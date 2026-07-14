"""
diagnostics.py
===============
Solver diagnostics for solve.py / solve_sweep.py: per-solve logging
(mesh stats, solver options, PETSc converged reason, iteration count,
residual history, an independently-computed true residual, field
magnitude stats, wall-clock time) plus a convergence plot.

Why an independently-computed true residual matters: PETSc's own
"converged reason" can be misleading if you don't know what to expect
from it (e.g. ksp_type=preonly + pc_type=lu always reports reason=4,
iterations=1 -- that's the CORRECT, documented behavior for a direct
solve, not a sign of trouble: PETSc's own docs for KSP_CONVERGED_ITS
say it's "used by the KSPPREONLY solver to indicate that the operator
(preconditioner) was applied to the right hand side, without actually
using the Krylov method"). What a direct solve's "converged" status
does NOT tell you is whether the factorization was actually accurate
for this particular system -- a singular or near-singular matrix can
still report success while the answer is poor. Computing the true
residual r = b - A*x directly (independent of whatever the KSP itself
reports) is solver-agnostic insurance against that.

For an iterative solver (CG, GMRES, ...), attach_monitor() also records
the actual per-iteration residual history, which IS a real convergence
trajectory worth plotting; for a direct solve that history is trivially
a single point, and the plot will show that honestly rather than fake a
curve.
"""
import json
import time

import numpy as np

KSP_REASON_NAMES = {
    1: "CONVERGED_RTOL_NORMAL", 9: "CONVERGED_ATOL_NORMAL",
    2: "CONVERGED_RTOL", 3: "CONVERGED_ATOL", 4: "CONVERGED_ITS",
    5: "CONVERGED_CG_NEG_CURVE", 6: "CONVERGED_CG_CONSTRAINED",
    7: "CONVERGED_STEP_LENGTH", 8: "CONVERGED_HAPPY_BREAKDOWN",
    -2: "DIVERGED_NULL", -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL",
    -5: "DIVERGED_BREAKDOWN", -6: "DIVERGED_BREAKDOWN_BICG",
    -7: "DIVERGED_NONSYMMETRIC", -8: "DIVERGED_INDEFINITE_PC",
    -9: "DIVERGED_NANORINF", -10: "DIVERGED_INDEFINITE_MAT",
    -11: "DIVERGED_PC_FAILED",
}


def reason_name(reason):
    return KSP_REASON_NAMES.get(int(reason), f"UNKNOWN({reason})")


class ConvergenceRecorder:
    """Callable for ksp.setMonitor(...) -- records every (iteration,
    residual norm) PETSc reports during a solve. For ksp_type=preonly
    this will just be a single trivial entry; for an iterative method
    it's the real convergence history."""

    def __init__(self):
        self.history = []

    def __call__(self, ksp, it, rnorm):
        self.history.append((int(it), float(rnorm)))

    def attach(self, ksp):
        ksp.setMonitor(self)
        return self


def true_residual(problem, x_petsc_vec):
    """
    Independently computes the true relative residual ||Ax - b|| / ||b||
    using the assembled matrix/RHS directly -- doesn't rely on whatever
    the KSP itself reports. Returns (abs_residual, rel_residual), or
    (None, None) with no exception if this dolfinx version exposes
    LinearProblem's matrix/vector under different attribute names (so a
    version mismatch here degrades to "no true-residual check" rather
    than crashing the whole solve).
    """
    try:
        A, b = problem.A, problem.b
        Ax = A.createVecLeft()
        A.mult(x_petsc_vec, Ax)
        r = b.copy()
        r.axpy(-1.0, Ax)
        abs_res = r.norm()
        b_norm = b.norm()
        rel_res = abs_res / b_norm if b_norm > 0 else abs_res
        return float(abs_res), float(rel_res)
    except AttributeError as e:
        return None, None


class SolveLog:
    """Accumulates one record per solve, then writes a human-readable
    text log, a machine-readable JSON log, and a convergence plot."""

    def __init__(self):
        self.header = {}
        self.records = []

    def set_header(self, **kwargs):
        self.header.update(kwargs)

    def add_record(self, **kwargs):
        self.records.append(kwargs)

    def log_solve(self, label, I_amps, ksp, recorder, problem, x_petsc_vec,
                  B_h, coil_cells, wall_time, petsc_options):
        reason = int(ksp.getConvergedReason())
        Bmag = np.linalg.norm(B_h.x.array.reshape(-1, 3)[coil_cells], axis=1)
        abs_res, rel_res = true_residual(problem, x_petsc_vec)
        B_p50 = float(np.median(Bmag))
        B_p99 = float(np.percentile(Bmag, 99))
        # ratio of the single worst cell to the 99th percentile -- a huge
        # ratio here (one cell wildly above essentially everything else)
        # is the specific signature of gauge/null-space pollution leaking
        # through curl(A)'s imperfect floating-point cancellation, as
        # opposed to a generically-too-high field (which would show up
        # as B_p99 itself being large, not just the single max).
        outlier_ratio = float(Bmag.max() / B_p99) if B_p99 > 0 else float("inf")
        A_norm = float(x_petsc_vec.norm())
        self.add_record(
            label=label, I_amps=float(I_amps),
            ksp_type=petsc_options.get("ksp_type"),
            pc_type=petsc_options.get("pc_type"),
            converged_reason=reason, converged_reason_name=reason_name(reason),
            iterations=int(ksp.getIterationNumber()),
            residual_history=list(recorder.history),
            true_abs_residual=abs_res, true_rel_residual=rel_res,
            B_min_T=float(Bmag.min()), B_max_T=float(Bmag.max()),
            B_mean_T=float(Bmag.mean()), B_p50_T=B_p50, B_p99_T=B_p99,
            outlier_ratio=outlier_ratio, A_norm=A_norm,
            wall_time_s=float(wall_time),
        )

    def write(self, txt_path, json_path):
        with open(json_path, "w") as f:
            json.dump({"header": self.header, "records": self.records}, f, indent=2)

        lines = ["=" * 70, "SOLVE LOG", "=" * 70, ""]
        for k, v in self.header.items():
            lines.append(f"{k}: {v}")
        lines.append("")
        for rec in self.records:
            lines.append("-" * 70)
            lines.append(f"[{rec['label']}]  I = {rec['I_amps']:.2f} A")
            lines.append(f"  solver: ksp_type={rec['ksp_type']}, pc_type={rec['pc_type']}")
            lines.append(f"  PETSc converged reason: {rec['converged_reason']} "
                         f"({rec['converged_reason_name']}), "
                         f"iterations: {rec['iterations']}")
            if rec['true_rel_residual'] is not None:
                lines.append(f"  true residual ||Ax-b||/||b||: {rec['true_rel_residual']:.3e} "
                             f"(absolute: {rec['true_abs_residual']:.3e})")
            else:
                lines.append("  true residual: could not compute "
                             "(LinearProblem.A/.b not found under expected "
                             "names in this dolfinx version)")
            if len(rec['residual_history']) > 1:
                lines.append(f"  residual history: {rec['residual_history']}")
            lines.append(f"  |B| on coil: min={rec['B_min_T']:.4g} T, "
                         f"max={rec['B_max_T']:.4g} T, mean={rec['B_mean_T']:.4g} T, "
                         f"p50={rec['B_p50_T']:.4g} T, p99={rec['B_p99_T']:.4g} T")
            lines.append(f"  ||A|| (solution vector norm): {rec['A_norm']:.3e}  "
                         f"|  max/p99 outlier ratio: {rec['outlier_ratio']:.2f}")
            if rec['B_max_T'] > 50.0 and rec['outlier_ratio'] > 20.0:
                lines.append(
                    f"  *** WARNING: |B|_max = {rec['B_max_T']:.3g} T is physically "
                    f"implausible, AND it's an isolated spike (max is "
                    f"{rec['outlier_ratio']:.0f}x the 99th percentile, not a "
                    "generally-too-high field). This specific pattern -- "
                    "near-correct field almost everywhere plus one wild "
                    "outlier cell -- is the signature of gauge/null-space "
                    "pollution leaking through curl(A)'s imperfect floating-"
                    "point cancellation, most likely because "
                    "gauge_regularization in params.py is too weak for this "
                    "mesh size. Try increasing it by several orders of "
                    "magnitude (e.g. 1e-8 -> 1e-3) and re-solving -- if the "
                    "outlier disappears and ||A|| drops to a more modest "
                    "value, that confirms it. If it persists, suspect a mesh "
                    "quality defect (a sliver/degenerate element) instead. ***")
            elif rec['B_max_T'] > 50.0:
                lines.append(f"  *** WARNING: |B|_max = {rec['B_max_T']:.3g} T is "
                             "physically implausible for this geometry/current "
                             "scale -- treat this solve's results as suspect "
                             "even though PETSc reported convergence. ***")
            lines.append(f"  wall time: {rec['wall_time_s']:.2f} s")
        lines.append("-" * 70)

        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Wrote {txt_path} and {json_path}")

    def plot(self, png_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        any_real_history = False
        for i, rec in enumerate(self.records):
            hist = rec["residual_history"]
            if len(hist) > 1:
                any_real_history = True
                its, rnorms = zip(*hist)
                ax.semilogy(its, rnorms, "o-", label=f"{rec['I_amps']:.0f} A")
            else:
                it, rnorm = hist[0] if hist else (0, np.nan)
                # direct solves all trivially land at it=0 -- jitter x by
                # index so overlapping markers are still distinguishable
                ax.semilogy([it + i * 0.02], [max(rnorm, 1e-300)], "o",
                           label=f"{rec['I_amps']:.0f} A")
        ax.set_xlabel("KSP iteration")
        ax.set_ylabel("residual norm (as reported by PETSc)")
        title = "KSP convergence history"
        if not any_real_history:
            title += "\n(direct solver -- single point per current, not a curve, by design)"
        ax.set_title(title, fontsize=11)
        if len(self.records) <= 8:
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        ax = axes[1]
        Is = [r["I_amps"] for r in self.records]
        rel_res = [r["true_rel_residual"] if r["true_rel_residual"] is not None
                   else np.nan for r in self.records]
        ax.semilogy(Is, rel_res, "o-", color="C3")
        ax.set_xlabel("current (A)")
        ax.set_ylabel("true relative residual ||Ax-b||/||b||")
        ax.set_title("Independent residual check\n(solver-agnostic, computed "
                     "directly from A, x, b)", fontsize=11)

        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png_path}")
