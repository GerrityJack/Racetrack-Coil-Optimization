"""
monolithic_historical_residual_check.py -- 2026-08-05 overnight follow-up.

Re-runs the EXACT historical case that produced monolithic_ta.py's
documented "+6800/+3669/+2012mT at three damping strengths" result
(docs/HISTORY.md's 2026-08-05 "fully monolithic block-Newton T-A" entry:
dt=params.ramp_duration=600s, I=params.I_design=196A, cold start,
step_relax in {0.3, 0.1, 0.03}, mirroring monolithic_step_relax_sweep.py
exactly) -- but this time with snes_monitor enabled so the RAW PETSc
residual is visible every outer iteration, not just SCIF.

Question: was that historical "clean asymptotic plateau" ALSO a case
where the true residual was exploding/plateauing at a huge nonzero value
while SCIF alone looked converged -- the same false-positive signature
found and retracted elsewhere in this session's monolithic_ta_diff
investigation -- or is the historical case genuinely different?
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    step_relax = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    max_outer = int(sys.argv[2]) if len(sys.argv) > 2 else 70

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    import numpy as np

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import monolithic_ta
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monohist_{os.getpid()}{ext}"
    print(f"building mesh (I_design={params.I_design}A, "
          f"dt={params.ramp_duration}s, step_relax={step_relax}) ...",
          flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    dt = params.ramp_duration

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    # snes_monitor=None turns on PETSc's own per-Newton-step residual print
    # ("0 SNES Function norm ...", "1 SNES Function norm ...") -- the exact
    # instrument that caught the false positive elsewhere in this
    # investigation. This is the FIRST time this specific historical
    # regression case has been checked this way.
    monolithic_ta.build_monolithic_problem(
        ta, snes_options=dict(snes_monitor=None), verbose=False)

    snes = ta["mono_snes"]

    # Patch monolithic_step's own per-iteration loop is not necessary --
    # snes_monitor prints directly to stdout on every problem.solve() call,
    # which happens once per outer iteration under the max_it=1 scheme.
    # We just need to also capture the EXPLICIT post-solve residual via
    # snes.getFunction()[0].norm() to log a clean per-outer-iteration
    # number alongside SCIF, since PETSc's own monitor output is a lot of
    # visual noise otherwise.
    orig_solve = ta["mono_problem"].solve
    fnorm_log = []

    def solve_and_log():
        result = orig_solve()
        fvec = snes.getFunction()[0]
        fnorm_log.append(fvec.norm())
        return result

    ta["mono_problem"].solve = solve_and_log

    info = monolithic_ta.monolithic_step(
        ta, domain, ic, nm, I, dt, uniform,
        max_outer=max_outer, min_outer=6, stall_tol=0.05, first=True,
        bootstrap_iters=30, verbose=True, step_relax=step_relax, debug=True)

    print("\n" + "=" * 78)
    print(f"RESULT step_relax={step_relax}: converged={info['converged']} "
          f"stop_reason={info['stop_reason']} n_outer={info['n_outer']} "
          f"SCIF={info['scif_mT']:+.2f} mT  (ground truth ~+641 mT)")
    print("=" * 78)
    print(f"Raw residual (||F||) trajectory, one value per outer iteration "
          f"({len(fnorm_log)} total):")
    for i, f in enumerate(fnorm_log):
        ratio = f / fnorm_log[i - 1] if i > 0 and fnorm_log[i - 1] > 0 else float("nan")
        print(f"  outer={i+1:3d}  ||F||={f:.6e}  ratio_to_prev={ratio:.4f}")
    if len(fnorm_log) >= 2:
        first, last = fnorm_log[0], fnorm_log[-1]
        print(f"\n||F|| first={first:.3e}  last={last:.3e}  "
              f"overall_ratio={last/max(first,1e-300):.3e}")
        # Heuristic verdict: a genuinely converging Newton residual should
        # DECREASE, typically by orders of magnitude, not grow or plateau
        # at a huge value.
        if last > 1.0 and last >= 0.1 * max(fnorm_log):
            print("VERDICT: residual is NOT decreasing to anything near "
                  "zero -- plateaued or still growing at a large value. "
                  "Consistent with the SAME false-positive signature found "
                  "elsewhere in this investigation.")
        else:
            print("VERDICT: residual genuinely decreased substantially -- "
                  "this historical case looks different from the retracted "
                  "one; needs more scrutiny before concluding either way.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
