"""
ni_closure_stability_check.py — regression probe for the NI circuit closure.

HISTORY (2026-08-04)
---------------------
A first version of this check found a persistent, non-decaying oscillation
in the closure's Picard loop (20-30 of 48 bins in the physical clip band
every iteration, no trend toward convergence over 250 iterations) that did
not respond to 5x stronger relaxation, reordering the update, or 20x weaker
contact coupling. Two real bugs were found and fixed:

  1. ni_circuit.py computed the induced field E_i explicitly each Picard
     iteration from a finite difference using the PREVIOUS iteration's
     (not-yet-converged) I_z, combined with E_p via ELEMENTWISE per-bin
     arithmetic. The per-turn mutual inductance matrix is dense and far
     from diagonally dominant (GMD regularisation ~0.91mm >> the 75um turn
     pitch, so most turns are nearly as coupled to each other as to
     themselves) -- an elementwise/Jacobi-style update on such a matrix is
     the textbook non-convergent case. FIXED by solving the linear
     (inductive) part of the closure EXACTLY each iteration (a 48x48 dense
     solve), leaving only the nonlinear E_p term Picard-lagged.

  2. Fixing (1) alone made things WORSE (48/48 bins clipping, E_p spiking to
     1e4 V/m). Instrumenting E_p directly during the "frozen" warmup phase
     showed the actual root cause: ta_transient.py's step() used ONLY the
     fine relaxation (alpha=0.15) throughout, never implementing
     ta_solve.py's own two-phase scheme (fast alpha=0.30 ramp-up until |dB|
     stops decreasing, THEN switch to 0.15). Without the fast phase, a fixed
     30-iteration warmup was nowhere near enough for the base (insulated-
     equivalent) solve to settle -- the SCIF was still swinging by
     thousands of mT iteration-to-iteration, the same pattern ta_solve's own
     k=1-10 shows before ITS phase-2 switch. The closure was switching on
     into a state still in its own early transient. FIXED by replacing the
     fixed-iteration warmup with ta_transient._picard_phase(), which reuses
     ta_solve's exact two-phase relaxation and observable-stall convergence
     criterion for BOTH the frozen warmup and the closure-coupled phase.

WHAT THIS SCRIPT DOES NOW
--------------------------
Runs ONE real step() call (not a hand-rolled copy of its internals, so this
tests the actual production code path) for the first cold-start step of the
champion's ramp, and reports whether the warmup phase and the closure phase
each genuinely converge (not just "didn't hit its cap").

Run:  <env>/bin/python3 transient/validation/ni_closure_stability_check.py
Expected: warmup_converged=True, converged=True, clip=0 (or very small).
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

I_TARGET = 32.666666667
DT = 100.0


def main():
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_transient as tt
    import tparams as tp
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    rho_ct = float(os.environ.get("RHO_CT_OVERRIDE", str(tp.RHO_CT_UOHM_CM2)))

    print("=" * 78)
    print(f"NI closure regression check: I={I_TARGET:.2f} A, dt={DT:.0f} s, "
          f"rho_c={rho_ct} uOhm.cm2, EP_FACTOR_MODE={tp.EP_FACTOR_MODE}")
    print("=" * 78)

    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_nicheck{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta, circuit, bins = tt.build(domain, cell_tags, facet_tags, uniform,
                                 rho_ct_uohm_cm2=rho_ct)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    tt._seed_cold(ta, uniform, I_TARGET)
    info = tt.step(ta, circuit, domain, ic, nm, I_TARGET, DT,
                   n_picard=tp.N_PICARD_FIRST, first=True, verbose=True)

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"warmup: {info['n_iters_warmup']} iters, "
          f"converged={info['warmup_converged']}")
    print(f"closure: {info['n_iters_closure']} iters, "
          f"converged={info['converged']}")
    print(f"final: SCIF={info['scif_mT']:+.2f} mT, "
          f"I_z=[{info['I_z_min']:.2f},{info['I_z_max']:.2f}] A, "
          f"I_r_mean={info['I_r_mean']:.4f} A, clip={info['n_clipped']}")

    ok = bool(info["warmup_converged"] and info["converged"]
             and info["n_clipped"] == 0)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
