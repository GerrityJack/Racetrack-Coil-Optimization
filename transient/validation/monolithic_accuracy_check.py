"""
monolithic_accuracy_check.py -- is the fully-coupled block-Newton
(monolithic_ta.py) formulation ACCURATE at the one regime this project has
ever validated a T-A solve against ground truth (a single implicit dt=600s
step from zero-field-cooled, I=params.I_design)?

Same discipline as transient/validation/accuracy_check_I196.py, which did
this for newton_ta.py's Gauss-Seidel hybrid: build ONE mesh, run the
UNMODIFIED production Picard path on it (ta_solve.solve_ta_at_current --
this project's ground truth methodology, unchanged here), then run
monolithic_ta.monolithic_step on a FRESH ta built on the SAME mesh, and
compare the resulting on-axis SCIF.

A. ta_solve.solve_ta_at_current() -- unmodified production Picard, ground truth.
B. monolithic_ta.monolithic_step() -- single coupled block-Newton system,
   Jc/n refreshed every step (see transient/monolithic_ta.py docstring).
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
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import monolithic_ta

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_mono196_{os.getpid()}{ext}"
    print("building mesh (shared by both A and B) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    print(f"I_design = {I} A, ramp_duration = {params.ramp_duration} s", flush=True)

    print("\n" + "=" * 78)
    print("A. PRODUCTION PICARD (ta_solve.solve_ta_at_current) -- ground truth")
    print("=" * 78)
    ta_a = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    A_h, B_h, T_h, info_a = ta_solve.solve_ta_at_current(
        domain, ta_a, uniform, I, ic, nm, verbose=True, warm_start=False)
    J_coil_a = ta_solve._J_from_T(ta_a, domain)
    J_unif_a = ta_a["t_hat_coil"] * (I / (ta_a["delta_SC"] * params.w))
    dJs_a = (J_coil_a - J_unif_a) * (ta_a["delta_SC"] / ta_a["Lambda"])
    scif_a = ta_solve.dB_bore_from_dJ(ta_a["coil_centroids"], dJs_a,
                                      ta_a["coil_vols"])[2] * 1e3
    print(f"\nPICARD RESULT: converged={info_a['converged']} "
          f"n_iters={info_a['n_iters']} on-axis SCIF={scif_a:+.2f} mT", flush=True)

    print("\n" + "=" * 78)
    print("B. MONOLITHIC BLOCK NEWTON, SAME discretization (dt=600, cold start)")
    print("=" * 78)
    ta_b = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    monolithic_ta.build_monolithic_problem(ta_b, verbose=False)
    info_b = monolithic_ta.monolithic_step(
        ta_b, domain, ic, nm, I, params.ramp_duration, uniform,
        max_outer=150, min_outer=6, stall_tol=0.05, first=True,
        bootstrap_iters=30, verbose=True)
    print(f"\nMONOLITHIC RESULT: converged={info_b['converged']} "
          f"stop_reason={info_b['stop_reason']} n_outer={info_b['n_outer']} "
          f"total_snes_iters={info_b['total_snes_iters']} "
          f"SCIF={info_b['scif_mT']:+.2f} mT", flush=True)
    print(f"SCIF trajectory tail: "
          f"{[round(s,2) for s in info_b['scif_hist_tail']]}", flush=True)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    diff_pct = 100.0 * abs(info_b["scif_mT"] - scif_a) / max(abs(scif_a), 1e-9)
    print(f"A (Picard, ground truth): {scif_a:+.2f} mT")
    print(f"B (monolithic block Newton): {info_b['scif_mT']:+.2f} mT")
    print(f"relative difference: {diff_pct:.2f}%")
    if diff_pct < 5.0:
        print("-> PASS: monolithic formulation agrees with the validated "
              "Picard ground truth to within 5% at the actual target "
              "current. The reformulation is accurate here, not just "
              "stable.")
    else:
        print("-> FAIL or INCONCLUSIVE: agreement is worse than 5% -- do "
              "NOT treat the monolithic solver's numbers as validated at "
              "this current until this is resolved.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
