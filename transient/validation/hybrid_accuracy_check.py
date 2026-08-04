"""
hybrid_accuracy_check.py — does newton_ta.hybrid_step() (the Newton-
INFORMED Picard hybrid, replacing the disproven t_relax scheme) actually
reach the validated Picard ground truth at I=196A?

Same test structure as accuracy_check_I196.py, which is what first caught
the t_relax scheme's 15.65% error. Ground truth: ta_solve.solve_ta_at_current(),
single dt=600s step from ZFC, I=196A -> +641.26 mT (independently confirmed
twice now: from ZFC, and from a from-Newton-drifted-state restart in
picard_from_newton_state.py).
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
    import newton_ta
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_hybacc_{os.getpid()}{ext}"
    print("building mesh (shared) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design

    print("\n" + "=" * 78)
    print("A. PRODUCTION PICARD (ta_solve.solve_ta_at_current) -- ground truth")
    print("=" * 78)
    ta_a = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    A_h, B_h, T_h, info_a = ta_solve.solve_ta_at_current(
        domain, ta_a, uniform, I, ic, nm, verbose=False, warm_start=False)
    J_coil_a = ta_solve._J_from_T(ta_a, domain)
    J_unif_a = ta_a["t_hat_coil"] * (I / (ta_a["delta_SC"] * params.w))
    dJs_a = (J_coil_a - J_unif_a) * (ta_a["delta_SC"] / ta_a["Lambda"])
    scif_a = ta_solve.dB_bore_from_dJ(ta_a["coil_centroids"], dJs_a,
                                      ta_a["coil_vols"])[2] * 1e3
    print(f"PICARD RESULT: converged={info_a['converged']} "
          f"n_iters={info_a['n_iters']} on-axis SCIF={scif_a:+.2f} mT",
          flush=True)

    print("\n" + "=" * 78)
    print("B. NEWTON-INFORMED PICARD HYBRID, same discretization "
          "(dt=600, cold start)")
    print("=" * 78)
    ta_b = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta_b, verbose=False)
    info_b = newton_ta.hybrid_step(ta_b, domain, ic, nm, I,
                                   params.ramp_duration, uniform,
                                   max_outer=100, min_outer=6,
                                   stall_tol=0.05, first=True,
                                   bootstrap_iters=30, verbose=True)
    print(f"\nHYBRID RESULT: converged={info_b['converged']} "
          f"stop_reason={info_b['stop_reason']} n_outer={info_b['n_outer']} "
          f"SCIF={info_b['scif_mT']:+.2f} mT  "
          f"newton_failures={info_b['n_newton_failures']}", flush=True)
    print(f"SCIF trajectory tail: "
          f"{[round(s,2) for s in info_b['scif_hist_tail']]}", flush=True)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    diff_pct = 100.0 * abs(info_b["scif_mT"] - scif_a) / max(abs(scif_a), 1e-9)
    print(f"A (Picard, ground truth):        {scif_a:+.2f} mT")
    print(f"B (Newton-informed hybrid):      {info_b['scif_mT']:+.2f} mT")
    print(f"relative difference: {diff_pct:.2f}%")
    if diff_pct < 2.0:
        print("-> PASS: the Newton-informed hybrid agrees with Picard ground "
              "truth to within 2% -- accurate, not just stable.")
    else:
        print("-> FAIL: still wrong. The hybrid mechanism needs further work.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
