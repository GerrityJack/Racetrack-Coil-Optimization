"""
hybrid_blend_sweep.py — sweep newton_blend to find a value that's both
stable AND accurate against the I=196A Picard ground truth (641.26 mT).
newton_blend=1.0 (pure Newton-informed rho) was tested and found to
overshoot then drift, settling somewhere between ~530-550 mT depending on
exact fix variant -- biased low, not matching truth. newton_blend=0.0
would be pure Picard (guaranteed correct, since it IS Picard, but gets
no benefit from Newton at all). Sweeping between.
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
    params.mesh_filename = f"{root}_blendsweep_{os.getpid()}{ext}"
    print("building mesh (shared) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design

    print("=" * 78)
    print("A. PRODUCTION PICARD -- ground truth")
    print("=" * 78)
    ta_a = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    ta_solve.solve_ta_at_current(domain, ta_a, uniform, I, ic, nm,
                                 verbose=False, warm_start=False)
    J_coil_a = ta_solve._J_from_T(ta_a, domain)
    J_unif_a = ta_a["t_hat_coil"] * (I / (ta_a["delta_SC"] * params.w))
    dJs_a = (J_coil_a - J_unif_a) * (ta_a["delta_SC"] / ta_a["Lambda"])
    scif_a = ta_solve.dB_bore_from_dJ(ta_a["coil_centroids"], dJs_a,
                                      ta_a["coil_vols"])[2] * 1e3
    print(f"PICARD ground truth: {scif_a:+.2f} mT", flush=True)

    results = []
    for blend in (0.5, 0.3, 0.15):
        print("\n" + "=" * 78)
        print(f"B. HYBRID, newton_blend={blend}, max_outer=120")
        print("=" * 78)
        ta_b = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags,
                                         uniform, per_layer=True,
                                         per_turn_bc=False)
        newton_ta.build_layer_newton_problems(ta_b, verbose=False)
        info_b = newton_ta.hybrid_step(ta_b, domain, ic, nm, I,
                                       params.ramp_duration, uniform,
                                       max_outer=120, min_outer=6,
                                       stall_tol=0.05, first=True,
                                       bootstrap_iters=30, verbose=True,
                                       newton_blend=blend)
        diff_pct = 100.0 * abs(info_b["scif_mT"] - scif_a) / abs(scif_a)
        print(f"  RESULT blend={blend}: converged={info_b['converged']} "
              f"stop={info_b['stop_reason']} n_outer={info_b['n_outer']} "
              f"SCIF={info_b['scif_mT']:+.2f} mT  diff={diff_pct:.2f}%")
        results.append((blend, info_b, diff_pct))

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    print(f"Picard ground truth: {scif_a:+.2f} mT")
    for blend, info_b, diff_pct in results:
        print(f"  blend={blend:.2f}: SCIF={info_b['scif_mT']:+8.2f} mT  "
              f"diff={diff_pct:6.2f}%  converged={info_b['converged']}  "
              f"stop={info_b['stop_reason']}  n_outer={info_b['n_outer']}")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
