"""
accuracy_diagnose.py — follow-up to accuracy_check_I196.py's 15.65% Picard
vs Newton-hybrid discrepancy at I=196A. Compares the actual converged
PHYSICAL fields (not just the summary SCIF) between both solvers on the
same mesh, to localize where they diverge rather than guess.
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def field_stats(label, ta, domain, ic_model, n_model, I, is_newton):
    import ta_solve
    from ic_model import angle_with_normal_deg

    coil = ta["coil_cells"]
    B_h = ta["B_fn"]
    B_h.interpolate(ta["curl_expr"])
    B_coil = B_h.x.array.reshape(-1, 3)[coil]
    Bmag = np.linalg.norm(B_coil, axis=1)

    J_coil = ta_solve._J_from_T(ta, domain)
    Jmag = np.linalg.norm(J_coil, axis=1)

    n_hat_coil = ta["n_hat_coil"]
    theta = angle_with_normal_deg(B_coil, n_hat_coil)
    Ic_arr, clip_frac = ic_model.critical_current(Bmag, theta)
    Jc_vol = Ic_arr / (ta["delta_SC"] * ic_model.tape_width)
    j_over_jc = Jmag / Jc_vol
    n_arr, _ = n_model.n_value(Bmag, theta)

    print(f"\n--- {label} ---")
    print(f"  |B| coil cells:  mean={Bmag.mean():.4f} T  max={Bmag.max():.4f} T  "
          f"min={Bmag.min():.4f} T  frac>8T={np.mean(Bmag>8.0):.4f}")
    print(f"  |J| coil cells:  mean={Jmag.mean():.4e} A/m^2  max={Jmag.max():.4e}")
    print(f"  Jc(B,theta):     mean={Jc_vol.mean():.4e}  max={Jc_vol.max():.4e}  "
          f"min={Jc_vol.min():.4e}")
    print(f"  n(B,theta):      mean={n_arr.mean():.3f}  max={n_arr.max():.3f}  "
          f"min={n_arr.min():.3f}")
    print(f"  J/Jc:            mean={j_over_jc.mean():.4f}  max={j_over_jc.max():.4f}  "
          f"frac>1={np.mean(j_over_jc>1.0):.4f}")
    print(f"  clip_frac (Ic model hit 8T boundary): {float(clip_frac):.4f}")

    # per-layer breakdown, since the discrepancy could be localized to one layer
    for layer, idx in enumerate(ta["layer_cell_idx"]):
        Bl = Bmag[idx]
        Jl = Jmag[idx]
        jjc_l = j_over_jc[idx]
        print(f"    layer {layer}: |B| mean={Bl.mean():.3f}T max={Bl.max():.3f}T  "
              f"J/Jc mean={jjc_l.mean():.3f} max={jjc_l.max():.3f}")

    return dict(Bmag=Bmag, Jmag=Jmag, Jc=Jc_vol, n=n_arr, j_over_jc=j_over_jc)


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
    params.mesh_filename = f"{root}_diag196_{os.getpid()}{ext}"
    print("building mesh ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design

    print("=" * 78)
    print("A. PRODUCTION PICARD -- solving ...")
    print("=" * 78)
    ta_a = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    ta_solve.solve_ta_at_current(domain, ta_a, uniform, I, ic, nm,
                                 verbose=False, warm_start=False)

    print("=" * 78)
    print("B. NEWTON-HYBRID t_relax=0.15 -- solving ...")
    print("=" * 78)
    ta_b = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta_b, verbose=False)
    newton_ta.step(ta_b, domain, ic, nm, I, params.ramp_duration, uniform,
                   max_outer=100, min_outer=3, stall_tol=0.05, first=True,
                   bootstrap_iters=30, verbose=False, spike_check=False,
                   t_relax=0.15)

    stats_a = field_stats("A. Picard (ground truth)", ta_a, domain, ic, nm, I,
                          is_newton=False)
    stats_b = field_stats("B. Newton-hybrid", ta_b, domain, ic, nm, I,
                          is_newton=True)

    print("\n" + "=" * 78)
    print("DIRECT COMPARISON")
    print("=" * 78)
    dB = stats_b["Bmag"].mean() - stats_a["Bmag"].mean()
    dJ = (stats_b["Jmag"].mean() - stats_a["Jmag"].mean()) / stats_a["Jmag"].mean() * 100
    djjc = stats_b["j_over_jc"].mean() - stats_a["j_over_jc"].mean()
    print(f"mean |B|: Picard={stats_a['Bmag'].mean():.4f}T  "
          f"Newton={stats_b['Bmag'].mean():.4f}T  diff={dB:+.4f}T")
    print(f"mean |J|: Picard={stats_a['Jmag'].mean():.4e}  "
          f"Newton={stats_b['Jmag'].mean():.4e}  diff={dJ:+.2f}%")
    print(f"mean J/Jc: Picard={stats_a['j_over_jc'].mean():.4f}  "
          f"Newton={stats_b['j_over_jc'].mean():.4f}  diff={djjc:+.4f}")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
