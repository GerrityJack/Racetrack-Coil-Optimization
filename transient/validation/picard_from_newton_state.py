"""
picard_from_newton_state.py — decisive test for the outer-loop drift found
in thin_layer_trend.py/tight_tol_trend.py (Newton-hybrid's true fixed point
at I=196A is ~-24 mT, not Picard's validated 641.26 mT).

Question: is -24 mT a second GENUINE physical fixed point of the T-A system
(meaning the underlying nonlinear problem is non-unique), or did the
Newton-hybrid's outer-loop iteration scheme (Jc(B)/n(B) Picard-lag +
t_relax T-damping) simply walk onto a spurious branch that the ORIGINAL,
validated Picard scheme would never reach?

Test: run the Newton-hybrid out to a clearly-drifted state (~240 outer
iters, SCIF ~20-50 mT, well off Picard's 641 mT), then TRANSPLANT that T
state into a fresh Picard `ta` (same mesh) and run ta_transient._picard_phase
-- the SAME validated, unmodified Picard iteration used everywhere else in
this project -- from there. Two possible outcomes:
  (a) Picard climbs back toward 641 mT -> Newton found a spurious branch;
      641 mT is the one true answer, the outer-loop scheme is buggy.
  (b) Picard ALSO heads toward something near -24 mT -> genuine
      non-uniqueness (or a bug shared by both, though the two schemes'
      inner mechanics are quite different, making a SHARED bug less
      likely) -- would need much deeper investigation.
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


def main():
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import newton_ta
    from ta_transient import _picard_phase
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_pfns_{os.getpid()}{ext}"
    print("building mesh (shared) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    dt = params.ramp_duration
    eps = float(getattr(params, "ta_eps_reg", 1.0))

    print("=" * 78)
    print("PHASE 1: drift the Newton-hybrid to a clearly-off-641 state (~240 iters)")
    print("=" * 78)
    ta_n = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta_n, verbose=False)

    info = newton_ta.step(ta_n, domain, ic, nm, I, dt, uniform, max_outer=30,
                          min_outer=30, stall_tol=1e-9, first=True,
                          bootstrap_iters=30, verbose=False,
                          spike_check=False, t_relax=0.15)
    print(f"  after bootstrap+30: SCIF={info['scif_mT']:+.2f} mT")
    for c in range(7):   # 30 + 7*30 = 240 outer iters total
        info = newton_ta.step(ta_n, domain, ic, nm, I, dt, uniform,
                              max_outer=30, min_outer=30, stall_tol=1e-9,
                              first=False, verbose=False, spike_check=False,
                              t_relax=0.15)
        print(f"  chunk {c+1}: SCIF={info['scif_mT']:+.2f} mT", flush=True)

    print(f"\nNewton-hybrid state after ~240 outer iters: "
          f"SCIF={info['scif_mT']:+.2f} mT (target: clearly off 641.26 mT "
          f"Picard ground truth)")

    print("\n" + "=" * 78)
    print("PHASE 2: transplant this T state into a FRESH Picard ta, seed "
          "consistently, run ta_transient._picard_phase (unmodified)")
    print("=" * 78)
    ta_p = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)

    # Transplant T -- same mesh/domain, same per-layer V_T construction, so
    # DOF layout matches; this is the actual physical state Newton reached.
    for T_p, T_n in zip(ta_p["layer_T_fns"], ta_n["layer_T_fns"]):
        T_p.x.array[:] = T_n.x.array
        T_p.x.scatter_forward()

    ta_p["dt_const"].value = float(dt)
    T_amp = I / (2.0 * ta_p["delta_SC"])
    ta_p["T_bot_val"].value = +T_amp
    ta_p["T_top_val"].value = -T_amp

    # Seed A/B/rho consistently with the TRANSPLANTED T (mirrors how
    # newton_ta.step()/ta_transient._seed_cold seed a fresh state, just
    # from this T instead of T=0).
    J_seed = ta_solve._J_from_T(ta_p, domain)
    ta_solve._update_Js(ta_p, J_seed)
    ta_solve._solve_A(ta_p, ta_p["L_A_form"])
    B_h = ta_p["B_fn"]
    B_h.interpolate(ta_p["curl_expr"])
    B_seed = B_h.x.array.reshape(-1, 3)[ta_p["coil_cells"]]
    ta_p["_rho_prev"] = None
    ta_solve._update_rho(ta_p, J_seed, B_seed, ic, nm, eps)

    dJs_seed = (J_seed - ta_p["t_hat_coil"] * (I / (ta_p["delta_SC"] * params.w))) \
        * (ta_p["delta_SC"] / ta_p["Lambda"])
    scif_seed = ta_solve.dB_bore_from_dJ(ta_p["coil_centroids"], dJs_seed,
                                         ta_p["coil_vols"])[2] * 1e3
    print(f"  seeded Picard state SCIF (should match Newton's ~{info['scif_mT']:.1f} mT): "
          f"{scif_seed:+.2f} mT", flush=True)

    print("\n  running _picard_phase from this state (max 150 iters) ...", flush=True)
    J_coil, n_iters, converged = _picard_phase(
        ta_p, domain, ic, nm, I, dt, J_coil=J_seed, closure=lambda: None,
        max_iters=150, min_iters=10,
        scif_tol=float(getattr(params, "ta_scif_stall_mT", 0.05)),
        label="picard-from-newton-state", verbose=True)

    dJs = (J_coil - ta_p["t_hat_coil"] * (I / (ta_p["delta_SC"] * params.w))) \
        * (ta_p["delta_SC"] / ta_p["Lambda"])
    scif_final = ta_solve.dB_bore_from_dJ(ta_p["coil_centroids"], dJs,
                                          ta_p["coil_vols"])[2] * 1e3

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"Picard ground truth (from ZFC):        +641.26 mT")
    print(f"Newton-hybrid drifted state (start):   {info['scif_mT']:+.2f} mT")
    print(f"Picard, continued FROM that state:     {scif_final:+.2f} mT "
          f"(converged={converged}, n_iters={n_iters})")
    if abs(scif_final - 641.26) < abs(scif_final - info['scif_mT']):
        print("-> (a) Picard climbed BACK toward 641 mT: Newton found a "
              "SPURIOUS branch. 641 mT is the one true answer.")
    elif scif_final < 100:
        print("-> (b) Picard ALSO stayed near/drifted toward the low value: "
              "possible genuine non-uniqueness or a shared issue -- needs "
              "deeper investigation, do not assume either mechanism yet.")
    else:
        print("-> INCONCLUSIVE: neither clearly matches (a) nor (b) -- report "
              "the raw trajectory above and investigate directly.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
