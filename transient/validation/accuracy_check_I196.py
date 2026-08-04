"""
accuracy_check_I196.py — is the t_relax-fixed Newton-hybrid ACCURATE, not
just stable? The overnight run (2026-08-04) validated stability across 74
step-solves but explicitly did NOT check any value against ground truth.

Rigorous, apples-to-apples check: both paths solve the IDENTICAL
discretization -- a single implicit dt=600s step from zero-field-cooled to
I=196A (params.I_design, the current used throughout tonight's runs) --
so they should agree closely if the Newton reformulation is correct. This
mirrors newton_ta_check.py's original regression (Picard 172.77 mT vs
Newton ~171.5 mT at I=32.667A, <1% agreement), just repeated at the actual
target current instead of I/6.

A. ta_solve.solve_ta_at_current() -- the UNMODIFIED, fully-validated
   production Picard path. This IS the project's ground truth methodology;
   nothing about it changes here.
B. newton_ta.step() -- single step, first=True, dt=600, I=196A,
   t_relax=0.15 (the validated fix), same mesh.

Also reports the SAME quantity from last night's multi-step dt=150 ramp
(548-580 mT, from a genuinely time-resolved 4-substep ramp) for context --
NOT expected to match A/B exactly (finer time resolution is a different,
presumably more accurate, discretization of the same continuous ramp), but
should be the same order of magnitude and sign if the model is sound.
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

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_acc196_{os.getpid()}{ext}"
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
    print("B. NEWTON-HYBRID, t_relax=0.15, SAME discretization (dt=600, cold start)")
    print("=" * 78)
    ta_b = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta_b, verbose=False)
    info_b = newton_ta.step(ta_b, domain, ic, nm, I, params.ramp_duration,
                            uniform, max_outer=100, min_outer=3,
                            stall_tol=0.05, first=True, bootstrap_iters=30,
                            verbose=True, spike_check=False, t_relax=0.15)
    print(f"\nNEWTON RESULT: converged={info_b['converged']} "
          f"stop_reason={info_b['stop_reason']} n_outer={info_b['n_outer']} "
          f"SCIF={info_b['scif_mT']:+.2f} mT", flush=True)
    print(f"SCIF trajectory tail: "
          f"{[round(s,2) for s in info_b['scif_hist_tail']]}", flush=True)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    diff_pct = 100.0 * abs(info_b["scif_mT"] - scif_a) / max(abs(scif_a), 1e-9)
    print(f"A (Picard, ground truth): {scif_a:+.2f} mT")
    print(f"B (Newton, t_relax=0.15): {info_b['scif_mT']:+.2f} mT")
    print(f"relative difference: {diff_pct:.2f}%")
    if diff_pct < 5.0:
        print("-> PASS: Newton-hybrid agrees with the validated Picard ground "
              "truth to within 5% at the actual target current. The "
              "reformulation is accurate here, not just stable.")
    else:
        print("-> FAIL or INCONCLUSIVE: agreement is worse than 5% -- do NOT "
              "treat the Newton-hybrid's numbers as validated at this "
              "current until this is resolved.")
    print(f"\n(context only, NOT expected to match A/B exactly -- different, "
          f"finer time discretization of the same continuous ramp) "
          f"last night's 4-step dt=150 ramp final-step SCIF range: "
          f"548-580 mT across 4 independent runs")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
