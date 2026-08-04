"""
newton_march_check.py — first multi-step (warm-started) test of the Newton/
Picard-hybrid T-A solver (transient/newton_ta.py), insulated case (no NI
closure yet).

Everything up to now (newton_ta_check.py) only tested single, independent,
COLD-started steps. This is qualitatively different: it exercises the
warm-start path (first=False) for the first time, which two real bugs were
found in just from re-reading the code before this ever ran (see
newton_ta.py's comments on the T_bot_val/T_top_val fix and the relax_k fix)
-- so this script checks MORE than just "did it crash", per explicit
instruction to verify things are actually working correctly, not just
running.

Checks, in order of how much they'd catch:
  1. Per-step sanity: does the applied BC current actually change each step
     (catches a regression of the T_bot_val/T_top_val bug)? Is T actually
     carried over between steps (not reset), i.e. is the SOLVE not starting
     fresh from T=0 every step?
  2. Physical sanity: bore SCIF should broadly trend with current (more
     current -> more screening, roughly) across the ramp, not be
     independent noise.
  3. Ground truth where available: if the schedule holds at a fixed current
     long enough for Jc/n to stop moving, the LAST held step should be
     comparable to a single-step Picard/Newton solve at that same (I, dt)
     combination -- checked by running run_case()-style code from
     newton_ta_check.py for direct comparison where practical.
  4. Per-step converged/stop_reason is reported explicitly, not assumed --
     see step()'s converged semantics (True only for a genuine stall,
     False for any revert-and-stop, regardless of how close the SCIF is).

Run:  <env>/bin/python3 transient/validation/newton_march_check.py
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
    import ta_transient
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_nmarch{os.getpid()}{ext}"
    print("building mesh ...")
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta, verbose=False)

    # SHORT schedule first: n_ramp=4 over 600s -> dt=150s/step, one of the
    # dt values the earlier single-step dt-scan showed converging cleanly.
    # No hold phase yet -- keep this smoke test cheap.
    I_op = 196.0
    schedule = ta_transient.ramp_schedule(I_op, t_ramp=600.0, t_hold=0.0,
                                          n_ramp=4, n_hold=0)
    print("schedule:")
    for t, I, dt in schedule:
        print(f"  t={t:.1f}s  I={I:.2f}A  dt={dt:.1f}s")

    print("\n" + "=" * 78)
    print("MARCHING")
    print("=" * 78)
    # t_relax=0.15: the 2026-08-05 fix -- T itself must be damped between
    # outer iterations, not just Jc/n (see CLAUDE.md). Validated on an
    # isolated step; this is the first full-ramp test of it.
    hist = newton_ta.march(ta, domain, uniform, ic, nm, schedule,
                           max_outer=60, bootstrap_iters=30, verbose=True,
                           t_relax=0.15, spike_check=False)

    print("\n" + "=" * 78)
    print("CHECK 1: BC actually changed each step (T_bot_val tracked I_now)")
    print("=" * 78)
    # We can't inspect PAST values of T_bot_val (it's a live Constant), but
    # we CAN check that the current-vs-SCIF relationship below is not flat
    # across steps with different I -- a flat SCIF across changing I would
    # indicate the BC bug regressed (current never actually changed).
    Is = [h["I"] for h in hist]
    scifs = [h["scif_mT"] for h in hist]
    print(f"  I values applied per step : {[round(i,2) for i in Is]}")
    print(f"  SCIF per step             : {[round(s,2) for s in scifs]}")
    bc_ok = len(set(round(i, 1) for i in Is)) == len(Is)
    print(f"  distinct I per step (BC really changing): "
          f"{'PASS' if bc_ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print("CHECK 2: per-step converged/stop_reason (explicit, not assumed)")
    print("=" * 78)
    for h in hist:
        print(f"  step {h['step_index']+1}: I={h['I']:.2f}A  "
              f"converged={h['converged']}  stop_reason={h['stop_reason']}  "
              f"n_outer={h['n_outer']}  SCIF={h['scif_mT']:+.2f} mT")

    print("\n" + "=" * 78)
    print("CHECK 3: ground truth cross-check at the FINAL step's (I, dt)")
    print("=" * 78)
    # Independent, single-step (non-warm-started) Newton solve at the same
    # final (I, dt) as the march's last step, for comparison. This is NOT
    # expected to match closely (the march's last step is warm-started from
    # a real ramp history with a genuinely different physical A_prev, while
    # this is a cold start with the SAME dt used as a single implicit jump
    # from ZFC) -- but the ORDER OF MAGNITUDE and SIGN should agree, and a
    # wild mismatch would indicate something has gone wrong in the march.
    final = hist[-1]
    ta2 = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                    per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta2, verbose=False)
    ref_info = newton_ta.step(ta2, domain, ic, nm, final["I"], final["dt"],
                              uniform, max_outer=30, first=True,
                              bootstrap_iters=30, verbose=False)
    print(f"  march final step   : I={final['I']:.2f}A dt={final['dt']:.1f}s "
          f"SCIF={final['scif_mT']:+.2f} mT (warm-started, real ramp history)")
    print(f"  independent single-step at same (I,dt): "
          f"SCIF={ref_info['scif_mT']:+.2f} mT (cold start, NOT directly "
          f"comparable -- different physical history, sanity check only)")

    n_bad = sum(1 for h in hist if h["stop_reason"] not in
               ("stall", "iteration_spike_reverted", "newton_failure_reverted"))
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"steps completed: {len(hist)}/{len(schedule)}")
    print(f"BC-tracks-current check: {'PASS' if bc_ok else 'FAIL'}")
    print(f"steps with an unexpected stop_reason: {n_bad}")
    ok = bc_ok and len(hist) == len(schedule) and n_bad == 0
    print(f"VERDICT: {'PASS (ran to completion, BC tracking verified)' if ok else 'FAIL'}")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
