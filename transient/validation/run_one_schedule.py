"""
run_one_schedule.py — CLI-driven, single-process runner for ONE
newton_ta.march() schedule, built for the 2026-08-05 overnight validation
of the t_relax fix (see CLAUDE.md's "FOUND IT" entry).

Writes a full step-by-step text log to stdout (redirect to a file when
launching) AND a machine-readable JSON summary to --out-json, so an
orchestrator can aggregate results without re-parsing prose. Every step's
diagnostics dict is written, not just a final pass/fail, so a human
checking mid-run can see exactly how each step behaved (SCIF trajectory
tail, n_outer, stop_reason) -- this is the file an overnight check-in
should tail.

Does NOT modify solve/ta_solve.py or any production path. Builds its own
fresh mesh (PID-suffixed) and cleans it up on exit, including on failure
(try/finally), so a crashed job never leaves stray .msh files for the next
job to trip over.
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--i-op", type=float, default=196.0)
    p.add_argument("--t-ramp", type=float, default=600.0)
    p.add_argument("--t-hold", type=float, default=0.0)
    p.add_argument("--n-ramp", type=int, default=4)
    p.add_argument("--n-hold", type=int, default=0)
    p.add_argument("--max-outer", type=int, default=100)
    p.add_argument("--min-outer", type=int, default=3)
    p.add_argument("--stall-tol", type=float, default=0.05)
    p.add_argument("--t-relax", type=float, default=0.15)
    p.add_argument("--bootstrap-iters", type=int, default=30)
    p.add_argument("--spike-check", action="store_true")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.monotonic()

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
    params.mesh_filename = f"{root}_ovn{os.getpid()}{ext}"

    print(f"=== {args.tag} ===", flush=True)
    print(f"config: i_op={args.i_op} t_ramp={args.t_ramp} t_hold={args.t_hold} "
          f"n_ramp={args.n_ramp} n_hold={args.n_hold} max_outer={args.max_outer} "
          f"min_outer={args.min_outer} stall_tol={args.stall_tol} "
          f"t_relax={args.t_relax} spike_check={args.spike_check}", flush=True)

    result = dict(tag=args.tag, args=vars(args), ok=False, error=None,
                 steps=[], wall_s=None)
    try:
        print("building mesh ...", flush=True)
        build_mesh.build(write_path=params.mesh_filename, verbose=False)
        md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
        domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
        uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
        ic = IcModel(params.csv_filename)
        nm = NValueModel(params.n_value_csv_filename)

        ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                       per_layer=True, per_turn_bc=False)
        newton_ta.build_layer_newton_problems(ta, verbose=False)

        schedule = ta_transient.ramp_schedule(
            args.i_op, t_ramp=args.t_ramp, t_hold=args.t_hold,
            n_ramp=args.n_ramp, n_hold=args.n_hold)
        print("schedule:", flush=True)
        for t, I, dt in schedule:
            print(f"  t={t:.1f}s  I={I:.2f}A  dt={dt:.1f}s", flush=True)

        hist = newton_ta.march(
            ta, domain, uniform, ic, nm, schedule,
            max_outer=args.max_outer, min_outer=args.min_outer,
            stall_tol=args.stall_tol, bootstrap_iters=args.bootstrap_iters,
            verbose=True, t_relax=args.t_relax, spike_check=args.spike_check)

        for h in hist:
            result["steps"].append(dict(
                step_index=h["step_index"], t=h["t"], I=h["I"], dt=h["dt"],
                converged=bool(h["converged"]), stop_reason=h["stop_reason"],
                n_outer=h["n_outer"], scif_mT=h["scif_mT"],
                scif_hist_tail=h["scif_hist_tail"],
                total_snes_iters=h["total_snes_iters"]))

        n_conv = sum(1 for s in result["steps"] if s["converged"])
        print(f"\n=== {args.tag} DONE: {n_conv}/{len(hist)} steps genuinely "
              f"converged (stall) ===", flush=True)
        result["ok"] = True
        result["n_converged"] = n_conv
        result["n_steps"] = len(hist)

    except Exception as exc:   # noqa: BLE001 -- must record, not crash silently
        import traceback
        print(f"\n=== {args.tag} FAILED: {exc} ===", flush=True)
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["wall_s"] = time.monotonic() - t_start
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        try:
            os.remove(params.mesh_filename)
        except OSError:
            pass

    print(f"wall time: {result['wall_s']:.1f}s", flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
