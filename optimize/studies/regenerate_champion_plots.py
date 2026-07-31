"""
regenerate_champion_plots.py — one-off, 2026-07-27
====================================================
The standard cmaes_{convergence,constraints,variables,overview}.png
figures were last auto-regenerated at 08:04 today by day_search.py's
Phase A n_layers=16 job (the LAST cmaes_search.py subprocess that ran) --
every cmaes_search.py run overwrites those files with ITS OWN history,
and n_layers=16 was one of the widened-search candidates that FAILED
real T-A validation in Phase B, not the actual champion. So the figures
currently on disk show a superseded/rejected run, not the real result.

This reconstructs the champion's own original search
(run_tag=run_20260723_124414, the double_pancake_search.py n_layers=6
job that found the current champion) from the cumulative master log
(optimize/runs/cmaes_all_evaluations.csv, append-only, never overwritten) and
re-runs cmaes_search.py's own _make_plots() against it -- reusing the
exact existing plotting code so the regenerated figures are visually
identical in style to what that run would have produced live, without
spending ~85 minutes re-running the actual CMA-ES search.

Run once: /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
    optimize/regenerate_champion_plots.py
"""
import csv, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUN_TAG = "run_20260723_124414"

_NUMERIC_FIELDS = ("eval", "fitness", "a_mm", "b_mm", "gap_mm",
                   "face_gap_mm", "n_total", "tape_km", "B_target_T",
                   "uniformity_pct", "hoop_MPa", "delam_MPa",
                   "delam_scr_MPa", "I_quench_A", "I_op_A", "clip_frac")
_BOOL_FIELDS = ("feasible", "all_constraints_ok")


def _coerce(row):
    out = dict(row)
    for k in _NUMERIC_FIELDS:
        # infeasible/geometry-violation rows never got the FEM-derived
        # fields populated (_record() only adds them "if r is not None") --
        # empty string there, same as the live in-memory row would have
        # simply omitted the key. NaN is a safe stand-in: _make_plots()
        # only ever reads these fields from feasible_rows, which already
        # excludes exactly these rows.
        out[k] = float(row[k]) if row[k] != "" else float("nan")
    out["eval"] = int(out["eval"])
    for k in _BOOL_FIELDS:
        out[k] = str(row[k]).strip().lower() == "true"
    return out


def main():
    import opt_config as cfg
    import cmaes_search as cs

    master_path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    with open(master_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["run_tag"] == RUN_TAG]
    if not rows:
        raise SystemExit(f"No rows found for run_tag={RUN_TAG!r} in {master_path}")

    history = [_coerce(r) for r in rows]
    history.sort(key=lambda r: r["eval"])

    best = None
    for r in history:
        if r["all_constraints_ok"] and (best is None or r["fitness"] < best["fitness"]):
            best = r
    if best is None:
        raise SystemExit("No all-constraints-ok row found -- can't set _best")

    cs._history = history
    cs._best = dict(best)
    cs._run_tag = RUN_TAG
    cs._eval_count = history[-1]["eval"]
    cs._last_flushed_idx = len(history)

    print(f"Loaded {len(history)} evaluations for {RUN_TAG}, "
         f"best tape={best['tape_km']:.4f}km at eval {best['eval']}")
    cs._make_plots()
    print("Done -- cmaes_convergence.png, cmaes_constraints.png, "
         "cmaes_variables.png, cmaes_overview.png regenerated from the "
         "champion's actual run.")


if __name__ == "__main__":
    main()
