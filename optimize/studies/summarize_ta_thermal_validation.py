"""
summarize_ta_thermal_validation.py -- quick readable table from
optimize/runs/ta_thermal_validation/summary.csv, the raw output of
ta_thermal_validation.py (2026-09-12 4.2K investigation). Read-only,
no solving.
"""
import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
CSV_PATH = os.path.join(_ROOT, "optimize", "runs", "ta_thermal_validation",
                        "summary.csv")

COLS = ["phase", "tag", "I_A", "alpha_high", "alpha_low", "n_iters",
        "converged_flag", "rel_err_final", "worst_margin", "p5_margin",
        "wall_s", "error"]


def main():
    with open(CSV_PATH) as fh:
        rows = list(csv.DictReader(fh))
    print(f"{len(rows)} rows in {CSV_PATH}\n")
    header = "  ".join(f"{c:>14s}" for c in COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = []
        for c in COLS:
            v = r.get(c, "")
            if c in ("rel_err_final",) and v not in ("", None):
                v = f"{float(v):.3e}"
            elif c in ("worst_margin", "p5_margin", "wall_s") and v not in ("", None):
                v = f"{float(v):.3f}"
            vals.append(f"{str(v):>14.14s}")
        print("  ".join(vals))


if __name__ == "__main__":
    main()
