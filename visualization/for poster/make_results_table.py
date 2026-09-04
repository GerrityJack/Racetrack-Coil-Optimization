"""
make_results_table.py
=======================
Poster figure: the champion's validated performance table (nominal value,
limit, and the 15-sample build-tolerance jitter range for each metric) --
matplotlib table rendered directly, white background, poster styling.

Numbers are the validated ones from CLAUDE.md / optimize/studies/
margin_design_search.py + jitter_margin_design.py (Kim Ic model, real
per-layer T-A box uniformity) -- NOT live-recomputed from any npz, since
the jitter sweep across 15 perturbed builds is a standalone study, not
something a single current run of this repo reproduces.

Output: visualization/for poster/results_table.png
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
sys.path.insert(0, _ROOT)
import params

ROWS = [
    ("Bore field\n(Kim Ic model)", "10.49 T", "≥ 10 T", "10.10 – 10.49 T", "PASS 15/15"),
    ("Box uniformity\n(T-A, 30×6mm box)", "0.495 %", "≤ 1 %", "0.338 – 0.517 %", "PASS 15/15"),
    ("Hoop stress", "113 MPa", "≤ 400 MPa", "102 – 113 MPa", "PASS 15/15"),
    ("Bend radius", "8.075 mm", "≥ 7.5 mm", "7.545 – 8.434 mm", "PASS 15/15"),
    ("Face gap", "3.40 mm", "≥ 3.0 mm", "3.00 – 3.84 mm", "PASS 15/15"),
]

COLS = ["Metric", "Nominal", "Limit", "Jitter range\n(15 perturbed builds)", "Result"]
COL_WIDTHS = [0.26, 0.16, 0.16, 0.26, 0.16]


def main():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=ROWS, colLabels=COLS, loc="center",
                    cellLoc="center", colLoc="center", colWidths=COL_WIDTHS)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 3.0)

    n_rows, n_cols = len(ROWS) + 1, len(COLS)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbb")
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("#2b3a55")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f4f8f4" if r % 2 == 0 else "white")
            if c == n_cols - 1:
                cell.set_text_props(color="#1e7d34", fontweight="bold")

    fig.text(0.5, 0.96,
              "Champion design -- validated performance "
              f"(a={params.a*1e3:.1f}mm, b={params.b*1e3:.1f}mm, "
              f"n_turns={params.n_turns}, I={params.I_design:.0f} A)",
              ha="center", va="top", fontsize=14, color="#111")
    fig.text(0.5, 0.02,
              "State bore field as ~10.5 T ± 0.5 T -- the Ic(B) "
              "extrapolation above the measured 8T dataset ceiling is "
              "the dominant uncertainty, not build tolerance.",
              ha="center", va="bottom", fontsize=9.5, style="italic",
              color="#555")

    out = os.path.join(_HERE, "results_table.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
