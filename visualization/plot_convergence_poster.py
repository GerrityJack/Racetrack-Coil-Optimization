"""
plot_convergence_poster.py
============================
Poster-quality convergence figure: ONE clean "best tape length found so
far" staircase, across every cmaes_search.py run to date (reads the
cumulative cfg.CMAES_MASTER_LOG, in the order rows were appended -- run 1,
then run 2, then run 3, ...), so it tells the whole search story rather
than just the last run.

2026-07-27: white background (poster theme, matching field_3d_poster.png)
and updated to correctly handle the master log's newest data -- the
2026-07-27 day_search.py widened search (n_layers in {6,8,10,12,14,16})
added ~16k more evaluations, several of which show all_constraints_ok=True
in the log at LOWER tape than the champion. That flag reflects only the
coarse screen's B_target+hoop check (uniformity was removed from the
CMA-ES fitness/constraints entirely once found unreliable -- see
CLAUDE.md's "Coarse-screen SCIF proxy found unreliable" section) -- every
one of those specific candidates was then checked with the real per-layer
T-A solve (optimize/day_search.py Phase B) and FAILED the true uniformity
target. Naively feeding the full log into the original "best so far" logic
would draw a dip below the champion's real value for a design that's
actually invalid. EXCLUDED_RUN_TAGS below names exactly those runs so the
trustworthy staircase correctly stays flat through them instead -- more
designs evaluated, confirmed no real improvement, not a fabricated one.

Deliberately does NOT show the raw per-evaluation scatter (fitness values
for infeasible/violating points range into the thousands from the penalty
term -- plotting them would force a log scale and bury the one message a
poster needs: it got better, and here's where it landed).

Output: visualization/cmaes_convergence_poster.png
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import opt_config as cfg

# 2026-07-27 day_search.py Phase A widened-search runs (n_layers in
# {6,8,10,12,14,16}, including the first, since-corrected 18mm-floor
# attempt for n_layers=6). Coarse-screen-only (all_constraints_ok there
# means B_target+hoop only, NOT uniformity), and every one of these
# specific candidates was later shown by Phase B's real T-A validation to
# fail the true uniformity target -- see optimize/day_search_report.md.
# Excluded from the "best so far" trajectory below so they can't
# masquerade as a real improvement; every other run (including the
# champion's own, run_20260723_124414) is unaffected.
EXCLUDED_RUN_TAGS = {
    "run_20260727_001227", "run_20260727_001402", "run_20260727_010926",
    "run_20260727_021209", "run_20260727_032221", "run_20260727_043344",
    "run_20260727_063037",
}


def _autocrop(path, pad=40, bg_color=(255, 255, 255)):
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, bg_color)
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(im.width, r + pad); b = min(im.height, b + pad)
    im.crop((l, t, r, b)).save(path)


def plot_convergence_poster(out_name="cmaes_convergence_poster.png"):
    path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    best_so_far = []
    cur = float("inf")
    for i, r in enumerate(rows, start=1):
        ok = str(r.get("all_constraints_ok", "")).strip().lower() == "true"
        if ok and r.get("run_tag") not in EXCLUDED_RUN_TAGS:
            try:
                tape = float(r["tape_km"])
                if tape < cur:
                    cur = tape
            except (TypeError, ValueError):
                pass
        if cur < float("inf"):
            best_so_far.append((i, cur))

    xs = [p[0] for p in best_so_far]
    ys = [p[1] for p in best_so_far]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(xs, ys, max(ys) * 1.05, color="#7d1ea8", alpha=0.12,
                    linewidth=0)
    ax.plot(xs, ys, color="#e8792a", linewidth=5, solid_capstyle="round")

    ax.scatter([xs[-1]], [ys[-1]], s=260, color="#e8792a",
              edgecolor="black", linewidth=2, zorder=5)
    ax.annotate(f"{ys[-1]:.2f} km",
               xy=(xs[-1], ys[-1]), xytext=(-18, 26),
               textcoords="offset points", ha="right",
               color="black", fontsize=30, fontweight="bold")
    ax.annotate(f"{ys[0]:.2f} km",
               xy=(xs[0], ys[0]), xytext=(18, 14),
               textcoords="offset points", ha="left",
               color="black", fontsize=22, alpha=0.85)

    for spine in ax.spines.values():
        spine.set_color("#888")
    ax.tick_params(colors="black", labelsize=18, length=8, width=1.2)
    ax.set_xlabel("designs evaluated", color="black", fontsize=26,
                 labelpad=12)
    ax.set_ylabel("tape needed  (km)", color="black", fontsize=26,
                 labelpad=12)
    ax.grid(True, alpha=0.2, color="#888")
    ax.set_xlim(0, xs[-1] * 1.03)
    ax.set_ylim(0, max(ys) * 1.15)

    out = os.path.join(params.VIZ_DIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _autocrop(out, pad=50, bg_color=(255, 255, 255))
    print(f"  Wrote {out}")
    return out


if __name__ == "__main__":
    plot_convergence_poster()
