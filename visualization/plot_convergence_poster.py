"""
plot_convergence_poster.py
============================
Poster-quality convergence figure: ONE clean "best tape length found so
far" staircase, across every cmaes_search.py run to date (reads the
cumulative cfg.CMAES_MASTER_LOG, in the order rows were appended -- run 1,
then run 2, then run 3), so it tells the whole search story rather than
just the last run. No title, minimal large text (axis labels + one
annotation on the final value), same black/orange/purple theme as
field_3d_poster.png for visual consistency on the poster.

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


def _autocrop(path, pad=40):
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (0, 0, 0))
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
        if ok:
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
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.fill_between(xs, ys, max(ys) * 1.05, color="#7d1ea8", alpha=0.25,
                    linewidth=0)
    ax.plot(xs, ys, color="#ff8f00", linewidth=5, solid_capstyle="round")

    ax.scatter([xs[-1]], [ys[-1]], s=260, color="#ff8f00",
              edgecolor="white", linewidth=2, zorder=5)
    ax.annotate(f"{ys[-1]:.2f} km",
               xy=(xs[-1], ys[-1]), xytext=(-18, 26),
               textcoords="offset points", ha="right",
               color="white", fontsize=30, fontweight="bold")
    ax.annotate(f"{ys[0]:.2f} km",
               xy=(xs[0], ys[0]), xytext=(18, 14),
               textcoords="offset points", ha="left",
               color="white", fontsize=22, alpha=0.85)

    for spine in ax.spines.values():
        spine.set_color("#555")
    ax.tick_params(colors="white", labelsize=18, length=8, width=1.2)
    ax.set_xlabel("designs evaluated", color="white", fontsize=26,
                 labelpad=12)
    ax.set_ylabel("tape needed  (km)", color="white", fontsize=26,
                 labelpad=12)
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xlim(0, xs[-1] * 1.03)
    ax.set_ylim(0, max(ys) * 1.15)

    out = os.path.join(params.VIZ_DIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    _autocrop(out, pad=50)
    print(f"  Wrote {out}")
    return out


if __name__ == "__main__":
    plot_convergence_poster()
