"""
plot_field_poster.py
=====================
Poster-quality figure: |B| field distribution, isometric view only.
No title, no legend, minimal text (colorbar label/ticks only, large font).
Orange/black/purple theme: inferno colormap truncated below its pale-yellow
tail (np.linspace(0, 0.88, ...)) so the brightest end stays vivid orange,
keeping the perceptually-uniform, colorblind-safe ramp already validated
by the magma/inferno family used elsewhere in this project.

Reuses the same npz + expansion-to-full-system logic as plot_3d.py's
field_3d.png (imports its helpers directly, no duplication).

Input: solve/racetrack_fields.npz (run solve/solve.py first if missing).
Output: visualization/field_3d_poster.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: registers projection
from PIL import Image, ImageChops

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from plot_3d import _expand_to_full_system, _racetrack_xy

# black -> deep purple -> red-orange -> vivid orange (inferno's own ramp,
# truncated before its pale-yellow tail so orange stays the brightest hue)
_POSTER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "inferno_poster", plt.cm.inferno(np.linspace(0.0, 0.88, 256)))


def plot_field_poster(npz_data, out_name="field_3d_poster.png"):
    centroids_raw = npz_data["coil_centroids"]
    Bmag_raw = np.linalg.norm(npz_data["coil_B"], axis=1)
    g = params.coil_half_gap

    if getattr(params, "use_eighth_symmetry", False):
        centroids, Bmag = _expand_to_full_system(centroids_raw, Bmag_raw)
    else:
        centroids, Bmag = centroids_raw, Bmag_raw
        if getattr(params, "two_coil_mode", False):
            c2 = centroids_raw.copy(); c2[:, 2] += 2.0 * g
            centroids = np.vstack([centroids_raw, c2])
            Bmag = np.tile(Bmag_raw, 2)

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0, 0, 0, 0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)

    # Subtle neutral geometry guides (layer boundary rings), no rainbow
    # per-layer colors -- kept translucent white so they don't compete with
    # the orange/black/purple field-color story.
    for z_off in (0.0, 2.0 * g):
        for i in range(params.n_layers):
            xo, yo = _racetrack_xy(params.a_out)
            for z_face in (params.layer_z_tops[i], params.layer_z_bottoms[i]):
                zo = np.full_like(xo, (z_face + z_off) * 1e3)
                ax.plot(xo * 1e3, yo * 1e3, zo,
                        color="white", lw=0.5, alpha=0.12)

    sc = ax.scatter(centroids[:, 0] * 1e3, centroids[:, 1] * 1e3,
                    centroids[:, 2] * 1e3, c=Bmag, cmap=_POSTER_CMAP,
                    s=5, alpha=0.65, linewidths=0, rasterized=True)

    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect((1, 1, 0.6))

    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, aspect=18)
    cb.set_label("|B|  (T)", color="white", fontsize=26, labelpad=14)
    cb.ax.tick_params(color="white", labelsize=20, length=6, width=1.2)
    plt.setp(cb.ax.get_yticklabels(), color="white")
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(1.0)

    out = os.path.join(params.VIZ_DIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    _autocrop(out, pad=40)
    print(f"  Wrote {out}")
    return out


def _autocrop(path, pad=40):
    """3-D axes reserve a lot of dead black margin regardless of framing --
    crop tight to the actual content (subject + colorbar) with a small
    padding, since matplotlib's bbox_inches='tight' doesn't help here."""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (0, 0, 0))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(im.width, r + pad); b = min(im.height, b + pad)
    im.crop((l, t, r, b)).save(path)


def main():
    npz_path = os.path.join(_ROOT, "solve", "racetrack_fields.npz")
    plot_field_poster(np.load(npz_path))


if __name__ == "__main__":
    main()
