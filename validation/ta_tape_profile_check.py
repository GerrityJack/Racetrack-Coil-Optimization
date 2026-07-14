"""
ta_tape_profile_check.py
========================
Does the solved screening-current distribution across the tape width look
like the physics says it should?

The textbook profile ("|J| high at both edges, low in the centre") is the
PURE-TRANSPORT thin-strip solution (Norris):

    b = (w/2)·sqrt(1 − i²),   i = I/Ic
    J(z) = Jc                                  for b < |z| < w/2
    J(z) = (2Jc/π)·atan( sqrt((a²−b²)/(b²−z²)) )   for |z| < b   (a = w/2)

Inside a winding the tape also sees a perpendicular (radial) field
component B_n that drives an ANTISYMMETRIC magnetization current
(+Jc penetration from one edge, −Jc from the other).  The physical
profile is the nonlinear combination: symmetric edge-peaked transport
where B_n ≈ 0, increasingly asymmetric (bulk offset + single-edge
reversal) where B_n dominates.

This script extracts J·t̂(z) across the tape width from the CONVERGED
per-layer T-A solution at several (pancake, radial-band) sample points,
annotates each with the local i = I/Ic and band-mean B_n, and overlays
the Norris pure-transport profile for the local i as the reference.

Output: visualization/ta_tape_profiles.png + console table.

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 validation/ta_tape_profile_check.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import solve as base_solve
import build_mesh
from ic_model import IcModel, NValueModel, angle_with_normal_deg
from ta_solve import setup_ta_problem, solve_ta_at_current, _J_from_T

I_OP = float(params.I_design)

# (pancake index, radial band [m], label)
SAMPLES = [
    (0, (0.050, 0.058), "pancake 0 (top), mid radius"),
    (0, (0.062, 0.068), "pancake 0 (top), outer turns"),
    (3, (0.050, 0.058), "pancake 3 (central), mid radius"),
    (3, (0.062, 0.068), "pancake 3 (central), outer turns"),
    (6, (0.062, 0.068), "pancake 6 (bottom), outer turns"),
    (3, (0.040, 0.047), "pancake 3 (central), inner turns"),
]


def norris_profile(z, half_w, i, Jc):
    """Pure-transport thin-strip current density profile (Norris)."""
    i = min(max(i, 1e-6), 0.999)
    b = half_w * np.sqrt(1.0 - i * i)
    J = np.full_like(z, Jc, dtype=float)
    core = np.abs(z) < b
    J[core] = (2.0 * Jc / np.pi) * np.arctan(
        np.sqrt((half_w**2 - b**2) / (b**2 - z[core]**2)))
    return J


def main():
    comm = MPI.COMM_WORLD
    build_mesh.build(write_path=params.mesh_filename)
    md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    domain = md.mesh
    ic_mod = IcModel()
    n_mod  = NValueModel(csv_path=params.n_value_csv_filename)
    us = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
    ta = setup_ta_problem(domain, md.cell_tags, md.facet_tags, us,
                          per_layer=True)
    *_, info = solve_ta_at_current(domain, ta, us, I_OP, ic_mod, n_mod,
                                   verbose=True, warm_start=False)
    print(f"\nsolve: k={info['n_iters']} converged={info['converged']} "
          f"SCIF={info['scif_mT']:+.2f} mT")

    J_TA  = _J_from_T(ta, domain)
    cents = ta["coil_centroids"]
    B     = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]
    t_hat = ta["t_hat_coil"]
    n_hat = ta["n_hat_coil"]
    J_t   = np.einsum("ij,ij->i", J_TA, t_hat)
    B_n   = np.einsum("ij,ij->i", B, n_hat)

    Bmag  = np.linalg.norm(B, axis=1)
    theta = angle_with_normal_deg(B, n_hat)
    Ic, _ = ic_mod.critical_current(Bmag, theta)
    Jc_vol = Ic / (ta["delta_SC"] * ic_mod.tape_width)

    zc = np.array([(t_ + b_) / 2 for t_, b_ in
                   zip(params.layer_z_tops, params.layer_z_bottoms)])
    assign = np.argmin(np.abs(cents[:, 2][:, None] - zc[None, :]), axis=1)
    L = params.b - params.a
    straight = np.abs(cents[:, 0]) < 0.6 * L
    half_w = params.w / 2

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.patch.set_facecolor("#111")

    print(f"\n{'sample':42s} {'i=I/Ic':>7} {'<B_n> [T]':>10} "
          f"{'J_edge/Jc':>10}")
    for ax, (li, (y0, y1), label) in zip(axes.ravel(), SAMPLES):
        ax.set_facecolor("#0d0d1a")
        m = straight & (assign == li) & (cents[:, 1] > y0) & (cents[:, 1] < y1)
        z_rel = (cents[m, 2] - zc[li])                     # [m]
        jt    = J_t[m]
        jc_b  = float(np.mean(Jc_vol[m]))
        bn_b  = float(np.mean(B_n[m]))
        i_loc = I_OP / (jc_b * ta["delta_SC"] * params.w)  # = J_unif/Jc

        # bin by z (slab structure)
        order = np.argsort(z_rel)
        bins = np.linspace(-half_w, half_w, 12)
        ib = np.digitize(z_rel, bins)
        zb = np.array([z_rel[ib == k].mean() for k in np.unique(ib)
                       if (ib == k).sum() >= 1])
        jb = np.array([jt[ib == k].mean() for k in np.unique(ib)
                       if (ib == k).sum() >= 1])

        ax.plot(z_rel * 1e3, jt / 1e9, ".", ms=3, color="tomato", alpha=0.35)
        ax.plot(zb * 1e3, jb / 1e9, "-o", ms=4, lw=1.8, color="tomato",
                label="T-A solution")

        zz = np.linspace(-half_w * 0.999, half_w * 0.999, 400)
        ax.plot(zz * 1e3, norris_profile(zz, half_w, i_loc, jc_b) / 1e9,
                "--", lw=1.4, color="#6fa8ff",
                label=f"Norris transport (i={i_loc:.2f})")
        ax.axhline(jc_b / 1e9, color="#888", lw=0.8, ls=":",
                   label="±Jc(B_loc)")
        ax.axhline(-jc_b / 1e9, color="#888", lw=0.8, ls=":")
        ax.axhline(I_OP / (params.delta_SC * params.w) / 1e9,
                   color="#555", lw=0.8, label="uniform J")

        ax.set_title(f"{label}\n⟨B_n⟩ = {bn_b:+.2f} T,  i = {i_loc:.2f}",
                     color="white", fontsize=9.5)
        ax.set_xlabel("z within tape [mm]", color="white", fontsize=8)
        ax.set_ylabel("J·t̂  [GA/m²]", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="#555")
        if li == SAMPLES[0][0] and (y0, y1) == SAMPLES[0][1]:
            ax.legend(fontsize=7, labelcolor="white", facecolor="#222",
                      framealpha=0.6)

        edge = np.abs(z_rel) > half_w * 0.6
        j_edge = np.abs(jt[edge]).max() / jc_b if edge.any() else np.nan
        print(f"{label:42s} {i_loc:7.2f} {bn_b:+10.3f} {j_edge:10.2f}")

    fig.suptitle(
        f"Screening-current profile across the tape width — converged "
        f"per-layer T-A, I = {I_OP:.0f} A, Δt = {params.ramp_duration:.0f} s\n"
        f"blue dashed = Norris pure-transport reference (valid where "
        f"⟨B_n⟩ ≈ 0); asymmetry/reversal grows with |B_n|",
        color="white", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(params.VIZ_DIR, "ta_tape_profiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
