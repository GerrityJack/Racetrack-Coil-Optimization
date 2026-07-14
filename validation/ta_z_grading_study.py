"""
ta_z_grading_study.py
=====================
Are graded sub-slabs (small cells at the tape edges, coarse bulk) the
right way to resolve the screening-current profile across the tape width?

Strip theory (Brandt/Norris): a strip carrying transport fraction
i = I/Ic keeps a flux-free core of half-width a·sqrt(1−i²); the screening
structure lives between there and the tape edges.  For i ~ 0.2–0.75 that
zone is ~0.1–0.7 mm from each edge of the 4 mm tape, so a graded mesh
[0.075, 0.15, 0.55, 0.15, 0.075]·w (0.3 mm edge cells) should capture it
at the cost of a uniform nz=5 mesh.

This study solves the per-layer T-A problem at I_design with the
SUB-CRITICAL (strong-tape / Low_Field) Ic dataset — the regime where
penetration fronts exist — on three meshes:

    uniform nz=3   (production default)
    uniform nz=5   (brute-force refinement)
    graded 5-slab  (same cell count as nz=5, edges resolved 2.7× finer)

and compares (a) the bore SCIF and (b) the tangential current profile
J·t̂(z) across the tape width in two pancakes (central + top).

Outputs:
  visualization/ta_z_grading_profiles.png
  console summary

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 validation/ta_z_grading_study.py
"""
import os, sys, time
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
params.ta_picard_tol = 1e-4
params.ta_n_picard   = 150

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import solve as base_solve
import build_mesh
from ic_model import IcModel, NValueModel
from ta_solve import (setup_ta_problem, solve_ta_at_current,
                      _J_from_T, dB_bore_from_dJ)

GRADED = [0.075, 0.15, 0.55, 0.15, 0.075]
CONFIGS = (
    ("uniform nz=3", dict(mesh_nz_per_layer=3, mesh_z_grading=None)),
    ("uniform nz=5", dict(mesh_nz_per_layer=5, mesh_z_grading=None)),
    ("graded 5-slab", dict(mesh_nz_per_layer=5, mesh_z_grading=GRADED)),
)
I_STUDY   = float(params.I_design)
PROF_LAYERS = (3, 0)          # central pancake, top pancake
Y_BAND    = (0.050, 0.058)    # mid-radius band, straight section [m]


def _z_centers():
    return np.array([(t_ + b_) / 2.0 for t_, b_ in
                     zip(params.layer_z_tops, params.layer_z_bottoms)])


def main():
    comm  = MPI.COMM_WORLD
    n_mod = NValueModel(csv_path=params.n_value_csv_filename)
    ic_mod = IcModel()   # strong tape (sub-critical at 200 A)
    L = params.b - params.a

    results = []
    for label, cfg in CONFIGS:
        for k, v in cfg.items():
            setattr(params, k, v)
        t0 = time.time()
        build_mesh.build(write_path=params.mesh_filename)
        md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
        domain = md.mesh
        us = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
        ta = setup_ta_problem(domain, md.cell_tags, md.facet_tags, us,
                              per_layer=True)
        *_, info = solve_ta_at_current(domain, ta, us, I_STUDY,
                                       ic_mod, n_mod, verbose=False,
                                       warm_start=False)
        J_TA   = _J_from_T(ta, domain)
        J_unif = ta["t_hat_coil"] * (I_STUDY / (ta["delta_SC"] * params.w))
        dJ_s   = (J_TA - J_unif) * (ta["delta_SC"] / ta["Lambda"])
        dB     = dB_bore_from_dJ(ta["coil_centroids"], dJ_s, ta["coil_vols"])

        cents  = ta["coil_centroids"]
        zc     = _z_centers()
        assign = np.argmin(np.abs(cents[:, 2][:, None] - zc[None, :]), axis=1)
        J_t    = np.einsum("ij,ij->i", J_TA, ta["t_hat_coil"])

        profiles = {}
        sel_band = ((np.abs(cents[:, 0]) < 0.6 * L)
                    & (cents[:, 1] > Y_BAND[0]) & (cents[:, 1] < Y_BAND[1]))
        for li in PROF_LAYERS:
            m = sel_band & (assign == li)
            z_rel = (cents[m, 2] - zc[li]) * 1e3     # mm within tape
            profiles[li] = (z_rel, J_t[m] / 1e9)      # GA/m²

        wall = time.time() - t0
        n_dofs = us["V"].dofmap.index_map.size_global
        print(f"  {label:14s}: SCIF dBz = {dB[2]*1e3:+8.2f} mT  "
              f"({n_dofs} dofs, {len(cents)} coil cells, "
              f"k={info['n_iters']}, conv={info['converged']}, {wall:.0f} s)")
        results.append(dict(label=label, dB=dB[2]*1e3, profiles=profiles,
                            dofs=n_dofs, ncells=len(cents)))

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(PROF_LAYERS), figsize=(13, 5))
    fig.patch.set_facecolor("#111")
    colors = ("#6fa8ff", "orchid", "tomato")
    J_u = I_STUDY / (params.delta_SC * params.w) / 1e9

    for ax, li in zip(np.atleast_1d(axes), PROF_LAYERS):
        ax.set_facecolor("#0d0d1a")
        for res, c in zip(results, colors):
            z_rel, jt = res["profiles"][li]
            order = np.argsort(z_rel)
            # bin to a line (0.15 mm bins), keep scatter faint
            ax.plot(z_rel, jt, ".", ms=2.5, color=c, alpha=0.25)
            bins = np.arange(-2.0, 2.01, 0.15)
            ib = np.digitize(z_rel, bins)
            zb = [z_rel[ib == b].mean() for b in np.unique(ib)
                  if (ib == b).sum() >= 2]
            jb = [jt[ib == b].mean() for b in np.unique(ib)
                  if (ib == b).sum() >= 2]
            ax.plot(zb, jb, "-o", ms=3.5, lw=1.6, color=c,
                    label=f"{res['label']}  (SCIF {res['dB']:+.1f} mT)")
        ax.axhline(J_u, color="#888", lw=0.8, ls="--",
                   label="uniform J = I/(δ·w)" if li == PROF_LAYERS[0] else None)
        ax.axvline(-2, color="#555", lw=0.7); ax.axvline(2, color="#555", lw=0.7)
        ax.set_xlabel("z within tape  [mm]", color="white")
        ax.set_ylabel("J·t̂  [GA/m²]", color="white")
        ax.set_title(f"pancake {li} ({params.n_turns[li]} turns)",
                     color="white", fontsize=11)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="#555")
        ax.legend(fontsize=7.5, labelcolor="white", facecolor="#222",
                  framealpha=0.6)

    fig.suptitle(
        f"Current profile across the tape width — sub-critical (strong-tape) "
        f"dataset, I = {I_STUDY:.0f} A, Δt = {params.ramp_duration:.0f} s\n"
        f"straight section, y ∈ [{Y_BAND[0]*1e3:.0f}, {Y_BAND[1]*1e3:.0f}] mm"
        f"  |  graded slabs: {GRADED} × w",
        color="white", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(params.VIZ_DIR, "ta_z_grading_profiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved profile figure → {out}")


if __name__ == "__main__":
    main()
