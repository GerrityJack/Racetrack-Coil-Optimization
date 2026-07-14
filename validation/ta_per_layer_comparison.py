"""
ta_per_layer_comparison.py
==========================
Accuracy study: replicated central-tape T-A vs true per-layer T-A.

The production solver (per_layer=False) solves T in ONE representative
tape (the central z-layer) and copies its screening-current pattern to
all other layers.  This script solves the same problem with a separate
T problem per layer (per_layer=True) and quantifies what the
replication approximation does to:

  1. the total bore SCIF ΔBz,
  2. each layer's SCIF contribution,
  3. the RADIAL screening-current profile in each layer — including the
     inner radial band of the 500-turn layers that the central (400-turn)
     layer cannot cover and the KD-tree replication must extrapolate.

Both solves are run COLD at I_design with a tightened Picard tolerance
so tolerance noise (±3-4 % at the production 1e-3) does not mask the
layer effect.

Outputs:
  visualization/ta_per_layer_comparison.png
  console summary table

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 validation/ta_per_layer_comparison.py
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
# Tight tolerance: SCIF carries ±3-4 % spread at the production 1e-3,
# comparable to the effect being measured.
params.ta_picard_tol = 1e-4
params.ta_n_picard   = 150

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import solve as base_solve
import build_mesh
from ic_model import IcModel, NValueModel
from ta_solve import (setup_ta_problem, solve_ta_at_current,
                      _J_from_T, dB_bore_from_dJ)

I_STUDY = float(params.I_design)


def main():
    comm = MPI.COMM_WORLD
    build_mesh.build(write_path=params.mesh_filename)
    md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    domain = md.mesh

    ic_mod = IcModel(csv_path=params.shanghai_csv_filename)
    n_mod  = NValueModel(csv_path=params.n_value_csv_filename)
    us = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)

    results = {}
    for mode, flag in (("replicated", False), ("per-layer", True)):
        print(f"\n{'='*62}\n  MODE: {mode}\n{'='*62}")
        ta = setup_ta_problem(domain, md.cell_tags, md.facet_tags, us,
                              per_layer=flag)
        t0 = time.time()
        *_, info = solve_ta_at_current(domain, ta, us, I_STUDY,
                                       ic_mod, n_mod, verbose=True,
                                       warm_start=False)
        wall = time.time() - t0

        J_TA   = _J_from_T(ta, domain)
        J_unif = ta["t_hat_coil"] * (I_STUDY / (ta["delta_SC"] * params.w))
        dJ_s   = (J_TA - J_unif) * (ta["delta_SC"] / ta["Lambda"])
        dB     = dB_bore_from_dJ(ta["coil_centroids"], dJ_s, ta["coil_vols"])

        results[mode] = dict(ta=ta, J_TA=J_TA, J_unif=J_unif, dJ_s=dJ_s,
                             dB=dB, wall=wall, info=info)
        print(f"  [{mode}] SCIF ΔBz = {dB[2]*1e3:+.2f} mT   "
              f"({info['n_iters']} iters, {wall:.1f} s, "
              f"converged={info['converged']})")

    _analyse(results)


def _layer_assignment(cents):
    z_centers = np.array([(t_ + b_) / 2.0 for t_, b_ in
                          zip(params.layer_z_tops, params.layer_z_bottoms)])
    return np.argmin(np.abs(cents[:, 2][:, None] - z_centers[None, :]), axis=1)


def _analyse(results):
    rep, lay = results["replicated"], results["per-layer"]
    ta     = lay["ta"]
    cents  = ta["coil_centroids"]
    dV     = ta["coil_vols"]
    t_hat  = ta["t_hat_coil"]
    assign = _layer_assignment(cents)
    nL     = params.n_layers
    L      = params.b - params.a
    central_layer = int(np.argmin(np.abs(
        np.array([(t_ + b_) / 2 for t_, b_ in
                  zip(params.layer_z_tops, params.layer_z_bottoms)]))))

    # ── per-layer SCIF contributions and J differences ───────────────────
    print(f"\n{'='*74}")
    print(f"  PER-LAYER COMPARISON  (I = {I_STUDY:.0f} A, tol = "
          f"{params.ta_picard_tol:.0e}, cold starts)")
    print(f"{'='*74}")
    print(f"  {'layer':>5} {'turns':>6} {'r_in [mm]':>10} "
          f"{'ΔBz rep [mT]':>13} {'ΔBz lay [mT]':>13} {'ΔJ L2 diff':>11}")

    dBz_rep_layers, dBz_lay_layers = [], []
    for i in range(nL):
        m = assign == i
        dBz_r = dB_bore_from_dJ(cents[m], rep["dJ_s"][m], dV[m])[2] * 1e3
        dBz_l = dB_bore_from_dJ(cents[m], lay["dJ_s"][m], dV[m])[2] * 1e3
        dBz_rep_layers.append(dBz_r)
        dBz_lay_layers.append(dBz_l)
        dJ_rep = rep["J_TA"][m] - rep["J_unif"][m]
        dJ_lay = lay["J_TA"][m] - lay["J_unif"][m]
        l2 = (np.linalg.norm(dJ_lay - dJ_rep)
              / max(np.linalg.norm(dJ_lay), 1e-30))
        mark = "  ← central (solved tape in replicated mode)" \
               if i == central_layer else ""
        print(f"  {i:>5} {params.n_turns[i]:>6} "
              f"{params.a_inner_list[i]*1e3:>10.2f} "
              f"{dBz_r:>13.2f} {dBz_l:>13.2f} {l2*100:>10.1f}%{mark}")

    tot_r, tot_l = rep["dB"][2] * 1e3, lay["dB"][2] * 1e3
    print(f"  {'-'*72}")
    print(f"  {'TOTAL':>23} {tot_r:>13.2f} {tot_l:>13.2f}   "
          f"(difference {abs(tot_l-tot_r)/abs(tot_l)*100:.1f}% of per-layer)")
    print(f"\n  runtimes: replicated {rep['wall']:.1f} s "
          f"({rep['info']['n_iters']} iters)  |  "
          f"per-layer {lay['wall']:.1f} s ({lay['info']['n_iters']} iters)")

    # ── figure: radial profiles per layer + SCIF bars ────────────────────
    straight = np.abs(cents[:, 0]) < 0.6 * L      # straight-section cells
    r_in_central = params.a_inner_list[central_layer]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor("#111")
    axes = axes.ravel()

    for i in range(nL):
        ax = axes[i]
        ax.set_facecolor("#0d0d1a")
        m = (assign == i) & straight
        y = cents[m, 1] * 1e3                     # radial coordinate [mm]
        for res, color, label in ((rep, "#6fa8ff", "replicated"),
                                  (lay, "tomato", "per-layer")):
            dJt = np.einsum("ij,ij->i",
                            res["J_TA"][m] - res["J_unif"][m], t_hat[m])
            order = np.argsort(y)
            # bin-average for a readable profile
            bins = np.arange(y.min() - 1, y.max() + 2, 2.0)
            ib   = np.digitize(y, bins)
            yb   = np.array([y[ib == b].mean() for b in np.unique(ib)])
            jb   = np.array([dJt[ib == b].mean() / 1e9
                             for b in np.unique(ib)])
            ax.plot(yb, jb, "-o", ms=3, lw=1.4, color=color, label=label)

        # shade the radial band the central layer cannot cover
        r_in_i = params.a_inner_list[i]
        if r_in_i < r_in_central - 1e-6:
            ax.axvspan(r_in_i * 1e3, r_in_central * 1e3, color="yellow",
                       alpha=0.15, label="extrapolated band")
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title(f"layer {i}  ({params.n_turns[i]} t)"
                     + ("  [central]" if i == central_layer else ""),
                     color="white", fontsize=10)
        ax.set_xlabel("y (radial)  [mm]", color="white", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("ΔJ·t̂  [GA/m²]", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="#555")
        if i == 0:
            ax.legend(fontsize=7, labelcolor="white", facecolor="#222",
                      framealpha=0.6)

    # last panel: per-layer SCIF contribution bars
    axb = axes[nL]
    axb.set_facecolor("#0d0d1a")
    x = np.arange(nL)
    axb.bar(x - 0.2, dBz_rep_layers, 0.4, color="#6fa8ff", label="replicated")
    axb.bar(x + 0.2, dBz_lay_layers, 0.4, color="tomato",  label="per-layer")
    axb.set_xticks(x)
    axb.set_xticklabels([f"{i}\n{params.n_turns[i]}t" for i in range(nL)],
                        fontsize=7)
    axb.set_ylabel("layer SCIF ΔBz  [mT]", color="white", fontsize=8)
    axb.set_title(f"SCIF by layer — totals: rep {tot_r:.0f} / "
                  f"lay {tot_l:.0f} mT", color="white", fontsize=10)
    axb.tick_params(colors="white", labelsize=7)
    for sp in axb.spines.values():
        sp.set_edgecolor("#444")
    axb.grid(True, alpha=0.2, color="#555", axis="y")
    axb.legend(fontsize=7, labelcolor="white", facecolor="#222",
               framealpha=0.6)

    fig.suptitle(
        f"Replicated central-tape vs per-layer T-A — screening current "
        f"ΔJ·t̂ radial profiles (straight section)\n"
        f"I = {I_STUDY:.0f} A,  Δt = {params.ramp_duration:.0f} s,  "
        f"tol = {params.ta_picard_tol:.0e}  |  yellow band = radii the "
        f"central tape cannot cover (KD-tree extrapolation)",
        color="white", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(params.VIZ_DIR, "ta_per_layer_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved comparison figure → {out}")


if __name__ == "__main__":
    main()
