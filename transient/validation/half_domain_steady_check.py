"""
half_domain_steady_check.py -- validation gate for the half-domain mesh.

Runs the SAME validated, unmodified single-implicit-dt=600s-step T-A solve
(ta_solve.solve_ta_at_current) that produces this project's ~641 mT ground
truth on the eighth-domain, but on the half-domain mesh (mesh/build_mesh_half.py)
instead. Reports field diagnostics comparable to the eighth-domain reference
(validation/loss_sanity_check.py's "j/jc mean 0.59, 26% over-critical") plus
the on-axis SCIF using a mirror count adapted for the half-domain (only the
z=coil_half_gap mirror applies -- coil_centroids here already cover the FULL
racetrack loop, so mirroring in x/y as ta_solve.dB_bore_from_dJ does would
double/quadruple count).

Also reports the |A| : |curl(A)| ratio, the direct measure of gauge
pollution documented in transient/induction.py (eighth-domain reference is
~1e10) -- this is the mechanism the half-domain hypothesis is meant to fix.

Usage: <env python3> transient/validation/half_domain_steady_check.py
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx import mesh as dmesh

import params
import solve as base_solve
import ta_solve
from ic_model import IcModel, NValueModel
import build_mesh_half


def dB_bore_from_dJ_half(cents, dJ_s, dV, bore_pt=None):
    """Bore Bz from screening current on the HALF domain (full racetrack
    loop for coil 1, no x/y quadrant mirroring -- only the z-mirror for
    coil 2). Mirrors ta_solve.dB_bore_from_dJ's z-mirror piece exactly."""
    g = float(params.coil_half_gap)
    if bore_pt is None:
        bore_pt = np.array([0.0, 0.0, g])
    bore_pt = np.asarray(bore_pt, dtype=float).reshape(3)

    dB = np.zeros(3)
    for coil in (1, 2):
        cc = cents.copy()
        Jc = dJ_s
        if coil == 2:
            cc[:, 2] = 2.0 * g - cc[:, 2]
            Jc = dJ_s * np.array([1.0, 1.0, -1.0])
        r = bore_pt[None, :] - cc
        r_mag = np.linalg.norm(r, axis=1)
        r_hat = r / np.maximum(r_mag, 1e-12)[:, None]
        dB += np.sum((1e-7 * dV / np.maximum(r_mag, 1e-12) ** 2)[:, None]
                     * np.cross(Jc, r_hat), axis=0)
    return dB


def main():
    mesh_path = os.path.join(params.MESH_DIR, "racetrack_mesh_half.msh")
    if not os.path.exists(mesh_path):
        print("[half-check] building half-domain mesh ...")
        build_mesh_half.build_half(write_path=mesh_path, verbose=False)

    comm = MPI.COMM_WORLD
    md = gmshio.read_from_msh(mesh_path, comm, rank=0, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    uniform_setup = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform_setup)

    ic_model = IcModel(params.shanghai_csv_filename)
    n_model = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    t0 = time.time()
    A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
        domain, ta, uniform_setup, I, ic_model, n_model, verbose=True,
        warm_start=False)
    wall = time.time() - t0

    coil = ta["coil_cells"]
    B_coil = B_h.x.array.reshape(-1, 3)[coil]
    Bmag = np.linalg.norm(B_coil, axis=-1)

    J_coil = ta_solve._J_from_T(ta, domain)
    n_hat = ta["n_hat_coil"]
    J_dot_n = np.einsum("ij,ij->i", J_coil, n_hat)
    J_inplane = J_coil - J_dot_n[:, None] * n_hat
    Jmag = np.linalg.norm(J_inplane, axis=-1)

    from ic_model import angle_with_normal_deg
    theta = angle_with_normal_deg(B_coil, n_hat)
    Ic_arr, _ = ic_model.critical_current(Bmag, theta)
    Jc_vol = Ic_arr / (ta["delta_SC"] * ic_model.tape_width)
    j_over_jc = Jmag / Jc_vol

    J_unif_vec = ta["t_hat_coil"] * (I / (ta["delta_SC"] * params.w))
    dJs = (J_coil - J_unif_vec) * (ta["delta_SC"] / ta["Lambda"])
    scif_half = dB_bore_from_dJ_half(ta["coil_centroids"], dJs,
                                     ta["coil_vols"])[2] * 1e3

    A_arr = A_h.x.array
    curlA_mag = Bmag  # already |curl(A)| at coil cells
    A_mag_mean = np.mean(np.abs(A_arr))
    curlA_mean = np.mean(curlA_mag)
    gauge_ratio = A_mag_mean / max(curlA_mean, 1e-300)

    print("\n===== HALF-DOMAIN STEADY VALIDATION GATE =====")
    print(f"converged={info['converged']}  n_iters={info['n_iters']}  "
         f"wall={wall:.1f}s")
    print(f"|B| mean over coil cells: {Bmag.mean():.4f} T  "
         f"(eighth-domain reference: 4.13 T)")
    print(f"frac(|B|>8T): {(Bmag>8).mean()*100:.1f}%  "
         f"(eighth-domain reference: 11.8%)")
    print(f"J/Jc mean: {j_over_jc.mean():.4f}  "
         f"(eighth-domain reference: 0.586)")
    print(f"frac(J/Jc>1): {(j_over_jc>1).mean()*100:.1f}%  "
         f"(eighth-domain reference: 26.0%)")
    print(f"on-axis SCIF (half-domain mirror count): {scif_half:+.2f} mT  "
         f"(eighth-domain reference: ~641 mT)")
    print(f"|A| mean (coil-independent, whole mesh): {A_mag_mean:.4e}")
    print(f"|curl(A)| mean at coil cells: {curlA_mean:.4e}")
    print(f"|A|:|curl(A)| ratio: {gauge_ratio:.4e}  "
         f"(eighth-domain reference: ~1e10, per transient/induction.py)")
    print(f"mesh: {len(coil)} coil cells, "
         f"{domain.topology.index_map(domain.topology.dim).size_local} total")


if __name__ == "__main__":
    main()
