"""
run_charge.py — T-A + NI transient charge of the champion.

What this adds over the Phase A circuit model (circuit/):
  * the hysteretic (screening-current) loss, which a lumped per-turn circuit
    cannot represent at all;
  * the actual current distribution across the tape WIDTH, not just radially;
  * a bore field taken straight from the solved current density rather than
    from a filament sum.

The bore field is evaluated by Biot-Savart over the solved J_s using
ta_solve.dB_bore_from_dJ() with the FULL current density (not the screening
difference), so it includes the radial redistribution and the screening
currents together.  Its DC value is a useful self-check: it should land near
the -11.5 T the independent filament model gives.

Run:  <env>/bin/python3 transient/run_charge.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tparams as tp        # noqa: E402


def main(rho_ct=None, t_ramp=None, tag="charge"):
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import ta_transient as tt
    import loss as loss_mod
    from ic_model import IcModel, NValueModel

    rho_ct = tp.RHO_CT_UOHM_CM2 if rho_ct is None else rho_ct
    t_ramp = tp.RAMP_S if t_ramp is None else t_ramp
    comm = MPI.COMM_WORLD

    print("=" * 78)
    print("T-A transient with no-insulation coupling — charge")
    print("=" * 78)
    print(f"rho_c = {rho_ct:.0f} uOhm.cm^2   ramp = {t_ramp:.0f} s   "
          f"I_op = {params.I_design:.2f} A")
    print(f"radial bins = {tp.N_RADIAL_BINS}/layer   "
          f"E_p factor mode = {tp.EP_FACTOR_MODE}\n")

    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_tr{os.getpid()}{ext}"
    print("building mesh ...")
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta, circuit, bins = tt.build(domain, cell_tags, facet_tags, uniform,
                                 rho_ct_uohm_cm2=rho_ct)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)
    print(f"  coil cells = {len(ta['coil_cells'])}, "
          f"A-dofs = {ta['V_A'].dofmap.index_map.size_global}")

    sched = tt.ramp_schedule(params.I_design, t_ramp=t_ramp)
    t0 = time.time()
    hist = tt.march(ta, circuit, domain, uniform, ic, nm, sched)
    wall = time.time() - t0

    # ── post: bore field and losses per step ────────────────────────────
    rows = []
    scale = ta["delta_SC"] / ta["Lambda"]
    for h in hist:
        rows.append(dict(t_s=h["t"], I_A=h["I"], dt_s=h["dt"],
                         n_iters=h["n_iters"], converged=h["converged"],
                         scif_mT=h["scif_mT"],
                         I_z_min=h["I_z_min"], I_z_max=h["I_z_max"],
                         I_r_mean=h["I_r_mean"],
                         I_r_frac_max=h["I_r_frac_max"],
                         n_out_of_range=h["n_out_of_range"]))

    # final-state quantities (the solved J is only available at the end)
    J_final = ta_solve._J_from_T(ta, domain)
    Bz_final = ta_solve.dB_bore_from_dJ(ta["coil_centroids"],
                                        J_final * scale,
                                        ta["coil_vols"])[2]
    P_sc = loss_mod.hysteretic_power(ta, J_final)
    P_c = loss_mod.contact_power(circuit)

    out_csv = os.path.join(tp.RUNS_DIR, f"{tag}_steps.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(
        os.path.join(tp.RUNS_DIR, f"{tag}_state.npz"),
        t=np.array([h["t"] for h in hist]),
        I=np.array([h["I"] for h in hist]),
        I_z=np.array([h["I_z"] for h in hist]),
        I_r=np.array([h["I_r"] for h in hist]),
        scif_mT=np.array([h["scif_mT"] for h in hist]),
        bin_turns=bins.flat(bins.turns), bin_r=bins.flat(bins.r_center),
        R_ct=circuit.R_ct, Bz_final=Bz_final, P_sc_final=P_sc,
        P_contact_final=P_c, rho_ct_uohm_cm2=rho_ct, t_ramp=t_ramp)

    print("\n" + "=" * 78)
    print(f"steps = {len(hist)}   wall = {wall/60:.1f} min   "
          f"non-converged steps = {sum(1 for h in hist if not h['converged'])}")
    print(f"final bore Bz (from the solved J) = {Bz_final:+.4f} T")
    print(f"final hysteretic power  P_sc      = {P_sc:.4g} W")
    print(f"final contact power     P_c       = {P_c:.4g} W")
    print(f"max |I_r| / I over the run        = "
          f"{max(h['I_r_frac_max'] for h in hist)*100:.2f}%")
    bad = sum(h["n_out_of_range"] for h in hist)
    print(f"bins with I_z outside [0, 1.5 I]  = {bad}"
          f"{'  <- CHECK' if bad else ''}")
    print(f"wrote {out_csv}")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
