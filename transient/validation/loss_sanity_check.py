"""
loss_sanity_check.py — sanity-check transient/loss.py's hysteretic_power()
against the FULLY CONVERGED steady-state T-A solve.

Why this test: the smoke-tested transient charge reported P_sc = 1182 W
(~3.5 W/m of tape), implausible for a 0.33 A/s ramp and far beyond any 20 K
cryocooler budget. Two candidate explanations:

  (a) the hysteretic_power() FORMULA (symmetry factors, the rho_fn*J^2*dV
      identity) is wrong;
  (b) the formula is fine, but the TRANSIENT STATE it was evaluated on was
      bad -- steps hit their iteration cap rather than converging, and the
      warmup/relaxation machinery could leave J transiently far from any
      physical state.

This isolates (a) by applying the identical formula to solve_ta_at_current()
at I_design -- the same call path used throughout this project's whole SCIF
history, known to converge cleanly (k~80, tol via ta_scif_stall_mT) and to
give physically sensible SCIF numbers. If P_sc is still huge here, the
formula itself is wrong. If it is small/sane here, the bug is transient-side
(b) -- most likely the steps that hit their Picard cap.

Also cross-checks the order of magnitude against the critical-state floor
power density E_c*Jc_typical (roughly the minimum resistive dissipation any
current-carrying superconducting cell has, since rho_fn is floored at
E_c/Jc), integrated over the coil volume, as a completely independent
ballpark.

Run:  <env>/bin/python3 transient/validation/loss_sanity_check.py
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import loss as loss_mod
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    print("=" * 76)
    print("Sanity check: hysteretic_power() on the CONVERGED steady solve")
    print("=" * 76)

    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_lsc{os.getpid()}{ext}"
    print("\nbuilding mesh ...")
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    print(f"\nsolving at I_design = {params.I_design:.2f} A "
          f"(the project's own well-converged reference point) ...")
    A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
        domain, ta, uniform, params.I_design, ic, nm, verbose=True,
        warm_start=False)
    print(f"\nconverged={info['converged']}  n_iters={info['n_iters']}  "
          f"rel_err={info['rel_err']:.2e}  scif_mT={info['scif_mT']:.2f}")

    J_coil = ta_solve._J_from_T(ta, domain)
    P_sc = loss_mod.hysteretic_power(ta, J_coil)

    # ── independent ballpark: critical-state floor power density ──────────
    B_coil = B_h.x.array.reshape(-1, 3)[ta["coil_cells"]]
    Bmag = np.linalg.norm(B_coil, axis=1)
    n_hat = ta["n_hat_coil"]
    from ic_model import angle_with_normal_deg
    theta = angle_with_normal_deg(B_coil, n_hat)
    Ic_arr, clip_frac = ic.critical_current(Bmag, theta)
    Jc_vol = Ic_arr / (ta["delta_SC"] * ic.tape_width)
    E_c = 1.0e-4
    P_floor_density = E_c * Jc_vol            # W/m^3 at j/jc = 1 (the floor)
    P_floor_total = loss_mod.CELL_SYMMETRY * float(
        np.sum(P_floor_density * ta["coil_vols"]))

    j_norm = np.linalg.norm(J_coil, axis=1) / Jc_vol
    print(f"\nlocal j/jc over the coil: "
          f"min={j_norm.min():.3f} mean={j_norm.mean():.3f} "
          f"median={np.median(j_norm):.3f} max={j_norm.max():.3f}")
    print(f"fraction of cells with j/jc > 1 (over-critical): "
          f"{np.mean(j_norm > 1.0)*100:.1f}%")
    print(f"Ic clip fraction (evaluated above 8T): {np.mean(clip_frac)*100:.1f}%")

    print(f"\nP_sc (hysteretic_power(), converged steady state) = "
          f"{P_sc:.4g} W  ({P_sc/params.tape_length_m:.4g} W/m of tape)")
    print(f"P_floor (E_c*Jc at j/jc=1, same cells/volumes)      = "
          f"{P_floor_total:.4g} W  "
          f"({P_floor_total/params.tape_length_m:.4g} W/m)")
    print(f"ratio P_sc / P_floor = {P_sc/max(P_floor_total,1e-30):.3f}  "
          f"(near 1 is physically sane at j/jc~1; >>1 tracks the "
          f"over-critical fraction and its (n-1) power-law amplification)")

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    plausible = P_sc < 100.0 * P_floor_total and np.isfinite(P_sc)
    print(f"P_sc within 100x of the critical-state floor estimate: "
          f"{'PASS' if plausible else 'FAIL'}")
    if plausible:
        print("-> the FORMULA is sane on a converged state. The 1182 W")
        print("   transient smoke-test number is therefore most likely a")
        print("   TRANSIENT-SIDE artifact (steps hit their Picard cap")
        print("   rather than converging), not a bug in loss.py itself.")
    else:
        print("-> the formula itself produces an implausible number even")
        print("   on a trusted, converged state. Re-derive it before")
        print("   trusting ANY hysteretic loss number from this project.")
    print("=" * 76)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
    return 0 if plausible else 1


if __name__ == "__main__":
    sys.exit(main())
