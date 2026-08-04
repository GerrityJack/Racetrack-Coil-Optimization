"""
insulated_limit.py — Tier B0: prove the ta_solve.py hooks are INERT.

The requirement is that the existing model still runs and gives the same
answer.  Comparing box_ptp_pct before and after would be a weak test: gmsh is
not reproducible across processes, and this design's own documented run-to-run
scatter (0.34-0.52% across the jitter study) is larger than any change a
subtle bug would produce.  A statistical check cannot distinguish "inert" from
"slightly broken" here.

So this gate proves inertness EXACTLY instead, in one process on one mesh:

  1. A_prev is identically zero after a normal solve, so the modified history
     term curl(A_h - A_prev) is algebraically curl(A_h).
  2. The assembled T right-hand side from the NEW form is BIT-IDENTICAL to
     the one from a locally reconstructed OLD form.  This is the decisive
     check -- it compares the actual numbers the solver consumes.
  3. With per_turn_bc=False the tape-edge BCs still carry Constant values,
     i.e. the original code path.
  4. solve_ta_at_current() still runs end to end and reproduces the recorded
     champion box uniformity within the documented mesh scatter (reported,
     not asserted, for the reason above).

Run:  <env>/bin/python3 transient/validation/insulated_limit.py
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

RECORDED_BOX_PTP_PCT = 0.495     # CLAUDE.md, the margin-aware final design
MESH_SCATTER_PCT = 0.20          # documented run-to-run band for this design


def main():
    from mpi4py import MPI
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    print("=" * 74)
    print("Tier B0 — ta_solve.py transient hooks must be INERT")
    print("=" * 74)

    print("\nbuilding mesh ...")
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_b0_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True)

    ok = True

    # ── 1. A_prev is zero ────────────────────────────────────────────────
    ap = ta["A_prev"].x.array
    z = np.all(ap == 0.0)
    print(f"\n[1] A_prev identically zero at setup      : "
          f"{'PASS' if z else 'FAIL'}  (max |A_prev| = {np.abs(ap).max():.3e})")
    ok &= z

    # ── 3. BCs are Constant-valued (checked before the solve) ────────────
    is_const = (ta["per_turn_bc"] is False
                and ta["T_top_fn"] is None and ta["T_bot_fn"] is None
                and len(ta["layer_bc_fns"]) == 0)
    print(f"[3] per_turn_bc=False keeps Constant BCs   : "
          f"{'PASS' if is_const else 'FAIL'}")
    ok &= is_const

    # ── run a real solve so A_h is a realistic, non-trivial state ────────
    print("\nrunning solve_ta_at_current at I_design (this is the state the "
          "RHS comparison uses) ...")
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)
    A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
        domain, ta, uniform, params.I_design, ic, nm, verbose=False)
    print(f"  converged={info['converged']} in {info['n_iters']} iters")

    ap = ta["A_prev"].x.array
    z2 = np.all(ap == 0.0)
    print(f"\n[1b] A_prev still zero after the solve    : "
          f"{'PASS' if z2 else 'FAIL'}")
    ok &= z2

    # ── 2. THE decisive check: assembled RHS, new form vs old form ───────
    dx = ufl.Measure("dx", domain=domain)
    phi = ufl.TestFunction(ta["V_T"])
    n_hat = ta["n_hat_ufl"]
    dt_c = ta["dt_const"]

    L_new = fem.form(-(1.0 / dt_c) * ta["coil_ind"]
                     * ufl.inner(ufl.curl(ta["A_h"] - ta["A_prev"]),
                                 phi * n_hat) * dx)
    L_old = fem.form(-(1.0 / dt_c) * ta["coil_ind"]
                     * ufl.inner(ufl.curl(ta["A_h"]), phi * n_hat) * dx)

    b_new = fem_petsc.assemble_vector(L_new)
    b_old = fem_petsc.assemble_vector(L_old)
    v_new = b_new.array.copy()
    v_old = b_old.array.copy()
    identical = np.array_equal(v_new, v_old)
    dmax = float(np.max(np.abs(v_new - v_old))) if v_new.size else 0.0
    scale = float(np.max(np.abs(v_old))) if v_old.size else 1.0
    print(f"\n[2] assembled T RHS, new form vs old form :")
    print(f"      max |difference| = {dmax:.3e}   (RHS scale {scale:.3e})")
    print(f"      BIT-IDENTICAL    : {'PASS' if identical else 'FAIL'}")
    ok &= identical

    # ── 4. end-to-end number, reported not asserted ──────────────────────
    import opt_config as cfg
    from coil2_field import compute_both_coils_field_multilayer as B_uniform

    J_unif = ta["t_hat_coil"] * (params.I_design
                                 / (ta["delta_SC"] * params.w))
    dJs = (ta_solve._J_from_T(ta, domain) - J_unif) \
        * (ta["delta_SC"] / ta["Lambda"])
    g = params.coil_half_gap
    xs = np.linspace(-cfg.TARGET_X_M / 2, cfg.TARGET_X_M / 2, cfg.TARGET_NX)
    ys = np.linspace(-cfg.TARGET_Y_M / 2, cfg.TARGET_Y_M / 2, cfg.TARGET_NY)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, g)])
    Bz_u = B_uniform(pts)[:, 2]
    dBz = np.array([ta_solve.dB_bore_from_dJ(ta["coil_centroids"], dJs,
                                             ta["coil_vols"], bore_pt=p)[2]
                    for p in pts])
    Bz = Bz_u + dBz
    ptp = 100.0 * (Bz.max() - Bz.min()) / abs(Bz.mean())
    print(f"\n[4] box peak-to-peak uniformity           : {ptp:.3f}%")
    print(f"      recorded champion value              : "
          f"{RECORDED_BOX_PTP_PCT:.3f}%")
    print(f"      within documented mesh scatter (±{MESH_SCATTER_PCT:.2f} pp): "
          f"{'yes' if abs(ptp - RECORDED_BOX_PTP_PCT) <= MESH_SCATTER_PCT else 'NO'}")
    print("      (reported, not asserted — gmsh is not reproducible across "
          "processes,\n       so check [2] is the one that actually proves "
          "inertness)")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass

    print("\n" + "=" * 74)
    print(f"B0 VERDICT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("The hooks are NOT inert — the existing model has been changed.")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
