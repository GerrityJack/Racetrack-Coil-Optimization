"""
evaluate.py — single-configuration evaluation for external optimizers
=====================================================================
THE entry point for optimization work.  One call = one design evaluated.

    OBJECTIVE   maximize  B_target  (mean |Bz| over the target box at the
                quench-limited operating current I_op = I_quench / SF)
    CONSTRAINT  peak-to-peak field uniformity over the target box ≤ 1 %
                (screening-current correction included)
    VARIABLES   a        cap radius [m]                    (continuous)
                b        centre → cap-centre length [m]    (continuous, b > a)
                n_turns  turns per pancake, top → bottom   (list of ints ≥ 1;
                         list length = number of pancakes, may also vary)

Geometric feasibility (inner radius > 0, i.e. a_out − n_i·t > 0) is
checked internally; infeasible designs return feasible=False with
objective None — treat as a constraint violation, not an error.

Stress limits are intentionally EXCLUDED here per current project
direction (quench + uniformity only).  The batch screen
(optimize_geometry.py) still carries them if needed later.

Command line:
    conda run -n fenicsx-env python3 optimize/evaluate.py \
        --a 0.050 --b 0.080 --n-turns 500,500,500,400,400,250,100 [--json]

Python:
    from evaluate import evaluate_configuration
    r = evaluate_configuration(0.050, 0.080, [500,500,500,400,400,250,100])
    r["objective_B_target_T"], r["uniformity_pct"], r["feasible"], ...

Runtime: ~8–12 s per evaluation (one FEM solve — the problem is exactly
linear in current, so quench is a root-find and everything else scales).
Evaluations are independent: parallelise across processes freely.
"""
import os, sys, json, time, argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import params
import opt_config as cfg


def evaluate_configuration(a, b, n_turns,
                           safety_factor=None,
                           uniformity_max_pct=None,
                           verbose=False):
    """
    Evaluate one (a, b, n_turns) design.  Returns a plain dict of floats /
    bools (JSON-serialisable).  Keys:

      feasible                geometry valid and evaluation completed
      objective_B_target_T    mean |Bz| over the target box at I_op
                              (None if infeasible)
      pass_constraints        uniformity constraint satisfied
      I_quench_A              min quench current over the winding
      I_op_A                  I_quench / safety_factor
      uniformity_pct          peak-to-peak |B| over the box [%], SCIF-corrected
      uniformity_geom_pct     same, geometric field only
      scif_mT                 screening-current field at box centre [mT]
      peak_conductor_B_T      max |B| on the conductor at I_op
      clip_fraction           fraction of Ic evaluations beyond the 8 T
                              dataset at I_op (data-quality warning: quench
                              is a FLOOR estimate when this is large)
      tape_km, n_turns_total, eval_seconds
    """
    from mpi4py import MPI
    import dolfinx.mesh as dmesh
    from dolfinx.io import gmsh as gmshio
    import solve as base_solve
    import build_mesh
    from ic_model import IcModel, angle_with_normal_deg
    from current_source import normal_xy
    import optimize_geometry as og

    # NOTE: uses EVALUATE_SAFETY_FACTOR, not SAFETY_FACTOR -- the latter was
    # repurposed for the internal CMA-ES search (opt_config.py); this frozen
    # constant keeps evaluate.py's external-team contract unchanged.
    sf   = safety_factor    if safety_factor    is not None \
        else cfg.EVALUATE_SAFETY_FACTOR
    umax = uniformity_max_pct if uniformity_max_pct is not None \
        else cfg.UNIFORMITY_MAX_PCT

    t0 = time.time()
    try:
        og.apply_geometry(dict(a=float(a), b=float(b),
                               n_turns=[int(n) for n in n_turns]))
    except AssertionError as e:
        return dict(feasible=False, objective_B_target_T=None,
                    pass_constraints=False, reason=str(e),
                    eval_seconds=time.time() - t0)

    comm = MPI.COMM_WORLD
    ic_model = IcModel()

    build_mesh.build(write_path=params.mesh_filename)
    md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    domain = md.mesh
    us = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
    _, B_h = base_solve.solve_at_current(
        domain, us, og.I_REF, comm,
        verbose_label="eval" if verbose else None)

    cells = us["coil_cells"]
    cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
    B_unit = B_h.x.array.reshape(-1, 3)[cells] / og.I_REF

    import ufl
    from dolfinx import fem
    Vv = fem.functionspace(domain, ("DG", 0))
    vf = fem.Function(Vv)
    vf.interpolate(fem.Expression(ufl.CellVolume(domain),
                                  Vv.element.interpolation_points))
    vol = vf.x.array[cells]

    # quench → operating point (quench + safety factor ONLY; no stress)
    L = params.b - params.a
    nx, ny = normal_xy(cents[:, 0], cents[:, 1], L)
    n_hat = np.column_stack([nx, ny, np.zeros(len(cells))])
    theta = angle_with_normal_deg(B_unit, n_hat)     # angle is I-independent
    B_umag = np.linalg.norm(B_unit, axis=1)
    I_q_cells, _ = og.quench_current(B_umag, theta, ic_model)
    I_q  = float(np.min(I_q_cells))
    I_op = I_q / sf

    # target box: geometric field + Bean screening correction.
    # Temporarily override the box size with evaluate.py's frozen
    # EVALUATE_TARGET_X_M/Y_M -- cfg.TARGET_X_M/Y_M were repurposed for the
    # internal CMA-ES search and must not leak into this entry point.
    _old_tx, _old_ty = cfg.TARGET_X_M, cfg.TARGET_Y_M
    cfg.TARGET_X_M, cfg.TARGET_Y_M = cfg.EVALUATE_TARGET_X_M, cfg.EVALUATE_TARGET_Y_M
    try:
        B_box, box_pts = og.target_box_field(I_op)
    finally:
        cfg.TARGET_X_M, cfg.TARGET_Y_M = _old_tx, _old_ty
    M_vec, _ = og.bean_moments(B_unit * I_op, n_hat, theta, ic_model, I_op)
    dB_box = og.dipole_field_mirrored(box_pts, cents, M_vec, vol)
    g = float(params.coil_half_gap)
    scif_mT = float(og.dipole_field_mirrored(
        np.array([[0.0, 0.0, g]]), cents, M_vec, vol)[0, 2] * 1e3)

    Bmag0 = np.linalg.norm(B_box, axis=1)
    Bmag  = np.linalg.norm(B_box + dB_box, axis=1)
    B_tgt = float(np.mean(np.abs((B_box + dB_box)[:, 2])))
    unif0 = float((Bmag0.max() - Bmag0.min()) / Bmag0.mean() * 100.0)
    unif  = float((Bmag.max() - Bmag.min()) / Bmag.mean() * 100.0)

    _, extra = ic_model.critical_current(B_umag * I_op, theta)
    try:
        clip = float(np.mean(extra))
    except Exception:
        clip = float("nan")

    return dict(
        feasible=True,
        objective_B_target_T=B_tgt,
        pass_constraints=bool(unif <= umax),
        I_quench_A=I_q, I_op_A=I_op,
        uniformity_pct=unif, uniformity_geom_pct=unif0,
        uniformity_limit_pct=umax,
        scif_mT=scif_mT,
        peak_conductor_B_T=float(np.max(B_umag) * I_op),
        clip_fraction=clip,
        tape_km=params.tape_length_m / 1e3,
        n_turns_total=int(params.n_turns_total),
        a_m=float(a), b_m=float(b), n_turns=[int(n) for n in n_turns],
        safety_factor=sf,
        eval_seconds=time.time() - t0)


def _summary(r):
    if not r["feasible"]:
        return (f"INFEASIBLE geometry: {r.get('reason', '')}\n"
                f"  (objective = None — treat as constraint violation)")
    warn = ""
    if r["clip_fraction"] > cfg.MAX_CLIP_FRACTION:
        warn = (f"\n  WARNING: {r['clip_fraction']*100:.0f}% of Ic values at "
                f"I_op lie beyond the 8 T dataset — quench (and therefore "
                f"the objective) is a floor estimate here")
    return (
        f"CONFIGURATION  a = {r['a_m']*1e3:.1f} mm   b = {r['b_m']*1e3:.1f} mm"
        f"   n_turns = {r['n_turns']}  ({r['n_turns_total']} turns, "
        f"{r['tape_km']:.2f} km tape)\n"
        f"  OBJECTIVE  B_target = {r['objective_B_target_T']:.3f} T   "
        f"(mean |Bz| over the target box at I_op)\n"
        f"  I_quench   = {r['I_quench_A']:.1f} A/turn  →  I_op = "
        f"{r['I_op_A']:.1f} A  (safety factor {r['safety_factor']})\n"
        f"  uniformity = {r['uniformity_pct']:.3f} %  "
        f"(limit {r['uniformity_limit_pct']:.1f} %)  →  "
        f"{'PASS' if r['pass_constraints'] else 'FAIL'}\n"
        f"  SCIF       = {r['scif_mT']:+.0f} mT at box centre "
        f"(screening correction included in uniformity)\n"
        f"  peak conductor field = {r['peak_conductor_B_T']:.1f} T"
        f"{warn}\n"
        f"  [{r['eval_seconds']:.0f} s]")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one racetrack-pair design "
                    "(objective: target field; constraint: uniformity).")
    ap.add_argument("--a", type=float, required=True,
                    help="cap radius [m], e.g. 0.050")
    ap.add_argument("--b", type=float, required=True,
                    help="centre-to-cap-centre length [m], must exceed a")
    ap.add_argument("--n-turns", type=str, required=True,
                    help="comma-separated turns per pancake, top to bottom, "
                         "e.g. 500,500,500,400,400,250,100")
    ap.add_argument("--safety-factor", type=float, default=None)
    ap.add_argument("--json", action="store_true",
                    help="print the full result dict as JSON on stdout")
    args = ap.parse_args()

    n_turns = [int(s) for s in args.n_turns.split(",")]
    r = evaluate_configuration(args.a, args.b, n_turns,
                               safety_factor=args.safety_factor)
    print(_summary(r))
    if args.json:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
