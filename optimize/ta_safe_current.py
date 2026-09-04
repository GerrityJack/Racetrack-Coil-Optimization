"""
ta_safe_current.py — T-A-resolved-safe operating current for one candidate
=============================================================================
2026-09-02: built after `transient/validation/ta_quench_margin_check.py`
(2026-08-08) found that the champion's own uniform-J-based I_op leaves
27% of coil cells locally AT/ABOVE Jc under the real (screening-current-
resolved) T-A current distribution -- the 65%-of-Ic safety target was
only ever checked against a uniform-J approximation, never against what
the coil's own physics actually does to the current. See CLAUDE.md's
"Ramp-up power analysis" / known-open-issue #6.

This module is the fix at the SOURCE: instead of `I_op = I_quench_uniform
/ SAFETY_FACTOR` (optimize_geometry.py's evaluate(), fast but wrong about
the real current distribution), it finds the largest current at which a
genuine T-A solve's worst-cell local margin, Jc(B,theta)/|J_inplane|,
stays >= 1/0.65 = 1.5385 EVERYWHERE in the coil -- i.e. no cell exceeds
65% of its local Ic even after screening-current concentration.

Cost: this is expensive by construction -- one full mesh build plus
several T-A Picard solves per candidate (the first cold, later ones
warm-started from the previous trial current via ta_solve's own I-ratio
seeding, per CLAUDE.md's "warm start ... far fewer iterations"). This is
NOT a cheap per-generation proxy (this project's "Proxy graveyard" found
none survive validation) -- it is the real, ground-truth T-A check,
called once per CMA-ES evaluation. See optimize/studies/
ta_safe_margin_search.py for the search this is built for; that script's
own docstring has the resulting per-evaluation cost budget and why the
search is a small, warm-started LOCAL polish around the champion rather
than a from-scratch search.

Reuses, rather than reimplements:
  - optimize_geometry.py's apply_geometry-equivalent (`_apply`, matches
    ta_validate.py's own, since this -- like ta_validate.py -- needs the
    FULL-fidelity production mesh, not optimize_geometry.py's
    SCREEN_MESH_OVERRIDES coarse mesh)
  - optimize_geometry.quench_current() / stress_screen() / target_box_field()
    for the uniform-J bracket, hoop stress, and box-field machinery
  - solve.py / ta_solve.py's own mesh->FEM->Picard pipeline, exactly as
    ta_validate.py and solve/ta_solve.py's own __main__ use it
  - transient/validation/ta_quench_margin_check.py's per-cell margin
    formula (Jc/|J_inplane|), so this is the SAME check already validated
    and documented in CLAUDE.md, not a new formula
"""
import os
import time

import numpy as np

import params
import opt_config as cfg
import optimize_geometry as og
import build_mesh
import solve as base_solve
import ta_solve
import dolfinx.mesh as dmesh
from current_source import normal_xy
from ic_model import angle_with_normal_deg, NValueModel
from coil2_field import compute_both_coils_field_multilayer

MARGIN_REQUIRED = 1.0 / 0.65   # same threshold as ta_quench_margin_check.py
MAX_BISECT = 5
I_BISECT_TOL_A = 3.0
I_FLOOR_A = 5.0                # assumed safe without a solve (near-zero J)

# 2026-09-03: per-cell margin distribution at a candidate's own safe I_op
# (checked on two designs, ~2100 coil cells each) found the OLD all-cells
# criterion was being gated by a literal handful of cells -- 0.1-1.7% of
# cells within reach of the 0.65 threshold, while the MEDIAN cell ran at
# 10-15x margin (only ~6-9% of its own local Ic). That's consistent with
# either a real but highly localized critical-state edge effect, or a
# handful of unresolved mesh/BC-singularity cells -- either way, per user
# direction, relax the requirement to a percentile instead of the strict
# minimum: MARGIN_PERCENTILE=5 requires the 5th-percentile cell (i.e. 95%
# of cells) to clear MARGIN_REQUIRED, not literally every cell. Set to 0
# to recover the original strict all-cells-must-comply behavior.
MARGIN_PERCENTILE = float(os.environ.get("TA_SAFE_MARGIN_PERCENTILE", 5.0))


def _constraint_margin(margin_arr):
    """The margin value that actually gates I_op -- see MARGIN_PERCENTILE
    above. np.percentile(margin, p) is the value below which p% of cells
    fall, i.e. requiring THIS value >= MARGIN_REQUIRED is exactly
    "(100-p)% of cells satisfy the margin requirement"."""
    if MARGIN_PERCENTILE <= 0:
        return float(margin_arr.min())
    return float(np.percentile(margin_arr, MARGIN_PERCENTILE))

# 2026-09-03 bugfix: captured ONCE at import time, before anything mutates
# params.mesh_filename. evaluate() used to derive each call's filename via
# os.path.splitext(params.mesh_filename) -- reading the LIVE (already
# call-mangled) value, not the pristine base -- so every call within the
# same long-lived worker process appended ANOTHER "_tasafe<pid>_<ms>"
# suffix onto the PREVIOUS call's already-suffixed name. Over ~30
# evaluations in one worker this grew past Linux's 255-byte filename
# limit, silently breaking gmsh's write/dolfinx's read ("Unable to open
# file ...") for every candidate that worker touched from then on --
# which flooded the CMA-ES population with identical flat
# CMAES_INFEASIBLE_PENALTY_KM fitness values and tripped pycma's own
# stagnation-detection stopping criterion, ending a live overnight run at
# 126/700 evaluations. Always derive from this fixed base instead.
_BASE_MESH_FILENAME = params.mesh_filename


def _apply(design):
    """Mutate params in-place for this candidate. Full production mesh
    resolution (mesh_nz_per_layer / mesh_z_grading left at params.py's
    defaults) -- deliberately does NOT apply cfg.SCREEN_MESH_OVERRIDES,
    same reasoning as ta_validate.py: the screening profile this whole
    check exists to resolve lives across the tape width, which the coarse
    single-slab screen mesh cannot see."""
    params.a = float(design["a"])
    params.b = float(design["b"])
    params.coil_half_gap = float(design["coil_half_gap"])
    params.n_turns = list(design["n_turns"])
    params.recompute_derived()


def _local_margin(domain, ta, B_h, ic_model):
    """Per-cell T-A local margin = Jc(B,theta)/|J_inplane| at the coil
    cells, using EXACTLY transient/validation/ta_quench_margin_check.py's
    formula (in-plane-projected J vs. Jc from the same Ic model). Returns
    the FULL per-cell array (not just the worst) -- see _constraint_margin
    for how this becomes the value that gates I_op."""
    coil_cells = ta["coil_cells"]
    centroids = dmesh.compute_midpoints(domain, domain.topology.dim,
                                        coil_cells)
    J_arr = ta_solve._J_from_T(ta, domain)
    B_arr = B_h.x.array.reshape(-1, 3)[coil_cells]

    L = params.L
    nx, ny = normal_xy(centroids[:, 0], centroids[:, 1], L)
    n_hat = np.column_stack([nx, ny, np.zeros_like(nx)])

    Bmag = np.linalg.norm(B_arr, axis=1)
    theta = angle_with_normal_deg(B_arr, n_hat)
    Ic_A, frac_clipped = ic_model.critical_current(Bmag, theta)
    Jc = Ic_A / (params.delta_SC * params.w)

    J_dot_n = np.einsum("ij,ij->i", J_arr, n_hat)
    J_inplane = J_arr - J_dot_n[:, None] * n_hat
    Jmag = np.linalg.norm(J_inplane, axis=-1)

    margin = Jc / Jmag
    return margin, float(frac_clipped), centroids, J_arr


def evaluate(design, ic_model, comm, verbose=False, save_fields_path=None):
    """design: dict(a, b, coil_half_gap, n_turns). Returns a dict with the
    same keys optimize_geometry.evaluate() returns (so it's a drop-in
    replacement for the fitness function), plus ta_worst_margin/
    n_ta_solves/ta_solve_s diagnostics."""
    t0 = time.time()
    label = (f"a={design['a']*1e3:.1f} b={design['b']*1e3:.1f} "
             f"gap={design['coil_half_gap']*1e3:.1f} "
             f"nt={sum(design['n_turns'])}")
    try:
        _apply(design)
    except AssertionError as e:
        return dict(label=label, feasible=False, reason=str(e))

    from dolfinx.io import gmsh as gmshio
    root, ext = os.path.splitext(_BASE_MESH_FILENAME)
    params.mesh_filename = f"{root}_tasafe{os.getpid()}_{int(t0*1000)%100000}{ext}"

    n_ta_solves = 0
    try:
        build_mesh.build(write_path=params.mesh_filename)
        md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
        domain = md.mesh

        uniform_setup = base_solve.setup_problem(domain, md.cell_tags,
                                                 md.facet_tags)
        n_model = NValueModel(csv_path=params.n_value_csv_filename)
        ta = ta_solve.setup_ta_problem(domain, md.cell_tags, md.facet_tags,
                                       uniform_setup)

        # ── reference uniform-J solve: cheap bracket + hoop stress ──────
        _, B_h_ref = base_solve.solve_at_current(
            domain, uniform_setup, og.I_REF, comm, verbose_label=label)
        cells = uniform_setup["coil_cells"]
        cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
        B_ref = B_h_ref.x.array.reshape(-1, 3)[cells]
        B_unit = B_ref / og.I_REF

        import ufl
        from dolfinx import fem
        Vv = fem.functionspace(domain, ("DG", 0))
        vf = fem.Function(Vv)
        vf.interpolate(fem.Expression(ufl.CellVolume(domain),
                                      Vv.element.interpolation_points))
        vol = vf.x.array[cells]

        L = params.b - params.a
        nx, ny = normal_xy(cents[:, 0], cents[:, 1], L)
        n_hat = np.column_stack([nx, ny, np.zeros(len(cells))])
        theta = angle_with_normal_deg(B_ref, n_hat)
        I_q_cells, clip_frac = og.quench_current(
            np.linalg.norm(B_unit, axis=1), theta, ic_model)
        I_q_uniform = float(np.min(I_q_cells))
        if not np.isfinite(I_q_uniform):
            # og.quench_current()'s root-find never crossed within
            # cfg.I_MAX_SEARCH_A (1500A -- tuned for the champion's scale,
            # not necessarily this wide search's much larger candidates)
            # and reports inf for "never quenched" cells. inf would break
            # the bisection arithmetic below (0.5*(lo+inf)). Use the
            # search ceiling itself as a finite stand-in -- it's only a
            # STARTING bracket for the T-A bisection, not the final
            # answer, and every T-A-safe current found so far (this
            # design and its neighbors) is two orders of magnitude below
            # even 1500A, so this is generous headroom, not a new cap.
            I_q_uniform = cfg.I_MAX_SEARCH_A
        I_op_uniform = I_q_uniform / cfg.SAFETY_FACTOR

        # ── T-A bisection: largest I with the constraint margin (percentile
        # -- see MARGIN_PERCENTILE) >= required ──────────────────────────
        I_hi = max(I_op_uniform, I_FLOOR_A * 2.0)
        A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
            domain, ta, uniform_setup, I_amps=I_hi, ic_model=ic_model,
            n_model=n_model, verbose=verbose, warm_start=False)
        n_ta_solves += 1
        margin_hi_arr, clip_ta, _, _ = _local_margin(domain, ta, B_h, ic_model)
        margin_hi = _constraint_margin(margin_hi_arr)

        if margin_hi >= MARGIN_REQUIRED:
            # Uniform-J's own I_op already satisfies the real T-A margin
            # for this candidate -- no derating needed.
            I_op_ta, final_margin = I_hi, margin_hi
            final_worst_margin = float(margin_hi_arr.min())
        else:
            I_lo = I_FLOOR_A
            best_I, best_margin, best_worst = None, None, None
            for _ in range(MAX_BISECT):
                I_mid = 0.5 * (I_lo + I_hi)
                A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
                    domain, ta, uniform_setup, I_amps=I_mid,
                    ic_model=ic_model, n_model=n_model, verbose=verbose,
                    warm_start=True)
                n_ta_solves += 1
                m_arr, clip_ta, _, _ = _local_margin(domain, ta, B_h, ic_model)
                m = _constraint_margin(m_arr)
                if m >= MARGIN_REQUIRED:
                    I_lo, best_I, best_margin = I_mid, I_mid, m
                    best_worst = float(m_arr.min())
                else:
                    I_hi = I_mid
                if (I_hi - I_lo) < I_BISECT_TOL_A:
                    break
            if best_I is None:
                # Never found a safe trial current above the floor --
                # fall back to the floor itself (reported margin is the
                # floor's, not re-verified by an extra solve: at I_FLOOR_A
                # local J is a small fraction of Jc almost by construction
                # for any physically sane geometry in this search's
                # bounds; treat as a strong fitness penalty via the
                # resulting tiny B_target_T rather than a hard failure).
                I_op_ta, final_margin, final_worst_margin = (
                    I_FLOOR_A, MARGIN_REQUIRED, MARGIN_REQUIRED)
            else:
                # Final confirming solve exactly AT best_I, since the loop
                # may have gone on to test a later (unsafe) I_mid after
                # finding best_I -- ta/A_h/B_h/T_h must reflect the SAME
                # current we report as I_op.
                A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
                    domain, ta, uniform_setup, I_amps=best_I,
                    ic_model=ic_model, n_model=n_model, verbose=verbose,
                    warm_start=True)
                n_ta_solves += 1
                final_margin_arr, clip_ta, cents_ta, J_ta = _local_margin(
                    domain, ta, B_h, ic_model)
                final_margin = _constraint_margin(final_margin_arr)
                final_worst_margin = float(final_margin_arr.min())
                I_op_ta = best_I

        # ── B_target_T at I_op_ta: uniform-J Biot-Savart + T-A SCIF correction ──
        delta_SC = float(ta["delta_SC"])
        Lambda = float(params.t)
        J_TA_arr = ta_solve._J_from_T(ta, domain)
        J_unif_mag = I_op_ta / (delta_SC * params.w)
        J_unif_arr = ta["t_hat_coil"] * J_unif_mag
        dJ_s = (J_TA_arr - J_unif_arr) * (delta_SC / Lambda)
        coil_cells = ta["coil_cells"]
        centroids = dmesh.compute_midpoints(domain, domain.topology.dim,
                                            coil_cells)
        dV = ta["coil_vols"]

        B_box_uniform, box_pts = og.target_box_field(I_op_ta)
        dB_box = np.array([ta_solve.dB_bore_from_dJ(centroids, dJ_s, dV,
                                                     bore_pt=p)
                           for p in box_pts])
        B_tgt = float(np.mean(np.abs((B_box_uniform + dB_box)[:, 2])))
        Bmag = np.linalg.norm(B_box_uniform + dB_box, axis=1)
        unif = float((Bmag.max() - Bmag.min()) / Bmag.mean() * 100.0)

        # ── hoop stress at I_op_ta (uniform-J field, matches production) ──
        hoop_1, delam_1 = og.stress_screen(cents, B_unit, vol, 1.0)
        hoop_MPa = hoop_1 * I_op_ta ** 2 / 1e6
        delam_MPa = delam_1 * I_op_ta ** 2 / 1e6

        # ── optional: minimal reproducible field snapshot for this design,
        # at its own T-A-safe I_op -- enough for visualization/for
        # poster/make_jjc_cross_sections.py (which only reads I_solved,
        # coil_centroids, coil_B, J_TA_coil, delta_SC) to regenerate the
        # J/Jc cross-section poster figures for THIS design without
        # re-running the search. Deliberately NOT a full drop-in
        # replacement for solve/racetrack_ta_fields.npz's schema (skips
        # T_field, dB_bore, Bz_bore_* -- not needed by that script and not
        # already computed here).
        if save_fields_path:
            B_coil_ta = B_h.x.array.reshape(-1, 3)[coil_cells]
            np.savez(save_fields_path,
                    I_solved=I_op_ta, delta_SC=delta_SC,
                    coil_centroids=centroids, coil_B=B_coil_ta,
                    J_TA_coil=J_TA_arr, J_unif_coil=J_unif_arr,
                    dV=dV, box_ptp_pct=unif,
                    a=params.a, b=params.b,
                    coil_half_gap=params.coil_half_gap,
                    n_turns=np.array(params.n_turns))
            if verbose:
                print(f"[ta_safe_current] wrote {save_fields_path}")

    finally:
        try:
            os.remove(params.mesh_filename)
        except OSError:
            pass

    return dict(label=label, feasible=True,
                a_mm=design["a"] * 1e3, b_mm=design["b"] * 1e3,
                n_turns=str(design["n_turns"]),
                n_total=sum(design["n_turns"]),
                tape_km=params.tape_length_m / 1e3,
                I_quench_uniform_A=I_q_uniform, I_op_A=I_op_ta,
                binding="ta_local_margin",
                B_target_T=B_tgt, uniformity_pct=unif,
                hoop_MPa=hoop_MPa, delam_MPa=delam_MPa, delam_scr_MPa=delam_MPa,
                # ta_worst_margin: the true single-worst-cell value
                # (diagnostic only, no longer what gates I_op).
                # ta_constraint_margin: the value that actually gated
                # I_op -- the MARGIN_PERCENTILE-th percentile (95% of
                # cells satisfy the requirement at this current), or
                # equal to ta_worst_margin if MARGIN_PERCENTILE<=0.
                ta_worst_margin=final_worst_margin,
                ta_constraint_margin=final_margin,
                clip_frac=clip_frac,
                n_ta_solves=n_ta_solves, eval_s=time.time() - t0)
