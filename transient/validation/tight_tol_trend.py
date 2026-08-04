"""
tight_tol_trend.py — root-cause test for thin_layer_trend.py's finding that
the Newton-hybrid's TRUE fixed point (Aitken-extrapolated from ~270
iterations: approx -24 mT) is wildly different from Picard's validated
ground truth (641.26 mT) at I=196A.

Hypothesis: DEFAULT_SNES_OPTIONS deliberately LOOSENED the per-layer inner
Newton tolerance (snes_rtol 1e-8->1e-6) to avoid spurious line-search
failures once the outer loop is close to converged. Fine for a handful of
iterations; over HUNDREDS of outer iterations, a small SYSTEMATIC (not
random-cancelling) per-iteration under-convergence could compound into
exactly the kind of slow, one-directional drift observed. Never tested at
this iteration count before -- the original validation (I=32.667A, <1%
agreement) only ran a few outer iterations before reaching the (correct,
in that case) answer.

Test: rerun the SAME chunked-march diagnostic with snes_rtol/atol/stol
tightened back to (and past) their original pre-loosening values, and see
whether the drift disappears (SCIF plateaus near Picard's 641 mT instead
of decaying toward a wildly different asymptote).
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


def per_layer_jjc(ta, domain, ic_model, n_model):
    import ta_solve
    from ic_model import angle_with_normal_deg

    coil = ta["coil_cells"]
    B_h = ta["B_fn"]
    B_h.interpolate(ta["curl_expr"])
    B_coil = B_h.x.array.reshape(-1, 3)[coil]
    Bmag = np.linalg.norm(B_coil, axis=1)
    J_coil = ta_solve._J_from_T(ta, domain)
    Jmag = np.linalg.norm(J_coil, axis=1)
    n_hat_coil = ta["n_hat_coil"]
    theta = angle_with_normal_deg(B_coil, n_hat_coil)
    Ic_arr, _ = ic_model.critical_current(Bmag, theta)
    Jc_vol = Ic_arr / (ta["delta_SC"] * ic_model.tape_width)
    jjc = Jmag / Jc_vol
    return [float(jjc[idx].mean()) for idx in ta["layer_cell_idx"]]


def main():
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import newton_ta
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_tighttol_{os.getpid()}{ext}"
    print("building mesh ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    # TIGHT tolerances -- past the ORIGINAL pre-loosening values, to make
    # this a decisive test (if the drift is due to accumulated
    # under-convergence, tightening should visibly slow/stop it).
    tight_opts = dict(snes_rtol=1e-12, snes_atol=1e-12, snes_stol=1e-13,
                      snes_max_it=100)
    newton_ta.build_layer_newton_problems(ta, snes_options=tight_opts,
                                          verbose=False)
    print(f"using tightened SNES options: {tight_opts}", flush=True)

    print("\n=== chunk 0 (bootstrap + first Newton pass, max_outer=30) ===", flush=True)
    info = newton_ta.step(ta, domain, ic, nm, I, params.ramp_duration, uniform,
                          max_outer=30, min_outer=3, stall_tol=1e-9,
                          first=True, bootstrap_iters=30, verbose=False,
                          spike_check=False, t_relax=0.15)
    jjc = per_layer_jjc(ta, domain, ic, nm)
    print(f"  SCIF={info['scif_mT']:+.2f} mT  stop={info['stop_reason']}  "
          f"per-layer J/Jc mean: {[round(x,4) for x in jjc]}", flush=True)

    N_CHUNKS = 10
    CHUNK = 30
    for c in range(1, N_CHUNKS + 1):
        info = newton_ta.step(ta, domain, ic, nm, I, params.ramp_duration,
                              uniform, max_outer=CHUNK, min_outer=CHUNK,
                              stall_tol=1e-9, first=False, verbose=False,
                              spike_check=False, t_relax=0.15)
        jjc = per_layer_jjc(ta, domain, ic, nm)
        print(f"=== chunk {c} (+{CHUNK} outer iters, total ~{30+c*CHUNK}) ===")
        print(f"  SCIF={info['scif_mT']:+.2f} mT  stop={info['stop_reason']}  "
              f"per-layer J/Jc mean: {[round(x,4) for x in jjc]}", flush=True)

    print("\nPicard ground-truth per-layer J/Jc: "
          "[0.204, 0.349, 0.653, 0.919, 0.949, 0.919], SCIF=641.26 mT", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
