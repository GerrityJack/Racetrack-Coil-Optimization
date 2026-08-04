"""
thin_layer_trend.py — is the 3x J/Jc gap on layers 4/5 (accuracy_diagnose.py)
a REAL wrong fixed point, or a FALSE STALL where t_relax's damping slows
those specific layers enough that the GLOBAL SCIF-based stall criterion
fires while they are still slowly climbing toward the true answer?

Test: call newton_ta.step() repeatedly in CHUNKS (each a fresh call with
first=False, so T/Jc/n persist and keep evolving across chunks -- step()
does not reset anything except its own internal iteration counter/stall
tracker), printing PER-LAYER J/Jc after every chunk. If layers 4/5 are
still visibly climbing chunk to chunk, the global stall check is firing
too early relative to their own (slower) convergence. If they've already
flattened near their current (wrong) value, the discrepancy is a genuine
different fixed point, not an early stop.
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
    params.mesh_filename = f"{root}_thintrend_{os.getpid()}{ext}"
    print("building mesh ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    n_layers = len(params.n_turns)
    print(f"n_turns = {params.n_turns}", flush=True)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta, verbose=False)

    print("\n=== chunk 0 (bootstrap + first Newton pass, max_outer=30) ===", flush=True)
    info = newton_ta.step(ta, domain, ic, nm, I, params.ramp_duration, uniform,
                          max_outer=30, min_outer=3, stall_tol=1e-9,
                          first=True, bootstrap_iters=30, verbose=False,
                          spike_check=False, t_relax=0.15)
    jjc = per_layer_jjc(ta, domain, ic, nm)
    print(f"  SCIF={info['scif_mT']:+.2f} mT  stop={info['stop_reason']}  "
          f"per-layer J/Jc mean: {[round(x,4) for x in jjc]}", flush=True)

    # Force many more chunks with an essentially-unreachable stall_tol so we
    # NEVER stop early -- always run the full chunk, to see the true trend.
    N_CHUNKS = 12
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

    print("\nPicard ground-truth per-layer J/Jc (from accuracy_diagnose.py): "
          "[0.204, 0.349, 0.653, 0.919, 0.949, 0.919]", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
