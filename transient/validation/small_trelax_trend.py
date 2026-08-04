"""
small_trelax_trend.py — is the outer-loop drift found in thin_layer_trend.py
(t_relax=0.15 walks AWAY from Picard's validated 641.26 mT, confirmed as
the one true fixed point by picard_from_newton_state.py) actually
stabilized by MORE damping (smaller t_relax), or does it just slow down
the same eventual failure?

If the Newton-hybrid's outer iteration Phi(T) is locally UNSTABLE near the
true solution (spectral radius > 1 in some mode), smaller t_relax slows
the walk-away rate but does NOT reverse its direction -- true instability,
just at lower speed. If smaller t_relax actually REVERSES the trend
(trajectory heads back toward 641 mT instead of away), that indicates
t_relax=0.15 was simply too weak a damping factor for a genuinely
stabilizable scheme, and there is some smaller t_relax that works.

Runs the SAME chunked long march as thin_layer_trend.py but at
t_relax=0.05 instead of 0.15, tracking both SCIF and per-layer J/Jc.
"""
import os
import sys

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
    import newton_ta
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_smalltr_{os.getpid()}{ext}"
    print("building mesh ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    T_RELAX = 0.05

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta, verbose=False)

    print(f"\n=== chunk 0 (bootstrap + first Newton pass, t_relax={T_RELAX}) ===",
          flush=True)
    info = newton_ta.step(ta, domain, ic, nm, I, params.ramp_duration, uniform,
                          max_outer=30, min_outer=30, stall_tol=1e-9,
                          first=True, bootstrap_iters=30, verbose=False,
                          spike_check=False, t_relax=T_RELAX)
    print(f"  SCIF={info['scif_mT']:+.2f} mT  (Picard ground truth: 641.26 mT)",
          flush=True)

    N_CHUNKS = 14
    CHUNK = 30
    prev = info["scif_mT"]
    for c in range(1, N_CHUNKS + 1):
        info = newton_ta.step(ta, domain, ic, nm, I, params.ramp_duration,
                              uniform, max_outer=CHUNK, min_outer=CHUNK,
                              stall_tol=1e-9, first=False, verbose=False,
                              spike_check=False, t_relax=T_RELAX)
        direction = "TOWARD 641" if abs(info["scif_mT"] - 641.26) < abs(prev - 641.26) else "AWAY from 641"
        print(f"=== chunk {c} (+{CHUNK}, total ~{30+c*CHUNK}) === "
              f"SCIF={info['scif_mT']:+.2f} mT  [{direction}]", flush=True)
        prev = info["scif_mT"]

    print(f"\nFinal after ~{30+N_CHUNKS*CHUNK} outer iters: SCIF={info['scif_mT']:+.2f} mT")
    print(f"Picard ground truth: +641.26 mT")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
