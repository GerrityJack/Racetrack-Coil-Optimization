"""
solve_sweep.py
================
Runs the FEM solve at several different currents (params.sweep_currents)
and saves each result separately into sweep/field_sweep/. quench_sweep.py
then *linearly* interpolates between these actual solved data points to
find the quench current at every location -- it does NOT rely on the
(true, but here deliberately not used) fact that this problem is exactly
linear in the current.

The mesh and the assembled FEM problem are built ONCE (mesh geometry and
the bilinear form a_form don't depend on current at all -- only the
source term's magnitude does), then solve.solve_at_current() just
rescales J and re-solves for each target current. This is both fast
(no redundant mesh-building/problem-assembly) and literally "running the
simulation several times" as requested -- each entry in field_sweep/ is
a genuine, independent FEM solve at that current, not a scaled copy.

Run:
    python3 solve_sweep.py
"""
from mpi4py import MPI

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import solve


def field_sweep_path(I_amps):
    return os.path.join(params.FIELD_SWEEP_DIR, f"I_{I_amps:07.2f}A.npz")


def main():
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(f"Solving at {len(params.sweep_currents)} currents: "
              f"{params.sweep_currents} A")
        print(f"Output folder: {params.FIELD_SWEEP_DIR}")

        # ── purge stale field_sweep files ───────────────────────────────────
        # Old files from a previous mesh (different n_turns / geometry) will
        # have mismatched coil_centroids and crash quench_sweep.py.
        # Delete every .npz that doesn't match the current sweep_currents set.
        expected_names = {
            os.path.basename(field_sweep_path(I)) for I in params.sweep_currents}
        removed = []
        for fname in os.listdir(params.FIELD_SWEEP_DIR):
            if fname.endswith(".npz") and fname not in expected_names:
                os.remove(os.path.join(params.FIELD_SWEEP_DIR, fname))
                removed.append(fname)
        if removed:
            print(f"  Removed {len(removed)} stale field_sweep file(s) "
                  f"from a previous run: {removed}")

    domain, cell_tags, facet_tags = solve.setup_mesh(comm)
    setup = solve.setup_problem(domain, cell_tags, facet_tags)

    log = None
    if comm.rank == 0:
        from diagnostics import SolveLog
        log = SolveLog()
        try:
            n_nodes = domain.topology.index_map(0).size_global
            n_cells = domain.topology.index_map(domain.topology.dim).size_global
        except Exception:
            n_nodes, n_cells = None, None
        log.set_header(
            mode="multi-current sweep (solve_sweep.py)",
            sweep_currents=list(params.sweep_currents),
            num_coil_cells=len(setup["coil_cells"]),
            num_V_dofs=setup["V"].dofmap.index_map.size_global,
            mesh_nodes=n_nodes, mesh_cells=n_cells,
            petsc_options=setup["petsc_options"],
            gauge_regularization=params.gauge_regularization,
        )

    for I_amps in params.sweep_currents:
        A_h, B_h = solve.solve_at_current(domain, setup, I_amps, comm,
                                           verbose_label=f"{I_amps} A", log=log)
        solve.extract_and_save(domain, cell_tags, setup["coil_cells"], B_h,
                                comm, field_sweep_path(I_amps), I_amps,
                                include_grid_slices=False)

    if comm.rank == 0:
        print(f"\nDone. {len(params.sweep_currents)} per-current datasets "
              f"written to {params.FIELD_SWEEP_DIR}")
        log.write(os.path.join(params.SWEEP_DIR, "solve_sweep_log.txt"),
                  os.path.join(params.SWEEP_DIR, "solve_sweep_log.json"))
        log.plot(os.path.join(params.SWEEP_DIR, "convergence_plot.png"))


if __name__ == "__main__":
    main()
