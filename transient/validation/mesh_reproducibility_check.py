"""
mesh_reproducibility_check.py -- does gmsh actually produce a different
mesh across two separate build_mesh.build() calls with IDENTICAL geometry,
for the CURRENT champion design?

This is the premise behind the "cross-process mesh sensitivity" hypothesis
raised to explain why an isolated re-run of adaptive_march_single.py 60
gave a completely different (and much worse) outcome than the original run
-- that hypothesis has been ASSUMED, drawing on this project's documented
history of gmsh non-reproducibility for OTHER designs/cases, but never
directly confirmed for this specific mesh. This script settles that in
seconds, no FEM solve needed: build the mesh twice, in the SAME process
(so nothing else differs), and compare.

Three levels of comparison, weakest to strongest claim:
  1. Byte-identical files?
  2. Same node count / cell count (same mesh topology size)?
  3. Same node coordinates and cell connectivity (same mesh, full stop)?
"""
import filecmp
import os
import sys

import numpy as np

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

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    path_a = f"{root}_reprod_A{ext}"
    path_b = f"{root}_reprod_B{ext}"

    print("Building mesh A ...", flush=True)
    build_mesh.build(write_path=path_a, verbose=False)
    print("Building mesh B (same geometry, same process) ...", flush=True)
    build_mesh.build(write_path=path_b, verbose=False)

    print("\n" + "=" * 78)
    print("LEVEL 1: byte-identical files?")
    print("=" * 78)
    byte_identical = filecmp.cmp(path_a, path_b, shallow=False)
    size_a = os.path.getsize(path_a)
    size_b = os.path.getsize(path_b)
    print(f"  file A: {size_a} bytes   file B: {size_b} bytes")
    print(f"  byte-identical: {byte_identical}")

    md_a = gmshio.read_from_msh(path_a, comm, gdim=3)
    md_b = gmshio.read_from_msh(path_b, comm, gdim=3)
    dom_a, dom_b = md_a.mesh, md_b.mesh

    x_a = dom_a.geometry.x
    x_b = dom_b.geometry.x
    tdim = dom_a.topology.dim
    n_cells_a = dom_a.topology.index_map(tdim).size_local
    n_cells_b = dom_b.topology.index_map(tdim).size_local

    print("\n" + "=" * 78)
    print("LEVEL 2: same node / cell counts?")
    print("=" * 78)
    print(f"  A: {x_a.shape[0]} nodes, {n_cells_a} cells")
    print(f"  B: {x_b.shape[0]} nodes, {n_cells_b} cells")
    same_counts = (x_a.shape[0] == x_b.shape[0]) and (n_cells_a == n_cells_b)
    print(f"  same counts: {same_counts}")

    print("\n" + "=" * 78)
    print("LEVEL 3: same node coordinates (if counts match)?")
    print("=" * 78)
    if same_counts:
        # Node ORDER is not guaranteed even for an identical mesh (dolfinx's
        # own partitioning/renumbering can differ run to run even from
        # identical input) -- compare via sorted coordinate sets, not
        # index-for-index, to avoid a false "different" from pure
        # renumbering.
        order_a = np.lexsort(x_a.T)
        order_b = np.lexsort(x_b.T)
        x_a_sorted = x_a[order_a]
        x_b_sorted = x_b[order_b]
        max_diff = float(np.max(np.abs(x_a_sorted - x_b_sorted)))
        print(f"  max |coordinate difference| after sort-matching: {max_diff:.6e} m")
        print(f"  numerically identical (< 1e-12 m): {max_diff < 1e-12}")
    else:
        print("  SKIPPED -- node/cell counts already differ, mesh is "
              "structurally different")
        max_diff = None

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if byte_identical:
        print("Mesh files are byte-identical. gmsh IS reproducible for "
              "this geometry, in this process. The mesh-sensitivity "
              "hypothesis for the earlier adaptive-marching discrepancy "
              "is NOT supported by this test -- look elsewhere.")
    elif same_counts and max_diff is not None and max_diff < 1e-12:
        print("Files differ (likely just node/element ID ordering or "
              "metadata) but the underlying mesh geometry is numerically "
              "identical. gmsh IS effectively reproducible here. The "
              "mesh-sensitivity hypothesis is NOT supported by this test.")
    else:
        print("The mesh is genuinely DIFFERENT between two builds of the "
              "identical geometry (different node/cell counts or "
              "different coordinates). This SUPPORTS the mesh-sensitivity "
              "hypothesis as at least plausible -- though it does not by "
              "itself prove the mesh difference is what flipped the "
              "adaptive-marching outcome; that would need the actual "
              "solve repeated on these two meshes.")

    for p in (path_a, path_b):
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
