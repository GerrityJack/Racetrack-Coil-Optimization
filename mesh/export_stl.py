"""
export_stl.py — geometry-only STL/STEP export of the racetrack coil pack,
for visual sanity-checking and CAD handoff (not a solver input).

Builds the same per-layer "stadium" solids (rectangle fused with two
semicircular end-cap disks) that build_mesh.py uses for the FEM mesh, but
skips the air box, symmetry clipping, and volume meshing.

Two output kinds:
  - .stl: a triangulated surface tessellation. Curvature-adaptive meshing
    is enabled (production build_mesh.py disables it, since it doesn't
    matter for the FEM field solve) so the racetrack ends render as smooth
    arcs instead of faceted polygons -- but an STL is still just triangles:
    no exact edges/radii survive, so it can only be eyeballed or measured
    approximately between mesh vertices.
  - .step/.stp: the exact analytic BREP geometry (true circular arcs,
    planes) straight from the OCC kernel, no tessellation -- the format to
    hand anyone who needs to actually measure or re-derive dimensions in a
    CAD tool. Written directly from the solids, no meshing step needed.

All coordinates are in METERS (params.py's native units) -- STL/STEP files
carry no unit metadata, so this is on the reader to know. Some CAD tools
assume STL is in mm by default; tell the recipient to import as meters.

Usage:
    conda run -n fenicsx-env python3 mesh/export_stl.py [output.stl ...] [--both-coils]

    Pass any number of output paths; the extension (.stl or .step/.stp)
    picks the format. Example:
        mesh/export_stl.py racetrack_coil.stl racetrack_coil.step --both-coils
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gmsh
import params


def stadium(occ, Lh, r, z):
    rect   = occ.addRectangle(-Lh, -r, z, 2*Lh, 2*r)
    disk_r = occ.addDisk( Lh, 0, z, r, r)
    disk_l = occ.addDisk(-Lh, 0, z, r, r)
    fused, _ = occ.fuse([(2, rect)], [(2, disk_r), (2, disk_l)])
    return fused[0][1]


def build_coil_solids(occ, L, a_out, both_coils):
    n_layers = params.n_layers
    vols = []
    for i in range(n_layers):
        z_start = params.layer_z_bottoms[i]
        a_in_i  = params.a_inner_list[i]
        outer_i = stadium(occ, L, a_out,  z_start)
        inner_i = stadium(occ, L, a_in_i, z_start)
        ring, _ = occ.cut([(2, outer_i)], [(2, inner_i)])
        ring_tag = ring[0][1]
        ext = occ.extrude([(2, ring_tag)], 0, 0, params.w)
        vols.append([tag for dim, tag in ext if dim == 3][0])

    if both_coils:
        g = params.coil_half_gap
        copies = occ.copy([(3, v) for v in vols])
        # mirror about the z=g midplane. gmsh's occ.mirror plane convention
        # is a*x+b*y+c*z+d=0, so the z=g plane needs d=-g (verified by
        # checking the resulting bounding box -- passing d=+g mirrors to
        # the wrong side, about z=-g instead).
        occ.mirror(copies, 0, 0, 1, -g)
        vols += [t for _, t in copies]
    return vols


def main(write_paths=("racetrack_coil.stl",), both_coils=False, verbose=True):
    if isinstance(write_paths, str):
        write_paths = (write_paths,)
    a, b, w  = params.a, params.b, params.w
    L        = b - a
    a_out    = params.a_out

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("racetrack_coil_stl")
    occ = gmsh.model.occ

    build_coil_solids(occ, L, a_out, both_coils)
    occ.synchronize()

    # Exact BREP formats -- write straight from the OCC kernel, no
    # tessellation, no meshing step needed.
    exact_paths = [p for p in write_paths if p.lower().endswith((".step", ".stp"))]
    if exact_paths:
        # STEP always declares its length unit as mm internally; without
        # telling gmsh the model it's holding is in meters, it writes our
        # raw meter-scale numbers under an mm label -- a 1000x error baked
        # into the file (worse than STL's no-unit-metadata case, since a
        # reader trusts the declared unit). Confirmed by writing a known
        # 4mm box both ways and diffing the CARTESIAN_POINT values.
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    for p in exact_paths:
        gmsh.write(p)

    # Tessellated formats (.stl) need a surface mesh first. Curvature-
    # adaptive sizing so the semicircular ends render as smooth arcs,
    # not the faceted look of a coarse/curvature-blind mesh.
    mesh_paths = [p for p in write_paths if p not in exact_paths]
    if mesh_paths:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)  # elements per 2*pi
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05 * a_out)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.15 * a_out)
        gmsh.model.mesh.generate(2)
        for p in mesh_paths:
            gmsh.write(p)

    gmsh.finalize()
    print(f"Wrote {list(write_paths)}"
          f" ({'both coils' if both_coils else 'coil 1 only'},"
          f" n_layers={params.n_layers}, units=meters)")
    return list(write_paths)


if __name__ == "__main__":
    args = sys.argv[1:]
    both = "--both-coils" in args
    args = [a for a in args if a != "--both-coils"]
    outs = args if args else ["racetrack_coil.stl"]
    main(write_paths=outs, both_coils=both)
