"""
build_mesh.py — multi-layer racetrack coil, optional 1/8 symmetry
===================================================================
Full-domain mode (use_eighth_symmetry=False):
  Builds the full 3-D winding pack (all layers, all quadrants) fragmented
  against a cubic air box.  Identical to the original single-layer code
  extended to N layers.

Eighth-symmetry mode (use_eighth_symmetry=True):
  Restricts the FEM domain to the octant (x≥0, y≥0, z≤coil_half_gap).
  Three symmetry planes:
    x=0  PEC  n×A=0   (racetrack transverse mid-plane)
    y=0  PEC  n×A=0   (racetrack long-axis mid-plane)
    z=g  PMC  natural  (midplane between the two coils; image principle
                        automatically includes coil-2 field)
  All PEC faces share outer_boundary_marker; the PMC face gets
  pmc_boundary_marker so solve.py knows to leave it free.
"""
import math
import gmsh

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params


def build(write_path=None, verbose=False, _algo3d=1):
    a, b, w  = params.a, params.b, params.w
    L        = b - a
    a_out    = params.a_out
    n_layers = params.n_layers
    eighth   = getattr(params, "use_eighth_symmetry", False)
    g        = getattr(params, "coil_half_gap", 0.0)

    # Re-entrant: allow repeated builds in one process (mesh studies)
    try:
        already = gmsh.isInitialized()
    except AttributeError:
        already = False
    if already:
        gmsh.clear()
    else:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("racetrack_coil")
    occ = gmsh.model.occ

    def stadium(Lh, r, z):
        """Stadium = rectangle fused with two end-cap disks, at height z."""
        rect   = occ.addRectangle(-Lh, -r, z, 2*Lh, 2*r)
        disk_r = occ.addDisk( Lh, 0, z, r, r)
        disk_l = occ.addDisk(-Lh, 0, z, r, r)
        fused, _ = occ.fuse([(2, rect)], [(2, disk_r), (2, disk_l)])
        return fused[0][1]

    big = (L + a_out) * 3    # quarter-mask size, safely larger than coil

    # ── Coil layer volumes ───────────────────────────────────────────────────
    # Each z-layer is built as nz_sub stacked sub-slab extrusions of w/nz.
    # The mesher must place conforming node planes at every sub-slab
    # interface, which controls the z-resolution across the tape width —
    # the direction the T-A screening-current profile lives in.  The
    # in-plane size still comes from the background Distance/Threshold
    # field, so this refines anisotropically (thin-slab tets) instead of
    # isotropically exploding the cell count.
    grading = getattr(params, "mesh_z_grading", None)
    if grading:
        fr = [float(f) for f in grading]
        assert all(f > 0 for f in fr) and abs(sum(fr) - 1.0) < 1e-9, \
            "mesh_z_grading fractions must be positive and sum to 1"
        slab_heights = [w * f for f in fr]
    else:
        nz_sub = max(1, int(getattr(params, "mesh_nz_per_layer", 1)))
        slab_heights = [w / nz_sub] * nz_sub

    coil_vols = []
    for i in range(n_layers):
        z_start = params.layer_z_bottoms[i]
        a_in_i  = params.a_inner_list[i]

        outer_i = stadium(L, a_out,  z_start)
        inner_i = stadium(L, a_in_i, z_start)
        ring_i, _ = occ.cut([(2, outer_i)], [(2, inner_i)])
        ring_tag  = ring_i[0][1]

        if eighth:
            # Clip to x≥0, y≥0 quadrant
            qmask = occ.addRectangle(0, 0, z_start, big, big)
            ring_q, _ = occ.intersect([(2, ring_tag)], [(2, qmask)],
                                       removeObject=True, removeTool=True)
            ring_tag = ring_q[0][1]

        # Chain sub-slab extrusions: each returns [top_face, volume, laterals...]
        base = ring_tag
        for dz_j in slab_heights:
            ext = occ.extrude([(2, base)], 0, 0, dz_j)
            coil_vols.append([tag for dim, tag in ext if dim == 3][0])
            base = [tag for dim, tag in ext if dim == 2][0]   # top face

    # ── Air box ──────────────────────────────────────────────────────────────
    box_edge = params.box_scale * b
    if eighth:
        # x∈[0, box/2], y∈[0, box/2], z∈[-box/2, g]
        box = occ.addBox(0, 0, -box_edge/2,
                          box_edge/2, box_edge/2, box_edge/2 + g)
    else:
        box = occ.addBox(-box_edge/2, -box_edge/2, -box_edge/2,
                          box_edge, box_edge, box_edge)

    occ.fragment([(3, box)], [(3, v) for v in coil_vols])
    occ.synchronize()

    # ── Volume classification ─────────────────────────────────────────────────
    vols   = gmsh.model.getEntities(3)
    masses = sorted((occ.getMass(d, t), t) for d, t in vols)
    air_tag   = masses[-1][1]
    coil_tags = [t for _, t in masses[:-1]]

    gmsh.model.addPhysicalGroup(3, coil_tags, params.coil_marker,  name="coil")
    gmsh.model.addPhysicalGroup(3, [air_tag], params.air_marker,   name="air")

    # ── Boundary surface classification ───────────────────────────────────────
    all_vols_ents = [(3, t) for t in [air_tag] + coil_tags]
    all_bnd = {tag for dim, tag in
               gmsh.model.getBoundary(all_vols_ents, oriented=False)
               if dim == 2}

    if eighth:
        pec_tags, pmc_tags = [], []
        tol = 1e-4 * (box_edge + g)
        for tag in all_bnd:
            com = occ.getCenterOfMass(2, tag)
            if abs(com[2] - g) < tol:
                pmc_tags.append(tag)   # z=g → PMC (natural, no BC imposed)
            else:
                pec_tags.append(tag)   # x=0, y=0, outer box → PEC n×A=0
        gmsh.model.addPhysicalGroup(2, pec_tags, params.outer_boundary_marker,
                                     name="outer_boundary")
        if pmc_tags:
            gmsh.model.addPhysicalGroup(2, pmc_tags, params.pmc_boundary_marker,
                                         name="pmc_boundary")
    else:
        air_bnd = gmsh.model.getBoundary([(3, air_tag)], oriented=False)
        box_surfs = [t for d, t in air_bnd
                     if max(abs(c) for c in occ.getCenterOfMass(d, t))
                     > 0.49 * box_edge]
        gmsh.model.addPhysicalGroup(2, box_surfs, params.outer_boundary_marker,
                                     name="outer_boundary")

    # ── Mesh refinement field ─────────────────────────────────────────────────
    coil_surfs = []
    for ct in coil_tags:
        coil_surfs += [t for d, t in
                       gmsh.model.getBoundary([(3, ct)], oriented=False)]

    fdist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fdist, "SurfacesList", coil_surfs)
    fthr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(fthr, "InField",   fdist)
    gmsh.model.mesh.field.setNumber(fthr, "SizeMin",   params.mesh_size_min)
    gmsh.model.mesh.field.setNumber(fthr, "SizeMax",   params.mesh_size_max)
    gmsh.model.mesh.field.setNumber(fthr, "DistMin",   params.mesh_dist_min)
    gmsh.model.mesh.field.setNumber(fthr, "DistMax",   params.mesh_dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", _algo3d)

    try:
        gmsh.model.mesh.generate(3)
    except Exception:
        # Delaunay (algo 1) boundary recovery can fail on thin coil
        # sub-slabs (slab thickness ≪ in-plane triangle size), and a
        # failed generate leaves the meshing context unusable — so
        # rebuild the whole model from scratch with Frontal (algo 4),
        # which is robust for these anisotropic thin volumes.
        if _algo3d == 1:
            gmsh.finalize()   # full teardown — a failed generate corrupts
            return build(write_path=write_path, verbose=verbose, _algo3d=4)
        raise

    if write_path:
        gmsh.write(write_path)

    info = dict(coil_marker=params.coil_marker, air_marker=params.air_marker,
                outer_boundary_marker=params.outer_boundary_marker,
                eighth_symmetry=eighth)
    return info


if __name__ == "__main__":
    info = build(write_path=params.mesh_filename, verbose=True)
    print("Physical group markers:", info)
    gmsh.finalize()
