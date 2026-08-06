"""
build_mesh_half.py — HALF-domain variant for the gauge-pollution investigation
================================================================================
NEW FILE, does not touch build_mesh.py's production eighth-symmetry path.

Drops the x=0/y=0 mirror cuts (which slice directly through each turn's own
racetrack current loop -- through the straight legs and the end-cap tips)
while KEEPING the z=coil_half_gap mirror (which sits in the physical air gap
between the two coils and does not cut through any turn's own loop).  This
models one full closed racetrack turn per layer instead of a quarter of one,
at ~4x the cell count of the eighth-symmetry mesh.

Boundary classification is IDENTICAL logic to build_mesh.py's eighth-symmetry
branch (z~=g -> PMC, everything else -> PEC/outer) and it is correct here
without modification: since there are no more x=0/y=0 internal cut faces at
all in this geometry, every non-PMC boundary face genuinely IS the far-field
outer box boundary, not a mislabeled symmetry plane.  solve.py's
setup_problem() reads facet_tags via params.outer_boundary_marker generically
and needs NO changes to work on meshes built by this file.
"""
import os
import sys

import gmsh

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params


def build_half(write_path=None, verbose=False, _algo3d=1,
               mesh_size_min=None, mesh_size_max=None,
               mesh_dist_min=None, mesh_dist_max=None, box_scale=None):
    """Half-domain mesh: full x/y racetrack footprint, z <= coil_half_gap.

    Optional mesh_size_*/box_scale overrides let a caller coarsen this
    research-spike mesh relative to params.py's production factors, to keep
    4x-the-cells runtimes tractable -- the production eighth-domain path in
    build_mesh.py is untouched either way.
    """
    a, b, w  = params.a, params.b, params.w
    L        = b - a
    a_out    = params.a_out
    n_layers = params.n_layers
    g        = getattr(params, "coil_half_gap", 0.0)

    size_min = mesh_size_min if mesh_size_min is not None else params.mesh_size_min
    size_max = mesh_size_max if mesh_size_max is not None else params.mesh_size_max
    dist_min = mesh_dist_min if mesh_dist_min is not None else params.mesh_dist_min
    dist_max = mesh_dist_max if mesh_dist_max is not None else params.mesh_dist_max
    bscale   = box_scale if box_scale is not None else params.box_scale

    try:
        already = gmsh.isInitialized()
    except AttributeError:
        already = False
    if already:
        gmsh.clear()
    else:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("racetrack_coil_half")
    occ = gmsh.model.occ

    def stadium(Lh, r, z):
        rect   = occ.addRectangle(-Lh, -r, z, 2*Lh, 2*r)
        disk_r = occ.addDisk( Lh, 0, z, r, r)
        disk_l = occ.addDisk(-Lh, 0, z, r, r)
        fused, _ = occ.fuse([(2, rect)], [(2, disk_r), (2, disk_l)])
        return fused[0][1]

    # ── Coil layer volumes -- NO quadrant clip (full x,y racetrack loop) ──
    grading = getattr(params, "mesh_z_grading", None)
    if grading:
        fr = [float(f) for f in grading]
        assert all(f > 0 for f in fr) and abs(sum(fr) - 1.0) < 1e-9
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
        # (no eighth-symmetry qmask intersect here -- this is the entire
        # difference in the coil geometry relative to build_mesh.build())

        base = ring_tag
        for dz_j in slab_heights:
            ext = occ.extrude([(2, base)], 0, 0, dz_j)
            coil_vols.append([tag for dim, tag in ext if dim == 3][0])
            base = [tag for dim, tag in ext if dim == 2][0]

    # ── Air box: full x,y extent, z in [-box/2, g] (z-mirror kept) ────────
    box_edge = bscale * b
    box = occ.addBox(-box_edge/2, -box_edge/2, -box_edge/2,
                     box_edge, box_edge, box_edge/2 + g)

    occ.fragment([(3, box)], [(3, v) for v in coil_vols])
    occ.synchronize()

    vols   = gmsh.model.getEntities(3)
    masses = sorted((occ.getMass(d, t), t) for d, t in vols)
    air_tag   = masses[-1][1]
    coil_tags = [t for _, t in masses[:-1]]

    gmsh.model.addPhysicalGroup(3, coil_tags, params.coil_marker,  name="coil")
    gmsh.model.addPhysicalGroup(3, [air_tag], params.air_marker,   name="air")

    all_vols_ents = [(3, t) for t in [air_tag] + coil_tags]
    all_bnd = {tag for dim, tag in
               gmsh.model.getBoundary(all_vols_ents, oriented=False)
               if dim == 2}

    pec_tags, pmc_tags = [], []
    tol = 1e-4 * (box_edge + g)
    for tag in all_bnd:
        com = occ.getCenterOfMass(2, tag)
        if abs(com[2] - g) < tol:
            pmc_tags.append(tag)
        else:
            pec_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, pec_tags, params.outer_boundary_marker,
                                name="outer_boundary")
    if pmc_tags:
        gmsh.model.addPhysicalGroup(2, pmc_tags, params.pmc_boundary_marker,
                                    name="pmc_boundary")

    coil_surfs = []
    for ct in coil_tags:
        coil_surfs += [t for d, t in
                       gmsh.model.getBoundary([(3, ct)], oriented=False)]

    fdist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fdist, "SurfacesList", coil_surfs)
    fthr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(fthr, "InField",   fdist)
    gmsh.model.mesh.field.setNumber(fthr, "SizeMin",   size_min)
    gmsh.model.mesh.field.setNumber(fthr, "SizeMax",   size_max)
    gmsh.model.mesh.field.setNumber(fthr, "DistMin",   dist_min)
    gmsh.model.mesh.field.setNumber(fthr, "DistMax",   dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", _algo3d)

    try:
        gmsh.model.mesh.generate(3)
    except Exception:
        if _algo3d == 1:
            gmsh.finalize()
            return build_half(write_path=write_path, verbose=verbose, _algo3d=4,
                              mesh_size_min=size_min, mesh_size_max=size_max,
                              mesh_dist_min=dist_min, mesh_dist_max=dist_max,
                              box_scale=bscale)
        raise

    if write_path:
        gmsh.write(write_path)

    info = dict(coil_marker=params.coil_marker, air_marker=params.air_marker,
                outer_boundary_marker=params.outer_boundary_marker,
                eighth_symmetry=False, half_domain=True)
    return info


if __name__ == "__main__":
    out = os.path.join(params.MESH_DIR, "racetrack_mesh_half.msh")
    info = build_half(write_path=out, verbose=True)
    print("Physical group markers:", info)
    print("Written to:", out)
    gmsh.finalize()
