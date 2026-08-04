"""
dcn_crosscheck.py — Tier B2: T-A + NI against the Phase A circuit model.

These two models solve the SAME closure

    I_r = (E_p - E_i) * l_turn / R_ct ,      I_z = I - I_r

and differ in exactly one respect: where E_p and E_i come from.

    circuit/ (DCN) : E_i from a lumped mutual-inductance matrix,
                     E_p from a lumped power law with one current per turn.
    transient/     : both from the T-A field solve, so they carry the
                     screening-current distribution across the tape width.

That makes the comparison interpretable rather than apples-to-oranges: a
large disagreement means the screening currents materially change the radial
redistribution, a small one means the lumped model already captured it.  It
is also the only cross-check available on the T-A side, since Phase A's
published-benchmark test (circuit/validation/he2025_racetrack.py) cannot be
run through the T-A path -- that coil is at 77 K and this repo has only 20 K
Ic data.

Compared at the end of the ramp, where the radial current is at its plateau:
  * mean radial current I_r
  * contact power
  * bore field deficit relative to DC

Run:  <env>/bin/python3 transient/validation/dcn_crosscheck.py
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

import tparams as tp     # noqa: E402

# circuit/ is loaded by explicit file path, NOT by putting it on sys.path:
# circuit/run_charge.py and transient/run_charge.py have the same module name
# and this repo has no packages, so a bare import silently resolves to
# whichever directory happens to come first.
_CIRC = os.path.join(_ROOT, "circuit")


def _load_circuit_module(name):
    import importlib.util
    if _CIRC not in sys.path:
        sys.path.append(_CIRC)      # append, so transient/ keeps priority
    spec = importlib.util.spec_from_file_location(
        f"_circuit_{name}", os.path.join(_CIRC, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

RHO_CT = 100.0
T_RAMP = 600.0
N_RAMP = 12
N_HOLD = 4


def run_dcn(rho_ct, t_ramp):
    import params
    dcn_mod = _load_circuit_module("dcn")
    circuit_charge = _load_circuit_module("run_charge")
    geometry = _load_circuit_module("geometry")
    CoilGeometry = geometry.CoilGeometry

    geom = CoilGeometry.from_params()
    d = dcn_mod.build(geom, rho_ct_uohm_cm2=rho_ct, verbose=False)
    r = circuit_charge.run_one(d, float(params.I_design), t_ramp,
                               t_hold=6 * t_ramp, n_out=400, verbose=False)
    end = np.argmin(np.abs(r["t"] - t_ramp))
    i_end = r["Y"][:, end]
    I_now = r["I"][end]
    P_c, P_s = d.power(i_end, I_now)
    return dict(I_r_mean=float(r["i_radial"][end]),
                P_contact=P_c, P_sc=P_s,
                deficit_pct=float(r["field_deficit_at_ramp_end_pct"]),
                tau_s=r["tau_s"])


def run_ta(rho_ct, t_ramp):
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import ta_transient as tt
    import loss as loss_mod
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_x{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta, circuit, bins = tt.build(domain, cell_tags, facet_tags, uniform,
                                 rho_ct_uohm_cm2=rho_ct)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    sched = tt.ramp_schedule(params.I_design, t_ramp=t_ramp, t_hold=0.0,
                             n_ramp=N_RAMP, n_hold=0)
    hist = tt.march(ta, circuit, domain, uniform, ic, nm, sched)

    J = ta_solve._J_from_T(ta, domain)
    scale = ta["delta_SC"] / ta["Lambda"]
    Bz = ta_solve.dB_bore_from_dJ(ta["coil_centroids"], J * scale,
                                  ta["coil_vols"])[2]
    out = dict(I_r_mean=hist[-1]["I_r_mean"],
               P_contact=loss_mod.contact_power(circuit),
               P_sc=loss_mod.hysteretic_power(ta, J),
               Bz_end=float(Bz),
               n_nonconv=sum(1 for h in hist if not h["converged"]))
    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
    return out


def main():
    print("=" * 76)
    print("Tier B2 — T-A + NI vs the Phase A DCN circuit model")
    print("=" * 76)
    print(f"rho_c = {RHO_CT:.0f} uOhm.cm^2, ramp = {T_RAMP:.0f} s, "
          f"compared at the end of the ramp\n")

    print("running the DCN (seconds) ...")
    D = run_dcn(RHO_CT, T_RAMP)
    print(f"  DCN: I_r_mean = {D['I_r_mean']:.4f} A, "
          f"P_contact = {D['P_contact']:.4g} W, "
          f"P_sc = {D['P_sc']:.3g} W, tau = {D['tau_s']:.2f} s")

    print("\nrunning the T-A transient (minutes) ...")
    T = run_ta(RHO_CT, T_RAMP)
    print(f"  T-A: I_r_mean = {T['I_r_mean']:.4f} A, "
          f"P_contact = {T['P_contact']:.4g} W, "
          f"P_sc = {T['P_sc']:.3g} W")
    print(f"       bore Bz at ramp end = {T['Bz_end']:+.4f} T, "
          f"non-converged steps = {T['n_nonconv']}")

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    r_I = rel(T["I_r_mean"], D["I_r_mean"])
    r_P = rel(T["P_contact"], D["P_contact"])
    print("\n" + "=" * 76)
    print(f"{'quantity':<22}{'T-A':>14}{'DCN':>14}{'rel diff':>12}")
    print("-" * 76)
    print(f"{'mean radial current':<22}{T['I_r_mean']:>14.4f}"
          f"{D['I_r_mean']:>14.4f}{r_I*100:>11.1f}%")
    print(f"{'contact power [W]':<22}{T['P_contact']:>14.4g}"
          f"{D['P_contact']:>14.4g}{r_P*100:>11.1f}%")
    print(f"{'hysteretic power [W]':<22}{T['P_sc']:>14.3g}"
          f"{D['P_sc']:>14.3g}{'n/a':>12}")
    print("-" * 76)
    print("The hysteretic column is NOT a comparison: the DCN cannot")
    print("represent current distribution across the tape width, so its")
    print("number is meaningless by construction.  That is precisely what")
    print("the T-A model is here to supply.")
    ok = r_I < 0.30
    print(f"\nradial current agrees within 30%: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("A large gap means screening currents materially change the")
        print("radial redistribution — interesting, but resolve it before")
        print("quoting either model's transient numbers.")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
