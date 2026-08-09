"""power_ramp_loss_plots.py -- 2026-08-08, AC-loss figures for the
constant-power ramp-up recommendation (circuit/power_ramp.py).

Two figures, both across the same rho_c sweep and ~600s-ramp power level
as circuit/power_ramp_plots.py:

  02_ac_loss_power_and_energy.png -- P_contact(t)/P_sc(t) (dissipated
      power) and cumulative E_contact(t)/E_sc(t), against the E_stored
      reference.
  03_ac_loss_vs_rho_c.png -- bar chart: total AC loss as a fraction of
      stored energy, at each rho_c. The headline sensitivity number.

P_contact/P_sc come from DCN.power(i_turn, I) -- already-validated
production machinery (energy balance closes to 0.00-0.08% per
CLAUDE.md), not new physics -- integrated with np.trapezoid over the
same ramp+hold trajectory circuit/power_ramp.py produces.

Run:  <env>/bin/python3 circuit/power_ramp_loss_plots.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "optimize"), os.path.join(_ROOT, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                      # noqa: E402
import cparams as cfg              # noqa: E402
import dcn as dcn_mod               # noqa: E402
import inductance as ind            # noqa: E402
import power_ramp as pr             # noqa: E402
from geometry import CoilGeometry   # noqa: E402
from postprocess import _ax, _legend, STYLES   # noqa: E402
from report_common import save_report           # noqa: E402

TARGET_T_RAMP = 600.0
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def find_P_for_ramp_time(d, I_design, target_t, t_guess0):
    lo, hi = 1.0, 5000.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        r = pr.run_power_ramp_auto_span(d, mid, I_design, t_guess0, verbose=False)
        if r["t_ramp_end"] > target_t:
            lo = mid
        else:
            hi = mid
    return hi


def compute_losses(d, r):
    t, Y, I = r["t"], r["Y"], r["I"]
    Pc = np.array([d.power(Y[:, j], I[j])[0] for j in range(len(t))])
    Ps = np.array([d.power(Y[:, j], I[j])[1] for j in range(len(t))])
    Ec = np.concatenate([[0.0], np.cumsum(0.5 * (Pc[1:] + Pc[:-1]) * np.diff(t))])
    Es = np.concatenate([[0.0], np.cumsum(0.5 * (Ps[1:] + Ps[:-1]) * np.diff(t))])
    return Pc, Ps, Ec, Es


def main():
    print("=" * 78)
    print("AC-loss figures for the constant-power ramp recommendation")
    print("=" * 78)
    geom = CoilGeometry.from_params()
    I_design = float(params.I_design)

    results = {}
    for rho in cfg.RHO_CT_SWEEP_UOHM_CM2:
        d = dcn_mod.build(geom, rho_ct_uohm_cm2=rho, verbose=False)
        P = find_P_for_ramp_time(d, I_design, TARGET_T_RAMP, 1200.0)
        r = pr.run_power_ramp_auto_span(d, P, I_design, 1200.0, verbose=False)
        Pc, Ps, Ec, Es = compute_losses(d, r)
        L = ind.total_inductance(d.M, d.groups, geom.two_coil)
        E_stored = 0.5 * L * I_design ** 2
        results[rho] = dict(t=r["t"], Pc=Pc, Ps=Ps, Ec=Ec, Es=Es,
                            P_target=P, t_ramp_end=r["t_ramp_end"],
                            E_stored=E_stored)
        print(f"  rho_c={rho:6.0f} uOhm.cm^2  P={P:6.2f} W  "
              f"E_contact={Ec[-1]:7.1f} J  E_sc={Es[-1]:.4f} J  "
              f"({100*(Ec[-1]+Es[-1])/E_stored:.2f}% of E_stored={E_stored:.0f}J)")

    # ── figure 1: power + cumulative energy ─────────────────────────────────
    fig, (ax_P, ax_E) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(cfg.FIG_BG)

    for k, rho in enumerate(cfg.RHO_CT_SWEEP_UOHM_CM2):
        res = results[rho]
        col, st = cfg.SERIES_COLORS[k], STYLES[k % len(STYLES)]
        lab = f"$\\rho_c$={rho:.0f} $\\mu\\Omega\\,$cm$^2$"
        ax_P.semilogy(res["t"], np.maximum(res["Pc"], 1e-6), st, color=col,
                     lw=2, label=f"{lab} (contact)")
        ax_P.semilogy(res["t"], np.maximum(res["Ps"], 1e-9), st, color=col,
                     lw=1, alpha=0.4)
        ax_E.plot(res["t"], res["Ec"], st, color=col, lw=2, label=lab)
        ax_P.axvline(res["t_ramp_end"], color=col, ls=":", lw=0.8, alpha=0.5)
        ax_E.axvline(res["t_ramp_end"], color=col, ls=":", lw=0.8, alpha=0.5)

    E_stored_ref = results[cfg.RHO_CT_SWEEP_UOHM_CM2[1]]["E_stored"]
    ax_E.axhline(E_stored_ref, color="#bbb", ls=":", lw=1.4,
                label=f"E$_{{stored}}$ = {E_stored_ref:.0f} J")

    t_ramp_ref = results[cfg.RHO_CT_SWEEP_UOHM_CM2[1]]["t_ramp_end"]
    xmax = t_ramp_ref * 2.0
    ax_P.set_xlim(0, xmax); ax_E.set_xlim(0, xmax)

    _ax(ax_P, "time [s]", "dissipated power [W]",
        "Contact/joint loss (bold) vs. SC hysteresis loss (faint) -- "
        "note the log scale:\ncontact loss dominates by 4-5 orders of magnitude")
    _legend(ax_P, loc="upper right")

    _ax(ax_E, "time [s]", "cumulative contact-loss energy [J]",
        "Energy dissipated in the NI joints during the ramp,\n"
        "vs. the coil's own stored magnetic energy")
    _legend(ax_E, loc="lower right")

    fig.suptitle(
        f"AC loss during the constant-power ramp to I_design={I_design:.0f} A "
        f"(~{TARGET_T_RAMP:.0f}s ramp) -- DCN circuit model",
        color="white", fontsize=12)
    fig.tight_layout()
    save_report(fig, "02_ac_loss_power_and_energy.png")

    # ── figure 2: AC loss fraction vs rho_c ─────────────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 5.5))
    fig2.patch.set_facecolor(cfg.FIG_BG)
    rhos = cfg.RHO_CT_SWEEP_UOHM_CM2
    fracs = [100 * (results[r]["Ec"][-1] + results[r]["Es"][-1]) / results[r]["E_stored"]
            for r in rhos]
    bar_colors = [cfg.SERIES_COLORS[k] for k in range(len(rhos))]
    bars = ax.bar([f"{r:.0f}" for r in rhos], fracs, color=bar_colors,
                 edgecolor="white", linewidth=0.6, width=0.55)
    for b, f in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, f + max(fracs) * 0.02,
               f"{f:.1f}%", ha="center", va="bottom", color="white", fontsize=11)
    _ax(ax, "contact resistivity $\\rho_c$ [$\\mu\\Omega\\,$cm$^2$]",
        "AC loss / stored energy [%]",
        f"Total AC loss (contact + SC) as a fraction of E$_{{stored}}$"
        f"={E_stored_ref:.0f} J\n(~{TARGET_T_RAMP:.0f}s ramp to "
        f"I_design={I_design:.0f} A)")
    ax.set_ylim(0, max(fracs) * 1.25)
    fig2.tight_layout()
    save_report(fig2, "03_ac_loss_vs_rho_c.png")

    print("=" * 78)


if __name__ == "__main__":
    main()
