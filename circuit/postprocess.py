"""
postprocess.py — figures for the NI transient DCN results.

Re-runs the scenarios rather than reloading a flattened .npz: a scenario costs
~0.5 s, so recomputing is cheaper and less error-prone than serialising the
full per-group state.

Figure conventions follow the rest of the repo (CLAUDE.md "Figure style"):
#111 figure background, #0d0d1a axes, white labels, #444 spines, dpi 150,
and savefig(facecolor=...) -- without that last one matplotlib writes a white
background regardless of fig.patch.

Charting rules applied: one measure per axis (never a second y-scale), a
legend whenever more than one series is drawn, categorical hues assigned in a
fixed order (never cycled), and line STYLE as a redundant encoding alongside
hue so the series stay separable in greyscale, in print, and under colour-
vision deficiency.  (The palette validator ships as a node script and node is
not installed in this environment, so the redundant encoding is not optional
here -- it is what makes the series safe without a machine check.)

Run:  <env>/bin/python3 circuit/postprocess.py
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
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                      # noqa: E402
import cparams as cfg              # noqa: E402
import dcn as dcn_mod              # noqa: E402
import inductance as ind           # noqa: E402
import run_charge as rc            # noqa: E402
import run_discharge as rd         # noqa: E402
from geometry import CoilGeometry  # noqa: E402

VIZ = params.VIZ_DIR
STYLES = ["-", "--", "-."]          # redundant encoding alongside hue


def _ax(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(cfg.AXES_BG)
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    if title:
        ax.set_title(title, color="white", fontsize=10)
    ax.grid(alpha=0.15, color="#666", lw=0.5)


def _legend(ax, **kw):
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222",
              edgecolor="#444", framealpha=0.6, **kw)


def _save(fig, name):
    path = os.path.join(VIZ, name)
    fig.savefig(path, dpi=cfg.FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {path}")


def figure_charge(results, t_ramp):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor(cfg.FIG_BG)
    a, b, c, d = axes.ravel()

    dc = abs(results[cfg.RHO_CT_SWEEP_UOHM_CM2[0]]["Bz_dc"])

    for k, rho in enumerate(cfg.RHO_CT_SWEEP_UOHM_CM2):
        r = results[rho]
        col, st = cfg.SERIES_COLORS[k], STYLES[k % len(STYLES)]
        lab = f"$\\rho_c$ = {rho:.0f} $\\mu\\Omega\\,$cm$^2$  ($\\tau$ = {r['tau_s']:.1f} s)"
        t = r["t"]
        a.plot(t, np.abs(r["Bz"]), st, color=col, lw=2, label=lab)
        deficit = 100 * np.abs(abs(r["Bz_dc"]) - np.abs(r["Bz"])) / dc
        b.semilogy(t, np.maximum(deficit, 1e-4), st, color=col, lw=2, label=lab)
        c.plot(t, r["i_radial"], st, color=col, lw=2, label=lab)
        d.plot(t, r["E_contact"], st, color=col, lw=2, label=lab)

    # zoom on the ramp plus the settling transient -- the multi-thousand-second
    # flat tail carries no information and squeezes everything interesting
    xmax = t_ramp + 6.0 * max(r["tau_s"] for r in results.values()
                              if np.isfinite(r["tau_s"]))
    for ax in (a, b, c, d):
        ax.set_xlim(0, xmax)

    a.axhline(dc, color="#bbb", ls=":", lw=1.2, label="DC steady state")
    a.axvline(t_ramp, color="#777", ls=":", lw=1)
    _ax(a, "time [s]", "bore |Bz| [T]",
        f"Bore field during a {t_ramp:.0f} s ramp to {params.I_design:.0f} A")
    _legend(a, loc="lower right")

    b.axvline(t_ramp, color="#777", ls=":", lw=1)
    b.axhline(1.0, color="#e57373", ls=":", lw=1.2)
    _ax(b, "time [s]", "field deficit [% of DC]",
        "Field lag (log scale); dotted red = 1%")
    _legend(b, loc="upper right")

    c.axvline(t_ramp, color="#777", ls=":", lw=1)
    _ax(c, "time [s]", "radial (leakage) current [A]",
        "Current bypassing the turns through the contacts")
    _legend(c, loc="upper right")

    d.axvline(t_ramp, color="#777", ls=":", lw=1)
    _ax(d, "time [s]", "cumulative contact loss [J]",
        "Energy deposited in the winding during startup")
    _legend(d, loc="lower right")

    fig.suptitle("NI charging transient — DCN circuit model",
                 color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, "circuit_charge.png")


def figure_discharge(results, W_stored):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor(cfg.FIG_BG)
    a, b, c = axes

    for k, rho in enumerate(cfg.RHO_CT_SWEEP_UOHM_CM2):
        r = results[rho]
        col, st = cfg.SERIES_COLORS[k], STYLES[k % len(STYLES)]
        lab = f"$\\rho_c$ = {rho:.0f}  ($\\tau$ = {r['tau_s']:.1f} s)"
        a.plot(r["t"], np.abs(r["Bz"]), st, color=col, lw=2, label=lab)
        b.semilogy(r["t"], np.maximum(r["P_contact"], 1e-3), st, color=col,
                   lw=2, label=lab)
        c.plot(r["t"], r["E_contact"] / 1e3, st, color=col, lw=2, label=lab)

    _ax(a, "time [s]", "bore |Bz| [T]", "Field decay after the supply opens")
    a.set_xlim(0, min(400, max(r['t'][-1] for r in results.values())))
    _legend(a)

    _ax(b, "time [s]", "contact power [W]", "Dissipation in the winding")
    b.set_xlim(0, min(400, max(r['t'][-1] for r in results.values())))
    _legend(b)

    c.axhline(W_stored / 1e3, color="#bbb", ls=":", lw=1.4,
              label=f"stored energy {W_stored/1e3:.2f} kJ")
    _ax(c, "time [s]", "cumulative energy [kJ]",
        "All of it lands in the winding")
    c.set_xlim(0, min(400, max(r['t'][-1] for r in results.values())))
    _legend(c, loc="lower right")

    fig.suptitle("NI sudden discharge — DCN circuit model",
                 color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, "circuit_discharge.png")


def figure_turn_currents(dcn, r, t_ramp):
    """Where the current actually goes, radially and per pancake.

    This is the panel that answers whether the champion's two 3-turn
    pancakes misbehave under NI: a 3-turn pancake has only 2 turn-to-turn
    interfaces, hence almost no bypass resistance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor(cfg.FIG_BG)
    a, b = axes

    t = r["t"]
    Y = r["Y"]
    picks = [np.argmin(np.abs(t - f * t_ramp))
             for f in (0.25, 0.5, 0.75, 1.0)]
    picks += [len(t) - 1]
    labels = [f"t = {t[p]:.0f} s" for p in picks[:-1]] + ["DC (end of hold)"]
    cmap = plt.get_cmap("magma")

    order = np.argsort(dcn.groups.r)
    for k, (p, lab) in enumerate(zip(picks, labels)):
        col = cmap(0.2 + 0.65 * k / max(1, len(picks) - 1))
        a.plot(dcn.groups.r[order] * 1e3, Y[order, p], "-o", color=col,
               lw=1.6, ms=3, label=lab)
    a.axhline(r["I_op"], color="#bbb", ls=":", lw=1.2,
              label=f"transport current {r['I_op']:.0f} A")
    _ax(a, "turn radius [mm]", "spiral current per turn [A]",
        "Radial current profile across the winding pack")
    _legend(a, loc="lower right")

    # per-layer mean current vs time
    n_layers = len(params.n_turns)
    for i in range(n_layers):
        m = dcn.groups.layer == i
        if not m.any():
            continue
        w = dcn.groups.n[m]
        cur = (Y[m, :] * w[:, None]).sum(axis=0) / w.sum()
        col = cfg.SERIES_COLORS[i % len(cfg.SERIES_COLORS)]
        b.plot(t, cur, "-", color=col, lw=2,
               label=f"layer {i} ({params.n_turns[i]} turns)")
    b.axvline(t_ramp, color="#777", ls=":", lw=1)
    _ax(b, "time [s]", "mean spiral current [A]",
        "Per-pancake charging — do the 3-turn pancakes track the rest?")
    _legend(b, loc="lower right", ncol=2)

    fig.suptitle(
        f"NI current redistribution  "
        f"($\\rho_c$ = {cfg.RHO_CT_UOHM_CM2:.0f} $\\mu\\Omega\\,$cm$^2$, "
        f"{t_ramp:.0f} s ramp)", color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, "circuit_turn_currents.png")


def main():
    geom = CoilGeometry.from_params()
    I_op = float(params.I_design)
    t_ramp = cfg.CHARGE_RAMP_S
    print("building DCNs and re-running scenarios for the figures ...")

    charge, discharge = {}, {}
    W_stored = None
    dcn_nominal = None
    for rho in cfg.RHO_CT_SWEEP_UOHM_CM2:
        d = dcn_mod.build(geom, rho_ct_uohm_cm2=rho, verbose=False)
        if W_stored is None:
            L = ind.total_inductance(d.M, d.groups, geom.two_coil)
            W_stored = 0.5 * L * I_op ** 2
        charge[rho] = rc.run_one(d, I_op, t_ramp, max(cfg.CHARGE_HOLD_S,
                                                     6 * t_ramp),
                                 n_out=500, verbose=False)
        discharge[rho] = rd.run_one(d, I_op, t_end=600.0, n_out=600,
                                    verbose=False)
        if rho == cfg.RHO_CT_UOHM_CM2:
            dcn_nominal = d
        print(f"  rho_c={rho:>5.0f}: charge tau={charge[rho]['tau_s']:.2f} s, "
              f"discharge tau={discharge[rho]['tau_s']:.2f} s")

    figure_charge(charge, t_ramp)
    figure_discharge(discharge, W_stored)
    if dcn_nominal is not None:
        figure_turn_currents(dcn_nominal, charge[cfg.RHO_CT_UOHM_CM2], t_ramp)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
