"""power_ramp_plots.py -- 2026-08-08, V(t)/I(t)/P(t) figure for the
constant-power ramp-up recommendation (circuit/power_ramp.py).

For each rho_c in the project's standard sweep, finds the constant power
that gives a ~600s ramp to I_design (matching the fastest ramp this
project has direct T-A evidence for -- see CLAUDE.md's "Ramp-up power
analysis"), then plots the supply current I(t), terminal voltage V(t)
(computed from DCN.terminal_voltage() -- the same validated, self-
consistent quantity the control law itself is built on, not re-derived),
and power P(t)=I(t)*V(t) across the ramp+hold trajectory.

Figure conventions match circuit/postprocess.py / CLAUDE.md's "Figure
style": #111 figure background, #0d0d1a axes, white labels, #444
spines, dpi 150, one measure per axis, hue+linestyle redundant encoding.

Run:  <env>/bin/python3 circuit/power_ramp_plots.py
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
import dcn as dcn_mod               # noqa: E402
import power_ramp as pr             # noqa: E402
from geometry import CoilGeometry   # noqa: E402
from postprocess import _ax, _legend, _save, STYLES   # noqa: E402

TARGET_T_RAMP = 600.0   # matches the fastest T-A-validated schedule (10x60s)


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


def compute_VIP(d, r):
    """V(t), P(t) from a run_power_ramp() result dict."""
    Y, I = r["Y"], r["I"]
    V = np.array([d.terminal_voltage(Y[:, j], I[j]) for j in range(Y.shape[1])])
    P = I * V
    return V, P


def main():
    print("=" * 78)
    print("V(t) / I(t) / P(t) for the constant-power ramp recommendation")
    print("=" * 78)
    geom = CoilGeometry.from_params()
    I_design = float(params.I_design)

    results = {}
    for rho in cfg.RHO_CT_SWEEP_UOHM_CM2:
        d = dcn_mod.build(geom, rho_ct_uohm_cm2=rho, verbose=False)
        P = find_P_for_ramp_time(d, I_design, TARGET_T_RAMP, 1200.0)
        r = pr.run_power_ramp_auto_span(d, P, I_design, 1200.0, verbose=False)
        V, Pt = compute_VIP(d, r)
        results[rho] = dict(t=r["t"], I=r["I"], V=V, P=Pt,
                            t_ramp_end=r["t_ramp_end"], P_target=P,
                            i_spiral_mean=r["i_spiral_mean"])
        print(f"  rho_c={rho:6.0f} uOhm.cm^2  P={P:6.2f} W  "
              f"t_ramp={r['t_ramp_end']:.1f} s")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    ax_I, ax_V, ax_P = axes
    fig.patch.set_facecolor(cfg.FIG_BG)

    t_ramp_ref = results[cfg.RHO_CT_SWEEP_UOHM_CM2[1]]["t_ramp_end"]  # nominal (100)
    xmax = t_ramp_ref * 2.5

    for k, rho in enumerate(cfg.RHO_CT_SWEEP_UOHM_CM2):
        r = results[rho]
        col, st = cfg.SERIES_COLORS[k], STYLES[k % len(STYLES)]
        lab = (f"$\\rho_c$={rho:.0f} $\\mu\\Omega\\,$cm$^2$  "
              f"(P={r['P_target']:.1f} W, t$_{{ramp}}$={r['t_ramp_end']:.0f} s)")
        ax_I.plot(r["t"], r["I"], st, color=col, lw=2, label=lab)
        ax_I.plot(r["t"], r["i_spiral_mean"], st, color=col, lw=1, alpha=0.5)
        ax_V.plot(r["t"], r["V"], st, color=col, lw=2, label=lab)
        ax_P.plot(r["t"], r["P"], st, color=col, lw=2, label=lab)
        ax_I.axvline(r["t_ramp_end"], color=col, ls=":", lw=1, alpha=0.6)

    ax_I.axhline(I_design, color="#bbb", ls=":", lw=1.2, label="I_design (196A)")
    _ax(ax_I, "", "current [A]",
        "Supply current I(t) (bold) and mean spiral current i_turn(t) (faint) --\n"
        "the gap between them is NI radial leakage")
    _legend(ax_I, loc="lower right")

    _ax(ax_V, "", "terminal voltage [V]",
        "V(t) = DCN.terminal_voltage(i_turn(t), I(t)) -- exact, self-consistent")
    _legend(ax_V, loc="upper right")

    _ax(ax_P, "time [s]", "power [W]",
        "P(t) = I(t)*V(t) -- constant during the ramp by construction, "
        "decays during hold as leakage vanishes")
    _legend(ax_P, loc="upper right")

    for ax in axes:
        ax.set_xlim(0, xmax)

    fig.suptitle(
        f"Constant-power ramp to I_design={I_design:.0f} A "
        f"(target ~{TARGET_T_RAMP:.0f}s ramp, matching the fastest\n"
        f"T-A-validated schedule) -- DCN circuit model, champion geometry",
        color="white", fontsize=11)
    fig.tight_layout()
    _save(fig, "circuit_power_ramp_VIP.png")

    print("=" * 78)


if __name__ == "__main__":
    main()
