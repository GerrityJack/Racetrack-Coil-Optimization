"""report_common.py -- shared style helpers for visualization/ramp_power_report/,
the constant-power ramp-up analysis report (circuit/power_ramp.py and
friends). Kept in its own folder, separate from the steady-state (non-
transient) solver's figures elsewhere in visualization/, per user request.

Reuses circuit/postprocess.py's _ax/_legend styling (identical dark-theme
convention as the rest of this project -- CLAUDE.md's "Figure style") but
saves into REPORT_DIR instead of the flat visualization/ directory.
"""
import os
import sys

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "circuit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                       # noqa: E402
import cparams as cfg                                # noqa: E402
from postprocess import _ax, _legend, STYLES          # noqa: E402, F401

REPORT_DIR = os.path.join(params.VIZ_DIR, "ramp_power_report")
os.makedirs(REPORT_DIR, exist_ok=True)


def save_report(fig, name):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=cfg.FIG_DPI, bbox_inches="tight",
               facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {path}")
    return path
