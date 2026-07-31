"""
mesh_convergence_champion.py -- is the champion's box uniformity converged?
============================================================================
2026-07-31. Focus has shifted from minimizing tape (already well below
target) to VALIDATING the design. The champion's T-A box peak-to-peak
uniformity has been reproduced four times at 0.686-0.688%, but every one
of those used the SAME production mesh settings -- that is repeat noise
(independent gmsh realizations of one resolution), NOT resolution
convergence. A first probe confirmed the concern is real: dropping to
mesh_nz_per_layer=2 moves the answer to 0.549%, a 0.14pp shift.

This matters more here than for a typical FEM quantity: the box
uniformity is built from dB_bore_from_dJ(), a NEAR-CANCELLING dipole sum
over screening currents (CLAUDE.md documents this sensitivity repeatedly
-- the on-axis SCIF was resolution-limited to +-4mT for exactly this
reason). A near-cancelling sum can look stable across repeats at fixed
resolution and still drift with resolution.

Two independent refinement axes, varied on the FIXED champion geometry:

  1. z-resolution ACROSS THE TAPE WIDTH -- the axis CLAUDE.md identifies
     as the one that matters for screening (the penetration front lives
     across w). Controlled by mesh_z_grading (graded sub-slabs) or
     mesh_nz_per_layer (uniform sub-slabs).
  2. IN-PLANE resolution -- mesh_size_min_factor / mesh_size_max_factor.

Each config is one ta_validate.py subprocess on the same design, so the
ONLY thing changing is the mesh. 1 repeat each: repeat noise at fixed
resolution is already characterized at <=0.003pp (perturbation study,
22 of 23 candidates), so a second repeat buys nothing here -- the
question is the trend across resolutions, not the scatter within one.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/studies/mesh_convergence_champion.py

Outputs: optimize/runs/mesh_convergence/{log.txt, results.csv} and
per-config T-A logs.
"""
import os, sys, json, re, csv, time, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "mesh_convergence")
LOGS_DIR = os.path.join(RUN_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
CSV_PATH = os.path.join(RUN_DIR, "results.csv")

TIMEOUT_S = 2700
N_CONCURRENT = 2          # 8-core box; each solve is largely serial
UNIF_LIMIT_PCT = 1.0

# The current champion (2026-07-30). Geometry FIXED for every config.
DESIGN = dict(a=0.022227029065529628, b=0.02726822715975084,
              coil_half_gap=0.013500289306395013,
              n_turns=[295, 295, 369, 369, 2, 2],
              I_design=223.88086308072167)

PROD_GRADING = [0.075, 0.15, 0.55, 0.15, 0.075]   # params.py production

# (label, axis, mesh-override dict). `None` grading => uniform nz slabs.
CONFIGS = [
    # ── axis 1: z-resolution across the tape width ──────────────────────
    ("z_uniform_nz2", "z", dict(mesh_z_grading=None, mesh_nz_per_layer=2)),
    ("z_uniform_nz3", "z", dict(mesh_z_grading=None, mesh_nz_per_layer=3)),
    ("z_graded5_PROD", "z", dict(mesh_z_grading=PROD_GRADING)),
    ("z_graded7", "z", dict(
        mesh_z_grading=[0.05, 0.09, 0.13, 0.46, 0.13, 0.09, 0.05])),
    ("z_graded9", "z", dict(
        mesh_z_grading=[0.035, 0.06, 0.09, 0.115, 0.40,
                        0.115, 0.09, 0.06, 0.035])),
    # ── axis 2: in-plane resolution (z held at production) ──────────────
    ("inplane_fine", "in-plane", dict(
        mesh_z_grading=PROD_GRADING,
        mesh_size_min_factor=0.125, mesh_size_max_factor=0.30)),
    ("inplane_finer", "in-plane", dict(
        mesh_z_grading=PROD_GRADING,
        mesh_size_min_factor=0.0625, mesh_size_max_factor=0.15)),
    # ── both refined together ───────────────────────────────────────────
    ("both_fine", "both", dict(
        mesh_z_grading=[0.05, 0.09, 0.13, 0.46, 0.13, 0.09, 0.05],
        mesh_size_min_factor=0.125, mesh_size_max_factor=0.30)),
]

_RE = re.compile(
    r"box_ptp_pct=(?P<box>[-\d.]+) onaxis_scif_pct=(?P<onaxis>[-\d.]+) "
    r"Bz_bore_uniform=(?P<bzu>[-\d.]+) Bz_bore_TA=(?P<bzta>[-\d.]+) "
    r"converged=(?P<conv>True|False) n_iters=(?P<niters>\d+) "
    r"solve_s=(?P<solves>[-\d.]+) n_coil_cells=(?P<cells>\d+) "
    r"n_dofs=(?P<dofs>-?\d+)")

FIELDS = ["label", "axis", "mesh", "box_ptp_pct", "onaxis_scif_pct",
          "Bz_bore_uniform", "Bz_bore_TA", "n_coil_cells", "n_dofs",
          "n_iters", "converged", "solve_s"]


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_config(cfg_tuple):
    label, axis, mesh = cfg_tuple
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(dict(label=label, mesh=mesh, **DESIGN))
    env["TA_VALIDATE_REPEATS"] = "1"
    log_path = os.path.join(LOGS_DIR, f"{label}.log")
    row = dict(label=label, axis=axis, mesh=json.dumps(mesh))
    try:
        out = subprocess.run([PYTHON_BIN, "-u", "optimize/ta_validate.py"],
                             cwd=_ROOT, env=env, capture_output=True,
                             text=True, timeout=TIMEOUT_S)
        with open(log_path, "w") as f:
            f.write(out.stdout + "\n--- stderr ---\n" + out.stderr)
        m = _RE.search(out.stdout)
        if not m:
            _log(f"  [{label}] NO RESULT (see {log_path})")
            return row
        row.update(box_ptp_pct=float(m.group("box")),
                   onaxis_scif_pct=float(m.group("onaxis")),
                   Bz_bore_uniform=float(m.group("bzu")),
                   Bz_bore_TA=float(m.group("bzta")),
                   n_coil_cells=int(m.group("cells")),
                   n_dofs=int(m.group("dofs")),
                   n_iters=int(m.group("niters")),
                   converged=m.group("conv") == "True",
                   solve_s=float(m.group("solves")))
        _log(f"  [{label:<15s}] box_ptp={row['box_ptp_pct']:.3f}%  "
             f"cells={row['n_coil_cells']:>6d} dofs={row['n_dofs']:>7d}  "
             f"iters={row['n_iters']:>3d} conv={row['converged']} "
             f"{row['solve_s']:.0f}s")
    except subprocess.TimeoutExpired:
        _log(f"  [{label}] TIMED OUT after {TIMEOUT_S}s")
    except Exception:
        _log(f"  [{label}] FAILED:\n{traceback.format_exc()}")
    return row


def main():
    t0 = time.time()
    open(LOG_PATH, "w").close()
    _log(f"mesh convergence of the champion's box uniformity: "
         f"{len(CONFIGS)} mesh configs, geometry FIXED")
    _log(f"  design: a={DESIGN['a']*1e3:.3f}mm b={DESIGN['b']*1e3:.3f}mm "
         f"gap={DESIGN['coil_half_gap']*1e3:.3f}mm "
         f"n_turns={DESIGN['n_turns']} I={DESIGN['I_design']:.2f}A")

    rows = []
    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        for r in ex.map(run_config, CONFIGS):
            rows.append(r)
            with open(CSV_PATH, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    done = [r for r in rows if "box_ptp_pct" in r]
    _log("")
    _log(f"{'label':<16}{'axis':<10}{'cells':>7}{'dofs':>8}"
         f"{'box%':>8}{'conv':>6}")
    for r in sorted(done, key=lambda r: (r["axis"], r["n_coil_cells"])):
        _log(f"{r['label']:<16}{r['axis']:<10}{r['n_coil_cells']:>7d}"
             f"{r['n_dofs']:>8d}{r['box_ptp_pct']:>8.3f}"
             f"{str(r['converged']):>6}")
    if done:
        vals = [r["box_ptp_pct"] for r in done]
        _log("")
        _log(f"spread across ALL mesh configs: {min(vals):.3f} - {max(vals):.3f}% "
             f"(range {max(vals)-min(vals):.3f}pp)")
        _log(f"all configs below the {UNIF_LIMIT_PCT}% target: "
             f"{all(v <= UNIF_LIMIT_PCT for v in vals)}")
        fine = [r for r in done if r["n_coil_cells"] >= 4000]
        if len(fine) >= 2:
            fv = [r["box_ptp_pct"] for r in fine]
            _log(f"spread among the {len(fine)} FINEST configs: "
                 f"{min(fv):.3f} - {max(fv):.3f}% (range {max(fv)-min(fv):.3f}pp)")
    _log(f"total {(time.time()-t0)/60:.1f} min -> {CSV_PATH}")


if __name__ == "__main__":
    main()
