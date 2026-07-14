"""
ic_model.py
============
Loads Ic(B, theta) or Kc(B, theta) angular-dependence data from a
manufacturer CSV and builds an interpolator, plus the helper that turns
a field vector + the tape's local "out-of-plane" (broad-face normal)
direction into the angle that data is parametrised by.

Two CSV formats are supported -- auto-detected from the column name:

  Format A — sheet current density (default dataset):
    Column "Critical current per cm width (A/cm)"
    Converted internally to Kc in A/m, then IcModel.critical_current()
    multiplies by tape_width (params.w, 4 mm) to give Ic_eff in amps.
    Ic_eff range with w=4 mm: ~220 A (worst field/angle) – ~1690 A (0 T).

  Format B — absolute critical current in amps (Shanghai Superconductor):
    Column "Critical current (A)"
    Used directly as Ic_eff. The tape_width argument is ignored for
    scaling, because the Ic values already incorporate the tape's own
    width (confirmed ~0.5 mm based on Ic_max≈211 A vs ~4220 A/cm in
    the sheet-density dataset: 4220 A/cm × 0.05 cm ≈ 211 A).
    Switch to this dataset by passing csv_path=params.shanghai_csv_filename
    when constructing IcModel.

Angle convention (both datasets): theta = 0/180 deg means B aligned
with the tape's out-of-plane vector (perpendicular to the broad face,
i.e. along the c-axis) — lowest Ic. theta = 90 deg means B in the tape
plane — highest Ic.

The CSV angle column runs 0-240 deg; data is periodic with period 180
deg. The physical angle returned by angle_with_normal_deg() is always
in [0, 180] deg, so we use only the 0-180 deg subset directly.

Interpolation: LINEAR, with hard output clamping to [Ic_min, Ic_max].
Cubic overshoots near the sharp theta=90 peak and can produce
implausibly high quench currents — confirmed by cross-validation.
Re-run cross_validate() to verify this still holds for a new dataset.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params

DEFAULT_CSV = params.csv_filename

# Format-A constants (sheet current density, A/cm)
_COL_KC_PER_CM  = "Critical current per cm width (A/cm)"
_A_PER_CM_TO_A_PER_M = 100.0   # 1 A/cm = 100 A/m


_COL_N_VALUE = "\U0001d6ee-value"   # Unicode italic small n + "-value"
# fallback logic will still find it. You can also just paste:
#   _COL_N_VALUE = "𝘯-value"
# directly (matching the CSV header exactly).

# Format-B constant (absolute Ic, A)
_COL_IC_ABS     = "Critical current (A)"

# Backwards-compatibility alias used by cross_validate() and external callers
# that reference the old constant name:
CSV_VALUE_COL = _COL_KC_PER_CM
A_PER_CM_TO_A_PER_M = _A_PER_CM_TO_A_PER_M


class IcModel:
    def __init__(self, csv_path=DEFAULT_CSV, method="linear", tape_width=None):
        """
        csv_path: path to either supported CSV format (auto-detected):
          - Format A: "Critical current per cm width (A/cm)" column.
            tape_width (default params.w) is used to convert Kc → Ic_eff.
          - Format B: "Critical current (A)" column (Shanghai Superconductor).
            Values are used directly as Ic_eff; tape_width is ignored for
            scaling (the absolute Ic already bakes in the tape's own width).

        tape_width: relevant only for Format A CSVs (Kc in A/cm). Ignored
        for Format B. Defaults to params.w (4 mm).
        """
        self.tape_width = tape_width if tape_width is not None else params.w

        df = pd.read_csv(csv_path)
        df = df[df["Applied field angle (°)"] <= 180]

        field_vals = np.sort(df["Applied field (T)"].unique())
        angle_vals = np.sort(df["Applied field angle (°)"].unique())

        # --- auto-detect column format ---
        if _COL_KC_PER_CM in df.columns:
            # Format A: sheet current density in A/cm → convert to Ic_eff in A
            self.csv_format = "kc_per_cm"
            raw = df.pivot_table(index="Applied field (T)",
                                 columns="Applied field angle (°)",
                                 values=_COL_KC_PER_CM).reindex(
                index=field_vals, columns=angle_vals).values
            if np.isnan(raw).any():
                raise ValueError(
                    f"Kc(B,theta) grid has missing entries in {csv_path} -- "
                    "expected a complete rectangular (field × angle) grid.")
            grid_Ic = raw * _A_PER_CM_TO_A_PER_M * self.tape_width  # A
            # also store internal Kc grid in A/m for sheet_critical_current()
            self.grid = raw * _A_PER_CM_TO_A_PER_M   # A/m
            self.Kc_min = float(self.grid.min())
            self.Kc_max = float(self.grid.max())

        elif _COL_IC_ABS in df.columns:
            # Format B: absolute Ic already in amps — no tape_width scaling
            self.csv_format = "ic_amps"
            grid_Ic = df.pivot_table(index="Applied field (T)",
                                     columns="Applied field angle (°)",
                                     values=_COL_IC_ABS).reindex(
                index=field_vals, columns=angle_vals).values
            if np.isnan(grid_Ic).any():
                raise ValueError(
                    f"Ic(B,theta) grid has missing entries in {csv_path} -- "
                    "expected a complete rectangular (field × angle) grid.")
            # Synthesise a Kc grid by dividing by tape_width so that
            # sheet_critical_current() remains callable (returns A/m),
            # but note this is an approximation since tape_width may differ
            # from the tape the CSV was measured on.
            self.grid = grid_Ic / self.tape_width  # A/m (approximate Kc)
            self.Kc_min = float(self.grid.min())
            self.Kc_max = float(self.grid.max())

        else:
            raise ValueError(
                f"Unrecognised CSV format in {csv_path}: expected a column "
                f"named '{_COL_KC_PER_CM}' (Format A, sheet current density) "
                f"or '{_COL_IC_ABS}' (Format B, absolute Ic in amps). "
                f"Found columns: {list(df.columns)}")

        self.field_vals = field_vals
        self.angle_vals = angle_vals
        self.B_max = field_vals.max()
        self.B_min = field_vals.min()
        self.Ic_min = float(grid_Ic.min())
        self.Ic_max = float(grid_Ic.max())
        self._grid_Ic = grid_Ic   # Ic in amps — used by critical_current()
        self.method = method
        self._interp = RegularGridInterpolator(
            (field_vals, angle_vals), grid_Ic,
            method=method, bounds_error=False, fill_value=None)

    def sheet_critical_current(self, B_tesla, theta_deg, clip_B=True):
        """
        Vectorized Kc(B, theta) in A/m (sheet current density).
        For Format-A CSVs this is the native data unit.
        For Format-B CSVs (Ic in amps) this is derived as Ic / tape_width
        and is approximate if tape_width differs from the measured tape.
        See critical_current() for the authoritative Ic in amps.
        """
        B_tesla = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        frac_clipped = 0.0
        if clip_B:
            out_of_range = (B_tesla < self.B_min) | (B_tesla > self.B_max)
            frac_clipped = float(np.mean(out_of_range))
            B_tesla = np.clip(B_tesla, self.B_min, self.B_max)
        pts = np.stack(np.broadcast_arrays(B_tesla, theta_deg), axis=-1)
        Kc = self._interp(pts) / self.tape_width   # undo Ic→Kc for Format-B
        if self.csv_format == "kc_per_cm":
            Kc = self._interp(pts)   # native A/m grid
        else:
            Kc = self._interp(pts) / self.tape_width
        Kc = np.clip(Kc, self.Kc_min, self.Kc_max)
        return Kc, frac_clipped

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        """
        Vectorized effective critical current Ic_eff(B, theta) in amps.
        This is the quantity directly comparable to a transport current I.

        For Format-A CSVs: Ic_eff = Kc(B,theta) [A/m] × tape_width [m].
        For Format-B CSVs: Ic_eff is read directly from the table (tape_width
        is NOT applied again — the dataset already encodes the tape's width).

        clip_B=True (default) clamps B into the measured range [B_min, B_max]
        rather than extrapolating beyond it. The OUTPUT is also always
        hard-clamped to [Ic_min, Ic_max] regardless of clip_B or method.

        Returns (Ic_eff, frac_clipped) where frac_clipped is the fraction of
        input points that needed B-clipping -- if large, params.sweep_currents
        is likely too high for this geometry.
        """
        B_tesla = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        frac_clipped = 0.0
        if clip_B:
            out_of_range = (B_tesla < self.B_min) | (B_tesla > self.B_max)
            frac_clipped = float(np.mean(out_of_range))
            B_tesla = np.clip(B_tesla, self.B_min, self.B_max)
        pts = np.stack(np.broadcast_arrays(B_tesla, theta_deg), axis=-1)
        Ic = self._interp(pts)   # amps — directly for Format-B, Kc×w for Format-A
        Ic = np.clip(Ic, self.Ic_min, self.Ic_max)
        return Ic, frac_clipped


def angle_with_normal_deg(B, n_hat):
    """
    B: (..., 3) field vectors. n_hat: (..., 3) unit "out-of-plane" tape
    normal vectors (broadcastable against B). Returns angle in degrees
    in [0, 180], matching the CSV's convention (0/180 = B parallel to
    the out-of-plane vector).
    """
    B = np.asarray(B, dtype=np.float64)
    n_hat = np.asarray(n_hat, dtype=np.float64)
    Bmag = np.linalg.norm(B, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.sum(B * n_hat, axis=-1) / Bmag
    cos_theta = np.clip(np.nan_to_num(cos_theta, nan=0.0), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def cross_validate(csv_path=DEFAULT_CSV, methods=("linear", "cubic", "pchip")):
    """
    Leave-one-angle-out and leave-one-field-out cross-validation.
    Operates on Ic in amps (not Kc in A/m) so it works correctly for
    both Format-A (sheet-density) and Format-B (absolute Ic) CSVs.
    Returns {method: {"angle_rmse": ..., "field_rmse": ...}} (RMSE in A).
    """
    df = pd.read_csv(csv_path)
    df = df[df["Applied field angle (°)"] <= 180]
    field_vals = np.sort(df["Applied field (T)"].unique())
    angle_vals = np.sort(df["Applied field angle (°)"].unique())

    # detect format and build Ic grid in amps
    if _COL_KC_PER_CM in df.columns:
        raw = df.pivot_table(index="Applied field (T)",
                             columns="Applied field angle (°)",
                             values=_COL_KC_PER_CM).reindex(
            index=field_vals, columns=angle_vals).values
        grid = raw * _A_PER_CM_TO_A_PER_M * params.w  # Ic in A (for w=4mm)
    elif _COL_IC_ABS in df.columns:
        grid = df.pivot_table(index="Applied field (T)",
                              columns="Applied field angle (°)",
                              values=_COL_IC_ABS).reindex(
            index=field_vals, columns=angle_vals).values
    else:
        raise ValueError(f"Unrecognised CSV format in {csv_path}")

    report = {}
    for method in methods:
        angle_sq_errs = []
        for j in range(len(angle_vals)):
            keep = np.arange(len(angle_vals)) != j
            interp = RegularGridInterpolator(
                (field_vals, angle_vals[keep]), grid[:, keep],
                method=method, bounds_error=False, fill_value=None)
            pts = np.column_stack([field_vals, np.full(len(field_vals), angle_vals[j])])
            pred = interp(pts)
            angle_sq_errs.append((pred - grid[:, j]) ** 2)
        angle_rmse = np.sqrt(np.mean(np.concatenate(angle_sq_errs)))

        field_sq_errs = []
        for i in range(len(field_vals)):
            keep = np.arange(len(field_vals)) != i
            interp = RegularGridInterpolator(
                (field_vals[keep], angle_vals), grid[keep, :],
                method=method, bounds_error=False, fill_value=None)
            pts = np.column_stack([np.full(len(angle_vals), field_vals[i]), angle_vals])
            pred = interp(pts)
            field_sq_errs.append((pred - grid[i, :]) ** 2)
        field_rmse = np.sqrt(np.mean(np.concatenate(field_sq_errs)))

        report[method] = {"angle_rmse": angle_rmse, "field_rmse": field_rmse}
    return report, field_vals, angle_vals, grid


def plot_interpolation_check(csv_path=DEFAULT_CSV,
                              B_levels=(0.05, 0.3, 1.0, 3.0, 8.0),
                              methods=("linear", "cubic")):
    """
    Visual check: at several measured field levels, plot the raw data
    points alongside each method's interpolated curve across angle.
    Operates in native sheet-current-density units (A/m), not the
    tape-width-scaled Ic_eff. Saves ic_interpolation_check.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df = df[df["Applied field angle (°)"] <= 180]
    theta_fine = np.linspace(0, 180, 400)

    fig, axes = plt.subplots(1, len(B_levels), figsize=(4 * len(B_levels), 3.5),
                              sharey=False)
    for ax, B in zip(axes, B_levels):
        sub = df[np.isclose(df["Applied field (T)"], B)].sort_values(
            "Applied field angle (°)")
        ax.scatter(sub["Applied field angle (°)"],
                   sub[CSV_VALUE_COL] * A_PER_CM_TO_A_PER_M,
                   color="k", s=14, zorder=3, label="measured")
        for method in methods:
            model = IcModel(csv_path, method=method)
            Kc, _ = model.sheet_critical_current(np.full_like(theta_fine, B),
                                                  theta_fine, clip_B=False)
            ax.plot(theta_fine, Kc, lw=1.5, label=method)
        ax.set_title(f"B = {B} T", fontsize=12)
        ax.set_xlabel("angle (deg)")
    axes[0].set_ylabel("Kc (A/m)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(params.VIZ_DIR, "ic_interpolation_check.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


_N_CSV_DEFAULT_PATH = None   # set to params.n_value_csv_filename when added to params.py

# Critical electric field criterion — standard for power-law E-J models
E_C = 1.0e-4   # V/m


class NValueModel:
    """
    Loads n(B, θ) data from the Shanghai Superconductor n-value CSV and
    builds a smoothed bicubic spline, enabling both fast point evaluation and
    continuous partial derivatives (∂n/∂B, ∂n/∂θ) needed for full Newton
    Jacobian computation.

    Column expected: '𝘯-value' (contains the Unicode italic n character).

    Grid convention: same as IcModel —
      θ = 0°   → B perpendicular to tape broad face (c-axis, lowest Ic)
      θ = 90°  → B parallel to tape broad face (highest Ic, highest n)

    Interpolation strategy:
      1. Restrict to θ ∈ [0, 180°] (physical half; data is 180°-periodic).
      2. Apply mild Gaussian smoothing (σ = 0.8 grid cells in each axis)
         to remove experimental noise before fitting the spline. Residuals
         at data points are ~0.3 (mean) to ~1.9 (max) — well within tape-
         to-tape variation and measurement uncertainty.
      3. Fit RectBivariateSpline(θ, B) with cubic order — gives analytic
         first and second derivatives for free.
      4. Hard-clamp output to [n_min, n_max] from the data.

    WHY NOT RegularGridInterpolator:
      That is used for IcModel because linear interpolation was found to be
      more robust than cubic near the sharp θ=90° Ic peak. For n(B,θ) the
      peak is smoother and the spline is well-behaved; the analytic
      derivative is the decisive advantage.

    WHY NOT full Newton (yet):
      For the Picard scheme, only point evaluations of n(B,θ) and Jc(B,θ)
      are needed — ρ_HTS is treated as a frozen DG0 coefficient updated each
      outer iteration. Full Newton would need ∂ρ/∂B (i.e. ∂n/∂B and ∂Jc/∂B)
      threaded into the Jacobian as UFL Coefficients; the spline gives those
      derivatives but the UFL wiring is a separate step.
    """

    def __init__(self, csv_path, smooth_sigma=0.8):
        """
        csv_path  : path to the n-value CSV file.
        smooth_sigma : Gaussian smoothing width in grid-cell units applied
                       before spline fitting. 0.8 is a good default —
                       enough to suppress noise without distorting the
                       field/angle dependence. Set to 0 to skip smoothing
                       (will use raw data; spline may ring near data kinks).
        """
        df = pd.read_csv(csv_path)

        if _COL_N_VALUE not in df.columns:
            # Try to find the column by stripping unicode oddities
            candidates = [c for c in df.columns if "value" in c.lower() or
                          c.strip().lower().startswith("n")]
            if not candidates:
                raise ValueError(
                    f"NValueModel: expected column '{_COL_N_VALUE}' in "
                    f"{csv_path}. Found: {list(df.columns)}")
            import warnings
            warnings.warn(f"Using column '{candidates[0]}' as n-value column.")
            col = candidates[0]
        else:
            col = _COL_N_VALUE

        # Restrict to physical half-period [0, 180°]
        df = df[df["Applied field angle (°)"] <= 180].copy()

        # Build rectangular grid: rows = angle, cols = B
        pivot = df.pivot_table(
            index="Applied field angle (°)",
            columns="Applied field (T)",
            values=col,
        )
        self.angles = pivot.index.to_numpy(dtype=float)   # (n_ang,) [0..180°]
        self.B_vals = pivot.columns.to_numpy(dtype=float) # (n_B,)   sorted
        n_grid = pivot.values.astype(float)               # (n_ang, n_B)

        if np.isnan(n_grid).any():
            raise ValueError(
                f"n(B,θ) grid has NaN entries in {csv_path}. "
                "Expected a complete (angle × field) rectangular grid.")

        self.n_min = float(n_grid.min())
        self.n_max = float(n_grid.max())
        self.B_min = float(self.B_vals.min())
        self.B_max = float(self.B_vals.max())
        self._smooth_sigma = smooth_sigma

        # Gaussian smoothing before spline fit
        if smooth_sigma > 0:
            n_fit = gaussian_filter(n_grid, sigma=smooth_sigma)
        else:
            n_fit = n_grid.copy()

        # Bicubic spline: RectBivariateSpline(x=angles, y=B_vals, z=n_grid)
        # kx=ky=3: cubic in both directions
        self._spline = RectBivariateSpline(
            self.angles, self.B_vals, n_fit, kx=3, ky=3)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def n_value(self, B_tesla, theta_deg, clip_B=True):
        """
        Vectorized n(B, θ) evaluation.

        Parameters
        ----------
        B_tesla   : array-like, field magnitude in tesla.
        theta_deg : array-like, angle in degrees [0, 180].
                    0° = B ⊥ tape face;  90° = B ∥ tape face.
        clip_B    : if True (default), clamp B to [B_min, B_max] rather than
                    extrapolating beyond the measured range.

        Returns
        -------
        n_arr : ndarray, n-values in [n_min, n_max].
        frac_clipped : float, fraction of input points that needed B-clamping.
        """
        B_tesla   = np.atleast_1d(np.asarray(B_tesla,   dtype=np.float64))
        theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B_tesla, theta_deg = np.broadcast_arrays(B_tesla, theta_deg)
        flat_shape = B_tesla.shape
        B_flat = B_tesla.ravel()
        th_flat = theta_deg.ravel()

        frac_clipped = 0.0
        if clip_B:
            out_of_range = (B_flat < self.B_min) | (B_flat > self.B_max)
            frac_clipped = float(np.mean(out_of_range))
            B_flat = np.clip(B_flat, self.B_min, self.B_max)

        # Clamp angle to [0, 180] (handle slight numeric overshoot)
        th_flat = np.clip(th_flat, 0.0, 180.0)

        n_arr = self._spline.ev(th_flat, B_flat)
        n_arr = np.clip(n_arr, self.n_min, self.n_max)
        return n_arr.reshape(flat_shape), frac_clipped

    def dn_dB(self, B_tesla, theta_deg, clip_B=True):
        """
        ∂n/∂B at each (B, θ) point. Same signature as n_value().
        Used for full Newton Jacobian (not needed for Picard scheme).
        Units: per tesla.
        """
        B_tesla   = np.atleast_1d(np.asarray(B_tesla,   dtype=np.float64))
        theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B_tesla, theta_deg = np.broadcast_arrays(B_tesla, theta_deg)
        B_flat = B_tesla.ravel().copy()
        th_flat = np.clip(theta_deg.ravel(), 0.0, 180.0)
        if clip_B:
            B_flat = np.clip(B_flat, self.B_min, self.B_max)
        # dy=1 → first derivative with respect to y (= B axis)
        dn = self._spline.ev(th_flat, B_flat, dy=1)
        return dn.reshape(B_tesla.shape)

    def dn_dtheta(self, B_tesla, theta_deg, clip_B=True):
        """
        ∂n/∂θ at each (B, θ) point. Units: per degree.
        Used for full Newton Jacobian.
        """
        B_tesla   = np.atleast_1d(np.asarray(B_tesla,   dtype=np.float64))
        theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B_tesla, theta_deg = np.broadcast_arrays(B_tesla, theta_deg)
        B_flat = B_tesla.ravel().copy()
        th_flat = np.clip(theta_deg.ravel(), 0.0, 180.0)
        if clip_B:
            B_flat = np.clip(B_flat, self.B_min, self.B_max)
        # dx=1 → first derivative with respect to x (= angle axis)
        dn = self._spline.ev(th_flat, B_flat, dx=1)
        return dn.reshape(B_tesla.shape)


# ------------------------------------------------------------------
# Helper: compute ρ_HTS (power-law resistivity) at coil cell centroids
# ------------------------------------------------------------------

def compute_rho_hts(J_vec, B_vec, n_hat_vec, ic_model, n_model,
                    delta_SC, Lambda,
                    E_c=E_C, eps_reg=1e-4):
    """
    Compute the DG0 power-law resistivity ρ_HTS at each coil cell centroid.
    Called once per Picard iteration to update the frozen coefficient.

    Parameters
    ----------
    J_vec      : (N, 3) current density in the homogenized SC domain [A/m²].
                 This is J_physical (not scaled), i.e. the actual current
                 density in the SC layer. For the first Picard step this is
                 the uniform J from the A-form solve.
    B_vec      : (N, 3) magnetic flux density at each centroid [T].
    n_hat_vec  : (N, 3) unit tape normal at each centroid (precomputed).
    ic_model   : IcModel instance (for J_c(B, θ)).
    n_model    : NValueModel instance (for n(B, θ)).
    delta_SC   : SC layer thickness [m] (~1e-6 for REBCO).
    Lambda     : homogenisation cell pitch [m] (= params.t = tape pitch).
    E_c        : critical electric field [V/m] (default 1e-4 V/m).
    eps_reg    : regularisation to avoid ρ → 0 when J → 0.
                 Effective floor: |J|/J_c is replaced by |J|/J_c + eps_reg.
                 Typical choice: 1e-4 (so ρ_floor ~ E_c/J_c * ε^(n-1) ≪ ρ_0).

    Returns
    -------
    rho_hts : (N,) array, resistivity values [Ω·m].
    Jmag    : (N,) |J| [A/m²], stored for convergence monitoring.
    Ic_arr  : (N,) J_c(B,θ) at each cell [A], for reference.
    """
    B_mag   = np.linalg.norm(B_vec,     axis=-1)          # |B| in T
    theta   = angle_with_normal_deg(B_vec, n_hat_vec)      # angle in [0,180°]

    # Critical current from IcModel → convert to critical current DENSITY
    # J_c = I_c / (delta_SC * tape_width); but IcModel returns I_c in amps
    # and the tape width is already baked into Format-B CSVs.
    # The correct volumetric J_c for the SC layer:
    #   J_c [A/m²] = I_c [A] / (delta_SC [m] * w [m])
    # where w = params.w (tape width, 4 mm).
    Ic_arr, _ = ic_model.critical_current(B_mag, theta)    # (N,) amps
    Jc_vol    = Ic_arr / (delta_SC * ic_model.tape_width)  # [A/m²]

    n_arr, _ = n_model.n_value(B_mag, theta)               # (N,) dimensionless

    J_in_plane = J_vec - np.einsum("ij,ij->i", J_vec, n_hat_vec)[:, None] * n_hat_vec
    Jmag = np.linalg.norm(J_in_plane, axis=-1)            # |J⊥n̂|

    # Power-law resistivity with regularisation
    j_norm = Jmag / Jc_vol + eps_reg                       # |J|/Jc + ε
    rho = (E_c / Jc_vol) * j_norm ** (n_arr - 1.0)

    # Scale from SC-layer resistivity to homogenised-volume resistivity:
    # ρ_homog = ρ_SC * (delta_SC / Lambda)
    # This keeps the A-source J_s = (delta_SC/Lambda)*J_phys consistent.
    rho_homog = rho * (delta_SC / Lambda)

    return rho_homog, Jmag, Ic_arr


if __name__ == "__main__":
    for label, csv in [("Original (Kc A/cm)", params.csv_filename),
                       ("Shanghai (Ic A)", params.shanghai_csv_filename)]:
        model = IcModel(csv_path=csv)
        print(f"\n--- {label} ---")
        print(f"  CSV format: {model.csv_format}")
        print(f"  Field range: {model.B_min}-{model.B_max} T, "
              f"{len(model.angle_vals)} angles, {len(model.field_vals)} fields")
        print(f"  Ic range: {model.Ic_min:.1f}-{model.Ic_max:.1f} A")
        for B, theta in [(0.0, 45.0), (1.0, 0.0), (1.0, 90.0), (5.0, 92.0)]:
            Ic, clipped = model.critical_current(B, theta)
            print(f"  Ic({B} T, {theta} deg) = {Ic[0]:.2f} A (clipped frac={clipped:.2f})")
