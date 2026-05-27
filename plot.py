"""
plot.py
=======
All visualization for the solar simulation results.

Compatible with the DataFrame schema produced by simulation.run_simulation():
    datetime, month, hour, irradiance_wm2, power_w, energy_wh

All public functions accept an optional `title` parameter so that the caller
(main.py) can label each scenario group without touching this file.
Unknown system names fall back to a deterministic hash-derived color so that
new systems added in scenarios.py never raise a KeyError here.
"""

from __future__ import annotations

import hashlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from typing import Sequence

# ---------------------------------------------------------------------------
# KU Leuven Official Brand Palette
# ---------------------------------------------------------------------------

_KUL_BLUES = {
    1: "#00407A",   # KUL primary dark blue
    2: "#185FA5",
    3: "#378ADD",
    4: "#52BDEC",
    5: "#85B7EB",
}

# A warm ramp to contrast the tracker types against the blue anchors
_KUL_WARM = {
    1: "#8B2000",   # deep rust
    2: "#C23B00",   # burnt orange
    3: "#E8720C",   # amber
}

SYSTEM_COLORS = {
    # Baselines — blue anchors, visually frame the comparison
    "Fixed":           _KUL_BLUES[1],   # darkest blue
    "Dual-Axis":       _KUL_BLUES[4],   # light blue ceiling

    # Single-axis trackers — warm ramp, clearly distinct from baselines
    "HSAT":            _KUL_WARM[1],
    "VSAT":            _KUL_WARM[2],
    "PSAT":            _KUL_WARM[3],

    # Vehicle orientations
    "EW Flat":         _KUL_BLUES[5],
    "EW Rear":         _KUL_BLUES[3],
    "EW Side":         _KUL_BLUES[1],
    "NS Flat (shaded)":         _KUL_BLUES[5],
    "NS Rear":         _KUL_BLUES[3],
    "NS Sides":         _KUL_BLUES[1],

    # Vehicle tracking
    "Single-Axis NS":  _KUL_BLUES[3],
    "Single-Axis EW":  _KUL_BLUES[5],
    "Single-Axis":     _KUL_BLUES[3],
}
BACKGROUND  = "#FFFFFF"
PANEL_BG    = "#FFFFFF"
TEXT_COLOR  = "#222222"
GRID_COLOR  = "#E5E5E5"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_color(name: str) -> str:
    """Return a brand color for known systems, or a deterministic fallback."""
    if name in SYSTEM_COLORS:
        return SYSTEM_COLORS[name]
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    return f"#{h:06X}"


def _apply_thesis_style(fig, axes_list) -> None:
    """Clean, print-friendly styling suitable for academic papers."""
    plt.rcParams["font.family"] = "serif"
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axes_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.title.set_fontsize(12)
        ax.title.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["left"].set_color("#555555")
        ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle=":")


# ---------------------------------------------------------------------------
# 1. Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary(
    results: dict[str, pd.DataFrame],
    title: str = "Total Energy Generated — System Comparison",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart comparing total kWh per system."""
    names  = list(results.keys())
    totals = [df["energy_wh"].sum() / 1000 for df in results.values()]
    colors = [_get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3.0, len(names) * 0.7)))
    bars = ax.barh(names, totals, color=colors, height=0.5, edgecolor="none")

    for bar, val in zip(bars, totals):
        ax.text(
            val + max(totals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} kWh",
            va="center", color=TEXT_COLOR, fontsize=10,
        )

    ax.set_xlabel("Total Energy (kWh)")
    ax.set_title(title, pad=12)
    ax.set_xlim(0, max(totals) * 1.18)

    _apply_thesis_style(fig, [ax])
    ax.grid(False, axis="y")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Monthly breakdown
# ---------------------------------------------------------------------------

def plot_monthly_breakdown(
    results: dict[str, pd.DataFrame],
    months: Sequence[int],
    title: str = "Monthly Energy Breakdown by System",
    save_path: str | None = None,
) -> None:
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    selected  = sorted(months)          # only the months passed in
    n_months  = len(selected)
    n_systems = len(results)
    width     = 0.8 / n_systems
    x         = np.arange(n_months)     # one slot per selected month, no gaps

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (name, df) in enumerate(results.items()):
        monthly_kwh = [
            df[df["month"] == m]["energy_wh"].sum() / 1000 for m in selected
        ]
        offset = (i - n_systems / 2 + 0.5) * width
        ax.bar(
            x + offset, monthly_kwh, width=width * 0.9,
            label=name, color=_get_color(name), edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m - 1] for m in selected])  # only selected months
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(title, pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    _apply_thesis_style(fig, [ax])
    ax.grid(False, axis="x")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------------
# 3. Tilt sweep — diminishing returns
# ---------------------------------------------------------------------------

def plot_tilt_sweep(
    sweeps: dict[str, pd.DataFrame],
    optimal_tilts: dict[str, float],
    title: str = "Energy vs Tilt Angle",
    save_path: str | None = None,
) -> None:
    """
    Line chart of kWh vs tilt angle with marginal-gain subplot.
    Accepts multiple systems for comparison (one curve per key in `sweeps`).
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    # Track vertical annotation positions to avoid overlap
    used_x_positions: list[float] = []

    for name, tilt_df in sweeps.items():
        color    = _get_color(name)
        tilts    = tilt_df["tilt_deg"].values
        kwhs     = tilt_df["total_kwh"].values
        marginal = np.gradient(kwhs, tilts)
        opt      = optimal_tilts[name]

        ax1.plot(tilts, kwhs, color=color, linewidth=2.5, zorder=3, label=name)
        ax1.fill_between(tilts, kwhs, alpha=0.08, color=color)
        ax1.axvline(opt, color=color, linewidth=1.5, linestyle="--", zorder=4)

        # Stagger label x slightly if positions are crowded
        x_label = opt + 1
        while any(abs(x_label - px) < 4 for px in used_x_positions):
            x_label += 4
        used_x_positions.append(x_label)

        y_annot = kwhs.min() + (kwhs.max() - kwhs.min()) * 0.05
        ax1.text(
            x_label, y_annot,
            f"  {name}\n  opt: {opt:.0f}°",
            color=color, fontsize=9, va="bottom",
        )

        ax2.plot(tilts, marginal, color=color, linewidth=2.0)
        ax2.axvline(opt, color=color, linewidth=1.5, linestyle="--")

    ax1.set_ylabel("Total Energy (kWh)")
    ax1.set_title(title, pad=10)
    ax1.legend(frameon=False, labelcolor=TEXT_COLOR)

    ax2.axhline(0, color="#888888", linewidth=1)
    ax2.set_ylabel("Marginal gain\n(kWh/deg)", fontsize=9)
    ax2.set_xlabel("Tilt Angle (°)")

    _apply_thesis_style(fig, [ax1, ax2])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Daily average power curve
# ---------------------------------------------------------------------------

def plot_daily_curve(
    results: dict[str, pd.DataFrame],
    months: Sequence[int],
    title: str = "Average Daily Power Curve by System",
    save_path: str | None = None,
) -> None:
    """Average power output by UTC hour (across all simulated days)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for name, df in results.items():
        df = df.copy()
        df["hour_bin"] = df["hour"].apply(lambda h: int(h))
        hourly = df.groupby("hour_bin")["power_w"].mean().reset_index()
        ax.plot(
            hourly["hour_bin"], hourly["power_w"],
            label=name, color=_get_color(name),
            linewidth=2.0, marker="o", markersize=4,
        )
        ax.fill_between(
            hourly["hour_bin"], hourly["power_w"],
            alpha=0.08, color=_get_color(name),
        )

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Average Power (W)")
    ax.set_title(title, pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    _apply_thesis_style(fig, [ax])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------------
# 5. Latitude sweep — energy vs latitude per tracking system
# ---------------------------------------------------------------------------

def plot_latitude_sweep(
    df: "pd.DataFrame",
    title: str = "Total Energy vs Latitude by Tracking System",
    save_path: "str | None" = None,
) -> None:
    """
    Line chart of total kWh vs latitude, one line per tracking system.

    Parameters
    ----------
    df : DataFrame returned by simulation.latitude_sweep()
        Columns: latitude, system, total_kwh
    """
    systems = df["system"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))

    for sys_name in systems:
        sub = df[df["system"] == sys_name].sort_values("latitude")
        ax.plot(
            sub["latitude"], sub["total_kwh"],
            label=sys_name,
            color=_get_color(sys_name),
            linewidth=2.5,
            marker="o",
            markersize=5,
        )
        ax.fill_between(
            sub["latitude"], sub["total_kwh"],
            alpha=0.06,
            color=_get_color(sys_name),
        )

    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Total Energy (kWh)")
    ax.set_title(title, pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    _apply_thesis_style(fig, [ax])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Shading Geometry Sanity Check
# ---------------------------------------------------------------------------

def plot_shading_sanity_check(
        lat: float,
        lon: float,
        H_flap: float,
        L_flap: float,
        d_flap: float,
        L_front: float = 1.5,
        d_front: float = 2.04,
        L_back:  float = 1.5,
        d_back:  float = 2.66,
        months: list[int] | None = None,
        save_path: str | None = None,
) -> None:
    """
    Thesis-ready sanity check plot for the two-panel shading model.

    Single axes with a stacked-area fill showing each panel's contribution
    to the combined daily average shaded area, plus a right-hand y-axis
    for the shade fraction of total panel area.

    Each hour is split evenly between the two driving directions:
      - 50 % driving North → front panel (cab side) is shaded
      - 50 % driving South → back panel (engine side) is shaded

    The stacked bands each represent 50 % of one panel's shadow, so their
    sum equals the combined average shaded area at every hour.

    The dotted reference line marks the total panel area ceiling — the
    shaded area must always stay below it (a useful sanity check).

    Side strips on the back panel ((d_back − d_flap)/2 on each side) are
    never shaded; the shadow width is bounded by d_flap.

    Save to figs/sanity_check.png by passing save_path='figs/sanity_check.png'.
    """
    import irradiance as irr
    import pandas as pd

    # Build the set of day-of-year values for the requested months only.
    # This ensures the declination angle (and thus shadow length) reflects
    # the actual season being simulated rather than a full-year average.
    if months is None:
        months = list(range(1, 13))
    _dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    days = _dates[_dates.month.isin(months)].day_of_year.to_numpy(dtype=float)

    hours = np.linspace(6, 20, 200)

    A_front = L_front * d_front
    A_back  = L_back  * d_back
    A_total = A_front + A_back

    # Max possible shadow on each panel (shadow width bounded by d_flap)
    A_front_max = L_front * d_front   # d_front == d_flap, no overhang
    A_back_max  = L_back  * d_flap    # side strips (d_back - d_flap) always clear

    # Per-hour annual-mean shaded areas for each driving direction
    shaded_front_mean = np.zeros(len(hours))   # driving North  → front shaded
    shaded_back_mean  = np.zeros(len(hours))   # driving South  → back shaded

    for i, h in enumerate(hours):
        decl  = irr.solar_declination(days)
        omega = irr.hour_angle(h, lon, days)
        alpha = irr.solar_altitude_rad(lat, decl, omega)
        theta = irr.solar_azimuth_from_south_rad(lat, decl, omega, alpha)

        mask = (alpha <= 0.0) | (np.abs(theta) > (np.pi / 2.0))

        L_sh_front = irr._shadow_length(alpha, H_flap, L_flap, L_front)
        L_sh_back  = irr._shadow_length(alpha, H_flap, L_flap, L_back)

        # Driving North → front panel shaded
        A_fn = irr._parallelogram_shaded_area(L_sh_front, theta, d_flap)
        A_fn = np.clip(A_fn, 0.0, A_front_max)
        A_fn = np.where(mask, 0.0, A_fn)

        # Driving South → back panel shaded
        A_bs = irr._parallelogram_shaded_area(L_sh_back, theta, d_flap)
        A_bs = np.clip(A_bs, 0.0, A_back_max)
        A_bs = np.where(mask, 0.0, A_bs)

        # Each direction covers 50 % of operating time
        shaded_front_mean[i] = np.mean(A_fn) * 0.5
        shaded_back_mean[i]  = np.mean(A_bs) * 0.5

    combined_mean = shaded_front_mean + shaded_back_mean
    combined_frac = combined_mean / A_total * 100.0

    # ------------------------------------------------------------------ #
    # Single axes with right-hand fraction axis                           #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    fs = 14

    # Single combined shaded area (average over both driving directions)
    ax.fill_between(hours, combined_mean, alpha=0.25, color=_KUL_BLUES[2])
    ax.plot(hours, combined_mean, color=_KUL_BLUES[2], lw=2.0,
            label="Combined average shaded area (50% North, 50% South)")

    # Total panel area ceiling
    ax.axhline(A_total, color="#FFA500", lw=1.0, linestyle="dashdot",
               label=f"Unshaded panel area = {A_total:.2f} m²  "
                     f"(front {A_front:.2f} + back {A_back:.2f})")

    ax.set_xlabel("Hour of Day (UTC)", fontsize=fs)
    ax.set_ylabel("Average Shaded Area (m²)", fontsize = fs)
    ax.set_xlim(6, 18)
    ax.set_ylim(0, A_total * 1.25)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=fs-2,
              loc="upper left")

    # Right-hand axis: shade fraction scale (mirrors the left axis curve)
    ax_r = ax.twinx()
    ax_r.set_ylim(0, 1.25 * 100)
    ax_r.set_ylabel("Combined shade fraction (%)", color=TEXT_COLOR, fontsize = fs)
    ax_r.set_yticks([0, 20, 40, 60, 80, 100])
    ax_r.tick_params(colors=TEXT_COLOR, labelsize=fs-2)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color("#555555")
    ax_r.spines["left"].set_visible(False)

    _apply_thesis_style(fig, [ax])
    ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle=":")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 7. Yearly breakdown
# ---------------------------------------------------------------------------

def plot_yearly_breakdown(
        results: dict[str, pd.DataFrame],
        years: Sequence[int],
        title: str = "Yearly Energy Breakdown by System",
        save_path: str | None = None,
) -> None:
    print(f"\n--- {title} ---")
    for name, df in results.items():
        if "sim_year" not in df.columns:
            continue
        print(f"\nSystem: {name}")
        for year in sorted(years):
            yearly_kwh = df[df["sim_year"] == year]["energy_wh"].sum() / 1000.0
            print(f"{yearly_kwh:.2f}")
    """Bar chart comparing total kWh per system per year."""
    selected = sorted(years)
    n_years = len(selected)
    n_systems = len(results)
    width = 0.8 / n_systems
    x = np.arange(n_years)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (name, df) in enumerate(results.items()):
        if "sim_year" not in df.columns:
            continue

        yearly_kwh = [
            df[df["sim_year"] == y]["energy_wh"].sum() / 1000.0 for y in selected
        ]
        offset = (i - n_systems / 2 + 0.5) * width
        ax.bar(
            x + offset, yearly_kwh, width=width * 0.9,
            label=name, color=_get_color(name), edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in selected])
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(title, pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    _apply_thesis_style(fig, [ax])
    ax.grid(False, axis="y")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()