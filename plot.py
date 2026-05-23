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
    "NS Flat":         _KUL_BLUES[5],
    "NS Rear":         _KUL_BLUES[3],
    "NS Side":         _KUL_BLUES[1],

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