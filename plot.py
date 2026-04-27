"""
plot.py
=======
All visualization for the solar simulation results.

Compatible with the DataFrame schema produced by simulation.run_simulation():
    datetime, month, hour, irradiance_wm2, power_w, energy_wh
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from typing import Sequence

# --- KU Leuven Official Brand Palette ---
SYSTEM_COLORS = {
    "Fixed":       "#DD8A2E",   # KU Leuven Accent (Orange/Bronze)
    "Single-Axis": "#52BDEC",   # KU Leuven Light Blue
    "Dual-Axis":   "#00407A",   # KU Leuven Primary Dark Blue
}
BACKGROUND  = "#FFFFFF"
PANEL_BG    = "#FFFFFF"
TEXT_COLOR  = "#222222"       # Soft black for readability
GRID_COLOR  = "#E5E5E5"       # Subtle grey for grid lines


def _apply_thesis_style(fig, axes_list):
    """Applies a clean, print-friendly styling suitable for academic papers."""
    # Use a serif font to match LaTeX typesetting
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

        # Despine the top and right to reduce chart junk
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

        ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle=":")


# ---------------------------------------------------------------------------
# 1. Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary(results: dict[str, pd.DataFrame], save_path: str | None = None):
    """Horizontal bar chart comparing total kWh per system."""
    names  = list(results.keys())
    totals = [df["energy_wh"].sum() / 1000 for df in results.values()]
    colors = [SYSTEM_COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(names, totals, color=colors, height=0.5, edgecolor="none")

    for bar, val in zip(bars, totals):
        ax.text(
            val + max(totals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} kWh",
            va="center", color=TEXT_COLOR, fontsize=10,
        )

    ax.set_xlabel("Total Energy (kWh)")
    ax.set_title("Total Energy Generated — System Comparison", pad=12)
    ax.set_xlim(0, max(totals) * 1.18)

    _apply_thesis_style(fig, [ax])
    ax.grid(False, axis='y') # Remove horizontal grid lines for bar chart

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight") # Upped DPI for print
    plt.show()


# ---------------------------------------------------------------------------
# 2. Monthly breakdown
# ---------------------------------------------------------------------------

def plot_monthly_breakdown(
    results: dict[str, pd.DataFrame],
    months: Sequence[int],
    save_path: str | None = None,
):
    """Grouped bar chart: kWh per month per system."""
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    selected  = sorted(months)
    n_months  = len(selected)
    n_systems = len(results)
    width     = 0.8 / n_systems
    x         = np.arange(n_months)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (name, df) in enumerate(results.items()):
        monthly_kwh = [
            df[df["month"] == m]["energy_wh"].sum() / 1000 for m in selected
        ]
        offset = (i - n_systems / 2 + 0.5) * width
        ax.bar(
            x + offset, monthly_kwh, width=width * 0.9,
            label=name, color=SYSTEM_COLORS.get(name, "#888"), edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m - 1] for m in selected])
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Monthly Energy Breakdown by System", pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    _apply_thesis_style(fig, [ax])
    ax.grid(False, axis='x') # Keep only horizontal grid lines

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Tilt sweep — diminishing returns
# ---------------------------------------------------------------------------

def plot_tilt_sweep(
    tilt_df: pd.DataFrame,
    optimal_tilt: float,
    save_path: str | None = None,
):
    """
    Line chart of kWh vs fixed tilt angle, with marginal gain subplot.
    """
    tilts    = tilt_df["tilt_deg"].values
    kwhs     = tilt_df["total_kwh"].values
    marginal = np.gradient(kwhs, tilts)   # dkWh / d(deg)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    # Top: total kWh
    ax1.plot(tilts, kwhs, color=SYSTEM_COLORS["Fixed"], linewidth=2.5, zorder=3)
    ax1.fill_between(tilts, kwhs, alpha=0.1, color=SYSTEM_COLORS["Fixed"])
    # Use Dark Blue for the optimal marker
    ax1.axvline(optimal_tilt, color=SYSTEM_COLORS["Dual-Axis"], linewidth=1.5, linestyle="--", zorder=4)

    y_annot = kwhs.min() + (kwhs.max() - kwhs.min()) * 0.05
    ax1.text(
        optimal_tilt + 1, y_annot,
        f"  optimal\n  {optimal_tilt:.0f}°",
        color=SYSTEM_COLORS["Dual-Axis"], fontsize=10, va="bottom",
    )
    ax1.set_ylabel("Total Energy (kWh)")
    ax1.set_title("Fixed System: Energy vs Tilt Angle (South-facing)", pad=10)

    # Bottom: marginal gain - using Light Blue
    ax2.plot(tilts, marginal, color=SYSTEM_COLORS["Single-Axis"], linewidth=2.0)
    ax2.axhline(0, color="#888888", linewidth=1)
    ax2.axvline(optimal_tilt, color=SYSTEM_COLORS["Dual-Axis"], linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Marginal gain\n(kWh/deg)", fontsize=9)
    ax2.set_xlabel("Fixed Tilt Angle (°)")

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
    save_path: str | None = None,
):
    """Average power output by UTC hour (across all simulated days)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for name, df in results.items():
        df = df.copy()
        df["hour_bin"] = df["hour"].apply(lambda h: int(h))
        hourly = df.groupby("hour_bin")["power_w"].mean().reset_index()
        ax.plot(
            hourly["hour_bin"], hourly["power_w"],
            label=name, color=SYSTEM_COLORS.get(name, "#888"),
            linewidth=2.0, marker="o", markersize=4,
        )
        ax.fill_between(
            hourly["hour_bin"], hourly["power_w"],
            alpha=0.08, color=SYSTEM_COLORS.get(name, "#888"),
        )

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Average Power (W)")
    ax.set_title("Average Daily Power Curve by System", pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    _apply_thesis_style(fig, [ax])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()