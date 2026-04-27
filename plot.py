"""
plot.py
All visualization for the solar simulation results.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from typing import Sequence


SYSTEM_COLORS = {
    "Fixed":       "#E07B39",
    "Single-Axis": "#4C9BE8",
    "Dual-Axis":   "#5DBE7E",
}
BACKGROUND = "#0F1117"
PANEL_BG   = "#1A1D27"
TEXT_COLOR  = "#E8EAF0"
GRID_COLOR  = "#2A2D3A"


def _apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axes_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--")


# ---------------------------------------------------------------------------
# 1. Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary(results: dict[str, pd.DataFrame], save_path: str | None = None):
    """Horizontal bar chart comparing total kWh per system."""
    names = list(results.keys())
    totals = [df["energy_wh"].sum() / 1000 for df in results.values()]
    colors = [SYSTEM_COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(names, totals, color=colors, height=0.5, edgecolor="none")

    for bar, val in zip(bars, totals):
        ax.text(val + max(totals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} kWh", va="center", color=TEXT_COLOR, fontsize=10, fontweight="bold")

    ax.set_xlabel("Total Energy (kWh)")
    ax.set_title("Total Energy Generated — System Comparison", fontsize=13, pad=12)
    ax.set_xlim(0, max(totals) * 1.18)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Monthly breakdown
# ---------------------------------------------------------------------------

def plot_monthly_breakdown(results: dict[str, pd.DataFrame],
                           months: Sequence[int],
                           save_path: str | None = None):
    """Grouped bar chart: kWh per month per system."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    selected = sorted(months)
    n_months = len(selected)
    n_systems = len(results)

    width = 0.8 / n_systems
    x = np.arange(n_months)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (name, df) in enumerate(results.items()):
        monthly_kwh = [df[df["month"] == m]["energy_wh"].sum() / 1000 for m in selected]
        offset = (i - n_systems / 2 + 0.5) * width
        ax.bar(x + offset, monthly_kwh, width=width * 0.9,
               label=name, color=SYSTEM_COLORS.get(name, "#888"), edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m - 1] for m in selected])
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Monthly Energy Breakdown by System", fontsize=13, pad=12)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Tilt sweep — diminishing returns
# ---------------------------------------------------------------------------

def plot_tilt_sweep(tilt_df: pd.DataFrame, optimal_tilt: float,
                    save_path: str | None = None):
    """
    Line chart of kWh vs fixed tilt angle.
    Annotates the optimal tilt and shows the marginal gain curve.
    """
    tilts = tilt_df["tilt_deg"].values
    kwhs  = tilt_df["total_kwh"].values
    marginal = np.gradient(kwhs, tilts)  # dkWh / d(deg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.5]})

    # Top: total kWh
    ax1.plot(tilts, kwhs, color=SYSTEM_COLORS["Fixed"], linewidth=2.5, zorder=3)
    ax1.fill_between(tilts, kwhs, alpha=0.15, color=SYSTEM_COLORS["Fixed"])
    ax1.axvline(optimal_tilt, color="#FFCC44", linewidth=1.5, linestyle="--", zorder=4)
    ax1.text(optimal_tilt + 1, ax1.get_ylim()[0] if ax1.get_ylim()[0] else kwhs.min(),
             f"  optimal\n  {optimal_tilt:.0f}°",
             color="#FFCC44", fontsize=9, va="bottom")
    ax1.set_ylabel("Total Energy (kWh)")
    ax1.set_title("Fixed System: Energy vs Tilt Angle (South-facing)", fontsize=13, pad=10)

    # Bottom: marginal gain
    ax2.plot(tilts, marginal, color="#C07BE8", linewidth=2.0)
    ax2.axhline(0, color=GRID_COLOR, linewidth=1)
    ax2.axvline(optimal_tilt, color="#FFCC44", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Marginal gain\n(kWh/deg)", fontsize=8)
    ax2.set_xlabel("Fixed Tilt Angle (°)")

    _apply_dark_style(fig, [ax1, ax2])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Daily average power curve
# ---------------------------------------------------------------------------

def plot_daily_curve(results: dict[str, pd.DataFrame],
                     months: Sequence[int],
                     save_path: str | None = None):
    """Average power output by hour (across all simulated days)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for name, df in results.items():
        # Bin by integer hour, average power
        df = df.copy()
        df["hour_bin"] = df["hour"].apply(lambda h: int(h))
        hourly = df.groupby("hour_bin")["power_w"].mean().reset_index()
        ax.plot(hourly["hour_bin"], hourly["power_w"],
                label=name, color=SYSTEM_COLORS.get(name, "#888"),
                linewidth=2.2, marker="o", markersize=3)
        ax.fill_between(hourly["hour_bin"], hourly["power_w"],
                        alpha=0.08, color=SYSTEM_COLORS.get(name, "#888"))

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Average Power (W)")
    ax.set_title("Average Daily Power Curve by System", fontsize=13, pad=12)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
