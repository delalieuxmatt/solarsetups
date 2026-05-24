"""
main.py
=======
Entry point for the solar panel kWh comparison tool — PVGIS edition.

Scenario groups are defined in scenarios.py.  Edit that file to change
what systems are compared on each plot.  This file handles CLI parsing,
simulation (with deduplication), summary printing, and plot dispatch.
"""

import argparse
import pandas as pd

from simulation import (
    run_simulation,
    tilt_sweep,
    latitude_sweep,
    tractor_system,
)
from scenarios import build_scenario_groups, LAT_SWEEP_CONFIG
import pvgis_client
import plot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Solar panel kWh comparison: Fixed vs Single-Axis vs Dual-Axis vs Tractor"
    )
    p.add_argument("--lat",         type=float, default=50.8,
                   help="Latitude in degrees (default: 50.8 — Belgium)")
    p.add_argument("--lon",         type=float, default=4.4,
                   help="Longitude in degrees (default: 4.4 — Belgium)")
    p.add_argument("--months",      type=int, nargs="+",
                   default=[4, 5, 6, 7, 8, 9],
                   help="Months to simulate (1-12). Default: Apr-Sep")
    p.add_argument("--start",       type=float, default=6.0,
                   help="Start hour UTC (default: 6.0)")
    p.add_argument("--end",         type=float, default=20.0,
                   help="End hour UTC (default: 20.0)")
    p.add_argument("--tilt",        type=float, default=None,
                   help="Fixed tilt in degrees (default: auto = latitude)")
    p.add_argument("--area",        type=float, default=1.0,
                   help="Panel area in m² (default: 1.0)")
    p.add_argument("--efficiency",  type=float, default=0.20,
                   help="Panel efficiency 0-1 (default: 0.20)")
    p.add_argument("--tilt-min",    type=float, default=0.0,
                   help="Tilt sweep start (default: 0°)")
    p.add_argument("--tilt-max",    type=float, default=90.0,
                   help="Tilt sweep end (default: 90°)")
    p.add_argument("--tilt-step",   type=float, default=5.0,
                   help="Tilt sweep step size (default: 5°)")
    p.add_argument("--no-sweep",    action="store_true",
                   help="Skip the tilt sweep analysis")
    p.add_argument("--no-lat-sweep", action="store_true",
                   help="Skip the latitude sweep analysis")
    p.add_argument("--year",        type=int, default=pvgis_client.DEFAULT_YEAR,
                   help=f"PVGIS data year (default: {pvgis_client.DEFAULT_YEAR})")
    p.add_argument("--raddatabase", type=str, default=pvgis_client.DEFAULT_DB,
                   help=f"PVGIS radiation DB (default: {pvgis_client.DEFAULT_DB})")
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable PVGIS response caching")
    # Add this line right before return p.parse_args():
    p.add_argument("--shading-sanity", action="store_true",
                   help="Plot the shading geometry sanity check and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary_table(group_name: str, results: dict[str, pd.DataFrame]) -> None:
    """Print a kWh summary table for one scenario group."""
    fixed_kwh = results.get("Fixed")
    fixed_kwh = fixed_kwh["energy_wh"].sum() / 1000.0 if fixed_kwh is not None else None

    print(f"\n  ── {group_name} ──")
    print(f"  {'System':<18} {'Total kWh':>10} {'vs Fixed':>10}")
    print(f"  {'-'*42}")
    for name, df in results.items():
        kwh = df["energy_wh"].sum() / 1000.0
        if fixed_kwh and fixed_kwh > 0:
            gain = (kwh / fixed_kwh - 1) * 100
            gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
        else:
            gain_str = "—"
        print(f"  {name:<18} {kwh:>10.3f} {gain_str:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # --- ADD THIS BLOCK ---
    if args.shading_sanity:
        from scenarios import H_FLAP, L_FLAP, D_FLAP, L_FRONT, D_FRONT, L_BACK, D_BACK
        print(f"  Generating shading sanity check plot for Lat={args.lat}°, Lon={args.lon}°...")
        plot.plot_shading_sanity_check(
            lat     = args.lat,
            lon     = args.lon,
            months = args.months,
            H_flap  = H_FLAP,
            L_flap  = L_FLAP,
            d_flap  = D_FLAP,
            L_front = L_FRONT,
            d_front = D_FRONT,
            L_back  = L_BACK,
            d_back  = D_BACK,
        )
        return
    # ----------------------

    fixed_tilt = args.tilt if args.tilt is not None else round(abs(args.lat))
    print("Fixed tilt is:", fixed_tilt)

    common = dict(
        months=args.months,
        start_hour=args.start,
        end_hour=args.end,
        latitude=args.lat,
        longitude=args.lon,
        panel_area_m2=args.area,
        efficiency=args.efficiency,
        year=args.year,
        raddatabase=args.raddatabase,
        use_cache=not args.no_cache,
    )

    # -----------------------------------------------------------------------
    # 1. Build scenario groups
    # -----------------------------------------------------------------------
    scenario_groups = build_scenario_groups(fixed_tilt, common)

    print("=" * 62)
    print("  Solar Panel kWh Comparison Tool  (PVGIS data source)")
    print("=" * 62)
    print(f"  Location       : {common["latitude"]}°N, {common["longitude"]}°E")
    print(f"  Months         : {common["months"]}")
    print(f"  Hours (UTC)    : {common["start_hour"]:04.1f} – {common["end_hour"]:04.1f}")
    print(f"  Panel          : {args.area} m²   efficiency {args.efficiency * 100:.0f}%")
    print(f"  PVGIS DB       : {args.raddatabase}  year {args.year}")
    print(f"  Cache          : {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # 2. Simulate — deduplicated across groups
    #    Systems that appear in multiple groups (Fixed, Single-Axis, Dual-Axis)
    #    are only fetched / computed once.
    # -----------------------------------------------------------------------
    sim_cache: dict[str, pd.DataFrame] = {}
    group_results: dict[str, dict[str, pd.DataFrame]] = {}

    for group_name, systems in scenario_groups.items():
        group_results[group_name] = {}
        for sys_name, cfg in systems.items():
            if sys_name not in sim_cache:
                print(f"\n  Simulating {sys_name}...", end="", flush=True)
                sim_cache[sys_name] = run_simulation(cfg, **common)
                kwh = sim_cache[sys_name]["energy_wh"].sum() / 1000.0
                print(f"  → {kwh:.3f} kWh")
            group_results[group_name][sys_name] = sim_cache[sys_name]

    # -----------------------------------------------------------------------
    # 3. Summary tables
    # -----------------------------------------------------------------------
    print()
    for group_name, results in group_results.items():
        _print_summary_table(group_name, results)
    print()

    # -----------------------------------------------------------------------
    # 4. Tilt sweeps (optional)
    # -----------------------------------------------------------------------
    sweep_results: dict[str, tuple[pd.DataFrame, float]] = {}   # name → (df, optimal_tilt)

    if not args.no_sweep:
        # Fixed south-facing sweep
        print("  Running Fixed tilt sweep...")
        fixed_sweep_df = tilt_sweep(**common,
                                    tilt_min=args.tilt_min,
                                    tilt_max=args.tilt_max,
                                    tilt_step=args.tilt_step)
        opt_fixed = float(fixed_sweep_df.loc[fixed_sweep_df["total_kwh"].idxmax(), "tilt_deg"])
        sweep_results["Fixed"] = (fixed_sweep_df, opt_fixed)
        print(f"  Optimal Fixed tilt (south-facing): {opt_fixed:.0f}°\n")

        # Tractor E-W sweep
        print("  Running Tractor E→W tilt sweep...")
        ew_factory = lambda t: tractor_system(tilt_deg=t, forward_azimuth_deg_from_north=90)
        ew_sweep_df = tilt_sweep(system_factory=ew_factory, **common,
                                 tilt_min=args.tilt_min,
                                 tilt_max=args.tilt_max,
                                 tilt_step=args.tilt_step)
        opt_ew = float(ew_sweep_df.loc[ew_sweep_df["total_kwh"].idxmax(), "tilt_deg"])
        sweep_results["Tractor E→W"] = (ew_sweep_df, opt_ew)
        print(f"  Optimal Tractor E→W tilt: {opt_ew:.0f}°\n")

        # Tractor N-S sweep
        print("  Running Tractor N→S tilt sweep...")
        ns_factory = lambda t: tractor_system(tilt_deg=t, forward_azimuth_deg_from_north=0)
        ns_sweep_df = tilt_sweep(system_factory=ns_factory, **common,
                                 tilt_min=args.tilt_min,
                                 tilt_max=args.tilt_max,
                                 tilt_step=args.tilt_step)
        opt_ns = float(ns_sweep_df.loc[ns_sweep_df["total_kwh"].idxmax(), "tilt_deg"])
        sweep_results["Tractor N→S"] = (ns_sweep_df, opt_ns)
        print(f"  Optimal Tractor N→S tilt: {opt_ns:.0f}°\n")

    # -----------------------------------------------------------------------
    # 5. Plots — one set per scenario group
    # -----------------------------------------------------------------------
    print("  Generating plots...")
    if True == False:
        for group_name, results in group_results.items():
            title_base = group_name.replace("_", " ").title()

            plot.plot_summary(
                results,
                title=f"Total Energy — {title_base}",
            )
            plot.plot_monthly_breakdown(
                results,
                common["months"],  # ← was args.months
                title=f"Monthly Breakdown — {title_base}",
            )
            plot.plot_daily_curve(
                results,
                args.months,
                title=f"Average Daily Power Curve — {title_base}",
            )

    if not args.no_sweep:
        sweeps      = {k: v[0] for k, v in sweep_results.items()}
        opt_tilts   = {k: v[1] for k, v in sweep_results.items()}
        plot.plot_tilt_sweep(sweeps, opt_tilts)

    # -----------------------------------------------------------------------
    # 6. Latitude sweep — energy vs latitude for all tracking systems
    # -----------------------------------------------------------------------
    if not args.no_lat_sweep:
        cfg = LAT_SWEEP_CONFIG
        print("  Running latitude sweep "
              f"({cfg['lat_min']}°–{cfg['lat_max']}°N, step {cfg['lat_step']}°, "
              f"lon={cfg['longitude']}°, months={cfg['months']})...")
        lat_df = latitude_sweep(
            months=cfg["months"],
            start_hour=cfg["start_hour"],
            end_hour=cfg["end_hour"],
            longitude=cfg["longitude"],
            lat_min=cfg["lat_min"],
            lat_max=cfg["lat_max"],
            lat_step=cfg["lat_step"],
            panel_area_m2=args.area,
            efficiency=args.efficiency,
            year=args.year,
            raddatabase=args.raddatabase,
            use_cache=not args.no_cache,
        )
        months_label = (
            "Full year" if cfg["months"] == list(range(1, 13))
            else f"Months {cfg['months']}"
        )
        plot.plot_latitude_sweep(
            lat_df,
            title=f"Total Energy vs Latitude — {months_label}",
        )

    print("  Done.")


if __name__ == "__main__":
    main()