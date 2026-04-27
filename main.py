"""
main.py
=======
Entry point for the solar panel kWh comparison tool — PVGIS edition.

All irradiance data comes from the PVGIS seriescalc API.  The previous
local geometric engine (solar_position.py + panel_systems.py) is no longer
used for the actual simulation; PVGIS handles geometry, atmosphere, and
tracking internally and returns G(i) — total in-plane irradiance — directly.

Usage examples:

    # Summer months, Belgium, 6 am – 8 pm UTC
    python main.py --lat 50.8 --lon 4.4 --months 5 6 7 8 --start 6 --end 20

    # Full year, Germany
    python main.py --lat 48.0 --lon 11.5 --months 1 2 3 4 5 6 7 8 9 10 11 12

    # Custom panel area / efficiency, skip tilt sweep
    python main.py --lat 35.0 --lon -120.0 --months 6 7 --area 1.7 --efficiency 0.22 --no-sweep

    # Use a specific PVGIS year and database
    python main.py --lat 50.8 --lon 4.4 --year 2019 --raddatabase PVGIS-ERA5
"""

import argparse
import pandas as pd

from simulation import (
    run_simulation,
    tilt_sweep,
    fixed_system,
    single_axis_system,
    dual_axis_system,
)
import pvgis_client
import plot


def parse_args():
    p = argparse.ArgumentParser(
        description="Solar panel kWh comparison: Fixed vs Single-Axis vs Dual-Axis"
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
                   help="Fixed tilt in degrees (default: auto = latitude, "
                        "rounded to nearest integer)")
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
    p.add_argument("--year",        type=int, default=pvgis_client.DEFAULT_YEAR,
                   help=f"PVGIS data year (default: {pvgis_client.DEFAULT_YEAR})")
    p.add_argument("--raddatabase", type=str, default=pvgis_client.DEFAULT_DB,
                   help=f"PVGIS radiation DB (default: {pvgis_client.DEFAULT_DB})")
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable PVGIS response caching")
    return p.parse_args()


def main():
    args = parse_args()

    # Default fixed tilt = latitude rounded to nearest degree
    fixed_tilt = args.tilt if args.tilt is not None else round(abs(args.lat))

    print("=" * 62)
    print("  Solar Panel kWh Comparison Tool  (PVGIS data source)")
    print("=" * 62)
    print(f"  Location    : {args.lat}°N, {args.lon}°E")
    print(f"  Months      : {args.months}")
    print(f"  Hours (UTC) : {args.start:04.1f} – {args.end:04.1f}")
    print(f"  Panel       : {args.area} m²   efficiency {args.efficiency*100:.0f}%")
    print(f"  Fixed tilt  : {fixed_tilt:.1f}°  (south-facing, azimuth 180°)")
    print(f"  PVGIS DB    : {args.raddatabase}  year {args.year}")
    print(f"  Cache       : {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 62)

    # Shared kwargs forwarded to every simulation / sweep call
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
    )
    use_cache = not args.no_cache
    common["use_cache"] = use_cache

    # --- Define systems ---
    systems = {
        "Fixed":       fixed_system(tilt_deg=fixed_tilt),
        "Single-Axis": single_axis_system(axis_tilt_deg=fixed_tilt),
        "Dual-Axis":   dual_axis_system(),
    }

    # --- Run simulations ---
    results: dict[str, pd.DataFrame] = {}
    for name, cfg in systems.items():
        print(f"\n  Simulating {name}...", end="", flush=True)
        df = run_simulation(cfg, **common)
        results[name] = df
        total_kwh = df["energy_wh"].sum() / 1000.0
        print(f"  → {total_kwh:.3f} kWh")

    # --- Summary table ---
    print()
    print(f"  {'System':<14} {'Total kWh':>10} {'vs Fixed':>10}")
    print(f"  {'-'*38}")
    fixed_kwh = results["Fixed"]["energy_wh"].sum() / 1000.0
    for name, df in results.items():
        kwh = df["energy_wh"].sum() / 1000.0
        gain = (kwh / fixed_kwh - 1) * 100 if fixed_kwh > 0 else 0.0
        gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
        print(f"  {name:<14} {kwh:>10.3f} {gain_str:>10}")
    print()

    # --- Tilt sweep ---
    optimal_tilt = fixed_tilt
    if not args.no_sweep:
        print("  Running tilt sweep (one PVGIS request per tilt angle)...")
        sweep_df = tilt_sweep(
            **common,
            tilt_min=args.tilt_min,
            tilt_max=args.tilt_max,
            tilt_step=args.tilt_step,
        )
        optimal_tilt = float(sweep_df.loc[sweep_df["total_kwh"].idxmax(), "tilt_deg"])
        print(f"\n  Optimal fixed tilt (south-facing): {optimal_tilt:.0f}°")
        print()

    # --- Plots ---
    print("  Generating plots...")
    plot.plot_summary(results)
    plot.plot_monthly_breakdown(results, args.months)
    if not args.no_sweep:
        plot.plot_tilt_sweep(sweep_df, optimal_tilt)
    plot.plot_daily_curve(results, args.months)
    print("  Done.")


if __name__ == "__main__":
    main()