"""
main.py
Entry point for the solar panel kWh comparison tool.

Usage examples:
    # Summer months, Belgium, 6am-8pm
    python main.py --lat 50.8 --lon 4.4 --months 5 6 7 8 --start 6 --end 20

    # Full year, fixed tilt 35°
    python main.py --lat 48.0 --lon 11.5 --months 1 2 3 4 5 6 7 8 9 10 11 12

    # Custom panel, skip tilt sweep
    python main.py --lat 35.0 --lon -120.0 --months 6 7 --area 1.7 --efficiency 0.22 --no-sweep
"""

import argparse
import numpy as np
import pandas as pd

from irradiance import ConstantIrradianceProvider
from panel_systems import FixedSystem, SingleAxisSystem, DualAxisSystem
from simulation import run_simulation, tilt_sweep, TIME_STEP_MINUTES
import plot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solar panel kWh comparison: Fixed vs Single-Axis vs Dual-Axis"
    )
    parser.add_argument("--lat",        type=float, default=50.8,
                        help="Latitude in degrees (default: 50.8 — Belgium)")
    parser.add_argument("--lon",        type=float, default=4.4,
                        help="Longitude in degrees (default: 4.4 — Belgium)")
    parser.add_argument("--months",     type=int,   nargs="+",
                        default=[4, 5, 6, 7, 8, 9],
                        help="Months to simulate (1-12). Default: Apr-Sep")
    parser.add_argument("--start",      type=float, default=6.0,
                        help="Start hour UTC (default: 6.0)")
    parser.add_argument("--end",        type=float, default=20.0,
                        help="End hour UTC (default: 20.0)")
    parser.add_argument("--tilt",       type=float, default=0,
                        help="Fixed system tilt in degrees (default: auto = latitude)")
    parser.add_argument("--area",       type=float, default=1.0,
                        help="Panel area in m² (default: 1.0)")
    parser.add_argument("--efficiency", type=float, default=0.20,
                        help="Panel efficiency 0-1 (default: 0.20)")
    parser.add_argument("--tilt-min",   type=float, default=0.0,
                        help="Tilt sweep start (default: 0°)")
    parser.add_argument("--tilt-max",   type=float, default=90.0,
                        help="Tilt sweep end (default: 90°)")
    parser.add_argument("--tilt-step",  type=float, default=5.0,
                        help="Tilt sweep step size (default: 5°)")
    parser.add_argument("--no-sweep",   action="store_true",
                        help="Skip the tilt sweep analysis")
    parser.add_argument("--provider",   type=str, default="constant",
                        choices=["constant", "pvgis"],
                        help="Irradiance data source: 'constant' (default) or 'pvgis'")
    parser.add_argument("--year",       type=int, default=2020,
                        help="Year to fetch from PVGIS (default: 2020, covers 2005-2020)")
    parser.add_argument("--raddatabase", type=str, default=None,
                        help="PVGIS radiation DB override, e.g. 'PVGIS-ERA5' (default: auto)")
    parser.add_argument("--validate", action="store_true",
                        help="Run PVGIS sanity checks for Fixed and Dual-Axis setups")
    return parser.parse_args()


def main():
    args = parse_args()

    # Default fixed tilt = latitude (rule of thumb for annual optimum)
    fixed_tilt = args.tilt if args.tilt is not None else abs(args.lat)

    print("=" * 60)
    print("  Solar Panel kWh Comparison Tool")
    print("=" * 60)
    print(f"  Location    : {args.lat}°N, {args.lon}°E")
    print(f"  Months      : {args.months}")
    print(f"  Hours (UTC) : {args.start:04.1f} – {args.end:04.1f}")
    print(f"  Panel area  : {args.area} m²   Efficiency: {args.efficiency*100:.0f}%")
    print(f"  Fixed tilt  : {fixed_tilt:.1f}°  (south-facing)")
    print("=" * 60)

    # Irradiance provider — swap here, nothing else changes
    if args.provider == "pvgis":
        from irradiance import PVGISProvider
        provider = PVGISProvider(year=args.year, raddatabase=args.raddatabase)
        provider_label = f"PVGIS ({args.year})"
    else:
        provider = ConstantIrradianceProvider()
        provider_label = "Constant (1370 W/m² + air mass)"

    print(f"  Provider    : {provider_label}")

    # Define systems
    systems = {
        "Fixed":       FixedSystem(tilt_deg=fixed_tilt, azimuth_deg=180.0),
        "Single-Axis": SingleAxisSystem(azimuth_deg=180.0),
        "Dual-Axis":   DualAxisSystem(),
    }

    # Run simulations
    results: dict[str, pd.DataFrame] = {}
    for name, system in systems.items():
        print(f"  Simulating {name}...", end="", flush=True)
        df = run_simulation(
            system, provider,
            months=args.months,
            start_hour=args.start,
            end_hour=args.end,
            latitude=args.lat,
            longitude=args.lon,
            panel_area_m2=args.area,
            efficiency=args.efficiency,
        )
        results[name] = df
        total_kwh = df["energy_wh"].sum() / 1000
        print(f"  → {total_kwh:.3f} kWh")

    # Print summary table
    print()
    print(f"  {'System':<14} {'Total kWh':>10} {'vs Fixed':>10}")
    print(f"  {'-'*36}")
    fixed_kwh = results["Fixed"]["energy_wh"].sum() / 1000
    for name, df in results.items():
        kwh = df["energy_wh"].sum() / 1000
        gain = (kwh / fixed_kwh - 1) * 100 if fixed_kwh > 0 else 0
        gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
        print(f"  {name:<14} {kwh:>10.3f} {gain_str:>10}")
    print()

    # --- PVGIS Sanity Check ---
    if args.validate:
        if args.provider != "pvgis":
            print("  --- PVGIS Sanity Check ---")
            print("  Validation requires '--provider pvgis'. Skipping.\n")
        else:
            print("  --- PVGIS Sanity Check ---")
            dt_step_h = TIME_STEP_MINUTES / 60.0

            # 1. Fixed Check (Tight Geometric Validation)
            # Convert local azimuth (clockwise from North, 180=South)
            # to PVGIS aspect (0=South, 90=West, -90=East)
            pvgis_aspect = 180.0 - 180.0  # Standard FixedSystem uses 180 (South)

            pvgis_fixed_kwh_m2 = provider.fetch_validation_total(
                lat=args.lat, lon=args.lon, trackingtype=0, angle=fixed_tilt, aspect=pvgis_aspect,
                months=args.months, start_hour=args.start, end_hour=args.end
            )
            # Extract raw irradiance sum from local engine (ignores area & efficiency)
            local_fixed_kwh_m2 = (results["Fixed"]["irradiance_wm2"].sum() * dt_step_h) / 1000

            fixed_diff = (local_fixed_kwh_m2 / pvgis_fixed_kwh_m2 - 1) * 100 if pvgis_fixed_kwh_m2 else 0

            print("  [Fixed Tilt - Tight Geometric Validation]")
            print(f"  Local Calculation : {local_fixed_kwh_m2:>8.2f} kWh/m²")
            print(f"  PVGIS Official    : {pvgis_fixed_kwh_m2:>8.2f} kWh/m²")
            print(f"  Difference        : {fixed_diff:>+8.1f}%   (Expected deviation: < 3%)")
            print()

            # 2. Dual-Axis Check (Upper Bound Sense Check)
            pvgis_dual_kwh_m2 = provider.fetch_validation_total(
                lat=args.lat, lon=args.lon, trackingtype=2,
                months=args.months, start_hour=args.start, end_hour=args.end
            )
            local_dual_kwh_m2 = (results["Dual-Axis"]["irradiance_wm2"].sum() * dt_step_h) / 1000

            dual_diff = (local_dual_kwh_m2 / pvgis_dual_kwh_m2 - 1) * 100 if pvgis_dual_kwh_m2 else 0

            print("  [Dual-Axis - Upper Bound Sense Check]")
            print(f"  Local Calculation : {local_dual_kwh_m2:>8.2f} kWh/m²")
            print(f"  PVGIS Official    : {pvgis_dual_kwh_m2:>8.2f} kWh/m²")
            print(f"  Difference        : {dual_diff:>+8.1f}%   (Expected deviation: 5-15%)")
            print()

    # Tilt sweep
    optimal_tilt = fixed_tilt
    if not args.no_sweep:
        print("  Running tilt sweep...", end="", flush=True)
        sweep_df = tilt_sweep(
            provider,
            months=args.months,
            start_hour=args.start,
            end_hour=args.end,
            latitude=args.lat,
            longitude=args.lon,
            panel_area_m2=args.area,
            efficiency=args.efficiency,
            tilt_min=args.tilt_min,
            tilt_max=args.tilt_max,
            tilt_step=args.tilt_step,
        )
        optimal_tilt = float(sweep_df.loc[sweep_df["total_kwh"].idxmax(), "tilt_deg"])
        print(f"  done.  Optimal fixed tilt: {optimal_tilt:.0f}°")

    # Plots
    print("  Generating plots...")
    plot.plot_summary(results)
    plot.plot_monthly_breakdown(results, args.months)
    if not args.no_sweep:
        plot.plot_tilt_sweep(sweep_df, optimal_tilt)
    plot.plot_daily_curve(results, args.months)
    print("  Done.")


if __name__ == "__main__":
    main()