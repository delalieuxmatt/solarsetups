"""
simulation.py
=============
Time-series simulation engine — PVGIS edition.

All irradiance values come directly from PVGIS G(i) (global in-plane
irradiance on the panel surface).  We never decompose into DNI/DHI or
compute angles of incidence locally.  That was the source of the flat-panel
bias in the previous version: the isotropic diffuse term
    DHI * (1 + cos(tilt)) / 2
is maximised at tilt=0, and at Belgium's latitude the diffuse component
dominates enough to make a flat panel appear optimal.

PVGIS tracking types used here:
    0  fixed plane    → requires angle (tilt) + aspect (azimuth from South)
    1  single-axis N-S horizontal tracker  → requires angle (axis tilt)
    2  dual-axis tracker

The tilt_sweep function sends one PVGIS request per tilt angle (south-facing,
fixed) and is the authoritative source for the optimum fixed tilt.

DataFrame schema returned by run_simulation() — identical to the old schema
so that plot.py needs no changes:
    datetime, month, hour, solar_altitude_deg, solar_azimuth_deg,
    panel_tilt_deg, panel_azimuth_deg, irradiance_wm2, power_w, energy_wh
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import time

import pvgis_client

# PVGIS returns one row per hour
_DT_STEP_H = 1.0


# ---------------------------------------------------------------------------
# System descriptors
# ---------------------------------------------------------------------------

class SystemConfig:
    """Lightweight descriptor passed to run_simulation()."""
    def __init__(
        self,
        name: str,
        trackingtype: int,
        angle: float = 0.0,
        aspect: float = 0.0,
        bidirectional: bool = False,
    ):
        self.name = name
        self.trackingtype = trackingtype
        self.angle  = angle   # tilt from horizontal (fixed) or axis tilt (SAT)
        self.aspect = aspect  # azimuth from South: 0=S, -90=E, +90=W
        self.bidirectional = bidirectional

def tractor_system(tilt_deg: float, forward_azimuth_deg_from_north: float = 90.0) -> SystemConfig:
    """
    Vehicle driving back and forth (e.g., East/West rows).
    Averages irradiance from forward and reverse directions.
    """
    aspect = forward_azimuth_deg_from_north - 180.0
    return SystemConfig("Tractor", trackingtype=0, angle=tilt_deg, aspect=aspect, bidirectional=True)

def fixed_system(tilt_deg: float, azimuth_deg_from_north: float = 180.0) -> SystemConfig:
    """
    Return a SystemConfig for a fixed panel.

    azimuth_deg_from_north : compass direction the panel faces
        (0=North, 90=East, 180=South, 270=West)
    PVGIS aspect            : 0=South, -90=East, +90=West
    Conversion              : aspect = azimuth_from_north - 180
    """
    aspect = azimuth_deg_from_north - 180.0
    return SystemConfig("Fixed", trackingtype=0, angle=tilt_deg, aspect=aspect)


def single_axis_system(axis_tilt_deg: float = 0.0) -> SystemConfig:
    """
    Single horizontal N-S axis tracker (PVGIS trackingtype=1).

    axis_tilt_deg is the fixed inclination of the tracking axis itself
    (0 = flat horizontal axis — the common utility-scale setup).
    The tracker rotates around this axis to minimise the angle of incidence
    throughout the day.
    """
    ttype = 1 if axis_tilt_deg > 0.0 else 1
    return SystemConfig("Single-Axis", trackingtype=ttype, angle=axis_tilt_deg, aspect=0.0)


def dual_axis_system() -> SystemConfig:
    """Two-axis tracker: always points directly at the sun."""
    return SystemConfig("Dual-Axis", trackingtype=2, angle=0.0, aspect=0.0)


def single_axis_ew_system() -> SystemConfig:
    """Single horizontal E-W axis tracker (PVGIS trackingtype=4)."""
    return SystemConfig("Single-Axis EW", trackingtype=4, angle=0.0, aspect=0.0)




# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(
        system: SystemConfig,
        *,
        months: Sequence[int],
        start_hour: float,
        end_hour: float,
        latitude: float,
        longitude: float,
        panel_area_m2: float = 1.0,
        efficiency: float = 0.20,
        year: int = pvgis_client.DEFAULT_YEAR,
        raddatabase: str = pvgis_client.DEFAULT_DB,
        use_cache: bool = True,
) -> pd.DataFrame:
    """
    Simulate energy production for one system over the requested months/hours.
    """
    # 1. Fetch the primary (forward) direction from PVGIS
    raw = pvgis_client.fetch_hourly(
        latitude, longitude,
        trackingtype=system.trackingtype,
        angle=system.angle,
        aspect=system.aspect,
        year=year,
        raddatabase=raddatabase,
        use_cache=use_cache,
    )

    # 1b. If it's a vehicle driving back and forth, fetch the reverse and average
    if getattr(system, "bidirectional", False):
        reverse_aspect = system.aspect + 180.0
        if reverse_aspect > 180.0:
            reverse_aspect -= 360.0  # Keep within typical API bounds

        raw_rev = pvgis_client.fetch_hourly(
            latitude, longitude,
            trackingtype=system.trackingtype,
            angle=system.angle,
            aspect=reverse_aspect,
            year=year,
            raddatabase=raddatabase,
            use_cache=use_cache,
        )

        # Average the in-plane irradiance of both directions
        raw = raw.copy()  # Make a copy to avoid SettingWithCopyWarning
        raw["G_i"] = (raw["G_i"] + raw_rev["G_i"]) / 2.0

    # 2. Filter to requested months and hour window
    df = raw.copy()
    df["month"] = df.index.month
    df["hour"] = df.index.hour + df.index.minute / 60.0

    mask = (
            df["month"].isin(months) &
            (df["hour"] >= start_hour) &
            (df["hour"] <= end_hour)
    )
    df = df[mask].copy()

    # 3. Compute power / energy
    df["irradiance_wm2"] = df["G_i"].clip(lower=0.0)
    df["power_w"] = df["irradiance_wm2"] * panel_area_m2 * efficiency
    df["energy_wh"] = df["power_w"] * _DT_STEP_H

    # Populate legacy columns
    df["datetime"] = df.index
    df["solar_altitude_deg"] = df["H_sun"]
    df["solar_azimuth_deg"] = np.nan
    df["panel_tilt_deg"] = system.angle
    df["panel_azimuth_deg"] = system.aspect + 180.0

    cols = [
        "datetime", "month", "hour",
        "solar_altitude_deg", "solar_azimuth_deg",
        "panel_tilt_deg", "panel_azimuth_deg",
        "irradiance_wm2", "power_w", "energy_wh",
    ]
    return df[cols].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Tilt sweep
# ---------------------------------------------------------------------------

def tilt_sweep(
    *,
    system_factory = None,
    months: Sequence[int],
    start_hour: float,
    end_hour: float,
    latitude: float,
    longitude: float,
    panel_area_m2: float = 1.0,
    efficiency: float = 0.20,
    tilt_min: float = 0.0,
    tilt_max: float = 90.0,
    tilt_step: float = 5.0,
    year: int = pvgis_client.DEFAULT_YEAR,
    raddatabase: str = pvgis_client.DEFAULT_DB,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Sweep south-facing fixed tilt angles and return total kWh for each.

    Each tilt is a separate PVGIS request (disk-cached after first run).
    Returns a DataFrame with columns: tilt_deg, total_kwh.
    """
    tilts = np.arange(tilt_min, tilt_max + tilt_step / 2, tilt_step)
    records = []

    for tilt in tilts:
        if system_factory:
            cfg = system_factory(float(tilt))
        else:
            cfg = fixed_system(tilt_deg=float(tilt), azimuth_deg_from_north=180.0)
        sim = run_simulation(
            cfg,
            months=months,
            start_hour=start_hour,
            end_hour=end_hour,
            latitude=latitude,
            longitude=longitude,
            panel_area_m2=panel_area_m2,
            efficiency=efficiency,
            year=year,
            raddatabase=raddatabase,
            use_cache=use_cache,
        )
        total_kwh = sim["energy_wh"].sum() / 1000.0
        records.append({"tilt_deg": float(tilt), "total_kwh": total_kwh})
        print(f"    tilt {tilt:5.1f}°  →  {total_kwh:.4f} kWh")
        time.sleep(1.0)

    return pd.DataFrame(records)