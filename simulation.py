"""
simulation.py
Time-step integration engine. Runs over selected months/hours and accumulates
kWh per system using minute-level resolution.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Sequence

from solar_position import sun_position
from irradiance import IrradianceProvider
from panel_systems import PanelSystem, irradiance_on_panel


# Simulation year — arbitrary non-leap year used for geometry
SIM_YEAR = 2023
TIME_STEP_MINUTES = 1


def build_timesteps(months: Sequence[int],
                    start_hour: float,
                    end_hour: float) -> list[datetime]:
    """
    Build a list of UTC datetime objects covering every minute of every day
    in the requested months between start_hour and end_hour (UTC).
    """
    steps: list[datetime] = []
    step = timedelta(minutes=TIME_STEP_MINUTES)

    for month in months:
        # Number of days in month for SIM_YEAR
        if month == 12:
            last_day = (datetime(SIM_YEAR + 1, 1, 1) - timedelta(days=1)).day
        else:
            last_day = (datetime(SIM_YEAR, month + 1, 1) - timedelta(days=1)).day

        for day in range(1, last_day + 1):
            start_dt = datetime(SIM_YEAR, month, day,
                                int(start_hour),
                                int((start_hour % 1) * 60),
                                tzinfo=timezone.utc)
            end_dt = datetime(SIM_YEAR, month, day,
                              int(end_hour),
                              int((end_hour % 1) * 60),
                              tzinfo=timezone.utc)
            dt = start_dt
            while dt <= end_dt:
                steps.append(dt)
                dt += step

    return steps


def run_simulation(system: PanelSystem,
                   provider: IrradianceProvider,
                   months: Sequence[int],
                   start_hour: float,
                   end_hour: float,
                   latitude: float,
                   longitude: float,
                   panel_area_m2: float = 1.0,
                   efficiency: float = 0.20) -> pd.DataFrame:
    """
    Run the time-step simulation for one system.

    Returns a DataFrame with columns:
        datetime, month, hour, solar_altitude_deg, solar_azimuth_deg,
        panel_tilt_deg, panel_azimuth_deg, irradiance_wm2, power_w, energy_wh
    """
    timesteps = build_timesteps(months, start_hour, end_hour)
    dt_step_h = TIME_STEP_MINUTES / 60.0  # hours per step → for Wh

    records = []
    for dt in timesteps:
        alt, az = sun_position(dt, latitude, longitude)

        if alt <= 0:
            records.append({
                "datetime": dt,
                "month": dt.month,
                "hour": dt.hour + dt.minute / 60,
                "solar_altitude_deg": np.degrees(alt),
                "solar_azimuth_deg": np.degrees(az),
                "panel_tilt_deg": 0.0,
                "panel_azimuth_deg": 0.0,
                "irradiance_wm2": 0.0,
                "power_w": 0.0,
                "energy_wh": 0.0,
            })
            continue

        irr = provider.get_irradiance(dt, latitude, longitude, alt)
        p_tilt, p_az = system.get_orientation(alt, az)

        g_panel = irradiance_on_panel(
            irr.dni, irr.dhi, alt, az, p_tilt, p_az
        )
        power_w = g_panel * panel_area_m2 * efficiency
        energy_wh = power_w * dt_step_h

        records.append({
            "datetime": dt,
            "month": dt.month,
            "hour": dt.hour + dt.minute / 60,
            "solar_altitude_deg": np.degrees(alt),
            "solar_azimuth_deg": np.degrees(az),
            "panel_tilt_deg": np.degrees(p_tilt),
            "panel_azimuth_deg": np.degrees(p_az),
            "irradiance_wm2": g_panel,
            "power_w": power_w,
            "energy_wh": energy_wh,
        })

    return pd.DataFrame(records)


def tilt_sweep(provider: IrradianceProvider,
               months: Sequence[int],
               start_hour: float,
               end_hour: float,
               latitude: float,
               longitude: float,
               panel_area_m2: float = 1.0,
               efficiency: float = 0.20,
               tilt_min: float = 0.0,
               tilt_max: float = 90.0,
               tilt_step: float = 5.0) -> pd.DataFrame:
    """
    Efficiently sweep fixed tilt angles (south-facing) by pre-computing sun
    positions and irradiance once, then evaluating AOI for each tilt.
    Returns a DataFrame: tilt_deg, total_kwh
    """
    from panel_systems import irradiance_on_panel

    tilts = np.arange(tilt_min, tilt_max + tilt_step, tilt_step)
    timesteps = build_timesteps(months, start_hour, end_hour)
    dt_step_h = TIME_STEP_MINUTES / 60.0
    az_fixed = np.radians(180.0)  # south-facing

    # Pre-compute sun position and irradiance for every timestep
    altitudes, solar_azs, dnis, dhis = [], [], [], []
    for dt in timesteps:
        alt, az = sun_position(dt, latitude, longitude)
        irr = provider.get_irradiance(dt, latitude, longitude, alt)
        altitudes.append(alt)
        solar_azs.append(az)
        dnis.append(irr.dni)
        dhis.append(irr.dhi)

    altitudes = np.array(altitudes)
    solar_azs = np.array(solar_azs)
    dnis      = np.array(dnis)
    dhis      = np.array(dhis)
    above     = altitudes > 0

    results = []
    for tilt_deg in tilts:
        tilt_r = np.radians(tilt_deg)
        # Vectorised AOI calculation for this tilt
        sx = np.cos(altitudes) * np.sin(solar_azs)
        sy = np.cos(altitudes) * np.cos(solar_azs)
        sz = np.sin(altitudes)
        nx = np.sin(tilt_r) * np.sin(az_fixed)
        ny = np.sin(tilt_r) * np.cos(az_fixed)
        nz = np.cos(tilt_r)
        dot = sx * nx + sy * ny + sz * nz
        aoi = np.arccos(np.clip(dot, -1.0, 1.0))

        direct  = dnis * np.maximum(np.cos(aoi), 0.0)
        diffuse = dhis * (1 + np.cos(tilt_r)) / 2
        g_panel = np.where(above, direct + diffuse, 0.0)
        total_kwh = (g_panel * panel_area_m2 * efficiency * dt_step_h).sum() / 1000
        results.append({"tilt_deg": tilt_deg, "total_kwh": total_kwh})

    return pd.DataFrame(results)
