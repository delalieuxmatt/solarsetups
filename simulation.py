"""
simulation.py
=============
Time-series simulation engine — PVGIS edition.

All irradiance values come directly from PVGIS G(i) (global in-plane
irradiance on the panel surface).  We never decompose into DNI/DHI or
compute angles of incidence locally — EXCEPT for the flat N/S vehicle panel,
where an obstacle casts a shadow that reduces the beam (Gb_i) component on
the flat surface.  The diffuse component (Gd_i) is assumed isotropic and
unaffected by the obstacle shadow.

PVGIS tracking types used here:
    0  fixed plane    → requires angle (tilt) + aspect (azimuth from South)
    1  single-axis N-S horizontal tracker  → requires angle (axis tilt)
    2  dual-axis tracker
    3  vertical-axis tracker
    4  single-axis E-W horizontal tracker
    5  single inclined axis, N-S aligned

Vehicle driving direction
-------------------------
ALL vehicle panels in this model drive N↔S.  This means:
  • For a FLAT panel  (tilt=0°):  PVGIS returns the same G(i) regardless of
    N-vs-S heading, so there is only one PVGIS fetch and no bidirectional
    averaging needed.  Shading correction is applied here instead.
  • For a TILTED/VERTICAL panel mounted on the side or rear of the vehicle
    (tilt > 0°, e.g. 90° for a vertical side panel): the panel faces either
    East or West depending on which side of the vehicle, so we average the
    forward and reverse PVGIS fetches as before.

ShadeParams dataclass
---------------------
Carries the obstacle geometry for the flat-panel shading model.
Set shade_params on a SystemConfig to enable the correction.  Only valid
when the panel is flat (tilt ≈ 0°).

DataFrame schema returned by run_simulation() — unchanged:
    datetime, month, hour, solar_altitude_deg, solar_azimuth_deg,
    panel_tilt_deg, panel_azimuth_deg, irradiance_wm2, power_w, energy_wh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import time

import pvgis_client
import irradiance as irr_geom   # shading geometry module

# PVGIS returns one row per hour
_DT_STEP_H = 1.0


# ---------------------------------------------------------------------------
# Obstacle / shade parameters
# ---------------------------------------------------------------------------

@dataclass
class ShadeParams:
    """
    Geometry of the flap obstacle and the two flat panels it shades.

    All dimensions in metres.

    Obstacle
    --------
    H_flap  : height of the flap above the panel surface
    L_flap  : depth of the flap along N-S (driving direction)
    d_flap  : width of the flap along E-W  — also the shadow width

    Front panel  (faces the direction of travel)
    ------------
    L_front : N-S depth of the front panel
    d_front : E-W width of the front panel
              For the combine harvester d_front == d_flap (no side overhang)

    Back panel   (behind the flap relative to direction of travel)
    ----------
    L_back  : N-S depth of the back panel
    d_back  : E-W width of the back panel
              d_back > d_flap → side strips of (d_back−d_flap)/2 on each
              side always receive full beam regardless of shadow length

    two_panel : if True, use two_panel_shaded_fraction() which handles
                both driving directions and the two different panel sizes.
                If False, fall back to the legacy single-panel shaded_fraction()
                using only d_flap / L_panel for backward compatibility.
    """
    H_flap:    float = 0.6
    L_flap:    float = 0.6
    d_flap:    float = 2.04

    # Two-panel geometry
    L_front:   float = 1.5          # example; set in scenarios.py
    d_front:   float = 2.04         # == d_flap for the combine harvester
    L_back:    float = 1.5          # example; set in scenarios.py
    d_back:    float = 2.66         # wider than d_flap → side overhang

    two_panel: bool  = True

    # Legacy single-panel fields (used only when two_panel=False)
    L_panel:   float | None = None


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
        shade_params: Optional[ShadeParams] = None,
    ):
        self.name          = name
        self.trackingtype  = trackingtype
        self.angle         = angle          # tilt from horizontal (fixed) or axis tilt (SAT)
        self.aspect        = aspect         # azimuth from South: 0=S, -90=E, +90=W
        self.bidirectional = bidirectional
        self.shade_params  = shade_params   # non-None → flat panel with obstacle shading
        # Optional override for total physical panel area (m²).
        # Used by flat_ns_vehicle_system(shade_params=None) to ensure the
        # unshaded baseline uses the same area as the shaded system.
        self.total_panel_area_m2: float | None = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def fixed_system(tilt_deg: float, azimuth_deg_from_north: float = 180.0) -> SystemConfig:
    """
    Fixed panel, south-facing by default.

    azimuth_deg_from_north : compass direction the panel faces
        (0=North, 90=East, 180=South, 270=West)
    PVGIS aspect convention : 0=South, -90=East, +90=West
    Conversion              : aspect = azimuth_from_north - 180
    """
    aspect = azimuth_deg_from_north - 180.0
    return SystemConfig("Fixed", trackingtype=0, angle=tilt_deg, aspect=aspect)


def single_axis_system(axis_tilt_deg: float = 0.0) -> SystemConfig:
    """
    Single horizontal N-S axis tracker (PVGIS trackingtype=1).

    axis_tilt_deg is the fixed inclination of the tracking axis itself
    (0 = flat horizontal axis — the common utility-scale setup).
    """
    return SystemConfig("Single-Axis", trackingtype=1, angle=axis_tilt_deg, aspect=0.0)


def dual_axis_system() -> SystemConfig:
    """Two-axis tracker: always points directly at the sun."""
    return SystemConfig("Dual-Axis", trackingtype=2, angle=0.0, aspect=0.0)


def single_axis_ew_system() -> SystemConfig:
    """Single horizontal E-W axis tracker (PVGIS trackingtype=4)."""
    return SystemConfig("Single-Axis EW", trackingtype=4, angle=0.0, aspect=0.0)


def flat_ns_vehicle_system(
    shade_params: Optional[ShadeParams] = None,
    name: str = "NS Flat (shaded)",
    total_panel_area_m2: float | None = None,
) -> SystemConfig:
    """
    Flat panel (tilt=0°) on a vehicle driving N↔S.

    A flat panel has no preferred azimuth — PVGIS returns the same G(i)
    for any aspect when angle=0, so no bidirectional averaging is needed.

    If shade_params is provided the beam component is corrected at each
    timestep for the shadow cast by the obstacle.  Pass shade_params=None
    to get the unshaded flat-panel baseline.

    total_panel_area_m2 : total physical panel area (m²) used when
        shade_params is None.  When shade_params is provided the area is
        derived from the ShadeParams geometry (L_front*d_front + L_back*d_back).
        If None and shade_params is None, falls back to the panel_area_m2
        passed to run_simulation() — this is usually wrong for the two-panel
        geometry, so always pass this explicitly for the unshaded baseline.
    """
    cfg = SystemConfig(
        name,
        trackingtype=0,
        angle=0.0,
        aspect=0.0,          # irrelevant for a horizontal panel
        bidirectional=False,
        shade_params=shade_params,
    )
    cfg.total_panel_area_m2 = total_panel_area_m2
    return cfg


def tilted_ns_vehicle_system(
    tilt_deg: float,
    forward_azimuth_deg_from_north: float,
    name: str | None = None,
) -> SystemConfig:
    """
    Tilted or vertical panel on a vehicle driving N↔S.

    Examples:
      • Rear panel  (faces backward along driving dir):
            forward_azimuth=0 → facing South when going North → aspect=−180 → +180 wrap
      • Side panel  (perpendicular to driving dir):
            forward_azimuth=90 → facing East → aspect=−90

    The vehicle drives both ways, so we average the forward and reverse
    PVGIS fetches (bidirectional=True).
    """
    aspect = forward_azimuth_deg_from_north - 180.0
    if name is None:
        name = f"NS Tilted {tilt_deg:.0f}°"
    return SystemConfig(
        name,
        trackingtype=0,
        angle=tilt_deg,
        aspect=aspect,
        bidirectional=True,
        shade_params=None,   # shading model only applies to flat panels
    )


def tractor_system(tilt_deg: float, forward_azimuth_deg_from_north: float = 90.0) -> SystemConfig:
    """
    Legacy helper: vehicle driving back and forth.
    Averages irradiance from forward and reverse directions.
    Retained for backward compatibility with scenarios that still use it.
    """
    aspect = forward_azimuth_deg_from_north - 180.0
    return SystemConfig("Tractor", trackingtype=0, angle=tilt_deg, aspect=aspect, bidirectional=True)


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
        panel_area_m2: float = 13.6,
        efficiency: float = 0.20,
        year: int = pvgis_client.DEFAULT_YEAR,
        raddatabase: str = pvgis_client.DEFAULT_DB,
        use_cache: bool = True,
) -> pd.DataFrame:
    """
    Simulate energy production for one system over the requested months/hours.

    For a flat N/S vehicle panel with ShadeParams, the beam irradiance (Gb_i)
    is reduced by the obstacle shadow fraction at each timestep:

        G_effective = Gb_i × (1 − f_shade) + Gd_i

    For all other systems the PVGIS G_i value is used directly (unchanged
    behaviour).
    """
    # ------------------------------------------------------------------
    # 1. Fetch primary PVGIS time-series
    # ------------------------------------------------------------------
    raw = pvgis_client.fetch_hourly(
        latitude, longitude,
        trackingtype=system.trackingtype,
        angle=system.angle,
        aspect=system.aspect,
        year=year,
        raddatabase=raddatabase,
        use_cache=use_cache,
    )

    # ------------------------------------------------------------------
    # 1b. Bidirectional panels: average forward + reverse irradiance
    #     (only for tilted/vertical vehicle panels, NOT flat ones)
    # ------------------------------------------------------------------
    if system.bidirectional:
        reverse_aspect = system.aspect + 180.0
        if reverse_aspect > 180.0:
            reverse_aspect -= 360.0

        raw_rev = pvgis_client.fetch_hourly(
            latitude, longitude,
            trackingtype=system.trackingtype,
            angle=system.angle,
            aspect=reverse_aspect,
            year=year,
            raddatabase=raddatabase,
            use_cache=use_cache,
        )
        raw = raw.copy()
        raw["G_i"]  = (raw["G_i"]  + raw_rev["G_i"])  / 2.0
        raw["Gb_i"] = (raw["Gb_i"] + raw_rev["Gb_i"]) / 2.0
        raw["Gd_i"] = (raw["Gd_i"] + raw_rev["Gd_i"]) / 2.0
        raw["Gr_i"] = (raw["Gr_i"] + raw_rev["Gr_i"]) / 2.0

    # ------------------------------------------------------------------
    # 2. Filter to requested months and hour window
    # ------------------------------------------------------------------
    df = raw.copy()
    df["month"] = df.index.month
    df["hour"]  = df.index.hour + df.index.minute / 60.0

    mask = (
        df["month"].isin(months)
        & (df["hour"] >= start_hour)
        & (df["hour"] <= end_hour)
    )
    df = df[mask].copy()

    # ------------------------------------------------------------------
    # 3. Shading correction for flat N/S vehicle panel
    # ------------------------------------------------------------------
    if system.shade_params is not None:
        sp = system.shade_params

        df = irr_geom.add_solar_position(df, lat_deg=latitude, lon_deg=longitude)

        if sp.two_panel:
            # Two-panel combine-harvester geometry: front + back panels,
            # averaged over both driving directions.
            f_shade = irr_geom.two_panel_shaded_fraction(
                alpha_rad = df["alpha_rad"].to_numpy(),
                theta_rad = df["theta_rad"].to_numpy(),
                H_flap    = sp.H_flap,
                L_flap    = sp.L_flap,
                d_flap    = sp.d_flap,
                L_front   = sp.L_front,
                d_front   = sp.d_front,
                L_back    = sp.L_back,
                d_back    = sp.d_back,
            )
            # A_total for energy calculation is the combined panel area
            effective_area = sp.L_front * sp.d_front + sp.L_back * sp.d_back
        else:
            # Legacy single-panel path
            f_shade = irr_geom.shaded_fraction(
                alpha_rad = df["alpha_rad"].to_numpy(),
                theta_rad = df["theta_rad"].to_numpy(),
                H_flap    = sp.H_flap,
                L_flap    = sp.L_flap,
                d_flap    = sp.d_flap,
                A_total   = panel_area_m2,
                L_panel   = sp.L_panel,
            )
            effective_area = panel_area_m2

        # Effective irradiance: beam reduced by shade, diffuse unaffected
        Gb = df["Gb_i"].clip(lower=0.0).to_numpy()
        Gd = df["Gd_i"].clip(lower=0.0).to_numpy()
        Gr = df["Gr_i"].clip(lower=0.0).to_numpy()

        G_effective = Gb * (1.0 - f_shade) + Gd + Gr

        df["irradiance_wm2"] = np.clip(G_effective, 0.0, None)
        df["shade_fraction"] = f_shade

        # Override panel_area_m2 for power calculation with actual panel area
        _area_for_power = effective_area

    else:
        # Standard path: use PVGIS G_i directly.
        # If the system carries an explicit total_panel_area_m2 (set by
        # flat_ns_vehicle_system for the unshaded baseline), use that so
        # the unshaded and shaded systems are compared on equal footing.
        df["irradiance_wm2"] = df["G_i"].clip(lower=0.0)
        df["shade_fraction"] = 0.0
        _area_for_power = (
            system.total_panel_area_m2
            if getattr(system, "total_panel_area_m2", None) is not None
            else panel_area_m2
        )

    # ------------------------------------------------------------------
    # 4. Power and energy
    # ------------------------------------------------------------------
    df["power_w"]   = df["irradiance_wm2"] * _area_for_power * efficiency
    df["energy_wh"] = df["power_w"] * _DT_STEP_H

    # ------------------------------------------------------------------
    # 5. Legacy / output columns
    # ------------------------------------------------------------------
    df["datetime"]         = df.index
    df["solar_altitude_deg"] = df["H_sun"]
    df["solar_azimuth_deg"]  = np.nan
    df["panel_tilt_deg"]     = system.angle
    df["panel_azimuth_deg"]  = system.aspect + 180.0

    cols = [
        "datetime", "month", "hour",
        "solar_altitude_deg", "solar_azimuth_deg",
        "panel_tilt_deg", "panel_azimuth_deg",
        "irradiance_wm2", "power_w", "energy_wh",
    ]
    return df[cols].reset_index(drop=True)


def run_multi_year_simulation(
        system: SystemConfig,
        *,
        years: Sequence[int],
        degradation_factor: float = 0.0,
        **kwargs
) -> pd.DataFrame:
    """
    Simulate energy production over multiple years with an annual degradation factor.
    """
    dfs = []
    # Sort years to ensure degradation is applied chronologically
    sorted_years = sorted(years)

    for i, y in enumerate(sorted_years):
        # Run standard simulation for the specific year
        df = run_simulation(system, year=y, **kwargs)

        # Apply degradation factor relative to the first year: (1 - degradation_factor)^i
        current_degradation = (1.0 - degradation_factor) ** i
        df["power_w"] = df["power_w"] * current_degradation
        df["energy_wh"] = df["energy_wh"] * current_degradation

        # Tag the DataFrame with the simulated year
        df["sim_year"] = y
        dfs.append(df)

    # Concatenate all years into a single DataFrame
    return pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------------------------
# Tilt sweep
# ---------------------------------------------------------------------------

def tilt_sweep(
    *,
    system_factory=None,
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
    Returns a DataFrame with columns: tilt_deg, total_kwh.
    """
    tilts   = np.arange(tilt_min, tilt_max + tilt_step / 2, tilt_step)
    records = []

    for tilt in tilts:
        cfg = system_factory(float(tilt)) if system_factory else \
              fixed_system(tilt_deg=float(tilt), azimuth_deg_from_north=180.0)
        sim = run_simulation(
            cfg,
            months=months, start_hour=start_hour, end_hour=end_hour,
            latitude=latitude, longitude=longitude,
            panel_area_m2=panel_area_m2, efficiency=efficiency,
            year=year, raddatabase=raddatabase, use_cache=use_cache,
        )
        total_kwh = sim["energy_wh"].sum() / 1000.0
        records.append({"tilt_deg": float(tilt), "total_kwh": total_kwh})
        print(f"    tilt {tilt:5.1f}°  →  {total_kwh:.4f} kWh")
        time.sleep(1.0)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Latitude sweep
# ---------------------------------------------------------------------------

def latitude_sweep(
    *,
    months: Sequence[int],
    start_hour: float,
    end_hour: float,
    longitude: float,
    panel_area_m2: float = 1.0,
    efficiency: float = 0.20,
    lat_min: float = 0.0,
    lat_max: float = 65.0,
    lat_step: float = 5.0,
    year: int = pvgis_client.DEFAULT_YEAR,
    raddatabase: str = pvgis_client.DEFAULT_DB,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Sweep latitudes and simulate all tracking systems at each latitude.
    Returns a DataFrame with columns: latitude, system, total_kwh.
    """
    from scenarios import single_axis_comparison   # local import — avoids circular dep

    latitudes = np.arange(lat_min, lat_max + lat_step / 2, lat_step)
    records: list[dict] = []

    common_sim = dict(
        months=months, start_hour=start_hour, end_hour=end_hour,
        longitude=longitude, panel_area_m2=panel_area_m2, efficiency=efficiency,
        year=year, raddatabase=raddatabase, use_cache=use_cache,
    )

    for lat in latitudes:
        lat     = float(lat)
        systems = single_axis_comparison(latitude=lat)
        print(f"  lat {lat:+6.1f}°", end="", flush=True)

        for sys_name, cfg in systems.items():
            sim       = run_simulation(cfg, latitude=lat, **common_sim)
            total_kwh = sim["energy_wh"].sum() / 1000.0
            records.append({"latitude": lat, "system": sys_name, "total_kwh": total_kwh})
            print(f"  {sys_name}={total_kwh:.2f}", end="", flush=True)
            time.sleep(0.3)

        print()

    return pd.DataFrame(records)