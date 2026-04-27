"""
panel_systems.py
Defines how each tracking system orients the panel at any moment.

Each system returns (panel_tilt, panel_azimuth) in radians given sun position.
  panel_tilt    : angle from horizontal (0 = flat, π/2 = vertical)
  panel_azimuth : azimuth the panel faces, clockwise from North
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def angle_of_incidence(solar_altitude: float, solar_azimuth: float,
                       panel_tilt: float, panel_azimuth: float) -> float:
    """
    Calculates the angle between the sun vector and the panel normal.

    All angles in radians.
    Returns AOI in radians (0 = sun perpendicular to panel = maximum power).
    """
    # Sun vector in Cartesian (x=East, y=North, z=Up)
    sx = np.cos(solar_altitude) * np.sin(solar_azimuth)
    sy = np.cos(solar_altitude) * np.cos(solar_azimuth)
    sz = np.sin(solar_altitude)

    # Panel normal vector
    nx = np.sin(panel_tilt) * np.sin(panel_azimuth)
    ny = np.sin(panel_tilt) * np.cos(panel_azimuth)
    nz = np.cos(panel_tilt)

    dot = sx * nx + sy * ny + sz * nz
    aoi = np.arccos(np.clip(dot, -1.0, 1.0))
    return aoi


def irradiance_on_panel(dni: float, dhi: float,
                        solar_altitude: float, solar_azimuth: float,
                        panel_tilt: float, panel_azimuth: float) -> float:
    """
    Total irradiance on the panel surface (W/m²).

    G = DNI * cos(AOI) + DHI * (1 + cos(panel_tilt)) / 2
    The second term is the isotropic sky diffuse model.
    """
    if solar_altitude <= 0:
        return 0.0

    aoi = angle_of_incidence(solar_altitude, solar_azimuth, panel_tilt, panel_azimuth)

    direct = dni * max(np.cos(aoi), 0.0)
    diffuse = dhi * (1 + np.cos(panel_tilt)) / 2  # isotropic sky model

    return direct + diffuse


# ---------------------------------------------------------------------------
# System base class
# ---------------------------------------------------------------------------

class PanelSystem:
    """Base class for all tracking systems."""
    name: str

    def get_orientation(self, solar_altitude: float, solar_azimuth: float,
                        **kwargs) -> tuple[float, float]:
        """Returns (panel_tilt, panel_azimuth) in radians."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Fixed system
# ---------------------------------------------------------------------------

class FixedSystem(PanelSystem):
    """
    Panel is bolted in place: fixed tilt and fixed azimuth.
    Typical install: tilt ≈ latitude, azimuth facing equator (South in NH).
    """
    name = "Fixed"

    def __init__(self, tilt_deg: float, azimuth_deg: float = 180.0):
        """
        Parameters
        ----------
        tilt_deg    : tilt from horizontal in degrees (0=flat, 90=vertical)
        azimuth_deg : direction panel faces, clockwise from North (180=South)
        """
        self.tilt = np.radians(tilt_deg)
        self.azimuth = np.radians(azimuth_deg)

    def get_orientation(self, solar_altitude: float, solar_azimuth: float,
                        **kwargs) -> tuple[float, float]:
        return self.tilt, self.azimuth


# ---------------------------------------------------------------------------
# 2. Single-axis tracking
# ---------------------------------------------------------------------------

class SingleAxisSystem(PanelSystem):
    """
    Rotates around a single horizontal N-S axis.
    The panel tracks the sun's East-West motion (changes tilt throughout the day),
    but the azimuth axis stays fixed (pointing South in NH by default).

    At each moment, the tilt is chosen to minimise the angle of incidence
    while keeping azimuth fixed.
    """
    name = "Single-Axis"

    def __init__(self, azimuth_deg: float = 180.0, max_tilt_deg: float = 60.0):
        """
        Parameters
        ----------
        azimuth_deg  : fixed azimuth the rotation axis faces (degrees from North)
        max_tilt_deg : mechanical tilt limit (degrees)
        """
        self.azimuth = np.radians(azimuth_deg)
        self.max_tilt = np.radians(max_tilt_deg)

    def get_orientation(self, solar_altitude: float, solar_azimuth: float,
                        **kwargs) -> tuple[float, float]:
        if solar_altitude <= 0:
            return 0.0, self.azimuth

        # Optimal tilt: project sun onto the fixed-azimuth plane
        # The ideal tilt minimises AOI for a fixed azimuth
        delta_az = solar_azimuth - self.azimuth
        optimal_tilt = np.arctan2(np.cos(solar_altitude) * np.cos(delta_az),
                                  np.sin(solar_altitude))
        optimal_tilt = np.clip(optimal_tilt, 0, self.max_tilt)
        return optimal_tilt, self.azimuth


# ---------------------------------------------------------------------------
# 3. Dual-axis tracking
# ---------------------------------------------------------------------------

class DualAxisSystem(PanelSystem):
    """
    Tracks the sun on both axes: always points directly at the sun.
    This is the theoretical maximum — the panel normal always aligns with
    the sun vector, giving AOI = 0 whenever the sun is above the horizon.
    """
    name = "Dual-Axis"

    def get_orientation(self, solar_altitude: float, solar_azimuth: float,
                        **kwargs) -> tuple[float, float]:
        if solar_altitude <= 0:
            return 0.0, np.pi  # park flat facing south

        # Tilt = complement of altitude (so normal points at sun)
        panel_tilt = np.pi / 2 - solar_altitude
        panel_azimuth = solar_azimuth  # face directly toward sun
        return panel_tilt, panel_azimuth
