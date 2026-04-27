"""
solar_position.py
Calculates sun altitude and azimuth for a given datetime and location.
Uses standard astronomical formulas (no external dependencies).
"""

import numpy as np
from datetime import datetime, timezone


def day_of_year(dt: datetime) -> int:
    return dt.timetuple().tm_yday


def equation_of_time(doy: int) -> float:
    """Returns equation of time in minutes."""
    B = np.radians(360 / 365 * (doy - 81))
    return 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)


def solar_declination(doy: int) -> float:
    """Returns solar declination in radians."""
    return np.arcsin(np.sin(np.radians(23.45)) * np.sin(np.radians(360 / 365 * (doy - 81))))


def solar_hour_angle(dt: datetime, longitude: float) -> float:
    """
    Returns the solar hour angle in radians.
    dt must be in UTC.
    """
    doy = day_of_year(dt)
    eot = equation_of_time(doy)  # minutes
    # Solar time in minutes from midnight
    utc_minutes = dt.hour * 60 + dt.minute + dt.second / 60
    solar_time = utc_minutes + 4 * longitude + eot  # 4 min per degree
    # Hour angle: 0 at solar noon, negative in morning
    hour_angle_deg = (solar_time / 60 - 12) * 15
    return np.radians(hour_angle_deg)


def sun_position(dt: datetime, latitude: float, longitude: float) -> tuple[float, float]:
    """
    Returns (altitude, azimuth) in radians for the sun at the given UTC datetime and location.

    altitude: angle above horizon (negative = below horizon)
    azimuth:  measured clockwise from North (0=N, π/2=E, π=S, 3π/2=W)
    """
    lat_r = np.radians(latitude)
    doy = day_of_year(dt)
    dec = solar_declination(doy)
    ha = solar_hour_angle(dt, longitude)

    # Altitude
    sin_alt = (np.sin(lat_r) * np.sin(dec) +
               np.cos(lat_r) * np.cos(dec) * np.cos(ha))
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))

    # Azimuth (from South, positive westward) — convert to from-North clockwise
    cos_az = (np.sin(dec) - np.sin(lat_r) * np.sin(altitude)) / (np.cos(lat_r) * np.cos(altitude) + 1e-9)
    az_from_south = np.arccos(np.clip(cos_az, -1, 1))

    if np.sin(ha) > 0:
        # Afternoon: sun is west of south
        azimuth = np.pi + az_from_south
    else:
        # Morning: sun is east of south
        azimuth = np.pi - az_from_south

    return altitude, azimuth
