"""
irradiance.py
=============
Pure-function shading geometry for a flat panel mounted on a vehicle
driving N↔S, with a rectangular obstacle (the "flap") rising above the
panel surface.

Physical setup
--------------
The panel lies flat (tilt = 0°).  The vehicle drives North–South, so the
obstacle is aligned E-W (its d_flap dimension runs East-West and its
L_flap dimension runs North-South along the driving direction).

Obstacle dimensions
-------------------
  H_flap  (m) — height of the obstacle above the panel surface
  L_flap  (m) — depth of the obstacle along the N-S driving direction
  d_flap  (m) — width of the obstacle along the E-W direction

The obstacle casts a shadow whose N-S projection onto the panel depends on
the solar altitude angle α (sun higher → shorter shadow).

Coordinate convention
---------------------
  α   — solar altitude angle (radians), 0 at horizon, π/2 at zenith
  θ   — solar azimuth offset from due South (radians)
        negative = sun is east of south, positive = sun is west of south
        used only for the triangular corner-relief term A_unshaded.

All public functions operate on numpy arrays so that an entire hourly
time-series can be processed in one vectorised call.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Solar position helpers
# ---------------------------------------------------------------------------

def solar_declination(day_of_year: np.ndarray | float) -> np.ndarray | float:
    """
    Solar declination δ in radians using the Spencer (1971) approximation.

    δ = 23.45° · sin(360/365 · (284 + doy))   [converted to radians]

    Parameters
    ----------
    day_of_year : 1-based integer day of year (1 = Jan 1, 365/366 = Dec 31)
    """
    B = np.radians(360.0 / 365.0 * (284.0 + day_of_year))
    return np.radians(23.45) * np.sin(B)


def equation_of_time_minutes(day_of_year: np.ndarray | float) -> np.ndarray | float:
    """
    Equation of Time in minutes (Spencer 1971).
    Corrects for the eccentricity of Earth's orbit and axial tilt.
    """
    B = 2.0 * np.pi * (day_of_year - 1) / 365.0
    return (
        229.18 * (
            0.000075
            + 0.001868 * np.cos(B)
            - 0.032077 * np.sin(B)
            - 0.014615 * np.cos(2 * B)
            - 0.04089  * np.sin(2 * B)
        )
    )


def hour_angle(
    hour_utc: np.ndarray | float,
    longitude_deg: float,
    day_of_year: np.ndarray | float,
) -> np.ndarray | float:
    """
    Solar hour angle ω in radians.

    Convention: negative before solar noon, positive after.

    Parameters
    ----------
    hour_utc      : UTC decimal hour (e.g. 13.5 = 13:30 UTC)
    longitude_deg : site longitude in degrees East (negative = West)
    day_of_year   : 1-based integer day of year
    """
    eot_min   = equation_of_time_minutes(day_of_year)
    # Local Apparent Solar Time in hours
    last_hour = hour_utc + longitude_deg / 15.0 + eot_min / 60.0
    # Hour angle: 15°/h, 0 at solar noon
    omega_deg = 15.0 * (last_hour - 12.0)
    return np.radians(omega_deg)


def solar_altitude_rad(
    lat_deg: float,
    declination_rad: np.ndarray | float,
    hour_angle_rad: np.ndarray | float,
) -> np.ndarray | float:
    """
    Solar altitude angle α in radians.

    sin(α) = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(ω)

    Parameters
    ----------
    lat_deg         : site latitude in degrees
    declination_rad : solar declination δ (radians)
    hour_angle_rad  : solar hour angle ω (radians)
    """
    phi   = np.radians(lat_deg)
    sin_a = (
        np.sin(phi) * np.sin(declination_rad)
        + np.cos(phi) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    # Clamp to [-1, 1] to guard against floating-point out-of-range
    sin_a = np.clip(sin_a, -1.0, 1.0)
    return np.arcsin(sin_a)


def solar_azimuth_from_south_rad(
    lat_deg: float,
    declination_rad: np.ndarray | float,
    hour_angle_rad: np.ndarray | float,
    altitude_rad: np.ndarray | float,
) -> np.ndarray | float:
    """
    Solar azimuth offset from due South in radians.

    θ < 0 : sun is East of South  (morning)
    θ > 0 : sun is West of South  (afternoon)
    |θ| > π/2 : sun is north of East/West line (northern sky for NH observers)

    The standard spherical-trigonometry formula gives azimuth from North:

        cos(Z_N) = [sin(δ) - sin(α)·sin(φ)] / [cos(α)·cos(φ)]

    Due-south corresponds to Z_N = 180°.  We convert:

        θ_from_south = Z_N − 180°

    so θ=0 is due south, θ=±90° is due East/West, |θ|>90° means the sun
    is in the northern sky.  Sign follows the hour angle convention
    (negative = morning = east of south, positive = afternoon = west of south).
    """
    phi     = np.radians(lat_deg)
    cos_alt = np.cos(altitude_rad)

    # Avoid division by zero near zenith
    safe_cos_alt = np.where(np.abs(cos_alt) < 1e-9, 1e-9, cos_alt) \
        if isinstance(cos_alt, np.ndarray) else \
        (1e-9 if abs(cos_alt) < 1e-9 else cos_alt)

    # cos(Z_from_north)
    cos_Z_north = (
        (np.sin(declination_rad) - np.sin(altitude_rad) * np.sin(phi))
        / (safe_cos_alt * np.cos(phi))
    )
    cos_Z_north = np.clip(cos_Z_north, -1.0, 1.0)

    # Z_from_north in [0, π]
    Z_north = np.arccos(cos_Z_north)

    # Convert to offset from South: θ = Z_north − π
    # Range: [−π, 0]  (always west of south after conversion without sign flip)
    # Apply sign from hour angle: negative ω → morning → east of south → θ < 0
    theta_abs = np.abs(Z_north - np.pi)   # magnitude of offset from south

    return np.where(hour_angle_rad < 0, -theta_abs, theta_abs) \
        if isinstance(hour_angle_rad, np.ndarray) else \
        (-theta_abs if hour_angle_rad < 0 else theta_abs)


# ---------------------------------------------------------------------------
# Shading model for a flat panel with an E-W obstacle
# ---------------------------------------------------------------------------

def _shadow_length(
    alpha_rad: np.ndarray,
    H_flap: float,
    L_flap: float,
    L_panel: float,
) -> np.ndarray:
    """
    N-S shadow length cast by the far (top) edge of the flap onto the
    panel surface, clamped to L_panel.

    The flap is a rectangular block of height H_flap (above the panel)
    and depth L_flap (along N-S).  The shadow of its farthest top edge
    (the edge furthest from the panel it is shading) is:

        L_shadow = H_flap / tan(α) + L_flap

    where α is the solar altitude.  The first term is the horizontal
    distance the top-edge shadow travels, and the second term adds the
    physical depth of the flap itself (the shadow starts L_flap away from
    the panel edge).

    This is the general formula valid for any H_flap / L_flap ratio.
    The previous approximation using a hardcoded 135° was only correct
    when H_flap == L_flap.
    """
    tan_a    = np.tan(alpha_rad)
    safe_tan = np.where(tan_a < 1e-6, 1e-6, tan_a)
    L_sh     = H_flap / safe_tan + L_flap
    return np.clip(L_sh, 0.0, L_panel)


def _parallelogram_shaded_area(
    L_shaded: np.ndarray,
    theta_rad: np.ndarray,
    d_shadow: float,
) -> np.ndarray:
    """
    Area of the shadow parallelogram that still overlaps a strip of width
    d_shadow (E-W).

    As the sun moves away from due-South by azimuth angle θ, the shadow
    shears sideways:

        Δy          = L_shaded · |tan(θ)|   (lateral shift of far edge)
        Δy_clipped  = min(Δy, d_shadow)
        A_shaded    = L_shaded · (d_shadow − Δy_clipped)  ≥ 0

    d_shadow is the *shadow-casting width* (= d_flap), NOT the panel width.
    The panel may be wider; that is handled by the caller.
    """
    abs_tan   = np.abs(np.tan(theta_rad))
    delta_y   = L_shaded * abs_tan
    delta_y_c = np.minimum(delta_y, d_shadow)
    return L_shaded * (d_shadow - delta_y_c)          # always ≥ 0


def shaded_fraction(
    alpha_rad: np.ndarray | float,
    theta_rad: np.ndarray | float,
    H_flap: float,
    L_flap: float,
    d_flap: float,
    A_total: float,
    L_panel: float | None = None,
    d_panel: float | None = None,
) -> np.ndarray | float:
    """
    Fraction of a *single* flat panel shaded by the flap obstacle.

    This is the single-panel entry point used by run_simulation() for a
    vehicle system with one panel.  For the two-panel combine-harvester
    geometry use ``two_panel_shaded_fraction`` instead.

    The shadow width is capped at d_flap (the obstacle width).  If the
    panel is wider than d_flap (d_panel > d_flap), the side strips always
    receive full beam; the shaded area is still only d_flap wide.

    Parameters
    ----------
    alpha_rad : solar altitude in radians  (scalar or array)
    theta_rad : solar azimuth from South   (scalar or array, signed)
    H_flap    : obstacle height above panel (m)
    L_flap    : obstacle depth along N-S   (m)
    d_flap    : obstacle (shadow) width along E-W (m)
    A_total   : total flat panel area (m²) — used only for the fraction
    L_panel   : panel N-S length (m); defaults to √(A_total)
    d_panel   : panel E-W width  (m); informational only (unused in calc,
                since the shadow is already bounded by d_flap)

    Returns
    -------
    f_shade   : shaded fraction ∈ [0, 1]
    """
    alpha = np.asarray(alpha_rad, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)

    if L_panel is None:
        L_panel = np.sqrt(A_total)

    below_horizon = alpha <= 0.0
    sun_is_north  = np.abs(theta) > (np.pi / 2.0)

    L_sh     = _shadow_length(alpha, H_flap, L_flap, L_panel)
    A_shaded = _parallelogram_shaded_area(L_sh, theta, d_flap)
    A_shaded = np.clip(A_shaded, 0.0, A_total)

    f_shade = A_shaded / A_total
    f_shade = np.where(sun_is_north,  0.0, f_shade)
    f_shade = np.where(below_horizon, 1.0, f_shade)

    if f_shade.ndim == 0:
        return float(f_shade)
    return f_shade


def two_panel_shaded_fraction(
    alpha_rad: np.ndarray | float,
    theta_rad: np.ndarray | float,
    H_flap: float,
    L_flap: float,
    d_flap: float,
    L_front: float,
    d_front: float,
    L_back: float,
    d_back: float,
) -> np.ndarray | float:
    """
    Combined shaded fraction for a vehicle with TWO flat panels separated
    by a central flap obstacle, driving N↔S equally in both directions.

    Physical layout (vehicle driving North, seen from above)
    ---------------------------------------------------------

        ┌──────────────────────────┐  ← front panel (d_front wide, L_front deep)
        │       FRONT PANEL        │    (the panel at the leading end of travel)
        └──────────────────────────┘
        ╔══════════════════════════╗  ← flap obstacle (d_flap wide, L_flap deep)
        ║         F L A P          ║    (canopy flap between the two panels)
        ╚══════════════════════════╝
        ┌──────────────────────────┐  ← back panel (d_back wide, L_back deep)
        │        BACK PANEL        │    d_back may be > d_flap; the extra
        │  [clear][shaded][clear]  │    (d_back−d_flap)/2 on each side always
        └──────────────────────────┘    receives full beam

    Driving directions and shadow logic
    ------------------------------------
    When driving North the front panel is the LEADING panel.  The flap
    sits above the gap between the panels.  The sun shines from the south
    and the flap blocks it — casting a shadow FORWARD (northward) onto the
    FRONT (leading) panel.  The back panel is fully in the clear because
    the flap does not overhang it in that direction.

    N→S  (driving North):  shadow falls onto the FRONT panel.
                           Back panel is unshaded.
    S→N  (driving South):  the roles swap — the back panel is now leading.
                           Shadow falls onto the BACK panel.
                           Front panel is unshaded.

    Since the vehicle spends equal time in both directions we average the
    two cases.

    E-W overhang of the back panel
    --------------------------------
    d_back > d_flap means the back panel has (d_back − d_flap)/2 of clear
    strip on each side.  The shadow is only d_flap wide, so:

        A_shaded_back = parallelogram_area(L_sh, θ, d_shadow=d_flap)
                        capped at L_back × d_flap   (not L_back × d_back)

    The side strips (total area L_back × (d_back − d_flap)) always have
    A_shaded = 0.

    Returns
    -------
    f_shade : combined shaded fraction of the *total* two-panel area,
              averaged over both driving directions.  ∈ [0, 1].
    """
    alpha = np.asarray(alpha_rad, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)

    A_front = L_front * d_front
    A_back  = L_back  * d_back
    A_total = A_front + A_back

    below_horizon = alpha <= 0.0
    sun_is_north  = np.abs(theta) > (np.pi / 2.0)

    # Shadow length capped to the respective panel's N-S depth
    L_sh_front = _shadow_length(alpha, H_flap, L_flap, L_front)
    L_sh_back  = _shadow_length(alpha, H_flap, L_flap, L_back)

    # --- Case A: driving North (N→S on map) — shadow falls on FRONT panel ---
    # The front panel has width d_front == d_flap (no side overhang).
    A_shaded_front_A = _parallelogram_shaded_area(L_sh_front, theta, d_flap)
    A_shaded_front_A = np.clip(A_shaded_front_A, 0.0, L_front * d_front)
    A_shaded_back_A  = np.zeros_like(alpha)   # back panel fully unshaded

    # --- Case B: driving South (S→N on map) — shadow falls on BACK panel ---
    # Back panel is wider than d_flap; shadow is still only d_flap wide.
    A_shaded_back_B  = _parallelogram_shaded_area(L_sh_back, theta, d_flap)
    A_shaded_back_B  = np.clip(A_shaded_back_B, 0.0, L_back * d_flap)
    A_shaded_front_B = np.zeros_like(alpha)   # front panel fully unshaded

    # Average over both driving directions
    A_shaded_total = 0.5 * (
        (A_shaded_front_A + A_shaded_back_A) +
        (A_shaded_front_B + A_shaded_back_B)
    )

    f_shade = A_shaded_total / A_total
    # Sun in northern sky or below horizon → no beam irradiance reaches panels
    f_shade = np.where(sun_is_north,  0.0, f_shade)
    f_shade = np.where(below_horizon, 1.0, f_shade)  # consistent with shaded_fraction()

    if f_shade.ndim == 0:
        return float(f_shade)
    return f_shade


# ---------------------------------------------------------------------------
# Convenience: compute solar position columns for a filtered PVGIS DataFrame
# ---------------------------------------------------------------------------

def add_solar_position(
    df: "pd.DataFrame",
    lat_deg: float,
    lon_deg: float,
) -> "pd.DataFrame":
    """
    Add columns  ``alpha_rad``  and  ``theta_rad``  to *df* in-place.

    *df* must have a UTC DatetimeIndex (as produced by pvgis_client).

    Parameters
    ----------
    df      : DataFrame with UTC DatetimeIndex
    lat_deg : site latitude  (degrees North)
    lon_deg : site longitude (degrees East)

    Returns
    -------
    The same DataFrame with two new columns added.
    """
    import pandas as pd   # local import so module stays light at import time

    doy       = df.index.day_of_year.to_numpy(dtype=float)
    hour_utc  = (df.index.hour + df.index.minute / 60.0).to_numpy(dtype=float)

    decl  = solar_declination(doy)
    omega = hour_angle(hour_utc, lon_deg, doy)
    alpha = solar_altitude_rad(lat_deg, decl, omega)
    print("Dit is alpha", alpha)
    theta = solar_azimuth_from_south_rad(lat_deg, decl, omega, alpha)

    df = df.copy()
    df["alpha_rad"] = alpha
    df["theta_rad"] = theta
    return df