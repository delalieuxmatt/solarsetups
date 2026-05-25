"""
scenarios.py
============
Scenario definitions for the solar simulation tool.

All vehicle scenarios drive N↔S exclusively.

  • Flat panel   (tilt=0°): a single PVGIS fetch suffices (aspect is
    irrelevant for a horizontal surface).  An obstacle shadow is applied
    via the ShadeParams / shading model in simulation.py + irradiance.py.

  • Tilted/vertical panels (tilt > 0°): the panel faces East or West
    depending on which side of the vehicle, so we average the forward
    (North-going) and reverse (South-going) PVGIS fetches.

Obstacle geometry (default values, edit here to change globally):
    H_FLAP = 0.6 m   — obstacle height above the panel surface
    L_FLAP = 0.6 m   — obstacle depth along the N-S driving direction
    D_FLAP = 2.04 m  — obstacle width along the E-W direction

To disable shading (unshaded flat-panel baseline) pass shade_params=None
to flat_ns_vehicle_system().
"""

from simulation import (
    SystemConfig,
    ShadeParams,
    fixed_system,
    single_axis_system,
    dual_axis_system,
    flat_ns_vehicle_system,
    tilted_ns_vehicle_system,
)


# ---------------------------------------------------------------------------
# Default obstacle + panel geometry
# ---------------------------------------------------------------------------

H_FLAP: float = 0.6    # m — height of flap above panel surface
L_FLAP: float = 0.6    # m — depth of flap along N-S driving direction
D_FLAP: float = 2.04   # m — width of flap along E-W (= shadow width)

# Front panel (faces direction of travel; same E-W width as the flap)
L_FRONT: float = 1.75   # m — N-S depth of front panel  ← set to real value
D_FRONT: float = 2.04  # m — E-W width of front panel  (== D_FLAP, no overhang)

# Back panel (wider than the flap; 0.31 m clear strip on each side)
L_BACK:  float = 1.8   # m — N-S depth of back panel   ← set to real value
D_BACK:  float = 2.66  # m — E-W width of back panel

DEFAULT_SHADE = ShadeParams(
    H_flap  = H_FLAP,
    L_flap  = L_FLAP,
    d_flap  = D_FLAP,
    L_front = L_FRONT,
    d_front = D_FRONT,
    L_back  = L_BACK,
    d_back  = D_BACK,
    two_panel = True,
)


# ---------------------------------------------------------------------------
# Single-axis tracker comparison (used by the latitude sweep)
# ---------------------------------------------------------------------------

def single_axis_comparison(latitude: float) -> dict:
    """
    Compares the three main single-axis tracker types against a fixed-tilt
    baseline.  Optimal tilt = latitude is used for Fixed / VSAT / PSAT.

    HSAT  – trackingtype=1, horizontal N-S axis (angle=0)
    VSAT  – trackingtype=3, vertical axis, module tilt = latitude
    PSAT  – trackingtype=5, inclined N-S axis, axis tilt = latitude
    """
    return {
        "Fixed":     SystemConfig("Fixed",    trackingtype=0, angle=latitude, aspect=0.0),
        "HSAT":      SystemConfig("HSAT",     trackingtype=1, angle=0.0,      aspect=0.0),
        "VSAT":      SystemConfig("VSAT",     trackingtype=3, angle=latitude, aspect=0.0),
        "PSAT":      SystemConfig("PSAT",     trackingtype=5, angle=latitude, aspect=0.0),
    }


# ---------------------------------------------------------------------------
# Vehicle scenario group  (always N↔S driving)
# ---------------------------------------------------------------------------

def ns_vehicle_systems(shade_params: ShadeParams = DEFAULT_SHADE) -> dict:
    """
    All panels mounted on a vehicle driving N↔S.

    Flat (shaded)   — front + back horizontal panels with obstacle shadow
    Flat (unshaded) — same total area, no obstacle (upper bound baseline)
    NS Side (East)  — 6.8 m² vertical panel facing East (right side going North)
    NS Side (West)  — 6.8 m² vertical panel facing West (left side going North)
    NS Side (Total) — combined energy of both side panels (sum of East + West)
    """
    SIDE_PANEL_AREA_M2 = 6.8  # m² per side

    _total_area = (shade_params.L_front * shade_params.d_front +
                   shade_params.L_back * shade_params.d_back)

    unshaded_cfg = flat_ns_vehicle_system(
        shade_params=None,
        name="NS Flat (unshaded)",
        total_panel_area_m2=_total_area,
    )

    return {
        # ── Flat roof panels (front + back, with and without shading) ──────
        "NS Flat (shaded)": flat_ns_vehicle_system(
            shade_params=shade_params,
            name="NS Flat (shaded)",
        ),
        #"NS Flat (unshaded)": unshaded_cfg,

        # ── Side panels: tilt=90°, each 6.8 m² ────────────────────────────
        # East-facing panel: right-hand side when driving North.
        # When the vehicle reverses (going South) this same physical panel
        # faces West, so PVGIS fetches for both azimuths are averaged
        # inside tilted_ns_vehicle_system.
        # "NS Side (East)": tilted_ns_vehicle_system(
        #     tilt_deg=90,
        #     forward_azimuth_deg_from_north=90,  # East when going N
        #     #panel_area_m2=SIDE_PANEL_AREA_M2,
        #     name="NS Side (East)",
        # ),

        # West-facing panel: left-hand side when driving North.
        # By symmetry its annual yield equals the East panel's, but we
        # model it explicitly so monthly/hourly breakdowns are available.
        "NS Sides": tilted_ns_vehicle_system(
            tilt_deg=90,
            forward_azimuth_deg_from_north=270,  # West when going N
            #panel_area_m2=SIDE_PANEL_AREA_M2,
            name="NS Side (West)",
        ),
    }
# ---------------------------------------------------------------------------
# Main scenario group builder
# ---------------------------------------------------------------------------

def build_scenario_groups(fixed_tilt: float, common: dict) -> dict:
    """
    Returns an ordered dict of scenario groups.

    Each group is a dict of {system_name: SystemConfig}.
    Simulation parameters (location, months, hours) come from the CLI
    via the common dict — nothing is overwritten here.
    """
    return {
        "ns_vehicle": ns_vehicle_systems(DEFAULT_SHADE),
    }


# ---------------------------------------------------------------------------
# Latitude sweep configuration
# ---------------------------------------------------------------------------

LAT_SWEEP_CONFIG = {
    "lat_min":    0.0,
    "lat_max":   55.0,
    "lat_step":   5.0,
    "longitude": 25.0,
    "months":    list(range(1, 13)),
    "start_hour": 6.0,
    "end_hour":  20.0,
}