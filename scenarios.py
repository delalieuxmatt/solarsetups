from simulation import fixed_system, single_axis_system, dual_axis_system, tractor_system, SystemConfig, single_axis_ew_system

def _vehicle_systems(forward_azimuth: float, label: str) -> dict:
    side_azimuth = (forward_azimuth + 90) % 360   # perpendicular to driving direction


    return {
        f"{label} Flat": SystemConfig(
            f"{label} Flat", trackingtype=0, angle=0.0, aspect=0.0
        ),
        f"{label} Rear": tractor_system(
            tilt_deg=90,
            forward_azimuth_deg_from_north=forward_azimuth,   # driving axis, bidirectional
        ),
        f"{label} Side": tractor_system(
            tilt_deg=90,
            forward_azimuth_deg_from_north=side_azimuth,      # perpendicular axis, bidirectional
        ),
    }


def single_axis_comparison(latitude: float) -> dict:
    """
    Compares the three main single-axis tracker types against fixed-tilt
    and dual-axis baselines. All tilt/axis angles use latitude as the
    optimal value, consistent with the text.

    HSAT  – trackingtype=1, axis horizontal (angle=0), N-S aligned
    VSAT  – trackingtype=3, module tilt = latitude (optimal fixed tilt on vertical axis)
    PSAT  – trackingtype=5, axis tilt = latitude (polar-aligned inclined axis)
    """
    return {
        "Fixed":     SystemConfig("Fixed",    trackingtype=0, angle=latitude,  aspect=0.0),
        "HSAT":      SystemConfig("HSAT",     trackingtype=1, angle=0.0,       aspect=0.0),
        "VSAT":      SystemConfig("VSAT",     trackingtype=3, angle=latitude,  aspect=0.0),
        "PSAT":      SystemConfig("PSAT",     trackingtype=5, angle=latitude,  aspect=0.0),
        #"Dual-Axis": dual_axis_system(),
    }

def build_scenario_groups(fixed_tilt: float, common: dict) -> dict:

    common["latitude"]   = 10
    common["longitude"]  = 5
    common["months"]     = [1,2,3,4,5,6,7,8,9,10,11,12]
    common["start_hour"] = 6.0
    common["end_hour"]   = 20.0


    base = {
        "Fixed":       fixed_system(tilt_deg=fixed_tilt),
        "Single-Axis": single_axis_system(axis_tilt_deg=fixed_tilt),
        "Dual-Axis":   dual_axis_system(),
    }

    vehicle_tracking = {
        "Dual-Axis": dual_axis_system(),  # should == stationary
        "Single-Axis NS": single_axis_system(),  # should == stationary (NS driving)
        "Single-Axis EW": single_axis_ew_system(),  # should be lower
    }

    ew_vehicle = _vehicle_systems(forward_azimuth=90, label="EW")   # drives E↔W
    ns_vehicle = _vehicle_systems(forward_azimuth=0, label="NS")    # drives N↔S


    return {

        "single_axis_comparison": single_axis_comparison(latitude=common["latitude"]),

    }

# ---------------------------------------------------------------------------
# Latitude sweep configuration
# ---------------------------------------------------------------------------
# Edit these values to control the latitude sweep behaviour.
# The sweep runs for every system in single_axis_comparison().

LAT_SWEEP_CONFIG = {
    "lat_min":    0,    # southernmost latitude to simulate (°N)
    "lat_max":   55.0,    # northernmost latitude to simulate (°N)
    "lat_step":   5.0,    # step size in degrees
    "longitude":  25.0,    # fixed longitude used for all latitudes
    "months":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # full year
    "start_hour": 6.0,
    "end_hour":  20.0,
}