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


def build_scenario_groups(fixed_tilt: float, common: dict) -> dict:

    common["latitude"]   = 65
    common["longitude"]  = 15
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
        "vehicle_ns": {
            **ns_vehicle,
        },


        #"stationary": base,

        #"vehicle_orientations": {
        #    "EW Flat": ew_vehicle["EW Flat"],
        #    "EW Rear": ew_vehicle["EW Rear"],
        #    "EW Side": ew_vehicle["EW Side"],
        #},

        #"vehicle_ew": {
        #
        #    **ew_vehicle,   # adds Flat, Rear, Side — named relative to EW driving
        #},





        #"vehicle_tracking": vehicle_tracking,
    }