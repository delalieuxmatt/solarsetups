"""
irradiance.py
Defines the IrradianceProvider abstraction and concrete implementations.

To add a new data source (e.g. PVGIS), subclass IrradianceProvider and
implement get_irradiance(). Then pass it to the Simulation in main.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class IrradianceData:
    """
    DNI  — Direct Normal Irradiance  (W/m²): direct beam on surface perpendicular to sun
    DHI  — Diffuse Horizontal Irradiance (W/m²): scattered sky radiation
    GHI  — Global Horizontal Irradiance (W/m²): total on horizontal surface
    """
    dni: float
    dhi: float
    ghi: float


class IrradianceProvider(ABC):
    """Abstract base class. Swap implementations without touching simulation code."""

    @abstractmethod
    def get_irradiance(self, dt: datetime, lat: float, lon: float,
                       solar_altitude_rad: float) -> IrradianceData:
        """
        Return irradiance components for a specific moment and location.

        Parameters
        ----------
        dt                : UTC datetime
        lat, lon          : degrees
        solar_altitude_rad: solar altitude in radians (pre-computed, saves re-work)
        """
        ...


# ---------------------------------------------------------------------------
# 1. Constant (physics-based) provider  ← used now
# ---------------------------------------------------------------------------

SOLAR_CONSTANT = 1370.0  # W/m²


class ConstantIrradianceProvider(IrradianceProvider):
    """
    Uses the solar constant (1370 W/m²) attenuated by the Kasten-Young air mass
    model. No real weather data — pure clear-sky physics.

    DNI = 1370 * 0.7^(AM^0.678)          (Meinel & Meinel empirical formula)
    DHI = 0.1 * DNI                       (simplified diffuse fraction)
    GHI = DNI * sin(altitude) + DHI
    """

    def __init__(self, solar_constant: float = SOLAR_CONSTANT):
        self.solar_constant = solar_constant

    def get_irradiance(self, dt: datetime, lat: float, lon: float,
                       solar_altitude_rad: float) -> IrradianceData:
        if solar_altitude_rad <= 0:
            return IrradianceData(dni=0.0, dhi=0.0, ghi=0.0)

        # Air mass (Kasten-Young 1989)
        alt_deg = np.degrees(solar_altitude_rad)
        am = 1.0 / (np.sin(solar_altitude_rad) +
                    0.50572 * (alt_deg + 6.07995) ** -1.6364)
        am = min(am, 38.0)  # cap at ~horizon

        dni = self.solar_constant * (0.7 ** (am ** 0.678))
        dhi = 0.1 * dni
        ghi = dni * np.sin(solar_altitude_rad) + dhi

        return IrradianceData(dni=dni, dhi=dhi, ghi=ghi)


# ---------------------------------------------------------------------------
# 2. PVGIS provider  ← stub, ready to implement
# ---------------------------------------------------------------------------

class PVGISProvider(IrradianceProvider):
    """
    Fetches hourly irradiance from the PVGIS REST API (European Commission).
    API docs: https://re.jrc.ec.europa.eu/api/v5_2/

    Usage (once implemented):
        provider = PVGISProvider(year=2020)
        sim = Simulation(provider=provider, ...)

    TODO:
        - Call https://re.jrc.ec.europa.eu/api/v5_2/seriescalc with
          lat, lon, startyear, endyear, outputformat=json
        - Cache the response (one API call per lat/lon/year)
        - Interpolate hourly data to per-minute values
        - Return DNI/DHI/GHI from the 'outputs.hourly' array
    """

    def __init__(self, year: int = 2020):
        self.year = year
        self._cache: dict = {}
        raise NotImplementedError(
            "PVGISProvider is not yet implemented. "
            "Use ConstantIrradianceProvider for now."
        )

    def get_irradiance(self, dt: datetime, lat: float, lon: float,
                       solar_altitude_rad: float) -> IrradianceData:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. NASA POWER provider  ← stub, ready to implement
# ---------------------------------------------------------------------------

class NASAPowerProvider(IrradianceProvider):
    """
    Fetches hourly irradiance from the NASA POWER API (free, global coverage).
    API docs: https://power.larc.nasa.gov/api/

    Parameters: ALLSKY_SFC_SW_DNI, ALLSKY_SFC_SW_DIFF, ALLSKY_SFC_SW_DWN
    Temporal resolution: hourly

    TODO:
        - Call https://power.larc.nasa.gov/api/temporal/hourly/point
          with parameters, lat, lon, start, end, community=RE
        - Cache the full response per location/year
        - Return DNI/DHI/GHI for the matching hour
    """

    def __init__(self, year: int = 2020):
        self.year = year
        self._cache: dict = {}
        raise NotImplementedError(
            "NASAPowerProvider is not yet implemented. "
            "Use ConstantIrradianceProvider for now."
        )

    def get_irradiance(self, dt: datetime, lat: float, lon: float,
                       solar_altitude_rad: float) -> IrradianceData:
        raise NotImplementedError
