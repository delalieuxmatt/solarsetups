"""
irradiance.py
Defines the IrradianceProvider abstraction and concrete implementations.

To add a new data source, subclass IrradianceProvider and implement
get_irradiance(). Then pass it to the Simulation in main.py — no other
changes required.

Provider overview
-----------------
ConstantIrradianceProvider  — Solar constant + Kasten-Young air mass model.
                               No network, no dependencies, always works.
                               Good for relative system comparisons.

PVGISProvider               — Real multi-year satellite irradiance from the
                               European Commission's PVGIS database.
                               Fetches horizontal DNI + DHI via seriescalc,
                               caches per (lat, lon, year), interpolates to
                               per-minute. Requires internet access.

NASAPowerProvider           — Stub. See class docstring for TODO.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


@dataclass
class IrradianceData:
    """
    DNI  — Direct Normal Irradiance  (W/m²): direct beam on a surface
           perpendicular to the sun direction.
    DHI  — Diffuse Horizontal Irradiance (W/m²): scattered sky radiation
           on a horizontal surface.
    GHI  — Global Horizontal Irradiance (W/m²): total on horizontal surface.
    """
    dni: float
    dhi: float
    ghi: float


class IrradianceProvider(ABC):
    """
    Abstract base class for irradiance data sources.

    Every provider returns the same IrradianceData(dni, dhi, ghi) for any
    given moment and location. The simulation layer uses DNI and DHI to
    compute plane-of-array irradiance via the panel's AOI — providers never
    need to know anything about panel orientation or tracking type.
    """

    @abstractmethod
    def get_irradiance(self, dt: datetime, lat: float, lon: float,
                       solar_altitude_rad: float) -> IrradianceData:
        """
        Return irradiance components for a specific moment and location.

        Parameters
        ----------
        dt                : UTC datetime
        lat, lon          : degrees
        solar_altitude_rad: solar altitude in radians (pre-computed by caller)
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
# 2. PVGIS provider  ← real satellite data from the European Commission
# ---------------------------------------------------------------------------

PVGIS_API = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"

# PVGIS field names returned for seriescalc when components=1
# Gb(i) = Direct in-plane (Horizontal beam since angle=0)
# Gd(i) = Diffuse in-plane (Horizontal diffuse)
# G(i)  = Global in-plane (Global horizontal)
_PVGIS_DNI_KEY = "Gb(i)"
_PVGIS_DHI_KEY = "Gd(i)"
_PVGIS_GHI_KEY = "G(i)"


class PVGISProvider(IrradianceProvider):
    """
    Real hourly irradiance from the PVGIS satellite database (EU JRC).
    API endpoint: https://re.jrc.ec.europa.eu/api/v5_2/seriescalc

    Design
    ------
    We always fetch horizontal component data (trackingtype=0, angle=0,
    components=1). This returns the raw DNI, DHI and GHI for the location,
    independent of any panel orientation. Our existing panel_systems.py AOI
    geometry then applies identically to the PVGIS data as it does to the
    ConstantIrradianceProvider — no changes needed in the simulation layer.

    We deliberately do NOT use PVGIS's built-in tracking plane calculations
    (trackingtype 1/2/5 in seriescalc) because that would require a separate
    API call per system type and bypass our geometry layer.

    Caching
    -------
    One API call is made per unique (lat, lon, year). The full hourly
    time-series (~8760 rows) is stored in self._cache keyed by
    (round(lat,4), round(lon,4), year). Subsequent get_irradiance() calls
    only perform a fast pandas index lookup + linear interpolation.

    Interpolation
    -------------
    PVGIS returns one value per hour (instantaneous, centred on the hour for
    SARAH2). We linearly interpolate between adjacent hourly values to produce
    per-minute estimates. Values before the first or after the last record in
    a day are filled with the nearest known value (no extrapolation below 0).

    Parameters
    ----------
    year        : Calendar year to fetch. PVGIS-SARAH2 covers 2005-2020.
                  ERA5 is available for earlier/later years.
    raddatabase : PVGIS radiation DB. Default uses the PVGIS automatic
                  selection (SARAH2 for Europe/Africa, ERA5 globally).
                  Pass e.g. "PVGIS-ERA5" to override.
    usehorizon  : Whether to include terrain horizon shading (default True).
    timeout_s   : HTTP request timeout in seconds.

    Usage
    -----
        provider = PVGISProvider(year=2020)
        # use exactly like ConstantIrradianceProvider:
        irr = provider.get_irradiance(dt, lat, lon, solar_altitude_rad)
    """

    def __init__(self,
                 year: int = 2020,
                 raddatabase: Optional[str] = None,
                 usehorizon: bool = True,
                 timeout_s: int = 60):
        self.year = year
        self.raddatabase = raddatabase
        self.usehorizon = usehorizon
        self.timeout_s = timeout_s
        # Cache: key = (lat_r, lon_r, year) → pd.DataFrame indexed by UTC datetime
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Public API — identical signature to every other IrradianceProvider
    # ------------------------------------------------------------------

    def get_irradiance(self, dt: datetime, lat: float, lon: float,
                       solar_altitude_rad: float) -> IrradianceData:
        if solar_altitude_rad <= 0:
            return IrradianceData(dni=0.0, dhi=0.0, ghi=0.0)

        cache_key = (round(lat, 4), round(lon, 4), self.year)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._fetch(lat, lon)

        return self._lookup(self._cache[cache_key], dt)

    def fetch_validation_total(self, lat: float, lon: float, trackingtype: int,
                               months: list[int] = None, start_hour: float = 0.0, end_hour: float = 24.0,
                               angle: float = 0.0, aspect: float = 0.0) -> float:
        """
        Fetches the total in-plane irradiance (kWh/m²) directly from PVGIS for the specified timeframe.
        Used for validation against local geometric calculations.
        """
        import requests

        params = {
            "lat": lat,
            "lon": lon,
            "startyear": self.year,
            "endyear": self.year,
            "trackingtype": trackingtype,
            "angle": angle,
            "aspect": aspect,
            "components": 1,
            "pvcalculation": 0,
            "usehorizon": int(self.usehorizon),
            "outputformat": "json",
        }
        if self.raddatabase:
            params["raddatabase"] = self.raddatabase

        mode_str = "Fixed" if trackingtype == 0 else "Dual-Axis"
        print(f"    [PVGIS] Fetching {mode_str} validation baseline...", end="", flush=True)
        resp = requests.get(PVGIS_API, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        print(" done.")

        hourly = data.get("outputs", {}).get("hourly", [])
        total_wh = 0.0

        for row in hourly:
            # Parse the time string "20200615:1011"
            raw = row["time"]
            date_part, time_part = raw.split(":")
            month = int(date_part[4:6])
            hour_float = int(time_part[0:2]) + int(time_part[2:4]) / 60.0

            # Filter out records outside the simulation timeframe
            if months and month not in months:
                continue
            if not (start_hour <= hour_float <= end_hour):
                continue

            if "G(i)" in row:
                g_i = float(row["G(i)"])
            elif "Gb(i)" in row and "Gd(i)" in row:
                g_i = float(row.get("Gb(i)", 0.0)) + float(row.get("Gd(i)", 0.0)) + float(row.get("Gr(i)", 0.0))
            else:
                raise KeyError(f"Validation failed: Expected 'G(i)' or components in PVGIS response.")

            total_wh += max(g_i, 0.0)

        return total_wh / 1000.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self, lat: float, lon: float):
        """
        Call PVGIS seriescalc and return a DataFrame indexed by UTC datetime
        with columns: dni, dhi, ghi.

        API parameters used
        -------------------
        trackingtype = 0        Use fixed horizontal plane — gives us raw
                                DNI/DHI/GHI components untouched by any
                                tracking assumption.
        angle        = 0        Horizontal reference plane.
        components   = 1        Return Gb(n), Gd(h), G(h) separately.
        pvcalculation= 0        Irradiance only — no PV power estimate.
        outputformat = json     Machine-readable response.
        startyear / endyear     Single year requested.

        Note on aspect convention
        -------------------------
        PVGIS uses 0=south, +90=west, -90=east.
        Our panel_systems.py uses clockwise-from-North (0=N, 180=S).
        Because we always request a horizontal plane (angle=0) the aspect
        value is irrelevant and we pass 0 (= south) as a safe default.
        """
        import requests

        params = {
            "lat":           lat,
            "lon":           lon,
            "startyear":     self.year,
            "endyear":       self.year,
            "trackingtype":  0,       # fixed horizontal — raw DNI/DHI/GHI
            "angle":         0,       # horizontal reference
            "aspect":        0,       # irrelevant for angle=0; 0=south
            "components":    1,       # return Gb(n), Gd(h), G(h) separately
            "pvcalculation": 0,       # radiation only
            "usehorizon":    int(self.usehorizon),
            "outputformat":  "json",
        }
        if self.raddatabase:
            params["raddatabase"] = self.raddatabase

        print(f"    [PVGIS] Fetching {self.year} data for ({lat}, {lon})…", end="", flush=True)
        resp = requests.get(PVGIS_API, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        print(" done.")

        return self._parse(data)

    def _parse(self, data: dict):
        import pandas as pd
        import numpy as np

        hourly = data["outputs"]["hourly"]

        records = []
        for row in hourly:
            # Parse "20200615:1011" → datetime(2020, 6, 15, 10, 11, tzinfo=UTC)
            raw = row["time"]
            date_part, time_part = raw.split(":")
            year = int(date_part[0:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            hour = int(time_part[0:2])
            minute = int(time_part[2:4])

            dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

            gb_i = max(float(row.get(_PVGIS_DNI_KEY, 0.0)), 0.0)
            h_sun = float(row.get("H_sun", 0.0))

            # Convert Direct Horizontal (Gb(i) at angle=0) to Direct Normal Irradiance
            if h_sun > 0:
                # Cap DNI at the solar constant (1370) to prevent dawn/dusk math spikes
                dni = min(gb_i / np.sin(np.radians(h_sun)), 1370.0)
            else:
                dni = 0.0

            records.append({
                "datetime": dt,
                "dni": dni,
                "dhi": max(float(row.get(_PVGIS_DHI_KEY, 0.0)), 0.0),
                "ghi": max(float(row.get(_PVGIS_GHI_KEY, 0.0)), 0.0),
            })

        df = pd.DataFrame(records).set_index("datetime").sort_index()
        return df

    def _lookup(self, df, dt: datetime) -> IrradianceData:
        """
        Return interpolated irradiance for an arbitrary UTC datetime.

        Strategy: find the two nearest hourly records that bracket `dt`
        and linearly interpolate. If `dt` falls outside the bracketing
        window (e.g. before first record of the day or after last), clamp
        to the nearest known value. All values are clamped to ≥ 0.

        Year normalisation
        ------------------
        The simulation engine uses a fixed SIM_YEAR (currently 2023) for
        all timesteps, while the PVGIS DataFrame is indexed with the actual
        fetched year (e.g. 2020). Without correction every lookup would land
        past the end of the index and silently return 0 W/m² (the last row
        after December 31st at night). We therefore replace the year in `dt`
        with the year of the first record in the DataFrame before searching.
        Only month/day/hour/minute matter for the lookup — cloud patterns
        from the real data year are preserved; day-of-year solar geometry is
        computed independently by solar_position.py.
        """
        import pandas as pd

        # Normalise dt to UTC-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Map simulation year → PVGIS data year (e.g. 2023 → 2020)
        data_year = df.index[0].year
        if dt.year != data_year:
            try:
                dt = dt.replace(year=data_year)
            except ValueError:
                # Feb 29 in a leap sim year mapped to non-leap data year → use Feb 28
                dt = dt.replace(year=data_year, day=28)

        idx = df.index

        # Exact match — common for on-the-hour steps
        if dt in idx:
            row = df.loc[dt]
            return IrradianceData(dni=row["dni"], dhi=row["dhi"], ghi=row["ghi"])

        # Find bracketing indices using searchsorted
        pos = idx.searchsorted(dt)

        if pos == 0:
            # Before the first record → use first record
            row = df.iloc[0]
            return IrradianceData(dni=row["dni"], dhi=row["dhi"], ghi=row["ghi"])

        if pos >= len(idx):
            # After the last record → use last record
            row = df.iloc[-1]
            return IrradianceData(dni=row["dni"], dhi=row["dhi"], ghi=row["ghi"])

        # Linear interpolation between t0 and t1
        t0 = idx[pos - 1]
        t1 = idx[pos]
        span = (t1 - t0).total_seconds()
        frac = (dt - t0).total_seconds() / span  # 0.0 … 1.0

        r0 = df.loc[t0]
        r1 = df.loc[t1]

        def lerp(a, b):
            return max(a + frac * (b - a), 0.0)

        return IrradianceData(
            dni=lerp(r0["dni"], r1["dni"]),
            dhi=lerp(r0["dhi"], r1["dhi"]),
            ghi=lerp(r0["ghi"], r1["ghi"]),
        )


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