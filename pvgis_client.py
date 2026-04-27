"""
pvgis_client.py
================
Thin wrapper around the PVGIS v5.2 seriescalc API.

PVGIS gives us hourly G(i) — the total in-plane irradiance already
accounting for beam, diffuse, and ground-reflected components for the
requested mounting geometry.  We never need to split into DNI/DHI or
compute angles of incidence ourselves; that is what caused the original
flat-panel bias.

Tracking types accepted by PVGIS:
    0  fixed plane             (angle + aspect required)
    1  single horizontal axis, N-S aligned  (angle = fixed tilt of axis)
    2  two-axis tracking       (always points at sun)
    3  vertical axis tracking
    4  single horizontal axis, E-W aligned
    5  single inclined axis, N-S aligned

Azimuth convention (PVGIS raw API `aspect` parameter):
    0 = South, -90 = East, +90 = West
    This is *different* from the pvlib convention (0=North).
    We accept the common "0=South" convention here and convert internally
    if needed (currently we just pass it straight through since our default
    is south-facing = 0).

The caller receives a pandas DataFrame indexed by UTC datetime with columns:
    G_i      W/m²   global in-plane irradiance on the panel surface
    Gb_i     W/m²   beam component
    Gd_i     W/m²   diffuse component
    Gr_i     W/m²   ground-reflected component
    H_sun    deg    sun height above horizon
    T2m      °C     air temperature at 2 m
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PVGIS_BASE = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"

# Simple on-disk cache so repeated runs with the same parameters don't hammer
# the API.  Cache lives in a subdirectory of the script's folder.
CACHE_DIR = Path(__file__).parent / ".pvgis_cache"

# PVGIS SARAH2 covers 2005-2023 for Europe.  We fetch a single representative
# year so that month-level comparisons are fair (same weather year).
DEFAULT_YEAR = 2023
DEFAULT_DB   = "PVGIS-SARAH2"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_hourly(
    lat: float,
    lon: float,
    trackingtype: int,
    *,
    angle: float = 0.0,
    aspect: float = 0.0,
    year: int = DEFAULT_YEAR,
    raddatabase: str = DEFAULT_DB,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch one full year of hourly in-plane irradiance from PVGIS.

    Parameters
    ----------
    lat, lon       : location in decimal degrees
    trackingtype   : 0=fixed, 1=single-axis N-S, 2=dual-axis, etc.
    angle          : panel tilt for fixed / axis tilt for single-axis (degrees)
    aspect         : azimuth offset from South (0=S, -90=E, +90=W)
                     ignored for dual-axis (trackingtype=2)
    year           : calendar year to fetch (must be in the DB range)
    raddatabase    : PVGIS radiation database name
    use_cache      : skip API call if identical request was cached

    Returns
    -------
    DataFrame indexed by UTC timestamp (hourly) with columns:
        G_i, Gb_i, Gd_i, Gr_i, H_sun, T2m
    """
    params = _build_params(lat, lon, trackingtype, angle, aspect, year, raddatabase)
    cache_key = _cache_key(params)

    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    df = _call_api(params)

    if use_cache:
        _save_cache(cache_key, df)

    return df


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_params(lat, lon, trackingtype, angle, aspect, year, raddatabase):
    params = {
        "lat":          round(lat, 4),
        "lon":          round(lon, 4),
        "startyear":    year,
        "endyear":      year,
        "trackingtype": trackingtype,
        "angle":        round(angle, 1),
        "aspect":       round(aspect, 1),
        "components":   1,          # return Gb(i), Gd(i), Gr(i) separately
        "outputformat": "json",
        "browser":      0,
        "raddatabase":  raddatabase,
    }
    # For dual-axis (trackingtype=2) angle/aspect are ignored by PVGIS,
    # but we send them anyway; it causes no harm.
    return params


def _call_api(params: dict, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            resp = requests.get(PVGIS_BASE, params=params, timeout=60)
            resp.raise_for_status()
            return _parse_response(resp.json())

        except requests.HTTPError as exc:
            # Surface the PVGIS error message if available
            try:
                msg = resp.json().get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"PVGIS API error: {msg}") from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"PVGIS unreachable after {retries} attempts: {exc}"
                ) from exc
            time.sleep(2 ** attempt)   # exponential back-off


def _parse_response(data: dict) -> pd.DataFrame:
    """Turn PVGIS JSON into a tidy DataFrame indexed by UTC datetime."""
    hourly_records = data["outputs"]["hourly"]
    print(f"DEBUG: First record keys: {hourly_records[0].keys()}")  # Add this line

    rows = []
    for rec in hourly_records:
        # Calculate G_i by summing components if G(i) is missing
        g_i = rec.get("G(i)", rec.get("Gi"))
        if g_i is None:
            g_i = float(rec.get("Gb(i)", 0)) + float(rec.get("Gd(i)", 0)) + float(rec.get("Gr(i)", 0))
        else:
            g_i = float(g_i)

        rows.append({
            "datetime": pd.to_datetime(rec["time"], format="%Y%m%d:%H%M", utc=True),
            "G_i": g_i,
            "Gb_i": float(rec.get("Gb(i)", 0.0)),
            "Gd_i": float(rec.get("Gd(i)", 0.0)),
            "Gr_i": float(rec.get("Gr(i)", 0.0)),
            "H_sun": float(rec.get("H_sun", 0.0)),
            "T2m": float(rec.get("T2m", 0.0)),
        })

    df = pd.DataFrame(rows).set_index("datetime")
    return df


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def _load_cache(key: str) -> Optional[pd.DataFrame]:
    # Try parquet first (smaller), fall back to pickle
    for suffix, reader in [(".parquet", pd.read_parquet), (".pkl", pd.read_pickle)]:
        path = CACHE_DIR / f"{key}{suffix}"
        if path.exists():
            try:
                return reader(path)
            except Exception:
                path.unlink(missing_ok=True)
    return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(CACHE_DIR / f"{key}.parquet")
        return
    except Exception:
        pass
    try:
        df.to_pickle(CACHE_DIR / f"{key}.pkl")
    except Exception:
        pass  # Cache write failure is non-fatal