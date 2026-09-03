"""
Match each SPC report to its atmospheric profile from the locally-downloaded
ERA5 files, then apply quality control.

Input:  data/processed/reports.parquet   (from parse_spc.py)
        data/raw/era5/era5_YYYYMM.nc     (from fetch_era5.py, one per month)
Output: data/processed/paired_dataset.parquet  -- one row per report, with
        the profile arrays (temperature, specific humidity, u/v wind,
        geopotential height at each pressure level) stored as list columns,
        plus the original label.

QC applied, matching the original paper's approach:
  1. Nonzero CAPE requirement (computed via MetPy on the extracted profile).
  2. Low-level relative humidity contamination filter -- if the profile is
     already too saturated near the surface, it's likely picking up the
     storm's own effects rather than the pre-storm environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from metpy.calc import (
    dewpoint_from_specific_humidity,
    parcel_profile,
    relative_humidity_from_specific_humidity,
    surface_based_cape_cin,
)
from metpy.units import units


def load_scope(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def open_era5_for_year_month(era5_dir: Path, year: int, month: int) -> xr.Dataset | None:
    path = era5_dir / f"era5_{year:04d}{month:02d}.nc"
    if not path.exists():
        return None
    return xr.open_dataset(path)


def extract_profile(ds: xr.Dataset, lat: float, lon: float, target_time: pd.Timestamp) -> dict | None:
    """Pull the nearest-gridpoint, nearest-hour profile as plain arrays."""
    try:
        point = ds.sel(latitude=lat, longitude=lon, time=target_time, method="nearest")
    except Exception:
        return None

    levels_hpa = point["level"].values  # pressure levels, hPa
    temp_k = point["t"].values          # Kelvin
    q_kgkg = point["q"].values          # specific humidity, kg/kg
    u_ms = point["u"].values
    v_ms = point["v"].values
    z_m = point["z"].values / 9.80665   # geopotential -> geopotential height

    if np.isnan(temp_k).any() or np.isnan(q_kgkg).any():
        return None

    return {
        "pressure_hpa": levels_hpa,
        "temperature_k": temp_k,
        "specific_humidity_kgkg": q_kgkg,
        "u_wind_ms": u_ms,
        "v_wind_ms": v_ms,
        "geopotential_height_m": z_m,
    }


def passes_qc(profile: dict, max_low_level_rh_pct: float) -> tuple[bool, str]:
    """Returns (passed, reason). Mirrors the original paper's two checks."""
    p = profile["pressure_hpa"] * units.hPa
    t = profile["temperature_k"] * units.kelvin
    q = profile["specific_humidity_kgkg"] * units("kg/kg")

    # Sort surface-to-top (MetPy expects decreasing pressure order upward;
    # ERA5 pressure levels are often stored high-to-low or low-to-high
    # depending on the download -- enforce ascending height / descending
    # pressure here).
    order = np.argsort(-p.magnitude)
    p, t, q = p[order], t[order], q[order]

    td = dewpoint_from_specific_humidity(p, t, q)

    # --- Check 1: nonzero CAPE ---
    try:
        prof = parcel_profile(p, t[0], td[0])
        cape, _cin = surface_based_cape_cin(p, t, td)
    except Exception as e:
        return False, f"CAPE calculation failed: {e}"

    if cape.magnitude <= 0:
        return False, "zero or undefined CAPE"

    # --- Check 2: low-level RH contamination filter ---
    rh = relative_humidity_from_specific_humidity(p, t, q).to("percent").magnitude
    low_level_mask = p.magnitude >= 850  # roughly surface to ~1.5km AGL
    if low_level_mask.any() and np.mean(rh[low_level_mask]) >= max_low_level_rh_pct:
        return False, "low-level RH contamination"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scope.yaml")
    parser.add_argument("--reports", default="data/processed/reports.parquet")
    parser.add_argument("--era5-dir", default="data/raw/era5")
    parser.add_argument("--out", default="data/processed/paired_dataset.parquet")
    args = parser.parse_args()

    scope = load_scope(args.config)
    lag_hours = scope["qc"]["proximity_lag_hours"]
    max_rh = scope["qc"]["max_low_level_rh_pct"]

    reports = pd.read_parquet(args.reports)
    era5_dir = Path(args.era5_dir)

    rows = []
    dropped = {"no_era5_file": 0, "extraction_failed": 0, "qc_failed": 0}
    open_files: dict[tuple[int, int], xr.Dataset] = {}

    for _, report in reports.iterrows():
        target_time = report["snap_hour_utc"] - pd.Timedelta(hours=lag_hours)
        key = (target_time.year, target_time.month)

        if key not in open_files:
            ds = open_era5_for_year_month(era5_dir, *key)
            open_files[key] = ds
        ds = open_files[key]

        if ds is None:
            dropped["no_era5_file"] += 1
            continue

        profile = extract_profile(ds, report["lat"], report["lon"], target_time)
        if profile is None:
            dropped["extraction_failed"] += 1
            continue

        ok, _reason = passes_qc(profile, max_rh)
        if not ok:
            dropped["qc_failed"] += 1
            continue

        rows.append({
            "hazard": report["hazard"],
            "label": report["label"],
            "lat": report["lat"],
            "lon": report["lon"],
            "report_time_utc": report["report_time_utc"],
            **{k: list(v) for k, v in profile.items()},
        })

    for ds in open_files.values():
        if ds is not None:
            ds.close()

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    print(f"Paired profiles kept: {len(out_df)}")
    print(f"Dropped -- no ERA5 file: {dropped['no_era5_file']}, "
          f"extraction failed: {dropped['extraction_failed']}, "
          f"QC failed: {dropped['qc_failed']}")
    if len(out_df):
        print(out_df.groupby(["hazard", "label"]).size())
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
