"""
Parse SPC (Storm Prediction Center) tornado and hail report files.

Download the source files yourself from:
    https://www.spc.noaa.gov/wcm/#data
Look for "1950-<latest>_actual_tornadoes.csv" and "1955-<latest>_hail.csv"
(the "actual" files, not the preliminary ones). Save them to
data/raw/ using the filenames in configs/scope.yaml.

SPC's published schema (same for both hazard files) is:
    om, yr, mo, dy, date, time, tz, st, stf, stn,
    mag, inj, fat, loss, closs,
    slat, slon, elat, elon,
    len, wid, ns, sn, sg, f1, f2, f3, f4, fc

Relevant fields for this project:
    date, time, tz  -> report timestamp (tz is an SPC timezone code, not
                        a standard one -- see TZ_TO_UTC_OFFSET below)
    slat, slon      -> start latitude/longitude of the report
    mag             -> for tornado files: (E)F-scale rating (int, -9 = unknown)
                        for hail files: hail diameter in inches (float)

This script does NOT hit the network -- it only reads local CSVs you've
already downloaded. That's deliberate: SPC's site isn't reachable from
this sandbox, so parsing logic is written and testable here, and you run
it for real once you have the files locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

# SPC's `tz` column uses old-style numeric timezone codes, not IANA names.
# 3 = CST, 4 = EST, 6 = MST, 7 = PST, 9 = GMT/UTC, 0 = unknown (assume CST,
# SPC's documented default for older records).
TZ_TO_UTC_OFFSET_HOURS = {
    0: 6,  # unknown -> assume CST (SPC convention)
    3: 6,  # CST
    4: 5,  # EST
    6: 7,  # MST
    7: 8,  # PST
    9: 0,  # GMT/UTC
}


def load_scope(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_timestamp_utc(df: pd.DataFrame) -> pd.Series:
    """Build a UTC timestamp from SPC's date, time, tz columns."""
    local = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    offset_hours = df["tz"].map(TZ_TO_UTC_OFFSET_HOURS)
    if offset_hours.isna().any():
        bad = sorted(df.loc[offset_hours.isna(), "tz"].unique().tolist())
        raise ValueError(
            f"Unrecognized SPC tz code(s) {bad} -- add them to "
            "TZ_TO_UTC_OFFSET_HOURS before proceeding."
        )
    return local + pd.to_timedelta(offset_hours, unit="h")


def parse_tornado(csv_path: str | Path, scope: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["mag"] >= 0].copy()  # drop unrated (-9) tornadoes
    df["report_time_utc"] = _parse_timestamp_utc(df)
    df["hazard"] = "tornado"
    df["lat"] = df["slat"]
    df["lon"] = df["slon"]
    df["magnitude"] = df["mag"].astype(float)

    thresholds = scope["labels"]["tornado"]
    df = df[df["magnitude"] >= thresholds["severe_min_mag"]]
    df["label"] = (df["magnitude"] >= thresholds["significant_min_mag"]).astype(int)

    return _finalize(df, scope)


def parse_hail(csv_path: str | Path, scope: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["report_time_utc"] = _parse_timestamp_utc(df)
    df["hazard"] = "hail"
    df["lat"] = df["slat"]
    df["lon"] = df["slon"]
    df["magnitude"] = df["mag"].astype(float)  # inches

    thresholds = scope["labels"]["hail"]
    df = df[df["magnitude"] >= thresholds["severe_min_in"]]
    df["label"] = (df["magnitude"] >= thresholds["significant_min_in"]).astype(int)

    return _finalize(df, scope)


def _finalize(df: pd.DataFrame, scope: dict) -> pd.DataFrame:
    """Shared cleanup: year range, region bbox, dedup, column selection."""
    start_year, end_year = scope["data"]["start_year"], scope["data"]["end_year"]
    df = df[df["report_time_utc"].dt.year.between(start_year, end_year)]

    lon_min, lon_max, lat_min, lat_max = scope["data"]["region_bbox"]
    df = df[
        df["lon"].between(lon_min, lon_max) & df["lat"].between(lat_min, lat_max)
    ]

    # Proximity extraction will later map each report to an ERA5 grid
    # cell + floored hour. Multiple reports can land on the same
    # (grid cell, hour) pair -- keep the highest-magnitude one, matching
    # the original paper's deduplication rule.
    df["snap_hour_utc"] = df["report_time_utc"].dt.floor("h")
    df = (
        df.sort_values("magnitude", ascending=False)
        .drop_duplicates(subset=["hazard", "snap_hour_utc", "lat", "lon"], keep="first")
    )

    cols = ["hazard", "report_time_utc", "snap_hour_utc", "lat", "lon", "magnitude", "label"]
    return df[cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scope.yaml")
    parser.add_argument("--out", default="data/processed/reports.parquet")
    args = parser.parse_args()

    scope = load_scope(args.config)

    tor = parse_tornado(scope["data"]["spc_tornado_csv"], scope)
    hail = parse_hail(scope["data"]["spc_hail_csv"], scope)
    combined = pd.concat([tor, hail], ignore_index=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)

    print(f"Tornado reports kept: {len(tor)}  (significant: {tor['label'].sum()})")
    print(f"Hail reports kept:    {len(hail)}  (significant: {hail['label'].sum()})")
    print(f"Saved combined dataset -> {args.out}")


if __name__ == "__main__":
    main()
