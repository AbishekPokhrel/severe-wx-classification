"""
Batch-pull ERA5 pressure-level reanalysis data for the study region and
period, one file per month. This is deliberately batched by month rather
than pulled per-report -- doing 100k+ individual small API calls would be
extremely slow and would also hammer CDS's rate limits.

Requires a free Copernicus CDS account and API key set up in ~/.cdsapirc
(see README.md for setup steps). This script makes real network calls to
cds.climate.copernicus.eu, so it must be run on a machine with normal
internet access -- it will not run inside this sandbox.

CDS request queueing means each month can take anywhere from a couple of
minutes to a couple of hours depending on their server load, so this is
written to run unattended and to skip months you've already downloaded
(safe to re-run if it gets interrupted).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi
import yaml

# 16 pressure levels spanning boundary layer through upper troposphere.
# Lighter than the original paper's full 137 hybrid-sigma levels, but
# detailed enough to capture the low/mid/upper-level structure the
# hazard-gating model needs to learn from.
PRESSURE_LEVELS_HPA = [
    "1000", "975", "950", "925", "900", "850", "800", "700",
    "600", "500", "400", "300", "250", "200", "150", "100",
]

VARIABLES = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]


def load_scope(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def month_range(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def fetch_month(client: cdsapi.Client, year: int, month: int, bbox, out_dir: Path) -> Path:
    lon_min, lon_max, lat_min, lat_max = bbox
    # CDS area format is [north, west, south, east]
    area = [lat_max, lon_min, lat_min, lon_max]

    out_path = out_dir / f"era5_{year:04d}{month:02d}.nc"
    if out_path.exists():
        print(f"  {out_path.name} already exists, skipping")
        return out_path

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": VARIABLES,
            "pressure_level": PRESSURE_LEVELS_HPA,
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
        },
        str(out_path),
    )
    print(f"  saved {out_path.name}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scope.yaml")
    parser.add_argument("--out-dir", default="data/raw/era5")
    args = parser.parse_args()

    scope = load_scope(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()  # reads credentials from ~/.cdsapirc

    start_year = scope["data"]["start_year"]
    end_year = scope["data"]["end_year"]
    bbox = scope["data"]["region_bbox"]

    for year, month in month_range(start_year, end_year):
        print(f"Fetching {year}-{month:02d} ...")
        fetch_month(client, year, month, bbox, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
