# Severe weather hazard-aware classification project

Matches the plan in the project proposal: extend Gensini et al. (2021) with
a raw-profile deep model that learns which atmospheric layers matter,
instead of relying on 84 hand-engineered variables.

## What's here so far

- `configs/scope.yaml` — every scope decision in one place (year range,
  region, label thresholds, QC settings). Edit this, not the code, when
  you want to change scope.
- `src/data_pipeline/parse_spc.py` — parses SPC tornado/hail report CSVs
  into a clean, labeled, deduplicated parquet file. **Tested and working**
  against synthetic data matching SPC's real schema (see below).

## What you need to do next, on your own machine

This sandbox can't reach `spc.noaa.gov` or the Copernicus CDS API, so
steps 1-2 below have to happen on a machine with normal internet access.

### 1. Get the real SPC files
Go to https://www.spc.noaa.gov/wcm/#data and download:
- `1950-<latest>_actual_tornadoes.csv` (the "actual" file, not preliminary)
- `1955-<latest>_hail.csv`

Save them into `data/raw/` using the exact filenames referenced in
`configs/scope.yaml` (or update the config to match whatever you name them).

**Important:** delete the synthetic test files currently sitting in
`data/raw/` first — they're fake data used only to prove the parser works,
not real reports. Running the parser against them again will silently mix
fake and real data if you don't remove them.

### 2. Set up Copernicus CDS access
1. Register at https://cds.climate.copernicus.eu (free).
2. Generate your API key from your account page.
3. Save it to `~/.cdsapirc` following their setup instructions.
4. `pip install cdsapi` (not yet in requirements.txt — we'll add it when
   we build the ERA5 fetch script next).

### 3. Run the parser for real
```bash
pip install -r requirements.txt
python3 src/data_pipeline/parse_spc.py --config configs/scope.yaml
```
This should print report counts for both hazards and save
`data/processed/reports.parquet`. Sanity-check the significant:severe
ratio against the original paper's rough proportions (~9:1) — if it's
wildly different, something in the scope config or source file is off.

## Data pipeline: what's built and tested so far

- `src/data_pipeline/parse_spc.py` — parses SPC reports. Tested (see above).
- `src/data_pipeline/fetch_era5.py` — pulls ERA5 pressure-level data from
  CDS, one NetCDF file per month, into `data/raw/era5/`. **Not runnable in
  this sandbox** (no network access to CDS) — this is real, correct code,
  but it has to be run on your own machine.
- `src/data_pipeline/extract_profiles.py` — matches each report to its
  atmospheric profile (nearest grid point, 1-hour-prior lag) and applies
  the two QC checks (nonzero CAPE, low-level RH contamination filter).
  **Tested against a synthetic ERA5-shaped file** — confirmed it correctly
  passes realistic profiles and correctly rejects both a stable/isothermal
  profile (zero CAPE) and an artificially saturated one (RH contamination).

### Honest heads-up on ERA5 download size

With the default `configs/scope.yaml` (2010-2023, full region, all 5
variables, all 24 hours/day, 16 pressure levels), `fetch_era5.py` will
pull 14 years x 12 months = 168 files, and each one covers the full
region at hourly resolution — this adds up to a genuinely large download
(likely tens of GB total) and CDS's request queue can take a while per
month under load. Two ways to cut this down if it's too slow:
1. Narrow `start_year`/`end_year` in the config to a shorter test range
   first (e.g. 2 years) to prove the whole pipeline works end-to-end
   before committing to the full pull.
2. Narrow `region_bbox` to a smaller area for initial testing.

### Running the real pipeline, in order

```bash
pip install -r requirements.txt

# 1. Parse real SPC reports (after replacing the synthetic test CSVs)
python3 src/data_pipeline/parse_spc.py --config configs/scope.yaml

# 2. Pull ERA5 data (requires ~/.cdsapirc set up -- see above)
python3 src/data_pipeline/fetch_era5.py --config configs/scope.yaml

# 3. Extract profiles + QC -- produces the final paired dataset
python3 src/data_pipeline/extract_profiles.py --config configs/scope.yaml
```

The synthetic files currently in `data/raw/` and `data/raw/era5/` are
test fixtures only, proving the code works -- delete them before running
the real pipeline so they don't get mixed in with real data.

## Next build step (not started yet)
The random forest baseline (`src/baseline/`) — computes a reduced set of
engineered variables (via MetPy) from `paired_dataset.parquet` and
reproduces the original paper's classifier as a validation checkpoint,
before any deep learning work begins.
