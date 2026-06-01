# Leonia data dictionary

This document is the single reference for every processed dataset
under `data/processed/`. It describes the canonical Parquet files
produced from raw StreetLight exports, the derived analytics tables
built on top of them, and how to load each one.

If you only want to **use** the data, start with
[Quick start](#quick-start). If you want to know what a specific
column means, jump to the file's section below. If you want to
rebuild everything from scratch, see
[Rebuilding the data lake](#rebuilding-the-data-lake).

---

## Quick start

```bash
# One-time: build the canonical data lake from raw streetlight/ exports
venv/bin/python scripts/00_build_datasets.py
```

```python
# Read a canonical tabular dataset (pandas)
import pandas as pd
za = pd.read_parquet("data/processed/streetlight/za_volume.parquet")

# Read a GeoParquet (geopandas)
import geopandas as gpd
lines = gpd.read_parquet("data/processed/streetlight/za_line_shapes.parquet")
# CRS is EPSG:4326 (WGS84 lon/lat) for every geospatial file in the lake.

# Or use the convenience helpers, which fall back to raw CSV if the
# canonical lake hasn't been built yet:
from leonia_traffic.data import za_streets_loader as za_ld
df = za_ld.load_za_main_cached()
```

GeoParquet files also open directly in QGIS (>=3.32) and DuckDB-spatial
without any Python.

---

## Directory layout

```
data/processed/
├── streetlight/              # canonical, raw-aligned tables (one per StreetLight CSV family)
│   ├── _manifest.json        # build metadata for every file in this folder
│   ├── streetscanner_segments.parquet
│   ├── bridge_od.parquet
│   ├── bridge_od_zones.parquet
│   ├── bridge_attributes.parquet
│   ├── congestion_links.parquet
│   ├── congestion_zones.parquet
│   ├── za_volume.parquet
│   ├── za_trips.parquet
│   ├── za_home_distance.parquet
│   ├── za_home_zips_top.parquet
│   ├── za_home_state.parquet
│   ├── za_work_distance.parquet
│   ├── za_work_block_groups.parquet
│   ├── za_tourist_summary.parquet
│   ├── za_line_shapes.parquet      (GeoParquet)
│   └── za_polygon_shapes.parquet   (GeoParquet)
└── derived/                  # downstream analytics tables
    ├── _manifest.json
    ├── cutthrough_index.parquet
    ├── hourly_profiles.parquet
    ├── peak_intensity_am.parquet
    └── peak_intensity_pm.parquet
```

`_manifest.json` records, for each file: row count, column count, file size, build timestamp (UTC), source paths consumed, and whether the file is a GeoParquet. Read it programmatically with `json.load(open("data/processed/streetlight/_manifest.json"))`.

---

## Conventions used across every file

These columns appear with the same meaning in many tables — they're listed once here instead of repeated in every section.

| Column | Type | Meaning |
|---|---|---|
| `zone_name` | str | StreetLight zone name, always `"<Street Name> / <OSM way id>"` (Bridge OD, ZA) or `"<Street Name> / <OSM way id> / <split #>"` (Street Scanner). Use as a join key wherever it appears. |
| `street_name` / `osm_name` / `road_name` | str | Parsed-out street name from `zone_name`. The aliases exist because the raw exports name this column differently; we preserve the source label per file. |
| `osm_way_id` | Int64 (nullable) | OSM way id parsed from `zone_name`. Join key for any OSM-based analysis (e.g. OSMnx network). |
| `day_type_raw` / `day_part_raw` | str | The original `"0: All Days (M-Su)"` / `"00: All Day (12am-12am)"` strings from the export. Preserved verbatim for traceability. |
| `day_type_code` / `day_part_code` | Int64 | Parsed integer part of the above. **Use these for filtering.** Codes: `day_type_code` 0=All Days, 1-7=Mon..Sun (the export omits Friday, code 5 is Saturday, 6 is Sunday — see caveat below). `day_part_code` 0=All Day, 1=12am-1am, 2=1am-2am, ..., 24=11pm-12am. |
| `day_type_label` / `day_part_label` | str | Human-readable label split out of `*_raw`. |
| `data_periods` | str | Date range covered by the export, e.g. `"Apr 01, 2025 - Mar 31, 2026"`. |
| `mode_of_travel` | str | Vehicle-mode of the underlying StreetLight metric, e.g. `"All Vehicles CVD Plus - StL All Vehicles Volume"`. |
| `filter` | str | `"Visitors"` or `"Residents"` — StreetLight's home/work-based segmentation. A *Visitor* on a Leonia residential street is the pass-through cut-through signal; a *Resident* drives that street as their home street. ZA files include both. |
| `intersection_type` | str | Almost always `"Trip All"` for our exports. StreetLight will sometimes split into "Trip Start" / "Trip End"; the Leonia exports do not. |

> **Friday caveat.** StreetLight's day-type codes 1..4 in this export are Mon..Thu. There is **no Friday** day_type — it gets rolled into the All-Days aggregate. Code 5 is Saturday, 6 is Sunday. Several analyses treat Thursday (code 4) as the canonical "typical weekday."

---

## Canonical datasets — `data/processed/streetlight/`

### `streetscanner_segments.parquet` (GeoParquet)

**Source:** `streetlight/` root + `streetlight/weekdays/` + `streetlight/weekend/` (StreetLight Street Scanner exports).
**Grain:** one row per (zone × source × day_type × day_part). Zones are short subsegments of named OSM ways; a single street name can have many split rows.
**Used by:** `scripts/02_build_network.py`, `scripts/03_calibrate.py`, all Pass-B simulation scoring.

| Column | Type | Notes |
|---|---|---|
| `zone_name` | str | `"<Name> / <osm_way_id> / <split_index>"` |
| `shp_road_class` | str | Road class as reported in the shapefile (e.g. `residential`, `tertiary`). |
| `road_class` | str | Road class as reported in the CSV (usually identical). |
| `road_name` / `osm_name` | str | Street name (alias of `street_name`). |
| `direction_deg` | float64 | Compass bearing of the zone. |
| `zone_direction_deg` | float64 | Same, rounded to whole degrees. |
| `shp_is_bidi` / `is_bidi` | bool | True if the zone is measured bidirectionally. |
| `geometry` | LineString | WGS84 LineString of the zone. |
| `city_county_state` | str | e.g. `"Leonia, Bergen, New Jersey"`. |
| `day_type` | str | `"All Days"`, `"Weekday"`, `"Weekend"`, `"Mon"`..`"Sun"`. |
| `day_part_raw` | str | `"All Day"` for daily exports; semicolon-joined list (e.g. `"AM Peak; PM Peak"`) when multiple parts were selected. |
| `speed_limit_mph` | float64 | Posted speed limit (may be `NaN`). |
| `avg_speed_mph` | float64 | Average measured speed across the period. |
| `avg_volume` | int64 | Average daily volume (vehicles/day) over the period. |
| `osm_way_id` | Int64 | Parsed from `zone_name`. |
| `split_index` | Int64 | Subsegment index within the OSM way. |
| `source` | str | Canonical label: `all_days`, `weekdays`, `weekend`. |
| `source_folder` | str | Absolute path of the export folder. |
| `filter_data_periods` / `filter_day_types` / `filter_day_parts` | str | Raw `Filters.txt` contents preserved for traceability. |

---

### `bridge_od.parquet`

**Source:** `streetlight/2036064_Destinations/*_od_all.csv` (StreetLight Destinations analysis 2036064, Apr 2025 – Mar 2026, 24-hour day parts).
**Grain:** one row per (origin × destination × day_type × day_part). With 24 hourly day-parts × 8 day-types × ~26 origin/destination pairs that's ~4,900 rows.
**Used by:** `scripts/05_bridge_od_report.py`, `scripts/07_bridge_od_report.py`, `leonia_traffic.sumo.demand_builder`.

> **Schema note:** This dataset replaces the legacy `streetlight/bridge_destination/` export (analysis 2034043) which used 5 fixed windows (Early AM / Peak AM / Mid-Day / Peak PM / Late PM). The new export uses **24 hourly windows** (`day_part_code` 1–24, each covering hour `[code-1, code)`), plus code 0 = All-Day total. See `BRIDGE_OD_WINDOWS` (24 entries) and `BRIDGE_OD_HOUR_RANGES` (named ranges like `PeakAM = [7, 8, 9, 10]`) in `leonia_traffic.sumo.demand_builder` for the canonical mapping.

| Column | Type | Notes |
|---|---|---|
| `origin_zone` / `destination_zone` | str | Full `zone_name`. |
| `origin_label` / `destination_label` | str | Parsed street name. |
| `origin_osm_way_id` / `destination_osm_way_id` | Int64 | Parsed OSM ids. |
| `origin_pass_through` / `destination_pass_through` | str | `"yes"` if the zone is a pass-through measurement. |
| `origin_direction_deg` / `destination_direction_deg` | int64 | Compass bearing of each zone. |
| `origin_bidi` / `destination_bidi` | str | `"yes"`/`"no"`. |
| `day_type_code` / `day_type_label` / `day_part_code` / `day_part_label` | per conventions | |
| `od_volume` | int64 | **Average daily O-D traffic** in this day-type × day-part. |
| `origin_total_volume` | int64 | Total daily volume at the origin (sum across all destinations). |
| `destination_total_volume` | int64 | Total daily volume at the destination. |
| `avg_travel_time_sec` | float64 | Average travel time from origin to destination, in seconds. May be NaN for low-sample OD pairs. |

---

### `bridge_od_zones.parquet` (GeoParquet)

**Source:** `streetlight/2036064_Destinations/Shapefile/*_origin_zones.zip` + `*_destination_zones.zip` merged.
**Grain:** one row per zone (14 total: 7 origin + 7 destination, both shapefiles include all 7 zones).

| Column | Type | Notes |
|---|---|---|
| `name` | str | StreetLight zone name. |
| `zone_role` | str | `"origin"` or `"destination"` — which shapefile the row came from. |
| `direction` | float64 | Compass bearing. |
| `is_pass` / `is_bidi` | int64 | 0/1 flags. |
| `road_type` | str | OSM-style road type when present, `"N/A"` otherwise. |
| `geom_len` | float64 | Geometry length in source units. |
| `gate_lat` / `gate_lon` / `gate_width` | str | Gate metadata (mostly `"N/A"`). |
| `geometry` | Polygon | WGS84 polygon. |

---

### `bridge_attributes.parquet`

**Source:** Six attribute CSVs under `streetlight/2036064_Destinations/` (`*_traveler_*` + `*_od_trip_all.csv`) joined onto the OD key.
**Grain:** one row per (origin × destination × day_type × day_part). Same key as `bridge_od.parquet`.
**Used by:** `scripts/07_bridge_od_report.py` for equity / household / income narratives.

Includes every column in `bridge_od.parquet` (joined on the OD key) plus 140+ attribute columns. The attribute columns are prefixed with their kind so you can filter them quickly:

| Prefix | Source CSV | Examples |
|---|---|---|
| `trip_purpose::` | `*_od_traveler_trip_purpose_all.csv` | `Home to Work`, `Home to Other`, `Non-Home Based Trip` (3 cols, sum to 1.0) |
| `equity::` | `*_od_traveler_equity_all.csv` | Race, ethnicity, foreign-born, English proficiency, disability shares (16 cols) |
| `household::` | `*_od_traveler_household_all.csv` | Kids, tenure, vehicles, unit structure (18 cols) |
| `income::` | `*_od_traveler_education_income_all.csv` | 16 income brackets + 7 education levels (23 cols) |
| `employment::` | `*_od_traveler_employment_all.csv` | 12 industries + 6 worker-class categories (18 cols) |
| `trip_stats::` | `*_od_trip_all.csv` | Trip-length / travel-time / speed / circuity bins (60+ cols) |

All `*::*` columns are float64 in [0, 1], representing the **share** of trips in that bracket for that OD pair × day-type × day-part. The 5 ACS-derived prefixes (`equity::`, `household::`, `income::`, `employment::`, `trip_purpose::`) sum to ~1.0 within each prefix; `trip_stats::` columns are independent distributions.

> **PII note.** Equity and income columns are *ACS small-area imputations* attached to each OD pair, not data about individual trips. They describe the population that *typically* makes that trip, not who actually drove on a given day.

---

### `congestion_links.parquet`

**Source:** `streetlight/congestion/*_link_metrics_all.csv` (StreetLight Congestion Trends).
**Grain:** one row per (link × day_type × day_part). `zone_name` here is `"<osm_road_type> / <osm_way_id>"` because the Congestion product names zones by class, not by street name.
**Used by:** `scripts/05_congestion_report.py`, `scripts/06_scenarios.py` (calibration target).

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `osm_name` / `osm_way_id` | per conventions | `osm_name` is usually the OSM road class (e.g. `tertiary`). |
| `road_class` | str | StreetLight-assigned class: `Local`, `Secondary`, `Primary`, etc. |
| `length_mi` | float64 | Link length in miles. |
| `direction_deg` | int64 | Compass bearing. |
| `is_pass_through` / `is_bidi` | str | `"yes"`/`"no"`. |
| `day_type_code` / `day_part_code` etc. | per conventions | |
| `avg_daily_volume` | int64 | Average daily volume in that day-type × day-part. |
| `avg_speed_mph` | float64 | Average measured speed. |
| `avg_travel_time_sec` | float64 | Average per-link travel time. |
| `free_flow_speed_mph` | float64 | StreetLight's free-flow reference speed (used to compute TTI). |
| `vmt` | float64 | Vehicle-miles travelled in the day-part. |
| `vhd` | float64 | Vehicle-hours of delay relative to free-flow. |
| `tti` | float64 | **Travel Time Index** — mean travel time ÷ free-flow travel time. `1.0` = free-flow, `>1.0` = slower than free-flow. |
| `tti_80` / `tti_90` | float64 | 80th / 90th-percentile TTI (variability). |
| `buffer_index` | float64 | (TTI_95 − TTI_50) / TTI_50; higher = less reliable. |
| `planning_time_index` | float64 | TTI_95 / free-flow TT. The travel time you'd need to budget to arrive on time 95% of the time. |
| `reliability_level` | float64 | StreetLight-internal reliability score; lower is better. |
| `is_reliable` | bool | Convenience flag — `reliability_level <= 1.5`. |
| `speed_p05` … `speed_p50` | float64 | 5th, 10th, 20th, 50th-percentile observed speeds. |

---

### `congestion_zones.parquet` (GeoParquet)

**Source:** `streetlight/congestion/Shapefile/*_congestion_zones.zip`.
**Grain:** one row per zone (136 zones — primarily classes/segments inside Leonia's study area).

| Column | Type | Notes |
|---|---|---|
| `segment_id` | float64 | StreetLight's internal segment id. |
| `name` / `osm_name` / `osm_way_id` | per conventions | |
| `segment_ty` | str | OSM segment type (e.g. `tertiary`, `secondary`). |
| `geometry` | LineString | WGS84 LineString. |

---

### Network Performance — `network_performance_*.parquet`

**Source:** `streetlight/2038116_leonia_network/` (StreetLight Network Performance, analysis 2038116).
**Why it exists:** the broadest segment-level product in the lake. Unlike Congestion Trends (arterials only) it covers **every selected OSM segment** — arterials, the GWB approach, *and* residential blocks — at **hourly** day-parts and **per-day-of-week** day types, so it supplies peak-hour volumes for calibration and a residential speed/volume layer.
**Zone-name format:** the 3-part OSM Derivative form `"<name> / <osm way id> / <split #>"` (e.g. `"1st Street / 1007650684 / 1"`). The loader parses the **middle** number as `osm_way_id` and the trailing number as `split_num` — do **not** reuse the 2-part `parse_bridge_zone_name` here.

#### `network_performance_segments.parquet` — main metrics table

**Grain:** one row per (zone × day_type × day_part). 8 day types (`0` = All Days, `1`–`7` = Mon–Sun), 25 day parts (`0` = All Day, `1`–`24` = clock hours). ~160k rows.

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `street_name` / `osm_way_id` | per conventions | `osm_way_id` parsed from the middle of the 3-part name. |
| `split_num` | Int64 | OSM-segment split index (the trailing number in the zone name). |
| `length_mi` | float64 | Segment length in miles. |
| `is_pass_through` / `is_bidi` | str | `"yes"`/`"no"`. |
| `direction_deg` | float64 | Compass bearing of the segment. |
| `day_type_code` / `day_type_label` | per conventions | `0`–`7`. |
| `day_part_code` / `day_part_label` | per conventions | `0`–`24` (hourly). |
| `inferred_volume` | str | `"true"` if the volume was inferred from nearby segments rather than directly observed. |
| `avg_daily_volume` | int64 | Average daily segment traffic (StL Volume) for that day-type × day-part. |
| `avg_speed_mph` | float64 | Average measured segment speed. |
| `avg_travel_time_sec` | float64 | Average per-segment travel time. |
| `free_flow_speed_mph` | float64 | Max average hourly speed observed in the data period. |
| `free_flow_factor` | float64 | Avg speed ÷ free-flow speed, 0–1. |
| `congestion` | float64 | Derived: `1 − free_flow_factor`. |
| `vmt` | float64 | Vehicle-miles of travel in the day-part. |
| `vhd` | float64 | Vehicle-hours of delay relative to free-flow. |
| `speed_p05` / `speed_p15` / `speed_p85` / `speed_p95` | float64 | 5th/15th/85th/95th-percentile observed speeds (the 85th is the standard speeding/design-speed reference). |

#### `network_performance_prediction.parquet`

**Source:** `*_seg_prediction_interval.csv`. **Grain:** one row per (zone × day_type × day_part).

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `street_name` / `osm_way_id` / `split_num` | per conventions | |
| `avg_daily_volume` | int64 | Point estimate of average daily volume. |
| `pred_lower_95` / `pred_upper_95` | int64 | Lower/upper bound of the 95% prediction interval around `avg_daily_volume`. |

#### `network_performance_zones.parquet`

**Source:** `Analysis Details/*_zones.csv`. **Grain:** one row per zone (815). Roster with StreetLight fingerprints (`fingerprint1`/`fingerprint2`) for de-duplication, plus the parsed `street_name` / `osm_way_id` / `split_num` and `length_mi`.

#### `network_performance_monthly.parquet` ⚠ large

**Source:** `*_seg_monthly_metrics.csv` (~560 MB CSV → ~2.4M rows). Same schema as `network_performance_segments.parquet` plus a `year_month` column (`"2025-01"` … `"2026-04"`). Skip with `--skip network_performance_monthly.parquet`. Note: the monthly file's `vmt`/`vhd`/speed-percentile columns are `N/A` in the raw export, so those land as null.

#### `network_performance_shapes.parquet` (GeoParquet)

**Source:** `Shapefile/*_segment_line.zip`. **Grain:** one row per segment (815). LineString geometry with parsed `street_name` / `osm_way_id` / `split_num`, plus `road_type`, `direction`, `geom_len`, and the gate lat/lon/width attributes (`"N/A"` for OSM-derived segments).

---

### ZA — `za_*.parquet`

The Zone Activity export covers **OSM tertiary segments inside Leonia** with both Visitor (pass-through) and Resident measurements. Ten files share a common key of `(zone_name, day_type_code, day_part_code, filter)` — different tables expose different attribute cross-tabs.

#### `za_volume.parquet` — main volume table

**Source:** `streetlight/2034227_leonia_streets/*_za_all.csv`.
**Grain:** one row per (zone × day_type × day_part × filter).
**Used by:** `scripts/09_leonia_streets_report.py` and the derived cut-through index.

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `street_name` / `osm_way_id` | per conventions | |
| `pass_through` | str | `"yes"`/`"no"`. |
| `direction_deg` | int64 | Compass bearing of the segment. |
| `bidi` | str | `"yes"`/`"no"`. |
| `filter` | str | `"Visitors"` or `"Residents"`. |
| `day_type_code` / `day_part_code` etc. | per conventions | |
| `zone_volume` | int64 | **Average daily volume** for this combination — i.e. how many of the filtered cohort's vehicles pass through this zone on a typical day_type/day_part. |
| `avg_travel_time_sec` | int64 | Average travel time through the zone (seconds). |
| `avg_all_travel_time_sec` | int64 | Including the full trip (origin to destination), not just the zone crossing. |
| `avg_trip_length_mi` | float64 | Average trip length **for trips that crossed this zone**. |
| `avg_all_trip_length_mi` | float64 | Average full-trip length. |

#### `za_trips.parquet` — trip-attribute distributions

**Source:** `streetlight/2034227_leonia_streets/*_zone_trip_all.csv`.
**Grain:** same as `za_volume.parquet` (it's an extended sibling).

In addition to all columns above, this table includes:

- `tt_min_0_10`, `tt_min_10_20`, …, `tt_min_150_plus` — travel-time bins (16 cols, share in each 10-min bucket).
- `len_mi_0_1`, `len_mi_1_2`, `len_mi_2_5`, …, `len_mi_100_plus` — trip-length bins (14 cols).
- `spd_mph_0_10`, `spd_mph_10_20`, …, `spd_mph_70_plus` — observed-speed bins (8 cols).
- `circuity_1_2`, `circuity_2_3`, …, `circuity_6_plus` — circuity bins (6 cols; ratio of trip length to straight-line distance).
- `avg_trip_speed_mph`, `avg_all_trip_speed_mph` — additional summary speeds.

Each `*_bin` value is a share in [0, 1]; the bins for a given metric (e.g. all `tt_min_*` columns) sum to ~1.0 per row.

#### `za_home_distance.parquet`

**Source:** `streetlight/2034227_leonia_streets/Home Work/*_home_distance_all.csv`.
**Grain:** one row per (zone × day_type × day_part × filter). Filter is overwhelmingly `Visitors` here (Residents always live within 0-1 mi of their own street, by definition).
**Used by:** `non_local_home_share()` in `cutthrough_streets.py`.

Adds bin columns (each is a share):
`Percent Home less than 1 mi`, `Percent Home 1 to 3 mi`, …, `Percent Home more than 100 mi` (8 cols, sum to ~1.0).

#### `za_home_zips_top.parquet`

**Source:** `streetlight/2034227_leonia_streets/Home Work/*_home_zip_codes_top_all.csv`.
**Grain:** **Multi-row** per zone — one row for each of the top home ZIP codes contributing to that zone's traffic, pre-ranked by StreetLight.
**Used by:** `visitor_demographics.origin_municipality_breakdown()`.

| Column | Type | Notes |
|---|---|---|
| `zip_code` | int64 | 5-digit ZIP (leading zeros dropped — handle when joining to ACS!). |
| `pct_home_location` | float64 | Share of the zone's filtered trips originating in this ZIP. |
| `rank` | int64 | 1 = top ZIP, 2 = second, etc. |

#### `za_home_state.parquet`

**Source:** `streetlight/2034227_leonia_streets/Home Work/*_home_state_all.csv`.
**Grain:** one row per (zone × day_type × day_part × filter × state). For a typical Leonia residential street, ~90% of Visitor home rows are New Jersey, ~7% New York.

| Column | Type | Notes |
|---|---|---|
| `state_name` | str | Full state name. |
| `pct_home_location` | float64 | Share of filtered trips originating in this state. |

#### `za_work_distance.parquet`

**Source:** `streetlight/2034227_leonia_streets/Home Work/*_work_distance_all.csv`.
**Grain:** same as `za_home_distance.parquet`. Filter here is `Residents` only (StreetLight only publishes a workplace distance for resident drivers).

Adds: `Percent Work less than 1 mi`, `Percent Work 1 to 3 mi`, …, `Percent Work more than 100 mi`.

> **Note.** The work-distance bins do *not* sum to 1.0 — StreetLight excludes the unknown-workplace cohort, so bin sums are typically 0.7–0.9.

#### `za_work_block_groups.parquet` ⚠ large

**Source:** `streetlight/2034227_leonia_streets/Home Work/*_work_block_groups_all.csv` (~2.86M rows, ~140 MB raw CSV, **13 MB** as Parquet).
**Grain:** one row per (zone × day_type × day_part × Census block group). Filter is `Residents` only.
**Used by:** `visitor_demographics.work_destination_breakdown()` (workplace-county aggregation, used as a destination proxy in the Pass-C report).

| Column | Type | Notes |
|---|---|---|
| `block_group_id` | str | 12-digit GEOID20 (state(2) + county(3) + tract(6) + block-group(1)). |
| `state_fips` | str | First 2 digits of `block_group_id`. |
| `county_fips` | str | First 5 digits (state + county). Join key for county-level aggregation. |
| `tract` | str | First 11 digits (state + county + tract). |
| `zone_volume` | int64 | Total daily volume of the (zone × day-type × day-part × filter) row — repeated on every block-group sub-row. |
| `pct_work_location` | float64 | Share of that zone-volume whose workplaces fall in this block group. |

Sub-rows for a given (zone × day_type × day_part) sum to ~1.0 across all block groups.

#### `za_tourist_summary.parquet`

**Source:** `streetlight/2034227_leonia_streets/Home Work/*_tourist_summary_all.csv`.
**Grain:** one row per (zone × day_type × day_part × filter × home-area type).

| Column | Type | Notes |
|---|---|---|
| `Percent Living in State` | float64 | Share of the cohort whose home block-group is in NJ. |
| `Percent Living out of State` | float64 | Complement. |
| `Percent Living in Local Metro Area` | float64 | Share in the NYC-Newark-Jersey City metro area. |
| `Percent Living in Other Metro Area` | float64 | Other metros. |
| `Percent Living in Rural Area` | float64 | Rural areas. |

#### `za_line_shapes.parquet` (GeoParquet)

**Source:** `streetlight/2034227_leonia_streets/Shapefile/*_zone_activity_line.zip`.
**Grain:** one row per zone (375).

| Column | Type | Notes |
|---|---|---|
| `name` | str | Full zone name. |
| `street_name` / `osm_way_id` | per conventions | |
| `direction` | float64 | Compass bearing. |
| `is_pass` / `is_bidi` | int64 | 0/1 flags. |
| `road_type` | str | OSM road type. |
| `geom_len` | float64 | Geometry length in source units. |
| `gate_lat` / `gate_lon` / `gate_width` | str | Mostly `"N/A"`. |
| `geometry` | LineString | WGS84. |

#### `za_polygon_shapes.parquet` (GeoParquet)

**Source:** `streetlight/2034227_leonia_streets/Shapefile/*_zone_activity_polygon.zip`.
**Grain:** one row per zone (375).

Same columns as the line shape file, minus the gate metadata, with `Polygon` geometry instead of `LineString`. Use the polygon shapes for mapping/colouring; use the line shapes for routing/spatial joins to the OSM network.

### `streetscanner_trend.parquet`

**Source:** `streetlight/streetscanner_trend/26658_*.csv` (StreetLight "Trend" export, monthly volume Jan 2023 → present).
**Grain:** one row per `(zone_name, year_month)` — long format. The wide `Change` column from the raw CSV is dropped (recoverable from `last_value` / `baseline_12mo_avg`).
**Used by:** `street_trend.parquet` derivation, accelerating-cut-through recommendation rule.

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `osm_name` / `osm_way_id` / `split_index` | per conventions | |
| `city` / `county` / `state` | object | Split from StreetLight's `City, County, State` field. |
| `road_class` / `road_name` | object | Passthrough. |
| `zone_direction_deg` | float64 | |
| `zone_bidi` | bool | |
| `day_type` / `day_part_raw` | object | Passthrough (single value across this export — "Weekday" / "All Day"). |
| `year_month` | datetime64[ns] | First-of-month. |
| `year` / `month` | Int64 | |
| `avg_volume` | float64 | Average daily vehicle volume that month. |

### `streetscanner_trend_shapes.parquet` (GeoParquet)

**Source:** matching shapefile bundled with the trend export.
**Grain:** one row per zone (same 3,061 zones as the trend table, deduplicated to 99,095 rows after split-index expansion).

`zone_name` joins to the long table. Geometry is `LineString` (EPSG:4326).

### `cutthrough_omd.parquet`

**Source:** `streetlight/2034993_cut_through/` — StreetLight O-D + Middle-Filter Analysis. This is the canonical confirmed-cut-through dataset: every row is a `(origin, middle, destination)` triple with the measured volume.
**Grain:** one row per `(origin × middle × destination × day_type × day_part)`. 67,635 rows in the current export.
**Used by:** `cutthrough_attribution.parquet`, `od_bypass_pairs.parquet`, `omd_confirmed_cutthrough` recommendation rule.

| Column | Type | Notes |
|---|---|---|
| `origin_zone` / `origin_label` / `origin_osm_way_id` | object / Int64 | |
| `middle_zone` / `middle_label` / `middle_osm_way_id` | object / Int64 | The Leonia tertiary street the trip routes through. |
| `destination_zone` / `destination_label` / `destination_osm_way_id` | object / Int64 | |
| `day_type_code` / `day_type_label` | Int64 / object | 0 = All Days, 1 = Mon, …, 7 = Sun. |
| `day_part_code` / `day_part_label` | Int64 / object | 0 = All Day, 1 = Off-Peak AM, 2 = Peak AM, … |
| `omd_volume` | float64 | Vehicles per day on this triple. |
| `avg_travel_time_sec` | float64 | Trip duration averaged across StreetLight observations. |

### `cutthrough_omd_trips.parquet`

**Source:** companion `_od_middle_trip_all.csv` — per-triple trip-attribute distributions.
**Grain:** one row per `(origin × middle × destination × day_type × day_part)`. Same key as `cutthrough_omd.parquet`.

Carries all StreetLight bucket columns (circuity 1-2 / 2-3 / 3-4 / 4-5 / 5-6 / 6+, trip length 0-1 / 1-2 / … / 20+ mi, trip speed 0-15 / 15-30 / 30-45 / 45+ mph), plus four pre-computed shares for convenience:

| Column | Type | Notes |
|---|---|---|
| `share_circuity_ge_3` | float64 | Sum of circuity 3-4, 4-5, 5-6, 6+ shares. Above 0.30 = strong cut-through signature. |
| `share_trip_le_2mi` / `share_trip_ge_5mi` | float64 | Short / long-trip shares. |
| `share_speed_ge_30` | float64 | Speeding share for trips ≥ 30 mph. |
| `avg_trip_length_mi` / `avg_trip_speed_mph` / `avg_trip_duration_min` | float64 | StreetLight provided. |

### `cutthrough_omd_zone_activity.parquet`

**Source:** `_zone_od_middle_all.csv` and `_zone_od_middle_trip_all.csv`.
**Grain:** one row per `(zone × zone_role × day_type × day_part)`.

The `zone_role` column is normalised to one of `origin`, `middle`, `destination` (the raw export uses `middle filter` which is rewritten to `middle`). Useful for "how busy is this zone overall, regardless of pairing".

### `cutthrough_omd_roster.parquet`

**Source:** `Analysis Details/*Zones.csv`.
**Grain:** one row per zone defined in the analysis (67 in the current export).

Lists zone id, label, role, and parsed osm_way_id where applicable. Acts as the lookup table between zone ids and street names.

### `cutthrough_omd_shapes.parquet` (GeoParquet)

**Source:** zipped shapefile bundles under `Shapefile/`.
**Grain:** one row per zone (union of origin, middle, and destination shapes; 67 rows).

`zone_role` column distinguishes the three classes. Geometry is `LineString` (EPSG:4326).

---

## Derived datasets — `data/processed/derived/`

These are recomputed from canonical data by `scripts/00_build_datasets.py --skip-derived` (skip flag) or simply by running the orchestrator with no flags.

### `cutthrough_index.parquet`

**Source:** `za_volume.parquet` + `za_trips.parquet` + `za_home_distance.parquet` + `za_line_shapes.parquet`, scoped to in-borough residential segments.
**Grain:** one row per in-borough Leonia residential segment.
**Used by:** `scripts/09_leonia_streets_report.py`, recommendation engine.

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `street_name` / `osm_way_id` | per conventions | |
| `thursday_volume` | float64 | Mon-Thu day-type Visitor volume. |
| `saturday_volume` | float64 | Saturday Visitor volume. |
| `weekday_weekend_ratio` | float64 | `thursday_volume / saturday_volume`. |
| `weekday_all_day_volume` | float64 | Mean of Mon-Thu Visitor volumes. |
| `long_trip_share_5mi` / `long_trip_share_10mi` | float64 | Share of Visitor trips longer than 5 mi / 10 mi (cut-through signature). |
| `avg_trip_length_mi` | float64 | Average Visitor trip length. |
| `speeding_share` | float64 | Share of Visitor trips in speed bins above the posted limit. |
| `posted_speed_mph` | int64 | Posted limit assumed for the segment (default 25 mph for residential). |
| `home_le_1mi_share` / `home_le_3mi_share` | float64 | Share of Visitor trips with home ≤ 1 mi / ≤ 3 mi. |
| `non_local_home_share` | float64 | `1 - home_le_3mi_share`. |
| `cutthrough_index` | float64 | Composite 0..1 index (1 = worst on every sub-metric). |
| `rank` | int64 | 1 = highest index. |

### `hourly_profiles.parquet`

**Source:** `za_volume.parquet` filtered to All-Days day-type.
**Grain:** one row per zone with hourly Visitor-volume columns. Only zones with sufficient sample for hourly disaggregation appear (152 of 375 zones in the current export).

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `street_name` / `osm_way_id` | per conventions | |
| `h00` … `h23` | float64 | Mean Visitor volume in each hour (24 columns). |

### `peak_intensity_am.parquet` / `peak_intensity_pm.parquet`

**Source:** `za_volume.parquet`.
**Grain:** one row per zone with sufficient hourly data.
**Used by:** `scripts/09_leonia_streets_report.py` peak-hour section.

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `street_name` / `osm_way_id` | per conventions | |
| `peak_total` | float64 | Sum of Visitor volume across the peak hours (7-10am for `_am`, 4-7pm for `_pm`). |
| `peak_per_hr` | float64 | `peak_total / 3` (per-hour rate). |
| `baseline_total` | float64 | Same shape, summed across midday (11am-2pm). |
| `baseline_per_hr` | float64 | `baseline_total / 3`. |
| `peak_intensity` | float64 | `peak_per_hr / baseline_per_hr`. NaN when either rate < 5 trips/hr (avoids small-denominator noise). A ratio of 3× or higher is a commuter-shortcut signature. |

### `street_trend.parquet`

**Source:** `streetscanner_trend.parquet`.
**Grain:** one row per Leonia street (3,060 in current export).
**Used by:** `09_leonia_streets_report.py`, `accelerating_cutthrough` recommendation rule.

Summarises the 35-month volume series for each street with both window comparisons and an OLS trend fit.

| Column | Type | Notes |
|---|---|---|
| `zone_name` / `osm_name` / `osm_way_id` | per conventions | |
| `recent_12mo_avg` | float64 | Mean monthly volume over the most-recent 12 months. |
| `baseline_12mo_avg` | float64 | Mean monthly volume over the prior 12 months (months 13-24 back). |
| `yoy_change_pct` | float64 | `(recent - baseline) / baseline × 100`. Primary "is this getting worse?" metric. |
| `trend_slope_per_year` | float64 | OLS slope, vehicles / year, over the full series. |
| `trend_r2` | float64 | OLS goodness-of-fit. Near 1 = clean linear trend; near 0 = noisy / cyclical / no trend. |
| `peak_year_month` / `peak_value` | date / float64 | The single highest month and its value. |
| `last_year_month` / `last_value` | date / float64 | Most recent point. |
| `share_recent_above_baseline_peak` | float64 | Share of recent-window months that exceeded the *worst* baseline month — an "always worse" indicator. |
| `yoy_rank` | int64 | 1 = largest YoY increase. |

### `cutthrough_attribution.parquet`

**Source:** `cutthrough_omd.parquet` + `cutthrough_omd_trips.parquet` (All-Days × All-Day slice).
**Grain:** one row per Leonia middle-filter street (24 in current export — only streets explicitly named in the OMD analysis).
**Used by:** `omd_confirmed_cutthrough` recommendation rule.

Reduces the OMD triple-product table along the **middle-street** axis: for every street, how many vehicles are routed through it, what share are heading to the GWB, what share are taking a heavily-detoured route, and which O-D pair dominates the volume.

| Column | Type | Notes |
|---|---|---|
| `middle_zone` / `middle_label` / `middle_osm_way_id` | object / Int64 | |
| `total_omd_vph` | float64 | Sum of OMD volume on this street across all OD pairs. |
| `n_od_pairs` | int64 | Number of distinct OD pairs that route through here. |
| `bridge_share` | float64 | Share of `total_omd_vph` whose destination matches "George Washington Bridge" / "GWB". |
| `top_origin_label` / `top_destination_label` | object | Origin and destination of the highest-volume OD pair on this street. |
| `top_od_pair_volume` | float64 | Volume of that single pair. |
| `high_circuity_share` | float64 | Volume-weighted share of trips with circuity ≥ 3 (direct cut-through signature). |
| `avg_trip_length_mi` / `avg_trip_speed_mph` | float64 | Volume-weighted means from the trips table. |
| `long_trip_share` / `speeding_share` | float64 | Volume-weighted shares of trips ≥ 5 mi / ≥ 30 mph. |
| `rank` | int64 | 1 = highest total routed volume. |

### `od_bypass_pairs.parquet`

**Source:** `cutthrough_omd.parquet` (All-Days × All-Day slice).
**Grain:** one row per `(origin, destination)` pair with at least 5 routed vph (86 in current export).
**Used by:** `07_bridge_od_report.py` for "which highway closure is driving which cut-through?" diagnoses.

| Column | Type | Notes |
|---|---|---|
| `origin_zone` / `origin_label` | object | |
| `destination_zone` / `destination_label` | object | |
| `total_routed_vph` | float64 | Sum across middle streets — total Leonia-routed volume for this pair. |
| `n_middle_streets` | int64 | How many distinct middle streets the pair touches. |
| `top_middle_label` / `top_middle_vph` | object / float64 | Dominant middle street and its volume. |
| `avg_travel_time_sec` | float64 | Mean across middle-street rows for this pair. |
| `rank` | int64 | 1 = highest total routed volume. |

---

## Crash overlay datasets — `data/processed/crashes/`

These files come from the **NJDOT Crash Data Dashboard**
(Numetric / AASHTOWare Safety) and are joined to the OSM ways
used by the SUMO network. They power the safety panel in the SUMO
stakeholder one-pager and the toggleable crash layer on the
animated map.

### Source: NJDOT Crash Data Dashboard

`scripts/14_build_crash_overlay.py` defaults to pulling rows
**2019–2026** straight from the dashboard's public search API. The
endpoint is anonymous-friendly when you include the dashboard's
`entityToken` query parameter (a signed Numetric token baked into
the public dashboard URL — no PII, no session cookie). The script
cursor-paginates 500 rows at a time using a `range` filter on
`id_cr` (the API caps each call at 500 regardless of any
`size`/`limit` parameter).

```bash
# default — pull from dashboard, full 1,711 rows (~13 MB cached JSON)
venv/bin/python scripts/14_build_crash_overlay.py

# force re-fetch (cache busts data/raw/njdot_dashboard/leonia_dashboard.json)
venv/bin/python scripts/14_build_crash_overlay.py --refresh

# drop Interstate / NJ Turnpike / state-highway crashes
# (the borough's "what happens on local streets" view)
venv/bin/python scripts/14_build_crash_overlay.py --local-only

# legacy fallback: 2017-2022 raw zip path
venv/bin/python scripts/14_build_crash_overlay.py --source zip
```

**Endpoint shape:**

```
POST https://njdot.aashtowaresafety.net/api/dashboards/<dashboard-id>
     /metrics/<metric-id>/search?entityToken=<signed-token>

Body: {
  "filters": [<group filter on Municipality>,
              <range filter on id_cr — used as the page cursor>],
  "sorting": {"direction": "desc", "sortBy": "id_cr"}
}
```

The dashboard's official CSV-export endpoint
(`/api/query/get-download-csv-id`) is paywalled behind authentication
and returns "Anonymous User" errors for public callers; the
read-side `search` endpoint is what the dashboard uses for its own
table view and is not authentication-gated.

The dashboard data offers four major upgrades over the legacy zip
pipeline:

| Feature | Dashboard | Legacy zip |
|---|---|---|
| Years available | 2019–2026 | 2017–2022 |
| Severity scale | KABCO (K/A/B/C/O) | F/I/P |
| Geocoding | ~92 % (raw + NJDOT-snapped fallback) | ~30 % raw + name-based fallback |
| Pre-computed network screening | yes (per-segment AADT, CRC, SHSP tags) | no |

### Data-quality note: the 2022–2023 reporting dip

The yearly counts the dashboard returns for Leonia look uneven —
**305 / 174 / 301 / 159 / 105 / 335 / 261 / 71** for 2019…2026 —
and 2023 in particular looks suspect. This **is not an extraction
bug**; it is a real reporting gap on NJDOT's side. We confirmed it
two ways:

1. The dashboard's own `totalRows` for Leonia is `1,711`, and our
   cursor-paginated pull returns exactly `1,711` unique `id_cr`
   values, with `year` and `dateofcrash` agreeing on every row.
2. The independent **legacy zip** for 2022 (`Bergen2022Accidents.zip`,
   parsed via `parse_crash_table`) returns **159 rows** for
   muni-code `29` (Leonia) — byte-for-byte the same total the
   dashboard reports. The two NJDOT publishing channels agree.

What's actually happening:

- **2022**: the local-jurisdiction reporting volume is roughly half
  of a "normal" year (51 Municipal vs 96–105 in 2019/2021). Likely
  a Leonia PD records-system / CDR-1 submission lag during the
  state-wide rollout of the new crash form.
- **2023**: only **3 of 105 rows** are on local streets — `101` are
  `State Authority` (NJ Turnpike / I-95, filed by NJ State Police
  who never lapsed). The borough essentially missed the calendar
  year of CDR-1 submissions; this matches what NJDOT staff have
  publicly noted about the 2022→2024 form/transmission transition.
- **2024 onwards** snaps back to the long-run baseline, including
  for local streets (97 Municipal in 2024).

So when reading the trend chart, **the partial-year flag for 2022
and 2023 is a *reporting-completeness* flag, not a *data-extraction*
flag** — there is nothing more to recover from the API. If we ever
want a backfill, NJDOT has indicated the missing CDR-1s for those
two years are unlikely to be retrofitted into the dashboard.

### Source: legacy NJDOT raw zips (`--source zip`)

Still supported for historic parity. URL pattern:
`https://www.nj.gov/transportation/refdata/accident/<YEAR>/Bergen<YEAR><Table>.zip`.
Data is fixed-width ASCII; field offsets defined in
`leonia_traffic/data/njdot_crash_loader.py:_CRASH_FIELDS`.

### `njdot_crashes.parquet`

**Grain:** one row per crash. Default municipality filter:
**Leonia Boro**. Typical row count from the dashboard pull: **1,711**
(2019–2026); from the legacy zip pull: ~1,500 (2017–2022).

| Column | Type | Notes |
|---|---|---|
| `crash_id` | object | `<year>_DASH_<id_cr>` (dashboard) or `<year>_<county><muni>_<case>` (legacy). |
| `data_source` | object | `njdot_dashboard` or `njdot_zip`. |
| `year`, `crash_date`, `crash_dow`, `crash_hour` | mixed | Calendar fields. |
| `muni_code`, `muni_name`, `county_code`, `county_name` | object | Verbatim from NJDOT. |
| `severity_code` | object | `K/A/B/C/O` (KABCO, dashboard) or `F/I/P` (legacy). |
| `severity_label` | object | Friendly name (`Fatal`, `Suspected Serious Injury`, …). |
| `epdo` | float64 | EPDO weight: K=542, A=66, B=11, C=11, O=1 (KABCO) or F=542, I=11, P=1 (legacy). |
| `total_killed`, `total_injured`, `ped_killed`, `ped_injured`, `total_vehicles` | Int64 | |
| `ped_involved`, `bike_involved`, `alcohol`, `hazmat`, `at_intersection` | bool | |
| `crash_location`, `cross_street` | object | NJDOT free-text. |
| `road_system` | object | `Municipal` / `County` / `NJDOT State Highway` / `State Authority` / `Interstate` / etc. |
| `latitude`, `longitude` | float64 | Raw NJDOT (~63 % filled in dashboard, ~30 % in zips). |
| `geocoded_lat`, `geocoded_lon` | float64 | After fallback chain. |
| `geocoded_osm_way_id` | Int64 | Snapped OSM way (NaN if no match within `--max-snap-m`, default 50 m). |
| `geocoded_method` | object | `raw` / `njdot_calculated` / `intersection` / `street` / `none`. |
| `shsp_emphasis_areas` | object | Comma-joined NJDOT Strategic Highway Safety Plan tags (dashboard only — e.g. `"Distracted Driving Related, Older Driver Involved"`). |
| `njdot_segment_id`, `njdot_segment_aadt`, `njdot_segment_crc`, `njdot_segment_cpmc`, `njdot_window_crc`, `njdot_intersection_tev` | mixed | Pre-computed network-screening metrics keyed to NJDOT's segment IDs (dashboard only). |
| `crash_type`, `first_harmful_event_label`, `surface_condition`, `light_condition`, `weather`, `road_surface_type`, `functional_class`, `urban_rural`, `posted_speed`, `posted_speed_cross`, … | object/Int64 | Free-text and categorical fields. |

### `crashes_by_segment.parquet`

**Grain:** one row per OSM way that had ≥ 1 geocoded crash. Typical
row count from the dashboard pull: **152** (or **118** with
`--local-only`). Sorted descending by `epdo_total`.

| Column | Type | Notes |
|---|---|---|
| `osm_way_id` | Int64 | Joins back to `streetscanner_segments.parquet` and the SUMO `leonia.edgedata.meta.csv`. |
| `street_name` | object | From the SUMO meta CSV. |
| `n_crashes` | int64 | Total crashes geocoded to this segment. |
| `n_fatal` | int64 | F (legacy) **or** K (KABCO). |
| `n_serious` | int64 | A — *Suspected Serious Injury* (KABCO data only). |
| `n_injury` | int64 | I (legacy) **or** A + B + C (KABCO). |
| `n_pdo` | int64 | P (legacy) **or** O (KABCO). |
| `n_ksi` | int64 | K + A (KABCO) or F + I (legacy). FHWA Killed-or-Seriously-Injured proxy. |
| `n_ped` | int64 | Pedestrian-involved. |
| `epdo_total` | float64 | Sum of the per-row EPDO weights. |
| `first_year`, `last_year`, `years_covered` | int64 | Time window present for this segment. |

### `njdot_pedestrian_crashes.parquet` (legacy zip path only)

**Grain:** one row per pedestrian victim from the Pedestrian
companion table; joins back to `njdot_crashes` on `crash_id`. Only
emitted when `--source zip` is used (the dashboard already encodes
all pedestrian fields directly on the parent crash row, so a
separate table is unnecessary).

| Column | Type | Notes |
|---|---|---|
| `crash_id` | object | Join key. |
| `year`, `case_number`, `muni_code`, `county_code` | mixed | |
| `ped_age` | float64 | |
| `ped_sex`, `ped_position` | object | Best-effort from the layout. |

### Geocoding chain

1. **`raw`** — NJDOT supplied a real `latitude`/`longitude`
   (dashboard: ~63 %; legacy zips: ~30 %).
2. **`njdot_calculated`** — NJDOT's own `Geopoint Calculated`
   fallback, snapped to their network-screening segment centerline
   (dashboard only; +29 % to ~92 % cumulative).
3. **`intersection`** — name-based fallback that takes the midpoint
   of the OSM intersection between `crash_location` and
   `cross_street`. Used when the previous two are blank.
4. **`street`** — last-resort midpoint of the OSM way matching
   `crash_location`.
5. **`none`** — unmatched. Counted in headline totals but not in
   `crashes_by_segment`.

For the current dashboard pull these come out to:
1,072 raw / 504 calculated / 60 intersection / 44 street / **31 unresolved**
(98 % overall placement).

### State-system filter

About 55 % of NJDOT crashes inside Leonia's polygon are on
state-jurisdiction roads (NJ Turnpike, I-95, NJ-93). These belong
to a different governance conversation and visually swamp the
borough-internal cluster on the animated map. The viz layer drops
them by default; `_safety_panel` aggregates *across* all rows so
the council can still see the absolute totals. Pass `--local-only`
to `scripts/14_build_crash_overlay.py` if you want the parquets
themselves restricted to the local-streets subset.

---

## Rebuilding the data lake

```bash
# Full rebuild (canonical + derived)
venv/bin/python scripts/00_build_datasets.py

# Only one product
venv/bin/python scripts/00_build_datasets.py --only za

# Skip the slow files when iterating
venv/bin/python scripts/00_build_datasets.py --skip za_work_block_groups.parquet

# Canonical only (skip derived)
venv/bin/python scripts/00_build_datasets.py --skip-derived
```

The script is idempotent — rerunning overwrites parquets and refreshes the manifest. Roughly 20s total on a quiet machine (95% of that is the 2.8M-row work-block-groups CSV; subsequent runs that skip it complete in ~2s).

---

## Adding a new StreetLight product

The orchestrator uses an **explicit** product list — adding a new product requires a code change (by design, so unexpected folders never silently slip into the lake). Steps:

1. Add a loader module under `leonia_traffic/data/` (mirror the pattern of `bridge_od_loader.py`).
2. Add a filename constant to `CanonicalFiles` in `leonia_traffic/data/dataset_io.py`.
3. Add a `build_<product>()` function to `scripts/00_build_datasets.py` and register it in `KNOWN_PRODUCTS`.
4. Document the new file(s) in a new subsection of this file.

---

## Exporting to Eclipse SUMO

The canonical lake + the derived layer can be exported to a
**self-contained SUMO project** under `data/processed/sumo/` with:

```bash
venv/bin/python scripts/11_export_sumo.py
```

The script writes the following files (all open with `sumo-gui`):

| File | What it is |
|---|---|
| `leonia.osm.xml` | Raw OSM extract for the slightly-widened Leonia bbox (includes the GWB approach). |
| `leonia.net.xml` | SUMO road network, built by `netconvert`. Edges carry the source OSM way ids as a `<param key="origId">` so we can join back to StreetLight measurements. |
| `leonia.poly.xml` | Leonia borough boundary + 375 ZA zone polygons, coloured by composite cut-through index (red = worst, grey = neutral). |
| `leonia.edgedata.xml` | Per-SUMO-edge observed averages from Street Scanner (`entered` = avg daily volume, `speed` = avg observed m/s). Schema-clean against SUMO's `meandata_file.xsd`. Load via File → Open EdgeData in sumo-gui to colour the network by real measurements. |
| `leonia.edgedata.meta.csv` | Metadata sidecar: pairs each `sumo_edge_id` with its `osm_way_id` and `street_name` (SUMO's `meandata` schema doesn't allow string attributes inside `<edge>`, so the names live here). |
| `leonia.flows.xml` | 49 Bridge OD flows across 5 day-part windows (Early AM, Peak AM, Mid-Day, Peak PM, Late PM). |
| `leonia.sumocfg` | Master config: load this with sumo-gui to open everything in one go. |
| `README_SUMO.md` | How to install SUMO + open the project + colour edges by observed data. |
| `_manifest.json` | Build provenance. |

The script auto-detects `netconvert` on PATH or at the macOS framework
install (`/Library/Frameworks/EclipseSUMO.framework`). If SUMO isn't
installed it still writes `leonia.osm.xml` and prints the one-liner to
finish the conversion later.

> **OSM-id drift caveat.** StreetLight's Bridge OD export uses OSM way
> ids from a snapshot that's slightly older than what Overpass returns
> today. The export script handles this with a spatial fallback: any
> Bridge OD zone whose id isn't found in the fresh OSM extract is
> matched to the nearest SUMO edge (within 300 m) by its zone
> geometry. The same trick is used in `calibration_match.py` for the
> UXsim simulation. Result: all 7 Bridge OD zones resolve to SUMO
> edges and all 49 flows route successfully.

## Running SUMO simulations with libsumo

The `leonia_traffic.sumo` package wraps `libsumo` so the static
SUMO project above becomes a real interactive simulator. Two CLI
entry points consume the canonical lake directly:

```bash
# Single-demand baseline + stakeholder one-pager (~30 s wallclock).
venv/bin/python scripts/12_sumo_baseline.py --demand peak_am_slice
venv/bin/python scripts/12_sumo_baseline.py --demand bridge_od_full

# Pass-B/C scenarios mirrored through SUMO + dual-compare maps.
venv/bin/python scripts/13_sumo_scenarios.py
```

Outputs land under `data/processed/sumo/runs/<timestamp>_<label>/`:

| File | What it is |
|---|---|
| `edge_history.parquet` | Long-format per-edge counters at the configured sample interval. |
| `edge_summary.parquet` | One row per SUMO edge with `peak_vph`, `mean_speed_mph`, joined to `osm_way_id` and `street_name`. |
| `scoring.parquet` | GEH per OSM way against `streetscanner_segments.parquet`. |
| `manifest.json` | Run config + headline scores (geh_mean, pct_lt_5, n_links_scored, wallclock). |
| `animated.html` | Self-contained folium time-slider map. |
| `stakeholder.html` | Council one-pager (KPIs, Plotly hourly chart + top-impacted bar, embedded animated map, demographic overlay). |
| `compare.html` *(scenarios only)* | Folium DualMap: baseline left, scenario right, synchronised pan/zoom. |
| `worker_stats.json` | Internal: subprocess-side simulation stats. |

Programmatic API:

```python
from leonia_traffic.sumo import DemandSource, SumoRuntime

with SumoRuntime.start(demand=DemandSource.PEAK_AM_SLICE) as rt:
    rt.run_until(7 * 3600 + 30 * 60)
    before = rt.edge_counters()                 # snapshot
    rt.apply_closure(osm_way_ids=[11586338])    # close mid-peak
    rt.run_to_end()
    history = rt.edge_history()                 # long-format
    summary = rt.edge_summary()                 # per-edge peak vph
```

> **libsumo ↔ pyarrow caveat.** `libsumo`'s C++ binding registers a
> competing pyarrow filesystem-scheme handler at import time, which
> permanently breaks `pd.read_parquet` / `gpd.read_parquet` /
> `df.to_parquet` in the same Python process. The CLIs work around
> this by running the simulation in a subprocess that writes CSV; the
> parent process then reads the CSVs and writes the final parquets.
> Notebook users should mirror that pattern: do all parquet reads
> before importing `libsumo`, or fork a subprocess for the simulation.

### 24-hour weekday/Sunday demand sources

`DemandSource.BRIDGE_OD_WEEKDAY_24H` and
`DemandSource.BRIDGE_OD_SUNDAY_24H` (used by
`scripts/15_sumo_weekday_vs_sunday.py`) are deliberately *composite*
demand sources:

1. Bridge OD flows filtered to the matching `day_type_code` cohort
   (Mon–Fri mean for weekday, Sunday-only for Sunday). These cover
   the gateway-to-gateway arterial backbone.
2. ZA hourly flows synthesised from `za_volume.parquet` filtered to
   the matching ZA `day_type_code` cohort (Mon–Thu mean = codes 1–4
   for weekday; Sunday = code 6) and **scaled by
   `ZA_VISIBILITY_SCALE_DEFAULT` (0.05)**. These are per-segment
   measurements at every ZA-tracked tertiary street, so summing
   them as new OD demand at full scale severely double-counts
   (one trip is observed at multiple ZA segments) and saturates
   the network. The 0.05 multiplier produces ~30k extra vehicles
   over 24 hours — enough for every ZA-tracked residential street
   to see vehicles in the animation, while keeping the simulation
   tractable (~80 s wall-clock per 24-hour run).

Without the ZA component the animated maps only paint the high-vph
arterials (Broad/Grand/Fort Lee Rd) because Bridge OD origins live
exclusively at gateway zones; vehicles route along the shortest
arterial path and never visit residential streets. The scaling
constant is tuned for *visualization fidelity*, not vehicle-count
fidelity — for analytical use of raw ZA Visitor counts, call
`_za_hourly_flows(..., scale=1.0)` or use `DemandSource.ZA_HOURLY`.

### Day-type code schemas: ZA vs. Bridge OD

> **Watch out:** the same `day_type_code` integer means different
> things in the two StreetLight datasets:
>
> | code | ZA volume          | Bridge OD          |
> |------|--------------------|--------------------|
> | 0    | All Days (M-Su)    | All Days (M-Su)    |
> | 1    | Monday             | Monday             |
> | 2    | Tuesday            | Tuesday            |
> | 3    | Wednesday          | Wednesday          |
> | 4    | Thursday           | Thursday           |
> | 5    | **Saturday**       | **Friday**         |
> | 6    | **Sunday**         | **Saturday**       |
> | 7    | _(not present)_    | **Sunday**         |
>
> The ZA dataset has 7 codes (no Friday); the Bridge OD has 8.
> `demand_builder.py` defines two separate constant sets to avoid
> mixing them up: `WEEKDAY_DAY_TYPE_CODES = (1,2,3,4,5)` and
> `SUNDAY_DAY_TYPE_CODE = 7` for Bridge OD; and
> `ZA_WEEKDAY_DAY_TYPE_CODES = (1,2,3,4)` and
> `ZA_SUNDAY_DAY_TYPE_CODE = 6` for ZA.

## Webapp precache datasets — `data/processed/sumo/runs_precache/`

The stakeholder web app (`webapp/`) is **read-only**: every map is a
precomputed JSON artefact served by FastAPI and rendered with deck.gl.
Nothing here is computed at request time. Three offline builders
populate this tree:

```
runs_precache/
├── catalog.json                  # index of everything below (rebuilt on every build)
├── <scenario_key>/               # one per (street × change × demand), e.g.
│   │                             #   willow_tree_road__speed_hump__weekday
│   ├── flow.json                 # ← animated Simulation-tab artefact
│   ├── edge_history.parquet      # long-format per-edge counters (build input)
│   ├── edge_summary.parquet      # per-edge peak vph + names
│   └── manifest.json             # run config + scenario spec + GEH score
├── _baseline__<demand>/          # no-change baseline per demand cohort
│   └── flow.json
├── _overlays/                    # StreetLight measured per-edge hourly vph
│   ├── streetlight_weekday.json
│   └── streetlight_sunday.json
└── _static/                      # ← Static Maps-tab artefacts
    ├── traffic_weekday.json
    ├── traffic_sunday.json
    └── crashes.json
```

Built by, respectively:

- `webapp/scripts/build_precache.py` → per-scenario / baseline `flow.json` + `catalog.json`
- `webapp/scripts/build_streetlight_overlay.py` → `_overlays/`
- `webapp/scripts/build_static_maps.py` → `_static/`

The deployable subset (`catalog.json` + every `flow.json` + `_static/`
+ `_overlays/`, minus the heavy `edge_history.parquet` build inputs) is
staged under `data/webapp/` and copied to the container's mounted
volume at deploy time. See [`webapp/README.md`](../webapp/README.md) for
the build/serve workflow.

### `catalog.json`

| Key | Notes |
|---|---|
| `built_at` | UTC build timestamp. |
| `scenarios` | `{scenario_key: entry}`. Each entry carries `street_name`, `street_slug`, `osm_way_ids`, `change_type`, `demand`, `demand_label`, `flow_json` (relative path or `null`), `animated_html`/`compare_html`/`animated_dual_html` (legacy folium fallbacks, usually `null`), `ok`, `warnings`. |
| `streets` | Street-picker list: `slug`, `name`, `cutthrough_rank`, `osm_way_ids`, `sumo_edge_ids`. |
| `change_types` | `[{value,label}]` — **`closure` + `speed_hump` only.** The one-way change type was dropped because its per-street `netconvert` rebuilds were unstable. |
| `demands` | `[{value,label}]` — `bridge_od_weekday_24h` / `bridge_od_sunday_24h`. |
| `baselines` | `{demand: {demand, demand_label, flow_json, animated_html}}`. |
| `static` | Static Maps index: `traffic.{weekday,sunday}` paths, `crash` path, `crash_years`, `peak_windows`. |

### `<scenario_key>/flow.json` and `_baseline__<demand>/flow.json`

Compact per-link, per-frame vehicles/hour for the animated Simulation
tab. Built from `edge_history` by
`leonia_traffic.sumo.visualizations.build_flow_payload`.

| Field | Notes |
|---|---|
| `meta` | `title`, `frame_minutes` (15), `vmax_vph` (95th-pctile cap for the colour scale), `center`, `zoom`, `n_active_edges`, `n_frames`, `has_baseline`. |
| `frames` | `["00:00","00:15", …]` — one HH:MM label per animation frame. |
| `skeleton` | Grey context polylines (`[[lon,lat], …]`) for every non-active edge, capped. |
| `edges` | `[{id, name, coords:[[lon,lat],…], vph:[per-frame int], base?:[per-frame int]}]`. `base` (the no-change baseline series) appears only on scenario files (`has_baseline=true`) and drives the impacted-street glow. |

### `_static/traffic_<demand>.json`

Average measured StreetLight volumes for the Static Maps **Traffic**
view. Coverage = Leonia (50 m border buffer recovers edge streets like
Bergen Blvd) ∪ the Fort Lee GWB-approach corridor. Geometry is snapped
to SUMO junction centres and collinear unnamed gaps are back-filled so
streets render continuously.

| Field | Notes |
|---|---|
| `meta` | `title`, `demand` (`weekday`/`sunday`), `metric` (`avg_vph`), `vmax_vph` (normalised to Leonia-internal streets only so the GWB approach doesn't wash out local contrast), `center`, `zoom`, `n_edges`, `peak_windows`. |
| `skeleton` | Grey context polylines for out-of-coverage edges. |
| `edges` | `[{id, name, coords, vals:{all_day, peak_am, peak_pm}}]` — average vph per day-part window. |

### `_static/crashes.json`

NJDOT crash points for the Static Maps **Crash** view (the default tab
on page load), filtered to the borough and excluding state-system roads.

| Field | Notes |
|---|---|
| `meta` | `title`, `center`, `zoom`, `n_points`. |
| `years` | Sorted list of years present — drives the Year dropdown. |
| `skeleton` | Grey context polylines. |
| `points` | `[{lat, lon, year, severity (KABCO K/A/B/C/O), severity_label, epdo, ped, label, date}]`. |

### `_overlays/streetlight_<demand>.json`

Per-edge 24-hour measured vph from StreetLight, keyed by SUMO edge id.
Each street's profile comes from its ZA hourly series where available,
else its StreetScanner daily total spread by the typical ZA hourly
shape.

| Field | Notes |
|---|---|
| `built_at` / `demand` | Build timestamp; `weekday` or `sunday`. |
| `hours` | `[0..23]`. |
| `by_edge` | `{sumo_edge_id: {street, source, daily_total, hourly_vph:[24]}}`, where `source` is `za` or `scanner+za_shape`. |
| `summary` | Counts: `n_streets_via_za`, `n_streets_via_scanner`, `n_streets_dropped`, `n_edges_with_data`. |

## Interactive notebooks

Two notebook sets sit on top of the data lake:

- **[notebooks/dev/](../notebooks/dev/)** — code-first companions for analysts.
  - `01_exploration.ipynb` — quick-look on every canonical parquet (< 30 s).
  - `02_scenario_sandbox.ipynb` — `ipywidgets` panel that closes / calms / lane-reduces any OSM way and shows the resulting diversion on a folium map. Built on the pure-Python NetworkX user-equilibrium engine in `leonia_traffic/assignment/`.
  - `03_recommendations.ipynb` — wraps `leonia_traffic.analysis.recommendations` and adds a diversion sanity check that flags >20 % spillover on any other Leonia street.
- **[notebooks/stakeholder/](../notebooks/stakeholder/)** — narrative versions of the same content, code cells tagged `remove-input` so the rendered HTML hides them.

To render the stakeholder set headless (e.g. in CI):

```bash
scripts/run_notebooks.sh stake   # → reports/notebooks/stakeholder/*.html
scripts/run_notebooks.sh dev     # → reports/notebooks/dev/*.html
scripts/run_notebooks.sh         # both
```

The traffic-assignment engine the sandbox is built on:

- Reuses the same `(nodes, links)` tuple `build_or_load_network()` already produces for UXsim — the network is bit-identical.
- Public API in `leonia_traffic/assignment/__init__.py`:
  - `build_assignment_graph(nodes, links)` — NetworkX DiGraph.
  - `bridge_od_to_demand(bridge_od_df, zones_gdf, G, day_type_code, day_part_code)` — Bridge-OD → `{(o_node, d_node): vph}`.
  - `run_ue(G, demand)` — Frank-Wolfe static UE with BPR cost.
  - `validate_against_streetscanner(result, segments)` — GEH validation against the Street Scanner counts.
  - `apply_scenarios_to_graph(nodes, links, scenarios)` — re-uses the existing `Closure / OneWayConversion / LaneReduction / SpeedHumpCalming` DSL from `leonia_traffic.simulation.scenarios`.
- Unit tests live in `tests/test_assignment.py`.

> **Demand caveat.** The assignment library currently feeds on Bridge OD only — i.e. traffic destined for the George Washington Bridge upper / lower decks. It's the strongest single signal for Leonia's cut-through problem, but it under-counts traffic that has nothing to do with the GWB. Treat the resulting diversion numbers as *floor estimates*. A richer demand assembled from the ZA volumes is left for future work.

## Open caveats

- **`zone_id`** is `NaN` across every export — StreetLight populates it for some products but not these. Use `zone_name` or `osm_way_id` as the join key.
- **ZIP codes** in `za_home_zips_top.parquet` are stored as `int64`, so a leading-zero ZIP like `07605` arrives as `7605`. Re-pad before joining to ACS or USPS data.
- **`zone_volume` in `za_work_block_groups.parquet`** is the *zone-level* total, repeated on every block-group sub-row. Don't sum it; weight by `pct_work_location` to allocate.
- **GeoParquet readers.** `geopandas>=0.13`, `pyarrow>=11`, QGIS >=3.32, DuckDB-spatial all read these files natively. Older tooling may need GeoPackage exports — see `dataset_io.write_geodataframe` for the canonical writer (replacing it with a `.gpkg` export is a one-line change).
