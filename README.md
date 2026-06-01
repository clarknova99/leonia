# Leonia, NJ — traffic analysis and mitigation framework

A reusable Python framework that combines **five StreetLight** data
products with a **UXsim** mesoscopic traffic simulation of Leonia, NJ
and its approaches to the George Washington Bridge:

- **Street Scanner** — segment-level speed/volume averages for
  inferring cut-through patterns from observed data alone.
- **Bridge-destination O-D Analysis** — per-day-part trip volumes from
  the 5 Leonia perimeter gates to the GWB, joined with rich traveler
  attributes (trip purpose, equity, household, income, education,
  employment, trip distribution stats).
- **Congestion Trends** — per-segment hourly Travel Time Index, Buffer
  Index, Vehicle Hours of Delay, and reliability classification on
  arterial / motorway approaches.
- **Leonia Streets Zone Activity** (added in Pass C) — per-zone
  Visitor pass-through volume, travel time, trip length, speed, and
  home-ZIP distributions on **OSM tertiary segments inside Leonia**
  (375 raw segments, 151 named municipal residential blocks after the
  jurisdiction filter).
- **Network Performance** — segment-level volume, speed, VMT, VHD,
  free-flow/congestion, and 5/15/85/95 speed percentiles on **every
  selected OSM segment** (arterials, the GWB approach, *and*
  residential blocks; 815 segments), at **hourly day-parts** and
  **per-day-of-week day types**. Supplies peak-hour volumes for
  calibration and a residential speed/volume layer that Congestion
  Trends lacks. Includes a 95% prediction-interval table and a
  per-month breakdown (Jan 2025 – Apr 2026).

The framework supports:

- exploratory analysis of observed traffic volumes and speeds,
- **evidence-first stakeholder reporting** built directly from the OD,
  congestion, and demographic exports (Pass A),
- a **simulation-first** calibrated baseline that drives UXsim from
  real per-day-part OD demand and observed link free-flow speeds
  (Pass B),
- **per-residential-street cut-through evidence** with composite-index
  ranking, home-ZIP origin breakdown, and residential rules folded
  back into the recommendation engine (Pass C),
- declarative mitigation scenarios (one-way conversions, traffic
  calming, residential closures, lane reductions),
- before/after comparison reports with folium maps and spillover flags,
- a **deck.gl stakeholder web app** (`webapp/`) with a *Static Maps* tab
  (measured StreetLight traffic + NJDOT crashes) and a *Simulation* tab
  that animates precomputed per-scenario SUMO flows over a token-free
  MapLibre basemap. See [`webapp/README.md`](webapp/README.md).

The framework is the Borough of Leonia's Safe Streets for All (SS4A)
data subscription product turned into a reproducible analysis pipeline.

---

## Quick start

```bash
# 1. Activate the Python 3.13 virtualenv.
source venv/bin/activate

# 2. Install dependencies (idempotent).
pip install -r requirements.txt

# Pass-0 (recommended first): build the canonical data lake from raw
# StreetLight exports. Writes parquets under data/processed/streetlight/
# (raw-aligned) and data/processed/derived/ (cut-through index, hourly
# profiles, peak intensity). See docs/DATA.md for the column-level
# schema of every output file.
python scripts/00_build_datasets.py

# Pass-1: Street Scanner exploration (purely data-driven).
python scripts/01_explore_data.py
python scripts/02_build_network.py

# Pass-A: evidence report from Bridge OD + Congestion Trends data.
# Produces reports/07_bridge_od.md with figures, maps, and
# rule-based recommendations.
python scripts/07_bridge_od_report.py
open reports/07_bridge_od.md

# Pass-C: per-residential-street cut-through evidence.
# Reads streetlight/2034227_leonia_streets/ and produces
# reports/09_leonia_streets.md with per-street profiles,
# composite-index ranking, origin municipalities, and maps.
# Caches data/processed/leonia_streets_cutthrough_index.parquet
# which Pass-A picks up to fire residential rules in
# reports/07_bridge_od.md.
python scripts/09_leonia_streets_report.py
python scripts/07_bridge_od_report.py    # re-run to pick up residential rules
open reports/09_leonia_streets.md

# Pass-B: simulation-first. Recalibrate UXsim with real OD demand +
# observed link free-flow speeds, then rerun the three sample scenarios.
python scripts/03_calibrate.py --mode v2 --maxiter 12 --deltan 20
python scripts/08_recalibrate_and_rerun.py --skip-calibrate
open reports/08_scenarios_calibrated.md

# Pass-C simulation: same as Pass-B but with Pass-C ZA-streets
# observations unioned into the GEH-scoring frame (residential coverage).
python scripts/03_calibrate.py --mode v3 --maxiter 12 --deltan 20
python scripts/08_recalibrate_and_rerun.py --mode v3 --skip-calibrate
open reports/10_scenarios_residential_calibrated.md

# Legacy Pass-1 simulation path (placeholder gateway demand):
python scripts/03_calibrate.py --mode v1 --maxiter 15 --deltan 20
python scripts/04_run_baseline.py
python scripts/06_compare_scenarios.py

# Optional: export the same data to Eclipse SUMO for an alternative
# simulator. Produces data/processed/sumo/leonia.{net,poly,flows,
# edgedata,sumocfg}.xml plus README_SUMO.md with the open-in-sumo-gui
# instructions. Requires SUMO installed locally.
python scripts/11_export_sumo.py
sumo-gui -c data/processed/sumo/leonia.sumocfg

# Pass-D: drive a real SUMO simulation through libsumo. Reads the
# canonical lake, builds a routes file, runs the simulation, scores
# it against Street Scanner, and emits a stakeholder bundle (animated
# folium map + Plotly one-pager). The `eclipse-sumo`, `libsumo`,
# `traci`, and `sumolib` PyPI wheels are pulled in via requirements.txt
# so no separate `brew install sumo` is required.
python scripts/12_sumo_baseline.py --demand peak_am_slice
open data/processed/sumo/runs/<latest>/stakeholder.html

# Same engine, applied to the three Pass-B/C scenarios.
python scripts/13_sumo_scenarios.py --demand bridge_od_full
open reports/13_sumo_scenarios.md

# Optional: build the NJDOT crash overlay (2019-2026 from the NJDOT
# Crash Data Dashboard, geocoded to OSM ways). Adds a "Safety overlay"
# panel to the stakeholder one-pager, the animated map's crash layer,
# and the web app's Static Maps crash view. Run once after
# 11_export_sumo.py; downstream consumers auto-detect the parquets.
python scripts/14_build_crash_overlay.py

# Web app (stakeholder map explorer). Build the offline precache, then
# serve with FastAPI. See webapp/README.md for the full workflow.
python webapp/scripts/build_streetlight_overlay.py   # _overlays/ (StreetLight per-edge hourly)
python webapp/scripts/build_precache.py --parallel 8 # per-scenario flow.json + catalog.json
python webapp/scripts/build_static_maps.py           # _static/ (traffic + crash JSON)
venv/bin/uvicorn webapp.app.main:app --reload --port 8000
```

---

## Repository layout

```
leonia/
  .gitignore               # excludes data/, streetlight/, venv/, caches
                           # (regenerable or too large for git)
  streetlight/             # raw StreetLight exports (git-ignored)
    (root)                 # all-days Street Scanner export
    weekdays/              # weekday Street Scanner export
    weekend/               # weekend Street Scanner export
    bridge_destination/    # Bridge-destination OD Analysis
    congestion/            # Congestion Trends export
    2034227_leonia_streets/  # Leonia-streets Zone Activity (Pass C)
    2038116_leonia_network/  # Network Performance (segment volume/speed)
  data/                    # all generated data — git-ignored, rebuilt by the pipeline
    processed/
      streetlight/         # canonical parquet lake (one file per StreetLight CSV family).
                           # Built by scripts/00_build_datasets.py.
                           # See docs/DATA.md for column-level schema.
      derived/             # downstream analytics (cut-through index, hourly profiles,
                           # peak intensity AM/PM). Rebuilt by the same script.
      crashes/             # NJDOT crash overlay (dashboard 2019-2026, geocoded
                           # to OSM ways). Built by scripts/14_build_crash_overlay.py.
      sumo/
        runs/              # per-timestamp libsumo runs (scripts/12,13_sumo_*.py)
        runs_precache/     # web-app precache: catalog.json + per-scenario flow.json
                           #   + _overlays/ + _static/. Built by webapp/scripts/.
    webapp/                # deployable subset of runs_precache/ — copied to the
                           # container's mounted volume at deploy time.
    network/               # cached OSM network, overrides.yaml,
                           # speed_overrides_*.parquet (auto-derived)
  docs/
    DATA.md                # data dictionary: every column in every parquet,
                           # with units, examples, and load snippets.
  leonia_traffic/          # main package
    config.py              # study area, paths, suspect streets
    data/
      dataset_io.py           # canonical parquet IO helpers + manifest (NEW)
      streetlight_loader.py   # Street Scanner discovery + join
      bridge_od_loader.py     # Bridge-destination OD loader (+ load_*_cached)
      congestion_loader.py    # Congestion Trends loader (+ load_*_cached)
      network_performance_loader.py  # Network Performance loader (+ load_*_cached)
      za_streets_loader.py    # Leonia-streets ZA loader (+ load_*_cached)
      od_loader.py            # compatibility shim over bridge_od_loader
      njdot_crash_loader.py   # NJDOT crash fixed-width parser + EPDO scoring +
                              # name-based geocoder fallback (NEW)
    network/
      osm_builder.py            # OSM -> UXsim World, manual overrides,
                                # apply_congestion_overrides (NEW)
      calibration_match.py      # OSM-way match + spatial fallback (NEW)
    simulation/
      demand.py                # gateway + OD + bridge_od demand models
      world_factory.py         # build_baseline / build_calibrated_baseline (v3 opt-in)
      calibration.py           # GEH scoring, Nelder-Mead (v1 + v2 + v3),
                               # per-source GEH breakdown
      scenarios.py             # mitigation scenarios + run/compare (v3 opt-in)
    analysis/
      cutthrough.py            # Street-Scanner cut-through detection
      od_cutthrough.py         # OD-driven cut-through evidence
      cutthrough_streets.py    # per-residential-street ZA analytics (NEW, Pass C)
      visitor_demographics.py  # ZIP→municipality lookup (NEW, Pass C)
      congestion.py            # worst hours, delay hotspots, overrides
      jurisdiction.py          # borough polygon + ownership filter
      equity.py                # demographic + equity exposure
      recommendations.py       # rule-based recommendation engine (+ residential rules)
      reports.py               # markdown + folium scenario reports
    sumo/                      # Pass-D: libsumo runtime + stakeholder viz (NEW)
      net_lookup.py            # OSM way ↔ SUMO edge resolution
      demand_builder.py        # DemandSource enum + routes.xml emitter
      runtime.py               # SumoRuntime — interactive libsumo wrapper
      scoring.py               # GEH vs Street Scanner (post-run)
      scenarios_sumo.py        # adapter from Scenario DSL → SUMO calls
      visualizations.py        # animated map / dual map / sparklines / stakeholder HTML
    viz/
      maps.py                  # folium volume / ratio / TTI / OD-flow maps
  scripts/                 # orchestrators that produce reports
    00_build_datasets.py   # build the canonical + derived data lake (NEW)
    01_explore_data.py
    11_export_sumo.py      # export the data lake to a SUMO-ready project (NEW)
    12_sumo_baseline.py    # Pass-D: libsumo runtime baseline + stakeholder bundle (NEW)
    13_sumo_scenarios.py   # Pass-D: libsumo scenario runner + dual-compare maps (NEW)
    14_build_crash_overlay.py # NJDOT crash overlay builder (Bergen 2017-2022) (NEW)
    02_build_network.py
    03_calibrate.py              # supports --mode v1 | v2 | v3
    04_run_baseline.py
    06_compare_scenarios.py
    07_bridge_od_report.py       # Pass-A evidence report
    08_recalibrate_and_rerun.py  # Pass-B/C simulation rerun (--mode v2|v3)
    09_leonia_streets_report.py  # Pass-C per-residential-street report (NEW)
  webapp/                  # stakeholder map explorer (FastAPI + deck.gl) (NEW)
    app/main.py            # FastAPI routes (serves the page + precache JSON)
    templates/stakeholder.html  # tabbed UI: Static Maps + Simulation
    static/                # deckgl_flow.js, deckgl_static.js, *_picker.js, styles.css
    scripts/
      build_precache.py         # offline per-scenario SUMO flow builder → flow.json
      build_streetlight_overlay.py  # StreetLight per-edge hourly → _overlays/
      build_static_maps.py      # measured-traffic + crash JSON → _static/
    Dockerfile
    README.md              # webapp build/serve/deploy guide
  reports/                 # generated outputs (markdown + figures + maps)
  tests/                   # pytest suite (~75 tests)
```

---

## Three-pass workflow

### Pass A — perimeter OD + arterial congestion evidence (`reports/07_bridge_od.md`)

Driven by `scripts/07_bridge_od_report.py`. Reads the bridge OD export
and congestion trends export and produces a single markdown deliverable
for stakeholders containing:

1. **Headline**: highest weekday Peak-AM volume into the GWB by origin
   gate (typically the 700-800 trips/day Fort Lee Road peak).
2. **Gateway profiles**: weekday/weekend imbalance ratios, peak vs.
   off-peak shares, circuity-based cut-through index.
3. **Trip purpose decomposition**: Home-to-Work / Home-to-Other /
   Non-Home-Based share per gate × day-part.
4. **Demographic profiles**: race, ethnicity, nativity, language,
   disability, household composition, income, education, employment
   industry/class. All attributes included per project policy.
5. **Congestion hotspots**: per-corridor worst-hour TTI, top delay
   hotspots, reliability classification by road class.
6. **Equity exposure**: per-gate flag for foreign-born, English
   limited, low-income, no-vehicle, and renter-occupied shares above
   configurable thresholds.
7. **Recommendations**: a rule-based engine ranks the findings into
   a short, prioritised action list. When the Pass-C per-street index
   has been built (cached at
   `data/processed/leonia_streets_cutthrough_index.parquet`), the
   engine also fires residential-cut-through and residential-speeding
   rules naming specific blocks (Willow Tree, Broad Ave, Schor,
   Pine Hill, Christie Heights, etc.).
8. Three interactive folium maps (OD flows, congestion TTI,
   reliability classification).

The script also caches the per-link speed override table consumed by
Pass B at `data/network/speed_overrides_weekday_peak_am.parquet`.

### Pass B — simulation-first calibrated baseline (`reports/08_scenarios_calibrated.md`)

Driven by `scripts/08_recalibrate_and_rerun.py`. Two phases:

1. **Recalibrate** with Nelder-Mead over the new parameter space:
   - `od_demand_scale` — global multiplier on observed OD volumes
     (should land near 1.0 if StreetLight numbers are well-calibrated).
   - `jam_density_factor` — multiplicative on UXsim's default jam
     density.
   - `intersection_capacity_factor` — multiplicative on each link's
     `capacity_out` (proxies signalized-intersection throughput).
   Results are persisted to
   `data/processed/calibration_best_params.json`.
2. **Rerun mitigation scenarios** (Broad Ave one-way, Grand+Fort Lee
   calming, west-residential closure) on the calibrated baseline via
   `run_scenario_v2`. Outputs land in `reports/scenarios_v2/` and
   `reports/maps/scenario_v2_*.html`. The top-level summary report
   sits next to a side-by-side comparison against the v1 placeholder
   baseline so you can see how the OD-driven model changes each
   scenario's spillover prediction.

### Pass C — per-residential-street evidence + recalibrated simulation

Pass C closes the residential-coverage gap that Pass B's GEH score
ignored. Two deliverables:

1. **`reports/09_leonia_streets.md`** — Driven by
   `scripts/09_leonia_streets_report.py`, this is the per-block
   stakeholder report. It loads the
   `streetlight/2034227_leonia_streets/` Zone Activity export
   (375 OSM tertiary segments), filters to ~150 named municipal
   residential blocks via the `jurisdiction` polygon + name/class
   filters, computes a composite cut-through index (weekday/weekend
   imbalance × non-local home share × long-trip share × speeding-bin
   share × volume), and produces:
   - top-20 ranking and per-street profiles (top 10),
   - origin-municipality bar chart (volume-weighted),
   - trip-length distribution small-multiples,
   - three folium maps (cut-through index, Visitor volume, speeding).
   The same per-street index is cached for the recommendation engine
   and Pass-A regeneration.
2. **`reports/10_scenarios_residential_calibrated.md`** — Driven by
   `scripts/08_recalibrate_and_rerun.py --mode v3`, this re-runs the
   three mitigation scenarios after re-calibrating with the ZA-streets
   observations unioned into the GEH-scoring frame
   (`include_za_streets_in_match=True`). Per-source GEH is logged so
   you can see whether residential coverage was dragging the prior v2
   calibration. The top-level summary compares v2↔v3 spillover counts
   per scenario.

### Pass D — interactive SUMO simulation via libsumo

Pass D replaces the "export only" relationship with Eclipse SUMO with
a real microscopic-simulator runtime driven by `libsumo`. It reuses
the canonical lake (no new StreetLight data), but offers:

- **Interactive control.** `leonia_traffic/sumo/runtime.py` exposes a
  `SumoRuntime` class with `start / step / run_until / run_to_end /
  apply_closure / restore / set_speed / set_traffic_light / close`.
  You can apply a road closure mid-simulation, watch the diversion
  develop, and read per-edge counters live.
- **Selectable demand.** `DemandSource` is an enum with five built-in
  strategies (`bridge_od_full`, `bridge_od_peak_am`, `za_hourly`,
  `bridge_od_plus_za`, `peak_am_slice`). The first two reuse the
  Bridge OD pairs the export script writes today; `za_hourly`
  synthesises additional demand from
  `data/processed/derived/hourly_profiles.parquet`; the slice is a
  fast-iterating 1-hour run that completes in seconds.
- **Stakeholder bundle.** Every CLI run emits a self-contained
  `animated.html` (folium time-slider map) and `stakeholder.html`
  (one-pager with KPIs, hourly volume curve, top-10 impacted streets,
  embedded animated map, and a demographic overlay sourced from
  `bridge_attributes.parquet`). Scenarios additionally produce a
  `compare.html` `DualMap` so a councillor can flip between baseline
  and intervention with synchronised pan/zoom.
- **NJDOT safety overlay.** If you've run
  `scripts/14_build_crash_overlay.py` once, the SUMO bundle picks
  up the resulting parquets automatically. The animated map gains
  a toggleable *NJDOT crashes 2017–22* layer (1.5k+ geocoded crashes,
  colour-coded by severity), and the stakeholder one-pager grows a
  *Safety overlay* panel that ranks the top-EPDO segments and tags
  any that also rank as cut-through corridors in the simulation.

Two CLI orchestrators:

- **`scripts/12_sumo_baseline.py`** runs a single-demand baseline and
  writes `reports/12_sumo_baseline.md` plus the stakeholder bundle
  under `data/processed/sumo/runs/<timestamp>_<demand>/`.
- **`scripts/13_sumo_scenarios.py`** mirrors `08_recalibrate_and_rerun.py`
  but in SUMO: runs the baseline + the three mitigation scenarios
  through `SumoRuntime`, writes per-scenario reports under
  `reports/scenarios_sumo/`, and a top-level summary at
  `reports/13_sumo_scenarios.md`.

> **Process model.** `libsumo`'s C++ binding registers a competing
> pyarrow filesystem-scheme handler at import time, which permanently
> breaks `pyarrow` in the same Python process. The orchestrators
> therefore spawn a worker subprocess for the simulation half (it
> writes plain CSV) and post-process (scoring, parquet output, HTML
> rendering) in the parent. Notebook users can mirror this pattern
> with `subprocess.run` or simply read CSVs back after `rt.close()`.

### Web app — stakeholder map explorer (`webapp/`)

A read-only FastAPI service that turns the simulation and measurement
layers into an interactive deck.gl map for council meetings. Every
artefact is precomputed offline; the server only maps UI selections to
cached JSON. Two tabs:

- **Static Maps** — non-animated, average-of-day views over a MapLibre
  basemap, with map-type-aware controls:
  - *Traffic* (measured StreetLight volumes): Day Type (Weekday or
    Sunday) × Day Part (All Day, Peak AM, Peak PM). Coloured by average
    vph; the colour scale is normalised to Leonia-internal streets so
    the high-volume GWB approach doesn't wash out local contrast.
  - *Crash* (NJDOT 2019–2026): Year (All or a specific year). Day
    Type / Day Part controls are hidden because they don't apply.
    **Crash is the default view on page load.**
  - Coverage is filtered to Leonia (a 50 m border buffer recovers
    edge streets like Bergen Blvd) plus the Fort Lee GWB-approach
    corridor. Edge geometry is snapped to SUMO junction centres and
    collinear unnamed gaps are back-filled so streets render continuous.
- **Simulation** — the animated per-scenario traffic flow. Pick a
  street, a change (close or speed-hump), and a demand cohort (average
  weekday or average Sunday); the page swaps to the precomputed SUMO
  comparison, highlights the selected street, glows impacted streets,
  and shows live vph on hover. With nothing selected it shows the
  baseline weekday simulation.

The offline build is three steps from the repo root (SUMO + the data
tree required for the precache step only):

```bash
python webapp/scripts/build_streetlight_overlay.py   # _overlays/ per-edge hourly vph
python webapp/scripts/build_precache.py --parallel 8 # per-scenario flow.json + catalog.json
python webapp/scripts/build_static_maps.py           # _static/ traffic + crash JSON
venv/bin/uvicorn webapp.app.main:app --reload --port 8000
```

`build_precache.py` crosses the cut-through-ranked streets with the two
change types (**closure, speed_hump**) and two demand cohorts
(weekday, Sunday) — the one-way change type was dropped from the
precache because its per-street `netconvert` rebuilds were unstable.
It is idempotent and resumable; `catalog.json` is rebuilt from on-disk
state on every invocation. The deployable subset is staged under
`data/webapp/`. See [`webapp/README.md`](webapp/README.md) for the full
build/serve/Docker guide and [`docs/DATA.md`](docs/DATA.md#webapp-precache-datasets--dataprocessedsumoruns_precache)
for the artefact schemas.

### Spatial fallback for stale OSM way IDs

The bridge OD, congestion, and ZA-streets exports reference OSM way
IDs that sometimes predate OSM way splits (e.g. Fort Lee Road / 590576).
When the current OSM extract does not contain an OSM way ID listed in
a StreetLight zone name, `apply_bridge_od_demand`,
`apply_congestion_overrides`, and `match_za_streets_to_links` all fall
back to a **nearest UXsim link** spatial match using the zone shapefile
geometry (see
`leonia_traffic.network.calibration_match.spatial_resolve_osm_way_ids`).
This means the pipeline is robust to OSM-vintage drift.

### Pulling additional StreetLight data

The current exports cover Apr 2025 – Mar 2026 (OD) and Jan 2025 –
Mar 2026 (Congestion). Refreshing the data is as simple as dropping a
new export into `streetlight/<product>/` and rerunning the Pass-A and
Pass-B scripts; auto-discovery picks up the new files.

### Per-Day-Part Street Scanner (optional)

The Street Scanner exports in `streetlight/weekdays/` and
`streetlight/weekend/` are still daily averages. To get true
AM-peak vs. PM-peak rows, re-pull **once per Day Part** into folders
such as `streetlight/weekday_peak_am/`, `weekday_peak_pm/`, etc. The
loader auto-discovers any new folder containing a `Filters.txt` plus
`*_streetscanner_*.csv` pair.

---

## Calibration target

Transport-model convention: **GEH < 5 on ≥85% of links** in the peak
hour. The calibrator (`scripts/03_calibrate.py`) runs Nelder-Mead in
one of two modes:

* `--mode v1` (legacy): three parameters of the placeholder gateway
  demand model (`daily_to_peak_factor`, `gwb_share`,
  `min_gateway_volume`).
* `--mode v2`: three parameters of the OD-driven model
  (`od_demand_scale`, `jam_density_factor`,
  `intersection_capacity_factor`). With real OD demand and observed
  link free-flow speeds already in place, this is the recommended
  Pass-B calibration path.
* `--mode v3`: same parameter space as `v2`, but the scoring frame is
  the **union of Street Scanner and Leonia-streets ZA Visitor
  observations**. Per-source GEH is logged each iteration so you can
  see whether residential links are dragging the calibration. Use this
  when you also want the simulation to track flows on Christie Heights,
  Willow Tree, Schor, Pine Hill, etc.

The objective is `geh_mean + 10 * (1 − pct_lt_5)` so the optimizer
favors solutions that lift the percentile metric over those that only
shave the average. The current v2 baseline reaches ≈48 % of links
under GEH 5 — better than the v1 placeholder's ≈55 % but on a much
more credible demand model.

---

## Known limitations

- **UXsim's OSM importer is officially experimental** ([upstream
  notes](https://toruseo.jp/UXsim/docs/notebooks/demo_notebook_04en_OpenStreetMap.html)).
  We patched a node-merging bug in `_postprocess_network` and use a
  conservative merge threshold; review `reports/02_network.md` and tune
  `data/network/overrides.yaml` if the network looks wrong.
- **OD origin zones are tight to 5 corridors only.** Trips entering
  Leonia from other directions (Glenwood, Hillside, Christie Heights,
  etc.) are not in the OD analysis. A second OD request may be needed
  to confirm those are non-trivial.
- **Congestion Trends does not cover residential streets.** Free-flow
  speed overrides therefore only refine arterials and motorway
  approaches; residential speeds remain at UXsim defaults
  (~8.3 m/s ≈ 30 km/h). **Pass C closes the coverage gap on the
  measurement side** — the new `2034227_leonia_streets` ZA export
  measures Visitor volumes, trip lengths, and speeds on 375 residential
  tertiary segments inside Leonia, and the v3 calibration mode uses
  those as additional GEH-scoring observations. The **Network
  Performance** export (`2038116_leonia_network`) extends this further:
  it carries per-segment average/free-flow speed and 5/15/85/95 speed
  percentiles on residential blocks as well as arterials, so it can
  seed residential free-flow speed overrides and supply hourly peak
  volumes for calibration. It is loaded into the canonical lake today;
  wiring it into a calibration mode / report is a follow-up.
- **Demographic attributes are StreetLight estimates** from device
  home locations, not a direct driver survey. Equity tables disclose
  this in the report.
- **Mesoscopic simulation.** UXsim cannot model microscopic phenomena
  like gap acceptance at unsignalized residential intersections, queue
  jumping, or pedestrian conflicts. Use SUMO or Vissim for NACTO-style
  micro-design analysis.
- **No precise Leonia polygon yet.** Gateway selection in the v1
  placeholder uses a band around the configured bbox. The v2 OD model
  bypasses this concern entirely by using explicit OSM-way → UXsim-link
  resolution with spatial fallback.

---

## Adding a new scenario

```python
from leonia_traffic.simulation.scenarios import (
    OneWayConversion, Closure, SpeedHumpCalming, LaneReduction,
    run_scenario, compare_scenarios,
)
from leonia_traffic.analysis.reports import delta_map, write_scenario_report

sc = SpeedHumpCalming(
    name="park_ave_calming",
    osm_way_ids=[11580528, 11580538, 11580547],   # Park Ave OSM ways
    free_flow_speed_factor=0.5,
)
baseline = run_scenario([], name="baseline")
result = run_scenario([sc], name=sc.name)
delta = compare_scenarios(baseline, result)
delta_map(baseline.world, delta).save("reports/maps/park_ave.html")
write_scenario_report(
    baseline_result=baseline, scenario_result=result,
    delta_df=delta, out_md=Path("reports/scenarios/park_ave.md"),
    map_html=Path("reports/maps/park_ave.html"),
)
```

OSM way IDs come from `scripts/02_build_network.py` output — open
`reports/02_network.md`, find the matching links, and copy the
`osm_way_id` values.

---

## Running tests

```bash
venv/bin/python -m pytest tests/ -v
```

The suite covers:

- StreetLight Street Scanner, Bridge OD, Congestion, and Leonia-streets
  ZA loaders;
- OD cut-through analysis (gateway peak imbalance, circuity index,
  day-of-week profile, trip-purpose decomposition);
- per-residential-street cut-through analytics (weekday/weekend
  ratio, long-trip share, speeding share, non-local home share,
  composite index, ZIP→municipality lookup);
- congestion analysis (worst hours, delay hotspots, reliability,
  speed-override extractor);
- jurisdiction filter (borough polygon + ownership exclusion);
- OSM-network postprocessing including unique-name disambiguation;
- the GEH statistic, scenario application, bridge-OD demand
  application, congestion-derived speed overrides, ZA-streets matcher
  (with stale-OSM spatial fallback), and the recommendation engine's
  residential rules.

No live network or simulation calls — runs in under 6 seconds.
