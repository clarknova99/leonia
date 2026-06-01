"""Pass B.4 / Pass C.3: recalibrate and rerun mitigation scenarios with
the OD-driven model.

Run from the repo root:

    venv/bin/python scripts/08_recalibrate_and_rerun.py [--mode v2|v3] [--skip-calibrate]

Modes:

* ``--mode v2`` (default) — Pass-B baseline: Street Scanner observations
  only in the GEH score, no residential observations.

* ``--mode v3`` — Pass-C baseline: Street Scanner *plus* Leonia-streets
  ZA Visitor volumes in the GEH score. Outputs land under
  ``reports/scenarios_v3/`` and the top-level summary becomes
  ``reports/10_scenarios_residential_calibrated.md`` (which also
  includes a v2↔v3 comparison panel when both are available).

Workflow:

1. (optional) Run a Pass-B/Pass-C Nelder-Mead calibration with the
   parameter space ``(od_demand_scale, jam_density_factor,
   intersection_capacity_factor)``. The best params are persisted to
   ``data/processed/calibration_best_params.json`` (and for v3 also to
   ``calibration_best_params_v3.json``).
2. Reuse those parameters as the calibrated baseline and rerun the same
   three sample mitigation scenarios that ``scripts/06_compare_scenarios.py``
   uses, via :func:`run_scenario_v2` (with
   ``include_za_streets_in_match=True`` in v3 mode).
3. Write the top-level summary and per-scenario detail outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from leonia_traffic.analysis.reports import delta_map, write_scenario_report
from leonia_traffic.config import DATA_PROCESSED_DIR, REPORTS_DIR
from leonia_traffic.simulation.calibration import (
    CalibrationParamsV2,
    calibrate_v2,
    calibrate_v3,
)
from leonia_traffic.simulation.scenarios import (
    Closure,
    OneWayConversion,
    SpeedHumpCalming,
    compare_scenarios,
    run_scenario_v2,
)

SCENARIO_V2_DIR = REPORTS_DIR / "scenarios_v2"
SCENARIO_V3_DIR = REPORTS_DIR / "scenarios_v3"
MAPS_DIR = REPORTS_DIR / "maps"
SCENARIO_V2_DIR.mkdir(parents=True, exist_ok=True)
SCENARIO_V3_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)


# Mirror the scenario set in scripts/06_compare_scenarios.py.
BROAD_AVE_OSM = [
    42508899, 42508901, 583818803, 584512144, 865209446,
    962179198, 1057638529, 1057638531, 1119076746, 1120251073,
    1177967222, 1177967224, 1297513620, 1356843744, 1361087818,
    1373551588, 1374082699, 1442271864,
]
GRAND_AVE_OSM = [
    11586081, 11586948, 11586957, 420520108, 420520109,
    420817632, 420817633, 542399384, 680797371, 680826584,
    702064599, 715293377, 954610888, 1072272145, 1086782503,
    1090667465, 1112172807, 1112172808, 1121962657,
    1329371709, 1361684843, 1474436857,
]
FORT_LEE_RD_OSM = [
    11585650, 11585651, 11586243, 61282869, 61282941,
    420815408, 420815409, 583818921, 1382939569, 1382939570,
]
HILLSIDE_AVE_OSM = [
    11587086, 11587103, 11587108, 11587117,
    573554220, 1356843743, 1363923902,
]
CHRISTIE_HEIGHTS_OSM = [11583456, 581438723, 866499450]


SCENARIOS = {
    "broad_ave_oneway_southbound": OneWayConversion(
        name="broad_ave_oneway_southbound",
        osm_way_ids=BROAD_AVE_OSM,
        allowed_bearing_deg=180.0,
        tolerance_deg=90.0,
    ),
    "calming_grand_and_fort_lee": SpeedHumpCalming(
        name="calming_grand_and_fort_lee",
        osm_way_ids=GRAND_AVE_OSM + FORT_LEE_RD_OSM,
        free_flow_speed_factor=0.5,
        min_free_flow_speed_ms=4.5,
    ),
    "close_west_residential": Closure(
        name="close_west_residential",
        osm_way_ids=HILLSIDE_AVE_OSM + CHRISTIE_HEIGHTS_OSM,
    ),
}


def _maybe_recalibrate(args) -> CalibrationParamsV2:
    """Return v2/v3 params, either freshly calibrated or read from disk."""
    mode = args.mode
    cache_name = (
        "calibration_best_params_v3.json" if mode == "v3"
        else "calibration_best_params.json"
    )
    cache = DATA_PROCESSED_DIR / cache_name
    if args.skip_calibrate and cache.exists():
        data = json.loads(cache.read_text())
        if data.get("mode") == mode:
            print(f"Reusing cached {mode} params from {cache}")
            params = data.get("params", {})
            return CalibrationParamsV2(
                od_demand_scale=float(params.get("od_demand_scale", 1.0)),
                jam_density_factor=float(params.get("jam_density_factor", 1.0)),
                intersection_capacity_factor=float(params.get("intersection_capacity_factor", 1.0)),
            )
        print(f"Cached params at {cache} are mode={data.get('mode')!r}; "
              "recalibrating.")

    runner = calibrate_v3 if mode == "v3" else calibrate_v2
    label = "Pass-C (v3)" if mode == "v3" else "Pass-B (v2)"
    print(f"Running {label} calibration (maxiter={args.maxiter}, deltan={args.deltan})")
    result = runner(
        initial=CalibrationParamsV2(),
        duration_hours=args.tmax / 3600.0,
        tmax=args.tmax,
        deltan=args.deltan,
        day_type_code=args.day_type_code,
        day_part_code=args.day_part_code,
        maxiter=args.maxiter,
        observed_to_hourly_factor=0.10 / (args.tmax / 3600.0),
    )
    cache.write_text(json.dumps({
        "mode": mode,
        "params": result.best_params.__dict__,
        "score": result.best_score.__dict__,
    }, indent=2), encoding="utf-8")
    print(f"Saved best {mode} params to {cache}")
    return result.best_params


def _load_v1_baseline_score() -> dict | None:
    """Read the v1 scenario summary (if any) for side-by-side comparison."""
    delta_files = list((REPORTS_DIR / "scenarios").glob("*_delta.parquet"))
    if not delta_files:
        return None
    summary: dict = {}
    for f in delta_files:
        name = f.stem.replace("_delta", "")
        df = pd.read_parquet(f)
        if df.empty:
            continue
        summary[name] = {
            "n_links_changed": int((df["abs_delta_vph"] > 5).sum()),
            "median_abs_delta_vph": float(df["abs_delta_vph"].median()),
            "n_spillover_links": int(df["spillover_flag"].sum()),
        }
    return summary or None


def _load_v2_summary() -> dict | None:
    """Read the v2 scenario summary parquets (for v3-vs-v2 comparison)."""
    delta_files = list(SCENARIO_V2_DIR.glob("*_delta.parquet"))
    if not delta_files:
        return None
    out: dict = {}
    for f in delta_files:
        name = f.stem.replace("_delta", "")
        df = pd.read_parquet(f)
        if df.empty:
            continue
        out[name] = {
            "n_links_changed": int((df["abs_delta_vph"] > 5).sum()),
            "median_abs_delta_vph": float(df["abs_delta_vph"].median()),
            "n_spillover_links": int(df["spillover_flag"].sum()),
        }
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("v2", "v3"), default="v2",
                        help="v2 = Pass-B (default); v3 = Pass-C "
                             "(union ZA-streets residential observations).")
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--deltan", type=int, default=20)
    parser.add_argument("--tmax", type=int, default=4 * 3600)
    parser.add_argument("--day-type-code", type=int, default=1)
    parser.add_argument("--day-part-code", type=int, default=2)
    parser.add_argument("--skip-calibrate", action="store_true",
                        help="Reuse cached best-params JSON if present.")
    args = parser.parse_args()

    mode = args.mode
    scenario_dir = SCENARIO_V3_DIR if mode == "v3" else SCENARIO_V2_DIR
    map_prefix = f"scenario_{mode}"
    include_za = mode == "v3"

    params = _maybe_recalibrate(args)
    print(f"Using params ({mode}): {params.__dict__}")

    run_kwargs = dict(
        duration_hours=args.tmax / 3600.0,
        tmax=args.tmax,
        deltan=args.deltan,
        day_type_code=args.day_type_code,
        day_part_code=args.day_part_code,
        demand_scale=params.od_demand_scale,
        jam_density_factor=params.jam_density_factor,
        intersection_capacity_factor=params.intersection_capacity_factor,
        include_za_streets_in_match=include_za,
    )

    print(f"\nRunning {mode} baseline...")
    baseline = run_scenario_v2(
        scenarios=[], name=f"baseline_{mode}",
        print_mode=0, **run_kwargs,
    )
    print(
        f"{mode} baseline: GEH mean={baseline.score.geh_mean:.2f}, "
        f"pct<5={baseline.score.pct_lt_5 * 100:.1f}% on "
        f"{baseline.score.n_links_scored} links"
    )
    # Per-source breakdown when the v3 frame has both sources tagged.
    if "source" in baseline.scoring_df.columns:
        from leonia_traffic.simulation.calibration import score_simulation_by_source
        by_source = score_simulation_by_source(baseline.scoring_df)
        for src, s in by_source.items():
            print(f"  [{src}] n={s.n_links_scored} mean GEH={s.geh_mean:.2f} "
                  f"pct<5={s.pct_lt_5 * 100:.0f}%")

    summary_rows = [{
        "scenario": f"baseline_{mode}",
        "geh_mean": baseline.score.geh_mean,
        "geh_p85": baseline.score.geh_p85,
        "pct_lt_5": baseline.score.pct_lt_5,
        "n_links_changed": 0,
        "median_abs_delta_vph": 0.0,
        "n_spillover_links": 0,
    }]

    for name, sc in SCENARIOS.items():
        print(f"\nRunning {mode} scenario: {name}...")
        scen = run_scenario_v2(
            scenarios=[sc], name=f"{mode}_{name}",
            print_mode=0, **run_kwargs,
        )

        delta = compare_scenarios(baseline, scen)
        delta_path = scenario_dir / f"{name}_delta.parquet"
        delta.to_parquet(delta_path)

        map_path = MAPS_DIR / f"{map_prefix}_{name}.html"
        delta_map(baseline.world, delta,
                  label=f"Δ veh/h vs. {mode} baseline ({name})").save(str(map_path))

        write_scenario_report(
            baseline_result=baseline,
            scenario_result=scen,
            delta_df=delta,
            out_md=scenario_dir / f"{name}.md",
            map_html=map_path,
        )

        summary_rows.append({
            "scenario": name,
            "geh_mean": scen.score.geh_mean,
            "geh_p85": scen.score.geh_p85,
            "pct_lt_5": scen.score.pct_lt_5,
            "n_links_changed": int((delta["abs_delta_vph"] > 5).sum()),
            "median_abs_delta_vph": float(delta["abs_delta_vph"].median()),
            "n_spillover_links": int(delta["spillover_flag"].sum()),
        })

    if mode == "v2":
        v1_summary = _load_v1_baseline_score()
        write_top_summary(summary_rows, v1_summary, mode="v2")
        print(f"\nDone. Top-level summary at {REPORTS_DIR / '08_scenarios_calibrated.md'}")
    else:
        v2_summary = _load_v2_summary()
        write_top_summary(summary_rows, v2_summary, mode="v3")
        print(
            f"\nDone. Top-level summary at "
            f"{REPORTS_DIR / '10_scenarios_residential_calibrated.md'}"
        )


def write_top_summary(rows: list[dict], prior_summary: dict | None, *, mode: str = "v2") -> None:
    is_v3 = mode == "v3"
    prior_mode = "v2" if is_v3 else "v1"

    if is_v3:
        title = "# Scenario comparison — calibrated baseline with residential ZA (Pass C)\n"
        intro = (
            "_Auto-generated by `scripts/08_recalibrate_and_rerun.py --mode v3`. "
            "The same three sample scenarios are rerun on the **v3 calibrated "
            "baseline**, which adds the Leonia-streets Zone-Activity (Pass C) "
            "Visitor observations to the GEH-scoring frame alongside the "
            "Street Scanner arterial observations. The summary below sits "
            "next to a v2↔v3 delta panel where the v2 baseline has also been "
            "computed._\n"
        )
        scen_subdir = "scenarios_v3"
        out_path = REPORTS_DIR / "10_scenarios_residential_calibrated.md"
    else:
        title = "# Scenario comparison — calibrated (Pass B) baseline\n"
        intro = (
            "_Auto-generated by `scripts/08_recalibrate_and_rerun.py`. The same "
            "three sample scenarios from `scripts/06_compare_scenarios.py` are "
            "rerun on the **v2 calibrated baseline** (real Bridge OD demand + "
            "observed link free-flow speeds). The summary table below sits "
            "next to a delta-vs-v1 panel where applicable._\n"
        )
        scen_subdir = "scenarios_v2"
        out_path = REPORTS_DIR / "08_scenarios_calibrated.md"

    lines = [title, intro]

    lines.append("## Jurisdictional note\n")
    lines.append(
        "> **Broad Avenue (CR 1)**, **Grand Avenue (CR 17/49)**, and "
        "**Fort Lee Road (CR 9 — signed locally as Main Street)** are "
        "Bergen County roads. Leonia has no authority to convert them "
        "to one-way, install speed humps, or modify their geometry or "
        "traffic controls. The scenarios `broad_ave_oneway_southbound` "
        "and `calming_grand_and_fort_lee` are **hypothetical county-"
        "coordination scenarios** that quantify the network effect if "
        "Bergen County were to implement such changes. Only "
        "`close_west_residential` (Hillside Ave + Christie Heights St) "
        "targets streets fully under Leonia's jurisdiction.\n"
    )
    lines.append("")
    lines.append(f"## {mode} baseline + scenarios\n")
    df = pd.DataFrame(rows)
    df["pct_lt_5"] = (df["pct_lt_5"] * 100).round(1).astype(str) + "%"
    cols = ["scenario", "geh_mean", "geh_p85", "pct_lt_5",
            "n_links_changed", "median_abs_delta_vph", "n_spillover_links"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if prior_summary is not None:
        lines.append(f"## {prior_mode} vs. {mode} — same scenarios, different baselines\n")
        lines.append(
            "Per-scenario flow changes are reported in both the prior model "
            f"({prior_mode}) and the {mode} model. Differences indicate where "
            "adding the alternate observation set (OD for v2, residential ZA "
            "for v3) shifts the predicted mitigation impact.\n"
        )
        cmp_cols = ["scenario",
                    f"{prior_mode}_n_links_changed", f"{mode}_n_links_changed",
                    f"{prior_mode}_median_abs_delta", f"{mode}_median_abs_delta",
                    f"{prior_mode}_n_spillover", f"{mode}_n_spillover"]
        lines.append("| " + " | ".join(cmp_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cmp_cols)) + " |")
        for r in rows[1:]:
            name = r["scenario"]
            prior = prior_summary.get(name, {})
            cells = [
                name,
                str(prior.get("n_links_changed", "")),
                str(r["n_links_changed"]),
                f"{prior.get('median_abs_delta_vph', float('nan')):.2f}" if prior else "",
                f"{r['median_abs_delta_vph']:.2f}",
                str(prior.get("n_spillover_links", "")),
                str(r["n_spillover_links"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines.append(f"## {prior_mode} vs. {mode} — comparison not available\n")
        lines.append(
            f"No {prior_mode} delta parquet files were found. "
            "Run the prior pipeline if you want a side-by-side comparison.\n"
        )

    lines.append("## Per-scenario reports\n")
    for r in rows[1:]:
        name = r["scenario"]
        lines.append(
            f"- [{name}]({scen_subdir}/{name}.md) — map at "
            f"`reports/maps/scenario_{mode}_{name}.html`"
        )
    lines.append("")

    lines.append("## Reading the spillover flag\n")
    lines.append(
        "A link is flagged as **spillover** when its baseline flow was <200 "
        "veh/h (typically residential) but the scenario adds >50 veh/h. A "
        "high spillover count means the scenario is pushing cut-through "
        "traffic onto smaller streets instead of eliminating it. "
    )
    if is_v3:
        lines.append(
            "Compared to v2, the v3 model also scores residential tertiary "
            "links against directly-measured Pass-C Visitor volumes, so any "
            "v3-vs-v2 spillover differences indicate links where the prior "
            "calibration was free to overshoot residential flow without "
            "penalty."
        )
    else:
        lines.append(
            "Compared to v1, the v2 model uses real OD volumes for the GWB "
            "approach, so spillover numbers should reflect the actual "
            "diversion that the real morning commuter flow would induce."
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
