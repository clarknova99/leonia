"""Phase 6: run three sample mitigation scenarios and write comparison reports.

Run from the repo root:

    venv/bin/python scripts/06_compare_scenarios.py

Outputs:

    reports/06_scenarios.md                 (top-level summary)
    reports/scenarios/<scenario>.md          (per-scenario detail)
    reports/maps/scenario_<scenario>.html    (folium delta map)
    reports/figures/cutthrough_share_<scenario>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from leonia_traffic.analysis.cutthrough import detect_simulated_cutthrough
from leonia_traffic.analysis.reports import delta_map, write_scenario_report
from leonia_traffic.config import REPORTS_DIR
from leonia_traffic.simulation.scenarios import (
    Closure,
    OneWayConversion,
    SpeedHumpCalming,
    compare_scenarios,
    run_scenario,
)

SCENARIO_DIR = REPORTS_DIR / "scenarios"
MAPS_DIR = REPORTS_DIR / "maps"
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)


# These OSM way IDs are derived from the cached UXsim network; see
# `scripts/02_build_network.py` output. Selections cover Broad Ave,
# Grand Ave, Fort Lee Rd, and a residential cluster between Broad and
# Grand. Refine these once a Leonia polygon is available.
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
        # Broad Ave runs roughly N-S; allow only southbound (~180°).
        allowed_bearing_deg=180.0,
        tolerance_deg=90.0,
    ),
    "calming_grand_and_fort_lee": SpeedHumpCalming(
        name="calming_grand_and_fort_lee",
        osm_way_ids=GRAND_AVE_OSM + FORT_LEE_RD_OSM,
        free_flow_speed_factor=0.5,
        min_free_flow_speed_ms=4.5,  # ~10 mph
    ),
    "close_west_residential": Closure(
        name="close_west_residential",
        osm_way_ids=HILLSIDE_AVE_OSM + CHRISTIE_HEIGHTS_OSM,
    ),
}


# Run parameters intentionally match calibration runs so scores are comparable.
RUN_KWARGS = dict(
    duration_hours=2.0,
    tmax=2 * 3600,
    deltan=20,
    daily_to_peak_factor=0.067,   # best from 5-iter calibration
    gwb_share=0.64,
    min_gateway_volume=566.7,
)


def main() -> None:
    print("Running baseline...")
    baseline = run_scenario(scenarios=[], name="baseline", print_mode=0, **RUN_KWARGS)
    print(
        f"Baseline: GEH mean={baseline.score.geh_mean:.2f}, "
        f"pct<5={baseline.score.pct_lt_5 * 100:.1f}%"
    )

    summary_rows = [
        {
            "scenario": "baseline",
            "geh_mean": baseline.score.geh_mean,
            "geh_p85": baseline.score.geh_p85,
            "pct_lt_5": baseline.score.pct_lt_5,
            "n_links_changed": 0,
            "median_abs_delta_vph": 0.0,
            "n_spillover_links": 0,
        }
    ]

    results: dict[str, object] = {"baseline": baseline}

    for name, sc in SCENARIOS.items():
        print(f"\nRunning scenario: {name}...")
        scen = run_scenario(scenarios=[sc], name=name, print_mode=0, **RUN_KWARGS)
        results[name] = scen

        delta = compare_scenarios(baseline, scen)
        delta_path = REPORTS_DIR / "scenarios" / f"{name}_delta.parquet"
        delta.to_parquet(delta_path)

        map_path = MAPS_DIR / f"scenario_{name}.html"
        delta_map(baseline.world, delta, label=f"Δ veh/h vs. baseline ({name})").save(
            str(map_path)
        )

        write_scenario_report(
            baseline_result=baseline,
            scenario_result=scen,
            delta_df=delta,
            out_md=SCENARIO_DIR / f"{name}.md",
            map_html=map_path,
        )

        summary_rows.append(
            {
                "scenario": name,
                "geh_mean": scen.score.geh_mean,
                "geh_p85": scen.score.geh_p85,
                "pct_lt_5": scen.score.pct_lt_5,
                "n_links_changed": int((delta["abs_delta_vph"] > 5).sum()),
                "median_abs_delta_vph": float(delta["abs_delta_vph"].median()),
                "n_spillover_links": int(delta["spillover_flag"].sum()),
            }
        )

    write_top_summary(summary_rows)
    print(f"\nDone. Top-level summary at {REPORTS_DIR / '06_scenarios.md'}")


def write_top_summary(rows: list[dict]) -> None:
    lines = ["# Scenario comparison\n"]
    lines.append(
        "_Auto-generated by `scripts/06_compare_scenarios.py`. Three sample "
        "mitigation scenarios are compared against the calibrated baseline. "
        "Each row links to a per-scenario detail report._\n"
    )

    lines.append("## Jurisdictional note\n")
    lines.append(
        "> **Broad Avenue (CR 1)**, **Grand Avenue (CR 17/49)**, and "
        "**Fort Lee Road (CR 9 — signed locally as Main Street)** are "
        "Bergen County roads. The Borough of Leonia has **no authority** "
        "to convert them to one-way, install speed humps, or otherwise "
        "modify their geometry or traffic controls. The scenarios "
        "`broad_ave_oneway_southbound` and `calming_grand_and_fort_lee` "
        "are therefore **hypothetical county-coordination scenarios** — "
        "they quantify what the network effect would be *if* Bergen "
        "County chose to implement such changes, and can be used as "
        "evidence when petitioning the county. They are **not** plans "
        "the borough can implement unilaterally. The "
        "`close_west_residential` scenario (Hillside Ave + Christie "
        "Heights St) targets streets fully under Leonia's jurisdiction "
        "and is the only scenario in this report that the borough can "
        "act on directly.\n"
    )
    lines.append("")

    lines.append("## Summary table\n")
    df = pd.DataFrame(rows)
    df["pct_lt_5"] = (df["pct_lt_5"] * 100).round(1).astype(str) + "%"
    cols = [
        "scenario",
        "geh_mean",
        "geh_p85",
        "pct_lt_5",
        "n_links_changed",
        "median_abs_delta_vph",
        "n_spillover_links",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-scenario reports\n")
    for r in rows[1:]:
        name = r["scenario"]
        lines.append(f"- [{name}](scenarios/{name}.md) — map at `reports/maps/scenario_{name}.html`")
    lines.append("")

    lines.append("## Reading the spillover flag\n")
    lines.append(
        "A link is flagged as **spillover** when its baseline flow was <200 veh/h "
        "(typically residential) but the scenario adds >50 veh/h to it. A high "
        "spillover count means the scenario is pushing cut-through traffic onto "
        "smaller streets instead of eliminating it."
    )
    lines.append("")

    (REPORTS_DIR / "06_scenarios.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
