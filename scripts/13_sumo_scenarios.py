"""Run the three Pass-B/C mitigation scenarios through the SUMO runtime.

Mirrors :mod:`scripts.08_recalibrate_and_rerun` but uses the libsumo
runtime instead of UXsim. Each scenario produces:

* A ``data/processed/sumo/runs/<ts>_<scenario>/`` directory with
  the standard analyst artefacts (``edge_history.parquet``,
  ``edge_summary.parquet``, ``scoring.parquet``, ``manifest.json``).
* A side-by-side ``compare.html`` against the baseline.
* A ``stakeholder.html`` one-pager.
* A markdown summary at ``reports/scenarios_sumo/<scenario>.md``.

A top-level summary lands at ``reports/13_sumo_scenarios.md``.

Usage
-----

::

    venv/bin/python scripts/13_sumo_scenarios.py
    venv/bin/python scripts/13_sumo_scenarios.py --demand bridge_od_full
    venv/bin/python scripts/13_sumo_scenarios.py --scenarios broad_ave_oneway_southbound

Like ``scripts/12_sumo_baseline.py``, this script spawns a worker
subprocess per simulation to keep ``libsumo`` from poisoning the
parent's ``pyarrow`` install.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Scenario set (mirrors scripts/08_recalibrate_and_rerun.py)
# ---------------------------------------------------------------------------


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


def _scenario_specs() -> dict[str, dict]:
    """Serialisable description of each scenario.

    Worker subprocesses re-hydrate this into ``Scenario`` instances.
    """
    return {
        "broad_ave_oneway_southbound": {
            "type": "OneWayConversion",
            "osm_way_ids": BROAD_AVE_OSM,
            "allowed_bearing_deg": 180.0,
            "tolerance_deg": 90.0,
        },
        "calming_grand_and_fort_lee": {
            "type": "SpeedHumpCalming",
            "osm_way_ids": GRAND_AVE_OSM + FORT_LEE_RD_OSM,
            "free_flow_speed_factor": 0.5,
            "min_free_flow_speed_ms": 4.5,
        },
        "close_west_residential": {
            "type": "Closure",
            "osm_way_ids": HILLSIDE_AVE_OSM + CHRISTIE_HEIGHTS_OSM,
        },
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _build_scenario(spec: dict):
    from leonia_traffic.simulation.scenarios import (
        Closure, OneWayConversion, SpeedHumpCalming,
    )

    t = spec["type"]
    if t == "OneWayConversion":
        return OneWayConversion(
            name=spec.get("name", "scenario"),
            osm_way_ids=spec["osm_way_ids"],
            allowed_bearing_deg=spec["allowed_bearing_deg"],
            tolerance_deg=spec["tolerance_deg"],
        )
    if t == "SpeedHumpCalming":
        return SpeedHumpCalming(
            name=spec.get("name", "scenario"),
            osm_way_ids=spec["osm_way_ids"],
            free_flow_speed_factor=spec["free_flow_speed_factor"],
            min_free_flow_speed_ms=spec["min_free_flow_speed_ms"],
        )
    if t == "Closure":
        return Closure(
            name=spec.get("name", "scenario"),
            osm_way_ids=spec["osm_way_ids"],
        )
    raise ValueError(f"Unknown scenario type: {t}")


def _run_worker(args: argparse.Namespace) -> int:
    """Subprocess entry: simulate one scenario, write CSVs, exit."""
    from leonia_traffic.sumo import DemandSource, SumoRuntime
    from leonia_traffic.sumo.scenarios_sumo import apply_scenarios

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path = Path(args.spec) if args.spec else None
    scenarios = []
    if spec_path is not None and spec_path.exists():
        spec = json.loads(spec_path.read_text())
        scenarios = [_build_scenario(spec)]

    end_t = int(args.end) if args.end else None
    rt = SumoRuntime.start(
        demand=DemandSource(args.demand),
        gui=args.gui,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        end_time_s=end_t,
    )
    applied_log: list[dict] = []
    t0 = time.time()
    try:
        if scenarios:
            applied = apply_scenarios(rt, scenarios)
            applied_log = [
                {
                    "scenario": type(a.scenario).__name__,
                    "n_affected_edges": len(a.affected_edges),
                    "notes": a.notes,
                }
                for a in applied
            ]
        rt.run_to_end()
        history = rt.edge_history()
        summary = rt.edge_summary()
        stats = rt.stats()
    finally:
        rt.close()

    history.to_csv(out_dir / "edge_history.csv", index=False)
    summary.to_csv(out_dir / "edge_summary.csv", index=False)
    stats["worker_wallclock_s"] = time.time() - t0
    (out_dir / "worker_stats.json").write_text(
        json.dumps(
            {"stats": stats, "applied_scenarios": applied_log},
            indent=2, default=str,
        )
    )
    print(f"[worker] {args.scenario_name or 'baseline'} done: {stats}")
    return 0


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def _spawn(args: argparse.Namespace, run_dir: Path,
           scenario_name: str | None,
           spec_path: Path | None) -> int:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--demand", args.demand,
        "--seed", str(args.seed),
        "--sample-interval", str(args.sample_interval),
        "--out", str(run_dir),
    ]
    if args.end:
        cmd += ["--end", str(args.end)]
    if args.gui:
        cmd += ["--gui"]
    if scenario_name:
        cmd += ["--scenario-name", scenario_name]
    if spec_path:
        cmd += ["--spec", str(spec_path)]
    print(f"[parent] spawning {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


def _post_process(args: argparse.Namespace, run_dir: Path,
                  scenario_name: str | None) -> dict:
    import pandas as pd

    history = pd.read_csv(run_dir / "edge_history.csv")
    summary = pd.read_csv(run_dir / "edge_summary.csv")
    worker = json.loads((run_dir / "worker_stats.json").read_text())

    from leonia_traffic.sumo.scoring import score_sumo_run, write_run_outputs

    sumo_score = score_sumo_run(
        summary, day_part=("peak_am" if args.demand in
                           ("bridge_od_peak_am", "peak_am_slice")
                           else "all_day")
    )
    manifest = {
        "demand": args.demand,
        "seed": args.seed,
        "sample_interval_s": args.sample_interval,
        "end_time_s": args.end,
        "scenario": scenario_name,
        "worker": worker,
        "command": " ".join(sys.argv),
    }
    write_run_outputs(
        run_dir,
        edge_history=history,
        edge_summary=summary,
        scoring_df=sumo_score.scoring_df,
        score=sumo_score.score,
        manifest=manifest,
    )
    return {
        "history": history, "summary": summary,
        "score": sumo_score.score, "scoring_df": sumo_score.scoring_df,
    }


def _build_visuals(args: argparse.Namespace,
                   run_dir: Path,
                   baseline_summary,
                   scenario_summary,
                   scenario_history,
                   score_dict: dict,
                   scenario_name: str | None) -> None:
    if args.no_stakeholder:
        return
    from leonia_traffic.sumo.visualizations import (
        build_animated_map,
        build_crash_map,
        build_dual_compare_map,
        build_stakeholder_html,
        load_crash_points_if_available,
        load_crash_segments_if_available,
    )

    animated = run_dir / "animated.html"
    try:
        build_animated_map(
            scenario_history, animated,
            sample_interval_s=args.sample_interval,
        )
    except Exception as exc:
        print(f"[parent] animated map failed: {exc}")
        animated = None  # type: ignore[assignment]

    crash_pts = load_crash_points_if_available()
    crash_seg = load_crash_segments_if_available()
    crash_map_path: Path | None = run_dir / "crashes.html"
    try:
        build_crash_map(crash_pts, crash_map_path, crash_segments=crash_seg)
    except Exception as exc:
        print(f"[parent] crash map failed: {exc}")
        crash_map_path = None

    if scenario_name and baseline_summary is not None:
        try:
            build_dual_compare_map(
                baseline_summary, scenario_summary,
                run_dir / "compare.html",
                title_left="Baseline", title_right=scenario_name,
            )
        except Exception as exc:
            print(f"[parent] dual map failed: {exc}")

    try:
        build_stakeholder_html(
            run_dir / "stakeholder.html",
            edge_history=scenario_history,
            edge_summary=scenario_summary,
            baseline_summary=baseline_summary,
            score=score_dict,
            animated_map=animated,
            crash_map=crash_map_path,
            title=(
                f"Leonia SUMO scenario — {scenario_name}"
                if scenario_name else "Leonia SUMO baseline"
            ),
            subtitle=(
                f"{scenario_summary.shape[0]:,} edges; "
                f"GEH<5 on {score_dict.get('pct_lt_5', 0) * 100:.0f}%"
            ),
            sample_interval_s=args.sample_interval,
        )
    except Exception as exc:
        print(f"[parent] stakeholder HTML failed: {exc}")


def _scenario_md(scenario_name: str, run_dir: Path,
                 baseline_summary, scenario_summary,
                 score_dict: dict) -> Path:
    """Per-scenario markdown report under reports/scenarios_sumo/."""
    import pandas as pd

    md_path = REPO_ROOT / "reports" / "scenarios_sumo" / f"{scenario_name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    rel_run = (
        run_dir.relative_to(REPO_ROOT)
        if run_dir.is_relative_to(REPO_ROOT) else run_dir
    )

    delta = scenario_summary[
        ["sumo_edge_id", "street_name", "osm_way_id", "peak_vph"]
    ].rename(columns={"peak_vph": "scenario_vph"}).merge(
        baseline_summary[["sumo_edge_id", "peak_vph"]].rename(
            columns={"peak_vph": "baseline_vph"}
        ),
        on="sumo_edge_id", how="outer",
    ).fillna(0.0)
    delta["delta_vph"] = delta["scenario_vph"] - delta["baseline_vph"]
    top = delta.reindex(
        delta["delta_vph"].abs().sort_values(ascending=False).index
    ).head(20)

    lines: list[str] = [
        f"# SUMO scenario — `{scenario_name}`",
        "",
        "_Auto-generated by `scripts/13_sumo_scenarios.py`._",
        "",
        "## Calibration",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| GEH mean | {score_dict.get('geh_mean', float('nan')):.2f} |",
        f"| Pct GEH < 5 | {score_dict.get('pct_lt_5', 0) * 100:.1f}% |",
        f"| Links scored | {score_dict.get('n_links_scored', 0):,} |",
        "",
        f"Run artefacts: `{rel_run}/`",
        "",
        "Stakeholder views:",
        "",
        f"- [Animated map](../../{rel_run}/animated.html)",
        f"- [Dual compare](../../{rel_run}/compare.html)",
        f"- [Stakeholder one-pager](../../{rel_run}/stakeholder.html)",
        "",
        "## Top 20 impacted edges (|Δ vph|)",
        "",
        "| Street | OSM way | baseline | scenario | Δ vph |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in top.iterrows():
        label = row.get("street_name") or "—"
        way = (
            int(row["osm_way_id"]) if pd.notna(row.get("osm_way_id"))
            else "—"
        )
        lines.append(
            f"| {label} | {way} | "
            f"{row['baseline_vph']:.0f} | {row['scenario_vph']:.0f} | "
            f"{row['delta_vph']:+.0f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--demand", default="peak_am_slice",
        choices=[
            "bridge_od_full", "bridge_od_peak_am",
            "za_hourly", "bridge_od_plus_za", "peak_am_slice",
        ],
    )
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="Subset of scenario names to run "
                        "(default: all three).")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-interval", type=int, default=60)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--no-stakeholder", action="store_true")
    p.add_argument("--worker", action="store_true",
                   help="Internal: simulate one scenario.")
    p.add_argument("--out", default=None)
    p.add_argument("--scenario-name", default=None)
    p.add_argument("--spec", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.worker:
        return _run_worker(args)

    specs = _scenario_specs()
    if args.scenarios:
        for name in args.scenarios:
            if name not in specs:
                print(f"Unknown scenario: {name}. "
                      f"Choose from {list(specs.keys())}", file=sys.stderr)
                return 2
        specs = {k: specs[k] for k in args.scenarios}

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base_dir = (
        REPO_ROOT / "data" / "processed" / "sumo" / "runs"
        / f"{ts}_baseline"
    ).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print("[parent] === baseline ===")
    rc = _spawn(args, base_dir, scenario_name=None, spec_path=None)
    if rc != 0:
        print(f"baseline failed (exit {rc})")
        return rc
    base_post = _post_process(args, base_dir, scenario_name=None)
    base_score_dict = {
        "geh_mean": base_post["score"].geh_mean,
        "pct_lt_5": base_post["score"].pct_lt_5,
        "n_links_scored": base_post["score"].n_links_scored,
    }
    _build_visuals(
        args, base_dir,
        baseline_summary=None,
        scenario_summary=base_post["summary"],
        scenario_history=base_post["history"],
        score_dict=base_score_dict,
        scenario_name=None,
    )

    summary_rows: list[dict] = []
    for name, spec in specs.items():
        spec_with_name = dict(spec)
        spec_with_name["name"] = name
        spec_path = base_dir.parent / f"{ts}_{name}_spec.json"
        spec_path.write_text(json.dumps(spec_with_name, indent=2))

        scen_dir = (
            REPO_ROOT / "data" / "processed" / "sumo" / "runs"
            / f"{ts}_{name}"
        ).resolve()
        scen_dir.mkdir(parents=True, exist_ok=True)

        print(f"[parent] === {name} ===")
        rc = _spawn(args, scen_dir, scenario_name=name, spec_path=spec_path)
        if rc != 0:
            print(f"{name} failed (exit {rc})")
            continue

        scen_post = _post_process(args, scen_dir, scenario_name=name)
        scen_score = {
            "geh_mean": scen_post["score"].geh_mean,
            "pct_lt_5": scen_post["score"].pct_lt_5,
            "n_links_scored": scen_post["score"].n_links_scored,
        }
        _build_visuals(
            args, scen_dir,
            baseline_summary=base_post["summary"],
            scenario_summary=scen_post["summary"],
            scenario_history=scen_post["history"],
            score_dict=scen_score,
            scenario_name=name,
        )
        md_path = _scenario_md(
            name, scen_dir,
            baseline_summary=base_post["summary"],
            scenario_summary=scen_post["summary"],
            score_dict=scen_score,
        )
        summary_rows.append({
            "scenario": name,
            "geh_mean": scen_score["geh_mean"],
            "pct_lt_5": scen_score["pct_lt_5"],
            "run_dir": str(scen_dir.relative_to(REPO_ROOT))
            if scen_dir.is_relative_to(REPO_ROOT) else str(scen_dir),
            "report": str(md_path.relative_to(REPO_ROOT))
            if md_path.is_relative_to(REPO_ROOT) else str(md_path),
        })

    # Top-level summary
    md_top = REPO_ROOT / "reports" / "13_sumo_scenarios.md"
    md_top.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# SUMO scenarios — summary",
        "",
        "_Auto-generated by `scripts/13_sumo_scenarios.py`._",
        "",
        f"Demand source: `{args.demand}` · Seed: {args.seed}",
        "",
        "## Headline",
        "",
        f"- Baseline GEH mean: {base_score_dict['geh_mean']:.2f} "
        f"(pct<5 = {base_score_dict['pct_lt_5'] * 100:.1f}%)",
        f"- Scenarios: {len(summary_rows)} run",
        "",
        "## Per-scenario reports",
        "",
        "| Scenario | GEH mean | Pct GEH < 5 | Report |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['scenario']}` | {row['geh_mean']:.2f} | "
            f"{row['pct_lt_5'] * 100:.1f}% | [{row['report']}]"
            f"(../{row['report']}) |"
        )
    md_top.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"Top-level report: {md_top}")
    for row in summary_rows:
        print(f"  - {row['scenario']:30s} GEH<5={row['pct_lt_5'] * 100:5.1f}%  "
              f"→ {row['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
