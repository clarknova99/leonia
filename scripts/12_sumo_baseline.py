"""Run a SUMO baseline simulation and emit the stakeholder bundle.

Usage
-----

::

    venv/bin/python scripts/12_sumo_baseline.py
    venv/bin/python scripts/12_sumo_baseline.py --demand bridge_od_full
    venv/bin/python scripts/12_sumo_baseline.py --demand peak_am_slice --no-stakeholder
    venv/bin/python scripts/12_sumo_baseline.py --gui

Outputs (under ``data/processed/sumo/runs/<timestamp>_<demand>/``):

* ``edge_history.parquet`` — long-format counters, one row per
  (sim-time bin × SUMO edge).
* ``edge_summary.parquet`` — one row per edge with peak-vph and
  mean-speed summaries.
* ``scoring.parquet`` — GEH per edge vs Street Scanner observations.
* ``manifest.json`` — run config + headline scores.
* ``animated.html`` — folium time-slider map (always built unless
  ``--no-stakeholder``).
* ``stakeholder.html`` — single-pager council brief embedding the
  animated map, KPIs, and demographic overlay.

Why subprocess?
---------------

libsumo's C++ binding re-registers pyarrow's ``LocalFileSystem``
scheme handler at import time, which permanently breaks
``pyarrow.read_parquet`` in the same Python process. We therefore
run the simulation in a fresh Python subprocess that writes plain
CSV, and post-process (scoring + visualisations) in the parent
where pyarrow still works.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess, talks to libsumo, writes CSVs
# ---------------------------------------------------------------------------


def _run_worker(args: argparse.Namespace) -> int:
    """Subprocess entry point — does all libsumo work, then exits."""
    from leonia_traffic.sumo import DemandSource, SumoRuntime

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_t = int(args.end) if args.end else None
    rt = SumoRuntime.start(
        demand=DemandSource(args.demand),
        gui=args.gui,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        end_time_s=end_t,
    )
    t0 = time.time()
    try:
        rt.run_to_end()
        history = rt.edge_history()
        summary = rt.edge_summary()
        stats = rt.stats()
    finally:
        rt.close()

    # Persist as CSV — pyarrow is broken inside this process.
    history.to_csv(out_dir / "edge_history.csv", index=False)
    summary.to_csv(out_dir / "edge_summary.csv", index=False)
    stats["worker_wallclock_s"] = time.time() - t0
    (out_dir / "worker_stats.json").write_text(
        json.dumps(stats, indent=2, default=str)
    )
    print(f"[worker] simulation done: {stats}")
    return 0


# ---------------------------------------------------------------------------
# Parent — orchestrates the worker + post-processing
# ---------------------------------------------------------------------------


def _spawn_worker(args: argparse.Namespace, run_dir: Path) -> int:
    """Run this script with ``--worker`` set in a subprocess."""
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
    print(f"[parent] spawning worker: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


def _post_process(args: argparse.Namespace, run_dir: Path) -> dict:
    """Read worker CSV outputs, score, visualise, write final artefacts."""
    import pandas as pd

    history = pd.read_csv(run_dir / "edge_history.csv")
    summary = pd.read_csv(run_dir / "edge_summary.csv")
    worker_stats = json.loads((run_dir / "worker_stats.json").read_text())

    # Score against Street Scanner.
    from leonia_traffic.sumo.scoring import score_sumo_run, write_run_outputs

    sumo_score = score_sumo_run(summary, day_part=_demand_to_day_part(args.demand))

    manifest = {
        "demand": args.demand,
        "seed": args.seed,
        "sample_interval_s": args.sample_interval,
        "end_time_s": args.end,
        "gui": args.gui,
        "worker": worker_stats,
        "stakeholder": not args.no_stakeholder,
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

    if not args.no_stakeholder:
        from leonia_traffic.sumo.visualizations import (
            build_animated_map,
            build_crash_map,
            build_stakeholder_html,
            load_crash_points_if_available,
            load_crash_segments_if_available,
        )

        animated_path = run_dir / "animated.html"
        try:
            build_animated_map(
                history, animated_path,
                sample_interval_s=args.sample_interval,
            )
        except Exception as exc:
            print(f"[parent] animated map failed: {exc}")
            animated_path = None  # type: ignore[assignment]

        crash_pts = load_crash_points_if_available()
        crash_seg = load_crash_segments_if_available()
        crash_map_path: Path | None = run_dir / "crashes.html"
        try:
            build_crash_map(crash_pts, crash_map_path,
                            crash_segments=crash_seg)
        except Exception as exc:
            print(f"[parent] crash map failed: {exc}")
            crash_map_path = None

        try:
            build_stakeholder_html(
                run_dir / "stakeholder.html",
                edge_history=history,
                edge_summary=summary,
                score={
                    "geh_mean": sumo_score.score.geh_mean,
                    "geh_p85": sumo_score.score.geh_p85,
                    "pct_lt_5": sumo_score.score.pct_lt_5,
                    "pct_lt_10": sumo_score.score.pct_lt_10,
                    "n_links_scored": sumo_score.score.n_links_scored,
                },
                animated_map=animated_path,
                crash_map=crash_map_path,
                title=f"Leonia SUMO baseline — {args.demand}",
                subtitle=(
                    f"{summary.shape[0]:,} edges, "
                    f"GEH<5 on {sumo_score.score.pct_lt_5 * 100:.0f}% of "
                    f"{sumo_score.score.n_links_scored:,} scored links."
                ),
                sample_interval_s=args.sample_interval,
            )
        except Exception as exc:
            print(f"[parent] stakeholder HTML failed: {exc}")

    return {
        "geh_mean": sumo_score.score.geh_mean,
        "pct_lt_5": sumo_score.score.pct_lt_5,
        "n_links_scored": sumo_score.score.n_links_scored,
        "rows": int(summary.shape[0]),
    }


def _demand_to_day_part(demand: str) -> str:
    if demand in ("bridge_od_peak_am", "peak_am_slice"):
        return "peak_am"
    # bridge_od_weekday_24h / bridge_od_sunday_24h / others all
    # cover the full 24-hour day, so score against the all-day
    # Street Scanner observation window.
    return "all_day"


def _write_markdown_report(args: argparse.Namespace, run_dir: Path,
                           summary_stats: dict) -> Path:
    """Write the analyst markdown summary at reports/12_sumo_baseline.md."""
    import pandas as pd

    md_path = REPO_ROOT / "reports" / "12_sumo_baseline.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(run_dir / "edge_summary.csv")
    scoring = (
        pd.read_parquet(run_dir / "scoring.parquet")
        if (run_dir / "scoring.parquet").exists() else pd.DataFrame()
    )

    top = (
        summary.dropna(subset=["street_name"])
        .sort_values("peak_vph", ascending=False)
        .head(20)[["street_name", "osm_way_id", "peak_vph", "mean_speed_mph"]]
    )

    rel_run = run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir

    lines: list[str] = []
    lines.append(f"# SUMO baseline — `{args.demand}`")
    lines.append("")
    lines.append("_Auto-generated by `scripts/12_sumo_baseline.py`._")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Demand source | `{args.demand}` |")
    lines.append(f"| Seed | {args.seed} |")
    lines.append(
        f"| GEH mean | {summary_stats.get('geh_mean', float('nan')):.2f} |"
    )
    lines.append(
        f"| Pct GEH < 5 | "
        f"{summary_stats.get('pct_lt_5', 0) * 100:.1f}% |"
    )
    lines.append(
        f"| Links scored | {summary_stats.get('n_links_scored', 0):,} |"
    )
    lines.append(f"| SUMO edges seen | {summary_stats.get('rows', 0):,} |")
    lines.append("")
    lines.append(f"Run artefacts: `{rel_run}/`")
    lines.append("")
    if not args.no_stakeholder:
        lines.append("Stakeholder views:")
        lines.append("")
        lines.append(f"- [Animated map](../{rel_run}/animated.html)")
        lines.append(
            f"- [Stakeholder one-pager](../{rel_run}/stakeholder.html)"
        )
        lines.append("")

    lines.append("## Top 20 highest-volume edges (simulated)")
    lines.append("")
    lines.append("| Street | OSM way | peak vph | mean speed (mph) |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['street_name']} | "
            f"{int(row['osm_way_id']) if pd.notna(row['osm_way_id']) else '—'} "
            f"| {row['peak_vph']:.0f} | {row['mean_speed_mph']:.1f} |"
        )
    lines.append("")

    if not scoring.empty:
        worst = scoring.sort_values("geh", ascending=False).head(10)
        lines.append("## Worst-fit edges (top GEH)")
        lines.append("")
        lines.append("| Street | sim vph | observed vph | GEH |")
        lines.append("| --- | --- | --- | --- |")
        for _, row in worst.iterrows():
            label = row.get("street_name") or row.get("osm_name") or "—"
            lines.append(
                f"| {label} | {row['sim_vph']:.0f} | "
                f"{row['observed_vph']:.0f} | {row['geh']:.2f} |"
            )
        lines.append("")

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
            "bridge_od_weekday_24h", "bridge_od_sunday_24h",
        ],
        help="DemandSource value to simulate.",
    )
    p.add_argument("--gui", action="store_true",
                   help="Launch sumo-gui instead of the headless binary.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-interval", type=int, default=60,
                   help="Per-edge sampling interval (seconds).")
    p.add_argument("--end", type=int, default=None,
                   help="Override the simulation end time (seconds).")
    p.add_argument("--no-stakeholder", action="store_true",
                   help="Skip the animated map and stakeholder one-pager.")
    p.add_argument("--out", default=None,
                   help="Override the run directory (defaults to "
                        "data/processed/sumo/runs/<ts>_<demand>/).")
    p.add_argument("--worker", action="store_true",
                   help="Internal: run the libsumo half of the pipeline.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.worker:
        return _run_worker(args)

    if args.out:
        run_dir = Path(args.out).resolve()
    else:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = (
            REPO_ROOT / "data" / "processed" / "sumo" / "runs"
            / f"{ts}_{args.demand}"
        ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    rc = _spawn_worker(args, run_dir)
    if rc != 0:
        print(f"[parent] worker failed with exit code {rc}")
        return rc

    summary_stats = _post_process(args, run_dir)
    md_path = _write_markdown_report(args, run_dir, summary_stats)
    print()
    print(f"Run dir:        {run_dir}")
    print(f"Markdown report: {md_path}")
    print(f"GEH mean: {summary_stats['geh_mean']:.2f} | "
          f"pct<5: {summary_stats['pct_lt_5'] * 100:.1f}% | "
          f"links scored: {summary_stats['n_links_scored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
