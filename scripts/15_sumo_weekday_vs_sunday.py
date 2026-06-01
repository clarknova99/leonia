"""Run two 24-hour SUMO baselines (weekday avg + Sunday) and emit a combined stakeholder report.

The weekday animation is built from the average of Monday–Friday
StreetLight bridge OD cohorts (``day_type_code`` ∈ {1,2,3,4,5}); the
Sunday animation is built from the Sunday-only cohort
(``day_type_code == 7``). Both run for the full 24-hour day on the
new 2036064_Destinations 24-hour-daypart export.

Outputs (under
``data/processed/sumo/runs/<ts>_weekday_vs_sunday/``):

* ``weekday/`` — full per-run artifacts for the weekday-avg run
  (edge_history.parquet, edge_summary.parquet, scoring.parquet,
  manifest.json, animated.html).
* ``sunday/`` — same shape, for the Sunday run.
* ``crashes.html`` — shared crash overlay (the safety story doesn't
  vary by day-of-week).
* ``stakeholder.html`` — combined one-pager embedding both
  animations side-by-side, the shared crash map, and the safety
  panel / trend chart computed from the weekday run.

Usage
-----

::

    venv/bin/python scripts/15_sumo_weekday_vs_sunday.py

    # Skip the Sunday simulation (e.g. while iterating on weekday)
    venv/bin/python scripts/15_sumo_weekday_vs_sunday.py --skip-sunday

The script wraps :mod:`scripts.12_sumo_baseline` rather than
duplicating its libsumo plumbing — each demand source is dispatched
to a fresh subprocess so libsumo's pyarrow conflict doesn't leak
into the combined-report rendering step.
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

BASELINE_SCRIPT = REPO_ROOT / "scripts" / "12_sumo_baseline.py"


def _run_baseline(*, demand: str, out_dir: Path, seed: int = 42) -> int:
    """Spawn ``12_sumo_baseline.py`` for a single demand source."""
    cmd = [
        sys.executable, str(BASELINE_SCRIPT),
        "--demand", demand,
        "--seed", str(seed),
        "--out", str(out_dir),
        # We render a *combined* stakeholder report below, so we
        # don't need each run to render its own. ``--no-stakeholder``
        # also skips the per-run animated map build, which we then
        # rebuild ourselves below at higher quality (always pointing
        # at a shared crash overlay path).
        "--no-stakeholder",
    ]
    print(f"\n=== running {demand} → {out_dir.name} ===")
    print(f"    {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


def _build_per_run_animated_map(run_dir: Path) -> Path | None:
    """Rebuild ``animated.html`` for a run using its cached parquets."""
    import pandas as pd

    from leonia_traffic.sumo.visualizations import build_animated_map

    history_path = run_dir / "edge_history.parquet"
    if not history_path.exists():
        # Worker writes CSV; post-processing converts to parquet.
        # If we ran with --no-stakeholder, post-processing already
        # ran and produced the parquet. If not, fall back to CSV.
        history_path = run_dir / "edge_history.csv"
    if not history_path.exists():
        return None

    if history_path.suffix == ".parquet":
        history = pd.read_parquet(history_path)
    else:
        history = pd.read_csv(history_path)

    out = run_dir / "animated.html"
    build_animated_map(history, out, sample_interval_s=60)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-weekday", action="store_true")
    p.add_argument("--skip-sunday", action="store_true")
    p.add_argument("--out", default=None,
                   help="Override the parent run directory (defaults to "
                        "data/processed/sumo/runs/<ts>_weekday_vs_sunday/).")
    args = p.parse_args(argv)

    if args.out:
        parent_dir = Path(args.out).resolve()
    else:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        parent_dir = (
            REPO_ROOT / "data" / "processed" / "sumo" / "runs"
            / f"{ts}_weekday_vs_sunday"
        ).resolve()
    parent_dir.mkdir(parents=True, exist_ok=True)

    weekday_dir = parent_dir / "weekday"
    sunday_dir = parent_dir / "sunday"

    if not args.skip_weekday:
        rc = _run_baseline(
            demand="bridge_od_weekday_24h",
            out_dir=weekday_dir, seed=args.seed,
        )
        if rc != 0:
            print(f"[parent] weekday run failed with code {rc}")
            return rc
        _build_per_run_animated_map(weekday_dir)

    if not args.skip_sunday:
        rc = _run_baseline(
            demand="bridge_od_sunday_24h",
            out_dir=sunday_dir, seed=args.seed,
        )
        if rc != 0:
            print(f"[parent] sunday run failed with code {rc}")
            return rc
        _build_per_run_animated_map(sunday_dir)

    # Combined stakeholder one-pager. Use the weekday run as the
    # "primary" data source for KPIs / trend chart / safety panel
    # (the council audience cares more about commuter-day patterns),
    # and embed both animations as separate panels.
    import pandas as pd

    from leonia_traffic.sumo.visualizations import (
        build_crash_map,
        build_stakeholder_html,
        load_crash_points_if_available,
        load_crash_segments_if_available,
    )

    history = pd.read_parquet(weekday_dir / "edge_history.parquet")
    summary = pd.read_parquet(weekday_dir / "edge_summary.parquet")
    manifest = json.loads((weekday_dir / "manifest.json").read_text())
    score = manifest.get("score") or {}

    crash_pts = load_crash_points_if_available()
    crash_seg = load_crash_segments_if_available()
    crash_map_path = parent_dir / "crashes.html"
    try:
        build_crash_map(crash_pts, crash_map_path, crash_segments=crash_seg)
    except Exception as exc:
        print(f"[parent] crash map failed: {exc}")
        crash_map_path = None  # type: ignore[assignment]

    animations = [
        {
            "title": "Average weekday — 24-hour traffic animation",
            "subtitle": (
                "Demand averaged across Monday–Friday StreetLight "
                "bridge OD cohorts. The slider runs 00:00 → 24:00 "
                "in 15-minute frames; expect a strong 7–9am peak "
                "on Broad/Grand/Fort Lee Rd toward the GW Bridge "
                "and a softer 4–6pm return-trip peak."
            ),
            "path": weekday_dir / "animated.html",
        },
        {
            "title": "Average Sunday — 24-hour traffic animation",
            "subtitle": (
                "Demand from the Sunday-only StreetLight bridge OD "
                "cohort. Volumes are roughly 1/10 of weekday peak "
                "in the morning but pick up in mid-afternoon as "
                "shopping / NJ-bound through-traffic returns east. "
                "Useful as a contrast for residents who only see "
                "the borough on weekends."
            ),
            "path": sunday_dir / "animated.html",
        },
    ]

    out_html = parent_dir / "stakeholder.html"
    build_stakeholder_html(
        out_html,
        edge_history=history,
        edge_summary=summary,
        score=score,
        animations=animations,
        crash_map=crash_map_path,
        title="Leonia simulated traffic — weekday vs. Sunday",
        subtitle=(
            "Two 24-hour SUMO baselines side-by-side. Weekday demand "
            "is the average of Mon–Fri StreetLight cohorts; Sunday is "
            "the Sunday-only cohort. KPI strip and safety/trend "
            "panels reflect the weekday run."
        ),
    )

    print()
    print(f"Run dir:   {parent_dir}")
    print(f"Combined:  {out_html}")
    if crash_map_path is not None and crash_map_path.exists():
        print(f"Crashes:   {crash_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
