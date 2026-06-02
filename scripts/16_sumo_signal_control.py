"""Run a SUMO simulation under an adaptive max-pressure signal controller.

This is the operational counterpart to ``scripts/12_sumo_baseline.py``:
instead of leaving SUMO's fixed-time signal programmes alone, it drives
every (or a chosen subset of) signalised intersection with the
:class:`~leonia_traffic.sumo.signal_control.AdaptivePressureController`
and emits the same stakeholder bundle, plus controller diagnostics.

Usage
-----

::

    venv/bin/python scripts/16_sumo_signal_control.py --demand peak_am_slice
    venv/bin/python scripts/16_sumo_signal_control.py \\
        --demand bridge_od_weekday_24h --intersections all
    venv/bin/python scripts/16_sumo_signal_control.py \\
        --demand peak_am_slice --baseline-run data/sumo/runs/<ts>_peak_am_slice

When ``--baseline-run`` points at an existing run directory (e.g. a
fixed-time baseline produced by script 12), the before/after comparison
(:mod:`leonia_traffic.sumo.comparison`) runs automatically and writes a
``compare/`` sub-directory.

Why subprocess? — same reason as script 12: ``libsumo`` permanently
breaks ``pyarrow.read_parquet`` in its own process, so the simulation
runs in a worker that writes CSV and the parent post-processes.
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
# Worker — runs in a subprocess, talks to libsumo, writes CSVs
# ---------------------------------------------------------------------------


def _run_worker(args: argparse.Namespace) -> int:
    """Subprocess entry point — runs the controller, then exits."""
    from leonia_traffic.sumo import DemandSource, SumoRuntime
    from leonia_traffic.sumo.signal_control import (
        AdaptivePressureController,
        AdaptiveSignalConfig,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_t = int(args.end) if args.end else None
    rt = SumoRuntime.start(
        demand=DemandSource(args.demand),
        gui=args.gui,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        end_time_s=end_t,
        tripinfo_path=out_dir / "tripinfo.xml",
    )

    all_tls = rt.traffic_light_ids()
    if args.intersections in (None, "", "all"):
        intersection_ids = all_tls
    else:
        wanted = {x.strip() for x in args.intersections.split(",") if x.strip()}
        intersection_ids = [t for t in all_tls if t in wanted]
        missing = wanted - set(intersection_ids)
        if missing:
            print(f"[worker] unknown TLS ids ignored: {sorted(missing)}")

    config = AdaptiveSignalConfig(
        min_green=args.min_green,
        max_green=args.max_green,
        pressure_threshold=args.pressure_threshold,
        control_interval=args.control_interval,
    )
    controller = AdaptivePressureController(
        rt.backend, intersection_ids, config,
    )

    t0 = time.time()
    try:
        rt.run_to_end(step_callback=controller.step)
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
            {
                "stats": stats,
                "signal_control": controller.diagnostics(),
                "n_tls_in_net": len(all_tls),
            },
            indent=2, default=str,
        )
    )
    print(f"[worker] simulation done: {stats}")
    print(f"[worker] signal control: {controller.diagnostics()}")
    return 0


# ---------------------------------------------------------------------------
# Parent — orchestrates the worker + post-processing
# ---------------------------------------------------------------------------


def _spawn_worker(args: argparse.Namespace, run_dir: Path) -> int:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--demand", args.demand,
        "--seed", str(args.seed),
        "--sample-interval", str(args.sample_interval),
        "--out", str(run_dir),
        "--min-green", str(args.min_green),
        "--max-green", str(args.max_green),
        "--pressure-threshold", str(args.pressure_threshold),
        "--control-interval", str(args.control_interval),
        "--intersections", args.intersections or "all",
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

    from leonia_traffic.sumo.scoring import score_sumo_run, write_run_outputs
    from leonia_traffic.sumo.trip_metrics import (
        compute_trip_kpis,
        parse_tripinfo,
        write_trip_metrics,
    )

    sumo_score = score_sumo_run(summary, day_part=_demand_to_day_part(args.demand))

    trip_df = parse_tripinfo(run_dir / "tripinfo.xml")
    trip_kpis = compute_trip_kpis(trip_df)
    write_trip_metrics(run_dir, trip_df, trip_kpis)

    manifest = {
        "demand": args.demand,
        "seed": args.seed,
        "sample_interval_s": args.sample_interval,
        "end_time_s": args.end,
        "gui": args.gui,
        "kind": "signal_control",
        "worker": worker_stats.get("stats", {}),
        "signal_control": worker_stats.get("signal_control", {}),
        "n_tls_in_net": worker_stats.get("n_tls_in_net"),
        "command": " ".join(sys.argv),
        "trip_kpis": trip_kpis.to_dict(),
    }
    write_run_outputs(
        run_dir,
        edge_history=history,
        edge_summary=summary,
        scoring_df=sumo_score.scoring_df,
        score=sumo_score.score,
        manifest=manifest,
    )

    # Deck.gl flow dataset (with baseline embedded for the impact view
    # when a baseline run is supplied).
    try:
        from leonia_traffic.sumo.demand_builder import DEFAULT_NET_PATH
        from leonia_traffic.sumo.visualizations import write_flow_json

        history_pq = pd.read_parquet(run_dir / "edge_history.parquet")
        baseline_history = None
        if args.baseline_run:
            bh = Path(args.baseline_run) / "edge_history.parquet"
            if bh.exists():
                baseline_history = pd.read_parquet(bh)
        write_flow_json(
            history_pq, DEFAULT_NET_PATH, run_dir / "flow.json",
            sample_interval_s=args.sample_interval,
            title=f"Adaptive signals · {args.demand}",
            baseline_history=baseline_history,
        )
    except Exception as exc:
        print(f"[parent] flow.json failed: {exc}")

    return {
        "geh_mean": sumo_score.score.geh_mean,
        "pct_lt_5": sumo_score.score.pct_lt_5,
        "n_links_scored": sumo_score.score.n_links_scored,
        "rows": int(summary.shape[0]),
        "trip_kpis": trip_kpis.to_dict(),
        "signal_control": worker_stats.get("signal_control", {}),
    }


def _demand_to_day_part(demand: str) -> str:
    if demand in ("bridge_od_peak_am", "peak_am_slice"):
        return "peak_am"
    return "all_day"


def _maybe_compare(args: argparse.Namespace, run_dir: Path) -> Path | None:
    """When --baseline-run is set, build the before/after comparison."""
    if not args.baseline_run:
        return None
    baseline_dir = Path(args.baseline_run).resolve()
    if not baseline_dir.exists():
        print(f"[parent] baseline run dir not found: {baseline_dir}")
        return None
    from leonia_traffic.sumo.comparison import build_compare_report, compare_runs

    result = compare_runs(baseline_dir, run_dir)
    compare_dir = run_dir / "compare"
    build_compare_report(
        compare_dir, result,
        label_baseline="Fixed-time baseline",
        label_scenario="Adaptive signals",
    )
    print(f"[parent] comparison -> {compare_dir}")
    return compare_dir


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
    p.add_argument(
        "--intersections", default="all",
        help="'all' (default) or a comma-separated list of TLS ids.",
    )
    p.add_argument("--min-green", type=int, default=25)
    p.add_argument("--max-green", type=int, default=60)
    p.add_argument("--pressure-threshold", type=int, default=3)
    p.add_argument("--control-interval", type=int, default=5,
                   help="Apply the controller every N simulation seconds.")
    p.add_argument(
        "--baseline-run", default=None,
        help="Existing run dir (e.g. a fixed-time baseline) to compare "
             "against; triggers the before/after report when supplied.",
    )
    p.add_argument("--out", default=None,
                   help="Override the run directory (defaults to "
                        "data/processed/sumo/runs/<ts>_<demand>_adaptive/).")
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
        from leonia_traffic.config import SUMO_RUNS_DIR
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = (SUMO_RUNS_DIR / f"{ts}_{args.demand}_adaptive").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    rc = _spawn_worker(args, run_dir)
    if rc != 0:
        print(f"[parent] worker failed with exit code {rc}")
        return rc

    summary_stats = _post_process(args, run_dir)
    compare_dir = _maybe_compare(args, run_dir)

    sc = summary_stats.get("signal_control", {})
    tk = summary_stats.get("trip_kpis", {})
    print()
    print(f"Run dir:        {run_dir}")
    print(
        f"Signal control: {sc.get('n_intersections', 0)} intersections, "
        f"{sc.get('n_switches', 0)} phase switches"
    )
    print(
        f"Trip KPIs: completion {tk.get('completion_rate', 0) * 100:.1f}% | "
        f"mean travel {tk.get('mean_travel_min', float('nan')):.1f} min | "
        f"delay {tk.get('total_delay_h', 0):.1f} veh·h"
    )
    if compare_dir:
        print(f"Comparison:     {compare_dir}/compare.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
