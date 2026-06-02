"""Compare two SUMO run directories (before vs after).

Produces a standalone comparison bundle — KPI delta table, travel-time
distribution overlay, hourly-delay-delta chart, and a self-contained
``compare.html`` — and prints the KPI delta table to the terminal.

Usage
-----

::

    venv/bin/python scripts/13_sumo_compare.py \\
        --baseline data/processed/sumo/runs/<ts>_peak_am_slice \\
        --scenario data/processed/sumo/runs/<ts>_peak_am_slice_adaptive

    # Custom labels + output location
    venv/bin/python scripts/13_sumo_compare.py \\
        --baseline <dir> --scenario <dir> \\
        --label-baseline "Fixed-time" --label-scenario "Adaptive signals" \\
        --out reports/compare_adaptive

Both run directories must contain ``trip_metrics.json`` /
``tripinfo.parquet`` (emitted by scripts 12 / 14). If ``--out`` is
omitted the bundle is written to ``<scenario>/compare/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", required=True, help="Baseline run directory.")
    p.add_argument("--scenario", required=True, help="Scenario run directory.")
    p.add_argument("--out", default=None,
                   help="Output dir (default: <scenario>/compare/).")
    p.add_argument("--label-baseline", default="Baseline")
    p.add_argument("--label-scenario", default="Scenario")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    from leonia_traffic.sumo.comparison import (
        build_compare_report,
        compare_runs,
        print_kpi_table,
    )

    baseline_dir = Path(args.baseline).resolve()
    scenario_dir = Path(args.scenario).resolve()
    for d in (baseline_dir, scenario_dir):
        if not d.exists():
            print(f"Run directory not found: {d}", file=sys.stderr)
            return 2

    result = compare_runs(
        baseline_dir, scenario_dir,
        label_baseline=args.label_baseline,
        label_scenario=args.label_scenario,
    )

    print_kpi_table(result)

    out_dir = Path(args.out).resolve() if args.out else scenario_dir / "compare"
    html_path = build_compare_report(out_dir, result)
    print()
    print(f"Comparison bundle: {out_dir}")
    print(f"Open: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
