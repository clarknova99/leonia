"""Phase 4 / Pass-B: Nelder-Mead calibration.

Two modes:

* ``--mode v1`` (default for backwards compatibility) — calibrates the
  legacy placeholder gateway demand against StreetLight Weekday flows.
  Parameters: ``daily_to_peak_factor``, ``gwb_share``,
  ``min_gateway_volume``.

* ``--mode v2`` — calibrates the **Pass-B** model that uses real Bridge
  OD demand and congestion-derived link free-flow speeds. Parameters:
  ``od_demand_scale``, ``jam_density_factor``,
  ``intersection_capacity_factor``.

* ``--mode v3`` — Pass-C. Same parameter space as v2, but the scoring
  frame is the *union* of Street Scanner (arterial) and Leonia-streets
  ZA (residential) observations. Per-source GEH is logged each
  iteration so you can see whether the calibration is being dragged by
  residential links.

Run from the repo root:

    venv/bin/python scripts/03_calibrate.py --mode v1 --maxiter 15
    venv/bin/python scripts/03_calibrate.py --mode v2 --maxiter 12
    venv/bin/python scripts/03_calibrate.py --mode v3 --maxiter 12

Outputs (mode-suffixed):

    reports/03_calibration.md
    reports/figures/calibration_history.png
    data/processed/calibration_history.parquet
    data/processed/calibration_best_params.json
    data/processed/calibration_best_params_v3.json (v3 only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from leonia_traffic.config import DATA_PROCESSED_DIR, REPORTS_DIR, REPORTS_FIG_DIR
from leonia_traffic.simulation.calibration import (
    CalibrationParams,
    CalibrationParamsV2,
    calibrate,
    calibrate_v2,
    calibrate_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--maxiter", type=int, default=15)
    parser.add_argument("--deltan", type=int, default=20,
                        help="Platoon size. Larger = faster, coarser.")
    parser.add_argument("--tmax", type=int, default=None,
                        help="Simulation horizon in seconds. Defaults to "
                             "2h (v1) or 4h (v2).")
    parser.add_argument("--day-type-code", type=int, default=1,
                        help="(v2 only) Day Type code from the bridge OD export.")
    parser.add_argument("--day-part-code", type=int, default=2,
                        help="(v2 only) Day Part code (2 = Peak AM).")
    args = parser.parse_args()

    if args.mode == "v1":
        tmax = args.tmax or 2 * 3600
        print(f"Running v1 (placeholder gateway) calibration "
              f"maxiter={args.maxiter} deltan={args.deltan} tmax={tmax}")
        result = calibrate(
            initial=CalibrationParams(
                daily_to_peak_factor=0.10,
                gwb_share=0.6,
                min_gateway_volume=500.0,
            ),
            duration_hours=tmax / 3600.0,
            tmax=tmax,
            deltan=args.deltan,
            maxiter=args.maxiter,
        )
        param_keys = ("daily_to_peak_factor", "gwb_share", "min_gateway_volume")
    else:
        tmax = args.tmax or 4 * 3600
        is_v3 = args.mode == "v3"
        label = "v3 (Pass-C: + residential ZA)" if is_v3 else "v2 (calibrated baseline)"
        print(f"Running {label} calibration "
              f"maxiter={args.maxiter} deltan={args.deltan} tmax={tmax} "
              f"day_type={args.day_type_code} day_part={args.day_part_code}")
        runner = calibrate_v3 if is_v3 else calibrate_v2
        result = runner(
            initial=CalibrationParamsV2(
                od_demand_scale=1.0,
                jam_density_factor=1.0,
                intersection_capacity_factor=1.0,
            ),
            duration_hours=tmax / 3600.0,
            tmax=tmax,
            deltan=args.deltan,
            day_type_code=args.day_type_code,
            day_part_code=args.day_part_code,
            maxiter=args.maxiter,
            observed_to_hourly_factor=0.10 / (tmax / 3600.0),
        )
        param_keys = ("od_demand_scale", "jam_density_factor", "intersection_capacity_factor")

    print("Best parameters:")
    print(json.dumps(result.best_params.__dict__, indent=2))
    print("Best score:")
    print(json.dumps(result.best_score.__dict__, indent=2))

    history = pd.DataFrame(result.history)
    history.to_parquet(DATA_PROCESSED_DIR / "calibration_history.parquet")

    best_summary = {
        "mode": args.mode,
        "params": result.best_params.__dict__,
        "score": result.best_score.__dict__,
    }
    (DATA_PROCESSED_DIR / "calibration_best_params.json").write_text(
        json.dumps(best_summary, indent=2), encoding="utf-8",
    )
    if args.mode == "v3":
        (DATA_PROCESSED_DIR / "calibration_best_params_v3.json").write_text(
            json.dumps(best_summary, indent=2), encoding="utf-8",
        )

    plot_history(history, REPORTS_FIG_DIR / "calibration_history.png", param_keys=param_keys)
    write_report(args.mode, result, history, param_keys=param_keys)
    print(f"Report at {REPORTS_DIR / '03_calibration.md'}")


def plot_history(history: pd.DataFrame, out_path: Path, *, param_keys: tuple[str, ...]) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(history["iter"], history["loss"], marker="o")
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("iter"); axes[0, 0].set_ylabel("loss")
    axes[0, 1].plot(history["iter"], history["geh_mean"], marker="o", color="C1")
    axes[0, 1].set_title("Mean GEH"); axes[0, 1].set_xlabel("iter")
    axes[1, 0].plot(history["iter"], history["pct_lt_5"] * 100, marker="o", color="C2")
    axes[1, 0].axhline(85, color="grey", linestyle="--", label="target 85%")
    axes[1, 0].set_title("% links with GEH < 5"); axes[1, 0].set_xlabel("iter"); axes[1, 0].legend()
    for i, key in enumerate(param_keys):
        if key in history.columns:
            axes[1, 1].plot(history["iter"], history[key], marker="o", label=key)
    axes[1, 1].set_title("Parameter trajectories")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(mode: str, result, history: pd.DataFrame, *, param_keys: tuple[str, ...]) -> None:
    lines = [f"# Phase 4 — calibration ({mode})\n"]
    lines.append(
        "_Auto-generated by `scripts/03_calibrate.py`. Each iteration runs a "
        "full UXsim simulation and scores it against StreetLight observed "
        "Weekday flows via the GEH statistic._\n"
    )
    lines.append("## Best parameters\n")
    p = result.best_params
    lines.append("| Parameter | Value |\n| --- | --- |")
    for k, v in p.__dict__.items():
        lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines.append("")

    lines.append("## Best score (vs. observed Weekday flows)\n")
    s = result.best_score
    lines.append("| Metric | Value |\n| --- | --- |")
    for k, v in s.__dict__.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.3f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append(
        "The transport-modeling target is **GEH < 5 on ≥85% of links**. "
        f"This run achieves {s.pct_lt_5 * 100:.1f}% on {s.n_links_scored} "
        "matched links."
    )
    if mode == "v2":
        lines.append(
            "\n\nWith real OD demand and observed link speeds applied, "
            "remaining error typically comes from network mismatches "
            "(e.g. pathfinder choosing different cut-through routes than "
            "drivers do). Inspect ``data/network/network_summary.md`` and "
            "consider adding manual ``set_link_attrs`` overrides in "
            "``data/network/overrides.yaml``."
        )
    else:
        lines.append(
            "\n\nLegacy v1 calibration. Use `--mode v2` for the bridge-OD "
            "+ congestion-trends calibration introduced in Pass B."
        )
    lines.append("")

    lines.append("## Iteration history\n")
    if not history.empty:
        cols = ["iter", "loss", "geh_mean", "geh_p85", "pct_lt_5"] + list(param_keys)
        cols = [c for c in cols if c in history.columns]
        show = history[cols].copy()
        for c in show.columns:
            if c != "iter":
                show[c] = show[c].astype(float).round(3)
        header = "| " + " | ".join(show.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
        lines.append(header)
        lines.append(sep)
        for _, r in show.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in show.columns) + " |")
    lines.append("")
    lines.append("![Calibration history](figures/calibration_history.png)")
    (REPORTS_DIR / "03_calibration.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
