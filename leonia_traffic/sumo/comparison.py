"""Before/after comparison of two SUMO runs.

Diffs a *baseline* run directory against a *scenario* run directory
(both produced by ``scripts/12_sumo_baseline.py``,
``scripts/16_sumo_signal_control.py``, or the webapp precache builder)
and produces:

* a KPI delta table (trip-level: completion, travel time, delay,
  waiting) — mirrors the Chișinău ``compare_runs.py`` schema;
* a per-edge ``peak_vph`` delta (which streets gained / lost traffic);
* matplotlib figures (travel-time distribution overlay, hourly delay
  delta);
* a self-contained ``compare.html`` plus a compact ``compare_kpis.json``
  the stakeholder webapp renders as a before/after panel.

Reads only JSON / parquet, so it runs in the parent post-processing
process (never alongside ``libsumo``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Display label, KPI key, format, and whether *lower is better*.
KPI_METRICS: list[tuple[str, str, str, bool]] = [
    ("Completion rate", "completion_rate", "pct", False),
    ("Mean travel time (min)", "mean_travel_min", "float", True),
    ("Median travel time (min)", "median_travel_min", "float", True),
    ("p90 travel time (min)", "p90_travel_min", "float", True),
    ("Total delay (veh·h)", "total_delay_h", "float", True),
    ("Mean waiting (min)", "mean_waiting_min", "float", True),
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_kpis(run_dir: Path) -> dict:
    p = run_dir / "trip_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    # Fall back to the manifest's embedded copy.
    man = run_dir / "manifest.json"
    if man.exists():
        data = json.loads(man.read_text())
        return data.get("trip_kpis", {}) or {}
    return {}


def _load_trips(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "tripinfo.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            logger.warning("could not read %s: %s", p, exc)
    return pd.DataFrame()


def _load_summary(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "edge_summary.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            logger.warning("could not read %s: %s", p, exc)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    label_baseline: str
    label_scenario: str
    baseline_kpis: dict
    scenario_kpis: dict
    baseline_trips: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenario_trips: pd.DataFrame = field(default_factory=pd.DataFrame)
    edge_delta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def kpi_rows(self) -> list[dict]:
        """Full per-metric rows for the table / kpi_delta.json."""
        rows: list[dict] = []
        for label, key, fmt, lower_better in KPI_METRICS:
            base = self.baseline_kpis.get(key)
            scen = self.scenario_kpis.get(key)
            if base is None or scen is None:
                continue
            try:
                delta = float(scen) - float(base)
            except (TypeError, ValueError):
                continue
            delta_pct = (delta / float(base) * 100.0) if base else 0.0
            improved = (delta < 0) if lower_better else (delta > 0)
            rows.append({
                "key": key,
                "label": label,
                "fmt": fmt,
                "base": float(base),
                "scenario": float(scen),
                "delta": delta,
                "delta_pct": delta_pct,
                "improved": bool(improved),
            })
        return rows

    def kpi_delta_payload(self) -> dict:
        """Compact dict the webapp's before/after panel consumes."""
        return {
            "labels": {
                "baseline": self.label_baseline,
                "scenario": self.label_scenario,
            },
            "metrics": self.kpi_rows(),
        }


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare_runs(
    baseline_dir: str | Path,
    scenario_dir: str | Path,
    *,
    label_baseline: str = "Baseline",
    label_scenario: str = "Scenario",
) -> ComparisonResult:
    """Diff two run directories into a :class:`ComparisonResult`."""
    baseline_dir = Path(baseline_dir)
    scenario_dir = Path(scenario_dir)

    base_kpis = _load_kpis(baseline_dir)
    scen_kpis = _load_kpis(scenario_dir)
    base_trips = _load_trips(baseline_dir)
    scen_trips = _load_trips(scenario_dir)

    edge_delta = _edge_delta(
        _load_summary(baseline_dir), _load_summary(scenario_dir),
    )

    return ComparisonResult(
        label_baseline=label_baseline,
        label_scenario=label_scenario,
        baseline_kpis=base_kpis,
        scenario_kpis=scen_kpis,
        baseline_trips=base_trips,
        scenario_trips=scen_trips,
        edge_delta=edge_delta,
    )


def _edge_delta(
    base_summary: pd.DataFrame, scen_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Per-edge ``peak_vph`` delta, sorted by absolute change."""
    if base_summary.empty or scen_summary.empty:
        return pd.DataFrame()
    cols = ["sumo_edge_id", "peak_vph"]
    keep = ["sumo_edge_id", "peak_vph", "street_name"]
    base = base_summary[[c for c in keep if c in base_summary.columns]].copy()
    scen = scen_summary[[c for c in keep if c in scen_summary.columns]].copy()
    if not set(cols).issubset(base.columns) or not set(cols).issubset(scen.columns):
        return pd.DataFrame()
    base = base.rename(columns={"peak_vph": "peak_vph_baseline"})
    scen = scen.rename(columns={"peak_vph": "peak_vph_scenario"})
    merged = base.merge(
        scen[["sumo_edge_id", "peak_vph_scenario"]],
        on="sumo_edge_id", how="outer",
    ).fillna({"peak_vph_baseline": 0.0, "peak_vph_scenario": 0.0})
    merged["delta_vph"] = (
        merged["peak_vph_scenario"] - merged["peak_vph_baseline"]
    )
    merged["abs_delta_vph"] = merged["delta_vph"].abs()
    return merged.sort_values("abs_delta_vph", ascending=False)


# ---------------------------------------------------------------------------
# Figures (matplotlib, Agg backend)
# ---------------------------------------------------------------------------


def fig_tt_overlay(result: ComparisonResult, out_path: Path) -> bool:
    """Travel-time distribution overlay (baseline vs scenario)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _completed(df: pd.DataFrame) -> pd.Series:
        if df.empty or "travel_time_min" not in df.columns:
            return pd.Series(dtype=float)
        if "completed" in df.columns:
            df = df[df["completed"]]
        return df["travel_time_min"].dropna()

    base = _completed(result.baseline_trips)
    scen = _completed(result.scenario_trips)
    if base.empty and scen.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    for series, label, color in [
        (base, result.label_baseline, "#2E86AB"),
        (scen, result.label_scenario, "#E84855"),
    ]:
        if series.empty:
            continue
        ax.hist(
            series, bins=60, alpha=0.6, color=color,
            label=f"{label} (p50={series.median():.1f}, "
                  f"p90={series.quantile(0.9):.1f})",
        )
        ax.axvline(series.median(), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Travel time (minutes)")
    ax.set_ylabel("Vehicles")
    ax.set_title("Travel-time distribution — before vs after")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_hourly_delay_delta(result: ComparisonResult, out_path: Path) -> bool:
    """Mean time-loss by hour for both runs, plus the per-hour delta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _hourly(df: pd.DataFrame) -> pd.Series:
        if df.empty or "loss_min" not in df.columns \
                or "depart_hour" not in df.columns:
            return pd.Series(dtype=float)
        sub = df[df["completed"]] if "completed" in df.columns else df
        sub = sub.dropna(subset=["depart_hour"])
        if sub.empty:
            return pd.Series(dtype=float)
        return sub.groupby(sub["depart_hour"].astype(int))["loss_min"].mean()

    h_base = _hourly(result.baseline_trips)
    h_scen = _hourly(result.scenario_trips)
    if h_base.empty and h_scen.empty:
        return False
    delta = h_scen.subtract(h_base, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if not h_base.empty:
        ax.plot(h_base.index, h_base.values, label=result.label_baseline,
                color="#2E86AB", linewidth=2)
    if not h_scen.empty:
        ax.plot(h_scen.index, h_scen.values, label=result.label_scenario,
                color="#E84855", linewidth=2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean time loss (min)")
    ax.set_title("Mean delay by hour")
    ax.legend()

    ax2 = axes[1]
    colors = ["#E84855" if v > 0 else "#90BE6D" for v in delta.values]
    ax2.bar(delta.index, delta.values, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Δ delay (min)  [positive = worse]")
    ax2.set_title(f"Delay change: {result.label_scenario} − {result.label_baseline}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_value(value: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _fmt_delta(row: dict) -> str:
    if row["fmt"] == "pct":
        return f"{row['delta'] * 100:+.1f} pts"
    return f"{row['delta']:+.2f} ({row['delta_pct']:+.1f}%)"


def _render_compare_html(result: ComparisonResult, has_figs: dict) -> str:
    rows = result.kpi_rows()
    body_rows = []
    for r in rows:
        cls = "improved" if r["improved"] else "worse"
        arrow = "▼" if r["delta"] < 0 else ("▲" if r["delta"] > 0 else "")
        body_rows.append(
            f"<tr><td>{r['label']}</td>"
            f"<td>{_fmt_value(r['base'], r['fmt'])}</td>"
            f"<td>{_fmt_value(r['scenario'], r['fmt'])}</td>"
            f"<td class='{cls}'>{arrow} {_fmt_delta(r)}</td></tr>"
        )
    table = "\n".join(body_rows)

    figs = []
    if has_figs.get("tt"):
        figs.append('<img src="fig_tt_overlay.png" alt="Travel time overlay" />')
    if has_figs.get("hourly"):
        figs.append('<img src="fig_hourly_delay.png" alt="Hourly delay delta" />')
    figs_html = "\n".join(figs)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Before / after — {result.label_scenario} vs {result.label_baseline}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem;
         color: #1a2233; background: #f7f9fc; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; background: #fff;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  th, td {{ padding: .55rem .9rem; text-align: right; border-bottom: 1px solid #e3e8f0; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #1a2233; color: #fff; }}
  td.improved {{ color: #157f3b; font-weight: 600; }}
  td.worse {{ color: #c0392b; font-weight: 600; }}
  img {{ max-width: 100%; margin: 1rem 0; border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
</style></head><body>
<h1>Before / after: {result.label_scenario} vs {result.label_baseline}</h1>
<table>
  <thead><tr><th>Metric</th><th>{result.label_baseline}</th>
  <th>{result.label_scenario}</th><th>Change</th></tr></thead>
  <tbody>
{table}
  </tbody>
</table>
{figs_html}
</body></html>
"""


def build_compare_report(
    out_dir: str | Path,
    result: ComparisonResult,
    *,
    label_baseline: str | None = None,
    label_scenario: str | None = None,
) -> Path:
    """Write the standalone comparison bundle into ``out_dir``.

    Produces ``fig_tt_overlay.png``, ``fig_hourly_delay.png``,
    ``kpi_delta.json``, ``compare_kpis.json``, and ``compare.html``.
    Returns the path to ``compare.html``.
    """
    if label_baseline:
        result.label_baseline = label_baseline
    if label_scenario:
        result.label_scenario = label_scenario

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_figs = {
        "tt": fig_tt_overlay(result, out_dir / "fig_tt_overlay.png"),
        "hourly": fig_hourly_delay_delta(result, out_dir / "fig_hourly_delay.png"),
    }

    (out_dir / "kpi_delta.json").write_text(
        json.dumps(result.kpi_rows(), indent=2, default=str)
    )
    (out_dir / "compare_kpis.json").write_text(
        json.dumps(result.kpi_delta_payload(), indent=2, default=str)
    )
    if not result.edge_delta.empty:
        try:
            result.edge_delta.head(200).to_csv(
                out_dir / "edge_delta_top.csv", index=False,
            )
        except Exception as exc:
            logger.warning("edge_delta write failed: %s", exc)

    html_path = out_dir / "compare.html"
    html_path.write_text(_render_compare_html(result, has_figs))
    return html_path


def print_kpi_table(result: ComparisonResult) -> None:
    """Print the Chișinău-style KPI delta table to stdout."""
    rows = result.kpi_rows()
    lb, ls = result.label_baseline, result.label_scenario
    print(f"\n{'Metric':35s}  {lb:>14s}  {ls:>14s}  {'Δ':>16s}")
    print("─" * 86)
    for r in rows:
        arrow = "▼" if r["delta"] < 0 else ("▲" if r["delta"] > 0 else " ")
        print(
            f"{r['label']:35s}  "
            f"{_fmt_value(r['base'], r['fmt']):>14s}  "
            f"{_fmt_value(r['scenario'], r['fmt']):>14s}  "
            f"{arrow} {_fmt_delta(r):>14s}"
        )
