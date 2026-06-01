"""Per-street trend analytics built on the Street Scanner Trend export.

The trend export carries a single ``avg_volume`` value per
``(zone_name, year_month)`` for Jan 2023 → present (35 + months).
We summarise each street with:

* ``recent_12mo_avg`` / ``baseline_12mo_avg`` — mean monthly volume in
  the most-recent 12 months vs the same 12-month window a year earlier.
* ``yoy_change_pct`` — percent change from baseline to recent.
* ``trend_slope_per_year`` — OLS slope (vehicles / year) over the full
  series.
* ``trend_r2`` — goodness-of-fit of the linear trend (1.0 = perfectly
  linear, near-0 = noisy / cyclical).
* ``peak_year_month`` — the single highest-volume month.
* ``last_value`` / ``last_year_month`` — most recent point.
* ``share_recent_above_baseline_peak`` — share of recent months that
  exceeded the *worst* baseline month (i.e. an "always worse" indicator).

The module exposes one public function, :func:`street_trend_metrics`,
plus a thin helper that picks the worst-N streets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _ols_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.all(np.isnan(y)):
        return (float("nan"), float("nan"))
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return (float("nan"), float("nan"))
    xv = x[mask]
    yv = y[mask]
    slope, intercept = np.polyfit(xv, yv, 1)
    yhat = slope * xv + intercept
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (float(slope), float(r2))


def street_trend_metrics(
    trend_df: pd.DataFrame,
    *,
    recent_months: int = 12,
) -> pd.DataFrame:
    """Reduce the long trend table to one row per street.

    Parameters
    ----------
    trend_df:
        Output of
        :func:`leonia_traffic.data.streetscanner_trend_loader.load_streetscanner_trend`.
        Expected columns include ``zone_name``, ``osm_name``,
        ``osm_way_id``, ``year_month``, ``avg_volume``.
    recent_months:
        Length (in months) of the recent / baseline windows.
    """
    if trend_df is None or trend_df.empty:
        return pd.DataFrame(columns=[
            "zone_name", "osm_name", "osm_way_id",
            "recent_12mo_avg", "baseline_12mo_avg",
            "yoy_change_pct", "trend_slope_per_year", "trend_r2",
            "peak_year_month", "peak_value",
            "last_year_month", "last_value",
            "share_recent_above_baseline_peak",
        ])

    df = trend_df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
    df = df.dropna(subset=["year_month", "avg_volume"])
    df = df.sort_values(["zone_name", "year_month"])

    if df.empty:
        return pd.DataFrame()

    max_month = df["year_month"].max()
    recent_cutoff = max_month - pd.DateOffset(months=recent_months - 1)
    baseline_end = recent_cutoff - pd.DateOffset(months=1)
    baseline_start = baseline_end - pd.DateOffset(months=recent_months - 1)

    out_rows: list[dict] = []
    for (zone, osm_name, osm_way_id), grp in df.groupby(
        ["zone_name", "osm_name", "osm_way_id"], dropna=False
    ):
        recent = grp[grp["year_month"] >= recent_cutoff]["avg_volume"]
        baseline = grp[(grp["year_month"] >= baseline_start)
                        & (grp["year_month"] <= baseline_end)]["avg_volume"]
        rmean = float(recent.mean()) if len(recent) else float("nan")
        bmean = float(baseline.mean()) if len(baseline) else float("nan")
        yoy = ((rmean - bmean) / bmean * 100.0) if bmean and bmean > 0 else float("nan")

        x = ((grp["year_month"] - grp["year_month"].min())
             .dt.days.to_numpy() / 365.25)
        y = grp["avg_volume"].to_numpy(dtype=float)
        slope, r2 = _ols_slope_r2(x, y)

        idxmax = grp["avg_volume"].idxmax()
        peak_month = grp.loc[idxmax, "year_month"] if pd.notna(idxmax) else None
        peak_val = grp["avg_volume"].max()
        last = grp.iloc[-1]

        share_above = float("nan")
        if len(recent) and len(baseline):
            bpeak = float(baseline.max())
            if bpeak > 0:
                share_above = float((recent > bpeak).sum()) / len(recent)

        out_rows.append({
            "zone_name": zone,
            "osm_name": osm_name,
            "osm_way_id": osm_way_id,
            "recent_12mo_avg": rmean,
            "baseline_12mo_avg": bmean,
            "yoy_change_pct": yoy,
            "trend_slope_per_year": slope,
            "trend_r2": r2,
            "peak_year_month": (peak_month.date()
                                 if peak_month is not None and pd.notna(peak_month)
                                 else None),
            "peak_value": float(peak_val),
            "last_year_month": last["year_month"].date(),
            "last_value": float(last["avg_volume"]),
            "share_recent_above_baseline_peak": share_above,
        })

    out = pd.DataFrame(out_rows)
    # Make osm_way_id properly nullable Int64.
    if "osm_way_id" in out.columns:
        out["osm_way_id"] = pd.array(out["osm_way_id"], dtype="Int64")
    out = out.sort_values("yoy_change_pct", ascending=False).reset_index(drop=True)
    out["yoy_rank"] = out.index + 1
    return out


def worsening_streets(
    trend_metrics_df: pd.DataFrame,
    *,
    min_yoy_pct: float = 15.0,
    min_recent_volume: float = 30.0,
) -> pd.DataFrame:
    """Filter to streets growing more than ``min_yoy_pct`` % year-over-year.

    Excludes very low-volume streets where small absolute changes inflate
    percent growth.
    """
    if trend_metrics_df is None or trend_metrics_df.empty:
        return trend_metrics_df
    mask = (
        (trend_metrics_df["yoy_change_pct"] >= min_yoy_pct)
        & (trend_metrics_df["recent_12mo_avg"] >= min_recent_volume)
    )
    return (
        trend_metrics_df.loc[mask]
        .sort_values("yoy_change_pct", ascending=False)
        .reset_index(drop=True)
    )


__all__ = [
    "street_trend_metrics",
    "worsening_streets",
]
