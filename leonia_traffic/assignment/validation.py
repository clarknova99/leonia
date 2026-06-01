"""Validate static-assignment output against StreetLight Street Scanner.

Reuses :func:`leonia_traffic.simulation.calibration.geh_array` to keep
the same GEH definition as the UXsim calibration pathway. Joins an
``AssignmentResult.by_osm_way()`` frame to the canonical
``streetscanner_segments.parquet`` on ``osm_way_id``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from leonia_traffic.assignment.assignment import AssignmentResult
from leonia_traffic.simulation.calibration import geh_array

logger = logging.getLogger(__name__)


@dataclass
class ValidationStats:
    """Aggregate validation statistics."""

    n_segments: int
    n_matched: int
    mean_geh: float
    median_geh: float
    pct_geh_lt_5: float       # share with GEH < 5
    pct_geh_lt_10: float      # share with GEH < 10
    r2: float
    rmse_vph: float
    bias_vph: float           # mean(simulated - observed)
    per_segment: pd.DataFrame  # joined table for drilldown

    def summary_dict(self) -> dict[str, float]:
        return {
            "n_segments": self.n_segments,
            "n_matched": self.n_matched,
            "mean_geh": self.mean_geh,
            "median_geh": self.median_geh,
            "pct_geh_lt_5": self.pct_geh_lt_5,
            "pct_geh_lt_10": self.pct_geh_lt_10,
            "r2": self.r2,
            "rmse_vph": self.rmse_vph,
            "bias_vph": self.bias_vph,
        }


def _r2(observed: np.ndarray, simulated: np.ndarray) -> float:
    o = np.asarray(observed, dtype=float)
    s = np.asarray(simulated, dtype=float)
    mask = np.isfinite(o) & np.isfinite(s)
    if mask.sum() < 2:
        return float("nan")
    o = o[mask]
    s = s[mask]
    ss_res = float(((o - s) ** 2).sum())
    ss_tot = float(((o - o.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def validate_against_streetscanner(
    result: AssignmentResult,
    segments: gpd.GeoDataFrame | pd.DataFrame,
    *,
    source_label: str = "weekdays",
    observed_to_hourly_factor: float = 0.10 / 4.0,
    bidi_split: float = 0.5,
) -> ValidationStats:
    """Compare per-OSM-way assigned flows to observed Street Scanner volumes.

    Parameters
    ----------
    result
        Output of :func:`run_ue`.
    segments
        ``streetscanner_segments.parquet``. The function filters to
        ``source == source_label`` and aggregates multiple "split" zones
        for the same OSM way by averaging (each split is a redundant
        observation of the same flow rate along the way).
    observed_to_hourly_factor
        Conversion from ``avg_volume`` (which is **vehicles per day**
        on the Street Scanner export) to **vehicles per hour** in the
        assignment window. Default 0.10 / 4 = 2.5 %/hour, consistent
        with a Peak-AM (4 h) window taking ~10 % of AADT — same factor
        used in ``leonia_traffic.simulation.demand``.
    bidi_split
        For bidirectional Street Scanner zones, share of the observed
        volume attributed to each direction. Default 0.5.

    Notes
    -----
    The assignment is directional (separate forward / reverse edges per
    OSM way) but the Street Scanner ``avg_volume`` is summed across
    directions for bidirectional zones. We therefore *halve* the
    observed value before comparing to a single-direction assigned
    flow. For zones flagged ``is_bidi == False`` the observed value is
    compared directly.
    """
    by_way = result.by_osm_way()
    if by_way.empty:
        logger.warning("Assignment has no edges with osm_way_id; cannot validate.")
        return ValidationStats(
            n_segments=0, n_matched=0, mean_geh=float("nan"),
            median_geh=float("nan"), pct_geh_lt_5=0.0, pct_geh_lt_10=0.0,
            r2=float("nan"), rmse_vph=float("nan"), bias_vph=float("nan"),
            per_segment=pd.DataFrame(),
        )

    seg = pd.DataFrame(segments).copy()
    if "source" in seg.columns:
        seg = seg[seg["source"] == source_label]
    seg = seg.dropna(subset=["osm_way_id", "avg_volume"]).copy()
    seg["osm_way_id"] = seg["osm_way_id"].astype(int)
    seg["avg_volume"] = pd.to_numeric(seg["avg_volume"], errors="coerce")

    # Aggregate splits.
    bidi_col = "is_bidi" if "is_bidi" in seg.columns else None
    agg_rows = []
    for osm_id, group in seg.groupby("osm_way_id"):
        observed_daily = float(group["avg_volume"].mean())
        observed_hourly_total = observed_daily * observed_to_hourly_factor
        is_bidi = bool(group[bidi_col].any()) if bidi_col else False
        observed_per_direction = (
            observed_hourly_total * bidi_split if is_bidi else observed_hourly_total
        )
        agg_rows.append({
            "osm_way_id": int(osm_id),
            "observed_daily": observed_daily,
            "observed_hourly": observed_hourly_total,
            "observed_hourly_per_dir": observed_per_direction,
            "is_bidi": is_bidi,
            "n_splits": len(group),
            "street_name": (
                str(group["road_name"].iloc[0])
                if "road_name" in group.columns and not group["road_name"].empty
                else ""
            ),
        })
    seg_agg = pd.DataFrame(agg_rows)

    merged = by_way.merge(seg_agg, on="osm_way_id", how="inner")
    if merged.empty:
        logger.warning("No osm_way_id overlap between assignment and street scanner.")
        return ValidationStats(
            n_segments=len(seg_agg), n_matched=0, mean_geh=float("nan"),
            median_geh=float("nan"), pct_geh_lt_5=0.0, pct_geh_lt_10=0.0,
            r2=float("nan"), rmse_vph=float("nan"), bias_vph=float("nan"),
            per_segment=merged,
        )

    sim = merged["assigned_volume_vph"].to_numpy(dtype=float)
    obs = merged["observed_hourly_per_dir"].to_numpy(dtype=float)
    merged["geh"] = geh_array(sim, obs)
    geh = merged["geh"].to_numpy(dtype=float)
    finite = np.isfinite(geh)

    n_matched = int(finite.sum())
    if n_matched == 0:
        return ValidationStats(
            n_segments=len(seg_agg), n_matched=0,
            mean_geh=float("nan"), median_geh=float("nan"),
            pct_geh_lt_5=0.0, pct_geh_lt_10=0.0,
            r2=float("nan"), rmse_vph=float("nan"), bias_vph=float("nan"),
            per_segment=merged,
        )

    diffs = sim[finite] - obs[finite]
    return ValidationStats(
        n_segments=len(seg_agg),
        n_matched=n_matched,
        mean_geh=float(np.nanmean(geh)),
        median_geh=float(np.nanmedian(geh)),
        pct_geh_lt_5=float((geh[finite] < 5).mean()),
        pct_geh_lt_10=float((geh[finite] < 10).mean()),
        r2=_r2(obs[finite], sim[finite]),
        rmse_vph=float(np.sqrt((diffs ** 2).mean())),
        bias_vph=float(diffs.mean()),
        per_segment=merged.sort_values("geh", ascending=False, na_position="last"),
    )


__all__ = ["ValidationStats", "validate_against_streetscanner"]
