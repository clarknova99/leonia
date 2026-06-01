"""GEH scoring of a SUMO run against StreetLight Street Scanner.

Mirrors :mod:`leonia_traffic.simulation.calibration` so a SUMO run
can use the same downstream reporting helpers as the UXsim pipeline.

The SUMO runtime hands us:

* an ``edge_history`` DataFrame with per-edge counters at fixed
  intervals,
* an ``edge_summary`` DataFrame with per-edge ``peak_vph`` /
  ``mean_speed_mph``.

The scorer joins those onto the StreetLight ``streetscanner_segments``
parquet (via ``leonia.edgedata.meta.csv``'s ``osm_way_id`` column)
and computes GEH per edge.

This file deliberately reads parquets — therefore it must run in a
process that has not yet imported ``libsumo``. Call it after
:meth:`SumoRuntime.close` or in a separate process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leonia_traffic.data.dataset_io import (
    CANONICAL_DIR,
    CanonicalFiles,
)
from leonia_traffic.simulation.calibration import (
    CalibrationScore,
    geh_array,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily-average → window vph conversion
# ---------------------------------------------------------------------------


# Default share of a daily total that lands in each named window. Mirrors
# BRIDGE_OD_WINDOWS in :mod:`leonia_traffic.sumo.demand_builder`. The
# values are taken from the StreetLight ZA hourly distributions on a
# canonical Leonia tertiary segment and round to ~100 % across the day.
DAY_PART_SHARE: dict[str, float] = {
    "all_day":  1.0,
    "early_am": 0.05,
    "peak_am":  0.20,
    "mid_day":  0.30,
    "peak_pm":  0.27,
    "late_pm":  0.18,
}

DAY_PART_HOURS: dict[str, int] = {
    "all_day":  24,
    "early_am": 6,
    "peak_am":  4,
    "mid_day":  5,
    "peak_pm":  4,
    "late_pm":  5,
}


def daily_to_vph(daily_total: float, day_part: str = "peak_am") -> float:
    """Convert a StreetLight daily total to a per-hour rate for a window."""
    share = DAY_PART_SHARE.get(day_part, DAY_PART_SHARE["peak_am"])
    hours = DAY_PART_HOURS.get(day_part, DAY_PART_HOURS["peak_am"])
    if hours <= 0:
        return 0.0
    return float(daily_total) * share / hours


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------


@dataclass
class SumoScore:
    """Same shape as :class:`CalibrationScore` so reports can reuse logic."""

    score: CalibrationScore
    scoring_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Score a SUMO run
# ---------------------------------------------------------------------------


def _load_observed_segments(source: str = "weekdays") -> pd.DataFrame:
    """Pull the StreetLight Street Scanner table down to one row per OSM way."""
    ss_path = CANONICAL_DIR / CanonicalFiles.streetscanner_segments
    if not ss_path.exists():
        logger.warning("streetscanner_segments.parquet not found at %s",
                       ss_path)
        return pd.DataFrame()
    import geopandas as gpd

    gdf = gpd.read_parquet(ss_path)
    if "source" in gdf.columns and source in set(gdf["source"].unique()):
        sub = gdf[gdf["source"] == source]
    else:
        sub = gdf
    sub = sub[(sub.get("day_type", "All Days") == "All Days")
              & (sub.get("day_part_raw", "All Day") == "All Day")]
    if sub.empty:
        sub = gdf
    obs = (
        sub.groupby("osm_way_id", dropna=True, as_index=False)
        .agg(
            observed_daily_volume=("avg_volume", "mean"),
            observed_speed_mph=("avg_speed_mph", "mean"),
            speed_limit_mph=("speed_limit_mph", "mean"),
            road_class=("road_class", "first"),
            osm_name=("osm_name", "first"),
        )
    )
    return obs


def score_sumo_run(
    edge_summary: pd.DataFrame,
    *,
    source: str = "weekdays",
    day_part: str = "peak_am",
    min_observed_daily: float = 50.0,
) -> SumoScore:
    """Compute GEH per SUMO edge against StreetLight Street Scanner.

    Parameters
    ----------
    edge_summary
        DataFrame from :meth:`SumoRuntime.edge_summary` (must contain
        ``sumo_edge_id``, ``osm_way_id``, ``peak_vph``,
        ``mean_speed_mph``).
    source
        StreetLight Street Scanner export label — one of
        ``"all_days"`` / ``"weekdays"`` / ``"weekend"``.
    day_part
        Which window the SUMO run represents — used to convert the
        StreetLight daily total to a per-hour reference rate.
    min_observed_daily
        Drop edges whose observed daily volume is below this threshold
        (avoids small-N inflation of GEH).

    Returns
    -------
    SumoScore
        ``score`` is a :class:`CalibrationScore` (same shape as
        the UXsim version), ``scoring_df`` is the per-edge join with
        ``sim_vph``, ``observed_vph``, ``geh``, ``street_name``.
    """
    if edge_summary is None or edge_summary.empty:
        empty_df = pd.DataFrame(
            columns=["sumo_edge_id", "osm_way_id", "street_name",
                     "sim_vph", "observed_vph", "geh"]
        )
        return SumoScore(
            score=CalibrationScore(
                geh_mean=float("nan"), geh_median=float("nan"),
                geh_p85=float("nan"), pct_lt_5=0.0, pct_lt_10=0.0,
                n_links_scored=0,
            ),
            scoring_df=empty_df,
        )

    obs = _load_observed_segments(source)
    if obs.empty:
        empty_df = pd.DataFrame(
            columns=["sumo_edge_id", "osm_way_id", "street_name",
                     "sim_vph", "observed_vph", "geh"]
        )
        return SumoScore(
            score=CalibrationScore(
                geh_mean=float("nan"), geh_median=float("nan"),
                geh_p85=float("nan"), pct_lt_5=0.0, pct_lt_10=0.0,
                n_links_scored=0,
            ),
            scoring_df=empty_df,
        )

    # Aggregate sim_vph to one value per OSM way (a way can split into
    # multiple SUMO edges; we use the max as the representative count).
    sim = edge_summary.dropna(subset=["osm_way_id"]).copy()
    sim["osm_way_id"] = sim["osm_way_id"].astype("Int64")
    sim_per_way = (
        sim.groupby("osm_way_id", as_index=False)
        .agg(
            sim_vph=("peak_vph", "max"),
            sim_mean_speed_mph=("mean_speed_mph", "mean"),
            street_name=("street_name", "first"),
        )
    )
    obs["osm_way_id"] = obs["osm_way_id"].astype("Int64")
    joined = sim_per_way.merge(obs, on="osm_way_id", how="inner")
    joined = joined[joined["observed_daily_volume"] >= min_observed_daily]
    joined["observed_vph"] = joined["observed_daily_volume"].apply(
        lambda v: daily_to_vph(v, day_part=day_part)
    )
    joined["geh"] = geh_array(
        joined["sim_vph"].to_numpy(),
        joined["observed_vph"].to_numpy(),
    )
    joined = joined.dropna(subset=["geh"])

    if joined.empty:
        score = CalibrationScore(
            geh_mean=float("nan"), geh_median=float("nan"),
            geh_p85=float("nan"), pct_lt_5=0.0, pct_lt_10=0.0,
            n_links_scored=0,
        )
    else:
        score = CalibrationScore(
            geh_mean=float(joined["geh"].mean()),
            geh_median=float(joined["geh"].median()),
            geh_p85=float(joined["geh"].quantile(0.85)),
            pct_lt_5=float((joined["geh"] < 5).mean()),
            pct_lt_10=float((joined["geh"] < 10).mean()),
            n_links_scored=int(len(joined)),
        )
    return SumoScore(score=score, scoring_df=joined.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def write_run_outputs(
    run_dir: Path,
    *,
    edge_history: pd.DataFrame,
    edge_summary: pd.DataFrame,
    scoring_df: pd.DataFrame,
    score: CalibrationScore,
    manifest: dict,
) -> None:
    """Write the canonical per-run output files.

    All files land under ``run_dir`` (created if missing):

    * ``edge_history.parquet`` — long-format per-edge per-bin counters.
    * ``edge_summary.parquet`` — one row per edge.
    * ``scoring.parquet`` — GEH join with StreetLight.
    * ``manifest.json`` — caller-supplied run metadata + score.
    """
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    if edge_history is not None and not edge_history.empty:
        edge_history.to_parquet(run_dir / "edge_history.parquet", index=False)
    if edge_summary is not None and not edge_summary.empty:
        edge_summary.to_parquet(run_dir / "edge_summary.parquet", index=False)
    if scoring_df is not None and not scoring_df.empty:
        scoring_df.to_parquet(run_dir / "scoring.parquet", index=False)
    full_manifest = dict(manifest)
    full_manifest["score"] = {
        "geh_mean": score.geh_mean,
        "geh_median": score.geh_median,
        "geh_p85": score.geh_p85,
        "pct_lt_5": score.pct_lt_5,
        "pct_lt_10": score.pct_lt_10,
        "n_links_scored": score.n_links_scored,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(full_manifest, indent=2, sort_keys=True, default=str)
    )
