"""Congestion-trends analysis for the Leonia bridge approach.

Pass A.3. Consumes the long-format DataFrame returned by
:func:`leonia_traffic.data.congestion_loader.load_congestion` and produces
operations-grade tables:

* ``worst_hours_per_corridor`` — top hours of the day per corridor by
  Travel Time Index (TTI).
* ``delay_hotspot_ranking`` — segments ranked by total weekday Vehicle
  Hours of Delay (the absolute time-cost-to-the-public metric).
* ``reliability_breakdown`` — counts of segments by reliability
  classification × road class.
* ``link_speed_overrides`` — extractor that produces the (osm_way_id →
  observed 50th-percentile speed) mapping consumed by Pass B's
  :func:`leonia_traffic.network.osm_builder.apply_congestion_overrides`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Day-part codes from the congestion CSV. Hourly codes are 1, 6-13, 15-20, 22-29.
# The named aggregates we care about:
DAY_PART_AGGREGATES = {
    0: "All Day (12am-12am)",
    2: "Early AM (12am-6am)",
    9: "Peak AM (6am-10am)",
    14: "Mid-Day (10am-4pm)",
    21: "Peak PM (4pm-8pm)",
    26: "Late PM (8pm-12am)",
}

# Pattern recognizing hourly day parts like "10: 7am (7am-8am)".
_HOURLY_DAY_PART = re.compile(r"\d{1,2}[ap]m \(")


def _is_hourly(day_part_label: pd.Series) -> pd.Series:
    return day_part_label.fillna("").str.match(_HOURLY_DAY_PART)


def worst_hours_per_corridor(
    df: pd.DataFrame,
    *,
    top_n_per_zone: int = 3,
    min_avg_speed_mph: float = 1.0,
    daytime_only: bool = True,
) -> pd.DataFrame:
    """For each zone, the top ``N`` hourly slots by Travel Time Index.

    Returns columns:
    ``zone_name``, ``osm_name``, ``osm_way_id``, ``road_class``,
    ``day_type_label``, ``day_part_label``, ``tti``, ``buffer_index``,
    ``speed_p50``, ``vhd``, ``avg_speed_mph``, ``free_flow_speed_mph``,
    sorted by ``tti`` descending.

    Only Weekday rows (day_type_code=1) and hourly day-parts are
    considered. Rows with no observed speed (``avg_speed_mph`` below
    ``min_avg_speed_mph``) are dropped because they produce
    pathologically large TTI values when StreetLight imputes from sparse
    overnight signal. Set ``daytime_only=True`` (default) to additionally
    restrict to 6 AM – 8 PM hourly buckets where stakeholders care.
    """
    if df.empty:
        return df

    sub = df[(df["day_type_code"] == 1) & _is_hourly(df["day_part_label"])].copy()
    if sub.empty:
        return pd.DataFrame()

    if min_avg_speed_mph is not None:
        sub = sub[sub["avg_speed_mph"] >= min_avg_speed_mph]
    if daytime_only:
        # Hourly codes 8..28 cover 6am–10pm in the StreetLight schema.
        sub = sub[(sub["day_part_code"] >= 8) & (sub["day_part_code"] <= 28)]
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(["zone_name", "tti"], ascending=[True, False])
    top = sub.groupby("zone_name", group_keys=False).head(top_n_per_zone)

    cols = [
        "zone_name", "osm_name", "osm_way_id", "road_class",
        "day_type_label", "day_part_label",
        "tti", "buffer_index", "speed_p50", "vhd",
        "avg_speed_mph", "free_flow_speed_mph",
    ]
    keep = [c for c in cols if c in top.columns]
    return top[keep].sort_values("tti", ascending=False).reset_index(drop=True)


def delay_hotspot_ranking(df: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Rank zones by total weekday Vehicle Hours of Delay.

    Aggregates over Weekday hourly day-parts (codes 1, 6-13, 15-20, 22-29).
    Returns columns:
    ``zone_name``, ``osm_name``, ``osm_way_id``, ``road_class``,
    ``length_mi``, ``weekday_vhd_total``, ``weekday_vmt_total``,
    ``delay_per_mile``, ``worst_hour_tti``, ``worst_hour_label``.
    """
    if df.empty:
        return df

    sub = df[(df["day_type_code"] == 1) & _is_hourly(df["day_part_label"])].copy()
    if sub.empty:
        return pd.DataFrame()

    grouped = sub.groupby(
        ["zone_name", "osm_name", "osm_way_id", "road_class", "length_mi"],
        dropna=False,
        as_index=False,
    ).agg(
        weekday_vhd_total=("vhd", "sum"),
        weekday_vmt_total=("vmt", "sum"),
    )

    # Worst-hour TTI per zone.
    worst = sub.sort_values("tti", ascending=False).drop_duplicates("zone_name")
    worst = worst[["zone_name", "tti", "day_part_label"]].rename(
        columns={"tti": "worst_hour_tti", "day_part_label": "worst_hour_label"}
    )
    out = grouped.merge(worst, on="zone_name", how="left")

    out["delay_per_mile"] = np.where(
        out["length_mi"] > 0,
        out["weekday_vhd_total"] / out["length_mi"],
        np.nan,
    )

    return out.sort_values("weekday_vhd_total", ascending=False).head(top_n).reset_index(drop=True)


def reliability_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of segments by reliability class × road class.

    Uses the All-Days × All-Day row per zone as the canonical
    classification, bucketing the numeric ``reliability_level`` (LOTTR)
    into ``Reliable`` (<1.5), ``Moderate`` (1.5–2.0), and ``Unreliable``
    (>=2.0). Returns a long DataFrame with columns ``road_class``,
    ``reliability_class``, ``n_segments``, ``share_of_road_class``.
    """
    from leonia_traffic.data.congestion_loader import classify_reliability

    if df.empty:
        return df

    canon = df[(df["day_type_code"] == 0) & (df["day_part_code"] == 0)].copy()
    if canon.empty:
        canon = df[(df["day_type_code"] == 1) & (df["day_part_code"] == 0)].copy()
    if canon.empty:
        return pd.DataFrame()

    canon["reliability_class"] = canon["reliability_level"].apply(classify_reliability)
    grouped = canon.groupby(["road_class", "reliability_class"], dropna=False, as_index=False).size()
    grouped = grouped.rename(columns={"size": "n_segments"})

    totals = grouped.groupby("road_class")["n_segments"].transform("sum")
    grouped["share_of_road_class"] = grouped["n_segments"] / totals

    rank = {"Reliable": 0, "Moderate": 1, "Unreliable": 2, "Unknown": 3}
    grouped["_order"] = grouped["reliability_class"].map(rank).fillna(99)
    out = grouped.sort_values(["road_class", "_order"]).drop(columns="_order").reset_index(drop=True)
    return out


def link_speed_overrides(
    df: pd.DataFrame,
    *,
    day_type_code: int = 1,
    day_part_code: int = 9,
    speed_field: str = "speed_p50",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Build an (osm_way_id → observed speed) mapping for UXsim Pass B.

    Parameters
    ----------
    df
        Long-format congestion DataFrame from ``load_congestion``.
    day_type_code
        Day type to filter on. Defaults to ``1`` (Weekday).
    day_part_code
        Day part to filter on. Defaults to ``9`` (Peak AM aggregate).
    speed_field
        Which speed column to use for the override. Defaults to
        ``speed_p50`` (median observed speed). Could also be
        ``avg_speed_mph`` (mean) or ``free_flow_speed_mph``.
    cache_path
        Optional parquet path; if provided the result is written there
        (parent directories created).

    Returns
    -------
    DataFrame
        Columns ``osm_way_id``, ``osm_name``, ``road_class``,
        ``observed_speed_mph``, ``free_flow_speed_mph``,
        ``observed_speed_ms`` (m/s, ready for UXsim's
        ``change_free_flow_speed``), and ``source_zone_name``.
        One row per unique OSM way ID (duplicates collapsed by averaging
        the speed across segments sharing the same way ID).
    """
    if df.empty:
        return pd.DataFrame()

    sub = df[(df["day_type_code"] == day_type_code) & (df["day_part_code"] == day_part_code)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub[sub["osm_way_id"].notna()]
    sub["osm_way_id"] = sub["osm_way_id"].astype("int64")

    grouped = sub.groupby("osm_way_id", as_index=False).agg(
        osm_name=("osm_name", "first"),
        road_class=("road_class", "first"),
        observed_speed_mph=(speed_field, "mean"),
        free_flow_speed_mph=("free_flow_speed_mph", "mean"),
        source_zone_name=("zone_name", "first"),
    )
    grouped["observed_speed_ms"] = grouped["observed_speed_mph"] * 0.44704

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_parquet(cache_path)
        logger.info("Wrote %d link speed overrides to %s", len(grouped), cache_path)

    return grouped


__all__ = [
    "DAY_PART_AGGREGATES",
    "delay_hotspot_ranking",
    "link_speed_overrides",
    "reliability_breakdown",
    "worst_hours_per_corridor",
]
