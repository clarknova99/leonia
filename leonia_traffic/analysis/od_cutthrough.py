"""OD-driven cut-through evidence for the bridge-destination dataset.

Pass A.2 of the bridge OD + congestion plan. Each function takes the
long-format DataFrame returned by
:func:`leonia_traffic.data.bridge_od_loader.load_bridge_od` (or the wide
joined frame from ``load_bridge_attributes``) and produces a tidy table
that is directly defensible to non-technical stakeholders.

Cut-through claims supported here:

* ``gateway_peak_imbalance`` — commuter cut-through manifests as a high
  weekday-Peak-AM / Saturday volume ratio. Pure neighborhood traffic
  should look much flatter.
* ``cutthrough_index_from_circuity`` — drivers who detour onto Leonia
  side streets to bypass arterials show up in the >2 circuity buckets.
* ``day_of_week_profile`` — a heatmap-shaped table that reveals
  Thursday/Friday peaks vs. flat weekends.
* ``trip_purpose_decomposition`` — commuter (Home-to-Work) vs.
  discretionary share by gate × day-part.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leonia_traffic.data.bridge_od_loader import (
    DAY_PART_CODES,
    DAY_TYPE_CODES,
    WEEKDAY_CODES,
    WEEKEND_CODES,
)

logger = logging.getLogger(__name__)


# Day-Part codes from the StreetLight export. The 2036064_Destinations
# analysis (active source as of 2026-05) uses 24 hourly day parts —
# code N covers the hour ``[N-1, N)``. Code 0 is the All-Day total.
#
# These wall-clock-aligned ranges replace the legacy 5-window codes
# (1=Early AM, 2=Peak AM, 3=Mid-Day, 4=Peak PM, 5=Late PM) so the
# rest of the analysis stack can keep referring to "Peak AM" /
# "Peak PM" semantically without caring whether the underlying
# export is 5-window or 24-hour.
PEAK_AM_CODES: list[int] = [7, 8, 9, 10]    # 6am-10am inclusive
PEAK_PM_CODES: list[int] = [16, 17, 18, 19]  # 3pm-7pm inclusive
MID_DAY_CODES: list[int] = [11, 12, 13, 14, 15]  # 10am-3pm

# Single-code aliases — the *busiest* hour in each window. Use these
# when a function genuinely needs one code (e.g. when filtering an
# attribute table whose row already represents a single day-part).
# 7-8am is the canonical Peak AM commuter hour for Leonia.
PEAK_AM_CODE: int = 8     # 7am-8am
PEAK_PM_CODE: int = 18    # 5pm-6pm
MID_DAY_CODE: int = 13    # 12pm-1pm


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.div(den.replace(0, np.nan))


def gateway_peak_imbalance(od_df: pd.DataFrame) -> pd.DataFrame:
    """Weekday peak vs. weekend volume per origin gate.

    Returns one row per origin gate with columns:

    * ``origin_zone``, ``origin_label``, ``origin_osm_way_id``
    * ``peak_am_weekday_avg`` — mean OD volume across Mon–Fri Peak-AM rows
    * ``peak_pm_weekday_avg`` — mean OD volume across Mon–Fri Peak-PM rows
    * ``weekend_peak_am_avg`` — mean of Sat & Sun Peak-AM volumes
    * ``weekend_all_day_avg`` — mean of Sat & Sun all-day volumes
    * ``weekday_all_day_avg`` — mean of Mon–Fri all-day volumes
    * ``peak_am_to_weekend_ratio`` — the headline cut-through signal
    * ``peak_pm_to_weekend_ratio``
    * ``weekday_to_weekend_ratio``

    Volumes are summed across destination zones so a gate's number is its
    total contribution to bridge-bound traffic.
    """
    df = od_df.copy()

    # Sum across all destinations per origin × day-type × day-part.
    agg = df.groupby(
        ["origin_zone", "origin_label", "origin_osm_way_id", "day_type_code", "day_part_code"],
        dropna=False,
        as_index=False,
    ).agg(volume=("od_volume", "sum"))

    def _mean(filtered: pd.DataFrame) -> pd.Series:
        return filtered.groupby(["origin_zone", "origin_label", "origin_osm_way_id"], dropna=False)["volume"].mean()

    # Peak AM / Peak PM are 4-hour ranges in the new 24-window
    # schema. ``_mean`` averages across all matched (day-type,
    # day-part) rows per origin, so summing across the 4 hours in
    # the range and averaging gives the same per-hour rate the
    # legacy single-code lookup produced.
    peak_am_wd = _mean(agg[
        agg["day_type_code"].isin(WEEKDAY_CODES)
        & agg["day_part_code"].isin(PEAK_AM_CODES)])
    peak_pm_wd = _mean(agg[
        agg["day_type_code"].isin(WEEKDAY_CODES)
        & agg["day_part_code"].isin(PEAK_PM_CODES)])
    weekend_peak_am = _mean(agg[
        agg["day_type_code"].isin(WEEKEND_CODES)
        & agg["day_part_code"].isin(PEAK_AM_CODES)])
    weekend_all_day = _mean(agg[(agg["day_type_code"].isin(WEEKEND_CODES)) & (agg["day_part_code"] == 0)])
    weekday_all_day = _mean(agg[(agg["day_type_code"].isin(WEEKDAY_CODES)) & (agg["day_part_code"] == 0)])

    out = pd.DataFrame({
        "peak_am_weekday_avg": peak_am_wd,
        "peak_pm_weekday_avg": peak_pm_wd,
        "weekend_peak_am_avg": weekend_peak_am,
        "weekend_all_day_avg": weekend_all_day,
        "weekday_all_day_avg": weekday_all_day,
    }).reset_index()

    out["peak_am_to_weekend_ratio"] = _safe_div(out["peak_am_weekday_avg"], out["weekend_peak_am_avg"])
    out["peak_pm_to_weekend_ratio"] = _safe_div(out["peak_pm_weekday_avg"], out["weekend_all_day_avg"])
    out["weekday_to_weekend_ratio"] = _safe_div(out["weekday_all_day_avg"], out["weekend_all_day_avg"])

    return out.sort_values("peak_am_weekday_avg", ascending=False).reset_index(drop=True)


def cutthrough_index_from_circuity(
    attr_df: pd.DataFrame,
    *,
    weekday_only: bool = True,
    peak_am_only: bool = True,
) -> pd.DataFrame:
    """Per-gate cut-through index from circuity buckets.

    StreetLight reports trip share by circuity bucket (1-2, 2-3, 3-4,
    4-5, 5-6, 6+). Trips with circuity > 2 traveled materially farther
    than the straight-line distance, which is a strong proxy for
    detoured/cut-through behavior.

    Returns one row per origin gate with:

    * ``circuity_low_pct`` — share of trips in 1–2 bucket (direct).
    * ``circuity_mid_pct`` — share in 2–3.
    * ``circuity_high_pct`` — share in 3–4, 4–5, 5–6, 6+.
    * ``cutthrough_circuity_index`` — share with circuity > 2.

    Parameters
    ----------
    attr_df
        Wide-frame from ``load_bridge_attributes`` (must contain the
        ``trip_stats::Circuity *`` columns).
    weekday_only
        Restrict to Day Type codes 1–5.
    peak_am_only
        Restrict to Day Part code 2 (Peak AM).
    """
    df = attr_df.copy()
    if weekday_only:
        df = df[df["day_type_code"].isin(WEEKDAY_CODES)]
    if peak_am_only:
        df = df[df["day_part_code"].isin(PEAK_AM_CODES)]

    bucket_cols = {
        "circuity_low_pct": "trip_stats::Circuity 1-2 (percent)",
        "circuity_mid_pct": "trip_stats::Circuity 2-3 (percent)",
        "circuity_3_4_pct": "trip_stats::Circuity 3-4 (percent)",
        "circuity_4_5_pct": "trip_stats::Circuity 4-5 (percent)",
        "circuity_5_6_pct": "trip_stats::Circuity 5-6 (percent)",
        "circuity_6plus_pct": "trip_stats::Circuity 6+ (percent)",
    }

    available = {k: v for k, v in bucket_cols.items() if v in df.columns}
    if not available:
        logger.warning("No circuity columns found in attribute frame")
        return pd.DataFrame()

    # Volume-weight the bucket shares across the included rows.
    df = df.assign(_w=df["od_volume"].fillna(0))
    grouped = df.groupby(
        ["origin_zone", "origin_label", "origin_osm_way_id"],
        dropna=False,
    )

    rows: list[dict] = []
    for keys, sub in grouped:
        w = sub["_w"].sum()
        rec: dict[str, float | int | str | None] = {
            "origin_zone": keys[0],
            "origin_label": keys[1],
            "origin_osm_way_id": keys[2],
            "trips_in_window": w,
        }
        if w <= 0:
            for out_col in available:
                rec[out_col] = np.nan
        else:
            for out_col, src in available.items():
                rec[out_col] = (sub[src].fillna(0) * sub["_w"]).sum() / w
        rec["circuity_high_pct"] = (
            (rec.get("circuity_3_4_pct") or 0)
            + (rec.get("circuity_4_5_pct") or 0)
            + (rec.get("circuity_5_6_pct") or 0)
            + (rec.get("circuity_6plus_pct") or 0)
        )
        rec["cutthrough_circuity_index"] = (rec.get("circuity_mid_pct") or 0) + rec["circuity_high_pct"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("cutthrough_circuity_index", ascending=False).reset_index(drop=True)


def day_of_week_profile(
    od_df: pd.DataFrame,
    *,
    day_part_code: int | list[int] = 0,
) -> pd.DataFrame:
    """Origin × day-of-week matrix of OD volumes for a fixed day-part window.

    Returns a wide DataFrame with one row per origin gate and columns
    ``Mon``, ``Tue``, ``Wed``, ``Thu``, ``Fri``, ``Sat``, ``Sun`` plus
    ``weekday_avg`` and ``weekend_avg``. Volumes are summed across all
    destinations and across the supplied day-part codes (so passing
    ``[7, 8, 9, 10]`` for the 24-hour Peak AM range correctly sums
    the four hourly slices into one Peak-AM total per day).

    Default ``day_part_code=0`` (All Day) gives the cleanest weekly
    rhythm; pass ``PEAK_AM_CODES`` for Peak AM, etc.
    """
    codes = (
        [int(day_part_code)]
        if isinstance(day_part_code, (int, np.integer))
        else [int(c) for c in day_part_code]
    )
    df = od_df[od_df["day_part_code"].isin(codes)].copy()

    agg = df.groupby(
        ["origin_zone", "origin_label", "origin_osm_way_id", "day_type_code"],
        dropna=False,
        as_index=False,
    ).agg(volume=("od_volume", "sum"))

    code_to_label = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    wide = agg[agg["day_type_code"].isin(code_to_label)].pivot_table(
        index=["origin_zone", "origin_label", "origin_osm_way_id"],
        columns="day_type_code",
        values="volume",
        aggfunc="sum",
    ).rename(columns=code_to_label).reset_index()

    for col in code_to_label.values():
        if col not in wide.columns:
            wide[col] = np.nan

    wide["weekday_avg"] = wide[["Mon", "Tue", "Wed", "Thu", "Fri"]].mean(axis=1)
    wide["weekend_avg"] = wide[["Sat", "Sun"]].mean(axis=1)
    wide["weekday_to_weekend_ratio"] = _safe_div(wide["weekday_avg"], wide["weekend_avg"])

    cols = ["origin_zone", "origin_label", "origin_osm_way_id",
            "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
            "weekday_avg", "weekend_avg", "weekday_to_weekend_ratio"]
    return wide[cols].sort_values("weekday_avg", ascending=False).reset_index(drop=True)


def trip_purpose_decomposition(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Trip-purpose share per gate × day-part (volume-weighted).

    Returns columns:
    ``origin_zone``, ``origin_label``, ``origin_osm_way_id``,
    ``day_part_code``, ``day_part_label``,
    ``home_to_work_pct``, ``home_to_other_pct``, ``non_home_based_pct``,
    ``weekday_trips`` (sum of OD volume across Mon–Fri rows).
    """
    cols_map = {
        "home_to_work_pct": "trip_purpose::Home to Work",
        "home_to_other_pct": "trip_purpose::Home to Other",
        "non_home_based_pct": "trip_purpose::Non-Home Based Trip",
    }
    available = {k: v for k, v in cols_map.items() if v in attr_df.columns}
    if not available:
        logger.warning("trip_purpose columns missing; returning empty frame")
        return pd.DataFrame()

    df = attr_df[attr_df["day_type_code"].isin(WEEKDAY_CODES)].copy()
    df["_w"] = df["od_volume"].fillna(0)

    grouped = df.groupby(
        ["origin_zone", "origin_label", "origin_osm_way_id",
         "day_part_code", "day_part_label"],
        dropna=False,
    )

    rows: list[dict] = []
    for keys, sub in grouped:
        w = sub["_w"].sum()
        rec: dict[str, float | int | str | None] = dict(zip(
            ["origin_zone", "origin_label", "origin_osm_way_id",
             "day_part_code", "day_part_label"], keys))
        rec["weekday_trips"] = w
        if w <= 0:
            for out_col in available:
                rec[out_col] = np.nan
        else:
            for out_col, src in available.items():
                rec[out_col] = (sub[src].fillna(0) * sub["_w"]).sum() / w
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out.sort_values(["day_part_code", "weekday_trips"], ascending=[True, False]).reset_index(drop=True)


__all__ = [
    "PEAK_AM_CODE",
    "PEAK_PM_CODE",
    "MID_DAY_CODE",
    "PEAK_AM_CODES",
    "PEAK_PM_CODES",
    "MID_DAY_CODES",
    "cutthrough_index_from_circuity",
    "day_of_week_profile",
    "gateway_peak_imbalance",
    "trip_purpose_decomposition",
]
