"""Per-residential-street cut-through analytics (Pass C.1).

Consumes the long-format frames produced by
``leonia_traffic.data.za_streets_loader`` and emits per-zone metrics
that, when combined into a composite index, defensibly rank Leonia's
residential streets by their cut-through severity.

The analyses use the **Visitors** filter exclusively — those are
trips by drivers whose home block-group is outside the analysis zone
set, i.e. non-resident pass-through traffic. The original raw frames
contain both Visitor and Resident rows; helpers in this module assume
the caller has filtered (or will pass already-filtered frames).

Each function returns a flat per-zone DataFrame keyed by
``zone_name``/``osm_way_id``/``street_name`` so the per-street report
and recommendation engine can join them on the same keys.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Day-type / day-part constants
# ---------------------------------------------------------------------------

ALL_DAY_PART = 0          # "00: All Day (12am-12am)"
ALL_DAYS_TYPE = 0         # "0: All Days (M-Su)"
WEEKDAY_TYPES = (1, 2, 3, 4)   # Monday..Thursday (the export omits Friday)
THURSDAY_TYPE = 4
SATURDAY_TYPE = 5
SUNDAY_TYPE = 6
WEEKEND_TYPES = (SATURDAY_TYPE, SUNDAY_TYPE)

# Day_part codes 1..24 represent hours 12am..11pm. Code 1 = 12am-1am,
# so code = hour + 1.  Peak AM = 7-10am = codes 8,9,10.  Peak PM =
# 4-7pm = codes 17,18,19.  Midday = 11am-2pm = codes 12,13,14 (the
# off-peak baseline against which peak intensity is measured).
PEAK_AM_HOURS = (8, 9, 10)
PEAK_PM_HOURS = (17, 18, 19)
MIDDAY_HOURS = (12, 13, 14)
ALL_HOURLY_CODES = tuple(range(1, 25))


KEY_COLUMNS = ("zone_name", "street_name", "osm_way_id")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _ensure_visitors(df: pd.DataFrame) -> pd.DataFrame:
    if "filter" in df.columns:
        return df[df["filter"] == "Visitors"].copy()
    return df.copy()


def _zone_keys(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in KEY_COLUMNS if c in df.columns]
    return df[keep].drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Volume / temporal ratios
# ---------------------------------------------------------------------------


def weekday_weekend_imbalance(za_df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone Thursday vs Saturday all-day Visitor volume ratio.

    A ratio significantly above 1 is the canonical "commuter
    cut-through" signature. Returns one row per zone with:

    * ``thursday_volume`` — Thursday all-day Visitor volume.
    * ``saturday_volume`` — Saturday all-day Visitor volume.
    * ``weekday_weekend_ratio`` — thursday / max(saturday, 1).
    """
    if za_df is None or za_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(za_df)
    sub = df[df["day_part_code"] == ALL_DAY_PART]
    piv = sub.pivot_table(
        index=list(KEY_COLUMNS),
        columns="day_type_code",
        values="zone_volume",
        aggfunc="mean",
    )
    out = piv.reset_index()
    out["thursday_volume"] = out.get(THURSDAY_TYPE, pd.Series(dtype=float))
    out["saturday_volume"] = out.get(SATURDAY_TYPE, pd.Series(dtype=float))
    out["weekday_weekend_ratio"] = (
        out["thursday_volume"].astype(float)
        / out["saturday_volume"].astype(float).clip(lower=1.0)
    )
    return out[
        list(KEY_COLUMNS) + ["thursday_volume", "saturday_volume", "weekday_weekend_ratio"]
    ].sort_values("weekday_weekend_ratio", ascending=False).reset_index(drop=True)


def _peak_volume(
    za_df: pd.DataFrame,
    hours: tuple[int, ...],
    out_col: str,
    *,
    day_types: tuple[int, ...] = WEEKDAY_TYPES,
) -> pd.DataFrame:
    df = _ensure_visitors(za_df)
    sub = df[df["day_type_code"].isin(day_types)
             & df["day_part_code"].isin(hours)]
    agg = sub.groupby(list(KEY_COLUMNS), dropna=False, as_index=False)[
        "zone_volume"
    ].sum().rename(columns={"zone_volume": out_col})
    return agg.sort_values(out_col, ascending=False).reset_index(drop=True)


def peak_am_volume(
    za_df: pd.DataFrame,
    *,
    day_types: tuple[int, ...] = WEEKDAY_TYPES,
) -> pd.DataFrame:
    """Peak-AM (7-10am) summed Visitor volume per zone.

    Defaults to Mon-Thu day types (the typical-weekday convention
    used elsewhere in the framework). Pass ``day_types=(0,)`` to use
    the All-Days aggregate when broader coverage is preferred over a
    strict weekday filter.
    """
    if za_df is None or za_df.empty:
        return pd.DataFrame()
    return _peak_volume(za_df, PEAK_AM_HOURS, "peak_am_volume",
                        day_types=day_types)


def peak_pm_volume(
    za_df: pd.DataFrame,
    *,
    day_types: tuple[int, ...] = WEEKDAY_TYPES,
) -> pd.DataFrame:
    """Peak-PM (4-7pm) summed Visitor volume per zone."""
    if za_df is None or za_df.empty:
        return pd.DataFrame()
    return _peak_volume(za_df, PEAK_PM_HOURS, "peak_pm_volume",
                        day_types=day_types)


# Minimum per-hour volume below which the peak-intensity ratio is
# considered unreliable (small-denominator noise). A ratio of "20× from
# 1 trip/hr to 20 trips/hr" is meaningless on a residential street.
_MIN_BASELINE_PER_HOUR = 5.0
_MIN_PEAK_PER_HOUR = 5.0


def peak_hour_intensity(
    za_df: pd.DataFrame,
    *,
    peak_hours: tuple[int, ...] = PEAK_AM_HOURS,
    baseline_hours: tuple[int, ...] = MIDDAY_HOURS,
    day_types: tuple[int, ...] = WEEKDAY_TYPES,
    min_per_hour: float = _MIN_BASELINE_PER_HOUR,
) -> pd.DataFrame:
    """Ratio of peak-hour to midday-hour Visitor volume, per zone.

    A residential street whose Visitor volume jumps 3-5× from midday
    to AM peak is exhibiting a commuter cut-through signature; a true
    "local errands" street stays flat across the day. Returns one row
    per zone with:

    * ``peak_total`` — summed Visitor volume across ``peak_hours``.
    * ``peak_per_hr`` — peak total divided by number of peak hours.
    * ``baseline_total`` — summed Visitor volume across ``baseline_hours``.
    * ``baseline_per_hr`` — baseline total divided by number of baseline hours.
    * ``peak_intensity`` — ``peak_per_hr / baseline_per_hr``. Set to
      NaN when either rate is below ``min_per_hour`` to avoid
      small-denominator noise (1 trip/hr → 10 trip/hr is *not* a
      meaningful 10× cut-through signal).
    """
    if za_df is None or za_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(za_df)
    df = df[df["day_type_code"].isin(day_types)]
    peak = df[df["day_part_code"].isin(peak_hours)].groupby(
        list(KEY_COLUMNS), dropna=False, as_index=False
    )["zone_volume"].sum().rename(columns={"zone_volume": "peak_total"})
    base = df[df["day_part_code"].isin(baseline_hours)].groupby(
        list(KEY_COLUMNS), dropna=False, as_index=False
    )["zone_volume"].sum().rename(columns={"zone_volume": "baseline_total"})
    out = peak.merge(base, on=list(KEY_COLUMNS), how="outer").fillna(0.0)
    out["peak_per_hr"] = out["peak_total"].astype(float) / max(len(peak_hours), 1)
    out["baseline_per_hr"] = out["baseline_total"].astype(float) / max(len(baseline_hours), 1)
    raw = out["peak_per_hr"] / out["baseline_per_hr"].clip(lower=1e-6)
    mask = (out["peak_per_hr"] >= min_per_hour) & (out["baseline_per_hr"] >= min_per_hour)
    out["peak_intensity"] = raw.where(mask, other=float("nan"))
    return out.sort_values("peak_intensity", ascending=False, na_position="last").reset_index(drop=True)


def weekday_hourly_profile(
    za_df: pd.DataFrame,
    *,
    day_types: tuple[int, ...] = (ALL_DAYS_TYPE,),
) -> pd.DataFrame:
    """Per-zone hourly Visitor-volume profile.

    Defaults to the **All-Days** day type because StreetLight only
    publishes hourly breakdowns for a subset of zones, and the
    All-Days aggregate has noticeably broader coverage than the
    individual weekday day types. Pass
    ``day_types=cutthrough_streets.WEEKDAY_TYPES`` to restrict to
    Mon-Thu, at the cost of fewer zones being reported.

    Returns a wide-format DataFrame keyed by ``zone_name`` with one
    column per hour (``h00``..``h23``) holding the mean Visitor
    volume in that hour across the chosen day types.
    """
    if za_df is None or za_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(za_df)
    df = df[df["day_type_code"].isin(day_types)
            & df["day_part_code"].isin(ALL_HOURLY_CODES)].copy()
    if df.empty:
        return pd.DataFrame()
    df["hour"] = (df["day_part_code"].astype(int) - 1)
    piv = df.pivot_table(
        index=list(KEY_COLUMNS),
        columns="hour",
        values="zone_volume",
        aggfunc="mean",
    )
    piv.columns = [f"h{int(c):02d}" for c in piv.columns]
    return piv.reset_index()


def weekday_all_day_volume(za_df: pd.DataFrame) -> pd.DataFrame:
    """Average weekday all-day Visitor volume per zone (mean over Mon-Thu)."""
    if za_df is None or za_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(za_df)
    sub = df[df["day_type_code"].isin(WEEKDAY_TYPES)
             & (df["day_part_code"] == ALL_DAY_PART)]
    agg = sub.groupby(list(KEY_COLUMNS), dropna=False, as_index=False)[
        "zone_volume"
    ].mean().rename(columns={"zone_volume": "weekday_all_day_volume"})
    return agg.sort_values("weekday_all_day_volume", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trip-attribute shares (operates on the per-zone trip CSV)
# ---------------------------------------------------------------------------


_LONG_TRIP_BINS_5MI = (
    "len_mi_5_10", "len_mi_10_20", "len_mi_20_30",
    "len_mi_30_40", "len_mi_40_50", "len_mi_50_60",
    "len_mi_60_70", "len_mi_70_80", "len_mi_80_90",
    "len_mi_90_100", "len_mi_100_plus",
)

_LONG_TRIP_BINS_10MI = (
    "len_mi_10_20", "len_mi_20_30",
    "len_mi_30_40", "len_mi_40_50", "len_mi_50_60",
    "len_mi_60_70", "len_mi_70_80", "len_mi_80_90",
    "len_mi_90_100", "len_mi_100_plus",
)

_SPEEDING_BINS_30 = (
    "spd_mph_30_40", "spd_mph_40_50", "spd_mph_50_60",
    "spd_mph_60_70", "spd_mph_70_plus",
)

_SPEEDING_BINS_25 = (
    "spd_mph_20_30", "spd_mph_30_40", "spd_mph_40_50",
    "spd_mph_50_60", "spd_mph_60_70", "spd_mph_70_plus",
)


def _sum_bins(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return df[cols].fillna(0).sum(axis=1)


def long_trip_share(trip_df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone share of pass-through trips that are long.

    Uses the all-day-all-days Visitor row per zone as the canonical
    distribution. Returns columns:

    * ``long_trip_share_5mi`` — share of trips with length >= 5 mi.
    * ``long_trip_share_10mi`` — share of trips with length >= 10 mi.
    * ``avg_trip_length_mi`` — passthrough (taken from the trip CSV).
    """
    if trip_df is None or trip_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(trip_df)
    sub = df[(df["day_type_code"] == ALL_DAYS_TYPE)
             & (df["day_part_code"] == ALL_DAY_PART)].copy()
    sub["long_trip_share_5mi"] = _sum_bins(sub, _LONG_TRIP_BINS_5MI)
    sub["long_trip_share_10mi"] = _sum_bins(sub, _LONG_TRIP_BINS_10MI)
    keep = list(KEY_COLUMNS) + [
        "long_trip_share_5mi", "long_trip_share_10mi",
    ]
    if "avg_trip_length_mi" in sub.columns:
        keep.append("avg_trip_length_mi")
    return sub[keep].drop_duplicates(subset=list(KEY_COLUMNS)).reset_index(drop=True)


def speeding_share(
    trip_df: pd.DataFrame,
    *,
    posted_speed_mph: int = 25,
) -> pd.DataFrame:
    """Per-zone share of pass-through trips above the posted speed.

    Because StreetLight reports speeds in 10-mph bins, we approximate
    "above 25 mph" by summing all bins at 20-30 mph and above (since
    half the 20-30 bin is technically below the posted limit, this is
    a slight over-estimate). For posted_speed_mph >= 30 the lower
    bound is the 30-40 mph bin which is a clean lower bound on
    "speeding".

    Returns columns ``speeding_share`` (0..1) and ``posted_speed_mph``.
    """
    if trip_df is None or trip_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(trip_df)
    sub = df[(df["day_type_code"] == ALL_DAYS_TYPE)
             & (df["day_part_code"] == ALL_DAY_PART)].copy()
    if posted_speed_mph < 30:
        bins = _SPEEDING_BINS_25
    else:
        bins = _SPEEDING_BINS_30
    sub["speeding_share"] = _sum_bins(sub, bins)
    sub["posted_speed_mph"] = int(posted_speed_mph)
    return sub[list(KEY_COLUMNS) + ["speeding_share", "posted_speed_mph"]].drop_duplicates(
        subset=list(KEY_COLUMNS)
    ).reset_index(drop=True)


def circuity_share(trip_df: pd.DataFrame) -> pd.DataFrame:
    """Share of pass-through trips with circuity >= 2 (detour) and >= 3."""
    if trip_df is None or trip_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(trip_df)
    sub = df[(df["day_type_code"] == ALL_DAYS_TYPE)
             & (df["day_part_code"] == ALL_DAY_PART)].copy()
    sub["circuity_share_ge2"] = _sum_bins(sub, (
        "circuity_2_3", "circuity_3_4", "circuity_4_5",
        "circuity_5_6", "circuity_6_plus",
    ))
    sub["circuity_share_ge3"] = _sum_bins(sub, (
        "circuity_3_4", "circuity_4_5",
        "circuity_5_6", "circuity_6_plus",
    ))
    return sub[list(KEY_COLUMNS) + [
        "circuity_share_ge2", "circuity_share_ge3"
    ]].drop_duplicates(subset=list(KEY_COLUMNS)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Home-location signals
# ---------------------------------------------------------------------------


def non_local_home_share(home_dist_df: pd.DataFrame) -> pd.DataFrame:
    """Share of Visitor trips with home located > 3 mi from the zone.

    Returns:

    * ``home_le_1mi_share``
    * ``home_le_3mi_share``  (i.e. < 3 mi from the zone)
    * ``non_local_home_share``  (≥ 3 mi)
    """
    if home_dist_df is None or home_dist_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(home_dist_df)
    sub = df[(df["day_type_code"] == ALL_DAYS_TYPE)
             & (df["day_part_code"] == ALL_DAY_PART)].copy()
    close_cols = (
        "Percent Home less than 1 mi",
        "Percent Home 1 to 3 mi",
    )
    far_cols = (
        "Percent Home 3 to 5 mi",
        "Percent Home 5 to 10 mi",
        "Percent Home 10 to 25 mi",
        "Percent Home 25 to 50 mi",
        "Percent Home 50 to 100 mi",
        "Percent Home more than 100 mi",
    )
    sub["home_le_1mi_share"] = sub.get("Percent Home less than 1 mi", 0.0)
    sub["home_le_3mi_share"] = _sum_bins(sub, close_cols)
    sub["non_local_home_share"] = _sum_bins(sub, far_cols)
    return sub[list(KEY_COLUMNS) + [
        "home_le_1mi_share", "home_le_3mi_share", "non_local_home_share"
    ]].drop_duplicates(subset=list(KEY_COLUMNS)).reset_index(drop=True)


def non_leonia_zip_share(
    home_zips_top_df: pd.DataFrame,
    *,
    leonia_zip: str = "07605",
) -> pd.DataFrame:
    """Share of top-ranked Visitor home ZIPs that are outside Leonia.

    StreetLight pre-ranks the top home ZIPs per zone; this aggregates
    them as a coarse "outsider share" signal. Returns ``leonia_zip_share``
    and its complement ``non_leonia_zip_share``.
    """
    if home_zips_top_df is None or home_zips_top_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(home_zips_top_df)
    sub = df[(df["day_type_code"] == ALL_DAYS_TYPE)
             & (df["day_part_code"] == ALL_DAY_PART)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_zip_str"] = sub["zip_code"].astype(str).str.zfill(5)
    sub["_is_leonia"] = sub["_zip_str"] == leonia_zip
    leonia = sub[sub["_is_leonia"]].groupby(
        list(KEY_COLUMNS), dropna=False, as_index=False
    )["pct_home_location"].sum().rename(
        columns={"pct_home_location": "leonia_zip_share"}
    )
    keys = _zone_keys(sub)
    out = keys.merge(leonia, on=list(KEY_COLUMNS), how="left")
    out["leonia_zip_share"] = out["leonia_zip_share"].fillna(0.0)
    out["non_leonia_zip_share"] = (1.0 - out["leonia_zip_share"]).clip(lower=0.0, upper=1.0)
    return out


# ---------------------------------------------------------------------------
# Composite cut-through index
# ---------------------------------------------------------------------------


# Weights for the composite index. Each component is normalised to
# [0, 1] before weighting, so weights are interpretable as relative
# importance shares. Sum doesn't have to equal 1 — the final value is
# divided by the sum of weights.
COMPOSITE_WEIGHTS = {
    "weekday_weekend_ratio": 0.25,
    "non_local_home_share": 0.25,
    "long_trip_share_5mi": 0.20,
    "speeding_share": 0.15,
    "weekday_all_day_volume": 0.15,
}


def _normalize_minmax(s: pd.Series) -> pd.Series:
    vals = s.astype(float)
    finite = vals.replace([float("inf"), -float("inf")], float("nan")).dropna()
    if finite.empty:
        return pd.Series([0.0] * len(s), index=s.index)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return pd.Series([0.0] * len(s), index=s.index)
    out = (vals - lo) / (hi - lo)
    return out.clip(lower=0.0, upper=1.0).fillna(0.0)


def composite_cutthrough_index(
    *,
    imbalance_df: pd.DataFrame,
    weekday_volume_df: pd.DataFrame,
    long_trip_df: pd.DataFrame,
    speeding_df: pd.DataFrame,
    home_dist_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Join the per-zone signals into a single ranked cut-through index.

    Each input frame must contain the standard ``zone_name``,
    ``street_name``, ``osm_way_id`` keys. Missing zones are tolerated
    via outer-merging — components that have no data for a zone get
    0 for that component.

    Returns one row per zone with all sub-metrics plus
    ``cutthrough_index`` (0..1) and a ``rank`` column (1 = worst).
    """
    weights = weights or COMPOSITE_WEIGHTS

    parts: list[pd.DataFrame] = []
    for d in (imbalance_df, weekday_volume_df, long_trip_df, speeding_df, home_dist_df):
        if d is not None and not d.empty:
            parts.append(d)

    if not parts:
        return pd.DataFrame()

    merged = parts[0].copy()
    for nxt in parts[1:]:
        merged = merged.merge(nxt, on=list(KEY_COLUMNS), how="outer")

    components = {
        "weekday_weekend_ratio": "weekday_weekend_ratio",
        "non_local_home_share": "non_local_home_share",
        "long_trip_share_5mi": "long_trip_share_5mi",
        "speeding_share": "speeding_share",
        "weekday_all_day_volume": "weekday_all_day_volume",
    }

    norm_cols: list[tuple[str, float]] = []
    for key, src in components.items():
        if src not in merged.columns:
            continue
        norm_name = f"_norm_{key}"
        merged[norm_name] = _normalize_minmax(merged[src])
        norm_cols.append((norm_name, float(weights.get(key, 0.0))))

    if not norm_cols:
        return merged

    total_weight = sum(w for _, w in norm_cols)
    if total_weight <= 0:
        merged["cutthrough_index"] = 0.0
    else:
        idx = sum(merged[c].fillna(0.0) * w for c, w in norm_cols) / total_weight
        merged["cutthrough_index"] = idx.clip(lower=0.0, upper=1.0)

    merged = merged.sort_values("cutthrough_index", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    # Drop the throwaway _norm_ helper columns
    drop = [c for c in merged.columns if c.startswith("_norm_")]
    return merged.drop(columns=drop)


def top_origin_zips(
    home_zips_top_df: pd.DataFrame,
    zone_name: str,
    *,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top-N home ZIPs for one zone (Visitors, all-days)."""
    if home_zips_top_df is None or home_zips_top_df.empty:
        return pd.DataFrame()
    df = _ensure_visitors(home_zips_top_df)
    sub = df[(df["day_type_code"] == ALL_DAYS_TYPE)
             & (df["day_part_code"] == ALL_DAY_PART)
             & (df["zone_name"] == zone_name)].copy()
    keep_cols = [c for c in (
        "rank", "zip_code", "zip_primary_state", "zip_primary_metro",
        "pct_home_location", "zone_volume",
    ) if c in sub.columns]
    return sub.sort_values("rank").head(n)[keep_cols].reset_index(drop=True)


__all__ = [
    "ALL_DAY_PART",
    "ALL_DAYS_TYPE",
    "ALL_HOURLY_CODES",
    "WEEKDAY_TYPES",
    "WEEKEND_TYPES",
    "PEAK_AM_HOURS",
    "PEAK_PM_HOURS",
    "MIDDAY_HOURS",
    "THURSDAY_TYPE",
    "SATURDAY_TYPE",
    "COMPOSITE_WEIGHTS",
    "composite_cutthrough_index",
    "circuity_share",
    "long_trip_share",
    "non_leonia_zip_share",
    "non_local_home_share",
    "peak_am_volume",
    "peak_pm_volume",
    "peak_hour_intensity",
    "speeding_share",
    "top_origin_zips",
    "weekday_all_day_volume",
    "weekday_hourly_profile",
    "weekday_weekend_imbalance",
]
