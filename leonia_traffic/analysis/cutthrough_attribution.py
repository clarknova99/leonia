"""Cut-through attribution analytics built on the O-D + Middle-Filter export.

The OMD export tells us, for every Leonia tertiary street, **which O-D
pairs use that street as a middle filter**. That's a direct measurement
of cut-through behaviour — no inference needed.

This module summarises that triple-product table along two axes:

* :func:`per_street_attribution` — one row per **middle street**:
  total volume routed through it, share that is bridge-bound, share with
  high circuity (3+), dominant origin/destination, dominant trip-length
  bucket. This is the table the recommendation engine consumes.
* :func:`top_od_bypass_pairs` — one row per **origin × destination pair**:
  total Leonia-routed volume, list of middle streets touched, share of
  the pair's trips that route through Leonia. Useful for "which highway
  closure is causing this?" diagnoses.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BRIDGE_KEYWORDS = ("george washington bridge", "gwb")


def _is_bridge_label(s: str) -> bool:
    if not isinstance(s, str):
        return False
    low = s.lower()
    return any(kw in low for kw in _BRIDGE_KEYWORDS)


def _safe_mode(s: pd.Series) -> str | None:
    s = s.dropna()
    if s.empty:
        return None
    return s.mode().iat[0]


# ---------------------------------------------------------------------------
# Per-street attribution
# ---------------------------------------------------------------------------


def per_street_attribution(
    omd_df: pd.DataFrame,
    omd_trips_df: pd.DataFrame | None = None,
    *,
    day_type_code: int = 0,
    day_part_code: int = 0,
) -> pd.DataFrame:
    """One row per Leonia middle-filter street.

    Defaults to All-Days × All-Day so the table reflects the full year of
    measurements. Pass ``day_type_code=1, day_part_code=2`` for Monday
    Peak-AM, etc.
    """
    if omd_df is None or omd_df.empty:
        return pd.DataFrame(columns=[
            "middle_zone", "middle_label", "middle_osm_way_id",
            "total_omd_vph", "n_od_pairs",
            "bridge_share", "top_origin_label", "top_destination_label",
            "top_od_pair_volume", "high_circuity_share",
            "avg_trip_length_mi", "avg_trip_speed_mph",
        ])

    d = omd_df[
        (omd_df["day_type_code"] == day_type_code)
        & (omd_df["day_part_code"] == day_part_code)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    d["is_bridge_dest"] = d["destination_label"].apply(_is_bridge_label)

    # Aggregate per middle street.
    grouped = d.groupby(
        ["middle_zone", "middle_label", "middle_osm_way_id"], dropna=False,
    )
    rows: list[dict] = []
    for keys, grp in grouped:
        middle_zone, middle_label, middle_osm = keys
        total = float(grp["omd_volume"].sum())
        bridge_vol = float(grp.loc[grp["is_bridge_dest"], "omd_volume"].sum())
        top_pair = grp.sort_values("omd_volume", ascending=False).iloc[0]
        rows.append({
            "middle_zone": middle_zone,
            "middle_label": middle_label,
            "middle_osm_way_id": middle_osm,
            "total_omd_vph": total,
            "n_od_pairs": int((grp["omd_volume"] > 0).sum()),
            "bridge_share": bridge_vol / total if total > 0 else float("nan"),
            "top_origin_label": top_pair["origin_label"],
            "top_destination_label": top_pair["destination_label"],
            "top_od_pair_volume": float(top_pair["omd_volume"]),
        })
    out = pd.DataFrame(rows)

    # Optional enrichment from the trip-distribution table.
    if omd_trips_df is not None and not omd_trips_df.empty:
        t = omd_trips_df[
            (omd_trips_df["day_type_code"] == day_type_code)
            & (omd_trips_df["day_part_code"] == day_part_code)
        ].copy()
        if not t.empty:
            # Weighted averages over triples (weight = omd_volume).
            def _weighted_mean(series, weights):
                w = weights.fillna(0)
                v = series
                mask = v.notna() & (w > 0)
                if not mask.any():
                    return float("nan")
                return float((v[mask] * w[mask]).sum() / w[mask].sum())

            agg = (
                t.groupby(["middle_zone", "middle_osm_way_id"], dropna=False)
                 .apply(lambda g: pd.Series({
                     "high_circuity_share": _weighted_mean(
                         g["share_circuity_ge_3"], g["omd_volume"],
                     ),
                     "avg_trip_length_mi": _weighted_mean(
                         g["avg_trip_length_mi"], g["omd_volume"],
                     ),
                     "avg_trip_speed_mph": _weighted_mean(
                         g["avg_trip_speed_mph"], g["omd_volume"],
                     ),
                     "long_trip_share": _weighted_mean(
                         g["share_trip_ge_5mi"], g["omd_volume"],
                     ),
                     "speeding_share": _weighted_mean(
                         g["share_speed_ge_30"], g["omd_volume"],
                     ),
                 }), include_groups=False)
                 .reset_index()
            )
            out = out.merge(agg, on=["middle_zone", "middle_osm_way_id"],
                              how="left")

    out = out.sort_values("total_omd_vph", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


# ---------------------------------------------------------------------------
# Top OD pairs
# ---------------------------------------------------------------------------


def top_od_bypass_pairs(
    omd_df: pd.DataFrame,
    *,
    day_type_code: int = 0,
    day_part_code: int = 0,
    min_volume: float = 5.0,
) -> pd.DataFrame:
    """One row per origin × destination pair routed through Leonia.

    Each row aggregates over middle-filter streets, reporting the total
    Leonia-routed volume and which streets carry it.
    """
    if omd_df is None or omd_df.empty:
        return pd.DataFrame()

    d = omd_df[
        (omd_df["day_type_code"] == day_type_code)
        & (omd_df["day_part_code"] == day_part_code)
        & (omd_df["omd_volume"] >= min_volume)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    grouped = d.groupby([
        "origin_zone", "origin_label",
        "destination_zone", "destination_label",
    ], dropna=False)

    rows: list[dict] = []
    for keys, grp in grouped:
        o_zone, o_label, dst_zone, dst_label = keys
        rows.append({
            "origin_zone": o_zone,
            "origin_label": o_label,
            "destination_zone": dst_zone,
            "destination_label": dst_label,
            "total_routed_vph": float(grp["omd_volume"].sum()),
            "n_middle_streets": int(len(grp)),
            "top_middle_label": grp.sort_values("omd_volume", ascending=False)
                                      .iloc[0]["middle_label"],
            "top_middle_vph": float(grp["omd_volume"].max()),
            "avg_travel_time_sec": float(grp["avg_travel_time_sec"].mean()),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("total_routed_vph", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


__all__ = [
    "per_street_attribution",
    "top_od_bypass_pairs",
]
