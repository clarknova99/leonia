"""Rule-based recommendation engine for the bridge OD evidence report.

Pass A.6. Consumes the tables produced by ``od_cutthrough``,
``congestion``, and ``equity`` and emits a ranked list of
``Recommendation`` records. Each rule is small, named, and individually
defensible — the goal is a short list a town engineer or council member
could walk through line-by-line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from leonia_traffic.analysis.equity import EQUITY_THRESHOLDS
from leonia_traffic.analysis.jurisdiction import is_county_state_arterial

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    """A single rule-engine output, ready for the report."""

    rank: int
    severity: str            # "high", "medium", "info"
    target: str              # name of the gate / corridor / metric
    rule: str                # short rule identifier
    rationale: str           # one-sentence justification
    metrics: dict = field(default_factory=dict)

    def to_markdown_row(self) -> str:
        bits = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
        return f"| {self.rank} | {self.severity.upper()} | {self.target} | {self.rule} | {self.rationale} | {bits} |"


# ---------------------------------------------------------------------------
# Thresholds — surfaced here so they can be overridden by a config later.
# ---------------------------------------------------------------------------


THRESHOLDS = {
    "primary_mitigation_peak_am_min": 300.0,
    "primary_mitigation_ratio_min": 5.0,
    "secondary_mitigation_peak_am_min": 100.0,
    "failing_corridor_tti_min": 2.0,
    "failing_corridor_buffer_min": 1.5,
    # Require non-trivial public time-cost in addition to a high TTI to
    # avoid surfacing weekend / low-volume noise on residential blocks.
    "failing_corridor_min_weekday_vhd": 10.0,
    "high_delay_vhd_min": 100.0,
    "commuter_share_min": 0.30,
    "cutthrough_circuity_index_min": 0.40,
    # Pass C: per-residential-street thresholds.
    "residential_cutthrough_index_min": 0.45,
    "residential_min_weekday_volume": 250.0,
    "residential_non_local_min": 0.50,
    "residential_speeding_min": 0.40,
    # Pass D: trend (streetscanner_trend) and OMD attribution thresholds.
    "trend_worsening_yoy_pct_min": 25.0,
    "trend_min_recent_volume": 50.0,
    "trend_persistent_above_baseline_min": 0.60,
    "omd_bridge_share_min": 0.50,
    "omd_high_circuity_share_min": 0.30,
    "omd_min_total_vph": 100.0,
    # Arterial-channeling thresholds (Broad / Grand / Fort Lee Rd).
    "arterial_diversion_local_min_vph": 75.0,
    "arterial_diversion_bridge_share_min": 0.30,
}


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _primary_mitigation_candidates(
    peak_imbalance_df: pd.DataFrame,
) -> list[Recommendation]:
    """Gates with weekday/weekend ratio > 5 and Peak-AM volume > 300."""
    out: list[Recommendation] = []
    if peak_imbalance_df.empty:
        return out

    qualifying = peak_imbalance_df[
        (peak_imbalance_df["peak_am_weekday_avg"] >= THRESHOLDS["primary_mitigation_peak_am_min"])
        & (peak_imbalance_df["peak_am_to_weekend_ratio"] >= THRESHOLDS["primary_mitigation_ratio_min"])
    ]

    for _, row in qualifying.iterrows():
        out.append(Recommendation(
            rank=0,
            severity="high",
            target=str(row["origin_label"]),
            rule="primary_mitigation_candidate",
            rationale=(
                f"Peak-AM weekday volume of {row['peak_am_weekday_avg']:.0f} trips/day "
                f"and weekday/weekend ratio of {row['peak_am_to_weekend_ratio']:.1f}× "
                "indicate concentrated commuter cut-through; this is the primary "
                "candidate for traffic-calming or routing changes."
            ),
            metrics={
                "peak_am_weekday": int(row["peak_am_weekday_avg"]),
                "weekend_peak_am": int(row["weekend_peak_am_avg"] or 0),
                "ratio": round(row["peak_am_to_weekend_ratio"], 1),
            },
        ))
    return out


def _secondary_mitigation_candidates(
    peak_imbalance_df: pd.DataFrame,
) -> list[Recommendation]:
    """Gates with material but not headline-level cut-through."""
    out: list[Recommendation] = []
    if peak_imbalance_df.empty:
        return out

    lo = THRESHOLDS["secondary_mitigation_peak_am_min"]
    hi = THRESHOLDS["primary_mitigation_peak_am_min"]
    qualifying = peak_imbalance_df[
        (peak_imbalance_df["peak_am_weekday_avg"] >= lo)
        & (peak_imbalance_df["peak_am_weekday_avg"] < hi)
        & (peak_imbalance_df["peak_am_to_weekend_ratio"] >= 2.0)
    ]
    for _, row in qualifying.iterrows():
        out.append(Recommendation(
            rank=0,
            severity="medium",
            target=str(row["origin_label"]),
            rule="secondary_mitigation_candidate",
            rationale=(
                f"Moderate Peak-AM volume ({row['peak_am_weekday_avg']:.0f}) with "
                f"weekday/weekend ratio {row['peak_am_to_weekend_ratio']:.1f}× — "
                "worth monitoring but not the principal cut-through corridor."
            ),
            metrics={
                "peak_am_weekday": int(row["peak_am_weekday_avg"]),
                "ratio": round(row["peak_am_to_weekend_ratio"], 1),
            },
        ))
    return out


def _failing_corridors(
    summary_df: pd.DataFrame,
) -> list[Recommendation]:
    """Already-failing corridors that should NOT absorb diverted traffic.

    Combines the delay-hotspot table (high VHD) with the per-link
    summary (high worst-hour TTI and Buffer Index) to identify links
    that are already operating beyond reasonable service.
    """
    out: list[Recommendation] = []
    if summary_df.empty:
        return out

    failing = summary_df[
        (summary_df["worst_tti"] >= THRESHOLDS["failing_corridor_tti_min"])
        & (summary_df["worst_buffer"].fillna(0) >= THRESHOLDS["failing_corridor_buffer_min"])
        & (summary_df["total_weekday_vhd"].fillna(0)
           >= THRESHOLDS["failing_corridor_min_weekday_vhd"])
    ]
    # Cap to the worst 10 to keep the recommendation list readable.
    failing = failing.sort_values("total_weekday_vhd", ascending=False).head(10)
    for _, row in failing.iterrows():
        out.append(Recommendation(
            rank=0,
            severity="high",
            target=str(row["osm_name"] or row["zone_name"]),
            rule="failing_corridor_exclude_from_diversion",
            rationale=(
                f"Worst-hour TTI {row['worst_tti']:.2f} and Buffer Index "
                f"{row['worst_buffer']:.2f} show this corridor is already failing; "
                "any mitigation that diverts traffic onto it would worsen conditions."
            ),
            metrics={
                "tti": round(row["worst_tti"], 2),
                "buffer": round(row["worst_buffer"], 2),
                "vhd": round(row["total_weekday_vhd"] or 0, 1),
            },
        ))
    return out


def _high_delay_warnings(delay_df: pd.DataFrame) -> list[Recommendation]:
    """Top-VHD corridors get an informational call-out."""
    out: list[Recommendation] = []
    if delay_df.empty:
        return out
    top = delay_df.head(5)
    for _, row in top.iterrows():
        if (row.get("weekday_vhd_total") or 0) < THRESHOLDS["high_delay_vhd_min"]:
            continue
        out.append(Recommendation(
            rank=0,
            severity="info",
            target=str(row["osm_name"] or row["zone_name"]),
            rule="high_delay_corridor",
            rationale=(
                f"{row['weekday_vhd_total']:.0f} vehicle-hours of weekday delay; "
                f"worst-hour TTI {row['worst_hour_tti']:.2f} at "
                f"{row['worst_hour_label']}. Total time-cost-to-public is "
                "non-trivial here."
            ),
            metrics={
                "weekday_vhd": int(row["weekday_vhd_total"]),
                "worst_tti": round(row["worst_hour_tti"] or 0, 2),
            },
        ))
    return out


def _equity_callouts(exposure_df: pd.DataFrame) -> list[Recommendation]:
    """Gates that trip equity thresholds need an explicit analysis."""
    out: list[Recommendation] = []
    if exposure_df.empty:
        return out

    for _, row in exposure_df.iterrows():
        if not row["any_equity_flag"]:
            continue
        flagged = [k.replace("_flag", "") for k in (
            "foreign_born_flag", "english_limited_flag",
            "low_income_flag", "no_vehicle_flag", "renter_occupied_flag",
        ) if row[k]]
        if not flagged:
            continue
        out.append(Recommendation(
            rank=0,
            severity="medium",
            target=str(row["origin_label"]),
            rule="explicit_equity_analysis_required",
            rationale=(
                "Travelers using this gate include sizable shares of "
                f"{', '.join(flagged)}; any mitigation here should include "
                "an explicit equity impact analysis before adoption."
            ),
            metrics={
                "fb": round(row["foreign_born_share"], 2),
                "el": round(row["english_limited_share"], 2),
                "lo_inc": round(row["low_income_under_50k_share"], 2),
                "no_veh": round(row["no_vehicle_share"], 2),
                "renter": round(row["renter_occupied_share"], 2),
                "peak_vol": int(row["weekday_peak_volume"]),
            },
        ))
    return out


def _residential_cutthrough_candidates(
    per_street_df: pd.DataFrame,
) -> list[Recommendation]:
    """Pass C: residential streets with strong cut-through evidence.

    A street qualifies when:

    * composite cut-through index >= ``residential_cutthrough_index_min``,
    * weekday all-day Visitor volume >= ``residential_min_weekday_volume``, and
    * non-local home share >= ``residential_non_local_min``.
    """
    out: list[Recommendation] = []
    if per_street_df is None or per_street_df.empty:
        return out
    df = per_street_df
    needed = {"cutthrough_index", "weekday_all_day_volume",
              "non_local_home_share", "street_name", "osm_way_id"}
    if not needed.issubset(set(df.columns)):
        logger.warning(
            "per_street_df missing columns %s; skipping residential rule",
            needed - set(df.columns),
        )
        return out

    qualifying = df[
        (df["cutthrough_index"].fillna(0) >= THRESHOLDS["residential_cutthrough_index_min"])
        & (df["weekday_all_day_volume"].fillna(0) >= THRESHOLDS["residential_min_weekday_volume"])
        & (df["non_local_home_share"].fillna(0) >= THRESHOLDS["residential_non_local_min"])
    ].sort_values("cutthrough_index", ascending=False).head(10)

    for _, row in qualifying.iterrows():
        out.append(Recommendation(
            rank=0,
            severity="high",
            target=str(row["street_name"]),
            rule="residential_cutthrough_candidate",
            rationale=(
                f"Composite cut-through index {row['cutthrough_index']:.2f} "
                f"({int(row['weekday_all_day_volume']):,} weekday Visitor trips/day, "
                f"{row['non_local_home_share'] * 100:.0f}% with home ≥3 mi away). "
                "Residential street with strong direct evidence of regional "
                "pass-through traffic; primary calming candidate."
            ),
            metrics={
                "osm_way": int(row["osm_way_id"]) if pd.notna(row["osm_way_id"]) else None,
                "index": round(float(row["cutthrough_index"]), 2),
                "wd_vol": int(row["weekday_all_day_volume"]),
                "non_local": round(float(row["non_local_home_share"]), 2),
                "wd_to_sat": (
                    round(float(row["weekday_weekend_ratio"]), 2)
                    if "weekday_weekend_ratio" in row and pd.notna(row["weekday_weekend_ratio"])
                    else None
                ),
            },
        ))
    return out


def _residential_speeding_callouts(
    per_street_df: pd.DataFrame,
) -> list[Recommendation]:
    """Pass C: residential streets with high speeding-bin share."""
    out: list[Recommendation] = []
    if per_street_df is None or per_street_df.empty:
        return out
    if "speeding_share" not in per_street_df.columns:
        return out
    df = per_street_df
    qualifying = df[
        (df["speeding_share"].fillna(0) >= THRESHOLDS["residential_speeding_min"])
        & (df["weekday_all_day_volume"].fillna(0) >= THRESHOLDS["residential_min_weekday_volume"])
    ].sort_values("speeding_share", ascending=False).head(10)

    for _, row in qualifying.iterrows():
        out.append(Recommendation(
            rank=0,
            severity="medium",
            target=str(row["street_name"]),
            rule="residential_speeding_callout",
            rationale=(
                f"{row['speeding_share'] * 100:.0f}% of non-resident pass-through "
                "trips fall in speed bins ≥25 mph on a residential block. "
                "Speed-management treatment (signage, calming, enforcement) is "
                "warranted regardless of volume disposition."
            ),
            metrics={
                "osm_way": int(row["osm_way_id"]) if pd.notna(row["osm_way_id"]) else None,
                "speed_share": round(float(row["speeding_share"]), 2),
                "wd_vol": int(row["weekday_all_day_volume"]),
            },
        ))
    return out


def _accelerating_cutthrough_callouts(
    street_trend_df: pd.DataFrame | None,
) -> list[Recommendation]:
    """Pass D: streets where weekday volume is climbing YoY.

    Uses the per-street trend table (Jan 2023 → present). Flags any
    street whose recent 12-month average is ≥``trend_worsening_yoy_pct_min``%
    above the prior 12 months AND has at least ``trend_min_recent_volume``
    recent-window vehicles/day. Severity is "medium" — these are early
    warnings rather than confirmed cut-through.
    """
    out: list[Recommendation] = []
    if street_trend_df is None or street_trend_df.empty:
        return out
    needed = {"zone_name", "osm_name", "yoy_change_pct", "recent_12mo_avg",
              "baseline_12mo_avg", "trend_slope_per_year"}
    if not needed.issubset(set(street_trend_df.columns)):
        logger.warning(
            "street_trend_df missing columns %s; skipping trend rule",
            needed - set(street_trend_df.columns),
        )
        return out
    df = street_trend_df
    qualifying = df[
        (df["yoy_change_pct"].fillna(0) >= THRESHOLDS["trend_worsening_yoy_pct_min"])
        & (df["recent_12mo_avg"].fillna(0) >= THRESHOLDS["trend_min_recent_volume"])
    ].sort_values("yoy_change_pct", ascending=False).head(10)

    for _, row in qualifying.iterrows():
        target = str(row.get("osm_name") or row["zone_name"])
        out.append(Recommendation(
            rank=0,
            severity="medium",
            target=target,
            rule="accelerating_cutthrough",
            rationale=(
                f"Weekday volume up {row['yoy_change_pct']:.0f}% YoY "
                f"({row['baseline_12mo_avg']:.0f} → {row['recent_12mo_avg']:.0f} "
                "vehicles/day on the 12-month rolling average). "
                "Cut-through pressure is increasing on this corridor — "
                "monitor and consider preemptive calming before it joins "
                "the high-severity list."
            ),
            metrics={
                "osm_way": (
                    int(row["osm_way_id"])
                    if "osm_way_id" in row and pd.notna(row["osm_way_id"])
                    else None
                ),
                "yoy_pct": round(float(row["yoy_change_pct"]), 1),
                "recent_avg": int(row["recent_12mo_avg"]),
                "baseline_avg": int(row["baseline_12mo_avg"]),
                "slope_per_year": (
                    round(float(row["trend_slope_per_year"]), 1)
                    if pd.notna(row["trend_slope_per_year"]) else None
                ),
            },
        ))
    return out


def _omd_attributed_cutthrough_callouts(
    cutthrough_attribution_df: pd.DataFrame | None,
) -> list[Recommendation]:
    """Pass D: streets the OMD export directly attributes to bridge cut-through.

    Uses the per-middle-street attribution table built from the
    O-D + Middle-Filter analysis. Flags streets where:

    * ``bridge_share`` ≥ ``omd_bridge_share_min`` (majority of routed
      volume terminates at the GWB), AND
    * ``high_circuity_share`` ≥ ``omd_high_circuity_share_min`` (drivers
      are taking ≥3× the straight-line distance), AND
    * ``total_omd_vph`` ≥ ``omd_min_total_vph`` (signal large enough to
      act on).

    Severity is "high" — this is a direct measurement, not an inference.
    """
    out: list[Recommendation] = []
    if cutthrough_attribution_df is None or cutthrough_attribution_df.empty:
        return out
    needed = {"middle_label", "bridge_share", "total_omd_vph"}
    if not needed.issubset(set(cutthrough_attribution_df.columns)):
        logger.warning(
            "cutthrough_attribution_df missing columns %s; skipping OMD rule",
            needed - set(cutthrough_attribution_df.columns),
        )
        return out

    df = cutthrough_attribution_df
    has_circuity = "high_circuity_share" in df.columns
    mask = (
        (df["bridge_share"].fillna(0) >= THRESHOLDS["omd_bridge_share_min"])
        & (df["total_omd_vph"].fillna(0) >= THRESHOLDS["omd_min_total_vph"])
    )
    if has_circuity:
        mask &= (
            df["high_circuity_share"].fillna(0)
            >= THRESHOLDS["omd_high_circuity_share_min"]
        )
    qualifying = df[mask].sort_values("total_omd_vph", ascending=False).head(10)

    for _, row in qualifying.iterrows():
        target = str(row["middle_label"])
        circuity_note = (
            f", and {row['high_circuity_share'] * 100:.0f}% of those trips "
            "take ≥3× the straight-line distance"
            if has_circuity and pd.notna(row.get("high_circuity_share"))
            else ""
        )
        top_o = row.get("top_origin_label", "")
        top_d = row.get("top_destination_label", "")
        out.append(Recommendation(
            rank=0,
            severity="high",
            target=target,
            rule="omd_confirmed_cutthrough",
            rationale=(
                f"OMD measurement: {row['bridge_share'] * 100:.0f}% of the "
                f"{int(row['total_omd_vph']):,} routed vehicles/day on this "
                f"street terminate at the GWB{circuity_note}. "
                f"Dominant flow: {top_o} → {top_d}. "
                "This is a directly observed cut-through corridor — "
                "calming and routing intervention are warranted."
            ),
            metrics={
                "osm_way": (
                    int(row["middle_osm_way_id"])
                    if "middle_osm_way_id" in row
                    and pd.notna(row["middle_osm_way_id"])
                    else None
                ),
                "bridge_share": round(float(row["bridge_share"]), 2),
                "total_vph": int(row["total_omd_vph"]),
                "n_od_pairs": (
                    int(row["n_od_pairs"])
                    if "n_od_pairs" in row and pd.notna(row["n_od_pairs"])
                    else None
                ),
                "high_circuity_share": (
                    round(float(row["high_circuity_share"]), 2)
                    if has_circuity and pd.notna(row.get("high_circuity_share"))
                    else None
                ),
            },
        ))
    return out


def _circuity_callouts(circuity_df: pd.DataFrame) -> list[Recommendation]:
    """Gates with high circuity (drivers detouring) get a separate signal."""
    out: list[Recommendation] = []
    if circuity_df.empty:
        return out
    qualifying = circuity_df[circuity_df["cutthrough_circuity_index"] >= THRESHOLDS["cutthrough_circuity_index_min"]]
    for _, row in qualifying.iterrows():
        out.append(Recommendation(
            rank=0,
            severity="medium",
            target=str(row["origin_label"]),
            rule="high_circuity_detour_evidence",
            rationale=(
                f"{row['cutthrough_circuity_index']*100:.0f}% of Peak-AM trips have "
                "circuity > 2, meaning drivers travelled materially farther than the "
                "straight-line distance — consistent with arterial-bypass behavior."
            ),
            metrics={
                "circuity_idx": round(row["cutthrough_circuity_index"], 3),
                "trips": int(row.get("trips_in_window", 0)),
            },
        ))
    return out


# ---------------------------------------------------------------------------
# Arterial-channeling rules (Broad / Grand / Fort Lee Rd)
# ---------------------------------------------------------------------------


_ARTERIAL_STRATEGY_RATIONALE = (
    "Borough strategy: channel cross-town and bridge-bound traffic onto "
    "Broad Avenue (CR 1), Grand Avenue (CR 17/49), and Fort Lee Road "
    "(CR 9) — the three county arterials sized to carry this volume — "
    "and protect the residential grid in between. Leonia has no "
    "authority to modify the arterials themselves (NJDOT / Bergen "
    "County jurisdiction), so action is concentrated on local-street "
    "calming, turn restrictions at arterial-to-local junctions, and "
    "signage / wayfinding that biases routing toward the arterials."
)


def _arterial_channeling_strategy() -> list[Recommendation]:
    """Top-level strategic recommendation fired once per report.

    Establishes the jurisdictional framing all other rules are read
    under: county-owned arterials are the desired channel; local
    Leonia-controlled streets are the protectable surface.
    """
    return [Recommendation(
        rank=0,
        severity="high",
        target="Borough-wide network strategy",
        rule="channel_to_county_arterials",
        rationale=_ARTERIAL_STRATEGY_RATIONALE,
        metrics={
            "arterials": "Broad Ave, Grand Ave, Fort Lee Rd",
            "authority": "Bergen County / NJDOT",
        },
    )]


def _local_to_arterial_diversion(
    cutthrough_attribution_df: pd.DataFrame | None,
) -> list[Recommendation]:
    """For each local street carrying bridge-bound cut-through, recommend
    turn-restriction / signage interventions that push that volume to
    the nearest county arterial.

    Fires only for streets **not** named Broad/Grand/Fort Lee — those
    are the channels, not the targets.
    """
    out: list[Recommendation] = []
    if cutthrough_attribution_df is None or cutthrough_attribution_df.empty:
        return out
    needed = {"middle_label", "total_omd_vph", "bridge_share"}
    if not needed.issubset(set(cutthrough_attribution_df.columns)):
        return out

    df = cutthrough_attribution_df.copy()
    df = df[~df["middle_label"].apply(is_county_state_arterial)]
    df = df[
        (df["total_omd_vph"].fillna(0)
         >= THRESHOLDS["arterial_diversion_local_min_vph"])
        & (df["bridge_share"].fillna(0)
           >= THRESHOLDS["arterial_diversion_bridge_share_min"])
    ].sort_values("total_omd_vph", ascending=False).head(10)

    for _, row in df.iterrows():
        target = str(row["middle_label"])
        circ = row.get("high_circuity_share")
        circ_note = (
            f" Circuity ≥3 on {circ * 100:.0f}% of trips — drivers are "
            "actively detouring off the arterials to use this street."
            if pd.notna(circ) else ""
        )
        out.append(Recommendation(
            rank=0,
            severity="high",
            target=target,
            rule="divert_local_to_arterial",
            rationale=(
                f"Local street carries {int(row['total_omd_vph']):,} "
                f"routed vph with {row['bridge_share'] * 100:.0f}% "
                "bridge-bound — most of which could be served by the "
                "parallel arterial (Broad Ave / Grand Ave / Fort Lee "
                f"Rd).{circ_note} Recommended Leonia-controlled "
                "interventions: peak-hour turn restrictions at the "
                "arterial-to-local junction feeding this corridor, "
                "wayfinding / signage steering GWB traffic to the "
                "arterials, and (if persistent) one-way conversion or "
                "diverters to break the through-route."
            ),
            metrics={
                "osm_way": (
                    int(row["middle_osm_way_id"])
                    if "middle_osm_way_id" in row
                    and pd.notna(row["middle_osm_way_id"]) else None
                ),
                "total_vph": int(row["total_omd_vph"]),
                "bridge_share": round(float(row["bridge_share"]), 2),
                "high_circuity_share": (
                    round(float(circ), 2) if pd.notna(circ) else None
                ),
                "top_origin": row.get("top_origin_label"),
            },
        ))
    return out


def _reclassify_arterial_targets(
    recs: list[Recommendation],
) -> list[Recommendation]:
    """Post-process recs: any rule whose target is a county arterial
    (Broad / Grand / Fort Lee Rd) is downgraded to ``info`` and reframed
    as a "monitor / petition county" action, since Leonia has no
    authority to act there directly.

    The *new* ``divert_local_to_arterial`` rule is unaffected — its
    target is always a local street, never an arterial.
    """
    rewritten: list[Recommendation] = []
    for r in recs:
        if r.rule == "divert_local_to_arterial":
            rewritten.append(r)
            continue
        if r.rule == "channel_to_county_arterials":
            rewritten.append(r)
            continue
        if is_county_state_arterial(r.target):
            new_metrics = dict(r.metrics)
            new_metrics["jurisdiction"] = "Bergen County / NJDOT"
            rewritten.append(Recommendation(
                rank=0,
                severity="info",
                target=r.target,
                rule=f"{r.rule}__arterial_monitor",
                rationale=(
                    "Evidence flagged this corridor, but it is a "
                    "county/state arterial — Leonia has no authority "
                    "to modify access, geometry, or signal timing here. "
                    "Action: forward measurements to Bergen County / "
                    "NJDOT and monitor. Original finding: "
                    f"{r.rationale}"
                ),
                metrics=new_metrics,
            ))
        else:
            rewritten.append(r)
    return rewritten


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_recommendations(
    *,
    peak_imbalance_df: pd.DataFrame,
    circuity_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    per_street_df: pd.DataFrame | None = None,
    street_trend_df: pd.DataFrame | None = None,
    cutthrough_attribution_df: pd.DataFrame | None = None,
) -> list[Recommendation]:
    """Run every rule and return the merged, ranked list.

    Severity sort: high > medium > info. Stable within severity by rule
    order in this function.

    Parameters
    ----------
    per_street_df:
        Optional per-residential-street cut-through table produced by
        :mod:`leonia_traffic.analysis.cutthrough_streets` (Pass C). When
        provided, residential-cut-through and residential-speeding rules
        fire in addition to the existing OD-gate and congestion rules.
        The frame is expected to already be filtered to streets under
        Borough of Leonia municipal jurisdiction.

    .. note::
        ``summary_df`` and ``delay_df`` are expected to be **already
        filtered to streets under Borough of Leonia jurisdiction** (use
        :func:`leonia_traffic.analysis.jurisdiction.filter_segments_to_leonia`
        upstream). The OD gates fed into ``peak_imbalance_df``,
        ``circuity_df``, and ``exposure_df`` are by construction Leonia
        perimeter zones, so no extra filter is needed there.
    """
    recs: list[Recommendation] = []
    recs.extend(_arterial_channeling_strategy())
    recs.extend(_local_to_arterial_diversion(cutthrough_attribution_df))
    recs.extend(_primary_mitigation_candidates(peak_imbalance_df))
    recs.extend(_omd_attributed_cutthrough_callouts(cutthrough_attribution_df))
    recs.extend(_residential_cutthrough_candidates(per_street_df))
    recs.extend(_failing_corridors(summary_df))
    recs.extend(_secondary_mitigation_candidates(peak_imbalance_df))
    recs.extend(_circuity_callouts(circuity_df))
    recs.extend(_accelerating_cutthrough_callouts(street_trend_df))
    recs.extend(_residential_speeding_callouts(per_street_df))
    recs.extend(_equity_callouts(exposure_df))
    recs.extend(_high_delay_warnings(delay_df))

    # Reclassify any rec targeting a county/state arterial as info-only
    # (Leonia cannot act there). The strategic + diversion rules above
    # are exempt.
    recs = _reclassify_arterial_targets(recs)

    sev_rank = {"high": 0, "medium": 1, "info": 2}
    recs.sort(key=lambda r: sev_rank.get(r.severity, 99))
    for i, r in enumerate(recs, start=1):
        r.rank = i
    return recs


def recommendations_to_markdown(recs: list[Recommendation]) -> str:
    """Render the recommendations as a markdown table."""
    if not recs:
        return "_No rule-based recommendations triggered._\n"
    lines = [
        "| # | Severity | Target | Rule | Rationale | Metrics |",
        "|---|----------|--------|------|-----------|---------|",
    ]
    for r in recs:
        lines.append(r.to_markdown_row())
    return "\n".join(lines) + "\n"


__all__ = [
    "Recommendation",
    "THRESHOLDS",
    "generate_recommendations",
    "recommendations_to_markdown",
]
