"""Equity and demographic profile of GWB-bound travelers.

Pass A.4. Builds on the wide attribute frame from
:func:`leonia_traffic.data.bridge_od_loader.load_bridge_attributes`. All
demographic columns are retained per the user's choice; the narrative is
rendered alongside the operational metrics rather than buried.

Two layers of output:

* :func:`gateway_user_profile` — for a single origin gate, a tidy
  one-column-per-attribute summary (volume-weighted share across the
  requested time window). Output goes into the report verbatim.
* :func:`equity_exposure_index` — for each candidate mitigation
  corridor, the demographic mix of *travelers using that corridor* is
  combined with operational volume to produce a single equity-exposure
  table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leonia_traffic.data.bridge_od_loader import WEEKDAY_CODES

logger = logging.getLogger(__name__)


# Column groups — readable labels mapped to the prefixed columns produced
# by ``load_bridge_attributes``.
EQUITY_COLUMNS: dict[str, dict[str, str]] = {
    "race": {
        "white": "equity::White",
        "black": "equity::Black",
        "american_indian": "equity::American Indian",
        "asian": "equity::Asian",
        "pacific_islander": "equity::Pacific Islander",
        "other_race": "equity::Other Race",
        "multiple_races": "equity::Multiple Races",
    },
    "ethnicity": {
        "hispanic": "equity::Hispanic",
        "non_hispanic": "equity::Non-Hispanic",
    },
    "nativity": {
        "foreign_born": "equity::Foreign Born",
        "non_foreign_born": "equity::Non-foreign Born",
    },
    "language": {
        "english_less_than_very_well": "equity::Speak English less than 'very well'",
    },
    "disability": {
        "with_disability": "equity::With a disability",
        "without_disability": "equity::Without a disability",
    },
}

HOUSEHOLD_COLUMNS: dict[str, dict[str, str]] = {
    "children": {
        "with_kids": "household::With Kids",
        "with_no_kids": "household::With No Kids",
        "with_kids_under_6": "household::With Kids under 6 years",
        "with_kids_6_17": "household::With Kids between 6-17 years",
    },
    "tenure": {
        "owner_occupied": "household::Owner occupied",
        "renter_occupied": "household::Renter occupied",
    },
    "vehicles": {
        "no_vehicle": "household::No vehicle available",
        "one_vehicle": "household::1 vehicle available",
        "two_vehicles": "household::2 vehicles available",
        "three_plus_vehicles": "household::3 or more vehicles available",
    },
    "unit_structure": {
        "single_unit": "household::1 Unit Structure",
        "duplex": "household::2 Unit Structure",
        "small_multi": "household::3-4 Unit Structure",
        "medium_multi": "household::5-9 Unit Structure",
        "large_multi_10_19": "household::10-19 Unit Structure",
        "large_multi_20_49": "household::20-49 Unit Structure",
        "fifty_plus_unit": "household::50+ Unit Structure",
        "mobile_other": "household::Mobile homes, RV, boat, van, other",
    },
}

INCOME_COLUMNS: dict[str, str] = {
    "lt_10k": "income::Income Less than 10K",
    "10_15k": "income::Income 10K to 15K",
    "15_20k": "income::Income 15K to 20K",
    "20_25k": "income::Income 20K to 25K",
    "25_30k": "income::Income 25K to 30K",
    "30_35k": "income::Income 30K to 35K",
    "35_40k": "income::Income 35K to 40K",
    "40_45k": "income::Income 40K to 45K",
    "45_50k": "income::Income 45K to 50K",
    "50_60k": "income::Income 50K to 60K",
    "60_75k": "income::Income 60K to 75K",
    "75_100k": "income::Income 75K to 100K",
    "100_125k": "income::Income 100K to 125K",
    "125_150k": "income::Income 125K to 150K",
    "150_200k": "income::Income 150K to 200K",
    "200k_plus": "income::Income More than 200K",
}

EDUCATION_COLUMNS: dict[str, str] = {
    "less_than_9th": "income::Less than 9th grade",
    "some_hs": "income::9th to 12th grade, no diploma",
    "hs_grad": "income::High school graduate",
    "some_college": "income::Some college, no degree",
    "associates": "income::Associate's degree",
    "bachelors": "income::Bachelor's degree",
    "grad_professional": "income::Graduate or professional degree",
}

EMPLOYMENT_INDUSTRY_COLUMNS: dict[str, str] = {
    "agriculture_mining": "employment::Agriculture, forestry, fishing, hunting, mining",
    "construction": "employment::Construction",
    "manufacturing": "employment::Manufacturing",
    "wholesale_trade": "employment::Wholesale trade",
    "retail_trade": "employment::Retail trade",
    "transportation_warehousing": "employment::Transportation, warehousing, utilities",
    "information": "employment::Information",
    "finance_insurance_real_estate": "employment::Finance, insurance, real estate rental and leasing",
    "professional_services": "employment::Professional, scientific, management, etc. services",
    "education_health_social": "employment::Educational services, health care, social assistance",
    "arts_entertainment": "employment::Arts, entertainment, recreation, etc. services",
    "other_services": "employment::Other services (except public administration)",
    "public_administration": "employment::Public administration",
    "military_industry": "employment::Military (Employment Industry)",
    "not_employed_industry": "employment::Not employed (Employment Industry)",
}

EMPLOYMENT_CLASS_COLUMNS: dict[str, str] = {
    "private_wage": "employment::Private wage and salary workers",
    "government_workers": "employment::Government workers",
    "self_employed": "employment::Self-employed workers",
    "unpaid_family": "employment::Unpaid family workers",
    "military_class": "employment::Military (Employment Class)",
    "not_employed_class": "employment::Not employed (Employment Class)",
}


# Threshold definitions used by :func:`equity_exposure_index` and the
# recommendation engine.
EQUITY_THRESHOLDS = {
    "foreign_born_high": 0.40,
    "english_limited_high": 0.20,
    "low_income_high": 0.30,    # share with HH income < $50K
    "no_vehicle_high": 0.15,
    "renter_occupied_high": 0.50,
}


# ---------------------------------------------------------------------------
# Weighted-share helpers
# ---------------------------------------------------------------------------


def _volume_weighted_share(sub: pd.DataFrame, value_col: str, weight_col: str = "od_volume") -> float:
    """Volume-weighted mean of ``value_col`` over rows in ``sub``."""
    if sub.empty or value_col not in sub.columns:
        return np.nan
    weights = sub[weight_col].fillna(0).to_numpy()
    values = sub[value_col].fillna(0).to_numpy()
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float((values * weights).sum() / total)


def _expand_section(
    sub: pd.DataFrame,
    column_map: dict[str, str],
) -> dict[str, float]:
    """Apply ``_volume_weighted_share`` to every column in ``column_map``."""
    return {label: _volume_weighted_share(sub, src) for label, src in column_map.items()}


# ---------------------------------------------------------------------------
# Per-gateway profile
# ---------------------------------------------------------------------------


def gateway_user_profile(
    attr_df: pd.DataFrame,
    origin_zone: str,
    *,
    weekday_only: bool = True,
    day_part_code: int | None = None,
) -> dict[str, dict[str, float]]:
    """Demographic profile of travelers using one origin gate.

    Returns a nested dict whose top-level keys mirror the column groups
    in this module (``race``, ``ethnicity``, ``nativity``, ``language``,
    ``disability``, ``household_children``, ``household_tenure``,
    ``household_vehicles``, ``household_unit_structure``, ``income``,
    ``education``, ``employment_industry``, ``employment_class``), with
    inner-dict values being volume-weighted shares.

    Parameters
    ----------
    attr_df
        Wide-frame from ``load_bridge_attributes``.
    origin_zone
        Origin zone string (e.g. ``"Fort Lee Road / 590576"``).
    weekday_only
        Restrict to Mon–Fri rows.
    day_part_code
        Optional specific day-part filter (e.g. ``2`` for Peak AM). If
        ``None``, the All-Day rows (code 0) are used.
    """
    sub = attr_df[attr_df["origin_zone"] == origin_zone]
    if weekday_only:
        sub = sub[sub["day_type_code"].isin(WEEKDAY_CODES)]
    if day_part_code is not None:
        sub = sub[sub["day_part_code"] == day_part_code]
    else:
        sub = sub[sub["day_part_code"] == 0]

    profile: dict[str, dict[str, float]] = {}
    for section, cols in EQUITY_COLUMNS.items():
        profile[section] = _expand_section(sub, cols)
    for section, cols in HOUSEHOLD_COLUMNS.items():
        profile[f"household_{section}"] = _expand_section(sub, cols)
    profile["income"] = _expand_section(sub, INCOME_COLUMNS)
    profile["education"] = _expand_section(sub, EDUCATION_COLUMNS)
    profile["employment_industry"] = _expand_section(sub, EMPLOYMENT_INDUSTRY_COLUMNS)
    profile["employment_class"] = _expand_section(sub, EMPLOYMENT_CLASS_COLUMNS)

    # Convenience aggregates.
    income = profile["income"]
    low_income_brackets = [
        "lt_10k", "10_15k", "15_20k", "20_25k", "25_30k",
        "30_35k", "35_40k", "40_45k", "45_50k",
    ]
    profile["aggregates"] = {
        "low_income_under_50k": sum(income.get(b, 0) or 0 for b in low_income_brackets),
        "foreign_born": profile["nativity"].get("foreign_born", np.nan),
        "english_limited": profile["language"].get("english_less_than_very_well", np.nan),
        "renter_occupied": profile["household_tenure"].get("renter_occupied", np.nan),
        "no_vehicle": profile["household_vehicles"].get("no_vehicle", np.nan),
    }
    return profile


def gateway_profiles_table(
    attr_df: pd.DataFrame,
    *,
    weekday_only: bool = True,
    day_part_code: int | None = None,
) -> pd.DataFrame:
    """Run :func:`gateway_user_profile` for every origin gate.

    Returns one row per gate × section.attribute, suitable for direct
    rendering. Columns:
    ``origin_zone``, ``origin_label``, ``origin_osm_way_id``, ``section``,
    ``attribute``, ``share`` (0..1).
    """
    rows: list[dict] = []
    origins = attr_df[["origin_zone", "origin_label", "origin_osm_way_id"]].drop_duplicates()
    for _, og in origins.iterrows():
        profile = gateway_user_profile(
            attr_df, og["origin_zone"],
            weekday_only=weekday_only, day_part_code=day_part_code,
        )
        for section, attrs in profile.items():
            for attribute, share in attrs.items():
                rows.append({
                    "origin_zone": og["origin_zone"],
                    "origin_label": og["origin_label"],
                    "origin_osm_way_id": og["origin_osm_way_id"],
                    "section": section,
                    "attribute": attribute,
                    "share": share,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Equity exposure index
# ---------------------------------------------------------------------------


def equity_exposure_index(
    attr_df: pd.DataFrame,
    *,
    candidate_origins: list[str] | None = None,
    day_part_code: int = 2,
) -> pd.DataFrame:
    """Per-gate exposure flags for the recommendation engine.

    For each origin gate, computes weekday Peak-AM volume + a handful of
    key equity flags. The result feeds the recommendation rule about
    "if more than 40% foreign-born ... require explicit equity
    analysis".

    Returns columns:
    ``origin_zone``, ``origin_label``, ``origin_osm_way_id``,
    ``weekday_peak_volume``, ``foreign_born_share``,
    ``english_limited_share``, ``low_income_under_50k_share``,
    ``no_vehicle_share``, ``renter_occupied_share``,
    ``foreign_born_flag``, ``english_limited_flag``,
    ``low_income_flag``, ``no_vehicle_flag``,
    ``renter_occupied_flag``, ``any_equity_flag``.
    """
    origins = attr_df["origin_zone"].dropna().unique()
    if candidate_origins is not None:
        origins = [o for o in origins if o in candidate_origins]

    rows: list[dict] = []
    for og in origins:
        sub_all = attr_df[(attr_df["origin_zone"] == og)
                          & (attr_df["day_type_code"].isin(WEEKDAY_CODES))]
        sub_peak = sub_all[sub_all["day_part_code"] == day_part_code]

        if sub_all.empty:
            continue

        first = sub_all.iloc[0]
        profile = gateway_user_profile(attr_df, og)
        agg = profile["aggregates"]

        fb = agg.get("foreign_born") or 0
        el = agg.get("english_limited") or 0
        li = agg.get("low_income_under_50k") or 0
        nv = agg.get("no_vehicle") or 0
        ro = agg.get("renter_occupied") or 0

        rec = {
            "origin_zone": og,
            "origin_label": first["origin_label"],
            "origin_osm_way_id": first["origin_osm_way_id"],
            "weekday_peak_volume": float(sub_peak["od_volume"].fillna(0).sum()),
            "foreign_born_share": fb,
            "english_limited_share": el,
            "low_income_under_50k_share": li,
            "no_vehicle_share": nv,
            "renter_occupied_share": ro,
            "foreign_born_flag": fb >= EQUITY_THRESHOLDS["foreign_born_high"],
            "english_limited_flag": el >= EQUITY_THRESHOLDS["english_limited_high"],
            "low_income_flag": li >= EQUITY_THRESHOLDS["low_income_high"],
            "no_vehicle_flag": nv >= EQUITY_THRESHOLDS["no_vehicle_high"],
            "renter_occupied_flag": ro >= EQUITY_THRESHOLDS["renter_occupied_high"],
        }
        rec["any_equity_flag"] = any(
            rec[k] for k in (
                "foreign_born_flag", "english_limited_flag", "low_income_flag",
                "no_vehicle_flag", "renter_occupied_flag",
            )
        )
        rows.append(rec)

    return pd.DataFrame(rows).sort_values("weekday_peak_volume", ascending=False).reset_index(drop=True)


__all__ = [
    "EQUITY_COLUMNS",
    "EQUITY_THRESHOLDS",
    "EDUCATION_COLUMNS",
    "EMPLOYMENT_CLASS_COLUMNS",
    "EMPLOYMENT_INDUSTRY_COLUMNS",
    "HOUSEHOLD_COLUMNS",
    "INCOME_COLUMNS",
    "equity_exposure_index",
    "gateway_profiles_table",
    "gateway_user_profile",
]
