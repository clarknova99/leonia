"""Visitor-demographic analytics for the Leonia-streets ZA export.

ZIP-code-only — no Census ACS join. The user's choice for Pass C is
to keep the analysis self-contained with a small static
ZIP-to-municipality lookup. If a richer demographic story is needed
later, ``leonia_traffic/analysis/equity.py`` already does block-group
joins for the Bridge OD export and can be extended.
"""

from __future__ import annotations

import logging

import pandas as pd

from leonia_traffic.analysis import cutthrough_streets as cs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static ZIP-to-municipality lookup
# ---------------------------------------------------------------------------

# ZIPs cover the top ~40 ZIPs we actually see in the home_zip_codes_top
# CSV. Anything else falls into the "Other (NJ)" or "Other (out-of-state)"
# bucket. Values are short labels that read well on a stacked bar chart.

LEONIA_ZIP = "07605"

NJ_BERGEN_ZIPS = {
    "07605": "Leonia",
    "07024": "Fort Lee",
    "07650": "Palisades Park",
    "07631": "Englewood",
    "07632": "Englewood Cliffs",
    "07010": "Cliffside Park",
    "07020": "Edgewater",
    "07666": "Teaneck",
    "07601": "Hackensack",
    "07657": "Ridgefield",
    "07660": "Ridgefield Park",
    "07644": "Lodi",
    "07670": "Tenafly",
    "07621": "Bergenfield",
    "07675": "Westwood",
    "07626": "Cresskill",
    "07628": "Dumont",
    "07640": "Harrington Park",
    "07641": "Haworth",
    "07642": "Hillsdale",
    "07643": "Little Ferry",
    "07645": "Montvale",
    "07646": "New Milford",
    "07647": "Norwood",
    "07648": "Norwood",
    "07649": "Paramus",
    "07652": "Paramus",
    "07603": "Bogota",
    "07604": "Hasbrouck Heights",
    "07407": "Elmwood Park",
    "07410": "Fair Lawn",
    "07417": "Franklin Lakes",
    "07423": "Glen Rock",
    "07452": "Glen Rock",
    "07450": "Ridgewood",
    "07451": "Ridgewood",
    "07458": "Saddle River",
    "07481": "Wyckoff",
}

NJ_OTHER_ZIPS = {
    "07026": "Garfield (Passaic)",
    "07047": "North Bergen",
    "07087": "Union City",
    "07093": "West New York",
    "07307": "Jersey City",
    "07442": "Pompton Lakes",
    "07463": "Waldwick",
    "07502": "Paterson",
    "07103": "Newark",
    "07042": "Montclair",
    "07052": "West Orange",
    "07070": "Rutherford",
}

NY_ZIPS = {
    "10312": "Staten Island, NY",
    "10927": "Haverstraw, NY",
    "11222": "Brooklyn, NY",
}


def _zip_str(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # The Home-Work CSV stores zips as ints; left-pad to 5 chars.
    return s.zfill(5)


def municipality_for_zip(zip_code) -> str:
    """Translate a ZIP code into a human label.

    Returns one of:

    * the municipality name from the static table,
    * ``"Other NJ"`` for unmatched NJ ZIPs (heuristic: starts with ``07``),
    * ``"Other NY"`` for unmatched NY ZIPs (starts with ``10`` or ``11``),
    * ``"Other"`` for anything else.
    """
    s = _zip_str(zip_code)
    if s is None:
        return "Unknown"
    if s == LEONIA_ZIP:
        return NJ_BERGEN_ZIPS[s]
    if s in NJ_BERGEN_ZIPS:
        return NJ_BERGEN_ZIPS[s]
    if s in NJ_OTHER_ZIPS:
        return NJ_OTHER_ZIPS[s]
    if s in NY_ZIPS:
        return NY_ZIPS[s]
    if s.startswith(("07", "08")):
        return "Other NJ"
    if s.startswith(("10", "11")):
        return "Other NY"
    return "Other"


def origin_municipality_breakdown(
    home_zips_top_df: pd.DataFrame,
    zone_name: str | None = None,
) -> pd.DataFrame:
    """Sum the top-ZIP shares per Visitor-origin municipality.

    If ``zone_name`` is given the breakdown is per-zone; if it's None
    the breakdown aggregates across all zones (volume-weighted by the
    average daily zone traffic).
    """
    if home_zips_top_df is None or home_zips_top_df.empty:
        return pd.DataFrame()
    df = home_zips_top_df.copy()
    if "filter" in df.columns:
        df = df[df["filter"] == "Visitors"]
    df = df[(df["day_type_code"] == cs.ALL_DAYS_TYPE)
            & (df["day_part_code"] == cs.ALL_DAY_PART)]
    if zone_name is not None:
        df = df[df["zone_name"] == zone_name]
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["municipality"] = df["zip_code"].apply(municipality_for_zip)
        agg = df.groupby("municipality", as_index=False)["pct_home_location"].sum()
        agg = agg.rename(columns={"pct_home_location": "share"})
        return agg.sort_values("share", ascending=False).reset_index(drop=True)

    # Cross-zone: weight by daily zone volume so a 10% share on a 1000-trip
    # street counts for more than a 10% share on a 50-trip street.
    df = df.copy()
    df["municipality"] = df["zip_code"].apply(municipality_for_zip)
    df["weighted_trips"] = df["pct_home_location"] * df["zone_volume"].fillna(0)
    by_muni = df.groupby("municipality", as_index=False).agg(
        weighted_trips=("weighted_trips", "sum"),
        total_visitor_trips=("zone_volume", "sum"),
    )
    total = float(by_muni["weighted_trips"].sum()) or 1.0
    by_muni["share"] = by_muni["weighted_trips"] / total
    return by_muni.sort_values("weighted_trips", ascending=False).reset_index(drop=True)


def state_split(
    tourist_summary_df: pd.DataFrame,
    zone_name: str | None = None,
) -> pd.DataFrame:
    """In-state / out-of-state / local-metro share per zone.

    Returns one row per zone, or one aggregated row if ``zone_name``
    is provided (in which case the single-row frame is returned for
    that zone).
    """
    if tourist_summary_df is None or tourist_summary_df.empty:
        return pd.DataFrame()
    df = tourist_summary_df.copy()
    if "filter" in df.columns:
        df = df[df["filter"] == "Visitors"]
    df = df[(df["day_type_code"] == cs.ALL_DAYS_TYPE)
            & (df["day_part_code"] == cs.ALL_DAY_PART)]
    keep = [
        "zone_name", "street_name", "osm_way_id", "zone_volume",
        "Percent Living in State", "Percent Living out of State",
        "Percent Living in Local Metro Area", "Percent Living in Other Metro Area",
        "Percent Living in Rural Area",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["zone_name"])
    rename = {
        "Percent Living in State": "in_state_share",
        "Percent Living out of State": "out_of_state_share",
        "Percent Living in Local Metro Area": "local_metro_share",
        "Percent Living in Other Metro Area": "other_metro_share",
        "Percent Living in Rural Area": "rural_share",
    }
    df = df.rename(columns=rename)
    if zone_name is not None:
        return df[df["zone_name"] == zone_name].reset_index(drop=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# County FIPS → label lookup (used by destination breakdowns)
# ---------------------------------------------------------------------------

# Census 5-digit county FIPS for everywhere material to Leonia's
# cut-through analysis. Anything not in this table is bucketed into
# "Other NJ", "Other NY", "Other CT/PA/Tri-state", or "Other".
COUNTY_FIPS_LABELS = {
    # New Jersey
    "34003": "Bergen, NJ",
    "34017": "Hudson, NJ",
    "34013": "Essex, NJ (Newark)",
    "34031": "Passaic, NJ",
    "34027": "Morris, NJ",
    "34039": "Union, NJ",
    "34023": "Middlesex, NJ",
    "34025": "Monmouth, NJ",
    "34041": "Warren, NJ",
    "34035": "Somerset, NJ",
    "34037": "Sussex, NJ",
    "34019": "Hunterdon, NJ",
    "34021": "Mercer, NJ",
    "34029": "Ocean, NJ",
    "34033": "Salem, NJ",
    "34005": "Burlington, NJ",
    "34007": "Camden, NJ",
    "34009": "Cape May, NJ",
    "34011": "Cumberland, NJ",
    "34015": "Gloucester, NJ",
    "34001": "Atlantic, NJ",
    # New York City + suburbs
    "36061": "Manhattan, NY",
    "36005": "Bronx, NY",
    "36047": "Brooklyn, NY",
    "36081": "Queens, NY",
    "36085": "Staten Island, NY",
    "36119": "Westchester, NY",
    "36087": "Rockland, NY",
    "36059": "Nassau, NY",
    "36103": "Suffolk, NY",
    "36071": "Orange, NY",
    "36079": "Putnam, NY",
    "36027": "Dutchess, NY",
    # Connecticut
    "09001": "Fairfield, CT",
    "09005": "Litchfield, CT",
    "09009": "New Haven, CT",
    # Pennsylvania
    "42103": "Pike, PA",
    "42089": "Monroe, PA",
}


def _state_fips_to_label(state_fips: str) -> str:
    if state_fips == "34":
        return "Other NJ"
    if state_fips == "36":
        return "Other NY"
    if state_fips == "09":
        return "Other CT"
    if state_fips == "42":
        return "Other PA"
    return "Other"


def county_label(county_fips) -> str:
    """Translate a 5-char county FIPS into a human label.

    Returns one of the entries in :data:`COUNTY_FIPS_LABELS`, falling
    back to ``"Other NJ"``/``"Other NY"``/etc. based on the state
    prefix, or ``"Unknown"`` if the value is missing.
    """
    if county_fips is None or (isinstance(county_fips, float) and pd.isna(county_fips)):
        return "Unknown"
    s = str(county_fips).strip().strip("'").zfill(5)
    if s in COUNTY_FIPS_LABELS:
        return COUNTY_FIPS_LABELS[s]
    return _state_fips_to_label(s[:2])


def work_destination_breakdown(
    work_bg_df: pd.DataFrame,
    zone_name: str | None = None,
    *,
    top_n: int | None = 15,
) -> pd.DataFrame:
    """County-level work-destination breakdown for Visitor pass-through.

    StreetLight only exposes a workplace block-group cross-tab — there
    is no direct "trip destination" CSV. For commuting cut-through
    that's still the right answer: the work block group is where the
    driver is headed in the AM peak. Returns one row per labeled
    county with a ``share`` column.

    When ``zone_name`` is None the breakdown aggregates across all
    zones, weighted by each zone's average daily Visitor volume so a
    10% share on a 1000-trip street counts for more than 10% on a
    50-trip street.
    """
    if work_bg_df is None or work_bg_df.empty:
        return pd.DataFrame()
    df = work_bg_df.copy()
    if "filter" in df.columns:
        df = df[df["filter"] == "Visitors"]
    df = df[(df["day_type_code"] == 0) & (df["day_part_code"] == 0)]
    if df.empty:
        return pd.DataFrame()

    df["county_label"] = df["county_fips"].apply(county_label)

    if zone_name is not None:
        sub = df[df["zone_name"] == zone_name]
        if sub.empty:
            return pd.DataFrame()
        agg = sub.groupby("county_label", as_index=False)["pct_work_location"].sum()
        agg = agg.rename(columns={"pct_work_location": "share"})
        agg = agg.sort_values("share", ascending=False).reset_index(drop=True)
        return agg.head(top_n) if top_n else agg

    # Cross-zone: weight by daily Visitor volume.
    df["weighted_trips"] = df["pct_work_location"] * df["zone_volume"].fillna(0)
    by_cty = df.groupby("county_label", as_index=False).agg(
        weighted_trips=("weighted_trips", "sum"),
    )
    total = float(by_cty["weighted_trips"].sum()) or 1.0
    by_cty["share"] = by_cty["weighted_trips"] / total
    by_cty = by_cty.sort_values("weighted_trips", ascending=False).reset_index(drop=True)
    return by_cty.head(top_n) if top_n else by_cty


def top_work_tracts(
    work_bg_df: pd.DataFrame,
    zone_name: str,
    *,
    n: int = 10,
) -> pd.DataFrame:
    """Top Census tracts where Visitor pass-through on ``zone_name`` works.

    Returns one row per tract, ranked by ``pct_work_location``,
    with the parent county label attached. Tracts are aggregated from
    block groups (StreetLight reports at the block-group level; tracts
    are a more legible spatial unit).
    """
    if work_bg_df is None or work_bg_df.empty:
        return pd.DataFrame()
    df = work_bg_df.copy()
    if "filter" in df.columns:
        df = df[df["filter"] == "Visitors"]
    df = df[(df["day_type_code"] == 0)
            & (df["day_part_code"] == 0)
            & (df["zone_name"] == zone_name)]
    if df.empty:
        return pd.DataFrame()
    df["county_label"] = df["county_fips"].apply(county_label)
    agg = df.groupby(["tract", "county_fips", "county_label"], as_index=False).agg(
        share=("pct_work_location", "sum"),
    )
    return agg.sort_values("share", ascending=False).head(n).reset_index(drop=True)


__all__ = [
    "LEONIA_ZIP",
    "NJ_BERGEN_ZIPS",
    "NJ_OTHER_ZIPS",
    "NY_ZIPS",
    "COUNTY_FIPS_LABELS",
    "county_label",
    "municipality_for_zip",
    "origin_municipality_breakdown",
    "state_split",
    "top_work_tracts",
    "work_destination_breakdown",
]
