"""Parse NJDOT raw crash files (the 2017+ "Crash Table" layout).

NJDOT publishes one fixed-width text file per (year × county × table)
at ``https://www.nj.gov/transportation/refdata/accident/<year>/<county><year><Table>.zip``.
Records are 470 characters wide (2017–present); each field has a
documented byte offset and the file is comma-padded so a naive CSV
parser would also work, but a fixed-width slicer is more robust to
embedded commas in free-text fields like ``Crash Location``.

Public surface
--------------

* :class:`Severity`, :func:`epdo_for` — translate the single-char
  severity code (``F``/``I``/``P``) into the standard NJDOT EPDO
  weighting (``542 / 11 / 1``).
* :func:`parse_crash_table` — read one ``BergenYYYYAccidents.txt``
  and return a :class:`pandas.DataFrame` with cleaned fields.
* :func:`parse_pedestrian_table` — same for the Pedestrian table.
* :func:`geocode_by_street_name` — fallback geocoder for the
  ~70% of records with empty lat/lon: spatial-join the
  ``(crash_location, cross_street)`` text fields against an OSM
  ways GeoDataFrame.
* :func:`assign_to_osm_way` — nearest-neighbour join of crashes
  (with lat/lon) to OSM ways for segment-level aggregation.
* :func:`aggregate_by_segment` — group crashes by ``osm_way_id``
  and emit per-segment counts + EPDO totals.

Why fixed-width and not pandas' built-in CSV reader? A handful of
rows have stray double-quotes or unmatched commas inside the
``Other Property Damage`` free-text field at byte 384–463 that would
break ``pd.read_csv``. The fixed-width slicer doesn't care.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field layout (NJDOT Crash Table 2017–present, 470 chars)
# ---------------------------------------------------------------------------


# Each entry: (column_name, start_byte_1based, length_in_bytes)
# Source: docs/AccidentTable2017+.pdf (NJDOT, 2018-10-12).
_CRASH_FIELDS: list[tuple[str, int, int]] = [
    ("year",                       1, 4),
    ("county_code",                5, 2),
    ("muni_code",                  7, 2),
    ("case_number",                9, 23),
    ("county_name",               33, 12),
    ("muni_name",                 46, 24),
    ("crash_date",                71, 10),   # MM/DD/YYYY
    ("crash_dow",                 82, 2),    # SU/MO/TU/...
    ("crash_time",                85, 4),    # HHMM (24h)
    ("police_dept_code",          90, 2),
    ("police_dept",               93, 25),
    ("police_station",           119, 15),
    ("total_killed",             135, 2),
    ("total_injured",            138, 2),
    ("ped_killed",               141, 2),
    ("ped_injured",              144, 2),
    ("severity_code",            147, 1),    # F/I/P (Fatal/Injury/PDO)
    ("intersection",             149, 1),    # I/N
    ("alcohol_involved",         151, 1),    # Y/N
    ("hazmat_involved",          153, 1),    # Y/N
    ("crash_type_code",          155, 2),
    ("total_vehicles",           158, 2),
    ("crash_location",           161, 50),
    ("location_direction",       212, 1),
    ("route",                    214, 4),
    ("route_suffix",             219, 1),
    ("sri",                      221, 16),
    ("milepost",                 238, 7),
    ("road_system",              246, 2),
    ("road_horiz_alignment",     252, 2),
    ("road_grade",               255, 2),
    ("road_surface_type",        258, 2),
    ("surface_condition",        261, 2),
    ("light_condition",          264, 2),
    ("environment",              267, 2),
    ("road_divided_by",          270, 2),
    ("temp_traffic_control",     273, 2),
    ("dist_to_cross",            276, 4),
    ("unit_of_measurement",      281, 2),
    ("dir_from_cross",           284, 1),
    ("cross_street",             286, 35),
    ("is_ramp",                  322, 1),
    ("ramp_route_name",          324, 25),
    ("ramp_route_dir",           350, 2),
    ("posted_speed",             353, 2),
    ("posted_speed_cross",       356, 2),
    ("first_harmful_event",      359, 2),
    ("latitude",                 362, 9),
    ("longitude",                372, 9),
    ("cell_phone",               382, 1),
    ("other_prop_damage",        384, 80),
    ("reporting_badge",          465, 5),
]


# Pedestrian table layout (NJDOT 2017+) — only the fields we need.
# The PDF embedding made it unreadable, so I derived these by
# inspecting raw bytes; the join keys on the first 4 fields are
# identical to the Crash table. Add more columns as needed later.
_PED_FIELDS: list[tuple[str, int, int]] = [
    ("year",          1, 4),
    ("county_code",   5, 2),
    ("muni_code",     7, 2),
    ("case_number",   9, 23),
    # Pedestrian-specific fields (best-effort — refine when needed).
    ("ped_age",      66, 3),
    ("ped_sex",      70, 1),
    ("ped_position", 76, 2),
]


# ---------------------------------------------------------------------------
# Severity / EPDO weighting
# ---------------------------------------------------------------------------


@dataclass
class Severity:
    """Map raw NJDOT severity codes to friendly labels + EPDO weights.

    The NJDOT Equivalent-Property-Damage-Only (EPDO) formula weights
    a fatality at 542× a PDO crash and an injury at 11× a PDO crash,
    matching the values in NJDOT's *Crash Reduction Factor Manual*.
    """

    code: str
    label: str
    epdo: float


SEVERITY_BY_CODE: dict[str, Severity] = {
    # Legacy NJDOT zip codes (collapsed scale).
    "F": Severity("F", "Fatal", 542.0),
    "I": Severity("I", "Injury", 11.0),
    "P": Severity("P", "PDO", 1.0),
    # KABCO codes returned by the NJDOT dashboard (Numetric).
    # Weights from FHWA HSM (Highway Safety Manual) Table A-1, NJ
    # statewide adjusted: K=542, A=66, B=11, C=11, O=1. This makes
    # severity-weighted comparisons compatible with both data
    # sources without a separate column.
    "K": Severity("K", "Fatal", 542.0),
    "A": Severity("A", "Suspected Serious Injury", 66.0),
    "B": Severity("B", "Suspected Minor Injury", 11.0),
    "C": Severity("C", "Possible Injury", 11.0),
    "O": Severity("O", "No Apparent Injury", 1.0),
}


# Dashboard JSON returns severity as `"Suspected Serious Injury (A)"`
# strings; pre-built lookup keeps the parser branch-free.
KABCO_LABEL_TO_CODE: dict[str, str] = {
    "Fatal Injury (K)": "K",
    "Suspected Serious Injury (A)": "A",
    "Suspected Minor Injury (B)": "B",
    "Possible Injury (C)": "C",
    "No Apparent Injury (O)": "O",
}


def epdo_for(severity_code: str) -> float:
    """Return the EPDO weight for a severity code (NaN-safe).

    Accepts both legacy ``F/I/P`` and KABCO ``K/A/B/C/O`` codes.
    """
    if not isinstance(severity_code, str):
        return 0.0
    return SEVERITY_BY_CODE.get(severity_code.strip().upper(),
                                Severity("?", "Unknown", 0.0)).epdo


def severity_label(severity_code: str) -> str:
    if not isinstance(severity_code, str):
        return "Unknown"
    return SEVERITY_BY_CODE.get(severity_code.strip().upper(),
                                Severity("?", "Unknown", 0.0)).label


def kabco_to_code(label_or_code: str) -> str:
    """Normalise a dashboard severity string to a single KABCO letter.

    ``"Suspected Serious Injury (A)"`` → ``"A"``;
    ``"K"`` → ``"K"``; anything unknown → ``"O"`` (the safe-low
    default — counted as a PDO when totalling EPDO).
    """
    if not isinstance(label_or_code, str):
        return "O"
    s = label_or_code.strip()
    if s in KABCO_LABEL_TO_CODE:
        return KABCO_LABEL_TO_CODE[s]
    if len(s) == 1 and s.upper() in SEVERITY_BY_CODE:
        return s.upper()
    return "O"


# ---------------------------------------------------------------------------
# Fixed-width slicer
# ---------------------------------------------------------------------------


def _slice_fields(line: str, layout: list[tuple[str, int, int]]
                  ) -> dict[str, str]:
    """Slice a raw line into a dict of trimmed string fields."""
    out: dict[str, str] = {}
    for name, start, length in layout:
        end = start - 1 + length
        if end > len(line):
            out[name] = ""
        else:
            out[name] = line[start - 1:end].strip()
    return out


# ---------------------------------------------------------------------------
# Crash table parser
# ---------------------------------------------------------------------------


def parse_crash_table(
    txt_path: Path,
    *,
    muni_codes: Iterable[str] | None = None,
    muni_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a NJDOT ``BergenYYYYAccidents.txt`` and clean the fields.

    Parameters
    ----------
    txt_path
        Path to the unzipped fixed-width ``.txt`` file.
    muni_codes
        Filter to crashes whose ``muni_code`` is in this set
        (e.g. ``{"29"}`` for Leonia). Strips leading zeros so callers
        can pass either ``"29"`` or ``"0229"`` if they want.
    muni_names
        Alternative filter on ``muni_name`` (case-insensitive
        substring match). Useful when a borough's code changes.
    """
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    code_filter: set[str] | None = None
    if muni_codes is not None:
        code_filter = {str(c).strip().lstrip("0") for c in muni_codes}
        # Normalise empty string back to "0" so muni_code "0" filter works.
        code_filter = {c or "0" for c in code_filter}

    name_filter: list[str] | None = None
    if muni_names is not None:
        name_filter = [str(n).strip().upper() for n in muni_names if n]

    # A record is the layout's max field end (469 chars in the 2017+
    # layout) plus the inter-field commas — but the trailing
    # ``Reporting Badge No.`` field is sometimes padded short, so we
    # accept anything ≥ ``min_record_len`` and let the slicer handle
    # missing tail bytes.
    min_record_len = 380   # well past Lat/Lon at 362–380
    rows: list[dict] = []
    with txt_path.open("r", encoding="latin-1") as fh:
        for line in fh:
            stripped = line.rstrip("\r\n")
            if len(stripped) < min_record_len:
                continue
            row = _slice_fields(stripped, _CRASH_FIELDS)

            if code_filter is not None:
                muni_norm = row["muni_code"].lstrip("0") or "0"
                if muni_norm not in code_filter:
                    continue
            if name_filter is not None:
                upper = row["muni_name"].upper()
                if not any(n in upper for n in name_filter):
                    continue
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ("year", "total_killed", "total_injured",
                "ped_killed", "ped_injured", "total_vehicles",
                "posted_speed", "posted_speed_cross", "dist_to_cross"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lat/lon: blank → NaN; strip any residual non-numeric.
    df["latitude"] = pd.to_numeric(
        df["latitude"].replace("", pd.NA), errors="coerce",
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"].replace("", pd.NA), errors="coerce",
    )
    # NJDOT publishes 6-decimal degrees; some rows have impossible 0.0
    # coordinates from form-blank. Drop those.
    bad_xy = (df["latitude"].fillna(0) == 0) | (df["longitude"].fillna(0) == 0)
    df.loc[bad_xy, ["latitude", "longitude"]] = pd.NA

    # Date / datetime.
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce",
                                      format="%m/%d/%Y")
    df["crash_time_hhmm"] = pd.to_numeric(df["crash_time"], errors="coerce")
    df["crash_hour"] = (df["crash_time_hhmm"] // 100).astype("Int64")

    # Severity → label + EPDO.
    df["severity_label"] = df["severity_code"].map(severity_label)
    df["epdo"] = df["severity_code"].map(epdo_for)

    # Booleans.
    for flag, col_in in (("alcohol", "alcohol_involved"),
                         ("hazmat", "hazmat_involved"),
                         ("at_intersection", "intersection")):
        df[flag] = df[col_in].isin(["Y", "I"])
    df["ped_involved"] = (
        df["ped_killed"].fillna(0) + df["ped_injured"].fillna(0) > 0
    )

    # Tidy text — collapse multi-space, upper-case for join keys.
    for col in ("crash_location", "cross_street", "muni_name"):
        df[col] = (
            df[col].fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Stable unique id (case_number itself isn't globally unique
    # across years — combine with year).
    df["crash_id"] = (
        df["year"].astype(str) + "_"
        + df["county_code"] + df["muni_code"] + "_"
        + df["case_number"].str.strip()
    )
    return df


# ---------------------------------------------------------------------------
# Pedestrian table parser
# ---------------------------------------------------------------------------


def parse_pedestrian_table(
    txt_path: Path,
    *,
    muni_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a NJDOT Pedestrian table and return age/sex per ped victim.

    Pedestrian rows join back to the Crash table via
    ``(year, county_code, muni_code, case_number)``.
    """
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    code_filter: set[str] | None = None
    if muni_codes is not None:
        code_filter = {str(c).strip().lstrip("0") or "0" for c in muni_codes}

    rows: list[dict] = []
    with txt_path.open("r", encoding="latin-1") as fh:
        for line in fh:
            stripped = line.rstrip("\r\n")
            if len(stripped) < 80:
                continue
            row = _slice_fields(stripped, _PED_FIELDS)
            if code_filter is not None:
                muni_norm = row["muni_code"].lstrip("0") or "0"
                if muni_norm not in code_filter:
                    continue
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["ped_age"] = pd.to_numeric(df["ped_age"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["crash_id"] = (
        df["year"].astype("Int64").astype(str) + "_"
        + df["county_code"] + df["muni_code"] + "_"
        + df["case_number"].str.strip()
    )
    return df


# ---------------------------------------------------------------------------
# NJDOT Crash Data Dashboard (Numetric/AASHTOWare Safety) JSON parser
# ---------------------------------------------------------------------------
#
# Maps the dashboard JSON shape to the **same column schema** as
# :func:`parse_crash_table` so the geocoder, segment aggregation, and
# visualisation layers can stay agnostic to which data source they
# came from. A few notable upgrades over the legacy fixed-width zip
# loader:
#
# * Severity is on the FHWA KABCO scale (K/A/B/C/O) instead of the
#   collapsed F/I/P scale, so we can distinguish "Suspected Serious
#   Injury" (the FHWA KSI denominator) from "Possible Injury".
# * ``Geopoint Calculated`` is a NJDOT-snapped fallback when the raw
#   ``latitude``/``longitude`` are missing, taking geocoding rates
#   from ~30% to ~92%.
# * Years 2019–2026 are available (vs 2017–2022 in the zips).
# * Pre-computed safety metrics: per-segment AADT, crash-rate
#   contribution (CRC), SHSP emphasis areas, all already keyed to
#   NJDOT's network-screening segment IDs.

# Maps dashboard column → output column (the schema produced by
# :func:`parse_crash_table`). Anything not in this map is dropped.
_DASHBOARD_FIELD_MAP: dict[str, str] = {
    "id_cr":                   "id_cr",
    "casenumber":              "case_number",
    "dateofcrash":             "_dateofcrash_str",
    "Time of Crash":           "_time_of_crash_str",
    "Day of Week":             "_dow_long",
    "year":                    "year",
    "County":                  "county_name",
    "Municipality":            "muni_name",
    "Severity Rating (5)":     "_severity_label_raw",
    "fatalcrashind":           "_fatal_ind",
    "fatalitycount":           "total_killed",
    "injurycount":             "total_injured",
    "pedestrianfatalitycount": "ped_killed",
    "pedestrianinjurycount":   "ped_injured",
    "vehiclecount":            "total_vehicles",
    "Crash Type":              "crash_type",
    "First Harmful Event":     "first_harmful_event_label",
    "Pedestrian Involved":     "_ped_involved_yn",
    "Bicyclist Involved":      "_bike_involved_yn",
    "Alcohol Involved":        "_alcohol_yn",
    "Hazmat Involved":         "_hazmat_yn",
    "At Intersection":         "_intersection_yn",
    "streetname":              "crash_location",
    "intersectstreetname":     "cross_street",
    "streetsri":               "sri",
    "milepost":                "milepost",
    "distancefromintersection": "dist_to_cross",
    "Direction From Intersection": "dir_from_cross",
    "Light Condition":         "light_condition",
    "Weather Condition":       "weather",
    "Surface Condition":       "surface_condition",
    "Road System":             "road_system",
    "Road Surface Type":       "road_surface_type",
    "Functional Class_first":  "functional_class",
    "Urban or Rural_first":    "urban_rural",
    "speedlimit":              "posted_speed",
    "intersectionspeedlimit":  "posted_speed_cross",
    "Unable to Geocode Crash": "_unable_to_geocode_yn",
    "SegmentID_first_2":       "njdot_segment_id",
    "__Nu_Segment_AADT__":     "njdot_segment_aadt",
    "__Nu_Segment_CRC__":      "njdot_segment_crc",
    "__Nu_Segment_CPMC__":     "njdot_segment_cpmc",
    "__Nu_Window_CRC__":       "njdot_window_crc",
    "__Nu_Intersection_TEV__": "njdot_intersection_tev",
}


def _yn_to_bool(v: object) -> bool:
    """Dashboard uses ``"Yes"``/``"No"`` strings; normalise to bool."""
    if isinstance(v, str):
        return v.strip().lower() == "yes"
    return bool(v)


def _coalesce_geopoint(row: dict) -> tuple[float | None, float | None,
                                            str | None]:
    """Extract the best (lat, lon, source) tuple for a dashboard row.

    Priority: raw ``latitude``/``longitude`` floats → ``Geopoint``
    dict → ``Geopoint Calculated`` dict. Returns ``(None, None,
    None)`` when nothing usable is present.
    """
    lat_raw = row.get("latitude")
    lon_raw = row.get("longitude")
    try:
        if isinstance(lat_raw, (int, float)) and float(lat_raw) != 0 and \
                isinstance(lon_raw, (int, float)) and float(lon_raw) != 0:
            return float(lat_raw), float(lon_raw), "raw"
    except (TypeError, ValueError):
        pass

    for key, source in (("Geopoint", "raw"),
                         ("Geopoint Calculated", "njdot_calculated")):
        g = row.get(key)
        if isinstance(g, dict):
            lat = g.get("lat")
            lon = g.get("lon")
            try:
                if lat and lon and float(lat) != 0 and float(lon) != 0:
                    return float(lat), float(lon), source
            except (TypeError, ValueError):
                continue
    return None, None, None


def parse_dashboard_json(
    json_path: "Path | str | dict",
    *,
    drop_state_system: bool = False,
) -> pd.DataFrame:
    """Parse the NJDOT dashboard's ``/api/.../search`` JSON.

    Accepts either a path on disk or a pre-loaded dict (so callers
    that have already run :mod:`scripts.14_build_crash_overlay` to
    fetch the rows can pass them in directly).

    Parameters
    ----------
    json_path
        Path to a JSON file with shape ``{"data": {"rows": [...]}}``,
        or a dict matching that shape.
    drop_state_system
        If True, drop crashes whose ``Road System`` is ``Interstate``
        / ``State Highway`` / ``State Park / Inst. / Authority``.
        Useful when the council story is "what's happening on
        Leonia *local* streets" — those are typically a separate
        jurisdictional conversation. Default ``False`` keeps every
        row so the data lake is complete; flip it on at the
        visualisation step.

    Returns
    -------
    DataFrame with the same columns produced by
    :func:`parse_crash_table` (so the geocoder, segment aggregation,
    and stakeholder viz can be source-agnostic), plus a few extra
    dashboard-only columns prefixed ``njdot_`` and
    ``shsp_emphasis_areas`` (list[str] of NJDOT's pre-computed
    Strategic Highway Safety Plan tags).
    """
    if isinstance(json_path, (str, Path)):
        with Path(json_path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = json_path

    rows_raw = (payload.get("data", {}) or {}).get("rows", []) or []
    if not rows_raw:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for r in rows_raw:
        out: dict[str, object] = {}
        for src, dst in _DASHBOARD_FIELD_MAP.items():
            out[dst] = r.get(src)
        # Coordinates with calculated-fallback.
        lat, lon, geo_source = _coalesce_geopoint(r)
        out["latitude"] = lat
        out["longitude"] = lon
        out["geocoded_lat"] = lat
        out["geocoded_lon"] = lon
        out["geocoded_method"] = (
            "raw" if geo_source == "raw"
            else ("njdot_calculated" if geo_source == "njdot_calculated"
                  else "none")
        )
        # SHSP emphasis areas often come back as a list; preserve.
        emphasis = r.get("SHSP Emphasis Areas") or []
        if isinstance(emphasis, list):
            out["shsp_emphasis_areas"] = ", ".join(str(x) for x in emphasis)
        else:
            out["shsp_emphasis_areas"] = str(emphasis or "")
        out_rows.append(out)

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    # Numeric coercions.
    for col in ("year", "total_killed", "total_injured", "ped_killed",
                "ped_injured", "total_vehicles",
                "posted_speed", "posted_speed_cross", "dist_to_cross",
                "milepost", "njdot_segment_aadt", "njdot_segment_crc",
                "njdot_segment_cpmc", "njdot_window_crc",
                "njdot_intersection_tev"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date / hour.
    df["crash_date"] = pd.to_datetime(
        df["_dateofcrash_str"], errors="coerce", format="%Y-%m-%d"
    )
    df["crash_hour"] = pd.to_numeric(
        df["_time_of_crash_str"].astype(str).str.slice(0, 2),
        errors="coerce",
    ).astype("Int64")
    df["crash_dow"] = df["_dow_long"].fillna("").str.slice(0, 2).str.upper()

    # KABCO severity → letter code → label + EPDO weight.
    df["severity_code"] = df["_severity_label_raw"].apply(kabco_to_code)
    df["severity_label"] = df["severity_code"].map(severity_label)
    df["epdo"] = df["severity_code"].map(epdo_for)

    # Yes/No flags → bools.
    df["alcohol"] = df["_alcohol_yn"].apply(_yn_to_bool)
    df["hazmat"] = df["_hazmat_yn"].apply(_yn_to_bool)
    df["at_intersection"] = df["_intersection_yn"].apply(_yn_to_bool)
    df["ped_involved"] = df["_ped_involved_yn"].apply(_yn_to_bool)
    df["bike_involved"] = df["_bike_involved_yn"].apply(_yn_to_bool)
    df["unable_to_geocode"] = df["_unable_to_geocode_yn"].apply(_yn_to_bool)

    # Tidy text — collapse multi-space, strip the legacy ``**``
    # markers that appear in some street names.
    for col in ("crash_location", "cross_street", "muni_name",
                "county_name"):
        if col in df.columns:
            df[col] = (
                df[col].fillna("")
                .astype(str)
                .str.replace(r"\*+", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # Stable id — match the legacy schema's `crash_id` shape so we
    # can union the two sources without collisions.
    df["crash_id"] = (
        df["year"].astype("Int64").astype(str) + "_DASH_"
        + df["id_cr"].astype("Int64").astype(str)
    )
    df["county_code"] = "02"
    df["muni_code"] = "29"
    df["data_source"] = "njdot_dashboard"

    # Optional: drop state-system rows (Interstate / NJ Turnpike /
    # State Highway). Leonia is bordered by I-95 and the GW Bridge
    # approach, which generates a lot of crashes that aren't a
    # municipal-government concern. Restricting to County + Municipal
    # gives the "what's happening on Leonia *local* streets" view.
    if drop_state_system:
        is_state = df["road_system"].fillna("").str.lower().isin({
            "state authority",         # NJ Turnpike / Garden State / etc.
            "njdot state highway",     # state-maintained routes
            "interstate",
            "state park / inst. / authority",
            "us govt property",
        })
        df = df[~is_state].copy()

    # Drop the temp ``_xxx_yn`` / ``_xxx_str`` scratch columns.
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")],
                 errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Name-based geocoder fallback
# ---------------------------------------------------------------------------


_STREET_TYPE_NORMALISATION = {
    "AVENUE": "AVE", "AVE": "AVE",
    "STREET": "ST", "ST": "ST",
    "ROAD": "RD", "RD": "RD",
    "PLACE": "PL", "PL": "PL",
    "BOULEVARD": "BLVD", "BLVD": "BLVD",
    "DRIVE": "DR", "DR": "DR",
    "TERRACE": "TER", "TER": "TER",
    "COURT": "CT", "CT": "CT",
    "PARKWAY": "PKWY", "PKWY": "PKWY",
    "HIGHWAY": "HWY", "HWY": "HWY",
    "LANE": "LN", "LN": "LN",
}


def _normalise_name(name: str) -> str:
    """Uppercase + collapse + drop trailing ``**`` and route prefixes."""
    if not isinstance(name, str):
        return ""
    s = name.upper()
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\bNJ\s*\d+\b", "", s)
    s = re.sub(r"\bROUTE\s*\d+\b", "", s)
    s = re.sub(r"\bUS\s*\d+\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Tail-token type normalisation.
    parts = s.split()
    if parts:
        last = parts[-1]
        if last in _STREET_TYPE_NORMALISATION:
            parts[-1] = _STREET_TYPE_NORMALISATION[last]
            s = " ".join(parts)
    return s


def geocode_by_street_name(
    crashes: pd.DataFrame,
    osm_ways: "pd.DataFrame",
    *,
    muni_polygon=None,
) -> pd.DataFrame:
    """Geocode crashes lacking lat/lon by name-matching to OSM ways.

    Parameters
    ----------
    crashes
        DataFrame from :func:`parse_crash_table`.
    osm_ways
        GeoDataFrame with at least ``osm_way_id``, ``street_name``,
        and a line ``geometry`` in WGS84. Typical input is the
        SUMO edge-geometry GeoDataFrame joined with
        ``leonia.edgedata.meta.csv`` (so each row has both an OSM
        way id and a clean street name).
    muni_polygon
        Optional shapely polygon — restrict the geocoder to candidate
        OSM ways inside the borough. Reduces false matches when a
        crash references a street name that exists in multiple
        municipalities.

    Returns
    -------
    DataFrame with the same rows as ``crashes`` plus
    ``geocoded_lat``, ``geocoded_lon``, ``geocoded_osm_way_id``,
    ``geocoded_method`` (``raw``/``street``/``intersection``/``none``).
    The original ``latitude``/``longitude`` are unchanged.
    """
    out = crashes.copy()
    out["geocoded_lat"] = out["latitude"]
    out["geocoded_lon"] = out["longitude"]
    out["geocoded_osm_way_id"] = pd.NA
    out["geocoded_method"] = pd.NA

    has_xy = out["latitude"].notna() & out["longitude"].notna()
    out.loc[has_xy, "geocoded_method"] = "raw"

    if osm_ways is None or osm_ways.empty:
        return out
    if "geometry" not in osm_ways.columns:
        return out

    needs = out[~has_xy].copy()
    if needs.empty:
        return out

    # Build a name → list[(osm_way_id, geometry)] map.
    candidates = osm_ways.copy()
    if muni_polygon is not None:
        # Keep only candidates that intersect (or lie within) the muni.
        try:
            candidates = candidates[candidates.geometry.intersects(muni_polygon)]
        except Exception:
            pass
    candidates["_norm"] = candidates["street_name"].fillna("").apply(_normalise_name)
    by_name: dict[str, list[tuple]] = {}
    for _, row in candidates.iterrows():
        nm = row["_norm"]
        if not nm:
            continue
        by_name.setdefault(nm, []).append(
            (row.get("osm_way_id"), row.geometry)
        )

    n_street = 0
    n_intx = 0
    for idx, row in needs.iterrows():
        primary = _normalise_name(row.get("crash_location", ""))
        cross = _normalise_name(row.get("cross_street", ""))
        if not primary:
            continue

        prim_cands = by_name.get(primary, [])
        if not prim_cands:
            # Fallback: substring containment (e.g. "BROAD AVE **" → "BROAD AVE").
            for k in by_name:
                if primary in k or k in primary:
                    prim_cands = by_name[k]
                    break
        if not prim_cands:
            continue

        # If we have a cross street, find the candidate whose centroid
        # is closest to *its* nearest cross-street geometry.
        cross_cands = by_name.get(cross, []) if cross else []
        if cross_cands:
            best_pair: tuple[float, object, object] | None = None
            for _, pg in prim_cands:
                if pg is None or pg.is_empty:
                    continue
                for _, cg in cross_cands:
                    if cg is None or cg.is_empty:
                        continue
                    try:
                        d = pg.distance(cg)
                    except Exception:
                        continue
                    if best_pair is None or d < best_pair[0]:
                        # Use the midpoint of the two segments as the
                        # crash location estimate.
                        try:
                            ip = pg.intersection(cg)
                            point = ip if not ip.is_empty else (
                                pg.interpolate(pg.project(cg.centroid))
                            )
                        except Exception:
                            point = pg.interpolate(pg.length / 2)
                        best_pair = (d, pg, point)
            if best_pair is not None:
                _, _, point = best_pair
                p = point if hasattr(point, "x") else point.centroid
                way_id = next(
                    (w for w, g in prim_cands if g is not None
                     and best_pair[1] is g),
                    None,
                )
                out.at[idx, "geocoded_lon"] = float(p.x)
                out.at[idx, "geocoded_lat"] = float(p.y)
                out.at[idx, "geocoded_osm_way_id"] = way_id
                out.at[idx, "geocoded_method"] = "intersection"
                n_intx += 1
                continue

        # Fallback: midpoint of the first matching primary segment.
        way_id, geom = prim_cands[0]
        if geom is None or geom.is_empty:
            continue
        mid = geom.interpolate(geom.length / 2)
        out.at[idx, "geocoded_lon"] = float(mid.x)
        out.at[idx, "geocoded_lat"] = float(mid.y)
        out.at[idx, "geocoded_osm_way_id"] = way_id
        out.at[idx, "geocoded_method"] = "street"
        n_street += 1

    out.loc[out["geocoded_method"].isna(), "geocoded_method"] = "none"
    logger.info(
        "geocoder: %d crashes resolved by intersection, %d by street name "
        "(of %d that lacked lat/lon)",
        n_intx, n_street, len(needs),
    )
    return out


# ---------------------------------------------------------------------------
# OSM-way assignment + segment aggregation
# ---------------------------------------------------------------------------


def assign_to_osm_way(
    crashes: pd.DataFrame,
    osm_ways: "pd.DataFrame",
    *,
    max_distance_m: float = 50.0,
) -> pd.DataFrame:
    """Snap each geocoded crash to the closest OSM way.

    Crashes that already have an ``geocoded_osm_way_id`` (from the
    name-based geocoder) keep it; the rest are matched spatially
    using the lat/lon. Snaps further than ``max_distance_m`` are
    discarded (the crash is left without a way id).
    """
    out = crashes.copy()
    if "geocoded_osm_way_id" not in out.columns:
        out["geocoded_osm_way_id"] = pd.NA

    have_xy = out["geocoded_lat"].notna() & out["geocoded_lon"].notna()
    needs_assign = have_xy & out["geocoded_osm_way_id"].isna()
    if not needs_assign.any() or osm_ways is None or osm_ways.empty:
        return out

    import geopandas as gpd
    from shapely.geometry import Point

    pts_gdf = gpd.GeoDataFrame(
        out.loc[needs_assign].assign(
            geometry=[Point(lon, lat) for lon, lat in
                      zip(out.loc[needs_assign, "geocoded_lon"],
                          out.loc[needs_assign, "geocoded_lat"])],
        ),
        geometry="geometry", crs="EPSG:4326",
    )
    edges_m = osm_ways.to_crs(3857)
    pts_m = pts_gdf.to_crs(3857)

    # ``sjoin_nearest`` is the fast path; it carries the matched way's
    # osm_way_id back onto each crash row.
    joined = gpd.sjoin_nearest(
        pts_m, edges_m[["osm_way_id", "geometry"]],
        how="left", max_distance=max_distance_m,
        distance_col="_dist_m",
    )
    for idx, way in zip(joined.index, joined["osm_way_id"]):
        if pd.notna(way):
            out.at[idx, "geocoded_osm_way_id"] = way
    return out


# ---------------------------------------------------------------------------
# Segment-level aggregation
# ---------------------------------------------------------------------------


def aggregate_by_segment(
    crashes: pd.DataFrame,
    *,
    osm_meta: "pd.DataFrame | None" = None,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Group crashes by ``geocoded_osm_way_id`` and emit per-segment counts.

    Parameters
    ----------
    crashes
        DataFrame from :func:`assign_to_osm_way`.
    osm_meta
        Optional ``leonia.edgedata.meta.csv``-shaped DataFrame to
        carry the canonical ``street_name`` onto every segment row.
    years
        Restrict to these years before aggregating (e.g. last 5).

    Returns
    -------
    DataFrame with columns:
        ``osm_way_id``, ``street_name``, ``n_crashes``, ``n_fatal``,
        ``n_injury``, ``n_pdo``, ``n_ksi`` (fatal + suspected
        serious-injury proxy = fatal + injury), ``n_ped``,
        ``epdo_total``, ``years_covered``.
    """
    df = crashes.copy()
    if "geocoded_osm_way_id" not in df.columns:
        return pd.DataFrame()
    df = df[df["geocoded_osm_way_id"].notna()]
    if years is not None:
        years_set = {int(y) for y in years}
        df = df[df["year"].astype("Int64").isin(years_set)]
    if df.empty:
        return pd.DataFrame()

    # Severity bucketing — accepts both legacy NJDOT zip scale
    # (F/I/P) and dashboard KABCO scale (K/A/B/C/O). Fatal includes
    # F + K; injury includes I + A + B + C; PDO is P + O. KSI uses
    # the FHWA convention (Fatal + Suspected Serious Injury, i.e.
    # K + A) when the data is on KABCO; collapses to Fatal + Injury
    # for the legacy F/I/P scale where suspected-serious vs minor
    # can't be told apart.
    fatal_codes = {"F", "K"}
    serious_codes = {"A"}
    other_injury_codes = {"I", "B", "C"}
    pdo_codes = {"P", "O"}

    def _count_codes(s, codes):
        return s.isin(codes).sum()

    grouped = df.groupby("geocoded_osm_way_id", as_index=False).agg(
        n_crashes=("crash_id", "count"),
        n_fatal=("severity_code", lambda s: _count_codes(s, fatal_codes)),
        n_serious=("severity_code", lambda s: _count_codes(s, serious_codes)),
        n_injury=("severity_code",
                  lambda s: _count_codes(s, other_injury_codes)),
        n_pdo=("severity_code", lambda s: _count_codes(s, pdo_codes)),
        n_ped=("ped_involved", "sum"),
        epdo_total=("epdo", "sum"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    # FHWA KSI = killed + suspected-serious-injury. Legacy F/I/P
    # data has no "serious" bucket so fall back to fatal + all injury.
    has_kabco = (df["severity_code"].isin({"K", "A", "B", "C", "O"}).any())
    if has_kabco:
        grouped["n_ksi"] = grouped["n_fatal"] + grouped["n_serious"]
        grouped["n_injury"] = grouped["n_injury"] + grouped["n_serious"]
    else:
        grouped["n_ksi"] = grouped["n_fatal"] + grouped["n_injury"]
    grouped["years_covered"] = (
        grouped["last_year"].astype("Int64")
        - grouped["first_year"].astype("Int64")
        + 1
    )
    grouped = grouped.rename(columns={"geocoded_osm_way_id": "osm_way_id"})

    if osm_meta is not None and not osm_meta.empty:
        meta = osm_meta[["osm_way_id", "street_name"]].drop_duplicates(
            "osm_way_id"
        )
        # Coerce both sides to nullable Int64 to avoid str/int mismatches.
        meta = meta.copy()
        meta["osm_way_id"] = pd.to_numeric(
            meta["osm_way_id"], errors="coerce"
        ).astype("Int64")
        grouped["osm_way_id"] = pd.to_numeric(
            grouped["osm_way_id"], errors="coerce"
        ).astype("Int64")
        grouped = grouped.merge(meta, on="osm_way_id", how="left")
    return grouped.sort_values("epdo_total", ascending=False).reset_index(
        drop=True
    )
