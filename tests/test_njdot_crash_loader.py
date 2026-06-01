"""Tests for the NJDOT crash loader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from leonia_traffic.data.njdot_crash_loader import (
    aggregate_by_segment,
    epdo_for,
    geocode_by_street_name,
    kabco_to_code,
    parse_crash_table,
    parse_dashboard_json,
    severity_label,
    _coalesce_geopoint,
    _normalise_name,
)


# ---------------------------------------------------------------------------
# EPDO scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code,label,epdo", [
    # Legacy NJDOT zip scale (F/I/P).
    ("F", "Fatal", 542.0),
    ("I", "Injury", 11.0),
    ("P", "PDO", 1.0),
    # KABCO scale (K/A/B/C/O) used by the NJDOT dashboard.
    ("K", "Fatal", 542.0),
    ("A", "Suspected Serious Injury", 66.0),
    ("B", "Suspected Minor Injury", 11.0),
    ("C", "Possible Injury", 11.0),
    ("O", "No Apparent Injury", 1.0),
    ("?", "Unknown", 0.0),
    ("",  "Unknown", 0.0),
])
def test_severity_table(code: str, label: str, epdo: float) -> None:
    """Severity codes should map to NJDOT-canonical EPDO weights."""
    assert severity_label(code) == label
    assert epdo_for(code) == pytest.approx(epdo)


def test_epdo_handles_non_string() -> None:
    """``epdo_for`` should not blow up on NaN / None inputs."""
    assert epdo_for(None) == 0.0  # type: ignore[arg-type]
    assert epdo_for(float("nan")) == 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("BROAD AVE **", "BROAD AVE"),
    ("Broad Avenue", "BROAD AVE"),
    ("NJ 93", ""),
    ("ROUTE 4 EAST", "EAST"),
    ("Glenwood   Ave", "GLENWOOD AVE"),
    ("Park Place", "PARK PL"),
])
def test_normalise_name(raw: str, expected: str) -> None:
    assert _normalise_name(raw) == expected


# ---------------------------------------------------------------------------
# Fixed-width parser
# ---------------------------------------------------------------------------


def _fixture_line(*, year: str = "2018", county: str = "02",
                  muni: str = "29", case: str = "18-99001",
                  county_name: str = "BERGEN",
                  muni_name: str = "LEONIA BORO",
                  date: str = "06/15/2018",
                  hhmm: str = "1730",
                  severity: str = "I",
                  total_killed: str = " 0",
                  total_injured: str = " 1",
                  ped_inv: str = "N",
                  loc: str = "BROAD AVE",
                  cross: str = "ELM ST",
                  lat: str = "         ",
                  lon: str = "         ") -> str:
    """Build a 470-byte NJDOT crash-table record for testing.

    Only the fields we care about for the smoke-test are wired up;
    the rest are filled with spaces or zeros.
    """
    line = bytearray(b" " * 470)

    def put(start_1based: int, length: int, value: str) -> None:
        b = value.encode("latin-1")[:length].ljust(length, b" ")
        line[start_1based - 1:start_1based - 1 + length] = b

    put(1, 4, year)
    put(5, 2, county)
    put(7, 2, muni)
    put(9, 23, case)
    put(32, 1, ",")
    put(33, 12, county_name)
    put(45, 1, ",")
    put(46, 24, muni_name)
    put(70, 1, ",")
    put(71, 10, date)
    put(81, 1, ",")
    put(82, 2, "FR")
    put(85, 4, hhmm)
    put(135, 2, total_killed)
    put(138, 2, total_injured)
    put(141, 2, " 0")
    put(144, 2, " 1" if ped_inv == "Y" else " 0")
    put(147, 1, severity)
    put(149, 1, "I")
    put(151, 1, "N")
    put(153, 1, ped_inv)
    put(155, 2, "06")
    put(158, 2, " 2")
    put(161, 50, loc)
    put(286, 35, cross)
    put(362, 9, lat)
    put(372, 9, lon)
    return line.decode("latin-1") + "\n"


def test_parse_crash_table_filters_muni(tmp_path: Path) -> None:
    """Muni filter should accept ``"29"`` or ``"029"`` interchangeably."""
    fixture = tmp_path / "B2018Accidents.txt"
    fixture.write_text(
        _fixture_line(muni="29", muni_name="LEONIA BORO")
        + _fixture_line(case="18-88001", muni="01",
                        muni_name="ALLENDALE BORO"),
        encoding="latin-1",
    )
    df_29 = parse_crash_table(fixture, muni_codes=["29"])
    df_029 = parse_crash_table(fixture, muni_codes=["029"])
    df_all = parse_crash_table(fixture)
    assert len(df_29) == 1
    assert len(df_029) == 1
    assert len(df_all) == 2
    assert df_29.iloc[0]["muni_name"] == "LEONIA BORO"


def test_parse_crash_table_severity_and_epdo(tmp_path: Path) -> None:
    """Severity codes should land in ``severity_label`` + ``epdo``."""
    fixture = tmp_path / "B2018Accidents.txt"
    fixture.write_text(
        _fixture_line(case="18-00001", severity="F", total_killed=" 1")
        + _fixture_line(case="18-00002", severity="I")
        + _fixture_line(case="18-00003", severity="P", total_injured=" 0"),
        encoding="latin-1",
    )
    df = parse_crash_table(fixture, muni_codes=["29"]).sort_values("case_number")
    assert df["severity_label"].tolist() == ["Fatal", "Injury", "PDO"]
    assert df["epdo"].tolist() == [542.0, 11.0, 1.0]


def test_parse_crash_table_handles_blank_latlon(tmp_path: Path) -> None:
    """Empty lat/lon bytes should become NaN, not 0.0."""
    fixture = tmp_path / "B2018Accidents.txt"
    fixture.write_text(
        _fixture_line(case="18-00001", lat="         ", lon="         ")
        + _fixture_line(case="18-00002", lat="40.870000", lon="-73.985000"),
        encoding="latin-1",
    )
    df = parse_crash_table(fixture, muni_codes=["29"]).sort_values("case_number")
    assert pd.isna(df.iloc[0]["latitude"])
    assert df.iloc[1]["latitude"] == pytest.approx(40.87)
    assert df.iloc[1]["longitude"] == pytest.approx(-73.985)


# ---------------------------------------------------------------------------
# Name-based geocoder fallback
# ---------------------------------------------------------------------------


def test_geocode_by_street_name_uses_intersection() -> None:
    """A crash at "BROAD AVE × ELM ST" should snap to the intersection."""
    from shapely.geometry import LineString
    import geopandas as gpd

    crashes = pd.DataFrame([
        {"crash_id": "x1", "crash_location": "BROAD AVE",
         "cross_street": "ELM ST", "latitude": pd.NA, "longitude": pd.NA},
    ])
    osm = gpd.GeoDataFrame(
        {
            "osm_way_id": [101, 202],
            "street_name": ["Broad Ave", "Elm St"],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(5, -5), (5, 5)]),
            ],
        },
        crs="EPSG:4326",
    )
    out = geocode_by_street_name(crashes, osm)
    row = out.iloc[0]
    assert row["geocoded_method"] == "intersection"
    assert row["geocoded_lat"] == pytest.approx(0.0)
    assert row["geocoded_lon"] == pytest.approx(5.0)
    assert row["geocoded_osm_way_id"] == 101


def test_geocode_falls_back_to_street_midpoint() -> None:
    """If no cross street matches, snap to the primary segment midpoint."""
    from shapely.geometry import LineString
    import geopandas as gpd

    crashes = pd.DataFrame([
        {"crash_id": "x1", "crash_location": "BROAD AVE",
         "cross_street": "NONEXISTENT WAY",
         "latitude": pd.NA, "longitude": pd.NA},
    ])
    osm = gpd.GeoDataFrame(
        {
            "osm_way_id": [101],
            "street_name": ["Broad Ave"],
            "geometry": [LineString([(0, 0), (10, 0)])],
        },
        crs="EPSG:4326",
    )
    out = geocode_by_street_name(crashes, osm)
    assert out.iloc[0]["geocoded_method"] == "street"
    assert out.iloc[0]["geocoded_lon"] == pytest.approx(5.0)


def test_geocode_preserves_raw_latlon() -> None:
    """Crashes that already have lat/lon should keep them and be flagged ``raw``."""
    from shapely.geometry import LineString
    import geopandas as gpd

    crashes = pd.DataFrame([
        {"crash_id": "x1", "crash_location": "BROAD AVE",
         "cross_street": "", "latitude": 40.87, "longitude": -73.985},
    ])
    osm = gpd.GeoDataFrame(
        {"osm_way_id": [101], "street_name": ["Broad Ave"],
         "geometry": [LineString([(0, 0), (1, 0)])]},
        crs="EPSG:4326",
    )
    out = geocode_by_street_name(crashes, osm)
    assert out.iloc[0]["geocoded_method"] == "raw"
    assert out.iloc[0]["geocoded_lat"] == pytest.approx(40.87)


# ---------------------------------------------------------------------------
# Segment aggregation
# ---------------------------------------------------------------------------


def test_aggregate_by_segment_groups_and_weights() -> None:
    """Per-segment EPDO and KSI should match the sum of inputs."""
    df = pd.DataFrame([
        # 3 crashes on way 101: 1 fatal, 1 injury, 1 PDO.
        {"crash_id": "a", "year": 2018, "geocoded_osm_way_id": 101,
         "severity_code": "F", "epdo": 542.0, "ped_involved": False},
        {"crash_id": "b", "year": 2019, "geocoded_osm_way_id": 101,
         "severity_code": "I", "epdo": 11.0, "ped_involved": True},
        {"crash_id": "c", "year": 2019, "geocoded_osm_way_id": 101,
         "severity_code": "P", "epdo": 1.0, "ped_involved": False},
        # 1 crash on way 202: PDO.
        {"crash_id": "d", "year": 2020, "geocoded_osm_way_id": 202,
         "severity_code": "P", "epdo": 1.0, "ped_involved": False},
    ])
    out = aggregate_by_segment(df).set_index("osm_way_id")
    assert out.loc[101, "n_crashes"] == 3
    assert out.loc[101, "n_fatal"] == 1
    assert out.loc[101, "n_ksi"] == 2  # fatal + injury
    assert out.loc[101, "n_ped"] == 1
    assert out.loc[101, "epdo_total"] == pytest.approx(554.0)
    assert out.loc[101, "years_covered"] == 2  # 2018-2019
    assert out.loc[202, "n_crashes"] == 1
    # Sorted descending by EPDO → way 101 should come first.
    assert out.index.tolist() == [101, 202]


def test_aggregate_by_segment_year_filter() -> None:
    """``years=`` should restrict the rows before aggregating."""
    df = pd.DataFrame([
        {"crash_id": "a", "year": 2018, "geocoded_osm_way_id": 101,
         "severity_code": "I", "epdo": 11.0, "ped_involved": False},
        {"crash_id": "b", "year": 2025, "geocoded_osm_way_id": 101,
         "severity_code": "F", "epdo": 542.0, "ped_involved": False},
    ])
    out = aggregate_by_segment(df, years=[2018]).set_index("osm_way_id")
    assert out.loc[101, "n_crashes"] == 1
    assert out.loc[101, "epdo_total"] == pytest.approx(11.0)


def test_aggregate_by_segment_kabco_scale() -> None:
    """KABCO data should split fatal / suspected-serious / other.

    With KABCO codes present, ``n_ksi`` should be Fatal + Suspected
    Serious only (FHWA convention) — not Fatal + every injury, which
    is what the legacy F/I/P scale collapses to.
    """
    df = pd.DataFrame([
        # Way 101: 1 fatal, 1 suspected serious, 1 minor injury, 1 PDO.
        {"crash_id": "a", "year": 2024, "geocoded_osm_way_id": 101,
         "severity_code": "K", "epdo": 542.0, "ped_involved": False},
        {"crash_id": "b", "year": 2025, "geocoded_osm_way_id": 101,
         "severity_code": "A", "epdo": 66.0, "ped_involved": True},
        {"crash_id": "c", "year": 2025, "geocoded_osm_way_id": 101,
         "severity_code": "B", "epdo": 11.0, "ped_involved": False},
        {"crash_id": "d", "year": 2025, "geocoded_osm_way_id": 101,
         "severity_code": "O", "epdo": 1.0, "ped_involved": False},
    ])
    out = aggregate_by_segment(df).set_index("osm_way_id")
    assert out.loc[101, "n_crashes"] == 4
    assert out.loc[101, "n_fatal"] == 1
    assert out.loc[101, "n_serious"] == 1
    # KSI should be K + A only (FHWA), not K + every injury.
    assert out.loc[101, "n_ksi"] == 2
    # n_injury should fold serious into "all-injury" for narrative use.
    assert out.loc[101, "n_injury"] == 2  # B + A
    assert out.loc[101, "n_pdo"] == 1
    assert out.loc[101, "n_ped"] == 1
    assert out.loc[101, "epdo_total"] == pytest.approx(620.0)


# ---------------------------------------------------------------------------
# KABCO normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inp,expected", [
    ("Fatal Injury (K)", "K"),
    ("Suspected Serious Injury (A)", "A"),
    ("Suspected Minor Injury (B)", "B"),
    ("Possible Injury (C)", "C"),
    ("No Apparent Injury (O)", "O"),
    ("K", "K"),
    ("a", "A"),
    ("not a real label", "O"),  # safe-low default
    (None, "O"),
    ("", "O"),
])
def test_kabco_to_code(inp, expected) -> None:
    assert kabco_to_code(inp) == expected


# ---------------------------------------------------------------------------
# Geopoint coalescing
# ---------------------------------------------------------------------------


def test_coalesce_geopoint_prefers_raw_floats() -> None:
    row = {"latitude": 40.87, "longitude": -73.985,
           "Geopoint Calculated": {"lat": 40.0, "lon": -74.0}}
    lat, lon, src = _coalesce_geopoint(row)
    assert (lat, lon, src) == (40.87, -73.985, "raw")


def test_coalesce_geopoint_falls_back_to_calculated() -> None:
    """Empty raw lat/lon should fall through to ``Geopoint Calculated``."""
    row = {"latitude": "", "longitude": "",
           "Geopoint": "",
           "Geopoint Calculated": {"lat": 40.86338719, "lon": -73.97722686}}
    lat, lon, src = _coalesce_geopoint(row)
    assert lat == pytest.approx(40.86338719)
    assert lon == pytest.approx(-73.97722686)
    assert src == "njdot_calculated"


def test_coalesce_geopoint_returns_none_when_all_missing() -> None:
    row = {"latitude": None, "longitude": None,
           "Geopoint": None, "Geopoint Calculated": None}
    assert _coalesce_geopoint(row) == (None, None, None)


def test_coalesce_geopoint_zero_coords_treated_as_missing() -> None:
    """Some NJDOT rows have ``0.0`` placeholders — drop them."""
    row = {"latitude": 0.0, "longitude": 0.0,
           "Geopoint": {"lat": 0, "lon": 0},
           "Geopoint Calculated": None}
    assert _coalesce_geopoint(row) == (None, None, None)


# ---------------------------------------------------------------------------
# Dashboard JSON parser
# ---------------------------------------------------------------------------


def _dashboard_row(*, id_cr=1000, year=2025, sev="No Apparent Injury (O)",
                   street="Broad Avenue", cross="Fort Lee Road",
                   lat=None, lon=None,
                   calc_lat=None, calc_lon=None,
                   road_system="Municipal", ped="No",
                   killed=0, injured=0, vehicles=2,
                   shsp=None) -> dict:
    """Construct a synthetic dashboard row for parser tests."""
    return {
        "id_cr": id_cr,
        "casenumber": f"25-{id_cr:05d}",
        "dateofcrash": f"{year}-06-15",
        "Time of Crash": "13:38:00",
        "Day of Week": "Friday",
        "year": year,
        "County": "Bergen",
        "Municipality": "Leonia Boro",
        "Severity Rating (5)": sev,
        "fatalcrashind": "Y" if killed else "N",
        "fatalitycount": killed,
        "injurycount": injured,
        "pedestrianfatalitycount": 0,
        "pedestrianinjurycount": 1 if ped == "Yes" else 0,
        "vehiclecount": vehicles,
        "Crash Type": "Same Direction (Side Swipe)",
        "First Harmful Event": "MV in Transport",
        "Pedestrian Involved": ped,
        "Bicyclist Involved": "No",
        "Alcohol Involved": "No",
        "Hazmat Involved": "No",
        "At Intersection": "Yes",
        "streetname": street,
        "intersectstreetname": cross,
        "streetsri": "02451653__",
        "milepost": 1.46,
        "distancefromintersection": 93,
        "Direction From Intersection": "South",
        "Light Condition": "Daylight",
        "Weather Condition": "Clear",
        "Surface Condition": "Dry",
        "Road System": road_system,
        "Road Surface Type": "Blacktop",
        "Functional Class_first": "Minor Arterial",
        "Urban or Rural_first": "Urban",
        "speedlimit": 25,
        "intersectionspeedlimit": 25,
        "latitude": lat,
        "longitude": lon,
        "Geopoint": ({"lat": lat, "lon": lon}
                     if lat is not None else ""),
        "Geopoint Calculated": ({"lat": calc_lat, "lon": calc_lon}
                                if calc_lat is not None else ""),
        "Unable to Geocode Crash": "No",
        "SegmentID_first_2": 60291,
        "__Nu_Segment_AADT__": 8575,
        "__Nu_Segment_CRC__": 0.0001,
        "__Nu_Segment_CPMC__": 0.27,
        "__Nu_Window_CRC__": 0.001,
        "__Nu_Intersection_TEV__": 26206,
        "SHSP Emphasis Areas": shsp or [],
    }


def test_parse_dashboard_json_basic_columns(tmp_path: Path) -> None:
    """Dashboard rows should land with the expected aliased columns."""
    payload = {"data": {"rows": [
        _dashboard_row(id_cr=1, sev="Fatal Injury (K)", killed=1,
                       calc_lat=40.86, calc_lon=-73.98),
        _dashboard_row(id_cr=2, sev="Suspected Serious Injury (A)",
                       injured=1, lat=40.87, lon=-73.99),
    ], "totalRows": 2}}
    p = tmp_path / "leonia.json"
    p.write_text(json.dumps(payload))

    df = parse_dashboard_json(p)
    assert len(df) == 2
    # Severity → KABCO + EPDO.
    assert df["severity_code"].tolist() == ["K", "A"]
    assert df["severity_label"].tolist() == [
        "Fatal", "Suspected Serious Injury",
    ]
    assert df["epdo"].tolist() == [542.0, 66.0]
    # Coords: row 0 gets calculated fallback, row 1 gets raw.
    assert df["geocoded_method"].tolist() == ["njdot_calculated", "raw"]
    assert df["geocoded_lat"].tolist() == pytest.approx([40.86, 40.87])
    # crash_id is unique with the dashboard prefix.
    assert df["crash_id"].is_unique
    assert all(cid.startswith(str(y) + "_DASH_")
               for cid, y in zip(df["crash_id"], df["year"]))
    # Booleans.
    assert df["at_intersection"].all()
    assert not df["alcohol"].any()
    # Source provenance.
    assert (df["data_source"] == "njdot_dashboard").all()


def test_parse_dashboard_json_drop_state_system(tmp_path: Path) -> None:
    """``drop_state_system=True`` should hide Interstate / NJ Tpk crashes."""
    payload = {"data": {"rows": [
        _dashboard_row(id_cr=1, road_system="Municipal",
                       lat=40.86, lon=-73.98),
        _dashboard_row(id_cr=2, road_system="State Authority",
                       lat=40.87, lon=-73.99),
        _dashboard_row(id_cr=3, road_system="NJDOT State Highway",
                       lat=40.88, lon=-73.97),
        _dashboard_row(id_cr=4, road_system="County",
                       lat=40.85, lon=-74.00),
    ], "totalRows": 4}}
    p = tmp_path / "leonia.json"
    p.write_text(json.dumps(payload))

    df_all = parse_dashboard_json(p)
    assert len(df_all) == 4

    df_local = parse_dashboard_json(p, drop_state_system=True)
    assert len(df_local) == 2
    assert df_local["road_system"].tolist() == ["Municipal", "County"]


def test_parse_dashboard_json_handles_empty_input(tmp_path: Path) -> None:
    """Empty input should produce an empty DataFrame, not raise."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"data": {"rows": [], "totalRows": 0}}))
    assert parse_dashboard_json(p).empty


def test_parse_dashboard_json_accepts_dict_payload() -> None:
    """The parser should also accept a pre-loaded dict, not just a path."""
    payload = {"data": {"rows": [_dashboard_row(lat=40.86, lon=-73.98)],
                        "totalRows": 1}}
    df = parse_dashboard_json(payload)
    assert len(df) == 1


def test_parse_dashboard_json_shsp_areas_serialised(tmp_path: Path) -> None:
    """SHSP emphasis areas should land as a comma-joined string."""
    payload = {"data": {"rows": [
        _dashboard_row(lat=40.86, lon=-73.98,
                       shsp=["Distracted Driving Related",
                             "Older Driver Involved"])
    ], "totalRows": 1}}
    p = tmp_path / "leonia.json"
    p.write_text(json.dumps(payload))

    df = parse_dashboard_json(p)
    assert df.iloc[0]["shsp_emphasis_areas"] == \
        "Distracted Driving Related, Older Driver Involved"


# ---------------------------------------------------------------------------
# Crash-only Leaflet map (visualizations)
# ---------------------------------------------------------------------------


def test_build_crash_map_writes_self_contained_html(tmp_path: Path) -> None:
    """The crash map should embed the crash payload + skeleton as JSON.

    No external CSV/parquet, no Folium plugin layer — just a single
    HTML file that opens cleanly in any browser.
    """
    from leonia_traffic.sumo.visualizations import build_crash_map

    df = pd.DataFrame([
        {"crash_id": "1", "year": 2024, "severity_code": "K",
         "epdo": 542.0, "ped_involved": False,
         "geocoded_lat": 40.862, "geocoded_lon": -73.988,
         "crash_location": "Broad Avenue", "cross_street": "Fort Lee Road",
         "crash_date": pd.Timestamp("2024-11-27"),
         "severity_label": "Fatal", "road_system": "Municipal"},
        {"crash_id": "2", "year": 2025, "severity_code": "A",
         "epdo": 66.0, "ped_involved": True,
         "geocoded_lat": 40.864, "geocoded_lon": -73.985,
         "crash_location": "Grand Avenue", "cross_street": "Park Place",
         "crash_date": pd.Timestamp("2025-04-15"),
         "severity_label": "Suspected Serious Injury",
         "road_system": "County"},
        {"crash_id": "3", "year": 2024, "severity_code": "O",
         "epdo": 1.0, "ped_involved": False,
         "geocoded_lat": 40.871, "geocoded_lon": -73.978,
         "crash_location": "I-95", "cross_street": "",
         "crash_date": pd.Timestamp("2024-08-01"),
         "severity_label": "No Apparent Injury",
         "road_system": "Interstate"},
    ])
    out = tmp_path / "crashes.html"
    build_crash_map(df, out)
    assert out.exists()
    html = out.read_text()

    # Self-contained: no external file refs.
    assert "<iframe" not in html
    assert "data/" not in html
    # Filter UI is present.
    assert 'id="year-lo"' in html and 'id="year-hi"' in html
    for sev in ("K", "A", "B", "C", "O"):
        assert f'data-sev="{sev}"' in html
    assert 'id="ped-only"' in html
    # The Interstate row should have been dropped by the default
    # ``drop_state_system=True``.
    import re as _re
    m = _re.search(r"const CRASHES = (\[.*?\]);", html, _re.DOTALL)
    assert m is not None
    crashes = json.loads(m.group(1))
    assert len(crashes) == 2
    sev_codes = {c["sev_code"] for c in crashes}
    assert sev_codes == {"K", "A"}
    # Year range comes from the actual data.
    assert 'min="2024"' in html
    assert 'max="2025"' in html


def test_build_crash_map_handles_empty_input(tmp_path: Path) -> None:
    """An empty / missing crash frame shouldn't blow up."""
    from leonia_traffic.sumo.visualizations import build_crash_map

    out = tmp_path / "crashes.html"
    build_crash_map(None, out)
    assert out.exists()
    assert "No crash data available" in out.read_text()

    out2 = tmp_path / "crashes2.html"
    build_crash_map(pd.DataFrame(), out2)
    assert "No crash data available" in out2.read_text()


def test_build_crash_map_can_keep_state_system(tmp_path: Path) -> None:
    """``drop_state_system=False`` should leave Interstate rows in."""
    from leonia_traffic.sumo.visualizations import build_crash_map

    df = pd.DataFrame([
        {"crash_id": "1", "year": 2024, "severity_code": "O",
         "epdo": 1.0, "ped_involved": False,
         "geocoded_lat": 40.87, "geocoded_lon": -73.98,
         "crash_location": "I-95", "cross_street": "",
         "crash_date": pd.NaT, "severity_label": "No Apparent Injury",
         "road_system": "Interstate"},
    ])
    out = tmp_path / "crashes.html"
    build_crash_map(df, out, drop_state_system=False)
    import re as _re
    m = _re.search(r"const CRASHES = (\[.*?\]);", out.read_text(), _re.DOTALL)
    assert m is not None
    assert len(json.loads(m.group(1))) == 1


# ---------------------------------------------------------------------------
# Stakeholder safety panel — local-streets-only filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,is_state", [
    ("New Jersey Turnpike", True),
    ("New Jersey Turnpike (Express Lanes)", True),
    ("I-95", True),
    ("I 95", True),
    ("I-95 Express", True),
    ("motorway_link", True),
    ("Broad Avenue", False),
    ("Grand Avenue", False),     # NJ-93 underneath — kept as local.
    ("NJ 93", False),            # also kept (local levers via signals).
    ("Fort Lee Road", False),
    ("Ridgeland Terrace", False),
    ("", False),
    (None, False),
])
def test_is_state_system_street(name, is_state) -> None:
    from leonia_traffic.sumo.visualizations import _is_state_system_street
    assert _is_state_system_street(name) is is_state


def test_safety_panel_filters_state_system_segments() -> None:
    """The safety table should drop NJ Turnpike / I-95 OSM ways and
    show the top *local* streets only.

    Headline totals must also reflect the local-streets subset so
    the table sums match the headline (no whiplash for readers).
    """
    from leonia_traffic.sumo.visualizations import _safety_panel

    seg = pd.DataFrame([
        # 4 NJ Turnpike / I-95 segments — should be excluded.
        {"street_name": "New Jersey Turnpike", "osm_way_id": 1,
         "n_crashes": 100, "n_fatal": 0, "n_serious": 0, "n_injury": 20,
         "n_pdo": 80, "n_ksi": 0, "n_ped": 0, "epdo_total": 300.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "New Jersey Turnpike (Express Lanes)",
         "osm_way_id": 2,
         "n_crashes": 80, "n_fatal": 0, "n_serious": 0, "n_injury": 15,
         "n_pdo": 65, "n_ksi": 0, "n_ped": 0, "epdo_total": 230.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "I-95", "osm_way_id": 3,
         "n_crashes": 60, "n_fatal": 0, "n_serious": 0, "n_injury": 10,
         "n_pdo": 50, "n_ksi": 0, "n_ped": 0, "epdo_total": 160.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "motorway_link", "osm_way_id": 4,
         "n_crashes": 40, "n_fatal": 0, "n_serious": 0, "n_injury": 5,
         "n_pdo": 35, "n_ksi": 0, "n_ped": 0, "epdo_total": 90.0,
         "first_year": 2019, "last_year": 2026},
        # 3 local streets — should appear in the table.
        {"street_name": "Broad Avenue", "osm_way_id": 5,
         "n_crashes": 45, "n_fatal": 2, "n_serious": 0, "n_injury": 7,
         "n_pdo": 36, "n_ksi": 2, "n_ped": 2, "epdo_total": 1197.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "Grand Avenue", "osm_way_id": 6,
         "n_crashes": 15, "n_fatal": 1, "n_serious": 1, "n_injury": 4,
         "n_pdo": 9, "n_ksi": 2, "n_ped": 2, "epdo_total": 661.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "Fort Lee Road", "osm_way_id": 7,
         "n_crashes": 39, "n_fatal": 0, "n_serious": 2, "n_injury": 13,
         "n_pdo": 24, "n_ksi": 2, "n_ped": 2, "epdo_total": 299.0,
         "first_year": 2019, "last_year": 2026},
    ])
    crash_pts = pd.DataFrame([{"year": 2019}, {"year": 2026}])
    html = _safety_panel(None, seg, crash_pts)

    # The state-system segments must not appear as table rows. The
    # footnote does mention "NJ Turnpike / I-95" by design, so we
    # assert against the table body specifically.
    import re as _re
    table_match = _re.search(r"<tbody>(.*?)</tbody>", html, _re.DOTALL)
    assert table_match is not None
    table_body = table_match.group(1)
    assert "New Jersey Turnpike" not in table_body
    assert "I-95" not in table_body
    assert "motorway_link" not in table_body
    # Local streets are present in the table body.
    assert "Broad Avenue" in table_body
    assert "Grand Avenue" in table_body
    assert "Fort Lee Road" in table_body
    # Headline totals = local-streets-only sum (45+15+39=99).
    assert "<b>99</b>" in html      # n_crashes
    assert "<b>3</b> fatal" in html  # 2+1+0
    assert "<b>3</b> suspected serious" in html  # 0+1+2
    assert "<b>6</b> KSI" in html    # 2+2+2
    assert "<b>6</b> pedestrian-involved" in html  # 2+2+2
    # Footnote disclosure of how many segments were filtered.
    assert "4 segments" in html
    assert "NJ Turnpike / I-95" in html
    # Title has the "local streets" framing + the year range.
    assert "Leonia local streets" in html
    assert "2019–2026" in html


def test_safety_panel_top_n_caps_at_default() -> None:
    """The default ``top_n`` is 50 — long enough to surface every
    Leonia street with measurable crash activity (the borough has
    on the order of ~40 distinct named local streets in the crash
    parquet), short enough to fit in a council-deck table without
    overwhelming the reader.
    """
    from leonia_traffic.sumo.visualizations import _safety_panel

    seg = pd.DataFrame([
        {"street_name": f"Local Street {i:02d}", "osm_way_id": 1000 + i,
         "n_crashes": 100 - i, "n_fatal": 0, "n_serious": 0,
         "n_injury": 5, "n_pdo": 95 - i, "n_ksi": 0, "n_ped": 0,
         "epdo_total": float(2000 - 100 * i),
         "first_year": 2020, "last_year": 2025}
        for i in range(60)  # 60 candidates, all local
    ])
    html = _safety_panel(None, seg)

    # Top 50 streets (00..49) make the table; 50..59 are dropped.
    assert "Local Street 00" in html
    assert "Local Street 49" in html
    assert "Local Street 50" not in html
    assert "Local Street 59" not in html


# ---------------------------------------------------------------------------
# Street-name canonicalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [
    # OSM short / full forms.
    ("Broad Avenue",         "BROAD AVE"),
    ("Broad Avenue",         "Broad Ave"),
    ("Broad Avenue",         "BROAD AVENUE"),
    # Compound NJDOT free-text.
    ("Broad Avenue",         "BROAD AVE / DANA PL"),
    # Parenthetical disambiguators.
    ("Fort Lee Road",        "Fort Lee Road (BERGEN COUNTY 56 3)"),
    # Suffix abbreviations all expand the same way.
    ("Park Place",           "PARK PL"),
    ("Linden Terrace",       "LINDEN TER"),
    ("Glenwood Avenue",      "GLENWOOD AV"),
    ("Christie Street",      "Christie St"),
    # NJDOT legacy ``**`` markers.
    ("Ridgeland Terrace",    "RIDGELAND TER **"),
    # Whitespace collapse.
    ("Glenwood   Avenue",    "Glenwood Avenue"),
])
def test_canonical_street_key_collisions(a, b) -> None:
    from leonia_traffic.sumo.visualizations import _canonical_street_key
    assert _canonical_street_key(a) == _canonical_street_key(b)
    assert _canonical_street_key(a) != ""


@pytest.mark.parametrize("a,b", [
    ("Broad Avenue",  "Grand Avenue"),
    ("Park Place",    "Park Avenue"),
    ("Fort Lee Road", "Fort Lee Avenue"),
])
def test_canonical_street_key_distinct_streets(a, b) -> None:
    from leonia_traffic.sumo.visualizations import _canonical_street_key
    assert _canonical_street_key(a) != _canonical_street_key(b)


def test_canonical_street_key_handles_blank_inputs() -> None:
    from leonia_traffic.sumo.visualizations import _canonical_street_key
    assert _canonical_street_key(None) == ""
    assert _canonical_street_key("") == ""
    assert _canonical_street_key("   ") == ""
    assert _canonical_street_key(123) == ""


def test_safety_panel_aggregates_osm_segments_per_street() -> None:
    """Multiple OSM ``way`` rows for the same street should collapse.

    OSM splits each road into multiple ``way`` records broken at
    intersections. The council table needs one row per street, with
    the column totals summed across all underlying segments and a
    ``(N segments)`` annotation so readers can tell whether a high
    count is one hot intersection or a corridor-wide pattern.
    """
    from leonia_traffic.sumo.visualizations import _safety_panel

    seg = pd.DataFrame([
        # Three OSM-way fragments of "Broad Avenue".
        {"street_name": "Broad Avenue",       "osm_way_id": 100,
         "n_crashes": 45, "n_fatal": 2, "n_serious": 0, "n_injury": 7,
         "n_pdo": 36, "n_ksi": 2, "n_ped": 2, "epdo_total": 1197.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "BROAD AVE / DANA PL", "osm_way_id": 101,
         "n_crashes": 30, "n_fatal": 0, "n_serious": 0, "n_injury": 5,
         "n_pdo": 25, "n_ksi": 0, "n_ped": 1, "epdo_total": 195.0,
         "first_year": 2020, "last_year": 2025},
        {"street_name": "BROAD AVE",          "osm_way_id": 102,
         "n_crashes": 25, "n_fatal": 0, "n_serious": 0, "n_injury": 4,
         "n_pdo": 21, "n_ksi": 0, "n_ped": 3, "epdo_total": 140.0,
         "first_year": 2021, "last_year": 2024},
        # One unrelated street.
        {"street_name": "Grand Avenue",       "osm_way_id": 200,
         "n_crashes": 15, "n_fatal": 1, "n_serious": 1, "n_injury": 4,
         "n_pdo": 9, "n_ksi": 2, "n_ped": 2, "epdo_total": 661.0,
         "first_year": 2019, "last_year": 2026},
    ])
    html = _safety_panel(None, seg)

    import re as _re
    table_body = _re.search(r"<tbody>(.*?)</tbody>", html,
                            _re.DOTALL).group(1)
    # Split into individual rows so a regex doesn't accidentally
    # match across the `</tr><tr>` boundary.
    all_rows = _re.findall(
        r"<tr>(?:(?!</tr>).)*</tr>", table_body, _re.DOTALL,
    )
    broad_rows = [r for r in all_rows if "Broad Avenue" in r]
    grand_rows = [r for r in all_rows if "Grand Avenue" in r]

    # Broad Avenue should appear exactly once as a table row.
    assert len(broad_rows) == 1
    broad_row = broad_rows[0]
    # Aggregated counts: 45+30+25=100, fatal 2, ksi 2, ped 6,
    # epdo 1197+195+140=1532.
    cells = _re.findall(r"<td[^>]*>([^<]*)</td>", broad_row)
    assert "100" in cells          # total crashes
    assert "2" in cells            # fatal
    assert "6" in cells            # ped
    assert "1532" in cells         # EPDO
    # Segments annotation present.
    assert "(3 segments)" in broad_row
    # The "BROAD AVE / DANA PL" / "BROAD AVE" spellings shouldn't
    # appear as separate rows or anywhere in the table body.
    assert "DANA PL" not in table_body
    # Grand Avenue stays a separate row…
    assert len(grand_rows) == 1
    # …and has no "N segments" annotation since it's a single way.
    assert "segments" not in grand_rows[0]


def test_safety_panel_default_sort_is_crashes() -> None:
    """The table should be sorted descending by raw crash count.

    Residents and council members read the table top-down; the most
    salient column is "how many crashes happen on my street". EPDO
    is the most decision-relevant column for capital prioritisation,
    but it's a derived index that's harder to explain. So we keep
    EPDO as a column but sort by crash count.
    """
    from leonia_traffic.sumo.visualizations import _safety_panel
    import re as _re

    # Construct a table where crash-count ranking and EPDO ranking
    # disagree. By crashes: B (50) > A (40) > C (30). By EPDO:
    # A (5000) > C (300) > B (200). Default sort must follow
    # crashes, not EPDO.
    seg = pd.DataFrame([
        {"street_name": "A St", "osm_way_id": 1,
         "n_crashes": 40, "n_fatal": 1, "n_serious": 1, "n_injury": 5,
         "n_pdo": 33, "n_ksi": 2, "n_ped": 2, "epdo_total": 5000.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "B St", "osm_way_id": 2,
         "n_crashes": 50, "n_fatal": 0, "n_serious": 0, "n_injury": 2,
         "n_pdo": 48, "n_ksi": 0, "n_ped": 0, "epdo_total": 200.0,
         "first_year": 2019, "last_year": 2026},
        {"street_name": "C St", "osm_way_id": 3,
         "n_crashes": 30, "n_fatal": 0, "n_serious": 0, "n_injury": 3,
         "n_pdo": 27, "n_ksi": 0, "n_ped": 1, "epdo_total": 300.0,
         "first_year": 2019, "last_year": 2026},
    ])
    html = _safety_panel(None, seg)
    table_body = _re.search(r"<tbody>(.*?)</tbody>", html,
                            _re.DOTALL).group(1)
    rows = _re.findall(r"<tr>(?:(?!</tr>).)*</tr>",
                       table_body, _re.DOTALL)
    # First row should be B St (50 crashes), then A St (40),
    # then C St (30) — strictly by crashes, not EPDO.
    assert "B St" in rows[0]
    assert "A St" in rows[1]
    assert "C St" in rows[2]


def test_safety_panel_totals_use_row_level_when_available() -> None:
    """Headline totals should come from the row-level parquet.

    This keeps the safety panel's KPI strip ("980 reported crashes")
    consistent with the crash map immediately below it. Without this,
    the panel sums the segment table (which excludes crashes that
    didn't snap to a specific OSM way), which can be ~10% smaller
    than the row-level count.
    """
    from leonia_traffic.sumo.visualizations import _safety_panel

    # Three local-street segments summing to 100 crashes.
    seg = pd.DataFrame([
        {"street_name": "Local Ave", "osm_way_id": 100,
         "n_crashes": 60, "n_fatal": 1, "n_serious": 1, "n_injury": 8,
         "n_pdo": 50, "n_ksi": 2, "n_ped": 3, "epdo_total": 700.0,
         "first_year": 2020, "last_year": 2025},
        {"street_name": "Side St", "osm_way_id": 200,
         "n_crashes": 40, "n_fatal": 0, "n_serious": 0, "n_injury": 5,
         "n_pdo": 35, "n_ksi": 0, "n_ped": 1, "epdo_total": 100.0,
         "first_year": 2020, "last_year": 2025},
    ])
    # Row-level data has 110 local rows: the 100 that snapped to
    # an OSM way, plus 10 extra that geocoded by name only.
    rows = []
    for _ in range(60):
        rows.append({"crash_id": f"a{_}", "year": 2024,
                     "geocoded_osm_way_id": 100,
                     "severity_code": "O", "ped_involved": False})
    for _ in range(40):
        rows.append({"crash_id": f"b{_}", "year": 2024,
                     "geocoded_osm_way_id": 200,
                     "severity_code": "O", "ped_involved": False})
    # The 10 unsegmented local rows.
    for _ in range(10):
        rows.append({"crash_id": f"c{_}", "year": 2024,
                     "geocoded_osm_way_id": float("nan"),
                     "severity_code": "K", "ped_involved": True})
    crash_pts = pd.DataFrame(rows)

    html = _safety_panel(None, seg, crash_pts)

    # Headline reflects the row-level count (110), including the 10
    # unsegmented rows that the segment table can't see — and the
    # severities they carry (10 fatal, 10 ped). Numbers may be
    # wrapped in formatting tags, so strip them before asserting.
    import re as _re
    plain = _re.sub(r"<[^>]+>", "", html)
    plain = _re.sub(r"\s+", " ", plain)
    assert "110 reported crashes" in plain
    assert "10 fatal" in plain
    assert "10 pedestrian" in plain


def test_crash_trend_chart_includes_all_years_and_flags_partial() -> None:
    """Year-over-year chart shows every year and flags incomplete ones.

    Specifically: the *current* calendar year is always partial,
    and any year with crash count < 30 percent of median is treated
    as a data-delivery gap (a real 70% drop is implausible).
    Other low-but-plausible years (e.g. 50% of median) must NOT be
    flagged — those could represent real safety improvements.
    """
    from leonia_traffic.sumo.visualizations import _crash_trend_chart
    from datetime import datetime, timezone as _tz

    current_year = datetime.now(_tz.utc).year
    rows = []

    def add_year(yr: int, n: int) -> None:
        for i in range(n):
            rows.append({
                "crash_id": f"{yr}-{i}", "year": yr,
                "geocoded_osm_way_id": 100,
                "severity_code": "O", "ped_involved": False,
            })

    add_year(2019, 200)
    add_year(2020, 200)
    add_year(2021, 5)              # data-delivery gap
    add_year(2022, 100)            # genuinely low (50% of median)
    add_year(2023, 200)
    add_year(current_year, 50)     # partial: current year
    crash_pts = pd.DataFrame(rows)

    seg = pd.DataFrame([
        {"street_name": "Local St", "osm_way_id": 100,
         "n_crashes": len(rows), "n_fatal": 0, "n_serious": 0,
         "n_injury": 0, "n_pdo": len(rows), "n_ksi": 0, "n_ped": 0,
         "epdo_total": float(len(rows)),
         "first_year": 2019, "last_year": current_year},
    ])

    html = _crash_trend_chart(crash_pts, seg)

    # Output is a Plotly HTML snippet — the year list is present
    # as an x-axis array.
    assert "Crashes per year on Leonia local streets" in html
    # 2021 (data gap) and current_year (partial) should be flagged.
    assert "2021" in html and "2023" in html and "2019" in html
    assert "partial" in html.lower()
    # 2022 (real but low) must NOT be in the partial list. The
    # subtitle text lives inside the chart title between
    # "partial / NJDOT-underreported:" and the closing ``</sub>``
    # tag. Plotly serialises ``/`` and ``<`` as unicode escapes
    # (``\u002f``, ``\u003c``), so split on the escaped marker.
    # Restrict the search to that span so we don't accidentally
    # match the categoryarray (which lists every year for the
    # x-axis, including 2022).
    parts = html.split("partial \\u002f NJDOT-underreported:")
    assert len(parts) >= 2, "partial-year footnote not present"
    partial_section = parts[-1].split("\\u003c\\u002fsub")[0]
    assert "2021" in partial_section            # flagged as gap
    assert str(current_year) in partial_section  # flagged as current
    assert "2022" not in partial_section        # plausible drop


def test_crash_trend_chart_returns_empty_for_no_data() -> None:
    """No row-level data → no chart (empty string), so the template skips it."""
    from leonia_traffic.sumo.visualizations import _crash_trend_chart

    assert _crash_trend_chart(None) == ""
    assert _crash_trend_chart(pd.DataFrame()) == ""


def test_crash_map_filters_state_system_by_osm_way_name() -> None:
    """Crash map must use the same OSM-way filter as the safety panel.

    Otherwise, NJ-93 crashes (which NJDOT tags as "State Highway"
    but which OSM resolves to "Grand Avenue") vanish from the map
    while still appearing in the safety table — confusing readers
    with a 200-row mismatch between two side-by-side panels.
    """
    from leonia_traffic.sumo.visualizations import build_crash_map
    import json
    import re as _re

    seg = pd.DataFrame([
        # Grand Avenue is local in OSM (driveable surface street),
        # even though NJDOT classifies it as "NJDOT State Highway".
        {"street_name": "Grand Avenue", "osm_way_id": 100,
         "n_crashes": 5, "n_fatal": 0, "n_serious": 0, "n_injury": 0,
         "n_pdo": 5, "n_ksi": 0, "n_ped": 0, "epdo_total": 5.0,
         "first_year": 2024, "last_year": 2024},
        # I-95 is a true state-system way that should still be hidden.
        {"street_name": "I-95 / NJ Turnpike", "osm_way_id": 200,
         "n_crashes": 5, "n_fatal": 0, "n_serious": 0, "n_injury": 0,
         "n_pdo": 5, "n_ksi": 0, "n_ped": 0, "epdo_total": 5.0,
         "first_year": 2024, "last_year": 2024},
    ])
    crash_pts = pd.DataFrame([
        # Two crashes on Grand Avenue, NJDOT-tagged as state highway.
        # OSM-aware filter must KEEP these.
        {"crash_id": "g1", "year": 2024, "geocoded_lat": 40.86,
         "geocoded_lon": -73.98, "geocoded_osm_way_id": 100,
         "severity_code": "O", "ped_involved": False,
         "road_system": "NJDOT State Highway",
         "crash_location": "GRAND AVE", "cross_street": ""},
        {"crash_id": "g2", "year": 2024, "geocoded_lat": 40.87,
         "geocoded_lon": -73.99, "geocoded_osm_way_id": 100,
         "severity_code": "O", "ped_involved": False,
         "road_system": "NJDOT State Highway",
         "crash_location": "GRAND AVE", "cross_street": ""},
        # One crash on I-95. OSM-aware filter must DROP this.
        {"crash_id": "i1", "year": 2024, "geocoded_lat": 40.85,
         "geocoded_lon": -73.97, "geocoded_osm_way_id": 200,
         "severity_code": "O", "ped_involved": False,
         "road_system": "NJDOT State Highway",
         "crash_location": "I-95", "cross_street": ""},
    ])

    out = tmp_path_for_test() / "crashes.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_crash_map(crash_pts, out, crash_segments=seg,
                    edges_geo=None)
    html = out.read_text()

    # Pull the marker payload and confirm both Grand Avenue crashes
    # made it in but the I-95 one did not.
    m = _re.search(r"const CRASHES = (\[.*?\]);", html, _re.DOTALL)
    assert m, "marker payload not found"
    crashes = json.loads(m.group(1))
    assert len(crashes) == 2
    crash_ids = {c.get("label", "") for c in crashes}
    assert all("GRAND" in cid.upper() for cid in crash_ids)


def test_crash_map_drops_njturnpike_outliers_outside_borough() -> None:
    """NJ-Turnpike crashes geocoded far outside Leonia must be dropped.

    The NJDOT dashboard sometimes resolves
    ``crash_location = "I-95; N.J. TURNPIKE"`` to a turnpike point
    in Trenton or Newark instead of the Leonia segment. Those rows
    have null ``geocoded_osm_way_id`` (no OSM snap) and
    ``road_system = "State Authority"``. They were leaking onto
    the map as scattered dots all over New Jersey before this fix
    — the 28 outliers visible in the May 2026 user report.
    """
    from leonia_traffic.sumo.visualizations import build_crash_map
    import json
    import re as _re

    seg = pd.DataFrame([
        {"street_name": "Broad Avenue", "osm_way_id": 100,
         "n_crashes": 5, "n_fatal": 0, "n_serious": 0, "n_injury": 0,
         "n_pdo": 5, "n_ksi": 0, "n_ped": 0, "epdo_total": 5.0,
         "first_year": 2024, "last_year": 2024},
    ])
    crash_pts = pd.DataFrame([
        # Inside the borough — must be KEPT.
        {"crash_id": "in1", "year": 2024, "geocoded_lat": 40.866,
         "geocoded_lon": -73.985, "geocoded_osm_way_id": 100,
         "severity_code": "O", "ped_involved": False,
         "road_system": "Municipal", "crash_location": "BROAD AVE",
         "cross_street": ""},
        # Outside bbox + state-system + no OSM snap — outlier from
        # the real NJDOT data. Must be DROPPED.
        {"crash_id": "tp1", "year": 2024, "geocoded_lat": 39.679,
         "geocoded_lon": -75.488, "geocoded_osm_way_id": float("nan"),
         "severity_code": "O", "ped_involved": False,
         "road_system": "State Authority",
         "crash_location": "I-95; N.J. TURNPIKE", "cross_street": ""},
        # Inside the bbox but no OSM snap and state_system — should
        # also be DROPPED (the road_system fallback kicks in when
        # OSM has nothing to say).
        {"crash_id": "tp2", "year": 2024, "geocoded_lat": 40.866,
         "geocoded_lon": -73.97, "geocoded_osm_way_id": float("nan"),
         "severity_code": "O", "ped_involved": False,
         "road_system": "State Authority",
         "crash_location": "I-95; N.J. TURNPIKE", "cross_street": ""},
    ])

    out = tmp_path_for_test() / "crashes.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_crash_map(crash_pts, out, crash_segments=seg, edges_geo=None)
    html = out.read_text()

    m = _re.search(r"const CRASHES = (\[.*?\]);", html, _re.DOTALL)
    assert m, "marker payload not found"
    crashes = json.loads(m.group(1))
    assert len(crashes) == 1
    assert "BROAD" in crashes[0]["label"].upper()


def test_filter_drops_state_system_rows_outside_polygon() -> None:
    """State-system crashes snapped to a friendly local-street OSM
    way must still be dropped if their lat/lon sits outside the
    Leonia polygon.

    Reproduces the May 2026 Fort Lee / Ridgefield Park / Englewood
    Cliffs leak: NJDOT classifies the row as ``State Authority``
    with ``crash_location = "I-95; N.J. TURNPIKE"``, but its
    ``geocoded_osm_way_id`` snaps to a local OSM way name (e.g.
    "Schlosser Street") because the spatial join's nearest-edge
    rule picked a Leonia centerline that happens to be 200m–1.8km
    from the actual lat/lon. The OSM-name filter alone keeps these
    (the snapped name is friendly), so we need the polygon-precise
    filter to catch them.
    """
    from leonia_traffic.sumo.visualizations import (
        _filter_crash_rows_to_borough,
    )

    # Segment table claims these OSM ways are friendly local
    # streets — the same name-mapping the safety panel sees.
    seg = pd.DataFrame([
        {"street_name": "Schlosser Street", "osm_way_id": 420537526,
         "n_crashes": 1, "n_fatal": 0, "n_serious": 0,
         "n_injury": 0, "n_pdo": 1, "n_ksi": 0, "n_ped": 0,
         "epdo_total": 1.0, "first_year": 2019, "last_year": 2019},
        {"street_name": "Christopher Columbus Highway",
         "osm_way_id": 759907485,
         "n_crashes": 1, "n_fatal": 0, "n_serious": 0,
         "n_injury": 0, "n_pdo": 1, "n_ksi": 0, "n_ped": 0,
         "epdo_total": 1.0, "first_year": 2022, "last_year": 2022},
        {"street_name": "Broad Avenue", "osm_way_id": 583818803,
         "n_crashes": 5, "n_fatal": 0, "n_serious": 0,
         "n_injury": 0, "n_pdo": 5, "n_ksi": 0, "n_ped": 0,
         "epdo_total": 5.0, "first_year": 2024, "last_year": 2024},
    ])
    crash_pts = pd.DataFrame([
        # Fort Lee centre — lon=-73.97, ~1.2 km from Leonia.
        {"crash_id": "leak_ftlee", "year": 2019,
         "geocoded_lat": 40.8509, "geocoded_lon": -73.9701,
         "geocoded_osm_way_id": 420537526,
         "severity_code": "O", "ped_involved": False,
         "road_system": "State Authority",
         "crash_location": "I-95; N.J. TURNPIKE", "cross_street": ""},
        # Ridgefield Park I-95 — lon=-74.025, ~1.85 km from Leonia.
        {"crash_id": "leak_ridgefield", "year": 2022,
         "geocoded_lat": 40.8666, "geocoded_lon": -74.0249,
         "geocoded_osm_way_id": 759907485,
         "severity_code": "O", "ped_involved": False,
         "road_system": "State Authority",
         "crash_location": "I-95; N.J. TURNPIKE", "cross_street": ""},
        # Englewood Cliffs — ~340 m north of Leonia.
        {"crash_id": "leak_eng", "year": 2021,
         "geocoded_lat": 40.8753, "geocoded_lon": -73.9781,
         "geocoded_osm_way_id": 583818803,
         "severity_code": "O", "ped_involved": False,
         "road_system": "State Authority",
         "crash_location": "I-95; N.J. TURNPIKE", "cross_street": ""},
        # Inside Leonia — must be KEPT despite road_system =
        # State Authority (e.g. an actual on-borough Turnpike
        # crash near the GW Bridge approach, snapped correctly).
        {"crash_id": "keep_in", "year": 2024,
         "geocoded_lat": 40.866, "geocoded_lon": -73.985,
         "geocoded_osm_way_id": 583818803,
         "severity_code": "O", "ped_involved": False,
         "road_system": "Municipal",  # local — never gets polygon-checked
         "crash_location": "BROAD AVE", "cross_street": ""},
    ])

    kept = _filter_crash_rows_to_borough(
        crash_pts, drop_state_system=True, crash_segments=seg,
    )
    # Only the local-jurisdiction inside-borough row survives.
    assert list(kept["crash_id"]) == ["keep_in"]


def test_filter_crash_rows_to_borough_uses_bbox() -> None:
    """The bbox filter alone should drop far-flung points.

    Even with ``drop_state_system=False`` (no road-system filter),
    a crash whose lat/lon lies outside the Leonia bbox must be
    rejected — that's the third defense layer.
    """
    from leonia_traffic.sumo.visualizations import (
        _filter_crash_rows_to_borough,
    )

    crash_pts = pd.DataFrame([
        # Inside.
        {"crash_id": "ok",   "geocoded_lat": 40.870, "geocoded_lon": -73.980,
         "geocoded_osm_way_id": 100, "severity_code": "O",
         "ped_involved": False, "road_system": "Municipal"},
        # Far outside (Wilmington-ish).
        {"crash_id": "far",  "geocoded_lat": 39.679, "geocoded_lon": -75.488,
         "geocoded_osm_way_id": 100, "severity_code": "O",
         "ped_involved": False, "road_system": "Municipal"},
        # Just outside the bbox to the south.
        {"crash_id": "edge", "geocoded_lat": 40.50,  "geocoded_lon": -73.98,
         "geocoded_osm_way_id": 100, "severity_code": "O",
         "ped_involved": False, "road_system": "Municipal"},
    ])

    kept = _filter_crash_rows_to_borough(crash_pts, drop_state_system=False)
    assert list(kept["crash_id"]) == ["ok"]


def tmp_path_for_test() -> Path:
    """Local helper: create a unique tempdir under tests/ for one test."""
    import tempfile
    return Path(tempfile.mkdtemp(prefix="crash_map_test_"))
