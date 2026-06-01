"""Tests for the Bridge-destination OD loader."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.data import bridge_od_loader as bol


def test_discover_bridge_od_finds_real_export():
    paths = bol.discover_bridge_od()
    assert paths is not None, "expected bridge OD export to be present on disk"
    assert paths.od_all.exists()
    # All six attribute CSVs should be found.
    assert paths.trip_purpose is not None
    assert paths.equity is not None
    assert paths.household is not None
    assert paths.education_income is not None
    assert paths.employment is not None
    assert paths.trip_stats is not None


def test_discover_bridge_od_returns_none_for_missing(tmp_path):
    assert bol.discover_bridge_od(tmp_path) is None


@pytest.mark.parametrize(
    "raw,expected_label,expected_id",
    [
        ("Fort Lee Road / 590576", "Fort Lee Road", 590576),
        ("Broad Avenue / 10030557", "Broad Avenue", 10030557),
        ("George Washington Bridge (lower level) / 590410",
         "George Washington Bridge (lower level)", 590410),
        ("No Slash Here", "No Slash Here", None),
        ("", "", None),
    ],
)
def test_parse_bridge_zone_name(raw, expected_label, expected_id):
    label, osm_id = bol.parse_bridge_zone_name(raw)
    assert label == expected_label
    assert osm_id == expected_id


@pytest.mark.parametrize(
    "raw,expected_code,expected_label",
    [
        ("2: Tuesday (Tu-Tu)", 2, "Tuesday (Tu-Tu)"),
        ("0: All Days (M-Su)", 0, "All Days (M-Su)"),
        ("2: Peak AM (6am-10am)", 2, "Peak AM (6am-10am)"),
        ("Free text", None, "Free text"),
        (None, None, None),
    ],
)
def test_parse_coded_value(raw, expected_code, expected_label):
    code, label = bol.parse_coded_value(raw)
    assert code == expected_code
    assert label == expected_label


def test_load_bridge_od_basic_shape():
    df = bol.load_bridge_od()
    assert not df.empty
    assert {
        "origin_zone", "destination_zone",
        "day_type_code", "day_type_label",
        "day_part_code", "day_part_label",
        "od_volume", "avg_travel_time_sec",
        "origin_osm_way_id", "destination_osm_way_id",
    }.issubset(df.columns)

    # Day Type codes 0..7. Day Part codes 0..24 in the
    # 2036064_Destinations export (was 0..5 in the legacy
    # bridge_destination 5-window export).
    assert set(df["day_type_code"].dropna().unique()).issubset(set(range(0, 8)))
    assert set(df["day_part_code"].dropna().unique()).issubset(set(range(0, 25)))

    # Volumes must be non-negative.
    assert (df["od_volume"].dropna() >= 0).all()


def test_load_bridge_od_headline_fort_lee_peak_am():
    """Sanity check the headline finding on the new 24-hour data.

    Fort Lee Road / 590576 should still be a major Peak-AM origin,
    but in the 24-hour schema "Peak AM" is the **sum** of hourly
    codes 7..10 (6am-10am) rather than a single 4-hour bucket. So
    we sum across those 4 hours per day-type and confirm at least
    one Mon-Thu day's Peak-AM total is >= 500 trips.
    """
    df = bol.load_bridge_od()
    fl = df[(df["origin_osm_way_id"] == 590576)
            & df["day_part_code"].isin([7, 8, 9, 10])  # Peak AM range
            & df["day_type_code"].isin([1, 2, 3, 4])]
    assert not fl.empty
    # Sum hourly slices into one Peak-AM total per day, then check
    # the busiest weekday clears the headline 500-trip threshold.
    daily_peak_am = fl.groupby("day_type_code")["od_volume"].sum()
    assert daily_peak_am.max() >= 500


def test_load_bridge_attributes_columns_prefixed():
    df = bol.load_bridge_attributes()
    assert not df.empty
    # At least one prefixed column from each bucket.
    prefixes = {c.split("::", 1)[0] for c in df.columns if "::" in c}
    assert {"trip_purpose", "equity", "household", "income", "employment", "trip_stats"}.issubset(prefixes)
    # Join keys still present (raw and renamed forms).
    assert "origin_zone" in df.columns
    assert "destination_zone" in df.columns


def test_load_bridge_attributes_include_filter():
    df = bol.load_bridge_attributes(include=("trip_purpose", "equity"))
    prefixes = {c.split("::", 1)[0] for c in df.columns if "::" in c}
    assert prefixes == {"trip_purpose", "equity"}


def test_load_bridge_zone_shapes_line_and_polygon():
    line = bol.load_bridge_zone_shapes(kind="line")
    assert not line.empty
    assert "zone_role" in line.columns
    assert set(line["zone_role"].unique()).issubset({"origin", "destination", "unknown"})
    # Polygon variant might not exist for every export, but the call
    # must not raise; if shapefiles are missing it returns an empty frame.
    poly = bol.load_bridge_zone_shapes(kind="polygon")
    assert isinstance(poly, type(line))
