"""Tests for borough-of-Leonia jurisdictional filtering."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from leonia_traffic.analysis.jurisdiction import (
    annotate_in_leonia,
    filter_segments_to_leonia,
    is_under_borough_jurisdiction,
)


# A small square polygon standing in for Leonia in unit tests.
TEST_BOROUGH = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])


def _make_zones() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "zone_name": ["A", "B", "C", "D", "E"],
            "osm_way_id": [1, 2, 3, 4, 5],
            "osm_name": [
                "Fort Lee Road",          # county arterial → exclude
                "George Washington Bridge Plaza",  # outside polygon
                "Some Local Street",
                "New Jersey Turnpike",    # state road → exclude
                "Main Street",            # local signing of Fort Lee Rd (CR 9) → exclude
            ],
            "geometry": [
                LineString([(2, 2), (3, 3)]),   # inside
                LineString([(20, 20), (21, 21)]),  # outside
                LineString([(5, 5), (6, 6)]),   # inside
                LineString([(2, 2), (4, 4)]),   # inside spatially, state road
                LineString([(3, 3), (5, 5)]),   # inside
            ],
        },
        crs="EPSG:4326",
    )


def _make_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_name": ["A", "B", "C", "D", "E"],
            "osm_way_id": [1, 2, 3, 4, 5],
            "osm_name": [
                "Fort Lee Road",
                "George Washington Bridge Plaza",
                "Some Local Street",
                "New Jersey Turnpike",
                "Main Street",
            ],
            "road_class": ["Secondary", "Motorway", "Tertiary", "Motorway", "Tertiary"],
            "worst_tti": [2.1, 4.0, 1.5, 3.0, 1.8],
        }
    )


def test_is_under_borough_jurisdiction_excludes_state_roads():
    assert is_under_borough_jurisdiction("Some Local Street", "Tertiary")
    # County arterials: Broad/Grand/Fort Lee/Main Street must be excluded.
    assert not is_under_borough_jurisdiction("Fort Lee Road", "Secondary")
    assert not is_under_borough_jurisdiction("Main Street", "Tertiary")
    assert not is_under_borough_jurisdiction("Broad Avenue", "Primary")
    assert not is_under_borough_jurisdiction("Grand Ave", "Primary")
    # State/federal facilities
    assert not is_under_borough_jurisdiction("New Jersey Turnpike", "Motorway")
    assert not is_under_borough_jurisdiction(
        "George Washington Bridge Plaza", "Motorway"
    )
    assert not is_under_borough_jurisdiction("US 1;US 9;US 46", "Motorway")
    # motorway class alone is enough to disqualify even if the name is generic
    assert not is_under_borough_jurisdiction("motorway_link", "On/Off Ramp")


def test_filter_segments_to_leonia_drops_outside_state_and_county_roads():
    summary = _make_summary()
    zones = _make_zones()

    filtered = filter_segments_to_leonia(
        summary, zones, borough_polygon=TEST_BOROUGH
    )

    kept_names = set(filtered["osm_name"].tolist())
    # Only the un-restricted local street should remain.
    assert "Some Local Street" in kept_names
    # County arterials — inside the polygon but not under borough authority.
    assert "Fort Lee Road" not in kept_names
    assert "Main Street" not in kept_names
    # Outside the polygon — drop
    assert "George Washington Bridge Plaza" not in kept_names
    # Inside the polygon spatially but a state road — drop
    assert "New Jersey Turnpike" not in kept_names


def test_filter_segments_to_leonia_handles_empty_input():
    empty = pd.DataFrame(columns=["zone_name", "osm_way_id", "osm_name", "road_class"])
    out = filter_segments_to_leonia(empty, _make_zones(), borough_polygon=TEST_BOROUGH)
    assert out.empty


def test_annotate_in_leonia_keeps_all_rows_with_flag():
    summary = _make_summary()
    zones = _make_zones()
    annotated = annotate_in_leonia(summary, zones, borough_polygon=TEST_BOROUGH)
    assert len(annotated) == len(summary)
    by_name = annotated.set_index("osm_name")["in_leonia_jurisdiction"].to_dict()
    # County arterials must be flagged False.
    assert not by_name["Fort Lee Road"]
    assert not by_name["Main Street"]
    # Local municipal street must be flagged True.
    assert by_name["Some Local Street"]
    # Outside polygon / state road → False.
    assert not by_name["George Washington Bridge Plaza"]
    assert not by_name["New Jersey Turnpike"]


def test_filter_falls_back_to_name_class_only_when_no_zones():
    summary = _make_summary()
    out = filter_segments_to_leonia(summary, None, borough_polygon=TEST_BOROUGH)
    names = out["osm_name"].tolist()
    # State-road exclusion still applies without geometries.
    assert "New Jersey Turnpike" not in names
    assert "George Washington Bridge Plaza" not in names
    # County arterial exclusion also applies without geometries.
    assert "Fort Lee Road" not in names
    assert "Main Street" not in names


def test_load_leonia_polygon_real_borough():
    """Smoke-test: the canonical Leonia polygon loads and covers the
    expected coordinates.
    """
    from leonia_traffic.config import load_leonia_polygon

    poly = load_leonia_polygon()
    # Borough hall is at roughly -74.000, 40.864
    assert poly.contains(Point(-74.000, 40.864))
    # GWB Plaza (Fort Lee) is well outside Leonia
    assert not poly.contains(Point(-73.972, 40.852))
