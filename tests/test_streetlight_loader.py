"""Tests for the StreetLight loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from leonia_traffic.config import STREETLIGHT_DIR
from leonia_traffic.data.streetlight_loader import (
    discover_sources,
    load_cached,
    parse_filters,
    parse_zone_name,
    restrict_to_study_area,
)


@pytest.fixture(scope="module")
def cached_gdf():
    return load_cached()


def test_parse_zone_name_basic():
    assert parse_zone_name("Oleri Terrace / 11586338 / 7") == (
        "Oleri Terrace",
        11586338,
        7,
    )


def test_parse_zone_name_with_extra_spaces():
    assert parse_zone_name("  Some Street/12345/3  ") == ("Some Street", 12345, 3)


def test_parse_zone_name_with_slash_in_name():
    name, osmid, split = parse_zone_name("Foo / Bar Street / 9999 / 2")
    assert osmid == 9999
    assert split == 2
    assert "Foo" in name and "Bar" in name


def test_parse_zone_name_no_pattern():
    name, osmid, split = parse_zone_name("just a label")
    assert osmid is None and split is None
    assert name == "just a label"


def test_parse_zone_name_empty():
    assert parse_zone_name("") == ("", None, None)


def test_parse_filters_recovers_sections(tmp_path: Path):
    p = tmp_path / "Filters.txt"
    p.write_text(
        "Subscription ID: 26600\n"
        "Organization: Test\n"
        "\n"
        "Day Types:\n"
        "  - Weekday\n"
        "  - Weekend Days\n"
        "\n"
        "Day Parts:\n"
        "  - Peak AM\n"
    )
    meta = parse_filters(p)
    assert meta.subscription_id == "26600"
    assert meta.organization == "Test"
    assert meta.day_types == ["Weekday", "Weekend Days"]
    assert meta.day_parts == ["Peak AM"]


@pytest.mark.skipif(
    not STREETLIGHT_DIR.exists(), reason="StreetLight data folder missing"
)
def test_discover_sources_finds_all_three():
    sources = discover_sources()
    labels = {s.label for s in sources}
    # We expect 'all_days', 'weekdays', 'weekend' to be present.
    expected = {"all_days", "weekdays", "weekend"}
    assert expected.issubset(labels), f"Missing: {expected - labels}"


@pytest.mark.skipif(
    not STREETLIGHT_DIR.exists(), reason="StreetLight data folder missing"
)
def test_load_cached_shape(cached_gdf):
    # Three sources, ~3000 segments each.
    assert len(cached_gdf) > 8000
    assert {"osm_way_id", "avg_volume", "source", "geometry"}.issubset(
        cached_gdf.columns
    )
    assert cached_gdf.crs.to_epsg() == 4326


@pytest.mark.skipif(
    not STREETLIGHT_DIR.exists(), reason="StreetLight data folder missing"
)
def test_osm_id_parsing_coverage(cached_gdf):
    # Every row's Zone Name should parse to an OSM way ID.
    assert cached_gdf["osm_way_id"].notna().mean() > 0.99


@pytest.mark.skipif(
    not STREETLIGHT_DIR.exists(), reason="StreetLight data folder missing"
)
def test_restrict_to_study_area_drops_remote(cached_gdf):
    study = restrict_to_study_area(cached_gdf)
    assert len(study) <= len(cached_gdf)
    # NY rows should be dropped because they're not in the study-area list.
    assert (study["city_county_state"] == "New York, New York, New York").sum() == 0
