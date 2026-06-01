"""Tests for the Congestion Trends loader."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.data import congestion_loader as cl


def test_discover_congestion_finds_real_export():
    paths = cl.discover_congestion()
    assert paths is not None
    assert paths.ct_csv.exists()
    assert paths.shapefile_zip is not None and paths.shapefile_zip.exists()


def test_discover_congestion_returns_none_for_missing(tmp_path):
    assert cl.discover_congestion(tmp_path) is None


def test_load_congestion_basic_shape():
    df = cl.load_congestion()
    assert not df.empty
    expected_cols = {
        "zone_name", "osm_way_id", "osm_name", "road_class",
        "day_type_code", "day_type_label", "day_part_code", "day_part_label",
        "tti", "buffer_index", "vhd", "speed_p50", "free_flow_speed_mph",
        "reliability_level",
    }
    assert expected_cols.issubset(df.columns)
    # 3 day types: All / Weekday / Weekend Day.
    assert set(df["day_type_code"].dropna().unique()).issubset({0, 1, 2})
    # 30 day parts (codes 0..29).
    assert set(df["day_part_code"].dropna().unique()).issubset(set(range(0, 30)))
    # TTI is bounded around 1.0; allow some slack since observed speeds can
    # exceed free-flow on light-traffic hours.
    assert df["tti"].dropna().min() >= 0.5


def test_load_congestion_zones_returns_geodataframe():
    gdf = cl.load_congestion_zones()
    assert not gdf.empty
    assert "geometry" in gdf.columns
    assert gdf.crs is not None
    assert "osm_way_id" in gdf.columns


def test_summarize_link_reliability():
    df = cl.load_congestion()
    summary = cl.summarize_link_reliability(df)
    assert not summary.empty
    expected = {
        "zone_name", "osm_way_id", "osm_name", "road_class",
        "worst_tti", "worst_buffer", "total_weekday_vhd",
        "median_speed_mph", "free_flow_speed_mph", "worst_lottr",
        "reliability_class",
    }
    assert expected.issubset(summary.columns)
    # Sorted by worst_tti descending.
    tti_values = summary["worst_tti"].dropna().values
    assert all(tti_values[i] >= tti_values[i + 1] for i in range(len(tti_values) - 1))


def test_summarize_empty_input():
    out = cl.summarize_link_reliability(pd.DataFrame())
    assert out.empty
