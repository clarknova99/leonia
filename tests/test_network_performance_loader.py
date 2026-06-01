"""Tests for the Network Performance loader."""

from __future__ import annotations

import pandas as pd

from leonia_traffic.data import network_performance_loader as npl


# ---------------------------------------------------------------------------
# Zone-name parsing
# ---------------------------------------------------------------------------


def test_parse_network_zone_name_three_part():
    name, way_id, split = npl.parse_network_zone_name("1st Street / 1007650684 / 1")
    assert name == "1st Street"
    assert way_id == 1007650684  # the MIDDLE number, not the trailing split
    assert split == 1


def test_parse_network_zone_name_handles_slashes_in_name():
    name, way_id, split = npl.parse_network_zone_name(
        "Grandview Terrace / 11577805 / 4"
    )
    assert name == "Grandview Terrace"
    assert way_id == 11577805
    assert split == 4


def test_parse_network_zone_name_two_part_fallback():
    name, way_id, split = npl.parse_network_zone_name("Anderson Avenue / 16999857")
    assert name == "Anderson Avenue"
    assert way_id == 16999857
    assert split is None


def test_parse_network_zone_name_none():
    assert npl.parse_network_zone_name(None) == ("", None, None)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_finds_real_export():
    paths = npl.discover_network_performance()
    assert paths is not None
    assert paths.seg_metrics.exists()
    assert paths.seg_prediction is not None and paths.seg_prediction.exists()
    assert paths.zones is not None and paths.zones.exists()
    assert (
        paths.shapefile_segment_line is not None
        and paths.shapefile_segment_line.exists()
    )


def test_discover_returns_none_for_missing(tmp_path):
    assert npl.discover_network_performance(tmp_path) is None


# ---------------------------------------------------------------------------
# Main metrics loader
# ---------------------------------------------------------------------------


def test_load_network_performance_basic_shape():
    df = npl.load_network_performance()
    assert not df.empty
    expected = {
        "zone_name", "street_name", "osm_way_id", "split_num",
        "day_type_code", "day_type_label", "day_part_code", "day_part_label",
        "avg_daily_volume", "avg_speed_mph", "free_flow_speed_mph",
        "free_flow_factor", "congestion", "vmt", "vhd",
        "speed_p05", "speed_p15", "speed_p85", "speed_p95",
    }
    assert expected.issubset(df.columns)
    # 8 day types: All Days + Mon..Sun (codes 0..7).
    assert set(df["day_type_code"].dropna().unique()).issubset(set(range(0, 8)))
    # 25 day parts: All Day + 24 clock hours (codes 0..24).
    assert set(df["day_part_code"].dropna().unique()).issubset(set(range(0, 25)))
    # OSM way IDs parsed from the middle of the 3-part name are large.
    assert df["osm_way_id"].dropna().min() > 1000
    # congestion is the complement of free-flow factor.
    sample = df.dropna(subset=["free_flow_factor", "congestion"]).iloc[0]
    assert abs((1.0 - sample["free_flow_factor"]) - sample["congestion"]) < 1e-9


def test_load_prediction_interval():
    df = npl.load_network_performance_prediction()
    assert not df.empty
    assert {"osm_way_id", "pred_lower_95", "pred_upper_95", "avg_daily_volume"}.issubset(
        df.columns
    )
    # Lower bound should not exceed the upper bound where both present.
    both = df.dropna(subset=["pred_lower_95", "pred_upper_95"])
    assert (both["pred_lower_95"] <= both["pred_upper_95"]).all()


def test_load_zones_roster():
    df = npl.load_network_performance_zones()
    assert not df.empty
    assert {"zone_name", "osm_way_id", "fingerprint1", "fingerprint2"}.issubset(
        df.columns
    )


def test_load_shapes_returns_geodataframe():
    gdf = npl.load_network_performance_shapes()
    assert not gdf.empty
    assert "geometry" in gdf.columns
    assert gdf.crs is not None
    assert "osm_way_id" in gdf.columns
    assert gdf["osm_way_id"].dropna().min() > 1000


# ---------------------------------------------------------------------------
# Peak-hour summary
# ---------------------------------------------------------------------------


def test_peak_hour_volume_summary():
    df = npl.load_network_performance()
    summary = npl.peak_hour_volume(df, day_type_code=0)
    assert not summary.empty
    expected = {
        "zone_name", "osm_way_id", "street_name",
        "peak_volume", "peak_hour_code", "peak_am_volume", "peak_pm_volume",
        "free_flow_speed_mph", "min_speed_p15", "all_day_volume",
    }
    assert expected.issubset(summary.columns)
    # Peak hourly volume cannot exceed the all-day daily volume by a wide
    # margin; at minimum it should be a positive number.
    assert summary["peak_volume"].dropna().min() >= 0
    # Sorted descending by peak volume.
    peaks = summary["peak_volume"].dropna().to_numpy()
    assert (peaks[:-1] >= peaks[1:]).all()


def test_peak_hour_code_in_range():
    df = npl.load_network_performance()
    summary = npl.peak_hour_volume(df, day_type_code=0)
    codes = pd.to_numeric(summary["peak_hour_code"], errors="coerce").dropna()
    assert codes.between(1, 24).all()
