"""Tests for the StreetLight Zone Activity (leonia-streets) loader."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.data import za_streets_loader as zl


def test_discover_za_streets_finds_real_export():
    paths = zl.discover_za_streets()
    if paths is None:
        pytest.skip("ZA streets export not present")
    assert paths.za_all.exists()
    assert paths.zone_trip_all is not None and paths.zone_trip_all.exists()
    assert paths.home_distance is not None and paths.home_distance.exists()
    assert paths.home_zips_top is not None and paths.home_zips_top.exists()
    assert paths.tourist_summary is not None and paths.tourist_summary.exists()
    assert paths.shapefile_line_zip is not None and paths.shapefile_line_zip.exists()


def test_discover_returns_none_for_missing(tmp_path):
    out = zl.discover_za_streets(tmp_path)
    assert out is None


def test_load_za_main_basic_shape():
    df = zl.load_za_main()
    if df.empty:
        pytest.skip("ZA streets export not present")
    expected = {
        "zone_name", "street_name", "osm_way_id",
        "filter", "day_type_code", "day_type_label",
        "day_part_code", "day_part_label", "zone_volume",
    }
    assert expected.issubset(set(df.columns))
    # Volume should be positive and finite for the vast majority of rows
    vol = df["zone_volume"].dropna()
    assert (vol > 0).mean() > 0.99
    # We should see hourly day parts (1..24) plus an All Day code 0
    codes = set(df["day_part_code"].dropna().astype(int).unique())
    assert 0 in codes
    assert 9 in codes  # 8am hour code per analysis txt


def test_load_za_main_christie_heights_present():
    df = zl.load_za_main()
    if df.empty:
        pytest.skip("ZA streets export not present")
    mask = df["zone_name"].str.contains("Christie Heights", na=False)
    assert mask.any(), "Christie Heights should be in the export"
    # The composite cut-through expectation: weekday Visitor volume on
    # Thursday all-day should exceed the Saturday volume by a meaningful
    # margin. (The loader doesn't compute this — we just sanity-check
    # the rows it produces.)
    sub = df[mask & (df["filter"] == "Visitors") &
             (df["day_part_code"] == 0)]
    assert sub["day_type_code"].dropna().nunique() >= 2


def test_load_za_trip_bins_short_names():
    df = zl.load_za_trip()
    if df.empty:
        pytest.skip("ZA streets export not present")
    # We expect the short-rename to produce bin columns like these
    sample_bins = {
        "tt_min_0_10", "tt_min_150_plus",
        "len_mi_0_1", "len_mi_100_plus",
        "spd_mph_0_10", "spd_mph_70_plus",
        "circuity_1_2", "circuity_6_plus",
    }
    assert sample_bins.issubset(set(df.columns))
    # Each bin column should be a fraction in [0, 1]
    for c in ("tt_min_0_10", "len_mi_5_10", "spd_mph_20_30", "circuity_2_3"):
        if c in df.columns:
            vals = df[c].dropna()
            assert (vals >= 0).all() and (vals <= 1.0 + 1e-6).all()


def test_load_za_home_distance_distance_columns_present():
    df = zl.load_za_home_distance()
    if df.empty:
        pytest.skip("ZA streets export not present")
    expected_cols = (
        "Percent Home less than 1 mi",
        "Percent Home 1 to 3 mi",
        "Percent Home 3 to 5 mi",
        "Percent Home more than 100 mi",
    )
    for c in expected_cols:
        assert c in df.columns
    # rows sum to ~1.0 across the distance bins
    dist_cols = [c for c in df.columns if c.startswith("Percent Home")]
    row_sums = df[dist_cols].sum(axis=1).dropna()
    assert ((row_sums > 0.95) & (row_sums < 1.05)).mean() > 0.95


def test_load_za_home_zips_top_returns_rank():
    df = zl.load_za_home_zips_top()
    if df.empty:
        pytest.skip("ZA streets export not present")
    assert "zip_code" in df.columns
    assert "pct_home_location" in df.columns
    assert "rank" in df.columns
    # Ranks should be small positive integers
    assert df["rank"].dropna().min() >= 1


def test_load_za_line_shapes_geometry_and_road_type():
    gdf = zl.load_za_line_shapes()
    if gdf.empty:
        pytest.skip("ZA streets export not present")
    assert "geometry" in gdf.columns
    assert "street_name" in gdf.columns
    assert "osm_way_id" in gdf.columns
    # At least one Christie Heights line segment
    assert gdf["street_name"].str.contains("Christie Heights", na=False).any()


def test_load_za_work_block_groups_shape_and_fips():
    df = zl.load_za_work_block_groups()
    if df.empty:
        pytest.skip("ZA work_block_groups not present")
    expected = {"zone_name", "filter", "block_group_id", "pct_work_location",
                "state_fips", "county_fips", "tract"}
    assert expected.issubset(set(df.columns))
    # All block group IDs are 12-char strings after the quote-strip.
    bg = df["block_group_id"].dropna().astype(str)
    assert (bg.str.len() == 12).mean() > 0.99
    # state_fips is the first 2 chars; "34" (NJ) should dominate.
    assert df["state_fips"].mode().iat[0] == "34"
    # county_fips column is 5 chars
    assert df["county_fips"].dropna().str.len().eq(5).all()


def test_load_za_work_distance_shape():
    df = zl.load_za_work_distance()
    if df.empty:
        pytest.skip("ZA work_distance not present")
    for c in ("Percent Work less than 1 mi", "Percent Work more than 100 mi"):
        assert c in df.columns
    distance_cols = [c for c in df.columns if c.startswith("Percent Work")]
    # Each bin is a non-negative fraction. Rows sum to <= 1; the
    # remainder ("workplace unknown") is not exposed as a column.
    for c in distance_cols:
        vals = df[c].dropna()
        assert (vals >= 0).all()
        assert (vals <= 1.0 + 1e-6).all()
    row_sums = df[distance_cols].sum(axis=1).dropna()
    assert (row_sums <= 1.05).all()


def test_visitors_only_filter():
    df = zl.load_za_main()
    if df.empty:
        pytest.skip("ZA streets export not present")
    vis = zl.visitors_only(df)
    assert (vis["filter"] == "Visitors").all()
    assert len(vis) < len(df) or (df["filter"] == "Visitors").all()
