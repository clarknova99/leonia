"""Tests for the Street Scanner Trend loader."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.data.streetscanner_trend_loader import (
    discover_streetscanner_trend,
    load_streetscanner_trend,
)


@pytest.fixture(scope="module")
def trend_df():
    paths = discover_streetscanner_trend()
    if paths is None:
        pytest.skip("streetscanner_trend folder not present")
    return load_streetscanner_trend()


def test_trend_loads_nonempty(trend_df):
    assert len(trend_df) > 1000


def test_trend_expected_columns(trend_df):
    expected = {
        "zone_name", "osm_name", "osm_way_id", "year_month",
        "year", "month", "avg_volume", "day_type", "day_part_raw",
    }
    assert expected.issubset(set(trend_df.columns))


def test_trend_year_month_parsed(trend_df):
    assert pd.api.types.is_datetime64_any_dtype(trend_df["year_month"])
    assert trend_df["year_month"].min() <= pd.Timestamp("2023-12-01")


def test_trend_long_format(trend_df):
    # Every zone should have multiple months.
    counts = trend_df.groupby("zone_name").size()
    assert counts.median() >= 12


def test_trend_no_change_column(trend_df):
    # "Change" wide column should be melted out.
    assert "change" not in [c.lower() for c in trend_df.columns]
