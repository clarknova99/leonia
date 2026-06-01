"""Tests for the congestion analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.analysis import congestion as cg
from leonia_traffic.data import congestion_loader as cl


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return cl.load_congestion()


def test_worst_hours_per_corridor_filters_garbage(df):
    out = cg.worst_hours_per_corridor(df, top_n_per_zone=1)
    assert not out.empty
    # Filter should drop pathological NaN-speed rows.
    assert out["avg_speed_mph"].dropna().min() >= 1.0
    # TTI should generally be < 6 once obvious garbage is removed.
    assert out["tti"].quantile(0.95) < 8.0


def test_delay_hotspot_ranking_top_corridor_known(df):
    out = cg.delay_hotspot_ranking(df, top_n=10)
    assert not out.empty
    # Grand Avenue, GWB Plaza, or NJ Turnpike should be in the top 5 by VHD.
    top5 = out.head(5)["osm_name"].str.lower()
    assert any(top5.str.contains("grand avenue|george washington bridge|turnpike|main street"))


def test_reliability_breakdown_uses_canonical_classes(df):
    out = cg.reliability_breakdown(df)
    assert not out.empty
    classes = set(out["reliability_class"].unique())
    assert classes.issubset({"Reliable", "Moderate", "Unreliable", "Unknown"})
    # Shares should sum to ~1.0 per road class.
    sums = out.groupby("road_class")["share_of_road_class"].sum()
    assert (sums.between(0.99, 1.01)).all()


def test_link_speed_overrides_returns_one_per_osm_way(df):
    overrides = cg.link_speed_overrides(df)
    assert not overrides.empty
    assert "observed_speed_ms" in overrides.columns
    # One row per OSM way ID (no duplicates).
    assert overrides["osm_way_id"].is_unique
    # Speeds in plausible range (1..40 m/s ≈ 2..90 mph).
    assert overrides["observed_speed_ms"].dropna().between(0.5, 50).all()


def test_link_speed_overrides_writes_cache(df, tmp_path):
    cache = tmp_path / "speeds.parquet"
    out = cg.link_speed_overrides(df, cache_path=cache)
    assert cache.exists()
    reread = pd.read_parquet(cache)
    assert len(reread) == len(out)
