"""Tests for :func:`leonia_traffic.network.osm_builder.apply_congestion_overrides`."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.network.osm_builder import apply_congestion_overrides


class FakeLink:
    def __init__(self, name: str, speed: float = 20.0):
        self.name = name
        self.free_flow_speed = speed
        self._changes: list[float] = []

    def change_free_flow_speed(self, new_value: float) -> None:
        self.free_flow_speed = new_value
        self._changes.append(new_value)


class FakeWorld:
    def __init__(self):
        self.LINKS: list[FakeLink] = []
        self._by_name: dict[str, FakeLink] = {}

    def add_link(self, name: str, speed: float = 20.0) -> FakeLink:
        link = FakeLink(name, speed)
        self.LINKS.append(link)
        self._by_name[name] = link
        return link

    def get_link(self, name: str) -> FakeLink:
        return self._by_name[name]


def test_apply_congestion_overrides_changes_matching_links():
    W = FakeWorld()
    W.add_link("Fort Lee Road-590576#1")
    W.add_link("Fort Lee Road-590576#2")
    W.add_link("Main Street-222585#1")
    W.add_link("Some Other-999#1")

    overrides = pd.DataFrame([
        {"osm_way_id": 590576, "observed_speed_ms": 10.0},
        {"osm_way_id": 222585, "observed_speed_ms": 8.0},
        {"osm_way_id": 12345, "observed_speed_ms": 12.0},  # no link match
    ])
    counts = apply_congestion_overrides(W, overrides)
    assert counts["n_overrides_seen"] == 3
    assert counts["n_links_changed"] == 3
    assert counts["n_osm_ways_with_match"] == 2
    assert counts["n_osm_ways_without_match"] == 1

    assert W.get_link("Fort Lee Road-590576#1").free_flow_speed == 10.0
    assert W.get_link("Fort Lee Road-590576#2").free_flow_speed == 10.0
    assert W.get_link("Main Street-222585#1").free_flow_speed == 8.0
    assert W.get_link("Some Other-999#1").free_flow_speed == 20.0  # untouched


def test_apply_congestion_overrides_floor_speed():
    W = FakeWorld()
    W.add_link("Foo-1#1")
    overrides = pd.DataFrame([
        {"osm_way_id": 1, "observed_speed_ms": -5.0},
    ])
    apply_congestion_overrides(W, overrides, min_speed_ms=2.0)
    assert W.get_link("Foo-1#1").free_flow_speed == 2.0


def test_apply_congestion_overrides_writes_cache(tmp_path):
    W = FakeWorld()
    W.add_link("Foo-1#1")
    overrides = pd.DataFrame([{"osm_way_id": 1, "observed_speed_ms": 9.0}])
    cache = tmp_path / "applied.parquet"
    apply_congestion_overrides(W, overrides, cache_path=cache)
    assert cache.exists()
    reread = pd.read_parquet(cache)
    assert len(reread) == 1
    assert reread.iloc[0]["uxsim_link_name"] == "Foo-1#1"


def test_apply_congestion_overrides_empty_input():
    W = FakeWorld()
    counts = apply_congestion_overrides(W, pd.DataFrame(columns=["osm_way_id", "observed_speed_ms"]))
    assert counts["n_overrides_seen"] == 0
    assert counts["n_links_changed"] == 0
