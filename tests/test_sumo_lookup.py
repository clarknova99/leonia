"""Tests for the OSM ↔ SUMO edge lookup helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from leonia_traffic.sumo.net_lookup import (
    edges_for_osm_ways,
    load_osm_to_sumo_lookup,
)


def _write_fixture_net(tmp_path: Path) -> Path:
    """Write a minimal SUMO ``.net.xml`` with three edges and origIds."""
    net = tmp_path / "fixture.net.xml"
    net.write_text("""\
<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
  <location netOffset="0,0" convBoundary="0,0,1,1" origBoundary="0,0,1,1"/>
  <edge id="A" function="normal">
    <param key="origId" value="11111"/>
    <lane id="A_0" shape="0,0 1,0"/>
  </edge>
  <edge id="B" function="normal">
    <lane id="B_0" shape="0,0 1,1">
      <param key="origId" value="22222"/>
    </lane>
  </edge>
  <edge id="-B" function="normal">
    <lane id="-B_0" shape="1,1 0,0">
      <param key="origId" value="22222"/>
    </lane>
  </edge>
  <edge id="C" function="normal" origId="33333 44444">
    <lane id="C_0" shape="0,1 1,1"/>
  </edge>
  <edge id=":internal" function="internal">
    <lane id=":internal_0" shape="0,0 0,0.1"/>
  </edge>
</net>
""", encoding="utf-8")
    return net


def test_load_osm_to_sumo_lookup_handles_three_origid_locations(tmp_path):
    """origId can live on the edge, on a <param>, or on the lane <param>."""
    net = _write_fixture_net(tmp_path)
    lookup = load_osm_to_sumo_lookup(net)
    # Edge A: param on the <edge>.
    assert lookup[11111] == ["A"]
    # Edge B: param on the <lane>. -B is the reverse direction with the
    # same origId, so the lookup contains both.
    assert sorted(lookup[22222]) == sorted(["B", "-B"])
    # Edge C: space-separated origId attribute on the edge.
    assert lookup[33333] == ["C"]
    assert lookup[44444] == ["C"]


def test_load_osm_to_sumo_lookup_skips_internal_edges(tmp_path):
    """``function="internal"`` edges must not appear in the lookup."""
    net = _write_fixture_net(tmp_path)
    lookup = load_osm_to_sumo_lookup(net)
    flat = [e for edges in lookup.values() for e in edges]
    assert ":internal" not in flat


def test_load_osm_to_sumo_lookup_returns_empty_for_missing_file(tmp_path):
    assert load_osm_to_sumo_lookup(tmp_path / "does_not_exist.xml") == {}


def test_load_osm_to_sumo_lookup_returns_empty_for_unparseable(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("not xml", encoding="utf-8")
    assert load_osm_to_sumo_lookup(bad) == {}


def test_edges_for_osm_ways_dedupes_and_skips_unknowns():
    lookup = {1: ["a", "b"], 2: ["b", "c"], 3: ["d"]}
    out = edges_for_osm_ways([1, 2, 99, 3], lookup)
    assert out == ["a", "b", "c", "d"]


def test_edges_for_osm_ways_handles_string_inputs():
    lookup = {1: ["a"]}
    assert edges_for_osm_ways(["1", 1.0], lookup) == ["a"]
    assert edges_for_osm_ways(["not-an-int"], lookup) == []
