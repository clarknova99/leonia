"""Tests for the SUMO-side scenario DSL adapter (no live libsumo)."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.simulation.scenarios import (
    Closure,
    LaneReduction,
    OneWayConversion,
    SpeedHumpCalming,
)
from leonia_traffic.sumo.scenarios_sumo import apply_scenario, apply_scenarios


class _FakeBackend:
    """Stand-in for ``libsumo.edge`` / ``libsumo.simulation`` etc.

    Records every speed change so tests can assert on the calls
    instead of running an actual SUMO process.
    """

    class _Edge:
        def __init__(self, parent):
            self._p = parent

        def getMaxSpeed(self, eid):
            return self._p.speeds.get(eid, 22.22)

        def setMaxSpeed(self, eid, v):
            self._p.speeds[eid] = v
            self._p.calls.append(("setMaxSpeed", eid, v))

    def __init__(self):
        self.speeds: dict[str, float] = {}
        self.calls: list[tuple] = []
        self.edge = self._Edge(self)


class _FakeRuntime:
    """Subset of :class:`SumoRuntime` used by the adapter."""

    def __init__(self, lookup: dict[int, list[str]]):
        self._osm_lookup = lookup
        self._backend = _FakeBackend()
        self._closed_ways: dict = {}

    @property
    def osm_lookup(self) -> dict[int, list[str]]:
        return dict(self._osm_lookup)

    def apply_closure(self, ways, *, crawl_speed_ms: float = 0.1):
        affected: list[str] = []
        for w in ways:
            for eid in self._osm_lookup.get(int(w), []):
                self._backend.setMaxSpeed = self._backend.edge.setMaxSpeed
                prev = self._backend.edge.getMaxSpeed(eid)
                self._backend.edge.setMaxSpeed(eid, crawl_speed_ms)
                affected.append(eid)
                self._closed_ways.setdefault(int(w), []).append((eid, prev))
        return affected

    def set_speed(self, ways, mph: float):
        target = max(0.1, float(mph) * 0.44704)
        affected: list[str] = []
        for w in ways:
            for eid in self._osm_lookup.get(int(w), []):
                self._backend.edge.setMaxSpeed(eid, target)
                affected.append(eid)
        return affected


def test_closure_applied_to_every_edge():
    rt = _FakeRuntime({100: ["E_a", "E_b"], 200: ["E_c"]})
    log = apply_scenario(rt, Closure(osm_way_ids=[100]))
    assert sorted(log.affected_edges) == ["E_a", "E_b"]
    # All E_a / E_b must now read crawl speed.
    assert rt._backend.speeds["E_a"] == pytest.approx(0.1)
    assert rt._backend.speeds["E_b"] == pytest.approx(0.1)


def test_speed_hump_calming_lowers_max_speed():
    rt = _FakeRuntime({100: ["E_a"]})
    sc = SpeedHumpCalming(
        osm_way_ids=[100],
        free_flow_speed_factor=0.5,
        min_free_flow_speed_ms=4.5,   # ~10 mph
    )
    log = apply_scenario(rt, sc)
    assert log.affected_edges == ["E_a"]
    # ``min_free_flow_speed_ms`` 4.5 m/s ≈ 10 mph → exactly 10 mph in
    # m/s = 4.4704. Adapter rounds the floor to ``max(10, ...)``.
    assert rt._backend.speeds["E_a"] == pytest.approx(10.0 * 0.44704, rel=0.05)


def test_oneway_conversion_blocks_only_reverse_direction():
    # Direct edge has no leading "-"; reverse edge id starts with "-".
    rt = _FakeRuntime({100: ["E_a", "-E_a"]})
    sc = OneWayConversion(osm_way_ids=[100], allowed_bearing_deg=180.0)
    log = apply_scenario(rt, sc)
    assert log.affected_edges == ["-E_a"]
    assert "-E_a" in rt._backend.speeds
    # The forward edge must not have been touched.
    assert "E_a" not in rt._backend.speeds


def test_lane_reduction_warns_and_falls_back():
    rt = _FakeRuntime({100: ["E_a"]})
    rt._backend.speeds["E_a"] = 22.22
    log = apply_scenario(rt, LaneReduction(osm_way_ids=[100], target_lanes=1))
    assert log.affected_edges == ["E_a"]
    assert any("LaneReduction" in n for n in log.notes)
    # 50 % cut applied as the fallback.
    assert rt._backend.speeds["E_a"] == pytest.approx(11.11, rel=0.01)


def test_unsupported_scenario_returns_note_with_no_affected_edges():
    class Mystery:
        name = "mystery"
    rt = _FakeRuntime({})
    log = apply_scenario(rt, Mystery())  # type: ignore[arg-type]
    assert log.affected_edges == []
    assert any("unsupported" in n for n in log.notes)


def test_apply_scenarios_returns_one_log_per_input():
    rt = _FakeRuntime({100: ["a"], 200: ["b"]})
    out = apply_scenarios(rt, [Closure(osm_way_ids=[100]),
                               Closure(osm_way_ids=[200])])
    assert len(out) == 2
    assert out[0].affected_edges == ["a"]
    assert out[1].affected_edges == ["b"]
