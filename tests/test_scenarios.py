"""Tests for scenario application logic (no UXsim simulation)."""

from __future__ import annotations

from leonia_traffic.simulation.scenarios import (
    Closure,
    LaneReduction,
    OneWayConversion,
    SpeedHumpCalming,
    apply_scenarios,
)


def _toy_network():
    # Two nodes for each of 3 streets; bearings 0° (N), 90° (E), 180° (S).
    nodes = [
        [1, 0.0, 0.0], [2, 0.0, 0.001],   # N: bearing 0
        [3, 0.0, 0.0], [4, 0.001, 0.0],   # E: bearing 90
        [5, 0.0, 0.0], [6, 0.0, -0.001],  # S: bearing 180
    ]
    links = [
        ["North-100", 1, 2, 2, 11.0, 0.001],
        ["North-100-reverse", 2, 1, 2, 11.0, 0.001],
        ["East-200", 3, 4, 2, 11.0, 0.001],
        ["South-300", 5, 6, 2, 11.0, 0.001],
    ]
    return nodes, links


def test_closure_removes_listed_ways():
    nodes, links = _toy_network()
    _, new_links = apply_scenarios(nodes, links, [Closure(osm_way_ids=[100])])
    remaining_ids = [name for name, *_ in new_links]
    assert all("100" not in name for name in remaining_ids)
    assert any("200" in name for name in remaining_ids)


def test_calming_reduces_speed():
    nodes, links = _toy_network()
    _, new_links = apply_scenarios(
        nodes, links, [SpeedHumpCalming(osm_way_ids=[100], free_flow_speed_factor=0.5)]
    )
    target = [l for l in new_links if "North-100" in l[0]]
    assert all(l[4] <= 11.0 * 0.5 + 1e-6 for l in target) or all(
        l[4] >= 4.5 - 1e-6 for l in target
    )


def test_one_way_keeps_only_allowed_bearing():
    nodes, links = _toy_network()
    _, new_links = apply_scenarios(
        nodes,
        links,
        [OneWayConversion(osm_way_ids=[100], allowed_bearing_deg=0.0, tolerance_deg=45)],
    )
    # Only the northbound link should remain for OSM way 100.
    target = [l for l in new_links if "-100" in l[0]]
    assert len(target) == 1
    assert "reverse" not in target[0][0]


def test_lane_reduction():
    nodes, links = _toy_network()
    _, new_links = apply_scenarios(
        nodes, links, [LaneReduction(osm_way_ids=[100], target_lanes=1)]
    )
    target = [l for l in new_links if "-100" in l[0]]
    for l in target:
        assert l[3] == 1


def test_scenarios_compose():
    nodes, links = _toy_network()
    out_nodes, out_links = apply_scenarios(
        nodes,
        links,
        [
            SpeedHumpCalming(osm_way_ids=[200], free_flow_speed_factor=0.4),
            Closure(osm_way_ids=[300]),
        ],
    )
    # 300 closed, 200 slowed.
    east = [l for l in out_links if "-200" in l[0]]
    assert east and east[0][4] <= 11.0 * 0.4 + 1e-6 or east[0][4] >= 4.5
    south = [l for l in out_links if "-300" in l[0]]
    assert not south
