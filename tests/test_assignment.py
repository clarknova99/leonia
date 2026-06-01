"""Tests for ``leonia_traffic.assignment``.

These tests exercise the User-Equilibrium pipeline on small, hand-built
graphs (no OSM access) plus a smoke test against the real Leonia network
when the cached parquet lake is present.
"""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import pytest

from leonia_traffic.assignment import (
    AssignmentResult,
    bpr_travel_time,
    build_assignment_graph,
    run_ue,
)
from leonia_traffic.assignment.assignment import _all_or_nothing
from leonia_traffic.assignment.network import graph_edges_summary


# ---------------------------------------------------------------------------
# Tiny hand-built networks
# ---------------------------------------------------------------------------


def _two_route_uxsim_data() -> tuple[list, list]:
    """A canonical Braess-style 4-node, 2-route network.

        s ---a---> A ---b---> t
        s ---c---> B ---d---> t

    Both routes have equal free-flow time; UE should split flow 50/50.
    """
    nodes = [
        ["s", -73.99, 40.86],
        ["A", -73.98, 40.87],
        ["B", -73.98, 40.85],
        ["t", -73.97, 40.86],
    ]
    # link = [name, from, to, lanes, free_flow_speed_ms, length_deg]
    # 0.01 deg ≈ 1100 m at this latitude.
    length_deg = 0.005
    links = [
        ["seg-100-a", "s", "A", 1, 13.4, length_deg],
        ["seg-101-b", "A", "t", 1, 13.4, length_deg],
        ["seg-102-c", "s", "B", 1, 13.4, length_deg],
        ["seg-103-d", "B", "t", 1, 13.4, length_deg],
    ]
    return nodes, links


def _linear_uxsim_data() -> tuple[list, list]:
    """s -> A -> t, single path. UE should put all flow on it."""
    nodes = [
        ["s", -73.99, 40.86],
        ["A", -73.98, 40.86],
        ["t", -73.97, 40.86],
    ]
    links = [
        ["seg-200-x", "s", "A", 2, 13.4, 0.005],
        ["seg-201-y", "A", "t", 2, 13.4, 0.005],
    ]
    return nodes, links


# ---------------------------------------------------------------------------
# Pure unit tests (no OSM / parquet dependency)
# ---------------------------------------------------------------------------


def test_bpr_monotone_increasing():
    t_ff = 100.0
    cap = 1000.0
    t0 = bpr_travel_time(0, t_ff, cap)
    t_at_cap = bpr_travel_time(cap, t_ff, cap)
    t_over = bpr_travel_time(2 * cap, t_ff, cap)
    assert t0 == pytest.approx(t_ff)
    # At v/c=1 the BPR formula gives t_ff * (1 + 0.15) = 115.
    assert t_at_cap == pytest.approx(t_ff * 1.15)
    assert t_over > t_at_cap


def test_build_assignment_graph_attrs():
    nodes, links = _linear_uxsim_data()
    G = build_assignment_graph(nodes, links)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    d = G["s"]["A"]
    assert d["lanes"] == 2
    # length_deg=0.005 * coef_degree_to_meter (~111000) ≈ 555 m.
    assert 500 < d["length_m"] < 700
    assert d["capacity_vph"] == pytest.approx(2 * 800.0)
    assert d["free_flow_time_s"] == pytest.approx(d["length_m"] / 13.4)


def test_run_ue_linear_assigns_all_flow():
    nodes, links = _linear_uxsim_data()
    G = build_assignment_graph(nodes, links)
    demand = {("s", "t"): 500.0}
    res = run_ue(G, demand, max_iter=10)
    # The single path has 2 edges; both should carry 500 vph.
    flows = res.edges.set_index(["u", "v"])["assigned_volume_vph"]
    assert flows.loc[("s", "A")] == pytest.approx(500.0)
    assert flows.loc[("A", "t")] == pytest.approx(500.0)
    assert res.converged


def test_run_ue_two_route_splits_evenly():
    """Symmetric two-route network: UE should split flow ~50/50."""
    nodes, links = _two_route_uxsim_data()
    G = build_assignment_graph(nodes, links)
    demand = {("s", "t"): 1500.0}  # ~2x per-link capacity → forces split
    res = run_ue(G, demand, max_iter=40, rel_gap=1e-4)
    flows = res.edges.set_index(["u", "v"])["assigned_volume_vph"]
    route_A = flows.loc[("s", "A")]
    route_B = flows.loc[("s", "B")]
    # 50/50 ± 5 % tolerance.
    assert route_A == pytest.approx(750.0, rel=0.05)
    assert route_B == pytest.approx(750.0, rel=0.05)
    assert route_A + route_B == pytest.approx(1500.0, rel=0.01)


def test_run_ue_handles_missing_origin():
    nodes, links = _linear_uxsim_data()
    G = build_assignment_graph(nodes, links)
    # 'XX' is not in the graph; should be skipped without crashing.
    demand = {("XX", "t"): 100.0, ("s", "t"): 200.0}
    res = run_ue(G, demand, max_iter=5)
    assert res.skipped_od_pairs == 1
    flows = res.edges.set_index(["u", "v"])["assigned_volume_vph"]
    assert flows.loc[("s", "A")] == pytest.approx(200.0)


def test_assignment_result_by_osm_way_aggregation():
    """Two parallel links sharing the same osm_way_id collapse to one row."""
    G = nx.DiGraph()
    G.add_node("a", lon=-73.98, lat=40.86)
    G.add_node("b", lon=-73.97, lat=40.86)
    G.add_node("c", lon=-73.96, lat=40.86)
    # Two edges, same osm_way_id 555
    G.add_edge("a", "b", link_name="x-555", osm_way_id=555,
               length_m=100, free_flow_speed_ms=13.4, free_flow_time_s=7.46,
               lanes=1, capacity_vph=800)
    G.add_edge("b", "c", link_name="y-555", osm_way_id=555,
               length_m=100, free_flow_speed_ms=13.4, free_flow_time_s=7.46,
               lanes=1, capacity_vph=800)
    demand = {("a", "c"): 400.0}
    res = run_ue(G, demand, max_iter=5)
    agg = res.by_osm_way()
    assert len(agg) == 1
    assert int(agg.iloc[0]["osm_way_id"]) == 555
    # Each edge carries 400; summed gives 800.
    assert agg.iloc[0]["assigned_volume_vph"] == pytest.approx(800.0)
    assert agg.iloc[0]["n_edges"] == 2


def test_all_or_nothing_empty_demand():
    G = nx.DiGraph()
    G.add_edge("a", "b", free_flow_time_s=1.0, _w=1.0)
    flows, skipped = _all_or_nothing(G, {}, "_w")
    assert flows == {}
    assert skipped == 0


# ---------------------------------------------------------------------------
# Integration smoke (skipped if the data lake / network cache is absent)
# ---------------------------------------------------------------------------


def _data_lake_available() -> bool:
    from pathlib import Path
    return (
        Path("data/processed/streetlight/bridge_od.parquet").exists()
        and Path("data/processed/streetlight/bridge_od_zones.parquet").exists()
        and Path("data/network/leonia_osm_network.pkl").exists()
    )


@pytest.mark.skipif(
    not _data_lake_available(),
    reason="Data lake or network cache not built (run scripts/00_build_datasets.py).",
)
def test_end_to_end_smoke_on_real_network():
    """End-to-end: load real network + bridge OD, assign, validate."""
    import geopandas as gpd
    from leonia_traffic.network.osm_builder import build_or_load_network
    from leonia_traffic.assignment import (
        bridge_od_to_demand,
        validate_against_streetscanner,
    )

    nodes, links, _ = build_or_load_network()
    G = build_assignment_graph(nodes, links)
    summary = graph_edges_summary(G)
    assert summary["n_nodes"] > 100
    assert summary["n_edges"] > 100
    assert summary["n_edges_with_osm_id"] > 0

    bod = pd.read_parquet("data/processed/streetlight/bridge_od.parquet")
    zones = gpd.read_parquet("data/processed/streetlight/bridge_od_zones.parquet")
    demand = bridge_od_to_demand(bod, zones, G, day_type_code=1, day_part_code=2)
    assert len(demand) > 0
    assert sum(demand.values()) > 0

    res = run_ue(G, demand, max_iter=20)
    assert isinstance(res, AssignmentResult)
    assert (res.edges["assigned_volume_vph"] > 0).sum() > 0

    seg = gpd.read_parquet(
        "data/processed/streetlight/streetscanner_segments.parquet"
    )
    stats = validate_against_streetscanner(res, seg, source_label="weekdays")
    # At least *some* street-scanner segments should match by OSM way id.
    assert stats.n_matched > 0
    # GEH < 10 on the majority of matched segments is the industry rule
    # of thumb. With Bridge-OD-only demand and a 4 h Peak-AM window
    # this is a loose floor; the realistic target sits >> 80 %.
    assert stats.pct_geh_lt_10 >= 0.5
