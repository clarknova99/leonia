"""Tests for ``apply_bridge_od_demand``."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from leonia_traffic.simulation.demand import apply_bridge_od_demand


@dataclass
class FakeNode:
    name: str


@dataclass
class FakeLink:
    name: str
    start_node: FakeNode
    end_node: FakeNode


class FakeWorld:
    """Minimal stand-in for ``uxsim.World`` for unit testing."""

    def __init__(self):
        self.links: dict[str, FakeLink] = {}
        self.demands: list[dict] = []

    def add_link(self, link_name: str, start: str, end: str) -> None:
        self.links[link_name] = FakeLink(link_name, FakeNode(start), FakeNode(end))

    def get_link(self, name: str) -> FakeLink:
        return self.links[name]

    def adddemand(self, orig, dest, t_start, t_end, *, volume):
        self.demands.append({
            "orig": orig, "dest": dest,
            "t_start": t_start, "t_end": t_end,
            "volume": volume,
        })


@dataclass
class FakeUXsimLinkRef:
    name: str
    osm_way_id: int


def test_apply_bridge_od_demand_basic():
    W = FakeWorld()
    W.add_link("L_origin_a", "N1", "N2")
    W.add_link("L_origin_b", "N3", "N4")
    W.add_link("L_dest", "N5", "N6")

    osm_to_uxsim = {
        100: [FakeUXsimLinkRef("L_origin_a", 100)],
        101: [FakeUXsimLinkRef("L_origin_b", 101)],
        500: [FakeUXsimLinkRef("L_dest", 500)],
    }

    od_df = pd.DataFrame([
        {"origin_osm_way_id": 100, "destination_osm_way_id": 500,
         "day_type_code": 1, "day_part_code": 2, "od_volume": 100.0},
        {"origin_osm_way_id": 101, "destination_osm_way_id": 500,
         "day_type_code": 1, "day_part_code": 2, "od_volume": 50.0},
        # Different day-part — should be filtered out:
        {"origin_osm_way_id": 100, "destination_osm_way_id": 500,
         "day_type_code": 1, "day_part_code": 4, "od_volume": 10.0},
        # Unknown OSM way — should be counted as skipped:
        {"origin_osm_way_id": 999, "destination_osm_way_id": 500,
         "day_type_code": 1, "day_part_code": 2, "od_volume": 5.0},
    ])

    summary = apply_bridge_od_demand(
        W, od_df, osm_to_uxsim, day_type_code=1, day_part_code=2,
    )
    assert summary.n_demands_added == 2
    assert summary.n_pairs_skipped_no_match == 1
    assert summary.n_origin_ways_resolved == 2
    assert summary.n_destination_ways_resolved == 1

    # Demands should use end_node of origin link and start_node of dest link.
    demands = {(d["orig"], d["dest"], d["volume"]) for d in W.demands}
    assert ("N2", "N5", 100.0) in demands
    assert ("N4", "N5", 50.0) in demands


def test_apply_bridge_od_demand_scale_and_zero_volume():
    W = FakeWorld()
    W.add_link("L_o", "A", "B")
    W.add_link("L_d", "C", "D")
    osm_to_uxsim = {
        1: [FakeUXsimLinkRef("L_o", 1)],
        2: [FakeUXsimLinkRef("L_d", 2)],
    }
    od_df = pd.DataFrame([
        {"origin_osm_way_id": 1, "destination_osm_way_id": 2,
         "day_type_code": 1, "day_part_code": 2, "od_volume": 10.0},
        # Zero-volume row should not produce a demand.
        {"origin_osm_way_id": 1, "destination_osm_way_id": 2,
         "day_type_code": 1, "day_part_code": 2, "od_volume": 0.0},
    ])
    summary = apply_bridge_od_demand(
        W, od_df, osm_to_uxsim, demand_scale=2.5, day_type_code=1, day_part_code=2,
    )
    assert summary.n_demands_added == 1
    assert W.demands[0]["volume"] == 25.0


def test_apply_bridge_od_demand_no_rows():
    W = FakeWorld()
    od_df = pd.DataFrame(columns=[
        "origin_osm_way_id", "destination_osm_way_id",
        "day_type_code", "day_part_code", "od_volume",
    ])
    summary = apply_bridge_od_demand(W, od_df, {})
    assert summary.n_demands_added == 0
    assert "No OD rows" in (summary.notes[0] if summary.notes else "")
