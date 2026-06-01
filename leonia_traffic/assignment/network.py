"""Build a NetworkX DiGraph from the UXsim (nodes, links) tuple.

The UXsim importer stores:

* ``nodes`` — list of ``[id, lon, lat]``
* ``links`` — list of ``[name, from_id, to_id, lanes, free_flow_speed_ms, length_deg]``

``length_deg`` is in WGS84 degrees and is converted to meters using the
same ``coef_degree_to_meter`` constant UXsim uses internally (see
``leonia_traffic.config.SIM_DEFAULTS``). The graph this module produces
is consumed by :mod:`leonia_traffic.assignment.assignment` for static
User-Equilibrium assignment.

Edge attributes
---------------

* ``osm_way_id`` — parsed from the UXsim link name (may be ``None``).
* ``length_m`` — link length in meters.
* ``free_flow_speed_ms`` — free-flow speed in m/s.
* ``free_flow_time_s`` — ``length_m / free_flow_speed_ms``.
* ``lanes`` — physical lane count (min 1).
* ``capacity_vph`` — directional capacity, ``lanes *
  DEFAULT_LANE_CAPACITY_VPH``. Per-lane capacity can be overridden by
  road class in :func:`build_assignment_graph`.
* ``link_name`` — original UXsim link name (kept for cross-engine joins).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx

from leonia_traffic.config import SIM_DEFAULTS
from leonia_traffic.network.osm_builder import parse_uxsim_link_name

# Highway Capacity Manual–style defaults. Per-lane vehicles-per-hour at
# capacity for a typical urban facility. These are intentionally
# conservative defaults; users can override via ``lane_capacity_vph``.
DEFAULT_LANE_CAPACITY_VPH: float = 800.0


@dataclass(frozen=True)
class AssignmentEdge:
    """Public view of a single edge as consumed by the assignment."""

    u: str
    v: str
    osm_way_id: int | None
    link_name: str
    length_m: float
    free_flow_speed_ms: float
    free_flow_time_s: float
    lanes: int
    capacity_vph: float


def build_assignment_graph(
    nodes: list,
    links: list,
    *,
    lane_capacity_vph: float = DEFAULT_LANE_CAPACITY_VPH,
    capacity_overrides: Mapping[str, float] | None = None,
) -> nx.DiGraph:
    """Return a NetworkX DiGraph ready for User-Equilibrium assignment.

    Parameters
    ----------
    nodes, links
        The tuple returned by
        :func:`leonia_traffic.network.osm_builder.build_or_load_network`.
    lane_capacity_vph
        Per-lane vehicles-per-hour at capacity. Used unless a more
        specific override is supplied. Default 800 vph (urban facility).
    capacity_overrides
        Optional ``{road_class: capacity_vph_per_lane}`` mapping, where
        ``road_class`` is the leading token of the UXsim link name
        (e.g. ``"residential"``). Unmatched links use
        ``lane_capacity_vph``. Road class isn't always present in the
        UXsim link name so this is best-effort.

    Notes
    -----
    The graph is directed. UXsim's post-processing already encodes both
    directions of a bidirectional road as separate links (the reverse
    direction's name ends in ``-reverse``), so we add edges as-is.
    """
    coef = SIM_DEFAULTS.coef_degree_to_meter
    capacity_overrides = capacity_overrides or {}

    G = nx.DiGraph()

    # Add nodes with WGS84 coordinates so downstream code (nearest-node
    # lookup, mapping) can use them directly.
    for n in nodes:
        node_id = str(n[0])
        G.add_node(node_id, lon=float(n[1]), lat=float(n[2]))

    for link in links:
        name = str(link[0])
        u = str(link[1])
        v = str(link[2])
        lanes = max(1, int(link[3]))
        speed_ms = max(0.1, float(link[4]))
        length_m = max(1.0, float(link[5]) * coef) if len(link) >= 6 else 1.0

        _, osm_id, _ = parse_uxsim_link_name(name)

        # Capacity: best-effort road-class override, else default.
        cap_per_lane = lane_capacity_vph
        for cls, override in capacity_overrides.items():
            if cls and cls.lower() in name.lower():
                cap_per_lane = float(override)
                break

        capacity = max(50.0, lanes * cap_per_lane)
        free_flow_time_s = length_m / speed_ms

        # Multiple UXsim links can share an (u, v) pair after the
        # node-merging post-process. NetworkX DiGraph keeps only the
        # last; merge by summing capacities and lanes and taking the
        # min free-flow time (parallel-edge approximation).
        if G.has_edge(u, v):
            existing = G[u][v]
            existing["lanes"] += lanes
            existing["capacity_vph"] += capacity
            existing["free_flow_time_s"] = min(
                existing["free_flow_time_s"], free_flow_time_s
            )
            existing["length_m"] = max(existing["length_m"], length_m)
            # Keep the first osm_way_id / link_name we saw.
            continue

        G.add_edge(
            u,
            v,
            link_name=name,
            osm_way_id=osm_id,
            length_m=length_m,
            free_flow_speed_ms=speed_ms,
            free_flow_time_s=free_flow_time_s,
            lanes=lanes,
            capacity_vph=capacity,
        )

    return G


def graph_edges_summary(G: nx.DiGraph) -> dict:
    """Quick diagnostic — node / edge counts + capacity stats."""
    caps = [d["capacity_vph"] for _, _, d in G.edges(data=True)]
    speeds = [d["free_flow_speed_ms"] for _, _, d in G.edges(data=True)]
    lengths = [d["length_m"] for _, _, d in G.edges(data=True)]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_edges_with_osm_id": sum(
            1 for _, _, d in G.edges(data=True) if d.get("osm_way_id") is not None
        ),
        "total_length_km": sum(lengths) / 1000.0,
        "min_capacity_vph": min(caps) if caps else 0.0,
        "max_capacity_vph": max(caps) if caps else 0.0,
        "min_speed_ms": min(speeds) if speeds else 0.0,
        "max_speed_ms": max(speeds) if speeds else 0.0,
    }


__all__ = [
    "AssignmentEdge",
    "DEFAULT_LANE_CAPACITY_VPH",
    "build_assignment_graph",
    "graph_edges_summary",
]
