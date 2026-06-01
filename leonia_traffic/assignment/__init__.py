"""Static traffic-assignment library for the Leonia notebooks.

Pure-Python (NetworkX) replacement for AequilibraE — AequilibraE has no
cp313 macOS-arm64 wheels and its source build needs OpenMP. This module
runs **User Equilibrium** assignment via Frank–Wolfe on the same
``(nodes, links)`` tuple that the UXsim baseline uses, so the
AequilibraE-flavoured notebooks are bit-identical in network topology to
the UXsim scripts under ``scripts/``.

Public API
----------

    from leonia_traffic.assignment import (
        build_assignment_graph,
        bridge_od_to_demand,
        run_ue,
        validate_against_streetscanner,
        apply_scenarios_to_graph,
    )

Workflow:

1. ``nodes, links, _ = build_or_load_network()`` from the existing
   ``leonia_traffic.network.osm_builder`` module.
2. ``G = build_assignment_graph(nodes, links)`` — NetworkX DiGraph with
   BPR-ready edge attributes (``length_m``, ``free_flow_time_s``,
   ``capacity_vph``, ``osm_way_id``, ``free_flow_speed_ms``).
3. ``demand = bridge_od_to_demand(bridge_od_df, zones_gdf, G)`` —
   ``{(o_node, d_node): trips_per_hour}``.
4. ``result = run_ue(G, demand)`` — returns ``AssignmentResult`` with
   per-edge volumes and travel times.
5. ``validate_against_streetscanner(result, segments_gdf)`` — GEH /
   RMSE / R² vs. observed Street Scanner volumes.

Scenarios:

    from leonia_traffic.simulation.scenarios import Closure
    new_nodes, new_links = apply_scenarios_to_graph(
        nodes, links, [Closure(osm_way_ids=[12345])]
    )
    G2 = build_assignment_graph(new_nodes, new_links)
    result2 = run_ue(G2, demand)
"""

from leonia_traffic.assignment.network import (
    AssignmentEdge,
    build_assignment_graph,
    DEFAULT_LANE_CAPACITY_VPH,
)
from leonia_traffic.assignment.od_matrices import (
    bridge_od_to_demand,
    map_bridge_od_to_graph_edges,
    map_zones_to_graph_nodes,
)
from leonia_traffic.assignment.assignment import (
    AssignmentResult,
    run_ue,
    bpr_travel_time,
)
from leonia_traffic.assignment.validation import (
    validate_against_streetscanner,
    ValidationStats,
)
from leonia_traffic.assignment.scenarios import (
    apply_scenarios_to_graph,
)

__all__ = [
    "AssignmentEdge",
    "AssignmentResult",
    "ValidationStats",
    "DEFAULT_LANE_CAPACITY_VPH",
    "build_assignment_graph",
    "bridge_od_to_demand",
    "map_bridge_od_to_graph_edges",
    "map_zones_to_graph_nodes",
    "run_ue",
    "bpr_travel_time",
    "validate_against_streetscanner",
    "apply_scenarios_to_graph",
]
