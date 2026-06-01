"""Frank–Wolfe User-Equilibrium static traffic assignment.

Implements the textbook algorithm:

1. Initial all-or-nothing assignment using free-flow times.
2. Iteration: compute current edge travel times via BPR, find shortest
   paths under those times, do another all-or-nothing assignment.
3. Line-search step size ``λ ∈ [0, 1]`` minimising the Beckmann
   objective ``∫₀ˣ t(s) ds`` summed over edges.
4. Repeat until the relative gap is below ``rel_gap`` or
   ``max_iter`` is reached.

Returns per-edge assigned volume (veh/h) and congested travel time. Edge
flows are keyed by ``(u, v)`` so they can be re-attached to the input
graph or aggregated by ``osm_way_id``.

Reference: Sheffi (1985), *Urban Transportation Networks*, ch. 5.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# BPR coefficients — standard values for urban networks.
BPR_ALPHA: float = 0.15
BPR_BETA: float = 4.0


def bpr_travel_time(
    flow_vph: float,
    free_flow_time_s: float,
    capacity_vph: float,
    *,
    alpha: float = BPR_ALPHA,
    beta: float = BPR_BETA,
) -> float:
    """Bureau of Public Roads volume-delay function.

        t(x) = t_ff * (1 + alpha * (x / c) ** beta)
    """
    if capacity_vph <= 0:
        return free_flow_time_s
    return free_flow_time_s * (1.0 + alpha * (flow_vph / capacity_vph) ** beta)


@dataclass
class AssignmentResult:
    """Output of :func:`run_ue`."""

    edges: pd.DataFrame      # one row per (u, v) with flow + congested time
    n_iterations: int
    converged: bool
    final_gap: float
    elapsed_s: float
    skipped_od_pairs: int = 0
    history: list[dict] = field(default_factory=list)

    def as_geodataframe(self, G: nx.DiGraph):
        """Return a GeoDataFrame keyed by (u, v) with LineString geometry."""
        import geopandas as gpd
        from shapely.geometry import LineString

        rows = []
        for _, r in self.edges.iterrows():
            u, v = r["u"], r["v"]
            if u not in G.nodes or v not in G.nodes:
                continue
            a = G.nodes[u]
            b = G.nodes[v]
            if "lon" not in a or "lon" not in b:
                continue
            rows.append({
                **r.to_dict(),
                "geometry": LineString(
                    [(a["lon"], a["lat"]), (b["lon"], b["lat"])]
                ),
            })
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def by_osm_way(self) -> pd.DataFrame:
        """Aggregate edge flows to OSM way granularity.

        For roads with parallel UXsim links sharing one ``osm_way_id``
        we sum flows and take the mean congested time weighted by free-
        flow time (proxy for length).
        """
        with_osm = self.edges.dropna(subset=["osm_way_id"]).copy()
        if with_osm.empty:
            return with_osm

        def _agg(group: pd.DataFrame) -> pd.Series:
            flow = group["assigned_volume_vph"].sum()
            weights = group["free_flow_time_s"]
            denom = weights.sum() if weights.sum() > 0 else 1.0
            cong_t = (group["congested_time_s"] * weights).sum() / denom
            ff_t = (group["free_flow_time_s"] * weights).sum() / denom
            voc = (group["voc"] * weights).sum() / denom
            return pd.Series({
                "assigned_volume_vph": flow,
                "congested_time_s": cong_t,
                "free_flow_time_s": ff_t,
                "voc": voc,
                "n_edges": len(group),
            })

        agg = with_osm.groupby("osm_way_id", as_index=False).apply(
            _agg, include_groups=False
        )
        # Newer pandas returns a frame already; older versions need reset.
        if "osm_way_id" not in agg.columns:
            agg = agg.reset_index()
        agg["osm_way_id"] = agg["osm_way_id"].astype(int)
        return agg


def _all_or_nothing(
    G: nx.DiGraph,
    demand: dict[tuple[str, str], float],
    weight_attr: str,
) -> tuple[dict[tuple[str, str], float], int]:
    """All-or-nothing: assign each OD's flow to its shortest path.

    Returns (flow_dict, n_skipped) where flow_dict maps (u, v) → flow.
    Pairs whose origin/destination is missing from the graph or is
    unreachable are skipped silently and counted.
    """
    flow: dict[tuple[str, str], float] = defaultdict(float)
    skipped = 0
    # Group demand by origin to reuse Dijkstra's single-source predecessor
    # tree across destinations.
    by_origin: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (o, d), vol in demand.items():
        by_origin[o].append((d, vol))

    for origin, dests in by_origin.items():
        if origin not in G:
            skipped += len(dests)
            continue
        try:
            lengths, paths = nx.single_source_dijkstra(
                G, origin, weight=weight_attr
            )
        except nx.NetworkXError:
            skipped += len(dests)
            continue
        for dest, vol in dests:
            if dest not in paths:
                skipped += 1
                continue
            path = paths[dest]
            for u, v in zip(path[:-1], path[1:]):
                flow[(u, v)] += vol
    return flow, skipped


def _bpr_objective(edge_flows: np.ndarray, ff_times: np.ndarray,
                   capacities: np.ndarray,
                   alpha: float, beta: float) -> float:
    """Beckmann integral ∫₀ˣ t(s) ds for the line search."""
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    ratio = edge_flows / safe_cap
    integral = ff_times * (
        edge_flows + alpha / (beta + 1.0) * safe_cap * ratio ** (beta + 1.0)
    )
    return float(integral.sum())


def run_ue(
    G: nx.DiGraph,
    demand: dict[tuple[str, str], float],
    *,
    max_iter: int = 30,
    rel_gap: float = 1e-3,
    alpha: float = BPR_ALPHA,
    beta: float = BPR_BETA,
    line_search_steps: int = 21,
    verbose: bool = False,
) -> AssignmentResult:
    """Run Frank–Wolfe User-Equilibrium assignment.

    Parameters
    ----------
    G
        Directed graph from :func:`build_assignment_graph` with
        ``free_flow_time_s`` and ``capacity_vph`` edge attributes.
    demand
        ``{(origin_node, dest_node): trips_per_hour}``.
    max_iter
        Cap on Frank–Wolfe iterations.
    rel_gap
        Convergence threshold on relative gap
        ``(UB - LB) / LB``. Typical UE results converge in 10–20
        iterations to ``1e-3`` for a network Leonia's size.
    line_search_steps
        Grid resolution for the 1-D line search on ``λ``. 21 = 5 %
        resolution; that's plenty for this size of network.

    Returns
    -------
    AssignmentResult
    """
    start = time.perf_counter()
    if not demand:
        logger.warning("run_ue called with empty demand")

    # Materialise edges as parallel numpy arrays keyed by (u, v).
    edge_keys: list[tuple[str, str]] = []
    ff_times: list[float] = []
    capacities: list[float] = []
    osm_ids: list[int | None] = []
    link_names: list[str] = []
    lengths_m: list[float] = []
    ff_speeds: list[float] = []
    lanes: list[int] = []
    for u, v, d in G.edges(data=True):
        edge_keys.append((u, v))
        ff_times.append(float(d["free_flow_time_s"]))
        capacities.append(float(d["capacity_vph"]))
        osm_ids.append(d.get("osm_way_id"))
        link_names.append(d.get("link_name", ""))
        lengths_m.append(float(d["length_m"]))
        ff_speeds.append(float(d["free_flow_speed_ms"]))
        lanes.append(int(d["lanes"]))

    n_edges = len(edge_keys)
    ff_arr = np.array(ff_times, dtype=float)
    cap_arr = np.array(capacities, dtype=float)
    edge_index = {key: i for i, key in enumerate(edge_keys)}

    # --- Initial all-or-nothing under free-flow times ----------------
    # Seed every edge's weight attribute with the free-flow time.
    for (u, v), t in zip(edge_keys, ff_times):
        G[u][v]["_w"] = t

    init_flow, skipped = _all_or_nothing(G, demand, "_w")
    x = np.zeros(n_edges)
    for key, vol in init_flow.items():
        if key in edge_index:
            x[edge_index[key]] = vol

    history: list[dict] = []
    converged = False
    final_gap = float("inf")

    for it in range(1, max_iter + 1):
        # Edge travel times under current flow.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(cap_arr > 0, x / cap_arr, 0.0)
            t_x = ff_arr * (1.0 + alpha * ratio ** beta)

        # Update graph weights.
        for i, (u, v) in enumerate(edge_keys):
            G[u][v]["_w"] = float(t_x[i])

        # Auxiliary all-or-nothing under t(x).
        aux_flow, skipped_iter = _all_or_nothing(G, demand, "_w")
        y = np.zeros(n_edges)
        for key, vol in aux_flow.items():
            if key in edge_index:
                y[edge_index[key]] = vol

        # Gap (Beckmann lower bound vs current upper bound).
        ub = float((t_x * x).sum())
        lb = float((t_x * y).sum())
        gap = (ub - lb) / max(lb, 1e-9)
        final_gap = gap

        # Line search on λ in [0, 1] minimising Beckmann objective.
        lambdas = np.linspace(0.0, 1.0, line_search_steps)
        best_lam = 0.0
        best_obj = float("inf")
        for lam in lambdas:
            trial = (1.0 - lam) * x + lam * y
            obj = _bpr_objective(trial, ff_arr, cap_arr, alpha, beta)
            if obj < best_obj:
                best_obj = obj
                best_lam = float(lam)

        x = (1.0 - best_lam) * x + best_lam * y

        history.append({
            "iter": it,
            "gap": gap,
            "lambda": best_lam,
            "beckmann": best_obj,
        })

        if verbose:
            logger.info(
                "FW iter=%d gap=%.4e lambda=%.3f obj=%.3e",
                it, gap, best_lam, best_obj,
            )

        if gap < rel_gap:
            converged = True
            break

    # Final per-edge travel times under the equilibrium flow.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cap_arr > 0, x / cap_arr, 0.0)
        t_x = ff_arr * (1.0 + alpha * ratio ** beta)

    # Tidy up scratch weight.
    for u, v in edge_keys:
        if "_w" in G[u][v]:
            del G[u][v]["_w"]

    edges_df = pd.DataFrame({
        "u": [k[0] for k in edge_keys],
        "v": [k[1] for k in edge_keys],
        "link_name": link_names,
        "osm_way_id": osm_ids,
        "length_m": lengths_m,
        "free_flow_speed_ms": ff_speeds,
        "lanes": lanes,
        "capacity_vph": cap_arr,
        "free_flow_time_s": ff_arr,
        "assigned_volume_vph": x,
        "congested_time_s": t_x,
        "voc": np.where(cap_arr > 0, x / cap_arr, 0.0),
    })

    return AssignmentResult(
        edges=edges_df,
        n_iterations=len(history),
        converged=converged,
        final_gap=final_gap,
        elapsed_s=time.perf_counter() - start,
        skipped_od_pairs=skipped,
        history=history,
    )


__all__ = [
    "AssignmentResult",
    "BPR_ALPHA",
    "BPR_BETA",
    "bpr_travel_time",
    "run_ue",
]
