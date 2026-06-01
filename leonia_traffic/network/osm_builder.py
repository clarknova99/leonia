"""Build a UXsim ``World`` from OpenStreetMap data for the Leonia study area.

Layered on top of ``uxsim.OSMImporter`` (which is officially experimental).
Adds:

  - bbox + custom_filter defaults tuned to a residential study area
  - an optional manual-overrides YAML for surgically deleting or relaxing
    links/nodes that the auto-import gets wrong
  - pickle-based caching of the postprocessed (nodes, links) tuple so we
    don't hit OSMnx on every run
  - parsing of UXsim's encoded link names back into ``(road_name, osm_way_id)``
    pairs (needed by ``calibration_match.py``)
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from uxsim import World
from uxsim.OSMImporter import OSMImporter

from leonia_traffic.config import (
    DATA_NETWORK_DIR,
    LEONIA_BBOX_WGS84,
    SIM_DEFAULTS,
)

logger = logging.getLogger(__name__)


# OSMnx >=2 expects bbox = (west, south, east, north) (left, bottom, right, top).
def _to_osmnx_bbox(wgs: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = wgs
    return (minx, miny, maxx, maxy)


# Highways relevant to a residential study area. We deliberately include
# ``residential`` and ``unclassified`` because cut-through traffic lives
# on them; we exclude ``service`` and ``living_street`` because they add
# clutter and rarely carry through traffic.
DEFAULT_HIGHWAY_FILTER = (
    '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link'
    '|secondary|secondary_link|tertiary|tertiary_link|residential|unclassified"]'
)


# UXsim encodes link names as "<street name>-<osm way id>". We append
# ``#N`` for uniqueness disambiguation in :func:`_postprocess_network`,
# and detect ``-reverse`` from optional bidirectional expansion.
_LINK_NAME_PATTERN = re.compile(
    r"^(?P<name>.*?)-(?P<osm_id>\d+)(?:-reverse)?(?:#\d+)?$"
)


def parse_uxsim_link_name(uxsim_name: str) -> tuple[str, int | None, bool]:
    """Return ``(road_name, osm_way_id, is_reverse)`` for a UXsim link name."""
    if not uxsim_name:
        return ("", None, False)
    base = uxsim_name.split("#")[0]
    is_reverse = base.endswith("-reverse")
    m = _LINK_NAME_PATTERN.match(uxsim_name)
    if not m:
        return (uxsim_name, None, is_reverse)
    return (m.group("name"), int(m.group("osm_id")), is_reverse)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


@dataclass
class NetworkOverrides:
    """Manual fixes applied to the postprocessed (nodes, links) lists.

    YAML schema (every key is optional):

    ```yaml
    delete_links:
      - osm_way_id: 11586338      # remove every UXsim link whose name encodes this id
      - name_contains: "Service"   # case-insensitive substring match on link name
    delete_nodes:
      - id: "1234567"              # remove the node and any links that touch it
    set_link_attrs:
      - osm_way_id: 11586338
        free_flow_speed: 8.33      # m/s
        lanes: 1
    add_links:                      # for hand-built corrections
      - name: "Custom-9999"
        from: "1234567"
        to:   "7654321"
        lanes: 1
        maxspeed: 11.11             # m/s
        length: 200.0               # meters
    ```
    """

    delete_links: list[dict] = field(default_factory=list)
    delete_nodes: list[dict] = field(default_factory=list)
    set_link_attrs: list[dict] = field(default_factory=list)
    add_links: list[dict] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "NetworkOverrides":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            delete_links=list(data.get("delete_links", []) or []),
            delete_nodes=list(data.get("delete_nodes", []) or []),
            set_link_attrs=list(data.get("set_link_attrs", []) or []),
            add_links=list(data.get("add_links", []) or []),
        )

    def is_empty(self) -> bool:
        return not (self.delete_links or self.delete_nodes or self.set_link_attrs or self.add_links)


def _link_matches_rule(link: list, rule: dict) -> bool:
    name, osmid, _ = parse_uxsim_link_name(link[0])
    if "osm_way_id" in rule:
        if osmid is None or osmid != int(rule["osm_way_id"]):
            return False
    if "name_contains" in rule:
        if rule["name_contains"].lower() not in (link[0] or "").lower():
            return False
    if "name_exact" in rule:
        if link[0] != rule["name_exact"]:
            return False
    return True


def apply_overrides(
    nodes: list, links: list, ov: NetworkOverrides
) -> tuple[list, list]:
    """Apply an overrides bundle. Returns new (nodes, links) lists."""
    if ov.is_empty():
        return nodes, links

    nodes = [list(n) for n in nodes]
    links = [list(l) for l in links]

    delete_node_ids = {str(r["id"]) for r in ov.delete_nodes if "id" in r}

    new_links = []
    for link in links:
        if any(_link_matches_rule(link, r) for r in ov.delete_links):
            continue
        if str(link[1]) in delete_node_ids or str(link[2]) in delete_node_ids:
            continue
        for rule in ov.set_link_attrs:
            if _link_matches_rule(link, rule):
                if "lanes" in rule:
                    link[3] = int(rule["lanes"])
                if "free_flow_speed" in rule:
                    link[4] = float(rule["free_flow_speed"])
                if "length" in rule and len(link) >= 6:
                    link[5] = float(rule["length"])
        new_links.append(link)

    for add in ov.add_links:
        new_links.append(
            [
                str(add.get("name", f"custom-{len(new_links)}")),
                str(add["from"]),
                str(add["to"]),
                int(add.get("lanes", 1)),
                float(add.get("maxspeed", 11.11)),
                float(add.get("length", 100.0)),
            ]
        )

    nodes = [n for n in nodes if str(n[0]) not in delete_node_ids]
    used = {l[1] for l in new_links} | {l[2] for l in new_links}
    nodes = [n for n in nodes if str(n[0]) in {str(u) for u in used}]

    return nodes, new_links


# ---------------------------------------------------------------------------
# Build / cache the (nodes, links) tuple
# ---------------------------------------------------------------------------


_CACHE_PATH = DATA_NETWORK_DIR / "leonia_osm_network.pkl"


def _backfill_missing_endpoints(
    nodes: list,
    links: list,
    bbox: tuple[float, float, float, float],
    highway_filter: str,
) -> list:
    """Ensure every link endpoint is present in ``nodes``.

    ``import_osm_data`` populates the internal ``nodes`` dict only with
    the start node of each retained edge. Postprocessing then looks up
    *both* endpoints — so any end-only node raises ``KeyError``. We
    re-query OSMnx for the original graph (the same one
    ``import_osm_data`` just pulled) and fill in the missing entries.
    """
    node_map: dict = {n[0]: list(n) for n in nodes}

    missing = set()
    for link in links:
        for endpoint in (link[1], link[2]):
            if endpoint not in node_map:
                missing.add(endpoint)
    if not missing:
        return [list(v) for v in node_map.values()]

    logger.info("Backfilling %d missing node positions from OSMnx", len(missing))
    import osmnx as ox

    G = ox.graph.graph_from_bbox(
        bbox=bbox, network_type="drive", custom_filter=highway_filter
    )
    for n in missing:
        if n in G.nodes:
            nd = G.nodes[n]
            node_map[n] = [n, nd["x"], nd["y"]]
    still_missing = missing - set(node_map.keys())
    if still_missing:
        logger.warning(
            "%d node IDs could not be recovered from OSMnx and will be "
            "dropped from links (sample: %s)",
            len(still_missing),
            list(still_missing)[:5],
        )
        # Drop links that reference unrecoverable endpoints.
        links[:] = [l for l in links if l[1] in node_map and l[2] in node_map]
    return [list(v) for v in node_map.values()]


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _postprocess_network(
    nodes: list,
    links: list,
    *,
    node_merge_threshold: float,
    node_merge_iteration: int = 5,
    enforce_bidirectional: bool = False,
) -> tuple[list, list]:
    """Merge nodes joined by very-short links; emit a clean (nodes, links).

    The output ``links`` are 6-tuples ``[name, from, to, lanes, maxspeed,
    length]`` measured in WGS84 degrees, matching what
    ``OSMImporter.osm_network_to_World`` expects.

    Algorithm:
      1. Build a union-find of nodes connected by links shorter than
         ``node_merge_threshold``.
      2. Pick a canonical representative per cluster (lowest id).
      3. Rewrite every link to use canonical endpoints.
      4. Drop self-loops; keep at most one link per (from, to) pair
         (preferring the longest, which is the most informative).
      5. Drop isolated nodes.
      6. Optionally add reverse links.

    Repeating for ``node_merge_iteration`` cycles lets long chains of
    short connectors collapse cleanly.
    """
    cur_nodes = [list(n) for n in nodes]
    cur_links = [list(l) for l in links]

    for _iter in range(node_merge_iteration):
        pos = {n[0]: (n[1], n[2]) for n in cur_nodes}

        parent: dict = {n[0]: n[0] for n in cur_nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Prefer the smaller id as the root for determinism.
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

        n_short = 0
        for link in cur_links:
            a, b = link[1], link[2]
            if a not in pos or b not in pos:
                continue
            if _euclid(pos[a], pos[b]) <= node_merge_threshold:
                union(a, b)
                n_short += 1

        if n_short == 0:
            break

        # Rewrite endpoints.
        new_pos: dict = {}
        for n in cur_nodes:
            root = find(n[0])
            new_pos.setdefault(root, pos[root])

        edge_map: dict[tuple, list] = {}
        for link in cur_links:
            a, b = find(link[1]), find(link[2])
            if a == b:
                continue
            length = _euclid(new_pos[a], new_pos[b])
            key = (a, b)
            new_record = [link[0], a, b, link[3], link[4], length]
            existing = edge_map.get(key)
            if existing is None or length > existing[5]:
                edge_map[key] = new_record

        cur_links = list(edge_map.values())
        # Drop now-isolated nodes.
        used = {l[1] for l in cur_links} | {l[2] for l in cur_links}
        cur_nodes = [[k, *new_pos[k]] for k in new_pos if k in used]

    # Make sure every link has the 6-tuple shape (length field present)
    # and that link names are globally unique. We append a stable
    # disambiguator ``#<from>-<to>`` so the OSM way ID encoded in the
    # original name stays intact (parse_uxsim_link_name will strip the
    # trailing token before looking up the OSM ID).
    pos_final = {n[0]: (n[1], n[2]) for n in cur_nodes}
    name_seen: dict[str, int] = {}
    final_links: list[list] = []
    for link in cur_links:
        if link[1] not in pos_final or link[2] not in pos_final:
            continue
        base_name = str(link[0])
        suffix_idx = name_seen.get(base_name, 0)
        name_seen[base_name] = suffix_idx + 1
        unique_name = base_name if suffix_idx == 0 else f"{base_name}#{suffix_idx}"
        if len(link) >= 6:
            final_links.append(
                [unique_name, link[1], link[2], link[3], link[4], link[5]]
            )
        else:
            length = _euclid(pos_final[link[1]], pos_final[link[2]])
            final_links.append(
                [unique_name, link[1], link[2], link[3], link[4], length]
            )

    if enforce_bidirectional:
        seen = {(l[1], l[2]) for l in final_links}
        reverse_links = []
        for l in final_links:
            if (l[2], l[1]) not in seen:
                reverse_links.append([f"{l[0]}-reverse", l[2], l[1], l[3], l[4], l[5]])
                seen.add((l[2], l[1]))
        final_links.extend(reverse_links)

    logger.info(
        "Postprocessed (custom): %d nodes, %d links", len(cur_nodes), len(final_links)
    )
    return cur_nodes, final_links


@dataclass
class OSMBuildConfig:
    bbox_wgs84: tuple[float, float, float, float] = LEONIA_BBOX_WGS84
    highway_filter: str = DEFAULT_HIGHWAY_FILTER
    node_merge_threshold_deg: float = 0.0005   # ~55 m
    node_merge_iteration: int = 5
    enforce_bidirectional: bool = False
    cache_path: Path = _CACHE_PATH
    overrides_path: Path | None = None


def build_or_load_network(
    cfg: OSMBuildConfig | None = None, *, rebuild: bool = False
) -> tuple[list, list, OSMBuildConfig]:
    """Build the postprocessed (nodes, links) tuple, caching to disk."""
    cfg = cfg or OSMBuildConfig()

    if cfg.cache_path.exists() and not rebuild:
        try:
            with cfg.cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            nodes = payload["nodes"]
            links = payload["links"]
            logger.info("Loaded cached OSM network: %d nodes, %d links",
                        len(nodes), len(links))
            return nodes, links, cfg
        except Exception as exc:  # pragma: no cover - corrupt cache fallback
            logger.warning("Cache load failed (%s); rebuilding", exc)

    bbox = _to_osmnx_bbox(cfg.bbox_wgs84)
    logger.info("Downloading OSM network for bbox=%s", bbox)
    nodes, links = OSMImporter.import_osm_data(
        bbox=bbox, custom_filter=cfg.highway_filter
    )
    logger.info("Raw OSM size: %d nodes, %d links", len(nodes), len(links))

    # UXsim's own ``osm_network_postprocessing`` has a known bug where
    # multi-iteration node-merging can leave links pointing at deleted
    # node IDs (https://github.com/toruseo/UXsim/issues — the importer is
    # marked experimental upstream). We use a small, deterministic
    # replacement that does the same intent: backfill missing endpoint
    # positions, merge nodes joined by very-short links, and emit links
    # with the 6-tuple layout expected by ``osm_network_to_World``.
    nodes = _backfill_missing_endpoints(nodes, links, bbox, cfg.highway_filter)
    nodes, links = _postprocess_network(
        nodes,
        links,
        node_merge_threshold=cfg.node_merge_threshold_deg,
        node_merge_iteration=cfg.node_merge_iteration,
        enforce_bidirectional=cfg.enforce_bidirectional,
    )
    logger.info("Postprocessed OSM size: %d nodes, %d links", len(nodes), len(links))

    if cfg.overrides_path is not None:
        ov = NetworkOverrides.from_yaml(cfg.overrides_path)
        if not ov.is_empty():
            nodes, links = apply_overrides(nodes, links, ov)
            logger.info("After overrides: %d nodes, %d links", len(nodes), len(links))

    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.cache_path.open("wb") as fh:
        pickle.dump({"nodes": nodes, "links": links, "config": cfg}, fh)

    return nodes, links, cfg


# ---------------------------------------------------------------------------
# Materialize a UXsim World
# ---------------------------------------------------------------------------


def make_world(
    *,
    name: str = "leonia",
    tmax: int = SIM_DEFAULTS.tmax_seconds,
    deltan: int = SIM_DEFAULTS.deltan,
    random_seed: int = SIM_DEFAULTS.random_seed,
    print_mode: int = 1,
    save_mode: int = 1,
    show_mode: int = 0,
) -> World:
    """Create a fresh UXsim ``World`` with consistent defaults."""
    return World(
        name=name,
        deltan=deltan,
        tmax=tmax,
        print_mode=print_mode,
        save_mode=save_mode,
        show_mode=show_mode,
        random_seed=random_seed,
    )


def world_from_osm(
    cfg: OSMBuildConfig | None = None,
    *,
    name: str = "leonia",
    tmax: int = SIM_DEFAULTS.tmax_seconds,
    deltan: int = SIM_DEFAULTS.deltan,
    rebuild_network: bool = False,
) -> tuple[World, list, list]:
    """End-to-end: build/load the network and load it into a fresh World."""
    nodes, links, cfg = build_or_load_network(cfg, rebuild=rebuild_network)
    W = make_world(name=name, tmax=tmax, deltan=deltan)
    OSMImporter.osm_network_to_World(
        W,
        nodes,
        links,
        default_jam_density=SIM_DEFAULTS.default_jam_density,
        coef_degree_to_meter=SIM_DEFAULTS.coef_degree_to_meter,
    )
    return W, nodes, links


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_network(nodes: list, links: list) -> dict:
    osm_ids = []
    deadend_in = {n[0]: 0 for n in nodes}
    deadend_out = {n[0]: 0 for n in nodes}
    for link in links:
        _, osmid, _ = parse_uxsim_link_name(link[0])
        if osmid is not None:
            osm_ids.append(osmid)
        deadend_out[link[1]] = deadend_out.get(link[1], 0) + 1
        deadend_in[link[2]] = deadend_in.get(link[2], 0) + 1
    isolated = sum(
        1 for n in nodes if deadend_in.get(n[0], 0) == 0 and deadend_out.get(n[0], 0) == 0
    )
    sinks = sum(1 for n in nodes if deadend_out.get(n[0], 0) == 0)
    sources = sum(1 for n in nodes if deadend_in.get(n[0], 0) == 0)
    return {
        "n_nodes": len(nodes),
        "n_links": len(links),
        "n_unique_osm_ways": len(set(osm_ids)),
        "n_isolated_nodes": isolated,
        "n_pure_sinks": sinks,
        "n_pure_sources": sources,
    }


# ---------------------------------------------------------------------------
# Congestion-derived link speed overrides (Pass B.2)
# ---------------------------------------------------------------------------


def apply_congestion_overrides(
    W: World,
    overrides_df,                 # pd.DataFrame, but pandas imported lazily
    *,
    osm_to_uxsim: dict[int, list] | None = None,
    zone_to_link_name: dict[str, str] | None = None,
    speed_col: str = "observed_speed_ms",
    min_speed_ms: float = 1.0,
    cache_path: Path | None = None,
) -> dict:
    """Override UXsim link free-flow speeds with observed congestion data.

    Consumes the DataFrame produced by
    :func:`leonia_traffic.analysis.congestion.link_speed_overrides`
    (one row per ``osm_way_id`` with ``observed_speed_ms``). For every
    UXsim link whose OSM way ID is in the override table, the link's
    free-flow speed is set to the observed value.

    Parameters
    ----------
    W
        The UXsim ``World`` to mutate.
    overrides_df
        Long-format DataFrame from ``link_speed_overrides``. Must
        contain ``osm_way_id`` and ``observed_speed_ms``.
    osm_to_uxsim
        Optional precomputed mapping from
        :func:`leonia_traffic.network.calibration_match.build_osm_to_uxsim_index`.
        If omitted, the function iterates over ``W.LINKS`` and parses
        each link name on the fly.
    speed_col
        Which override column to apply. Defaults to
        ``observed_speed_ms`` (already in m/s).
    min_speed_ms
        Speeds below this floor are clipped (UXsim warns/ignores
        non-positive speeds).
    cache_path
        If provided, the resolved override table (with the actually
        applied UXsim link names) is written to this parquet file.

    Returns
    -------
    dict
        Diagnostic counts: ``n_overrides_seen``, ``n_links_changed``,
        ``n_osm_ways_with_match``, ``n_osm_ways_without_match``.
    """
    import pandas as pd

    seen = len(overrides_df)
    counts = {
        "n_overrides_seen": int(seen),
        "n_links_changed": 0,
        "n_osm_ways_with_match": 0,
        "n_osm_ways_without_match": 0,
    }
    if seen == 0:
        return counts

    if osm_to_uxsim is None:
        index: dict[int, list[str]] = {}
        for link in getattr(W, "LINKS", []):
            _, osmid, _ = parse_uxsim_link_name(getattr(link, "name", ""))
            if osmid is None:
                continue
            index.setdefault(osmid, []).append(link.name)
    else:
        index = {k: [r.name for r in v] for k, v in osm_to_uxsim.items()}

    zone_to_link_name = zone_to_link_name or {}

    resolved_rows: list[dict] = []
    for _, row in overrides_df.iterrows():
        ow = row.get("osm_way_id")
        speed = float(row.get(speed_col, 0.0) or 0.0)
        if speed < min_speed_ms:
            speed = min_speed_ms

        link_names: list[str] = []
        if not pd.isna(ow):
            link_names = index.get(int(ow), [])

        if not link_names:
            # Spatial fallback: look up by zone name.
            zone_name = str(row.get("source_zone_name", "") or "")
            fallback = zone_to_link_name.get(zone_name)
            if fallback is not None:
                link_names = [fallback]

        if not link_names:
            counts["n_osm_ways_without_match"] += 1
            continue
        counts["n_osm_ways_with_match"] += 1

        for link_name in link_names:
            try:
                link = W.get_link(link_name)
                link.change_free_flow_speed(speed)
                counts["n_links_changed"] += 1
                resolved_rows.append({
                    "osm_way_id": (int(ow) if not pd.isna(ow) else None),
                    "uxsim_link_name": link_name,
                    "applied_speed_ms": speed,
                })
            except Exception as exc:  # pragma: no cover
                logger.debug("Skipping %s (%.2f m/s): %s", link_name, speed, exc)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_rows:
            pd.DataFrame(resolved_rows).to_parquet(cache_path)
            logger.info("Wrote %d applied overrides to %s", len(resolved_rows), cache_path)
        else:
            logger.warning("No overrides were applied; not writing %s", cache_path)

    return counts


__all__ = [
    "DEFAULT_HIGHWAY_FILTER",
    "NetworkOverrides",
    "OSMBuildConfig",
    "apply_congestion_overrides",
    "apply_overrides",
    "build_or_load_network",
    "make_world",
    "parse_uxsim_link_name",
    "summarize_network",
    "world_from_osm",
]
