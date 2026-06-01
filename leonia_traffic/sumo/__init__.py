"""SUMO / libsumo runtime layer for the Leonia traffic simulation.

Public API (lazy-loaded — see ``__getattr__`` below):

- :class:`SumoRuntime` — interactive libsumo wrapper.
- :class:`DemandSource` — enum of supported demand builders.
- :func:`build_routes` — synthesise a SUMO ``<routes>`` XML from one
  or more :class:`DemandSource` values.
- :func:`load_osm_to_sumo_lookup`,
  :func:`load_sumo_edge_geometries`,
  :func:`load_meta_lookup`,
  :func:`spatial_resolve_zones` — shared OSM ↔ SUMO edge resolution
  helpers.
- :func:`score_sumo_run` — GEH against StreetLight Street Scanner.

Lazy importing keeps ``import leonia_traffic.sumo`` cheap and avoids
pulling ``libsumo`` (which loads a 50 MB native library) unless the
caller actually needs the runtime.
"""

from __future__ import annotations

__all__ = [
    "BRIDGE_OD_WINDOWS",
    "DemandSource",
    "SumoRuntime",
    "build_routes",
    "edges_for_osm_ways",
    "load_crash_points_if_available",
    "load_crash_segments_if_available",
    "load_meta_lookup",
    "load_osm_to_sumo_lookup",
    "load_sumo_edge_geometries",
    "score_sumo_run",
    "spatial_resolve_zones",
]


_LAZY = {
    "BRIDGE_OD_WINDOWS": ("demand_builder", "BRIDGE_OD_WINDOWS"),
    "DemandSource": ("demand_builder", "DemandSource"),
    "build_routes": ("demand_builder", "build_routes"),
    "edges_for_osm_ways": ("net_lookup", "edges_for_osm_ways"),
    "load_crash_points_if_available":
        ("visualizations", "load_crash_points_if_available"),
    "load_crash_segments_if_available":
        ("visualizations", "load_crash_segments_if_available"),
    "load_meta_lookup": ("net_lookup", "load_meta_lookup"),
    "load_osm_to_sumo_lookup": ("net_lookup", "load_osm_to_sumo_lookup"),
    "load_sumo_edge_geometries": ("net_lookup", "load_sumo_edge_geometries"),
    "score_sumo_run": ("scoring", "score_sumo_run"),
    "spatial_resolve_zones": ("net_lookup", "spatial_resolve_zones"),
    "SumoRuntime": ("runtime", "SumoRuntime"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule, attr = _LAZY[name]
    import importlib

    mod = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
