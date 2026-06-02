"""Engine-neutral declarative mitigation-scenario DSL.

A scenario is a small, serialisable description of a network change the
borough might make to a set of OSM ways — close them, calm them with
speed humps, convert them to one-way, or drop a lane. The objects here
are **pure data carriers**: they hold the targeted ``osm_way_ids`` and a
few parameters and carry no simulation-engine logic.

Each engine adapts these to its own world:

* SUMO applies them at *runtime* through libsumo edge controls — see
  :mod:`leonia_traffic.sumo.scenarios_sumo`.

Keeping the DSL here (rather than inside a specific engine package) lets
callers describe a single scenario list and hand it to whichever engine
is in use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Closure",
    "LaneReduction",
    "OneWayConversion",
    "Scenario",
    "SpeedHumpCalming",
]


@dataclass
class Scenario:
    """Base class for all scenarios. Carries a human-readable ``name``."""

    name: str = "scenario"


@dataclass
class OneWayConversion(Scenario):
    """Convert a two-way OSM way to one-way.

    Keeps only the direction whose bearing is within ``tolerance_deg`` of
    ``allowed_bearing_deg``; the other-direction edge is removed/blocked
    by the engine adapter.
    """

    osm_way_ids: list[int] = field(default_factory=list)
    allowed_bearing_deg: float = 0.0
    tolerance_deg: float = 90.0


@dataclass
class SpeedHumpCalming(Scenario):
    """Reduce free-flow speed on listed OSM ways to mimic traffic calming."""

    osm_way_ids: list[int] = field(default_factory=list)
    free_flow_speed_factor: float = 0.5
    min_free_flow_speed_ms: float = 4.5  # ~10 mph


@dataclass
class Closure(Scenario):
    """Fully close listed OSM ways (remove all matching edges)."""

    osm_way_ids: list[int] = field(default_factory=list)


@dataclass
class LaneReduction(Scenario):
    """Reduce ``lanes`` to ``target_lanes`` on listed OSM ways."""

    osm_way_ids: list[int] = field(default_factory=list)
    target_lanes: int = 1
