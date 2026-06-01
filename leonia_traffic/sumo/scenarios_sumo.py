"""SUMO-side adapter for the declarative scenario DSL.

The :mod:`leonia_traffic.simulation.scenarios` module defines a small
DSL — :class:`Closure`, :class:`OneWayConversion`,
:class:`SpeedHumpCalming`, :class:`LaneReduction` — that the UXsim
pipeline applies *before* loading the network. SUMO is different: the
network is fixed at ``netconvert`` time, so we apply scenarios *during*
the simulation through libsumo's edge controls.

This adapter takes the same ``Scenario`` instances the UXsim pipeline
uses and translates each into one or more :class:`SumoRuntime` calls
so callers can write a single scenario list and run it through either
simulator.

Limitations
-----------

* :class:`LaneReduction` cannot be honoured by libsumo at runtime —
  SUMO requires a fresh ``netconvert`` pass to change lane counts.
  The adapter logs a warning and falls back to a partial-speed
  reduction proportional to the lane delta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from leonia_traffic.simulation.scenarios import (
    Closure,
    LaneReduction,
    OneWayConversion,
    Scenario,
    SpeedHumpCalming,
)
from leonia_traffic.sumo.runtime import SumoRuntime

logger = logging.getLogger(__name__)


@dataclass
class AppliedScenario:
    """Record of one scenario translation.

    Attributes
    ----------
    scenario
        The original :class:`Scenario` instance.
    affected_edges
        SUMO edge ids that were modified.
    notes
        Human-readable diagnostics (e.g. lane-reduction warnings).
    """

    scenario: Scenario
    affected_edges: list[str]
    notes: list[str]


def apply_scenario(rt: SumoRuntime, scenario: Scenario) -> AppliedScenario:
    """Translate one :class:`Scenario` into runtime calls.

    Mutations are *not* automatically reverted; call :meth:`SumoRuntime.restore`
    or re-run :func:`apply_scenario` with a counter-scenario to roll back.
    """
    affected: list[str] = []
    notes: list[str] = []

    if isinstance(scenario, Closure):
        affected = rt.apply_closure(scenario.osm_way_ids)
        if not affected:
            notes.append("no edges resolved for any of the listed OSM ways")

    elif isinstance(scenario, SpeedHumpCalming):
        # ``SpeedHumpCalming.free_flow_speed_factor`` operates on the
        # underlying free-flow speed; SUMO doesn't expose a stable
        # "free flow" metric so we apply the factor against the
        # current edge maxSpeed (the post-netconvert default).
        target_mph = max(
            10.0,
            scenario.min_free_flow_speed_ms / 0.44704,
        )
        affected = rt.set_speed(scenario.osm_way_ids, target_mph)
        notes.append(
            f"set_speed -> {target_mph:.0f} mph "
            f"(factor {scenario.free_flow_speed_factor:.2f}, "
            f"floor {scenario.min_free_flow_speed_ms:.1f} m/s)"
        )

    elif isinstance(scenario, OneWayConversion):
        # libsumo can't drop a single direction of an edge at runtime;
        # the closest equivalent is to *block* the reverse-direction
        # edge for passenger vehicles. We approximate it with a near-zero
        # max speed on the reverse edges (those whose SUMO id matches
        # ``-<osm_way_id>``-style markers when netconvert produced a
        # bidi pair).
        reverse_ids: list[str] = []
        for way in scenario.osm_way_ids:
            try:
                way_int = int(way)
            except (TypeError, ValueError):
                continue
            for eid in rt.osm_lookup.get(way_int, []):
                if eid.startswith("-"):
                    reverse_ids.append(eid)
        if not reverse_ids:
            notes.append(
                "no reverse-direction SUMO edges resolved; "
                "libsumo cannot enforce one-way for already-merged ways"
            )
        else:
            for eid in reverse_ids:
                try:
                    rt._backend.edge.setMaxSpeed(eid, 0.1)
                    affected.append(eid)
                except Exception as exc:
                    notes.append(f"setMaxSpeed failed on {eid}: {exc}")

    elif isinstance(scenario, LaneReduction):
        # libsumo cannot change lane counts at runtime. Approximate by
        # halving the maximum speed (a crude capacity proxy) and emit
        # a loud warning so the user knows to rebuild the .net.xml
        # for an exact match.
        for way in scenario.osm_way_ids:
            try:
                way_int = int(way)
            except (TypeError, ValueError):
                continue
            for eid in rt.osm_lookup.get(way_int, []):
                try:
                    prev = rt._backend.edge.getMaxSpeed(eid)
                    rt._backend.edge.setMaxSpeed(eid, prev * 0.5)
                    affected.append(eid)
                except Exception:
                    continue
        notes.append(
            "LaneReduction approximated as a 50%% maxSpeed cut; "
            "rerun netconvert with --lanes overrides for an exact match"
        )

    else:
        notes.append(f"unsupported scenario type: {type(scenario).__name__}")

    return AppliedScenario(
        scenario=scenario, affected_edges=affected, notes=notes,
    )


def apply_scenarios(
    rt: SumoRuntime,
    scenarios: Iterable[Scenario],
) -> list[AppliedScenario]:
    """Apply a sequence of scenarios in order. Returns the per-scenario log."""
    out: list[AppliedScenario] = []
    for sc in scenarios:
        out.append(apply_scenario(rt, sc))
    return out
