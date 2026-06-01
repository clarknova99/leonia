"""Scenario translator for the assignment library.

Reuses the dataclasses from
:mod:`leonia_traffic.simulation.scenarios` (`Closure`, `OneWayConversion`,
`SpeedHumpCalming`, `LaneReduction`) so a single ``Scenario`` object can
drive either the UXsim mesoscopic run or the NetworkX UE assignment.

Because every existing scenario class implements ``apply(nodes, links)
-> (nodes, links)`` against the UXsim-shaped tuple, we can simply call
that method and then feed the result back through
:func:`leonia_traffic.assignment.network.build_assignment_graph`. No
new translation table is needed.
"""

from __future__ import annotations

from typing import Iterable

from leonia_traffic.simulation.scenarios import Scenario, apply_scenarios


def apply_scenarios_to_graph(
    nodes: list,
    links: list,
    scenarios: Iterable[Scenario],
) -> tuple[list, list]:
    """Apply scenarios to ``(nodes, links)`` for downstream graph build.

    Thin re-export of :func:`leonia_traffic.simulation.scenarios.apply_scenarios`
    that lives in the assignment package so notebooks only need to
    import from ``leonia_traffic.assignment``.
    """
    return apply_scenarios(nodes, links, scenarios)


__all__ = ["apply_scenarios_to_graph"]
