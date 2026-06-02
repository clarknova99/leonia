"""Adaptive max-pressure traffic-signal controller for the SUMO runtime.

Ported from the Chișinău project's ``adaptive_pressure.py`` but adapted
to drive :class:`leonia_traffic.sumo.runtime.SumoRuntime` directly: the
controller receives the runtime's backend module (``libsumo`` or
``traci``) at construction time rather than importing a global ``traci``.

Algorithm (per intersection, per control step)
-----------------------------------------------

1. For each phase ``P`` compute its *pressure* — the number of vehicles
   halting on the incoming lanes that phase serves.
2. If the current phase has been green for at least ``min_green`` and a
   different phase has pressure exceeding the current phase's by at
   least ``pressure_threshold``, switch to the highest-pressure phase.
3. Never hold a phase past ``max_green`` (force a round-robin advance).

Wire it into a run via the runtime's per-step hook::

    rt = SumoRuntime.start(demand=..., tripinfo_path=...)
    ctl = AdaptivePressureController(rt.backend, rt.traffic_light_ids())
    rt.run_to_end(step_callback=ctl.step)
    diagnostics = ctl.diagnostics()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSignalConfig:
    """Tunable parameters for :class:`AdaptivePressureController`."""

    min_green: int = 25
    max_green: int = 60
    pressure_threshold: int = 3
    control_interval: int = 5


class AdaptivePressureController:
    """Max-pressure adaptive signal controller driven by a SUMO backend.

    Parameters
    ----------
    backend
        The ``libsumo`` / ``traci`` module from
        :attr:`SumoRuntime.backend`. Must expose ``trafficlight`` and
        ``lane`` namespaces (both bindings do, identically).
    intersection_ids
        TLS ids to control. Pass :meth:`SumoRuntime.traffic_light_ids`
        for citywide control, or a subset for a pilot.
    config
        An :class:`AdaptiveSignalConfig` (defaults used when omitted).
    """

    def __init__(
        self,
        backend,
        intersection_ids: list[str],
        config: AdaptiveSignalConfig | None = None,
    ):
        self._backend = backend
        self.intersection_ids = list(intersection_ids)
        cfg = config or AdaptiveSignalConfig()
        self.min_green = cfg.min_green
        self.max_green = cfg.max_green
        self.pressure_threshold = cfg.pressure_threshold
        self.control_interval = max(1, cfg.control_interval)

        # Runtime state, keyed by intersection id.
        self._phase_start: dict[str, int] = {}
        self._current_phase: dict[str, int] = {}
        self._phase_lanes: dict[str, dict[int, list[str]]] = {}

        self._initialized = False
        self._n_switches = 0
        self._n_control_steps = 0

    # ------------------------------------------------------------------
    # Initialisation (runs lazily on the first control step)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        tl = self._backend.trafficlight
        for jid in self.intersection_ids:
            try:
                defs = tl.getCompleteRedYellowGreenDefinition(jid)
                n = len(defs[0].phases) if defs else 4
            except Exception:
                n = 4
            self._current_phase[jid] = 0
            self._phase_start[jid] = 0
            self._phase_lanes[jid] = self._build_phase_lane_map(jid, n)
        self._initialized = True
        logger.info(
            "[signal_control] initialised for %d intersections",
            len(self.intersection_ids),
        )

    def _build_phase_lane_map(
        self, jid: str, n_phases: int,
    ) -> dict[int, list[str]]:
        """Map each phase index to the incoming lanes it serves."""
        try:
            links = self._backend.trafficlight.getControlledLinks(jid)
        except Exception:
            return {i: [] for i in range(max(n_phases, 1))}

        n = max(n_phases, 1)
        phase_map: dict[int, list[str]] = {i: [] for i in range(n)}
        for idx, link_list in enumerate(links):
            phase_idx = idx % n
            for link in link_list:
                if not link:
                    continue
                incoming = link[0]
                if incoming and incoming not in phase_map[phase_idx]:
                    phase_map[phase_idx].append(incoming)
        return phase_map

    # ------------------------------------------------------------------
    # Per-step control
    # ------------------------------------------------------------------

    def step(self, sim_time: float) -> None:
        """Per-step hook — pass as ``step_callback`` to the runtime."""
        if not self._initialized:
            self._initialize()

        t = int(sim_time)
        if t % self.control_interval != 0:
            return
        self._n_control_steps += 1
        for jid in self.intersection_ids:
            self._control_intersection(jid, t)

    def _control_intersection(self, jid: str, sim_time: int) -> None:
        current_phase = self._current_phase.get(jid, 0)
        phase_start = self._phase_start.get(jid, 0)
        green_duration = sim_time - phase_start

        # Enforce minimum green.
        if green_duration < self.min_green:
            return

        lanes_by_phase = self._phase_lanes.get(jid, {})
        if not lanes_by_phase:
            return

        # Enforce maximum green — force a round-robin advance.
        if green_duration >= self.max_green:
            next_phase = (current_phase + 1) % len(lanes_by_phase)
            self._switch_phase(jid, next_phase, sim_time)
            return

        # Compute pressure (halting count on incoming lanes) per phase.
        pressures: dict[int, int] = {}
        for phase_idx, lanes in lanes_by_phase.items():
            q_in = 0
            for lane in lanes:
                try:
                    q_in += self._backend.lane.getLastStepHaltingNumber(lane)
                except Exception:
                    continue
            pressures[phase_idx] = q_in

        if not pressures:
            return

        best_phase = max(pressures, key=pressures.get)
        best_pressure = pressures[best_phase]
        curr_pressure = pressures.get(current_phase, 0)

        if (best_phase != current_phase
                and best_pressure - curr_pressure >= self.pressure_threshold):
            self._switch_phase(jid, best_phase, sim_time)

    def _switch_phase(self, jid: str, phase: int, sim_time: int) -> None:
        try:
            self._backend.trafficlight.setPhase(jid, phase)
        except Exception as exc:
            logger.debug("[signal_control] %s phase switch failed: %s",
                         jid, exc)
            return
        self._current_phase[jid] = phase
        self._phase_start[jid] = sim_time
        self._n_switches += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Summary stats for the manifest / logs."""
        return {
            "n_intersections": len(self.intersection_ids),
            "n_switches": self._n_switches,
            "n_control_steps": self._n_control_steps,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "pressure_threshold": self.pressure_threshold,
            "control_interval": self.control_interval,
        }
