"""Interactive libsumo wrapper for the Leonia SUMO project.

The :class:`SumoRuntime` exposes a stepwise controller around
``libsumo`` so callers can:

* Start a simulation from a generated routes file (or a pre-existing
  ``.sumocfg``) without writing scratch shell commands.
* Drive the clock with :meth:`step`, :meth:`run_until`, or
  :meth:`run_to_end` while reading per-edge counters back as a pandas
  DataFrame at any time.
* Apply scenarios *during* the simulation:

  - :meth:`apply_closure` disables an OSM way by name (translated to
    every matching SUMO edge);
  - :meth:`restore` reverses a closure;
  - :meth:`set_speed` lowers the maximum speed on listed ways
    (speed-hump calming);
  - :meth:`set_traffic_light` swaps a fixed-timing TLS programme.

* Collect a long-format edge-history DataFrame across the run so we
  can render an animated map and compute GEH afterwards.

The wrapper prefers ``libsumo`` (in-process bindings, fastest) and
falls back to ``traci`` (separate sumo process) when ``libsumo`` is
unavailable. Both expose the same surface so downstream code only
talks to :class:`SumoRuntime`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from leonia_traffic.sumo.demand_builder import (
    DEFAULT_NET_PATH,
    SUMO_DIR,
    DemandSource,
    build_routes,
    default_routes_path,
)
from leonia_traffic.sumo.net_lookup import (
    edges_for_osm_ways,
    load_meta_lookup,
    load_osm_to_sumo_lookup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _import_backend() -> tuple[object, str]:
    """Return the ``libsumo`` or ``traci`` module and a label.

    ``libsumo`` is preferred (in-process, no socket overhead). Falls
    back to ``traci`` automatically.
    """
    try:
        import libsumo  # type: ignore

        return libsumo, "libsumo"
    except ImportError:
        import traci  # type: ignore

        return traci, "traci"


def _resolve_sumo_binary(gui: bool = False) -> str:
    """Locate the appropriate ``sumo`` / ``sumo-gui`` binary."""
    name = "sumo-gui" if gui else "sumo"
    try:
        from sumolib import checkBinary

        path = checkBinary(name)
        if path:
            return path
    except ImportError:
        pass
    # Fall back to the bare command name; the OS will resolve it via PATH.
    return name


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class _RunStats:
    n_steps: int = 0
    n_inserted: int = 0
    n_arrived: int = 0
    wallclock_start: float = field(default_factory=time.time)


class SumoRuntime:
    """Stepwise libsumo / traci wrapper.

    Use :meth:`SumoRuntime.start` (classmethod) as the primary entry
    point — it builds routes if needed, picks a backend, and returns
    a started runtime ready to step.

    Example
    -------

    >>> with SumoRuntime.start(demand=DemandSource.PEAK_AM_SLICE) as rt:
    ...     rt.run_until(7 * 3600 + 30 * 60)
    ...     before = rt.edge_counters()
    ...     rt.apply_closure(osm_way_ids=[11586338])
    ...     rt.run_to_end()
    ...     history = rt.edge_history()
    """

    def __init__(
        self,
        backend,
        backend_name: str,
        net_path: Path,
        meta_path: Path,
        sample_interval_s: int = 60,
    ):
        self._backend = backend
        self._backend_name = backend_name
        self._net_path = net_path
        self._meta_path = meta_path
        self._sample_interval_s = int(sample_interval_s)

        self._osm_lookup: dict[int, list[str]] = {}
        self._meta_df: pd.DataFrame = pd.DataFrame()
        self._edge_ids: list[str] = []
        self._stats = _RunStats()
        self._history_rows: list[dict] = []
        self._history_df_cache: pd.DataFrame | None = None
        self._tls_backups: dict[str, str] = {}
        self._closed_ways: dict[int, list[tuple[str, float]]] = {}
        self._stopped = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        sumocfg: Path | None = None,
        demand: DemandSource | None = None,
        routes_path: Path | None = None,
        net_path: Path = DEFAULT_NET_PATH,
        meta_path: Path | None = None,
        gui: bool = False,
        seed: int = 42,
        step_length: float = 1.0,
        sample_interval_s: int = 60,
        end_time_s: int | None = None,
        tripinfo_path: Path | None = None,
        extra_args: list[str] | None = None,
    ) -> "SumoRuntime":
        """Boot a SUMO simulation and return a controlling runtime.

        Provide *one* of ``sumocfg``, ``routes_path``, or ``demand``.

        * ``sumocfg`` — load an existing ``.sumocfg``. The SUMO
          project at ``data/processed/sumo/leonia.sumocfg`` is the
          default when nothing else is supplied.
        * ``routes_path`` — load a specific routes XML alongside
          the canonical net file.
        * ``demand`` — build the routes file on the fly (writes to
          ``data/processed/sumo/leonia.routes_<source>.xml``).

        ``end_time_s`` overrides the ``<end>`` value in the config.
        ``tripinfo_path`` enables SUMO's per-trip ``--tripinfo-output``
        (parsed later by :mod:`leonia_traffic.sumo.trip_metrics`).
        ``extra_args`` are appended verbatim to the SUMO command line
        for power users.

        .. note::

           ``libsumo`` re-registers pyarrow's filesystem-scheme
           factory at import time, which permanently breaks
           ``pyarrow.fs.LocalFileSystem`` in the current process. We
           therefore (1) build the routes file (reads parquets) and
           (2) load every parquet-backed lookup *before* importing
           the backend, then never touch pyarrow again. All reads
           after :meth:`start` returns must use plain CSV / XML.
        """
        if not net_path.exists():
            raise FileNotFoundError(
                f"SUMO net file not found: {net_path}. Run "
                "scripts/11_export_sumo.py first."
            )

        if meta_path is None:
            meta_path = SUMO_DIR / "leonia.edgedata.meta.csv"

        # Build the routes file *before* the backend import (see note
        # above about the libsumo↔pyarrow filesystem-scheme conflict).
        if sumocfg is None and routes_path is None and demand is None:
            sumocfg = SUMO_DIR / "leonia.sumocfg"

        if demand is not None and routes_path is None:
            routes_path = default_routes_path(demand)
            n = build_routes(demand, routes_path, net_path=net_path)
            logger.info("Built %d flows for %s -> %s", n, demand.value,
                        routes_path)

        # Now safe to pull the backend in.
        backend, backend_name = _import_backend()
        binary = _resolve_sumo_binary(gui=gui)

        cmd = [binary]
        if sumocfg is not None and routes_path is None:
            cmd += ["-c", str(sumocfg)]
        else:
            cmd += [
                "--net-file", str(net_path),
                "--route-files", str(routes_path),
            ]
            poly_path = SUMO_DIR / "leonia.poly.xml"
            if poly_path.exists():
                cmd += ["--additional-files", str(poly_path)]
        cmd += [
            "--seed", str(seed),
            "--step-length", str(step_length),
            "--ignore-route-errors", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "300",
        ]
        if end_time_s is not None:
            cmd += ["--end", str(end_time_s)]
        if tripinfo_path is not None:
            # SUMO writes the per-trip summary at simulation end; the
            # ``write-unfinished`` flag also emits a row for vehicles
            # still in the network so completion-rate stays meaningful.
            cmd += [
                "--tripinfo-output", str(tripinfo_path),
                "--tripinfo-output.write-unfinished", "true",
            ]
        if extra_args:
            cmd += list(extra_args)

        logger.info("Starting SUMO via %s: %s", backend_name, " ".join(cmd))
        backend.start(cmd)

        rt = cls(
            backend=backend,
            backend_name=backend_name,
            net_path=net_path,
            meta_path=meta_path,
            sample_interval_s=sample_interval_s,
        )
        rt._post_start()
        return rt

    def _post_start(self) -> None:
        """One-time setup after the backend has started."""
        self._osm_lookup = load_osm_to_sumo_lookup(self._net_path)
        self._meta_df = load_meta_lookup(self._meta_path)
        all_edges = self._backend.edge.getIDList()
        self._edge_ids = [e for e in all_edges if not e.startswith(":")]
        logger.info("SUMO ready: %d non-internal edges, %d OSM ways mapped",
                    len(self._edge_ids), len(self._osm_lookup))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "SumoRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Cleanly shut the simulation down."""
        if self._stopped:
            return
        try:
            self._backend.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.warning("Error during SUMO close(): %s", exc)
        self._stopped = True

    # ------------------------------------------------------------------
    # Time / stepping
    # ------------------------------------------------------------------

    def sim_time_s(self) -> float:
        return float(self._backend.simulation.getTime())

    def step(self, seconds: int | float = 1) -> None:
        """Advance the simulation by ``seconds`` seconds."""
        end_t = self.sim_time_s() + float(seconds)
        self.run_until(end_t)

    def run_until(
        self,
        target_s: float,
        *,
        step_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Step the simulation until clock ≥ ``target_s``.

        Per-edge counters are sampled every ``sample_interval_s`` and
        appended to the in-memory history. ``step_callback``, when
        supplied, is invoked with the current sim time after every
        ``simulationStep()`` — the hook a live signal controller uses
        to read queues and switch phases.
        """
        last_sample = self._next_sample_boundary(self.sim_time_s())
        while True:
            t_now = self.sim_time_s()
            if t_now >= target_s:
                break
            try:
                self._backend.simulationStep()
            except Exception as exc:
                logger.warning("simulationStep raised %s; stopping run_until",
                               exc)
                break
            self._stats.n_steps += 1
            self._stats.n_inserted += int(
                self._backend.simulation.getDepartedNumber()
            )
            self._stats.n_arrived += int(
                self._backend.simulation.getArrivedNumber()
            )
            t_after = self.sim_time_s()
            if step_callback is not None:
                step_callback(t_after)
            if t_after >= last_sample:
                self._record_sample(t_after)
                last_sample = self._next_sample_boundary(t_after)
            # Safety: if the sim refuses to advance (no vehicles loaded
            # or very long step length) we don't want an infinite loop.
            if t_after <= t_now:
                break

    def run_to_end(
        self,
        *,
        step_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Run until SUMO reports no expected vehicles remain.

        ``step_callback`` is invoked with the current sim time after
        every step (used by the adaptive signal controller).
        """
        while True:
            try:
                self._backend.simulationStep()
            except Exception:
                break
            self._stats.n_steps += 1
            t_now = self.sim_time_s()
            if step_callback is not None:
                step_callback(t_now)
            if t_now % self._sample_interval_s < 1:
                self._record_sample(t_now)
            self._stats.n_inserted += int(
                self._backend.simulation.getDepartedNumber()
            )
            self._stats.n_arrived += int(
                self._backend.simulation.getArrivedNumber()
            )
            min_expected = self._backend.simulation.getMinExpectedNumber()
            if min_expected <= 0:
                break

    def _next_sample_boundary(self, t: float) -> float:
        """Round ``t`` up to the next multiple of the sample interval."""
        ivl = self._sample_interval_s
        return (int(t // ivl) + 1) * ivl

    # ------------------------------------------------------------------
    # Live readouts
    # ------------------------------------------------------------------

    def edge_counters(self) -> pd.DataFrame:
        """Snapshot of every active edge's counters at the current sim time."""
        rows: list[dict] = []
        t_now = self.sim_time_s()
        for eid in self._edge_ids:
            try:
                rows.append({
                    "t_sim_s": t_now,
                    "sumo_edge_id": eid,
                    "vehicles": self._backend.edge.getLastStepVehicleNumber(eid),
                    "mean_speed_ms":
                        self._backend.edge.getLastStepMeanSpeed(eid),
                    "occupancy":
                        self._backend.edge.getLastStepOccupancy(eid),
                    "waiting_time_s":
                        self._backend.edge.getWaitingTime(eid),
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df["mean_speed_mph"] = df["mean_speed_ms"] / 0.44704
            if not self._meta_df.empty:
                df = df.merge(
                    self._meta_df[["sumo_edge_id", "osm_way_id",
                                   "street_name"]],
                    on="sumo_edge_id", how="left",
                )
        return df

    def edge_history(self) -> pd.DataFrame:
        """Long-format DataFrame of all sampled counters during the run."""
        if self._history_df_cache is None:
            df = pd.DataFrame(self._history_rows)
            if not df.empty and not self._meta_df.empty:
                df = df.merge(
                    self._meta_df[["sumo_edge_id", "osm_way_id",
                                   "street_name"]],
                    on="sumo_edge_id", how="left",
                )
            self._history_df_cache = df
        return self._history_df_cache.copy()

    def stats(self) -> dict:
        """Return aggregate run statistics."""
        return {
            "backend": self._backend_name,
            "sim_time_s": self.sim_time_s(),
            "n_steps": self._stats.n_steps,
            "n_inserted": self._stats.n_inserted,
            "n_arrived": self._stats.n_arrived,
            "wallclock_s": time.time() - self._stats.wallclock_start,
        }

    def _record_sample(self, t_now: float) -> None:
        """Append a per-edge snapshot row to the in-memory history."""
        self._history_df_cache = None
        # Bucket the sample to the interval boundary (so the same
        # boundary doesn't get two records if simulationStep runs in
        # half-second increments).
        ivl = self._sample_interval_s
        t_bin = int(t_now // ivl) * ivl
        for eid in self._edge_ids:
            try:
                self._history_rows.append({
                    "t_bin_s": t_bin,
                    "sumo_edge_id": eid,
                    "vehicles":
                        self._backend.edge.getLastStepVehicleNumber(eid),
                    "mean_speed_ms":
                        self._backend.edge.getLastStepMeanSpeed(eid),
                    "occupancy":
                        self._backend.edge.getLastStepOccupancy(eid),
                })
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Scenario controls
    # ------------------------------------------------------------------

    def _resolve(self, osm_way_ids: Iterable[int]) -> list[str]:
        """OSM ways → flat list of SUMO edge ids."""
        return edges_for_osm_ways(osm_way_ids, self._osm_lookup)

    def apply_closure(
        self,
        osm_way_ids: Iterable[int],
        *,
        crawl_speed_ms: float = 0.1,
    ) -> list[str]:
        """Close every SUMO edge mapped to the given OSM ways.

        Implemented as ``setMaxSpeed(crawl)`` so vehicles already on
        the link can drain — a hard ``setDisallowed`` would strand
        them. Returns the affected edge ids; pass the same list to
        :meth:`restore` to undo the closure.
        """
        affected: list[str] = []
        for way in osm_way_ids:
            try:
                way_int = int(way)
            except (TypeError, ValueError):
                continue
            edge_ids = self._osm_lookup.get(way_int, [])
            if not edge_ids:
                continue
            backups: list[tuple[str, float]] = []
            for eid in edge_ids:
                # libsumo.edge has setMaxSpeed but no getMaxSpeed; the
                # canonical pre-mutation max speed lives on the edge's
                # first lane. Fall back to the configured default if
                # the lane probe fails for any reason.
                prev: float | None = None
                try:
                    n_lanes = self._backend.edge.getLaneNumber(eid)
                except Exception:
                    n_lanes = 0
                if n_lanes > 0:
                    try:
                        prev = self._backend.lane.getMaxSpeed(f"{eid}_0")
                    except Exception:
                        prev = None
                if prev is None:
                    try:
                        prev = self._backend.edge.getLastStepMeanSpeed(eid)
                    except Exception:
                        prev = 13.4  # ≈ 30 mph fallback for restore
                try:
                    self._backend.edge.setMaxSpeed(eid, crawl_speed_ms)
                except Exception:
                    continue
                backups.append((eid, float(prev)))
                affected.append(eid)
            if backups:
                self._closed_ways[way_int] = backups
        return affected

    def restore(self, osm_way_ids: Iterable[int]) -> int:
        """Undo a previous :meth:`apply_closure`. Returns # restored edges."""
        n = 0
        for way in osm_way_ids:
            try:
                way_int = int(way)
            except (TypeError, ValueError):
                continue
            backups = self._closed_ways.pop(way_int, None)
            if not backups:
                continue
            for eid, prev in backups:
                try:
                    self._backend.edge.setMaxSpeed(eid, prev)
                    n += 1
                except Exception:
                    continue
        return n

    def set_speed(
        self,
        osm_way_ids: Iterable[int],
        mph: float,
    ) -> list[str]:
        """Set the maximum allowed speed on listed OSM ways.

        Used for speed-hump calming. Reversible with another
        :meth:`set_speed` call (or :meth:`restore` if you also called
        :meth:`apply_closure`).
        """
        target_ms = max(0.1, float(mph) * 0.44704)
        affected: list[str] = []
        for eid in self._resolve(osm_way_ids):
            try:
                self._backend.edge.setMaxSpeed(eid, target_ms)
                affected.append(eid)
            except Exception:
                continue
        return affected

    def set_traffic_light(self, tls_id: str, state: str) -> bool:
        """Override a traffic light's signal state.

        ``state`` follows SUMO's ``rRgGyY`` per-approach encoding;
        callers typically read it from ``trafficlight.getRedYellowGreenState``
        before mutating. The previous state is backed up so a follow-up
        call to :meth:`restore_traffic_light` reverts cleanly.
        """
        try:
            prev = self._backend.trafficlight.getRedYellowGreenState(tls_id)
        except Exception:
            return False
        self._tls_backups.setdefault(tls_id, prev)
        try:
            self._backend.trafficlight.setRedYellowGreenState(tls_id, state)
            return True
        except Exception:
            return False

    def restore_traffic_light(self, tls_id: str) -> bool:
        """Revert a TLS to the state it had before :meth:`set_traffic_light`."""
        prev = self._tls_backups.pop(tls_id, None)
        if prev is None:
            return False
        try:
            self._backend.trafficlight.setRedYellowGreenState(tls_id, prev)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Convenience: derive vph from the recorded history
    # ------------------------------------------------------------------

    def edge_summary(self) -> pd.DataFrame:
        """Per-edge summary of the run.

        One row per SUMO edge with:

        * ``total_vehicles`` — sum of every sample's vehicle count
          (proxy for vehicle-bin-seconds, *not* a unique vehicle count).
        * ``peak_vph`` — highest per-bin vehicle count, scaled to vph.
        * ``mean_speed_mph`` — sample-weighted mean speed (mph).
        * ``vph_observed_in_window`` — alias for callers that prefer
          the explicit name.
        """
        hist = self.edge_history()
        if hist.empty:
            return pd.DataFrame(
                columns=["sumo_edge_id", "osm_way_id", "street_name",
                         "total_vehicles", "peak_vph", "mean_speed_mph"]
            )
        ivl = self._sample_interval_s
        scale_to_hour = 3600.0 / ivl
        agg = hist.groupby("sumo_edge_id", as_index=False).agg(
            total_vehicles=("vehicles", "sum"),
            peak_vehicles=("vehicles", "max"),
            mean_speed_ms=("mean_speed_ms", "mean"),
        )
        agg["peak_vph"] = agg["peak_vehicles"] * scale_to_hour
        agg["mean_speed_mph"] = agg["mean_speed_ms"] / 0.44704
        if not self._meta_df.empty:
            agg = agg.merge(
                self._meta_df[["sumo_edge_id", "osm_way_id",
                               "street_name"]],
                on="sumo_edge_id", how="left",
            )
        return agg

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def traffic_light_ids(self) -> list[str]:
        """Every traffic-light (TLS) id netconvert produced for the net."""
        try:
            return list(self._backend.trafficlight.getIDList())
        except Exception as exc:  # pragma: no cover - backend probe
            logger.warning("trafficlight.getIDList failed: %s", exc)
            return []

    @property
    def backend(self):
        """The raw ``libsumo`` / ``traci`` module (for live controllers)."""
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def osm_lookup(self) -> dict[int, list[str]]:
        return dict(self._osm_lookup)

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta_df.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def silenced_sumo_warnings():
    """Silence SUMO's noisy stderr while a block runs (best-effort)."""
    devnull = open(os.devnull, "w")
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        devnull.close()
