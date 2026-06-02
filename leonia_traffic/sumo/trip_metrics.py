"""Trip-level KPIs from SUMO's ``tripinfo`` output.

The SUMO runtime already exposes per-edge volume / speed counters; this
module adds the complementary *per-trip* lens (commute time, delay,
completion rate) by parsing the ``tripinfo.xml`` SUMO writes when the
simulation is started with ``--tripinfo-output``.

Everything here uses :mod:`xml.etree.ElementTree` and pandas only — no
``libsumo``/pyarrow read coupling — so it is safe to call in the parent
post-processing process (after the worker subprocess has closed SUMO).

The KPI keys mirror the Chișinău ``compare_runs.py`` schema so the
before/after comparison in :mod:`leonia_traffic.sumo.comparison` can
diff two runs without per-run special-casing.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_tripinfo(path: str | Path) -> pd.DataFrame:
    """Parse a SUMO ``tripinfo.xml`` into a tidy per-trip DataFrame.

    Columns
    -------
    ``id, depart, arrival, duration, route_length, waiting_time,
    time_loss, depart_hour, completed, travel_time_min, loss_min,
    waiting_min``.

    A trip is ``completed`` when SUMO recorded a non-negative
    ``arrival`` time. With ``--tripinfo-output.write-unfinished`` the
    vehicles still in the network at sim end appear with
    ``arrival == -1``; we keep them (so completion-rate is meaningful)
    but flag them as not completed.

    Returns an empty (typed) DataFrame when the file is missing or has
    no ``<tripinfo>`` elements.
    """
    cols = [
        "id", "depart", "arrival", "duration", "route_length",
        "waiting_time", "time_loss", "depart_hour", "completed",
        "travel_time_min", "loss_min", "waiting_min",
    ]
    path = Path(path)
    if not path.exists():
        logger.warning("tripinfo not found at %s", path)
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    # iterparse keeps memory flat on large (100k+ trip) files.
    for _event, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "tripinfo":
            continue
        depart = _to_float(elem.get("depart"))
        arrival = _to_float(elem.get("arrival"), default=-1.0)
        duration = _to_float(elem.get("duration"))
        rows.append({
            "id": elem.get("id"),
            "depart": depart,
            "arrival": arrival,
            "duration": duration,
            "route_length": _to_float(elem.get("routeLength")),
            "waiting_time": _to_float(elem.get("waitingTime"), 0.0),
            "time_loss": _to_float(elem.get("timeLoss"), 0.0),
        })
        elem.clear()

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["completed"] = df["arrival"] >= 0
    df["depart_hour"] = (df["depart"] // 3600).astype("Int64")
    df["travel_time_min"] = df["duration"] / 60.0
    df["loss_min"] = df["time_loss"] / 60.0
    df["waiting_min"] = df["waiting_time"] / 60.0
    return df[cols]


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


@dataclass
class TripKpis:
    """Headline per-trip KPIs for one run.

    Travel-time / delay / waiting statistics are computed over the
    *completed* trips only (an unfinished trip has no meaningful travel
    time); ``completion_rate`` captures how many trips finished.
    """

    n_trips: int
    n_completed: int
    completion_rate: float
    mean_travel_min: float
    median_travel_min: float
    p90_travel_min: float
    total_delay_h: float
    mean_waiting_min: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_trip_kpis(df: pd.DataFrame) -> TripKpis:
    """Reduce a :func:`parse_tripinfo` DataFrame to headline KPIs."""
    n_trips = int(len(df))
    if n_trips == 0:
        return TripKpis(
            n_trips=0, n_completed=0, completion_rate=0.0,
            mean_travel_min=float("nan"), median_travel_min=float("nan"),
            p90_travel_min=float("nan"), total_delay_h=0.0,
            mean_waiting_min=float("nan"),
        )

    done = df[df["completed"]]
    n_completed = int(len(done))
    travel = done["travel_time_min"]
    return TripKpis(
        n_trips=n_trips,
        n_completed=n_completed,
        completion_rate=float(n_completed / n_trips) if n_trips else 0.0,
        mean_travel_min=float(travel.mean()) if n_completed else float("nan"),
        median_travel_min=(
            float(travel.median()) if n_completed else float("nan")
        ),
        p90_travel_min=(
            float(travel.quantile(0.90)) if n_completed else float("nan")
        ),
        total_delay_h=float(done["loss_min"].sum() / 60.0),
        mean_waiting_min=(
            float(done["waiting_min"].mean()) if n_completed else float("nan")
        ),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_trip_metrics(
    run_dir: str | Path,
    df: pd.DataFrame,
    kpis: TripKpis,
) -> None:
    """Write ``tripinfo.parquet`` + ``trip_metrics.json`` under ``run_dir``.

    Safe to call from the parent process only (writes parquet). The raw
    per-trip frame is persisted so the comparison module can rebuild the
    travel-time distribution and hourly-delay curves without re-parsing
    the (large) XML.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if df is not None and not df.empty:
        df.to_parquet(run_dir / "tripinfo.parquet", index=False)
    (run_dir / "trip_metrics.json").write_text(
        json.dumps(kpis.to_dict(), indent=2, default=str)
    )
