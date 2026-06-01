"""Build the StreetLight measured-volume overlay payload for the webapp.

The dual-map iframe has historically only shown the SUMO simulation's
**modelled** traffic, which routes vehicles between a small set of OD
pairs. Most of Leonia's residential streets carry zero modelled
traffic (no OD pair routes through them), so the maps render those
streets as grey skeleton even though real-world traffic counts on
those streets are non-trivial.

This module emits a small JSON payload — one file per demand profile
— that pairs each SUMO ``edge_id`` with a **measured** hourly vph
profile derived from StreetLight data. The webapp's dual map loads
this payload at boot and draws a static dark-grey overlay underneath
the simulation frames, swapping the active hour as the slider moves.
Stakeholders see both *what real traffic looks like today* (left =
baseline) and *how the simulated change shifts that traffic* (right =
scenario).

Data sources
------------

We prefer **Network Performance** and fall back to a two-source
approximation only where it has no coverage:

* **Network Performance (2038116_leonia_network)** — *preferred.*
  Real per-segment hourly volume for every selected OSM segment
  (arterials, the GWB approach, *and* residential blocks; 815
  segments), at hourly day-parts and per-day-of-week day types. Where
  a street is covered here we use its **measured** 24-hour curve
  directly — no shape extrapolation, no borrowed level.
* **Zone Activity (2034227_leonia_streets)** — *fallback shape.*
  hourly, by day-of-week, but only ~22 Leonia streets (mostly
  tertiary+ classes). Provides the **time-of-day shape** of typical
  Leonia traffic for streets Network Performance misses.
* **StreetScanner (26600_*_streetscanner_speed_volume.csv)** —
  *fallback level.* every Leonia street including residential, but
  only "All Days / All Day" (an annualised daily total per zone).
  Provides the **absolute level** for streets in neither of the above.

Per street, the 24-hour profile is taken from the first source that
covers it, in priority order: Network Performance → ZA hourly →
StreetScanner daily total × ZA-derived shape. Every profile is keyed
to the SAME slider the simulation uses.

Output files
------------

``data/processed/sumo/runs_precache/_overlays/streetlight_weekday.json``
``data/processed/sumo/runs_precache/_overlays/streetlight_sunday.json``

Both files have the same shape::

    {
      "built_at": "2026-05-29T18:00:00Z",
      "demand": "weekday",
      "hours": [0, 1, ..., 23],
      "by_edge": {
        "<sumo_edge_id>": {
          "street": "Glenwood Avenue",
          "source": "scanner+za_shape",     // or "za" if direct
          "daily_total": 5840,
          "hourly_vph": [12.4, 8.1, ..., 306.0, ..., 18.2]
        },
        ...
      }
    }

Aggregation rules
-----------------

* **Network Performance (preferred, per edge)**: each NP segment's
  hourly vph is the mean ``avg_daily_volume`` over the requested day
  types (weekday = ``day_type_code`` 1-4; Sunday = ``7``). Each SUMO
  edge is matched to the nearest NP segment **on the same street**
  (see :func:`_match_edges_to_np_segments`) so the along-street volume
  falloff is preserved instead of being averaged away.
* **Weekday shape (fallback)**: mean of Mon/Tue/Wed/Thu
  (``day_type_code`` 1-4) in the ZA export, normalised so the 24
  hourly values sum to 1 (``shape[h] = vph[h] / sum(vph)``).
* **Sunday shape (fallback)**: ZA ``day_type_code`` 6, normalised the
  same way. (Note ZA's Sunday is 6; Network Performance's Sunday is 7.)
* **Per-street level (StreetScanner fallback)**: mean of
  ``Average Volume`` across every zone on that street with ``City``
  starting with "Leonia". Treated as vehicles/day, then redistributed
  by the ZA shape.
* **Priority per edge**: Network Performance (per segment) → ZA hourly
  (per street) → StreetScanner × shape (per street).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from leonia_traffic.config import DATA_PROCESSED_DIR, STREETLIGHT_DIR
from leonia_traffic.data.network_performance_loader import (
    load_network_performance_cached,
    load_network_performance_shapes_cached,
)
from leonia_traffic.data.za_streets_loader import (
    load_za_main_cached,
    visitors_only,
)
from leonia_traffic.sumo.net_lookup import (
    load_meta_lookup,
    load_sumo_edge_geometries,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

PRECACHE_DIR = DATA_PROCESSED_DIR / "sumo" / "runs_precache"
OVERLAY_DIR = PRECACHE_DIR / "_overlays"
META_CSV = DATA_PROCESSED_DIR / "sumo" / "leonia.edgedata.meta.csv"
NET_PATH = DATA_PROCESSED_DIR / "sumo" / "leonia.net.xml"
STREETSCANNER_CSV = (
    STREETLIGHT_DIR
    / "26600_20260514_212812_streetscanner_speed_volume.csv"
)


# ``day_type_code`` schedule for the **ZA** export. Codes line up with
# the ZA export header — see ``load_za_main`` for the parser.
WEEKDAY_DAY_CODES: tuple[int, ...] = (1, 2, 3, 4)  # Mon, Tue, Wed, Thu
SUNDAY_DAY_CODE: int = 6

# ``day_type_code`` schedule for the **Network Performance** export.
# NP uses 0=All Days, 1..7 = Mon..Sun, so Sunday is 7 here (not 6 as in
# the ZA export). Weekday is still Mon-Thu.
NP_WEEKDAY_DAY_CODES: tuple[int, ...] = (1, 2, 3, 4)
NP_SUNDAY_DAY_CODE: int = 7

# Per-street profile source labels (written into each ``by_edge`` entry).
SOURCE_NP = "network_performance"
SOURCE_ZA = "za"
SOURCE_SCANNER = "scanner+za_shape"


# ---------------------------------------------------------------------------
# Aggregation: ZA hourly  →  per-street hourly profile
# ---------------------------------------------------------------------------


def _aggregate_hourly_by_street(
    za: pd.DataFrame,
    day_codes: tuple[int, ...],
) -> pd.DataFrame:
    """Average StreetLight hourly volumes across the requested day codes.

    Returns a long-format frame with one row per ``(street_name, hour)``
    where ``hour`` is 0..23 (we shift StreetLight's 1..24 down by one
    so it matches Python ``datetime.hour`` semantics) and ``vph`` is the
    mean of the per-zone, per-day volumes for that hour. Missing
    combinations are simply absent — the webapp treats a missing edge
    as "don't draw" rather than "0 vph".
    """
    if za.empty:
        return pd.DataFrame(columns=["street_name", "hour", "vph"])
    sub = za[
        (za["day_type_code"].isin(day_codes))
        & (za["day_part_code"].between(1, 24))
        & (za["zone_volume"].notna())
        & (za["street_name"].notna())
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["street_name", "hour", "vph"])
    # day_part_code 1..24 → hour 0..23 (so JS Math.floor(seconds / 3600)
    # lines up with the array index without an off-by-one shim).
    sub["hour"] = sub["day_part_code"].astype(int) - 1
    grouped = (
        sub.groupby(["street_name", "hour"], as_index=False)["zone_volume"]
        .mean()
        .rename(columns={"zone_volume": "vph"})
    )
    return grouped


def _aggregate_np_hourly_by_zone(
    np_df: pd.DataFrame,
    day_codes: tuple[int, ...],
) -> dict[str, dict[int, float]]:
    """Per-**segment** measured hourly vph from the Network Performance export.

    Returns ``{zone_name: {hour: vph}}`` where ``zone_name`` is the full
    StreetLight segment identifier (e.g. ``"Park Avenue / 11580547 / 4"``),
    ``hour`` is 0..23 (StreetLight's day-part codes 1..24 shifted down by
    one to match ``datetime.hour``), and ``vph`` is the mean
    ``avg_daily_volume`` across the requested day codes.

    Keying by segment — rather than averaging across a street's segments
    — preserves the real along-street volume falloff (a corridor can drop
    from ~1,500 to ~120 vph between its first and last block); the
    edge→segment matcher then assigns each SUMO edge its own segment's
    profile.
    """
    if np_df is None or np_df.empty:
        return {}
    sub = np_df[
        (np_df["day_type_code"].isin(day_codes))
        & (np_df["day_part_code"].between(1, 24))
        & (np_df["avg_daily_volume"].notna())
        & (np_df["zone_name"].notna())
    ].copy()
    if sub.empty:
        return {}
    sub["hour"] = sub["day_part_code"].astype(int) - 1
    grouped = (
        sub.groupby(["zone_name", "hour"], as_index=False)["avg_daily_volume"]
        .mean()
        .rename(columns={"avg_daily_volume": "vph"})
    )
    out: dict[str, dict[int, float]] = {}
    for zone, group in grouped.groupby("zone_name"):
        out[str(zone)] = {
            int(row["hour"]): float(row["vph"]) for _, row in group.iterrows()
        }
    return out


def _match_edges_to_np_segments(
    edge_geom,
    name_of: dict[str, str],
    np_shapes,
    np_by_zone: dict[str, dict[int, float]],
) -> dict[str, tuple[str, str, list[float], float]]:
    """Map each SUMO edge to its nearest Network Performance segment.

    Both the SUMO network and Network Performance subdivide each OSM way
    into multiple sub-segments (SUMO ``<way>#0,#1,…`` edges; NP
    ``… / <way> / <split>`` zones). To preserve the per-segment volume
    falloff we assign each SUMO edge the profile of the NP segment whose
    geometry is closest to the edge's midpoint, restricting candidates to
    NP segments on the **same street name** (segments of a street are
    collinear, so nearest-by-distance is unambiguous, and the street
    constraint avoids matching a parallel road).

    Returns ``{edge_id: (zone_name, street_name, hourly_vph[24], daily)}``
    for every edge that matched. Edges whose street has no NP coverage
    are absent (the caller fills them via the ZA/StreetScanner fallback).
    """
    from collections import defaultdict

    if np_shapes is None or len(np_shapes) == 0 or edge_geom is None or edge_geom.empty:
        return {}

    segs_by_street: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for _, row in np_shapes.iterrows():
        zone = row.get("name")
        street = row.get("street_name")
        geom = row.geometry
        if (
            zone in np_by_zone
            and geom is not None
            and not geom.is_empty
            and isinstance(street, str)
        ):
            segs_by_street[street].append((str(zone), geom))
    if not segs_by_street:
        return {}

    out: dict[str, tuple[str, str, list[float], float]] = {}
    for _, erow in edge_geom.iterrows():
        eid = str(erow["edge_id"])
        geom = erow.geometry
        if geom is None or geom.is_empty:
            continue
        street = name_of.get(eid)
        cands = segs_by_street.get(street)
        if not cands:
            continue
        midpoint = geom.interpolate(0.5, normalized=True)
        best_zone = min(cands, key=lambda zs, m=midpoint: zs[1].distance(m))[0]
        hours = np_by_zone[best_zone]
        slots = [round(hours.get(h, 0.0), 1) for h in range(24)]
        out[eid] = (best_zone, street, slots, sum(slots))
    return out


def _normalised_hourly_shape(
    hourly_by_street: pd.DataFrame,
) -> list[float]:
    """Build a 24-element shape vector summing to 1.0.

    Averages the per-street, per-hour vph across every street the ZA
    export covers, then normalises so ``sum(shape) == 1``. This shape
    represents "what fraction of the daily traffic happens in each
    hour" for a typical Leonia street on the requested day type.
    Streets without their own hourly data inherit this shape.

    If the input is empty (extremely unlikely — would mean the ZA
    parquet is missing) we fall back to a flat ``1/24`` distribution.
    """
    if hourly_by_street.empty:
        logger.warning(
            "No ZA hourly data available; falling back to flat shape."
        )
        return [1.0 / 24] * 24

    by_hour = (
        hourly_by_street.groupby("hour")["vph"].mean().reindex(range(24))
    )
    # Where the export has no measurement for an hour (e.g. 3 AM on
    # tertiary streets), fill with 0 — a plausible value that won't
    # blow up the normaliser.
    by_hour = by_hour.fillna(0.0)
    total = float(by_hour.sum())
    if total <= 0:
        return [1.0 / 24] * 24
    return [float(by_hour.iloc[h]) / total for h in range(24)]


# ---------------------------------------------------------------------------
# Aggregation: StreetScanner daily totals
# ---------------------------------------------------------------------------


def _load_scanner_daily_totals() -> dict[str, float]:
    """Mean of StreetScanner ``Average Volume`` per Leonia road name.

    The CSV groups multiple zones per street (each block of Glenwood
    is its own zone). We use the **mean** so a long street with
    high-variance segments doesn't get over-counted; the result is
    "the typical block of this street carries X vehicles per day."
    """
    if not STREETSCANNER_CSV.exists():
        logger.warning(
            "StreetScanner CSV not found at %s; daily totals will only "
            "come from the ZA export (smaller coverage).",
            STREETSCANNER_CSV,
        )
        return {}
    df = pd.read_csv(STREETSCANNER_CSV)
    sub = df[
        df["City, County, State"].astype(str).str.startswith("Leonia", na=False)
        & df["Day Type"].eq("All Days")
        & df["Day Part"].eq("All Day")
        & df["Average Volume"].notna()
    ]
    if sub.empty:
        return {}
    return (
        sub.groupby("Road Name")["Average Volume"].mean().to_dict()
    )


# ---------------------------------------------------------------------------
# Edge mapping
# ---------------------------------------------------------------------------


def _street_to_edges(meta: pd.DataFrame) -> dict[str, list[str]]:
    """Map ``street_name`` to the list of SUMO ``edge_id`` strings on it.

    Mirrors the same mapping the scenario builder uses (see
    ``_load_streets`` in ``build_precache.py``) so the overlay
    references the same edges that the dual-map skeleton uses for
    highlights and frames.
    """
    if meta.empty:
        return {}
    sub = meta.dropna(subset=["street_name", "sumo_edge_id"])
    return (
        sub.groupby("street_name")["sumo_edge_id"]
        .apply(lambda s: sorted({str(x) for x in s}))
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Overlay builder
# ---------------------------------------------------------------------------


def _resolve_street_profile(
    *,
    za_hours: dict[int, float],
    scanner_total: float | None,
    shape: list[float],
) -> tuple[list[float], str, float] | None:
    """Pick the best street-level 24-hour profile for one street.

    Used as the **fallback** for streets/edges that Network Performance
    (handled per-edge upstream) does not cover. Returns
    ``(hourly_vph[24], source, daily_total)`` using the higher-priority
    source that covers the street, or ``None`` if none do. Priority:
    ZA hourly → StreetScanner × shape.
    """
    if za_hours:
        # Use measured vph where present; for the holes, spread
        # (daily_total × shape) to keep slider continuity. The daily
        # total is the measured hours plus a shape-implied extrapolation
        # for the rest.
        measured_total = sum(za_hours.values())
        measured_share = sum(shape[h] for h in za_hours)
        implied_daily = (
            measured_total / measured_share if measured_share > 0 else measured_total
        )
        # Prefer scanner_total when bigger — it captures more zones than
        # ZA's tertiary-only sample, so tends to be a fuller daily figure.
        if scanner_total is not None:
            implied_daily = max(implied_daily, scanner_total)
        slots = [
            round(za_hours.get(h, shape[h] * implied_daily), 1) for h in range(24)
        ]
        return slots, SOURCE_ZA, implied_daily

    if scanner_total is not None:
        slots = [round(shape[h] * scanner_total, 1) for h in range(24)]
        return slots, SOURCE_SCANNER, scanner_total

    return None


def _build_overlay_payload(
    za: pd.DataFrame,
    day_codes: tuple[int, ...],
    name_to_edges: dict[str, list[str]],
    scanner_daily: dict[str, float],
    demand_label: str,
    edge_np: dict[str, tuple[str, str, list[float], float]] | None = None,
) -> dict:
    """Compose the ``streetlight_<demand>.json`` payload.

    Each SUMO edge's 24-hour profile comes from one of these paths,
    in priority order:

    1. **Network Performance (per edge)** — if the edge was matched to a
       Network Performance segment (see
       :func:`_match_edges_to_np_segments`), use that segment's measured
       per-hour vph directly. This is resolved **per edge**, so the
       along-street volume falloff is preserved (e.g. Park Avenue
       dropping from ~1,500 to ~120 vph between its blocks).
    2. **ZA hourly (per street)** — for edges on a street in the ZA
       export but with no NP match, use the street's measured per-hour
       vph. Sparse hours are filled by spreading the daily total via the
       ZA shape so no edge ends up with a hole in the slider.
    3. **StreetScanner × ZA shape (per street)** — for edges on a street
       in StreetScanner but not the above, use its daily total scaled by
       the typical-street ZA shape. Best for residential streets.
    4. **Excluded** — edges on a street in no source get no overlay
       polyline; the simulation skeleton still shows through.
    """
    edge_np = edge_np or {}
    hourly = _aggregate_hourly_by_street(za, day_codes)
    shape = _normalised_hourly_shape(hourly)

    # Per-street ZA hourly (where available).
    za_by_street: dict[str, dict[int, float]] = {}
    for street, group in hourly.groupby("street_name"):
        za_by_street[street] = {
            int(row["hour"]): float(row["vph"])
            for _, row in group.iterrows()
        }

    by_edge: dict[str, dict] = {}

    # 1) Per-edge Network Performance — preserves segment-level variation.
    np_streets: set[str] = set()
    for eid, (_zone, street, slots, daily) in edge_np.items():
        by_edge[str(eid)] = {
            "street": street,
            "source": SOURCE_NP,
            "daily_total": round(daily, 0),
            "hourly_vph": slots,
        }
        np_streets.add(street)
    n_np_edges = len(by_edge)

    # 2) Per-street ZA / StreetScanner fallback for edges NP didn't cover.
    n_dropped = 0
    fallback_streets = set(za_by_street) | set(scanner_daily)
    candidates = set(name_to_edges) & fallback_streets
    candidates_outside_net = fallback_streets - set(name_to_edges)
    if candidates_outside_net:
        logger.info(
            "%d fallback streets had StreetLight data but no matching SUMO "
            "edges (likely outside the borough or renamed): %s",
            len(candidates_outside_net),
            ", ".join(sorted(candidates_outside_net)[:8])
            + ("…" if len(candidates_outside_net) > 8 else ""),
        )

    source_counts = {SOURCE_ZA: 0, SOURCE_SCANNER: 0}
    for street in sorted(candidates):
        edge_ids = [e for e in name_to_edges.get(street, []) if e not in by_edge]
        if not edge_ids:
            continue
        profile = _resolve_street_profile(
            za_hours=za_by_street.get(street, {}),
            scanner_total=scanner_daily.get(street),
            shape=shape,
        )
        if profile is None:
            n_dropped += 1
            continue
        slots, source, daily = profile
        source_counts[source] += 1
        entry = {
            "street": street,
            "source": source,
            "daily_total": round(daily, 0),
            "hourly_vph": slots,
        }
        for eid in edge_ids:
            by_edge[eid] = entry

    return {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "demand": demand_label,
        "hours": list(range(24)),
        "by_edge": by_edge,
        "summary": {
            "n_streets_via_network_performance": len(np_streets),
            "n_edges_via_network_performance": n_np_edges,
            "n_streets_via_za": source_counts[SOURCE_ZA],
            "n_streets_via_scanner": source_counts[SOURCE_SCANNER],
            "n_streets_dropped": n_dropped,
            "n_edges_with_data": len(by_edge),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_overlays(out_dir: Path = OVERLAY_DIR) -> dict[str, Path]:
    """Build both weekday and sunday overlay files. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    za = visitors_only(load_za_main_cached())
    if za.empty:
        raise RuntimeError(
            "No StreetLight ZA data available. Run "
            "scripts/00_build_datasets.py to populate the canonical lake."
        )

    meta = load_meta_lookup(META_CSV)
    if meta.empty:
        raise FileNotFoundError(
            f"SUMO meta CSV not found at {META_CSV}; run "
            "scripts/11_export_sumo.py first."
        )
    name_to_edges = _street_to_edges(meta)
    if not name_to_edges:
        raise RuntimeError(
            "Empty street→edge mapping; check the meta CSV columns."
        )

    scanner_daily = _load_scanner_daily_totals()
    if scanner_daily:
        logger.info(
            "StreetScanner: %d Leonia streets with daily totals.",
            len(scanner_daily),
        )
    else:
        logger.warning(
            "StreetScanner data unavailable — overlay will only cover "
            "streets that are in the ZA / Network Performance exports."
        )

    np_df = load_network_performance_cached()
    np_shapes = load_network_performance_shapes_cached()
    edge_geom = load_sumo_edge_geometries(NET_PATH)
    np_ready = (
        np_df is not None and not np_df.empty
        and np_shapes is not None and len(np_shapes) > 0
        and edge_geom is not None and not edge_geom.empty
    )
    if not np_ready:
        logger.warning(
            "Network Performance segments/shapes/SUMO geometry unavailable "
            "— falling back to the ZA-shape + StreetScanner overlay. Run "
            "scripts/00_build_datasets.py --only network_performance and "
            "scripts/11_export_sumo.py to populate them."
        )
        np_df = pd.DataFrame()
    else:
        logger.info(
            "Network Performance: %d rows + %d segment shapes; matching to "
            "%d SUMO edges.",
            len(np_df), len(np_shapes), len(edge_geom),
        )

    # SUMO edge → street name, for constraining the per-edge NP match.
    name_of = dict(
        zip(meta["sumo_edge_id"].astype(str), meta["street_name"])
    )

    written: dict[str, Path] = {}
    for demand_label, za_codes, np_codes in (
        ("weekday", WEEKDAY_DAY_CODES, NP_WEEKDAY_DAY_CODES),
        ("sunday", (SUNDAY_DAY_CODE,), (NP_SUNDAY_DAY_CODE,)),
    ):
        edge_np: dict[str, tuple[str, str, list[float], float]] = {}
        if np_ready:
            np_by_zone = _aggregate_np_hourly_by_zone(np_df, np_codes)
            edge_np = _match_edges_to_np_segments(
                edge_geom, name_of, np_shapes, np_by_zone,
            )
        payload = _build_overlay_payload(
            za, za_codes, name_to_edges, scanner_daily, demand_label,
            edge_np=edge_np,
        )
        path = out_dir / f"streetlight_{demand_label}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=False))
        written[demand_label] = path
        s = payload["summary"]
        logger.info(
            "Wrote %s — Network Performance %d edges across %d streets "
            "+ %d streets via ZA + %d via StreetScanner = %d edges "
            "(dropped %d fallback streets).",
            path, s["n_edges_via_network_performance"],
            s["n_streets_via_network_performance"],
            s["n_streets_via_za"], s["n_streets_via_scanner"],
            s["n_edges_with_data"], s["n_streets_dropped"],
        )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir", type=Path, default=OVERLAY_DIR,
        help=f"Output directory (default: {OVERLAY_DIR}).",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    build_overlays(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
