"""Precompute the SUMO scenario cache the stakeholder webapp serves.

For every (street, change_type, demand) triple, run a SUMO scenario
end-to-end and save the comparison artefacts under
``data/processed/sumo/runs_precache/<key>/``. The webapp then maps
dropdown selections to cached artefact URLs without ever invoking
SUMO at request time.

Scope
-----

The default scope mirrors the plan:

* **Streets** — every distinct ``street_name`` in
  ``data/processed/leonia_streets_cutthrough_index.parquet`` (90
  ranked Leonia local streets). Streets with multiple OSM ways are
  modified as a unit.
* **Change types** — ``closure`` (hard close), ``speed_hump`` (drop
  free-flow speed to ~10 mph), ``oneway`` (convert two-way to
  one-way; needs a fresh ``netconvert`` pass per scenario).
* **Demand cohorts** — ``bridge_od_weekday_24h`` (Mon–Fri average)
  and ``bridge_od_sunday_24h`` (Sunday only).

Total: ``90 streets × 3 change_types × 2 demands = 540 runs``. With
``--parallel 8`` this is roughly 2 hours of wallclock.

Output layout
-------------

::

    data/processed/sumo/runs_precache/
        baseline_weekday/                  # one shared baseline per demand
            edge_summary.parquet
            animated.html
            manifest.json
        baseline_sunday/
            ...
        willow_tree_road__closure__weekday/
            edge_summary.parquet
            animated.html
            compare.html                   # side-by-side vs baseline_weekday
            manifest.json
        ...
        catalog.json                       # what the webapp reads at boot

The catalog enumerates every successfully-built scenario and points
the webapp at the right HTML files.

Process model
-------------

Each individual SUMO run is launched as a worker subprocess
(``--worker``). The parent process never imports ``libsumo`` so its
``pyarrow`` filesystem registration stays intact. With ``--parallel
N`` the parent uses ``concurrent.futures.ProcessPoolExecutor`` to
fan out across N worker shells. Each worker run is fully independent
and writes to its own ``runs_precache/<key>/`` directory.

Resumability
------------

A run is considered complete iff ``runs_precache/<key>/manifest.json``
exists. The script skips already-complete runs on every invocation,
so ``Ctrl-C`` and rerun is safe. The catalog is regenerated from the
filesystem at the end of every invocation.

Usage
-----

::

    # Default: 90 streets x 3 change types x 2 demands, parallel 8
    venv/bin/python webapp/scripts/build_precache.py

    # Smoke test — one street, one change type, one demand
    venv/bin/python webapp/scripts/build_precache.py \\
        --streets willow_tree_road \\
        --change-types closure \\
        --demands bridge_od_weekday_24h \\
        --parallel 1

    # Top N most-trafficked streets only (cuts precache build time)
    venv/bin/python webapp/scripts/build_precache.py --top-n 30

"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports below are all inside-leonia and may transitively load
# parquets — that's fine, this script is the *parent* and never
# imports libsumo (workers do).
from leonia_traffic.config import DATA_PROCESSED_DIR  # noqa: E402

SUMO_DIR = DATA_PROCESSED_DIR / "sumo"
PRECACHE_DIR = SUMO_DIR / "runs_precache"
DEFAULT_NET_PATH = SUMO_DIR / "leonia.net.xml"
DEFAULT_OSM_PATH = SUMO_DIR / "leonia.osm.xml"
CUTTHROUGH_INDEX_PATH = (
    DATA_PROCESSED_DIR / "leonia_streets_cutthrough_index.parquet"
)

logger = logging.getLogger("precache")

CHANGE_TYPES = ("closure", "speed_hump", "oneway")
DEMANDS = ("bridge_od_weekday_24h", "bridge_od_sunday_24h")

DEMAND_LABELS = {
    "bridge_od_weekday_24h": "weekday",
    "bridge_od_sunday_24h": "sunday",
}

CHANGE_TYPE_LABELS = {
    "closure": "Close the street",
    "speed_hump": "Add speed humps (slow to ~10 mph)",
    "oneway": "Convert to one-way",
}


# ---------------------------------------------------------------------------
# Slugs and keys
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """ASCII-lowercase-underscore slug for filesystem/URL safety."""
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unnamed"


def _scenario_key(street_slug: str, change_type: str, demand: str) -> str:
    """Stable key used in catalog + on-disk run dir name."""
    return f"{street_slug}__{change_type}__{DEMAND_LABELS[demand]}"


def _baseline_key(demand: str) -> str:
    return f"baseline__{DEMAND_LABELS[demand]}"


# ---------------------------------------------------------------------------
# Scenario enumeration
# ---------------------------------------------------------------------------


@dataclass
class StreetSpec:
    """A single street in the precache (may span multiple OSM ways)."""
    slug: str
    name: str
    osm_way_ids: list[int]
    cutthrough_rank: int
    # SUMO edge IDs that belong to this street. Surfaced in the
    # catalog so the dual-map iframe can outline the selected
    # street without having to re-derive the OSM-way mapping
    # client-side.
    sumo_edge_ids: list[str] = field(default_factory=list)


@dataclass
class RunSpec:
    """A single (street, change_type, demand) precache run."""
    key: str
    street: StreetSpec
    change_type: str
    demand: str

    @property
    def run_dir(self) -> Path:
        return PRECACHE_DIR / self.key


def _restrict_edges_to_borough(
    name_to_edges: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Drop edges whose geometry doesn't intersect the Leonia borough.

    Several Leonia streets share a name with a neighbour (Teaneck's
    Glenwood Avenue, Englewood's Grand Avenue, etc.). The SUMO net
    extends slightly past the borough to keep through-traffic
    flows realistic, so a naïve ``street_name`` lookup returns
    edges from both jurisdictions. The dropdown is selecting a
    Leonia street, so the highlight should also be Leonia-only.

    A small geometry-buffer (~10 m) on the borough polygon
    tolerates GPS/edge-shape noise at the boundary so that an
    edge whose centre vertex happens to fall a couple metres
    outside the polygon (because of edge digitisation slop) still
    counts as "in Leonia".

    If we can't load the polygon or geometries we return the
    input unchanged — degraded behaviour (highlight may extend
    past the border) is preferable to no highlight at all.
    """
    try:
        import geopandas as gpd  # local: precache build is the only caller
        from leonia_traffic.config import load_leonia_polygon
        from leonia_traffic.sumo.net_lookup import load_sumo_edge_geometries
    except Exception as exc:
        logger.warning(
            "Borough filter imports unavailable (%s); skipping.", exc,
        )
        return name_to_edges

    try:
        borough = load_leonia_polygon()
    except Exception as exc:
        logger.warning(
            "Could not load Leonia borough polygon (%s); skipping "
            "borough filter on street highlights.", exc,
        )
        return name_to_edges

    try:
        edges_geo = load_sumo_edge_geometries(DEFAULT_NET_PATH)
    except Exception as exc:
        logger.warning(
            "Could not load SUMO edge geometries (%s); skipping "
            "borough filter on street highlights.", exc,
        )
        return name_to_edges

    if edges_geo.empty:
        return name_to_edges

    # Project to a metric CRS (NJ State Plane North) to buffer in
    # metres rather than degrees, then test intersection back in
    # WGS84. The 10 m buffer absorbs OSM geometry noise without
    # accidentally pulling in Teaneck side streets.
    try:
        metric_crs = "EPSG:32118"  # NAD83 / New Jersey
        borough_metric = (
            gpd.GeoSeries([borough], crs="EPSG:4326")
            .to_crs(metric_crs)
            .iloc[0]
            .buffer(10.0)
        )
        borough_buffered = (
            gpd.GeoSeries([borough_metric], crs=metric_crs)
            .to_crs("EPSG:4326")
            .iloc[0]
        )
    except Exception as exc:
        logger.warning(
            "Could not buffer borough polygon (%s); using raw polygon.", exc,
        )
        borough_buffered = borough

    edges_geo = edges_geo.copy()
    edges_geo["edge_id"] = edges_geo["edge_id"].astype(str)
    edges_geo["_in_borough"] = edges_geo.geometry.intersects(borough_buffered)
    in_borough = set(
        edges_geo.loc[edges_geo["_in_borough"], "edge_id"].tolist()
    )

    filtered: dict[str, list[str]] = {}
    n_dropped_edges = 0
    n_streets_clipped = 0
    for name, edge_ids in name_to_edges.items():
        kept = [eid for eid in edge_ids if eid in in_borough]
        if len(kept) != len(edge_ids):
            n_dropped_edges += len(edge_ids) - len(kept)
            n_streets_clipped += 1
        # Streets with zero kept edges (entirely outside Leonia)
        # are not highlighted — but we still keep an empty list
        # so the catalog reports the street and the dropdown can
        # show it with a "no highlight available" experience.
        filtered[name] = kept
    if n_dropped_edges:
        logger.info(
            "Borough filter trimmed %d edges across %d streets "
            "(out-of-borough segments will not be highlighted).",
            n_dropped_edges, n_streets_clipped,
        )
    return filtered


def _load_streets(top_n: int | None = None) -> list[StreetSpec]:
    """Return one StreetSpec per distinct ``street_name`` in the index.

    Multi-way streets (e.g. Lakeview Avenue spans 5 OSM ways) are
    returned as a single StreetSpec carrying all way IDs, so a
    closure / speed-hump / oneway is applied to every segment of the
    street together.

    Cross-resolution against the SUMO meta CSV
    -------------------------------------------

    The cutthrough index uses OSM way IDs from a *historical* OSM
    extract (the one that was current when the StreetLight zones
    were defined). The active SUMO network was built from a
    *newer* OSM extract whose way IDs don't overlap with the older
    ones. We therefore resolve each ``street_name`` to its
    *current* SUMO-net way IDs by joining through
    ``leonia.edgedata.meta.csv`` (which carries the live
    ``street_name → osm_way_id`` mapping).

    A street that exists in the cutthrough index but not in the
    SUMO net is dropped, with a warning, because we have no edges
    to apply the scenario to.
    """
    import pandas as pd

    from leonia_traffic.sumo.net_lookup import load_meta_lookup

    if not CUTTHROUGH_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Cutthrough index not found at {CUTTHROUGH_INDEX_PATH}. "
            "Run scripts/09_leonia_streets_report.py first."
        )
    df = pd.read_parquet(CUTTHROUGH_INDEX_PATH)

    meta = load_meta_lookup(SUMO_DIR / "leonia.edgedata.meta.csv")
    if meta.empty:
        raise FileNotFoundError(
            "leonia.edgedata.meta.csv not found; run "
            "scripts/11_export_sumo.py first."
        )
    name_to_ways: dict[str, list[int]] = (
        meta.dropna(subset=["street_name", "osm_way_id"])
        .groupby("street_name")["osm_way_id"]
        .apply(lambda s: sorted({int(x) for x in s}))
        .to_dict()
    )
    # Used by the dual-map highlight overlay: from a street_name
    # we want every SUMO edge_id that belongs to it (both
    # directions of a two-way road, every segment of a multi-way
    # street). Filtered to the Leonia borough below so streets
    # that share a name across the border (e.g. "Glenwood Avenue"
    # also exists in Teaneck) only highlight Leonia segments.
    name_to_edges: dict[str, list[str]] = (
        meta.dropna(subset=["street_name", "sumo_edge_id"])
        .groupby("street_name")["sumo_edge_id"]
        .apply(lambda s: sorted({str(x) for x in s}))
        .to_dict()
    )
    name_to_edges = _restrict_edges_to_borough(name_to_edges)

    grouped = (
        df.sort_values("rank")
        .groupby("street_name", sort=False)
        .agg(cutthrough_rank=("rank", "min"))
        .reset_index()
        .sort_values("cutthrough_rank")
    )

    streets: list[StreetSpec] = []
    n_dropped = 0
    for _, row in grouped.iterrows():
        name = row["street_name"]
        net_ways = name_to_ways.get(name)
        if not net_ways:
            n_dropped += 1
            logger.warning(
                "Street %r in cutthrough index but missing from SUMO net; "
                "skipping.", name,
            )
            continue
        streets.append(StreetSpec(
            slug=_slugify(name),
            name=name,
            osm_way_ids=net_ways,
            cutthrough_rank=int(row["cutthrough_rank"]),
            sumo_edge_ids=name_to_edges.get(name, []),
        ))
    if n_dropped:
        logger.warning(
            "%d/%d streets dropped due to net mismatch.",
            n_dropped, len(grouped),
        )
    if top_n is not None:
        streets = streets[:top_n]
    return streets


def _enumerate_runs(
    streets: list[StreetSpec],
    change_types: list[str],
    demands: list[str],
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for street in streets:
        for change_type in change_types:
            for demand in demands:
                key = _scenario_key(street.slug, change_type, demand)
                runs.append(
                    RunSpec(
                        key=key,
                        street=street,
                        change_type=change_type,
                        demand=demand,
                    )
                )
    return runs


# ---------------------------------------------------------------------------
# OneWay netconvert rebuild
# ---------------------------------------------------------------------------


def _find_netconvert() -> str | None:
    """Locate the ``netconvert`` binary (mirror of scripts/11)."""
    if (p := shutil.which("netconvert")) is not None:
        return p
    if (sumo_home := os.environ.get("SUMO_HOME")):
        candidate = Path(sumo_home) / "bin" / "netconvert"
        if candidate.exists():
            return str(candidate)
    framework_root = Path(
        "/Library/Frameworks/EclipseSUMO.framework/Versions"
    )
    if framework_root.exists():
        for version_dir in sorted(framework_root.iterdir(), reverse=True):
            candidate = version_dir / "EclipseSUMO" / "bin" / "netconvert"
            if candidate.exists():
                return str(candidate)
    return None


def _build_oneway_osm(
    base_osm_path: Path,
    way_ids: list[int],
    out_osm_path: Path,
) -> None:
    """Clone the base OSM XML and force ``oneway=yes`` on listed ways.

    Removes any preexisting ``oneway`` tag on the targeted ways and
    inserts a fresh ``<tag k="oneway" v="yes"/>`` so netconvert will
    keep only the forward direction. Ways not in the list are left
    untouched.
    """
    tree = ET.parse(base_osm_path)
    root = tree.getroot()
    target_ids = {str(w) for w in way_ids}
    n_modified = 0
    for way in root.iter("way"):
        if way.get("id") not in target_ids:
            continue
        existing = [c for c in way if c.tag == "tag" and c.get("k") == "oneway"]
        for c in existing:
            way.remove(c)
        oneway_tag = ET.SubElement(way, "tag")
        oneway_tag.set("k", "oneway")
        oneway_tag.set("v", "yes")
        n_modified += 1
    if n_modified == 0:
        raise RuntimeError(
            f"None of the ways {way_ids} were found in {base_osm_path}"
        )
    out_osm_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_osm_path, encoding="utf-8", xml_declaration=True)


def _run_netconvert(
    osm_path: Path, net_path: Path,
    netconvert_binary: str,
) -> None:
    """Invoke netconvert with the same flags as the canonical export."""
    cmd = [
        netconvert_binary,
        "--osm-files", str(osm_path),
        "--output-file", str(net_path),
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--no-turnarounds",
        "--proj.utm",
        "--output.original-names",
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        last_err = (
            res.stderr.strip().splitlines()[-1]
            if res.stderr.strip() else "(no stderr)"
        )
        raise RuntimeError(f"netconvert failed: {last_err}")


# ---------------------------------------------------------------------------
# Worker: simulate one (street, change_type, demand)
# ---------------------------------------------------------------------------


def _worker_run_baseline(
    *,
    demand: str,
    out_dir: Path,
    net_path: Path,
    seed: int,
    sample_interval_s: int,
) -> dict:
    """Worker entry point: simulate the *baseline* (no scenario applied).

    Same demand sources as the scenario runs, but no scenario
    mutation. We need a baseline per demand cohort because the
    compare maps overlay scenario vs baseline.

    Output is *raw* (parquet for the dataframes, JSON for stats)
    so a downstream parent process can read them with pyarrow
    after libsumo has muddied this process's filesystem registry.
    Parquet writes happen *before* libsumo is imported, so this
    works fine — pyarrow's read path is what breaks.
    """
    from leonia_traffic.sumo import DemandSource, SumoRuntime

    out_dir.mkdir(parents=True, exist_ok=True)
    rt = SumoRuntime.start(
        demand=DemandSource(demand),
        net_path=net_path,
        seed=seed,
        sample_interval_s=sample_interval_s,
        tripinfo_path=out_dir / "tripinfo.xml",
    )
    t0 = time.time()
    try:
        rt.run_to_end()
        history = rt.edge_history()
        summary = rt.edge_summary()
        stats = rt.stats()
    finally:
        rt.close()
    stats["worker_wallclock_s"] = time.time() - t0

    # CSV (not parquet) — pyarrow's filesystem resolver is broken
    # in this process by libsumo. Parent re-reads + re-emits parquet.
    history.to_csv(out_dir / "edge_history.csv", index=False)
    summary.to_csv(out_dir / "edge_summary.csv", index=False)
    (out_dir / "worker_stats.json").write_text(
        json.dumps({
            "kind": "baseline",
            "demand": demand,
            "seed": seed,
            "sample_interval_s": sample_interval_s,
            "stats": stats,
        }, indent=2, default=str)
    )
    return stats


def _worker_run_scenario(
    *,
    spec_path: Path,
    out_dir: Path,
    net_path: Path,
    seed: int,
    sample_interval_s: int,
) -> dict:
    """Worker entry point: simulate one scenario from a JSON spec file.

    Writes raw parquet outputs only. Scoring + final manifest get
    assembled by the parent process after the worker exits, because
    libsumo permanently breaks pyarrow's read path inside its own
    process.
    """
    from leonia_traffic.simulation.scenarios import (
        Closure, SpeedHumpCalming,
    )
    from leonia_traffic.sumo import DemandSource, SumoRuntime
    from leonia_traffic.sumo.scenarios_sumo import apply_scenarios

    spec = json.loads(spec_path.read_text())
    scenario_obj = None
    if spec["change_type"] == "closure":
        scenario_obj = Closure(
            name=spec["key"], osm_way_ids=spec["osm_way_ids"],
        )
    elif spec["change_type"] == "speed_hump":
        scenario_obj = SpeedHumpCalming(
            name=spec["key"],
            osm_way_ids=spec["osm_way_ids"],
            free_flow_speed_factor=0.5,
            min_free_flow_speed_ms=4.5,
        )
    elif spec["change_type"] == "oneway":
        # OneWayConversion is *not* applied at runtime here — the
        # mutation already happened upstream by rebuilding the .net
        # via netconvert with ``oneway=yes`` injected into the OSM.
        # The runtime doesn't need to do anything; ``net_path``
        # already reflects the change.
        scenario_obj = None
    else:
        raise ValueError(f"Unknown change_type: {spec['change_type']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rt = SumoRuntime.start(
        demand=DemandSource(spec["demand"]),
        net_path=net_path,
        seed=seed,
        sample_interval_s=sample_interval_s,
        tripinfo_path=out_dir / "tripinfo.xml",
    )
    applied_log: list[dict] = []
    t0 = time.time()
    try:
        if scenario_obj is not None:
            applied = apply_scenarios(rt, [scenario_obj])
            applied_log = [
                {
                    "scenario": type(a.scenario).__name__,
                    "n_affected_edges": len(a.affected_edges),
                    "notes": a.notes,
                }
                for a in applied
            ]
        rt.run_to_end()
        history = rt.edge_history()
        summary = rt.edge_summary()
        stats = rt.stats()
    finally:
        rt.close()
    stats["worker_wallclock_s"] = time.time() - t0

    history.to_csv(out_dir / "edge_history.csv", index=False)
    summary.to_csv(out_dir / "edge_summary.csv", index=False)
    (out_dir / "worker_stats.json").write_text(
        json.dumps({
            "kind": "scenario",
            "demand": spec["demand"],
            "seed": seed,
            "sample_interval_s": sample_interval_s,
            "scenario": {
                "key": spec["key"],
                "street_name": spec["street_name"],
                "street_slug": spec["street_slug"],
                "change_type": spec["change_type"],
                "osm_way_ids": spec["osm_way_ids"],
                "applied": applied_log,
            },
            "stats": stats,
        }, indent=2, default=str)
    )
    return stats


# ---------------------------------------------------------------------------
# Worker process entry point — dispatched via ``--worker``
# ---------------------------------------------------------------------------


def _worker_main(argv: list[str]) -> int:
    """Subprocess entry: run a single simulation and exit.

    The parent invokes us as ``python build_precache.py --worker
    --kind {baseline,scenario} --out … --net …``. We never read the
    parent's process state — everything we need is in argv.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["baseline", "scenario"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-interval", type=int, default=60)
    p.add_argument("--demand", default=None)
    p.add_argument("--spec", default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    net_path = Path(args.net)

    if args.kind == "baseline":
        if not args.demand:
            print("--demand required for baseline", file=sys.stderr)
            return 2
        stats = _worker_run_baseline(
            demand=args.demand,
            out_dir=out_dir,
            net_path=net_path,
            seed=args.seed,
            sample_interval_s=args.sample_interval,
        )
    else:
        if not args.spec:
            print("--spec required for scenario", file=sys.stderr)
            return 2
        stats = _worker_run_scenario(
            spec_path=Path(args.spec),
            out_dir=out_dir,
            net_path=net_path,
            seed=args.seed,
            sample_interval_s=args.sample_interval,
        )
    print(f"[worker] done: {stats}")
    return 0


# ---------------------------------------------------------------------------
# Parent: spawn workers
# ---------------------------------------------------------------------------


def _spawn_worker(cmd: list[str]) -> tuple[int, str]:
    """Run a worker subprocess with output captured. Returns (rc, log)."""
    res = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    log = (res.stdout or "") + (res.stderr or "")
    return res.returncode, log


def _maybe_build_baseline(
    *,
    demand: str,
    seed: int,
    sample_interval_s: int,
    force: bool,
) -> Path:
    """Run the per-demand baseline once. Idempotent."""
    out_dir = PRECACHE_DIR / _baseline_key(demand)
    manifest = out_dir / "manifest.json"
    if manifest.exists() and not force:
        logger.info("baseline %s: skip (already built)", demand)
        return out_dir
    logger.info("baseline %s: building...", demand)
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--kind", "baseline",
        "--out", str(out_dir),
        "--net", str(DEFAULT_NET_PATH),
        "--seed", str(seed),
        "--sample-interval", str(sample_interval_s),
        "--demand", demand,
    ]
    rc, log = _spawn_worker(cmd)
    if rc != 0:
        logger.error("baseline %s FAILED (exit %d):\n%s", demand, rc, log)
        raise RuntimeError(f"baseline {demand} failed")
    _post_process_baseline(out_dir, demand)
    return out_dir


def _post_process_baseline(out_dir: Path, demand: str) -> None:
    """Read worker outputs, score, and write final manifest.

    Runs in the parent process so pyarrow is still functional.
    """
    import pandas as pd
    from leonia_traffic.sumo.scoring import score_sumo_run, write_run_outputs
    from leonia_traffic.sumo.trip_metrics import (
        compute_trip_kpis,
        parse_tripinfo,
        write_trip_metrics,
    )

    summary = pd.read_csv(out_dir / "edge_summary.csv")
    history = pd.read_csv(out_dir / "edge_history.csv")
    worker_stats = json.loads((out_dir / "worker_stats.json").read_text())

    sumo_score = score_sumo_run(summary, day_part="all_day")
    trip_df = parse_tripinfo(out_dir / "tripinfo.xml")
    trip_kpis = compute_trip_kpis(trip_df)
    write_trip_metrics(out_dir, trip_df, trip_kpis)
    manifest = {
        "demand": demand,
        "seed": worker_stats.get("seed"),
        "sample_interval_s": worker_stats.get("sample_interval_s"),
        "scenario": None,
        "kind": "baseline",
        "worker": worker_stats.get("stats", {}),
        "trip_kpis": trip_kpis.to_dict(),
    }
    write_run_outputs(
        out_dir,
        edge_history=history,
        edge_summary=summary,
        scoring_df=sumo_score.scoring_df,
        score=sumo_score.score,
        manifest=manifest,
    )

    # Emit the deck.gl flow dataset for the baseline too, so the
    # stakeholder page can show the unchanged ("no street selected")
    # network on load. No baseline_history / pins: this *is* the
    # baseline, so there's no delta to embed.
    try:
        from leonia_traffic.sumo.visualizations import write_flow_json

        history_pq = pd.read_parquet(out_dir / "edge_history.parquet")
        payload = write_flow_json(
            history_pq, DEFAULT_NET_PATH, out_dir / "flow.json",
            sample_interval_s=60,
            title=f"Leonia baseline · {DEMAND_LABELS[demand]} (no changes)",
        )
        logger.info(
            "baseline %s flow.json: %d active edges, %d frames",
            demand, payload["meta"]["n_active_edges"],
            payload["meta"]["n_frames"],
        )
    except Exception as exc:
        logger.warning("baseline %s flow.json failed: %s", demand, exc)


def _build_scenario_net(
    run: RunSpec,
    netconvert_binary: str | None,
    force: bool,
) -> tuple[Path, str | None]:
    """Resolve the per-scenario .net.xml path. Returns (path, warning).

    For ``closure`` and ``speed_hump`` we reuse the canonical
    ``leonia.net.xml`` — the mutation happens inside libsumo at
    runtime. For ``oneway`` we rebuild the OSM extract with
    ``oneway=yes`` on the targeted ways and re-run netconvert.

    The per-scenario net file lives next to the run dir to keep the
    precache directory self-contained.
    """
    if run.change_type != "oneway":
        return DEFAULT_NET_PATH, None

    if netconvert_binary is None:
        return DEFAULT_NET_PATH, "netconvert binary not found; oneway not enforceable"

    scenario_net_dir = PRECACHE_DIR / "_nets" / run.street.slug
    scenario_net = scenario_net_dir / f"{run.street.slug}.net.xml"
    scenario_osm = scenario_net_dir / f"{run.street.slug}.osm.xml"

    if scenario_net.exists() and not force:
        return scenario_net, None

    try:
        _build_oneway_osm(
            DEFAULT_OSM_PATH, run.street.osm_way_ids, scenario_osm,
        )
        _run_netconvert(scenario_osm, scenario_net, netconvert_binary)
    except Exception as exc:
        return DEFAULT_NET_PATH, f"oneway rebuild failed: {exc}"
    return scenario_net, None


def _run_one_scenario(
    run: RunSpec,
    *,
    seed: int,
    sample_interval_s: int,
    force: bool,
    netconvert_binary: str | None,
    legacy_maps: bool = False,
) -> dict:
    """End-to-end build for one (street, change_type, demand) triple.

    Steps:

    1. Skip if ``manifest.json`` already exists (resumability).
    2. Resolve the .net file (per-scenario rebuild for oneway).
    3. Write a JSON spec to disk for the worker.
    4. Spawn the SUMO worker subprocess.
    5. Build the compare map vs the baseline.

    Returns a dict suitable for catalog.json's ``scenarios[key]``.
    """
    run.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run.run_dir / "manifest.json"
    spec_path = run.run_dir / "spec.json"
    warnings: list[str] = []

    if manifest_path.exists() and not force:
        logger.info("[%s] skip (already built)", run.key)
        return _catalog_entry_from_disk(run, warnings)

    net_path, net_warning = _build_scenario_net(
        run, netconvert_binary, force=force,
    )
    if net_warning:
        warnings.append(net_warning)

    spec_path.write_text(json.dumps({
        "key": run.key,
        "street_name": run.street.name,
        "street_slug": run.street.slug,
        "osm_way_ids": run.street.osm_way_ids,
        "change_type": run.change_type,
        "demand": run.demand,
    }, indent=2))

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--kind", "scenario",
        "--out", str(run.run_dir),
        "--net", str(net_path),
        "--seed", str(seed),
        "--sample-interval", str(sample_interval_s),
        "--spec", str(spec_path),
    ]
    t0 = time.time()
    rc, log = _spawn_worker(cmd)
    elapsed = time.time() - t0
    if rc != 0:
        logger.error("[%s] FAILED (%.0fs): %s", run.key, elapsed, log[-500:])
        warnings.append(f"simulation failed (exit {rc})")
        return {
            "key": run.key,
            "street_name": run.street.name,
            "street_slug": run.street.slug,
            "osm_way_ids": run.street.osm_way_ids,
            "change_type": run.change_type,
            "demand": run.demand,
            "demand_label": DEMAND_LABELS[run.demand],
            "run_dir": run.key,
            "compare_html": None,
            "animated_html": None,
            "warnings": warnings,
            "ok": False,
        }
    try:
        _post_process_scenario(run.run_dir)
    except Exception as exc:
        logger.error("[%s] post-process failed: %s", run.key, exc)
        warnings.append(f"post-process failed: {exc}")
    logger.info("[%s] done in %.0fs", run.key, elapsed)
    result = _build_visuals(run, warnings, net_path, legacy_maps=legacy_maps)
    _trim_run_artefacts(run.run_dir)
    return result


# Filenames the webapp serves or the dual-animation builder needs
# downstream; everything else is intermediate and can be deleted
# after the visuals are rendered. edge_history.parquet (~30 MB) is
# kept because the synchronised dual animation (baseline + scenario)
# is rendered at *visual-build* time, but if you ever want to
# re-render with a different style you'd re-read it. edge_history.csv
# (~460 MB) is the heavy worker artefact and gets trimmed.
_PERSIST_FILES = {
    "compare.html",
    "animated.html",
    "animated_dual.html",
    "flow.json",
    "manifest.json",
    "edge_summary.parquet",
    "edge_history.parquet",
    "spec.json",
    "trip_metrics.json",
    "compare_kpis.json",
}


def _trim_run_artefacts(run_dir: Path) -> None:
    """Drop heavyweight intermediates (edge_history.*, etc.) post-render.

    The webapp only ever serves the HTML artefacts and the catalog
    refers to ``edge_summary.parquet``. ``edge_history`` is a
    multi-hundred-MB file used solely as input to the animated map
    builder; once ``animated.html`` is on disk we don't need it
    again.
    """
    if not run_dir.exists():
        return
    for child in run_dir.iterdir():
        if child.is_file() and child.name not in _PERSIST_FILES:
            try:
                child.unlink()
            except OSError as exc:
                logger.warning(
                    "could not remove %s: %s", child, exc,
                )


def _post_process_scenario(out_dir: Path) -> None:
    """Read worker scenario outputs, score, write manifest.json."""
    import pandas as pd
    from leonia_traffic.sumo.scoring import score_sumo_run, write_run_outputs
    from leonia_traffic.sumo.trip_metrics import (
        compute_trip_kpis,
        parse_tripinfo,
        write_trip_metrics,
    )

    summary = pd.read_csv(out_dir / "edge_summary.csv")
    history = pd.read_csv(out_dir / "edge_history.csv")
    worker_stats = json.loads((out_dir / "worker_stats.json").read_text())

    sumo_score = score_sumo_run(summary, day_part="all_day")
    trip_df = parse_tripinfo(out_dir / "tripinfo.xml")
    trip_kpis = compute_trip_kpis(trip_df)
    write_trip_metrics(out_dir, trip_df, trip_kpis)
    manifest = {
        "demand": worker_stats.get("demand"),
        "seed": worker_stats.get("seed"),
        "sample_interval_s": worker_stats.get("sample_interval_s"),
        "kind": "scenario",
        "scenario": worker_stats.get("scenario", {}),
        "worker": worker_stats.get("stats", {}),
        "trip_kpis": trip_kpis.to_dict(),
    }
    write_run_outputs(
        out_dir,
        edge_history=history,
        edge_summary=summary,
        scoring_df=sumo_score.scoring_df,
        score=sumo_score.score,
        manifest=manifest,
    )


# Module-level per-process cache. ProcessPoolExecutor workers are
# reused across many scenarios, so reading each demand's ~30 MB baseline
# edge_history once per worker (instead of once per scenario) is a large
# saving for the flow.json baseline delta on a multi-hundred-run build.
_BASELINE_HISTORY_CACHE: dict = {}


def _load_baseline_history_cached(path: Path):
    """Read (and memoise per process) a baseline edge_history parquet."""
    import pandas as pd
    key = str(path)
    df = _BASELINE_HISTORY_CACHE.get(key)
    if df is None:
        df = pd.read_parquet(path)
        _BASELINE_HISTORY_CACHE[key] = df
    return df


def _build_visuals(
    run: RunSpec,
    warnings: list[str],
    net_path: Path,
    *,
    legacy_maps: bool = False,
) -> dict:
    """Build the per-scenario visual artefacts.

    The stakeholder page renders ``flow.json`` exclusively (deck.gl), so
    that is always produced and is the only thing on the critical path.
    The legacy folium maps (``animated.html`` / ``animated_dual.html`` /
    ``compare.html``) are no longer used by the UI and are skipped unless
    ``legacy_maps`` is set — they roughly double the render cost (the
    dual map alone re-parses the baseline history and builds per-frame
    GeoJSON for both panes), so skipping them materially speeds up a
    full build.

    Each step is independent — failures are logged into the catalog
    ``warnings`` list rather than aborting the run.
    """
    import pandas as pd
    from leonia_traffic.sumo.visualizations import write_flow_json

    baseline_dir = PRECACHE_DIR / _baseline_key(run.demand)
    baseline_summary_path = baseline_dir / "edge_summary.parquet"
    baseline_history_path = baseline_dir / "edge_history.parquet"
    scenario_summary_path = run.run_dir / "edge_summary.parquet"
    history_path = run.run_dir / "edge_history.parquet"

    animated_html = run.run_dir / "animated.html"
    animated_dual_html = run.run_dir / "animated_dual.html"
    compare_html = run.run_dir / "compare.html"
    flow_json = run.run_dir / "flow.json"

    try:
        scenario_history = pd.read_parquet(history_path)
    except Exception as exc:
        logger.warning("[%s] could not read history: %s", run.key, exc)
        warnings.append(f"history read failed: {exc}")
        scenario_history = None

    # --- flow.json: the artefact the deck.gl page actually renders -----
    if scenario_history is not None:
        baseline_history_for_flow = None
        if baseline_history_path.exists():
            try:
                baseline_history_for_flow = _load_baseline_history_cached(
                    baseline_history_path
                )
            except Exception as exc:
                logger.warning(
                    "[%s] baseline history read for flow.json failed: %s",
                    run.key, exc,
                )
        try:
            payload = write_flow_json(
                scenario_history, net_path, flow_json,
                sample_interval_s=60,
                title=f"{run.street.name} · {run.change_type}",
                pin_edge_ids=run.street.sumo_edge_ids,
                baseline_history=baseline_history_for_flow,
            )
            logger.info(
                "[%s] flow.json: %d active edges, %d frames",
                run.key, payload["meta"]["n_active_edges"],
                payload["meta"]["n_frames"],
            )
        except Exception as exc:
            logger.warning("[%s] flow.json failed: %s", run.key, exc)
            warnings.append(f"flow.json failed: {exc}")
            flow_json = None
    else:
        flow_json = None

    # --- compare_kpis.json: before/after trip KPIs vs the baseline -----
    compare_kpis_json = run.run_dir / "compare_kpis.json"
    if (run.run_dir / "trip_metrics.json").exists() and (
        baseline_dir / "trip_metrics.json"
    ).exists():
        try:
            from leonia_traffic.sumo.comparison import compare_runs

            result = compare_runs(
                baseline_dir, run.run_dir,
                label_baseline=f"Baseline ({DEMAND_LABELS[run.demand]})",
                label_scenario=f"{run.street.name} · {run.change_type}",
            )
            compare_kpis_json.write_text(
                json.dumps(result.kpi_delta_payload(), indent=2, default=str)
            )
        except Exception as exc:
            logger.warning("[%s] compare_kpis.json failed: %s", run.key, exc)
            compare_kpis_json = None
    else:
        compare_kpis_json = None

    # --- legacy folium maps (opt-in; not used by the current UI) -------
    if not legacy_maps:
        animated_html = None
        animated_dual_html = None
        compare_html = None
    elif scenario_history is not None:
        from leonia_traffic.sumo.visualizations import (
            build_animated_dual_map,
            build_animated_map,
            build_dual_compare_map,
        )
        try:
            build_animated_map(
                scenario_history, animated_html,
                sample_interval_s=60,
                title=f"{run.street.name} — {run.change_type}",
            )
        except Exception as exc:
            logger.warning("[%s] animated map failed: %s", run.key, exc)
            warnings.append(f"animated map failed: {exc}")
            animated_html = None

        if baseline_history_path.exists():
            try:
                baseline_history = _load_baseline_history_cached(
                    baseline_history_path
                )
                build_animated_dual_map(
                    baseline_history, scenario_history,
                    animated_dual_html,
                    net_path=net_path,
                    sample_interval_s=60,
                    title_left=f"Baseline · {DEMAND_LABELS[run.demand]}",
                    title_right=(
                        f"{run.street.name} · {run.change_type} "
                        f"({DEMAND_LABELS[run.demand]})"
                    ),
                )
            except Exception as exc:
                logger.warning("[%s] dual animation failed: %s", run.key, exc)
                warnings.append(f"dual animation failed: {exc}")
                animated_dual_html = None
        else:
            warnings.append(
                "baseline edge_history.parquet missing; skipped dual animation"
            )
            animated_dual_html = None

        try:
            if not baseline_summary_path.exists():
                raise FileNotFoundError(
                    f"baseline summary missing: {baseline_summary_path}"
                )
            scenario_summary = pd.read_parquet(scenario_summary_path)
            baseline_summary = pd.read_parquet(baseline_summary_path)
            build_dual_compare_map(
                baseline_summary, scenario_summary,
                compare_html,
                net_path=net_path,
                title_left=f"Baseline ({DEMAND_LABELS[run.demand]})",
                title_right=f"{run.street.name} — {run.change_type}",
            )
        except Exception as exc:
            logger.warning("[%s] compare map failed: %s", run.key, exc)
            warnings.append(f"compare map failed: {exc}")
            compare_html = None
    else:
        animated_html = None
        animated_dual_html = None
        compare_html = None

    return {
        "key": run.key,
        "street_name": run.street.name,
        "street_slug": run.street.slug,
        "osm_way_ids": run.street.osm_way_ids,
        "change_type": run.change_type,
        "demand": run.demand,
        "demand_label": DEMAND_LABELS[run.demand],
        "run_dir": run.key,
        "compare_html": (
            f"{run.key}/compare.html" if compare_html else None
        ),
        "animated_html": (
            f"{run.key}/animated.html" if animated_html else None
        ),
        "animated_dual_html": (
            f"{run.key}/animated_dual.html" if animated_dual_html else None
        ),
        "flow_json": (
            f"{run.key}/flow.json" if flow_json else None
        ),
        "compare_kpis": (
            f"{run.key}/compare_kpis.json" if compare_kpis_json else None
        ),
        "warnings": warnings,
        "ok": True,
    }


def _catalog_entry_from_disk(run: RunSpec, warnings: list[str]) -> dict:
    """Reconstitute a catalog entry from an already-completed run dir."""
    return {
        "key": run.key,
        "street_name": run.street.name,
        "street_slug": run.street.slug,
        "osm_way_ids": run.street.osm_way_ids,
        "change_type": run.change_type,
        "demand": run.demand,
        "demand_label": DEMAND_LABELS[run.demand],
        "run_dir": run.key,
        "compare_html": (
            f"{run.key}/compare.html"
            if (run.run_dir / "compare.html").exists() else None
        ),
        "animated_html": (
            f"{run.key}/animated.html"
            if (run.run_dir / "animated.html").exists() else None
        ),
        "animated_dual_html": (
            f"{run.key}/animated_dual.html"
            if (run.run_dir / "animated_dual.html").exists() else None
        ),
        "flow_json": (
            f"{run.key}/flow.json"
            if (run.run_dir / "flow.json").exists() else None
        ),
        "compare_kpis": (
            f"{run.key}/compare_kpis.json"
            if (run.run_dir / "compare_kpis.json").exists() else None
        ),
        "warnings": warnings,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------


def _write_catalog(
    streets: list[StreetSpec],
    change_types: list[str],
    demands: list[str],
    entries: list[dict],
) -> Path:
    """Serialise the catalog the webapp reads at boot."""
    PRECACHE_DIR.mkdir(parents=True, exist_ok=True)
    catalog_path = PRECACHE_DIR / "catalog.json"
    scenarios = {e["key"]: e for e in entries}
    catalog = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenarios": scenarios,
        "streets": [
            {
                "slug": s.slug,
                "name": s.name,
                "cutthrough_rank": s.cutthrough_rank,
                "osm_way_ids": s.osm_way_ids,
                "sumo_edge_ids": s.sumo_edge_ids,
            }
            for s in streets
        ],
        "change_types": [
            {"value": ct, "label": CHANGE_TYPE_LABELS[ct]}
            for ct in change_types
        ],
        "demands": [
            {
                "value": d,
                "label": "Average weekday" if "weekday" in d else "Average Sunday",
            }
            for d in demands
        ],
        "baselines": {
            d: {
                "demand": d,
                "demand_label": DEMAND_LABELS[d],
                "flow_json": (
                    f"{_baseline_key(d)}/flow.json"
                    if (PRECACHE_DIR / _baseline_key(d) / "flow.json").exists()
                    else None
                ),
                "animated_html": (
                    f"{_baseline_key(d)}/animated.html"
                    if (PRECACHE_DIR / _baseline_key(d) / "animated.html").exists()
                    else None
                ),
            }
            for d in demands
        },
    }
    # Static-map artefacts (Static Maps tab). Built separately by
    # webapp/scripts/build_static_maps.py; reflect whatever is on disk.
    try:
        from build_static_maps import static_catalog_block
        catalog["static"] = static_catalog_block(PRECACHE_DIR)
    except Exception as exc:  # pragma: no cover - non-fatal
        logger.warning("could not attach static-map catalog block: %s", exc)
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=False))
    return catalog_path


def _collect_all_runs_on_disk(
    streets_in_scope: list[StreetSpec],
    change_types_in_scope: list[str],
    demands_in_scope: list[str],
) -> tuple[list[StreetSpec], list[str], list[str], list[dict]]:
    """Scan the precache dir and return every successful run as a
    catalog entry, not just the ones from this invocation.

    Resumability requires partial invocations (``--streets foo``) to
    *extend* the catalog rather than replace it. Each on-disk
    ``<key>/manifest.json`` becomes a catalog entry; we infer
    street/change/demand from the directory name.
    """
    full_streets: dict[str, StreetSpec] = {s.slug: s for s in streets_in_scope}
    full_change_types: list[str] = list(change_types_in_scope)

    # Pre-load the meta lookup once so we can backfill
    # ``sumo_edge_ids`` for streets discovered from manifests only
    # (i.e. not present in ``streets_in_scope``). If the file isn't
    # available we just leave the field empty — the highlight
    # overlay will degrade gracefully (no outline) rather than
    # breaking the page.
    _name_to_edges: dict[str, list[str]] = {}
    try:
        from leonia_traffic.sumo.net_lookup import load_meta_lookup
        _meta = load_meta_lookup(SUMO_DIR / "leonia.edgedata.meta.csv")
        if not _meta.empty:
            _name_to_edges = (
                _meta.dropna(subset=["street_name", "sumo_edge_id"])
                .groupby("street_name")["sumo_edge_id"]
                .apply(lambda s: sorted({str(x) for x in s}))
                .to_dict()
            )
    except Exception as exc:
        logger.warning(
            "Could not load meta lookup for catalog backfill: %s", exc,
        )
    full_demands: list[str] = list(demands_in_scope)
    full_entries: list[dict] = []

    if not PRECACHE_DIR.exists():
        return (
            list(full_streets.values()),
            full_change_types, full_demands, full_entries,
        )

    demand_label_to_value = {v: k for k, v in DEMAND_LABELS.items()}

    for child in sorted(PRECACHE_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if child.name.startswith("baseline__"):
            continue
        manifest = child / "manifest.json"
        if not manifest.exists():
            continue
        try:
            parts = child.name.split("__")
            if len(parts) != 3:
                continue
            street_slug, change_type, demand_label = parts
            demand = demand_label_to_value.get(demand_label)
            if demand is None:
                continue
            data = json.loads(manifest.read_text())
            scen_meta = data.get("scenario") or {}
            street_name = scen_meta.get("street_name", street_slug)
            osm_way_ids = scen_meta.get("osm_way_ids", [])
        except Exception as exc:
            logger.warning("Could not read %s: %s", manifest, exc)
            continue

        if street_slug not in full_streets:
            full_streets[street_slug] = StreetSpec(
                slug=street_slug,
                name=street_name,
                osm_way_ids=list(osm_way_ids),
                cutthrough_rank=10**6,  # sort discovered-only streets last
                sumo_edge_ids=_name_to_edges.get(street_name, []),
            )
        else:
            existing = full_streets[street_slug]
            if not existing.sumo_edge_ids:
                # Backfill from meta in case the in-scope spec was
                # built before sumo_edge_ids was a populated field.
                existing.sumo_edge_ids = _name_to_edges.get(street_name, [])
        if change_type not in full_change_types:
            full_change_types.append(change_type)
        if demand not in full_demands:
            full_demands.append(demand)

        full_entries.append({
            "key": child.name,
            "street_name": street_name,
            "street_slug": street_slug,
            "osm_way_ids": list(osm_way_ids),
            "change_type": change_type,
            "demand": demand,
            "demand_label": demand_label,
            "run_dir": child.name,
            "compare_html": (
                f"{child.name}/compare.html"
                if (child / "compare.html").exists() else None
            ),
            "animated_html": (
                f"{child.name}/animated.html"
                if (child / "animated.html").exists() else None
            ),
            "animated_dual_html": (
                f"{child.name}/animated_dual.html"
                if (child / "animated_dual.html").exists() else None
            ),
            "flow_json": (
                f"{child.name}/flow.json"
                if (child / "flow.json").exists() else None
            ),
            "compare_kpis": (
                f"{child.name}/compare_kpis.json"
                if (child / "compare_kpis.json").exists() else None
            ),
            "warnings": [],
            "ok": True,
        })

    streets_sorted = sorted(
        full_streets.values(), key=lambda s: s.cutthrough_rank,
    )
    return streets_sorted, full_change_types, full_demands, full_entries


def _build_baseline_visuals(demand: str) -> None:
    """Render the baseline animated map (no compare needed)."""
    import pandas as pd
    from leonia_traffic.sumo.visualizations import build_animated_map

    baseline_dir = PRECACHE_DIR / _baseline_key(demand)
    history_path = baseline_dir / "edge_history.parquet"
    animated = baseline_dir / "animated.html"
    if not animated.exists() and history_path.exists():
        try:
            history = pd.read_parquet(history_path)
            build_animated_map(history, animated, sample_interval_s=60)
        except Exception as exc:
            logger.warning("baseline %s animated failed: %s", demand, exc)
    # Trim heavy intermediates the same way scenario dirs are trimmed.
    _trim_run_artefacts(baseline_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--streets", nargs="*", default=None,
        help="Subset of street slugs to build (default: all from "
             "the cutthrough index).",
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help="Limit to the top N streets by cutthrough rank.",
    )
    p.add_argument(
        "--change-types", nargs="*", default=list(CHANGE_TYPES),
        choices=CHANGE_TYPES,
    )
    p.add_argument(
        "--demands", nargs="*", default=list(DEMANDS),
        choices=DEMANDS,
    )
    p.add_argument(
        "--parallel", type=int, default=8,
        help="Number of worker subprocesses to run concurrently. "
             "Each peaks around 1 GB RAM; tune down on smaller hosts.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-interval", type=int, default=60)
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild every scenario even if already cached. "
             "Default skips runs whose manifest.json exists.",
    )
    p.add_argument(
        "--legacy-maps", action="store_true",
        help="Also render the old folium maps (animated.html, "
             "animated_dual.html, compare.html). The deck.gl UI only "
             "uses flow.json, so these are skipped by default for a "
             "faster build.",
    )
    p.add_argument("--worker", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    # Worker dispatch happens *before* logging is configured because
    # the worker subprocess is short-lived and inherits stdout.
    if "--worker" in argv:
        argv.remove("--worker")
        return _worker_main(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse(argv)

    PRECACHE_DIR.mkdir(parents=True, exist_ok=True)

    netconvert_binary = _find_netconvert()
    if "oneway" in args.change_types and netconvert_binary is None:
        logger.warning(
            "netconvert not found; oneway scenarios will be flagged with a "
            "warning and fall back to the unmodified network."
        )

    # 1. Resolve streets
    all_streets = _load_streets(top_n=args.top_n)
    if args.streets:
        wanted = set(args.streets)
        streets = [s for s in all_streets if s.slug in wanted]
        missing = wanted - {s.slug for s in streets}
        if missing:
            logger.warning("unknown street slugs: %s", sorted(missing))
    else:
        streets = all_streets
    if not streets:
        logger.error("No streets selected — refusing to continue.")
        return 2
    logger.info(
        "Streets: %d  change_types: %s  demands: %s  total runs: %d",
        len(streets), args.change_types, args.demands,
        len(streets) * len(args.change_types) * len(args.demands),
    )

    # 2. Build per-demand baselines (sequentially — they're prerequisites
    #    for the dual-compare maps).
    for demand in args.demands:
        _maybe_build_baseline(
            demand=demand, seed=args.seed,
            sample_interval_s=args.sample_interval,
            force=args.force,
        )
        _build_baseline_visuals(demand)

    # 3. Enumerate scenario runs and execute via process pool.
    runs = _enumerate_runs(streets, args.change_types, args.demands)
    entries: list[dict] = []
    t_start = time.time()

    if args.parallel <= 1:
        for run in runs:
            entries.append(_run_one_scenario(
                run,
                seed=args.seed,
                sample_interval_s=args.sample_interval,
                force=args.force,
                netconvert_binary=netconvert_binary,
                legacy_maps=args.legacy_maps,
            ))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.parallel,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(
                    _run_one_scenario,
                    run,
                    seed=args.seed,
                    sample_interval_s=args.sample_interval,
                    force=args.force,
                    netconvert_binary=netconvert_binary,
                    legacy_maps=args.legacy_maps,
                ): run
                for run in runs
            }
            for fut in concurrent.futures.as_completed(futures):
                run = futures[fut]
                try:
                    entries.append(fut.result())
                except Exception as exc:
                    logger.error("[%s] uncaught: %s", run.key, exc)
                    entries.append({
                        "key": run.key,
                        "street_name": run.street.name,
                        "street_slug": run.street.slug,
                        "osm_way_ids": run.street.osm_way_ids,
                        "change_type": run.change_type,
                        "demand": run.demand,
                        "demand_label": DEMAND_LABELS[run.demand],
                        "run_dir": run.key,
                        "compare_html": None,
                        "animated_html": None,
                        "warnings": [f"uncaught: {exc}"],
                        "ok": False,
                    })

    elapsed = time.time() - t_start

    # 4. Build a *complete* catalog by scanning the precache directory
    #    for every previously-built run, not just this invocation.
    #    Without this, partial runs (--streets foo) would clobber the
    #    catalog with only their own subset.
    full_streets, full_change_types, full_demands, full_entries = (
        _collect_all_runs_on_disk(streets, args.change_types, args.demands)
    )
    catalog_path = _write_catalog(
        full_streets, full_change_types, full_demands, full_entries,
    )

    n_ok_now = sum(1 for e in entries if e.get("ok"))
    n_fail_now = len(entries) - n_ok_now
    n_total_catalog = len(full_entries)
    logger.info(
        "This invocation: %d scenarios (%d ok, %d failed) in %.0fs. "
        "Catalog now contains %d total scenarios at %s",
        len(entries), n_ok_now, n_fail_now, elapsed,
        n_total_catalog, catalog_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
