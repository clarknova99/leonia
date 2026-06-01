"""Export the Leonia data lake to a SUMO-ready project.

Produces a self-contained directory under ``data/processed/sumo/``
that can be opened with ``sumo-gui`` (Eclipse SUMO) without any
further conversion.

Usage
-----

::

    venv/bin/python scripts/11_export_sumo.py            # full export
    venv/bin/python scripts/11_export_sumo.py --no-net   # skip OSM fetch + netconvert
    venv/bin/python scripts/11_export_sumo.py --no-demand
    venv/bin/python scripts/11_export_sumo.py --bbox <minx miny maxx maxy>

Output layout::

    data/processed/sumo/
        leonia.osm.xml          # raw OSM extract (input to netconvert)
        leonia.net.xml          # SUMO road network (created by netconvert)
        leonia.poly.xml         # borough boundary + ZA zone polygons
        leonia.edgedata.xml     # per-edge observed volumes / speeds
        leonia.flows.xml        # hourly OD demand from Bridge OD
        leonia.sumocfg          # one-click load file for sumo-gui
        README_SUMO.md          # how to open / extend the project
        _manifest.json          # build provenance

What the script does
--------------------

1. Fetch a fresh OSM extract for the configured Leonia bounding box
   (same bbox the UXsim builder uses). The extract is written as
   ``leonia.osm.xml`` so it can be consumed by external tools too.
2. If ``netconvert`` is on PATH, run it to produce ``leonia.net.xml``.
   Otherwise, the script prints the exact command to run after the
   user installs SUMO (``brew install sumo`` on macOS).
3. Build ``leonia.poly.xml`` from the Leonia borough polygon plus the
   ZA zone polygons (color-coded by composite cut-through index when
   the derived table is present).
4. Build ``leonia.edgedata.xml`` keyed by OSM way id, carrying the
   observed Street Scanner ``avg_volume`` + ``avg_speed_mph`` for
   every link we've measured. ``sumo-gui`` can colour the network by
   either attribute via Edit > Edge Visualisation.
5. Build ``leonia.flows.xml`` from the Bridge OD canonical parquet —
   one ``<flow>`` per (origin, destination, day-part hour), spread
   over a 24-hour simulation horizon.
6. Write a ``.sumocfg`` so ``sumo-gui -c leonia.sumocfg`` opens
   everything at once.

The script is **read-only** with respect to ``data/processed/`` — it
only reads canonical parquets produced by
``scripts/00_build_datasets.py``. The SUMO files all live in
``data/processed/sumo/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from leonia_traffic.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    LEONIA_BBOX_WGS84,
    load_leonia_polygon,
)
from leonia_traffic.data.dataset_io import (  # noqa: E402
    CANONICAL_DIR,
    DERIVED_DIR,
    CanonicalFiles,
    DerivedFiles,
)
from leonia_traffic.sumo.net_lookup import (  # noqa: E402
    load_osm_to_sumo_lookup as _load_osm_to_sumo_lookup,
    load_sumo_edge_geometries as _load_sumo_edge_geometries,
    spatial_resolve_zones as _spatial_resolve_zones,
)

SUMO_DIR = DATA_PROCESSED_DIR / "sumo"

# Output filenames
OSM_NAME = "leonia.osm.xml"
NET_NAME = "leonia.net.xml"
POLY_NAME = "leonia.poly.xml"
EDGEDATA_NAME = "leonia.edgedata.xml"
FLOWS_NAME = "leonia.flows.xml"
CFG_NAME = "leonia.sumocfg"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _say(msg: str) -> None:
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# 1. OSM extract
# ---------------------------------------------------------------------------


def fetch_osm_xml(out_path: Path,
                  bbox: tuple[float, float, float, float]) -> bool:
    """Download a fresh OSM extract for the Leonia bbox.

    Uses osmnx (which wraps the Overpass API) to grab a drivable
    network and writes it as OSM XML — the format netconvert
    consumes.

    Returns True if the file was written, False on any error.
    """
    _section("OSM extract")
    try:
        import osmnx as ox
    except ImportError:
        _say("osmnx not installed; skipping (the file is also produced by "
             "scripts/02_build_network.py)")
        return False
    try:
        # netconvert reads OSM XML best when every way is emitted in
        # its native orientation rather than reversed for bidirectional
        # edges. osmnx warns about this if it isn't set.
        ox.settings.all_oneway = True

        minx, miny, maxx, maxy = bbox
        _say(f"querying Overpass for bbox=({minx:.4f}, {miny:.4f}, "
             f"{maxx:.4f}, {maxy:.4f})")
        # Newer osmnx (>=2.0) accepts (west, south, east, north).
        G = ox.graph.graph_from_bbox(
            bbox=(minx, miny, maxx, maxy),
            network_type="drive",
            simplify=False,
            retain_all=True,
        )
        _say(f"got graph: {len(G.nodes):,} nodes / {len(G.edges):,} edges")
        ox.save_graph_xml(G, filepath=str(out_path))
        _say(f"wrote {out_path.name} ({out_path.stat().st_size // 1024} KB)")
        return True
    except Exception as exc:
        _say(f"OSM fetch failed: {exc}")
        _say("Tip: rerun with --no-net if you only need polygons / demand.")
        return False


def _find_netconvert() -> str | None:
    """Locate the ``netconvert`` binary.

    SUMO on macOS installs to ``/Library/Frameworks/EclipseSUMO.framework``
    but doesn't add itself to PATH by default. Check that location as
    well as the standard PATH lookup. Honour ``SUMO_HOME`` if set.
    """
    import os

    if (p := shutil.which("netconvert")) is not None:
        return p
    if (sumo_home := os.environ.get("SUMO_HOME")):
        candidate = Path(sumo_home) / "bin" / "netconvert"
        if candidate.exists():
            return str(candidate)
    # macOS framework install (DLR-distributed `.pkg` or `.dmg`).
    framework_root = Path("/Library/Frameworks/EclipseSUMO.framework/Versions")
    if framework_root.exists():
        for version_dir in sorted(framework_root.iterdir(), reverse=True):
            candidate = version_dir / "EclipseSUMO" / "bin" / "netconvert"
            if candidate.exists():
                return str(candidate)
    return None


def run_netconvert(osm_path: Path, net_path: Path) -> bool:
    """Convert OSM XML to SUMO .net.xml using netconvert, if installed.

    Returns True on success, False if netconvert is missing or fails.
    """
    netconvert = _find_netconvert()
    if netconvert is None:
        _say("netconvert not found on PATH or in standard install paths.")
        _say("Install SUMO from https://eclipse.dev/sumo/ (or "
             "`brew install --cask sumo` on macOS), then run:")
        _say(f"  netconvert --osm-files {osm_path.name} "
             f"--output-file {net_path.name} "
             "--geometry.remove --ramps.guess --junctions.join "
             "--tls.guess-signals --tls.discard-simple --tls.join "
             "--no-turnarounds --proj.utm")
        return False
    _say(f"using netconvert at {netconvert}")
    cmd = [
        netconvert,
        "--osm-files", str(osm_path),
        "--output-file", str(net_path),
        # Reasonable defaults for an urban residential study area.
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--no-turnarounds",
        "--proj.utm",   # project to UTM so units are metres
        # Preserve OSM way IDs on the resulting edges so we can join
        # back to the StreetLight measurements. The OSM way id ends
        # up as a <param key="origId" value="..."/> on each edge.
        "--output.original-names",
    ]
    _say(f"running: {' '.join(cmd[:3])} ... (this takes a few seconds)")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        _say(f"netconvert failed (exit {res.returncode}):")
        _say(res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "")
        return False
    _say(f"wrote {net_path.name} ({net_path.stat().st_size // 1024} KB)")
    return True


# ---------------------------------------------------------------------------
# 2. Polygons
# ---------------------------------------------------------------------------


def _polygon_to_sumo_shape(geom) -> str:
    """Turn a shapely (Multi)Polygon into a SUMO ``shape`` attribute.

    SUMO shape syntax is ``x1,y1 x2,y2 ...`` (space-separated pairs).
    We feed it lon,lat — SUMO will use the network's projection to
    translate on the fly if the network was built with --proj.utm
    (which we do above) and the polygon file is loaded with
    ``--geo`` semantics via polyconvert. Here we just write the
    coordinate pairs and let the user run polyconvert or rely on
    the in-process loader.
    """
    if geom.geom_type == "MultiPolygon":
        # Use the largest sub-polygon for visualization.
        geom = max(geom.geoms, key=lambda p: p.area)
    coords = list(geom.exterior.coords)
    return " ".join(f"{x:.6f},{y:.6f}" for x, y in coords)


def build_polygons(out_path: Path) -> int:
    """Write the borough boundary + ZA zone polygons as polyconvert XML.

    Output is a SUMO ``<additional>`` XML that polyconvert can
    consume directly. If the user has SUMO installed, the
    ``.sumocfg`` will reference this file as a ``--poly-files``
    additional and sumo-gui will draw the polygons as a coloured
    background layer.

    Returns the number of polygons written.
    """
    _section("Polygons")

    polygons: list[tuple[str, str, str, str]] = []
    # (id, type, color, shape)

    # 1. Borough boundary (always present in this repo).
    try:
        leonia = load_leonia_polygon()
        polygons.append((
            "leonia_borough", "boundary", "0,80,200,80",
            _polygon_to_sumo_shape(leonia),
        ))
        _say("added Leonia borough boundary")
    except Exception as exc:
        _say(f"could not load borough polygon: {exc}")

    # 2. ZA zone polygons coloured by composite cut-through index
    # (if the derived table is present). Otherwise plain.
    za_poly_path = CANONICAL_DIR / CanonicalFiles.za_polygon_shapes
    if za_poly_path.exists():
        import geopandas as gpd
        import pandas as pd

        gdf = gpd.read_parquet(za_poly_path)
        gdf = gdf[gdf.geometry.notna()].copy()
        # Optional: join the cut-through index for colour-coding.
        idx_path = DERIVED_DIR / DerivedFiles.cutthrough_index
        if idx_path.exists():
            idx = pd.read_parquet(idx_path)[
                ["osm_way_id", "cutthrough_index", "rank"]
            ]
            gdf = gdf.merge(idx, on="osm_way_id", how="left")
        else:
            gdf["cutthrough_index"] = float("nan")
            gdf["rank"] = float("nan")

        for _, row in gdf.iterrows():
            score = row.get("cutthrough_index")
            color = _index_to_rgba(score) if score is not None else "120,120,120,90"
            way = row.get("osm_way_id")
            pid = f"za_{int(way)}" if way is not None and not pd.isna(way) else f"za_{row.name}"
            polygons.append((
                pid, "za_zone", color,
                _polygon_to_sumo_shape(row.geometry),
            ))
        _say(f"added {len(gdf)} ZA zone polygons")
    else:
        _say("ZA polygons not built; run scripts/00_build_datasets.py")

    # 3. Emit XML
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        fh.write('            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')
        for pid, ptype, color, shape in polygons:
            fh.write(
                f'  <poly id="{escape(pid)}" type="{escape(ptype)}" '
                f'color="{color}" fill="1" layer="-1.00" '
                f'shape="{shape}"/>\n'
            )
        fh.write('</additional>\n')
    _say(f"wrote {out_path.name} ({len(polygons)} polygons)")
    return len(polygons)


def _index_to_rgba(score) -> str:
    """Map a 0..1 cut-through index to an RGBA colour (red→amber→grey)."""
    if score is None:
        return "180,180,180,80"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "180,180,180,80"
    import math
    if math.isnan(s):
        return "180,180,180,80"
    # 0 = grey, 0.5 = amber, 1.0 = deep red
    s = max(0.0, min(1.0, s))
    r = int(180 + (255 - 180) * s)
    g = int(180 + (50 - 180) * s)
    b = int(180 + (50 - 180) * s)
    return f"{r},{g},{b},140"


# ---------------------------------------------------------------------------
# 3. Edge data (observed counts/speeds per OSM way)
# ---------------------------------------------------------------------------


def build_edgedata(out_path: Path,
                   osm_lookup: dict[int, list[str]] | None = None) -> int:
    """Write per-OSM-way observed volume + speed.

    SUMO ``.net.xml`` edge IDs follow the convention ``<osm_way_id>``
    or ``-<osm_way_id>`` for the reverse direction when produced by
    ``netconvert --osm-files``. Our parquet rows are keyed by OSM
    way id, so we emit one ``<edge id="<id>">`` per measured way
    with both observed averages.

    The user can colour the network by ``avg_volume`` or
    ``avg_speed_mph`` via View > Edit Edge Visualisation in
    sumo-gui after loading the file.
    """
    _section("Edge data")
    ss_path = CANONICAL_DIR / CanonicalFiles.streetscanner_segments
    if not ss_path.exists():
        _say("streetscanner_segments not built; "
             "run scripts/00_build_datasets.py first.")
        return 0

    import geopandas as gpd
    import pandas as pd

    gdf = gpd.read_parquet(ss_path)
    # Use only "All Days / All Day" rows for the headline averages so the
    # edgedata reflects a single typical-day picture. Users can rebuild
    # with --day_part if they want hour-of-day breakdowns.
    sub = gdf[(gdf["day_type"] == "All Days")
              & (gdf["day_part_raw"] == "All Day")]
    if sub.empty:
        _say("no All Days / All Day rows found in streetscanner; using all rows")
        sub = gdf

    # Aggregate to one row per (osm_way_id, direction). SUMO's
    # bidi splitting is handled in the .net.xml; we attach the
    # average volume to both forward (<id>) and reverse (-<id>) edges.
    agg = sub.groupby("osm_way_id", dropna=True, as_index=False).agg(
        avg_volume=("avg_volume", "mean"),
        avg_speed_mph=("avg_speed_mph", "mean"),
        speed_limit_mph=("speed_limit_mph", "mean"),
        osm_name=("osm_name", "first"),
    )
    agg = agg.dropna(subset=["osm_way_id"])
    n_ways = len(agg)

    # SUMO's interval-based edge-data format. Load with the GUI via
    # File → Open EdgeData (or programmatically with ``--edgedata-files``).
    # Each <edge> element is keyed by the **SUMO edge id** (not the
    # OSM way id) so sumo-gui can colour-match without any manual
    # translation.
    #
    # The root element must be ``<meandata>`` (per SUMO's
    # ``meandata_file.xsd``); ``<data>`` causes "no declaration found
    # for element 'data'" in the strict XML validator. The ``<edge>``
    # type also does not permit ``<param>`` children — so we publish
    # the human-readable metadata (street name, posted limit, etc.)
    # to a sibling CSV (``leonia.edgedata.meta.csv``) for analysts who
    # want to join it back.
    meta_path = out_path.with_name(out_path.stem + ".meta.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows_written = 0
    n_ways_resolved = 0
    n_ways_unresolved = 0
    meta_rows: list[dict] = []
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<meandata>\n')
        fh.write('  <interval id="streetlight_observed" begin="0" end="86400">\n')
        for _, row in agg.iterrows():
            way = int(row["osm_way_id"])
            vol = row["avg_volume"]
            spd = row["avg_speed_mph"]
            lim = row["speed_limit_mph"]
            name = row["osm_name"] or ""

            # Resolve OSM way → SUMO edge id(s). When the lookup is
            # unavailable (i.e. no .net.xml yet) we fall back to the
            # bare OSM id — netconvert often *does* keep it on
            # residential streets that aren't merged.
            if osm_lookup:
                sumo_ids = osm_lookup.get(way, [])
                if sumo_ids:
                    n_ways_resolved += 1
                else:
                    n_ways_unresolved += 1
                    continue
            else:
                sumo_ids = [str(way), f"-{way}"]

            for sid in sumo_ids:
                attrs = [f'id="{sid}"']
                if pd.notna(vol):
                    attrs.append(f'entered="{int(round(float(vol)))}"')
                if pd.notna(spd):
                    attrs.append(f'speed="{float(spd) * 0.44704:.3f}"')
                fh.write(f'    <edge {" ".join(attrs)}/>\n')
                meta_rows.append({
                    "sumo_edge_id": sid,
                    "osm_way_id": way,
                    "street_name": name,
                    "avg_volume_per_day": (
                        int(round(float(vol))) if pd.notna(vol) else None
                    ),
                    "avg_speed_mph": (
                        round(float(spd), 1) if pd.notna(spd) else None
                    ),
                    "speed_limit_mph": (
                        round(float(lim), 1) if pd.notna(lim) else None
                    ),
                })
                n_rows_written += 1
        fh.write('  </interval>\n')
        fh.write('</meandata>\n')

    # Write the metadata sidecar.
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    if osm_lookup:
        _say(f"wrote {out_path.name} ({n_ways_resolved:,}/"
             f"{n_ways_resolved + n_ways_unresolved:,} OSM ways "
             f"resolved → {n_rows_written:,} SUMO edges)")
    else:
        _say(f"wrote {out_path.name} ({n_ways:,} OSM ways × 2 "
             "directions, no SUMO lookup available)")
    if meta_rows:
        _say(f"wrote {meta_path.name} ({len(meta_rows):,} edges with street names)")
    return n_rows_written


# ---------------------------------------------------------------------------
# 4. Demand (Bridge OD -> SUMO flows)
# ---------------------------------------------------------------------------


# Bridge OD day-part windows (canonical export uses 5 named windows,
# not 24 hourly bins). Each tuple is (label, begin_hour, end_hour).
BRIDGE_OD_WINDOWS: dict[int, tuple[str, int, int]] = {
    1: ("EarlyAM", 0, 6),
    2: ("PeakAM",  6, 10),
    3: ("MidDay", 10, 15),
    4: ("PeakPM", 15, 19),
    5: ("LatePM", 19, 24),
}


def _spatial_resolve_bridge_zones(
    osm_lookup: dict[int, list[str]],
    net_path: Path,
) -> dict[int, list[str]]:
    """Augment the OSM-way lookup with spatial matches for the Bridge OD zones.

    Thin wrapper around
    :func:`leonia_traffic.sumo.net_lookup.spatial_resolve_zones` that
    loads the Bridge OD zone parquet + SUMO edge geometries, and
    reports how many zones were rescued.
    """
    bz_path = CANONICAL_DIR / CanonicalFiles.bridge_od_zones
    if not bz_path.exists():
        return osm_lookup

    import geopandas as gpd
    import pandas as pd

    zones = gpd.read_parquet(bz_path)
    if zones.empty:
        return osm_lookup
    if "osm_way_id" not in zones.columns:
        zones["osm_way_id"] = zones["name"].str.extract(r"/ (\d+)").astype("Int64")

    edges = _load_sumo_edge_geometries(net_path)
    if edges.empty or edges.crs is None:
        return osm_lookup

    augmented = _spatial_resolve_zones(osm_lookup, zones, edges,
                                       max_distance_m=300.0)
    n_added = len(augmented) - len(osm_lookup)
    if n_added > 0:
        _say(f"  spatial fallback: matched {n_added} Bridge OD zones to "
             "the nearest SUMO edge.")
    return augmented


def build_flows(out_path: Path,
                osm_lookup: dict[int, list[str]] | None = None) -> int:
    """Write Bridge OD flows as SUMO ``<flow>`` entries.

    The Bridge OD export uses five named day-part windows (Early AM,
    Peak AM, Mid-Day, Peak PM, Late PM) rather than 24 hourly bins.
    Each ``(origin, destination, day_part_code)`` row in the canonical
    parquet becomes one ``<flow>`` element with:

    * ``begin``/``end`` matching the window in seconds since midnight
      (e.g. Peak AM → 21600..36000).
    * ``vehsPerHour`` equal to ``od_volume / window_hours``. The
      StreetLight ``od_volume`` is the *total* vehicles in the window,
      so we normalise to a per-hour rate that SUMO can spread over the
      window using its built-in stochastic departure scheduler.
    * ``from``/``to`` set to the OSM way id of each zone — these
      match the SUMO edge ids netconvert produces when reading the
      OSM input.

    The output also includes a ``<vType>`` for ``passenger`` that
    SUMO needs to know how to model the vehicles.
    """
    _section("Demand (Bridge OD flows)")
    od_path = CANONICAL_DIR / CanonicalFiles.bridge_od
    if not od_path.exists():
        _say("bridge_od not built; run scripts/00_build_datasets.py first.")
        return 0

    import pandas as pd

    od = pd.read_parquet(od_path)
    # Use the All-Days day type with the 5 known windows
    # (day_type_code=0, day_part_code in {1..5}). day_part_code=0 is
    # the All-Day total — redundant with the sum of the 5 windows, so
    # we skip it to avoid double-counting.
    sub = od[
        (od["day_type_code"] == 0)
        & (od["day_part_code"].isin(BRIDGE_OD_WINDOWS.keys()))
        & (od["origin_osm_way_id"].notna())
        & (od["destination_osm_way_id"].notna())
        & (od["od_volume"] > 0)
    ].copy()
    if sub.empty:
        _say("no window-level OD rows found; nothing to write.")
        return 0

    n_flows = 0
    n_skipped_no_origin = 0
    n_skipped_no_dest = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        fh.write('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        fh.write('  <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" '
                 'length="5.0" maxSpeed="22.22" guiShape="passenger"/>\n')

        for _, row in sub.iterrows():
            o = int(row["origin_osm_way_id"])
            d = int(row["destination_osm_way_id"])

            # Resolve OSM way ids to SUMO edge ids.
            if osm_lookup:
                o_edges = osm_lookup.get(o, [])
                d_edges = osm_lookup.get(d, [])
                if not o_edges:
                    n_skipped_no_origin += 1
                    continue
                if not d_edges:
                    n_skipped_no_dest += 1
                    continue
                # Use the first SUMO edge as a representative endpoint.
                # SUMO will route through the network from there.
                from_edge = o_edges[0]
                to_edge = d_edges[0]
            else:
                from_edge = str(o)
                to_edge = str(d)

            code = int(row["day_part_code"])
            label, hr_start, hr_end = BRIDGE_OD_WINDOWS[code]
            window_hours = hr_end - hr_start
            begin = hr_start * 3600
            end = hr_end * 3600
            vph = float(row["od_volume"]) / max(window_hours, 1)
            if vph <= 0:
                continue
            flow_id = f"od_{o}_to_{d}_{label}"
            fh.write(
                f'  <flow id="{flow_id}" type="passenger" '
                f'from="{from_edge}" to="{to_edge}" '
                f'begin="{begin}" end="{end}" '
                f'vehsPerHour="{vph:.2f}" '
                f'departLane="best" departSpeed="max"/>\n'
            )
            n_flows += 1
        fh.write('</routes>\n')
    if osm_lookup and (n_skipped_no_origin + n_skipped_no_dest) > 0:
        _say(f"  skipped {n_skipped_no_origin} flows with unmapped origin, "
             f"{n_skipped_no_dest} with unmapped destination")
    _say(f"wrote {out_path.name} ({n_flows:,} flows across "
         f"{len(BRIDGE_OD_WINDOWS)} time windows)")
    return n_flows


# ---------------------------------------------------------------------------
# 5. .sumocfg + README
# ---------------------------------------------------------------------------


def build_sumocfg(out_path: Path, *, have_net: bool,
                  have_polys: bool, have_flows: bool) -> None:
    """Write the master .sumocfg file that ties everything together.

    ``leonia.edgedata.xml`` is intentionally *not* listed in
    ``<additional-files>`` — it's an observed-data overlay (SUMO
    meandata schema), which the user loads from sumo-gui via
    File → Open EdgeData rather than at network-load time. Wiring
    it in as an additional file would cause SUMO to misinterpret
    the <edge> elements as network-edge definitions.
    """
    _section("SUMO config")
    additional_files: list[str] = []
    if have_polys:
        additional_files.append(POLY_NAME)

    route_files = [FLOWS_NAME] if have_flows else []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        fh.write('               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
        fh.write('  <input>\n')
        if have_net:
            fh.write(f'    <net-file value="{NET_NAME}"/>\n')
        else:
            fh.write(f'    <!-- net-file value="{NET_NAME}" '
                     '(produced by running netconvert on leonia.osm.xml) -->\n')
        if route_files:
            fh.write(f'    <route-files value="{",".join(route_files)}"/>\n')
        if additional_files:
            fh.write(f'    <additional-files value="{",".join(additional_files)}"/>\n')
        fh.write('  </input>\n')
        fh.write('  <time>\n')
        fh.write('    <begin value="0"/>\n')
        fh.write('    <end value="86400"/>\n')
        fh.write('  </time>\n')
        fh.write('  <processing>\n')
        fh.write('    <ignore-route-errors value="true"/>\n')
        fh.write('  </processing>\n')
        fh.write('  <report>\n')
        fh.write('    <verbose value="true"/>\n')
        fh.write('    <no-step-log value="true"/>\n')
        fh.write('  </report>\n')
        fh.write('</configuration>\n')
    _say(f"wrote {out_path.name}")


def write_readme(out_path: Path, *, have_net: bool, n_flows: int,
                 n_polys: int, n_edges: int) -> None:
    out_path.write_text(f"""# Leonia SUMO project

Generated by ``scripts/11_export_sumo.py`` from the canonical data
lake under ``data/processed/streetlight/``.

## Files

| File | Purpose |
|---|---|
| ``{OSM_NAME}`` | Raw OSM extract (input to netconvert). |
| ``{NET_NAME}`` | SUMO road network. {"Already built." if have_net else "**Build me:** run the netconvert command in the install section below."} |
| ``{POLY_NAME}`` | {n_polys} polygons: Leonia borough boundary + ZA zone polygons coloured by cut-through index. |
| ``{EDGEDATA_NAME}`` | {n_edges:,} SUMO edges with observed Street Scanner ``entered`` (avg daily volume) + ``speed`` (m/s). |
| ``leonia.edgedata.meta.csv`` | Per-edge metadata sidecar: ``sumo_edge_id``, ``osm_way_id``, ``street_name``, raw averages. SUMO's meandata schema doesn't allow string columns inside ``<edge>``, so the names live here. Useful for joining results back to street names in post-processing. |
| ``{FLOWS_NAME}`` | {n_flows:,} hourly OD flows from the Bridge OD analysis. |
| ``{CFG_NAME}`` | Master config — load this with sumo-gui to open everything. |

## Open in sumo-gui

```
# 1. Install SUMO if you haven't:
brew install sumo          # macOS
sudo apt install sumo sumo-tools sumo-doc     # Linux

# 2. (Skip if leonia.net.xml already exists.)
cd data/processed/sumo
netconvert --osm-files {OSM_NAME} --output-file {NET_NAME} \\
    --geometry.remove --ramps.guess --junctions.join \\
    --tls.guess-signals --tls.discard-simple --tls.join \\
    --no-turnarounds --proj.utm

# 3. Open the project (loads net + polys + edgedata + flows).
sumo-gui -c {CFG_NAME}

# 4. To run the simulation headlessly:
sumo -c {CFG_NAME}
```

## Colouring the network by observed data

The ``{EDGEDATA_NAME}`` file is an observed-data overlay (SUMO's
``meandata`` schema), not part of the network definition. Load it
separately:

1. ``sumo-gui -c {CFG_NAME}`` to open the network + demand + polygons.
2. File → Open EdgeData… and choose ``{EDGEDATA_NAME}``.
3. Edit → Edge Visualisation Settings → **Colour edges by** →
   *meandata* → pick ``entered`` (= observed average daily volume)
   or ``speed`` (= observed mean speed, m/s).
4. Click Apply. Heavily-trafficked corridors light up; cut-through
   residential blocks stand out against the local-only ones.

The companion ``leonia.edgedata.meta.csv`` file pairs each
``sumo_edge_id`` with its ``osm_way_id`` and ``street_name`` plus
the original mph values. Join it to SUMO simulation output (or to
the ``entered`` / ``speed`` attributes above) to label results by
street.

## Re-running the simulation with different assumptions

* **Different time window** — edit the ``<time>`` block in
  ``{CFG_NAME}`` (``begin`` / ``end`` are seconds since midnight).
* **Different demand** — replace ``{FLOWS_NAME}`` with a custom
  routes file. The exporter writes one ``<flow>`` per (origin,
  destination, hour); you can hand-edit or regenerate from the
  ZA hourly profiles for a peak-only run.
* **Speed-bump scenarios** — append ``<edge ... speed="x.x"/>``
  overrides to ``{EDGEDATA_NAME}`` (or to a dedicated
  ``leonia.additional.xml``) and add it to ``<additional-files>``
  in the config.

## Limitations

* Bridge OD only covers trips that touch the GWB approach.
  Internal Leonia-to-Leonia traffic is not in this demand file.
  See ``docs/DATA.md`` for the full coverage caveats.
* SUMO edge ids match OSM way ids, but ``netconvert`` may merge
  adjacent ways with identical attributes. If an edge from the
  flows file resolves to a merged segment, SUMO will still find
  the right edge as long as ``ignore-route-errors`` is on
  (it is, in the generated config).
* The ``speed`` column in ``{EDGEDATA_NAME}`` is the **observed
  mean speed**, not a target free-flow speed. Do not use it as a
  ``speed`` override on the live edges — use it only for
  visualisation. Speed-limit overrides should go in a separate
  ``additional`` file with ``<edge id="..." speed="x"/>``.
""", encoding="utf-8")
    _say(f"wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-net", action="store_true",
                        help="Skip OSM fetch + netconvert. Useful if you "
                             "already have leonia.net.xml.")
    parser.add_argument("--no-demand", action="store_true",
                        help="Skip writing leonia.flows.xml.")
    parser.add_argument("--no-polygons", action="store_true",
                        help="Skip writing leonia.poly.xml.")
    parser.add_argument("--no-edgedata", action="store_true",
                        help="Skip writing leonia.edgedata.xml.")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="Override the OSM extract bbox. Defaults to "
                             "config.LEONIA_BBOX_WGS84.")
    args = parser.parse_args(argv)

    SUMO_DIR.mkdir(parents=True, exist_ok=True)
    # SUMO bbox is slightly wider than LEONIA_BBOX_WGS84 to include the
    # full GWB approach (lower- and upper-level destination zones in
    # Bridge OD sit just east of Leonia's east edge). Override with
    # --bbox if you only want the borough proper.
    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        minx, miny, maxx, maxy = LEONIA_BBOX_WGS84
        # Widen east + south to capture the full GWB approach (Bridge
        # OD has destinations at the lower- and upper-level toll
        # plazas just outside the borough's east/south edge).
        bbox = (minx, miny - 0.005, maxx + 0.020, maxy)

    t0 = time.time()
    counts: dict[str, int] = {}
    have_net = False
    have_polys = False
    have_flows = False
    have_edgedata = False

    if not args.no_net:
        osm_ok = fetch_osm_xml(SUMO_DIR / OSM_NAME, bbox)
        if osm_ok:
            have_net = run_netconvert(
                SUMO_DIR / OSM_NAME, SUMO_DIR / NET_NAME,
            )
    else:
        have_net = (SUMO_DIR / NET_NAME).exists()

    # Build the OSM-way → SUMO-edge lookup once, after the network is
    # available. Both build_edgedata() and build_flows() use it.
    osm_lookup: dict[int, list[str]] = {}
    if (SUMO_DIR / NET_NAME).exists():
        osm_lookup = _load_osm_to_sumo_lookup(SUMO_DIR / NET_NAME)
        _say(f"OSM-way → SUMO-edge lookup: {len(osm_lookup):,} ways "
             f"mapped to {sum(len(v) for v in osm_lookup.values()):,} edges")
        # Extend the lookup with spatially-resolved matches for the
        # Bridge OD zones, since StreetLight's OSM ids don't always
        # match a fresh OSM extract.
        osm_lookup = _spatial_resolve_bridge_zones(
            osm_lookup, SUMO_DIR / NET_NAME,
        )

    if not args.no_polygons:
        counts["polygons"] = build_polygons(SUMO_DIR / POLY_NAME)
        have_polys = counts["polygons"] > 0

    if not args.no_edgedata:
        counts["edges"] = build_edgedata(SUMO_DIR / EDGEDATA_NAME, osm_lookup)
        have_edgedata = counts["edges"] > 0

    if not args.no_demand:
        counts["flows"] = build_flows(SUMO_DIR / FLOWS_NAME, osm_lookup)
        have_flows = counts["flows"] > 0

    build_sumocfg(
        SUMO_DIR / CFG_NAME,
        have_net=have_net, have_polys=have_polys,
        have_flows=have_flows,
    )
    write_readme(
        SUMO_DIR / "README_SUMO.md",
        have_net=have_net,
        n_flows=counts.get("flows", 0),
        n_polys=counts.get("polygons", 0),
        n_edges=counts.get("edges", 0),
    )

    manifest = {
        "built_at": _now_iso(),
        "bbox_wgs84": list(bbox),
        "files": {
            f.name: {
                "exists": (SUMO_DIR / f.name).exists(),
                "bytes": (SUMO_DIR / f.name).stat().st_size
                if (SUMO_DIR / f.name).exists() else 0,
            }
            for f in [
                Path(OSM_NAME), Path(NET_NAME), Path(POLY_NAME),
                Path(EDGEDATA_NAME), Path(FLOWS_NAME), Path(CFG_NAME),
            ]
        },
        "counts": counts,
        "have_net": have_net,
    }
    (SUMO_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print(f"\nBuilt SUMO project in {time.time() - t0:.1f}s")
    print(f"Output:  {SUMO_DIR}")
    print(f"Open with: sumo-gui -c {SUMO_DIR / CFG_NAME}")
    if not have_net:
        print("\n  NOTE: leonia.net.xml not built (netconvert missing or OSM fetch failed).")
        print("  See README_SUMO.md for the one-liner to finish the conversion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
