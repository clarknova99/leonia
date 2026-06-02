"""Recreate the full 24-hour weekday baseline and pack per-block flow for
the front-end prototypes — including every residential block that carries
any traffic at all, no matter how small.

Approach: run the real `sumo` binary on the existing weekday-24h routes with
an ``edgeData`` detector that writes per-edge throughput every 15 minutes.
We use mesoscopic mode (``--mesosim``) so a 24-hour, ~30k-vehicle run finishes
in seconds, and we keep **every edge that ever sees a vehicle** (no vph floor),
so the smallest residential streets light up too.

This calls `sumo` as a subprocess (not libsumo), so there is no pyarrow
filesystem-scheme conflict and nothing parquet-related to manage.

Run from the repo root:

    venv/bin/python prototypes/run_baseline_24h.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from sumolib import checkBinary

from leonia_traffic.sumo import demand_builder as db
from leonia_traffic.sumo.net_lookup import (
    load_meta_lookup,
    load_sumo_edge_geometries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = REPO_ROOT / "data/sumo/base"
NET = SUMO_DIR / "leonia.net.xml"
META = SUMO_DIR / "leonia.edgedata.meta.csv"
OUT_JS = Path(__file__).resolve().parent / "leonia_flow_data.js"
# Prototype-local routes: weekday bridge-OD magnitude + ALL-DAYS residential
# hourly shape (≈152 ZA zones vs. the 59 in the Mon-Thu production demand).
# Kept separate so we never mutate the webapp's precache demand.
PROTO_ROUTES = SUMO_DIR / "leonia.routes_prototype_weekday_allday.xml"


def build_prototype_routes(out: Path) -> tuple[int, int]:
    """Write a routes file = weekday Bridge-OD + All-Days ZA hourly shape.

    The production ``bridge_od_weekday_24h`` demand draws its residential
    shape from the Mon-Thu ZA hourly profile, which StreetLight only
    publishes for ~59 zones. Swapping to the All-Days hourly profile
    (``day_type_codes=None``) lifts residential coverage to ~152 zones
    while keeping the weekday Bridge-OD arterial magnitudes. Returns
    ``(n_flows, n_unique_za_origins)``.
    """
    osm = db._load_bridge_od_lookup(NET)
    flows = list(db._bridge_od_flows(
        osm, day_type_codes=db.WEEKDAY_DAY_TYPE_CODES, label_suffix="wkd",
    ))
    flows += db._za_hourly_flows(
        osm, NET, day_type_codes=None, label_suffix="allday",
        scale=db.ZA_VISIBILITY_SCALE_DEFAULT,
    )
    flows.sort(key=lambda f: (f.begin_s, f.end_s, f.flow_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write(
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '        xsi:noNamespaceSchemaLocation="'
            'http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        )
        fh.write(
            '  <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" '
            'length="5.0" maxSpeed="22.22" guiShape="passenger"/>\n'
        )
        for f in flows:
            fh.write(f.to_xml())
        fh.write('</routes>\n')
    za_origins = {f.flow_id.rsplit("_h", 1)[0] for f in flows
                  if f.flow_id.startswith("za_")}
    return len(flows), len(za_origins)


def _round_coords(coords, ndigits: int = 5):
    return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in coords]


def run_sumo_edgedata(routes: Path, edgedata_out: Path, add_file: Path,
                      begin: int, end: int, period: int, seed: int,
                      meso: bool) -> None:
    add_file.write_text(
        '<additional>\n'
        f'  <edgeData id="flow24" file="{edgedata_out}" period="{period}" '
        f'begin="{begin}" end="{end}" excludeEmpty="true"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )
    sumo = checkBinary("sumo")
    cmd = [
        sumo,
        "--net-file", str(NET),
        "--route-files", str(routes),
        "--additional-files", str(add_file),
        "--begin", str(begin),
        "--end", str(end),
        "--seed", str(seed),
        "--ignore-route-errors", "true",
        "--no-step-log", "true",
        "--time-to-teleport", "300",
    ]
    if meso:
        cmd += ["--mesosim"]
    print("Running SUMO:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_edgedata(edgedata_out: Path, period: int):
    """Stream edgeData XML -> {edge_id: {frame_s: (vph, speed_ms)}} and frame list.

    vph is derived from ``entered`` (vehicles entering the edge in the
    interval) scaled to an hourly rate; falls back to ``left`` then to a
    density-based proxy so meso/micro attribute differences don't drop edges.
    """
    scale = 3600.0 / period
    by_edge: dict[str, dict[int, tuple[float, float]]] = {}
    frame_set: set[int] = set()
    for _evt, interval in ET.iterparse(edgedata_out, events=("end",)):
        if interval.tag != "interval":
            continue
        fs = int(float(interval.get("begin", 0.0)))
        frame_set.add(fs)
        for e in interval.findall("edge"):
            eid = e.get("id")
            entered = e.get("entered")
            left = e.get("left")
            if entered is not None:
                count = float(entered)
            elif left is not None:
                count = float(left)
            else:
                count = float(e.get("departed", 0) or 0) + float(e.get("arrived", 0) or 0)
            speed = float(e.get("speed", 0.0) or 0.0)
            if count <= 0 and speed <= 0:
                continue
            by_edge.setdefault(eid, {})[fs] = (count * scale, speed)
        interval.clear()
    return by_edge, sorted(frame_set)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--za-shape", choices=["all_days", "weekday"],
                    default="all_days",
                    help="residential hourly shape: 'all_days' (~152 zones, "
                         "default) or 'weekday' (~59 zones, matches production)")
    ap.add_argument("--routes", type=Path, default=None,
                    help="explicit routes file (overrides --za-shape)")
    ap.add_argument("--begin", type=int, default=0)
    ap.add_argument("--end", type=int, default=90000)   # 24h + drain buffer
    ap.add_argument("--period", type=int, default=900)  # 15-min frames
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro", action="store_true",
                    help="use microscopic mode (slower) instead of mesosim")
    ap.add_argument("--edgedata", type=Path, default=Path("/tmp/leonia_edgedata24.xml"))
    args = ap.parse_args()

    if args.routes is not None:
        routes = args.routes
    elif args.za_shape == "all_days":
        n_flows, n_za = build_prototype_routes(PROTO_ROUTES)
        print(f"Built prototype routes (All-Days ZA shape): {n_flows} flows, "
              f"{n_za} residential origins -> {PROTO_ROUTES.name}")
        routes = PROTO_ROUTES
    else:
        routes = SUMO_DIR / "leonia.routes_bridge_od_weekday_24h.xml"

    run_sumo_edgedata(
        routes, args.edgedata, Path("/tmp/leonia_edgedata24_add.xml"),
        args.begin, args.end, args.period, args.seed, meso=not args.micro,
    )

    print(f"Parsing {args.edgedata} ...")
    by_edge, frame_keys = parse_edgedata(args.edgedata, args.period)
    # Drop the post-midnight drain frames so the clock reads 00:00-23:45.
    frame_keys = [f for f in frame_keys if f < 86400]
    frame_index = {fs: i for i, fs in enumerate(frame_keys)}
    n_frames = len(frame_keys)
    print(f"  {len(by_edge)} edges with any flow, {n_frames} frames")

    print("Loading geometry + names ...")
    geo = load_sumo_edge_geometries(NET).set_index("edge_id")
    meta = load_meta_lookup(META)
    name_by_edge = {}
    if not meta.empty:
        for r in meta.itertuples(index=False):
            nm = getattr(r, "street_name", None)
            if isinstance(nm, str) and nm and nm != "nan":
                name_by_edge[r.sumo_edge_id] = nm

    edges_out = []
    all_vmax = []
    for eid, frames in by_edge.items():
        if eid not in geo.index:
            continue
        geom = geo.loc[eid, "geometry"]
        if hasattr(geom, "iloc"):
            geom = geom.iloc[0]
        if geom is None or geom.is_empty:
            continue
        vph = [0] * n_frames
        for fs, (v, _spd) in frames.items():
            if fs in frame_index:
                vph[frame_index[fs]] = int(round(v))
        peak = max(vph) if vph else 0
        if peak <= 0:
            continue
        all_vmax.append(peak)
        edges_out.append({
            "name": name_by_edge.get(eid, eid),
            "coords": _round_coords(list(geom.coords)),
            "vph": vph,
        })

    # vmax for the colour ramp = p95 of per-edge peaks (keeps the busy
    # arterials from washing out the residential signal).
    all_vmax.sort()
    vmax = max(50, int(all_vmax[int(len(all_vmax) * 0.95)])) if all_vmax else 200

    # Thin grey skeleton: edges with no flow at all, for map context.
    active_ids = {e["name"] for e in edges_out}  # not used directly
    flow_edge_ids = set(by_edge.keys())
    skeleton = []
    for eid, srow in geo.iterrows():
        if eid in flow_edge_ids:
            continue
        g = srow.geometry
        if g is None or g.is_empty:
            continue
        skeleton.append(_round_coords(list(g.coords)))

    pts = [p for e in edges_out for p in e["coords"]]
    center = [round(sum(p[0] for p in pts) / len(pts), 5),
              round(sum(p[1] for p in pts) / len(pts), 5)]

    labels = []
    for fs in frame_keys:
        h, m = divmod(fs // 60, 60)
        labels.append(f"{h:02d}:{m:02d}")

    payload = {
        "meta": {
            "title": "Leonia simulated traffic — 24h weekday baseline",
            "frame_minutes": args.period // 60,
            "vmax_vph": vmax,
            "center": center,
            "zoom": 14,
            "n_active_edges": len(edges_out),
            "n_frames": n_frames,
        },
        "frames": labels,
        "skeleton": skeleton,
        "edges": edges_out,
    }
    OUT_JS.write_text(
        "// Auto-generated by prototypes/run_baseline_24h.py — 24h weekday baseline.\n"
        "window.LEONIA_FLOW = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    size_mb = OUT_JS.stat().st_size / 1e6
    print(f"\nWrote {OUT_JS}  ({size_mb:.2f} MB)")
    print(f"  active edges: {len(edges_out)}  skeleton: {len(skeleton)}  "
          f"frames: {n_frames}  vmax_vph: {vmax}")


if __name__ == "__main__":
    main()
