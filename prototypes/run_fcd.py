"""Run a real SUMO simulation with --fcd-output and pack the per-vehicle
trajectories for the deck.gl TripsLayer prototype.

Unlike the aggregated edge-flow extractor, this captures **every vehicle's
position over time** (floating-car data), which is exactly what deck.gl's
TripsLayer wants. We drive the existing peak-AM slice demand (07:00-08:00,
~218 vehicles) so the run is fast and the output stays small.

Key SUMO flags:
  --fcd-output PATH        write per-step vehicle positions
  --fcd-output.geo true    emit lon/lat instead of projected x/y
  --device.fcd.period N    subsample to every N seconds (smaller file)

We call the `sumo` binary directly (not libsumo) so there's no pyarrow
filesystem-scheme conflict and no in-process state to manage.

Run from the repo root:

    venv/bin/python prototypes/run_fcd.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from sumolib import checkBinary

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = REPO_ROOT / "data/sumo/base"
NET = SUMO_DIR / "leonia.net.xml"
OUT_JS = Path(__file__).resolve().parent / "leonia_fcd_data.js"


def run_sumo_fcd(routes: Path, fcd_out: Path, begin: int, end: int,
                 period: int, seed: int) -> None:
    sumo = checkBinary("sumo")
    cmd = [
        sumo,
        "--net-file", str(NET),
        "--route-files", str(routes),
        "--fcd-output", str(fcd_out),
        "--fcd-output.geo", "true",
        "--device.fcd.period", str(period),
        "--begin", str(begin),
        "--end", str(end),
        "--step-length", "1",
        "--seed", str(seed),
        "--ignore-route-errors", "true",
        "--no-step-log", "true",
        "--time-to-teleport", "300",
    ]
    print("Running SUMO:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_fcd(fcd_out: Path, t0: int, precision: int = 5):
    """Stream the FCD XML into per-vehicle trajectories.

    Returns ``{veh_id: {"path": [[lon,lat],...], "ts": [s,...],
    "spd": [m/s,...]}}`` with timestamps rebased so the first sampled
    second is 0.
    """
    trips: dict[str, dict] = {}
    for _evt, elem in ET.iterparse(fcd_out, events=("end",)):
        if elem.tag != "timestep":
            continue
        t = float(elem.get("time", 0.0)) - t0
        for v in elem.findall("vehicle"):
            vid = v.get("id")
            lon = float(v.get("x"))   # geo output: x=lon, y=lat
            lat = float(v.get("y"))
            spd = float(v.get("speed", 0.0))
            d = trips.setdefault(vid, {"path": [], "ts": [], "spd": []})
            d["path"].append([round(lon, precision), round(lat, precision)])
            d["ts"].append(round(t, 1))
            d["spd"].append(round(spd, 2))
        elem.clear()
    return trips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routes", type=Path,
                    default=SUMO_DIR / "leonia.routes_peak_am_slice.xml")
    ap.add_argument("--begin", type=int, default=25200)   # 07:00
    ap.add_argument("--end", type=int, default=29400)     # 08:10 (let cars clear)
    ap.add_argument("--period", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fcd", type=Path, default=Path("/tmp/leonia_fcd.xml"))
    ap.add_argument("--reuse-fcd", action="store_true",
                    help="skip the SUMO run and re-parse an existing --fcd file")
    ap.add_argument("--per-origin-cap", type=int, default=0,
                    help="max trips kept per origin (0 = no cap). Keeps every "
                         "street present while thinning busy ones — preserves "
                         "residential coverage at a fraction of the size.")
    ap.add_argument("--coord-precision", type=int, default=5)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if not args.reuse_fcd:
        run_sumo_fcd(args.routes, args.fcd, args.begin, args.end,
                     args.period, args.seed)
    else:
        print(f"Reusing existing FCD file {args.fcd} (skipping SUMO).")

    print(f"Parsing {args.fcd} ...")
    trips_raw = parse_fcd(args.fcd, t0=args.begin, precision=args.coord_precision)
    print(f"  {len(trips_raw)} vehicles")

    # Keep trips with at least two waypoints (a single point can't draw).
    trips = []
    for vid, d in trips_raw.items():
        if len(d["path"]) < 2:
            continue
        mean_spd = sum(d["spd"]) / len(d["spd"])
        trips.append({
            "id": vid,
            "path": d["path"],
            "ts": d["ts"],
            "mean_mph": round(mean_spd / 0.44704, 1),
        })

    # Per-origin subsampling: a full 24h run is ~33k vehicles / >100 MB of
    # JSON — far too heavy for the browser, and most of those are short
    # residential trips. Capping trips *per origin* keeps every street
    # present (coverage) while thinning the busy ones (size). Trips are
    # spread across the day per origin so the diurnal pattern survives.
    # This mirrors the production advice: subsample for the client, keep
    # the full set on the host.
    n_before = len(trips)
    if args.per_origin_cap:
        import random
        random.seed(args.seed)
        # Origin key = vehicle id minus the hour token and the .N replica
        # suffix: "za_3356462_h21_allday.1" -> "za_3356462";
        # "od_10030557_to_590551_H07_wkd.4" -> "od_10030557_to_590551".
        groups: dict[str, list] = {}
        for t in trips:
            key = re.sub(r"_[hH]\d+.*$", "", t["id"])
            groups.setdefault(key, []).append(t)
        capped = []
        for key, members in groups.items():
            if len(members) > args.per_origin_cap:
                members = random.sample(members, args.per_origin_cap)
            capped.extend(members)
        trips = capped
        print(f"  per-origin cap {args.per_origin_cap}: {n_before} -> "
              f"{len(trips)} vehicles across {len(groups)} origins")

    durations = [t["ts"][-1] for t in trips]
    max_t = max(durations) if durations else (args.end - args.begin)
    waypoints = sum(len(t["path"]) for t in trips)

    title = args.title or "Leonia FCD — real vehicle trajectories"
    payload = {
        "meta": {
            "title": title,
            "begin_s": args.begin,
            "max_t": max_t,
            "n_vehicles": len(trips),
            "n_vehicles_total": n_before,
            "n_waypoints": waypoints,
            "period_s": args.period,
        },
        "trips": trips,
    }
    OUT_JS.write_text(
        "// Auto-generated by prototypes/run_fcd.py — real SUMO FCD. Do not edit.\n"
        "window.LEONIA_FCD = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    size_mb = OUT_JS.stat().st_size / 1e6
    print(f"\nWrote {OUT_JS}  ({size_mb:.2f} MB)")
    print(f"  vehicles: {len(trips)}  waypoints: {waypoints}  "
          f"window: 0..{max_t:.0f}s")


if __name__ == "__main__":
    main()
