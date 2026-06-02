"""Stakeholder-facing visual outputs for SUMO simulation runs.

Public API:

* :func:`build_animated_map` — folium time-slider map of edge volumes.
* :func:`build_dual_compare_map` — synchronised baseline-vs-scenario
  side-by-side map.
* :func:`build_sparkline` — Plotly mini-chart per suspect street.
* :func:`build_stakeholder_html` — single self-contained council-meeting
  one-pager that embeds all of the above plus KPIs and a demographic
  overlay.

Like :mod:`leonia_traffic.sumo.scoring`, these functions read parquets
(via geopandas / pandas) — they must therefore run in a process that
has not yet imported ``libsumo``. Always call them after
:meth:`SumoRuntime.close`, never alongside an active runtime.

The functions intentionally accept *DataFrames* rather than file paths
so they're easy to unit-test and easy to call from a notebook.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from leonia_traffic.config import SUMO_BASE_DIR
from leonia_traffic.data.dataset_io import (
    CANONICAL_DIR,
    CRASHES_DIR,
    DERIVED_DIR,
    CanonicalFiles,
    CrashFiles,
    DerivedFiles,
)
from leonia_traffic.sumo.net_lookup import load_sumo_edge_geometries

logger = logging.getLogger(__name__)


SUMO_DIR = SUMO_BASE_DIR
DEFAULT_NET_PATH = SUMO_DIR / "leonia.net.xml"


# ---------------------------------------------------------------------------
# Borough bounding box
# ---------------------------------------------------------------------------
#
# Rough geographic extent of Leonia plus a ~1.3 km buffer on every
# side. The SUMO network covers roughly lon [-74.030, -73.940] ×
# lat [40.847, 40.895]; the buffer here is generous enough to admit
# crashes geocoded just outside the borough (e.g. on a side street
# whose centerline is on the Palisades Park / Englewood line) but
# tight enough to reject the NJDOT rows that geocoded to a faraway
# stretch of I-95 / NJ Turnpike. Used by the crash visualisations
# as a defense-in-depth filter on top of the road-system check.
LEONIA_BBOX_MIN_LON: float = -74.05
LEONIA_BBOX_MAX_LON: float = -73.92
LEONIA_BBOX_MIN_LAT: float = 40.83
LEONIA_BBOX_MAX_LAT: float = 40.92

# Leonia borough polygon buffered by ~150 m, in WGS84. Used as a
# defense-in-depth filter for state-system crashes whose snap
# misses far outside the borough (see filter layer 4 in
# :func:`_filter_crash_rows_to_borough`). Cached lazily because
# loading the geojson + projecting to UTM is ~30 ms and we hit
# this on every crash render.
_LEONIA_POLYGON_BUFFER_M: float = 150.0
_LEONIA_POLYGON_BUFFERED = None


def _leonia_polygon_buffered():
    """Return Leonia's polygon buffered by ~150m in WGS84 (cached).

    Returns ``None`` if the borough geojson is missing — callers
    must treat that as "skip the polygon filter, fall back to bbox
    only" so the test harness and minimal installs still work.
    """
    global _LEONIA_POLYGON_BUFFERED
    if _LEONIA_POLYGON_BUFFERED is not None:
        return _LEONIA_POLYGON_BUFFERED

    try:
        from leonia_traffic.config import load_leonia_polygon
        import geopandas as _gpd

        poly = load_leonia_polygon()
    except Exception:
        return None

    try:
        gs = _gpd.GeoSeries([poly], crs="EPSG:4326")
        buf_m = gs.to_crs("EPSG:32618").buffer(_LEONIA_POLYGON_BUFFER_M)
        _LEONIA_POLYGON_BUFFERED = buf_m.to_crs("EPSG:4326").iloc[0]
    except Exception:
        return None
    return _LEONIA_POLYGON_BUFFERED


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _vph_to_color(vph: float, vmax: float = 800.0) -> str:
    """Green → yellow → red ramp, returns a CSS hex string."""
    if vph is None or pd.isna(vph) or vmax <= 0:
        return "#cccccc"
    t = max(0.0, min(1.0, float(vph) / vmax))
    # Piecewise lerp for legibility on a folium dark/light tile.
    if t < 0.5:
        # green → yellow
        r = int(255 * (t * 2))
        g = 200
        b = 50
    else:
        # yellow → red
        r = 255
        g = int(200 * (1 - (t - 0.5) * 2))
        b = 50
    return f"#{r:02x}{g:02x}{b:02x}"


# Absolute vph thresholds calibrated to Leonia's road classes:
#
#   ≤  20 vph  — pure green (residential idle)
#   ≤  80 vph  — green → yellow-green (residential busy)
#   ≤ 150 vph  — yellow-green → yellow (collector emerging)
#   ≤ 300 vph  — yellow → orange (collector / minor arterial)
#   ≤ 600 vph  — orange → red (arterial)
#   > 600 vph  — saturating red
#
# Encoded as (vph, "#hex") anchors; we lerp linearly between them.
# Stops are tuned so that "green is fine, yellow means worth a look,
# red means heavy traffic" matches a stakeholder's intuition.
_ABSOLUTE_VPH_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0,    (0,   180, 80)),    # vivid green
    (20.0,   (60,  200, 60)),    # still distinctly green
    (80.0,   (180, 220, 40)),    # green-yellow
    (150.0,  (255, 215, 0)),     # yellow
    (300.0,  (255, 140, 0)),     # orange
    (600.0,  (220, 30,  30)),    # red
    (1200.0, (170, 0,   0)),     # deep red (saturating)
)


def _vph_to_color_absolute(vph: float) -> str:
    """Map a vph value to a colour using absolute Leonia thresholds.

    See :data:`_ABSOLUTE_VPH_STOPS` for the calibration. Anything
    above the top stop pins to the deep-red anchor so saturation
    doesn't rotate around the colour wheel.
    """
    if vph is None or pd.isna(vph):
        return "#cccccc"
    v = float(vph)
    if v <= _ABSOLUTE_VPH_STOPS[0][0]:
        r, g, b = _ABSOLUTE_VPH_STOPS[0][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    if v >= _ABSOLUTE_VPH_STOPS[-1][0]:
        r, g, b = _ABSOLUTE_VPH_STOPS[-1][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    for (v_lo, c_lo), (v_hi, c_hi) in zip(
        _ABSOLUTE_VPH_STOPS, _ABSOLUTE_VPH_STOPS[1:],
    ):
        if v_lo <= v < v_hi:
            t = (v - v_lo) / (v_hi - v_lo)
            r = int(c_lo[0] + (c_hi[0] - c_lo[0]) * t)
            g = int(c_lo[1] + (c_hi[1] - c_lo[1]) * t)
            b = int(c_lo[2] + (c_hi[2] - c_lo[2]) * t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#cccccc"


def _vph_to_width_absolute(vph: float) -> float:
    """Stroke width on the absolute scale: 2 px (idle) → 9 px (saturating).

    Uses the same anchor schedule as the colour ramp so width and
    colour escalate together — readers don't have to do mental
    math to see "thicker = busier" matching "redder = busier".
    """
    if vph is None or pd.isna(vph):
        return 2.0
    v = float(vph)
    if v <= 20.0:
        return 2.0
    if v >= 600.0:
        return 9.0
    if v <= 150.0:
        return 2.0 + (v - 20.0) / (150.0 - 20.0) * 3.0  # 2 → 5
    if v <= 300.0:
        return 5.0 + (v - 150.0) / (300.0 - 150.0) * 2.0  # 5 → 7
    return 7.0 + (v - 300.0) / (600.0 - 300.0) * 2.0  # 7 → 9


def _vph_to_width(vph: float, vmax: float = 800.0) -> float:
    """Stroke width 1 px (low) → 8 px (saturating)."""
    if vph is None or pd.isna(vph) or vmax <= 0:
        return 1.0
    t = max(0.0, min(1.0, float(vph) / vmax))
    return 1.0 + 7.0 * t


def _seconds_to_clock_label(seconds: float) -> str:
    """Format an offset-from-midnight as ``H:MM AM/PM``.

    Used by the animation player's clock and edge tooltips.
    Stakeholders read the dual map as a wall-clock day, not a
    SUMO-second offset, and the 12-hour AM/PM convention is what
    everyone in town uses. ``00:00`` becomes ``12:00 AM``,
    ``13:30`` becomes ``1:30 PM``. Frames past 24:00 (an
    extended SUMO run) wrap modulo 24 so the label keeps making
    sense.
    """
    sec = int(seconds) % (24 * 3600)
    hour24 = sec // 3600
    minute = (sec % 3600) // 60
    suffix = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12}:{minute:02d} {suffix}"


def load_crash_points_if_available() -> pd.DataFrame | None:
    """Best-effort load of geocoded crashes for the animated map overlay.

    Returns ``None`` if ``data/processed/crashes/njdot_crashes.parquet``
    hasn't been built yet — the caller should treat that as "skip the
    safety overlay this run." Run :mod:`scripts.14_build_crash_overlay`
    once to populate it.
    """
    path = CRASHES_DIR / CrashFiles.crashes
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("could not read crash parquet (%s): %s", path, exc)
        return None


def load_crash_segments_if_available() -> pd.DataFrame | None:
    """Best-effort load of the per-segment crash aggregate."""
    path = CRASHES_DIR / CrashFiles.crashes_by_segment
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("could not read crashes_by_segment (%s): %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Animated map
# ---------------------------------------------------------------------------


def _edge_history_to_frames(
    edge_history: pd.DataFrame,
    *,
    frame_seconds: int = 900,
    sample_interval_s: int = 60,
) -> pd.DataFrame:
    """Aggregate the per-bin history into per-frame vehicle counts.

    The runtime writes per-bin samples (default 60 s); for the
    animated map we want coarser frames (default 15 min). We sum
    each edge's vehicle counts across the bins inside each frame
    and convert to a vph rate (the visual property the map encodes).
    """
    if edge_history.empty:
        return edge_history.copy()
    df = edge_history.copy()
    df["frame_s"] = (df["t_bin_s"] // frame_seconds) * frame_seconds
    grouped = df.groupby(
        ["frame_s", "sumo_edge_id"], as_index=False
    ).agg(
        vehicles_peak=("vehicles", "max"),
        mean_speed_ms=("mean_speed_ms", "mean"),
    )
    # Treat the per-bin sample as "vehicles present at the bin" (a
    # snapshot, not an arrival count). The flow rate is therefore the
    # *peak* vehicles seen in any bin within the frame, scaled to vph.
    scale_to_hour = 3600.0 / sample_interval_s
    grouped["vph"] = grouped["vehicles_peak"] * scale_to_hour
    grouped["mean_speed_mph"] = grouped["mean_speed_ms"] / 0.44704

    # Carry street_name / osm_way_id through if the runtime joined
    # them onto edge_history. We pick the first observation per edge
    # since these are static metadata.
    label_cols = [c for c in ("street_name", "osm_way_id")
                  if c in df.columns]
    if label_cols:
        labels = df.groupby("sumo_edge_id", as_index=False)[label_cols].first()
        grouped = grouped.merge(labels, on="sumo_edge_id", how="left")
    return grouped


def _round_coords(
    coords: Iterable[tuple[float, float]], ndigits: int = 5
) -> list[list[float]]:
    """[(lon, lat), ...] -> [[lon, lat], ...] rounded to keep payloads small."""
    return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in coords]


def build_flow_payload(
    edge_history: pd.DataFrame,
    net_path: str | Path,
    *,
    frame_minutes: int = 15,
    sample_interval_s: int = 60,
    min_vph: float = 20.0,
    max_skeleton: int = 4000,
    title: str = "Leonia simulated traffic",
    zoom: int = 14,
    pin_edge_ids: Iterable[str] | None = None,
    baseline_history: pd.DataFrame | None = None,
) -> dict:
    """Build the compact flow dataset the deck.gl front-end consumes.

    This is the single source of truth for the ``window.LEONIA_FLOW`` /
    ``flow.json`` schema. It is shared by the offline prototype
    extractor (``prototypes/extract_flow_data.py``) and the webapp
    precache builder so both stay byte-for-byte compatible.

    Returns a dict with:

    * ``meta`` — title, vmax_vph (95th pct), centre, zoom, counts,
      ``has_baseline``.
    * ``frames`` — ``["00:00", "00:15", ...]`` clock labels.
    * ``skeleton`` — grey context line geometries (no traffic).
    * ``edges`` — active edges, each ``{id, name, coords, vph[]}`` with
      one vph slot per frame (0 where idle). When ``baseline_history``
      is supplied each edge also carries ``base[]`` (the baseline vph
      aligned to the same frames) so the front-end can show the change
      this scenario caused (e.g. closing a street -> more vph on the
      parallel route).

    ``pin_edge_ids`` forces those edges into ``edges`` even when they
    carry little or no traffic (e.g. a closed street, which would
    otherwise vanish into the grey skeleton). This keeps the selected
    street highlightable on the front-end regardless of its load.
    """
    geo = load_sumo_edge_geometries(Path(net_path)).set_index("edge_id")
    pinned = {str(e) for e in (pin_edge_ids or [])}

    frame_seconds = frame_minutes * 60
    frames = _edge_history_to_frames(
        edge_history,
        frame_seconds=frame_seconds,
        sample_interval_s=sample_interval_s,
    )

    if frames.empty:
        return {
            "meta": {
                "title": title, "frame_minutes": frame_minutes,
                "vmax_vph": 50, "center": [-73.99, 40.86], "zoom": zoom,
                "n_active_edges": 0, "n_frames": 0, "has_baseline": False,
            },
            "frames": [], "skeleton": [], "edges": [],
        }

    frame_keys = sorted(int(s) for s in frames["frame_s"].unique())
    frame_index = {fs: i for i, fs in enumerate(frame_keys)}
    n_frames = len(frame_keys)

    # Optional baseline (unchanged-network) frames, used to compute the
    # per-edge change this scenario caused. Baseline frame_s buckets are
    # the same 15-min, seconds-of-day grid, so they align to the
    # scenario's frame_index directly.
    base_frames = None
    base_active: set[str] = set()
    if baseline_history is not None and not baseline_history.empty:
        base_frames = _edge_history_to_frames(
            baseline_history,
            frame_seconds=frame_seconds,
            sample_interval_s=sample_interval_s,
        )
        if base_frames.empty:
            base_frames = None
        else:
            bpeak = base_frames.groupby("sumo_edge_id")["vph"].max()
            base_active = set(bpeak[bpeak >= min_vph].index)
    has_baseline = base_frames is not None

    # Per-edge peak vph across the day decides who is an "active" edge.
    # Pinned edges (the scenario's target street) are always promoted so
    # they stay highlightable even at zero load (e.g. a closure), and
    # baseline-active edges are included so a street that *lost* all its
    # traffic in the scenario is still present (and shows up as impacted).
    peak = frames.groupby("sumo_edge_id")["vph"].max()
    active_ids = set(peak[peak >= min_vph].index)
    include_ids = (
        active_ids
        | {e for e in pinned if e in geo.index}
        | {e for e in base_active if e in geo.index}
    )

    positive = frames[frames["vph"] > 0]["vph"]
    vmax = max(50.0, round(float(positive.quantile(0.95)))) if not positive.empty else 50.0

    act = frames[frames["sumo_edge_id"].isin(include_ids)]
    vph_by_edge: dict[str, list[int]] = {eid: [0] * n_frames for eid in include_ids}
    names: dict[str, str] = {}

    def _record_name(eid: str, nm: object) -> None:
        if (
            isinstance(nm, str) and nm and nm != "nan"
            and eid not in names
        ):
            names[eid] = nm

    for row in act.itertuples(index=False):
        slot = frame_index[int(row.frame_s)]
        vph_by_edge[row.sumo_edge_id][slot] = int(round(row.vph))
        _record_name(row.sumo_edge_id, getattr(row, "street_name", None))

    base_by_edge: dict[str, list[int]] = {}
    if has_baseline:
        base_by_edge = {eid: [0] * n_frames for eid in include_ids}
        bact = base_frames[base_frames["sumo_edge_id"].isin(include_ids)]
        for row in bact.itertuples(index=False):
            slot = frame_index.get(int(row.frame_s))
            if slot is None:
                continue
            base_by_edge[row.sumo_edge_id][slot] = int(round(row.vph))
            _record_name(row.sumo_edge_id, getattr(row, "street_name", None))

    edges_out: list[dict] = []
    for eid in include_ids:
        if eid not in geo.index:
            continue
        geom = geo.loc[eid, "geometry"]
        if isinstance(geom, pd.Series):
            geom = geom.iloc[0]
        if geom is None or geom.is_empty:
            continue
        edge_obj = {
            "id": str(eid),
            "name": names.get(eid, str(eid)),
            "coords": _round_coords(list(geom.coords)),
            "vph": vph_by_edge[eid],
        }
        if has_baseline:
            edge_obj["base"] = base_by_edge[eid]
        edges_out.append(edge_obj)

    # Grey context skeleton: every other edge with geometry, capped.
    skeleton: list[list[list[float]]] = []
    for eid, srow in geo.iterrows():
        if eid in include_ids:
            continue
        geom = srow.geometry
        if geom is None or geom.is_empty:
            continue
        skeleton.append(_round_coords(list(geom.coords)))
        if len(skeleton) >= max_skeleton:
            break

    all_pts = [pt for e in edges_out for pt in e["coords"]]
    if all_pts:
        center = [
            round(sum(p[0] for p in all_pts) / len(all_pts), 5),
            round(sum(p[1] for p in all_pts) / len(all_pts), 5),
        ]
    else:
        center = [-73.99, 40.86]

    labels = []
    for fs in frame_keys:
        h, m = divmod(fs // 60, 60)
        labels.append(f"{h:02d}:{m:02d}")

    return {
        "meta": {
            "title": title,
            "frame_minutes": frame_minutes,
            "vmax_vph": int(vmax),
            "center": center,
            "zoom": zoom,
            "n_active_edges": len(edges_out),
            "n_frames": n_frames,
            "has_baseline": has_baseline,
        },
        "frames": labels,
        "skeleton": skeleton,
        "edges": edges_out,
    }


def write_flow_json(
    edge_history: pd.DataFrame,
    net_path: str | Path,
    out_path: str | Path,
    **kwargs,
) -> dict:
    """Build the flow payload and write it as compact ``flow.json``.

    Returns the payload so callers can log counts. See
    :func:`build_flow_payload` for the schema and tuning kwargs.
    """
    payload = build_flow_payload(edge_history, net_path, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return payload


def _seconds_to_iso(seconds: int, *, base_date: datetime | None = None) -> str:
    """Translate seconds-since-midnight into an ISO timestamp.

    folium's TimestampedGeoJson wants ISO strings; we synthesise a
    fictional date (today by default) so the time slider shows a
    24-hour clock.
    """
    if base_date is None:
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dt = base_date + timedelta(seconds=int(seconds))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def build_animated_map(
    edge_history: pd.DataFrame,
    out_html: Path,
    *,
    edges_geo: gpd.GeoDataFrame | None = None,
    net_path: Path = DEFAULT_NET_PATH,
    frame_minutes: int = 15,
    sample_interval_s: int = 60,
    vmax_vph: float | None = None,
    min_vph_for_animation: float = 1.0,
    title: str = "Leonia simulated traffic",
    center: tuple[float, float] | None = None,
    zoom: int = 14,
    crash_points: pd.DataFrame | None = None,
) -> Path:
    """Render an animated folium map with a time-slider control.

    The previous implementation used :class:`folium.plugins.TimestampedGeoJson`,
    but that plugin is built around point-geometry animation: it
    drops a marker on every LineString endpoint by default, doesn't
    reliably honour per-feature ``style`` for LineString features
    when the underlying ``leaflet-timedimension`` library splits
    the geometry by time, and offers no way to render a static base
    layer underneath the animation.

    This version builds a self-contained Leaflet map with a custom
    time slider:

    * **Static skeleton** — every SUMO edge as a 1.2 px light-grey
      line so the viewer has a stable mental map.
    * **One feature group per frame** — only edges with measurable
      flow (``vph >= min_vph_for_animation``) are rendered, with a
      saturating green→yellow→red colour ramp and weight 3–10 px.
    * **Custom HTML slider + play/pause** — ``input[type=range]``
      with a play button. Each tick toggles which feature group is
      visible. Frames are driven by sim time, so the slider label
      reads "07:00 → 07:15" etc.
    * **Pre-computed counts** in the slider tooltip — the viewer can
      see "07:30 · 41 active edges · peak 360 vph" instead of having
      to mouse over individual segments.
    * **Optional crash overlay** — pass ``crash_points`` (the rows
      from ``njdot_crashes.parquet`` with ``geocoded_lat/lon`` filled
      in) to add a toggleable safety layer with circle markers
      sized + coloured by NJDOT severity. Hidden by default so it
      doesn't clutter the traffic story until the viewer asks for it.
    """
    import folium

    if edges_geo is None:
        edges_geo = load_sumo_edge_geometries(net_path)
    if edges_geo.empty:
        logger.warning("No SUMO edge geometries; cannot build animated map")
        out_html.write_text(
            "<html><body>No edge geometries available.</body></html>"
        )
        return out_html

    if center is None:
        try:
            cb = edges_geo.geometry.unary_union.centroid
            center = (float(cb.y), float(cb.x))
        except Exception:
            center = (40.864, -73.980)

    frame_seconds = int(frame_minutes * 60)
    frames = _edge_history_to_frames(
        edge_history,
        frame_seconds=frame_seconds,
        sample_interval_s=sample_interval_s,
    )
    if frames.empty:
        logger.warning("Empty frame data; rendering placeholder map")
        m = folium.Map(location=list(center), zoom_start=zoom)
        m.save(str(out_html))
        return out_html

    if vmax_vph is None:
        positive = frames[frames["vph"] >= min_vph_for_animation]["vph"]
        if positive.empty:
            vmax_vph = max(50.0, float(frames["vph"].max() or 50.0))
        else:
            vmax_vph = max(50.0, float(positive.quantile(0.95)))

    edges_idx = edges_geo.set_index("edge_id")
    per_frame = _build_per_frame_features(
        frames, edges_idx, vmax_vph=vmax_vph,
        min_vph_for_animation=min_vph_for_animation,
    )

    skeleton_lines = _skeleton_lines_from_edges_geo(edges_geo)
    crash_payload = _crash_points_to_payload(crash_points)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        _render_animated_html(
            title=title,
            center=center,
            zoom=zoom,
            vmax_vph=vmax_vph,
            skeleton_lines=skeleton_lines,
            per_frame=per_frame,
            crash_points=crash_payload,
        ),
        encoding="utf-8",
    )
    return out_html


def _build_per_frame_features(
    frames: pd.DataFrame,
    edges_idx: gpd.GeoDataFrame,
    *,
    vmax_vph: float,
    min_vph_for_animation: float = 1.0,
    color_mode: str = "relative",
) -> list[dict]:
    """Build the per-frame feature payload Leaflet animates over.

    Returns one dict per frame, each carrying ``frame_s`` (seconds
    since midnight), a human ``label`` (``"07:00"``), aggregate
    counters (``n_active``, ``peak_vph``), and a list of
    LineString ``features`` styled by current vph.

    ``color_mode``:

    * ``"relative"`` — green→yellow→red ramp normalised against
      ``vmax_vph`` (the run's p95). Matches the long-standing
      single-pane animation.
    * ``"absolute"`` — fixed Leonia-calibrated thresholds (see
      :data:`_ABSOLUTE_VPH_STOPS`). Used by the dual map so a
      stakeholder's interpretation of "green ≈ idle, red ≈ busy"
      is consistent across runs and across both panes.

    Factored out of :func:`build_animated_map` so the dual-map
    variant can build two of these (left = baseline, right =
    scenario) with the same colour normalisation.
    """
    active = frames[frames["vph"] >= min_vph_for_animation].copy()
    frame_keys = sorted(int(s) for s in active["frame_s"].unique())
    if not frame_keys:
        logger.warning(
            "No edges met min_vph_for_animation=%s; rendering skeleton only.",
            min_vph_for_animation,
        )

    use_absolute = color_mode == "absolute"

    per_frame: list[dict] = []
    for fs in frame_keys:
        sub = active[active["frame_s"] == fs]
        feats: list[dict] = []
        for _, row in sub.iterrows():
            eid = row["sumo_edge_id"]
            if eid not in edges_idx.index:
                continue
            geom = edges_idx.loc[eid, "geometry"]
            if isinstance(geom, pd.Series):
                geom = geom.iloc[0]
            if geom is None or geom.is_empty:
                continue
            coords = [[float(y), float(x)] for x, y in geom.coords]
            vph = float(row["vph"])
            if use_absolute:
                color = _vph_to_color_absolute(vph)
                weight = _vph_to_width_absolute(vph)
            else:
                color = _vph_to_color(vph, vmax_vph)
                weight = 3.0 + 7.0 * min(1.0, vph / max(vmax_vph, 1.0))
            label = row.get("street_name")
            if not isinstance(label, str) or not label or label == "nan":
                label = str(eid)
            feats.append({
                "coords": coords,
                "color": color,
                "weight": weight,
                "label": label,
                "vph": round(vph),
                "speed_mph": round(float(row["mean_speed_mph"]), 1),
                "edge_id": eid,
            })
        per_frame.append({
            "frame_s": fs,
            "label": _seconds_to_clock_label(fs),
            "n_active": len(feats),
            "peak_vph": int(round(sub["vph"].max())) if not sub.empty else 0,
            "features": feats,
        })
    return per_frame


def _align_dual_frames(
    left: list[dict], right: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Make ``left`` and ``right`` agree on a single frame timeline.

    The two SUMO runs may have generated slightly different sets of
    ``frame_s`` keys (e.g. a closure scenario can drop a frame
    where no remaining edge meets ``min_vph_for_animation``). For a
    single shared slider we need both lists to have an entry at
    every frame the union covers; missing frames get an empty
    ``features`` list so the slider still ticks but that side
    shows skeleton-only.
    """
    keys = sorted({f["frame_s"] for f in left} | {f["frame_s"] for f in right})
    by_left = {f["frame_s"]: f for f in left}
    by_right = {f["frame_s"]: f for f in right}

    def _empty(fs: int) -> dict:
        return {
            "frame_s": fs,
            "label": _seconds_to_clock_label(fs),
            "n_active": 0,
            "peak_vph": 0,
            "features": [],
        }

    aligned_left = [by_left.get(k) or _empty(k) for k in keys]
    aligned_right = [by_right.get(k) or _empty(k) for k in keys]
    return aligned_left, aligned_right


def build_animated_dual_map(
    baseline_history: pd.DataFrame,
    scenario_history: pd.DataFrame,
    out_html: Path,
    *,
    edges_geo: gpd.GeoDataFrame | None = None,
    net_path: Path = DEFAULT_NET_PATH,
    frame_minutes: int = 15,
    sample_interval_s: int = 60,
    vmax_vph: float | None = None,
    min_vph_for_animation: float = 1.0,
    title_left: str = "Baseline",
    title_right: str = "Scenario",
    center: tuple[float, float] | None = None,
    zoom: int = 14,
) -> Path:
    """Render a side-by-side animated dual map.

    The output is a single HTML file containing two synchronised
    Leaflet maps:

    * **Left** — the baseline simulation (no scenario applied).
    * **Right** — the scenario simulation (closure / speed hump /
      one-way).

    Both maps share one time slider; advancing the slider toggles
    the matching feature group on each side. Pan and zoom are
    synchronised so the two maps stay aligned as the viewer
    explores the borough.

    Colour normalisation uses a *shared* ``vmax_vph`` so the same
    colour means the same vph on both sides — otherwise a
    closure-induced drop on the scenario side would erroneously
    look "as red as the baseline" because each side would
    auto-scale to its own peak.
    """
    if edges_geo is None:
        edges_geo = load_sumo_edge_geometries(net_path)
    if edges_geo.empty:
        logger.warning("No SUMO edge geometries; cannot build dual animation")
        out_html.write_text(
            "<html><body>No edge geometries available.</body></html>"
        )
        return out_html

    if center is None:
        try:
            cb = edges_geo.geometry.unary_union.centroid
            center = (float(cb.y), float(cb.x))
        except Exception:
            center = (40.864, -73.980)

    frame_seconds = int(frame_minutes * 60)
    base_frames = _edge_history_to_frames(
        baseline_history,
        frame_seconds=frame_seconds,
        sample_interval_s=sample_interval_s,
    )
    scen_frames = _edge_history_to_frames(
        scenario_history,
        frame_seconds=frame_seconds,
        sample_interval_s=sample_interval_s,
    )

    if base_frames.empty and scen_frames.empty:
        logger.warning("Empty frame data on both sides; placeholder map")
        out_html.write_text(
            "<html><body>No simulation history available.</body></html>"
        )
        return out_html

    if vmax_vph is None:
        # Compute a *shared* vmax across both sides so the colour
        # ramp is comparable left-vs-right. p95 of positive vph
        # filters out the long tail without losing perceptual
        # range on residential streets.
        positive_left = base_frames[
            base_frames["vph"] >= min_vph_for_animation
        ]["vph"]
        positive_right = scen_frames[
            scen_frames["vph"] >= min_vph_for_animation
        ]["vph"]
        positive = pd.concat([positive_left, positive_right], ignore_index=True)
        if positive.empty:
            vmax_vph = 50.0
        else:
            vmax_vph = max(50.0, float(positive.quantile(0.95)))

    edges_idx = edges_geo.set_index("edge_id")
    left_frames = _build_per_frame_features(
        base_frames, edges_idx,
        vmax_vph=vmax_vph,
        min_vph_for_animation=min_vph_for_animation,
        color_mode="absolute",
    )
    right_frames = _build_per_frame_features(
        scen_frames, edges_idx,
        vmax_vph=vmax_vph,
        min_vph_for_animation=min_vph_for_animation,
        color_mode="absolute",
    )
    left_frames, right_frames = _align_dual_frames(left_frames, right_frames)

    skeleton_lines = _skeleton_lines_from_edges_geo(edges_geo)
    skeleton_payload = _skeleton_with_edge_ids(edges_geo)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        _render_animated_dual_html(
            title_left=title_left,
            title_right=title_right,
            center=center,
            zoom=zoom,
            vmax_vph=vmax_vph,
            skeleton_lines=skeleton_lines,
            skeleton_with_ids=skeleton_payload,
            left_frames=left_frames,
            right_frames=right_frames,
        ),
        encoding="utf-8",
    )
    return out_html


_STATE_SYSTEM_LABELS = {
    "state authority", "njdot state highway", "interstate",
    "state park / inst. / authority", "us govt property",
}


def _is_state_system(road_system: object) -> bool:
    """Return True if a crash row is on a state-jurisdiction roadway."""
    if not isinstance(road_system, str):
        return False
    return road_system.strip().lower() in _STATE_SYSTEM_LABELS


def _filter_crash_rows_to_borough(
    crash_points: pd.DataFrame,
    *,
    drop_state_system: bool = True,
    crash_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply Leonia-specific filters to a row-level crash DataFrame.

    Returns a fresh DataFrame (with index reset) containing only the
    rows that should appear on the borough's crash visualisations.
    Four filters are applied in sequence:

    (1) **Coordinate present** — drop rows where ``geocoded_lat`` /
        ``geocoded_lon`` (or the legacy ``latitude`` / ``longitude``
        columns) are null.
    (2) **Borough bounding box** — keep only rows inside the Leonia
        bbox plus a ~1.3 km buffer. This catches NJDOT rows that
        were geocoded to a far-away stretch of I-95 / NJ Turnpike;
        the dataset's address parser sometimes resolves
        ``"I-95; N.J. TURNPIKE"`` to a Trenton / Newark turnpike
        point instead of a Leonia segment. Without this guard, ~28
        out-of-borough crashes leaked onto the map.
    (3) **State-system filter** (when ``drop_state_system=True``) —
        drop rows that resolve to a state-system OSM way name OR
        whose row-level ``road_system`` column says state-system.
        Both signals are checked because either alone misses cases:
        OSM-only misses rows with ``geocoded_osm_way_id`` null
        (NJDOT didn't snap them), and ``road_system``-only misses
        local streets that NJDOT misclassifies.
    (4) **Polygon-precise filter for state-system rows** — when
        the row-level ``road_system`` flags an Interstate / NJ
        Turnpike / State Authority crash, also require that the
        lat/lon falls inside the actual Leonia borough polygon
        (with a ~150 m buffer for centerline jitter). This catches
        the residual handful of "I-95; N.J. TURNPIKE" crashes that
        NJDOT snapped to a friendly local-street OSM way (e.g.
        Schlosser St, Christopher Columbus Hwy, Broad Ave) but
        whose actual coordinates are 200 m – 1.8 km outside the
        borough (Fort Lee centre / Ridgefield Park / Englewood
        Cliffs). Local-jurisdiction rows skip this layer because
        their OSM-snapped coordinates can sit just outside the
        polygon when the underlying centerline crosses into a
        neighbouring municipality.

    The output's row alignment matches what
    :func:`_crash_points_to_payload` produces from the same input,
    so callers can zip the two together by positional index.
    """
    if crash_points is None or crash_points.empty:
        return crash_points.iloc[0:0].reset_index(drop=True) \
            if crash_points is not None \
            else pd.DataFrame()

    df = crash_points.copy()
    if "geocoded_lat" in df.columns and "geocoded_lon" in df.columns:
        lat_col, lon_col = "geocoded_lat", "geocoded_lon"
    elif "latitude" in df.columns and "longitude" in df.columns:
        lat_col, lon_col = "latitude", "longitude"
    else:
        return df.iloc[0:0].reset_index(drop=True)

    df = df[df[lat_col].notna() & df[lon_col].notna()]
    if df.empty:
        return df.reset_index(drop=True)

    df = df[
        (df[lat_col] >= LEONIA_BBOX_MIN_LAT)
        & (df[lat_col] <= LEONIA_BBOX_MAX_LAT)
        & (df[lon_col] >= LEONIA_BBOX_MIN_LON)
        & (df[lon_col] <= LEONIA_BBOX_MAX_LON)
    ]
    if df.empty:
        return df.reset_index(drop=True)

    if drop_state_system:
        # OSM is the **primary** signal — when an OSM way name is
        # available, trust it. This matters for NJ-93 / Grand
        # Avenue: NJDOT classifies it as a state highway but OSM
        # resolves it to "Grand Avenue" (a borough surface street),
        # and the safety panel keeps it because Leonia has policy
        # levers there. We follow the same convention here.
        #
        # ``road_system`` is the **fallback** signal — used only
        # when OSM is unavailable (``geocoded_osm_way_id`` is null
        # or the segment table doesn't have a matching way). This
        # catches the 28 NJ-Turnpike crashes that NJDOT
        # geocoded to a far-away turnpike point and never snapped
        # to OSM. Without the fallback, those leak through.
        has_osm_lookup = (
            crash_segments is not None and not crash_segments.empty
            and "geocoded_osm_way_id" in df.columns
            and "osm_way_id" in crash_segments.columns
        )
        if has_osm_lookup:
            way_to_name = (
                crash_segments[["osm_way_id", "street_name"]]
                .drop_duplicates("osm_way_id")
                .set_index("osm_way_id")["street_name"]
                .to_dict()
            )
            way_ids = pd.to_numeric(
                df["geocoded_osm_way_id"], errors="coerce",
            )
            osm_name = way_ids.map(way_to_name)
            has_osm_name = osm_name.notna()
            is_state_osm = osm_name.apply(_is_state_system_street)
        else:
            has_osm_name = pd.Series(False, index=df.index)
            is_state_osm = pd.Series(False, index=df.index)

        if "road_system" in df.columns:
            is_state_njdot = df["road_system"].apply(_is_state_system)
        else:
            is_state_njdot = pd.Series(False, index=df.index)

        # Drop rule:
        #   - if OSM name is known: drop iff OSM says state-system
        #     (ignore road_system — OSM is authoritative)
        #   - if OSM name is unknown: drop iff road_system says
        #     state-system (no OSM signal to trust)
        # On-road text drop: NJDOT crashes whose *on-road* is itself a
        # limited-access state facility (I-95 / NJ Turnpike / Interstate /
        # express lanes). These are routinely mis-geocoded onto a parallel
        # borough street, so the OSM-name signal alone keeps them (e.g. a
        # "I-95; N.J. TURNPIKE" crash snapped to Broad/Grand Avenue). We key
        # off crash_location only, so a local-street crash that merely
        # *crosses* I-95 at the interchange is retained.
        if "crash_location" in df.columns:
            is_nonlocal_onroad = df["crash_location"].apply(
                _is_nonlocal_crash_onroad
            )
        else:
            is_nonlocal_onroad = pd.Series(False, index=df.index)

        is_state = (
            (has_osm_name & is_state_osm)
            | (~has_osm_name & is_state_njdot)
            | is_nonlocal_onroad
        )
        df = df[~is_state]
        if df.empty:
            return df.reset_index(drop=True)

        # Layer (4): polygon-precise filter for residual
        # state-system rows. By the time we get here the rows
        # whose OSM way name is itself state-system have already
        # been dropped above; what remains is the snap-mismatch
        # case — ``road_system = "State Authority"`` with a
        # friendly-named OSM way. Verify those by lat/lon.
        if "road_system" in df.columns and is_state_njdot.any():
            poly_buf = _leonia_polygon_buffered()
            if poly_buf is not None:
                from shapely.geometry import Point as _Point

                njdot_state = df["road_system"].apply(_is_state_system)
                if njdot_state.any():
                    inside = df.loc[njdot_state].apply(
                        lambda r: poly_buf.contains(
                            _Point(r[lon_col], r[lat_col])
                        ),
                        axis=1,
                    )
                    drop_idx = njdot_state[njdot_state].index[~inside.values]
                    if len(drop_idx) > 0:
                        df = df.drop(index=drop_idx)

    return df.reset_index(drop=True)


def _crash_points_to_payload(
    crash_points: pd.DataFrame | None,
    *,
    drop_state_system: bool = True,
    crash_segments: pd.DataFrame | None = None,
) -> list[dict]:
    """Reduce a crashes DataFrame to the minimum the renderer needs.

    Accepts either the dashboard schema (KABCO codes K/A/B/C/O,
    ``Geopoint Calculated`` fallback already merged into
    ``geocoded_lat``/``geocoded_lon``) or the legacy zip schema
    (F/I/P codes). Rows without coordinates are dropped silently.

    ``drop_state_system=True`` (the default) hides crashes whose
    OSM way is a state-system road (Interstate / NJ Turnpike /
    motorway link). It uses **the same OSM-way-name filter as the
    safety panel** so the borough totals match: 980 reported
    crashes in the table → 980 markers on the map.

    Earlier versions of this function used the NJDOT
    ``road_system`` column to filter, but that disagreed with the
    safety panel for crashes on NJ-93 (= Grand Avenue): NJDOT
    classifies NJ-93 as a "State Highway" so the crashes vanished
    from the map, but the safety panel kept them because Leonia
    has policy levers on Grand Avenue. Switching to OSM-way-based
    filtering reconciles the two views.

    ``crash_segments`` (optional) is the per-segment parquet from
    :data:`leonia_traffic.data.dataset_io.CrashFiles.crashes_by_segment`.
    When supplied, it provides the canonical OSM ``street_name``
    for each ``geocoded_osm_way_id`` so the filter can match the
    safety panel's logic exactly. When omitted, we fall back to
    the row-level ``road_system`` filter (legacy behaviour).
    """
    if crash_points is None or crash_points.empty:
        return []

    df = crash_points.copy()
    if "geocoded_lat" in df.columns and "geocoded_lon" in df.columns:
        df["lat"] = df["geocoded_lat"]
        df["lon"] = df["geocoded_lon"]
    elif "latitude" in df.columns and "longitude" in df.columns:
        df["lat"] = df["latitude"]
        df["lon"] = df["longitude"]
    else:
        return []
    df = _filter_crash_rows_to_borough(
        df,
        drop_state_system=drop_state_system,
        crash_segments=crash_segments,
    )
    if df.empty:
        return []

    # Visual encoding: fatal = big red, suspected serious = bigger
    # orange, possible / minor injury = orange, PDO = small grey.
    # Both KABCO (K/A/B/C/O) and legacy F/I/P scales supported.
    severity_style = {
        "K": ("#b71c1c", 10.0),  # fatal
        "F": ("#b71c1c", 10.0),
        "A": ("#d32f2f",  8.0),  # suspected serious injury (KSI)
        "B": ("#ef6c00",  6.0),  # suspected minor injury
        "C": ("#f57c00",  5.0),  # possible injury
        "I": ("#ef6c00",  6.0),  # legacy injury
        "O": ("#616161",  3.5),  # no apparent injury (PDO)
        "P": ("#616161",  3.5),
    }

    payload: list[dict] = []
    for _, r in df.iterrows():
        sev = (r.get("severity_code") or "O").strip().upper() \
            if isinstance(r.get("severity_code"), str) else "O"
        color, radius = severity_style.get(sev, ("#616161", 3.5))
        date = r.get("crash_date")
        try:
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d") \
                if pd.notna(date) else ""
        except Exception:
            date_str = ""
        loc = r.get("crash_location") or ""
        cross = r.get("cross_street") or ""
        label = (loc + (" × " + cross if cross else "")).strip() or "(unknown)"
        payload.append({
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "color": color,
            "radius": radius,
            "label": label,
            "date": date_str,
            "severity": r.get("severity_label", sev),
            "ped": bool(r.get("ped_involved", False)),
        })
    return payload


# ---------------------------------------------------------------------------
# Self-contained HTML template for the animated map
# ---------------------------------------------------------------------------


_ANIMATED_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body { margin: 0; padding: 0; height: 100%;
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                              Roboto, Helvetica, Arial, sans-serif; }
    #map { position: absolute; top: 0; bottom: 76px; left: 0; right: 0; }
    #ctrl { position: absolute; bottom: 0; left: 0; right: 0; height: 76px;
            background: #fff; border-top: 1px solid #e1e1e4;
            display: flex; align-items: center; padding: 0 16px; gap: 12px;
            z-index: 1000; }
    #ctrl button { width: 36px; height: 36px; border-radius: 4px;
                   border: 1px solid #c0c0c4; background: #fff;
                   cursor: pointer; font-size: 16px; }
    #ctrl button:hover { background: #f3f3f5; }
    #ctrl input[type=range] { flex: 1; }
    #frame-label { font-weight: 600; color: #222; min-width: 90px; }
    #frame-stats { color: #666; font-size: 12px; min-width: 200px; }
    #title { position: absolute; top: 12px; left: 60px; z-index: 999;
             background: rgba(255,255,255,0.92); padding: 6px 12px;
             border-radius: 4px; font-size: 14px; font-weight: 600;
             color: #222; }
    #legend { position: absolute; top: 12px; right: 12px; z-index: 999;
              background: rgba(255,255,255,0.92); padding: 8px 12px;
              border-radius: 4px; font-size: 11px; line-height: 1.5;
              color: #222; min-width: 180px; }
    #legend .gradient-bar {
      width: 100%; height: 10px; border-radius: 2px; margin: 4px 0 2px;
      background: linear-gradient(to right,
        __C_LO__ 0%, __C_Q__ 25%, __C_MID__ 50%, __C_HI__ 100%);
    }
    #legend .gradient-ticks {
      display: flex; justify-content: space-between;
      font-size: 10px; color: #555;
    }
    #legend .swatch { display: inline-block; width: 14px; height: 3px;
                      vertical-align: middle; margin-right: 6px;
                      background: #cccccc; }
  </style>
</head>
<body>
  <div id="title">__TITLE__</div>
  <div id="legend">
    <b>Traffic intensity (vehicles per hour)</b>
    <div class="gradient-bar"></div>
    <div class="gradient-ticks">
      <span>__V_LO__</span>
      <span>__V_Q__</span>
      <span>__V_MID__</span>
      <span>__V_HI__+</span>
    </div>
    <div style="color:#666;margin-top:4px;font-size:10px">
      Color = current rate (smooth scale). Hover a road for exact vph.
    </div>
    <div style="margin-top:4px"><span class="swatch"></span><span style="color:#999">network (no traffic)</span></div>
    <div id="crash-legend" style="margin-top:6px;border-top:1px solid #e1e1e4;
                                  padding-top:6px;display:none">
      <b>NJDOT crashes (local streets)</b><br>
      <span style="display:inline-block;width:10px;height:10px;background:#b71c1c;
                   border-radius:50%;vertical-align:middle;margin-right:6px"></span>fatal (K)<br>
      <span style="display:inline-block;width:8px;height:8px;background:#d32f2f;
                   border-radius:50%;vertical-align:middle;margin-right:6px"></span>suspected serious (A)<br>
      <span style="display:inline-block;width:6px;height:6px;background:#ef6c00;
                   border-radius:50%;vertical-align:middle;margin-right:6px"></span>injury (B/C)<br>
      <span style="display:inline-block;width:4px;height:4px;background:#616161;
                   border-radius:50%;vertical-align:middle;margin-right:6px"></span>no apparent injury
    </div>
  </div>
  <div id="map"></div>
  <div id="ctrl">
    <button id="play" title="Play / pause">▶</button>
    <button id="prev" title="Previous frame">◀</button>
    <button id="next" title="Next frame">▶|</button>
    <span id="frame-label">--:--</span>
    <input id="slider" type="range" min="0" max="0" value="0" step="1">
    <span id="frame-stats">—</span>
  </div>
<script>
const SKELETON = __SKELETON_JSON__;
const FRAMES = __FRAMES_JSON__;
const CRASHES = __CRASHES_JSON__;
const CENTER = __CENTER_JSON__;
const ZOOM = __ZOOM__;

const map = L.map('map', { zoomControl: true }).setView(CENTER, ZOOM);
L.tileLayer(
  'https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
  { attribution: '© OpenStreetMap contributors © CARTO', maxZoom: 19 }
).addTo(map);

const skeletonLayer = L.layerGroup();
SKELETON.forEach(coords => {
  L.polyline(coords, { color: '#bdbdbd', weight: 1.2, opacity: 0.55,
                       interactive: false }).addTo(skeletonLayer);
});
skeletonLayer.addTo(map);

// Crashes layer (off by default; user toggles it via the layer control).
const crashLayer = L.layerGroup();
CRASHES.forEach(c => {
  const m = L.circleMarker([c.lat, c.lon], {
    radius: c.radius, color: c.color, fillColor: c.color,
    fillOpacity: 0.85, weight: 1, opacity: 0.95,
  });
  const pedTag = c.ped ? ' <span style="color:#b71c1c">(pedestrian)</span>' : '';
  m.bindTooltip(
    `<b>${c.severity}</b>${pedTag}<br>${c.label}<br>${c.date}`,
    { sticky: true }
  );
  m.addTo(crashLayer);
});
if (CRASHES.length > 0) {
  L.control.layers(null, { 'NJDOT crashes 2017–22': crashLayer },
                   { collapsed: false, position: 'topright' }).addTo(map);
  // Reveal the dedicated legend block when the layer is on.
  const cl = document.getElementById('crash-legend');
  map.on('overlayadd', e => {
    if (e.layer === crashLayer && cl) cl.style.display = 'block';
  });
  map.on('overlayremove', e => {
    if (e.layer === crashLayer && cl) cl.style.display = 'none';
  });
}

const frameLayers = FRAMES.map(frame => {
  const layer = L.layerGroup();
  frame.features.forEach(f => {
    const line = L.polyline(f.coords, {
      color: f.color, weight: f.weight, opacity: 0.95,
    });
    line.bindTooltip(
      `<b>${f.label}</b><br>${frame.label} — ${f.vph} vph @ ${f.speed_mph} mph`,
      { sticky: true }
    );
    line.addTo(layer);
  });
  return layer;
});

const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const prevBtn = document.getElementById('prev');
const nextBtn = document.getElementById('next');
const frameLabel = document.getElementById('frame-label');
const frameStats = document.getElementById('frame-stats');

slider.max = Math.max(0, FRAMES.length - 1);

let currentIdx = -1;
let playing = false;
let timer = null;

function showFrame(idx) {
  if (idx < 0 || idx >= FRAMES.length) return;
  if (currentIdx >= 0) {
    map.removeLayer(frameLayers[currentIdx]);
  }
  currentIdx = idx;
  frameLayers[idx].addTo(map);
  slider.value = String(idx);
  const f = FRAMES[idx];
  frameLabel.textContent = f.label;
  frameStats.textContent =
    `${f.n_active} active edge${f.n_active === 1 ? '' : 's'} · peak ${f.peak_vph} vph`;
}

function play() {
  playing = true;
  playBtn.textContent = '❚❚';
  if (timer) clearInterval(timer);
  timer = setInterval(() => {
    let next = currentIdx + 1;
    if (next >= FRAMES.length) next = 0;
    showFrame(next);
  }, 700);
}

function pause() {
  playing = false;
  playBtn.textContent = '▶';
  if (timer) { clearInterval(timer); timer = null; }
}

playBtn.addEventListener('click', () => playing ? pause() : play());
prevBtn.addEventListener('click', () => {
  pause();
  showFrame((currentIdx - 1 + FRAMES.length) % FRAMES.length);
});
nextBtn.addEventListener('click', () => {
  pause();
  showFrame((currentIdx + 1) % FRAMES.length);
});
slider.addEventListener('input', e => {
  pause();
  showFrame(Number(e.target.value));
});

if (FRAMES.length > 0) {
  showFrame(0);
  play();
} else {
  frameLabel.textContent = '—';
  frameStats.textContent = 'no traffic in window';
}
</script>
</body>
</html>
"""


_ANIMATED_DUAL_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE_LEFT__ vs __TITLE_RIGHT__</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body { margin: 0; padding: 0; height: 100%;
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                              Roboto, Helvetica, Arial, sans-serif; }
    #stage { position: absolute; top: 0; bottom: 76px; left: 0; right: 0;
             display: flex; }
    .pane { flex: 1 1 50%; position: relative;
            border-right: 1px solid #d8d8da; min-width: 0; }
    .pane:last-child { border-right: 0; }
    .pane .map { position: absolute; top: 0; bottom: 0; left: 0; right: 0; }
    .pane .pane-title {
      position: absolute; top: 12px; left: 60px; z-index: 999;
      background: rgba(255,255,255,0.94); padding: 6px 12px;
      border-radius: 4px; font-size: 13px; font-weight: 600;
      color: #222; max-width: 60%;
    }
    .pane .pane-stats {
      position: absolute; bottom: 12px; left: 12px; z-index: 999;
      background: rgba(255,255,255,0.94); padding: 4px 10px;
      border-radius: 4px; font-size: 11px; color: #444;
    }
    #ctrl { position: absolute; bottom: 0; left: 0; right: 0; height: 76px;
            background: #fff; border-top: 1px solid #e1e1e4;
            display: flex; align-items: center; padding: 0 16px; gap: 12px;
            z-index: 1000; }
    #ctrl button { width: 36px; height: 36px; border-radius: 4px;
                   border: 1px solid #c0c0c4; background: #fff;
                   cursor: pointer; font-size: 16px; }
    #ctrl button:hover { background: #f3f3f5; }
    #ctrl input[type=range] { flex: 1; }
    #frame-label { font-weight: 600; color: #222; min-width: 90px; }
    /* The "selected street" outline pulses subtly so the eye picks
       it up over the live traffic colours without obscuring them. */
    .leaflet-interactive.street-outline {
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div id="stage">
    <div class="pane" id="pane-left">
      <div class="pane-title">__TITLE_LEFT__</div>
      <div class="map" id="map-left"></div>
      <div class="pane-stats" id="stats-left">—</div>
    </div>
    <div class="pane" id="pane-right">
      <div class="pane-title">__TITLE_RIGHT__</div>
      <div class="map" id="map-right"></div>
      <div class="pane-stats" id="stats-right">—</div>
    </div>
  </div>
  <div id="ctrl">
    <button id="play" title="Play / pause">▶</button>
    <button id="prev" title="Previous frame">◀</button>
    <button id="next" title="Next frame">▶|</button>
    <span id="frame-label">--:--</span>
    <input id="slider" type="range" min="0" max="0" value="0" step="1">
  </div>
<script>
const SKELETON = __SKELETON_JSON__;
const SKELETON_WITH_IDS = __SKELETON_WITH_IDS_JSON__;
const LEFT_FRAMES = __LEFT_FRAMES_JSON__;
const RIGHT_FRAMES = __RIGHT_FRAMES_JSON__;
const CENTER = __CENTER_JSON__;
const ZOOM = __ZOOM__;

const N_FRAMES = Math.max(LEFT_FRAMES.length, RIGHT_FRAMES.length);

// ---- maps -----------------------------------------------------------
const tileUrl = 'https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png';
const tileAttr = '© OpenStreetMap contributors © CARTO';

function makeMap(elId) {
  const m = L.map(elId, { zoomControl: true }).setView(CENTER, ZOOM);
  L.tileLayer(tileUrl, { attribution: tileAttr, maxZoom: 19 }).addTo(m);
  const skeletonLayer = L.layerGroup();
  SKELETON.forEach(coords => {
    L.polyline(coords, { color: '#bdbdbd', weight: 1.2, opacity: 0.55,
                         interactive: false }).addTo(skeletonLayer);
  });
  skeletonLayer.addTo(m);
  return m;
}
const mapLeft = makeMap('map-left');
const mapRight = makeMap('map-right');

// Build a single index of edge_id → coords so the highlight overlay
// can resolve a list of edge IDs to polylines without scanning.
const COORDS_BY_EDGE = new Map();
SKELETON_WITH_IDS.forEach(s => COORDS_BY_EDGE.set(s.edge_id, s.coords));

// ---- pan/zoom synchronisation ---------------------------------------
let _syncing = false;
function syncFromTo(src, dst) {
  src.on('move', () => {
    if (_syncing) return;
    _syncing = true;
    dst.setView(src.getCenter(), src.getZoom(), { animate: false });
    _syncing = false;
  });
}
syncFromTo(mapLeft, mapRight);
syncFromTo(mapRight, mapLeft);

// ---- "Selected street" highlight overlay ----------------------------
// The overlay is a static layer that sits on top of the live traffic
// frames; it doesn't toggle with the slider. Two passes per pane are
// used: first a thick semi-transparent white halo to lift the
// segment off the basemap, then a thinner saturated-blue outline so
// the eye locks onto the chosen street.
const highlightLeft = L.layerGroup().addTo(mapLeft);
const highlightRight = L.layerGroup().addTo(mapRight);

function setHighlightedEdges(edgeIds) {
  highlightLeft.clearLayers();
  highlightRight.clearLayers();
  if (!Array.isArray(edgeIds) || edgeIds.length === 0) return;
  edgeIds.forEach(eid => {
    const coords = COORDS_BY_EDGE.get(String(eid));
    if (!coords) return;
    [highlightLeft, highlightRight].forEach(layer => {
      // Soft white halo underneath, then a desaturated slate
      // outline. The earlier saturated-blue + dashed treatment
      // shouted over the underlying traffic colours; this version
      // reads as "this is the street you picked" without
      // dominating the live data.
      L.polyline(coords, {
        color: '#ffffff', weight: 9, opacity: 0.45,
        className: 'street-outline', interactive: false,
      }).addTo(layer);
      L.polyline(coords, {
        color: '#5b6dab', weight: 2.5, opacity: 0.75,
        className: 'street-outline', interactive: false,
      }).addTo(layer);
    });
  });
}

// ---- "Measured StreetLight volume" overlay --------------------------
// A second skeleton-style layer that shows the *real-world* hourly
// traffic volume on every street with measured data, regardless of
// whether the SUMO simulation routes vehicles through it. Most of
// Leonia's local streets carry zero modelled traffic (the demand
// only routes between OD pairs), so without this overlay
// stakeholders see a misleading sea of grey.
//
// The overlay is loaded asynchronously: we fetch a tiny JSON
// (~200 KB gzipped) from the same precache server, build 24
// per-hour LayerGroups, and swap the active hour as the slider
// moves. Each polyline is rendered as a dark-grey ribbon whose
// width scales with the measured vph — colour stays free for the
// simulation. The overlay renders BELOW the simulation frames so
// that streets the model does route through still pop in red.
const stlLeft = L.layerGroup().addTo(mapLeft);
const stlRight = L.layerGroup().addTo(mapRight);
let stlByHour = null;     // [hour] -> {left: LayerGroup, right: LayerGroup}
let stlVisible = true;    // toggled by parent
let stlCurrentHour = -1;

function _stlWidthFromVph(vph) {
  // Logarithmic-ish so a 5 vph cul-de-sac is just visible (1.0 px)
  // and a 600 vph arterial maxes out around 5 px. Stays narrower
  // than the simulation strokes (max ~9 px) so the overlay reads
  // as background context rather than competing with the live data.
  if (!vph || vph <= 0) return 0;
  if (vph <= 5)   return 1.0;
  if (vph <= 20)  return 1.5;
  if (vph <= 50)  return 2.2;
  if (vph <= 100) return 3.0;
  if (vph <= 200) return 3.7;
  if (vph <= 400) return 4.5;
  return 5.0;
}

function _buildStlLayers(byEdge) {
  // 24 LayerGroup pairs, one per hour. Each contains a polyline
  // for every edge that has a positive vph in that hour.
  const layers = Array.from({length: 24}, () => ({
    left: L.layerGroup(),
    right: L.layerGroup(),
  }));
  Object.keys(byEdge).forEach(eid => {
    const entry = byEdge[eid];
    const coords = COORDS_BY_EDGE.get(eid);
    if (!coords) return;
    entry.hourly_vph.forEach((vph, h) => {
      const w = _stlWidthFromVph(vph);
      if (w <= 0) return;
      const tooltip = `<b>${entry.street}</b><br>` +
        `Measured ${Math.round(vph)} vph at ${h}:00<br>` +
        `<span style="color:#888">~${entry.daily_total} vehicles/day · ${entry.source}</span>`;
      ['left', 'right'].forEach(side => {
        const line = L.polyline(coords, {
          color: '#3a3a3a', weight: w, opacity: 0.45,
          interactive: true,
        });
        line.bindTooltip(tooltip, { sticky: true });
        layers[h][side].addLayer(line);
      });
    });
  });
  return layers;
}

function _setStlHour(hour) {
  if (!stlByHour || !stlVisible) return;
  if (hour === stlCurrentHour) return;
  // Remove previous hour from the visible group.
  if (stlCurrentHour >= 0) {
    stlLeft.clearLayers();
    stlRight.clearLayers();
  }
  stlCurrentHour = hour;
  if (hour < 0 || hour >= 24) return;
  stlByHour[hour].left.eachLayer(l => stlLeft.addLayer(l));
  stlByHour[hour].right.eachLayer(l => stlRight.addLayer(l));
  // The overlay is conceptually "below" simulation frames; Leaflet's
  // SVG renderer puts later-added paths on top, so without
  // intervention the overlay would obscure the live colours. We
  // bring frame layers AND the highlight to the front instead, in
  // showFrame(), to enforce the right z-order.
}

function _showStl() {
  stlVisible = true;
  if (stlCurrentHour >= 0 && stlByHour) {
    const h = stlCurrentHour;
    stlCurrentHour = -1;          // force _setStlHour to re-add
    _setStlHour(h);
  }
}

function _hideStl() {
  stlVisible = false;
  stlLeft.clearLayers();
  stlRight.clearLayers();
}

// Fetch the overlay payload. The DEMAND_PARAM placeholder is
// substituted from the URL (?demand=weekday|sunday) by the parent
// page when it sets iframe.src. If the fetch fails (file missing,
// no network), we silently degrade: the simulation still works,
// stakeholders just don't get the measured-volume context.
const _params = new URLSearchParams(window.location.search);
const _demand = _params.get('demand') || 'weekday';
fetch(`../_overlays/streetlight_${_demand}.json`)
  .then(r => r.ok ? r.json() : Promise.reject(r.status))
  .then(data => {
    stlByHour = _buildStlLayers(data.by_edge || {});
    // Initial paint: pick the hour matching the current frame.
    if (currentIdx >= 0 && (LEFT_FRAMES[currentIdx] || RIGHT_FRAMES[currentIdx])) {
      const fs = (LEFT_FRAMES[currentIdx] || RIGHT_FRAMES[currentIdx]).frame_s;
      _setStlHour(Math.floor(fs / 3600));
    } else {
      _setStlHour(0);
    }
  })
  .catch(err => {
    // Soft-fail: log and move on. The dual map remains useful
    // even without the measured-volume context.
    console.warn('StreetLight overlay unavailable:', err);
  });

// ---- Parent-page postMessage hooks ----------------------------------
window.addEventListener('message', (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'highlightStreet') {
    setHighlightedEdges(msg.edgeIds);
  } else if (msg.type === 'toggleStreetLight') {
    if (msg.visible) {
      _showStl();
    } else {
      _hideStl();
    }
  }
});

// ---- frame layers ---------------------------------------------------
function buildFrameLayers(frames) {
  return frames.map(frame => {
    const layer = L.layerGroup();
    frame.features.forEach(f => {
      const line = L.polyline(f.coords, {
        color: f.color, weight: f.weight, opacity: 0.95,
      });
      line.bindTooltip(
        `<b>${f.label}</b><br>${frame.label} — ${f.vph} vph @ ${f.speed_mph} mph`,
        { sticky: true }
      );
      line.addTo(layer);
    });
    return layer;
  });
}
const layersLeft = buildFrameLayers(LEFT_FRAMES);
const layersRight = buildFrameLayers(RIGHT_FRAMES);

const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const prevBtn = document.getElementById('prev');
const nextBtn = document.getElementById('next');
const frameLabel = document.getElementById('frame-label');
const statsLeft = document.getElementById('stats-left');
const statsRight = document.getElementById('stats-right');

slider.max = Math.max(0, N_FRAMES - 1);

let currentIdx = -1;
let playing = false;
let timer = null;

function paneStats(frame) {
  if (!frame) return '—';
  return `${frame.n_active} active edge${frame.n_active === 1 ? '' : 's'} · peak ${frame.peak_vph} vph`;
}

function showFrame(idx) {
  if (idx < 0 || idx >= N_FRAMES) return;
  if (currentIdx >= 0) {
    if (layersLeft[currentIdx]) mapLeft.removeLayer(layersLeft[currentIdx]);
    if (layersRight[currentIdx]) mapRight.removeLayer(layersRight[currentIdx]);
  }
  currentIdx = idx;
  if (layersLeft[idx]) layersLeft[idx].addTo(mapLeft);
  if (layersRight[idx]) layersRight[idx].addTo(mapRight);
  // Sync the StreetLight overlay to the current hour. The hour is
  // derived from the frame_s (seconds since midnight) so the
  // measured volume changes naturally as the user scrubs through
  // the day.
  const fs = (LEFT_FRAMES[idx] || RIGHT_FRAMES[idx]).frame_s;
  _setStlHour(Math.floor(fs / 3600));
  // Re-raise simulation frame layers above the StreetLight overlay
  // so an actively-modelled street still dominates over the
  // measured ribbon. LayerGroups expose ``eachLayer`` but not
  // ``bringToFront``, so we iterate.
  if (layersLeft[idx])  layersLeft[idx].eachLayer(l => l.bringToFront && l.bringToFront());
  if (layersRight[idx]) layersRight[idx].eachLayer(l => l.bringToFront && l.bringToFront());
  // Highlight is rendered last so it sits on top of everything.
  highlightLeft.eachLayer(l => l.bringToFront && l.bringToFront());
  highlightRight.eachLayer(l => l.bringToFront && l.bringToFront());
  slider.value = String(idx);
  const lf = LEFT_FRAMES[idx];
  const rf = RIGHT_FRAMES[idx];
  // Both sides share a frame timeline (see _align_dual_frames in
  // visualizations.py), so either's label works for the clock.
  frameLabel.textContent = (lf || rf).label;
  statsLeft.textContent = paneStats(lf);
  statsRight.textContent = paneStats(rf);
}

function play() {
  playing = true;
  playBtn.textContent = '❚❚';
  if (timer) clearInterval(timer);
  timer = setInterval(() => {
    let next = currentIdx + 1;
    if (next >= N_FRAMES) next = 0;
    showFrame(next);
  }, 700);
}

function pause() {
  playing = false;
  playBtn.textContent = '▶';
  if (timer) { clearInterval(timer); timer = null; }
}

playBtn.addEventListener('click', () => playing ? pause() : play());
prevBtn.addEventListener('click', () => {
  pause();
  showFrame((currentIdx - 1 + N_FRAMES) % N_FRAMES);
});
nextBtn.addEventListener('click', () => {
  pause();
  showFrame((currentIdx + 1) % N_FRAMES);
});
slider.addEventListener('input', e => {
  pause();
  showFrame(Number(e.target.value));
});

if (N_FRAMES > 0) {
  showFrame(0);
  play();
} else {
  frameLabel.textContent = '—';
}

// Tell the parent we're ready to receive highlight messages.
// (Parent stores the desired edge IDs and replays the message on
// 'iframe-ready' so the highlight survives an iframe swap.)
try {
  window.parent.postMessage({ type: 'iframe-ready' }, '*');
} catch (_e) { /* sandboxed iframes may forbid this; ignore */ }
</script>
</body>
</html>
"""


def _render_animated_dual_html(
    *,
    title_left: str,
    title_right: str,
    center: tuple[float, float],
    zoom: int,
    vmax_vph: float,  # retained for signature parity; absolute scale ignores it
    skeleton_lines: list[list[list[float]]],
    skeleton_with_ids: list[dict],
    left_frames: list[dict],
    right_frames: list[dict],
) -> str:
    """Inject the dual-frame payloads into the dual-map template.

    The dual map uses the absolute Leonia colour scale (see
    :data:`_ABSOLUTE_VPH_STOPS`), so ``vmax_vph`` is unused here
    — the legend swatches are baked into the template's CSS.
    """
    del vmax_vph  # absolute scale ignores it
    return (
        _ANIMATED_DUAL_HTML_TEMPLATE
        .replace("__TITLE_LEFT__", html_escape(title_left))
        .replace("__TITLE_RIGHT__", html_escape(title_right))
        .replace("__SKELETON_JSON__", json.dumps(skeleton_lines))
        .replace("__SKELETON_WITH_IDS_JSON__", json.dumps(skeleton_with_ids))
        .replace("__LEFT_FRAMES_JSON__", json.dumps(left_frames))
        .replace("__RIGHT_FRAMES_JSON__", json.dumps(right_frames))
        .replace("__CENTER_JSON__", json.dumps([center[0], center[1]]))
        .replace("__ZOOM__", str(int(zoom)))
    )


def _render_animated_html(
    *,
    title: str,
    center: tuple[float, float],
    zoom: int,
    vmax_vph: float,
    skeleton_lines: list[list[list[float]]],
    per_frame: list[dict],
    crash_points: list[dict] | None = None,
) -> str:
    """Inject data into ``_ANIMATED_HTML_TEMPLATE`` (no Jinja).

    We use plain string substitution because the body contains a lot
    of CSS/JS braces that confuse Jinja, and the template is fully
    static apart from a handful of clearly-named placeholders.
    """
    crashes = crash_points or []
    return (
        _ANIMATED_HTML_TEMPLATE
        .replace("__TITLE__", html_escape(title))
        .replace("__SKELETON_JSON__", json.dumps(skeleton_lines))
        .replace("__FRAMES_JSON__", json.dumps(per_frame))
        .replace("__CRASHES_JSON__", json.dumps(crashes))
        .replace("__CENTER_JSON__", json.dumps([center[0], center[1]]))
        .replace("__ZOOM__", str(int(zoom)))
        # Four anchors along the green→yellow→red ramp so the legend
        # bar visually matches the gradient drawn on the streets.
        # ``vmax_vph`` is the run's p95 of positive vph, so the "+"
        # on the high end honestly indicates "≥ this rate" rather
        # than "exactly this rate".
        .replace("__C_LO__", _vph_to_color(1.0, vmax_vph))
        .replace("__C_Q__", _vph_to_color(vmax_vph * 0.25, vmax_vph))
        .replace("__C_MID__", _vph_to_color(vmax_vph * 0.5, vmax_vph))
        .replace("__C_HI__", _vph_to_color(vmax_vph, vmax_vph))
        .replace("__V_LO__", "1")
        .replace("__V_Q__", f"{vmax_vph * 0.25:.0f}")
        .replace("__V_MID__", f"{vmax_vph * 0.5:.0f}")
        .replace("__V_HI__", f"{vmax_vph:.0f}")
    )


def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _skeleton_lines_from_edges_geo(
    edges_geo: gpd.GeoDataFrame,
) -> list[list[list[float]]]:
    """Reduce a GeoDataFrame of OSM ways to ``[[lat, lon], ...]`` polylines.

    Used by both the traffic-animation and the crash-only maps so
    every renderer shows the same ground-truth network skeleton.
    """
    out: list[list[list[float]]] = []
    for _, e in edges_geo.iterrows():
        geom = e.geometry
        if geom is None or geom.is_empty:
            continue
        out.append([[float(y), float(x)] for x, y in geom.coords])
    return out


def _skeleton_with_edge_ids(
    edges_geo: gpd.GeoDataFrame,
) -> list[dict]:
    """Skeleton variant that retains the SUMO ``edge_id``.

    Used by the dual map's "highlight selected street" overlay:
    the parent page passes a list of edge IDs (resolved from the
    catalog's ``osm_way_ids → sumo_edge_ids`` map) and the iframe
    finds the matching skeleton polylines to outline.

    Each item is ``{ "edge_id": str, "coords": [[lat, lon], ...] }``.
    """
    out: list[dict] = []
    if "edge_id" not in edges_geo.columns:
        return out
    for _, e in edges_geo.iterrows():
        geom = e.geometry
        if geom is None or geom.is_empty:
            continue
        out.append({
            "edge_id": str(e["edge_id"]),
            "coords": [[float(y), float(x)] for x, y in geom.coords],
        })
    return out


# ---------------------------------------------------------------------------
# Crash-only map (separate from the traffic-animation map so the
# two stories don't compete for the viewer's attention).
# ---------------------------------------------------------------------------


def build_crash_map(
    crash_points: pd.DataFrame | None,
    out_html: Path,
    *,
    edges_geo: gpd.GeoDataFrame | None = None,
    net_path: Path = DEFAULT_NET_PATH,
    title: str = "Leonia crash overlay",
    subtitle: str | None = None,
    center: tuple[float, float] | None = None,
    zoom: int = 14,
    drop_state_system: bool = True,
    crash_segments: pd.DataFrame | None = None,
) -> Path:
    """Render a self-contained crash-only Leaflet map.

    The map is intentionally narrower in scope than the animated
    traffic map: no time slider, no edge volumes — just a road
    skeleton with crash dots on top, plus inline filter controls
    (year range, severity, pedestrian-only). This keeps the
    safety story uncluttered for the council audience.

    The same KABCO colour ramp + radius scale that the animated
    map's optional crash layer used:

    * **K** (fatal) — large dark red
    * **A** (suspected serious) — medium red
    * **B/C** (minor / possible injury) — orange
    * **O** (no apparent injury / PDO) — small grey

    ``drop_state_system=True`` (the default) hides crashes on
    Interstate / NJ Turnpike / state-highway segments so the
    overlay focuses on borough-internal safety. The headline KPI
    strip below the map shows the *visible* totals (post-filter)
    and updates live as the user toggles the controls.
    """
    if crash_points is None or crash_points.empty:
        logger.warning("No crash points; skipping crash map")
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(
            "<html><body><p style='font-family:sans-serif;color:#666'>"
            "No crash data available — run "
            "<code>scripts/14_build_crash_overlay.py</code> first."
            "</p></body></html>"
        )
        return out_html

    if edges_geo is None:
        try:
            edges_geo = load_sumo_edge_geometries(net_path)
        except Exception:
            edges_geo = gpd.GeoDataFrame(geometry=[])

    if center is None:
        try:
            if edges_geo is not None and not edges_geo.empty:
                cb = edges_geo.geometry.union_all().centroid
                center = (float(cb.y), float(cb.x))
            else:
                lat = pd.to_numeric(
                    crash_points.get("geocoded_lat",
                                      crash_points.get("latitude")),
                    errors="coerce",
                ).dropna()
                lon = pd.to_numeric(
                    crash_points.get("geocoded_lon",
                                      crash_points.get("longitude")),
                    errors="coerce",
                ).dropna()
                center = (
                    (float(lat.mean()), float(lon.mean()))
                    if not lat.empty and not lon.empty
                    else (40.864, -73.980)
                )
        except Exception:
            center = (40.864, -73.980)

    skeleton = (
        _skeleton_lines_from_edges_geo(edges_geo)
        if edges_geo is not None and not edges_geo.empty
        else []
    )

    payload = _crash_points_to_payload(
        crash_points,
        drop_state_system=drop_state_system,
        crash_segments=crash_segments,
    )
    # Augment each marker with year and ped flag so the JS filters
    # don't have to re-derive them from the label. Run the row-level
    # data through the **same** filter logic the payload helper used
    # so the row alignment stays in sync — otherwise a row dropped
    # by the payload filter (e.g. an out-of-bbox NJ-Turnpike crash)
    # would still leak its year/ped into the next surviving marker.
    df = _filter_crash_rows_to_borough(
        crash_points,
        drop_state_system=drop_state_system,
        crash_segments=crash_segments,
    )
    for i, marker in enumerate(payload):
        if i >= len(df):
            break
        row = df.iloc[i]
        try:
            marker["year"] = (
                int(row["year"]) if pd.notna(row.get("year")) else None
            )
        except (TypeError, ValueError):
            marker["year"] = None
        marker["sev_code"] = (
            str(row.get("severity_code") or "O").strip().upper()
        )
        marker["ped"] = bool(row.get("ped_involved", False))

    # Year range for the slider — clamp to whatever the data
    # actually covers so we don't show empty-trailing years.
    years = (
        sorted({m["year"] for m in payload if m.get("year") is not None})
        if payload else []
    )
    year_lo = years[0] if years else None
    year_hi = years[-1] if years else None

    if subtitle is None:
        n = len(payload)
        n_fatal = sum(1 for m in payload if m["sev_code"] == "K")
        n_ksi = sum(1 for m in payload
                    if m["sev_code"] in ("K", "A"))
        n_ped = sum(1 for m in payload if m["ped"])
        bits = [
            f"{n:,} crashes",
            f"{n_fatal} fatal",
            f"{n_ksi} KSI",
            f"{n_ped} pedestrian-involved",
        ]
        if year_lo and year_hi and year_lo != year_hi:
            bits.insert(0, f"{year_lo}–{year_hi}")
        elif year_lo:
            bits.insert(0, str(year_lo))
        if drop_state_system:
            bits.append("local streets only")
        subtitle = " · ".join(bits)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        _render_crash_map_html(
            title=title,
            subtitle=subtitle,
            center=center,
            zoom=zoom,
            skeleton=skeleton,
            crashes=payload,
            year_lo=year_lo or 2019,
            year_hi=year_hi or 2026,
        ),
        encoding="utf-8",
    )
    return out_html


def _render_crash_map_html(
    *,
    title: str,
    subtitle: str,
    center: tuple[float, float],
    zoom: int,
    skeleton: list[list[list[float]]],
    crashes: list[dict],
    year_lo: int,
    year_hi: int,
) -> str:
    return (
        _CRASH_MAP_HTML_TEMPLATE
        .replace("__TITLE__", html_escape(title))
        .replace("__SUBTITLE__", html_escape(subtitle))
        .replace("__SKELETON_JSON__", json.dumps(skeleton))
        .replace("__CRASHES_JSON__", json.dumps(crashes))
        .replace("__CENTER_JSON__", json.dumps([center[0], center[1]]))
        .replace("__ZOOM__", str(int(zoom)))
        .replace("__YEAR_LO__", str(int(year_lo)))
        .replace("__YEAR_HI__", str(int(year_hi)))
    )


_CRASH_MAP_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {
      margin: 0; padding: 0; height: 100%;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                   Roboto, Helvetica, Arial, sans-serif;
      color: #222; background: #f7f7f8;
    }
    body { display: flex; flex-direction: column; }
    header {
      padding: 10px 14px; background: #fff;
      border-bottom: 1px solid #e1e1e4;
    }
    header h1 { font-size: 16px; margin: 0; }
    header .subtitle { color: #666; font-size: 12px; margin-top: 2px; }
    #controls {
      display: flex; gap: 16px; flex-wrap: wrap;
      padding: 8px 14px; background: #fff;
      border-bottom: 1px solid #e1e1e4; font-size: 13px;
    }
    #controls .group { display: flex; align-items: center; gap: 6px; }
    #controls label { color: #555; }
    #controls input[type="range"] { vertical-align: middle; }
    #controls .check { display: inline-flex; align-items: center; gap: 4px; }
    #map { flex: 1 1 auto; min-height: 480px; }
    #legend {
      position: absolute; bottom: 14px; right: 14px; z-index: 1000;
      background: rgba(255,255,255,0.95); padding: 8px 10px;
      border: 1px solid #d0d0d4; border-radius: 4px; font-size: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    #legend .row { display: flex; align-items: center; margin: 2px 0; }
    #legend .dot {
      display: inline-block; border-radius: 50%; margin-right: 6px;
    }
    #kpis {
      display: flex; gap: 18px; flex-wrap: wrap;
      padding: 6px 14px; background: #fafafb;
      border-top: 1px solid #e1e1e4; font-size: 12px; color: #555;
    }
    #kpis b { color: #222; }
    .leaflet-popup-content {
      font-size: 12px; line-height: 1.4; margin: 8px 10px;
    }
    .leaflet-popup-content b { font-size: 13px; }
    .leaflet-popup-content .sev { display: inline-block; padding: 1px 6px;
      border-radius: 3px; font-size: 10px; font-weight: 600;
      margin-left: 4px; vertical-align: middle; }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="subtitle">__SUBTITLE__</div>
  </header>
  <div id="controls">
    <div class="group">
      <label for="year-lo">From</label>
      <input id="year-lo" type="range"
             min="__YEAR_LO__" max="__YEAR_HI__" value="__YEAR_LO__">
      <span id="year-lo-val">__YEAR_LO__</span>
    </div>
    <div class="group">
      <label for="year-hi">To</label>
      <input id="year-hi" type="range"
             min="__YEAR_LO__" max="__YEAR_HI__" value="__YEAR_HI__">
      <span id="year-hi-val">__YEAR_HI__</span>
    </div>
    <div class="group">
      <label>Severity</label>
      <label class="check"><input type="checkbox" data-sev="K" checked>K</label>
      <label class="check"><input type="checkbox" data-sev="A" checked>A</label>
      <label class="check"><input type="checkbox" data-sev="B" checked>B</label>
      <label class="check"><input type="checkbox" data-sev="C" checked>C</label>
      <label class="check"><input type="checkbox" data-sev="O" checked>O</label>
    </div>
    <div class="group">
      <label class="check">
        <input id="ped-only" type="checkbox">Pedestrian / bicyclist only
      </label>
    </div>
    <div class="group" style="margin-left:auto">
      <button id="reset" style="font-size:12px;padding:3px 8px;cursor:pointer">
        Reset filters
      </button>
    </div>
  </div>
  <div id="map">
    <div id="legend">
      <div class="row"><span class="dot"
        style="width:10px;height:10px;background:#b71c1c"></span>
        Fatal (K)</div>
      <div class="row"><span class="dot"
        style="width:8px;height:8px;background:#d32f2f"></span>
        Suspected serious (A)</div>
      <div class="row"><span class="dot"
        style="width:6px;height:6px;background:#ef6c00"></span>
        Minor injury (B)</div>
      <div class="row"><span class="dot"
        style="width:5px;height:5px;background:#f57c00"></span>
        Possible injury (C)</div>
      <div class="row"><span class="dot"
        style="width:4px;height:4px;background:#616161"></span>
        No apparent injury (O)</div>
    </div>
  </div>
  <div id="kpis">
    <span><b id="kpi-total">0</b> visible</span>
    <span><b id="kpi-fatal">0</b> fatal</span>
    <span><b id="kpi-ksi">0</b> KSI</span>
    <span><b id="kpi-ped">0</b> ped/bike</span>
    <span style="margin-left:auto;color:#888;font-size:11px">
      Source: NJDOT Crash Data Dashboard. Click a marker for details.
    </span>
  </div>
  <script>
    const SKELETON = __SKELETON_JSON__;
    const CRASHES = __CRASHES_JSON__;
    const map = L.map('map', { zoomControl: true })
                 .setView(__CENTER_JSON__, __ZOOM__);
    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      { attribution:
          '&copy; OpenStreetMap, &copy; CARTO',
        maxZoom: 19 }
    ).addTo(map);

    const skeletonLayer = L.layerGroup().addTo(map);
    SKELETON.forEach(coords => {
      L.polyline(coords, { color: '#bdbdbd', weight: 1.2,
                           opacity: 0.85 }).addTo(skeletonLayer);
    });

    const crashLayer = L.layerGroup().addTo(map);
    const SEV_LABEL = {
      'K': 'Fatal',
      'A': 'Suspected serious',
      'B': 'Minor injury',
      'C': 'Possible injury',
      'O': 'No apparent injury',
      'F': 'Fatal',
      'I': 'Injury',
      'P': 'PDO'
    };
    const SEV_BG = {
      'K': '#b71c1c', 'F': '#b71c1c',
      'A': '#d32f2f',
      'B': '#ef6c00', 'I': '#ef6c00',
      'C': '#f57c00',
      'O': '#616161', 'P': '#616161'
    };

    let allMarkers = [];
    function buildMarkers() {
      crashLayer.clearLayers();
      allMarkers = CRASHES.map(c => {
        const m = L.circleMarker([c.lat, c.lon], {
          radius: c.radius,
          color: c.color,
          weight: 1,
          fillColor: c.color,
          fillOpacity: 0.7,
        });
        const sevColor = SEV_BG[c.sev_code] || '#999';
        const sevLabel = SEV_LABEL[c.sev_code] || c.sev_code;
        const pedTag = c.ped
          ? ' <span style="background:#ffe082;color:#7a4f00;padding:1px 5px;'
            + 'border-radius:3px;font-size:10px">ped</span>'
          : '';
        const html = '<b>' + (c.label || '(unknown)') + '</b>'
          + '<span class="sev" style="background:' + sevColor
          + ';color:#fff">' + (c.sev_code || '?') + '</span>'
          + pedTag
          + '<br><span style="color:#666">' + sevLabel + '</span>'
          + (c.date ? '<br><span style="color:#888">' + c.date
                    + '</span>' : '')
          + (c.year ? ' <span style="color:#888">('
                    + c.year + ')</span>' : '');
        m.bindPopup(html);
        return { m, data: c };
      });
      applyFilters();
    }

    function applyFilters() {
      const lo = parseInt(document.getElementById('year-lo').value, 10);
      const hi = parseInt(document.getElementById('year-hi').value, 10);
      const sevChecked = new Set();
      document.querySelectorAll('#controls input[data-sev]').forEach(cb => {
        if (cb.checked) sevChecked.add(cb.dataset.sev);
      });
      const pedOnly = document.getElementById('ped-only').checked;

      let nVisible = 0, nFatal = 0, nKsi = 0, nPed = 0;
      allMarkers.forEach(({ m, data }) => {
        const sev = data.sev_code || 'O';
        const yearOk = data.year == null
          || (data.year >= lo && data.year <= hi);
        const sevOk = sevChecked.has(sev)
          || (sev === 'F' && sevChecked.has('K'))
          || (sev === 'I' && sevChecked.has('B'))
          || (sev === 'P' && sevChecked.has('O'));
        const pedOk = !pedOnly || data.ped;
        if (yearOk && sevOk && pedOk) {
          if (!map.hasLayer(m)) m.addTo(crashLayer);
          nVisible += 1;
          if (sev === 'K' || sev === 'F') nFatal += 1;
          if (sev === 'K' || sev === 'A' || sev === 'F') nKsi += 1;
          if (data.ped) nPed += 1;
        } else {
          if (map.hasLayer(m)) crashLayer.removeLayer(m);
        }
      });
      document.getElementById('kpi-total').textContent =
        nVisible.toLocaleString();
      document.getElementById('kpi-fatal').textContent = nFatal;
      document.getElementById('kpi-ksi').textContent = nKsi;
      document.getElementById('kpi-ped').textContent = nPed;
    }

    // Wire up controls.
    const yearLo = document.getElementById('year-lo');
    const yearHi = document.getElementById('year-hi');
    const yearLoVal = document.getElementById('year-lo-val');
    const yearHiVal = document.getElementById('year-hi-val');
    function syncYearLabels() {
      // Keep lo <= hi.
      if (parseInt(yearLo.value, 10) > parseInt(yearHi.value, 10)) {
        if (this === yearLo) yearHi.value = yearLo.value;
        else yearLo.value = yearHi.value;
      }
      yearLoVal.textContent = yearLo.value;
      yearHiVal.textContent = yearHi.value;
    }
    yearLo.addEventListener('input', e => { syncYearLabels.call(yearLo);
                                            applyFilters(); });
    yearHi.addEventListener('input', e => { syncYearLabels.call(yearHi);
                                            applyFilters(); });
    document.querySelectorAll('#controls input[data-sev], #ped-only')
      .forEach(cb => cb.addEventListener('change', applyFilters));
    document.getElementById('reset').addEventListener('click', () => {
      yearLo.value = __YEAR_LO__;
      yearHi.value = __YEAR_HI__;
      document.querySelectorAll('#controls input[data-sev]')
        .forEach(cb => cb.checked = true);
      document.getElementById('ped-only').checked = false;
      syncYearLabels();
      applyFilters();
    });

    buildMarkers();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dual compare map
# ---------------------------------------------------------------------------


def _edge_summary_to_geo(
    edge_summary: pd.DataFrame,
    edges_geo: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Join an edge_summary DataFrame to per-edge geometry."""
    if edge_summary.empty or edges_geo.empty:
        return gpd.GeoDataFrame(geometry=[], crs=edges_geo.crs)
    merged = edges_geo.merge(
        edge_summary, left_on="edge_id", right_on="sumo_edge_id",
        how="left",
    )
    return merged


def build_dual_compare_map(
    baseline_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    out_html: Path,
    *,
    net_path: Path = DEFAULT_NET_PATH,
    edges_geo: gpd.GeoDataFrame | None = None,
    spillover_threshold: float = 0.20,
    spillover_min_baseline_vph: float = 30.0,
    title_left: str = "Baseline",
    title_right: str = "Scenario",
    center: tuple[float, float] = (40.864, -73.980),
    zoom: int = 14,
) -> Path:
    """Render a synchronised baseline-vs-scenario folium DualMap.

    Both sub-maps share pan/zoom. Edges are coloured by simulated
    ``peak_vph`` on each side; residential streets that gain more than
    ``spillover_threshold`` (default 20%) flow on the scenario side
    are highlighted with a red dashed outline.
    """
    import folium
    from folium.plugins import DualMap

    if edges_geo is None:
        edges_geo = load_sumo_edge_geometries(net_path)
    if edges_geo.empty:
        out_html.write_text(
            "<html><body>No edge geometries available.</body></html>"
        )
        return out_html

    base_geo = _edge_summary_to_geo(baseline_summary, edges_geo)
    scen_geo = _edge_summary_to_geo(scenario_summary, edges_geo)

    delta = base_geo[["edge_id", "peak_vph"]].rename(
        columns={"peak_vph": "baseline_vph"}
    ).merge(
        scen_geo[["edge_id", "peak_vph"]].rename(
            columns={"peak_vph": "scenario_vph"}
        ),
        on="edge_id", how="outer",
    ).fillna(0.0)
    delta["delta_vph"] = delta["scenario_vph"] - delta["baseline_vph"]
    safe_baseline = delta["baseline_vph"].clip(lower=spillover_min_baseline_vph)
    delta["pct_change"] = delta["delta_vph"] / safe_baseline
    spillover_ids = set(
        delta[
            (delta["baseline_vph"] >= spillover_min_baseline_vph)
            & (delta["pct_change"] >= spillover_threshold)
        ]["edge_id"].astype(str).tolist()
    )

    vmax = max(
        50.0,
        float(np.nanpercentile(
            np.concatenate([
                base_geo["peak_vph"].fillna(0).to_numpy(),
                scen_geo["peak_vph"].fillna(0).to_numpy(),
            ]), 95,
        )),
    )

    dual = DualMap(location=list(center), zoom_start=zoom,
                   tiles="cartodbpositron")

    def _add_layer(side, geo_df: gpd.GeoDataFrame, label: str) -> None:
        for _, row in geo_df.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            vph = float(row.get("peak_vph", 0.0) or 0.0)
            color = _vph_to_color(vph, vmax)
            weight = _vph_to_width(vph, vmax)
            spillover = str(row["edge_id"]) in spillover_ids
            tooltip = (
                f"{row.get('street_name') or row['edge_id']} — "
                f"{vph:.0f} vph"
            )
            folium.PolyLine(
                [(y, x) for x, y in geom.coords],
                color=color, weight=weight, opacity=0.9,
                tooltip=tooltip,
            ).add_to(side)
            if spillover and label == title_right:
                folium.PolyLine(
                    [(y, x) for x, y in geom.coords],
                    color="#cc0000", weight=weight + 2,
                    opacity=0.7, dash_array="6,6",
                    tooltip=f"SPILLOVER: {tooltip}",
                ).add_to(side)
        folium.map.Marker(
            list(center),
            icon=folium.DivIcon(html=(
                '<div style="font-size:13px; font-weight:bold; '
                'color:#222; background:rgba(255,255,255,0.85); '
                'padding:3px 6px; border-radius:4px;">'
                f'{label}</div>'
            )),
        ).add_to(side)

    _add_layer(dual.m1, base_geo, title_left)
    _add_layer(dual.m2, scen_geo, title_right)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    dual.save(str(out_html))
    return out_html


# ---------------------------------------------------------------------------
# Sparklines
# ---------------------------------------------------------------------------


def build_sparkline(
    zone_name: str,
    simulated_hourly: pd.Series | dict[int, float] | None,
    observed_hourly: pd.Series | dict[int, float] | None,
    *,
    width_px: int = 220,
    height_px: int = 60,
    tolerance_pct: float = 0.10,
) -> str:
    """Render a per-street sparkline as inline-HTML Plotly.

    Returns an HTML snippet (``<div>...</div>``) that can be embedded
    directly into a markdown report or the stakeholder one-pager.
    Use multiple in a row to compare suspect streets.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    sim = pd.Series(simulated_hourly).reindex(range(24)) if simulated_hourly is not None else pd.Series([np.nan] * 24)
    obs = pd.Series(observed_hourly).reindex(range(24)) if observed_hourly is not None else pd.Series([np.nan] * 24)
    sim = sim.fillna(0.0)
    obs = obs.fillna(0.0)
    hours = list(range(24))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=obs, mode="lines",
        name="observed",
        line=dict(color="#888888", width=1.5, dash="dot"),
        hovertemplate="obs %{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=sim, mode="lines",
        name="simulated",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="sim %{y:.0f}<extra></extra>",
    ))

    # Within-tolerance check. Compare peak hours; if peak diff < tol → tick.
    peak_obs = float(obs.max())
    peak_sim = float(sim.max())
    if peak_obs > 0:
        rel_err = abs(peak_sim - peak_obs) / peak_obs
        within = rel_err <= tolerance_pct
    else:
        within = False
    tick = "✓" if within else "≠"
    title = f"{zone_name} {tick}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=10), x=0.0, xanchor="left"),
        margin=dict(l=4, r=4, t=18, b=4),
        width=width_px, height=height_px,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return pio.to_html(
        fig, include_plotlyjs="cdn", full_html=False,
        div_id=f"sparkline_{abs(hash(zone_name)) % 10**8}",
    )


# ---------------------------------------------------------------------------
# Stakeholder HTML
# ---------------------------------------------------------------------------


def _hourly_volume_chart(edge_history: pd.DataFrame,
                         sample_interval_s: int = 60) -> str:
    """Plotly hourly citywide volume curve. Returns an HTML snippet."""
    import plotly.graph_objects as go
    import plotly.io as pio

    if edge_history.empty:
        return "<div>(no hourly data)</div>"
    df = edge_history.copy()
    df["hour"] = df["t_bin_s"] // 3600
    hourly = df.groupby("hour", as_index=False).agg(
        vehicles=("vehicles", "sum"),
    )
    bins_per_hour = max(3600 // sample_interval_s, 1)
    hourly["vph"] = hourly["vehicles"] / bins_per_hour
    fig = go.Figure(
        go.Scatter(
            x=hourly["hour"], y=hourly["vph"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
        )
    )
    if not hourly.empty:
        peak_idx = hourly["vph"].idxmax()
        peak_hour = int(hourly.loc[peak_idx, "hour"])
        peak_val = float(hourly.loc[peak_idx, "vph"])
        fig.add_annotation(
            x=peak_hour, y=peak_val,
            text=f"peak {peak_hour:02d}:00 — {peak_val:.0f} vph",
            arrowhead=1, ax=0, ay=-30,
        )
    fig.update_layout(
        title="Citywide simulated volume by hour",
        xaxis_title="hour of day", yaxis_title="vehicles / hour",
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def _top_impacted_chart(
    summary: pd.DataFrame,
    *,
    baseline_summary: pd.DataFrame | None = None,
    top_n: int = 10,
    spillover_threshold: float = 0.20,
) -> str:
    """Plotly horizontal bar of top-impacted streets."""
    import plotly.graph_objects as go
    import plotly.io as pio

    if summary.empty:
        return "<div>(no edge summary)</div>"
    if baseline_summary is None or baseline_summary.empty:
        ranked = summary.sort_values("peak_vph", ascending=False).head(top_n)
        labels = ranked.apply(
            lambda r: f"{r.get('street_name') or r['sumo_edge_id']}",
            axis=1,
        )
        values = ranked["peak_vph"]
        colors = ["#1f77b4"] * len(ranked)
        title = "Top-10 simulated peak volumes"
        x_title = "vehicles / hour"
    else:
        merged = summary[["sumo_edge_id", "peak_vph", "street_name"]].rename(
            columns={"peak_vph": "scenario_vph"}
        ).merge(
            baseline_summary[["sumo_edge_id", "peak_vph"]].rename(
                columns={"peak_vph": "baseline_vph"}
            ),
            on="sumo_edge_id", how="outer",
        ).fillna(0.0)
        merged["delta_vph"] = merged["scenario_vph"] - merged["baseline_vph"]
        merged["abs_delta"] = merged["delta_vph"].abs()
        ranked = merged.sort_values("abs_delta", ascending=False).head(top_n)
        labels = ranked.apply(
            lambda r: f"{r.get('street_name') or r['sumo_edge_id']}",
            axis=1,
        )
        values = ranked["delta_vph"]
        colors = [
            "#cc0000" if (v > 0 and v / max(b, 30) >= spillover_threshold)
            else "#1f77b4" if v > 0
            else "#2ca02c"
            for v, b in zip(ranked["delta_vph"], ranked["baseline_vph"])
        ]
        title = "Top-10 impacted streets (Δ vehicles/hour)"
        x_title = "Δ vehicles / hour"

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        title=title, xaxis_title=x_title,
        margin=dict(l=160, r=20, t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(autorange="reversed"),
        height=380,
    )
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def _demographic_overlay(top_streets: list[str]) -> str:
    """Tabular demographic / trip-purpose context for the top streets.

    Reads ``bridge_attributes.parquet`` and computes volume-weighted
    shares across the OD pairs whose origin or destination matches
    one of ``top_streets``. Returns an HTML table snippet.
    """
    attrs_path = CANONICAL_DIR / CanonicalFiles.bridge_attributes
    if not attrs_path.exists() or not top_streets:
        return ""
    df = pd.read_parquet(attrs_path)
    df = df[(df["day_type_code"] == 0) & (df["day_part_code"] == 0)]
    streets_lower = {s.lower() for s in top_streets}
    mask = (
        df["origin_label"].fillna("").str.lower().isin(streets_lower)
        | df["destination_label"].fillna("").str.lower().isin(streets_lower)
    )
    sub = df[mask]
    if sub.empty:
        return ""
    weight = sub["od_volume"].fillna(0).astype(float)
    if weight.sum() <= 0:
        return ""

    def wmean(col: str) -> float:
        if col not in sub.columns:
            return float("nan")
        vals = sub[col].fillna(0).astype(float)
        return float((vals * weight).sum() / weight.sum())

    rows = [
        ("Home → Work", "trip_purpose::Home to Work"),
        ("Home → Other", "trip_purpose::Home to Other"),
        ("Non-home-based", "trip_purpose::Non-Home Based Trip"),
        ("Foreign-born", "equity::Foreign Born"),
        ("English limited", "equity::Limited English"),
        ("Disabled", "equity::Disability"),
        ("No vehicle", "household::No Vehicle"),
        ("Renter-occupied", "household::Renter Occupied"),
    ]
    body = []
    for label, col in rows:
        v = wmean(col)
        if pd.isna(v):
            continue
        body.append(
            f"<tr><td>{label}</td>"
            f"<td style='text-align:right'>{v * 100:.1f}%</td></tr>"
        )
    if not body:
        return ""
    table = (
        "<table style='border-collapse:collapse'>"
        "<thead><tr><th style='text-align:left'>Cohort</th>"
        "<th style='text-align:right'>Share</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )
    note = (
        "<p style='font-size:11px;color:#666'>Volume-weighted ACS small-area "
        "imputations from <code>bridge_attributes.parquet</code> for OD pairs "
        "touching the top impacted streets.</p>"
    )
    return f"<h3>Who is driving these trips?</h3>{table}{note}"


def _suspect_streets_sparklines(
    edge_history: pd.DataFrame,
    *,
    sample_interval_s: int = 60,
    max_streets: int = 8,
) -> str:
    """Build sparklines for streets named in the cut-through index."""
    idx_path = DERIVED_DIR / DerivedFiles.cutthrough_index
    if not idx_path.exists() or edge_history.empty:
        return ""
    idx = pd.read_parquet(idx_path)
    suspects = idx.dropna(subset=["osm_way_id"]).head(max_streets)

    hp_path = DERIVED_DIR / DerivedFiles.hourly_profiles
    hp = pd.read_parquet(hp_path) if hp_path.exists() else pd.DataFrame()

    df = edge_history.copy()
    df["hour"] = df["t_bin_s"] // 3600
    bins_per_hour = max(3600 // sample_interval_s, 1)

    snippets: list[str] = []
    for _, row in suspects.iterrows():
        way = int(row["osm_way_id"])
        zone_label = row.get("street_name") or row.get("zone_name") or f"OSM {way}"
        sim_hourly = (
            df[df["osm_way_id"] == way]
            .groupby("hour")["vehicles"].sum()
            / bins_per_hour
        )
        if sim_hourly.empty:
            continue
        if not hp.empty and way in set(hp["osm_way_id"].dropna().astype(int)):
            hp_row = hp[hp["osm_way_id"].astype("Int64") == way].iloc[0]
            obs_hourly = pd.Series({
                hr: float(hp_row.get(f"h{hr:02d}", float("nan")))
                for hr in range(24)
            })
        else:
            obs_hourly = pd.Series([float("nan")] * 24)
        snippets.append(
            build_sparkline(zone_label, sim_hourly, obs_hourly)
        )
    if not snippets:
        return ""
    return (
        "<h3>Per-street simulated vs observed (Visitor cohort)</h3>"
        "<div style='display:grid; grid-template-columns:repeat(2,1fr); "
        "gap:6px;'>" + "".join(snippets) + "</div>"
    )


# Street-name patterns that identify state-system / non-borough roads.
# Matched case-insensitively against the OSM ``street_name`` column on
# ``crashes_by_segment``. NJ-93 (= Grand Avenue locally) is intentionally
# **not** matched here — it's an NJDOT State Highway on paper but the
# borough has real policy levers on it (signal timing, parking, signage,
# pedestrian crossings) so it belongs in the local-streets ranking.
_STATE_SYSTEM_NAME_PATTERNS = [
    r"\bturnpike\b",
    r"\bi[-\s]?95\b",
    r"\bexpress\s*lanes?\b",
    r"\bmotorway[_\s]?link\b",
    r"^motorway$",
    r"\bgw\s*bridge\b",
    r"\bgeorge washington bridge\b",
]


def _is_state_system_street(name: object) -> bool:
    """True for OSM street names that clearly belong to a state-system road."""
    if not isinstance(name, str) or not name.strip():
        return False
    import re as _re
    s = name.strip().lower()
    return any(_re.search(p, s) for p in _STATE_SYSTEM_NAME_PATTERNS)


# NJDOT crash on-road (``crash_location``) free-text patterns for
# limited-access state facilities that never run on a Leonia surface
# street. Superset of the OSM-name patterns plus ``interstate`` (NJDOT
# writes "INTERSTATE 95" where OSM uses "I-95"). Used to reject crashes
# that NJDOT geocoded/snapped onto a parallel borough street (Broad /
# Grand Avenue) even though the crash is actually on I-95 / the Turnpike.
_CRASH_ONROAD_NONLOCAL_PATTERNS = _STATE_SYSTEM_NAME_PATTERNS + [
    r"\binterstate\b",
]


def _is_nonlocal_crash_onroad(text: object) -> bool:
    """True when a crash's on-road text names a non-Leonia state facility.

    Keyed off the *on-road* only (``crash_location``). A crash whose
    on-road is a local street but whose cross-street merely mentions I-95
    is a genuine local crash at the interchange and is **not** flagged.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    import re as _re
    s = text.strip().lower()
    return any(_re.search(p, s) for p in _CRASH_ONROAD_NONLOCAL_PATTERNS)


# OSM ``way`` records are split at intersections, so a single street
# like "Broad Avenue" appears as 5+ separate rows in
# ``crashes_by_segment``. To collapse them into one row in the
# council table we need a normalisation that matches:
#
# * "Broad Avenue", "BROAD AVE", "BROAD AVENUE", "Broad Ave"
# * "BROAD AVE / DANA PL" → matches "Broad Avenue" (compound names
#   from old NJDOT free-text coexist with the OSM short form)
# * "Fort Lee Road (BERGEN COUNTY 56 3)" → matches "Fort Lee Road"
# * Non-overlapping single-quote / double-quote variants.
#
# The trick is to (a) take the first segment before any ``/``, (b)
# strip parenthetical disambiguators, (c) drop punctuation, (d)
# expand the standard street-suffix abbreviations to full words so
# "Ave" and "Avenue" hash the same.
_STREET_SUFFIX_EXPANSIONS = {
    "AVE": "AVENUE", "AV": "AVENUE",
    "ST": "STREET", "STR": "STREET",
    "RD": "ROAD",
    "DR": "DRIVE",
    "PL": "PLACE",
    "TER": "TERRACE", "TERR": "TERRACE",
    "BLVD": "BOULEVARD",
    "CT": "COURT",
    "LN": "LANE",
    "PKWY": "PARKWAY",
    "HWY": "HIGHWAY",
    "CIR": "CIRCLE",
}


def _canonical_street_key(name: object) -> str:
    """Map free-text street names to a stable canonical key.

    The key is intentionally lossy: it's only used as a groupby key
    in :func:`_safety_panel`, never displayed. Returns ``""`` for
    non-string / blank inputs (callers should fall back to the OSM
    way id in that case).
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    import re as _re
    s = name.strip()
    # Drop parentheticals: "Fort Lee Road (BERGEN COUNTY 56 3)".
    s = _re.sub(r"\s*\([^)]*\)", "", s)
    # Take the head before a slash: "BROAD AVE / DANA PL" → "BROAD AVE".
    s = s.split("/")[0]
    s = s.upper()
    # Drop punctuation, the legacy NJDOT ``**`` markers, and collapse
    # internal whitespace.
    s = _re.sub(r"[^A-Z0-9\s]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    parts = s.split(" ")
    parts = [_STREET_SUFFIX_EXPANSIONS.get(p, p) for p in parts]
    return " ".join(parts)


def _safety_totals(
    crash_points: pd.DataFrame | None,
    crash_segments: pd.DataFrame,
    state_system_mask: pd.Series,
    has_kabco: bool,
) -> tuple[int, int, int, int, int]:
    """Compute (total, fatal, ksi, ped, serious) for the safety panel headline.

    Prefers row-level counts (matching the crash map's denominator);
    falls back to segment-table sums when row-level data isn't
    available. Returns ``(n_total, n_fatal, n_ksi, n_ped, n_serious)``;
    ``n_serious`` is always 0 for legacy F/I/P data.
    """
    seg_local_all = crash_segments[~state_system_mask]

    # Fast path: no row-level data → fall back to segment sums.
    if (crash_points is None or crash_points.empty
            or "geocoded_osm_way_id" not in crash_points.columns):
        return (
            int(seg_local_all["n_crashes"].sum()),
            int(seg_local_all["n_fatal"].sum()),
            int(seg_local_all["n_ksi"].sum()),
            int(seg_local_all["n_ped"].sum()),
            int(seg_local_all["n_serious"].sum()) if has_kabco else 0,
        )

    way_to_name = (
        crash_segments[["osm_way_id", "street_name"]]
        .drop_duplicates("osm_way_id")
        .set_index("osm_way_id")["street_name"]
        .to_dict()
    )
    way_ids = pd.to_numeric(
        crash_points["geocoded_osm_way_id"], errors="coerce",
    )
    osm_name = way_ids.map(way_to_name)
    is_state = osm_name.apply(_is_state_system_street)
    local = crash_points[~is_state]
    sev = local.get("severity_code", pd.Series(dtype=object))
    n_total = int(len(local))
    n_fatal = int(sev.isin(["K", "F"]).sum())
    n_serious = int((sev == "A").sum()) if has_kabco else 0
    n_ksi = (
        int(sev.isin(["K", "A"]).sum()) if has_kabco
        else int(sev.isin(["F", "I"]).sum())
    )
    n_ped = int(local.get(
        "ped_involved", pd.Series(dtype=bool)
    ).fillna(False).astype(bool).sum())
    return (n_total, n_fatal, n_ksi, n_ped, n_serious)


def _crash_trend_chart(
    crash_points: pd.DataFrame | None,
    crash_segments: pd.DataFrame | None = None,
    *,
    drop_state_system: bool = True,
    partial_year_threshold: float = 0.30,
) -> str:
    """Year-over-year crash volume chart for the stakeholder one-pager.

    Stacked bars: PDO (no apparent injury) at the bottom in grey,
    other-injury (B/C) above it in orange, KSI (K + A) at the top
    in red. A line overlay tracks pedestrian-involved crashes on
    the same axis. Years that look incomplete (e.g. the current
    partial year, or a NJDOT data-delivery gap) get an explicit
    "partial year" hatch overlay so the council doesn't read a
    data hole as a policy win.

    The "incomplete" heuristic is intentionally simple: any year
    whose crash count is below ``partial_year_threshold`` of the
    median full-year count gets flagged. This catches both the
    current-year-in-progress case and the 2023 data-delivery gap
    visible in the Leonia dataset without requiring a hand-curated
    list of bad years.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    if crash_points is None or crash_points.empty:
        return ""
    df = crash_points.copy()

    # Apply the same OSM-way-based local-streets filter as
    # ``_safety_panel`` so the trend reflects the same denominator.
    if drop_state_system:
        if (crash_segments is not None and not crash_segments.empty
                and "geocoded_osm_way_id" in df.columns
                and "osm_way_id" in crash_segments.columns):
            way_to_name = (
                crash_segments[["osm_way_id", "street_name"]]
                .drop_duplicates("osm_way_id")
                .set_index("osm_way_id")["street_name"]
                .to_dict()
            )
            way_ids = pd.to_numeric(
                df["geocoded_osm_way_id"], errors="coerce",
            )
            osm_name = way_ids.map(way_to_name)
            df = df[~osm_name.apply(_is_state_system_street)]
        elif "road_system" in df.columns:
            df = df[~df["road_system"].apply(_is_state_system)]

    df = df[pd.to_numeric(df["year"], errors="coerce").notna()].copy()
    if df.empty:
        return ""
    df["year"] = df["year"].astype(int)

    df["sev"] = df.get("severity_code", pd.Series(dtype=object))
    df["is_fatal"] = df["sev"].isin(["K", "F"])
    df["is_serious"] = df["sev"] == "A"
    df["is_other_injury"] = df["sev"].isin(["B", "C", "I"])
    df["is_pdo"] = df["sev"].isin(["O", "P"])
    df["is_ped"] = df.get("ped_involved", False).astype(bool)

    yearly = df.groupby("year").agg(
        n=("crash_id", "count"),
        n_fatal=("is_fatal", "sum"),
        n_serious=("is_serious", "sum"),
        n_other_injury=("is_other_injury", "sum"),
        n_pdo=("is_pdo", "sum"),
        n_ped=("is_ped", "sum"),
    ).reset_index()
    yearly = yearly.sort_values("year")
    if yearly.empty:
        return ""
    yearly["n_ksi"] = yearly["n_fatal"] + yearly["n_serious"]

    # Identify partial / incomplete years using two heuristics:
    # 1. **Current calendar year** is always partial-by-definition.
    # 2. **Obvious data-delivery gaps** — years with < 30% of the
    #    median count are very unlikely to be a real 70% drop and
    #    almost always reflect a NJDOT pipeline gap (we observe this
    #    for 2023 in the Leonia data, where only 6 rows came through
    #    vs ~200 in every adjacent year).
    # We deliberately don't flag years like 2022 (97 vs median ~200,
    # ~50% of median) — those could be real reductions.
    from datetime import datetime as _dt
    current_year = _dt.now(timezone.utc).year
    median_n = float(yearly["n"].median()) if not yearly.empty else 0.0
    yearly["is_partial"] = (
        yearly["year"] >= current_year
    ) | (
        (yearly["n"] < partial_year_threshold * median_n)
        if median_n > 0 else False
    )

    years_str = yearly["year"].astype(str).tolist()
    full_mask = ~yearly["is_partial"]

    # Coerce the y-values to plain Python int lists. Plotly 6+
    # auto-encodes small integer numpy/pandas arrays as base64
    # `bdata` for performance, but for our 6–8-element series
    # that encoding has been observed to cause Plotly.js (CDN
    # 3.x) to render the bars at y=0 even though the data is
    # round-trippable. Plain JSON arrays sidestep the issue.
    pdo_y = [int(v) for v in yearly["n_pdo"].tolist()]
    inj_y = [int(v) for v in yearly["n_other_injury"].tolist()]
    ksi_y = [int(v) for v in yearly["n_ksi"].tolist()]
    ped_y = [int(v) for v in yearly["n_ped"].tolist()]

    fig = go.Figure()
    fig.add_bar(
        name="No apparent injury", x=years_str, y=pdo_y,
        marker_color="#bdbdbd",
        hovertemplate="%{x}<br>PDO: %{y}<extra></extra>",
    )
    fig.add_bar(
        name="Injury (B/C)", x=years_str, y=inj_y,
        marker_color="#ef6c00",
        hovertemplate="%{x}<br>Injury: %{y}<extra></extra>",
    )
    fig.add_bar(
        name="KSI (K + A)", x=years_str, y=ksi_y,
        marker_color="#b71c1c",
        hovertemplate="%{x}<br>KSI: %{y}<extra></extra>",
    )
    # Pedestrian-involved as a line overlay so it stays comparable
    # year-over-year regardless of stack height.
    fig.add_scatter(
        name="Pedestrian-involved",
        x=years_str, y=ped_y,
        mode="lines+markers",
        line=dict(color="#7b1fa2", width=2.5),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="%{x}<br>Ped: %{y}<extra></extra>",
    )
    # "Partial year" annotations — anchor at the top of the stack
    # so the badge sits just above the bar regardless of stack
    # composition. ``yshift`` pushes it 18px clear of the bar.
    for _, row in yearly.iterrows():
        if bool(row["is_partial"]):
            fig.add_annotation(
                x=str(int(row["year"])),
                y=int(row["n"]),
                text="partial<br>year",
                showarrow=False, yshift=18,
                font=dict(size=10, color="#666"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1, borderpad=2,
            )

    # Linear trend line on full years only — gives the council a
    # one-glance answer to "is this getting better or worse?"
    sub_full = yearly[full_mask]
    if len(sub_full) >= 2:
        import numpy as _np
        coeffs = _np.polyfit(sub_full["year"], sub_full["n"], 1)
        slope = float(coeffs[0])
        x_full = sub_full["year"].astype(int).tolist()
        y_fit = [float(coeffs[0] * y + coeffs[1]) for y in x_full]
        trend_color = "#2e7d32" if slope < 0 else "#c62828"
        fig.add_scatter(
            name=(f"Full-year trend ({slope:+.0f}/yr)"),
            x=[str(y) for y in x_full],
            y=y_fit,
            mode="lines",
            line=dict(color=trend_color, width=2, dash="dash"),
            hoverinfo="skip",
        )

    n_full = int(full_mask.sum())
    subtitle_bits: list[str] = []
    if not sub_full.empty:
        median = int(sub_full["n"].median())
        subtitle_bits.append(
            f"median full-year count = {median:,} crashes/yr "
            f"across {n_full} year{'s' if n_full != 1 else ''}"
        )
    if (yearly["is_partial"]).any():
        bad_years = yearly.loc[yearly["is_partial"], "year"].astype(int).tolist()
        subtitle_bits.append(
            "partial / NJDOT-underreported: "
            + ", ".join(str(y) for y in bad_years)
        )
    subtitle = " · ".join(subtitle_bits)

    # Pin both axis ranges explicitly. Without this, Plotly.js
    # (3.x via the CDN) computes the wrong autorange on first paint
    # when the chart's container has ``width: 100%`` and the parent
    # hasn't settled its width yet — the bars end up squished into
    # the leftmost x position. Manually zooming forces a relayout
    # that recomputes the range from scratch and fixes it. By
    # pinning the ranges up front we get the post-zoom layout on
    # first paint.
    n_years = len(years_str)
    stack_max = max(
        (a + b + c for a, b, c in zip(pdo_y, inj_y, ksi_y)),
        default=0,
    )
    y_max = max(stack_max, max(ped_y, default=0))
    # Add 12% headroom so the "partial year" badges sit clear of
    # the bar tops.
    y_top = max(int(y_max * 1.12) + 1, 10)

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=("Crashes per year on Leonia local streets"
                  + (f"<br><sub style='color:#666'>{subtitle}</sub>"
                     if subtitle else "")),
            x=0, xanchor="left",
        ),
        yaxis=dict(title="crashes", range=[0, y_top], autorange=False),
        margin=dict(l=40, r=20, t=70, b=80),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, yanchor="top"),
        height=380,
    )
    # ``type="category"`` is mandatory: numeric strings like "2019"
    # can otherwise be coerced to a numeric axis at render time,
    # which collapses all 8 bars onto x=0 because Plotly.js's
    # numeric-axis autorange logic gets confused when the same x
    # appears across stacked traces. ``update_xaxes`` reliably sets
    # the axis ``type`` (passing it via ``update_layout(xaxis=...)``
    # silently ignores ``type`` in some plotly.py versions when
    # the figure was first laid out as numeric).
    #
    # ``range`` is also pinned with a small fractional pad on each
    # side: ``[-0.5, n-0.5]`` makes Plotly.js treat the categorical
    # axis as a closed numeric range internally — which sidesteps
    # the same width-not-yet-settled bug as the y-axis.
    fig.update_xaxes(
        title_text="year",
        type="category",
        categoryorder="array",
        categoryarray=years_str,
        range=[-0.5, n_years - 0.5],
        autorange=False,
    )
    # ``post_script`` runs after Plotly draws the chart. Calling
    # ``Plotly.Plots.resize`` forces a relayout against the now-
    # settled container width, which is exactly the fix the user
    # was doing manually by zooming in. With the ranges pinned
    # above this is mostly a belt-and-braces measure for browsers
    # that lay out asynchronously.
    return pio.to_html(
        fig, include_plotlyjs="cdn", full_html=False,
        post_script=(
            "setTimeout(function() {"
            "  if (window.Plotly && document.getElementById('{plot_id}')) {"
            "    Plotly.Plots.resize(document.getElementById('{plot_id}'));"
            "  }"
            "}, 50);"
        ),
    )


def _safety_panel(
    edge_summary: pd.DataFrame | None,
    crash_segments: pd.DataFrame | None,
    crash_points: pd.DataFrame | None = None,
    *,
    top_n: int = 50,
) -> str:
    """Render the NJDOT crash overlay block for the stakeholder one-pager.

    The panel pulls triple duty: (1) headline borough crash totals
    using the FHWA KABCO scale when available, (2) top-EPDO **local**
    corridors table (Interstate / NJ Turnpike / motorway-link
    segments are filtered out — they're NJTA/NJDOT jurisdiction and
    crowd out the streets where the borough actually has policy
    levers), (3) flags any corridor that also ranks as a simulated
    cut-through so the council can see at a glance that "Broad
    Avenue is both the busiest cut-through and the highest-EPDO
    street" without flipping between visualisations.

    ``crash_points`` (optional) is the row-level parquet — used to
    derive the year-range subtitle. ``crash_segments`` is the
    pre-aggregated per-OSM-way table from
    :func:`leonia_traffic.data.njdot_crash_loader.aggregate_by_segment`.
    """
    if crash_segments is None or crash_segments.empty:
        return ""

    seg = crash_segments.copy()
    if "epdo_total" not in seg.columns:
        return ""

    # Drop non-borough state-system rows (NJ Turnpike / I-95 / etc.)
    # so the top-10 reflects streets the borough can actually act on.
    state_system_mask = seg.get(
        "street_name", pd.Series([], dtype=object)
    ).apply(_is_state_system_street)
    n_state_dropped = int(state_system_mask.sum())
    seg_no_state = seg[~state_system_mask].copy()

    # OSM splits each road into a chain of ``way`` records broken at
    # intersections, so a single street like "Broad Avenue" appears
    # as 5+ separate rows in ``crashes_by_segment``. Collapse them
    # with a normalised street-name key so the council table reads
    # as "one row per actual street" — that's the mental model
    # they're working with.
    seg_no_state["_street_key"] = (
        seg_no_state.get(
            "street_name", pd.Series([], dtype=object)
        ).fillna("").apply(_canonical_street_key)
    )
    # Use an OSM way id as a stable fallback when the OSM
    # ``street_name`` is blank (rare but it does happen on some
    # service roads / driveways).
    seg_no_state.loc[seg_no_state["_street_key"] == "", "_street_key"] = (
        "way_" + seg_no_state.get(
            "osm_way_id", pd.Series([], dtype=object)
        ).astype(str)
    )

    def _pick_display_name(s: pd.Series) -> str:
        # Pick the most-common spelling among the OSM-way fragments;
        # break ties by preferring (a) Title-Case over ALL-CAPS (the
        # NJDOT zip data is all-caps, OSM is Title Case and reads
        # better in a council slide) and then (b) the longest
        # spelling so abbreviations like "BROAD AVE" lose to
        # "Broad Avenue".
        clean = s.dropna()
        if clean.empty:
            return ""
        counts = clean.value_counts()
        top_n = counts[counts == counts.iloc[0]].index.tolist()
        top_n.sort(key=lambda x: (
            x.isupper(),       # Title-Case wins over ALL-CAPS
            -len(x),           # then longest
        ))
        return top_n[0]

    agg_spec = {
        "n_crashes": "sum", "n_fatal": "sum",
        "n_injury": "sum", "n_pdo": "sum", "n_ksi": "sum",
        "n_ped": "sum", "epdo_total": "sum",
        "first_year": "min", "last_year": "max",
        "street_name": _pick_display_name,
    }
    if "n_serious" in seg_no_state.columns:
        agg_spec["n_serious"] = "sum"

    seg_by_street = (
        seg_no_state.groupby("_street_key", as_index=False, sort=False)
        .agg({**agg_spec,
              # n_segments = how many OSM ways merged into this row.
              "osm_way_id": "count"})
        .rename(columns={"osm_way_id": "n_segments"})
    )
    seg_by_street["years_covered"] = (
        seg_by_street["last_year"].astype("Int64")
        - seg_by_street["first_year"].astype("Int64") + 1
    )
    # Default sort by raw crash count — that's the metric residents
    # see in their daily life (more crashes on my street = bigger
    # problem). EPDO is still the most decision-relevant column for
    # capital prioritisation, but it's a derived index and harder to
    # explain in a council slide. Keep it as the rightmost column so
    # the eye still lands on it.
    seg_local = seg_by_street.sort_values(
        "n_crashes", ascending=False,
    ).head(top_n)

    sim_streets: set[str] = set()
    if edge_summary is not None and not edge_summary.empty:
        ranked = edge_summary.sort_values("peak_vph", ascending=False).head(15)
        sim_streets = {
            s.strip().lower()
            for s in ranked.get("street_name", pd.Series([])).dropna().tolist()
            if isinstance(s, str)
        }

    # Derive the year range from the row-level data when available;
    # fall back to the segment table's first_year/last_year.
    year_lo = year_hi = None
    if crash_points is not None and "year" in crash_points.columns:
        years = pd.to_numeric(crash_points["year"], errors="coerce").dropna()
        if not years.empty:
            year_lo = int(years.min())
            year_hi = int(years.max())
    if year_lo is None and "first_year" in crash_segments.columns:
        year_lo = int(crash_segments["first_year"].min())
        year_hi = int(crash_segments["last_year"].max())
    year_label = (
        f"{year_lo}–{year_hi}" if year_lo and year_hi and year_lo != year_hi
        else (str(year_lo) if year_lo else "")
    )

    # Detect KABCO data (presence of an `n_serious` column from
    # ``aggregate_by_segment``) so we can show separate "fatal" /
    # "suspected serious" / "other injury" buckets.
    seg_local_all = crash_segments[~state_system_mask]
    has_kabco = "n_serious" in crash_segments.columns

    # Headline totals come from the **row-level** parquet when it's
    # available (filtered the same way the crash map filters), so
    # this panel and the crash map below it always show the same
    # denominator. Falls back to the segment-table sum when the
    # row-level data isn't supplied — that's a strictly smaller
    # number because it excludes rows that didn't snap to any OSM
    # way, but it's still internally consistent with the per-street
    # rows in the table.
    n_total, n_fatal, n_ksi, n_ped, n_serious = _safety_totals(
        crash_points, crash_segments, state_system_mask, has_kabco,
    )
    epdo_total = float(seg_local_all["epdo_total"].sum())

    # Compare against the *normalised* simulated cut-through keys so
    # we still flag "Broad Avenue" even when the simulation reports
    # "BROAD AVE / DANA PL" or similar.
    sim_street_keys = {
        _canonical_street_key(s) for s in sim_streets
    } - {""}

    rows_html: list[str] = []
    for _, r in seg_local.iterrows():
        street = r.get("street_name") or f"OSM way {r.get('_street_key')}"
        street_key = r.get("_street_key", "")
        is_sim_hot = isinstance(street_key, str) and street_key in sim_street_keys
        flag = (
            ' <span style="background:#fff3cd;color:#7a5a00;font-size:11px;'
            'padding:1px 6px;border-radius:3px;margin-left:4px">'
            'simulated cut-through</span>'
            if is_sim_hot else ""
        )
        # Show how many OSM segments rolled up into this street so
        # the council can tell whether a high count is concentrated
        # at a single intersection or spread along the corridor.
        n_segments = int(r.get("n_segments", 1))
        seg_note = (
            f' <span style="color:#888;font-size:11px">'
            f'({n_segments} segments)</span>'
            if n_segments > 1 else ""
        )
        ksi_cell = (
            f"<td style='text-align:right'>{int(r.get('n_ksi', 0))}</td>"
        )
        rows_html.append(
            f"<tr><td style='font-weight:600'>"
            f"{html_escape(str(street))}{seg_note}{flag}</td>"
            f"<td style='text-align:right'>{int(r['n_crashes'])}</td>"
            f"<td style='text-align:right'>{int(r['n_fatal'])}</td>"
            + ksi_cell
            + f"<td style='text-align:right'>{int(r['n_ped'])}</td>"
              f"<td style='text-align:right;font-weight:600'>"
              f"{float(r['epdo_total']):.0f}</td></tr>"
        )

    headline_bits = [f"<b>{n_total:,}</b> reported crashes",
                     f"<b>{n_fatal}</b> fatal"]
    if has_kabco:
        headline_bits.append(f"<b>{n_serious}</b> suspected serious")
    headline_bits.extend([
        f"<b>{n_ksi}</b> KSI",
        f"<b>{n_ped}</b> pedestrian-involved",
        f"local-streets EPDO = <b>{epdo_total:,.0f}</b>",
    ])
    headline = " · ".join(headline_bits)

    weight_note = (
        "Fatal ×542, suspected serious ×66, minor/possible injury ×11, "
        "no-apparent-injury ×1 (FHWA KABCO scale, NJDOT-adjusted)."
        if has_kabco
        else "Fatal ×542, injury ×11, PDO ×1 (NJDOT F/I/P scale)."
    )

    title = (
        f"Safety overlay — NJDOT crashes on Leonia local streets "
        f"{year_label}".strip(" —")
        if year_label
        else "Safety overlay — NJDOT crashes on Leonia local streets"
    )

    state_system_note = (
        f" {n_state_dropped} segment"
        f"{'s' if n_state_dropped != 1 else ''} on the NJ Turnpike / "
        f"I-95 corridor are excluded — they're NJTA / NJDOT "
        f"jurisdiction (see the crash map below for the full picture)."
        if n_state_dropped else ""
    )

    return (
        f"<h3>{title}</h3>"
        f"<p style='color:#666;margin-top:0'>{headline}</p>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr style='border-bottom:1px solid #c0c0c4;'>"
        f"<th style='text-align:left'>Street</th>"
        f"<th style='text-align:right'>Crashes</th>"
        f"<th style='text-align:right'>Fatal</th>"
        f"<th style='text-align:right'>KSI</th>"
        f"<th style='text-align:right'>Ped</th>"
        f"<th style='text-align:right'>EPDO</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
        f"<p style='color:#888;font-size:11px;margin-top:8px'>"
        f"EPDO = Equivalent Property-Damage-Only. {weight_note} "
        f"KSI = Killed + Suspected Serious Injury (FHWA convention). "
        f"Source: NJDOT Crash Data Dashboard, geocoded to OSM ways."
        f"{state_system_note}</p>"
    )


_STAKEHOLDER_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  {% raw %}
  <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        margin: 0; padding: 24px; color: #222; background: #f7f7f8;
        max-width: 1200px; margin: 0 auto;
    }
    h1, h2, h3 { margin: 16px 0 8px; }
    .kpis { display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px; margin-bottom: 16px; }
    .kpi { background: #fff; border: 1px solid #e1e1e4;
           border-radius: 6px; padding: 12px; }
    .kpi .label { color: #666; font-size: 12px; text-transform: uppercase; }
    .kpi .value { font-size: 24px; font-weight: bold; margin-top: 4px; }
    .panel { background: #fff; border: 1px solid #e1e1e4;
             border-radius: 6px; padding: 12px; margin-bottom: 16px; }
    iframe { border: none; width: 100%; height: 540px;
             border-radius: 4px; }
    table { font-size: 13px; }
    th, td { padding: 4px 10px; border-bottom: 1px solid #eee; }
    @media print { body { background: #fff; } .panel { break-inside: avoid; } }
  </style>
  {% endraw %}
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <p style="color:#666">{{ subtitle }}</p>
  </header>

  <section class="kpis">
    {% for kpi in kpis %}
    <div class="kpi">
      <div class="label">{{ kpi.label }}</div>
      <div class="value">{{ kpi.value }}</div>
    </div>
    {% endfor %}
  </section>

  {% if crash_map_iframe %}
  <section class="panel">
    <h3>Where crashes happen</h3>
    <p style="color:#666;margin-top:0;font-size:13px">
      Reported NJDOT crashes geocoded to the borough's road network.
      Use the controls to filter by year range, severity, or to focus
      on pedestrian / bicyclist involvement only. NJ Turnpike / I-95
      / motorway-link segments are hidden by default — they're NJTA
      jurisdiction. Grand Avenue (NJ-93) is kept because the borough
      has policy levers there (signal timing, parking, signage).
    </p>
    {{ crash_map_iframe|safe }}
  </section>
  {% endif %}

  {% if crash_trend %}
  <section class="panel">{{ crash_trend|safe }}</section>
  {% endif %}

  {% if safety %}
  <section class="panel">{{ safety|safe }}</section>
  {% endif %}

  {% if hourly_chart %}
  <section class="panel">{{ hourly_chart|safe }}</section>
  {% endif %}

  {% if top_chart %}
  <section class="panel">{{ top_chart|safe }}</section>
  {% endif %}

  {% if animations %}
  {% for anim in animations %}
  <section class="panel">
    <h3>{{ anim.title }}</h3>
    {% if anim.subtitle %}
    <p style="color:#666;margin-top:0;font-size:13px">{{ anim.subtitle }}</p>
    {% endif %}
    {{ anim.iframe|safe }}
  </section>
  {% endfor %}
  {% endif %}

  {% if demographics %}
  <section class="panel">{{ demographics|safe }}</section>
  {% endif %}

  {% if sparklines %}
  <section class="panel">{{ sparklines|safe }}</section>
  {% endif %}

  <footer style="color:#666; font-size: 12px; margin-top: 24px;">
    Generated by leonia_traffic.sumo.visualizations on
    {{ generated_at }}.
  </footer>
</body>
</html>
"""


def build_stakeholder_html(
    out_html: Path,
    *,
    edge_history: pd.DataFrame,
    edge_summary: pd.DataFrame,
    score: dict | None = None,
    baseline_summary: pd.DataFrame | None = None,
    animated_map: Path | None = None,
    animations: list[dict] | None = None,
    crash_map: Path | None = None,
    title: str = "Leonia simulated traffic",
    subtitle: str = "",
    sample_interval_s: int = 60,
) -> Path:
    """Render the council-meeting one-pager.

    All inputs are DataFrames except the various ``Path`` args,
    which (when given) are embedded as ``<iframe>``s so a single
    self-contained open of ``out_html`` shows them too. Pass them
    as relative paths when possible so the HTML stays portable.

    Animation embedding accepts two shapes for backward compat:

    * ``animated_map=Path`` — single animation, embedded under the
      header *Hour-by-hour traffic animation*. This is the legacy
      shape used by ``scripts/12_sumo_baseline.py`` and the SUMO
      scenarios script when only one demand is being simulated.
    * ``animations=[{"title": ..., "subtitle": ..., "path": Path}, ...]``
      — multiple animations rendered as separate panels in order.
      Used by the combined weekday-vs-Sunday driver to embed both
      24-hour animations side-by-side. When both arguments are
      provided ``animations`` wins.
    """
    from jinja2 import Template

    n_inserted = (
        int(edge_history["vehicles"].sum())
        if not edge_history.empty else 0
    )
    mean_speed = (
        float(edge_history["mean_speed_ms"].mean()) / 0.44704
        if not edge_history.empty else 0.0
    )
    if score is None:
        score = {"pct_lt_5": float("nan"), "n_links_scored": 0}
    geh_pass = score.get("pct_lt_5", float("nan"))
    geh_pass_str = (
        f"{geh_pass * 100:.0f}%" if pd.notna(geh_pass) else "—"
    )
    if not edge_history.empty:
        df = edge_history.copy()
        df["hour"] = df["t_bin_s"] // 3600
        bins_per_hour = max(3600 // sample_interval_s, 1)
        hourly = df.groupby("hour")["vehicles"].sum() / bins_per_hour
        peak_hour = int(hourly.idxmax()) if not hourly.empty else 0
        peak_label = f"{peak_hour:02d}:00"
    else:
        peak_label = "—"

    kpis = [
        {"label": "Total vehicle-bins", "value": f"{n_inserted:,}"},
        {"label": "Mean speed", "value": f"{mean_speed:.1f} mph"},
        {"label": "GEH < 5", "value": geh_pass_str},
        {"label": "Links scored",
         "value": f"{score.get('n_links_scored', 0):,}"},
        {"label": "Peak hour", "value": peak_label},
    ]

    hourly_chart = _hourly_volume_chart(edge_history,
                                        sample_interval_s=sample_interval_s)
    top_chart = _top_impacted_chart(edge_summary,
                                    baseline_summary=baseline_summary)

    def _iframe_src(p: Path) -> str:
        # Prefer a relative path so the HTML is portable.
        try:
            return str(p.relative_to(out_html.parent))
        except ValueError:
            return str(p)

    # Resolve the animation list. New callers pass ``animations=[...]``;
    # legacy callers pass a single ``animated_map=Path`` which we
    # promote to the same one-element list shape.
    if animations is None and animated_map is not None:
        animations = [{
            "title": "Hour-by-hour traffic animation",
            "subtitle": "",
            "path": animated_map,
        }]
    rendered_animations: list[dict] = []
    for anim in (animations or []):
        p = anim.get("path")
        if p is None or not Path(p).exists():
            continue
        rendered_animations.append({
            "title": anim.get("title", "Hour-by-hour traffic animation"),
            "subtitle": anim.get("subtitle", ""),
            "iframe": f'<iframe src="{_iframe_src(Path(p))}"></iframe>',
        })
    # Crash map gets a tall iframe so the filter strip + KPI footer
    # are all visible without inner-iframe scrolling. This is the
    # top panel of the stakeholder report, so giving it room pays
    # off — the council should see the geographic story first.
    crash_map_iframe = (
        f'<iframe src="{_iframe_src(crash_map)}" '
        f'style="height:840px"></iframe>'
        if crash_map is not None and crash_map.exists()
        else ""
    )

    top_streets: list[str] = []
    if not edge_summary.empty:
        ranked = edge_summary.sort_values("peak_vph", ascending=False).head(5)
        top_streets = [
            s for s in ranked.get("street_name", pd.Series([])).tolist()
            if isinstance(s, str) and s
        ]
    demographics = _demographic_overlay(top_streets)

    sparklines = _suspect_streets_sparklines(
        edge_history, sample_interval_s=sample_interval_s,
    )

    crash_segments_df = load_crash_segments_if_available()
    crash_points_df = load_crash_points_if_available()
    safety = _safety_panel(
        edge_summary, crash_segments_df, crash_points_df,
    )
    crash_trend = _crash_trend_chart(crash_points_df, crash_segments_df)

    template = Template(_STAKEHOLDER_TEMPLATE)
    html = template.render(
        title=title,
        subtitle=subtitle,
        kpis=kpis,
        hourly_chart=hourly_chart,
        top_chart=top_chart,
        animations=rendered_animations,
        crash_map_iframe=crash_map_iframe,
        safety=safety,
        crash_trend=crash_trend,
        demographics=demographics,
        sparklines=sparklines,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html
