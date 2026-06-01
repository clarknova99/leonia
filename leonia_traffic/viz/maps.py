"""Folium-based interactive maps for Leonia traffic data."""

from __future__ import annotations

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np


_LEONIA_CENTER = (40.864, -73.985)


def _first_present(row, candidates: tuple[str, ...], default: str = "—") -> str:
    """Return the first column value from ``candidates`` that's set on ``row``.

    StreetLight, StreetScanner, OSM, and the ZA exports each call the
    same field by different names (e.g. ``road_name`` vs ``street_name``
    vs ``osm_name``). Tooltips need to work across all of them, so we
    try each candidate in order and skip ``None`` / ``NaN`` / ``"N/A"``
    / empty strings.
    """
    for col in candidates:
        if col not in row.index:
            continue
        val = row[col]
        if val is None:
            continue
        # NaN check that doesn't blow up on strings.
        try:
            if isinstance(val, float) and np.isnan(val):
                continue
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        if not s or s.lower() in ("n/a", "nan", "none"):
            continue
        return s
    return default


_NAME_COLS = ("road_name", "osm_name", "street_name", "name")
_PLACE_COLS = ("city_county_state", "city", "municipality", "county")
_SPEED_OBS_COLS = ("avg_speed_mph", "speed_avg_mph", "avg_speed")
_SPEED_LIMIT_COLS = ("speed_limit_mph", "posted_speed_mph", "posted_speed")


def _segment_label(row) -> str:
    """Build a multi-line HTML tooltip header for a street segment."""
    name = _first_present(row, _NAME_COLS)
    place = _first_present(row, _PLACE_COLS, default="")
    parts = [name]
    if place:
        parts.append(place)
    return "<br>".join(parts)


def _make_base_map(center: tuple[float, float] = _LEONIA_CENTER) -> folium.Map:
    return folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )


def _quantile_breaks(values: np.ndarray, n: int = 6) -> list[float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return [0.0, 1.0]
    qs = np.linspace(0, 1, n + 1)
    breaks = np.unique(np.quantile(finite, qs)).tolist()
    if len(breaks) < 2:
        breaks = [float(finite.min()), float(finite.max()) + 1.0]
    return breaks


def volume_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    label: str,
    *,
    min_volume: float = 0.0,
    line_weight: tuple[float, float] = (1.0, 6.0),
) -> folium.Map:
    """Render a per-segment volume choropleth as line widths + colors."""
    g = gdf.loc[gdf[value_col].fillna(0) >= min_volume].copy()
    values = g[value_col].fillna(0).to_numpy()

    breaks = _quantile_breaks(values)
    colormap = cm.linear.YlOrRd_09.scale(min(breaks), max(breaks))
    colormap.caption = label

    vmin, vmax = float(min(values, default=1)), float(max(values, default=1))
    span = max(vmax - vmin, 1.0)
    wmin, wmax = line_weight

    fmap = _make_base_map()

    for _, row in g.iterrows():
        v = float(row[value_col]) if row[value_col] is not None else 0.0
        if not np.isfinite(v):
            continue
        weight = wmin + (wmax - wmin) * ((v - vmin) / span)
        color = colormap(v)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            coord_lists = [list(ls.coords) for ls in geom.geoms]
        else:
            coord_lists = [list(geom.coords)]
        for coords in coord_lists:
            latlon = [(y, x) for x, y in coords]
            speed_obs = _first_present(row, _SPEED_OBS_COLS, default="")
            speed_lim = _first_present(row, _SPEED_LIMIT_COLS, default="")
            speed_line = (
                f"<br>Speed: {speed_obs} / {speed_lim} mph"
                if speed_obs or speed_lim else ""
            )
            folium.PolyLine(
                locations=latlon,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=folium.Tooltip(
                    f"{_segment_label(row)}<br>"
                    f"{label}: {v:,.0f}"
                    f"{speed_line}",
                    sticky=True,
                ),
            ).add_to(fmap)

    fmap.add_child(colormap)
    return fmap


def ratio_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    label: str,
    *,
    midpoint: float = 1.0,
    line_weight: float = 3.0,
) -> folium.Map:
    """Render a per-segment ratio as a diverging colormap centered on ``midpoint``."""
    g = gdf.loc[gdf[value_col].notna()].copy()
    values = g[value_col].to_numpy()

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return _make_base_map()

    vmin = float(np.quantile(finite, 0.05))
    vmax = float(np.quantile(finite, 0.95))
    half = max(midpoint - vmin, vmax - midpoint, 0.1)
    colormap = cm.LinearColormap(
        colors=["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
        vmin=midpoint - half,
        vmax=midpoint + half,
        caption=label,
    )

    fmap = _make_base_map()
    for _, row in g.iterrows():
        v = float(row[value_col])
        if not np.isfinite(v):
            continue
        color = colormap(v)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            coord_lists = [list(ls.coords) for ls in geom.geoms]
        else:
            coord_lists = [list(geom.coords)]
        for coords in coord_lists:
            latlon = [(y, x) for x, y in coords]
            folium.PolyLine(
                locations=latlon,
                color=color,
                weight=line_weight,
                opacity=0.85,
                tooltip=folium.Tooltip(
                    f"{_segment_label(row)}<br>"
                    f"{label}: {v:.2f}",
                    sticky=True,
                ),
            ).add_to(fmap)

    fmap.add_child(colormap)
    return fmap


def od_flow_map(
    origin_gdf: gpd.GeoDataFrame,
    destination_gdf: gpd.GeoDataFrame,
    flows: list[dict],
    *,
    label: str = "OD volume",
) -> folium.Map:
    """Draw weighted origin→destination flow lines on top of the zone gates.

    Parameters
    ----------
    origin_gdf
        GeoDataFrame of origin gate geometries (must include ``name``).
    destination_gdf
        GeoDataFrame of destination gate geometries (must include ``name``).
    flows
        List of dicts with keys ``origin``, ``destination``, ``volume``,
        ``label`` (tooltip text).
    """
    fmap = _make_base_map()

    def _centroid_latlon(geom) -> tuple[float, float] | None:
        if geom is None or geom.is_empty:
            return None
        c = geom.centroid
        return (c.y, c.x)

    origin_index = {row["name"]: _centroid_latlon(row.geometry) for _, row in origin_gdf.iterrows()}
    dest_index = {row["name"]: _centroid_latlon(row.geometry) for _, row in destination_gdf.iterrows()}

    for _, row in origin_gdf.iterrows():
        ll = _centroid_latlon(row.geometry)
        if ll is None:
            continue
        folium.CircleMarker(
            location=ll, radius=6, color="#1a9850", fill=True,
            tooltip=f"Origin: {row.get('name', '?')}",
        ).add_to(fmap)

    for _, row in destination_gdf.iterrows():
        ll = _centroid_latlon(row.geometry)
        if ll is None:
            continue
        folium.CircleMarker(
            location=ll, radius=8, color="#b2182b", fill=True,
            tooltip=f"Destination: {row.get('name', '?')}",
        ).add_to(fmap)

    if flows:
        max_vol = max((f.get("volume", 0) for f in flows), default=1)
        max_vol = max(max_vol, 1)
        for f in flows:
            o = origin_index.get(f["origin"])
            d = dest_index.get(f["destination"])
            v = f.get("volume", 0) or 0
            if o is None or d is None or v <= 0:
                continue
            weight = 1 + 7 * (v / max_vol)
            folium.PolyLine(
                locations=[o, d],
                color="#762a83",
                weight=weight,
                opacity=0.7,
                tooltip=f.get("label", f"{f['origin']} → {f['destination']}: {v:.0f}"),
            ).add_to(fmap)

    return fmap


def tti_map(gdf: gpd.GeoDataFrame, *, label: str = "Travel Time Index (worst hour)") -> folium.Map:
    """Render per-segment congestion TTI as colored line widths.

    ``gdf`` must include the ``worst_tti`` column (e.g. the output of
    :func:`leonia_traffic.data.congestion_loader.summarize_link_reliability`
    joined to its zone geometry).
    """
    g = gdf.loc[gdf["worst_tti"].notna()].copy()
    values = g["worst_tti"].to_numpy()

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return _make_base_map()

    vmin = max(1.0, float(np.quantile(finite, 0.05)))
    vmax = float(np.quantile(finite, 0.95))
    colormap = cm.linear.YlOrRd_09.scale(vmin, max(vmax, vmin + 0.1))
    colormap.caption = label

    fmap = _make_base_map()
    for _, row in g.iterrows():
        v = float(row["worst_tti"])
        if not np.isfinite(v):
            continue
        color = colormap(v)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            coord_lists = [list(ls.coords) for ls in geom.geoms]
        else:
            coord_lists = [list(geom.coords)]
        for coords in coord_lists:
            latlon = [(y, x) for x, y in coords]
            folium.PolyLine(
                locations=latlon,
                color=color,
                weight=4.5,
                opacity=0.85,
                tooltip=folium.Tooltip(
                    f"{_segment_label(row)} ({_first_present(row, ('road_class',), default='?')})<br>"
                    f"Worst-hour TTI: {v:.2f}<br>"
                    f"Worst Buffer Idx: {row.get('worst_buffer', float('nan')):.2f}<br>"
                    f"Weekday VHD: {row.get('total_weekday_vhd', 0):.1f}<br>"
                    f"Reliability: {row.get('reliability_class', 'Unknown')}",
                    sticky=True,
                ),
            ).add_to(fmap)

    fmap.add_child(colormap)
    return fmap


def reliability_map(gdf: gpd.GeoDataFrame, *, label: str = "Reliability class") -> folium.Map:
    """Render per-segment reliability classification with discrete colors.

    ``gdf`` must include the ``reliability_class`` column with values
    ``Reliable``, ``Moderate``, ``Unreliable``, ``Unknown``.
    """
    color_map = {
        "Reliable": "#1a9850",
        "Moderate": "#fdae61",
        "Unreliable": "#d73027",
        "Unknown": "#999999",
    }
    fmap = _make_base_map()
    for _, row in gdf.iterrows():
        cls = row.get("reliability_class", "Unknown")
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            coord_lists = [list(ls.coords) for ls in geom.geoms]
        else:
            coord_lists = [list(geom.coords)]
        for coords in coord_lists:
            latlon = [(y, x) for x, y in coords]
            folium.PolyLine(
                locations=latlon,
                color=color_map.get(cls, "#999999"),
                weight=4.5,
                opacity=0.85,
                tooltip=folium.Tooltip(
                    f"{_segment_label(row)} ({_first_present(row, ('road_class',), default='?')})<br>"
                    f"Reliability: {cls} (LOTTR={row.get('worst_lottr', float('nan')):.2f})<br>"
                    f"Worst TTI: {row.get('worst_tti', float('nan')):.2f}",
                    sticky=True,
                ),
            ).add_to(fmap)
    return fmap


__all__ = ["volume_map", "ratio_map", "od_flow_map", "tti_map", "reliability_map"]
