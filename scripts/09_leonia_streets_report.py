"""Pass C.1: per-residential-street cut-through evidence report.

Run from the repo root:

    venv/bin/python scripts/09_leonia_streets_report.py

Produces:

    reports/09_leonia_streets.md
    reports/figures/street_cutthrough_topN.png
    reports/figures/street_origin_municipalities.png
    reports/figures/street_trip_length_distribution.png
    reports/maps/street_cutthrough_index.html
    reports/maps/street_visitor_volume.html
    reports/maps/street_speeding.html

The report is the primary stakeholder deliverable for Pass C. Pass C.2
(recommendations) and Pass C.3 (simulation calibration) consume the
same per-street tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leonia_traffic.analysis import cutthrough_streets as cs
from leonia_traffic.analysis import visitor_demographics as vd
from leonia_traffic.analysis.jurisdiction import filter_segments_to_leonia
from leonia_traffic.config import DATA_STAGE2_DIR, REPORTS_DIR, REPORTS_FIG_DIR
from leonia_traffic.data import za_streets_loader as zl
from leonia_traffic.data.dataset_io import DERIVED_DIR, DerivedFiles
from leonia_traffic.viz.maps import volume_map

REPORTS_MAPS_DIR = REPORTS_DIR / "maps"
REPORTS_MAPS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATA_STAGE2_DIR
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Formatting helpers (small subset of those in 07_bridge_od_report.py)
# ---------------------------------------------------------------------------


def _fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return ""


def _fmt_float(x, digits: int = 2) -> str:
    try:
        f = float(x)
        if not np.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _fmt_pct(x, digits: int = 1) -> str:
    try:
        f = float(x)
        if not np.isfinite(f):
            return ""
        return f"{f * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return ""


_FMT_MAP = {
    "int": _fmt_int,
    "float2": lambda v: _fmt_float(v, 2),
    "float3": lambda v: _fmt_float(v, 3),
    "pct": _fmt_pct,
    "str": lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v),
}


def _table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    headers = [h for _, h, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col, _, fmt in columns:
            v = row.get(col) if col in row else None
            cells.append(_FMT_MAP.get(fmt, _FMT_MAP["str"])(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figure_top_n_bar(index_df: pd.DataFrame, fig_path: Path, n: int = 20) -> None:
    if index_df.empty:
        return
    sub = index_df.head(n).iloc[::-1].copy()  # ascending for hbar
    labels = [f"{s} ({int(o) if pd.notna(o) else '-'})"
              for s, o in zip(sub["street_name"], sub["osm_way_id"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(sub)), sub["cutthrough_index"].astype(float), color="#a50026")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Composite cut-through index (0..1)")
    ax.set_title(f"Top {n} Leonia residential streets by composite cut-through index")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _figure_origin_municipalities(
    home_zips_top: pd.DataFrame,
    fig_path: Path,
    *,
    top_n: int = 15,
) -> None:
    df = vd.origin_municipality_breakdown(home_zips_top, zone_name=None)
    if df.empty:
        return
    df = df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(df)), df["share"].astype(float) * 100.0, color="#3182bd")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["municipality"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of weighted Visitor trips (%)")
    ax.set_title("Where Leonia's residential-street pass-through traffic comes from")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _figure_destination_counties(
    work_bg_df: pd.DataFrame,
    fig_path: Path,
    *,
    top_n: int = 15,
) -> None:
    df = vd.work_destination_breakdown(work_bg_df, zone_name=None, top_n=top_n)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(df)), df["share"].astype(float) * 100.0, color="#e6550d")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["county_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of weighted Visitor trips (%)")
    ax.set_title("Where Leonia's residential-street pass-through traffic is going")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


_LEN_BIN_ORDER = (
    ("len_mi_0_1", "0-1"),
    ("len_mi_1_2", "1-2"),
    ("len_mi_2_5", "2-5"),
    ("len_mi_5_10", "5-10"),
    ("len_mi_10_20", "10-20"),
    ("len_mi_20_30", "20-30"),
    ("len_mi_30_40", "30-40"),
    ("len_mi_40_50", "40-50"),
    ("len_mi_50_60", "50-60"),
    ("len_mi_60_70", "60-70"),
    ("len_mi_70_80", "70-80"),
    ("len_mi_80_90", "80-90"),
    ("len_mi_90_100", "90-100"),
    ("len_mi_100_plus", "100+"),
)


def _figure_hourly_profiles(
    hourly_df: pd.DataFrame,
    index_df: pd.DataFrame,
    fig_path: Path,
    *,
    n: int = 8,
) -> None:
    """Small-multiple hourly Visitor-volume curves for the top-N
    cut-through streets, with AM-peak (7-10am) and PM-peak (4-7pm)
    windows shaded for context."""
    if hourly_df is None or hourly_df.empty or index_df is None or index_df.empty:
        return
    hour_cols = [f"h{h:02d}" for h in range(24)]
    present = [c for c in hour_cols if c in hourly_df.columns]
    if not present:
        return
    top = index_df.head(n)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.2 * rows),
                             sharex=True, sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, (_, r) in enumerate(top.iterrows()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        sub = hourly_df[hourly_df["osm_way_id"] == r["osm_way_id"]]
        if sub.empty:
            ax.set_visible(False)
            continue
        y = sub.iloc[0][present].fillna(0).astype(float).values
        x = [int(c[1:]) for c in present]
        ax.axvspan(7, 10, color="#fee08b", alpha=0.45, label="AM peak")
        ax.axvspan(16, 19, color="#fdae61", alpha=0.45, label="PM peak")
        ax.plot(x, y, color="#08519c", linewidth=1.6)
        ax.fill_between(x, 0, y, color="#08519c", alpha=0.15)
        ax.set_title(
            f"{r['street_name']} ({int(r['osm_way_id'])})", fontsize=9,
        )
        ax.set_ylabel("trips/hr", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(0, 23)
    for ax in axes_flat[len(top):]:
        ax.set_visible(False)
    axes_flat[-1].set_xlabel("Hour of day", fontsize=9)
    if rows > 1:
        axes_flat[-2].set_xlabel("Hour of day", fontsize=9)
    fig.suptitle(
        "Hourly Visitor volume (All Days) — top cut-through residential streets",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _figure_trip_length_smallmultiples(
    trip_df: pd.DataFrame,
    index_df: pd.DataFrame,
    fig_path: Path,
    *,
    n: int = 10,
) -> None:
    if trip_df is None or trip_df.empty or index_df is None or index_df.empty:
        return
    top = index_df.head(n)
    trip = trip_df[(trip_df["filter"] == "Visitors")
                   & (trip_df["day_type_code"] == cs.ALL_DAYS_TYPE)
                   & (trip_df["day_part_code"] == cs.ALL_DAY_PART)]
    n_rows = (n + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.4 * n_rows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    bin_cols = [c for c, _ in _LEN_BIN_ORDER if c in trip.columns]
    labels = [lbl for c, lbl in _LEN_BIN_ORDER if c in trip.columns]
    for i, (_, r) in enumerate(top.iterrows()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        sub = trip[trip["zone_name"] == r["zone_name"]]
        if sub.empty:
            ax.set_visible(False)
            continue
        vals = sub.iloc[0][bin_cols].fillna(0).astype(float).values * 100.0
        ax.bar(range(len(bin_cols)), vals, color="#08519c")
        ax.set_title(f"{r['street_name']} ({int(r['osm_way_id'])})", fontsize=9)
        ax.set_ylabel("% of trips", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes_flat[len(top):]:
        ax.set_visible(False)
    axes_flat[-1].set_xticks(range(len(labels)))
    axes_flat[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    if n_rows > 1:
        axes_flat[-2].set_xticks(range(len(labels)))
        axes_flat[-2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    fig.suptitle("Trip-length distribution for top-10 cut-through residential streets")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def _build_maps(
    line_gdf,
    index_df: pd.DataFrame,
    speeding_df: pd.DataFrame,
    weekday_vol_df: pd.DataFrame,
):
    """Return three folium maps: index, volume, speeding. May return None
    if the line shapefile is missing.
    """
    if line_gdf is None or line_gdf.empty:
        return None, None, None
    g = line_gdf.copy()
    g["osm_way_id"] = g["osm_way_id"].astype("Int64")

    merged_idx = g.merge(
        index_df[["osm_way_id", "cutthrough_index", "street_name"]].drop_duplicates(
            subset=["osm_way_id"]
        ),
        on="osm_way_id",
        how="left",
        suffixes=("", "_idx"),
    )
    map_idx = volume_map(
        merged_idx.dropna(subset=["cutthrough_index"]),
        "cutthrough_index",
        "Composite cut-through index",
        line_weight=(1.5, 7.0),
    )

    merged_vol = g.merge(
        weekday_vol_df[["osm_way_id", "weekday_all_day_volume"]].drop_duplicates(
            subset=["osm_way_id"]
        ),
        on="osm_way_id",
        how="left",
    )
    map_vol = volume_map(
        merged_vol.dropna(subset=["weekday_all_day_volume"]),
        "weekday_all_day_volume",
        "Visitor weekday daily volume",
        line_weight=(1.5, 7.0),
    )

    merged_speed = g.merge(
        speeding_df[["osm_way_id", "speeding_share"]].drop_duplicates(
            subset=["osm_way_id"]
        ),
        on="osm_way_id",
        how="left",
    )
    map_speed = volume_map(
        merged_speed.dropna(subset=["speeding_share"]),
        "speeding_share",
        "Share of Visitor trips above posted-speed bin",
        line_weight=(1.5, 7.0),
    )
    return map_idx, map_vol, map_speed


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _scope_block() -> str:
    return (
        "> **Scope and jurisdiction.** This report covers OSM tertiary "
        "segments inside the Borough of Leonia, measured by StreetLight as "
        "*pass-through* trips by drivers whose home block-group is outside "
        "the analysis zone set (Visitors). State and federal facilities "
        "crossing the borough (NJ Turnpike, GWB approaches, US 1/9/46, NJ 4) "
        "are excluded from recommendations even when they appear in the "
        "data, because Leonia has no authority over them.\n\n"
    )


def _headline_section(index_df: pd.DataFrame, total_visitor_volume: float) -> str:
    if index_df.empty:
        return "_No cut-through data available._\n\n"
    top = index_df.iloc[0]
    second = index_df.iloc[1] if len(index_df) > 1 else None
    second_clause = ""
    if second is not None:
        second_clause = (
            f" The next-ranked street is **{second['street_name']}** "
            f"(index {second['cutthrough_index']:.2f})."
        )
    return (
        "**Headline finding (Pass C.1):** Across **"
        f"{int(index_df['osm_way_id'].nunique()):,} OSM tertiary segments** "
        f"inside Leonia, an average of **{int(round(total_visitor_volume)):,} "
        "pass-through Visitor trips/day** are made by non-resident drivers. "
        f"The corridor with the highest composite cut-through index is "
        f"**{top['street_name']}** (OSM way `{int(top['osm_way_id'])}`) at "
        f"**{top['cutthrough_index']:.2f}** (0..1 scale, 1 = worst on every "
        f"sub-metric)." + second_clause + "\n\n"
    )


def _ranking_section(index_df: pd.DataFrame, top_n: int = 20) -> str:
    md = "## Top cut-through streets\n\n"
    if index_df.empty:
        return md + "_No data._\n\n"
    md += (
        "The composite index combines weekday/weekend volume imbalance, "
        "non-local home share, the share of pass-through trips that are "
        ">5 mi (a cut-through signature, since local-only traffic is "
        "<2 mi), the share of trips in speed bins above the posted "
        "residential limit, and overall daily volume. Each sub-metric "
        "is min-max normalised before weighting.\n\n"
    )
    md += _table(index_df.head(top_n), [
        ("rank", "#", "int"),
        ("street_name", "Street", "str"),
        ("osm_way_id", "OSM way", "int"),
        ("cutthrough_index", "Composite index", "float3"),
        ("weekday_all_day_volume", "Avg weekday vol.", "int"),
        ("weekday_weekend_ratio", "Wkdy / Sat ratio", "float2"),
        ("non_local_home_share", "Non-local home", "pct"),
        ("long_trip_share_5mi", "Trips >5mi", "pct"),
        ("speeding_share", "Speeding bins", "pct"),
    ])
    md += "\n![Top 20 ranking](figures/street_cutthrough_topN.png)\n\n"
    return md


def _per_street_pages(
    index_df: pd.DataFrame,
    home_zips_top: pd.DataFrame,
    work_bg_df: pd.DataFrame,
    n: int = 10,
) -> str:
    md = "## Per-street profiles (top 10)\n\n"
    if index_df.empty:
        return md + "_No data._\n\n"
    for _, row in index_df.head(n).iterrows():
        zone = row["zone_name"]
        md += (
            f"### {row['street_name']} (OSM way {int(row['osm_way_id'])}) "
            f"— rank #{int(row['rank'])}\n\n"
        )
        md += (
            f"- Composite cut-through index: **{row['cutthrough_index']:.3f}**\n"
            f"- Weekday all-day Visitor volume: **{_fmt_int(row.get('weekday_all_day_volume'))}** trips/day\n"
            f"- Thursday / Saturday ratio: **{_fmt_float(row.get('weekday_weekend_ratio'))}×**\n"
            f"- Non-local home share (≥3 mi): **{_fmt_pct(row.get('non_local_home_share'))}**\n"
            f"- Trips with length >5 mi: **{_fmt_pct(row.get('long_trip_share_5mi'))}**, "
            f">10 mi: **{_fmt_pct(row.get('long_trip_share_10mi'))}**\n"
            f"- Speeding-bin share (≥25 mph): **{_fmt_pct(row.get('speeding_share'))}**\n"
        )
        muni = vd.origin_municipality_breakdown(home_zips_top, zone)
        if not muni.empty:
            top_muni = muni.head(5)
            md += "\n  Top 5 home municipalities (Visitor share):\n\n"
            for _, m in top_muni.iterrows():
                md += f"  - {m['municipality']}: {_fmt_pct(m['share'])}\n"
        dest = vd.work_destination_breakdown(work_bg_df, zone, top_n=5)
        if not dest.empty:
            md += "\n  Top 5 work destinations (Visitor share, by county):\n\n"
            for _, d in dest.iterrows():
                md += f"  - {d['county_label']}: {_fmt_pct(d['share'])}\n"
        md += "\n"
    md += "![Trip-length distributions](figures/street_trip_length_distribution.png)\n\n"
    return md


def _peak_hours_section(
    index_df: pd.DataFrame,
    peak_am_df: pd.DataFrame,
    peak_pm_df: pd.DataFrame,
    intensity_am_df: pd.DataFrame,
    intensity_pm_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> str:
    """Peak-hour deep dive: AM peak (7-10am), PM peak (4-7pm), and
    peak-vs-midday intensity ratios. StreetLight only publishes an
    hourly breakdown for a subset of the highest-volume zones, so
    these tables cover fewer rows than the headline ranking."""
    md = "## Peak-hour deep dive\n\n"
    md += (
        "StreetLight reports volume in 1-hour day-parts, but only for "
        "zones with enough sample to support hourly disaggregation. "
        "The tables below use the **All-Days** day-type aggregate for "
        "the widest possible coverage, then restrict to the **top "
        f"{top_n}** Leonia residential segments by composite cut-through "
        "index that have hourly data available.\n\n"
        "- **Peak AM** = 7:00–10:00 AM (StreetLight day-parts 8-10)\n"
        "- **Peak PM** = 4:00–7:00 PM (day-parts 17-19)\n"
        "- **Midday baseline** = 11:00 AM – 2:00 PM (day-parts 12-14), "
        "used as the off-peak comparison\n"
        "- **Peak intensity** = mean peak-hour rate ÷ mean midday-hour "
        "rate. A ratio of 1.0 means the street has flat traffic "
        "(local-errand pattern); 3× or higher is a strong "
        "commuter-cut-through signature.\n\n"
    )

    top_keys = index_df[["osm_way_id", "street_name", "rank",
                         "cutthrough_index"]]

    def _peak_table(
        vol_df: pd.DataFrame,
        intensity_df: pd.DataFrame,
        vol_col: str,
        peak_label: str,
    ) -> str:
        if vol_df is None or vol_df.empty:
            return f"_No {peak_label} peak-hour data available._\n\n"
        sub = top_keys.merge(
            vol_df[["osm_way_id", vol_col]],
            on="osm_way_id", how="left",
        )
        if intensity_df is not None and not intensity_df.empty:
            sub = sub.merge(
                intensity_df[["osm_way_id", "peak_per_hr",
                              "baseline_per_hr", "peak_intensity"]],
                on="osm_way_id", how="left",
            )
        else:
            sub["peak_per_hr"] = float("nan")
            sub["baseline_per_hr"] = float("nan")
            sub["peak_intensity"] = float("nan")
        sub = sub.dropna(subset=[vol_col])
        if sub.empty:
            return f"_No {peak_label} hourly data for the top-ranked streets._\n\n"
        sub = sub.sort_values(vol_col, ascending=False).head(top_n).reset_index(drop=True)
        cols = [
            ("rank", "Cut-through #", "int"),
            ("street_name", "Street", "str"),
            ("osm_way_id", "OSM way", "int"),
            (vol_col, f"{peak_label} 3-hr vol.", "int"),
            ("peak_per_hr", f"{peak_label} trips/hr", "float2"),
            ("baseline_per_hr", "Midday trips/hr", "float2"),
            ("peak_intensity", "Peak ÷ midday", "float2"),
        ]
        return _table(sub, cols)

    md += f"### Peak AM (7–10 AM) — top {top_n} by peak-AM volume\n\n"
    md += _peak_table(peak_am_df, intensity_am_df, "peak_am_volume", "AM")
    md += (
        "\n_Read this as: which Leonia residential blocks carry the "
        "most pass-through traffic during the morning commute, and "
        "how much sharper that signal is than midday._\n\n"
    )

    md += f"### Peak PM (4–7 PM) — top {top_n} by peak-PM volume\n\n"
    md += _peak_table(peak_pm_df, intensity_pm_df, "peak_pm_volume", "PM")
    md += (
        "\n_PM-peak volumes are systematically higher than AM-peak on "
        "Leonia residential streets (longer evening rush window plus "
        "school pick-up). A street that's high on both lists is "
        "carrying bidirectional commuter traffic._\n\n"
    )

    # Combined intensity leaderboard — streets where the *ratio* is
    # high, regardless of absolute volume, are the clearest examples
    # of commuter cut-through.
    if intensity_am_df is not None and not intensity_am_df.empty:
        am_subset = (
            top_keys
            .merge(
                intensity_am_df[["osm_way_id", "peak_intensity", "peak_per_hr"]],
                on="osm_way_id", how="left",
            )
            .rename(columns={
                "peak_intensity": "am_intensity",
                "peak_per_hr": "am_per_hr",
            })
        )
    else:
        am_subset = top_keys.assign(am_intensity=float("nan"), am_per_hr=float("nan"))
    if intensity_pm_df is not None and not intensity_pm_df.empty:
        am_subset = am_subset.merge(
            intensity_pm_df[["osm_way_id", "peak_intensity", "peak_per_hr"]],
            on="osm_way_id", how="left",
        ).rename(columns={
            "peak_intensity": "pm_intensity",
            "peak_per_hr": "pm_per_hr",
        })
    else:
        am_subset["pm_intensity"] = float("nan")
        am_subset["pm_per_hr"] = float("nan")
    am_subset = am_subset.dropna(subset=["am_intensity", "pm_intensity"], how="all")
    if not am_subset.empty:
        am_subset["max_intensity"] = am_subset[
            ["am_intensity", "pm_intensity"]
        ].max(axis=1)
        am_subset = am_subset.sort_values(
            "max_intensity", ascending=False
        ).head(top_n).reset_index(drop=True)
        md += "### Streets with the sharpest peak-vs-midday spike\n\n"
        md += _table(am_subset, [
            ("rank", "Cut-through #", "int"),
            ("street_name", "Street", "str"),
            ("osm_way_id", "OSM way", "int"),
            ("am_per_hr", "AM peak trips/hr", "float2"),
            ("am_intensity", "AM ÷ midday", "float2"),
            ("pm_per_hr", "PM peak trips/hr", "float2"),
            ("pm_intensity", "PM ÷ midday", "float2"),
        ])
        md += (
            "\nStreets near the top of this table behave like commuter "
            "shortcuts: very quiet at midday, but bursting in the "
            "AM or PM peak. Speed humps, traffic calming, or "
            "peak-period turn restrictions on these blocks would have "
            "the largest relative effect on residents' lived "
            "experience.\n\n"
        )
    else:
        md += "_Insufficient hourly data to compute peak intensity._\n\n"

    if hourly_df is not None and not hourly_df.empty:
        md += "![Hourly profiles](figures/street_hourly_profiles.png)\n\n"

    md += (
        "**Coverage caveat.** Only "
        f"{int(hourly_df['osm_way_id'].nunique()) if hourly_df is not None and not hourly_df.empty else 0} "
        "of the residential segments in the Pass-C export have an hourly "
        "breakdown (the rest only have All-Day totals). Segments missing "
        "from the tables above are not zero — they simply do not have "
        "enough sampled trips in any one hour for StreetLight to publish "
        "an hourly figure.\n\n"
    )
    return md


def _origin_section(home_zips_top: pd.DataFrame) -> str:
    md = "## Where the cut-through is coming from\n\n"
    df = vd.origin_municipality_breakdown(home_zips_top, zone_name=None)
    if df.empty:
        return md + "_No home-ZIP data available._\n\n"
    md += (
        "Volume-weighted share of pass-through Visitor trips across all "
        "Leonia residential streets, grouped by the driver's home "
        "municipality (ZIP-derived):\n\n"
    )
    md += _table(df.head(15), [
        ("municipality", "Home municipality", "str"),
        ("share", "Share of weighted Visitor trips", "pct"),
    ])
    md += "\n![Origin municipalities](figures/street_origin_municipalities.png)\n\n"
    return md


def _destination_section(work_bg_df: pd.DataFrame) -> str:
    md = "## Where the cut-through is going\n\n"
    if work_bg_df is None or work_bg_df.empty:
        return md + "_No work-block-group data available._\n\n"
    df = vd.work_destination_breakdown(work_bg_df, zone_name=None, top_n=15)
    if df.empty:
        return md + "_No destination data available._\n\n"
    md += (
        "Volume-weighted share of pass-through Visitor trips across all "
        "Leonia residential streets, grouped by the driver's *workplace* "
        "county (derived from the StreetLight work-block-group "
        "cross-tab). StreetLight does not expose an explicit trip "
        "destination, but for the AM-peak cut-through pattern the "
        "workplace location is a strong proxy for where the trip is "
        "headed.\n\n"
    )
    md += _table(df, [
        ("county_label", "Work destination county", "str"),
        ("share", "Share of weighted Visitor trips", "pct"),
    ])
    md += "\n![Destination counties](figures/street_destination_counties.png)\n\n"
    return md


def _coverage_section(line_gdf) -> str:
    md = "## Coverage update\n\n"
    n_segments = len(line_gdf) if line_gdf is not None else 0
    n_streets = (
        line_gdf["street_name"].nunique() if line_gdf is not None and not line_gdf.empty else 0
    )
    md += (
        f"This export covers **{n_segments} OSM tertiary segments** across "
        f"**{n_streets} unique street names** inside (or immediately touching) "
        "Leonia. The previous Congestion Trends export covered 7 named "
        "in-borough streets only — Pass C closes that gap. Residential "
        "blocks (Christie Heights, Hillside, Glenwood, Park, Highwood, "
        "Schor, Willow Tree, Crescent, Walnut, Birch, etc.) now have "
        "direct pass-through measurements.\n\n"
        "The residential-coverage caveat at the bottom of "
        "`reports/07_bridge_od.md` should be read alongside this report.\n\n"
    )
    return md


def _limitations_section() -> str:
    return (
        "## Notes on data limitations\n\n"
        "* **Visitor definition** — StreetLight tags a trip as Visitor "
        "when the driver's home block-group is outside the analysis zone "
        "set. A non-resident commuting *to* a Leonia destination is "
        "therefore still a Visitor, even if the trip doesn't continue "
        "through. The composite index weights `long_trip_share_5mi` "
        "precisely to distinguish through-trips from short destination "
        "trips.\n"
        "* **Speed bins are coarse** (10-mph wide). The `speeding_share` "
        "for posted ≤25 mph uses the 20-30 mph bin as the lower bound, "
        "which slightly over-states speeding; the report exposes the raw "
        "trip-speed distribution per street so readers can inspect "
        "individual streets.\n"
        "* **Day-type \"weekday\" excludes Friday** — the export's day "
        "types are All / Mon / Tue / Wed / Thu / Sat / Sun. We use Thu as "
        "the canonical weekday day and Mon-Thu for Peak-AM aggregations "
        "to stay consistent with Pass A.\n"
        "* **ZIP-only home location** — Pass C uses StreetLight's "
        "pre-ranked top home ZIPs (no ACS join). The static ZIP→"
        "municipality lookup in `analysis/visitor_demographics.py` covers "
        "the top ~40 ZIPs; unmatched ZIPs fall into `Other NJ`, `Other "
        "NY`, or `Other`.\n"
        "* **Destination = workplace, not trip-end** — the Zone Activity "
        "product does not expose trip destination directly. The "
        "destination columns in this report come from StreetLight's "
        "work-block-group cross-tab, which characterises the driver's "
        "workplace location. For AM-peak commuter cut-through that is "
        "the right answer; for non-work pass-through (errands, evening "
        "trips) the workplace is a less reliable proxy and the "
        "destination story should be read with that caveat.\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading data...")
    za = zl.load_za_main()
    trip = zl.load_za_trip()
    home_dist = zl.load_za_home_distance()
    home_zips_top = zl.load_za_home_zips_top()
    work_bg = zl.load_za_work_block_groups()
    line_gdf = zl.load_za_line_shapes()

    if za.empty:
        raise SystemExit(
            "Leonia-streets ZA export not found at "
            "streetlight/2034227_leonia_streets/."
        )

    print(f"ZA main rows: {len(za):,}")
    print(f"Trip rows: {len(trip):,}")
    print(f"Home-distance rows: {len(home_dist):,}")
    print(f"Home-zip-top rows: {len(home_zips_top):,}")
    print(f"Work block-group rows: {len(work_bg):,}")
    print(f"Line shapes: {len(line_gdf):,}")

    print("Computing per-street analytics...")
    imbalance = cs.weekday_weekend_imbalance(za)
    weekday_vol = cs.weekday_all_day_volume(za)
    peak_am = cs.peak_am_volume(za)
    peak_pm = cs.peak_pm_volume(za)
    # All-Days hourly aggregates have meaningfully broader StreetLight
    # coverage than Mon-Thu for the peak/midday ratio.
    intensity_am = cs.peak_hour_intensity(
        za, peak_hours=cs.PEAK_AM_HOURS, day_types=(cs.ALL_DAYS_TYPE,),
    )
    intensity_pm = cs.peak_hour_intensity(
        za, peak_hours=cs.PEAK_PM_HOURS, day_types=(cs.ALL_DAYS_TYPE,),
    )
    hourly_profile = cs.weekday_hourly_profile(
        za, day_types=(cs.ALL_DAYS_TYPE,),
    )
    long_trip = cs.long_trip_share(trip)
    speeding = cs.speeding_share(trip)
    home_share = cs.non_local_home_share(home_dist)
    leonia_zip_share = cs.non_leonia_zip_share(home_zips_top)

    print("Filtering to Leonia municipal residential streets...")
    # Drop segments whose "name" is actually an OSM highway tag for a
    # state/federal facility or a non-thoroughfare class. These names
    # appear when the OSM way had no `name` tag: ``motorway_link``,
    # ``trunk_link``, ``primary_link``, ``service`` (driveways/parking
    # connectors), ``tertiary`` (unnamed tertiaries which are not
    # signposted residential streets).
    EXCLUDE_NAME_TAGS = {
        "motorway_link", "trunk_link", "primary_link",
        "secondary_link", "tertiary_link",
        "service", "track", "unclassified",
        # Unnamed tertiary segments — these are real residential blocks
        # but lack an OSM `name` tag, so we cannot cite them to
        # stakeholders by street name. Drop from recommendation tables.
        "tertiary",
    }
    pre_drop = len(imbalance)
    for df in (imbalance, weekday_vol, peak_am, peak_pm,
               intensity_am, intensity_pm, hourly_profile,
               long_trip, speeding, home_share, leonia_zip_share):
        if df is not None and not df.empty and "street_name" in df.columns:
            df.drop(
                df.index[df["street_name"].isin(EXCLUDE_NAME_TAGS)],
                inplace=True,
            )
    print(
        f"  Non-residential OSM tags dropped: ~{pre_drop - len(imbalance)} rows."
    )

    # Cut-through ranking: reuse the jurisdiction-filtered, ranked composite
    # index that scripts/00_build_datasets.build_derived already writes
    # (DerivedFiles.cutthrough_index) rather than recomputing it here. Fall
    # back to an in-line recompute if the derived lake has not been built yet
    # so this report still runs standalone.
    derived_index_path = DERIVED_DIR / DerivedFiles.cutthrough_index
    if derived_index_path.exists():
        index_df_leonia = pd.read_parquet(derived_index_path)
        print(
            f"  Loaded cut-through index from {derived_index_path.name} "
            f"({len(index_df_leonia)} in-borough segments)."
        )
    else:
        index_df = cs.composite_cutthrough_index(
            imbalance_df=imbalance,
            weekday_volume_df=weekday_vol,
            long_trip_df=long_trip,
            speeding_df=speeding,
            home_dist_df=home_share,
        )
        # Spatial filter: only segments whose geometry actually sits inside
        # the Leonia borough polygon. The earlier name-based drop already
        # removed state-road tags; the polygon filter catches any
        # tertiary-named segment that lies outside the borough.
        index_df_leonia = filter_segments_to_leonia(
            index_df.rename(columns={"street_name": "osm_name"}),
            line_gdf.rename(columns={"name": "zone_name"}),
        ).rename(columns={"osm_name": "street_name"})
        index_df_leonia = index_df_leonia.sort_values(
            "cutthrough_index", ascending=False
        ).reset_index(drop=True)
        index_df_leonia["rank"] = index_df_leonia.index + 1
        print(
            f"  Jurisdiction filter: {len(index_df)}\u2192{len(index_df_leonia)} "
            "in-borough municipal residential segments."
        )

    # Report-only augmentation: peak-AM volume + Leonia-ZIP origin shares.
    # (build_derived's table is the ranking; these columns are for the
    # report tables/figures only.)
    if not peak_am.empty:
        index_df_leonia = index_df_leonia.merge(
            peak_am, on=list(cs.KEY_COLUMNS), how="left",
        )
    if not leonia_zip_share.empty:
        index_df_leonia = index_df_leonia.merge(
            leonia_zip_share, on=list(cs.KEY_COLUMNS), how="left",
        )
    index_df_leonia = index_df_leonia.sort_values(
        "cutthrough_index", ascending=False
    ).reset_index(drop=True)
    index_df_leonia["rank"] = index_df_leonia.index + 1

    # Restrict the peak-hour dataframes to in-borough OSM ways so the
    # peak-hour section reflects the same jurisdictional scope as the
    # headline ranking.
    in_borough_ids = set(
        index_df_leonia["osm_way_id"].dropna().astype("Int64").astype(int).tolist()
    )

    def _restrict_to_borough(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "osm_way_id" not in df.columns:
            return df
        ids = df["osm_way_id"].astype("Int64")
        return df[ids.isin(in_borough_ids)].reset_index(drop=True)

    peak_am = _restrict_to_borough(peak_am)
    peak_pm = _restrict_to_borough(peak_pm)
    intensity_am = _restrict_to_borough(intensity_am)
    intensity_pm = _restrict_to_borough(intensity_pm)
    hourly_profile = _restrict_to_borough(hourly_profile)

    # Cache the per-street index for Pass C.2 / C.3.
    cache_path = PROCESSED_DIR / "leonia_streets_cutthrough_index.parquet"
    index_df_leonia.to_parquet(cache_path, index=False)
    print(f"  Cached per-street index to {cache_path}")

    print("Generating figures...")
    _figure_top_n_bar(index_df_leonia, REPORTS_FIG_DIR / "street_cutthrough_topN.png", n=20)
    _figure_origin_municipalities(home_zips_top, REPORTS_FIG_DIR / "street_origin_municipalities.png")
    _figure_destination_counties(work_bg, REPORTS_FIG_DIR / "street_destination_counties.png")
    _figure_trip_length_smallmultiples(
        trip, index_df_leonia,
        REPORTS_FIG_DIR / "street_trip_length_distribution.png",
        n=10,
    )
    _figure_hourly_profiles(
        hourly_profile, index_df_leonia,
        REPORTS_FIG_DIR / "street_hourly_profiles.png",
        n=8,
    )

    print("Generating maps...")
    map_idx, map_vol, map_speed = _build_maps(
        line_gdf, index_df_leonia, speeding, weekday_vol
    )
    if map_idx is not None:
        map_idx.save(str(REPORTS_MAPS_DIR / "street_cutthrough_index.html"))
        map_vol.save(str(REPORTS_MAPS_DIR / "street_visitor_volume.html"))
        map_speed.save(str(REPORTS_MAPS_DIR / "street_speeding.html"))

    print("Writing report...")
    total_visitor_volume = float(
        weekday_vol["weekday_all_day_volume"].sum()
    ) if not weekday_vol.empty else 0.0

    md_parts: list[str] = []
    md_parts.append("# Leonia residential streets: cut-through evidence (Pass C)\n\n")
    md_parts.append(
        "Generated by `scripts/09_leonia_streets_report.py`. Source: "
        "`streetlight/2034227_leonia_streets/` (Zone Activity on OSM "
        "Tertiary Segments, Apr 2025 – Mar 2026). Companion report: "
        "`reports/07_bridge_od.md` (Pass A perimeter OD + congestion).\n\n"
    )
    md_parts.append(_scope_block())
    md_parts.append(_headline_section(index_df_leonia, total_visitor_volume))
    md_parts.append(_ranking_section(index_df_leonia, top_n=20))
    md_parts.append(_per_street_pages(index_df_leonia, home_zips_top, work_bg, n=10))
    md_parts.append(_peak_hours_section(
        index_df_leonia, peak_am, peak_pm,
        intensity_am, intensity_pm, hourly_profile,
        top_n=15,
    ))
    md_parts.append(_origin_section(home_zips_top))
    md_parts.append(_destination_section(work_bg))
    md_parts.append(_coverage_section(line_gdf))

    md_parts.append("## Interactive maps\n\n")
    md_parts.append("- [Composite cut-through index](maps/street_cutthrough_index.html)\n")
    md_parts.append("- [Visitor weekday volume](maps/street_visitor_volume.html)\n")
    md_parts.append("- [Speeding-bin share](maps/street_speeding.html)\n\n")

    md_parts.append(_limitations_section())

    out_path = REPORTS_DIR / "09_leonia_streets.md"
    out_path.write_text("".join(md_parts), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"  {len(index_df_leonia)} in-borough residential segments ranked")


if __name__ == "__main__":
    main()
