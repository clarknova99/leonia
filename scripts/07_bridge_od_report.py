"""Pass A.5: Bridge OD + congestion evidence report.

Run from the repo root:

    venv/bin/python scripts/07_bridge_od_report.py

Outputs:

    reports/07_bridge_od.md
    reports/figures/od_dow_profile.png
    reports/figures/od_daypart_profile.png
    reports/figures/circuity_histogram.png
    reports/figures/trip_purpose_stack.png
    reports/figures/income_distribution_per_gate.png
    reports/figures/delay_hotspots.png
    reports/maps/od_flows.html
    reports/maps/congestion_tti.html
    reports/maps/congestion_reliability.html

The report is the primary stakeholder deliverable for this phase.
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

from leonia_traffic.analysis import congestion as cg
from leonia_traffic.analysis import od_cutthrough as oc
from leonia_traffic.analysis.jurisdiction import (
    annotate_in_leonia,
    filter_segments_to_leonia,
)
from leonia_traffic.analysis.equity import (
    EDUCATION_COLUMNS,
    EMPLOYMENT_CLASS_COLUMNS,
    EMPLOYMENT_INDUSTRY_COLUMNS,
    EQUITY_COLUMNS,
    HOUSEHOLD_COLUMNS,
    INCOME_COLUMNS,
    equity_exposure_index,
    gateway_user_profile,
)
from leonia_traffic.analysis.recommendations import (
    generate_recommendations,
    recommendations_to_markdown,
)
from leonia_traffic.config import (
    DATA_NETWORK_DIR,
    DATA_STAGE2_DIR,
    REPORTS_DIR,
    REPORTS_FIG_DIR,
)
from leonia_traffic.data.bridge_od_loader import (
    DAY_PART_CODES,
    load_bridge_attributes,
    load_bridge_od,
    load_bridge_zone_shapes,
)
from leonia_traffic.data.congestion_loader import (
    classify_reliability,
    load_congestion,
    load_congestion_zones,
    summarize_link_reliability,
)
from leonia_traffic.viz.maps import od_flow_map, reliability_map, tti_map

REPORTS_MAPS_DIR = REPORTS_DIR / "maps"
REPORTS_MAPS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Formatting helpers (copied/extended from scripts/01_explore_data.py)
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
        return f"{f*100:.{digits}f}%"
    except (TypeError, ValueError):
        return ""


def _table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    """Markdown table.

    ``columns`` is a list of ``(column_name, header_label, format)``
    tuples. Format codes: ``"int"``, ``"float2"``, ``"float3"``,
    ``"pct"``, ``"str"``.
    """
    headers = [h for _, h, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    fmt_map = {
        "int": _fmt_int,
        "float2": lambda v: _fmt_float(v, 2),
        "float3": lambda v: _fmt_float(v, 3),
        "pct": _fmt_pct,
        "str": lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v),
    }
    for _, row in df.iterrows():
        cells = []
        for col, _, fmt in columns:
            v = row.get(col)
            cells.append(fmt_map.get(fmt, fmt_map["str"])(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _heatmap(matrix: np.ndarray, x_labels, y_labels, title: str, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(y_labels)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, _fmt_int(v), ha="center", va="center",
                        color="white" if v > np.nanmax(matrix) * 0.5 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="trips/day")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _make_dow_heatmap(od_df: pd.DataFrame, fig_path: Path) -> None:
    dow = oc.day_of_week_profile(od_df, day_part_code=oc.PEAK_AM_CODE)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    y = dow["origin_label"].tolist()
    m = dow[days].to_numpy(dtype=float)
    _heatmap(m, days, y, "OD volume by origin × day-of-week (Peak AM weekday)", fig_path)


def _make_daypart_heatmap(od_df: pd.DataFrame, fig_path: Path) -> None:
    sub = od_df[od_df["day_type_code"].isin((1, 2, 3, 4, 5))]
    pivot = sub.pivot_table(
        index=["origin_zone", "origin_label"],
        columns="day_part_code",
        values="od_volume",
        aggfunc="sum",
    ).fillna(0)
    day_part_order = [0, 1, 2, 3, 4, 5]
    pivot = pivot.reindex(columns=day_part_order, fill_value=0)
    y = [lbl for _, lbl in pivot.index.tolist()]
    x = [DAY_PART_CODES.get(c, str(c)).split(" (")[0] for c in pivot.columns]
    _heatmap(pivot.to_numpy(dtype=float), x, y,
             "OD volume by origin × day-part (weekday total)", fig_path)


def _make_circuity_histogram(circuity_df: pd.DataFrame, fig_path: Path) -> None:
    if circuity_df.empty:
        return
    buckets = ["circuity_low_pct", "circuity_mid_pct", "circuity_3_4_pct",
               "circuity_4_5_pct", "circuity_5_6_pct", "circuity_6plus_pct"]
    labels = ["1-2", "2-3", "3-4", "4-5", "5-6", "6+"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = np.zeros(len(circuity_df))
    colors = ["#1a9850", "#a6d96a", "#fee08b", "#fdae61", "#f46d43", "#a50026"]
    for col, c, lbl in zip(buckets, colors, labels):
        vals = (circuity_df[col].fillna(0) * 100).to_numpy()
        ax.bar(range(len(circuity_df)), vals, bottom=bottoms, color=c, label=lbl)
        bottoms = bottoms + vals
    ax.set_xticks(range(len(circuity_df)))
    ax.set_xticklabels(circuity_df["origin_label"], rotation=20, ha="right")
    ax.set_ylabel("Share of trips (%)")
    ax.set_title("Trip circuity distribution by origin (Peak AM weekday)")
    ax.legend(title="Circuity bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _make_trip_purpose_figure(tp_df: pd.DataFrame, fig_path: Path) -> None:
    if tp_df.empty:
        return
    peak = tp_df[tp_df["day_part_code"] == oc.PEAK_AM_CODE].copy()
    if peak.empty:
        return
    peak = peak.sort_values("weekday_trips", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = np.zeros(len(peak))
    purposes = [
        ("home_to_work_pct", "Home → Work", "#1f78b4"),
        ("home_to_other_pct", "Home → Other", "#33a02c"),
        ("non_home_based_pct", "Non-home based", "#ff7f00"),
    ]
    for col, lbl, c in purposes:
        vals = (peak[col].fillna(0) * 100).to_numpy()
        ax.bar(range(len(peak)), vals, bottom=bottoms, color=c, label=lbl)
        bottoms = bottoms + vals
    ax.set_xticks(range(len(peak)))
    ax.set_xticklabels(peak["origin_label"], rotation=20, ha="right")
    ax.set_ylabel("Share of trips (%)")
    ax.set_title("Trip purpose by origin (Peak AM weekday)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _make_income_figure(attr_df: pd.DataFrame, fig_path: Path) -> None:
    origins = attr_df[["origin_zone", "origin_label"]].drop_duplicates()
    if origins.empty:
        return
    bracket_keys = list(INCOME_COLUMNS.keys())
    rows = []
    for _, og in origins.iterrows():
        prof = gateway_user_profile(attr_df, og["origin_zone"])
        income = prof.get("income", {})
        rows.append({
            "origin": og["origin_label"],
            **{b: income.get(b, 0) for b in bracket_keys},
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4))
    bottoms = np.zeros(len(df))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(bracket_keys)))
    short_labels = [b.replace("_", "-") for b in bracket_keys]
    for c, color, lbl in zip(bracket_keys, cmap, short_labels):
        vals = (df[c].fillna(0) * 100).to_numpy()
        ax.bar(range(len(df)), vals, bottom=bottoms, color=color, label=lbl)
        bottoms = bottoms + vals
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["origin"], rotation=20, ha="right")
    ax.set_ylabel("Share of households (%)")
    ax.set_title("Household income distribution per origin gate")
    ax.legend(title="Income", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _make_delay_hotspot_figure(delay_df: pd.DataFrame, fig_path: Path) -> None:
    if delay_df.empty:
        return
    sub = delay_df.head(15).iloc[::-1]   # ascending order for horizontal bars
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(sub)), sub["weekday_vhd_total"], color="#a50026")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f"{n} ({c})" for n, c in zip(sub["osm_name"], sub["road_class"])])
    ax.set_xlabel("Weekday Vehicle Hours of Delay (sum across hourly buckets)")
    ax.set_title("Top 15 delay hotspots")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------


def _headline_section(od_df: pd.DataFrame) -> str:
    peak_am = od_df[(od_df["day_part_code"] == oc.PEAK_AM_CODE)
                    & (od_df["day_type_code"].isin([1, 2, 3, 4, 5, 6, 7]))]
    by_origin_dow = peak_am.groupby(
        ["origin_label", "origin_osm_way_id", "day_type_code", "day_type_label"],
        dropna=False, as_index=False,
    )["od_volume"].sum()
    top = by_origin_dow.sort_values("od_volume", ascending=False).head(1)
    if top.empty:
        return "_No data yet — bridge OD export not present._\n\n"

    rec = top.iloc[0]
    return (
        f"**Headline finding (Pass A.1):** The single largest source of Peak-AM "
        f"weekday traffic into the GWB from the Leonia perimeter is "
        f"**{rec['origin_label']}** (OSM way `{int(rec['origin_osm_way_id'])}`), "
        f"with up to **{int(rec['od_volume']):,} trips** during a single "
        f"Peak-AM window on {rec['day_type_label']}. The four other Leonia gates "
        f"combined typically contribute under 200 trips during the same window. "
        f"Cut-through to the GWB is concentrated on **one corridor**.\n\n"
    )


def _gateway_profiles_section(peak_imbalance: pd.DataFrame, circuity: pd.DataFrame) -> str:
    md = "## Gateway profiles\n\n"
    md += "Volumes are summed across both GWB upper and lower-level destinations.\n\n"
    if peak_imbalance.empty:
        return md + "_No data yet._\n\n"
    md += _table(peak_imbalance, [
        ("origin_label", "Origin gate", "str"),
        ("origin_osm_way_id", "OSM way", "int"),
        ("weekday_all_day_avg", "Weekday all-day avg", "int"),
        ("weekend_all_day_avg", "Weekend all-day avg", "int"),
        ("peak_am_weekday_avg", "Peak AM (weekday)", "int"),
        ("weekend_peak_am_avg", "Peak AM (weekend)", "int"),
        ("peak_pm_weekday_avg", "Peak PM (weekday)", "int"),
        ("peak_am_to_weekend_ratio", "Peak-AM weekday/weekend ratio", "float2"),
    ])
    md += "\n### Circuity-based cut-through evidence\n\n"
    if not circuity.empty:
        md += _table(circuity, [
            ("origin_label", "Origin gate", "str"),
            ("circuity_low_pct", "Direct (1-2)", "pct"),
            ("circuity_mid_pct", "Mild detour (2-3)", "pct"),
            ("circuity_high_pct", "Heavy detour (3+)", "pct"),
            ("cutthrough_circuity_index", "Cut-through circuity index", "float3"),
            ("trips_in_window", "Peak-AM trips analyzed", "int"),
        ])
    return md + "\n"


def _trip_purpose_section(tp_df: pd.DataFrame) -> str:
    md = "## Trip purpose decomposition\n\n"
    if tp_df.empty:
        return md + "_No data yet._\n\n"
    peak = tp_df[tp_df["day_part_code"] == oc.PEAK_AM_CODE].copy()
    peak["origin"] = peak["origin_label"]
    md += "Weekday Peak-AM purpose mix by origin gate:\n\n"
    md += _table(peak.sort_values("weekday_trips", ascending=False), [
        ("origin", "Origin gate", "str"),
        ("weekday_trips", "Mon-Fri Peak-AM trips", "int"),
        ("home_to_work_pct", "Home → Work", "pct"),
        ("home_to_other_pct", "Home → Other", "pct"),
        ("non_home_based_pct", "Non-home based", "pct"),
    ])
    md += "\n![Trip purpose](figures/trip_purpose_stack.png)\n\n"
    return md


def _demographics_section(attr_df: pd.DataFrame) -> str:
    md = "## Demographic profiles\n\n"
    md += "Shares are volume-weighted across weekday all-day OD trips. "
    md += "StreetLight estimates these attributes from device home locations; "
    md += "they are not a direct survey of drivers.\n\n"
    origins = attr_df[["origin_zone", "origin_label", "origin_osm_way_id"]].drop_duplicates()
    if origins.empty:
        return md + "_No data yet._\n\n"

    for _, og in origins.iterrows():
        prof = gateway_user_profile(attr_df, og["origin_zone"])
        md += f"### {og['origin_label']} (OSM way {int(og['origin_osm_way_id'])})\n\n"

        # Race + ethnicity
        md += "**Race / ethnicity / nativity / language / disability**:\n\n"
        sections = ["race", "ethnicity", "nativity", "language", "disability"]
        for sec in sections:
            attrs = prof.get(sec, {})
            row = ", ".join(f"{k}: {_fmt_pct(v)}" for k, v in attrs.items() if v is not None)
            md += f"- {sec.capitalize()} — {row}\n"

        md += "\n**Household** (children, tenure, vehicles, structure):\n\n"
        for sec in ("household_children", "household_tenure",
                    "household_vehicles", "household_unit_structure"):
            attrs = prof.get(sec, {})
            row = ", ".join(f"{k}: {_fmt_pct(v)}" for k, v in attrs.items() if v is not None)
            md += f"- {sec.replace('household_', '').capitalize()} — {row}\n"

        md += "\n**Income / education**:\n\n"
        income = prof.get("income", {})
        md += "- Income brackets: "
        md += ", ".join(f"{k}: {_fmt_pct(v)}" for k, v in income.items() if v) + "\n"
        edu = prof.get("education", {})
        md += "- Education: " + ", ".join(f"{k}: {_fmt_pct(v)}" for k, v in edu.items() if v) + "\n"

        md += "\n**Employment industry / class**:\n\n"
        ind = prof.get("employment_industry", {})
        md += "- Industry: " + ", ".join(f"{k}: {_fmt_pct(v)}" for k, v in ind.items() if v) + "\n"
        cls = prof.get("employment_class", {})
        md += "- Class: " + ", ".join(f"{k}: {_fmt_pct(v)}" for k, v in cls.items() if v) + "\n"

        agg = prof.get("aggregates", {})
        md += "\n**Equity aggregates**: "
        md += f"low-income (<$50K): {_fmt_pct(agg.get('low_income_under_50k'))}, "
        md += f"foreign-born: {_fmt_pct(agg.get('foreign_born'))}, "
        md += f"English limited: {_fmt_pct(agg.get('english_limited'))}, "
        md += f"renter-occupied: {_fmt_pct(agg.get('renter_occupied'))}, "
        md += f"no vehicle: {_fmt_pct(agg.get('no_vehicle'))}.\n\n"

    md += "![Income distribution](figures/income_distribution_per_gate.png)\n\n"
    return md


def _congestion_section(
    cdf: pd.DataFrame,
    summary: pd.DataFrame,
    delay: pd.DataFrame,
    reliability: pd.DataFrame,
    summary_leonia: pd.DataFrame,
    delay_leonia: pd.DataFrame,
) -> str:
    md = "## Congestion hotspots\n\n"
    if summary.empty:
        return md + "_No congestion data yet._\n\n"

    md += (
        "> **Jurisdictional scope.** The borough of Leonia can act only on "
        "streets located inside the municipal boundary and not owned by NJDOT, "
        "the New Jersey Turnpike Authority, or the federal government (NJ Turnpike, "
        "GWB Plaza & approaches, US 1/9/46, NJ 4, Mackay Highway, and motorway "
        "ramps). Tables below show every congestion segment in the export for "
        "context; the recommendations engine restricts attention to in-borough "
        "municipal streets only.\n\n"
    )

    md += "### All observed corridors (within ~1 mile of Leonia)\n\n"
    md += "Top 15 corridors by weekday-hourly worst Travel Time Index "
    md += "(TTI > 1.0 means slower than free-flow; > 2.0 typically indicates "
    md += "service-level failure):\n\n"
    md += _table(summary.head(15), [
        ("osm_name", "Corridor", "str"),
        ("road_class", "Road class", "str"),
        ("in_leonia_jurisdiction", "In Leonia jurisdiction?", "str"),
        ("worst_tti", "Worst TTI", "float2"),
        ("worst_buffer", "Worst Buffer Idx", "float2"),
        ("median_speed_mph", "Median speed (mph)", "float2"),
        ("free_flow_speed_mph", "Free-flow speed", "float2"),
        ("worst_lottr", "LOTTR (peak)", "float2"),
        ("reliability_class", "Reliability", "str"),
    ])

    md += "\n### In-Leonia congestion (recommendation scope)\n\n"
    if summary_leonia.empty:
        md += "_No in-Leonia segments tripped the TTI / Buffer thresholds._\n\n"
    else:
        md += _table(summary_leonia.head(15), [
            ("osm_name", "Corridor", "str"),
            ("road_class", "Road class", "str"),
            ("worst_tti", "Worst TTI", "float2"),
            ("worst_buffer", "Worst Buffer Idx", "float2"),
            ("median_speed_mph", "Median speed (mph)", "float2"),
            ("free_flow_speed_mph", "Free-flow speed", "float2"),
            ("worst_lottr", "LOTTR (peak)", "float2"),
            ("reliability_class", "Reliability", "str"),
        ])

    md += "\n### Top delay hotspots — all observed\n\n"
    md += _table(delay.head(15), [
        ("osm_name", "Corridor", "str"),
        ("road_class", "Road class", "str"),
        ("in_leonia_jurisdiction", "In Leonia jurisdiction?", "str"),
        ("length_mi", "Length (mi)", "float2"),
        ("weekday_vhd_total", "Weekday VHD total", "float2"),
        ("delay_per_mile", "Delay per mile", "float2"),
        ("worst_hour_tti", "Worst-hour TTI", "float2"),
        ("worst_hour_label", "Worst hour", "str"),
    ])

    md += "\n### Top delay hotspots — within Leonia jurisdiction\n\n"
    if delay_leonia.empty:
        md += "_No in-Leonia segments above the delay threshold._\n\n"
    else:
        md += _table(delay_leonia.head(15), [
            ("osm_name", "Corridor", "str"),
            ("road_class", "Road class", "str"),
            ("length_mi", "Length (mi)", "float2"),
            ("weekday_vhd_total", "Weekday VHD total", "float2"),
            ("delay_per_mile", "Delay per mile", "float2"),
            ("worst_hour_tti", "Worst-hour TTI", "float2"),
            ("worst_hour_label", "Worst hour", "str"),
        ])
    md += "\n![Delay hotspots](figures/delay_hotspots.png)\n\n"

    md += "\n### Reliability classification\n\n"
    md += _table(reliability, [
        ("road_class", "Road class", "str"),
        ("reliability_class", "Class", "str"),
        ("n_segments", "Segments", "int"),
        ("share_of_road_class", "Share", "pct"),
    ])
    return md + "\n"


def _equity_section(exposure: pd.DataFrame) -> str:
    md = "## Equity exposure\n\n"
    if exposure.empty:
        return md + "_No data yet._\n\n"
    md += "Per-gate equity flags. A flag fires when the corresponding share "
    md += "exceeds the threshold listed in `analysis/equity.py:EQUITY_THRESHOLDS`.\n\n"
    md += _table(exposure, [
        ("origin_label", "Origin gate", "str"),
        ("weekday_peak_volume", "Peak-AM weekday vol.", "int"),
        ("foreign_born_share", "Foreign-born", "pct"),
        ("english_limited_share", "English limited", "pct"),
        ("low_income_under_50k_share", "Under $50K HH", "pct"),
        ("no_vehicle_share", "No vehicle", "pct"),
        ("renter_occupied_share", "Renter occupied", "pct"),
        ("any_equity_flag", "Any flag?", "str"),
    ])
    return md + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading data...")
    od_df = load_bridge_od()
    attr_df = load_bridge_attributes()
    cdf = load_congestion()
    congestion_zones = load_congestion_zones()
    origin_zones = load_bridge_zone_shapes(kind="line")

    print(f"OD rows: {len(od_df)}")
    print(f"Attribute rows: {len(attr_df)}")
    print(f"Congestion rows: {len(cdf)}")

    if od_df.empty or cdf.empty:
        raise SystemExit("Required exports not found — see streetlight/ folder.")

    print("Computing analyses...")
    peak_imbalance = oc.gateway_peak_imbalance(od_df)
    circuity = oc.cutthrough_index_from_circuity(attr_df)
    tp_decomp = oc.trip_purpose_decomposition(attr_df)

    summary = summarize_link_reliability(cdf)
    delay = cg.delay_hotspot_ranking(cdf)
    reliability = cg.reliability_breakdown(cdf)
    overrides = cg.link_speed_overrides(
        cdf,
        cache_path=DATA_NETWORK_DIR / "speed_overrides_weekday_peak_am.parquet",
    )

    summary = annotate_in_leonia(summary, congestion_zones)
    delay = annotate_in_leonia(delay, congestion_zones)

    summary_leonia = filter_segments_to_leonia(summary, congestion_zones)
    delay_leonia = filter_segments_to_leonia(delay, congestion_zones)
    print(
        f"Jurisdiction filter: {len(summary)}→{len(summary_leonia)} reliability rows, "
        f"{len(delay)}→{len(delay_leonia)} delay rows in-Leonia."
    )

    exposure = equity_exposure_index(attr_df)

    print("Generating figures...")
    _make_dow_heatmap(od_df, REPORTS_FIG_DIR / "od_dow_profile.png")
    _make_daypart_heatmap(od_df, REPORTS_FIG_DIR / "od_daypart_profile.png")
    _make_circuity_histogram(circuity, REPORTS_FIG_DIR / "circuity_histogram.png")
    _make_trip_purpose_figure(tp_decomp, REPORTS_FIG_DIR / "trip_purpose_stack.png")
    _make_income_figure(attr_df, REPORTS_FIG_DIR / "income_distribution_per_gate.png")
    _make_delay_hotspot_figure(delay, REPORTS_FIG_DIR / "delay_hotspots.png")

    print("Generating maps...")
    origin_gdf = origin_zones[origin_zones["zone_role"] == "origin"]
    dest_gdf = origin_zones[origin_zones["zone_role"] == "destination"]

    flows = []
    peak_rows = od_df[(od_df["day_type_code"].isin((1, 2, 3, 4, 5)))
                      & (od_df["day_part_code"] == oc.PEAK_AM_CODE)]
    flow_totals = peak_rows.groupby(
        ["origin_zone", "destination_zone"], as_index=False,
    )["od_volume"].sum()
    for _, row in flow_totals.iterrows():
        flows.append({
            "origin": row["origin_zone"],
            "destination": row["destination_zone"],
            "volume": float(row["od_volume"]),
            "label": (
                f"{row['origin_zone']} → {row['destination_zone']}<br>"
                f"Weekday Peak-AM total: {int(row['od_volume']):,} trips"
            ),
        })

    if not origin_gdf.empty and not dest_gdf.empty:
        fmap_flow = od_flow_map(origin_gdf, dest_gdf, flows, label="Peak AM OD volume")
        fmap_flow.save(str(REPORTS_MAPS_DIR / "od_flows.html"))

    if not congestion_zones.empty and not summary.empty:
        merged = congestion_zones.merge(
            summary, left_on="osm_way_id", right_on="osm_way_id",
            how="left", suffixes=("_zone", ""),
        )
        merged["reliability_class"] = merged["worst_lottr"].apply(classify_reliability)
        fmap_tti = tti_map(merged)
        fmap_tti.save(str(REPORTS_MAPS_DIR / "congestion_tti.html"))
        fmap_rel = reliability_map(merged)
        fmap_rel.save(str(REPORTS_MAPS_DIR / "congestion_reliability.html"))

    print("Generating recommendations (Leonia jurisdiction only)...")
    # Load the Pass-C per-residential-street index if it's been built;
    # otherwise the residential rules are silently skipped.
    per_street_df = None
    per_street_path = DATA_STAGE2_DIR / "leonia_streets_cutthrough_index.parquet"
    if per_street_path.exists():
        try:
            per_street_df = pd.read_parquet(per_street_path)
            print(
                f"  Loaded Pass-C per-street index ({len(per_street_df)} rows) "
                "for residential rules."
            )
        except (OSError, ValueError) as exc:
            print(f"  Pass-C parquet read failed ({exc}); skipping residential rules.")

    # Load Pass-D derived tables (trend + OMD attribution) if present.
    street_trend_df = None
    cutthrough_attribution_df = None
    derived_dir = DATA_STAGE2_DIR
    trend_path = derived_dir / "street_trend.parquet"
    if trend_path.exists():
        try:
            street_trend_df = pd.read_parquet(trend_path)
            print(
                f"  Loaded street trend table ({len(street_trend_df)} rows) "
                "for accelerating-cutthrough rule."
            )
        except (OSError, ValueError) as exc:
            print(f"  Trend parquet read failed ({exc}); skipping trend rule.")
    attr_path = derived_dir / "cutthrough_attribution.parquet"
    if attr_path.exists():
        try:
            cutthrough_attribution_df = pd.read_parquet(attr_path)
            print(
                f"  Loaded OMD attribution ({len(cutthrough_attribution_df)} rows) "
                "for confirmed-cutthrough rule."
            )
        except (OSError, ValueError) as exc:
            print(f"  Attribution parquet read failed ({exc}); skipping OMD rule.")

    recs = generate_recommendations(
        peak_imbalance_df=peak_imbalance,
        circuity_df=circuity,
        delay_df=delay_leonia,
        summary_df=summary_leonia,
        exposure_df=exposure,
        per_street_df=per_street_df,
        street_trend_df=street_trend_df,
        cutthrough_attribution_df=cutthrough_attribution_df,
    )
    recs_md = recommendations_to_markdown(recs)

    print("Writing report...")
    md_parts: list[str] = []
    md_parts.append("# Bridge OD + Congestion evidence report\n\n")
    md_parts.append(
        "Generated by `scripts/07_bridge_od_report.py`. Sources: "
        "`streetlight/bridge_destination/` (OD Analysis, Apr 2025 – Mar 2026) "
        "and `streetlight/congestion/` (Congestion Trends, Jan 2025 – Mar 2026). "
        f"Recommendations: **{len(recs)}** rules triggered.\n\n"
    )
    md_parts.append(
        "> **Scope of recommendations.** The Borough of Leonia has authority "
        "only over municipal streets located inside the borough limits. "
        "State and federal facilities crossing the borough (NJ Turnpike, "
        "George Washington Bridge & approaches, US 1/9/46, NJ 4, NJDOT ramps) "
        "are governed by NJDOT, the New Jersey Turnpike Authority, and the "
        "Port Authority of NY & NJ. The recommendation engine below filters "
        "all corridor-level rules to in-borough municipal streets only; "
        "OD-gate recommendations (Fort Lee Rd, Grand Ave, Broad Ave) target "
        "the Leonia-perimeter approach links the borough does control. "
        "State-road congestion is still surfaced in the data tables for "
        "situational awareness but is **not** subject to borough action.\n\n"
    )

    md_parts.append("## Headline\n\n")
    md_parts.append(_headline_section(od_df))
    md_parts.append("![Day-of-week × origin (Peak AM)](figures/od_dow_profile.png)\n\n")
    md_parts.append("![Day-part × origin (weekday total)](figures/od_daypart_profile.png)\n\n")

    md_parts.append(_gateway_profiles_section(peak_imbalance, circuity))
    md_parts.append("![Circuity distribution](figures/circuity_histogram.png)\n\n")

    md_parts.append(_trip_purpose_section(tp_decomp))
    md_parts.append(_demographics_section(attr_df))
    md_parts.append(_congestion_section(
        cdf, summary, delay, reliability,
        summary_leonia=summary_leonia,
        delay_leonia=delay_leonia,
    ))
    md_parts.append(_equity_section(exposure))

    md_parts.append("## Interactive maps\n\n")
    md_parts.append("- [OD flows (Peak-AM weekday)](maps/od_flows.html)\n")
    md_parts.append("- [Worst-hour Travel Time Index](maps/congestion_tti.html)\n")
    md_parts.append("- [Reliability classification](maps/congestion_reliability.html)\n\n")

    md_parts.append("## Recommendations\n\n")
    md_parts.append(
        "_All targets below are inside the Borough of Leonia and under "
        "municipal jurisdiction. State and federal facilities (NJ Turnpike, "
        "GWB Plaza, US 1/9/46, NJ 4) are excluded by design — Leonia has "
        "no authority over them. Equity and circuity rules target OD-gate "
        "approach links at the borough perimeter (also municipal). "
        "Residential-cut-through rules (rule `residential_cutthrough_candidate` "
        "and `residential_speeding_callout`) come from Pass C — see "
        "`reports/09_leonia_streets.md` for the per-street evidence base._\n\n"
    )
    md_parts.append(recs_md)
    md_parts.append("\n")

    md_parts.append("## Coverage and data limitations\n\n")
    md_parts.append(
        "### Street-level coverage inside Leonia\n\n"
        "Leonia has **87 named drivable streets** (613 OSM edges, ~71% "
        "classified residential). The four StreetLight exports now cover "
        "the borough at three levels:\n\n"
        "| Dataset | In-Leonia segments | Named streets covered | Notes |\n"
        "| --- | --- | --- | --- |\n"
        "| Bridge OD | 5 origin gates | 3 (Fort Lee Rd, Grand Ave, Broad Ave) | "
        "Perimeter OD only; says nothing about interior routing. |\n"
        "| Congestion Trends | 39 segments | 7 named arterials + 16 unnamed tertiary stubs | "
        "Drives TTI / Buffer / VHD recommendations on arterials. |\n"
        "| Street Scanner | ~389 zones | broadest coverage of the three, but still concentrated on arterials/collectors | Older baseline, used for Pass-B calibration on arterials. |\n"
        "| **Leonia streets (Pass C)** | **151 OSM tertiary segments after the residential filter** | **Christie Heights, Willow Tree, Schor, Pine Hill, Hoefleys, Lakeview, Nordhoff, Lowe, Crescent, Walnut, Birch and dozens more residential blocks** | **Direct per-zone Visitor pass-through volume, trip-length, speed, and home-ZIP distributions.** |\n\n"
        "**Where to look for what.** TTI / Buffer / VHD findings still come "
        "from Congestion Trends and apply to the 7 arterials it covers. "
        "Per-residential-street cut-through evidence — the heart of the "
        "borough's traffic-calming question — now comes from "
        "`reports/09_leonia_streets.md`, where each candidate block is "
        "ranked by a composite cut-through index combining "
        "weekday/weekend imbalance, non-local home share, long-trip share, "
        "speeding-bin share, and absolute volume. Residential rules in the "
        "recommendations table above (`residential_cutthrough_candidate`, "
        "`residential_speeding_callout`) are driven by that report.\n\n"
        "### Other limitations\n\n"
        "* OD origin zones in this analysis are 5 Leonia-perimeter "
        "corridors. Trips entering Leonia from other directions are not "
        "included.\n"
        "* Demographic attributes are StreetLight estimates from device "
        "home locations, not a direct driver survey.\n"
        "* Recommendations are filtered to streets under Borough of Leonia "
        "municipal jurisdiction (borough polygon + exclusion of NJDOT, "
        "Turnpike Authority, and Port Authority facilities). State-road "
        "congestion is reported for situational awareness only.\n"
        f"* Per-OSM-way speed overrides cached at "
        f"`data/network/speed_overrides_weekday_peak_am.parquet` "
        f"({len(overrides)} rows) for Pass B.\n"
    )

    out_path = REPORTS_DIR / "07_bridge_od.md"
    out_path.write_text("".join(md_parts), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"  {len(recs)} recommendations")


if __name__ == "__main__":
    main()
