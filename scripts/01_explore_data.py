"""Phase 1c: exploratory analysis of the StreetLight Street Scanner data.

Run from the repo root:

    venv/bin/python scripts/01_explore_data.py

Outputs:

    reports/01_exploration.md
    reports/figures/volume_by_road_class_<source>.png
    reports/figures/speed_over_limit_hist_<source>.png
    reports/figures/weekday_weekend_ratio_hist.png
    reports/maps/volume_<source>.html
    reports/maps/weekday_weekend_ratio.html
    reports/maps/cutthrough_suspects.html

The report is the primary deliverable for this phase.
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

from leonia_traffic.analysis.cutthrough import (
    add_weekday_weekend_signals,
    rank_cutthrough_suspects,
    rank_speed_over_limit,
    residential_volume_percentiles,
)
from leonia_traffic.config import REPORTS_DIR, REPORTS_FIG_DIR
from leonia_traffic.data.streetlight_loader import (
    discover_sources,
    load_cached,
    pivot_by_source,
    restrict_to_study_area,
)
from leonia_traffic.viz.maps import ratio_map, volume_map

REPORTS_MAPS_DIR = REPORTS_DIR / "maps"
REPORTS_MAPS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "?"


def _table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    """Render a pandas frame as a GitHub markdown table.

    ``columns`` is a list of ``(column_name, header_label)`` tuples.
    """
    headers = [h for _, h in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col, _ in columns:
            v = row.get(col)
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("")
                elif abs(v) >= 100:
                    cells.append(f"{v:,.0f}")
                else:
                    cells.append(f"{v:.2f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot_volume_by_road_class(gdf, source: str, out_path: Path) -> None:
    g = gdf[gdf["source"] == source]
    if g.empty:
        return
    classes = sorted(g["road_class"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [g.loc[g["road_class"] == c, "avg_volume"].dropna().values for c in classes]
    ax.boxplot(data, tick_labels=classes, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Average volume (log scale)")
    ax.set_title(f"Volume distribution by road class — source: {source}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_speed_over_limit_hist(gdf, source: str, out_path: Path) -> None:
    g = gdf[gdf["source"] == source].copy()
    g = g[g["road_class"].isin(["residential", "tertiary"])]
    g["speed_over_limit"] = g["avg_speed_mph"] - g["speed_limit_mph"]
    g = g.dropna(subset=["speed_over_limit"])
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(g["speed_over_limit"], bins=40, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Average speed minus posted speed limit (mph)")
    ax.set_ylabel("Number of residential/tertiary segments")
    ax.set_title(f"Speed-over-limit on residential & tertiary roads — {source}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_wd_we_ratio_hist(piv, out_path: Path) -> None:
    g = piv.dropna(subset=["weekday_weekend_ratio"]).copy()
    g = g[(g["weekday_weekend_ratio"] > 0) & (g["weekday_weekend_ratio"] < 10)]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, label, color in [
        ("residential", "Residential", "#1f77b4"),
        ("tertiary", "Tertiary", "#ff7f0e"),
        ("primary", "Primary", "#2ca02c"),
    ]:
        sub = g.loc[g["road_class"] == cls, "weekday_weekend_ratio"]
        if not sub.empty:
            ax.hist(sub, bins=40, alpha=0.6, label=f"{label} (n={len(sub)})", color=color)
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Weekday volume / Weekend volume")
    ax.set_ylabel("Number of segments")
    ax.set_title("Weekday-to-weekend volume ratio by road class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    print("Discovering sources...")
    sources = discover_sources()
    for s in sources:
        print(f"  - {s.label}: {s.folder.name}")

    print("Loading (with parquet cache)...")
    gdf = load_cached()
    gdf = restrict_to_study_area(gdf)
    print(f"Loaded {len(gdf):,} rows across {gdf['source'].nunique()} sources")

    print("Pivoting by source...")
    piv = pivot_by_source(gdf, value_col="avg_volume")
    piv = add_weekday_weekend_signals(piv)
    print(f"Pivot table: {len(piv):,} segments")

    print("Computing rankings...")
    cutthrough = rank_cutthrough_suspects(
        piv, road_classes=("residential", "tertiary"), min_weekday_volume=200, top_n=30
    )
    speeders_wd = rank_speed_over_limit(gdf, source="weekdays", top_n=20)
    speeders_we = rank_speed_over_limit(gdf, source="weekend", top_n=20)
    pct_table_wd = residential_volume_percentiles(gdf, source="weekdays")
    pct_table_we = residential_volume_percentiles(gdf, source="weekend")
    pct_table_all = residential_volume_percentiles(gdf, source="all_days")

    print("Rendering figures...")
    for src in ("all_days", "weekdays", "weekend"):
        _plot_volume_by_road_class(
            gdf, src, REPORTS_FIG_DIR / f"volume_by_road_class_{src}.png"
        )
        _plot_speed_over_limit_hist(
            gdf, src, REPORTS_FIG_DIR / f"speed_over_limit_hist_{src}.png"
        )
    _plot_wd_we_ratio_hist(piv, REPORTS_FIG_DIR / "weekday_weekend_ratio_hist.png")

    print("Rendering maps...")
    for src, label in [
        ("all_days", "All-days avg volume"),
        ("weekdays", "Weekday avg volume (across 5 day parts)"),
        ("weekend", "Weekend avg volume (across 5 day parts)"),
    ]:
        sub = gdf[gdf["source"] == src]
        fmap = volume_map(sub, value_col="avg_volume", label=label, min_volume=50)
        fmap.save(str(REPORTS_MAPS_DIR / f"volume_{src}.html"))

    ratio_map(
        piv.dropna(subset=["weekday_weekend_ratio"]),
        value_col="weekday_weekend_ratio",
        label="Weekday / Weekend volume ratio",
        midpoint=1.0,
    ).save(str(REPORTS_MAPS_DIR / "weekday_weekend_ratio.html"))

    residential = piv[piv["road_class"].isin(["residential", "tertiary"])].copy()
    residential = residential.dropna(subset=["weekday_weekend_ratio"])
    ratio_map(
        residential,
        value_col="weekday_weekend_ratio",
        label="Weekday/Weekend ratio — residential & tertiary only",
        midpoint=1.0,
        line_weight=4.0,
    ).save(str(REPORTS_MAPS_DIR / "cutthrough_suspects.html"))

    print("Writing report...")
    write_report(
        gdf=gdf,
        piv=piv,
        cutthrough=cutthrough,
        speeders_wd=speeders_wd,
        speeders_we=speeders_we,
        pct_table_wd=pct_table_wd,
        pct_table_we=pct_table_we,
        pct_table_all=pct_table_all,
        sources=sources,
    )
    print(f"Done. See {REPORTS_DIR / '01_exploration.md'}")


def write_report(
    *,
    gdf,
    piv,
    cutthrough,
    speeders_wd,
    speeders_we,
    pct_table_wd,
    pct_table_we,
    pct_table_all,
    sources,
) -> None:
    lines: list[str] = []
    lines.append("# Leonia traffic data — Phase 1 exploration\n")
    lines.append(
        "_Auto-generated by `scripts/01_explore_data.py`. Source: StreetLight "
        "Street Scanner exports under `streetlight/`._\n"
    )

    lines.append("## Data sources loaded\n")
    src_rows = []
    for s in sources:
        sub = gdf[gdf["source"] == s.label]
        src_rows.append(
            {
                "source": s.label,
                "folder": s.folder.name or "streetlight/",
                "rows_in_study_area": len(sub),
                "day_type": sub["day_type"].iloc[0] if len(sub) else "",
                "day_part_raw": (sub["day_part_raw"].iloc[0] if len(sub) else "")[:80],
            }
        )
    src_df = pd.DataFrame(src_rows)
    lines.append(
        _table(
            src_df,
            [
                ("source", "Source label"),
                ("folder", "Folder"),
                ("rows_in_study_area", "Rows (study area)"),
                ("day_type", "Day Type"),
                ("day_part_raw", "Day Part(s)"),
            ],
        )
    )
    lines.append("")
    lines.append(
        "> **Caveat.** The Weekday and Weekend exports list five day parts but "
        "contain only one row per segment — Street Scanner averaged across the "
        "selected parts. To get true per-day-part rows, re-pull each Day Part "
        "in its own export (see plan Part A.1)."
    )
    lines.append("")

    lines.append("## City/Town coverage\n")
    coverage = (
        gdf[gdf["source"] == "weekdays"]
        .groupby("city_county_state")["zone_name"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"zone_name": "segments"})
    )
    lines.append(
        _table(
            coverage,
            [("city_county_state", "City, County, State"), ("segments", "Segments")],
        )
    )
    lines.append("")

    lines.append("## Volume distribution by road class\n")
    for label, table in [
        ("All days", pct_table_all),
        ("Weekdays", pct_table_wd),
        ("Weekend", pct_table_we),
    ]:
        lines.append(f"**{label}**\n")
        lines.append("```")
        lines.append(table.to_string())
        lines.append("```")
        lines.append("")
    lines.append("Figures:")
    for src in ("all_days", "weekdays", "weekend"):
        lines.append(f"- `reports/figures/volume_by_road_class_{src}.png`")
    lines.append("")

    lines.append("## Weekday vs. Weekend asymmetry (cut-through signal)\n")
    lines.append(
        "Segments with a high **weekday / weekend volume ratio** are likely "
        "carrying commuter traffic rather than local trips. Residential and "
        "tertiary streets at the top of this list are prime cut-through "
        "suspects."
    )
    lines.append("")
    lines.append(
        f"Of {(piv['road_class'].isin(['residential','tertiary'])).sum():,} "
        f"residential/tertiary segments in the study area, "
        f"{int((piv['weekday_weekend_ratio'] > 1.5).sum()):,} have a "
        f"weekday/weekend ratio above 1.5 — strong commuter-bias signal."
    )
    lines.append("")
    lines.append("![Weekday/Weekend ratio histogram](figures/weekday_weekend_ratio_hist.png)\n")

    show_cols = [
        ("road_name", "Road"),
        ("city_county_state", "City"),
        ("road_class", "Class"),
        ("avg_volume__weekdays", "Wkdy vol"),
        ("avg_volume__weekend", "Wknd vol"),
        ("weekday_weekend_ratio", "Wkdy/Wknd"),
        ("speed_limit_mph", "Limit"),
        ("is_suspected_cutthrough_street", "On suspect list?"),
    ]
    lines.append("### Top 30 cut-through suspects (residential + tertiary)\n")
    lines.append(_table(cutthrough, show_cols))
    lines.append("")

    suspect_summary = (
        cutthrough.groupby("road_name")
        .agg(
            n_segments=("zone_name", "count"),
            mean_ratio=("weekday_weekend_ratio", "mean"),
            mean_weekday_vol=("avg_volume__weekdays", "mean"),
            on_suspect_list=("is_suspected_cutthrough_street", "any"),
        )
        .sort_values("mean_ratio", ascending=False)
        .reset_index()
    )
    lines.append("### Aggregated by street name\n")
    lines.append(
        _table(
            suspect_summary,
            [
                ("road_name", "Road"),
                ("n_segments", "# top-30 segments"),
                ("mean_ratio", "Mean Wkdy/Wknd"),
                ("mean_weekday_vol", "Mean weekday vol"),
                ("on_suspect_list", "Prior suspect?"),
            ],
        )
    )
    lines.append("")

    lines.append("## Speeding (avg speed above posted limit)\n")
    lines.append(
        "Average speed exceeding the posted limit on residential/tertiary "
        "streets is a secondary cut-through signal — commuters under time "
        "pressure tend to speed."
    )
    lines.append("")
    for label, df in [("Weekday", speeders_wd), ("Weekend", speeders_we)]:
        lines.append(f"### Top 20 speeding segments — {label}\n")
        lines.append(
            _table(
                df,
                [
                    ("road_name", "Road"),
                    ("city_county_state", "City"),
                    ("road_class", "Class"),
                    ("speed_limit_mph", "Limit"),
                    ("avg_speed_mph", "Avg speed"),
                    ("speed_over_limit", "Over limit"),
                    ("avg_volume", "Avg volume"),
                ],
            )
        )
        lines.append("")

    lines.append("## Interactive maps\n")
    lines.append("Open these in a browser:\n")
    lines.append("- `reports/maps/volume_all_days.html`")
    lines.append("- `reports/maps/volume_weekdays.html`")
    lines.append("- `reports/maps/volume_weekend.html`")
    lines.append("- `reports/maps/weekday_weekend_ratio.html`")
    lines.append("- `reports/maps/cutthrough_suspects.html` _(residential + tertiary only)_")
    lines.append("")

    lines.append("## Headline findings\n")
    top_road = (
        suspect_summary.iloc[0] if not suspect_summary.empty else None
    )
    leonia_residential = piv[
        (piv["city_county_state"] == "Leonia, Bergen, New Jersey")
        & (piv["road_class"] == "residential")
    ]
    leonia_high_ratio = leonia_residential[
        leonia_residential["weekday_weekend_ratio"] > 1.5
    ]
    findings = []
    if top_road is not None:
        findings.append(
            f"- The street with the highest mean Weekday/Weekend ratio in the "
            f"top-30 list is **{top_road['road_name']}** (mean ratio "
            f"{top_road['mean_ratio']:.2f} across {int(top_road['n_segments'])} "
            f"segments), with mean weekday volume "
            f"{_fmt_int(top_road['mean_weekday_vol'])}."
        )
    findings.append(
        f"- In **Leonia itself**, {len(leonia_high_ratio)} of "
        f"{len(leonia_residential)} residential segments have weekday/weekend "
        f"ratio > 1.5."
    )
    speeders_top_wd = speeders_wd.head(3)
    if not speeders_top_wd.empty:
        names = ", ".join(
            f"{r['road_name']} (+{r['speed_over_limit']:.1f} mph)"
            for _, r in speeders_top_wd.iterrows()
        )
        findings.append(f"- Top weekday speeders: {names}.")
    lines.extend(findings)
    lines.append("")
    lines.append(
        "These are pure data-side heuristics. Phase 6 will combine these with "
        "UXsim trajectory analysis to confirm true cut-through behavior."
    )
    lines.append("")

    (REPORTS_DIR / "01_exploration.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
