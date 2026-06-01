"""Tests for street_trend and cutthrough_attribution analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.analysis.cutthrough_attribution import (
    per_street_attribution,
    top_od_bypass_pairs,
)
from leonia_traffic.analysis.street_trend import (
    street_trend_metrics,
    worsening_streets,
)


# ---------------------------------------------------------------------------
# street_trend
# ---------------------------------------------------------------------------


def _make_trend_fixture():
    months = pd.date_range("2023-01-01", periods=30, freq="MS")
    rows = []
    # Street A: steady ~ 100/day
    for m in months:
        rows.append({
            "zone_name": "A / 1 / 1", "osm_name": "A St",
            "osm_way_id": 1, "year_month": m, "avg_volume": 100.0,
        })
    # Street B: doubling — climbs linearly from 50 to 200
    for i, m in enumerate(months):
        rows.append({
            "zone_name": "B / 2 / 1", "osm_name": "B Ave",
            "osm_way_id": 2, "year_month": m,
            "avg_volume": 50.0 + (150.0 * i / (len(months) - 1)),
        })
    return pd.DataFrame(rows)


def test_trend_metrics_columns():
    df = _make_trend_fixture()
    out = street_trend_metrics(df)
    for col in ("zone_name", "recent_12mo_avg", "baseline_12mo_avg",
                "yoy_change_pct", "trend_slope_per_year", "trend_r2"):
        assert col in out.columns


def test_trend_metrics_detects_growth():
    df = _make_trend_fixture()
    out = street_trend_metrics(df).set_index("zone_name")
    # Street A should be ~ 0% YoY; street B should be strongly positive.
    assert abs(out.loc["A / 1 / 1", "yoy_change_pct"]) < 1.0
    assert out.loc["B / 2 / 1", "yoy_change_pct"] > 20.0
    # B's linear fit should have r² near 1.
    assert out.loc["B / 2 / 1", "trend_r2"] > 0.95


def test_worsening_streets_filters():
    df = _make_trend_fixture()
    out = street_trend_metrics(df)
    w = worsening_streets(out, min_yoy_pct=15.0, min_recent_volume=30.0)
    names = set(w["zone_name"].tolist())
    assert "B / 2 / 1" in names
    assert "A / 1 / 1" not in names


def test_trend_metrics_empty():
    out = street_trend_metrics(pd.DataFrame())
    assert isinstance(out, pd.DataFrame)
    assert out.empty


# ---------------------------------------------------------------------------
# cutthrough_attribution
# ---------------------------------------------------------------------------


def _make_omd_fixture():
    rows = []
    # Street M1: 100% bridge-bound, two OD pairs
    rows.append({
        "middle_zone": "M1 / 10 / 1", "middle_label": "M1 St",
        "middle_osm_way_id": 10,
        "origin_zone": "O1", "origin_label": "Fort Lee Rd Gate",
        "destination_zone": "D1", "destination_label": "George Washington Bridge",
        "day_type_code": 0, "day_part_code": 0,
        "omd_volume": 60.0, "avg_travel_time_sec": 90.0,
    })
    rows.append({
        "middle_zone": "M1 / 10 / 1", "middle_label": "M1 St",
        "middle_osm_way_id": 10,
        "origin_zone": "O2", "origin_label": "Hoefley's Lane Gate",
        "destination_zone": "D1", "destination_label": "GWB",
        "day_type_code": 0, "day_part_code": 0,
        "omd_volume": 40.0, "avg_travel_time_sec": 80.0,
    })
    # Street M2: bridge share 50%
    rows.append({
        "middle_zone": "M2 / 11 / 1", "middle_label": "M2 St",
        "middle_osm_way_id": 11,
        "origin_zone": "O3", "origin_label": "Some Gate",
        "destination_zone": "D2", "destination_label": "George Washington Bridge",
        "day_type_code": 0, "day_part_code": 0,
        "omd_volume": 25.0, "avg_travel_time_sec": 70.0,
    })
    rows.append({
        "middle_zone": "M2 / 11 / 1", "middle_label": "M2 St",
        "middle_osm_way_id": 11,
        "origin_zone": "O3", "origin_label": "Some Gate",
        "destination_zone": "D3", "destination_label": "Englewood",
        "day_type_code": 0, "day_part_code": 0,
        "omd_volume": 25.0, "avg_travel_time_sec": 60.0,
    })
    return pd.DataFrame(rows)


def test_per_street_attribution_basic():
    df = _make_omd_fixture()
    out = per_street_attribution(df)
    assert len(out) == 2
    m1 = out[out["middle_zone"] == "M1 / 10 / 1"].iloc[0]
    m2 = out[out["middle_zone"] == "M2 / 11 / 1"].iloc[0]
    assert m1["total_omd_vph"] == pytest.approx(100.0)
    assert m1["bridge_share"] == pytest.approx(1.0)
    assert m1["n_od_pairs"] == 2
    assert m2["bridge_share"] == pytest.approx(0.5)


def test_per_street_attribution_empty():
    out = per_street_attribution(pd.DataFrame())
    assert out.empty


def test_top_od_bypass_pairs():
    df = _make_omd_fixture()
    out = top_od_bypass_pairs(df, min_volume=1.0)
    assert len(out) >= 3
    # Highest pair should be O1 → D1 (60).
    top = out.iloc[0]
    assert top["origin_zone"] == "O1"
    assert top["total_routed_vph"] == pytest.approx(60.0)


def test_top_od_bypass_min_volume():
    df = _make_omd_fixture()
    out = top_od_bypass_pairs(df, min_volume=50.0)
    # Only the 60-volume pair survives.
    assert len(out) == 1
