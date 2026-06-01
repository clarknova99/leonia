"""Tests for the OD cut-through analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.analysis import od_cutthrough as oc
from leonia_traffic.data import bridge_od_loader as bol


@pytest.fixture(scope="module")
def od_df() -> pd.DataFrame:
    return bol.load_bridge_od()


@pytest.fixture(scope="module")
def attr_df() -> pd.DataFrame:
    return bol.load_bridge_attributes()


def test_gateway_peak_imbalance_finds_fort_lee(od_df):
    out = oc.gateway_peak_imbalance(od_df)
    assert not out.empty
    expected_cols = {
        "origin_zone", "origin_label", "origin_osm_way_id",
        "peak_am_weekday_avg", "weekend_peak_am_avg",
        "peak_am_to_weekend_ratio",
    }
    assert expected_cols.issubset(out.columns)

    # Fort Lee Road / 590576 should be the top origin by Peak AM weekday volume.
    top = out.iloc[0]
    assert top["origin_osm_way_id"] == 590576
    # And its weekday/weekend ratio should be markedly > 1.
    assert top["peak_am_to_weekend_ratio"] > 3.0


def test_cutthrough_index_from_circuity_returns_per_gate_rows(attr_df):
    out = oc.cutthrough_index_from_circuity(attr_df)
    assert not out.empty
    assert {"cutthrough_circuity_index", "circuity_low_pct", "circuity_high_pct"}.issubset(out.columns)
    # Percent columns from StreetLight are 0..1 proportions.
    idx = out["cutthrough_circuity_index"].dropna()
    assert (idx >= 0).all()
    assert (idx <= 1.0 + 1e-6).all()


def test_day_of_week_profile_columns(od_df):
    out = oc.day_of_week_profile(od_df)
    for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
              "weekday_avg", "weekend_avg", "weekday_to_weekend_ratio"):
        assert d in out.columns

    # The commuter cut-through signal lives in the Peak-AM slice, not All Day.
    peak = oc.day_of_week_profile(od_df, day_part_code=oc.PEAK_AM_CODE)
    fl = peak[peak["origin_osm_way_id"] == 590576]
    assert not fl.empty
    assert fl["weekday_avg"].iloc[0] > 5 * fl["weekend_avg"].iloc[0]


def test_trip_purpose_decomposition_shares_sum_to_100(attr_df):
    out = oc.trip_purpose_decomposition(attr_df)
    assert not out.empty
    needed = {"home_to_work_pct", "home_to_other_pct", "non_home_based_pct"}
    assert needed.issubset(out.columns)

    # Each (gate, day_part) row's purpose shares should sum to ~100 when
    # there's any traffic in that window.
    nonzero = out[out["weekday_trips"] > 0].copy()
    nonzero["sum_pct"] = (
        nonzero["home_to_work_pct"].fillna(0)
        + nonzero["home_to_other_pct"].fillna(0)
        + nonzero["non_home_based_pct"].fillna(0)
    )
    # StreetLight purpose columns are normalised to 0..1 (proportion).
    assert (nonzero["sum_pct"].between(0.95, 1.05)).all()


def test_trip_purpose_decomposition_peak_am_skews_commuter(attr_df):
    """Peak-AM weekday trips should be Home-to-Work-heavy on the headline gate."""
    out = oc.trip_purpose_decomposition(attr_df)
    fl_peak = out[(out["origin_osm_way_id"] == 590576) & (out["day_part_code"] == oc.PEAK_AM_CODE)]
    assert not fl_peak.empty
    # Commuter share should be a material fraction (shares are 0..1).
    assert fl_peak["home_to_work_pct"].iloc[0] >= 0.25
