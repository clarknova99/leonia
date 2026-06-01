"""Tests for visitor-demographic ZIP-only analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.analysis import visitor_demographics as vd
from leonia_traffic.data import za_streets_loader as zl


def _need_export():
    return zl.discover_za_streets() is None


@pytest.fixture(scope="module")
def home_zips_top():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = zl.load_za_home_zips_top()
    if df.empty:
        pytest.skip("home_zip_codes_top not present")
    return df


@pytest.fixture(scope="module")
def work_bg():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = zl.load_za_work_block_groups()
    if df.empty:
        pytest.skip("work_block_groups not present")
    return df


@pytest.fixture(scope="module")
def tourist_summary():
    if _need_export():
        pytest.skip("ZA streets export not present")
    return zl.load_za_tourist_summary()


def test_municipality_for_zip_known_values():
    assert vd.municipality_for_zip("07605") == "Leonia"
    assert vd.municipality_for_zip(7605) == "Leonia"  # int → padded
    assert vd.municipality_for_zip("07024") == "Fort Lee"
    assert vd.municipality_for_zip("07026").startswith("Garfield")


def test_municipality_for_zip_fallback_buckets():
    assert vd.municipality_for_zip("07999") == "Other NJ"
    assert vd.municipality_for_zip("10000") == "Other NY"
    assert vd.municipality_for_zip("90210") == "Other"
    assert vd.municipality_for_zip(None) == "Unknown"


def test_origin_municipality_breakdown_christie_heights(home_zips_top):
    zones = home_zips_top["zone_name"].dropna().unique()
    ch_zone = next((z for z in zones if "Christie Heights" in str(z)), None)
    if ch_zone is None:
        pytest.skip("Christie Heights zone not in zips file")
    df = vd.origin_municipality_breakdown(home_zips_top, ch_zone)
    assert not df.empty
    assert df.iloc[0]["municipality"] == "Leonia"
    next_two = set(df["municipality"].iloc[1:3])
    assert any("Fort Lee" == m or "Englewood" == m or "Garfield" in m or "Edgewater" == m
               or "Cliffside Park" == m or "Palisades Park" == m for m in next_two)


def test_origin_municipality_breakdown_overall_aggregation(home_zips_top):
    df = vd.origin_municipality_breakdown(home_zips_top, zone_name=None)
    assert not df.empty
    assert (df["share"] >= 0).all()
    assert df["share"].sum() == pytest.approx(1.0, rel=0.01) or df["share"].sum() < 1.05


def test_state_split_columns(tourist_summary):
    df = vd.state_split(tourist_summary)
    if df.empty:
        pytest.skip("tourist_summary not present")
    for c in ("in_state_share", "out_of_state_share",
              "local_metro_share", "other_metro_share"):
        assert c in df.columns
    # In-state share for Leonia residential streets should dominate
    assert (df["in_state_share"].dropna() > 0.7).mean() > 0.8


def test_origin_municipality_breakdown_handles_empty():
    out = vd.origin_municipality_breakdown(pd.DataFrame(), "anything")
    assert out.empty


def test_county_label_known_values():
    assert vd.county_label("34003") == "Bergen, NJ"
    assert vd.county_label("36061") == "Manhattan, NY"
    assert vd.county_label("09001") == "Fairfield, CT"


def test_county_label_state_fallbacks():
    # Unknown NJ county → "Other NJ"
    assert vd.county_label("34999") == "Other NJ"
    # Unknown NY county → "Other NY"
    assert vd.county_label("36999") == "Other NY"
    # Unknown state → "Other"
    assert vd.county_label("48999") == "Other"
    assert vd.county_label(None) == "Unknown"


def test_county_label_strips_quotes_and_pads():
    # StreetLight occasionally wraps FIPS in single-quotes; loader strips
    # them but the label function should be robust either way.
    assert vd.county_label("'34003'") == "Bergen, NJ"
    # Int input padded to 5 chars
    assert vd.county_label(34003) == "Bergen, NJ"


def test_work_destination_breakdown_cross_zone(work_bg):
    df = vd.work_destination_breakdown(work_bg, zone_name=None, top_n=15)
    assert not df.empty
    assert df.iloc[0]["county_label"] == "Bergen, NJ"
    assert (df["share"] >= 0).all()
    assert (df["share"] <= 1.0 + 1e-6).all()
    top_labels = set(df["county_label"].head(6))
    assert any("NJ" in lab for lab in top_labels)
    assert any("NY" in lab for lab in top_labels)


def test_work_destination_breakdown_per_zone_willow_tree(work_bg):
    zones = work_bg["zone_name"].dropna().unique()
    wt = next((z for z in zones if "Willow Tree" in str(z)), None)
    if wt is None:
        pytest.skip("Willow Tree zone not present")
    df = vd.work_destination_breakdown(work_bg, zone_name=wt, top_n=5)
    assert not df.empty
    assert df.iloc[0]["county_label"] == "Bergen, NJ"


def test_top_work_tracts(work_bg):
    zones = work_bg["zone_name"].dropna().unique()
    wt = next((z for z in zones if "Willow Tree" in str(z)), None)
    if wt is None:
        pytest.skip("Willow Tree zone not present")
    df = vd.top_work_tracts(work_bg, wt, n=5)
    assert not df.empty
    assert "tract" in df.columns
    assert "county_label" in df.columns
    assert df.iloc[0]["county_label"] == "Bergen, NJ"
    assert df["tract"].str.len().eq(11).all()


def test_work_destination_breakdown_handles_empty():
    out = vd.work_destination_breakdown(pd.DataFrame())
    assert out.empty
    out = vd.top_work_tracts(pd.DataFrame(), "anything")
    assert out.empty
