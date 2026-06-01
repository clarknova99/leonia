"""Tests for per-residential-street cut-through analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from leonia_traffic.analysis import cutthrough_streets as cs
from leonia_traffic.data import za_streets_loader as zl


def _need_export():
    return zl.discover_za_streets() is None


def test_weekday_weekend_imbalance_christie_heights_high():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.weekday_weekend_imbalance(zl.load_za_main())
    assert not df.empty
    ch = df[df["street_name"] == "Christie Heights Street"]
    assert not ch.empty
    # Known signature: weekday >> Saturday on Christie Heights
    assert ch.iloc[0]["weekday_weekend_ratio"] >= 2.0


def test_peak_am_volume_returns_one_row_per_zone():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.peak_am_volume(zl.load_za_main())
    assert not df.empty
    assert df["zone_name"].is_unique
    assert (df["peak_am_volume"] >= 0).all()


def test_weekday_all_day_volume_positive():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.weekday_all_day_volume(zl.load_za_main())
    assert not df.empty
    assert (df["weekday_all_day_volume"] > 0).mean() > 0.95


def test_long_trip_share_columns_and_ranges():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.long_trip_share(zl.load_za_trip())
    assert not df.empty
    for c in ("long_trip_share_5mi", "long_trip_share_10mi"):
        assert c in df.columns
        vals = df[c].dropna()
        assert (vals >= 0).all() and (vals <= 1.0 + 1e-6).all()
    # The >5 share must be >= the >10 share for every zone
    assert (df["long_trip_share_5mi"] + 1e-9 >= df["long_trip_share_10mi"]).all()


def test_speeding_share_25mph_default():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.speeding_share(zl.load_za_trip())
    assert not df.empty
    assert "speeding_share" in df.columns
    vals = df["speeding_share"].dropna()
    assert (vals >= 0).all() and (vals <= 1.0 + 1e-6).all()


def test_non_local_home_share_sums_make_sense():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.non_local_home_share(zl.load_za_home_distance())
    assert not df.empty
    # home_le_3mi_share + non_local_home_share should be ~1 (sum of all
    # distance bins)
    s = df["home_le_3mi_share"].fillna(0) + df["non_local_home_share"].fillna(0)
    assert ((s > 0.95) & (s < 1.05)).mean() > 0.95


def test_non_leonia_zip_share_christie_heights():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.non_leonia_zip_share(zl.load_za_home_zips_top())
    if df.empty:
        pytest.skip("home_zip_codes_top not present")
    ch = df[df["street_name"] == "Christie Heights Street"]
    assert not ch.empty
    # Leonia ZIP share on Christie Heights is known to be ~0.36;
    # so non-Leonia share should be substantial.
    assert ch.iloc[0]["non_leonia_zip_share"] >= 0.3


def test_composite_cutthrough_index_bounds_and_ranking():
    if _need_export():
        pytest.skip("ZA streets export not present")
    za = zl.load_za_main()
    trip = zl.load_za_trip()
    home_dist = zl.load_za_home_distance()
    idx = cs.composite_cutthrough_index(
        imbalance_df=cs.weekday_weekend_imbalance(za),
        weekday_volume_df=cs.weekday_all_day_volume(za),
        long_trip_df=cs.long_trip_share(trip),
        speeding_df=cs.speeding_share(trip),
        home_dist_df=cs.non_local_home_share(home_dist),
    )
    assert not idx.empty
    assert (idx["cutthrough_index"] >= 0).all()
    assert (idx["cutthrough_index"] <= 1.0 + 1e-6).all()
    assert idx["rank"].iloc[0] == 1
    assert idx["rank"].is_monotonic_increasing


def test_top_origin_zips_returns_ranked_rows():
    if _need_export():
        pytest.skip("ZA streets export not present")
    home_zips = zl.load_za_home_zips_top()
    if home_zips.empty:
        pytest.skip("home_zip_codes_top not present")
    # Find any Christie Heights zone name
    zones = home_zips["zone_name"].dropna().unique()
    ch_zone = next((z for z in zones if "Christie Heights" in str(z)), None)
    if ch_zone is None:
        pytest.skip("Christie Heights zone not in zips file")
    top = cs.top_origin_zips(home_zips, ch_zone, n=10)
    assert not top.empty
    assert top["rank"].is_monotonic_increasing
    assert top["pct_home_location"].iloc[0] > 0


def test_composite_empty_inputs():
    empty = pd.DataFrame()
    out = cs.composite_cutthrough_index(
        imbalance_df=empty,
        weekday_volume_df=empty,
        long_trip_df=empty,
        speeding_df=empty,
        home_dist_df=empty,
    )
    assert out.empty


def test_peak_pm_volume_returns_one_row_per_zone():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.peak_pm_volume(zl.load_za_main())
    assert not df.empty
    assert df["zone_name"].is_unique
    assert (df["peak_pm_volume"] >= 0).all()


def test_peak_hour_intensity_columns_and_noise_floor():
    if _need_export():
        pytest.skip("ZA streets export not present")
    za = zl.load_za_main()
    df = cs.peak_hour_intensity(za, day_types=(0,))
    if df.empty:
        pytest.skip("no hourly day_type=0 rows present")
    for c in ("peak_total", "peak_per_hr", "baseline_total",
              "baseline_per_hr", "peak_intensity"):
        assert c in df.columns
    valid = df.dropna(subset=["peak_intensity"])
    assert (valid["peak_intensity"] > 0).all()
    assert (valid["peak_intensity"] < 1e3).all()
    assert (valid["peak_per_hr"] >= 5.0 - 1e-9).all()
    assert (valid["baseline_per_hr"] >= 5.0 - 1e-9).all()


def test_peak_hour_intensity_birch_lane_signature():
    if _need_export():
        pytest.skip("ZA streets export not present")
    za = zl.load_za_main()
    df = cs.peak_hour_intensity(za, day_types=(0,))
    if df.empty:
        pytest.skip("no hourly rows present")
    birch = df[df["street_name"] == "Birch Lane"]
    if birch.empty or birch["peak_intensity"].isna().all():
        pytest.skip("Birch Lane peak intensity not measurable in export")
    # Known: Birch Lane in current export has ~7-8× peak-AM vs midday;
    # require at least 3× to flag a cut-through signal.
    assert birch["peak_intensity"].dropna().iloc[0] >= 3.0


def test_weekday_hourly_profile_has_24_hour_columns():
    if _need_export():
        pytest.skip("ZA streets export not present")
    df = cs.weekday_hourly_profile(zl.load_za_main())
    if df.empty:
        pytest.skip("no hourly rows present")
    hours = [c for c in df.columns if c.startswith("h") and len(c) == 3]
    assert len(hours) == 24
    assert hours[0] == "h00" and hours[-1] == "h23"
    vals = df[hours].to_numpy()
    assert (vals[~pd.isna(vals)] >= 0).all()


def test_peak_intensity_empty_input():
    empty = pd.DataFrame()
    assert cs.peak_hour_intensity(empty).empty
    assert cs.peak_pm_volume(empty).empty
    assert cs.weekday_hourly_profile(empty).empty
