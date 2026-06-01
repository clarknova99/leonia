"""Tests for the O-D + Middle-Filter cut-through loader."""

from __future__ import annotations

import pytest

from leonia_traffic.data.cutthrough_omd_loader import (
    discover_cutthrough_omd,
    load_cutthrough_omd,
    load_cutthrough_omd_roster,
    load_cutthrough_omd_trips,
    load_cutthrough_omd_zone_activity,
)


@pytest.fixture(scope="module")
def omd_paths():
    paths = discover_cutthrough_omd()
    if paths is None:
        pytest.skip("cutthrough OMD folder not present")
    return paths


@pytest.fixture(scope="module")
def omd_df(omd_paths):
    return load_cutthrough_omd(omd_paths.folder)


def test_omd_nonempty(omd_df):
    assert len(omd_df) > 1000


def test_omd_has_triple_keys(omd_df):
    for col in ("origin_zone", "middle_zone", "destination_zone",
                "day_type_code", "day_part_code", "omd_volume"):
        assert col in omd_df.columns


def test_omd_volume_numeric(omd_df):
    assert omd_df["omd_volume"].dtype.kind in ("f", "i")
    assert (omd_df["omd_volume"].fillna(0) >= 0).all()


def test_omd_zone_role_normalised(omd_paths):
    zact = load_cutthrough_omd_zone_activity(omd_paths.folder)
    if zact is None or zact.empty:
        pytest.skip("zone activity table empty")
    roles = set(zact["zone_role"].dropna().str.lower().unique())
    assert "middle" in roles or "middle filter" not in roles


def test_omd_trips_has_shares(omd_paths):
    tdf = load_cutthrough_omd_trips(omd_paths.folder)
    assert "share_circuity_ge_3" in tdf.columns
    assert "share_trip_ge_5mi" in tdf.columns
    assert "share_speed_ge_30" in tdf.columns


def test_omd_roster_loads(omd_paths):
    r = load_cutthrough_omd_roster(omd_paths.folder)
    if r is None:
        pytest.skip("roster file missing")
    assert len(r) > 0
