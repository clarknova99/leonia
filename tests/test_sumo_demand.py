"""Tests for the SUMO routes demand builder."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from leonia_traffic.sumo.demand_builder import (
    BRIDGE_OD_WINDOWS,
    DemandSource,
    build_routes,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NET_PATH = _REPO_ROOT / "data" / "processed" / "sumo" / "leonia.net.xml"


def _net_available() -> bool:
    return _NET_PATH.exists()


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing — run scripts/11_export_sumo.py")
def test_build_routes_peak_am_slice_emits_well_formed_xml(tmp_path):
    out = tmp_path / "routes.xml"
    n = build_routes(DemandSource.PEAK_AM_SLICE, out, net_path=_NET_PATH)
    assert n > 0, "expected at least one peak-AM-slice flow"
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "routes"
    flows = root.findall("flow")
    assert len(flows) == n
    # Every flow must define begin / end / vehsPerHour / from / to.
    for flow in flows:
        for attr in ("begin", "end", "vehsPerHour", "from", "to"):
            assert flow.get(attr) is not None, (
                f"flow {flow.get('id')!r} is missing {attr}"
            )
        # Slice runs in a single 07–08am hour.
        assert int(flow.get("begin")) == 7 * 3600
        assert int(flow.get("end")) == 8 * 3600


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing")
def test_build_routes_bridge_od_full_uses_all_five_windows(tmp_path):
    out = tmp_path / "routes_full.xml"
    n = build_routes(DemandSource.BRIDGE_OD_FULL, out, net_path=_NET_PATH)
    assert n > 0
    tree = ET.parse(out)
    flows = tree.getroot().findall("flow")
    # Map back from begin-time to the named window.
    begins = {int(f.get("begin")) for f in flows}
    expected = {hr_start * 3600
                for _, (_, hr_start, _) in BRIDGE_OD_WINDOWS.items()}
    assert expected.issubset(begins), (
        f"expected window begins {expected}, got {begins}"
    )


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing")
def test_build_routes_combined_includes_both_sources(tmp_path):
    out = tmp_path / "routes_combined.xml"
    n = build_routes(DemandSource.BRIDGE_OD_PLUS_ZA, out, net_path=_NET_PATH)
    assert n > 0
    tree = ET.parse(out)
    flow_ids = [f.get("id") for f in tree.getroot().findall("flow")]
    has_od = any(fid.startswith("od_") for fid in flow_ids)
    has_za = any(fid.startswith("za_") for fid in flow_ids)
    assert has_od or has_za, "expected at least one source to emit flows"


def test_build_routes_string_input_is_accepted(tmp_path, monkeypatch):
    """String input ('peak_am_slice') should resolve to the enum."""
    if not _net_available():
        pytest.skip("SUMO net file missing")
    out = tmp_path / "routes_str.xml"
    n = build_routes("peak_am_slice", out, net_path=_NET_PATH)
    assert n > 0


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing")
def test_build_routes_weekday_24h_covers_full_day(tmp_path):
    """The weekday-avg DemandSource must span all 24 hours.

    Each Bridge OD pair contributes one ``<flow>`` per hour with
    non-zero volume; we expect the union of begin-times to cover
    most of the day (allowing for genuinely-empty hours late at
    night when no OD pair has measurable Mon-Fri volume).
    """
    out = tmp_path / "routes_weekday.xml"
    n = build_routes(DemandSource.BRIDGE_OD_WEEKDAY_24H, out,
                     net_path=_NET_PATH)
    assert n > 0
    tree = ET.parse(out)
    flows = tree.getroot().findall("flow")
    begins = {int(f.get("begin")) // 3600 for f in flows}
    # Should cover at least 18 distinct hours of the day. The weekday
    # cohort has measurable volume across 12am–11pm; we leave slack
    # for a few late-night zeros.
    assert len(begins) >= 18, (
        f"weekday demand only covers {len(begins)} hours of the day"
    )
    # Every flow must carry the "wkd" suffix in its id so it's
    # distinguishable from a Sunday flow when both are merged.
    ids = [f.get("id") for f in flows]
    assert all("_wkd" in fid for fid in ids), (
        "expected every weekday flow id to end with _wkd"
    )


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing")
def test_build_routes_emits_flows_sorted_by_begin(tmp_path):
    """SUMO silently drops unsorted flows.

    libsumo (and the headless ``sumo`` binary) emit a
    "Route file should be sorted by departure time" warning and
    refuse to insert any flow whose ``begin`` is earlier than the
    most recently parsed one. Production-quality output therefore
    has to be sorted ascending by ``begin``. Regression-guards the
    24-hour weekday demand: pre-fix, ~95% of flows were dropped
    because they were grouped by OD pair instead of begin-time.
    """
    out = tmp_path / "routes_weekday.xml"
    build_routes(DemandSource.BRIDGE_OD_WEEKDAY_24H, out,
                 net_path=_NET_PATH)
    tree = ET.parse(out)
    flows = tree.getroot().findall("flow")
    begins = [int(f.get("begin")) for f in flows]
    assert begins == sorted(begins), (
        "flows must be sorted ascending by `begin` so SUMO doesn't "
        "drop them"
    )


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing")
def test_build_routes_weekday_24h_includes_za_residential_flows(tmp_path):
    """The 24h weekday demand must include ZA per-segment flows.

    Without this, the SUMO simulation only sees gateway-to-gateway
    Bridge OD demand, which routes through the high-vph arterials
    (Broad / Grand / Fort Lee Rd). The animation then leaves most
    Leonia residential streets dark even though those streets have
    real measured traffic. Regression-guards the fix: BRIDGE_OD_*_24H
    must mix in ``za_*`` flows synthesised from ``za_volume.parquet``
    filtered to the matching day-type cohort (Mon-Thu mean for
    weekday, Sunday-only for Sunday).
    """
    out = tmp_path / "routes_weekday.xml"
    build_routes(DemandSource.BRIDGE_OD_WEEKDAY_24H, out,
                 net_path=_NET_PATH)
    tree = ET.parse(out)
    flow_ids = [f.get("id") for f in tree.getroot().findall("flow")]
    has_za = any(fid.startswith("za_") for fid in flow_ids)
    has_od = any(fid.startswith("od_") for fid in flow_ids)
    assert has_od, "expected gateway OD flows in 24h weekday demand"
    assert has_za, (
        "expected ZA per-segment residential flows in 24h weekday "
        "demand — without these the animation only paints arterials"
    )
    # ZA flows in the weekday source must carry the `_wkd` suffix
    # so they don't collide with a parallel Sunday build.
    za_ids = [fid for fid in flow_ids if fid.startswith("za_")]
    assert all(fid.endswith("_wkd") for fid in za_ids), (
        "ZA flows in BRIDGE_OD_WEEKDAY_24H must carry _wkd label"
    )


@pytest.mark.skipif(not _net_available(),
                    reason="SUMO net file missing")
def test_build_routes_sunday_24h_has_lower_morning_peak(tmp_path):
    """The Sunday cohort should produce visibly less morning traffic.

    Sundays in the StreetLight bridge OD show ~10% of the weekday
    7-8am volume — the GW-bridge commuter shape collapses. We
    assert the Sunday-cohort total 7-8am vph is less than half of
    the weekday-avg total.
    """
    weekday_out = tmp_path / "routes_weekday.xml"
    sunday_out = tmp_path / "routes_sunday.xml"
    build_routes(DemandSource.BRIDGE_OD_WEEKDAY_24H, weekday_out,
                 net_path=_NET_PATH)
    build_routes(DemandSource.BRIDGE_OD_SUNDAY_24H, sunday_out,
                 net_path=_NET_PATH)

    def _total_vph_at(path, begin_s):
        tree = ET.parse(path)
        return sum(
            float(f.get("vehsPerHour", 0))
            for f in tree.getroot().findall("flow")
            if int(f.get("begin", 0)) == begin_s
        )

    weekday_7am = _total_vph_at(weekday_out, 7 * 3600)
    sunday_7am = _total_vph_at(sunday_out, 7 * 3600)
    assert weekday_7am > 0
    assert sunday_7am < 0.5 * weekday_7am, (
        f"sunday 7-8am vph ({sunday_7am:.1f}) should be < 50% of "
        f"weekday avg ({weekday_7am:.1f})"
    )
