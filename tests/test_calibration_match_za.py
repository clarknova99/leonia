"""Tests for ``match_za_streets_to_links`` (Pass C.3)."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from leonia_traffic.network import calibration_match as cm


# UXsim link tuples are (name, from_node, to_node, lanes, ffs, length_deg)
# matching what OSMImporter produces. The matcher only reads via
# ``index_uxsim_links``.

def _fake_links() -> list:
    # UXsim link names parse as ``<name>-<osm_id>[-reverse][#n]`` —
    # see ``parse_uxsim_link_name``.
    return [
        ("ChristieHeights-9954832", "n1", "n2", 1, 11.176, 0.001),
        ("WillowTree-3356462", "n3", "n4", 1, 11.176, 0.001),
        ("MainSt-1234", "n5", "n6", 2, 22.352, 0.001),
    ]


def _fake_za_main() -> pd.DataFrame:
    return pd.DataFrame([
        # Christie Heights, Thu all-day Visitors → should match by OSM id
        {
            "zone_name": "Christie Heights Street / 9954832",
            "street_name": "Christie Heights Street",
            "osm_way_id": 9954832,
            "filter": "Visitors",
            "day_type_code": 4,
            "day_part_code": 0,
            "zone_volume": 642.0,
            "avg_trip_speed_mph": 23.0,
        },
        # Willow Tree, but with a fake/stale OSM id — must fall back to spatial
        {
            "zone_name": "Willow Tree Road / 99999999",
            "street_name": "Willow Tree Road",
            "osm_way_id": 99999999,
            "filter": "Visitors",
            "day_type_code": 4,
            "day_part_code": 0,
            "zone_volume": 1250.0,
            "avg_trip_speed_mph": 26.0,
        },
        # Same zone, Saturday — must NOT be included (day_type_code != 4)
        {
            "zone_name": "Christie Heights Street / 9954832",
            "street_name": "Christie Heights Street",
            "osm_way_id": 9954832,
            "filter": "Visitors",
            "day_type_code": 5,
            "day_part_code": 0,
            "zone_volume": 235.0,
            "avg_trip_speed_mph": 24.0,
        },
        # Resident row on Christie Heights — must NOT be included
        {
            "zone_name": "Christie Heights Street / 9954832",
            "street_name": "Christie Heights Street",
            "osm_way_id": 9954832,
            "filter": "Residents",
            "day_type_code": 4,
            "day_part_code": 0,
            "zone_volume": 410.0,
            "avg_trip_speed_mph": 22.0,
        },
    ])


def _fake_line_gdf() -> gpd.GeoDataFrame:
    """ZA line shapes — used only for spatial fallback."""
    return gpd.GeoDataFrame(
        [
            {
                "name": "Willow Tree Road / 99999999",
                "street_name": "Willow Tree Road",
                "osm_way_id": 99999999,
                "geometry": LineString([(-74.0, 40.86), (-74.0, 40.861)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


class _FakeNode:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _FakeLink:
    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.start_node = _FakeNode(x, y)
        self.end_node = _FakeNode(x, y + 0.0001)


class _FakeWorld:
    """Just enough World to satisfy ``build_link_midpoint_gdf``."""

    def __init__(self, links):
        self.LINKS = links

    def get_link(self, name: str):  # pragma: no cover - unused here
        for l in self.LINKS:
            if l.name == name:
                return l
        raise KeyError(name)


def _fake_world():
    # Place WillowTree's midpoint very close to the line geometry above.
    return _FakeWorld([
        _FakeLink("ChristieHeights-9954832", -74.005, 40.865),
        _FakeLink("WillowTree-3356462", -74.0, 40.8605),
        _FakeLink("MainSt-1234", -74.01, 40.870),
    ])


def test_match_za_streets_direct_osm_id_match():
    matched = cm.match_za_streets_to_links(
        _fake_world(),
        _fake_za_main(),
        _fake_links(),
        line_gdf=None,
    )
    assert not matched.empty
    assert "ChristieHeights-9954832" in matched.index
    row = matched.loc["ChristieHeights-9954832"]
    assert row["osm_way_id"] == 9954832
    assert row["observed_volume"] == 642.0
    assert row["street_name"] == "Christie Heights Street"
    assert row["source"] == "za_streets"


def test_match_za_streets_filters_filter_and_day():
    matched = cm.match_za_streets_to_links(
        _fake_world(),
        _fake_za_main(),
        _fake_links(),
        line_gdf=None,
    )
    if "ChristieHeights-9954832" in matched.index:
        assert matched.loc["ChristieHeights-9954832", "observed_volume"] == 642.0


def test_match_za_streets_spatial_fallback_for_stale_osm_id():
    matched = cm.match_za_streets_to_links(
        _fake_world(),
        _fake_za_main(),
        _fake_links(),
        line_gdf=_fake_line_gdf(),
        spatial_fallback_max_distance_m=500.0,
    )
    assert "WillowTree-3356462" in matched.index
    assert matched.loc["WillowTree-3356462", "observed_volume"] == 1250.0
    assert matched.loc["WillowTree-3356462", "source"] == "za_streets"


def test_match_za_streets_returns_empty_on_empty_input():
    matched = cm.match_za_streets_to_links(
        _fake_world(),
        pd.DataFrame(),
        _fake_links(),
    )
    assert matched.empty


def test_match_za_streets_no_match_when_osm_id_absent_and_no_fallback():
    df = _fake_za_main()
    # Mangle every OSM id so nothing matches directly
    df["osm_way_id"] = 11111111
    matched = cm.match_za_streets_to_links(
        _fake_world(),
        df,
        _fake_links(),
        line_gdf=None,  # disable fallback
    )
    assert matched.empty


def test_match_za_streets_real_data_smoke():
    """Smoke-test against the real export if it's present."""
    from leonia_traffic.data.za_streets_loader import (
        discover_za_streets,
        load_za_line_shapes,
        load_za_main,
    )

    if discover_za_streets() is None:
        import pytest
        pytest.skip("ZA streets export not present")

    za = load_za_main()
    line_gdf = load_za_line_shapes()
    # Use the line geometries as a stand-in for UXsim links so we can
    # at least confirm the matcher returns rows on real data. We turn
    # each line into a fake UXsim link with its OSM id.
    links = []
    for _, row in line_gdf.iterrows():
        if pd.isna(row.get("osm_way_id")):
            continue
        way_id = int(row["osm_way_id"])
        name = f"{row['street_name']}-{way_id}"
        # Tiny synthetic length; we don't simulate.
        links.append((name, f"n{way_id}a", f"n{way_id}b", 1, 11.176, 0.001))

    if not links:
        import pytest
        pytest.skip("No usable line shapes")

    matched = cm.match_za_streets_to_links(
        _fake_world(), za, links, line_gdf=None,
        day_type_code=4, day_part_code=0,
    )
    assert not matched.empty
    assert (matched["source"] == "za_streets").all()
    # Christie Heights way id 9954832 should be in there
    assert (matched["osm_way_id"] == 9954832).any()
