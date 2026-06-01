"""Tests for OSM builder utilities (no network calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from leonia_traffic.network.osm_builder import (
    NetworkOverrides,
    _postprocess_network,
    apply_overrides,
    parse_uxsim_link_name,
)


def test_parse_uxsim_link_name_simple():
    assert parse_uxsim_link_name("Broad Avenue-42508899") == (
        "Broad Avenue",
        42508899,
        False,
    )


def test_parse_uxsim_link_name_with_disambiguator():
    assert parse_uxsim_link_name("Broad Avenue-42508899#3") == (
        "Broad Avenue",
        42508899,
        False,
    )


def test_parse_uxsim_link_name_reverse():
    name, osmid, is_rev = parse_uxsim_link_name("Broad Avenue-42508899-reverse")
    assert osmid == 42508899
    assert is_rev


def test_parse_uxsim_link_name_no_osm_id():
    name, osmid, is_rev = parse_uxsim_link_name("Some Random Label")
    assert osmid is None
    assert name == "Some Random Label"
    assert not is_rev


def test_parse_uxsim_link_name_empty():
    assert parse_uxsim_link_name("") == ("", None, False)


def _make_toy_network():
    # 4 nodes in a line, 3 short links + 1 long link with a duplicate name.
    nodes = [
        [1, 0.0, 0.0],
        [2, 0.0001, 0.0],   # very close to 1 — will merge under threshold
        [3, 0.01, 0.0],
        [4, 0.02, 0.0],
    ]
    links = [
        ["Foo-100", 1, 2, 1, 11.0],   # short, will be merged out
        ["Foo-100", 2, 3, 1, 11.0],   # duplicate name
        ["Bar-200", 3, 4, 1, 11.0],
    ]
    return nodes, links


def test_postprocess_merges_short_links():
    nodes, links = _make_toy_network()
    new_nodes, new_links = _postprocess_network(
        nodes, links, node_merge_threshold=0.0005, node_merge_iteration=2,
    )
    # Node 1 and 2 are 0.0001 apart -> should merge.
    assert len(new_nodes) < len(nodes)
    # All link records should be 6-tuples.
    assert all(len(l) == 6 for l in new_links)


def test_postprocess_assigns_unique_names():
    nodes, links = _make_toy_network()
    # Add a third Foo-100 link with a different path so duplicates survive merging.
    links.append(["Foo-100", 4, 3, 1, 11.0])
    _, new_links = _postprocess_network(
        nodes, links, node_merge_threshold=1e-9, node_merge_iteration=1,
    )
    names = [l[0] for l in new_links]
    assert len(names) == len(set(names)), f"duplicate names: {names}"
    # OSM ID survives the disambiguator.
    for name in names:
        _, osmid, _ = parse_uxsim_link_name(name)
        assert osmid in (100, 200), f"OSM id parse failed for {name!r}"


def test_overrides_from_yaml_empty(tmp_path: Path):
    p = tmp_path / "ov.yaml"
    p.write_text("")
    ov = NetworkOverrides.from_yaml(p)
    assert ov.is_empty()


def test_overrides_apply_delete_link(tmp_path: Path):
    nodes = [[1, 0.0, 0.0], [2, 0.01, 0.0], [3, 0.02, 0.0]]
    links = [
        ["Foo-100", 1, 2, 1, 11.0, 0.01],
        ["Bar-200", 2, 3, 1, 11.0, 0.01],
    ]
    ov_path = tmp_path / "ov.yaml"
    ov_path.write_text(yaml.safe_dump({"delete_links": [{"osm_way_id": 100}]}))
    ov = NetworkOverrides.from_yaml(ov_path)
    _, new_links = apply_overrides(nodes, links, ov)
    assert len(new_links) == 1
    assert "Bar-200" in new_links[0][0]


def test_overrides_apply_set_link_attrs():
    nodes = [[1, 0.0, 0.0], [2, 0.01, 0.0]]
    links = [["Foo-100", 1, 2, 1, 11.0, 0.01]]
    ov = NetworkOverrides(
        set_link_attrs=[{"osm_way_id": 100, "free_flow_speed": 5.0, "lanes": 2}]
    )
    _, new_links = apply_overrides(nodes, links, ov)
    assert new_links[0][3] == 2          # lanes
    assert abs(new_links[0][4] - 5.0) < 1e-9  # free_flow_speed
