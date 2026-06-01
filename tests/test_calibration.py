"""Tests for the GEH statistic and scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from leonia_traffic.simulation.calibration import (
    geh,
    geh_array,
    score_simulation,
)


def test_geh_identical_is_zero():
    assert math.isclose(geh(100, 100), 0.0)


def test_geh_symmetric():
    assert math.isclose(geh(100, 200), geh(200, 100))


def test_geh_known_value():
    # GEH(100, 200) = sqrt(2 * 100^2 / 300) = sqrt(66.67) ~= 8.165
    assert math.isclose(geh(100, 200), math.sqrt(2 * 100 ** 2 / 300), rel_tol=1e-9)


def test_geh_zero_pair_is_nan():
    assert math.isnan(geh(0, 0))


def test_geh_array_vectorized():
    out = geh_array(np.array([100.0, 200.0, 0.0]), np.array([100.0, 100.0, 0.0]))
    assert math.isclose(out[0], 0.0)
    assert math.isclose(out[1], geh(200, 100))
    assert math.isnan(out[2])


def test_score_simulation_with_perfect_match():
    matched = pd.DataFrame(
        {
            "observed_volume": [1000.0, 2000.0, 3000.0],
            "osm_way_id": [1, 2, 3],
        },
        index=["a", "b", "c"],
    )
    matched.index.name = "uxsim_link_name"
    # observed_to_hourly = 0.05 -> observed_vph = [50, 100, 150]
    sim_flow = pd.Series({"a": 50.0, "b": 100.0, "c": 150.0})
    score, df = score_simulation(
        sim_flow, matched, observed_to_hourly_factor=0.05, min_observed=10.0
    )
    assert score.n_links_scored == 3
    assert math.isclose(score.geh_mean, 0.0)
    assert math.isclose(score.pct_lt_5, 1.0)


def test_score_simulation_drops_low_volume():
    matched = pd.DataFrame(
        {"observed_volume": [10.0, 5000.0], "osm_way_id": [1, 2]},
        index=["a", "b"],
    )
    matched.index.name = "uxsim_link_name"
    sim_flow = pd.Series({"a": 1.0, "b": 250.0})
    score, df = score_simulation(
        sim_flow, matched, observed_to_hourly_factor=0.05, min_observed=100.0
    )
    assert score.n_links_scored == 1
    assert df.index.tolist() == ["b"]
