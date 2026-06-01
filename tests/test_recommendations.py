"""Tests for the recommendation engine, focused on Pass-C extensions."""

from __future__ import annotations

import pandas as pd

from leonia_traffic.analysis import recommendations as rec


def _empty_od_inputs() -> dict[str, pd.DataFrame]:
    """Empty frames in the shapes the engine expects for non-Pass-C rules."""
    empty = pd.DataFrame()
    return {
        "peak_imbalance_df": empty,
        "circuity_df": empty,
        "delay_df": empty,
        "summary_df": empty,
        "exposure_df": empty,
    }


def _make_per_street_df() -> pd.DataFrame:
    """Three synthetic streets — one strong, one weak, one speeding-only."""
    return pd.DataFrame([
        # Strong residential cut-through: should fire both index and speeding
        {
            "zone_name": "Christie Heights Street / 1",
            "street_name": "Christie Heights Street",
            "osm_way_id": 1,
            "cutthrough_index": 0.62,
            "weekday_all_day_volume": 800.0,
            "weekday_weekend_ratio": 2.7,
            "non_local_home_share": 0.55,
            "long_trip_share_5mi": 0.70,
            "speeding_share": 0.45,
        },
        # Low-index, low-volume — should not fire either rule
        {
            "zone_name": "Quiet Place / 2",
            "street_name": "Quiet Place",
            "osm_way_id": 2,
            "cutthrough_index": 0.15,
            "weekday_all_day_volume": 80.0,
            "weekday_weekend_ratio": 0.9,
            "non_local_home_share": 0.20,
            "long_trip_share_5mi": 0.10,
            "speeding_share": 0.10,
        },
        # Speeding-only: high speeding, but low cut-through index and
        # below the non-local threshold. Should fire speeding callout
        # but not the primary residential rule.
        {
            "zone_name": "Race Street / 3",
            "street_name": "Race Street",
            "osm_way_id": 3,
            "cutthrough_index": 0.35,
            "weekday_all_day_volume": 600.0,
            "weekday_weekend_ratio": 1.5,
            "non_local_home_share": 0.30,
            "long_trip_share_5mi": 0.40,
            "speeding_share": 0.60,
        },
    ])


def test_residential_cutthrough_fires_on_high_index_street():
    per_street = _make_per_street_df()
    recs = rec.generate_recommendations(per_street_df=per_street, **_empty_od_inputs())
    rules = [r.rule for r in recs]
    targets = [r.target for r in recs]
    assert "residential_cutthrough_candidate" in rules
    assert "Christie Heights Street" in targets


def test_residential_cutthrough_skips_low_index_street():
    per_street = _make_per_street_df()
    recs = rec.generate_recommendations(per_street_df=per_street, **_empty_od_inputs())
    cutthrough = [r for r in recs if r.rule == "residential_cutthrough_candidate"]
    quiet_targets = [r.target for r in cutthrough if r.target == "Quiet Place"]
    assert not quiet_targets, "Quiet Place must not fire the cut-through rule"


def test_residential_cutthrough_severity_is_high():
    per_street = _make_per_street_df()
    recs = rec.generate_recommendations(per_street_df=per_street, **_empty_od_inputs())
    cutthrough = [r for r in recs if r.rule == "residential_cutthrough_candidate"]
    assert all(r.severity == "high" for r in cutthrough)


def test_residential_speeding_callout_fires():
    per_street = _make_per_street_df()
    recs = rec.generate_recommendations(per_street_df=per_street, **_empty_od_inputs())
    speeding = [r for r in recs if r.rule == "residential_speeding_callout"]
    targets = {r.target for r in speeding}
    # Both Christie Heights and Race Street exceed speeding + volume thresholds
    assert "Race Street" in targets
    assert all(r.severity == "medium" for r in speeding)
    # Quiet Place must not fire
    assert "Quiet Place" not in targets


def test_residential_rules_silent_when_per_street_df_missing():
    recs = rec.generate_recommendations(**_empty_od_inputs(), per_street_df=None)
    assert all(r.rule != "residential_cutthrough_candidate" for r in recs)
    assert all(r.rule != "residential_speeding_callout" for r in recs)


def test_residential_rules_silent_when_per_street_df_empty():
    recs = rec.generate_recommendations(
        per_street_df=pd.DataFrame(),
        **_empty_od_inputs(),
    )
    assert all(r.rule != "residential_cutthrough_candidate" for r in recs)


def test_residential_rules_skip_when_required_columns_missing():
    bad = pd.DataFrame([
        {"street_name": "X", "osm_way_id": 1, "cutthrough_index": 0.7},
    ])
    recs = rec.generate_recommendations(per_street_df=bad, **_empty_od_inputs())
    # Missing non_local_home_share and weekday_all_day_volume → silent skip
    assert all(r.rule != "residential_cutthrough_candidate" for r in recs)


def test_existing_signature_keeps_working_without_per_street():
    # No per_street_df kwarg at all — must not raise.
    recs = rec.generate_recommendations(**_empty_od_inputs())
    assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# Arterial-channeling rules (Broad / Grand / Fort Lee Rd)
# ---------------------------------------------------------------------------


def _make_attribution_df() -> pd.DataFrame:
    """Two streets: one local (Christie Heights), one arterial (Broad Ave)."""
    return pd.DataFrame([
        {
            "middle_zone": "Christie Heights Street / 1 / 1",
            "middle_label": "Christie Heights Street",
            "middle_osm_way_id": 1,
            "total_omd_vph": 200.0,
            "n_od_pairs": 5,
            "bridge_share": 0.65,
            "top_origin_label": "Hoefley's Lane Gate",
            "top_destination_label": "George Washington Bridge",
            "top_od_pair_volume": 90.0,
            "high_circuity_share": 0.35,
        },
        {
            "middle_zone": "Broad Avenue / 99 / 1",
            "middle_label": "Broad Avenue",
            "middle_osm_way_id": 99,
            "total_omd_vph": 800.0,
            "n_od_pairs": 12,
            "bridge_share": 0.70,
            "top_origin_label": "Fort Lee Rd Gate",
            "top_destination_label": "George Washington Bridge",
            "top_od_pair_volume": 300.0,
            "high_circuity_share": 0.40,
        },
    ])


def test_arterial_strategy_always_fires():
    recs = rec.generate_recommendations(**_empty_od_inputs())
    assert any(r.rule == "channel_to_county_arterials" for r in recs)
    strat = [r for r in recs if r.rule == "channel_to_county_arterials"][0]
    assert strat.severity == "high"
    assert "Broad" in strat.rationale and "Fort Lee" in strat.rationale


def test_local_to_arterial_diversion_fires_for_local_streets():
    attribution = _make_attribution_df()
    recs = rec.generate_recommendations(
        **_empty_od_inputs(), cutthrough_attribution_df=attribution,
    )
    divert = [r for r in recs if r.rule == "divert_local_to_arterial"]
    targets = {r.target for r in divert}
    assert "Christie Heights Street" in targets
    # Broad Ave is an arterial — must NOT be a diversion target.
    assert "Broad Avenue" not in targets


def test_arterial_targets_downgraded_to_info():
    """An OMD-confirmed rule targeting Broad Ave should be reclassified."""
    attribution = _make_attribution_df()
    recs = rec.generate_recommendations(
        **_empty_od_inputs(), cutthrough_attribution_df=attribution,
    )
    # The omd_confirmed_cutthrough rule would normally fire HIGH for
    # Broad Ave (bridge_share 0.70, circuity 0.40, vph 800). After
    # reclassification it should be info-only with the monitor suffix.
    broad_recs = [r for r in recs if r.target == "Broad Avenue"]
    assert broad_recs, "Broad Ave should still appear in some form"
    assert all(r.severity == "info" for r in broad_recs)
    assert all("__arterial_monitor" in r.rule for r in broad_recs)
    assert all(r.metrics.get("jurisdiction") == "Bergen County / NJDOT"
               for r in broad_recs)


def test_diversion_skips_low_volume_local_streets():
    df = pd.DataFrame([{
        "middle_zone": "Tiny St / 5 / 1",
        "middle_label": "Tiny Street",
        "middle_osm_way_id": 5,
        "total_omd_vph": 10.0,            # below threshold
        "n_od_pairs": 1,
        "bridge_share": 0.80,
        "top_origin_label": "X", "top_destination_label": "GWB",
        "top_od_pair_volume": 10.0,
        "high_circuity_share": 0.30,
    }])
    recs = rec.generate_recommendations(
        **_empty_od_inputs(), cutthrough_attribution_df=df,
    )
    assert all(r.rule != "divert_local_to_arterial" for r in recs)


def test_is_county_state_arterial_helper():
    from leonia_traffic.analysis.jurisdiction import is_county_state_arterial
    assert is_county_state_arterial("Broad Avenue")
    assert is_county_state_arterial("Broad Ave")
    assert is_county_state_arterial("Grand Avenue")
    assert is_county_state_arterial("Fort Lee Rd")
    assert is_county_state_arterial("Fort Lee Road / 123 / 1")
    # Main Street in Leonia is the local signing of Fort Lee Road
    # (Bergen CR 9) — must be treated as a county arterial too.
    assert is_county_state_arterial("Main Street")
    assert is_county_state_arterial("Main St")
    assert is_county_state_arterial("Main Street / 13031696 / 1")
    assert not is_county_state_arterial("Christie Heights Street")
    assert not is_county_state_arterial("")
    assert not is_county_state_arterial(None)
