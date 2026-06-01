"""Integration tests for :class:`leonia_traffic.sumo.runtime.SumoRuntime`.

``libsumo``'s C++ binding registers a competing pyarrow-filesystem
factory at import time, which permanently breaks ``pyarrow`` in the
hosting Python process. To avoid poisoning the rest of the suite we
run every libsumo-touching assertion in a **fresh subprocess** via
:mod:`subprocess`. The main pytest process only reads the worker's
exit status and stdout; it never imports libsumo itself.

Tests are gated on:

* ``libsumo`` being importable (skipped otherwise);
* the canonical ``leonia.net.xml`` existing (skipped otherwise — run
  ``scripts/11_export_sumo.py`` first).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NET_PATH = _REPO_ROOT / "data" / "processed" / "sumo" / "leonia.net.xml"


def _libsumo_available() -> bool:
    try:
        # Trial import in a subprocess so we don't break pyarrow here.
        proc = subprocess.run(
            [sys.executable, "-c", "import libsumo"],
            capture_output=True, text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _libsumo_available(),
                       reason="libsumo not importable"),
    pytest.mark.skipif(not _NET_PATH.exists(),
                       reason="SUMO net file missing — run "
                              "scripts/11_export_sumo.py"),
]


def _run_in_subprocess(snippet: str) -> subprocess.CompletedProcess:
    """Execute ``snippet`` in a fresh Python process rooted at the repo."""
    full = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_REPO_ROOT)!r})
        {textwrap.indent(textwrap.dedent(snippet), "        ").strip()}
    """)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        cwd=str(_REPO_ROOT),
    )


def test_runtime_starts_and_steps():
    res = _run_in_subprocess("""
        from leonia_traffic.sumo import DemandSource, SumoRuntime

        rt = SumoRuntime.start(
            demand=DemandSource.PEAK_AM_SLICE,
            sample_interval_s=300,
            end_time_s=8 * 3600,
            seed=1,
        )
        try:
            assert rt.backend_name in {"libsumo", "traci"}
            assert rt.sim_time_s() == 0.0
            rt.run_until(7 * 3600 + 30 * 60)
            assert rt.sim_time_s() >= 7 * 3600 + 30 * 60
            cnt = rt.edge_counters()
            assert "sumo_edge_id" in cnt.columns
            assert len(cnt) > 0
            print("OK rows={}".format(len(cnt)))
        finally:
            rt.close()
    """)
    assert res.returncode == 0, (
        f"subprocess failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    )
    assert "OK rows=" in res.stdout


def test_runtime_produces_history_and_summary():
    res = _run_in_subprocess("""
        from leonia_traffic.sumo import DemandSource, SumoRuntime

        rt = SumoRuntime.start(
            demand=DemandSource.PEAK_AM_SLICE,
            sample_interval_s=300,
            end_time_s=8 * 3600,
            seed=1,
        )
        try:
            rt.run_to_end()
            hist = rt.edge_history()
            summary = rt.edge_summary()
        finally:
            rt.close()
        for col in ("t_bin_s", "sumo_edge_id", "vehicles", "mean_speed_ms"):
            assert col in hist.columns, col
        for col in ("sumo_edge_id", "peak_vph", "mean_speed_mph"):
            assert col in summary.columns, col
        assert (summary["peak_vph"].fillna(0) >= 0).all()
        print("OK history={} summary={}".format(len(hist), len(summary)))
    """)
    assert res.returncode == 0, (
        f"subprocess failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    )
    assert "OK history=" in res.stdout


def test_apply_closure_then_restore_round_trips():
    res = _run_in_subprocess("""
        from leonia_traffic.sumo import DemandSource, SumoRuntime

        rt = SumoRuntime.start(
            demand=DemandSource.PEAK_AM_SLICE,
            sample_interval_s=300,
            end_time_s=8 * 3600,
            seed=1,
        )
        try:
            rt.run_until(7 * 3600 + 5 * 60)
            any_way = next(iter(rt.osm_lookup.keys()))
            edges_before = list(rt.osm_lookup[any_way])
            affected = rt.apply_closure([any_way])
            assert affected, "apply_closure should affect >=1 edge"
            assert set(affected) <= set(edges_before)
            for eid in affected:
                # libsumo.edge has no getMaxSpeed; max speed lives on lanes.
                spd = rt._backend.lane.getMaxSpeed(eid + "_0")
                assert abs(spd - 0.1) < 1e-3, spd
            n = rt.restore([any_way])
            assert n == len(affected)
            for eid in affected:
                assert rt._backend.lane.getMaxSpeed(eid + "_0") > 1.0
            print("OK affected={}".format(len(affected)))
        finally:
            rt.close()
    """)
    assert res.returncode == 0, (
        f"subprocess failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    )
    assert "OK affected=" in res.stdout
