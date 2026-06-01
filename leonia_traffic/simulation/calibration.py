"""GEH-based calibration of the placeholder gateway demand model.

The calibration tunes a small number of global parameters to minimize a
GEH-derived loss between simulated and observed link flows.

GEH statistic
-------------
    GEH(M, C) = sqrt( 2 (M - C)^2 / (M + C) )

A model is considered "well calibrated" when ≥85% of matched links have
GEH < 5. We score with the percentile and the mean for a smooth loss.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from uxsim import World

from leonia_traffic.network.osm_builder import OSMBuildConfig
from leonia_traffic.simulation.world_factory import build_baseline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GEH statistic
# ---------------------------------------------------------------------------


def geh(simulated: float, observed: float) -> float:
    """Compute the GEH statistic for one observation pair."""
    if simulated is None or observed is None:
        return float("nan")
    s = float(simulated)
    o = float(observed)
    if s + o <= 0:
        return float("nan")
    return math.sqrt(2.0 * (s - o) ** 2 / (s + o))


def geh_array(simulated: np.ndarray, observed: np.ndarray) -> np.ndarray:
    s = np.asarray(simulated, dtype=float)
    o = np.asarray(observed, dtype=float)
    denom = s + o
    out = np.full_like(s, np.nan, dtype=float)
    mask = denom > 0
    out[mask] = np.sqrt(2.0 * (s[mask] - o[mask]) ** 2 / denom[mask])
    return out


# ---------------------------------------------------------------------------
# Link-flow extraction
# ---------------------------------------------------------------------------


def extract_simulated_flows(
    W: World,
    *,
    t_start_s: float | None = None,
    t_end_s: float | None = None,
) -> pd.Series:
    """Return simulated flow (veh/h) per UXsim link over the analysis window.

    Uses ``arrival_count(t)`` which is the cumulative vehicle count
    through time ``t``. Flow is ``(arrivals(t_end) - arrivals(t_start)) /
    duration_h``.
    """
    if t_start_s is None:
        t_start_s = 0.0
    if t_end_s is None:
        t_end_s = float(W.TMAX)
    duration_h = max((t_end_s - t_start_s) / 3600.0, 1e-6)

    rows: dict[str, float] = {}
    for link in W.LINKS:
        try:
            arrivals_end = float(link.arrival_count(t_end_s))
            arrivals_start = (
                float(link.arrival_count(t_start_s)) if t_start_s > 0 else 0.0
            )
            n = max(arrivals_end - arrivals_start, 0.0)
            rows[link.name] = n / duration_h
        except Exception:
            rows[link.name] = 0.0
    return pd.Series(rows, name="sim_flow_vph")


# ---------------------------------------------------------------------------
# Score a calibration run
# ---------------------------------------------------------------------------


@dataclass
class CalibrationScore:
    geh_mean: float
    geh_median: float
    geh_p85: float
    pct_lt_5: float
    pct_lt_10: float
    n_links_scored: int


def score_simulation_by_source(
    scored_df: pd.DataFrame,
) -> dict[str, CalibrationScore]:
    """Break a scored DataFrame down by ``source`` column.

    Returns a ``{source_label: CalibrationScore}`` map. If the
    DataFrame has no ``source`` column or only one source, returns a
    single-entry dict keyed by ``"all"``.
    """
    if scored_df is None or scored_df.empty:
        return {}
    if "source" not in scored_df.columns:
        return {}

    out: dict[str, CalibrationScore] = {}
    for src, sub in scored_df.groupby("source", dropna=False):
        if pd.isna(src) or src is None:
            src = "unknown"
        sub = sub.dropna(subset=["geh"])
        if sub.empty:
            continue
        out[str(src)] = CalibrationScore(
            geh_mean=float(sub["geh"].mean()),
            geh_median=float(sub["geh"].median()),
            geh_p85=float(sub["geh"].quantile(0.85)),
            pct_lt_5=float((sub["geh"] < 5).mean()),
            pct_lt_10=float((sub["geh"] < 10).mean()),
            n_links_scored=int(len(sub)),
        )
    return out


def score_simulation(
    sim_flow: pd.Series,
    matched: pd.DataFrame,
    *,
    observed_to_hourly_factor: float = 0.10 / 2.0,
    min_observed: float = 50.0,
) -> tuple[CalibrationScore, pd.DataFrame]:
    """Score one simulation run vs. observed flows.

    ``observed_to_hourly_factor`` converts the StreetLight "average
    volume" (daily total) to vehicles/hour matching the simulated
    window. Default assumes 10 % AM-peak factor over a 2-h window.
    """
    df = matched.copy()
    df = df.dropna(subset=["observed_volume"])
    df = df[df["observed_volume"] >= min_observed]
    df["observed_vph"] = df["observed_volume"] * observed_to_hourly_factor
    df["sim_vph"] = df.index.map(sim_flow).astype(float).fillna(0.0)
    df["geh"] = geh_array(df["sim_vph"].to_numpy(), df["observed_vph"].to_numpy())

    df = df.dropna(subset=["geh"])
    score = CalibrationScore(
        geh_mean=float(df["geh"].mean()),
        geh_median=float(df["geh"].median()),
        geh_p85=float(df["geh"].quantile(0.85)),
        pct_lt_5=float((df["geh"] < 5).mean()),
        pct_lt_10=float((df["geh"] < 10).mean()),
        n_links_scored=int(len(df)),
    )
    return score, df


# ---------------------------------------------------------------------------
# Parameter optimizer
# ---------------------------------------------------------------------------


@dataclass
class CalibrationParams:
    """The free parameters Nelder-Mead optimizes."""

    daily_to_peak_factor: float = 0.10
    gwb_share: float = 0.6
    min_gateway_volume: float = 500.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.daily_to_peak_factor, self.gwb_share, self.min_gateway_volume],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CalibrationParams":
        return cls(
            daily_to_peak_factor=max(min(float(arr[0]), 0.5), 0.005),
            gwb_share=max(min(float(arr[1]), 1.0), 0.0),
            min_gateway_volume=max(float(arr[2]), 50.0),
        )


@dataclass
class CalibrationResult:
    best_params: CalibrationParams
    best_score: CalibrationScore
    history: list[dict] = field(default_factory=list)


def _objective(
    arr: np.ndarray,
    *,
    network_cfg: OSMBuildConfig | None,
    duration_hours: float,
    observed_to_hourly_factor_factory,
    history: list[dict],
    streetlight_source: str,
    tmax: int,
    deltan: int,
) -> float:
    params = CalibrationParams.from_array(arr)
    build = build_baseline(
        name=f"calib_iter_{len(history)}",
        tmax=tmax,
        deltan=deltan,
        network_cfg=network_cfg,
        streetlight_source=streetlight_source,
        duration_hours=duration_hours,
        daily_to_peak_factor=params.daily_to_peak_factor,
        gwb_share=params.gwb_share,
        min_gateway_volume=params.min_gateway_volume,
        print_mode=0,
    )
    W = build.world
    W.exec_simulation()

    sim_flow = extract_simulated_flows(W, t_start_s=0, t_end_s=W.TMAX)
    score, _ = score_simulation(
        sim_flow,
        build.matched,
        observed_to_hourly_factor=observed_to_hourly_factor_factory(params),
    )

    # Loss: combine mean GEH with a penalty for low "pct_lt_5".
    loss = score.geh_mean + 10.0 * (1.0 - score.pct_lt_5)
    history.append(
        {
            "iter": len(history),
            **params.__dict__,
            "geh_mean": score.geh_mean,
            "geh_p85": score.geh_p85,
            "pct_lt_5": score.pct_lt_5,
            "pct_lt_10": score.pct_lt_10,
            "n_links_scored": score.n_links_scored,
            "loss": loss,
        }
    )
    logger.info(
        "iter=%d loss=%.3f geh_mean=%.2f pct<5=%.1f%% params=%s",
        len(history),
        loss,
        score.geh_mean,
        100 * score.pct_lt_5,
        params.__dict__,
    )
    return float(loss)


def calibrate(
    *,
    network_cfg: OSMBuildConfig | None = None,
    initial: CalibrationParams | None = None,
    streetlight_source: str = "weekdays",
    duration_hours: float = 2.0,
    tmax: int = 2 * 3600,
    deltan: int = 10,
    maxiter: int = 30,
) -> CalibrationResult:
    """Run Nelder-Mead over a few demand parameters.

    Each iteration triggers a full UXsim simulation, so ``maxiter`` is
    intentionally modest. The simulation is ~30s on the Leonia network.
    """
    initial = initial or CalibrationParams()
    history: list[dict] = []

    def obs_factor(p: CalibrationParams) -> float:
        return p.daily_to_peak_factor / duration_hours

    def objective(arr):
        return _objective(
            arr,
            network_cfg=network_cfg,
            duration_hours=duration_hours,
            observed_to_hourly_factor_factory=obs_factor,
            history=history,
            streetlight_source=streetlight_source,
            tmax=tmax,
            deltan=deltan,
        )

    result = minimize(
        objective,
        initial.as_array(),
        method="Nelder-Mead",
        options={
            "maxiter": maxiter,
            "xatol": 1e-3,
            "fatol": 1e-2,
            "disp": True,
        },
    )

    best_params = CalibrationParams.from_array(result.x)
    best_score = CalibrationScore(
        geh_mean=history[-1]["geh_mean"],
        geh_median=float("nan"),
        geh_p85=history[-1]["geh_p85"],
        pct_lt_5=history[-1]["pct_lt_5"],
        pct_lt_10=history[-1]["pct_lt_10"],
        n_links_scored=history[-1]["n_links_scored"],
    )
    # Use the best history entry rather than the final one.
    if history:
        best = min(history, key=lambda h: h["loss"])
        best_params = CalibrationParams(
            daily_to_peak_factor=best["daily_to_peak_factor"],
            gwb_share=best["gwb_share"],
            min_gateway_volume=best["min_gateway_volume"],
        )
        best_score = CalibrationScore(
            geh_mean=best["geh_mean"],
            geh_median=float("nan"),
            geh_p85=best["geh_p85"],
            pct_lt_5=best["pct_lt_5"],
            pct_lt_10=best["pct_lt_10"],
            n_links_scored=best["n_links_scored"],
        )

    return CalibrationResult(best_params=best_params, best_score=best_score, history=history)


# ---------------------------------------------------------------------------
# Pass-B calibration: real OD demand + congestion-derived link speeds
# ---------------------------------------------------------------------------


@dataclass
class CalibrationParamsV2:
    """Reduced parameter space for the calibrated Pass-B baseline.

    With observed OD volumes and observed link free-flow speeds in
    place, only three global knobs remain:

    * ``od_demand_scale`` — global multiplier on OD volume; should land
      near 1.0 if the StreetLight OD numbers are well-calibrated.
    * ``jam_density_factor`` — multiplicative on UXsim's default
      ``Link.kappa`` (jam density per lane).
    * ``intersection_capacity_factor`` — multiplicative on each link's
      outflow capacity (``capacity_out``); proxies signalized-intersection
      throughput.
    """

    od_demand_scale: float = 1.0
    jam_density_factor: float = 1.0
    intersection_capacity_factor: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.od_demand_scale, self.jam_density_factor, self.intersection_capacity_factor],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CalibrationParamsV2":
        return cls(
            od_demand_scale=max(min(float(arr[0]), 5.0), 0.1),
            jam_density_factor=max(min(float(arr[1]), 3.0), 0.3),
            intersection_capacity_factor=max(min(float(arr[2]), 3.0), 0.3),
        )


@dataclass
class CalibrationResultV2:
    best_params: CalibrationParamsV2
    best_score: CalibrationScore
    history: list[dict] = field(default_factory=list)


def _apply_intersection_capacity(W: World, factor: float) -> None:
    if factor == 1.0:
        return
    for link in W.LINKS:
        cap = getattr(link, "capacity_out", None)
        if cap is None or not np.isfinite(cap):
            continue
        try:
            link.capacity_out = float(cap) * factor
        except Exception:  # pragma: no cover
            pass


def _objective_v2(
    arr: np.ndarray,
    *,
    network_cfg: OSMBuildConfig | None,
    duration_hours: float,
    day_type_code: int,
    day_part_code: int,
    history: list[dict],
    streetlight_source: str,
    tmax: int,
    deltan: int,
    observed_to_hourly_factor: float,
    include_za_streets_in_match: bool = False,
) -> float:
    from leonia_traffic.simulation.world_factory import build_calibrated_baseline

    params = CalibrationParamsV2.from_array(arr)
    build = build_calibrated_baseline(
        name=f"calib_v2_iter_{len(history)}",
        tmax=tmax,
        deltan=deltan,
        network_cfg=network_cfg,
        streetlight_source=streetlight_source,
        day_type_code=day_type_code,
        day_part_code=day_part_code,
        duration_hours=duration_hours,
        demand_scale=params.od_demand_scale,
        jam_density_factor=params.jam_density_factor,
        include_za_streets_in_match=include_za_streets_in_match,
        print_mode=0,
    )
    W = build.world
    _apply_intersection_capacity(W, params.intersection_capacity_factor)
    W.exec_simulation()

    sim_flow = extract_simulated_flows(W, t_start_s=0, t_end_s=W.TMAX)
    score, scored = score_simulation(
        sim_flow, build.matched,
        observed_to_hourly_factor=observed_to_hourly_factor,
    )

    loss = score.geh_mean + 10.0 * (1.0 - score.pct_lt_5)
    per_source = score_simulation_by_source(scored)
    history.append({
        "iter": len(history),
        **params.__dict__,
        "geh_mean": score.geh_mean,
        "geh_p85": score.geh_p85,
        "pct_lt_5": score.pct_lt_5,
        "pct_lt_10": score.pct_lt_10,
        "n_links_scored": score.n_links_scored,
        "loss": loss,
        "per_source": {
            src: {
                "geh_mean": s.geh_mean,
                "pct_lt_5": s.pct_lt_5,
                "n": s.n_links_scored,
            } for src, s in per_source.items()
        },
    })
    if per_source:
        per_source_str = ", ".join(
            f"{src}: n={s.n_links_scored} mean={s.geh_mean:.2f} "
            f"pct<5={s.pct_lt_5 * 100:.0f}%"
            for src, s in per_source.items()
        )
        logger.info(
            "iter=%d loss=%.3f geh_mean=%.2f pct<5=%.1f%% [%s] params=%s",
            len(history), loss, score.geh_mean, 100 * score.pct_lt_5,
            per_source_str, params.__dict__,
        )
    else:
        logger.info(
            "iter=%d loss=%.3f geh_mean=%.2f pct<5=%.1f%% params=%s",
            len(history), loss, score.geh_mean, 100 * score.pct_lt_5,
            params.__dict__,
        )
    return float(loss)


def calibrate_v2(
    *,
    network_cfg: OSMBuildConfig | None = None,
    initial: CalibrationParamsV2 | None = None,
    streetlight_source: str = "weekdays",
    day_type_code: int = 1,
    day_part_code: int = 2,
    duration_hours: float = 4.0,
    tmax: int = 4 * 3600,
    deltan: int = 10,
    maxiter: int = 25,
    observed_to_hourly_factor: float = 0.10 / 4.0,
    include_za_streets_in_match: bool = False,
) -> CalibrationResultV2:
    """Nelder-Mead calibration over the Pass-B parameter space.

    Each iteration triggers a full simulation including OD demand
    injection and congestion-derived speed overrides, so ``maxiter``
    should remain modest.
    """
    initial = initial or CalibrationParamsV2()
    history: list[dict] = []

    def objective(arr):
        return _objective_v2(
            arr,
            network_cfg=network_cfg,
            duration_hours=duration_hours,
            day_type_code=day_type_code,
            day_part_code=day_part_code,
            history=history,
            streetlight_source=streetlight_source,
            tmax=tmax,
            deltan=deltan,
            observed_to_hourly_factor=observed_to_hourly_factor,
            include_za_streets_in_match=include_za_streets_in_match,
        )

    result = minimize(
        objective,
        initial.as_array(),
        method="Nelder-Mead",
        options={
            "maxiter": maxiter,
            "xatol": 1e-2,
            "fatol": 1e-2,
            "disp": True,
        },
    )

    best_params = CalibrationParamsV2.from_array(result.x)
    if history:
        best = min(history, key=lambda h: h["loss"])
        best_params = CalibrationParamsV2(
            od_demand_scale=best["od_demand_scale"],
            jam_density_factor=best["jam_density_factor"],
            intersection_capacity_factor=best["intersection_capacity_factor"],
        )
        best_score = CalibrationScore(
            geh_mean=best["geh_mean"],
            geh_median=float("nan"),
            geh_p85=best["geh_p85"],
            pct_lt_5=best["pct_lt_5"],
            pct_lt_10=best["pct_lt_10"],
            n_links_scored=best["n_links_scored"],
        )
    else:
        best_score = CalibrationScore(
            geh_mean=float("nan"), geh_median=float("nan"),
            geh_p85=float("nan"), pct_lt_5=0.0, pct_lt_10=0.0,
            n_links_scored=0,
        )

    return CalibrationResultV2(best_params=best_params, best_score=best_score, history=history)


def calibrate_v3(
    *,
    network_cfg: OSMBuildConfig | None = None,
    initial: CalibrationParamsV2 | None = None,
    streetlight_source: str = "weekdays",
    day_type_code: int = 1,
    day_part_code: int = 2,
    duration_hours: float = 4.0,
    tmax: int = 4 * 3600,
    deltan: int = 10,
    maxiter: int = 25,
    observed_to_hourly_factor: float = 0.10 / 4.0,
) -> CalibrationResultV2:
    """Pass-C calibration: same parameter space as v2 but with Pass-C
    ZA-streets observations unioned into the scoring frame so
    calibration also pulls on residential link flows.
    """
    return calibrate_v2(
        network_cfg=network_cfg,
        initial=initial,
        streetlight_source=streetlight_source,
        day_type_code=day_type_code,
        day_part_code=day_part_code,
        duration_hours=duration_hours,
        tmax=tmax,
        deltan=deltan,
        maxiter=maxiter,
        observed_to_hourly_factor=observed_to_hourly_factor,
        include_za_streets_in_match=True,
    )


__all__ = [
    "CalibrationParams",
    "CalibrationParamsV2",
    "CalibrationResult",
    "CalibrationResultV2",
    "CalibrationScore",
    "calibrate",
    "calibrate_v2",
    "calibrate_v3",
    "extract_simulated_flows",
    "geh",
    "geh_array",
    "score_simulation",
    "score_simulation_by_source",
]
