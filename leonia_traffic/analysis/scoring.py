"""Engine-neutral GEH scoring primitives.

The GEH statistic compares a modelled flow ``M`` to an observed count
``C``::

    GEH(M, C) = sqrt( 2 (M - C)^2 / (M + C) )

A link is conventionally "well matched" when GEH < 5. These helpers are
pure NumPy/pandas and carry no simulation-engine dependency, so both the
SUMO scorer (:mod:`leonia_traffic.sumo.scoring`) and any analytics code
can reuse them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CalibrationScore",
    "geh",
    "geh_array",
    "score_simulation_by_source",
]


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
    """Vectorised GEH over arrays of (simulated, observed) flows."""
    s = np.asarray(simulated, dtype=float)
    o = np.asarray(observed, dtype=float)
    denom = s + o
    out = np.full_like(s, np.nan, dtype=float)
    mask = denom > 0
    out[mask] = np.sqrt(2.0 * (s[mask] - o[mask]) ** 2 / denom[mask])
    return out


@dataclass
class CalibrationScore:
    """Summary GEH statistics for one scored run (or one source within it)."""

    geh_mean: float
    geh_median: float
    geh_p85: float
    pct_lt_5: float
    pct_lt_10: float
    n_links_scored: int


def score_simulation_by_source(
    scored_df: pd.DataFrame,
) -> dict[str, CalibrationScore]:
    """Break a scored DataFrame down by its ``source`` column.

    Returns a ``{source_label: CalibrationScore}`` map. If the DataFrame
    is empty or has no ``source`` column, returns an empty dict.
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
