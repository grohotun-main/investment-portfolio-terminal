# parsers/tail_risk.py
"""Extreme-Value-Theory tail model for the conditional further-fall. Fits a
Generalized Pareto Distribution to losses over a high threshold (Peaks-Over-
Threshold) and extrapolates tail-loss quantiles beyond the empirical sample.
Pure functions; no I/O. See the buy-the-dip design spec.

Losses are passed as POSITIVE magnitudes (e.g. -further_fall). Returned
quantiles are positive loss magnitudes; the caller negates for display.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

_MIN_EXCEEDANCES_CONFIDENT = 30


def fit_gpd_tail(losses, threshold_q: float = 0.90) -> dict:
    """POT-GPD fit to the upper tail of `losses` (positive magnitudes)."""
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 10:
        return {"xi": float("nan"), "beta": float("nan"), "threshold": float("nan"),
                "n_total": int(x.size), "n_exceedances": 0, "confident": False}
    u = float(np.quantile(x, threshold_q))
    exc = x[x > u] - u
    if exc.size < 5:
        return {"xi": float("nan"), "beta": float("nan"), "threshold": u,
                "n_total": int(x.size), "n_exceedances": int(exc.size), "confident": False}
    xi, _loc, beta = genpareto.fit(exc, floc=0.0)
    return {
        "xi": float(xi), "beta": float(beta), "threshold": u,
        "n_total": int(x.size), "n_exceedances": int(exc.size),
        "confident": exc.size >= _MIN_EXCEEDANCES_CONFIDENT,
    }


def tail_loss_quantile(fit: dict, p: float) -> float:
    """Loss magnitude at tail probability `p` (e.g. p=0.01 -> worst-1% loss),
    via the standard POT/GPD quantile. Returns NaN on a degenerate fit."""
    xi, beta, u = fit.get("xi"), fit.get("beta"), fit.get("threshold")
    n, nu = fit.get("n_total", 0), fit.get("n_exceedances", 0)
    if not (np.isfinite(xi) and np.isfinite(beta) and nu > 0 and n > 0):
        return float("nan")
    ratio = (n / nu) * p
    if abs(xi) < 1e-8:
        return float(u + beta * (-np.log(ratio)))
    return float(u + (beta / xi) * (ratio ** (-xi) - 1.0))
