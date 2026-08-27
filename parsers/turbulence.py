"""Financial Turbulence (Kritzman & Li, 2010): Mahalanobis distance of a daily
cross-asset return vector vs its historical mean/covariance — "is today a normal
wiggle or an abnormal regime?" Plus a single-asset realized-volatility regime
classifier used to condition the dip stats over deep history. Pure; no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Dip-card regime conditioning: classify each day by the asset's own trailing
# realized vol; "stressed" = vol in the top (1-q) of the asset's own history.
DIP_VOL_REGIME_WINDOW = 21      # one trading month
DIP_VOL_REGIME_Q = 0.80         # top-quintile vol = stressed


def turbulence_index(returns: pd.DataFrame) -> pd.Series:
    """Per-day Mahalanobis distance vs the full-sample mean/covariance."""
    R = returns.dropna()
    if len(R) < 2 or R.shape[1] == 0:
        return pd.Series(dtype=float)
    mu = R.mean().to_numpy()
    cov = np.cov(R.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(np.atleast_2d(cov))
    d = R.to_numpy() - mu
    md = np.einsum("ij,jk,ik->i", d, inv, d)
    return pd.Series(md, index=R.index)


def turbulence_now(returns: pd.DataFrame) -> dict:
    """Latest turbulence value, its percentile, and a regime label."""
    t = turbulence_index(returns)
    if t.empty:
        return {"value": float("nan"), "percentile": float("nan"), "regime": "unknown"}
    cur = float(t.iloc[-1])
    if not np.isfinite(cur) or t.nunique() <= 1:
        return {"value": cur, "percentile": float("nan"), "regime": "calm"}
    pct = float((t <= cur).mean() * 100.0)
    regime = "abnormal" if pct >= 90 else "elevated" if pct >= 75 else "calm"
    return {"value": cur, "percentile": pct, "regime": regime}


def vol_regime(
    price: pd.Series,
    window: int = DIP_VOL_REGIME_WINDOW,
    q: float = DIP_VOL_REGIME_Q,
) -> pd.Series:
    """Label each day calm/stressed by trailing realized volatility.

    Threshold = the q-th percentile of the in-sample rolling annualized vol.
    Deep-history compatible (single-asset), so it can condition the dip stats
    over the full price history where the multi-asset turbulence index has no
    data. Returns a Series of "calm"/"stressed" strings aligned to price.index;
    the first `window` entries are "calm" (NaN vol < any finite threshold).
    """
    ret = price.astype(float).pct_change(fill_method=None)
    rv = ret.rolling(window).std() * np.sqrt(252)
    thr = float(rv.quantile(q))
    if not np.isfinite(thr) or thr <= 0.0:
        return pd.Series("calm", index=price.index, dtype=object)
    labels = np.where(rv >= thr, "stressed", "calm")
    return pd.Series(labels, index=price.index, dtype=object)
