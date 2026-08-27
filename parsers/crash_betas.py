"""Per-name crash beta vs SPY across a fixed set of historical windows.

Crash beta = median across (window-1, ..., window-N) of
  (per_name_return_over_window) / (spy_return_over_window)

Why median across windows (not pooled regression)? Two reasons:
  1. Sample size is tiny (4 windows). A single noisy beta from one
     idiosyncratic episode (e.g. an earnings miss inside the window)
     can dominate a least-squares fit; the median is robust.
  2. The user has stated preference for robust estimators (memory).

Windows where SPY didn't move (return ≈ 0) are skipped to avoid
divide-by-zero / sign-instability. A NaN beta is returned if all
windows are unusable.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SPY_ZERO_RETURN_EPS = 1e-4


def compute_window_return(price_series: pd.Series,
                          start: str | pd.Timestamp,
                          end: str | pd.Timestamp) -> float:
    """Compute total return over [start, end] for a price series.

    Forward-fills missing start/end dates to the next/prior available
    trading day. Returns NaN if no valid endpoints exist.
    """
    s = price_series.dropna()
    if s.empty:
        return float("nan")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # Slice [start, end]; the first observation >= start is the open,
    # the last observation <= end is the close.
    in_window = s.loc[(s.index >= start_ts) & (s.index <= end_ts)]
    if in_window.empty:
        return float("nan")
    open_px = float(in_window.iloc[0])
    close_px = float(in_window.iloc[-1])
    if open_px <= 0:
        return float("nan")
    return (close_px - open_px) / open_px


def compute_crash_betas(daily_prices: pd.DataFrame,
                         tickers: Iterable[str],
                         spy_ticker: str = "SPY",
                         windows: Iterable[tuple[str, str]] = (),
                         ) -> dict[str, float]:
    """For each ticker in `tickers`, return crash_beta = median across
    `windows` of (ticker_return / spy_return).

    Args:
        daily_prices: wide frame indexed by date, ticker columns.
        tickers: which tickers to compute beta for.
        spy_ticker: the SPY column name in daily_prices.
        windows: list of (start, end) date strings/timestamps.

    Returns a dict {ticker: float}; NaN for tickers absent from
    `daily_prices` or with no usable windows.
    """
    windows = list(windows)
    out: dict[str, float] = {}
    if spy_ticker not in daily_prices.columns:
        return {t: float("nan") for t in tickers}
    spy_series = daily_prices[spy_ticker]
    spy_window_returns = [
        compute_window_return(spy_series, s, e) for (s, e) in windows
    ]
    for ticker in tickers:
        if ticker not in daily_prices.columns:
            out[ticker] = float("nan")
            continue
        ticker_series = daily_prices[ticker]
        betas: list[float] = []
        for (s, e), spy_r in zip(windows, spy_window_returns):
            if not np.isfinite(spy_r) or abs(spy_r) < SPY_ZERO_RETURN_EPS:
                continue
            t_r = compute_window_return(ticker_series, s, e)
            if not np.isfinite(t_r):
                continue
            betas.append(t_r / spy_r)
        out[ticker] = float(np.median(betas)) if betas else float("nan")
    return out


def portfolio_crash_scenarios(daily_prices: pd.DataFrame,
                              weights: pd.Series,
                              windows: Iterable[tuple[str, str]],
                              spy_ticker: str = "SPY") -> dict:
    """Historical crash-window replay for a weighted book (v2-S4).

    implied_drop_pct per window = sum(w_hat_i * crash_beta_i) * spy_drop,
    where crash betas come from compute_crash_betas and w_hat are the
    weights RENORMALIZED over names with a finite beta (NaN-beta names are
    excluded and disclosed — never zero-filled). Windows whose SPY return
    is non-finite or >= 0 are skipped. A modelling replay, not a forecast."""
    windows = list(windows)
    w = weights.dropna() if weights is not None else pd.Series(dtype=float)
    if w.empty or spy_ticker not in getattr(daily_prices, "columns", []):
        # Nothing usable: every provided weight counts as excluded (0.0/0
        # when weights are empty — literally true, not fabricated).
        return {"available": False, "scenarios": [],
                "excluded_weight_pct": round(float(w.sum()) * 100, 1),
                "n_excluded": int(len(w))}
    betas = compute_crash_betas(daily_prices, tickers=list(w.index),
                                spy_ticker=spy_ticker, windows=windows)
    usable = {t: b for t, b in betas.items() if np.isfinite(b)}
    excluded_w = float(sum(float(w[t]) for t in w.index if t not in usable))
    total_usable = float(sum(float(w[t]) for t in usable))
    if total_usable <= 0:
        return {"available": False, "scenarios": [],
                "excluded_weight_pct": round(excluded_w * 100, 1),
                "n_excluded": int(len(w) - len(usable))}
    port_beta = sum(float(w[t]) / total_usable * b
                    for t, b in usable.items())
    spy_series = daily_prices[spy_ticker]
    scenarios = []
    for (s, e) in windows:
        spy_r = compute_window_return(spy_series, s, e)
        if not np.isfinite(spy_r) or spy_r >= 0:
            continue
        scenarios.append({"window": f"{s}→{e}",
                          "spy_drop_pct": round(spy_r * 100, 2),
                          "implied_drop_pct": round(port_beta * spy_r * 100,
                                                    2)})
    if not scenarios:
        return {"available": False, "scenarios": [],
                "excluded_weight_pct": round(excluded_w * 100, 1),
                "n_excluded": int(len(w) - len(usable))}
    return {"available": True, "scenarios": scenarios,
            "excluded_weight_pct": round(excluded_w * 100, 1),
            "n_excluded": int(len(w) - len(usable))}
