"""Per-position return contribution (approximate Brinson-lite).

contribution_pp = current_weight × window_total_return × 100.

Deliberate approximations, disclosed to every consumer:
- TOTAL returns from the loaders' adjusted daily closes (distributions
  reinvested at the ex-date, split-scaled — parsers/total_return.py); a
  symbol without a dividend file stays price-only there.
- CURRENT weights held constant across the window (mid-window buys/sells
  smear; a full transaction-based attribution is a different engine).
- "ytd" slices each series by ITS OWN latest year: a stale-priced name
  can contribute a prior-year window (such names usually fall to the
  <2-observation exclusion anyway).
Symbols absent from daily_prices (or with <2 in-window observations) are
EXCLUDED, never zero-filled (the B3 ratio-honesty rule); their combined
weight rides in the result frame's ``attrs["excluded_weight_pct"]``.
NaN-WEIGHT symbols are dropped too, but their magnitude is unknowable, so
they are disclosed as a COUNT in ``attrs["n_dropped_nan_weights"]``.
"""
from __future__ import annotations

import pandas as pd


def _window_slice(s: pd.Series, spec) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    if spec == "ytd":
        idx = pd.DatetimeIndex(pd.to_datetime(s.index))
        return s[idx.year == idx.year.max()]
    return s.tail(int(spec))


def position_return_contribution(daily_prices: pd.DataFrame,
                                 weights: pd.Series,
                                 windows: dict) -> dict:
    """{window_label: DataFrame[symbol × (weight_pct, return_pct,
    contrib_pp)] sorted by contrib_pp desc, attrs[excluded_weight_pct]}."""
    out: dict = {}
    w = weights.dropna() if weights is not None else pd.Series(dtype=float)
    n_nan = int(weights.isna().sum()) if weights is not None else 0
    for label, spec in windows.items():
        rows, excluded_w = [], 0.0
        for sym, wt in w.items():
            col = daily_prices.get(sym) if not daily_prices.empty else None
            win = _window_slice(col, spec) if col is not None else pd.Series(
                dtype=float)
            if len(win) < 2 or float(win.iloc[0]) <= 0:
                excluded_w += float(wt)
                continue
            r = float(win.iloc[-1]) / float(win.iloc[0]) - 1.0
            rows.append((str(sym), float(wt) * 100.0, r * 100.0,
                         float(wt) * r * 100.0))
        df = pd.DataFrame(rows, columns=["symbol", "weight_pct",
                                         "return_pct", "contrib_pp"])
        df = df.set_index("symbol").sort_values("contrib_pp",
                                                ascending=False)
        df.attrs["excluded_weight_pct"] = excluded_w * 100.0
        df.attrs["n_dropped_nan_weights"] = n_nan
        out[label] = df
    return out
