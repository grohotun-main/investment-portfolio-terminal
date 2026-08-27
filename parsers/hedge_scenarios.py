"""Pure math for the Options Hedging Phase 2 scenarios table.

Given a portfolio's equity/cash split, per-name weights, and per-name
crash betas, invert any target portfolio drawdown into an implied SPY
drop and per-name drops. No I/O.
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd


class HoldingClass(str, Enum):
    EQUITY = "equity"
    CASH_EQUIVALENT = "cash_equivalent"


# Ultra-short Treasury / money-market ETFs treated as cash. Conservative
# list — only the symbols actually held in this portfolio (plus the
# common neighbors). Anything else falls through to EQUITY.
_CASH_EQUIVALENT_TICKERS = {
    # ultra-short Treasury ETFs
    "SGOV", "BIL", "SHV", "GOVT", "TLH",
    # Fidelity money-market sweep symbols
    "SPAXX", "FZDXX", "FDRXX",
    # CDs / treasuries appear as multi-character strings in some statements;
    # those rows are excluded from the equity slice upstream in the
    # synthesize_interim_positions pipeline. This set covers the public
    # tickers; the caller may augment it.
}


def classify_holdings(ticker: str) -> HoldingClass:
    """Classify a single ticker as EQUITY or CASH_EQUIVALENT.

    Defaults to EQUITY — anything not explicitly cash-equivalent is
    treated as risky. Caller is responsible for upstream filtering of
    things like raw CD CUSIPs that have no ticker.
    """
    t = (ticker or "").strip().upper()
    if t in _CASH_EQUIVALENT_TICKERS:
        return HoldingClass.CASH_EQUIVALENT
    return HoldingClass.EQUITY


def invert_portfolio_drawdown(*,
                              target_drawdown: float,
                              equity_weight: float,
                              weights_in_equity: pd.Series,
                              crash_betas: Mapping[str, float],
                              ) -> float:
    """Given a target portfolio drawdown, return the implied SPY drop.

    Math:
      portfolio_drawdown = equity_weight × equity_drawdown
      equity_drawdown = Σ weights_in_equity[i] × beta[i] × spy_drop
                     = spy_drop × Σ weights_in_equity[i] × beta[i]
                     = spy_drop × weighted_avg_beta_in_equity

    Therefore:
      spy_drop = (target_drawdown / equity_weight) / weighted_avg_beta_in_equity

    Names with no usable crash beta (absent from `crash_betas`, or a NaN
    value — e.g. a listing newer than the most recent crash window) are
    *excluded* and the remaining equity weights renormalized over the kept
    names, rather than nuking the whole result. A single brand-new position
    should not blind the user to the entire scenarios table. The exclusion is
    not silent: callers surface it via `equity_names_without_crash_history`,
    so the renormalization is a disclosed transform (this supersedes the
    earlier strict-NaN-propagation contract).

    Returns NaN when equity_weight ≤ 0, when weights_in_equity is empty, when
    NO name has a usable crash beta, or when |weighted_avg_beta| < 1e-6
    (degenerate division).

    `weights_in_equity` should sum to 1.0 across the equity slice (caller is
    responsible for that initial normalization; the kept-name renormalization
    below is layered on top).
    """
    if equity_weight <= 0:
        return float("nan")
    if weights_in_equity.empty:
        return float("nan")
    kept = [t for t in weights_in_equity.index
            if t in crash_betas and np.isfinite(float(crash_betas[t]))]
    if not kept:
        return float("nan")
    w = weights_in_equity.loc[kept].astype(float)
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or abs(w_sum) < 1e-12:
        return float("nan")
    w = w / w_sum  # renormalize over the kept (has-crash-history) names
    b = pd.Series({t: float(crash_betas[t]) for t in kept})
    weighted_beta = float((w * b).sum(skipna=False))
    if not np.isfinite(weighted_beta) or abs(weighted_beta) < 1e-6:
        return float("nan")
    equity_drawdown = target_drawdown / equity_weight
    return equity_drawdown / weighted_beta


def equity_names_without_crash_history(*,
                                       weights_in_equity: pd.Series,
                                       crash_betas: Mapping[str, float],
                                       ) -> list[tuple[str, float]]:
    """Equity names that `invert_portfolio_drawdown` excludes for lack of a
    usable crash beta (absent from `crash_betas` or a NaN value).

    Returns ``[(ticker, weight_in_equity), ...]`` sorted by weight descending,
    where ``weight_in_equity`` is the name's pre-renormalization share of the
    equity slice. Callers render this as a caption so the renormalization the
    inverter performs is disclosed rather than hidden. Empty list when every
    name has a usable crash beta.
    """
    out: list[tuple[str, float]] = []
    for t in weights_in_equity.index:
        b = crash_betas.get(t, float("nan"))
        if not np.isfinite(float(b)):
            out.append((t, float(weights_in_equity.loc[t])))
    out.sort(key=lambda tw: tw[1], reverse=True)
    return out


def per_name_scenario_drops(*,
                             spy_drop: float,
                             crash_betas: Mapping[str, float],
                             ) -> dict[str, float]:
    """For a given SPY drop, scale per-name drops by crash beta."""
    return {t: spy_drop * float(b) for t, b in crash_betas.items()}
