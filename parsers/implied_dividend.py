"""Resolve the continuous dividend yield `q` for an options-chain expiry.

Background — why this module exists
-----------------------------------
Phase A's verification used put-call parity on a SINGLE most-liquid strike
per expiry to back-solve q. That gave nonsensical NEGATIVE q for QQQ
(−1.18%), NVDA (−1.90%), AAPL (−0.12%) — the result of one option pair's
bid-ask noise dominating a single-point solve. Vega ∝ S·e^(−qT)·φ(d1)·√T
is the most q-sensitive Greek, so vega was the only Greek that didn't
validate cleanly (72% vs 96-100% on the others).

This module fixes that with a multi-tier solver:

  1. **PCP-median** — solve put-call parity at EVERY liquid near-ATM
     strike, take the median q across them. Bid-ask noise on individual
     strikes averages out; the systematic q is what's left. Returns this
     if the result lands in a sane band.
  2. **Hardcoded trailing yield** — fall back to a per-ticker constant
     when PCP can't produce something sane. These are trailing-12mo SEC
     yields for ETFs / forward-annualized declared dividends for single
     names, sourced once from public yield data and re-checked annually.
  3. **Zero** — terminal fallback if the ticker is unknown.

Reusable by both the verifier and Phase B's Monte Carlo (where each
underlying needs a single yield for the projection horizon).

Schema expected for `chain_df`: one row per contract, columns
`contract_type`, `strike`, `polygon_price`, `polygon_open_interest`,
`underlying_price`, `dte`. (Matches what fetch_options_chains.py emits.)
"""
from __future__ import annotations

import math
from typing import Literal, TypedDict

import numpy as np
import pandas as pd


YieldMethod = Literal["pcp-median", "hardcoded", "zero"]


class YieldResult(TypedDict):
    q: float
    method: YieldMethod
    n_strikes: int


# Trailing-12mo distribution yield (ETFs) / forward-annualized declared
# dividend (single names). Sourced from issuer disclosure pages, last
# refreshed 2026-05-24. If the chain produces a sane PCP-median, that
# wins — these are only the fallback. Re-check yearly.
HARDCODED_YIELDS: dict[str, float] = {
    "SPY":  0.013,
    "QQQ":  0.005,
    "NVDA": 0.0003,
    "AAPL": 0.0045,
}

# Band the PCP-median must land in to be accepted. Slightly negative is
# allowed (legitimate borrow-cost premium). Above 8% is suspicious for the
# tickers in scope here and signals bad data. Tune if extending to REITs
# or utility ETFs.
SANE_Q_LOW  = -0.005
SANE_Q_HIGH =  0.08

# Strike-selection knobs.
ATM_BAND        = 0.10   # keep strikes within ±10% of spot (most liquid)
MIN_OI_PER_LEG  = 10     # both call and put need at least this much OI
MIN_STRIKES_PCP = 3      # need this many usable pairs for median to mean anything
MAX_Q_FOR_INCL  = 0.50   # per-strike sanity gate — drop pairs giving |q|>50%


def _pcp_median(chain_df: pd.DataFrame, r: float) -> YieldResult | None:
    """Median of put-call-parity-implied q across near-ATM liquid strikes.

    Returns None if fewer than MIN_STRIKES_PCP usable pairs were found, or
    if the resulting median lands outside [SANE_Q_LOW, SANE_Q_HIGH].
    """
    if chain_df.empty:
        return None
    spot = chain_df["underlying_price"].iloc[0]
    dte  = chain_df["dte"].iloc[0]
    if pd.isna(spot) or pd.isna(dte) or spot <= 0 or dte <= 0:
        return None
    T = float(dte) / 365.0

    calls = chain_df[chain_df["contract_type"] == "call"].set_index("strike")
    puts  = chain_df[chain_df["contract_type"] == "put"].set_index("strike")
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        return None

    lo, hi = float(spot) * (1.0 - ATM_BAND), float(spot) * (1.0 + ATM_BAND)
    qs: list[float] = []
    for K in common_strikes:
        if not (lo <= K <= hi):
            continue
        c, p = calls.loc[K, "polygon_price"], puts.loc[K, "polygon_price"]
        oi_c = calls.loc[K, "polygon_open_interest"]
        oi_p = puts.loc[K, "polygon_open_interest"]
        if pd.isna(c) or pd.isna(p) or pd.isna(oi_c) or pd.isna(oi_p):
            continue
        if oi_c < MIN_OI_PER_LEG or oi_p < MIN_OI_PER_LEG:
            continue
        arg = (float(c) - float(p) + float(K) * math.exp(-r * T)) / float(spot)
        if arg <= 0:
            continue
        q = -math.log(arg) / T
        if abs(q) > MAX_Q_FOR_INCL:
            continue
        qs.append(q)

    if len(qs) < MIN_STRIKES_PCP:
        return None
    q_med = float(np.median(qs))
    if not (SANE_Q_LOW <= q_med <= SANE_Q_HIGH):
        return None
    return {"q": q_med, "method": "pcp-median", "n_strikes": len(qs)}


def solve_q(chain_df: pd.DataFrame, ticker: str, r: float) -> YieldResult:
    """Resolve q for a single (underlying, expiry) chain slice.

    chain_df must contain rows for ONE underlying and ONE expiration. The
    function does not enforce this (it just uses the first row's spot/dte).

    Always returns a result — never None. The `method` field tells the
    caller which tier was used.
    """
    pcp = _pcp_median(chain_df, r)
    if pcp is not None:
        return pcp
    fallback = HARDCODED_YIELDS.get(ticker)
    if fallback is not None:
        return {"q": fallback, "method": "hardcoded", "n_strikes": 0}
    return {"q": 0.0, "method": "zero", "n_strikes": 0}
