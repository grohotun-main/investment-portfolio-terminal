"""IV-at-buy ranking for the Options Hedging tab caption.

`leg_iv_at_buy` ranks a single leg's IV against the 52w window ending
on its `open_date` — i.e. "when this leg was bought, where was IV in
its recent range?" That's the textbook interpretation of "bought at
rank N".

`book_iv_at_buy_rank` weights per-leg ranks by current market value
across the sleeve. Legs missing from history (e.g. open_date too far
back, or underlying we never fetched) are surfaced via `skipped_legs`
so the caller can warn.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd


@dataclass
class LegAtBuy:
    underlying: str
    open_date: pd.Timestamp
    iv_at_buy: float
    rank_at_buy: float


@dataclass
class BookAtBuy:
    rank: float
    covered_legs: int
    skipped_legs: int


WINDOW_DAYS = 365


def leg_iv_at_buy(history: pd.DataFrame, underlying: str,
                  open_date: date | pd.Timestamp) -> Optional[LegAtBuy]:
    """Find the leg's underlying ATM IV on open_date and rank it within
    the trailing 52w window ending on that day.

    Returns None when:
      * history has no rows for the underlying, OR
      * open_date is before any history row for the underlying.

    When open_date isn't a trading day, uses the most recent history
    row at or before open_date (matches forward-fill semantics).
    """
    if history.empty or "underlying" not in history.columns:
        return None
    sel = history[history["underlying"] == underlying].copy()
    if sel.empty:
        return None
    sel["date"] = pd.to_datetime(sel["date"])
    sel = sel.dropna(subset=["atm_iv"]).sort_values("date")

    od = pd.Timestamp(open_date)
    at_or_before = sel[sel["date"] <= od]
    if at_or_before.empty:
        return None
    iv_at_buy = float(at_or_before.iloc[-1]["atm_iv"])

    window_start = od - pd.Timedelta(days=WINDOW_DAYS)
    window = sel[(sel["date"] >= window_start) & (sel["date"] <= od)]
    iv_min = float(window["atm_iv"].min())
    iv_max = float(window["atm_iv"].max())
    if iv_max <= iv_min:
        rank = float("nan")
    else:
        rank = 100.0 * (iv_at_buy - iv_min) / (iv_max - iv_min)
    return LegAtBuy(
        underlying=underlying, open_date=od,
        iv_at_buy=iv_at_buy, rank_at_buy=rank,
    )


def book_iv_at_buy_rank(history: pd.DataFrame,
                        legs: Iterable[dict],
                        *, as_of: date) -> Optional[BookAtBuy]:
    """Book-wide "bought avg at rank N", weighted by current MV.

    `legs` is an iterable of dicts with `underlying`, `open_date`,
    and `market_value`. Order doesn't matter.
    """
    contributions: list[tuple[float, float]] = []  # (weight, rank)
    skipped = 0
    for leg in legs:
        mv = float(leg.get("market_value") or 0.0)
        if mv <= 0.0:
            skipped += 1
            continue
        r = leg_iv_at_buy(history, leg["underlying"], leg["open_date"])
        if r is None or pd.isna(r.rank_at_buy):
            skipped += 1
            continue
        contributions.append((mv, r.rank_at_buy))

    if not contributions:
        return None
    total_w = sum(c[0] for c in contributions)
    if total_w <= 0.0:
        return None
    rank = sum(c[0] * c[1] for c in contributions) / total_w
    return BookAtBuy(rank=rank, covered_legs=len(contributions),
                     skipped_legs=skipped)
