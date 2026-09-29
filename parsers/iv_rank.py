"""IV Rank math for the Options Hedging tab caption.

Two functions:
  * `iv_rank(history, underlying, as_of)` — single-underlying rank
    formula `(today_IV - 52w_min) / (52w_max - 52w_min)` scaled to
    0-100. Returns None if history has no rows for the underlying;
    rank is NaN if the 52w range collapsed to a single value.
  * `book_iv_rank(history, positions_mv, as_of)` — MV-weighted average
    of per-underlying ranks across the sleeve, with skip-tracking so
    the UI can flag underlyings missing from history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import math
import pandas as pd


@dataclass
class UnderlyingRank:
    underlying: str
    rank: float
    iv_today: float
    iv_min: float
    iv_max: float
    as_of: date


@dataclass
class BookRank:
    rank: float
    iv_today_weighted: float
    iv_min_weighted: float
    iv_max_weighted: float
    as_of: date
    covered_underlyings: list[str] = field(default_factory=list)
    skipped_underlyings: list[str] = field(default_factory=list)


def _select_underlying(history: pd.DataFrame, underlying: str) -> pd.DataFrame:
    if history.empty or "underlying" not in history.columns:
        return pd.DataFrame()
    return history[history["underlying"] == underlying].copy()


def iv_rank(history: pd.DataFrame, underlying: str,
            *, as_of: date) -> Optional[UnderlyingRank]:
    """Rank today's ATM IV within the trailing 52w range.

    Returns None when there's no data at all for the underlying.
    Returns a UnderlyingRank with rank=NaN when min==max (avoids
    a divide-by-zero on flat synthetic histories).
    """
    rows = _select_underlying(history, underlying)
    if rows.empty:
        return None
    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows.dropna(subset=["atm_iv"]).sort_values("date")
    if rows.empty:
        return None
    iv_today = float(rows.iloc[-1]["atm_iv"])
    iv_min = float(rows["atm_iv"].min())
    iv_max = float(rows["atm_iv"].max())
    if iv_max <= iv_min:
        return UnderlyingRank(
            underlying=underlying, rank=float("nan"),
            iv_today=iv_today, iv_min=iv_min, iv_max=iv_max, as_of=as_of,
        )
    rank = 100.0 * (iv_today - iv_min) / (iv_max - iv_min)
    return UnderlyingRank(
        underlying=underlying, rank=rank,
        iv_today=iv_today, iv_min=iv_min, iv_max=iv_max, as_of=as_of,
    )


def book_iv_rank(history: pd.DataFrame, positions_mv: dict[str, float],
                 *, as_of: date) -> Optional[BookRank]:
    """MV-weighted IV rank across the sleeve. Underlyings missing
    from history are listed in `skipped_underlyings` so the caller
    can warn the user without silently dropping them.
    """
    if not positions_mv:
        return None
    per_rank: list[tuple[str, float, float, float, float, float]] = []
    skipped: list[str] = []
    for u, mv in positions_mv.items():
        r = iv_rank(history, u, as_of=as_of)
        if r is None or math.isnan(r.rank):
            skipped.append(u)
            continue
        per_rank.append((u, mv, r.rank, r.iv_today, r.iv_min, r.iv_max))

    if not per_rank:
        return None
    total_mv = sum(p[1] for p in per_rank)
    if total_mv <= 0.0:
        return None
    rank = sum(p[1] * p[2] for p in per_rank) / total_mv
    iv_today = sum(p[1] * p[3] for p in per_rank) / total_mv
    iv_min = sum(p[1] * p[4] for p in per_rank) / total_mv
    iv_max = sum(p[1] * p[5] for p in per_rank) / total_mv
    return BookRank(
        rank=rank, iv_today_weighted=iv_today,
        iv_min_weighted=iv_min, iv_max_weighted=iv_max, as_of=as_of,
        covered_underlyings=[p[0] for p in per_rank],
        skipped_underlyings=skipped,
    )


def format_iv_rank_caption(book: Optional[BookRank],
                           at_buy: object) -> str:
    """Render the caption shown under the Weighted IV tile.

    Format: "Vol rank **N** (52w range L%–H%) · bought avg at rank **M**"
    Falls back gracefully when at_buy is None or the whole history is
    missing. Skipped underlyings (in history) are surfaced parenthetically
    so the user knows the rank isn't whole-book.
    """
    if book is None or math.isnan(book.rank):
        return "Vol rank not available — refresh ATM IV history to populate."

    parts = [
        f"Vol rank **{book.rank:.0f}** "
        f"(52w range {book.iv_min_weighted:.1%}–{book.iv_max_weighted:.1%})"
    ]
    if at_buy is not None and not math.isnan(getattr(at_buy, "rank", float("nan"))):
        parts.append(f"bought avg at rank **{at_buy.rank:.0f}**")
    caption = " · ".join(parts)
    if book.skipped_underlyings:
        skipped = ", ".join(book.skipped_underlyings)
        caption += f" *(excludes {skipped} — no IV history)*"
    return caption


def iv_percentile(history: pd.DataFrame, underlying: str,
                  *, as_of: date, window_days: int = 252) -> float:
    """Empirical percentile (rank-order) of the most recent ATM IV at or
    before ``as_of``, computed against the trailing ``window_days`` trading
    SESSIONS (rows) ending on ``as_of``. Returns NaN when there's no data
    in the window.

    Window is row-based, not calendar-based: ``window_days=252`` means the
    last 252 observed sessions — a true trading year regardless of holidays
    or data gaps. NaN rows (dead days from the constant-maturity derive
    step) are dropped BEFORE the tail, so the window always holds
    ``window_days`` real observations, never NaN-padded ones.

    Tie handling: `pandas.Series.rank(method="average")`.
    """
    if history is None or history.empty or "underlying" not in history.columns:
        return float("nan")
    sel = history[history["underlying"] == underlying].copy()
    if sel.empty:
        return float("nan")
    sel["date"] = pd.to_datetime(sel["date"])
    sel = sel.dropna(subset=["atm_iv"]).sort_values("date")
    as_of_ts = pd.Timestamp(as_of)
    sel = sel[sel["date"] <= as_of_ts]
    if sel.empty:
        return float("nan")
    win = sel.tail(window_days)
    if len(win) < 2:
        return float("nan")
    ranks = win["atm_iv"].rank(method="average")
    today_rank = float(ranks.iloc[-1])
    return 100.0 * (today_rank - 1.0) / (len(win) - 1)


@dataclass
class BookPercentile:
    """MV-weighted empirical IV percentile across the sleeve. Unlike
    `BookRank` (min/max), this blends per-underlying rank-order
    percentiles, so deep history doesn't get anchored to a single old
    spike. `skipped_underlyings` lists names absent from the history so
    the caller can warn instead of silently dropping them.

    `approx_underlyings` lists names whose CURRENT (as-of) reading is a
    one-sided constant-maturity approximation (quality "approx" — nearest
    single expiry, not a true bracketed 90-day point), so the caller can
    flag that the gauge point isn't fully apples-to-apples for that name."""
    percentile: float
    as_of: date
    covered_underlyings: list[str] = field(default_factory=list)
    skipped_underlyings: list[str] = field(default_factory=list)
    approx_underlyings: list[str] = field(default_factory=list)


def _today_quality(history: pd.DataFrame, underlying: str,
                   *, as_of: date) -> Optional[str]:
    """The `quality` flag of the most recent CM reading at/before `as_of`
    for `underlying`. None when the history has no `quality` column
    (pre-constant-maturity data) or no usable row."""
    if history is None or history.empty or "quality" not in history.columns:
        return None
    sel = history[history["underlying"] == underlying].copy()
    if sel.empty:
        return None
    sel["date"] = pd.to_datetime(sel["date"])
    sel = sel.dropna(subset=["atm_iv"]).sort_values("date")
    sel = sel[sel["date"] <= pd.Timestamp(as_of)]
    if sel.empty:
        return None
    return sel.iloc[-1]["quality"]


def book_iv_percentile(history: pd.DataFrame,
                       positions_mv: dict[str, float],
                       *, as_of: date,
                       window_days: int = 252) -> Optional[BookPercentile]:
    """MV-weighted empirical IV percentile across the sleeve.

    Each underlying's `iv_percentile` over the trailing `window_days` is
    weighted by its position market value. Underlyings missing from the
    history (NaN percentile) are dropped from the weighting and surfaced
    in `skipped_underlyings`. Returns None when no positions are given or
    none of them have usable history.
    """
    if not positions_mv:
        return None
    contributions: list[tuple[str, float, float]] = []  # (sym, mv, pct)
    skipped: list[str] = []
    for sym, mv in positions_mv.items():
        if mv is None or mv <= 0.0:
            skipped.append(sym)
            continue
        pct = iv_percentile(history, sym, as_of=as_of, window_days=window_days)
        if math.isnan(pct):
            skipped.append(sym)
            continue
        contributions.append((sym, float(mv), pct))

    if not contributions:
        return None
    total_mv = sum(c[1] for c in contributions)
    if total_mv <= 0.0:
        return None
    weighted = sum(c[1] * c[2] for c in contributions) / total_mv
    covered = [c[0] for c in contributions]
    approx = [sym for sym in covered
              if _today_quality(history, sym, as_of=as_of) == "approx"]
    return BookPercentile(
        percentile=weighted, as_of=as_of,
        covered_underlyings=covered,
        skipped_underlyings=skipped,
        approx_underlyings=approx,
    )


def book_iv_percentile_series(history: pd.DataFrame,
                              positions_mv: dict[str, float],
                              *, as_of: date,
                              window_days: int = 252,
                              span_days: int = 252) -> pd.DataFrame:
    """Rolling trajectory of the MV-weighted book percentile — the data
    behind the IV-percentile sparkline.

    Returns a `[date, percentile]` frame with one row per session over the
    trailing `span_days` ending at `as_of`. Each row is the book percentile
    *as of that day*, ranked against its own trailing `window_days` window.

    Computed by replaying the tested `book_iv_percentile` at each session
    date, so the **last row equals today's headline gauge** by construction.
    Days where the book percentile is undefined (no usable history yet —
    e.g. before the window fills) are skipped. Returns an empty
    `[date, percentile]` frame when there are no positions or no usable
    history.
    """
    empty = pd.DataFrame(columns=["date", "percentile"])
    if not positions_mv or history is None or history.empty:
        return empty
    if "underlying" not in history.columns or "date" not in history.columns:
        return empty

    covered_syms = [s for s, mv in positions_mv.items()
                    if mv is not None and mv > 0.0]
    if not covered_syms:
        return empty

    # Distinct session dates the sleeve actually has data for, on/before
    # as_of, ascending — these are the candidate x-axis points.
    sel = history[history["underlying"].isin(covered_syms)].copy()
    sel["date"] = pd.to_datetime(sel["date"])
    sel = sel.dropna(subset=["atm_iv"])
    sel = sel[sel["date"] <= pd.Timestamp(as_of)]
    if sel.empty:
        return empty
    session_dates = sorted(sel["date"].dt.normalize().unique())
    span_dates = session_dates[-span_days:]

    rows: list[dict] = []
    for ts in span_dates:
        d = pd.Timestamp(ts).date()
        book = book_iv_percentile(history, positions_mv, as_of=d,
                                  window_days=window_days)
        if book is None or math.isnan(book.percentile):
            continue
        rows.append({"date": pd.Timestamp(ts), "percentile": book.percentile})
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=["date", "percentile"])


def format_weighted_iv_caption(weighted_iv: float,
                               book_pct: Optional[BookPercentile],
                               *, window_days: int = 252) -> str:
    """Caption under the 'Weighted IV' tile: place today's MV-weighted IV
    level in its trailing-window percentile context.

    Falls back to a refresh hint when the percentile gauge isn't
    populated.
    """
    if book_pct is None or math.isnan(book_pct.percentile):
        return ("Percentile gauge not available — refresh ATM IV history "
                "to populate.")
    parts = [
        f"**{book_pct.percentile:.0f}%** of the trailing {window_days} "
        f"sessions had lower 90-day constant-maturity IV"
    ]
    if book_pct.skipped_underlyings:
        parts.append(
            "excl. " + ", ".join(sorted(book_pct.skipped_underlyings))
            + " (no IV history)"
        )
    if book_pct.approx_underlyings:
        parts.append(
            ", ".join(sorted(book_pct.approx_underlyings))
            + " on nearest expiry (approx — no 90-day bracket today)"
        )
    return " · ".join(parts)
