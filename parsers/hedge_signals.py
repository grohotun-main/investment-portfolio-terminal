"""Hedge-signal alerts for the Options Hedging tab: IV-cheap × high-MCR.

A name "fires" when it is one of the hedge recommender's targets (SPY in
the systematic bucket, plus the idiosyncratic excess-MCR names) AND its own
ATM IV is currently cheap by historical standards (percentile below
``CHEAP_PERCENTILE_CUTOFF`` of the trailing year).

Pure module — no Streamlit, no I/O. Reuses the IV-percentile machinery that
already drives the gauge/sparkline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Optional

import pandas as pd

from iv_rank import _today_quality, iv_percentile

# Bottom-quartile of the trailing year. Matches the cheap edge of the
# 20/80 reference band drawn on the IV sparkline directly above the panel.
CHEAP_PERCENTILE_CUTOFF = 25.0


@dataclass(frozen=True)
class HedgeSignal:
    """One row of the hedge-signal panel."""
    ticker: str
    mcr_kind: Literal["systematic", "idiosyncratic"]
    mcr_share_pct: float            # systematic share for SPY; excess-MCR pctr_pct otherwise
    iv_percentile: float            # NaN when no usable IV history
    is_cheap: bool                  # has_iv AND iv_percentile < cutoff
    has_iv: bool                    # False -> "no IV data", never fires
    quality: Optional[str]          # "interp" | "approx" | None (from the CM gauge)


def hedge_signal_universe(
        diagnostics: Mapping) -> list[tuple[str, str, float]]:
    """Reshape ``build_hedge_basket`` diagnostics into the signal universe of
    ``(ticker, mcr_kind, mcr_share_pct)`` tuples — the names worth hedging,
    each with its MCR share.

    SPY is normally returned by ``identify_excess_mcr_names`` itself (its
    weight in the SPY-constituents file is 0, so any SPY holding exceeds the
    1.5x threshold), carrying the largest MCR — so it's surfaced from
    ``excess_names`` like any other name, just labelled ``systematic``.
    Only when SPY is absent from ``excess_names`` do we fall back to the
    ``spy_systematic_mcr_pct`` residual. Missing keys yield an empty universe.
    """
    per_name = diagnostics.get("per_name_mcr_pct", {}) or {}
    excess = list(diagnostics.get("excess_names", []) or [])
    universe: list[tuple[str, str, float]] = [
        (name, "systematic" if name == "SPY" else "idiosyncratic",
         float(per_name.get(name, 0.0)))
        for name in excess
    ]
    if "SPY" not in excess:
        spy_share = float(diagnostics.get("spy_systematic_mcr_pct", 0.0) or 0.0)
        if spy_share > 0.0:
            universe.insert(0, ("SPY", "systematic", spy_share))
    return universe


def build_hedge_signals(*, universe: list[tuple[str, str, float]],
                        iv_history: pd.DataFrame,
                        as_of,
                        cheap_cutoff: float = CHEAP_PERCENTILE_CUTOFF,
                        window_days: int = 252) -> list[HedgeSignal]:
    """Join a hedge ``universe`` of ``(ticker, mcr_kind, mcr_share_pct)`` with
    each name's trailing-``window_days`` IV percentile, classify "cheap"
    (percentile strictly below ``cheap_cutoff``), and return the signals
    ranked cheapest-first. Names without usable IV history sink to the
    bottom (``has_iv=False``) and never fire.
    """
    signals: list[HedgeSignal] = []
    for ticker, kind, mcr_share in universe:
        pct = iv_percentile(iv_history, ticker, as_of=as_of,
                            window_days=window_days)
        has_iv = not math.isnan(pct)
        quality = _today_quality(iv_history, ticker, as_of=as_of) if has_iv else None
        # An approx (one-sided constant-maturity) reading — typical of thin,
        # illiquid chains — is too noisy to assert "cheap". The row still
        # shows its percentile + ⚠ quality; it just doesn't fire the signal.
        is_cheap = has_iv and pct < cheap_cutoff and quality != "approx"
        signals.append(HedgeSignal(
            ticker=ticker, mcr_kind=kind, mcr_share_pct=float(mcr_share),
            iv_percentile=pct, is_cheap=is_cheap, has_iv=has_iv,
            quality=quality))

    # Cheapest-first: with-IV names by ascending percentile (so cheap ones
    # lead), then no-IV names last, alphabetically.
    signals.sort(key=lambda s: (not s.has_iv,
                                s.iv_percentile if s.has_iv else math.inf,
                                s.ticker))
    return signals


def signals_to_table_rows(signals: list[HedgeSignal]) -> list[dict]:
    """Render-ready rows for the panel table — one dict per signal, in the
    order given. Keeps the cheap / no-IV / approx display branching in
    tested code rather than in the Streamlit wrapper."""
    rows = []
    for s in signals:
        rows.append({
            "Name":      s.ticker,
            "IV %ile":   f"{s.iv_percentile:.0f}" if s.has_iv else "—",
            "Signal":    ("🟢 cheap" if s.is_cheap
                          else ("—" if s.has_iv else "no IV data")),
            "MCR share": f"{s.mcr_share_pct:.0f}%",
            "Quality":   "⚠ approx" if s.quality == "approx" else "",
        })
    return rows


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 22 -> '22nd'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_signal_headline(signals: list[HedgeSignal]) -> tuple[str, str]:
    """One-line headline above the panel, as ``(level, text)`` where
    ``level`` is ``"green"`` (something cheap), ``"amber"`` (nothing cheap),
    or ``"grey"`` (no IV data / nothing to hedge)."""
    if not signals:
        return ("grey", "No concentrated names to hedge right now")
    with_iv = [s for s in signals if s.has_iv]
    if not with_iv:
        return ("grey", "No IV history for your concentrated names yet")
    cheap = sorted((s for s in with_iv if s.is_cheap),
                   key=lambda s: s.iv_percentile)
    if cheap:
        names = ", ".join(
            f"{s.ticker} ({_ordinal(round(s.iv_percentile))} pct)"
            for s in cheap)
        noun = "name" if len(cheap) == 1 else "names"
        return ("green", f"{len(cheap)} {noun} cheap to hedge now: {names}")
    cheapest = min(with_iv, key=lambda s: s.iv_percentile)
    return ("amber",
            f"Nothing cheap right now (cheapest: {cheapest.ticker}, "
            f"{_ordinal(round(cheapest.iv_percentile))} pct)")
