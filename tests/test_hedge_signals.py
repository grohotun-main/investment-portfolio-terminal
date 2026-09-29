"""Tests for parsers/hedge_signals.py — the IV-cheap × high-MCR hedge
signal panel on the Options Hedging tab.

Three pure surfaces:
  * `build_hedge_signals` — join a hedge universe with per-name IV
    percentile + cheap classification, ranked cheapest-first.
  * `hedge_signal_universe` — reshape build_hedge_basket diagnostics into
    the (ticker, kind, mcr_share) universe.
  * `format_signal_headline` — the one-line headline above the panel.
"""
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from hedge_signals import (  # noqa: E402
    HedgeSignal,
    build_hedge_signals,
    format_signal_headline,
    hedge_signal_universe,
    signals_to_table_rows,
)

AS_OF = date(2026, 5, 26)


def _sig(ticker, pct, cheap, has_iv=True, kind="idiosyncratic", mcr=5.0,
         quality="interp"):
    return HedgeSignal(ticker=ticker, mcr_kind=kind, mcr_share_pct=mcr,
                       iv_percentile=pct, is_cheap=cheap, has_iv=has_iv,
                       quality=quality)


def _hist(values, underlying, quality=None):
    """One row per recent session for `underlying`; today == values[-1]."""
    rows = []
    for i, v in enumerate(reversed(values)):
        row = {"date": pd.Timestamp(AS_OF - timedelta(days=i)).isoformat(),
               "underlying": underlying, "atm_iv": v}
        if quality is not None:
            row["quality"] = quality
        rows.append(row)
    return pd.DataFrame(rows)


def _concat(*frames):
    return pd.concat(frames, ignore_index=True)


class TestBuildHedgeSignals(unittest.TestCase):
    def test_cheap_name_flagged(self):
        # today (0.10) is the min of 5 -> percentile 0 -> cheap (<25)
        hist = _hist([0.50, 0.40, 0.30, 0.20, 0.10], "SPY")
        sigs = build_hedge_signals(
            universe=[("SPY", "systematic", 62.0)],
            iv_history=hist, as_of=AS_OF)
        self.assertEqual(len(sigs), 1)
        s = sigs[0]
        self.assertEqual(s.ticker, "SPY")
        self.assertEqual(s.mcr_kind, "systematic")
        self.assertAlmostEqual(s.mcr_share_pct, 62.0)
        self.assertTrue(s.has_iv)
        self.assertTrue(s.is_cheap)
        self.assertAlmostEqual(s.iv_percentile, 0.0)

    def test_at_cutoff_not_cheap(self):
        # today (0.20) is 2nd-smallest of 5 -> percentile 25.0 -> NOT cheap
        # (cutoff is strict `< 25`)
        hist = _hist([0.10, 0.30, 0.40, 0.50, 0.20], "GLD")
        sigs = build_hedge_signals(
            universe=[("GLD", "idiosyncratic", 9.0)],
            iv_history=hist, as_of=AS_OF)
        self.assertAlmostEqual(sigs[0].iv_percentile, 25.0)
        self.assertFalse(sigs[0].is_cheap)

    def test_expensive_name_not_cheap(self):
        # today (0.50) is the max -> percentile 100 -> not cheap
        hist = _hist([0.10, 0.20, 0.30, 0.40, 0.50], "GOOG")
        sigs = build_hedge_signals(
            universe=[("GOOG", "idiosyncratic", 7.0)],
            iv_history=hist, as_of=AS_OF)
        self.assertAlmostEqual(sigs[0].iv_percentile, 100.0)
        self.assertFalse(sigs[0].is_cheap)

    def test_name_without_history_has_no_iv(self):
        hist = _hist([0.50, 0.10], "SPY")
        sigs = build_hedge_signals(
            universe=[("CLS", "idiosyncratic", 5.0)],
            iv_history=hist, as_of=AS_OF)
        s = sigs[0]
        self.assertFalse(s.has_iv)
        self.assertFalse(s.is_cheap)
        self.assertTrue(math.isnan(s.iv_percentile))

    def test_sorted_cheapest_first_then_no_iv_last(self):
        hist = _concat(
            _hist([0.50, 0.40, 0.30, 0.20, 0.10], "SPY"),   # today 0.10 -> pct 0
            _hist([0.10, 0.20, 0.40, 0.50, 0.30], "GLD"),   # today 0.30 -> pct 50
        )
        universe = [
            ("GLD", "idiosyncratic", 9.0),   # 50, listed first on purpose
            ("SPY", "systematic", 62.0),     # 0
            ("CLS", "idiosyncratic", 5.0),   # no IV history
        ]
        sigs = build_hedge_signals(universe=universe, iv_history=hist, as_of=AS_OF)
        self.assertEqual([s.ticker for s in sigs], ["SPY", "GLD", "CLS"])

    def test_approx_quality_surfaced(self):
        hist = _hist([0.50, 0.10], "SPY", quality="approx")
        sigs = build_hedge_signals(
            universe=[("SPY", "systematic", 62.0)],
            iv_history=hist, as_of=AS_OF)
        self.assertEqual(sigs[0].quality, "approx")

    def test_approx_today_reading_does_not_fire_cheap(self):
        # A thin/illiquid name whose today reading is a one-sided CM
        # approximation is too noisy to assert "cheap": it still shows its
        # percentile + ⚠ quality, but must NOT earn the 🟢 cheap signal.
        hist = _hist([0.50, 0.40, 0.30, 0.20, 0.10], "VISN", quality="approx")
        sigs = build_hedge_signals(
            universe=[("VISN", "idiosyncratic", 3.0)],
            iv_history=hist, as_of=AS_OF)
        s = sigs[0]
        self.assertAlmostEqual(s.iv_percentile, 0.0)  # cheap by percentile…
        self.assertEqual(s.quality, "approx")
        self.assertTrue(s.has_iv)
        self.assertFalse(s.is_cheap)                  # …but gated by approx

    def test_interp_quality_still_fires_cheap(self):
        hist = _hist([0.50, 0.40, 0.30, 0.20, 0.10], "SPY", quality="interp")
        sigs = build_hedge_signals(
            universe=[("SPY", "systematic", 40.0)],
            iv_history=hist, as_of=AS_OF)
        self.assertTrue(sigs[0].is_cheap)

    def test_empty_universe_returns_empty(self):
        hist = _hist([0.50, 0.10], "SPY")
        sigs = build_hedge_signals(universe=[], iv_history=hist, as_of=AS_OF)
        self.assertEqual(sigs, [])


class TestFormatSignalHeadline(unittest.TestCase):
    def test_green_lists_cheap_names_with_ordinal_percentiles(self):
        sigs = [
            _sig("SPY", 18.0, True, kind="systematic", mcr=62.0),
            _sig("GLD", 22.0, True),
            _sig("GOOG", 41.0, False),
        ]
        level, text = format_signal_headline(sigs)
        self.assertEqual(level, "green")
        self.assertEqual(
            text,
            "2 names cheap to hedge now: SPY (18th pct), GLD (22nd pct)")

    def test_green_singular_for_one_cheap_name(self):
        sigs = [_sig("SPY", 1.0, True, kind="systematic", mcr=62.0),
                _sig("GOOG", 80.0, False)]
        level, text = format_signal_headline(sigs)
        self.assertEqual(level, "green")
        self.assertEqual(text, "1 name cheap to hedge now: SPY (1st pct)")

    def test_amber_when_none_cheap_names_cheapest(self):
        sigs = [_sig("GOOG", 41.0, False), _sig("SPY", 60.0, False,
                                                 kind="systematic", mcr=62.0)]
        level, text = format_signal_headline(sigs)
        self.assertEqual(level, "amber")
        self.assertEqual(text, "Nothing cheap right now (cheapest: GOOG, 41st pct)")

    def test_grey_when_no_iv_history(self):
        sigs = [_sig("CLS", float("nan"), False, has_iv=False, quality=None),
                _sig("IEF", float("nan"), False, has_iv=False, quality=None)]
        level, text = format_signal_headline(sigs)
        self.assertEqual(level, "grey")
        self.assertEqual(text, "No IV history for your concentrated names yet")

    def test_grey_when_empty(self):
        level, text = format_signal_headline([])
        self.assertEqual(level, "grey")
        self.assertEqual(text, "No concentrated names to hedge right now")


class TestHedgeSignalUniverse(unittest.TestCase):
    def test_includes_spy_systematic_then_excess_names(self):
        diagnostics = {
            "excess_names": ["GOOG", "GLD"],
            "per_name_mcr_pct": {"GOOG": 7.0, "GLD": 9.0},
            "spy_systematic_mcr_pct": 62.0,
        }
        uni = hedge_signal_universe(diagnostics)
        self.assertEqual(uni, [
            ("SPY", "systematic", 62.0),
            ("GOOG", "idiosyncratic", 7.0),
            ("GLD", "idiosyncratic", 9.0),
        ])

    def test_spy_in_excess_names_is_systematic_with_its_own_mcr(self):
        # Reality check: identify_excess_mcr_names DOES return SPY — its weight
        # in the SPY-constituents file is 0, so any SPY holding trivially
        # exceeds the 1.5x threshold. SPY then carries the biggest MCR, and
        # spy_systematic_mcr_pct is 0 (SPY's MCR is already inside the excess
        # sum). SPY must surface — labelled systematic — not be dropped.
        diagnostics = {
            "excess_names": ["SPY", "MU", "GLD"],
            "per_name_mcr_pct": {"SPY": 39.8, "MU": 14.8, "GLD": 10.9},
            "spy_systematic_mcr_pct": 0.0,
        }
        uni = hedge_signal_universe(diagnostics)
        self.assertEqual(uni, [
            ("SPY", "systematic", 39.8),
            ("MU", "idiosyncratic", 14.8),
            ("GLD", "idiosyncratic", 10.9),
        ])

    def test_skips_spy_when_systematic_share_zero(self):
        diagnostics = {
            "excess_names": ["GOOG"],
            "per_name_mcr_pct": {"GOOG": 7.0},
            "spy_systematic_mcr_pct": 0.0,
        }
        uni = hedge_signal_universe(diagnostics)
        self.assertEqual(uni, [("GOOG", "idiosyncratic", 7.0)])

    def test_excess_name_without_mcr_defaults_to_zero(self):
        diagnostics = {
            "excess_names": ["CLS"],
            "per_name_mcr_pct": {},
            "spy_systematic_mcr_pct": 0.0,
        }
        uni = hedge_signal_universe(diagnostics)
        self.assertEqual(uni, [("CLS", "idiosyncratic", 0.0)])

    def test_missing_keys_returns_empty(self):
        self.assertEqual(hedge_signal_universe({}), [])


class TestSignalsToTableRows(unittest.TestCase):
    def test_cheap_row(self):
        rows = signals_to_table_rows(
            [_sig("SPY", 0.0, True, kind="systematic", mcr=62.0)])
        self.assertEqual(rows, [{
            "Name": "SPY", "IV %ile": "0", "Signal": "🟢 cheap",
            "MCR share": "62%", "Quality": "",
        }])

    def test_not_cheap_with_iv_row(self):
        row = signals_to_table_rows([_sig("GOOG", 41.0, False, mcr=7.0)])[0]
        self.assertEqual(row["Signal"], "—")
        self.assertEqual(row["IV %ile"], "41")

    def test_no_iv_row(self):
        row = signals_to_table_rows(
            [_sig("CLS", float("nan"), False, has_iv=False, quality=None)])[0]
        self.assertEqual(row["IV %ile"], "—")
        self.assertEqual(row["Signal"], "no IV data")

    def test_approx_quality_row(self):
        row = signals_to_table_rows(
            [_sig("GLD", 5.0, True, quality="approx")])[0]
        self.assertEqual(row["Quality"], "⚠ approx")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
