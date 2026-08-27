"""Tests for parsers/dip_ladder.py — the dry-powder rotation state machine.

Synthetic hand-computable paths (tr == price, no dividends, so round-trip
arithmetic checks by hand). Conventions under test (spec 2026-07-18):
fractions are OF REMAINING powder; one tranche per band entry, re-armed
only when that tranche exits; exit = price >= the tranche's drawdown-anchor
peak; on one eval day the band tranche deploys before the ★-rule tranche.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from dip_ladder import LADDER_FRACTIONS, simulate_ladder  # noqa: E402


def _series(vals, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series([float(v) for v in vals], index=idx)


def _evals(price, entries):
    """entries: list of (day_index, band, tk_rule)."""
    return pd.DataFrame([
        {"date": price.index[i], "band": band, "tk_rule": tk}
        for i, band, tk in entries
    ])


def _path(*segments):
    """[(value, n_days), ...] -> flat list."""
    out = []
    for v, n in segments:
        out.extend([v] * n)
    return out


class LadderTests(unittest.TestCase):
    def test_no_signals_tracks_cash_leg(self):
        price = _series(_path((100, 40)))
        cash_rets = pd.Series(0.001, index=price.index)
        out = simulate_ladder(price, price, cash_rets, _evals(price, []))
        expected = (1.0 + cash_rets).cumprod()
        pd.testing.assert_series_equal(out["wealth"], expected,
                                       check_names=False)
        self.assertEqual(out["summary"]["n_tranches"], 0)

    def test_single_buy_round_trip(self):
        # peak 100 -> dip to 90 -> recover to 100 at day 30.
        price = _series(_path((100, 10), (90, 20), (100, 30)))
        zero = pd.Series(0.0, index=price.index)
        out = simulate_ladder(price, price, zero,
                              _evals(price, [(12, "neutral", False)]))
        self.assertEqual(out["summary"]["n_tranches"], 1)
        t = out["tranches"][0]
        self.assertEqual(t["band"], "neutral")
        self.assertEqual(t["entry_date"], price.index[12])
        self.assertEqual(t["exit_date"], price.index[30])
        self.assertAlmostEqual(t["round_trip_return"], 100.0 / 90.0 - 1.0,
                               places=10)
        # 0.25 deployed at 90, exits at 100 -> wealth 0.75 + 0.25*10/9.
        self.assertAlmostEqual(float(out["wealth"].iloc[-1]),
                               0.75 + 0.25 * 100.0 / 90.0, places=10)
        # While the tranche is open and tr is flat, wealth stays 1.0.
        self.assertAlmostEqual(float(out["wealth"].iloc[20]), 1.0, places=10)

    def test_rearm_only_after_exit(self):
        # Two dips; consecutive evals in dip 1 deploy ONCE; dip 2 re-deploys.
        price = _series(_path((100, 10), (90, 20), (100, 5), (90, 10),
                              (100, 15)))
        zero = pd.Series(0.0, index=price.index)
        out = simulate_ladder(
            price, price, zero,
            _evals(price, [(12, "neutral", False), (17, "neutral", False),
                           (37, "neutral", False)]))
        self.assertEqual(out["summary"]["n_tranches"], 2)
        self.assertEqual(out["tranches"][0]["entry_date"], price.index[12])
        self.assertEqual(out["tranches"][1]["entry_date"], price.index[37])

    def test_fractions_of_remaining_powder(self):
        price = _series(_path((100, 10), (90, 20), (100, 30)))
        zero = pd.Series(0.0, index=price.index)
        out = simulate_ladder(
            price, price, zero,
            _evals(price, [(12, "neutral", False), (14, "strong", False)]))
        self.assertEqual(out["summary"]["n_tranches"], 2)
        sizes = [t["deployed"] for t in out["tranches"]]
        self.assertAlmostEqual(sizes[0], 0.25, places=10)          # 25% of 1.0
        self.assertAlmostEqual(sizes[1], 0.50 * 0.75, places=10)   # 50% of rest
        # Both exit at day 30; wealth = 0.375 + 0.625 * 10/9.
        self.assertAlmostEqual(float(out["wealth"].iloc[-1]),
                               0.375 + 0.625 * 100.0 / 90.0, places=10)

    def test_tk_rule_takes_the_rest_and_empty_fire_is_noop(self):
        price = _series(_path((100, 10), (90, 20), (100, 30)))
        zero = pd.Series(0.0, index=price.index)
        out = simulate_ladder(
            price, price, zero,
            _evals(price, [(12, "strong", True), (17, "strong", True)]))
        # Day 12: strong tranche 50%, then tk_rule 100% of the rest -> all-in.
        self.assertEqual(out["summary"]["n_tranches"], 2)
        self.assertAlmostEqual(sum(t["deployed"] for t in out["tranches"]),
                               1.0, places=10)
        # Day 17: both rungs disarmed AND powder empty -> no-op, logged.
        self.assertEqual(out["summary"]["skipped_deploys"], 0)
        # A later eval with powder empty but rung ARMED logs a skip:
        out2 = simulate_ladder(
            price, price, zero,
            _evals(price, [(12, "strong", True), (17, "neutral", False)]))
        self.assertEqual(out2["summary"]["skipped_deploys"], 1)

    def test_never_recovered_marks_at_final(self):
        price = _series(_path((100, 10), (90, 20), (95, 30)))
        zero = pd.Series(0.0, index=price.index)
        out = simulate_ladder(price, price, zero,
                              _evals(price, [(12, "neutral", False)]))
        t = out["tranches"][0]
        self.assertIsNone(t["exit_date"])
        self.assertAlmostEqual(float(out["wealth"].iloc[-1]),
                               0.75 + 0.25 * 95.0 / 90.0, places=10)

    def test_cash_flat_carry_on_missing_dates(self):
        price = _series(_path((100, 40)))
        # cash series covers only the first 20 days at 0.001/day.
        cash_rets = pd.Series(0.001, index=price.index[:20])
        out = simulate_ladder(price, price, cash_rets, _evals(price, []))
        expected_final = (1.001) ** 20
        self.assertAlmostEqual(float(out["wealth"].iloc[-1]),
                               expected_final, places=10)

    def test_default_fractions_are_the_registered_claim(self):
        self.assertEqual(LADDER_FRACTIONS,
                         {"neutral": 0.25, "strong": 0.50, "tk_rule": 1.0})

    def test_summary_reports_exposure(self):
        price = _series(_path((100, 10), (90, 20), (100, 30)))
        zero = pd.Series(0.0, index=price.index)
        out = simulate_ladder(price, price, zero,
                              _evals(price, [(12, "neutral", False)]))
        s = out["summary"]
        for k in ("n_tranches", "skipped_deploys", "avg_equity_exposure",
                  "final_wealth"):
            self.assertIn(k, s)
        self.assertGreater(s["avg_equity_exposure"], 0.0)
        self.assertLess(s["avg_equity_exposure"], 1.0)


if __name__ == "__main__":
    unittest.main()
