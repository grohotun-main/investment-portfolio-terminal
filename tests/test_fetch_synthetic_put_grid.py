"""Unit tests for parsers/fetch_synthetic_put_grid.py.

Network-touching helpers (``fetch_spy_put_chain_as_of``,
``fetch_one_contract``) are NOT exercised here — they're integration
tested via the live fetch in ``--write`` mode. We cover the pure-Python
logic: third-Friday math, chain picker, fallback picker, target planning.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.fetch_synthetic_put_grid import (  # noqa: E402
    GridTarget,
    _list_third_fridays,
    pick_best_contract,
    pick_fallback_contract,
    plan_targets,
    third_friday,
)


class TestThirdFriday(unittest.TestCase):
    def test_known_dates(self):
        # Hand-checked against a calendar.
        self.assertEqual(third_friday(2025, 6), date(2025, 6, 20))
        self.assertEqual(third_friday(2025, 12), date(2025, 12, 19))
        self.assertEqual(third_friday(2026, 1), date(2026, 1, 16))
        self.assertEqual(third_friday(2026, 3), date(2026, 3, 20))

    def test_third_friday_is_always_friday(self):
        for y in (2024, 2025, 2026, 2027):
            for m in range(1, 13):
                d = third_friday(y, m)
                self.assertEqual(d.weekday(), 4, f"{d} is not a Friday")
                # 3rd Friday is always in [15, 21] of the month.
                self.assertGreaterEqual(d.day, 15)
                self.assertLessEqual(d.day, 21)

    def test_list_inclusive_window(self):
        # Both endpoints fall on/after the 1st of the month — should include both 3rd Fridays.
        out = _list_third_fridays(date(2025, 6, 1), date(2025, 9, 30))
        self.assertEqual(out, [
            date(2025, 6, 20), date(2025, 7, 18),
            date(2025, 8, 15), date(2025, 9, 19),
        ])


class TestPlanTargets(unittest.TestCase):
    def test_weekly_snaps_to_monday(self):
        # June 1 2025 is Sun → snap to Mon Jun 2 → step 7 days.
        targets = plan_targets(date(2025, 6, 1), date(2025, 6, 30), 90, 0.05, "weekly")
        self.assertEqual([t.as_of for t in targets], [
            date(2025, 6, 2), date(2025, 6, 9), date(2025, 6, 16),
            date(2025, 6, 23), date(2025, 6, 30),
        ])
        self.assertTrue(all(t.target_dte == 90 for t in targets))
        self.assertTrue(all(t.target_moneyness == 0.05 for t in targets))

    def test_daily_skips_weekends(self):
        targets = plan_targets(date(2025, 6, 7), date(2025, 6, 13), 90, 0.05, "daily")
        # Jun 7 = Sat → snap to Mon Jun 9. Then 9, 10, 11, 12, 13 = Mon-Fri.
        self.assertEqual([t.as_of for t in targets], [
            date(2025, 6, 9), date(2025, 6, 10), date(2025, 6, 11),
            date(2025, 6, 12), date(2025, 6, 13),
        ])

    def test_monthly_cadence(self):
        # Monthly cadence stepping by 30 days; weekday gate may drop some.
        targets = plan_targets(date(2025, 1, 6), date(2025, 5, 31), 90, 0.05, "monthly")
        # Each tick: Jan 6 (Mon), Feb 5 (Wed), Mar 7 (Fri), Apr 6 (Sun-skip), May 6 (Tue).
        self.assertEqual([t.as_of for t in targets], [
            date(2025, 1, 6), date(2025, 2, 5), date(2025, 3, 7),
            date(2025, 5, 6),
        ])


class TestPickBestContract(unittest.TestCase):
    def test_minimizes_joint_distance(self):
        # Target: 90 DTE, 5% OTM (K=550 when spot=580).
        chain = [
            {"strike_price": 540, "expiration_date": "2025-08-15"},  # ~60 DTE, 6.9% OTM
            {"strike_price": 550, "expiration_date": "2025-09-19"},  # 95 DTE, 5.2% OTM — best
            {"strike_price": 560, "expiration_date": "2025-09-19"},  # 95 DTE, 3.4% OTM
            {"strike_price": 500, "expiration_date": "2025-11-21"},  # 158 DTE, 13.8% OTM
        ]
        pick = pick_best_contract(chain, date(2025, 6, 15), spot=580.0,
                                  target_dte=90, target_moneyness=0.05)
        self.assertEqual(pick["strike_price"], 550)

    def test_skips_expired_contracts(self):
        chain = [
            {"strike_price": 550, "expiration_date": "2025-05-01"},  # already expired
            {"strike_price": 560, "expiration_date": "2025-09-19"},
        ]
        pick = pick_best_contract(chain, date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertEqual(pick["strike_price"], 560)

    def test_empty_chain_returns_none(self):
        self.assertIsNone(pick_best_contract([], date(2025, 6, 15), 580.0, 90, 0.05))

    def test_missing_fields_skipped(self):
        chain = [
            {"strike_price": None, "expiration_date": "2025-09-19"},
            {"strike_price": 550, "expiration_date": None},
            {"strike_price": 555, "expiration_date": "2025-09-19"},
        ]
        pick = pick_best_contract(chain, date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertEqual(pick["strike_price"], 555)

    def test_strike_error_weighted_more_than_dte_error(self):
        # 1% strike error (K=574 at spot=580 is ~1% OTM, target 5%) costs 40 vs target — d=40
        # vs a 30-day DTE miss at the right strike — d=30. Strike-correct contract wins.
        chain = [
            {"strike_price": 574, "expiration_date": "2025-09-19"},  # 95 DTE, ~1% OTM — bad strike
            {"strike_price": 551, "expiration_date": "2025-08-15"},  # 60 DTE, 5% OTM — bad DTE
        ]
        # Distance for #1: |95-90| + 10*|0.0103-0.05|*100 = 5 + 39.7 = ~44.7
        # Distance for #2: |60-90| + 10*|0.05-0.05|*100 = 30 + 0 = 30
        pick = pick_best_contract(chain, date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertEqual(pick["strike_price"], 551)


class TestPickFallbackContract(unittest.TestCase):
    def test_rounds_strike_to_nearest_5(self):
        # spot=580, 5% OTM target = 551 → round to 550.
        pick = pick_fallback_contract(date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertEqual(pick["strike_price"], 550.0)
        self.assertEqual(pick["contract_type"], "put")
        self.assertEqual(pick["underlying_ticker"], "SPY")

    def test_picks_nearest_third_friday(self):
        # June 15 + 90 days = Sep 13 → nearest 3rd Friday is Sep 19.
        pick = pick_fallback_contract(date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertEqual(pick["expiration_date"], "2025-09-19")

    def test_builds_correct_occ_ticker(self):
        pick = pick_fallback_contract(date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertEqual(pick["ticker"], "O:SPY250919P00550000")


if __name__ == "__main__":
    unittest.main()
