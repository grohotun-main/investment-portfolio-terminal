"""Tests for parsers/nav_basis.py — canonical (marked) vs return-basis
(statement) NAV reconciliation (AUDIT-NAV)."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from nav_basis import canonical_nav, return_basis_nav, nav_reconciliation  # noqa: E402


class TestCanonicalNav(unittest.TestCase):
    def _marked(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"statement_date": pd.Timestamp("2026-05-31"), "account_id": "A",
             "market_value": 100.0},
            {"statement_date": pd.Timestamp("2026-06-11"), "account_id": "A",
             "market_value": 1000.0},
            {"statement_date": pd.Timestamp("2026-06-11"), "account_id": "TEST-1",
             "market_value": 9999.0},
        ])

    def test_sums_latest_snapshot_only(self) -> None:
        self.assertAlmostEqual(canonical_nav(self._marked()), 10999.0, places=2)

    def test_excludes_demo_test_accounts(self) -> None:
        self.assertAlmostEqual(
            canonical_nav(self._marked(), exclude_account_ids={"TEST-1"}),
            1000.0, places=2)

    def test_empty_frame_returns_zero(self) -> None:
        self.assertEqual(canonical_nav(pd.DataFrame()), 0.0)

    def test_missing_statement_date_column_returns_zero(self) -> None:
        # No date column -> no "latest snapshot" to sum; guard returns 0.0.
        df = pd.DataFrame([{"account_id": "A", "market_value": 100.0}])
        self.assertEqual(canonical_nav(df), 0.0)


class TestReturnBasisNav(unittest.TestCase):
    def test_takes_last_nav(self) -> None:
        twr = pd.DataFrame({"month": ["2026-04", "2026-05"],
                            "nav": [3400000.00, 3500000.00]})
        self.assertAlmostEqual(return_basis_nav(twr), 3500000.00, places=2)

    def test_empty_or_missing_column_returns_zero(self) -> None:
        self.assertEqual(return_basis_nav(pd.DataFrame()), 0.0)
        self.assertEqual(return_basis_nav(pd.DataFrame({"month": ["x"]})), 0.0)


class TestNavReconciliation(unittest.TestCase):
    def test_gap_and_caption(self) -> None:
        rec = nav_reconciliation(3425155.22, 3500000.00)
        self.assertAlmostEqual(rec["gap"], -74844.78, places=2)
        self.assertIn("Return-basis NAV", rec["caption"])
        self.assertIn("marked value", rec["caption"])
