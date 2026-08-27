"""Tests for parsers/fetch_spy_holdings.py."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from fetch_spy_holdings import normalize_holdings_frame  # noqa: E402


class NormalizeHoldingsFrameTests(unittest.TestCase):
    def test_drops_cash_without_renormalizing(self):
        # SSGA frames include a "CASH" / "USD" line at the bottom; we drop it.
        raw = pd.DataFrame({
            "Ticker": ["NVDA", "AAPL", "MSFT", "USD", "CASH_USD"],
            "Name":   ["NVIDIA", "APPLE", "MICROSOFT", "USD", "CASH"],
            "Weight": [7.5, 6.8, 6.2, 0.1, 0.0],
            "Sector": ["Tech", "Tech", "Tech", "Cash", "Cash"],
        })
        out = normalize_holdings_frame(raw)
        self.assertNotIn("USD", out["ticker"].tolist())
        self.assertNotIn("CASH_USD", out["ticker"].tolist())
        # Weights remain on a 0-100 scale (no renormalization — SPY is
        # never exactly 100% due to cash drag; preserve audit fidelity).
        self.assertAlmostEqual(out["weight_pct"].sum(), 7.5 + 6.8 + 6.2, places=2)

    def test_uppercases_ticker(self):
        raw = pd.DataFrame({
            "Ticker": ["nvda", "Aapl"],
            "Name":   ["NVIDIA", "APPLE"],
            "Weight": [7.5, 6.8],
            "Sector": ["Tech", "Tech"],
        })
        out = normalize_holdings_frame(raw)
        self.assertEqual(out["ticker"].tolist(), ["NVDA", "AAPL"])

    def test_returns_two_columns(self):
        raw = pd.DataFrame({
            "Ticker": ["SPY"],
            "Name":   ["dummy"],
            "Weight": [100.0],
            "Sector": ["x"],
        })
        out = normalize_holdings_frame(raw)
        self.assertListEqual(sorted(out.columns.tolist()), ["ticker", "weight_pct"])


if __name__ == "__main__":
    unittest.main()
