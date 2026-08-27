"""Tests for parsers/implied_dividend.py.

Uses synthetic option chains built from Black-Scholes with a known q. Since
BS prices for the same (S, K, T, r, q, σ) satisfy put-call parity exactly,
the solver should recover q to high precision when the chain is clean, and
gracefully fall through the multi-tier fallback when it isn't.

No network, no Polygon.
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from implied_dividend import (  # noqa: E402
    HARDCODED_YIELDS,
    MIN_OI_PER_LEG,
    SANE_Q_HIGH,
    solve_q,
)
from options_pricer import black_scholes  # noqa: E402


def _make_chain(spot: float, strikes: list[float], T: float, r: float,
                q: float, sigma: float = 0.20,
                oi: int = 100) -> pd.DataFrame:
    """Build a clean call+put chain priced via BS with a known q. Calls and
    puts at each strike satisfy PCP exactly (to FP precision)."""
    rows = []
    dte = int(round(T * 365))
    for K in strikes:
        for opt in ("call", "put"):
            res = black_scholes(spot, K, T, r, q, sigma, opt)
            rows.append({
                "contract_type": opt,
                "strike": float(K),
                "polygon_price": res["price"],
                "polygon_open_interest": oi,
                "underlying_price": spot,
                "dte": dte,
            })
    return pd.DataFrame(rows)


class TestPCPMedianRecoversKnownQ(unittest.TestCase):
    """Clean synthetic chains — solver should hit the true q to ~4 decimals."""

    def test_recovers_q_2pct(self):
        chain = _make_chain(100.0, [95, 97.5, 100, 102.5, 105],
                            T=0.25, r=0.04, q=0.02)
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "pcp-median")
        self.assertAlmostEqual(r["q"], 0.02, places=3)
        self.assertGreaterEqual(r["n_strikes"], 3)

    def test_recovers_q_zero(self):
        chain = _make_chain(100.0, [95, 100, 105], T=0.25, r=0.04, q=0.0)
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "pcp-median")
        self.assertAlmostEqual(r["q"], 0.0, places=3)

    def test_recovers_long_dated(self):
        # Long-dated options are q-sensitive — make sure we're robust
        chain = _make_chain(500.0, [475, 490, 500, 510, 525],
                            T=2.0, r=0.04, q=0.015)
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "pcp-median")
        self.assertAlmostEqual(r["q"], 0.015, places=3)


class TestFallbackTiering(unittest.TestCase):
    """When PCP can't produce a sane result, the right fallback fires."""

    def _no_pcp_chain(self) -> pd.DataFrame:
        """A chain that PCP-median will reject (one strike, zero OI)."""
        return _make_chain(100.0, [50], T=0.25, r=0.04, q=0.02, oi=0)

    def test_too_few_strike_pairs_falls_to_hardcoded(self):
        # 2 strikes only → below MIN_STRIKES_PCP (3) → fallback
        chain = _make_chain(100.0, [95, 105], T=0.25, r=0.04, q=0.02)
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "hardcoded")
        self.assertEqual(r["q"], HARDCODED_YIELDS["SPY"])

    def test_far_from_spot_strikes_fall_to_hardcoded(self):
        # All strikes well outside ATM band → no pairs survive filter
        chain = _make_chain(100.0, [50, 60, 150, 160],
                            T=0.25, r=0.04, q=0.02)
        r = solve_q(chain, "QQQ", r=0.04)
        self.assertEqual(r["method"], "hardcoded")
        self.assertEqual(r["q"], HARDCODED_YIELDS["QQQ"])

    def test_zero_oi_falls_to_hardcoded(self):
        chain = _make_chain(100.0, [95, 100, 105],
                            T=0.25, r=0.04, q=0.02,
                            oi=MIN_OI_PER_LEG - 1)
        r = solve_q(chain, "AAPL", r=0.04)
        self.assertEqual(r["method"], "hardcoded")
        self.assertEqual(r["q"], HARDCODED_YIELDS["AAPL"])

    def test_unknown_ticker_no_pcp_returns_zero(self):
        chain = self._no_pcp_chain()
        r = solve_q(chain, "XYZ-NOT-A-TICKER", r=0.04)
        self.assertEqual(r["method"], "zero")
        self.assertEqual(r["q"], 0.0)
        self.assertEqual(r["n_strikes"], 0)

    def test_empty_chain_uses_hardcoded_for_known(self):
        empty = pd.DataFrame(columns=[
            "contract_type", "strike", "polygon_price",
            "polygon_open_interest", "underlying_price", "dte",
        ])
        r = solve_q(empty, "SPY", r=0.04)
        self.assertEqual(r["method"], "hardcoded")

    def test_empty_chain_unknown_ticker_returns_zero(self):
        empty = pd.DataFrame(columns=[
            "contract_type", "strike", "polygon_price",
            "polygon_open_interest", "underlying_price", "dte",
        ])
        r = solve_q(empty, "XYZ", r=0.04)
        self.assertEqual(r["method"], "zero")

    def test_implied_q_outside_sane_band_falls_back(self):
        # Build chain priced with q above the sane upper bound -> reject
        bad_q = SANE_Q_HIGH + 0.05
        chain = _make_chain(100.0, [95, 100, 105], T=0.25, r=0.04, q=bad_q)
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "hardcoded")

    def test_nan_prices_skipped_then_fall_back(self):
        # Replace all option prices with NaN — every pair gets filtered
        chain = _make_chain(100.0, [95, 100, 105], T=0.25, r=0.04, q=0.02)
        chain["polygon_price"] = float("nan")
        r = solve_q(chain, "QQQ", r=0.04)
        self.assertEqual(r["method"], "hardcoded")


class TestRobustness(unittest.TestCase):
    """Median is robust to a single bad pair — that's the whole point."""

    def test_one_corrupted_strike_does_not_break_median(self):
        chain = _make_chain(100.0, [92.5, 95, 100, 102.5, 105],
                            T=0.25, r=0.04, q=0.02)
        # Corrupt the 92.5-call price by adding $5. With 5 good strikes and
        # one bad, the median should still recover q≈2% within ~1pp.
        mask = (chain["contract_type"] == "call") & (chain["strike"] == 92.5)
        chain.loc[mask, "polygon_price"] = (
            chain.loc[mask, "polygon_price"].iloc[0] + 5.0
        )
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "pcp-median")
        self.assertAlmostEqual(r["q"], 0.02, places=2)

    def test_strike_only_call_no_put_pair_skipped(self):
        # Build chain, drop one put — that strike's PCP solve should be
        # skipped without breaking the others.
        chain = _make_chain(100.0, [95, 97.5, 100, 102.5, 105],
                            T=0.25, r=0.04, q=0.02)
        chain = chain[~((chain["contract_type"] == "put")
                        & (chain["strike"] == 95))].reset_index(drop=True)
        r = solve_q(chain, "SPY", r=0.04)
        self.assertEqual(r["method"], "pcp-median")
        self.assertAlmostEqual(r["q"], 0.02, places=3)


class TestReturnStructure(unittest.TestCase):
    """Public contract — always returns a dict with all three keys."""

    def test_always_returns_dict_never_none(self):
        # Every code path should produce a YieldResult, never None.
        chains_and_tickers = [
            (_make_chain(100, [95, 100, 105], 0.25, 0.04, 0.02), "SPY"),  # PCP
            (_make_chain(100, [50], 0.25, 0.04, 0.02, oi=0), "SPY"),      # hardcoded
            (_make_chain(100, [50], 0.25, 0.04, 0.02, oi=0), "XYZ"),      # zero
        ]
        for chain, ticker in chains_and_tickers:
            r = solve_q(chain, ticker, r=0.04)
            self.assertIsInstance(r, dict)
            self.assertIn("q", r)
            self.assertIn("method", r)
            self.assertIn("n_strikes", r)
            self.assertIn(r["method"], {"pcp-median", "hardcoded", "zero"})


if __name__ == "__main__":
    unittest.main()
