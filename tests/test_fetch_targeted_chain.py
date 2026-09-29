"""Tests for parsers/fetch_targeted_chain.py.

The HTTP layer is exercised by manual run + integration test (Task 14).
These tests cover the pure-Python selection logic.
"""
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from fetch_targeted_chain import pick_nearest_contract  # noqa: E402


def _contract(strike, expiry, *, price=1.0, iv=0.20, oi=100, contract_type="put"):
    return {
        "details": {"strike_price": strike, "expiration_date": expiry,
                    "contract_type": contract_type,
                    "ticker": f"O:{expiry}P{strike}"},
        "day":     {"close": price},
        "implied_volatility": iv,
        "greeks":  {"delta": -0.30, "gamma": 0.01, "vega": 0.20, "theta": -0.05},
        "open_interest": oi,
        "last_quote": {"bid": price - 0.05, "ask": price + 0.05},
    }


class PickNearestContractTests(unittest.TestCase):
    def test_picks_exact_strike_and_expiry_match(self):
        chain = [
            _contract(100, "2026-12-18"),
            _contract(105, "2026-12-18"),
            _contract(110, "2026-12-18"),
        ]
        got = pick_nearest_contract(chain, target_strike=105.0,
                                     target_expiry=date(2026, 12, 18),
                                     contract_type="put")
        self.assertEqual(got["strike"], 105.0)

    def test_prefers_expiry_close_then_strike_close(self):
        # Two strikes equidistant; expiry tie-break by closest expiry.
        chain = [
            _contract(100, "2026-12-18"),
            _contract(110, "2026-12-18"),  # |105-110|=5, |105-100|=5
            _contract(105, "2027-01-15"),  # exact strike but wrong expiry
        ]
        got = pick_nearest_contract(chain, target_strike=105.0,
                                     target_expiry=date(2026, 12, 18),
                                     contract_type="put")
        # Expiry match is the primary key — should pick from 2026-12-18.
        self.assertEqual(str(got["expiration_date"])[:10], "2026-12-18")

    def test_filters_by_contract_type(self):
        chain = [
            _contract(105, "2026-12-18", contract_type="call"),
            _contract(105, "2026-12-18", contract_type="put"),
        ]
        got = pick_nearest_contract(chain, target_strike=105.0,
                                     target_expiry=date(2026, 12, 18),
                                     contract_type="put")
        self.assertEqual(got["contract_type"], "put")

    def test_skips_zero_oi_when_alternative_exists(self):
        chain = [
            _contract(105, "2026-12-18", oi=0),
            _contract(105, "2026-12-18", oi=500),
        ]
        got = pick_nearest_contract(chain, target_strike=105.0,
                                     target_expiry=date(2026, 12, 18),
                                     contract_type="put")
        self.assertEqual(got["open_interest"], 500)

    def test_returns_none_when_no_contracts(self):
        got = pick_nearest_contract([], target_strike=105.0,
                                     target_expiry=date(2026, 12, 18),
                                     contract_type="put")
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()
