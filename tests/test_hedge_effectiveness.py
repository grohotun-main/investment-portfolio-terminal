"""Tests for parsers/hedge_effectiveness.py.

Covers:
* Lot reconstruction: BUY-open + SELL-close, partial closes, cross-account
  pairing, strike resolution via positions, txns with already-negative qty
  for SELLs, missing-strike legacy Harbor BUYs.
* Drawdown episode detection: peak/trough/recover boundaries, threshold,
  end-of-data unrecovered episodes, overlapping nested troughs.
* Sleeve MV aggregation across multiple open lots.
* Episode-attach: peak/trough/recover lookups with weekend fallback.
"""
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.hedge_effectiveness import (  # noqa: E402
    Lot,
    attach_sleeve_to_episodes,
    build_daily_sleeve_mv,
    build_strike_resolver,
    compare_to_statement_mv,
    find_coverage_gaps,
    find_drawdown_episodes,
    reconstruct_lots,
    reprice_lots_daily,
    _build_close_lookup,
    _lookup_close,
)


def _mk_txn(rows):
    """Build a transactions DataFrame from terse tuples:
    (settlement_date, transaction_type, qty, amount, description, account_id).
    """
    return pd.DataFrame([
        {
            "settlement_date": pd.Timestamp(d),
            "trade_date": pd.Timestamp(d),
            "broker": "harbor",
            "account_id": acc,
            "transaction_type": tt,
            "symbol": None,
            "cusip": None,
            "description": desc,
            "quantity": qty,
            "price": 0.0,
            "amount": amt,
            "source_file": "test",
        }
        for d, tt, qty, amt, desc, acc in rows
    ])


def _mk_pos(rows):
    """(statement_date, asset_class, account_id, qty, mv, description)."""
    return pd.DataFrame([
        {
            "statement_date": pd.Timestamp(d),
            "broker": "harbor",
            "account_id": acc,
            "asset_class": ac,
            "symbol": "X",
            "description": desc,
            "quantity": qty,
            "market_value": mv,
        }
        for d, ac, acc, qty, mv, desc in rows
    ])


class LotReconstructionTests(unittest.TestCase):

    def test_simple_buy_then_sell_closes_lot(self):
        # BUY 10 contracts SPY 600P on 2025-06-01, SELL all on 2025-07-01.
        # Both descriptions carry inline strike — no resolver needed.
        txn = _mk_txn([
            ("2025-06-01", "buy",  10, -10000,
             "PUT SPY 09/19/25 600 OPEN CONTRACT", "A"),
            ("2025-07-01", "sell", -10,  6000,
             "PUT SPY 09/19/25 600 CLOSE CONTRACT", "A"),
        ])
        lots = reconstruct_lots(txn)
        self.assertEqual(len(lots), 1)
        lot = lots[0]
        self.assertEqual(lot.underlying, "SPY")
        self.assertEqual(lot.opt_type, "put")
        self.assertEqual(lot.strike, 600.0)
        self.assertEqual(lot.expiry, date(2025, 9, 19))
        self.assertEqual(lot.open_date, pd.Timestamp("2025-06-01"))
        self.assertEqual(lot.close_date, pd.Timestamp("2025-07-01"))
        # Premium = 10000 / (10 * 100) = $10/share
        self.assertAlmostEqual(lot.open_premium, 10.0)

    def test_partial_sells_then_full_close(self):
        txn = _mk_txn([
            ("2025-06-01", "buy",  10, -10000,
             "PUT SPY 09/19/25 600 OPEN", "A"),
            ("2025-07-01", "sell", -3,   2200,
             "PUT SPY 09/19/25 600 CLOSE", "A"),
            ("2025-08-01", "sell", -7,   4500,
             "PUT SPY 09/19/25 600 CLOSE", "A"),
        ])
        lots = reconstruct_lots(txn)
        self.assertEqual(len(lots), 1)
        lot = lots[0]
        # Close date is when net qty first hits 0 — second SELL.
        self.assertEqual(lot.close_date, pd.Timestamp("2025-08-01"))
        # qty_at checks mid-life
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-06-15")), 10)
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-07-15")), 7)
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-08-15")), 0)

    def test_sell_with_positive_qty_normalized(self):
        # Some brokers emit SELL with positive qty + tt=sell rather than
        # negative qty. Reconstruct should handle both conventions.
        txn = _mk_txn([
            ("2025-06-01", "buy",  10, -10000,
             "PUT SPY 09/19/25 600 OPEN", "A"),
            ("2025-07-01", "sell", 10,   6000,   # +10 not -10
             "PUT SPY 09/19/25 600 CLOSE", "A"),
        ])
        lots = reconstruct_lots(txn)
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0].close_date, pd.Timestamp("2025-07-01"))

    def test_strike_resolved_from_positions_when_txn_omits_it(self):
        # Harbor PDF descriptions sometimes drop strike from BUY/SELL lines.
        # Resolver uses positions to recover it.
        txn = _mk_txn([
            ("2025-06-01", "buy", 10, -10000,
             "PUT SPY 09/19/25 OPEN CONTRACT", "A"),  # no strike
            ("2025-07-01", "sell", -10, 6000,
             "PUT SPY 09/19/25 CLOSE CONTRACT", "A"),  # no strike
        ])
        pos = _mk_pos([
            ("2025-06-30", "option_put", "A", 10, 8000,
             "PUT SPY 09/19/25 600 STANDARD POORS"),
        ])
        lots = reconstruct_lots(txn, positions=pos)
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0].strike, 600.0)
        self.assertEqual(lots[0].close_date, pd.Timestamp("2025-07-01"))

    def test_cross_account_buy_sell_pair(self):
        # BUY recorded in account A, SELL recorded in account B —
        # Harbor book-and-allocate quirk. Should still close the lot.
        txn = _mk_txn([
            ("2025-06-01", "buy", 10, -10000,
             "PUT SPY 09/19/25 600 OPEN", "A"),
            ("2025-07-01", "sell", -10, 6000,
             "PUT SPY 09/19/25 600 CLOSE", "B"),
        ])
        lots = reconstruct_lots(txn)
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0].close_date, pd.Timestamp("2025-07-01"))

    def test_no_strike_no_resolver_drops_row(self):
        txn = _mk_txn([
            ("2025-06-01", "buy", 10, -10000,
             "PUT SPY 09/19/25 ETF OPEN CONTRACT", "A"),
        ])
        # No positions — can't resolve strike.
        lots = reconstruct_lots(txn)
        self.assertEqual(lots, [])

    def test_ambiguous_strike_resolver_skips(self):
        # Two distinct strikes for same (type, ul, expiry) in positions —
        # resolver returns None.
        txn = _mk_txn([
            ("2025-06-01", "buy", 10, -10000,
             "PUT SPY 09/19/25 OPEN CONTRACT", "A"),
        ])
        pos = _mk_pos([
            ("2025-06-30", "option_put", "A", 10, 8000,
             "PUT SPY 09/19/25 600 STANDARD POORS"),
            ("2025-06-30", "option_put", "B", 5, 3000,
             "PUT SPY 09/19/25 580 STANDARD POORS"),  # different K!
        ])
        lots = reconstruct_lots(txn, positions=pos)
        self.assertEqual(lots, [])

    def test_test_broker_excluded_from_resolver(self):
        # TEST-FID / Harbor Test rows shouldn't poison the resolver.
        pos = pd.DataFrame([
            {"statement_date": pd.Timestamp("2025-06-30"),
             "broker": "Harbor Test", "account_id": "TEST-Harbor",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 09/19/25 580 STANDARD POORS",
             "quantity": 1, "market_value": 800},
            {"statement_date": pd.Timestamp("2025-06-30"),
             "broker": "harbor", "account_id": "A",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 09/19/25 600 STANDARD POORS",
             "quantity": 10, "market_value": 8000},
        ])
        resolver = build_strike_resolver(pos)
        # Only the real broker's strike should be present.
        key = ("put", "SPY", date(2025, 9, 19))
        self.assertEqual(resolver[key], {600.0})

    def test_empty_or_no_put_txns_returns_empty(self):
        self.assertEqual(reconstruct_lots(None), [])
        self.assertEqual(reconstruct_lots(pd.DataFrame()), [])
        # All non-PUT
        txn = _mk_txn([
            ("2025-06-01", "buy", 1, -100, "AAPL COMMON STOCK", "A"),
        ])
        self.assertEqual(reconstruct_lots(txn), [])


class LotQtyAtTests(unittest.TestCase):
    """qty_at should be 0 outside the lot life, ramp through partial sells."""

    def _mk_lot(self):
        lot = Lot(
            opt_type="put", underlying="SPY", expiry=date(2025, 9, 19),
            strike=600.0, open_date=pd.Timestamp("2025-06-01"),
            close_date=pd.Timestamp("2025-08-01"),
            open_qty=10, open_premium=10.0,
            qty_changes=[
                (pd.Timestamp("2025-06-01"), 10),
                (pd.Timestamp("2025-07-01"), -3),
                (pd.Timestamp("2025-08-01"), -7),
            ],
        )
        return lot

    def test_pre_open_is_zero(self):
        lot = self._mk_lot()
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-05-31")), 0)

    def test_at_open_full_qty(self):
        lot = self._mk_lot()
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-06-01")), 10)

    def test_during_partial_close(self):
        lot = self._mk_lot()
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-07-15")), 7)

    def test_after_full_close_is_zero(self):
        lot = self._mk_lot()
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-08-15")), 0)

    def test_after_expiry_is_zero(self):
        lot = self._mk_lot()
        # Force the close to None so only expiry filtering matters
        lot.close_date = None
        # Beyond expiry the lot contributes 0 regardless of qty_changes.
        self.assertEqual(lot.qty_at(pd.Timestamp("2025-09-20")), 0)


class DrawdownEpisodeTests(unittest.TestCase):

    def _series(self, values, start="2025-01-01"):
        dates = pd.bdate_range(start, periods=len(values))
        return pd.DataFrame({"date": dates, "close": values})

    def test_single_episode_recovered(self):
        # 100 -> 95 -> 90 -> 95 -> 100 -> 105 (recover at index 4)
        spy = self._series([100, 95, 90, 95, 100, 105])
        ep = find_drawdown_episodes(spy, threshold_pct=5.0)
        self.assertEqual(len(ep), 1)
        row = ep.iloc[0]
        self.assertEqual(row["peak_close"], 100.0)
        self.assertEqual(row["trough_close"], 90.0)
        self.assertAlmostEqual(row["decline_pct"], -10.0)
        self.assertTrue(row["recovered"])

    def test_below_threshold_not_recorded(self):
        spy = self._series([100, 99, 100, 105])
        ep = find_drawdown_episodes(spy, threshold_pct=3.0)
        self.assertEqual(len(ep), 0)

    def test_ongoing_at_end_of_data_kept(self):
        # Drops 10% and never recovers.
        spy = self._series([100, 95, 90])
        ep = find_drawdown_episodes(spy, threshold_pct=5.0)
        self.assertEqual(len(ep), 1)
        self.assertFalse(ep.iloc[0]["recovered"])
        self.assertIsNone(ep.iloc[0]["recover_date"])

    def test_multiple_episodes(self):
        # 100->90->100 (recover) ->85 (new ep, below 100) ->100
        spy = self._series([100, 90, 100, 85, 100])
        ep = find_drawdown_episodes(spy, threshold_pct=5.0)
        self.assertEqual(len(ep), 2)
        self.assertAlmostEqual(ep.iloc[0]["decline_pct"], -10.0)
        self.assertAlmostEqual(ep.iloc[1]["decline_pct"], -15.0)

    def test_nested_deeper_trough_updates(self):
        # Same peak, deeper trough later — same episode, lower trough.
        spy = self._series([100, 95, 92, 90, 88, 100])
        ep = find_drawdown_episodes(spy, threshold_pct=5.0)
        self.assertEqual(len(ep), 1)
        self.assertEqual(ep.iloc[0]["trough_close"], 88.0)

    def test_empty_history(self):
        ep = find_drawdown_episodes(
            pd.DataFrame(columns=["date", "close"]), threshold_pct=3.0
        )
        self.assertTrue(ep.empty)


class SleeveAggregationTests(unittest.TestCase):
    """build_daily_sleeve_mv should sum across all open lots on each date,
    looking up daily close from option_history."""

    def _mk_history(self, rows):
        """Rows: (opt_type, underlying, expiry_date, strike, date, close)."""
        return pd.DataFrame([
            {"opt_type": tt, "underlying": und,
             "expiry": pd.Timestamp(exp), "strike": float(K),
             "date": pd.Timestamp(d), "close": float(c)}
            for tt, und, exp, K, d, c in rows
        ])

    def test_single_lot_mv_from_close(self):
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
        ])
        # Close fluctuates day-to-day; sleeve_mv tracks it.
        hist = self._mk_history([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-03", 6.5),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-04", 4.2),
        ])
        out = build_daily_sleeve_mv(
            txn, hist, start_date="2025-06-02", end_date="2025-06-04"
        )
        self.assertEqual(len(out), 3)
        # mv = qty * 100 * close = 5 * 100 * close
        self.assertAlmostEqual(out.iloc[0]["sleeve_mv"], 2500.0)   # 5*100*5.0
        self.assertAlmostEqual(out.iloc[1]["sleeve_mv"], 3250.0)   # 5*100*6.5
        self.assertAlmostEqual(out.iloc[2]["sleeve_mv"], 2100.0)   # 5*100*4.2

    def test_two_lots_sum_per_day(self):
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
            ("2025-06-02", "buy", 3, -900,
             "PUT QQQ 09/19/25 400 OPEN", "A"),
        ])
        hist = self._mk_history([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
            ("put", "QQQ", "2025-09-19", 400.0, "2025-06-02", 3.0),
        ])
        out = build_daily_sleeve_mv(
            txn, hist, start_date="2025-06-02", end_date="2025-06-02"
        )
        self.assertEqual(len(out), 1)
        # SPY: 5*100*5 = 2500; QQQ: 3*100*3 = 900; total = 3400
        self.assertAlmostEqual(out.iloc[0]["sleeve_mv"], 3400.0)
        self.assertEqual(out.iloc[0]["n_open_lots"], 2)

    def test_close_forward_fills_across_gap(self):
        # Lookup should forward-fill from the most-recent earlier close
        # when an exact-date row is missing (weekend / zero-volume day).
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
        ])
        hist = self._mk_history([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
            # Skip 06-03; 06-04 has next close.
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-04", 6.5),
        ])
        out = build_daily_sleeve_mv(
            txn, hist, start_date="2025-06-02", end_date="2025-06-04"
        )
        # Date grid uses dates from hist when trading_dates not provided,
        # so 06-03 isn't in the grid — out has 06-02 and 06-04 only.
        self.assertEqual(len(out), 2)

    def test_closed_lot_contributes_zero_after_close(self):
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
            ("2025-06-04", "sell", -5, 3000,
             "PUT SPY 09/19/25 600 CLOSE", "A"),
        ])
        hist = self._mk_history([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-03", 5.5),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-04", 6.0),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-05", 7.0),
        ])
        out = build_daily_sleeve_mv(
            txn, hist, start_date="2025-06-02", end_date="2025-06-05"
        )
        # Lot is open 06-02 and 06-03, closes 06-04, contributes 0 on 06-05.
        # qty_at on close_date returns 0 (per Lot.qty_at filter), so the
        # row on 06-04 already shows 0 contribution — drops out of per-lot.
        # The aggregation only includes dates with at least one open lot.
        dates = sorted(out["date"].tolist())
        self.assertEqual(dates, [pd.Timestamp("2025-06-02"),
                                 pd.Timestamp("2025-06-03")])

    def test_empty_history_returns_empty(self):
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
        ])
        out = build_daily_sleeve_mv(txn, pd.DataFrame())
        self.assertTrue(out.empty)


class EpisodeAttachTests(unittest.TestCase):

    def _mk_sleeve(self, rows):
        return pd.DataFrame([
            {"date": pd.Timestamp(d), "sleeve_mv": v, "n_open_lots": 1}
            for d, v in rows
        ])

    def test_lookup_at_episode_dates(self):
        episodes = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-06-01"),
            "peak_close": 100.0,
            "trough_date": pd.Timestamp("2025-06-10"),
            "trough_close": 90.0,
            "decline_pct": -10.0,
            "recover_date": pd.Timestamp("2025-06-20"),
            "recovered": True,
            "duration_days": 19,
        }])
        sleeve = self._mk_sleeve([
            ("2025-06-01", 5000),
            ("2025-06-10", 12000),
            ("2025-06-20", 4000),
        ])
        out = attach_sleeve_to_episodes(episodes, sleeve)
        self.assertEqual(out.iloc[0]["sleeve_mv_peak"], 5000)
        self.assertEqual(out.iloc[0]["sleeve_mv_trough"], 12000)
        self.assertEqual(out.iloc[0]["sleeve_mv_recover"], 4000)
        self.assertEqual(out.iloc[0]["sleeve_gain_peak_to_trough"], 7000)
        self.assertAlmostEqual(out.iloc[0]["sleeve_gain_pct"], 140.0)

    def test_weekend_lookup_falls_back_to_prior(self):
        # Episode peak on a weekend — function falls back to prior weekday.
        episodes = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-06-07"),  # Saturday
            "peak_close": 100.0,
            "trough_date": pd.Timestamp("2025-06-10"),
            "trough_close": 90.0,
            "decline_pct": -10.0,
            "recover_date": None,
            "recovered": False,
            "duration_days": 3,
        }])
        sleeve = self._mk_sleeve([
            ("2025-06-06", 5000),  # Friday
            ("2025-06-09", 8000),
            ("2025-06-10", 12000),
        ])
        out = attach_sleeve_to_episodes(episodes, sleeve)
        # Saturday peak — should pick up Friday's 5000.
        self.assertEqual(out.iloc[0]["sleeve_mv_peak"], 5000)
        self.assertEqual(out.iloc[0]["sleeve_mv_trough"], 12000)

    def test_empty_sleeve_returns_nan_columns(self):
        episodes = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-06-01"),
            "peak_close": 100.0,
            "trough_date": pd.Timestamp("2025-06-10"),
            "trough_close": 90.0,
            "decline_pct": -10.0,
            "recover_date": pd.Timestamp("2025-06-20"),
            "recovered": True,
            "duration_days": 19,
        }])
        out = attach_sleeve_to_episodes(
            episodes, pd.DataFrame(columns=["date", "sleeve_mv", "n_open_lots"])
        )
        self.assertTrue(pd.isna(out.iloc[0]["sleeve_mv_peak"]))
        self.assertTrue(pd.isna(out.iloc[0]["sleeve_gain_peak_to_trough"]))


class StatementComparisonTests(unittest.TestCase):

    def test_basic_compare(self):
        sleeve = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-30"), "sleeve_mv": 4500,
             "n_open_lots": 1},
            {"date": pd.Timestamp("2025-07-31"), "sleeve_mv": 6200,
             "n_open_lots": 1},
        ])
        pos = _mk_pos([
            ("2025-06-30", "option_put", "A", 10, 5000, "PUT SPY 600"),
            ("2025-07-31", "option_put", "A", 10, 6000, "PUT SPY 600"),
        ])
        comp = compare_to_statement_mv(sleeve, pos)
        self.assertEqual(len(comp), 2)
        # Repriced 4500 vs statement 5000 = -10% error.
        self.assertAlmostEqual(comp.iloc[0]["error_pct"], -10.0, places=1)
        # Repriced 6200 vs statement 6000 = +3.33%
        self.assertAlmostEqual(comp.iloc[1]["error_pct"], 100/30, places=1)

    def test_empty_inputs(self):
        comp = compare_to_statement_mv(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(comp.empty)


class CloseLookupTests(unittest.TestCase):
    """The close-lookup primitives that replace the old IV/pricer flow."""

    def test_build_close_lookup_keys_match_lot_format(self):
        hist = pd.DataFrame([
            {"opt_type": "put", "underlying": "SPY",
             "expiry": pd.Timestamp("2025-09-19"), "strike": 600.0,
             "date": pd.Timestamp("2025-06-02"), "close": 5.0},
            {"opt_type": "put", "underlying": "SPY",
             "expiry": pd.Timestamp("2025-09-19"), "strike": 600.0,
             "date": pd.Timestamp("2025-06-03"), "close": 5.5},
        ])
        lookup = _build_close_lookup(hist)
        # Key uses datetime.date for expiry (matching Lot.expiry).
        key = ("put", "SPY", date(2025, 9, 19), 600.0)
        self.assertIn(key, lookup)
        self.assertEqual(len(lookup[key]), 2)

    def test_lookup_close_exact_date(self):
        df = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-02"), "close": 5.0},
            {"date": pd.Timestamp("2025-06-03"), "close": 6.5},
        ])
        # Exact-date match returns days_stale=0.
        self.assertEqual(_lookup_close(df, pd.Timestamp("2025-06-02")), (5.0, 0))
        self.assertEqual(_lookup_close(df, pd.Timestamp("2025-06-03")), (6.5, 0))

    def test_lookup_close_forward_fills(self):
        # Weekend fallback: 06-07 is Saturday, falls back to Friday 06-06.
        df = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-06"), "close": 4.0},
            {"date": pd.Timestamp("2025-06-09"), "close": 4.2},
        ])
        # Sat from Fri = 1 day stale, Sun from Fri = 2 days stale.
        self.assertEqual(_lookup_close(df, pd.Timestamp("2025-06-07")), (4.0, 1))
        self.assertEqual(_lookup_close(df, pd.Timestamp("2025-06-08")), (4.0, 2))

    def test_lookup_close_before_first_returns_nan(self):
        df = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-02"), "close": 5.0},
        ])
        # Pre-history sentinel: (NaN, -1) to distinguish from "exact match".
        close, days = _lookup_close(df, pd.Timestamp("2025-05-01"))
        self.assertTrue(pd.isna(close))
        self.assertEqual(days, -1)

    def test_lookup_close_long_gap(self):
        # 5-day gap (illiquid contract): from Mon to following Mon = 7 stale.
        df = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-02"), "close": 5.0},
        ])
        self.assertEqual(_lookup_close(df, pd.Timestamp("2025-06-09")), (5.0, 7))

    def test_empty_history_returns_empty_lookup(self):
        self.assertEqual(_build_close_lookup(pd.DataFrame()), {})
        self.assertEqual(_build_close_lookup(None), {})


class RepriceStalenessTests(unittest.TestCase):
    """reprice_lots_daily should carry per-row days_stale; aggregation
    should surface n_stale_lots / frac_stale_mv / max_days_stale per day."""

    def _mk_history(self, rows):
        return pd.DataFrame([
            {"opt_type": tt, "underlying": und,
             "expiry": pd.Timestamp(exp), "strike": float(K),
             "date": pd.Timestamp(d), "close": float(c)}
            for tt, und, exp, K, d, c in rows
        ])

    def test_reprice_carries_days_stale_zero_when_fresh(self):
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
        ])
        hist = self._mk_history([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-03", 6.0),
        ])
        out = reprice_lots_daily(
            reconstruct_lots(txn), hist,
            start_date=pd.Timestamp("2025-06-02"),
            end_date=pd.Timestamp("2025-06-03"),
        )
        self.assertIn("days_stale", out.columns)
        self.assertTrue((out["days_stale"] == 0).all())

    def test_aggregation_flags_stale_day(self):
        # Two lots; one has fresh closes, the other goes stale Wed→Fri.
        txn = _mk_txn([
            ("2025-06-02", "buy", 5, -2500,
             "PUT SPY 09/19/25 600 OPEN", "A"),
            ("2025-06-02", "buy", 4, -1200,
             "PUT QQQ 09/19/25 400 OPEN", "A"),
        ])
        # SPY fresh every day. QQQ fresh Mon-Tue, then no bar until Fri.
        hist = self._mk_history([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-03", 5.0),
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-04", 5.0),
            ("put", "QQQ", "2025-09-19", 400.0, "2025-06-02", 3.0),
            ("put", "QQQ", "2025-09-19", 400.0, "2025-06-03", 3.0),
            # No QQQ on 06-04 — forward-fill triggers.
        ])
        out = build_daily_sleeve_mv(
            txn, hist,
            start_date=pd.Timestamp("2025-06-02"),
            end_date=pd.Timestamp("2025-06-04"),
            trading_dates=pd.DatetimeIndex(pd.bdate_range(
                "2025-06-02", "2025-06-04")),
        )
        # Mon, Tue both fresh.
        mon = out[out["date"] == pd.Timestamp("2025-06-02")].iloc[0]
        self.assertEqual(int(mon["n_stale_lots"]), 0)
        self.assertEqual(int(mon["max_days_stale"]), 0)
        self.assertAlmostEqual(float(mon["frac_stale_mv"]), 0.0)
        # Wed: QQQ row uses Tue's close (1 day stale).
        wed = out[out["date"] == pd.Timestamp("2025-06-04")].iloc[0]
        self.assertEqual(int(wed["n_stale_lots"]), 1)
        self.assertEqual(int(wed["max_days_stale"]), 1)
        # SPY MV = 5*100*5 = 2500; QQQ MV = 4*100*3 = 1200; stale share = 1200/3700.
        self.assertAlmostEqual(
            float(wed["frac_stale_mv"]), 1200.0 / 3700.0, places=4
        )


class EpisodeStalenessTests(unittest.TestCase):
    """attach_sleeve_to_episodes should propagate per-date staleness from
    the matched sleeve row to peak/trough/recover_days_stale columns."""

    def test_episode_inherits_stale_from_matched_row(self):
        episodes = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-06-02"),
            "peak_close": 100.0,
            "trough_date": pd.Timestamp("2025-06-04"),
            "trough_close": 90.0,
            "decline_pct": -10.0,
            "recover_date": pd.Timestamp("2025-06-06"),
            "recovered": True,
            "duration_days": 4,
        }])
        # Trough date has a stale close (max_days_stale=2).
        sleeve = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-02"), "sleeve_mv": 5000,
             "n_open_lots": 1, "n_stale_lots": 0, "frac_stale_mv": 0.0,
             "max_days_stale": 0},
            {"date": pd.Timestamp("2025-06-04"), "sleeve_mv": 12000,
             "n_open_lots": 1, "n_stale_lots": 1, "frac_stale_mv": 1.0,
             "max_days_stale": 2},
            {"date": pd.Timestamp("2025-06-06"), "sleeve_mv": 4000,
             "n_open_lots": 1, "n_stale_lots": 0, "frac_stale_mv": 0.0,
             "max_days_stale": 0},
        ])
        out = attach_sleeve_to_episodes(episodes, sleeve)
        self.assertEqual(int(out.iloc[0]["peak_days_stale"]), 0)
        self.assertEqual(int(out.iloc[0]["trough_days_stale"]), 2)
        self.assertEqual(int(out.iloc[0]["recover_days_stale"]), 0)

    def test_no_stale_column_yields_none(self):
        # Legacy callers with sleeve frames lacking max_days_stale —
        # episode staleness columns should be present but None.
        episodes = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-06-02"),
            "peak_close": 100.0,
            "trough_date": pd.Timestamp("2025-06-04"),
            "trough_close": 90.0,
            "decline_pct": -10.0,
            "recover_date": pd.Timestamp("2025-06-06"),
            "recovered": True,
            "duration_days": 4,
        }])
        sleeve = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-02"), "sleeve_mv": 5000,
             "n_open_lots": 1},
            {"date": pd.Timestamp("2025-06-04"), "sleeve_mv": 12000,
             "n_open_lots": 1},
            {"date": pd.Timestamp("2025-06-06"), "sleeve_mv": 4000,
             "n_open_lots": 1},
        ])
        out = attach_sleeve_to_episodes(episodes, sleeve)
        self.assertIsNone(out.iloc[0]["peak_days_stale"])
        self.assertIsNone(out.iloc[0]["trough_days_stale"])


class CoverageGapsTests(unittest.TestCase):
    """find_coverage_gaps surfaces lots that the back-test would silently
    underreport (no_history) or partially miss (pre_history)."""

    def _mk_lot(self, opt_type="put", underlying="SPY",
                expiry=date(2025, 9, 19), strike=600.0,
                open_date="2025-06-01"):
        return Lot(
            opt_type=opt_type, underlying=underlying, expiry=expiry,
            strike=strike, open_date=pd.Timestamp(open_date),
            close_date=None, open_qty=1, open_premium=1.0,
            qty_changes=[(pd.Timestamp(open_date), 1)],
        )

    def _mk_hist(self, rows):
        return pd.DataFrame([
            {"opt_type": tt, "underlying": und,
             "expiry": pd.Timestamp(exp), "strike": float(K),
             "date": pd.Timestamp(d), "close": float(c)}
            for tt, und, exp, K, d, c in rows
        ])

    def test_full_coverage_returns_empty(self):
        lot = self._mk_lot(open_date="2025-06-02")
        hist = self._mk_hist([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
        ])
        self.assertEqual(find_coverage_gaps([lot], hist), [])

    def test_pre_history_flagged(self):
        # Lot opens 2025-01-01 but first bar isn't until 2025-06-02.
        lot = self._mk_lot(open_date="2025-01-01")
        hist = self._mk_hist([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
        ])
        gaps = find_coverage_gaps([lot], hist)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["kind"], "pre_history")
        self.assertEqual(gaps[0]["first_bar_date"], pd.Timestamp("2025-06-02"))
        # Jan 1 -> Jun 2 = 152 days.
        self.assertEqual(gaps[0]["gap_days"], 152)

    def test_no_history_flagged(self):
        # Lot has no matching contract in history (different strike).
        lot = self._mk_lot(strike=600.0)
        hist = self._mk_hist([
            ("put", "SPY", "2025-09-19", 580.0, "2025-06-02", 5.0),  # K=580
        ])
        gaps = find_coverage_gaps([lot], hist)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["kind"], "no_history")
        self.assertIsNone(gaps[0]["first_bar_date"])
        self.assertIsNone(gaps[0]["gap_days"])

    def test_mixed_kinds_both_returned(self):
        lot_pre = self._mk_lot(strike=600.0, open_date="2025-01-01")
        lot_no = self._mk_lot(strike=580.0, open_date="2025-06-02")
        hist = self._mk_hist([
            ("put", "SPY", "2025-09-19", 600.0, "2025-06-02", 5.0),
        ])
        gaps = find_coverage_gaps([lot_pre, lot_no], hist)
        kinds = sorted(g["kind"] for g in gaps)
        self.assertEqual(kinds, ["no_history", "pre_history"])

    def test_empty_inputs(self):
        self.assertEqual(find_coverage_gaps([], pd.DataFrame()), [])
        self.assertEqual(find_coverage_gaps([self._mk_lot()], pd.DataFrame()), [])
        self.assertEqual(find_coverage_gaps([], self._mk_hist([])), [])


if __name__ == "__main__":
    unittest.main()
