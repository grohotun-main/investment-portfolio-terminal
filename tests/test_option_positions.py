"""Tests for parsers/option_positions.py.

The parser walks two broker formats:
  - JPM: position description carries (type, ticker, MM/DD/YY, strike).
    BUY rows carry (type, ticker, MM/DD/YY) — strike absent.
  - Fidelity: position description carries (type, name) only.
    BUY rows carry (type, (TICKER), name, expiry like "DEC 18 26", "$strike").

Tests lock down regex fragility, cost-basis recovery across multiple buys,
edge cases that have actually appeared in production statements (parenthesized
quantities, NaN symbol, "ADJ 10:1 STOCK SPLIT" continuations).
"""
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from option_positions import (  # noqa: E402
    CONTRACT_MULT,
    ParsedOption,
    build_option_position_table,
    option_book_aggregates,
    parse_fidelity_buy_desc,
    parse_jpm_buy_desc,
    parse_jpm_option_desc,
)


class JPMPositionDescTests(unittest.TestCase):
    """Canonical JPM position description shapes."""

    def test_basic_put(self):
        got = parse_jpm_option_desc(
            "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 -- ETF"
        )
        self.assertEqual(
            got, ParsedOption("put", "SPY", date(2026, 12, 18), 560.0)
        )

    def test_call(self):
        got = parse_jpm_option_desc(
            "CALL TSLA 06/19/26 250 TESLA INC COM"
        )
        self.assertEqual(
            got, ParsedOption("call", "TSLA", date(2026, 6, 19), 250.0)
        )

    def test_stock_split_continuation_does_not_swallow_strike(self):
        # "ADJ 10:1 STOCK SPLIT" tail must not consume strike; the regex
        # is anchored on the FIRST bare number after the date.
        got = parse_jpm_option_desc(
            "PUT NVDA 12/18/26 135 NVIDIA CORPORATION ADJ 10:1 STOCK SPLIT * tax lots"
        )
        self.assertEqual(
            got, ParsedOption("put", "NVDA", date(2026, 12, 18), 135.0)
        )

    def test_decimal_strike(self):
        got = parse_jpm_option_desc("PUT LDI 12/19/25 2.50 LOANDEPOT INC")
        self.assertEqual(
            got, ParsedOption("put", "LDI", date(2025, 12, 19), 2.50)
        )

    def test_lowercase_input_accepted(self):
        # Case-insensitive — defensively, even though the parser ships
        # are all upper from PDF parsers.
        got = parse_jpm_option_desc("put spy 12/18/26 560 etf")
        self.assertEqual(got and got.opt_type, "put")
        self.assertEqual(got and got.underlying, "SPY")

    def test_returns_none_for_unrelated_row(self):
        self.assertIsNone(parse_jpm_option_desc("SPDR S&P 500 ETF"))

    def test_returns_none_for_non_string(self):
        self.assertIsNone(parse_jpm_option_desc(None))
        self.assertIsNone(parse_jpm_option_desc(float("nan")))

    def test_returns_none_for_invalid_date(self):
        # 13th month — datetime.strptime raises
        self.assertIsNone(parse_jpm_option_desc(
            "PUT SPY 13/18/26 575 something"
        ))

    def test_returns_none_for_zero_strike(self):
        self.assertIsNone(parse_jpm_option_desc(
            "PUT SPY 12/18/26 0 something"
        ))


class JPMBuyDescTests(unittest.TestCase):
    """JPM BUY-transaction descriptions: type + ticker + expiry, no strike."""

    def test_strike_is_nan(self):
        got = parse_jpm_buy_desc(
            "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT Exchange Listed Option"
        )
        self.assertIsNotNone(got)
        self.assertEqual(got.opt_type, "put")
        self.assertEqual(got.underlying, "SPY")
        self.assertEqual(got.expiry, date(2026, 12, 18))
        import math
        self.assertTrue(math.isnan(got.strike))

    def test_stock_split_tag_after_expiry(self):
        got = parse_jpm_buy_desc(
            "PUT NVDA 12/18/26 STOCK SPLIT UNSOLICITED OPEN CONTRACT"
        )
        self.assertEqual(got and got.opt_type, "put")
        self.assertEqual(got and got.underlying, "NVDA")
        self.assertEqual(got and got.expiry, date(2026, 12, 18))

    def test_returns_none_when_no_match(self):
        self.assertIsNone(parse_jpm_buy_desc("BUY SPY 100 shares"))
        self.assertIsNone(parse_jpm_buy_desc(None))


class FidelityBuyDescTests(unittest.TestCase):
    """Fidelity BUY descriptions carry parens-wrapped ticker, written-out
    date, dollar-sign strike."""

    def test_basic_put(self):
        got = parse_fidelity_buy_desc(
            "PUT (SPY) SPDR S&P500 ETF 1234567AB You Bought DEC 18 26 $560 "
            "(100 SHS) OPENING refer to confirm for Lot detail TRANSACTION"
        )
        self.assertEqual(
            got, ParsedOption("put", "SPY", date(2026, 12, 18), 560.0)
        )

    def test_call(self):
        got = parse_fidelity_buy_desc(
            "CALL (AAPL) APPLE INC ABC123 You Bought JAN 16 26 $200 (100 SHS) OPENING"
        )
        self.assertEqual(
            got, ParsedOption("call", "AAPL", date(2026, 1, 16), 200.0)
        )

    def test_strike_with_comma(self):
        got = parse_fidelity_buy_desc(
            "PUT (NVDA) NVIDIA CORPORATION X You Bought DEC 18 26 $1,200 (100 SHS) OPENING"
        )
        self.assertEqual(got and got.strike, 1200.0)

    def test_strike_with_decimal(self):
        got = parse_fidelity_buy_desc(
            "PUT (GME) GAMESTOP X You Bought DEC 18 26 $22.50 (100 SHS) OPENING"
        )
        self.assertEqual(got and got.strike, 22.50)

    def test_four_digit_year(self):
        got = parse_fidelity_buy_desc(
            "PUT (SPY) SPDR X You Bought DEC 18 2026 $560 (100 SHS) OPENING"
        )
        self.assertEqual(got and got.expiry, date(2026, 12, 18))

    def test_returns_none_for_position_only(self):
        # Position-style description (no expiry, no strike) — must return
        # None so callers walk back to the BUY txn instead.
        self.assertIsNone(parse_fidelity_buy_desc("PUT NVIDIA CORPORATION"))

    def test_returns_none_for_unrelated(self):
        self.assertIsNone(parse_fidelity_buy_desc(
            "DIVIDEND RECEIVED ON SPY"
        ))


class BuildTableTests(unittest.TestCase):
    """End-to-end: positions + transactions → parsed table.

    Synthetic micro-fixtures cover JPM-only, Fidelity-only, both brokers
    holding the same contract, and an unparsed orphan.
    """

    def _make_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        positions = pd.DataFrame([
            {"account_id": "JPM-A", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 9, "market_value": 7335.0},
            {"account_id": "FID-A", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": "NVDA",
             "description": "PUT NVIDIA CORPORATION",
             "quantity": 11, "market_value": 2585.0},
            {"account_id": "ORPH", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": None,
             "description": "PUT something unknown",
             "quantity": 1, "market_value": 100.0},
            # An equity row — must be excluded.
            {"account_id": "JPM-A", "statement_date": "2026-04-30",
             "asset_class": "equity_etf", "symbol": "SPY",
             "description": "SPY ETF", "quantity": 100,
             "market_value": 74564.0},
            # Stale statement date — must be excluded when as_of=latest.
            {"account_id": "JPM-A", "statement_date": "2026-03-31",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 560 OLD",
             "quantity": 9, "market_value": 9000.0},
        ])
        transactions = pd.DataFrame([
            # JPM buy — strike absent in description (matched on expiry).
            {"account_id": "JPM-A", "settlement_date": "2026-04-10",
             "transaction_type": "buy", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT",
             "quantity": 9, "amount": -13143.60},
            # Fidelity buy — full data in description.
            {"account_id": "FID-A", "settlement_date": "2026-04-14",
             "transaction_type": "buy", "symbol": None,
             "description": "PUT (NVDA) NVIDIA CORPORATION 7654321CD "
                            "You Bought DEC 18 26 $135 (100 SHS) OPENING",
             "quantity": 11, "amount": -3286.80},
            # Unrelated dividend — must be ignored by _is_option_buy.
            {"account_id": "JPM-A", "settlement_date": "2026-04-15",
             "transaction_type": "dividend", "symbol": "SPY",
             "description": "DIVIDEND",
             "quantity": float("nan"), "amount": 12.50},
        ])
        return positions, transactions

    def test_parsing_and_cost_basis(self):
        positions, transactions = self._make_data()
        tbl = build_option_position_table(positions, transactions)

        # 3 option rows on the latest statement date (the equity row and
        # the stale 2026-03-31 row are excluded).
        self.assertEqual(len(tbl), 3)

        jpm = tbl[tbl["account_id"] == "JPM-A"].iloc[0]
        self.assertEqual(jpm["source"], "jpm")
        self.assertEqual(jpm["opt_type"], "put")
        self.assertEqual(jpm["underlying"], "SPY")
        self.assertEqual(jpm["strike"], 560.0)
        self.assertEqual(jpm["expiry"], date(2026, 12, 18))
        # Cost basis = 13143.60 / (9 * 100) = 14.604
        self.assertAlmostEqual(jpm["cost_basis_per_share"], 14.604, places=3)
        self.assertAlmostEqual(jpm["cost_basis_total"], 13143.60, places=2)

        fid = tbl[tbl["account_id"] == "FID-A"].iloc[0]
        self.assertEqual(fid["source"], "fidelity")
        self.assertEqual(fid["strike"], 135.0)
        # Cost basis = 3286.80 / (11 * 100) = 2.988
        self.assertAlmostEqual(fid["cost_basis_per_share"], 2.988, places=3)

        orph = tbl[tbl["account_id"] == "ORPH"].iloc[0]
        self.assertEqual(orph["source"], "unparsed")
        # NaN strike for unparsed rows
        import math
        self.assertTrue(math.isnan(orph["strike"]))

    def test_premium_per_share_mv_back_derives_correctly(self):
        positions, transactions = self._make_data()
        tbl = build_option_position_table(positions, transactions)
        jpm = tbl[tbl["account_id"] == "JPM-A"].iloc[0]
        # 7335.0 / (9 * 100) = 8.15
        self.assertAlmostEqual(jpm["premium_per_share_mv"], 8.15, places=4)

    def test_empty_positions(self):
        tbl = build_option_position_table(pd.DataFrame(columns=[
            "account_id", "statement_date", "asset_class", "symbol",
            "description", "quantity", "market_value",
        ]), pd.DataFrame())
        self.assertTrue(tbl.empty)
        # Schema should still be the standard shape so callers can rely on
        # column names even on an empty result.
        for col in ("opt_type", "underlying", "expiry", "strike",
                    "cost_basis_per_share", "source"):
            self.assertIn(col, tbl.columns)

    def test_as_of_filter(self):
        positions, transactions = self._make_data()
        # Force an earlier as-of date — the stale row becomes the latest.
        tbl = build_option_position_table(
            positions, transactions, as_of="2026-03-31"
        )
        # One statement row (the stale JPM SPY 560) and one synthesized
        # post-statement row (the Fidelity NVDA buy on 2026-04-14 carries
        # full strike/expiry). The JPM buy on 2026-04-10 has no strike
        # in its desc so it's skipped, not synthesized.
        stmt_rows = tbl[tbl["source"] != "post_statement"]
        post_rows = tbl[tbl["source"] == "post_statement"]
        self.assertEqual(len(stmt_rows), 1)
        self.assertEqual(stmt_rows.iloc[0]["account_id"], "JPM-A")
        import math
        self.assertTrue(math.isnan(stmt_rows.iloc[0]["cost_basis_per_share"]))
        self.assertEqual(len(post_rows), 1)
        self.assertEqual(post_rows.iloc[0]["underlying"], "NVDA")

    def test_multiple_buys_pooled(self):
        """Two BUY tranches for the same contract → cost basis aggregates."""
        positions = pd.DataFrame([
            {"account_id": "FID-A", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPDR S&P500 ETF",
             "quantity": 10, "market_value": 8800.0},
        ])
        transactions = pd.DataFrame([
            {"account_id": "FID-A", "settlement_date": "2026-03-15",
             "transaction_type": "buy", "symbol": None,
             "description": "PUT (SPY) SPDR X You Bought DEC 18 26 $560 (100 SHS) OPENING",
             "quantity": 6, "amount": -7200.0},
            {"account_id": "FID-A", "settlement_date": "2026-04-15",
             "transaction_type": "buy", "symbol": None,
             "description": "PUT (SPY) SPDR X You Bought DEC 18 26 $560 (100 SHS) OPENING",
             "quantity": 4, "amount": -5200.0},
        ])
        tbl = build_option_position_table(positions, transactions)
        self.assertEqual(len(tbl), 1)
        # Pooled: ($7200 + $5200) / (10 * 100) = $12.40/sh
        self.assertAlmostEqual(
            tbl.iloc[0]["cost_basis_per_share"], 12.40, places=4
        )
        self.assertAlmostEqual(tbl.iloc[0]["cost_basis_total"], 12400.0, places=2)


class RollCostBasisTests(unittest.TestCase):
    """A roll (sell one strike, open others under the same
    (account, type, underlying, expiry) key) poisons the txn-pool fallback
    two ways: sells are never netted, and JPM strike-less buys match EVERY
    strike's row, double-counting the pool across rows. The broker-stated
    statement ``cost_basis`` is authoritative when present."""

    def _roll_data(self, with_statement_basis: bool):
        pos_rows = [
            {"account_id": "JPM-A", "statement_date": "2026-06-30",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 500 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 6, "market_value": 5000.0},
            {"account_id": "JPM-A", "statement_date": "2026-06-30",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 520 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 4, "market_value": 4500.0},
        ]
        if with_statement_basis:
            pos_rows[0]["cost_basis"] = 6000.0
            pos_rows[1]["cost_basis"] = 5000.0
        transactions = pd.DataFrame([
            # The original lot, later rolled away — strike-less JPM buy desc.
            {"account_id": "JPM-A", "settlement_date": "2026-04-10",
             "transaction_type": "buy", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT",
             "quantity": 12, "amount": -15000.0},
            # The roll's sell — must never feed cost basis.
            {"account_id": "JPM-A", "settlement_date": "2026-06-03",
             "transaction_type": "sell", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 ETF UNSOLICITED CLOSE CONTRACT",
             "quantity": -12, "amount": 14000.0},
            # Replacement lots (strike-less descs — each matches BOTH rows).
            {"account_id": "JPM-A", "settlement_date": "2026-06-03",
             "transaction_type": "buy", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT",
             "quantity": 6, "amount": -6000.0},
            {"account_id": "JPM-A", "settlement_date": "2026-06-03",
             "transaction_type": "buy", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT",
             "quantity": 4, "amount": -5000.0},
        ])
        return pd.DataFrame(pos_rows), transactions

    def test_statement_cost_basis_preferred(self):
        positions, transactions = self._roll_data(with_statement_basis=True)
        tbl = build_option_position_table(positions, transactions)
        self.assertEqual(len(tbl), 2)
        by_strike = {r["strike"]: r for _, r in tbl.iterrows()}
        self.assertAlmostEqual(by_strike[500.0]["cost_basis_total"], 6000.0,
                               places=2)
        self.assertAlmostEqual(by_strike[520.0]["cost_basis_total"], 5000.0,
                               places=2)
        # per-share = total / (qty * 100)
        self.assertAlmostEqual(by_strike[500.0]["cost_basis_per_share"], 10.0,
                               places=4)
        self.assertAlmostEqual(by_strike[520.0]["cost_basis_per_share"], 12.5,
                               places=4)
        agg = option_book_aggregates(tbl, date(2026, 7, 1))
        self.assertAlmostEqual(agg["cost_basis"], 11000.0, places=2)

    def test_txn_pool_fallback_without_statement_basis(self):
        """Without a statement basis the legacy pool engages, with its two
        documented limitations (no sell netting; strike-blind double count).
        Pinned so the fallback's semantics are explicit, not accidental."""
        positions, transactions = self._roll_data(with_statement_basis=False)
        tbl = build_option_position_table(positions, transactions)
        pool = 15000.0 + 6000.0 + 5000.0
        for _, r in tbl.iterrows():
            self.assertAlmostEqual(r["cost_basis_total"], pool, places=2)


class PostStatementOpenTests(unittest.TestCase):
    """Options opened (or closed) AFTER the latest statement date.

    Interim CSV ingest pulls broker activity that lands between monthly
    statements, but until this hook existed the option position table was
    derived purely from statement snapshots — anything traded mid-cycle
    was invisible. These cases cover the round trip.
    """

    def _stmt_positions(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": "JPM-A", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 9, "market_value": 7335.0},
        ])

    def _stmt_buy(self) -> dict:
        return {"account_id": "JPM-A", "settlement_date": "2026-04-10",
                "transaction_type": "buy", "symbol": "SPY",
                "description": "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT",
                "quantity": 9, "amount": -13143.60, "price": 14.60}

    def test_open_after_statement_creates_new_row(self):
        positions = self._stmt_positions()
        txns = pd.DataFrame([
            self._stmt_buy(),
            # NEW: post-statement open — strike present (interim CSV format).
            {"account_id": "JPM-A", "settlement_date": "2026-05-15",
             "transaction_type": "buy", "symbol": "SPY NOV 26 PUT 640.00",
             "description": "PUT SPY 11/20/26 640 STATE STREET SPDR S&P 500 ETF "
                            "UNSOLICITED OPEN CONTRACT",
             "quantity": 4, "amount": -6610.80, "price": 16.53},
        ])
        tbl = build_option_position_table(positions, txns)
        post = tbl[tbl["source"] == "post_statement"]
        self.assertEqual(len(post), 1)
        row = post.iloc[0]
        self.assertEqual(row["underlying"], "SPY")
        self.assertEqual(row["opt_type"], "put")
        self.assertEqual(row["strike"], 640.0)
        self.assertEqual(row["expiry"], date(2026, 11, 20))
        self.assertEqual(row["quantity"], 4.0)
        # Per-share cost = 6610.80 / (4 * 100) = 16.527
        self.assertAlmostEqual(row["cost_basis_per_share"], 16.527, places=3)
        self.assertAlmostEqual(row["cost_basis_total"], 6610.80, places=2)
        self.assertEqual(row["asset_class"], "option_put")
        # Bootstrap MV = qty * 100 * last_buy_price = 4 * 100 * 16.53
        self.assertAlmostEqual(row["market_value"], 6612.0, places=2)

    def test_round_trip_within_window_is_skipped(self):
        # User opens then closes the same contract before next statement —
        # should not appear in the table at all.
        positions = self._stmt_positions()
        txns = pd.DataFrame([
            self._stmt_buy(),
            {"account_id": "JPM-A", "settlement_date": "2026-05-07",
             "transaction_type": "buy", "symbol": "USO MAY 26 PUT 95.00",
             "description": "PUT USO 05/15/26 95 UNITED STATES OIL FUND LP "
                            "UNSOLICITED OPEN CONTRACT",
             "quantity": 300, "amount": -3297.00, "price": 0.11},
            {"account_id": "JPM-A", "settlement_date": "2026-05-12",
             "transaction_type": "sell", "symbol": "USO MAY 26 PUT 95.00",
             "description": "PUT USO 05/15/26 95 UNITED STATES OIL FUND LP "
                            "UNSOLICITED CLOSING CONTRACT",
             "quantity": -300, "amount": 201.0, "price": 0.01},
        ])
        tbl = build_option_position_table(positions, txns)
        self.assertFalse((tbl["underlying"] == "USO").any())
        self.assertEqual(len(tbl[tbl["source"] == "post_statement"]), 0)

    def test_add_to_existing_position_pools_into_one_row(self):
        # Statement shows 12 SPY 575 puts; user buys 4 more after statement.
        # Result: single row with qty=16, pooled cost basis.
        positions = self._stmt_positions()
        txns = pd.DataFrame([
            self._stmt_buy(),
            {"account_id": "JPM-A", "settlement_date": "2026-05-10",
             "transaction_type": "buy", "symbol": "SPY DEC 26 PUT 560.00",
             "description": "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 ETF "
                            "UNSOLICITED OPEN CONTRACT",
             "quantity": 4, "amount": -5200.0, "price": 13.0},
        ])
        tbl = build_option_position_table(positions, txns)
        spy_560 = tbl[
            (tbl["underlying"] == "SPY") & (tbl["strike"] == 560.0)
        ]
        # Should be exactly one row, not two.
        self.assertEqual(len(spy_560), 1)
        row = spy_560.iloc[0]
        self.assertEqual(row["quantity"], 13.0)
        # Pooled cost basis: (13143.60 + 5200.0) / (13 * 100) ≈ 14.1105
        self.assertAlmostEqual(row["cost_basis_per_share"], 14.1105, places=3)
        self.assertAlmostEqual(row["cost_basis_total"], 18343.60, places=2)
        # Source stays "jpm" (the existing-row mutation path doesn't relabel).
        self.assertEqual(row["source"], "jpm")

    def test_jpm_buy_without_strike_is_skipped_gracefully(self):
        # Legacy PDF-parsed JPM BUY without strike (e.g. "PUT SPY 12/18/26
        # ETF OPEN CONTRACT") can't be grouped, so it should be skipped —
        # the existing statement row still renders, no crash.
        positions = self._stmt_positions()
        txns = pd.DataFrame([
            self._stmt_buy(),
            {"account_id": "JPM-A", "settlement_date": "2026-05-10",
             "transaction_type": "buy", "symbol": "AMZN",
             "description": "PUT AMZN 06/19/26 OPEN CONTRACT Exchange Listed",
             "quantity": 5, "amount": -2500.0, "price": 5.0},
        ])
        tbl = build_option_position_table(positions, txns)
        # AMZN unfilled — no row for it.
        self.assertFalse((tbl["underlying"] == "AMZN").any())
        # Original statement row still present.
        self.assertEqual(len(tbl[tbl["underlying"] == "SPY"]), 1)

    def test_synth_rolled_snapshot_trusts_the_rolled_rows(self):
        # app.py / the terminal run synthesize_interim_positions BEFORE this
        # function: it rolls the 2026-04-30 option rows forward to a
        # snapshot_date == max(interim.settlement_date), say 2026-05-22, and
        # books every interim open / close / expiry into that snapshot. The
        # table must take those rows as-is — re-synthesizing the in-window
        # BUYs on top doubled them (Aug 2026: 2 SNDK calls became 4).
        positions = pd.DataFrame([
            # Statement row.
            {"account_id": "JPM-A", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 9, "market_value": 7335.0},
            # synthesize_interim_positions carry-forward to the synth date.
            # Same description, quantity, MV — only statement_date changes.
            {"account_id": "JPM-A", "statement_date": "2026-05-22",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 9, "market_value": 7335.0},
            # The in-window SPY 640 open, booked by the roll at its premium.
            {"account_id": "JPM-A", "statement_date": "2026-05-22",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 11/20/26 640 STATE STREET SPDR S&P 500 ETF "
                            "UNSOLICITED OPEN CONTRACT",
             "quantity": 4, "market_value": 6610.80},
        ])
        txns = pd.DataFrame([
            # Original opening BUY for SPY 575.
            {"account_id": "JPM-A", "settlement_date": "2026-04-10",
             "transaction_type": "buy", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT",
             "quantity": 9, "amount": -13143.60, "price": 14.60},
            # NEW post-statement open, dated BEFORE the synth roll date.
            {"account_id": "JPM-A", "settlement_date": "2026-05-15",
             "transaction_type": "buy", "symbol": "SPY NOV 26 PUT 640.00",
             "description": "PUT SPY 11/20/26 640 STATE STREET SPDR S&P 500 ETF "
                            "UNSOLICITED OPEN CONTRACT",
             "quantity": 4, "amount": -6610.80, "price": 16.53},
            # An unrelated late-settled txn that defines the synth-roll date.
            {"account_id": "JPM-A", "settlement_date": "2026-05-22",
             "transaction_type": "dividend", "symbol": "SPY",
             "description": "DIVIDEND",
             "quantity": float("nan"), "amount": 25.0},
        ])
        tbl = build_option_position_table(positions, txns)
        # The carry-forwarded SPY 560 row appears once (at the synth date).
        spy_560 = tbl[tbl["strike"] == 560.0]
        self.assertEqual(len(spy_560), 1)
        # The SPY 640 opened between the 04-30 statement and the 05-22 synth
        # date is already in the rolled snapshot — it must appear ONCE, at
        # the rolled quantity, and never as a post_statement duplicate.
        spy_640 = tbl[tbl["strike"] == 640.0]
        self.assertEqual(len(spy_640), 1)
        self.assertEqual(spy_640.iloc[0]["source"], "jpm")
        self.assertEqual(spy_640.iloc[0]["quantity"], 4.0)
        self.assertEqual(len(tbl[tbl["source"] == "post_statement"]), 0)
        self.assertEqual(len(tbl), 2)


class DualDateMonthTests(unittest.TestCase):
    """WSF-2: two brokers' option legs in the same month on different dates."""

    def test_dual_date_latest_month_keeps_both_brokers(self) -> None:
        positions = pd.DataFrame([
            # JPM leg on the last business day.
            {"account_id": "JPM-A", "statement_date": "2026-04-29",
             "asset_class": "option_put", "symbol": "SPY",
             "description": "PUT SPY 12/18/26 560 STATE STREET SPDR S&P 500 -- ETF",
             "quantity": 10, "market_value": 9000.0},
            # Fidelity leg on the calendar month-end.
            {"account_id": "FID-A", "statement_date": "2026-04-30",
             "asset_class": "option_put", "symbol": "TSLA",
             "description": "PUT TESLA INC",
             "quantity": 5, "market_value": 4000.0},
        ])
        txns = pd.DataFrame(columns=[
            "account_id", "settlement_date", "transaction_type",
            "symbol", "description", "quantity", "amount",
        ])
        # as_of=None -> max option date = 2026-04-30. Month-resolution must
        # still include the 2026-04-29 JPM leg (exact-match dropped it).
        tbl = build_option_position_table(positions, txns)
        self.assertEqual(set(tbl["account_id"]), {"JPM-A", "FID-A"},
                         "month-resolution must keep the 04-29 JPM leg")
        self.assertIn("SPY", set(tbl["underlying"]))


class OptionBookAggregatesTests(unittest.TestCase):
    """WS-E: aggregate-exposure tiles over the LIVE long-option book.

    The audit found the Options-tab aggregate tiles (a) summed statement
    market_value instead of the live Polygon mid for "Premium at risk" and
    (b) included expired + zero-quantity rows, overstating the loss. These
    pin the corrected behavior of the pure aggregate helper.
    """

    TODAY = date(2026, 6, 17)
    COLUMNS = ["opt_type", "underlying", "expiry", "strike", "quantity",
               "market_value", "cost_basis_total", "premium_mid"]

    @staticmethod
    def _row(opt_type="put", underlying="SPY", expiry=date(2026, 9, 18),
             strike=500.0, quantity=2.0, market_value=2400.0,
             cost_basis_total=3000.0, premium_mid=12.50) -> dict:
        return {
            "opt_type": opt_type, "underlying": underlying, "expiry": expiry,
            "strike": strike, "quantity": quantity,
            "market_value": market_value, "cost_basis_total": cost_basis_total,
            "premium_mid": premium_mid,
        }

    def test_excludes_expired_and_zero_quantity_rows(self):
        # One live put + one expired put + one closed (zero-qty) row.
        tbl = pd.DataFrame([
            self._row(),  # live, future expiry
            self._row(strike=400.0, expiry=date(2026, 5, 1), quantity=1.0,
                      market_value=5000.0, cost_basis_total=6000.0,
                      premium_mid=float("nan")),  # expired
            self._row(strike=450.0, expiry=date(2026, 12, 18), quantity=0.0,
                      market_value=1000.0, cost_basis_total=2000.0,
                      premium_mid=5.0),  # zero-qty (closed, still on stmt)
        ])
        agg = option_book_aggregates(tbl, self.TODAY)
        self.assertEqual(agg["n_live"], 1)
        self.assertEqual(agg["n_excluded"], 2)
        # cost must be ONLY the live row — not 3000+6000+2000.
        self.assertAlmostEqual(agg["cost_basis"], 3000.0)
        # premium-at-risk uses the live mid (2*100*12.50), not statement 2400.
        self.assertAlmostEqual(agg["premium_at_risk"], 2500.0)
        self.assertAlmostEqual(agg["unrealized_pnl"], -500.0)
        self.assertAlmostEqual(agg["notional_protected"], 100000.0)

    def test_premium_at_risk_uses_live_mid_with_statement_fallback(self):
        tbl = pd.DataFrame([
            # Live mid present -> qty*100*mid = 3000 (NOT statement 9999).
            self._row(quantity=3.0, market_value=9999.0, premium_mid=10.0),
            # No live mid -> fall back to statement market_value 800.
            self._row(underlying="QQQ", strike=300.0, quantity=1.0,
                      market_value=800.0, premium_mid=float("nan")),
        ])
        agg = option_book_aggregates(tbl, self.TODAY)
        self.assertEqual(agg["n_live"], 2)
        self.assertAlmostEqual(agg["premium_at_risk"], 3800.0)

    def test_notional_protected_is_puts_only(self):
        tbl = pd.DataFrame([
            self._row(opt_type="put", strike=500.0, quantity=2.0),
            self._row(opt_type="call", strike=600.0, quantity=1.0),
        ])
        agg = option_book_aggregates(tbl, self.TODAY)
        # Puts only: 2*500*100; the call's notional is excluded.
        self.assertAlmostEqual(agg["notional_protected"], 100000.0)
        self.assertEqual(agg["n_live"], 2)  # both still count for premium

    def test_weighted_dte_over_live_book_only(self):
        tbl = pd.DataFrame([
            self._row(),  # live, expiry 2026-09-18 -> dte 93 from 2026-06-17
            self._row(strike=400.0, expiry=date(2026, 5, 1),
                      premium_mid=float("nan")),  # expired, must not drag it
        ])
        agg = option_book_aggregates(tbl, self.TODAY)
        self.assertAlmostEqual(
            agg["weighted_dte"], (date(2026, 9, 18) - self.TODAY).days)

    def test_empty_table_returns_zeroed_aggregates(self):
        tbl = pd.DataFrame(columns=self.COLUMNS)
        agg = option_book_aggregates(tbl, self.TODAY)
        self.assertEqual(agg["n_live"], 0)
        self.assertEqual(agg["n_excluded"], 0)
        self.assertEqual(agg["premium_at_risk"], 0.0)
        self.assertEqual(agg["cost_basis"], 0.0)
        self.assertEqual(agg["unrealized_pnl"], 0.0)
        self.assertEqual(agg["notional_protected"], 0.0)
        self.assertTrue(pd.isna(agg["weighted_dte"]))


class TestGreekDollarColumns(unittest.TestCase):
    def _tbl(self):
        return pd.DataFrame({
            "quantity": [2.0, -1.0], "model_gamma": [0.01, 0.02],
            "model_vega": [0.5, 0.3], "model_theta": [-0.4, -0.2],
            "spot": [100.0, 50.0]})

    def test_columns_and_formulas(self):
        from option_positions import greek_dollar_columns
        out = greek_dollar_columns(self._tbl())
        self.assertAlmostEqual(out["gamma_dollar_per_1pct"].iloc[0],
                               2.0 * 100.0 * 0.01 * (100.0 ** 2) * 0.01)
        self.assertAlmostEqual(out["vega_dollar_per_volpt"].iloc[1],
                               -1.0 * 100.0 * 0.3 * 0.01)
        self.assertAlmostEqual(out["theta_dollar_per_day"].iloc[0],
                               2.0 * 100.0 * -0.4 / 365.0)
        # the seam does NOT set unrealized_pnl
        self.assertNotIn("unrealized_pnl", out.columns)


if __name__ == "__main__":
    unittest.main()
