"""Unit tests for parsers.lot_engine — constructed frames, synthetic values only."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from parsers import lot_engine
# Reconciliation bands treated as clean by the lot-build gate (an amortizing
# instrument may legitimately land in the accretion/wash-consistent bands).
OK_BANDS = ("ok", "accretion_ok", "wash_consistent")
from parsers.lot_engine import (
    EXCEPTION_COLUMNS,
    LONG_TERM_DAYS,
    LOT_COLUMNS,
    OPEN_COLUMNS,
    REALIZATION_COLUMNS,
    RECON_COLUMNS,
    RELIEF_COLUMNS,
    LotLedgerResult,
    average_cost_remaining_basis,
    build_key_resolvers,
    build_lot_ledger,
    build_name_resolver,
    cash_keys_from_positions,
    classify_basis,
    classify_term,
    corporate_split_events,
    days_to_long_term,
    instrument_key,
    is_cash_like,
    lot_rows,
    reconcile_lots,
    relief_check,
    relief_method,
    symbol_fold,
    vsp_hints,
)

TX_COLUMNS = ["settlement_date", "trade_date", "broker", "account_id",
              "transaction_type", "symbol", "cusip", "description",
              "quantity", "price", "amount", "source_file", "flow_scope",
              "pair_id"]

POS_COLUMNS = ["statement_date", "broker", "account_id", "account_type",
               "symbol", "cusip", "description", "asset_class", "quantity",
               "price", "market_value", "cost_basis", "unrealized_gl",
               "est_annual_income", "currency", "source_file"]


def tx(**kw) -> dict:
    """One transactions.csv-shaped row with sane defaults."""
    row = {c: np.nan for c in TX_COLUMNS}
    row.update({"broker": "testbrok", "account_id": "ACC-1",
                "source_file": "synthetic.pdf"})
    row.update(kw)
    return row


def tx_frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=TX_COLUMNS)


RELIEF_TX_COLUMNS = TX_COLUMNS + ["closing_method", "closing_cost"]


def tx5(**kw) -> dict:
    """A transactions row that also carries the slice-5 relief columns."""
    row = tx(**kw)
    row.setdefault("closing_method", np.nan)
    row.setdefault("closing_cost", np.nan)
    return row


def tx5_frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=RELIEF_TX_COLUMNS)


def pos(**kw) -> dict:
    row = {c: np.nan for c in POS_COLUMNS}
    row.update({"broker": "testbrok", "account_id": "ACC-1",
                "account_type": "individual_tod", "asset_class": "equity_stock",
                "currency": "USD", "source_file": "synthetic.pdf"})
    row.update(kw)
    return row


def pos_frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=POS_COLUMNS)


class InstrumentKeyTests(unittest.TestCase):
    def test_symbol_wins(self):
        self.assertEqual(instrument_key("aaa", "123456789", "SOME FUND"),
                         ("AAA", "symbol"))

    def test_cusip_fallback(self):
        self.assertEqual(instrument_key(np.nan, "123456789", "SOME FUND"),
                         ("123456789", "cusip"))

    def test_desc_slug_fallback(self):
        key, source = instrument_key(np.nan, np.nan, "  Some  Fund, Cl A ")
        self.assertEqual(source, "desc")
        self.assertEqual(key, "SOME FUND CL A")

    def test_all_missing_is_empty_key(self):
        key, source = instrument_key(np.nan, np.nan, np.nan)
        self.assertEqual(key, "")
        self.assertEqual(source, "desc")


class CashClassTests(unittest.TestCase):
    def test_cash_keys_from_positions(self):
        p = pos_frame(pos(symbol="CORE1", asset_class="cash"),
                      pos(symbol="AAA", asset_class="equity_stock"))
        self.assertEqual(cash_keys_from_positions(p), {"CORE1"})

    def test_is_cash_like_by_key_and_by_description(self):
        self.assertTrue(is_cash_like("whatever", "CORE1", {"CORE1"}, "symbol"))
        self.assertTrue(is_cash_like("GOVT MONEY MARKET FUND", "XX", set(), "desc"))
        self.assertFalse(is_cash_like("COMMON STOCK", "AAA", set(), "symbol"))
        # description heuristic must NOT fire on symbol-keyed rows
        self.assertFalse(is_cash_like("GOVT MONEY MARKET FUND", "SPAXX",
                                      set(), "symbol"))

    def test_depositary_receipts_are_not_cash_like(self):
        self.assertFalse(is_cash_like("AMERICAN DEPOSITARY SHARES", "XX", set(), "desc"))
        self.assertFalse(is_cash_like("SPONSORED ADR AMERICAN DEPOSITARY RECEIPT", "XX", set(), "desc"))
        self.assertTrue(is_cash_like("CHASE DEPOSIT SWEEP", "XX", set(), "desc"))

    def test_literal_cash_symbol_is_cash_like(self):
        self.assertTrue(is_cash_like("YOU SOLD CASH", "CASH", set(), "symbol"))
        self.assertTrue(is_cash_like("CASH You Sold CASH @ 1.00",
                                     "CASH YOU SOLD CASH 1 00", set(), "desc"))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="CASH",
               quantity=100.0, price=1.0, amount=100.0),
            tx(trade_date="2024-02-10", transaction_type="sell", symbol="CASH",
               quantity=-100.0, price=1.0, amount=100.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)


class FifoCoreTests(unittest.TestCase):
    def test_buy_opens_lot(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0)))
        self.assertIsInstance(res, LotLedgerResult)
        self.assertEqual(list(res.open_lots.columns), OPEN_COLUMNS)
        self.assertEqual(list(res.realizations.columns), REALIZATION_COLUMNS)
        self.assertEqual(list(res.exceptions.columns), EXCEPTION_COLUMNS)
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "AAA")
        self.assertEqual(lot["origin"], "buy")
        self.assertAlmostEqual(lot["quantity_remaining"], 10.0)
        self.assertAlmostEqual(lot["basis_remaining"], 100.0)
        self.assertEqual(str(lot["acquired_date"])[:10], "2024-01-10")

    def test_fifo_close_spans_lots_with_partial_split(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=20.0, amount=200.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="AAA",
               quantity=-15.0, price=30.0, amount=450.0)))
        # first lot fully closed, second half closed
        self.assertEqual(len(res.realizations), 2)
        r0, r1 = res.realizations.iloc[0], res.realizations.iloc[1]
        self.assertAlmostEqual(r0["quantity_closed"], 10.0)
        self.assertAlmostEqual(r0["basis_closed"], 100.0)
        self.assertAlmostEqual(r0["proceeds"], 300.0)   # 10/15 of 450
        self.assertAlmostEqual(r0["realized_gl"], 200.0)
        self.assertAlmostEqual(r1["quantity_closed"], 5.0)
        self.assertAlmostEqual(r1["basis_closed"], 100.0)  # half of lot 2
        self.assertAlmostEqual(r1["proceeds"], 150.0)
        self.assertEqual(len(res.open_lots), 1)
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"], 5.0)
        self.assertAlmostEqual(res.open_lots.iloc[0]["basis_remaining"], 100.0)
        self.assertEqual(len(res.exceptions), 0)

    def test_sell_quantity_sign_is_ignored(self):
        # sells arrive with negative quantity in the data; positive must work too
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="AAA",
               quantity=4.0, price=30.0, amount=120.0)))
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"], 6.0)

    def test_sell_underflow_closes_available_and_logs_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="AAA",
               quantity=-25.0, price=30.0, amount=750.0)))
        self.assertEqual(len(res.realizations), 1)
        self.assertAlmostEqual(res.realizations.iloc[0]["quantity_closed"], 10.0)
        self.assertAlmostEqual(res.realizations.iloc[0]["proceeds"], 300.0)  # 10/25
        self.assertEqual(len(res.open_lots), 0)
        exc = res.exceptions
        self.assertEqual(len(exc), 1)
        self.assertEqual(exc.iloc[0]["reason"], "sell_underflow")
        self.assertAlmostEqual(exc.iloc[0]["quantity"], 15.0)

    def test_sell_with_missing_amount_is_missing_fields_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="AAA")))
        self.assertEqual(len(res.exceptions), 1)
        self.assertEqual(res.exceptions.iloc[0]["reason"], "missing_fields")
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"], 10.0)

    def test_cash_like_rows_are_excluded(self):
        p = pos_frame(pos(symbol="CORE1", asset_class="cash"))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="CORE1",
               quantity=100.0, price=1.0, amount=100.0),
            tx(trade_date="2024-01-11", transaction_type="buy", symbol=np.nan,
               cusip=np.nan, description="GOVT MONEY MARKET FUND",
               quantity=50.0, price=1.0, amount=50.0)),
            opening_positions=p)
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)

    def test_term_classification(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2022-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-01-05", transaction_type="buy", symbol="BBB",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="AAA",
               quantity=-10.0, price=30.0, amount=300.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="BBB",
               quantity=-10.0, price=30.0, amount=300.0)))
        by_key = res.realizations.set_index("instrument_key")
        self.assertEqual(by_key.loc["AAA", "term"], "long")
        self.assertEqual(by_key.loc["BBB", "term"], "short")
        self.assertGreater(by_key.loc["AAA", "holding_days"], LONG_TERM_DAYS)

    def test_schema_breakage_raises(self):
        with self.assertRaises(ValueError):
            build_lot_ledger(pd.DataFrame({"trade_date": []}))

    def test_dividends_do_not_touch_lots(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-01", transaction_type="dividend",
               symbol="AAA", amount=5.0)))
        self.assertEqual(len(res.open_lots), 1)
        self.assertEqual(len(res.realizations), 0)
        self.assertEqual(len(res.exceptions), 0)

    def test_other_type_is_unhandled_event_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="other", symbol="AAA",
               quantity=-10.0, amount=100.0)))
        self.assertEqual(res.exceptions.iloc[0]["reason"], "unhandled_event")

    def test_cash_in_lieu_does_not_touch_lots(self):
        # fractional-share cash (2026-07 parser slice): the fraction was
        # never a lot — the one lot-fragment case closes via its exchange
        # out-leg — so the row is skipped, not an unhandled_event exception
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-01", transaction_type="cash_in_lieu",
               symbol="AAA", amount=5.55)))
        self.assertEqual(len(res.open_lots), 1)
        self.assertEqual(len(res.realizations), 0)
        self.assertEqual(len(res.exceptions), 0)

    def test_leap_year_exact_anniversary_is_short(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2025-01-10", transaction_type="sell", symbol="AAA",
               quantity=-10.0, price=30.0, amount=300.0)))
        row = res.realizations.iloc[0]
        self.assertEqual(row["holding_days"], 366)  # spans 2024-02-29
        self.assertEqual(row["term"], "short")      # exactly one year


class ReinvestAndSplitTests(unittest.TestCase):
    def test_reinvestment_opens_lot_without_price(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="reinvestment",
               symbol="AAA", quantity=2.0, price=np.nan, amount=50.0)))
        self.assertEqual(len(res.open_lots), 1)
        self.assertEqual(res.open_lots.iloc[0]["origin"], "reinvestment")
        self.assertAlmostEqual(res.open_lots.iloc[0]["basis_remaining"], 50.0)

    def test_split_adds_shares_pro_rata_basis_unchanged(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-10", transaction_type="buy", symbol="AAA",
               quantity=30.0, price=10.0, amount=300.0),
            tx(trade_date="2024-03-01", transaction_type="stock_split",
               symbol="AAA", quantity=40.0, amount=0.0)))
        # 40 held + 40 added = 2:1 split applied pro-rata to both lots
        lots = res.open_lots.sort_values("open_date")
        self.assertAlmostEqual(lots.iloc[0]["quantity_remaining"], 20.0)
        self.assertAlmostEqual(lots.iloc[1]["quantity_remaining"], 60.0)
        self.assertAlmostEqual(lots.iloc[0]["basis_remaining"], 100.0)
        self.assertAlmostEqual(lots.iloc[1]["basis_remaining"], 300.0)

    def test_split_after_split_sell_uses_split_quantities(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="stock_split",
               symbol="AAA", quantity=10.0, amount=0.0),
            tx(trade_date="2024-04-01", transaction_type="sell", symbol="AAA",
               quantity=-20.0, price=10.0, amount=200.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)
        self.assertAlmostEqual(res.realizations.iloc[0]["basis_closed"], 100.0)

    def test_split_without_position_is_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-03-01", transaction_type="stock_split",
               symbol="AAA", quantity=10.0, amount=0.0)))
        self.assertEqual(res.exceptions.iloc[0]["reason"],
                         "split_without_position")

    def test_spinoff_receipt_for_a_positions_key_is_not_rescued_to_parent(self):
        # A spinoff child's receipt row prints under the CHILD's own symbol,
        # but the child holds no lots and its description leads with the
        # PARENT's issuer word, so the >=6-char one-token name rescue would
        # re-key it onto the parent and multiply the parent's lots (the
        # 2026-06 HONA/HON case). A printed key that IS a positions
        # instrument is not identifier drift — believe the broker; the row
        # lands as split_without_position like every other spinoff receipt.
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="HLX", quantity=4.0,
                cost_basis=40.0, description="HELICON INTL INC"),
            pos(statement_date="2024-06-30", symbol="HLXA", quantity=2.0,
                cost_basis=np.nan, description="HELICON AEROSPACE INC"))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="HLX",
               description="HELICON INTL INC", quantity=4.0, price=10.0,
               amount=40.0),
            tx(trade_date="2024-07-01", transaction_type="stock_split",
               symbol="HLXA",
               description="HELICON AEROSPACE INC COMMON STOCK SPINOFF ON "
                           "4 SHS HELICON INTL INC REC 06/15/24 PAY 06/29/24",
               quantity=2.0, amount=0.0)),
            opening_positions=p)
        parent = res.open_lots[res.open_lots["instrument_key"] == "HLX"]
        self.assertEqual(len(parent), 1)
        self.assertAlmostEqual(parent.iloc[0]["quantity_remaining"], 4.0)
        flagged = res.exceptions[
            res.exceptions["reason"] == "split_without_position"]
        self.assertEqual(list(flagged["instrument_key"]), ["HLXA"])


class MergerRedemptionTests(unittest.TestCase):
    def test_cash_merger_closes_all_lots(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=20.0, amount=200.0),
            tx(trade_date="2024-06-01", transaction_type="merger",
               symbol="AAA", quantity=-20.0, amount=500.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)
        self.assertEqual(set(res.realizations["close_reason"]), {"merger_cash"})
        self.assertAlmostEqual(res.realizations["proceeds"].sum(), 500.0)
        self.assertAlmostEqual(res.realizations["realized_gl"].sum(), 200.0)

    def test_merger_without_cash_is_unrecognized_shape(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-06-01", transaction_type="merger",
               symbol="AAA", quantity=-10.0, amount=0.0)))
        self.assertEqual(res.exceptions.iloc[0]["reason"],
                         "merger_unrecognized_shape")
        self.assertEqual(len(res.open_lots), 1)  # untouched

    def test_redemption_closes_at_proceeds(self):
        # exactly one calendar year -> short (365 days; 2023 is not a leap year)
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2023-01-10", transaction_type="buy", symbol=np.nan,
               cusip="123456AB1", description="SYNTH CORP NOTE 5%",
               quantity=5000.0, price=0.998, amount=4990.0),
            tx(trade_date="2024-01-10", transaction_type="redemption",
               symbol=np.nan, cusip="123456AB1",
               description="SYNTH CORP NOTE 5%", quantity=-5000.0,
               amount=5000.0)))
        self.assertEqual(len(res.open_lots), 0)
        row = res.realizations.iloc[0]
        self.assertEqual(row["close_reason"], "redemption")
        self.assertAlmostEqual(row["realized_gl"], 10.0)
        self.assertEqual(row["term"], "short")  # exactly 365 days is short

    def test_cash_merger_without_position_is_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-06-01", transaction_type="merger",
               symbol="AAA", quantity=-10.0, amount=500.0)))
        self.assertEqual(res.exceptions.iloc[0]["reason"],
                         "merger_without_position")
        self.assertEqual(len(res.realizations), 0)


class ExchangeTransferTests(unittest.TestCase):
    def test_exchange_carries_basis_and_dates(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2022-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="AAA", quantity=-10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="BBB", quantity=4.0, amount=0.0)))
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "BBB")
        self.assertEqual(lot["origin"], "exchange_in")
        self.assertAlmostEqual(lot["quantity_remaining"], 4.0)
        self.assertAlmostEqual(lot["basis_remaining"], 100.0)
        self.assertEqual(str(lot["acquired_date"])[:10], "2022-01-10")
        out = res.realizations.iloc[0]
        self.assertEqual(out["close_reason"], "exchange_out")
        self.assertAlmostEqual(out["realized_gl"], 0.0)

    def test_unpaired_exchange_is_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="AAA", quantity=-10.0, amount=0.0)))
        self.assertEqual(res.exceptions.iloc[0]["reason"], "exchange_unpaired")
        self.assertEqual(len(res.open_lots), 1)  # untouched

    def test_transfer_moves_lots_between_accounts_via_pair_id(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2022-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="AAA", quantity=-10.0, amount=0.0, pair_id="P1"),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               account_id="ACC-2", symbol="AAA", quantity=10.0, amount=0.0,
               pair_id="P1"),
            tx(trade_date="2024-04-01", transaction_type="sell",
               account_id="ACC-2", symbol="AAA", quantity=-10.0, price=30.0,
               amount=300.0)))
        self.assertEqual(len(res.open_lots), 0)
        sell = res.realizations[res.realizations["close_reason"] == "sell"].iloc[0]
        self.assertEqual(sell["account_id"], "ACC-2")
        self.assertAlmostEqual(sell["basis_closed"], 100.0)
        self.assertEqual(sell["term"], "long")   # acquired 2022 survives the move

    def test_transfer_fallback_match_without_pair_id(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="AAA", quantity=-10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               account_id="ACC-2", symbol="AAA", quantity=10.0, amount=0.0)))
        self.assertEqual(len(res.exceptions), 0)
        self.assertEqual(res.open_lots.iloc[0]["account_id"], "ACC-2")

    def test_cash_transfers_have_no_lot_impact(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               amount=1000.0),
            tx(trade_date="2024-03-05", transaction_type="transfer_out",
               amount=-500.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)

    def test_unmatched_inkind_transfer_is_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="AAA", quantity=-10.0, amount=0.0, pair_id="P9")))
        self.assertEqual(res.exceptions.iloc[0]["reason"], "transfer_unmatched")
        self.assertEqual(len(res.open_lots), 1)  # untouched

    def test_same_day_transfer_in_then_sell_uses_statement_order(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2022-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="AAA", quantity=-10.0, amount=0.0, pair_id="P1"),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               account_id="ACC-2", symbol="AAA", quantity=10.0, amount=0.0,
               pair_id="P1"),
            tx(trade_date="2024-03-01", transaction_type="sell",
               account_id="ACC-2", symbol="AAA", quantity=-10.0, price=30.0,
               amount=300.0)))
        self.assertEqual(len(res.exceptions), 0)
        sell = res.realizations[res.realizations["close_reason"] == "sell"].iloc[0]
        self.assertAlmostEqual(sell["basis_closed"], 100.0)
        self.assertEqual(sell["term"], "long")

    def test_ambiguous_exchange_group_refuses_to_pair(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="BBB",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="AAA", quantity=-10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="BBB", quantity=-10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="CCC", quantity=4.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="DDD", quantity=4.0, amount=0.0)))
        self.assertEqual(len(res.exceptions), 4)
        self.assertEqual(set(res.exceptions["reason"]), {"exchange_unpaired"})
        self.assertEqual(len(res.open_lots), 2)  # AAA and BBB untouched

    def test_ambiguous_transfer_fallback_refuses(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-01-10", transaction_type="buy",
               account_id="ACC-3", symbol="AAA", quantity=10.0, price=20.0,
               amount=200.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="AAA", quantity=-10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               account_id="ACC-3", symbol="AAA", quantity=-10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               account_id="ACC-2", symbol="AAA", quantity=10.0, amount=0.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               account_id="ACC-4", symbol="AAA", quantity=10.0, amount=0.0)))
        self.assertEqual(len(res.exceptions), 4)
        self.assertEqual(set(res.exceptions["reason"]), {"transfer_unmatched"})
        self.assertEqual(len(res.open_lots), 2)  # both source lots untouched

    def test_move_underflow_scales_in_side(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=5.0, price=10.0, amount=50.0),
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="AAA", quantity=-10.0, amount=0.0, pair_id="P1"),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               account_id="ACC-2", symbol="AAA", quantity=10.0, amount=0.0,
               pair_id="P1")))
        self.assertEqual(res.exceptions.iloc[0]["reason"], "sell_underflow")
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["account_id"], "ACC-2")
        self.assertAlmostEqual(lot["quantity_remaining"], 5.0)  # covered half only
        self.assertAlmostEqual(lot["basis_remaining"], 50.0)    # per-share 10 preserved

    def test_degenerate_exchange_leg_is_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-03-01", transaction_type="exchange",
               symbol="AAA", quantity=np.nan, amount=0.0)))
        self.assertEqual(res.exceptions.iloc[0]["reason"], "exchange_unpaired")


class OpeningSynthesisTests(unittest.TestCase):
    def test_full_shortfall_gets_opening_lot(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          quantity=10.0, cost_basis=150.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-06-01", transaction_type="sell", symbol="AAA",
               quantity=-10.0, price=30.0, amount=300.0)),
            opening_positions=p)
        self.assertEqual(len(res.exceptions), 0)
        row = res.realizations.iloc[0]
        self.assertAlmostEqual(row["basis_closed"], 150.0)
        self.assertEqual(row["term"], "unknown")   # acquired_date is NaT
        self.assertTrue(np.isnan(row["holding_days"]))

    def test_partial_shortfall_basis_is_reported_minus_reconstructed(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          quantity=10.0, cost_basis=150.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=4.0, price=10.0, amount=40.0)),
            opening_positions=p)
        lots = res.open_lots.sort_values("origin")  # buy, opening
        self.assertEqual(len(lots), 2)
        opening = lots[lots["origin"] == "opening"].iloc[0]
        self.assertAlmostEqual(opening["quantity_remaining"], 6.0)
        self.assertAlmostEqual(opening["basis_remaining"], 110.0)  # 150-40
        self.assertTrue(pd.isna(opening["acquired_date"]))

    def test_opening_basis_floored_at_zero(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          quantity=10.0, cost_basis=30.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=4.0, price=10.0, amount=40.0)),
            opening_positions=p)
        opening = res.open_lots[res.open_lots["origin"] == "opening"].iloc[0]
        self.assertAlmostEqual(opening["basis_remaining"], 0.0)

    def test_opening_missing_basis_is_exception(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          quantity=10.0, cost_basis=np.nan))
        res = build_lot_ledger(tx_frame(), opening_positions=p)
        self.assertEqual(res.exceptions.iloc[0]["reason"],
                         "opening_basis_missing")
        opening = res.open_lots.iloc[0]
        self.assertTrue(np.isnan(opening["basis_remaining"]))

    def test_only_first_statement_month_synthesizes(self):
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=150.0),
            pos(statement_date="2024-02-29", symbol="AAA", quantity=25.0,
                cost_basis=400.0))
        res = build_lot_ledger(tx_frame(), opening_positions=p)
        self.assertEqual(len(res.open_lots), 1)   # no second-month synthesis
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"], 10.0)

    def test_cash_and_zero_qty_positions_are_skipped(self):
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="CORE1",
                asset_class="cash", quantity=100.0, cost_basis=np.nan),
            pos(statement_date="2024-01-31", symbol="AAA", quantity=0.0,
                cost_basis=0.0))
        res = build_lot_ledger(tx_frame(), opening_positions=p)
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)

    def test_positions_schema_breakage_raises_valueerror(self):
        bad = pd.DataFrame({"statement_date": ["2024-01-31"],
                            "account_id": ["ACC-1"]})
        with self.assertRaises(ValueError):
            build_lot_ledger(tx_frame(), opening_positions=bad)

    def test_same_date_buy_processes_before_opening_synthesis(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          quantity=10.0, cost_basis=150.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-31", transaction_type="buy", symbol="AAA",
               quantity=4.0, price=10.0, amount=40.0)),
            opening_positions=p)
        opening = res.open_lots[res.open_lots["origin"] == "opening"]
        self.assertEqual(len(opening), 1)
        self.assertAlmostEqual(opening.iloc[0]["quantity_remaining"], 6.0)
        self.assertAlmostEqual(opening.iloc[0]["basis_remaining"], 110.0)

    def test_two_symbols_synthesize_in_one_call(self):
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=150.0),
            pos(statement_date="2024-01-31", symbol="BBB", quantity=3.0,
                cost_basis=60.0))
        res = build_lot_ledger(tx_frame(), opening_positions=p)
        self.assertEqual(len(res.open_lots), 2)
        by_key = res.open_lots.set_index("instrument_key")
        self.assertAlmostEqual(by_key.loc["AAA", "basis_remaining"], 150.0)
        self.assertAlmostEqual(by_key.loc["BBB", "basis_remaining"], 60.0)

    def test_per_lot_first_statement_rows_aggregate(self):
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=100.0),
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=120.0),
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=140.0))
        res = build_lot_ledger(tx_frame(), opening_positions=p)
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertAlmostEqual(lot["quantity_remaining"], 30.0)
        self.assertAlmostEqual(lot["basis_remaining"], 360.0)


class ReconcileTests(unittest.TestCase):
    def _ledger(self):
        return build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-01-10", transaction_type="buy", symbol=np.nan,
               cusip=np.nan, description="SYNTH PRIVATE NOTE",
               quantity=5.0, price=10.0, amount=50.0)))

    def test_bands_ok_watch_error_known(self):
        self.assertEqual(classify_basis(100.0, 100.5), "ok")     # <= $1 dust
        self.assertEqual(classify_basis(100.0, 100.0), "ok")
        self.assertEqual(classify_basis(102.0, 100.0), "watch")  # $2, 2% <= error_pct
        self.assertEqual(classify_basis(100.0, 5000.0), "error") # 98%, $4900
        self.assertEqual(classify_basis(100.0, 5000.0, tol_pct=99.0), "known")
        self.assertEqual(classify_basis(0.4, 0.0), "ok")         # <= $1 forgiveness
        self.assertEqual(classify_basis(5000.0, 0.0), "error")   # reported 0

    def test_reconcile_joins_and_flags_both_directions(self):
        res = self._ledger()
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=100.0),
            pos(statement_date="2024-01-31", symbol="BBB", quantity=5.0,
                cost_basis=500.0))
        recon = reconcile_lots(res.open_lots, p)
        self.assertEqual(list(recon.columns), RECON_COLUMNS)
        by_key = recon.set_index("instrument_key")
        self.assertEqual(by_key.loc["AAA", "band"], "ok")
        self.assertEqual(by_key.loc["BBB", "band"], "uncovered")
        self.assertEqual(by_key.loc["SYNTH PRIVATE NOTE", "band"], "unjoinable")

    def test_reconcile_empty_inputs(self):
        res = self._ledger()
        empty_recon = reconcile_lots(res.open_lots.iloc[0:0], pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=1.0,
                cost_basis=10.0)))
        self.assertEqual(list(empty_recon.columns), RECON_COLUMNS)
        self.assertEqual(set(empty_recon["band"]), {"uncovered"})
        none_recon = reconcile_lots(res.open_lots, pos_frame(pos())[0:0])
        self.assertEqual(set(none_recon["band"]), {"unjoinable"})

    def test_allowlist_marks_known(self):
        res = self._ledger()
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          quantity=10.0, cost_basis=200.0))
        recon = reconcile_lots(res.open_lots, p,
                               allowlist={("ACC-1", "AAA"): 60.0})
        self.assertEqual(
            recon.set_index("instrument_key").loc["AAA", "band"], "known")

    def test_nan_basis_lot_bands_as_basis_unknown(self):
        first = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                              quantity=10.0, cost_basis=np.nan))
        res = build_lot_ledger(tx_frame(), opening_positions=first)
        later = pos_frame(pos(statement_date="2024-02-29", symbol="AAA",
                              quantity=10.0, cost_basis=150.0))
        recon = reconcile_lots(res.open_lots, later)
        row = recon.set_index("instrument_key").loc["AAA"]
        self.assertEqual(row["band"], "basis_unknown")
        self.assertTrue(np.isnan(row["reconstructed"]))
        self.assertTrue(np.isnan(row["diff_usd"]))

    def test_multi_row_position_aggregates_before_compare(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0)))
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=6.0,
                cost_basis=60.0),
            pos(statement_date="2024-01-31", symbol="AAA", quantity=4.0,
                cost_basis=40.0))
        recon = reconcile_lots(res.open_lots, p)
        self.assertEqual(len(recon), 1)
        row = recon.iloc[0]
        self.assertEqual(row["band"], "ok")
        self.assertAlmostEqual(row["reported"], 100.0)


class AverageCostDiagnosticTests(unittest.TestCase):
    def test_average_cost_remaining_basis(self):
        rows = tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=30.0, amount=300.0),
            tx(trade_date="2024-03-10", transaction_type="sell", symbol="AAA",
               quantity=-10.0, price=30.0, amount=300.0))
        # pool: 20 sh / $400 -> sell 10 removes $200; FIFO would remove $100
        self.assertAlmostEqual(average_cost_remaining_basis(rows), 200.0)


class QuantityBandTests(unittest.TestCase):
    """Basis cannot be judged where the share count is wrong.

    Slice 3 learned this the hard way: fixing instrument keys turned a
    -75% basis band into +35% because the transaction history was missing
    sells, and a basis percentage on a quantity-mismatched position is not a
    near-miss but a meaningless number. `qty_mismatch` says so instead.
    """

    def _lots(self, qty, basis, key="AAA"):
        return build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol=key,
               quantity=qty, price=basis / qty, amount=-basis))).open_lots

    def test_agreement_bands_ok_and_populates_columns(self):
        recon = reconcile_lots(self._lots(10.0, 100.0), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=100.0)))
        row = recon.iloc[0]
        self.assertEqual(row["band"], "ok")
        self.assertAlmostEqual(row["reconstructed_qty"], 10.0)
        self.assertAlmostEqual(row["reported_qty"], 10.0)
        self.assertAlmostEqual(row["qty_diff"], 0.0)
        for col in ("reconstructed_qty", "reported_qty", "qty_diff"):
            self.assertIn(col, RECON_COLUMNS)

    def test_quantity_mismatch_overrides_ok_basis(self):
        # basis agrees exactly, quantity does not -> the basis agreement is a
        # coincidence and must not be reported as ok
        recon = reconcile_lots(self._lots(10.0, 100.0), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=7.0,
                cost_basis=100.0)))
        row = recon.iloc[0]
        self.assertEqual(row["band"], "qty_mismatch")
        self.assertAlmostEqual(row["qty_diff"], 3.0)

    def test_quantity_mismatch_overrides_error_basis(self):
        recon = reconcile_lots(self._lots(10.0, 100.0), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=2.0,
                cost_basis=9000.0)))
        self.assertEqual(recon.iloc[0]["band"], "qty_mismatch")

    def test_fractional_noise_below_epsilon_stays_ok(self):
        recon = reconcile_lots(self._lots(10.0, 100.0), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0005,
                cost_basis=100.0)))
        self.assertEqual(recon.iloc[0]["band"], "ok")

    def test_bond_face_scale_exact_match_stays_ok(self):
        # face value in the tens of thousands must not manufacture a mismatch
        recon = reconcile_lots(self._lots(36000.0, 36153.29), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=36000.0,
                cost_basis=36153.29)))
        self.assertEqual(recon.iloc[0]["band"], "ok")

    def test_tolerance_is_absolute_at_every_scale(self):
        """Pins the epsilon on both sides and at bond scale.

        The tolerance is deliberately a flat 0.001 shares: a relative term is
        inert (float noise on these sums is ~1e-9 even at face values in the
        tens of thousands) and a wide one would silence exactly the
        whole-share corporate-action gaps this band exists to surface.
        """
        from parsers.lot_engine import _QTY_RECON_EPS, quantity_mismatch
        self.assertAlmostEqual(_QTY_RECON_EPS, 0.001)
        # boundary: exactly at the floor is agreement, just past it is not
        self.assertFalse(quantity_mismatch(10.0, 10.0 + _QTY_RECON_EPS))
        self.assertTrue(quantity_mismatch(10.0, 10.002))
        # a whole share is a mismatch no matter how large the position
        self.assertTrue(quantity_mismatch(1_000_000.0, 999_999.0))
        self.assertTrue(quantity_mismatch(36000.0, 36001.0))
        # and 0.002 shares on a bond-scale position is still a mismatch
        recon = reconcile_lots(self._lots(36000.0, 36153.29), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=36000.002,
                cost_basis=36153.29)))
        self.assertEqual(recon.iloc[0]["band"], "qty_mismatch")

    def test_absent_reported_quantity_is_not_a_mismatch_nor_a_zero(self):
        """An all-NaN reported quantity must stay NaN, never become 0.0.

        Summing it to zero would print a share count the broker never stated
        and band a phantom mismatch — the quantity analogue of the
        basis_unknown honesty rule.
        """
        recon = reconcile_lots(self._lots(10.0, 100.0), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA",
                quantity=np.nan, cost_basis=100.0)))
        row = recon.iloc[0]
        self.assertTrue(np.isnan(row["reported_qty"]))
        self.assertNotEqual(row["band"], "qty_mismatch")

    def test_basis_unknown_still_wins_over_qty_mismatch(self):
        first = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                              quantity=10.0, cost_basis=np.nan))
        res = build_lot_ledger(tx_frame(), opening_positions=first)
        later = pos_frame(pos(statement_date="2024-02-29", symbol="AAA",
                             quantity=99.0, cost_basis=150.0))
        self.assertEqual(reconcile_lots(res.open_lots, later).iloc[0]["band"],
                         "basis_unknown")

    def test_allowlist_does_not_tolerate_quantity_mismatch(self):
        recon = reconcile_lots(self._lots(10.0, 100.0), pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=4.0,
                cost_basis=200.0)),
            allowlist={("ACC-1", "AAA"): 60.0})
        self.assertEqual(recon.iloc[0]["band"], "qty_mismatch")

    def test_one_sided_rows_carry_coherent_quantity_columns(self):
        lots = self._lots(10.0, 100.0)
        unjoined = reconcile_lots(lots, pos_frame(pos())[0:0]).iloc[0]
        self.assertEqual(unjoined["band"], "unjoinable")
        self.assertAlmostEqual(unjoined["reconstructed_qty"], 10.0)
        self.assertTrue(np.isnan(unjoined["reported_qty"]))
        self.assertTrue(np.isnan(unjoined["qty_diff"]))
        uncovered = reconcile_lots(lots.iloc[0:0], pos_frame(
            pos(statement_date="2024-01-31", symbol="BBB", quantity=5.0,
                cost_basis=50.0))).iloc[0]
        self.assertEqual(uncovered["band"], "uncovered")
        self.assertAlmostEqual(uncovered["reported_qty"], 5.0)
        self.assertAlmostEqual(uncovered["reconstructed_qty"], 0.0)


class OptionExclusionTests(unittest.TestCase):
    """Options are out of scope for the lot ledger (parent spec §6) — but
    both brokers key option confirms and option positions by the
    UNDERLYING's symbol, so without an explicit guard they flow into the
    underlying's equity lots (slice-2 spec §4). One predicate, every
    surface: replay, pairing, opening synthesis, reconciliation.
    """

    def test_predicates_match_option_shapes_only(self):
        from parsers.lot_engine import is_option_position, is_option_row
        # JPM confirm + positions shapes, Fidelity confirm/positions shape
        self.assertTrue(is_option_row("CALL ZZZ 01/17/25 OPEN CONTRACT"))
        self.assertTrue(is_option_row("PUT ZZZ 12/18/26 115 ZEBRA CORP ADJ 10:1"))
        self.assertTrue(is_option_row("PUT (ZZZ) ZEBRA CORP DEC 18 26 $115 (100 SHS)"))
        # near-misses that must stay in scope
        self.assertFalse(is_option_row("CALLON PETROLEUM CO COM"))
        self.assertFalse(is_option_row("PUTNAM MUNICIPAL OPPORTUNITIES TR"))
        self.assertFalse(is_option_row("SYNTH NOTES CALL 05/01/26"))
        self.assertFalse(is_option_row(np.nan))
        # positions predicate: asset_class belt OR description
        self.assertTrue(is_option_position(
            {"asset_class": "option_call", "description": "ZEBRA CORP"}))
        self.assertTrue(is_option_position(
            {"asset_class": "other",
             "description": "PUT (ZZZ) ZEBRA CORP DEC 18 26 $115 (100 SHS)"}))
        self.assertFalse(is_option_position(
            {"asset_class": "equity_stock", "description": "ZEBRA CORP"}))

    def test_option_trades_excluded_both_desc_shapes(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=-100.0,
               description="ALPHA CORP COMMON STOCK"),
            tx(trade_date="2024-02-10", transaction_type="buy", symbol="AAA",
               quantity=5.0, price=2.0, amount=-10.0,
               description="CALL AAA 03/15/24 OPEN CONTRACT AVG PRICE SHOWN"),
            tx(trade_date="2024-02-20", transaction_type="sell", symbol="AAA",
               quantity=-5.0, price=3.0, amount=15.0,
               description="PUT (AAA) ALPHA CORP CLOSING TRANSACTION")))
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertAlmostEqual(lot["quantity_remaining"], 10.0)
        self.assertAlmostEqual(lot["basis_remaining"], 100.0)
        self.assertEqual(len(res.realizations), 0)
        self.assertEqual(res.exceptions["reason"].tolist(),
                         ["option_excluded", "option_excluded"])
        self.assertEqual(res.exceptions["instrument_key"].tolist(),
                         ["AAA", "AAA"])

    def test_option_transfer_legs_not_paired(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-03-01", transaction_type="transfer_out",
               symbol="CCC", quantity=-10.0, pair_id="p1",
               description="PUT CCC 03/20/26 GAMMA INC TO 999-88888"),
            tx(trade_date="2024-03-01", transaction_type="transfer_in",
               symbol="CCC", quantity=10.0, pair_id="p1", account_id="ACC-2",
               description="PUT CCC 03/20/26 GAMMA INC FROM 111-22222")))
        self.assertEqual(len(res.open_lots), 0)
        reasons = res.exceptions["reason"].tolist()
        self.assertEqual(reasons, ["option_excluded", "option_excluded"])

    def test_near_miss_rows_still_processed(self):
        # includes a bond whose redemption is a CALL event: the word CALL
        # appears mid-description, the type is redemption — must be
        # processed as a real lot close, never option-excluded
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-05", transaction_type="buy", symbol="CPE",
               quantity=10.0, price=5.0, amount=-50.0,
               description="CALLON PETROLEUM CO COM"),
            tx(trade_date="2024-01-06", transaction_type="buy",
               cusip="746763AD3", quantity=10.0, price=10.0, amount=-100.0,
               description="PUTNAM MUNICIPAL OPPORTUNITIES TR"),
            tx(trade_date="2024-02-01", transaction_type="buy",
               cusip="912828XX5", quantity=10.0, price=100.0, amount=-1000.0,
               description="SYNTH POWER CO 1ST MTG 4.75% DUE 2031"),
            tx(trade_date="2024-03-01", transaction_type="redemption",
               cusip="912828XX5", quantity=-10.0, amount=1000.0,
               description="SYNTH POWER CO 1ST MTG 4.75% CALLED @ 100")))
        self.assertEqual(len(res.open_lots), 2)  # bond fully redeemed
        self.assertEqual(len(res.realizations), 1)
        self.assertEqual(len(res.exceptions), 0)

    def test_opening_synthesis_ignores_option_positions(self):
        p = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA", quantity=10.0,
                cost_basis=100.0, description="ALPHA CORP"),
            pos(statement_date="2024-01-31", symbol="AAA", quantity=1.0,
                cost_basis=500.0, asset_class="option_put",
                description="PUT (AAA) ALPHA CORP DEC 18 26 $100 (100 SHS)"))
        res = build_lot_ledger(tx_frame(), opening_positions=p)
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "AAA")
        self.assertAlmostEqual(lot["quantity_remaining"], 10.0)
        self.assertAlmostEqual(lot["basis_remaining"], 100.0)

    def test_reconcile_drops_option_positions(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=10.0, amount=-100.0,
               description="ALPHA CORP")))
        month = pos_frame(
            pos(statement_date="2024-06-30", symbol="AAA", quantity=10.0,
                cost_basis=100.0, description="ALPHA CORP"),
            pos(statement_date="2024-06-30", symbol="AAA", quantity=1.0,
                cost_basis=500.0, asset_class="option_put",
                description="PUT (AAA) ALPHA CORP DEC 18 26 $100 (100 SHS)"),
            pos(statement_date="2024-06-30", symbol="ZZZ", quantity=2.0,
                cost_basis=300.0, asset_class="option_call",
                description="CALL (ZZZ) ZEBRA CORP JAN 15 27 $50 (100 SHS)"))
        recon = reconcile_lots(res.open_lots, month)
        self.assertEqual(len(recon), 1)
        row = recon.iloc[0]
        self.assertEqual(row["instrument_key"], "AAA")
        self.assertAlmostEqual(row["reported"], 100.0)
        self.assertEqual(row["band"], "ok")


class KeyResolutionTests(unittest.TestCase):
    """Fidelity prints a security's NAME + cusip on confirms but its NAME +
    ticker in holdings, so transactions and positions land in disjoint key
    spaces and nothing joins (slice-3 spec §3). The resolver learns
    name->symbol from positions and canonicalizes transaction keys onto it.
    A symbol the statement actually printed always wins, and anything
    ambiguous refuses to resolve rather than guessing (audit WSF-7).
    """

    FID_DESC = "SYNTH ALPHA TRUST UNIT 11111ZZZ1 You Bought DEPOSITARY RECEIPT"

    def test_cusip_keyed_buy_joins_symbol_keyed_position(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT",
                          quantity=10.0, cost_basis=100.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy",
               cusip="11111ZZZ1", description=self.FID_DESC,
               quantity=10.0, price=10.0, amount=-100.0)),
            opening_positions=p)
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "AAA")
        self.assertEqual(lot["key_source"], "resolved")
        later = pos_frame(pos(statement_date="2024-02-29", symbol="AAA",
                             description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT",
                             quantity=10.0, cost_basis=100.0))
        recon = reconcile_lots(res.open_lots, later)
        self.assertEqual(len(recon), 1)
        self.assertEqual(recon.iloc[0]["band"], "ok")

    def test_printed_symbol_never_overridden(self):
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        p = pos_frame(pos(symbol="BBB", description="SYNTH ALPHA TRUST UNIT"))
        resolver = build_name_resolver(p)
        row = pd.Series(tx(transaction_type="buy", symbol="AAA",
                           description="SYNTH ALPHA TRUST UNIT 11111ZZZ1"))
        self.assertEqual(resolve_instrument_key(row, resolver),
                         ("AAA", "symbol"))

    def test_ambiguous_learned_name_resolves_nothing(self):
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        p = pos_frame(
            pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT"),
            pos(symbol="ZZZ", description="SYNTH ALPHA TRUST UNIT"))
        resolver = build_name_resolver(p)
        row = pd.Series(tx(transaction_type="buy", cusip="11111ZZZ1",
                           description=self.FID_DESC))
        self.assertEqual(resolve_instrument_key(row, resolver),
                         ("11111ZZZ1", "cusip"))

    def test_short_and_nonmatching_names_resolve_nothing(self):
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        p = pos_frame(pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT"),
                      pos(symbol="SH", description="SH"))
        resolver = build_name_resolver(p)
        short = pd.Series(tx(transaction_type="buy", cusip="22222ZZZ2",
                             description="SH 22222ZZZ2 You Bought"))
        self.assertEqual(resolve_instrument_key(short, resolver),
                         ("22222ZZZ2", "cusip"))
        other = pd.Series(tx(transaction_type="buy", cusip="33333ZZZ3",
                             description="WHOLLY UNRELATED CORP 33333ZZZ3"))
        self.assertEqual(resolve_instrument_key(other, resolver),
                         ("33333ZZZ3", "cusip"))

    def test_disagreeing_prefix_candidates_refuse_agreeing_resolve(self):
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        # two learned names, both extending the row's name, different
        # symbols -> refuse
        p = pos_frame(
            pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT CLASS A"),
            pos(symbol="ZZZ", description="SYNTH ALPHA TRUST UNIT CL Z SHS"))
        row = pd.Series(tx(transaction_type="buy", cusip="11111ZZZ1",
                           description="SYNTH ALPHA TRUST 11111ZZZ1"))
        self.assertEqual(resolve_instrument_key(row, build_name_resolver(p)),
                         ("11111ZZZ1", "cusip"))
        # same shape but both learned names map to ONE symbol -> resolve
        p2 = pos_frame(
            pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT CLASS A"),
            pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT SHS"))
        self.assertEqual(resolve_instrument_key(row, build_name_resolver(p2)),
                         ("AAA", "resolved"))

    def test_row_name_extending_a_learned_name_refuses(self):
        """A row carrying EXTRA tokens is a different instrument.

        Real collisions this prevents: a contingent-value right and a
        cash-merger stub, both named "<ISSUER> …" beyond the common stock's
        holdings name, were merging onto the stock's ticker and pulling
        unrelated basis into its lots.
        """
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        p = pos_frame(pos(symbol="HOLX", description="SYNTH HOLDINGS INC"))
        resolver = build_name_resolver(p)
        cvr = pd.Series(tx(transaction_type="buy", cusip="436CVR021",
                           description="SYNTH HOLDINGS INCORPO CVR RIGHTS"))
        self.assertEqual(resolve_instrument_key(cvr, resolver),
                         ("436CVR021", "cusip"))
        stub = pd.Series(tx(transaction_type="buy", cusip="43644ZZZ1",
                            description="SYNTH HOLDINGS INC CSM 1 76P S"))
        self.assertEqual(resolve_instrument_key(stub, resolver),
                         ("43644ZZZ1", "cusip"))
        # the legitimate direction still resolves
        exact = pd.Series(tx(transaction_type="buy", cusip="43644ZZZ2",
                             description="SYNTH HOLDINGS 43644ZZZ2 You Bought"))
        self.assertEqual(resolve_instrument_key(exact, resolver),
                         ("HOLX", "resolved"))

    def test_resolution_applies_to_sells_and_transfers(self):
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT",
                          quantity=10.0, cost_basis=100.0))
        # symbol-keyed buy, then a cusip-keyed sell of the same security
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-02-01", transaction_type="buy", symbol="AAA",
               description="SYNTH ALPHA TRUST UNIT", quantity=10.0,
               price=10.0, amount=-100.0),
            tx(trade_date="2024-03-01", transaction_type="sell",
               cusip="11111ZZZ1", description=self.FID_DESC,
               quantity=-4.0, price=12.0, amount=48.0)),
            opening_positions=p)
        self.assertEqual(len(res.realizations), 1)
        self.assertNotIn("sell_underflow", set(res.exceptions["reason"]))
        # paired in-kind transfer where the two legs print different ids
        res2 = build_lot_ledger(tx_frame(
            tx(trade_date="2024-02-01", transaction_type="buy", symbol="AAA",
               description="SYNTH ALPHA TRUST UNIT", quantity=10.0,
               price=10.0, amount=-100.0),
            tx(trade_date="2024-04-01", transaction_type="transfer_out",
               cusip="11111ZZZ1", description=self.FID_DESC, quantity=-10.0,
               pair_id="p9"),
            tx(trade_date="2024-04-01", transaction_type="transfer_in",
               symbol="AAA", description="SYNTH ALPHA TRUST UNIT",
               quantity=10.0, pair_id="p9", account_id="ACC-2")),
            opening_positions=p)
        moved = res2.open_lots[res2.open_lots["account_id"] == "ACC-2"]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved.iloc[0]["instrument_key"], "AAA")
        self.assertNotIn("transfer_unmatched", set(res2.exceptions["reason"]))

    def test_match_must_be_a_prefix_not_a_substring(self):
        # kills the substring mutant: an unrelated security whose holdings
        # name merely CONTAINS the row's name must not resolve
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        p = pos_frame(pos(symbol="ZZZ",
                          description="GLOBAL SYNTH BETA HOLDINGS PLC"))
        row = pd.Series(tx(transaction_type="buy", cusip="22222ZZZ2",
                           description="SYNTH BETA 22222ZZZ2 You Bought"))
        self.assertEqual(resolve_instrument_key(row, build_name_resolver(p)),
                         ("22222ZZZ2", "cusip"))

    def test_min_name_length_guard_blocks_a_short_prefix(self):
        # kills the mutant that drops the length floor: a 3-char row name IS
        # a prefix of the learned name but is far too weak to identify it
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        p = pos_frame(pos(symbol="AAA", description="SYN ALPHA TRUST UNIT"))
        row = pd.Series(tx(transaction_type="buy", cusip="22222ZZZ2",
                           description="SYN 22222ZZZ2 You Bought"))
        self.assertEqual(resolve_instrument_key(row, build_name_resolver(p)),
                         ("22222ZZZ2", "cusip"))

    def test_cusip_resolver_rescues_rows_whose_desc_omits_the_cusip(self):
        """DRIP and account-transfer rows print no cusip in the description.

        Their name is then the whole row text, which cannot match a holdings
        name, so without cusip propagation they strand under the raw cusip
        while the same instrument's buys resolve — one instrument, two keys,
        and in-kind transfers silently stop moving lots.
        """
        # statement dated after both rows, so the shortfall rule synthesizes
        # nothing and the two transaction lots stand alone
        p = pos_frame(pos(statement_date="2024-03-31", symbol="AAA",
                          description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT",
                          quantity=15.0, cost_basis=150.0))
        res = build_lot_ledger(tx_frame(
            # confirm: carries the cusip in the description -> resolves by name
            tx(trade_date="2024-01-10", transaction_type="buy",
               cusip="11111ZZZ1", description=self.FID_DESC,
               quantity=10.0, price=10.0, amount=-100.0),
            # DRIP: same cusip in the column, no cusip in the description
            tx(trade_date="2024-02-10", transaction_type="reinvestment",
               cusip="11111ZZZ1", quantity=5.0, price=10.0, amount=-50.0,
               description="SYNTH ALPHA TRUST UNIT TRADE DATE 02 10 24 DIV"),
        ), opening_positions=p)
        self.assertEqual(len(res.open_lots), 2)
        self.assertEqual(set(res.open_lots["instrument_key"]), {"AAA"})
        self.assertEqual(set(res.open_lots["key_source"]), {"resolved"})
        recon = reconcile_lots(res.open_lots, p)
        self.assertEqual(len(recon), 1)
        self.assertEqual(recon.iloc[0]["band"], "ok")

    def test_contradicted_cusip_proof_is_dropped(self):
        # an option confirm can carry a neighbouring row's cusip; a cusip with
        # contradictory proofs must teach the resolver nothing
        from parsers.lot_engine import build_cusip_resolver, build_name_resolver
        p = pos_frame(pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT"),
                      pos(symbol="BBB", description="SYNTH BETA CORPORATION"))
        names = build_name_resolver(p)
        frame = tx_frame(
            tx(transaction_type="buy", cusip="11111ZZZ1",
               description="SYNTH ALPHA TRUST UNIT 11111ZZZ1 You Bought"),
            tx(transaction_type="buy", cusip="11111ZZZ1",
               description="SYNTH BETA CORPORATION 11111ZZZ1 You Bought"))
        self.assertEqual(build_cusip_resolver(frame, names), {})
        # ... while an uncontradicted one is learned
        ok_frame = tx_frame(
            tx(transaction_type="buy", cusip="11111ZZZ1",
               description="SYNTH ALPHA TRUST UNIT 11111ZZZ1 You Bought"))
        self.assertEqual(build_cusip_resolver(ok_frame, names),
                         {"11111ZZZ1": "AAA"})

    def test_option_confirms_never_prove_a_cusip(self):
        # build_key_resolvers must exclude option rows from the proving pass:
        # their description names a derivative and their cusip column is
        # occasionally mis-attributed from an adjacent row
        from parsers.lot_engine import build_key_resolvers
        p = pos_frame(pos(symbol="AAA", description="SYNTH ALPHA TRUST UNIT"))
        frame = tx_frame(
            tx(transaction_type="buy", cusip="11111ZZZ1",
               description="CALL (AAA) SYNTH ALPHA TRUST UNIT 11111ZZZ1 "
                           "You Bought JAN 17 25 $50 (100 SHS)"))
        _names, cusips = build_key_resolvers(frame, p)
        self.assertEqual(cusips, {})

    def test_transfer_pairs_by_shape_across_key_spaces(self):
        # _pair_events._shape must use the resolver: with no pair_id the legs
        # pair on (key, date, qty), so a cusip-keyed out-leg and a
        # symbol-keyed in-leg must land on the same key to pair at all
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT",
                          quantity=10.0, cost_basis=100.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-02-01", transaction_type="buy",
               cusip="11111ZZZ1", description=self.FID_DESC, quantity=10.0,
               price=10.0, amount=-100.0),
            tx(trade_date="2024-04-01", transaction_type="transfer_out",
               cusip="11111ZZZ1", description=self.FID_DESC, quantity=-10.0),
            tx(trade_date="2024-04-01", transaction_type="transfer_in",
               symbol="AAA", description="SYNTH ALPHA TRUST UNIT",
               quantity=10.0, account_id="ACC-2"),
        ), opening_positions=p)
        moved = res.open_lots[res.open_lots["account_id"] == "ACC-2"]
        self.assertEqual(len(moved), 1)
        self.assertAlmostEqual(moved.iloc[0]["basis_remaining"], 100.0)
        self.assertEqual(set(res.exceptions["reason"]) - {"option_excluded"},
                         set())

    def test_moved_lot_lands_on_the_resolved_key(self):
        # _move keys the IN-side row itself; with a cusip-keyed in-leg the
        # moved lot must still land under the resolved symbol
        p = pos_frame(pos(statement_date="2024-01-31", symbol="AAA",
                          description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT",
                          quantity=10.0, cost_basis=100.0))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-02-01", transaction_type="buy", symbol="AAA",
               description="SYNTH ALPHA TRUST UNIT", quantity=10.0,
               price=10.0, amount=-100.0),
            tx(trade_date="2024-04-01", transaction_type="transfer_out",
               symbol="AAA", description="SYNTH ALPHA TRUST UNIT",
               quantity=-10.0, pair_id="p7"),
            tx(trade_date="2024-04-01", transaction_type="transfer_in",
               cusip="11111ZZZ1", description=self.FID_DESC, quantity=10.0,
               pair_id="p7", account_id="ACC-2"),
        ), opening_positions=p)
        moved = res.open_lots[res.open_lots["account_id"] == "ACC-2"]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved.iloc[0]["instrument_key"], "AAA")
        self.assertEqual(moved.iloc[0]["key_source"], "resolved")

    def test_excluded_and_unpaired_exceptions_carry_resolved_keys(self):
        p = pos_frame(pos(symbol="AAA",
                          description="SYNTH ALPHA TRUST UNIT DEPOSITARY RECEIPT"),
                      pos(symbol="BBB", description="SYNTH BETA CORPORATION"))
        res = build_lot_ledger(tx_frame(
            # proves 11111ZZZ1 -> AAA by name
            tx(trade_date="2024-01-10", transaction_type="buy",
               cusip="11111ZZZ1", description=self.FID_DESC,
               quantity=10.0, price=10.0, amount=-100.0),
            # option row on the proved cusip -> excluded, but keyed AAA
            tx(trade_date="2024-02-10", transaction_type="buy",
               cusip="11111ZZZ1", quantity=1.0, price=2.0, amount=-200.0,
               description="CALL AAA 03/15/24 OPEN CONTRACT"),
            # unpaired transfer leg, cusip-keyed -> exception keyed AAA
            tx(trade_date="2024-03-10", transaction_type="transfer_out",
               cusip="11111ZZZ1", description=self.FID_DESC, quantity=-3.0,
               pair_id="lonely"),
        ), opening_positions=p)
        keys = set(res.exceptions["instrument_key"])
        self.assertEqual(keys, {"AAA"})
        self.assertEqual(set(res.exceptions["reason"]),
                         {"option_excluded", "transfer_unmatched"})

    def test_resolution_keeps_the_money_market_cash_guard(self):
        # A sweep row that RESOLVES onto a non-cash-classed holding must still
        # be dropped as cash. The description heuristic keys off key_source,
        # so resolution would otherwise re-admit rows the pre-resolver engine
        # dropped. The row name here is a strict prefix of the holdings name,
        # so it does resolve (key_source "resolved").
        p = pos_frame(pos(statement_date="2024-01-31", symbol="MMF",
                          description="SYNTH GOVT MONEY MARKET FUND",
                          asset_class="equity_etf", quantity=1.0,
                          cost_basis=1.0))
        from parsers.lot_engine import build_name_resolver, resolve_instrument_key
        row = pd.Series(tx(transaction_type="buy",
                           description="SYNTH GOVT MONEY MARKET"))
        self.assertEqual(resolve_instrument_key(row, build_name_resolver(p)),
                         ("MMF", "resolved"))
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-02-10", transaction_type="buy",
               description="SYNTH GOVT MONEY MARKET",
               quantity=100.0, price=1.0, amount=-100.0)),
            opening_positions=p)
        # only the opening-synthesis lot from the positions row survives
        self.assertEqual(list(res.open_lots["origin"]), ["opening"])
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"], 1.0)

    def test_no_positions_means_no_resolution(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy",
               cusip="11111ZZZ1", description=self.FID_DESC,
               quantity=10.0, price=10.0, amount=-100.0)))
        self.assertEqual(res.open_lots.iloc[0]["instrument_key"], "11111ZZZ1")
        self.assertEqual(res.open_lots.iloc[0]["key_source"], "cusip")

    def test_name_helpers(self):
        from parsers.lot_engine import (
            normalize_security_name, security_name_from_description,
        )
        self.assertEqual(normalize_security_name("  SPDR  S&P500, ETF-TRUST "),
                         "SPDR S P500 ETF TRUST")
        self.assertEqual(
            security_name_from_description(self.FID_DESC),
            "SYNTH ALPHA TRUST UNIT")
        # no cusip token -> whole description
        self.assertEqual(security_name_from_description("SYNTH ALPHA CORP"),
                         "SYNTH ALPHA CORP")
        self.assertEqual(security_name_from_description(np.nan), "")


class BondBasisTests(unittest.TestCase):
    """A bond trade's amount is principal PLUS accrued interest paid to the
    seller, and accrued interest is not cost basis (it washes against the
    first coupon). Bond quantity is face value and price is per 100 face, so
    principal is qty*price/100 — verified to the penny against every
    cusip-keyed position on the real book (slice-3 spec §4).
    """

    def test_bond_buy_opens_lot_at_principal_not_amount(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-08-19", transaction_type="buy",
               cusip="912828ZZ9", description="SYNTH TREASURY NOTE",
               quantity=36000.0, price=100.4258, amount=-36553.61)))
        lot = res.open_lots.iloc[0]
        self.assertAlmostEqual(lot["basis_remaining"], 36153.29, places=2)
        self.assertAlmostEqual(lot["quantity_remaining"], 36000.0)

    def test_equity_buy_unchanged_including_commission(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-10", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=100.0, amount=-1000.0),
            tx(trade_date="2024-01-11", transaction_type="buy", symbol="BBB",
               quantity=10.0, price=100.0, amount=-1004.95)))
        by = res.open_lots.set_index("instrument_key")["basis_remaining"]
        self.assertAlmostEqual(by.loc["AAA"], 1000.0, places=2)
        self.assertAlmostEqual(by.loc["BBB"], 1004.95, places=2)

    def test_principal_outside_band_or_no_price_falls_back_to_amount(self):
        from parsers.lot_engine import bond_principal_basis
        # no price / zero price
        self.assertAlmostEqual(bond_principal_basis(1000.0, np.nan, -990.0),
                               990.0)
        self.assertAlmostEqual(bond_principal_basis(1000.0, 0.0, -990.0), 990.0)
        # a non-numeric price must not raise (frames built from dicts)
        self.assertAlmostEqual(bond_principal_basis(1000.0, "", -990.0), 990.0)
        self.assertAlmostEqual(bond_principal_basis(1000.0, "n/a", -990.0),
                               990.0)
        # principal more than 10% away from amount -> keep amount
        self.assertAlmostEqual(bond_principal_basis(1000.0, 100.0, -1500.0),
                               1500.0)
        # genuine bond shape -> principal
        self.assertAlmostEqual(bond_principal_basis(36000.0, 100.4258,
                                                   -36553.61), 36153.29,
                               places=2)

    def test_bond_partial_sale_relieves_principal_proportionally(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-08-19", transaction_type="buy",
               cusip="912828ZZ9", description="SYNTH TREASURY NOTE",
               quantity=36000.0, price=100.4258, amount=-36553.61),
            tx(trade_date="2026-01-15", transaction_type="sell",
               cusip="912828ZZ9", description="SYNTH TREASURY NOTE",
               quantity=-18000.0, price=100.0, amount=18150.0)))
        lot = res.open_lots.iloc[0]
        self.assertAlmostEqual(lot["basis_remaining"], 36153.29 / 2, places=2)
        self.assertAlmostEqual(res.realizations.iloc[0]["basis_closed"],
                               36153.29 / 2, places=2)


def _buy(date, qty, amount):
    return tx5(trade_date=date, settlement_date=date, transaction_type="buy",
               symbol="AAA", description="ALPHA CORP", quantity=qty,
               price=amount / qty, amount=amount)


def _sell(date, qty, amount, method=np.nan, cost=np.nan,
          description="ALPHA CORP"):
    return tx5(trade_date=date, settlement_date=date, transaction_type="sell",
               symbol="AAA", description=description, quantity=-qty,
               price=amount / qty, amount=amount, closing_method=method,
               closing_cost=cost)


class ReliefMethodTests(unittest.TestCase):
    def test_recognised_token_is_used(self):
        self.assertEqual(relief_method(pd.Series({"closing_method": "hc"})),
                         "HC")

    def test_unrecognised_token_falls_back_to_fifo(self):
        self.assertEqual(relief_method(pd.Series({"closing_method": "ZZZ"})),
                         "FIFO")

    def test_missing_column_falls_back_to_fifo(self):
        self.assertEqual(relief_method(pd.Series({"symbol": "AAA"})), "FIFO")

    def test_nan_falls_back_to_fifo(self):
        self.assertEqual(relief_method(pd.Series({"closing_method": np.nan})),
                         "FIFO")


class ReliefOrderTests(unittest.TestCase):
    """The broker prints the relief convention per sell; the replay honours
    it instead of assuming FIFO."""

    def test_fifo_is_the_default_and_unchanged(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0)))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 100.0,
                               places=6)

    def test_high_cost_closes_the_expensive_lot_first(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="HC")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 300.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 100.0,
                               places=6)

    def test_low_cost_closes_the_cheap_lot_first(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 300.0),
            _buy("2024-02-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="LC")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 100.0,
                               places=6)

    def test_lifo_closes_the_newest_lot_first(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="LIFO")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 300.0,
                               places=6)

    def test_lthc_prefers_a_long_held_lot_over_a_pricier_recent_one(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2022-01-02", 10, 200.0),      # long term
            _buy("2024-02-02", 10, 900.0),      # short term, pricier
            _sell("2024-03-02", 10, 250.0, method="LTHC")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 200.0,
                               places=6)

    def test_vsp_closes_the_lot_named_in_the_row_text(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="VSP",
                  description="ALPHA CORP UNSOLICITED VS 020224 10 @30.0")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 300.0,
                               places=6)
        self.assertEqual(set(res.exceptions["reason"]), set())

    def test_vsp_without_a_matching_lot_falls_back_and_logs(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="VSP",
                  description="ALPHA CORP VS 090924 10 @99.0")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 100.0,
                               places=6)
        self.assertIn("vsp_lot_unmatched", set(res.exceptions["reason"]))

    def test_multi_lot_hint_chain_closes_every_named_lot(self):
        # the broker names three lots and their quantities; a FIFO replay
        # would take 10 from the first lot instead
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),      # 10/share
            _buy("2024-02-02", 10, 300.0),      # 30/share
            _buy("2024-03-02", 10, 500.0),      # 50/share
            _sell("2024-04-02", 6, 400.0,
                  description="ALPHA CORP VS 010224 1 @10.0 VS 020224 2 @30.0 "
                              "VS 030224 3 @50.0 ROME: X")))
        # 1*10 + 2*30 + 3*50 = 220
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 220.0,
                               places=6)
        self.assertEqual(set(res.realizations["closing_method"]), {"VSP"})
        self.assertEqual(len(res.realizations), 3)

    def test_named_lots_drive_relief_even_with_no_method_token(self):
        # multi-lot specific matches print the lots but leave the column blank
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 5, 150.0,
                  description="ALPHA CORP VS 020224 5 @30.0 ROME: X")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 150.0,
                               places=6)
        self.assertEqual(set(res.realizations["closing_method"]), {"VSP"})

    def test_one_unmatched_hint_logs_and_the_rest_still_apply(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 4, 120.0, method="VSP",
                  description="ALPHA CORP VS 020224 2 @30.0 "
                              "VS 090924 2 @99.0 ROME: X")))
        self.assertIn("vsp_lot_unmatched", set(res.exceptions["reason"]))
        # 2 from the named 30/share lot, 2 spilling to FIFO at 10/share
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 80.0,
                               places=6)

    def test_a_hint_never_names_the_same_lot_twice(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 2, 60.0,
                  description="ALPHA CORP VS 020224 1 @30.0 "
                              "VS 020224 1 @30.0 ROME: X")))
        # both hints name the same acquisition date/price; only one lot exists,
        # so the second finds nothing and is logged rather than double-counted
        self.assertIn("vsp_lot_unmatched", set(res.exceptions["reason"]))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 60.0,
                               places=6)

    def test_malformed_hint_fragment_is_ignored(self):
        # a wrap can strip a hint's quantity; a partial hint identifies no lot
        self.assertEqual(
            [h[1] for h in vsp_hints("ALPHA VS @488.71 VS 020224 3 @30.0")],
            [3.0])

    def test_vsp_price_mismatch_is_not_a_match(self):
        # right date, wrong cost per share -> refuse rather than guess
        res = build_lot_ledger(tx5_frame(
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="VSP",
                  description="ALPHA CORP VS 020224 10 @11.0")))
        self.assertIn("vsp_lot_unmatched", set(res.exceptions["reason"]))

    def test_unsupported_method_falls_back_and_logs(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="PRO")))
        self.assertIn("relief_unsupported", set(res.exceptions["reason"]))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 100.0,
                               places=6)

    def test_specific_share_without_lot_detail_falls_back_and_logs(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="SPEC")))
        self.assertIn("relief_lot_unspecified", set(res.exceptions["reason"]))

    def test_frame_without_the_column_still_replays_fifo(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-01-02", settlement_date="2024-01-02",
               transaction_type="buy", symbol="AAA", description="ALPHA CORP",
               quantity=10.0, price=10.0, amount=100.0),
            tx(trade_date="2024-02-02", settlement_date="2024-02-02",
               transaction_type="buy", symbol="AAA", description="ALPHA CORP",
               quantity=10.0, price=30.0, amount=300.0),
            tx(trade_date="2024-03-02", settlement_date="2024-03-02",
               transaction_type="sell", symbol="AAA",
               description="ALPHA CORP", quantity=-10.0, price=25.0,
               amount=250.0)))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 100.0,
                               places=6)

    def test_realizations_carry_method_and_source_row(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="HC")))
        row = res.realizations.iloc[0]
        self.assertEqual(row["closing_method"], "HC")
        self.assertEqual(row["source_row"], 1)
        self.assertEqual(list(res.realizations.columns), REALIZATION_COLUMNS)

    def test_partial_close_leaves_the_rest_of_the_expensive_lot(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 4, 120.0, method="HC")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 120.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 280.0,
                               places=6)

    def test_high_cost_spills_into_the_next_lot(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 15, 400.0, method="HC")))
        # all of the 300 lot, then half of the 100 lot
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 350.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 50.0,
                               places=6)


class ReliefCheckTests(unittest.TestCase):
    """Per-sell: does the reconstructed relieved basis reproduce the number
    the broker printed on the same row?"""

    def _run(self, frame):
        res = build_lot_ledger(frame)
        return relief_check(res.realizations, frame, res.exceptions)

    def test_exact_match(self):
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=100.0)))
        self.assertEqual(len(chk), 1)
        row = chk.iloc[0]
        self.assertEqual(row["status"], "compared")
        self.assertTrue(row["matched"])
        self.assertAlmostEqual(row["diff"], 0.0, places=6)
        self.assertEqual(list(chk.columns), RELIEF_COLUMNS)

    def test_wrong_convention_shows_as_a_mismatch(self):
        # the broker relieved the expensive lot; a FIFO replay would not
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=300.0)))
        row = chk.iloc[0]
        self.assertFalse(row["matched"])
        self.assertAlmostEqual(row["diff"], -200.0, places=6)

    def test_tolerance_boundary_both_sides(self):
        inside = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=100.05)))
        self.assertTrue(inside.iloc[0]["matched"])
        outside = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=100.06)))
        self.assertFalse(outside.iloc[0]["matched"])

    def test_multi_lot_sell_aggregates_to_one_comparison(self):
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 20, 500.0, method="FIFO", cost=400.0)))
        self.assertEqual(len(chk), 1)
        self.assertTrue(chk.iloc[0]["matched"])
        self.assertAlmostEqual(chk.iloc[0]["reconstructed_basis"], 400.0,
                               places=6)

    def test_sell_without_a_printed_cost_is_not_in_the_frame(self):
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO")))
        self.assertTrue(chk.empty)

    def test_underflowed_sell_is_excluded_and_counted(self):
        chk = self._run(tx5_frame(
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=100.0)))
        self.assertEqual(list(chk["status"]), ["excluded_underflow"])
        self.assertFalse(bool(chk.iloc[0]["matched"]))

    def test_option_sell_is_excluded_and_counted(self):
        chk = self._run(tx5_frame(
            _sell("2024-03-02", 1, 250.0, method="FIFO", cost=100.0,
                  description="CALL AAA 03/15/24 100 ALPHA CORP")))
        self.assertEqual(list(chk["status"]), ["excluded_option"])

    def test_empty_inputs_return_an_empty_frame_with_columns(self):
        out = relief_check(pd.DataFrame(columns=REALIZATION_COLUMNS),
                           pd.DataFrame(), None)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), RELIEF_COLUMNS)

    def test_transactions_without_the_cost_column_return_empty(self):
        frame = tx_frame(
            tx(trade_date="2024-01-02", settlement_date="2024-01-02",
               transaction_type="buy", symbol="AAA", description="ALPHA CORP",
               quantity=10.0, price=10.0, amount=100.0))
        res = build_lot_ledger(frame)
        self.assertTrue(relief_check(res.realizations, frame).empty)

    def test_method_reported_is_the_one_applied(self):
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="HC", cost=300.0)))
        self.assertEqual(chk.iloc[0]["closing_method"], "HC")
        self.assertTrue(chk.iloc[0]["matched"])

    def test_resolved_key_labels_match_the_reconciliation(self):
        # a cusip-keyed row must be labelled by the resolved symbol, or the
        # relief section names an instrument the reconciliation calls
        # something else
        frame = tx5_frame(
            tx5(trade_date="2024-01-02", settlement_date="2024-01-02",
                transaction_type="buy", symbol=np.nan, cusip="123456789",
                description="ALPHA CORP 123456789 You Bought", quantity=10.0,
                price=10.0, amount=100.0),
            tx5(trade_date="2024-03-02", settlement_date="2024-03-02",
                transaction_type="sell", symbol=np.nan, cusip="123456789",
                description="ALPHA CORP 123456789 You Sold", quantity=-10.0,
                price=25.0, amount=250.0, closing_method="FIFO",
                closing_cost=100.0))
        positions = pos_frame(pos(statement_date="2024-03-31", symbol="AAA",
                                  cusip="", description="ALPHA CORP",
                                  quantity=0.0, cost_basis=0.0))
        res = build_lot_ledger(frame, opening_positions=positions)
        names, cusips = build_key_resolvers(frame, positions)
        chk = relief_check(res.realizations, frame, res.exceptions,
                           names, cusips)
        self.assertEqual(chk.iloc[0]["instrument_key"], "AAA")

    def test_fallback_reports_both_the_printed_and_the_executed_method(self):
        # SPEC cannot be executed; the row relieves FIFO. Both facts are
        # reported - a specific-share sell relieved FIFO must never read as a
        # sell the broker itself relieved FIFO.
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="SPEC", cost=100.0)))
        self.assertEqual(chk.iloc[0]["closing_method"], "FIFO")
        self.assertEqual(chk.iloc[0]["printed_method"], "SPEC")

    def test_printed_and_executed_agree_on_an_executable_method(self):
        chk = self._run(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 10, 250.0, method="HC", cost=300.0)))
        self.assertEqual(chk.iloc[0]["printed_method"], "HC")
        self.assertEqual(chk.iloc[0]["closing_method"], "HC")


class ReportedUnknownBandTests(unittest.TestCase):
    """A carry-forward positions month reprices market value but carries no
    cost basis. Summing an all-NaN column to 0.0 asserts the broker reported
    zero cost - the same fabrication the slice-4 quantity fix removed."""

    def _lots(self, quantity=10.0, basis=100.0):
        return pd.DataFrame([{
            "account_id": "ACC-1", "instrument_key": "AAA",
            "key_source": "symbol", "symbol": "AAA",
            "open_date": pd.Timestamp("2024-01-02"),
            "acquired_date": pd.Timestamp("2024-01-02"), "origin": "buy",
            "quantity_open": quantity, "quantity_remaining": quantity,
            "basis_open": basis, "basis_remaining": basis, "source_row": 0}],
            columns=OPEN_COLUMNS)

    def test_all_nan_reported_basis_bands_reported_unknown(self):
        out = reconcile_lots(self._lots(), pos_frame(pos(
            statement_date="2024-06-30", symbol="AAA", description="ALPHA",
            quantity=10.0, cost_basis=float("nan"))))
        row = out.iloc[0]
        self.assertEqual(row["band"], "reported_unknown")
        self.assertTrue(pd.isna(row["reported"]))
        self.assertTrue(pd.isna(row["diff_pct"]))

    def test_present_basis_still_bands_normally(self):
        out = reconcile_lots(self._lots(), pos_frame(pos(
            statement_date="2024-06-30", symbol="AAA", description="ALPHA",
            quantity=10.0, cost_basis=100.0)))
        self.assertEqual(out.iloc[0]["band"], "ok")

    def test_partially_nan_group_still_sums_the_present_rows(self):
        out = reconcile_lots(self._lots(), pos_frame(
            pos(statement_date="2024-06-30", symbol="AAA",
                description="ALPHA", quantity=6.0, cost_basis=60.0),
            pos(statement_date="2024-06-30", symbol="AAA",
                description="ALPHA", quantity=4.0, cost_basis=float("nan"))))
        row = out.iloc[0]
        self.assertEqual(row["reported"], 60.0)
        self.assertNotEqual(row["band"], "reported_unknown")

    def test_quantity_mismatch_outranks_a_missing_reported_basis(self):
        # a carry-forward row still reports QUANTITY, so that comparison
        # stays valid and must not hide behind the missing basis
        out = reconcile_lots(self._lots(), pos_frame(pos(
            statement_date="2024-06-30", symbol="AAA", description="ALPHA",
            quantity=7.0, cost_basis=float("nan"))))
        self.assertEqual(out.iloc[0]["band"], "qty_mismatch")


class CloseLotsRestructureTests(unittest.TestCase):
    """decide-then-apply must not move a single number."""

    def test_plan_then_spill_is_unchanged(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 15, 400.0, method="VSP",
                  description="ALPHA CORP VS 020224 10 @30.0 ROME: X")))
        # the named lot (300) in full, then 5 shares of the 100 lot at 10/sh
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 350.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 50.0,
                               places=6)

    def test_two_takes_from_one_lot_split_its_basis_pro_rata(self):
        # the plan names 4 shares of a lot; the ordering pass then needs 3
        # more from the SAME lot - together they must remove exactly 7/10ths
        res = build_lot_ledger(tx5_frame(
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 7, 200.0, method="VSP",
                  description="ALPHA CORP VS 020224 4 @30.0 ROME: X")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 210.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 90.0,
                               places=6)


class PrintedPoolReliefTests(unittest.TestCase):
    """A synthesized opening lot pools an unknown set of real lots at their
    average cost, so FIFO inside it IS average cost. Where the broker printed
    what it relieved, draw that from the pool instead of the average."""

    def _positions(self, remaining_qty, remaining_basis):
        return pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA",
                description="ALPHA CORP", quantity=100.0, cost_basis=4000.0),
            pos(statement_date="2024-06-30", symbol="AAA",
                description="ALPHA CORP", quantity=remaining_qty,
                cost_basis=remaining_basis))

    def _pool_book(self, sell_qty, printed, extra_buy=None):
        rows = []
        if extra_buy is not None:
            rows.append(extra_buy)
        rows.append(_sell("2024-06-10", sell_qty, sell_qty * 50.0,
                          method="FIFO", cost=printed))
        return build_lot_ledger(
            tx5_frame(*rows),
            opening_positions=self._positions(100.0 - sell_qty, 3700.0))

    def test_pool_only_sell_relieves_exactly_the_printed_cost(self):
        # the pool averages 40/share; the broker relieved 30/share
        res = self._pool_book(10, 300.0)
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 300.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"printed_pool"})
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 3700.0,
                               places=6)

    def test_no_printed_cost_still_relieves_the_pool_average(self):
        res = build_lot_ledger(
            tx5_frame(_sell("2024-06-10", 10, 500.0, method="FIFO")),
            opening_positions=self._positions(90.0, 3600.0))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 400.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"reconstructed"})

    def test_mixed_pool_and_known_lot_takes_printed_minus_known(self):
        # the pool holds only 6 shares (at 40/sh) and sorts first in FIFO, so
        # a sell of 10 drains it and then takes 4 from a known buy of 4 @ 25.
        # The known lot's 100 is exact; the pool supplies printed - 100.
        positions = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA",
                description="ALPHA CORP", quantity=6.0, cost_basis=240.0),
            pos(statement_date="2024-06-30", symbol="AAA",
                description="ALPHA CORP", quantity=0.0, cost_basis=0.0))
        res = build_lot_ledger(
            tx5_frame(_buy("2024-05-02", 4, 100.0),
                      _sell("2024-06-10", 10, 500.0, method="FIFO",
                            cost=300.0)),
            opening_positions=positions)
        by_source = res.realizations.groupby("basis_source")[
            "basis_closed"].sum().to_dict()
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 300.0,
                               places=6)
        self.assertAlmostEqual(by_source["printed_pool"], 200.0, places=6)
        self.assertAlmostEqual(by_source["reconstructed"], 100.0, places=6)

    def test_printed_cost_larger_than_the_pool_clamps_and_logs(self):
        res = self._pool_book(10, 99_000.0)
        self.assertIn("printed_basis_exceeds_lots",
                      set(res.exceptions["reason"]))
        self.assertLessEqual(res.realizations["basis_closed"].sum(), 4000.0)
        self.assertGreaterEqual(
            float(res.open_lots["basis_remaining"].sum()), 0.0)

    def test_quantity_accounting_is_untouched_by_the_basis_override(self):
        res = self._pool_book(10, 300.0)
        self.assertAlmostEqual(res.realizations["quantity_closed"].sum(), 10.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["quantity_remaining"].sum(), 90.0,
                               places=6)

    def test_known_lot_sells_are_never_overridden(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=999.0)))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 100.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"reconstructed"})


class UnknowableLotReliefTests(unittest.TestCase):
    """Fidelity's specific-share flag means the broker relieved a lot it does
    not name ("refer to confirm for Lot detail"). The ledger cannot execute
    that, so where the broker printed what it relieved, that figure is the
    only non-guessing basis to draw. The lot stays unknown either way."""

    def _two_lots_then_spec(self, cost=np.nan, qty=10):
        return build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _buy("2024-02-02", 10, 2000.0),
            _sell("2024-03-02", qty, 5000.0, method="SPEC", cost=cost)))

    def test_spec_sell_relieves_exactly_the_printed_cost(self):
        res = self._two_lots_then_spec(cost=1800.0)
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 1800.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"printed_unknowable"})

    def test_spec_without_a_printed_cost_relieves_fifo_unchanged(self):
        res = self._two_lots_then_spec()
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 1000.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"reconstructed"})

    def test_shares_closed_are_unaffected_by_the_basis_override(self):
        res = self._two_lots_then_spec(cost=1800.0)
        self.assertAlmostEqual(res.realizations["quantity_closed"].sum(), 10.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["quantity_remaining"].sum(), 10.0,
                               places=6)

    def test_printed_unknowable_is_distinct_from_printed_pool(self):
        pool = build_lot_ledger(
            tx5_frame(_sell("2024-06-10", 10, 500.0, method="FIFO",
                            cost=300.0)),
            opening_positions=pos_frame(
                pos(statement_date="2024-01-31", symbol="AAA",
                    description="ALPHA CORP", quantity=100.0,
                    cost_basis=4000.0),
                pos(statement_date="2024-06-30", symbol="AAA",
                    description="ALPHA CORP", quantity=90.0,
                    cost_basis=3700.0)))
        self.assertEqual(set(pool.realizations["basis_source"]),
                         {"printed_pool"})

    def test_printed_cost_spills_beyond_the_closing_shares(self):
        # specific-share identification exists precisely to close something
        # other than the oldest shares, so the printed figure routinely
        # exceeds what the FIFO-selected lot holds. Shares still follow FIFO;
        # the basis comes from the instrument.
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _buy("2024-02-02", 10, 2000.0),
            _sell("2024-03-02", 5, 3000.0, method="SPEC", cost=1500.0)))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 1500.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 1500.0,
                               places=6)
        self.assertAlmostEqual(res.open_lots["quantity_remaining"].sum(), 15.0,
                               places=6)

    def test_no_lot_is_left_holding_negative_basis(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _buy("2024-02-02", 10, 2000.0),
            _sell("2024-03-02", 3, 3000.0, method="SPEC", cost=2900.0)))
        self.assertTrue((res.open_lots["basis_remaining"] >= -1e-9).all())
        self.assertAlmostEqual(res.open_lots["basis_remaining"].sum(), 100.0,
                               places=6)

    def test_printed_cost_exceeding_the_lots_clamps_and_logs(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=99_000.0)))
        self.assertIn("printed_basis_exceeds_lots",
                      set(res.exceptions["reason"]))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 1000.0,
                               places=6)

    def test_lot_still_logs_that_no_lot_was_identified(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=900.0)))
        self.assertIn("relief_lot_unspecified", set(res.exceptions["reason"]))

    def test_exhausting_a_lot_with_basis_left_over_is_logged_not_dropped(self):
        # printed relief below the lots' reconstructed basis leaves basis
        # attached to zero shares; the lot then leaves the queue and carries
        # that residue out of the ledger. It is a finding about the lots.
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=600.0)))
        self.assertIn("printed_relief_basis_residue",
                      set(res.exceptions["reason"]))

    def test_a_named_lot_outranks_the_unknowable_path(self):
        # a VSP hint names the lot, so relief IS executable and the printed
        # cost must not override what the broker itself identified
        res = build_lot_ledger(tx5_frame(
            _buy("2024-02-02", 10, 300.0),
            _sell("2024-03-02", 4, 200.0, method="SPEC", cost=999.0,
                  description="ALPHA CORP VS 020224 4 @30.0 ROME: X")))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 120.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"reconstructed"})

    def test_an_unknown_basis_lot_blocks_the_instrument_level_spread(self):
        # the opening lot's basis is unknown, so the instrument has no total
        # to re-spread; turning that one NaN into every lot's basis would
        # destroy the reconstruction the `basis_unknown` band exists to name
        res = build_lot_ledger(
            tx5_frame(_buy("2024-02-02", 10, 1000.0),
                      _sell("2024-06-10", 5, 500.0, method="SPEC",
                            cost=300.0)),
            opening_positions=pos_frame(
                pos(statement_date="2024-01-31", symbol="AAA",
                    description="ALPHA CORP", quantity=10.0,
                    cost_basis=float("nan")),
                pos(statement_date="2024-06-30", symbol="AAA",
                    description="ALPHA CORP", quantity=15.0,
                    cost_basis=800.0)))
        # relief falls through to the pool path and is labelled as such: the
        # request was "unknowable", but that is not where the basis came from
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"printed_pool"})
        # the known lot keeps its real basis rather than becoming NaN
        known = res.open_lots[res.open_lots["origin"] == "buy"]
        self.assertFalse(bool(known["basis_remaining"].isna().any()))
        self.assertAlmostEqual(float(known["basis_remaining"].iloc[0]),
                               1000.0, places=6)

    def test_other_unsupported_methods_do_not_take_printed_relief(self):
        # MLMG/PRO are executable in principle, just not implemented — taking
        # the printed figure there would assert knowledge we do not have
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="MLMG", cost=600.0)))
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(), 1000.0,
                               places=6)
        self.assertEqual(set(res.realizations["basis_source"]),
                         {"reconstructed"})


class BasisEvidenceTests(unittest.TestCase):
    """Which side of the two anchors an instrument falls on: `reconstructed`
    only when nothing about its basis came from a printed figure."""

    def test_unaided_instrument_is_reconstructed(self):
        res = build_lot_ledger(tx5_frame(_buy("2024-01-02", 10, 1000.0)))
        self.assertEqual(set(res.open_lots["basis_evidence"]),
                         {"reconstructed"})

    def test_printed_relief_marks_the_instrument(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _buy("2024-02-02", 10, 2000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=1800.0)))
        self.assertEqual(set(res.open_lots["basis_evidence"]), {"printed"})

    def test_evidence_survives_the_relieved_lot_being_closed_out(self):
        # the printed-relieved lot is fully consumed and leaves the queue; a
        # later buy opens a fresh, wholly reconstructed lot. The instrument is
        # still printed-touched, and reading only the surviving lots would
        # leak printed evidence into the strict anchor.
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=900.0),
            _buy("2024-04-02", 5, 500.0)))
        self.assertEqual(set(res.open_lots["basis_evidence"]), {"printed"})

    def test_pool_relief_also_counts_as_printed_evidence(self):
        res = build_lot_ledger(
            tx5_frame(_sell("2024-06-10", 10, 500.0, method="FIFO",
                            cost=300.0)),
            opening_positions=pos_frame(
                pos(statement_date="2024-01-31", symbol="AAA",
                    description="ALPHA CORP", quantity=100.0,
                    cost_basis=4000.0),
                pos(statement_date="2024-06-30", symbol="AAA",
                    description="ALPHA CORP", quantity=90.0,
                    cost_basis=3700.0)))
        self.assertEqual(set(res.open_lots["basis_evidence"]), {"printed"})

    def test_another_instrument_is_untouched_by_the_mark(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=900.0),
            tx5(trade_date="2024-04-02", settlement_date="2024-04-02",
                transaction_type="buy", symbol="BBB", description="BETA CORP",
                quantity=4.0, price=25.0, amount=100.0)))
        by_key = dict(zip(res.open_lots["instrument_key"],
                          res.open_lots["basis_evidence"]))
        self.assertEqual(by_key["BBB"], "reconstructed")

    def test_recon_marks_a_partially_touched_instrument_printed(self):
        lots = pd.DataFrame(
            [{"account_id": "ACC-1", "instrument_key": "AAA",
              "basis_remaining": 100.0, "quantity_remaining": 5.0,
              "basis_evidence": "reconstructed"},
             {"account_id": "ACC-1", "instrument_key": "AAA",
              "basis_remaining": 100.0, "quantity_remaining": 5.0,
              "basis_evidence": "printed"}])
        month = pos_frame(pos(statement_date="2024-06-30", symbol="AAA",
                              description="ALPHA CORP", quantity=10.0,
                              cost_basis=200.0))
        recon = reconcile_lots(lots, month)
        self.assertEqual(recon.iloc[0]["basis_evidence"], "printed")

    def test_recon_column_is_present_and_defaults_to_reconstructed(self):
        self.assertIn("basis_evidence", RECON_COLUMNS)
        self.assertIn("basis_evidence", OPEN_COLUMNS)
        lots = pd.DataFrame(
            [{"account_id": "ACC-1", "instrument_key": "AAA",
              "basis_remaining": 200.0, "quantity_remaining": 10.0}])
        month = pos_frame(pos(statement_date="2024-06-30", symbol="AAA",
                              description="ALPHA CORP", quantity=10.0,
                              cost_basis=200.0))
        self.assertEqual(reconcile_lots(lots, month).iloc[0]["basis_evidence"],
                         "reconstructed")

    def test_unjoinable_and_uncovered_rows_carry_the_column(self):
        lots = pd.DataFrame(
            [{"account_id": "ACC-1", "instrument_key": "AAA",
              "basis_remaining": 200.0, "quantity_remaining": 10.0,
              "basis_evidence": "printed"}])
        month = pos_frame(pos(statement_date="2024-06-30", symbol="BBB",
                              description="BETA CORP", quantity=4.0,
                              cost_basis=100.0))
        recon = reconcile_lots(lots, month)
        by_band = dict(zip(recon["band"], recon["basis_evidence"]))
        self.assertEqual(by_band["unjoinable"], "printed")
        self.assertEqual(by_band["uncovered"], "reconstructed")


class LotRowsTests(unittest.TestCase):
    """The lots.csv frame: every open lot, each labelled with the band and
    evidence of the instrument it belongs to."""

    def _book(self):
        return build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _buy("2024-02-02", 5, 700.0)))

    def _month(self, quantity=15.0, cost_basis=1700.0):
        return pos_frame(pos(statement_date="2024-06-30", symbol="AAA",
                             description="ALPHA CORP", quantity=quantity,
                             cost_basis=cost_basis))

    def test_columns_are_the_open_lot_columns_plus_band(self):
        res = self._book()
        rows = lot_rows(res.open_lots,
                        reconcile_lots(res.open_lots, self._month()))
        self.assertEqual(list(rows.columns), LOT_COLUMNS)
        self.assertEqual(LOT_COLUMNS, OPEN_COLUMNS + ["band"])

    def test_one_row_per_open_lot(self):
        res = self._book()
        rows = lot_rows(res.open_lots,
                        reconcile_lots(res.open_lots, self._month()))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows["basis_remaining"].sum(), 1700.0, places=6)

    def test_every_lot_of_an_instrument_carries_its_band(self):
        res = self._book()
        # reported basis disagrees wildly -> the pair bands `error`
        rows = lot_rows(res.open_lots, reconcile_lots(
            res.open_lots, self._month(cost_basis=100.0)))
        self.assertEqual(set(rows["band"]), {"error"})

    def test_an_unjoinable_pair_carries_the_unjoinable_band(self):
        res = self._book()
        month = pos_frame(pos(statement_date="2024-06-30", symbol="BBB",
                              description="BETA CORP", quantity=4.0,
                              cost_basis=100.0))
        rows = lot_rows(res.open_lots, reconcile_lots(res.open_lots, month))
        self.assertEqual(set(rows["band"]), {"unjoinable"})

    def test_printed_evidence_rides_through_from_the_lot(self):
        res = build_lot_ledger(tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _buy("2024-02-02", 10, 2000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=1800.0)))
        rows = lot_rows(res.open_lots, reconcile_lots(
            res.open_lots, self._month(quantity=10.0, cost_basis=1200.0)))
        self.assertEqual(set(rows["basis_evidence"]), {"printed"})

    def test_a_lot_with_no_reconciliation_row_raises(self):
        # every open lot's pair is in recon by construction; a lot without one
        # means the caller paired frames from different runs, and writing a
        # blank band would ship an unlabelled row
        res = self._book()
        recon = reconcile_lots(res.open_lots, self._month())
        with self.assertRaises(ValueError):
            lot_rows(res.open_lots, recon.iloc[0:0])

    def test_empty_lots_give_an_empty_frame_with_the_columns(self):
        rows = lot_rows(pd.DataFrame(columns=OPEN_COLUMNS),
                        pd.DataFrame(columns=RECON_COLUMNS))
        self.assertTrue(rows.empty)
        self.assertEqual(list(rows.columns), LOT_COLUMNS)


class UnknowableReliefCheckTests(unittest.TestCase):
    """A lot-unknowable sell takes the printed figure by construction, so
    comparing it against that same figure scores the check's own input —
    the reason slice 6 excluded pool-relieved sells, carried across."""

    def test_unknowable_sell_is_reported_but_not_scored(self):
        frame = tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=900.0))
        res = build_lot_ledger(frame)
        chk = relief_check(res.realizations, frame, res.exceptions)
        row = chk.iloc[0]
        self.assertEqual(row["status"], "printed_unknowable")
        self.assertFalse(bool(row["matched"]))
        self.assertEqual(int((chk["status"] == "compared").sum()), 0)

    def test_an_ordinary_sell_is_still_scored(self):
        frame = tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="FIFO", cost=1000.0))
        res = build_lot_ledger(frame)
        chk = relief_check(res.realizations, frame, res.exceptions)
        self.assertEqual(chk.iloc[0]["status"], "compared")
        self.assertTrue(bool(chk.iloc[0]["matched"]))

    def test_the_printed_method_is_still_reported_as_spec(self):
        frame = tx5_frame(
            _buy("2024-01-02", 10, 1000.0),
            _sell("2024-03-02", 10, 5000.0, method="SPEC", cost=900.0))
        res = build_lot_ledger(frame)
        chk = relief_check(res.realizations, frame, res.exceptions)
        self.assertEqual(chk.iloc[0]["printed_method"], "SPEC")


class PrintedPoolCheckTests(unittest.TestCase):
    """A pool-relieved sell takes the printed figure by construction, so
    scoring it would be the check marking its own homework."""

    def test_pool_relieved_sell_is_reported_but_not_scored(self):
        positions = pos_frame(
            pos(statement_date="2024-01-31", symbol="AAA",
                description="ALPHA CORP", quantity=100.0, cost_basis=4000.0),
            pos(statement_date="2024-06-30", symbol="AAA",
                description="ALPHA CORP", quantity=90.0, cost_basis=3700.0))
        frame = tx5_frame(_sell("2024-06-10", 10, 500.0, method="FIFO",
                                cost=300.0))
        res = build_lot_ledger(frame, opening_positions=positions)
        chk = relief_check(res.realizations, frame, res.exceptions)
        row = chk.iloc[0]
        self.assertEqual(row["status"], "printed_pool")
        self.assertFalse(bool(row["matched"]))
        self.assertEqual(int((chk["status"] == "compared").sum()), 0)

    def test_source_row_is_present_and_joins_back(self):
        frame = tx5_frame(
            _buy("2024-01-02", 10, 100.0),
            _sell("2024-03-02", 10, 250.0, method="FIFO", cost=100.0))
        res = build_lot_ledger(frame)
        chk = relief_check(res.realizations, frame, res.exceptions)
        self.assertIn("source_row", chk.columns)
        idx = int(chk.iloc[0]["source_row"])
        self.assertEqual(frame.loc[idx, "transaction_type"], "sell")
        self.assertEqual(chk.iloc[0]["status"], "compared")


class TestClassifyTerm(unittest.TestCase):
    """The ledger's one term rule: long strictly AFTER the calendar
    anniversary (IRS 'more than one year'). A >365-day rule misfires on
    anniversary day whenever the span crosses a leap day."""

    def test_anniversary_day_is_short(self):
        self.assertEqual(classify_term(pd.Timestamp("2023-06-15"),
                                       pd.Timestamp("2024-06-15")), "short")

    def test_day_after_anniversary_is_long(self):
        self.assertEqual(classify_term(pd.Timestamp("2023-06-15"),
                                       pd.Timestamp("2024-06-16")), "long")

    def test_leap_span_day_366_is_still_the_anniversary(self):
        span = (pd.Timestamp("2024-06-15") - pd.Timestamp("2023-06-15")).days
        self.assertEqual(span, 366)
        self.assertEqual(classify_term(pd.Timestamp("2023-06-15"),
                                       pd.Timestamp("2024-06-15")), "short")

    def test_missing_acquired_date_is_unknown(self):
        self.assertEqual(classify_term(pd.NaT,
                                       pd.Timestamp("2024-01-01")), "unknown")
        self.assertEqual(classify_term(None,
                                       pd.Timestamp("2024-01-01")), "unknown")


class SymbolFoldTests(unittest.TestCase):
    """Identity fold built from TICKER_HISTORY + CORPORATE_ACTIONS."""

    def test_fold_from_ticker_history(self):
        fold = symbol_fold({"NEWCO": [{"prior_symbol": "OLDCO",
                                       "effective_date": "2026-01-14"}]})
        self.assertEqual(fold, {"OLDCO": "NEWCO"})

    def test_fold_follows_chains_to_terminal(self):
        fold = symbol_fold({
            "MID": [{"prior_symbol": "FIRST", "effective_date": "2020-01-01"}],
            "LAST": [{"prior_symbol": "MID", "effective_date": "2024-01-01"}]})
        self.assertEqual(fold["FIRST"], "LAST")
        self.assertEqual(fold["MID"], "LAST")

    def test_fold_cycle_is_a_loud_config_error(self):
        with self.assertRaises(ValueError):
            symbol_fold({
                "AAA": [{"prior_symbol": "BBB", "effective_date": "2020-01-01"}],
                "BBB": [{"prior_symbol": "AAA", "effective_date": "2021-01-01"}]})

    def test_corporate_action_cusips_fold_to_ticker(self):
        fold = symbol_fold(None, {"AAA": [
            {"kind": "split", "effective_date": "2024-10-01", "ratio": 10.0,
             "cusips": ["999999AB9"]}]})
        self.assertEqual(fold, {"999999AB9": "AAA"})

    def test_empty_inputs_fold_nothing(self):
        self.assertEqual(symbol_fold(None, None), {})
        self.assertEqual(symbol_fold({}, {}), {})

    def test_instrument_key_applies_fold(self):
        fold = {"OLDCO": "NEWCO"}
        self.assertEqual(instrument_key("OLDCO", np.nan, "X", fold=fold),
                         ("NEWCO", "symbol"))
        self.assertEqual(instrument_key("OTHER", np.nan, "X", fold=fold),
                         ("OTHER", "symbol"))


class RenameFoldReplayTests(unittest.TestCase):
    def test_fold_closes_renamed_symbol_lot(self):
        fold = {"OLDCO": "NEWCO"}
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-10-20", transaction_type="buy",
               symbol="OLDCO", quantity=10.0, price=25.0, amount=-250.0),
            tx(trade_date="2025-11-11", transaction_type="sell",
               symbol="NEWCO", quantity=-10.0, price=31.0, amount=310.0)),
            fold=fold)
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)
        row = res.realizations.iloc[0]
        self.assertEqual(row["instrument_key"], "NEWCO")
        self.assertAlmostEqual(row["basis_closed"], 250.0)

    def test_no_fold_is_todays_behavior(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-10-20", transaction_type="buy",
               symbol="OLDCO", quantity=10.0, price=25.0, amount=-250.0),
            tx(trade_date="2025-11-11", transaction_type="sell",
               symbol="NEWCO", quantity=-10.0, price=31.0, amount=310.0)))
        self.assertEqual(len(res.open_lots), 1)
        self.assertEqual(res.exceptions.iloc[0]["reason"], "sell_underflow")

    def test_folded_lot_carries_canonical_symbol(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-04-01", transaction_type="buy",
               symbol="OLDCO", quantity=4.0, price=50.0, amount=-200.0)),
            fold={"OLDCO": "NEWCO"})
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "NEWCO")
        self.assertEqual(lot["symbol"], "NEWCO")

    def test_unfolded_lot_symbol_is_untouched(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-04-01", transaction_type="buy",
               symbol="aaa", quantity=4.0, price=50.0, amount=-200.0)))
        self.assertEqual(res.open_lots.iloc[0]["symbol"], "aaa")

    def test_fold_applies_to_positions_side_of_reconcile(self):
        lots = build_lot_ledger(tx_frame(
            tx(trade_date="2026-04-01", transaction_type="buy",
               symbol="OLDCO", quantity=4.0, price=50.0, amount=-200.0)),
            fold={"OLDCO": "NEWCO"}).open_lots
        month = pos_frame(pos(statement_date="2026-06-30", symbol="OLDCO",
                              description="OLD NAME CORP", quantity=4.0,
                              cost_basis=200.0))
        recon = reconcile_lots(lots, month, fold={"OLDCO": "NEWCO"})
        self.assertEqual(len(recon), 1)
        self.assertEqual(recon.iloc[0]["band"], "ok")
        self.assertEqual(recon.iloc[0]["instrument_key"], "NEWCO")

    def test_fold_repairs_resolver_rename_ambiguity(self):
        # the same security name maps to OLD in early months and NEW later;
        # unfolded that is two symbols for one name -> dropped; folded it is
        # one symbol and resolves
        frame = pos_frame(
            pos(statement_date="2025-10-31", symbol="OLDCO",
                description="FOLDING CORP COMMON STOCK", quantity=10.0),
            pos(statement_date="2025-12-31", symbol="NEWCO",
                description="FOLDING CORP COMMON STOCK", quantity=9.0))
        self.assertEqual(build_name_resolver(frame), {})
        folded = build_name_resolver(frame, fold={"OLDCO": "NEWCO"})
        self.assertEqual(folded, {"FOLDING CORP COMMON STOCK": "NEWCO"})


class CorporateSplitConfigTests(unittest.TestCase):
    def test_split_events_parse_and_fold(self):
        events = corporate_split_events(
            {"OLDCO": [{"kind": "split", "effective_date": "2024-10-01",
                        "ratio": 10.0, "cusips": ["999999AB9"]}]},
            fold={"OLDCO": "NEWCO"})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["key"], "NEWCO")
        self.assertEqual(events[0]["ratio"], 10.0)
        self.assertEqual(events[0]["date"], pd.Timestamp("2024-10-01"))

    def test_bad_ratio_or_kind_raises(self):
        with self.assertRaises(ValueError):
            corporate_split_events({"AAA": [
                {"kind": "split", "effective_date": "2024-10-01",
                 "ratio": 0.0}]})
        with self.assertRaises(ValueError):
            corporate_split_events({"AAA": [
                {"kind": "reverse_merger", "effective_date": "2024-10-01",
                 "ratio": 2.0}]})

    def test_config_split_multiplies_open_quantities(self):
        splits = corporate_split_events({"AAA": [
            {"kind": "split", "effective_date": "2024-10-01", "ratio": 10.0}]})
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-08-29", transaction_type="buy", symbol="AAA",
               quantity=12.345, price=400.0, amount=-4938.0),
            tx(trade_date="2024-10-31", transaction_type="sell", symbol="AAA",
               quantity=-123.45, price=45.0, amount=5555.25)),
            splits=splits)
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)
        self.assertAlmostEqual(res.realizations["basis_closed"].sum(),
                               4938.0, places=2)

    def test_config_split_applies_before_same_day_sell(self):
        splits = corporate_split_events({"AAA": [
            {"kind": "split", "effective_date": "2024-10-01", "ratio": 2.0}]})
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-08-29", transaction_type="buy", symbol="AAA",
               quantity=10.0, price=100.0, amount=-1000.0),
            tx(trade_date="2024-10-01", transaction_type="sell", symbol="AAA",
               quantity=-20.0, price=50.0, amount=1000.0)),
            splits=splits)
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)

    def test_cusip_alias_reaches_the_same_lots(self):
        # post-action rows key by a new cusip; the alias folds them home
        fold = symbol_fold(None, {"AAA": [
            {"kind": "split", "effective_date": "2024-10-01", "ratio": 10.0,
             "cusips": ["999999AB9"]}]})
        splits = corporate_split_events({"AAA": [
            {"kind": "split", "effective_date": "2024-10-01", "ratio": 10.0,
             "cusips": ["999999AB9"]}]}, fold=fold)
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2024-08-29", transaction_type="buy", symbol="AAA",
               quantity=12.345, price=400.0, amount=-4938.0),
            tx(trade_date="2024-10-31", transaction_type="sell",
               symbol=np.nan, cusip="999999AB9",
               description="ALIASED CORP COM NEW 999999AB9 You Sold",
               quantity=-123.45, price=45.0, amount=5555.25)),
            fold=fold, splits=splits)
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)


class CorporateActionKeyRescueTests(unittest.TestCase):
    """Merger/redemption/split/exchange rows whose key drifted from the lots."""

    def test_merger_cusip_row_rescued_by_name(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-08-20", transaction_type="buy", symbol="KKK",
               quantity=3.0, price=70.0, amount=-210.0),
            tx(trade_date="2025-12-11", transaction_type="merger",
               symbol=np.nan, cusip="999888772",
               description="KAPPAFOODS K163682 CMR $85P/S",
               quantity=-3.0, amount=255.0)),
            opening_positions=pos_frame(
                pos(statement_date="2025-08-31", symbol="KKK",
                    description="KAPPAFOODS COMMON STOCK EST YIELD:",
                    quantity=3.0, cost_basis=210.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertNotIn("merger_without_position",
                         set(res.exceptions["reason"]))
        row = res.realizations.iloc[-1]
        self.assertEqual(row["close_reason"], "merger_cash")
        self.assertEqual(row["instrument_key"], "KKK")
        self.assertAlmostEqual(row["proceeds"], 255.0)

    def test_rescue_single_leading_token_needs_six_chars(self):
        # HONEYCOMB INTL INC vs HONEYCOMB INTERNATIONAL INC: one shared
        # leading token, >=6 chars, unique holder -> rescued
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-08-20", transaction_type="buy", symbol="HHH",
               quantity=4.0, price=100.0, amount=-400.0),
            tx(trade_date="2026-06-29", transaction_type="merger",
               symbol=np.nan, cusip="999888773",
               description="HONEYCOMB INTL INC CMR $120P/S",
               quantity=-4.0, amount=480.0)),
            opening_positions=pos_frame(
                pos(statement_date="2025-08-31", symbol="HHH",
                    description="HONEYCOMB INTERNATIONAL INC",
                    quantity=4.0, cost_basis=400.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(res.realizations.iloc[-1]["close_reason"],
                         "merger_cash")

    def test_rescue_refuses_ambiguous_name(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-08-20", transaction_type="buy", symbol="GG1",
               quantity=3.0, price=10.0, amount=-30.0),
            tx(trade_date="2025-08-20", transaction_type="buy", symbol="GG2",
               quantity=3.0, price=10.0, amount=-30.0),
            tx(trade_date="2025-12-11", transaction_type="merger",
               symbol=np.nan, cusip="999888775",
               description="GAMMATRON K163682 CMR $10P/S",
               quantity=-3.0, amount=30.0)),
            opening_positions=pos_frame(
                pos(statement_date="2025-08-31", symbol="GG1",
                    description="GAMMATRON CORP", quantity=3.0,
                    cost_basis=30.0),
                pos(statement_date="2025-08-31", symbol="GG2",
                    description="GAMMATRON GROUP", quantity=3.0,
                    cost_basis=30.0)))
        self.assertIn("merger_without_position", set(res.exceptions["reason"]))
        self.assertEqual(len(res.open_lots), 2)

    def test_rescue_requires_quantity_within_holdings(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-08-20", transaction_type="buy", symbol="KKK",
               quantity=3.0, price=70.0, amount=-210.0),
            tx(trade_date="2025-12-11", transaction_type="merger",
               symbol=np.nan, cusip="999888772",
               description="KAPPAFOODS K163682 CMR $85P/S",
               quantity=-30.0, amount=2550.0)),
            opening_positions=pos_frame(
                pos(statement_date="2025-08-31", symbol="KKK",
                    description="KAPPAFOODS COMMON STOCK", quantity=3.0,
                    cost_basis=210.0)))
        self.assertIn("merger_without_position", set(res.exceptions["reason"]))
        self.assertEqual(len(res.open_lots), 1)

    def test_exchange_out_leg_rescued_by_cusip_holding(self):
        # the successor's SYMBOL prints on the out leg (IPG->OMC shape); the
        # cusip column still names the instrument whose lots exist
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-11-11", transaction_type="buy", symbol=np.nan,
               cusip="999888771", description="INTERPUB GROUP OF COS",
               quantity=9.0, price=25.0, amount=-225.0),
            tx(trade_date="2025-12-01", transaction_type="exchange",
               symbol="OMX", cusip="999888771",
               description="INTERPUB GROUP OF COS @0.344 OMNI GROUP INC",
               quantity=-9.0, amount=0.0),
            tx(trade_date="2025-12-01", transaction_type="exchange",
               symbol="OMX", cusip=np.nan, description="OMNI GROUP INC @0.344",
               quantity=3.0, amount=0.0)))
        self.assertEqual(len(res.exceptions), 0)
        self.assertEqual(len(res.open_lots), 1)
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "OMX")
        self.assertAlmostEqual(lot["quantity_remaining"], 3.0)
        self.assertAlmostEqual(lot["basis_remaining"], 225.0)
        self.assertEqual(str(lot["acquired_date"])[:10], "2025-11-11")

    def test_exchange_out_leg_rescued_by_name(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-10-20", transaction_type="buy", symbol="CTE",
               quantity=10.0, price=23.0, amount=-230.0),
            tx(trade_date="2026-05-08", transaction_type="exchange",
               symbol=np.nan, cusip="999888774",
               description="COTENERGY INC D005253 SMR @0.7",
               quantity=-10.0, amount=0.0),
            tx(trade_date="2026-05-08", transaction_type="exchange",
               symbol="DVX", cusip=np.nan,
               description="DEVOTED ENERGY CORPORATION SMR @0.7",
               quantity=7.0, amount=0.0)),
            opening_positions=pos_frame(
                pos(statement_date="2025-10-31", symbol="CTE",
                    description="COTENERGY INC COMMON STOCK",
                    quantity=10.0, cost_basis=230.0)))
        self.assertEqual(len(res.exceptions), 0)
        lot = res.open_lots.iloc[0]
        self.assertEqual(lot["instrument_key"], "DVX")
        self.assertAlmostEqual(lot["quantity_remaining"], 7.0)
        self.assertAlmostEqual(lot["basis_remaining"], 230.0)

    def test_split_veto_rekeys_mislabeled_symbol_row(self):
        # the CVNA-split-printed-as-DVN shape: the symbol column names an
        # instrument the description contradicts; the description names
        # exactly one other lot-holding instrument
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-01-10", transaction_type="buy", symbol="DVX",
               quantity=5.0, price=10.0, amount=-50.0),
            tx(trade_date="2026-01-10", transaction_type="buy", symbol="CVX1",
               quantity=3.0, price=100.0, amount=-300.0),
            tx(trade_date="2026-05-08", transaction_type="stock_split",
               symbol="DVX", quantity=6.0, amount=0.0,
               description="CARVATRON CO SPLIT ON 1 SHS REC 05/06/26")),
            opening_positions=pos_frame(
                pos(statement_date="2026-01-31", symbol="DVX",
                    description="DEVOTED ENERGY CORPORATION", quantity=5.0,
                    cost_basis=50.0),
                pos(statement_date="2026-01-31", symbol="CVX1",
                    description="CARVATRON CO CLASS A COMMON STOCK",
                    quantity=3.0, cost_basis=300.0)))
        self.assertEqual(len(res.exceptions), 0)
        by_key = {r["instrument_key"]: r for _, r in res.open_lots.iterrows()}
        self.assertAlmostEqual(by_key["DVX"]["quantity_remaining"], 5.0)
        self.assertAlmostEqual(by_key["CVX1"]["quantity_remaining"], 9.0)

    def test_veto_without_unique_match_skips_and_logs(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-01-10", transaction_type="buy", symbol="DVX",
               quantity=5.0, price=10.0, amount=-50.0),
            tx(trade_date="2026-05-08", transaction_type="stock_split",
               symbol="DVX", quantity=6.0, amount=0.0,
               description="NOBODYCO SPLIT ON 1 SHS REC 05/06/26")),
            opening_positions=pos_frame(
                pos(statement_date="2026-01-31", symbol="DVX",
                    description="DEVOTED ENERGY CORPORATION", quantity=5.0,
                    cost_basis=50.0)))
        self.assertIn("corporate_action_key_mismatch",
                      set(res.exceptions["reason"]))
        self.assertAlmostEqual(
            res.open_lots.iloc[0]["quantity_remaining"], 5.0)


class RedemptionMaturityRescueTests(unittest.TestCase):
    BILL_A = ("UNITED STATES TREASURY 02/17/2026 MATURITY DATE 08/19/25 "
              "DATED DATE UNSOLICITED")
    BILL_B = ("UNITED STATES TREASURY 10/15/2025 MATURITY DATE 04/16/25 "
              "DATED DATE UNSOLICITED")

    def _bills(self):
        return [
            tx(trade_date="2025-08-19", transaction_type="buy", symbol=np.nan,
               cusip="912797AA1", description=self.BILL_A, quantity=15000.0,
               price=98.5, amount=-14775.0),
            tx(trade_date="2025-08-19", transaction_type="buy", symbol=np.nan,
               cusip="912797BB2", description=self.BILL_B, quantity=15000.0,
               price=99.0, amount=-14850.0)]

    def test_cusipless_redemption_closes_by_maturity(self):
        res = build_lot_ledger(tx_frame(
            *self._bills(),
            tx(trade_date="2025-10-15", transaction_type="redemption",
               symbol=np.nan, cusip=np.nan,
               description="UNITED STATES TREASURY REDEMPTION",
               quantity=-15000.0, amount=15000.0),
            tx(trade_date="2026-02-17", transaction_type="redemption",
               symbol=np.nan, cusip=np.nan,
               description="UNITED STATES TREASURY REDEMPTION",
               quantity=-15000.0, amount=15000.0)))
        self.assertEqual(len(res.open_lots), 0)
        self.assertEqual(len(res.exceptions), 0)
        reasons = set(res.realizations["close_reason"])
        self.assertEqual(reasons, {"redemption"})
        keys = set(res.realizations["instrument_key"])
        self.assertEqual(keys, {"912797AA1", "912797BB2"})

    def test_ambiguous_maturity_refuses(self):
        rows = self._bills()
        # both bills now mature the same day the redemption lands
        rows[1] = dict(rows[1], description=self.BILL_A, cusip="912797BB2")
        res = build_lot_ledger(tx_frame(
            *rows,
            tx(trade_date="2026-02-17", transaction_type="redemption",
               symbol=np.nan, cusip=np.nan,
               description="UNITED STATES TREASURY REDEMPTION",
               quantity=-15000.0, amount=15000.0)))
        self.assertIn("redemption_unmatched", set(res.exceptions["reason"]))
        self.assertEqual(len(res.open_lots), 2)

    def test_maturity_match_requires_face_agreement(self):
        res = build_lot_ledger(tx_frame(
            self._bills()[0],
            tx(trade_date="2026-02-17", transaction_type="redemption",
               symbol=np.nan, cusip=np.nan,
               description="UNITED STATES TREASURY REDEMPTION",
               quantity=-9000.0, amount=9000.0)))
        self.assertIn("redemption_unmatched", set(res.exceptions["reason"]))
        self.assertEqual(len(res.open_lots), 1)


class PrincipalPaymentTests(unittest.TestCase):
    def test_principal_reduces_basis_pro_rata_by_share(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-12-26", transaction_type="buy", symbol="VVV",
               quantity=10.0, price=10.0, amount=-100.0),
            tx(trade_date="2026-01-05", transaction_type="buy", symbol="VVV",
               quantity=10.0, price=30.0, amount=-300.0),
            tx(trade_date="2026-04-27", transaction_type="principal_pmt",
               symbol="VVV", quantity=np.nan, amount=40.0,
               description="VVV CORP 04/27 RT 2.000 PRINCIPAL")))
        self.assertEqual(len(res.exceptions), 0)
        lots = res.open_lots.sort_values("open_date")
        self.assertAlmostEqual(lots.iloc[0]["basis_remaining"], 80.0)
        self.assertAlmostEqual(lots.iloc[1]["basis_remaining"], 280.0)
        self.assertAlmostEqual(lots.iloc[0]["quantity_remaining"], 10.0)
        self.assertEqual(len(res.realizations), 0)

    def test_principal_exceeding_basis_clamps_and_logs(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-12-26", transaction_type="buy", symbol="VVV",
               quantity=10.0, price=10.0, amount=-100.0),
            tx(trade_date="2026-04-27", transaction_type="principal_pmt",
               symbol="VVV", quantity=np.nan, amount=150.0,
               description="VVV CORP PRINCIPAL")))
        self.assertIn("principal_exceeds_basis", set(res.exceptions["reason"]))
        self.assertAlmostEqual(res.open_lots.iloc[0]["basis_remaining"], 0.0)

    def test_principal_without_position_logs(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-04-27", transaction_type="principal_pmt",
               symbol="VVV", quantity=np.nan, amount=150.0,
               description="VVV CORP PRINCIPAL")))
        self.assertIn("principal_without_position",
                      set(res.exceptions["reason"]))


class LoneExchangeLegTests(unittest.TestCase):
    def test_lone_subshare_out_leg_closes_unrealized(self):
        # a cash-in-lieu fractional removal: no matching in-leg; closes
        # without inventing a realized loss from the printed 0.00
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-05-08", transaction_type="buy", symbol="DVX",
               quantity=7.0, price=40.0, amount=-280.0),
            tx(trade_date="2026-05-11", transaction_type="exchange",
               symbol="DVX", quantity=-0.1, amount=0.0,
               description="DEVOTED ENERGY CORPORATION FROM C014585")))
        self.assertEqual(len(res.exceptions), 0)
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"],
                               6.9)
        row = res.realizations.iloc[-1]
        self.assertEqual(row["close_reason"], "exchange_out")
        self.assertAlmostEqual(row["basis_closed"], 280.0 * 0.1 / 7.0,
                               places=4)
        self.assertTrue(pd.isna(row["proceeds"]))
        self.assertAlmostEqual(row["realized_gl"], 0.0)

    def test_lone_whole_share_out_leg_still_strands(self):
        # a whole-share lone leg is a lost pairing, not a CIL — basis must
        # not be quietly destroyed (test_unpaired_exchange_is_exception
        # pins the same shape at full-position size)
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-05-08", transaction_type="buy", symbol="DVX",
               quantity=7.0, price=40.0, amount=-280.0),
            tx(trade_date="2026-05-11", transaction_type="exchange",
               symbol="DVX", quantity=-7.0, amount=0.0,
               description="DEVOTED ENERGY CORPORATION")))
        self.assertIn("exchange_unpaired", set(res.exceptions["reason"]))
        self.assertAlmostEqual(res.open_lots.iloc[0]["quantity_remaining"],
                               7.0)

    def test_lone_in_leg_is_still_an_exception(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2026-04-08", transaction_type="exchange",
               symbol=np.nan, cusip="999CVR021",
               description="CONTRA CVR UNIT", quantity=8.0, amount=0.0)))
        self.assertIn("exchange_unpaired", set(res.exceptions["reason"]))
        self.assertEqual(len(res.open_lots), 0)


class TestSchemaAdditions(unittest.TestCase):
    def test_open_lots_carry_maturity_column(self):
        # a synthetic bond buy whose description carries the maturity text
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-01-10", transaction_type="buy",
               cusip="000000AA1", quantity=10_000, price=94.0,
               amount=-9_400.00,
               description="SYNTH TREASURY NOTE 06/30/2026 MATURITY")))
        self.assertIn("maturity", res.open_lots.columns)
        got = pd.to_datetime(res.open_lots["maturity"]).iloc[0]
        self.assertEqual(got, pd.Timestamp("2026-06-30"))

    def test_realizations_carry_open_source_row(self):
        res = build_lot_ledger(tx_frame(
            tx(trade_date="2025-01-10", transaction_type="buy",
               symbol="SYNA", quantity=10, price=50.0, amount=-500.00),
            tx(trade_date="2025-02-10", transaction_type="sell",
               symbol="SYNA", quantity=-4, price=40.0, amount=160.00)))
        self.assertIn("open_source_row", res.realizations.columns)
        opener_row = res.open_lots["source_row"].iloc[0]
        self.assertEqual(res.realizations["open_source_row"].iloc[0],
                         opener_row)


def _bond_group(clean, face, open_date, maturity, qty_rem=None):
    qty_open = face
    return pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "000000AA1",
        "open_date": pd.Timestamp(open_date),
        "maturity": pd.Timestamp(maturity),
        "quantity_open": qty_open,
        "quantity_remaining": qty_open if qty_rem is None else qty_rem,
        "basis_open": clean, "basis_remaining": clean,
        "source_row": 1, "basis_evidence": "reconstructed"}])


class TestAccretionWindow(unittest.TestCase):
    def test_discount_window_halfway(self):
        from parsers.lot_engine import _accretion_window
        g = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01")
        lo, hi = _accretion_window(g, pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(lo, 9_400.0, places=2)
        self.assertAlmostEqual(hi, 9_700.0, places=2)   # halfway to face

    def test_premium_window_mirrors(self):
        from parsers.lot_engine import _accretion_window
        g = _bond_group(10_600.0, 10_000.0, "2025-01-01", "2027-01-01")
        lo, hi = _accretion_window(g, pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(lo, 10_300.0, places=2)  # amortizing down
        self.assertAlmostEqual(hi, 10_600.0, places=2)

    def test_matured_window_stays_full_width(self):
        from parsers.lot_engine import _accretion_window
        g = _bond_group(9_400.0, 10_000.0, "2024-01-01", "2025-01-01")
        lo, hi = _accretion_window(g, pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(lo, 9_400.0, places=2)
        self.assertAlmostEqual(hi, 10_000.0, places=2)

    def test_partial_remaining_scales(self):
        from parsers.lot_engine import _accretion_window
        g = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01",
                        qty_rem=5_000.0)
        lo, hi = _accretion_window(g, pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(lo, 4_700.0, places=2)
        self.assertAlmostEqual(hi, 4_850.0, places=2)

    def test_missing_maturity_returns_none(self):
        from parsers.lot_engine import _accretion_window
        g = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01")
        g.loc[0, "maturity"] = pd.NaT
        self.assertIsNone(_accretion_window(g, pd.Timestamp("2026-01-01")))

    def test_missing_maturity_column_returns_none(self):
        from parsers.lot_engine import _accretion_window
        g = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01")
        g = g.drop(columns=["maturity"])
        self.assertIsNone(_accretion_window(g, pd.Timestamp("2026-01-01")))

    def test_equity_price_shape_refused(self):
        # share-price-shaped lot (basis/qty = 50.0) with a maturity must not
        # get an envelope — the bond premise gate
        g = _bond_group(5_000.0, 100.0, "2025-01-01", "2027-01-01")
        self.assertIsNone(
            lot_engine._accretion_window(g, pd.Timestamp("2026-01-01")))

    def test_below_price_floor_refused(self):
        # clean/face = 0.3, below _BOND_PRICE_PER_FACE_MIN (0.5) — the
        # bond-premise gate's lower bound, mirroring the upper-bound check
        # above
        g = _bond_group(3_000.0, 10_000.0, "2025-01-01", "2027-01-01")
        self.assertIsNone(
            lot_engine._accretion_window(g, pd.Timestamp("2026-01-01")))


class TestAccretionBand(unittest.TestCase):
    def _recon(self, reported):
        lots = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01")
        pos = pd.DataFrame([{
            "account_id": "ACCT-1", "symbol": "", "cusip": "000000AA1",
            "description": "SYNTH TREASURY NOTE", "asset_class": "bond",
            "quantity": 10_000.0, "cost_basis": reported,
            "market_value": reported,
            "statement_date": "2026-01-01"}])
        return reconcile_lots(lots, pos)

    def test_reported_inside_window_bands_accretion_ok(self):
        # halfway accreted print: inside [9400, 9700]
        rec = self._recon(9_650.0)
        self.assertEqual(rec["band"].iloc[0], "accretion_ok")

    def test_purchase_cost_month_stays_ok_not_accretion(self):
        # exact purchase-cost print is plain ok — the envelope only fires
        # where classify_basis said watch/error
        rec = self._recon(9_400.0)
        self.assertEqual(rec["band"].iloc[0], "ok")

    def test_reported_outside_window_still_fails(self):
        rec = self._recon(9_950.0)   # above the halfway ceiling + tol
        self.assertEqual(rec["band"].iloc[0], "watch")

    def test_reported_below_floor_still_fails(self):
        rec = self._recon(9_100.0)   # below clean purchase cost - tol
        self.assertEqual(rec["band"].iloc[0], "watch")


def _wash_fixture(delta_matches=True, detected=True, self_match=False):
    """One loss sell + one still-open replacement lot, synthetic numbers.

    Sold 4 sh at a 260.00 loss; replacement 6 sh bought inside the window.
    Broker folds the disallowed loss into reported basis: reported =
    reconstructed + 260.00 when delta_matches.
    """
    opener_row = 11
    repl_row = opener_row if self_match else 12
    lots = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "open_date": pd.Timestamp("2026-03-25"),
        "maturity": pd.NaT,
        "quantity_open": 6.0, "quantity_remaining": 6.0,
        "basis_open": 540.0, "basis_remaining": 540.0,
        "source_row": repl_row, "basis_evidence": "reconstructed"}])
    rz = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "close_reason": "sell" if detected else "merger_cash",
        "close_date": pd.Timestamp("2026-03-30"),
        "quantity_closed": 4.0, "realized_gl": -260.00,
        "open_source_row": opener_row}])
    reported = 540.0 + (260.00 if delta_matches else 137.55)
    pos = pd.DataFrame([{
        "account_id": "ACCT-1", "symbol": "SYNW", "cusip": "",
        "description": "SYNTH CORP", "asset_class": "equity",
        "quantity": 6.0, "cost_basis": reported, "market_value": reported,
        "statement_date": "2026-04-30"}])
    return lots, pos, rz


def _wash_double_fixture(reported_delta):
    """One still-open replacement lot (bought 2026-03-25, quantity_open=6,
    quantity_remaining=5 -- one of its own shares already sold elsewhere)
    facing TWO loss sells of the same account/instrument, 3 days apart.

    Correct capacity-consumed delta is 260.00*(4/4) + 260.00*(1/4) =
    325.00: the first sell (2026-03-28) claims 4 of the lot's 5 still-held
    shares, leaving 1; the second sell (2026-03-31) claims only that 1.
    Two numbers a broken implementation could produce instead: 260+260=
    520.00 if capacity is never consumed across sells (each sell sees the
    full lot independently -- the pre-fix bug), or 260*(4/4)+260*(2/4)=
    390.00 if capacity IS consumed but keyed on quantity_open (6.0) rather
    than quantity_remaining (5.0). Neither is 325.00.
    """
    lots = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "open_date": pd.Timestamp("2026-03-25"),
        "maturity": pd.NaT,
        "quantity_open": 6.0, "quantity_remaining": 5.0,
        "basis_open": 540.0, "basis_remaining": 450.0,
        "source_row": 20, "basis_evidence": "reconstructed"}])
    rz = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "close_reason": "sell",
        "close_date": pd.Timestamp("2026-03-28"),
        "quantity_closed": 4.0, "realized_gl": -260.00,
        "open_source_row": 21},
        {"account_id": "ACCT-1", "instrument_key": "SYNW",
        "close_reason": "sell",
        "close_date": pd.Timestamp("2026-03-31"),
        "quantity_closed": 4.0, "realized_gl": -260.00,
        "open_source_row": 22}])
    reported = 450.0 + reported_delta
    pos = pd.DataFrame([{
        "account_id": "ACCT-1", "symbol": "SYNW", "cusip": "",
        "description": "SYNTH CORP", "asset_class": "equity",
        "quantity": 5.0, "cost_basis": reported, "market_value": reported,
        "statement_date": "2026-04-30"}])
    return lots, pos, rz


def _wash_single_sell_fixture(realized_gl, reported_delta=260.00):
    """One loss-shaped sell against one still-open replacement lot, for
    pinning the guards that gate entry into the capacity-consumption loop
    (positive gains, NaN realized_gl) rather than the consumption math
    itself."""
    lots = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "open_date": pd.Timestamp("2026-03-25"),
        "maturity": pd.NaT,
        "quantity_open": 6.0, "quantity_remaining": 6.0,
        "basis_open": 540.0, "basis_remaining": 540.0,
        "source_row": 12, "basis_evidence": "reconstructed"}])
    rz = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "close_reason": "sell",
        "close_date": pd.Timestamp("2026-03-30"),
        "quantity_closed": 4.0, "realized_gl": realized_gl,
        "open_source_row": 11}])
    reported = 540.0 + reported_delta
    pos = pd.DataFrame([{
        "account_id": "ACCT-1", "symbol": "SYNW", "cusip": "",
        "description": "SYNTH CORP", "asset_class": "equity",
        "quantity": 6.0, "cost_basis": reported, "market_value": reported,
        "statement_date": "2026-04-30"}])
    return lots, pos, rz


def _wash_multi_lot_fixture():
    """One loss sell of 6 sh spread across TWO still-open replacement lots
    (4 sh each, distinct rows, both in-window). claim = min(8, 6) = 6;
    disallowed = 300.00. FIFO-by-acquisition: lot A (03-25) gives 4 ->
    200.00, lot B (03-26) gives 2 -> 100.00. Broker folds it into reported
    basis: (400 + 400) + 300 = 1100.00."""
    lots = pd.DataFrame([
        {"account_id": "ACCT-1", "instrument_key": "SYNW",
         "open_date": pd.Timestamp("2026-03-25"), "maturity": pd.NaT,
         "quantity_open": 4.0, "quantity_remaining": 4.0,
         "basis_open": 400.0, "basis_remaining": 400.0,
         "source_row": 20, "basis_evidence": "reconstructed"},
        {"account_id": "ACCT-1", "instrument_key": "SYNW",
         "open_date": pd.Timestamp("2026-03-26"), "maturity": pd.NaT,
         "quantity_open": 4.0, "quantity_remaining": 4.0,
         "basis_open": 400.0, "basis_remaining": 400.0,
         "source_row": 21, "basis_evidence": "reconstructed"}])
    rz = pd.DataFrame([{
        "account_id": "ACCT-1", "instrument_key": "SYNW",
        "close_reason": "sell", "close_date": pd.Timestamp("2026-03-30"),
        "quantity_closed": 6.0, "realized_gl": -300.00,
        "open_source_row": 10}])
    reported = 800.0 + 300.0
    pos = pd.DataFrame([{
        "account_id": "ACCT-1", "symbol": "SYNW", "cusip": "",
        "description": "SYNTH CORP", "asset_class": "equity",
        "quantity": 8.0, "cost_basis": reported, "market_value": reported,
        "statement_date": "2026-04-30"}])
    return lots, pos, rz


class TestWashConsistentBand(unittest.TestCase):
    def test_matching_delta_bands_wash_consistent(self):
        lots, pos, rz = _wash_fixture()
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertEqual(rec["band"].iloc[0], "wash_consistent")

    def test_near_miss_delta_fails_normally(self):
        lots, pos, rz = _wash_fixture(delta_matches=False)
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertIn(rec["band"].iloc[0], ("watch", "error"))

    def test_no_detection_never_bands(self):
        # same wash-shaped delta but the close is not a sell — no detection
        lots, pos, rz = _wash_fixture(detected=False)
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertIn(rec["band"].iloc[0], ("watch", "error"))

    def test_self_match_excluded(self):
        # the "replacement" is the sold lot's own remainder -> excluded
        lots, pos, rz = _wash_fixture(self_match=True)
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertIn(rec["band"].iloc[0], ("watch", "error"))

    def test_no_realizations_frame_is_inert(self):
        lots, pos, _rz = _wash_fixture()
        rec = lot_engine.reconcile_lots(lots, pos)
        self.assertIn(rec["band"].iloc[0], ("watch", "error"))

    def test_window_days_matches_tax_scanner(self):
        from parsers import tax_scanner
        self.assertEqual(lot_engine._WASH_WINDOW_DAYS,
                         tax_scanner.WINDOW_DAYS)

    def test_shared_replacement_lot_not_double_counted(self):
        # capacity is quantity_remaining (5.0), consumed chronologically:
        # sell 1 (2026-03-28) claims 4/4 leaving 1.0; sell 2 (2026-03-31,
        # 3 days later) claims only 1/4 of its own loss. 260*(4/4) +
        # 260*(1/4) = 325.00. Using quantity_open (6.0) instead of
        # quantity_remaining would give 260*(4/4) + 260*(2/4) = 390.00 --
        # not 325.00 -- so the capacity base matters as much as the
        # consumption bookkeeping (see _wash_double_fixture's docstring).
        lots, pos, rz = _wash_double_fixture(325.00)
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertEqual(rec["band"].iloc[0], "wash_consistent")

        # Discriminator: the pre-fix per-sell-independent code let BOTH
        # sells claim the full still-open lot with no consumption between
        # them, summing to 260+260=520.00. That inflated sum must NOT
        # register as wash-consistent -- this assertion bands
        # "wash_consistent" (wrongly) on the pre-fix code.
        lots2, pos2, rz2 = _wash_double_fixture(520.00)
        rec2 = lot_engine.reconcile_lots(lots2, pos2, realizations=rz2)
        self.assertIn(rec2["band"].iloc[0], ("watch", "error"))

    def test_positive_gain_sell_never_counts(self):
        lots, pos, rz = _wash_single_sell_fixture(260.00)
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertIn(rec["band"].iloc[0], ("watch", "error"))

    def test_nan_realized_gl_inert(self):
        lots, pos, rz = _wash_single_sell_fixture(float("nan"))
        rec = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertIn(rec["band"].iloc[0], ("watch", "error"))


class TestWashFoldPlan(unittest.TestCase):
    PAIR = ("ACCT-1", "SYNW")

    def test_single_wash_plan_totals_and_lines(self):
        lots, _pos, rz = _wash_fixture()
        plan = lot_engine._wash_fold_plan(rz, self.PAIR, lots)
        self.assertAlmostEqual(plan["total"], 260.00, places=2)
        self.assertEqual(set(plan["disallowed_by_sell"]), {0})
        self.assertAlmostEqual(plan["disallowed_by_sell"][0], 260.00, places=2)
        self.assertAlmostEqual(plan["added_by_lot"][0], 260.00, places=2)

    def test_delta_delegates_to_plan_total(self):
        lots, _pos, rz = _wash_fixture()
        self.assertAlmostEqual(
            lot_engine._wash_consistent_delta(rz, self.PAIR, lots),
            260.00, places=2)

    def test_double_sell_consumes_capacity_and_splits_to_one_lot(self):
        lots, _pos, rz = _wash_double_fixture(325.00)
        plan = lot_engine._wash_fold_plan(rz, self.PAIR, lots)
        self.assertAlmostEqual(plan["total"], 325.00, places=2)
        self.assertAlmostEqual(plan["disallowed_by_sell"][0], 260.00, places=2)
        self.assertAlmostEqual(plan["disallowed_by_sell"][1], 65.00, places=2)
        self.assertAlmostEqual(plan["added_by_lot"][0], 325.00, places=2)

    def test_positive_gain_sell_yields_empty_plan(self):
        lots, _pos, rz = _wash_single_sell_fixture(260.00)  # positive gl
        plan = lot_engine._wash_fold_plan(rz, self.PAIR, lots)
        self.assertEqual(plan["total"], 0.0)
        self.assertEqual(plan["disallowed_by_sell"], {})

    def test_no_realizations_frame_is_inert(self):
        lots, _pos, _rz = _wash_fixture()
        self.assertEqual(
            lot_engine._wash_fold_plan(None, self.PAIR, lots),
            {"disallowed_by_sell": {}, "added_by_lot": {}, "total": 0.0})

    def test_single_sell_splits_across_two_lots(self):
        lots, _pos, rz = _wash_multi_lot_fixture()
        plan = lot_engine._wash_fold_plan(rz, self.PAIR, lots)
        self.assertAlmostEqual(plan["total"], 300.00, places=2)
        self.assertEqual(set(plan["disallowed_by_sell"]), {0})
        self.assertAlmostEqual(plan["added_by_lot"][0], 200.00, places=2)
        self.assertAlmostEqual(plan["added_by_lot"][1], 100.00, places=2)


class TestApplyWashFolds(unittest.TestCase):
    def _ledger(self, lots, rz):
        return lot_engine.LotLedgerResult(
            open_lots=lots.assign(wash_adjustment=0.0),
            realizations=rz.assign(disallowed_wash=0.0),
            exceptions=pd.DataFrame())

    def test_fold_moves_loss_into_basis_and_re_reconciles_ok(self):
        lots, pos, rz = _wash_fixture()
        recon = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertEqual(recon["band"].iloc[0], "wash_consistent")
        ledger = self._ledger(lots, rz)
        lot_engine.apply_wash_folds(ledger, recon)
        lot = ledger.open_lots.iloc[0]
        self.assertAlmostEqual(lot["basis_remaining"], 800.00, places=2)
        self.assertAlmostEqual(lot["wash_adjustment"], 260.00, places=2)
        sale = ledger.realizations.iloc[0]
        self.assertAlmostEqual(sale["realized_gl"], 0.00, places=2)
        self.assertAlmostEqual(sale["disallowed_wash"], 260.00, places=2)
        recon2 = lot_engine.reconcile_lots(
            ledger.open_lots, pos, realizations=ledger.realizations)
        self.assertEqual(recon2["band"].iloc[0], "ok")

    def test_non_wash_pair_is_untouched(self):
        lots, pos, rz = _wash_fixture(delta_matches=False)
        recon = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertIn(recon["band"].iloc[0], ("watch", "error"))
        ledger = self._ledger(lots, rz)
        lot_engine.apply_wash_folds(ledger, recon)
        self.assertAlmostEqual(
            ledger.open_lots.iloc[0]["basis_remaining"], 540.00, places=2)
        self.assertAlmostEqual(
            ledger.open_lots.iloc[0]["wash_adjustment"], 0.00, places=2)
        self.assertAlmostEqual(
            ledger.realizations.iloc[0]["realized_gl"], -260.00, places=2)

    def test_fold_distributes_across_multiple_lots(self):
        lots, pos, rz = _wash_multi_lot_fixture()
        recon = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertEqual(recon["band"].iloc[0], "wash_consistent")
        ledger = self._ledger(lots, rz)
        lot_engine.apply_wash_folds(ledger, recon)
        a, b = ledger.open_lots.iloc[0], ledger.open_lots.iloc[1]
        self.assertAlmostEqual(a["basis_remaining"], 600.00, places=2)
        self.assertAlmostEqual(a["wash_adjustment"], 200.00, places=2)
        self.assertAlmostEqual(b["basis_remaining"], 500.00, places=2)
        self.assertAlmostEqual(b["wash_adjustment"], 100.00, places=2)
        self.assertAlmostEqual(
            ledger.realizations.iloc[0]["realized_gl"], 0.00, places=2)
        recon2 = lot_engine.reconcile_lots(
            ledger.open_lots, pos, realizations=ledger.realizations)
        self.assertEqual(recon2["band"].iloc[0], "ok")

    def test_apply_is_idempotent_on_partial_capacity(self):
        # Partial: sell 10 sh at -500 but only 4 replacement shares -> claim 4,
        # disallowed 200; realized_gl stays -300 (< 0) after the fold, and the
        # replacement's quantity_remaining is unchanged, so a naive SECOND call
        # on the same (stale) recon would re-fold. The guard must make it a
        # no-op.
        lots = pd.DataFrame([{
            "account_id": "ACCT-1", "instrument_key": "SYNW",
            "open_date": pd.Timestamp("2026-03-25"), "maturity": pd.NaT,
            "quantity_open": 4.0, "quantity_remaining": 4.0,
            "basis_open": 400.0, "basis_remaining": 400.0,
            "source_row": 20, "basis_evidence": "reconstructed"}])
        rz = pd.DataFrame([{
            "account_id": "ACCT-1", "instrument_key": "SYNW",
            "close_reason": "sell", "close_date": pd.Timestamp("2026-03-30"),
            "quantity_closed": 10.0, "realized_gl": -500.00,
            "open_source_row": 10}])
        pos = pd.DataFrame([{
            "account_id": "ACCT-1", "symbol": "SYNW", "cusip": "",
            "description": "SYNTH CORP", "asset_class": "equity",
            "quantity": 4.0, "cost_basis": 600.0, "market_value": 600.0,
            "statement_date": "2026-04-30"}])
        recon = lot_engine.reconcile_lots(lots, pos, realizations=rz)
        self.assertEqual(recon["band"].iloc[0], "wash_consistent")
        ledger = self._ledger(lots, rz)
        lot_engine.apply_wash_folds(ledger, recon)        # first fold
        self.assertAlmostEqual(
            ledger.open_lots.iloc[0]["basis_remaining"], 600.00, places=2)
        self.assertAlmostEqual(
            ledger.open_lots.iloc[0]["wash_adjustment"], 200.00, places=2)
        self.assertAlmostEqual(
            ledger.realizations.iloc[0]["realized_gl"], -300.00, places=2)
        lot_engine.apply_wash_folds(ledger, recon)        # SAME stale recon
        self.assertAlmostEqual(                            # must be a no-op
            ledger.open_lots.iloc[0]["basis_remaining"], 600.00, places=2)
        self.assertAlmostEqual(
            ledger.open_lots.iloc[0]["wash_adjustment"], 200.00, places=2)
        self.assertAlmostEqual(
            ledger.realizations.iloc[0]["realized_gl"], -300.00, places=2)


class TestRedemptionFaceRelief(unittest.TestCase):
    def _ledger(self, redeem_date):
        return build_lot_ledger(tx_frame(
            tx(trade_date="2025-01-10", transaction_type="buy",
               account_id="ACCT-1", symbol=np.nan, cusip="000000AA1",
               quantity=10_000, price=94.0, amount=-9_400.00,
               description="SYNTH TREASURY NOTE 06/30/2026 MATURITY"),
            tx(trade_date=redeem_date, transaction_type="redemption",
               account_id="ACCT-1", symbol=np.nan, cusip="000000AA1",
               quantity=-10_000, amount=10_000.00,
               description="SYNTH TREASURY NOTE FULL CALL")))

    def test_maturity_redemption_relieves_at_face(self):
        ledger = self._ledger("2026-06-30")
        rz = ledger.realizations
        r = rz[rz["close_reason"] == "redemption"].iloc[0]
        self.assertAlmostEqual(r["basis_closed"], 10_000.00, places=2)
        self.assertAlmostEqual(r["realized_gl"], 0.00, places=2)
        self.assertEqual(r["basis_source"], "amortized_face")

    def test_lot_fully_empties_no_negative_basis(self):
        ledger = self._ledger("2026-06-30")
        pair_lots = ledger.open_lots[
            ledger.open_lots["instrument_key"].str.contains("000000AA1")]
        self.assertTrue(pair_lots.empty)   # fully closed, nothing stranded

    def test_pre_maturity_redemption_unchanged(self):
        ledger = self._ledger("2025-09-30")   # well before maturity
        rz = ledger.realizations
        r = rz[rz["close_reason"] == "redemption"].iloc[0]
        self.assertAlmostEqual(r["basis_closed"], 9_400.00, places=2)
        self.assertEqual(r["basis_source"], "reconstructed")

    def test_share_priced_instrument_never_face_relieved(self):
        # a preferred-like lot: 100 shares at $50, maturity text on the
        # opener, called at par -- the bond-premise gate must refuse face
        # relief (basis/qty = 50, far outside the [0.5, 2.0] band)
        ledger = build_lot_ledger(tx_frame(
            tx(trade_date="2025-01-10", transaction_type="buy",
               account_id="ACCT-1", symbol=np.nan, cusip="000000CC3",
               quantity=100, price=50.0, amount=-5_000.00,
               description="SYNTH PREFERRED 06/30/2026 MATURITY"),
            tx(trade_date="2026-06-30", transaction_type="redemption",
               account_id="ACCT-1", symbol=np.nan, cusip="000000CC3",
               quantity=-100, amount=5_000.00,
               description="SYNTH PREFERRED FULL CALL")))
        rz = ledger.realizations
        r = rz[rz["close_reason"] == "redemption"].iloc[0]
        self.assertAlmostEqual(r["basis_closed"], 5_000.00, places=2)
        self.assertAlmostEqual(r["realized_gl"], 0.00, places=2)
        self.assertEqual(r["basis_source"], "reconstructed")


class TestAlternatingMonths(unittest.TestCase):
    """The bistable-printing shape: SAME lots judged against a purchase-cost
    month and an amortized month must BOTH land in OK_BANDS."""
    def _pos(self, statement_date, reported):
        return pd.DataFrame([{
            "account_id": "ACCT-1", "symbol": "", "cusip": "000000AA1",
            "description": "SYNTH TREASURY NOTE", "asset_class": "bond",
            "quantity": 10_000.0, "cost_basis": reported,
            "market_value": reported, "statement_date": statement_date}])

    def test_both_month_kinds_pass(self):
        lots = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01")
        # purchase-cost month
        rec1 = lot_engine.reconcile_lots(lots, self._pos("2026-01-31",
                                                         9_400.00))
        self.assertIn(rec1["band"].iloc[0], OK_BANDS)
        # amortized month (a print inside the elapsed-fraction ceiling)
        rec2 = lot_engine.reconcile_lots(lots, self._pos("2026-01-31",
                                                         9_680.00))
        self.assertIn(rec2["band"].iloc[0], OK_BANDS)

    def test_corrupted_report_stays_caught(self):
        # a wrong-extraction basis far outside the envelope must still band
        # error — the envelope does not eat real errors
        lots = _bond_group(9_400.0, 10_000.0, "2025-01-01", "2027-01-01")
        rec = lot_engine.reconcile_lots(lots, self._pos("2026-01-31",
                                                        12_800.00))
        self.assertEqual(rec["band"].iloc[0], "error")


class TestCollectBondMaturities(unittest.TestCase):
    def _tx(self, descs):
        return pd.DataFrame([
            {"account_id": "ACCT-1", "symbol": "", "cusip": "000000AA1",
             "description": d, "transaction_type": "interest",
             "quantity": float("nan"), "price": float("nan"),
             "amount": 4.7, "settlement_date": "2025-06-30"}
            for d in descs])

    def test_single_due_date_enriches(self):
        result = lot_engine.collect_bond_maturities(
            self._tx(["SYNTH NOTE DUE 06/30/2026 INTEREST"]))
        self.assertEqual(result, {"000000AA1": pd.Timestamp("2026-06-30")})

    def test_conflicting_dates_skip(self):
        result = lot_engine.collect_bond_maturities(
            self._tx(["DUE 06/30/2026 X", "DUE 09/30/2026 Y"]))
        self.assertNotIn("000000AA1", result)

    def test_no_evidence_inert(self):
        result = lot_engine.collect_bond_maturities(
            self._tx(["SYNTH NOTE COUPON PAYMENT"]))
        self.assertNotIn("000000AA1", result)

    def test_unrelated_instrument_rows_ignored(self):
        tx = self._tx(["DUE 06/30/2026 X"])
        tx.loc[0, "cusip"] = "000000BB2"
        result = lot_engine.collect_bond_maturities(tx)
        self.assertNotIn("000000AA1", result)

    def test_invalid_calendar_date_skipped(self):
        result = lot_engine.collect_bond_maturities(
            self._tx(["SYNTH NOTE DUE 13/45/2026 X"]))
        self.assertNotIn("000000AA1", result)

    def test_invalid_beside_valid_still_enriches(self):
        result = lot_engine.collect_bond_maturities(
            self._tx(["SYNTH DUE 13/45/2026 X", "SYNTH DUE 06/30/2026 Y"]))
        self.assertEqual(result, {"000000AA1": pd.Timestamp("2026-06-30")})


class TestMaturityByKeyIntegration(unittest.TestCase):
    """collect_bond_maturities -> build_lot_ledger(maturity_by_key=...),
    wired PRE-replay (FIX 2): the exact seam that lets a note-class
    redemption face-relieve even though its own opener prints no maturity
    text."""

    def test_note_class_redemption_face_relieves_from_interest_row(self):
        # the note-class scenario the finding reproduced: the OPENER prints
        # no MATURITY text at all (a "note" shape, unlike the bill/treasury
        # shape RE_MATURITY covers), but an interest row elsewhere in the
        # same instrument's history prints DUE -- collected PRE-replay, that
        # evidence must still reach the lot in time for the redemption to
        # face-relieve.
        frame = tx_frame(
            tx(trade_date="2025-01-10", transaction_type="buy",
               account_id="ACCT-1", symbol=np.nan, cusip="000000AA1",
               quantity=10_000, price=94.0, amount=-9_400.00,
               description="SYNTH TREASURY NOTE"),
            tx(trade_date="2025-06-30", transaction_type="interest",
               account_id="ACCT-1", symbol=np.nan, cusip="000000AA1",
               quantity=np.nan, amount=4.7,
               description="SYNTH TREASURY NOTE DUE 06/30/2026 INTEREST"),
            tx(trade_date="2026-06-30", transaction_type="redemption",
               account_id="ACCT-1", symbol=np.nan, cusip="000000AA1",
               quantity=-10_000, amount=10_000.00,
               description="SYNTH TREASURY NOTE FULL CALL"))
        resolver, cusip_resolver = build_key_resolvers(frame, None)
        maturity_by_key = lot_engine.collect_bond_maturities(
            frame, resolver, cusip_resolver)
        ledger = build_lot_ledger(frame, maturity_by_key=maturity_by_key)
        rz = ledger.realizations
        r = rz[rz["close_reason"] == "redemption"].iloc[0]
        self.assertAlmostEqual(r["basis_closed"], 10_000.00, places=2)
        self.assertAlmostEqual(r["realized_gl"], 0.00, places=2)
        self.assertEqual(r["basis_source"], "amortized_face")

    def test_parsed_maturity_wins_over_conflicting_map_entry(self):
        frame = tx_frame(
            tx(trade_date="2025-01-10", transaction_type="buy",
               account_id="ACCT-1", symbol=np.nan, cusip="000000AA1",
               quantity=10_000, price=94.0, amount=-9_400.00,
               description="SYNTH TREASURY NOTE 03/31/2027 MATURITY"))
        conflicting_map = {"000000AA1": pd.Timestamp("2099-01-01")}
        ledger = build_lot_ledger(frame, maturity_by_key=conflicting_map)
        lot = ledger.open_lots.iloc[0]
        self.assertEqual(pd.Timestamp("2027-03-31"),
                         pd.to_datetime(lot["maturity"]))
        self.assertEqual(ledger.n_maturity_enriched, 0)


class TestDaysToLongTerm(unittest.TestCase):
    """days_to_long_term counts down to the LT boundary and stays in
    lockstep with classify_term (they share the anniversary)."""

    def test_boundary_day_before_on_after(self):
        acq = pd.Timestamp("2025-06-28")
        # anniversary = 2026-06-28; first LT day = 2026-06-29
        self.assertEqual(days_to_long_term(acq, pd.Timestamp("2026-06-27")), 2)
        self.assertEqual(days_to_long_term(acq, pd.Timestamp("2026-06-28")), 1)  # anniversary: still short
        self.assertIsNone(days_to_long_term(acq, pd.Timestamp("2026-06-29")))    # first LT day

    def test_missing_acquired_is_none(self):
        self.assertIsNone(days_to_long_term(None, pd.Timestamp("2026-06-28")))
        self.assertIsNone(days_to_long_term(pd.NaT, pd.Timestamp("2026-06-28")))

    def test_leap_year_anchor_agrees_with_classify_term(self):
        acq = pd.Timestamp("2024-02-29")  # anniversary rolls to 2025-02-28
        for as_of in (pd.Timestamp("2025-02-27"), pd.Timestamp("2025-02-28"),
                      pd.Timestamp("2025-03-01")):
            with self.subTest(as_of=as_of):
                d = days_to_long_term(acq, as_of)
                is_short = classify_term(acq, as_of) == "short"
                self.assertEqual(d is not None, is_short)
                if d is not None:
                    self.assertGreaterEqual(d, 1)

    def test_consistency_invariant_across_grid(self):
        acqs = [pd.Timestamp("2024-02-29"), pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-06-28"), pd.Timestamp("2025-12-31")]
        as_ofs = [pd.Timestamp("2025-06-28"), pd.Timestamp("2026-01-01"),
                  pd.Timestamp("2026-06-28"), pd.Timestamp("2026-12-31")]
        for acq in acqs:
            for as_of in as_ofs:
                with self.subTest(acq=acq, as_of=as_of):
                    d = days_to_long_term(acq, as_of)
                    term = classify_term(acq, as_of)
                    if term == "short":
                        self.assertIsNotNone(d)
                        self.assertGreaterEqual(d, 1)
                    else:  # "long" or "unknown"
                        self.assertIsNone(d)

    def test_missing_as_of_is_none(self):
        # as_of is always concrete in production; a NaT/None reference date
        # deliberately yields None (an undefined countdown), by design NOT
        # classify_term's "short".
        acq = pd.Timestamp("2025-06-28")
        self.assertIsNone(days_to_long_term(acq, None))
        self.assertIsNone(days_to_long_term(acq, pd.NaT))


class TestWashFoldSchema(unittest.TestCase):
    def test_open_columns_carries_wash_adjustment(self):
        self.assertIn("wash_adjustment", lot_engine.OPEN_COLUMNS)

    def test_realization_columns_carries_disallowed_wash(self):
        self.assertIn("disallowed_wash", lot_engine.REALIZATION_COLUMNS)

    def test_lot_open_row_defaults_wash_adjustment_zero(self):
        lot = lot_engine._Lot(
            account_id="ACCT-1", instrument_key="SYNW", key_source="symbol",
            symbol="SYNW", open_date=pd.Timestamp("2026-03-25"),
            acquired_date=pd.Timestamp("2026-03-25"), origin="buy",
            quantity_open=6.0, quantity_remaining=6.0,
            basis_open=540.0, basis_remaining=540.0, source_row=12)
        self.assertEqual(lot.open_row()["wash_adjustment"], 0.0)

    def test_close_defaults_disallowed_wash_zero(self):
        replay = lot_engine._Replay(set())
        lot = lot_engine._Lot(
            account_id="ACCT-1", instrument_key="SYNW", key_source="symbol",
            symbol="SYNW", open_date=pd.Timestamp("2026-03-01"),
            acquired_date=pd.Timestamp("2026-03-01"), origin="buy",
            quantity_open=4.0, quantity_remaining=4.0,
            basis_open=400.0, basis_remaining=400.0, source_row=0)
        replay._close(lot, 4.0, 400.0, 140.0,
                      pd.Timestamp("2026-03-30"), "sell")
        self.assertEqual(replay.realizations[-1]["disallowed_wash"], 0.0)


if __name__ == "__main__":
    unittest.main()
