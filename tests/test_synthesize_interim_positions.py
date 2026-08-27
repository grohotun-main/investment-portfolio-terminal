"""Tests for parsers/synthesize_interim_positions.py.

The interim-positions roller is live: `data/transactions_interim.csv` (managed
by parsers/ingest_csv_activity.py and the dashboard's "Pull interim
transactions" button) rolls the latest statement positions forward to a
unified mid-month snapshot. These tests cover the identity helper, the
per-transaction roll-forward invariants, and the per-account-base /
cash-landing / option-leg behavior that keeps a lagging account from
vanishing from the snapshot.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from synthesize_interim_positions import (  # noqa: E402
    _key_for,
    _is_corporate_out_leg,
    _is_security_exchange_leg,
    _rescue_by_name,
    synthesize_interim_positions,
)


POSITIONS_COLS = [
    "statement_date", "broker", "account_id", "account_type",
    "symbol", "cusip", "description", "asset_class", "quantity",
    "price", "market_value", "cost_basis", "unrealized_gl",
    "est_annual_income", "currency", "source_file",
]


def _pos(symbol, cusip, asset_class: str, quantity: float,
         market_value: float, account_id: str = "TEST-1",
         date: str = "2026-04-30") -> dict:
    row = {c: None for c in POSITIONS_COLS}
    row.update({
        "statement_date": pd.Timestamp(date), "broker": "fidelity",
        "account_id": account_id, "account_type": "BROKERAGE",
        "symbol": symbol, "cusip": cusip, "description": symbol or "",
        "asset_class": asset_class, "quantity": quantity,
        "price": (market_value / quantity) if quantity else 0.0,
        "market_value": market_value, "cost_basis": market_value,
        "unrealized_gl": 0.0, "est_annual_income": 0.0,
        "currency": "USD", "source_file": "test.pdf",
    })
    return row


def _txn(date: str, ttype: str, symbol, cusip, quantity, amount,
         account_id: str = "TEST-1", price: float = 0.0) -> dict:
    return {
        "settlement_date": pd.Timestamp(date), "broker": "fidelity",
        "account_id": account_id, "transaction_type": ttype,
        "symbol": symbol, "cusip": cusip, "description": ttype,
        "quantity": quantity, "price": price, "amount": amount,
        "source_file": "interim.csv", "trade_date": None,
    }


class TestKeyFor(unittest.TestCase):
    def test_symbol_present_uses_symbol(self) -> None:
        self.assertEqual(_key_for("SPY", None), "SYM:SPY")

    def test_cusip_used_when_symbol_missing(self) -> None:
        # Bare-CUSIP Treasuries arrive with symbol=NaN. Falling back to
        # cusip keeps them addressable.
        self.assertEqual(_key_for(None, "912828X88"), "CUSIP:912828X88")

    def test_both_missing_returns_none(self) -> None:
        # Sweep journals etc. can have neither — caller treats None-key
        # rows as untouched.
        self.assertIsNone(_key_for(None, None))
        self.assertIsNone(_key_for(float("nan"), float("nan")))

    def test_whitespace_only_treated_as_missing(self) -> None:
        # "   " in symbol shouldn't shadow a valid CUSIP fallback.
        self.assertEqual(_key_for("  ", "912828X88"), "CUSIP:912828X88")


class TestSynthesizeInterimPositions(unittest.TestCase):
    def test_empty_interim_returns_empty_frame(self) -> None:
        positions = pd.DataFrame([_pos("SPY", None, "equity_etf", 10, 5000.0)])
        interim = pd.DataFrame(columns=[
            "settlement_date", "broker", "account_id", "transaction_type",
            "symbol", "cusip", "description", "quantity", "price",
            "amount", "source_file", "trade_date",
        ])
        out = synthesize_interim_positions(positions, interim)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), list(positions.columns))

    def test_quantity_delta_applied_to_existing_position(self) -> None:
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", quantity=10, market_value=5000.0),
            _pos("CASH", None, "cash", quantity=1000.0, market_value=1000.0),
        ])
        interim = pd.DataFrame([
            _txn("2026-05-15", "buy", "SPY", None,
                 quantity=2, amount=-1100.0, price=550.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        spy = out[out["symbol"] == "SPY"].iloc[0]
        # 10 + 2 = 12 shares. Unit value preserved at 5000/10 = 500, so
        # MV = 12 * 500 = 6000. (mark_to_market overlays real prices
        # later; the synthesizer just rolls quantities forward.)
        self.assertEqual(spy["quantity"], 12)
        self.assertAlmostEqual(spy["market_value"], 6000.0)

    def test_brand_new_symbol_creates_position_row(self) -> None:
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", quantity=10, market_value=5000.0),
            _pos("CASH", None, "cash", quantity=2000.0, market_value=2000.0),
        ])
        # User buys VTI for the first time mid-month.
        interim = pd.DataFrame([
            _txn("2026-05-15", "buy", "VTI", None,
                 quantity=5, amount=-1250.0, price=250.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        vti = out[out["symbol"] == "VTI"]
        self.assertEqual(len(vti), 1)
        self.assertEqual(vti.iloc[0]["quantity"], 5)
        # cost_basis = abs(amount) for the buy.
        self.assertAlmostEqual(vti.iloc[0]["cost_basis"], 1250.0)

    def test_cash_position_absorbs_per_account_net_cash_delta(self) -> None:
        # A $1000 deposit + $200 buy nets to +$800 cash for the account.
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", quantity=10, market_value=5000.0),
            _pos("CASH", None, "cash", quantity=1000.0, market_value=1000.0),
        ])
        interim = pd.DataFrame([
            _txn("2026-05-10", "transfer_in", None, None,
                 quantity=None, amount=1000.0),
            _txn("2026-05-15", "buy", "SPY", None,
                 quantity=0.4, amount=-200.0, price=500.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        cash = out[out["asset_class"] == "cash"].iloc[0]
        # 1000 + 1000 - 200 = 1800.
        self.assertAlmostEqual(cash["quantity"], 1800.0)
        self.assertAlmostEqual(cash["market_value"], 1800.0)


class TestLaggingAccountAndHardening(unittest.TestCase):
    """Regression tests for the Z10-000008 disappearance (Jun 2026).

    Fidelity issued May statements for most accounts but not for the
    individual-stocks sleeve (last real statement Apr-30). With interim June
    activity present, the roller dropped that account's entire base and showed
    only the net of June trades. Root cause: the per-broker base was filtered to
    the broker's *global* latest statement_date, excluding any account lagging
    behind it. These tests pin the per-account-latest base, cash-delta
    landing, and option-leg handling.
    """

    def test_lagging_account_rolls_forward_from_its_own_latest_base(self) -> None:
        # FRESH has a May-31 statement; LAG's last real statement is Apr-30
        # (its May statement was never issued). LAG buys 10 more AAPL in June.
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", quantity=10, market_value=5000.0,
                 account_id="FRESH", date="2026-05-31"),
            _pos("AAPL", None, "equity_stock", quantity=100, market_value=15000.0,
                 account_id="LAG", date="2026-04-30"),
            _pos("CASH", None, "cash", quantity=500.0, market_value=500.0,
                 account_id="LAG", date="2026-04-30"),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-02", "buy", "AAPL", None,
                 quantity=10, amount=-1600.0, price=160.0, account_id="LAG"),
        ])
        out = synthesize_interim_positions(positions, interim)
        aapl = out[(out["account_id"] == "LAG") & (out["symbol"] == "AAPL")]
        self.assertEqual(len(aapl), 1)
        # Base 100 + interim 10 = 110 — NOT a brand-new 10-share row that
        # silently discards the lagging account's existing holdings.
        self.assertEqual(aapl.iloc[0]["quantity"], 110)
        # LAG's untouched base cash row must also survive the roll-forward.
        lag_cash = out[(out["account_id"] == "LAG") & (out["asset_class"] == "cash")]
        self.assertEqual(len(lag_cash), 1)

    def test_cash_delta_without_existing_cash_row_creates_one(self) -> None:
        # Account holds only a stock — no cash position. A $300 dividend has
        # nowhere to land under the old code and is silently dropped.
        positions = pd.DataFrame([
            _pos("AAPL", None, "equity_stock", quantity=10, market_value=1500.0),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-02", "dividend", None, None,
                 quantity=None, amount=300.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        cash = out[out["asset_class"] == "cash"]
        self.assertEqual(len(cash), 1)
        self.assertAlmostEqual(cash.iloc[0]["market_value"], 300.0)

    def test_option_sell_closes_base_row_and_buy_opens_new_option_row(self) -> None:
        # Base: 6 SPY puts (symbol is the underlying, per statement format).
        # June: sell all 6 of the P560 strike, buy 4 of the P635 strike.
        # Correct result: the base put is closed (qty 0), a new option_put row
        # holds the 4 P635s at premium, and NOTHING is mislabeled "other".
        positions = pd.DataFrame([
            _pos("SPY", None, "option_put", quantity=6, market_value=4920.0),
            _pos("CASH", None, "cash", quantity=2000.0, market_value=2000.0),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-03", "sell", "-SPY261218P560", None,
                 quantity=-6, amount=3030.00, price=5.05),
            _txn("2026-06-03", "buy", "-SPY261218P635", None,
                 quantity=4, amount=-4180.00, price=10.45),
        ])
        out = synthesize_interim_positions(positions, interim)
        # Option legs must never be booked as asset_class "other".
        self.assertEqual(len(out[out["asset_class"] == "other"]), 0)
        puts = out[out["asset_class"] == "option_put"]
        # Closed P560 (0) + opened P635 (4) = 4 contracts net.
        self.assertAlmostEqual(puts["quantity"].sum(), 4.0)
        # Closed $4920 base → $0; opened P635 at its $4180.00 premium.
        self.assertAlmostEqual(puts["market_value"].sum(), 4180.00, places=2)

    # ---- JPM display-format legs (Aug 2026 phantom-puts incident) ---------
    # JPM's interim CSV carries the contract as a DISPLAY symbol ("SPY DEC 26
    # PUT 650.00") with the full contract in the description ("PUT SPY
    # 12/18/26 650 ..."); the statement row carries the bare underlying with
    # the same description shape. Before the fix these legs fell into the
    # equity path as "brand-new symbols", booking NEGATIVE phantom rows while
    # the statement puts were carried forward untouched.

    @staticmethod
    def _jpm_put(strike: int, qty: float, mv: float, cost: float) -> dict:
        row = _pos("SPY", None, "option_put", quantity=qty, market_value=mv)
        row["broker"] = "jpm"
        row["description"] = (f"PUT SPY 12/18/26 {strike} STATE STREET SPDR "
                              "S&P 500 ETF")
        row["cost_basis"] = cost
        return row

    @staticmethod
    def _jpm_leg(date: str, ttype: str, strike: int, qty: float,
                 amount: float, *, month: str = "DEC") -> dict:
        leg = _txn(date, ttype, f"SPY {month} 26 PUT {strike}.00", None,
                   quantity=qty, amount=amount)
        leg["broker"] = "jpm"
        verb = "CLOSING" if qty < 0 else "OPEN"
        leg["description"] = (f"PUT SPY 12/18/26 {strike} STATE STREET SPDR "
                              f"S&P 500 ETF UNSOLICITED {verb} CONTRACT")
        return leg

    def test_jpm_display_format_sell_closes_statement_put(self) -> None:
        cash = _pos("CASH", None, "cash", 2000.0, 2000.0)
        cash["broker"] = "jpm"
        positions = pd.DataFrame([self._jpm_put(650, 6, 4572.0, 6688.04), cash])
        interim = pd.DataFrame([self._jpm_leg("2026-08-12", "sell", 650, -6,
                                              3235.95)])
        out = synthesize_interim_positions(positions, interim)
        # No display-symbol phantom row, no "other" row.
        self.assertFalse((out["symbol"] == "SPY DEC 26 PUT 650.00").any())
        self.assertEqual(len(out[out["asset_class"] == "other"]), 0)
        puts = out[out["asset_class"] == "option_put"]
        self.assertEqual(len(puts), 1)
        self.assertAlmostEqual(float(puts.iloc[0]["quantity"]), 0.0)
        self.assertAlmostEqual(float(puts.iloc[0]["market_value"]), 0.0)
        # A full close leaves no statement cost basis behind (the Options tab
        # treats a positive statement cost_basis as authoritative).
        self.assertAlmostEqual(float(puts.iloc[0]["cost_basis"]), 0.0)
        # Proceeds land in cash.
        cash_mv = float(out[out["asset_class"] == "cash"]["market_value"].sum())
        self.assertAlmostEqual(cash_mv, 2000.0 + 3235.95, places=2)

    def test_jpm_display_format_close_matches_its_own_strike(self) -> None:
        # 650 ×6 and 670 ×4 on the statement; the 670s are sold. The
        # description carries expiry + strike, so the close binds to the 670
        # row exactly — never "first free same-underlying row".
        positions = pd.DataFrame([self._jpm_put(650, 6, 4572.0, 6688.04),
                                  self._jpm_put(670, 4, 3816.0, 5374.69)])
        interim = pd.DataFrame([self._jpm_leg("2026-08-12", "sell", 670, -4,
                                              2689.29)])
        out = synthesize_interim_positions(positions, interim)
        puts = out[out["asset_class"] == "option_put"].set_index("description")
        self.assertAlmostEqual(
            float(puts.loc["PUT SPY 12/18/26 650 STATE STREET SPDR S&P 500 ETF",
                           "quantity"]), 6.0)
        self.assertAlmostEqual(
            float(puts.loc["PUT SPY 12/18/26 670 STATE STREET SPDR S&P 500 ETF",
                           "quantity"]), 0.0)

    def test_partial_close_scales_statement_cost_basis(self) -> None:
        positions = pd.DataFrame([self._jpm_put(650, 6, 4572.0, 6688.04)])
        interim = pd.DataFrame([self._jpm_leg("2026-08-12", "sell", 650, -2,
                                              1078.65)])
        out = synthesize_interim_positions(positions, interim)
        put = out[out["asset_class"] == "option_put"].iloc[0]
        self.assertAlmostEqual(float(put["quantity"]), 4.0)
        self.assertAlmostEqual(float(put["market_value"]), 4572.0 * 4 / 6, places=2)
        self.assertAlmostEqual(float(put["cost_basis"]), 6688.04 * 4 / 6, places=2)

    def test_option_expiry_other_row_closes_the_contract(self) -> None:
        # Expirations land as `other` (Fidelity "EXPIRED ..." qty<0 amount 0;
        # JPM "Journal" qty<0 amount NaN). They carry no cash, but the
        # contracts DID leave the account — the quantity applies.
        base = _pos("META", None, "option_call", quantity=5, market_value=0.0)
        base["broker"] = "jpm"
        base["description"] = "CALL META 07/31/26 650 META PLATFORMS INC CL A"
        base["cost_basis"] = 3153.32
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0)
        cash["broker"] = "jpm"
        journal = _txn("2026-08-03", "other", "META JUL 26 CALL 650.00", None,
                       quantity=-5, amount=float("nan"))
        journal["broker"] = "jpm"
        journal["description"] = "CALL META 07/31/26 650 META PLATFORMS INC CL A"
        out = synthesize_interim_positions(pd.DataFrame([base, cash]),
                                           pd.DataFrame([journal]))
        calls = out[out["asset_class"] == "option_call"]
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(float(calls.iloc[0]["quantity"]), 0.0)
        self.assertAlmostEqual(float(calls.iloc[0]["cost_basis"]), 0.0)
        self.assertEqual(len(out[out["asset_class"] == "other"]), 0)
        cash_mv = float(out[out["asset_class"] == "cash"]["market_value"].sum())
        self.assertAlmostEqual(cash_mv, 1000.0)

    def test_occ_open_then_expiry_nets_to_nothing(self) -> None:
        # Fidelity: buy 2 SNDK calls in the window, then "EXPIRED" (other,
        # qty -2, amount 0). Net 0 contracts → no live option row; the
        # premium is the only cash impact.
        positions = pd.DataFrame([_pos("CASH", None, "cash", 5000.0, 5000.0)])
        buy = _txn("2026-08-06", "buy", "-SNDK260807C1800", None,
                   quantity=2, amount=-1811.32, price=9.0)
        exp = _txn("2026-08-10", "other", "-SNDK260807C1800", None,
                   quantity=-2, amount=0.0)
        out = synthesize_interim_positions(positions, pd.DataFrame([buy, exp]))
        opt = out[out["asset_class"].astype(str).str.startswith("option")]
        self.assertAlmostEqual(float(opt["quantity"].sum()), 0.0)
        self.assertAlmostEqual(float(opt["market_value"].sum()), 0.0)
        self.assertEqual(len(out[out["asset_class"] == "other"]), 0)
        cash_mv = float(out[out["asset_class"] == "cash"]["market_value"].sum())
        self.assertAlmostEqual(cash_mv, 5000.0 - 1811.32, places=2)

    def test_display_format_add_on_buy_pools_into_statement_row(self) -> None:
        # Buying more of a contract already on the statement adds to THAT
        # row (qty + cost basis) instead of booking a duplicate.
        positions = pd.DataFrame([self._jpm_put(650, 6, 4572.0, 6688.04)])
        interim = pd.DataFrame([self._jpm_leg("2026-08-12", "buy", 650, 2,
                                              -1100.00)])
        out = synthesize_interim_positions(positions, interim)
        puts = out[out["asset_class"] == "option_put"]
        self.assertEqual(len(puts), 1)
        self.assertAlmostEqual(float(puts.iloc[0]["quantity"]), 8.0)
        self.assertAlmostEqual(float(puts.iloc[0]["cost_basis"]), 6688.04 + 1100.0,
                               places=2)
        self.assertAlmostEqual(float(puts.iloc[0]["market_value"]),
                               4572.0 * 8 / 6, places=2)


class TestStalePreStatementActivity(unittest.TestCase):
    """The interim CSV is a rolling activity window that can reach back BEFORE
    an account's latest statement (e.g. a dividend-reinvestment on a position
    sold before the statement). Those rows are already reflected in the
    statement; re-applying them resurrects phantom holdings and double-counts
    cash. Only activity dated AFTER the account's base statement rolls forward.

    Real incident (Jun 2026): the Roth sold JEPQ before its May-31 statement,
    but the interim file still carried Apr/May JEPQ DRIP rows (blank settlement
    date, trade_date pre-statement). With no date guard they booked a ~2.4-share
    phantom JEPQ, which — being a non-SPY name (natural weight 0) — auto-flagged
    as "excess concentration" and surfaced in the Options-tab Hedge signals.
    """

    def test_pre_statement_reinvestment_does_not_resurrect_sold_symbol(self):
        positions = pd.DataFrame([
            _pos("MU", None, "equity_stock", quantity=24, market_value=23900.0,
                 account_id="ROTH", date="2026-05-31"),
            _pos("CASH", None, "cash", quantity=2.0, market_value=2.0,
                 account_id="ROTH", date="2026-05-31"),
        ])
        # Legit post-base buy (also sets the unified snapshot date) ...
        good = _txn("2026-06-04", "buy", "MU", None,
                    quantity=1, amount=-970.0, account_id="ROTH", price=970.0)
        # ... and the stale JEPQ DRIP: blank settlement, trade_date pre-base.
        drip = _txn("2026-06-05", "reinvestment", "JEPQ", None,
                    quantity=1.208, amount=-68.98, account_id="ROTH",
                    price=57.10)
        drip["settlement_date"] = pd.NaT
        drip["trade_date"] = pd.Timestamp("2026-05-05")
        interim = pd.DataFrame([good, drip])
        out = synthesize_interim_positions(positions, interim)
        self.assertEqual(
            len(out[out["symbol"] == "JEPQ"]), 0,
            "stale pre-statement DRIP resurrected a phantom JEPQ position")
        # The legit post-base buy still applied (MU 24 -> 25).
        mu = out[(out["account_id"] == "ROTH")
                 & (out["symbol"] == "MU")].iloc[0]
        self.assertEqual(mu["quantity"], 25)

    def test_pre_statement_cash_dividend_is_not_double_counted(self):
        positions = pd.DataFrame([
            _pos("AAPL", None, "equity_stock", quantity=100,
                 market_value=15000.0, account_id="ACCT", date="2026-05-31"),
            _pos("CASH", None, "cash", quantity=500.0, market_value=500.0,
                 account_id="ACCT", date="2026-05-31"),
        ])
        stale_div = _txn("2026-06-05", "dividend", None, None,
                         quantity=None, amount=120.0, account_id="ACCT")
        stale_div["settlement_date"] = pd.NaT
        stale_div["trade_date"] = pd.Timestamp("2026-05-20")  # pre-base
        post_div = _txn("2026-06-03", "dividend", None, None,
                        quantity=None, amount=80.0, account_id="ACCT")
        interim = pd.DataFrame([stale_div, post_div])
        out = synthesize_interim_positions(positions, interim)
        cash = out[(out["account_id"] == "ACCT")
                   & (out["asset_class"] == "cash")].iloc[0]
        # Only the post-base $80 lands: 500 + 80 = 580 (NOT 700).
        self.assertAlmostEqual(cash["market_value"], 580.0)

    def test_activity_with_no_determinable_date_is_kept(self):
        # If BOTH settlement_date and trade_date are blank we can't tell whether
        # the row predates the base — keep it rather than silently dropping real
        # activity. (Rare; most rows carry at least one date.)
        positions = pd.DataFrame([
            _pos("AAPL", None, "equity_stock", quantity=100,
                 market_value=15000.0, account_id="ACCT", date="2026-05-31"),
            _pos("CASH", None, "cash", quantity=500.0, market_value=500.0,
                 account_id="ACCT", date="2026-05-31"),
        ])
        dateless = _txn("2026-06-05", "buy", "AAPL", None,
                        quantity=3, amount=-450.0, account_id="ACCT",
                        price=150.0)
        dateless["settlement_date"] = pd.NaT
        dateless["trade_date"] = pd.NaT
        anchor = _txn("2026-06-02", "dividend", None, None, quantity=None,
                      amount=10.0, account_id="ACCT")  # sets snapshot date
        interim = pd.DataFrame([dateless, anchor])
        out = synthesize_interim_positions(positions, interim)
        aapl = out[(out["account_id"] == "ACCT")
                   & (out["symbol"] == "AAPL")].iloc[0]
        self.assertEqual(aapl["quantity"], 103)  # kept: 100 + 3


class TestWSD4ExchangeLeg(unittest.TestCase):
    """The FXAIX round-trip must net to 0 shares and +$14.18 cash, not a
    -12.335 phantom short with double-counted cash."""

    def _book(self):
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 100.0, 100.0, account_id="F1", date="2026-05-31"),
        ])
        sell = _txn("2026-06-05", "sell", "FXAIX", None, -12.335, 3088.68,
                    account_id="F1", price=250.40)
        buy = {
            "settlement_date": pd.NaT, "trade_date": pd.Timestamp("2026-06-03"),
            "broker": "fidelity", "account_id": "F1", "transaction_type": "other",
            "symbol": "FXAIX", "cusip": None, "description": "FIDELITY 500 INDEX FUND",
            "quantity": 12.335, "price": float("nan"), "amount": 3074.50,
            "source_file": "interim.csv",
        }
        return positions, pd.DataFrame([sell, buy])

    def test_exchange_pair_nets_to_zero_shares(self) -> None:
        out = synthesize_interim_positions(*self._book())
        fx = out[out["symbol"] == "FXAIX"]
        self.assertAlmostEqual(fx["quantity"].sum(), 0.0, places=6)
        self.assertAlmostEqual(fx["market_value"].sum(), 0.0, places=2)

    def test_exchange_pair_cash_nets_to_proceeds_minus_cost(self) -> None:
        out = synthesize_interim_positions(*self._book())
        cash = out[(out["account_id"] == "F1") & (out["asset_class"] == "cash")]
        self.assertAlmostEqual(cash["market_value"].sum(), 114.18, places=2)


class TestWSD4Guards(unittest.TestCase):
    """The exchange-leg path must not disturb the `other` rows it was carved
    out from: Carnival NaN-amount renames, internal cash journals."""

    def test_carnival_nan_amount_other_row_is_noop_on_qty(self) -> None:
        positions = pd.DataFrame([
            _pos("CCL", None, "equity", 100.0, 2000.0, account_id="F1", date="2026-05-31"),
        ])
        carnival = {
            "settlement_date": pd.Timestamp("2026-06-03"), "trade_date": None,
            "broker": "fidelity", "account_id": "F1", "transaction_type": "other",
            "symbol": "CCL", "cusip": None, "description": "CARNIVAL CORP",
            "quantity": 25.0, "price": float("nan"), "amount": float("nan"),
            "source_file": "interim.csv",
        }
        out = synthesize_interim_positions(positions, pd.DataFrame([carnival]))
        ccl = out[out["symbol"] == "CCL"]
        self.assertAlmostEqual(ccl["quantity"].sum(), 100.0, places=6)

    def test_internal_cash_journal_other_row_unchanged(self) -> None:
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 500.0, 500.0, account_id="F1", date="2026-05-31"),
        ])
        journal = {
            "settlement_date": pd.Timestamp("2026-06-03"), "trade_date": None,
            "broker": "fidelity", "account_id": "F1", "transaction_type": "other",
            "symbol": None, "cusip": None, "description": "JOURNALED CASH",
            "quantity": float("nan"), "price": float("nan"), "amount": -200.0,
            "source_file": "interim.csv",
        }
        out = synthesize_interim_positions(positions, pd.DataFrame([journal]))
        cash = out[out["asset_class"] == "cash"]
        self.assertAlmostEqual(cash["market_value"].sum(), 300.0, places=2)


class TestSecurityExchangeLegPredicate(unittest.TestCase):
    """A mis-typed Fidelity fund-exchange leg (`other` + qty + real amount +
    security key, not an option) must be recognized; cash journals and Carnival
    share-class renames must not be."""

    def _row(self, **kw) -> dict:
        base = {"transaction_type": "other", "symbol": "FXAIX", "cusip": None,
                "quantity": 12.335, "amount": 3074.50}
        base.update(kw)
        return base

    def test_fund_exchange_in_leg_is_true(self) -> None:
        self.assertTrue(_is_security_exchange_leg(self._row()))

    def test_non_other_type_is_false(self) -> None:
        self.assertFalse(_is_security_exchange_leg(self._row(transaction_type="buy")))

    def test_nan_amount_is_false(self) -> None:
        self.assertFalse(_is_security_exchange_leg(self._row(amount=float("nan"))))

    def test_zero_quantity_is_false(self) -> None:
        self.assertFalse(_is_security_exchange_leg(self._row(quantity=0)))

    def test_no_security_key_is_false(self) -> None:
        self.assertFalse(_is_security_exchange_leg(self._row(symbol=None, cusip=None)))

    def test_option_leg_is_false(self) -> None:
        self.assertFalse(_is_security_exchange_leg(self._row(symbol="-SPY261218P575")))

    def test_display_format_option_leg_is_false(self) -> None:
        # JPM display-format legs are option legs too — never an equity leg.
        self.assertFalse(_is_security_exchange_leg(
            self._row(symbol="SPY DEC 26 PUT 650.00")))

    def test_zero_amount_is_false(self) -> None:
        # "COLLATERAL DELV..." lending placeholders carry qty but $0 amount.
        self.assertFalse(_is_security_exchange_leg(self._row(amount=0.0)))

    def test_internal_flow_scope_is_false(self) -> None:
        self.assertFalse(_is_security_exchange_leg(self._row(flow_scope="internal")))


class TestWSD4BalancedRoundTrip(unittest.TestCase):
    """Balanced `other` round-trips (e.g. securities-lending in/out) should net
    to zero shares and zero cash impact — they must not create phantom positions
    or drain the cash balance."""

    def test_balanced_other_round_trip_nets_to_zero(self) -> None:
        # Securities-lending style: two `other` legs, same symbol, opposite qty,
        # signed amounts -> 0 shares and 0 cash (no phantom position or cash).
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 100.0, 100.0, account_id="F1", date="2026-05-31"),
        ])
        leg_in = {
            "settlement_date": pd.Timestamp("2026-06-03"), "trade_date": None,
            "broker": "fidelity", "account_id": "F1", "transaction_type": "other",
            "symbol": "DRAM", "cusip": None, "description": "ROUNDHILL MEMORY ETF",
            "quantity": 216.0, "price": float("nan"), "amount": 14191.20,
            "source_file": "interim.csv",
        }
        leg_out = dict(leg_in, quantity=-216.0, amount=-14191.20)
        out = synthesize_interim_positions(positions, pd.DataFrame([leg_in, leg_out]))
        dram = out[out["symbol"] == "DRAM"]
        self.assertAlmostEqual(dram["quantity"].sum(), 0.0, places=6)
        cash = out[out["asset_class"] == "cash"]
        self.assertAlmostEqual(cash["market_value"].sum(), 100.0, places=2)


class TestSupersededAccounts(unittest.TestCase):
    """A statement at/after the interim snapshot supersedes the roll-forward.

    2026-06 incident: a stale interim file (max settlement 2026-06-02) met
    freshly-ingested 2026-06-30 statements. The roller emitted a backdated
    copy of each June-30 book at June-02, so every month-sliced consumer
    (options composition, option table, Data Health extracted totals)
    double-counted those accounts."""

    def test_account_with_newer_statement_is_not_emitted(self) -> None:
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", 10, 5000.0,
                 account_id="NEW-1", date="2026-06-30"),
            _pos("VTI", None, "equity_etf", 4, 1200.0,
                 account_id="LAG-1", date="2026-05-31"),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-02", "buy", "VTI", None,
                 quantity=1, amount=-300.0, account_id="LAG-1", price=300.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        # Superseded account: no June-02 copy of its June-30 book.
        self.assertNotIn("NEW-1", set(out["account_id"]))
        # Lagging account still rolls forward.
        vti = out[(out["account_id"] == "LAG-1") & (out["symbol"] == "VTI")]
        self.assertEqual(len(vti), 1)
        self.assertEqual(vti.iloc[0]["quantity"], 5)

    def test_statement_on_snapshot_date_is_not_duplicated(self) -> None:
        # Equality counts as superseded: relabeling to the SAME date would
        # emit every row twice at that date.
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", 10, 5000.0,
                 account_id="EQ-1", date="2026-06-02"),
            _pos("VTI", None, "equity_etf", 4, 1200.0,
                 account_id="LAG-1", date="2026-05-31"),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-02", "buy", "VTI", None,
                 quantity=1, amount=-300.0, account_id="LAG-1", price=300.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        self.assertNotIn("EQ-1", set(out["account_id"]))

    def test_relabel_branch_respects_superseded(self) -> None:
        # A broker with NO interim rows takes the wholesale-relabel path;
        # it must skip superseded accounts too (JPM June book stays put
        # once its June statement lands) while still bridging lagging ones.
        positions = pd.DataFrame([
            dict(_pos("VOO", None, "equity_etf", 3, 900.0,
                      account_id="J-NEW", date="2026-06-30"), broker="jpm"),
            dict(_pos("BND", None, "equity_etf", 5, 500.0,
                      account_id="J-LAG", date="2026-05-31"), broker="jpm"),
            _pos("VTI", None, "equity_etf", 4, 1200.0,
                 account_id="F-LAG", date="2026-05-31"),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-02", "buy", "VTI", None,
                 quantity=1, amount=-300.0, account_id="F-LAG", price=300.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        self.assertNotIn("J-NEW", set(out["account_id"]))
        j_lag = out[out["account_id"] == "J-LAG"]
        self.assertEqual(len(j_lag), 1)
        self.assertEqual(pd.Timestamp(j_lag.iloc[0]["statement_date"]),
                         pd.Timestamp("2026-06-02"))

    def test_fully_stale_interim_returns_empty(self) -> None:
        # Every account superseded -> empty frame with the positions columns
        # (regression: pd.DataFrame([])[cols] used to raise KeyError).
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", 10, 5000.0,
                 account_id="NEW-1", date="2026-06-30"),
        ])
        interim = pd.DataFrame([
            _txn("2026-06-02", "buy", "SPY", None,
                 quantity=1, amount=-500.0, account_id="NEW-1", price=500.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), list(positions.columns))

    def test_undated_txn_for_superseded_account_spawns_nothing(self) -> None:
        # An undated interim row normally survives the stale-activity guard
        # ("don't drop genuinely new activity"); for a superseded account the
        # statement covers the whole interim window, so it must not spawn a
        # brand-new row either.
        positions = pd.DataFrame([
            _pos("SPY", None, "equity_etf", 10, 5000.0,
                 account_id="NEW-1", date="2026-06-30"),
            _pos("VTI", None, "equity_etf", 4, 1200.0,
                 account_id="LAG-1", date="2026-05-31"),
        ])
        undated = _txn("2026-06-02", "buy", "NVDA", None,
                       quantity=2, amount=-2000.0, account_id="NEW-1",
                       price=1000.0)
        undated["settlement_date"] = pd.NaT
        interim = pd.DataFrame([
            _txn("2026-06-02", "buy", "VTI", None,
                 quantity=1, amount=-300.0, account_id="LAG-1", price=300.0),
            undated,
        ])
        out = synthesize_interim_positions(positions, interim)
        self.assertNotIn("NEW-1", set(out["account_id"]))


class TestNewRowAssetClassInference(unittest.TestCase):
    """Brand-new interim rows infer a real asset class instead of the blanket
    "other" placeholder (2026-07: a batch of mid-month buys of new symbols
    rendered as a spurious "Other" allocation slice). Inference order: inherit
    from the book's own statement history by key; UST description ->
    fixed_income; plain ticker -> equity_stock. Display-format option legs and
    unknown bonds keep "other" (the former so the reclass_asset WSD-3 rescue
    still fires downstream)."""

    def test_new_symbol_inherits_most_recent_statement_class(self) -> None:
        # Re-entry: held long ago (equity_etf in 2023, equity_stock in 2025),
        # absent from the latest statement, re-bought mid-month. The most
        # recent history row's class wins.
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 2000.0, 2000.0),
            _pos("RNTR", None, "equity_etf", 20, 1000.0, date="2023-08-31"),
            _pos("RNTR", None, "equity_stock", 30, 1500.0, date="2025-01-31"),
        ])
        interim = pd.DataFrame([
            _txn("2026-05-15", "buy", "RNTR", None,
                 quantity=10, amount=-5000.0, price=500.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        rntr = out[out["symbol"] == "RNTR"]
        self.assertEqual(len(rntr), 1)
        self.assertEqual(rntr.iloc[0]["asset_class"], "equity_stock")

    def test_new_ust_cusip_classified_fixed_income(self) -> None:
        # A brand-new Treasury note (cusip-keyed, no symbol) is recognizable
        # from its description alone.
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 25000.0, 25000.0),
        ])
        buy = _txn("2026-05-15", "buy", None, "91282CNEW",
                   quantity=10000, amount=-9938.00, price=99.38)
        buy["description"] = ("UNITED STATES TREASURY NOTE DUE 08/15/2030 "
                              "04.250%FA 15")
        interim = pd.DataFrame([buy])
        out = synthesize_interim_positions(positions, interim)
        ust = out[out["cusip"] == "91282CNEW"]
        self.assertEqual(len(ust), 1)
        self.assertEqual(ust.iloc[0]["asset_class"], "fixed_income")

    def test_new_plain_ticker_defaults_to_equity_stock(self) -> None:
        # Brand-new ticker with no history signal: default to equity_stock;
        # the display reclass layer still maps known ETFs and commodity
        # tickers on top of this.
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 80000.0, 80000.0),
        ])
        interim = pd.DataFrame([
            _txn("2026-05-15", "buy", "NEWCO", None,
                 quantity=100, amount=-6500.0, price=65.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        newco = out[out["symbol"] == "NEWCO"]
        self.assertEqual(len(newco), 1)
        self.assertEqual(newco.iloc[0]["asset_class"], "equity_stock")

    def test_option_history_is_not_inherited_for_bare_symbol(self) -> None:
        # Statement option rows carry the bare underlying as symbol (SPY,
        # option_put). A bare-symbol SPY buy in another account is the
        # underlying, never the derivative — option classes must not inherit.
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 5000.0, 5000.0),
            _pos("SPY", None, "option_put", -4, -800.0, account_id="TEST-2"),
        ])
        interim = pd.DataFrame([
            _txn("2026-05-15", "buy", "SPY", None,
                 quantity=2, amount=-1100.0, price=550.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        new_spy = out[(out["account_id"] == "TEST-1") & (out["symbol"] == "SPY")]
        self.assertEqual(len(new_spy), 1)
        self.assertEqual(new_spy.iloc[0]["asset_class"], "equity_stock")

    def test_other_history_is_not_inherited(self) -> None:
        # JPM statements tag commodity tickers "other" — no information to
        # inherit. The synthesizer defaults to equity_stock and the display
        # layer's commodity map (GLD -> gold) corrects it downstream.
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 5000.0, 5000.0),
            _pos("GLD", None, "other", 10, 3000.0, date="2024-06-30"),
        ])
        interim = pd.DataFrame([
            _txn("2026-05-15", "buy", "GLD", None,
                 quantity=5, amount=-1500.0, price=300.0),
        ])
        out = synthesize_interim_positions(positions, interim)
        gld = out[out["symbol"] == "GLD"]
        self.assertEqual(len(gld), 1)
        self.assertEqual(gld.iloc[0]["asset_class"], "equity_stock")

    def test_display_format_option_leg_books_option_row(self) -> None:
        # Interim JPM option legs arrive in DISPLAY format ("SPY DEC 26 PUT
        # 650.00"). They used to stay an "other" placeholder (rescued into
        # option_put at render time, WSD-3) — which also meant they never
        # netted against the statement's option rows (the Aug 2026 phantom
        # puts). They now take the option route: a real option_put row keyed
        # on the underlying, the same shape the OCC route produces.
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 5000.0, 5000.0),
        ])
        buy = _txn("2026-05-15", "buy", "SPY DEC 26 PUT 650.00", None,
                   quantity=4, amount=-800.0, price=2.0)
        buy["description"] = ("PUT SPY 12/18/26 650 STATE STREET SPDR S&P 500 "
                              "ETF UNSOLICITED OPEN CONTRACT")
        out = synthesize_interim_positions(positions, pd.DataFrame([buy]))
        self.assertEqual(len(out[out["asset_class"] == "other"]), 0)
        self.assertFalse((out["symbol"] == "SPY DEC 26 PUT 650.00").any())
        leg = out[out["asset_class"] == "option_put"]
        self.assertEqual(len(leg), 1)
        self.assertEqual(leg.iloc[0]["symbol"], "SPY")
        self.assertAlmostEqual(float(leg.iloc[0]["quantity"]), 4.0)
        self.assertAlmostEqual(float(leg.iloc[0]["market_value"]), 800.0)
        self.assertAlmostEqual(float(leg.iloc[0]["cost_basis"]), 800.0)

    def test_unrecognized_cusip_only_row_stays_other(self) -> None:
        # A cusip-keyed buy that is not recognizably a Treasury (e.g. a
        # corporate bond) has no safe default — keep the honest "other".
        positions = pd.DataFrame([
            _pos("CASH", None, "cash", 5000.0, 5000.0),
        ])
        buy = _txn("2026-05-15", "buy", None, "X99999AA1",
                   quantity=5000, amount=-4980.0, price=99.6)
        buy["description"] = "ACME CORP 5.500% NOTES DUE 2030"
        interim = pd.DataFrame([buy])
        out = synthesize_interim_positions(positions, interim)
        bond = out[out["cusip"] == "X99999AA1"]
        self.assertEqual(len(bond), 1)
        self.assertEqual(bond.iloc[0]["asset_class"], "other")


class TestCorporateOutLegPredicate(unittest.TestCase):
    """A merger / redemption row joins the quantity path ONLY in the cash-out
    shape (shares leave, cash arrives). Any other shape stays inert."""

    def _row(self, **kw) -> dict:
        base = {"transaction_type": "merger", "symbol": None,
                "cusip": "X99999AA1", "quantity": -2.0, "amount": 100.0}
        base.update(kw)
        return base

    def test_cash_merger_is_true(self) -> None:
        self.assertTrue(_is_corporate_out_leg(self._row()))

    def test_redemption_is_true(self) -> None:
        self.assertTrue(_is_corporate_out_leg(
            self._row(transaction_type="redemption", cusip="912828X88",
                      quantity=-25000.0, amount=25000.0)))

    def test_share_receiving_merger_leg_is_false(self) -> None:
        # Shares arriving (quantity > 0) with cash alongside — the shape of a
        # share-for-share merger leg with cash in lieu — is not an out-leg.
        self.assertFalse(_is_corporate_out_leg(self._row(quantity=3.0)))
        self.assertFalse(_is_corporate_out_leg(
            self._row(quantity=3.0, amount=float("nan"))))

    def test_missing_amount_is_false(self) -> None:
        self.assertFalse(_is_corporate_out_leg(self._row(amount=float("nan"))))
        self.assertFalse(_is_corporate_out_leg(self._row(amount=0.0)))

    def test_other_types_are_false(self) -> None:
        self.assertFalse(_is_corporate_out_leg(self._row(transaction_type="sell")))
        self.assertFalse(_is_corporate_out_leg(self._row(transaction_type="other")))
        self.assertFalse(_is_corporate_out_leg(self._row(transaction_type="exchange")))


class TestRescueByName(unittest.TestCase):
    """The lot engine's corporate-action name rule (spec 2026-07-30 §3.3)
    ported to statement rows: strict unique leading-token run >= 2, or one
    token >= 6 chars; qty-sane; symbol-keyed, non-option, non-cash rows only.
    Verdict contract (spec Update 2026-08-23): ("match"|"claimed"|
    "blocker"|"refuse", payload) — "claimed" and "blocker" carry the
    target for the caller's emitted-row quantity gate."""

    MERGER_DESC = "ACME WIDGETS INC A123456 TO A123456 CMR $50P/S"

    def _rec(self, symbol, description, quantity=2.0, cusip=None,
             asset_class="equity_stock", account_id="J1") -> dict:
        return {"account_id": account_id, "symbol": symbol, "cusip": cusip,
                "description": description, "asset_class": asset_class,
                "quantity": quantity}

    def test_unique_leading_run_matches(self) -> None:
        recs = [self._rec("ZZZ", "ZETA HOLDINGS"),
                self._rec("ACME", "ACME WIDGETS INC")]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("match", 1))

    def test_tie_refuses(self) -> None:
        recs = [self._rec("ACME", "ACME WIDGETS INC"),
                self._rec("ACMB", "ACME WIDGETS INC CL B")]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))

    def test_qty_above_holding_refuses(self) -> None:
        recs = [self._rec("ACME", "ACME WIDGETS INC", quantity=1.0)]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))

    def test_option_and_cash_rows_never_match(self) -> None:
        recs = [self._rec("ACME", "ACME WIDGETS INC PUT 12/18/26 40",
                          asset_class="option_put"),
                self._rec("CASH", "ACME WIDGETS INC", asset_class="cash",
                          quantity=5000.0)]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))

    def test_cusip_keyed_rows_never_match(self) -> None:
        # Two different cusips are two different instruments — the rescue
        # bridges symbol<->cusip identifier drift only.
        recs = [self._rec(None, "ACME WIDGETS INC 5.5% NOTES DUE 2030",
                          cusip="Y11111AA1", asset_class="fixed_income")]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))

    def test_single_long_issuer_token_matches(self) -> None:
        recs = [self._rec("ZV", "ZETAVOLT", quantity=3.0)]
        self.assertEqual(_rescue_by_name(
            recs, "J1", "ZETAVOLT Z163682 CMR $80P/S", 3.0, set()),
            ("match", 0))

    def test_single_short_token_refuses(self) -> None:
        recs = [self._rec("ACME", "ACME CORP")]
        self.assertEqual(_rescue_by_name(
            recs, "J1", "ACME HOLDINGS LTD A1 CMR $10P/S", 2.0, set()),
            ("refuse", None))

    def test_claimed_row_verdict_and_other_account_rows_skipped(self) -> None:
        # A claimed unique winner is no longer a flat refusal: the verdict
        # names it so the caller can accumulate onto the EMITTED row
        # (quantity gated there, not here — the base qty is stale).
        recs = [self._rec("ACME", "ACME WIDGETS INC"),
                self._rec("ACME", "ACME WIDGETS INC", account_id="J2")]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, {0}),
                         ("claimed", 0))
        self.assertEqual(_rescue_by_name(recs, "J2", self.MERGER_DESC, 2.0, set()),
                         ("match", 1))

    def test_blank_description_refuses(self) -> None:
        recs = [self._rec("ACME", "ACME WIDGETS INC")]
        self.assertEqual(_rescue_by_name(recs, "J1", None, 2.0, set()),
                         ("refuse", None))
        self.assertEqual(_rescue_by_name(recs, "J1", float("nan"), 2.0, set()),
                         ("refuse", None))

    def test_qty_short_winner_refuses_rather_than_promoting_runner_up(self) -> None:
        # The true target (run 3) cannot supply the shares; the runner-up
        # (run 2) must NOT be promoted — the quantity check vetoes the
        # winner, it does not re-run the contest without it.
        recs = [self._rec("ACME", "ACME WIDGETS INC", quantity=1.0),
                self._rec("ACMH", "ACME WIDGETS HOLDINGS", quantity=5.0)]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))
        recs[0]["quantity"] = 5.0
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("match", 0))

    def test_higher_run_beats_lower_run(self) -> None:
        recs = [self._rec("ACMC", "ACME CORP"),              # run 1
                self._rec("ACME", "ACME WIDGETS INC")]       # run 3
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("match", 1))

    def test_nan_quantity_winner_is_vetoed(self) -> None:
        # A NaN quantity coerces to 0 held: the row still wins the contest
        # and is then vetoed (never a silent close, never a promotion).
        recs = [self._rec("ACME", "ACME WIDGETS INC", quantity=float("nan"))]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))

    def test_keyless_record_is_skipped(self) -> None:
        recs = [self._rec(None, "ACME WIDGETS INC", cusip=None)]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 2.0, set()),
                         ("refuse", None))

    def test_claimed_winner_named_never_the_runner_up(self) -> None:
        # The true target (run 3) is already claimed; the runner-up (run 2)
        # must NOT be promoted — the verdict names the claimed winner for
        # the caller's emitted-row gate.
        recs = [self._rec("ACME", "ACME WIDGETS INC", quantity=10.0),
                self._rec("ACMH", "ACME WIDGETS HOLDINGS", quantity=5.0)]
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 3.0, {0}),
                         ("claimed", 0))
        self.assertEqual(_rescue_by_name(recs, "J1", self.MERGER_DESC, 3.0, set()),
                         ("match", 0))

    def test_in_window_position_blocking_names_the_blocker(self) -> None:
        # The statement holds class A; class C was bought inside the window
        # (no statement row) and is the security actually being merged. A
        # STRICTLY better unique blocker is named for the caller's netting;
        # unrelated blockers do not interfere.
        recs = [self._rec("ZETA", "ZETA HOLDINGS INC CL A", quantity=10.0)]
        leg = "ZETA HOLDINGS INC CL C Z1 CMR $50P/S"
        self.assertEqual(_rescue_by_name(
            recs, "J1", leg, 5.0, set(),
            blockers=["ZETA HOLDINGS INC CL C"]),
            ("blocker", "ZETA HOLDINGS INC CL C"))
        self.assertEqual(_rescue_by_name(
            recs, "J1", leg, 5.0, set(), blockers=["UNRELATED NAME"]),
            ("match", 0))
        self.assertEqual(_rescue_by_name(recs, "J1", leg, 5.0, set()),
                         ("match", 0))

    def test_blocker_tie_with_statement_winner_refuses(self) -> None:
        # A blocker matching only AS WELL as the statement winner is not
        # decisive about which holding was merged — refuse (the pre-Update
        # veto semantics for ties survive).
        recs = [self._rec("ZETA", "ZETA HOLDINGS INC", quantity=10.0)]
        leg = "ZETA HOLDINGS INC Z1 CMR $50P/S"
        self.assertEqual(_rescue_by_name(
            recs, "J1", leg, 5.0, set(), blockers=["ZETA HOLDINGS INC"]),
            ("refuse", None))

    def test_two_strictly_better_blockers_refuse(self) -> None:
        # Two in-window positions both out-name the statement winner: no
        # way to pick — refuse, never guess.
        recs = [self._rec("ZETA", "ZETA HOLDINGS INC CL A", quantity=10.0)]
        leg = "ZETA HOLDINGS INC CL C Z1 CMR $50P/S"
        self.assertEqual(_rescue_by_name(
            recs, "J1", leg, 5.0, set(),
            blockers=["ZETA HOLDINGS INC CL C",
                      "ZETA HOLDINGS INC CL C SERIES 2"]),
            ("refuse", None))

    def test_claimed_winner_still_yields_to_a_strictly_better_blocker(self) -> None:
        # Claimed AND out-named: the stronger name wins — the verdict is
        # the blocker (the merged shares are more likely the fresh ones).
        recs = [self._rec("ZETA", "ZETA HOLDINGS INC CL A", quantity=10.0)]
        leg = "ZETA HOLDINGS INC CL C Z1 CMR $50P/S"
        self.assertEqual(_rescue_by_name(
            recs, "J1", leg, 5.0, {0},
            blockers=["ZETA HOLDINGS INC CL C"]),
            ("blocker", "ZETA HOLDINGS INC CL C"))


class TestCorporateActionOutLegs(unittest.TestCase):
    """A cash-merger / redemption out-leg closes the statement row it names.
    The EA take-private (Aug 2026) arrived cusip-keyed while the statement
    keyed EA by symbol: the roll booked a fresh NEGATIVE `other` row beside
    the carried-forward holding. Now the leg re-keys by security name when
    the exact key finds nothing; unmatched legs keep today's visible
    fresh-row artifact (it nets in NAV — a silent skip would double count)."""

    MERGER_DESC = "ACME WIDGETS INC A123456 TO A123456 CMR $50P/S"

    def _book(self, acme_qty: float = 2.0) -> pd.DataFrame:
        acme = _pos("ACME", None, "equity_stock", acme_qty, 50.0 * acme_qty,
                    account_id="J1", date="2026-07-31")
        acme["description"] = "ACME WIDGETS INC"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        return pd.DataFrame([acme, cash])

    def _leg(self, ttype: str = "merger", **kw) -> dict:
        row = _txn("2026-08-05", ttype, None, "X99999AA1",
                   quantity=-2.0, amount=100.0, account_id="J1")
        row["description"] = self.MERGER_DESC
        row.update(kw)
        return row

    def test_cash_merger_closes_the_symbol_keyed_statement_row(self) -> None:
        out = synthesize_interim_positions(self._book(), pd.DataFrame([self._leg()]))
        acme = out[out["symbol"] == "ACME"]
        self.assertEqual(len(acme), 1)
        self.assertAlmostEqual(acme.iloc[0]["quantity"], 0.0)
        self.assertAlmostEqual(acme.iloc[0]["market_value"], 0.0)
        self.assertFalse((out["cusip"] == "X99999AA1").any())   # no phantom row
        cash = out[out["asset_class"] == "cash"]
        self.assertAlmostEqual(cash["market_value"].sum(), 1100.0, places=2)
        self.assertEqual(len(out), 2)

    def test_on_disk_other_shape_closes_too(self) -> None:
        # Until the export is re-pulled under the repaired map the row on
        # disk is typed `other` (the security-exchange-leg path).
        out = synthesize_interim_positions(
            self._book(), pd.DataFrame([self._leg(ttype="other")]))
        acme = out[out["symbol"] == "ACME"].iloc[0]
        self.assertAlmostEqual(acme["quantity"], 0.0)
        self.assertFalse((out["cusip"] == "X99999AA1").any())
        self.assertEqual(len(out), 2)

    def test_ambiguous_name_keeps_todays_fresh_row(self) -> None:
        book = self._book()
        twin = _pos("ACMB", None, "equity_stock", 2.0, 100.0,
                    account_id="J1", date="2026-07-31")
        twin["description"] = "ACME WIDGETS INC CL B"
        book = pd.concat([book, pd.DataFrame([twin])], ignore_index=True)
        out = synthesize_interim_positions(book, pd.DataFrame([self._leg()]))
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 2.0)
        self.assertAlmostEqual(out[out["symbol"] == "ACMB"].iloc[0]["quantity"], 2.0)
        phantom = out[out["cusip"] == "X99999AA1"]
        self.assertEqual(len(phantom), 1)
        self.assertAlmostEqual(phantom.iloc[0]["quantity"], -2.0)

    def test_qty_above_holding_keeps_todays_fresh_row(self) -> None:
        out = synthesize_interim_positions(
            self._book(acme_qty=1.0), pd.DataFrame([self._leg()]))
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 1.0)
        self.assertEqual(len(out[out["cusip"] == "X99999AA1"]), 1)

    def test_rescue_never_targets_option_or_cash_rows(self) -> None:
        put = _pos("ACME", None, "option_put", 1.0, 100.0,
                   account_id="J1", date="2026-07-31")
        put["description"] = "ACME WIDGETS INC PUT 12/18/26 40"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        cash["description"] = "ACME WIDGETS INC"
        out = synthesize_interim_positions(
            pd.DataFrame([put, cash]), pd.DataFrame([self._leg()]))
        self.assertAlmostEqual(
            out[out["asset_class"] == "option_put"].iloc[0]["quantity"], 1.0)
        self.assertEqual(len(out[out["cusip"] == "X99999AA1"]), 1)
        self.assertAlmostEqual(
            out[out["asset_class"] == "cash"].iloc[0]["market_value"], 1100.0, places=2)

    def test_exact_cusip_key_wins_over_a_name_match(self) -> None:
        book = self._book()
        bond = _pos(None, "X99999AA1", "fixed_income", 2.0, 200.0,
                    account_id="J1", date="2026-07-31")
        bond["description"] = "ACME WIDGETS INC 5.5% NOTES DUE 2030"
        book = pd.concat([book, pd.DataFrame([bond])], ignore_index=True)
        leg = self._leg(ttype="redemption", amount=200.0)
        leg["description"] = "ACME WIDGETS INC 5.5% NOTES REDEMPTION"
        out = synthesize_interim_positions(book, pd.DataFrame([leg]))
        self.assertAlmostEqual(out[out["cusip"] == "X99999AA1"].iloc[0]["quantity"], 0.0)
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 2.0)
        self.assertEqual(len(out), 3)

    def test_redemption_closes_the_bond_and_lands_its_face(self) -> None:
        # Today a row typed `redemption` is not a quantity type: the bond is
        # carried forward AND the face lands in cash — a double count.
        bond = _pos(None, "912828X88", "fixed_income", 25000.0, 24950.0,
                    account_id="J1", date="2026-07-31")
        bond["description"] = "UNITED STATES TREAS BILLS DUE 08/05/2026"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        leg = _txn("2026-08-05", "redemption", None, "912828X88",
                   quantity=-25000.0, amount=25000.0, account_id="J1")
        leg["description"] = "UNITED STATES TREASURY BILL REDEMPTION"
        out = synthesize_interim_positions(
            pd.DataFrame([bond, cash]), pd.DataFrame([leg]))
        self.assertAlmostEqual(out[out["cusip"] == "912828X88"].iloc[0]["quantity"], 0.0)
        self.assertAlmostEqual(
            out[out["asset_class"] == "cash"].iloc[0]["market_value"], 26000.0, places=2)
        self.assertEqual(len(out), 2)

    def test_merger_without_the_cash_shape_is_quantity_inert(self) -> None:
        leg = _txn("2026-08-05", "merger", "NEWCO", None,
                   quantity=3.0, amount=float("nan"), account_id="J1")
        leg["description"] = "NEWCO HOLDINGS N1 FROM A123456 SMR @1.5"
        out = synthesize_interim_positions(self._book(), pd.DataFrame([leg]))
        self.assertFalse((out["symbol"] == "NEWCO").any())
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 2.0)
        self.assertAlmostEqual(
            out[out["asset_class"] == "cash"].iloc[0]["market_value"], 1000.0, places=2)

    def test_plain_sell_with_an_unmatched_cusip_is_not_rescued(self) -> None:
        # A sell's printed key is authoritative (lot-engine precedent).
        out = synthesize_interim_positions(
            self._book(), pd.DataFrame([self._leg(ttype="sell")]))
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 2.0)
        self.assertEqual(len(out[out["cusip"] == "X99999AA1"]), 1)

    def test_single_issuer_token_rescues_only_when_long(self) -> None:
        k = _pos("ZV", None, "equity_stock", 3.0, 250.5,
                 account_id="J1", date="2026-07-31")
        k["description"] = "ZETAVOLT"
        acme = _pos("ACME", None, "equity_stock", 2.0, 100.0,
                    account_id="J1", date="2026-07-31")
        acme["description"] = "ACME CORP"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        k_leg = _txn("2026-08-05", "merger", None, "Z1234567X",
                     quantity=-3.0, amount=250.5, account_id="J1")
        k_leg["description"] = "ZETAVOLT Z163682 CMR $80P/S"
        short_leg = _txn("2026-08-05", "merger", None, "A9999999X",
                         quantity=-2.0, amount=20.0, account_id="J1")
        short_leg["description"] = "ACME HOLDINGS LTD A1 CMR $10P/S"
        out = synthesize_interim_positions(
            pd.DataFrame([k, acme, cash]), pd.DataFrame([k_leg, short_leg]))
        self.assertAlmostEqual(out[out["symbol"] == "ZV"].iloc[0]["quantity"], 0.0)
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 2.0)
        self.assertEqual(len(out[out["cusip"] == "A9999999X"]), 1)
        self.assertFalse((out["cusip"] == "Z1234567X").any())

    def test_same_window_symbol_trade_accumulates_never_duplicates(self) -> None:
        # A symbol-keyed partial sell and the cusip-keyed cash merger of the
        # same security in one window: exact-key deltas apply first, the
        # holding is emitted once whatever the interim row order, and the
        # merger now ACCUMULATES onto the already-emitted row (spec Update
        # 2026-08-23 B) — the emitted 8 shares supply the leg's 8 exactly,
        # so the position closes and no fresh row is booked.
        acme = _pos("ACME", None, "equity_stock", 10.0, 500.0,
                    account_id="J1", date="2026-07-31")
        acme["description"] = "ACME WIDGETS INC"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        sell = _txn("2026-08-03", "sell", "ACME", None,
                    quantity=-2.0, amount=100.0, account_id="J1")
        merger = self._leg(quantity=-8.0, amount=400.0)
        for order in ((merger, sell), (sell, merger)):
            out = synthesize_interim_positions(
                pd.DataFrame([acme, cash]), pd.DataFrame(list(order)))
            label = order[0]["transaction_type"] + " first"
            rows = out[out["symbol"] == "ACME"]
            self.assertEqual(len(rows), 1, label)
            self.assertAlmostEqual(rows.iloc[0]["quantity"], 0.0, msg=label)
            self.assertFalse((out["cusip"] == "X99999AA1").any(), label)
            self.assertAlmostEqual(
                out[out["asset_class"] == "cash"].iloc[0]["market_value"],
                1500.0, places=2, msg=label)
            self.assertAlmostEqual(out["market_value"].sum(), 1500.0,
                                   places=2, msg=label)

    def test_share_short_emitted_row_keeps_todays_fresh_row(self) -> None:
        # The emitted row cannot supply the leg's shares after the exact-key
        # sell (10 - 9 = 1 < 8): the accumulate gate refuses and today's
        # NAV-neutral fresh row survives (spec Update 2026-08-23 B).
        acme = _pos("ACME", None, "equity_stock", 10.0, 500.0,
                    account_id="J1", date="2026-07-31")
        acme["description"] = "ACME WIDGETS INC"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        sell = _txn("2026-08-03", "sell", "ACME", None,
                    quantity=-9.0, amount=450.0, account_id="J1")
        merger = self._leg(quantity=-8.0, amount=400.0)
        out = synthesize_interim_positions(
            pd.DataFrame([acme, cash]), pd.DataFrame([sell, merger]))
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 1.0)
        phantom = out[out["cusip"] == "X99999AA1"]
        self.assertEqual(len(phantom), 1)
        self.assertAlmostEqual(phantom.iloc[0]["quantity"], -8.0)

    def test_rescue_matches_on_the_out_legs_own_description(self) -> None:
        # Under one cusip key an `other` in-leg settling LATER carries an
        # unrelated description; the rescue must read the out-leg's own.
        in_leg = _txn("2026-08-06", "other", None, "X99999AA1",
                      quantity=1.0, amount=-50.0, account_id="J1")
        in_leg["description"] = "UNRELATED NAME LEG"
        out_leg = self._leg(quantity=-3.0, amount=150.0)
        out = synthesize_interim_positions(
            self._book(), pd.DataFrame([out_leg, in_leg]))
        acme = out[out["symbol"] == "ACME"].iloc[0]
        self.assertAlmostEqual(acme["quantity"], 0.0)
        self.assertFalse((out["cusip"] == "X99999AA1").any())
        self.assertAlmostEqual(
            out[out["asset_class"] == "cash"].iloc[0]["market_value"],
            1100.0, places=2)

    def test_claimed_target_accumulates_never_the_sibling(self) -> None:
        # A same-window symbol sell claims ACME first; the cusip-keyed merger
        # naming ACME now applies ON TOP of the emitted ACME row (spec
        # Update 2026-08-23 B) — the same-issuer sibling ACMH is still
        # never touched and no fresh row is booked.
        acme = _pos("ACME", None, "equity_stock", 10.0, 500.0,
                    account_id="J1", date="2026-07-31")
        acme["description"] = "ACME WIDGETS INC"
        acmh = _pos("ACMH", None, "equity_stock", 5.0, 100.0,
                    account_id="J1", date="2026-07-31")
        acmh["description"] = "ACME WIDGETS HOLDINGS"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        sell = _txn("2026-08-03", "sell", "ACME", None,
                    quantity=-2.0, amount=100.0, account_id="J1")
        merger = self._leg(quantity=-3.0, amount=150.0)
        out = synthesize_interim_positions(
            pd.DataFrame([acme, acmh, cash]), pd.DataFrame([sell, merger]))
        self.assertAlmostEqual(out[out["symbol"] == "ACMH"].iloc[0]["quantity"], 5.0)
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 5.0)
        self.assertFalse((out["cusip"] == "X99999AA1").any())
        self.assertAlmostEqual(
            out[out["asset_class"] == "cash"].iloc[0]["market_value"],
            1250.0, places=2)
        self.assertAlmostEqual(out["market_value"].sum(), 1600.0, places=2)

    def test_qty_short_target_never_closes_a_sibling_position(self) -> None:
        # The statement row the merger names holds fewer shares than the leg
        # removes; the same-issuer sibling with enough shares must not be
        # closed instead — the leg keeps the NAV-neutral fresh row.
        acme = _pos("ACME", None, "equity_stock", 2.0, 100.0,
                    account_id="J1", date="2026-07-31")
        acme["description"] = "ACME WIDGETS INC"
        acmh = _pos("ACMH", None, "equity_stock", 5.0, 100.0,
                    account_id="J1", date="2026-07-31")
        acmh["description"] = "ACME WIDGETS HOLDINGS"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        merger = self._leg(quantity=-3.0, amount=150.0)
        out = synthesize_interim_positions(
            pd.DataFrame([acme, acmh, cash]), pd.DataFrame([merger]))
        self.assertAlmostEqual(out[out["symbol"] == "ACMH"].iloc[0]["quantity"], 5.0)
        self.assertAlmostEqual(out[out["symbol"] == "ACME"].iloc[0]["quantity"], 2.0)
        phantom = out[out["cusip"] == "X99999AA1"]
        self.assertEqual(len(phantom), 1)
        self.assertAlmostEqual(phantom.iloc[0]["quantity"], -3.0)
        self.assertAlmostEqual(out["market_value"].sum(), 1200.0, places=2)

    def test_in_window_buy_of_the_merged_class_nets_never_the_sibling(self) -> None:
        # Class C is bought inside the window (no statement row) and then
        # cash-merged under a cusip; the out-leg now NETS into that fresh
        # class-C position (spec Update 2026-08-23 A — its name is strictly
        # better than the statement's class A), whatever the interim row
        # order. The statement's class A is still never touched.
        zeta = _pos("ZETA", None, "equity_stock", 10.0, 500.0,
                    account_id="J1", date="2026-07-31")
        zeta["description"] = "ZETA HOLDINGS INC CL A"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        buy = _txn("2026-08-03", "buy", "ZETAC", None,
                   quantity=5.0, amount=-250.0, account_id="J1", price=50.0)
        buy["description"] = "ZETA HOLDINGS INC CL C"
        merger = _txn("2026-08-10", "merger", None, "Z9999999X",
                      quantity=-5.0, amount=250.0, account_id="J1")
        merger["description"] = "ZETA HOLDINGS INC CL C Z1 CMR $50P/S"
        for order in ((buy, merger), (merger, buy)):
            out = synthesize_interim_positions(
                pd.DataFrame([zeta, cash]), pd.DataFrame(list(order)))
            label = order[0]["transaction_type"] + " first"
            self.assertAlmostEqual(
                out[out["symbol"] == "ZETA"].iloc[0]["quantity"], 10.0, msg=label)
            self.assertAlmostEqual(
                out[out["symbol"] == "ZETAC"].iloc[0]["quantity"], 0.0, msg=label)
            self.assertFalse((out["cusip"] == "Z9999999X").any(), label)
            self.assertAlmostEqual(
                out[out["asset_class"] == "cash"].iloc[0]["market_value"],
                1000.0, places=2, msg=label)
            self.assertAlmostEqual(out["market_value"].sum(), 1500.0,
                                   places=2, msg=label)

    def test_blocker_net_share_short_keeps_todays_fresh_row(self) -> None:
        # The fresh class-C position holds fewer shares than the leg
        # removes: the netting gate refuses and today's fresh-row artifact
        # survives; neither position is touched (spec Update 2026-08-23 A).
        zeta = _pos("ZETA", None, "equity_stock", 10.0, 500.0,
                    account_id="J1", date="2026-07-31")
        zeta["description"] = "ZETA HOLDINGS INC CL A"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        buy = _txn("2026-08-03", "buy", "ZETAC", None,
                   quantity=3.0, amount=-150.0, account_id="J1", price=50.0)
        buy["description"] = "ZETA HOLDINGS INC CL C"
        merger = _txn("2026-08-10", "merger", None, "Z9999999X",
                      quantity=-5.0, amount=250.0, account_id="J1")
        merger["description"] = "ZETA HOLDINGS INC CL C Z1 CMR $50P/S"
        out = synthesize_interim_positions(
            pd.DataFrame([zeta, cash]), pd.DataFrame([buy, merger]))
        self.assertAlmostEqual(out[out["symbol"] == "ZETA"].iloc[0]["quantity"], 10.0)
        self.assertAlmostEqual(out[out["symbol"] == "ZETAC"].iloc[0]["quantity"], 3.0)
        phantom = out[out["cusip"] == "Z9999999X"]
        self.assertEqual(len(phantom), 1)
        self.assertAlmostEqual(phantom.iloc[0]["quantity"], -5.0)

    def test_blocker_tie_keeps_todays_fresh_row_and_both_positions(self) -> None:
        # The fresh buy's description only TIES the statement winner's run:
        # which holding was merged is not decisive — refuse; both positions
        # survive untouched beside the visible fresh row.
        zeta = _pos("ZETA", None, "equity_stock", 10.0, 500.0,
                    account_id="J1", date="2026-07-31")
        zeta["description"] = "ZETA HOLDINGS INC"
        cash = _pos("CASH", None, "cash", 1000.0, 1000.0,
                    account_id="J1", date="2026-07-31")
        buy = _txn("2026-08-03", "buy", "ZETAB", None,
                   quantity=5.0, amount=-250.0, account_id="J1", price=50.0)
        buy["description"] = "ZETA HOLDINGS INC"
        merger = _txn("2026-08-10", "merger", None, "Z9999999X",
                      quantity=-5.0, amount=250.0, account_id="J1")
        merger["description"] = "ZETA HOLDINGS INC Z1 CMR $50P/S"
        out = synthesize_interim_positions(
            pd.DataFrame([zeta, cash]), pd.DataFrame([buy, merger]))
        self.assertAlmostEqual(out[out["symbol"] == "ZETA"].iloc[0]["quantity"], 10.0)
        self.assertAlmostEqual(out[out["symbol"] == "ZETAB"].iloc[0]["quantity"], 5.0)
        self.assertEqual(len(out[out["cusip"] == "Z9999999X"]), 1)


if __name__ == "__main__":
    unittest.main()
