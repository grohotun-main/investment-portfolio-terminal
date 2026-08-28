"""Tests for parsers/protection_cost.py.

Locks down: identity (paid - recv - mv == cost), snapshot anchor handling,
history_start rebase + anchor row, Harbor/Alpine month-end-quirk aggregation,
empty inputs.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from protection_cost import (  # noqa: E402
    build_protection_cost_timeline,
)


def _mk_txn(rows):
    """Build a transactions DataFrame from terse tuples.

    Each row: (date, broker, amount, description). Other columns are
    filled to match real CSV shape but aren't used by the function.
    """
    return pd.DataFrame([
        {"settlement_date": pd.Timestamp(d), "broker": b,
         "amount": a, "description": desc,
         "account_id": "ACC", "symbol": "SPY", "transaction_type": "buy",
         "quantity": 1, "trade_date": pd.Timestamp(d)}
        for d, b, a, desc in rows
    ])


def _mk_pos(rows):
    """Build a positions DataFrame. Rows: (statement_date, broker,
    asset_class, market_value)."""
    return pd.DataFrame([
        {"statement_date": pd.Timestamp(d), "broker": b,
         "asset_class": ac, "market_value": mv,
         "account_id": "ACC", "symbol": "SPY", "quantity": 1}
        for d, b, ac, mv in rows
    ])


class IdentityTests(unittest.TestCase):
    """gross_paid - gross_received - sleeve_mv must equal cost_to_date."""

    def test_identity_holds_lifetime(self):
        txn = _mk_txn([
            ("2025-01-15", "harbor", -5000, "PUT SPY 03/21/25 580 OPEN"),
            ("2025-03-21", "harbor", +1200, "PUT SPY 03/21/25 580 CLOSE EXPIRE"),
            ("2025-06-10", "harbor", -8000, "PUT SPY 09/19/25 540 OPEN"),
        ])
        pos = _mk_pos([
            ("2025-01-31", "harbor", "option_put", 4900),
            ("2025-02-28", "harbor", "option_put", 4200),
            ("2025-03-31", "harbor", "option_put", 0),
            ("2025-06-30", "harbor", "option_put", 7800),
        ])
        df = build_protection_cost_timeline(txn, pos)
        for _, r in df.iterrows():
            self.assertAlmostEqual(
                r["cost_to_date"],
                r["gross_paid"] - r["gross_received"] - r["sleeve_mv"],
                places=2,
                msg=f"identity violated at {r['date'].date()}",
            )

    def test_identity_with_snapshot(self):
        txn = _mk_txn([
            ("2025-01-15", "harbor", -5000, "PUT SPY 03/21/25 580 OPEN"),
        ])
        pos = _mk_pos([("2025-01-31", "harbor", "option_put", 4800)])
        df = build_protection_cost_timeline(
            txn, pos,
            snapshot_today_mv=4500.0, today=pd.Timestamp("2025-02-15"),
        )
        last = df.iloc[-1]
        self.assertEqual(last["date"], pd.Timestamp("2025-02-15"))
        self.assertAlmostEqual(last["sleeve_mv"], 4500.0)
        self.assertAlmostEqual(
            last["cost_to_date"],
            last["gross_paid"] - last["gross_received"] - last["sleeve_mv"],
            places=2,
        )


class SnapshotAnchorTests(unittest.TestCase):

    def test_snapshot_added_when_today_past_latest_statement(self):
        txn = _mk_txn([("2025-01-15", "harbor", -5000, "PUT OPEN")])
        pos = _mk_pos([("2025-01-31", "harbor", "option_put", 4800)])
        df = build_protection_cost_timeline(
            txn, pos,
            snapshot_today_mv=4200.0, today=pd.Timestamp("2025-02-15"),
        )
        self.assertEqual(len(df), 2)  # Jan-31 statement + Feb-15 snapshot
        self.assertEqual(df.iloc[-1]["date"], pd.Timestamp("2025-02-15"))

    def test_snapshot_skipped_when_today_at_or_before_latest_statement(self):
        # Equal case — today == latest statement date, should NOT append a
        # row (would create a duplicate / contradictory point).
        txn = _mk_txn([("2025-01-15", "harbor", -5000, "PUT OPEN")])
        pos = _mk_pos([("2025-01-31", "harbor", "option_put", 4800)])
        df = build_protection_cost_timeline(
            txn, pos,
            snapshot_today_mv=4200.0, today=pd.Timestamp("2025-01-31"),
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[-1]["date"], pd.Timestamp("2025-01-31"))
        # Statement MV wins, snapshot ignored
        self.assertAlmostEqual(df.iloc[-1]["sleeve_mv"], 4800.0)


class HistoryStartRebaseTests(unittest.TestCase):

    def test_rebase_starts_at_zero(self):
        txn = _mk_txn([
            ("2025-01-15", "harbor", -5000, "PUT OPEN"),
            ("2025-06-10", "harbor", -8000, "PUT OPEN"),
        ])
        pos = _mk_pos([
            ("2025-01-31", "harbor", "option_put", 4500),  # cost = $500
            ("2025-03-31", "harbor", "option_put", 3200),  # cost = $1800
            ("2025-06-30", "harbor", "option_put", 10800), # cost = $2200
            ("2025-12-31", "harbor", "option_put", 8000),  # cost = $5000
        ])
        df = build_protection_cost_timeline(
            txn, pos, history_start=pd.Timestamp("2025-06-01"),
        )
        # Anchor row at 2025-06-01 with cost == 0
        self.assertEqual(df.iloc[0]["date"], pd.Timestamp("2025-06-01"))
        self.assertAlmostEqual(df.iloc[0]["cost_to_date"], 0.0)
        # 2025-06-30 lifetime cost was 2200; baseline was 1800 (3/31);
        # rebased cost at 6/30 = 2200 - 1800 = 400
        jun = df[df["date"] == pd.Timestamp("2025-06-30")].iloc[0]
        self.assertAlmostEqual(jun["cost_to_date"], 400.0, places=1)
        # 2025-12-31 rebased = 5000 - 1800 = 3200
        dec = df[df["date"] == pd.Timestamp("2025-12-31")].iloc[0]
        self.assertAlmostEqual(dec["cost_to_date"], 3200.0, places=1)
        # Pre-cutoff rows dropped
        self.assertFalse((df["date"] < pd.Timestamp("2025-06-01")).any())

    def test_rebase_with_no_prior_activity(self):
        # history_start before first put activity → baseline is 0,
        # series unchanged except for an anchor at the cutoff with cost 0.
        txn = _mk_txn([("2025-06-10", "harbor", -8000, "PUT OPEN")])
        pos = _mk_pos([
            ("2025-06-30", "harbor", "option_put", 7500),
        ])
        df = build_protection_cost_timeline(
            txn, pos, history_start=pd.Timestamp("2025-01-01"),
        )
        self.assertEqual(df.iloc[0]["date"], pd.Timestamp("2025-01-01"))
        self.assertAlmostEqual(df.iloc[0]["cost_to_date"], 0.0)
        # 2025-06-30 cost = 8000 - 7500 = 500, baseline = 0
        self.assertAlmostEqual(df.iloc[-1]["cost_to_date"], 500.0, places=1)


class MonthEndAggregationTests(unittest.TestCase):
    """Harbor (business-day) + Alpine (calendar-day) end-of-month quirk."""

    def test_two_statement_dates_in_one_month_combine(self):
        # May 2025: Harbor books 2025-05-30, Alpine books 2025-05-31.
        # The function must take both into the May month-end MV.
        txn = _mk_txn([
            ("2025-05-01", "harbor",      -3000, "PUT OPEN"),
            ("2025-05-02", "alpine", -2000, "PUT OPEN"),
        ])
        pos = _mk_pos([
            ("2025-05-30", "harbor",      "option_put", 2800),
            ("2025-05-31", "alpine", "option_put", 1900),
        ])
        df = build_protection_cost_timeline(txn, pos)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["date"], pd.Timestamp("2025-05-31"))
        self.assertAlmostEqual(row["sleeve_mv"], 4700.0)  # 2800 + 1900
        self.assertAlmostEqual(row["gross_paid"], 5000.0)  # 3000 + 2000
        self.assertAlmostEqual(row["cost_to_date"], 300.0)  # 5000 - 0 - 4700

    def test_latest_per_broker_within_month_wins(self):
        # If Harbor has two statements in a single month (unusual but possible
        # during onboarding), use the latest one only.
        txn = _mk_txn([("2025-04-01", "harbor", -1000, "PUT OPEN")])
        pos = _mk_pos([
            ("2025-04-15", "harbor", "option_put", 800),  # mid-month: ignored
            ("2025-04-30", "harbor", "option_put", 700),  # latest: used
        ])
        df = build_protection_cost_timeline(txn, pos)
        row = df.iloc[0]
        self.assertAlmostEqual(row["sleeve_mv"], 700.0)
        self.assertAlmostEqual(row["cost_to_date"], 300.0)

    def test_row_date_is_actual_statement_not_month_end(self):
        # Critical for the snapshot-append logic: the row's date column
        # must be the real latest statement_date within the month, not the
        # calendar month-end. Otherwise synth-rolled positions (dated 5/15
        # by synthesize_interim_positions) get bucketed under 5/31, and the
        # today=5/25 snapshot append silently doesn't fire because
        # `today > last_date` is false for last_date=5/31.
        txn = _mk_txn([("2025-05-15", "harbor", -1000, "PUT OPEN")])
        pos = _mk_pos([
            ("2025-05-15", "harbor", "option_put", 950),  # synth-rolled date
        ])
        df = build_protection_cost_timeline(
            txn, pos,
            snapshot_today_mv=900.0, today=pd.Timestamp("2025-05-25"),
        )
        # Expect TWO rows: 5/15 (synth-rolled) and 5/25 (snapshot append).
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["date"], pd.Timestamp("2025-05-15"))
        self.assertEqual(df.iloc[-1]["date"], pd.Timestamp("2025-05-25"))
        self.assertAlmostEqual(df.iloc[-1]["sleeve_mv"], 900.0)
        self.assertAlmostEqual(df.iloc[-1]["cost_to_date"], 100.0)  # 1000-0-900  # 1000 - 700


class EmptyAndEdgeCaseTests(unittest.TestCase):

    def test_no_put_transactions(self):
        # Stock-only universe — function returns empty
        txn = _mk_txn([
            ("2025-01-15", "harbor", -5000, "BOUGHT SPY ETF"),
        ])
        pos = _mk_pos([("2025-01-31", "harbor", "etf", 5100)])
        df = build_protection_cost_timeline(txn, pos)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), [
            "date", "gross_paid", "gross_received",
            "sleeve_mv", "cost_to_date"
        ])

    def test_empty_inputs(self):
        df = build_protection_cost_timeline(
            pd.DataFrame(columns=["settlement_date", "broker", "amount",
                                  "description"]),
            pd.DataFrame(columns=["statement_date", "broker", "asset_class",
                                  "market_value"]),
        )
        self.assertTrue(df.empty)

    def test_calls_not_counted(self):
        # CALL rows in description must be filtered out — only PUTs count
        # toward "cost of PROTECTION".
        txn = _mk_txn([
            ("2025-01-15", "harbor", -5000, "CALL TSLA 03/21/25 OPEN"),
            ("2025-02-10", "harbor", -3000, "PUT SPY 03/21/25 OPEN"),
        ])
        pos = _mk_pos([
            ("2025-02-28", "harbor", "option_put",  2800),
            ("2025-02-28", "harbor", "option_call", 4500),
        ])
        df = build_protection_cost_timeline(txn, pos)
        # CALL txn ignored on gross_paid; CALL position ignored on sleeve_mv
        row = df.iloc[0]
        self.assertAlmostEqual(row["gross_paid"], 3000.0)
        self.assertAlmostEqual(row["sleeve_mv"], 2800.0)


if __name__ == "__main__":
    unittest.main()
