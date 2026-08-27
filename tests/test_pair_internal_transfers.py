"""
Tests for parsers/pair_internal_transfers.py.

547 LOC of pipeline-critical logic that was previously untested. Covers:

  - `_find_counterparty` / `_looks_like_sweep` / `_make_pair_id` (helpers)
  - `pair_transfers` — counterparty-named pairing, sweep detection,
    leftover→external default, non-flow rows untouched
  - `_pair_within_broker_same_day` — JPM Cash Flow Summary pairing
    (exact match, partial split with residual, same-account skip)
  - `synthesize_in_kind_flows` — NAV-delta inference, min(recv,sent) safety,
    day-1 settle date, idempotent re-run, pre-tracking-debut skip
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))
sys.path.insert(0, str(ROOT))

import pair_internal_transfers as pit  # noqa: E402


# ---------------------------------------------------------------------------
# Helper to build transaction fixtures
# ---------------------------------------------------------------------------
def _txn(rows: list[dict]) -> pd.DataFrame:
    """Build a transaction DataFrame with the columns pair_transfers expects."""
    defaults = {
        "settlement_date": pd.Timestamp("2026-01-15"),
        "trade_date": pd.Timestamp("2026-01-15"),
        "broker": "fidelity",
        "account_id": "X10-000007",
        "transaction_type": "transfer_in",
        "symbol": pd.NA,
        "cusip": pd.NA,
        "description": "",
        "quantity": pd.NA,
        "price": pd.NA,
        "amount": 0.0,
        "source_file": "test",
    }
    filled = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(filled)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _positions(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["statement_date"] = pd.to_datetime(df["statement_date"])
    return df


def _empty_txn() -> pd.DataFrame:
    """Empty transactions frame with the right dtypes for the in-kind detector."""
    return pd.DataFrame({
        "settlement_date": pd.Series([], dtype="datetime64[ns]"),
        "trade_date":      pd.Series([], dtype="datetime64[ns]"),
        "broker":          pd.Series([], dtype="object"),
        "account_id":      pd.Series([], dtype="object"),
        "transaction_type": pd.Series([], dtype="object"),
        "symbol":          pd.Series([], dtype="object"),
        "cusip":           pd.Series([], dtype="object"),
        "description":     pd.Series([], dtype="object"),
        "quantity":        pd.Series([], dtype="float64"),
        "price":           pd.Series([], dtype="float64"),
        "amount":          pd.Series([], dtype="float64"),
        "source_file":     pd.Series([], dtype="object"),
        "flow_scope":      pd.Series([], dtype="object"),
        "pair_id":         pd.Series([], dtype="object"),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestFindCounterparty(unittest.TestCase):
    def test_extracts_other_account(self) -> None:
        desc = "JOURNALED SHARES Z10-000008-1 INTERNAL TRANSFER"
        self.assertEqual(pit._find_counterparty(desc, "X10-000007"), "Z10-000008")

    def test_self_account_in_description_is_ignored(self) -> None:
        # The account_id of the row itself must not be picked up as counterparty.
        desc = "X10-000007 self-reference; real counterparty Z10-000008"
        self.assertEqual(pit._find_counterparty(desc, "X10-000007"), "Z10-000008")

    def test_returns_none_when_no_match(self) -> None:
        self.assertIsNone(pit._find_counterparty("EFT FROM CHASE BANK", "X10-000007"))

    def test_non_string_returns_none(self) -> None:
        self.assertIsNone(pit._find_counterparty(None, "X10-000007"))
        self.assertIsNone(pit._find_counterparty(float("nan"), "X10-000007"))


class TestLooksLikeSweep(unittest.TestCase):
    def test_matches_known_patterns(self) -> None:
        self.assertTrue(pit._looks_like_sweep("CASH Transferred to FCASH"))
        self.assertTrue(pit._looks_like_sweep("FCASH IS LIQUIDATED"))
        self.assertTrue(pit._looks_like_sweep("FIDELITY GOVERNMENT MONEY MARKET"))

    def test_matches_margin_to_cash_journal(self) -> None:
        # WSF-8: same-account margin<->cash journal must read as a sweep
        # (internal), not external — the only 2 of 58 external flows that
        # actually look internal.
        self.assertTrue(pit._looks_like_sweep("MARGIN TO CASH A/C Journaled"))

    def test_rejects_non_sweep_descriptions(self) -> None:
        self.assertFalse(pit._looks_like_sweep("EFT FROM CHASE BANK"))
        self.assertFalse(pit._looks_like_sweep(None))


class TestMakePairId(unittest.TestCase):
    def test_deterministic(self) -> None:
        d = pd.Timestamp("2026-03-15")
        a = pit._make_pair_id(d, 100.0, "X10-000007", "Z10-000008")
        b = pit._make_pair_id(d, 100.0, "X10-000007", "Z10-000008")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)

    def test_account_order_invariant(self) -> None:
        d = pd.Timestamp("2026-03-15")
        a = pit._make_pair_id(d, 100.0, "X10-000007", "Z10-000008")
        b = pit._make_pair_id(d, 100.0, "Z10-000008", "X10-000007")
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# pair_transfers — counterparty-named pairing + sweep + external default
# ---------------------------------------------------------------------------
class TestPairTransfers(unittest.TestCase):
    def test_counterparty_named_pair_marked_internal(self) -> None:
        # X10-000007 sends $1,000 to Z10-000008, both rows name the other side.
        df = _txn([
            {"account_id": "X10-000007", "transaction_type": "transfer_out",
             "amount": -1000.0,
             "description": "JOURNAL TO Z10-000008 INTERNAL TRANSFER"},
            {"account_id": "Z10-000008", "transaction_type": "transfer_in",
             "amount": 1000.0,
             "description": "JOURNAL FROM X10-000007 INTERNAL TRANSFER"},
        ])
        out = pit.pair_transfers(df)
        scopes = out["flow_scope"].tolist()
        self.assertEqual(scopes, ["internal", "internal"])
        self.assertEqual(out["pair_id"].iloc[0], out["pair_id"].iloc[1])
        self.assertIsNotNone(out["pair_id"].iloc[0])

    def test_sweep_pattern_marked_sweep(self) -> None:
        df = _txn([
            {"account_id": "X10-000007", "transaction_type": "transfer_out",
             "amount": -500.0, "description": "CASH Transferred to FGMM"},
        ])
        out = pit.pair_transfers(df)
        self.assertEqual(out["flow_scope"].iloc[0], "sweep")

    def test_unmatched_flow_defaults_external(self) -> None:
        # A real bank wire with no counterparty hint must remain external.
        df = _txn([
            {"account_id": "X10-000007", "transaction_type": "transfer_in",
             "amount": 10000.0, "description": "EFT FROM CHASE BANK"},
        ])
        out = pit.pair_transfers(df)
        self.assertEqual(out["flow_scope"].iloc[0], "external")
        self.assertIsNone(out["pair_id"].iloc[0])

    def test_non_flow_rows_get_empty_scope(self) -> None:
        # buy/sell/dividend etc. aren't flows; pair_transfers must not touch them.
        df = _txn([
            {"account_id": "X10-000007", "transaction_type": "buy",
             "amount": -100.0, "description": "AAPL purchase"},
            {"account_id": "X10-000007", "transaction_type": "dividend",
             "amount": 2.50, "description": "AAPL DIVIDEND"},
        ])
        out = pit.pair_transfers(df)
        self.assertEqual(out["flow_scope"].tolist(), ["", ""])


# ---------------------------------------------------------------------------
# _pair_within_broker_same_day — JPM CFS pairing
# ---------------------------------------------------------------------------
class TestPairWithinBrokerSameDay(unittest.TestCase):
    def test_exact_match_same_day_marked_internal(self) -> None:
        # Two JPM accounts on the same day with equal-and-opposite amounts,
        # no counterparty in description — must still pair.
        df = _txn([
            {"broker": "jpm", "account_id": "100-00001",
             "transaction_type": "transfer_out", "amount": -816800.0,
             "description": "CASH WITHDRAWAL"},
            {"broker": "jpm", "account_id": "100-00003",
             "transaction_type": "transfer_in", "amount": 816800.0,
             "description": "CASH DEPOSIT"},
        ])
        # Need to call full pair_transfers (it tags external first, then
        # _pair_within_broker_same_day promotes to internal).
        out = pit.pair_transfers(df)
        self.assertEqual(out["flow_scope"].tolist(), ["internal", "internal"])
        self.assertEqual(out["pair_id"].iloc[0], out["pair_id"].iloc[1])

    def test_partial_pair_splits_larger_with_external_residual(self) -> None:
        # $1000 deposit into A pairs against $700 withdrawal from B (same broker,
        # same day, distinct accounts). $700 of the deposit becomes internal;
        # the $300 residual must remain external (a real bank wire).
        df = _txn([
            {"broker": "jpm", "account_id": "AAA-11111",
             "transaction_type": "transfer_in", "amount": 1000.0,
             "description": "DEPOSIT"},
            {"broker": "jpm", "account_id": "BBB-22222",
             "transaction_type": "transfer_out", "amount": -700.0,
             "description": "WITHDRAWAL"},
        ])
        out = pit.pair_transfers(df)
        scopes = out["flow_scope"].value_counts().to_dict()
        # Should have: 1 internal pair (2 rows) + 1 external residual = 3 rows total
        self.assertEqual(scopes.get("internal", 0), 2)
        self.assertEqual(scopes.get("external", 0), 1)
        # The internal-paired amounts must reconcile to $0 net.
        internal = out[out["flow_scope"] == "internal"]
        self.assertAlmostEqual(internal["amount"].sum(), 0.0, places=2)
        # The remaining external row must be the $300 residual on AAA.
        external = out[out["flow_scope"] == "external"]
        self.assertEqual(len(external), 1)
        self.assertAlmostEqual(external["amount"].iloc[0], 300.0, places=2)
        self.assertEqual(external["account_id"].iloc[0], "AAA-11111")

    def test_same_account_not_paired(self) -> None:
        # Two flows on the same day in the SAME account must not pair against
        # each other (that would zero out a real round-trip).
        df = _txn([
            {"broker": "jpm", "account_id": "AAA-11111",
             "transaction_type": "transfer_in", "amount": 500.0,
             "description": "DEPOSIT"},
            {"broker": "jpm", "account_id": "AAA-11111",
             "transaction_type": "transfer_out", "amount": -500.0,
             "description": "WITHDRAWAL"},
        ])
        out = pit.pair_transfers(df)
        # Both stay external; no pair_id assigned.
        self.assertEqual(out["flow_scope"].tolist(), ["external", "external"])
        self.assertTrue(out["pair_id"].isna().all())


# ---------------------------------------------------------------------------
# synthesize_in_kind_flows — NAV-delta inference
# ---------------------------------------------------------------------------
class TestSynthesizeInKindFlows(unittest.TestCase):
    def _setup_in_kind_event(self, send_amt: float, recv_amt: float,
                              donor_dec_nav: float = 500_000.0):
        """Donor account loses send_amt of NAV with no flows; receiver gains
        recv_amt of NAV with no flows. Both must trip the in-kind detector.

        donor_dec_nav defaults to $500K so send_amt of $108K+ exceeds the
        15% pct-of-prev-NAV threshold and the absolute-$ threshold; the
        receiver is always a first-month debut so it trips on the abs alone.
        """
        positions = _positions([
            {"statement_date": "2025-12-31", "account_id": "100-00001",
             "broker": "jpm", "market_value": donor_dec_nav},
            {"statement_date": "2026-01-31", "account_id": "100-00001",
             "broker": "jpm", "market_value": donor_dec_nav - send_amt},
            {"statement_date": "2026-01-31", "account_id": "100-00002",
             "broker": "jpm", "market_value": recv_amt},
        ])
        # No flows in transactions at all — pure in-kind shift.
        txn = _empty_txn()
        return positions, txn

    def test_paired_synthesis_uses_min_recv_sent(self) -> None:
        # Receiver gained $127K, donor lost $108K → synthesize $108K (the min).
        # The $19K residual is left attributed to market movement.
        positions, txn = self._setup_in_kind_event(
            send_amt=108_000.0, recv_amt=127_000.0)
        augmented, count = pit.synthesize_in_kind_flows(txn, positions)
        self.assertEqual(count, 1)
        # Two new rows (the pair) appended.
        self.assertEqual(len(augmented), 2)
        amounts = sorted(augmented["amount"].tolist())
        self.assertAlmostEqual(amounts[0], -108_000.0, places=2)
        self.assertAlmostEqual(amounts[1], +108_000.0, places=2)
        # Both rows share a pair_id.
        self.assertEqual(augmented["pair_id"].iloc[0], augmented["pair_id"].iloc[1])

    def test_settle_date_is_day_one_of_month(self) -> None:
        # In-kind settle date MUST be day-1 of the destination month so
        # modified Dietz gives the flow weight ~1 (covers a known regression).
        positions, txn = self._setup_in_kind_event(
            send_amt=200_000.0, recv_amt=200_000.0)
        augmented, _ = pit.synthesize_in_kind_flows(txn, positions)
        for d in augmented["settlement_date"]:
            self.assertEqual(pd.Timestamp(d).day, 1)
            self.assertEqual(pd.Timestamp(d).month, 1)
            self.assertEqual(pd.Timestamp(d).year, 2026)

    def test_idempotent_rerun_strips_prior_synthetic(self) -> None:
        # Running twice must not produce duplicate synthetic pairs.
        positions, txn = self._setup_in_kind_event(
            send_amt=200_000.0, recv_amt=200_000.0)
        first, n1 = pit.synthesize_in_kind_flows(txn, positions)
        second, n2 = pit.synthesize_in_kind_flows(first, positions)
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 1)
        # After the second pass: same row count as after the first.
        self.assertEqual(len(first), len(second))

    def test_no_suspects_returns_input_unchanged(self) -> None:
        # NAV moves but every move is below the absolute threshold → no synth.
        positions = _positions([
            {"statement_date": "2025-12-31", "account_id": "100-00001",
             "broker": "jpm", "market_value": 1_000_000.0},
            {"statement_date": "2026-01-31", "account_id": "100-00001",
             "broker": "jpm", "market_value": 1_010_000.0},  # +$10K, < $50K
        ])
        txn = _empty_txn()
        out, count = pit.synthesize_in_kind_flows(txn, positions)
        self.assertEqual(count, 0)
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
