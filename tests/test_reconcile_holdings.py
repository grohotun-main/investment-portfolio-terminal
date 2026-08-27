"""Tests for parsers/reconcile_holdings.py — the pure core of the holdings
reconciliation guard.

The guard compares each account's EXTRACTED position value (summed
market_value) against the statement's REPORTED account total, in two bands,
so a silently-wrong extraction (the JPM May +$234K phantom) is caught at
ingest while legitimate residuals stay quiet.

The residuals that shape the bands:
  * accrued income / unpriced positions make extracted != reported even when
    extraction is perfect (~0.1% on the live book),
  * 100-00004 (Parametric, 300+ direct-index lots) carries ~0.2-0.5% genuine
    lot-rounding noise,
  * the +$234K phantom was +15%.

So: WATCH at |diff%| > 0.30%, ERROR (blocks) at |diff%| > 2% AND |diff$| >
$10,000, and a per-account allowlist tolerance reclassifies known noise as
"known" until it grows past the tolerance (then it escalates through the
normal bands again).
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from reconcile_holdings import (  # noqa: E402
    classify, reconcile, format_table, load_allowlist,
    upsert_summaries, SUMMARIES_COLUMNS,
    LaggingRow, lagging_accounts, format_lagging,
)


class TestClassify(unittest.TestCase):
    def test_large_drift_in_both_pct_and_dollars_is_error(self) -> None:
        # The JPM May 100-00001 phantom shape: extracted $1,150,000 vs reported
        # $1,000,000 = +$150,000 / +15%. Trips both ERROR clauses.
        self.assertEqual(classify(1_150_000.00, 1_000_000.00), "error")

    def test_negative_large_drift_is_error(self) -> None:
        # ERROR is on |diff|, so an under-extraction of the same magnitude
        # must also fire — easy to regress to a signed comparison.
        self.assertEqual(classify(1_000_000.00, 1_150_000.00), "error")

    def test_subthreshold_parametric_drift_is_watch(self) -> None:
        # +0.48% with no allowlist: above the 0.30% WATCH floor, below ERROR.
        self.assertEqual(classify(100_480.0, 100_000.0), "watch")

    def test_drift_within_allowlist_tolerance_is_known(self) -> None:
        # Same +0.48%, but the account is allowlisted at 0.6% known-noise.
        self.assertEqual(classify(100_480.0, 100_000.0, tol_pct=0.6), "known")

    def test_drift_above_allowlist_tolerance_escalates(self) -> None:
        # +0.70% exceeds the 0.6% tolerance -> normal bands resume -> WATCH.
        # This is the "revisit if it grows" escalation.
        self.assertEqual(classify(100_700.0, 100_000.0, tol_pct=0.6), "watch")

    def test_accrued_income_residual_is_ok(self) -> None:
        # -0.09% (extracted just under reported by accrued income) is below
        # the WATCH floor and must stay quiet every month.
        self.assertEqual(classify(99_910.0, 100_000.0), "ok")

    def test_small_dollar_high_pct_is_watch_not_error(self) -> None:
        # +3% but only +$3,000 — above ERROR's pct clause but below its
        # $10,000 floor, so the AND keeps it out of ERROR. Surfaced as WATCH.
        self.assertEqual(classify(103_000.0, 100_000.0), "watch")

    def test_large_dollar_low_pct_is_watch_not_error(self) -> None:
        # +$150,000 but only +1.5% on a $10M account — above ERROR's dollar
        # floor but below its 2% clause, so again not ERROR.
        self.assertEqual(classify(10_150_000.0, 10_000_000.0), "watch")

    def test_zero_reported_with_extracted_is_error_no_raise(self) -> None:
        # A reported total of 0 for an account we extracted value for is
        # unreconcilable; flag it ERROR rather than dividing by zero.
        self.assertEqual(classify(5_000.0, 0.0), "error")

    def test_zero_reported_and_zero_extracted_is_ok(self) -> None:
        # Both zero: nothing to reconcile, no false alarm, no ZeroDivision.
        self.assertEqual(classify(0.0, 0.0), "ok")


class TestReconcile(unittest.TestCase):
    def _fixture(self):
        extracted = {
            ("jpm", "100-00001", "2026-05"): 1_780_953.92,   # phantom -> error
            ("jpm", "100-00004", "2026-05"): 100_480.0,       # +0.48% parametric
            ("fidelity", "X10-000007", "2026-05"): 99_910.0,  # -0.09% accrued
        }
        reported = {
            ("jpm", "100-00001", "2026-05"): 1_546_937.88,
            ("jpm", "100-00004", "2026-05"): 100_000.0,
            ("fidelity", "X10-000007", "2026-05"): 100_000.0,
        }
        allowlist = {"100-00004": {"max_pct": 0.6,
                                   "reason": "Parametric TLH lot-rounding"}}
        return extracted, reported, allowlist

    def test_maps_dicts_to_rows_with_bands_and_allowlist(self) -> None:
        extracted, reported, allowlist = self._fixture()
        rows = reconcile(extracted, reported, allowlist)
        by_acct = {r.account_id: r for r in rows}
        self.assertEqual(by_acct["100-00001"].band, "error")
        self.assertEqual(by_acct["100-00004"].band, "known")  # within 0.6% tol
        self.assertEqual(by_acct["X10-000007"].band, "ok")

    def test_row_carries_signed_diff_in_dollars_and_pct(self) -> None:
        extracted, reported, allowlist = self._fixture()
        by_acct = {r.account_id: r for r in reconcile(extracted, reported, allowlist)}
        self.assertAlmostEqual(by_acct["100-00001"].diff_usd, 234_016.04, places=2)
        self.assertAlmostEqual(by_acct["100-00001"].diff_pct, 15.1277, places=3)
        self.assertEqual(by_acct["100-00001"].broker, "jpm")
        self.assertEqual(by_acct["100-00001"].month, "2026-05")

    def test_account_reported_but_not_extracted_is_error(self) -> None:
        # Reported has the account; extraction produced nothing for it. A
        # silently-dropped account must surface as ERROR, not be omitted.
        extracted = {}
        reported = {("jpm", "100-00003", "2026-05"): 400_000.0}
        rows = reconcile(extracted, reported, {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].band, "error")


class TestFormatTable(unittest.TestCase):
    def test_includes_account_ids_and_band_labels(self) -> None:
        extracted = {("jpm", "100-00001", "2026-05"): 1_780_953.92}
        reported = {("jpm", "100-00001", "2026-05"): 1_546_937.88}
        out = format_table(reconcile(extracted, reported, {}))
        self.assertIn("100-00001", out)
        self.assertIn("error", out.lower())

    def test_handles_empty_rows(self) -> None:
        # No rows (nothing to reconcile) must yield a string, not crash.
        self.assertIsInstance(format_table([]), str)


class TestLoadAllowlist(unittest.TestCase):
    """The guard's per-account tolerances come from config_local (gitignored).
    Loading is optional — {} when config_local or the constant is absent — and
    every entry must carry a positive numeric max_pct that reconcile() can use."""
    def test_returns_a_dict(self) -> None:
        self.assertIsInstance(load_allowlist(), dict)

    def test_every_entry_has_a_positive_numeric_max_pct(self) -> None:
        for acct, spec in load_allowlist().items():
            self.assertIn("max_pct", spec, acct)
            self.assertIsInstance(spec["max_pct"], (int, float), acct)
            self.assertGreater(spec["max_pct"], 0, acct)


class TestUpsertSummaries(unittest.TestCase):
    """summaries.csv has been Phase-0-frozen (written by nothing). The guard
    advances it with each ingest's reported totals — adding new (broker,
    account, month) keys and refreshing existing ones, never dropping history
    for other keys (so a corrected re-download replaces cleanly, not duplicates)."""
    def _rec(self, date, broker, acct, total, src):
        return dict(zip(SUMMARIES_COLUMNS, [date, broker, acct, total, src]))

    def _seed(self, path, rows):
        pd.DataFrame(rows, columns=SUMMARIES_COLUMNS).to_csv(path, index=False)

    def test_adds_new_month_preserving_history(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "summaries.csv"
            self._seed(p, [self._rec("2026-04-30", "jpm", "AAA-11111",
                                     400000.0, "apr.pdf")])
            upsert_summaries([self._rec("2026-05-31", "jpm", "AAA-11111",
                                        410000.0, "may.pdf")], p)
            df = pd.read_csv(p, dtype=str)
            months = set(df["statement_date"].str.slice(0, 7))
            self.assertEqual(months, {"2026-04", "2026-05"})

    def test_replaces_same_broker_account_month(self) -> None:
        # A corrected re-ingest of the same key must overwrite, not duplicate.
        with TemporaryDirectory() as td:
            p = Path(td) / "summaries.csv"
            self._seed(p, [self._rec("2026-05-31", "jpm", "AAA-11111",
                                     100.0, "bad.pdf")])
            upsert_summaries([self._rec("2026-05-31", "jpm", "AAA-11111",
                                        110.0, "fixed.pdf")], p)
            df = pd.read_csv(p, dtype=str)
            self.assertEqual(len(df), 1)
            self.assertEqual(float(df.iloc[0]["reported_total"]), 110.0)

    def test_empty_records_is_noop(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "summaries.csv"
            self._seed(p, [self._rec("2026-04-30", "fid", "BBB-22222",
                                     1.0, "a.pdf")])
            upsert_summaries([], p)
            self.assertEqual(len(pd.read_csv(p)), 1)

    def test_creates_file_when_absent(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "summaries.csv"
            upsert_summaries([self._rec("2026-05-31", "fid", "BBB-22222",
                                        50.0, "x.pdf")], p)
            self.assertTrue(p.exists())
            self.assertEqual(len(pd.read_csv(p)), 1)


class TestLaggingAccounts(unittest.TestCase):
    """Per-broker frontier: an account whose newest statement month is behind
    its OWN broker's newest is lagging (its broker advanced and it didn't come
    along). Brokers are scored independently; a suppress set silences closed ones."""

    def test_account_behind_broker_frontier_is_flagged(self) -> None:
        latest = {("fidelity", "Z10-000008"): "2026-04",
                  ("fidelity", "Z10-000009"): "2026-04",
                  ("fidelity", "X10-000007"): "2026-05",
                  ("fidelity", "100-000006"): "2026-05"}
        rows = lagging_accounts(latest)
        self.assertEqual({r.account_id for r in rows}, {"Z10-000008", "Z10-000009"})
        for r in rows:
            self.assertEqual(r.broker, "fidelity")
            self.assertEqual((r.last_month, r.broker_latest), ("2026-04", "2026-05"))

    def test_brokers_are_independent(self) -> None:
        # jpm sits entirely at April; fidelity at May. jpm accounts must NOT be
        # flagged just because fidelity is ahead — each broker has its own frontier.
        latest = {("jpm", "100-00003"): "2026-04", ("jpm", "100-00004"): "2026-04",
                  ("fidelity", "X10-000007"): "2026-05", ("fidelity", "Z10-000008"): "2026-04"}
        flagged = {(r.broker, r.account_id) for r in lagging_accounts(latest)}
        self.assertEqual(flagged, {("fidelity", "Z10-000008")})

    def test_all_current_is_empty(self) -> None:
        latest = {("fidelity", "X10-000007"): "2026-05", ("fidelity", "100-000006"): "2026-05"}
        self.assertEqual(lagging_accounts(latest), [])

    def test_single_account_broker_never_lags(self) -> None:
        self.assertEqual(lagging_accounts({("jpm", "100-00001"): "2026-05"}), [])

    def test_suppress_silences_named_account(self) -> None:
        latest = {("fidelity", "X10-000007"): "2026-05", ("fidelity", "OLD-99999"): "2026-01"}
        self.assertEqual(lagging_accounts(latest, suppress={"OLD-99999"}), [])

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(lagging_accounts({}), [])

    def test_two_brokers_each_with_a_laggard(self) -> None:
        # Each broker is scored against its OWN frontier, so a laggard in each
        # broker is flagged independently in the same call.
        latest = {("fidelity", "X10-000007"): "2026-05",
                  ("fidelity", "Z10-000008"): "2026-04",
                  ("jpm", "100-00001"): "2026-05",
                  ("jpm", "100-00002"): "2026-03"}
        flagged = {(r.broker, r.account_id) for r in lagging_accounts(latest)}
        self.assertEqual(flagged, {("fidelity", "Z10-000008"),
                                   ("jpm", "100-00002")})


class TestFormatLagging(unittest.TestCase):
    def test_lists_accounts_and_count_when_lagging(self) -> None:
        rows = [LaggingRow("fidelity", "Z10-000008", "2026-04", "2026-05"),
                LaggingRow("fidelity", "Z10-000009", "2026-04", "2026-05")]
        out = format_lagging(rows)
        self.assertIn("Z10-000008", out)
        self.assertIn("Z10-000009", out)
        self.assertIn("2 carried forward", out)
        self.assertIn("behind their broker's latest statement", out)

    def test_all_current_message_when_empty(self) -> None:
        self.assertIn("all accounts current", format_lagging([]).lower())

    def test_single_account_message(self) -> None:
        out = format_lagging([LaggingRow("jpm", "100-00004", "2026-04", "2026-05")])
        self.assertIn("100-00004", out)
        self.assertIn("1 carried forward", out)
        self.assertIn("missing statement and", out)   # singular, no "(s)"


if __name__ == "__main__":
    unittest.main()
