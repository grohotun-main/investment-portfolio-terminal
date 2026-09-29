"""Unit tests for parsers/data_health.py (pure; inline frames, no I/O)."""
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from data_health import (  # noqa: E402
    AccountHealth,
    HealthReport,
    build_health_report,
    format_health_headline,
    health_rows_to_table,
)
from mark_to_market import mark_to_market  # noqa: E402

TODAY = date(2026, 5, 10)
SUMM_COLS = ["statement_date", "broker", "account_id", "reported_total", "source_file"]
POS_COLS = ["statement_date", "broker", "account_id", "market_value"]


def _summaries(rows):
    return pd.DataFrame(rows, columns=SUMM_COLS)


def _positions(rows):
    return pd.DataFrame(rows, columns=POS_COLS)


def _by_id(report):
    return {a.account_id: a for a in report.accounts}


class TestBuildHealthReport(unittest.TestCase):
    def test_empty_summaries_is_grey_and_unavailable(self):
        rep = build_health_report(
            _positions([("2026-04-30", "alpine", "A", 100.0)]),
            _summaries([]),
            today=TODAY,
        )
        self.assertFalse(rep.recon_available)
        self.assertEqual(rep.worst_level, "grey")
        self.assertEqual(rep.accounts, [])

    def test_verified_ok_when_extracted_equals_reported(self):
        rep = build_health_report(
            _positions([("2026-04-30", "alpine", "A", 100_000.0)]),
            _summaries([("2026-04-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        self.assertTrue(rep.recon_available)
        self.assertEqual(rep.as_of_month, "2026-04")
        a = _by_id(rep)["A"]
        self.assertEqual(a.state, "verified")
        self.assertEqual(a.band, "ok")
        self.assertFalse(a.lagging)
        self.assertEqual(rep.worst_level, "green")
        self.assertEqual(rep.n_ok, 1)

    def test_verified_watch_band(self):
        # +0.5% drift -> watch (>0.30%, not the >2% AND >$10k error band)
        rep = build_health_report(
            _positions([("2026-04-30", "alpine", "A", 100_500.0)]),
            _summaries([("2026-04-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        a = _by_id(rep)["A"]
        self.assertEqual(a.band, "watch")
        self.assertEqual(rep.n_watch, 1)
        self.assertEqual(rep.worst_level, "amber")

    def test_verified_error_band(self):
        # +15% and +$15k -> error
        rep = build_health_report(
            _positions([("2026-04-30", "alpine", "A", 115_000.0)]),
            _summaries([("2026-04-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        a = _by_id(rep)["A"]
        self.assertEqual(a.band, "error")
        self.assertEqual(rep.n_error, 1)
        self.assertEqual(rep.worst_level, "red")

    def test_known_band_within_allowlist_tolerance(self):
        # +0.4% drift, allowlisted tol 0.5% -> known (info, not watch)
        rep = build_health_report(
            _positions([("2026-04-30", "harbor", "P", 100_400.0)]),
            _summaries([("2026-04-30", "harbor", "P", 100_000.0, "s")]),
            today=TODAY,
            allowlist={"P": {"max_pct": 0.5}},
        )
        a = _by_id(rep)["P"]
        self.assertEqual(a.band, "known")
        self.assertEqual(rep.n_known, 1)
        self.assertEqual(rep.worst_level, "green")

    def test_missing_account_reported_but_not_extracted(self):
        # Reported at M, zero extracted -> state "missing", counts as error.
        rep = build_health_report(
            _positions([("2026-04-30", "alpine", "A", 100_000.0)]),
            _summaries([
                ("2026-04-30", "alpine", "A", 100_000.0, "s"),
                ("2026-04-30", "alpine", "B", 50_000.0, "s"),
            ]),
            today=TODAY,
        )
        b = _by_id(rep)["B"]
        self.assertEqual(b.state, "missing")
        self.assertGreaterEqual(rep.n_error, 1)
        self.assertEqual(rep.worst_level, "red")

    def test_carried_and_lagging_when_broker_advanced(self):
        # F1 has April; F2 only March -> F2 carried AND lagging (broker frontier
        # advanced to April without it).
        rep = build_health_report(
            _positions([
                ("2026-04-30", "alpine", "F1", 10_000.0),
                ("2026-03-31", "alpine", "F2", 5_000.0),
            ]),
            _summaries([
                ("2026-04-30", "alpine", "F1", 10_000.0, "s"),
                ("2026-03-31", "alpine", "F2", 5_000.0, "s"),
            ]),
            today=TODAY,
        )
        f2 = _by_id(rep)["F2"]
        self.assertEqual(f2.state, "carried")
        self.assertTrue(f2.lagging)
        self.assertIsNone(f2.band)
        self.assertEqual(f2.last_verified_month, "2026-03")
        self.assertEqual(f2.days_since, (TODAY - date(2026, 3, 31)).days)
        self.assertEqual(rep.n_carried, 1)
        self.assertEqual(rep.worst_level, "amber")

    def test_carried_but_not_lagging_when_whole_broker_behind(self):
        # alpine reached April (sets M); harbor's only account is at March, but
        # harbor's own frontier is March -> J1 carried (behind M) yet not lagging.
        rep = build_health_report(
            _positions([
                ("2026-04-30", "alpine", "F1", 10_000.0),
                ("2026-03-31", "harbor", "J1", 7_000.0),
            ]),
            _summaries([
                ("2026-04-30", "alpine", "F1", 10_000.0, "s"),
                ("2026-03-31", "harbor", "J1", 7_000.0, "s"),
            ]),
            today=TODAY,
        )
        j1 = _by_id(rep)["J1"]
        self.assertEqual(j1.state, "carried")
        self.assertFalse(j1.lagging)

    def test_worst_level_error_beats_watch_beats_carried(self):
        rep = build_health_report(
            _positions([
                ("2026-04-30", "alpine", "ERR", 115_000.0),
                ("2026-04-30", "alpine", "WAT", 100_500.0),
                ("2026-03-31", "alpine", "CAR", 5_000.0),
            ]),
            _summaries([
                ("2026-04-30", "alpine", "ERR", 100_000.0, "s"),
                ("2026-04-30", "alpine", "WAT", 100_000.0, "s"),
                ("2026-03-31", "alpine", "CAR", 5_000.0, "s"),
            ]),
            today=TODAY,
        )
        self.assertEqual(rep.worst_level, "red")
        # error/missing sorts first
        self.assertEqual(rep.accounts[0].account_id, "ERR")

    def test_unreconciled_newer_holdings_flagged(self):
        # Real account A has an April statement (reconciled) AND May holdings
        # with no May reported total -> May is unreconciled, verdict amber.
        rep = build_health_report(
            _positions([
                ("2026-04-30", "alpine", "A", 100_000.0),
                ("2026-05-31", "alpine", "A", 101_000.0),
            ]),
            _summaries([("2026-04-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        self.assertEqual(rep.as_of_month, "2026-04")
        self.assertEqual(rep.unreconciled_months, ["2026-05"])
        self.assertEqual(rep.worst_level, "amber")
        a = _by_id(rep)["A"]
        self.assertEqual(a.state, "verified")
        self.assertEqual(a.band, "ok")

    def test_newer_holdings_for_non_statement_account_not_flagged(self):
        # An account with May holdings but NO summaries rows at all (e.g. the
        # demo-broker overlay) is not a real account -> excluded from the
        # roster AND from unreconciled_months.
        rep = build_health_report(
            _positions([
                ("2026-04-30", "alpine", "A", 100_000.0),
                ("2026-05-31", "alpine", "DEMO", 5_000.0),
            ]),
            _summaries([("2026-04-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        self.assertEqual(rep.unreconciled_months, [])
        self.assertEqual(rep.worst_level, "green")
        self.assertNotIn("DEMO", _by_id(rep))


class TestStatementBasisExtraction(unittest.TestCase):
    """Extracted must stay on the STATEMENT basis (matching the ingest gate).

    Both UIs pass positions frames whose latest snapshot was re-marked to live
    prices at load; mark_to_market stashes the pre-mark values in
    `market_value_stmt` and build_health_report must prefer that column, so a
    real market move since the statement date cannot read as reconciliation
    drift (2026-07-13: memory-sector names fell ~20-26% after the June-30
    statements and produced false ERROR-band rows).
    """

    def test_extracted_prefers_statement_basis_stash(self):
        # Marked value alone would be a -22% ERROR; the stash reconciles.
        pos = _positions([("2026-04-30", "alpine", "A", 78_000.0)])
        pos["market_value_stmt"] = [100_000.0]
        rep = build_health_report(
            pos,
            _summaries([("2026-04-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        a = _by_id(rep)["A"]
        self.assertEqual(a.band, "ok")
        self.assertEqual(a.extracted, 100_000.0)
        self.assertEqual(rep.worst_level, "green")

    def test_stash_nan_rows_fall_back_to_market_value(self):
        # A row missing the stash (e.g. concatenated after marking) must still
        # contribute its market_value instead of dropping out of the sum.
        pos = _positions([
            ("2026-04-30", "alpine", "A", 60_000.0),
            ("2026-04-30", "alpine", "A", 40_000.0),
        ])
        pos["market_value_stmt"] = [59_000.0, float("nan")]
        rep = build_health_report(
            pos,
            _summaries([("2026-04-30", "alpine", "A", 99_000.0, "s")]),
            today=TODAY,
        )
        a = _by_id(rep)["A"]
        self.assertEqual(a.extracted, 99_000.0)
        self.assertEqual(a.band, "ok")

    def test_marked_frame_reconciles_end_to_end(self):
        # THE regression: statement says $100k; live price fell 25% before the
        # dashboard loaded. mark_to_market rewrites market_value on the latest
        # snapshot; the health report must still reconcile to the statement.
        pos = pd.DataFrame([{
            "statement_date": "2026-06-30", "broker": "alpine",
            "account_id": "A", "symbol": "MEM", "quantity": 1000.0,
            "price": 100.0, "market_value": 100_000.0,
            "cost_basis": 90_000.0, "unrealized_gl": 10_000.0,
        }])
        pos["statement_date"] = pd.to_datetime(pos["statement_date"])
        marked = mark_to_market(
            pos, pd.DataFrame([{"symbol": "MEM", "close": 75.0,
                                "status": "ok"}]))
        # Sanity: the live mark really did move the frame -25%.
        self.assertAlmostEqual(float(marked["market_value"].sum()), 75_000.0)
        rep = build_health_report(
            marked,
            _summaries([("2026-06-30", "alpine", "A", 100_000.0, "s")]),
            today=TODAY,
        )
        a = _by_id(rep)["A"]
        self.assertEqual(a.state, "verified")
        self.assertEqual(a.band, "ok")
        self.assertEqual(a.extracted, 100_000.0)
        self.assertEqual(rep.worst_level, "green")


class TestFormatHealthHeadline(unittest.TestCase):
    def _rep(self, **kw):
        base = dict(as_of_month="2026-04", recon_available=True, accounts=[],
                    n_ok=0, n_known=0, n_watch=0, n_error=0, n_carried=0,
                    worst_level="green")
        base.update(kw)
        return HealthReport(**base)

    def test_grey_when_unavailable(self):
        level, text = format_health_headline(self._rep(recon_available=False, worst_level="grey"))
        self.assertEqual(level, "grey")
        self.assertIn("unavailable", text.lower())

    def test_green_all_reconcile(self):
        acc = [AccountHealth("A", "A", "alpine", "verified", False, "ok",
                             100.0, 100.0, 0.0, 0.0, "2026-04", 5)]
        level, text = format_health_headline(self._rep(accounts=acc, n_ok=1))
        self.assertEqual(level, "green")
        self.assertIn("Apr 2026", text)

    def test_red_when_error(self):
        level, text = format_health_headline(self._rep(n_error=1, worst_level="red"))
        self.assertEqual(level, "red")

    def test_amber_when_watch(self):
        level, text = format_health_headline(self._rep(n_watch=2, worst_level="amber"))
        self.assertEqual(level, "amber")
        self.assertIn("watch", text.lower())

    def test_amber_names_carried_account(self):
        acc = [AccountHealth("Z", "Roth IRA", "alpine", "carried", True, None,
                             None, None, None, None, "2026-03", 40)]
        level, text = format_health_headline(
            self._rep(accounts=acc, n_carried=1, worst_level="amber"))
        self.assertEqual(level, "amber")
        self.assertIn("Roth IRA", text)
        self.assertIn("Mar 2026", text)

    def test_amber_names_unreconciled_newer_month(self):
        level, text = format_health_headline(
            self._rep(unreconciled_months=["2026-05"], worst_level="amber"))
        self.assertEqual(level, "amber")
        self.assertIn("May 2026", text)
        self.assertIn("not yet reconciled", text)


class TestHealthRowsToTable(unittest.TestCase):
    def test_carried_row_has_dashes_verified_has_numbers(self):
        rep = HealthReport(
            as_of_month="2026-04", recon_available=True,
            accounts=[
                AccountHealth("A", "A", "alpine", "verified", False, "ok",
                              100_000.0, 100_000.0, 0.0, 0.0, "2026-04", 5),
                AccountHealth("C", "C", "alpine", "carried", True, None,
                              None, None, None, None, "2026-03", 40),
            ],
            n_ok=1, n_known=0, n_watch=0, n_error=0, n_carried=1,
            worst_level="amber",
        )
        rows = health_rows_to_table(rep)
        by_acct = {r["Account"]: r for r in rows}
        self.assertEqual(by_acct["A"]["State"], "Verified")
        self.assertEqual(by_acct["A"]["Verdict"], "ok")
        self.assertEqual(by_acct["C"]["State"], "Carried forward")
        self.assertEqual(by_acct["C"]["Extracted"], "—")
        self.assertEqual(by_acct["C"]["Δ%"], "—")


if __name__ == "__main__":
    unittest.main()
