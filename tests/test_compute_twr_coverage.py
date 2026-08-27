"""Tests for the Fidelity coverage helpers in parsers/compute_twr.py.

`_load_fidelity_coverage` and `_account_covered_in_month` are the load-bearing
helpers behind the "is this gap month genuinely missing, or just rolled into
the next PDF?" distinction surfaced on the Performance tab. They shipped in
PR #31 without dedicated coverage; this file fills that gap.

Spanning rule (paraphrased from the helper docstring): a coverage row spans
`month` only if `period_start <= month_start` AND `period_end >= month_end`.
Real-world example: an account that opened mid-Dec 2025 (period_start =
2025-12-12), so Dec 2025 is NOT covered end-to-end and should return False —
the audit finding that #31 was built to surface.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import compute_twr  # noqa: E402
from compute_twr import (  # noqa: E402
    _account_covered_in_month,
    _load_fidelity_coverage,
)


def _coverage(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a coverage frame from (account_id, period_start, period_end)
    triples. Mirrors the schema written by fidelity_txn_parser."""
    return pd.DataFrame(
        [{"broker": "fidelity", "account_id": acct,
          "period_start": pd.Timestamp(ps), "period_end": pd.Timestamp(pe),
          "source_file": "Statement.pdf"}
         for acct, ps, pe in rows]
    )


class TestLoadFidelityCoverage(unittest.TestCase):
    def test_returns_empty_frame_when_csv_missing(self) -> None:
        with TemporaryDirectory() as td:
            missing = Path(td) / "no_such_file.csv"
            with patch.object(compute_twr, "FIDELITY_COVERAGE_CSV", missing):
                df = _load_fidelity_coverage()
        self.assertTrue(df.empty)
        self.assertEqual(
            list(df.columns),
            ["broker", "account_id", "period_start", "period_end", "source_file"],
        )

    def test_loads_csv_when_present(self) -> None:
        with TemporaryDirectory() as td:
            csv = Path(td) / "coverage.csv"
            pd.DataFrame([
                {"broker": "fidelity", "account_id": "X10-000007",
                 "period_start": "2026-02-01", "period_end": "2026-03-31",
                 "source_file": "Statement3312026.pdf"},
            ]).to_csv(csv, index=False)
            with patch.object(compute_twr, "FIDELITY_COVERAGE_CSV", csv):
                df = _load_fidelity_coverage()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["account_id"], "X10-000007")

    def test_period_columns_parsed_as_datetime(self) -> None:
        # The spanning comparison in _account_covered_in_month relies on
        # period_start/period_end being Timestamps, not strings.
        with TemporaryDirectory() as td:
            csv = Path(td) / "coverage.csv"
            pd.DataFrame([
                {"broker": "fidelity", "account_id": "X10-000007",
                 "period_start": "2026-02-01", "period_end": "2026-02-28",
                 "source_file": "Statement2282026.pdf"},
            ]).to_csv(csv, index=False)
            with patch.object(compute_twr, "FIDELITY_COVERAGE_CSV", csv):
                df = _load_fidelity_coverage()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["period_start"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["period_end"]))


class TestAccountCoveredInMonth(unittest.TestCase):
    def test_returns_false_when_coverage_empty(self) -> None:
        empty = pd.DataFrame(columns=[
            "broker", "account_id", "period_start", "period_end", "source_file",
        ])
        self.assertFalse(
            _account_covered_in_month("X10-000007", pd.Period("2026-02"), empty)
        )

    def test_returns_false_when_account_not_in_coverage(self) -> None:
        cov = _coverage([("Z10-000008", "2026-02-01", "2026-02-28")])
        self.assertFalse(
            _account_covered_in_month("X10-000007", pd.Period("2026-02"), cov)
        )

    def test_single_month_statement_covers_month(self) -> None:
        # Normal case: period == month. Helper docstring acknowledges this
        # returns True even though the caller only consults the helper for
        # forward-filled months (which have no positions row); coverage of
        # a single month then means "parser saw a statement the user didn't
        # ingest into positions.csv".
        cov = _coverage([("X10-000007", "2026-02-01", "2026-02-28")])
        self.assertTrue(
            _account_covered_in_month("X10-000007", pd.Period("2026-02"), cov)
        )

    def test_combined_statement_covers_first_month(self) -> None:
        # X10-000007's Feb 2026 PDF was rolled into the Mar 2026 combined
        # statement: period_start = 2026-02-01, period_end = 2026-03-31 spans
        # both. The Feb 2026 gap on the Performance tab gets the quieter
        # "rolled into next PDF" annotation because of this branch.
        cov = _coverage([("X10-000007", "2026-02-01", "2026-03-31")])
        self.assertTrue(
            _account_covered_in_month("X10-000007", pd.Period("2026-02"), cov)
        )
        self.assertTrue(
            _account_covered_in_month("X10-000007", pd.Period("2026-03"), cov)
        )

    def test_partial_month_statement_not_covered(self) -> None:
        # Z10-000008 opened on 2025-12-12. Dec 2025's "statement" is the stub
        # from account open to month-end — it does NOT span Dec 1 to Dec 31.
        # The helper must return False, which lets the Performance tab flag
        # the partial-month coverage rather than silently report a full-month
        # return for Dec 2025.
        cov = _coverage([("Z10-000008", "2025-12-12", "2025-12-31")])
        self.assertFalse(
            _account_covered_in_month("Z10-000008", pd.Period("2025-12"), cov)
        )

    def test_month_after_all_periods_not_covered(self) -> None:
        cov = _coverage([("X10-000007", "2026-01-01", "2026-04-30")])
        self.assertFalse(
            _account_covered_in_month("X10-000007", pd.Period("2026-05"), cov)
        )

    def test_uses_any_when_multiple_periods_overlap_month(self) -> None:
        # Two statements both nominally cover Feb 2026 — the helper should
        # OR them (.any()), not insist on a unique match.
        cov = _coverage([
            ("X10-000007", "2026-01-01", "2026-02-28"),
            ("X10-000007", "2026-02-01", "2026-03-31"),
        ])
        self.assertTrue(
            _account_covered_in_month("X10-000007", pd.Period("2026-02"), cov)
        )


if __name__ == "__main__":
    unittest.main()
