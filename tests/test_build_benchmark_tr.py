"""Tests for parsers/build_benchmark_total_return.py::build_tr.

Constructs a dividend-reinvested total-return series from a price CSV and a
dividends CSV. The May 2026 audit caught a stale comparison_spy.csv hiding
~4 pp/yr of underperformance — and the TR builder itself had no unit tests
before this. Pin the reinvestment math, the share accumulation, and the
ticker -> filename mapping.

`build_tr` reads from `DATA_DIR / "benchmark_<ticker>.csv"` and
`DATA_DIR / "dividends_<ticker>.csv"`, so the tests patch the module-level
DATA_DIR to a tempdir and write fixture CSVs there.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import build_benchmark_total_return as btr  # noqa: E402
from build_benchmark_total_return import build_tr  # noqa: E402
from build_benchmark_total_return import build_blended_tr  # noqa: E402


def _write_fixture(td: Path, ticker: str,
                   prices: list[tuple[str, float]],
                   dividends: list[tuple[str, float]]) -> None:
    pd.DataFrame(prices, columns=["date", "close"]).to_csv(
        td / f"benchmark_{ticker.lower()}.csv", index=False)
    pd.DataFrame(dividends, columns=["ex_dividend_date", "cash_amount"]).to_csv(
        td / f"dividends_{ticker.lower()}.csv", index=False)


class TestBuildTr(unittest.TestCase):
    def test_no_dividends_tr_value_tracks_close(self) -> None:
        with TemporaryDirectory() as td:
            _write_fixture(Path(td), "SPY",
                           prices=[("2026-01-02", 100.0),
                                   ("2026-01-03", 101.0),
                                   ("2026-01-04", 102.0)],
                           dividends=[])
            with patch.object(btr, "DATA_DIR", Path(td)):
                df = build_tr("SPY")
        # With no dividends, shares stays at 1.0 and tr_value == close.
        self.assertEqual(list(df["shares"]), [1.0, 1.0, 1.0])
        self.assertEqual(list(df["tr_value"]), [100.0, 101.0, 102.0])

    def test_dividend_reinvested_grows_shares(self) -> None:
        # On 2026-01-03: cash = 1.0 * $2.00 = $2.00 reinvested at close $100
        # -> shares += 0.02 -> 1.02. tr_value = 1.02 * 100 = $102.
        # On 2026-01-04: tr_value = 1.02 * 105 = $107.10.
        with TemporaryDirectory() as td:
            _write_fixture(Path(td), "SPY",
                           prices=[("2026-01-02", 100.0),
                                   ("2026-01-03", 100.0),
                                   ("2026-01-04", 105.0)],
                           dividends=[("2026-01-03", 2.0)])
            with patch.object(btr, "DATA_DIR", Path(td)):
                df = build_tr("SPY")
        self.assertAlmostEqual(df["shares"].iloc[0], 1.0)
        self.assertAlmostEqual(df["shares"].iloc[1], 1.02)
        self.assertAlmostEqual(df["shares"].iloc[2], 1.02)
        self.assertAlmostEqual(df["tr_value"].iloc[1], 102.0)
        self.assertAlmostEqual(df["tr_value"].iloc[2], 107.10)

    def test_tr_index_normalized_to_100_at_start(self) -> None:
        with TemporaryDirectory() as td:
            _write_fixture(Path(td), "SPY",
                           prices=[("2026-01-02", 250.0),
                                   ("2026-01-03", 275.0)],
                           dividends=[])
            with patch.object(btr, "DATA_DIR", Path(td)):
                df = build_tr("SPY")
        self.assertAlmostEqual(df["tr_index"].iloc[0], 100.0)
        self.assertAlmostEqual(df["tr_index"].iloc[1], 110.0)  # 275/250 * 100

    def test_daily_return_matches_pct_change(self) -> None:
        with TemporaryDirectory() as td:
            _write_fixture(Path(td), "SPY",
                           prices=[("2026-01-02", 100.0),
                                   ("2026-01-03", 110.0),
                                   ("2026-01-04", 99.0)],
                           dividends=[])
            with patch.object(btr, "DATA_DIR", Path(td)):
                df = build_tr("SPY")
        # First row's pct_change is NaN by definition.
        self.assertTrue(pd.isna(df["daily_return"].iloc[0]))
        self.assertAlmostEqual(df["daily_return"].iloc[1], 0.10)
        self.assertAlmostEqual(df["daily_return"].iloc[2], -0.10)

    def test_multiple_dividends_same_day_are_summed(self) -> None:
        # Two dividend rows on the same ex-date — fetch_dividends sometimes
        # emits split rows for special distributions. The groupby+sum in
        # build_tr must aggregate them before reinvestment.
        with TemporaryDirectory() as td:
            _write_fixture(Path(td), "SPY",
                           prices=[("2026-01-02", 100.0),
                                   ("2026-01-03", 100.0)],
                           dividends=[("2026-01-03", 1.0),
                                      ("2026-01-03", 1.5)])
            with patch.object(btr, "DATA_DIR", Path(td)):
                df = build_tr("SPY")
        # Combined dividend $2.50 reinvested at $100 -> shares 1.025.
        self.assertAlmostEqual(df["shares"].iloc[1], 1.025)

    def test_ticker_argument_is_lowercased_for_filenames(self) -> None:
        # build_tr should read benchmark_spy.csv even when called with "SPY".
        with TemporaryDirectory() as td:
            _write_fixture(Path(td), "spy",  # lowercase fixture filename
                           prices=[("2026-01-02", 100.0)],
                           dividends=[])
            with patch.object(btr, "DATA_DIR", Path(td)):
                df = build_tr("SPY")
        self.assertEqual(len(df), 1)


def _tr(dates, values):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tr_value": values})
    return df


class TestBuildBlendedTr(unittest.TestCase):
    def test_all_weight_on_one_leg_reproduces_that_leg(self):
        spy = _tr(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 110.0, 121.0])
        agg = _tr(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 100.0])
        out = build_blended_tr([(1.0, spy), (0.0, agg)])
        # 100% SPY: index tracks SPY's returns exactly (rebased to 100)
        self.assertAlmostEqual(out["tr_index"].iloc[0], 100.0, places=6)
        self.assertAlmostEqual(out["tr_index"].iloc[-1], 121.0, places=6)

    def test_two_day_sixty_forty_blend_by_hand(self):
        # day1 returns: SPY +10%, AGG 0%.  60/40 -> +6%.
        spy = _tr(["2024-01-02", "2024-01-03"], [100.0, 110.0])
        agg = _tr(["2024-01-02", "2024-01-03"], [100.0, 100.0])
        out = build_blended_tr([(0.6, spy), (0.4, agg)])
        self.assertAlmostEqual(out["daily_return"].iloc[1], 0.06, places=9)
        self.assertAlmostEqual(out["tr_index"].iloc[-1], 106.0, places=6)

    def test_inner_aligns_on_common_dates(self):
        spy = _tr(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 110.0, 120.0])
        agg = _tr(["2024-01-03", "2024-01-04", "2024-01-05"], [100.0, 100.0, 100.0])
        out = build_blended_tr([(0.6, spy), (0.4, agg)])
        # common dates are 01-03 and 01-04 only
        self.assertEqual(len(out), 2)
        self.assertEqual(pd.Timestamp(out["date"].iloc[0]), pd.Timestamp("2024-01-03"))

    def test_weights_must_sum_to_one(self):
        spy = _tr(["2024-01-02", "2024-01-03"], [100.0, 110.0])
        with self.assertRaises(ValueError):
            build_blended_tr([(0.6, spy), (0.3, spy)])

    def test_empty_component_returns_empty(self):
        spy = _tr(["2024-01-02", "2024-01-03"], [100.0, 110.0])
        empty = pd.DataFrame(columns=["date", "tr_value"])
        self.assertTrue(build_blended_tr([(0.6, spy), (0.4, empty)]).empty)


if __name__ == "__main__":
    unittest.main()
