"""
Tests for parsers/whatif_data.py — candidate ticker fetch + proxy splice.

Covers:
  - fetch_candidate_history: provider-driven happy path, empty bars,
    stale-cache fallback, fresh-cache short-circuit.
  - splice_with_proxy: empty inputs, no pre-candidate overlap, normal
    rebase math (return continuity), candidate-overlap-day precedence.
  - build_augmented_price_matrix: empty/non-empty combinations and
    column alignment to daily_prices index.

Run from phase1_build/ with:
    py -m unittest discover tests
"""
import os
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import whatif_data as wd  # noqa: E402


def _bdays(n: int, end: str = "2026-04-30") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


class TestFetchCandidateHistory(unittest.TestCase):
    def test_provider_returns_bars_writes_cache(self) -> None:
        bars = [
            {"date": "2025-01-02", "close": 14.10},
            {"date": "2025-01-03", "close": 14.25},
            {"date": "2025-01-06", "close": 14.30},
        ]
        called = {"count": 0}

        def provider(ticker, start, end):
            called["count"] += 1
            return bars

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            series = wd.fetch_candidate_history(
                "PDBC", cache_dir=cache_dir, bars_provider=provider
            )
            self.assertEqual(called["count"], 1)
            self.assertEqual(len(series), 3)
            self.assertEqual(series.name, "PDBC")
            self.assertAlmostEqual(float(series.iloc[0]), 14.10, places=6)
            # Cache written
            self.assertTrue((cache_dir / "PDBC.csv").exists())

    def test_actions_provider_makes_candidate_total_return(self) -> None:
        # Total-return basis (spec 2026-08-22): a $0.50 distribution going ex
        # on the 14.25 bar is reinvested at that close; the last level stays
        # the real close; the pre-split $2.00 dividend is scaled by the 1->2.
        bars = [
            {"date": "2025-01-02", "close": 14.10},
            {"date": "2025-01-03", "close": 14.25},
            {"date": "2025-01-06", "close": 14.30},
        ]

        def actions(ticker, start, end):
            return ([{"ex_dividend_date": "2025-01-03", "cash_amount": 0.5}], [])

        with tempfile.TemporaryDirectory() as tmp:
            series = wd.fetch_candidate_history(
                "PDBC", cache_dir=Path(tmp), bars_provider=lambda t, s, e: bars,
                actions_provider=actions)
            r = series.pct_change()
            self.assertAlmostEqual(float(r.iloc[1]), (14.25 + 0.5) / 14.10 - 1.0, places=12)
            self.assertAlmostEqual(float(series.iloc[-1]), 14.30, places=12)
            # the cache holds the adjusted series, so a fresh read agrees
            again = wd.fetch_candidate_history("PDBC", cache_dir=Path(tmp),
                                               bars_provider=lambda t, s, e: [])
            self.assertAlmostEqual(float(again.iloc[0]), float(series.iloc[0]), places=12)

    def test_injected_bars_without_actions_stay_price_only(self) -> None:
        bars = [{"date": "2025-01-02", "close": 10.0}, {"date": "2025-01-03", "close": 11.0}]
        with tempfile.TemporaryDirectory() as tmp:
            series = wd.fetch_candidate_history(
                "PDBC", cache_dir=Path(tmp), bars_provider=lambda t, s, e: bars)
            self.assertAlmostEqual(float(series.iloc[0]), 10.0, places=12)

    def test_empty_bars_returns_empty_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            series = wd.fetch_candidate_history(
                "ZZZZ", cache_dir=Path(tmp),
                bars_provider=lambda t, s, e: [],
            )
            self.assertTrue(series.empty)
            self.assertEqual(series.name, "ZZZZ")

    def test_fresh_cache_short_circuits_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            seed = pd.Series(
                [10.0, 10.5, 10.9],
                index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
                name="close",
            )
            wd._write_cache(seed, cache_dir / "PDBC.csv")

            def boom(*a, **k):
                raise AssertionError("provider should not be called on fresh cache")

            series = wd.fetch_candidate_history(
                "PDBC", cache_dir=cache_dir, bars_provider=boom
            )
            self.assertEqual(len(series), 3)
            self.assertEqual(series.name, "PDBC")

    def test_stale_cache_used_when_provider_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache_path = cache_dir / "PDBC.csv"
            seed = pd.Series(
                [9.5, 9.6],
                index=pd.to_datetime(["2024-12-30", "2024-12-31"]),
                name="close",
            )
            wd._write_cache(seed, cache_path)
            # Backdate the cache far enough to be stale.
            stale = time.time() - (wd.CACHE_TTL_DAYS + 1) * 86400
            os.utime(cache_path, (stale, stale))

            series = wd.fetch_candidate_history(
                "PDBC", cache_dir=cache_dir,
                bars_provider=lambda t, s, e: [],
            )
            self.assertEqual(len(series), 2)
            self.assertEqual(series.name, "PDBC")


class TestSpliceWithProxy(unittest.TestCase):
    def test_empty_candidate_returns_empty(self) -> None:
        proxy = pd.Series([1.0, 1.1], index=_bdays(2), name="GSG")
        out = wd.splice_with_proxy(pd.Series(dtype=float, name="PDBC"), proxy)
        self.assertTrue(out.empty)

    def test_empty_proxy_returns_candidate_unchanged(self) -> None:
        cand_idx = _bdays(3)
        cand = pd.Series([10.0, 10.5, 11.0], index=cand_idx, name="PDBC")
        out = wd.splice_with_proxy(cand, pd.Series(dtype=float, name="GSG"))
        pd.testing.assert_series_equal(out, cand)

    def test_proxy_with_no_pre_candidate_data_returns_candidate(self) -> None:
        cand_idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
        cand = pd.Series([10.0, 10.2], index=cand_idx, name="PDBC")
        # Proxy starts AFTER candidate.
        proxy_idx = pd.to_datetime(["2025-01-06", "2025-01-07"])
        proxy = pd.Series([20.0, 20.1], index=proxy_idx, name="GSG")
        out = wd.splice_with_proxy(cand, proxy)
        pd.testing.assert_series_equal(out, cand)

    def test_normal_rebase_preserves_candidate_first_level(self) -> None:
        # Candidate starts 2025-01-06 at 100. Proxy extends back to
        # 2025-01-02. Proxy must be rebased so the spliced series passes
        # through 100 on 2025-01-06.
        cand_idx = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"])
        cand = pd.Series([100.0, 101.0, 99.5], index=cand_idx, name="PDBC")
        proxy_idx = pd.to_datetime([
            "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
        ])
        proxy = pd.Series([50.0, 51.0, 52.0, 53.0], index=proxy_idx,
                          name="GSG")
        out = wd.splice_with_proxy(cand, proxy)
        # Pre-inception values rebased by scale = 100/52.
        scale = 100.0 / 52.0
        self.assertAlmostEqual(
            float(out.loc[pd.Timestamp("2025-01-02")]),
            50.0 * scale, places=6,
        )
        self.assertAlmostEqual(
            float(out.loc[pd.Timestamp("2025-01-03")]),
            51.0 * scale, places=6,
        )
        # Candidate values untouched.
        self.assertEqual(float(out.loc[pd.Timestamp("2025-01-06")]), 100.0)
        self.assertEqual(float(out.loc[pd.Timestamp("2025-01-07")]), 101.0)
        self.assertEqual(float(out.loc[pd.Timestamp("2025-01-08")]), 99.5)

    def test_pre_inception_returns_match_proxy_returns(self) -> None:
        # The whole point of this splice: daily returns in the
        # pre-inception window equal proxy's actual daily returns.
        cand_idx = pd.to_datetime(["2025-01-06", "2025-01-07"])
        cand = pd.Series([100.0, 101.0], index=cand_idx, name="PDBC")
        proxy_idx = pd.to_datetime([
            "2025-01-02", "2025-01-03", "2025-01-06",
        ])
        proxy = pd.Series([50.0, 51.0, 52.0], index=proxy_idx, name="GSG")
        spliced = wd.splice_with_proxy(cand, proxy)
        proxy_ret_jan3 = 51.0 / 50.0 - 1.0
        # Spliced jan3 / spliced jan2 should equal proxy_ret_jan3 + 1.
        spliced_ret_jan3 = (
            float(spliced.loc[pd.Timestamp("2025-01-03")])
            / float(spliced.loc[pd.Timestamp("2025-01-02")])
            - 1.0
        )
        self.assertAlmostEqual(spliced_ret_jan3, proxy_ret_jan3, places=10)


class TestBuildAugmentedPriceMatrix(unittest.TestCase):
    def test_empty_daily_prices_creates_one_column_frame(self) -> None:
        cand = pd.Series(
            [10.0, 10.5],
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
            name="PDBC",
        )
        out = wd.build_augmented_price_matrix(pd.DataFrame(), "PDBC", cand)
        self.assertEqual(list(out.columns), ["PDBC"])
        self.assertEqual(len(out), 2)

    def test_empty_candidate_adds_nan_column(self) -> None:
        idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
        dp = pd.DataFrame({"VTI": [200.0, 201.0]}, index=idx)
        out = wd.build_augmented_price_matrix(
            dp, "PDBC", pd.Series(dtype=float, name="PDBC")
        )
        self.assertIn("PDBC", out.columns)
        self.assertTrue(out["PDBC"].isna().all())
        # Original column untouched.
        pd.testing.assert_series_equal(out["VTI"], dp["VTI"])

    def test_candidate_joined_on_daily_prices_index(self) -> None:
        idx = pd.to_datetime([
            "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
        ])
        dp = pd.DataFrame({"VTI": [200.0, 201.0, 202.0, 203.0]}, index=idx)
        # Candidate only has 3 of the 4 dates.
        cand = pd.Series(
            [10.0, 10.5, 11.0],
            index=pd.to_datetime(["2025-01-02", "2025-01-06", "2025-01-07"]),
            name="PDBC",
        )
        out = wd.build_augmented_price_matrix(dp, "PDBC", cand)
        self.assertEqual(list(out.columns), ["VTI", "PDBC"])
        self.assertEqual(float(out["PDBC"].iloc[0]), 10.0)
        self.assertTrue(np.isnan(float(out["PDBC"].iloc[1])))  # 2025-01-03 missing
        self.assertEqual(float(out["PDBC"].iloc[2]), 10.5)
        self.assertEqual(float(out["PDBC"].iloc[3]), 11.0)


class TestBuildMultiAugmentedPriceMatrix(unittest.TestCase):
    def test_splices_all_columns(self):
        idx = pd.date_range("2022-01-03", periods=5, freq="B")
        base = pd.DataFrame({"AAA": np.arange(5.0)}, index=idx)
        s1 = pd.Series([10.0, 11, 12, 13, 14], index=idx, name="NEWC")
        s2 = pd.Series([20.0, 21, 22], index=idx[:3], name="PXY")
        out = wd.build_multi_augmented_price_matrix(base, {"NEWC": s1, "PXY": s2})
        self.assertEqual(list(out.columns), ["AAA", "NEWC", "PXY"])
        self.assertEqual(out["NEWC"].tolist(), [10, 11, 12, 13, 14])
        # short candidate gets NaN on the days it has no bar
        self.assertTrue(bool(out["PXY"].iloc[3:].isna().all()))
        # empty dict is a no-op passthrough
        self.assertTrue(wd.build_multi_augmented_price_matrix(base, {}).equals(base))


if __name__ == "__main__":
    unittest.main()
