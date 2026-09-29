"""Tests for parsers/total_return.py — total-return close adjustment.

Convention under test (spec 2026-08-22-total-return-basis-design §3): a
distribution with cash D going ex on a bar with close P_u is reinvested at
that close, so the day's total return is (P_u + D) / P_{u-1} - 1 and the
adjusted series is rebased so its LAST level equals the actual close.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from total_return import (  # noqa: E402
    apply_total_return,
    covered_symbols,
    load_distributions,
    load_splits,
    split_scale_cash,
    total_return_adjust,
)

FIXTURE = ROOT / "tests" / "fixtures" / "synth_data"


def _prices(closes: dict, dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.DataFrame(closes, index=idx)


def _dist(rows) -> pd.DataFrame:
    """rows: (symbol, ex_date, cash_amount[, pay_date])."""
    rows = [tuple(r) + (None,) * (4 - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=["symbol", "ex_date", "cash_amount", "pay_date"])
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["pay_date"] = pd.to_datetime(df["pay_date"])
    return df


DATES = ["2026-04-24", "2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"]


class TestAdjust(unittest.TestCase):
    def test_ex_date_return_is_total_return(self) -> None:
        # VISN: 19.53 -> 9.90 with a $10 distribution = +1.9 %, not -49 %.
        px = _prices({"VISN": [19.40, 19.53, 9.90, 10.47, 12.795]}, DATES)
        out = total_return_adjust(px, _dist([("VISN", "2026-04-28", 10.0)]))
        r = out["VISN"].pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-28"]),
                               (9.90 + 10.0) / 19.53 - 1.0, places=12)
        # Other days' returns are the price returns.
        self.assertAlmostEqual(float(r.loc["2026-04-29"]), 10.47 / 9.90 - 1.0, places=12)
        self.assertAlmostEqual(float(r.loc["2026-04-27"]), 19.53 / 19.40 - 1.0, places=12)

    def test_last_level_is_the_actual_close(self) -> None:
        px = _prices({"VISN": [19.40, 19.53, 9.90, 10.47, 12.795]}, DATES)
        out = total_return_adjust(px, _dist([("VISN", "2026-04-28", 10.0)]))
        self.assertEqual(float(out["VISN"].iloc[-1]), 12.795)
        # Levels strictly before the ex-date are scaled DOWN by 1/(1 + D/P_u).
        f = 1.0 / (1.0 + 10.0 / 9.90)
        self.assertAlmostEqual(float(out["VISN"].loc["2026-04-27"]), 19.53 * f, places=10)
        self.assertAlmostEqual(float(out["VISN"].loc["2026-04-24"]), 19.40 * f, places=10)
        self.assertEqual(list(out.columns), ["VISN"])
        self.assertTrue(out.index.equals(px.index))

    def test_non_payer_and_uncovered_columns_unchanged(self) -> None:
        px = _prices({"AAA": [10, 11, 12, 13, 14], "BBB": [5, 5, 5, 5, 5]}, DATES)
        out = total_return_adjust(px, _dist([("AAA", "2026-04-28", 0.5)]))
        pd.testing.assert_series_equal(out["BBB"], px["BBB"])
        self.assertEqual(out.attrs["total_return"]["adjusted"], ["AAA"])

    def test_nan_bars_preserved_and_prev_valid_bar_used(self) -> None:
        px = _prices({"AAA": [10.0, float("nan"), 12.0, 13.0, 14.0]}, DATES)
        out = total_return_adjust(px, _dist([("AAA", "2026-04-28", 1.0)]))
        self.assertTrue(pd.isna(out["AAA"].loc["2026-04-27"]))
        # prev valid close is 04-24's 10.0
        r = out["AAA"].dropna().pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-28"]), (12.0 + 1.0) / 10.0 - 1.0, places=12)

    def test_ex_date_on_non_trading_day_applies_on_next_bar(self) -> None:
        # 04-25/26 are a weekend; ex 04-26 lands on the 04-27 bar.
        px = _prices({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, DATES)
        out = total_return_adjust(px, _dist([("AAA", "2026-04-26", 1.0)]))
        r = out["AAA"].pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-27"]), (11.0 + 1.0) / 10.0 - 1.0, places=12)

    def test_distribution_outside_window_ignored(self) -> None:
        px = _prices({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, DATES)
        out = total_return_adjust(px, _dist([("AAA", "2026-04-24", 1.0),    # first bar: no prior
                                            ("AAA", "2026-05-15", 1.0)]))   # after last bar
        pd.testing.assert_frame_equal(out, px, check_names=False)
        self.assertEqual(out.attrs["total_return"]["adjusted"], [])

    def test_impossible_distribution_skipped_and_recorded(self) -> None:
        px = _prices({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, DATES)
        out = total_return_adjust(px, _dist([("AAA", "2026-04-28", 11.0)]))   # >= prev close
        pd.testing.assert_frame_equal(out, px, check_names=False)
        skipped = out.attrs["total_return"]["skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][0], "AAA")

    def test_two_distributions_on_one_bar_sum(self) -> None:
        px = _prices({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, DATES)
        out = total_return_adjust(px, _dist([("AAA", "2026-04-28", 0.4),
                                            ("AAA", "2026-04-28", 0.6)]))
        r = out["AAA"].pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-28"]), (12.0 + 1.0) / 11.0 - 1.0, places=12)

    # ---- due-bill rule (large distributions) ------------------------------
    def test_large_distribution_without_price_drop_is_skipped(self) -> None:
        # VISN Aug 2026: Polygon listed a $5 ex-date on a $11.53 stock while
        # the price went UP — applying it would fabricate +44 %.
        px = _prices({"VISN": [11.35, 11.53, 11.66, 11.38, 11.20]}, DATES)
        out = total_return_adjust(px, _dist([("VISN", "2026-04-28", 5.0, "2026-05-08")]))
        pd.testing.assert_frame_equal(out, px, check_names=False)
        skipped = out.attrs["total_return"]["skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][0], "VISN")
        self.assertEqual(skipped[0][-1], "unconfirmed_large")

    def test_large_distribution_applies_after_pay_date_when_confirmed_there(self) -> None:
        # Listed ex 04-27 (no drop), pay 04-28 → the true ex-date is the next
        # bar, 04-29, where the drop shows. Reinvested there.
        px = _prices({"XYZ": [20.0, 20.2, 20.1, 15.2, 15.4]}, DATES)
        out = total_return_adjust(px, _dist([("XYZ", "2026-04-27", 5.0, "2026-04-28")]))
        r = out["XYZ"].pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-29"]), (15.2 + 5.0) / 20.1 - 1.0, places=12)
        self.assertAlmostEqual(float(r.loc["2026-04-27"]), 20.2 / 20.0 - 1.0, places=12)
        self.assertEqual(out.attrs["total_return"]["skipped"], [])

    def test_large_distribution_confirmed_on_listed_ex_date(self) -> None:
        # VISN Apr 2026: pay 04-27, ex 04-28 = pay + 1 — the drop is there.
        px = _prices({"VISN": [19.40, 19.53, 9.90, 10.47, 12.795]}, DATES)
        out = total_return_adjust(px, _dist([("VISN", "2026-04-28", 10.0, "2026-04-27")]))
        r = out["VISN"].pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-28"]), (9.90 + 10.0) / 19.53 - 1.0, places=12)

    def test_small_distribution_applies_regardless_of_price_move(self) -> None:
        px = _prices({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, DATES)   # rising through ex
        out = total_return_adjust(px, _dist([("AAA", "2026-04-28", 0.5)]))
        r = out["AAA"].pct_change()
        self.assertAlmostEqual(float(r.loc["2026-04-28"]), (12.0 + 0.5) / 11.0 - 1.0, places=12)

    def test_matches_the_spy_tr_builder_rule(self) -> None:
        # shares += cash / close on the ex-date (build_benchmark_total_return)
        # must give the same daily returns as the adjusted series.
        closes = [100.0, 102.0, 101.0, 103.0, 104.0]
        px = _prices({"SPY": closes}, DATES)
        out = total_return_adjust(px, _dist([("SPY", "2026-04-28", 1.5)]))
        shares, values = 1.0, []
        for d, c in zip(DATES, closes):
            if d == "2026-04-28":
                shares += shares * 1.5 / c
            values.append(shares * c)
        tr = pd.Series(values, index=px.index).pct_change()
        pd.testing.assert_series_equal(out["SPY"].pct_change(), tr, check_names=False)


class TestSplitScaling(unittest.TestCase):
    def test_pre_split_dividend_scaled_post_split_untouched(self) -> None:
        dist = _dist([("AVGO", "2024-06-24", 5.25), ("AVGO", "2024-09-19", 0.53)])
        splits = pd.DataFrame([{"symbol": "AVGO",
                                "execution_date": pd.Timestamp("2024-07-15"),
                                "ratio": 10.0}])
        out = split_scale_cash(dist, splits)
        by = out.set_index("ex_date")["cash_amount"]
        self.assertAlmostEqual(float(by.loc["2024-06-24"]), 0.525, places=12)
        self.assertAlmostEqual(float(by.loc["2024-09-19"]), 0.53, places=12)

    def test_reverse_split_scales_up_and_cumulative(self) -> None:
        dist = _dist([("XYZ", "2020-01-10", 1.0)])
        splits = pd.DataFrame([
            {"symbol": "XYZ", "execution_date": pd.Timestamp("2021-01-01"), "ratio": 0.1},   # 10->1
            {"symbol": "XYZ", "execution_date": pd.Timestamp("2022-01-01"), "ratio": 2.0},   # 1->2
        ])
        out = split_scale_cash(dist, splits)
        self.assertAlmostEqual(float(out["cash_amount"].iloc[0]), 1.0 / (0.1 * 2.0), places=12)

    def test_no_splits_frame_is_identity(self) -> None:
        dist = _dist([("AAA", "2024-06-24", 1.0)])
        out = split_scale_cash(dist, pd.DataFrame(columns=["symbol", "execution_date", "ratio"]))
        pd.testing.assert_frame_equal(out, dist)


class TestLoaders(unittest.TestCase):
    def _tmp_with_dividends(self, tmp: Path, rows, splits=None) -> None:
        # Copy the fixture WITHOUT its dividend files (it ships dividends_spy.csv
        # for the benchmark-TR / income tests) so each case controls coverage.
        for csv in FIXTURE.glob("*.csv"):
            if csv.name.startswith("dividends_"):
                continue
            shutil.copy2(csv, tmp / csv.name)
        for sym, ex, cash in rows:
            p = tmp / f"dividends_{sym.lower()}.csv"
            pd.DataFrame([{"cash_amount": cash, "currency": "USD",
                           "dividend_type": "CD", "ex_dividend_date": ex,
                           "frequency": 4, "id": "x", "pay_date": ex,
                           "record_date": ex, "ticker": sym}]).to_csv(p, index=False)
        if splits is not None:
            pd.DataFrame(splits, columns=["symbol", "execution_date", "split_from",
                                          "split_to"]).to_csv(tmp / "splits.csv", index=False)

    def test_load_distributions_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._tmp_with_dividends(tmp, [("AAA", "2026-03-10", 0.5)])
            # header-only non-payer file
            pd.DataFrame(columns=["cash_amount", "ex_dividend_date", "ticker"]).to_csv(
                tmp / "dividends_bbb.csv", index=False)
            dist = load_distributions(tmp)
            self.assertEqual(list(dist.columns), ["symbol", "ex_date", "cash_amount", "pay_date"])
            self.assertEqual(dist["symbol"].tolist(), ["AAA"])
            self.assertEqual(str(dist["pay_date"].iloc[0].date()), "2026-03-10")
            self.assertEqual(covered_symbols(tmp), {"AAA", "BBB"})

    def test_load_splits_absent_and_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self.assertTrue(load_splits(tmp).empty)
            self._tmp_with_dividends(tmp, [], splits=[("AVGO", "2024-07-15", 1, 10)])
            s = load_splits(tmp)
            self.assertEqual(float(s["ratio"].iloc[0]), 10.0)

    def test_apply_total_return_is_noop_without_dividend_files(self) -> None:
        from terminal import holdings_service as hs
        raw = pd.read_csv(FIXTURE / "daily_prices.csv", parse_dates=["date"]).pivot(
            index="date", columns="symbol", values="close").sort_index()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._tmp_with_dividends(tmp, [])            # fixture minus dividend files
            out = apply_total_return(raw, tmp)
            pd.testing.assert_frame_equal(out, raw)
            self.assertEqual(out.attrs["total_return"]["adjusted"], [])
            self.assertEqual(out.attrs["total_return"]["covered"], 0)
            # and the loader's matrix is the same price matrix
            frames = hs.load_frames(tmp)
        pd.testing.assert_frame_equal(frames.daily_prices[raw.columns], raw)

    def test_fixture_adjusts_spy_only(self) -> None:
        # The committed fixture ships dividends_spy.csv (benchmark-TR / income
        # tests) with ex-dates inside its price window, so SPY IS adjusted
        # there — which is why the risk-family goldens moved when this landed.
        # AAA / BBB have no file and stay price-only.
        raw = pd.read_csv(FIXTURE / "daily_prices.csv", parse_dates=["date"]).pivot(
            index="date", columns="symbol", values="close").sort_index()
        out = apply_total_return(raw, FIXTURE)
        info = out.attrs["total_return"]
        self.assertEqual(info["adjusted"], ["SPY"])
        # dividends_aaa.csv is a header-only non-payer file: covered, nothing
        # to apply; BBB has no file at all.
        self.assertEqual(info["uncovered"], ["BBB"])
        self.assertEqual(info["covered"], 2)
        pd.testing.assert_series_equal(out["AAA"], raw["AAA"])
        self.assertEqual(float(out["SPY"].dropna().iloc[-1]), float(raw["SPY"].dropna().iloc[-1]))
        # Levels before the last ex-date (2026-03-20) are scaled down; from
        # that bar on the backward factor is 1 and levels equal the raw close.
        last_ex = pd.Timestamp("2026-03-20")
        before = out.index < last_ex
        self.assertTrue((out.loc[before, "SPY"] < raw.loc[before, "SPY"]).all())
        pd.testing.assert_series_equal(out.loc[~before, "SPY"], raw.loc[~before, "SPY"])

    def test_loader_applies_dividend_from_temp_data_dir(self) -> None:
        from terminal import holdings_service as hs
        raw = pd.read_csv(FIXTURE / "daily_prices.csv", parse_dates=["date"])
        aaa = raw[raw["symbol"] == "AAA"].sort_values("date")
        self.assertGreater(len(aaa), 3, "fixture needs AAA daily prices")
        ex_row = aaa.iloc[2]
        prev_close = float(aaa.iloc[1]["close"])
        ex_date = pd.Timestamp(ex_row["date"]).strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._tmp_with_dividends(tmp, [("AAA", ex_date, 0.25)])
            frames = hs.load_frames(tmp)
        col = frames.daily_prices["AAA"]
        r = col.pct_change()
        self.assertAlmostEqual(float(r.loc[ex_date]),
                               (float(ex_row["close"]) + 0.25) / prev_close - 1.0, places=12)
        self.assertEqual(float(col.dropna().iloc[-1]), float(aaa.iloc[-1]["close"]))
        self.assertIn("AAA", frames.daily_prices.attrs["total_return"]["adjusted"])


if __name__ == "__main__":
    unittest.main()
