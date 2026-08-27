import tempfile
import unittest
from pathlib import Path

import pandas as pd

from parsers import dip_adhoc


class NormalizeTests(unittest.TestCase):
    def test_upper_strip(self):
        self.assertEqual(dip_adhoc.normalize_ticker("  googl "), "GOOGL")

    def test_blank_and_none(self):
        self.assertEqual(dip_adhoc.normalize_ticker(""), "")
        self.assertEqual(dip_adhoc.normalize_ticker(None), "")


class SliceSymbolTests(unittest.TestCase):
    def _frames(self):
        hist = pd.DataFrame({
            "symbol": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-02", "2023-01-02"]),
            "close": [11.0, 10.0, 99.0],
            "adj_close": [11.0, 10.0, 99.0],
        })
        divs = pd.DataFrame({"symbol": ["AAA"],
                             "ex_date": pd.to_datetime(["2023-01-02"]),
                             "amount": [0.5]})
        return hist, divs

    def test_slice_sorts_and_filters(self):
        hist, divs = self._frames()
        price, tr, dser = dip_adhoc.slice_symbol(hist, divs, "AAA")
        self.assertEqual(list(price.values), [10.0, 11.0])  # date-sorted
        self.assertEqual(list(tr.values), [10.0, 11.0])
        self.assertEqual(list(dser.values), [0.5])

    def test_slice_unknown_symbol_is_empty(self):
        hist, divs = self._frames()
        price, _, dser = dip_adhoc.slice_symbol(hist, divs, "ZZZ")
        self.assertEqual(len(price), 0)
        self.assertEqual(len(dser), 0)


class UpsertTests(unittest.TestCase):
    def test_replaces_only_that_ticker(self):
        existing = pd.DataFrame({
            "symbol": ["SPY", "SPY", "X"],
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-02"]),
            "close": [1.0, 2.0, 3.0], "adj_close": [1.0, 2.0, 3.0]})
        fresh = pd.DataFrame({"symbol": ["X"],
                              "date": pd.to_datetime(["2023-02-01"]),
                              "close": [9.0], "adj_close": [9.0]})
        out = dip_adhoc.adhoc_upsert(existing, "X", fresh)
        self.assertEqual(int((out["symbol"] == "X").sum()), 1)
        self.assertEqual(out.loc[out["symbol"] == "X", "close"].iloc[0], 9.0)
        self.assertEqual(int((out["symbol"] == "SPY").sum()), 2)

    def test_upsert_into_empty(self):
        empty = pd.DataFrame(columns=["symbol", "date", "close", "adj_close"])
        fresh = pd.DataFrame({"symbol": ["X"],
                              "date": pd.to_datetime(["2023-02-01"]),
                              "close": [9.0], "adj_close": [9.0]})
        out = dip_adhoc.adhoc_upsert(empty, "X", fresh)
        self.assertEqual(len(out), 1)


class StaleTests(unittest.TestCase):
    def _df(self, dates):
        return pd.DataFrame({"symbol": ["X"] * len(dates),
                             "date": pd.to_datetime(dates),
                             "close": [1.0] * len(dates),
                             "adj_close": [1.0] * len(dates)})

    def test_absent_is_stale(self):
        self.assertTrue(dip_adhoc.adhoc_is_stale(self._df(["2023-01-02"]), "Y", "2023-01-02"))

    def test_older_is_stale(self):
        self.assertTrue(dip_adhoc.adhoc_is_stale(self._df(["2023-01-02"]), "X", "2023-01-05"))

    def test_fresh_is_not_stale(self):
        self.assertFalse(dip_adhoc.adhoc_is_stale(self._df(["2023-01-05"]), "X", "2023-01-05"))

    def test_empty_frame_is_stale(self):
        empty = pd.DataFrame(columns=["symbol", "date", "close", "adj_close"])
        self.assertTrue(dip_adhoc.adhoc_is_stale(empty, "X", "2023-01-05"))


class OfflineFetcherTests(unittest.TestCase):
    def test_reads_source_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "src.csv"
            pd.DataFrame({"symbol": ["Q", "Q"],
                          "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
                          "close": [10.0, 11.0],
                          "adj_close": [10.0, 11.0]}).to_csv(p, index=False)
            price_fn, div_fn = dip_adhoc.offline_fetchers(p)
            got = price_fn("Q", None, None)
            self.assertEqual(list(got["close"]), [10.0, 11.0])
            self.assertEqual(list(got.columns), ["date", "close", "adj_close"])
            self.assertEqual(len(div_fn("Q")), 0)  # no dividends sidecar -> empty


class ResolveAdhocTests(unittest.TestCase):
    """resolve_adhoc with injected fetchers + a temp sidecar dir (no network)."""

    def _long_price_fn(self, n=300, last_close=120.0):
        """A fetcher returning n business days of synthetic prices, ending at last_close."""
        def _fn(ticker, start, end):
            idx = pd.bdate_range("2021-01-01", periods=n)
            close = [100.0 + (i % 50) for i in range(n - 1)] + [last_close]
            return pd.DataFrame({"date": idx, "close": close, "adj_close": close})
        return _fn

    def _empty_div_fn(self):
        def _fn(ticker):
            return pd.DataFrame(columns=["ex_date", "amount"])
        return _fn

    def _raise_fn(self):
        def _fn(ticker, start, end):
            raise RuntimeError("network down")
        return _fn

    def test_ok_from_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            res = dip_adhoc.resolve_adhoc(
                Path(d), "NVDA", "2021-12-31",
                self._long_price_fn(), self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=False)
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["n_days"], dip_adhoc.MIN_HISTORY_DAYS)
        self.assertIsNotNone(res["asof"])

    def test_empty_unknown_ticker(self):
        def _empty_price(ticker, start, end):
            return pd.DataFrame(columns=["date", "close", "adj_close"])
        with tempfile.TemporaryDirectory() as d:
            res = dip_adhoc.resolve_adhoc(
                Path(d), "ZZZZ", "2021-12-31",
                _empty_price, self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=False)
        self.assertEqual(res["status"], "empty")

    def test_short_history(self):
        with tempfile.TemporaryDirectory() as d:
            res = dip_adhoc.resolve_adhoc(
                Path(d), "IPO", "2021-12-31",
                self._long_price_fn(n=50), self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=False)
        self.assertEqual(res["status"], "short")
        self.assertEqual(res["n_days"], 50)

    def test_persist_writes_then_fresh_sidecar_skips_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            # First call persists the sidecar.
            dip_adhoc.resolve_adhoc(
                data_dir, "AAA", "2021-12-31",
                self._long_price_fn(), self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=True)
            self.assertTrue((data_dir / "dip_adhoc_history.csv").exists())
            # Second call with a fetcher that would RAISE if called -> must use
            # the fresh sidecar instead (ref_date <= sidecar's last date).
            res = dip_adhoc.resolve_adhoc(
                data_dir, "AAA", "2021-01-04",
                self._raise_fn(), self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=True)
        self.assertEqual(res["status"], "ok")  # no exception -> sidecar served it

    def test_fetch_error_with_no_sidecar_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            res = dip_adhoc.resolve_adhoc(
                Path(d), "AAA", "2021-12-31",
                self._raise_fn(), self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("network down", res["msg"])

    def test_fetch_error_falls_back_to_stale_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            # Seed a stale sidecar for AAA.
            dip_adhoc.resolve_adhoc(
                data_dir, "AAA", "2021-12-31",
                self._long_price_fn(), self._empty_div_fn(),
                pd.Timestamp("2022-01-01"), persist=True)
            # Now a LATER ref_date makes it stale; the live fetch raises -> stale
            # fallback should still serve AAA, flagged stale.
            res = dip_adhoc.resolve_adhoc(
                data_dir, "AAA", "2099-01-01",
                self._raise_fn(), self._empty_div_fn(),
                pd.Timestamp("2099-01-02"), persist=True)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["stale"])


if __name__ == "__main__":
    unittest.main()
