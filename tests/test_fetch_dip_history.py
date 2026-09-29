import unittest
from unittest import mock
import pandas as pd
from parsers import fetch_dip_history as fdh


def _fake_price(ticker, start, end):
    idx = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame({"date": idx,
                         "close": [10, 11, 12, 11, 13],
                         "adj_close": [9, 10, 11, 10, 12]})


def _fake_div(ticker):
    return pd.DataFrame({"ex_date": pd.to_datetime(["2020-01-03"]), "amount": [0.25]})


class BuildHistoryTests(unittest.TestCase):
    def test_build_history_stacks_symbols(self):
        hist, divs = fdh.build_history(["SPY", "SCHD"], _fake_price, _fake_div,
                                       today=pd.Timestamp("2020-01-08"))
        self.assertEqual(set(hist["symbol"]), {"SPY", "SCHD"})
        self.assertListEqual(list(hist.columns), ["symbol", "date", "close", "adj_close"])
        self.assertEqual(set(divs["symbol"]), {"SPY", "SCHD"})
        self.assertListEqual(list(divs.columns), ["symbol", "ex_date", "amount"])

    def test_build_history_drops_trailing_nan_close(self):
        """Yahoo writes a trailing NaN bar for the unsettled session; it must be
        dropped so the latest stored row is always a real settled close."""
        def _price_with_nan(ticker, start, end):
            idx = pd.bdate_range("2020-01-01", periods=4)
            return pd.DataFrame({"date": idx,
                                 "close": [10.0, 11.0, 12.0, float("nan")],
                                 "adj_close": [10.0, 11.0, 12.0, float("nan")]})
        hist, _ = fdh.build_history(["SPY"], _price_with_nan, _fake_div,
                                    today=pd.Timestamp("2020-01-08"))
        self.assertEqual(int(hist["close"].isna().sum()), 0)
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist["close"].iloc[-1], 12.0)


class ValidationTests(unittest.TestCase):
    def _spy(self, closes, start="2021-01-01"):
        idx = pd.bdate_range(start, periods=len(closes))
        return pd.DataFrame({"date": idx, "close": closes})

    def test_matching_series_pass(self):
        a = self._spy([100, 101, 102, 103, 104])
        b = self._spy([100, 101, 102, 103, 104])
        out = fdh.validate_against_polygon(a, b)
        self.assertTrue(out["ok"])
        self.assertGreater(out["return_corr"], 0.99)

    def test_divergent_series_flagged(self):
        a = self._spy([100, 101, 102, 103, 104])
        b = self._spy([100, 90, 120, 80, 130])
        out = fdh.validate_against_polygon(a, b)
        self.assertFalse(out["ok"])


class DipUniverseTests(unittest.TestCase):
    def test_dip_universe_dedupes_and_orders(self):
        """SPY/SCHD/GLD always first; extras upper-cased, stripped, deduped."""
        # Patch config_local away so the test never depends on the real one.
        with mock.patch.dict("sys.modules", {"config_local": None}):
            result = fdh.dip_universe(["aapl", "AAPL", " spy "])
        self.assertEqual(result, ["SPY", "SCHD", "GLD", "AAPL"])


class ValidateEdgeTests(unittest.TestCase):
    def _spy(self, closes, start="2021-01-01"):
        idx = pd.bdate_range(start, periods=len(closes))
        return pd.DataFrame({"date": idx, "close": closes})

    def test_validate_flat_series_not_ok(self):
        """Constant-price series → corr is NaN → ok is False, no crash/warning."""
        flat = self._spy([100, 100, 100, 100, 100])
        out = fdh.validate_against_polygon(flat, flat.copy())
        self.assertFalse(out["ok"])

    def test_validate_short_overlap_not_ok(self):
        """Only 2 overlapping dates → ok False and n_overlap == 2."""
        a = self._spy([100, 101], start="2021-01-04")
        b = self._spy([100, 101], start="2021-01-04")
        out = fdh.validate_against_polygon(a, b)
        self.assertFalse(out["ok"])
        self.assertEqual(out["n_overlap"], 2)


class IndexSidecarTests(unittest.TestCase):
    """Spec 2026-07-19 §4a: index series ride the same --write run into a
    SIDECAR csv — never into dip_history.csv (every symbol there becomes a
    UI card)."""

    @staticmethod
    def _fetch(ticker, start, end):
        import pandas as pd
        idx = pd.bdate_range("1950-01-03", periods=1200)
        lvl = pd.Series(range(1, 1201), index=idx, dtype=float)
        return pd.DataFrame({"date": idx, "close": lvl.values,
                             "adj_close": lvl.values})

    def test_build_index_history_shape(self):
        import pandas as pd
        hist = fdh.build_index_history(self._fetch,
                                       pd.Timestamp("2026-07-19"))
        self.assertEqual(list(hist.columns),
                         ["symbol", "date", "close", "adj_close"])
        self.assertEqual(sorted(hist["symbol"].unique()),
                         sorted(fdh.INDEX_TICKERS))

    def test_index_history_ok(self):
        import pandas as pd
        good = fdh.build_index_history(self._fetch,
                                       pd.Timestamp("2026-07-19"))
        self.assertTrue(fdh.index_history_ok(good))
        self.assertFalse(fdh.index_history_ok(good[good["symbol"] == "^GSPC"]))
        self.assertFalse(fdh.index_history_ok(good.head(10)))
        self.assertFalse(
            fdh.index_history_ok(good.iloc[0:0]))

    def test_index_tickers_never_in_default_universe(self):
        for t in fdh.INDEX_TICKERS:
            self.assertNotIn(t, fdh.dip_universe())


class NoClobberGuardTests(unittest.TestCase):
    """2026-08-19 incident: a total Yahoo TLS failure (Norton MITM root
    rotation) made every per-ticker fetch return empty, and --write truncated
    BOTH primary csvs to header-only. The index sidecar survived because
    index_history_ok refused the empty overwrite; these tests lock the same
    no-clobber contract onto dip_history.csv / dip_dividends.csv."""

    CORE = ["SPY", "SCHD", "GLD"]

    def _frames(self):
        return fdh.build_history(self.CORE, _fake_price, _fake_div,
                                 today=pd.Timestamp("2020-01-08"))

    @staticmethod
    def _empty_frames():
        return (pd.DataFrame(columns=["symbol", "date", "close", "adj_close"]),
                pd.DataFrame(columns=["symbol", "ex_date", "amount"]))

    # --- history_ok: is the fetched frame trustworthy enough to write? ---

    def test_history_ok_true_on_full_core(self):
        hist, _ = self._frames()
        self.assertTrue(fdh.history_ok(hist, self.CORE))

    def test_history_ok_false_on_empty(self):
        hist, _ = self._empty_frames()
        self.assertFalse(fdh.history_ok(hist, self.CORE))
        self.assertFalse(fdh.history_ok(hist, ["AAPL"]))

    def test_history_ok_false_when_core_symbol_missing(self):
        hist, _ = self._frames()
        self.assertFalse(fdh.history_ok(hist[hist["symbol"] != "SCHD"],
                                        self.CORE))

    def test_history_ok_ignores_missing_watchlist_extras(self):
        """A bogus/delisted watchlist ticker returning nothing must NOT block
        the write — only the always-on core tickers gate it."""
        hist, _ = self._frames()
        self.assertTrue(fdh.history_ok(hist, self.CORE + ["ZZZDELISTED"]))

    def test_history_ok_tickers_override_without_core(self):
        """--tickers AAPL never asked for the core; non-empty is enough."""
        hist, _ = fdh.build_history(["AAPL"], _fake_price, _fake_div,
                                    today=pd.Timestamp("2020-01-08"))
        self.assertTrue(fdh.history_ok(hist, ["AAPL"]))

    # --- emit_outputs: guarded writes for the two primary csvs ---

    def _paths(self, tmp):
        from pathlib import Path
        return Path(tmp) / "dip_history.csv", Path(tmp) / "dip_dividends.csv"

    def test_healthy_write(self):
        import tempfile
        hist, divs = self._frames()
        with tempfile.TemporaryDirectory() as tmp:
            h_csv, d_csv = self._paths(tmp)
            rc = fdh.emit_outputs(hist, divs, self.CORE, h_csv, d_csv)
            self.assertEqual(rc, 0)
            self.assertEqual(len(pd.read_csv(h_csv)), len(hist))
            self.assertEqual(len(pd.read_csv(d_csv)), len(divs))

    def test_refuses_to_clobber_populated_files(self):
        import tempfile
        hist, divs = self._frames()
        empty_h, empty_d = self._empty_frames()
        with tempfile.TemporaryDirectory() as tmp:
            h_csv, d_csv = self._paths(tmp)
            fdh.emit_outputs(hist, divs, self.CORE, h_csv, d_csv)
            before = (h_csv.read_bytes(), d_csv.read_bytes())
            rc = fdh.emit_outputs(empty_h, empty_d, self.CORE, h_csv, d_csv)
            self.assertEqual(rc, 1)
            self.assertEqual((h_csv.read_bytes(), d_csv.read_bytes()), before)

    def test_degraded_core_refuses_even_when_nonempty(self):
        """A partial outage (a core ticker came back empty) must not shrink a
        populated file either — stale-complete beats fresh-partial for a file
        whose every symbol is a UI card."""
        import tempfile
        hist, divs = self._frames()
        with tempfile.TemporaryDirectory() as tmp:
            h_csv, d_csv = self._paths(tmp)
            fdh.emit_outputs(hist, divs, self.CORE, h_csv, d_csv)
            before = h_csv.read_bytes()
            rc = fdh.emit_outputs(hist[hist["symbol"] == "GLD"], divs,
                                  self.CORE, h_csv, d_csv)
            self.assertEqual(rc, 1)
            self.assertEqual(h_csv.read_bytes(), before)

    def test_bootstrap_write_when_nothing_to_protect(self):
        """First run on a broken network: nothing to keep, so the (empty)
        artifacts are still written — but the exit code says the fetch failed."""
        import tempfile
        empty_h, empty_d = self._empty_frames()
        with tempfile.TemporaryDirectory() as tmp:
            h_csv, d_csv = self._paths(tmp)
            rc = fdh.emit_outputs(empty_h, empty_d, self.CORE, h_csv, d_csv)
            self.assertEqual(rc, 1)
            self.assertTrue(h_csv.exists() and d_csv.exists())
            self.assertEqual(len(pd.read_csv(h_csv)), 0)

    def test_empty_dividends_keep_existing(self):
        """Good history + dead dividend endpoint: history advances, the
        populated dividend file is kept, and the run still reports failure."""
        import tempfile
        hist, divs = self._frames()
        empty_d = self._empty_frames()[1]
        with tempfile.TemporaryDirectory() as tmp:
            h_csv, d_csv = self._paths(tmp)
            fdh.emit_outputs(hist, divs, self.CORE, h_csv, d_csv)
            d_before = d_csv.read_bytes()
            hist2 = hist.copy()
            hist2["close"] = hist2["close"] + 1.0
            rc = fdh.emit_outputs(hist2, empty_d, self.CORE, h_csv, d_csv)
            self.assertEqual(rc, 1)
            self.assertEqual(d_csv.read_bytes(), d_before)
            self.assertEqual(pd.read_csv(h_csv)["close"].iloc[0],
                             hist2["close"].iloc[0])


if __name__ == "__main__":
    unittest.main()
