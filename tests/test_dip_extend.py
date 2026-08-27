"""Tests for parsers/dip_extend.py — the SPX extended-series builder — and
the committed dividend-yield table it consumes.
Spec: docs/superpowers/specs/2026-07-19-dip-spx-history-extension-design.md."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from parsers import dip_extend as de

ROOT = Path(__file__).resolve().parents[1]
YIELD_CSV = ROOT / "parsers" / "spx_dividend_yield_monthly.csv"


class YieldTableTests(unittest.TestCase):
    """The committed table is public immutable history; these pins are its
    integrity contract (spec §4a / §6.2)."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(YIELD_CSV, dtype={"month": str})

    def test_schema(self):
        self.assertEqual(list(self.df.columns), ["month", "yield"])

    def test_months_unique_ascending(self):
        m = self.df["month"].tolist()
        self.assertEqual(m, sorted(m))
        self.assertEqual(len(m), len(set(m)))
        self.assertTrue(all(len(x) == 7 and x[4] == "-" for x in m))

    def test_span(self):
        self.assertLessEqual(self.df["month"].iloc[0], "1949-01")
        self.assertGreaterEqual(self.df["month"].iloc[-1], "2024-12")

    def test_plausible_bounds_production_span(self):
        prod = self.df[self.df["month"] <= "1987-12"]["yield"]
        self.assertTrue(bool(((prod > 0.0) & (prod < 0.10)).all()))

    def test_spot_pins_known_history(self):
        s = self.df.set_index("month")["yield"]
        # Wide, historically safe ranges — era-defining levels, not decimals.
        self.assertTrue(0.045 <= s["1982-06"] <= 0.075)   # early-80s highs
        self.assertTrue(0.020 <= s["1973-01"] <= 0.040)   # pre-oil-shock
        self.assertTrue(0.008 <= s["2000-03"] <= 0.020)   # dot-com lows


def _mk_components(years=8, start="2000-01-03", spy_frac=0.6, tr_frac=0.35,
                   yld=0.03):
    """Deterministic synthetic components shaped like the real inputs.
    gspc spans the whole window; sptr starts at tr_frac, spy at spy_frac
    (both later than gspc, mirroring 1988/1993 vs 1950)."""
    idx = pd.bdate_range(start, periods=252 * years)
    n = len(idx)
    ret = np.where(np.arange(n) % 2 == 0, 0.001, -0.0005)
    gspc = pd.Series(100.0 * np.cumprod(1.0 + ret), index=idx, name="^GSPC")
    spy_start = idx[int(n * spy_frac)]
    tr_start = idx[int(n * tr_frac)]
    spy_close = (gspc[gspc.index >= spy_start] * 0.1).rename("SPY")
    spy_adj = (spy_close * 0.8).rename("SPY")
    sptr = (gspc[gspc.index >= tr_start] * 1.7).rename("^SP500TR")
    months = sorted(set(idx.strftime("%Y-%m")))
    yields = pd.Series(yld, index=pd.Index(months, name="month"))
    return {"spy_close": spy_close, "spy_adj": spy_adj, "gspc_close": gspc,
            "sptr_close": sptr, "yields": yields}


class YieldLoaderTests(unittest.TestCase):
    def test_loads_committed_table(self):
        s = de.load_yield_table()
        self.assertIsInstance(s, pd.Series)
        self.assertLessEqual(s.index[0], "1949-01")
        self.assertTrue(float(s.loc["1975-06"]) > 0.0)

    def test_missing_file_raises_with_message(self):
        with self.assertRaises(FileNotFoundError) as cm:
            de.load_yield_table(Path("nonexistent_dir") / "nope.csv")
        self.assertIn("yield table", str(cm.exception))


class SyntheticTrTests(unittest.TestCase):
    def test_accrual_formula(self):
        idx = pd.bdate_range("1980-01-01", periods=5)
        px = pd.Series([100.0, 101.0, 100.0, 102.0, 102.0], index=idx)
        yields = pd.Series(0.0252, index=pd.Index(
            sorted(set(idx.strftime("%Y-%m")))))
        lvl = de.synthetic_tr_levels(px, yields)
        # daily tr ret = px ret + 0.0252/252 = px ret + 0.0001
        exp1 = (101.0 / 100.0 - 1.0) + 0.0001
        self.assertAlmostEqual(lvl.iloc[1] / lvl.iloc[0] - 1.0, exp1, places=12)
        # flat price day still accrues the dividend
        self.assertAlmostEqual(lvl.iloc[4] / lvl.iloc[3] - 1.0, 0.0001,
                               places=12)

    def test_missing_month_raises(self):
        idx = pd.bdate_range("1980-01-01", periods=30)  # spans Jan+Feb
        px = pd.Series(100.0, index=idx)
        yields = pd.Series(0.03, index=pd.Index(["1980-01"]))
        with self.assertRaises(ValueError) as cm:
            de.synthetic_tr_levels(px, yields)
        self.assertIn("1980-02", str(cm.exception))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            de.synthetic_tr_levels(pd.Series(dtype=float),
                                   pd.Series(dtype=float))


class BuildExtendedSpxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _mk_components()
        cls.out = de.build_extended_spx(cls.c["spy_close"], cls.c["spy_adj"],
                                        cls.c["gspc_close"],
                                        cls.c["sptr_close"], cls.c["yields"])

    def test_shape_and_shared_index(self):
        price, tr = self.out["price"], self.out["tr"]
        self.assertTrue(price.index.equals(tr.index))
        self.assertFalse(price.isna().any())
        self.assertFalse(tr.isna().any())
        self.assertEqual(price.name, "SPX")
        self.assertEqual(tr.name, "SPX")

    def test_price_segments_verbatim(self):
        price = self.out["price"]
        spy = self.c["spy_close"]
        gspc = self.c["gspc_close"]
        join = spy.index.min()
        # post-join: SPY levels verbatim
        pd.testing.assert_series_equal(price[price.index >= join],
                                       spy, check_names=False)
        # pre-join: gspc returns verbatim
        pre = price[price.index < join]
        exp = gspc[gspc.index < join]
        np.testing.assert_allclose(pre.pct_change().dropna().to_numpy(),
                                   exp.pct_change().dropna().to_numpy(),
                                   rtol=1e-12)

    def test_tr_segments_returns(self):
        tr = self.out["tr"]
        spy_a = self.c["spy_adj"]
        sptr = self.c["sptr_close"]
        spy_join = spy_a.index.min()
        tr_join = sptr.index.min()
        # SPY adj segment verbatim
        pd.testing.assert_series_equal(tr[tr.index >= spy_join], spy_a,
                                       check_names=False)
        # sptr segment returns verbatim
        mid = tr[(tr.index >= tr_join) & (tr.index < spy_join)]
        exp = sptr[sptr.index < spy_join]
        np.testing.assert_allclose(mid.pct_change().dropna().to_numpy(),
                                   exp.pct_change().dropna().to_numpy(),
                                   rtol=1e-12)
        # synthetic segment: tr return = price return + accrual
        pre = tr[tr.index < tr_join].pct_change().dropna()
        px_pre = self.c["gspc_close"][
            self.c["gspc_close"].index < tr_join].pct_change().dropna()
        np.testing.assert_allclose(pre.to_numpy(),
                                   px_pre.to_numpy() + 0.03 / 252.0,
                                   atol=1e-12)

    def test_meta_block(self):
        m = self.out["meta"]
        self.assertEqual(m["id"], "SPX")
        self.assertEqual(m["start"],
                         str(self.c["gspc_close"].index.min().date()))
        self.assertEqual(m["tr_segments"]["sp500tr_from"],
                         str(self.c["sptr_close"].index.min().date()))
        self.assertIn("yield_span", m)

    def test_spx_start_trim(self):
        c = _mk_components(start="1948-01-05")
        out = de.build_extended_spx(c["spy_close"], c["spy_adj"],
                                    c["gspc_close"], c["sptr_close"],
                                    c["yields"])
        self.assertGreaterEqual(out["price"].index.min(),
                                pd.Timestamp(de.SPX_START))

    def test_no_extension_raises(self):
        c = _mk_components()
        gspc_late = c["gspc_close"][
            c["gspc_close"].index >= c["spy_close"].index.min()]
        with self.assertRaises(ValueError) as cm:
            de.build_extended_spx(c["spy_close"], c["spy_adj"], gspc_late,
                                  c["sptr_close"], c["yields"])
        self.assertIn("GSPC", str(cm.exception))

    def test_empty_component_raises(self):
        c = _mk_components()
        with self.assertRaises(ValueError) as cm:
            de.build_extended_spx(c["spy_close"], c["spy_adj"],
                                  pd.Series(dtype=float), c["sptr_close"],
                                  c["yields"])
        self.assertIn("gspc_close", str(cm.exception))


class GateMeasurementTests(unittest.TestCase):
    def test_overlap_corr_identical_series(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        a = pd.Series(np.linspace(100, 130, 300), index=idx)
        r = de.overlap_return_corr(a, a * 3.0)
        self.assertAlmostEqual(r["corr"], 1.0, places=9)
        self.assertEqual(r["n_overlap"], 299)

    def test_overlap_corr_too_short(self):
        idx = pd.bdate_range("2020-01-01", periods=2)
        a = pd.Series([1.0, 2.0], index=idx)
        r = de.overlap_return_corr(a, a)
        self.assertTrue(np.isnan(r["corr"]))

    def test_tracking_identical_is_zero(self):
        idx = pd.bdate_range("2018-01-01", periods=600)
        ret = np.where(np.arange(600) % 2 == 0, 0.002, -0.001)
        lvl = pd.Series(np.cumprod(1.0 + ret), index=idx)
        t = de.tracking_measurements(lvl, lvl * 2.0)
        self.assertAlmostEqual(t["ann_diff_bps"], 0.0, places=6)
        self.assertAlmostEqual(t["p95_roll252_bps"], 0.0, places=6)

    def test_tracking_known_gap(self):
        idx = pd.bdate_range("2018-01-01", periods=600)
        base = pd.Series(np.cumprod(np.full(600, 1.0004)), index=idx)
        drift = pd.Series(np.cumprod(np.full(600, 1.0004 + 0.0050 / 252.0)),
                          index=idx)
        t = de.tracking_measurements(drift, base)
        # 50 bps/yr constructed gap; compounding keeps it within a few bps
        self.assertGreater(t["ann_diff_bps"], 35.0)
        self.assertLess(t["ann_diff_bps"], 65.0)

    def test_tracking_too_short_is_nan(self):
        idx = pd.bdate_range("2018-01-01", periods=100)
        lvl = pd.Series(1.0, index=idx)
        t = de.tracking_measurements(lvl, lvl)
        self.assertTrue(np.isnan(t["ann_diff_bps"]))


def _write_component_csvs(tmp, c):
    hist = pd.concat([
        pd.DataFrame({"symbol": "SPY", "date": c["spy_close"].index,
                      "close": c["spy_close"].values,
                      "adj_close": c["spy_adj"].values}),
        pd.DataFrame({"symbol": "GLD",
                      "date": c["spy_close"].index[:50],
                      "close": 100.0, "adj_close": 100.0}),
    ], ignore_index=True)
    hist.to_csv(Path(tmp) / "dip_history.csv", index=False)
    idx = pd.concat([
        pd.DataFrame({"symbol": "^GSPC", "date": c["gspc_close"].index,
                      "close": c["gspc_close"].values,
                      "adj_close": c["gspc_close"].values}),
        pd.DataFrame({"symbol": "^SP500TR", "date": c["sptr_close"].index,
                      "close": c["sptr_close"].values,
                      "adj_close": c["sptr_close"].values}),
    ], ignore_index=True)
    idx.to_csv(Path(tmp) / "dip_index_history.csv", index=False)
    yp = Path(tmp) / "yields.csv"
    c["yields"].rename("yield").rename_axis("month").reset_index() \
        .to_csv(yp, index=False)
    return yp


class LoadExtendedSpxTests(unittest.TestCase):
    def test_round_trip(self):
        c = _mk_components()
        with tempfile.TemporaryDirectory() as tmp:
            yp = _write_component_csvs(tmp, c)
            out = de.load_extended_spx(tmp, yield_path=yp)
        direct = de.build_extended_spx(c["spy_close"], c["spy_adj"],
                                       c["gspc_close"], c["sptr_close"],
                                       c["yields"])
        pd.testing.assert_series_equal(out["price"], direct["price"],
                                       check_names=False, check_freq=False)
        pd.testing.assert_series_equal(out["tr"], direct["tr"],
                                       check_names=False, check_freq=False)
        self.assertEqual(out["meta"], direct["meta"])
        self.assertEqual(sorted(out["components"]),
                         ["gspc_close", "sptr_close", "spy_adj", "spy_close",
                          "yields"])

    def test_missing_sidecar_raises_with_fetch_command(self):
        c = _mk_components()
        with tempfile.TemporaryDirectory() as tmp:
            yp = _write_component_csvs(tmp, c)
            (Path(tmp) / "dip_index_history.csv").unlink()
            with self.assertRaises(FileNotFoundError) as cm:
                de.load_extended_spx(tmp, yield_path=yp)
        self.assertIn("fetch_dip_history.py --write", str(cm.exception))

    def test_missing_symbol_raises(self):
        c = _mk_components()
        with tempfile.TemporaryDirectory() as tmp:
            yp = _write_component_csvs(tmp, c)
            p = Path(tmp) / "dip_index_history.csv"
            df = pd.read_csv(p)
            df[df["symbol"] == "^GSPC"].to_csv(p, index=False)
            with self.assertRaises(ValueError) as cm:
                de.load_extended_spx(tmp, yield_path=yp)
        self.assertIn("^SP500TR", str(cm.exception))


class RunDataGatesTests(unittest.TestCase):
    def test_all_ok_on_consistent_synthetic_book(self):
        c = _mk_components(yld=0.0)   # zero yield -> synth == sptr returns
        g = de.run_data_gates(c)
        self.assertTrue(g["corr_gspc_vs_spy"]["ok"])
        self.assertTrue(g["corr_sptr_vs_spy"]["ok"])
        self.assertGreaterEqual(g["corr_gspc_vs_spy"]["corr_2002p"], 0.99)
        self.assertTrue(g["tracking"]["ok"])
        self.assertTrue(g["all_ok"])

    def test_tracking_gate_fails_on_big_gap(self):
        c = _mk_components(yld=0.0)
        # corrupt sptr with a large persistent return drift
        drift = pd.Series(
            np.cumprod(np.full(len(c["sptr_close"]), 1.0 + 0.20 / 252.0)),
            index=c["sptr_close"].index)
        c["sptr_close"] = c["sptr_close"] * drift
        g = de.run_data_gates(c)
        self.assertFalse(g["tracking"]["ok"])
        self.assertFalse(g["all_ok"])

    def test_full_window_floor_fails_on_unrelated_series(self):
        c = _mk_components()
        rng = np.random.default_rng(7)
        noise = pd.Series(
            100.0 * np.cumprod(1.0 + rng.normal(0, 0.02,
                                                len(c["spy_close"]))),
            index=c["spy_close"].index)
        c["spy_close"] = noise
        g = de.run_data_gates(c)
        self.assertFalse(g["corr_gspc_vs_spy"]["ok"])
        self.assertFalse(g["all_ok"])

    def test_modern_window_floor_binds(self):
        # degrade ONLY the post-2002 segment (the synthetic components span
        # 2000-2008, so MODERN_CORR_START splits them): the strict modern
        # floor must fail and the gate must fail with it, whatever the
        # full-window number does
        c = _mk_components()
        rng = np.random.default_rng(11)
        s = c["spy_close"].copy()
        m = s.index >= pd.Timestamp(de.MODERN_CORR_START)
        jitter = 1.0 + rng.normal(0.0, 0.004, int(m.sum()))
        s.loc[m] = s.loc[m] * np.cumprod(jitter)
        c["spy_close"] = s
        g = de.run_data_gates(c)
        self.assertLess(g["corr_gspc_vs_spy"]["corr_2002p"], 0.99)
        self.assertFalse(g["corr_gspc_vs_spy"]["ok"])
        self.assertFalse(g["all_ok"])

    def test_bounds_are_frozen_literals(self):
        self.assertGreater(de.TRACKING_MAX_ANN_BPS, 0.0)
        self.assertGreater(de.TRACKING_MAX_P95_BPS, 0.0)
