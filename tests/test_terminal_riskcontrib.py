# tests/test_terminal_riskcontrib.py
import dataclasses, json, math, os, sys, unittest
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "parsers"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import riskcontrib_service as rcs


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (same as the Factor golden:
    the rolling beta/alpha + synthesis pass through many float ops that aren't
    bit-reproducible across BLAS builds, so pin structure + formatted strings
    exactly but compare raw floats with a relative tolerance)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a is b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            m = _deep_close(a[k], b[k], rel=rel, abs_=abs_, path=f"{path}.{k}")
            if m: return m
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = _deep_close(x, y, rel=rel, abs_=abs_, path=f"{path}[{i}]")
            if m: return m
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (None if math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
                else f"{path}: {a!r} !~ {b!r}")
    return None if a == b else f"{path}: {a!r} != {b!r}"


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = rcs.build_riskcontrib_view(cls.frames)

    def test_keys(self):
        self.assertEqual(set(self.view.keys()), {
            "meta", "info_html", "caption_html", "state", "filter_note",
            "treasury_ladder", "controls", "combos", "benchmarks",
            "dr_in_context", "dr_regime", "correlations"})

    def test_correlations_present_and_jsonable(self):
        c = self.view["correlations"]
        self.assertIsNotNone(c)
        self.assertIn("major", c)
        self.assertIsNotNone(c["stress"])       # Slice 3b now fills this
        self.assertEqual(set(c["major"]["top15"]),
                         {"ewma_lw", "ewma", "rolling"})
        json.dumps(c, allow_nan=False)

    def test_correlations_fallback_state_on_fixture(self):
        c = self.view["correlations"]["major"]
        # fixture has no long_history_prices.csv -> Big-3 falls back
        self.assertTrue(c["big3"]["fallback"])
        self.assertEqual(c["big3"]["reason"], "no_long_history")
        # each estimator's Top-15 block resolves to a concrete state
        for est in ("ewma_lw", "ewma", "rolling"):
            blk = c["top15"][est]
            self.assertIn(blk["available"], (True, False))
            if blk["available"]:
                self.assertIsNotNone(blk["heatmap"])
            else:
                self.assertEqual(blk["reason"], "insufficient_names")

    def test_dr_in_context_available_and_jsonable(self):
        dr = self.view["dr_in_context"]
        self.assertTrue(dr["available"])          # fixture has ≥252d burn-in
        self.assertEqual(len(dr["tiles"]), 4)
        self.assertEqual(set(dr["thresholds"]), {"fixed", "percentile", "zscore"})
        json.dumps(dr, allow_nan=False)

    def test_dr_regime_no_inputs_on_fixture(self):
        dr = self.view["dr_regime"]
        # fixture has no vix_history.csv -> no_inputs (DR itself is available)
        self.assertFalse(dr["available"])
        self.assertEqual(dr["reason"], "no_inputs")
        self.assertIn("vix_history.csv", dr["message"])
        json.dumps(dr, allow_nan=False)

    def test_controls_defaults(self):
        c = self.view["controls"]
        self.assertEqual(c["estimators"][0]["id"], "ewma_lw")
        self.assertEqual([e["id"] for e in c["es_levels"]], ["0.05", "0.025", "0.01"])
        self.assertEqual([t["id"] for t in c["thresholds"]], ["0.0", "-0.005", "-0.01"])
        self.assertEqual(c["benchmarks"][0]["id"], "SPY")

    def test_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)  # raises if any NaN leaks

    def test_stress_populated_on_fixture(self):
        s = self.view["correlations"]["stress"]
        self.assertIsNotNone(s)
        self.assertIn("big3", s)
        self.assertEqual(set(s["top15"]), {"ewma_lw", "ewma", "rolling"})
        # fixture has no long_history -> Big-3 stress is no_inputs (silent)
        self.assertEqual(s["big3"]["reason"], "no_inputs")
        self.assertIsNone(s["big3"]["message"])
        json.dumps(s, allow_nan=False)


class TestEmptyStates(unittest.TestCase):
    def test_no_twr(self):
        frames = hs.load_frames(FIXTURE)
        f2 = dataclasses.replace(frames, twr_portfolio=frames.twr_portfolio.iloc[0:0])
        view = rcs.build_riskcontrib_view(f2)
        self.assertFalse(view["state"]["available"])
        self.assertEqual(view["state"]["unavailable"], "no_twr")
        self.assertEqual(view["combos"], {})

    def test_daily_empty(self):
        frames = hs.load_frames(FIXTURE)
        f2 = dataclasses.replace(frames, daily_prices=frames.daily_prices.iloc[0:0])
        view = rcs.build_riskcontrib_view(f2)
        self.assertEqual(view["state"]["unavailable"], "daily_empty")


class TestFilterNote(unittest.TestCase):
    def test_filter_flips_note(self):
        frames = hs.load_frames(FIXTURE)
        base = rcs.build_riskcontrib_view(frames)
        opts = base["meta"]["classes"]
        if not opts:
            self.skipTest("no class options on fixture")
        filtered = rcs.build_riskcontrib_view(frames, asset_class=opts[0]["id"])
        self.assertTrue(filtered["meta"]["class_filter_active"])
        self.assertIsNotNone(filtered["filter_note"])


class TestComboParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = rcs.build_riskcontrib_view(cls.frames)

    def _bundle(self):
        from terminal.performance_service import _resolve_filter
        snap_all = hs._current_snap(self.frames)
        bf, cf, _i, aa, ca = _resolve_filter(self.frames, snap_all, "all", "all")
        from terminal import risk_service as rs
        return rs._bundle(self.frames, bf, cf, aa, ca)

    def test_default_combo_top_tiles(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        from risk_metrics import compute_risk_contributions
        b = self._bundle()
        rc = compute_risk_contributions(b["weights"], self.frames.daily_prices,
                                        window=rcs.RC_WINDOW, estimator="ewma_lw")
        per = rc["per_symbol"]
        combo = self.view["combos"]["ewma_lw|0.05|0.0"]
        tiles = combo["top_tiles"]
        self.assertEqual(tiles[0]["value"], str(per.index[0]))
        self.assertEqual(tiles[1]["value"], f"{per['pctr_pct'].head(3).sum():.1f}%")
        self.assertEqual(tiles[3]["value"],
                         f"{rc['dr']:.2f}×" if np.isfinite(rc["dr"]) else "—")

    def test_default_combo_table_first_row(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        from risk_metrics import compute_risk_contributions
        b = self._bundle()
        rc = compute_risk_contributions(b["weights"], self.frames.daily_prices,
                                        window=rcs.RC_WINDOW, estimator="ewma_lw")
        per = rc["per_symbol"]
        row0 = self.view["combos"]["ewma_lw|0.05|0.0"]["table"]["rows"][0]
        self.assertEqual(row0["symbol"], str(per.index[0]))
        self.assertEqual(row0["weight"], f"{per['weight_pct'].iloc[0]:.2f}%")
        self.assertEqual(row0["pctr"], f"{per['pctr_pct'].iloc[0]:.2f}%")
        self.assertEqual(row0["risk_delta"]["text"], f"{per['diff_pp'].iloc[0]:+.2f}")

    def test_weight_vs_pctr_top10_union(self):
        """Pairbar rows = top-10 by PCTR ∪ top-10 by weight — a big-weight,
        near-zero-PCTR holding (the T-bill sleeve) must stay on the chart."""
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        wv = self.view["combos"]["ewma_lw|0.05|0.0"]["weight_vs_pctr"]
        self.assertEqual(len(wv["symbols"]), len(wv["weight"]))
        self.assertEqual(len(wv["symbols"]), len(wv["pctr"]))
        self.assertLessEqual(len(wv["symbols"]), 20)
        from risk_metrics import compute_risk_contributions
        b = self._bundle()
        rc = compute_risk_contributions(b["weights"], self.frames.daily_prices,
                                        window=rcs.RC_WINDOW, estimator="ewma_lw")
        per = rc["per_symbol"]
        heavy = per.sort_values("weight_pct", ascending=False).head(10).index
        for sym in heavy:
            self.assertIn(str(sym), wv["symbols"],
                          f"top-10-by-weight {sym} missing from the pairbar")

    def test_pairbar_rows_union_constructed(self):
        """12-name panel: the #12-by-PCTR / #1-by-weight name is kept, order
        stays PCTR-descending, and pure-top-10 inputs pass through unchanged."""
        n = 12
        per = pd.DataFrame({
            # PCTR descending 12..1; weights inverted so the LOWEST-PCTR name
            # is the single biggest weight (the SGOV shape).
            "pctr_pct": np.arange(n, 0, -1, dtype=float),
            "weight_pct": np.arange(1, n + 1, dtype=float),
        }, index=[f"S{i:02d}" for i in range(n)])
        sel = rcs._pairbar_rows(per)
        self.assertIn("S11", sel.index)          # rank-12 PCTR, rank-1 weight
        self.assertIn("S10", sel.index)          # rank-11 PCTR, rank-2 weight
        self.assertEqual(list(sel.index), sorted(
            sel.index, key=lambda s: -per.loc[s, "pctr_pct"]))
        self.assertEqual(len(sel), 12)           # 10 + the 2 heavy stragglers
        top8 = per.head(8)
        self.assertEqual(list(rcs._pairbar_rows(top8).index), list(top8.index))

    def test_all_27_combos_present(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        self.assertEqual(len(self.view["combos"]), 27)
        self.assertLessEqual(len(self.view["combos"]["ewma_lw|0.05|0.0"]["table"]["rows"]), 15)


class TestBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = rcs.build_riskcontrib_view(cls.frames)

    def test_spy_present_and_dirs_valid(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        bm = self.view["benchmarks"]
        self.assertIn("SPY", bm)
        for est in ("ewma_lw", "ewma", "rolling"):
            cmp = bm["SPY"]["vol"][est]
            self.assertIn(cmp["dir"], ("up", "down", "flat", None))
        # every benchmark id appears in the controls list
        ids = {x["id"] for x in self.view["controls"]["benchmarks"]}
        self.assertEqual(set(bm.keys()), ids)

    def test_bench_vol_matches_engine(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        from risk_metrics import series_vol_ann
        from terminal import risk_service as rs
        from terminal.performance_service import _resolve_filter
        snap = hs._current_snap(self.frames)
        bf, cf, _i, aa, ca = _resolve_filter(self.frames, snap, "all", "all")
        b = rs._bundle(self.frames, bf, cf, aa, ca)
        v = series_vol_ann(b["spy_rets"], estimator="ewma_lw", window=rcs.RC_WINDOW,
                           lambda_param=0.94)
        cmp = self.view["benchmarks"]["SPY"]["vol"]["ewma_lw"]
        if cmp["value"] is not None:
            self.assertEqual(cmp["value"], f"{v * 100:.2f}%")

    def test_no_priceless_benchmarks(self):
        # A benchmark must have a usable price series. A TICKER_HISTORY rename
        # whose symbols are absent from the data dir injects a phantom all-NaN
        # column; config_local.py (TICKER_HISTORY) is gitignored, so without
        # filtering it the fixture's benchmark universe differs Windows vs CI
        # and the golden won't reproduce. (Guards the _benchmarks empty-series skip.)
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        daily = self.frames.daily_prices
        for bm in self.view["controls"]["benchmarks"]:
            if bm["id"] in ("SPY", "60_40"):
                continue
            self.assertIn(bm["id"], daily.columns)
            self.assertGreater(daily[bm["id"]].pct_change().dropna().size, 0,
                               f"benchmark {bm['id']} has no usable price series")


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_ok(self):
        r = self.client.get("/api/riskcontrib")
        self.assertEqual(r.status_code, 200)
        self.assertIn("combos", r.json())

    def test_unknown_account_422(self):
        r = self.client.get("/api/riskcontrib", params={"account": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/riskcontrib")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_riskcontrib_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = rcs.build_riskcontrib_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        mismatch = _deep_close(view, expected)
        self.assertIsNone(mismatch, f"riskcontrib view diverges from golden at {mismatch}")


class TestEsPctrTint(unittest.TestCase):
    """app.py tints the "ES PCTR" column via Styler .map(subset=[..., "ES PCTR"]),
    which feeds each cell its OWN value — so ES PCTR is colored by the raw ES PCTR
    value, NOT by (ES − PCTR). The synth fixture's ES column is all "—", so this
    bug is invisible there; pin it with a constructed panel where ES PCTR spans
    the tint thresholds."""
    def test_es_pctr_tinted_on_raw_value_not_delta(self):
        idx = ["AAA", "BBB", "CCC"]
        per = pd.DataFrame({
            "weight_pct": [10.0, 10.0, 10.0],
            "standalone_vol_ann": [0.2, 0.2, 0.2],
            "pctr_pct": [40.0, 5.0, 1.0],
            "diff_pp": [30.0, -5.0, -9.0],
            "n_obs_with_price": [300, 300, 300],
            "n_obs_in_window": [300, 300, 300],
        }, index=idx)
        rc = {"per_symbol": per, "n_days": 300}
        rd = {"per_symbol_down": pd.DataFrame(
            {"pctr_pct_down": [42.0, 4.0, 1.0]}, index=idx)}
        raw_es = [35.0, 3.0, 1.0]
        re_ = {"per_symbol_es": pd.DataFrame(
            {"pctr_es_pct": raw_es, "n_obs_in_window": [300, 300, 300]},
            index=idx), "n_days_window": 252}
        rows = rcs._table_rows(rc, rd, re_)
        # ES PCTR cls is keyed off its OWN value, not (es − pctr). Each row's
        # tint must equal _delta_cls(raw_es), spanning the thresholds:
        #   35 → loss-strong (≥5) · 3 → loss-mild (≥2) · 1 → "" (within ±2).
        for row, raw in zip(rows, raw_es):
            self.assertEqual(row["es_pctr"]["cls"], rcs._delta_cls(raw))
        self.assertEqual(rows[0]["es_pctr"]["cls"], "loss-strong")
        self.assertEqual(rows[1]["es_pctr"]["cls"], "loss-mild")
        self.assertEqual(rows[2]["es_pctr"]["cls"], "")
        # Regression guard: the buggy code tinted row 0 off the delta
        # (35−40=−5 → gain-strong); the raw value gives loss-strong instead.
        self.assertNotEqual(rows[0]["es_pctr"]["cls"], "gain-strong")
        # bold flag still uses the delta: 35-40=-5 → not >= 5 → False
        self.assertFalse(rows[0]["es_pctr"]["bold"])
        # sanity: es_delta still uses the delta (35-40=-5 → gain-strong)
        self.assertEqual(rows[0]["es_delta"]["cls"], "gain-strong")


class TestParity(unittest.TestCase):
    """Slow (boots Streamlit). The render_compare-style benchmark line + the
    estimator strip use st.markdown/st.caption, so they're covered by the
    engine-recompute gates; here we cross-check the st.metric tiles and the
    st.dataframe table at default controls."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = rcs.build_riskcontrib_view(cls.frames)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        at.session_state["active_tab"] = "Risk Contribution"
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_top_tiles_and_portfolio_metrics_present(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        metric_values = {m.value for m in self.at.metric}
        combo = self.view["combos"]["ewma_lw|0.05|0.0"]
        # the 4 top-contributor st.metric values + portfolio vol
        for t in combo["top_tiles"]:
            self.assertIn(t["value"], metric_values,
                          f"top tile {t['label']}={t['value']} not in Streamlit metrics")
        self.assertIn(combo["portfolio"]["vol"]["value"], metric_values)

    def test_table_first_row_matches_streamlit(self):
        if not self.view["state"]["available"]:
            self.skipTest("decomposition unavailable on fixture")
        # find the Top-15 dataframe (the one with a 'PCTR' column)
        df = None
        for d in self.at.dataframe:
            cols = list(getattr(d.value, "columns", []))
            if "PCTR" in cols and "Symbol" in cols:
                df = d.value
                break
        if df is None:
            self.skipTest("Risk Contribution table not rendered on fixture")
        row0 = self.view["combos"]["ewma_lw|0.05|0.0"]["table"]["rows"][0]
        self.assertEqual(str(df.iloc[0]["Symbol"]), row0["symbol"])
        self.assertEqual(f"{float(df.iloc[0]['Weight']):.2f}%", row0["weight"])
        self.assertEqual(f"{float(df.iloc[0]['PCTR']):.2f}%", row0["pctr"])

    def test_dr_tiles_present(self):
        dr = self.view["dr_in_context"]
        if not dr["available"]:
            self.skipTest("DR-in-context unavailable on fixture")
        metric_values = {m.value for m in self.at.metric}
        # DR short/medium/long + Max DR are st.metric values in app.py 6045-6064;
        # they are threshold-method-independent so they match at default controls.
        for t in dr["tiles"]:
            if t["value"] != "—":
                self.assertIn(t["value"], metric_values,
                              f"DR tile {t['label']}={t['value']} not in Streamlit metrics")

    def test_regime_unavailable_guard_matches(self):
        # The fixture has no vix_history.csv, so BOTH UIs show the regime
        # section's "need vix_history.csv" guard. (The populated grid is covered
        # by TestDrRegime + the real-data browser smoke — the fixture can't reach
        # it, same as the Benchmark slice's filtered-parity gap.)
        self.assertEqual(self.view["dr_regime"]["reason"], "no_inputs")
        alerts = [getattr(a, "value", "") for a in self.at.info] \
            + [getattr(a, "value", "") for a in self.at.warning]
        self.assertTrue(any("vix_history.csv" in str(v) for v in alerts),
                        "Streamlit regime no-inputs guard not found")


class TestDrInContext(unittest.TestCase):
    """Engine-recompute parity on a constructed 300-day panel (the synth
    fixture's DR series is clipped to ~22 portfolio days, so a longer panel
    exercises the ceiling + all-3-threshold paths deterministically)."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2024-01-01", periods=300)
        shared = rng.normal(0.0004, 0.011, size=(300, 1))      # common factor
        idio = rng.normal(0.0, 0.008, size=(300, 3))           # idiosyncratic
        rets = shared + idio                                    # -> DR > 1
        prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
        cls.daily = pd.DataFrame(prices, index=idx, columns=["AAA", "BBB", "SPY"])
        cls.weights = pd.Series({"AAA": 0.3, "BBB": 0.2, "SPY": 0.5})
        # port_rets only contributes its index.min() (the display clip); take a
        # 40-day tail so the clipped DR window sits well past the 252d burn-in.
        cls.port_rets = pd.Series(rets[:, 0], index=idx).tail(40)

    def test_available_and_tiles(self):
        from terminal import riskcontrib_dr as rcd
        from risk_metrics import compute_dr_time_series, compute_max_dr
        view = rcd.build_dr_in_context(self.weights, self.daily, self.port_rets)
        self.assertTrue(view["available"])
        dr = compute_dr_time_series(self.weights, self.daily, windows=(21, 63, 252))
        dr = dr.loc[dr.index >= self.port_rets.index.min()]
        last = dr.dropna(how="all").tail(1)
        dr_s = float(last["dr_21d"].iloc[0]); dr_l = float(last["dr_252d"].iloc[0])
        dr_m = float(dr["dr_63d"].dropna().iloc[-1])
        mx = compute_max_dr(self.weights, self.daily, window=252)
        self.assertEqual(view["tiles"][0]["value"], f"{dr_s:.2f}×")
        self.assertEqual(view["tiles"][1]["value"], f"{dr_m:.2f}×")
        self.assertEqual(view["tiles"][2]["value"], f"{dr_l:.2f}×")
        self.assertEqual(view["tiles"][3]["value"], f"{mx['max_dr']:.2f}×")
        self.assertIn(f"{mx['n_symbols']} symbols", view["tiles"][3]["sub"])

    def test_chart_and_ratio_series(self):
        from terminal import riskcontrib_dr as rcd
        view = rcd.build_dr_in_context(self.weights, self.daily, self.port_rets)
        series = view["dr_chart"]["series"]
        self.assertEqual(len(series), 4)            # short/med/long + ceiling
        self.assertEqual(len({len(s["points"]) for s in series}), 1)  # common index
        self.assertIsNotNone(view["dr_chart"]["baseline"])
        self.assertGreater(len(view["ratio_series"]), 0)

    def test_thresholds_three_methods(self):
        from terminal import riskcontrib_dr as rcd
        from risk_metrics import (compute_dr_time_series, compute_dr_ratio_series,
                                  compute_dr_regime_thresholds, classify_dr_regime)
        view = rcd.build_dr_in_context(self.weights, self.daily, self.port_rets)
        thr = view["thresholds"]
        self.assertEqual(set(thr), {"fixed", "percentile", "zscore"})
        self.assertAlmostEqual(thr["fixed"]["stress_thr"], 0.90)
        self.assertAlmostEqual(thr["fixed"]["calm_thr"], 1.10)
        dr = compute_dr_time_series(self.weights, self.daily, windows=(21, 63, 252))
        dr = dr.loc[dr.index >= self.port_rets.index.min()]
        ratio = compute_dr_ratio_series(dr, "dr_21d", "dr_252d")
        last = dr.dropna(how="all").tail(1)
        dr_s = float(last["dr_21d"].iloc[0]); dr_l = float(last["dr_252d"].iloc[0])
        for m in ("percentile", "zscore"):
            ti = compute_dr_regime_thresholds(ratio, method=m)
            self.assertAlmostEqual(thr[m]["stress_thr"], ti["stress_thr"])
            self.assertAlmostEqual(thr[m]["calm_thr"], ti["calm_thr"])
            reg = classify_dr_regime(dr_s, dr_l, stress_thr=ti["stress_thr"],
                                     calm_thr=ti["calm_thr"])
            self.assertEqual(thr[m]["regime"]["label"], reg["label"])
            self.assertAlmostEqual(thr[m]["gauge"]["value"], reg["ratio"])

    def test_empty_state_short_history(self):
        from terminal import riskcontrib_dr as rcd
        idx = pd.bdate_range("2024-01-01", periods=30)
        rng = np.random.default_rng(3)
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(30, 3)), axis=0)
        daily = pd.DataFrame(prices, index=idx, columns=["AAA", "BBB", "SPY"])
        port = pd.Series(rng.normal(0, 0.01, 30), index=idx).tail(10)
        view = rcd.build_dr_in_context(self.weights, daily, port)
        self.assertFalse(view["available"])
        self.assertIn("252 trading days", view["message"])
        self.assertEqual(view["tiles"], [])
        self.assertEqual([o["id"] for o in view["control"]["options"]],
                         ["fixed", "percentile", "zscore"])

    def test_compute_dr_frames_shape(self):
        from terminal import riskcontrib_dr as rcd
        f = rcd.compute_dr_frames(self.weights, self.daily, self.port_rets)
        self.assertEqual(set(f), {"dr_ts", "max_dr_ts", "ratio_ts",
                                  "available", "dr_s", "dr_l"})
        self.assertTrue(f["available"])
        self.assertFalse(f["dr_ts"].empty)
        self.assertGreater(f["ratio_ts"].dropna().shape[0], 0)
        self.assertTrue(np.isfinite(f["dr_s"]) and np.isfinite(f["dr_l"]))


class TestDrRegime(unittest.TestCase):
    """Constructed-panel engine-recompute (the fixture has no VIX/long-history,
    so it can't reach the populated grid — that path is covered here + by the
    real-data browser smoke)."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        idx = pd.bdate_range("2022-01-01", periods=700)
        shared = rng.normal(0.0003, 0.011, size=(700, 1))
        idio = rng.normal(0.0, 0.008, size=(700, 3))
        rets = shared + idio
        prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
        cls.daily = pd.DataFrame(prices, index=idx, columns=["AAA", "BBB", "SPY"])
        cls.weights = pd.Series({"AAA": 0.3, "BBB": 0.2, "SPY": 0.5})
        cls.port_rets = pd.Series(rets[:, 0], index=idx).tail(420)  # ~2y display window
        vix_level = 18.0 + 9.0 * np.abs(rng.normal(0, 1, size=700)).cumsum() / 40.0
        cls.vix = pd.Series(vix_level, index=idx, name="VIX")
        cls.long_history = pd.DataFrame()   # force the daily["SPY"] fallback

    def test_available_heatmap_and_recompute(self):
        from terminal import riskcontrib_regime as rcr
        from terminal.riskcontrib_dr import compute_dr_frames
        from risk_metrics import (classify_market_regime,
                                  compute_regime_conditional_dr, interpret_regime_dr,
                                  SPY_STATES, VIX_STATES)
        view = rcr.build_dr_regime(self.weights, self.daily, self.port_rets,
                                   self.vix, self.long_history)
        if not view["available"]:
            self.skipTest(f"constructed panel did not populate a grid: {view['reason']}")
        f = compute_dr_frames(self.weights, self.daily, self.port_rets)
        labels = classify_market_regime(spy_series=self.daily["SPY"], vix_series=self.vix)
        hw = "dr_63d"
        tail = f["dr_ts"][hw].dropna().tail(252)
        baseline = float(tail.mean())
        cond = compute_regime_conditional_dr(
            f["dr_ts"], labels, dr_ratio_series=f["ratio_ts"], min_n_per_cell=20,
            headline_window_col=hw, baseline_dr=baseline)
        interp = interpret_regime_dr(cond)
        self.assertEqual(view["character"]["headline"], interp["headline"])
        self.assertEqual(view["heatmap"]["rows"], list(SPY_STATES))
        self.assertEqual(view["heatmap"]["cols"], list(VIX_STATES))
        self.assertEqual(len(view["heatmap"]["cells"]), 3)
        self.assertTrue(all(len(r) == 3 for r in view["heatmap"]["cells"]))
        present = [c for row in view["heatmap"]["cells"] for c in row if c["present"]]
        self.assertTrue(present, "no populated cells")
        self.assertTrue(present[0]["color"].startswith("#") and len(present[0]["color"]) == 7)
        self.assertIn("N=", present[0]["text_html"])
        self.assertEqual(len(view["tiles"]), 3)
        json.dumps(view, allow_nan=False)

    def test_color_interp_endpoints(self):
        from terminal import riskcontrib_regime as rcr
        self.assertEqual(rcr._diverging_color(-0.4).upper(), "#FB6F63")
        self.assertEqual(rcr._diverging_color(0.0).upper(), "#E6B450")
        self.assertEqual(rcr._diverging_color(0.4).upper(), "#2FD79A")
        self.assertEqual(rcr._diverging_color(-99).upper(), "#FB6F63")  # clamps

    def test_no_inputs_when_vix_empty(self):
        from terminal import riskcontrib_regime as rcr
        view = rcr.build_dr_regime(self.weights, self.daily, self.port_rets,
                                   pd.Series(dtype=float), pd.DataFrame())
        self.assertFalse(view["available"])
        self.assertEqual(view["reason"], "no_inputs")
        self.assertIn("vix_history.csv", view["message"])


class TestCorrColor(unittest.TestCase):
    def test_corr_color_endpoints_and_clamp(self):
        from terminal import riskcontrib_corr as rcc
        # Stretched fixed scale: -0.6 -> full teal, 1.0 -> full coral.
        self.assertEqual(rcc._corr_color(-0.6).upper(), "#2FD79A")
        self.assertEqual(rcc._corr_color(1.0).upper(), "#FB6F63")
        # clamp beyond the anchors
        self.assertEqual(rcc._corr_color(9.0).upper(), "#FB6F63")
        self.assertEqual(rcc._corr_color(-9.0).upper(), "#2FD79A")
        # NaN / None -> None (front-end paints a neutral cell)
        self.assertIsNone(rcc._corr_color(float("nan")))
        self.assertIsNone(rcc._corr_color(None))

    def test_corr_t_stretch_widens_crowded_band(self):
        from terminal import riskcontrib_corr as rcc
        # Monotone through the anchors, 0..1 at the ends.
        vals = [-0.6, -0.2, 0.0, 0.4, 0.7, 1.0]
        ts = [rcc._corr_t(v) for v in vals]
        self.assertEqual(ts, sorted(ts))
        self.assertAlmostEqual(ts[0], 0.0)
        self.assertAlmostEqual(ts[-1], 1.0)
        # The crowded 0.4..1.0 band gets over half the ramp (the point of the
        # stretch); linear on [-0.6, 1.0] would give it 0.375.
        self.assertGreater(ts[-1] - ts[3], 0.5)

    def test_corr_gradient_is_css(self):
        from terminal import riskcontrib_corr as rcc
        g = rcc._corr_gradient()
        self.assertTrue(g.startswith("linear-gradient(to right,"))
        self.assertIn("#2FD79A", g)  # teal (low) ... coral (high)
        self.assertIn("#FB6F63", g)


class TestCorrMajor(unittest.TestCase):
    """Constructed-panel engine-recompute — the fixture has no long_history, so
    the primary Big-3 + full 15×15 paths are covered here + real-data smoke."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        # long_history: ~2600 bdays of SPY/SGOV/GLD/BIL for the primary Big-3 path
        lidx = pd.bdate_range("2014-01-01", periods=2600)
        lr = rng.normal(0.0003, 0.01, size=(2600, 4))
        lpx = 100.0 * np.cumprod(1.0 + lr, axis=0)
        cls.long_history = pd.DataFrame(lpx, index=lidx,
                                        columns=["SPY", "SGOV", "GLD", "BIL"])
        # daily: last ~520 bdays, Big-3 + 5 holdings
        didx = pd.bdate_range("2023-01-01", periods=520)
        dr = rng.normal(0.0004, 0.011, size=(520, 8))
        dpx = 100.0 * np.cumprod(1.0 + dr, axis=0)
        cols = ["SPY", "SGOV", "GLD", "AAA", "BBB", "CCC", "DDD", "EEE"]
        cls.daily = pd.DataFrame(dpx, index=didx, columns=cols)
        cls.port_rets = pd.Series(dr[:, 0], index=didx).tail(400)
        # vol_blocks stub: per_symbol index (PCTR order) + differing n_days
        order = ["AAA", "BBB", "SPY", "CCC", "GLD", "DDD", "EEE", "SGOV"]
        per = pd.DataFrame(index=pd.Index(order, name="symbol"))
        cls.vol_blocks = {
            "ewma_lw": {"per_symbol": per, "n_days": 504},
            "ewma": {"per_symbol": per, "n_days": 504},
            "rolling": {"per_symbol": per, "n_days": 252},
        }

    def _build(self):
        from terminal import riskcontrib_corr as rcc
        return rcc.build_major_correlations(
            self.vol_blocks, self.daily, self.long_history, self.port_rets)

    def test_big3_primary_heatmap_matches_engine(self):
        from risk_metrics import compute_correlation_matrix, splice_sgov_with_bil
        view = self._build()["major"]["big3"]
        self.assertTrue(view["available"])
        self.assertFalse(view["fallback"])
        hm = view["heatmap"]
        self.assertEqual(hm["rows"], ["SPY", "SGOV", "GLD"])
        self.assertEqual(hm["cols"], ["SPY", "SGOV", "GLD"])
        self.assertEqual(hm["cells"][0][0]["text_html"], "1.00")  # diagonal
        self.assertTrue(hm["cells"][0][0]["color"].startswith("#"))
        # independent recompute of an off-diagonal cell
        sgov_ext = splice_sgov_with_bil(self.long_history)
        big3 = pd.concat({"SPY": self.long_history["SPY"].dropna(),
                          "SGOV": sgov_ext.dropna(),
                          "GLD": self.long_history["GLD"].dropna()},
                         axis=1).dropna(how="all")
        corr = compute_correlation_matrix(big3, ["SPY", "SGOV", "GLD"])
        self.assertEqual(hm["cells"][0][1]["text_html"],
                         f"{corr.loc['SPY', 'SGOV']:.2f}")
        json.dumps(view, allow_nan=False)

    def test_big3_rolling_series_and_windows(self):
        view = self._build()["major"]["big3"]
        roll = view["rolling"]
        self.assertIsNotNone(roll)
        self.assertEqual(len(roll["series"]), 3)
        self.assertEqual({s["name"] for s in roll["series"]},
                         {"SPY–SGOV", "SPY–GLD", "SGOV–GLD"})
        self.assertEqual([o["id"] for o in roll["window_options"]],
                         ["All", "5y", "3y", "2y", "1y", "2025+", "YTD"])
        self.assertIsNone(roll["window_options"][0]["start"])  # "All"
        self.assertTrue(all(p["v"] is None or -1.0 <= p["v"] <= 1.0
                            for p in roll["series"][0]["points"]))

    def test_big3_fallback_when_no_long_history(self):
        from terminal import riskcontrib_corr as rcc
        view = rcc.build_major_correlations(
            self.vol_blocks, self.daily, pd.DataFrame(), self.port_rets
        )["major"]["big3"]
        self.assertTrue(view["fallback"])
        self.assertEqual(view["reason"], "no_long_history")
        self.assertIn("fetch_long_history", view["message"])
        # SPY/SGOV/GLD are in the constructed daily -> a fallback heatmap renders
        self.assertTrue(view["available"])
        self.assertIsNone(view["rolling"])

    def test_top15_per_estimator_matches_engine(self):
        from risk_metrics import compute_correlation_matrix
        top15 = self._build()["major"]["top15"]
        self.assertEqual(set(top15), {"ewma_lw", "ewma", "rolling"})
        blk = top15["rolling"]  # n_days = 252
        self.assertTrue(blk["available"])
        order = blk["heatmap"]["rows"]
        # membership = top-15 PCTR order restricted to daily columns (all 8
        # fixture symbols are present; app.py applies no SGOV-specific exclusion)
        self.assertEqual(order,
                         [s for s in ["AAA", "BBB", "SPY", "CCC", "GLD", "DDD", "EEE", "SGOV"]
                          if s in self.daily.columns][:15])
        corr = compute_correlation_matrix(self.daily.tail(252), order)
        self.assertEqual(blk["heatmap"]["cells"][0][1]["text_html"],
                         f"{corr.loc[order[0], order[1]]:.2f}")
        self.assertIsNotNone(blk["avg_roll"])
        self.assertEqual(len(blk["avg_roll"]["series"]), 1)

    def test_top15_window_differs_by_estimator(self):
        top15 = self._build()["major"]["top15"]
        # rolling uses n_days=252, ewma_lw uses 504 -> different windows ->
        # generally different off-diagonal cell values (1:1 with Streamlit rc)
        a = top15["rolling"]["heatmap"]["cells"][0][1]["text_html"]
        b = top15["ewma_lw"]["heatmap"]["cells"][0][1]["text_html"]
        # both present; the point is they are computed independently per estimator
        self.assertNotEqual(top15["rolling"]["caption_html"],
                            top15["ewma_lw"]["caption_html"])  # window count differs

    def test_top15_insufficient_names(self):
        from terminal import riskcontrib_corr as rcc
        per = pd.DataFrame(index=pd.Index(["AAA", "ZZZ"], name="symbol"))  # <3 in daily
        blk = rcc._top15_for(per, 252, self.daily, self.port_rets)
        self.assertFalse(blk["available"])
        self.assertEqual(blk["reason"], "insufficient_names")
        self.assertIsNone(blk["heatmap"])


class TestCorrStress(unittest.TestCase):
    def test_delta_color_endpoints_and_clamp(self):
        from terminal import riskcontrib_corr as rcc
        self.assertEqual(rcc._delta_color(-0.5).upper(), "#2FD79A")  # fell -> teal (good)
        self.assertEqual(rcc._delta_color(0.5).upper(), "#FB6F63")   # spiked -> coral (bad)
        self.assertEqual(rcc._delta_color(0.0).upper(), "#E6B450")   # unchanged -> amber
        self.assertEqual(rcc._delta_color(9.0).upper(), "#FB6F63")   # clamp +
        self.assertEqual(rcc._delta_color(-9.0).upper(), "#2FD79A")  # clamp -
        self.assertIsNone(rcc._delta_color(float("nan")))
        self.assertIsNone(rcc._delta_color(None))

    def test_delta_gradient_is_css_teal_to_coral(self):
        from terminal import riskcontrib_corr as rcc
        g = rcc._delta_gradient()
        self.assertTrue(g.startswith("linear-gradient(to right, #2FD79A"))  # teal (lo)
        self.assertTrue(g.rstrip(") ").endswith("#FB6F63"))                  # coral (hi)

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        lidx = pd.bdate_range("2014-01-01", periods=2600)
        lr = rng.normal(0.0003, 0.011, size=(2600, 4))
        lpx = 100.0 * np.cumprod(1.0 + lr, axis=0)
        cls.long_history = pd.DataFrame(lpx, index=lidx,
                                        columns=["SPY", "SGOV", "GLD", "BIL"])
        didx = pd.bdate_range("2022-01-01", periods=520)
        dr = rng.normal(0.0004, 0.012, size=(520, 8))
        dpx = 100.0 * np.cumprod(1.0 + dr, axis=0)
        cls.daily = pd.DataFrame(
            dpx, index=didx,
            columns=["SPY", "SGOV", "GLD", "AAA", "BBB", "CCC", "DDD", "EEE"])
        order = ["AAA", "BBB", "SPY", "CCC", "GLD", "DDD", "EEE", "SGOV"]
        per = pd.DataFrame(index=pd.Index(order, name="symbol"))
        cls.vol_blocks = {
            "ewma_lw": {"per_symbol": per, "n_days": 504},
            "ewma": {"per_symbol": per, "n_days": 504},
            "rolling": {"per_symbol": per, "n_days": 252},
        }

    def _build(self):
        from terminal import riskcontrib_corr as rcc
        return rcc.build_stress_correlations(
            self.vol_blocks, self.daily, self.long_history)

    def test_stress_big3_matches_engine(self):
        from terminal import riskcontrib_corr as rcc
        from risk_metrics import compute_conditional_correlation_matrix
        big3 = self._build()["big3"]
        self.assertTrue(big3["available"])
        hm = big3["heatmap"]
        self.assertEqual(hm["rows"], ["SPY", "SGOV", "GLD"])
        self.assertEqual(hm["legend"]["title"], "Δρ")
        self.assertEqual(hm["cells"][0][0]["text_html"], "0.00")   # diagonal Δ
        frame = rcc._big3_frame(self.long_history)
        cond = compute_conditional_correlation_matrix(
            frame, ["SPY", "SGOV", "GLD"], condition_symbol="SPY", z_threshold=-1.5)
        self.assertTrue(cond["enough"])
        self.assertEqual(hm["cells"][0][1]["text_html"],
                         f"{cond['delta'].loc['SPY', 'SGOV']:.2f}")
        self.assertIn(f"{cond['n_stress']:,}", big3["caption_html"])
        json.dumps(self._build(), allow_nan=False)

    def test_stress_top15_per_estimator(self):
        top15 = self._build()["top15"]
        self.assertEqual(set(top15), {"ewma_lw", "ewma", "rolling"})
        blk = top15["rolling"]
        self.assertTrue(blk["available"])
        self.assertEqual(blk["heatmap"]["legend"]["title"], "Δρ")
        self.assertTrue(blk["howto_html"])
        # window differs by estimator (252 vs 504) -> different caption
        self.assertNotEqual(top15["rolling"]["caption_html"],
                            top15["ewma_lw"]["caption_html"])
        # independent engine-recompute of a Top-15 Δ cell (symmetry with Big-3)
        from risk_metrics import compute_conditional_correlation_matrix
        order = blk["heatmap"]["rows"]
        cond = compute_conditional_correlation_matrix(
            self.daily.tail(252), order, condition_symbol="SPY", z_threshold=-1.5)
        self.assertTrue(cond["enough"])
        self.assertEqual(blk["heatmap"]["cells"][0][1]["text_html"],
                         f"{cond['delta'].loc[order[0], order[1]]:.2f}")

    def test_stress_top15_insufficient_names(self):
        from terminal import riskcontrib_corr as rcc
        per = pd.DataFrame(index=pd.Index(["AAA", "ZZZ"], name="symbol"))
        blk = rcc._top15_stress(per, 252, self.daily)
        self.assertFalse(blk["available"])
        self.assertEqual(blk["reason"], "insufficient_names")
        self.assertIsNone(blk["heatmap"])

    def test_stress_big3_no_inputs_silent(self):
        from terminal import riskcontrib_corr as rcc
        big3 = rcc._big3_stress(pd.DataFrame())   # no long_history
        self.assertFalse(big3["available"])
        self.assertEqual(big3["reason"], "no_inputs")
        self.assertIsNone(big3["message"])        # app.py has no else -> no callout
        self.assertIsNone(big3["heatmap"])

    def test_big3_all_nan_trio_gate_matches_streamlit(self):
        # trio columns PRESENT but all-NaN: static -> insufficient_overlap (NOT
        # the no-long-history fallback); stress -> insufficient_stress "have 0"
        # (NOT no_inputs). app.py gates Big-3 on column-presence, so parity needs
        # the column check, not frame-emptiness (regression guard for the 3b refactor).
        from terminal import riskcontrib_corr as rcc
        idx = pd.bdate_range("2015-01-01", periods=300)
        lh = pd.DataFrame({"SPY": np.nan, "SGOV": np.nan, "GLD": np.nan,
                           "BIL": np.nan}, index=idx)
        st = rcc._big3(lh, self.daily)
        self.assertFalse(st["fallback"])
        self.assertEqual(st["reason"], "insufficient_overlap")
        ss = rcc._big3_stress(lh)
        self.assertEqual(ss["reason"], "insufficient_stress")
        self.assertIn("have 0", ss["message"])


class TestRegimeLoaderTotalReturnBasis(unittest.TestCase):
    """DA-B-1: the regime long-history loader must read the same total-
    return basis as every other close-matrix loader. The 2026-08-22 TR
    switch missed exactly this one — the terminal's Big-3 correlations
    and DR-regime conditioning ran price-only while Streamlit ran TR
    (SGOV vol 3.4x apart, one live correlation sign flip) — so the two
    long-history loaders are pinned EQUAL on a dir where the adjustment
    is not a no-op (one covered symbol with one distribution)."""

    def test_regime_loader_matches_holdings_loader_with_dividends(self):
        from tempfile import TemporaryDirectory

        from terminal import riskcontrib_regime as rcr

        with TemporaryDirectory() as td:
            d = Path(td)
            dates = pd.bdate_range("2025-06-16", periods=6)
            rows = []
            for i, dt in enumerate(dates):
                rows.append({"date": dt.date(), "symbol": "DIVY",
                             "close": 100.0})
                rows.append({"date": dt.date(), "symbol": "NOPE",
                             "close": 50.0 + i})
            pd.DataFrame(rows).to_csv(d / "long_history_prices.csv",
                                      index=False)
            pd.DataFrame([{
                "cash_amount": 1.0, "currency": "USD",
                "declaration_date": "2025-06-10", "dividend_type": "CD",
                "ex_dividend_date": str(dates[3].date()), "frequency": 4,
                "id": "T1", "pay_date": "2025-07-01",
                "record_date": str(dates[4].date()), "ticker": "DIVY",
            }]).to_csv(d / "dividends_divy.csv", index=False)
            got = rcr._load_long_history(d)
            want = hs._load_long_history_prices(d)
            pd.testing.assert_frame_equal(got, want)
            # and the basis is genuinely total-return, not price-only:
            # reinvesting the distribution rebases pre-ex closes DOWN
            self.assertLess(got["DIVY"].iloc[0], 100.0 - 1e-9)


if __name__ == "__main__":
    unittest.main()
