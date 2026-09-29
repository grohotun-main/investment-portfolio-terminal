# tests/test_terminal_factor.py
import dataclasses
import json
import math
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import factor_service as fs
from factor_regression import (FACTOR_LABELS, MODELS, align_returns_with_factors,
                               align_twr_with_factors, attribution,
                               attribution_timeseries, per_holding_regressions,
                               rolling_factor_regressions, run_factor_regression)
from risk_metrics import synthesize_portfolio_returns


def _aligned_full(frames):
    ff_m, _ = fs._load_ff(frames.data_dir)
    return align_twr_with_factors(frames.twr_portfolio, ff_m, None)


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare; returns None if equal else a
    path describing the first mismatch.

    The factor numbers are ``np.linalg.lstsq`` (SVD) outputs, which are NOT
    bit-reproducible across LAPACK builds (Windows dev box vs Linux CI) — the last
    few ULPs drift. So the golden pins STRUCTURE (keys / list lengths) and every
    FORMATTED string exactly, but compares raw floats with a relative tolerance.
    The engine-parity tests (which recompute on the SAME machine the view is built
    on) stay the tight numeric gate; this golden guards shape + formatting +
    approximate magnitude. (The earlier tabs' goldens compare exactly because
    their math is elementwise/sum arithmetic, which IS bit-stable; Factor is the
    first tab whose golden holds SVD outputs.)"""
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a is b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            m = _deep_close(a[k], b[k], rel=rel, abs_=abs_, path=f"{path}.{k}")
            if m:
                return m
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = _deep_close(x, y, rel=rel, abs_=abs_, path=f"{path}[{i}]")
            if m:
                return m
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (None if math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
                else f"{path}: {a!r} !~ {b!r}")
    return None if a == b else f"{path}: {a!r} != {b!r}"


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = fs.build_factor_view(cls.frames)

    def test_contract_keys(self):
        self.assertEqual(set(self.view.keys()),
                         {"meta", "caption", "state", "by_window", "methodology"})

    def test_state_vocab(self):
        st = self.view["state"]
        self.assertTrue(st["available"])
        self.assertEqual(st["windows"],
                         ["Full history", "Last 60 months", "Last 36 months"])
        self.assertEqual(st["models"],
                         ["CAPM", "FF3", "Carhart 4", "FF5", "FF5 + Mom"])
        self.assertEqual(st["default_model"], "FF5 + Mom")
        self.assertEqual(st["default_window"], "Full history")
        self.assertEqual(st["roll_windows"], [24, 36, 60])
        self.assertEqual(st["default_roll"], 36)
        self.assertEqual(st["default_attr_view"], "Cumulative")

    def test_every_window_has_all_models(self):
        for w in self.view["state"]["windows"]:
            wb = self.view["by_window"][w]
            if wb["aligned_empty"]:
                continue
            self.assertEqual(list(wb["models"].keys()), self.view["state"]["models"])
            self.assertEqual([e["model"] for e in wb["strip"]],
                             self.view["state"]["models"])

    def test_attribution_series_aligned(self):
        for w in self.view["state"]["windows"]:
            wb = self.view["by_window"][w]
            if wb["aligned_empty"]:
                continue
            for m, mb in wb["models"].items():
                if not mb["available"]:
                    continue
                attr = mb["attribution"]
                for s in attr["series"]:
                    self.assertEqual(len(s["values"]), len(attr["x"]),
                                     f"{w}/{m} attr series {s['name']} ragged")

    def test_rolling_gaps_are_null_not_nan(self):
        # Any structurally-skipped rolling window must serialize as null, never
        # a NaN token (which allow_nan=False forbids and JSON.parse rejects).
        body = json.dumps(self.view, allow_nan=False)  # raises if any NaN leaks
        self.assertNotIn("NaN", body)

    def test_full_view_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)


class TestEngineParity(unittest.TestCase):
    """Terminal values equal an independent recompute of the engine on the same
    inputs — the '1:1 numbers' guarantee at the data layer. Exercises the Full
    history window for every model + the default model's daily sections."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = fs.build_factor_view(cls.frames)
        cls.aligned = _aligned_full(cls.frames)
        cls.results = {m: run_factor_regression(cls.aligned, m) for m in MODELS}
        cls.wb = cls.view["by_window"]["Full history"]
        _, cls.ff_d = fs._load_ff(cls.frames.data_dir)
        cls.weights = fs._factor_weights(cls.frames)

    def test_strip_values(self):
        for e in self.wb["strip"]:
            res = self.results[e["model"]]
            if res is None:
                self.assertEqual(e["value"], "—")
            else:
                self.assertEqual(e["value"], f"{res.alpha_annual * 100:+.1f}%")
                lo, hi = res.alpha_ci_annual
                self.assertEqual(e["delta"], f"±{(hi - lo) / 2 * 100:.1f} pp")

    def test_detail_metrics_and_betas(self):
        res = self.results["FF5 + Mom"]
        mb = self.wb["models"]["FF5 + Mom"]
        self.assertEqual(mb["metrics"],
                         {"n": f"{res.n}", "r2": f"{res.r2:.2f}",
                          "adj_r2": f"{res.adj_r2:.2f}"})
        self.assertEqual(len(mb["beta_numeric"]), len(res.factors))
        for bn, c in zip(mb["beta_numeric"], res.factors):
            self.assertEqual(bn["factor"], c)
            self.assertAlmostEqual(bn["beta"], res.betas[c], places=9)
            self.assertAlmostEqual(bn["se"], res.se[c], places=9)
            self.assertAlmostEqual(bn["t"], res.tstats[c], places=9)
        # rendered strings match the .style.format rules
        for row, c in zip(mb["beta_table"]["rows"], res.factors):
            self.assertEqual(row["Factor"], FACTOR_LABELS[c])
            self.assertEqual(row["β"], f"{res.betas[c]:+.2f}")
            self.assertEqual(row["t"], f"{res.tstats[c]:+.2f}")
            self.assertEqual(row["Significant (|t|>2)"],
                             "✓" if abs(res.tstats[c]) > 2 else "")

    def test_waterfall(self):
        res = self.results["FF5 + Mom"]
        mb = self.wb["models"]["FF5 + Mom"]
        contribs = attribution(res)
        self.assertEqual(len(mb["waterfall"]["items"]), len(contribs))
        for item, (lab, v) in zip(mb["waterfall"]["items"], contribs):
            self.assertEqual(item["label"], lab)
            self.assertAlmostEqual(item["value_pp"], v * 100, places=9)
            self.assertEqual(item["text"], f"{v * 100:+.1f}")
        self.assertAlmostEqual(mb["waterfall"]["total_pp"],
                               res.mean_return_monthly * 12 * 100, places=9)

    def test_attribution_series(self):
        res = self.results["FF5 + Mom"]
        mb = self.wb["models"]["FF5 + Mom"]
        ats = attribution_timeseries(res, self.aligned)
        attr = mb["attribution"]
        self.assertEqual(attr["x"], [str(m) for m in ats["month"]])
        by_key = {s["key"]: s["values"] for s in attr["series"]}
        self.assertEqual(by_key["rf"], [float(v) for v in ats["rf"]])
        self.assertEqual(by_key["unexplained"],
                         [float(v) for v in ats["unexplained"]])
        for c in res.factors:
            self.assertEqual(by_key[f"contrib_{c}"],
                             [float(v) for v in ats[f"contrib_{c}"]])

    def test_rolling(self):
        mb = self.wb["models"]["FF5 + Mom"]
        roll = rolling_factor_regressions(self.aligned, "FF5 + Mom", 36)
        r36 = mb["rolling"]["by_roll"]["36"]
        self.assertTrue(r36["available"])
        self.assertEqual(r36["x"], [str(m) for m in roll["month"]])
        by_name = {s["name"]: s["values"] for s in r36["series"]}
        # NaN engine rows become JSON null
        exp_alpha = [None if (v is None or (isinstance(v, float) and math.isnan(v)))
                     else float(v) for v in roll["alpha_annual"]]
        self.assertEqual(by_name["α (annualized)"], exp_alpha)

    def test_per_holding(self):
        mb = self.wb["models"]["FF5 + Mom"]
        ph = mb["per_holding"]
        if not ph["available"]:
            self.skipTest("per-holding empty on fixture")
        roster = [s for s in self.weights.index
                  if s in self.frames.daily_prices.columns]
        ph_table, _ = per_holding_regressions(
            self.frames.daily_prices[roster], self.ff_d, "FF5 + Mom", None)
        disp = ph_table.copy()
        disp.insert(1, "weight", disp["symbol"].map(self.weights))
        disp = disp.sort_values("weight", ascending=False)
        self.assertEqual([r["Symbol"] for r in ph["table"]["rows"]],
                         [str(s) for s in disp["symbol"]])
        self.assertEqual([r["Weight"] for r in ph["table"]["rows"]],
                         [f"{w:.1%}" for w in disp["weight"]])

    def test_cross_check(self):
        mb = self.wb["models"]["FF5 + Mom"]
        cc = mb["cross_check"]
        if not cc["available"]:
            self.skipTest("cross-check unavailable on fixture")
        res_m = self.results["FF5 + Mom"]
        port_daily = synthesize_portfolio_returns(self.weights,
                                                  self.frames.daily_prices)
        aligned_d = align_returns_with_factors(port_daily, self.ff_d, None)
        res_d = run_factor_regression(aligned_d, "FF5 + Mom", periods_per_year=252)
        beta_rows = [r for r in cc["table"]["rows"]
                     if r["Metric"].startswith("β ")]
        for r, c in zip(beta_rows, res_m.factors):
            self.assertEqual(r["Monthly (real TWR)"], f"{res_m.betas[c]:+.2f}")
            self.assertEqual(r["Daily (current wts, synthetic)"],
                             f"{res_d.betas[c]:+.2f}")


class TestEmptyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_no_twr(self):
        frames2 = dataclasses.replace(self.frames,
                                      twr_portfolio=self.frames.twr_portfolio.iloc[0:0])
        view = fs.build_factor_view(frames2)
        self.assertFalse(view["state"]["available"])
        self.assertEqual(view["state"]["unavailable"], "no_twr")
        self.assertEqual(view["by_window"], {})

    def test_no_factor_files(self):
        # A data dir with no ff_factors_*.csv -> no_factors top-level state.
        empty_dir = ROOT / "tests" / "fixtures"  # no ff_factors here
        frames2 = dataclasses.replace(self.frames, data_dir=str(empty_dir))
        view = fs.build_factor_view(frames2)
        self.assertFalse(view["state"]["available"])
        self.assertEqual(view["state"]["unavailable"], "no_factors")
        self.assertEqual(view["by_window"], {})


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

    def test_factor_ok(self):
        r = self.client.get("/api/factor")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("by_window", body)
        self.assertIn("state", body)

    def test_unknown_params_ignored_not_500(self):
        r = self.client.get("/api/factor", params={"account": "nope"})
        self.assertEqual(r.status_code, 200)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/factor")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_factor_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = fs.build_factor_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        # Float-tolerant (lstsq/SVD outputs aren't bit-reproducible across
        # platforms); structure + formatted strings are still pinned exactly.
        mismatch = _deep_close(view, expected)
        self.assertIsNone(mismatch,
                          f"factor view diverges from golden at {mismatch}")


class TestDailyPriceSplice(unittest.TestCase):
    """The Factor tab's per-holding betas + cross-check read the daily-price
    matrix through daily regressions, so a renamed ticker's prior-symbol history
    must be spliced in (app.py.load_daily_prices does this; the terminal must
    match 1:1 on real data). The synth fixture has NO renamed ticker, so the
    golden/AppTest gates can't see this path — exercise it directly here."""

    def test_load_daily_prices_applies_ticker_history_splice(self):
        import csv
        import tempfile
        import config_local as cfg
        from risk_metrics import splice_ticker_history

        tmp = Path(tempfile.mkdtemp())
        # Prior symbol COMM straddling a (synthetic) rename to VISN; the raw file
        # has NO VISN column — the splice must graft one from COMM's history.
        rows = [("COMM", "2026-01-05", 10.0), ("COMM", "2026-01-06", 11.0),
                ("SPY", "2026-01-05", 400.0), ("SPY", "2026-01-06", 401.0)]
        with open(tmp / "daily_prices.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "date", "close"])
            w.writerows(rows)

        hist = {"VISN": [{"prior_symbol": "COMM", "effective_date": "2026-06-01"}]}
        saved = getattr(cfg, "TICKER_HISTORY", {})
        cfg.TICKER_HISTORY = hist
        try:
            got = hs._load_daily_prices(tmp)
            raw = (pd.read_csv(tmp / "daily_prices.csv", parse_dates=["date"])
                   .pivot(index="date", columns="symbol", values="close")
                   .sort_index())
            expected = splice_ticker_history(raw, hist)
        finally:
            cfg.TICKER_HISTORY = saved
        # The splice fired (VISN didn't exist in the raw file) AND the loader's
        # output is exactly the engine's spliced matrix — i.e. the wiring is live.
        self.assertIn("VISN", got.columns,
                      "splice not applied — VISN not grafted from COMM")
        pd.testing.assert_frame_equal(got, expected)


class TestOrchestrationSeams(unittest.TestCase):
    """Unit-pin each orchestration seam app.py and the service now share, by
    recomputing the same engine-call sequence directly (tight numeric parity —
    same machine, so bit-identical)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.ff_m, cls.ff_d = fs._load_ff(cls.frames.data_dir)
        cls.weights = fs._factor_weights(cls.frames)
        cls.model = list(MODELS)[-1]

    def test_factor_results_matches_direct(self):
        aligned, results = fs.factor_results(
            self.frames.twr_portfolio, self.ff_m, "Full history")
        exp_aligned = align_twr_with_factors(
            self.frames.twr_portfolio, self.ff_m, None)
        pd.testing.assert_frame_equal(aligned, exp_aligned)
        self.assertEqual(list(results), list(MODELS))
        exp = run_factor_regression(exp_aligned, "CAPM")
        self.assertEqual(results["CAPM"].alpha_annual, exp.alpha_annual)

    def test_factor_results_window_applies(self):
        aligned, _ = fs.factor_results(
            self.frames.twr_portfolio, self.ff_m, "Last 36 months")
        exp = align_twr_with_factors(self.frames.twr_portfolio, self.ff_m, 36)
        pd.testing.assert_frame_equal(aligned, exp)

    def test_per_holding_result_matches_direct(self):
        disp, skipped = fs.per_holding_result(
            self.weights, self.frames.daily_prices, self.ff_d, self.model, None)
        roster = [s for s in self.weights.index
                  if s in self.frames.daily_prices.columns]
        ph, ph_sk = per_holding_regressions(
            self.frames.daily_prices[roster], self.ff_d, self.model, None)
        if ph.empty:
            self.assertTrue(disp.empty)
        else:
            exp = ph.copy()
            exp.insert(1, "weight", exp["symbol"].map(self.weights))
            exp = exp.sort_values("weight", ascending=False)
            pd.testing.assert_frame_equal(disp, exp)
        pd.testing.assert_frame_equal(skipped, ph_sk)

    def test_cross_check_daily_matches_direct(self):
        res_d, n = fs.cross_check_daily(
            self.weights, self.frames.daily_prices, self.ff_d, self.model, None)
        port = synthesize_portfolio_returns(self.weights, self.frames.daily_prices)
        aligned_d = align_returns_with_factors(port, self.ff_d, None)
        exp = run_factor_regression(aligned_d, self.model, periods_per_year=252)
        self.assertEqual(n, len(aligned_d))
        if exp is None:
            self.assertIsNone(res_d)
        else:
            self.assertEqual(res_d.alpha_annual, exp.alpha_annual)
            self.assertEqual(res_d.n, exp.n)

    def test_per_holding_result_empty_roster(self):
        # A weights index disjoint from the price columns -> empty roster -> the
        # seam returns the raw empty ph_table BEFORE the weight-insert/sort (the
        # fixture never trips this via _factor_weights, so pin it directly).
        disp, skipped = fs.per_holding_result(
            pd.Series({"ZZZZ_NOPE": 1.0}), self.frames.daily_prices, self.ff_d,
            self.model, None)
        self.assertTrue(disp.empty)
        self.assertTrue(skipped.empty)

    def test_cross_check_daily_too_few_days(self):
        # A 5-day window can't fit a multi-factor model -> run_factor_regression
        # returns None; the seam passes it through with n = the aligned count.
        res_d, n = fs.cross_check_daily(
            self.weights, self.frames.daily_prices, self.ff_d, self.model, 5)
        self.assertIsNone(res_d)
        self.assertEqual(n, 5)


class TestCaptionBasisHonesty(unittest.TestCase):
    """DA-G-1/G-2: the 2026-08-22 total-return switch flipped the daily
    panels to TR closes but left this module's captions describing the
    price-only world ("α is understated by each name's distribution
    yield", "SGOV prints α ≈ −RF" — on screen SGOV prints α ≈ 0), and the
    golden regenerated in the same commit re-blessed the stale text (31
    occurrences). Golden regeneration re-blesses whatever the source
    emits, so the claim is pinned HERE, at the source: no factor caption
    may assert the retired price-only basis. ("stays price-only" for a
    name without a dividend file remains a true, allowed statement.)"""

    BANNED = ("price-only returns", "price-only synthesis",
              "(no dividends)", "omits dividends", "α ≈ −RF")

    def test_factor_service_captions_do_not_claim_price_only(self):
        import inspect
        src = inspect.getsource(fs)
        for phrase in self.BANNED:
            self.assertNotIn(phrase, src, f"stale price-only claim: {phrase!r}")

    def test_terminal_static_chrome_does_not_claim_price_only(self):
        # The live smoke for this fix found a FIFTH stale surface the
        # audit's own count missed: a static section subtitle in the
        # front-end ("daily price-only returns"). Server-side caption
        # tests can't see FE literals, so scan the static files too.
        static = ROOT / "terminal" / "static"
        for name in ("index.html", "app.js"):
            src = (static / name).read_text(encoding="utf-8")
            for phrase in self.BANNED:
                self.assertNotIn(phrase, src,
                                 f"{name}: stale claim {phrase!r}")


if __name__ == "__main__":
    unittest.main()
