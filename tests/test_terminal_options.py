# tests/test_terminal_options.py
import dataclasses
import json
import math
import os
import sys
import types
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import options_service as ops
from option_positions import option_book_aggregates
from iv_rank import book_iv_percentile

# Pinned so the golden + recompute parity are deterministic (DTE, Greeks and the
# staleness age are genuinely clock-dependent). The live route + AppTest parity use
# the real today/now instead, so they agree with the Streamlit body on the same run.
TODAY = date(2026, 5, 1)
NOW = pd.Timestamp("2026-05-01T17:00:00+00:00")
IV_PCT_WINDOW_DAYS = 252


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (the #204 golden helper:
    binomial-pricer Greeks aren't bit-reproducible across BLAS builds)."""
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
        cls.view = ops.build_options_view(cls.frames, today=TODAY, now=NOW)

    def test_contract_keys(self):
        self.assertEqual(
            set(self.view.keys()),
            {"meta", "empty", "empty_message", "staleness",
             "aggregates", "iv_percentile", "footer"})

    def test_meta_keys(self):
        for k in ("title", "caption", "accounts", "classes",
                  "available_dates", "synthetic", "filter"):
            self.assertIn(k, self.view["meta"])

    def test_not_empty_on_fixture(self):
        # The synth fixture carries a parseable long put (#123), so the read half
        # is populated, not the empty state.
        self.assertFalse(self.view["empty"])
        self.assertIsNone(self.view["empty_message"])
        self.assertIsNotNone(self.view["aggregates"])

    def test_staleness_always_present(self):
        for key in ("snapshot", "atm_iv"):
            self.assertIn(key, self.view["staleness"])
            self.assertIn("chip", self.view["staleness"][key])
            self.assertIsInstance(self.view["staleness"][key]["chip"], str)

    def test_aggregates_keys(self):
        agg = self.view["aggregates"]
        for k in ("notional_protected", "notional_pct_nav", "premium_at_risk",
                  "cost_basis", "n_excluded", "unrealized_pnl", "pnl_pct_cost",
                  "weighted_dte", "gamma_dollar", "vega_dollar", "theta_dollar",
                  "weighted_iv", "greeks_missing"):
            self.assertIn(k, agg)

    def test_full_view_jsonable_no_nan(self):
        # allow_nan=False fails loudly if any NaN/inf leaked past _jnum.
        json.dumps(self.view, allow_nan=False)


class TestEngineParity(unittest.TestCase):
    """The view's aggregate numbers equal an independent call of the standalone
    engines on the service's assembled table + same pinned today."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ops.build_options_view(cls.frames, today=TODAY, now=NOW)
        cls.opt_tbl = ops._assemble_opt_tbl(cls.frames, today=TODAY)

    def test_aggregates_match_engine(self):
        agg = option_book_aggregates(self.opt_tbl, TODAY)
        v = self.view["aggregates"]
        for k in ("notional_protected", "premium_at_risk", "cost_basis",
                  "unrealized_pnl"):
            self.assertIsNone(_deep_close(v[k], agg[k]), k)
        self.assertEqual(v["n_excluded"], int(agg["n_excluded"]))
        # weighted_dte is NaN->None when there is no live MV.
        if math.isfinite(agg["weighted_dte"]):
            self.assertIsNone(_deep_close(v["weighted_dte"], agg["weighted_dte"]))
        else:
            self.assertIsNone(v["weighted_dte"])

    def test_book_iv_percentile_matches_engine(self):
        mv = (self.opt_tbl.loc[self.opt_tbl["market_value"] > 0]
              .groupby("underlying")["market_value"].sum().to_dict())
        ref = book_iv_percentile(
            _read_atm(self.frames), mv, as_of=TODAY,
            window_days=IV_PCT_WINDOW_DAYS)
        got = self.view["iv_percentile"]["last_percentile"]
        if ref is None or math.isnan(ref.percentile):
            self.assertIsNone(got)
        else:
            self.assertIsNone(_deep_close(got, ref.percentile))

    def test_iv_series_aligned(self):
        # House chart-point shape is {"x": ISO-date, "v": ...} — attachAxes'
        # date ticks and the crosshair read p.x, so a "date" key silently
        # renders an axis-less chart (the TK 2026-07-17 feedback item).
        s = self.view["iv_percentile"]["series"]
        for pt in s:
            self.assertIn("x", pt)
            self.assertIn("v", pt)
            self.assertNotIn("date", pt)


def _read_atm(frames):
    p = Path(frames.data_dir) / "atm_iv_history.csv"
    return (pd.read_csv(p, parse_dates=["date", "fetched_at"])
            if p.exists() else pd.DataFrame())


class TestEmptyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_empty_when_no_option_positions(self):
        # Drop every option row -> build_option_position_table returns empty ->
        # the read half is the empty state, but staleness still renders.
        pos = self.frames.positions
        no_opts = pos[~pos["asset_class"].astype(str).str.contains(
            "option", case=False, na=False)]
        frames2 = dataclasses.replace(self.frames, positions=no_opts)
        view = ops.build_options_view(frames2, today=TODAY, now=NOW)
        self.assertTrue(view["empty"])
        self.assertIsInstance(view["empty_message"], str)
        self.assertIsNone(view["aggregates"])
        self.assertIsNone(view["iv_percentile"])
        self.assertIn("snapshot", view["staleness"])  # chips still present

    def test_empty_when_every_option_is_closed(self):
        # An interim roll can net every contract to qty 0 (all puts sold,
        # calls expired) while the rows stay listed. That is an EMPTY book —
        # not a grid of $0 tiles reading as "existing puts at -100% P&L".
        pos = self.frames.positions.copy()
        is_opt = pos["asset_class"].astype(str).str.contains(
            "option", case=False, na=False)
        pos.loc[is_opt, ["quantity", "market_value"]] = 0.0
        frames2 = dataclasses.replace(self.frames, positions=pos)
        view = ops.build_options_view(frames2, today=TODAY, now=NOW)
        self.assertTrue(view["empty"])
        self.assertIn("No open option positions", view["empty_message"])
        self.assertIn("closed or expired", view["empty_message"])
        self.assertIsNone(view["aggregates"])
        self.assertIsNone(view["iv_percentile"])
        self.assertIn("snapshot", view["staleness"])

    def test_missing_atm_history_no_iv_gauge(self):
        # A data_dir with no atm_iv_history.csv -> the IV caption is None but the
        # tiles still compute (Section renders without the gauge).
        empty_dir = ROOT / "tests" / "fixtures"  # no atm_iv_history.csv here
        frames2 = dataclasses.replace(self.frames, data_dir=str(empty_dir),
                                      positions=self.frames.positions)
        # positions/transactions still from the loaded fixture; only data_dir moved
        view = ops.build_options_view(frames2, today=TODAY, now=NOW)
        self.assertIsNone(view["iv_percentile"]["caption"])
        self.assertEqual(view["iv_percentile"]["series"], [])


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

    def test_options_ok(self):
        r = self.client.get("/api/options")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("aggregates", body)
        self.assertIn("staleness", body)

    def test_unknown_params_ignored_not_500(self):
        r = self.client.get("/api/options", params={"account": "nope"})
        self.assertEqual(r.status_code, 200)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/options")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_options_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = ops.build_options_view(frames, today=TODAY, now=NOW)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        mismatch = _deep_close(view, expected)
        self.assertIsNone(mismatch, mismatch)


class TestParity(unittest.TestCase):
    """Cross-check the terminal view against the Streamlit Options tab's rendered
    metric tiles. Uses the real today on both sides (Streamlit hardcodes today).
    Slow (boots Streamlit) — intentional."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        from terminal.holdings_service import fmt_money
        cls.fmt_money = staticmethod(fmt_money)
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ops.build_options_view(cls.frames)  # real today/now, like Streamlit
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        at.session_state["active_tab"] = "Options Hedging"
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def _metric_values(self):
        return {m.value for m in self.at.metric}

    def test_no_uncaught_exception(self):
        self.assertFalse(self.at.exception,
                         f"Streamlit raised: {[e.value for e in self.at.exception]}")

    def test_money_tiles_present_among_metrics(self):
        # notional / premium-at-risk / unrealized-P&L tiles use fmt_money on both
        # sides (app.py 8357/8370/8387). Assert each appears among the rendered
        # metric values. Skip a tile whose value is None (missing data).
        agg = self.view["aggregates"]
        self.assertIsNotNone(agg, "fixture should populate aggregates")
        mvals = self._metric_values()
        fm = self.fmt_money
        self.assertIn(fm(agg["notional_protected"]), mvals)
        self.assertIn(fm(agg["premium_at_risk"]), mvals)
        # Unrealized P&L is rendered with a leading '+' when >= 0 (app.py 8384-8387).
        pnl = agg["unrealized_pnl"]
        sign = "+" if pnl >= 0 else ""
        self.assertIn(f"{sign}{fm(pnl)}", mvals)

    def test_weighted_dte_tile_present(self):
        dte = self.view["aggregates"]["weighted_dte"]
        if dte is not None:
            self.assertIn(f"{dte:.0f} days", self._metric_values())


from hedge_recommender import build_hedge_basket, EXIT_RULE_MULTIPLIER

# Anchored recommend inputs (pinned like the read-half TODAY).
REC_MODE = "A"
REC_TARGET = 0.10


class TestRecommendContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ops.build_recommend_view(cls.frames, mode=REC_MODE,
                                            target=REC_TARGET, today=TODAY)

    def test_keys(self):
        self.assertEqual(
            set(self.view.keys()),
            {"meta", "composition", "coverage_caption", "warnings",
             "chain_error", "recommendation", "hedge_signals"})

    def test_meta(self):
        m = self.view["meta"]
        self.assertEqual(m["mode"], "A")
        self.assertAlmostEqual(m["target"], 0.10)
        self.assertEqual(m["target_label"], "10%")
        self.assertEqual(m["defaults"], {"mode": "A", "target": 0.10})

    def test_composition_keys(self):
        for k in ("portfolio_value", "equity_mv", "equity_pct", "cash_mv",
                  "cash_pct", "options_mv", "options_pct"):
            self.assertIn(k, self.view["composition"])

    def test_recommendation_present_on_fixture(self):
        # The fixture has spy_holdings.csv + hedge_chain_fixture.csv + a put, so a
        # recommendation builds offline.
        rec = self.view["recommendation"]
        self.assertIsNotNone(rec)
        for k in ("scenarios", "existing_puts", "new_puts", "current_cap_pct",
                  "combined_cap_pct", "total_new_premium", "diagnostics"):
            self.assertIn(k, rec)
        self.assertEqual(len(rec["scenarios"]), 5)  # 5 SCENARIO_DRAWDOWNS

    def test_existing_puts_carry_roll_and_sell_at(self):
        for ep in self.view["recommendation"]["existing_puts"]:
            for k in ("roll_by", "roll_into", "sell_at"):
                self.assertIn(k, ep)

    def test_hedge_signals_shape(self):
        hs_ = self.view["hedge_signals"]
        self.assertIn(hs_["level"], ("green", "amber", "grey"))
        self.assertIsInstance(hs_["rows"], list)

    def test_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)

    def test_headline_and_scenario_notes_present(self):
        rec = self.view["recommendation"]
        self.assertIn(rec["headline"]["level"],
                      ("success", "warn", "note", "info"))
        self.assertIsInstance(rec["headline"]["html"], str)
        self.assertIsInstance(rec["scenario_notes"], list)

    def test_meta_has_as_of_iso(self):
        as_of = self.view["meta"]["as_of"]
        self.assertEqual(as_of, TODAY.isoformat())   # deterministic under pinned today
        # ISO YYYY-MM-DD parses
        import datetime as _dt
        _dt.date.fromisoformat(as_of)


class TestRecommendEngineParity(unittest.TestCase):
    """The view's recommendation equals an independent build_hedge_basket call on
    the same injected inputs (assembled the way app.py assembles them)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ops.build_recommend_view(cls.frames, mode=REC_MODE,
                                            target=REC_TARGET, today=TODAY)

    def test_scenarios_match_engine(self):
        inp = ops._recommend_inputs(self.frames, mode=REC_MODE,
                                    target=REC_TARGET, today=TODAY)
        rec = build_hedge_basket(
            mode=REC_MODE, target=REC_TARGET, holdings=inp["holdings"],
            existing_options=inp["existing_options"],
            per_symbol_mcr=inp["per_symbol_mcr"], spy_holdings=inp["spy_holdings"],
            chain_premiums=inp["chain_premiums"], crash_betas=inp["crash_betas"],
            today=TODAY, spot_prices=inp["spot_prices"])
        got = [s["combined_pnl"] for s in self.view["recommendation"]["scenarios"]]
        exp = [ (None if math.isnan(s.combined_pnl) else s.combined_pnl)
                for s in rec.scenarios ]
        self.assertIsNone(_deep_close(got, exp))
        self.assertIsNone(_deep_close(self.view["recommendation"]["total_new_premium"],
                                      rec.total_new_premium))


class TestHeadline(unittest.TestCase):
    """Direct unit test of the 5-branch headline (app.py 8861-8943 port). Pure
    function over primitives, so every branch is deterministic without a full rec."""

    def test_caps_unavailable_warn(self):
        h = ops._build_headline(
            mode="A", current_cap_pct=float("nan"), target_cap_pct=0.10,
            combined_cap_pct=float("nan"), current_worst_pnl_pct=-0.10,
            combined_worst_pnl_pct=-0.10, has_new_puts=True, target=0.10)
        self.assertEqual(h["level"], "warn")
        self.assertIn("Scenarios unavailable", h["html"])

    def test_mode_a_already_covered_success(self):
        h = ops._build_headline(
            mode="A", current_cap_pct=0.08, target_cap_pct=0.10,
            combined_cap_pct=0.08, current_worst_pnl_pct=-0.08,
            combined_worst_pnl_pct=-0.08, has_new_puts=False, target=0.10)
        self.assertEqual(h["level"], "success")
        self.assertIn("Already covered", h["html"])
        self.assertIn("a loss of 8.0%", h["html"])

    def test_mode_a_general_over_covers_note(self):
        h = ops._build_headline(
            mode="A", current_cap_pct=0.18, target_cap_pct=0.10,
            combined_cap_pct=0.06, current_worst_pnl_pct=-0.18,
            combined_worst_pnl_pct=0.02, has_new_puts=True, target=0.10)
        self.assertEqual(h["level"], "note")
        self.assertIn("Your target cap:", h["html"])
        self.assertIn("past", h["html"])                 # over-covers "why"
        self.assertIn("a loss of 18.0%", h["html"])
        self.assertIn("a gain of +2.0%", h["html"])      # convex-positive worst

    def test_mode_a_general_no_over_cover_note(self):
        h = ops._build_headline(
            mode="A", current_cap_pct=0.12, target_cap_pct=0.10,
            combined_cap_pct=0.10, current_worst_pnl_pct=-0.12,
            combined_worst_pnl_pct=-0.10, has_new_puts=True, target=0.10)
        self.assertEqual(h["level"], "note")
        self.assertIn("Your target cap:", h["html"])
        self.assertNotIn("past", h["html"])

    def test_mode_b_ladder_note(self):
        h = ops._build_headline(
            mode="B", current_cap_pct=0.30, target_cap_pct=0.28,
            combined_cap_pct=0.28, current_worst_pnl_pct=-0.30,
            combined_worst_pnl_pct=-0.28, has_new_puts=True, target=0.010)
        self.assertEqual(h["level"], "note")
        self.assertIn("tail-hedge ladder", h["html"])
        self.assertIn("1.00%/yr", h["html"])
        self.assertIn("not</strong> a loss cap", h["html"])

    def test_mode_b_budget_too_small_info(self):
        h = ops._build_headline(
            mode="B", current_cap_pct=0.30, target_cap_pct=0.30,
            combined_cap_pct=0.30, current_worst_pnl_pct=-0.30,
            combined_worst_pnl_pct=-0.30, has_new_puts=False, target=0.010)
        self.assertEqual(h["level"], "info")
        self.assertIn("Budget too small", h["html"])
        self.assertIn("1.00%/yr", h["html"])


class TestHeadlineDecision(unittest.TestCase):
    """headline_decision is the UI-neutral branch DECISION extracted from
    _build_headline (Phase D) — both app.py and the terminal will call it and
    render their own markup. Covers all 5 cases + both thresholds."""

    def d(self, **kw):
        base = dict(mode="A", current_cap_pct=0.10, target_cap_pct=0.10,
                    combined_cap_pct=0.10, has_new_puts=False)
        base.update(kw)
        return ops.headline_decision(**base)

    def test_unavailable_nan(self):
        self.assertEqual(self.d(current_cap_pct=float("nan"))["case"], "unavailable")

    def test_already_covered_threshold(self):
        # current == target + 0.001 -> still already_covered
        self.assertEqual(self.d(current_cap_pct=0.101, target_cap_pct=0.10)["case"],
                         "already_covered")
        # current == target + 0.002 -> mode_a
        self.assertEqual(self.d(current_cap_pct=0.102, target_cap_pct=0.10)["case"],
                         "mode_a")

    def test_mode_a_over_covers_threshold(self):
        r = self.d(current_cap_pct=0.20, has_new_puts=True,
                   combined_cap_pct=0.09, target_cap_pct=0.10)  # 0.09 < 0.10-0.005
        self.assertEqual(r["case"], "mode_a")
        self.assertTrue(r["over_covers"])

    def test_mode_b_ladder_and_budget(self):
        self.assertEqual(self.d(mode="B", has_new_puts=True)["case"], "mode_b_ladder")
        self.assertEqual(self.d(mode="B", has_new_puts=False)["case"], "budget_too_small")


class TestScenarioNotes(unittest.TestCase):
    """`_scenario_notes` builds the data-driven captions under the table:
    excluded-no-history (any mode) then the Mode-A cap-precision note."""

    def _rec(self, excluded, mode, cap_note):
        return types.SimpleNamespace(
            diagnostics={"scenario_excluded_no_history": excluded},
            mode=mode, cap_precision_note=cap_note)

    def test_excluded_and_cap_note(self):
        notes = ops._scenario_notes(
            self._rec([("AAA", 0.30)], "A", "cap note text"))
        self.assertEqual(len(notes), 2)
        self.assertIn("AAA (30% of equity)", notes[0])
        self.assertIn("No crash-window history", notes[0])
        self.assertEqual(notes[1], "cap note text")

    def test_cap_note_only_when_present(self):
        notes = ops._scenario_notes(self._rec([], "A", "cap note text"))
        self.assertEqual(notes, ["cap note text"])

    def test_mode_b_drops_cap_note(self):
        notes = ops._scenario_notes(self._rec([], "B", "cap note text"))
        self.assertEqual(notes, [])

    def test_no_cap_note(self):
        notes = ops._scenario_notes(self._rec([], "A", None))
        self.assertEqual(notes, [])


class TestChainSeams(unittest.TestCase):
    """build_chain_targets + parse_chain_premiums: the two seams shared between
    _recommend_inputs and (later) app.py's recommendation block."""

    def test_mode_a_strike_depth_and_targets(self):
        daily = pd.DataFrame({"SPY": [390.0, 395.0, 400.0],
                              "AAA": [98.0, 99.0, 100.0]})
        out = ops.build_chain_targets(daily, ["AAA", "SPY"], ["AAA"],
                                      mode="A", target=0.10, today=TODAY)
        self.assertEqual(out["spot_prices"]["SPY"], 400.0)
        self.assertEqual(out["spot_prices"]["AAA"], 100.0)
        self.assertEqual(out["strike_depth"], 0.10)
        by_ticker = {t: strike for t, strike, _, _ in out["chain_targets"]}
        self.assertAlmostEqual(by_ticker["SPY"], 360.0)   # 400 * (1 - 0.10)
        self.assertAlmostEqual(by_ticker["AAA"], 90.0)     # 100 * (1 - 0.10)
        self.assertEqual(len(out["chain_targets"]), 2)
        self.assertTrue(all(kind == "put" for _, _, _, kind in out["chain_targets"]))

    def test_mode_b_strike_depth_fixed_at_20pct(self):
        daily = pd.DataFrame({"SPY": [400.0], "AAA": [100.0]})
        out = ops.build_chain_targets(daily, ["AAA", "SPY"], ["AAA"],
                                      mode="B", target=0.005, today=TODAY)
        self.assertEqual(out["strike_depth"], 0.20)
        by_ticker = {t: strike for t, strike, _, _ in out["chain_targets"]}
        self.assertAlmostEqual(by_ticker["SPY"], 320.0)    # 400 * (1 - 0.20)


class TestParseChainPremiums(unittest.TestCase):

    def test_skips_missing_premium_keeps_valid_row(self):
        chain_df = pd.DataFrame({
            "request_underlying": ["SPY", "AAA"],
            "request_strike": [360.0, 90.0],
            "polygon_price": [5.0, None],
            "expiration_date": ["2026-09-18", "2026-09-18"]})
        out = ops.parse_chain_premiums(chain_df)
        self.assertEqual(list(out.keys()), ["SPY"])
        self.assertEqual(out["SPY"]["premium"], 5.0)
        self.assertEqual(out["SPY"]["strike"], 360.0)
        self.assertEqual(out["SPY"]["expiry"], date(2026, 9, 18))


class TestRecommendServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_default_ok(self):
        r = self.client.get("/api/options/recommend",
                            params={"mode": "A", "target": 0.10})
        self.assertEqual(r.status_code, 200)
        self.assertIn("recommendation", r.json())

    def test_mode_b_ok(self):
        r = self.client.get("/api/options/recommend",
                            params={"mode": "B", "target": 0.010})
        self.assertEqual(r.status_code, 200)

    def test_bad_mode_422(self):
        r = self.client.get("/api/options/recommend",
                            params={"mode": "Z", "target": 0.10})
        self.assertEqual(r.status_code, 422)

    def test_target_not_in_mode_set_422(self):
        # 0.010 is a Mode-B budget, not a Mode-A cap.
        r = self.client.get("/api/options/recommend",
                            params={"mode": "A", "target": 0.010})
        self.assertEqual(r.status_code, 422)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/options/recommend",
                                params={"mode": "A", "target": 0.10})
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestRecommendGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_options_recommend_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = ops.build_recommend_view(frames, mode=REC_MODE,
                                        target=REC_TARGET, today=TODAY)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        mismatch = _deep_close(view, expected)
        self.assertIsNone(mismatch, mismatch)


class TestRecommendParity(unittest.TestCase):
    """Cross-check the recommend view against the Streamlit Options tab: the
    composition metric tiles + the hedge-signals headline. Real today on both
    sides (Streamlit hardcodes today); Mode A / 10% are the Streamlit defaults."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ops.build_recommend_view(cls.frames, mode="A", target=0.10)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        at.session_state["active_tab"] = "Options Hedging"
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_no_exception(self):
        self.assertFalse(self.at.exception,
                         f"Streamlit raised: {[e.value for e in self.at.exception]}")

    def test_composition_tiles_present(self):
        mvals = {m.value for m in self.at.metric}
        c = self.view["composition"]
        # app.py 8553-8558: f"${v:,.0f}" for the value.
        self.assertIn(f"${c['portfolio_value']:,.0f}", mvals)
        self.assertIn(f"${c['equity_mv']:,.0f}", mvals)

    def test_hedge_headline_text_present(self):
        # The hedge-signals headline text appears somewhere in the rendered
        # markdown/success/info/caption. Compare the core text (drop any emoji
        # prefix Streamlit adds). Skip if grey/no-signals empty headline.
        head = self.view["hedge_signals"]["headline"]
        if not head:
            self.skipTest("no hedge headline on this fixture")
        blobs = " ".join(
            (getattr(el, "value", "") or "") for el in
            (list(self.at.markdown) + list(self.at.success)
             + list(self.at.info) + list(self.at.caption)))
        self.assertIn(head, blobs)

    def test_scenario_headline_worst_phrases_match_streamlit(self):
        # The service headline and the Streamlit render derive from the SAME rec,
        # so the worst-case phrases (via _fmt_worst) must appear on both sides.
        rec = self.view["recommendation"]
        cur = ops._fmt_worst(rec["current_worst_pnl_pct"])
        comb = ops._fmt_worst(rec["combined_worst_pnl_pct"])
        html = rec["headline"]["html"]
        self.assertIn(cur, html)                         # service side
        self.assertIn(comb, html)
        blobs = " ".join(
            (getattr(e, "value", "") or "") for e in
            (list(self.at.markdown) + list(self.at.success)
             + list(self.at.info) + list(self.at.warning) + list(self.at.caption)))
        self.assertIn(cur, blobs)                        # Streamlit side
        self.assertIn(comb, blobs)

    def test_scenario_notes_appear_in_streamlit_captions(self):
        notes = self.view["recommendation"]["scenario_notes"]
        if not notes:
            self.skipTest("no scenario notes on this fixture")
        caps = " ".join((getattr(c, "value", "") or "")
                        for c in self.at.caption)
        for n in notes:
            self.assertIn(n, caps)

    def test_scenario_numbers_match_streamlit_headline_branch(self):
        # On the fixture (Mode A, over-covering basket) the branch is the neutral
        # 3-line readout -> Streamlit renders it via st.markdown, level "note".
        self.assertEqual(self.view["recommendation"]["headline"]["level"], "note")
        md = " ".join((getattr(m, "value", "") or "") for m in self.at.markdown)
        self.assertIn("Your target cap:", md)

    def test_coverage_tiles_match_streamlit(self):
        rec = self.view["recommendation"]
        mvals = {m.value for m in self.at.metric}
        # app.py 9022-9027: f"${...:,.0f}" and f"{...*100:.2f}%"
        self.assertIn(f"${rec['total_new_premium']:,.0f}", mvals)
        self.assertIn(f"{rec['total_new_drag_pct'] * 100:.2f}%", mvals)
        self.assertIn(f"{rec['total_combined_drag_pct'] * 100:.2f}%", mvals)


if __name__ == "__main__":
    unittest.main()
