# tests/test_terminal_risksim.py
import json, math, os, sys, unittest
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "parsers"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import risksim_service as rss
from terminal import factor_service as fs
from terminal import risk_service as rsvc
from whatif_engine import WhatIfScenario, compute_before_after
from min_variance import suggest_min_variance_grid
from risk_parity import suggest_risk_parity_grid
from opt_curve import trace_cap_curve
from frontier import trace_frontier, capm_expected_returns
import whatif_data as wd


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (the Factor/#204 golden helper:
    covariance/beta/correlation outputs aren't bit-reproducible across BLAS builds)."""
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
        cls.view = rss.build_risksim_view(cls.frames)

    def test_keys(self):
        self.assertEqual(set(self.view.keys()),
                         {"meta", "caption_html", "state", "grid", "optimizer"})

    def test_optimizer_seed_block(self):
        opt = self.view["optimizer"]
        self.assertIsNotNone(opt)
        self.assertIn("Min-variance", opt["caption_html"])
        self.assertGreaterEqual(opt["cap_default_pct"], 0.0)
        self.assertLessEqual(opt["cap_default_pct"], 100.0)
        for b in opt["buckets"]:
            self.assertEqual(set(b), {"key", "label", "floor_default_pct"})
            self.assertNotEqual(b["key"], "other")
            self.assertEqual(b["label"], b["key"].replace("_", " ").title())

    def test_available_and_seed_rows_match_weights(self):
        self.assertTrue(self.view["state"]["available"])
        weights = rss._bundle_for(self.frames, "all", "all")["weights"]
        rows = self.view["grid"]["rows"]
        self.assertEqual([r["ticker"] for r in rows], list(weights.index))
        for r, (t, w) in zip(rows, weights.items()):
            self.assertAlmostEqual(r["now_pct"], float(w) * 100.0, places=6)

    def test_jsonable(self):
        json.dumps(self.view, allow_nan=False)


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.weights = rss._bundle_for(cls.frames, "all", "all")["weights"]
        # A fixed deterministic reweight: shift 5pp from the largest name to the
        # smallest, renormalized to 100%. Weight-only, so always simulatable.
        w = (cls.weights * 100.0).sort_values(ascending=False)
        cls.new_pct = {str(t): float(v) for t, v in w.items()}
        big, small = str(w.index[0]), str(w.index[-1])
        cls.new_pct[big] -= 5.0
        cls.new_pct[small] += 5.0

    def test_headline_matches_direct_engine(self):
        view = rss.run_simulation(self.frames, self.new_pct)
        self.assertIsNone(view["error"])
        cur = self.weights
        new = pd.Series({k: v / 100.0 for k, v in self.new_pct.items()})
        new = new[new > 1e-9]
        bundle = rss._bundle_for(self.frames, "all", "all")
        res = compute_before_after(
            WhatIfScenario(candidate_ticker=None, current_weights=cur,
                           new_weights=new),
            self.frames.daily_prices, pd.Series(dtype=float),
            bench_tr=bundle["bench_tr"], rf_series=rss._load_rf(self.frames.data_dir),
            history_start=None)
        self.assertIsNone(res["error"])
        # The vol tile's "after" string equals the engine value formatted.
        vol_tile = view["headline"]["risk"][0]
        self.assertEqual(vol_tile["value"], rss._fmt_pct(res["headline"]["vol"]["after"]))
        self.assertEqual(vol_tile["label"], "Portfolio vol (ann.)")

    def test_weight_bars_and_coverage(self):
        view = rss.run_simulation(self.frames, self.new_pct)
        wb = view["weight_bars"]
        self.assertEqual([p["x"] for p in wb["port"]], [p["x"] for p in wb["bench"]])
        self.assertIn("overlap", view["coverage_html"].lower())

    def test_over_allocated_returns_error_not_crash(self):
        bad = dict(self.new_pct)
        bad[next(iter(bad))] += 40.0   # Sigma != 100
        view = rss.run_simulation(self.frames, bad)
        self.assertIsNotNone(view["error"])
        self.assertIsNone(view["headline"])

    def test_nothing_to_simulate_returns_error(self):
        same = {str(t): float(v) * 100.0 for t, v in self.weights.items()}
        view = rss.run_simulation(self.frames, same)
        self.assertIsNotNone(view["error"])


class TestDetail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        w = (rss._bundle_for(cls.frames, "all", "all")["weights"] * 100.0
             ).sort_values(ascending=False)
        np_ = {str(t): float(v) for t, v in w.items()}
        np_[str(w.index[0])] -= 5.0
        np_[str(w.index[-1])] += 5.0
        cls.view = rss.run_simulation(cls.frames, np_)

    def test_detail_shape(self):
        d = self.view["detail"]
        self.assertEqual(set(d), {"vol_table", "diversification", "tail", "stress"})
        self.assertEqual([r["metric"] for r in d["vol_table"]["rows"]],
                         ["Portfolio vol (ann.)", "Sharpe", "Sortino"])
        self.assertEqual([r["metric"] for r in d["tail"]["tail_table"]["rows"]],
                         ["Max drawdown", "VaR 95% (daily)", "CVaR 95% (daily)"])
        self.assertEqual([r["metric"] for r in d["stress"]["stress_table"]["rows"]],
                         ["Conditional avg corr", "Down-β vs SPY", "Stressed DR"])

    def test_corr_heatmap_cells(self):
        cb = self.view["detail"]["diversification"]["corr_before"]
        if cb is not None:            # >= 2 priced symbols overlap
            self.assertEqual(cb["cells"][0][0]["text_html"], "1.00")  # diagonal
            self.assertEqual(cb["legend"]["title"], "ρ")

    def test_full_result_jsonable(self):
        json.dumps(self.view, allow_nan=False)   # allow_nan=False-clean end to end


class TestOptimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.bundle = rss._bundle_for(cls.frames, "all", "all")
        seed = rss.build_risksim_view(cls.frames)["optimizer"]
        cls.cap = seed["cap_default_pct"]
        cls.floors = {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}

    def _direct_floors(self):
        class_of = rss._mv_class_of(self.bundle)
        buckets = {b for b in class_of.values() if b != "other"}
        return {b: self.floors.get(b, 0.0) / 100.0 for b in buckets}

    def test_min_variance_matches_engine(self):
        out = rss.run_optimize(self.frames, optimizer="min_variance",
                               cap_pct=self.cap, floors=self.floors)
        direct = suggest_min_variance_grid(
            self.frames.daily_prices, self.bundle["weights"],
            rss._mv_class_of(self.bundle),
            name_cap=self.cap / 100.0, class_floors=self._direct_floors())
        self.assertEqual(out["kind"], direct["kind"])
        self.assertEqual(out["message"], direct["message"])
        exp = ({str(k): float(v) for k, v in direct["new_pct"].items()}
               if direct["new_pct"] is not None else None)
        self.assertIsNone(_deep_close(out["new_pct"], exp))

    def test_risk_parity_matches_engine(self):
        out = rss.run_optimize(self.frames, optimizer="risk_parity",
                               cap_pct=self.cap, floors={})
        direct = suggest_risk_parity_grid(
            self.frames.daily_prices, self.bundle["weights"],
            name_cap=self.cap / 100.0)
        self.assertEqual(out["kind"], direct["kind"])
        self.assertEqual(out["message"], direct["message"])
        exp = ({str(k): float(v) for k, v in direct["new_pct"].items()}
               if direct["new_pct"] is not None else None)
        self.assertIsNone(_deep_close(out["new_pct"], exp))

    def test_infeasible_cap_is_error_not_raise(self):
        out = rss.run_optimize(self.frames, optimizer="min_variance",
                               cap_pct=1.0, floors={})
        self.assertEqual(out["kind"], "error")
        self.assertIsNone(out["new_pct"])

    def test_result_jsonable(self):
        out = rss.run_optimize(self.frames, optimizer="min_variance",
                               cap_pct=self.cap, floors=self.floors)
        json.dumps(out, allow_nan=False)

    def _direct_caps(self, caps_pct):
        class_of = rss._mv_class_of(self.bundle)
        buckets = {b for b in class_of.values() if b != "other"}
        return {b: caps_pct.get(b, 100.0) / 100.0 for b in buckets}

    def test_min_variance_caps_match_engine(self):
        caps_pct = {"fixed_income": 5.0}
        out = rss.run_optimize(self.frames, optimizer="min_variance",
                               cap_pct=self.cap, floors=self.floors,
                               caps=caps_pct)
        direct = suggest_min_variance_grid(
            self.frames.daily_prices, self.bundle["weights"],
            rss._mv_class_of(self.bundle),
            name_cap=self.cap / 100.0, class_floors=self._direct_floors(),
            class_caps=self._direct_caps(caps_pct))
        self.assertEqual(out["kind"], direct["kind"])
        self.assertEqual(out["message"], direct["message"])
        exp = ({str(k): float(v) for k, v in direct["new_pct"].items()}
               if direct["new_pct"] is not None else None)
        self.assertIsNone(_deep_close(out["new_pct"], exp))

    def test_caps_absent_identical_to_empty(self):
        a = rss.run_optimize(self.frames, optimizer="min_variance",
                             cap_pct=self.cap, floors=self.floors)
        b = rss.run_optimize(self.frames, optimizer="min_variance",
                             cap_pct=self.cap, floors=self.floors, caps={})
        self.assertEqual(a, b)

    def test_risk_parity_caps_match_engine(self):
        caps_pct = {"fixed_income": 5.0}
        out = rss.run_optimize(self.frames, optimizer="risk_parity",
                               cap_pct=self.cap, floors={}, caps=caps_pct)
        direct = suggest_risk_parity_grid(
            self.frames.daily_prices, self.bundle["weights"],
            name_cap=self.cap / 100.0,
            class_of=rss._mv_class_of(self.bundle),
            class_caps=self._direct_caps(caps_pct))
        self.assertEqual(out["kind"], direct["kind"])
        self.assertEqual(out["message"], direct["message"])
        exp = ({str(k): float(v) for k, v in direct["new_pct"].items()}
               if direct["new_pct"] is not None else None)
        self.assertIsNone(_deep_close(out["new_pct"], exp))

    def test_risk_parity_budgets_match_engine(self):
        budgets_pct = {"fixed_income": 20.0}
        out = rss.run_optimize(self.frames, optimizer="risk_parity",
                               cap_pct=self.cap, floors={},
                               budgets=budgets_pct)
        direct = suggest_risk_parity_grid(
            self.frames.daily_prices, self.bundle["weights"],
            name_cap=self.cap / 100.0,
            class_of=rss._mv_class_of(self.bundle),
            class_caps={b: 1.0 for b in rss._mv_class_of(
                self.bundle).values() if b != "other"},
            class_risk_budgets={"fixed_income": 0.2})
        self.assertEqual(out["kind"], direct["kind"])
        self.assertEqual(out["message"], direct["message"])
        exp = ({str(k): float(v) for k, v in direct["new_pct"].items()}
               if direct["new_pct"] is not None else None)
        self.assertIsNone(_deep_close(out["new_pct"], exp))

    def test_budgets_zero_entry_equals_unset(self):
        # Service-level zero-filter boundary: an explicit 0 budget is UNSET
        # (plain ERC), identical to the absent-key path. The route rejects a
        # 0 with 422; this pins the seam other callers hit directly.
        b0 = sorted({b for b in rss._mv_class_of(self.bundle).values()
                     if b != "other"})[0]
        base = rss.run_optimize(self.frames, optimizer="risk_parity",
                                cap_pct=self.cap, floors={}, budgets={})
        zeroed = rss.run_optimize(self.frames, optimizer="risk_parity",
                                  cap_pct=self.cap, floors={},
                                  budgets={b0: 0.0})
        self.assertEqual(zeroed["kind"], base["kind"])
        self.assertEqual(zeroed["message"], base["message"])
        self.assertIsNone(_deep_close(zeroed["new_pct"], base["new_pct"]))

    def test_min_variance_includes_candidate(self):
        view = rss.run_optimize(
            self.frames, optimizer="min_variance", cap_pct=40.0, floors={},
            candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        self.assertEqual(view["kind"], "success")
        self.assertIn("NEWC", view["new_pct"])           # in the opportunity set
        self.assertEqual(view["warnings"], [])

    def test_risk_parity_sizes_candidate_positive(self):
        view = rss.run_optimize(
            self.frames, optimizer="risk_parity", cap_pct=40.0, floors={},
            candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        self.assertEqual(view["kind"], "success")
        self.assertIn("NEWC", view["new_pct"])
        self.assertGreater(view["new_pct"]["NEWC"], 0.0)  # ERC gives every name weight

    def test_optimize_no_candidates_has_no_warnings_key(self):
        view = rss.run_optimize(self.frames, optimizer="min_variance",
                                cap_pct=40.0, floors={})
        self.assertNotIn("warnings", view)                # shape unchanged w/o candidates

    def test_candidate_only_bucket_floor_is_honored(self):
        # NEWC as gold in an equity/FI book: a gold floor must push weight onto it.
        view = rss.run_optimize(
            self.frames, optimizer="min_variance", cap_pct=60.0,
            floors={"gold": 10.0},
            candidates=[{"ticker": "NEWC", "asset_class": "gold", "proxy": ""}])
        self.assertEqual(view["kind"], "success")
        self.assertGreaterEqual(view["new_pct"].get("NEWC", 0.0), 9.99)


class TestTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.bundle = rss._bundle_for(cls.frames, "all", "all")
        seed = rss.build_risksim_view(cls.frames)["optimizer"]
        cls.cap = seed["cap_default_pct"]
        cls.floors = {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}

    def test_shape_and_jsonable(self):
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        self.assertEqual(set(out), {"error", "series", "current", "skipped_n",
                                    "skipped_reasons", "empty_message",
                                    "caption_html"})
        self.assertIsNone(out["error"])
        for s in out["series"]:
            self.assertEqual(set(s), {"key", "label", "points"})
            self.assertIn(s["key"], ("min_variance", "risk_parity"))
            caps = [p["cap"] for p in s["points"]]
            self.assertEqual(caps, sorted(caps))   # cap-ascending
            for p in s["points"]:
                self.assertEqual(set(p),
                                 {"cap", "vol", "effective_n", "max_weight",
                                  "converged", "weights_pct"})
        json.dumps(out, allow_nan=False)           # allow_nan=False-clean

    def test_points_carry_weights_pct(self):
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        for s in out["series"]:
            for p in s["points"]:
                self.assertIn("weights_pct", p)
                w = p["weights_pct"]
                self.assertIsInstance(w, dict)
                self.assertAlmostEqual(sum(w.values()), 100.0, places=4)
        json.dumps(out, allow_nan=False)

    def test_skipped_reasons_grouped_and_disclosed(self):
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        self.assertIn("skipped_reasons", out)
        self.assertEqual(sum(g["n"] for g in out["skipped_reasons"]),
                         out["skipped_n"])
        for g in out["skipped_reasons"]:
            self.assertEqual(set(g), {"reason", "n"})
            self.assertGreater(g["n"], 0)
        if out["skipped_n"] > 0:
            # Partial sweeps disclose the reason in the caption, not just a count.
            self.assertIn("skipped —", out["caption_html"])
            self.assertIn(out["skipped_reasons"][0]["reason"].rstrip("."),
                          out["caption_html"])

    def test_series_match_direct_engine(self):
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        class_of = rss._mv_class_of(self.bundle)
        buckets = sorted({b for b in class_of.values() if b != "other"})
        class_floors = {b: self.floors.get(b, 0.0) / 100.0 for b in buckets}
        ladder = sorted({*rss._TRACE_LADDER_PCT, float(self.cap)})
        direct = trace_cap_curve(self.frames.daily_prices, self.bundle["weights"],
                                 class_of, caps=[c / 100.0 for c in ladder],
                                 class_floors=class_floors)
        got = {(s["key"], round(p["cap"], 10)): p
               for s in out["series"] for p in s["points"]}
        exp = {(r["optimizer"], round(float(r["cap"]), 10)): r
               for _, r in direct["points"].iterrows()}
        self.assertEqual(set(got), set(exp))
        for k in got:
            self.assertIsNone(_deep_close(
                [got[k]["vol"], got[k]["effective_n"], got[k]["max_weight"]],
                [float(exp[k]["vol"]), float(exp[k]["effective_n"]),
                 float(exp[k]["max_weight"])]))

    def test_current_star_present(self):
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        self.assertIsNotNone(out["current"])
        self.assertEqual(set(out["current"]), {"vol", "effective_n", "max_weight"})

    def test_skipped_and_empty_message_invariant(self):
        # The ladder always includes a 100% (non-binding) cap, so on the fixture at
        # least the unconstrained point survives → series non-empty, empty_message
        # None. Assert the invariant either way; a tight cap raises skipped_n.
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        if out["series"]:
            self.assertIsNone(out["empty_message"])
        else:
            self.assertIsNotNone(out["empty_message"])
        self.assertGreaterEqual(out["skipped_n"], 0)
        if out["skipped_n"] > 0:
            self.assertIn("skipped", out["caption_html"])

    def test_empty_sweep_joins_all_reasons_into_empty_message(self):
        # 1% class caps make every rung infeasible -> series fully empty;
        # empty_message must be the "; "-joined reason (n) form, matching
        # skipped_reasons exactly (the branch no fixture-default run hits).
        buckets = sorted({b for b in rss._mv_class_of(self.bundle).values()
                          if b != "other"})
        out = rss.run_trace(self.frames, cap_pct=self.cap,
                            floors=self.floors,
                            caps={b: 1.0 for b in buckets})
        self.assertEqual(out["series"], [])
        self.assertGreater(out["skipped_n"], 0)
        exp = "; ".join(f"{g['reason'].rstrip('.')} ({g['n']})"
                        for g in out["skipped_reasons"])
        self.assertEqual(out["empty_message"], exp)
        self.assertIn("skipped —", out["caption_html"])

    def test_trace_caps_match_direct_engine(self):
        caps_pct = {"fixed_income": 5.0}
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors,
                            caps=caps_pct)
        class_of = rss._mv_class_of(self.bundle)
        buckets = sorted({b for b in class_of.values() if b != "other"})
        class_floors = {b: self.floors.get(b, 0.0) / 100.0 for b in buckets}
        class_caps = {b: caps_pct.get(b, 100.0) / 100.0 for b in buckets}
        ladder = sorted({*rss._TRACE_LADDER_PCT, float(self.cap)})
        direct = trace_cap_curve(self.frames.daily_prices,
                                 self.bundle["weights"], class_of,
                                 caps=[c / 100.0 for c in ladder],
                                 class_floors=class_floors,
                                 class_caps=class_caps)
        got = {(s["key"], round(p["cap"], 10)): p
               for s in out["series"] for p in s["points"]}
        exp = {(r["optimizer"], round(float(r["cap"]), 10)): r
               for _, r in direct["points"].iterrows()}
        self.assertEqual(set(got), set(exp))
        for k in got:
            self.assertIsNone(_deep_close(
                [got[k]["vol"], got[k]["effective_n"]],
                [float(exp[k]["vol"]), float(exp[k]["effective_n"])]))

    def test_caption_names_active_class_caps_only_when_active(self):
        base = rss.run_trace(self.frames, cap_pct=self.cap,
                             floors=self.floors)
        self.assertNotIn("Class caps applied", base["caption_html"])
        # Derive a bucket that actually exists on the fixture book.
        b0 = sorted({b for b in rss._mv_class_of(self.bundle).values()
                     if b != "other"})[0]
        capped = rss.run_trace(self.frames, cap_pct=self.cap,
                               floors=self.floors, caps={b0: 40.0})
        self.assertIn(f"Class caps applied: {b0.replace('_', ' ')} 40%",
                      capped["caption_html"])
        absent = rss.run_trace(self.frames, cap_pct=self.cap,
                               floors=self.floors, caps={})
        self.assertEqual(base["caption_html"], absent["caption_html"])

    def test_caption_names_budgets_only_when_entered(self):
        base = rss.run_trace(self.frames, cap_pct=self.cap,
                             floors=self.floors)
        self.assertNotIn("Risk budgets applied", base["caption_html"])
        b0 = sorted({b for b in rss._mv_class_of(self.bundle).values()
                     if b != "other"})[0]
        budgeted = rss.run_trace(self.frames, cap_pct=self.cap,
                                 floors=self.floors, budgets={b0: 20.0})
        self.assertIn(
            f"Risk budgets applied: {b0.replace('_', ' ')} 20%",
            budgeted["caption_html"])
        absent = rss.run_trace(self.frames, cap_pct=self.cap,
                               floors=self.floors, budgets={})
        self.assertEqual(base["caption_html"], absent["caption_html"])
        zeroed = rss.run_trace(self.frames, cap_pct=self.cap,
                               floors=self.floors, budgets={b0: 0.0})
        self.assertEqual(base["caption_html"], zeroed["caption_html"])

    def test_trace_includes_candidate_in_weights(self):
        view = rss.run_trace(
            self.frames, cap_pct=40.0, floors={},
            candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        self.assertIsNone(view["error"])
        self.assertEqual(view["warnings"], [])
        self.assertTrue(view["series"])                 # at least one optimizer series
        pt = view["series"][0]["points"][0]
        self.assertIn("NEWC", pt["weights_pct"])


class TestSweepProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(cls.frames)["optimizer"]
        cls.cap = seed["cap_default_pct"]
        cls.floors = {b["key"]: b["floor_default_pct"]
                      for b in seed["buckets"]}

    def tearDown(self):
        rss._sweep_end()          # never leak a held slot into other tests

    def test_idle_shape(self):
        self.assertEqual(rss.sweep_progress(), {"running": False})

    def test_begin_tick_end_lifecycle(self):
        self.assertTrue(rss._sweep_begin("trace", 28))
        self.assertEqual(rss.sweep_progress(),
                         {"running": True, "op": "trace", "done": 0,
                          "total": 28})
        rss._sweep_tick(12, 28)
        self.assertEqual(rss.sweep_progress()["done"], 12)
        rss._sweep_end()
        self.assertEqual(rss.sweep_progress(), {"running": False})

    def test_second_begin_rejected_while_held(self):
        self.assertTrue(rss._sweep_begin("trace", 28))
        self.assertFalse(rss._sweep_begin("frontier", 12))

    def test_run_trace_under_held_slot_errors_and_leaves_slot(self):
        self.assertTrue(rss._sweep_begin("frontier", 12))
        out = rss.run_trace(self.frames, cap_pct=self.cap,
                            floors=self.floors)
        self.assertEqual(out["error"], rss._SWEEP_BUSY_MSG)
        self.assertEqual(out["series"], [])
        # The refused run must NOT release the holder's slot.
        self.assertTrue(rss.sweep_progress()["running"])

    def test_run_trace_ticks_and_clears(self):
        seen = []
        real_begin = rss._sweep_begin
        def spying_begin(op, total):
            ok = real_begin(op, total)
            seen.append(("begin", op, total, ok))
            return ok
        rss._sweep_begin = spying_begin
        try:
            out = rss.run_trace(self.frames, cap_pct=self.cap,
                                floors=self.floors)
        finally:
            rss._sweep_begin = real_begin
        self.assertIsNone(out["error"])
        self.assertEqual(seen[0][1], "trace")
        self.assertGreater(seen[0][2], 0)                 # a real total
        self.assertEqual(rss.sweep_progress(), {"running": False})


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)
        cls.frames = hs.load_frames(FIXTURE)
        w = (rss._bundle_for(cls.frames, "all", "all")["weights"] * 100.0
             ).sort_values(ascending=False)
        cls.new_pct = {str(t): float(v) for t, v in w.items()}
        cls.new_pct[str(w.index[0])] -= 5.0
        cls.new_pct[str(w.index[-1])] += 5.0

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_seed_ok(self):
        r = self.client.get("/api/risksim")
        self.assertEqual(r.status_code, 200)
        self.assertIn("grid", r.json())

    def test_seed_unknown_account_422(self):
        r = self.client.get("/api/risksim", params={"account": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_simulate_ok(self):
        r = self.client.post("/api/risksim/simulate",
                             json={"weights": self.new_pct})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["error"])
        self.assertIsNotNone(r.json()["headline"])

    def test_simulate_multi_account_ok(self):
        ids = [o["id"] for o in
               self.client.get("/api/risksim").json()["meta"]["accounts"]][:2]
        if len(ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        # Weights must be within the FILTERED universe (else the universe gate
        # 422s on out-of-subset tickers) — derive them from the same 2-account
        # bundle the handler resolves, proving the list reaches _bundle_for.
        w = rss._bundle_for(self.frames, ids, "all")["weights"] * 100.0
        weights = {str(t): float(v) for t, v in w.items()}
        r = self.client.post("/api/risksim/simulate",
                             json={"weights": weights, "account": ids})
        self.assertEqual(r.status_code, 200)

    def test_simulate_unknown_ticker_422(self):
        r = self.client.post("/api/risksim/simulate",
                             json={"weights": {**self.new_pct, "ZZZZ": 0.0}})
        self.assertEqual(r.status_code, 422)

    def test_simulate_malformed_body_422(self):
        r = self.client.post("/api/risksim/simulate",
                             json={"weights": {"AAA": "not-a-number"}})
        self.assertEqual(r.status_code, 422)

    def test_simulate_over_allocated_200_with_error(self):
        bad = dict(self.new_pct); bad[str(next(iter(bad)))] += 40.0
        r = self.client.post("/api/risksim/simulate", json={"weights": bad})
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(r.json()["error"])

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/risksim")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)

    def test_frontier_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.post("/api/risksim/frontier",
                                 json=self._frontier_body())
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)

    def test_frontier_response_carries_sig(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("sig", body)
        self.assertIsInstance(body["sig"], str)
        self.assertEqual(len(body["sig"]), 16)

    def test_sig_dim_rejected_on_nonfrontier_section(self):
        r = self.client.get("/api/ai/explain",
                            params={"section": "risk", "sig": "abc123"})
        self.assertEqual(r.status_code, 422)

    def _opt_body(self, **kw):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        body = {"optimizer": "min_variance", "cap_pct": seed["cap_default_pct"],
                "floors": {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}}
        body.update(kw)
        return body

    def test_optimize_minvar_ok(self):
        r = self.client.post("/api/risksim/optimize", json=self._opt_body())
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["kind"], ("success", "error"))
        self.assertIn("message", r.json())

    def test_optimize_riskparity_ok(self):
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(optimizer="risk_parity", floors={}))
        self.assertEqual(r.status_code, 200)

    def test_optimize_infeasible_cap_200_error(self):
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(cap_pct=1.0, floors={}))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kind"], "error")
        self.assertIsNone(r.json()["new_pct"])

    def test_optimize_bad_optimizer_422(self):
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(optimizer="sharpe_max"))
        self.assertEqual(r.status_code, 422)

    def test_optimize_cap_out_of_range_422(self):
        r = self.client.post("/api/risksim/optimize", json=self._opt_body(cap_pct=0.0))
        self.assertEqual(r.status_code, 422)

    def test_optimize_unknown_bucket_422(self):
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(floors={"unobtainium": 10.0}))
        self.assertEqual(r.status_code, 422)

    def test_optimize_floor_over_100_422(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(floors={b0: 150.0}))
        self.assertEqual(r.status_code, 422)

    def test_optimize_caps_ok(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(caps={b0: 40.0}))
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["kind"], ("success", "error"))

    def test_optimize_zero_cap_is_valid_at_route(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(caps={b0: 0.0}))
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["kind"], ("success", "error"))

    def test_optimize_unknown_cap_bucket_422(self):
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(caps={"unobtainium": 10.0}))
        self.assertEqual(r.status_code, 422)

    def test_optimize_cap_bucket_over_100_422(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(caps={b0: 150.0}))
        self.assertEqual(r.status_code, 422)

    def test_optimize_unknown_account_422(self):
        # A one-element LIST (not a scalar): passes the list[str] body model so
        # the request reaches the handler's _validate_filter_ids domain check
        # (a scalar would 422 at the Pydantic type layer, never exercising it).
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(account=["nope"]))
        self.assertEqual(r.status_code, 422)

    def test_optimize_multi_account_ok(self):
        ids = [o["id"] for o in
               self.client.get("/api/risksim").json()["meta"]["accounts"]][:2]
        if len(ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        r = self.client.post("/api/risksim/optimize",
                             json=self._opt_body(account=ids, floors={}))
        self.assertEqual(r.status_code, 200)

    def test_optimize_extra_field_422(self):
        body = self._opt_body(); body["surprise"] = 1
        r = self.client.post("/api/risksim/optimize", json=body)
        self.assertEqual(r.status_code, 422)

    def test_optimize_candidate_offline_ok(self):
        body = self._opt_body(candidates=[
            {"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        r = self.client.post("/api/risksim/optimize", json=body)
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual(out["kind"], "success")
        self.assertIn("NEWC", out["new_pct"])
        self.assertIn("warnings", out)

    def test_optimize_candidate_bad_class_422(self):
        body = self._opt_body(candidates=[
            {"ticker": "NEWC", "asset_class": "crypto", "proxy": ""}])
        self.assertEqual(self.client.post("/api/risksim/optimize", json=body).status_code, 422)

    def test_optimize_candidate_bad_charset_422(self):
        body = self._opt_body(candidates=[
            {"ticker": "AB<C", "asset_class": "equity", "proxy": ""}])
        self.assertEqual(self.client.post("/api/risksim/optimize", json=body).status_code, 422)

    def test_optimize_too_many_candidates_422(self):
        body = self._opt_body(candidates=[
            {"ticker": f"C{i}", "asset_class": "equity", "proxy": ""} for i in range(4)])
        self.assertEqual(self.client.post("/api/risksim/optimize", json=body).status_code, 422)

    def test_optimize_candidate_extra_field_422(self):
        body = self._opt_body(candidates=[
            {"ticker": "NEWC", "asset_class": "equity", "proxy": "", "surprise": 1}])
        self.assertEqual(self.client.post("/api/risksim/optimize", json=body).status_code, 422)

    def test_optimize_candidate_only_bucket_floor_ok(self):
        # A gold floor is valid because the NEWC(gold) candidate adds the bucket.
        body = self._opt_body(cap_pct=60.0, floors={"gold": 10.0}, candidates=[
            {"ticker": "NEWC", "asset_class": "gold", "proxy": ""}])
        r = self.client.post("/api/risksim/optimize", json=body)
        self.assertEqual(r.status_code, 200)      # not 422 on the gold floor key

    def _trace_body(self, **kw):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        body = {"cap_pct": seed["cap_default_pct"],
                "floors": {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}}
        body.update(kw)
        return body

    def test_trace_ok(self):
        r = self.client.post("/api/risksim/trace", json=self._trace_body())
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertIsNone(j["error"])
        self.assertIsInstance(j["series"], list)
        # ladder always includes a 100% (non-binding) cap → ≥1 plotted point
        self.assertTrue(any(s["points"] for s in j["series"]))

    def test_trace_cap_out_of_range_422(self):
        r = self.client.post("/api/risksim/trace", json=self._trace_body(cap_pct=0.0))
        self.assertEqual(r.status_code, 422)

    def test_trace_floor_over_100_422(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(floors={b0: 150.0}))
        self.assertEqual(r.status_code, 422)

    def test_trace_unknown_bucket_422(self):
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(floors={"unobtainium": 10.0}))
        self.assertEqual(r.status_code, 422)

    def test_trace_caps_ok_and_422_twins(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(caps={b0: 40.0}))
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(caps={"unobtainium": 10.0}))
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(caps={b0: -1.0}))
        self.assertEqual(r.status_code, 422)

    def test_trace_unknown_account_422(self):
        # One-element LIST so the domain check runs (see optimize twin above).
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(account=["nope"]))
        self.assertEqual(r.status_code, 422)

    def test_trace_multi_account_ok(self):
        ids = [o["id"] for o in
               self.client.get("/api/risksim").json()["meta"]["accounts"]][:2]
        if len(ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        r = self.client.post("/api/risksim/trace",
                             json=self._trace_body(account=ids, floors={}))
        self.assertEqual(r.status_code, 200)

    def test_trace_extra_field_422(self):
        body = self._trace_body(); body["surprise"] = 1
        r = self.client.post("/api/risksim/trace", json=body)
        self.assertEqual(r.status_code, 422)

    def test_optimize_budgets_ok_and_422s(self):
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/optimize", json=self._opt_body(
            optimizer="risk_parity", floors={}, budgets={b0: 20.0}))
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["kind"], ("success", "error"))
        r = self.client.post("/api/risksim/optimize", json=self._opt_body(
            budgets={"unobtainium": 10.0}))
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/risksim/optimize", json=self._opt_body(
            budgets={b0: 0.0}))
        self.assertEqual(r.status_code, 422)          # 0 = omit, not send
        r = self.client.post("/api/risksim/optimize", json=self._opt_body(
            budgets={b0: 150.0}))
        self.assertEqual(r.status_code, 422)

    def test_trace_budgets_ok_and_422s(self):
        # Full twin of test_optimize_budgets_ok_and_422s on the trace route
        # (the O2 fast-follow: only the unknown-bucket leg existed).
        seed = self.client.get("/api/risksim").json()["optimizer"]
        if not seed["buckets"]:
            self.skipTest("no floor buckets on fixture")
        b0 = seed["buckets"][0]["key"]
        r = self.client.post("/api/risksim/trace", json=self._trace_body(
            budgets={b0: 20.0}))
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["error"])
        r = self.client.post("/api/risksim/trace", json=self._trace_body(
            budgets={"unobtainium": 10.0}))
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/risksim/trace", json=self._trace_body(
            budgets={b0: 0.0}))
        self.assertEqual(r.status_code, 422)          # 0 = omit, not send
        r = self.client.post("/api/risksim/trace", json=self._trace_body(
            budgets={b0: 150.0}))
        self.assertEqual(r.status_code, 422)

    def _frontier_body(self, **kw):
        seed = rss.build_risksim_view(hs.load_frames(FIXTURE))["optimizer"]
        body = {"cap_pct": seed["cap_default_pct"],
                "floors": {b["key"]: b["floor_default_pct"]
                           for b in seed["buckets"]},
                "erp_pct": 4.5}
        body.update(kw)
        return body

    def test_frontier_ok(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body())
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertIsNone(out["error"])
        self.assertEqual(len(out["series"]), 1)
        self.assertIn("CAPM", out["caption_html"])

    def test_frontier_erp_zero_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(erp_pct=0.0))
        self.assertEqual(r.status_code, 422)

    def test_frontier_erp_over_20_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(erp_pct=20.1))
        self.assertEqual(r.status_code, 422)

    def test_frontier_cap_out_of_range_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(cap_pct=0.0))
        self.assertEqual(r.status_code, 422)

    def test_frontier_unknown_floor_bucket_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(floors={"crypto": 10.0}))
        self.assertEqual(r.status_code, 422)

    def test_frontier_unknown_cap_bucket_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(caps={"crypto": 10.0}))
        self.assertEqual(r.status_code, 422)

    def test_frontier_extra_field_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(budgets={"equity": 50.0}))
        self.assertEqual(r.status_code, 422)

    def test_frontier_unknown_account_422(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(account=["nope"]))
        self.assertEqual(r.status_code, 422)

    def test_frontier_infeasible_cap_200_with_empty(self):
        r = self.client.post("/api/risksim/frontier",
                             json=self._frontier_body(cap_pct=1.0))
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertIsNone(out["error"])
        self.assertEqual(out["series"], [])
        self.assertIsNotNone(out["empty_message"])

    def test_frontier_candidate_offline_ok(self):
        body = self._frontier_body(candidates=[
            {"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        r = self.client.post("/api/risksim/frontier", json=body)
        self.assertEqual(r.status_code, 200)
        self.assertIn("warnings", r.json())

    def _cand_body(self, cand="NEWC", w=10.0, asset_class="equity", proxy="", **kw):
        base = rss._bundle_for(self.frames, "all", "all")["weights"] * 100.0
        weights = {str(t): float(v) * (100.0 - w) / 100.0 for t, v in base.items()}
        weights[cand] = w
        body = {"weights": weights,
               "candidates": [{"ticker": cand, "asset_class": asset_class, "proxy": proxy}]}
        body.update(kw)
        return body

    def test_simulate_candidate_ok(self):
        r = self.client.post("/api/risksim/simulate", json=self._cand_body())
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["error"])
        self.assertIsNotNone(r.json()["headline"])

    def test_simulate_candidate_not_flagged_unknown_ticker(self):
        r = self.client.post("/api/risksim/simulate", json=self._cand_body())
        self.assertNotEqual(r.status_code, 422)

    def test_simulate_candidate_already_held_200_error(self):
        held = str(rss._bundle_for(self.frames, "all", "all")["weights"].index[0])
        r = self.client.post(
            "/api/risksim/simulate",
            json={"weights": self.new_pct,
                 "candidates": [{"ticker": held, "asset_class": "equity", "proxy": ""}]})
        self.assertEqual(r.status_code, 200)
        self.assertIn("already held", r.json()["error"])

    def test_simulate_candidate_extra_field_422(self):
        body = self._cand_body(); body["surprise"] = 1
        r = self.client.post("/api/risksim/simulate", json=body)
        self.assertEqual(r.status_code, 422)

    def test_simulate_candidate_bad_charset_422(self):
        r = self.client.post(
            "/api/risksim/simulate",
            json={"weights": self.new_pct,
                 "candidates": [{"ticker": "BAD<script>", "asset_class": "equity",
                                 "proxy": ""}]})
        self.assertEqual(r.status_code, 422)

    def test_simulate_proxy_bad_charset_422(self):
        r = self.client.post(
            "/api/risksim/simulate",
            json={"weights": self.new_pct,
                 "candidates": [{"ticker": "NEWC", "asset_class": "equity",
                                 "proxy": "x;y"}]})
        self.assertEqual(r.status_code, 422)

    def test_simulate_two_candidates_offline_ok(self):
        base = rss._bundle_for(self.frames, "all", "all")["weights"] * 100.0
        weights = {str(t): float(v) * 90.0 / 100.0 for t, v in base.items()}
        weights["NEWC"] = 5.0
        weights["PXY"] = 5.0
        r = self.client.post(
            "/api/risksim/simulate",
            json={"weights": weights,
                 "candidates": [{"ticker": "NEWC", "asset_class": "equity", "proxy": ""},
                                {"ticker": "PXY", "asset_class": "equity", "proxy": ""}]})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["error"])

    def test_simulate_too_many_candidates_422(self):
        r = self.client.post(
            "/api/risksim/simulate",
            json={"weights": self.new_pct,
                 "candidates": [{"ticker": t, "asset_class": "equity", "proxy": ""}
                                for t in ("A1", "A2", "A3", "A4")]})
        self.assertEqual(r.status_code, 422)

    def test_progress_idle_shape(self):
        r = self.client.get("/api/risksim/progress")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"running": False})

    def test_progress_reflects_a_held_slot(self):
        self.assertTrue(rss._sweep_begin("trace", 28))
        try:
            rss._sweep_tick(12, 28)
            r = self.client.get("/api/risksim/progress")
            self.assertEqual(r.json(), {"running": True, "op": "trace",
                                        "done": 12, "total": 28})
        finally:
            rss._sweep_end()

    def test_progress_survives_missing_data_dir(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/risksim/progress")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {"running": False})
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_risksim_golden.json")

    def _payload(self):
        frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(frames)
        w = (rss._bundle_for(frames, "all", "all")["weights"] * 100.0
             ).sort_values(ascending=False)
        np_ = {str(t): float(v) for t, v in w.items()}
        np_[str(w.index[0])] -= 5.0
        np_[str(w.index[-1])] += 5.0
        opt = seed["optimizer"]
        cap = opt["cap_default_pct"]
        floors = {b["key"]: b["floor_default_pct"] for b in opt["buckets"]}
        cand_pct = {str(t): float(v) * 0.9 for t, v in
                    (rss._bundle_for(frames, "all", "all")["weights"] * 100.0).items()}
        cand_pct["NEWC"] = 10.0
        cand_pct2 = {str(t): float(v) * 0.85 for t, v in
                    (rss._bundle_for(frames, "all", "all")["weights"] * 100.0).items()}
        cand_pct2["NEWC"] = 10.0
        cand_pct2["PXY"] = 5.0
        return {
            "seed": seed,
            "run": rss.run_simulation(frames, np_),
            "run_candidate": rss.run_simulation(
                frames, cand_pct,
                candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}]),
            "run_multi_candidate": rss.run_simulation(
                frames, cand_pct2,
                candidates=[{"ticker": "NEWC", "proxy": ""},
                           {"ticker": "PXY", "proxy": ""}]),
            "optimize_minvar": rss.run_optimize(
                frames, optimizer="min_variance", cap_pct=cap, floors=floors),
            "optimize_riskparity": rss.run_optimize(
                frames, optimizer="risk_parity", cap_pct=cap, floors={}),
            "trace": rss.run_trace(frames, cap_pct=cap, floors=floors),
            "frontier": rss.run_frontier(frames, cap_pct=cap, floors=floors,
                                         erp_pct=4.5),
            "optimize_minvar_candidate": rss.run_optimize(
                frames, optimizer="min_variance", cap_pct=cap, floors=floors,
                candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}]),
            "frontier_candidate": rss.run_frontier(
                frames, cap_pct=cap, floors=floors,
                candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}]),
        }

    def test_matches_golden(self):
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        mismatch = _deep_close(self._payload(), expected)
        self.assertIsNone(mismatch, f"risksim view diverges from golden at {mismatch}")


class TestFrontierCapm(unittest.TestCase):
    def test_capm_block_present_and_shaped(self):
        frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(frames)
        opt = seed["optimizer"]
        floors = {b["key"]: b["floor_default_pct"] for b in opt["buckets"]}
        out = rss.run_frontier(frames, cap_pct=opt["cap_default_pct"],
                               floors=floors, erp_pct=4.5)
        self.assertIsNone(out["error"])
        capm = out["capm"]
        self.assertEqual(capm["erp_pct"], 4.5)
        self.assertIsInstance(capm["rf_pct"], float)
        self.assertIsInstance(capm["beta_years"], int)
        self.assertIsInstance(capm["assumed_beta_names"], list)
        self.assertIn(capm["rf_src"], ("FRED DGS3MO", "fallback"))


class TestFrontierMemo(unittest.TestCase):
    def _sig(self, **over):
        base = dict(data_version="dv1", broker=["all"], history_start="all",
                    account=["all"], asset_class=["all"], cap_pct=40.0,
                    floors={"fixed_income": 10.0}, caps={}, erp_pct=4.5)
        base.update(over)
        return rss.frontier_sig(**base)

    def test_sig_deterministic_and_order_insensitive(self):
        self.assertEqual(self._sig(), self._sig())
        self.assertEqual(self._sig(broker=["a", "b"]), self._sig(broker=["b", "a"]))
        self.assertNotEqual(self._sig(), self._sig(erp_pct=5.0))
        self.assertEqual(len(self._sig()), 16)

    def test_memo_roundtrip_and_eviction(self):
        sigs = [self._sig(erp_pct=float(i)) for i in range(1, 9)]
        for s in sigs:
            rss.frontier_memo_put(s, {"series": [s]}, ["all"], ["all"])
        got = rss.frontier_memo_get(sigs[-1])
        self.assertEqual(got["payload"], {"series": [sigs[-1]]})
        self.assertEqual(got["account"], ["all"])
        self.assertLessEqual(len(rss._FRONTIER_MEMO), rss._FRONTIER_MEMO_MAX)
        self.assertIsNone(rss.frontier_memo_get("no-such-sig"))
        self.assertIsNone(rss.frontier_memo_get(None))


class TestSimulateMemo(unittest.TestCase):
    def test_sig_deterministic_and_sensitive(self):
        base = dict(data_version="dv1", broker=["all"], history_start="all",
                    account=["all"], asset_class=["all"],
                    weights={"AAA": 60.0, "BBB": 40.0}, candidates=[])
        s = rss.simulate_sig(**base)
        self.assertEqual(len(s), 16)
        self.assertEqual(s, rss.simulate_sig(**base))                 # deterministic
        self.assertNotEqual(s, rss.simulate_sig(**{**base,            # weight change
                            "weights": {"AAA": 61.0, "BBB": 39.0}}))
        self.assertNotEqual(s, rss.simulate_sig(**{**base,            # scope change
                            "account": ["acct1"]}))
        self.assertEqual(s, rss.simulate_sig(**{**base,               # weight order-insensitive
                            "weights": {"BBB": 40.0, "AAA": 60.0}}))

    def test_sig_hashes_candidates(self):
        base = dict(data_version="v", broker=["all"], history_start="all",
                    account=["all"], asset_class=["all"], weights={"A": 50.0, "B": 50.0})
        s1 = rss.simulate_sig(**base, candidates=[{"ticker": "X", "proxy": ""},
                                                  {"ticker": "Y", "proxy": "Z"}])
        s2 = rss.simulate_sig(**base, candidates=[{"ticker": "Y", "proxy": "Z"},
                                                  {"ticker": "X", "proxy": ""}])
        s3 = rss.simulate_sig(**base, candidates=[{"ticker": "X", "proxy": ""}])
        self.assertEqual(s1, s2)          # order-insensitive
        self.assertNotEqual(s1, s3)       # candidate set matters

    def test_memo_roundtrip_blank_and_cap(self):
        rss.simulate_memo_put("sig-x", {"vol_pct": 1}, ["all"], ["fixed_income"])
        got = rss.simulate_memo_get("sig-x")
        self.assertEqual(got["facts"], {"vol_pct": 1})
        self.assertEqual(got["account"], ["all"])
        self.assertEqual(got["asset_class"], ["fixed_income"])
        self.assertIsNone(rss.simulate_memo_get(""))                 # blank -> None
        self.assertIsNone(rss.simulate_memo_get("never-stored"))
        for i in range(rss._SIMULATE_MEMO_MAX + 3):                  # LRU cap
            rss.simulate_memo_put(f"k{i}", {"i": i}, ["all"], ["all"])
        self.assertLessEqual(len(rss._SIMULATE_MEMO), rss._SIMULATE_MEMO_MAX)


class TestSimulateAiFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        w = (rss._bundle_for(cls.frames, "all", "all")["weights"] * 100.0
             ).sort_values(ascending=False)
        cls.new_pct = {str(t): float(v) for t, v in w.items()}
        cls.new_pct[str(w.index[0])] -= 5.0
        cls.new_pct[str(w.index[-1])] += 5.0
        cls.view = rss.run_simulation(cls.frames, cls.new_pct)

    def test_ai_facts_present_and_keys(self):
        f = self.view["ai_facts"]
        for k in ("vol_pct", "sharpe", "sortino", "max_dd_pct", "var95_pct",
                  "cvar95_pct", "dr", "down_beta", "effective_n", "top5_pct",
                  "max_pct", "stressed_corr_avg", "stressed_dr", "weight_moves"):
            self.assertIn(k, f)
        for k in ("vol_pct", "sharpe", "top5_pct"):
            self.assertEqual(set(f[k]), {"before", "after", "delta"})

    def test_ai_facts_units_match_display_tiles(self):
        # the unit trap: vol is a fraction (x100), top-5 is already percent.
        f = self.view["ai_facts"]
        vol_after = self.view["headline"]["risk"][0]["value"]          # "12.34%"
        self.assertAlmostEqual(f["vol_pct"]["after"],
                               float(vol_after.rstrip("%")), places=2)
        top5_after = self.view["headline"]["concentration"][1]["value"]  # "45.2%"
        self.assertAlmostEqual(f["top5_pct"]["after"],
                               float(top5_after.rstrip("%")), places=1)

    def test_weight_moves_only_changed_sorted(self):
        moves = self.view["ai_facts"]["weight_moves"]
        self.assertTrue(all(abs(m["delta_pp"]) > 0.01 for m in moves))
        self.assertLessEqual(len(moves), 6)
        self.assertEqual(moves, sorted(moves,
                         key=lambda m: (-abs(m["delta_pp"]), m["ticker"])))

    def test_ai_facts_absent_on_error(self):
        bad = dict(self.new_pct); bad[next(iter(bad))] += 40.0
        v = rss.run_simulation(self.frames, bad)
        self.assertIsNotNone(v["error"])
        self.assertIsNone(v.get("ai_facts"))

    def test_ai_facts_json_nan_safe(self):
        json.dumps(self.view["ai_facts"], allow_nan=False)   # must not raise


class TestSimulateRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil, tempfile
        from fastapi.testclient import TestClient
        from terminal import server
        cls.ddir = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        cls.client = TestClient(server.app)
        w = (rss._bundle_for(hs.load_frames(cls.ddir), "all", "all")["weights"]
             * 100.0).sort_values(ascending=False)
        cls.reweight = {str(t): float(v) for t, v in w.items()}
        cls.reweight[str(w.index[0])] -= 5.0
        cls.reweight[str(w.index[-1])] += 5.0

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_success_carries_sig_and_warms_memo(self):
        r = self.client.post("/api/risksim/simulate",
                             json={"weights": self.reweight})
        self.assertEqual(r.status_code, 200)
        sig = r.json()["sig"]
        self.assertRegex(sig, r"^[0-9a-f]{16}$")
        self.assertNotIn("ai_facts", r.json())          # popped from the wire
        memo = rss.simulate_memo_get(sig)
        self.assertIsNotNone(memo)
        self.assertIn("vol_pct", memo["facts"])

    def test_error_run_has_no_sig(self):
        bad = dict(self.reweight); bad[next(iter(bad))] += 40.0
        r = self.client.post("/api/risksim/simulate", json={"weights": bad})
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(r.json()["error"])
        self.assertNotIn("sig", r.json())


class TestParity(unittest.TestCase):
    """Slow (boots Streamlit). Cross-checks the terminal before/after headline
    against the Streamlit Risk Simulation tab's st.metric values at the same
    reweight (seeded via whatif_new_pct → the same code path Run uses)."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        cls.frames = hs.load_frames(FIXTURE)
        w = (rss._bundle_for(cls.frames, "all", "all")["weights"] * 100.0
             ).sort_values(ascending=False)
        np_ = {str(t): float(v) for t, v in w.items()}
        np_[str(w.index[0])] -= 5.0
        np_[str(w.index[-1])] += 5.0
        cls.new_pct = np_
        cls.view = rss.run_simulation(cls.frames, np_)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        at.session_state["active_tab"] = "Risk Simulation"
        at.session_state["whatif_new_pct"] = np_   # seed the New % override
        at = at.run()
        # Click "Run simulation" (the handler has no st.rerun → one run suffices).
        for b in at.button:
            if b.label == "Run simulation":
                b.click(); break
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_headline_after_values_match_streamlit(self):
        if self.view["error"]:
            self.skipTest("scenario unavailable on fixture")
        metric_values = {m.value for m in self.at.metric}
        for group in ("risk", "diversification", "concentration"):
            for tile in self.view["headline"][group]:
                if tile["value"] == "—":
                    continue
                self.assertIn(tile["value"], metric_values,
                              f"{tile['label']}={tile['value']} not in Streamlit metrics")


class TestOptimizeParity(unittest.TestCase):
    """Slow (boots Streamlit). The terminal optimizer output must match the
    Streamlit ⚖ Optimizer Suggest buttons at identical cap/floors (both sides use
    the anchored slider defaults = the seed defaults)."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        cls.frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(cls.frames)["optimizer"]
        cls.cap = seed["cap_default_pct"]
        cls.floors = {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        at.session_state["active_tab"] = "Risk Simulation"
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def _streamlit_suggest(self, label):
        at = self.at
        clicked = False
        for b in at.button:
            if b.label == label:
                b.click(); clicked = True; break
        self.assertTrue(clicked, f"button not found: {label!r}")
        at2 = at.run()
        # SafeSessionState (AppTest's session_state wrapper) has no .get() —
        # __getattr__ blindly forwards to __getitem__, which raises on a
        # missing key. Use `in` (it defines __contains__) instead.
        ss = at2.session_state
        return (ss["whatif_new_pct"] if "whatif_new_pct" in ss else None,
                ss["whatif_opt_msg"] if "whatif_opt_msg" in ss else None)

    def test_min_variance_matches_streamlit(self):
        new_pct, msg = self._streamlit_suggest("Suggest min-variance weights")
        out = rss.run_optimize(self.frames, optimizer="min_variance",
                               cap_pct=self.cap, floors=self.floors)
        if out["kind"] != "success":
            self.skipTest(f"min-variance infeasible on fixture: {out['message']}")
        self.assertEqual(out["message"], msg[1])
        self.assertIsNone(_deep_close(
            out["new_pct"], {str(k): float(v) for k, v in new_pct.items()}))

    def test_risk_parity_matches_streamlit(self):
        new_pct, msg = self._streamlit_suggest("Suggest risk-parity weights")
        out = rss.run_optimize(self.frames, optimizer="risk_parity",
                               cap_pct=self.cap, floors={})
        if out["kind"] != "success":
            self.skipTest(f"risk-parity infeasible on fixture: {out['message']}")
        self.assertEqual(out["message"], msg[1])
        self.assertIsNone(_deep_close(
            out["new_pct"], {str(k): float(v) for k, v in new_pct.items()}))


class TestTraceParity(unittest.TestCase):
    """Slow (boots Streamlit). The terminal cap-curve trace must match the
    Streamlit "Trace vol vs concentration" output (points + current) at identical
    cap/floors (both sides use the anchored slider defaults = the seed defaults)."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        cls.frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(cls.frames)["optimizer"]
        cls.cap = seed["cap_default_pct"]
        cls.floors = {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        at.session_state["active_tab"] = "Risk Simulation"
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_trace_points_match_streamlit(self):
        at = self.at
        clicked = False
        for b in at.button:
            if b.label == "Trace vol vs concentration":
                b.click(); clicked = True; break
        self.assertTrue(clicked, "Trace button not found")
        ss = at.run().session_state
        self.assertIn("whatif_curve", ss)
        cv = ss["whatif_curve"]
        out = rss.run_trace(self.frames, cap_pct=self.cap, floors=self.floors)
        got = {(s["key"], round(p["cap"], 10)): p
               for s in out["series"] for p in s["points"]}
        exp = {(r["optimizer"], round(float(r["cap"]), 10)): r
               for _, r in cv["points"].iterrows()}
        self.assertEqual(set(got), set(exp))
        for k in got:
            self.assertIsNone(_deep_close(
                [got[k]["vol"], got[k]["effective_n"]],
                [float(exp[k]["vol"]), float(exp[k]["effective_n"])]))
        if out["current"] is not None and cv["current"] is not None:
            self.assertIsNone(_deep_close(
                [out["current"]["vol"], out["current"]["effective_n"]],
                [float(cv["current"]["vol"]), float(cv["current"]["effective_n"])]))


class TestCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_offline_provider_serves_bars(self):
        provider, cache_dir = rss._candidate_provider(self.frames.data_dir)
        self.assertIsNotNone(provider, "sidecar present -> offline provider expected")
        bars = provider("NEWC", date(2000, 1, 1), date(2040, 1, 1))
        self.assertTrue(bars)
        self.assertEqual(set(bars[0]), {"date", "close"})
        s = wd.fetch_candidate_history("NEWC", cache_dir=cache_dir, bars_provider=provider)
        self.assertFalse(s.empty)
        self.assertEqual(s.name, "NEWC")

    def _cand_pct(self, cand="NEWC", w=10.0):
        base = rss._bundle_for(self.frames, "all", "all")["weights"] * 100.0
        scaled = {str(t): float(v) * (100.0 - w) / 100.0 for t, v in base.items()}
        scaled[cand] = w
        return scaled

    def test_candidate_headline_matches_direct_engine(self):
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": ""}])
        self.assertIsNone(view["error"])
        provider, cache_dir = rss._candidate_provider(self.frames.data_dir)
        hist = wd.fetch_candidate_history("NEWC", cache_dir=cache_dir, bars_provider=provider)
        cur = rss._bundle_for(self.frames, "all", "all")["weights"]
        new = pd.Series({k: v / 100.0 for k, v in self._cand_pct().items()})
        new = new[new > 1e-9]
        bundle = rss._bundle_for(self.frames, "all", "all")
        res = compute_before_after(
            WhatIfScenario(candidate_ticker="NEWC", current_weights=cur, new_weights=new),
            self.frames.daily_prices, hist, bench_tr=bundle["bench_tr"],
            rf_series=rss._load_rf(self.frames.data_dir), history_start=None)
        self.assertIsNone(res["error"])
        div = view["headline"]["diversification"]
        mcr = [t for t in div if t["label"] == "MCR(NEWC)"]
        if res["headline"]["mcr_verdict"] != "unknown":
            self.assertEqual(len(mcr), 1)
            self.assertEqual(mcr[0]["value"], rss._fmt_pct(res["headline"]["mcr_candidate"]))

    def test_candidate_already_held_is_error(self):
        held = str(rss._bundle_for(self.frames, "all", "all")["weights"].index[0])
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": held, "proxy": ""}])
        self.assertIsNotNone(view["error"])
        self.assertIn("already held", view["error"])
        self.assertIsNone(view["headline"])

    def test_candidate_self_proxy_is_error(self):
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": "NEWC"}])
        self.assertIsNotNone(view["error"])
        self.assertIn("proxy", view["error"].lower())
        self.assertIsNone(view["headline"])

    def test_candidate_spliced_runs(self):
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": "PXY"}])
        self.assertIsNone(view["error"])
        self.assertIn("spliced", (view["coverage_html"] or "").lower())

    def test_candidate_proxy_empty_sets_note(self):
        # A proxy absent from the sidecar returns no bars -> soft-degrade to
        # unspliced with a `note` (present only in that case, so pure reweight is unchanged).
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": "ZZZ"}])
        self.assertIsNone(view["error"])
        self.assertIn("note", view)
        self.assertIn("ZZZ", view["note"])
        self.assertNotIn("spliced", (view["coverage_html"] or "").lower())

    def test_pure_reweight_shape_unchanged(self):
        base = rss._bundle_for(self.frames, "all", "all")["weights"] * 100.0
        np_ = {str(t): float(v) for t, v in base.sort_values(ascending=False).items()}
        np_[str(base.index[0])] -= 5.0; np_[str(base.index[-1])] += 5.0
        view = rss.run_simulation(self.frames, np_)
        self.assertNotIn("note", view)
        self.assertEqual(set(view),
                         {"error", "coverage_html", "weight_bars", "headline",
                          "detail", "ai_facts"})

    def test_candidate_detail_and_coverage(self):
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": ""}])
        self.assertIsNone(view["error"])
        self.assertIn("candidate <b>NEWC</b> from", (view["coverage_html"] or ""))
        div = view["detail"]["diversification"]
        if "mcr_html" in div:
            self.assertIn("MCR(NEWC)", div["mcr_html"])
        json.dumps(view, allow_nan=False)

    def test_single_candidate_ai_facts_list(self):
        # A one-entry candidates list still ends up in ai_facts['candidates']
        # as a one-entry list — 'candidate' (singular) is gone.
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": ""}])
        self.assertIsNone(view["error"])
        self.assertNotIn("candidate", view["ai_facts"])
        cands = view["ai_facts"]["candidates"]
        div = view["headline"]["diversification"]
        mcr = [t for t in div if t["label"] == "MCR(NEWC)"]
        if mcr:
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["ticker"], "NEWC")
        else:
            self.assertEqual(cands, [])

    def _two_cand_pct(self, w1=5.0, w2=5.0):
        base = rss._bundle_for(self.frames, "all", "all")["weights"] * 100.0
        scaled = {str(t): float(v) * (100.0 - w1 - w2) / 100.0 for t, v in base.items()}
        scaled["NEWC"] = w1
        scaled["PXY"] = w2
        return scaled

    def test_two_candidates_simulate_ok(self):
        new_pct = self._two_cand_pct()
        view = rss.run_simulation(self.frames, new_pct,
                                  candidates=[{"ticker": "NEWC", "proxy": ""},
                                              {"ticker": "PXY", "proxy": ""}])
        self.assertIsNone(view["error"])
        labels = [t["label"] for t in view["headline"]["diversification"]]
        self.assertIn("MCR(NEWC)", labels)
        self.assertIn("MCR(PXY)", labels)
        self.assertEqual(len(view["ai_facts"]["candidates"]), 2)
        json.dumps(view, allow_nan=False)

    def test_single_candidate_via_list_still_ok(self):
        view = rss.run_simulation(self.frames, self._cand_pct(),
                                  candidates=[{"ticker": "NEWC", "proxy": ""}])
        self.assertIsNone(view["error"])


class TestCandidateParity(unittest.TestCase):
    """Slow (boots Streamlit). Candidate MCR parity: the terminal MCR tile matches
    the Streamlit tab's MCR metric at the same candidate. app.py fetches offline by
    pre-seeding the 7-day cache from the committed sidecar (both sides read it)."""

    @classmethod
    def setUpClass(cls):
        # addClassCleanup (LIFO, runs even if setUpClass raises partway) so the env
        # var and the pre-seeded synthetic cache files never leak into other tests /
        # data/ on a mid-setup failure (e.g. the AppTest boot timing out).
        cls.addClassCleanup(os.environ.pop, "APP_DATA_DIR", None)
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        cls.frames = hs.load_frames(FIXTURE)
        prov = rss._offline_candidate_provider(FIXTURE / "whatif_candidate_source.csv")
        for t in ("NEWC", "PXY"):
            p = wd._cache_path(t, wd.DEFAULT_CACHE_DIR)
            s = wd.fetch_candidate_history(t, cache_dir=wd.DEFAULT_CACHE_DIR, bars_provider=prov)
            if not s.empty:
                cls.addClassCleanup(p.unlink, missing_ok=True)
        base = (rss._bundle_for(cls.frames, "all", "all")["weights"] * 100.0)
        cand_pct = {str(t): float(v) * 0.9 for t, v in base.items()}
        cand_pct["NEWC"] = 10.0
        cls.view = rss.run_simulation(cls.frames, cand_pct,
                                      candidates=[{"ticker": "NEWC", "proxy": ""}])
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
        at.session_state["active_tab"] = "Risk Simulation"
        at.session_state["whatif_new_pct"] = cand_pct
        at.session_state["whatif_candidate"] = "NEWC"
        at = at.run()
        for b in at.button:
            if b.label == "Run simulation":
                b.click(); break
        cls.at = at.run()

    def test_mcr_verdict_matches_streamlit(self):
        if self.view["error"]:
            self.skipTest(f"candidate scenario unavailable on fixture: {self.view['error']}")
        div = self.view["headline"]["diversification"]
        mcr = [t for t in div if t["label"] == "MCR(NEWC)"]
        if not mcr:
            self.skipTest("MCR verdict unknown on fixture (short overlap)")
        verdict = mcr[0]["sub"].split(" ·")[0]
        md = " ".join(getattr(m, "value", "") for m in self.at.markdown)
        self.assertIn(verdict, md,
                      f"terminal MCR verdict {verdict!r} not found in Streamlit markdown")


class TestFrontier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(cls.frames)["optimizer"]
        cls.cap = seed["cap_default_pct"]
        cls.floors = {b["key"]: b["floor_default_pct"] for b in seed["buckets"]}

    def _run(self, **kw):
        kw.setdefault("cap_pct", self.cap)
        kw.setdefault("floors", self.floors)
        return rss.run_frontier(self.frames, **kw)

    def test_points_carry_lam_and_weights_pct(self):
        out = self._run()
        pts = out["series"][0]["points"]
        lams = [p["lam"] for p in pts]
        self.assertEqual(lams, sorted(lams, reverse=True))  # ladder order
        for p in pts:
            self.assertIn("weights_pct", p)
            self.assertAlmostEqual(sum(p["weights_pct"].values()),
                                   100.0, places=4)
        json.dumps(out, allow_nan=False)

    def test_frontier_skipped_reasons_grouped(self):
        out = self._run(cap_pct=1.0)   # infeasible cap -> everything skipped
        self.assertEqual(out["series"], [])
        self.assertEqual(sum(g["n"] for g in out["skipped_reasons"]),
                         out["skipped_n"])
        self.assertIsNotNone(out["empty_message"])
        self.assertIn(out["skipped_reasons"][0]["reason"].rstrip("."),
                      out["empty_message"])

    def test_caption_caps_fragment_matches_trace_wording(self):
        b0 = sorted({b for b in rss._mv_class_of(
            rss._bundle_for(self.frames, "all", "all")).values()
            if b != "other"})[0]
        out = self._run(caps={b0: 40.0})
        self.assertIn(f"Class caps applied: {b0.replace('_', ' ')} 40%",
                      out["caption_html"])
        self.assertNotIn("; class caps", out["caption_html"])

    def test_lede_no_longer_claims_missing_slices(self):
        seed = rss.build_risksim_view(self.frames)
        self.assertNotIn("coming in later slices", seed["caption_html"])

    def test_shape_and_jsonable(self):
        out = self._run()
        self.assertEqual(set(out.keys()),
                         {"error", "series", "current", "markers", "capm",
                          "skipped_n", "skipped_reasons", "empty_message",
                          "caption_html"})
        json.dumps(out, allow_nan=False)          # the route's exact encoder

    def test_series_matches_direct_engine(self):
        out = self._run(erp_pct=5.0)
        bundle = rss._bundle_for(self.frames, "all", "all")
        weights = bundle["weights"]
        class_of = rss._mv_class_of(bundle)
        buckets = sorted({b for b in class_of.values() if b != "other"})
        _ffm, ffd = fs._load_ff(self.frames.data_dir)
        rf_s = rsvc._load_rf(self.frames.data_dir)
        rf = (float(rf_s.iloc[-1]) if not rf_s.empty
              else rsvc.RF_FALLBACK_ANNUAL)
        mu = capm_expected_returns(self.frames.daily_prices, ffd,
                                   list(weights.index), rf_annual=rf,
                                   erp=0.05)["mu"]
        direct = trace_frontier(
            self.frames.daily_prices, weights, class_of, mu,
            name_cap=self.cap / 100.0,
            class_floors={b: float(self.floors.get(b, 0.0)) / 100.0
                          for b in buckets},
            class_caps={b: 1.0 for b in buckets})
        svc_pts = out["series"][0]["points"]
        self.assertEqual(len(svc_pts), len(direct["points"]))
        for got, (_i, exp) in zip(svc_pts, direct["points"].iterrows()):
            self.assertIsNone(_deep_close(got["vol"], float(exp["vol"])))
            self.assertIsNone(_deep_close(got["exp_return"],
                                          float(exp["exp_return"])))

    def test_series_is_a_single_frontier_series(self):
        out = self._run()
        self.assertEqual(len(out["series"]), 1)
        self.assertEqual(out["series"][0]["key"], "frontier")

    def test_markers_and_current_present(self):
        out = self._run()
        self.assertIsNotNone(out["current"])
        self.assertIn("exp_return", out["current"])
        self.assertTrue({m["key"] for m in out["markers"]}
                        <= {"min_variance", "risk_parity"})

    def test_caption_discloses_estimator_rf_and_erp(self):
        out = self._run(erp_pct=6.0)
        cap_html = out["caption_html"]
        self.assertIn("estimates, not", cap_html)
        self.assertIn("6.0%", cap_html)
        self.assertIn("CAPM", cap_html)

    def test_caption_names_assumed_beta_holdings(self):
        """The fixture's unpriced holding (SGOV) takes beta = 1.0."""
        out = self._run()
        weights = rss._bundle_for(self.frames, "all", "all")["weights"]
        unpriced = [str(s) for s in weights.index
                    if s not in self.frames.daily_prices.columns]
        if unpriced:
            self.assertIn("1.0 assumed", out["caption_html"])
            self.assertIn(unpriced[0], out["caption_html"])

    def test_higher_erp_lifts_returns_and_slides_points_up_the_frontier(self):
        """Scaling mu by c is equivalent to scaling lambda by 1/c, so a higher
        ERP does not move the frontier LOCUS -- it lifts every point's expected
        return and slides the ladder's samples toward the higher-vol end. Vol
        per ladder point is therefore non-decreasing, NOT fixed."""
        a = self._run(erp_pct=4.5)["series"][0]["points"]
        b = self._run(erp_pct=9.0)["series"][0]["points"]
        self.assertEqual(len(a), len(b))
        for pa, pb in zip(a, b):
            self.assertGreater(pb["exp_return"], pa["exp_return"])
            self.assertGreaterEqual(pb["vol"], pa["vol"] - 1e-9)

    def test_infeasible_cap_is_empty_not_error(self):
        out = self._run(cap_pct=1.0)
        self.assertIsNone(out["error"])
        self.assertEqual(out["series"], [])
        self.assertIsNotNone(out["empty_message"])
        self.assertGreater(out["skipped_n"], 0)

    def test_caption_discloses_floorless_erc_marker_when_floors_active(self):
        """Floors bind the frontier and its min-variance marker but not the
        risk-parity optimizer, so the ERC marker can land outside the traced
        frontier's feasible set. The fixture's default equity floor is
        active, so the disclosure should be present alongside the marker."""
        out = self._run()
        self.assertTrue({m["key"] for m in out["markers"]}
                        >= {"risk_parity"})
        self.assertIn("traced without class floors", out["caption_html"])

    def test_caption_omits_floorless_erc_note_when_floors_are_zero(self):
        """With every floor at 0%, the frontier and the ERC optimizer share
        the same feasible set, so the disclosure would be noise."""
        floors = {b: 0.0 for b in self.floors}
        out = self._run(floors=floors)
        self.assertNotIn("traced without class floors", out["caption_html"])

    def test_missing_factor_file_is_error_not_raise(self):
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("positions.csv", "positions_monthly.csv",
                         "daily_prices.csv", "twr_portfolio.csv",
                         "prices_latest.csv", "risk_free_rate.csv",
                         "accounts.csv", "transactions.csv",
                         "benchmark_spy_tr.csv", "irr_per_account.csv"):
                src = FIXTURE / name
                if src.exists():
                    shutil.copy(src, Path(tmp) / name)
            frames = hs.load_frames(tmp)
            out = rss.run_frontier(frames, cap_pct=self.cap,
                                   floors=self.floors)
            self.assertIsNotNone(out["error"])
            self.assertIn("fetch_ff_factors", out["error"])

    def test_frontier_includes_candidate(self):
        view = rss.run_frontier(
            self.frames, cap_pct=40.0, floors={},
            candidates=[{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        self.assertIsNone(view["error"])
        self.assertEqual(view["warnings"], [])
        self.assertTrue(view["series"])
        pt = view["series"][0]["points"][0]
        self.assertIn("NEWC", pt["weights_pct"])


class TestAugmentCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)          # sidecar present -> offline NEWC/PXY
        cls.bundle = rss._bundle_for(cls.frames, "all", "all")
        cls.held = str(cls.bundle["weights"].index[0])

    def test_happy_path_seeds_candidate_at_zero_with_class(self):
        prices, weights, class_of, warns = rss._augment_with_candidates(
            self.frames, self.bundle,
            [{"ticker": "NEWC", "asset_class": "equity", "proxy": ""}])
        self.assertIn("NEWC", weights.index)
        self.assertEqual(float(weights["NEWC"]), 0.0)
        self.assertEqual(class_of["NEWC"], "equity")
        self.assertIn("NEWC", prices.columns)
        # existing book preserved
        self.assertTrue(set(self.bundle["weights"].index).issubset(set(weights.index)))
        self.assertEqual(warns, [])

    def test_already_held_is_skipped_with_warning(self):
        _p, weights, _c, warns = rss._augment_with_candidates(
            self.frames, self.bundle,
            [{"ticker": self.held, "asset_class": "equity", "proxy": ""}])
        self.assertEqual(list(weights.index), list(self.bundle["weights"].index))
        self.assertTrue(any("already held" in w for w in warns))

    def test_duplicate_slot_skipped_once(self):
        _p, weights, _c, warns = rss._augment_with_candidates(
            self.frames, self.bundle,
            [{"ticker": "NEWC", "asset_class": "equity", "proxy": ""},
             {"ticker": "NEWC", "asset_class": "gold", "proxy": ""}])
        self.assertEqual(list(weights.index).count("NEWC"), 1)
        self.assertTrue(any("more than once" in w for w in warns))

    def test_unfetchable_ticker_skipped_with_warning(self):
        _p, weights, _c, warns = rss._augment_with_candidates(
            self.frames, self.bundle,
            [{"ticker": "ZZZNOPE", "asset_class": "equity", "proxy": ""}])
        self.assertNotIn("ZZZNOPE", weights.index)
        self.assertTrue(any("ZZZNOPE" in w for w in warns))

    def test_empty_candidates_is_passthrough(self):
        prices, weights, class_of, warns = rss._augment_with_candidates(
            self.frames, self.bundle, [])
        self.assertIs(prices, self.frames.daily_prices)
        self.assertTrue(weights.equals(self.bundle["weights"]))
        self.assertEqual(warns, [])

    def test_blank_slot_skipped_with_warning(self):
        _p, weights, _c, warns = rss._augment_with_candidates(
            self.frames, self.bundle,
            [{"ticker": "", "asset_class": "equity", "proxy": ""}])
        self.assertEqual(list(weights.index), list(self.bundle["weights"].index))
        self.assertTrue(any("blank" in w.lower() for w in warns))

    def test_short_history_no_proxy_dropped(self):
        import tempfile
        from pathlib import Path
        # A synthetic provider returning only ~120 bars for TINY (< 253 -> dropped).
        dates = pd.bdate_range("2023-01-02", periods=120)
        def prov(ticker, start, end):
            if ticker.upper() != "TINY":
                return []
            return [{"date": d.strftime("%Y-%m-%d"), "close": 100.0 + i}
                    for i, d in enumerate(dates)]
        cache = Path(tempfile.mkdtemp(prefix="cand_short_"))
        _p, weights, _c, warns = rss._augment_with_candidates(
            self.frames, self.bundle,
            [{"ticker": "TINY", "asset_class": "equity", "proxy": ""}],
            provider_override=(prov, cache))
        self.assertNotIn("TINY", weights.index)
        self.assertTrue(any("TINY" in w and "history" in w for w in warns))
