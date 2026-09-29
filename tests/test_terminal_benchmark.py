# tests/test_terminal_benchmark.py
import dataclasses
import json
import math
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import benchmark_service as bs
from compare_to_benchmark import build_twr_comparison
from risk_metrics import aggregate_periodic_returns


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (copied from
    test_terminal_factor). The benchmark-facts golden holds vol/CAGR values
    computed via numpy upstream (benchmark_service -> aggregate_periodic_returns),
    which are NOT bit-reproducible across BLAS builds, so raw floats compare
    within rel_tol; structure + every formatted string stay exact."""
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


def _engine_summary(frames):
    """The same comparison the service builds on, computed directly from the
    importable engine — the recompute-parity reference."""
    port = hs.load_frames  # noqa: F841 (keep import path warm)
    from terminal import performance_service as ps
    p = ps._prepare_portfolio_twr(frames.twr_portfolio)
    tr = hs._bench_tr_series(frames)
    return build_twr_comparison(p, tr, base_amount=100_000.0)


class TestHeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = bs.build_benchmark_view(cls.frames)
        cls.cmp = _engine_summary(cls.frames)

    def test_state_ok_on_fixture(self):
        self.assertEqual(self.view["meta"]["state"], "ok")

    def test_four_cards_in_order(self):
        keys = [c["key"] for c in self.view["headline"]]
        self.assertEqual(keys, ["twr", "irr", "winrate", "wealth"])

    def test_twr_value_matches_engine(self):
        s = self.cmp["summary"]
        self.assertEqual(self.view["headline"][0]["value"],
                         hs.fmt_pct(s["port_twr_ann"] * 100, 2))

    def test_twr_color_matches_sign(self):
        s = self.cmp["summary"]
        self.assertEqual(self.view["headline"][0]["color"],
                         "gain" if s["port_twr_ann"] >= 0 else "loss")

    def test_wealth_value_matches_engine(self):
        s = self.cmp["summary"]
        self.assertEqual(self.view["headline"][3]["value"],
                         hs.fmt_money(s["port_wealth_final"]))

    def test_wealth_delta_sign(self):
        s = self.cmp["summary"]
        port_final, bench_final = s["port_wealth_final"], s["bench_wealth_final"]
        self.assertEqual(self.view["headline"][3]["delta_dir"],
                         "up" if port_final >= bench_final else "down")

    def test_jsonable(self):
        json.dumps(self.view["headline"])


class TestWinRateDefinition(unittest.TestCase):
    """Benchmark win-rate = months portfolio BEAT SPY (spread>0), NOT months
    the portfolio was up. Guards the gotcha vs performance_service._periodic."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = bs.build_benchmark_view(cls.frames)
        cls.cmp = _engine_summary(cls.frames)

    def test_winrate_numerator_is_spread_wins(self):
        s = self.cmp["summary"]
        wins, losses = int(s["win_months"]), int(s["loss_months"])
        expected = f"{wins}/{wins + losses}  ({wins / max(1, wins + losses) * 100:.0f}%)"
        self.assertEqual(self.view["headline"][2]["value"], expected)

    def test_winrate_format(self):
        self.assertRegex(self.view["headline"][2]["value"],
                         r"^\d+/\d+  \(\d+%\)$")


class TestGrowth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = bs.build_benchmark_view(cls.frames)
        cls.cmp = _engine_summary(cls.frames)

    def test_series_carry_leading_base_point(self):
        comp = self.cmp["comp"]
        for ser in self.view["growth"]["series"]:
            # base point prepended -> len == len(comp) + 1
            self.assertEqual(len(ser["points"]), len(comp) + 1)
            self.assertAlmostEqual(ser["points"][0]["v"],
                                   self.view["growth"]["base"], places=6)

    def test_final_port_wealth_matches_summary(self):
        s = self.cmp["summary"]
        port = next(x for x in self.view["growth"]["series"] if x["key"] == "port")
        self.assertAlmostEqual(port["points"][-1]["v"],
                               float(s["port_wealth_final"]), places=4)


class TestDrawdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = bs.build_benchmark_view(cls.frames)
        cls.cmp = _engine_summary(cls.frames)

    def test_all_points_nonpositive(self):
        for ser in self.view["drawdown"]["series"]:
            for p in ser["points"]:
                self.assertLessEqual(p["dd"], 1e-9)

    def test_trio_values_match_summary(self):
        s = self.cmp["summary"]
        trio = self.view["drawdown"]["trio"]
        self.assertEqual(trio["port"]["value"], f"{s['port_max_dd']:+.1f}%")
        self.assertEqual(trio["bench"]["value"], f"{s['bench_max_dd']:+.1f}%")

    def test_spread_word_matches_sign(self):
        s = self.cmp["summary"]
        word = "shallower" if (s["port_max_dd"] - s["bench_max_dd"]) > 0 else "deeper"
        self.assertIn(word, self.view["drawdown"]["trio"]["spread"]["value"])


class TestPeriodic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.per = bs.build_benchmark_view(cls.frames)["periodic"]
        cls.cmp = _engine_summary(cls.frames)

    def test_three_granularities(self):
        self.assertEqual(set(self.per.keys()),
                         {"monthly", "quarterly", "yearly"})

    def test_monthly_count_matches_comp(self):
        comp = self.cmp["comp"]
        self.assertEqual(len(self.per["monthly"]["port"]), len(comp))
        self.assertEqual(len(self.per["monthly"]["bench"]), len(comp))
        self.assertEqual(len(self.per["monthly"]["spread"]), len(comp))

    def test_quarterly_via_aggregate(self):
        comp = self.cmp["comp"]
        agg, _ = aggregate_periodic_returns(
            comp["port_return"], comp["statement_date"], "Q")
        self.assertEqual(len(self.per["quarterly"]["port"]), len(agg))

    def test_spread_equals_port_minus_bench(self):
        for gran in ("monthly", "quarterly", "yearly"):
            b = self.per[gran]
            for sp, pv, bv in zip(b["spread"], b["port"], b["bench"]):
                self.assertAlmostEqual(sp["v"], pv["v"] - bv["v"], places=6)

    def test_first_monthly_bench_bar_from_raw_tr(self):
        # Independent of build_twr_comparison: recompute the first period's
        # benchmark return straight from the raw SPY tr_value series and confirm
        # the service's first monthly bench bar matches (checks a real point on
        # the curve against market data, not just the engine's own output).
        comp = self.cmp["comp"].sort_values("statement_date").reset_index(drop=True)
        tr = hs._bench_tr_series(self.frames)
        end, prev = comp.iloc[0]["statement_date"], comp.iloc[0]["prev_stmt_date"]
        expected = (tr.loc[end] / tr.loc[prev] - 1.0) * 100.0
        self.assertAlmostEqual(self.per["monthly"]["bench"][0]["v"], expected,
                               places=6)


class TestFiltered(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        base = bs.build_benchmark_view(cls.frames)
        cls.acct_id = base["meta"]["accounts"][0]["id"]
        cls.class_id = base["meta"]["classes"][0]["id"]

    def test_account_filter_marks_active_and_hides_irr(self):
        v = bs.build_benchmark_view(self.frames, account=self.acct_id)
        self.assertTrue(v["meta"]["holdings_filter_active"])
        self.assertTrue(v["meta"]["account_filter_active"])
        # IRR is hidden under any holdings filter, in every non-error path.
        if v["meta"]["state"] == "ok":
            self.assertEqual(v["headline"][1]["value"], "—")
            self.assertEqual(v["headline"][1]["sub"], "Holdings filter active")
            self.assertIn("Holdings-filter slice",
                          v["disclosures"]["methodology"])

    def test_filtered_view_jsonable_no_nan(self):
        for opt in (("account", self.acct_id), ("asset_class", self.class_id)):
            v = bs.build_benchmark_view(self.frames, **{opt[0]: opt[1]})
            json.dumps(v, allow_nan=False)


class TestFilteredIrrHidden(unittest.TestCase):
    """The filtered NUMERIC path can't be exercised end-to-end on this synth
    fixture: every single Account/Asset-class filter yields no SPY overlap
    (a 1-row synthetic slice has no >=2 dated months), so the filtered view
    never reaches state=='ok'. That is correct parity (the filtered path reuses
    the same parity-gated _filtered_twr_view + build_twr_comparison app.py uses);
    real accounts have multi-month overlap. We document that fixture limitation
    here and test the filter-dependent code paths (IRR hidden) at the unit
    level instead."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.cmp = _engine_summary(cls.frames)

    def test_no_single_filter_reaches_ok_on_fixture(self):
        # Acknowledge the coverage gap explicitly: if a future fixture change
        # makes a filter overlap, this flips and a real filtered-parity test
        # should be added.
        base = bs.build_benchmark_view(self.frames)
        states = set()
        for o in base["meta"]["accounts"]:
            states.add(bs.build_benchmark_view(
                self.frames, account=o["id"])["meta"]["state"])
        for o in base["meta"]["classes"]:
            states.add(bs.build_benchmark_view(
                self.frames, asset_class=o["id"])["meta"]["state"])
        self.assertNotIn("ok", states)
        self.assertTrue(states <= {"no_overlap", "empty_filtered"})

    def test_headline_hides_irr_under_holdings_filter(self):
        # Exercise the IRR-hidden branch directly, independent of fixture overlap.
        s = self.cmp["summary"]
        cards = bs._headline(s, irr_cmp=None, holdings_filter_active=True, short="SPY")
        irr = next(c for c in cards if c["key"] == "irr")
        self.assertEqual(irr["value"], "—")
        self.assertEqual(irr["sub"], "Holdings filter active")
        self.assertIsNone(irr["delta"])

    def test_headline_irr_insufficient_when_unfiltered_none(self):
        s = self.cmp["summary"]
        cards = bs._headline(s, irr_cmp=None, holdings_filter_active=False, short="SPY")
        irr = next(c for c in cards if c["key"] == "irr")
        self.assertEqual(irr["value"], "—")
        self.assertEqual(irr["sub"], "Insufficient cashflow data for this subset")


class TestEmptyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_no_bench_when_tr_absent(self):
        # Drop the SPY TR frame -> tr_lookup empty -> error-level no_bench.
        frames2 = dataclasses.replace(self.frames, spy_tr=pd.DataFrame())
        v = bs.build_benchmark_view(frames2)
        self.assertEqual(v["meta"]["state"], "no_bench")
        self.assertEqual(v["meta"]["message_level"], "error")
        self.assertTrue(v["message"])
        self.assertEqual(v["headline"], [])
        self.assertIsNone(v["growth"])

    def test_cash_class_filter_is_benign_empty(self):
        # A cash-only slice has no priceable holdings -> benign info state.
        base = bs.build_benchmark_view(self.frames)
        cash = next((o["id"] for o in base["meta"]["classes"]
                     if "cash" in o["id"].lower()), None)
        if cash is None:
            self.skipTest("no cash class in fixture")
        v = bs.build_benchmark_view(self.frames, asset_class=cash)
        self.assertEqual(v["meta"]["state"], "empty_filtered")
        self.assertEqual(v["meta"]["message_level"], "info")

    def test_empty_state_keeps_filter_options(self):
        # meta.accounts/classes must survive every early return so the endpoint
        # 422 validation still works.
        frames2 = dataclasses.replace(self.frames, spy_tr=pd.DataFrame())
        v = bs.build_benchmark_view(frames2)
        self.assertTrue(v["meta"]["accounts"])
        self.assertTrue(v["meta"]["classes"])


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = bs.build_benchmark_view(cls.frames)

    def test_contract_keys(self):
        self.assertEqual(
            set(self.view.keys()),
            {"meta", "message", "disclosures", "headline", "growth",
             "drawdown", "periodic", "returns_table"})

    def test_meta_filter_options_have_id_label(self):
        for opt in self.view["meta"]["accounts"] + self.view["meta"]["classes"]:
            self.assertIn("id", opt)
            self.assertIn("label", opt)

    def test_full_view_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)


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

    def test_benchmark_ok(self):
        r = self.client.get("/api/benchmark")
        self.assertEqual(r.status_code, 200)
        self.assertIn("headline", r.json())

    def test_rejects_unknown_account(self):
        r = self.client.get("/api/benchmark", params={"account": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_rejects_unknown_class(self):
        r = self.client.get("/api/benchmark", params={"asset_class": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_benchmark_default_is_spy(self):
        r = self.client.get("/api/benchmark")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["benchmark"]["id"], "spy")

    def test_benchmark_explicit_6040(self):
        r = self.client.get("/api/benchmark", params={"benchmark": "60_40"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["benchmark"]["id"], "60_40")

    def test_benchmark_bad_422(self):
        r = self.client.get("/api/benchmark", params={"benchmark": "bogus"})
        self.assertEqual(r.status_code, 422)

    def test_benchmark_all_sentinel_422(self):
        # "all" is a valid wildcard for account/asset_class/broker but not a
        # benchmark id — must NOT silently resolve to auto (_validate_filter_ids
        # exempts "all", so the route uses a direct membership check instead).
        r = self.client.get("/api/benchmark", params={"benchmark": "all"})
        self.assertEqual(r.status_code, 422)

    def test_harbor_broker_auto_stays_spy_when_not_bond_heavy(self):
        # auto is composition-driven: the fixture's harbor book is ~35%
        # fixed income (below the majority threshold), so a scoped auto
        # still compares against SPY.
        r = self.client.get("/api/benchmark", params={"broker": "harbor"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["benchmark"]["id"], "spy")


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_benchmark_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = bs.build_benchmark_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(view, expected)


class TestBenchMultiLabel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        _, acct_by_id = hs._account_options(hs._current_snap(cls.frames))
        cls.ids = list(acct_by_id)

    def test_subset_label_joins_multiple(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        v = bs.build_benchmark_view(self.frames, account=self.ids[:2])
        label = v["meta"]["subset_label"]
        _, acct_by_id = hs._account_options(hs._current_snap(self.frames))
        for i in self.ids[:2]:
            self.assertIn(acct_by_id[i], label)


class TestBrokerScopedCaptions(unittest.TestCase):
    """DA-D-3: under a broker scope the tab used to caption itself "Full
    combined portfolio (Alpine + Harbor)" and claim the canonical
    compute_twr series while actually rendering the NAV-weighted
    recompute (the docstring premise "no broker selector in the terminal"
    went stale at #323). Scope labels and the methodology's series
    description must track frames.broker_scope."""

    @classmethod
    def setUpClass(cls):
        frames = hs.load_frames(FIXTURE)
        snap = hs._current_snap(frames)
        broker_opts, cls.by_id = hs._broker_options(snap)
        cls.ids = [o["id"] for o in broker_opts]
        if len(cls.ids) >= 2:
            cls.scoped = hs.apply_global_filters(frames, [cls.ids[0]], "all")
            cls.view = bs.build_benchmark_view(cls.scoped)

    def test_scoped_labels_name_the_scope(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        scope_lbl = " + ".join(self.scoped.broker_scope)
        meta = self.view["meta"]
        self.assertIn(scope_lbl, meta["subset_label"])
        self.assertNotIn("Alpine + Harbor", meta["subset_label"])
        self.assertIn(scope_lbl, meta["filter_caption"])
        self.assertNotIn("Alpine + Harbor", meta["filter_caption"])

    def test_scoped_methodology_discloses_the_recompute(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        meth = self.view["disclosures"]["methodology"]
        self.assertIn("NAV-weighted", meth)
        self.assertNotIn("canonical compute_twr.py output", meth)

    def test_canonical_view_keeps_the_canonical_claim(self):
        v = bs.build_benchmark_view(hs.load_frames(FIXTURE))
        meth = v["disclosures"]["methodology"]
        self.assertIn("canonical compute_twr.py output", meth)


class TestAggFixture(unittest.TestCase):
    def test_agg_tr_fixture_aligns_with_spy_dates(self):
        spy = pd.read_csv(FIXTURE / "benchmark_spy_tr.csv", parse_dates=["date"])
        agg = pd.read_csv(FIXTURE / "benchmark_agg_tr.csv", parse_dates=["date"])
        self.assertEqual(list(agg.columns),
                         ["date", "close", "shares", "tr_value", "tr_index", "daily_return"])
        self.assertEqual(agg["date"].tolist(), spy["date"].tolist())
        self.assertAlmostEqual(float(agg["tr_index"].iloc[0]), 100.0, places=4)
        # bond-like: far less cumulative growth than the SPY fixture
        self.assertLess(float(agg["tr_value"].iloc[-1]), float(spy["tr_value"].iloc[-1]))


class TestBenchmarkResolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_registry_ids(self):
        self.assertEqual(set(hs.BENCHMARKS), {"spy", "60_40"})

    def _scoped_frames(self, fi_share: float):
        """Minimal broker-scoped Frames whose current snapshot (the sliced
        ``positions_monthly``) holds ``fi_share`` of MV in fixed income."""
        from dataclasses import replace
        pm = pd.DataFrame([
            {"month": "2026-04", "statement_date": pd.Timestamp("2026-04-30"),
             "broker": "harbor", "account_id": "H-1", "symbol": "COREB",
             "asset_class": "fixed_income",
             "market_value": fi_share * 100.0},
            {"month": "2026-04", "statement_date": pd.Timestamp("2026-04-30"),
             "broker": "harbor", "account_id": "H-1", "symbol": "SPY",
             "asset_class": "equity_etf",
             "market_value": (1.0 - fi_share) * 100.0},
        ])
        return replace(self.frames, positions_monthly=pm,
                       available_dates=["2026-04-30"],
                       broker_scope=("Harbor",))

    def test_auto_bond_heavy_scope_picks_6040(self):
        f = self._scoped_frames(0.65)
        self.assertEqual(
            hs.resolve_benchmark("auto", f.broker_scope, frames=f), "60_40")

    def test_auto_equity_scope_picks_spy(self):
        f = self._scoped_frames(0.30)
        self.assertEqual(
            hs.resolve_benchmark("auto", f.broker_scope, frames=f), "spy")

    def test_auto_fixture_harbor_scope_is_not_bond_heavy(self):
        # The committed fixture's harbor book is ~35% fixed income — below
        # the threshold, so a scoped auto still compares against SPY.
        f = hs.apply_global_filters(hs.load_frames(FIXTURE), ["harbor"], "all")
        self.assertEqual(f.broker_scope, ("Harbor",))
        self.assertEqual(
            hs.resolve_benchmark("auto", f.broker_scope, frames=f), "spy")

    def test_auto_whole_book_picks_spy(self):
        self.assertEqual(
            hs.resolve_benchmark("auto", None, frames=self.frames), "spy")

    def test_explicit_ids_pass_through(self):
        self.assertEqual(hs.resolve_benchmark("spy", ("Harbor",)), "spy")
        self.assertEqual(hs.resolve_benchmark("60_40", None), "60_40")

    def test_agg_tr_loaded_on_fixture(self):
        self.assertFalse(self.frames.agg_tr.empty)

    def test_blended_series_differs_from_spy(self):
        spy = hs._bench_tr_series(self.frames, "spy")
        blend = hs._bench_tr_series(self.frames, "60_40")
        self.assertFalse(blend.empty)
        # blended TR grows less than pure SPY over the fixture bull window
        self.assertLess(blend.iloc[-1] / blend.iloc[0], spy.iloc[-1] / spy.iloc[0])

    def test_default_series_is_spy(self):
        a = hs._bench_tr_series(self.frames)
        b = hs._bench_tr_series(self.frames, "spy")
        self.assertTrue(a.equals(b))


from period_returns import WINDOWS as _PR_WINDOWS


class TestBenchmarkSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.spy = bs.build_benchmark_view(cls.frames, benchmark="spy")
        cls.blend = bs.build_benchmark_view(cls.frames, benchmark="60_40")

    def test_default_is_spy_for_whole_book(self):
        v = bs.build_benchmark_view(self.frames)   # benchmark="auto", broker_scope None
        self.assertEqual(v["meta"]["benchmark"]["id"], "spy")
        self.assertEqual(v["meta"]["ticker"], "SPY")

    def test_explicit_6040_labels_and_id(self):
        self.assertEqual(self.blend["meta"]["benchmark"]["id"], "60_40")
        self.assertEqual(self.blend["meta"]["benchmark"]["short"], "60/40")
        self.assertEqual(self.blend["meta"]["ticker"], "60/40")

    def test_6040_bench_numbers_differ_from_spy(self):
        spy_ann = self.spy["headline"][0]["sub"]        # "SPY TR ann.: …"
        blend_ann = self.blend["headline"][0]["sub"]    # "60/40 TR ann.: …"
        self.assertNotEqual(spy_ann, blend_ann)
        self.assertIn("60/40", blend_ann)

    def test_6040_returns_table_itd_bench_differs_from_spy(self):
        # test_6040_bench_numbers_differ_from_spy above only proves the
        # formatted STRINGS differ, which the "SPY"/"60/40" label prefix alone
        # guarantees even if the underlying number were identical (a
        # relabel-without-feeding-the-blend regression). Pin the raw NUMBER
        # instead: the ITD row's bench value must differ between the two
        # benchmark views.
        spy_rows = {r["key"]: r for r in self.spy["returns_table"]["rows"]}
        blend_rows = {r["key"]: r for r in self.blend["returns_table"]["rows"]}
        self.assertTrue(spy_rows["itd"]["available"])
        self.assertTrue(blend_rows["itd"]["available"])
        self.assertNotAlmostEqual(spy_rows["itd"]["bench"],
                                  blend_rows["itd"]["bench"])

    def test_auto_resolves_6040_under_bond_heavy_scope(self):
        # auto keys on the scoped snapshot's composition, not the broker
        # name: scale the fixture's harbor fixed-income sleeve up until it
        # is the majority of the scoped book and the blend takes over.
        pm = self.frames.positions_monthly
        pm = pm[pm["broker"] == "harbor"].copy()
        pm.loc[pm["asset_class"] == "fixed_income", "market_value"] *= 10.0
        harbor = dataclasses.replace(self.frames, positions_monthly=pm,
                                     broker_scope=("Harbor",))
        v = bs.build_benchmark_view(harbor, benchmark="auto")
        self.assertEqual(v["meta"]["benchmark"]["id"], "60_40")

    def test_6040_unavailable_falls_back_to_spy(self):
        no_agg = dataclasses.replace(self.frames, agg_tr=pd.DataFrame())
        v = bs.build_benchmark_view(no_agg, benchmark="60_40")
        self.assertEqual(v["meta"]["state"], "ok")
        self.assertEqual(v["meta"]["benchmark"]["id"], "spy")
        self.assertTrue(v["meta"]["benchmark"]["unavailable_fallback"])

    def test_jsonable_no_nan(self):
        json.dumps(self.blend, allow_nan=False)


class TestReturnsTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = bs.build_benchmark_view(cls.frames, benchmark="spy")

    def test_rows_cover_all_windows_in_order(self):
        keys = [r["key"] for r in cls_rows(self)]
        self.assertEqual(keys, [w[0] for w in _PR_WINDOWS])

    def test_ytd_1y_itd_available_on_28mo_fixture(self):
        rows = {r["key"]: r for r in cls_rows(self)}
        self.assertTrue(rows["ytd"]["available"])
        self.assertTrue(rows["1y"]["available"])
        self.assertTrue(rows["itd"]["available"])

    def test_3y_5y_unavailable_on_28mo_fixture(self):
        rows = {r["key"]: r for r in cls_rows(self)}
        self.assertFalse(rows["3y"]["available"])
        self.assertFalse(rows["5y"]["available"])

    def test_bench_label_present(self):
        self.assertEqual(self.view["returns_table"]["bench_label"], "SPY")


def cls_rows(case):
    return case.view["returns_table"]["rows"]


BENCHMARK_FACTS_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                          / "terminal_ai_benchmark_facts_golden.json")


class TestBenchmarkFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_registered_section(self):
        from terminal import ai_service as ai
        self.assertIn("benchmark", ai.SECTIONS)
        self.assertEqual(ai.SECTIONS["benchmark"]["dims"], ("benchmark",))

    def test_scrub_clean_and_golden(self):
        from terminal import ai_service as ai
        facts = ai.build_facts("benchmark", self.frames, history_start="all",
                               broker=["all"], dims={"benchmark": "spy"})
        ai.scrub_gate(facts)
        self.assertTrue(BENCHMARK_FACTS_GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        golden = json.loads(BENCHMARK_FACTS_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_facts_available_and_dollar_free(self):
        from terminal import ai_service as ai
        facts = ai.build_facts("benchmark", self.frames, history_start="all",
                               broker=["all"], dims={"benchmark": "spy"})
        self.assertTrue(facts["available"])
        self.assertEqual(facts["section"], "benchmark")
        self.assertIn("returns", facts)
        blob = json.dumps(facts)
        self.assertNotIn("$", blob)                       # no dollar glyph
        # no raw dollar/nav channels leaked from the view
        for banned in ("wealth", "nav", "_usd", "amount", "growth"):
            self.assertNotIn(banned, blob.lower())

    def test_bad_benchmark_value_raises_dim_error(self):
        from terminal import ai_service as ai
        with self.assertRaises(ai.AIDimError):
            ai.build_facts("benchmark", self.frames, history_start="all",
                           broker=["all"], dims={"benchmark": "nope"})

    def test_consistency_block(self):
        from terminal import ai_service as ai
        f = ai._facts_benchmark(self.frames, "all", None, None)
        c = f["consistency"]
        self.assertTrue(c["available"])
        self.assertEqual(c["window_months"], 12)
        for k in ("hit_rate_pct", "tracking_error_pct", "n_windows"):
            self.assertIn(k, c)
        ai.scrub_gate(f)

    def test_consistency_uses_the_filtered_series_not_whole_book(self):
        """The Task-6 judgment call: consistency is built from
        view["periodic"]["monthly"] — the SAME (possibly-filtered) aligned
        pair build_benchmark_view already computed for the returns table/
        chart — rather than re-deriving a whole-book-only pair from
        frames.twr_portfolio (_facts_portfolio's route, which ignores any
        account/asset_class filter). No single filter on this synthetic
        fixture reaches meta.state=='ok' (pinned by
        TestFilteredIrrHidden.test_no_single_filter_reaches_ok_on_fixture),
        so the real filtered numeric path can't be exercised end-to-end here
        — same documented gap as the rest of this file. Stand in a synthetic
        'ok' filtered view whose monthly pair is deliberately DIFFERENT from
        the real whole-book series (port beats bench every month, a shape
        the real fixture's series does not have) and prove that shape — not
        the whole book's — is what reaches FACTS.consistency."""
        from terminal import ai_service as ai
        real_view = ai.bs.build_benchmark_view(self.frames, benchmark="spy")
        fake_meta = dict(real_view["meta"])
        fake_meta["holdings_filter_active"] = True
        fake_meta["window"] = {"n_months": 12, "years": 1.0,
                               "start": "Jan 31, 2025", "end": "Dec 31, 2025"}
        months = [f"2025-{m:02d}" for m in range(1, 13)]
        fake_view = {
            "meta": fake_meta,
            "returns_table": {"rows": real_view["returns_table"]["rows"]},
            "periodic": {"monthly": {
                "port": [{"x": mo, "v": 5.0} for mo in months],
                "bench": [{"x": mo, "v": 1.0} for mo in months],
            }},
        }
        with mock.patch.object(ai.bs, "build_benchmark_view",
                               return_value=fake_view):
            f = ai._facts_benchmark(self.frames, "all", None,
                                    {"account": "test_a"})
        self.assertTrue(f["holdings_filter_active"])
        c = f["consistency"]
        self.assertTrue(c["available"])
        # Port (5%/mo) beat bench (1%/mo) in all 12 months of the FAKE
        # filtered series -> 100% hit rate. The real whole-book series
        # (golden: n_windows 16, hit_rate 100.0) can't distinguish this from
        # a whole-book leak by hit-rate alone, so pin TE/IR too: a 4pp/mo
        # constant spread is a materially different (near-zero-variance)
        # active-return series from the real book's.
        self.assertEqual(c["hit_rate_pct"], 100.0)
        self.assertEqual(c["n_windows"], 1)         # exactly 12 months in
        self.assertAlmostEqual(c["tracking_error_pct"], 0.0, places=2)
        ai.scrub_gate(f)


if __name__ == "__main__":
    unittest.main()
