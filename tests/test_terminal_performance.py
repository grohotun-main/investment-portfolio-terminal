# tests/test_terminal_performance.py
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import performance_service as ps


class TestHeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ps.build_performance_view(cls.frames)

    def test_five_cards_in_order(self):
        keys = [c["key"] for c in self.view["headline"]]
        self.assertEqual(
            keys, ["cum_twr", "ann_twr", "irr", "max_dd", "best_worst"])

    def test_cum_twr_matches_link_returns(self):
        from compute_twr import link_returns
        port = ps._prepare_portfolio_twr(self.frames.twr_portfolio)
        expected = link_returns(port["return_pct"]) * 100.0
        # service formats with a signed pct; recover the number
        shown = self.view["headline"][0]["value"]
        self.assertEqual(shown, hs._spct(expected))

    def test_cum_color_matches_sign(self):
        c = self.view["headline"][0]
        cum = ps._prepare_portfolio_twr(self.frames.twr_portfolio)["cum_return"].iloc[-1]
        self.assertEqual(c["color"], "gain" if cum >= 0 else "loss")

    def test_max_dd_color_loss(self):
        self.assertEqual(self.view["headline"][3]["color"], "loss")

    def test_jsonable(self):
        json.dumps(self.view["headline"])


class TestCashFlows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_unfiltered_has_three_tiles(self):
        cf = ps.build_performance_view(self.frames)["cashflows"]
        self.assertIsNotNone(cf)
        for k in ("deposits", "withdrawals", "net"):
            self.assertIn(k, cf)
            self.assertTrue(cf[k]["value"].startswith(("$", "+$", "-$")))

    def test_deposits_match_external_positive(self):
        tx = self.frames.transactions
        ext = tx[tx["flow_scope"] == "external"]
        expected = float(ext.loc[ext["amount"] > 0, "amount"].sum())
        cf = ps.build_performance_view(self.frames)["cashflows"]
        self.assertEqual(cf["deposits"]["value"], ps.fmt_money(expected))
        self.assertEqual(cf["deposits"]["n"], int((ext["amount"] > 0).sum()))

    def test_hidden_under_class_filter(self):
        # any real class id from meta
        cid = ps.build_performance_view(self.frames)["meta"]["classes"][0]["id"]
        cf = ps.build_performance_view(self.frames, asset_class=cid)["cashflows"]
        self.assertIsNone(cf)


class TestSeries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ps.build_performance_view(cls.frames)

    def test_cum_twr_points_match_frame(self):
        port = ps._prepare_portfolio_twr(self.frames.twr_portfolio)
        pts = self.view["cum_twr"]["points"]
        self.assertEqual(len(pts), len(port))
        self.assertAlmostEqual(pts[-1]["v"], port["cum_return"].iloc[-1] * 100.0, places=6)

    def test_drawdown_points_nonpositive(self):
        for p in self.view["drawdown"]["points"]:
            self.assertLessEqual(p["dd"], 1e-9)

    def test_markers_have_x_and_label(self):
        for m in self.view["cum_twr"]["markers"]:
            self.assertIn("x", m)
            self.assertTrue(m["label"].startswith("+"))

    def test_jsonable(self):
        json.dumps({k: self.view[k] for k in ("cum_twr", "drawdown")})


class TestPeriodic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.per = ps.build_performance_view(cls.frames)["periodic"]

    def test_three_granularities(self):
        self.assertEqual(set(self.per.keys()), {"monthly", "quarterly", "yearly"})

    def test_monthly_bars_match_return_pct(self):
        port = ps._prepare_portfolio_twr(self.frames.twr_portfolio)
        base = port.dropna(subset=["return_pct"])
        self.assertEqual(len(self.per["monthly"]["bars"]), len(base))

    def test_quarterly_via_aggregate(self):
        from risk_metrics import aggregate_periodic_returns
        port = ps._prepare_portfolio_twr(self.frames.twr_portfolio)
        base = port.dropna(subset=["return_pct"])
        agg_ret, _ = aggregate_periodic_returns(base["return_pct"], base["month_end"], "Q")
        self.assertEqual(len(self.per["quarterly"]["bars"]), len(agg_ret))

    def test_winrate_format(self):
        self.assertRegex(self.per["monthly"]["winrate"], r"^\d+ / \d+ \(\d+%\)$")


class TestPerAccount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.pa = ps.build_performance_view(cls.frames)["per_account"]

    def test_rows_sorted_desc_by_cum(self):
        vals = [float(r["cum_twr"].rstrip("%").replace("+", "").replace("−", "-"))
                for r in self.pa["rows"]]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_short_history_suppresses_ann_and_irr(self):
        for r in self.pa["rows"]:
            if r["months"] < 12:
                self.assertEqual(r["ann_twr"], "—")
                self.assertEqual(r["irr"], "—")

    def test_skip_accounts_excluded(self):
        accts = {r["account"].rstrip(" †") for r in self.pa["rows"]}
        for sk in ps.SKIP_FROM_TWR_SUMMARY:
            label = ps.ACCOUNT_DISPLAY.get(sk, sk)
            self.assertNotIn(label, accts)

    def test_jsonable(self):
        json.dumps(self.pa)


class TestNav(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.nav = ps.build_performance_view(cls.frames)["nav"]

    def test_trio_keys(self):
        self.assertEqual(set(self.nav["trio"].keys()), {"current", "peak", "months"})

    def test_points_match_monthly_totals(self):
        ts = ps.nav_totals_raw(self.frames.positions_monthly)
        self.assertEqual(len(self.nav["points"]), len(ts))
        self.assertAlmostEqual(self.nav["points"][-1]["v"], float(ts.iloc[-1]["total"]), places=2)

    def test_markers_have_dollar_label(self):
        for m in self.nav["markers"]:
            self.assertRegex(m["label"], r"\(\+\$[\d,]+K\)")

    def test_jsonable(self):
        json.dumps(self.nav)


class TestFilteredSynthesis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.acct_id = ps.build_performance_view(cls.frames)["meta"]["accounts"][0]["id"]

    def test_unfiltered_view_equals_canonical(self):
        # all-selected must use the byte-identical canonical frame
        view = ps.build_performance_view(self.frames)
        port = ps._prepare_portfolio_twr(self.frames.twr_portfolio)
        self.assertEqual(view["headline"][0]["value"],
                         hs._spct(port["cum_return"].iloc[-1] * 100.0))

    def test_filtered_marks_holdings_filter_active(self):
        view = ps.build_performance_view(self.frames, account=self.acct_id)
        self.assertTrue(view["meta"]["holdings_filter_active"])
        self.assertIsNotNone(view["disclosures"]["holdings_filter"])
        self.assertEqual(view["headline"][2]["value"], "—")  # IRR hidden

    def test_filtered_twr_view_shape(self):
        snap_all = hs._current_snap(self.frames)
        bf, cf, _, _, _ = ps._resolve_filter(self.frames, snap_all, self.acct_id, "all")
        port = ps._prepare_portfolio_twr(self.frames.twr_portfolio)
        out = ps._filtered_twr_view(self.frames, port, bf, cf)
        # synthesized frame is twr_portfolio-shaped (or empty for a no-priceable slice)
        if not out.empty:
            for c in ("statement_date", "return_pct", "cum_return", "twr_dd_pct", "month_end"):
                self.assertIn(c, out.columns)


class TestDisclosuresAndContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ps.build_performance_view(cls.frames)

    def test_contract_keys(self):
        self.assertEqual(
            set(self.view.keys()),
            {"meta", "disclosures", "headline", "cashflows", "cum_twr",
             "drawdown", "periodic", "per_account", "nav"})

    def test_disclosures_keys(self):
        self.assertEqual(set(self.view["disclosures"].keys()),
                         {"holdings_filter", "combined_statement"})

    def test_combined_statement_unfiltered_only(self):
        # The synth fixture carries the bookkeeping columns but has no combined
        # months, so the note stays None — but it must NEVER be a string under a
        # holdings filter (gating is unfiltered-only).
        self.assertIsNone(self.view["disclosures"]["holdings_filter"])
        cid = self.view["meta"]["classes"][0]["id"]
        filtered = ps.build_performance_view(self.frames, asset_class=cid)
        self.assertIsNone(filtered["disclosures"]["combined_statement"])

    def test_meta_filter_options_have_id_label(self):
        for opt in self.view["meta"]["accounts"] + self.view["meta"]["classes"]:
            self.assertIn("id", opt)
            self.assertIn("label", opt)

    def test_full_view_jsonable_no_nan(self):
        # mirrors server allow_nan=False
        json.dumps(self.view, allow_nan=False)

    def test_filtered_view_jsonable_no_nan(self):
        cid = self.view["meta"]["classes"][0]["id"]
        filtered = ps.build_performance_view(self.frames, asset_class=cid)
        json.dumps(filtered, allow_nan=False)


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

    def test_performance_ok(self):
        r = self.client.get("/api/performance")
        self.assertEqual(r.status_code, 200)
        self.assertIn("headline", r.json())

    def test_rejects_unknown_account(self):
        r = self.client.get("/api/performance", params={"account": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_rejects_unknown_class(self):
        r = self.client.get("/api/performance", params={"asset_class": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_multi_account_ok(self):
        ids = [o["id"] for o in self.client.get("/api/performance").json()["meta"]["accounts"]][:2]
        if len(ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        r = self.client.get("/api/performance",
                            params=[("account", ids[0]), ("account", ids[1])])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["filter"]["account"], ids)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_performance_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = ps.build_performance_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(view, expected)


class TestRawSeams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.port = ps._prepare_portfolio_twr(cls.frames.twr_portfolio)

    def test_headline_raw_values(self):
        hd = ps.headline_raw(self.port, self.frames.irr_table, False)
        self.assertEqual(hd["cum"], float(self.port["cum_return"].iloc[-1]))
        self.assertEqual(hd["n"], int(self.port["return_pct"].notna().sum()))
        self.assertEqual(hd["mdd"], float(self.port["twr_dd_pct"].min()))
        self.assertEqual(hd["max_dd_month"],
                         self.port.loc[self.port["twr_dd_pct"].idxmin(), "month"])
        self.assertIsNotNone(hd["worst_ret"])
        self.assertIsNotNone(hd["best_ret"])

    def test_headline_raw_irr_hidden_when_filtered(self):
        hd = ps.headline_raw(self.port, self.frames.irr_table, True)
        self.assertTrue(math.isnan(hd["irr"]))

    def test_headline_raw_empty_view_is_none(self):
        import pandas as pd
        self.assertIsNone(ps.headline_raw(pd.DataFrame(), self.frames.irr_table, False))

    def test_headline_formatter_wraps_raw(self):
        cards = ps._headline(self.port, self.frames.irr_table, False)
        self.assertEqual([c["key"] for c in cards],
                         ["cum_twr", "ann_twr", "irr", "max_dd", "best_worst"])

    def test_cashflows_raw_matches_external(self):
        tx = self.frames.transactions
        ext = tx[tx["flow_scope"] == "external"]
        exp_dep = float(ext.loc[ext["amount"] > 0, "amount"].sum())
        cf = ps.cashflows_raw(tx, self.port, [], False, False)
        self.assertAlmostEqual(cf["deposits"], exp_dep)
        self.assertAlmostEqual(cf["net"], cf["deposits"] + cf["withdrawals"])

    def test_cashflows_raw_none_under_class_filter(self):
        self.assertIsNone(
            ps.cashflows_raw(self.frames.transactions, self.port, [], False, True))

    def test_events_grouped_raw_matches_wrapper(self):
        events = ps._onboarding_events(self.frames.positions_monthly)
        raw = ps.events_grouped_raw(events, [], False, False)
        self.assertEqual(list(raw.columns),
                         ["first_month", "account_display", "join_value"])
        wrapped = ps._events_grouped(self.frames, [], False, False)
        self.assertTrue(raw.reset_index(drop=True).equals(wrapped.reset_index(drop=True)))

    def test_events_grouped_raw_empty_under_class_filter(self):
        events = ps._onboarding_events(self.frames.positions_monthly)
        raw = ps.events_grouped_raw(events, [], False, True)
        self.assertTrue(raw.empty)

    def test_per_account_raw_cum_twr_matches_direct_compound(self):
        twr_account = ps._load_twr_account(self.frames.data_dir)
        raw = ps.per_account_raw(twr_account, self.frames.irr_table, [], False)
        self.assertFalse(raw.empty)
        acct = raw.iloc[0]["account_id"]
        g = twr_account[twr_account["account_id"].astype(str) == acct].sort_values("month")
        valid = g.dropna(subset=["return_pct"])
        expected = float(np.prod(1.0 + valid["return_pct"]) - 1.0)
        row = raw[raw["account_id"] == acct].iloc[0]
        self.assertAlmostEqual(row["cum_twr"], expected, places=12)

    def test_per_account_raw_sorted_desc_by_cum_twr(self):
        twr_account = ps._load_twr_account(self.frames.data_dir)
        raw = ps.per_account_raw(twr_account, self.frames.irr_table, [], False)
        vals = raw["cum_twr"].tolist()
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_per_account_raw_excludes_skip_accounts(self):
        twr_account = ps._load_twr_account(self.frames.data_dir)
        raw = ps.per_account_raw(twr_account, self.frames.irr_table, [], False)
        accts = set(raw["account_id"])
        for sk in ps.SKIP_FROM_TWR_SUMMARY:
            self.assertNotIn(str(sk), accts)

    def test_per_account_raw_ann_twr_nan_under_12_months(self):
        twr_account = ps._load_twr_account(self.frames.data_dir)
        raw = ps.per_account_raw(twr_account, self.frames.irr_table, [], False)
        for _, r in raw.iterrows():
            if r["months"] < 12:
                self.assertTrue(math.isnan(r["ann_twr"]))

    def test_nav_totals_raw_columns_and_sums(self):
        ts = ps.nav_totals_raw(self.frames.positions_monthly)
        self.assertEqual(list(ts.columns), ["month", "total", "n_accts", "month_end"])
        expected = self.frames.positions_monthly.groupby("month")["market_value"].sum()
        got = ts.set_index("month")["total"]
        self.assertTrue(got.reindex(expected.index).equals(expected))


class TestPerAccountDegenerateTWR(unittest.TestCase):
    """An account whose linked TWR is below -100% (a Dietz small-denominator
    artifact — e.g. a drained backdoor-IRA) must NOT crash the endpoint: the
    negative-base annualization would otherwise go complex and 500 the whole
    Performance tab for any such account not hidden by SKIP_FROM_TWR_SUMMARY
    (caught live on the demo-showcase dataset, 2026-07-13)."""

    @staticmethod
    def _degen_frame():
        import pandas as pd
        months = [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01", "2026-02"]
        rets = [np.nan] + [-0.02] * 12 + [-1.5]   # linked < -100%, 13 valid
        return pd.DataFrame({
            "account_id": ["DEGEN-X"] * 14,
            "month": months,
            "return_pct": rets,
            "nav": [1000.0] * 13 + [1.0],
            "net_external_flow": [0.0] * 14,
        })

    def test_ann_twr_is_nan_not_complex(self):
        import pandas as pd
        raw = ps.per_account_raw(self._degen_frame(), pd.DataFrame(), [], False)
        row = raw[raw["account_id"] == "DEGEN-X"].iloc[0]
        self.assertLess(row["cum_twr"], -1.0)
        self.assertNotIsInstance(row["ann_twr"], complex)
        self.assertTrue(pd.isna(row["ann_twr"]))

    def test_formatter_renders_dash_without_raising(self):
        import pandas as pd
        view = ps._per_account(self._degen_frame(), pd.DataFrame(), [], False)
        row = next(r for r in view["rows"] if "DEGEN-X" in r["account"])
        self.assertEqual(row["ann_twr"], "—")
        self.assertTrue(row["cum_twr"].startswith("-"))


class TestResolveFilterMulti(unittest.TestCase):
    """_resolve_filter accepts a scalar OR a list; multi takes the union; the
    *_active flags are proper-subset checks (mirrors app.py 1860-1861)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.snap_all = hs._current_snap(cls.frames)
        _, cls.acct_by_id = hs._account_options(cls.snap_all)
        cls.ids = list(cls.acct_by_id)  # every account option id

    def test_all_scalar_unfiltered(self):
        bf, cf, _, a_active, c_active = ps._resolve_filter(
            self.frames, self.snap_all, "all", "all")
        choices = sorted(self.frames.positions["bucket"].dropna().astype(str).unique())
        self.assertEqual(sorted(bf), choices)
        self.assertFalse(a_active)
        self.assertFalse(c_active)

    def test_all_list_equals_scalar(self):
        bf, *_ = ps._resolve_filter(self.frames, self.snap_all, ["all"], "all")
        bf2, *_ = ps._resolve_filter(self.frames, self.snap_all, "all", "all")
        self.assertEqual(sorted(bf), sorted(bf2))

    def test_two_ids_union(self):
        if len(self.ids) <= 2:
            self.skipTest("fixture has <3 account buckets (needs a proper subset)")
        bf, _, _, a_active, _ = ps._resolve_filter(
            self.frames, self.snap_all, self.ids[:2], "all")
        want = sorted(self.acct_by_id[i] for i in self.ids[:2])
        self.assertEqual(sorted(bf), want)
        self.assertTrue(a_active)  # a proper subset (fixture has >2 buckets)

    def test_selecting_every_id_reads_as_unfiltered(self):
        _, _, _, a_active, _ = ps._resolve_filter(
            self.frames, self.snap_all, self.ids, "all")
        self.assertFalse(a_active)  # full set != subset → not active

    def test_single_id_still_works(self):
        bf, _, _, a_active, _ = ps._resolve_filter(
            self.frames, self.snap_all, self.ids[0], "all")
        self.assertEqual(bf, [self.acct_by_id[self.ids[0]]])


class TestMultiActiveFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        _, acct_by_id = hs._account_options(hs._current_snap(cls.frames))
        cls.ids = list(acct_by_id)

    def test_multi_marks_active_and_echo(self):
        if len(self.ids) <= 2:
            self.skipTest("fixture has <3 account buckets (needs a proper subset)")
        v = ps.build_performance_view(self.frames, account=self.ids[:2])
        self.assertTrue(v["meta"]["account_filter_active"])
        self.assertTrue(v["meta"]["holdings_filter_active"])
        self.assertEqual(v["meta"]["filter"]["account"], self.ids[:2])

    def test_all_ids_not_active(self):
        v = ps.build_performance_view(self.frames, account=self.ids)
        # every id selected = not a proper subset -> not "active"...
        self.assertFalse(v["meta"]["account_filter_active"])
        # ...but the echo truthfully reflects the selection (orthogonal concern)
        self.assertEqual(v["meta"]["filter"]["account"], self.ids)


class TestScopedIrrHeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hs, cls.ps = hs, ps
        cls.frames = hs.load_frames(FIXTURE)

    def _irr_card(self, cards):
        return next(c for c in cards if c["key"] == "irr")

    def _twr_view(self, frames):
        # the same portfolio view build_performance_view feeds _headline on
        # the unfiltered path (port_view = port) — reuse the module's own
        # preparation helper so the unit test cannot drift from the real
        # call path.
        return self.ps._prepare_portfolio_twr(frames.twr_portfolio)

    def test_canonical_sub_unchanged(self):
        cards = self.ps._headline(self._twr_view(self.frames),
                                  self.frames.irr_table, False)
        self.assertEqual(self._irr_card(cards)["sub"], "Money-weighted")

    def test_holdings_filter_still_wins(self):
        cards = self.ps._headline(self._twr_view(self.frames),
                                  self.frames.irr_table, True,
                                  broker_scope=("Fidelity",))
        self.assertEqual(self._irr_card(cards)["sub"], "Holdings filter active")

    def test_scoped_sub_matrix(self):
        import pandas as pd
        finite = pd.DataFrame([{"account_id": "PORTFOLIO", "irr": 0.12}])
        cards = self.ps._headline(self._twr_view(self.frames), finite, False,
                                  broker_scope=("Fidelity",))
        card = self._irr_card(cards)
        self.assertEqual(card["sub"], "Money-weighted · Fidelity")
        self.assertNotEqual(card["value"], "—")
        empty = pd.DataFrame(columns=["account_id", "irr"])
        cards = self.ps._headline(self._twr_view(self.frames), empty, False,
                                  broker_scope=("Fidelity", "JPM"))
        card = self._irr_card(cards)
        self.assertEqual(card["sub"], "Money-weighted · n/a for this selection")
        self.assertEqual(card["value"], "—")

    def test_route_threads_broker_scope(self):
        opts = self.hs._broker_options(self.hs._current_snap(self.frames))[0]
        one = [o["id"] for o in opts if o["id"] != "all"][:1]
        if not one:
            self.skipTest("no broker options on fixture")
        f2 = self.hs.apply_global_filters(self.frames, one, "all")
        view = self.ps.build_performance_view(f2)
        card = self._irr_card(view["headline"])
        lbl = " + ".join(f2.broker_scope)
        self.assertIn(card["sub"],
                      {f"Money-weighted · {lbl}",
                       "Money-weighted · n/a for this selection"})
        has_row = (not f2.irr_table.empty and
                   "PORTFOLIO" in set(f2.irr_table["account_id"].astype(str)))
        self.assertEqual(card["value"] != "—", has_row)


if __name__ == "__main__":
    unittest.main()
