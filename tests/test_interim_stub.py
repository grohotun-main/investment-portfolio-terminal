# tests/test_interim_stub.py — provisional interim stub period (spec 2026-08-22)
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

import interim_stub as ist                    # noqa: E402
from compute_twr import modified_dietz_period  # noqa: E402


def _flows(*rows):
    return pd.DataFrame(list(rows), columns=["settlement_date", "amount"]).assign(
        settlement_date=lambda d: pd.to_datetime(d["settlement_date"]))


class TestComputeInterimStub(unittest.TestCase):
    D0, D1 = pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-21")

    def test_no_flows_is_simple_ratio(self):
        s = ist.compute_interim_stub(self.D0, self.D1, 1000.0, 1030.0, _flows())
        self.assertAlmostEqual(s.return_pct, 0.03, places=12)
        self.assertEqual((s.days, s.n_flows, s.net_flow, s.flows_through), (21, 0, 0.0, None))
        self.assertEqual(s.as_facts()["net_flow_sign"], "none")

    def test_mid_period_withdrawal_matches_dietz_by_hand(self):
        # T=21 days; -100 on day 7 -> weight (21-7)/21 = 2/3
        fl = _flows(("2026-08-07", -100.0))
        s = ist.compute_interim_stub(self.D0, self.D1, 1000.0, 930.0, fl)
        expected = (930.0 - 1000.0 + 100.0) / (1000.0 - 100.0 * (14 / 21))
        self.assertAlmostEqual(s.return_pct, expected, places=12)
        self.assertAlmostEqual(
            s.return_pct,
            modified_dietz_period(1000.0, 930.0, fl, self.D0, self.D1), places=12)
        self.assertEqual(s.as_facts()["net_flow_sign"], "out")
        self.assertEqual(s.flows_through, pd.Timestamp("2026-08-07"))

    def test_flow_after_end_gets_zero_weight(self):
        # cash already inside nav_end: subtracted in the numerator, no weight
        fl = _flows(("2026-08-24", 500.0))
        s = ist.compute_interim_stub(self.D0, self.D1, 1000.0, 1530.0, fl)
        self.assertAlmostEqual(s.return_pct, 0.03, places=12)

    def test_gates(self):
        self.assertIsNone(ist.compute_interim_stub(self.D1, self.D0, 1.0, 1.0, _flows()))
        self.assertIsNone(ist.compute_interim_stub(self.D0, self.D0, 1.0, 1.0, _flows()))
        self.assertIsNone(ist.compute_interim_stub(self.D0, self.D1, float("nan"), 1.0, _flows()))
        self.assertIsNone(ist.compute_interim_stub(self.D0, self.D1, 0.0, 1.0, _flows()))
        self.assertIsNone(ist.compute_interim_stub(self.D0, self.D1, 1.0, -1.0, _flows()))

    def test_as_facts_is_deny_clean(self):
        import re
        deny = re.compile(r"(usd|dollar|amount|market_value|cost_basis|basis|proceeds|nav\b|"
                          r"balance|value|cost|price|gain|loss|_gl\b|equity)", re.I)
        s = ist.compute_interim_stub(self.D0, self.D1, 1000.0, 1030.0, _flows())
        for k in s.as_facts():
            self.assertIsNone(deny.search(k), k)
        self.assertTrue(s.as_facts()["provisional"])


class TestStubFlows(unittest.TestCase):
    def test_filters_type_scope_date_and_nan(self):
        tx = pd.DataFrame({
            "settlement_date": ["2026-07-31", "2026-08-03", "2026-08-05",
                                "2026-08-06", "2026-08-07", "2026-08-08"],
            "transaction_type": ["transfer_in", "transfer_out", "dividend",
                                 "transfer_in", "contribution", "transfer_out"],
            "flow_scope": [None, None, None, "internal", "", None],
            "amount": [50.0, -100.0, 3.0, 40.0, 25.0, float("nan")],
        })
        fl = ist.stub_flows(tx, after=pd.Timestamp("2026-07-31"))
        # on/before cutoff, dividend, internal leg and NaN amount all dropped
        self.assertEqual(list(fl["amount"]), [-100.0, 25.0])
        self.assertEqual(list(fl.columns), ["settlement_date", "amount"])

    def test_empty_and_missing_columns(self):
        self.assertTrue(ist.stub_flows(pd.DataFrame(), after=pd.Timestamp("2026-07-31")).empty)
        self.assertTrue(ist.stub_flows(None, after=pd.Timestamp("2026-07-31")).empty)


class TestAppendStub(unittest.TestCase):
    def _prepared(self):
        return pd.DataFrame({
            "month": pd.PeriodIndex(["2026-06", "2026-07"], freq="M"),
            "statement_date": pd.to_datetime(["2026-06-30", "2026-07-31"]),
            "prev_stmt_date": pd.to_datetime(["2026-05-31", "2026-06-30"]),
            "nav": [1000.0, 1100.0], "prev_nav": [990.0, 1000.0],
            "net_external_flow": [0.0, 0.0], "return_pct": [0.0101, 0.10],
            "n_flows": [0, 0], "is_real_statement": [True, True],
            "cum_return": [0.0101, 0.1111], "wealth_index": [1.0101, 1.1111],
            "wealth_peak": [1.0101, 1.1111], "twr_dd_pct": [0.0, 0.0],
            "month_end": pd.to_datetime(["2026-06-30", "2026-07-31"]),
        })

    def test_none_adds_only_the_flag_column(self):
        out = ist.append_stub(self._prepared(), None)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["provisional"]), [False, False])

    def test_appended_row_shape_and_chaining(self):
        stub = ist.compute_interim_stub("2026-07-31", "2026-08-21", 1100.0, 1133.0, _flows())
        out = ist.append_stub(self._prepared(), stub)
        self.assertEqual(len(out), 3)
        last = out.iloc[-1]
        self.assertEqual(str(last["month"]), "2026-08")
        self.assertEqual(last["statement_date"], pd.Timestamp("2026-08-21"))
        self.assertEqual(last["prev_stmt_date"], pd.Timestamp("2026-07-31"))
        self.assertEqual(last["month_end"], pd.Timestamp("2026-08-21"))
        self.assertTrue(bool(last["provisional"]) and not bool(last["is_real_statement"]))
        self.assertAlmostEqual(float(last["return_pct"]), 0.03, places=12)
        self.assertAlmostEqual(float(last["cum_return"]), 1.0101 * 1.10 * 1.03 - 1, places=6)
        self.assertEqual(list(out["provisional"][:2]), [False, False])
        self.assertEqual(out["wealth_peak"].iloc[-1], out["wealth_index"].max())

    def test_raw_string_dates_are_respected(self):
        raw = pd.DataFrame({"month": ["2026-07"], "statement_date": ["2026-07-31"],
                            "prev_stmt_date": ["2026-06-30"], "nav": [1100.0],
                            "return_pct": [0.1]})
        stub = ist.compute_interim_stub("2026-07-31", "2026-08-21", 1100.0, 1133.0, _flows())
        out = ist.append_stub(raw, stub)
        self.assertEqual(out.iloc[-1]["statement_date"], "2026-08-21")
        self.assertEqual(out.iloc[-1]["month"], "2026-08")


class TestToDateCagr(unittest.TestCase):
    def test_day_count(self):
        self.assertAlmostEqual(ist.to_date_cagr(0.10, 365), 0.10, places=12)
        self.assertAlmostEqual(ist.to_date_cagr(0.21, 730), math.sqrt(1.21) - 1, places=12)
        self.assertTrue(math.isnan(ist.to_date_cagr(0.1, 0)))
        self.assertTrue(math.isnan(ist.to_date_cagr(float("nan"), 10)))


# --------------------------------------------------------------------------- #
# Frames seam + consumers — on a temp copy of the synth fixture whose price
# date is pushed past the last statement (the committed fixture's price date
# EQUALS its last statement, so the gate is closed there and goldens hold).
# --------------------------------------------------------------------------- #
import shutil                                   # noqa: E402
import tempfile                                 # noqa: E402

from terminal import holdings_service as hs     # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "synth_data"


def stub_fixture(tmpdir: Path, *, bump_days: int = 20, interim: bool = True) -> Path:
    """Copy of the synth fixture with the price date moved past the last
    statement (+bump_days) and, optionally, a small interim CSV: a dividend,
    a transfer_out on day 5, a transfer_in dated after the price date, and an
    internal-tagged pair that must be ignored. Accounts come from the fixture
    itself so the roll-forward resolves."""
    dst = tmpdir / "synth_data"
    shutil.copytree(FIXTURE, dst)
    pl = pd.read_csv(dst / "prices_latest.csv", parse_dates=["as_of_date"])
    pl["as_of_date"] = pl["as_of_date"] + pd.Timedelta(days=bump_days)
    pl.to_csv(dst / "prices_latest.csv", index=False)
    # The benchmark TR series must reach the new price date too (+0.05%/day),
    # else the benchmark view correctly refuses the stub.
    for name in ("benchmark_spy_tr.csv", "benchmark_agg_tr.csv"):
        f = dst / name
        if not f.exists():
            continue
        tr = pd.read_csv(f, parse_dates=["date"])
        last = tr.iloc[-1]
        rows = []
        for i in range(1, bump_days + 1):
            g = 1.0005 ** i
            rows.append({"date": last["date"] + pd.Timedelta(days=i),
                         "close": float(last["close"]) * g, "shares": last["shares"],
                         "tr_value": float(last["tr_value"]) * g,
                         "tr_index": float(last["tr_index"]) * g, "daily_return": 0.0005})
        pd.concat([tr, pd.DataFrame(rows)], ignore_index=True).to_csv(f, index=False)
    pos = pd.read_csv(dst / "positions.csv")
    last = pos["statement_date"].max()
    acct = sorted(pos.loc[pos["statement_date"] == last, "account_id"].unique())[0]
    d0 = pd.Timestamp(last)
    if interim:
        rows = [
            (d0 + pd.Timedelta(days=3), "dividend", 12.5, None),
            (d0 + pd.Timedelta(days=5), "transfer_out", -500.0, None),
            (d0 + pd.Timedelta(days=bump_days + 2), "transfer_in", 800.0, None),
            (d0 + pd.Timedelta(days=6), "transfer_out", -300.0, "internal"),
            (d0 + pd.Timedelta(days=6), "transfer_in", 300.0, "internal"),
        ]
        pd.DataFrame([{
            "settlement_date": d.strftime("%Y-%m-%d"), "trade_date": d.strftime("%Y-%m-%d"),
            "broker": "fidelity", "account_id": acct, "transaction_type": t,
            "symbol": "", "cusip": "", "description": f"synthetic {t}",
            "quantity": 0.0, "price": 0.0, "amount": amt,
            "source_file": "test", "flow_scope": (scope or ""),
            "pair_id": ("p1" if scope else ""),
        } for d, t, amt, scope in rows]).to_csv(dst / "transactions_interim.csv", index=False)
    return dst


class TestFramesSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = stub_fixture(Path(cls.td.name))
        cls.frames = hs.apply_global_filters(hs.load_frames(cls.ddir), ["all"], "all")

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_fixture_itself_has_no_stub(self):
        self.assertIsNone(hs.interim_stub(hs.load_frames(FIXTURE)))

    def test_stub_wires_snapshots_flows_and_dates(self):
        f = self.frames
        s = hs.interim_stub(f)
        self.assertIsNotNone(s)
        d0 = pd.Timestamp(pd.to_datetime(f.twr_portfolio["statement_date"]).max())
        self.assertEqual(s.start_date, d0)
        self.assertEqual(s.end_date, pd.Timestamp(f.prices_as_of).normalize())
        self.assertEqual(s.days, 20)
        snap0 = hs._current_snap(f, d0.strftime("%Y-%m-%d"))
        exp0 = float(snap0["market_value_stmt"].where(
            snap0["market_value_stmt"].notna(), snap0["market_value"]).sum())
        self.assertAlmostEqual(s.nav_start, exp0, places=6)
        self.assertAlmostEqual(s.nav_end, float(hs._current_snap(f)["market_value"].sum()),
                               places=6)
        # flows: the -500 (day 5) and the +800 (after the price date); internal pair ignored
        self.assertEqual(s.n_flows, 2)
        self.assertAlmostEqual(s.net_flow, 300.0)
        self.assertEqual(s.flows_through, d0 + pd.Timedelta(days=22))
        fl = ist.stub_flows(f.transactions, after=d0)
        self.assertAlmostEqual(
            s.return_pct,
            modified_dietz_period(s.nav_start, s.nav_end, fl, s.start_date, s.end_date),
            places=12)

    def test_no_interim_file_still_stubs_on_the_marked_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            ddir = stub_fixture(Path(td), interim=False)
            f = hs.load_frames(ddir)
            s = hs.interim_stub(f)
            self.assertIsNotNone(s)
            self.assertEqual(s.n_flows, 0)
            snap = hs._current_snap(f)
            self.assertAlmostEqual(s.nav_start, float(snap["market_value_stmt"].sum()), places=6)
            self.assertAlmostEqual(s.nav_end, float(snap["market_value"].sum()), places=6)

    def test_scoped_frames_get_their_own_stub(self):
        opts = hs._broker_options(hs._current_snap(self.frames))[0]
        ids = [o["id"] for o in opts if o["id"] != "all"]
        if len(ids) < 2:
            self.skipTest("fixture has a single broker")
        sub = hs.apply_global_filters(hs.load_frames(self.ddir), [ids[0]], "all")
        s = hs.interim_stub(sub)
        self.assertIsNotNone(s)
        self.assertLess(s.nav_end, hs.interim_stub(self.frames).nav_end)

    def test_history_start_scope_keeps_the_stub(self):
        hist = hs._history_start_options(hs.load_frames(self.ddir))
        other = next((o["id"] for o in hist if o["id"] != "all"), None)
        if other is None:
            self.skipTest("fixture has a single history option")
        sub = hs.apply_global_filters(hs.load_frames(self.ddir), ["all"], other)
        self.assertIsNotNone(hs.interim_stub(sub))


from terminal import performance_service as ps    # noqa: E402


class TestPerformanceStub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = stub_fixture(Path(cls.td.name))
        cls.frames = hs.apply_global_filters(hs.load_frames(cls.ddir), ["all"], "all")
        cls.view = ps.build_performance_view(cls.frames)
        cls.base = ps.build_performance_view(hs.load_frames(FIXTURE))

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_fixture_payload_has_null_stub_and_no_provisional_point(self):
        self.assertIsNone(self.base.get("stub"))        # key absent without a stub
        self.assertFalse(any(p.get("provisional") for p in self.base["cum_twr"]["points"]))
        cards = {c["key"]: c for c in self.base["headline"]}
        self.assertNotIn("provisional", cards["cum_twr"]["sub"])

    def test_headline_cards_show_to_date_values(self):
        s = hs.interim_stub(self.frames)
        cards = {c["key"]: c for c in self.view["headline"]}
        port_view = ps.twr_view_for(self.frames)[0]
        stmt = ps.headline_raw(port_view, self.frames.irr_table, False)
        cum_td = (1 + stmt["cum"]) * (1 + s.return_pct) - 1
        self.assertEqual(cards["cum_twr"]["value"], hs._spct(cum_td * 100.0))
        self.assertIn(f"to {s.end_date.strftime('%Y-%m-%d')} · provisional",
                      cards["cum_twr"]["sub"])
        hd = ps.headline_raw(port_view, self.frames.irr_table, False, stub=s)
        self.assertAlmostEqual(hd["cum_to_date"], cum_td, places=12)
        self.assertAlmostEqual(hd["ann_to_date"],
                               ist.to_date_cagr(cum_td, hd["days_to_date"]), places=12)
        self.assertEqual(hd["cum"], stmt["cum"])                 # statement keys untouched
        self.assertEqual(cards["ann_twr"]["value"], hs._spct(hd["ann_to_date"] * 100.0))
        self.assertIn("provisional", cards["ann_twr"]["sub"])

    def test_cum_chart_last_point_is_provisional(self):
        pts = self.view["cum_twr"]["points"]
        self.assertTrue(pts[-1]["provisional"])
        self.assertFalse(any(p.get("provisional") for p in pts[:-1]))
        s = hs.interim_stub(self.frames)
        self.assertEqual(pts[-1]["x"], s.end_date.strftime("%Y-%m-%d"))
        self.assertIn("provisional", self.view["cum_twr"]["head"]["sub"])
        base_pts = self.base["cum_twr"]["points"]
        self.assertEqual(len(pts), len(base_pts) + 1)

    def test_payload_stub_block(self):
        st = self.view["stub"]
        self.assertTrue(st["provisional"])
        self.assertEqual(st["stub_days"], 20)
        self.assertIn("unaudited", st["caption"])
        self.assertEqual(st["net_flow_sign"], "in")

    def test_holdings_filter_has_no_stub(self):
        opts = hs._account_options(hs._current_snap(self.frames))[0]
        acct = next(o["id"] for o in opts if o["id"] != "all")   # option slug, not raw id
        v = ps.build_performance_view(self.frames, account=acct)
        self.assertTrue(v["meta"]["holdings_filter_active"])
        self.assertIsNone(v.get("stub"))
        self.assertFalse(any(p.get("provisional") for p in v["cum_twr"]["points"]))

    def test_untouched_sections_match_a_stub_free_build(self):
        import dataclasses
        s = hs.interim_stub(self.frames)
        pinned = dataclasses.replace(self.frames, prices_as_of=s.start_date)
        self.assertIsNone(hs.interim_stub(pinned))
        stmt_view = ps.build_performance_view(pinned)
        for key in ("periodic", "drawdown", "per_account"):
            self.assertEqual(self.view[key], stmt_view[key], key)


from terminal import benchmark_service as bs     # noqa: E402


class TestBenchmarkStub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = stub_fixture(Path(cls.td.name))
        cls.frames = hs.apply_global_filters(hs.load_frames(cls.ddir), ["all"], "all")
        cls.stub = hs.interim_stub(cls.frames)
        cls.view = bs.build_benchmark_view(cls.frames)
        # the same frames with the price date pinned to the statement = no stub
        cls.stmt = bs.build_benchmark_view(
            dataclasses.replace(cls.frames, prices_as_of=cls.stub.start_date))
        cls.base = bs.build_benchmark_view(hs.load_frames(FIXTURE))

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_fixture_rows_have_no_to_date_keys(self):
        self.assertIsNone(self.base.get("stub"))
        for r in self.base["returns_table"]["rows"]:
            self.assertNotIn("to_date", r)
        self.assertEqual(len(self.base["growth"]["series"]), 2)
        self.assertNotIn("caption", self.base["returns_table"])

    def test_rows_chain_the_stub_and_keep_statement_values(self):
        s = self.stub
        base_rows = {r["key"]: r for r in self.stmt["returns_table"]["rows"]}
        tr = hs._bench_tr_series(self.frames, "spy")
        b_stub = float(tr.loc[s.end_date] / tr.loc[s.start_date] - 1.0)
        seen = 0
        for r in self.view["returns_table"]["rows"]:
            b = base_rows[r["key"]]
            if not r["available"]:
                self.assertNotIn("to_date", r)
                continue
            seen += 1
            self.assertEqual(r["port"], b["port"])                 # statement keys untouched
            self.assertEqual(r["bench"], b["bench"])
            self.assertEqual(r["port_vol"], b["port_vol"])
            self.assertTrue(r["provisional"])
            self.assertEqual(r["to_date"], s.end_date.strftime("%Y-%m-%d"))
            if r["annualized"]:
                # calendar-day re-annualisation from the window's own origin
                # (the headline's basis), never n/12 + stub days
                from compare_to_benchmark import build_twr_comparison
                comp = build_twr_comparison(ps.twr_view_for(self.frames)[0], tr,
                                            base_amount=100_000.0)["comp"]
                n = r["n_months"]
                cum_p = (1 + b["port"]) ** (n / 12) - 1
                cum_b = (1 + b["bench"]) ** (n / 12) - 1
                days = (s.end_date - bs._window_origin(comp, r["key"])).days
                self.assertGreater(days, n * 28)
                self.assertAlmostEqual(
                    r["port_to_date"], ist.to_date_cagr(ist.chain(cum_p, s.return_pct), days), places=9)
                self.assertAlmostEqual(
                    r["bench_to_date"], ist.to_date_cagr(ist.chain(cum_b, b_stub), days), places=9)
            else:
                self.assertAlmostEqual(r["port_to_date"], (1 + b["port"]) * (1 + s.return_pct) - 1, places=9)
                self.assertAlmostEqual(r["bench_to_date"], (1 + b["bench"]) * (1 + b_stub) - 1, places=9)
            self.assertAlmostEqual(r["spread_to_date"], r["port_to_date"] - r["bench_to_date"], places=12)
        self.assertGreater(seen, 0)

    def test_growth_has_provisional_point_on_both_main_series(self):
        s = self.stub
        series = self.view["growth"]["series"]
        self.assertEqual([x["name"] for x in series], [x["name"] for x in self.stmt["growth"]["series"]])
        for x, x0 in zip(series, self.stmt["growth"]["series"]):
            self.assertEqual(len(x["points"]), len(x0["points"]) + 1)
            self.assertEqual(x["points"][:-1], x0["points"])        # statement points untouched
            self.assertTrue(x["points"][-1]["provisional"])
            self.assertEqual(x["points"][-1]["x"], s.end_date.strftime("%Y-%m-%d"))
        tr = hs._bench_tr_series(self.frames, "spy")
        b_stub = float(tr.loc[s.end_date] / tr.loc[s.start_date] - 1.0)
        self.assertAlmostEqual(series[0]["points"][-1]["v"], series[0]["points"][-2]["v"] * (1 + s.return_pct), places=6)
        self.assertAlmostEqual(series[1]["points"][-1]["v"], series[1]["points"][-2]["v"] * (1 + b_stub), places=6)

    def test_headline_and_methodology_are_to_date_consistent(self):
        s = self.stub
        twr = next(c for c in self.view["headline"] if c["key"] == "twr")
        self.assertIn(f"to {s.end_date:%Y-%m-%d} · provisional", twr["sub"])
        w = self.view["meta"]["window"]
        meth = self.view["disclosures"]["methodology"]
        self.assertIn(f"({w['years']:.2f}y)", meth)
        self.assertIn(s.end_date.strftime("%b %d, %Y"), meth)

    def test_summary_is_to_date_and_growth_has_dashed_tail(self):
        s = self.stub
        w, w0 = self.view["meta"]["window"], self.stmt["meta"]["window"]
        self.assertEqual(w["n_months"], w0["n_months"])                  # statement count
        self.assertEqual(w["start"], w0["start"])
        # to-date years are day-count based (the statement view's are n/12)
        self.assertAlmostEqual(w["years"], (s.end_date - pd.to_datetime(w["start"])).days / 365.0,
                               places=9)
        self.assertGreater(w["years"], w0["years"])
        self.assertIn("(provisional)", w["end"])
        st = self.view["stub"]
        tr = hs._bench_tr_series(self.frames, "spy")
        b_stub = float(tr.loc[s.end_date] / tr.loc[s.start_date] - 1.0)
        self.assertAlmostEqual(st["bench_stub_return"], b_stub, places=12)
        self.assertAlmostEqual(st["years_to_date"], w["years"], places=12)
        # to-date cum = statement cum chained with the stub (both sides)
        base_growth = self.stmt["growth"]["series"]
        port_stmt_cum = base_growth[0]["points"][-1]["v"] / self.stmt["growth"]["base"] - 1.0
        bench_stmt_cum = base_growth[1]["points"][-1]["v"] / self.stmt["growth"]["base"] - 1.0
        self.assertAlmostEqual(st["port_twr_cum_to_date"], (1 + port_stmt_cum) * (1 + s.return_pct) - 1, places=9)
        self.assertAlmostEqual(st["bench_twr_cum_to_date"], (1 + bench_stmt_cum) * (1 + b_stub) - 1, places=9)
        self.assertAlmostEqual(st["port_twr_ann_to_date"],
                               ist.to_date_cagr(st["port_twr_cum_to_date"], int(round(w["years"] * 365))), places=9)
        self.assertIn("provisional", self.view["returns_table"]["caption"])
        self.assertTrue(self.view["stub"]["provisional"])
        self.assertIn("provisional", self.view["meta"]["window"]["end"])

    def test_drawdown_and_periodic_stay_statement_only(self):
        self.assertEqual(self.view["drawdown"], self.stmt["drawdown"])
        self.assertEqual(self.view["periodic"], self.stmt["periodic"])

    def test_holdings_filter_has_no_stub(self):
        opts = hs._account_options(hs._current_snap(self.frames))[0]
        acct = next(o["id"] for o in opts if o["id"] != "all")
        v = bs.build_benchmark_view(self.frames, account=acct)
        self.assertTrue(v["meta"]["holdings_filter_active"])
        self.assertIsNone(v.get("stub"))


class TestStubGapCaption(unittest.TestCase):
    """DA-C-10: a stub the benchmark series cannot cover must be NAMED, not
    silently dropped — the live shape was a stale AGG leg truncating the
    60/40 blend, leaving a statement-anchored vs-Benchmark next to a
    provisional Performance tab with no explanation."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = stub_fixture(Path(cls.td.name))
        # Cut the SPY TR extension back so the series ends BEFORE the stub
        # window (stub_fixture extended it by bump_days to let stubs chain).
        f = cls.ddir / "benchmark_spy_tr.csv"
        tr = pd.read_csv(f, parse_dates=["date"]).iloc[:-20]
        tr.to_csv(f, index=False)
        cls.bench_end = tr["date"].max().strftime("%Y-%m-%d")
        cls.frames = hs.apply_global_filters(hs.load_frames(cls.ddir),
                                             ["all"], "all")

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_benchmark_view_names_the_gap(self):
        self.assertIsNotNone(hs.interim_stub(self.frames))  # stub exists
        v = bs.build_benchmark_view(self.frames)
        self.assertNotIn("stub", v)                          # not chained
        cap = v["returns_table"].get("caption") or ""
        self.assertIn("Provisional segment unavailable", cap)
        self.assertIn(self.bench_end, cap)
        self.assertNotIn("(provisional)", v["meta"]["window"]["end"])
        for r in v["returns_table"]["rows"]:
            self.assertNotIn("to_date", r)                   # honest rows

    def test_ai_facts_relay_the_reason(self):
        from terminal import ai_service as ai_mod
        pf = ai_mod.build_facts("portfolio", self.frames,
                                history_start="all", broker=["all"], dims=None)
        self.assertIn("before the provisional period",
                      pf.get("to_date_unavailable") or "")
        self.assertNotIn("stub", pf)
        bf = ai_mod.build_facts("benchmark", self.frames,
                                history_start="all", broker=["all"], dims=None)
        self.assertIn("Provisional segment unavailable",
                      bf.get("to_date_unavailable") or "")
        ai_mod.scrub_gate(pf)
        ai_mod.scrub_gate(bf)


class TestTapeStub(unittest.TestCase):
    def test_tape_cells_to_date_when_stubbed(self):
        from twr_aggregate import portfolio_twr_headline
        with tempfile.TemporaryDirectory() as td:
            f = hs.apply_global_filters(hs.load_frames(stub_fixture(Path(td))), ["all"], "all")
            s = hs.interim_stub(f)
            tape = {c["key"]: c for c in hs._kpi_tape(f)}
            base = {c["key"]: c for c in hs._kpi_tape(hs.load_frames(FIXTURE))}
            self.assertIn(f"to {s.end_date:%Y-%m-%d} · prov.", tape["cum_twr"]["sub"])
            self.assertIn("prov.", tape["annualized"]["sub"])
            self.assertNotIn("prov.", base["cum_twr"]["sub"])
            self.assertNotIn("prov.", base["annualized"]["sub"])
            h = portfolio_twr_headline(f.twr_portfolio)
            cum_td = (1 + h.cum) * (1 + s.return_pct) - 1
            self.assertEqual(tape["cum_twr"]["value"], hs._spct(cum_td * 100.0))
            self.assertNotEqual(tape["annualized"]["value"], base["annualized"]["value"])
            # untouched cells
            for key in ("irr", "max_dd"):
                if key in tape and key in base:
                    self.assertEqual(tape[key]["value"], base[key]["value"])


from terminal import ai_service as ai            # noqa: E402


class TestAiFactsStub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.frames = hs.apply_global_filters(
            hs.load_frames(stub_fixture(Path(cls.td.name))), ["all"], "all")
        cls.stub = hs.interim_stub(cls.frames)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def _facts(self, section, **dims):
        return ai.build_facts(section, self.frames, history_start="all",
                              broker=["all"], dims=dims or None)

    def test_performance_headline_to_date(self):
        f = self._facts("performance")
        td = f["headline"]["to_date"]
        self.assertTrue(td["provisional"])
        self.assertEqual(td["stub_days"], 20)
        self.assertEqual(td["end"], self.stub.end_date.strftime("%Y-%m-%d"))
        self.assertIsInstance(td["cum_twr_pct"], float)
        self.assertIsInstance(td["ann_twr_pct"], float)
        self.assertAlmostEqual(td["stub_return_pct"], round(self.stub.return_pct * 100, 2), places=6)
        self.assertIn("cum_twr_pct", f["headline"])            # statement key stays
        ai.scrub_gate(f)

    def test_portfolio_windows_to_date(self):
        f = self._facts("portfolio")
        rows = [w for w in f["windows"] if w["available"]]
        self.assertTrue(rows)
        for w in rows:
            td = w["to_date"]
            self.assertEqual(td["stub_days"], 20)
            self.assertTrue(td["provisional"])
            self.assertAlmostEqual(
                td["portfolio_cum_pct"],
                round(((1 + w["portfolio"]["twr_cum_pct"] / 100) * (1 + self.stub.return_pct) - 1) * 100, 2),
                places=6)
            self.assertIsInstance(td["benchmark_cum_pct"], float)
            self.assertIn("twr_cum_pct", w["portfolio"])
        for w in f["windows"]:
            if not w["available"]:
                self.assertNotIn("to_date", w)
        self.assertTrue(f["stub"]["provisional"])
        ai.scrub_gate(f)

    def test_benchmark_rows_to_date(self):
        f = self._facts("benchmark")
        rows = [r for r in f["returns"] if r["available"]]
        self.assertTrue(rows)
        for r in rows:
            self.assertTrue(r["provisional"])
            self.assertIsInstance(r["port_to_date_pct"], float)
            self.assertIsInstance(r["bench_to_date_pct"], float)
            self.assertAlmostEqual(r["spread_to_date_pp"],
                                   round(r["port_to_date_pct"] - r["bench_to_date_pct"], 2), places=1)
            self.assertIn("port_return_pct", r)                 # statement key stays
        ai.scrub_gate(f)

    def test_fixture_facts_have_no_to_date(self):
        base = hs.load_frames(FIXTURE)
        f = ai.build_facts("performance", base, history_start="all", broker=["all"], dims=None)
        self.assertNotIn("to_date", f["headline"])
        p = ai.build_facts("portfolio", base, history_start="all", broker=["all"], dims=None)
        self.assertNotIn("stub", p)
        self.assertFalse(any("to_date" in w for w in p["windows"]))
        b = ai.build_facts("benchmark", base, history_start="all", broker=["all"], dims=None)
        self.assertFalse(any("to_date" in r for r in b.get("returns", [])))

    def test_prompts_mention_provisional(self):
        for s in (ai._SYSTEM_PROMPT, ai._CHAT_SYSTEM, ai._BRIEF_SYSTEM):
            self.assertIn("provisional", s)
            self.assertIn(ai._TO_DATE_CLAUSE, s)


class TestHelpers(unittest.TestCase):
    """Review fix wave 2026-08-22: the shared to-date helpers."""

    def _stub(self, start="2026-07-31", end="2026-08-21"):
        return ist.compute_interim_stub(start, end, 1000.0, 1030.0, _flows())

    def test_to_date_span_uses_prev_stmt_date(self):
        port = pd.DataFrame({"month": pd.PeriodIndex(["2021-01", "2021-02"], freq="M"),
                             "statement_date": pd.to_datetime(["2021-01-31", "2021-02-28"]),
                             "prev_stmt_date": [pd.NaT, pd.Timestamp("2021-01-31")],
                             "return_pct": [float("nan"), 0.01]})
        origin, days = ist.to_date_span(port, self._stub())
        self.assertEqual(origin, pd.Timestamp("2021-01-31"))
        self.assertEqual(days, (pd.Timestamp("2026-08-21") - pd.Timestamp("2021-01-31")).days)

    def test_to_date_span_recomputed_frame_row0_has_return(self):
        # scoped frames (history_start / broker subsets): row 0 carries a return, NaT prev
        port = pd.DataFrame({"month": pd.PeriodIndex(["2021-01", "2021-02"], freq="M"),
                             "statement_date": pd.to_datetime(["2021-01-31", "2021-02-28"]),
                             "prev_stmt_date": [pd.NaT, pd.Timestamp("2021-01-31")],
                             "return_pct": [0.005, 0.01]})
        origin, _ = ist.to_date_span(port, self._stub())
        self.assertEqual(origin, pd.Timestamp("2020-12-31"))        # month-end BEFORE row 0
        raw = pd.DataFrame({"month": ["2021-01"], "statement_date": ["2021-01-31"],
                            "return_pct": [0.005]})
        self.assertEqual(ist.to_date_span(raw, self._stub())[0], pd.Timestamp("2020-12-31"))

    def test_chain(self):
        self.assertAlmostEqual(ist.chain(0.10, 0.05), 0.155, places=12)
        self.assertTrue(math.isnan(ist.chain(float("nan"), 0.05)))

    def test_ytd_to_date_same_vs_next_year(self):
        same = self._stub("2026-07-31", "2026-08-21")
        p, b = ist.ytd_to_date(0.10, 0.12, same, 0.02)
        self.assertAlmostEqual(p, 1.10 * 1.03 - 1, places=12)
        self.assertAlmostEqual(b, 1.12 * 1.02 - 1, places=12)
        nxt = self._stub("2026-12-31", "2027-01-15")
        self.assertEqual(ist.ytd_to_date(0.10, 0.12, nxt, 0.02), (nxt.return_pct, 0.02))

    def test_bench_stub_return_gates(self):
        s = self._stub()
        idx = pd.date_range("2026-07-31", "2026-08-21")
        tr = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
        self.assertAlmostEqual(ist.bench_stub_return(tr, s), tr.iloc[-1] / tr.iloc[0] - 1, places=12)
        self.assertIsNone(ist.bench_stub_return(tr.iloc[1:], s))     # starts after the stub start
        self.assertIsNone(ist.bench_stub_return(tr.iloc[:-1], s))    # ends before the stub end
        self.assertIsNone(ist.bench_stub_return(None, s))
        self.assertIsNone(ist.bench_stub_return(tr, None))


class TestReturnsTableYtdBoundary(unittest.TestCase):
    def test_ytd_row_is_stub_only_across_new_year(self):
        ends = pd.period_range("2025-12", "2026-12", freq="M").to_timestamp(how="end").normalize()
        comp = pd.DataFrame({"month": [str(d.to_period("M")) for d in ends[1:]],
                             "statement_date": ends[1:], "prev_stmt_date": ends[:-1],
                             "port_return": [0.01] * 12, "bench_return": [0.02] * 12})
        stub = ist.compute_interim_stub("2026-12-31", "2027-01-15", 1000.0, 1010.0, _flows())
        rt = bs._returns_table(comp, "SPY", stub=stub, bench_stub=0.005)
        ytd = next(r for r in rt["rows"] if r["key"] == "ytd")
        self.assertAlmostEqual(ytd["port"], 1.01 ** 12 - 1, places=12)      # statement: full 2026
        self.assertAlmostEqual(ytd["port_to_date"], 0.01, places=12)        # to-date: January alone
        self.assertAlmostEqual(ytd["bench_to_date"], 0.005, places=12)
        itd = next(r for r in rt["rows"] if r["key"] == "itd")
        days = (pd.Timestamp("2027-01-15") - pd.Timestamp("2025-12-31")).days
        self.assertAlmostEqual(itd["port_to_date"],
                               ist.to_date_cagr(1.01 ** 12 * 1.01 - 1, days), places=9)
        self.assertIn("provisional", rt["caption"])


class TestDualDateGuard(unittest.TestCase):
    def test_dual_date_latest_month_has_no_stub(self):
        import dataclasses
        with tempfile.TemporaryDirectory() as td:
            f = hs.apply_global_filters(hs.load_frames(stub_fixture(Path(td))), ["all"], "all")
            self.assertIsNotNone(hs.interim_stub(f))
            pm = f.positions_monthly.copy()
            latest = pm["statement_date"].max()
            idx = pm.index[pm["statement_date"] == latest][:1]
            pm.loc[idx, "statement_date"] = latest - pd.Timedelta(days=1)
            self.assertIsNone(hs.interim_stub(dataclasses.replace(f, positions_monthly=pm)))


if __name__ == "__main__":
    unittest.main()
