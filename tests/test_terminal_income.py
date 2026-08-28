# tests/test_terminal_income.py
import dataclasses
import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

import config_local as cfg
from terminal import holdings_service as hs
from terminal import income_service as ins
from terminal.holdings_service import fmt_money
from income_analytics import (forward_income, income_timeseries,
                              latest_ex_date_through, load_div_history,
                              trailing_income)

# Pinned so the golden + recompute parity are deterministic (the rendered Income
# surface is genuinely today-dependent). The live route + AppTest parity use the
# real today instead, so they agree with the Streamlit body on the same run.
ASOF = date(2026, 6, 28)
ASOF_TS = pd.Timestamp(ASOF)
SLEEVES = {cfg.TREASURY_LADDER_ACCOUNT_ID: "Treasury Ladder",
           cfg.TLH_ACCOUNT_ID: "Tax Loss Harvesting"}


def _recompute(frames):
    """Recompute the engine outputs the service is supposed to surface — the
    independent parity reference, same pinned asof."""
    tx = frames.transactions
    inc_ts = income_timeseries(tx)
    pm = frames.positions_monthly
    book_ts = pm["statement_date"].max()
    book = pm[pm["statement_date"] == book_ts]
    div_hist = load_div_history(FIXTURE)
    fwd_df, roll = forward_income(book, div_hist, ASOF, sleeves=SLEEVES)
    return {
        "inc_ts": inc_ts,
        "t12m": trailing_income(tx, ASOF_TS),
        "ytd": float(inc_ts.loc[inc_ts.index >= pd.Timestamp(ASOF.year, 1, 1),
                                "net"].sum()),
        "fwd_df": fwd_df, "roll": roll, "div_hist": div_hist,
        "book_ts": book_ts,
    }


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ins.build_income_view(cls.frames, asof=ASOF)

    def test_contract_keys(self):
        self.assertEqual(set(self.view.keys()),
                         {"meta", "caption", "received", "forward", "methodology"})

    def test_meta_keys(self):
        for k in ("accounts", "classes", "available_dates", "synthetic",
                  "filter", "book_date"):
            self.assertIn(k, self.view["meta"])

    def test_meta_filter_options_have_id_label(self):
        for opt in self.view["meta"]["accounts"] + self.view["meta"]["classes"]:
            self.assertIn("id", opt)
            self.assertIn("label", opt)

    def test_received_chart_series_aligned(self):
        # Every series in every view has len(values) == len(x).
        chart = self.view["received"]["chart"]
        for split in ("components", "by_account"):
            for vname in ("monthly", "yearly", "cumulative"):
                v = chart[split][vname]
                for s in v["series"]:
                    self.assertEqual(len(s["values"]), len(v["x"]),
                                     f"{split}.{vname} series {s['name']} ragged")

    def test_full_view_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)


class TestEngineParity(unittest.TestCase):
    """The terminal view values equal an independent recompute of the engine
    outputs on the SAME inputs + asof — the '1:1 numbers' guarantee at the data
    layer."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ins.build_income_view(cls.frames, asof=ASOF)
        cls.ref = _recompute(cls.frames)

    def test_received_kpis(self):
        kpis = self.view["received"]["kpis"]
        self.assertEqual(kpis[0]["value"], fmt_money(self.ref["t12m"]))
        self.assertEqual(kpis[1]["value"], fmt_money(self.ref["ytd"]))
        self.assertEqual(kpis[2]["value"],
                         fmt_money(self.ref["roll"]["projected_12m"]))

    def test_components_monthly_series_match_inc_ts(self):
        inc_ts = self.ref["inc_ts"]
        monthly = self.view["received"]["chart"]["components"]["monthly"]
        self.assertEqual(monthly["x"],
                         [i.strftime("%b %Y") for i in inc_ts.index])
        by_name = {s["name"]: s["values"] for s in monthly["series"]}
        for col, lab in zip(["dividends", "interest", "withholding"],
                            ["Dividends", "Interest", "Withholding"]):
            self.assertEqual(by_name[lab],
                             [float(v) for v in inc_ts[col].tolist()])

    def test_cumulative_has_net_line(self):
        cum = self.view["received"]["chart"]["components"]["cumulative"]
        net = [s for s in cum["series"] if s["name"] == "Net (cumulative)"]
        self.assertEqual(len(net), 1)
        self.assertEqual(net[0]["width"], 4)
        self.assertEqual(net[0]["values"],
                         [float(v) for v in self.ref["inc_ts"]["net"].cumsum().tolist()])

    def test_forward_kpis(self):
        roll = self.ref["roll"]
        kpis = {k["label"]: k["value"] for k in self.view["forward"]["kpis"]}
        self.assertEqual(kpis["Projected 12M income"],
                         fmt_money(roll["projected_12m"]))
        self.assertEqual(kpis["Yield (covered MV)"],
                         f"{roll['yield_on_covered_mv'] * 100:.2f}%")
        self.assertEqual(kpis["Coverage (% of NAV)"],
                         f"{roll['coverage_pct_nav'] * 100:.0f}%")

    def test_forward_detail_matches_payers(self):
        payers = self.ref["fwd_df"][self.ref["fwd_df"]["projected"] > 0]
        rows = self.view["forward"]["detail"]["rows"]
        self.assertEqual(len(rows), len(payers))
        # symbols + projected, in the engine's sorted (desc) order
        self.assertEqual([r["Symbol"] for r in rows],
                         [str(s) for s in payers["symbol"]])
        self.assertEqual([r["Projected 12M"] for r in rows],
                         [f"${v:,.0f}" for v in payers["projected"]])

    def test_history_caption(self):
        ex = latest_ex_date_through(self.ref["div_hist"], ASOF_TS)
        cap = self.view["forward"]["history_through_caption"]
        self.assertIn(ex.strftime("%b %d, %Y"), cap)

    def test_book_date(self):
        self.assertEqual(self.view["meta"]["book_date"],
                         pd.Timestamp(self.ref["book_ts"]).strftime("%b %d, %Y"))


class TestEmptyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_received_empty_when_no_income_rows(self):
        tx = self.frames.transactions
        no_income = tx[~tx["transaction_type"].isin(
            ("dividend", "interest", "withholding"))]
        frames2 = dataclasses.replace(self.frames, transactions=no_income)
        view = ins.build_income_view(frames2, asof=ASOF)
        self.assertTrue(view["received"]["empty"])
        self.assertEqual(view["received"]["kpis"], [])
        self.assertIsNone(view["received"]["chart"])

    def test_forward_unavailable_when_no_div_history(self):
        # A data_dir with no dividends_*.csv -> load_div_history returns {} ->
        # the forward section is unavailable (Section A still renders).
        empty_dir = ROOT / "tests" / "fixtures"  # no dividends_*.csv here
        frames2 = dataclasses.replace(self.frames, data_dir=str(empty_dir))
        view = ins.build_income_view(frames2, asof=ASOF)
        self.assertFalse(view["forward"]["available"])
        self.assertEqual(view["forward"]["kpis"], [])
        self.assertIsNone(view["forward"]["detail"])
        self.assertFalse(view["received"]["empty"])  # actuals still present


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

    def test_income_ok(self):
        r = self.client.get("/api/income")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("received", body)
        self.assertIn("forward", body)

    def test_unknown_params_ignored_not_500(self):
        r = self.client.get("/api/income", params={"account": "nope"})
        self.assertEqual(r.status_code, 200)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/income")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)

    def test_broker_param_accepted(self):
        r = self.client.get("/api/income", params={"broker": "all"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("brokers", r.json()["meta"])


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_income_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = ins.build_income_view(frames, asof=ASOF)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(view, expected)


class TestRawSeams(unittest.TestCase):
    """Unit-pin each public compute seam app.py and the service now share."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.inc_ts = income_timeseries(cls.frames.transactions)
        cls.ts_by = income_timeseries(cls.frames.transactions, by="account_id")

    def test_latest_book(self):
        pm = self.frames.positions_monthly
        book, book_ts = ins.latest_book(pm)
        self.assertEqual(book_ts, pm["statement_date"].max())
        self.assertTrue((book["statement_date"] == book_ts).all())

    def test_latest_book_empty(self):
        book, book_ts = ins.latest_book(self.frames.positions_monthly.iloc[0:0])
        self.assertIsNone(book_ts)
        self.assertTrue(book.empty)

    def test_ytd_income(self):
        got = ins.ytd_income(self.inc_ts, ASOF_TS)
        exp = float(self.inc_ts.loc[
            self.inc_ts.index >= pd.Timestamp(ASOF.year, 1, 1), "net"].sum())
        self.assertEqual(got, exp)

    def test_forward_payers_filters_and_adds_pct_mv(self):
        book, _ = ins.latest_book(self.frames.positions_monthly)
        fwd_df, roll = forward_income(
            book, load_div_history(FIXTURE), ASOF,
            sleeves={cfg.TREASURY_LADDER_ACCOUNT_ID: "Treasury Ladder",
                     cfg.TLH_ACCOUNT_ID: "Tax Loss Harvesting"})
        payers = ins.forward_payers(fwd_df, roll["nav"])
        self.assertTrue((payers["projected"] > 0).all())
        self.assertIn("pct_mv", payers.columns)
        exp = fwd_df.loc[fwd_df["projected"] > 0, "market_value"] / roll["nav"]
        pd.testing.assert_series_equal(payers["pct_mv"], exp, check_names=False)

    def test_components_frame_monthly_is_inc_ts(self):
        pd.testing.assert_frame_equal(
            ins.components_frame(self.inc_ts, "monthly"), self.inc_ts)

    def test_components_frame_yearly_resamples(self):
        pd.testing.assert_frame_equal(
            ins.components_frame(self.inc_ts, "yearly"),
            self.inc_ts.resample("YS").sum())

    def test_by_account_frame_monthly(self):
        got = ins.by_account_frame(self.ts_by, "monthly")
        exp = self.ts_by.reset_index()
        exp["account"] = (exp["account_id"].map(ins.ACCOUNT_DISPLAY)
                          .fillna(exp["account_id"]).astype(str))
        pd.testing.assert_frame_equal(got, exp)

    def test_by_account_frame_yearly_rolls(self):
        got = ins.by_account_frame(self.ts_by, "yearly")
        self.assertEqual(list(got.columns), ["month", "account", "net"])
        self.assertTrue(
            (got["month"] == got["month"].dt.to_period("Y").dt.to_timestamp()).all())


if __name__ == "__main__":
    unittest.main()
