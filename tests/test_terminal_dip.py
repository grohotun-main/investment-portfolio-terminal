# tests/test_terminal_dip.py
import dataclasses
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import dip_service as ds
import dip_analytics
import tail_risk
import turbulence


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (copied from
    test_terminal_factor). The dip golden holds EVT/GPD-fit + bootstrap-quantile
    outputs, which are NOT bit-reproducible across LAPACK/scipy builds, so raw
    floats compare within rel_tol; structure + every formatted string stay exact."""
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
        cls.view = ds.build_dip_view(cls.frames)

    def test_contract_keys(self):
        self.assertEqual(set(self.view.keys()),
                         {"meta", "caption", "turbulence", "legend", "empty", "cards"})

    def test_cards_present_and_keyed(self):
        # GLD has <MIN_HISTORY_DAYS rows in the fixture, so only 2 cards in
        # the trio; DISPLAY order is _TRIO's own (SPY before SCHD) while the
        # BUILD order stays trio-first-alpha (what the parity test pairs on).
        self.assertEqual([c["symbol"] for c in self.view["cards"]], ["SPY", "SCHD"])
        base = {"symbol", "today_regime", "regime_chip", "verdict",
                "kpis", "bridge_text", "forward_table", "further_fall",
                "time_underwater", "track_record", "underwater"}
        for c in self.view["cards"]:
            extra = set(c.keys()) - base
            # the registered referee block rides ONLY the artifact's ticker
            self.assertEqual(extra, {"referee"} if c["symbol"] == "SPY"
                             else set(),
                             f"unexpected card keys on {c['symbol']}")

    def test_verdict_shape(self):
        v = self.view["cards"][0]["verdict"]
        self.assertEqual(set(v.keys()),
                         {"band", "level", "text", "omega", "baseline_omega",
                          "ci_lo", "ci_hi", "n"})
        self.assertIn(v["band"],
                      {"strong", "neutral", "weak", "shallow", "inconclusive"})
        self.assertIn(v["level"], {"success", "info", "warning"})

    def test_forward_table_aligned(self):
        ft = self.view["cards"][0]["forward_table"]
        self.assertEqual(len(ft["rows"]), len(ft["tints"]))
        self.assertEqual(len(ft["rows"]), len(ft["numeric"]))

    def test_underwater_xy_aligned(self):
        uw = self.view["cards"][0]["underwater"]
        self.assertTrue(len(uw) > 0)
        self.assertEqual(set(uw[0].keys()), {"x", "v"})

    def test_full_view_jsonable_no_nan(self):
        body = json.dumps(self.view, allow_nan=False)  # raises if any NaN leaks
        self.assertNotIn("NaN", body)


class TestTrDrawdownSub(unittest.TestCase):
    """current_dd's TR-basis sub-line: shown only when the total-return
    drawdown differs from the price drawdown at display precision (the house
    _dual taste — no repeated numbers), absent entirely when TR is NaN."""

    def _kpis(self, tr_dd=float("nan")):
        state = {"current_dd": -0.021, "frac_below_52w_high": 0.021,
                 "pct_history_shallower": 35.0, "pct_history_deeper": 65.0,
                 "n_days": 1000}
        ymet = {"current_yield": float("nan"), "percentile": float("nan")}
        return ds._kpis(state, 3, ymet, "2026-07-17", tr_dd)

    def test_sub_present_when_tr_differs(self):
        k = self._kpis(tr_dd=-0.019)["current_dd"]
        self.assertEqual(k["sub"], "-1.9% with dividends reinvested")
        self.assertIn("total-return basis: -1.9%", k["help"])

    def test_sub_hidden_when_equal_at_display_precision(self):
        k = self._kpis(tr_dd=-0.0212)["current_dd"]   # both format to -2.1%
        self.assertIsNone(k["sub"])
        self.assertIn("total-return basis: -2.1%", k["help"])

    def test_sub_and_help_suffix_absent_when_tr_nan(self):
        k = self._kpis()["current_dd"]
        self.assertIsNone(k["sub"])
        self.assertNotIn("total-return", k["help"])


class TestTurbulence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ds.build_dip_view(cls.frames)

    def test_turbulence_matches_engine_or_none(self):
        dp = self.frames.daily_prices
        cols = [c for c in ds.MACRO if not dp.empty and c in dp.columns]
        rets = (dp[cols].sort_index().pct_change(fill_method=None).dropna()
                if cols else pd.DataFrame())
        tb = self.view["turbulence"]
        if not (rets.shape[1] >= 2 and len(rets) > 30):
            self.assertIsNone(tb)
            return
        exp = turbulence.turbulence_now(rets)
        self.assertEqual(tb["regime"], exp["regime"])
        self.assertEqual(tb["n"], len(rets))
        if np.isfinite(exp["percentile"]):
            self.assertAlmostEqual(tb["percentile"], exp["percentile"], places=9)
        else:
            self.assertIsNone(tb["percentile"])


class TestTurbulenceSnapshotSeam(unittest.TestCase):
    def test_matches_turbulence_now(self):
        frames = hs.load_frames(FIXTURE)
        snap = ds.turbulence_snapshot(frames.daily_prices)
        dp = frames.daily_prices
        cols = [c for c in ds.MACRO if not dp.empty and c in dp.columns]
        rets = (dp[cols].sort_index().pct_change(fill_method=None).dropna()
                if cols else pd.DataFrame())
        if not (rets.shape[1] >= 2 and len(rets) > 30):
            self.assertIsNone(snap)
        else:
            exp = turbulence.turbulence_now(rets)
            self.assertEqual(snap["regime"], exp["regime"])
            self.assertEqual(snap["n"], len(rets))


class TestEngineParity(unittest.TestCase):
    """Terminal card values equal an independent recompute of the engine on the
    same inputs — the '1:1 numbers' guarantee at the data layer."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ds.build_dip_view(cls.frames)
        import dip_adhoc
        cls.hist, cls.divs = ds._load_dip_csvs(cls.frames.data_dir)
        cls.adhoc = dip_adhoc

    def _slice(self, sym):
        return self.adhoc.slice_symbol(self.hist, self.divs, sym)

    def test_each_card_matches_engine(self):
        da = dip_analytics
        for card in self.view["cards"]:
            sym = card["symbol"]
            price, tr, dser = self._slice(sym)
            state = da.drawdown_state(price)
            # KPI numbers
            self.assertEqual(card["kpis"]["current_dd"]["value"],
                             f"{state['current_dd'] * 100:.1f}%")
            self.assertEqual(card["kpis"]["deeper_than"]["value"],
                             f"{state['pct_history_shallower']:.0f}% of history")
            # verdict band + omega
            labels = turbulence.vol_regime(price)
            reg_ent, today_regime = da.regime_conditioned_entries(
                price, state["current_dd"], labels)
            in_reg = (labels == today_regime).to_numpy()
            ff_full = da.conditional_further_fall(price, state["current_dd"])
            ff_reg = da.conditional_further_fall(price, state["current_dd"],
                                                 in_regime=in_reg)
            ff_head = ff_reg if ff_reg["n_complete"] >= da.REGIME_MIN_N else ff_full
            H = da.VERDICT_HORIZON
            use_v = da.forward_return_stats(tr, reg_ent, horizons=(H,))[H]["n"] >= da.REGIME_MIN_N
            v_ent = reg_ent if use_v else da.entry_index(price, state["current_dd"])
            verdict = da.dip_buy_verdict(
                da.forward_returns(tr, v_ent, H),
                da.forward_returns(tr, tr.index, H),
                depth_pctile=state["pct_history_shallower"],
                rr_percentile=da.reward_risk_depth_percentile(price, tr, state["current_dd"]),
                n_recovered_further_fall=ff_head["n_complete"])
            self.assertEqual(card["verdict"]["band"], verdict["band"])
            self.assertEqual(card["today_regime"], today_regime)
            if np.isfinite(verdict["omega"]):
                self.assertAlmostEqual(card["verdict"]["omega"], verdict["omega"],
                                       places=9)
            # forward table first data row matches a direct recompute
            fwd_full = da.forward_return_stats(tr, da.entry_index(price, state["current_dd"]),
                                               horizons=ds.HORIZONS)
            fwd_reg = da.forward_return_stats(tr, reg_ent, horizons=ds.HORIZONS)
            first_h = next(h for h in ds.HORIZONS if fwd_full[h]["n"] > 0)
            use = fwd_reg[first_h]["n"] >= da.REGIME_MIN_N
            src = fwd_reg[first_h] if use else fwd_full[first_h]
            exp_med = (ds._dual(src["median"], fwd_full[first_h]["median"], ds._pct)
                       if use else ds._pct(src["median"]))
            row0 = card["forward_table"]["rows"][0]
            self.assertEqual(row0["Typical Median Forward Return"], exp_med)


class TestEmptyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_no_dip_history(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        # data dir with no dip_history.csv
        frames2 = dataclasses.replace(self.frames, data_dir=str(tmp))
        view = ds.build_dip_view(frames2)
        self.assertEqual(view["cards"], [])
        self.assertIsNotNone(view["empty"])
        self.assertIn("No dip history", view["empty"]["message"])
        self.assertIsNone(view["turbulence"])

    def test_short_symbol_skipped(self):
        # A symbol with < MIN_HISTORY_DAYS rows must produce NO card. Build a temp
        # data dir = the fixture's real dip history + a synthetic short-history
        # symbol, then assert the short symbol is dropped while the real ones
        # still render (exercises the build_dip_view continue branch).
        import tempfile
        import dip_adhoc
        hist, divs = ds._load_dip_csvs(self.frames.data_dir)
        n_short = dip_adhoc.MIN_HISTORY_DAYS - 1
        short = pd.DataFrame({
            "symbol": ["SHRT"] * n_short,
            "date": pd.bdate_range("2023-01-02", periods=n_short),
            "close": np.linspace(100.0, 90.0, n_short),
            "adj_close": np.linspace(100.0, 90.0, n_short),
        })
        tmp = Path(tempfile.mkdtemp())
        pd.concat([hist, short], ignore_index=True).to_csv(
            tmp / "dip_history.csv", index=False)
        divs.to_csv(tmp / "dip_dividends.csv", index=False)
        frames2 = dataclasses.replace(self.frames, data_dir=str(tmp))

        view = ds.build_dip_view(frames2)
        syms = [c["symbol"] for c in view["cards"]]
        self.assertNotIn("SHRT", syms)              # the short symbol is skipped
        self.assertEqual(syms, ["SPY", "SCHD"])     # the real ones still render (display order)
        # And every rendered card cleared the floor.
        for c in view["cards"]:
            price, _, _ = dip_adhoc.slice_symbol(hist, divs, c["symbol"])
            self.assertGreaterEqual(len(price), dip_adhoc.MIN_HISTORY_DAYS)


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

    def test_dip_ok(self):
        r = self.client.get("/api/dip")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("cards", body)
        self.assertIn("turbulence", body)

    def test_unknown_params_ignored_not_500(self):
        r = self.client.get("/api/dip", params={"account": "nope"})
        self.assertEqual(r.status_code, 200)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/dip")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestCardDataSeam(unittest.TestCase):
    """Pin the card-orchestration seam against direct engine recompute (a few
    representative fields; the golden covers the full integration)."""

    @classmethod
    def setUpClass(cls):
        import dip_adhoc
        cls.frames = hs.load_frames(FIXTURE)
        hist, divs = ds._load_dip_csvs(cls.frames.data_dir)
        cls.price, cls.tr, cls.dser = dip_adhoc.slice_symbol(hist, divs, "SPY")
        cls.d = ds.dip_card_data("SPY", cls.price, cls.tr, cls.dser)

    def test_state_matches_engine(self):
        self.assertEqual(self.d.state, dip_analytics.drawdown_state(self.price))

    def test_verdict_band_valid(self):
        self.assertIn(self.d.verdict["band"],
                      ("strong", "neutral", "weak", "shallow", "inconclusive"))
        self.assertIn(self.d.today_regime, ("calm", "stressed"))

    def test_forward_full_matches_engine(self):
        ent = dip_analytics.entry_index(self.price, self.d.state["current_dd"])
        exp = dip_analytics.forward_return_stats(self.tr, ent, horizons=ds.HORIZONS)
        # NaN-aware (SPY's 100% hit-rate at 6mo/12mo -> cond_loss is NaN, and
        # nan != nan under assertEqual/_deep_close's math.isclose).
        np.testing.assert_equal(self.d.fwd_full, exp)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_dip_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = ds.build_dip_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        # Float-tolerant: the dip golden holds EVT/GPD + bootstrap-quantile
        # outputs, NOT bit-reproducible across LAPACK/scipy (the Factor #204
        # CI lesson). Structure + every formatted string stay pinned exactly.
        mismatch = _deep_close(view, expected)
        self.assertIsNone(mismatch, f"dip view diverges from golden at {mismatch}")


class TestLegendNoDrift(unittest.TestCase):
    """The served legend is `_LEGEND_HTML` (hand-converted to HTML so the vanilla
    terminal renders the `**bold**`/bullets Streamlit's st.markdown parses for
    free). `LEGEND_BODY` is kept as the canonical, copied-verbatim-from-app.py
    source of the wording — but nothing else ties the two together, so guard
    against silent drift: their visible WORDS must stay identical. Normalize each
    to plain text (strip markdown markers / HTML tags, both -> a space so block
    boundaries match, then collapse whitespace) and compare. If this fails, the
    HTML and the Streamlit-verbatim source have diverged — re-sync them."""

    @staticmethod
    def _from_md(s: str) -> str:
        import re
        s = s.replace("*", "")              # drop **bold** / *italic* markers
        s = re.sub(r"(?m)^- ", "", s)       # drop bullet markers (line-start "- ")
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _from_html(s: str) -> str:
        import re
        s = re.sub(r"<[^>]+>", " ", s)      # tags -> space (block boundaries)
        return re.sub(r"\s+", " ", s).strip()

    def test_html_words_match_source(self):
        self.assertEqual(self._from_md(ds.LEGEND_BODY),
                         self._from_html(ds._LEGEND_HTML),
                         "_LEGEND_HTML wording drifted from LEGEND_BODY")


class TestLookupService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    _CARD_KEYS = {"symbol", "today_regime", "regime_chip", "verdict", "kpis",
                  "bridge_text", "forward_table", "further_fall",
                  "time_underwater", "track_record", "underwater"}

    def test_ok_status_and_card(self):
        out = ds.build_dip_lookup(self.frames, "testq")  # lowercase -> normalized
        self.assertEqual(out["ticker"], "TESTQ")
        self.assertEqual(out["status"], "ok")
        self.assertIsNotNone(out["card"])
        self.assertEqual(set(out["card"].keys()), self._CARD_KEYS)
        self.assertEqual(out["card"]["symbol"], "TESTQ")
        json.dumps(out, allow_nan=False)  # raises if any NaN leaks

    def test_already_for_trio_symbol(self):
        out = ds.build_dip_lookup(self.frames, "SPY")
        self.assertEqual(out["status"], "already")
        self.assertIsNone(out["card"])
        self.assertIn("already shown", out["note"])

    def test_short(self):
        out = ds.build_dip_lookup(self.frames, "SHRT")
        self.assertEqual(out["status"], "short")
        self.assertGreater(out["n_days"], 0)
        self.assertIsNone(out["card"])

    def test_empty_unknown_symbol(self):
        out = ds.build_dip_lookup(self.frames, "ZZZZ")
        self.assertEqual(out["status"], "empty")
        self.assertIsNone(out["card"])

    def test_error_when_fetch_raises(self):
        def _raise(*a, **k):
            raise RuntimeError("network down")
        orig = ds._adhoc_fetchers
        ds._adhoc_fetchers = lambda data_dir: (_raise, _raise, False)
        try:
            out = ds.build_dip_lookup(self.frames, "NOCACHE")
        finally:
            ds._adhoc_fetchers = orig
        self.assertEqual(out["status"], "error")
        self.assertIsNone(out["card"])
        self.assertIn("network down", out["note"])

    def test_stale_returns_cached_card(self):
        import tempfile
        import dip_adhoc
        tmp = Path(tempfile.mkdtemp())
        hist, _divs = ds._load_dip_csvs(self.frames.data_dir)
        hist.to_csv(tmp / "dip_history.csv", index=False)  # vintage + watch source
        # A >=252-row TESTX sidecar dated OLDER than vintage: reuse real SPY rows
        # minus the last 30, relabeled, so resolve_adhoc sees it as stale and
        # tries the (raising) live fetch, then falls back to this cached copy.
        spy = hist[hist["symbol"] == "SPY"].sort_values("date")
        side = spy.iloc[:-30].copy()
        side["symbol"] = "TESTX"
        side.to_csv(tmp / "dip_adhoc_history.csv", index=False)
        frames2 = dataclasses.replace(self.frames, data_dir=str(tmp))

        def _raise(*a, **k):
            raise RuntimeError("live down")
        orig = ds._adhoc_fetchers
        ds._adhoc_fetchers = lambda data_dir: (_raise, _raise, False)
        try:
            out = ds.build_dip_lookup(frames2, "TESTX")
        finally:
            ds._adhoc_fetchers = orig
        self.assertEqual(out["status"], "stale")
        self.assertIsNotNone(out["card"])
        self.assertIn("cached", out["note"])


class TestLookupParity(unittest.TestCase):
    """The typed ok card == an independent _build_card recompute on the same
    offline-sliced TESTQ series — the '1:1 numbers' guarantee for the typed card.
    (No Streamlit AppTest cross-check: the typed card reuses the SAME _build_card
    the static-half TestParity already pins against _render_dip_card.)"""

    def test_ok_card_matches_build_card(self):
        import dip_adhoc
        frames = hs.load_frames(FIXTURE)
        out = ds.build_dip_lookup(frames, "TESTQ")
        self.assertEqual(out["status"], "ok")

        src = Path(frames.data_dir) / "dip_adhoc_source.csv"
        price_fn, div_fn = dip_adhoc.offline_fetchers(src)
        h_t, d_t = dip_adhoc.build_history(
            ["TESTQ"], price_fn, div_fn, pd.Timestamp.today().normalize())
        price, tr, dser = dip_adhoc.slice_symbol(h_t, d_t, "TESTQ")
        expected = ds._build_card("TESTQ", price, tr, dser)
        self.assertIsNone(_deep_close(out["card"], expected))


class TestResolveAdhocSeam(unittest.TestCase):
    def test_wires_fetchers_and_resolve(self):
        # resolve_adhoc_card goes through _adhoc_fetchers then resolve_adhoc,
        # returning a status-bearing dict. "SPY" is in the fixture's dip
        # universe so this resolves without a live fetch.
        frames = hs.load_frames(FIXTURE)
        out = ds.resolve_adhoc_card(frames.data_dir, "SPY", "2020-01-01")
        self.assertIn("status", out)


class TestLookupServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_lookup_ok(self):
        r = self.client.get("/api/dip/lookup", params={"ticker": "TESTQ"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_lookup_already(self):
        r = self.client.get("/api/dip/lookup", params={"ticker": "SPY"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "already")

    def test_lookup_short(self):
        r = self.client.get("/api/dip/lookup", params={"ticker": "SHRT"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "short")

    def test_lookup_empty(self):
        r = self.client.get("/api/dip/lookup", params={"ticker": "ZZZZ"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "empty")

    def test_lookup_invalid_input_422(self):
        for bad in ["../etc", "a;b", "TOOLONGTICKER", ""]:
            r = self.client.get("/api/dip/lookup", params={"ticker": bad})
            self.assertEqual(r.status_code, 422, f"{bad!r} should be 422")

    def test_lookup_missing_ticker_422(self):
        r = self.client.get("/api/dip/lookup")
        self.assertEqual(r.status_code, 422)

    def test_lookup_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/dip/lookup", params={"ticker": "TESTQ"})
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestLookupGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_dip_lookup_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        out = ds.build_dip_lookup(frames, "TESTQ")
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        # Float-tolerant (EVT/bootstrap not bit-reproducible across LAPACK — #204);
        # structure + formatted strings stay exact.
        self.assertIsNone(_deep_close(out, expected))


class TestHistoryDepthCaveatWiring(unittest.TestCase):
    """The fixture is 2y of history and both cards land 'strong' — exactly the
    thin-history edge claim the referee warns about, so the caveat must appear.
    The fixture can ONLY exercise the thin branch; the deep branch is covered by
    a constructed series (a degenerate fixture must not be the only witness)."""

    def test_fixture_card_discloses_thin_history(self):
        frames = hs.load_frames(FIXTURE)
        view = ds.build_dip_view(frames)
        cards = view["cards"]
        self.assertTrue(cards)
        for card in cards:
            with self.subTest(symbol=card["symbol"]):
                self.assertEqual(card["verdict"]["band"], "strong")
                self.assertIn("Thin history", card["verdict"]["text"])

    def test_deep_history_card_has_no_caveat(self):
        txt = ds._verdict_block(
            "SPY",
            {"band": "strong", "omega": 2.0, "baseline_omega": 1.2,
             "omega_ci": {"lo": 1.1, "hi": 3.0}, "n": 40},
            {"pct_history_shallower": 90.0},
            18.0)["text"]
        self.assertNotIn("Thin history", txt)

    def test_thin_history_non_edge_band_has_no_caveat(self):
        txt = ds._verdict_block(
            "SPY",
            {"band": "weak", "omega": 0.6, "baseline_omega": 1.2,
             "omega_ci": {"lo": 0.3, "hi": 0.9}, "n": 40},
            {"pct_history_shallower": 90.0},
            2.0)["text"]
        self.assertNotIn("Thin history", txt)


class TestRefereeBlock(unittest.TestCase):
    """SPY-only registered referee table (spec 2026-07-16 §5): shape, values
    recomputed from the committed artifact, and silent degradation when the
    artifact is unavailable."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = ds.build_dip_view(cls.frames)
        cls.cards = {c["symbol"]: c for c in cls.view["cards"]}

    def test_spy_card_carries_referee_block(self):
        ref = self.cards["SPY"].get("referee")
        self.assertIsNotNone(ref)
        self.assertEqual(set(ref.keys()), {"columns", "rows", "caption"})
        self.assertEqual(ref["columns"],
                         ["Verdict Band", "Days", "Episodes", "Median 12m",
                          "Hit 12m", "Omega 12m"])
        self.assertEqual(len(ref["rows"]), 7)   # 5 bands + all + TK rule
        self.assertIn("registered 2026-07-14", ref["caption"])
        self.assertIn("SPY only", ref["caption"])

    def test_rows_recompute_from_committed_artifact(self):
        from parsers import dip_backtest as db
        art = db.load_registered_artifact()
        ref = self.cards["SPY"]["referee"]
        strong = art["referee"]["strong"]
        self.assertEqual(ref["rows"][0]["Verdict Band"], "\U0001f7e2 Strong buy")
        self.assertEqual(ref["rows"][0]["Days"], f"{strong['n_days']:,}")
        self.assertEqual(ref["rows"][0]["Episodes"],
                         f"{strong['n_episodes']:,}")
        if strong.get("omega_252_inf"):
            self.assertEqual(ref["rows"][0]["Omega 12m"], "∞")
        alln = art["referee"]["all"]
        self.assertEqual(ref["rows"][5]["Verdict Band"], "— All days")
        self.assertEqual(ref["rows"][5]["Days"], f"{alln['n_days']:,}")
        self.assertTrue(ref["rows"][6]["Verdict Band"].startswith("★"))

    def test_non_spy_cards_carry_no_referee(self):
        for sym, c in self.cards.items():
            if sym != "SPY":
                self.assertNotIn("referee", c)

    def test_missing_artifact_degrades_silently(self):
        orig = ds._registered_artifact
        ds._registered_artifact = lambda: None
        try:
            view = ds.build_dip_view(hs.load_frames(FIXTURE))
            for c in view["cards"]:
                self.assertNotIn("referee", c)
        finally:
            ds._registered_artifact = orig

    def test_view_still_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
