# tests/test_terminal_ai.py
import dataclasses
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "terminal_ai_facts_golden.json"
PORTFOLIO_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                    / "terminal_ai_portfolio_facts_golden.json")

import _config
from terminal import ai_service as ai
from terminal import holdings_service as hs
from terminal import risksim_service as rss
from terminal import risk_service as rs
from terminal import tax_service as txs


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (returns None or first
    mismatch path). The factors facts carry lstsq (SVD) outputs, which drift
    a few ULPs across LAPACK builds (Windows dev vs Linux CI) — same doctrine
    as the factor golden: pin structure + formatted strings exactly, compare
    raw floats with tolerance."""
    import math
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


class TestFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_factors_facts_scrub_clean_and_golden(self):
        facts = ai.build_facts("factors", self.frames)
        ai.scrub_gate(facts)                      # never raises on real output
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_unknown_section_raises(self):
        with self.assertRaises(KeyError):
            ai.build_facts("nope", self.frames)

    def test_scope_labels_percent_only(self):
        facts = ai.build_facts("factors", self.frames)
        self.assertIn("scope", facts)
        self.assertIn("broker", facts["scope"])

    def test_history_start_threads_into_scope(self):
        # Review-2: Frames carries no history_start attribute — the request
        # value must be threaded explicitly or the scope block lies.
        frames = hs.apply_global_filters(hs.load_frames(str(FIXTURE)),
                                         ["all"], "2021+")
        facts = ai.build_facts("factors", frames, history_start="2021+")
        self.assertEqual(facts["scope"]["history_start"], "2021+")


import pandas as pd


class TestPortfolioFacts(unittest.TestCase):
    ASOF = date(2026, 6, 28)   # pin the today-dependent surface — the
                               # test_terminal_income.ASOF convention

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai._facts_portfolio(self.frames, "all", ["all"], None,
                                    asof=self.ASOF)
        ai.scrub_gate(facts)
        golden = json.loads(PORTFOLIO_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_asof_pins_income_and_tax_posture(self):
        # The seam must reach both sub-views. Tax: the TEST-A SPY lot
        # acquired 2025-11-03 turns long-term after 2026-11-03, moving
        # unrealized magnitude from the short to the long bucket. Income:
        # the fixture's forward model drifts once dividends age out of
        # its inference window (first on 2026-09-19: 1.09% -> 0.90%).
        # Without the thread both blocks follow date.today() and would
        # have flipped the golden from 2026-09-19 onward.
        pinned = ai._facts_portfolio(self.frames, "all", ["all"], None,
                                     asof=self.ASOF)
        later = ai._facts_portfolio(self.frames, "all", ["all"], None,
                                    asof=date(2026, 11, 10))
        self.assertGreater(later["tax_posture"]["long_share_pct"],
                           pinned["tax_posture"]["long_share_pct"])
        self.assertLess(later["tax_posture"]["short_share_pct"],
                        pinned["tax_posture"]["short_share_pct"])
        self.assertNotEqual(
            later["income"]["forward_yield_on_covered_mv"],
            pinned["income"]["forward_yield_on_covered_mv"])

    def test_windows_are_the_five(self):
        facts = ai.build_facts("portfolio", self.frames,
                               history_start="all", broker=["all"])
        self.assertEqual([w["window"] for w in facts["windows"]],
                         ["Full history", "5y", "3y", "1y", "YTD"])

    def test_scope_threads(self):
        f = hs.apply_global_filters(hs.load_frames(str(FIXTURE)),
                                    ["all"], "2026+")
        facts = ai.build_facts("portfolio", f, history_start="2026+",
                               broker=["all"])
        self.assertEqual(facts["scope"]["history_start"], "2026+")

    def test_factors_reducer_still_works_with_broker_kwarg(self):
        facts = ai.build_facts("factors", self.frames,
                               history_start="all", broker=["all"])
        self.assertEqual(facts["section"], "factors")


class TestPortfolioDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frames = hs.apply_global_filters(hs.load_frames(str(FIXTURE)),
                                         ["all"], "all")
        cls.facts = ai.build_facts("portfolio", frames,
                                   history_start="all", broker=["all"])
        cls.frames = frames

    def test_window_table_shape_and_strings(self):
        d = ai.portfolio_display(self.facts)
        t = d["window_table"]
        self.assertEqual(t["columns"][0], "Window")
        self.assertEqual(len(t["rows"]), 5)
        for row in t["rows"]:
            for col in t["columns"]:
                self.assertIsInstance(row[col], str)

    def test_tiles_are_label_value_strings(self):
        for tile in ai.portfolio_display(self.facts)["tiles"]:
            self.assertIsInstance(tile["label"], str)
            self.assertIsInstance(tile["value"], str)

    def test_meta_has_picker_vocab(self):
        meta = ai.portfolio_meta(self.frames)
        for k in ("accounts", "classes", "brokers", "available_dates",
                  "filter", "synthetic"):
            self.assertIn(k, meta)


class TestBundleBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_default_is_spy(self):
        b = rs._bundle(self.frames, None, None, False, False)
        b_spy = rs._bundle(self.frames, None, None, False, False,
                           benchmark="spy")
        # default == explicit spy on every key
        self.assertTrue(b["spy_monthly"].equals(b_spy["spy_monthly"]))
        self.assertTrue(b["spy_rets"].equals(b_spy["spy_rets"]))

    def test_blend_moves_bench_side_only(self):
        b_spy = rs._bundle(self.frames, None, None, False, False,
                           benchmark="spy")
        b_60 = rs._bundle(self.frames, None, None, False, False,
                          benchmark="60_40")
        # portfolio side is benchmark-independent
        self.assertTrue(b_spy["port_rets"].equals(b_60["port_rets"]))
        self.assertTrue(b_spy["monthly"].equals(b_60["monthly"]))
        # benchmark side changes (fixture AGG differs from flat SPY-TR)
        self.assertFalse(b_spy["spy_monthly"].equals(b_60["spy_monthly"]))


class TestWindowGrid(unittest.TestCase):
    def _mk(self, vals, start="2024-01-31"):
        idx = pd.date_range(start, periods=len(vals), freq="ME")
        return pd.Series(vals, index=idx, dtype=float)

    def test_slice_full_tail_ytd(self):
        m = self._mk([0.01] * 30)                      # 2024-01 … 2026-06
        self.assertEqual(len(ai._win_slice_monthly(m, None)), 30)
        self.assertEqual(len(ai._win_slice_monthly(m, 12)), 12)
        ytd = ai._win_slice_monthly(m, "ytd")
        self.assertTrue((ytd.index.year == 2026).all())
        self.assertEqual(len(ytd), 6)                  # 2026-01 … 2026-06

    def test_row_identical_series_beta_one_corr_one(self):
        m = self._mk([0.02, -0.01, 0.03, 0.01, -0.02, 0.015] * 4)
        d = pd.Series([0.001, -0.002, 0.0015] * 200,
                      index=pd.date_range("2024-01-01", periods=600, freq="B"))
        row = ai._window_row("Full history", m, m.copy(), d, d.copy(),
                             0.0, None)
        self.assertTrue(row["available"])
        self.assertAlmostEqual(row["beta"], 1.0, places=6)
        self.assertAlmostEqual(row["correlation"], 1.0, places=6)
        self.assertAlmostEqual(row["portfolio"]["twr_cum_pct"],
                               row["benchmark"]["twr_cum_pct"], places=9)

    def test_row_drawdown_is_percent_scale(self):
        # +100% then −50%: max drawdown must read −50.0 (percent), not −0.5
        m = self._mk([1.0, -0.5, 0.0, 0.0])
        d = pd.Series([0.0] * 40,
                      index=pd.date_range("2026-01-01", periods=40, freq="B"))
        row = ai._window_row("Full history", m, m.copy(), d, d.copy(), 0.0, None)
        self.assertAlmostEqual(row["portfolio"]["max_dd_pct"], -50.0, places=6)

    def test_row_unavailable_when_too_short(self):
        m = self._mk([0.01])
        row = ai._window_row("3y", m, m.copy(), pd.Series(dtype=float),
                             pd.Series(dtype=float), 0.0, 756)
        self.assertFalse(row["available"])
        self.assertEqual(row["n_months"], 1)
        self.assertNotIn("portfolio", row)

    def test_row_unavailable_when_window_underfilled(self):
        # Review-S2-1: a 3y row built from 7 months must NOT present itself
        # as available — the scope cannot fill the requested window.
        m = self._mk([0.01] * 7)
        d = pd.Series([0.001] * 120,
                      index=pd.date_range("2026-01-01", periods=120, freq="B"))
        row = ai._window_row("3y", m, m.copy(), d, d.copy(), 0.0, 756,
                             requested_months=36)
        self.assertFalse(row["available"])
        self.assertEqual(row["n_months"], 7)
        self.assertEqual(row["requested_months"], 36)

    def test_row_emits_daily_count_used(self):
        m = self._mk([0.01] * 24)
        d = pd.Series([0.001] * 130,
                      index=pd.date_range("2026-01-01", periods=130, freq="B"))
        row = ai._window_row("1y", m.tail(12), m.tail(12).copy(),
                             d, d.copy(), 0.0, 252, requested_months=12)
        self.assertTrue(row["available"])
        self.assertEqual(row["n_days_used"], 130)   # < the 252 requested

    def test_facts_windows_use_aligned_series(self):
        # Review-S2-2: the fixture's SPY monthly series is shorter than the
        # portfolio's; windows must be built on the INNER-ALIGNED pair so
        # portfolio and SPY stats always cover the same months.
        frames = hs.apply_global_filters(hs.load_frames(str(FIXTURE)),
                                         ["all"], "all")
        facts = ai.build_facts("portfolio", frames,
                               history_start="all", broker=["all"])
        full = facts["windows"][0]
        three = next(w for w in facts["windows"] if w["window"] == "3y")
        self.assertTrue(full["available"])
        # aligned span (27 months on the fixture) < 36 -> 3y is honest now
        self.assertFalse(three["available"])
        self.assertEqual(three["requested_months"], 36)


class TestCache(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.ddir = Path(self.td.name)
        (self.ddir / "positions.csv").write_text("a,b\n1,2\n")
        (self.ddir / "transactions.csv").write_text("c,d\n3,4\n")

    def tearDown(self):
        self.td.cleanup()

    def test_version_stable_then_invalidates_on_touch(self):
        v1 = ai.data_version(self.ddir)
        self.assertEqual(v1, ai.data_version(self.ddir))
        t = time.time() + 5
        os.utime(self.ddir / "positions.csv", (t, t))
        self.assertNotEqual(v1, ai.data_version(self.ddir))

    def test_cache_file_itself_never_invalidates(self):
        v1 = ai.data_version(self.ddir)
        ai.cache_put(self.ddir, "factors", "broker=all", v1, "hello")
        self.assertEqual(v1, ai.data_version(self.ddir))

    def test_put_get_roundtrip(self):
        v = ai.data_version(self.ddir)
        e = ai.cache_put(self.ddir, "factors", "broker=all", v, "narration")
        got = ai.cache_get(self.ddir, "factors", "broker=all")
        self.assertEqual(got["text"], "narration")
        self.assertEqual(got["data_version"], v)
        self.assertEqual(got["model"], ai.MODEL)
        self.assertEqual(e["text"], "narration")

    def test_corrupt_cache_is_empty(self):
        (self.ddir / "ai_cache.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(ai.cache_get(self.ddir, "factors", "broker=all"))

    def test_scope_key_canonical(self):
        self.assertEqual(ai.scope_key(["b", "a"], "2021+"),
                         ai.scope_key(["a", "b"], "2021+"))
        self.assertNotEqual(ai.scope_key(["all"], "all"),
                            ai.scope_key(["all"], "2021+"))

    def test_scope_key_comma_value_cannot_collide(self):
        # Review-3: a single value containing a comma must not produce the
        # same key as the two-value selection (cache-warmth-dependent 422).
        self.assertNotEqual(ai.scope_key(["alpine,harbor"], "all"),
                            ai.scope_key(["alpine", "harbor"], "all"))

    def test_missing_dir_raises_fnf(self):
        with self.assertRaises(FileNotFoundError):
            ai.data_version(self.ddir / "gone")


class TestCacheFmt(unittest.TestCase):
    """S1-v2: the narrative TEXT contract changed (structured JSON), so
    freshness = same data_version AND current fmt. Legacy entries (no fmt)
    regenerate once, lazily; scope keys never change."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.ddir = Path(td.name)
        (self.ddir / "positions.csv").write_text("a\n1\n")
        self.dv = ai.data_version(self.ddir)

    def _strip_fmt(self, key: str):
        p = self.ddir / "ai_cache.json"
        blob = json.loads(p.read_text(encoding="utf-8"))
        del blob[key]["fmt"]
        p.write_text(json.dumps(blob), encoding="utf-8")

    def test_put_stamps_current_fmt(self):
        e = ai.cache_put(self.ddir, "factors", "k", self.dv, "t")
        self.assertEqual(e["fmt"], ai.CACHE_FMT)

    def test_entry_fresh_requires_dv_and_fmt(self):
        e = ai.cache_put(self.ddir, "factors", "k", self.dv, "t")
        self.assertTrue(ai.entry_fresh(e, self.dv))
        legacy = {k: v for k, v in e.items() if k != "fmt"}
        self.assertFalse(ai.entry_fresh(legacy, self.dv))
        self.assertFalse(ai.entry_fresh(e, "other-dv"))
        self.assertFalse(ai.entry_fresh(None, self.dv))

    def test_legacy_entry_regenerates_not_hit(self):
        ai.cache_put(self.ddir, "factors", "k", self.dv, "old prose")
        self._strip_fmt("factors|k")       # simulate a pre-v2 entry
        c = FakeClient(_FakeMsg("new structured"))
        entry, hit = ai.generate_cached(self.ddir, "factors", "k", self.dv,
                                        {"section": "factors"}, client=c)
        self.assertFalse(hit)
        self.assertEqual(entry["text"], "new structured")
        self.assertEqual(entry["fmt"], ai.CACHE_FMT)


class TestChatPack(unittest.TestCase):
    """S2: the whole-book facts pack the chat answers from. Frames-derivable
    sections only; scrubbed as a whole; memoized by (dv, broker, history)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def setUp(self):
        ai._CHAT_PACK_MEMO.clear()
        ai._CHAT_PACK_BUILDING.clear()
        ai._CHAT_PACK_FAILED.clear()

    def test_pack_covers_the_thirteen_sections_and_scrubs(self):
        pack = ai.build_chat_pack(self.frames, "all", ["all"])
        self.assertEqual(sorted(pack), sorted(ai._CHAT_SECTIONS))
        self.assertEqual(sorted(ai._CHAT_SECTIONS),
                         sorted(["portfolio", "performance", "benchmark",
                                 "risk", "riskcontrib", "factors",
                                 "income", "tax", "dip",
                                 "tax_detail", "holdings_detail",
                                 "options", "health"]))
        self.assertEqual(pack["tax_detail"]["section"], "tax_detail")
        self.assertEqual(pack["holdings_detail"]["section"],
                         "holdings_detail")
        self.assertEqual(pack["options"]["section"], "options")
        self.assertEqual(pack["health"]["section"], "health")
        ai.scrub_gate(pack)          # must not raise (already scrubbed)

    def test_detail_sections_are_chat_only(self):
        for s in ("tax_detail", "holdings_detail", "options", "health"):
            self.assertNotIn(s, ai.SECTIONS)
            self.assertIn(s, ai._CHAT_DETAIL_REDUCERS)

    def test_chat_system_prompt_names_the_detail_sections(self):
        self.assertIn("FACTS.tax_detail", ai._CHAT_SYSTEM)
        self.assertIn("FACTS.holdings_detail", ai._CHAT_SYSTEM)
        self.assertIn("wash_if_sold_before", ai._CHAT_SYSTEM)
        self.assertIn("FACTS.options", ai._CHAT_SYSTEM)
        self.assertIn("FACTS.health", ai._CHAT_SYSTEM)

    def test_sig_sections_absent(self):
        pack = ai.build_chat_pack(self.frames, "all", ["all"])
        self.assertNotIn("frontier", pack)
        self.assertNotIn("risksim", pack)

    def test_memo_roundtrip_and_key_separation(self):
        self.assertIsNone(ai.chat_pack_get("dv1", "all", ["all"]))
        built = ai.chat_pack_build(self.frames, "dv1", "all", ["all"])
        got = ai.chat_pack_get("dv1", "all", ["all"])
        self.assertIs(got, built)
        self.assertIsNone(ai.chat_pack_get("dv2", "all", ["all"]))
        self.assertIsNone(ai.chat_pack_get("dv1", "2022", ["all"]))
        self.assertIsNone(ai.chat_pack_get("dv1", "all", ["alpine"]))

    def test_memo_broker_order_canonical(self):
        ai.chat_pack_build(self.frames, "dv1", "all", ["b", "a"])
        self.assertIsNotNone(ai.chat_pack_get("dv1", "all", ["a", "b"]))

    def test_memo_bounded(self):
        for i in range(ai._CHAT_PACK_MAX + 2):
            ai.chat_pack_build(self.frames, f"dv{i}", "all", ["all"])
        self.assertLessEqual(len(ai._CHAT_PACK_MEMO), ai._CHAT_PACK_MAX)

    def test_scrub_failure_propagates_and_never_stored(self):
        with mock.patch.object(ai, "scrub_gate",
                               side_effect=ai.AIScrubError("dirty")):
            with self.assertRaises(ai.AIScrubError):
                ai.chat_pack_build(self.frames, "dvX", "all", ["all"])
        self.assertIsNone(ai.chat_pack_get("dvX", "all", ["all"]))

    def test_reducer_failure_propagates_and_never_stored(self):
        with mock.patch.object(ai, "build_facts",
                               side_effect=RuntimeError("reducer boom")):
            with self.assertRaises(RuntimeError):
                ai.chat_pack_build(self.frames, "dvY", "all", ["all"])
        self.assertIsNone(ai.chat_pack_get("dvY", "all", ["all"]))

    # --- pack pre-warm seam (2026-08-22) --------------------------------
    def test_state_missing_then_ready(self):
        self.assertEqual(ai.chat_pack_state("dvS", "all", ["all"]), "missing")
        ai.chat_pack_build(self.frames, "dvS", "all", ["all"])
        self.assertEqual(ai.chat_pack_state("dvS", "all", ["all"]), "ready")

    def test_ensure_hit_never_calls_frames_fn(self):
        built = ai.chat_pack_build(self.frames, "dvE", "all", ["all"])
        def boom():
            raise AssertionError("frames_fn on a memo hit")
        self.assertIs(ai.chat_pack_ensure("dvE", "all", ["all"], boom), built)

    def test_ensure_miss_builds_once_and_memoizes(self):
        calls = []
        def frames_fn():
            calls.append(1)
            return self.frames
        p1 = ai.chat_pack_ensure("dvM", "all", ["all"], frames_fn)
        p2 = ai.chat_pack_ensure("dvM", "all", ["all"], frames_fn)
        self.assertIs(p1, p2)
        self.assertEqual(calls, [1])
        self.assertIs(ai.chat_pack_get("dvM", "all", ["all"]), p1)

    def test_ensure_failure_releases_lock_and_next_call_rebuilds(self):
        with mock.patch.object(ai, "build_chat_pack",
                               side_effect=RuntimeError("reducer boom")):
            with self.assertRaises(RuntimeError):
                ai.chat_pack_ensure("dvF", "all", ["all"], lambda: self.frames)
        self.assertEqual(ai.chat_pack_state("dvF", "all", ["all"]), "missing")
        p = ai.chat_pack_ensure("dvF", "all", ["all"], lambda: self.frames)
        self.assertEqual(sorted(p), sorted(ai._CHAT_SECTIONS))
        self.assertEqual(ai.chat_pack_state("dvF", "all", ["all"]), "ready")

    def test_ensure_concurrent_caller_waits_and_shares_one_build(self):
        import threading
        gate, started = threading.Event(), threading.Event()
        real = ai.build_chat_pack
        n = []
        def slow(frames, history_start, broker):
            n.append(1); started.set(); gate.wait(5)
            return real(frames, history_start, broker)
        out = {}
        with mock.patch.object(ai, "build_chat_pack", side_effect=slow):
            t1 = threading.Thread(
                target=lambda: out.setdefault("a", ai.chat_pack_ensure(
                    "dvC", "all", ["all"], lambda: self.frames)))
            t1.start()
            self.assertTrue(started.wait(5))
            self.assertEqual(ai.chat_pack_state("dvC", "all", ["all"]), "building")
            t2 = threading.Thread(
                target=lambda: out.setdefault("b", ai.chat_pack_ensure(
                    "dvC", "all", ["all"], lambda: self.frames)))
            t2.start(); t2.join(0.3)
            self.assertTrue(t2.is_alive())          # blocked on the in-flight build
            gate.set(); t1.join(30); t2.join(30)
        self.assertIs(out["a"], out["b"])
        self.assertEqual(n, [1])                    # exactly one build
        self.assertEqual(ai.chat_pack_state("dvC", "all", ["all"]), "ready")

    def test_warm_swallows_and_logs_build_failure_and_marks_failed(self):
        with mock.patch.object(ai, "build_chat_pack",
                               side_effect=RuntimeError("reducer boom")):
            with self.assertLogs("terminal.ai_service", level="WARNING") as cm:
                ai.chat_pack_warm("dvW", "all", ["all"], lambda: self.frames)
        self.assertIn("reducer boom", "\n".join(cm.output))
        self.assertEqual(ai.chat_pack_state("dvW", "all", ["all"]), "failed")
        # the chat path still rebuilds, and a success clears the mark
        p = ai.chat_pack_ensure("dvW", "all", ["all"], lambda: self.frames)
        self.assertEqual(sorted(p), sorted(ai._CHAT_SECTIONS))
        self.assertEqual(ai.chat_pack_state("dvW", "all", ["all"]), "ready")

    def test_warm_builds_on_miss(self):
        ai.chat_pack_warm("dvW2", "all", ["all"], lambda: self.frames)
        self.assertEqual(ai.chat_pack_state("dvW2", "all", ["all"]), "ready")

    def test_memo_hit_refreshes_recency(self):
        # LRU: the scope being chatted in (hit) outlives passively warmed ones.
        ai.chat_pack_build(self.frames, "dvL0", "all", ["all"])
        for i in range(1, ai._CHAT_PACK_MAX):
            ai.chat_pack_build(self.frames, f"dvL{i}", "all", ["all"])
        self.assertIsNotNone(ai.chat_pack_get("dvL0", "all", ["all"]))  # touch
        ai.chat_pack_build(self.frames, "dvLx", "all", ["all"])           # evicts ONE
        self.assertIsNotNone(ai.chat_pack_get("dvL0", "all", ["all"]))
        self.assertIsNone(ai.chat_pack_get("dvL1", "all", ["all"]))

    def test_stale_dv_locks_pruned_after_build(self):
        ai.chat_pack_ensure("dvOld", "all", ["all"], lambda: self.frames)
        self.assertIn(ai._chat_pack_key("dvOld", "all", ["all"]),
                      ai._CHAT_PACK_BUILDING)
        ai.chat_pack_ensure("dvNew", "all", ["all"], lambda: self.frames)
        self.assertNotIn(ai._chat_pack_key("dvOld", "all", ["all"]),
                         ai._CHAT_PACK_BUILDING)
        self.assertIn(ai._chat_pack_key("dvNew", "all", ["all"]),
                      ai._CHAT_PACK_BUILDING)


_S1_ASOF = date(2026, 6, 28)
_S1_NOW = pd.Timestamp("2026-06-28T12:00:00+00:00")


class TestOptionsFacts(unittest.TestCase):
    """Full-gate S1: chat-only options section (spec 2026-08-23 §4.1)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def _facts(self, **kw):
        kw.setdefault("asof", _S1_ASOF)
        kw.setdefault("now", _S1_NOW)
        return ai._facts_options(self.frames, "all", ["all"], None, **kw)

    def test_structure_and_fixture_put(self):
        f = self._facts()
        self.assertEqual(f["section"], "options")
        self.assertEqual(f["scope"]["history_start"], "all")
        self.assertTrue(f["available"])
        self.assertFalse(f["empty"])
        rows = f["contracts"]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["underlying"], "SPY")
        self.assertEqual(r["opt_type"], "put")
        self.assertEqual(r["side"], "long")
        self.assertEqual(r["contracts"], 2)
        self.assertEqual(r["expiry"], "2026-09-18")
        self.assertEqual(r["dte"], 82)          # 2026-06-28 -> 2026-09-18
        # strike 500 vs spot 570 -> (500/570 - 1) * 100 = -12.3
        self.assertAlmostEqual(r["strike_vs_spot_pct"], -12.3, places=1)

    def test_aggregates_pct_only(self):
        agg = self._facts()["aggregates"]
        for k in ("notional_coverage_pct", "premium_at_risk_pct",
                  "pnl_on_cost_pct", "weighted_dte", "weighted_iv_pct",
                  "n_live", "n_excluded", "greeks_missing"):
            self.assertIn(k, agg)
        self.assertEqual(agg["n_live"], 1)
        # fixture: premium_mid 12.50 x 2 x 100 = 2500 at risk on 3000 cost
        self.assertAlmostEqual(agg["pnl_on_cost_pct"], -16.7, places=1)
        # weighted_iv 0.18 -> 18.0
        self.assertAlmostEqual(agg["weighted_iv_pct"], 18.0, places=1)

    def test_iv_and_staleness_blocks(self):
        f = self._facts()
        self.assertIn("last_percentile", f["iv_percentile"])
        self.assertIn("window_days", f["iv_percentile"])
        self.assertNotIn("series", f["iv_percentile"])
        for k in ("snapshot", "atm_iv"):
            self.assertIn("chip", f["staleness"][k])

    def test_asof_threads_liveness(self):
        # #367 idiom: a later asof past the 2026-09-18 expiry must flip the
        # book to the all-closed empty state — proves `today` reaches both
        # the view and the live-mask.
        past = self._facts(asof=date(2026, 11, 4))
        self.assertTrue(past["empty"])
        self.assertIn("closed or expired", past["empty_message"])

    def test_scrubs_and_json_safe(self):
        f = self._facts()
        ai.scrub_gate(f)                       # must not raise
        json.dumps(f, allow_nan=False)         # no NaN/Inf leaks


class TestHealthFacts(unittest.TestCase):
    """Full-gate S1: chat-only health section (spec 2026-08-23 §4.2 +
    Update #1 — structured fields, no headline text)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def _facts(self, **kw):
        kw.setdefault("asof", _S1_ASOF)
        return ai._facts_health(self.frames, "all", ["all"], None, **kw)

    def test_structure(self):
        f = self._facts()
        self.assertEqual(f["section"], "health")
        self.assertTrue(f["available"])
        self.assertIn(f["headline_level"],
                      ("green", "amber", "red", "grey"))
        self.assertTrue(f["recon_available"])
        for k in ("n_ok", "n_known", "n_watch", "n_error", "n_carried",
                  "worst_level"):
            self.assertIn(k, f["summary"])
        self.assertGreaterEqual(len(f["accounts"]), 1)
        row = f["accounts"][0]
        for k in ("account", "broker", "state", "verdict"):
            self.assertIn(k, row)
        # dollar columns must NOT survive the compaction
        for banned in ("extracted", "reported", "diff_usd"):
            self.assertNotIn(banned, row)

    def test_calendar_stable(self):
        # `today` feeds only days_since (unrendered, un-emitted) — the
        # section must be byte-identical across distant asof values.
        self.assertEqual(self._facts(),
                         self._facts(asof=date(2027, 3, 1)))

    def test_scrubs_and_json_safe(self):
        f = self._facts()
        ai.scrub_gate(f)
        json.dumps(f, allow_nan=False)

    def test_scrub_guards_unlabeled_digit_run_account(self):
        # Finding-1 guard: build_health_report falls back to the RAW
        # account id (label_by_account.get(acct, acct)) when ACCOUNT_DISPLAY
        # has no entry for it — a 5+ digit run that would otherwise trip
        # _TICKER_DIGIT_RUN_RE and fail the whole chat pack closed on every
        # scope. _facts_health must route a.label through the same
        # _scrub_safe_label guard _facts_holdings_detail/_facts_tax_detail
        # already use.
        from data_health import AccountHealth, HealthReport
        from terminal import health_service as hlth
        bad = AccountHealth(
            account_id="87654321", label="87654321", broker="alpine",
            state="verified", lagging=False, band="ok",
            extracted=1000.0, reported=1000.0, diff_usd=0.0, diff_pct=0.0,
            last_verified_month="2026-07", days_since=20)
        report = HealthReport(
            as_of_month="2026-07", recon_available=True, accounts=[bad],
            n_ok=1, n_known=0, n_watch=0, n_error=0, n_carried=0,
            worst_level="green")
        with mock.patch.object(hlth, "build_health_report_for_frames",
                               return_value=report):
            f = self._facts()
        self.assertEqual(f["accounts"][0]["account"], "unlabeled account")
        ai.scrub_gate(f)          # would raise AIScrubError pre-fix


class TestDetailTopics(unittest.TestCase):
    """Full-gate S2: fetch_detail topic reducers — riskcontrib + income
    (spec §5.2) and the run_detail dispatch contract."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_run_detail_unknown_topic_is_structured_error(self):
        out = ai.run_detail(self.frames, "all", ["all"], "nope", None,
                            asof=_S1_ASOF)
        self.assertIn("error", out)
        self.assertIn("nope", out["error"])

    def test_riskcontrib_full_table_and_filter(self):
        out = ai.run_detail(self.frames, "all", ["all"], "riskcontrib",
                            None, asof=_S1_ASOF)
        self.assertEqual(out["topic"], "riskcontrib")
        self.assertTrue(out["available"])
        self.assertEqual(len(out["rows"]), out["names_total"])
        self.assertGreaterEqual(out["names_total"], 1)
        tickers = [r["ticker"] for r in out["rows"]]
        self.assertIn("SPY", tickers)
        for r in out["rows"]:
            self.assertIn("weight_pct", r)
            self.assertIn("risk_share_pct", r)
        one = ai.run_detail(self.frames, "all", ["all"], "riskcontrib",
                            "spy", asof=_S1_ASOF)
        self.assertEqual([r["ticker"] for r in one["rows"]], ["SPY"])
        # *_total keys are the PRE-filter table size, not the row count —
        # the rename rider's whole point (was n_names, misreadable as
        # len(rows) on a ticker-filtered result).
        self.assertEqual(one["names_total"], out["names_total"])
        miss = ai.run_detail(self.frames, "all", ["all"], "riskcontrib",
                             "ZZZQ", asof=_S1_ASOF)
        self.assertIs(miss["found"], False)

    def test_income_payers_rows(self):
        out = ai.run_detail(self.frames, "all", ["all"], "income", None,
                            asof=_S1_ASOF)
        self.assertEqual(out["topic"], "income")
        self.assertTrue(out["available"])
        self.assertGreaterEqual(out["payers_total"], 1)
        shares = [r["share_of_forward_pct"] for r in out["rows"]
                  if r["share_of_forward_pct"] is not None]
        self.assertLess(abs(sum(shares) - 100.0), 1.0)
        for r in out["rows"]:
            for k in ("symbol", "weight_pct", "yield_pct", "yoc_pct"):
                self.assertIn(k, r)
        top = out["rows"][0]["symbol"]
        one = ai.run_detail(self.frames, "all", ["all"], "income",
                            top.lower(), asof=_S1_ASOF)
        self.assertEqual([r["symbol"] for r in one["rows"]], [top])
        self.assertEqual(one["payers_total"], out["payers_total"])
        miss = ai.run_detail(self.frames, "all", ["all"], "income",
                             "ZZZQ", asof=_S1_ASOF)
        self.assertIs(miss["found"], False)
        self.assertEqual(miss["payers_total"], out["payers_total"])

    def test_results_scrub_and_json_safe(self):
        for topic, tick in (("riskcontrib", None), ("income", None)):
            out = ai.run_detail(self.frames, "all", ["all"], topic, tick,
                                asof=_S1_ASOF)
            ai.scrub_gate(out)      # run_detail already gates; must be idempotent
            json.dumps(out, allow_nan=False)

    def test_ticker_shape_precheck_rejects_cusip_and_dollar(self):
        # A model-supplied cusip or dollar-shaped "ticker" must be
        # rejected before it ever reaches a reducer (and could echo
        # into the payload and trip scrub_gate, killing the whole turn
        # after paying full reducer cost) — two different topics, no
        # exception either way.
        cusip = ai.run_detail(self.frames, "all", ["all"], "lots",
                              "912828XG8", asof=_S1_ASOF)
        self.assertEqual(cusip, {"error": "not a ticker-shaped symbol"})
        dollar = ai.run_detail(self.frames, "all", ["all"], "income",
                               "$50000", asof=_S1_ASOF)
        self.assertEqual(dollar, {"error": "not a ticker-shaped symbol"})


class TestDetailLots(unittest.TestCase):
    """Full-gate S2: lots topic — every open lot for one ticker."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_spy_lots_rows(self):
        out = ai.run_detail(self.frames, "all", ["all"], "lots", "SPY",
                            asof=_S1_ASOF)
        self.assertEqual(out["topic"], "lots")
        self.assertTrue(out["available"])
        self.assertTrue(out["found"])
        self.assertGreaterEqual(out["n_lots"], 1)
        r = out["rows"][0]
        for k in ("account", "acquired_date", "term",
                  "days_to_long_term", "unrealized_pct"):
            self.assertIn(k, r)
        self.assertNotIn("wash_status", r)
        # the fixture's known lot: TEST-A SPY acquired 2025-11-03
        self.assertIn("2025-11-03", [x["acquired_date"] for x in out["rows"]])
        for x in out["rows"]:
            self.assertNotIn("quantity", x)
        self.assertIn("taxable accounts only", out["note"])
        self.assertIn("IRA lots are excluded", out["note"])

    def test_asof_flips_term(self):
        # #367 idiom: the 2025-11-03 lot is short at the pin, long a year on
        short = ai.run_detail(self.frames, "all", ["all"], "lots", "SPY",
                              asof=_S1_ASOF)
        longr = ai.run_detail(self.frames, "all", ["all"], "lots", "SPY",
                              asof=date(2026, 12, 1))
        def term_of(res):
            return {x["acquired_date"]: x["term"] for x in res["rows"]}
        self.assertEqual(term_of(short).get("2025-11-03"), "short")
        self.assertEqual(term_of(longr).get("2025-11-03"), "long")

    def test_unknown_ticker_found_false(self):
        out = ai.run_detail(self.frames, "all", ["all"], "lots", "ZZZQ",
                            asof=_S1_ASOF)
        self.assertIs(out["found"], False)
        # IRA-caveat rider: the view is taxable-only, so "no open lots"
        # must not read as "not held" — the name may sit in an IRA.
        self.assertIn("held only in an IRA", out["note"])

    def test_found_false_carries_stale_and_note(self):
        # On today's real book the ledger IS stale; a "no open lots"
        # answer for a freshly-bought name must carry that caveat.
        with mock.patch.object(
                txs, "build_tax_view",
                return_value={"kind": "tax", "meta": {"stale": True},
                             "lots": []}):
            out = ai.run_detail(self.frames, "all", ["all"], "lots",
                                "ZZZQ", asof=_S1_ASOF)
        self.assertIs(out["found"], False)
        self.assertIs(out["stale"], True)
        self.assertIn("ledger may lag recent buys", out["note"])
        self.assertIn("held only in an IRA", out["note"])

    def test_missing_required_ticker_is_structured_error(self):
        # lives in Task 2, not Task 1: "lots" only enters _DETAIL_TOPICS
        # here, and run_detail checks unknown-topic before needs-ticker.
        out = ai.run_detail(self.frames, "all", ["all"], "lots", None,
                            asof=_S1_ASOF)
        self.assertEqual(out, {"error": "ticker required for this topic"})
        blank = ai.run_detail(self.frames, "all", ["all"], "lots", "  ",
                              asof=_S1_ASOF)
        self.assertEqual(blank, {"error": "ticker required for this topic"})

    def test_scrubs_and_json_safe(self):
        out = ai.run_detail(self.frames, "all", ["all"], "lots", "SPY",
                            asof=_S1_ASOF)
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)


class TestDetailTransactions(unittest.TestCase):
    """Full-gate S2: transactions topic — one ticker's wash-window rows
    (dates/types/accounts, no quantities)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_structure_on_fixture(self):
        out = ai.run_detail(self.frames, "all", ["all"], "transactions",
                            "SPY", asof=_S1_ASOF)
        self.assertEqual(out["topic"], "transactions")
        self.assertTrue(out["available"])
        self.assertEqual(out["window_days"], txs.WINDOW_DAYS)
        for r in out.get("rows", []):
            for k in ("date", "type", "account"):
                self.assertIn(k, r)
            self.assertNotIn("quantity", r)
            self.assertNotIn("qty", r)

    def test_injected_window_row_appears(self):
        # Deterministic end-to-end proof on a temp copy of the fixture:
        # append one in-window buy for a fresh ticker and see the row.
        import shutil as _sh
        with tempfile.TemporaryDirectory() as td:
            _sh.copytree(FIXTURE, td, dirs_exist_ok=True)
            with open(Path(td) / "transactions.csv", "a",
                      encoding="utf-8", newline="") as f:
                f.write("2026-06-10,2026-06-10,alpine,TEST-A,buy,ZZT,,"
                        "ZZ TEST CO,5,10.00,-50.00,synth,,\n")
            frames = hs.load_frames(td)
            out = ai.run_detail(frames, "all", ["all"], "transactions",
                                "ZZT", asof=_S1_ASOF)
        self.assertTrue(out["found"])
        rows = out["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["type"], "buy")
        self.assertEqual(rows[0]["date"], "2026-06-10")
        self.assertNotIn("quantity", rows[0])
        self.assertIn("account", rows[0])
        self.assertNotRegex(rows[0]["account"], r"\d{5,}")
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)


class TestPct100Helper(unittest.TestCase):
    """Rider (#383 ledger): _pct100 promoted from three identical
    function-local copies to one module-level helper (decimal ->
    rounded percent, None/NaN-safe via _num)."""

    def test_contract(self):
        self.assertIsNone(ai._pct100(None))
        self.assertEqual(ai._pct100(0.12345), 12.35)
        self.assertEqual(ai._pct100(0.1234, 1), 12.3)
        self.assertIsNone(ai._pct100(float("nan")))


class TestDetailMemoKeyNormalization(unittest.TestCase):
    """Rider (#382 ledger): the per-turn detail memo keyed on the RAW
    model-supplied ticker, so "strl" then "STRL" double-paid a ~20 s
    reducer on the real book. The memo key now goes through
    ai.normalize_detail_ticker — the same rule run_detail applies
    internally before dispatch."""

    def test_normalize_matches_run_detail_rule(self):
        self.assertEqual(ai.normalize_detail_ticker(" strl "), "STRL")
        self.assertIsNone(ai.normalize_detail_ticker(None))
        self.assertIsNone(ai.normalize_detail_ticker("   "))
        self.assertIsNone(ai.normalize_detail_ticker(""))

    def test_case_variants_hit_one_reducer_call(self):
        from terminal import server
        calls = []

        def fake_run_detail(frames, history_start, broker, topic, ticker,
                            **kw):
            calls.append((topic, ticker))
            return {"topic": topic, "ok": True}

        with mock.patch.object(server.hs, "load_frames",
                               return_value="FRAMES"), \
             mock.patch.object(server.hs, "apply_global_filters",
                               return_value="SCOPED"), \
             mock.patch.object(ai, "run_detail",
                               side_effect=fake_run_detail):
            fn = server._chat_detail_fn("ddir", ["all"], "all")
            a = fn("lots", "strl")
            b = fn("lots", "STRL")
            c = fn("lots", " Strl ")
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual(len(calls), 1)


class TestDetailRest(unittest.TestCase):
    """Full-gate S3: the last five fetch_detail topics (spec §6)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def _run(self, topic, ticker=None):
        return ai.run_detail(self.frames, "all", ["all"], topic, ticker,
                             asof=_S1_ASOF)

    def test_performance_monthly_yearly_accounts(self):
        out = self._run("performance")
        self.assertEqual(out["topic"], "performance")
        self.assertTrue(out["available"])
        self.assertGreaterEqual(len(out["monthly"]), 3)
        m = out["monthly"][0]
        self.assertIn("month", m)
        self.assertIn("return_pct", m)
        self.assertGreaterEqual(len(out["by_year"]), 1)
        y = out["by_year"][0]
        self.assertIn("year", y)
        self.assertIn("return_pct", y)
        self.assertGreaterEqual(len(out["accounts"]), 1)
        a = out["accounts"][0]
        for k in ("account", "cum_twr_pct", "months"):
            self.assertIn(k, a)
        for banned in ("start_nav", "end_nav", "net_flow"):
            self.assertNotIn(banned, a)
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)

    def test_performance_accounts_scrubs_digit_run_label(self):
        # Finding-1 guard: per_account_raw falls back to the RAW account id
        # (ACCOUNT_DISPLAY.get(acct, acct)) when the display map has no
        # entry for it — a 5+ digit run would otherwise trip
        # _TICKER_DIGIT_RUN_RE and fail the whole chat pack closed on every
        # performance-fetch turn. The accounts row must route
        # account_label through the same _scrub_safe_label guard
        # _detail_lots/_detail_health already use.
        from terminal import performance_service as ps
        bad = pd.DataFrame([{"account_label": "87654321",
                             "cum_twr": 0.1, "months": 5}])
        with mock.patch.object(ps, "per_account_raw", return_value=bad):
            out = self._run("performance")
        self.assertEqual(out["accounts"][0]["account"], "unlabeled account")
        ai.scrub_gate(out)          # would raise AIScrubError pre-fix
        json.dumps(out, allow_nan=False)

    def test_dip_referee_and_skips(self):
        out = self._run("dip")
        self.assertEqual(out["topic"], "dip")
        self.assertTrue(out["available"])
        ref = out["referee"]
        self.assertIsNotNone(ref)
        self.assertEqual(ref["columns"][0], "Verdict Band")
        self.assertGreaterEqual(len(ref["rows"]), 7)
        self.assertIn("walk-forward referee", ref["caption"])
        self.assertIn("skipped_symbols", out)
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)

    def test_by_year_drops_nan_inception_month(self):
        # compute_twr deterministically seeds return_pct=NaN for the
        # portfolio's first tracked month (no prior NAV). Blank the FIRST
        # data row's return_pct (row kept) on a temp copy of the fixture
        # and prove the inception year still reports a real return_pct,
        # compounded over the surviving months only (review catch: the
        # original loop let one NaN poison the whole calendar year).
        import csv as _csv
        import shutil as _sh
        with tempfile.TemporaryDirectory() as td:
            _sh.copytree(FIXTURE, td, dirs_exist_ok=True)
            path = Path(td) / "twr_portfolio.csv"
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(_csv.reader(f))
            header, data = rows[0], rows[1:]
            ret_idx = header.index("return_pct")
            month_idx = header.index("month")
            year = data[0][month_idx].split("-")[0]
            same_year = sum(1 for r in data
                            if r[month_idx].split("-")[0] == year)
            data[0][ret_idx] = ""    # blank the value, keep the row
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(header)
                w.writerows(data)
            frames = hs.load_frames(td)
            out = ai.run_detail(frames, "all", ["all"], "performance", None,
                                asof=_S1_ASOF)
        by_year = {y["year"]: y for y in out["by_year"]}
        self.assertIn(int(year), by_year)
        entry = by_year[int(year)]
        self.assertIsNotNone(entry["return_pct"])
        self.assertEqual(entry["months"], same_year - 1)
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)

    def test_factor_blocks(self):
        out = self._run("factor")
        self.assertEqual(out["topic"], "factor")
        self.assertTrue(out["available"])
        self.assertGreaterEqual(len(out["windows"]), 1)
        w = out["windows"][0]
        self.assertIn("window", w)
        self.assertGreaterEqual(len(w["models"]), 1)
        m = w["models"][0]
        for k in ("model", "n", "r2", "adj_r2", "betas"):
            self.assertIn(k, m)
        if m["betas"]:
            b = m["betas"][0]
            for k in ("factor", "beta", "se", "t"):
                self.assertIn(k, b)
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)

    def test_options_contracts_fixture_put(self):
        out = self._run("options_contracts")
        self.assertEqual(out["topic"], "options_contracts")
        self.assertTrue(out["available"])
        self.assertFalse(out["empty"])
        rows = out["contracts"]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["underlying"], "SPY")
        self.assertEqual(r["dte"], 82)
        self.assertAlmostEqual(r["pnl_on_cost_pct"], -16.7, places=1)
        self.assertEqual(out["live_total"], len(rows))
        filt = self._run("options_contracts", "ZZZQ")
        self.assertIs(filt["found"], False)
        # live_total is the PRE-filter live-contract count (rename rider)
        self.assertEqual(filt["live_total"], out["live_total"])
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)

    def test_options_contracts_asof_threads(self):
        past = ai.run_detail(self.frames, "all", ["all"],
                             "options_contracts", None,
                             asof=date(2026, 11, 4))
        self.assertTrue(past["empty"])

    def test_health_rows_with_lagging(self):
        out = self._run("health")
        self.assertEqual(out["topic"], "health")
        self.assertTrue(out["available"])
        self.assertIn(out["headline_level"],
                      ("green", "amber", "red", "grey"))
        self.assertGreaterEqual(len(out["accounts"]), 1)
        row = out["accounts"][0]
        for k in ("account", "broker", "state", "verdict", "lagging"):
            self.assertIn(k, row)
        for banned in ("extracted", "reported", "diff_usd"):
            self.assertNotIn(banned, row)
        ai.scrub_gate(out)
        json.dumps(out, allow_nan=False)

    def test_registry_and_enum_agree(self):
        self.assertEqual(
            sorted(ai._DETAIL_TOPICS),
            sorted(ai._FETCH_DETAIL_TOOL["input_schema"]["properties"]
                   ["topic"]["enum"]))


class TestScrub(unittest.TestCase):
    """scrub_gate: the outbound-payload privacy boundary. Fail closed on
    dollar-bearing keys, $-formatted strings, and 5+-digit runs (the account
    -mask shape); pass clean percent/ratio payloads through unchanged."""

    def test_clean_payload_passes_unchanged(self):
        p = {"alpha_annual": "+4.2%", "betas": [{"factor": "Mkt-RF",
             "beta": 1.02, "se": 0.05, "t": 20.4}],
             "stats": "t = 1.23 · R² = 0.87 · 95% CI +1.2% … +9.8%",
             "window": "Full history", "months": "74"}
        self.assertIs(ai.scrub_gate(p), p)

    def test_dollar_key_raises(self):
        # Includes this repo's REAL payload key names (tax_service lots rows) —
        # the review-1 gap: `basis\b` can't match basis_remaining because `_`
        # is a word character.
        for key in ("market_value", "cost_basis", "nav", "amount_usd",
                    "proceeds", "balance", "basis_remaining", "unrealized_gl",
                    "realized_gl", "value", "cost", "price", "total_gain",
                    "loss_amt", "equity"):
            with self.assertRaises(ai.AIScrubError, msg=key):
                ai.scrub_gate({"outer": [{key: 1.0}]})

    def test_percent_suffixed_keys_pass(self):
        # The percent family is legal by construction — the deny tokens are
        # exempted when the key names a percent/ratio unit.
        ai.scrub_gate({"gains_pct": 4.2, "loss_pct": -1.1,
                       "value_pp": 0.7, "cost_ratio": 0.01})

    def test_dollar_string_raises(self):
        with self.assertRaises(ai.AIScrubError):
            ai.scrub_gate({"note": "up $1,234 this month"})

    def test_digit_run_raises(self):
        with self.assertRaises(ai.AIScrubError):
            ai.scrub_gate({"label": "account 123456"})

    def test_dates_and_years_pass(self):
        ai.scrub_gate({"caption": "Aligned window 2021-01 → 2026-07 · 1260d"})

    def test_s3_payload_dollar_keys_denied(self):
        # S1 catch-#1 discipline: real key names from the S3 source
        # payloads must trip the gate if a reducer ever leaks them.
        for k in ("basis_remaining", "market_value", "unrealized_gl",
                  "cost_basis", "proceeds", "gains", "losses",
                  "total_unrealized_loss", "priced_basis", "nav"):
            with self.assertRaises(ai.AIScrubError, msg=k):
                ai.scrub_gate({k: 1.0})

    def test_b3_payload_dollar_keys_denied(self):
        # B3 source payloads' dollar-bearing names must stay deny-listed if a
        # reducer ever leaks them (S1 catch-#1 discipline). coverage_pct_nav
        # documents WHY the shipped key is coverage_of_book_pct.
        for k in ("start_nav", "end_nav", "join_value", "deposits_usd",
                  "coverage_pct_nav", "market_value", "projected_amount"):
            with self.assertRaises(ai.AIScrubError, msg=k):
                ai.scrub_gate({k: 1.0})


class TestDims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scope_key_dimless_byte_identical_to_s1(self):
        self.assertEqual(ai.scope_key(["all"], "all"),
                         '{"broker":["all"],"history_start":"all"}')
        self.assertEqual(ai.scope_key(["all"], "all", None),
                         ai.scope_key(["all"], "all"))
        self.assertEqual(ai.scope_key(["all"], "all", {}),
                         ai.scope_key(["all"], "all"))

    def test_scope_key_dims_canonical_and_distinct(self):
        k1 = ai.scope_key(["all"], "all", {"window": "3y", "model": "CAPM"})
        k2 = ai.scope_key(["all"], "all", {"model": "CAPM", "window": "3y"})
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, ai.scope_key(["all"], "all"))
        self.assertEqual(json.loads(k1)["dims"],
                         {"window": "3y", "model": "CAPM"})

    def test_scope_key_dims_comma_value_cannot_collide(self):
        a = ai.scope_key(["all"], "all", {"window": "a,b"})
        b = ai.scope_key(["all"], "all", {"window": "a", "model": "b"})
        self.assertNotEqual(a, b)

    def test_factors_declares_dims(self):
        self.assertEqual(ai.SECTIONS["factors"].get("dims"),
                         ("window", "model"))

    def test_factors_default_dims_equal_omitted(self):
        from terminal import factor_service as fs
        st = fs.build_factor_view(self.frames)["state"]
        explicit = ai.build_facts(
            "factors", self.frames,
            dims={"window": st["default_window"],
                  "model": st["default_model"]})
        m = _deep_close(explicit, ai.build_facts("factors", self.frames))
        self.assertIsNone(m, m)

    def test_factors_bad_dim_value_raises(self):
        with self.assertRaises(ai.AIDimError):
            ai.build_facts("factors", self.frames, dims={"window": "bogus"})
        with self.assertRaises(ai.AIDimError):
            ai.build_facts("factors", self.frames, dims={"model": "bogus"})

    def test_factors_nondefault_window_honored_or_honest(self):
        from terminal import factor_service as fs
        st = fs.build_factor_view(self.frames)["state"]
        others = [w for w in st["windows"] if w != st["default_window"]]
        if not others:
            self.skipTest("fixture offers a single window")
        facts = ai.build_facts("factors", self.frames,
                               dims={"window": others[0]})
        if facts["available"]:
            self.assertEqual(facts["window"], others[0])
        else:
            self.assertEqual(facts["reason"], "window_unavailable")


class _FakeBlock:
    def __init__(self, text):
        self.type, self.text = "text", text


class _FakeMsg:
    def __init__(self, text, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.content = [_FakeBlock(text)] if text is not None else []


class _FakeMessages:
    def __init__(self, msg=None, exc=None):
        self.msg, self.exc, self.calls = msg, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.msg


class FakeClient:
    """Duck-types the two attributes generate() touches:
    client.beta.messages.create(...)."""
    def __init__(self, msg=None, exc=None):
        self.beta = type("B", (), {})()
        self.beta.messages = _FakeMessages(msg, exc)


class _FakeToolBlock:
    def __init__(self, id, input):
        self.type, self.name = "tool_use", "fetch_detail"
        self.id, self.input = id, input


class _FakeToolMsg:
    def __init__(self, blocks):
        self.stop_reason = "tool_use"
        self.content = list(blocks)


class _FakeSeqMessages:
    """Scripted sequence: each create() pops the next message."""
    def __init__(self, msgs):
        self.msgs, self.calls = list(msgs), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.msgs.pop(0)


class FakeSeqClient:
    def __init__(self, msgs):
        self.beta = type("B", (), {})()
        self.beta.messages = _FakeSeqMessages(msgs)


class TestGenerate(unittest.TestCase):
    FACTS = {"section": "factors", "available": True, "window": "Full history"}

    def test_happy_path_returns_text_and_request_shape(self):
        c = FakeClient(_FakeMsg("  The book is market-driven.  "))
        out = ai.generate("factors", self.FACTS, client=c)
        self.assertEqual(out, "The book is market-driven.")
        kw = c.beta.messages.calls[0]
        self.assertEqual(kw["model"], ai.MODEL)
        self.assertEqual(kw["model"], "claude-fable-5")   # TK 2026-08-20: Fable everywhere
        self.assertEqual(kw["max_tokens"], 16000)
        self.assertEqual(kw["fallbacks"], "default")
        self.assertIn(ai._FALLBACK_BETA, kw["betas"])
        self.assertIn("FACTS:", kw["messages"][0]["content"])
        self.assertNotIn("thinking", kw)     # Fable 5: thinking always on; omit the param

    def test_refusal_raises(self):
        c = FakeClient(_FakeMsg("partial", stop_reason="refusal"))
        with self.assertRaises(ai.AIGenerationError):
            ai.generate("factors", self.FACTS, client=c)

    def test_api_error_raises_generation_error(self):
        c = FakeClient(exc=RuntimeError("boom"))
        with self.assertRaises(ai.AIGenerationError):
            ai.generate("factors", self.FACTS, client=c)

    def test_empty_content_raises(self):
        c = FakeClient(_FakeMsg(None))
        with self.assertRaises(ai.AIGenerationError):
            ai.generate("factors", self.FACTS, client=c)

    def test_resolve_client_none_without_key(self):
        with mock.patch.object(ai._config, "get_anthropic_key", return_value=""):
            self.assertIsNone(ai.resolve_client())


class TestResolveClientMemo(unittest.TestCase):
    """#371 review ledger, TK-raised: constructing anthropic.Anthropic loads
    the Windows cert store (~0.7-1 s measured on the real box) and ran on
    EVERY /api/ai/* request — resolve_client memoizes the built client
    module-wide, keyed by the API key so a renewed key rebuilds and a
    missing key is never cached."""

    def setUp(self):
        self._saved = ai._CLIENT_MEMO
        ai._CLIENT_MEMO = None
        self.addCleanup(self._restore)

    def _restore(self):
        ai._CLIENT_MEMO = self._saved

    @staticmethod
    def _stub_module():
        from types import SimpleNamespace
        built = []

        class Anthropic:
            def __init__(self, api_key):
                built.append(api_key)
                self.api_key = api_key

        return SimpleNamespace(Anthropic=Anthropic), built

    def test_same_key_reuses_client(self):
        stub, built = self._stub_module()
        with mock.patch.object(ai, "anthropic", stub), \
             mock.patch.object(ai._config, "get_anthropic_key",
                               return_value="sk-test-a"):
            first = ai.resolve_client()
            second = ai.resolve_client()
        self.assertIs(first, second)
        self.assertEqual(built, ["sk-test-a"])

    def test_key_change_rebuilds(self):
        stub, built = self._stub_module()
        with mock.patch.object(ai, "anthropic", stub):
            with mock.patch.object(ai._config, "get_anthropic_key",
                                   return_value="sk-test-a"):
                first = ai.resolve_client()
            with mock.patch.object(ai._config, "get_anthropic_key",
                                   return_value="sk-test-b"):
                second = ai.resolve_client()
                third = ai.resolve_client()
        self.assertIsNot(first, second)
        self.assertIs(second, third)
        self.assertEqual(built, ["sk-test-a", "sk-test-b"])

    def test_missing_key_not_cached(self):
        stub, built = self._stub_module()
        with mock.patch.object(ai, "anthropic", stub):
            with mock.patch.object(ai._config, "get_anthropic_key",
                                   return_value=""):
                self.assertIsNone(ai.resolve_client())
            self.assertEqual(built, [])
            with mock.patch.object(ai._config, "get_anthropic_key",
                                   return_value="sk-test-a"):
                self.assertIsNotNone(ai.resolve_client())
        self.assertEqual(built, ["sk-test-a"])


class TestGenerateCoalescing(unittest.TestCase):
    """Review-S2-3: concurrent same-scope cold requests must coalesce onto
    ONE API call (double-checked cache under a per-key lock)."""

    def test_two_threads_one_generation(self):
        import threading
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        ddir = Path(td.name)
        (ddir / "positions.csv").write_text("a\n1\n")
        dv = ai.data_version(ddir)
        calls = []

        class SlowMessages:
            def create(self, **kw):
                calls.append(1)
                time.sleep(0.4)
                return _FakeMsg("coalesced text")

        c = FakeClient()
        c.beta.messages = SlowMessages()
        results = []

        def hit():
            results.append(ai.generate_cached(
                ddir, "factors", "k", dv, {"section": "factors"}, client=c))

        t1 = threading.Thread(target=hit)
        t2 = threading.Thread(target=hit)
        t1.start(); time.sleep(0.05); t2.start()
        t1.join(); t2.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 2)
        texts = sorted((e["text"], hit_flag) for e, hit_flag in results)
        self.assertEqual([t for t, _ in texts],
                         ["coalesced text", "coalesced text"])
        self.assertEqual(sorted(h for _, h in results), [False, True])


class TestAsyncJobs(unittest.TestCase):
    """B1a: the _JOBS registry + start_generation background-job orchestration.
    _SPAWN is patched inline so the job runs synchronously and deterministically;
    the running-guard / error-recording / clear-on-success behaviour is what we
    assert, not thread timing."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.ddir = Path(td.name)
        (self.ddir / "positions.csv").write_text("a\n1\n")
        self.dv = ai.data_version(self.ddir)
        ai.clear_job("factors", "k")
        self.addCleanup(ai.clear_job, "factors", "k")

    def _inline(self):
        return mock.patch.object(ai, "_SPAWN", lambda fn: fn())

    def test_success_populates_cache_and_clears_job(self):
        c = FakeClient(_FakeMsg("async text"))
        with self._inline():
            ai.start_generation(self.ddir, "factors", "k", self.dv,
                                {"section": "factors"}, client=c)
        self.assertIsNone(ai.job_status("factors", "k"))      # cleared on success
        entry = ai.cache_get(self.ddir, "factors", "k")
        self.assertEqual(entry["text"], "async text")

    def test_error_recorded_in_registry(self):
        c = FakeClient(exc=RuntimeError("api down"))
        with self._inline():
            ai.start_generation(self.ddir, "factors", "k", self.dv,
                                {"section": "factors"}, client=c)
        st = ai.job_status("factors", "k")
        self.assertEqual(st["status"], "error")
        self.assertIn("api down", st["error"])

    def test_running_guard_blocks_second_spawn(self):
        # a no-op _SPAWN leaves the job in 'running'; the second call must not
        # spawn again (idempotent) — assert only one runnable was handed to _SPAWN.
        spawned = []
        with mock.patch.object(ai, "_SPAWN", lambda fn: spawned.append(fn)):
            ai.start_generation(self.ddir, "factors", "k", self.dv,
                                {"section": "factors"}, client=FakeClient(_FakeMsg("x")))
            ai.start_generation(self.ddir, "factors", "k", self.dv,
                                {"section": "factors"}, client=FakeClient(_FakeMsg("x")))
        self.assertEqual(len(spawned), 1)
        self.assertEqual(ai.job_status("factors", "k")["status"], "running")

    def test_non_aierror_recorded_not_wedged(self):
        # A non-AIError failure (unknown section -> KeyError in generate, before
        # the client is used) must be recorded as error, not leave the job wedged
        # at "running" forever.
        with self._inline():
            ai.start_generation(self.ddir, "no_such_section", "k2", self.dv,
                                {"section": "no_such_section"},
                                client=FakeClient(_FakeMsg("x")))
        st = ai.job_status("no_such_section", "k2")
        self.assertIsNotNone(st)
        self.assertEqual(st["status"], "error")
        self.addCleanup(ai.clear_job, "no_such_section", "k2")


class TestChatJobs(unittest.TestCase):
    """S2: chat jobs ride the same _JOBS registry under chat|<id> keys.
    DOCUMENTED DEVIATION from the B1a 'registry never holds text' rule:
    chat answers are deliberately uncached, so a finished entry holds its
    text until the first successful poll pops it."""

    PACK = {"risk": {"available": True}}
    MSGS = [{"role": "user", "content": "q"}]

    def setUp(self):
        ai.clear_chat("t1")
        self.addCleanup(ai.clear_chat, "t1")

    def _inline(self):
        return mock.patch.object(ai, "_SPAWN", lambda fn: fn())

    def test_done_holds_text(self):
        with self._inline():
            ai.start_chat("t1", self.MSGS, self.PACK,
                          client=FakeClient(_FakeMsg("the answer")))
        st = ai.chat_status("t1")
        self.assertEqual(st, {"status": "done", "text": "the answer"})

    def test_error_recorded_not_wedged(self):
        with self._inline():
            ai.start_chat("t1", self.MSGS, self.PACK,
                          client=FakeClient(exc=RuntimeError("api down")))
        st = ai.chat_status("t1")
        self.assertEqual(st["status"], "error")
        self.assertIn("api down", st["error"])

    def test_running_guard_idempotent(self):
        spawned = []
        with mock.patch.object(ai, "_SPAWN", lambda fn: spawned.append(fn)):
            ai.start_chat("t1", self.MSGS, self.PACK,
                          client=FakeClient(_FakeMsg("x")))
            ai.start_chat("t1", self.MSGS, self.PACK,
                          client=FakeClient(_FakeMsg("x")))
        self.assertEqual(len(spawned), 1)
        self.assertEqual(ai.chat_status("t1")["status"], "running")

    def test_clear_pops(self):
        with self._inline():
            ai.start_chat("t1", self.MSGS, self.PACK,
                          client=FakeClient(_FakeMsg("x")))
        ai.clear_chat("t1")
        self.assertIsNone(ai.chat_status("t1"))


class TestNarrative(unittest.TestCase):
    GOOD = ('{"verdict": "v", "why": ["w1", "w2"], '
            '"changes": "c", "watch": ["x"]}')

    def test_parse_good(self):
        d = ai.parse_narrative(self.GOOD)
        self.assertEqual(d, {"verdict": "v", "why": ["w1", "w2"],
                             "changes": "c", "watch": ["x"]})

    def test_parse_bad(self):
        self.assertIsNone(ai.parse_narrative("not json"))
        self.assertIsNone(ai.parse_narrative('{"verdict": "v"}'))
        self.assertIsNone(ai.parse_narrative('{"verdict": 1, "why": ["w"], '
                                             '"changes": "c", "watch": ["x"]}'))
        # legacy 4-string shape (pre-v2 cache) is malformed now:
        self.assertIsNone(ai.parse_narrative('{"verdict": "v", "why": "w", '
                                             '"changes": "c", "watch": "x"}'))
        # empty or non-string bullets are malformed:
        self.assertIsNone(ai.parse_narrative('{"verdict": "v", "why": [], '
                                             '"changes": "c", "watch": ["x"]}'))
        self.assertIsNone(ai.parse_narrative('{"verdict": "v", "why": [1], '
                                             '"changes": "c", "watch": ["x"]}'))

    def test_generate_passes_schema_and_system_for_portfolio(self):
        c = FakeClient(_FakeMsg(self.GOOD))
        out = ai.generate("portfolio", {"section": "portfolio"}, client=c)
        self.assertEqual(out, self.GOOD)
        kw = c.beta.messages.calls[0]
        self.assertEqual(kw["output_config"]["format"]["type"], "json_schema")
        props = kw["output_config"]["format"]["schema"]["properties"]
        self.assertEqual(props["why"]["type"], "array")
        self.assertEqual(props["watch"]["type"], "array")
        self.assertEqual(props["verdict"]["type"], "string")
        self.assertIn("JSON", kw["system"])

    def test_generate_box_sections_get_box_schema(self):
        c = FakeClient(_FakeMsg('{"headline": "h", "bullets": ["b"]}'))
        ai.generate("factors", {"section": "factors"}, client=c)
        kw = c.beta.messages.calls[0]
        fmt = kw["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertIn("headline", fmt["schema"]["properties"])
        self.assertIn("bullets", fmt["schema"]["properties"])
        self.assertIn("JSON", kw["system"])


class TestBoxParse(unittest.TestCase):
    """parse_box: the structured-box seatbelt. None = legacy prose (the FE
    renders those as a plain paragraph via the STALE path)."""

    def test_good_with_caveat(self):
        d = ai.parse_box('{"headline": "h", "bullets": ["a", "b"], '
                         '"caveat": "c"}')
        self.assertEqual(d, {"headline": "h", "bullets": ["a", "b"],
                             "caveat": "c", "watch": None})

    def test_good_without_caveat(self):
        d = ai.parse_box('{"headline": "h", "bullets": ["a"]}')
        self.assertEqual(d, {"headline": "h", "bullets": ["a"],
                             "caveat": None, "watch": None})

    def test_legacy_prose_none(self):
        self.assertIsNone(ai.parse_box("plain prose paragraph"))
        self.assertIsNone(ai.parse_box(""))

    def test_malformed_none(self):
        self.assertIsNone(ai.parse_box('{"headline": "h"}'))
        self.assertIsNone(ai.parse_box('{"headline": "h", "bullets": []}'))
        self.assertIsNone(ai.parse_box(
            '{"headline": "h", "bullets": ["a"], "extra": 1}'))
        self.assertIsNone(ai.parse_box('{"headline": 1, "bullets": ["a"]}'))
        self.assertIsNone(ai.parse_box(
            '{"headline": "h", "bullets": ["a", 2]}'))
        self.assertIsNone(ai.parse_box(
            '{"headline": "h", "bullets": ["a"], "caveat": 3}'))

    def test_watch_accepted_and_returned(self):
        d = ai.parse_box('{"headline": "h", "bullets": ["a"], '
                         '"watch": ["w1", "w2"]}')
        self.assertEqual(d, {"headline": "h", "bullets": ["a"],
                             "caveat": None, "watch": ["w1", "w2"]})

    def test_watch_absent_is_none(self):
        d = ai.parse_box('{"headline": "h", "bullets": ["a"]}')
        self.assertIsNone(d["watch"])

    def test_watch_malformed_none(self):
        self.assertIsNone(ai.parse_box(
            '{"headline": "h", "bullets": ["a"], "watch": []}'))
        self.assertIsNone(ai.parse_box(
            '{"headline": "h", "bullets": ["a"], "watch": [1]}'))
        self.assertIsNone(ai.parse_box(
            '{"headline": "h", "bullets": ["a"], "watch": "w"}'))


class TestGenerateChat(unittest.TestCase):
    PACK = {"risk": {"section": "risk", "available": True, "vol_pct": 16.3}}
    MSGS = [{"role": "user", "content": "How risky is the book?"},
            {"role": "assistant", "content": "Moderately."},
            {"role": "user", "content": "And versus SPY?"}]

    def test_request_shape(self):
        c = FakeClient(_FakeMsg("  From FACTS, beta is near 1.  "))
        out = ai.generate_chat(self.MSGS, self.PACK, client=c)
        self.assertEqual(out, "From FACTS, beta is near 1.")
        kw = c.beta.messages.calls[0]
        self.assertEqual(kw["model"], "claude-fable-5")
        self.assertEqual(kw["max_tokens"], 16000)
        self.assertEqual(kw["fallbacks"], "default")
        self.assertIn(ai._FALLBACK_BETA, kw["betas"])
        self.assertNotIn("thinking", kw)
        self.assertEqual(kw["output_config"], {"effort": ai._CHAT_EFFORT})
        self.assertEqual(ai._CHAT_EFFORT, "medium")   # TK 2026-08-22: faster turns
        self.assertNotIn("format", kw["output_config"])   # free prose — no schema
        self.assertEqual(kw["system"], ai._CHAT_SYSTEM)

    def test_pack_rides_first_user_block_with_cache_control(self):
        c = FakeClient(_FakeMsg("ok"))
        ai.generate_chat(self.MSGS, self.PACK, client=c)
        convo = c.beta.messages.calls[0]["messages"]
        self.assertEqual(len(convo), 3)
        self.assertEqual(convo[0]["role"], "user")
        first_blocks = convo[0]["content"]
        self.assertEqual(first_blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertTrue(first_blocks[0]["text"].startswith("FACTS:\n"))
        self.assertIn('"vol_pct": 16.3', first_blocks[0]["text"])
        self.assertEqual(first_blocks[1]["text"], "How risky is the book?")
        # later turns ride as plain strings, no pack duplication:
        self.assertEqual(convo[1], {"role": "assistant",
                                    "content": "Moderately."})
        self.assertEqual(convo[2], {"role": "user",
                                    "content": "And versus SPY?"})

    def test_refusal_and_empty_raise(self):
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK,
                             client=FakeClient(_FakeMsg("x", stop_reason="refusal")))
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK,
                             client=FakeClient(_FakeMsg(None)))

    def test_api_error_raises(self):
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK,
                             client=FakeClient(exc=RuntimeError("down")))

    def test_system_prompt_carries_doctrine(self):
        for phrase in ("dollar", "never recommend", "FACTS"):
            self.assertIn(phrase, ai._CHAT_SYSTEM)


class TestChatToolLoop(unittest.TestCase):
    """Full-gate S2: the bounded fetch_detail loop (spec §5.3)."""

    PACK = {"portfolio": {"section": "portfolio"}}
    MSGS = [{"role": "user", "content": "hi"}]

    def test_no_detail_fn_no_tools_param(self):
        c = FakeSeqClient([_FakeMsg("plain answer")])
        out = ai.generate_chat(self.MSGS, self.PACK, client=c)
        self.assertEqual(out, "plain answer")
        self.assertNotIn("tools", c.beta.messages.calls[0])

    def test_tools_param_present_with_detail_fn(self):
        c = FakeSeqClient([_FakeMsg("done")])
        ai.generate_chat(self.MSGS, self.PACK, client=c,
                         detail_fn=lambda t, k: {"ok": True})
        kw = c.beta.messages.calls[0]
        self.assertEqual(kw["tools"], [ai._FETCH_DETAIL_TOOL])
        self.assertTrue(ai._FETCH_DETAIL_TOOL["strict"])
        self.assertEqual(
            ai._FETCH_DETAIL_TOOL["input_schema"]["properties"]["topic"]
            ["enum"],
            ["riskcontrib", "income", "lots", "transactions",
             "performance", "dip", "factor", "options_contracts",
             "health"])

    def test_one_round_executes_and_replays_verbatim(self):
        calls = []
        def fn(topic, ticker):
            calls.append((topic, ticker))
            return {"topic": topic, "rows": [1]}
        tool_msg = _FakeToolMsg([_FakeToolBlock("tu_1",
                                                {"topic": "riskcontrib",
                                                 "ticker": "STRL"})])
        c = FakeSeqClient([tool_msg, _FakeMsg("answered")])
        out = ai.generate_chat(self.MSGS, self.PACK, client=c, detail_fn=fn)
        self.assertEqual(out, "answered")
        self.assertEqual(calls, [("riskcontrib", "STRL")])
        second = c.beta.messages.calls[1]["messages"]
        self.assertIs(second[-2]["content"], tool_msg.content)  # verbatim
        results = second[-1]["content"]
        self.assertEqual(second[-1]["role"], "user")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "tool_result")
        self.assertEqual(results[0]["tool_use_id"], "tu_1")
        self.assertNotIn("is_error", results[0])
        self.assertIn('"rows": [1]', results[0]["content"])

    def test_parallel_blocks_one_result_message(self):
        tool_msg = _FakeToolMsg([
            _FakeToolBlock("tu_1", {"topic": "income"}),
            _FakeToolBlock("tu_2", {"topic": "lots", "ticker": "SPY"})])
        c = FakeSeqClient([tool_msg, _FakeMsg("ok")])
        ai.generate_chat(self.MSGS, self.PACK, client=c,
                         detail_fn=lambda t, k: {"topic": t})
        results = c.beta.messages.calls[1]["messages"][-1]["content"]
        self.assertEqual([r["tool_use_id"] for r in results],
                         ["tu_1", "tu_2"])

    def test_error_dict_becomes_is_error(self):
        tool_msg = _FakeToolMsg([_FakeToolBlock("tu_1", {"topic": "nope"})])
        c = FakeSeqClient([tool_msg, _FakeMsg("ok")])
        ai.generate_chat(self.MSGS, self.PACK, client=c,
                         detail_fn=lambda t, k: {"error": "unknown topic"})
        r = c.beta.messages.calls[1]["messages"][-1]["content"][0]
        self.assertTrue(r["is_error"])
        self.assertIn("unknown topic", r["content"])

    def test_scrub_error_fails_turn_closed(self):
        def fn(topic, ticker):
            raise ai.AIScrubError("denied key")
        tool_msg = _FakeToolMsg([_FakeToolBlock("tu_1",
                                                {"topic": "income"})])
        c = FakeSeqClient([tool_msg])
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK, client=c, detail_fn=fn)

    def test_round_cap_nudges_then_fails(self):
        tool = lambda i: _FakeToolMsg([_FakeToolBlock(f"tu_{i}",
                                                      {"topic": "income"})])
        executed = []
        def fn(topic, ticker):
            executed.append(topic)
            return {"topic": topic}
        # 5 consecutive tool_use turns: 3 executed, 4th nudged, 5th raises
        c = FakeSeqClient([tool(1), tool(2), tool(3), tool(4), tool(5)])
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK, client=c, detail_fn=fn)
        self.assertEqual(len(executed), 3)
        nudge = c.beta.messages.calls[4]["messages"][-1]["content"][0]
        self.assertTrue(nudge["is_error"])
        self.assertIn("budget exhausted", nudge["content"])
        self.assertEqual(len(c.beta.messages.calls), 5)

    def test_refusal_still_raises(self):
        c = FakeSeqClient([_FakeMsg("partial", stop_reason="refusal")])
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK, client=c,
                             detail_fn=lambda t, k: {})

    def test_nan_in_result_fails_serialization_closed(self):
        tool_msg = _FakeToolMsg([_FakeToolBlock("tu_1",
                                                {"topic": "income"})])
        c = FakeSeqClient([tool_msg, _FakeMsg("ok")])
        with self.assertRaises(ai.AIGenerationError):
            ai.generate_chat(self.MSGS, self.PACK, client=c,
                             detail_fn=lambda t, k: {"x": float("nan")})


class TestQuestions(unittest.TestCase):
    """A-feedback 2026-08-20: every AI surface shows the question it answers."""

    def test_every_section_declares_question(self):
        for name, sec in ai.SECTIONS.items():
            q = sec.get("question")
            self.assertIsInstance(q, str, name)
            self.assertTrue(q.strip().endswith("?"), name)

    def test_portfolio_questions_names_benchmark(self):
        qs = ai.portfolio_questions("SPY")
        self.assertEqual(set(qs), {"verdict", "why", "changes", "watch"})
        self.assertIn("SPY", qs["verdict"])
        for v in qs.values():
            self.assertTrue(v.strip().endswith("?"))


class TestConfig(unittest.TestCase):
    """get_anthropic_key(): optional, never raises, placeholder-guarded.

    The unset case mocks load_env — the dev box's real .env may hold a real
    key, and this test must never read it."""

    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            self.assertEqual(_config.get_anthropic_key(), "sk-ant-test123")

    def test_placeholder_is_unset(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "PASTE_YOUR_KEY_HERE"}):
            self.assertEqual(_config.get_anthropic_key(), "")

    def test_unset_returns_empty(self):
        # All three sources mocked empty — the dev box's real key (env file
        # or User registry) must never reach this test, or its failure
        # message could print it.
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), \
             mock.patch.object(_config, "load_env", return_value={}), \
             mock.patch.object(_config, "_user_scope_env", return_value=""):
            self.assertEqual(_config.get_anthropic_key(), "")

    def test_registry_fallback_when_env_and_file_empty(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), \
             mock.patch.object(_config, "load_env", return_value={}), \
             mock.patch.object(_config, "_user_scope_env",
                               return_value="sk-ant-from-registry"):
            self.assertEqual(_config.get_anthropic_key(),
                             "sk-ant-from-registry")


class TestRoutes(unittest.TestCase):
    """Route tests run against a TEMP COPY of the fixture so cache writes
    never touch the committed fixture dir, and with resolve_client patched —
    tests must never read the dev box's real .env key."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def setUp(self):
        cache = self.ddir / "ai_cache.json"
        if cache.exists():
            cache.unlink()
        p = mock.patch.object(ai, "_SPAWN", lambda fn: fn())   # inline = deterministic
        p.start(); self.addCleanup(p.stop)
        ai.clear_job("factors", ai.scope_key(["all"], "all", None))

    def _fake(self, text="canned narration"):
        return mock.patch.object(ai, "resolve_client",
                                 return_value=FakeClient(_FakeMsg(text)))

    def test_explain_generating_then_cached(self):
        # async: first call kicks the (inline) job and returns generating;
        # the job has populated the cache, so the second call (poll) is ok.
        with self._fake("canned narration"):
            r1 = self.client.get("/api/ai/explain?section=factors")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["kind"], "generating")
        self.assertIsNone(r1.json()["text"])
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("hit path resolved")):
            r2 = self.client.get("/api/ai/explain?section=factors")
        d2 = r2.json()
        self.assertEqual(d2["kind"], "ok")
        self.assertTrue(d2["cached"])
        self.assertEqual(d2["text"], "canned narration")

    def test_legacy_cache_entry_regenerates_on_explain(self):
        # A fresh-dv entry WITHOUT fmt (pre-v2) must be treated as a miss:
        # the route spawns a regen instead of serving the prose as current.
        skey = ai.scope_key(["all"], "all", None)
        dv = ai.data_version(self.ddir)
        ai.cache_put(self.ddir, "factors", skey, dv, "legacy prose")
        p = self.ddir / "ai_cache.json"
        blob = json.loads(p.read_text(encoding="utf-8"))
        del blob[f"factors|{skey}"]["fmt"]
        p.write_text(json.dumps(blob), encoding="utf-8")
        with self._fake("fresh v2"):
            r1 = self.client.get("/api/ai/explain?section=factors")
            self.assertEqual(r1.json()["kind"], "generating")
            r2 = self.client.get("/api/ai/explain?section=factors")
        self.assertEqual(r2.json()["text"], "fresh v2")

    def test_frontier_section_generates_with_sig(self):
        # warm the memo via a real frontier POST, then narrate that result
        pr = self.client.post("/api/risksim/frontier",
                              json={"cap_pct": 40.0, "floors": {}})
        self.assertEqual(pr.status_code, 200)
        sig = pr.json()["sig"]
        with self._fake("frontier narration"):
            r1 = self.client.get(f"/api/ai/explain?section=frontier&sig={sig}")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("hit path resolved")):
            r2 = self.client.get(f"/api/ai/explain?section=frontier&sig={sig}")
        self.assertEqual(r2.json()["kind"], "ok")
        self.assertEqual(r2.json()["text"], "frontier narration")

    def test_frontier_cold_sig_still_200(self):
        # a cold/garbage sig -> facts available:false, still a 200 (not 422)
        with self._fake("stale narration"):
            r = self.client.get("/api/ai/explain?section=frontier"
                                "&sig=deadbeefdeadbeef")
        self.assertEqual(r.status_code, 200)

    def test_unknown_section_422(self):
        self.assertEqual(
            self.client.get("/api/ai/explain?section=nope").status_code, 422)

    def test_unknown_broker_422(self):
        with self._fake():
            r = self.client.get("/api/ai/explain?section=factors&broker=nope")
        self.assertEqual(r.status_code, 422)

    def test_off_state_without_client(self):
        with mock.patch.object(ai, "resolve_client", return_value=None):
            r = self.client.get("/api/ai/explain?section=factors")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertFalse(d["enabled"])
        self.assertIsNone(d["text"])

    def test_regenerate_forces_new_text(self):
        with self._fake("first"):
            self.client.get("/api/ai/explain?section=factors")   # generating -> cached "first"
            self.client.get("/api/ai/explain?section=factors")   # ok "first"
        with self._fake("second"):
            rp = self.client.post("/api/ai/regenerate", json={"section": "factors"})
            self.assertEqual(rp.json()["kind"], "generating")     # force spawns, returns generating
            r = self.client.get("/api/ai/explain?section=factors")
        self.assertEqual(r.json()["text"], "second")

    def test_regenerate_extra_field_422(self):
        r = self.client.post("/api/ai/regenerate",
                             json={"section": "factors", "bogus": 1})
        self.assertEqual(r.status_code, 422)

    def test_api_failure_serves_stale(self):
        with self._fake("good text"):
            self.client.get("/api/ai/explain?section=factors")    # generating
            self.client.get("/api/ai/explain?section=factors")    # ok "good text"
        (self.ddir / "positions.csv").touch()                     # invalidate the version
        failing = FakeClient(exc=RuntimeError("api down"))
        with mock.patch.object(ai, "resolve_client", return_value=failing):
            self.client.get("/api/ai/explain?section=factors")    # kicks a job that fails
            r = self.client.get("/api/ai/explain?section=factors")  # poll surfaces the error
        d = r.json()
        self.assertEqual(d["kind"], "error")
        self.assertTrue(d["stale"])
        self.assertEqual(d["text"], "good text")

    def test_failed_force_regen_surfaces_error_not_ok(self):
        # running->error->cache: a failed force regen with a still-fresh same-dv
        # cache must return kind:error + stale old text (not silently ok "old"),
        # and must clear the _JOBS error entry (no leak).
        with self._fake("old"):
            self.client.get("/api/ai/explain?section=factors")   # generating (inline caches "old")
            self.client.get("/api/ai/explain?section=factors")   # ok "old"
        failing = FakeClient(exc=RuntimeError("regen api down"))
        with mock.patch.object(ai, "resolve_client", return_value=failing):
            self.assertEqual(
                self.client.post("/api/ai/regenerate", json={"section": "factors"}).json()["kind"],
                "generating")                                    # force job spawned (inline, fails)
            r = self.client.get("/api/ai/explain?section=factors")   # poll surfaces the error
        d = r.json()
        self.assertEqual(d["kind"], "error")
        self.assertTrue(d["stale"])
        self.assertEqual(d["text"], "old")
        # derive skey exactly as the handler does for this request, then assert cleared:
        skey = ai.scope_key(["all"], "all", None)
        self.assertIsNone(ai.job_status("factors", skey))

    def test_explain_portfolio_returns_parsed_narrative(self):
        # B1b: /api/ai/explain?section=portfolio is async AND returns a parsed
        # 4-field narrative (so the AI-tab poll can render the grid).
        with self._fake(self.PORT_JSON):
            r1 = self.client.get("/api/ai/explain?section=portfolio")
        self.assertEqual(r1.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("hit path resolved")):
            r2 = self.client.get("/api/ai/explain?section=portfolio")
        d2 = r2.json()
        self.assertEqual(d2["kind"], "ok")
        self.assertEqual(d2["narrative"], {"verdict": "not riskier",
                                           "why": ["w"],
                                           "changes": "c", "watch": ["x"]})

    def test_explain_factors_has_no_narrative_key(self):
        # box sections keep raw text, no parsed narrative
        with self._fake("box text"):
            self.client.get("/api/ai/explain?section=factors")
            r = self.client.get("/api/ai/explain?section=factors")
        d = r.json()
        self.assertEqual(d["text"], "box text")
        self.assertNotIn("narrative", d)

    BOX_JSON = ('{"headline": "Risk is moderate at 15.4% vol.", '
                '"bullets": ["Beta 1.1 vs SPY", "Top-10 hold 62% of risk"], '
                '"caveat": "1y window has 9 months."}')

    def test_explain_box_structured_parsed(self):
        with self._fake(self.BOX_JSON):
            self.client.get("/api/ai/explain?section=factors")
            r = self.client.get("/api/ai/explain?section=factors")
        d = r.json()
        self.assertEqual(d["kind"], "ok")
        self.assertEqual(d["structured"]["headline"],
                         "Risk is moderate at 15.4% vol.")
        self.assertEqual(len(d["structured"]["bullets"]), 2)
        self.assertEqual(d["structured"]["caveat"], "1y window has 9 months.")

    def test_explain_box_legacy_prose_structured_none(self):
        # Legacy prose (pre-v2 cache surfacing) must serve untouched with
        # structured:null and kind ok — the FE falls back to a paragraph.
        with self._fake("plain legacy prose"):
            self.client.get("/api/ai/explain?section=factors")
            r = self.client.get("/api/ai/explain?section=factors")
        d = r.json()
        self.assertEqual(d["kind"], "ok")
        self.assertIsNone(d["structured"])
        self.assertEqual(d["text"], "plain legacy prose")

    def test_explain_carries_question(self):
        with self._fake("q text"):
            r = self.client.get("/api/ai/explain?section=factors")
        self.assertEqual(r.status_code, 200)
        q = r.json()["question"]
        self.assertIsInstance(q, str)
        self.assertTrue(q.endswith("?"))

    def test_portfolio_payload_carries_questions(self):
        with self._fake(self.PORT_JSON):
            r = self.client.get("/api/ai/portfolio")
        d = r.json()
        qs = d["questions"]
        self.assertEqual(set(qs), {"verdict", "why", "changes", "watch"})
        self.assertIn(d["meta"]["benchmark"]["short"], qs["verdict"])

    def test_explain_portfolio_malformed_cache_is_error(self):
        # Review fix: a cached-but-unparseable narrative must not surface as
        # kind:"ok" (a frontend keyed on kind=="error" would spin forever).
        # resolve_client is guarded so a skey/dv mismatch (cache miss) fails
        # loudly here instead of silently risking a real network call.
        skey = ai.scope_key(["all"], "all", None)
        dv = ai.data_version(self.ddir)
        ai.cache_put(self.ddir, "portfolio", skey, dv, "not valid json")
        try:
            with mock.patch.object(ai, "resolve_client",
                                   side_effect=AssertionError("hit path resolved")):
                r = self.client.get("/api/ai/explain?section=portfolio")
        finally:
            ai.clear_job("portfolio", skey)
        d = r.json()
        self.assertEqual(d["kind"], "error")
        self.assertIsNone(d["narrative"])

    PORT_JSON = ('{"verdict": "not riskier", "why": ["w"], '
                 '"changes": "c", "watch": ["x"]}')

    def test_portfolio_route_generating_then_ok(self):
        with self._fake(self.PORT_JSON):
            r1 = self.client.get("/api/ai/portfolio")
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertEqual(d1["kind"], "generating")
        self.assertIsNone(d1["narrative"])
        self.assertEqual(len(d1["display"]["window_table"]["rows"]), 5)  # facts served
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("hit resolved")):
            r2 = self.client.get("/api/ai/portfolio")
        d2 = r2.json()
        self.assertEqual(d2["narrative"]["verdict"], "not riskier")
        self.assertTrue(d2["narrative_meta"]["cached"])

    def test_portfolio_off_state_still_serves_facts(self):
        with mock.patch.object(ai, "resolve_client", return_value=None):
            r = self.client.get("/api/ai/portfolio")
        d = r.json()
        self.assertFalse(d["enabled"])
        self.assertIsNone(d["narrative"])
        self.assertTrue(d["facts"]["available"])
        self.assertEqual(len(d["display"]["window_table"]["rows"]), 5)

    def test_portfolio_unknown_broker_422(self):
        with self._fake(self.PORT_JSON):
            r = self.client.get("/api/ai/portfolio?broker=nope")
        self.assertEqual(r.status_code, 422)

    def test_portfolio_bad_narrative_is_error_not_500(self):
        # B1b: the first GET only spawns (inline _SPAWN completes the job and
        # caches the malformed text, but the response is already built as
        # "generating" before that mount happens) -- the poll is what surfaces
        # the malformed-cache error, same as test_explain_portfolio_malformed_cache_is_error.
        with self._fake("not json at all"):
            r1 = self.client.get("/api/ai/portfolio")
        self.assertEqual(r1.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("hit resolved")):
            r2 = self.client.get("/api/ai/portfolio")
        d = r2.json()
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(d["kind"], "error")
        self.assertIsNone(d["narrative"])
        self.assertTrue(d["facts"]["available"])

    def test_regenerate_portfolio_section_works(self):
        # B1b: portfolio regenerate is async too now — force spawns a job and
        # returns generating; the poll then serves ok with a parsed narrative.
        with self._fake(self.PORT_JSON):
            rp = self.client.post("/api/ai/regenerate",
                                  json={"section": "portfolio"})
            self.assertEqual(rp.json()["kind"], "generating")
            r = self.client.get("/api/ai/explain?section=portfolio")
        d = r.json()
        self.assertEqual(d["kind"], "ok")
        self.assertEqual(d["narrative"]["verdict"], "not riskier")

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(Path(self.td.name) / "gone")
        try:
            with self._fake():
                r = self.client.get("/api/ai/explain?section=factors")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(self.ddir)

    def test_dim_on_dimless_section_422(self):
        with self._fake():
            r = self.client.get("/api/ai/explain?section=portfolio&window=3y")
        self.assertEqual(r.status_code, 422)

    def test_undeclared_dim_name_422(self):
        with self._fake():
            r = self.client.get(
                "/api/ai/explain?section=factors&estimator=ewma_lw")
        self.assertEqual(r.status_code, 422)

    def test_bad_dim_value_422_on_generating_path(self):
        with self._fake():
            r = self.client.get("/api/ai/explain?section=factors&window=bogus")
        self.assertEqual(r.status_code, 422)

    def test_dims_cache_isolated_from_default(self):
        with self._fake("default text"):
            self.client.get("/api/ai/explain?section=factors")   # generating -> cached
            self.client.get("/api/ai/explain?section=factors")   # ok "default text"
        fv = self.client.get("/api/factor").json()
        others = [w for w in fv["state"]["windows"]
                  if w != fv["state"]["default_window"]]
        if not others:
            self.skipTest("fixture offers a single window")
        with self._fake("dim text"):
            self.client.get(
                f"/api/ai/explain?section=factors&window={others[0]}")  # generating
            r = self.client.get(
                f"/api/ai/explain?section=factors&window={others[0]}")  # ok "dim text"
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "dim text")
        # The default-scope entry is untouched — unpatched GET must cache-hit.
        r2 = self.client.get("/api/ai/explain?section=factors")
        self.assertTrue(r2.json()["cached"])
        self.assertEqual(r2.json()["text"], "default text")

    def test_regenerate_with_dims(self):
        fv = self.client.get("/api/factor").json()
        w0 = fv["state"]["default_window"]
        with self._fake("first"):
            self.client.get(f"/api/ai/explain?section=factors&window={w0}")   # generating -> cached "first"
            self.client.get(f"/api/ai/explain?section=factors&window={w0}")   # ok "first"
        with self._fake("second"):
            rp = self.client.post("/api/ai/regenerate",
                                  json={"section": "factors", "window": w0})
            self.assertEqual(rp.json()["kind"], "generating")     # force spawns, returns generating
            r = self.client.get(f"/api/ai/explain?section=factors&window={w0}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "second")

    def _risksim_sig(self):
        w = (rss._bundle_for(hs.load_frames(self.ddir), "all", "all")["weights"]
             * 100.0).sort_values(ascending=False)
        rw = {str(t): float(v) for t, v in w.items()}
        rw[str(w.index[0])] -= 5.0
        rw[str(w.index[-1])] += 5.0
        return self.client.post("/api/risksim/simulate",
                                json={"weights": rw}).json()["sig"]

    def test_risksim_section_generates_with_sig(self):
        sig = self._risksim_sig()
        with self._fake("risksim narration"):
            r1 = self.client.get(f"/api/ai/explain?section=risksim&sig={sig}")
        self.assertEqual(r1.status_code, 200)
        with self._fake("risksim narration"):
            r2 = self.client.get(f"/api/ai/explain?section=risksim&sig={sig}")
        self.assertEqual(r2.json()["kind"], "ok")
        self.assertEqual(r2.json()["text"], "risksim narration")
        self.assertTrue(r2.json()["cached"])

    def test_risksim_cold_sig_still_200(self):
        with self._fake("stale"):
            r = self.client.get("/api/ai/explain?section=risksim"
                                "&sig=deadbeefdeadbeef")
        self.assertEqual(r.status_code, 200)

    def test_risksim_rejects_foreign_dim(self):
        self.assertEqual(self.client.get(
            "/api/ai/explain?section=risksim&window=5y").status_code, 422)

    def test_explain_brief_ok_and_dim_rejected(self):
        with self._fake('{"headline": "h", "bullets": ["b"], '
                        '"watch": ["w"]}'):
            self.client.get("/api/ai/explain?section=brief")
            r = self.client.get("/api/ai/explain?section=brief")
        d = r.json()
        self.assertEqual(d["kind"], "ok")
        self.assertEqual(d["structured"]["watch"], ["w"])
        self.assertTrue(d["question"].endswith("?"))
        self.assertEqual(
            self.client.get("/api/ai/explain?section=brief&window=1Y")
            .status_code, 422)


class TestChatRoutes(unittest.TestCase):
    """S2: POST /api/ai/chat + GET poll. Off-state skips semantic
    validation (doctrine); >12 messages TRIM silently; oversize/role
    violations 422; ok/error polls pop the registry entry."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def setUp(self):
        ai._CHAT_PACK_MEMO.clear()
        p = mock.patch.object(ai, "_SPAWN", lambda fn: fn())
        p.start(); self.addCleanup(p.stop)

    def _fake(self, text="a grounded answer"):
        return mock.patch.object(ai, "resolve_client",
                                 return_value=FakeClient(_FakeMsg(text)))

    def _post(self, msgs, **extra):
        return self.client.post("/api/ai/chat",
                                json={"messages": msgs, **extra})

    def test_turn_generates_then_ok_then_gone(self):
        with self._fake("beta is 1.08"):
            r = self._post([{"role": "user", "content": "how risky?"}])
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["kind"], "generating")
        cid = d["chat_id"]
        r2 = self.client.get(f"/api/ai/chat?id={cid}")
        d2 = r2.json()
        self.assertEqual(d2["kind"], "ok")
        self.assertEqual(d2["text"], "beta is 1.08")
        self.assertEqual(d2["model"], "claude-fable-5")
        self.assertEqual(self.client.get(f"/api/ai/chat?id={cid}").status_code,
                         404)                     # popped on the ok read

    def test_error_surfaced_and_popped(self):
        failing = FakeClient(exc=RuntimeError("api down"))
        with mock.patch.object(ai, "resolve_client", return_value=failing):
            cid = self._post([{"role": "user", "content": "q"}]).json()["chat_id"]
        d = self.client.get(f"/api/ai/chat?id={cid}").json()
        self.assertEqual(d["kind"], "error")
        self.assertIn("api down", d["error"])
        self.assertEqual(self.client.get(f"/api/ai/chat?id={cid}").status_code,
                         404)

    def test_multi_turn_and_silent_trim(self):
        msgs = []
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "user", "content": "final"})   # 17 total
        with self._fake("trimmed fine"):
            cid = self._post(msgs).json()["chat_id"]
        d = self.client.get(f"/api/ai/chat?id={cid}").json()
        self.assertEqual(d["kind"], "ok")

    def test_trim_window_reaches_generate_chat(self):
        seen = {}
        real = ai.generate_chat
        def spy(messages, pack, *, client, detail_fn=None):
            seen["n"] = len(messages)
            seen["first_role"] = messages[0]["role"]
            return real(messages, pack, client=client, detail_fn=detail_fn)
        msgs = []
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "user", "content": "final"})   # 17 total
        with self._fake("ok"), mock.patch.object(ai, "generate_chat",
                                                 side_effect=spy):
            self._post(msgs)
        self.assertLessEqual(seen["n"], 12)
        self.assertEqual(seen["first_role"], "user")

    def test_422_shapes(self):
        with self._fake():
            self.assertEqual(self._post([]).status_code, 422)
            self.assertEqual(self._post(
                [{"role": "user", "content": "q"},
                 {"role": "assistant", "content": "a"}]).status_code, 422)
            self.assertEqual(self._post(
                [{"role": "system", "content": "x"},
                 {"role": "user", "content": "q"}]).status_code, 422)
            self.assertEqual(self._post(
                [{"role": "user", "content": "x" * 2001}]).status_code, 422)
            self.assertEqual(self._post(
                [{"role": "user", "content": "   "}]).status_code, 422)
            self.assertEqual(self.client.post(
                "/api/ai/chat", json={"messages": [
                    {"role": "user", "content": "q"}], "bogus": 1}
                ).status_code, 422)
            self.assertEqual(self._post(
                [{"role": "user", "content": "q"}],
                broker=["nope"]).status_code, 422)

    def test_oversize_assistant_echo_accepted(self):
        # Assistant echoes are server-authored context — the 2000-char cap
        # binds USER messages only (one long answer must never wedge
        # subsequent turns; final-review Issue 1).
        msgs = [{"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a" * 3000},
                {"role": "user", "content": "q2"}]
        with self._fake("fine"):
            r = self._post(msgs)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kind"], "generating")

    def test_off_state_skips_semantic_validation(self):
        with mock.patch.object(ai, "resolve_client", return_value=None):
            r = self._post([{"role": "assistant", "content": "backwards"}])
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["enabled"])

    def test_unknown_id_404(self):
        self.assertEqual(self.client.get("/api/ai/chat?id=nope").status_code,
                         404)

    def test_second_turn_hits_pack_memo(self):
        with self._fake("one"):
            cid = self._post([{"role": "user", "content": "q1"}]).json()["chat_id"]
            self.client.get(f"/api/ai/chat?id={cid}")
        with self._fake("two"), \
             mock.patch.object(hs, "load_frames",
                               side_effect=AssertionError("frames on memo hit")):
            r = self._post([{"role": "user", "content": "q1"},
                            {"role": "assistant", "content": "one"},
                            {"role": "user", "content": "q2"}])
        self.assertEqual(r.json()["kind"], "generating")

    def test_chat_post_threads_detail_fn(self):
        captured = {}
        real = ai.start_chat
        def spy(chat_id, messages, pack, *, client, detail_fn=None):
            captured["detail_fn"] = detail_fn
            return real(chat_id, messages, pack, client=client,
                        detail_fn=None)     # don't run a fake tool turn
        with self._fake("ok"), \
             mock.patch.object(ai, "start_chat", side_effect=spy):
            r = self._post([{"role": "user", "content": "hi"}])
        self.assertEqual(r.status_code, 200)
        fn = captured["detail_fn"]
        self.assertTrue(callable(fn))
        out = fn("riskcontrib", None)       # lazy frames load happens HERE
        self.assertEqual(out["topic"], "riskcontrib")
        self.assertTrue(out["available"])
        err = fn("nope", None)
        self.assertIn("error", err)

    def test_detail_fn_contains_reducer_exceptions(self):
        from terminal import server
        fn = server._chat_detail_fn(os.environ["APP_DATA_DIR"],
                                    ["all"], "all")
        with mock.patch.object(ai, "run_detail",
                               side_effect=RuntimeError("boom")):
            out = fn("income", None)
        self.assertEqual(out, {"error": "income failed: RuntimeError"})

    def test_detail_fn_memoizes_success_per_turn(self):
        # Riskcontrib costs ~21-26s on the real book; a repeat/nudge-
        # round call within the same turn must not pay full price twice.
        from terminal import server
        fn = server._chat_detail_fn(os.environ["APP_DATA_DIR"],
                                    ["all"], "all")
        with mock.patch.object(
                ai, "run_detail",
                side_effect=[{"topic": "riskcontrib", "ok": 1}]) as m:
            out1 = fn("riskcontrib", None)
            out2 = fn("riskcontrib", None)
        self.assertEqual(m.call_count, 1)
        self.assertEqual(out1, out2)

    def test_prompt_names_the_tool(self):
        self.assertIn("fetch_detail", ai._CHAT_SYSTEM)
        self.assertIn("IRA lots excluded", ai._CHAT_SYSTEM)
        for t in ("performance", "dip", "factor", "options_contracts",
                  "health"):
            self.assertIn(t, ai._CHAT_SYSTEM)


class TestChatWarmRoute(unittest.TestCase):
    """2026-08-22: POST /api/ai/chat/warm pre-builds the pack for one scope.
    Off-state before validation; memo hit -> ready; in-flight -> building
    with no second spawn; miss -> validate ids (422) then spawn -> building;
    a chat POST after a warm is frames-free."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def setUp(self):
        ai._CHAT_PACK_MEMO.clear()
        ai._CHAT_PACK_BUILDING.clear()
        ai._CHAT_PACK_FAILED.clear()
        p = mock.patch.object(ai, "_SPAWN", lambda fn: fn())
        p.start(); self.addCleanup(p.stop)
        c = mock.patch.object(ai, "resolve_client",
                              return_value=FakeClient(_FakeMsg("answer")))
        c.start(); self.addCleanup(c.stop)

    def _warm(self, **body):
        return self.client.post("/api/ai/chat/warm", json=body)

    def _dv(self):
        return ai.data_version(str(self.ddir))

    def test_miss_builds_then_ready(self):
        r = self._warm()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"enabled": True, "status": "building"})
        self.assertEqual(ai.chat_pack_state(self._dv(), "all", ["all"]), "ready")
        self.assertEqual(self._warm().json(), {"enabled": True, "status": "ready"})

    def test_in_flight_reports_building_without_second_spawn(self):
        import threading
        key = ai._chat_pack_key(self._dv(), "all", ["all"])
        lock = ai._CHAT_PACK_BUILDING.setdefault(key, threading.Lock())
        lock.acquire()
        spawned = []
        try:
            with mock.patch.object(ai, "_SPAWN", lambda fn: spawned.append(fn)):
                r = self._warm()
        finally:
            lock.release()
        self.assertEqual(r.json(), {"enabled": True, "status": "building"})
        self.assertEqual(spawned, [])

    def test_recheck_after_validation_skips_spawn(self):
        # A chat turn that raced in during load/validate built the pack:
        # the post-validation re-check answers ready and spawns nothing.
        from terminal import server
        real = server._chat_scope_frames_fn
        def build_then_return(ddir, broker, history_start):
            fn = real(ddir, broker, history_start)
            ai.chat_pack_build(fn(), self._dv(), history_start, broker)
            return fn
        spawned = []
        with mock.patch.object(server, "_chat_scope_frames_fn",
                               side_effect=build_then_return), \
             mock.patch.object(ai, "_SPAWN", lambda fn: spawned.append(fn)):
            r = self._warm()
        self.assertEqual(r.json(), {"enabled": True, "status": "ready"})
        self.assertEqual(spawned, [])

    def test_off_state_skips_validation(self):
        with mock.patch.object(ai, "resolve_client", return_value=None):
            r = self._warm(broker=["nope"], history_start="bogus")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"enabled": False})

    def test_422_shapes(self):
        self.assertEqual(self._warm(broker=["nope"]).status_code, 422)
        self.assertEqual(self._warm(history_start="1999+").status_code, 422)
        self.assertEqual(self._warm(bogus=1).status_code, 422)
        self.assertEqual(ai._CHAT_PACK_MEMO, {})          # nothing built

    def test_503_when_positions_missing(self):
        with mock.patch.object(ai, "data_version",
                               side_effect=FileNotFoundError("gone")):
            self.assertEqual(self._warm().status_code, 503)

    def test_build_failure_is_logged_not_raised(self):
        with mock.patch.object(ai, "build_chat_pack",
                               side_effect=RuntimeError("reducer boom")), \
             self.assertLogs("terminal.ai_service", level="WARNING"):
            r = self._warm()
        self.assertEqual(r.json(), {"enabled": True, "status": "building"})
        self.assertEqual(ai.chat_pack_state(self._dv(), "all", ["all"]), "failed")
        # a later render must NOT re-spawn the failing build ...
        spawned = []
        with mock.patch.object(ai, "_SPAWN", lambda fn: spawned.append(fn)):
            r2 = self._warm()
        self.assertEqual(r2.json(), {"enabled": True, "status": "failed"})
        self.assertEqual(spawned, [])
        # ... while the chat turn still tries (here: succeeds), clearing it
        r3 = self.client.post("/api/ai/chat", json={
            "messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(r3.json()["kind"], "generating")
        self.assertEqual(self._warm().json(), {"enabled": True, "status": "ready"})

    def test_chat_after_warm_is_frames_free(self):
        self._warm()
        with mock.patch.object(hs, "load_frames",
                               side_effect=AssertionError("frames on a warm hit")):
            r = self.client.post("/api/ai/chat", json={
                "messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kind"], "generating")

    def test_scope_key_honours_broker_and_history(self):
        hist = hs._history_start_options(hs.load_frames(str(self.ddir)))
        other = next(o["id"] for o in hist if o["id"] != "all")
        self._warm(history_start=other)
        self.assertEqual(ai.chat_pack_state(self._dv(), other, ["all"]), "ready")
        self.assertEqual(ai.chat_pack_state(self._dv(), "all", ["all"]), "missing")


class TestChatWarmRealThread(unittest.TestCase):
    """Real _SPAWN: a chat POST landing during an in-flight warm waits on
    the same build (one build_chat_pack call) and then proceeds."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def test_chat_waits_on_in_flight_warm_single_build(self):
        import threading
        ai._CHAT_PACK_MEMO.clear(); ai._CHAT_PACK_BUILDING.clear()
        ai._CHAT_PACK_FAILED.clear()
        gate, started = threading.Event(), threading.Event()
        real = ai.build_chat_pack
        n = []
        def slow(frames, history_start, broker):
            n.append(1); started.set(); gate.wait(10)
            return real(frames, history_start, broker)
        fake = FakeClient(_FakeMsg("late"))
        out = {}
        try:
            with mock.patch.object(ai, "build_chat_pack", side_effect=slow), \
                 mock.patch.object(ai, "resolve_client", return_value=fake):
                r = self.client.post("/api/ai/chat/warm", json={})
                self.assertEqual(r.json()["status"], "building")
                self.assertTrue(started.wait(5), "warm thread never built")
                t = threading.Thread(target=lambda: out.setdefault(
                    "r", self.client.post("/api/ai/chat", json={
                        "messages": [{"role": "user", "content": "q"}]})))
                t.start(); t.join(0.5)
                self.assertTrue(t.is_alive())       # waiting on the warm's lock
                # The POST waits on the warm's lock, then the fixture pack
                # build (~6 s idle, minutes on a loaded box): a generous join,
                # and a named failure rather than a KeyError on `out`.
                gate.set(); t.join(180)
        finally:
            gate.set()
        self.assertIn("r", out, "chat POST did not return within 180 s")
        self.assertEqual(out["r"].json()["kind"], "generating")
        self.assertEqual(n, [1])
        cid = out["r"].json()["chat_id"]
        for _ in range(50):
            d = self.client.get(f"/api/ai/chat?id={cid}").json()
            if d["kind"] != "generating":
                break
            time.sleep(0.1)
        self.assertEqual(d["kind"], "ok")


class TestChatRealThread(unittest.TestCase):
    """Real _SPAWN: a concurrent poll must see generating (not 404, not a
    half-written entry) while the API call is in flight — the B1a class of
    race that inline-_SPAWN tests structurally cannot expose."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def test_poll_sees_generating_then_ok(self):
        import threading
        gate = threading.Event()
        released = threading.Event()

        class BlockingMessages:
            def create(self, **kw):
                gate.set()
                released.wait(5)
                return _FakeMsg("late answer")

        blocking = FakeClient()
        blocking.beta.messages = BlockingMessages()
        ai._CHAT_PACK_MEMO.clear()
        try:
            with mock.patch.object(ai, "resolve_client",
                                   return_value=blocking):
                cid = self.client.post("/api/ai/chat", json={
                    "messages": [{"role": "user", "content": "q"}]
                }).json()["chat_id"]
            self.assertTrue(gate.wait(5), "background chat never started")
            d1 = self.client.get(f"/api/ai/chat?id={cid}").json()
            self.assertEqual(d1["kind"], "generating")
        finally:
            released.set()
        deadline = time.time() + 5
        while (ai.chat_status(cid) or {}).get("status") == "running" \
                and time.time() < deadline:
            time.sleep(0.02)
        d2 = self.client.get(f"/api/ai/chat?id={cid}").json()
        self.assertEqual(d2["kind"], "ok")
        self.assertEqual(d2["text"], "late answer")


class TestRoutesAsyncRealThread(unittest.TestCase):
    """Regression guard (final whole-branch review): TestRoutes patches
    _SPAWN inline, so a force job always finishes INSIDE the POST itself and
    can never expose a cache-check-before-job-status-check ordering bug --
    a warm same-dv cache would win over an in-flight forced regeneration, so
    a Regenerate poll would show the PRE-regen text. This class runs with the
    real _SPAWN (an actual daemon thread, never patched here) so a concurrent
    poll can observe the job mid-flight, still "running"."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def setUp(self):
        cache = self.ddir / "ai_cache.json"
        if cache.exists():
            cache.unlink()
        self.skey = ai.scope_key(["all"], "all", None)
        ai.clear_job("factors", self.skey)
        self.addCleanup(ai.clear_job, "factors", self.skey)

    def _fake(self, text):
        return mock.patch.object(ai, "resolve_client",
                                 return_value=FakeClient(_FakeMsg(text)))

    def test_regenerate_poll_sees_generating_not_stale_cache(self):
        import threading

        # 1. Warm the cache to "old" -- inline _SPAWN locally (started/stopped
        # around just this block) so the warm-up itself is deterministic and
        # unrelated to the real-thread behaviour under test below.
        p = mock.patch.object(ai, "_SPAWN", lambda fn: fn())
        p.start()
        try:
            with self._fake("old"):
                self.client.get("/api/ai/explain?section=factors")   # generating -> cached "old"
                r0 = self.client.get("/api/ai/explain?section=factors")
        finally:
            p.stop()
        self.assertEqual(r0.json()["kind"], "ok")
        self.assertEqual(r0.json()["text"], "old")

        # 2. Regenerate with the REAL _SPAWN (unpatched in this class) and a
        # client whose create() blocks on an Event, so the background job is
        # still "running" when we issue the concurrent poll in step 3.
        gate = threading.Event()
        released = threading.Event()

        class BlockingMessages:
            def create(self, **kw):
                gate.set()
                released.wait(5)
                return _FakeMsg("new")

        blocking_client = FakeClient()
        blocking_client.beta.messages = BlockingMessages()

        try:
            with mock.patch.object(ai, "resolve_client",
                                   return_value=blocking_client):
                rp = self.client.post("/api/ai/regenerate",
                                      json={"section": "factors"})
            self.assertEqual(rp.json()["kind"], "generating")
            self.assertTrue(gate.wait(5), "background job never started")
            st = ai.job_status("factors", self.skey)
            self.assertIsNotNone(st)
            self.assertEqual(st["status"], "running")

            # 3. REGRESSION GUARD: a concurrent poll (force=False, same dv)
            # must see the in-flight job, never the pre-regen cached "old"
            # text under kind:"ok" -- this is exactly what the pre-fix
            # ordering (cache-hit check before job-status check) gets wrong.
            r1 = self.client.get("/api/ai/explain?section=factors")
            d1 = r1.json()
            self.assertEqual(d1["kind"], "generating")
            self.assertIsNone(d1["text"])
        finally:
            released.set()   # let the background job finish regardless

        # 4. Bounded wait for the job to finish (_SPAWN discards the thread
        # handle, so a cleared job_status is the completion signal), then
        # confirm a final poll serves the NEW text.
        deadline = time.time() + 5
        while ai.job_status("factors", self.skey) is not None and time.time() < deadline:
            time.sleep(0.02)
        self.assertIsNone(ai.job_status("factors", self.skey),
                          "background job never finished")
        r2 = self.client.get("/api/ai/explain?section=factors")
        d2 = r2.json()
        self.assertEqual(d2["kind"], "ok")
        self.assertEqual(d2["text"], "new")


RISK_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
               / "terminal_ai_risk_facts_golden.json")
RISK_FILTERED_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                        / "terminal_ai_risk_facts_filtered_golden.json")


class TestRiskFacts(unittest.TestCase):
    ASOF = date(2026, 6, 28)   # pin the today-dependent surface — the
                               # test_terminal_income.ASOF convention

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai.build_facts("risk", self.frames,
                               history_start="all", broker=["all"])
        ai.scrub_gate(facts)
        golden = json.loads(RISK_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_risk_is_dimless(self):
        self.assertNotIn("dims", ai.SECTIONS["risk"])

    def test_concentration_matches_portfolio_facts(self):
        # One shared helper — the risk box must cite the same concentration
        # numbers the AI tab (and Risk tab) show.
        risk = ai.build_facts("risk", self.frames)
        port = ai.build_facts("portfolio", self.frames)
        m = _deep_close(risk["concentration"], port["concentration"])
        self.assertIsNone(m, m)

    def test_portfolio_golden_unchanged_by_extraction(self):
        facts = ai._facts_portfolio(self.frames, "all", ["all"], None,
                                    asof=self.ASOF)
        golden = json.loads(PORTFOLIO_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_window_disclosure_counts(self):
        facts = ai.build_facts("risk", self.frames)
        ra = facts["risk_adjusted"]
        self.assertLessEqual(ra["months_used_1y"], 12)
        self.assertLessEqual(ra["months_used_3y"], 36)
        if facts["daily_available"]:
            self.assertLessEqual(facts["daily"]["n_days_60d"], 60)
            self.assertLessEqual(facts["daily"]["n_days_252d"], 252)

    def test_window_disclosure_reads_short_on_filtered_scope(self):
        frames = hs.apply_global_filters(hs.load_frames(str(FIXTURE)),
                                         ["all"], "2026+")
        facts = ai.build_facts("risk", frames, history_start="2026+")
        ra = facts["risk_adjusted"]
        self.assertLess(ra["months_used_3y"], 36)
        self.assertLess(ra["months_used_1y"], 12)

    def test_filtered_risk_reads_too_short_and_names_scope(self):
        # Fixture reality: a proper class filter collapses monthly TWR to 1 row.
        facts = ai.build_facts("risk", self.frames,
                               dims={"asset_class": ["equity_stock"]})
        ai.scrub_gate(facts)
        self.assertFalse(facts["available"])
        self.assertEqual(facts["reason"], "too_short")
        self.assertEqual(facts["scope"]["asset_class"], "Individual Stocks")

    def test_filtered_risk_golden(self):
        facts = ai.build_facts("risk", self.frames,
                               dims={"asset_class": ["equity_stock"]})
        golden = json.loads(RISK_FILTERED_GOLDEN.read_text(encoding="utf-8"))
        self.assertIsNone(_deep_close(facts, golden))

    def test_all_accounts_selected_is_available_and_named(self):
        # filter_requested=True but account_active=False -> whole-book monthly
        # count (40), still names the accounts in scope; NOT asserted byte-equal
        # to the None-based whole-book facts (spec §2 edge).
        facts = ai.build_facts("risk", self.frames,
                               dims={"account": ["test_a", "test_b",
                                                 "test_c"]})
        self.assertTrue(facts["available"])
        self.assertIn("account", facts["scope"])
        self.assertEqual(facts["risk_adjusted"]["months_used_3y"], 36)

    def test_whole_book_scope_has_no_filter_keys(self):
        facts = ai.build_facts("risk", self.frames)
        self.assertNotIn("account", facts["scope"])
        self.assertNotIn("asset_class", facts["scope"])

    def test_stress_block(self):
        f = ai._facts_risk(self.frames, "all", None, None)
        st = f["stress"]
        self.assertIn("available", st)
        if st["available"]:
            for sc in st["scenarios"]:
                self.assertEqual(set(sc), {"window", "spy_drop_pct",
                                           "implied_drop_pct"})
        ai.scrub_gate(f)


RC_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
             / "terminal_ai_riskcontrib_facts_golden.json")
RC_FILTERED_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                      / "terminal_ai_riskcontrib_facts_filtered_golden.json")


class TestRiskContribFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai.build_facts("riskcontrib", self.frames,
                               history_start="all", broker=["all"])
        ai.scrub_gate(facts)
        golden = json.loads(RC_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_declares_estimator_benchmark_dims(self):
        self.assertEqual(ai.SECTIONS["riskcontrib"]["dims"],
                         ("estimator", "benchmark"))

    def test_estimator_dim_honored(self):
        from terminal import riskcontrib_service as rcs
        view = rcs.build_riskcontrib_view(self.frames)
        ests = [e["id"] for e in view["controls"]["estimators"]]
        if len(ests) < 2:
            self.skipTest("single estimator")
        facts = ai.build_facts("riskcontrib", self.frames,
                               dims={"estimator": ests[1]})
        self.assertEqual(facts["estimator"], ests[1])

    def test_bad_dim_values_raise(self):
        with self.assertRaises(ai.AIDimError):
            ai.build_facts("riskcontrib", self.frames,
                           dims={"estimator": "bogus"})
        with self.assertRaises(ai.AIDimError):
            ai.build_facts("riskcontrib", self.frames,
                           dims={"benchmark": "bogus"})

    def test_invariance_no_untracked_seg_values(self):
        # Spec §3.2: ES-confidence and downside-threshold are UNTRACKED —
        # nothing alpha- or threshold-dependent may appear in the facts.
        facts = ai.build_facts("riskcontrib", self.frames)

        def walk(o, path="root"):
            if isinstance(o, dict):
                for k, v in o.items():
                    kl = str(k).lower()
                    for bad in ("es_", "shortfall", "threshold", "downside"):
                        self.assertNotIn(bad, kl, f"{path}.{k}")
                    walk(v, f"{path}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
        walk(facts)
        self.assertNotIn("sample_warnings", facts)

    def test_whole_book_golden_unchanged(self):
        # B2 must not move the whole-book RC facts.
        facts = ai.build_facts("riskcontrib", self.frames,
                               history_start="all", broker=["all"])
        golden = json.loads(RC_GOLDEN.read_text(encoding="utf-8"))
        self.assertIsNone(_deep_close(facts, golden))

    def test_class_filter_honored_and_named_in_scope(self):
        from terminal import riskcontrib_service as rcs
        facts = ai.build_facts("riskcontrib", self.frames,
                               dims={"asset_class": ["equity_stock"]})
        ai.scrub_gate(facts)                          # labels must be scrub-safe
        self.assertTrue(facts["available"])
        self.assertEqual(facts["scope"]["asset_class"], "Individual Stocks")
        # box == tab: the reduced contributors come from the filtered view.
        ref = rcs.build_riskcontrib_view(self.frames, asset_class=["equity_stock"])
        ref_est = ref["controls"]["estimators"][0]["id"]
        ref_a = ref["controls"]["es_levels"][0]["id"]
        ref_t = ref["controls"]["thresholds"][0]["id"]
        ref_syms = ref["combos"][f"{ref_est}|{ref_a}|{ref_t}"]["weight_vs_pctr"]["symbols"]
        self.assertEqual(len(facts["top_risk_contributors"]), min(5, len(ref_syms)))

    def test_class_filtered_golden(self):
        facts = ai.build_facts("riskcontrib", self.frames,
                               dims={"asset_class": ["equity_stock"]})
        golden = json.loads(RC_FILTERED_GOLDEN.read_text(encoding="utf-8"))
        self.assertIsNone(_deep_close(facts, golden))

    def test_account_filter_names_scope(self):
        facts = ai.build_facts("riskcontrib", self.frames,
                               dims={"account": ["test_a"]})
        self.assertEqual(facts["scope"]["account"], "TEST-A")

    def test_whole_book_scope_has_no_filter_keys(self):
        facts = ai.build_facts("riskcontrib", self.frames)
        self.assertNotIn("account", facts["scope"])
        self.assertNotIn("asset_class", facts["scope"])

    def test_contributors_carry_return_pairing(self):
        f = ai._facts_riskcontrib(self.frames, "all", None, None)
        rows = f["top_risk_contributors"]
        self.assertTrue(rows)
        for r in rows:
            self.assertIn("contrib_252d_pp", r)
        ai.scrub_gate(f)


TAX_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_ai_tax_facts_golden.json")


class TestTaxFacts(unittest.TestCase):
    ASOF = date(2026, 6, 28)   # pin the today-dependent surface — the
                               # test_terminal_income.ASOF convention

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai._facts_tax(self.frames, "all", ["all"], None,
                              asof=self.ASOF)
        ai.scrub_gate(facts)
        golden = json.loads(TAX_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_asof_pins_ripening_window(self):
        # The seam must reach build_tax_view: the TEST-A SPY lot acquired
        # 2025-11-03 turns long-term after 2026-11-03, so it sits outside
        # the 60-day ripening window at the pinned ASOF and inside it by
        # 2026-09-10. Without the thread the count follows date.today()
        # and would have flipped the golden around 2026-09-05.
        pinned = ai._facts_tax(self.frames, "all", ["all"], None,
                               asof=self.ASOF)
        later = ai._facts_tax(self.frames, "all", ["all"], None,
                              asof=date(2026, 9, 10))
        self.assertEqual(pinned["ripening_to_long_within_60d"], 0)
        self.assertGreaterEqual(later["ripening_to_long_within_60d"], 1)

    def test_tax_is_dimless(self):
        self.assertNotIn("dims", ai.SECTIONS["tax"])

    def test_no_dollar_leaves_anywhere(self):
        # Belt-and-braces beyond scrub_gate: no key from the tax view's
        # dollar families may survive the reduction at any depth.
        facts = ai.build_facts("tax", self.frames)

        def walk(o, path="root"):
            if isinstance(o, dict):
                for k, v in o.items():
                    self.assertNotIn(str(k), ("gains", "losses", "net",
                                              "basis", "market_value",
                                              "unrealized_gl"), path)
                    walk(v, f"{path}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
        walk(facts)

    def test_unavailable_without_lots(self):
        with tempfile.TemporaryDirectory() as td:
            dd = Path(td) / "d"
            shutil.copytree(FIXTURE, dd)
            (dd / "lots.csv").unlink()
            frames = hs.apply_global_filters(
                hs.load_frames(str(dd)), ["all"], "all")
            facts = ai.build_facts("tax", frames)
            self.assertFalse(facts["available"])
            self.assertTrue(facts["reason"])

    def test_unrealized_matches_portfolio_tax_posture(self):
        # One shared helper — same numbers as the AI tab's tax_posture.
        tax = ai.build_facts("tax", self.frames)
        port = ai.build_facts("portfolio", self.frames)
        m = _deep_close(tax["unrealized"], port["tax_posture"])
        self.assertIsNone(m, m)


TAX_DETAIL_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                     / "terminal_ai_tax_detail_facts_golden.json")


class TestTaxDetailFacts(unittest.TestCase):
    """Chat-only per-ticker tax facts (spec 2026-08-22): wash calendar from
    transactions, harvest grouped by ticker, actionable lot detail."""
    ASOF = date(2026, 6, 28)     # the TestTaxFacts pin — calendar-stable

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")
        cls.facts = ai._facts_tax_detail(cls.frames, "all", ["all"], None,
                                         asof=cls.ASOF)

    def test_shape_and_scrub(self):
        f = self.facts
        ai.scrub_gate(f)
        self.assertEqual(f["section"], "tax_detail")
        self.assertTrue(f["available"])
        self.assertEqual(set(f["wash_calendar"]),
                         {"as_of", "window_days", "tx_frontier",
                          "unkeyed_rows_omitted", "note", "tickers"})
        self.assertEqual(f["wash_calendar"]["as_of"], "2026-06-28")
        self.assertEqual(f["wash_calendar"]["tx_frontier"], "2026-04-15")
        # the fixture has no trades inside the pinned window
        self.assertEqual(f["wash_calendar"]["tickers"], [])
        self.assertEqual(set(f["harvest"]),
                         {"available", "candidates", "candidates_omitted",
                          "window_observed_pct", "clear_means", "tickers"})
        self.assertEqual(set(f["lots"]), {"available", "coverage_note",
                                          "tickers"})
        self.assertEqual(set(f["ledger"]), {"stale", "stale_reason_present"})
        json.dumps(f, allow_nan=False)

    def test_golden(self):
        golden = json.loads(TAX_DETAIL_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(self.facts, golden)
        self.assertIsNone(m, m)

    def test_lots_block_is_actionable_only(self):
        lots = self.facts["lots"]
        self.assertIn("of", lots["coverage_note"])
        # Compaction rider: empty-valued keys are omitted per row, so the
        # shape is core-keys-always plus optional non-empty extras.
        for t in lots["tickers"]:
            self.assertLessEqual({"ticker", "accounts", "lots"}, set(t))
            self.assertLessEqual(set(t), {"ticker", "accounts", "lots",
                                          "short_lots", "long_lots",
                                          "marked_lots",
                                          "min_days_to_long_term",
                                          "unrealized_direction",
                                          "ripening_within_60d"})

    def test_calendar_survives_a_missing_ledger(self):
        # The calendar is the ledger-independent part: with lots.csv absent
        # the harvest/lots blocks degrade, the section stays available.
        with tempfile.TemporaryDirectory() as td:
            for name in ("transactions.csv", "positions.csv"):
                shutil.copy(FIXTURE / name, Path(td) / name)
            frames = dataclasses.replace(self.frames, data_dir=td)
            f = ai._facts_tax_detail(frames, "all", ["all"], None,
                                     asof=self.ASOF)
        self.assertTrue(f["available"])
        self.assertEqual(f["wash_calendar"]["tx_frontier"], "2026-04-15")
        self.assertFalse(f["harvest"]["available"])
        self.assertFalse(f["lots"]["available"])
        ai.scrub_gate(f)

    def test_asof_is_forwarded_to_the_tax_view(self):
        later = ai._facts_tax_detail(self.frames, "all", ["all"], None,
                                     asof=date(2026, 9, 10))
        # TestTaxFacts pins the ripening flip between these two dates
        # (absent key == 0 under the compaction rider)
        pinned_total = sum(t.get("ripening_within_60d", 0)
                           for t in self.facts["lots"]["tickers"])
        later_total = sum(t.get("ripening_within_60d", 0)
                          for t in later["lots"]["tickers"])
        self.assertEqual(pinned_total, 0)
        self.assertGreaterEqual(later_total, 1)

    def test_reductions_on_a_constructed_tax_view(self):
        cands = [
            {"symbol": "AAA", "instrument_key": "AAA", "account_label": "Acct Two",
             "term": "short", "wash_status": "clear", "window_ends": "2026-07-10",
             "blocking_buys": [], "is_ira_blocked": False},
            {"symbol": "AAA", "instrument_key": "AAA", "account_label": "Acct One",
             "term": "long", "wash_status": "blocked", "window_ends": "2026-07-20",
             "blocking_buys": [{"date": "2026-06-20"}, {"date": "2026-06-20"}],
             "is_ira_blocked": True},
            {"symbol": "BBB", "instrument_key": "BBB", "account_label": "Acct One",
             "term": "long", "wash_status": "clear", "window_ends": "2026-07-01",
             "blocking_buys": [], "is_ira_blocked": False},
            {"symbol": "", "instrument_key": "912828X88", "account_label": "Acct One",
             "term": "short", "wash_status": "clear", "window_ends": "2026-07-01",
             "blocking_buys": [], "is_ira_blocked": False},
            {"symbol": "", "instrument_key": "CASH REINVESTMENT SWEEP",
             "account_label": "Acct One", "term": "short", "wash_status": "clear",
             "window_ends": "2026-07-01", "blocking_buys": [],
             "is_ira_blocked": False},
            {"symbol": "CCC", "instrument_key": "CCC", "account_label": "Acct One",
             "term": "short", "wash_status": "clear", "window_ends": "2026-07-05",
             "blocking_buys": [], "is_ira_blocked": False},
        ]
        lots = [
            {"symbol": "AAA", "instrument_key": "AAA", "account_label": "Acct One",
             "term": "short", "days_to_long_term": 10, "unrealized_gl": 5.0},
            {"symbol": "AAA", "instrument_key": "AAA", "account_label": "Acct Two",
             "term": "short", "days_to_long_term": 200, "unrealized_gl": None},
            {"symbol": "AAA", "instrument_key": "AAA", "account_label": "Acct One",
             "term": "long", "days_to_long_term": None, "unrealized_gl": 3.0},
            {"symbol": "BBB", "instrument_key": "BBB", "account_label": "Acct One",
             "term": "long", "days_to_long_term": None, "unrealized_gl": None},
            {"symbol": "", "instrument_key": "912828X88", "account_label": "Acct One",
             "term": "short", "days_to_long_term": 30, "unrealized_gl": 1.0},
            {"symbol": "", "instrument_key": "CASH REINVESTMENT SWEEP",
             "account_label": "Acct One", "term": "short",
             "days_to_long_term": 30, "unrealized_gl": 1.0},
            # CCC: a short LOSS lot flipping long-term TODAY — min_days 0
            # is a meaningful zero the omission must keep; actionable via
            # its harvest candidacy above
            {"symbol": "CCC", "instrument_key": "CCC",
             "account_label": "Acct One", "term": "short",
             "days_to_long_term": 0, "unrealized_gl": -2.0},
        ]
        fake = {"kind": "tax", "meta": {"stale": False, "stale_reason": None},
                "harvest": {"candidates": cands, "summary": {"candidates": 5},
                            "semantics": {"window_observed_pct": 29.0,
                                          "clear_means": "observed only"}},
                "lots": lots}
        with mock.patch.object(ai.txs, "build_tax_view", return_value=fake):
            f = ai._facts_tax_detail(self.frames, "all", ["all"], None,
                                     asof=self.ASOF)
        ai.scrub_gate(f)
        h = f["harvest"]
        self.assertTrue(h["available"])
        self.assertEqual(h["candidates"], 5)
        self.assertEqual(h["candidates_omitted"], 2)
        self.assertEqual([t["ticker"] for t in h["tickers"]],
                         ["AAA", "BBB", "CCC"])
        aaa = h["tickers"][0]
        self.assertEqual(aaa["accounts"], ["Acct One", "Acct Two"])
        self.assertEqual(aaa["terms"], ["long", "short"])
        self.assertEqual(aaa["wash_status"], "blocked")
        self.assertEqual(aaa["window_ends"], "2026-07-20")
        self.assertEqual(aaa["blocking_buy_dates"], ["2026-06-20"])
        self.assertTrue(aaa["ira_blocked"])
        bbb = h["tickers"][1]
        # Compaction rider: BBB's empty harvest values leave the row
        self.assertNotIn("blocking_buy_dates", bbb)
        self.assertNotIn("ira_blocked", bbb)
        self.assertEqual(bbb["wash_status"], "clear")
        l = f["lots"]
        self.assertTrue(l["available"])
        self.assertEqual([t["ticker"] for t in l["tickers"]],
                         ["AAA", "BBB", "CCC"])
        a = l["tickers"][0]
        self.assertEqual((a["lots"], a["short_lots"], a["long_lots"],
                          a["marked_lots"]), (3, 2, 1, 2))
        self.assertEqual(a["min_days_to_long_term"], 10)
        self.assertEqual(a["ripening_within_60d"], 1)
        self.assertEqual(a["unrealized_direction"], "gain")
        b = l["tickers"][1]
        # Compaction rider: BBB is one long unmarked lot — every zero/None
        # key is absent, the non-empty ones stay
        self.assertEqual((b["lots"], b["long_lots"]), (1, 1))
        for absent in ("short_lots", "marked_lots", "min_days_to_long_term",
                       "unrealized_direction", "ripening_within_60d"):
            self.assertNotIn(absent, b)
        ccc = l["tickers"][2]
        # a meaningful zero survives the omission
        self.assertEqual(ccc["min_days_to_long_term"], 0)
        self.assertEqual(ccc["unrealized_direction"], "loss")
        self.assertNotIn("ripening_within_60d", ccc)
        self.assertIn("covers 3 of 3 ticker-keyed positions", l["coverage_note"])
        self.assertIn("(2 cusip-keyed bills/notes or unnamed positions "
                      "excluded)", l["coverage_note"])
        self.assertIn("absent row keys mean zero", l["coverage_note"])

    def test_harvest_unavailable_reason_when_scan_did_not_run(self):
        # tax_ok True (kind: tax) but the harvest sub-block itself never
        # produced a summary — distinct from a tax_view-level error, which
        # carries tax_reason instead.
        fake = {"kind": "tax", "meta": {"stale": False, "stale_reason": None},
                "harvest": {"candidates": [], "summary": {}, "semantics": {},
                            "unavailable": "x"},
                "lots": []}
        with mock.patch.object(ai.txs, "build_tax_view", return_value=fake):
            f = ai._facts_tax_detail(self.frames, "all", ["all"], None,
                                     asof=self.ASOF)
        self.assertFalse(f["harvest"]["available"])
        self.assertEqual(f["harvest"]["reason"], "harvest_scan_unavailable")

    def test_interim_file_feeds_the_calendar(self):
        header = ("settlement_date,trade_date,broker,account_id,transaction_type,"
                  "symbol,cusip,description,quantity,price,amount,source_file,"
                  "flow_scope,pair_id\n")
        rows = ("2026-06-20,2026-06-20,alpine,TEST-A,buy,AAA,,Synthetic Equity A,"
                "5,90.00,-450.00,interim,,\n"
                "2026-06-25,2026-06-25,alpine,TEST-A,sell,AAA,,Synthetic Equity A,"
                "-2,95.00,190.00,interim,,\n")
        with tempfile.TemporaryDirectory() as td:
            for name in ("transactions.csv", "positions.csv"):
                shutil.copy(FIXTURE / name, Path(td) / name)
            (Path(td) / "transactions_interim.csv").write_text(header + rows,
                                                                encoding="utf-8")
            frames = dataclasses.replace(self.frames, data_dir=td)
            f = ai._facts_tax_detail(frames, "all", ["all"], None,
                                     asof=self.ASOF)
        ai.scrub_gate(f)
        cal = f["wash_calendar"]
        self.assertEqual(cal["tx_frontier"], "2026-06-25")
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["AAA"])
        t = cal["tickers"][0]
        self.assertEqual(t["acquisitions_in_window"], 1)
        self.assertEqual(t["wash_if_sold_before"], "2026-07-20")
        self.assertEqual(t["sells_in_window"], 1)
        self.assertEqual(t["wash_if_rebought_before"], "2026-07-25")
        self.assertFalse(f["harvest"]["available"])     # no ledger in td
        self.assertIn("reason", f["harvest"])
        self.assertFalse(f["lots"]["available"])

    def test_sleeve_only_calendar_names_are_not_lot_actionable(self):
        # Sleeve rider: a ticker whose only in-window trades are the
        # direct-index sleeve's own churn must not drag its lots into the
        # actionable lot detail via the traded-in-window arm. The
        # candidates/ripening arms are untouched.
        fake_cal = {"as_of": str(self.ASOF), "window_days": 30,
                    "tx_frontier": str(self.ASOF),
                    "unkeyed_rows_omitted": 0, "note": "n",
                    "tickers": [{"ticker": "SLVX", "accounts": ["Direct Index"],
                                 "acquisitions_in_window": 2,
                                 "sleeve_only": True},
                                {"ticker": "MIXD", "accounts": ["Acct One"],
                                 "acquisitions_in_window": 1}]}
        lots = [
            {"symbol": "SLVX", "instrument_key": "SLVX",
             "account_label": "Direct Index", "term": "long",
             "days_to_long_term": None, "unrealized_gl": 1.0},
            {"symbol": "MIXD", "instrument_key": "MIXD",
             "account_label": "Acct One", "term": "long",
             "days_to_long_term": None, "unrealized_gl": 1.0},
        ]
        fake_tv = {"kind": "tax", "meta": {"stale": False,
                                           "stale_reason": None},
                   "harvest": {"candidates": [], "summary": {"candidates": 0},
                               "semantics": {}},
                   "lots": lots}
        with mock.patch.object(ai.txs, "wash_calendar",
                               return_value=fake_cal), \
             mock.patch.object(ai.txs, "build_tax_view",
                               return_value=fake_tv):
            f = ai._facts_tax_detail(self.frames, "all", ["all"], None,
                                     asof=self.ASOF)
        self.assertEqual([t["ticker"] for t in f["lots"]["tickers"]],
                         ["MIXD"])
        self.assertIn("sleeve-only", f["lots"]["coverage_note"])

    def test_sleeve_accounts_come_from_config(self):
        with mock.patch.object(ai.hs.cfg, "TLH_ACCOUNT_ID", "SLV-9",
                               create=True):
            self.assertEqual(ai._sleeve_account_ids(), frozenset({"SLV-9"}))
        with mock.patch.object(ai.hs.cfg, "TLH_ACCOUNT_ID", "",
                               create=True):
            self.assertEqual(ai._sleeve_account_ids(), frozenset())


class TestChatPackTaxViewSeam(unittest.TestCase):
    """Compaction rider: the pack built the tax view three times
    (portfolio posture / tax / tax_detail). build_chat_pack now builds it
    once and threads it via the reducers' tax_view= keyword (the #262
    bundle= seam shape); reducers without the keyword still self-build,
    so every asof-pinned test and the box path are untouched."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_pack_builds_the_tax_view_exactly_once(self):
        real = txs.build_tax_view
        calls = []

        def counting(*a, **k):
            calls.append(k)
            return real(*a, **k)

        with mock.patch.object(ai.txs, "build_tax_view",
                               side_effect=counting):
            ai.build_chat_pack(self.frames, "all", ["all"])
        self.assertEqual(len(calls), 1)

    def test_seam_parity_per_reducer(self):
        asof = date(2026, 6, 28)
        tv = txs.build_tax_view(self.frames, self.frames.data_dir,
                                broker=["all"], asof=asof)
        for fn in (ai._facts_portfolio, ai._facts_tax,
                   ai._facts_tax_detail):
            without = fn(self.frames, "all", ["all"], None, asof=asof)
            with_tv = fn(self.frames, "all", ["all"], None, asof=asof,
                         tax_view=tv)
            m = _deep_close(without, with_tv)
            self.assertIsNone(m, f"{fn.__name__}: {m}")


HOLDINGS_DETAIL_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                          / "terminal_ai_holdings_detail_facts_golden.json")


class TestHoldingsDetailFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")
        cls.facts = ai._facts_holdings_detail(cls.frames, "all", ["all"], None)

    def test_pct_label_parsing(self):
        self.assertEqual(ai._pct_label_to_num("75.2%"), 75.2)
        self.assertEqual(ai._pct_label_to_num("-3.0%"), -3.0)
        self.assertEqual(ai._pct_label_to_num("1,234.5%"), 1234.5)
        self.assertIsNone(ai._pct_label_to_num("—"))
        self.assertIsNone(ai._pct_label_to_num(None))
        self.assertIsNone(ai._pct_label_to_num("inf%"))
        self.assertIsNone(ai._pct_label_to_num("nan%"))

    def test_shape_scrub_and_golden(self):
        f = self.facts
        ai.scrub_gate(f)
        self.assertEqual(f["section"], "holdings_detail")
        self.assertTrue(f["available"])
        self.assertTrue(f["positions"])
        for p in f["positions"]:
            self.assertEqual(set(p), {"ticker", "account", "class",
                                      "weight_pct", "unrealized_pct",
                                      "unrealized_direction"})
        # weights describe the scoped book and sum to ~100
        self.assertAlmostEqual(sum(p["weight_pct"] for p in f["positions"])
                               + f["omitted_weight_pct"], 100.0, delta=0.5)
        self.assertEqual(f["omitted_weight_pct"], 0.0)
        json.dumps(f, allow_nan=False)
        golden = json.loads(HOLDINGS_DETAIL_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(f, golden)
        self.assertIsNone(m, m)

    def test_occ_option_rows_list_under_their_underlying(self):
        tickers = [p["ticker"] for p in self.facts["positions"]]
        self.assertNotIn("SPY260918P00500000", tickers)
        self.assertEqual(tickers.count("SPY"), 3)      # two SPY ETF rows + the put
        self.assertTrue(all(not re.search(r"\d{5,}", t) for t in tickers))

    def test_omission_remap_and_sanitizing_on_a_constructed_view(self):
        fake = {"meta": {"as_of": "2026-04-30"}, "positions": {"rows": [
            {"symbol": "AAA", "account": "Other (X99-12345678)", "class_label": "Stocks",
             "weight_pct": 97.7, "ugl_pct": "1.0%", "ugl_pct_dir": "up"},
            {"symbol": "SPY DEC 26 PUT 650.00", "account": "Acct",
             "class_label": "Options (puts)", "weight_pct": 0.5,
             "ugl_pct": "—", "ugl_pct_dir": "flat"},
            {"symbol": "US912828P12345", "account": "Acct", "class_label": "Stocks",
             "weight_pct": 0.3, "ugl_pct": "—", "ugl_pct_dir": "flat"},
            {"symbol": "912828X88", "account": "Acct", "class_label": "Treasuries",
             "weight_pct": 1.5, "ugl_pct": "—", "ugl_pct_dir": "flat"}]}}
        with mock.patch.object(ai.hs, "build_holdings_view", return_value=fake):
            f = ai._facts_holdings_detail(self.frames, "all", ["all"], None)
        ai.scrub_gate(f)
        self.assertEqual([p["ticker"] for p in f["positions"]], ["AAA", "SPY"])
        self.assertEqual(f["positions"][0]["account"], "unlabeled account")
        self.assertIsNone(f["positions"][1]["unrealized_pct"])
        self.assertEqual(f["omitted_weight_pct"], 1.8)
        self.assertTrue(f["available"])
        self.assertNotIn("reason", f)

    def test_all_omitted_is_unavailable_with_a_reason(self):
        fake = {"meta": {"as_of": "2026-04-30"}, "positions": {"rows": [
            {"symbol": "912828X88", "account": "Acct", "class_label": "Treasuries",
             "weight_pct": 100.0, "ugl_pct": "—", "ugl_pct_dir": "flat"}]}}
        with mock.patch.object(ai.hs, "build_holdings_view", return_value=fake):
            f = ai._facts_holdings_detail(self.frames, "all", ["all"], None)
        ai.scrub_gate(f)
        self.assertFalse(f["available"])
        self.assertEqual(f["reason"], "no_nameable_positions")
        self.assertEqual(f["omitted_weight_pct"], 100.0)


class TestPortfolioBenchmarkAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_declares_benchmark_dim(self):
        self.assertEqual(ai.SECTIONS["portfolio"].get("dims"), ("benchmark",))

    def test_default_is_spy(self):
        facts = ai.build_facts("portfolio", self.frames,
                               history_start="all", broker=["all"])
        self.assertEqual(facts["benchmark"]["id"], "spy")
        self.assertEqual(facts["benchmark"]["label"], hs.BENCHMARKS["spy"])

    def test_explicit_60_40(self):
        facts = ai.build_facts("portfolio", self.frames, history_start="all",
                               broker=["all"], dims={"benchmark": "60_40"})
        self.assertEqual(facts["benchmark"]["id"], "60_40")

    def test_60_40_window_numbers_differ_from_spy(self):
        spy = ai.build_facts("portfolio", self.frames, history_start="all",
                             broker=["all"], dims={"benchmark": "spy"})
        b60 = ai.build_facts("portfolio", self.frames, history_start="all",
                             broker=["all"], dims={"benchmark": "60_40"})
        sfull = next(w for w in spy["windows"] if w["window"] == "Full history")
        bfull = next(w for w in b60["windows"] if w["window"] == "Full history")
        self.assertNotEqual(sfull["benchmark"]["twr_cum_pct"],
                            bfull["benchmark"]["twr_cum_pct"])
        # portfolio side is benchmark-independent → unchanged
        self.assertEqual(sfull["portfolio"], bfull["portfolio"])

    def test_five_windows_including_5y(self):
        facts = ai.build_facts("portfolio", self.frames, history_start="all",
                               broker=["all"])
        self.assertEqual([w["window"] for w in facts["windows"]],
                         ["Full history", "5y", "3y", "1y", "YTD"])
        five = next(w for w in facts["windows"] if w["window"] == "5y")
        # fixture has 27 aligned months < 60 -> honest unavailable
        self.assertFalse(five["available"])
        self.assertEqual(five["requested_months"], 60)

    def test_bench_side_key_renamed(self):
        facts = ai.build_facts("portfolio", self.frames, history_start="all",
                               broker=["all"])
        full = next(w for w in facts["windows"] if w["available"])
        self.assertIn("benchmark", full)
        self.assertNotIn("spy", full)
        self.assertIn("bench_pct", facts["latest_month"])
        self.assertNotIn("spy_pct", facts["latest_month"])

    def test_untracked_facts_invariant_to_benchmark(self):
        # concentration / cash / income / tax are benchmark-INDEPENDENT and
        # must not move when only the benchmark dim changes (campaign
        # invariance rule).
        spy = ai.build_facts("portfolio", self.frames, history_start="all",
                             broker=["all"], dims={"benchmark": "spy"})
        b60 = ai.build_facts("portfolio", self.frames, history_start="all",
                             broker=["all"], dims={"benchmark": "60_40"})
        for k in ("concentration", "income", "tax_posture", "cash_weight_pct"):
            self.assertEqual(spy.get(k), b60.get(k), k)

    def test_bad_benchmark_value_raises_dim_error(self):
        with self.assertRaises(ai.AIDimError):
            ai.build_facts("portfolio", self.frames, history_start="all",
                           broker=["all"], dims={"benchmark": "bogus"})

    def test_60_40_falls_back_to_spy_when_agg_absent(self):
        # M4: _facts_portfolio degrades 60/40 -> SPY when the AGG bond leg is
        # missing. The GET route resolves this before build_facts; a direct
        # 60/40 regenerate reaches build_facts unresolved, so the fallback must
        # live here too (Frames is frozen -> replace agg_tr with an empty frame).
        frames = dataclasses.replace(self.frames,
                                     agg_tr=self.frames.agg_tr.iloc[0:0])
        facts = ai.build_facts("portfolio", frames, history_start="all",
                               broker=["all"], dims={"benchmark": "60_40"})
        self.assertEqual(facts["benchmark"]["id"], "spy")
        self.assertEqual(facts["benchmark"]["label"], hs.BENCHMARKS["spy"])

    def test_spy_window_numbers_unchanged(self):
        # regression guard: the existing windows' portfolio-side numbers do
        # not move (benchmark side is spy by default).
        facts = ai.build_facts("portfolio", self.frames, history_start="all",
                               broker=["all"])
        full = next(w for w in facts["windows"] if w["window"] == "Full history")
        self.assertEqual(full["portfolio"]["twr_cum_pct"], 49.41)
        self.assertEqual(full["benchmark"]["twr_cum_pct"], 13.45)


class TestPortfolioRouteBenchmark(unittest.TestCase):
    """GET /api/ai/portfolio benchmark param: 422 validation, auto
    resolution, AGG-missing fallback, and benchmark-keyed scope_key
    separation. Mirrors the TestRoutes harness (temp APP_DATA_DIR copy +
    patched resolve_client) so cache writes never touch the committed
    fixture and no test ever reaches the real Anthropic key/network."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def setUp(self):
        cache = self.ddir / "ai_cache.json"
        if cache.exists():
            cache.unlink()

    def _fake(self, text=TestRoutes.PORT_JSON):
        return mock.patch.object(ai, "resolve_client",
                                 return_value=FakeClient(_FakeMsg(text)))

    def test_unknown_benchmark_422(self):
        r = self.client.get("/api/ai/portfolio?benchmark=bogus")
        self.assertEqual(r.status_code, 422)

    def test_all_is_not_a_benchmark_422(self):
        r = self.client.get("/api/ai/portfolio?benchmark=all")
        self.assertEqual(r.status_code, 422)

    def test_explicit_60_40_in_meta(self):
        with self._fake():
            r = self.client.get("/api/ai/portfolio?benchmark=60_40")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["benchmark"]["id"], "60_40")

    def test_auto_stays_spy_for_non_bond_heavy_harbor_scope(self):
        # auto is composition-driven (majority-fixed-income scopes pull the
        # blend); the fixture's harbor book is ~35% fixed income, so its
        # scoped auto still compares against SPY.
        with self._fake():
            j = self.client.get("/api/ai/portfolio").json()
            harbor = next(o["id"] for o in j["meta"]["brokers"]
                       if o["id"].lower() == "harbor")
            r = self.client.get(
                f"/api/ai/portfolio?broker={harbor}&benchmark=auto")
        self.assertEqual(r.json()["meta"]["benchmark"]["id"], "spy")

    def test_scope_key_separates_spy_and_60_40(self):
        self.assertNotEqual(
            ai.scope_key(["all"], "all", {"benchmark": "spy"}),
            ai.scope_key(["all"], "all", {"benchmark": "60_40"}))

    def test_agg_missing_falls_back_to_spy(self):
        # temp data dir WITHOUT benchmark_agg_tr.csv -> 60/40 degrades to spy
        d = tempfile.mkdtemp()
        try:
            shutil.copytree(str(FIXTURE), os.path.join(d, "synth_data"))
            os.remove(os.path.join(d, "synth_data", "benchmark_agg_tr.csv"))
            with mock.patch.dict(os.environ,
                                 {"APP_DATA_DIR": os.path.join(d, "synth_data")}):
                from fastapi.testclient import TestClient
                from terminal import server as srv
                c = TestClient(srv.app)
                with self._fake():
                    r = c.get("/api/ai/portfolio?benchmark=60_40")
                self.assertEqual(r.status_code, 200)
                m = r.json()["meta"]["benchmark"]
                self.assertEqual(m["id"], "spy")
                self.assertTrue(m["unavailable_fallback"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class _RouteFilterHarness:
    """Shared fixture-copy + inline-_SPAWN + fake-client harness for the
    filter-threading route classes (B2 + B3 — the duplication was a
    deferred #338 Minor). A mixin, not a TestCase: unittest must not
    discover it standalone. Each subclass gets its own temp copy of the
    fixture via setUpClass (per-class, so cache files never cross)."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.ddir = Path(cls.td.name) / "synth_data"
        shutil.copytree(FIXTURE, cls.ddir)
        os.environ["APP_DATA_DIR"] = str(cls.ddir)
        from fastapi.testclient import TestClient
        from terminal import server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)
        cls.td.cleanup()

    def setUp(self):
        cache = self.ddir / "ai_cache.json"
        if cache.exists():
            cache.unlink()
        p = mock.patch.object(ai, "_SPAWN", lambda fn: fn())   # inline
        p.start(); self.addCleanup(p.stop)

    def _fake(self, text="canned"):
        return mock.patch.object(ai, "resolve_client",
                                 return_value=FakeClient(_FakeMsg(text)))


class TestRoutesFilterThreading(_RouteFilterHarness, unittest.TestCase):
    """B2: account/asset_class threaded into the risk/riskcontrib boxes."""

    def test_filtered_rc_generating_then_cached(self):
        with self._fake("rc filtered"):
            r1 = self.client.get("/api/ai/explain?section=riskcontrib"
                                 "&asset_class=equity_stock")
            self.assertEqual(r1.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            r2 = self.client.get("/api/ai/explain?section=riskcontrib"
                                 "&asset_class=equity_stock")
        self.assertEqual(r2.json()["kind"], "ok")
        self.assertEqual(r2.json()["text"], "rc filtered")

    def test_filtered_and_whole_book_cache_separately(self):
        with self._fake("whole book"):
            self.client.get("/api/ai/explain?section=riskcontrib")
            self.client.get("/api/ai/explain?section=riskcontrib")   # -> ok cached
        with self._fake("filtered"):
            # different scope -> a miss -> generating (not the whole-book text)
            r = self.client.get("/api/ai/explain?section=riskcontrib"
                                "&asset_class=equity_stock")
            self.assertEqual(r.json()["kind"], "generating")

    def test_bogus_account_422(self):
        with self._fake():
            r = self.client.get("/api/ai/explain?section=risk&account=nope")
        self.assertEqual(r.status_code, 422)

    def test_bogus_class_422(self):
        with self._fake():
            r = self.client.get("/api/ai/explain?section=riskcontrib"
                                "&asset_class=nope")
        self.assertEqual(r.status_code, 422)

    def test_regenerate_threads_class(self):
        # Whole-book baseline FIRST — the #338-deferred strengthening. The
        # write/read steps below share their filter params, so without this
        # baseline the test passes even if the class filter were dropped
        # from the scope key (everything would just share one key).
        with self._fake("wb"):
            self.client.get("/api/ai/explain?section=riskcontrib")
        with self._fake("first"):
            self.client.get("/api/ai/explain?section=riskcontrib&asset_class=equity_stock")
            self.client.get("/api/ai/explain?section=riskcontrib&asset_class=equity_stock")
        with self._fake("second"):
            rp = self.client.post("/api/ai/regenerate",
                                  json={"section": "riskcontrib",
                                        "asset_class": ["equity_stock"]})
            self.assertEqual(rp.json()["kind"], "generating")
            r = self.client.get("/api/ai/explain?section=riskcontrib&asset_class=equity_stock")
        self.assertEqual(r.json()["text"], "second")
        # The filtered regen must NOT have touched the whole-book entry.
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            wb = self.client.get("/api/ai/explain?section=riskcontrib")
        self.assertEqual(wb.json()["text"], "wb")


class TestScopeKeyFilter(unittest.TestCase):
    def test_whole_book_key_byte_identical(self):
        # cache-preservation guard: the pre-B2 dimless key is unchanged.
        self.assertEqual(ai.scope_key(["all"], "all", None),
                         ai.scope_key(["all"], "all", {}))

    def test_filter_dim_changes_key(self):
        base = ai.scope_key(["all"], "all", None)
        filt = ai.scope_key(["all"], "all", {"asset_class": ["equity_stock"]})
        self.assertNotEqual(base, filt)

    def test_account_only_differs_from_class_only(self):
        a = ai.scope_key(["all"], "all", {"account": ["test_a"]})
        c = ai.scope_key(["all"], "all", {"asset_class": ["equity_stock"]})
        self.assertNotEqual(a, c)

    def test_canon_filter_drops_all(self):
        self.assertEqual(ai._canon_filter(["all"]), [])
        self.assertEqual(ai._canon_filter(["equity_stock", "all"]), ["equity_stock"])
        self.assertEqual(ai._canon_filter("equity_stock"), ["equity_stock"])


PERF_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
               / "terminal_ai_performance_facts_golden.json")
PERF_FILTERED_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                        / "terminal_ai_performance_facts_filtered_golden.json")


class TestPerformanceFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai.build_facts("performance", self.frames,
                               history_start="all", broker=["all"])
        ai.scrub_gate(facts)
        golden = json.loads(PERF_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(facts, golden)

    def test_contributors_block(self):
        f = ai._facts_performance(self.frames, "all", None, None)
        c = f["contributors"]
        self.assertTrue(c["available"])
        self.assertIn("total-return", c["method"])
        for wlabel in ("60d", "ytd", "252d"):
            wblk = c["windows"][wlabel]
            self.assertLessEqual(len(wblk["top"]), 3)
            self.assertLessEqual(len(wblk["bottom"]), 3)
            for row in wblk["top"] + wblk["bottom"]:
                self.assertEqual(set(row), {"ticker", "weight_pct",
                                            "return_pct", "contrib_pp"})
            tops = [r["contrib_pp"] for r in wblk["top"]]
            self.assertEqual(tops, sorted(tops, reverse=True))
            top_set = {r["ticker"] for r in wblk["top"]}
            for r in wblk["bottom"]:
                self.assertNotIn(r["ticker"], top_set)
        ai.scrub_gate(f)

    def test_performance_is_dimless(self):
        self.assertNotIn("dims", ai.SECTIONS["performance"])

    def test_irr_whole_book_mirrors_tab(self):
        # Whole book: headline_raw carries the PORTFOLIO IRR; the fact must
        # equal that same source value (box==tab), rounded like _num does —
        # or be null exactly when the source is NaN (fixture-honest).
        from terminal import performance_service as ps
        facts = ai.build_facts("performance", self.frames)
        (pv, _bf, _cf, sel, aa, ca) = ps.twr_view_for(self.frames)
        hd = ps.headline_raw(pv, self.frames.irr_table, False)
        if hd["irr"] == hd["irr"]:   # finite source -> fact mirrors it
            self.assertEqual(facts["headline"]["irr_pct"],
                             round(hd["irr"] * 100.0, 2))
        else:                        # NaN source -> fact is null
            self.assertIsNone(facts["headline"]["irr_pct"])

    def test_filtered_facts_hide_irr_and_name_scope(self):
        # Fixture reality: the class filter collapses the synthesized series;
        # whatever availability results, IRR must be ABSENT-null (the tab
        # hides IRR under any Holdings filter) and the scope names the class.
        facts = ai.build_facts("performance", self.frames,
                               dims={"asset_class": ["equity_stock"]})
        ai.scrub_gate(facts)
        self.assertEqual(facts["scope"]["asset_class"], "Individual Stocks")
        if facts["available"]:
            self.assertIsNone(facts["headline"]["irr_pct"])
            self.assertTrue(facts["synthesized_twr"])

    def test_filtered_golden(self):
        facts = ai.build_facts("performance", self.frames,
                               dims={"asset_class": ["equity_stock"]})
        golden = json.loads(PERF_FILTERED_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(facts, golden)

    def test_whole_book_scope_has_no_filter_keys(self):
        facts = ai.build_facts("performance", self.frames)
        self.assertNotIn("account", facts["scope"])
        self.assertNotIn("asset_class", facts["scope"])

    def test_twr_view_for_refactor_is_pure(self):
        # The seam extraction must not change the tab payload at all — the
        # performance tab golden is the real gate; this is the fast local pin.
        from terminal import performance_service as ps
        view = ps.build_performance_view(self.frames)
        golden = json.loads(
            (Path(__file__).resolve().parent / "fixtures"
             / "terminal_performance_golden.json").read_text(encoding="utf-8"))
        self.assertEqual(view, golden)


INCOME_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                 / "terminal_ai_income_facts_golden.json")


class TestIncomeFacts(unittest.TestCase):
    ASOF = date(2026, 6, 28)   # pin the today-dependent surface — the
                               # test_terminal_income.ASOF convention

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai._facts_income(self.frames, "all", ["all"], None,
                                 asof=self.ASOF)
        ai.scrub_gate(facts)
        golden = json.loads(INCOME_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(facts, golden)

    def test_income_is_dimless(self):
        self.assertNotIn("dims", ai.SECTIONS["income"])

    def test_ignores_account_class_dims(self):
        # Whole-book by design: dims carrying account/class must not change
        # the facts (the server never merges them for income, but the
        # reducer itself must also be indifferent).
        a = ai._facts_income(self.frames, "all", ["all"], None,
                             asof=self.ASOF)
        b = ai._facts_income(self.frames, "all", ["all"],
                             {"account": ["test_a"]}, asof=self.ASOF)
        self.assertEqual(a, b)

    def test_no_dollar_keys_anywhere(self):
        # Belt-and-braces on the most dollar-dense tab: the deny regex
        # itself must find nothing (scrub_gate passing implies this, but
        # pin it against the real payload's key walk).
        facts = ai._facts_income(self.frames, "all", ["all"], None,
                                 asof=self.ASOF)
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    ks = str(k)
                    self.assertFalse(
                        ai._DENY_KEY_RE.search(ks)
                        and not ks.lower().endswith(ai._PCT_SUFFIXES), ks)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(facts)

    def test_forward_rollup_refactor_is_pure(self):
        from terminal import income_service as incs
        view = incs.build_income_view(self.frames, asof=self.ASOF)
        golden = json.loads(
            (Path(__file__).resolve().parent / "fixtures"
             / "terminal_income_golden.json").read_text(encoding="utf-8"))
        self.assertEqual(view, golden)

    def test_received_empty_branch_honest(self):
        # Constructed frames: zero income-type transactions -> received
        # unavailable, no mix/ratio keys fabricated.
        frames2 = dataclasses.replace(self.frames,
                                      transactions=self.frames.transactions.iloc[0:0])
        facts = ai._facts_income(frames2, "all", ["all"], None,
                                 asof=self.ASOF)
        ai.scrub_gate(facts)
        self.assertFalse(facts["received"]["available"])
        for k in ("dividends_share_pct", "interest_share_pct",
                  "projected_vs_ttm_ratio"):
            self.assertNotIn(k, facts["received"])

    def test_forward_unavailable_branch_honest(self):
        # Constructed frames: data dir with no dividend-history files ->
        # forward unavailable AND no ratio fabricated from a zero projection
        # (the final-review catch: the ratio must be gated on div_hist).
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            frames2 = dataclasses.replace(self.frames, data_dir=str(td))
            facts = ai._facts_income(frames2, "all", ["all"], None,
                                     asof=self.ASOF)
        ai.scrub_gate(facts)
        self.assertFalse(facts["forward"]["available"])
        self.assertNotIn("yield_on_covered_mv_pct", facts["forward"])
        self.assertNotIn("projected_vs_ttm_ratio", facts["received"])
        self.assertTrue(facts["received"]["available"])

    def test_latest_full_year_growth_present_when_two_full_years(self):
        # Constructed ledger: coverage from 2023-12 (on/before Jan 1 2024,
        # so 2024 AND 2025 are both FULL years under the TK-locked
        # definition), dividends 1000 in 2024 vs 1200 in 2025 -> +20.0%.
        tx = pd.DataFrame({
            "transaction_type": ["dividend"] * 3,
            "settlement_date": ["2023-12-15", "2024-06-15", "2025-06-15"],
            "amount": [50.0, 1000.0, 1200.0],
        })
        frames2 = dataclasses.replace(self.frames, transactions=tx)
        facts = ai._facts_income(frames2, "all", ["all"], None,
                                 asof=self.ASOF)
        ai.scrub_gate(facts)
        self.assertEqual(
            facts["received"]["latest_full_year_growth_pct"], 20.0)

    def test_latest_full_year_growth_omitted_on_partial_first_year(self):
        # Coverage starts mid-2024 -> 2024 is a PARTIAL year; the naive
        # 2025-vs-2024 ratio would overstate growth (the exact distortion
        # that got the fact descoped in B3), so it must be omitted.
        tx = pd.DataFrame({
            "transaction_type": ["dividend"] * 2,
            "settlement_date": ["2024-06-15", "2025-06-15"],
            "amount": [1000.0, 1200.0],
        })
        frames2 = dataclasses.replace(self.frames, transactions=tx)
        facts = ai._facts_income(frames2, "all", ["all"], None,
                                 asof=self.ASOF)
        self.assertNotIn("latest_full_year_growth_pct", facts["received"])


DIP_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_ai_dip_facts_golden.json")


class TestDipFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.apply_global_filters(
            hs.load_frames(str(FIXTURE)), ["all"], "all")

    def test_scrub_clean_and_golden(self):
        facts = ai.build_facts("dip", self.frames,
                               history_start="all", broker=["all"])
        ai.scrub_gate(facts)
        golden = json.loads(DIP_GOLDEN.read_text(encoding="utf-8"))
        # Bootstrap CI + GPD fit pass through optimizers — cross-BLAS ULP
        # drift is possible, so the dip golden uses _deep_close (same
        # doctrine as the dip TAB golden).
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_dip_is_dimless(self):
        self.assertNotIn("dims", ai.SECTIONS["dip"])

    def test_symbols_match_the_tab(self):
        # Box==tab: the facts narrate exactly the cards the tab shows.
        from terminal import dip_service as ds
        facts = ai.build_facts("dip", self.frames)
        view = ds.build_dip_view(self.frames)
        self.assertEqual([s["ticker"] for s in facts["symbols"]],
                         view["meta"]["symbols"])

    def test_bands_match_the_tab(self):
        from terminal import dip_service as ds
        facts = ai.build_facts("dip", self.frames)
        view = ds.build_dip_view(self.frames)
        tab_bands = {c["symbol"]: c["verdict"]["band"] for c in view["cards"]}
        for s in facts["symbols"]:
            self.assertEqual(s["band"], tab_bands[s["ticker"]])

    def test_ignores_account_class_dims(self):
        a = ai.build_facts("dip", self.frames)
        b = ai.build_facts("dip", self.frames,
                           dims={"asset_class": ["equity_stock"]})
        m = _deep_close(a, b)
        self.assertIsNone(m, m)


BRIEF_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                / "terminal_ai_brief_facts_golden.json")


class TestBriefFacts(unittest.TestCase):
    """S3: the executive brief — composition of the five existing
    reducers, whole-book, dimless, availability-honest per block."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.facts = ai._facts_brief(cls.frames, "all", None, None,
                                    asof=date(2026, 6, 28))

    def test_registered_dimless_with_question_and_schema(self):
        sec = ai.SECTIONS["brief"]
        self.assertNotIn("dims", sec)
        self.assertTrue(sec["question"].endswith("?"))
        props = sec["schema"]["properties"]
        self.assertEqual(props["watch"]["type"], "array")
        self.assertNotIn("caveat", props)
        # S1 lesson: maxItems ANYWHERE in an output schema 400s — walk the
        # whole schema serialization, not just top-level property dicts.
        self.assertNotIn('"maxItems"', json.dumps(sec["schema"]))
        self.assertIn("watch", sec["system"])

    def test_scrub_clean_and_golden(self):
        ai.scrub_gate(self.facts)
        golden = json.loads(BRIEF_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(self.facts, golden)
        self.assertIsNone(m, m)

    def test_blocks_present_and_composed_not_recomputed(self):
        f = self.facts
        self.assertTrue(f["available"])
        for k in ("performance", "risk", "vs_benchmark", "income", "tax"):
            self.assertIn(k, f)
        # composed = the sub-blocks are the sub-reducers' own values:
        perf = ai._facts_performance(self.frames, "all", None, None)
        self.assertEqual(f["performance"]["headline"], perf["headline"])
        risk = ai._facts_risk(self.frames, "all", None, None)
        self.assertEqual(f["risk"]["daily"], risk["daily"])

    def test_sub_unavailable_is_honest_and_brief_survives(self):
        with mock.patch.object(ai, "_facts_tax",
                               return_value={"section": "tax",
                                             "available": False,
                                             "reason": "no_lots"}):
            f = ai._facts_brief(self.frames, "all", None, None,
                                asof=date(2026, 6, 28))
        self.assertTrue(f["available"])
        self.assertEqual(f["tax"], {"available": False,
                                    "reason": "no_lots"})

    def test_unavailable_when_no_twr(self):
        with mock.patch.object(ai, "_facts_performance",
                               return_value={"available": False,
                                             "reason": "no_twr"}), \
             mock.patch.object(ai, "_facts_risk",
                               return_value={"available": False,
                                             "reason": "no_twr"}):
            f = ai._facts_brief(self.frames, "all", None, None,
                                asof=date(2026, 6, 28))
        self.assertFalse(f["available"])
        self.assertEqual(f["reason"], "no_data")

    def test_s4_blocks_ride_into_the_brief(self):
        f = ai._facts_brief(self.frames, "all", None, None,
                            asof=date(2026, 6, 28))
        self.assertIn("contributors", f["performance"])
        self.assertIn("stress", f["risk"])
        self.assertIn("consistency", f["vs_benchmark"])


class TestRoutesFilterThreadingB3(_RouteFilterHarness, unittest.TestCase):
    """B3: performance joins the B2 filter-threaded set; income/dip do NOT."""

    def test_filtered_performance_caches_separately(self):
        with self._fake("whole book"):
            self.client.get("/api/ai/explain?section=performance")
            r_wb = self.client.get("/api/ai/explain?section=performance")
        self.assertEqual(r_wb.json()["kind"], "ok")
        with self._fake("filtered"):
            r = self.client.get("/api/ai/explain?section=performance"
                                "&asset_class=equity_stock")
            self.assertEqual(r.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            r2 = self.client.get("/api/ai/explain?section=performance"
                                 "&asset_class=equity_stock")
        self.assertEqual(r2.json()["text"], "filtered")

    def test_bogus_account_422_performance(self):
        with self._fake():
            r = self.client.get("/api/ai/explain?section=performance"
                                "&account=nope")
        self.assertEqual(r.status_code, 422)

    def test_regenerate_threads_class_performance(self):
        # Whole-book baseline FIRST — the #338-deferred strengthening (the
        # write/read steps share filter params, so without the baseline the
        # test passes even with the predicate reverted).
        with self._fake("wb"):
            self.client.get("/api/ai/explain?section=performance")
        with self._fake("first"):
            self.client.get("/api/ai/explain?section=performance"
                            "&asset_class=equity_stock")
            self.client.get("/api/ai/explain?section=performance"
                            "&asset_class=equity_stock")
        with self._fake("second"):
            rp = self.client.post("/api/ai/regenerate",
                                  json={"section": "performance",
                                        "asset_class": ["equity_stock"]})
            self.assertEqual(rp.json()["kind"], "generating")
            r = self.client.get("/api/ai/explain?section=performance"
                                "&asset_class=equity_stock")
        self.assertEqual(r.json()["text"], "second")
        # The filtered regen must NOT have touched the whole-book entry.
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            wb = self.client.get("/api/ai/explain?section=performance")
        self.assertEqual(wb.json()["text"], "wb")

    def test_income_account_param_does_not_fork_the_cache(self):
        # income is whole-book: a (valid) account param must land on the SAME
        # scope key — the second call is a cache hit, not a new generation.
        with self._fake("income text"):
            r1 = self.client.get("/api/ai/explain?section=income"
                                 "&account=test_a")
            self.assertEqual(r1.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            r2 = self.client.get("/api/ai/explain?section=income")
        self.assertEqual(r2.json()["kind"], "ok")
        self.assertEqual(r2.json()["text"], "income text")

    def test_dip_class_param_does_not_fork_the_cache(self):
        with self._fake("dip text"):
            self.client.get("/api/ai/explain?section=dip"
                            "&asset_class=equity_stock")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            r2 = self.client.get("/api/ai/explain?section=dip")
        self.assertEqual(r2.json()["kind"], "ok")
        self.assertEqual(r2.json()["text"], "dip text")

    def test_explain_ok_all_three_sections(self):
        for section in ("performance", "income", "dip"):
            with self._fake(section + " text"):
                r1 = self.client.get("/api/ai/explain?section=" + section)
                self.assertEqual(r1.json()["kind"], "generating", section)
                r2 = self.client.get("/api/ai/explain?section=" + section)
            self.assertEqual(r2.json()["kind"], "ok", section)
            self.assertEqual(r2.json()["text"], section + " text")

    def test_regenerate_income_updates_cache(self):
        with self._fake("first"):
            self.client.get("/api/ai/explain?section=income")
            self.client.get("/api/ai/explain?section=income")
        with self._fake("second"):
            rp = self.client.post("/api/ai/regenerate",
                                  json={"section": "income"})
            self.assertEqual(rp.json()["kind"], "generating")
            r = self.client.get("/api/ai/explain?section=income")
        self.assertEqual(r.json()["text"], "second")

    def test_regenerate_dip_updates_cache(self):
        with self._fake("first"):
            self.client.get("/api/ai/explain?section=dip")
            self.client.get("/api/ai/explain?section=dip")
        with self._fake("second"):
            rp = self.client.post("/api/ai/regenerate",
                                  json={"section": "dip"})
            self.assertEqual(rp.json()["kind"], "generating")
            r = self.client.get("/api/ai/explain?section=dip")
        self.assertEqual(r.json()["text"], "second")


class TestExplainBenchmarkRoute(_RouteFilterHarness, unittest.TestCase):
    """Task 5: 'benchmark' joins the filter-threaded-section tuple in
    _ai_response (box==tab under account/class filters, same as
    performance/riskcontrib in B2/B3). These route tests lock the
    200/422 shape of the contract; the actual per-scope caching
    behavior is proven in the real-data closing smoke."""

    def test_explain_benchmark_ok_shape(self):
        with self._fake("bench text"):
            r = self.client.get("/api/ai/explain?section=benchmark&benchmark=spy")
        self.assertEqual(r.status_code, 200)           # enabled:false w/o key, or generating
        self.assertEqual(r.json()["section"], "benchmark")

    def test_explain_benchmark_rejects_foreign_dim(self):
        r = self.client.get("/api/ai/explain?section=benchmark&estimator=ewma")
        self.assertEqual(r.status_code, 422)           # benchmark takes no 'estimator'

    def test_explain_benchmark_accepts_account_filter(self):
        # a valid account id threads without 422 (value from the fixture's options)
        with self._fake():
            r = self.client.get("/api/ai/explain?section=benchmark&account=all")
        self.assertEqual(r.status_code, 200)

    def test_filtered_and_whole_book_cache_separately(self):
        # Review fix: account=all above is canon-stripped to [] and never
        # exercises the merge-into-dims branch. This uses a REAL non-"all"
        # class id (mirrors TestRoutesFilterThreading /
        # TestRoutesFilterThreadingB3's asset_class=equity_stock precedent)
        # to prove the filtered call actually lands on a DIFFERENT scope_key
        # than the whole-book call — if "benchmark" is ever dropped from the
        # server.py filter-threaded-section tuple, dims stays unmerged, both
        # calls share the whole-book scope_key, and the filtered GET below
        # would cache-hit "whole book" (kind "ok") instead of missing
        # ("generating"), failing this test.
        with self._fake("whole book"):
            self.client.get("/api/ai/explain?section=benchmark")
            r_wb = self.client.get("/api/ai/explain?section=benchmark")
        self.assertEqual(r_wb.json()["kind"], "ok")
        with self._fake("filtered"):
            r = self.client.get("/api/ai/explain?section=benchmark"
                                "&asset_class=equity_stock")
            self.assertEqual(r.json()["kind"], "generating")
        with mock.patch.object(ai, "resolve_client",
                               side_effect=AssertionError("should hit cache")):
            r2 = self.client.get("/api/ai/explain?section=benchmark"
                                 "&asset_class=equity_stock")
        self.assertEqual(r2.json()["text"], "filtered")


FRONTIER_FACTS_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                         / "terminal_ai_frontier_facts_golden.json")


class TestFrontierFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        seed = rss.build_risksim_view(cls.frames)
        opt = seed["optimizer"]
        cls.floors = {b["key"]: b["floor_default_pct"] for b in opt["buckets"]}
        cls.cap = opt["cap_default_pct"]
        out = rss.run_frontier(cls.frames, cap_pct=cls.cap, floors=cls.floors,
                               erp_pct=4.5)
        cls.sig = rss.frontier_sig(data_version=ai.data_version(FIXTURE),
                                   broker=["all"], history_start="all",
                                   account=["all"], asset_class=["all"],
                                   cap_pct=cls.cap, floors=cls.floors, caps={},
                                   erp_pct=4.5)
        rss.frontier_memo_put(cls.sig, out, ["all"], ["all"])

    def _facts(self):
        return ai.build_facts("frontier", self.frames, history_start="all",
                              broker=["all"], dims={"sig": self.sig})

    def test_registered_section(self):
        self.assertIn("frontier", ai.SECTIONS)
        self.assertEqual(ai.SECTIONS["frontier"]["dims"], ("sig",))

    def test_scrub_clean_and_golden(self):
        facts = self._facts()
        ai.scrub_gate(facts)                    # never raises on real output
        self.assertTrue(FRONTIER_FACTS_GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        golden = json.loads(FRONTIER_FACTS_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_available_and_dollar_free(self):
        facts = self._facts()
        self.assertTrue(facts["available"])
        self.assertEqual(facts["section"], "frontier")
        self.assertIn("at_your_vol", facts)
        blob = json.dumps(facts).lower()
        self.assertNotIn("$", blob)
        for banned in ("wealth", "nav", "_usd", "amount", "growth", "cash",
                       "gains", "losses"):
            self.assertNotIn(banned, blob)

    def test_frontier_fact_keys_allowed(self):
        # S1 catch-#1 discipline: every real frontier fact key must pass the
        # deny-regex (a future rename introducing a banned substring is caught).
        for k in ("rf_pct", "erp_pct", "beta_years", "assumed_beta_names",
                  "vol_pct", "exp_return_pct", "effective_n", "point_vol_pct",
                  "frontier_exp_return_pct", "gap_pp", "max_weight_pct",
                  "min_vol_pct", "min_vol_exp_return_pct", "max_return_pct",
                  "max_return_vol_pct", "n_points", "skipped_n"):
            ai.scrub_gate({k: 1.0})             # must not raise

    def test_cold_memo_unavailable_not_raise(self):
        facts = ai.build_facts("frontier", self.frames, history_start="all",
                               broker=["all"], dims={"sig": "deadbeefdeadbeef"})
        self.assertFalse(facts["available"])
        self.assertEqual(facts["reason"], "stale")


RISKSIM_FACTS_GOLDEN = (Path(__file__).resolve().parent / "fixtures"
                        / "terminal_ai_risksim_facts_golden.json")


class TestRisksimFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(str(FIXTURE))
        w = (rss._bundle_for(cls.frames, "all", "all")["weights"] * 100.0
             ).sort_values(ascending=False)
        new_pct = {str(t): float(v) for t, v in w.items()}
        new_pct[str(w.index[0])] -= 5.0
        new_pct[str(w.index[-1])] += 5.0
        facts = rss.run_simulation(cls.frames, new_pct)["ai_facts"]
        cls.sig = rss.simulate_sig(
            data_version=ai.data_version(str(FIXTURE)),
            broker=["all"], history_start="all", account=["all"],
            asset_class=["all"], weights=new_pct, candidates=[])
        rss.simulate_memo_put(cls.sig, facts, ["all"], ["all"])

    def _facts(self):
        return ai.build_facts("risksim", self.frames, history_start="all",
                              broker=["all"], dims={"sig": self.sig})

    def test_registered_section(self):
        self.assertIn("risksim", ai.SECTIONS)
        self.assertEqual(ai.SECTIONS["risksim"]["dims"], ("sig",))

    def test_scrub_clean_and_golden(self):
        facts = self._facts()
        ai.scrub_gate(facts)                    # keys must pass the dollar gate
        self.assertTrue(RISKSIM_FACTS_GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        golden = json.loads(RISKSIM_FACTS_GOLDEN.read_text(encoding="utf-8"))
        m = _deep_close(facts, golden)
        self.assertIsNone(m, m)

    def test_available_and_scope(self):
        facts = self._facts()
        self.assertTrue(facts["available"])
        self.assertEqual(facts["section"], "risksim")
        self.assertIn("scope", facts)

    def test_cold_memo_unavailable_not_raise(self):
        facts = ai.build_facts("risksim", self.frames, history_start="all",
                               broker=["all"], dims={"sig": "deadbeefdeadbeef"})
        self.assertFalse(facts["available"])
        self.assertEqual(facts["reason"], "stale")

    def test_candidate_facts_list_shape(self):
        # Coverage gap close: run_simulation WITH a candidate, memoized and
        # read back through the same memo -> _facts_risksim pass-through as
        # the pure-reweight tests above, must surface facts["candidates"] as
        # a non-empty [{ticker, mcr_pct, verdict}] list (offline sidecar
        # fixture; 'candidate' singular is gone post-migration).
        base = rss._bundle_for(self.frames, "all", "all")["weights"] * 100.0
        new_pct = {str(t): float(v) * 95.0 / 100.0 for t, v in base.items()}
        new_pct["NEWC"] = 5.0
        result = rss.run_simulation(
            self.frames, new_pct, candidates=[{"ticker": "NEWC", "proxy": ""}])
        self.assertIsNone(result["error"])
        sig = rss.simulate_sig(
            data_version=ai.data_version(str(FIXTURE)),
            broker=["all"], history_start="all", account=["all"],
            asset_class=["all"], weights=new_pct,
            candidates=[{"ticker": "NEWC", "proxy": ""}])
        rss.simulate_memo_put(sig, result["ai_facts"], ["all"], ["all"])
        facts = ai.build_facts("risksim", self.frames, history_start="all",
                               broker=["all"], dims={"sig": sig})
        self.assertTrue(facts["available"])
        self.assertNotIn("candidate", facts)
        cands = facts["candidates"]
        self.assertIsInstance(cands, list)
        self.assertTrue(cands, "expected a known MCR verdict for NEWC")
        for c in cands:
            self.assertEqual(set(c), {"ticker", "mcr_pct", "verdict"})
        self.assertEqual(cands[0]["ticker"], "NEWC")


if __name__ == "__main__":
    unittest.main()
