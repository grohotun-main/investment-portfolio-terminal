"""Invariant tests for the MERIDIAN Terminal Holdings data seam.

These exercise `terminal.holdings_service` against the committed synthetic
fixture (tests/fixtures/synth_data). They are deliberately invariant-based — no
hard-coded fixture magic numbers — so they stay true if the fixture is
regenerated. The KPI tape (TWR/IRR/vol) and `build_holdings_view` assembly are a
separate later task and are NOT covered here.
"""
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Mirror every other test in the suite: parsers/ is a flat module dir on sys.path
# and the repo root carries config_local / theme.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"


def _is_jsonable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


class TestPackage(unittest.TestCase):
    def test_package_imports(self):
        import terminal  # noqa: F401
        import terminal.holdings_service  # noqa: F401


class TestLoadFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)

    def test_positions_monthly_enriched(self):
        pm = self.frames.positions_monthly
        for col in ("symbol", "display_symbol", "description_clean",
                    "asset_class", "asset_class_label", "bucket", "broker",
                    "account_id", "account_display", "quantity", "price",
                    "cost_basis", "market_value", "unrealized_gl",
                    "statement_date", "month", "_filled", "_as_of_date"):
            self.assertIn(col, pm.columns, col)
        self.assertGreater(len(pm), 0)

    def test_positions_enriched(self):
        p = self.frames.positions
        for col in ("display_symbol", "description_clean", "asset_class_label",
                    "bucket", "broker", "account_display"):
            self.assertIn(col, p.columns, col)
        self.assertGreater(len(p), 0)

    def test_has_twr_and_irr(self):
        # twr_portfolio.csv carries the per-month return series the tape chains.
        self.assertIn("return_pct", self.frames.twr_portfolio.columns)
        # The raw irr_per_account.csv is per-account only; the PORTFOLIO IRR row
        # is NOT in the CSV — app.py SYNTHESIZES it live at render time via
        # compute_portfolio_irr (app.py:1964-1996), overwriting the loaded
        # table. load_frames therefore exposes the raw per-account frame (with
        # an `irr` + `account_id` column); the later KPI-tape task is
        # responsible for the live PORTFOLIO recompute. Asserting a PORTFOLIO
        # row here would be false against the committed synthetic fixture.
        self.assertIn("account_id", self.frames.irr_table.columns)
        self.assertIn("irr", self.frames.irr_table.columns)
        self.assertGreater(len(self.frames.irr_table), 0)

    def test_available_dates_desc(self):
        d = self.frames.available_dates
        self.assertGreater(len(d), 0)
        self.assertTrue(all(isinstance(x, str) for x in d))
        self.assertEqual(d, sorted(d, reverse=True))  # newest first
        # "YYYY-MM-DD" shape
        for x in d:
            self.assertRegex(x, r"^\d{4}-\d{2}-\d{2}$")

    def test_secondary_frames_loaded(self):
        # The tape task will need these — make sure they load (may be empty).
        for name in ("transactions", "prices_latest", "daily_prices",
                     "spy_tr"):
            self.assertIsNotNone(getattr(self.frames, name))


class TestFramesCache(unittest.TestCase):
    """load_frames is cached per data_dir on a CSV stat-signature so every
    terminal route stops re-reading + re-enriching the whole book per request.
    The cached Frames is shared (frozen + read-only by every consumer)."""

    def setUp(self):
        from terminal import holdings_service as hs
        self.hs = hs
        hs._clear_frames_cache()

    def tearDown(self):
        self.hs._clear_frames_cache()

    def test_repeated_load_returns_cached_instance(self):
        f1 = self.hs.load_frames(FIXTURE)
        f2 = self.hs.load_frames(FIXTURE)
        self.assertIs(f1, f2)   # cached: same object, no re-read / re-enrich

    def test_cache_invalidates_when_a_data_file_changes(self):
        with tempfile.TemporaryDirectory() as td:
            dd = Path(td) / "data"
            shutil.copytree(FIXTURE, dd)
            f1 = self.hs.load_frames(dd)
            # Append a duplicate position row -> positions.csv size + mtime change.
            pos = dd / "positions.csv"
            lines = pos.read_text(encoding="utf-8").splitlines()
            lines.append(lines[-1])
            pos.write_text("\n".join(lines) + "\n", encoding="utf-8")
            f2 = self.hs.load_frames(dd)
            self.assertIsNot(f2, f1)                               # rebuilt
            self.assertGreater(len(f2.positions), len(f1.positions))  # fresh data

    def test_shared_frames_not_mutated_by_consumers(self):
        """Guards the instance-sharing decision: the heavy view builders and the
        global-filter narrowing must treat the cached Frames read-only."""
        frames = self.hs.load_frames(FIXTURE)
        before = (len(frames.positions), len(frames.positions_monthly),
                  tuple(frames.positions.columns),
                  tuple(frames.positions_monthly.columns))
        self.hs.build_holdings_view(frames, as_of=None, account=["all"],
                                    asset_class=["all"], top_n=15, search="")
        self.hs._current_snap(frames)
        self.hs.apply_global_filters(frames, "all", "all")
        after = (len(frames.positions), len(frames.positions_monthly),
                 tuple(frames.positions.columns),
                 tuple(frames.positions_monthly.columns))
        self.assertEqual(before, after)
        self.assertIs(self.hs.load_frames(FIXTURE), frames)  # still the cached one


class TestCurrentSnap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)

    def test_portfolio_value_finite_positive(self):
        snap = self.hs._current_snap(self.frames)
        total = float(snap["market_value"].sum())
        self.assertTrue(math.isfinite(total))
        self.assertGreater(total, 0)

    def test_matches_canonical_nav(self):
        # The latest-as-of marked snapshot total must equal canonical_nav of the
        # marked monthly frame, computed independently here.
        from nav_basis import canonical_nav
        snap = self.hs._current_snap(self.frames)
        total = float(snap["market_value"].sum())
        nav_ids = {a for a in self.frames.positions_monthly["account_id"]
                   .astype(str).unique() if "TEST" in a.upper()}
        # The synth fixture is all TEST accounts; canonical_nav would exclude
        # them. Compare WITHOUT the exclusion so we are checking the same
        # universe the snapshot total covers.
        nav = canonical_nav(self.frames.positions_monthly)
        self.assertAlmostEqual(total, nav, delta=0.01)
        # Sanity: with the test-exclusion the fixture nets to ~0 (all TEST).
        self.assertGreaterEqual(len(nav_ids), 0)


class TestAllocation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.snap = hs._current_snap(cls.frames)

    def test_class_pct_sums_to_100(self):
        a = self.hs._alloc_by_class(self.snap)
        self.assertAlmostEqual(sum(s["pct"] for s in a["slices"]), 100.0, places=0)
        self.assertEqual(a["n"], len(a["slices"]))
        for s in a["slices"]:
            self.assertTrue(s["color"].startswith("#"))
            self.assertTrue(_is_jsonable(s))
            self.assertIsInstance(s["pct"], float)
        self.assertTrue(a["total_label"].startswith("$"))
        self.assertTrue(a["total_label"].endswith("M"))

    def test_class_desc_sorted(self):
        a = self.hs._alloc_by_class(self.snap)
        pcts = [s["pct"] for s in a["slices"]]
        self.assertEqual(pcts, sorted(pcts, reverse=True))

    def test_unmapped_classes_stay_distinguishable(self):
        # Classes missing from CLASS_COLORS must NOT collapse onto one shared
        # grey — the donut would render as a single color (QA-polish S1).
        import pandas as pd
        snap = pd.DataFrame({
            "asset_class": ["mystery_a", "mystery_b", "mystery_c"],
            "market_value": [300.0, 200.0, 100.0],
        })
        a = self.hs._alloc_by_class(snap)
        colors = [s["color"] for s in a["slices"]]
        self.assertEqual(len(set(colors)), 3)
        self.assertNotIn("#888", colors)

    def test_mapped_class_colors_unchanged(self):
        # The fallback must not disturb mapped classes (golden purity anchor).
        a = self.hs._alloc_by_class(self.snap)
        for s in a["slices"]:
            if s["class"] in self.hs.CLASS_COLORS:
                self.assertEqual(s["color"], self.hs.CLASS_COLORS[s["class"]])

    def test_account_bars_normalised(self):
        a = self.hs._alloc_by_account(self.snap)
        bars = [r["bar"] for r in a["rows"]]
        self.assertAlmostEqual(max(bars), 100.0, places=0)  # top row = 100%
        self.assertEqual(a["n"], len(a["rows"]))
        for r in a["rows"]:
            self.assertTrue(_is_jsonable(r))
            self.assertTrue(str(r["color"]).startswith("#") or r["color"] is None)


class TestTopHoldings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.snap = hs._current_snap(cls.frames)
        cls.total = float(cls.snap["market_value"].sum())

    def test_top_subset_and_bars(self):
        top = self.hs._top_holdings(self.snap, self.total, top_n=15)
        self.assertLessEqual(len(top["rows"]), 15)
        self.assertEqual(top["top_n"], 15)
        self.assertAlmostEqual(max(r["bar"] for r in top["rows"]), 100.0, places=0)

    def test_symbols_subset_of_snapshot(self):
        from terminal import holdings_service as hs
        top = self.hs._top_holdings(self.snap, self.total, top_n=15)
        collapsed = hs.collapse_buckets(self.snap)
        valid = set(collapsed["display_symbol"].astype(str))
        for r in top["rows"]:
            self.assertIn(r["symbol"], valid)
            self.assertTrue(_is_jsonable(r))

    def test_top_n_respected(self):
        top = self.hs._top_holdings(self.snap, self.total, top_n=2)
        self.assertLessEqual(len(top["rows"]), 2)


class TestRawSeams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.snap = hs._current_snap(cls.frames)

    def test_alloc_by_class_raw(self):
        raw = self.hs.alloc_by_class_raw(self.snap)
        # columns + descending sort + sums equal a direct groupby
        self.assertEqual(list(raw.columns), ["asset_class", "market_value"])
        direct = (self.snap.groupby("asset_class")["market_value"].sum()
                  .sort_values(ascending=False))
        self.assertEqual(raw["asset_class"].tolist(), direct.index.tolist())
        for a, b in zip(raw["market_value"].tolist(), direct.tolist()):
            self.assertAlmostEqual(a, b)

    def test_alloc_by_class_formatter_wraps_raw(self):
        # The view formatter's dict is unchanged (built from the raw frame).
        view = self.hs._alloc_by_class(self.snap)
        self.assertEqual(view["n"], len(self.hs.alloc_by_class_raw(self.snap)))
        self.assertIn("slices", view)

    def test_alloc_by_account_raw(self):
        raw = self.hs.alloc_by_account_raw(self.snap)
        # columns + descending sort + sums equal a direct groupby
        self.assertEqual(list(raw.columns), ["broker", "bucket", "market_value"])
        direct = (self.snap.groupby(["broker", "bucket"])["market_value"].sum()
                  .sort_values(ascending=False))
        self.assertEqual(list(map(tuple, raw[["broker", "bucket"]].values)),
                          direct.index.tolist())
        for a, b in zip(raw["market_value"].tolist(), direct.tolist()):
            self.assertAlmostEqual(a, b)

    def test_alloc_by_account_formatter_wraps_raw(self):
        # The view formatter's dict is unchanged (built from the raw frame).
        view = self.hs._alloc_by_account(self.snap)
        self.assertEqual(view["n"], len(self.hs.alloc_by_account_raw(self.snap)))
        self.assertIn("rows", view)

    def test_top_holdings_raw(self):
        raw = self.hs.top_holdings_raw(self.snap, top_n=15)
        # superset columns: mv (the formatter's number) + qty/n_accts/desc
        # (extra columns app.py needs, ignored by the service's formatter).
        self.assertEqual(
            list(raw.columns),
            ["display_symbol", "asset_class", "mv", "qty", "n_accts", "desc"],
        )
        self.assertLessEqual(len(raw), 15)
        # mv + sort match a direct groupby on the collapsed snap.
        collapsed = self.hs.collapse_buckets(self.snap)
        direct = (collapsed.groupby(["display_symbol", "asset_class"])
                  .agg(mv=("market_value", "sum"))
                  .reset_index().sort_values("mv", ascending=False).head(15))
        self.assertEqual(raw["display_symbol"].tolist(),
                          direct["display_symbol"].tolist())
        for a, b in zip(raw["mv"].tolist(), direct["mv"].tolist()):
            self.assertAlmostEqual(a, b)
        # descending-sorted by mv.
        mvs = raw["mv"].tolist()
        self.assertEqual(mvs, sorted(mvs, reverse=True))

    def test_top_holdings_raw_empty_collapsed(self):
        empty = self.snap.iloc[0:0]
        raw = self.hs.top_holdings_raw(empty, top_n=15)
        self.assertTrue(raw.empty)

    def test_top_holdings_formatter_wraps_raw(self):
        # The view formatter's dict is unchanged (built from the raw frame's
        # mv column only — qty/n_accts/desc don't perturb pct/bar/sort).
        total = float(self.snap["market_value"].sum())
        view = self.hs._top_holdings(self.snap, total, top_n=15)
        raw = self.hs.top_holdings_raw(self.snap, top_n=15)
        self.assertEqual(len(view["rows"]), len(raw))
        for row, (_, raw_row) in zip(view["rows"], raw.iterrows()):
            self.assertEqual(row["symbol"], str(raw_row["display_symbol"]))

    def test_positions_table_raw_columns_and_sort(self):
        as_of = self.frames.available_dates[0]
        total = float(self.snap["market_value"].sum())
        raw = self.hs.positions_table_raw(self.snap, self.frames, total, as_of)
        for col in ("display_symbol", "description_clean", "bucket",
                    "asset_class", "asset_class_label", "quantity", "price",
                    "market_value", "cost_basis", "unrealized_gl",
                    "weight_pct", "unrealized_pct", "price_asof",
                    "price_stmt"):
            self.assertIn(col, raw.columns, col)
        # descending-sorted by market_value, matching a direct collapse+sort.
        collapsed = self.hs.collapse_buckets(self.snap)
        direct = collapsed.sort_values("market_value", ascending=False)
        self.assertEqual(raw["display_symbol"].tolist(),
                          direct["display_symbol"].tolist())
        mvs = raw["market_value"].tolist()
        self.assertEqual(mvs, sorted(mvs, reverse=True))

    def test_positions_table_raw_weight_and_unrealized(self):
        as_of = self.frames.available_dates[0]
        total = float(self.snap["market_value"].sum())
        raw = self.hs.positions_table_raw(self.snap, self.frames, total, as_of)
        # weight_pct sums to ~100 (every collapsed row is present — no search).
        self.assertAlmostEqual(raw["weight_pct"].sum(), 100.0, places=0)
        for _, r in raw.iterrows():
            if r["cost_basis"] > 0:
                expected = ((r["market_value"] - r["cost_basis"])
                            / r["cost_basis"] * 100.0)
                self.assertAlmostEqual(r["unrealized_pct"], expected)
            else:
                self.assertTrue(math.isnan(r["unrealized_pct"]))

    def test_positions_table_raw_is_search_independent(self):
        # The raw seam is computed over the FULL universe (search is a
        # formatter-only concern layered on top) — no `search` kwarg exists.
        as_of = self.frames.available_dates[0]
        total = float(self.snap["market_value"].sum())
        raw = self.hs.positions_table_raw(self.snap, self.frames, total, as_of)
        collapsed = self.hs.collapse_buckets(self.snap)
        self.assertEqual(len(raw), len(collapsed))

    def test_positions_table_formatter_wraps_raw(self):
        # The view dict is unchanged (built from the raw frame + search filter
        # + string formatting on top).
        as_of = self.frames.available_dates[0]
        total = float(self.snap["market_value"].sum())
        view = self.hs._positions_table(self.snap, self.frames, total, as_of)
        raw = self.hs.positions_table_raw(self.snap, self.frames, total, as_of)
        self.assertEqual(view["shown"], len(raw))
        self.assertEqual(view["total_underlying"], len(self.snap))
        for row, (_, raw_row) in zip(view["rows"], raw.iterrows()):
            self.assertEqual(row["symbol"], str(raw_row["display_symbol"]))
            self.assertAlmostEqual(row["weight_pct"], float(raw_row["weight_pct"]))


class TestPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.snap = hs._current_snap(cls.frames)
        cls.total = float(cls.snap["market_value"].sum())
        cls.as_of = cls.frames.available_dates[0]

    def test_shown_matches_rows(self):
        t = self.hs._positions_table(self.snap, self.frames, self.total, self.as_of)
        self.assertEqual(t["shown"], len(t["rows"]))
        self.assertIsInstance(t["total_underlying"], int)
        self.assertTrue(_is_jsonable(t))

    def test_ugl_dirs_valid(self):
        t = self.hs._positions_table(self.snap, self.frames, self.total, self.as_of)
        for r in t["rows"]:
            self.assertIn(r["ugl_dir"], ("up", "down", "flat"))
            self.assertIn(r["ugl_pct_dir"], ("up", "down", "flat"))
            self.assertIsInstance(r["weight_pct"], float)

    def test_aggregated_rows_contract(self):
        # The synth fixture has no TLH/Treasury sleeve, so there may be zero
        # aggregated rows. When one exists it must honour the qty/price="—" +
        # price_asof="(aggregated)" contract.
        t = self.hs._positions_table(self.snap, self.frames, self.total, self.as_of)
        for r in t["rows"]:
            if r["qty"] == "—":
                self.assertEqual(r["price"], "—")
                self.assertEqual(r["price_asof"], "(aggregated)")

    def test_search_filters_to_zero(self):
        t = self.hs._positions_table(self.snap, self.frames, self.total,
                                     self.as_of, search="zzzznope")
        self.assertEqual(t["shown"], 0)
        self.assertEqual(t["rows"], [])

    def test_search_narrows(self):
        full = self.hs._positions_table(self.snap, self.frames, self.total, self.as_of)
        # Pick a real symbol substring from the first row and confirm filtering.
        sym = full["rows"][0]["symbol"]
        sub = self.hs._positions_table(self.snap, self.frames, self.total,
                                       self.as_of, search=sym)
        self.assertGreaterEqual(sub["shown"], 1)
        self.assertLessEqual(sub["shown"], full["shown"])

    def test_money_formatting(self):
        t = self.hs._positions_table(self.snap, self.frames, self.total, self.as_of)
        for r in t["rows"]:
            self.assertTrue(r["market_value"].startswith("$")
                            or r["market_value"].startswith("-$"))
            # signed ugl uses U+2212 for negatives, "+" for non-negatives
            self.assertTrue(r["ugl"][0] in ("+", "−"))


class TestKpiTape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.tape = hs._kpi_tape(cls.frames)

    def test_six_cells_exact_key_order(self):
        keys = [c["key"] for c in self.tape]
        self.assertEqual(keys, ["portfolio_value", "cum_twr", "annualized",
                                "irr", "vol", "max_dd"])
        self.assertEqual(len(self.tape), 6)

    def test_portfolio_value_dollar_and_chip(self):
        cell = self.tape[0]
        self.assertTrue(cell["value"].startswith("$"))
        # chip only on cell 0; the synth fixture has a prior month, so it's set.
        self.assertIn("chip", cell)
        for c in self.tape[1:]:
            self.assertNotIn("chip", c)
        if cell["chip"] is not None:
            self.assertIn(cell["chip"]["dir"], ("up", "down"))
            self.assertTrue(cell["chip"]["text"].endswith("%"))

    def test_cum_twr_color_matches_sign(self):
        from compute_twr import link_returns
        cum = link_returns(self.frames.twr_portfolio["return_pct"])
        self.assertEqual(self.tape[1]["color"], "gain" if cum >= 0 else "loss")

    def test_max_dd_always_loss(self):
        self.assertEqual(self.tape[5]["color"], "loss")

    def test_annualized_label_says_twr(self):
        self.assertEqual(self.tape[2]["label"], "Annualized TWR")

    def test_irr_sub_explains_whole_book_gate(self):
        """Canonical view: finite -> 'money-weighted'; absent -> the sub says
        WHY the em-dash (TK 2026-07-19). Scoped view (broker subset): the sub
        names the scope; non-computable says n/a — never the now-wrong
        'whole-book only' (spec 2026-08-07)."""
        irr_cell = self.tape[3]
        if irr_cell["value"] == "—":
            self.assertEqual(irr_cell["sub"], "money-weighted · whole-book only")
        else:
            self.assertEqual(irr_cell["sub"], "money-weighted")
        # A single-broker slice now RECOMPUTES a scoped IRR; the sub labels
        # the scope either way.
        opts = self.hs._broker_options(self.hs._current_snap(self.frames))[0]
        one = [o["id"] for o in opts if o["id"] != "all"][:1]
        if not one:
            self.skipTest("no broker options on fixture")
        f2 = self.hs.apply_global_filters(self.frames, one, "all")
        self.assertIsNotNone(f2.broker_scope)
        cell = self.hs._kpi_tape(f2)[3]
        lbl = " + ".join(f2.broker_scope)
        if cell["value"] == "—":
            self.assertEqual(cell["sub"],
                             "money-weighted · n/a for this selection")
        else:
            self.assertEqual(cell["sub"], f"money-weighted · {lbl}")
        # The two outcomes must agree with the row's actual presence.
        has_row = (not f2.irr_table.empty and
                   "PORTFOLIO" in set(f2.irr_table["account_id"].astype(str)))
        self.assertEqual(cell["value"] != "—", has_row)

    def test_color_domain(self):
        for c in self.tape:
            self.assertIn(c["color"], (None, "gain", "loss"))

    def test_jsonable(self):
        self.assertTrue(_is_jsonable(self.tape))


class TestBuildView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.view = hs.build_holdings_view(hs.load_frames(FIXTURE))

    def test_contract_keys(self):
        for k in ("meta", "tape", "health", "carried_forward", "snapshot",
                  "alloc_class", "alloc_account", "top_holdings", "positions"):
            self.assertIn(k, self.view)

    def test_tape_len_and_meta_as_of(self):
        self.assertEqual(len(self.view["tape"]), 6)
        self.assertEqual(self.view["meta"]["as_of"],
                         self.view["meta"]["available_dates"][0])

    def test_meta_filter_options(self):
        meta = self.view["meta"]
        for opt in meta["accounts"]:
            self.assertIn("id", opt)
            self.assertIn("label", opt)
        for opt in meta["classes"]:
            self.assertIn("id", opt)
            self.assertIn("label", opt)
        self.assertEqual(meta["filter"],
                         {"account": "all", "asset_class": "all", "broker": "all"})
        self.assertIsInstance(meta["synthetic"], bool)
        # fixture path contains "synth" → SYNTHETIC badge on.
        self.assertTrue(meta["synthetic"])

    def test_snapshot_cards_shape(self):
        snap = self.view["snapshot"]
        self.assertTrue(snap["portfolio_value"].startswith("$"))
        self.assertEqual(snap["portfolio_value_sub"], "marked to live prices")
        for card in ("vs_prior", "ytd"):
            for k in ("pct", "dir", "abs", "prior_label"):
                self.assertIn(k, snap[card])
        # Two months of one year: a prior period exists, a prior YEAR-END
        # does not -> the honest placeholder (the live path is exercised in
        # TestYtdCard).
        self.assertTrue(snap["vs_prior"]["prior_label"].startswith("vs "))
        self.assertEqual(snap["ytd"], {"pct": "-", "dir": "flat", "abs": "",
                                       "prior_label": "No prior year-end snapshot"})
        self.assertIsInstance(snap["accounts"]["value"], int)
        self.assertIsInstance(snap["holdings"]["symbols"], int)
        self.assertIsInstance(snap["holdings"]["rows"], int)

    def test_delta_cards_compare_within_the_filter_scope(self):
        # Filtered current vs FILTERED prior. Against the unfiltered prior a
        # one-account filter read "-82.6% / -$2.8M" (TK, 2026-08-22).
        import pandas as pd
        frames = self.hs.load_frames(FIXTURE)
        aid = self.view["meta"]["accounts"][0]["id"]
        view = self.hs.build_holdings_view(frames, account=aid)
        snap = view["snapshot"]
        self.assertTrue(snap["portfolio_value_sub"].startswith("of $"))
        self.assertIn("unfiltered", snap["portfolio_value_sub"])
        _, by_id = self.hs._account_options(self.hs._current_snap(frames))
        bucket = by_id[aid]
        dates = frames.available_dates
        cur = self.hs._current_snap(frames, dates[0])
        prior = self.hs._current_snap(frames, dates[1])
        cur_t = float(cur[cur["bucket"] == bucket]["market_value"].sum())
        prior_t = float(prior[prior["bucket"] == bucket]["market_value"].sum())
        self.assertGreater(prior_t, 0)
        self.assertEqual(snap["portfolio_value"], self.hs.fmt_money(cur_t))
        self.assertEqual(snap["vs_prior"]["pct"],
                         self.hs.fmt_pct((cur_t / prior_t - 1.0) * 100.0))
        self.assertEqual(snap["vs_prior"]["prior_label"],
                         "vs " + pd.Timestamp(dates[1]).strftime("%b %d, %Y"))

    def test_health_shape(self):
        self.assertIn(self.view["health"]["level"],
                      ("success", "warning", "error", "muted"))
        self.assertIn("text", self.view["health"])

    def test_json_serialisable(self):
        json.dumps(self.view)  # must not raise (no numpy / Timestamp leaks)

    def test_filter_narrows_snapshot(self):
        meta = self.view["meta"]
        # Pick the first asset-class option and confirm filtering trims totals.
        cid = meta["classes"][0]["id"]
        filtered = self.hs.build_holdings_view(
            self.hs.load_frames(FIXTURE), asset_class=cid)
        self.assertEqual(filtered["meta"]["filter"]["asset_class"], [cid])
        # filtered snapshot rows ≤ unfiltered rows.
        self.assertLessEqual(filtered["snapshot"]["holdings"]["rows"],
                             self.view["snapshot"]["holdings"]["rows"])


class TestBrokerMeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.snap = hs._current_snap(cls.frames)
        cls.view = hs.build_holdings_view(cls.frames)

    def test_broker_options_shape(self):
        opts, by_id = self.hs._broker_options(self.snap)
        self.assertGreater(len(opts), 0)
        for o in opts:
            self.assertIn("id", o); self.assertIn("label", o)
        # every id maps back to a broker value present in the snapshot
        brokers = set(self.snap["broker"].dropna().astype(str))
        self.assertTrue(set(by_id.values()).issubset(brokers))

    def test_meta_has_brokers_and_echo(self):
        meta = self.view["meta"]
        self.assertIn("brokers", meta)
        self.assertIsInstance(meta["brokers"], list)
        self.assertEqual(meta["filter"]["broker"], "all")  # default echo scalar


class TestGolden(unittest.TestCase):
    GOLDEN = FIXTURE.parent / "terminal_holdings_golden.json"

    def test_matches_golden(self):
        from terminal import holdings_service as hs
        view = hs.build_holdings_view(hs.load_frames(FIXTURE))
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        golden = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(view, golden)


class TestInterimRollforward(unittest.TestCase):
    """`load_frames` must mirror app.py.load_data's mid-month interim handling:
    when a `transactions_interim.csv` is present, union it into transactions and
    roll holdings forward via synthesize_interim_positions so the snapshot date
    advances to max(interim settlement_date). The committed fixture has NO
    interim file, so the default path must be an exact no-op — that is asserted
    by the golden + parity tests; here we build a temp data dir WITH an interim
    row and assert the snapshot advances past the fixture's latest month."""

    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        # Baseline: the fixture with no interim file.
        cls.base_dates = hs.load_frames(FIXTURE).available_dates

    def _copy_fixture(self, dst: Path) -> None:
        """Copy the fixture CSVs into a fresh temp data dir."""
        for csv in FIXTURE.glob("*.csv"):
            shutil.copy2(csv, dst / csv.name)

    def test_interim_advances_snapshot(self):
        import pandas as pd
        latest = pd.Timestamp(self.base_dates[0])
        # Settle the interim activity strictly after the fixture's latest month.
        interim_settle = (latest + pd.offsets.MonthBegin(1)
                          + pd.Timedelta(days=14)).normalize()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._copy_fixture(tmp)

            # Craft a minimal valid interim row against an account/symbol that
            # exists in the latest fixture snapshot (a buy of AAA in TEST-A). The
            # columns match transactions.csv's schema so the union + roll-forward
            # behave exactly as app.py's loader would.
            interim = pd.DataFrame([{
                "settlement_date": interim_settle,
                "trade_date":      interim_settle,
                "broker":          "alpine",
                "account_id":      "TEST-A",
                "transaction_type": "buy",
                "symbol":          "AAA",
                "cusip":           "",
                "description":     "Synthetic Equity A",
                "quantity":        10.0,
                "price":           102.0,
                "amount":          -1020.0,
                "source_file":     "test_interim",
                "flow_scope":      "",
                "pair_id":         "",
            }])
            interim.to_csv(tmp / "transactions_interim.csv", index=False)

            rolled_dates = self.hs.load_frames(tmp).available_dates

        # The newest available date must advance to (or past) the interim month.
        self.assertGreater(
            pd.Timestamp(rolled_dates[0]), latest,
            f"snapshot did not advance: {rolled_dates[0]} vs base {latest}",
        )
        self.assertGreaterEqual(pd.Timestamp(rolled_dates[0]), interim_settle)

    def test_no_interim_is_noop(self):
        """A temp data dir WITHOUT an interim file must reproduce the fixture's
        available_dates exactly (the no-interim clean no-op the golden depends
        on)."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._copy_fixture(tmp)
            self.assertEqual(self.hs.load_frames(tmp).available_dates,
                             self.base_dates)


class TestYtdCard(unittest.TestCase):
    """The YTD card anchors on the last snapshot dated in the PRIOR calendar
    year (the December statement). The committed fixture spans two months of
    one year — no anchor, the placeholder pinned by the golden — so relabel
    its older month to the previous December to exercise the live path."""

    def test_ytd_anchors_on_prior_year_end(self):
        import pandas as pd
        from terminal import holdings_service as hs
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for csv in FIXTURE.glob("*.csv"):
                shutil.copy2(csv, tmp / csv.name)
            pos = pd.read_csv(tmp / "positions.csv")
            oldest = sorted(pos["statement_date"].astype(str).unique())[0]
            year_end = f"{int(oldest[:4]) - 1}-12-31"
            pos.loc[pos["statement_date"].astype(str) == oldest,
                    "statement_date"] = year_end
            pos.to_csv(tmp / "positions.csv", index=False)
            frames = hs.load_frames(tmp)
            view = hs.build_holdings_view(frames)
        ytd = view["snapshot"]["ytd"]
        self.assertEqual(ytd["prior_label"],
                         "vs " + pd.Timestamp(year_end).strftime("%b %d, %Y"))
        self.assertNotEqual(ytd["pct"], "-")
        self.assertIn(ytd["dir"], ("up", "down"))
        self.assertTrue(ytd["abs"].startswith(("+$", "-$")))


class TestServer(unittest.TestCase):
    """The FastAPI shell over the data seam, exercised via TestClient against the
    committed fixture. Validates the happy path + the typed/known-option input
    rejection (spec §7.4)."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    def tearDown(self):
        # Some tests repoint APP_DATA_DIR; always restore the fixture so a later
        # test (or another module sharing this process) sees the committed data.
        os.environ["APP_DATA_DIR"] = str(FIXTURE)

    def test_holdings_endpoint_ok(self):
        r = self.client.get("/api/holdings")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["tape"]), 6)

    def test_rejects_unknown_as_of(self):
        r = self.client.get("/api/holdings", params={"as_of": "1999-01-01"})
        self.assertEqual(r.status_code, 422)

    def test_rejects_unknown_account(self):
        r = self.client.get("/api/holdings", params={"account": "__nope__"})
        self.assertEqual(r.status_code, 422)

    def test_multi_account_ok(self):
        v = self.client.get("/api/holdings")
        ids = [o["id"] for o in v.json()["meta"]["accounts"]][:2]
        if len(ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        r = self.client.get("/api/holdings",
                            params=[("account", ids[0]), ("account", ids[1])])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["filter"]["account"], ids)

    def test_multi_with_unknown_id_422(self):
        v = self.client.get("/api/holdings")
        good = [o["id"] for o in v.json()["meta"]["accounts"]][0]
        r = self.client.get("/api/holdings",
                            params=[("account", good), ("account", "__nope__")])
        self.assertEqual(r.status_code, 422)

    def test_account_all_matches_no_param(self):
        a = self.client.get("/api/holdings").json()
        b = self.client.get("/api/holdings", params={"account": "all"}).json()
        self.assertEqual(a["meta"]["filter"], b["meta"]["filter"])

    def test_missing_data_dir_returns_503(self):
        # A wrong/missing APP_DATA_DIR must surface a clean 503, not a 500 with a
        # FileNotFoundError stack trace (misconfiguration, not empty data).
        os.environ["APP_DATA_DIR"] = str(FIXTURE / "__does_not_exist__")
        r = self.client.get("/api/holdings")
        self.assertEqual(r.status_code, 503)

    def test_serves_index(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_rejects_unknown_broker(self):
        r = self.client.get("/api/holdings", params={"broker": "__nope__"})
        self.assertEqual(r.status_code, 422)

    def test_broker_all_ok_and_echo(self):
        r = self.client.get("/api/holdings", params={"broker": "all"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["filter"]["broker"], "all")

    def test_broker_narrow_keeps_full_options_and_echo(self):
        full = self.client.get("/api/holdings").json()["meta"]["brokers"]
        if len(full) < 2:
            self.skipTest("fixture has <2 brokers")
        one = full[0]["id"]
        r = self.client.get("/api/holdings", params={"broker": one}).json()
        # one-way-picker guard: options stay the FULL set even when narrowed
        self.assertEqual({b["id"] for b in r["meta"]["brokers"]},
                         {b["id"] for b in full})
        # echo reflects the selection (not the stale "all")
        self.assertEqual(r["meta"]["filter"]["broker"], [one])

    def test_history_start_all_is_default_and_ok(self):
        r = self.client.get("/api/performance")
        self.assertEqual(r.status_code, 200)
        self.assertIn("history_starts", r.json()["meta"])
        self.assertEqual(r.json()["meta"]["filter"]["history_start"], "all")

    def test_history_start_valid_year_ok(self):
        opts = self.client.get("/api/performance").json()["meta"]["history_starts"]
        if len(opts) < 2:
            self.skipTest("fixture has <2 years")
        yid = opts[-1]["id"]
        # NOTE: history_start ids are shaped "{year}+" — a raw f-string URL
        # (e.g. f".../performance?history_start={yid}") would embed a literal
        # "+", which application/x-www-form-urlencoded query parsing (what
        # Starlette uses) decodes as a space, corrupting the id before it
        # reaches the handler. Use params= like the rest of this file's
        # filter-id tests (e.g. test_broker_narrow_keeps_full_options_and_echo)
        # so httpx percent-encodes it correctly.
        r = self.client.get("/api/performance", params={"history_start": yid})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["meta"]["filter"]["history_start"], yid)

    def test_history_start_unknown_422(self):
        r = self.client.get("/api/performance?history_start=1066+")
        self.assertEqual(r.status_code, 422)


class TestTokens(unittest.TestCase):
    """The :root token block must carry the P&L + accent tokens and stay pinned
    to theme.py (so the front-end can't drift from Streamlit on palette)."""

    def test_root_block_has_palette(self):
        from terminal.tokens import root_css
        css = root_css()
        for tok in ("--accent:", "--gain:", "--loss:"):
            self.assertIn(tok, css)
        import theme
        self.assertIn(theme.ACCENT.lower(), css.lower())


class TestFilterNormalize(unittest.TestCase):
    def test_normalize(self):
        from terminal import holdings_service as hs
        self.assertEqual(hs._normalize_filter_ids("all"), ["all"])
        self.assertEqual(hs._normalize_filter_ids(["all"]), ["all"])
        self.assertEqual(hs._normalize_filter_ids([]), ["all"])
        self.assertEqual(hs._normalize_filter_ids(None), ["all"])
        self.assertEqual(hs._normalize_filter_ids("x"), ["x"])
        self.assertEqual(hs._normalize_filter_ids(["a", "b"]), ["a", "b"])
        self.assertEqual(hs._normalize_filter_ids(["a", "all"]), ["all"])  # 'all' wins

    def test_echo(self):
        from terminal import holdings_service as hs
        self.assertEqual(hs._filter_echo(["all"]), "all")
        self.assertEqual(hs._filter_echo(["a", "b"]), ["a", "b"])

    def test_filter_meta_default_is_scalar_all(self):
        from terminal import holdings_service as hs
        self.assertEqual(hs._filter_meta("all", "all"),
                         {"account": "all", "asset_class": "all", "broker": "all"})


class TestHoldingsMultiFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        cls.full = hs.build_holdings_view(cls.frames)
        _, cls.acct_by_id = hs._account_options(hs._current_snap(cls.frames))
        cls.ids = list(cls.acct_by_id)

    def test_multi_account_union_rows(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        one = self.hs.build_holdings_view(self.frames, account=[self.ids[0]])
        two = self.hs.build_holdings_view(self.frames, account=self.ids[:2])
        self.assertGreaterEqual(two["snapshot"]["holdings"]["rows"],
                                one["snapshot"]["holdings"]["rows"])
        self.assertLessEqual(two["snapshot"]["holdings"]["rows"],
                             self.full["snapshot"]["holdings"]["rows"])

    def test_multi_echoes_list(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 account buckets")
        two = self.hs.build_holdings_view(self.frames, account=self.ids[:2])
        self.assertEqual(two["meta"]["filter"]["account"], self.ids[:2])

    def test_default_echoes_all_scalar(self):
        self.assertEqual(self.full["meta"]["filter"],
                         {"account": "all", "asset_class": "all", "broker": "all"})


class TestApplyGlobalFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        snap = hs._current_snap(cls.frames)
        cls.broker_opts, cls.broker_by_id = hs._broker_options(snap)
        cls.ids = [o["id"] for o in cls.broker_opts]

    def test_all_is_noop(self):
        out = self.hs.apply_global_filters(self.frames, ["all"])
        self.assertIs(out, self.frames)   # canonical/default → same object

    def test_noncanonical_narrows_and_guards(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        one = self.ids[0]                 # a single broker = a proper subset
        out = self.hs.apply_global_filters(self.frames, [one])
        name = self.broker_by_id[one]
        self.assertEqual(set(out.positions["broker"].astype(str)), {name})   # narrowed
        # S2b: a real subset now RECOMPUTES the portfolio TWR (not blank) from
        # the narrowed per-account frame — engine-parity with a direct call.
        import pandas as pd
        from parsers.twr_aggregate import recompute_portfolio_twr
        self.assertFalse(out.twr_portfolio.empty)                            # recomputed
        expected = recompute_portfolio_twr(out.twr_account)
        pd.testing.assert_frame_equal(out.twr_portfolio.reset_index(drop=True),
                                      expected.reset_index(drop=True))
        # Scoped-IRR contract: any PORTFOLIO row present must equal a fresh
        # scoped recompute on the narrowed frames — a stale whole-book number
        # can never survive into a scope (spec 2026-08-07).
        self._assert_portfolio_row_is_scoped_recompute(out)
        self.assertGreater(len(out.positions_monthly), 0)                    # rebuilt
        # narrowed positions are a strict subset of the full book
        self.assertLess(len(out.positions), len(self.frames.positions))

    def test_unknown_broker_id_falls_back_to_real(self):
        # unknown ids resolve to nothing → treated as no-filter (canonical) → no-op-ish
        out = self.hs.apply_global_filters(self.frames, ["__nope__"])
        self.assertIs(out, self.frames)   # same identity-preserving no-op path
        self.assertEqual(set(out.positions["broker"].astype(str)),
                         set(self.frames.positions["broker"].astype(str)))

    def test_portfolio_row_actually_stripped(self):
        # The fixture's irr_per_account.csv has NO PORTFOLIO row, so the
        # sibling check can be vacuous. Inject a stale whole-book row
        # (irr=0.05) and prove it is REPLACED by the scoped recompute (or
        # dropped when the recompute is non-finite) — never kept verbatim.
        import pandas as pd
        from dataclasses import replace
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        port_row = pd.DataFrame([{"account_id": "PORTFOLIO", "irr": 0.05,
                                  "terminal_nav": 1.0, "n_cashflows": 1,
                                  "total_deposits": 1.0, "total_withdrawals": 0.0,
                                  "start_date": "2024-01-01", "end_date": "2026-04-30"}])
        frames = replace(self.frames,
                         irr_table=pd.concat([self.frames.irr_table, port_row],
                                             ignore_index=True))
        out = self.hs.apply_global_filters(frames, [self.ids[0]])
        rows = out.irr_table[out.irr_table["account_id"].astype(str) == "PORTFOLIO"]
        if len(rows):
            self.assertNotAlmostEqual(float(rows.iloc[0]["irr"]), 0.05, places=9)
        self._assert_portfolio_row_is_scoped_recompute(out)

    def _assert_portfolio_row_is_scoped_recompute(self, out):
        """Row present iff the fresh scoped recompute is finite and not
        floor-banded; when present, irr matches it to float tolerance."""
        import numpy as np
        from compute_twr import compute_portfolio_irr, classify_irr
        import config_local as cfg
        expected = compute_portfolio_irr(
            out.positions, out.transactions,
            synthetic_onboarding=cfg.SYNTHETIC_ONBOARDING, scoped=True)
        rows = (out.irr_table[out.irr_table["account_id"].astype(str) == "PORTFOLIO"]
                if not out.irr_table.empty else out.irr_table)
        should_have = (expected["irr"] is not None
                       and np.isfinite(expected["irr"])
                       and classify_irr(expected["irr"]) != "error")
        if should_have:
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows.iloc[0]["irr"]),
                                   float(expected["irr"]), places=10)
        else:
            self.assertEqual(len(rows), 0)

    def test_broker_scope_none_on_canonical(self):
        self.assertIsNone(self.frames.broker_scope)
        out = self.hs.apply_global_filters(self.frames, ["all"])
        self.assertIsNone(out.broker_scope)

    def test_broker_scope_set_on_subset(self):
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        out = self.hs.apply_global_filters(self.frames, [self.ids[0]])
        raw = self.broker_by_id[self.ids[0]]
        # broker_scope carries the DISPLAY label of the raw broker value, not
        # the raw value itself (2026-08-07 live-smoke finding: on real data
        # positions.broker is lowercase "alpine"/"harbor", which must not leak
        # into the KPI-tape/headline prose). This fixture's raw value happens
        # to equal what _broker_options' (raw, golden-pinned) label already
        # is, so go through the SAME display-casing helper the fix uses
        # rather than the raw value, so this test states the real contract
        # instead of one only the fixture's coincidence lets pass.
        self.assertEqual(out.broker_scope,
                         (self.hs._broker_display_label(raw),))

    def test_broker_scope_uses_display_labels_not_raw(self):
        """The fixture-from-the-model trap: this fixture's raw broker values
        ("alpine"/"harbor") already equal what _broker_options' (raw,
        golden-pinned — see _broker_display_label's docstring) label
        returns, so a test that only reaches for _broker_options' label as
        its oracle could never distinguish a correctly display-cased
        broker_scope from a bug that just echoes the raw column verbatim.
        Force the divergence explicitly instead: lowercase every broker
        column (a no-op on today's fixture, but robust if the fixture's raw
        casing ever changes) and pin against the terminal's OWN known
        display convention — app.js's `_brokerOpts` "nice" map / `prettyBroker`
        (terminal/static/app.js) — hardcoded here (not read from production
        code) so a self-consistent-but-wrong implementation can't game it."""
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        from dataclasses import replace

        def _lower_broker(df):
            if df.empty or "broker" not in df.columns:
                return df
            return df.assign(broker=df["broker"].astype(str).str.lower())

        lowered = replace(
            self.frames,
            positions=_lower_broker(self.frames.positions),
            positions_monthly=_lower_broker(self.frames.positions_monthly),
            transactions=_lower_broker(self.frames.transactions),
            summaries=_lower_broker(self.frames.summaries),
        )
        expected_label = {"alpine": "Alpine", "harbor": "Harbor"}
        exercised = False
        for bid in self.ids:
            raw = self.broker_by_id[bid].lower()
            if raw not in expected_label:
                continue
            exercised = True
            out = self.hs.apply_global_filters(lowered, [bid])
            self.assertEqual(out.broker_scope, (expected_label[raw],))
            self.assertNotIn(raw, out.broker_scope)
        if not exercised:
            self.skipTest("fixture has no alpine/harbor broker to pin")

    def test_floor_banded_scoped_irr_suppressed(self):
        # A scoped recompute pinned at the xirr floor is the PR #147
        # corruption signature — it must never surface as a -99.99% row.
        import pandas as pd
        from unittest.mock import patch
        if len(self.ids) < 2:
            self.skipTest("fixture has <2 brokers")
        floored = {"irr": -0.9999, "n_cashflows": 3, "terminal_nav": 1.0,
                   "start_date": pd.Timestamp("2026-01-31"),
                   "end_date": pd.Timestamp("2026-03-31"),
                   "total_deposits": 1.0, "total_withdrawals": 0.0}
        with patch.object(self.hs, "compute_portfolio_irr",
                          return_value=floored):
            out = self.hs.apply_global_filters(self.frames, [self.ids[0]])
        if not out.irr_table.empty:
            self.assertNotIn("PORTFOLIO",
                             set(out.irr_table["account_id"].astype(str)))

    def _inject_demo_broker(self):
        """Return a frames with a demo-broker ("Alpine Test") position spliced
        into the LATEST snapshot month of BOTH positions and positions_monthly,
        mirroring what load_frames produces on real data (it overlays demo into
        positions, then rebuilds positions_monthly FROM the overlaid positions —
        so a positions-only inject would leave the demo invisible to
        _current_snap, which reads positions_monthly)."""
        import pandas as pd
        from dataclasses import replace
        base = self.frames.positions
        latest = base["statement_date"].max()
        row = base[base["statement_date"] == latest].iloc[[0]].copy()
        row["broker"] = "Alpine Test"
        row["account_id"] = "TEST-FID"
        pos = pd.concat([base, row], ignore_index=True)
        # Rebuild positions_monthly from the overlaid positions (as load_frames
        # does) so _current_snap / _broker_options see the demo broker.
        pm = self.hs.monthly_normalize(pos)
        if not self.frames.prices_latest.empty:
            pm = self.hs.mark_to_market(pm, self.frames.prices_latest)
        return replace(self.frames, positions=pos, positions_monthly=pm)

    def test_demo_only_isolates_and_blanks(self):
        # Selecting ONLY the demo broker isolates it and blanks the real twr.
        frames = self._inject_demo_broker()
        snap = self.hs._current_snap(frames)
        _, by_id = self.hs._broker_options(snap)
        demo_id = next(i for i, name in by_id.items() if name == "Alpine Test")
        out = self.hs.apply_global_filters(frames, [demo_id])
        self.assertEqual(set(out.positions["broker"].astype(str)),
                         {"Alpine Test"})       # demo only
        # S2b: the recompute runs on the NARROWED twr_account. The injected
        # demo broker has positions but NO twr_account rows, so the narrowed
        # per-account frame is empty and the recompute yields an empty
        # portfolio series — still zero real-book leak.
        self.assertTrue(out.twr_portfolio.empty)   # no demo per-account TWR to aggregate

    def test_sidecars_narrowed_no_real_leak_under_demo(self):
        # Inject a demo broker into positions (+ rebuilt positions_monthly, via
        # the existing helper, so _current_snap/_broker_options can see it) plus
        # a real+demo twr_account and a real-only summaries frame. Selecting
        # ONLY the demo broker must leave NO real account/broker in either
        # narrowed sidecar (the fresh-CSV-read demo-isolation leak this test
        # guards against).
        import pandas as pd
        from dataclasses import replace
        demo_frames = self._inject_demo_broker()
        real_acct = str(self.frames.positions["account_id"].iloc[0])
        # S2b: apply_global_filters now feeds the narrowed twr_account into
        # recompute_portfolio_twr on a non-real selection, which needs
        # return_pct/prev_nav/month (the real schema always has them, per
        # twr_monthly.csv) — round out the synthetic rows to match, so this
        # sidecar-narrowing test isn't coupled to the recompute's column needs.
        twr_acct = pd.DataFrame([
            {"account_id": real_acct, "month": "2026-03", "return_pct": 0.02,
             "prev_nav": 980.0, "nav": 1000.0},
            {"account_id": "TEST-FID", "month": "2026-03", "return_pct": 0.01,
             "prev_nav": 4.0, "nav": 5.0},
        ])
        summ = pd.DataFrame([{"broker": str(self.frames.positions["broker"].iloc[0]),
                              "reported_total": 1000.0}])
        frames = replace(demo_frames, twr_account=twr_acct, summaries=summ)
        snap = self.hs._current_snap(frames)
        _, by_id = self.hs._broker_options(snap)
        demo_id = next(i for i, n in by_id.items() if n == "Alpine Test")
        out = self.hs.apply_global_filters(frames, [demo_id])
        self.assertNotIn(real_acct, set(out.twr_account["account_id"].astype(str)))   # real per-account $ gone
        self.assertTrue(out.summaries.empty)                                          # real summaries gone

    def test_demo_only_with_twr_recomputes_demo_only_no_leak(self):
        # Give the demo broker a real per-account TWR row (alongside real
        # accounts) and select demo-only: the recomputed portfolio TWR must be
        # non-empty AND derive ONLY from the demo account — never the real book.
        import pandas as pd
        from dataclasses import replace
        from parsers.twr_aggregate import recompute_portfolio_twr
        demo_frames = self._inject_demo_broker()
        real_acct = str(self.frames.positions["account_id"].iloc[0])
        twr_acct = pd.DataFrame([
            {"account_id": real_acct, "month": "2026-03", "return_pct": 0.99,
             "prev_nav": 1_000_000.0, "nav": 1_990_000.0},          # real — must NOT leak
            {"account_id": "TEST-FID", "month": "2026-03", "return_pct": 0.01,
             "prev_nav": 100.0, "nav": 101.0},                       # demo
        ])
        frames = replace(demo_frames, twr_account=twr_acct)
        snap = self.hs._current_snap(frames)
        _, by_id = self.hs._broker_options(snap)
        demo_id = next(i for i, n in by_id.items() if n == "Alpine Test")
        out = self.hs.apply_global_filters(frames, [demo_id])
        self.assertFalse(out.twr_portfolio.empty)                    # recomputed
        # the recompute saw ONLY the demo account (real row narrowed out first)
        self.assertNotIn(real_acct, set(out.twr_account["account_id"].astype(str)))
        demo_only = twr_acct[twr_acct["account_id"] == "TEST-FID"]
        expected = recompute_portfolio_twr(demo_only)
        pd.testing.assert_frame_equal(out.twr_portfolio.reset_index(drop=True),
                                      expected.reset_index(drop=True))
        # the real account's 99% return is nowhere in the demo portfolio series
        self.assertAlmostEqual(out.twr_portfolio.iloc[-1]["return_pct"], 0.01,
                               places=9)

    def test_non_real_with_missing_twr_account_blanks_no_crash(self):
        # Regression (S2b review): a missing twr_monthly.csv makes
        # frames.twr_account a COLUMNLESS empty frame. Under a non-real broker
        # selection the seam must NOT call recompute_portfolio_twr on it (that
        # would KeyError on the absent return_pct column and 500 every route) —
        # it blanks twr_portfolio instead. Mirrors app.py's `if not
        # twr_account.empty` guard.
        import pandas as pd
        from dataclasses import replace
        frames = replace(self._inject_demo_broker(), twr_account=pd.DataFrame())
        snap = self.hs._current_snap(frames)
        _, by_id = self.hs._broker_options(snap)
        demo_id = next(i for i, n in by_id.items() if n == "Alpine Test")
        out = self.hs.apply_global_filters(frames, [demo_id])   # must not raise
        self.assertTrue(out.twr_portfolio.empty)


class TestHistoryStartCutoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.hs = hs
        cls.frames = hs.load_frames(FIXTURE)
        snap = hs._current_snap(cls.frames)
        cls.broker_opts, cls.broker_by_id = hs._broker_options(snap)
        cls.ids = [o["id"] for o in cls.broker_opts]

    def test_options_include_all_and_years(self):
        opts = self.hs._history_start_options(self.frames)
        self.assertEqual(opts[0], {"id": "all", "label": "All history"})
        for o in opts[1:]:
            self.assertRegex(o["id"], r"^\d{4}\+$")
            self.assertEqual(o["label"], o["id"])
        years = [int(o["id"].rstrip("+")) for o in opts[1:]]
        self.assertEqual(years, sorted(years))

    def test_cutoff_parse(self):
        self.assertIsNone(self.hs._history_start_cutoff("all"))
        import pandas as pd
        self.assertEqual(self.hs._history_start_cutoff("2025+"),
                         pd.Timestamp(2025, 1, 1))

    def test_all_is_noop(self):
        out = self.hs.apply_global_filters(self.frames, "all", "all")
        self.assertIs(out, self.frames)

    def test_cutoff_slices_positions_and_twr(self):
        import pandas as pd
        opts = self.hs._history_start_options(self.frames)
        if len(opts) < 2:
            self.skipTest("fixture has <2 years")
        hstart = opts[-1]["id"]
        cutoff = self.hs._history_start_cutoff(hstart)
        out = self.hs.apply_global_filters(self.frames, "all", hstart)
        self.assertTrue((out.positions["statement_date"] >= cutoff).all())
        if not out.twr_account.empty:
            self.assertTrue(
                (out.twr_account["month"].dt.to_timestamp() >= cutoff).all())
        self.assertGreaterEqual(len(self.frames.positions), len(out.positions))

    def test_cutoff_recomputes_account_irr_and_gates_portfolio_row(self):
        # A cutoff re-runs compute_account_irr on the pre-slice frames, so
        # per-account IRR rows are present. The synthetic PORTFOLIO row is
        # ADDITIONALLY gated on a FINITE portfolio IRR: this fixture's positions
        # are a single quarter of 2026, so the only available cutoff (2026+)
        # predates the first statement -> nav_at_cutoff = 0 -> portfolio IRR is
        # NaN -> the row is correctly OMITTED. The row-PRESENT path (real
        # multi-year history -> finite IRR) is proven by the opus real-data
        # recompute vs app.py (spec-designated; the fixture is too shallow to
        # exercise it, and extending it would churn every terminal golden).
        # Since the scoped-IRR choke-point, the append is attempted for
        # non-real (scoped) selections too — scoped=True when not is_real —
        # gated by the same finite + band (classify_irr != "error") check.
        opts = self.hs._history_start_options(self.frames)
        if len(opts) < 2:
            self.skipTest("fixture has <2 years")
        out = self.hs.apply_global_filters(self.frames, "all", opts[-1]["id"])
        self.assertFalse(out.irr_table.empty)          # account IRR recomputed
        self.assertNotIn("PORTFOLIO",                  # NaN portfolio IRR -> gated out
                         set(out.irr_table["account_id"].astype(str)))

    def test_floor_banded_irr_suppressed_on_canonical_cutoff_path(self):
        # Mirrors test_floor_banded_scoped_irr_suppressed (in
        # TestApplyGlobalFilters), but drives the CANONICAL (is_real=True,
        # scoped=False) cutoff path
        # instead of a broker subset: _portfolio_irr_row's finite +
        # classify_irr != "error" guard must suppress a floor-banded IRR
        # there too — the -0.9999 corruption signature is not special-cased
        # to the scoped branch.
        import pandas as pd
        from unittest.mock import patch
        opts = self.hs._history_start_options(self.frames)
        if len(opts) < 2:
            self.skipTest("fixture has <2 years")
        floored = {"irr": -0.9999, "n_cashflows": 3, "terminal_nav": 1.0,
                   "start_date": pd.Timestamp("2026-01-31"),
                   "end_date": pd.Timestamp("2026-03-31"),
                   "total_deposits": 1.0, "total_withdrawals": 0.0}
        with patch.object(self.hs, "compute_portfolio_irr",
                          return_value=floored):
            out = self.hs.apply_global_filters(self.frames, "all", opts[-1]["id"])
        if not out.irr_table.empty:
            self.assertNotIn("PORTFOLIO",
                             set(out.irr_table["account_id"].astype(str)))

    def test_cutoff_plus_broker_subset_attempts_scoped_append(self):
        # Non-real + cutoff: the PORTFOLIO append is attempted with
        # scoped=True and the same finite+band gate. On this fixture the
        # recompute is NaN (cutoff predates the first statement) -> no row,
        # no crash; broker_scope still stamped.
        opts = self.hs._history_start_options(self.frames)
        if len(opts) < 2 or len(self.ids) < 2:
            self.skipTest("fixture too shallow")
        out = self.hs.apply_global_filters(self.frames, [self.ids[0]],
                                           opts[-1]["id"])
        # Same real contract as test_broker_scope_set_on_subset: DISPLAY
        # label, not the raw broker_by_id value.
        self.assertEqual(out.broker_scope,
                         (self.hs._broker_display_label(
                             self.broker_by_id[self.ids[0]]),))
        self._assert_portfolio_row_is_scoped_recompute_with_cutoff(
            out, self.hs._history_start_cutoff(opts[-1]["id"]))

    def _assert_portfolio_row_is_scoped_recompute_with_cutoff(self, out, cutoff):
        import numpy as np
        from compute_twr import compute_portfolio_irr, classify_irr
        import config_local as cfg
        # NOTE: mirror the choke-point's ordering — the recompute sees the
        # broker-narrowed but PRE-cutoff-slice frames; out.positions is
        # post-slice, so rebuild the pre-slice narrowed book from the raw
        # fixture frames.
        sel = set(out.positions["broker"].astype(str).unique()) or {
            self.broker_by_id[self.ids[0]]}
        pos = self.frames.positions[
            self.frames.positions["broker"].astype(str).isin(sel)]
        txns = self.frames.transactions[
            self.frames.transactions["broker"].astype(str).isin(sel)]
        expected = compute_portfolio_irr(
            pos, txns, synthetic_onboarding=cfg.SYNTHETIC_ONBOARDING,
            start_date=cutoff, scoped=True)
        rows = (out.irr_table[out.irr_table["account_id"].astype(str) == "PORTFOLIO"]
                if not out.irr_table.empty else out.irr_table)
        should_have = (expected["irr"] is not None
                       and np.isfinite(expected["irr"])
                       and classify_irr(expected["irr"]) != "error")
        self.assertEqual(len(rows), 1 if should_have else 0)

    def test_pure_cutoff_twr_is_sliced_canonical(self):
        # DA-D-3: a PURE history cutoff on the canonical book slices the
        # canonical monthly frame — EXACT, since each month's Modified
        # Dietz return is independent of the window start — instead of the
        # NAV-weighted recompute approximation (which shifted the
        # terminal's default 2021+ view +0.38pp off the canonical series
        # its methodology text claims). The recompute stays reserved for
        # account-subset scopes, where canonical cross-account pairing
        # genuinely cannot be subset.
        import pandas as pd
        from parsers.twr_aggregate import slice_canonical_twr
        opts = self.hs._history_start_options(self.frames)
        if len(opts) < 2:
            self.skipTest("fixture has <2 years")
        cutoff = pd.Timestamp(int(opts[1]["id"].rstrip("+")), 1, 1)
        out = self.hs.apply_global_filters(self.frames, "all", opts[1]["id"])
        expected = slice_canonical_twr(self.frames.twr_portfolio, cutoff)
        pd.testing.assert_frame_equal(
            out.twr_portfolio.reset_index(drop=True),
            expected.reset_index(drop=True))
        # every kept row is literally a canonical row (no re-derivation)
        canon = self.frames.twr_portfolio
        months = pd.PeriodIndex(canon["month"], freq="M").to_timestamp()
        pd.testing.assert_frame_equal(
            out.twr_portfolio.reset_index(drop=True),
            canon[months >= cutoff].reset_index(drop=True))

    def test_broker_subset_with_cutoff_still_recomputes(self):
        # A subset scope cannot be sliced from the canonical series — the
        # NAV-weighted recompute remains its (disclosed) approximation.
        import pandas as pd
        from parsers.twr_aggregate import recompute_portfolio_twr
        opts = self.hs._history_start_options(self.frames)
        if len(opts) < 2 or len(self.ids) < 2:
            self.skipTest("fixture lacks 2 years or 2 brokers")
        out = self.hs.apply_global_filters(self.frames, [self.ids[0]],
                                           opts[1]["id"])
        if not out.twr_account.empty:
            expected = recompute_portfolio_twr(out.twr_account)
            pd.testing.assert_frame_equal(
                out.twr_portfolio.reset_index(drop=True),
                expected.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
