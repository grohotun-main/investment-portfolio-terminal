# tests/test_terminal_tax.py
"""Terminal Tax tab (tax S2 slice 2b): service, route, golden.

First consumer of the gated lot ledger. The synthetic fixture carries
lots.csv + lots_meta.json beside the synth book: reconstructed and printed
evidence, long/short/unknown terms, an unpriced symbol (CCC), and a TEST-X
account absent from positions (fail-closed exclusion). IRA exclusion has no
fixture account, so it is pinned on constructed frames.
"""
import json
import math
import os
import shutil
import sys
import unittest
from datetime import date, timedelta
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from parsers.lot_engine import classify_term     # noqa: E402
from parsers import tax_estimate                 # noqa: E402
from parsers.tax_scanner import price_map as scanner_price_map  # noqa: E402
from terminal import holdings_service as hs      # noqa: E402
from terminal import tax_service as txs          # noqa: E402

# Pinned so terms and the golden are deterministic (the LT/ST boundary is
# genuinely today-dependent). The live route uses the real today.
ASOF = date(2026, 6, 28)


def _view(**kw):
    frames = hs.load_frames(FIXTURE)
    return txs.build_tax_view(frames, FIXTURE, asof=ASOF, **kw)


def _estimate_view():
    """Deterministic estimate payload for the golden: fixture book,
    FULL synthetic profile via overrides (every field explicit, so a
    dev box's real TAX_PROFILE can never leak in), one 1-share sim leg
    on the first priced known-term lot."""
    tmp = TemporaryDirectory()
    d = _fixture_copy_dir(tmp)
    meta = json.loads((d / "lots_meta.json").read_text(encoding="utf-8"))
    meta["realized_ytd"] = {
        "year": 2026,
        "by_account": {"TEST-A": {"short": {
            "gains": 1_000.0, "losses": -400.0, "net": 600.0,
            "closes": 2}}},
        "notes": {"excludes_alpine_options": True,
                  "options_source": "harbor_printed_confirms",
                  "broker_unresolved": 0}}
    (d / "lots_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    frames = hs.load_frames(d)
    with mock.patch.object(txs, "_config_profile", return_value={}):
        lots = txs.build_tax_view(frames, d, asof=ASOF)["lots"]
        lot = next(x for x in lots if x["market_value"] is not None
                   and x["term"] in ("short", "long"))
        out = txs.build_tax_estimate(
            frames, d, overrides=dict(FULL_PROFILE),
            sim=[{"lot_id": lot["lot_id"], "qty": 1.0}], asof=ASOF)
    tmp.cleanup()
    return out


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (house convention,
    copied from test_terminal_factor) — the golden's derived floats must
    reproduce across BLAS/pandas builds, structure and strings exactly."""
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
        return (None if math.isclose(float(a), float(b), rel_tol=rel,
                                     abs_tol=abs_)
                else f"{path}: {a!r} !~ {b!r}")
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _fixture_copy_dir(tmp: TemporaryDirectory) -> Path:
    """A temp data dir seeded with the synth fixture's files."""
    d = Path(tmp.name)
    for name in ("positions.csv", "transactions.csv", "prices_latest.csv",
                 "twr_monthly.csv", "twr_portfolio.csv", "summaries.csv",
                 "lots.csv", "lots_meta.json"):
        src = FIXTURE / name
        if src.exists():
            shutil.copy(src, d / name)
    return d


class TestTaxContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.view = _view()

    def test_contract_keys(self):
        self.assertEqual(self.view["kind"], "tax")
        for key in ("meta", "summary", "lots"):
            self.assertIn(key, self.view)
        for key in ("built_at", "open_lots", "gate", "joined_bands",
                    "stale", "stale_reason", "asof", "filter"):
            self.assertIn(key, self.view["meta"])
        for key in ("accounts", "totals", "by_evidence", "by_band",
                    "excluded", "silent_share_note"):
            self.assertIn(key, self.view["summary"])
        # deep-link support: whole-book tabs ship the global picker option
        # lists so a ?tab=tax-first load doesn't leave the pickers empty
        for key in ("accounts", "classes"):
            self.assertIn(key, self.view["meta"])

    def test_full_view_jsonable_no_nan(self):
        # allow_nan=False is the route's serialization contract
        json.dumps(self.view, allow_nan=False)

    def test_lot_row_shape(self):
        row = self.view["lots"][0]
        for key in ("account_id", "account_label", "symbol", "instrument_key",
                    "type", "origin", "acquired_date", "term",
                    "days_to_long_term",
                    "quantity_remaining", "basis_remaining", "price",
                    "price_source", "price_asof", "market_value",
                    "unrealized_gl", "basis_evidence", "band"):
            self.assertIn(key, row)

    def test_days_to_long_term_matches_term(self):
        for row in self.view["lots"]:
            d = row["days_to_long_term"]
            if row["term"] == "short":
                self.assertIsInstance(d, int)
                self.assertGreaterEqual(d, 1)
            else:  # long / unknown
                self.assertIsNone(d)

    def test_realized_by_account_present_and_fail_closed(self):
        rz = self.view["summary"]["realized_ytd"]
        self.assertIn("by_account", rz)
        ids = {r["account_id"] for r in rz["by_account"]}
        # TEST-X carries a slot in the fixture meta but has no positions
        # row, so its type is unknowable -> excluded fail-closed
        self.assertEqual(ids, {"TEST-A", "TEST-B", "TEST-C"})
        self.assertEqual(rz["options_uncovered"], 3)


class TestTaxNumbers(unittest.TestCase):
    """The fixture book, computed by hand (prices: SPY 570, AAA 102,
    BBB 50.5; CCC unpriced)."""

    @classmethod
    def setUpClass(cls):
        cls.view = _view()
        cls.by_key = {(r["account_id"], r["symbol"], r["acquired_date"]): r
                      for r in cls.view["lots"]}

    def test_taxable_lots_only_and_sorted(self):
        accts = [r["account_id"] for r in self.view["lots"]]
        self.assertEqual(sorted(set(accts)), ["TEST-A", "TEST-B", "TEST-C"])
        self.assertNotIn("TEST-X", accts)
        self.assertEqual(len(self.view["lots"]), 6)

    def test_unrealized_math(self):
        spy_a = self.by_key[("TEST-A", "SPY", "2024-05-10")]
        self.assertEqual(spy_a["market_value"], 34_200.0)      # 60 * 570
        self.assertEqual(spy_a["unrealized_gl"], 7_200.0)      # - 27,000
        bbb = self.by_key[("TEST-B", "BBB", "2026-01-05")]
        self.assertEqual(bbb["unrealized_gl"], -150.0)         # a loss row

    def test_terms(self):
        self.assertEqual(self.by_key[("TEST-A", "SPY", "2024-05-10")]["term"],
                         "long")
        self.assertEqual(self.by_key[("TEST-A", "SPY", "2025-11-03")]["term"],
                         "short")
        self.assertEqual(self.by_key[("TEST-A", "AAA", None)]["term"],
                         "unknown")

    def test_unpriced_lot_is_null_never_zero(self):
        ccc = self.by_key[("TEST-C", "CCC", "2026-02-10")]
        self.assertIsNone(ccc["price"])
        self.assertIsNone(ccc["market_value"])
        self.assertIsNone(ccc["unrealized_gl"])
        self.assertEqual(ccc["basis_remaining"], 900.0)

    def test_totals(self):
        t = self.view["summary"]["totals"]
        self.assertEqual(t["basis"], 99_200.0)
        self.assertEqual(t["market_value"], 121_050.0)   # priced lots only
        self.assertEqual(t["unrealized_gl"], 22_750.0)
        self.assertEqual(t["priced_lots"], 5)
        self.assertEqual(t["unpriced_lots"], 1)

    def test_evidence_and_band_census_exclude_nontaxable(self):
        self.assertEqual(self.view["summary"]["by_evidence"],
                         {"reconstructed": 5, "printed": 1})
        self.assertEqual(self.view["summary"]["by_band"],
                         {"ok": 5, "watch": 1})

    def test_unknown_type_account_excluded_fail_closed(self):
        self.assertEqual(self.view["summary"]["excluded"],
                         {"ira_accounts": 0, "unknown_accounts": 1})

    def test_meta_carries_gate_and_freshness(self):
        m = self.view["meta"]
        self.assertEqual(m["open_lots"], 7)
        self.assertTrue(m["gate"]["passed"])
        self.assertFalse(m["stale"])
        self.assertIsNone(m["stale_reason"])
        self.assertEqual(m["asof"], "2026-06-28")

    def test_silent_note_states_both_gate_terms(self):
        # reworded 2026-07-30 (TK read "silent on 31.68%" as a third of
        # the lots being INVISIBLE): the note now leads with "every lot
        # is shown" and describes the remainder as the broker's own
        # printed figures — provenance, not visibility
        note = self.view["summary"]["silent_share_note"]
        self.assertIn("99.5", note)
        self.assertIn("70.0", note)
        self.assertIn("every lot is shown", note)
        self.assertIn("printed figures", note)

    def test_unpriced_only_account_summary_is_null_never_zero(self):
        # TEST-C's single lot has no live mark: the account summary must
        # say "no market value", not $0.00
        by_id = {a["id"]: a for a in self.view["summary"]["accounts"]}
        self.assertIsNone(by_id["TEST-C"]["market_value"])
        self.assertIsNone(by_id["TEST-C"]["unrealized_gl"])
        self.assertEqual(by_id["TEST-C"]["basis"], 900.0)


class TestTaxCore(unittest.TestCase):
    """Constructed-frame units for what the fixture cannot carry."""

    def test_term_boundary_365_366(self):
        self.assertEqual(txs.term_of(pd.Timestamp("2025-06-28"),
                                     date(2026, 6, 28)), "short")   # 365d
        self.assertEqual(txs.term_of(pd.Timestamp("2025-06-27"),
                                     date(2026, 6, 28)), "long")    # 366d
        self.assertEqual(txs.term_of(pd.NaT, date(2026, 6, 28)), "unknown")

    def test_the_tab_and_the_ledger_agree_on_a_leap_spanning_anniversary(self):
        # The two views of one lot must never disagree about its term. A
        # day-count rule reads 366 days here (Feb 29 2024 falls inside the
        # span) and calls it long; the IRS rule is "more than one year",
        # so ON the anniversary it is still SHORT — long starts the day
        # after. The ledger's classify_term is the single rule; the tab
        # must return exactly what it returns, for every date.
        acquired = pd.Timestamp("2024-02-28")
        for asof, expected in ((date(2025, 2, 27), "short"),
                               (date(2025, 2, 28), "short"),   # anniversary
                               (date(2025, 3, 1), "long")):
            with self.subTest(asof=asof):
                self.assertEqual(txs.term_of(acquired, asof), expected)
                self.assertEqual(txs.term_of(acquired, asof),
                                 classify_term(acquired, asof))

    def test_the_service_shares_the_ledgers_price_map(self):
        # one pricing path for the whole tab: the same helper the harvest
        # scanner marks with, not a second copy that could drift
        self.assertIs(txs.price_map, scanner_price_map)

    def test_ira_accounts_are_excluded(self):
        self.assertIs(txs.taxable_of("roth_ira"), False)
        self.assertIs(txs.taxable_of("traditional_ira"), False)
        self.assertIs(txs.taxable_of("individual_tod"), True)
        self.assertIs(txs.taxable_of("harbor_brokerage"), True)
        self.assertIs(txs.taxable_of(""), None)
        self.assertIs(txs.taxable_of(None), None)

    def test_account_type_latest_wins(self):
        pos = pd.DataFrame({
            "account_id": ["A-1", "A-1"],
            "account_type": ["individual_tod", "roth_ira"],
            "statement_date": ["2025-01-31", "2026-01-31"],
        })
        self.assertEqual(txs.account_types(pos), {"A-1": "roth_ira"})

    def test_staleness_transactions_moved(self):
        meta = {"inputs": {"transactions_rows": 999,
                           "positions_max_month": "2026-04"}}
        stale, reason = txs.freshness(meta, tx_rows=11,
                                      positions_max_month="2026-04")
        self.assertTrue(stale)
        self.assertIn("transactions", reason)

    def test_staleness_frontier_moved(self):
        meta = {"inputs": {"transactions_rows": 11,
                           "positions_max_month": "2026-04"}}
        stale, reason = txs.freshness(meta, tx_rows=11,
                                      positions_max_month="2026-05")
        self.assertTrue(stale)
        self.assertIn("2026-05", reason)

    def test_staleness_missing_meta(self):
        stale, reason = txs.freshness(None, tx_rows=11,
                                      positions_max_month="2026-04")
        self.assertTrue(stale)
        self.assertIn("metadata", reason)

    def test_fresh_when_inputs_match(self):
        meta = {"inputs": {"transactions_rows": 11,
                           "positions_max_month": "2026-04"}}
        self.assertEqual(txs.freshness(meta, tx_rows=11,
                                       positions_max_month="2026-04"),
                         (False, None))


class TestTaxMissingFiles(unittest.TestCase):
    def test_missing_lots_csv_is_an_error_view(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("positions.csv", "transactions.csv",
                         "prices_latest.csv", "twr_monthly.csv",
                         "twr_portfolio.csv", "summaries.csv"):
                src = FIXTURE / name
                if src.exists():
                    (d / name).write_bytes(src.read_bytes())
            frames = hs.load_frames(d)
            view = txs.build_tax_view(frames, d, asof=ASOF)
        self.assertEqual(view["kind"], "error")
        self.assertIn("build_lots", view["reason"])
        self.assertIn("filter", view["meta"])

    def test_missing_meta_serves_lots_stale(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("positions.csv", "transactions.csv",
                         "prices_latest.csv", "twr_monthly.csv",
                         "twr_portfolio.csv", "summaries.csv", "lots.csv"):
                src = FIXTURE / name
                if src.exists():
                    (d / name).write_bytes(src.read_bytes())
            frames = hs.load_frames(d)
            view = txs.build_tax_view(frames, d, asof=ASOF)
        self.assertEqual(view["kind"], "tax")
        self.assertTrue(view["meta"]["stale"])
        self.assertIn("metadata", view["meta"]["stale_reason"])
        self.assertEqual(len(view["lots"]), 6)


class TestTaxEndToEndExclusions(unittest.TestCase):
    """Constructed-data-dir drives of the exclusion + broker paths the synth
    fixture cannot carry (its accounts are all taxable and its broker labels
    happen to equal their own slugs — the fixture-from-the-model trap)."""

    def _dir_with(self, *, extra_positions=(), extra_lots=(),
                  relabel_broker=None):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        pos = pd.read_csv(d / "positions.csv")
        if relabel_broker:
            pos["broker"] = pos["broker"].replace(relabel_broker)
        if extra_positions:
            pos = pd.concat([pos, pd.DataFrame(list(extra_positions))],
                            ignore_index=True)
        pos.to_csv(d / "positions.csv", index=False)
        if extra_lots:
            lots = pd.read_csv(d / "lots.csv")
            lots = pd.concat([lots, pd.DataFrame(list(extra_lots))],
                             ignore_index=True)
            lots.to_csv(d / "lots.csv", index=False)
        return d

    _IRA_POS = {"statement_date": "2026-04-30", "broker": "alpine",
                "account_id": "TEST-D", "account_type": "roth_ira",
                "symbol": "SPY", "asset_class": "equity_etf",
                "quantity": 5, "market_value": 2850.0}
    _IRA_LOT = {"account_id": "TEST-D", "instrument_key": "SPY",
                "key_source": "symbol", "symbol": "SPY",
                "open_date": "2024-02-02", "acquired_date": "2024-02-02",
                "origin": "buy", "quantity_open": 5,
                "quantity_remaining": 5, "basis_open": 2000.0,
                "basis_remaining": 2000.0, "source_row": 99,
                "basis_evidence": "reconstructed", "band": "ok"}

    def test_ira_lot_never_appears_and_is_counted(self):
        # end-to-end through build_tax_view — the locked decision's actual
        # enforcement branch, not just the taxable_of predicate
        d = self._dir_with(extra_positions=[self._IRA_POS],
                           extra_lots=[self._IRA_LOT])
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        self.assertNotIn("TEST-D", {r["account_id"] for r in view["lots"]})
        self.assertEqual(view["summary"]["excluded"]["ira_accounts"], 1)
        self.assertEqual(len(view["lots"]), 6)   # the taxable book unchanged

    def test_exclusion_counted_before_broker_narrowing(self):
        # spec Update: exclusions are tax-ineligibility facts, not filter
        # state — an IRA in a broker OUTSIDE the selection still counts
        d = self._dir_with(extra_positions=[self._IRA_POS],
                           extra_lots=[self._IRA_LOT])
        frames = hs.load_frames(d)
        opts, _ = hs._broker_options(hs._current_snap(frames))
        harbor_id = next(o["id"] for o in opts if o["label"] == "harbor")
        view = txs.build_tax_view(frames, d, asof=ASOF, broker=[harbor_id])
        self.assertEqual({r["account_id"] for r in view["lots"]}, {"TEST-B"})
        self.assertEqual(view["summary"]["excluded"]["ira_accounts"], 1)

    def test_broker_ids_resolve_to_labels_before_narrowing(self):
        # the route validates broker OPTION IDS (slugs); positions carries
        # raw labels. With a label that is NOT its own slug the two
        # namespaces diverge — narrowing must still work.
        d = self._dir_with(relabel_broker={"alpine": "Alpine Co."})
        frames = hs.load_frames(d)
        opts, _ = hs._broker_options(hs._current_snap(frames))
        fid_id = next(o["id"] for o in opts if o["label"] == "Alpine Co.")
        self.assertNotEqual(fid_id, "Alpine Co.")   # namespaces differ
        view = txs.build_tax_view(frames, d, asof=ASOF, broker=[fid_id])
        accts = {r["account_id"] for r in view["lots"]}
        self.assertEqual(accts, {"TEST-A", "TEST-C"})   # not silently empty

    def test_corrupt_meta_degrades_stale_never_raises(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        (d / "lots_meta.json").write_text("{not json", encoding="utf-8")
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        self.assertEqual(view["kind"], "tax")
        self.assertTrue(view["meta"]["stale"])
        self.assertIn("unreadable", view["meta"]["stale_reason"])
        self.assertEqual(len(view["lots"]), 6)
        json.dumps(view, allow_nan=False)

    def test_missing_transactions_is_an_error_view_not_a_crash(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        (d / "transactions.csv").unlink()
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        self.assertEqual(view["kind"], "error")
        self.assertIn("incomplete", view["reason"])


class TestTaxHarvest(unittest.TestCase):
    """S3b: the harvest view and its cross-account wash guard.

    The synth fixture cannot carry real wash shapes, so every blocking case
    is driven through a constructed data dir and through `build_tax_view`
    itself — not through `scan_harvest_candidates` in isolation. That is the
    S2b review lesson: the IRA branch there survived mutation-deletion with
    a green suite because nothing drove it end-to-end.

    The fixture's one candidate is TEST-B / BBB (300 sh, basis 15,300,
    marked 50.50 -> 15,150, a 150.00 loss). The backward wash window at the
    pinned ASOF (2026-06-28) is [2026-05-29, 2026-06-28].
    """

    IN_WINDOW = "2026-06-10"
    EDGE_IN = "2026-05-29"     # exactly 30 days before ASOF
    EDGE_OUT = "2026-05-28"    # 31 days — outside

    def _dir_with(self, *, txns=(), positions=()):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        if txns:
            tx = pd.read_csv(d / "transactions.csv")
            # appended, so existing positional indices — which lots.csv's
            # source_row values point at — keep pointing at the same rows
            tx = pd.concat([tx, pd.DataFrame(list(txns))], ignore_index=True)
            tx.to_csv(d / "transactions.csv", index=False)
        if positions:
            pos = pd.read_csv(d / "positions.csv")
            pos = pd.concat([pos, pd.DataFrame(list(positions))],
                            ignore_index=True)
            pos.to_csv(d / "positions.csv", index=False)
        return d

    @staticmethod
    def _buy(account, date_str, *, kind="buy", symbol="BBB", broker="harbor",
             qty=10):
        return {"settlement_date": date_str, "trade_date": date_str,
                "broker": broker, "account_id": account,
                "transaction_type": kind, "symbol": symbol, "cusip": "",
                "description": f"{symbol} purchase", "quantity": qty,
                "price": 50.0, "amount": -qty * 50.0, "source_file": "synth",
                "flow_scope": "", "pair_id": ""}

    @staticmethod
    def _ira_position(account="TEST-IRA"):
        return {"statement_date": "2026-04-30", "broker": "harbor",
                "account_id": account, "account_type": "roth_ira",
                "symbol": "BBB", "asset_class": "equity",
                "quantity": 10, "market_value": 505.0}

    def _harvest(self, d):
        return txs.build_tax_view(hs.load_frames(d), d,
                                  asof=ASOF)["harvest"]

    def _only(self, d):
        cands = self._harvest(d)["candidates"]
        self.assertEqual(len(cands), 1, cands)
        return cands[0]

    # ---- the clear baseline ----

    def test_the_fixtures_one_loss_lot_is_a_clear_candidate(self):
        c = self._only(self._dir_with())
        self.assertEqual((c["account_id"], c["symbol"]), ("TEST-B", "BBB"))
        self.assertEqual(c["wash_status"], "clear")
        self.assertEqual(c["blocking_buys"], [])
        self.assertFalse(c["is_ira_blocked"])
        self.assertAlmostEqual(c["unrealized_gl"], -150.0)

    def test_gain_and_flat_lots_never_become_candidates(self):
        h = self._harvest(self._dir_with())
        self.assertEqual([c["symbol"] for c in h["candidates"]], ["BBB"])
        self.assertGreater(h["summary"]["excluded_gain_or_flat"], 0)

    def test_an_unpriced_lot_is_counted_not_rendered(self):
        # TEST-C/CCC has no live mark: a candidate list that silently
        # dropped it would understate what the scan is blind to
        h = self._harvest(self._dir_with())
        self.assertNotIn("CCC", {c["symbol"] for c in h["candidates"]})
        self.assertEqual(h["summary"]["excluded_unpriced"], 1)

    def test_a_printed_evidence_lot_is_excluded_but_still_an_open_lot(self):
        d = self._dir_with()
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        self.assertIn("AAA", {r["symbol"] for r in view["lots"]})
        self.assertNotIn("AAA",
                         {c["symbol"] for c in view["harvest"]["candidates"]})
        self.assertEqual(
            view["harvest"]["summary"]["excluded_printed_evidence"], 1)

    # ---- blocking ----

    def test_a_same_account_buy_in_the_window_blocks(self):
        c = self._only(self._dir_with(
            txns=[self._buy("TEST-B", self.IN_WINDOW)]))
        self.assertEqual(c["wash_status"], "blocked")
        self.assertEqual([b["account_id"] for b in c["blocking_buys"]],
                         ["TEST-B"])
        self.assertFalse(c["is_ira_blocked"])
        # the block ages out 31 days after the blocking buy, not after today
        self.assertEqual(c["window_ends"], "2026-07-11")

    def test_a_buy_in_a_DIFFERENT_account_blocks(self):
        # the campaign's whole premise: no broker sees across institutions,
        # so a replacement bought elsewhere must still block
        c = self._only(self._dir_with(
            txns=[self._buy("TEST-A", self.IN_WINDOW, broker="alpine")]))
        self.assertEqual(c["wash_status"], "blocked")
        self.assertEqual([b["account_id"] for b in c["blocking_buys"]],
                         ["TEST-A"])

    def test_an_IRA_buy_blocks_and_is_flagged_permanent_end_to_end(self):
        # driven through build_tax_view, not scan_harvest_candidates: this
        # is the branch a consumer must not present as "wait out the window"
        d = self._dir_with(txns=[self._buy("TEST-IRA", self.IN_WINDOW)],
                           positions=[self._ira_position()])
        c = self._only(d)
        self.assertEqual(c["wash_status"], "blocked")
        self.assertTrue(c["is_ira_blocked"])
        self.assertEqual([b["is_ira"] for b in c["blocking_buys"]], [True])
        self.assertEqual(self._harvest(d)["summary"]["ira_blocked"], 1)

    def test_a_dividend_reinvestment_blocks(self):
        # a DRIP is an acquisition; a few fractional shares can disallow the
        # whole loss, which is exactly the trap this view exists to surface
        c = self._only(self._dir_with(
            txns=[self._buy("TEST-B", self.IN_WINDOW, kind="reinvestment",
                            qty=0.42)]))
        self.assertEqual(c["wash_status"], "blocked")
        self.assertEqual(c["blocking_buys"][0]["transaction_type"],
                         "reinvestment")
        self.assertAlmostEqual(c["blocking_buys"][0]["quantity"], 0.42)

    def test_a_transfer_in_does_not_block(self):
        # moving shares between your own accounts acquires nothing
        c = self._only(self._dir_with(
            txns=[self._buy("TEST-B", self.IN_WINDOW, kind="transfer_in")]))
        self.assertEqual(c["wash_status"], "clear")

    def test_a_buy_of_a_DIFFERENT_instrument_does_not_block(self):
        # same-instrument matching only; "substantially identical" is not
        # attempted, and the expander says so
        c = self._only(self._dir_with(
            txns=[self._buy("TEST-B", self.IN_WINDOW, symbol="SPY")]))
        self.assertEqual(c["wash_status"], "clear")

    def test_a_buy_AFTER_the_as_of_date_does_not_block(self):
        # The scan looks BACKWARD only, on purpose. Its question is "if I
        # sell today, is this loss already disallowed by something that has
        # happened" — the forward half of the wash window is unknowable at
        # that moment, which is exactly why `clear` is disclosed as a lower
        # bound rather than a verdict. A two-sided window here would let a
        # row dated ahead of the as-of (a future settle, a mis-dated row)
        # silently veto a harvest that nothing has actually blocked.
        c = self._only(self._dir_with(
            txns=[self._buy("TEST-B", "2026-07-03")]))     # 5 days ahead
        self.assertEqual(c["wash_status"], "clear")
        self.assertEqual(c["blocking_buys"], [])

    def test_the_window_edges_30_in_31_out(self):
        c_in = self._only(self._dir_with(
            txns=[self._buy("TEST-B", self.EDGE_IN)]))
        self.assertEqual(c_in["wash_status"], "blocked")
        c_out = self._only(self._dir_with(
            txns=[self._buy("TEST-B", self.EDGE_OUT)]))
        self.assertEqual(c_out["wash_status"], "clear")

    def test_the_lots_own_opening_buy_never_blocks_itself(self):
        # S3a's correctness fix, re-pinned at the view level: selling shares
        # you just bought, holding nothing else, is not a wash sale
        d = self._dir_with()
        tx = pd.read_csv(d / "transactions.csv")
        lots = pd.read_csv(d / "lots.csv")
        own = int(lots.loc[lots["symbol"] == "BBB", "source_row"].iloc[0])
        # drag the opening buy INTO the window; it is still not a blocker
        tx.loc[own, ["trade_date", "settlement_date"]] = self.IN_WINDOW
        tx.to_csv(d / "transactions.csv", index=False)
        self.assertEqual(self._only(d)["wash_status"], "clear")

    # ---- honesty + contract ----

    def test_semantics_state_how_much_of_the_window_is_observable(self):
        sem = self._harvest(self._dir_with())["semantics"]
        self.assertEqual(sem["window_days"], 30)
        self.assertEqual(sem["window_days_total"], 31)
        # the fixture's transactions end well before ASOF, so the guard can
        # see NONE of the backward window — `clear` above must not read as
        # "safe", and the strip has to be able to say so
        self.assertEqual(sem["tx_frontier"], "2026-04-15")
        self.assertEqual(sem["window_days_observed"], 0)
        self.assertEqual(sem["window_observed_pct"], 0.0)
        self.assertIn("OBSERVED", sem["clear_means"])

    def test_a_stale_ledger_says_a_clear_verdict_may_be_wrong(self):
        # lots.csv's source_row indexes transactions.csv POSITIONALLY and
        # combine_txns resets those positions on every rebuild
        d = self._dir_with(txns=[self._buy("TEST-B", self.EDGE_OUT)])
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        self.assertTrue(view["meta"]["stale"])
        self.assertIn("stale", view["harvest"]["semantics"]["stale_note"])

    def test_a_candidate_with_no_printed_acquisition_date_serializes(self):
        # A synthesized opening lot carries no acquired_date (the slice-1
        # SHORTFALL rule never guesses one). The scanner emits None, but
        # DataFrame construction turns that into NaN — and a NaN reaching
        # allow_nan=False is a 500 on the WHOLE tab, open lots included.
        # Neither the fixture nor today's real book happens to contain a
        # loss lot of this shape, so nothing else would have caught it.
        d = self._dir_with()
        lots = pd.read_csv(d / "lots.csv")
        row = lots[lots["symbol"] == "BBB"].iloc[0].to_dict()
        row.update({"acquired_date": None, "source_row": None,
                    "origin": "opening"})
        pd.concat([lots, pd.DataFrame([row])], ignore_index=True).to_csv(
            d / "lots.csv", index=False)
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        unknown = [c for c in view["harvest"]["candidates"]
                   if c["term"] == "unknown"]
        self.assertEqual(len(unknown), 1)
        self.assertIsNone(unknown[0]["acquired_date"])
        json.dumps(view, allow_nan=False)      # the actual regression

    def test_harvest_rows_carry_the_account_label(self):
        self.assertIn("account_label", self._only(self._dir_with()))

    def test_broker_narrowing_moves_lots_and_harvest_together(self):
        d = self._dir_with()
        frames = hs.load_frames(d)
        opts, _ = hs._broker_options(hs._current_snap(frames))
        fid = next(o["id"] for o in opts if o["label"] == "alpine")
        view = txs.build_tax_view(frames, d, asof=ASOF, broker=[fid])
        # BBB lives in the harbor account, so both views must drop it
        self.assertNotIn("TEST-B", {r["account_id"] for r in view["lots"]})
        self.assertEqual(view["harvest"]["candidates"], [])
        self.assertEqual(view["harvest"]["summary"]["candidates"], 0)
        # ...while the whole-book exclusion counts stay tax facts
        self.assertEqual(
            view["harvest"]["summary"]["excluded_printed_evidence"], 1)

    def test_a_malformed_transactions_file_degrades_with_a_reason(self):
        # open lots reads two columns; harvest reads the whole frame and
        # joins it, so it can fail on shapes the tab used to tolerate. It
        # must say so — never 500, and never a silent empty candidate list
        # that would read as "nothing to harvest".
        d = self._dir_with()
        tx = pd.read_csv(d / "transactions.csv").drop(
            columns=["transaction_type"])
        tx.to_csv(d / "transactions.csv", index=False)
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        self.assertEqual(view["kind"], "tax")
        self.assertEqual(len(view["lots"]), 6)          # open lots unharmed
        self.assertIn("unavailable", view["harvest"])
        self.assertEqual(view["harvest"]["candidates"], [])
        json.dumps(view, allow_nan=False)


class TestTaxPricingFallbacks(unittest.TestCase):
    """TK feedback 2026-07-30: resolved-key (empty-symbol) lots and the
    cusip-keyed treasury ladder rendered 'unpriced', so the header paired
    an all-lots basis with a priced-only market value and read as a large
    loss. Rung 1: the live lookup falls back to instrument_key — the
    scanner's own rule (two views of one lot must never price
    differently). Rung 2: the latest statement month's own marks
    (market_value / quantity per key), flagged price_source='statement'
    with their as-of month. Both rungs missing -> null, never 0."""

    LOT_RESOLVED = ("TEST-C,SPY,resolved,,2026-01-02,2026-01-02,buy,"
                    "10,10,5000.00,5000.00,90,reconstructed,ok\r\n")
    LOT_CUSIP = ("TEST-C,912TEST111,cusip,,2025-12-01,2025-12-01,buy,"
                 "20000,20000,19900.00,19900.00,91,reconstructed,ok\r\n")
    POS_BILL = ("2026-04-30,alpine,TEST-C,individual_tod,,912TEST111,"
                "Synthetic Bill,fixed_income,{q},0.99,{mv},9900.00,0.00,"
                "0.00,USD,synth\r\n")
    PRICE_LIVE = "912TEST111,2026-04-30,1.01,synth,ok\r\n"

    def _view_with(self, *, lots="", positions="", prices=""):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        for name, extra in (("lots.csv", lots),
                            ("positions.csv", positions),
                            ("prices_latest.csv", prices)):
            if extra:
                p = d / name
                p.write_bytes(p.read_bytes() + extra.encode("ascii"))
        frames = hs.load_frames(d)
        return txs.build_tax_view(frames, d, asof=ASOF)

    @staticmethod
    def _lot(view, ikey):
        return next(r for r in view["lots"]
                    if r["account_id"] == "TEST-C"
                    and r["instrument_key"] == ikey)

    def test_resolved_key_lot_prices_from_the_live_feed(self):
        # the real-book shape: a crosswalk-resolved lot has symbol=""
        # and its ticker in instrument_key; bare-symbol lookup left it
        # unpriced while SPY sat 'ok' in prices_latest
        view = self._view_with(lots=self.LOT_RESOLVED)
        lot = next(r for r in view["lots"]
                   if r["account_id"] == "TEST-C" and r["symbol"] == ""
                   and r["instrument_key"] == "SPY")
        self.assertEqual(lot["price"], 570.0)
        self.assertEqual(lot["price_source"], "live")
        self.assertIsNone(lot["price_asof"])
        self.assertEqual(lot["market_value"], 5700.0)
        self.assertEqual(lot["unrealized_gl"], 700.0)

    def test_statement_mark_prices_a_cusip_keyed_lot(self):
        # two positions rows for one instrument (some brokers print one
        # row per LOT) — the unit price must aggregate, not double
        pos = (self.POS_BILL.format(q=10000, mv="9900.00")
               + self.POS_BILL.format(q=10000, mv="9900.00"))
        view = self._view_with(lots=self.LOT_CUSIP, positions=pos)
        lot = self._lot(view, "912TEST111")
        self.assertEqual(lot["price_source"], "statement")
        self.assertEqual(lot["price_asof"], "2026-04")
        self.assertEqual(lot["price"], 0.99)  # 19,800 / 20,000
        self.assertEqual(lot["market_value"], 19_800.0)
        self.assertEqual(lot["unrealized_gl"], -100.0)
        self.assertEqual(view["summary"]["totals"]["stmt_priced_lots"], 1)

    def test_live_beats_statement_when_both_exist(self):
        pos = self.POS_BILL.format(q=20000, mv="19800.00")
        view = self._view_with(lots=self.LOT_CUSIP, positions=pos,
                               prices=self.PRICE_LIVE)
        lot = self._lot(view, "912TEST111")
        self.assertEqual(lot["price_source"], "live")
        self.assertEqual(lot["price"], 1.01)

    def test_totals_are_one_consistent_universe(self):
        # mv - priced_basis == unrealized_gl, and the basis split names
        # the unmarked remainder instead of silently deflating mv
        t = _view()["summary"]["totals"]
        self.assertEqual(t["priced_basis"] + t["unpriced_basis"],
                         t["basis"])
        self.assertAlmostEqual(t["market_value"] - t["priced_basis"],
                               t["unrealized_gl"], places=2)
        self.assertEqual(t["priced_basis"], 98_300.0)
        self.assertEqual(t["unpriced_basis"], 900.0)
        self.assertEqual(t["stmt_priced_lots"], 0)

    def test_both_rungs_missing_stays_null_with_no_source(self):
        ccc = next(r for r in _view()["lots"] if r["symbol"] == "CCC")
        self.assertIsNone(ccc["price"])
        self.assertIsNone(ccc["price_source"])
        self.assertIsNone(ccc["price_asof"])

    def test_silent_note_says_every_lot_is_shown(self):
        note = _view()["summary"]["silent_share_note"]
        self.assertIn("every lot is shown", note)
        self.assertIn("broker's own printed figures", note)

    def test_stmt_map_reads_the_latest_month_only(self):
        pos = pd.DataFrame([
            {"statement_date": "2026-03-31", "symbol": "ZZ", "cusip": "",
             "quantity": 10.0, "market_value": 50.0},
            {"statement_date": "2026-04-30", "symbol": "ZZ", "cusip": "",
             "quantity": 10.0, "market_value": 100.0},
        ])
        m, asof = txs.stmt_price_map(pos)
        self.assertEqual(asof, "2026-04")
        self.assertEqual(m["ZZ"], 10.0)

    def test_stmt_map_never_fabricates_a_price(self):
        pos = pd.DataFrame([
            {"statement_date": "2026-04-30", "symbol": "ZZ", "cusip": "",
             "quantity": 0.0, "market_value": 100.0},
            {"statement_date": "2026-04-30", "symbol": "", "cusip": "C1",
             "quantity": 10.0, "market_value": float("nan")},
        ])
        m, asof = txs.stmt_price_map(pos)
        self.assertEqual(m, {})
        self.assertEqual(asof, "2026-04")


class TestTaxRealizedYtd(unittest.TestCase):
    """Round 2 §3b: `summary.realized_ytd` aggregates lots_meta.json's
    build-time by_account block over accounts that are BOTH provably
    taxable (the lots loop's own fail-closed predicate — an IRA's or
    unknown-type account's realized never lands) AND inside the global
    broker narrowing. All values synthetic (#310)."""

    _IRA_POS = {"statement_date": "2026-04-30", "broker": "alpine",
                "account_id": "TEST-D", "account_type": "roth_ira",
                "symbol": "SPY", "asset_class": "equity_etf",
                "quantity": 5, "market_value": 2850.0}

    @staticmethod
    def _slot(gains, losses, closes=1, **extra):
        s = {"gains": gains, "losses": losses,
             "net": round(gains + losses, 2), "closes": closes}
        s.update(extra)
        return s

    def _dir(self, by_account=None, *, drop_key=False, year=2026,
             extra_positions=()):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        if extra_positions:
            pos = pd.read_csv(d / "positions.csv")
            pos = pd.concat([pos, pd.DataFrame(list(extra_positions))],
                            ignore_index=True)
            pos.to_csv(d / "positions.csv", index=False)
        meta = json.loads((d / "lots_meta.json").read_text(encoding="utf-8"))
        if drop_key:
            meta.pop("realized_ytd", None)
        else:
            meta["realized_ytd"] = {
                "year": year, "by_account": by_account or {},
                "notes": {"excludes_alpine_options": True,
                          "options_source": "harbor_printed_confirms",
                          "broker_unresolved": 0}}
        (d / "lots_meta.json").write_text(json.dumps(meta),
                                          encoding="utf-8")
        return d

    def _realized(self, d, **kw):
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF, **kw)
        return view["summary"]["realized_ytd"]

    def test_ira_and_unknown_type_accounts_never_land(self):
        # TEST-D is an IRA; TEST-X has no positions row so its type is
        # unknowable — both blocks must vanish fail-closed, exactly like
        # their lots do in the open-lots loop
        d = self._dir({"TEST-A": {"short": self._slot(100.0, -40.0, 2)},
                       "TEST-B": {"long": self._slot(500.0, 0.0)},
                       "TEST-D": {"short": self._slot(999.0, 0.0)},
                       "TEST-X": {"short": self._slot(777.0, 0.0)}},
                      extra_positions=[self._IRA_POS])
        rz = self._realized(d)
        self.assertEqual(rz["year"], 2026)
        self.assertEqual(rz["gains"], 600.0)
        self.assertEqual(rz["losses"], -40.0)
        self.assertEqual(rz["net"], 560.0)
        self.assertEqual(rz["by_term"]["short"],
                         {"gains": 100.0, "losses": -40.0, "net": 60.0})
        self.assertEqual(rz["by_term"]["long"],
                         {"gains": 500.0, "losses": 0.0, "net": 500.0})
        self.assertNotIn("unknown", rz["by_term"])   # never touched
        self.assertIsNone(rz["unavailable"])
        self.assertIs(rz["options_in"], True)
        self.assertEqual(rz["options_uncovered"], 0)

    def test_broker_narrowing_drops_an_accounts_realized(self):
        d = self._dir({"TEST-A": {"short": self._slot(100.0, -40.0, 2)},
                       "TEST-B": {"long": self._slot(500.0, 0.0)}})
        frames = hs.load_frames(d)
        opts, _ = hs._broker_options(hs._current_snap(frames))
        harbor_id = next(o["id"] for o in opts if o["label"] == "harbor")
        view = txs.build_tax_view(frames, d, asof=ASOF, broker=[harbor_id])
        rz = view["summary"]["realized_ytd"]
        self.assertEqual(rz["gains"], 500.0)
        self.assertEqual(rz["net"], 500.0)
        self.assertEqual(set(rz["by_term"]), {"long"})   # TEST-A's short gone

    def test_options_fold_into_short_long_and_never_unknown(self):
        d = self._dir({"TEST-B": {
            "short": self._slot(120.0, -20.0, 3),
            "long": self._slot(0.0, -75.5),
            "unknown": self._slot(10.0, 0.0),
            "options_short": self._slot(30.0, -5.0, 2, uncovered=1),
            "options_long": self._slot(0.0, -12.25, 1, uncovered=2),
        }})
        rz = self._realized(d)
        self.assertEqual(rz["by_term"]["short"],
                         {"gains": 150.0, "losses": -25.0, "net": 125.0})
        self.assertEqual(rz["by_term"]["long"],
                         {"gains": 0.0, "losses": -87.75, "net": -87.75})
        # options carry their own printed ST/LT tag: they can never land
        # in "unknown" (that bucket means the LEDGER could not date it)
        self.assertEqual(rz["by_term"]["unknown"],
                         {"gains": 10.0, "losses": 0.0, "net": 10.0})
        self.assertEqual(rz["gains"], 160.0)
        self.assertEqual(rz["losses"], -112.75)
        self.assertEqual(rz["net"], 47.25)
        self.assertEqual(rz["options_uncovered"], 3)

    def test_money_rerounds_at_the_boundary(self):
        # summing cent-rounded per-account figures reintroduces float dust
        d = self._dir({"TEST-A": {"short": self._slot(0.1, 0.0)},
                       "TEST-B": {"short": self._slot(0.2, 0.0)}})
        rz = self._realized(d)
        self.assertEqual(rz["by_term"]["short"]["gains"], 0.3)
        self.assertEqual(rz["gains"], 0.3)
        self.assertEqual(rz["net"], 0.3)

    def test_zero_money_bucket_with_closes_still_shows(self):
        # a zero-gain sale is activity, not absence: closes > 0 keeps the
        # term bucket even though every money field is 0.0
        d = self._dir({"TEST-A": {"short": self._slot(0.0, 0.0)}})
        rz = self._realized(d)
        self.assertEqual(rz["by_term"]["short"],
                         {"gains": 0.0, "losses": 0.0, "net": 0.0})

    def test_uncovered_only_options_never_invent_a_bucket(self):
        # option closes whose confirms printed no usable figure: no money
        # to show (never guessed), but the uncovered count must surface
        d = self._dir({"TEST-B": {"options_short": self._slot(
            0.0, 0.0, 0, uncovered=2)}})
        rz = self._realized(d)
        self.assertEqual(rz["by_term"], {})
        self.assertEqual(rz["options_uncovered"], 2)
        self.assertEqual(rz["gains"], 0.0)

    def test_fixture_realized_block_matches_the_enriched_truth(self):
        # Task 2: the fixture's lots_meta.json now carries a fully
        # synthetic, non-empty realized-YTD block (previously all-zero) so
        # the golden actually guards summary.realized_ytd.by_account. This
        # pins the topline cross-foot and the fold-then-emit account rows
        # against the real fixture, not just a constructed temp dir.
        # Round-4 review finding 2: TEST-C also carries an options_long
        # slot (folds into its long row) so the fixture jointly covers all
        # five slot names, short/long/unknown/options_short/options_long.
        rz = _view()["summary"]["realized_ytd"]
        self.assertEqual(rz, {
            "year": 2026,
            "gains": 2231.45, "losses": -566.35, "net": 1665.10,
            "by_term": {
                "short": {"gains": 925.25, "losses": -320.50,
                          "net": 604.75},
                "long": {"gains": 1245.80, "losses": -245.85,
                         "net": 999.95},
                "unknown": {"gains": 60.40, "losses": 0.0, "net": 60.40},
            },
            "by_account": [
                {"account_id": "TEST-A", "account_label": "TEST-A",
                 "term": "short", "closes": 4, "gains": 925.25,
                 "losses": -320.50, "net": 604.75},
                {"account_id": "TEST-B", "account_label": "TEST-B",
                 "term": "long", "closes": 2, "gains": 1200.00,
                 "losses": -150.75, "net": 1049.25},
                {"account_id": "TEST-B", "account_label": "TEST-B",
                 "term": "unknown", "closes": 1, "gains": 60.40,
                 "losses": 0.0, "net": 60.40},
                {"account_id": "TEST-C", "account_label": "TEST-C",
                 "term": "long", "closes": 2, "gains": 45.80,
                 "losses": -95.10, "net": -49.30},
            ],
            # TEST-X carries a slot but no positions row (unknowable type)
            # and never appears above -> fail-closed exclusion pinned here
            "options_in": True,
            "options_uncovered": 3,
            "unrecognized_slots": 0,
            "broker_unresolved": 0,
            "unavailable": None})

    def test_meta_without_the_key_degrades_named(self):
        # a pre-round-2 meta: name the fix, fabricate NOTHING else
        d = self._dir(drop_key=True)
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        rz = view["summary"]["realized_ytd"]
        self.assertEqual(set(rz), {"unavailable"})
        self.assertIn("build_lots.py --write", rz["unavailable"])
        self.assertIn("month close", rz["unavailable"])
        json.dumps(view, allow_nan=False)

    def test_missing_meta_degrades_the_same_way(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        (d / "lots_meta.json").unlink()
        rz = self._realized(d)
        self.assertEqual(set(rz), {"unavailable"})
        self.assertIn("build_lots.py --write", rz["unavailable"])

    def test_malformed_block_degrades_named_not_500(self):
        # by_account as a LIST: the realized tiles go down WITH a reason;
        # Open lots and Harvest render on (the _harvest_or_unavailable
        # contract applied to this block — review finding 4)
        d = self._dir({"TEST-A": {"short": self._slot(10.0, 0.0)}})
        meta_path = d / "lots_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["realized_ytd"]["by_account"] = ["not", "a", "mapping"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        rz = view["summary"]["realized_ytd"]
        self.assertEqual(set(rz), {"unavailable"})
        self.assertIn("could not be read", rz["unavailable"])
        self.assertIn("build_lots.py --write", rz["unavailable"])
        self.assertTrue(view["lots"])            # open lots survived
        json.dumps(view, allow_nan=False)

    def test_unrecognized_slots_and_broker_unresolved_are_disclosed(self):
        # a newer meta read by older fold rules must COUNT what it skips
        # (the swallowed-error rule — review finding 3), and the build's
        # own dropped-row count rides through to the surface; neither may
        # leak money into the totals
        d = self._dir({"TEST-A": {"short": self._slot(10.0, 0.0)}})
        meta_path = d / "lots_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        acct = meta["realized_ytd"]["by_account"]["TEST-A"]
        acct["mystery_bucket"] = {"gains": 5.0, "losses": 0.0,
                                  "net": 5.0, "closes": 1}
        acct["long"] = "not-a-dict"
        meta["realized_ytd"]["notes"]["broker_unresolved"] = 3
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        rz = self._realized(d)
        self.assertEqual(rz["unrecognized_slots"], 2)
        self.assertEqual(rz["broker_unresolved"], 3)
        self.assertEqual(rz["gains"], 10.0)
        self.assertNotIn("long", rz["by_term"])

    def _rows(self, d, **kw):
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF, **kw)
        return view["summary"]["realized_ytd"]["by_account"]

    def test_by_account_rows_fold_emit_and_order(self):
        # TEST-A's short + options_short fold into ONE short row; order is
        # label (test ids fall back to themselves), then term rank, then id
        d = self._dir({
            "TEST-B": {"long": self._slot(500.0, 0.0)},
            "TEST-A": {"short": self._slot(100.0, -40.0, 2),
                       "options_short": self._slot(30.0, -5.0, 1,
                                                   uncovered=1),
                       "unknown": self._slot(12.5, 0.0)},
        })
        self.assertEqual(
            self._rows(d),
            [{"account_id": "TEST-A", "account_label": "TEST-A",
              "term": "short", "closes": 3, "gains": 130.0,
              "losses": -45.0, "net": 85.0},
             {"account_id": "TEST-A", "account_label": "TEST-A",
              "term": "unknown", "closes": 1, "gains": 12.5,
              "losses": 0.0, "net": 12.5},
             {"account_id": "TEST-B", "account_label": "TEST-B",
              "term": "long", "closes": 1, "gains": 500.0,
              "losses": 0.0, "net": 500.0}])

    def test_shared_label_orders_term_major_then_id(self):
        # Round-4 review finding 3: every fixture account's label equals
        # its own id, so the sort's third key (account_id) never actually
        # discriminates anything there. Force two accounts under ONE
        # display label so the id tie-break engages, and confirm the
        # order is term-major (both shorts before either long) — id only
        # breaks the tie WITHIN a term rank, never overrides it.
        d = self._dir({
            "TEST-B": {"short": self._slot(300.0, -30.0, 1),
                       "long": self._slot(400.0, -40.0, 1)},
            "TEST-A": {"short": self._slot(100.0, -10.0, 1),
                       "long": self._slot(200.0, -20.0, 1)},
        })
        with mock.patch.object(hs, "ACCOUNT_DISPLAY",
                               {"TEST-A": "Shared", "TEST-B": "Shared"},
                               create=True):
            rows = self._rows(d)
        self.assertEqual(
            [(r["account_id"], r["term"]) for r in rows],
            [("TEST-A", "short"), ("TEST-B", "short"),
             ("TEST-A", "long"), ("TEST-B", "long")])

    def test_by_account_excludes_what_the_fold_excludes(self):
        # IRA (TEST-D) and type-unknowable (TEST-X) never emit rows —
        # the same fail-closed predicate the by_term fold applies
        d = self._dir({"TEST-A": {"short": self._slot(100.0, 0.0)},
                       "TEST-D": {"short": self._slot(999.0, 0.0)},
                       "TEST-X": {"short": self._slot(777.0, 0.0)}},
                      extra_positions=[self._IRA_POS])
        self.assertEqual([r["account_id"] for r in self._rows(d)],
                         ["TEST-A"])

    def test_by_account_respects_broker_narrowing(self):
        d = self._dir({"TEST-A": {"short": self._slot(100.0, -40.0, 2)},
                       "TEST-B": {"long": self._slot(500.0, 0.0)}})
        frames = hs.load_frames(d)
        opts, _ = hs._broker_options(hs._current_snap(frames))
        harbor_id = next(o["id"] for o in opts if o["label"] == "harbor")
        view = txs.build_tax_view(frames, d, asof=ASOF, broker=[harbor_id])
        rows = view["summary"]["realized_ytd"]["by_account"]
        self.assertEqual([r["account_id"] for r in rows], ["TEST-B"])

    def test_by_account_empty_year_is_empty_list(self):
        self.assertEqual(self._rows(self._dir({})), [])

    def test_empty_year_is_true_zeros_not_the_degrade_path(self):
        # Round-4 review finding 1: pin the COMPLETE empty-year dict
        # against a CONSTRUCTED empty dir (never the shared fixture, so
        # this can't go stale the way enriching the fixture just did to
        # its predecessor). Per the module docstring: an empty by_account
        # aggregates to true zeros with by_term {} — a statement about
        # the YEAR, not the degrade path (unavailable stays None here; a
        # mutation returning None money / options_in False would pass
        # every other test but must fail this one).
        rz = self._realized(self._dir({}))
        self.assertEqual(rz, {
            "year": 2026,
            "gains": 0.0, "losses": 0.0, "net": 0.0,
            "by_term": {}, "by_account": [],
            "options_in": True,
            "options_uncovered": 0,
            "unrecognized_slots": 0,
            "broker_unresolved": 0,
            "unavailable": None})

    def test_by_account_crossfoots_to_by_term_exactly(self):
        # cent-clean slots in, cent-clean rows out: refolding the rows
        # reproduces by_term EXACTLY (assertEqual, not approx)
        d = self._dir({
            "TEST-A": {"short": self._slot(850.0, -320.5, 3),
                       "options_short": self._slot(75.25, 0.0, 1,
                                                   uncovered=1)},
            "TEST-B": {"long": self._slot(1200.0, -150.75, 2),
                       "unknown": self._slot(60.4, 0.0)},
            "TEST-C": {"long": self._slot(0.0, -95.1)},
        })
        rz = self._realized(d)
        refold = {}
        for r in rz["by_account"]:
            b = refold.setdefault(r["term"], {"gains": 0.0,
                                              "losses": 0.0, "net": 0.0})
            b["gains"] = round(b["gains"] + r["gains"], 2)
            b["losses"] = round(b["losses"] + r["losses"], 2)
            b["net"] = round(b["net"] + r["net"], 2)
        self.assertEqual(refold, rz["by_term"])

    def test_by_account_uses_account_display_labels(self):
        d = self._dir({"TEST-A": {"short": self._slot(100.0, 0.0)}})
        with mock.patch.object(hs, "ACCOUNT_DISPLAY",
                               {"TEST-A": "Alpha"}, create=True):
            rows = self._rows(d)
        self.assertEqual(rows[0]["account_label"], "Alpha")

    def test_unrecognized_slots_never_emit_rows(self):
        d = self._dir({"TEST-A": {"mystery": self._slot(50.0, 0.0),
                                  "short": "not-a-dict"}})
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        rz = view["summary"]["realized_ytd"]
        self.assertEqual(rz["by_account"], [])
        self.assertEqual(rz["unrecognized_slots"], 2)

    def test_degrade_path_has_no_by_account_key(self):
        # malformed block -> named unavailable, no by_account key, and
        # Open lots + Harvest survive in the same payload
        d = self._dir({})
        meta = json.loads((d / "lots_meta.json")
                          .read_text(encoding="utf-8"))
        meta["realized_ytd"] = ["not", "a", "dict"]
        (d / "lots_meta.json").write_text(json.dumps(meta),
                                          encoding="utf-8")
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        rz = view["summary"]["realized_ytd"]
        self.assertTrue(rz["unavailable"])
        self.assertNotIn("by_account", rz)
        self.assertTrue(view["lots"])
        self.assertIn("harvest", view)


class TestTaxInstrumentType(unittest.TestCase):
    """Round 2 §3b: every lots[] and harvest.candidates[] row carries
    `type` ("stock" | "etf" | "other") derived from positions' own
    asset_class — latest-wins PER KEY across the whole positions history,
    symbol and cusip both keyed, row lookup via the pricing rule's
    sym-or-ikey fallback. No ticker lists anywhere (the Sage rule)."""

    # constructed rows: 14 lots.csv columns / 16 positions.csv columns,
    # appended as bytes the way TestTaxPricingFallbacks does
    LOT_RESOLVED = ("TEST-C,SPY,resolved,,2026-01-02,2026-01-02,buy,"
                    "10,10,5000.00,5000.00,90,reconstructed,ok\r\n")
    LOT_MYSTERY = ("TEST-C,MYST,symbol,MYST,2026-02-01,2026-02-01,buy,"
                   "5,5,500.00,500.00,91,reconstructed,ok\r\n")
    LOT_CUSIP = ("TEST-C,912TEST111,cusip,,2025-12-01,2025-12-01,buy,"
                 "100,100,95.00,95.00,92,reconstructed,ok\r\n")
    POS_CUSIP = ("2026-04-30,alpine,TEST-C,individual_tod,,912TEST111,"
                 "Synthetic Note,equity_stock,100,1.00,100.00,95.00,5.00,"
                 "0.00,USD,synth\r\n")

    def _view_with(self, *, lots="", positions=""):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        for name, extra in (("lots.csv", lots),
                            ("positions.csv", positions)):
            if extra:
                p = d / name
                p.write_bytes(p.read_bytes() + extra.encode("ascii"))
        return txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)

    # ---- _type_map unit ----

    def test_latest_wins_per_key_across_history_not_latest_month(self):
        # OLD's last sighting is an old month: a stranded instrument must
        # classify from ITS OWN max statement_date, not vanish behind the
        # newest month's row set
        pos = pd.DataFrame([
            {"statement_date": "2026-04-30", "symbol": "SPY", "cusip": "",
             "asset_class": "equity_etf"},
            {"statement_date": "2026-01-31", "symbol": "OLD", "cusip": "",
             "asset_class": "equity_stock"},
        ])
        m = txs._type_map(pos)
        self.assertEqual(m["OLD"], "stock")
        self.assertEqual(m["SPY"], "etf")

    def test_reclassed_ticker_classifies_from_its_newest_row(self):
        # the key's newest row wins even when the frame is not date-sorted
        # (raw equity_etf here; the broker-misfile corrections are covered
        # by the reclass tests below)
        pos = pd.DataFrame([
            {"statement_date": "2026-04-30", "symbol": "RRR", "cusip": "",
             "asset_class": "equity_etf"},
            {"statement_date": "2026-01-31", "symbol": "RRR", "cusip": "",
             "asset_class": "equity_stock"},
        ])
        self.assertEqual(txs._type_map(pos)["RRR"], "etf")
        self.assertEqual(txs._type_map(pos.iloc[::-1])["RRR"], "etf")

    # ---- the reclass layer (2026-07-31 live-smoke fix) ----
    # positions.csv carries the broker's RAW tag; Alpine files ETFs under
    # Common Stock (PR #131), so on the real book raw-class typing left the
    # ETFs view EMPTY. Type runs through reclass_asset + the config ETF
    # sets; these pin the true path (raw tag WRONG, correction supplied).

    @staticmethod
    def _one_row(sym: str, cls: str) -> pd.DataFrame:
        return pd.DataFrame([{"statement_date": "2026-04-30", "symbol": sym,
                              "cusip": "", "asset_class": cls}])

    def test_raw_broker_misfile_reclassifies_via_etf_class(self):
        pos = self._one_row("QQZ", "equity_stock")
        self.assertEqual(txs._type_map(pos)["QQZ"], "stock")
        self.assertEqual(
            txs._type_map(pos, etf_class={"QQZ": "equity_etf"})["QQZ"],
            "etf")

    def test_bond_etf_mapped_fixed_income_still_types_etf(self):
        # an etf_class entry mapping a ticker to fixed_income (risk
        # bucketing) is still the user saying "this is an ETF"; a plain
        # fixed_income instrument with no entry stays other
        m = txs._type_map(
            pd.concat([self._one_row("SGVX", "equity_stock"),
                       self._one_row("BND", "fixed_income")]),
            etf_class={"SGVX": "fixed_income"})
        self.assertEqual(m["SGVX"], "etf")
        self.assertEqual(m["BND"], "other")

    def test_commodity_builtin_types_etf(self):
        # GLD-family tickers reclass to the gold bucket wherever the
        # broker filed them — no config needed (asset_reclass builtin)
        pos = self._one_row("GLD", "fixed_income")
        self.assertEqual(txs._type_map(pos)["GLD"], "etf")

    def test_core_etf_symbols_membership_types_etf(self):
        pos = self._one_row("CLSX", "equity_stock")
        self.assertEqual(txs._type_map(pos)["CLSX"], "stock")
        self.assertEqual(
            txs._type_map(pos, core_etf_symbols={"CLSX"})["CLSX"], "etf")

    def test_etf_class_mapping_back_to_stock_does_not_force_etf(self):
        pos = self._one_row("REAL", "equity_stock")
        self.assertEqual(
            txs._type_map(pos, etf_class={"REAL": "equity_stock"})["REAL"],
            "stock")

    # ---- is_tlh (round 3, filter-only) ----

    def test_tlh_account_rows_flag_is_tlh_and_type_stays_truthful(self):
        # the Type FILTER carves the TLH sleeve out of Individual stocks;
        # the type field itself stays instrument truth (a TLH lot is
        # still typed by its instrument), and the payload carries only a
        # BOOLEAN — never the account id itself
        with mock.patch.object(txs, "_tlh_account_id",
                               return_value="TEST-A"):
            view = _view()
        for r in view["lots"]:
            self.assertEqual(r["is_tlh"], r["account_id"] == "TEST-A")
        spy = next(r for r in view["lots"]
                   if r["account_id"] == "TEST-A" and r["symbol"] == "SPY")
        self.assertEqual(spy["type"], "etf")     # truth untouched
        self.assertTrue(spy["is_tlh"])

    def test_harvest_candidates_flag_is_tlh_identically(self):
        cand_acct = _view()["harvest"]["candidates"][0]["account_id"]
        with mock.patch.object(txs, "_tlh_account_id",
                               return_value=cand_acct):
            view = _view()
        self.assertTrue(all(c["is_tlh"] == (c["account_id"] == cand_acct)
                            for c in view["harvest"]["candidates"]))
        self.assertTrue(view["harvest"]["candidates"][0]["is_tlh"])

    def test_unset_tlh_config_flags_nothing(self):
        with mock.patch.object(txs, "_tlh_account_id", return_value=""):
            view = _view()
        self.assertFalse(any(r["is_tlh"] for r in view["lots"]))
        self.assertFalse(any(c["is_tlh"]
                             for c in view["harvest"]["candidates"]))

    def test_display_option_leg_is_not_a_key_at_all(self):
        # display-format option legs are dropped from the map entirely
        # (they are never type contenders); consumers' .get(key, "other")
        # types such a lot "other" by default
        pos = self._one_row("SPY DEC 26 PUT 650.00", "other")
        self.assertNotIn("SPY DEC 26 PUT 650.00", txs._type_map(pos))

    def test_option_leg_is_never_a_type_contender_for_its_underlying(self):
        # a covered call's positions row carries the UNDERLYING's ticker
        # as its symbol; at a same-date tie the winner must not be
        # positions.csv row order (branch review: 3 real keys ambiguous
        # this way, masked only by config membership)
        rows = pd.DataFrame([
            {"statement_date": "2026-04-30", "symbol": "QQZ", "cusip": "",
             "asset_class": "equity_etf"},
            {"statement_date": "2026-04-30", "symbol": "QQZ", "cusip": "",
             "asset_class": "option_call"},
        ])
        self.assertEqual(txs._type_map(rows)["QQZ"], "etf")
        self.assertEqual(txs._type_map(rows.iloc[::-1])["QQZ"], "etf")

    def test_same_date_class_tie_resolves_by_preference_not_row_order(self):
        rows = pd.DataFrame([
            {"statement_date": "2026-04-30", "symbol": "QQZ", "cusip": "",
             "asset_class": "equity_etf"},
            {"statement_date": "2026-04-30", "symbol": "QQZ", "cusip": "",
             "asset_class": "equity_stock"},
        ])
        self.assertEqual(txs._type_map(rows)["QQZ"], "etf")
        self.assertEqual(txs._type_map(rows.iloc[::-1])["QQZ"], "etf")
        # a NEWER date still beats a preferred class at an older one —
        # preference only breaks same-date ties
        newer = pd.concat([rows, pd.DataFrame([
            {"statement_date": "2026-05-31", "symbol": "QQZ", "cusip": "",
             "asset_class": "equity_stock"}])], ignore_index=True)
        self.assertEqual(txs._type_map(newer)["QQZ"], "stock")

    def test_cusip_keys_and_everything_else_is_other(self):
        pos = pd.DataFrame([
            {"statement_date": "2026-04-30", "symbol": "", "cusip": "C-1",
             "asset_class": "equity_stock"},
            {"statement_date": "2026-04-30", "symbol": "BND",
             "cusip": float("nan"), "asset_class": "fixed_income"},
            {"statement_date": "2026-04-30", "symbol": "NANC", "cusip": "",
             "asset_class": float("nan")},
        ])
        m = txs._type_map(pos)
        self.assertEqual(m["C-1"], "stock")      # cusip-keyed rows key too
        self.assertEqual(m["BND"], "other")      # fixed_income -> other
        self.assertEqual(m["NANC"], "other")     # NaN class -> other
        self.assertNotIn("", m)                  # no empty/NaN keys
        self.assertNotIn("nan", m)

    # ---- end-to-end row stamping ----

    def test_every_lot_row_is_stamped_from_positions(self):
        view = _view()
        types = {(r["account_id"], r["symbol"]): r["type"]
                 for r in view["lots"]}
        self.assertEqual(types[("TEST-A", "SPY")], "etf")
        self.assertEqual(types[("TEST-A", "AAA")], "stock")
        self.assertEqual(types[("TEST-B", "BBB")], "other")  # fixed_income
        self.assertEqual(types[("TEST-C", "CCC")], "stock")
        self.assertTrue(all(r["type"] in ("stock", "etf", "other")
                            for r in view["lots"]))

    def test_resolved_key_lot_classifies_through_instrument_key(self):
        # crosswalk shape: empty symbol, ticker in the KEY — the same
        # sym-or-ikey rule pricing uses; two views of one lot must never
        # disagree, about price or about type
        view = self._view_with(lots=self.LOT_RESOLVED)
        lot = next(r for r in view["lots"] if r["symbol"] == ""
                   and r["instrument_key"] == "SPY")
        self.assertEqual(lot["type"], "etf")

    def test_cusip_keyed_lot_classifies_through_the_cusip(self):
        view = self._view_with(lots=self.LOT_CUSIP,
                               positions=self.POS_CUSIP)
        lot = next(r for r in view["lots"]
                   if r["instrument_key"] == "912TEST111")
        self.assertEqual(lot["type"], "stock")

    def test_unknown_key_lands_other(self):
        # a key positions has never carried classifies "other", never a
        # guess and never a KeyError
        view = self._view_with(lots=self.LOT_MYSTERY)
        lot = next(r for r in view["lots"] if r["symbol"] == "MYST")
        self.assertEqual(lot["type"], "other")

    def test_harvest_candidate_stamped_identically(self):
        view = _view()
        cand = next(c for c in view["harvest"]["candidates"]
                    if c["symbol"] == "BBB")
        lot = next(r for r in view["lots"] if r["symbol"] == "BBB")
        self.assertIn(cand["type"], ("stock", "etf", "other"))
        self.assertEqual(cand["type"], lot["type"])   # one map, two views


class TestTaxLotIds(unittest.TestCase):
    """Estimator prereq: a stable per-lot id — the lots.csv positional
    row index — so sim selection survives client sorting/filtering and
    broker narrowing. NOT display order: rows are sorted after build."""

    def _view(self, **kw):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        return txs.build_tax_view(hs.load_frames(d), d, asof=ASOF, **kw), d

    def test_every_lot_carries_its_csv_row_index(self):
        view, d = self._view()
        lots = pd.read_csv(d / "lots.csv")
        self.assertTrue(view["lots"])
        for row in view["lots"]:
            i = row["lot_id"]
            self.assertIsInstance(i, int)
            src = lots.iloc[i]
            self.assertEqual(str(src["instrument_key"]),
                             row["instrument_key"])
            self.assertEqual(str(src["account_id"]), row["account_id"])

    def test_lot_ids_unique(self):
        view, _ = self._view()
        ids = [r["lot_id"] for r in view["lots"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_lot_id_stable_under_broker_narrowing(self):
        whole, _ = self._view()
        keys = [(r["account_id"], r["instrument_key"],
                 r["acquired_date"], r["quantity_remaining"])
                for r in whole["lots"]]
        # a silent last-wins collision here would misjoin the narrowed
        # compare below — fail loudly if the fixture ever grows twins
        self.assertEqual(len(keys), len(set(keys)),
                         "fixture grew colliding lots — rekey this test")
        by_key = {(r["account_id"], r["instrument_key"],
                   r["acquired_date"], r["quantity_remaining"]): r["lot_id"]
                  for r in whole["lots"]}
        view, d = self._view()
        frames = hs.load_frames(d)
        opts, _ = hs._broker_options(hs._current_snap(frames))
        one = [o["id"] for o in opts if o["id"] != "all"][:1]
        narrowed = txs.build_tax_view(frames, d, asof=ASOF, broker=one)
        for r in narrowed["lots"]:
            k = (r["account_id"], r["instrument_key"],
                 r["acquired_date"], r["quantity_remaining"])
            self.assertEqual(r["lot_id"], by_key[k])


class TestTaxServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_tax_ok(self):
        r = self.client.get("/api/tax")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["kind"], "tax")
        self.assertIn("brokers", body["meta"])
        self.assertIn("history_starts", body["meta"])

    def test_unknown_broker_422(self):
        r = self.client.get("/api/tax", params={"broker": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_broker_filter_narrows_accounts(self):
        r = self.client.get("/api/tax", params={"broker": "harbor"})
        self.assertEqual(r.status_code, 200)
        accts = {row["account_id"] for row in r.json()["lots"]}
        self.assertEqual(accts, {"TEST-B"})

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/tax")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)

    def test_payload_carries_realized_ytd_and_row_type(self):
        # round 2 §3b: the new keys must survive the route's own
        # serialization, not just build_tax_view
        r = self.client.get("/api/tax")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        rz = body["summary"]["realized_ytd"]
        self.assertEqual(rz["year"], 2026)
        self.assertIsNone(rz["unavailable"])
        # Task 2: fixture's realized block is enriched, not empty. Round-4
        # finding 2 added TEST-C's options_long slot, which folds into
        # "long" alongside TEST-B's and TEST-C's plain long slots.
        self.assertEqual(rz["by_term"], {
            "short": {"gains": 925.25, "losses": -320.50, "net": 604.75},
            "long": {"gains": 1245.80, "losses": -245.85, "net": 999.95},
            "unknown": {"gains": 60.40, "losses": 0.0, "net": 60.40},
        })
        self.assertTrue(body["lots"])
        self.assertIn(body["lots"][0]["type"], ("stock", "etf", "other"))

    def test_pre_round2_meta_degrades_realized_over_the_route(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        meta = json.loads((d / "lots_meta.json").read_text(encoding="utf-8"))
        meta.pop("realized_ytd", None)
        (d / "lots_meta.json").write_text(json.dumps(meta),
                                          encoding="utf-8")
        os.environ["APP_DATA_DIR"] = str(d)
        try:
            r = self.client.get("/api/tax")
            self.assertEqual(r.status_code, 200)
            rz = r.json()["summary"]["realized_ytd"]
            self.assertEqual(set(rz), {"unavailable"})
            self.assertIn("build_lots.py --write", rz["unavailable"])
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestTaxGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_tax_golden.json")

    def test_matches_golden(self):
        view = _view()
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        # house convention: float-tolerant, structure/strings exact
        self.assertIsNone(_deep_close(view, expected))


FULL_PROFILE = {"filing_status": "single", "w2_income": 100_000.0,
                "state": "CA", "deduction": "standard",
                "carryforward_loss": 0.0, "qualified_dividend_pct": 1.0,
                "unknown_term_assumption": "long"}


class TestTaxEstimateAssembly(unittest.TestCase):
    """build_tax_estimate: profile merge + degrades, realized wiring,
    income window + treasury split, all-or-nothing sim validation,
    wash flag. All values synthetic (#310)."""

    def _dir(self, by_account=None, *, extra_tx=(), extra_lots=(),
             extra_positions=(), drop_realized=False):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = _fixture_copy_dir(tmp)
        meta = json.loads((d / "lots_meta.json").read_text(encoding="utf-8"))
        if drop_realized:
            meta.pop("realized_ytd", None)
        else:
            meta["realized_ytd"] = {
                "year": 2026, "by_account": by_account or {},
                "notes": {"excludes_alpine_options": True,
                          "options_source": "harbor_printed_confirms",
                          "broker_unresolved": 0}}
        (d / "lots_meta.json").write_text(json.dumps(meta),
                                          encoding="utf-8")
        if extra_tx:
            tx = pd.read_csv(d / "transactions.csv")
            template = {c: "" for c in tx.columns}
            rows = [{**template, **r} for r in extra_tx]
            tx = pd.concat([tx, pd.DataFrame(rows)], ignore_index=True)
            tx.to_csv(d / "transactions.csv", index=False)
        if extra_lots:
            lots = pd.read_csv(d / "lots.csv")
            lots = pd.concat([lots, pd.DataFrame(list(extra_lots))],
                             ignore_index=True)
            lots.to_csv(d / "lots.csv", index=False)
        if extra_positions:
            pos = pd.read_csv(d / "positions.csv")
            pos = pd.concat([pos, pd.DataFrame(list(extra_positions))],
                            ignore_index=True)
            pos.to_csv(d / "positions.csv", index=False)
        return d

    def _est(self, d, **kw):
        kw.setdefault("overrides", dict(FULL_PROFILE))
        with mock.patch.object(txs, "_config_profile", return_value={}):
            return txs.build_tax_estimate(hs.load_frames(d), d,
                                          asof=ASOF, **kw)

    @staticmethod
    def _slot(gains, losses, closes=1):
        return {"gains": gains, "losses": losses,
                "net": round(gains + losses, 2), "closes": closes}

    def test_unconfigured_profile_names_the_degrade(self):
        d = self._dir()
        with mock.patch.object(txs, "_config_profile", return_value={}):
            out = txs.build_tax_estimate(hs.load_frames(d), d, asof=ASOF)
        self.assertEqual(out["kind"], "error")
        self.assertIn("TAX_PROFILE", out["reason"])
        self.assertIn("config_example", out["reason"])

    def test_missing_realized_block_degrades_named(self):
        d = self._dir(drop_realized=True)
        out = self._est(d)
        self.assertEqual(out["kind"], "error")
        self.assertIn("realized_ytd", out["reason"])

    def test_engine_valueerror_becomes_named_degrade(self):
        d = self._dir()
        out = self._est(d, overrides={**FULL_PROFILE, "state": "NY"})
        self.assertEqual(out["kind"], "error")
        self.assertIn("state", out["reason"])

    def test_config_profile_typeerror_becomes_named_degrade(self):
        # A hand-edited config_local TAX_PROFILE with `"w2_income": None`
        # passes the missing-fields check (str(None) == "None" is
        # truthy-non-blank) but blows up `float(None)` inside the engine
        # with a TypeError, not a ValueError — the profile guard must
        # catch both, not just the one the engine happens to raise today.
        d = self._dir()
        full = {**FULL_PROFILE, "w2_income": None}
        with mock.patch.object(txs, "_config_profile", return_value=full):
            out = txs.build_tax_estimate(hs.load_frames(d), d, asof=ASOF)
        self.assertEqual(out["kind"], "error")
        self.assertIn("NoneType", out["reason"])

    def test_baseline_matches_direct_engine_recompute(self):
        d = self._dir({"TEST-A": {"short": self._slot(1_000.0, -400.0)},
                       "TEST-B": {"long": self._slot(5_000.0, 0.0)}})
        out = self._est(d)
        self.assertEqual(out["kind"], "estimate")
        expected = tax_estimate.estimate_year_tax(
            dict(FULL_PROFILE),
            {"short": 600.0, "long": 5_000.0, "unknown": 0.0},
            out["income"])
        self.assertIsNone(_deep_close(out["baseline"], expected))
        self.assertIsNone(out["with_sim"])
        self.assertEqual(out["sim_rejected"], [])
        self.assertEqual(out["year"], tax_estimate.TAX_YEAR)

    def test_income_window_year_taxable_and_treasury_split(self):
        # DELTA-based: the fixture may carry its own 2026 income rows,
        # so assert what the injected rows ADD, not absolute sums.
        base = self._est(self._dir())["income"]
        d = self._dir(extra_tx=[
            {"account_id": "TEST-A", "transaction_type": "dividend",
             "amount": 1_200.0, "settlement_date": "2026-03-10",
             "symbol": "SPY", "description": "DIVIDEND"},
            {"account_id": "TEST-A", "transaction_type": "interest",
             "amount": 800.0, "settlement_date": "2026-05-04",
             "symbol": "912797GL5",
             "description": "UNITED STATES TREASURY BILL DUE 09/15/2026"},
            {"account_id": "TEST-B", "transaction_type": "interest",
             "amount": 300.0, "settlement_date": "2026-06-15",
             "symbol": "", "description": "CREDIT INTEREST"},
            {"account_id": "TEST-A", "transaction_type": "dividend",
             "amount": 999.0, "settlement_date": "2025-11-20",
             "symbol": "SPY", "description": "PRIOR YEAR"},
            {"account_id": "TEST-A", "transaction_type": "withholding",
             "amount": -45.0, "settlement_date": "2026-04-02",
             "symbol": "", "description": "FOREIGN TAX"}])
        out = self._est(d)
        inc = out["income"]
        self.assertAlmostEqual(inc["dividends"] - base["dividends"],
                               1_200.0, places=2)   # 2025 row excluded
        self.assertAlmostEqual(inc["interest"] - base["interest"],
                               1_100.0, places=2)
        self.assertAlmostEqual(
            inc["treasury_interest"] - base["treasury_interest"],
            800.0, places=2)
        self.assertAlmostEqual(inc["withholding"] - base["withholding"],
                               -45.0, places=2)
        self.assertTrue(inc["through"] >= "2026-06-15")

    def test_distribution_rows_never_feed_the_estimate_dividends(self):
        # DA-F-1: a `principal_pmt` distribution on held shares counts as
        # yield on the Income tab (Distributions S2 rule) but is return
        # of capital to the IRS — taxed once, via the basis reduction the
        # lot engine already applies. income_timeseries folds qualifying
        # distribution rows into `dividends`, so feeding it the raw frame
        # taxed the same dollars twice in-model (fed pref + CA + NIIT on
        # top of the future capital gain from the lowered basis).
        base = self._est(self._dir())["income"]
        d = self._dir(extra_tx=[
            {"account_id": "TEST-A", "transaction_type": "principal_pmt",
             "amount": 27_500.0, "settlement_date": "2026-04-27",
             "symbol": "ROCF", "description": "RT 10.000 PRINCIPAL"},
            {"account_id": "TEST-A", "transaction_type": "dividend",
             "amount": 500.0, "settlement_date": "2026-04-28",
             "symbol": "SPY", "description": "DIVIDEND"}])
        inc = self._est(d)["income"]
        self.assertAlmostEqual(inc["dividends"] - base["dividends"],
                               500.0, places=2)   # distribution excluded
        self.assertAlmostEqual(inc["interest"] - base["interest"],
                               0.0, places=2)

    def test_blank_treasury_amount_does_not_poison_the_estimate(self):
        # NaN is truthy in Python, so `pd.to_numeric(...) or 0.0` never
        # falls back for a blank/garbled amount — it lets NaN into
        # treasury_interest, which then poisons every downstream figure
        # and 500s the whole estimate at the json.dumps(allow_nan=False)
        # boundary. A blank amount must be skipped, not summed as NaN.
        base = self._est(self._dir())["income"]["treasury_interest"]
        d = self._dir(extra_tx=[
            {"account_id": "TEST-A", "transaction_type": "interest",
             "amount": "", "settlement_date": "2026-05-10",
             "symbol": "912797GL5",
             "description": "UNITED STATES TREASURY BILL DUE 09/15/2026"},
            {"account_id": "TEST-A", "transaction_type": "interest",
             "amount": 500.0, "settlement_date": "2026-05-11",
             "symbol": "912797GL5",
             "description": "UNITED STATES TREASURY BILL DUE 09/15/2026"}])
        out = self._est(d)
        self.assertEqual(out["kind"], "estimate")
        self.assertAlmostEqual(
            out["income"]["treasury_interest"] - base, 500.0, places=2)

    def test_unknown_type_account_income_never_lands(self):
        # An account with no positions row has UNKNOWN type -> fail-closed
        # exclusion, same predicate as the open-lots loop.
        base = self._est(self._dir())["income"]
        d = self._dir(extra_tx=[
            {"account_id": "TEST-NOPOS", "transaction_type": "dividend",
             "amount": 5_000.0, "settlement_date": "2026-03-10",
             "symbol": "SPY", "description": "DIVIDEND"}])
        out = self._est(d)
        self.assertAlmostEqual(out["income"]["dividends"],
                               base["dividends"], places=2)

    def _first_priced_lot(self, d, prefer_loss=False):
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        cands = [r for r in view["lots"]
                 if r["market_value"] is not None
                 and r["term"] in ("short", "long")
                 and r["quantity_remaining"] > 1]
        if prefer_loss:
            losses = [r for r in cands
                      if (r["unrealized_gl"] or 0) < 0]
            if losses:
                return losses[0]
        return cands[0]

    def test_valid_sim_leg_prorates_basis_and_recomputes(self):
        d = self._dir()
        lot = self._first_priced_lot(d)
        qty = round(lot["quantity_remaining"] / 2.0, 4)
        out = self._est(d, sim=[{"lot_id": lot["lot_id"], "qty": qty}])
        self.assertEqual(out["sim_rejected"], [])
        leg = out["sim_legs"][0]
        frac = qty / lot["quantity_remaining"]
        self.assertAlmostEqual(leg["proceeds"], qty * lot["price"], 2)
        self.assertAlmostEqual(leg["basis_part"],
                               lot["basis_remaining"] * frac, 2)
        self.assertAlmostEqual(leg["gl"],
                               leg["proceeds"] - leg["basis_part"], 2)
        self.assertEqual(leg["term"], lot["term"])
        expected = tax_estimate.estimate_year_tax(
            dict(FULL_PROFILE), {"short": 0.0, "long": 0.0,
                                 "unknown": 0.0},
            out["income"],
            [{"gl": leg["gl"], "term": leg["term"]}])
        self.assertIsNone(_deep_close(out["with_sim"], expected))

    def test_sim_rejection_is_all_or_nothing(self):
        d = self._dir()
        lot = self._first_priced_lot(d)
        out = self._est(d, sim=[
            {"lot_id": lot["lot_id"], "qty": 1.0},
            {"lot_id": 999_999, "qty": 1.0}])
        self.assertIsNone(out["with_sim"])
        self.assertEqual(len(out["sim_rejected"]), 1)
        self.assertIn("999999", out["sim_rejected"][0])
        self.assertEqual(out["baseline"]["year"], tax_estimate.TAX_YEAR)

    def test_qty_exceeding_remaining_rejects(self):
        d = self._dir()
        lot = self._first_priced_lot(d)
        out = self._est(d, sim=[{"lot_id": lot["lot_id"],
                                 "qty": lot["quantity_remaining"] + 1}])
        self.assertIsNone(out["with_sim"])
        self.assertIn("exceeds", out["sim_rejected"][0])

    def test_wash_flag_on_loss_leg_with_recent_buy(self):
        d0 = self._dir()
        lot = self._first_priced_lot(d0, prefer_loss=True)
        buy = {"account_id": lot["account_id"],
               "transaction_type": "buy", "amount": -100.0,
               "quantity": 1.0, "symbol": lot["symbol"]
               or lot["instrument_key"],
               "trade_date": (ASOF - timedelta(days=5)).isoformat(),
               "settlement_date": (ASOF - timedelta(days=3)).isoformat(),
               "description": "BUY"}
        d = self._dir(extra_tx=[buy])
        lot2 = self._first_priced_lot(d, prefer_loss=True)
        out = self._est(d, sim=[{"lot_id": lot2["lot_id"],
                                 "qty": lot2["quantity_remaining"]}])
        leg = out["sim_legs"][0]
        if leg["gl"] < 0:
            self.assertTrue(leg["wash_observed"])
        else:
            # fixture lot is a gainer: flag must be False on gains
            self.assertFalse(leg["wash_observed"])

    def test_wash_flag_fires_on_a_reinvestment_not_just_a_buy(self):
        # parsers.tax_scanner.ACQUISITION_TYPES is ("buy", "reinvestment")
        # — a DRIP reinvestment is a replacement acquisition too, and is
        # exactly the row shape most likely to print no symbol. A vocab
        # that only matched "buy" silently missed it (review finding).
        d0 = self._dir()
        lot = self._first_priced_lot(d0, prefer_loss=True)
        reinvest = {"account_id": lot["account_id"],
                   "transaction_type": "reinvestment", "amount": -21.21,
                   "quantity": 0.42, "symbol": lot["symbol"]
                   or lot["instrument_key"],
                   "trade_date": (ASOF - timedelta(days=5)).isoformat(),
                   "settlement_date": (ASOF - timedelta(days=3)).isoformat(),
                   "description": "REINVESTMENT"}
        d = self._dir(extra_tx=[reinvest])
        lot2 = self._first_priced_lot(d, prefer_loss=True)
        out = self._est(d, sim=[{"lot_id": lot2["lot_id"],
                                 "qty": lot2["quantity_remaining"]}])
        leg = out["sim_legs"][0]
        if leg["gl"] < 0:
            self.assertTrue(leg["wash_observed"])
        else:
            # fixture lot is a gainer: flag must be False on gains
            self.assertFalse(leg["wash_observed"])

    def test_wash_flag_matches_resolved_identity_not_raw_symbol(self):
        # A crosswalk-resolved lot (empty symbol, ticker in
        # instrument_key) whose replacement acquisition ALSO prints no
        # symbol (a DRIP/account-transfer shape) must still flag — the
        # old raw `buys["symbol"] == sym_or_ikey` compare could never see
        # this (review finding). The fixture's own positions.csv already
        # ties the name "Synthetic Bond Fund" to symbol BBB uniquely, so
        # a buy row with blank symbol/cusip and that exact description
        # resolves to instrument_key "BBB" by NAME
        # (parsers.lot_engine.resolve_instrument_key) — deterministic,
        # no extra fixture rows needed beyond the lot and the buy.
        lot_row = {"account_id": "TEST-C", "instrument_key": "BBB",
                  "key_source": "resolved", "symbol": "",
                  "open_date": "2026-01-20", "acquired_date": "2026-01-20",
                  "origin": "buy", "quantity_open": 50,
                  "quantity_remaining": 50, "basis_open": 3_000.0,
                  "basis_remaining": 3_000.0, "source_row": "",
                  "basis_evidence": "reconstructed", "band": "ok"}
        buy = {"account_id": "TEST-C", "transaction_type": "buy",
              "amount": -505.0, "quantity": 10.0, "symbol": "",
              "cusip": "", "description": "Synthetic Bond Fund",
              "trade_date": (ASOF - timedelta(days=5)).isoformat(),
              "settlement_date": (ASOF - timedelta(days=3)).isoformat()}
        d = self._dir(extra_lots=[lot_row], extra_tx=[buy])
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        lot = next(r for r in view["lots"]
                  if r["account_id"] == "TEST-C"
                  and r["instrument_key"] == "BBB" and r["symbol"] == "")
        self.assertLess(lot["unrealized_gl"], 0)   # confirm it IS a loss
        out = self._est(d, sim=[{"lot_id": lot["lot_id"],
                                 "qty": lot["quantity_remaining"]}])
        leg = out["sim_legs"][0]
        self.assertLess(leg["gl"], 0)
        self.assertTrue(leg["wash_observed"])

    def test_wash_flag_excludes_the_lots_own_opening_purchase(self):
        # A loss lot ACQUIRED within the 30-day window is exactly what a
        # user is most likely to simulate selling. Without excluding the
        # lot's OWN opening buy from its own replacement-acquisition
        # hits, that buy matches itself and flags wash_observed=True
        # with zero genuine replacement activity — the false positive
        # this fix kills, mirroring scan_harvest_candidates' own
        # self-exclusion (parsers/tax_scanner.py ~line 303-310). AAA is
        # already priced (102.00) and its only OTHER transaction row is
        # from 2024, well outside the window, so reusing it needs no new
        # prices_latest.csv row and cannot pick up a stray match.
        base_rows = len(pd.read_csv(FIXTURE / "transactions.csv"))
        acquired = (ASOF - timedelta(days=5)).isoformat()
        lot_row = {"account_id": "TEST-C", "instrument_key": "AAA",
                  "key_source": "symbol", "symbol": "AAA",
                  "open_date": acquired, "acquired_date": acquired,
                  "origin": "buy", "quantity_open": 50,
                  "quantity_remaining": 50, "basis_open": 6_000.0,
                  "basis_remaining": 6_000.0, "source_row": base_rows,
                  "basis_evidence": "reconstructed", "band": "ok"}
        own_buy = {"account_id": "TEST-C", "transaction_type": "buy",
                  "amount": -6_000.0, "quantity": 50.0, "symbol": "AAA",
                  "cusip": "", "description": "AAA purchase",
                  "trade_date": acquired, "settlement_date": acquired}
        d = self._dir(extra_lots=[lot_row], extra_tx=[own_buy])
        view = txs.build_tax_view(hs.load_frames(d), d, asof=ASOF)
        lot = next(r for r in view["lots"]
                  if r["account_id"] == "TEST-C"
                  and r["instrument_key"] == "AAA")
        self.assertLess(lot["unrealized_gl"], 0)   # confirm it IS a loss
        out = self._est(d, sim=[{"lot_id": lot["lot_id"],
                                 "qty": lot["quantity_remaining"]}])
        leg = out["sim_legs"][0]
        self.assertLess(leg["gl"], 0)
        self.assertFalse(leg["wash_observed"])

    def test_profile_used_echoes_resolved_fields_only(self):
        d = self._dir()
        out = self._est(d)
        self.assertEqual(set(out["profile_used"]), set(FULL_PROFILE))


class TestTaxEstimateServer(unittest.TestCase):
    """POST /api/tax/estimate contract rows (spec §5.2). Mirrors
    TestTaxServer's mechanism: the env-var seam (raw ``os.environ``
    assignment, popped in cleanup) and a locally-imported TestClient
    over `terminal.server.app` — that class is the authority, not the
    plan's fallback sketch. A per-test temp dir (rather than
    TestTaxServer's shared class-level FIXTURE) is required here
    because every test needs a `realized_ytd` block injected — the
    estimator degrades named without one."""

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = _fixture_copy_dir(tmp)
        meta = json.loads((self.dir / "lots_meta.json")
                          .read_text(encoding="utf-8"))
        meta["realized_ytd"] = {
            "year": 2026,
            "by_account": {"TEST-A": {"short": {
                "gains": 1_000.0, "losses": -400.0, "net": 600.0,
                "closes": 2}}},
            "notes": {"excludes_alpine_options": True,
                      "options_source": "harbor_printed_confirms",
                      "broker_unresolved": 0}}
        (self.dir / "lots_meta.json").write_text(json.dumps(meta),
                                                 encoding="utf-8")
        os.environ["APP_DATA_DIR"] = str(self.dir)
        self.addCleanup(os.environ.pop, "APP_DATA_DIR", None)
        self.cfg = mock.patch.object(txs, "_config_profile",
                                     return_value={})
        self.cfg.start()
        self.addCleanup(self.cfg.stop)
        from fastapi.testclient import TestClient
        from terminal.server import app
        self.client = TestClient(app)

    def _post(self, body):
        # httpx's `json=` shortcut hardcodes allow_nan=False and raises
        # client-side (ValueError) before the request is even sent, which
        # would make the nan/inf cases below untestable through the wire
        # path they exist to exercise. Build the body ourselves — stdlib
        # json.dumps defaults to allow_nan=True, matching what a real
        # non-httpx client could send — so a non-finite value actually
        # reaches the route's own isfinite checks.
        return self.client.post(
            "/api/tax/estimate", content=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"})

    def test_happy_path_with_overrides(self):
        r = self._post({"overrides": FULL_PROFILE})
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual(out["kind"], "estimate")
        self.assertIn("baseline", out)
        self.assertIsNone(out["with_sim"])

    def test_empty_body_degrades_unconfigured(self):
        r = self._post({})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kind"], "error")
        self.assertIn("TAX_PROFILE", r.json()["reason"])

    def test_sim_leg_flows_through(self):
        view = self.client.get("/api/tax").json()
        lot = next(x for x in view["lots"]
                   if x["market_value"] is not None
                   and x["term"] in ("short", "long"))
        r = self._post({"overrides": FULL_PROFILE,
                        "sim": [{"lot_id": lot["lot_id"], "qty": 1.0}]})
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertIsNotNone(out["with_sim"])
        self.assertEqual(out["sim_legs"][0]["lot_id"], lot["lot_id"])

    def test_sim_leg_unknown_lot_id_200_with_sim_rejected(self):
        # Well-typed (passes TaxSimLeg's own Field constraints) but
        # semantically invalid — an id no lot in this book carries. That
        # is a 200 named rejection (spec §5.1 all-or-nothing), never a
        # 422: the shape is valid, only the reference is stale/wrong.
        r = self._post({"overrides": FULL_PROFILE,
                        "sim": [{"lot_id": 999_999, "qty": 1.0}]})
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual(out["kind"], "estimate")
        self.assertIsNone(out["with_sim"])
        self.assertTrue(out["sim_rejected"])

    def test_missing_data_dir_503(self):
        # Mirrors TestTaxServer.test_missing_data_dir_503's mechanism
        # (~line 1523) for the estimate route: hs.load_frames raising
        # FileNotFoundError -> 503, not the lots-not-built kind:"error"
        # degrade (that is a valid-request-but-empty-book case; this is
        # a misconfigured data dir).
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self._post({"overrides": FULL_PROFILE})
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(self.dir)

    def test_unknown_override_field_422s(self):
        r = self._post({"overrides": {**FULL_PROFILE, "ssn": "x"}})
        self.assertEqual(r.status_code, 422)

    def test_malformed_sim_leg_422s(self):
        self.assertEqual(self._post({"sim": [{"lot_id": 1}]}).status_code,
                         422)
        self.assertEqual(self._post({"sim": [{"lot_id": 1, "qty": 0}]}
                                    ).status_code, 422)
        self.assertEqual(self._post({"sim": [{"lot_id": 1,
                                              "qty": float("nan")}]}
                                    ).status_code, 422)

    def test_nonfinite_override_422s(self):
        r = self._post({"overrides": {**FULL_PROFILE,
                                      "w2_income": float("inf")}})
        self.assertEqual(r.status_code, 422)

    def test_nan_lot_id_422s_not_500(self):
        # a NaN INT field is rejected by pydantic itself (finite_number),
        # same crash class as qty one field over — the app-wide handler
        # must sanitize it before the 422 body serializes
        r = self._post({"sim": [{"lot_id": float("nan"), "qty": 1.0}]})
        self.assertEqual(r.status_code, 422)

    def test_nan_filing_status_422s_not_500(self):
        # a NaN into a plain `str | None` field fails pydantic's own
        # string-type check with the raw NaN echoed in the error detail —
        # unreachable by any route-level isfinite loop, since the route
        # body never runs when request validation itself rejects the shape
        r = self._post({"overrides": {**FULL_PROFILE,
                                      "filing_status": float("nan")}})
        self.assertEqual(r.status_code, 422)

    def test_nonfinite_deduction_422s(self):
        # unlike the other override floats, `deduction`'s `str | float`
        # union accepts a NaN/Inf float member cleanly — pydantic never
        # rejects it, so only the route's own isfinite loop catches it
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                r = self._post({"overrides": {**FULL_PROFILE,
                                              "deduction": bad}})
                self.assertEqual(r.status_code, 422)

    def test_optimize_route_nan_cap_pct_422s_not_500(self):
        # proves the app-wide handler GENERALIZES: OptimizeBody.cap_pct
        # (Field(ge=1.0, le=100.0), server.py) predates this task and
        # carries the identical NaN-echo crash exposure
        body = {"optimizer": "min_variance", "cap_pct": float("nan")}
        r = self.client.post(
            "/api/risksim/optimize", content=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 422)


class TestTaxEstimateGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_tax_estimate_golden.json")

    def test_matches_golden(self):
        view = _estimate_view()
        self.assertTrue(self.GOLDEN.exists(),
                        "estimate golden missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertIsNone(_deep_close(view, expected))


class TestWashCalendar(unittest.TestCase):
    """Ledger-independent per-ticker wash windows from transactions alone."""

    ASOF = date(2026, 8, 22)
    # SLV-1 is a test-shaped stand-in for the direct-index sleeve account
    # (a real account id previously sat here — de-identified, #384 riders).
    LABELS = {"A1": "Broker One Stocks", "B1": "Broker Two Stocks",
              "SLV-1": "Direct Index"}

    @staticmethod
    def _row(acct, ttype, symbol, qty, amount, trade, *, cusip=None,
             description=None, flag=None) -> dict:
        return {"account_id": acct, "transaction_type": ttype,
                "symbol": symbol, "cusip": cusip,
                "description": description or f"{symbol} COMMON STOCK",
                "quantity": qty, "amount": amount,
                "trade_date": trade, "settlement_date": trade,
                "tax_flag": flag}

    def _cal(self, rows, **kw):
        tx = pd.DataFrame(rows)
        return txs.wash_calendar(tx, asof=self.ASOF, labels=self.LABELS,
                                 resolver=kw.get("resolver", {}),
                                 cusip_resolver=kw.get("cusip_resolver", {}),
                                 fold=None)

    def test_buys_and_sells_inside_the_window_set_both_dates(self) -> None:
        cal = self._cal([
            self._row("A1", "buy", "STK1", 9.0, -5000.0, "2026-07-27"),
            self._row("A1", "sell", "STK1", -4.0, 2000.0, "2026-07-29", flag="W"),
            self._row("A1", "sell", "STK1", -10.0, 5900.0, "2026-08-14"),
        ])
        self.assertEqual(cal["as_of"], "2026-08-22")
        self.assertEqual(cal["window_days"], 30)
        self.assertEqual(cal["tx_frontier"], "2026-08-14")
        self.assertEqual(len(cal["tickers"]), 1)
        t = cal["tickers"][0]
        self.assertEqual(t["ticker"], "STK1")
        self.assertEqual(t["accounts"], ["Broker One Stocks"])
        self.assertEqual(t["acquisitions_in_window"], 1)
        self.assertEqual(t["last_acquisition"], "2026-07-27")
        self.assertEqual(t["wash_if_sold_before"], "2026-08-26")
        self.assertEqual(t["sells_in_window"], 2)
        self.assertEqual(t["last_sell"], "2026-08-14")
        self.assertEqual(t["wash_if_rebought_before"], "2026-09-13")
        self.assertEqual(t["broker_flagged_wash_sells"], 1)

    def test_activity_outside_the_window_is_ignored(self) -> None:
        cal = self._cal([
            self._row("A1", "buy", "OLD", 1.0, -10.0, "2026-07-01"),   # 52 days back
            self._row("A1", "buy", "NEW", 1.0, -10.0, "2026-07-23"),   # exactly 30 back
        ])
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["NEW"])

    def test_option_rows_and_cusip_shaped_keys_never_list(self) -> None:
        cal = self._cal([
            self._row("A1", "buy", "-SPYP575", 2.0, -300.0, "2026-08-10",
                      description=
                      "PUT (SPY) SPDR S&P 500 ETF DEC 18 26 $575 (100 SHS)"),
            self._row("SLV-1", "sell", None, -25000.0, 25000.0, "2026-08-10",
                      cusip="912828X88",
                      description="UNITED STATES TREASURY BILL"),
            self._row("A1", "buy", "AAA", 1.0, -10.0, "2026-08-11"),
        ])
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["AAA"])

    def test_accounts_merge_across_brokers_and_labels_stay_scrub_safe(self) -> None:
        cal = self._cal([
            self._row("A1", "buy", "AAA", 1.0, -10.0, "2026-08-11"),
            self._row("B1", "sell", "AAA", -1.0, 11.0, "2026-08-12"),
            self._row("999-55555", "buy", "AAA", 1.0, -10.0, "2026-08-13"),  # unlabeled
        ])
        t = cal["tickers"][0]
        self.assertEqual(t["accounts"],
                         ["Broker One Stocks", "Broker Two Stocks",
                          "unlabeled account"])
        self.assertEqual(t["acquisitions_in_window"], 2)
        self.assertEqual(t["sells_in_window"], 1)

    def test_symbolless_drip_row_keys_through_the_resolver(self) -> None:
        cal = self._cal([
            self._row("A1", "reinvestment", None, 0.5, -45.0, "2026-08-05",
                      description="SYNTHETIC EQUITY A"),
        ], resolver={"SYNTHETIC EQUITY A": "AAA"})
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["AAA"])
        self.assertEqual(cal["tickers"][0]["acquisitions_in_window"], 1)

    def test_missing_tax_flag_column_means_zero_flags(self) -> None:
        rows = [self._row("A1", "sell", "AAA", -1.0, 11.0, "2026-08-12")]
        for r in rows:
            r.pop("tax_flag")
        cal = self._cal(rows)
        # Compaction rider: zero/None keys are omitted from rows — a
        # sell-only ticker carries no flag count and no acquisition keys.
        t = cal["tickers"][0]
        self.assertNotIn("broker_flagged_wash_sells", t)
        self.assertNotIn("last_acquisition", t)
        self.assertNotIn("wash_if_sold_before", t)

    def test_empty_valued_row_keys_are_omitted(self) -> None:
        # Compaction rider (#380 final review, −15.5 KB on the real book):
        # None / 0 / empty keys leave the row; absent means zero/none and
        # the calendar note says so.
        cal = self._cal([
            self._row("A1", "buy", "BUYO", 1.0, -10.0, "2026-08-11"),
            self._row("A1", "sell", "SELO", -1.0, 11.0, "2026-08-12"),
        ])
        by = {t["ticker"]: t for t in cal["tickers"]}
        buy_only, sell_only = by["BUYO"], by["SELO"]
        for absent in ("sells_in_window", "last_sell",
                       "wash_if_rebought_before",
                       "broker_flagged_wash_sells"):
            self.assertNotIn(absent, buy_only)
        self.assertEqual(buy_only["acquisitions_in_window"], 1)
        self.assertEqual(buy_only["last_acquisition"], "2026-08-11")
        for absent in ("acquisitions_in_window", "last_acquisition",
                       "wash_if_sold_before"):
            self.assertNotIn(absent, sell_only)
        self.assertEqual(sell_only["sells_in_window"], 1)
        self.assertIn("absent from a row mean zero", cal["note"])

    def test_sleeve_only_activity_is_flagged(self) -> None:
        # Sleeve rider (#380 final review, −9.9 KB): a ticker whose ONLY
        # in-window activity sits in the sleeve account carries
        # sleeve_only true; mixed or non-sleeve activity carries no key
        # (False is omitted with the other empty values).
        tx = pd.DataFrame([
            self._row("SLV-1", "buy", "SLVA", 1.0, -10.0, "2026-08-11"),
            self._row("SLV-1", "sell", "SLVA", -1.0, 11.0, "2026-08-12"),
            self._row("SLV-1", "buy", "MIXD", 1.0, -10.0, "2026-08-11"),
            self._row("A1", "buy", "MIXD", 1.0, -10.0, "2026-08-12"),
            self._row("A1", "buy", "PLAIN", 1.0, -10.0, "2026-08-13"),
        ])
        cal = txs.wash_calendar(tx, asof=self.ASOF, labels=self.LABELS,
                                resolver={}, cusip_resolver={}, fold=None,
                                sleeve_accounts=frozenset({"SLV-1"}))
        by = {t["ticker"]: t for t in cal["tickers"]}
        self.assertIs(by["SLVA"].get("sleeve_only"), True)
        self.assertNotIn("sleeve_only", by["MIXD"])
        self.assertNotIn("sleeve_only", by["PLAIN"])

    def test_no_sleeve_accounts_means_no_flag(self) -> None:
        cal = self._cal([
            self._row("SLV-1", "buy", "SLVA", 1.0, -10.0, "2026-08-11"),
        ])
        self.assertNotIn("sleeve_only", cal["tickers"][0])

    def test_ticker_shape_accepts_slash_share_classes(self) -> None:
        # `/` rider: slash-printed share classes are ticker-shaped; cusips
        # and description slugs still are not.
        self.assertTrue(txs._ticker_like("BRK/B"))
        self.assertTrue(txs._ticker_like("HEI/A"))
        self.assertFalse(txs._ticker_like("912828X88"))
        self.assertFalse(txs._ticker_like("CASH REINVESTMENT SWEEP"))
        cal = self._cal([
            self._row("A1", "buy", "BRK/B", 1.0, -10.0, "2026-08-11"),
        ])
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["BRK/B"])

    def test_empty_ledger(self) -> None:
        cal = self._cal([])
        self.assertEqual(cal["tickers"], [])
        self.assertIsNone(cal["tx_frontier"])

    def test_same_day_buy_and_sell_set_both_dates(self) -> None:
        cal = self._cal([
            self._row("A1", "buy", "AAA", 1.0, -10.0, "2026-08-12"),
            self._row("A1", "sell", "AAA", -1.0, 11.0, "2026-08-12"),
        ])
        t = cal["tickers"][0]
        self.assertEqual(t["wash_if_sold_before"], "2026-09-11")
        self.assertEqual(t["wash_if_rebought_before"], "2026-09-11")

    def test_datetime_asof_is_normalized_to_the_day(self) -> None:
        from datetime import datetime
        tx = pd.DataFrame([self._row("A1", "buy", "AAA", 1.0, -10.0, "2026-07-23")])
        cal = txs.wash_calendar(tx, asof=datetime(2026, 8, 22, 14, 30),
                                labels=self.LABELS, resolver={},
                                cusip_resolver={}, fold=None)
        self.assertEqual(cal["as_of"], "2026-08-22")
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["AAA"])

    def test_unkeyed_rows_are_omitted_and_counted(self) -> None:
        # No symbol, no cusip, no resolver hit: the key is a description slug
        # (a cash sweep, an unresolved reinvestment) — never a ticker.
        cal = self._cal([
            self._row("A1", "buy", None, 1.0, -10.0, "2026-08-11",
                      description="CASH REINVESTMENT SWEEP"),
            self._row("A1", "reinvestment", None, 0.5, -5.0, "2026-08-12",
                      description="SOME UNRESOLVED EQUITY NAME"),
            self._row("A1", "buy", "AAA", 1.0, -10.0, "2026-08-13"),
        ])
        self.assertEqual([t["ticker"] for t in cal["tickers"]], ["AAA"])
        self.assertEqual(cal["unkeyed_rows_omitted"], 2)

    def test_none_frame_is_an_empty_calendar(self) -> None:
        cal = txs.wash_calendar(None, asof=self.ASOF, labels=self.LABELS,
                                resolver={}, cusip_resolver={}, fold=None)
        self.assertEqual(cal["tickers"], [])
        self.assertEqual(cal["unkeyed_rows_omitted"], 0)

    def test_digit_run_guard_matches_the_ai_scrub(self) -> None:
        from terminal import ai_service as ai
        self.assertEqual(txs._TICKER_DIGIT_RUN_RE.pattern, ai._DIGIT_RUN_RE.pattern)


if __name__ == "__main__":
    unittest.main()
