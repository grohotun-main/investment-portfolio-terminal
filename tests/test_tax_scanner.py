"""Tax S3a: harvest scanner + wash-sale detector unit tests.

Constructed frames only — synthetic accounts/symbols, no fixture reuse.
Spec: docs/superpowers/specs/2026-07-27-tax-s3-tlh-wash-sale-design.md §4.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsers.tax_scanner import (  # noqa: E402
    _lots_freshness_warning, _print_report, account_types,
    keyed_acquisitions, main as scanner_main, price_map, replacement_buys,
    scan_harvest_candidates, taxable_of, wash_check,
)


def _tx(rows):
    """Minimal canonical transactions frame for the scanner."""
    base = {"settlement_date": None, "trade_date": None, "broker": "harbor",
            "account_id": "TAX-1", "transaction_type": "buy", "symbol": "AAA",
            "cusip": None, "description": "ALPHA TRUST", "quantity": 1.0,
            "price": 10.0, "amount": -10.0, "source_file": "t.pdf",
            "closing_method": None, "closing_cost": None, "tax_flag": None}
    return pd.DataFrame([{**base, **r} for r in rows])


class TestKeyedAcquisitions(unittest.TestCase):
    def test_buy_and_reinvestment_key_in(self):
        acq = keyed_acquisitions(_tx([
            {"trade_date": "2026-07-01", "transaction_type": "buy"},
            {"trade_date": "2026-07-02", "transaction_type": "reinvestment"},
        ]))
        self.assertEqual(len(acq), 2)
        self.assertEqual(set(acq["instrument_key"]), {"AAA"})

    def test_sells_transfers_and_splits_do_not(self):
        acq = keyed_acquisitions(_tx([
            {"trade_date": "2026-07-01", "transaction_type": "sell"},
            {"trade_date": "2026-07-02", "transaction_type": "transfer_in"},
            {"trade_date": "2026-07-03", "transaction_type": "stock_split"},
        ]))
        self.assertTrue(acq.empty)

    def test_option_rows_never_match(self):
        # an option acquisition keyed by its underlying must not block an
        # equity harvest — options live outside the lot ledger
        acq = keyed_acquisitions(_tx([
            {"trade_date": "2026-07-01",
             "description": "CALL ZZZ 01/17/25 OPEN CONTRACT",
             "symbol": "ZZZ"},
        ]))
        self.assertTrue(acq.empty)

    def test_wash_date_prefers_trade_falls_back_settlement(self):
        acq = keyed_acquisitions(_tx([
            {"trade_date": "2026-07-01", "settlement_date": "2026-07-03"},
            {"trade_date": None, "settlement_date": "2026-07-04"},
        ]))
        self.assertEqual([str(d.date()) for d in acq["wash_date"]],
                         ["2026-07-01", "2026-07-04"])

    def test_cusip_resolution_uses_the_ledger_key_space(self):
        # a DRIP row keyed by cusip resolves onto the symbol the ledger uses
        acq = keyed_acquisitions(
            _tx([{"trade_date": "2026-07-01", "symbol": None,
                  "cusip": "123456789", "transaction_type": "reinvestment"}]),
            resolver={}, cusip_resolver={"123456789": "AAA"})
        self.assertEqual(list(acq["instrument_key"]), ["AAA"])


class TestReplacementBuys(unittest.TestCase):
    def _acq(self):
        return keyed_acquisitions(_tx([
            {"trade_date": "2026-06-01", "account_id": "TAX-1"},
            {"trade_date": "2026-07-01", "account_id": "IRA-1"},
            {"trade_date": "2026-07-31", "account_id": "TAX-2"},
            {"trade_date": "2026-08-01", "account_id": "TAX-2"},
            {"trade_date": "2026-07-01", "symbol": "BBB",
             "description": "BETA FUND"},
        ]))

    def test_window_is_inclusive_both_sides(self):
        hits = replacement_buys(self._acq(), "AAA", "2026-07-01")
        # 06-01 is day 30 back (in), 07-31 is day 30 forward (in),
        # 08-01 is day 31 forward (out); BBB never matches
        self.assertEqual([str(d.date()) for d in hits["wash_date"]],
                         ["2026-06-01", "2026-07-01", "2026-07-31"])

    def test_day_31_back_is_out(self):
        hits = replacement_buys(self._acq(), "AAA", "2026-07-02")
        self.assertNotIn("2026-06-01",
                         [str(d.date()) for d in hits["wash_date"]])

    def test_backward_mode_ignores_later_buys(self):
        hits = replacement_buys(self._acq(), "AAA", "2026-07-01",
                                sides="backward")
        self.assertEqual([str(d.date()) for d in hits["wash_date"]],
                         ["2026-06-01", "2026-07-01"])

    def test_cross_account_buys_count(self):
        hits = replacement_buys(self._acq(), "AAA", "2026-07-01")
        self.assertIn("IRA-1", set(hits["account_id"]))

    def test_empty_frame_yields_empty(self):
        hits = replacement_buys(keyed_acquisitions(_tx([])), "AAA",
                                "2026-07-01")
        self.assertTrue(hits.empty)


class TestAccountClassification(unittest.TestCase):
    def test_taxable_of(self):
        self.assertTrue(taxable_of("Brokerage"))
        self.assertFalse(taxable_of("Roth IRA"))
        self.assertFalse(taxable_of("TRADITIONAL IRA"))
        self.assertIsNone(taxable_of(None))
        self.assertIsNone(taxable_of(""))

    def test_account_types_latest_wins(self):
        pos = pd.DataFrame([
            {"account_id": "TAX-1", "account_type": "Individual",
             "statement_date": "2026-01-31"},
            {"account_id": "TAX-1", "account_type": "Brokerage",
             "statement_date": "2026-06-30"},
        ])
        self.assertEqual(account_types(pos), {"TAX-1": "Brokerage"})


class TestPriceMap(unittest.TestCase):
    def test_ok_and_cash_fixed_rows(self):
        frame = pd.DataFrame([
            {"symbol": "AAA", "status": "ok", "close": 12.5},
            {"symbol": "SWEEP", "status": "cash_fixed_1", "close": None},
            {"symbol": "BAD", "status": "stale", "close": 9.9},
        ])
        self.assertEqual(price_map(frame), {"AAA": 12.5, "SWEEP": 1.0})

    def test_none_and_empty(self):
        self.assertEqual(price_map(None), {})
        self.assertEqual(price_map(pd.DataFrame()), {})


def _lots(rows):
    base = {"account_id": "TAX-1", "instrument_key": "AAA",
            "key_source": "symbol", "symbol": "AAA",
            "open_date": "2024-01-10", "acquired_date": "2024-01-10",
            "origin": "buy", "quantity_open": 100.0,
            "quantity_remaining": 100.0, "basis_open": 5000.0,
            "basis_remaining": 5000.0, "source_row": 1,
            "basis_evidence": "reconstructed", "band": "ok"}
    return pd.DataFrame([{**base, **r} for r in rows])


TYPES = {"TAX-1": "Brokerage", "TAX-2": "Individual", "IRA-1": "Roth IRA"}
PRICES = {"AAA": 40.0, "BBB": 8.0}


class TestScanHarvestCandidates(unittest.TestCase):
    def test_a_clear_loss_lot_is_a_candidate(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES)
        cand = out["candidates"]
        self.assertEqual(len(cand), 1)
        row = cand.iloc[0]
        self.assertEqual(row["wash_status"], "clear")
        self.assertEqual(row["unrealized_gl"], 40.0 * 100 - 5000.0)
        self.assertEqual(row["term"], "long")
        self.assertEqual(row["window_ends"], "2026-08-27")
        self.assertEqual(out["summary"]["candidates"], 1)

    def test_gain_lots_do_not_surface(self):
        out = scan_harvest_candidates(
            _lots([{"basis_remaining": 1000.0}]), _tx([]), PRICES,
            as_of="2026-07-27", account_type_of=TYPES)
        self.assertTrue(out["candidates"].empty)
        self.assertEqual(out["summary"]["excluded_gain_or_flat"], 1)

    def test_printed_evidence_is_excluded_and_counted(self):
        out = scan_harvest_candidates(
            _lots([{"basis_evidence": "printed"}]), _tx([]), PRICES,
            as_of="2026-07-27", account_type_of=TYPES)
        self.assertTrue(out["candidates"].empty)
        self.assertEqual(out["summary"]["excluded_printed_evidence"], 1)

    def test_ira_and_unknown_account_lots_are_excluded(self):
        out = scan_harvest_candidates(
            _lots([{"account_id": "IRA-1"}, {"account_id": "MYSTERY-9"}]),
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        self.assertTrue(out["candidates"].empty)
        self.assertEqual(out["summary"]["excluded_non_taxable_accounts"], 2)

    def test_unpriced_lot_is_excluded_and_counted(self):
        out = scan_harvest_candidates(
            _lots([{"instrument_key": "NOPRICE", "symbol": "NOPRICE"}]),
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        self.assertTrue(out["candidates"].empty)
        self.assertEqual(out["summary"]["excluded_unpriced"], 1)

    def test_empty_symbol_falls_back_to_instrument_key(self):
        # real lots.csv rows carry symbol="" on resolved keys
        out = scan_harvest_candidates(
            _lots([{"symbol": ""}]), _tx([]), PRICES,
            as_of="2026-07-27", account_type_of=TYPES)
        self.assertEqual(len(out["candidates"]), 1)

    def test_empty_symbol_falls_back_after_a_real_csv_round_trip(self):
        # parsers/build_lots.py writes data/lots.csv with to_csv(); every
        # consumer (this scanner included) reads it back with read_csv().
        # That round trip turns a symbol="" cell into NaN, and
        # bool(float("nan")) is True in Python — the naive
        # `str(x or "") or fallback` idiom silently stringifies it to
        # "nan" instead of falling back to instrument_key. Reproduce the
        # real pipeline (not a hand-built NaN) so this catches it.
        lots = _lots([{"symbol": ""}])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "lots.csv"
            lots.to_csv(csv_path, index=False)
            reloaded = pd.read_csv(csv_path)
        self.assertTrue(pd.isna(reloaded["symbol"].iloc[0]))
        out = scan_harvest_candidates(
            reloaded, _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES)
        self.assertEqual(len(out["candidates"]), 1)
        self.assertEqual(out["summary"]["excluded_unpriced"], 0)

    def test_whitespace_only_symbol_falls_back_to_instrument_key(self):
        # a whitespace-only cell is not a symbol either
        out = scan_harvest_candidates(
            _lots([{"symbol": "   "}]), _tx([]), PRICES,
            as_of="2026-07-27", account_type_of=TYPES)
        self.assertEqual(len(out["candidates"]), 1)

    def test_recent_same_account_buy_blocks(self):
        out = scan_harvest_candidates(
            _lots([{}]),
            _tx([{"trade_date": "2026-07-10", "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        row = out["candidates"].iloc[0]
        self.assertEqual(row["wash_status"], "blocked")
        self.assertEqual(row["blocking_buys"][0]["date"], "2026-07-10")
        self.assertFalse(row["blocking_buys"][0]["is_ira"])
        # newest blocking buy + 31 days
        self.assertEqual(row["window_ends"], "2026-08-10")

    def test_a_cross_account_ira_drip_blocks_and_is_flagged(self):
        out = scan_harvest_candidates(
            _lots([{}]),
            _tx([{"trade_date": "2026-07-20", "account_id": "IRA-1",
                  "transaction_type": "reinvestment"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        row = out["candidates"].iloc[0]
        self.assertEqual(row["wash_status"], "blocked")
        self.assertTrue(row["blocking_buys"][0]["is_ira"])
        self.assertEqual(out["summary"]["ira_blocked"], 1)

    def test_an_unknown_account_type_buy_blocks_but_is_not_flagged_ira(self):
        # locked design: blocking acquisitions count from ANY account —
        # taxable, IRA, or unknown-type — fail-closed the dangerous way
        # round (a block we cannot rule out is still shown). Only a
        # PROVEN IRA earns is_ira: True; "MYSTERY-9" is absent from
        # account_type_of entirely, so it must still block, unflagged.
        out = scan_harvest_candidates(
            _lots([{}]),
            _tx([{"trade_date": "2026-07-15", "account_id": "MYSTERY-9",
                  "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        row = out["candidates"].iloc[0]
        self.assertEqual(row["wash_status"], "blocked")
        self.assertEqual(len(row["blocking_buys"]), 1)
        self.assertEqual(row["blocking_buys"][0]["account_id"], "MYSTERY-9")
        self.assertFalse(row["blocking_buys"][0]["is_ira"])
        self.assertEqual(out["summary"]["ira_blocked"], 0)

    def test_a_lots_own_opening_buy_does_not_block_it(self):
        # the buy that CREATED this lot (transactions row 0, mirrored onto
        # the lot's own source_row) is the acquisition of the shares being
        # sold -- not a replacement purchase. Selling a lot bought 17 days
        # ago, holding nothing else of it, is not a wash sale.
        out = scan_harvest_candidates(
            _lots([{"source_row": 0}]),
            _tx([{"trade_date": "2026-07-10", "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        row = out["candidates"].iloc[0]
        self.assertEqual(row["wash_status"], "clear")
        self.assertEqual(row["blocking_buys"], [])

    def test_a_distinct_second_purchase_still_blocks_alongside_the_opener(self):
        # narrow the fix: a genuinely SECOND acquisition (a different
        # transactions row) inside the window must still block, even when
        # the lot's own opener is also present and also inside the window
        out = scan_harvest_candidates(
            _lots([{"source_row": 0}]),
            _tx([{"trade_date": "2026-07-10",
                  "transaction_type": "buy"},          # row 0: the opener
                 {"trade_date": "2026-07-15",
                  "transaction_type": "buy"}]),         # row 1: a real add
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        row = out["candidates"].iloc[0]
        self.assertEqual(row["wash_status"], "blocked")
        self.assertEqual(len(row["blocking_buys"]), 1)
        self.assertEqual(row["blocking_buys"][0]["date"], "2026-07-15")

    def test_nan_source_row_on_synthesized_lot_does_not_crash(self):
        # synthesized opening lots (pre-history shortfall reconstruction)
        # carry no real transactions row for source_row -- nothing of the
        # lot's own to exclude, but a genuine replacement buy must still
        # block, and the NaN must not raise
        out = scan_harvest_candidates(
            _lots([{"source_row": None}]),
            _tx([{"trade_date": "2026-07-10", "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        row = out["candidates"].iloc[0]
        self.assertEqual(row["wash_status"], "blocked")
        self.assertEqual(row["blocking_buys"][0]["date"], "2026-07-10")

    def test_a_buy_31_days_ago_does_not_block(self):
        out = scan_harvest_candidates(
            _lots([{}]),
            _tx([{"trade_date": "2026-06-26", "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        self.assertEqual(out["candidates"].iloc[0]["wash_status"], "clear")

    def test_unknown_acquired_date_lists_with_unknown_term(self):
        out = scan_harvest_candidates(
            _lots([{"acquired_date": None}]), _tx([]), PRICES,
            as_of="2026-07-27", account_type_of=TYPES)
        self.assertEqual(out["candidates"].iloc[0]["term"], "unknown")

    def test_deepest_loss_sorts_first(self):
        out = scan_harvest_candidates(
            _lots([{}, {"instrument_key": "BBB", "symbol": "BBB",
                        "basis_remaining": 20000.0}]),
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        self.assertEqual(list(out["candidates"]["instrument_key"]),
                         ["BBB", "AAA"])

    def test_ira_and_unknown_accounts_are_separately_counted(self):
        # excluded_non_taxable_accounts merges "proven IRA" with "cannot
        # prove taxable" -- the honesty strip needs the IRA count on its
        # own, so the two must also be independently available
        out = scan_harvest_candidates(
            _lots([{"account_id": "IRA-1"}, {"account_id": "MYSTERY-9"}]),
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        s = out["summary"]
        self.assertEqual(s["excluded_ira_accounts"], 1)
        self.assertEqual(s["excluded_unknown_accounts"], 1)
        # combined key kept too -- unchanged, still reconciles
        self.assertEqual(s["excluded_non_taxable_accounts"], 2)

    def test_no_shares_remaining_is_separated_from_gain_or_flat(self):
        # excluded_gain_or_flat used to also absorb zero/NaN quantity
        # lots -- "no shares left to sell" and "at or above cost" are
        # different facts and must be counted separately
        out = scan_harvest_candidates(
            _lots([{"quantity_remaining": 0.0},        # no shares remaining
                   {"basis_remaining": 1000.0}]),       # at/above cost (gain)
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        s = out["summary"]
        self.assertEqual(s["excluded_no_shares_remaining"], 1)
        self.assertEqual(s["excluded_gain_or_flat"], 1)

    def test_all_exclusion_counters_reconcile_with_candidates_to_lots_seen(self):
        # the fine-grained buckets (never the combined keys, which would
        # double count) plus candidates must always foot to lots_seen
        out = scan_harvest_candidates(
            _lots([
                {},                                            # candidate
                {"account_id": "IRA-1"},                        # ira
                {"account_id": "MYSTERY-9"},                    # unknown acct
                {"basis_evidence": "printed"},                  # printed
                {"instrument_key": "NOPRICE", "symbol": "NOPRICE"},  # unpriced
                {"basis_remaining": None},                      # no basis
                {"quantity_remaining": 0.0},                    # no shares
                {"basis_remaining": 1000.0},                    # gain/flat
            ]),
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        s = out["summary"]
        self.assertEqual(s["lots_seen"], 8)
        self.assertEqual(s["candidates"], 1)
        self.assertEqual(s["excluded_ira_accounts"], 1)
        self.assertEqual(s["excluded_unknown_accounts"], 1)
        self.assertEqual(s["excluded_printed_evidence"], 1)
        self.assertEqual(s["excluded_unpriced"], 1)
        self.assertEqual(s["excluded_no_basis"], 1)
        self.assertEqual(s["excluded_no_shares_remaining"], 1)
        self.assertEqual(s["excluded_gain_or_flat"], 1)
        fine_grained_total = (
            s["excluded_ira_accounts"] + s["excluded_unknown_accounts"]
            + s["excluded_printed_evidence"] + s["excluded_unpriced"]
            + s["excluded_no_basis"] + s["excluded_no_shares_remaining"]
            + s["excluded_gain_or_flat"])
        self.assertEqual(s["candidates"] + fine_grained_total,
                         s["lots_seen"])

    def test_summary_blocked_count_and_total_unrealized_loss(self):
        # both are headline numbers and were previously unasserted
        out = scan_harvest_candidates(
            _lots([{},                                          # blocks
                   {"instrument_key": "BBB", "symbol": "BBB",
                    "basis_remaining": 900.0}]),                 # clear
            _tx([{"trade_date": "2026-07-10", "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        s = out["summary"]
        self.assertEqual(s["candidates"], 2)
        self.assertEqual(s["blocked"], 1)
        aaa_unrl = 40.0 * 100 - 5000.0
        bbb_unrl = 8.0 * 100 - 900.0
        self.assertEqual(s["total_unrealized_loss"],
                         round(aaa_unrl + bbb_unrl, 2))

    def test_tx_frontier_nat_behaves_like_unknown(self):
        # the defensive NaT branch, exercised directly rather than only
        # via the None-input tests above
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9, tx_frontier=pd.NaT)
        s = out["summary"]
        self.assertIsNone(s["window_days_observed"])
        self.assertIsNone(s["window_observed_pct"])
        self.assertIsNone(s["tx_frontier"])


class TestWindowObservability(unittest.TestCase):
    """tx_frontier: the ledger's data frontier vs. the backward wash
    window. Additive to the summary only — must never move wash_status."""

    def test_partial_observability_reports_observed_and_pct(self):
        # window_days=9 -> a 10-day backward window [as_of-9, as_of];
        # frontier sits 4 days in from the start, so 5 days are observed
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9, tx_frontier="2026-07-05")
        s = out["summary"]
        self.assertEqual(s["window_days_total"], 10)
        self.assertEqual(s["window_days_observed"], 5)
        self.assertEqual(s["window_observed_pct"], 50.0)
        self.assertEqual(s["tx_frontier"], "2026-07-05")

    def test_real_data_shaped_scenario_four_of_thirtyone(self):
        # the motivating case: statement frontier 2026-06-30, scan as_of
        # 2026-07-27, default 30-day window -> 4 of 31 days observed
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES, tx_frontier="2026-06-30")
        s = out["summary"]
        self.assertEqual(s["window_days_total"], 31)
        self.assertEqual(s["window_days_observed"], 4)
        self.assertEqual(s["window_observed_pct"], round(4 / 31 * 100, 1))

    def test_full_observability_when_frontier_at_as_of(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9, tx_frontier="2026-07-10")
        s = out["summary"]
        self.assertEqual(s["window_days_observed"], s["window_days_total"])
        self.assertEqual(s["window_observed_pct"], 100.0)

    def test_full_observability_when_frontier_after_as_of(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9, tx_frontier="2026-08-01")
        s = out["summary"]
        self.assertEqual(s["window_days_observed"], s["window_days_total"])
        self.assertEqual(s["window_observed_pct"], 100.0)

    def test_frontier_entirely_before_window_is_zero_observed(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9, tx_frontier="2026-06-01")
        s = out["summary"]
        self.assertEqual(s["window_days_observed"], 0)
        self.assertEqual(s["window_observed_pct"], 0.0)

    def test_unknown_frontier_yields_none_and_does_not_crash(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9, tx_frontier=None)
        s = out["summary"]
        self.assertIsNone(s["window_days_observed"])
        self.assertIsNone(s["window_observed_pct"])
        self.assertIsNone(s["tx_frontier"])
        self.assertEqual(s["window_days_total"], 10)  # always known

    def test_tx_frontier_defaults_to_none(self):
        # callers that omit tx_frontier get the same "unknown" behavior
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9)
        self.assertIsNone(out["summary"]["window_days_observed"])

    def test_accepts_a_date_object_not_just_a_string(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-10",
            account_type_of=TYPES, window_days=9,
            tx_frontier=date(2026, 7, 10))
        self.assertEqual(out["summary"]["tx_frontier"], "2026-07-10")

    def test_existing_summary_and_candidates_unaffected_by_tx_frontier(self):
        # additive only: every pre-existing key/value and the candidate
        # rows themselves are identical with or without tx_frontier
        base = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES)
        withf = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES, tx_frontier="2026-06-30")
        pre_existing = ["as_of", "window_days", "lots_seen",
                        "excluded_non_taxable_accounts",
                        "excluded_printed_evidence", "excluded_unpriced",
                        "excluded_no_basis", "excluded_gain_or_flat",
                        "candidates", "blocked", "ira_blocked",
                        "total_unrealized_loss"]
        for key in pre_existing:
            self.assertEqual(base["summary"][key], withf["summary"][key])
        pd.testing.assert_frame_equal(base["candidates"], withf["candidates"])


class TestPrintReportBlockers(unittest.TestCase):
    """_print_report's per-blocker line: quantity visibility + a
    status-keyed window label that cannot be misread as an expiry."""

    def _render(self, out_dict, tx_frontier="2026-07-27"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_report(out_dict, tx_frontier)
        return buf.getvalue()

    def test_blocking_buy_quantity_is_printed(self):
        # a 0.8-share dividend reinvestment currently vetoes an
        # arbitrarily large harvest with no visible scale -- the
        # replacement's quantity must reach the report
        out = scan_harvest_candidates(
            _lots([{}]),
            _tx([{"trade_date": "2026-07-10",
                  "transaction_type": "reinvestment", "quantity": 0.8}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        text = self._render(out)
        self.assertIn("0.8000", text)

    def test_window_labels_are_status_keyed_not_misreadable_as_expiry(self):
        # "clear ... window ends 2026-08-27" scans as "clear UNTIL
        # 2026-08-27" -- backwards from the intended "if you sell today,
        # don't repurchase before 2026-08-27". The label must differ by
        # wash_status so a clear row's date is never read as an expiry
        # of its clearness. AAA blocks (source_row default 1 != tx row
        # 0); BBB has no acquisitions at all and stays clear.
        out = scan_harvest_candidates(
            _lots([{}, {"instrument_key": "BBB", "symbol": "BBB"}]),
            _tx([{"trade_date": "2026-07-10", "transaction_type": "buy"}]),
            PRICES, as_of="2026-07-27", account_type_of=TYPES)
        text = self._render(out)
        self.assertIn("clears 2026-08-10", text)
        self.assertIn("unless rebought before 2026-08-27", text)
        self.assertNotIn("window ends", text)
        # window_ends is the first SAFE rebuy day (day 31 out): "by" reads
        # as on-or-before, which would wrongly taint that day too
        self.assertNotIn("rebought by", text)

    def test_clear_row_does_not_stutter_the_status_word(self):
        # the wash_status column already prints "clear"; the note beside
        # it must not repeat the word ("clear    clear unless..." stutters)
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES)
        text = self._render(out)
        row_line = next(ln for ln in text.splitlines() if "AAA" in ln)
        self.assertEqual(row_line.count("clear"), 1)

    def test_exclusion_line_prints_split_counters_without_the_doubled_word(self):
        # "excluded — printed-evidence lots excluded: N" repeats
        # "excluded" for no reason (no other item in the line does); and
        # the IRA/unknown-account split (item 5) must both be visible
        out = scan_harvest_candidates(
            _lots([{"account_id": "IRA-1"}, {"account_id": "MYSTERY-9"}]),
            _tx([]), PRICES, as_of="2026-07-27", account_type_of=TYPES)
        text = self._render(out)
        self.assertIn("printed-evidence lots: 0", text)
        self.assertNotIn("printed-evidence lots excluded:", text)
        self.assertIn("IRA accounts: 1", text)
        self.assertIn("unknown-type accounts: 1", text)


class TestPrintReportObservabilityCaveat(unittest.TestCase):
    """The observability caveat must fire whenever the scan cannot prove
    it saw the whole backward window -- including when it doesn't even
    know how much it saw. Silent only at genuine 100%."""

    def _render(self, out_dict, tx_frontier="unknown"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_report(out_dict, tx_frontier)
        return buf.getvalue()

    def test_unknown_observability_still_prints_a_caveat(self):
        # the LEAST informed case (frontier unknown) must not be the one
        # that goes silent -- a bare 'clear' with no caveat at all is the
        # exact overstatement the doctrine forbids
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES, tx_frontier=None)
        text = self._render(out)
        self.assertIn("observability unknown", text)
        self.assertIn("'clear' means", text)

    def test_full_observability_stays_silent(self):
        out = scan_harvest_candidates(
            _lots([{}]), _tx([]), PRICES, as_of="2026-07-27",
            account_type_of=TYPES, tx_frontier="2026-07-27")
        text = self._render(out, tx_frontier="2026-07-27")
        self.assertNotIn("observability unknown", text)
        self.assertNotIn("backward-window days", text)
        self.assertNotIn("'clear' means", text)


def _realizations(rows):
    base = {"account_id": "TAX-1", "instrument_key": "AAA",
            "open_date": "2026-01-05", "acquired_date": "2026-01-05",
            "close_date": "2026-07-01", "close_reason": "sell",
            "quantity_closed": 10.0, "basis_closed": 500.0,
            "proceeds": 400.0, "realized_gl": -100.0, "holding_days": 178,
            "term": "short", "closing_method": "FIFO", "source_row": 5,
            "basis_source": "reconstructed"}
    return pd.DataFrame([{**base, **r} for r in rows])


class TestWashCheck(unittest.TestCase):
    def _tx_with_sell(self, tax_flag, extra_rows=()):
        # index 5 = the sell row realizations point at via source_row
        rows = [{"trade_date": f"2026-0{i+1}-01"} for i in range(5)]
        rows.append({"trade_date": "2026-07-01", "transaction_type": "sell",
                     "quantity": -10.0, "amount": 400.0,
                     "tax_flag": tax_flag})
        rows.extend(extra_rows)
        return _tx(rows)

    def test_agree_wash(self):
        tx = self._tx_with_sell("W", [{"trade_date": "2026-07-10",
                                       "transaction_type": "buy"}])
        chk = wash_check(_realizations([{}]), tx)
        self.assertEqual(list(chk["bucket"]), ["agree_wash"])

    def test_agree_clean(self):
        # the early-January buys are outside the +/-30d window of 07-01
        chk = wash_check(_realizations([{}]), self._tx_with_sell(None))
        self.assertEqual(list(chk["bucket"]), ["agree_clean"])

    def test_broker_only_is_the_falsifier(self):
        chk = wash_check(_realizations([{}]), self._tx_with_sell("W"))
        self.assertEqual(list(chk["bucket"]), ["broker_only"])
        self.assertTrue(bool(chk.iloc[0]["printed_wash"]))
        self.assertFalse(bool(chk.iloc[0]["detector_wash"]))

    def test_detector_only(self):
        tx = self._tx_with_sell(None, [{"trade_date": "2026-06-20",
                                        "transaction_type": "buy"}])
        chk = wash_check(_realizations([{}]), tx)
        self.assertEqual(list(chk["bucket"]), ["detector_only"])
        # the window buy sits in the SAME account (TAX-1, the default) as
        # the sell -- something a single-broker view could see too
        self.assertFalse(bool(chk.iloc[0]["cross_account"]))

    def test_detector_only_cross_account_when_buy_is_in_another_account(self):
        # cross-account matches are precisely what no broker can see --
        # this feature's whole premise -- so they must be distinguishable
        # from a same-account detector_only row
        tx = self._tx_with_sell(None, [{"trade_date": "2026-06-20",
                                        "transaction_type": "buy",
                                        "account_id": "TAX-2"}])
        chk = wash_check(_realizations([{}]), tx)
        self.assertEqual(list(chk["bucket"]), ["detector_only"])
        self.assertTrue(bool(chk.iloc[0]["cross_account"]))

    def test_gain_sells_are_not_judged(self):
        chk = wash_check(_realizations([{"realized_gl": 50.0}]),
                         self._tx_with_sell("W"))
        self.assertTrue(chk.empty)

    def test_one_verdict_per_sell_row_not_per_lot_slice(self):
        # a multi-lot sell emits several realizations sharing source_row 5
        chk = wash_check(_realizations([{}, {"realized_gl": -40.0}]),
                         self._tx_with_sell("W"))
        self.assertEqual(len(chk), 1)
        self.assertEqual(chk.iloc[0]["realized_gl"], -140.0)

    def test_alpine_loss_sells_are_outside_the_denominator(self):
        tx = self._tx_with_sell("W")
        tx.loc[5, "broker"] = "alpine"
        chk = wash_check(_realizations([{}]), tx)
        self.assertTrue(chk.empty)
        # the falsifier's twin: nothing counted the excluded row before --
        # the report read "638 judged" as if that were the whole loss-sell
        # population, silently dropping every Alpine row off the books
        self.assertEqual(chk.attrs.get("excluded_other_broker"), 1)

    def test_mixed_broker_loss_sells_excludes_only_non_harbor(self):
        # a Harbor sell alongside a Alpine one must still be judged; only
        # the Alpine row leaves the judged universe
        rows = [{"trade_date": f"2026-0{i+1}-01"} for i in range(5)]
        rows.append({"trade_date": "2026-07-01", "transaction_type": "sell",
                     "quantity": -10.0, "amount": 400.0,
                     "broker": "alpine"})
        rows.append({"trade_date": "2026-07-01", "transaction_type": "sell",
                     "quantity": -10.0, "amount": 400.0, "broker": "harbor"})
        tx = _tx(rows)
        chk = wash_check(
            _realizations([{"source_row": 5}, {"source_row": 6}]), tx)
        self.assertEqual(list(chk["source_row"]), [6])
        self.assertEqual(chk.attrs.get("excluded_other_broker"), 1)

    def test_missing_tax_flag_column_returns_none(self):
        tx = self._tx_with_sell(None).drop(columns=["tax_flag"])
        self.assertIsNone(wash_check(_realizations([{}]), tx))

    def test_empty_transactions_with_tax_flag_column_is_not_column_absent(self):
        # zero rows, but the schema (including tax_flag) is present -- a
        # genuinely empty book, not a pre-S3 one. wash_check used to
        # collapse both onto the same None, so build_lots printed "tax_flag
        # column absent" for a book that had the column all along.
        tx = self._tx_with_sell(None).iloc[0:0]
        self.assertIn("tax_flag", tx.columns)
        chk = wash_check(_realizations([]).iloc[0:0], tx)
        self.assertIsNotNone(chk)
        self.assertTrue(chk.empty)

    def test_empty_realizations_yield_an_empty_frame(self):
        chk = wash_check(_realizations([]).iloc[0:0],
                         self._tx_with_sell(None))
        self.assertTrue(chk.empty)
        self.assertIsNotNone(chk)

    def test_forward_window_past_the_frontier_is_counted_unobserved(self):
        # the sell IS the transactions frontier (latest trade_date in the
        # whole frame), so its forward 30-day half necessarily runs past
        # what this book has actually seen
        tx = self._tx_with_sell(None)
        chk = wash_check(_realizations([{"close_date": "2026-07-01"}]), tx)
        self.assertEqual(chk.attrs.get("forward_unobserved"), 1)
        self.assertEqual(chk.attrs.get("tx_frontier"), "2026-07-01")

    def test_forward_window_within_the_frontier_is_observed(self):
        # a later buy pushes the frontier out past the sell's own +30d, so
        # the forward half is fully inside observed history
        tx = self._tx_with_sell(None, [{"trade_date": "2026-08-15",
                                        "transaction_type": "buy"}])
        chk = wash_check(_realizations([{"close_date": "2026-07-01"}]), tx)
        self.assertEqual(chk.attrs.get("forward_unobserved"), 0)

    def test_broker_frontiers_are_recorded_per_broker(self):
        # alpine's own latest observed row lags harbor's by two weeks --
        # the per-broker breakdown the report needs to show that the
        # max-over-frame frontier is not every broker's own frontier
        tx = self._tx_with_sell(None)
        extra = tx.iloc[[0]].copy()
        extra["trade_date"] = "2026-06-15"
        extra["broker"] = "alpine"
        tx = pd.concat([tx, extra], ignore_index=True)
        chk = wash_check(_realizations([{}]), tx)
        self.assertEqual(chk.attrs.get("broker_frontiers"),
                         {"harbor": "2026-07-01", "alpine": "2026-06-15"})

    def test_broker_frontiers_single_broker_book(self):
        chk = wash_check(_realizations([{}]), self._tx_with_sell(None))
        self.assertEqual(chk.attrs.get("broker_frontiers"),
                         {"harbor": "2026-07-01"})


class TestScannerCli(unittest.TestCase):
    def _write_book(self, d, with_lots=True):
        if with_lots:
            _lots([{}, {"basis_evidence": "printed"}]).to_csv(
                Path(d) / "lots.csv", index=False)
        _tx([{"trade_date": "2026-07-10", "transaction_type": "buy"}]).to_csv(
            Path(d) / "transactions.csv", index=False)
        pd.DataFrame([
            {"account_id": "TAX-1", "account_type": "Brokerage",
             "statement_date": "2026-06-30", "symbol": "AAA",
             "description": "ALPHA TRUST", "quantity": 100.0,
             "market_value": 4000.0, "cost_basis": 5000.0},
        ]).to_csv(Path(d) / "positions.csv", index=False)
        pd.DataFrame([
            {"symbol": "AAA", "status": "ok", "close": 40.0},
        ]).to_csv(Path(d) / "prices_latest.csv", index=False)

    def _write_ira_block_book(self, d):
        # same loss lot as _write_book, but the only replacement buy sits
        # in an IRA account instead of TAX-1 itself — exercises the
        # permanent-disallowance leg (vs. a taxable-account wash, deferred)
        _lots([{}, {"basis_evidence": "printed"}]).to_csv(
            Path(d) / "lots.csv", index=False)
        _tx([{"trade_date": "2026-07-10", "transaction_type": "buy",
              "account_id": "IRA-1"}]).to_csv(
            Path(d) / "transactions.csv", index=False)
        pd.DataFrame([
            {"account_id": "TAX-1", "account_type": "Brokerage",
             "statement_date": "2026-06-30", "symbol": "AAA",
             "description": "ALPHA TRUST", "quantity": 100.0,
             "market_value": 4000.0, "cost_basis": 5000.0},
            {"account_id": "IRA-1", "account_type": "Roth IRA",
             "statement_date": "2026-06-30", "symbol": "AAA",
             "description": "ALPHA TRUST", "quantity": 10.0,
             "market_value": 400.0, "cost_basis": 100.0},
        ]).to_csv(Path(d) / "positions.csv", index=False)
        pd.DataFrame([
            {"symbol": "AAA", "status": "ok", "close": 40.0},
        ]).to_csv(Path(d) / "prices_latest.csv", index=False)

    def _run(self, d, argv=None):
        buf = io.StringIO()
        env = {"APP_DATA_DIR": str(d)}
        with unittest.mock.patch.dict(os.environ, env):
            with contextlib.redirect_stdout(buf):
                code = scanner_main(argv or ["--as-of", "2026-07-27"])
        return code, buf.getvalue()

    def test_report_prints_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("== Harvest candidates ==", out)
        self.assertIn("blocked", out)          # the 07-10 buy blocks
        self.assertIn("printed-evidence lots: 1", out)

    def test_missing_lots_csv_exits_one_with_instructions(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d, with_lots=False)
            code, out = self._run(d)
        self.assertEqual(code, 1)
        self.assertIn("build_lots", out)

    def test_ira_block_prints_the_permanent_disallowance_legend(self):
        # a candidate blocked by an IRA replacement must carry a legend
        # saying the loss is permanently disallowed, not deferred — the
        # taxable-wash case (test below) must NOT print it
        with tempfile.TemporaryDirectory() as d:
            self._write_ira_block_book(d)
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("[IRA]", out)
        self.assertIn("permanently disallowed", out)
        self.assertIn("not deferred", out)

    def test_non_ira_block_omits_the_permanent_disallowance_legend(self):
        # blocked by TAX-1's OWN buy (taxable, not IRA) — the legend must
        # fire on an IRA block specifically, not merely on ANY block
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("blocked", out)
        self.assertNotIn("permanently disallowed", out)

    def test_partial_observability_prints_the_caveat(self):
        # _write_book's only tx row trades 2026-07-10; as-of 2026-07-27
        # with the default 30d window leaves the frontier short of as_of,
        # so the backward window is only partially observed: 14 of 31 days
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("14 of 31 backward-window days", out)
        self.assertIn("45.2%", out)
        self.assertIn("observed part of the window", out)

    def test_full_observability_omits_the_caveat(self):
        # as-of pinned to the tx frontier itself (2026-07-10) -> the
        # backward window sits entirely inside observed history
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            code, out = self._run(d, argv=["--as-of", "2026-07-10"])
        self.assertEqual(code, 0)
        self.assertNotIn("backward-window days", out)
        self.assertNotIn("observed part of the window", out)

    def test_unknown_frontier_at_cli_level_still_prints_the_caveat(self):
        # no transactions row has a parseable trade/settlement date ->
        # main() computes tx_frontier=None end to end. The least-informed
        # case must still surface the caveat, not go silent.
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            tx_path = Path(d) / "transactions.csv"
            tx = pd.read_csv(tx_path)
            tx["trade_date"] = None
            tx["settlement_date"] = None
            tx.to_csv(tx_path, index=False)
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("guard sees statement history through unknown", out)
        self.assertIn("observability unknown", out)

    def test_missing_lots_meta_prints_unverified_coupling(self):
        # no lots_meta.json in this fixture dir at all -- the CLI must say
        # the source_row coupling is unverified, not stay silent about it
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("lots_meta.json", out)
        self.assertIn("unverified", out)

    def test_stale_lots_meta_warns_before_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            (Path(d) / "lots_meta.json").write_text(
                json.dumps({"inputs": {"transactions_rows": 999}}),
                encoding="utf-8")
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertIn("WARNING", out)
        self.assertIn("STALE", out)
        self.assertIn("py parsers/build_lots.py --write", out)

    def test_fresh_lots_meta_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_book(d)
            tx_rows = len(pd.read_csv(Path(d) / "transactions.csv"))
            (Path(d) / "lots_meta.json").write_text(
                json.dumps({"inputs": {"transactions_rows": tx_rows}}),
                encoding="utf-8")
            code, out = self._run(d)
        self.assertEqual(code, 0)
        self.assertNotIn("STALE", out)
        self.assertNotIn("unverified", out)


class TestLotsFreshnessWarning(unittest.TestCase):
    """The source_row row-identity join (scan_harvest_candidates excluding
    a lot's own opener by transactions row position) only holds while
    lots.csv and the loaded transactions.csv describe the SAME positional
    ordering. combine_txns.py re-sorts and reset_index(drop=True)s on
    every rebuild, so a lots.csv built from an earlier transactions.csv
    can point source_row at the wrong row entirely once a re-ingest
    shifts everything after it. This is the standalone row-count check
    (parsers/ must never import terminal/, so this cannot reuse
    terminal/tax_service.py's freshness() -- it is a from-scratch copy of
    just that check's transactions-row half)."""

    def test_meta_absent_says_unverified(self):
        msg = _lots_freshness_warning(None, tx_rows=10)
        self.assertIsNotNone(msg)
        self.assertIn("lots_meta.json", msg)
        self.assertIn("unverified", msg)

    def test_matching_row_count_is_fresh(self):
        meta = {"inputs": {"transactions_rows": 10}}
        self.assertIsNone(_lots_freshness_warning(meta, tx_rows=10))

    def test_mismatched_row_count_warns_stale(self):
        meta = {"inputs": {"transactions_rows": 10}}
        msg = _lots_freshness_warning(meta, tx_rows=14)
        self.assertIsNotNone(msg)
        self.assertIn("STALE", msg)
        self.assertIn("source_row", msg)
        self.assertIn("py parsers/build_lots.py --write", msg)

    def test_meta_present_but_missing_the_field_is_treated_as_unverified(
            self):
        # a malformed/older meta file with no inputs.transactions_rows at
        # all must not print "built from None transactions row(s)"
        msg = _lots_freshness_warning({"inputs": {}}, tx_rows=10)
        self.assertIsNotNone(msg)
        self.assertNotIn("None", msg)


class TestModuleImportHasNoStdioSideEffect(unittest.TestCase):
    """tax_scanner.py is imported by terminal/tax_service.py inside the
    long-running terminal server process. An unconditional
    sys.stdout/stderr.reconfigure() at module top would mutate that
    process's console encoding as a mere side effect of importing this
    module — it must fire only for a direct script run, the same guard
    already used for the sys.path insert beside it. A fresh subprocess is
    required: sys.stdout is a C-level io.TextIOWrapper whose reconfigure
    method cannot be monkeypatched in-process."""

    def test_importing_as_a_package_module_does_not_reconfigure_stdio(self):
        script = (
            "import sys\n"
            "class Fake:\n"
            "    def __init__(self): self.calls = 0\n"
            "    def reconfigure(self, **kw): self.calls += 1\n"
            "    def write(self, s): return len(s)\n"
            "    def flush(self): pass\n"
            "fake_out, fake_err = Fake(), Fake()\n"
            "real_out = sys.stdout\n"
            "sys.stdout, sys.stderr = fake_out, fake_err\n"
            "import parsers.tax_scanner\n"
            "sys.stdout = real_out\n"
            "print(fake_out.calls, fake_err.calls)\n"
        )
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(repo_root),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0 0", result.stderr)


class TestKeyedRowsTypes(unittest.TestCase):
    """`types=` keys other row types (sells, for the chat wash calendar) in
    the same instrument-key space; the default stays buy/reinvestment."""

    def _tx(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": "A1", "transaction_type": "buy", "symbol": "AAA",
             "cusip": None, "description": "SYNTHETIC EQUITY A",
             "quantity": 10.0, "amount": -900.0,
             "trade_date": "2026-06-01", "settlement_date": "2026-06-02"},
            {"account_id": "A1", "transaction_type": "sell", "symbol": "AAA",
             "cusip": None, "description": "SYNTHETIC EQUITY A",
             "quantity": -4.0, "amount": 380.0,
             "trade_date": "2026-06-10", "settlement_date": "2026-06-11"},
            {"account_id": "A1", "transaction_type": "dividend", "symbol": "AAA",
             "cusip": None, "description": "SYNTHETIC EQUITY A DIVIDEND",
             "quantity": 0.0, "amount": 5.0,
             "trade_date": "2026-06-15", "settlement_date": "2026-06-15"},
        ])

    def test_default_is_acquisitions_only(self) -> None:
        out = keyed_acquisitions(self._tx())
        self.assertEqual(list(out["transaction_type"]), ["buy"])

    def test_types_keys_sells_with_same_columns(self) -> None:
        out = keyed_acquisitions(self._tx(), types=("sell",))
        self.assertEqual(list(out["transaction_type"]), ["sell"])
        self.assertEqual(list(out["instrument_key"]), ["AAA"])
        self.assertEqual(float(out["quantity"].iloc[0]), -4.0)
        self.assertEqual(str(out["wash_date"].iloc[0].date()), "2026-06-10")
        self.assertEqual(int(out["source_row"].iloc[0]), 1)
        self.assertEqual(list(out.columns),
                         ["account_id", "instrument_key", "wash_date",
                          "quantity", "transaction_type", "source_row"])


if __name__ == "__main__":
    unittest.main()
