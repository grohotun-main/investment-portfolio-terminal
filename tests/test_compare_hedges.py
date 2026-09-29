"""Tests for parsers/compare_hedges.py.

Two pure surfaces to lock down:
  * `parse_candidate_spec` — the CLI input contract; users hand-type these,
    so error messages must be specific.
  * `format_compare_table` — the user-visible deliverable; layout and
    per-mode metadata (atm skew, mixed expiries) must render correctly.

The CLI's `main()` does Polygon IO and is not exercised here — it's
covered by the end-to-end smoke run before merge.
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from compare_hedges import (  # noqa: E402
    CandidateSpec,
    _candidate_label,
    _sort_key,
    format_compare_table,
    parse_candidate_spec,
)
from stress_hedge import Hedge, Scenario, evaluate_hedge  # noqa: E402


class TestParseCandidateSpec(unittest.TestCase):

    def test_happy_path_put(self) -> None:
        s = parse_candidate_spec("525:2026-08-21:put")
        self.assertEqual(s.strike, 525.0)
        self.assertEqual(s.expiry, date(2026, 8, 21))
        self.assertEqual(s.option_type, "put")
        self.assertIsNone(s.premium)
        self.assertIsNone(s.sigma)
        self.assertIsNone(s.sigma_atm)

    def test_happy_path_call(self) -> None:
        s = parse_candidate_spec("100:2026-12-31:call")
        self.assertEqual(s.strike, 100.0)
        self.assertEqual(s.expiry, date(2026, 12, 31))
        self.assertEqual(s.option_type, "call")

    def test_type_case_insensitive(self) -> None:
        self.assertEqual(parse_candidate_spec("525:2026-08-21:PUT").option_type, "put")

    def test_fractional_strike(self) -> None:
        self.assertEqual(parse_candidate_spec("542.5:2026-08-21:put").strike, 542.5)

    def test_wrong_part_count_too_few(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("525:2026-08-21")
        self.assertIn("expected 3-6", str(cm.exception))

    def test_wrong_part_count_too_many(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("525:2026-08-21:put:1.16:0.38:0.16:extra")
        self.assertIn("expected 3-6", str(cm.exception))

    def test_non_numeric_strike(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("ATM:2026-08-21:put")
        self.assertIn("strike", str(cm.exception))

    def test_zero_strike(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("0:2026-08-21:put")
        self.assertIn("positive", str(cm.exception))

    def test_negative_strike(self) -> None:
        with self.assertRaises(ValueError):
            parse_candidate_spec("-10:2026-08-21:put")

    def test_bad_expiry_format(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("525:08/21/2026:put")
        self.assertIn("ISO", str(cm.exception))

    def test_bad_option_type(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("525:2026-08-21:option")
        self.assertIn("put", str(cm.exception))

    def test_optional_premium_only(self) -> None:
        s = parse_candidate_spec("525:2026-08-21:put:1.16")
        self.assertEqual(s.premium, 1.16)
        self.assertIsNone(s.sigma)
        self.assertIsNone(s.sigma_atm)

    def test_optional_premium_and_sigma(self) -> None:
        s = parse_candidate_spec("525:2026-08-21:put:1.16:0.38")
        self.assertEqual(s.premium, 1.16)
        self.assertEqual(s.sigma, 0.38)
        self.assertIsNone(s.sigma_atm)

    def test_full_six_fields(self) -> None:
        s = parse_candidate_spec("525:2026-08-21:put:1.16:0.38:0.16")
        self.assertEqual(s.premium, 1.16)
        self.assertEqual(s.sigma, 0.38)
        self.assertEqual(s.sigma_atm, 0.16)

    def test_empty_trailing_field_is_none(self) -> None:
        # Allows skipping the middle override (only-atm, no sigma):
        s = parse_candidate_spec("525:2026-08-21:put:1.16::0.16")
        self.assertEqual(s.premium, 1.16)
        self.assertIsNone(s.sigma)
        self.assertEqual(s.sigma_atm, 0.16)

    def test_negative_premium_rejected(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("525:2026-08-21:put:-1.16")
        self.assertIn("premium", str(cm.exception))

    def test_non_numeric_sigma_rejected(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse_candidate_spec("525:2026-08-21:put:1.16:hi")
        self.assertIn("sigma", str(cm.exception))


class TestCandidateLabel(unittest.TestCase):

    def test_single_letter(self) -> None:
        self.assertEqual(_candidate_label(0), "A")
        self.assertEqual(_candidate_label(1), "B")
        self.assertEqual(_candidate_label(25), "Z")

    def test_double_letter(self) -> None:
        self.assertEqual(_candidate_label(26), "AA")
        self.assertEqual(_candidate_label(27), "AB")
        self.assertEqual(_candidate_label(51), "AZ")
        self.assertEqual(_candidate_label(52), "BA")


class TestSortKey(unittest.TestCase):
    """_sort_key returns the column value evaluate_hedge produced. All
    sort modes are ascending — the user can read the comparison
    intuitively from cheapest/safest at top to richest/most-aggressive at bottom."""

    def _build_evs(self):
        today = date(2026, 5, 24)
        exp = today + timedelta(days=90)
        # Three evals with deliberately different premium / COI / breakeven
        # / COVID payouts so each sort key induces a distinct ordering.
        evA = _build_eval(475.0, exp, 0.18, 5.00, today, n=10)  # cheapest
        evB = _build_eval(490.0, exp, 0.16, 9.00, today, n=10)  # mid
        evC = _build_eval(500.0, exp, 0.15, 13.00, today, n=10) # priciest
        return [evA, evB, evC]

    def test_sort_by_premium_ascending(self) -> None:
        evs = self._build_evs()
        keys = [_sort_key(ev, "premium") for ev in evs]
        self.assertEqual(keys, sorted(keys))  # already in ascending premium
        self.assertEqual(keys, [5.0, 9.0, 13.0])

    def test_sort_by_coi_matches_pct(self) -> None:
        evs = self._build_evs()
        for ev in evs:
            self.assertEqual(_sort_key(ev, "coi"), ev.cost_of_insurance_pct)

    def test_sort_by_breakeven_decline_matches(self) -> None:
        evs = self._build_evs()
        for ev in evs:
            self.assertEqual(_sort_key(ev, "breakeven-decline"),
                             ev.breakeven_decline_pct)

    def test_sort_by_covid_pnl_uses_last_scenario(self) -> None:
        # _sort_key uses the last scenario, which is COVID-style in defaults.
        evs = self._build_evs()
        for ev in evs:
            self.assertEqual(_sort_key(ev, "covid-pnl"), ev.scenarios[-1].pnl_total)

    def test_unknown_sort_key_raises(self) -> None:
        evs = self._build_evs()
        with self.assertRaises(ValueError):
            _sort_key(evs[0], "moneyness")


def _build_eval(strike: float, expiry: date, sigma: float, premium: float,
                today: date, spot: float = 500.0, n: int = 1,
                opt: str = "put", vol_baseline: str = "contract",
                sigma_atm: float | None = None):
    h = Hedge(ticker="SPY", option_type=opt, strike=strike, expiry=expiry,
              n_contracts=n, premium_per_share=premium)
    return evaluate_hedge(
        h, spot_today=spot, sigma_today=sigma, r=0.045, q=0.013,
        today=today, vol_baseline=vol_baseline, sigma_atm=sigma_atm,
    )


class TestFormatCompareTable(unittest.TestCase):
    """Format checks: structural elements must appear; per-mode metadata
    must show up; numeric values must match the underlying eval objects."""

    def _today(self) -> date:
        return date(2026, 5, 24)

    def test_renders_ticker_spot_and_rate(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today)
        out = format_compare_table(
            ticker="SPY", spot=500.0, r=0.045,
            evals=[ev1, ev2],
            q_by_expiry={exp: 0.013},
            today=today,
        )
        self.assertIn("SPY", out)
        self.assertIn("500.00", out)
        self.assertIn("4.50%", out)

    def test_per_expiry_q_line(self) -> None:
        today = self._today()
        exp_a = today + timedelta(days=88)
        exp_b = today + timedelta(days=117)
        ev1 = _build_eval(475.0, exp_a, 0.18, 5.00, today)
        ev2 = _build_eval(475.0, exp_b, 0.18, 6.50, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp_a: 0.012, exp_b: 0.014},
            today=today,
        )
        self.assertIn(exp_a.isoformat(), out)
        self.assertIn(exp_b.isoformat(), out)
        self.assertIn("+1.200%", out)
        self.assertIn("+1.400%", out)

    def test_candidate_labels_appear_in_order(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today)
        ev3 = _build_eval(500.0, exp, 0.15, 12.00, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2, ev3],
            q_by_expiry={exp: 0.013}, today=today,
        )
        a_pos = out.find("A) K=475")
        b_pos = out.find("B) K=490")
        c_pos = out.find("C) K=500")
        self.assertGreater(a_pos, 0)
        self.assertGreater(b_pos, a_pos)
        self.assertGreater(c_pos, b_pos)

    def test_scenario_rows_present(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        # All four DEFAULT_SCENARIOS by name
        self.assertIn("Mild correction", out)
        self.assertIn("Liberation-Day-style", out)
        self.assertIn("Moderate crash", out)
        self.assertIn("COVID-style", out)

    def test_pnl_dollar_values_match_evals(self) -> None:
        # The COVID-style total $ must appear in the row, formatted with
        # commas + sign. Round-trip the value from the eval object.
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today, n=10)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today, n=10)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        for ev in (ev1, ev2):
            covid = next(s for s in ev.scenarios if s.scenario.name == "COVID-style")
            expected = f"{covid.pnl_total:+,.0f}"
            self.assertIn(expected, out,
                          msg=f"missing COVID P&L {expected} for K={ev.hedge.strike}")

    def test_atm_mode_shows_skew(self) -> None:
        # In atm mode the candidate block must display ATM IV + skew points.
        today = self._today()
        exp = today + timedelta(days=90)
        ev_wing = _build_eval(450.0, exp, 0.38, 1.20, today,
                              vol_baseline="atm", sigma_atm=0.16)
        ev_close = _build_eval(490.0, exp, 0.18, 5.00, today,
                               vol_baseline="atm", sigma_atm=0.16)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev_wing, ev_close],
            q_by_expiry={exp: 0.013},
            atm_iv_by_expiry={exp: 0.16},
            today=today,
        )
        self.assertIn("ATM 16.00%", out)  # in candidate row
        # Skew: wing contract IV 38% - ATM 16% = +22.00 pp
        self.assertIn("+22.00pp", out)
        # Close-to-ATM: 18% - 16% = +2.00 pp
        self.assertIn("+2.00pp", out)
        # Mode line
        self.assertIn("vol_baseline=atm", out)

    def test_contract_mode_no_skew_display(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        # Contract mode: no "skew" or "ATM" annotation in candidate rows
        self.assertNotIn("skew", out)
        # Mode line
        self.assertIn("vol_baseline=contract", out)

    def test_mixed_expiry_header_changes(self) -> None:
        today = self._today()
        exp_a = today + timedelta(days=60)
        exp_b = today + timedelta(days=120)
        ev1 = _build_eval(475.0, exp_a, 0.18, 4.00, today)
        ev2 = _build_eval(475.0, exp_b, 0.18, 6.00, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp_a: 0.013, exp_b: 0.013}, today=today,
        )
        self.assertIn("Per-expiry market frame", out)

    def test_single_expiry_header_changes(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        self.assertIn("Expiry frame", out)
        self.assertNotIn("Per-expiry market frame", out)

    def test_rejects_mismatched_scenario_counts(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        custom = (Scenario("Just one", 0.95, 1.5),)
        h1 = Hedge("SPY", "put", 475.0, exp, 1, 5.00)
        h2 = Hedge("SPY", "put", 490.0, exp, 1, 8.00)
        ev1 = evaluate_hedge(h1, 500.0, 0.18, 0.045, 0.013, today=today,
                             scenarios=custom)
        ev2 = evaluate_hedge(h2, 500.0, 0.16, 0.045, 0.013, today=today)
        # ev2 has 4 default scenarios, ev1 has 1 → mismatch
        with self.assertRaises(ValueError) as cm:
            format_compare_table(
                "SPY", 500.0, 0.045, [ev1, ev2],
                q_by_expiry={exp: 0.013}, today=today,
            )
        self.assertIn("same scenario", str(cm.exception))

    def test_varying_n_label(self) -> None:
        # n differs across candidates → header says "n varies", not "n=X each"
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today, n=10)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today, n=5)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        self.assertIn("n varies", out)

    def test_same_n_label(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today, n=7)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today, n=7)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        self.assertIn("n=7", out)

    def test_mtm_breakeven_displayed(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        # Both labels appear (at-expiry and MTM) and neither shows n/a:
        self.assertIn("BE@exp", out)
        self.assertIn("BE@mtm", out)
        self.assertNotIn("BE@mtm n/a", out)

    def test_mtm_breakeven_na_at_expiry(self) -> None:
        # Expiry-day hedge has T=0 → MTM BE is None → display "BE@mtm n/a"
        today = self._today()
        h = Hedge("SPY", "put", 525.0, today, 1, 25.0)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        h2 = Hedge("SPY", "put", 530.0, today, 1, 30.0)
        ev2 = evaluate_hedge(h2, 500.0, 0.18, 0.045, 0.013, today=today)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev, ev2],
            q_by_expiry={today: 0.013}, today=today,
        )
        self.assertIn("BE@mtm n/a", out)

    def test_best_in_scenario_marker_appears_once_per_row(self) -> None:
        today = self._today()
        exp = today + timedelta(days=90)
        # Three candidates with distinct payoffs — one winner per row.
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today, n=10)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today, n=10)
        ev3 = _build_eval(500.0, exp, 0.15, 13.00, today, n=10)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2, ev3],
            q_by_expiry={exp: 0.013}, today=today,
        )
        # Each scenario row contains exactly one " *" marker. Count by line.
        scenario_names = [s.scenario.name for s in ev1.scenarios]
        for name in scenario_names:
            # Find the row that starts with this scenario name
            row = next(line for line in out.splitlines()
                       if line.startswith(name))
            self.assertEqual(row.count(" *"), 1,
                             msg=f"row {name!r} should have exactly one marker, "
                                 f"got: {row!r}")

    def test_best_in_scenario_marker_at_actual_winner(self) -> None:
        # Construct evals where we know which candidate wins each row.
        # Higher strike put pays more on the same crash → ev2 (K=490) wins
        # every default-scenario row vs ev1 (K=475) at same n.
        today = self._today()
        exp = today + timedelta(days=90)
        ev1 = _build_eval(475.0, exp, 0.18, 5.00, today, n=10)
        ev2 = _build_eval(490.0, exp, 0.16, 8.00, today, n=10)
        # Confirm ev2 beats ev1 in every scenario (else the test is meaningless)
        for s1, s2 in zip(ev1.scenarios, ev2.scenarios):
            self.assertGreater(s2.pnl_total, s1.pnl_total)
        out = format_compare_table(
            "SPY", 500.0, 0.045, [ev1, ev2],
            q_by_expiry={exp: 0.013}, today=today,
        )
        # The B-column's cells should carry " *"; the A-column shouldn't.
        for name in (s.scenario.name for s in ev1.scenarios):
            row = next(line for line in out.splitlines()
                       if line.startswith(name))
            # The marker should be at the very end (B is the second/last col)
            self.assertTrue(row.rstrip().endswith("*"),
                            msg=f"row {name!r} should end with marker (B won): {row!r}")


if __name__ == "__main__":
    unittest.main()
