"""Engine tests for parsers/tax_estimate.py — every spec §4.3 rule.

All figures synthetic (#310). Expected values are hand-walked from the
module's own bracket constants; a constants update (new tax year) is
EXPECTED to break these — that is the pin working.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parsers"))

from parsers.tax_estimate import (  # noqa: E402
    TAX_YEAR, estimate_year_tax, is_treasury_income)

NO_INCOME = {"dividends": 0.0, "interest": 0.0,
             "treasury_interest": 0.0, "withholding": 0.0}
NO_REALIZED = {"short": 0.0, "long": 0.0, "unknown": 0.0}


def prof(**kw):
    base = {"filing_status": "single", "w2_income": 100_000.0,
            "state": "CA", "deduction": "standard",
            "carryforward_loss": 0.0, "qualified_dividend_pct": 1.0,
            "unknown_term_assumption": "long"}
    base.update(kw)
    return base


def est(profile=None, realized=None, income=None, sim=None):
    return estimate_year_tax(profile or prof(),
                             realized or dict(NO_REALIZED),
                             income or dict(NO_INCOME), sim)


class TestBaseline(unittest.TestCase):
    def test_year_constant(self):
        self.assertEqual(TAX_YEAR, 2026)

    def test_zero_portfolio_is_zero_everywhere(self):
        # The delta-vs-wages definition: W-2 alone contributes nothing.
        r = est()
        for k in ("total", "federal", "state", "niit", "ftc"):
            self.assertEqual(r[k], 0, k)

    def test_w2_size_alone_never_changes_the_estimate(self):
        lo = est(prof(w2_income=50_000.0))
        hi = est(prof(w2_income=400_000.0))
        self.assertEqual(lo["total"], 0)
        self.assertEqual(hi["total"], 0)

    def test_totals_compose_from_rounded_leaves(self):
        # fractional inputs that round the raw sum differently than the
        # leaf sum — total/federal must equal their parts EXACTLY
        for income in ({**NO_INCOME, "interest": 10_001.61},
                       {**NO_INCOME, "dividends": 10_002.86}):
            r = est(prof(qualified_dividend_pct=0.6), income=income)
            self.assertEqual(r["federal"],
                             r["federal_ordinary"]
                             + r["federal_preferential"])
            self.assertEqual(r["total"],
                             r["federal"] + r["state"] + r["niit"]
                             - r["ftc"])


class TestFederalOrdinary(unittest.TestCase):
    def test_interest_inside_one_bracket(self):
        # single, W-2 100,000, std 16,100 -> taxable 83,900 (22% band);
        # +10,000 interest stays inside the band -> 2,200.
        r = est(income={**NO_INCOME, "interest": 10_000.0})
        self.assertEqual(r["federal_ordinary"], 2_200)
        self.assertEqual(r["state"], 930)      # CA 9.3% band x 10,000
        self.assertEqual(r["niit"], 0)         # MAGI 110,000 < 200,000

    def test_bracket_crossing_splits_the_rate(self):
        # single, W-2 65,500 -> taxable 49,400; +2,000 interest crosses
        # the 12%->22% edge at 50,400: 1,000@12% + 1,000@22% = 340.
        r = est(prof(w2_income=65_500.0),
                income={**NO_INCOME, "interest": 2_000.0})
        self.assertEqual(r["federal_ordinary"], 340)

    def test_marginal_rates_reported(self):
        r = est(income={**NO_INCOME, "interest": 10_000.0})
        self.assertEqual(r["marginal"]["fed_ordinary"], 0.22)
        self.assertEqual(r["marginal"]["ca"], 0.093)

    def test_married_joint_tables(self):
        # MFJ, W-2 200,000, std 32,200 -> taxable 167,800 (22% band
        # 100,801-211,400); +10,000 interest stays inside -> 2,200.
        # CA MFJ: 200,000-11,412=188,588 (9.3% band from 145,448);
        # +10,000 -> 930. NIIT: MAGI 210,000 < 250,000 -> 0.
        r = est(prof(filing_status="married_joint", w2_income=200_000.0),
                income={**NO_INCOME, "interest": 10_000.0})
        self.assertEqual(r["federal_ordinary"], 2_200)
        self.assertEqual(r["state"], 930)
        self.assertEqual(r["niit"], 0)


class TestFederalPreferential(unittest.TestCase):
    def test_lt_zero_band_stacking(self):
        # single, W-2 46,100 -> ordinary taxable 30,000. LT gain 30,000
        # occupies [30,000, 60,000): 19,450 in the 0% band (edge 49,450),
        # 10,550 @15% = 1,582.50 -> 1582 rounded.
        r = est(prof(w2_income=46_100.0),
                realized={**NO_REALIZED, "long": 30_000.0})
        self.assertEqual(r["federal_preferential"], 1_582)

    def test_lt_15_to_20_straddle(self):
        # single, ordinary taxable 500,000 (W-2 516,100). LT 100,000:
        # 45,500 @15% + 54,500 @20% = 17,725.
        r = est(prof(w2_income=516_100.0),
                realized={**NO_REALIZED, "long": 100_000.0})
        self.assertEqual(r["federal_preferential"], 17_725)

    def test_qualified_dividend_pct_splits_layers(self):
        # 10,000 dividends at qdp 0.6 -> 4,000 ordinary + 6,000 preferential.
        r = est(prof(qualified_dividend_pct=0.6),
                income={**NO_INCOME, "dividends": 10_000.0})
        self.assertEqual(r["federal_ordinary"], 880)        # 4,000 x 22%
        self.assertEqual(r["federal_preferential"], 900)    # 6,000 x 15%


class TestItemizedDeduction(unittest.TestCase):
    def test_itemized_deduction_applied_to_both_jurisdictions(self):
        # single, W-2 100,000, itemized deduction 30,000 (both fed and CA
        # per _core: ded = ca_ded = float(deduction) when not "standard").
        # fed_base = 100,000 - 30,000 = 70,000; +10,000 interest ->
        # fed_with = 80,000. Both endpoints sit inside FED_BRACKETS
        # single's 22% band (50,400, 105,700] (the "22% band starts
        # 50,401" edge), so the whole $10,000 layer is one rate:
        #   10,000 x 22% = 2,200.00 -> 2,200
        #
        # CA: ca_base = 70,000 sits inside CA_BRACKETS single's 8% band
        # (57,542, 72,724]; the +10,000 layer crosses that edge at
        # 72,724, splitting into two rates:
        #   72,724 - 70,000 = 2,724 @ 8%   = 217.92
        #   80,000 - 72,724 = 7,276 @ 9.3% = 676.668
        #   total                          = 894.588 -> round -> 895
        r = est(prof(deduction=30_000.0),
                income={**NO_INCOME, "interest": 10_000.0})
        self.assertEqual(r["federal_ordinary"], 2_200)
        self.assertEqual(r["state"], 895)


class TestNetting(unittest.TestCase):
    def test_cross_absorb_st_loss_into_lt_gain(self):
        r = est(realized={"short": -10_000.0, "long": 15_000.0,
                          "unknown": 0.0})
        self.assertEqual(r["netting"]["st_net"], 0.0)
        self.assertEqual(r["netting"]["lt_net"], 5_000.0)
        self.assertEqual(r["federal_preferential"], 750)    # 5,000 @15%

    def test_ordinary_offset_capped_at_3000(self):
        # net -8,000: offset -3,000, carryforward_out -5,000, and the
        # offset REDUCES tax vs wages-only (negative attributable is real).
        r = est(realized={"short": -8_000.0, "long": 0.0, "unknown": 0.0})
        self.assertEqual(r["netting"]["ordinary_offset"], -3_000.0)
        self.assertEqual(r["netting"]["carryforward_out"], -5_000.0)
        self.assertEqual(r["federal_ordinary"], -660)       # -3,000 x 22%
        self.assertEqual(r["state"], -279)                  # -3,000 x 9.3%
        self.assertEqual(r["niit"], 0)                      # NII clamps at 0
        self.assertEqual(r["total"], -939)

    def test_carryforward_loss_reduces_lt(self):
        r = est(prof(carryforward_loss=2_000.0),
                realized={**NO_REALIZED, "long": 10_000.0})
        self.assertEqual(r["netting"]["lt_net"], 8_000.0)
        self.assertEqual(r["federal_preferential"], 1_200)  # 8,000 @15%

    def test_cross_absorb_lt_loss_into_st_gain(self):
        r = est(realized={"short": 15_000.0, "long": -10_000.0,
                          "unknown": 0.0})
        self.assertEqual(r["netting"]["st_net"], 5_000.0)
        self.assertEqual(r["netting"]["lt_net"], 0.0)
        self.assertEqual(r["federal_ordinary"], 1_100)   # 5,000 x 22%


class TestUnknownTerm(unittest.TestCase):
    def test_swing_positive_when_short_would_cost_more(self):
        r = est(realized={**NO_REALIZED, "unknown": 50_000.0})
        self.assertEqual(r["unknown_term"]["assumption"], "long")
        self.assertEqual(r["unknown_term"]["amount"], 50_000.0)
        self.assertGreater(r["unknown_term"]["swing_if_other"], 0)

    def test_swing_symmetry_under_flipped_assumption(self):
        long_first = est(realized={**NO_REALIZED, "unknown": 50_000.0})
        short_first = est(prof(unknown_term_assumption="short"),
                          realized={**NO_REALIZED, "unknown": 50_000.0})
        self.assertEqual(short_first["total"],
                         long_first["total"]
                         + long_first["unknown_term"]["swing_if_other"])
        self.assertEqual(short_first["unknown_term"]["swing_if_other"],
                         -long_first["unknown_term"]["swing_if_other"])


class TestCalifornia(unittest.TestCase):
    def test_treasury_interest_exempt_from_ca_only(self):
        # 20,000 interest, 15,000 of it treasury: CA taxes 5,000 (465 at
        # 9.3%); fed taxes all 20,000 across the 24% bracket (crossing
        # into 32%): 17,875@24% + 2,125@32% = 4,970.
        r = est(prof(w2_income=200_000.0),
                income={**NO_INCOME, "interest": 20_000.0,
                        "treasury_interest": 15_000.0})
        self.assertEqual(r["state"], 465)
        self.assertEqual(r["federal_ordinary"], 4_970)
        # Pin: treasury interest IS in NII (§4.3.5) — `nii` in _core sums
        # the FULL `interest` figure, not `interest - treasury` (that
        # subtraction is CA-only). magi = w2 200,000 + nii 20,000 =
        # 220,000; excess over the single NIIT_THRESHOLD (200,000) =
        # 20,000; niit = 3.8% x min(nii=20,000, excess=20,000) =
        # 3.8% x 20,000 = 760.
        self.assertEqual(r["niit"], 760)

    def test_lt_gain_is_ordinary_for_ca(self):
        # CA has no preferential rate: a pure LT gain still lands in state.
        r = est(realized={**NO_REALIZED, "long": 10_000.0})
        self.assertEqual(r["state"], 930)                   # 9.3% band

    def test_mental_health_surtax_band(self):
        # single, W-2 1,050,000, LT 100,000: fed ord delta 0; LT all @20%
        # = 20,000; CA 100,000 x 12.3% + 1% MHST on the added 100,000
        # = 13,300; NIIT 3,800. Total 37,100.
        r = est(prof(w2_income=1_050_000.0),
                realized={**NO_REALIZED, "long": 100_000.0})
        self.assertEqual(r["federal_ordinary"], 0)
        self.assertEqual(r["federal_preferential"], 20_000)
        self.assertEqual(r["state"], 13_300)
        self.assertEqual(r["niit"], 3_800)
        self.assertEqual(r["total"], 37_100)


class TestNiitAndFtc(unittest.TestCase):
    def test_niit_threshold_straddle(self):
        # single, W-2 190,000 + NII 30,000 -> MAGI 220,000, excess 20,000:
        # 3.8% x min(30,000, 20,000) = 760.
        r = est(prof(w2_income=190_000.0),
                income={**NO_INCOME, "interest": 30_000.0})
        self.assertEqual(r["niit"], 760)

    def test_foreign_withholding_credits_approximately(self):
        r = est(income={**NO_INCOME, "dividends": 10_000.0,
                        "withholding": -250.0})
        self.assertEqual(r["ftc"], 250)
        self.assertEqual(r["total"],
                         r["federal"] + r["state"] + r["niit"] - 250)


class TestSimLegs(unittest.TestCase):
    def test_sim_legs_aggregate_into_terms(self):
        via_sim = est(sim=[{"gl": 1_000.0, "term": "short"},
                           {"gl": -400.0, "term": "long"},
                           {"gl": 2_500.0, "term": "long"}])
        via_realized = est(realized={"short": 1_000.0, "long": 2_100.0,
                                     "unknown": 0.0})
        self.assertEqual(via_sim["total"], via_realized["total"])


class TestValidation(unittest.TestCase):
    def test_unsupported_filing_status_raises(self):
        with self.assertRaises(ValueError):
            est(prof(filing_status="married_separate"))

    def test_non_ca_state_raises(self):
        with self.assertRaises(ValueError):
            est(prof(state="NY"))

    def test_assumptions_list_is_nonempty_strings(self):
        r = est()
        self.assertTrue(r["assumptions"])
        self.assertTrue(all(isinstance(a, str) and a for a in r["assumptions"]))


class TestTreasuryMatcher(unittest.TestCase):
    def test_matches_structurally_never_by_ticker_list(self):
        self.assertTrue(is_treasury_income("912797GL5", ""))
        self.assertTrue(is_treasury_income("91282CJK8", ""))
        self.assertTrue(is_treasury_income(
            "", "UNITED STATES TREASURY BILL DUE 09/15/2026"))
        self.assertTrue(is_treasury_income("", "us treasury note 4.125%"))
        self.assertFalse(is_treasury_income("SPY", "SPDR S&P 500 ETF"))
        self.assertFalse(is_treasury_income("912", ""))       # not 9 chars
        self.assertFalse(is_treasury_income("", ""))


if __name__ == "__main__":
    unittest.main()
