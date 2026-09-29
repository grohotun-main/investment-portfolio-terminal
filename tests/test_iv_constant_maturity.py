"""Tests for parsers/iv_constant_maturity.py — the pure constant-maturity
ATM IV interpolation feeding the IV-percentile gauge.

One public surface:
  * `constant_maturity_iv(points, target_days=90) -> (iv, quality)` —
    interpolate a single day's option term structure to a fixed maturity
    in TOTAL VARIANCE (sigma^2 * T, additive in time).

Quality taxonomy:
  * "interp" — target bracketed by two expiries (or hit exactly).
  * "approx" — one-sided; nearest single expiry used (real, flagged).
  * "none"   — no usable points; iv is NaN.

All hand-computed expected values below use T = dte/365 and
sigma_target = sqrt( w_target / T_target ), where total variance
w = sigma^2 * T is interpolated linearly in DTE.
"""
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from iv_constant_maturity import (  # noqa: E402
    CM_CSV_COLS,
    QUALITY_APPROX,
    QUALITY_INTERP,
    QUALITY_NONE,
    constant_maturity_iv,
    derive_cm_history,
)


class TestBracketedInterpolation(unittest.TestCase):
    """The good case: two expiries straddle the target → interpolate."""

    def test_matches_hand_calc_77_105_to_90(self):
        # brackets 22% @ 77d, 24% @ 105d → CM-90.
        #   w_lo = .22^2 * 77/365  = .01021041
        #   w_hi = .24^2 * 105/365 = .01656986
        #   frac = (90-77)/(105-77) = 13/28 = .46428571
        #   w_t  = .01316301 ; T_t = 90/365 = .24657534
        #   sigma = sqrt(.01316301/.24657534) = .231048
        iv, q = constant_maturity_iv([(77, 0.22), (105, 0.24)], target_days=90)
        self.assertEqual(q, QUALITY_INTERP)
        self.assertAlmostEqual(iv, 0.231048, delta=1e-4)

    def test_picks_tightest_bracket_from_four_points(self):
        # {20:.30, 50:.26, 80:.23, 110:.21}, target 90 → bracket 80/110, not
        # 50/110. Hand-calc with the 80/110 pair:
        #   w_lo = .23^2 * 80/365  = .01159452
        #   w_hi = .21^2 * 110/365 = .01329041
        #   frac = (90-80)/(110-80) = 1/3
        #   w_t  = .01215982 ; T_t = .24657534
        #   sigma = sqrt(.01215982/.24657534) = .222069
        pts = [(20, 0.30), (50, 0.26), (80, 0.23), (110, 0.21)]
        iv, q = constant_maturity_iv(pts, target_days=90)
        self.assertEqual(q, QUALITY_INTERP)
        self.assertAlmostEqual(iv, 0.222069, delta=1e-4)

    def test_distinguishes_variance_from_linear_vol_interp(self):
        # A naive linear-in-vol interp of the 77/105 case gives
        #   .22 + .46428571*(.24-.22) = .229286 — materially different from
        # the correct .231048. This pins the variance formula.
        iv, _ = constant_maturity_iv([(77, 0.22), (105, 0.24)], target_days=90)
        self.assertNotAlmostEqual(iv, 0.229286, delta=5e-4)

    def test_exact_target_hit_returns_that_point(self):
        # An expiry sitting exactly on the target is a true CM reading and
        # must not divide-by-zero on the bracket width.
        iv, q = constant_maturity_iv([(60, 0.25), (90, 0.27), (120, 0.28)],
                                     target_days=90)
        self.assertEqual(q, QUALITY_INTERP)
        self.assertAlmostEqual(iv, 0.27, places=10)

    def test_default_target_is_90(self):
        # Same inputs as the 77/105 case but without passing target_days.
        iv, q = constant_maturity_iv([(77, 0.22), (105, 0.24)])
        self.assertEqual(q, QUALITY_INTERP)
        self.assertAlmostEqual(iv, 0.231048, delta=1e-4)


class TestOneSidedApprox(unittest.TestCase):
    """All expiries on one side of target → nearest single expiry, flagged."""

    def test_all_below_target_uses_largest_dte(self):
        # target 90, all expiries shorter → nearest is the 80d point.
        iv, q = constant_maturity_iv([(20, 0.30), (50, 0.26), (80, 0.23)],
                                     target_days=90)
        self.assertEqual(q, QUALITY_APPROX)
        self.assertAlmostEqual(iv, 0.23, places=10)

    def test_all_above_target_uses_smallest_dte(self):
        # target 90, all expiries longer → nearest is the 110d point.
        iv, q = constant_maturity_iv([(110, 0.21), (140, 0.20), (170, 0.19)],
                                     target_days=90)
        self.assertEqual(q, QUALITY_APPROX)
        self.assertAlmostEqual(iv, 0.21, places=10)


class TestNoUsableData(unittest.TestCase):
    """Degenerate inputs → (NaN, "none"). The percentile gauge NaN-skips."""

    def test_empty_points(self):
        iv, q = constant_maturity_iv([], target_days=90)
        self.assertEqual(q, QUALITY_NONE)
        self.assertTrue(math.isnan(iv))

    def test_all_nan_iv_filtered_out(self):
        iv, q = constant_maturity_iv([(77, float("nan")), (105, float("nan"))],
                                     target_days=90)
        self.assertEqual(q, QUALITY_NONE)
        self.assertTrue(math.isnan(iv))

    def test_nonpositive_dte_dropped_before_bracketing(self):
        # The 0-DTE point is dropped; only one valid point (105) remains →
        # one-sided above 90 → approx on 105, not an interp using dte=0.
        iv, q = constant_maturity_iv([(0, 0.50), (105, 0.24)], target_days=90)
        self.assertEqual(q, QUALITY_APPROX)
        self.assertAlmostEqual(iv, 0.24, places=10)

    def test_nonpositive_iv_dropped(self):
        iv, q = constant_maturity_iv([(77, 0.0), (105, -0.1)], target_days=90)
        self.assertEqual(q, QUALITY_NONE)
        self.assertTrue(math.isnan(iv))


class TestDeriveCmHistory(unittest.TestCase):
    """Frame projection: raw term history (several expiries per day) → one
    constant-maturity row per (date, underlying)."""

    def _term_row(self, d, u, expiry, dte, iv, fetched="2026-02-01T00:00:00"):
        return {"date": d, "underlying": u, "expiry": expiry,
                "dte_days": dte, "atm_strike": 500.0, "spot": 500.0,
                "close": 5.0, "atm_iv": iv, "fetched_at": fetched}

    def test_one_row_per_date_underlying_with_cm_value(self):
        # 77/105 bracket → CM-90 = 0.231048 (same hand-calc as the scalar test).
        term = pd.DataFrame([
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, 0.22),
            self._term_row("2026-01-02", "SPY", "2026-04-17", 105, 0.24),
        ])
        cm = derive_cm_history(term, target_days=90)
        self.assertEqual(len(cm), 1)
        self.assertEqual(list(cm.columns), CM_CSV_COLS)
        row = cm.iloc[0]
        self.assertEqual(row["underlying"], "SPY")
        self.assertAlmostEqual(float(row["atm_iv"]), 0.231048, delta=1e-4)
        self.assertEqual(row["quality"], QUALITY_INTERP)
        self.assertEqual(int(row["target_days"]), 90)

    def test_groups_by_date_and_underlying(self):
        term = pd.DataFrame([
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, 0.22),
            self._term_row("2026-01-02", "SPY", "2026-04-17", 105, 0.24),
            self._term_row("2026-01-02", "NVDA", "2026-03-20", 77, 0.50),
            self._term_row("2026-01-02", "NVDA", "2026-04-17", 105, 0.52),
            self._term_row("2026-01-05", "SPY", "2026-03-20", 74, 0.23),
            self._term_row("2026-01-05", "SPY", "2026-04-17", 102, 0.25),
        ])
        cm = derive_cm_history(term, target_days=90)
        self.assertEqual(len(cm), 3)  # (1/2,SPY) (1/2,NVDA) (1/5,SPY)
        keys = set(zip(cm["date"].astype(str), cm["underlying"]))
        self.assertEqual(
            keys,
            {("2026-01-02", "SPY"), ("2026-01-02", "NVDA"),
             ("2026-01-05", "SPY")},
        )

    def test_output_sorted_by_underlying_then_date(self):
        term = pd.DataFrame([
            self._term_row("2026-01-05", "SPY", "2026-03-20", 74, 0.23),
            self._term_row("2026-01-05", "SPY", "2026-04-17", 102, 0.25),
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, 0.22),
            self._term_row("2026-01-02", "SPY", "2026-04-17", 105, 0.24),
        ])
        cm = derive_cm_history(term, target_days=90)
        self.assertEqual(list(cm["date"].astype(str)),
                         ["2026-01-02", "2026-01-05"])

    def test_fetched_at_is_max_within_group(self):
        # A day re-touched by a later fetch keeps the newest stamp.
        term = pd.DataFrame([
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, 0.22,
                           fetched="2026-02-01T00:00:00"),
            self._term_row("2026-01-02", "SPY", "2026-04-17", 105, 0.24,
                           fetched="2026-02-05T00:00:00"),
        ])
        cm = derive_cm_history(term, target_days=90)
        self.assertEqual(cm.iloc[0]["fetched_at"], "2026-02-05T00:00:00")

    def test_one_sided_day_flagged_approx(self):
        # Only short-dated expiries available → approx on nearest, not interp.
        term = pd.DataFrame([
            self._term_row("2026-01-02", "SPY", "2026-02-20", 49, 0.22),
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, 0.23),
        ])
        cm = derive_cm_history(term, target_days=90)
        self.assertEqual(cm.iloc[0]["quality"], QUALITY_APPROX)
        self.assertAlmostEqual(float(cm.iloc[0]["atm_iv"]), 0.23, places=6)

    def test_dead_day_emitted_with_nan_and_none(self):
        # A day where every close failed to invert → NaN row, quality "none".
        # Emitted (not dropped) so the record is honest; the percentile gauge
        # already NaN-skips it.
        term = pd.DataFrame([
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, float("nan")),
            self._term_row("2026-01-02", "SPY", "2026-04-17", 105, float("nan")),
        ])
        cm = derive_cm_history(term, target_days=90)
        self.assertEqual(len(cm), 1)
        self.assertTrue(math.isnan(float(cm.iloc[0]["atm_iv"])))
        self.assertEqual(cm.iloc[0]["quality"], QUALITY_NONE)

    def test_empty_term_frame_returns_empty_with_columns(self):
        cm = derive_cm_history(pd.DataFrame(columns=[
            "date", "underlying", "expiry", "dte_days", "atm_strike",
            "spot", "close", "atm_iv", "fetched_at"]), target_days=90)
        self.assertTrue(cm.empty)
        self.assertEqual(list(cm.columns), CM_CSV_COLS)

    def test_target_days_column_reflects_argument(self):
        term = pd.DataFrame([
            self._term_row("2026-01-02", "SPY", "2026-02-20", 49, 0.26),
            self._term_row("2026-01-02", "SPY", "2026-03-20", 77, 0.24),
            self._term_row("2026-01-02", "SPY", "2026-04-17", 105, 0.22),
        ])
        cm = derive_cm_history(term, target_days=60)
        self.assertEqual(int(cm.iloc[0]["target_days"]), 60)
        self.assertEqual(cm.iloc[0]["quality"], QUALITY_INTERP)


if __name__ == "__main__":
    unittest.main()
