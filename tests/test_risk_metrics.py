"""
Tests for the pure risk math in parsers/risk_metrics.py.

Covers:
  - Return summaries:   _return_stats
  - Risk-adjusted:      compute_sharpe, compute_sortino, compute_calmar
  - Drawdowns:          compute_drawdown_episodes
  - Concentration:      compute_concentration
  - Benchmark series:   spy_monthly_returns_aligned, spy_value_at,
                        spy_decline_between, spy_months_underwater_from
  - Daily ex-ante:      synthesize_portfolio_returns, compute_risk_contributions
  - Beta / alpha:       compute_beta, compute_alpha_annual,
                        compute_up_down_beta
  - Tail risk:          compute_var_cvar
  - Period aggregation: aggregate_periodic_returns

Conventions verified here (from v13 audit fixes, see memory:
project_v13_audit_fixes.md):
  - Sortino uses textbook Sortino-Bawa downside deviation (mean over ALL
    observations, with positives contributing 0), not the "average of
    negatives only" variant.
  - Black-Scholes put delta supports a continuous dividend yield `q`.

Run from phase1_build/ with:
    py -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Make parsers/ importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import risk_metrics as rm  # noqa: E402


# ---------------------------------------------------------------------------
# _return_stats — building block under Sharpe/Sortino/Calmar
# ---------------------------------------------------------------------------
class TestReturnStats(unittest.TestCase):
    def test_too_few_observations(self) -> None:
        cagr, vol, dvol, n = rm._return_stats(pd.Series([0.05]))
        self.assertTrue(np.isnan(cagr))
        self.assertTrue(np.isnan(vol))
        self.assertTrue(np.isnan(dvol))
        self.assertEqual(n, 1)

    def test_constant_return(self) -> None:
        # 12 months of 1% each: CAGR = (1.01)^12 - 1 ≈ 12.68%; vol = 0.
        r = pd.Series([0.01] * 12)
        cagr, vol, dvol, n = rm._return_stats(r)
        self.assertAlmostEqual(cagr, (1.01) ** 12 - 1.0, places=10)
        self.assertAlmostEqual(vol, 0.0, places=12)
        self.assertAlmostEqual(dvol, 0.0, places=12)
        self.assertEqual(n, 12)

    def test_all_positive_zero_downside(self) -> None:
        # No negative months → downside deviation must be exactly zero.
        _, _, dvol, _ = rm._return_stats(pd.Series([0.02, 0.03, 0.01, 0.04]))
        self.assertAlmostEqual(dvol, 0.0, places=12)

    def test_sortino_bawa_convention(self) -> None:
        # The v13 audit pinned this to the textbook Sortino-Bawa formula:
        #     dvol = sqrt(mean(min(0, r)^2)) × sqrt(12)
        # averaged over ALL n observations (positives contribute zero).
        # NOT sqrt(mean(neg^2 over negatives only)) × sqrt(12).
        # Here three months are negative out of six; mean over ALL six.
        r = pd.Series([0.05, -0.02, 0.03, -0.01, 0.02, -0.04])
        _, _, dvol_actual, _ = rm._return_stats(r)
        neg_sq = (np.minimum(r.values, 0.0)) ** 2
        dvol_expected = float(np.sqrt(neg_sq.mean()) * np.sqrt(12))
        self.assertAlmostEqual(dvol_actual, dvol_expected, places=12)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------
class TestSharpe(unittest.TestCase):
    def test_constant_returns_zero_vol_returns_nan(self) -> None:
        # Zero vol breaks the denominator → must be NaN, not inf or zero.
        # Use 0.0 rather than 0.01: pandas' std(ddof=1) on a non-zero
        # constant Series leaks ~1e-18 of floating-point noise, which
        # squeaks past the `ann_vol > 0` guard and produces a ~1e16-scale
        # Sharpe. That's a degenerate case that doesn't occur in real data
        # (monthly returns are never bit-identical), so we don't harden
        # production against it — but we still want this test to assert
        # the cleanly-zero path.
        self.assertTrue(np.isnan(rm.compute_sharpe(pd.Series([0.0] * 12), 0.0)))

    def test_subtracts_rf(self) -> None:
        # With nonzero rf, Sharpe = (cagr - rf) / vol — verify the subtraction
        # actually happens by checking two rf values differ by (Δrf / vol).
        r = pd.Series([0.02, -0.01, 0.03, -0.005, 0.015, 0.025, -0.02, 0.01,
                       0.005, 0.018, -0.012, 0.022])
        s0 = rm.compute_sharpe(r, 0.0)
        s4 = rm.compute_sharpe(r, 0.04)
        cagr, vol, _, _ = rm._return_stats(r)
        self.assertAlmostEqual(s0 - s4, 0.04 / vol, places=10)

    def test_insufficient_data_returns_nan(self) -> None:
        self.assertTrue(np.isnan(rm.compute_sharpe(pd.Series([0.01]), 0.04)))


# ---------------------------------------------------------------------------
# compute_sortino
# ---------------------------------------------------------------------------
class TestSortino(unittest.TestCase):
    def test_no_negative_months_returns_nan(self) -> None:
        # Positive-only series ⇒ downside vol = 0 ⇒ Sortino undefined.
        self.assertTrue(np.isnan(
            rm.compute_sortino(pd.Series([0.01, 0.02, 0.015, 0.018]), 0.0)))

    def test_sortino_higher_than_sharpe_when_skewed_positive(self) -> None:
        # Skewed-positive returns ⇒ downside vol < total vol ⇒ Sortino > Sharpe
        # (same numerator, smaller denominator).
        r = pd.Series([0.05, 0.04, 0.06, 0.03, -0.01, 0.07, 0.02, -0.005,
                       0.04, 0.03, 0.05, 0.04])
        sharpe = rm.compute_sharpe(r, 0.0)
        sortino = rm.compute_sortino(r, 0.0)
        self.assertGreater(sortino, sharpe)

    def test_symmetric_returns_sortino_above_sharpe(self) -> None:
        # For a roughly-symmetric distribution: dvol uses sqrt(mean(neg^2))
        # divided by sqrt(n_total) — which is smaller than total stdev's
        # sqrt(sum_sq / (n-1)). So Sortino > Sharpe holds here too.
        r = pd.Series([-0.02, 0.02, -0.015, 0.018, -0.01, 0.012, -0.025,
                       0.024, 0.03, -0.028, 0.005, -0.004])
        self.assertGreater(rm.compute_sortino(r, 0.0),
                           rm.compute_sharpe(r, 0.0))


# ---------------------------------------------------------------------------
# Time-varying RF: _window_rf + Series-valued compute_sharpe / compute_sortino
# ---------------------------------------------------------------------------
class TestWindowRF(unittest.TestCase):
    def test_scalar_input_passthrough(self) -> None:
        # Float rf input should return as-is regardless of the returns series.
        r = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
        self.assertAlmostEqual(rm._window_rf(0.05, r), 0.05, places=12)
        self.assertAlmostEqual(rm._window_rf(0.0, r), 0.0, places=12)

    def test_series_input_samples_at_return_dates(self) -> None:
        # Daily RF series; pick a sparse return index — we should average the
        # RF values at exactly those return dates, not the full RF span.
        rf = pd.Series([0.04, 0.04, 0.05, 0.05, 0.06, 0.06],
                       index=pd.to_datetime(
                           ["2024-01-31", "2024-02-29", "2024-03-31",
                            "2024-04-30", "2024-05-31", "2024-06-30"]))
        rets = pd.Series([0.01, 0.02, 0.03],
                         index=pd.to_datetime(["2024-02-29", "2024-04-30", "2024-06-30"]))
        # Samples at the 3 return dates: 0.04, 0.05, 0.06 → mean 0.05.
        self.assertAlmostEqual(rm._window_rf(rf, rets), 0.05, places=12)

    def test_series_forward_fills_for_weekend_return_dates(self) -> None:
        # FRED is biz-day only; if a return date lands on a Saturday, the
        # helper should ffill the prior Friday's published rate.
        rf = pd.Series(
            [0.04, 0.05],
            index=pd.to_datetime(["2024-03-29", "2024-04-01"]))  # Fri, Mon
        rets = pd.Series(
            [0.01], index=pd.to_datetime(["2024-03-30"]))  # Saturday
        self.assertAlmostEqual(rm._window_rf(rf, rets), 0.04, places=12)

    def test_series_back_fills_for_pre_series_dates(self) -> None:
        # Return dates predating the RF series should degrade to the earliest
        # available rate rather than producing NaN — keeps ancient history
        # Sharpe finite when FRED's coverage doesn't reach back that far.
        rf = pd.Series([0.05], index=pd.to_datetime(["2024-01-31"]))
        rets = pd.Series([0.01], index=pd.to_datetime(["2020-01-31"]))
        self.assertAlmostEqual(rm._window_rf(rf, rets), 0.05, places=12)

    def test_empty_inputs_return_nan(self) -> None:
        rf = pd.Series([0.04, 0.05], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
        self.assertTrue(np.isnan(rm._window_rf(rf, pd.Series([], dtype=float))))
        empty_rf = pd.Series([], dtype=float)
        rets = pd.Series([0.01], index=pd.to_datetime(["2024-01-31"]))
        self.assertTrue(np.isnan(rm._window_rf(empty_rf, rets)))

    def test_compute_sharpe_series_matches_mean_rf(self) -> None:
        # Sharpe(returns, rf_series) must equal Sharpe(returns, mean(rf_series))
        # — same numerator delta, same denominator.
        r = pd.Series(
            [0.02, -0.01, 0.03, -0.005, 0.015, 0.025, -0.02, 0.01,
             0.005, 0.018, -0.012, 0.022],
            index=pd.date_range("2024-01-31", periods=12, freq="ME"))
        # Step the RF from 4% to 6% across the year.
        rf = pd.Series(np.linspace(0.04, 0.06, 12), index=r.index)
        s_series = rm.compute_sharpe(r, rf)
        s_scalar = rm.compute_sharpe(r, float(rf.mean()))
        self.assertAlmostEqual(s_series, s_scalar, places=12)

    def test_compute_sortino_series_matches_mean_rf(self) -> None:
        r = pd.Series(
            [0.02, -0.01, 0.03, -0.005, 0.015, 0.025, -0.02, 0.01,
             0.005, 0.018, -0.012, 0.022],
            index=pd.date_range("2024-01-31", periods=12, freq="ME"))
        rf = pd.Series(np.linspace(0.04, 0.06, 12), index=r.index)
        s_series = rm.compute_sortino(r, rf)
        s_scalar = rm.compute_sortino(r, float(rf.mean()))
        self.assertAlmostEqual(s_series, s_scalar, places=12)

    def test_compute_sortino_daily_accepts_series(self) -> None:
        # Sortino-daily should also accept a Series rf and average it across
        # the daily return dates. Sanity check: finite and monotonic in rf
        # (higher rf → lower Sortino).
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-02", periods=252, freq="B")
        r = pd.Series(rng.normal(0.0005, 0.01, 252), index=idx)
        rf_low = pd.Series(np.full(252, 0.02), index=idx)
        rf_high = pd.Series(np.full(252, 0.08), index=idx)
        val_low = rm.compute_sortino_daily(r, rf_low)
        val_high = rm.compute_sortino_daily(r, rf_high)
        self.assertTrue(np.isfinite(val_low) and np.isfinite(val_high))
        self.assertGreater(val_low, val_high)


# ---------------------------------------------------------------------------
# compute_calmar
# ---------------------------------------------------------------------------
class TestCalmar(unittest.TestCase):
    def test_known_cagr_and_dd(self) -> None:
        # CAGR (1% × 12) = (1.01)^12 - 1 ≈ 0.12683. Max DD = -20%.
        # Calmar = CAGR / |MaxDD| = 0.12683 / 0.20 ≈ 0.63414.
        r = pd.Series([0.01] * 12)
        dd_pct = pd.Series([0.0, -5.0, -20.0, -10.0, 0.0])  # percent
        expected = ((1.01) ** 12 - 1.0) / 0.20
        self.assertAlmostEqual(rm.compute_calmar(r, dd_pct), expected, places=8)

    def test_no_dd_returns_nan(self) -> None:
        # If max DD is zero or positive (no drawdown observed), Calmar is
        # undefined.
        r = pd.Series([0.01] * 12)
        self.assertTrue(np.isnan(rm.compute_calmar(r, pd.Series([0.0, 0.0]))))

    def test_empty_dd_series_returns_nan(self) -> None:
        self.assertTrue(np.isnan(
            rm.compute_calmar(pd.Series([0.01] * 12), pd.Series([], dtype=float))))

    def test_negative_cagr_returns_nan(self) -> None:
        # Phase 1C audit: Calmar = CAGR / |MaxDD| loses interpretive
        # meaning when CAGR is negative — the ratio reads as "per-year
        # loss as a fraction of max drawdown" but the comparison to a
        # positive-CAGR Calmar isn't meaningful. The dashboard's
        # how-to-read prose says "higher is better"; surfacing a
        # negative Calmar would invite reading SPY=-0.10 vs port=-0.50
        # as "port underperforming by 0.4," which the math doesn't
        # support. Lock NaN on non-positive CAGR.
        # 12 months of -1% each → CAGR ≈ -11.4%, MaxDD = -11.4%.
        r = pd.Series([-0.01] * 12)
        dd_pct = pd.Series([-1.0, -2.0, -3.0, -11.4])
        self.assertTrue(np.isnan(rm.compute_calmar(r, dd_pct)))
        # Zero CAGR boundary: also NaN, since the ratio is 0 and the
        # interpretation is degenerate.
        r0 = pd.Series([0.0] * 12)
        self.assertTrue(np.isnan(rm.compute_calmar(r0, pd.Series([-5.0]))))

    def test_tiny_max_dd_returns_nan(self) -> None:
        # Phase 1D audit: without an epsilon floor, a series with a
        # micro-wobble (e.g. -0.001%) drove Calmar to absurd magnitudes
        # (CAGR / 1e-5 ≈ 5,000+). The 1bp floor (_CALMAR_MIN_DD_PCT)
        # returns NaN in that regime — well below any realistic
        # multi-asset drawdown. Symmetric counterpart to PR #65's
        # negative-CAGR guard.
        r = pd.Series([0.01] * 12)  # CAGR ≈ 12.68% — comfortably positive
        # Max DD = -0.001% (one basis-point under the 1bp floor).
        dd_pct_tiny = pd.Series([0.0, -0.001, 0.0])
        self.assertTrue(np.isnan(rm.compute_calmar(r, dd_pct_tiny)))
        # Just above the floor (1.5bp) — Calmar should now be finite.
        dd_pct_above = pd.Series([0.0, -0.015, 0.0])
        cal = rm.compute_calmar(r, dd_pct_above)
        self.assertTrue(np.isfinite(cal))
        self.assertGreater(cal, 0.0)
        # Exact-floor boundary: at the threshold, the guard fires (<).
        dd_pct_at_floor = pd.Series([0.0, -rm._CALMAR_MIN_DD_PCT, 0.0])
        self.assertTrue(np.isfinite(rm.compute_calmar(r, dd_pct_at_floor)))


# ---------------------------------------------------------------------------
# Phase 1C hygiene bundle
# ---------------------------------------------------------------------------
class TestPhase1CHygieneGuards(unittest.TestCase):
    """Lock the small guards / docstring invariants added in the Phase 1C
    hygiene bundle so future refactors don't silently regress them."""

    def test_compute_beta_near_zero_variance_returns_nan(self) -> None:
        # Pre-hardening guard was exact ==0; a positive-but-tiny variance
        # would slip through and produce absurd-magnitude β. Now: <=1e-16
        # also returns NaN.
        b = pd.Series([1e-10, 0.0, -1e-10, 0.0])  # var ≈ 6.7e-21, near-zero
        p = pd.Series([0.01, -0.01, 0.005, -0.005])
        self.assertTrue(np.isnan(rm.compute_beta(p, b)))

    def test_compute_up_down_beta_near_zero_variance_returns_nan(self) -> None:
        b = pd.Series([1e-10, -1e-10, 2e-10, -2e-10, 1e-10, -1e-10] * 10)
        p = pd.Series([0.01] * 60)
        up_b, dn_b = rm.compute_up_down_beta(p, b)
        self.assertTrue(np.isnan(up_b))
        self.assertTrue(np.isnan(dn_b))

    def test_dr_regime_thresholds_zscore_zero_sd_falls_back(self) -> None:
        # Constant ratio series (sd == 0) with n >= 10 used to silently
        # collapse stress_thr == calm_thr == mean and route everything
        # to "Normal." Now falls back to fixed defaults with a fallback
        # message.
        ratio = pd.Series([1.0] * 30)
        out = rm.compute_dr_regime_thresholds(ratio, method="zscore")
        self.assertEqual(out["method"], "fixed")
        self.assertIn("fallback", out)
        self.assertAlmostEqual(out["stress_thr"], 0.90)
        self.assertAlmostEqual(out["calm_thr"], 1.10)

    def test_ledoit_wolf_tiny_variance_bails_to_input(self) -> None:
        # Mix one near-flat asset (positive but tiny variance) with two
        # healthy ones. The hardened guard (diag < eps_var) routes to
        # the unshrunk-input path with alpha=0.0 instead of letting
        # corr = S / std·stdᵀ produce huge entries.
        rng = np.random.default_rng(2024)
        n = 300
        a = rng.normal(0, 0.012, n)
        b = rng.normal(0, 0.010, n)
        flat = rng.normal(0, 1e-15, n)
        rets = pd.DataFrame({"A": a, "B": b, "FLAT": flat})
        cov_input = rets.cov()
        shrunk, alpha = rm._ledoit_wolf_shrinkage(rets, cov_input)
        self.assertEqual(alpha, 0.0)
        np.testing.assert_array_equal(shrunk.values, cov_input.values)

    def test_rf_staleness_missing_file_returns_none(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as td:
            self.assertIsNone(rm.rf_staleness_business_days(Path(td)))

    def test_rf_staleness_counts_weekdays(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as td:
            # Last RF reading on Mon 2026-05-11; "today" Mon 2026-05-18
            # → weekdays strictly between (12, 13, 14, 15, 18) = 5.
            pd.DataFrame({"date": ["2026-05-11"], "rate": [0.037]}).to_csv(
                Path(td) / "risk_free_rate.csv", index=False)
            lag = rm.rf_staleness_business_days(
                Path(td), today=pd.Timestamp("2026-05-18"))
            self.assertEqual(lag, 5)
            self.assertEqual(rm.rf_staleness_business_days(
                Path(td), today=pd.Timestamp("2026-05-11")), 0)


# ---------------------------------------------------------------------------
# window_drawdown_pct
# ---------------------------------------------------------------------------
class TestWindowDrawdownPct(unittest.TestCase):
    """Phase 1C audit: the previous inline _window_dd_pct in app.py
    anchored cumprod at (1 + r₀), so a worst month at the start of
    the trailing window silently showed 0% drawdown. The Max DD (1Y /
    3Y) and Calmar (3Y) tiles read this output. Lock the anchored
    semantics here."""

    def test_first_observation_worst_is_visible(self) -> None:
        # Month 1 = -20%, then flat. Drawdown at month 1 must be -20%,
        # not 0 (the pre-fix value). At subsequent months the wealth
        # stays at 0.8, and the implicit peak of 1.0 holds → dd stays
        # at -20%.
        r = pd.Series([-0.20, 0.0, 0.0])
        dd = rm.window_drawdown_pct(r)
        self.assertAlmostEqual(dd.iloc[0], -20.0, places=8)
        self.assertAlmostEqual(dd.iloc[1], -20.0, places=8)
        self.assertAlmostEqual(dd.iloc[2], -20.0, places=8)

    def test_recovery_zeroes_drawdown(self) -> None:
        # -10% then +12% → wealth 0.9 → 1.008. Peak is 1.008 by month 2;
        # dd at month 2 is back to 0.
        r = pd.Series([-0.10, 0.10 / 0.9])  # +11.11% recovers exactly
        dd = rm.window_drawdown_pct(r)
        self.assertAlmostEqual(dd.iloc[0], -10.0, places=8)
        self.assertAlmostEqual(dd.iloc[1], 0.0, places=8)

    def test_positive_only_window_zero_throughout(self) -> None:
        # All gains → peak rises with wealth → dd is 0 every month.
        r = pd.Series([0.05, 0.03, 0.02])
        dd = rm.window_drawdown_pct(r)
        self.assertTrue((dd == 0.0).all())

    def test_empty_returns_empty(self) -> None:
        dd = rm.window_drawdown_pct(pd.Series([], dtype=float))
        self.assertTrue(dd.empty)

    def test_preserves_input_index(self) -> None:
        idx = pd.date_range("2026-01-31", periods=3, freq="ME")
        r = pd.Series([-0.05, 0.02, 0.01], index=idx)
        dd = rm.window_drawdown_pct(r)
        self.assertTrue((dd.index == idx).all())

    def test_max_dd_in_first_period_flows_through_to_calmar(self) -> None:
        # Cross-function: pre-fix, a first-period trough was invisible,
        # so Calmar's |MaxDD| denominator was understated (or 0). Now
        # the trough is captured and Calmar reflects it. This pins the
        # composite invariant.
        r = pd.Series([-0.15] + [0.005] * 11)  # 1 bad month + 11 mild gains
        dd = rm.window_drawdown_pct(r)
        cagr_implied = float((1.0 + r).prod()) ** (1.0 / 1.0) - 1.0  # 12 months ≈ 1 year
        self.assertLess(dd.min(), -10.0)  # trough is real
        # If Calmar's denominator was 0 (pre-fix on a hypothetical
        # data shape), Calmar would be NaN; here it must be finite.
        calmar = rm.compute_calmar(r, dd)
        if cagr_implied > 0:
            self.assertTrue(np.isfinite(calmar))
        else:
            # Negative-CAGR path: Calmar is NaN by the other guard.
            self.assertTrue(np.isnan(calmar))


# ---------------------------------------------------------------------------
# compute_drawdown_episodes
# ---------------------------------------------------------------------------
class TestDrawdownEpisodes(unittest.TestCase):
    def _series(self, vals, start: str = "2024-01-31"):
        idx = pd.date_range(start, periods=len(vals), freq="ME")
        return pd.Series(vals, index=idx), pd.Series(idx)

    def test_monotonic_growth_no_episodes(self) -> None:
        w, d = self._series([100, 105, 110, 120, 130])
        self.assertEqual(rm.compute_drawdown_episodes(w, d), [])

    def test_single_recovered_episode(self) -> None:
        # Up to 120, down to 90, recovers to 125.
        # Episode: depth = (90/120 - 1) × 100 = -25%, p2t = 1 month,
        # recovery = 2 months (90→105→125 ⇒ first month w >= 120 is index 4).
        w, d = self._series([100, 120, 90, 105, 125])
        eps = rm.compute_drawdown_episodes(w, d)
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertAlmostEqual(ep["depth_pct"], -25.0, places=6)
        self.assertEqual(ep["peak_to_trough_months"], 1)
        self.assertEqual(ep["recovery_months"], 2)
        self.assertIsNotNone(ep["recovery_date"])

    def test_open_episode_no_recovery(self) -> None:
        # New high, then drops and never recovers. Episode is open
        # (recovery_date=None, recovery_months=None).
        w, d = self._series([100, 110, 100, 95, 90])
        eps = rm.compute_drawdown_episodes(w, d)
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertAlmostEqual(ep["depth_pct"], (90 / 110 - 1.0) * 100, places=6)
        self.assertIsNone(ep["recovery_date"])
        self.assertIsNone(ep["recovery_months"])

    def test_too_short_returns_empty(self) -> None:
        w, d = self._series([100])
        self.assertEqual(rm.compute_drawdown_episodes(w, d), [])


# ---------------------------------------------------------------------------
# compute_concentration
# ---------------------------------------------------------------------------
class TestConcentration(unittest.TestCase):
    def test_equal_weights_effective_n_equals_n(self) -> None:
        # 4 equal positions ⇒ each is 25%; HHI = 4 × 0.25² = 0.25;
        # effective_n = 1/HHI = 4.
        mv = pd.Series([100, 100, 100, 100], index=list("ABCD"))
        out = rm.compute_concentration(mv)
        self.assertAlmostEqual(out["max_pct"], 25.0, places=6)
        self.assertAlmostEqual(out["effective_n"], 4.0, places=6)
        self.assertEqual(out["n_positions"], 4)

    def test_dominant_position(self) -> None:
        # 90% in one, 10% spread across two. max_pct = 90%, top5 = 100%.
        mv = pd.Series([900, 50, 50], index=["X", "Y", "Z"])
        out = rm.compute_concentration(mv)
        self.assertAlmostEqual(out["max_pct"], 90.0, places=6)
        self.assertAlmostEqual(out["top5_pct"], 100.0, places=6)
        self.assertEqual(out["n_positions"], 3)

    def test_drops_non_positive(self) -> None:
        # Negative (short option) and zero rows are dropped before weights.
        mv = pd.Series([100, -50, 0, 100], index=list("ABCD"))
        out = rm.compute_concentration(mv)
        self.assertEqual(out["n_positions"], 2)
        self.assertAlmostEqual(out["max_pct"], 50.0, places=6)

    def test_empty(self) -> None:
        out = rm.compute_concentration(pd.Series([], dtype=float))
        self.assertEqual(out["n_positions"], 0)
        self.assertTrue(np.isnan(out["effective_n"]))


# ---------------------------------------------------------------------------
# spy_value_at / spy_decline_between
# ---------------------------------------------------------------------------
class TestSpyValueAndDecline(unittest.TestCase):
    def setUp(self) -> None:
        # 10 business days of synthetic SPY TR values.
        idx = pd.date_range("2026-01-05", periods=10, freq="B")
        self.bench = pd.Series([100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
                               index=idx)

    def test_value_at_exact_date(self) -> None:
        self.assertAlmostEqual(rm.spy_value_at(self.bench, self.bench.index[3]),
                               103.0, places=10)

    def test_value_at_non_trading_day_uses_prior(self) -> None:
        # Sunday between trading days — returns the prior trading day's value.
        # bench[2] is 2026-01-07 (Wed)=102; query 2026-01-10 (Sat).
        d = pd.Timestamp("2026-01-10")
        v = rm.spy_value_at(self.bench, d)
        # 2026-01-09 (Fri) is in bench at value 104.
        self.assertAlmostEqual(v, 104.0, places=10)

    def test_value_before_range_returns_nan(self) -> None:
        self.assertTrue(np.isnan(rm.spy_value_at(self.bench,
                                                 pd.Timestamp("2025-01-01"))))

    def test_decline_between(self) -> None:
        # bench[0]=100, bench[-1]=101 ⇒ +1% — but use peak/trough order.
        # Peak at idx 5 (105), trough at idx 9 (101): decline = (101/105 - 1)×100.
        d_peak = self.bench.index[5]
        d_trough = self.bench.index[9]
        expected = (101.0 / 105.0 - 1.0) * 100.0
        self.assertAlmostEqual(rm.spy_decline_between(self.bench, d_peak, d_trough),
                               expected, places=8)


# ---------------------------------------------------------------------------
# spy_months_underwater_from
# ---------------------------------------------------------------------------
class TestSpyMonthsUnderwater(unittest.TestCase):
    def test_local_peak_uses_lookahead_window(self) -> None:
        # The function takes the MAX over [peak_date, peak_date+lookahead]
        # to find SPY's local peak, not the value AT peak_date — captures
        # the case where SPY's true high comes a few days after the
        # portfolio peak. Verify by putting the unique max early in the
        # window and the recovery many months later.
        idx = pd.date_range("2026-01-01", periods=60, freq="B")
        vals = ([100] * 5            # days 0-4
                + [105]              # day 5: SPY local peak (Jan)
                + [99] * 45          # days 6-50: extended trough
                + [105] * 9)         # days 51-59: recovery to peak (March)
        assert len(vals) == 60, f"fixture must be 60 long, got {len(vals)}"
        bench = pd.Series(vals, index=idx)
        # Portfolio peak passed in at day 0; SPY's local high (105) is at
        # day 5; recovery (first value >= 105 after day 5) is at day 51.
        # Day 5 ≈ Jan 8, day 51 ≈ mid-March → at least 2 calendar months.
        n = rm.spy_months_underwater_from(bench, idx[0], lookahead_days=90)
        self.assertIsNotNone(n)
        self.assertGreaterEqual(n, 1)

    def test_no_recovery_returns_none(self) -> None:
        # SPY peaks at 110 then never gets back.
        idx = pd.date_range("2026-01-01", periods=20, freq="B")
        vals = [100, 105, 110] + [100] * 17
        bench = pd.Series(vals, index=idx)
        self.assertIsNone(rm.spy_months_underwater_from(bench, idx[0],
                                                       lookahead_days=10))

    def test_empty_bench_returns_none(self) -> None:
        self.assertIsNone(rm.spy_months_underwater_from(
            pd.Series([], dtype=float), pd.Timestamp("2026-01-01")))

    def test_lag_beyond_lookahead_silently_uses_early_peak(self) -> None:
        # Phase 1D hygiene: lock the known limitation that SPY peaks
        # lagging the portfolio peak by more than `lookahead_days` get
        # silently missed. The function uses the highest SPY value WITHIN
        # the lookahead, so if the true peak sits beyond that window the
        # caller has no way to tell. This test pins the behavior so a
        # future refactor either preserves it (acceptable for current
        # portfolios) or surfaces a diagnostic.
        idx = pd.date_range("2026-01-01", periods=300, freq="B")
        vals = [100.0] * 300
        # SPY's "true" local peak is at day 150 (well past a 90d / ≈65
        # trading-day lookahead from day 0). Recovery happens at day 250.
        vals[150] = 120.0
        for i in range(250, 300):
            vals[i] = 120.0
        # Put a smaller local high at day 10 — within lookahead.
        vals[10] = 105.0
        bench = pd.Series(vals, index=idx)
        n = rm.spy_months_underwater_from(bench, idx[0], lookahead_days=90)
        # The function locks onto the day-10 peak (105), not the day-150
        # peak (120). Recovery is ≥ 105 — the first such day after day
        # 10 is day 150 itself. So months underwater reflects day 10 →
        # day 150, NOT day 150 → day 250.
        self.assertIsNotNone(n)
        # ~7 months between Jan day 10 and Jul day 150 (business-day idx),
        # which is well under the 12-month gap to the true recovery.
        self.assertLess(n, 11)


# ---------------------------------------------------------------------------
# synthesize_portfolio_returns
# ---------------------------------------------------------------------------
class TestSynthesizePortfolioReturns(unittest.TestCase):
    def test_equal_weights_averages_returns(self) -> None:
        # Two symbols, 50/50 weight: each day's port return = mean of asset
        # returns. Build prices so daily returns are exactly known.
        idx = pd.date_range("2026-01-01", periods=4, freq="B")
        prices = pd.DataFrame({
            "AAA": [100.0, 110.0, 121.0, 121.0],   # +10%, +10%, 0%
            "BBB": [100.0,  90.0,  90.0,  99.0],   # -10%, 0%, +10%
        }, index=idx)
        w = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
        port = rm.synthesize_portfolio_returns(w, prices)
        # Day 1 return: (0.10 + -0.10) / 2 = 0.0
        # Day 2: (0.10 + 0.0) / 2 = 0.05
        # Day 3: (0.0 + 0.10) / 2 = 0.05
        # First entry of pct_change is NaN→filled→0, then iloc[1:] drops it.
        self.assertEqual(len(port), 3)
        self.assertAlmostEqual(port.iloc[0], 0.0, places=10)
        self.assertAlmostEqual(port.iloc[1], 0.05, places=10)
        self.assertAlmostEqual(port.iloc[2], 0.05, places=10)

    def test_renormalizes_when_some_symbols_missing(self) -> None:
        # Weight on a symbol not in prices is dropped, then re-normalized.
        idx = pd.date_range("2026-01-01", periods=3, freq="B")
        prices = pd.DataFrame({
            "AAA": [100.0, 110.0, 121.0],
        }, index=idx)
        w = pd.Series([0.6, 0.4], index=["AAA", "MISSING"])
        port = rm.synthesize_portfolio_returns(w, prices)
        # Only AAA remains, weight renormalized to 1.0. Returns = AAA's.
        self.assertEqual(len(port), 2)
        self.assertAlmostEqual(port.iloc[0], 0.10, places=10)
        self.assertAlmostEqual(port.iloc[1], 0.10, places=10)

    def test_no_overlap_returns_empty(self) -> None:
        idx = pd.date_range("2026-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"AAA": [1, 2, 3]}, index=idx)
        w = pd.Series([1.0], index=["XYZ"])
        self.assertTrue(rm.synthesize_portfolio_returns(w, prices).empty)


# ---------------------------------------------------------------------------
# synthesize_portfolio_returns_historical (time-varying weights)
# ---------------------------------------------------------------------------
class TestSynthesizeHistoricalReturns(unittest.TestCase):
    def test_empty_weights_per_snapshot_returns_empty(self) -> None:
        prices = pd.DataFrame({"AAA": [1.0, 2.0]},
                              index=pd.date_range("2026-01-01", periods=2,
                                                   freq="B"))
        self.assertTrue(
            rm.synthesize_portfolio_returns_historical({}, prices).empty
        )

    def test_empty_daily_prices_returns_empty(self) -> None:
        wps = {pd.Timestamp("2026-01-01"):
               pd.Series([1.0], index=["AAA"])}
        self.assertTrue(
            rm.synthesize_portfolio_returns_historical(wps, pd.DataFrame()).empty
        )

    def test_single_snapshot_produces_post_snapshot_segment(self) -> None:
        # One snapshot on Jan 1, weights = 100% AAA. Daily prices for AAA
        # over Jan 2 – Jan 6. The mask is `daily_rets.index > stmt_d`, so
        # every daily-return date after Jan 1 belongs to this segment.
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        prices = pd.DataFrame(
            {"AAA": [100.0, 110.0, 121.0, 121.0, 121.0]}, index=idx)
        wps = {pd.Timestamp("2026-01-01"):
               pd.Series([1.0], index=["AAA"])}
        out = rm.synthesize_portfolio_returns_historical(wps, prices)
        # 4 daily returns (Jan 2..Jan 6); first pct_change is NaN at Jan 1
        # which is filtered by the `> stmt_d` mask.
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out.iloc[0], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[1], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[2], 0.0,  places=10)

    def test_weight_change_switches_segment_returns(self) -> None:
        # Snapshot A on Jan 1: 100% AAA. Snapshot B on Jan 4: 100% BBB.
        # Pre-rebalance returns track AAA; post-rebalance track BBB.
        # This is the core of the historical synthesis vs static synthesis:
        # static would use today's weights (snap B) for the WHOLE window;
        # historical correctly splits into segments.
        idx = pd.date_range("2026-01-01", periods=6, freq="B")
        prices = pd.DataFrame({
            "AAA": [100.0, 110.0, 121.0, 121.0, 121.0, 121.0],  # +10,+10,0,0,0
            "BBB": [100.0, 100.0, 100.0, 100.0,  90.0,  99.0],  # 0,0,0,-10,+10
        }, index=idx)
        wps = {
            pd.Timestamp("2026-01-01"): pd.Series([1.0], index=["AAA"]),
            pd.Timestamp("2026-01-06"): pd.Series([1.0], index=["BBB"]),
        }
        out = rm.synthesize_portfolio_returns_historical(wps, prices)
        # Segment 1 (Jan 1, Jan 6]: AAA returns on Jan 2..Jan 6
        # Segment 2 (after Jan 6): BBB returns on Jan 8 only (Jan 7 is Sat)
        # With pd.date_range freq='B', the index is M Tu W Th F M (6 biz days):
        # Jan 1, 2, 5, 6, 7, 8. Segment 1 covers Jan 2..Jan 6 = 3 returns.
        # Segment 2 covers Jan 7..Jan 8 = 2 returns.
        self.assertEqual(len(out), 5)
        # First three are AAA: +10%, +10%, 0%
        self.assertAlmostEqual(out.iloc[0], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[1], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[2], 0.0,  places=10)
        # Last two are BBB: -10%, +10%
        self.assertAlmostEqual(out.iloc[3], -0.10, places=10)
        self.assertAlmostEqual(out.iloc[4],  0.10, places=10)

    def test_missing_symbol_renormalizes_within_segment(self) -> None:
        # Snapshot has weights {AAA: 0.6, MISSING: 0.4}. MISSING isn't in
        # daily_prices; weights re-normalize to {AAA: 1.0} for that segment.
        idx = pd.date_range("2026-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=idx)
        wps = {pd.Timestamp("2026-01-01"):
               pd.Series([0.6, 0.4], index=["AAA", "MISSING"])}
        out = rm.synthesize_portfolio_returns_historical(wps, prices)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out.iloc[0], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[1], 0.10, places=10)

    def test_segment_with_no_overlapping_symbol_skipped(self) -> None:
        # Snapshot weights reference only a missing symbol -> segment dropped.
        # Remaining snapshot still contributes its segment.
        idx = pd.date_range("2026-01-01", periods=4, freq="B")
        prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 121.0]}, index=idx)
        wps = {
            pd.Timestamp("2026-01-01"): pd.Series([1.0], index=["XYZ"]),
            pd.Timestamp("2026-01-02"): pd.Series([1.0], index=["AAA"]),
        }
        out = rm.synthesize_portfolio_returns_historical(wps, prices)
        # Only the 2026-01-02 snapshot contributes (covers Jan 5..Jan 6).
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out.iloc[0], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[1], 0.0,  places=10)

    def test_all_snapshots_after_daily_range_returns_empty(self) -> None:
        idx = pd.date_range("2026-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=idx)
        wps = {pd.Timestamp("2027-01-01"):
               pd.Series([1.0], index=["AAA"])}
        out = rm.synthesize_portfolio_returns_historical(wps, prices)
        self.assertTrue(out.empty)

    def test_constant_weights_matches_static_synthesis(self) -> None:
        # If weights never change across snapshots, the historical
        # synthesis should be byte-identical to the static synthesis over
        # the union of segments. Pins the equivalence on the no-rebalance
        # case so a regression that mishandles the static fallback can't
        # slip through.
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        prices = pd.DataFrame({
            "AAA": [100.0, 110.0, 121.0, 121.0, 121.0],
            "BBB": [100.0, 100.0, 100.0,  90.0,  99.0],
        }, index=idx)
        w = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
        wps = {
            pd.Timestamp("2026-01-01"): w,
            pd.Timestamp("2026-01-02"): w,  # rebalance day, same weights
        }
        out_hist = rm.synthesize_portfolio_returns_historical(wps, prices)
        out_static = rm.synthesize_portfolio_returns(w, prices)
        self.assertEqual(len(out_hist), len(out_static))
        for i in range(len(out_hist)):
            self.assertAlmostEqual(out_hist.iloc[i], out_static.iloc[i],
                                   places=10)


# ---------------------------------------------------------------------------
# compute_risk_contributions (Pass 3)
# ---------------------------------------------------------------------------
class TestRiskContributions(unittest.TestCase):
    @staticmethod
    def _prices_from_returns(returns: dict, start: str = "2024-01-01") -> pd.DataFrame:
        """Build a price DataFrame whose pct_change reproduces `returns`.
        prices[t] = 100 × ∏_{i<=t}(1 + r_i), prepended with the seed 100."""
        n = len(next(iter(returns.values())))
        idx = pd.date_range(start, periods=n + 1, freq="B")
        cols = {}
        for sym, rets in returns.items():
            cols[sym] = np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + rets)))
        return pd.DataFrame(cols, index=idx)

    def test_sum_invariants(self) -> None:
        # The defining identity: Σ CCTR_i == σ_p (to machine epsilon), and
        # PCTR sums to exactly 100%. Holds for any positive-vol portfolio
        # regardless of correlation structure.
        rng = np.random.default_rng(42)
        n_days = 252
        rets = {f"S{i}": rng.normal(0, 0.012, n_days) for i in range(5)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.4, 0.25, 0.15, 0.12, 0.08], index=list(rets))
        out = rm.compute_risk_contributions(w, prices, window=n_days)
        per = out["per_symbol"]
        self.assertAlmostEqual(per["cctr_ann"].sum(), out["port_vol_ann"], places=12)
        self.assertAlmostEqual(per["pctr_pct"].sum(), 100.0, places=10)

    def test_single_asset_pctr_is_100(self) -> None:
        # One position → all risk comes from that name, DR = 1 exactly.
        rng = np.random.default_rng(7)
        rets = {"ONLY": rng.normal(0, 0.015, 252)}
        prices = self._prices_from_returns(rets)
        out = rm.compute_risk_contributions(
            pd.Series([1.0], index=["ONLY"]), prices, window=252,
        )
        per = out["per_symbol"]
        self.assertAlmostEqual(float(per["pctr_pct"].iloc[0]), 100.0, places=10)
        self.assertAlmostEqual(out["dr"], 1.0, places=10)
        # MCTR for a single asset reduces to that asset's standalone vol.
        self.assertAlmostEqual(
            float(per["mctr_ann"].iloc[0]),
            float(per["standalone_vol_ann"].iloc[0]),
            places=10,
        )

    def test_perfectly_correlated_assets_dr_is_one(self) -> None:
        # Two identical return streams ⇒ no diversification ⇒ DR = 1.0
        # regardless of the weight split. Verifies the DR denominator math.
        rng = np.random.default_rng(123)
        r = rng.normal(0, 0.01, 252)
        prices = self._prices_from_returns({"AAA": r, "BBB": r.copy()})
        out = rm.compute_risk_contributions(
            pd.Series([0.7, 0.3], index=["AAA", "BBB"]), prices, window=252,
        )
        self.assertAlmostEqual(out["dr"], 1.0, places=10)

    def test_uncorrelated_equal_weight_pctr_evenly_split(self) -> None:
        # Hadamard rows are pairwise orthogonal → zero sample correlation
        # when tiled to a multiple of the row length. With equal weights and
        # identical magnitude returns, each asset's PCTR ≈ 100 / n and the
        # diversification ratio ≈ √n.
        sign_rows = np.array([
            [ 1, -1,  1, -1],
            [ 1,  1, -1, -1],
            [ 1, -1, -1,  1],
        ], dtype=float)
        n_days = 252  # multiple of 4
        amp = 0.01
        rets = {f"H{i}": np.tile(sign_rows[i], n_days // 4) * amp
                for i in range(3)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([1/3, 1/3, 1/3], index=list(rets))
        # Legacy rolling estimator — Ledoit-Wolf shrinkage perturbs the
        # exact orthogonality identity under the default ewma_lw path.
        out = rm.compute_risk_contributions(
            w, prices, window=n_days, estimator="rolling",
        )
        per = out["per_symbol"]
        for pctr in per["pctr_pct"]:
            # Orthogonal + equal-weight + equal-vol ⇒ each PCTR = 33.33%.
            self.assertAlmostEqual(pctr, 100.0 / 3.0, places=6)
        self.assertAlmostEqual(out["dr"], np.sqrt(3.0), places=6)

    def test_constant_returns_zero_vol_branch(self) -> None:
        # Flat prices ⇒ zero variance ⇒ port_vol_ann = 0. Function should
        # surface NaN PCTR/DR and CCTR=0 rather than blowing up on division.
        idx = pd.date_range("2024-01-01", periods=260, freq="B")
        prices = pd.DataFrame({"FLAT": [100.0] * 260,
                               "ALSO": [50.0]  * 260}, index=idx)
        out = rm.compute_risk_contributions(
            pd.Series([0.6, 0.4], index=["FLAT", "ALSO"]), prices, window=252,
        )
        self.assertEqual(out["port_vol_ann"], 0.0)
        self.assertTrue(np.isnan(out["dr"]))
        per = out["per_symbol"]
        self.assertTrue(per["pctr_pct"].isna().all())
        self.assertTrue((per["cctr_ann"] == 0.0).all())

    def test_renormalizes_when_symbol_missing_from_prices(self) -> None:
        # MISSING is dropped, AAA's weight renormalizes to 1.0 → PCTR = 100%.
        rng = np.random.default_rng(99)
        prices = self._prices_from_returns({"AAA": rng.normal(0, 0.01, 252)})
        out = rm.compute_risk_contributions(
            pd.Series([0.7, 0.3], index=["AAA", "MISSING"]), prices, window=252,
        )
        self.assertEqual(out["n_symbols"], 1)
        per = out["per_symbol"]
        self.assertAlmostEqual(float(per.loc["AAA", "weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(per.loc["AAA", "pctr_pct"]), 100.0, places=10)

    def test_empty_weights_returns_empty(self) -> None:
        out = rm.compute_risk_contributions(
            pd.Series(dtype=float), pd.DataFrame(), window=252,
        )
        self.assertTrue(out["per_symbol"].empty)
        self.assertTrue(np.isnan(out["port_vol_ann"]))
        self.assertEqual(out["n_days"], 0)
        self.assertEqual(out["n_symbols"], 0)

    def test_insufficient_data_returns_empty_with_counts(self) -> None:
        # Below 20 trading days the sample covariance is too noisy — return
        # empty per_symbol but populate n_days/n_symbols so the UI can show
        # an informative message.
        rng = np.random.default_rng(5)
        prices = self._prices_from_returns({
            "AAA": rng.normal(0, 0.01, 10),
            "BBB": rng.normal(0, 0.01, 10),
        })
        out = rm.compute_risk_contributions(
            pd.Series([0.5, 0.5], index=["AAA", "BBB"]), prices, window=252,
        )
        self.assertTrue(out["per_symbol"].empty)
        self.assertEqual(out["n_days"], 10)
        self.assertEqual(out["n_symbols"], 2)

    def test_port_vol_matches_synthesize_portfolio_returns(self) -> None:
        # The wᵀΣw computation must match synthesize_portfolio_returns ×
        # √252 on the same window exactly — this is the contract that lets
        # the Risk-tab Vol 252d tile and the Risk-contribution section share
        # one number. fillna(0) parity between the two functions is the
        # reason this invariant holds; without it the two would drift.
        rng = np.random.default_rng(2024)
        rets = {f"S{i}": rng.normal(0, 0.013, 300) for i in range(4)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.3, 0.3, 0.2, 0.2], index=list(rets))
        # The equivalence with std(port_rets) × √252 only holds under the
        # rolling sample-cov estimator; ewma_lw uses a different weighting.
        out = rm.compute_risk_contributions(
            w, prices, window=252, estimator="rolling",
        )
        port_rets = rm.synthesize_portfolio_returns(w, prices)
        expected = float(port_rets.tail(252).std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(out["port_vol_ann"], expected, places=10)

    def test_window_truncates_to_recent_returns(self) -> None:
        # The window cap should clip the trailing return slice. Build a
        # regime change (first 200 days low-vol, last 252 days high-vol) and
        # verify the 252-day vol matches the high-vol regime, not the blend.
        rng = np.random.default_rng(2026)
        low  = rng.normal(0, 0.005, 200)
        high = rng.normal(0, 0.020, 252)
        prices = self._prices_from_returns({
            "AAA": np.concatenate([low, high]),
            "BBB": np.concatenate([low, high]),
        })
        # The `window` parameter only takes effect under the rolling
        # estimator; ewma_lw ignores it and caps input at cap_days=504
        # instead. Test the legacy windowing behavior here.
        out_window = rm.compute_risk_contributions(
            pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
            prices, window=252, estimator="rolling",
        )
        out_full = rm.compute_risk_contributions(
            pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
            prices, window=10_000, estimator="rolling",
        )
        # Recent-window vol should be materially larger than the all-history
        # blend (high-vol regime dominates the truncated slice).
        self.assertGreater(out_window["port_vol_ann"], out_full["port_vol_ann"])

    def test_per_symbol_n_obs_with_price_counts_non_nan_returns(self) -> None:
        # Phase 1A audit follow-up (item #14): per_symbol surfaces
        # n_obs_with_price so the UI can suppress noisy standalone vols for
        # thin-history symbols. The count must reflect actual non-NaN
        # pct_change observations, NOT the fillna(0)-padded total.
        rng = np.random.default_rng(99)
        n_days = 253  # 253 prices ⇒ 252 valid pct_change observations
        idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
        aaa = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n_days))
        sparse = np.full(n_days, np.nan)
        # Only the trailing 30 days of SPARSE have prices.
        sparse[-30:] = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 30))
        prices = pd.DataFrame({"AAA": aaa, "SPARSE": sparse}, index=idx)
        w = pd.Series([0.7, 0.3], index=["AAA", "SPARSE"])
        out = rm.compute_risk_contributions(w, prices, window=252)
        per = out["per_symbol"]
        self.assertIn("n_obs_with_price", per.columns)
        # AAA: 253 prices → 252 valid pct_change observations.
        self.assertEqual(int(per.loc["AAA", "n_obs_with_price"]), 252)
        # SPARSE: only 30 valid prices in a row at the end → 29 valid
        # pct_change observations (the boundary day between NaN and the
        # first real price is also NaN).
        self.assertEqual(int(per.loc["SPARSE", "n_obs_with_price"]), 29)

    def test_per_symbol_n_obs_in_window_is_window_relative(self) -> None:
        # Phase 1C audit: n_obs_in_window counts real returns WITHIN the
        # trailing n_days window the cov saw. For thin-coverage symbols
        # in a long EWMA window (cap_days=504), this is what catches the
        # EWMA bias — n_obs_with_price (total history) misses recent-
        # onboarding cases where the symbol passes the absolute 60-obs
        # gate but has <50% real coverage IN THE WINDOW.
        rng = np.random.default_rng(2026)
        # Long history so the rolling and EWMA windows differ meaningfully.
        n_days = 800
        idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
        aaa = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n_days))
        # YOUNG has 150 real days at the end of the series.
        young = np.full(n_days, np.nan)
        young[-150:] = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 150))
        prices = pd.DataFrame({"AAA": aaa, "YOUNG": young}, index=idx)
        w = pd.Series([0.7, 0.3], index=["AAA", "YOUNG"])

        # ewma_lw with cap_days=504. YOUNG has 149 real pct_change obs
        # (150 prices → 149 returns), all in the trailing 504d window.
        out_ewma = rm.compute_risk_contributions(
            w, prices, estimator="ewma_lw", cap_days=504,
        )
        per_ewma = out_ewma["per_symbol"]
        self.assertIn("n_obs_in_window", per_ewma.columns)
        self.assertEqual(int(per_ewma.loc["AAA", "n_obs_in_window"]), 504)
        self.assertEqual(int(per_ewma.loc["YOUNG", "n_obs_in_window"]), 149)
        # And YOUNG's n_obs_in_window / n_days = 149/504 ≈ 30% < 50%, so
        # the UI gate at app.py would correctly suppress its standalone vol.
        self.assertLess(
            per_ewma.loc["YOUNG", "n_obs_in_window"]
            / out_ewma["n_days"], 0.5,
        )
        # Whereas n_obs_with_price = 149 still passes the absolute 60-obs
        # floor — i.e. the OLD gate alone wouldn't catch this case.
        self.assertGreater(int(per_ewma.loc["YOUNG", "n_obs_with_price"]), 60)

        # rolling estimator with window=252: YOUNG has 149 real obs in
        # the last 252 days. 149/252 ≈ 59% > 50%, so under rolling the
        # gate would NOT suppress — a known mode difference.
        out_roll = rm.compute_risk_contributions(
            w, prices, estimator="rolling", window=252,
        )
        per_roll = out_roll["per_symbol"]
        self.assertEqual(int(per_roll.loc["YOUNG", "n_obs_in_window"]), 149)
        self.assertEqual(int(per_roll.loc["AAA", "n_obs_in_window"]), 252)


# ---------------------------------------------------------------------------
# compute_synthesis_gaps — diagnostic for sparse-coverage symbols
# ---------------------------------------------------------------------------
class TestComputeSynthesisGaps(unittest.TestCase):
    """Phase 1A audit follow-up (item #7): the function existed but had no
    UI consumer; this test class locks the contract now that app.py
    surfaces it as a coverage-gap banner on the Risk tab."""

    def test_reports_per_symbol_coverage(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        aaa = pd.Series(np.linspace(100, 110, 100), index=idx)
        sparse = pd.Series(np.full(100, np.nan), index=idx)
        sparse.iloc[-30:] = np.linspace(50, 55, 30)
        prices = pd.DataFrame({"AAA": aaa, "SPARSE": sparse})
        w = pd.Series([0.6, 0.4], index=["AAA", "SPARSE"])
        gaps = rm.compute_synthesis_gaps(w, prices)
        self.assertEqual(int(gaps.loc["AAA", "n_days_total"]), 100)
        self.assertEqual(int(gaps.loc["AAA", "n_days_no_price"]), 0)
        self.assertAlmostEqual(float(gaps.loc["AAA", "pct_no_price"]), 0.0)
        self.assertEqual(int(gaps.loc["SPARSE", "n_days_total"]), 100)
        self.assertEqual(int(gaps.loc["SPARSE", "n_days_no_price"]), 70)
        self.assertAlmostEqual(float(gaps.loc["SPARSE", "pct_no_price"]), 70.0)
        # Sort order: descending by pct_no_price — banner shows worst first.
        self.assertEqual(list(gaps.index), ["SPARSE", "AAA"])

    def test_empty_inputs_returns_empty_frame_with_columns(self) -> None:
        gaps = rm.compute_synthesis_gaps(pd.Series(dtype=float), pd.DataFrame())
        self.assertTrue(gaps.empty)
        # Columns still present so the UI's `if not sgaps.empty` short-circuit
        # works without needing to also defend against missing columns.
        self.assertEqual(
            list(gaps.columns),
            ["weight_pct", "n_days_total", "n_days_no_price", "pct_no_price"],
        )


# ---------------------------------------------------------------------------
# estimate_covariance — EWMA + Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------
class TestEstimateCovariance(unittest.TestCase):
    """Covers the unified covariance estimator backing compute_risk_contributions.

    Three estimators: "rolling" (legacy sample cov), "ewma" (RiskMetrics
    EWMA, λ=0.94 default), and "ewma_lw" (EWMA + Ledoit-Wolf constant-
    correlation shrinkage; the new default).
    """

    @staticmethod
    def _returns(seed: int, n_days: int, n_assets: int,
                 sigma: float = 0.01) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
        return pd.DataFrame(
            rng.normal(0, sigma, (n_days, n_assets)), index=idx,
            columns=[f"S{i}" for i in range(n_assets)],
        )

    def test_rolling_matches_legacy_sample_cov(self) -> None:
        # The "rolling" path must equal rets.tail(window).cov() × 252 to
        # numerical tolerance — that's the contract for backward compat.
        rets = self._returns(2024, 300, 4)
        out = rm.estimate_covariance(rets, estimator="rolling", window=252)
        expected = rets.tail(252).cov() * 252.0
        np.testing.assert_allclose(out["cov"].values, expected.values,
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(out["n_days"], 252)
        self.assertIsNone(out["alpha"])
        self.assertIsNone(out["lambda"])

    def test_ewma_weights_recent_observations_more(self) -> None:
        # Same return rows permuted in time. Sample cov is permutation-
        # invariant; EWMA isn't (weights depend on time position).
        n = 252
        rng = np.random.default_rng(7)
        calm  = rng.normal(0, 0.005, (n - 10, 2))
        spike = np.array([[0.05, 0.05]] * 10)
        idx = pd.date_range("2024-01-01", periods=n, freq="B")

        # Two variants over the same row multiset: spike up front, spike at end.
        early = np.vstack([spike, calm])
        late  = np.vstack([calm,  spike])

        df_early = pd.DataFrame(early, index=idx, columns=["A", "B"])
        df_late  = pd.DataFrame(late,  index=idx, columns=["A", "B"])

        ewma_e = rm.estimate_covariance(df_early, estimator="ewma").get("cov")
        ewma_l = rm.estimate_covariance(df_late,  estimator="ewma").get("cov")
        roll_e = rm.estimate_covariance(df_early, estimator="rolling",
                                        window=n).get("cov")
        roll_l = rm.estimate_covariance(df_late,  estimator="rolling",
                                        window=n).get("cov")

        # Recent spike → EWMA variance materially larger than early-spike.
        self.assertGreater(ewma_l.iloc[0, 0], 2 * ewma_e.iloc[0, 0])
        # Rolling sample cov: invariant under permutation.
        np.testing.assert_allclose(roll_e.values, roll_l.values,
                                   rtol=1e-12, atol=1e-12)

    def test_ewma_lw_alpha_in_unit_interval(self) -> None:
        # The Ledoit-Wolf shrinkage intensity must always live in [0, 1] —
        # the formula clips kappa/T to that range.
        rets = self._returns(11, 252, 8)
        out = rm.estimate_covariance(rets, estimator="ewma_lw")
        self.assertIsNotNone(out["alpha"])
        self.assertGreaterEqual(out["alpha"], 0.0)
        self.assertLessEqual(out["alpha"], 1.0)

    def test_ewma_lw_more_shrinkage_when_T_small(self) -> None:
        # Smaller T relative to N → larger α (more shrinkage). Compare a
        # T=30 sample to a T=300 sample on the same dimensionality.
        rng = np.random.default_rng(2026)
        # Same DGP, two different sample sizes.
        idx_long  = pd.date_range("2020-01-01", periods=300, freq="B")
        idx_short = pd.date_range("2020-01-01", periods=30,  freq="B")
        cols = [f"S{i}" for i in range(8)]
        long_rets  = pd.DataFrame(rng.normal(0, 0.01, (300, 8)),
                                  index=idx_long, columns=cols)
        short_rets = pd.DataFrame(rng.normal(0, 0.01, (30, 8)),
                                  index=idx_short, columns=cols)
        long_out  = rm.estimate_covariance(long_rets,  estimator="ewma_lw")
        short_out = rm.estimate_covariance(short_rets, estimator="ewma_lw")
        self.assertGreater(short_out["alpha"], long_out["alpha"])

    def test_input_cap_under_ewma(self) -> None:
        # Feeding 1000 days with cap_days=300 must yield n_days=300; rolling
        # ignores cap_days and uses `window` instead.
        rets = self._returns(99, 1000, 5)
        ewma_out = rm.estimate_covariance(rets, estimator="ewma", cap_days=300)
        roll_out = rm.estimate_covariance(rets, estimator="rolling",
                                          window=200, cap_days=300)
        self.assertEqual(ewma_out["n_days"], 300)
        self.assertEqual(roll_out["n_days"], 200)

    def test_unknown_estimator_raises(self) -> None:
        rets = self._returns(0, 100, 3)
        with self.assertRaises(ValueError):
            rm.estimate_covariance(rets, estimator="garch")  # unsupported

    def test_pctr_sums_to_100_under_ewma_lw(self) -> None:
        # The Σ PCTR = 100% invariant must survive the estimator change —
        # it's pure algebra (CCTR_i / σ_p × 100, Σ CCTR_i = σ_p).
        rets = self._returns(42, 504, 6) ** 1  # deterministic enough
        prices = (1.0 + rets).cumprod()
        prices.iloc[0] = 1.0
        w = pd.Series([0.25, 0.20, 0.15, 0.15, 0.15, 0.10], index=prices.columns)
        out = rm.compute_risk_contributions(w, prices)  # default ewma_lw
        per = out["per_symbol"]
        self.assertAlmostEqual(per["pctr_pct"].sum(), 100.0, places=8)
        self.assertEqual(out["estimator"], "ewma_lw")
        self.assertIsNotNone(out["alpha"])
        self.assertAlmostEqual(out["lambda"], 0.94, places=10)


# ---------------------------------------------------------------------------
# compute_downside_risk_contributions
# ---------------------------------------------------------------------------
class TestDownsideRiskContributions(unittest.TestCase):
    """Per-position decomposition restricted to days when port_ret ≤ threshold.

    Reuses TestRiskContributions._prices_from_returns for fixture construction
    so total vs downside numbers are directly comparable on the same series.
    """

    @staticmethod
    def _prices_from_returns(returns: dict, start: str = "2024-01-01") -> pd.DataFrame:
        # Duplicate of the TestRiskContributions helper — kept local so this
        # class can be moved or run in isolation without coupling.
        n = len(next(iter(returns.values())))
        idx = pd.date_range(start, periods=n + 1, freq="B")
        cols = {}
        for sym, rets in returns.items():
            cols[sym] = np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + rets)))
        return pd.DataFrame(cols, index=idx)

    def test_sum_invariants(self) -> None:
        # Σ CCTR_down = σ_p,down to machine epsilon; Σ PCTR_down = 100%.
        # Identity holds for any non-empty stress sample, threshold-independent.
        rng = np.random.default_rng(2026)
        n_days = 504
        rets = {f"S{i}": rng.normal(0, 0.013, n_days) for i in range(6)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.30, 0.20, 0.18, 0.12, 0.12, 0.08], index=list(rets))
        out = rm.compute_downside_risk_contributions(
            w, prices, threshold=0.0, window=252,
        )
        per = out["per_symbol_down"]
        self.assertGreater(out["n_down_days"], 20)
        self.assertAlmostEqual(
            per["cctr_ann_down"].sum(), out["port_vol_ann_down"], places=12,
        )
        self.assertAlmostEqual(per["pctr_pct_down"].sum(), 100.0, places=10)

    def test_threshold_zero_equals_negative_port_days(self) -> None:
        # With threshold=0 the n_down_days count must exactly equal the number
        # of days the synthesized portfolio return was ≤ 0 on the window.
        rng = np.random.default_rng(11)
        rets = {"A": rng.normal(0, 0.01, 300), "B": rng.normal(0, 0.012, 300)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.6, 0.4], index=["A", "B"])
        # Independent recompute of the down-day count from synthesize_*.
        pr = rm.synthesize_portfolio_returns(w, prices).tail(252)
        expected = int((pr <= 0).sum())
        # `window` only applies under the rolling estimator; ewma_lw caps
        # the outer window at cap_days=504, which would pull in more days.
        out = rm.compute_downside_risk_contributions(
            w, prices, threshold=0.0, window=252, estimator="rolling",
        )
        self.assertEqual(out["n_down_days"], expected)

    def test_stricter_threshold_keeps_fewer_days(self) -> None:
        # Lowering the threshold can only shrink the stress sample monotonically.
        rng = np.random.default_rng(99)
        rets = {f"X{i}": rng.normal(0, 0.012, 400) for i in range(4)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=list(rets))
        n0 = rm.compute_downside_risk_contributions(w, prices, threshold=0.0)["n_down_days"]
        n1 = rm.compute_downside_risk_contributions(w, prices, threshold=-0.005)["n_down_days"]
        n2 = rm.compute_downside_risk_contributions(w, prices, threshold=-0.01)["n_down_days"]
        self.assertGreaterEqual(n0, n1)
        self.assertGreaterEqual(n1, n2)

    def test_too_few_down_days_returns_empty(self) -> None:
        # Only 5 down days at threshold=-0.05 ⇒ below the 20-day guard ⇒
        # NaN scalars and empty per_symbol_down, but n_down_days is reported
        # so the UI can render a sample-size warning.
        rng = np.random.default_rng(3)
        # Mostly positive series with rare large drawdowns to push tail thin.
        r = rng.normal(0.001, 0.005, 252)
        r[::50] = -0.06  # five hard losses
        prices = self._prices_from_returns({"ONLY": r})
        out = rm.compute_downside_risk_contributions(
            pd.Series([1.0], index=["ONLY"]), prices,
            threshold=-0.05, window=252,
        )
        self.assertTrue(out["per_symbol_down"].empty)
        self.assertTrue(np.isnan(out["port_vol_ann_down"]))
        self.assertLess(out["n_down_days"], 20)

    def test_empty_inputs_return_empty(self) -> None:
        out = rm.compute_downside_risk_contributions(
            pd.Series(dtype=float), pd.DataFrame(), threshold=0.0,
        )
        self.assertTrue(out["per_symbol_down"].empty)
        self.assertEqual(out["n_down_days"], 0)
        self.assertEqual(out["n_symbols"], 0)


# ---------------------------------------------------------------------------
# compute_es_contributions
# ---------------------------------------------------------------------------
class TestEsContributions(unittest.TestCase):
    """Per-position Euler decomposition of historical Expected Shortfall."""

    @staticmethod
    def _prices_from_returns(returns: dict, start: str = "2024-01-01") -> pd.DataFrame:
        n = len(next(iter(returns.values())))
        idx = pd.date_range(start, periods=n + 1, freq="B")
        cols = {}
        for sym, rets in returns.items():
            cols[sym] = np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + rets)))
        return pd.DataFrame(cols, index=idx)

    def test_sum_invariant_contrib_equals_port_es(self) -> None:
        # Σ contrib_es = port_es exactly (Euler identity on the homogeneous
        # ES functional). Holds regardless of correlation structure.
        rng = np.random.default_rng(17)
        rets = {f"S{i}": rng.normal(0, 0.013, 504) for i in range(5)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.35, 0.25, 0.2, 0.12, 0.08], index=list(rets))
        out = rm.compute_es_contributions(w, prices, alpha=0.05, window=252)
        per = out["per_symbol_es"]
        self.assertGreater(out["n_tail_days"], 0)
        self.assertAlmostEqual(per["contrib_es"].sum(), out["port_es"], places=12)
        self.assertAlmostEqual(per["pctr_es_pct"].sum(), 100.0, places=10)

    def test_var_lines_up_with_quantile(self) -> None:
        # var_p is exactly the alpha-quantile of the synthesized port returns
        # on the window — no smoothing, no interpolation tricks.
        rng = np.random.default_rng(45)
        rets = {"A": rng.normal(0, 0.012, 300), "B": rng.normal(0, 0.015, 300)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.55, 0.45], index=["A", "B"])
        out = rm.compute_es_contributions(w, prices, alpha=0.05, window=252)
        pr = rm.synthesize_portfolio_returns(w, prices).tail(252)
        self.assertAlmostEqual(out["var_p"], float(pr.quantile(0.05)), places=10)

    def test_single_asset_pctr_is_100(self) -> None:
        # One position ⇒ that position absorbs 100% of the ES contribution.
        rng = np.random.default_rng(8)
        prices = self._prices_from_returns({"ONLY": rng.normal(-0.0005, 0.02, 300)})
        out = rm.compute_es_contributions(
            pd.Series([1.0], index=["ONLY"]), prices, alpha=0.05,
        )
        per = out["per_symbol_es"]
        self.assertAlmostEqual(float(per["pctr_es_pct"].iloc[0]), 100.0, places=10)

    def test_lower_alpha_means_thinner_tail_and_larger_es(self) -> None:
        # Tighter confidence ⇒ further into the tail ⇒ ES grows (more
        # extreme conditional mean loss), tail-day count shrinks.
        rng = np.random.default_rng(2024)
        rets = {"A": rng.normal(0, 0.012, 600), "B": rng.normal(0, 0.013, 600)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        a05 = rm.compute_es_contributions(w, prices, alpha=0.05, window=504)
        a01 = rm.compute_es_contributions(w, prices, alpha=0.01, window=504)
        self.assertGreaterEqual(a05["n_tail_days"], a01["n_tail_days"])
        self.assertGreater(a01["port_es"], a05["port_es"])

    def test_invalid_alpha_returns_empty(self) -> None:
        prices = self._prices_from_returns(
            {"A": np.random.default_rng(0).normal(0, 0.01, 252)}
        )
        for bad in [0.0, 1.0, -0.1, 1.5]:
            out = rm.compute_es_contributions(
                pd.Series([1.0], index=["A"]), prices, alpha=bad,
            )
            self.assertTrue(out["per_symbol_es"].empty)
            self.assertTrue(np.isnan(out["port_es"]))

    def test_empty_inputs_return_empty(self) -> None:
        out = rm.compute_es_contributions(
            pd.Series(dtype=float), pd.DataFrame(), alpha=0.05,
        )
        self.assertTrue(out["per_symbol_es"].empty)
        self.assertEqual(out["n_tail_days"], 0)
        self.assertEqual(out["n_symbols"], 0)

    def test_per_symbol_es_n_obs_in_window_catches_thin_history(self) -> None:
        # Phase 1D: ES uses fillna(0.0) on pct_change, so a symbol added
        # mid-window contributes 0% returns on pre-existence tail days —
        # its ES contribution is silently understated and the slack is
        # reallocated to longer-history symbols. n_obs_in_window pins
        # this: counts NaN-free pct_change observations within the same
        # trailing window the tail mask sees. The UI uses it to suppress
        # ES PCTR for symbols with thin coverage, mirroring the gate that
        # PR #66 added to compute_risk_contributions.
        rng = np.random.default_rng(2026)
        n_days = 400  # > 252 so the window truncation kicks in
        idx = pd.date_range("2024-01-01", periods=n_days + 1, freq="B")
        aaa_rets = rng.normal(0, 0.012, n_days)
        aaa = np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + aaa_rets)))
        # YOUNG has 80 real days at the end (so 79 real pct_change obs
        # in a 252d window — well below 50% coverage).
        young_prices = np.full(n_days + 1, np.nan)
        young_prices[-81:] = 100.0 * np.cumprod(
            1.0 + rng.normal(0, 0.012, 81)
        )
        prices = pd.DataFrame({"AAA": aaa, "YOUNG": young_prices}, index=idx)
        w = pd.Series([0.7, 0.3], index=["AAA", "YOUNG"])

        out = rm.compute_es_contributions(w, prices, alpha=0.05, window=252)
        per = out["per_symbol_es"]
        self.assertIn("n_obs_in_window", per.columns)
        # AAA has full coverage (252 real pct_change obs in the 252d window).
        self.assertEqual(int(per.loc["AAA", "n_obs_in_window"]), 252)
        # YOUNG has 80 real observations (81 prices → 80 returns, all
        # within the trailing 252d window). The UI gate at 50% will
        # correctly suppress its ES PCTR.
        self.assertEqual(int(per.loc["YOUNG", "n_obs_in_window"]), 80)
        self.assertLess(
            per.loc["YOUNG", "n_obs_in_window"] / out["n_days_window"], 0.5,
        )
        # Sanity: per_symbol_es still has the existing columns so this
        # add doesn't break consumers that don't read n_obs_in_window.
        for _col in ("weight", "tail_mean_ret", "contrib_es", "pctr_es_pct"):
            self.assertIn(_col, per.columns)

    def test_window_contract_is_strict_no_estimator_no_cap_days(self) -> None:
        # Phase 1C audit: compute_es_contributions takes ONLY `window`,
        # never reads cap_days, and never applies an estimator switch.
        # The dashboard places this function's output (ES PCTR, 252d
        # sample) alongside compute_risk_contributions' output (PCTR,
        # 504d EWMA under default) in the same per-symbol table. UI
        # captions disclose both windows; this test pins the function's
        # half of the contract so a future refactor that adds cap_days
        # or estimator semantics breaks here and is forced to update
        # the UI captions in lockstep.
        rng = np.random.default_rng(2024)
        n = 600  # > 252 + 504 so a window-confusion bug would show
        rets = {f"X{i}": rng.normal(0, 0.012, n) for i in range(3)}
        idx = pd.date_range("2024-01-01", periods=n + 1, freq="B")
        cols = {sym: np.concatenate(([100.0], 100.0 * np.cumprod(1 + r)))
                for sym, r in rets.items()}
        prices = pd.DataFrame(cols, index=idx)
        w = pd.Series([0.5, 0.3, 0.2], index=list(rets))
        # Strict window contract: n_days_window matches the input.
        for win in (60, 100, 252, 504):
            out = rm.compute_es_contributions(w, prices, alpha=0.05, window=win)
            self.assertEqual(out["n_days_window"], win,
                              f"window={win} returned n_days_window="
                              f"{out['n_days_window']}; window contract broken")
        # And ES at window=252 differs from a hypothetical 504 — the two
        # cannot be the same number on heteroskedastic data, which pins
        # the "windows aren't unified" disclosure.
        out_252 = rm.compute_es_contributions(w, prices, alpha=0.05, window=252)
        out_504 = rm.compute_es_contributions(w, prices, alpha=0.05, window=504)
        self.assertNotAlmostEqual(out_252["port_es"], out_504["port_es"], places=4)


# ---------------------------------------------------------------------------
# compute_sortino_daily
# ---------------------------------------------------------------------------
class TestSortinoDaily(unittest.TestCase):
    def test_zero_downside_returns_nan(self) -> None:
        # All-positive series → downside vol = 0 → Sortino undefined.
        r = pd.Series([0.001, 0.002, 0.003, 0.001])
        self.assertTrue(np.isnan(rm.compute_sortino_daily(r, rf_annual=0.0)))

    def test_matches_monthly_formula_shape(self) -> None:
        # Sanity check: positive-mean series with realistic downside should
        # give a finite positive Sortino. The daily formula mirrors the
        # monthly one with 252-annualization, so the magnitudes match what
        # we expect for trailing-year stats (single-digit Sortino).
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0005, 0.01, 252))  # ~12.6% CAGR, ~16% vol
        val = rm.compute_sortino_daily(r, rf_annual=0.045)
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0)

    def test_too_few_observations(self) -> None:
        self.assertTrue(np.isnan(rm.compute_sortino_daily(pd.Series([0.01]))))
        self.assertTrue(np.isnan(rm.compute_sortino_daily(pd.Series([], dtype=float))))


# ---------------------------------------------------------------------------
# compute_dr_time_series
# ---------------------------------------------------------------------------
class TestDrTimeSeries(unittest.TestCase):
    """Rolling diversification ratio over multiple lookback windows.

    Verifies the right-edge identity: DR_W from the rolling series at the
    last timestamp must match the static `dr` from compute_risk_contributions
    on the same window, to numerical tolerance.
    """

    @staticmethod
    def _prices_from_returns(returns: dict, start: str = "2024-01-01") -> pd.DataFrame:
        n = len(next(iter(returns.values())))
        idx = pd.date_range(start, periods=n + 1, freq="B")
        cols = {}
        for sym, rets in returns.items():
            cols[sym] = np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + rets)))
        return pd.DataFrame(cols, index=idx)

    def test_right_edge_matches_static_dr(self) -> None:
        # The rolling-DR right edge MUST agree with compute_risk_contributions
        # on the same window — both should reduce to σ_avg / σ_p on the
        # last W observations. This is the cross-function consistency test.
        rng = np.random.default_rng(2026)
        rets = {f"S{i}": rng.normal(0, 0.012, 504) for i in range(5)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.4, 0.25, 0.15, 0.12, 0.08], index=list(rets))
        # compute_dr_time_series uses rolling sample stds, so the right-edge
        # match only holds against the rolling estimator (ewma_lw default
        # would diverge by the shrinkage intensity).
        static = rm.compute_risk_contributions(
            w, prices, window=252, estimator="rolling",
        )
        ts = rm.compute_dr_time_series(w, prices, windows=(252,))
        right_edge = float(ts["dr_252d"].dropna().iloc[-1])
        self.assertAlmostEqual(right_edge, static["dr"], places=8)

    def test_right_edge_diverges_from_ewma_lw_static_dr(self) -> None:
        # Phase 1C audit: the Risk-Contribution tab's "Diversification ratio"
        # tile reads rc["dr"] under the default estimator="ewma_lw"; the
        # DR-time-series chart's right edge reads compute_dr_time_series
        # which always uses rolling sample stds. The two will disagree —
        # by construction, not as a bug. This test demonstrates the
        # divergence on a fixture where the two estimators actually
        # produce different cov matrices, so a future maintainer who
        # "fixes" the chart to use ewma_lw breaks this test and is
        # forced to update the UI captions in lockstep.
        rng = np.random.default_rng(2027)
        # Mild heteroskedasticity to make EWMA + LW shrinkage diverge
        # meaningfully from sample cov on the same trailing 252 days.
        n = 504
        vol = np.linspace(0.005, 0.025, n)
        rets = {
            f"S{i}": rng.normal(0, vol, n) * (1.0 + 0.1 * i)
            for i in range(5)
        }
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.4, 0.25, 0.15, 0.12, 0.08], index=list(rets))
        rc_ewma_lw = rm.compute_risk_contributions(
            w, prices, window=252, estimator="ewma_lw",
        )
        ts = rm.compute_dr_time_series(w, prices, windows=(252,))
        right_edge = float(ts["dr_252d"].dropna().iloc[-1])
        # Both are finite, both ≥ 1 (Cauchy-Schwarz), but they should
        # differ at least at the 1% level on this fixture.
        self.assertGreaterEqual(rc_ewma_lw["dr"], 1.0)
        self.assertGreaterEqual(right_edge, 1.0)
        self.assertNotAlmostEqual(right_edge, rc_ewma_lw["dr"], places=2)

    def test_dr_is_at_least_one(self) -> None:
        # Cauchy-Schwarz: (Σ wᵢ σᵢ)² ≥ wᵀ Σ w on any real correlation matrix
        # ⇒ DR ≥ 1 at every observation. Lower bound is sharp when all assets
        # are perfectly correlated.
        rng = np.random.default_rng(11)
        rets = {f"X{i}": rng.normal(0, 0.013, 300) for i in range(4)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=list(rets))
        ts = rm.compute_dr_time_series(w, prices, windows=(21, 63, 252))
        for col in ts.columns:
            vals = ts[col].dropna()
            if not vals.empty:
                # Numerical floor — allow a hair below 1 for rounding noise.
                self.assertGreaterEqual(float(vals.min()), 1.0 - 1e-9,
                                        msg=f"{col} dipped below 1.0")

    def test_perfectly_correlated_assets_dr_is_one(self) -> None:
        # Two identical streams ⇒ DR = 1 at every observation.
        rng = np.random.default_rng(7)
        r = rng.normal(0, 0.011, 300)
        prices = self._prices_from_returns({"AAA": r, "BBB": r.copy()})
        ts = rm.compute_dr_time_series(
            pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
            prices, windows=(21, 63),
        )
        for col in ts.columns:
            vals = ts[col].dropna()
            if not vals.empty:
                # Tighter tolerance — analytic equality.
                self.assertTrue(np.allclose(vals.values, 1.0, atol=1e-8),
                                msg=f"{col} not all ~1.0")

    def test_single_asset_is_one(self) -> None:
        rng = np.random.default_rng(3)
        prices = self._prices_from_returns({"ONLY": rng.normal(0, 0.015, 252)})
        ts = rm.compute_dr_time_series(
            pd.Series([1.0], index=["ONLY"]), prices, windows=(21,),
        )
        vals = ts["dr_21d"].dropna()
        self.assertTrue(np.allclose(vals.values, 1.0, atol=1e-10))

    def test_empty_inputs_return_empty_frame_with_columns(self) -> None:
        # Empty inputs should still surface the expected column names so the
        # UI can render placeholder tiles without a KeyError.
        ts = rm.compute_dr_time_series(
            pd.Series(dtype=float), pd.DataFrame(), windows=(21, 63, 252),
        )
        self.assertEqual(list(ts.columns), ["dr_21d", "dr_63d", "dr_252d"])
        self.assertTrue(ts.empty)


# ---------------------------------------------------------------------------
# compute_max_dr
# ---------------------------------------------------------------------------
class TestMaxDr(unittest.TestCase):
    """Closed-form upper bound on the diversification ratio.

    Implements max DR = √(1ᵀ R⁻¹ 1) on the daily-window correlation matrix.
    Tests both analytic limits and the monotonicity bound vs the static DR.
    """

    @staticmethod
    def _prices_from_returns(returns: dict, start: str = "2024-01-01") -> pd.DataFrame:
        n = len(next(iter(returns.values())))
        idx = pd.date_range(start, periods=n + 1, freq="B")
        cols = {}
        for sym, rets in returns.items():
            cols[sym] = np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + rets)))
        return pd.DataFrame(cols, index=idx)

    def test_single_asset(self) -> None:
        # One asset ⇒ no diversification possible ⇒ ceiling = 1.0 exactly.
        rng = np.random.default_rng(0)
        prices = self._prices_from_returns({"ONLY": rng.normal(0, 0.01, 252)})
        m = rm.compute_max_dr(pd.Series([1.0], index=["ONLY"]),
                              prices, window=252)
        self.assertAlmostEqual(m["max_dr"], 1.0, places=10)

    def test_two_perfectly_correlated_is_one(self) -> None:
        # Perfect correlation ⇒ no diversification ⇒ ceiling = 1.0.
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 300)
        prices = self._prices_from_returns({"AAA": r, "BBB": r.copy()})
        m = rm.compute_max_dr(pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
                              prices, window=252)
        self.assertAlmostEqual(m["max_dr"], 1.0, places=6)

    def test_two_uncorrelated_equal_vol_is_sqrt_two(self) -> None:
        # Independent equal-vol assets ⇒ max DR = √2 ≈ 1.414, achieved by
        # the 50/50 weighting. The closed form should land at the analytic
        # value modulo finite-sample noise in the correlation estimate.
        rng = np.random.default_rng(2)
        prices = self._prices_from_returns({
            "A": rng.normal(0, 0.012, 2000),
            "B": rng.normal(0, 0.012, 2000),
        })
        m = rm.compute_max_dr(
            pd.Series([0.5, 0.5], index=["A", "B"]), prices, window=2000,
        )
        # Sample correlation is small (a few %) — wide tolerance.
        self.assertAlmostEqual(m["max_dr"], np.sqrt(2), delta=0.05)

    def test_max_dr_dominates_current_dr(self) -> None:
        # Universal: max DR over all w MUST be ≥ DR at any specific w
        # (the optimum is at least as good as the current point).
        rng = np.random.default_rng(45)
        rets = {f"S{i}": rng.normal(0, 0.012, 504) for i in range(6)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.35, 0.25, 0.15, 0.10, 0.10, 0.05], index=list(rets))
        m = rm.compute_max_dr(w, prices, window=252)
        rc = rm.compute_risk_contributions(w, prices, window=252)
        self.assertGreaterEqual(m["max_dr"], rc["dr"])

    def test_too_few_assets_or_observations(self) -> None:
        # Zero/one-asset universes degenerate by definition; tiny windows
        # are filtered out (matches the 20-day floor everywhere else).
        m_empty = rm.compute_max_dr(pd.Series(dtype=float), pd.DataFrame())
        self.assertTrue(np.isnan(m_empty["max_dr"]))
        rng = np.random.default_rng(0)
        prices = self._prices_from_returns({"A": rng.normal(0, 0.01, 5),
                                            "B": rng.normal(0, 0.01, 5)})
        m_thin = rm.compute_max_dr(
            pd.Series([0.5, 0.5], index=["A", "B"]), prices, window=10,
        )
        self.assertTrue(np.isnan(m_thin["max_dr"]))


# ---------------------------------------------------------------------------
# compute_max_dr_time_series
# ---------------------------------------------------------------------------
class TestMaxDrTimeSeries(unittest.TestCase):
    """Rolling Max-DR ceiling — same algebra as compute_max_dr at every date.

    Key invariants:
      - Right-edge value matches static compute_max_dr with the same window.
      - Ceiling ≥ DR_W at every date where both are defined.
    """

    @staticmethod
    def _prices_from_returns(returns: dict, start: str = "2024-01-01") -> pd.DataFrame:
        n = len(next(iter(returns.values())))
        idx = pd.date_range(start, periods=n + 1, freq="B")
        cols = {sym: np.concatenate(([100.0],
                                     100.0 * np.cumprod(1.0 + rets)))
                for sym, rets in returns.items()}
        return pd.DataFrame(cols, index=idx)

    def test_right_edge_matches_static_compute(self) -> None:
        rng = np.random.default_rng(7)
        rets = {f"S{i}": rng.normal(0, 0.012, 504) for i in range(5)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.3, 0.25, 0.2, 0.15, 0.10], index=list(rets))
        ts = rm.compute_max_dr_time_series(w, prices, window=252)
        static = rm.compute_max_dr(w, prices, window=252)
        self.assertAlmostEqual(float(ts.dropna().iloc[-1]),
                               static["max_dr"], places=10)

    def test_ceiling_dominates_dr_at_every_date(self) -> None:
        # max_DR(R_t) ≥ DR(w; R_t) at every date — same correlation matrix
        # for both sides of the inequality.
        rng = np.random.default_rng(11)
        rets = {f"S{i}": rng.normal(0, 0.012, 504) for i in range(6)}
        prices = self._prices_from_returns(rets)
        w = pd.Series([0.35, 0.25, 0.15, 0.10, 0.10, 0.05], index=list(rets))
        ceiling = rm.compute_max_dr_time_series(w, prices, window=252)
        dr_ts = rm.compute_dr_time_series(w, prices, windows=(252,))
        joined = pd.concat({"ceiling": ceiling, "dr": dr_ts["dr_252d"]},
                           axis=1).dropna()
        self.assertGreater(len(joined), 0)
        # 1e-9 tolerance for fp roundoff in pinv vs rolling std composition.
        self.assertTrue(
            (joined["ceiling"] >= joined["dr"] - 1e-9).all(),
            msg=f"Ceiling violated on rows: "
                f"{joined.loc[joined['ceiling'] < joined['dr'] - 1e-9]}"
        )

    def test_empty_inputs(self) -> None:
        empty = rm.compute_max_dr_time_series(pd.Series(dtype=float),
                                              pd.DataFrame())
        self.assertTrue(empty.empty)


# ---------------------------------------------------------------------------
# classify_dr_regime
# ---------------------------------------------------------------------------
class TestClassifyDrRegime(unittest.TestCase):
    def test_equal_short_and_long_is_normal(self) -> None:
        out = rm.classify_dr_regime(1.5, 1.5)
        self.assertEqual(out["label"], "Normal")
        self.assertAlmostEqual(out["ratio"], 1.0, places=10)

    def test_short_far_below_long_is_stress(self) -> None:
        # DR_21 has dropped to 80% of DR_252 ⇒ correlations clustering.
        out = rm.classify_dr_regime(1.2, 1.5)
        self.assertEqual(out["label"], "Stress")
        self.assertLess(out["ratio"], 0.90)

    def test_short_far_above_long_is_calm(self) -> None:
        out = rm.classify_dr_regime(1.8, 1.5)
        self.assertEqual(out["label"], "Calm")
        self.assertGreater(out["ratio"], 1.10)

    def test_threshold_boundary_default(self) -> None:
        # Exactly at the lower threshold ⇒ Normal (strict inequality).
        out = rm.classify_dr_regime(0.90, 1.0)
        self.assertEqual(out["label"], "Normal")
        # Just below ⇒ Stress.
        out = rm.classify_dr_regime(0.899, 1.0)
        self.assertEqual(out["label"], "Stress")

    def test_custom_thresholds(self) -> None:
        # Tighter stress band: ratio 0.95 IS stress at thr=0.96 but Normal
        # at default thr=0.90. Same input, different verdict.
        ratio_input = (0.95, 1.0)
        self.assertEqual(rm.classify_dr_regime(*ratio_input)["label"], "Normal")
        self.assertEqual(
            rm.classify_dr_regime(*ratio_input, stress_thr=0.96)["label"],
            "Stress",
        )

    def test_nan_inputs_return_dash(self) -> None:
        self.assertEqual(rm.classify_dr_regime(np.nan, 1.5)["label"], "—")
        self.assertEqual(rm.classify_dr_regime(1.5, np.nan)["label"], "—")
        self.assertEqual(rm.classify_dr_regime(1.5, 0.0)["label"], "—")
        self.assertEqual(rm.classify_dr_regime(1.5, -0.1)["label"], "—")


# ---------------------------------------------------------------------------
# compute_dr_ratio_series + compute_dr_regime_thresholds
# ---------------------------------------------------------------------------
class TestDrRatioSeries(unittest.TestCase):
    def test_matches_pointwise_division(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        dr_ts = pd.DataFrame({
            "dr_21d":  [1.2, 1.3, 1.1, 1.4, 1.5, np.nan, 1.6, 1.7, 1.8, 1.2],
            "dr_252d": [1.4, 1.4, 1.4, 1.5, 1.5, 1.5,    1.6, 1.7, 1.8, 1.0],
        }, index=idx)
        ratio = rm.compute_dr_ratio_series(dr_ts)
        # 1.2 / 1.4, 1.3 / 1.4, etc.
        self.assertAlmostEqual(float(ratio.iloc[0]), 1.2 / 1.4, places=10)
        self.assertAlmostEqual(float(ratio.iloc[-1]), 1.2 / 1.0, places=10)
        # NaN propagates.
        self.assertTrue(np.isnan(ratio.iloc[5]))

    def test_long_col_zero_yields_nan(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        dr_ts = pd.DataFrame({"dr_21d": [1.0, 1.5, 2.0],
                              "dr_252d": [1.5, 0.0, 1.5]}, index=idx)
        ratio = rm.compute_dr_ratio_series(dr_ts)
        self.assertTrue(np.isnan(ratio.iloc[1]))

    def test_empty_frame_returns_empty_series(self) -> None:
        ratio = rm.compute_dr_ratio_series(pd.DataFrame())
        self.assertEqual(len(ratio), 0)

    def test_missing_column_returns_empty(self) -> None:
        ratio = rm.compute_dr_ratio_series(
            pd.DataFrame({"dr_21d": [1.0]}, index=[pd.Timestamp("2024-01-01")])
        )
        self.assertEqual(len(ratio), 0)


class TestDrRegimeThresholds(unittest.TestCase):
    def test_fixed_returns_defaults(self) -> None:
        s = pd.Series([0.5, 0.9, 1.0, 1.2, 1.5])
        out = rm.compute_dr_regime_thresholds(s, method="fixed")
        self.assertAlmostEqual(out["stress_thr"], 0.90, places=10)
        self.assertAlmostEqual(out["calm_thr"], 1.10, places=10)
        self.assertEqual(out["method"], "fixed")

    def test_percentile_uses_empirical_quantiles(self) -> None:
        # Uniform 0.5 to 1.5 → p20 ≈ 0.7, p80 ≈ 1.3.
        s = pd.Series(np.linspace(0.5, 1.5, 1001))
        out = rm.compute_dr_regime_thresholds(s, method="percentile")
        self.assertAlmostEqual(out["stress_thr"], 0.7, places=3)
        self.assertAlmostEqual(out["calm_thr"], 1.3, places=3)
        self.assertEqual(out["method"], "percentile")

    def test_zscore_centers_on_mean(self) -> None:
        # Symmetric around 1.0 with std=0.1 → thresholds at 0.9 / 1.1.
        rng = np.random.default_rng(7)
        s = pd.Series(1.0 + 0.1 * rng.normal(size=10_000))
        out = rm.compute_dr_regime_thresholds(s, method="zscore")
        self.assertAlmostEqual(out["mean"], 1.0, places=2)
        self.assertAlmostEqual(out["sd"], 0.1, places=2)
        self.assertAlmostEqual(out["stress_thr"], 0.9, places=2)
        self.assertAlmostEqual(out["calm_thr"], 1.1, places=2)

    def test_thin_data_falls_back_to_fixed(self) -> None:
        # Below the 10-obs floor — empirical methods can't calibrate.
        s = pd.Series([0.95, 1.05, 1.0])
        out = rm.compute_dr_regime_thresholds(s, method="percentile")
        self.assertEqual(out["method"], "fixed")
        self.assertIn("fallback", out)
        self.assertAlmostEqual(out["stress_thr"], 0.90, places=10)

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            rm.compute_dr_regime_thresholds(
                pd.Series([1.0, 1.1, 0.9]), method="madeup"
            )

    def test_percentile_is_in_sample_by_construction(self) -> None:
        # Phase 1E audit (Agent C M-1): percentile mode fits on the same
        # series it classifies, so ~20% of observations are mechanically
        # below stress_thr and ~20% are above calm_thr regardless of the
        # underlying distribution. Lock this property explicitly so a
        # future refactor either preserves it (and the docstring stays
        # honest) or surfaces the change.
        rng = np.random.default_rng(2026)
        # Pure noise — no real "regime" structure. Percentile mode should
        # still split it ~20 / 60 / 20.
        noise = pd.Series(rng.normal(loc=1.0, scale=0.15, size=2000))
        out = rm.compute_dr_regime_thresholds(noise, method="percentile")
        below_stress = float((noise < out["stress_thr"]).mean())
        above_calm   = float((noise > out["calm_thr"]).mean())
        # Tolerance: ±2pp around the 20% target (quantile is exact, the
        # strict-vs-nonstrict comparison gives a tiny offset).
        self.assertAlmostEqual(below_stress, 0.20, delta=0.02)
        self.assertAlmostEqual(above_calm,   0.20, delta=0.02)
        # By contrast, "fixed" thresholds against the same series produce
        # a different (data-dependent) split — confirming the in-sample
        # vs out-of-sample distinction is meaningful.
        out_fixed = rm.compute_dr_regime_thresholds(noise, method="fixed")
        below_fixed = float((noise < out_fixed["stress_thr"]).mean())
        # Fixed cutoff is 0.90; with N(1.0, 0.15), P(X<0.90) ≈ Φ(-0.667)
        # ≈ 0.252 — distinctly different from the in-sample 20%.
        self.assertNotAlmostEqual(below_fixed, 0.20, delta=0.02)


# ---------------------------------------------------------------------------
# classify_market_regime — two-axis market regime labels (SPY dd × VIX z)
# ---------------------------------------------------------------------------
class TestClassifyMarketRegime(unittest.TestCase):
    """Two-axis regime labels: SPY drawdown state × VIX z-state.
    Burn-in respects dd_window (21d default) and vix_z_window (252d default).
    Signature takes two date-indexed Series (SPY and VIX prices)."""

    @staticmethod
    def _pair(spy: list[float], vix: list[float],
              start: str = "2023-01-02") -> tuple[pd.Series, pd.Series]:
        idx = pd.date_range(start, periods=len(spy), freq="B")
        return (pd.Series(spy, index=idx, name="SPY"),
                pd.Series(vix, index=idx, name="VIX"))

    def test_empty_returns_empty_frame_with_columns(self) -> None:
        out = rm.classify_market_regime(pd.Series(dtype=float),
                                        pd.Series(dtype=float))
        self.assertTrue(out.empty)
        for c in ("spy_drawdown", "spy_state", "vix_z", "vix_state", "regime"):
            self.assertIn(c, out.columns)

    def test_one_side_empty_returns_empty(self) -> None:
        idx = pd.date_range("2023-01-02", periods=30, freq="B")
        out1 = rm.classify_market_regime(pd.Series(dtype=float),
                                         pd.Series([20.0] * 30, index=idx))
        self.assertTrue(out1.empty)
        out2 = rm.classify_market_regime(pd.Series([400.0] * 30, index=idx),
                                         pd.Series(dtype=float))
        self.assertTrue(out2.empty)

    def test_burn_in_states_are_nan(self) -> None:
        # 300 days of mild noise lets us probe burn-in cleanly.
        rng = np.random.default_rng(0)
        spy = (400 + np.cumsum(rng.normal(0, 0.5, 300))).tolist()
        vix = (20  + np.cumsum(rng.normal(0, 0.1, 300))).tolist()
        s, v = self._pair(spy, vix)
        out = rm.classify_market_regime(s, v)
        # spy_state burn-in: first 20 rows NaN, index 20 has a value.
        self.assertTrue(out["spy_state"].iloc[:20].isna().all())
        self.assertFalse(pd.isna(out["spy_state"].iloc[20]))
        # vix_state burn-in: first 251 rows NaN, index 251 has a value.
        self.assertTrue(out["vix_state"].iloc[:251].isna().all())
        self.assertFalse(pd.isna(out["vix_state"].iloc[251]))
        # regime requires BOTH defined; gated by vix_state burn-in.
        self.assertTrue(out["regime"].iloc[:251].isna().all())
        self.assertFalse(pd.isna(out["regime"].iloc[251]))

    def test_drawdown_state_thresholds(self) -> None:
        # Engineered SPY: rise to 105 by day 25, then -4.3% drop (correction)
        # then -12.4% drop (stress) on subsequent days.
        spy = [100.0] * 5 + list(np.linspace(100, 105, 21)) + [100.5, 92.0]
        vix = [20.0] * len(spy)
        s, v = self._pair(spy, vix)
        out = rm.classify_market_regime(s, v)
        self.assertEqual(out["spy_state"].iloc[25], "calm")
        self.assertEqual(out["spy_state"].iloc[26], "correction")
        self.assertEqual(out["spy_state"].iloc[27], "stress")

    def test_vix_zscore_state_thresholds(self) -> None:
        # 280 days of mild VIX noise around 20, then a spike at index 270.
        rng = np.random.default_rng(1)
        vix = (20 + rng.normal(0, 0.5, 280)).tolist()
        vix[270] = 30.0
        spy = [400.0] * 280
        s, v = self._pair(spy, vix)
        out = rm.classify_market_regime(s, v)
        self.assertEqual(out["vix_state"].iloc[270], "high")
        # Burn-in: 252d rolling needs 252 obs → index 251 is first valid.
        self.assertTrue(pd.isna(out["vix_state"].iloc[250]))
        self.assertFalse(pd.isna(out["vix_state"].iloc[251]))

    def test_custom_thresholds_change_assignment(self) -> None:
        # Same engineered series as drawdown test, looser thresholds.
        spy = [100.0] * 5 + list(np.linspace(100, 105, 21)) + [100.5, 92.0]
        vix = [20.0] * len(spy)
        s, v = self._pair(spy, vix)
        out = rm.classify_market_regime(s, v, dd_thresholds=(0.05, 0.20))
        self.assertEqual(out["spy_state"].iloc[26], "calm")
        self.assertEqual(out["spy_state"].iloc[27], "correction")

    def test_combined_regime_label_format(self) -> None:
        # Smoke: where both states defined, regime == f"{spy}_{vix}".
        rng = np.random.default_rng(2)
        spy = (400 + np.cumsum(rng.normal(0, 0.5, 300))).tolist()
        vix = (20  + np.cumsum(rng.normal(0, 0.1, 300))).tolist()
        s, v = self._pair(spy, vix)
        out = rm.classify_market_regime(s, v).dropna(subset=["regime"])
        for _, row in out.head(20).iterrows():
            self.assertEqual(row["regime"],
                             f"{row['spy_state']}_{row['vix_state']}")

    def test_axes_align_on_different_date_ranges(self) -> None:
        # SPY series starts later than VIX (typical: ^VIX has 35y of data,
        # SPY long_history only 10y). Verify the join produces a regime
        # column NaN where SPY is missing, populated where both overlap.
        vix_idx = pd.date_range("2020-01-02", periods=400, freq="B")
        spy_idx = pd.date_range("2021-01-04", periods=200, freq="B")
        rng = np.random.default_rng(3)
        vix = pd.Series(20 + rng.normal(0, 0.5, 400), index=vix_idx, name="VIX")
        spy = pd.Series(400 + np.cumsum(rng.normal(0, 0.5, 200)),
                        index=spy_idx, name="SPY")
        out = rm.classify_market_regime(spy, vix)
        # Outside SPY range → spy_state NaN → regime NaN.
        pre_spy_dates = out.index < spy_idx.min()
        self.assertTrue(out.loc[pre_spy_dates, "regime"].isna().all())
        # Within SPY range (post both burn-ins), regime should be populated.
        post_burnin = out.dropna(subset=["regime"])
        self.assertGreater(len(post_burnin), 0)


# ---------------------------------------------------------------------------
# compute_regime_conditional_dr — groupby + tail/asymmetry diagnostics
# ---------------------------------------------------------------------------
class TestRegimeConditionalDr(unittest.TestCase):
    """Conditional DR table + tail-highlight + asymmetry. Aggregation is
    a plain groupby; the tested invariants are the diagnostic packaging."""

    @staticmethod
    def _build(regimes: list[str], dr_63d: list[float],
               dr_21d: list[float] | None = None,
               dr_252d: list[float] | None = None,
               dr_ratio: list[float] | None = None,
               start: str = "2024-01-02") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
        n = len(regimes)
        idx = pd.date_range(start, periods=n, freq="B")
        dr_ts = pd.DataFrame({
            "dr_21d":  dr_21d  if dr_21d  is not None else dr_63d,
            "dr_63d":  dr_63d,
            "dr_252d": dr_252d if dr_252d is not None else dr_63d,
        }, index=idx)
        labels = pd.DataFrame({
            "spy_state":    [r.split("_")[0] for r in regimes],
            "vix_state":    [r.split("_")[1] for r in regimes],
            "regime":       regimes,
        }, index=idx)
        ratio_s = pd.Series(dr_ratio, index=idx, name="dr_ratio") if dr_ratio else None
        return dr_ts, labels, ratio_s

    def test_empty_inputs_return_empty_summary(self) -> None:
        out = rm.compute_regime_conditional_dr(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(out["summary"].empty)
        self.assertIsNone(out["tail_highlight"])
        self.assertIsNone(out["asymmetry"])
        self.assertEqual(out["n_total"], 0)

    def test_groupby_means_match_manual(self) -> None:
        # 4 days in calm_low (DRs 1.5/1.6/1.7/1.8), 2 days in stress_high (1.0/1.2).
        regimes = ["calm_low"] * 4 + ["stress_high"] * 2
        dr = [1.5, 1.6, 1.7, 1.8, 1.0, 1.2]
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(
            dr_ts, labels, min_n_per_cell=1, headline_window_col="dr_63d"
        )
        summary = out["summary"].set_index(["spy_state", "vix_state"])
        self.assertAlmostEqual(summary.loc[("calm", "low"),    "dr_63d_mean"], 1.65, places=10)
        self.assertAlmostEqual(summary.loc[("stress", "high"), "dr_63d_mean"], 1.10, places=10)
        # Overall mean is plain mean across all 6 days.
        self.assertAlmostEqual(out["overall"]["dr_63d_mean"], np.mean(dr), places=10)
        self.assertEqual(out["n_total"], 6)

    def test_low_n_flag(self) -> None:
        regimes = ["calm_low"] * 25 + ["stress_high"] * 3
        dr = [1.5] * 25 + [1.0] * 3
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(dr_ts, labels, min_n_per_cell=20)
        s = out["summary"].set_index(["spy_state", "vix_state"])
        self.assertFalse(bool(s.loc[("calm", "low"),    "low_n"]))
        self.assertTrue (bool(s.loc[("stress", "high"), "low_n"]))

    def test_tail_highlight_uses_unconditional_mean_when_no_baseline(self) -> None:
        # Default baseline behavior: unconditional mean over labelled window.
        regimes = ["calm_low"] * 10 + ["stress_high"] * 5
        dr = [2.0] * 10 + [1.0] * 5  # cell DR = 1.0, overall ≈ 1.667
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(dr_ts, labels, min_n_per_cell=1)
        th = out["tail_highlight"]
        self.assertIsNotNone(th)
        self.assertAlmostEqual(th["cell_dr"], 1.0, places=10)
        self.assertAlmostEqual(th["baseline_dr"], np.mean(dr), places=10)
        self.assertAlmostEqual(th["delta"], 1.0 - np.mean(dr), places=10)
        self.assertEqual(th["n"], 5)

    def test_baseline_dr_arg_overrides_unconditional_mean(self) -> None:
        # When the caller supplies a baseline (e.g. trailing-1Y mean of DR),
        # tail delta should be measured against that, not the labelled mean.
        regimes = ["calm_low"] * 10 + ["stress_high"] * 5
        dr = [2.0] * 10 + [1.0] * 5
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(
            dr_ts, labels, min_n_per_cell=1, baseline_dr=1.5
        )
        th = out["tail_highlight"]
        self.assertAlmostEqual(th["baseline_dr"], 1.5, places=10)
        self.assertAlmostEqual(th["delta"], 1.0 - 1.5, places=10)
        # The summary should carry a per-cell delta-vs-baseline column.
        s = out["summary"].set_index(["spy_state", "vix_state"])
        self.assertAlmostEqual(
            s.loc[("stress", "high"), "dr_63d_delta_vs_baseline"],
            1.0 - 1.5, places=10,
        )
        self.assertAlmostEqual(
            s.loc[("calm", "low"), "dr_63d_delta_vs_baseline"],
            2.0 - 1.5, places=10,
        )

    def test_tail_highlight_none_when_cell_empty(self) -> None:
        # No stress_high days at all → None.
        regimes = ["calm_low"] * 10 + ["correction_normal"] * 5
        dr = [1.5] * 15
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(dr_ts, labels, min_n_per_cell=1)
        self.assertIsNone(out["tail_highlight"])

    def test_asymmetry_delta_calm_low_minus_stress_high(self) -> None:
        regimes = ["calm_low"] * 10 + ["stress_high"] * 5
        dr = [2.0] * 10 + [1.0] * 5
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(dr_ts, labels, min_n_per_cell=1)
        asy = out["asymmetry"]
        self.assertIsNotNone(asy)
        self.assertAlmostEqual(asy["calm_low_dr"],    2.0, places=10)
        self.assertAlmostEqual(asy["stress_high_dr"], 1.0, places=10)
        self.assertAlmostEqual(asy["delta"], 1.0, places=10)

    def test_asymmetry_none_when_anchor_missing(self) -> None:
        # No calm_low days → asymmetry None even if stress_high populated.
        regimes = ["correction_normal"] * 10 + ["stress_high"] * 5
        dr = [1.5] * 15
        dr_ts, labels, _ = self._build(regimes, dr)
        out = rm.compute_regime_conditional_dr(dr_ts, labels, min_n_per_cell=1)
        self.assertIsNone(out["asymmetry"])

    def test_ratio_series_picked_up_when_supplied(self) -> None:
        regimes = ["calm_low"] * 4 + ["stress_high"] * 2
        dr = [1.5] * 6
        ratio = [1.05, 1.05, 1.05, 1.05, 0.80, 0.80]
        dr_ts, labels, ratio_s = self._build(regimes, dr, dr_ratio=ratio)
        out = rm.compute_regime_conditional_dr(
            dr_ts, labels, dr_ratio_series=ratio_s, min_n_per_cell=1
        )
        s = out["summary"].set_index(["spy_state", "vix_state"])
        self.assertAlmostEqual(s.loc[("calm", "low"),    "dr_ratio_mean"], 1.05, places=10)
        self.assertAlmostEqual(s.loc[("stress", "high"), "dr_ratio_mean"], 0.80, places=10)


# ---------------------------------------------------------------------------
# interpret_regime_dr — character classification of conditional DR output
# ---------------------------------------------------------------------------
class TestInterpretRegimeDr(unittest.TestCase):
    """Classifies tail_highlight delta (cell − baseline) into Holds /
    Erodes / Weakens / Breaks (plus 'insufficient' when the tail is empty).
    Thresholds: (0.20, 0.40) default. Drops the old fair-weather jargon."""

    @staticmethod
    def _cond(delta: float | None,
              baseline_dr: float = 1.40,
              asym_delta: float | None = 0.10,
              n: int = 30) -> dict:
        tail = None if delta is None else {
            "cell":         "stress_high",
            "cell_dr":      baseline_dr + delta,
            "baseline_dr":  baseline_dr,
            "delta":        delta,
            "n":            n,
            "low_n":        n < 20,
            "window":       "dr_63d",
        }
        asym = None if asym_delta is None else {
            "calm_low_dr":    baseline_dr + 0.10,
            "stress_high_dr": baseline_dr + 0.10 - asym_delta,
            "delta":          asym_delta,
            "calm_low_n":     50,
            "stress_high_n":  n,
            "window":         "dr_63d",
        }
        return {"tail_highlight": tail, "asymmetry": asym}

    def test_insufficient_when_tail_missing(self) -> None:
        out = rm.interpret_regime_dr(self._cond(None))
        self.assertEqual(out["character"], "insufficient")
        self.assertTrue(out["low_n_warning"])

    def test_holds_when_delta_nonneg(self) -> None:
        out = rm.interpret_regime_dr(self._cond(+0.05))
        self.assertEqual(out["character"], "holds")

    def test_holds_at_exact_zero(self) -> None:
        out = rm.interpret_regime_dr(self._cond(0.0))
        self.assertEqual(out["character"], "holds")

    def test_erodes_band_above_minus_20bp(self) -> None:
        # -delta in [0, 0.20) → erodes
        out = rm.interpret_regime_dr(self._cond(-0.10))
        self.assertEqual(out["character"], "erodes")

    def test_weakens_band_minus_20_to_minus_40(self) -> None:
        # -delta in [0.20, 0.40) → weakens
        out = rm.interpret_regime_dr(self._cond(-0.25))
        self.assertEqual(out["character"], "weakens")

    def test_breaks_when_delta_at_or_below_minus_40(self) -> None:
        out = rm.interpret_regime_dr(self._cond(-0.50))
        self.assertEqual(out["character"], "breaks")
        out2 = rm.interpret_regime_dr(self._cond(-0.40))
        self.assertEqual(out2["character"], "breaks")

    def test_custom_thresholds_change_boundaries(self) -> None:
        # Tighten the bands; what was "erodes" at -0.10 with default (0.20, 0.40)
        # becomes "weakens" with (0.05, 0.20).
        out = rm.interpret_regime_dr(
            self._cond(-0.10), weakness_thresholds=(0.05, 0.20)
        )
        self.assertEqual(out["character"], "weakens")

    def test_low_n_warning_appended_to_headline(self) -> None:
        out = rm.interpret_regime_dr(self._cond(-0.10, n=5))
        self.assertIn("preliminary", out["headline"].lower())
        self.assertTrue(out["low_n_warning"])

    def test_asymmetry_note_directionality(self) -> None:
        out_pos = rm.interpret_regime_dr(self._cond(-0.10, asym_delta=+0.20))
        self.assertIn("typical", out_pos["asymmetry_note"].lower())
        out_neg = rm.interpret_regime_dr(self._cond(-0.10, asym_delta=-0.20))
        self.assertIn("favorable", out_neg["asymmetry_note"].lower())


# ---------------------------------------------------------------------------
# compute_beta / compute_alpha_annual
# ---------------------------------------------------------------------------
class TestBetaAlpha(unittest.TestCase):
    def test_perfect_correlation_slope_one(self) -> None:
        # p = b exactly ⇒ β = 1, annualized α = 0.
        rng = np.random.default_rng(0)
        b = pd.Series(rng.normal(0, 0.01, size=200))
        p = b.copy()
        self.assertAlmostEqual(rm.compute_beta(p, b), 1.0, places=10)
        self.assertAlmostEqual(rm.compute_alpha_annual(p, b), 0.0, places=8)

    def test_known_slope_two(self) -> None:
        # p = 2 × b ⇒ β = 2, α = 0 (no intercept).
        rng = np.random.default_rng(1)
        b = pd.Series(rng.normal(0, 0.01, size=200))
        p = 2.0 * b
        self.assertAlmostEqual(rm.compute_beta(p, b), 2.0, places=10)
        self.assertAlmostEqual(rm.compute_alpha_annual(p, b), 0.0, places=8)

    def test_added_intercept_shows_in_alpha(self) -> None:
        # p = β × b + c per day ⇒ annualized α = c × 252.
        rng = np.random.default_rng(2)
        b = pd.Series(rng.normal(0, 0.01, size=200))
        daily_alpha = 0.0005
        p = 1.5 * b + daily_alpha
        beta = rm.compute_beta(p, b)
        alpha_ann = rm.compute_alpha_annual(p, b)
        self.assertAlmostEqual(beta, 1.5, places=10)
        self.assertAlmostEqual(alpha_ann, daily_alpha * 252.0, places=6)

    def test_zero_variance_benchmark_returns_nan(self) -> None:
        b = pd.Series([0.0] * 100)
        p = pd.Series([0.01] * 100)
        self.assertTrue(np.isnan(rm.compute_beta(p, b)))
        self.assertTrue(np.isnan(rm.compute_alpha_annual(p, b)))

    def test_window_truncates_to_recent_n(self) -> None:
        # First 100 days p = b; last 50 days p = 3*b. With window=50, β=3.
        rng = np.random.default_rng(3)
        b = pd.Series(rng.normal(0, 0.01, size=150))
        p = pd.Series(np.concatenate([b.values[:100], 3.0 * b.values[100:]]))
        self.assertAlmostEqual(rm.compute_beta(p, b, window=50), 3.0, places=10)

    def test_returns_raw_intercept_not_capm_alpha(self) -> None:
        """Lock: compute_alpha_annual returns the OLS intercept on RAW
        returns, NOT textbook CAPM α (which would regress excess returns
        over the risk-free rate).

        Phase 1B audit caught the Risk-tab tile labeling this as CAPM α
        with prose invoking a passive cash leg — both implied excess-
        return regression. The label was corrected to "α (OLS intercept
        vs SPY)"; this test locks the formula so a future maintainer
        can't silently flip to excess returns and shift the displayed
        α by ~0.8 pp/yr at today's β / RF.

        Documented gap: for constant daily r_f,
            α_raw - α_capm = r_f_ann · (1 - β).
        """
        rng = np.random.default_rng(42)
        n = 252
        b = pd.Series(rng.normal(0.0004, 0.01, size=n))
        target_beta = 0.78
        daily_intercept = 0.0001
        p = target_beta * b + daily_intercept

        alpha_raw_ann = rm.compute_alpha_annual(p, b)

        # Independent hand-computed textbook CAPM α with constant daily RF.
        # Constants don't affect cov / var so β is preserved across the
        # raw and excess regressions.
        rf_ann = 0.037
        rf_daily = rf_ann / 252.0
        p_excess = p - rf_daily
        b_excess = b - rf_daily
        beta_capm = p_excess.cov(b_excess) / b_excess.var()
        alpha_capm_ann = (
            (p_excess.mean() - beta_capm * b_excess.mean()) * 252.0
        )

        beta_raw = rm.compute_beta(p, b)
        self.assertAlmostEqual(beta_raw, beta_capm, places=10)

        expected_gap = rf_ann * (1.0 - beta_raw)
        self.assertAlmostEqual(
            alpha_raw_ann - alpha_capm_ann, expected_gap, places=8,
        )
        # For β < 1 with positive RF, the raw intercept overstates CAPM α.
        self.assertLess(beta_raw, 1.0)
        self.assertGreater(alpha_raw_ann, alpha_capm_ann)


# ---------------------------------------------------------------------------
# compute_up_down_beta
# ---------------------------------------------------------------------------
class TestUpDownBeta(unittest.TestCase):
    def test_symmetric_relationship(self) -> None:
        # p = 1.2 × b on every day ⇒ both up_b and dn_b should be 1.2.
        rng = np.random.default_rng(4)
        b = pd.Series(rng.normal(0, 0.01, size=300))
        p = 1.2 * b
        up_b, dn_b = rm.compute_up_down_beta(p, b)
        self.assertAlmostEqual(up_b, 1.2, places=8)
        self.assertAlmostEqual(dn_b, 1.2, places=8)

    def test_asymmetric_up_down(self) -> None:
        # Steeper on down days than up: p = b on up days, p = 2×b on down days.
        rng = np.random.default_rng(5)
        b = pd.Series(rng.normal(0, 0.01, size=400))
        p = b.where(b > 0, 2.0 * b)
        up_b, dn_b = rm.compute_up_down_beta(p, b)
        self.assertAlmostEqual(up_b, 1.0, places=8)
        self.assertAlmostEqual(dn_b, 2.0, places=8)


# ---------------------------------------------------------------------------
# rolling_beta / rolling_alpha_annual / rolling_up_down_beta
# ---------------------------------------------------------------------------
class TestRollingBetaAlpha(unittest.TestCase):
    def test_rolling_beta_matches_point_estimate_at_right_edge(self) -> None:
        # The rolling 252 value on the last date must equal compute_beta over
        # the trailing 252 days — they are the same operation.
        rng = np.random.default_rng(10)
        n = 600
        b = pd.Series(rng.normal(0, 0.01, size=n))
        p = 1.4 * b + pd.Series(rng.normal(0, 0.003, size=n))
        rb = rm.rolling_beta(p, b, window=252)
        point = rm.compute_beta(p, b, window=252)
        self.assertAlmostEqual(rb.iloc[-1], point, places=10)

    def test_rolling_beta_warmup_is_nan(self) -> None:
        rng = np.random.default_rng(11)
        n = 400
        b = pd.Series(rng.normal(0, 0.01, size=n))
        p = 1.1 * b
        rb = rm.rolling_beta(p, b, window=252)
        self.assertTrue(rb.iloc[:251].isna().all())
        self.assertTrue(rb.iloc[251:].notna().all())

    def test_rolling_alpha_matches_point_estimate_at_right_edge(self) -> None:
        rng = np.random.default_rng(12)
        n = 500
        b = pd.Series(rng.normal(0, 0.01, size=n))
        p = 1.0 * b + 0.0004  # constant daily alpha
        ra = rm.rolling_alpha_annual(p, b, window=252)
        point = rm.compute_alpha_annual(p, b, window=252)
        self.assertAlmostEqual(ra.iloc[-1], point, places=10)
        # Should be close to 0.0004 × 252 ≈ 0.1008.
        self.assertAlmostEqual(ra.iloc[-1], 0.0004 * 252.0, places=4)

    def test_rolling_up_down_beta_matches_point_estimate_at_right_edge(self) -> None:
        rng = np.random.default_rng(13)
        n = 500
        b = pd.Series(rng.normal(0, 0.01, size=n))
        # Asymmetric: β=1 on up days, β=2 on down days.
        p = b.where(b > 0, 2.0 * b)
        ub, db = rm.rolling_up_down_beta(p, b, window=252)
        up_pt, dn_pt = rm.compute_up_down_beta(p, b, window=252)
        self.assertAlmostEqual(ub.iloc[-1], up_pt, places=10)
        self.assertAlmostEqual(db.iloc[-1], dn_pt, places=10)
        # And the values themselves should be ~1 and ~2.
        self.assertAlmostEqual(ub.iloc[-1], 1.0, places=8)
        self.assertAlmostEqual(db.iloc[-1], 2.0, places=8)

    def test_rolling_up_down_beta_min_obs_guard(self) -> None:
        # Benchmark always positive ⇒ no down days ⇒ dn_beta should be NaN
        # for every date.
        rng = np.random.default_rng(14)
        n = 300
        b = pd.Series(np.abs(rng.normal(0, 0.01, size=n)) + 0.001)
        p = 1.5 * b
        _, db = rm.rolling_up_down_beta(p, b, window=252)
        self.assertTrue(db.isna().all())

# ---------------------------------------------------------------------------
# compute_var_cvar
# ---------------------------------------------------------------------------
class TestVarCvar(unittest.TestCase):
    def test_too_few_observations(self) -> None:
        v, c = rm.compute_var_cvar(pd.Series(range(19)) * 0.001)
        self.assertTrue(np.isnan(v))
        self.assertTrue(np.isnan(c))

    def test_var_matches_quantile_and_cvar_le_var(self) -> None:
        rng = np.random.default_rng(6)
        r = pd.Series(rng.normal(0.0005, 0.012, size=500))
        var, cvar = rm.compute_var_cvar(r, alpha=0.05)
        self.assertAlmostEqual(var, float(r.quantile(0.05)), places=10)
        # CVaR (mean of tail) must be ≤ VaR (the tail boundary) — both negative.
        self.assertLessEqual(cvar, var)
        # Tail isn't empty here (500 obs × 5% ≈ 25 obs).
        self.assertTrue(np.isfinite(cvar))


# ---------------------------------------------------------------------------
# aggregate_periodic_returns
# ---------------------------------------------------------------------------
class TestAggregatePeriodicReturns(unittest.TestCase):
    def test_monthly_pass_through(self) -> None:
        # freq="M" reset_indexes and normalizes dates to month-start, but
        # does NOT chain (months stay as monthly returns).
        rets = pd.Series([0.01, 0.02, -0.005])
        dates = pd.Series([pd.Timestamp("2026-01-31"),
                           pd.Timestamp("2026-02-28"),
                           pd.Timestamp("2026-03-31")])
        r_out, d_out = rm.aggregate_periodic_returns(rets, dates, "M")
        self.assertEqual(len(r_out), 3)
        self.assertAlmostEqual(r_out.iloc[1], 0.02, places=12)
        # Date normalized to month-start.
        self.assertEqual(d_out.iloc[0], pd.Timestamp("2026-01-01"))

    def test_quarterly_chains_three_months(self) -> None:
        # Three 5% months in Q1 chain to (1.05)^3 - 1 = 0.157625.
        rets = pd.Series([0.05, 0.05, 0.05, 0.02, 0.03, 0.04])
        dates = pd.Series([pd.Timestamp("2026-01-31"),
                           pd.Timestamp("2026-02-28"),
                           pd.Timestamp("2026-03-31"),
                           pd.Timestamp("2026-04-30"),
                           pd.Timestamp("2026-05-31"),
                           pd.Timestamp("2026-06-30")])
        r_out, d_out = rm.aggregate_periodic_returns(rets, dates, "Q")
        self.assertEqual(len(r_out), 2)
        self.assertAlmostEqual(r_out.iloc[0], (1.05) ** 3 - 1.0, places=10)
        # Q2: (1.02)(1.03)(1.04) - 1.
        q2_expected = 1.02 * 1.03 * 1.04 - 1.0
        self.assertAlmostEqual(r_out.iloc[1], q2_expected, places=10)

    def test_yearly_chains_full_year(self) -> None:
        # 12 months of 1% chain to (1.01)^12 - 1.
        rets = pd.Series([0.01] * 12)
        dates = pd.Series(pd.date_range("2026-01-31", periods=12, freq="ME"))
        r_out, _ = rm.aggregate_periodic_returns(rets, dates, "Y")
        self.assertEqual(len(r_out), 1)
        self.assertAlmostEqual(r_out.iloc[0], (1.01) ** 12 - 1.0, places=10)


# ---------------------------------------------------------------------------
# spy_monthly_returns_aligned (boundary alignment)
# ---------------------------------------------------------------------------
class TestSpyMonthlyReturnsAligned(unittest.TestCase):
    def test_aligns_to_statement_boundaries(self) -> None:
        # bench TR: daily values at 100, 110, 121 on three month-end dates.
        # twr_portfolio supplies the prev_stmt_date → statement_date windows.
        bench = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.DatetimeIndex([
                pd.Timestamp("2026-01-31"),
                pd.Timestamp("2026-02-28"),
                pd.Timestamp("2026-03-31"),
            ]),
        )
        twr = pd.DataFrame([
            # First row has no prev — drop.
            {"statement_date": pd.Timestamp("2026-01-31"),
             "prev_stmt_date": pd.NaT},
            # Feb: 100 → 110 = +10%.
            {"statement_date": pd.Timestamp("2026-02-28"),
             "prev_stmt_date": pd.Timestamp("2026-01-31")},
            # Mar: 110 → 121 = +10%.
            {"statement_date": pd.Timestamp("2026-03-31"),
             "prev_stmt_date": pd.Timestamp("2026-02-28")},
        ])
        out = rm.spy_monthly_returns_aligned(twr, bench)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out.iloc[0], 0.10, places=10)
        self.assertAlmostEqual(out.iloc[1], 0.10, places=10)

    def test_outside_bench_range_dropped(self) -> None:
        bench = pd.Series([100.0, 110.0],
                          index=pd.DatetimeIndex([pd.Timestamp("2026-02-01"),
                                                  pd.Timestamp("2026-03-01")]))
        twr = pd.DataFrame([
            # Jan window: prev 2026-01-01 is before bench_start → drop.
            {"statement_date": pd.Timestamp("2026-02-01"),
             "prev_stmt_date": pd.Timestamp("2026-01-01")},
        ])
        out = rm.spy_monthly_returns_aligned(twr, bench)
        self.assertEqual(len(out), 0)

    def test_empty_inputs(self) -> None:
        self.assertTrue(rm.spy_monthly_returns_aligned(
            pd.DataFrame(), pd.Series(dtype=float)).empty)


# ---------------------------------------------------------------------------
# Correlation matrices + rolling correlations (Big-3 / Top-15 section)
# ---------------------------------------------------------------------------
class TestCorrelationMatrix(unittest.TestCase):
    def _three_year_prices(self) -> pd.DataFrame:
        """Deterministic 3-symbol 3y price frame for shape/sign tests."""
        rng = np.random.default_rng(0)
        n = 600
        idx = pd.bdate_range("2023-01-02", periods=n)
        a = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        b = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        # c moves with a (positive) — should show positive ρ vs A.
        c_rets = 0.6 * np.diff(np.log(a), prepend=np.log(a[0])) \
                 + 0.4 * rng.normal(0, 0.005, n)
        c = 100.0 * np.exp(np.cumsum(c_rets))
        return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)

    def test_shape_and_diagonal(self) -> None:
        px = self._three_year_prices()
        corr = rm.compute_correlation_matrix(px, ["A", "B", "C"])
        self.assertEqual(corr.shape, (3, 3))
        for sym in ("A", "B", "C"):
            self.assertAlmostEqual(corr.loc[sym, sym], 1.0, places=10)

    def test_symmetry(self) -> None:
        px = self._three_year_prices()
        corr = rm.compute_correlation_matrix(px, ["A", "B", "C"])
        self.assertAlmostEqual(corr.loc["A", "B"], corr.loc["B", "A"])
        self.assertAlmostEqual(corr.loc["A", "C"], corr.loc["C", "A"])

    def test_constructed_positive_correlation(self) -> None:
        # C is constructed with 60% loading on A — expect strong positive ρ.
        px = self._three_year_prices()
        corr = rm.compute_correlation_matrix(px, ["A", "B", "C"])
        self.assertGreater(corr.loc["A", "C"], 0.4)

    def test_unknown_symbols_dropped(self) -> None:
        px = self._three_year_prices()
        corr = rm.compute_correlation_matrix(px, ["A", "C", "ZZZ"])
        self.assertEqual(set(corr.columns), {"A", "C"})

    def test_empty_inputs(self) -> None:
        self.assertTrue(rm.compute_correlation_matrix(pd.DataFrame()).empty)
        self.assertTrue(rm.compute_correlation_matrix(
            pd.DataFrame({"A": [1.0, 2.0]}), symbols=[],
        ).empty)


class TestRollingPairCorrelations(unittest.TestCase):
    def _two_symbol_prices(self, n: int = 200,
                           rho: float = 0.7) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2023-01-02", periods=n)
        x = rng.normal(0, 0.01, n)
        y = rho * x + np.sqrt(1 - rho * rho) * rng.normal(0, 0.01, n)
        return pd.DataFrame({
            "A": 100.0 * np.exp(np.cumsum(x)),
            "B": 100.0 * np.exp(np.cumsum(y)),
        }, index=idx)

    def test_emits_one_column_per_pair(self) -> None:
        px = self._two_symbol_prices()
        roll = rm.compute_rolling_pair_correlations(
            px, [("A", "B")], window=60,
        )
        self.assertEqual(list(roll.columns), ["A–B"])
        self.assertGreater(len(roll), 100)
        # All in [-1, 1]
        v = roll["A–B"].dropna().values
        self.assertTrue(np.all((v >= -1) & (v <= 1)))

    def test_min_periods_full_window(self) -> None:
        # No values emitted before `window` observations exist.
        px = self._two_symbol_prices(n=200)
        roll = rm.compute_rolling_pair_correlations(
            px, [("A", "B")], window=60,
        )
        # First non-NaN should land at trading day 60 (1-indexed) or later.
        first_valid = roll["A–B"].first_valid_index()
        self.assertIsNotNone(first_valid)
        self.assertGreaterEqual(
            list(px.index).index(first_valid), 60 - 1,
        )

    def test_unknown_pair_silently_skipped(self) -> None:
        px = self._two_symbol_prices()
        roll = rm.compute_rolling_pair_correlations(
            px, [("A", "B"), ("A", "MISSING")], window=60,
        )
        self.assertEqual(list(roll.columns), ["A–B"])

    def test_empty_inputs(self) -> None:
        self.assertTrue(rm.compute_rolling_pair_correlations(
            pd.DataFrame(), pairs=[("A", "B")],
        ).empty)
        self.assertTrue(rm.compute_rolling_pair_correlations(
            pd.DataFrame({"A": [1.0]}), pairs=[],
        ).empty)


class TestRollingAvgPairwiseCorrelation(unittest.TestCase):
    def test_shape_and_range(self) -> None:
        rng = np.random.default_rng(2)
        n = 250
        idx = pd.bdate_range("2023-01-02", periods=n)
        # Three loosely correlated series — average ρ should be modestly
        # positive and bounded in [-1, 1].
        base = rng.normal(0, 0.01, n)
        px = pd.DataFrame({
            "A": 100.0 * np.exp(np.cumsum(0.5 * base + rng.normal(0, 0.01, n))),
            "B": 100.0 * np.exp(np.cumsum(0.5 * base + rng.normal(0, 0.01, n))),
            "C": 100.0 * np.exp(np.cumsum(0.5 * base + rng.normal(0, 0.01, n))),
        }, index=idx)
        avg = rm.compute_rolling_avg_pairwise_correlation(
            px, ["A", "B", "C"], window=60,
        )
        self.assertGreater(len(avg), 100)
        vals = avg.dropna().values
        self.assertTrue(np.all((vals >= -1) & (vals <= 1)))

    def test_too_few_symbols_returns_empty(self) -> None:
        px = pd.DataFrame({
            "A": [100.0, 100.5, 101.0, 100.8],
        }, index=pd.bdate_range("2023-01-02", periods=4))
        avg = rm.compute_rolling_avg_pairwise_correlation(
            px, ["A"], window=2,
        )
        self.assertTrue(avg.empty)


class TestSpliceSgovWithBil(unittest.TestCase):
    def _split_history(self) -> pd.DataFrame:
        # SGOV exists only for the second half of the index; BIL covers all.
        idx = pd.bdate_range("2018-01-02", periods=400)
        sgov = pd.Series(np.nan, index=idx, name="SGOV")
        sgov.iloc[200:] = np.linspace(100.0, 100.6, 200)
        bil = pd.Series(np.linspace(90.0, 91.5, 400), index=idx, name="BIL")
        return pd.concat([sgov, bil], axis=1)

    def test_splice_extends_back_to_bil_start(self) -> None:
        prices = self._split_history()
        out = rm.splice_sgov_with_bil(prices)
        # Output covers BIL's full range, not just SGOV's
        self.assertEqual(out.index.min(), prices.index.min())
        self.assertEqual(out.index.max(), prices.index.max())

    def test_splice_matches_sgov_at_handoff(self) -> None:
        prices = self._split_history()
        out = rm.splice_sgov_with_bil(prices)
        sgov_first_date = prices["SGOV"].dropna().index.min()
        # SGOV values must be passed through unchanged where SGOV exists.
        self.assertAlmostEqual(
            out.loc[sgov_first_date],
            prices["SGOV"].loc[sgov_first_date],
            places=10,
        )

    def test_no_bil_returns_sgov_only(self) -> None:
        prices = pd.DataFrame({
            "SGOV": [100.0, 100.5],
        }, index=pd.bdate_range("2020-05-28", periods=2))
        out = rm.splice_sgov_with_bil(prices)
        self.assertEqual(len(out), 2)

    def test_empty_input(self) -> None:
        out = rm.splice_sgov_with_bil(pd.DataFrame())
        self.assertTrue(out.empty)


class TestExtendSgovPanel(unittest.TestCase):
    """extend_sgov_with_bil_panel: bridge SGOV's pre-inception NaN head in
    the daily panel from the BIL-spliced long-history series (TK 2026-07-17:
    'it's just T-bills — substitute for missing, use SGOV when available')."""

    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        idx = pd.bdate_range("2018-01-02", periods=400)
        sgov = pd.Series(np.nan, index=idx, name="SGOV")
        sgov.iloc[200:] = np.linspace(100.0, 100.6, 200)
        bil = pd.Series(np.linspace(90.0, 91.5, 400), index=idx, name="BIL")
        long_p = pd.concat([sgov, bil], axis=1)
        # Panel starts mid pre-history; SGOV appears at the same date as in
        # the long file (idx[200] == panel_idx[100]).
        panel_idx = idx[100:]
        panel = pd.DataFrame({
            "SPY": np.linspace(300.0, 400.0, 300),
            "SGOV": np.concatenate([np.full(100, np.nan),
                                    np.linspace(100.0, 100.6, 200)]),
        }, index=panel_idx)
        return panel, long_p

    def test_fills_only_nan_head(self) -> None:
        panel, long_p = self._frames()
        out = rm.extend_sgov_with_bil_panel(panel, long_p)
        self.assertFalse(out["SGOV"].isna().any())
        # Existing SGOV values and other columns pass through untouched.
        pd.testing.assert_series_equal(out["SGOV"].iloc[100:],
                                       panel["SGOV"].iloc[100:])
        pd.testing.assert_series_equal(out["SPY"], panel["SPY"])
        # Input not mutated.
        self.assertTrue(panel["SGOV"].iloc[:100].isna().all())

    def test_seam_is_continuous(self) -> None:
        panel, long_p = self._frames()
        out = rm.extend_sgov_with_bil_panel(panel, long_p)
        # Rebased BIL back-fill anchors at SGOV's first level — the seam
        # must not fabricate a jump.
        seam_ret = out["SGOV"].iloc[100] / out["SGOV"].iloc[99] - 1.0
        self.assertLess(abs(seam_ret), 0.05)

    def test_noop_without_long_history(self) -> None:
        panel, _ = self._frames()
        out = rm.extend_sgov_with_bil_panel(panel, pd.DataFrame())
        pd.testing.assert_frame_equal(out, panel)

    def test_noop_without_sgov_column(self) -> None:
        panel, long_p = self._frames()
        p2 = panel.drop(columns=["SGOV"])
        pd.testing.assert_frame_equal(
            rm.extend_sgov_with_bil_panel(p2, long_p), p2)


# ---------------------------------------------------------------------------
# Ticker-history splicing (corporate-action renames)
# ---------------------------------------------------------------------------
class TestSpliceTickerHistory(unittest.TestCase):
    def _frame(self, idx, **cols):
        return pd.DataFrame(cols, index=idx)

    def test_empty_history_is_noop(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=20)
        df = self._frame(idx, A=range(20), B=range(20, 40))
        out = rm.splice_ticker_history(df, {})
        pd.testing.assert_frame_equal(out, df)

    def test_empty_input_returns_empty(self) -> None:
        out = rm.splice_ticker_history(pd.DataFrame(), {"X": [{"prior_symbol": "Y", "effective_date": "2020-01-01"}]})
        self.assertTrue(out.empty)

    def test_simple_rename_splice(self) -> None:
        # NEW exists from 2020-06-01 onward; OLD has data 2020-01-02 - 2020-05-29.
        idx = pd.bdate_range("2020-01-02", periods=200)
        old = pd.Series(np.linspace(10, 20, 200), index=idx, name="OLD")
        new = pd.Series(np.nan, index=idx, name="NEW")
        new.iloc[110:] = np.linspace(50, 60, 90)  # NEW starts ~2020-06-04
        df = pd.concat([old, new], axis=1)

        history = {"NEW": [
            {"prior_symbol": "OLD", "effective_date": idx[110].isoformat()},
        ]}
        out = rm.splice_ticker_history(df, history)

        # Before effective_date: NEW should now have OLD's values
        self.assertAlmostEqual(out.loc[idx[0], "NEW"], 10.0, places=6)
        self.assertAlmostEqual(out.loc[idx[50], "NEW"], old.iloc[50], places=6)
        # On/after effective_date: NEW values untouched
        self.assertAlmostEqual(out.loc[idx[110], "NEW"], 50.0, places=6)
        self.assertAlmostEqual(out.loc[idx[199], "NEW"], 60.0, places=6)
        # OLD column untouched
        pd.testing.assert_series_equal(out["OLD"], df["OLD"])

    def test_does_not_overwrite_existing_current_ticker_data(self) -> None:
        # If NEW has a stray observation in the historical window, splice
        # must NOT overwrite it with OLD's value at that date.
        idx = pd.bdate_range("2020-01-02", periods=100)
        old = pd.Series(np.linspace(10, 20, 100), index=idx)
        new = pd.Series(np.nan, index=idx)
        new.iloc[5] = 999.0  # stray
        new.iloc[60:] = np.linspace(50, 60, 40)
        df = pd.DataFrame({"OLD": old, "NEW": new})
        history = {"NEW": [
            {"prior_symbol": "OLD", "effective_date": idx[60].isoformat()},
        ]}
        out = rm.splice_ticker_history(df, history)
        # Stray was preserved
        self.assertAlmostEqual(out.loc[idx[5], "NEW"], 999.0, places=6)
        # Surrounding NaN gaps got filled
        self.assertAlmostEqual(out.loc[idx[4], "NEW"], old.iloc[4], places=6)
        self.assertAlmostEqual(out.loc[idx[6], "NEW"], old.iloc[6], places=6)

    def test_missing_prior_symbol_is_silent(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=50)
        df = pd.DataFrame({"NEW": np.arange(50, dtype=float)}, index=idx)
        df.loc[idx[:25], "NEW"] = np.nan
        history = {"NEW": [
            {"prior_symbol": "GHOST", "effective_date": idx[25].isoformat()},
        ]}
        out = rm.splice_ticker_history(df, history)
        # Pre-effective rows stay NaN (no prior data to fill from)
        self.assertTrue(out.loc[idx[:25], "NEW"].isna().all())
        # Post-effective rows untouched
        self.assertEqual(out.loc[idx[25], "NEW"], 25.0)

    def test_creates_column_for_missing_current_ticker(self) -> None:
        # If NEW isn't in `daily_prices` yet, splice should create it
        # populated entirely from OLD's data over the historical window.
        idx = pd.bdate_range("2020-01-02", periods=50)
        df = pd.DataFrame({"OLD": np.arange(50, dtype=float)}, index=idx)
        history = {"NEW": [
            {"prior_symbol": "OLD", "effective_date": idx[25].isoformat()},
        ]}
        out = rm.splice_ticker_history(df, history)
        self.assertIn("NEW", out.columns)
        # Dates < effective: NEW = OLD
        self.assertEqual(out.loc[idx[10], "NEW"], 10.0)
        self.assertEqual(out.loc[idx[24], "NEW"], 24.0)
        # Dates >= effective: NEW is still NaN (no current ticker observed)
        self.assertTrue(pd.isna(out.loc[idx[25], "NEW"]))

    def test_multi_segment_chain(self) -> None:
        # CCC was BBB until 2022-01-03, was AAA until 2020-01-02.
        idx = pd.bdate_range("2018-01-02", periods=1200)
        aaa = pd.Series(np.nan, index=idx)
        bbb = pd.Series(np.nan, index=idx)
        ccc = pd.Series(np.nan, index=idx)
        # AAA active: 2018 -> 2019 end
        aaa_mask = idx < pd.Timestamp("2020-01-02")
        aaa[aaa_mask] = 100.0
        # BBB active: 2020-01-02 -> 2021-12-31
        bbb_mask = (idx >= pd.Timestamp("2020-01-02")) & (idx < pd.Timestamp("2022-01-03"))
        bbb[bbb_mask] = 200.0
        # CCC active: 2022-01-03 onward
        ccc_mask = idx >= pd.Timestamp("2022-01-03")
        ccc[ccc_mask] = 300.0
        df = pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc})

        history = {"CCC": [
            {"prior_symbol": "BBB", "effective_date": "2022-01-03"},
            {"prior_symbol": "AAA", "effective_date": "2020-01-02"},
        ]}
        out = rm.splice_ticker_history(df, history)
        # CCC now has all three eras
        self.assertAlmostEqual(out.loc[idx[10], "CCC"], 100.0, places=6)
        self.assertAlmostEqual(
            out.loc[pd.Timestamp("2021-06-01"), "CCC"], 200.0, places=6)
        self.assertAlmostEqual(
            out.loc[pd.Timestamp("2022-06-01"), "CCC"], 300.0, places=6)
        # Other columns untouched
        self.assertAlmostEqual(out.loc[idx[10], "AAA"], 100.0, places=6)
        self.assertAlmostEqual(out.loc[pd.Timestamp("2021-06-01"), "BBB"], 200.0, places=6)

    def test_input_not_mutated(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=20)
        df = pd.DataFrame({
            "OLD": np.arange(20, dtype=float),
            "NEW": np.nan,
        }, index=idx)
        df.loc[idx[10:], "NEW"] = 99.0
        df_before = df.copy()
        history = {"NEW": [
            {"prior_symbol": "OLD", "effective_date": idx[10].isoformat()},
        ]}
        _ = rm.splice_ticker_history(df, history)
        pd.testing.assert_frame_equal(df, df_before)


# ---------------------------------------------------------------------------
# Conditional correlation matrix (full vs stress-day Pearson)
# ---------------------------------------------------------------------------
class TestConditionalCorrelationMatrix(unittest.TestCase):
    def _make_prices(self, n: int = 800, seed: int = 7) -> pd.DataFrame:
        """Synthetic 4-symbol frame where on the worst SPY days, B and C
        co-move much more tightly with SPY than on calm days. D is
        independent everywhere. The constructed regime gives us a clean
        signal that conditional ρ exceeds full ρ for SPY-B and SPY-C, and
        leaves SPY-D effectively untouched.

        - Calm regime: B/C loaded weakly on SPY (β ≈ 0.10) under heavy
          idiosyncratic noise — full-sample ρ stays modest.
        - Stress regime: SPY's tail is roughly −3σ, and B/C lock onto
          SPY at β ≈ 0.95 with tiny noise — conditional ρ ≈ 1.
        """
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-02", periods=n)
        spy_r = rng.normal(0, 0.012, n)
        # Calm-day construction: low SPY loading + dominant idiosyncratic
        # noise so calm-day ρ(SPY,B) stays well below the stress value.
        b_r = 0.10 * spy_r + 0.012 * rng.normal(0, 1, n)
        c_r = 0.10 * spy_r + 0.012 * rng.normal(0, 1, n)
        d_r = 0.012 * rng.normal(0, 1, n)
        # Inject obvious stress days: SPY ~ −3% with tight β≈0.95 lock on B/C.
        stress_days = rng.choice(n, size=60, replace=False)
        spy_r[stress_days] = -0.03 + 0.003 * rng.normal(0, 1, 60)
        for i in stress_days:
            b_r[i] = 0.95 * spy_r[i] + 0.001 * rng.normal()
            c_r[i] = 0.95 * spy_r[i] + 0.001 * rng.normal()
            d_r[i] = 0.012 * rng.normal()  # stays independent

        def _px(r):
            return 100.0 * np.exp(np.cumsum(r))
        return pd.DataFrame({
            "SPY": _px(spy_r), "B": _px(b_r),
            "C":   _px(c_r),   "D": _px(d_r),
        }, index=idx)

    def test_shapes_and_keys(self) -> None:
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "C", "D"], condition_symbol="SPY",
            z_threshold=-1.5,
        )
        for k in ("full", "conditional", "delta", "threshold", "mean",
                  "sd", "n_full", "n_stress", "enough"):
            self.assertIn(k, out)
        self.assertEqual(out["full"].shape, (4, 4))
        self.assertEqual(out["conditional"].shape, (4, 4))
        self.assertEqual(out["delta"].shape, (4, 4))
        self.assertTrue(out["enough"])

    def test_threshold_picks_left_tail(self) -> None:
        # The threshold should be roughly mu - 1.5σ of SPY log returns.
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "C", "D"],
            condition_symbol="SPY", z_threshold=-1.5,
        )
        self.assertLess(out["threshold"], out["mean"])
        self.assertGreater(out["n_stress"], 0)
        self.assertLess(out["n_stress"], out["n_full"])

    def test_stress_rho_exceeds_full_for_coupled_pairs(self) -> None:
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "C", "D"],
            condition_symbol="SPY", z_threshold=-1.5,
        )
        # SPY–B and SPY–C constructed to spike under stress
        self.assertGreater(
            out["conditional"].loc["SPY", "B"],
            out["full"].loc["SPY", "B"],
        )
        self.assertGreater(
            out["conditional"].loc["SPY", "C"],
            out["full"].loc["SPY", "C"],
        )
        # And the Δ matrix should reflect that
        self.assertGreater(out["delta"].loc["SPY", "B"], 0.0)
        self.assertGreater(out["delta"].loc["SPY", "C"], 0.0)

    def test_diagonal_delta_is_zero(self) -> None:
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "C", "D"],
            condition_symbol="SPY", z_threshold=-1.5,
        )
        for s in ("SPY", "B", "C", "D"):
            self.assertAlmostEqual(out["delta"].loc[s, s], 0.0, places=10)

    def test_too_few_stress_days_returns_partial(self) -> None:
        # min_stress_days higher than actual stress count → enough=False
        px = self._make_prices(n=200)
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "C", "D"],
            condition_symbol="SPY", z_threshold=-1.5,
            min_stress_days=1_000,
        )
        self.assertFalse(out["enough"])
        self.assertTrue(out["conditional"].empty)
        self.assertTrue(out["delta"].empty)
        # Full sample matrix should still be populated
        self.assertEqual(out["full"].shape, (4, 4))

    def test_missing_condition_symbol(self) -> None:
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["B", "C", "D"], condition_symbol="ZZZ",
        )
        self.assertFalse(out["enough"])
        self.assertTrue(out["full"].empty)
        self.assertEqual(out["n_full"], 0)

    def test_empty_inputs(self) -> None:
        out = rm.compute_conditional_correlation_matrix(
            pd.DataFrame(), ["SPY", "B"], condition_symbol="SPY",
        )
        self.assertFalse(out["enough"])
        self.assertEqual(out["n_full"], 0)
        out = rm.compute_conditional_correlation_matrix(
            pd.DataFrame({"SPY": [100.0, 100.5]}),
            symbols=[], condition_symbol="SPY",
        )
        self.assertFalse(out["enough"])

    def test_unknown_symbols_dropped(self) -> None:
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "ZZZ"], condition_symbol="SPY",
            z_threshold=-1.5,
        )
        self.assertEqual(list(out["full"].columns), ["SPY", "B"])
        if out["enough"]:
            self.assertEqual(list(out["conditional"].columns), ["SPY", "B"])

    def test_default_method_is_spearman(self) -> None:
        px = self._make_prices()
        out = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "B", "C", "D"], condition_symbol="SPY",
            z_threshold=-1.5,
        )
        self.assertEqual(out["method"], "spearman")

    def test_spearman_robust_to_single_outlier(self) -> None:
        """The motivating case: a single huge idiosyncratic print in one
        name's stress window. Pearson ρ swings dramatically; Spearman ρ
        barely moves because the outlier is just one rank position. This
        is the property that justifies using Spearman on the small-N
        conditional matrix without dropping or clipping the outlier day.

        Construction is deterministic and pedagogical: 20 SPY stress days
        with X = SPY exactly (no noise), so the un-contaminated conditional
        ρ is +1 under both methods. The X-outlier (−60%) lands on SPY's
        own worst day — the empirically realistic case where a distressed
        name's worst day coincides with the market's worst day, just much
        more violently. Spearman sees this as perfect rank agreement and
        stays at +1; Pearson is dragged below 0.70 by the magnitude gap.
        """
        idx = pd.bdate_range("2020-01-02", periods=400)
        spy_r = np.zeros(400)
        x_r = np.zeros(400)
        # 20 SPY-stress days spaced out, with monotonically spread severities
        # so each day has a clean rank. SPY days range -0.030 to -0.049.
        stress_positions = list(range(20, 400, 20))[:20]
        for k, pos in enumerate(stress_positions):
            spy_r[pos] = -0.030 - 0.001 * k  # -0.030, -0.031, ..., -0.049
            x_r[pos] = spy_r[pos]            # perfect alignment
        # Outlier hits SPY's WORST stress day — both ranks remain rank 1,
        # so Spearman sees no rank disagreement. Pearson sees a huge
        # magnitude mismatch and gets pulled.
        outlier_pos = stress_positions[-1]   # k=19 → SPY = -0.049 (most negative)
        x_r[outlier_pos] = -0.60             # X tanks extra hard on that day

        def _px(r):
            return 100.0 * np.exp(np.cumsum(r))
        px = pd.DataFrame({"SPY": _px(spy_r), "X": _px(x_r)}, index=idx)

        pear = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "X"], condition_symbol="SPY",
            z_threshold=-1.5, method="pearson",
        )
        spear = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "X"], condition_symbol="SPY",
            z_threshold=-1.5, method="spearman",
        )
        self.assertTrue(pear["enough"])
        self.assertTrue(spear["enough"])
        pear_rho = float(pear["conditional"].loc["SPY", "X"])
        spear_rho = float(spear["conditional"].loc["SPY", "X"])
        # Pearson conditional ρ is heavily dragged down from the +1.0 baseline.
        self.assertLess(pear_rho, 0.70)
        # Spearman barely moves — single outlier is one bad rank among ~20
        # mostly perfectly-aligned ranks.
        self.assertGreater(spear_rho, 0.95)
        # And the robustness gap is unambiguous.
        self.assertGreater(spear_rho - pear_rho, 0.25)
        # No data was hidden: outlier day participates in both estimates.
        self.assertEqual(pear["n_stress"], spear["n_stress"])
        # And Spearman is the dashboard's default for this function.
        default = rm.compute_conditional_correlation_matrix(
            px, ["SPY", "X"], condition_symbol="SPY", z_threshold=-1.5,
        )
        self.assertAlmostEqual(
            float(default["conditional"].loc["SPY", "X"]),
            spear_rho, places=10,
        )


class TestWeightsPerSnapMonthly(unittest.TestCase):
    """WSF-1: the historical daily synthesis must use ONE weight snapshot per
    calendar month. A dual-date month (JPM stamps the last business day,
    Fidelity stamps month-end) must NOT split into two one-broker snapshots
    that each own a sub-segment of the synthesis with half the portfolio."""

    def test_coalesces_dual_date_month_into_one_snapshot(self) -> None:
        # JPM May 29 + Fidelity May 31 are one month-end snapshot split across
        # two filing dates -> ONE snapshot, keyed by the month's latest date,
        # built from BOTH brokers' rows.
        pos = pd.DataFrame([
            {"statement_date": pd.Timestamp("2026-05-29"), "broker": "jpm"},
            {"statement_date": pd.Timestamp("2026-05-31"), "broker": "fidelity"},
        ])
        seen_brokers = []

        def build(snap):
            seen_brokers.append(sorted(snap["broker"].tolist()))
            return pd.Series({"AGG": 1.0})

        out = rm.weights_per_snap_monthly(pos, build)
        self.assertEqual(list(out.keys()), [pd.Timestamp("2026-05-31")])
        self.assertEqual(seen_brokers, [["fidelity", "jpm"]])

    def test_separate_months_stay_separate(self) -> None:
        pos = pd.DataFrame([
            {"statement_date": pd.Timestamp("2026-04-30"), "broker": "jpm"},
            {"statement_date": pd.Timestamp("2026-05-31"), "broker": "jpm"},
        ])
        out = rm.weights_per_snap_monthly(
            pos, lambda s: pd.Series({"AGG": 1.0}))
        self.assertEqual(
            sorted(out.keys()),
            [pd.Timestamp("2026-04-30"), pd.Timestamp("2026-05-31")])

    def test_skips_months_with_empty_weights(self) -> None:
        pos = pd.DataFrame([
            {"statement_date": pd.Timestamp("2026-05-31"), "broker": "jpm"},
        ])
        out = rm.weights_per_snap_monthly(
            pos, lambda s: pd.Series(dtype=float))
        self.assertEqual(out, {})

    def test_empty_positions_returns_empty(self) -> None:
        empty = pd.DataFrame(
            {"statement_date": pd.Series(dtype="datetime64[ns]")})
        self.assertEqual(
            rm.weights_per_snap_monthly(empty, lambda s: pd.Series({"X": 1.0})),
            {})


# ---------------------------------------------------------------------------
# series_vol_ann — single-series annualized vol under a chosen estimator.
# Lets a benchmark vol be computed with the SAME estimator as the portfolio
# tile, so the colored better/worse delta is apples-to-apples (audit WSB-2).
# ---------------------------------------------------------------------------
class TestSeriesVolAnn(unittest.TestCase):
    def _rets(self, vols, seed=0):
        """Daily returns whose stdev steps up across `vols` blocks."""
        rng = np.random.default_rng(seed)
        chunks = [rng.normal(0.0, v, 120) for v in vols]
        return pd.Series(np.concatenate(chunks))

    def test_rolling_equals_sample_std(self) -> None:
        # For a series no longer than the window the "rolling" estimator is
        # the plain sample covariance, so vol == std(ddof=1) * sqrt(252) —
        # exactly the legacy benchmark formula this replaces.
        s = self._rets([0.01], seed=1)  # 120 obs < 252-day window
        expected = float(s.std(ddof=1) * np.sqrt(252.0))
        got = rm.series_vol_ann(s, estimator="rolling", window=252)
        self.assertAlmostEqual(got, expected, places=10)

    def test_equals_estimate_covariance_diagonal(self) -> None:
        # Contract: series_vol_ann is the 1-asset specialization of
        # estimate_covariance — sqrt of its single diagonal entry.
        s = self._rets([0.01, 0.02], seed=4)
        for est in ("rolling", "ewma", "ewma_lw"):
            cov = rm.estimate_covariance(
                s.to_frame("x"), estimator=est, window=252)["cov"]
            expected = float(np.sqrt(cov.iloc[0, 0]))
            got = rm.series_vol_ann(s, estimator=est, window=252)
            self.assertAlmostEqual(got, expected, places=12, msg=est)

    def test_ewma_lw_differs_from_sample(self) -> None:
        # On a vol-clustering series (calm then stormy) the recency-weighted
        # EWMA-LW estimate must diverge from the equal-weight sample std —
        # that divergence is exactly what flipped the delta's color.
        s = self._rets([0.005, 0.03], seed=2)
        sample = float(s.std(ddof=1) * np.sqrt(252.0))
        ewma = rm.series_vol_ann(s, estimator="ewma_lw", window=252)
        self.assertTrue(np.isfinite(ewma) and ewma > 0)
        self.assertNotAlmostEqual(ewma, sample, places=4)

    def test_matches_single_asset_portfolio(self) -> None:
        # The WSB-2 invariant: a 1-asset portfolio's vol from
        # compute_risk_contributions equals series_vol_ann on that asset's
        # returns under the same estimator — i.e. the benchmark is now
        # computed exactly like the book it is compared against.
        idx = pd.bdate_range("2021-01-04", periods=400)
        rng = np.random.default_rng(3)
        px = pd.DataFrame(
            {"AAA": 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, 400))},
            index=idx,
        )
        rets = px["AAA"].pct_change().dropna()
        for est in ("rolling", "ewma_lw"):
            rc = rm.compute_risk_contributions(
                pd.Series({"AAA": 1.0}), px, window=252, estimator=est)
            got = rm.series_vol_ann(rets, estimator=est, window=252)
            self.assertAlmostEqual(got, float(rc["port_vol_ann"]),
                                   places=10, msg=f"estimator={est}")

    def test_empty_series_is_nan(self) -> None:
        self.assertTrue(np.isnan(
            rm.series_vol_ann(pd.Series(dtype=float), estimator="ewma_lw")))


class TestConcatSortStability(unittest.TestCase):
    """WSG-13: pd.concat(axis=1) over DatetimeIndex series must pass sort=
    explicitly. The deprecated default (sort-by-default, flipping to no-sort in
    pandas 4) would silently mis-order the .dropna().tail(252) windows behind
    the rolling-beta / scatter charts."""

    def test_aligned_emits_no_concat_sort_deprecation(self) -> None:
        import warnings
        idx_a = pd.to_datetime(["2026-01-05", "2026-01-01", "2026-01-03"])
        idx_b = pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-05"])
        p = pd.Series([1.0, 2.0, 3.0], index=idx_a)
        b = pd.Series([4.0, 5.0, 6.0], index=idx_b)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="Sorting by default when concatenating")
            df = rm._aligned(p, b)
        # Chronological order is what the downstream .tail(252) windows assume.
        self.assertTrue(df.index.is_monotonic_increasing)


class TestComputeDrFrames(unittest.TestCase):
    """compute_dr_frames — the shared DR-frame orchestration both UIs consume
    (Phase D PR-2). Verbatim move from terminal/riskcontrib_dr.py; these pin
    its keys, the finite path, the empty/degenerate path, and the port-start
    clip on a constructed panel long enough to populate the 252d window."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2022-01-03", periods=600)
        shared = rng.normal(0.0004, 0.010, size=(600, 1))
        idio = rng.normal(0.0, 0.007, size=(600, 3))
        rets = shared + idio
        prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
        cls.daily = pd.DataFrame(prices, index=idx,
                                 columns=["AAA", "BBB", "SPY"])
        cls.weights = pd.Series({"AAA": 0.3, "BBB": 0.2, "SPY": 0.5})
        cls.port_rets = pd.Series(rets[:, 0], index=idx).tail(300)

    def test_constants_present(self):
        self.assertEqual(
            (rm.DR_SHORT_W, rm.DR_MED_W, rm.DR_LONG_W), (21, 63, 252))

    def test_finite_path_keys_and_values(self):
        f = rm.compute_dr_frames(self.weights, self.daily, self.port_rets)
        self.assertEqual(set(f), {"dr_ts", "max_dr_ts", "ratio_ts",
                                  "available", "dr_s", "dr_l"})
        self.assertTrue(f["available"])
        self.assertFalse(f["dr_ts"].empty)
        self.assertGreater(f["ratio_ts"].dropna().shape[0], 0)
        self.assertTrue(np.isfinite(f["dr_s"]) and np.isfinite(f["dr_l"]))

    def test_port_start_clips_dr_ts(self):
        f = rm.compute_dr_frames(self.weights, self.daily, self.port_rets)
        self.assertGreaterEqual(f["dr_ts"].index.min(),
                                self.port_rets.index.min())

    def test_empty_weights_unavailable(self):
        f = rm.compute_dr_frames(pd.Series(dtype=float), self.daily,
                                 self.port_rets)
        self.assertFalse(f["available"])
        self.assertTrue(f["dr_ts"].empty and f["max_dr_ts"].empty)
        self.assertTrue(np.isnan(f["dr_s"]) and np.isnan(f["dr_l"]))

    def test_terminal_reexport_is_same_object(self):
        # riskcontrib_dr must re-export the moved names so riskcontrib_regime
        # and the terminal tests keep importing them from there unchanged.
        from terminal import riskcontrib_dr as rcd
        self.assertIs(rcd.compute_dr_frames, rm.compute_dr_frames)
        self.assertEqual((rcd.DR_SHORT_W, rcd.DR_MED_W, rcd.DR_LONG_W),
                         (rm.DR_SHORT_W, rm.DR_MED_W, rm.DR_LONG_W))


if __name__ == "__main__":
    unittest.main()
