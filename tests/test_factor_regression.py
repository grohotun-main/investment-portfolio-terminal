"""Tests for parsers/factor_regression.py — pure-numpy factor OLS.

Synthetic-data philosophy: generate factor returns and build the portfolio
return as an exact linear combination plus tiny iid noise, then assert the
regression recovers the construction. Tiny-but-nonzero noise keeps OLS SEs
positive (zero residuals -> se=0 -> t-stats blow up with divide warnings).
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import factor_regression as fr  # noqa: E402


def _synth_frames(n_months: int = 120, alpha: float = 0.002,
                  betas: dict | None = None, noise_sd: float = 1e-6,
                  seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(twr_portfolio-like, factors-like) frames with a KNOWN structure.

    Portfolio return = rf + alpha + sum(beta_i * factor_i) + eps.
    """
    betas = betas or {"mkt_rf": 1.10, "smb": 0.30, "hml": -0.20,
                      "rmw": 0.00, "cma": 0.00, "mom": 0.15}
    rng = np.random.default_rng(seed)
    months = pd.period_range("2016-01", periods=n_months, freq="M")
    factors = pd.DataFrame({
        "month": months.astype(str),
        "mkt_rf": rng.normal(0.006, 0.04, n_months),
        "smb": rng.normal(0.0, 0.02, n_months),
        "hml": rng.normal(0.0, 0.025, n_months),
        "rmw": rng.normal(0.001, 0.015, n_months),
        "cma": rng.normal(0.0, 0.012, n_months),
        "mom": rng.normal(0.002, 0.03, n_months),
        "rf": np.full(n_months, 0.003),
    })
    ret = (factors["rf"] + alpha
           + sum(b * factors[c] for c, b in betas.items())
           + rng.normal(0.0, noise_sd, n_months))
    twr = pd.DataFrame({"month": months,  # PeriodIndex, like load_twr's frame
                        "return_pct": ret.to_numpy()})
    return twr, factors


class TestTCrit(unittest.TestCase):
    def test_knot_values(self) -> None:
        self.assertAlmostEqual(fr.t_crit_975(10), 2.228, places=3)
        self.assertAlmostEqual(fr.t_crit_975(120), 1.980, places=3)

    def test_above_table_uses_normal(self) -> None:
        self.assertAlmostEqual(fr.t_crit_975(500), 1.96, places=2)

    def test_between_knots_uses_lower_knot(self) -> None:
        # dof=70 sits between knots 60 and 80 -> conservative (wider) 2.000.
        self.assertAlmostEqual(fr.t_crit_975(70), 2.000, places=3)

    def test_monotone_decreasing(self) -> None:
        vals = [fr.t_crit_975(d) for d in (1, 2, 5, 10, 30, 60, 120, 200)]
        self.assertEqual(vals, sorted(vals, reverse=True))


class TestModelsRegistry(unittest.TestCase):
    def test_models_are_nested_subsets_of_known_columns(self) -> None:
        known = set(fr.FACTOR_LABELS)
        for name, cols in fr.MODELS.items():
            self.assertTrue(set(cols).issubset(known), name)
        self.assertEqual(fr.MODELS["CAPM"], ("mkt_rf",))
        self.assertEqual(len(fr.MODELS["FF5 + Mom"]), 6)


class TestAlign(unittest.TestCase):
    def test_inner_join_on_month_handles_period_dtype(self) -> None:
        twr, factors = _synth_frames(24)
        aligned = fr.align_twr_with_factors(twr, factors)
        self.assertEqual(len(aligned), 24)
        self.assertEqual(list(aligned.columns),
                         ["month", "ret", "rf"] + list(fr.FACTOR_LABELS))

    def test_nan_rows_dropped(self) -> None:
        twr, factors = _synth_frames(24)
        factors.loc[3, "mom"] = np.nan
        twr.loc[5, "return_pct"] = np.nan
        aligned = fr.align_twr_with_factors(twr, factors)
        self.assertEqual(len(aligned), 22)

    def test_window_applied_after_alignment(self) -> None:
        twr, factors = _synth_frames(60)
        factors = factors[factors["month"] >= "2017-01"]  # kill 12 months
        aligned = fr.align_twr_with_factors(twr, factors, window_months=24)
        self.assertEqual(len(aligned), 24)
        self.assertEqual(aligned["month"].iloc[-1], "2020-12")

    def test_empty_inputs_give_empty_frame(self) -> None:
        twr, factors = _synth_frames(12)
        self.assertTrue(
            fr.align_twr_with_factors(pd.DataFrame(), factors).empty)
        self.assertTrue(
            fr.align_twr_with_factors(twr, pd.DataFrame()).empty)


class TestRegression(unittest.TestCase):
    def test_recovers_known_betas_and_alpha(self) -> None:
        twr, factors = _synth_frames(120, alpha=0.002)
        aligned = fr.align_twr_with_factors(twr, factors)
        res = fr.run_factor_regression(aligned, "FF5 + Mom")
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res.betas["mkt_rf"], 1.10, places=3)
        self.assertAlmostEqual(res.betas["smb"], 0.30, places=3)
        self.assertAlmostEqual(res.betas["hml"], -0.20, places=3)
        self.assertAlmostEqual(res.alpha_monthly, 0.002, places=4)
        self.assertAlmostEqual(res.alpha_annual, 0.024, places=3)
        self.assertGreater(res.r2, 0.999)

    def test_smaller_model_loads_omitted_factors_into_alpha_or_betas(self) -> None:
        # CAPM on multi-factor data: alpha absorbs the unmodeled premia, so
        # CAPM alpha != FF alpha — the entire point of the tab. Just assert
        # it runs and reports the right factor set.
        twr, factors = _synth_frames(120)
        aligned = fr.align_twr_with_factors(twr, factors)
        res = fr.run_factor_regression(aligned, "CAPM")
        self.assertIsNotNone(res)
        self.assertEqual(res.factors, ("mkt_rf",))
        self.assertEqual(set(res.betas), {"mkt_rf"})

    def test_alpha_ci_brackets_true_alpha(self) -> None:
        twr, factors = _synth_frames(120, alpha=0.004, noise_sd=0.01)
        aligned = fr.align_twr_with_factors(twr, factors)
        res = fr.run_factor_regression(aligned, "FF3")
        lo, hi = res.alpha_ci_annual
        self.assertLess(lo, hi)
        self.assertLess(lo, 0.004 * 12)
        self.assertGreater(hi, 0.004 * 12 * 0.2)  # loose: noisy small sample

    def test_big_alpha_has_big_t(self) -> None:
        twr, factors = _synth_frames(120, alpha=0.01, noise_sd=0.002)
        aligned = fr.align_twr_with_factors(twr, factors)
        res = fr.run_factor_regression(aligned, "FF5 + Mom")
        self.assertGreater(res.alpha_t, 5.0)

    def test_n_floor_returns_none(self) -> None:
        twr, factors = _synth_frames(7)  # FF5+Mom needs k+2 = 8
        aligned = fr.align_twr_with_factors(twr, factors)
        self.assertIsNone(fr.run_factor_regression(aligned, "FF5 + Mom"))
        self.assertIsNotNone(fr.run_factor_regression(aligned, "FF3"))

    def test_rank_deficient_returns_none(self) -> None:
        twr, factors = _synth_frames(60)
        factors["hml"] = factors["smb"]  # exact collinearity
        aligned = fr.align_twr_with_factors(twr, factors)
        self.assertIsNone(fr.run_factor_regression(aligned, "FF3"))

    def test_constant_excess_return_gives_nan_r2(self) -> None:
        # ret and rf both constant -> excess return is numerically constant,
        # but float summation leaves ss_tot ~1e-35 rather than exact 0. The
        # guard must classify that as "no variance to explain" (NaN r2), not
        # report a ratio of rounding noise.
        twr, factors = _synth_frames(30)
        twr["return_pct"] = 0.007  # rf is constant 0.003 -> excess 0.004
        aligned = fr.align_twr_with_factors(twr, factors)
        res = fr.run_factor_regression(aligned, "FF3")
        self.assertIsNotNone(res)
        self.assertTrue(math.isnan(res.r2))
        self.assertTrue(math.isnan(res.adj_r2))

    def test_unknown_model_raises_keyerror(self) -> None:
        twr, factors = _synth_frames(24)
        aligned = fr.align_twr_with_factors(twr, factors)
        with self.assertRaises(KeyError):
            fr.run_factor_regression(aligned, "FF7")


class TestAttribution(unittest.TestCase):
    def _result(self):
        twr, factors = _synth_frames(120, alpha=0.002, noise_sd=0.005)
        aligned = fr.align_twr_with_factors(twr, factors)
        return fr.run_factor_regression(aligned, "FF5 + Mom")

    def test_sums_exactly_to_arithmetic_annualized_mean(self) -> None:
        # ret = rf + alpha + Σβf + ε and OLS-with-intercept residuals
        # average zero, so the decomposition is an identity, not an approx.
        res = self._result()
        contribs = fr.attribution(res)
        self.assertAlmostEqual(sum(v for _, v in contribs),
                               res.mean_return_monthly * 12.0, places=10)

    def test_rf_first_alpha_last_factors_in_model_order(self) -> None:
        res = self._result()
        labels = [lab for lab, _ in fr.attribution(res)]
        self.assertTrue(labels[0].startswith("Risk-free"))
        self.assertEqual(labels[-1], "Alpha (unexplained)")
        self.assertEqual(labels[1:-1],
                         [fr.FACTOR_LABELS[c] for c in res.factors])

    def test_alpha_component_is_annualized_alpha(self) -> None:
        res = self._result()
        contribs = dict(fr.attribution(res))
        self.assertAlmostEqual(contribs["Alpha (unexplained)"],
                               res.alpha_annual, places=12)


class TestAttributionTimeseries(unittest.TestCase):
    def _fit(self, model: str = "FF5 + Mom"):
        twr, factors = _synth_frames(120, alpha=0.002, noise_sd=0.005)
        aligned = fr.align_twr_with_factors(twr, factors)
        return fr.run_factor_regression(aligned, model), aligned

    def test_rows_sum_exactly_to_raw_return_every_model(self) -> None:
        # unexplained is computed as the remainder, so the per-month
        # decomposition is an identity for every model, not an approx.
        for model in fr.MODELS:
            res, aligned = self._fit(model)
            ts = fr.attribution_timeseries(res, aligned)
            row_sums = ts.drop(columns=["month"]).sum(axis=1).to_numpy()
            np.testing.assert_allclose(
                row_sums, aligned["ret"].to_numpy(), rtol=0, atol=1e-12,
                err_msg=f"row sums != ret for {model}")

    def test_column_means_reproduce_waterfall(self) -> None:
        # Column means ×12 ARE the waterfall bars (attribution()).
        res, aligned = self._fit()
        ts = fr.attribution_timeseries(res, aligned)
        cols = (["rf"] + [f"contrib_{c}" for c in res.factors]
                + ["unexplained"])
        for (label, annual), col in zip(fr.attribution(res), cols):
            self.assertAlmostEqual(ts[col].mean() * 12.0, annual,
                                   places=10, msg=f"{label} vs {col}")

    def test_unexplained_mean_is_alpha(self) -> None:
        # OLS-with-intercept residuals average zero -> mean(α + e) = α.
        res, aligned = self._fit()
        ts = fr.attribution_timeseries(res, aligned)
        self.assertAlmostEqual(ts["unexplained"].mean() * 12.0,
                               res.alpha_annual, places=10)

    def test_column_order(self) -> None:
        # month, rf, factors in model order, unexplained — the UI renders
        # this order directly.
        res, aligned = self._fit("FF3")
        ts = fr.attribution_timeseries(res, aligned)
        self.assertEqual(list(ts.columns),
                         ["month", "rf", "contrib_mkt_rf", "contrib_smb",
                          "contrib_hml", "unexplained"])

    def test_window_mismatch_raises(self) -> None:
        # Passing any frame other than the exact fitted window is a
        # programmer error -> loud ValueError, not silent wrong numbers.
        res, aligned = self._fit()
        with self.assertRaises(ValueError):
            fr.attribution_timeseries(res, aligned.iloc[1:])

    def test_non_default_index_slice_decomposes_positionally(self) -> None:
        # A non-reset .iloc tail slice (RangeIndex starting at 24) must
        # decompose identically — the function is positional by contract;
        # index-aligned pandas arithmetic would NaN this.
        twr, factors = _synth_frames(120, alpha=0.002, noise_sd=0.005)
        aligned = fr.align_twr_with_factors(twr, factors).iloc[24:]
        res = fr.run_factor_regression(aligned, "FF5 + Mom")
        ts = fr.attribution_timeseries(res, aligned)
        np.testing.assert_allclose(
            ts.drop(columns=["month"]).sum(axis=1).to_numpy(),
            aligned["ret"].to_numpy(), rtol=0, atol=1e-12)


class TestReferenceValues(unittest.TestCase):
    """Pin run_factor_regression's OLS output to hard-coded constants derived
    INDEPENDENTLY via the simple-regression closed form (not by calling the
    module):

        beta_hat  = Sxy / Sxx
        alpha_hat = ybar - beta_hat * xbar
        sigma2    = SSR / (n - 2)          [dof = n - k - 1 = 8 - 1 - 1 = 6]
        se_alpha  = sqrt(sigma2 * (1/n + xbar^2/Sxx))
        se_beta   = sqrt(sigma2 / Sxx)
        r2        = 1 - SSR/SST
        adj_r2    = 1 - (1-r2)*(n-1)/dof
        CI        = (alpha_monthly +/- t975(6=2.447) * se_alpha) * 12

    Any swap of dof (e.g. n-k-1 vs n-k), missing sqrt, or wrong t-table
    value breaks at least one assertAlmostEqual(..., places=10) here.
    """

    def _make_aligned(self) -> pd.DataFrame:
        """8-month CAPM dataset with fully deterministic, hand-pickable numbers.

        ret = rf + 0.005 + 0.8*mkt_rf + e  (no rng)
        rf  = 0.004 constant
        e   = [+0.001, -0.001, +0.001, -0.001, +0.001, -0.001, +0.001, -0.001]
        Other factor columns are filled with fixed distinct values; they are
        present (required by align schema) but ignored by the CAPM regression.
        """
        mkt_rf = [0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.04, -0.03]
        rf_c = 0.004
        e = [+0.001, -0.001, +0.001, -0.001, +0.001, -0.001, +0.001, -0.001]
        ret = [rf_c + 0.005 + 0.8 * mkt_rf[i] + e[i] for i in range(8)]
        return pd.DataFrame({
            "month":  [f"2025-{i+1:02d}" for i in range(8)],
            "ret":    ret,
            "rf":     [rf_c] * 8,
            "mkt_rf": mkt_rf,
            # Non-NaN filler so align_twr_with_factors schema is satisfied;
            # CAPM ignores these columns entirely.
            "smb":  [0.001 * i for i in range(8)],
            "hml":  [0.002 * i for i in range(8)],
            "rmw":  [0.003 * i for i in range(8)],
            "cma":  [0.004 * i for i in range(8)],
            "mom":  [0.005 * i for i in range(8)],
        })

    def test_capm_matches_hand_computed_ols(self) -> None:
        # Constants derived independently via textbook closed form above —
        # NOT copied from module output.  Any dof/sqrt/t-table regression
        # breaks this test.
        aligned = self._make_aligned()
        res = fr.run_factor_regression(aligned, "CAPM")
        self.assertIsNotNone(res)

        self.assertEqual(res.n, 8)

        self.assertAlmostEqual(res.betas["mkt_rf"],  0.838095238095, places=10)
        self.assertAlmostEqual(res.se["mkt_rf"],     0.008694008849, places=10)
        self.assertAlmostEqual(res.tstats["mkt_rf"], 96.399170120909, places=10)

        self.assertAlmostEqual(res.alpha_monthly, 0.004809523810, places=10)
        self.assertAlmostEqual(res.alpha_t,       23.588518008094, places=10)
        self.assertAlmostEqual(res.alpha_annual,  0.057714285714, places=10)

        lo, hi = res.alpha_ci_annual
        self.assertAlmostEqual(lo, 0.051727183977, places=10)
        self.assertAlmostEqual(hi, 0.063701387451, places=10)

        self.assertAlmostEqual(res.r2,     0.999354755452, places=10)
        self.assertAlmostEqual(res.adj_r2, 0.999247214694, places=10)


def _aligned_beta_switch(n: int = 36, switch: int = 18,
                         b1: float = 0.5, b2: float = 1.5,
                         alpha: float = 0.004) -> pd.DataFrame:
    """Aligned-frame builder with an EXACT (zero-noise) CAPM structure whose
    market beta switches from b1 to b2 at row `switch`. Zero residual makes
    every fully-inside window recover its half's beta to float precision
    (se=0 -> t=inf is fine: rolling output only reads betas + alpha)."""
    rng = np.random.default_rng(3)
    mkt = rng.normal(0.005, 0.04, n)
    rf = 0.003
    b = np.where(np.arange(n) < switch, b1, b2)
    ret = rf + alpha + b * mkt
    months = pd.period_range("2020-01", periods=n, freq="M").astype(str)
    return pd.DataFrame({
        "month": months, "ret": ret, "rf": rf, "mkt_rf": mkt,
        "smb": rng.normal(0, 0.02, n), "hml": rng.normal(0, 0.02, n),
        "rmw": rng.normal(0, 0.02, n), "cma": rng.normal(0, 0.02, n),
        "mom": rng.normal(0, 0.02, n),
    })


class TestRollingFactorRegressions(unittest.TestCase):
    def test_schema_and_row_count(self) -> None:
        aligned = _aligned_beta_switch(n=36)
        roll = fr.rolling_factor_regressions(aligned, "FF3", 12)
        self.assertEqual(list(roll.columns),
                         ["month", "alpha_annual",
                          "beta_mkt_rf", "beta_smb", "beta_hml"])
        self.assertEqual(len(roll), 36 - 12 + 1)
        # End-dated: first window ends at row 11, last at row 35.
        self.assertEqual(roll["month"].iloc[0], aligned["month"].iloc[11])
        self.assertEqual(roll["month"].iloc[-1], aligned["month"].iloc[-1])

    def test_recovers_exact_betas_each_side_of_switch(self) -> None:
        aligned = _aligned_beta_switch(n=36, switch=18, b1=0.5, b2=1.5,
                                       alpha=0.004)
        roll = fr.rolling_factor_regressions(aligned, "CAPM", 12)
        # Window ending at aligned row 11 (months 0-11) is fully pre-switch.
        first = roll.iloc[0]
        self.assertAlmostEqual(first["beta_mkt_rf"], 0.5, places=10)
        self.assertAlmostEqual(first["alpha_annual"], 0.004 * 12, places=10)
        # Window ending at the last row (months 24-35) is fully post-switch.
        last = roll.iloc[-1]
        self.assertAlmostEqual(last["beta_mkt_rf"], 1.5, places=10)
        self.assertAlmostEqual(last["alpha_annual"], 0.004 * 12, places=10)
        # A straddling window lands strictly between the two betas.
        mid = roll[roll["month"] == aligned["month"].iloc[20]].iloc[0]
        self.assertGreater(mid["beta_mkt_rf"], 0.5)
        self.assertLess(mid["beta_mkt_rf"], 1.5)

    def test_too_few_rows_gives_empty_frame(self) -> None:
        aligned = _aligned_beta_switch(n=10)
        roll = fr.rolling_factor_regressions(aligned, "CAPM", 12)
        self.assertTrue(roll.empty)
        self.assertEqual(list(roll.columns),
                         ["month", "alpha_annual", "beta_mkt_rf"])

    def test_rank_deficient_window_yields_nan_row(self) -> None:
        aligned = _aligned_beta_switch(n=24)
        # First 12 months: constant market column -> X=[1, const] is rank 1.
        aligned.loc[:11, "mkt_rf"] = 0.01
        aligned.loc[:11, "ret"] = 0.003 + 0.004 + 0.5 * 0.01
        roll = fr.rolling_factor_regressions(aligned, "CAPM", 12)
        self.assertTrue(math.isnan(roll.iloc[0]["beta_mkt_rf"]))
        self.assertTrue(math.isnan(roll.iloc[0]["alpha_annual"]))
        # Later windows (containing variation) produce numbers again.
        self.assertFalse(math.isnan(roll.iloc[-1]["beta_mkt_rf"]))


class TestPeriodsPerYear(unittest.TestCase):
    """ppy only rescales annualization — per-period stats are identical."""

    def _aligned(self) -> pd.DataFrame:
        twr, factors = _synth_frames(120, alpha=0.002, noise_sd=0.005)
        return fr.align_twr_with_factors(twr, factors)

    def test_default_is_12_and_field_stored(self) -> None:
        res = fr.run_factor_regression(self._aligned(), "FF3")
        self.assertEqual(res.periods_per_year, 12)

    def test_alpha_annualization_scales_with_ppy(self) -> None:
        aligned = self._aligned()
        res12 = fr.run_factor_regression(aligned, "FF3")
        res252 = fr.run_factor_regression(aligned, "FF3",
                                          periods_per_year=252)
        self.assertAlmostEqual(res252.alpha_monthly, res12.alpha_monthly,
                               places=14)
        self.assertAlmostEqual(res252.alpha_annual,
                               res12.alpha_monthly * 252, places=12)
        lo12, hi12 = res12.alpha_ci_annual
        lo252, hi252 = res252.alpha_ci_annual
        self.assertAlmostEqual(lo252, lo12 / 12 * 252, places=12)
        self.assertAlmostEqual(hi252, hi12 / 12 * 252, places=12)
        # Per-period stats untouched by ppy.
        self.assertAlmostEqual(res252.alpha_t, res12.alpha_t, places=14)
        self.assertEqual(res252.betas, res12.betas)
        self.assertEqual(res252.r2, res12.r2)

    def test_attribution_identity_holds_at_252(self) -> None:
        res = fr.run_factor_regression(self._aligned(), "FF5 + Mom",
                                       periods_per_year=252)
        contribs = fr.attribution(res)
        self.assertAlmostEqual(sum(v for _, v in contribs),
                               res.mean_return_monthly * 252.0, places=9)
        self.assertAlmostEqual(dict(contribs)["Alpha (unexplained)"],
                               res.alpha_annual, places=12)


def _synth_daily_factors(n_days: int = 300, seed: int = 11,
                         start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "mkt_rf": rng.normal(0.0004, 0.011, n_days),
        "smb": rng.normal(0.0, 0.005, n_days),
        "hml": rng.normal(0.0, 0.006, n_days),
        "rmw": rng.normal(0.0, 0.004, n_days),
        "cma": rng.normal(0.0, 0.004, n_days),
        "mom": rng.normal(0.0001, 0.007, n_days),
        "rf": np.full(n_days, 0.0002),
    })


class TestAlignReturnsWithFactors(unittest.TestCase):
    def test_datetimeindex_series_joins_on_date(self) -> None:
        factors = _synth_daily_factors(50)
        idx = pd.DatetimeIndex(pd.to_datetime(factors["date"]))
        rets = pd.Series(np.full(50, 0.001), index=idx)
        aligned = fr.align_returns_with_factors(rets, factors)
        self.assertEqual(len(aligned), 50)
        # Same schema as the monthly align — key column is named 'month'
        # even at daily frequency (run_factor_regression treats it as an
        # opaque period label).
        self.assertEqual(list(aligned.columns),
                         ["month", "ret", "rf"] + list(fr.FACTOR_LABELS))
        self.assertEqual(aligned["month"].iloc[0], factors["date"].iloc[0])

    def test_partial_overlap_and_window(self) -> None:
        factors = _synth_daily_factors(50)
        idx = pd.DatetimeIndex(pd.to_datetime(factors["date"].iloc[10:]))
        rets = pd.Series(np.full(40, 0.001), index=idx)
        aligned = fr.align_returns_with_factors(rets, factors, window=20)
        self.assertEqual(len(aligned), 20)
        self.assertEqual(aligned["month"].iloc[-1],
                         factors["date"].iloc[-1])

    def test_empty_inputs(self) -> None:
        factors = _synth_daily_factors(10)
        self.assertTrue(fr.align_returns_with_factors(
            pd.Series(dtype=float), factors).empty)
        idx = pd.DatetimeIndex(pd.to_datetime(factors["date"]))
        rets = pd.Series(np.full(10, 0.001), index=idx)
        self.assertTrue(fr.align_returns_with_factors(
            rets, pd.DataFrame()).empty)

    def test_nan_rows_dropped(self) -> None:
        factors = _synth_daily_factors(30)
        factors.loc[5, "mom"] = np.nan
        idx = pd.DatetimeIndex(pd.to_datetime(factors["date"]))
        vals = np.full(30, 0.001)
        vals[7] = np.nan
        rets = pd.Series(vals, index=idx)
        aligned = fr.align_returns_with_factors(rets, factors)
        self.assertEqual(len(aligned), 28)

    def test_monthly_frame_with_month_column_also_accepted(self) -> None:
        # The monthly factor file (key column 'month') goes through the
        # same helper — proves align_twr_with_factors can delegate.
        twr, factors = _synth_frames(24)
        rets = pd.Series(twr["return_pct"].to_numpy(),
                         index=twr["month"])  # PeriodIndex-ish labels
        aligned = fr.align_returns_with_factors(rets, factors)
        self.assertEqual(len(aligned), 24)


def _prices_from_returns(rets: np.ndarray,
                         dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=dates)


class TestPerHoldingRegressions(unittest.TestCase):
    def _setup(self, n_days: int = 300):
        factors = _synth_daily_factors(n_days)
        dates = pd.DatetimeIndex(pd.to_datetime(factors["date"]))
        rng = np.random.default_rng(5)
        noise = rng.normal(0.0, 1e-6, n_days)
        # AAA: known structure rf + 0.0002 + 1.2*mkt - 0.4*hml (+ tiny noise)
        aaa_r = (factors["rf"] + 0.0002 + 1.2 * factors["mkt_rf"]
                 - 0.4 * factors["hml"] + noise).to_numpy()
        prices = pd.DataFrame({
            "AAA": _prices_from_returns(aaa_r, dates),
            # SHORTY: only the last 50 days have prices -> 49 returns < 126
            "SHORTY": _prices_from_returns(
                aaa_r, dates).where(pd.Series(
                    np.arange(n_days) >= n_days - 50, index=dates).values),
        })
        return prices, factors

    def test_recovers_known_betas_and_skips_short_history(self) -> None:
        prices, factors = self._setup()
        table, skipped = fr.per_holding_regressions(
            prices, factors, "FF3", window=None, min_obs=126)
        self.assertEqual(list(table["symbol"]), ["AAA"])
        row = table.iloc[0]
        self.assertAlmostEqual(row["beta_mkt_rf"], 1.2, places=3)
        self.assertAlmostEqual(row["beta_hml"], -0.4, places=3)
        self.assertAlmostEqual(row["alpha_annual"], 0.0002 * 252, places=3)
        self.assertEqual(row["n"], 299)  # n_days - 1 pct_change row
        self.assertEqual(list(skipped["symbol"]), ["SHORTY"])
        self.assertEqual(int(skipped.iloc[0]["n"]), 49)

    def test_schema(self) -> None:
        prices, factors = self._setup()
        table, skipped = fr.per_holding_regressions(
            prices, factors, "CAPM")
        self.assertEqual(list(table.columns),
                         ["symbol", "n", "start", "end", "alpha_annual",
                          "alpha_t", "r2", "beta_mkt_rf", "t_mkt_rf"])
        self.assertEqual(list(skipped.columns), ["symbol", "n"])

    def test_nan_price_gap_drops_two_returns(self) -> None:
        prices, factors = self._setup()
        # One NaN print -> that day's return AND the next day's are NaN.
        prices.iloc[100, prices.columns.get_loc("AAA")] = np.nan
        table, _ = fr.per_holding_regressions(prices, factors, "CAPM")
        self.assertEqual(table.iloc[0]["n"], 297)  # 299 - 2

    def test_window_trims_after_alignment(self) -> None:
        prices, factors = self._setup()
        table, _ = fr.per_holding_regressions(
            prices, factors, "CAPM", window=150)
        self.assertEqual(table.iloc[0]["n"], 150)
        self.assertEqual(table.iloc[0]["end"], factors["date"].iloc[-1])

    def test_empty_inputs_give_empty_frames(self) -> None:
        _, factors = self._setup(10)
        t1, s1 = fr.per_holding_regressions(pd.DataFrame(), factors, "CAPM")
        self.assertTrue(t1.empty and s1.empty)
        prices, _ = self._setup(10)
        t2, s2 = fr.per_holding_regressions(prices, pd.DataFrame(), "CAPM")
        self.assertTrue(t2.empty and s2.empty)


if __name__ == "__main__":
    unittest.main()
