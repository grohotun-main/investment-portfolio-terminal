# tests/test_dip_analytics.py
import math
import unittest
import numpy as np
import pandas as pd
from parsers import dip_analytics as da
from parsers import dip_backtest as db


def _series(vals, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


class UnderwaterTests(unittest.TestCase):
    def test_underwater_zero_at_new_highs(self):
        s = _series([100, 101, 102, 103])
        uw = da.underwater(s)
        self.assertTrue((uw == 0.0).all())

    def test_underwater_measures_drop_from_peak(self):
        s = _series([100, 120, 90])          # peak 120, then 90
        uw = da.underwater(s)
        self.assertAlmostEqual(uw.iloc[-1], 90 / 120 - 1.0)

    def test_drawdown_state_percentiles_complement(self):
        # deep current dip: most history is shallower
        s = _series([100, 110, 120, 118, 115, 80])   # now ~ -33% from 120
        st = da.drawdown_state(s)
        self.assertAlmostEqual(st["current_dd"], 80 / 120 - 1.0, places=6)
        self.assertGreater(st["pct_history_shallower"], 50.0)
        self.assertEqual(st["n_days"], 6)


class EpisodeCountTests(unittest.TestCase):
    def test_counts_distinct_episodes_reaching_depth(self):
        # two separate ~-20% dips that each recover, one shallow -5% dip
        s = _series([100, 80, 100, 95, 100, 78, 100])
        n = da.episodes_reaching(s, -0.15)        # >= 15% deep
        self.assertEqual(n, 2)

    def test_shallow_threshold_excludes_minor_wobble(self):
        s = _series([100, 99, 100, 98, 100])
        self.assertEqual(da.episodes_reaching(s, -0.15), 0)


class FurtherFallTests(unittest.TestCase):
    def test_further_fall_to_trough_before_recovery(self):
        # peak 100 -> enter at 88 (uw=-0.12, passes -0.10 threshold) -> trough 80
        #           -> enter at 80 (uw=-0.20) with 0 further fall -> recover to 100
        # n_complete=2: day@88 (further fall 80/88-1≈-0.0909) + day@80 (0.0)
        # median of {-0.0909, 0.0} is strictly negative
        s = _series([100, 88, 80, 90, 100])
        out = da.conditional_further_fall(s, current_dd=-0.10)
        self.assertEqual(out["n_complete"], 2)
        self.assertEqual(out["n_censored"], 0)
        self.assertLess(out["quantiles"][0.5], 0.0)

    def test_unrecovered_dip_is_censored(self):
        s = _series([100, 90, 70])           # never recovers to 100
        out = da.conditional_further_fall(s, current_dd=-0.05)
        self.assertEqual(out["n_complete"], 0)
        self.assertGreaterEqual(out["n_censored"], 1)


class ForwardReturnTests(unittest.TestCase):
    def test_forward_stats_from_entry_days(self):
        price = _series([100, 90, 95, 100, 105, 110, 120])   # drawdown then up
        tr = price.copy()                                    # TR == price here
        ent = da.entry_index(price, current_dd=-0.05)        # days >= 5% deep
        out = da.forward_return_stats(tr, ent, horizons=(2,))
        self.assertIn(2, out)
        self.assertGreaterEqual(out[2]["hit_rate"], 0.0)
        self.assertEqual(out[2]["n"], out[2]["n"])           # finite count

    def test_entry_index_selects_deep_days(self):
        price = _series([100, 99, 80, 100])
        ent = da.entry_index(price, current_dd=-0.10)
        self.assertIn(price.index[2], ent)                   # the -20% day
        self.assertNotIn(price.index[1], ent)                # the -1% day

    def test_conditional_loss_is_median_of_down_outcomes(self):
        # Forward returns from entries[:5] at h=1: +10%, -9.09%, -4.04%, -5.26%, +10%.
        # cond_loss = median of the three NEGATIVE outcomes (= 90/95-1), and is
        # distinct from p10 (an unconditional percentile over all five).
        tr = _series([100, 110, 99, 95, 90, 99])
        out = da.forward_return_stats(tr, tr.index[:5], horizons=(1,))[1]
        downs = sorted(d for d in
                       (110/100 - 1, 99/110 - 1, 95/99 - 1, 90/95 - 1, 99/90 - 1)
                       if d < 0)
        self.assertAlmostEqual(out["cond_loss"], downs[len(downs) // 2], places=9)
        self.assertAlmostEqual(out["cond_loss"], 90/95 - 1, places=9)
        self.assertNotAlmostEqual(out["cond_loss"], out["p10"], places=4)
        self.assertEqual(out["n"], 5)

    def test_conditional_loss_nan_when_never_down(self):
        tr = _series([100, 101, 102, 103])                   # only gains
        out = da.forward_return_stats(tr, tr.index[:3], horizons=(1,))[1]
        self.assertTrue(math.isnan(out["cond_loss"]))
        self.assertEqual(out["hit_rate"], 1.0)


class YieldAndRecoveryTests(unittest.TestCase):
    def test_ttm_yield(self):
        divs = pd.Series([0.5, 0.5, 0.5, 0.5],
                         index=pd.to_datetime(["2025-03-20", "2025-06-20",
                                               "2025-09-19", "2025-12-19"]))
        y = da.ttm_yield(price_latest=100.0, dividends=divs,
                         asof=pd.Timestamp("2026-01-15"))
        self.assertAlmostEqual(y, 2.0 / 100.0)

    def test_yield_percentile_high_when_price_low(self):
        price = _series([100, 100, 50], start="2024-01-01")   # yield doubles at the dip
        divs = pd.Series([2.0, 2.0, 2.0],
                         index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        out = da.yield_percentile(price, divs)
        self.assertGreaterEqual(out["percentile"], 50.0)

    def test_yield_percentile_empty_dividends_returns_nans(self):
        # non-dividend-paying stock: must not raise, must return NaN sentinels
        out = da.yield_percentile(_series([100, 90, 80]), pd.Series(dtype=float))
        self.assertTrue(math.isnan(out["current_yield"]))
        self.assertTrue(math.isnan(out["percentile"]))

    def test_recovery_rate_counts_recovered_dips(self):
        s = _series([100, 80, 100, 78, 100])     # two >=5% dips, both recover
        out = da.recovery_rate(s)
        self.assertEqual(out["n_episodes"], 2)
        self.assertEqual(out["recovered"], 2)
        self.assertAlmostEqual(out["recovery_rate"], 1.0)


class FurtherFallRegimeTests(unittest.TestCase):
    def test_in_regime_none_matches_default(self):
        # Backward-compat pin: the optional mask defaulting to None must be a
        # byte-identical no-op versus the current call.
        s = _series([100, 88, 80, 90, 100])
        self.assertEqual(
            da.conditional_further_fall(s, current_dd=-0.10),
            da.conditional_further_fall(s, current_dd=-0.10, in_regime=None),
        )

    def test_mask_restricts_completed_entries(self):
        s = _series([100, 88, 80, 90, 100, 88, 70, 100])
        full = da.conditional_further_fall(s, current_dd=-0.10)
        mask = np.array([True, True, True, True, True, False, False, False])
        masked = da.conditional_further_fall(s, current_dd=-0.10, in_regime=mask)
        self.assertLess(masked["n_complete"], full["n_complete"])

    def test_losses_helper_respects_mask(self):
        s = _series([100, 88, 80, 90, 100, 88, 70, 100])
        full = da._further_fall_losses(s, -0.10)
        masked = da._further_fall_losses(
            s, -0.10, in_regime=np.array([True] * 5 + [False] * 3))
        self.assertLess(masked.size, full.size)


class DipBuyVerdictTests(unittest.TestCase):
    BASE = np.array([0.03] * 60 + [-0.025] * 40)        # Omega ~ 1.8 baseline

    def _verdict(self, cond, depth=90.0, rr=0.9, n_rec=50):
        return da.dip_buy_verdict(
            np.asarray(cond, float), self.BASE,
            depth_pctile=depth, rr_percentile=rr, n_recovered_further_fall=n_rec,
            n_boot=200, seed=7)

    def test_shallow_overrides(self):
        self.assertEqual(self._verdict([0.05] * 30, depth=30.0)["band"], "shallow")

    def test_small_sample_inconclusive(self):
        self.assertEqual(self._verdict([0.05] * 5)["band"], "inconclusive")

    def test_nothing_recovered_inconclusive(self):
        self.assertEqual(self._verdict([0.05] * 30, n_rec=0)["band"], "inconclusive")

    def test_strong_finite(self):
        # huge Omega (tiny losses) -> CI lower bound >> baseline, rr high
        self.assertEqual(self._verdict([0.10] * 28 + [-0.001] * 2)["band"], "strong")

    def test_strong_all_gains_inf(self):
        self.assertEqual(self._verdict([0.05] * 20)["band"], "strong")

    def test_neutral_when_rr_low(self):
        self.assertEqual(
            self._verdict([0.10] * 28 + [-0.001] * 2, rr=0.30)["band"], "neutral")

    def test_weak_when_omega_below_one(self):
        self.assertEqual(self._verdict([0.02] * 10 + [-0.08] * 20)["band"], "weak")

    def test_inconclusive_when_omega_nan_ci(self):
        # all break-even -> omega=nan -> bootstrap CI nan -> inconclusive
        self.assertEqual(self._verdict(np.zeros(15))["band"], "inconclusive")


class RewardRiskPercentileTests(unittest.TestCase):
    def test_in_unit_range(self):
        # uptrend with periodic dips; current depth = a real historical depth
        s = _series([100, 90, 110, 100, 85, 120, 110, 95, 130] * 4)
        rr = da.reward_risk_depth_percentile(s, s, current_dd=-0.15, horizon=3)
        self.assertTrue(0.0 <= rr <= 1.0)

    def test_deeper_than_all_history_is_nan(self):
        s = _series([100, 95, 100, 98, 100] * 5)   # never deeper than ~-5%
        rr = da.reward_risk_depth_percentile(s, s, current_dd=-0.50, horizon=3)
        self.assertTrue(math.isnan(rr))


class BootstrapCITests(unittest.TestCase):
    def test_deterministic_under_seed(self):
        x = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.15, 0.0, 0.1, -0.05])
        a = da.stationary_block_bootstrap_ci(np.mean, x, n_boot=200, seed=1)
        b = da.stationary_block_bootstrap_ci(np.mean, x, n_boot=200, seed=1)
        self.assertEqual(a, b)

    def test_point_equals_stat_and_ci_brackets(self):
        x = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.15, 0.0, 0.1, -0.05])
        out = da.stationary_block_bootstrap_ci(np.mean, x, n_boot=500, seed=2)
        self.assertAlmostEqual(out["point"], float(np.mean(x)))
        self.assertLessEqual(out["lo"], out["point"])
        self.assertLessEqual(out["point"], out["hi"])

    def test_constant_series_zero_width(self):
        x = np.array([5.0, 5.0, 5.0, 5.0])
        out = da.stationary_block_bootstrap_ci(np.mean, x, n_boot=100, seed=3)
        self.assertEqual(out["lo"], 5.0)
        self.assertEqual(out["hi"], 5.0)

    def test_degenerate_small_n_is_nan(self):
        out = da.stationary_block_bootstrap_ci(np.mean, np.array([1.0]), seed=4)
        self.assertTrue(math.isnan(out["lo"]) and math.isnan(out["hi"]))


class OmegaRatioTests(unittest.TestCase):
    def test_symmetric_is_one(self):
        self.assertAlmostEqual(da.omega_ratio(np.array([0.1, -0.1])), 1.0)

    def test_all_gains_is_inf(self):
        self.assertEqual(da.omega_ratio(np.array([0.1, 0.2])), float("inf"))

    def test_mixed_hand_value(self):
        # gains=0.4, losses=0.1 -> 4.0
        self.assertAlmostEqual(da.omega_ratio(np.array([0.2, 0.2, -0.1])), 4.0)

    def test_empty_is_nan(self):
        self.assertTrue(math.isnan(da.omega_ratio(np.array([]))))


class RecoveryTimeTests(unittest.TestCase):
    def test_days_to_break_even(self):
        # peak 100; uw=[0,-.10,-.05,0,0]; entries (<=-.05): idx1(90), idx2(95)
        # idx1 -> first day >=90 is idx2 (1 day); idx2 -> first >=95 is idx3 (1 day)
        s = _series([100, 90, 95, 100, 105])
        out = da.conditional_recovery_time(s, current_dd=-0.05)
        self.assertEqual(out["n_complete"], 2)
        self.assertEqual(out["n_censored"], 0)
        self.assertEqual(out["median_days"], 1.0)

    def test_unrecovered_is_censored(self):
        s = _series([100, 90, 80])               # never regains 90 or 80
        out = da.conditional_recovery_time(s, current_dd=-0.05)
        self.assertEqual(out["n_complete"], 0)
        self.assertGreaterEqual(out["n_censored"], 1)
        self.assertTrue(math.isnan(out["median_days"]))

    def test_regime_mask_restricts_entries(self):
        s = _series([100, 90, 100, 100, 90, 100])   # two identical dips
        mask = np.array([True, True, True, False, False, False])
        masked = da.conditional_recovery_time(s, current_dd=-0.05, in_regime=mask)
        full = da.conditional_recovery_time(s, current_dd=-0.05)
        self.assertEqual(masked["n_complete"], 1)   # only the first dip's entry
        self.assertEqual(full["n_complete"], 2)


class ForwardReturnsHelperTests(unittest.TestCase):
    def test_returns_raw_vector_at_horizon(self):
        tr = _series([100, 110, 121, 100, 90])   # idx0..4
        ent = pd.Index([tr.index[0], tr.index[1]])
        r = da.forward_returns(tr, ent, horizon=2)
        # idx0: 121/100-1=0.21 ; idx1: 100/110-1≈-0.0909
        self.assertEqual(r.size, 2)
        self.assertAlmostEqual(r[0], 0.21, places=6)
        self.assertAlmostEqual(r[1], 100 / 110 - 1.0, places=6)

    def test_drops_entries_without_full_horizon(self):
        tr = _series([100, 110, 121])
        ent = tr.index                            # last 2 lack a 2-day horizon
        r = da.forward_returns(tr, ent, horizon=2)
        self.assertEqual(r.size, 1)               # only idx0 qualifies


class RegimeConditionedEntriesTests(unittest.TestCase):
    def test_isolates_same_regime_deep_entries_and_today_label(self):
        # calm dip block, then a deeper stressed dip block, then today at -15% (calm)
        price = _series([100, 90, 80, 100,    # idx0-3: calm, dips to -10%/-20%
                         100, 70, 50, 100,    # idx4-7: stressed, dips to -30%/-50%
                         100, 85])            # idx8-9: calm, today -15%
        labels = pd.Series(["calm"] * 4 + ["stressed"] * 4 + ["calm"] * 2,
                           index=price.index, dtype=object)
        # -0.09 (not -0.10): 90/100-1 = -0.0999… floats just above -0.10, so a
        # -0.10 threshold would exclude the idx1 calm entry we assert is included.
        ent, today = da.regime_conditioned_entries(price, current_dd=-0.09, labels=labels)
        self.assertEqual(today, "calm")
        self.assertIn(price.index[1], ent)       # ~-10% calm (uw=-0.0999... <= -0.09)
        self.assertIn(price.index[2], ent)       # -20% calm
        self.assertIn(price.index[9], ent)       # -15% calm (today's depth day)
        self.assertNotIn(price.index[5], ent)    # -30% but stressed -> excluded
        self.assertNotIn(price.index[6], ent)    # -50% but stressed -> excluded

    def test_today_stressed_selects_stressed_entries(self):
        price = _series([100, 80, 100, 60, 100, 80])   # today idx5 = -20%
        labels = pd.Series(["calm", "calm", "calm", "stressed", "stressed", "stressed"],
                           index=price.index, dtype=object)
        ent, today = da.regime_conditioned_entries(price, current_dd=-0.10, labels=labels)
        self.assertEqual(today, "stressed")
        self.assertIn(price.index[3], ent)       # -40% stressed
        self.assertIn(price.index[5], ent)       # -20% stressed (today)
        self.assertNotIn(price.index[1], ent)    # -20% but calm -> excluded


class TimeUnderwaterCaptionTests(unittest.TestCase):
    """Per-card Buy-the-Dip 'Time underwater' caption text (three branches)."""

    def test_shallow_says_no_real_dip_not_a_median(self):
        # band 'shallow' = dip in the shallower half of history, so "dips this
        # deep" is ~every down-day and break-even is near-immediate. A "median
        # 0 mo" reads like nothing; say there's no real dip and name the depth.
        cap = da.time_underwater_caption(
            "SPY", "shallow", -0.027, median_text="0 mo", p90_text="1 mo",
            n_complete=3759, n_censored=1)
        self.assertIn("Time underwater", cap)
        self.assertIn("no real dip", cap.lower())
        self.assertIn("2.7%", cap)               # depth off the high, named
        self.assertNotIn("median", cap.lower())  # no misleading months figure

    def test_no_recovery_branch_when_n_complete_zero(self):
        cap = da.time_underwater_caption(
            "SPY", "weak", -0.45, median_text="—", p90_text="—",
            n_complete=0, n_censored=12)
        self.assertIn("no dip this deep has returned to break-even", cap)
        self.assertIn("SPY", cap)

    def test_meaningful_branch_reports_median_p90_and_censored(self):
        cap = da.time_underwater_caption(
            "SCHD", "neutral", -0.18, median_text="2 mo", p90_text="5 mo",
            n_complete=177, n_censored=1)
        self.assertIn("median **2 mo**", cap)
        self.assertIn("**5 mo** in the slow 1-in-10 case", cap)
        self.assertIn("1 of 178 comparable dips never did", cap)

    def test_meaningful_branch_omits_censored_clause_when_none(self):
        cap = da.time_underwater_caption(
            "SCHD", "strong", -0.18, median_text="2 mo", p90_text="5 mo",
            n_complete=200, n_censored=0)
        self.assertNotIn("never did", cap)


def _reference_indices(rng, n, p_new):
    """The pre-vectorization per-resample index loop, verbatim (Politis-Romano
    stationary bootstrap). Kept here as the behavioral reference."""
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p_new
    starts = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
    return idx


class StationaryBootstrapVectorizationTests(unittest.TestCase):
    """S1 gate: the vectorized bootstrap must be BIT-IDENTICAL to the original.
    test_pinned_ci_outputs passes before AND after the refactor
    (characterization); test_indices_match_reference fails before (helper
    absent) and locks the draw order + index arithmetic after."""

    def test_pinned_ci_outputs(self):
        cases = [
            (np.linspace(-0.05, 0.05, 30),
             (1.0, 0.22149944465, 3.552667509482)),
            (np.sin(np.arange(200)) * 0.02,
             (1.014366699533, 0.920877433447, 1.113689590112)),
            (np.concatenate([np.full(25, 0.01), np.full(25, -0.02)]),
             (0.5, 0.140224358974, 1.772727272727)),
        ]
        for x, (pt, lo, hi) in cases:
            r = da.stationary_block_bootstrap_ci(da.omega_ratio, x)
            self.assertAlmostEqual(r["point"], pt, places=12)
            self.assertAlmostEqual(r["lo"], lo, places=12)
            self.assertAlmostEqual(r["hi"], hi, places=12)

    def test_indices_match_reference(self):
        for n, p_new, seed in [(5, 0.5, 0), (30, 1.0 / 21, 1),
                               (200, 1.0 / 21, 2), (3, 1.0, 3)]:
            r1 = np.random.default_rng(seed)
            r2 = np.random.default_rng(seed)
            for _ in range(5):   # consecutive resamples share rng state
                got = da._stationary_bootstrap_indices(r1, n, p_new)
                ref = _reference_indices(r2, n, p_new)
                np.testing.assert_array_equal(got, ref)


def _dip_series(n1=300, n2=60, n3=120):
    """Ramp 100->180, crash to 135, partial recovery to 155: today is a real
    ~14% dip with multi-depth history. tr == price is fine for equality tests.
    n3=120 leaves no conditional entry a full 252d forward window (degenerate
    verdict); n3=400 gives ~190 completed outcomes (finite-Omega verdict)."""
    px = np.concatenate([np.linspace(100.0, 180.0, n1),
                         np.linspace(180.0, 135.0, n2),
                         np.linspace(135.0, 155.0, n3)])
    idx = pd.bdate_range("2019-01-01", periods=px.size)
    price = pd.Series(px, index=idx)
    return price, price.copy()


def _nan_eq(a, b):
    """Deep equality where NaN == NaN (dicts/sequences/floats)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_nan_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


class DipVerdictBlockTests(unittest.TestCase):
    """S2 gate: the extracted verdict block must equal dip_card_data's fields
    on identical inputs - single-source by construction."""

    def test_block_matches_card(self):
        from terminal.dip_service import dip_card_data
        for n3 in (120, 400):    # degenerate corner + finite-Omega verdict
            price, tr = _dip_series(n3=n3)
            blk = da.dip_verdict_block(price, tr, horizons=(21, 63, 126, 252))
            card = dip_card_data("TEST", price, tr, pd.Series(dtype=float))
            pairs = [("verdict", card.verdict), ("fwd_full", card.fwd_full),
                     ("fwd_reg", card.fwd_reg), ("ff_full", card.ff_full),
                     ("ff_reg", card.ff_reg), ("use_reg_ff", card.use_reg_ff),
                     ("ff_head", card.ff_head),
                     ("today_regime", card.today_regime),
                     ("state", card.state)]
            for key, card_val in pairs:
                self.assertTrue(_nan_eq(blk[key], card_val),
                                f"n3={n3} field {key}: {blk[key]!r} != {card_val!r}")

    def test_verdict_horizon_required(self):
        price, tr = _dip_series()
        with self.assertRaises(ValueError):
            da.dip_verdict_block(price, tr, horizons=(21, 63))


class TestHistoryDepthCaveat(unittest.TestCase):
    """The referee's load-bearing caveat as product disclosure: edge-claiming
    verdicts on histories thinner than the burn-in that earned 'validated'
    (10y; 5y returned not_supported) must say so. Disclosure only — no band or
    number changes."""

    @staticmethod
    def _series(years: float) -> pd.Series:
        # Span the requested CALENDAR years, then take the business days inside
        # them. Do NOT build this as `periods=years*252`: 252 is the trading-day
        # convention (holidays excluded) but bdate_range only drops weekends
        # (~261/yr), so a 252-per-year count silently spans ~3.5% short — 10y
        # would come out 9.65y. days/365.25 mirrors history_span_years' own
        # formula, so the fixture round-trips exactly.
        start = pd.Timestamp("2000-01-03")
        end = start + pd.Timedelta(days=round(years * 365.25))
        idx = pd.bdate_range(start, end)
        return pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)

    def test_span_years_matches_calendar_span(self):
        s = self._series(10.0)
        got = da.history_span_years(s)
        self.assertAlmostEqual(got, (s.index[-1] - s.index[0]).days / 365.25,
                               places=9)
        self.assertAlmostEqual(got, 10.0, delta=0.15)

    def test_span_years_two_years(self):
        self.assertAlmostEqual(da.history_span_years(self._series(2.0)),
                               2.0, delta=0.1)

    def test_span_years_degenerate_series(self):
        idx = pd.bdate_range("2000-01-03", periods=1)
        self.assertEqual(da.history_span_years(pd.Series([1.0], index=idx)),
                         0.0)
        self.assertEqual(
            da.history_span_years(pd.Series(dtype=float)), 0.0)

    def test_caveat_fires_on_edge_bands(self):
        for band in ("strong", "neutral"):
            with self.subTest(band=band):
                txt = da.history_depth_caveat("SPY", band, 2.0)
                self.assertTrue(txt)
                self.assertIn("2 years", txt)
                self.assertIn("SPY", txt)

    def test_caveat_silent_on_non_edge_bands(self):
        for band in ("weak", "shallow", "inconclusive"):
            with self.subTest(band=band):
                self.assertEqual(
                    da.history_depth_caveat("SPY", band, 2.0), "")

    def test_caveat_silent_on_deep_history(self):
        self.assertEqual(da.history_depth_caveat("SPY", "strong", 12.0),
                         "")

    def test_caveat_silent_at_threshold(self):
        # >= the trusted depth is trusted; the boundary itself must not fire.
        self.assertEqual(
            da.history_depth_caveat(
                "SPY", "strong", da.VERDICT_TRUSTED_HISTORY_YEARS), "")

    def test_caveat_fires_just_below_threshold(self):
        txt = da.history_depth_caveat(
            "SPY", "strong", da.VERDICT_TRUSTED_HISTORY_YEARS - 0.01)
        self.assertTrue(txt)
        # Truthiness alone would let a self-contradicting sentence through.
        self.assertIn("9 years", txt)

    def test_caveat_span_never_reaches_the_threshold(self):
        # The [9.5, 10.0) trap: rounding would render these as "10 years" and
        # produce "only 10 years — demonstrated only on 10+ years". The rendered
        # span must stay strictly below the threshold for every sub-threshold
        # input, or the disclosure contradicts itself at its own boundary.
        thr = da.VERDICT_TRUSTED_HISTORY_YEARS
        for years in (9.5, 9.7, 9.99, thr - 1e-9):
            with self.subTest(years=years):
                txt = da.history_depth_caveat("SPY", "strong", years)
                self.assertTrue(txt)
                self.assertIn("9 years", txt)
                self.assertNotIn(f"only {thr:.0f} years", txt)

    def test_caveat_silent_on_unknown_years(self):
        self.assertEqual(
            da.history_depth_caveat("SPY", "strong", float("nan")), "")

    def test_caveat_sub_year_reads_naturally(self):
        txt = da.history_depth_caveat("TESTQ", "strong", 0.75)
        self.assertIn("under 1 year", txt)
        self.assertNotIn("0 years", txt)

    def test_caveat_singular_year(self):
        txt = da.history_depth_caveat("TESTQ", "strong", 1.2)
        self.assertIn("only 1 year of", txt)
        self.assertNotIn("only 1 years of", txt)

    def test_caveat_count_is_accurate_not_floored(self):
        # The fixture's real span is 729d = 1.9959y: flooring would disclose
        # "1 year", a ~50% understatement of the very fact being disclosed. Round.
        txt = da.history_depth_caveat("SCHD", "strong", 1.9959)
        self.assertIn("2 years", txt)
        self.assertNotIn("1 year of", txt)

    def test_caveat_never_overstates_the_evidence(self):
        # The referee tested 5y and 10y only — it never located the boundary.
        # The wording must claim what was demonstrated, not a proven minimum.
        txt = da.history_depth_caveat("SPY", "strong", 2.0)
        self.assertIn("10+ years", txt)

    def test_edge_bands_agree_with_the_referee(self):
        # dip_backtest defines EDGE_BANDS per its own spec §4; the caveat fires
        # on exactly the set the registered primary metric measured. If these
        # ever diverge, the disclosure stops matching the evidence.
        self.assertEqual(set(da.VERDICT_EDGE_BANDS),
                         set(db.EDGE_BANDS))


class TestVerdictBlockHistoryYears(unittest.TestCase):
    """dip_verdict_block carries the as-of history span so the card can disclose
    it. Additive: the referee replays this same block on truncated inputs, so
    each evaluation naturally reports its own as-of depth."""

    @classmethod
    def setUpClass(cls):
        n = 252 * 3
        idx = pd.bdate_range("2010-01-04", periods=n)
        rng = np.random.default_rng(0)
        walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        cls.price = pd.Series(walk, index=idx)
        cls.tr = cls.price.copy()

    def test_block_exposes_history_years(self):
        blk = da.dip_verdict_block(self.price, self.tr)
        self.assertIn("history_years", blk)
        self.assertAlmostEqual(blk["history_years"],
                               da.history_span_years(self.price),
                               places=9)

    def test_block_history_years_tracks_truncation(self):
        # The no-look-ahead property the referee depends on: a truncated input
        # reports the truncated span, never the full one.
        half = self.price.iloc[: len(self.price) // 2]
        blk = da.dip_verdict_block(half, self.tr.iloc[: len(half)])
        self.assertLess(blk["history_years"],
                        da.history_span_years(self.price))
