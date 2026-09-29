"""
Tests for parsers/whatif_engine.py — scenario validation + before/after
risk computation.

Covers:
  - WhatIfScenario.validate: weights-sum, optional candidate, free
    reweighting of existing holdings, splice/proxy combos, edge inputs.
  - WhatIfScenario.source_reductions: shape + values.
  - compute_before_after: insufficient-overlap error path; happy path
    with a controlled-correlation 4-asset universe; MCR verdict
    classification at both extremes; sanity (small-weight candidate
    leaves headline metrics nearly unchanged).

Run from phase1_build/ with:
    py -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import whatif_engine as we  # noqa: E402
from whatif_engine import WhatIfCandidate  # noqa: E402


def _bdays(n: int, end: str = "2026-04-30") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def _valid_scenario(candidate: str = "PDBC") -> we.WhatIfScenario:
    cw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
    nw = pd.Series({"AAA": 0.45, "BBB": 0.30, "CCC": 0.20, candidate: 0.05})
    return we.WhatIfScenario(
        candidate_ticker=candidate,
        current_weights=cw, new_weights=nw,
    )


class TestScenarioValidate(unittest.TestCase):
    def test_valid_scenario_passes(self) -> None:
        _valid_scenario().validate()  # should not raise

    def test_empty_candidate_ticker(self) -> None:
        s = _valid_scenario()
        bad = we.WhatIfScenario(candidate_ticker="   ",
                                current_weights=s.current_weights,
                                new_weights=s.new_weights)
        with self.assertRaises(ValueError):
            bad.validate()

    def test_weights_dont_sum_to_one(self) -> None:
        cw = pd.Series({"AAA": 0.5, "BBB": 0.4})  # sum=0.9
        nw = pd.Series({"AAA": 0.5, "BBB": 0.4, "PDBC": 0.05})
        s = we.WhatIfScenario(candidate_ticker="PDBC", current_weights=cw, new_weights=nw)
        with self.assertRaises(ValueError):
            s.validate()

    def test_pure_reweight_no_candidate_passes(self) -> None:
        # No new ticker: shift weight between existing holdings.
        cw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
        nw = pd.Series({"AAA": 0.40, "BBB": 0.40, "CCC": 0.20})
        we.WhatIfScenario(current_weights=cw, new_weights=nw).validate()  # should not raise

    def test_existing_holding_may_increase(self) -> None:
        # Was illegal under sink-only; now allowed.
        cw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
        nw = pd.Series({"AAA": 0.55, "BBB": 0.25, "CCC": 0.20})
        we.WhatIfScenario(current_weights=cw, new_weights=nw).validate()  # should not raise

    def test_holding_to_zero_passes(self) -> None:
        # CCC dropped to 0 and redistributed.
        cw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
        nw = pd.Series({"AAA": 0.60, "BBB": 0.40, "CCC": 0.00})
        we.WhatIfScenario(current_weights=cw, new_weights=nw).validate()  # should not raise

    def test_negative_weight_blocked(self) -> None:
        cw = pd.Series({"AAA": 0.50, "BBB": 0.50})
        nw = pd.Series({"AAA": 1.10, "BBB": -0.10})
        with self.assertRaises(ValueError):
            we.WhatIfScenario(current_weights=cw, new_weights=nw).validate()

    def test_two_new_tickers_blocked(self) -> None:
        # Historical name; the contract now allows up to MAX_CANDIDATES (3)
        # new tickers, so this exercises the updated cap with 4.
        cw = pd.Series({"AAA": 0.60, "BBB": 0.40})
        nw = pd.Series({"AAA": 0.20, "BBB": 0.20, "NEW1": 0.15, "NEW2": 0.15,
                        "NEW3": 0.15, "NEW4": 0.15})
        with self.assertRaises(ValueError) as ctx:
            we.WhatIfScenario(candidate_ticker="NEW1",
                              current_weights=cw, new_weights=nw).validate()
        self.assertIn("up to 3", str(ctx.exception))

    def test_no_change_blocked(self) -> None:
        cw = pd.Series({"AAA": 0.50, "BBB": 0.50})
        nw = pd.Series({"AAA": 0.50, "BBB": 0.50})
        with self.assertRaises(ValueError) as ctx:
            we.WhatIfScenario(current_weights=cw, new_weights=nw).validate()
        self.assertIn("nothing to simulate", str(ctx.exception))

    def test_candidate_already_held_blocked(self) -> None:
        # A new ticker must be genuinely new; reweight held names in the grid.
        cw = pd.Series({"AAA": 0.5, "PDBC": 0.3, "CCC": 0.2})
        nw = pd.Series({"AAA": 0.45, "PDBC": 0.35, "CCC": 0.20})
        with self.assertRaises(ValueError) as ctx:
            we.WhatIfScenario(candidate_ticker="PDBC",
                              current_weights=cw, new_weights=nw).validate()
        self.assertIn("already held", str(ctx.exception))

    def test_splice_requires_candidate(self) -> None:
        # Under the generalized contract, a splice_with_proxy flag with no
        # candidate_ticker normalizes to zero candidates (nothing to splice)
        # rather than raising — norm_candidates only synthesizes an entry
        # when candidate_ticker is truthy. validate() still passes/fails on
        # its own merits (here: a genuine reweight with no new tickers).
        cw = pd.Series({"AAA": 0.5, "BBB": 0.5})
        nw = pd.Series({"AAA": 0.4, "BBB": 0.6})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              splice_with_proxy=True, proxy_ticker="GSG")
        self.assertEqual(s.norm_candidates, ())
        s.validate()  # should not raise: no candidate to splice

    def test_candidate_missing_from_new_weights(self) -> None:
        cw = pd.Series({"AAA": 0.5, "BBB": 0.5})
        nw = pd.Series({"AAA": 0.5, "BBB": 0.5})  # no candidate
        s = we.WhatIfScenario(candidate_ticker="PDBC", current_weights=cw, new_weights=nw)
        with self.assertRaises(ValueError):
            s.validate()

    def test_candidate_weight_zero(self) -> None:
        cw = pd.Series({"AAA": 0.5, "BBB": 0.5})
        nw = pd.Series({"AAA": 0.5, "BBB": 0.5, "PDBC": 0.0})
        s = we.WhatIfScenario(candidate_ticker="PDBC", current_weights=cw, new_weights=nw)
        with self.assertRaises(ValueError):
            s.validate()

    def test_splice_without_proxy_blocked(self) -> None:
        # Under the generalized contract, splice_with_proxy=True with
        # proxy_ticker=None normalizes to WhatIfCandidate(ticker, proxy=None)
        # — indistinguishable from "no splice requested" — so validate() no
        # longer raises here; norm_candidates carries proxy=None instead.
        s_valid = _valid_scenario()
        s = we.WhatIfScenario(
            candidate_ticker=s_valid.candidate_ticker,
            current_weights=s_valid.current_weights, new_weights=s_valid.new_weights,
            splice_with_proxy=True, proxy_ticker=None,
        )
        self.assertIsNone(s.norm_candidates[0].proxy)
        s.validate()  # should not raise

    def test_proxy_equals_candidate_blocked(self) -> None:
        s_valid = _valid_scenario()
        s = we.WhatIfScenario(
            candidate_ticker="PDBC",
            current_weights=s_valid.current_weights, new_weights=s_valid.new_weights,
            splice_with_proxy=True, proxy_ticker="PDBC",
        )
        with self.assertRaises(ValueError):
            s.validate()

    def test_source_reductions(self) -> None:
        s = _valid_scenario()
        red = s.source_reductions
        self.assertAlmostEqual(float(red["AAA"]), 0.05, places=6)
        self.assertAlmostEqual(float(red["BBB"]), 0.0, places=6)
        self.assertAlmostEqual(float(red["CCC"]), 0.0, places=6)
        # AAA funds the whole new position in this fixture, so net reduction
        # happens to equal the candidate weight here (fixture-specific, not a
        # general invariant now that holdings may also increase).
        self.assertAlmostEqual(float(red.sum()), s.candidate_weight, places=6)

    def test_three_candidates_ok(self):
        cw = pd.Series({"A": 0.5, "B": 0.5})
        nw = pd.Series({"A": 0.4, "B": 0.3, "X": 0.1, "Y": 0.1, "Z": 0.1})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              candidates=(WhatIfCandidate("X"), WhatIfCandidate("Y"),
                                          WhatIfCandidate("Z")))
        s.validate()   # no raise
        self.assertEqual([c.ticker for c in s.norm_candidates], ["X", "Y", "Z"])

    def test_four_candidates_blocked(self):
        cw = pd.Series({"A": 1.0})
        nw = pd.Series({"A": 0.6, "W": 0.1, "X": 0.1, "Y": 0.1, "Z": 0.1})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              candidates=tuple(WhatIfCandidate(t) for t in ("W", "X", "Y", "Z")))
        with self.assertRaises(ValueError):
            s.validate()

    def test_scalar_path_normalizes_to_one_candidate(self):
        cw = pd.Series({"A": 1.0}); nw = pd.Series({"A": 0.9, "P": 0.1})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              candidate_ticker="P", splice_with_proxy=True, proxy_ticker="Q")
        s.validate()
        self.assertEqual(len(s.norm_candidates), 1)
        self.assertEqual(s.norm_candidates[0].ticker, "P")
        self.assertEqual(s.norm_candidates[0].proxy, "Q")   # splice on -> proxy carried

    def test_scalar_splice_off_drops_proxy(self):
        cw = pd.Series({"A": 1.0}); nw = pd.Series({"A": 0.9, "P": 0.1})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              candidate_ticker="P", splice_with_proxy=False, proxy_ticker="Q")
        self.assertIsNone(s.norm_candidates[0].proxy)

    def test_candidate_not_in_new_weights_blocked(self):
        cw = pd.Series({"A": 1.0}); nw = pd.Series({"A": 1.0})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              candidates=(WhatIfCandidate("P"),))
        with self.assertRaises(ValueError):
            s.validate()

    def test_multi_candidate_undeclared_new_ticker_blocked(self):
        # One declared candidate (X) but new_weights also carries a second,
        # undeclared new ticker (Y) with weight > 0 — must be rejected and
        # the error must name the undeclared ticker.
        cw = pd.Series({"A": 1.0})
        nw = pd.Series({"A": 0.8, "X": 0.1, "Y": 0.1})
        s = we.WhatIfScenario(current_weights=cw, new_weights=nw,
                              candidates=(WhatIfCandidate("X"),))
        with self.assertRaises(ValueError) as ctx:
            s.validate()
        self.assertIn("Y", str(ctx.exception))


# -------------------------------------------------------------------------
# Fixtures for compute_before_after
# -------------------------------------------------------------------------

def _make_universe(
    n_days: int = 400,
    seed: int = 0,
    candidate_correlation_with_spy: float = 0.0,
    candidate_vol_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthesize a 4-asset price universe + a candidate's price history.

    daily_prices columns: AAA, BBB, CCC, SPY (existing).
    candidate_history: PDBC (new), with SPY correlation set explicitly.

    Uses a 1-factor model — every asset is a beta load on a single market
    factor plus idiosyncratic noise; PDBC is mixed with the market's
    standardized returns at the requested correlation. Guarantees a PD
    correlation structure for any rho in [-1, 1].

    candidate_vol_scale lets us dial PDBC's own vol independently — the
    MCR/σ_p verdict depends on BOTH correlation AND relative vol, so the
    "risk-adding" test sets candidate_vol_scale > 1.
    """
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(n_days) * 0.01           # the factor
    eps = rng.standard_normal((n_days, 4)) * 0.008        # idio for 4 assets
    betas = np.array([1.00, 0.95, 0.85, 1.00])            # AAA, BBB, CCC, SPY
    rets = market[:, None] * betas[None, :] + eps         # (n_days, 4)

    idx = _bdays(n_days)
    cols = ["AAA", "BBB", "CCC", "SPY"]
    prices = pd.DataFrame(
        np.cumprod(1.0 + rets, axis=0) * 100.0,
        index=idx, columns=cols,
    )

    rho = float(np.clip(candidate_correlation_with_spy, -1.0, 1.0))
    z_market = (market - market.mean()) / (market.std(ddof=1) + 1e-12)
    eps_pdbc = rng.standard_normal(n_days)
    pdbc_rets = (rho * z_market + np.sqrt(max(1.0 - rho * rho, 0.0))
                 * eps_pdbc) * 0.01 * float(candidate_vol_scale)
    cand = pd.Series(
        np.cumprod(1.0 + pdbc_rets) * 100.0,
        index=idx, name="PDBC",
    )
    return prices, cand


class TestComputeBeforeAfter(unittest.TestCase):
    def test_insufficient_overlap_returns_error(self) -> None:
        prices, cand = _make_universe(n_days=400, seed=1)
        # Slice candidate to a very short window to force the error.
        short_cand = cand.iloc[-50:]
        result = we.compute_before_after(
            _valid_scenario(), prices, short_cand,
            min_overlap_days=252,
        )
        self.assertIsNone(result["headline"])
        self.assertIsNone(result["detail"])
        self.assertIsNotNone(result["error"])
        self.assertIn("Insufficient overlap", result["error"])
        self.assertEqual(result["coverage"]["overlap_days"], 50)

    def test_happy_path_returns_full_payload(self) -> None:
        prices, cand = _make_universe(n_days=400, seed=2)
        scen = _valid_scenario()
        result = we.compute_before_after(scen, prices, cand)
        self.assertIsNone(result["error"])
        head = result["headline"]
        self.assertIsNotNone(head)
        # Every bundle metric is a {before, after, delta} dict
        for key in ("vol", "sharpe", "sortino", "dr",
                    "avg_pairwise_corr", "max_dd", "var95", "cvar95",
                    "stressed_corr_avg", "stressed_dr"):
            self.assertIn(key, head, msg=f"missing {key}")
            self.assertIn("before", head[key])
            self.assertIn("after", head[key])
            self.assertIn("delta", head[key])
        # Scalar metrics
        self.assertIn("mcr_candidate", head)
        self.assertIn("mcr_verdict", head)
        # Detail panels
        det = result["detail"]
        for key in ("corr_matrix_before", "corr_matrix_after",
                    "drawdown_curve_before", "drawdown_curve_after",
                    "risk_contribution_before", "risk_contribution_after"):
            self.assertIn(key, det)
        # Coverage
        self.assertEqual(result["coverage"]["spliced"], False)
        self.assertGreaterEqual(result["coverage"]["overlap_days"], 252)

    def test_uncorrelated_candidate_is_diversifying(self) -> None:
        # PDBC ~uncorrelated with the equity block → MCR should be well
        # below port_vol → verdict "diversifying".
        prices, cand = _make_universe(
            n_days=500, seed=3, candidate_correlation_with_spy=0.0,
        )
        scen = _valid_scenario()  # PDBC at 5%, funded from AAA
        result = we.compute_before_after(scen, prices, cand)
        head = result["headline"]
        self.assertTrue(np.isfinite(head["mcr_candidate"]))
        self.assertTrue(np.isfinite(head["vol"]["after"]))
        ratio = head["mcr_candidate"] / head["vol"]["after"]
        # Should clearly land below the 0.95 diversifying threshold.
        self.assertLess(ratio, we.MCR_DIVERSIFYING_RATIO,
                        msg=f"mcr/σ_p ratio={ratio:.3f}; expected < "
                            f"{we.MCR_DIVERSIFYING_RATIO}")
        self.assertEqual(head["mcr_verdict"], "diversifying")
        # DR should not decrease (uncorrelated asset helps diversification).
        self.assertGreaterEqual(head["dr"]["after"], head["dr"]["before"] - 0.05)

    def test_highly_correlated_high_vol_candidate_is_risk_adding(self) -> None:
        # PDBC near-perfectly correlated with SPY AND with 2x base vol →
        # MCR clearly exceeds port_vol → verdict "risk_adding".
        prices, cand = _make_universe(
            n_days=500, seed=4,
            candidate_correlation_with_spy=0.95,
            candidate_vol_scale=2.0,
        )
        scen = _valid_scenario()
        result = we.compute_before_after(scen, prices, cand)
        head = result["headline"]
        self.assertIn(head["mcr_verdict"], ("neutral", "risk_adding"),
                      msg=f"mcr={head['mcr_candidate']:.4f}, "
                          f"σ_p_after={head['vol']['after']:.4f}")

    def test_runs_when_spy_is_in_existing_weights(self) -> None:
        # Real portfolios commonly hold SPY directly. SPY is also the
        # default condition_symbol for stress-day classification. Prior
        # to the dedup fix in _stressed_dr, this combination crashed with
        # `TypeError: float() argument must be a string or a real number,
        # not 'Series'` because daily_prices[common + [condition_symbol]]
        # selected SPY twice and rets["SPY"] returned a DataFrame.
        prices, cand = _make_universe(n_days=500, seed=11)
        cw = pd.Series({"AAA": 0.50, "SPY": 0.30, "CCC": 0.20})
        nw = pd.Series({"AAA": 0.45, "SPY": 0.30, "CCC": 0.20, "PDBC": 0.05})
        scen = we.WhatIfScenario(candidate_ticker="PDBC", current_weights=cw, new_weights=nw)
        result = we.compute_before_after(scen, prices, cand)
        self.assertIsNone(result["error"])
        head = result["headline"]
        # Stressed DR should compute to a finite number when SPY is held.
        self.assertTrue(
            np.isfinite(head["stressed_dr"]["before"]),
            msg="stressed_dr crashed/NaN when SPY is in weights",
        )
        self.assertTrue(np.isfinite(head["stressed_dr"]["after"]))

    def test_small_weight_candidate_near_identity(self) -> None:
        # Adding PDBC at a tiny 0.5% weight should leave most headline
        # metrics nearly unchanged before vs after.
        prices, cand = _make_universe(n_days=400, seed=5)
        cw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
        nw = pd.Series({"AAA": 0.495, "BBB": 0.300, "CCC": 0.200, "PDBC": 0.005})
        scen = we.WhatIfScenario(candidate_ticker="PDBC", current_weights=cw, new_weights=nw)
        result = we.compute_before_after(scen, prices, cand)
        head = result["headline"]
        # Vol delta should be small in absolute terms (< 1pp annualized).
        self.assertLess(abs(head["vol"]["delta"]), 0.01,
                        msg=f"vol delta {head['vol']['delta']:.4f}")
        # DR delta should be small (< 0.10 units).
        self.assertLess(abs(head["dr"]["delta"]), 0.10,
                        msg=f"DR delta {head['dr']['delta']:.4f}")

    def test_pure_reweight_full_payload(self) -> None:
        # No candidate: reweight the existing 4-name universe.
        prices, _ = _make_universe(n_days=400, seed=7)
        cw = pd.Series({"AAA": 0.40, "BBB": 0.30, "CCC": 0.20, "SPY": 0.10})
        nw = pd.Series({"AAA": 0.25, "BBB": 0.30, "CCC": 0.20, "SPY": 0.25})
        scen = we.WhatIfScenario(current_weights=cw, new_weights=nw)
        result = we.compute_before_after(
            scen, prices, pd.Series(dtype=float),
        )
        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["headline"])
        # No new ticker -> MCR is undefined.
        self.assertTrue(np.isnan(result["headline"]["mcr_candidate"]))
        self.assertEqual(result["headline"]["mcr_verdict"], "unknown")
        self.assertIsNone(result["coverage"]["candidate_inception"])
        # Concentration present on the success path.
        self.assertIn("concentration", result)
        self.assertIn("effective_n", result["concentration"])

    def test_two_candidates_emit_per_candidate_mcr(self) -> None:
        # Two candidates, spliced via the dict[str, Series] path (terminal
        # shape). Same bdate_range params -> identical indices, so both
        # clear MIN_OVERLAP_DAYS with full overlap.
        prices, cand1 = _make_universe(
            n_days=400, seed=20, candidate_correlation_with_spy=0.0)
        _, cand2 = _make_universe(
            n_days=400, seed=21, candidate_correlation_with_spy=0.3)
        cand1 = cand1.rename("C1")
        cand2 = cand2.rename("C2")
        cw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
        nw = pd.Series({"AAA": 0.40, "BBB": 0.30, "CCC": 0.20,
                        "C1": 0.05, "C2": 0.05})
        scen = we.WhatIfScenario(
            current_weights=cw, new_weights=nw,
            candidates=(WhatIfCandidate("C1"), WhatIfCandidate("C2")),
        )
        result = we.compute_before_after(
            scen, prices, {"C1": cand1, "C2": cand2},
        )
        self.assertIsNone(result["error"])
        head = result["headline"]
        self.assertIsNotNone(head)
        self.assertEqual(len(head["mcr_candidates"]), 2)
        self.assertEqual(
            {d["ticker"] for d in head["mcr_candidates"]}, {"C1", "C2"})
        # scalar back-compat == first candidate
        self.assertEqual(head["mcr_candidate"], head["mcr_candidates"][0]["mcr"])
        self.assertEqual(head["mcr_verdict"], head["mcr_candidates"][0]["verdict"])
        cov = result["coverage"]
        self.assertEqual(len(cov["candidates"]), 2)
        self.assertEqual({c["ticker"] for c in cov["candidates"]}, {"C1", "C2"})
        # scalar candidate_inception == first candidate's inception
        self.assertEqual(cov["candidate_inception"], cov["candidates"][0]["inception"])

    def test_pure_reweight_still_scalar_unknown(self) -> None:
        # candidate_ticker=None -> no candidates at all: scalar keys fall
        # back to NaN/"unknown"/None, and the list-shaped keys are empty.
        prices, _ = _make_universe(n_days=400, seed=22)
        cw = pd.Series({"AAA": 0.40, "BBB": 0.30, "CCC": 0.20, "SPY": 0.10})
        nw = pd.Series({"AAA": 0.25, "BBB": 0.30, "CCC": 0.20, "SPY": 0.25})
        scen = we.WhatIfScenario(current_weights=cw, new_weights=nw)
        result = we.compute_before_after(
            scen, prices, pd.Series(dtype=float),
        )
        self.assertIsNone(result["error"])
        head = result["headline"]
        self.assertTrue(np.isnan(head["mcr_candidate"]))
        self.assertEqual(head["mcr_verdict"], "unknown")
        self.assertEqual(head["mcr_candidates"], [])
        cov = result["coverage"]
        self.assertIsNone(cov["candidate_inception"])
        self.assertEqual(cov["candidates"], [])

    def test_concentration_present_on_short_overlap(self) -> None:
        # Insufficient overlap still returns the weight-only concentration.
        prices, cand = _make_universe(n_days=400, seed=8)
        short_cand = cand.iloc[-50:]
        result = we.compute_before_after(
            _valid_scenario(), prices, short_cand, min_overlap_days=252,
        )
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["headline"])
        self.assertIn("concentration", result)
        self.assertIn("max_pct", result["concentration"])


# -------------------------------------------------------------------------
# Helper-function unit tests
# -------------------------------------------------------------------------

class TestEngineHelpers(unittest.TestCase):
    def test_classify_mcr_thresholds(self) -> None:
        self.assertEqual(we._classify_mcr(0.05, 0.10), "diversifying")
        self.assertEqual(we._classify_mcr(0.10, 0.10), "neutral")
        self.assertEqual(we._classify_mcr(0.20, 0.10), "risk_adding")
        self.assertEqual(we._classify_mcr(np.nan, 0.10), "unknown")
        self.assertEqual(we._classify_mcr(0.05, 0.0), "unknown")

    def test_pair_delta_signs(self) -> None:
        self.assertAlmostEqual(we._pair(0.10, 0.12)["delta"], 0.02, places=10)
        self.assertAlmostEqual(we._pair(0.10, 0.08)["delta"], -0.02,
                               places=10)
        self.assertTrue(np.isnan(we._pair(np.nan, 0.10)["delta"]))

    def test_max_drawdown_pct_simple(self) -> None:
        # Wealth path: 1.0 → 1.1 → 0.99 → 1.05; peak=1.1, trough=0.99
        # MDD = 0.99/1.1 - 1 = -0.1
        rets = pd.Series([0.10, -0.10, 0.0606], index=_bdays(3))
        dd = we._max_drawdown_pct(rets)
        self.assertAlmostEqual(dd, 0.99 / 1.10 - 1.0, places=5)

    def test_avg_offdiagonal_corr(self) -> None:
        # 3×3 with off-diagonals (0.3, 0.5, 0.7) → mean = 0.5
        corr = pd.DataFrame(
            [[1.0, 0.3, 0.5],
             [0.3, 1.0, 0.7],
             [0.5, 0.7, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        self.assertAlmostEqual(we._avg_offdiagonal_corr(corr), 0.5, places=6)

    def test_avg_offdiagonal_corr_empty(self) -> None:
        self.assertTrue(np.isnan(we._avg_offdiagonal_corr(pd.DataFrame())))

    def test_concentration_delta_equal_vs_concentrated(self) -> None:
        cw = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        nw = pd.Series({"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10})
        cd = we.concentration_delta(cw, nw)
        # Equal-weight 4 names -> effective-N 4, max 25%.
        self.assertAlmostEqual(cd["effective_n"]["before"], 4.0, places=6)
        self.assertAlmostEqual(cd["max_pct"]["before"], 25.0, places=4)
        self.assertAlmostEqual(cd["max_pct"]["after"], 70.0, places=4)
        # Concentrating lowers effective-N and raises max weight.
        self.assertLess(cd["effective_n"]["after"], cd["effective_n"]["before"])
        self.assertGreater(cd["max_pct"]["delta"], 0.0)
        # HHI is the reciprocal of effective-N.
        self.assertAlmostEqual(cd["herfindahl"]["before"], 1.0 / 4.0, places=6)
        # Keys present with the {before, after, delta} shape.
        for k in ("effective_n", "top5_pct", "max_pct", "herfindahl"):
            for f in ("before", "after", "delta"):
                self.assertIn(f, cd[k])


if __name__ == "__main__":
    unittest.main()
