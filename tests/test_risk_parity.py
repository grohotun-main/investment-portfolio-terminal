"""
Tests for parsers/risk_parity.py — equal-risk-contribution (ERC) optimizer.

Run from phase1_build/ with:
    py -m unittest tests.test_risk_parity
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import risk_parity as rp  # noqa: E402
import min_variance as mv  # noqa: E402


def _spd(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 0.01
    labels = [f"S{i}" for i in range(n)]
    return pd.DataFrame(cov, index=labels, columns=labels)


def _rc(weights: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    """Per-name risk contribution RCᵢ = wᵢ·(Σw)ᵢ."""
    w = weights.to_numpy(dtype=float)
    s = cov.to_numpy(dtype=float)
    return w * (s @ w)


class TestSolveRiskParity(unittest.TestCase):
    def test_two_asset_uncorrelated_is_inverse_vol(self) -> None:
        # Uncorrelated 2-asset ERC -> weights ∝ 1/σᵢ; σ=(0.2,0.3) -> (0.6,0.4).
        sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]],
                             index=["A", "B"], columns=["A", "B"])
        res = rp.solve_risk_parity(sigma)
        self.assertTrue(res["feasible"])
        self.assertAlmostEqual(float(res["weights"]["A"]), 0.6, places=4)
        self.assertAlmostEqual(float(res["weights"]["B"]), 0.4, places=4)
        self.assertAlmostEqual(float(res["weights"].sum()), 1.0, places=6)
        self.assertTrue(res["converged"])

    def test_single_asset_is_fully_invested(self) -> None:
        # n=1 edge: the lone holding takes 100%, converges immediately.
        sigma = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        res = rp.solve_risk_parity(sigma)
        self.assertTrue(res["feasible"])
        self.assertTrue(res["converged"])
        self.assertAlmostEqual(float(res["weights"]["A"]), 1.0, places=9)
        self.assertAlmostEqual(res["vol"], 0.2, places=6)

    def test_diagonal_n_asset_inverse_vol(self) -> None:
        sigma = pd.DataFrame(np.diag([0.01, 0.04, 0.09]),
                             index=list("ABC"), columns=list("ABC"))
        res = rp.solve_risk_parity(sigma)
        inv_vol = 1.0 / np.sqrt(np.array([0.01, 0.04, 0.09]))
        expect = inv_vol / inv_vol.sum()
        np.testing.assert_allclose(res["weights"].to_numpy(), expect, atol=1e-5)

    def test_equal_risk_contributions(self) -> None:
        for seed in (1, 2, 3):
            sigma = _spd(seed, 6)
            res = rp.solve_risk_parity(sigma)
            self.assertTrue(res["feasible"])
            rc = _rc(res["weights"], sigma)
            self.assertLess((rc.max() - rc.min()) / rc.mean(), 1e-8)
            self.assertAlmostEqual(float(res["weights"].sum()), 1.0, places=6)
            self.assertTrue((res["weights"].to_numpy() > 0).all())
            self.assertLess(res["rc_dispersion"], 1e-10)

    def test_deterministic(self) -> None:
        sigma = _spd(7, 8)
        r1 = rp.solve_risk_parity(sigma)
        r2 = rp.solve_risk_parity(sigma)
        np.testing.assert_allclose(
            r1["weights"].to_numpy(), r2["weights"].to_numpy(), atol=0.0)

    def test_correlation_reduces_weight(self) -> None:
        # Three equal-vol assets; A,B 0.9-correlated, C independent.
        # ERC down-weights the redundant pair vs the independent name.
        v = 0.04
        sigma = pd.DataFrame(
            [[v, 0.9 * v, 0.0], [0.9 * v, v, 0.0], [0.0, 0.0, v]],
            index=list("ABC"), columns=list("ABC"))
        res = rp.solve_risk_parity(sigma)
        w = res["weights"]
        self.assertLess(float(w["A"]), float(w["C"]))
        self.assertLess(float(w["B"]), float(w["C"]))

    def test_cross_check_vs_min_variance(self) -> None:
        # 1) ERC vol >= GMV vol always (GMV is the global variance floor).
        sigma = _spd(11, 7)
        class_of = {s: "equity" for s in sigma.index}
        gmv = mv.solve_min_variance(
            sigma, name_cap=1.0, class_of=class_of, class_floors={})
        erc = rp.solve_risk_parity(sigma)
        self.assertGreaterEqual(erc["vol"], gmv["vol"] - 1e-9)
        # 2) With one near-riskless asset, GMV concentrates into it while ERC
        #    stays diversified -> ERC has the higher effective N (1/Σwᵢ²).
        diag = pd.DataFrame(np.diag([0.0004, 0.04, 0.05, 0.06]),
                            index=list("ABCD"), columns=list("ABCD"))
        cof = {s: "equity" for s in diag.index}
        gmv2 = mv.solve_min_variance(
            diag, name_cap=1.0, class_of=cof, class_floors={})
        erc2 = rp.solve_risk_parity(diag)
        eff_n = lambda w: 1.0 / float((w.to_numpy() ** 2).sum())
        self.assertGreater(eff_n(erc2["weights"]), eff_n(gmv2["weights"]))

    def test_non_finite_cov_bails(self) -> None:
        sigma = pd.DataFrame([[0.04, np.nan], [np.nan, 0.09]],
                             index=["A", "B"], columns=["A", "B"])
        res = rp.solve_risk_parity(sigma)
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["weights"])

    def test_non_pd_cov_bails(self) -> None:
        # A singular Σ (riskless asset C) is not strictly PD -> bail.
        sigma = pd.DataFrame(np.diag([0.04, 0.09, 0.0]),
                             index=list("ABC"), columns=list("ABC"))
        res = rp.solve_risk_parity(sigma)
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["weights"])


def _prices(seed: int = 1, n_days: int = 300,
            syms: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-04-30", periods=n_days)
    data = {s: 100.0 * np.exp(np.cumsum(rng.standard_normal(n_days) * 0.01))
            for s in syms}
    return pd.DataFrame(data, index=idx)


class TestSuggestRiskParityGrid(unittest.TestCase):
    def test_success_sums_to_100(self) -> None:
        px = _prices(n_days=300, syms=("A", "B", "SGOV"))
        weights = pd.Series({"A": 0.5, "B": 0.4, "SGOV": 0.1})
        out = rp.suggest_risk_parity_grid(px, weights, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertIsNotNone(out["new_pct"])
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=6)
        self.assertIn("Risk-parity loaded", out["message"])
        self.assertIn("vol", out)

    def test_equalizes_contributions_on_mature_book(self) -> None:
        # No young names -> the suggested weights ARE the ERC weights, so their
        # risk contributions are near-equal on the same Σ the engine built.
        px = _prices(n_days=300, syms=("A", "B", "C"))
        weights = pd.Series({"A": 0.6, "B": 0.3, "C": 0.1})
        out = rp.suggest_risk_parity_grid(px, weights, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        cov = mv.build_covariance(px, ["A", "B", "C"],
                                  min_overlap_days=200)["cov"]
        w = pd.Series({s: out["new_pct"][s] / 100.0 for s in cov.index})
        rc = _rc(w, cov)
        self.assertLess((rc.max() - rc.min()) / rc.mean(), 1e-6)

    def test_insufficient_history_returns_error_kind(self) -> None:
        px = _prices(n_days=50, syms=("A", "B", "SGOV"))
        weights = pd.Series({"A": 0.5, "B": 0.4, "SGOV": 0.1})
        out = rp.suggest_risk_parity_grid(px, weights, min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIsNone(out["new_pct"])

    def test_holds_young_names_at_current_weight(self) -> None:
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))  # 40 trading days
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.45, "B": 0.35, "C": 0.20})
        out = rp.suggest_risk_parity_grid(merged, weights, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertAlmostEqual(out["new_pct"]["C"], 20.0, places=4)  # held
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=6)
        self.assertIn("too new to model", out["message"])

    def test_all_young_returns_error(self) -> None:
        # Modelable names exist but carry ~no weight; all the weight sits in a
        # holding too young to model -> nothing to optimize (the mature_budget
        # guard, distinct from build_covariance's no-mature-name error).
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.0, "B": 0.0, "C": 1.0})
        out = rp.suggest_risk_parity_grid(merged, weights, min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIsNone(out["new_pct"])
        self.assertIn("nothing to optimize", out["message"])

    def test_cap_bounds_grid_and_discloses(self) -> None:
        # A near-riskless name (tiny daily moves) -> uncapped ERC over-weights it;
        # name_cap bounds every weight and the banner names the capped holding.
        rng = np.random.default_rng(5)
        idx = pd.bdate_range(end="2026-04-30", periods=300)
        df = pd.DataFrame({
            "A": 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01)),
            "B": 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.012)),
            "CASH": 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.0004)),
        }, index=idx)
        weights = pd.Series({"A": 0.3, "B": 0.3, "CASH": 0.4})
        out = rp.suggest_risk_parity_grid(df, weights, name_cap=0.45,
                                          min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertTrue(all(v <= 45.0 + 1e-3 for v in out["new_pct"].values()),
                        out["new_pct"])
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=6)
        self.assertIn("Capped at the per-name limit", out["message"])
        self.assertIn("CASH", out["message"])

    def test_young_and_cap_compose(self) -> None:
        # A young held name + a capped mature near-riskless name in one book.
        rng = np.random.default_rng(6)
        idx = pd.bdate_range(end="2026-04-30", periods=300)
        mature = pd.DataFrame({
            "A": 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01)),
            "CASH": 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.0004)),
        }, index=idx)
        young = _prices(seed=9, n_days=40, syms=("NEW",))
        df = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.4, "CASH": 0.4, "NEW": 0.2})
        out = rp.suggest_risk_parity_grid(df, weights, name_cap=0.5,
                                          min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertAlmostEqual(out["new_pct"]["NEW"], 20.0, places=4)   # young held
        self.assertTrue(all(v <= 50.0 + 1e-3 for v in out["new_pct"].values()),
                        out["new_pct"])
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=6)
        # confirm the cap actually bound (non-vacuous): CASH is near-riskless so
        # uncapped ERC would over-weight it.
        self.assertIn("Capped at the per-name limit", out["message"])
        self.assertIn("CASH", out["message"])


class TestSuggestGridClassCaps(unittest.TestCase):
    def _book(self):
        px = _prices(n_days=300, syms=("A", "B", "SGOV"))
        weights = pd.Series({"A": 0.5, "B": 0.4, "SGOV": 0.1})
        class_of = {"A": "equity", "B": "equity", "SGOV": "fixed_income"}
        return px, weights, class_of

    def test_class_cap_respected_on_full_book(self) -> None:
        px, weights, class_of = self._book()
        out = rp.suggest_risk_parity_grid(
            px, weights, name_cap=1.0, class_of=class_of,
            class_caps={"fixed_income": 0.05}, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertLessEqual(out["new_pct"]["SGOV"], 5.0 + 1e-2)
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=2)
        self.assertIn("Class-cap held", out["message"])

    def test_message_identical_when_caps_absent_or_inert(self) -> None:
        px, weights, class_of = self._book()
        base = rp.suggest_risk_parity_grid(
            px, weights, name_cap=0.5, min_overlap_days=200)
        inert = rp.suggest_risk_parity_grid(
            px, weights, name_cap=0.5, class_of=class_of,
            class_caps={"fixed_income": 1.0}, min_overlap_days=200)
        self.assertEqual(base["message"], inert["message"])
        self.assertNotIn("Class-cap held", base["message"])

    def test_young_alone_exceeding_cap_errors(self) -> None:
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        class_of = {"A": "equity", "B": "equity", "C": "fixed_income"}
        out = rp.suggest_risk_parity_grid(
            merged, weights, name_cap=1.0, class_of=class_of,
            class_caps={"fixed_income": 0.1}, min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIn("too new to model already hold", out["message"])

    def test_young_in_capped_bucket_transform(self) -> None:
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.6, "B": 0.3, "C": 0.1})
        class_of = {"A": "equity", "B": "fixed_income", "C": "fixed_income"}
        out = rp.suggest_risk_parity_grid(
            merged, weights, name_cap=1.0, class_of=class_of,
            class_caps={"fixed_income": 0.25}, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertAlmostEqual(out["new_pct"]["C"], 10.0, places=4)  # held
        fi = out["new_pct"]["B"] + out["new_pct"]["C"]
        self.assertLessEqual(fi, 25.0 + 1e-2)


class TestSuggestRiskBudgets(unittest.TestCase):
    def _book(self):
        px = _prices(n_days=300, syms=("A", "B", "SGOV"))
        weights = pd.Series({"A": 0.5, "B": 0.4, "SGOV": 0.1})
        class_of = {"A": "equity", "B": "equity", "SGOV": "fixed_income"}
        return px, weights, class_of

    def test_budget_message_and_realized_shares(self) -> None:
        px, weights, class_of = self._book()
        out = rp.suggest_risk_parity_grid(
            px, weights, name_cap=1.0, class_of=class_of,
            class_risk_budgets={"fixed_income": 0.2},
            min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertIn("budget-shaped across", out["message"])
        self.assertIn("Risk budgets: fixed income 20.0% (realized ",
                      out["message"])
        self.assertNotIn("equalized across", out["message"])

    def test_message_identical_when_budgets_absent(self) -> None:
        px, weights, class_of = self._book()
        base = rp.suggest_risk_parity_grid(
            px, weights, name_cap=0.5, min_overlap_days=200)
        off = rp.suggest_risk_parity_grid(
            px, weights, name_cap=0.5, class_of=class_of,
            class_risk_budgets=None, min_overlap_days=200)
        self.assertEqual(base["message"], off["message"])
        self.assertNotIn("Risk budgets", base["message"])

    def test_limited_by_caps_clause(self) -> None:
        px, weights, class_of = self._book()
        out = rp.suggest_risk_parity_grid(
            px, weights, name_cap=1.0, class_of=class_of,
            class_caps={"fixed_income": 0.05},
            class_risk_budgets={"fixed_income": 0.5},
            min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertIn("Risk budgets limited by caps for: fixed income",
                      out["message"])

    def test_gap_metric_covers_free_names_only(self) -> None:
        # G0 (tiny vol, 20% budget) pins at the 25% name cap -> its own
        # target gap is huge; the headline gap must come from the FREE
        # names instead (well under 50%). name_cap 0.25 x 5 = 1.25 >= 1.
        # G0 built near-riskless (the _cash_prices pattern from
        # tests/test_opt_curve.py) since plain _prices (comparable-vol
        # random walks) never makes G0 pin the cap.
        idx = ["G0", "A", "B", "C", "D"]
        rng = np.random.default_rng(1)
        n_days = 300
        px = _prices(n_days=n_days, syms=("A", "B", "C", "D"))
        g0 = pd.Series(100.0 * np.exp(
            np.cumsum(rng.standard_normal(n_days) * 0.0004)),
            index=px.index, name="G0")
        px = px.join(g0)[idx]
        weights = pd.Series({s: 0.2 for s in idx})
        class_of = {"G0": "gold", "A": "equity", "B": "equity",
                    "C": "equity", "D": "equity"}
        out = rp.suggest_risk_parity_grid(
            px, weights, name_cap=0.25, class_of=class_of,
            class_risk_budgets={"gold": 0.2}, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertIn("Capped at the per-name limit: G0", out["message"])
        m = re.search(r"max gap to target (\d+\.\d+)%", out["message"])
        self.assertIsNotNone(m, out["message"])
        self.assertLess(float(m.group(1)), 50.0, out["message"])


class TestCappedRiskParity(unittest.TestCase):
    # Diagonal Σ with one near-riskless name (A): uncapped ERC over-weights A.
    SIGMA = pd.DataFrame(np.diag([0.0004, 0.04, 0.05, 0.06]),
                         index=list("ABCD"), columns=list("ABCD"))

    def test_cap_binds_and_reports(self) -> None:
        res = rp.solve_risk_parity(self.SIGMA, name_cap=0.4)
        w = res["weights"]
        self.assertTrue(res["feasible"])
        self.assertTrue((w.to_numpy() <= 0.4 + 1e-6).all())
        self.assertIn("A", res["binding"]["name_cap"])
        self.assertAlmostEqual(float(w["A"]), 0.4, places=4)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_cap_restores_de_concentration(self) -> None:
        uncapped = rp.solve_risk_parity(self.SIGMA)            # name_cap=1.0
        capped = rp.solve_risk_parity(self.SIGMA, name_cap=0.4)
        eff_n = lambda w: 1.0 / float((w.to_numpy() ** 2).sum())
        self.assertGreater(eff_n(capped["weights"]), eff_n(uncapped["weights"]))

    def test_free_names_stay_equalized(self) -> None:
        # A is pinned at the cap; B,C,D are free and keep equal RC among themselves.
        res = rp.solve_risk_parity(self.SIGMA, name_cap=0.4)
        self.assertIn("A", res["binding"]["name_cap"])
        self.assertLess(res["rc_dispersion"], 1e-6)

    def test_infeasible_cap_errors(self) -> None:
        sigma = pd.DataFrame(np.eye(3), index=list("ABC"), columns=list("ABC"))
        res = rp.solve_risk_parity(sigma, name_cap=0.2)       # 0.2*3 = 0.6 < 1
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["weights"])
        self.assertIsNotNone(res["error"])

    def test_no_cap_matches_unconstrained(self) -> None:
        sigma = _spd(11, 7)
        a = rp.solve_risk_parity(sigma)
        b = rp.solve_risk_parity(sigma, name_cap=1.0)
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=1e-12)
        self.assertEqual(b["binding"]["name_cap"], [])


class TestClassCapsPlumbing(unittest.TestCase):
    def test_no_caps_identical_to_omitted(self) -> None:
        sigma = _spd(11, 6)
        a = rp.solve_risk_parity(sigma, name_cap=0.4)
        b = rp.solve_risk_parity(sigma, name_cap=0.4, class_of=None,
                                 class_caps=None)
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=0.0)
        self.assertEqual(b["binding"]["class_caps"], {})

    def test_caps_too_tight_is_error(self) -> None:
        # 2 capped names (bucket cap 0.1) + 1 free name at name-cap 0.5:
        # reach = 0.1 + 0.5 = 0.6 < 1.
        sigma = _spd(12, 3)
        cof = {s: ("equity" if i < 2 else "other")
               for i, s in enumerate(sigma.index)}
        res = rp.solve_risk_parity(sigma, name_cap=0.5, class_of=cof,
                                   class_caps={"equity": 0.1})
        self.assertFalse(res["feasible"])
        self.assertIn("too little room", res["error"])
        self.assertEqual(res["binding"]["class_caps"], {})

    def test_per_name_cap_message_unchanged(self) -> None:
        sigma = _spd(13, 3)
        res = rp.solve_risk_parity(sigma, name_cap=0.2)
        self.assertFalse(res["feasible"])
        self.assertIn("Per-name cap 20% x 3 holdings", res["error"])

    def test_single_bucket_pins_at_cap(self) -> None:
        # B, C near-riskless fixed_income -> unconstrained ERC overweights
        # them; cap the bucket at 0.3.
        sigma = pd.DataFrame(
            [[0.09, 0.0, 0.0, 0.0], [0.0, 0.0004, 0.0001, 0.0],
             [0.0, 0.0001, 0.0005, 0.0], [0.0, 0.0, 0.0, 0.08]],
            index=list("ABCD"), columns=list("ABCD"))
        cof = {"A": "equity", "B": "fixed_income", "C": "fixed_income",
               "D": "equity"}
        res = rp.solve_risk_parity(sigma, name_cap=1.0, class_of=cof,
                                   class_caps={"fixed_income": 0.3})
        self.assertTrue(res["feasible"])
        w = res["weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        fi = float(w["B"] + w["C"])
        self.assertLessEqual(fi, 0.3 + 1e-6)
        self.assertGreaterEqual(fi, 0.3 - 1e-4)      # pinned AT the cap
        self.assertTrue(res["binding"]["class_caps"]["fixed_income"])
        # Outside names carry the rest and both hold real weight.
        self.assertGreater(float(w["A"]), 0.0)
        self.assertGreater(float(w["D"]), 0.0)

    def test_name_cap_enforced_inside_pinned_bucket(self) -> None:
        # Bucket cap 0.4 with name cap 0.25: B alone would absorb the whole
        # bucket budget (C is 100x riskier), but must stop at 0.25.
        # D, E are plain uncapped equity padding: with only A outside the
        # bucket, 3 names x 0.25 name-cap tops out at 75% and the global
        # feasibility floor (name_cap * n_holdings >= 1) bails before the
        # bucket logic ever runs; D/E give the free set enough room to
        # reach 100% so the in-bucket dynamic under test is reachable.
        sigma = pd.DataFrame(
            [[0.09, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0001, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.01, 0.0, 0.0], [0.0, 0.0, 0.0, 0.05, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.07]],
            index=list("ABCDE"), columns=list("ABCDE"))
        cof = {"A": "equity", "B": "fixed_income", "C": "fixed_income",
               "D": "equity", "E": "equity"}
        res = rp.solve_risk_parity(sigma, name_cap=0.25, class_of=cof,
                                   class_caps={"fixed_income": 0.4})
        self.assertTrue(res["feasible"])
        w = res["weights"]
        self.assertLessEqual(float(w["B"]), 0.25 + 1e-6)
        self.assertLessEqual(float(w["B"] + w["C"]), 0.4 + 1e-6)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertIn("B", res["binding"]["name_cap"])

    def test_zero_cap_zeroes_the_bucket(self) -> None:
        sigma = _spd(14, 4)
        cof = {s: ("gold" if i == 0 else "equity")
               for i, s in enumerate(sigma.index)}
        res = rp.solve_risk_parity(sigma, name_cap=1.0, class_of=cof,
                                   class_caps={"gold": 0.0})
        self.assertTrue(res["feasible"])
        gold = [s for s in sigma.index if cof[s] == "gold"]
        self.assertLessEqual(float(res["weights"][gold].sum()), 1e-9)
        self.assertTrue(res["binding"]["class_caps"]["gold"])

    def test_unreachable_cap_never_pins(self) -> None:
        # Bucket of 1 name at name-cap 0.2 can reach at most 0.2 < cap 0.5:
        # the cap can never bind; result equals the caps-free solve.
        # n=6 (not 4): at name_cap=0.2 the global feasibility floor needs
        # name_cap * n_holdings >= 1, i.e. n >= 5; 4 names bails before the
        # caps-free baseline solve even runs.
        sigma = _spd(15, 6)
        cof = {s: ("gold" if i == 0 else "equity")
               for i, s in enumerate(sigma.index)}
        a = rp.solve_risk_parity(sigma, name_cap=0.2)
        b = rp.solve_risk_parity(sigma, name_cap=0.2, class_of=cof,
                                 class_caps={"gold": 0.5})
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=1e-12)
        self.assertFalse(b["binding"]["class_caps"]["gold"])

    def test_cascade_second_bucket_pins_after_first(self) -> None:
        # A TRUE round-2 cascade: on the diagonal Σ, unconstrained ERC gives
        # w ∝ 1/vol → B ≈ .64 (over its .25 cap), C ≈ .32 (UNDER its .35
        # cap). Round 1 pins only B; the freed budget re-concentrates into C
        # (≈ .66), driving it over .35 in round 2. Both end pinned.
        sigma = pd.DataFrame(np.diag([0.09, 0.0001, 0.0004, 0.08]),
                             index=list("ABCD"), columns=list("ABCD"))
        cof = {"A": "equity", "B": "fixed_income", "C": "gold", "D": "equity"}
        res = rp.solve_risk_parity(sigma, name_cap=1.0, class_of=cof,
                                   class_caps={"fixed_income": 0.25,
                                               "gold": 0.35})
        self.assertTrue(res["feasible"])
        w = res["weights"]
        self.assertLessEqual(float(w["B"]), 0.25 + 1e-6)
        self.assertLessEqual(float(w["C"]), 0.35 + 1e-6)
        self.assertTrue(res["binding"]["class_caps"]["fixed_income"])
        self.assertTrue(res["binding"]["class_caps"]["gold"])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)


class TestFullSigmaRcDispersion(unittest.TestCase):
    """WSB-4: the Suggest banner must report the dispersion of risk
    contributions on the FULL covariance with the final weights, not the
    optimizer's reduced free-set sub-problem metric (which ignores capped
    names' cross-covariance and reads ~0 even when the realized spread is a
    couple percent)."""

    def test_equal_contributions_is_zero(self):
        sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]],
                             index=["A", "B"], columns=["A", "B"])
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        self.assertAlmostEqual(
            rp.full_sigma_rc_dispersion(w, sigma, ["A", "B"]), 0.0, places=10)

    def test_unequal_contributions_coefficient_of_variation(self):
        # RC_A = 0.5*(0.04*0.5) = 0.01; RC_B = 0.5*(0.16*0.5) = 0.04.
        # mean = 0.025, population std([0.01,0.04]) = 0.015 -> CV = 0.6.
        sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.16]],
                             index=["A", "B"], columns=["A", "B"])
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        self.assertAlmostEqual(
            rp.full_sigma_rc_dispersion(w, sigma, ["A", "B"]), 0.6, places=6)

    def test_only_free_names_count(self):
        # C is capped/pinned and excluded from the dispersion; A,B are equal.
        sigma = pd.DataFrame(np.diag([0.04, 0.04, 0.25]),
                             index=["A", "B", "C"], columns=["A", "B", "C"])
        w = pd.Series([0.4, 0.4, 0.2], index=["A", "B", "C"])
        self.assertAlmostEqual(
            rp.full_sigma_rc_dispersion(w, sigma, ["A", "B"]), 0.0, places=10)

    def test_fewer_than_two_free_names_returns_zero(self):
        sigma = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        w = pd.Series([1.0], index=["A"])
        self.assertEqual(rp.full_sigma_rc_dispersion(w, sigma, ["A"]), 0.0)


class TestBudgetedUnconstrained(unittest.TestCase):
    def test_budgets_none_identical_to_omitted(self) -> None:
        sigma = _spd(21, 6).to_numpy()
        a = rp._solve_erc_unconstrained(sigma)
        b = rp._solve_erc_unconstrained(sigma, budgets=None)
        np.testing.assert_allclose(a[0], b[0], atol=0.0, rtol=0.0)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])

    def test_diagonal_two_name_analytic(self) -> None:
        # diag(0.04, 0.01): equal weights carry risk 4:1, i.e. (0.8, 0.2) —
        # so budgets (0.8, 0.2) must return ~equal weights, shares on target.
        sigma = np.diag([0.04, 0.01])
        w, converged, disp = rp._solve_erc_unconstrained(
            sigma, budgets=np.array([0.8, 0.2]))
        self.assertTrue(converged)
        rc = w * (sigma @ w)
        shares = rc / rc.sum()
        np.testing.assert_allclose(shares, [0.8, 0.2], atol=1e-6)
        np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-4)

    def test_general_pd_sigma_hits_targets(self) -> None:
        sigma = _spd(22, 4).to_numpy()
        b = np.array([0.4, 0.3, 0.2, 0.1])
        w, converged, disp = rp._solve_erc_unconstrained(sigma, budgets=b)
        self.assertTrue(converged)
        rc = w * (sigma @ w)
        shares = rc / rc.sum()
        self.assertLess(float(np.max(np.abs(shares / b - 1.0))), 1e-6)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)

    def test_zero_or_misshaped_budgets_raise(self) -> None:
        sigma = np.diag([0.04, 0.01])
        with self.assertRaises(ValueError):
            rp._solve_erc_unconstrained(sigma, budgets=np.array([0.5, 0.0]))
        with self.assertRaises(ValueError):
            rp._solve_erc_unconstrained(sigma, budgets=np.array([1.0]))

    def test_pinned_none_identical_to_omitted(self) -> None:
        sigma = _spd(23, 5).to_numpy()
        a = rp._erc_name_pinned(sigma, 0.3)
        b = rp._erc_name_pinned(sigma, 0.3, budgets=None)
        np.testing.assert_allclose(a[0], b[0], atol=0.0, rtol=0.0)
        self.assertEqual(a[3], b[3])

    def test_budgeted_pin_renormalizes_free_budgets(self) -> None:
        # Name 0 near-riskless with a huge budget -> wants far past the cap,
        # pins; free names 1,2 keep their 0.3/0.1 ratio (renormed 0.75/0.25)
        # on the free sub-problem. Diagonal => shares computable exactly.
        # name_cap * n = 0.4 * 3 = 1.2 >= 1 (feasibility hand-checked).
        sigma = np.diag([0.0001, 0.04, 0.01])
        w, converged, disp, pinned = rp._erc_name_pinned(
            sigma, 0.4, budgets=np.array([0.6, 0.3, 0.1]))
        self.assertEqual(pinned, {0})
        self.assertAlmostEqual(float(w[0]), 0.4, places=9)
        sub = sigma[np.ix_([1, 2], [1, 2])]
        u = w[[1, 2]] / w[[1, 2]].sum()
        rc = u * (sub @ u)
        shares = rc / rc.sum()
        np.testing.assert_allclose(shares, [0.75, 0.25], atol=1e-6)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)


class TestClassRiskBudgets(unittest.TestCase):
    def _diag_book(self):
        # 4 names, 2 classes; diagonal => shares exactly computable.
        # name_cap defaults 1.0 -> cap*n = 4 >= 1.
        sigma = pd.DataFrame(np.diag([0.04, 0.02, 0.01, 0.03]),
                             index=list("ABCD"), columns=list("ABCD"))
        cof = {"A": "equity", "B": "equity", "C": "fixed_income",
               "D": "fixed_income"}
        return sigma, cof

    def test_none_identical_to_omitted(self) -> None:
        sigma, cof = self._diag_book()
        a = rp.solve_risk_parity(sigma, class_of=cof)
        b = rp.solve_risk_parity(sigma, class_of=cof,
                                 class_risk_budgets=None)
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=0.0, rtol=0.0)
        self.assertEqual(b["binding"]["budget_limited"], [])
        self.assertIsNone(b["budgets"])

    def test_partial_entry_splits_and_hits_targets(self) -> None:
        sigma, cof = self._diag_book()
        res = rp.solve_risk_parity(sigma, class_of=cof,
                                   class_risk_budgets={"fixed_income": 0.3})
        self.assertTrue(res["feasible"])
        b = res["budgets"]
        # FI names C,D: 0.15 each; unset equity A,B: 0.35 each.
        self.assertAlmostEqual(float(b["C"]), 0.15, places=12)
        self.assertAlmostEqual(float(b["D"]), 0.15, places=12)
        self.assertAlmostEqual(float(b["A"]), 0.35, places=12)
        w = res["weights"].to_numpy()
        s = sigma.to_numpy()
        rc = w * (s @ w)
        shares = rc / rc.sum()
        self.assertLess(float(np.max(np.abs(
            shares / b.to_numpy() - 1.0))), 1e-6)

    def test_validation_messages(self) -> None:
        sigma, cof = self._diag_book()
        cases = [
            ({"equity": -0.1}, "must be positive"),
            ({"equity": 0.9, "fixed_income": 0.3}, "sum to 120%"),
            ({"gold": 0.2}, "No modelable holdings in gold"),
            ({"equity": 0.4, "fixed_income": 0.3}, None),  # legal partial? NO:
        ]
        # legal case: equity 0.4 + fixed_income 0.3 covers EVERY class with
        # remainder 0.3 and no unset names -> validation 4 fires.
        for budgets, frag in cases[:3]:
            res = rp.solve_risk_parity(sigma, class_of=cof,
                                       class_risk_budgets=budgets)
            self.assertFalse(res["feasible"], budgets)
            self.assertIn(frag, res["error"], budgets)
        res = rp.solve_risk_parity(sigma, class_of=cof,
                                   class_risk_budgets={"equity": float("nan")})
        self.assertFalse(res["feasible"])
        self.assertIn("must be positive", res["error"])
        self.assertIsNone(res["budgets"])
        self.assertEqual(res["binding"]["budget_limited"], [])
        res = rp.solve_risk_parity(sigma, class_of=cof,
                                   class_risk_budgets=cases[3][0])
        self.assertFalse(res["feasible"])
        self.assertIn("cover every class", res["error"])

    def test_budgets_total_100_with_unset_class_errors(self) -> None:
        sigma, cof = self._diag_book()
        res = rp.solve_risk_parity(sigma, class_of=cof,
                                   class_risk_budgets={"equity": 1.0})
        self.assertFalse(res["feasible"])
        self.assertIn("carry none", res["error"])

    def test_budget_limited_when_class_cap_pins(self) -> None:
        # FI near-riskless wants big weight; a tight FI weight cap pins the
        # bucket; the 0.5 FI risk budget can't be honored -> budget_limited.
        sigma = pd.DataFrame(np.diag([0.09, 0.0001, 0.0004, 0.08]),
                             index=list("ABCD"), columns=list("ABCD"))
        cof = {"A": "equity", "B": "fixed_income", "C": "fixed_income",
               "D": "equity"}
        res = rp.solve_risk_parity(
            sigma, class_of=cof, class_caps={"fixed_income": 0.2},
            class_risk_budgets={"fixed_income": 0.5})
        self.assertTrue(res["feasible"])
        self.assertEqual(res["binding"]["budget_limited"], ["fixed_income"])
        fi = float(res["weights"][["B", "C"]].sum())
        self.assertLessEqual(fi, 0.2 + 1e-6)

    def test_none_identical_on_caps_path_too(self) -> None:
        sigma, cof = self._diag_book()
        kw = dict(class_of=cof, class_caps={"fixed_income": 0.3})
        a = rp.solve_risk_parity(sigma, **kw)
        b = rp.solve_risk_parity(sigma, **kw, class_risk_budgets=None)
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=0.0, rtol=0.0)


class TestRealizedPhrasing(unittest.TestCase):
    def test_fmt_realized_contract(self) -> None:
        self.assertEqual(rp._fmt_realized(20.04), "20.0%")
        self.assertEqual(rp._fmt_realized(-0.04), "0.0%")   # signed-zero clamp
        self.assertEqual(rp._fmt_realized(0.04), "0.0%")
        self.assertEqual(
            rp._fmt_realized(-3.2),
            "-3.2% — risk contribution currently negative")
        self.assertEqual(rp._fmt_realized(0.05), "0.1%")
        self.assertEqual(
            rp._fmt_realized(-0.05),
            "-0.1% — risk contribution currently negative")

    def test_gap_fragment_filters_and_omits(self) -> None:
        shares = pd.Series({"A": 0.50, "B": 0.0, "C": -0.001, "D": 0.51})
        b_ser = pd.Series({"A": 0.40, "B": 0.10, "C": 0.10, "D": 0.40})
        # B (zero) and C (negative) are excluded: gap = max(|.5/.4-1|,
        # |.51/.4-1|) = 27.50%, not the nonsense 101.0% inclusion would give.
        self.assertEqual(rp._gap_fragment(shares, b_ser, list(shares.index)),
                         " (max gap to target 27.50%)")
        # No qualifying free name -> the parenthetical is omitted entirely.
        self.assertEqual(
            rp._gap_fragment(shares, b_ser, ["B", "C"]), "")
        self.assertEqual(rp._gap_fragment(shares, None, ["A"]), "")

    def _hedged_book(self):
        # Cap-pinned anti-correlated bucket with a budget: the verified recipe
        # that drives a genuinely negative realized RC share (probe 2026-08-16:
        # class_cap=0.05 -> realized -5.2% pre-fix).
        rng = np.random.default_rng(7)
        n = 320
        idx = pd.bdate_range("2024-01-02", periods=n)
        ra = rng.normal(0.0005, 0.012, n)
        rb = rng.normal(0.0004, 0.010, n)
        rc = -ra + rng.normal(0.0, 0.001, n)
        px = pd.DataFrame({"A": 100.0 * np.cumprod(1.0 + ra),
                           "B": 100.0 * np.cumprod(1.0 + rb),
                           "C": 100.0 * np.cumprod(1.0 + rc)}, index=idx)
        weights = pd.Series({"A": 0.55, "B": 0.35, "C": 0.10})
        class_of = {"A": "equity", "B": "equity", "C": "hedge"}
        return px, weights, class_of

    def test_negative_realized_is_named_not_bare(self) -> None:
        px, weights, class_of = self._hedged_book()
        out = rp.suggest_risk_parity_grid(
            px, weights, name_cap=0.60, class_of=class_of,
            class_caps={"hedge": 0.05},
            class_risk_budgets={"hedge": 0.2},
            min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertIn("— risk contribution currently negative",
                      out["message"])
        self.assertRegex(out["message"],
                         r"\(realized -\d+\.\d% — risk contribution")
        self.assertNotIn("realized -0.0%", out["message"])
        # Both free names (A, B) have positive realized share, so the gap
        # fragment is present (exercises the qualifying-names branch).
        self.assertIn("(max gap to target ", out["message"])


if __name__ == "__main__":
    unittest.main()
