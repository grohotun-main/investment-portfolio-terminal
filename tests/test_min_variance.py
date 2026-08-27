"""
Tests for parsers/min_variance.py — constrained minimum-variance optimizer.

Run from phase1_build/ with:
    py -m unittest tests.test_min_variance
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import min_variance as mv  # noqa: E402


class TestCappedSimplex(unittest.TestCase):
    def test_projects_to_sum_one_within_box(self) -> None:
        w = mv._project_capped_simplex(np.array([0.9, 0.05, 0.05]), 0.5)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertTrue((w <= 0.5 + 1e-6).all())
        self.assertTrue((w >= -1e-9).all())
        self.assertAlmostEqual(float(w[0]), 0.5, places=4)  # hits the cap

    def test_already_on_simplex_is_unchanged(self) -> None:
        v = np.array([0.4, 0.35, 0.25])
        w = mv._project_capped_simplex(v, 1.0)
        np.testing.assert_allclose(w, v, atol=1e-6)

    def test_infeasible_cap_raises(self) -> None:
        with self.assertRaises(ValueError):
            mv._project_capped_simplex(np.zeros(3), 0.2)  # 0.2*3 < 1

    def test_is_exact_projection_variational_inequality(self) -> None:
        # w* is THE Euclidean projection of v onto {sum=1, 0<=w<=cap} iff
        # (v - w*) . (w - w*) <= 0 for every feasible w (the projection VI).
        rng = np.random.default_rng(7)
        def feasible(k, cap):
            for _ in range(2000):
                w = rng.dirichlet(np.ones(k))
                if (w <= cap + 1e-12).all():
                    return w
            return np.full(k, 1.0 / k)
        for _ in range(200):
            k = int(rng.integers(3, 12))
            v = rng.standard_normal(k)
            cap = float(rng.uniform(1.0 / k + 1e-3, 1.0))
            ws = mv._project_capped_simplex(v, cap)
            self.assertAlmostEqual(float(ws.sum()), 1.0, places=9)
            self.assertTrue((ws >= -1e-12).all() and (ws <= cap + 1e-12).all())
            for _ in range(4):
                wf = feasible(k, cap)
                vi = float((v - ws) @ (wf - ws))
                self.assertLessEqual(vi, 1e-9, f"VI violated: {vi}")


class TestProjectFeasible(unittest.TestCase):
    def test_no_groups_equals_capped_simplex(self) -> None:
        v = np.array([0.9, 0.05, 0.05])
        a = mv._project_feasible(v, 0.5, [])
        b = mv._project_capped_simplex(v, 0.5)
        np.testing.assert_allclose(a, b, atol=1e-8)

    def test_floor_half_space_enforced(self) -> None:
        # wants asset 2 (index 2); equity bucket {0,1} must hold >= 0.5
        v = np.array([0.05, 0.05, 0.90])
        w = mv._project_feasible(v, 1.0, [(np.array([0, 1]), 0.5)])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=4)
        self.assertGreaterEqual(float(w[[0, 1]].sum()), 0.5 - 1e-4)
        self.assertTrue((w >= -1e-6).all())

    def test_single_asset_floor_enforced(self) -> None:
        # One holding must be >= 60% of a 2-asset book; floor group has 1 member.
        w = mv._project_feasible(np.array([0.1, 0.9]), 0.7, [(np.array([0]), 0.6)])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertGreaterEqual(float(w[0]), 0.6 - 1e-4)
        self.assertTrue((w <= 0.7 + 1e-6).all())

    def test_cap_half_space_enforced(self) -> None:
        # wants bucket {0,1} at 0.95; class cap holds it to <= 0.4
        v = np.array([0.55, 0.40, 0.05])
        w = mv._project_feasible(v, 1.0, [], [(np.array([0, 1]), 0.4)])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=4)
        self.assertLessEqual(float(w[[0, 1]].sum()), 0.4 + 1e-4)
        self.assertTrue((w >= -1e-6).all())

    def test_floor_and_cap_same_bucket_both_hold(self) -> None:
        # bucket {0,1} squeezed into [0.3, 0.5]; v wants ~0.9 there
        v = np.array([0.45, 0.45, 0.10])
        w = mv._project_feasible(v, 1.0, [(np.array([0, 1]), 0.3)],
                                 [(np.array([0, 1]), 0.5)])
        s = float(w[[0, 1]].sum())
        self.assertAlmostEqual(float(w.sum()), 1.0, places=4)
        self.assertGreaterEqual(s, 0.3 - 1e-4)
        self.assertLessEqual(s, 0.5 + 1e-4)

    def test_no_cap_groups_identical_to_before(self) -> None:
        v = np.array([0.05, 0.05, 0.90])
        a = mv._project_feasible(v, 1.0, [(np.array([0, 1]), 0.5)])
        b = mv._project_feasible(v, 1.0, [(np.array([0, 1]), 0.5)], [])
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestFeasibility(unittest.TestCase):
    def test_cap_too_small(self) -> None:
        self.assertIsNotNone(mv._check_feasibility(3, 0.2, {}, {}))

    def test_floors_exceed_100(self) -> None:
        msg = mv._check_feasibility(
            3, 1.0, {"equity": 2, "fixed_income": 1},
            {"equity": 0.6, "fixed_income": 0.6})
        self.assertIsNotNone(msg)

    def test_floor_exceeds_bucket_capacity(self) -> None:
        # equity has 1 holding; at 40% cap it can hold at most 40% < 50% floor.
        msg = mv._check_feasibility(3, 0.4, {"equity": 1}, {"equity": 0.5})
        self.assertIsNotNone(msg)

    def test_feasible_returns_none(self) -> None:
        self.assertIsNone(
            mv._check_feasibility(3, 0.5, {"equity": 2}, {"equity": 0.4}))

    def test_floor_exceeds_class_cap(self) -> None:
        msg = mv._check_feasibility(4, 1.0, {"equity": 2}, {"equity": 0.5},
                                    {"equity": 0.4})
        self.assertIsNotNone(msg)
        self.assertIn("exceeds its", msg)

    def test_caps_too_tight_to_reach_100(self) -> None:
        # 2 capped equity names (cap 0.2) + 1 free name at name-cap 0.5:
        # reach = 0.2 + 0.5 = 0.7 < 1.
        msg = mv._check_feasibility(3, 0.5, {"equity": 2}, {},
                                    {"equity": 0.2})
        self.assertIsNotNone(msg)
        self.assertIn("too little room", msg)

    def test_caps_reachable_returns_none(self) -> None:
        self.assertIsNone(
            mv._check_feasibility(4, 0.5, {"equity": 2}, {"equity": 0.2},
                                  {"equity": 0.5}))

    def test_no_caps_message_unchanged(self) -> None:
        # The no-class-caps path keeps today's exact message text.
        msg = mv._check_feasibility(3, 0.2, {}, {})
        self.assertIn("Per-name cap 20% x 3 holdings", msg)


def _spd(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 0.01
    labels = [f"S{i}" for i in range(n)]
    return pd.DataFrame(cov, index=labels, columns=labels)


class TestSolveMinVariance(unittest.TestCase):
    def test_two_asset_analytic_gmv(self) -> None:
        sigma = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]],
                             index=["A", "B"], columns=["A", "B"])
        res = mv.solve_min_variance(
            sigma, name_cap=1.0,
            class_of={"A": "equity", "B": "equity"}, class_floors={})
        w_a = (0.09 - 0.006) / (0.04 + 0.09 - 2 * 0.006)
        self.assertTrue(res["feasible"])
        self.assertAlmostEqual(float(res["weights"]["A"]), w_a, places=3)
        self.assertAlmostEqual(float(res["weights"].sum()), 1.0, places=6)
        self.assertTrue(res["converged"])

    def test_long_only_clips_shorts(self) -> None:
        # Unconstrained GMV shorts B (A,B ~0.95 correlated, B higher vol).
        sigma = pd.DataFrame(
            [[0.04, 0.0475, 0.0], [0.0475, 0.0625, 0.0], [0.0, 0.0, 0.02]],
            index=list("ABC"), columns=list("ABC"))
        s = sigma.to_numpy()
        w_unc = np.linalg.solve(s, np.ones(3))
        w_unc /= w_unc.sum()
        self.assertLess(float(w_unc.min()), 0.0)  # scenario is non-trivial
        res = mv.solve_min_variance(
            sigma, name_cap=1.0,
            class_of={c: "equity" for c in "ABC"}, class_floors={})
        w = res["weights"]
        self.assertGreaterEqual(float(w.min()), -1e-9)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        ew = float(np.sqrt(np.ones(3) / 3 @ s @ (np.ones(3) / 3)))
        self.assertLessEqual(res["vol"], ew + 1e-9)

    def test_name_cap_binds(self) -> None:
        sigma = pd.DataFrame(np.diag([0.01, 0.10, 0.12]),
                             index=list("ABC"), columns=list("ABC"))
        res = mv.solve_min_variance(
            sigma, name_cap=0.5,
            class_of={c: "equity" for c in "ABC"}, class_floors={})
        w = res["weights"]
        self.assertTrue((w <= 0.5 + 1e-6).all())
        self.assertIn("A", res["binding"]["name_cap"])
        self.assertAlmostEqual(float(w["A"]), 0.5, places=3)

    def test_class_floor_binds_and_costs_vol(self) -> None:
        sigma = pd.DataFrame(np.diag([0.04, 0.05, 0.0001]),
                             index=list("ABC"), columns=list("ABC"))
        class_of = {"A": "equity", "B": "equity", "C": "fixed_income"}
        base = mv.solve_min_variance(
            sigma, name_cap=1.0, class_of=class_of, class_floors={})
        res = mv.solve_min_variance(
            sigma, name_cap=1.0, class_of=class_of,
            class_floors={"equity": 0.5})
        w = res["weights"]
        self.assertGreaterEqual(float(w[["A", "B"]].sum()), 0.5 - 1e-4)
        self.assertTrue(res["binding"]["floors"]["equity"])
        self.assertGreaterEqual(res["vol"], base["vol"] - 1e-9)

    def test_slack_floor_not_flagged_binding(self) -> None:
        # GMV piles into near-riskless C, so the fixed_income sum (~0.995)
        # sits far above its 0.2 floor: satisfied with slack, not binding.
        sigma = pd.DataFrame(np.diag([0.04, 0.05, 0.0001]),
                             index=list("ABC"), columns=list("ABC"))
        res = mv.solve_min_variance(
            sigma, name_cap=1.0,
            class_of={"A": "equity", "B": "equity", "C": "fixed_income"},
            class_floors={"fixed_income": 0.2})
        self.assertTrue(res["feasible"])
        self.assertGreater(float(res["weights"]["C"]), 0.9)
        self.assertFalse(res["binding"]["floors"]["fixed_income"])

    def test_infeasible_cap_returns_reason(self) -> None:
        sigma = pd.DataFrame(np.eye(3), index=list("ABC"), columns=list("ABC"))
        res = mv.solve_min_variance(
            sigma, name_cap=0.2,
            class_of={c: "equity" for c in "ABC"}, class_floors={})
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["weights"])
        self.assertIsNotNone(res["error"])

    def test_infeasible_floor_capacity(self) -> None:
        sigma = pd.DataFrame(np.eye(3), index=list("ABC"), columns=list("ABC"))
        res = mv.solve_min_variance(
            sigma, name_cap=0.4,
            class_of={"A": "equity", "B": "fixed_income", "C": "fixed_income"},
            class_floors={"equity": 0.5})
        self.assertFalse(res["feasible"])

    def test_non_finite_cov_bails(self) -> None:
        sigma = pd.DataFrame([[0.04, np.nan], [np.nan, 0.09]],
                             index=["A", "B"], columns=["A", "B"])
        res = mv.solve_min_variance(
            sigma, name_cap=1.0,
            class_of={"A": "equity", "B": "equity"}, class_floors={})
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["weights"])

    def test_non_psd_cov_bails(self) -> None:
        # An indefinite / negative-definite matrix is not a valid covariance.
        sigma = pd.DataFrame([[-1.0, 0.0], [0.0, -1.0]],
                             index=["A", "B"], columns=["A", "B"])
        res = mv.solve_min_variance(
            sigma, name_cap=1.0,
            class_of={"A": "equity", "B": "equity"}, class_floors={})
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["weights"])

    def test_invariants_and_determinism(self) -> None:
        sigma = _spd(42, 8)
        class_of = {s: "equity" for s in sigma.index}
        r1 = mv.solve_min_variance(
            sigma, name_cap=0.3, class_of=class_of, class_floors={})
        r2 = mv.solve_min_variance(
            sigma, name_cap=0.3, class_of=class_of, class_floors={})
        w = r1["weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertTrue((w >= -1e-9).all())
        self.assertTrue((w <= 0.3 + 1e-6).all())
        np.testing.assert_allclose(
            r1["weights"].to_numpy(), r2["weights"].to_numpy(), atol=0.0)

    def test_class_cap_binds_and_reports(self) -> None:
        # C is near-riskless -> unconstrained GMV piles into fixed_income {C}.
        sigma = pd.DataFrame(
            [[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.0001]],
            index=list("ABC"), columns=list("ABC"))
        class_of = {"A": "equity", "B": "equity", "C": "fixed_income"}
        res = mv.solve_min_variance(
            sigma, name_cap=1.0, class_of=class_of, class_floors={},
            class_caps={"fixed_income": 0.3})
        self.assertTrue(res["feasible"])
        w = res["weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertLessEqual(float(w["C"]), 0.3 + 1e-4)
        self.assertTrue(res["binding"]["class_caps"]["fixed_income"])

    def test_class_caps_none_identical_to_omitted(self) -> None:
        sigma = _spd(3, 6)
        cof = {s: ("equity" if i % 2 else "fixed_income")
               for i, s in enumerate(sigma.index)}
        a = mv.solve_min_variance(sigma, name_cap=0.4, class_of=cof,
                                  class_floors={"equity": 0.2})
        b = mv.solve_min_variance(sigma, name_cap=0.4, class_of=cof,
                                  class_floors={"equity": 0.2},
                                  class_caps=None)
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=0.0)
        self.assertEqual(b["binding"]["class_caps"], {})

    def test_cap_of_one_is_inert(self) -> None:
        sigma = _spd(4, 5)
        cof = {s: "equity" for s in sigma.index}
        a = mv.solve_min_variance(sigma, name_cap=0.5, class_of=cof,
                                  class_floors={})
        b = mv.solve_min_variance(sigma, name_cap=0.5, class_of=cof,
                                  class_floors={}, class_caps={"equity": 1.0})
        np.testing.assert_allclose(a["weights"].to_numpy(),
                                   b["weights"].to_numpy(), atol=0.0)
        self.assertEqual(b["binding"]["class_caps"], {})

    def test_zero_cap_zeroes_the_bucket(self) -> None:
        sigma = _spd(5, 4)
        cof = {s: ("gold" if i == 0 else "equity")
               for i, s in enumerate(sigma.index)}
        res = mv.solve_min_variance(sigma, name_cap=1.0, class_of=cof,
                                    class_floors={}, class_caps={"gold": 0.0})
        self.assertTrue(res["feasible"])
        gold = [s for s in sigma.index if cof[s] == "gold"]
        self.assertLessEqual(float(res["weights"][gold].sum()), 1e-4)

    def test_floor_over_cap_is_infeasible_with_message(self) -> None:
        sigma = _spd(6, 4)
        cof = {s: "equity" for s in sigma.index}
        res = mv.solve_min_variance(
            sigma, name_cap=1.0, class_of=cof,
            class_floors={"equity": 0.5}, class_caps={"equity": 0.4})
        self.assertFalse(res["feasible"])
        self.assertIn("exceeds its", res["error"])
        self.assertEqual(res["binding"]["class_caps"], {})

    def test_two_active_caps_both_enforced_and_reported(self) -> None:
        # Two low-vol buckets that unconstrained GMV loads up; both caps bind.
        sigma = pd.DataFrame(
            [[0.09, 0.0, 0.0, 0.0], [0.0, 0.0001, 0.0, 0.0],
             [0.0, 0.0, 0.0002, 0.0], [0.0, 0.0, 0.0, 0.08]],
            index=list("ABCD"), columns=list("ABCD"))
        class_of = {"A": "equity", "B": "fixed_income", "C": "gold",
                    "D": "equity"}
        res = mv.solve_min_variance(
            sigma, name_cap=1.0, class_of=class_of, class_floors={},
            class_caps={"fixed_income": 0.3, "gold": 0.2})
        self.assertTrue(res["feasible"])
        w = res["weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertLessEqual(float(w["B"]), 0.3 + 1e-4)
        self.assertLessEqual(float(w["C"]), 0.2 + 1e-4)
        self.assertTrue(res["binding"]["class_caps"]["fixed_income"])
        self.assertTrue(res["binding"]["class_caps"]["gold"])


class TestSolverPerformance(unittest.TestCase):
    @staticmethod
    def _ill_conditioned(n: int = 10, kappa: float = 1e4, seed: int = 3):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eig = np.geomspace(1.0, 1.0 / kappa, n)          # wide eigen-spread -> high kappa
        cov = (q * eig) @ q.T
        cov = 0.5 * (cov + cov.T)                         # re-symmetrize
        labels = [f"S{i}" for i in range(n)]
        return pd.DataFrame(cov, index=labels, columns=labels)

    def test_converges_in_bounded_iterations_on_ill_conditioning(self) -> None:
        cov = self._ill_conditioned()
        class_of = {s: ("equity" if i % 2 else "fixed_income")
                    for i, s in enumerate(cov.index)}
        calls = {"n": 0}
        orig = mv._project_feasible
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        mv._project_feasible = counting
        try:
            res = mv.solve_min_variance(
                cov, name_cap=0.30, class_of=class_of,
                class_floors={"equity": 0.4})
        finally:
            mv._project_feasible = orig
        self.assertTrue(res["feasible"])
        self.assertTrue(res["converged"], "solver did not converge")
        # Plain PGD needs many hundreds of projections on this conditioning;
        # accelerated FISTA must finish in far fewer. Guards the speedup.
        self.assertLess(calls["n"], 400,
                        f"solver used {calls['n']} projections (perf regression?)")


class TestBucketAndDefaults(unittest.TestCase):
    def test_to_floor_bucket(self) -> None:
        self.assertEqual(mv.to_floor_bucket("equity_stock"), "equity")
        self.assertEqual(mv.to_floor_bucket("equity_etf"), "equity")
        self.assertEqual(mv.to_floor_bucket("tax_loss_harvesting"), "equity")
        self.assertEqual(mv.to_floor_bucket("fixed_income"), "fixed_income")
        self.assertEqual(mv.to_floor_bucket("cash"), "fixed_income")
        self.assertEqual(mv.to_floor_bucket("gold"), "gold")
        self.assertEqual(mv.to_floor_bucket(""), "other")
        self.assertEqual(mv.to_floor_bucket("mystery"), "other")

    def test_anchored_defaults_cap_rounds_up_floor_rounds_down(self) -> None:
        weights = pd.Series({"A": 0.513, "B": 0.300, "C": 0.187})
        class_of = {"A": "equity", "B": "equity", "C": "fixed_income"}
        out = mv.anchored_defaults(weights, class_of)
        # current largest 51.3% -> cap rounds UP to 52%
        self.assertAlmostEqual(out["cap_default"], 0.52, places=6)
        # current equity 81.3% -> floor rounds DOWN to 81%
        self.assertAlmostEqual(out["equity_floor_default"], 0.81, places=6)

    def test_anchored_defaults_empty(self) -> None:
        out = mv.anchored_defaults(pd.Series(dtype=float), {})
        self.assertEqual(out["cap_default"], 1.0)
        self.assertEqual(out["equity_floor_default"], 0.0)

    def test_anchored_defaults_requires_bucketed_class_values(self) -> None:
        # Contract: class_of values must be floor buckets, not raw asset_class.
        weights = pd.Series({"A": 0.6, "B": 0.4})
        bucketed = mv.anchored_defaults(
            weights, {"A": "equity", "B": "fixed_income"})
        self.assertAlmostEqual(bucketed["equity_floor_default"], 0.60, places=6)
        # Raw (un-bucketed) classes do NOT match -> 0 equity floor.
        raw = mv.anchored_defaults(
            weights, {"A": "equity_stock", "B": "fixed_income"})
        self.assertAlmostEqual(raw["equity_floor_default"], 0.0, places=6)


def _prices(seed: int = 1, n_days: int = 300,
            syms: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-04-30", periods=n_days)
    data = {s: 100.0 * np.exp(np.cumsum(rng.standard_normal(n_days) * 0.01))
            for s in syms}
    return pd.DataFrame(data, index=idx)


class TestBuildCovariance(unittest.TestCase):
    def test_builds_symmetric_finite_cov(self) -> None:
        res = mv.build_covariance(_prices(), ["A", "B", "C"],
                                  min_overlap_days=200)
        self.assertIsNone(res["error"])
        cov = res["cov"]
        self.assertEqual(list(cov.index), ["A", "B", "C"])
        self.assertTrue(np.all(np.isfinite(cov.to_numpy())))
        np.testing.assert_allclose(
            cov.to_numpy(), cov.to_numpy().T, atol=1e-12)

    def test_insufficient_overlap_errors(self) -> None:
        res = mv.build_covariance(_prices(n_days=50), ["A", "B", "C"],
                                  min_overlap_days=200)
        self.assertIsNotNone(res["error"])
        self.assertIsNone(res["cov"])

    def test_filters_symbols_without_history(self) -> None:
        res = mv.build_covariance(_prices(), ["A", "ZZZ"],
                                  min_overlap_days=100)
        self.assertEqual(list(res["cov"].index), ["A"])

    def test_excludes_young_names(self) -> None:
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))  # only 40 trading days
        merged = mature.join(young, how="outer")
        res = mv.build_covariance(merged, ["A", "B", "C"], min_overlap_days=200)
        self.assertIsNone(res["error"])
        self.assertEqual(res["excluded"], ["C"])
        self.assertEqual(list(res["cov"].index), ["A", "B"])


class TestSuggestGrid(unittest.TestCase):
    def _book(self):
        px = _prices(n_days=300, syms=("A", "B", "SGOV"))
        weights = pd.Series({"A": 0.5, "B": 0.4, "SGOV": 0.1})
        class_of = {"A": "equity", "B": "equity", "SGOV": "fixed_income"}
        return px, weights, class_of

    def test_success_payload_sums_to_100_and_respects_floor(self) -> None:
        px, weights, class_of = self._book()
        out = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.6,
            class_floors={"equity": 0.3}, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertIsNotNone(out["new_pct"])
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=2)
        self.assertGreaterEqual(
            out["new_pct"]["A"] + out["new_pct"]["B"], 30.0 - 1e-2)
        self.assertIn("vol", out)
        self.assertIn("Min-variance loaded", out["message"])

    def test_infeasible_returns_error_kind(self) -> None:
        px, weights, class_of = self._book()
        out = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.2,
            class_floors={}, min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIsNone(out["new_pct"])

    def test_insufficient_history_returns_error_kind(self) -> None:
        px = _prices(n_days=50, syms=("A", "B", "SGOV"))
        weights = pd.Series({"A": 0.5, "B": 0.4, "SGOV": 0.1})
        class_of = {"A": "equity", "B": "equity", "SGOV": "fixed_income"}
        out = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.6,
            class_floors={}, min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIsNone(out["new_pct"])

    def test_holds_young_names_at_current_weight(self) -> None:
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.45, "B": 0.35, "C": 0.20})
        class_of = {"A": "equity", "B": "equity", "C": "equity"}
        out = mv.suggest_min_variance_grid(
            merged, weights, class_of, name_cap=0.8,
            class_floors={}, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertAlmostEqual(out["new_pct"]["C"], 20.0, places=4)  # held
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=2)
        self.assertIn("too new to model", out["message"])

    def test_full_book_equity_floor_holds_with_young_equity_held(self) -> None:
        # A (equity, mature), B (fixed_income, mature), C (equity, YOUNG).
        # Equity floor must hold on the FULL book: A + C (equities) >= 55%.
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.40, "B": 0.40, "C": 0.20})
        class_of = {"A": "equity", "B": "fixed_income", "C": "equity"}
        out = mv.suggest_min_variance_grid(
            merged, weights, class_of, name_cap=1.0,
            class_floors={"equity": 0.55}, min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=4)
        equity_pct = out["new_pct"]["A"] + out["new_pct"]["C"]
        self.assertGreaterEqual(equity_pct, 55.0 - 0.1)
        self.assertAlmostEqual(out["new_pct"]["C"], 20.0, places=4)  # young held

    def test_floor_unsatisfiable_by_young_alone_errors(self) -> None:
        # FI floor 30% but the only fixed_income holding is a YOUNG 10% name
        # and no mature FI name exists to fill it -> honest infeasibility.
        mature = _prices(n_days=300, syms=("A", "B"))   # both equity, mature
        young = _prices(seed=9, n_days=40, syms=("C",))  # fixed_income, young
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.5, "B": 0.4, "C": 0.1})
        class_of = {"A": "equity", "B": "equity", "C": "fixed_income"}
        out = mv.suggest_min_variance_grid(
            merged, weights, class_of, name_cap=1.0,
            class_floors={"fixed_income": 0.30}, min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIsNone(out["new_pct"])

    def test_class_cap_respected_on_full_book(self) -> None:
        px, weights, class_of = self._book()
        out = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=1.0,
            class_floors={}, class_caps={"fixed_income": 0.05},
            min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertLessEqual(out["new_pct"]["SGOV"], 5.0 + 1e-2)
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=2)

    def test_message_gains_class_cap_clause_only_when_active(self) -> None:
        px, weights, class_of = self._book()
        base = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.6,
            class_floors={"equity": 0.3}, min_overlap_days=200)
        inert = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.6,
            class_floors={"equity": 0.3}, class_caps={"fixed_income": 1.0},
            min_overlap_days=200)
        self.assertEqual(base["message"], inert["message"])   # byte-identical
        self.assertNotIn("class caps", base["message"])
        active = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.6,
            class_floors={"equity": 0.3}, class_caps={"fixed_income": 0.05},
            min_overlap_days=200)
        self.assertIn("class caps", active["message"])

    def test_young_in_capped_bucket_transform(self) -> None:
        # C (fixed_income) is YOUNG at 10%; full-book FI cap 25% leaves the
        # mature FI name B at most 15% of the book.
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.6, "B": 0.3, "C": 0.1})
        class_of = {"A": "equity", "B": "fixed_income", "C": "fixed_income"}
        out = mv.suggest_min_variance_grid(
            merged, weights, class_of, name_cap=1.0,
            class_floors={}, class_caps={"fixed_income": 0.25},
            min_overlap_days=200)
        self.assertEqual(out["kind"], "success")
        self.assertAlmostEqual(out["new_pct"]["C"], 10.0, places=4)  # held
        fi = out["new_pct"]["B"] + out["new_pct"]["C"]
        self.assertLessEqual(fi, 25.0 + 1e-2)

    def test_young_alone_exceeding_cap_errors(self) -> None:
        mature = _prices(n_days=300, syms=("A", "B"))
        young = _prices(seed=9, n_days=40, syms=("C",))
        merged = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        class_of = {"A": "equity", "B": "equity", "C": "fixed_income"}
        out = mv.suggest_min_variance_grid(
            merged, weights, class_of, name_cap=1.0,
            class_floors={}, class_caps={"fixed_income": 0.1},
            min_overlap_days=200)
        self.assertEqual(out["kind"], "error")
        self.assertIn("too new to model already hold", out["message"])


class TestScalarizedObjective(unittest.TestCase):
    """lambda*w'Sigma*w - w'mu. The mu=None path must not move a single bit."""

    def _cov(self, seed=0, n=6):
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, (n, n))
        s = (a @ a.T) / n * 0.04 + np.eye(n) * 0.01
        syms = [f"S{i}" for i in range(n)]
        return pd.DataFrame(s, index=syms, columns=syms)

    def _legacy_solve(self, cov, cap, class_of, floors, caps,
                      max_iter=10000, tol=1e-6):
        """The pre-O3 FISTA loop, verbatim, using the (unchanged) projection.

        This is the bit-identity tripwire: if anyone rewrites the default
        gradient expression, these weights stop matching exactly.
        """
        syms = list(cov.index)
        n = len(syms)
        sigma = np.asarray(cov.to_numpy(), dtype=float)
        floors_d = {b: float(f) for b, f in (floors or {}).items()
                    if f and float(f) > 0.0}
        caps_d = {b: float(c) for b, c in (caps or {}).items()
                  if c is not None and float(c) < 1.0}
        all_buckets = set(floors_d) | set(caps_d)
        group_idx = {b: np.array([i for i, s in enumerate(syms)
                                  if class_of.get(s) == b], dtype=int)
                     for b in all_buckets}
        groups = [(group_idx[b], floors_d[b]) for b in floors_d]
        cap_groups = [(group_idx[b], caps_d[b]) for b in caps_d]
        lam_max = float(np.linalg.eigvalsh(sigma)[-1])
        eta = 1.0 / (2.0 * lam_max) if lam_max > 0 else 1.0
        x = mv._project_feasible(np.full(n, 1.0 / n), cap, groups, cap_groups)
        y = x.copy()
        t = 1.0
        for _ in range(max_iter):
            x_new = mv._project_feasible(y - eta * (2.0 * (sigma @ y)), cap,
                                         groups, cap_groups)
            if np.max(np.abs(x_new - x)) < tol:
                return x_new
            if float((y - x_new) @ (x_new - x)) > 0.0:
                t = 1.0
                y = x_new
            else:
                t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                y = x_new + ((t - 1.0) / t_new) * (x_new - x)
                t = t_new
            x = x_new
        return x

    def test_mu_none_is_bit_identical_to_legacy_solver(self):
        cov = self._cov()
        syms = list(cov.index)
        class_of = {s: ("equity" if i < 3 else "fixed_income")
                    for i, s in enumerate(syms)}
        configs = [
            (1.0, {}, {}),
            (0.30, {}, {}),
            (0.40, {"equity": 0.50}, {}),
            (0.40, {}, {"equity": 0.60}),
            (0.35, {"equity": 0.40}, {"fixed_income": 0.50}),
            (0.50, {"fixed_income": 0.20}, {"equity": 0.70}),
        ]
        for cap, floors, caps in configs:
            with self.subTest(cap=cap, floors=floors, caps=caps):
                res = mv.solve_min_variance(cov, name_cap=cap,
                                            class_of=class_of,
                                            class_floors=floors,
                                            class_caps=caps)
                self.assertTrue(res["feasible"])
                legacy = self._legacy_solve(cov, cap, class_of, floors, caps)
                self.assertTrue(
                    np.array_equal(res["weights"].to_numpy(), legacy),
                    f"mu=None path moved at cap={cap} floors={floors} "
                    f"caps={caps}")

    def test_risk_aversion_is_inert_without_mu(self):
        cov = self._cov()
        class_of = {s: "equity" for s in cov.index}
        a = mv.solve_min_variance(cov, name_cap=0.4, class_of=class_of,
                                  class_floors={})
        b = mv.solve_min_variance(cov, name_cap=0.4, class_of=class_of,
                                  class_floors={}, risk_aversion=97.0)
        self.assertTrue(np.array_equal(a["weights"].to_numpy(),
                                       b["weights"].to_numpy()))

    def test_high_risk_aversion_converges_to_min_variance(self):
        cov = self._cov()
        class_of = {s: "equity" for s in cov.index}
        mu = pd.Series(np.linspace(0.03, 0.12, len(cov.index)),
                       index=cov.index)
        mvw = mv.solve_min_variance(cov, name_cap=0.5, class_of=class_of,
                                    class_floors={})["weights"]
        hi = mv.solve_min_variance(cov, name_cap=0.5, class_of=class_of,
                                   class_floors={}, mu=mu,
                                   risk_aversion=1e5)["weights"]
        self.assertLess(float(np.max(np.abs(hi - mvw))), 1e-4)

    def test_low_risk_aversion_tilts_to_the_highest_mu_name(self):
        cov = self._cov()
        class_of = {s: "equity" for s in cov.index}
        mu = pd.Series(np.linspace(0.03, 0.12, len(cov.index)),
                       index=cov.index)
        best = mu.idxmax()
        lo = mv.solve_min_variance(cov, name_cap=0.5, class_of=class_of,
                                   class_floors={}, mu=mu,
                                   risk_aversion=0.05)["weights"]
        hi = mv.solve_min_variance(cov, name_cap=0.5, class_of=class_of,
                                   class_floors={}, mu=mu,
                                   risk_aversion=1e5)["weights"]
        self.assertGreater(float(lo[best]), float(hi[best]))
        self.assertAlmostEqual(float(lo[best]), 0.5, places=4)

    def test_constraints_still_hold_with_mu(self):
        cov = self._cov()
        syms = list(cov.index)
        class_of = {s: ("equity" if i < 3 else "fixed_income")
                    for i, s in enumerate(syms)}
        mu = pd.Series(np.linspace(0.03, 0.12, len(syms)), index=syms)
        res = mv.solve_min_variance(cov, name_cap=0.30, class_of=class_of,
                                    class_floors={"fixed_income": 0.25},
                                    class_caps={"equity": 0.60}, mu=mu,
                                    risk_aversion=0.2)
        w = res["weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertLessEqual(float(w.max()), 0.30 + 1e-6)
        fi = float(sum(w[s] for s in syms if class_of[s] == "fixed_income"))
        eq = float(sum(w[s] for s in syms if class_of[s] == "equity"))
        self.assertGreaterEqual(fi, 0.25 - 1e-6)
        self.assertLessEqual(eq, 0.60 + 1e-6)

    def test_nonfinite_mu_is_error_not_raise(self):
        cov = self._cov()
        class_of = {s: "equity" for s in cov.index}
        mu = pd.Series(np.linspace(0.03, 0.12, len(cov.index)),
                       index=cov.index)
        mu.iloc[0] = np.nan
        res = mv.solve_min_variance(cov, name_cap=0.5, class_of=class_of,
                                    class_floors={}, mu=mu, risk_aversion=1.0)
        self.assertFalse(res["feasible"])
        self.assertIsNotNone(res["error"])
        self.assertIsNone(res["weights"])

    def test_bad_risk_aversion_is_error_not_raise(self):
        cov = self._cov()
        class_of = {s: "equity" for s in cov.index}
        mu = pd.Series(0.05, index=cov.index)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                res = mv.solve_min_variance(cov, name_cap=0.5,
                                            class_of=class_of,
                                            class_floors={}, mu=mu,
                                            risk_aversion=bad)
                self.assertFalse(res["feasible"])
                self.assertIsNotNone(res["error"])


class TestGridScalarized(unittest.TestCase):
    """mu threading through the young-sleeve transform (effective lambda)."""

    def _book(self, n_days=400, seed=3):
        """Prices where SPY/AAA/BBB are mature and YOUNG has 60 days."""
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2022-01-03", periods=n_days)
        cols = {}
        for sym, drift in (("SPY", 0.0004), ("AAA", 0.0005), ("BBB", 0.0003)):
            cols[sym] = 100.0 * np.cumprod(
                1.0 + rng.normal(drift, 0.01, n_days))
        px = pd.DataFrame(cols, index=idx)
        young = np.full(n_days, np.nan)
        young[-60:] = 50.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, 60))
        px["YOUNG"] = young
        weights = pd.Series({"SPY": 0.4, "AAA": 0.25, "BBB": 0.25,
                             "YOUNG": 0.10})
        class_of = {"SPY": "equity", "AAA": "equity", "BBB": "fixed_income",
                    "YOUNG": "equity"}
        mu = pd.Series({"SPY": 0.085, "AAA": 0.11, "BBB": 0.05,
                        "YOUNG": 0.09})
        return px, weights, class_of, mu

    def test_grid_solves_the_sleeve_at_lambda_times_mature_budget(self):
        px, weights, class_of, mu = self._book()
        lam = 3.0
        out = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.60, class_floors={},
            mu=mu, risk_aversion=lam)
        self.assertEqual(out["kind"], "success")

        covres = mv.build_covariance(px, list(weights.index))
        cov = covres["cov"]
        mature = list(cov.index)
        self.assertNotIn("YOUNG", mature)          # the fixture must be young
        mb = 1.0 - float(weights["YOUNG"])
        direct = mv.solve_min_variance(
            cov, name_cap=min(1.0, 0.60 / mb),
            class_of={s: class_of[s] for s in mature}, class_floors={},
            mu=mu.reindex(mature), risk_aversion=lam * mb)
        self.assertTrue(direct["feasible"])
        for s in mature:
            self.assertAlmostEqual(out["new_pct"][s] / 100.0,
                                   float(direct["weights"][s]) * mb, places=9)

    def test_young_holding_is_held_at_current_weight(self):
        px, weights, class_of, mu = self._book()
        out = mv.suggest_min_variance_grid(
            px, weights, class_of, name_cap=0.60, class_floors={},
            mu=mu, risk_aversion=3.0)
        self.assertAlmostEqual(out["new_pct"]["YOUNG"], 10.0, places=9)
        self.assertAlmostEqual(sum(out["new_pct"].values()), 100.0, places=6)

    def test_grid_without_mu_ignores_risk_aversion(self):
        px, weights, class_of, _mu = self._book()
        a = mv.suggest_min_variance_grid(px, weights, class_of, name_cap=0.60,
                                         class_floors={})
        b = mv.suggest_min_variance_grid(px, weights, class_of, name_cap=0.60,
                                         class_floors={}, risk_aversion=41.0)
        self.assertEqual(a["new_pct"], b["new_pct"])
        self.assertEqual(a["message"], b["message"])

    def test_lower_lambda_raises_the_high_mu_name(self):
        px, weights, class_of, mu = self._book()
        hi = mv.suggest_min_variance_grid(px, weights, class_of, name_cap=0.60,
                                          class_floors={}, mu=mu,
                                          risk_aversion=500.0)["new_pct"]
        lo = mv.suggest_min_variance_grid(px, weights, class_of, name_cap=0.60,
                                          class_floors={}, mu=mu,
                                          risk_aversion=0.2)["new_pct"]
        self.assertGreater(lo["AAA"], hi["AAA"])     # AAA has the highest mu
