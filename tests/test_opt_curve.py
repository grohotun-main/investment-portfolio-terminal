"""
Tests for parsers/opt_curve.py — cap-sweep tracer (vol vs concentration)
plus the covres pass-in kwarg it relies on in both suggest functions.

Run from phase1_build/ with:
    py -m unittest tests.test_opt_curve
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import min_variance as mv  # noqa: E402
import risk_parity as rp  # noqa: E402
import opt_curve as oc  # noqa: E402


def _prices(seed: int = 1, n_days: int = 300,
            syms: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-04-30", periods=n_days)
    data = {s: 100.0 * np.exp(np.cumsum(rng.standard_normal(n_days) * 0.01))
            for s in syms}
    return pd.DataFrame(data, index=idx)


def _cash_prices(seed: int = 5, n_days: int = 300) -> pd.DataFrame:
    """Two risky names + one near-riskless (CASH): the fixture where loose
    caps let both optimizers concentrate into CASH (the SGOV degeneracy)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-04-30", periods=n_days)
    return pd.DataFrame({
        "A": 100.0 * np.exp(np.cumsum(rng.standard_normal(n_days) * 0.01)),
        "B": 100.0 * np.exp(np.cumsum(rng.standard_normal(n_days) * 0.012)),
        "CASH": 100.0 * np.exp(
            np.cumsum(rng.standard_normal(n_days) * 0.0004)),
    }, index=idx)


class TestCovresPassIn(unittest.TestCase):
    """covres=<prebuilt> must be identical to letting the suggest build Σ."""

    def setUp(self) -> None:
        self.px = _cash_prices()
        self.weights = pd.Series({"A": 0.3, "B": 0.3, "CASH": 0.4})
        self.class_of = {"A": "equity", "B": "equity",
                         "CASH": "fixed_income"}
        self.covres = mv.build_covariance(
            self.px, list(self.weights.index), min_overlap_days=200)
        self.assertIsNone(self.covres["error"], self.covres["error"])

    def test_min_variance_covres_equivalent(self) -> None:
        kw = dict(name_cap=0.5, class_floors={"equity": 0.2},
                  min_overlap_days=200)
        a = mv.suggest_min_variance_grid(
            self.px, self.weights, self.class_of, **kw)
        b = mv.suggest_min_variance_grid(
            self.px, self.weights, self.class_of, covres=self.covres, **kw)
        self.assertEqual(a["kind"], "success")
        self.assertEqual(a["kind"], b["kind"])
        self.assertEqual(a["converged"], b["converged"])
        self.assertEqual(a["message"], b["message"])
        for s in a["new_pct"]:
            self.assertEqual(a["new_pct"][s], b["new_pct"][s], s)
        self.assertEqual(a["vol"], b["vol"])

    def test_risk_parity_covres_equivalent(self) -> None:
        kw = dict(name_cap=0.5, min_overlap_days=200)
        a = rp.suggest_risk_parity_grid(self.px, self.weights, **kw)
        b = rp.suggest_risk_parity_grid(self.px, self.weights,
                                        covres=self.covres, **kw)
        self.assertEqual(a["kind"], "success")
        self.assertEqual(a["kind"], b["kind"])
        self.assertEqual(a["converged"], b["converged"])
        self.assertEqual(a["message"], b["message"])
        for s in a["new_pct"]:
            self.assertEqual(a["new_pct"][s], b["new_pct"][s], s)
        self.assertEqual(a["vol"], b["vol"])


class TestTraceCapCurve(unittest.TestCase):
    def setUp(self) -> None:
        self.px = _cash_prices()
        self.weights = pd.Series({"A": 0.3, "B": 0.3, "CASH": 0.4})
        self.class_of = {"A": "equity", "B": "equity",
                         "CASH": "fixed_income"}

    def _trace(self, caps, **kw):
        return oc.trace_cap_curve(
            self.px, self.weights, self.class_of, caps=caps,
            class_floors={}, min_overlap_days=200, **kw)

    def test_shape_and_both_optimizers(self) -> None:
        out = self._trace([0.5, 1.0])
        self.assertIsNone(out["error"])
        pts = out["points"]
        self.assertEqual(list(pts.columns), oc.POINT_COLUMNS)
        self.assertEqual(len(pts), 4)  # 2 caps x 2 optimizers
        for cap in (0.5, 1.0):
            sub = pts[pts["cap"] == cap]
            self.assertEqual(sorted(sub["optimizer"]),
                             ["min_variance", "risk_parity"])
        self.assertTrue((pts["vol"] > 0).all())
        self.assertTrue((pts["effective_n"] >= 1.0).all())
        self.assertTrue(((pts["max_weight"] > 0)
                         & (pts["max_weight"] <= 1.0 + 1e-9)).all())

    def test_current_book_point(self) -> None:
        out = self._trace([1.0])
        cur = out["current"]
        self.assertGreater(cur["vol"], 0.0)
        # 1 / (0.3^2 + 0.3^2 + 0.4^2) = 1 / 0.34
        self.assertAlmostEqual(cur["effective_n"], 1.0 / 0.34, places=6)
        self.assertAlmostEqual(cur["max_weight"], 0.4, places=9)

    def test_on_point_progress(self) -> None:
        calls: list[tuple[int, int]] = []
        self._trace([0.5, 1.0], on_point=lambda d, t: calls.append((d, t)))
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[-1], (4, 4))
        self.assertTrue(all(t == 4 for _, t in calls))

    def test_caps_deduped_and_sorted(self) -> None:
        out = self._trace([1.0, 0.5, 1.0])  # duplicate + unsorted
        pts = out["points"]
        self.assertEqual(len(pts), 4)  # dupes collapsed
        mv_caps = list(pts[pts["optimizer"] == "min_variance"]["cap"])
        self.assertEqual(mv_caps, sorted(mv_caps))

    def test_concentration_degenerate_inputs_nan(self) -> None:
        # Unreachable via the suggests' success path; the helper must fail
        # soft (NaNs), not raise or return inf.
        for bad in ({}, {"A": 0.0, "B": 0.0}, {"A": float("nan"), "B": 50.0}):
            en, mx = oc._concentration(bad)
            self.assertTrue(np.isnan(en), bad)
            self.assertTrue(np.isnan(mx), bad)


class TestTraceProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.px = _cash_prices()
        self.weights = pd.Series({"A": 0.3, "B": 0.3, "CASH": 0.4})
        self.class_of = {"A": "equity", "B": "equity",
                         "CASH": "fixed_income"}

    def _trace(self, caps, weights=None, px=None):
        return oc.trace_cap_curve(
            px if px is not None else self.px,
            weights if weights is not None else self.weights,
            self.class_of, caps=caps, class_floors={},
            min_overlap_days=200)

    def test_min_variance_vol_non_increasing_in_cap(self) -> None:
        # Loosening the cap only enlarges the feasible set -> GMV vol can
        # only fall (within solver tolerance).
        pts = self._trace([0.4, 0.5, 0.75, 1.0])["points"]
        d = pts[pts["optimizer"] == "min_variance"].sort_values("cap")
        vols = d["vol"].to_numpy()
        self.assertEqual(len(vols), 4)
        self.assertTrue((np.diff(vols) <= 1e-6).all(), vols)

    def test_tight_cap_de_concentrates_both_optimizers(self) -> None:
        # The SGOV lesson as a curve property: at cap=1.0 both objectives
        # pile into near-riskless CASH; a tight cap forces spread.
        pts = self._trace([0.4, 1.0])["points"]
        for opt in ("min_variance", "risk_parity"):
            d = pts[pts["optimizer"] == opt]
            tight = d[d["cap"] == 0.4].iloc[0]
            loose = d[d["cap"] == 1.0].iloc[0]
            self.assertGreater(tight["effective_n"],
                               loose["effective_n"], opt)
            self.assertLessEqual(tight["max_weight"], 0.4 + 1e-6, opt)
            self.assertGreater(tight["max_weight"], 0.1, opt)

    def test_infeasible_rungs_skipped_not_fatal(self) -> None:
        # cap 0.2 x 3 names = 0.6 < 1 -> infeasible for both optimizers.
        out = self._trace([0.2, 0.5, 1.0])
        self.assertIsNone(out["error"])
        self.assertEqual(len(out["points"]), 4)  # 0.5 and 1.0 survive
        self.assertEqual(len(out["skipped"]), 2)
        self.assertEqual({s[0] for s in out["skipped"]}, {0.2})
        self.assertEqual(sorted(s[1] for s in out["skipped"]),
                         ["min_variance", "risk_parity"])
        self.assertTrue(all(isinstance(s[2], str) and s[2]
                            for s in out["skipped"]))

    def test_all_rungs_infeasible_empty_points_no_error(self) -> None:
        # Every cap infeasible -> empty points but error stays None (the
        # UI words this case itself); current is still computed.
        out = self._trace([0.1])
        self.assertIsNone(out["error"])
        self.assertTrue(out["points"].empty)
        self.assertEqual(len(out["skipped"]), 2)
        self.assertIsNotNone(out["current"])

    def test_sigma_failure_sets_error(self) -> None:
        short = _cash_prices(n_days=50)
        out = self._trace([0.5, 1.0], px=short)
        self.assertIsNotNone(out["error"])
        self.assertTrue(out["points"].empty)
        self.assertIsNone(out["current"])
        self.assertEqual(out["skipped"], [])

    def test_young_holding_held_at_every_point(self) -> None:
        # Young NEW (40 days) carries 60% -- more than any mature name can
        # reach (mature budget is 40%) -- so max_weight == 0.6 at every
        # point iff the hold rule applied throughout. Also exercises the
        # disclosed "held young weight can exceed the cap" edge (cap 0.5).
        mature = _cash_prices()[["A", "CASH"]]
        young = _prices(seed=9, n_days=40, syms=("NEW",))
        px = mature.join(young, how="outer")
        weights = pd.Series({"A": 0.25, "CASH": 0.15, "NEW": 0.60})
        out = oc.trace_cap_curve(
            px, weights, {"A": "equity", "CASH": "fixed_income",
                          "NEW": "equity"},
            caps=[0.5, 1.0], class_floors={}, min_overlap_days=200)
        self.assertIsNone(out["error"])
        pts = out["points"]
        self.assertEqual(len(pts), 4)
        self.assertTrue(np.allclose(pts["max_weight"], 0.60, atol=1e-6),
                        pts[["cap", "optimizer", "max_weight"]])


class TestClassCapsThreading(unittest.TestCase):
    def setUp(self) -> None:
        self.px = _cash_prices()
        self.weights = pd.Series({"A": 0.3, "B": 0.3, "CASH": 0.4})
        self.class_of = {"A": "equity", "B": "equity",
                         "CASH": "fixed_income"}

    def _trace(self, caps, **kw):
        return oc.trace_cap_curve(
            self.px, self.weights, self.class_of, caps=caps,
            class_floors={}, min_overlap_days=200, **kw)

    def test_both_optimizers_respond_to_class_cap(self) -> None:
        base = self._trace([0.5, 1.0])
        capped = self._trace([0.5, 1.0], class_caps={"fixed_income": 0.05})
        b, c = base["points"], capped["points"]
        for opt in ("min_variance", "risk_parity"):
            ob = b[b["optimizer"] == opt].reset_index(drop=True)
            oc_ = c[c["optimizer"] == opt].reset_index(drop=True)
            self.assertEqual(len(ob), len(oc_), opt)
            self.assertFalse(
                np.allclose(ob["vol"].to_numpy(), oc_["vol"].to_numpy()),
                f"{opt} points unchanged by an active class cap")

    def test_class_caps_absent_identical(self) -> None:
        a = self._trace([0.5, 1.0])
        b = self._trace([0.5, 1.0], class_caps=None)
        pd.testing.assert_frame_equal(a["points"], b["points"])

    def test_class_risk_budgets_none_identical(self) -> None:
        # The O2 None-twin: budgets=None must match the kwarg-absent path
        # bit-for-bit, like the class_caps twin above.
        a = self._trace([0.5, 1.0])
        b = self._trace([0.5, 1.0], class_risk_budgets=None)
        pd.testing.assert_frame_equal(a["points"], b["points"])

    def test_erc_responds_to_budgets_min_var_does_not(self) -> None:
        base = self._trace([0.5, 1.0])
        budgeted = self._trace([0.5, 1.0],
                               class_risk_budgets={"fixed_income": 0.1})
        b, c = base["points"], budgeted["points"]
        mv_b = b[b["optimizer"] == "min_variance"].reset_index(drop=True)
        mv_c = c[c["optimizer"] == "min_variance"].reset_index(drop=True)
        pd.testing.assert_frame_equal(mv_b, mv_c)          # budgets ERC-only
        erc_b = b[b["optimizer"] == "risk_parity"].reset_index(drop=True)
        erc_c = c[c["optimizer"] == "risk_parity"].reset_index(drop=True)
        self.assertEqual(len(erc_b), len(erc_c))
        self.assertFalse(np.allclose(erc_b["vol"].to_numpy(),
                                     erc_c["vol"].to_numpy()))


class TestPointWeights(unittest.TestCase):
    def setUp(self) -> None:
        self.px = _cash_prices()
        self.weights = pd.Series({"A": 0.3, "B": 0.3, "CASH": 0.4})
        self.class_of = {"A": "equity", "B": "equity",
                         "CASH": "fixed_income"}

    def test_points_carry_full_book_weights(self) -> None:
        out = oc.trace_cap_curve(self.px, self.weights, self.class_of,
                                 caps=[0.5, 1.0], class_floors={},
                                 min_overlap_days=200)
        pts = out["points"]
        self.assertIn("weights", pts.columns)
        self.assertFalse(pts.empty)
        for _, r in pts.iterrows():
            w = r["weights"]
            self.assertIsInstance(w, dict)
            self.assertTrue(all(isinstance(k, str) for k in w))
            self.assertTrue(np.isclose(sum(w.values()), 100.0, atol=1e-6), w)

    def test_point_weights_match_the_suggest_at_that_cap(self) -> None:
        from min_variance import build_covariance, suggest_min_variance_grid
        covres = build_covariance(self.px, list(self.weights.index),
                                  min_overlap_days=200)
        out = oc.trace_cap_curve(self.px, self.weights, self.class_of,
                                 caps=[0.5], class_floors={},
                                 min_overlap_days=200)
        row = out["points"]
        row = row[row["optimizer"] == "min_variance"].iloc[0]
        direct = suggest_min_variance_grid(
            self.px, self.weights, self.class_of, name_cap=0.5,
            class_floors={}, min_overlap_days=200, covres=covres)
        self.assertEqual(row["weights"], direct["new_pct"])


if __name__ == "__main__":
    unittest.main()
