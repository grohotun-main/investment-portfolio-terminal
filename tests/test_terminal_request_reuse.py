# tests/test_terminal_request_reuse.py
"""Request-scoped reuse seams (2026-07-15 spec): filter_option_ids must
reproduce every tab view's meta id sets, and the risksim bundle= seam must be
output-identical to the compute-internally path."""
import math, sys, types, unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "parsers"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import performance_service as ps
from terminal import benchmark_service as bs
from terminal import risk_service as rs
from terminal import riskcontrib_service as rcs
from terminal import risksim_service as rss


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-tolerant deep compare (same helper as the other
    terminal test files — covariance outputs aren't bit-reproducible across
    BLAS builds)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a is b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            m = _deep_close(a[k], b[k], rel=rel, abs_=abs_, path=f"{path}.{k}")
            if m: return m
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = _deep_close(x, y, rel=rel, abs_=abs_, path=f"{path}[{i}]")
            if m: return m
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (None if math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
                else f"{path}: {a!r} !~ {b!r}")
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _meta_ids(view):
    return ({o["id"] for o in view["meta"]["accounts"]},
            {o["id"] for o in view["meta"]["classes"]})


class TestFilterOptionIds(unittest.TestCase):
    """filter_option_ids == the id sets each tab view's meta lists. If a service
    ever diverges from the shared option source, its test here goes red — the
    helper must never silently loosen 422 validation."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_matches_holdings_latest(self):
        view = hs.build_holdings_view(self.frames)
        self.assertEqual(hs.filter_option_ids(self.frames), _meta_ids(view))

    def test_matches_holdings_non_latest_as_of(self):
        dates = list(self.frames.available_dates)
        if len(dates) < 2:
            self.skipTest("fixture has a single as-of date")
        older = dates[-1]  # available_dates is newest-first
        view = hs.build_holdings_view(self.frames, as_of=older)
        self.assertEqual(hs.filter_option_ids(self.frames, as_of=older),
                         _meta_ids(view))

    def test_matches_performance(self):
        self.assertEqual(hs.filter_option_ids(self.frames),
                         _meta_ids(ps.build_performance_view(self.frames)))

    def test_matches_benchmark(self):
        self.assertEqual(hs.filter_option_ids(self.frames),
                         _meta_ids(bs.build_benchmark_view(self.frames)))

    def test_matches_risk(self):
        self.assertEqual(hs.filter_option_ids(self.frames),
                         _meta_ids(rs.build_risk_view(self.frames)))

    def test_matches_riskcontrib(self):
        self.assertEqual(hs.filter_option_ids(self.frames),
                         _meta_ids(rcs.build_riskcontrib_view(self.frames)))

    def test_matches_risksim(self):
        self.assertEqual(hs.filter_option_ids(self.frames),
                         _meta_ids(rss.build_risksim_view(self.frames)))

    def test_fixture_discriminates_as_of(self):
        """Guards the non-latest as_of test's power: if a fixture regen ever
        makes the oldest month's option sets identical to the latest, the
        as_of equivalence test silently goes vacuous — this makes it loud."""
        dates = list(self.frames.available_dates)
        if len(dates) < 2:
            self.skipTest("fixture has a single as-of date")
        self.assertNotEqual(hs.filter_option_ids(self.frames, as_of=dates[-1]),
                            hs.filter_option_ids(self.frames))


class TestBundleSeam(unittest.TestCase):
    """Each risksim entry point must return the identical result whether it
    computes its own bundle or receives the handler's precomputed one."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.bundle = rss._bundle_for(cls.frames, "all", "all")
        view = rss.build_risksim_view(cls.frames)
        raw = {r["ticker"]: r["now_pct"] for r in view["grid"]["rows"]}
        cls.cap_default = float(view["optimizer"]["cap_default_pct"])
        # A genuine reweight: move up to 1pp from the largest name to the
        # smallest (sum stays 100 so Run validates, but it is NOT a no-op —
        # posting the grid back unchanged trips the engine's
        # "new_weights == current_weights: nothing to simulate" guard).
        ordered = sorted(raw, key=raw.get)
        lo, hi = ordered[0], ordered[-1]
        shift = min(1.0, raw[hi] / 2.0)
        cls.weights = dict(raw)
        cls.weights[hi] -= shift
        cls.weights[lo] += shift

    def test_view_identical_with_bundle(self):
        a = rss.build_risksim_view(self.frames)
        b = rss.build_risksim_view(self.frames, bundle=self.bundle)
        self.assertIsNone(_deep_close(a, b))

    def test_simulate_identical_with_bundle(self):
        a = rss.run_simulation(self.frames, self.weights)
        b = rss.run_simulation(self.frames, self.weights, bundle=self.bundle)
        self.assertIsNone(a["error"])  # must exercise the happy path, not a guard
        self.assertIsNone(_deep_close(a, b))

    def test_simulate_guard_path_identical(self):
        a = rss.run_simulation(self.frames, {})
        b = rss.run_simulation(self.frames, {}, bundle=self.bundle)
        self.assertEqual(a, b)
        self.assertIn("error", a)
        self.assertIsNotNone(a["error"])

    def test_optimize_identical_with_bundle(self):
        kw = dict(optimizer="min_variance", cap_pct=self.cap_default, floors={})
        a = rss.run_optimize(self.frames, **kw)
        b = rss.run_optimize(self.frames, **kw, bundle=self.bundle)
        self.assertEqual(a["kind"], "success")  # success path must cross the seam
        self.assertIsNone(_deep_close(a, b))

    def test_optimize_infeasible_cap_identical(self):
        kw = dict(optimizer="min_variance", cap_pct=25.0, floors={})
        a = rss.run_optimize(self.frames, **kw)
        b = rss.run_optimize(self.frames, **kw, bundle=self.bundle)
        self.assertEqual(a["kind"], "error")  # in-band domain failure, same both ways
        self.assertIsNone(_deep_close(a, b))

    def test_optimize_guard_path_identical(self):
        kw = dict(optimizer="nope", cap_pct=25.0, floors={})
        a = rss.run_optimize(self.frames, **kw)
        b = rss.run_optimize(self.frames, **kw, bundle=self.bundle)
        self.assertEqual(a, b)
        self.assertEqual(a["kind"], "error")

    def test_trace_identical_with_bundle(self):
        kw = dict(cap_pct=25.0, floors={})
        a = rss.run_trace(self.frames, **kw)
        b = rss.run_trace(self.frames, **kw, bundle=self.bundle)
        self.assertIsNone(a["error"])  # solver path, not an error-shape compare
        self.assertIsNone(_deep_close(a, b))


class TestFloorBuckets(unittest.TestCase):
    """floor_buckets_for must equal the view-derived floor-key set the optimize/
    trace handlers validated against before this change — including the guard
    corner where the seed block is absent (empty set -> any floor key 422s)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.bundle = rss._bundle_for(cls.frames, "all", "all")

    def test_matches_view_seed_buckets(self):
        view = rss.build_risksim_view(self.frames)
        known = {b["key"] for b in (view.get("optimizer") or {}).get("buckets", [])}
        self.assertEqual(rss.floor_buckets_for(self.frames, self.bundle), known)

    def test_empty_on_missing_twr_guard(self):
        stub = types.SimpleNamespace(twr_portfolio=None,
                                     daily_prices=self.frames.daily_prices)
        self.assertEqual(rss.floor_buckets_for(stub, self.bundle), set())

    def test_empty_on_empty_weights(self):
        bundle = {"weights": pd.Series(dtype=float)}
        self.assertEqual(rss.floor_buckets_for(self.frames, bundle), set())

    def test_empty_on_missing_daily_prices(self):
        stub = types.SimpleNamespace(twr_portfolio=self.frames.twr_portfolio,
                                     daily_prices=pd.DataFrame())
        self.assertEqual(rss.floor_buckets_for(stub, self.bundle), set())


if __name__ == "__main__":
    unittest.main()
