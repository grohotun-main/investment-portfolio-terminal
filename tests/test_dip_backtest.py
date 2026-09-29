# tests/test_dip_backtest.py
"""Walk-forward referee harness (parsers/dip_backtest.py) — mechanics,
referee/primary aggregation branches, CLI smoke. Spec:
docs/superpowers/specs/2026-07-14-dip-verdict-backtest-design.md"""
import contextlib
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from parsers import dip_backtest as db
from parsers import dip_analytics as da

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "synth_data"


def _mk_series(n=900, dip_at=400, dip_len=60, dip_frac=0.12, start="2019-01-01"):
    """Gently rising series with one mid-depth dip that stays underwater to
    the last bar. n=900 puts the dip INSIDE the evaluable window (the 252d
    frontier censor would hide a dip in the last year). Deterministic."""
    px = np.linspace(100.0, 160.0, n)
    trough = px[dip_at] * (1.0 - dip_frac)
    px[dip_at:dip_at + dip_len] = np.linspace(px[dip_at], trough, dip_len)
    px[dip_at + dip_len:] = np.linspace(trough, trough * 1.06,
                                        n - dip_at - dip_len)
    idx = pd.bdate_range(start, periods=n)
    price = pd.Series(px, index=idx)
    return price, price.copy()


class WalkForwardMechanicsTests(unittest.TestCase):
    def test_burn_in_stride_and_censor_boundaries(self):
        price, tr = _mk_series()
        rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)
        n = len(price)
        first_ok = price.index[0] + pd.DateOffset(years=1)
        i0 = int(np.searchsorted(price.index, first_ok))
        eligible = [i for i in range(i0, n) if i + 252 < n]
        expect_dates = [price.index[i] for i in eligible[::5]]
        self.assertEqual(list(rows["date"]), expect_dates)
        # censor: every row has a full realized 252d forward window
        self.assertFalse(rows["fwd_252"].isna().any())

    def test_realized_forwards_are_positional_total_returns(self):
        price, tr = _mk_series()
        rows = db.walk_forward(price, tr, stride=7, burn_in_years=1)
        r = rows.iloc[3]
        i = int(price.index.get_loc(r["date"]))
        arr = tr.to_numpy(dtype=float)
        for h in (21, 63, 126, 252):
            self.assertAlmostEqual(r[f"fwd_{h}"], arr[i + h] / arr[i] - 1.0,
                                   places=12)

    def test_no_look_ahead(self):
        """THE keystone test (spec S8): appending an adversarial future —
        including a crash deeper than anything before — must not change any
        already-evaluated row."""
        price, tr = _mk_series()
        base = db.walk_forward(price, tr, stride=5, burn_in_years=1)
        crash = np.linspace(price.iloc[-1], price.iloc[-1] * 0.55, 200)
        ext_idx = pd.bdate_range(price.index[-1] + pd.offsets.BDay(1),
                                 periods=200)
        price2 = pd.concat([price, pd.Series(crash, index=ext_idx)])
        rows2 = db.walk_forward(price2, price2.copy(), stride=5,
                                burn_in_years=1)
        merged = base.merge(rows2, on="date", suffixes=("_a", "_b"))
        self.assertEqual(len(merged), len(base))     # every base date re-eval'd
        for col in ("band", "depth_pctile", "omega", "edge", "edge_ci_lo",
                    "rr_pct", "n_cond", "episode_id", "fwd_21", "fwd_63",
                    "fwd_126", "fwd_252"):
            a, b = merged[f"{col}_a"], merged[f"{col}_b"]
            if col == "band":   # string column (pandas-3 "str" dtype != object)
                self.assertTrue((a == b).all(), col)
            else:
                np.testing.assert_allclose(a.astype(float), b.astype(float),
                                           rtol=0, atol=0, err_msg=col)

    def test_deterministic(self):
        price, tr = _mk_series()
        r1 = db.walk_forward(price, tr, stride=10, burn_in_years=1)
        r2 = db.walk_forward(price, tr, stride=10, burn_in_years=1)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rows_carry_edge_ci_lo(self):
        """Gate-experiment seam (spec 2026-07-18): every row records the
        verdict's edge-CI lower bound so candidate gates can be relabeled
        from stored fields without re-running the replay."""
        price, tr = _mk_series()
        rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)
        self.assertIn("edge_ci_lo", rows.columns)
        r = rows.iloc[5]
        i = int(price.index.get_loc(r["date"]))
        blk = da.dip_verdict_block(price.iloc[:i + 1], tr.iloc[:i + 1],
                                   horizons=(da.VERDICT_HORIZON,))
        expect = blk["verdict"]["edge_ci"]["lo"]
        if math.isnan(expect):
            self.assertTrue(math.isnan(r["edge_ci_lo"]))
        else:
            self.assertAlmostEqual(float(r["edge_ci_lo"]), expect, places=12)

    def test_episode_ids_cluster_the_dip(self):
        price, tr = _mk_series()
        rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)
        in_dip = rows[rows["date"] >= price.index[401]]
        self.assertTrue((~in_dip["episode_id"].isna()).all())
        self.assertEqual(in_dip["episode_id"].nunique(), 1)

    def test_insufficient_history_raises(self):
        price, tr = _mk_series(n=300, dip_at=200, dip_len=30)
        with self.assertRaises(ValueError):
            db.walk_forward(price, tr, stride=5, burn_in_years=10)

    def test_mismatched_index_raises(self):
        price, tr = _mk_series()
        with self.assertRaises(ValueError):
            db.walk_forward(price, tr.iloc[:-1], stride=5, burn_in_years=1)


def _gate_rows(specs):
    """Eval-rows frame from (band, omega, edge, edge_ci_lo, rr_pct) tuples —
    full control over the verdict internals gate_relabel reads."""
    n = len(specs)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "band": [s[0] for s in specs],
        "depth_pctile": [80.0] * n,
        "omega": [float(s[1]) for s in specs],
        "edge": [float(s[2]) for s in specs],
        "edge_ci_lo": [float(s[3]) for s in specs],
        "rr_pct": [float(s[4]) for s in specs],
        "n_cond": [30] * n,
        "episode_id": [float(i) for i in range(n)],
        "fwd_21": [0.01] * n, "fwd_63": [0.01] * n,
        "fwd_126": [0.01] * n, "fwd_252": [0.01] * n,
    })


class GateRelabelTests(unittest.TestCase):
    """Candidate gates as exact relabelings (spec 2026-07-18 strong-gate
    recalibration): strong/neutral re-split from stored verdict internals;
    bands upstream of the strong condition never promote."""

    INF = float("inf")
    NAN = float("nan")

    def _relabel(self, specs, **gate):
        return list(db.gate_relabel(_gate_rows(specs), **gate))

    def test_candidate_set_is_the_registration(self):
        self.assertEqual(
            [(c["id"], c["strong_rr"], c["requires_ci"], c["preference"])
             for c in db.GATE_CANDIDATES],
            [("rr0", 0.0, True, 1), ("rr50", 0.5, True, 2),
             ("pt67", 0.67, False, 3)])

    def test_non_edge_bands_never_promote(self):
        specs = [("shallow", 5.0, 4.0, 3.0, 0.9),
                 ("inconclusive", self.INF, self.NAN, self.NAN, 0.9),
                 ("weak", 0.5, -1.0, -2.0, 0.9)]
        for cand in db.GATE_CANDIDATES:
            got = self._relabel(specs, strong_rr=cand["strong_rr"],
                                requires_ci=cand["requires_ci"])
            self.assertEqual(got, ["shallow", "inconclusive", "weak"],
                             cand["id"])

    def test_inf_omega_branch_uses_rank_only(self):
        specs = [("neutral", self.INF, self.NAN, self.NAN, 0.5),
                 ("strong", self.INF, self.NAN, self.NAN, 0.8),
                 ("neutral", self.INF, self.NAN, self.NAN, self.NAN)]
        self.assertEqual(self._relabel(specs, strong_rr=0.0, requires_ci=True),
                         ["strong", "strong", "neutral"])
        self.assertEqual(self._relabel(specs, strong_rr=0.5, requires_ci=True),
                         ["strong", "strong", "neutral"])
        self.assertEqual(self._relabel(specs, strong_rr=0.67,
                                       requires_ci=False),
                         ["neutral", "strong", "neutral"])

    def test_both_axes_on_finite_path(self):
        low_rank = ("neutral", 2.0, 1.0, 0.4, 0.3)    # CI cleared, rank low
        no_ci = ("neutral", 3.0, 2.0, -0.5, 0.9)      # CI failed, rank high
        nan_ci = ("neutral", 3.0, 2.0, self.NAN, 0.9)  # CI NaN, rank high
        inc_strong = ("strong", 4.0, 3.0, 1.5, 0.7)   # incumbent strong
        specs = [low_rank, no_ci, nan_ci, inc_strong]
        self.assertEqual(self._relabel(specs, strong_rr=0.0, requires_ci=True),
                         ["strong", "neutral", "neutral", "strong"])
        self.assertEqual(self._relabel(specs, strong_rr=0.5, requires_ci=True),
                         ["neutral", "neutral", "neutral", "strong"])
        self.assertEqual(self._relabel(specs, strong_rr=0.67,
                                       requires_ci=False),
                         ["neutral", "strong", "strong", "strong"])

    def test_identity_at_incumbent_on_walkforward_rows(self):
        """The identity control (spec): relabeling at the shipped gate must
        reproduce the stored bands exactly."""
        price, tr = _mk_series()
        rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)
        got = db.gate_relabel(rows, strong_rr=da.VERDICT_STRONG_RR,
                              requires_ci=True)
        self.assertTrue((got == rows["band"]).all())


def _rows(bands, episodes, fwd, depth=None):
    """Hand-built eval-rows frame with full control over bands/episodes/
    outcomes (mirrors walk_forward's columns)."""
    n = len(bands)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "band": bands,
        "depth_pctile": depth if depth is not None else [80.0] * n,
        "omega": [1.0] * n, "edge": [0.0] * n, "rr_pct": [0.5] * n,
        "n_cond": [30] * n, "episode_id": episodes,
        "fwd_21": fwd, "fwd_63": fwd, "fwd_126": fwd, "fwd_252": fwd,
    })


class RefereeTableTests(unittest.TestCase):
    def test_per_band_and_all_rows(self):
        rows = _rows(bands=["neutral", "neutral", "weak", "shallow"],
                     episodes=[0.0, 1.0, 1.0, float("nan")],
                     fwd=[0.10, 0.20, -0.10, 0.05])
        t = db.referee_table(rows)
        self.assertEqual(list(t.columns), list(db._AGG_COLS))
        self.assertEqual(list(t.index),
                         ["strong", "neutral", "weak", "inconclusive",
                          "shallow", "all"])
        self.assertEqual(t.loc["neutral", "n_days"], 2)
        self.assertEqual(t.loc["neutral", "n_episodes"], 2)
        self.assertAlmostEqual(t.loc["neutral", "med_252"], 0.15)
        self.assertAlmostEqual(t.loc["neutral", "hit_252"], 1.0)
        self.assertEqual(t.loc["strong", "n_days"], 0)
        self.assertTrue(math.isnan(t.loc["strong", "med_252"]))
        self.assertEqual(t.loc["all", "n_days"], 4)
        self.assertEqual(t.loc["all", "n_episodes"], 2)   # NaN id not counted
        self.assertAlmostEqual(t.loc["all", "hit_252"], 0.75)

    def test_tk_rule_needs_depth_and_edge(self):
        rows = _rows(bands=["neutral", "neutral", "weak", "strong"],
                     episodes=[0.0, 1.0, 1.0, 2.0],
                     fwd=[0.10, 0.20, -0.10, 0.30],
                     depth=[90.0, 70.0, 95.0, 99.0])
        t = db.tk_rule_row(rows)
        self.assertEqual(t.loc["tk_rule", "n_days"], 2)    # rows 0 and 3 only
        self.assertAlmostEqual(t.loc["tk_rule", "med_252"], 0.20)

    def test_depth_decile_curve_includes_100(self):
        rows = _rows(bands=["weak"] * 3, episodes=[0.0, 1.0, 2.0],
                     fwd=[0.01, 0.02, 0.03], depth=[0.0, 95.0, 100.0])
        c = db.depth_decile_curve(rows)
        self.assertEqual(c.loc["0-10", "n_days"], 1)
        self.assertEqual(c.loc["90-100", "n_days"], 2)     # 100.0 included
        self.assertEqual(c["n_days"].sum(), 3)


class PrimaryMetricTests(unittest.TestCase):
    def test_block_rows_scale_with_stride(self):
        rows = _rows(bands=["weak"] * 10, episodes=[float(i) for i in range(10)],
                     fwd=[0.01] * 10)
        self.assertEqual(db.primary_metric(rows, stride=5)["block_rows"], 51)
        self.assertEqual(db.primary_metric(rows, stride=21)["block_rows"], 12)
        self.assertEqual(db.primary_metric(rows, stride=1)["block_rows"], 252)

    def test_episode_gate_forces_inconclusive(self):
        bands = (["neutral"] * 20) + (["weak"] * 20)
        eps = [float(i % 4) for i in range(20)] + [9.0] * 20   # 4 edge episodes
        fwd = [0.10] * 18 + [-0.01] * 2 + [-0.05, 0.01] * 10
        pm = db.primary_metric(_rows(bands, eps, fwd))
        self.assertEqual(pm["n_edge_episodes"], 4)
        self.assertEqual(pm["outcome"], "inconclusive")

    def test_validated_when_ci_clears_and_episodes_suffice(self):
        bands = (["neutral"] * 20) + (["weak"] * 20)
        eps = [float(i % 6) for i in range(20)] + [9.0] * 20   # 6 edge episodes
        fwd = [0.10] * 18 + [-0.01] * 2 + [-0.05, 0.01] * 10
        pm = db.primary_metric(_rows(bands, eps, fwd))
        self.assertEqual(pm["n_edge_episodes"], 6)
        self.assertGreater(pm["stat"], 0.0)
        self.assertGreater(pm["ci_lo"], 0.0)
        self.assertEqual(pm["outcome"], "validated")

    def test_not_supported_when_no_edge_difference(self):
        bands = (["neutral"] * 20) + (["weak"] * 20)
        eps = [float(i % 6) for i in range(20)] + [9.0] * 20
        fwd = [0.02, -0.02] * 20                # identical mix, no edge
        pm = db.primary_metric(_rows(bands, eps, fwd))
        self.assertEqual(pm["outcome"], "not_supported")

    def test_no_edge_rows_is_inconclusive_nan_stat(self):
        rows = _rows(bands=["weak"] * 12, episodes=[float(i) for i in range(12)],
                     fwd=[0.01, -0.01] * 6)
        pm = db.primary_metric(rows)
        self.assertTrue(math.isnan(pm["stat"]))
        self.assertEqual(pm["n_edge_days"], 0)
        self.assertEqual(pm["outcome"], "inconclusive")

    def test_deterministic(self):
        bands = (["neutral"] * 20) + (["weak"] * 20)
        eps = [float(i % 6) for i in range(20)] + [9.0] * 20
        fwd = [0.10] * 18 + [-0.01] * 2 + [-0.05, 0.01] * 10
        self.assertEqual(db.primary_metric(_rows(bands, eps, fwd)),
                         db.primary_metric(_rows(bands, eps, fwd)))


class StrongPrimaryMetricTests(unittest.TestCase):
    """Per-candidate registered primary (spec 2026-07-18): Omega(strong_G)
    − Omega(all) under the referee's bootstrap conventions; episode gate 5;
    an all-win strong set validates via the declared omega_inf branch."""

    def _tape(self, n_eps):
        bands = (["strong"] * 20) + (["weak"] * 20)
        eps = [float(i % n_eps) for i in range(20)] + [9.0] * 20
        fwd = [0.10] * 18 + [-0.01] * 2 + [-0.05, 0.01] * 10
        return _rows(bands, eps, fwd)

    def test_block_rows_scale_with_stride(self):
        rows = _rows(bands=["weak"] * 10,
                     episodes=[float(i) for i in range(10)], fwd=[0.01] * 10)
        bands = rows["band"].astype(str)
        for stride, expect in ((5, 51), (21, 12), (1, 252)):
            pm = db.strong_primary_metric(rows, bands, stride=stride)
            self.assertEqual(pm["block_rows"], expect)

    def test_episode_gate_forces_inconclusive(self):
        rows = self._tape(4)
        pm = db.strong_primary_metric(rows, rows["band"].astype(str))
        self.assertEqual(pm["n_strong_episodes"], 4)
        self.assertEqual(pm["outcome"], "inconclusive")

    def test_validated_when_ci_clears(self):
        rows = self._tape(6)
        pm = db.strong_primary_metric(rows, rows["band"].astype(str))
        self.assertEqual(pm["n_strong_episodes"], 6)
        self.assertEqual(pm["n_strong_days"], 20)
        self.assertGreater(pm["stat"], 0.0)
        self.assertGreater(pm["ci_lo"], 0.0)
        self.assertFalse(pm["omega_inf"])
        self.assertEqual(pm["outcome"], "validated")

    def test_all_win_strong_set_validates_with_flag(self):
        """No losing strong outcomes across >=5 episodes → the pre-declared
        omega_inf branch: validated even though every bootstrap stat is
        inf/nan and the CI is empty."""
        bands = (["strong"] * 12) + (["weak"] * 28)
        eps = [float(i % 6) for i in range(12)] + [9.0] * 28
        fwd = [0.10] * 12 + [-0.05, 0.01] * 14
        rows = _rows(bands, eps, fwd)
        pm = db.strong_primary_metric(rows, rows["band"].astype(str))
        self.assertTrue(pm["omega_inf"])
        self.assertTrue(math.isinf(pm["stat"]) and pm["stat"] > 0)
        self.assertEqual(pm["outcome"], "validated")

    def test_composes_with_gate_relabel(self):
        """CI-cleared low-rank neutrals: empty strong set at the incumbent
        (inconclusive), a 20-episode validated strong set under rr0."""
        specs = ([("neutral", 2.0, 1.0, 0.4, 0.3)] * 20
                 + [("weak", 0.5, -1.0, -2.0, 0.9)] * 20)
        rows = _gate_rows(specs)
        inc = db.strong_primary_metric(
            rows, db.gate_relabel(rows, strong_rr=0.67, requires_ci=True))
        self.assertEqual(inc["n_strong_days"], 0)
        self.assertEqual(inc["outcome"], "inconclusive")
        cand = db.strong_primary_metric(
            rows, db.gate_relabel(rows, strong_rr=0.0, requires_ci=True))
        self.assertEqual(cand["n_strong_days"], 20)
        self.assertEqual(cand["n_strong_episodes"], 20)
        self.assertEqual(cand["outcome"], "validated")

    def test_deterministic(self):
        rows = self._tape(6)
        bands = rows["band"].astype(str)
        self.assertEqual(db.strong_primary_metric(rows, bands),
                         db.strong_primary_metric(rows, bands))


class CliTests(unittest.TestCase):
    def _run(self, argv):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), \
                contextlib.redirect_stderr(buf_err):
            code = db.main(argv)
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_smoke_on_synth_fixture(self):
        code, out, _ = self._run(["--ticker", "SPY",
                                  "--data-dir", str(FIXTURE_DIR),
                                  "--burn-in-years", "1", "--stride", "5"])
        self.assertEqual(code, 0)
        self.assertIn("referee table", out)
        self.assertIn("primary metric", out)
        self.assertIn("depth-decile", out)

    def test_unknown_ticker_exits_nonzero(self):
        code, _, err = self._run(["--ticker", "NOPE",
                                  "--data-dir", str(FIXTURE_DIR),
                                  "--burn-in-years", "1"])
        self.assertEqual(code, 1)
        self.assertIn("NOPE", err)

    def test_insufficient_history_exits_nonzero(self):
        code, _, err = self._run(["--ticker", "SPY",
                                  "--data-dir", str(FIXTURE_DIR),
                                  "--burn-in-years", "10"])
        self.assertEqual(code, 1)
        self.assertIn("insufficient history", err)

    def test_write_dumps_rows_csv(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(FIXTURE_DIR / "dip_history.csv", td)
            code, _, _ = self._run(["--ticker", "SPY", "--data-dir", td,
                                    "--burn-in-years", "1", "--stride", "5",
                                    "--write"])
            self.assertEqual(code, 0)
            out_csv = Path(td) / "dip_backtest_SPY.csv"
            self.assertTrue(out_csv.exists())
            dumped = pd.read_csv(out_csv)
            self.assertIn("band", dumped.columns)
            self.assertIn("fwd_252", dumped.columns)


class RegisteredGuardFailFastTests(unittest.TestCase):
    """Every registered writer refuses a non-SPY run as an ARGUMENT check —
    before any data load, walk-forward or bootstrap. Proven against an EMPTY
    data dir: a guard placed after the load would report the missing CSV
    instead (and on a real data dir would burn a whole walk-forward +
    bootstrap replay only to refuse)."""

    FLAGS = ("--write-registered", "--write-ladder-registered",
             "--write-gate-registered", "--write-ladder-v2-registered")

    def _run(self, argv):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), \
                contextlib.redirect_stderr(buf_err):
            code = db.main(argv)
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_non_spy_refused_before_any_data_is_read(self):
        with tempfile.TemporaryDirectory() as td:      # deliberately empty
            for flag in self.FLAGS:
                with self.subTest(flag=flag):
                    code, out, err = self._run(
                        ["--ticker", "SCHD", "--data-dir", td, flag])
                    self.assertEqual(code, 1)
                    self.assertIn("SPY-only", err)
                    self.assertNotIn("not found", err)   # never reached IO
                    self.assertNotIn("referee table", out)
                    self.assertEqual(list(Path(td).iterdir()), [])

    def test_extended_spx_refused_by_write_registered_before_any_load(self):
        with tempfile.TemporaryDirectory() as td:      # deliberately empty
            code, out, err = self._run(["--ticker", "SPX", "--extended-spx",
                                        "--data-dir", td,
                                        "--write-registered"])
        self.assertEqual(code, 1)
        self.assertIn("SPY-only", err)
        self.assertNotIn("not found", err)
        self.assertNotIn("referee table", out)


class RegisteredArtifactTests(unittest.TestCase):
    """build_registered_artifact: schema, json-safety, determinism (spec
    2026-07-16 §4). No wall clock anywhere — two calls must be identical."""

    @classmethod
    def setUpClass(cls):
        price, tr = _mk_series()
        cls.rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)

    def test_schema_and_determinism(self):
        a = db.build_registered_artifact(self.rows, ticker="tst",
                                         stride=5, burn_in_years=1)
        b = db.build_registered_artifact(self.rows, ticker="tst",
                                         stride=5, burn_in_years=1)
        self.assertEqual(a, b)
        self.assertEqual(a["schema"], 1)
        self.assertEqual(a["ticker"], "TST")
        self.assertEqual(a["registered"], db.REGISTERED_DATE)
        self.assertEqual(a["config"], {"stride": 5, "burn_in_years": 1,
                                       "censor": 252, "horizon": 252})
        self.assertEqual(a["evals"]["n"], len(self.rows))
        self.assertEqual(a["evals"]["first"],
                         str(self.rows["date"].iloc[0].date()))
        self.assertEqual(a["evals"]["last"],
                         str(self.rows["date"].iloc[-1].date()))
        self.assertEqual(set(a["referee"]), set(db.BAND_ORDER) | {"all"})
        for rec in a["referee"].values():
            self.assertTrue({"n_days", "n_episodes", "med_252", "hit_252",
                             "omega_252"} <= set(rec))
        self.assertIn("outcome", a["primary"])
        self.assertIn("tk_rule", a)
        json.dumps(a, allow_nan=False)   # end-to-end json-safe, no NaN/inf

    def test_json_safe_agg_inf_and_nan(self):
        rec = {"n_days": 3, "n_episodes": 1, "med_252": float("nan"),
               "hit_252": 1.0, "omega_252": float("inf"),
               "med_126": 0.1, "med_63": 0.05, "med_21": 0.01}
        safe = db._json_safe_agg(rec)
        self.assertIsNone(safe["med_252"])
        self.assertIsNone(safe["omega_252"])
        self.assertTrue(safe["omega_252_inf"])
        self.assertEqual(safe["hit_252"], 1.0)
        json.dumps(safe, allow_nan=False)

    def test_json_safe_agg_finite_passthrough_no_flag(self):
        rec = {"n_days": 5, "n_episodes": 2, "med_252": 0.2, "hit_252": 0.9,
               "omega_252": 4.5, "med_126": 0.1, "med_63": 0.04, "med_21": 0.01}
        safe = db._json_safe_agg(rec)
        self.assertEqual(safe, rec)          # no omega_252_inf key when finite


class RegisteredLoaderTests(unittest.TestCase):
    """load_registered_artifact: round-trip, and None on every failure mode —
    a broken artifact must never take the dip tab down (spec §7)."""

    def _artifact(self):
        price, tr = _mk_series()
        rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)
        return db.build_registered_artifact(rows, ticker="TST", stride=5,
                                            burn_in_years=1)

    def test_round_trip_and_missing(self):
        art = self._artifact()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reg.json"
            p.write_bytes((json.dumps(art, indent=2) + "\n").encode("ascii"))
            self.assertEqual(db.load_registered_artifact(p), art)
            self.assertIsNone(
                db.load_registered_artifact(Path(td) / "absent.json"))

    def test_corrupt_and_schema_drift_return_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reg.json"
            p.write_text("{not json", encoding="ascii")
            self.assertIsNone(db.load_registered_artifact(p))
            p.write_text(json.dumps({"schema": 2, "ticker": "SPY"}),
                         encoding="ascii")
            self.assertIsNone(db.load_registered_artifact(p))
            art = self._artifact()
            del art["referee"]["weak"]        # band missing -> drift -> None
            p.write_text(json.dumps(art), encoding="ascii")
            self.assertIsNone(db.load_registered_artifact(p))
            art = self._artifact()
            art["referee"] = None             # json-valid, wrong type -> None
            p.write_text(json.dumps(art), encoding="ascii")
            self.assertIsNone(db.load_registered_artifact(p))
            art = self._artifact()
            del art["primary"]                # required key missing -> None
            p.write_text(json.dumps(art), encoding="ascii")
            self.assertIsNone(db.load_registered_artifact(p))

    def test_cli_write_and_spy_guard(self):
        """--write-registered writes the SPY incumbent verdict and refuses a
        non-SPY ticker (spec 2026-07-16) — the same SPY-only guard the ladder/
        gate/ladder-v2 registered writers carry."""
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(FIXTURE_DIR / "dip_history.csv", td)
            orig = db.REGISTERED_PATH
            db.REGISTERED_PATH = Path(td) / "reg.json"
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = db.main(["--ticker", "SPY", "--data-dir", td,
                                  "--burn-in-years", "1", "--stride", "5",
                                  "--write-registered"])
                self.assertEqual(rc, 0)
                self.assertIn("wrote", buf.getvalue())
                art = db.load_registered_artifact(db.REGISTERED_PATH)
                self.assertIsNotNone(art)
                self.assertEqual(art["ticker"], "SPY")
                b = db.REGISTERED_PATH.read_bytes()
                self.assertNotIn(b"\r\n", b)          # LF only, deterministic
                self.assertTrue(b.endswith(b"\n"))
                self.assertTrue(all(x < 128 for x in b))   # pure ASCII
                # non-SPY: exit 1 with the guard's reason, nothing written
                db.REGISTERED_PATH = Path(td) / "reg2.json"
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(err):
                    rc2 = db.main(["--ticker", "SCHD", "--data-dir", td,
                                   "--burn-in-years", "1", "--stride", "5",
                                   "--write-registered"])
                self.assertEqual(rc2, 1)
                self.assertIn("SPY-only", err.getvalue())
                self.assertFalse(db.REGISTERED_PATH.exists())
            finally:
                db.REGISTERED_PATH = orig


class GateArtifactTests(unittest.TestCase):
    """build_gate_registered_artifact + CLI + loader (spec 2026-07-18):
    schema, ordered selection, the incumbent identity guard, json-safety,
    determinism; SPY-only CLI guard; loader never raises."""

    @classmethod
    def setUpClass(cls):
        price, tr = _mk_series()
        cls.rows = db.walk_forward(price, tr, stride=5, burn_in_years=1)

    def test_schema_and_determinism(self):
        a = db.build_gate_registered_artifact(self.rows, ticker="tst",
                                              stride=5, burn_in_years=1)
        b = db.build_gate_registered_artifact(self.rows, ticker="tst",
                                              stride=5, burn_in_years=1)
        self.assertEqual(a, b)
        self.assertEqual(a["schema"], 1)
        self.assertEqual(a["kind"], "gate")
        self.assertEqual(a["ticker"], "TST")
        self.assertEqual(a["registered"], db.GATE_REGISTERED_DATE)
        self.assertEqual(a["config"]["incumbent"],
                         {"strong_rr": da.VERDICT_STRONG_RR,
                          "requires_ci": True})
        self.assertEqual(a["config"]["min_episodes"], db.MIN_EDGE_EPISODES)
        self.assertEqual([c["id"] for c in a["candidates"]],
                         ["rr0", "rr50", "pt67"])
        for c in a["candidates"]:
            self.assertIn("outcome", c["primary"])
            self.assertIn("strong", c["descriptive"])
            self.assertIn("residual_neutral", c["descriptive"])
        self.assertIn("primary", a["incumbent"])
        self.assertIn(a["selected"], (None, "rr0", "rr50", "pt67"))
        self.assertIn("selection_rule", a)
        json.dumps(a, allow_nan=False)   # end-to-end json-safe

    def test_incumbent_identity_guard(self):
        """A rows frame whose stored bands the incumbent relabel cannot
        reproduce means the seam drifted — the builder must refuse."""
        rows = self.rows.copy()
        rows.loc[rows.index[0], ["band", "omega", "edge", "edge_ci_lo"]] = \
            ["strong", 0.5, -1.0, -1.0]
        with self.assertRaises(ValueError):
            db.build_gate_registered_artifact(rows, ticker="TST", stride=5,
                                              burn_in_years=1)

    def test_selection_prefers_first_validated(self):
        specs = ([("neutral", 2.0, 1.0, 0.4, 0.3)] * 20
                 + [("weak", 0.5, -1.0, -2.0, 0.9)] * 20)
        art = db.build_gate_registered_artifact(_gate_rows(specs),
                                                ticker="TST", stride=5,
                                                burn_in_years=1)
        self.assertEqual(art["selected"], "rr0")
        by_id = {c["id"]: c for c in art["candidates"]}
        self.assertEqual(by_id["rr0"]["primary"]["outcome"], "validated")
        self.assertEqual(by_id["rr50"]["primary"]["outcome"], "inconclusive")
        self.assertEqual(by_id["pt67"]["primary"]["outcome"], "inconclusive")

    def test_selection_falls_through_to_pt67(self):
        specs = ([("neutral", 3.0, 2.0, -0.5, 0.9)] * 20
                 + [("weak", 0.5, -1.0, -2.0, 0.9)] * 20)
        art = db.build_gate_registered_artifact(_gate_rows(specs),
                                                ticker="TST", stride=5,
                                                burn_in_years=1)
        self.assertEqual(art["selected"], "pt67")
        by_id = {c["id"]: c for c in art["candidates"]}
        self.assertEqual(by_id["rr0"]["primary"]["outcome"], "inconclusive")
        self.assertEqual(by_id["rr50"]["primary"]["outcome"], "inconclusive")

    def test_selection_none_when_nothing_validates(self):
        specs = [("weak", 0.5, -1.0, -2.0, 0.9)] * 30
        art = db.build_gate_registered_artifact(_gate_rows(specs),
                                                ticker="TST", stride=5,
                                                burn_in_years=1)
        self.assertIsNone(art["selected"])
        for c in art["candidates"]:
            self.assertEqual(c["primary"]["outcome"], "inconclusive")

    def test_cli_write_and_spy_guard(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(FIXTURE_DIR / "dip_history.csv", td)
            orig = db.GATE_REGISTERED_PATH
            db.GATE_REGISTERED_PATH = Path(td) / "gate.json"
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = db.main(["--ticker", "SPY", "--data-dir", td,
                                  "--burn-in-years", "1", "--stride", "5",
                                  "--write-gate-registered"])
                self.assertEqual(rc, 0)
                self.assertIn("strong-gate experiment", buf.getvalue())
                art = db.load_gate_registered_artifact(db.GATE_REGISTERED_PATH)
                self.assertIsNotNone(art)
                self.assertEqual(art["ticker"], "SPY")
                b = db.GATE_REGISTERED_PATH.read_bytes()
                self.assertNotIn(b"\r\n", b)          # LF only
                self.assertTrue(b.endswith(b"\n"))
                self.assertTrue(all(x < 128 for x in b))   # pure ASCII
                # non-SPY: exit 1 with the guard's reason, nothing written
                db.GATE_REGISTERED_PATH = Path(td) / "gate2.json"
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(err):
                    rc2 = db.main(["--ticker", "SCHD", "--data-dir", td,
                                   "--burn-in-years", "1", "--stride", "5",
                                   "--write-gate-registered"])
                self.assertEqual(rc2, 1)
                self.assertIn("SPY-only", err.getvalue())
                self.assertFalse(db.GATE_REGISTERED_PATH.exists())
            finally:
                db.GATE_REGISTERED_PATH = orig

    def test_loader_round_trip_missing_corrupt_drift(self):
        art = db.build_gate_registered_artifact(self.rows, ticker="TST",
                                                stride=5, burn_in_years=1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gate.json"
            p.write_bytes((json.dumps(art, indent=2) + "\n").encode("ascii"))
            self.assertEqual(db.load_gate_registered_artifact(p), art)
            self.assertIsNone(
                db.load_gate_registered_artifact(Path(td) / "absent.json"))
            p.write_text("{not json", encoding="ascii")
            self.assertIsNone(db.load_gate_registered_artifact(p))
            bad = dict(art)
            bad["kind"] = "ladder"
            p.write_text(json.dumps(bad), encoding="ascii")
            self.assertIsNone(db.load_gate_registered_artifact(p))
            bad = dict(art)
            del bad["candidates"]
            p.write_text(json.dumps(bad), encoding="ascii")
            self.assertIsNone(db.load_gate_registered_artifact(p))
            bad = dict(art)
            bad["candidates"] = {"rr0": {}}   # json-valid, wrong type
            p.write_text(json.dumps(bad), encoding="ascii")
            self.assertIsNone(db.load_gate_registered_artifact(p))


class CommittedArtifactConformanceTests(unittest.TestCase):
    """Regen tripwire (spec §8): the COMMITTED artifact must stay the
    registered configuration with the registered outcome. A regen that drifts
    ticker/config/outcome fails here and forces the change-control
    conversation instead of shipping silently."""

    @classmethod
    def setUpClass(cls):
        cls.art = db.load_registered_artifact()

    def test_committed_artifact_is_the_registration(self):
        self.assertIsNotNone(
            self.art, "parsers/dip_backtest_registered.json missing/invalid — "
            "regenerate deliberately: py parsers/dip_backtest.py --ticker SPY "
            "--write-registered (spec 2026-07-16 §9)")
        self.assertEqual(self.art["ticker"], "SPY")
        self.assertEqual(self.art["registered"], "2026-07-14")
        self.assertEqual(self.art["config"]["stride"], 5)
        self.assertEqual(self.art["config"]["burn_in_years"], 10)
        self.assertEqual(self.art["primary"]["outcome"], "validated")
        self.assertGreaterEqual(self.art["primary"]["n_edge_episodes"], 5)


class LadderArtifactConformanceTests(unittest.TestCase):
    """Regen tripwire for the LADDER registration (spec 2026-07-18): the
    committed artifact must stay the registered configuration with the
    registered outcome — including the HONEST negative. The 2026-07-18 SPY
    run: final wealth beat the matched constant mix (5.77 vs 3.78) but the
    pre-registered Omega primary did NOT clear (stat −6.8, CI −17.7…+119.7)
    → not_supported → the UI card does not ship. A regen that flips any of
    this fails here and forces the change-control conversation."""

    @classmethod
    def setUpClass(cls):
        cls.art = db.load_ladder_registered_artifact()

    def test_committed_ladder_artifact_is_the_registration(self):
        self.assertIsNotNone(
            self.art, "parsers/dip_ladder_registered.json missing/invalid — "
            "regenerate deliberately: py parsers/dip_backtest.py --ticker "
            "SPY --write-ladder-registered (spec 2026-07-18)")
        self.assertEqual(self.art["ticker"], "SPY")
        self.assertEqual(self.art["registered"], "2026-07-18")
        cfg = self.art["config"]
        self.assertEqual(cfg["stride"], 5)
        self.assertEqual(cfg["burn_in_years"], 10)
        self.assertEqual(cfg["horizon"], 252)
        self.assertEqual(cfg["cash_leg"], "ff_rf_daily")
        self.assertEqual(cfg["exit"], "recovery_to_anchor_peak")
        self.assertEqual(self.art["fractions"],
                         {"neutral": 0.25, "strong": 0.50, "tk_rule": 1.0})
        self.assertEqual(self.art["evals"]["n"], 1128)
        self.assertEqual(self.art["ladder"]["n_tranches"], 9)
        self.assertEqual(self.art["primary"]["outcome"], "not_supported")


class LadderV2MetricTests(unittest.TestCase):
    """V2 ladder primary (spec 2026-07-18 v2 metric): terminal-wealth
    log-ratio with a stationary-block CI on the paired daily log-diff
    series — every day contributes finitely, no inf−inf."""

    def _wealth(self, rets):
        s = pd.Series(rets, index=pd.bdate_range("2020-01-01",
                                                 periods=len(rets)))
        return (1.0 + s).cumprod()

    def test_stat_telescopes_to_final_log_ratio(self):
        lw = self._wealth([0.01, 0.00, 0.02, -0.01])
        bw = self._wealth([0.00, 0.01, 0.00, 0.01])
        m = db.ladder_wealth_metric(lw, bw, n_tranches=3)
        self.assertAlmostEqual(m["stat"],
                               math.log(lw.iloc[-1] / bw.iloc[-1]),
                               places=12)
        self.assertEqual(m["n_days"], 4)
        self.assertEqual(m["block_days"], db.LADDER_V2_BLOCK_DAYS)

    def test_day_zero_measured_from_capital_start(self):
        m = db.ladder_wealth_metric(self._wealth([0.05]),
                                    self._wealth([0.01]), n_tranches=3)
        self.assertAlmostEqual(m["stat"],
                               math.log(1.05) - math.log(1.01), places=12)

    def test_validated_on_dominating_tape(self):
        n = 600
        m = db.ladder_wealth_metric(self._wealth([0.002] * n),
                                    self._wealth([0.001] * n), n_tranches=3)
        self.assertGreater(m["ci_lo"], 0.0)
        self.assertEqual(m["outcome"], "validated")

    def test_tranche_gate_overrides_ci(self):
        n = 600
        m = db.ladder_wealth_metric(self._wealth([0.002] * n),
                                    self._wealth([0.001] * n), n_tranches=2)
        self.assertEqual(m["outcome"], "inconclusive")

    def test_identical_series_not_supported(self):
        lw = self._wealth([0.001] * 400)
        m = db.ladder_wealth_metric(lw, lw.copy(), n_tranches=3)
        self.assertAlmostEqual(m["stat"], 0.0, places=12)
        self.assertAlmostEqual(m["ci_lo"], 0.0, places=12)
        self.assertEqual(m["outcome"], "not_supported")

    def test_block_length_changes_the_ci(self):
        lw = self._wealth([0.004] * 150 + [0.000] * 150)
        bw = self._wealth([0.000] * 150 + [0.003] * 150)
        a = db.ladder_wealth_metric(lw, bw, n_tranches=3, block_days=21)
        c = db.ladder_wealth_metric(lw, bw, n_tranches=3, block_days=252)
        self.assertNotAlmostEqual(a["ci_lo"], c["ci_lo"], places=6)

    def test_deterministic(self):
        lw = self._wealth([0.002, -0.001] * 100)
        bw = self._wealth([0.001, 0.0005] * 100)
        self.assertEqual(db.ladder_wealth_metric(lw, bw, n_tranches=3),
                         db.ladder_wealth_metric(lw, bw, n_tranches=3))


class LadderDeployedOmegaTests(unittest.TestCase):
    """Deployed-only Omega descriptive (spec 2026-07-18 v2 §4): eval-day
    deployment mask from the tranche log (entry day deployed, exit day
    NOT, never-recovered deploys through the tail) + fwd-252 wealth
    Omega difference restricted to deployed rows. NON-gating."""

    def test_mask_boundaries(self):
        idx = pd.bdate_range("2021-01-01", periods=6)
        tranches = [{"entry_date": idx[1], "exit_date": idx[3]}]
        got = db.ladder_deployed_mask(pd.Series(idx), tranches)
        self.assertEqual(list(got), [False, True, True, False, False, False])

    def test_mask_open_tranche_deploys_through_tail(self):
        idx = pd.bdate_range("2021-01-01", periods=6)
        tranches = [{"entry_date": idx[1], "exit_date": idx[2]},
                    {"entry_date": idx[4], "exit_date": None}]
        got = db.ladder_deployed_mask(pd.Series(idx), tranches)
        self.assertEqual(list(got), [False, True, False, False, True, True])

    def _wealth_with_points(self, n, points):
        idx = pd.bdate_range("2020-01-01", periods=n)
        vals = np.ones(n)
        for i, v in points.items():
            vals[i] = v
        return pd.Series(vals, index=idx)

    def test_restriction_censor_and_stat(self):
        n, H = 600, 252
        lw = self._wealth_with_points(n, {50 + H: 0.95, 60 + H: 1.10})
        bw = self._wealth_with_points(n, {50 + H: 0.99, 60 + H: 1.04})
        idx = lw.index
        rows = pd.DataFrame({"date": [idx[50], idx[60], idx[70], idx[400]]})
        # one tranche covers rows 50/60; a second covers the censored 400
        tranches = [{"entry_date": idx[45], "exit_date": idx[65]},
                    {"entry_date": idx[395], "exit_date": None}]
        res = db.ladder_deployed_omega(rows, lw, bw, tranches, stride=5)
        # row 70 undeployed; row 400 deployed but 400+252 >= 600 -> censored
        self.assertEqual(res["n_rows"], 2)
        self.assertEqual(res["block_rows"], 51)
        # fl = [-5%, +10%] -> omega 2.0; fb = [-1%, +4%] -> omega 4.0
        self.assertAlmostEqual(res["stat"], 2.0 - 4.0, places=12)

    def test_deterministic(self):
        n, H = 600, 252
        lw = self._wealth_with_points(n, {50 + H: 0.95, 60 + H: 1.10})
        bw = self._wealth_with_points(n, {50 + H: 0.99, 60 + H: 1.04})
        idx = lw.index
        rows = pd.DataFrame({"date": [idx[50], idx[60]]})
        tranches = [{"entry_date": idx[45], "exit_date": None}]
        self.assertEqual(
            db.ladder_deployed_omega(rows, lw, bw, tranches, stride=5),
            db.ladder_deployed_omega(rows, lw, bw, tranches, stride=5))


class GateArtifactConformanceTests(unittest.TestCase):
    """Regen tripwire for the GATE experiment (spec 2026-07-18): the
    committed artifact must stay the registered configuration with the
    registered outcome — the HONEST negative. The 2026-07-18 SPY run:
    loosening the rank clause widened strong 74→104 days (still all-win,
    hit 100%) but only 2→3 episodes; pt67's CI cleared on 2 episodes (the
    overlapping-window illusion the episode gate exists to catch). No
    candidate reached the pre-registered 5-episode adoption gate →
    selected=null → the incumbent double gate stays shipped. A regen that
    flips any of this forces the change-control conversation."""

    @classmethod
    def setUpClass(cls):
        cls.art = db.load_gate_registered_artifact()

    def test_committed_gate_artifact_is_the_registration(self):
        self.assertIsNotNone(
            self.art, "parsers/dip_gate_registered.json missing/invalid — "
            "regenerate deliberately: py parsers/dip_backtest.py --ticker "
            "SPY --write-gate-registered (spec 2026-07-18)")
        self.assertEqual(self.art["ticker"], "SPY")
        self.assertEqual(self.art["registered"], "2026-07-18")
        cfg = self.art["config"]
        self.assertEqual(cfg["stride"], 5)
        self.assertEqual(cfg["burn_in_years"], 10)
        self.assertEqual(cfg["incumbent"],
                         {"strong_rr": 0.67, "requires_ci": True})
        self.assertEqual(cfg["min_episodes"], 5)
        self.assertEqual(self.art["evals"]["n"], 1128)
        prim = {c["id"]: c["primary"] for c in self.art["candidates"]}
        self.assertEqual({k: v["outcome"] for k, v in prim.items()},
                         {"rr0": "inconclusive", "rr50": "inconclusive",
                          "pt67": "inconclusive"})
        self.assertEqual({k: v["n_strong_episodes"] for k, v in prim.items()},
                         {"rr0": 3, "rr50": 3, "pt67": 2})
        self.assertEqual({k: v["n_strong_days"] for k, v in prim.items()},
                         {"rr0": 104, "rr50": 104, "pt67": 75})
        self.assertIsNone(self.art["selected"])


class LadderHarnessTests(unittest.TestCase):
    """S2 (spec 2026-07-18): ladder replay over eval rows + the
    pre-registered primary vs a constant-mix baseline."""

    def _market(self):
        # Two clean dips inside a long flat-peak tape, both recovered, with
        # >252d of frontier after the last eval so fwd windows exist.
        vals = ([100.0] * 120 + [90.0] * 80 + [100.0] * 120
                + [90.0] * 80 + [100.0] * 300)
        idx = pd.bdate_range("2019-06-03", periods=len(vals))
        price = pd.Series(vals, index=idx)
        return price, price.copy()

    def _rows_at(self, price, entries):
        """entries: (day_index, band, depth_pctile) — only the columns
        ladder_backtest reads, mirroring walk_forward's shape."""
        n = len(entries)
        return pd.DataFrame({
            "date": [price.index[i] for i, _, _ in entries],
            "band": [b for _, b, _ in entries],
            "depth_pctile": [float(d) for _, _, d in entries],
            "omega": [1.0] * n, "edge": [0.0] * n, "rr_pct": [0.5] * n,
            "n_cond": [30] * n, "episode_id": [float(k) for k in range(n)],
            "fwd_252": [0.0] * n,
        })

    def test_ladder_evals_mask(self):
        price, _ = self._market()
        rows = self._rows_at(price, [(130, "neutral", 90.0),
                                     (135, "neutral", 50.0),
                                     (140, "weak", 95.0)])
        ev = db.ladder_evals(rows)
        self.assertEqual(list(ev["tk_rule"]), [True, False, False])
        self.assertEqual(list(ev["band"]), ["neutral", "neutral", "weak"])

    def test_backtest_keys_and_determinism(self):
        price, tr = self._market()
        zero = pd.Series(0.0, index=price.index)
        rows = self._rows_at(price, [(130, "neutral", 60.0),
                                     (340, "neutral", 60.0)])
        a = db.ladder_backtest(rows, price, tr, zero, n_boot=64)
        b = db.ladder_backtest(rows, price, tr, zero, n_boot=64)
        for k in ("stat", "ci_lo", "ci_hi", "n_evals", "n_tranches",
                  "skipped_deploys", "avg_equity_exposure", "final_wealth",
                  "baseline_final_wealth", "block_rows", "outcome",
                  "tranches"):
            self.assertIn(k, a)
        # NaN-safe exact equality: on this all-positive tape both Omegas are
        # inf -> stat is NaN by the referee's own semantics; determinism is
        # the property under test.
        np.testing.assert_equal(a["stat"], b["stat"])
        np.testing.assert_equal(a["ci_lo"], b["ci_lo"])
        self.assertEqual(a["n_tranches"], 2)
        self.assertGreater(a["avg_equity_exposure"], 0.0)
        self.assertLess(a["avg_equity_exposure"], 1.0)
        # Both dips recover +11.1% while the baseline holds a constant mix —
        # the ladder's final wealth beats it on this hand-built tape.
        self.assertGreater(a["final_wealth"], 1.0)

    def test_outcome_inconclusive_below_min_tranches(self):
        price, tr = self._market()
        zero = pd.Series(0.0, index=price.index)
        rows = self._rows_at(price, [(130, "neutral", 60.0)])
        res = db.ladder_backtest(rows, price, tr, zero, n_boot=32)
        self.assertEqual(res["n_tranches"], 1)
        self.assertEqual(res["outcome"], "inconclusive")

    def test_artifact_roundtrip_and_loader_never_raises(self):
        price, tr = self._market()
        zero = pd.Series(0.0, index=price.index)
        rows = self._rows_at(price, [(130, "neutral", 60.0),
                                     (340, "strong", 90.0)])
        art = db.build_ladder_registered_artifact(
            rows, price, tr, zero, ticker="spy", stride=5, burn_in_years=10)
        self.assertEqual(art["schema"], 1)
        self.assertEqual(art["kind"], "ladder")
        self.assertEqual(art["ticker"], "SPY")
        self.assertEqual(art["fractions"],
                         {"neutral": 0.25, "strong": 0.50, "tk_rule": 1.0})
        self.assertIn(art["primary"]["outcome"],
                      ("validated", "not_supported", "inconclusive"))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "art.json"
            p.write_bytes((json.dumps(art, indent=2) + "\n").encode("ascii"))
            back = db.load_ladder_registered_artifact(p)
            self.assertEqual(back, json.loads(p.read_text()))
            # never raises: missing / corrupt / wrong kind
            self.assertIsNone(
                db.load_ladder_registered_artifact(Path(td) / "nope.json"))
            bad = Path(td) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertIsNone(db.load_ladder_registered_artifact(bad))
            wrong = Path(td) / "wrong.json"
            wrong.write_text(json.dumps({"schema": 1, "kind": "verdict"}),
                             encoding="utf-8")
            self.assertIsNone(db.load_ladder_registered_artifact(wrong))


class LadderV2ArtifactTests(unittest.TestCase):
    """build_ladder_v2_registered_artifact + CLI + loader (spec 2026-07-18
    v2): schema, the spoilage disclosure, sensitivity blocks, telescoping
    consistency, json-safety, determinism; SPY-only CLI guard; loader
    never raises. v1 plumbing untouched by construction."""

    def _market(self):
        vals = ([100.0] * 120 + [90.0] * 80 + [100.0] * 120
                + [90.0] * 80 + [100.0] * 300)
        idx = pd.bdate_range("2019-06-03", periods=len(vals))
        price = pd.Series(vals, index=idx)
        return price, price.copy()

    def _rows_at(self, price, entries):
        n = len(entries)
        return pd.DataFrame({
            "date": [price.index[i] for i, _, _ in entries],
            "band": [b for _, b, _ in entries],
            "depth_pctile": [float(d) for _, _, d in entries],
            "omega": [1.0] * n, "edge": [0.0] * n, "rr_pct": [0.5] * n,
            "n_cond": [30] * n, "episode_id": [float(k) for k in range(n)],
            "fwd_252": [0.0] * n,
        })

    def _artifact(self):
        price, tr = self._market()
        zero = pd.Series(0.0, index=price.index)
        rows = self._rows_at(price, [(130, "neutral", 60.0),
                                     (340, "strong", 90.0)])
        return db.build_ladder_v2_registered_artifact(
            rows, price, tr, zero, ticker="spy", stride=5, burn_in_years=10)

    def test_schema_disclosure_and_determinism(self):
        a = self._artifact()
        b = self._artifact()
        self.assertEqual(a, b)
        self.assertEqual(a["schema"], 1)
        self.assertEqual(a["kind"], "ladder_v2")
        self.assertEqual(a["ticker"], "SPY")
        self.assertEqual(a["registered"], db.LADDER_V2_REGISTERED_DATE)
        cfg = a["config"]
        self.assertEqual(cfg["gate"], {"strong_rr": da.VERDICT_STRONG_RR,
                                       "requires_ci": True})
        self.assertEqual(cfg["metric"],
                         {"primary": "terminal_wealth_log_ratio",
                          "block_days": db.LADDER_V2_BLOCK_DAYS})
        self.assertEqual(cfg["cash_leg"], "ff_rf_daily")
        self.assertEqual(cfg["min_tranches"], db.LADDER_MIN_TRANCHES)
        self.assertEqual(a["fractions"],
                         {"neutral": 0.25, "strong": 0.50, "tk_rule": 1.0})
        self.assertIn("5.77", a["prior_observation"])
        self.assertEqual(set(a["sensitivity"]),
                         {"block_21", "block_63", "block_252"})
        self.assertEqual(a["primary"]["block_days"],
                         db.LADDER_V2_BLOCK_DAYS)
        self.assertIn(a["primary"]["outcome"],
                      ("validated", "not_supported", "inconclusive"))
        self.assertIn("n_rows", a["descriptive"]["deployed_omega"])
        json.dumps(a, allow_nan=False)

    def test_primary_telescopes_to_final_wealth_ratio(self):
        a = self._artifact()
        self.assertAlmostEqual(
            a["primary"]["stat"],
            math.log(a["ladder"]["final_wealth"]
                     / a["ladder"]["baseline_final_wealth"]),
            places=10)

    def test_cli_write_and_spy_guard(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(FIXTURE_DIR / "dip_history.csv", td)
            hist = pd.read_csv(FIXTURE_DIR / "dip_history.csv",
                               parse_dates=["date"])
            dates = hist["date"].drop_duplicates().sort_values()
            pd.DataFrame({"date": dates, "rf": 0.0001}).to_csv(
                Path(td) / "ff_factors_daily.csv", index=False)
            orig = db.LADDER_V2_REGISTERED_PATH
            db.LADDER_V2_REGISTERED_PATH = Path(td) / "v2.json"
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = db.main(["--ticker", "SPY", "--data-dir", td,
                                  "--burn-in-years", "1", "--stride", "5",
                                  "--write-ladder-v2-registered"])
                self.assertEqual(rc, 0)
                self.assertIn("ladder v2", buf.getvalue())
                art = db.load_ladder_v2_registered_artifact(
                    db.LADDER_V2_REGISTERED_PATH)
                self.assertIsNotNone(art)
                self.assertEqual(art["ticker"], "SPY")
                b = db.LADDER_V2_REGISTERED_PATH.read_bytes()
                self.assertNotIn(b"\r\n", b)
                self.assertTrue(b.endswith(b"\n"))
                self.assertTrue(all(x < 128 for x in b))
                db.LADDER_V2_REGISTERED_PATH = Path(td) / "v2b.json"
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(err):
                    rc2 = db.main(["--ticker", "SCHD", "--data-dir", td,
                                   "--burn-in-years", "1", "--stride", "5",
                                   "--write-ladder-v2-registered"])
                self.assertEqual(rc2, 1)
                self.assertIn("SPY-only", err.getvalue())
                self.assertFalse(db.LADDER_V2_REGISTERED_PATH.exists())
            finally:
                db.LADDER_V2_REGISTERED_PATH = orig

    def test_loader_never_raises(self):
        art = self._artifact()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "v2.json"
            p.write_bytes((json.dumps(art, indent=2) + "\n").encode("ascii"))
            self.assertEqual(db.load_ladder_v2_registered_artifact(p),
                             json.loads(p.read_text()))
            self.assertIsNone(db.load_ladder_v2_registered_artifact(
                Path(td) / "nope.json"))
            bad = Path(td) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertIsNone(db.load_ladder_v2_registered_artifact(bad))
            v1kind = dict(art)
            v1kind["kind"] = "ladder"      # a v1 file must NOT load as v2
            bad.write_text(json.dumps(v1kind), encoding="utf-8")
            self.assertIsNone(db.load_ladder_v2_registered_artifact(bad))
            drift = dict(art)
            del drift["prior_observation"]
            bad.write_text(json.dumps(drift), encoding="utf-8")
            self.assertIsNone(db.load_ladder_v2_registered_artifact(bad))


class LadderV2ArtifactConformanceTests(unittest.TestCase):
    """Regen tripwire for the LADDER V2 registration (spec 2026-07-18 v2):
    the committed artifact must stay the registered configuration with
    the registered outcome — the SECOND honest ladder negative. The
    2026-07-19 SPY run: stat +0.4233 (= ln(5.77/3.78), telescoping) but
    90% CI [−0.269, +1.218] at block 126 — and the non-gating sensitivity
    rows (21/63/252d) straddle zero near-identically, so the width is not
    a block-length artifact: ~9 tranches in 33y cannot be separated from
    timing luck at 90% confidence. The deployed-only Omega descriptive
    degenerated to inf−inf over its 390 deployed rows (both sides nearly
    all-win from dip levels) — the metric not picked would have been
    blind here too. The S3 card does NOT ship. A regen that flips any of
    this forces the change-control conversation."""

    @classmethod
    def setUpClass(cls):
        cls.art = db.load_ladder_v2_registered_artifact()

    def test_committed_v2_artifact_is_the_registration(self):
        self.assertIsNotNone(
            self.art, "parsers/dip_ladder_v2_registered.json missing/"
            "invalid — regenerate deliberately: py parsers/dip_backtest.py "
            "--ticker SPY --write-ladder-v2-registered (spec 2026-07-18 v2)")
        self.assertEqual(self.art["ticker"], "SPY")
        self.assertEqual(self.art["registered"], "2026-07-19")
        cfg = self.art["config"]
        self.assertEqual(cfg["stride"], 5)
        self.assertEqual(cfg["burn_in_years"], 10)
        self.assertEqual(cfg["cash_leg"], "ff_rf_daily")
        self.assertEqual(cfg["exit"], "recovery_to_anchor_peak")
        self.assertEqual(cfg["gate"], {"strong_rr": 0.67,
                                       "requires_ci": True})
        self.assertEqual(cfg["metric"],
                         {"primary": "terminal_wealth_log_ratio",
                          "block_days": 126})
        self.assertEqual(self.art["fractions"],
                         {"neutral": 0.25, "strong": 0.50, "tk_rule": 1.0})
        self.assertEqual(self.art["evals"]["n"], 1128)
        self.assertEqual(self.art["ladder"]["n_tranches"], 9)
        self.assertIn("5.77", self.art["prior_observation"])
        self.assertEqual(self.art["primary"]["outcome"], "not_supported")
        self.assertEqual(self.art["primary"]["n_days"], 8407)
        self.assertEqual(set(self.art["sensitivity"]),
                         {"block_21", "block_63", "block_252"})
        self.assertEqual(
            self.art["descriptive"]["deployed_omega"]["n_rows"], 390)


from parsers import dip_extend as de


_SPX_FIXTURE_CACHE: dict = {}


def _mk_spx_rows(n_days=2500, seed_dips=((700, 80), (1600, 90))):
    """Walk-forward rows over a synthetic extended pair: gspc whole-span,
    sptr from 35%, spy from 60% (the _mk_components geometry), two dips so
    bands vary. Returns (rows, rows5, built).

    MEMOIZED per (n_days, seed_dips) — this is the most expensive fixture in
    the suite and it is deterministic, so rebuilding it per test bought
    nothing. Two `walk_forward` passes here, and each
    `build_spx_registered_artifact` on top re-runs the referee for the replay,
    the 5y sensitivity and every gate candidate.

    The frames are returned as COPIES so callers keep the same isolation they
    had when every call rebuilt from scratch — a cached frame handed out by
    reference would let one test's edit leak into the next. The copy is
    microseconds against minutes of compute. `built` is read-only by
    convention (tests only read `built["meta"]`).
    """
    key = (n_days, tuple(seed_dips))
    if key not in _SPX_FIXTURE_CACHE:
        _SPX_FIXTURE_CACHE[key] = _build_spx_rows(n_days, seed_dips)
    rows, rows5, built = _SPX_FIXTURE_CACHE[key]
    return rows.copy(), rows5.copy(), built


def _build_spx_rows(n_days, seed_dips):
    idx = pd.bdate_range("1990-01-02", periods=n_days)
    n = len(idx)
    ret = np.where(np.arange(n) % 2 == 0, 0.0012, -0.0006).copy()
    for at, ln in seed_dips:
        ret[at:at + ln] = -0.004
    gspc = pd.Series(100.0 * np.cumprod(1.0 + ret), index=idx, name="^GSPC")
    spy_start, tr_start = idx[int(n * 0.6)], idx[int(n * 0.35)]
    spy_close = (gspc[gspc.index >= spy_start] * 0.1).rename("SPY")
    spy_adj = (spy_close * 0.8).rename("SPY")
    sptr = (gspc[gspc.index >= tr_start] * 1.7).rename("^SP500TR")
    months = sorted(set(idx.strftime("%Y-%m")))
    yields = pd.Series(0.03, index=pd.Index(months, name="month"))
    built = de.build_extended_spx(spy_close, spy_adj, gspc, sptr, yields)
    rows = db.walk_forward(built["price"], built["tr"], stride=5,
                           burn_in_years=3)
    rows5 = db.walk_forward(built["price"], built["tr"], stride=5,
                            burn_in_years=2)
    return rows, rows5, built


_SPX_GATES = {"corr_gspc_vs_spy": {"corr": 1.0, "n_overlap": 999, "ok": True},
              "corr_sptr_vs_spy": {"corr": 1.0, "n_overlap": 999, "ok": True},
              "tracking": {"ann_diff_bps": 1.0, "p95_roll252_bps": 10.0,
                           "n_overlap": 999, "ok": True},
              "all_ok": True}

_SPX_ARTIFACT_CACHE: list = []


def _spx_artifact():
    """The reference SPX artifact over the default fixture, built ONCE.

    Shared by the builder tests and the loader tests: the loader only needs a
    structurally valid artifact to round-trip and corrupt, so giving it its own
    series bought a second full build of the suite's most expensive fixture.
    Treat as read-only -- the tests that mutate it already `dict(art)` first.
    """
    if not _SPX_ARTIFACT_CACHE:
        rows, rows5, built = _mk_spx_rows()
        _SPX_ARTIFACT_CACHE.append(db.build_spx_registered_artifact(
            rows, rows5, stride=5, burn_in_years=3,
            series_meta=built["meta"], data_gates=_SPX_GATES))
    return _SPX_ARTIFACT_CACHE[0]


class SpxArtifactBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.rows5, cls.built = _mk_spx_rows()
        cls.gates = _SPX_GATES
        cls.art = _spx_artifact()

    def test_schema_and_identity(self):
        a = self.art
        self.assertEqual(a["schema"], 1)
        self.assertEqual(a["kind"], "spx_extension")
        self.assertEqual(a["ticker"], "SPX")
        self.assertEqual(a["registered"], db.SPX_REGISTERED_DATE)
        self.assertEqual(a["consequence"], db.SPX_CONSEQUENCE)
        self.assertEqual(a["config"]["incumbent_window"],
                         list(db.INCUMBENT_WINDOW))
        self.assertEqual(a["series"]["meta"], self.built["meta"])
        self.assertTrue(a["series"]["data_gates"]["all_ok"])

    def test_replay_block(self):
        r = self.art["replay"]
        self.assertEqual(
            set(r["referee"]), set(db.BAND_ORDER) | {"all"})
        self.assertIn(r["primary"]["outcome"],
                      ("validated", "not_supported", "inconclusive"))
        # synthetic span predates 2003 -> incumbent-window slice is empty
        self.assertIsNone(r["incumbent_window"])

    def test_sensitivity_block(self):
        s = self.art["sensitivity"]["burn_in_5y"]
        self.assertEqual(s["evals"]["n"], len(self.rows5))
        self.assertIn("outcome", s["primary"])

    def test_gate_block_matches_gate_machinery(self):
        g = self.art["gate"]
        self.assertEqual([c["id"] for c in g["candidates"]],
                         [c["id"] for c in db.GATE_CANDIDATES])
        self.assertEqual(g["selection_rule"], db.GATE_SELECTION_RULE)
        for c in g["candidates"]:
            self.assertIn(c["primary"]["outcome"],
                          ("validated", "not_supported", "inconclusive"))

    def test_incumbent_identity_guard(self):
        bad = self.rows.copy()
        # Force an impossible stored-strong row (rank below any threshold,
        # CI not cleared, omega <= 1): the incumbent relabel cannot
        # reproduce it, so the builder must refuse to build.
        i = bad.index[0]
        bad.loc[i, ["band", "omega", "edge", "edge_ci_lo", "rr_pct"]] = \
            ["strong", 0.5, -1.0, -1.0, -1.0]
        with self.assertRaises(ValueError):
            db.build_spx_registered_artifact(
                bad, self.rows5, stride=5, burn_in_years=3,
                series_meta=self.built["meta"], data_gates=self.gates)

    def test_json_safe_and_deterministic(self):
        s1 = json.dumps(self.art, indent=2)
        art2 = db.build_spx_registered_artifact(
            self.rows, self.rows5, stride=5, burn_in_years=3,
            series_meta=self.built["meta"], data_gates=self.gates)
        self.assertEqual(s1, json.dumps(art2, indent=2))

    def test_incumbent_window_populated_when_rows_overlap(self):
        rows_iw = self.rows.copy()
        shift = pd.Timestamp("2004-01-01") - rows_iw["date"].min()
        rows_iw["date"] = rows_iw["date"] + shift
        art = db.build_spx_registered_artifact(
            rows_iw, self.rows5, stride=5, burn_in_years=3,
            series_meta=self.built["meta"], data_gates=self.gates)
        self.assertIsNotNone(art["replay"]["incumbent_window"])


class SpxLoaderTests(unittest.TestCase):
    """Loader round-trip + rejection. These assert on the artifact's SHAPE, not
    on any property of the series behind it, so they share the module-level
    reference artifact instead of building a second one (this class was 224s
    for 2 tests before that change)."""

    def _valid(self):
        return _spx_artifact()

    def test_round_trip(self):
        art = self._valid()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.json"
            p.write_text(json.dumps(art), encoding="utf-8")
            self.assertEqual(db.load_spx_registered_artifact(p), art)

    def test_missing_corrupt_wrong_kind_missing_key(self):
        art = self._valid()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                db.load_spx_registered_artifact(Path(tmp) / "no.json"))
            p = Path(tmp) / "a.json"
            p.write_text("{not json", encoding="utf-8")
            self.assertIsNone(db.load_spx_registered_artifact(p))
            bad = dict(art); bad["kind"] = "gate"
            p.write_text(json.dumps(bad), encoding="utf-8")
            self.assertIsNone(db.load_spx_registered_artifact(p))
            bad = dict(art); del bad["consequence"]
            p.write_text(json.dumps(bad), encoding="utf-8")
            self.assertIsNone(db.load_spx_registered_artifact(p))
            bad = dict(art)
            bad["gate"] = {**art["gate"],
                           "candidates": art["gate"]["candidates"][:2]}
            p.write_text(json.dumps(bad), encoding="utf-8")
            self.assertIsNone(db.load_spx_registered_artifact(p))


class SpxCliTests(unittest.TestCase):
    def _datadir(self, tmp):
        # reuse the extend-test writer shapes inline (no cross-file import).
        # 5y span = the shortest that still clears the happy path's 3y
        # burn-in + 252d censor with a real stride-5 eval window (45 evals).
        # Only that test reads the data; the three refusal tests exit on an
        # argument check before any load.
        idx = pd.bdate_range("2000-01-03", periods=252 * 5)
        n = len(idx)
        ret = np.where(np.arange(n) % 2 == 0, 0.001, -0.0005)
        gspc = pd.Series(100.0 * np.cumprod(1.0 + ret), index=idx)
        spy_start, tr_start = idx[int(n * 0.6)], idx[int(n * 0.35)]
        spy_close = gspc[gspc.index >= spy_start] * 0.1
        spy_adj = spy_close * 0.8
        sptr = gspc[gspc.index >= tr_start] * 1.7
        pd.concat([
            pd.DataFrame({"symbol": "SPY", "date": spy_close.index,
                          "close": spy_close.values,
                          "adj_close": spy_adj.values}),
        ], ignore_index=True).to_csv(Path(tmp) / "dip_history.csv",
                                     index=False)
        pd.concat([
            pd.DataFrame({"symbol": "^GSPC", "date": gspc.index,
                          "close": gspc.values, "adj_close": gspc.values}),
            pd.DataFrame({"symbol": "^SP500TR", "date": sptr.index,
                          "close": sptr.values, "adj_close": sptr.values}),
        ], ignore_index=True).to_csv(Path(tmp) / "dip_index_history.csv",
                                     index=False)

    def test_extended_requires_ticker_spx(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._datadir(tmp)
            rc = db.main(["--ticker", "SPY", "--extended-spx",
                          "--data-dir", tmp])
        self.assertEqual(rc, 1)

    def test_write_spx_requires_extended_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._datadir(tmp)
            rc = db.main(["--ticker", "SPY", "--write-spx-registered",
                          "--data-dir", tmp])
        self.assertEqual(rc, 1)

    def test_extended_happy_path_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._datadir(tmp)
            rc = db.main(["--ticker", "SPX", "--extended-spx",
                          "--burn-in-years", "3", "--data-dir", tmp])
        self.assertEqual(rc, 0)

    def test_write_registered_refuses_extended(self):
        """--write-registered must refuse --extended-spx: an SPX extended run
        must never overwrite the SPY incumbent verdict artifact (spec
        2026-07-16 + 2026-07-19 §4d)."""
        with tempfile.TemporaryDirectory() as tmp:
            self._datadir(tmp)
            orig = db.REGISTERED_PATH
            db.REGISTERED_PATH = Path(tmp) / "reg.json"
            try:
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(err):
                    rc = db.main(["--ticker", "SPX", "--extended-spx",
                                  "--burn-in-years", "3", "--data-dir", tmp,
                                  "--write-registered"])
                self.assertEqual(rc, 1)
                self.assertIn("SPY-only", err.getvalue())
                self.assertFalse(db.REGISTERED_PATH.exists())
            finally:
                db.REGISTERED_PATH = orig


class SpxArtifactConformanceTests(unittest.TestCase):
    """Regen tripwire for the SPX-extension registration (spec 2026-07-19):
    the COMMITTED artifact must stay the registered run — the fourth
    registered honest negative on the dip registered-claims ledger (after
    #273/#274/#275).
    The 2026-07-20 SPX run: 3,299 evals 1960-01-04->2025-07-14 (10y burn-in
    over a 1950+ spliced index — far more episodes than #274's 22y sample:
    42 edge episodes vs ~6). Full-span replay primary: edge-claimed minus
    all-days Omega +0.0656, 90% CI [-2.101, +5.290] -> not_supported — the
    CI is wide open despite the much larger episode count. All three #274
    gate candidates (rr0/rr50/pt67) also not_supported on the extended span
    (point stats now NEGATIVE: -0.60/-0.58/-1.15, versus #274's positive-
    trending pt67 +259.5 on only 2 episodes) -> selected=null, same as the
    incumbent. Burn-in-5y sensitivity: also not_supported (-0.214, CI
    [-2.370, +4.154], 50 episodes). Descriptive-only: the incumbent-window
    row (2003-01-29->2025-06-23, same dates as the #258 registration) still
    reads validated (+25.98, CI [+8.83, +352.46], 9 episodes vs the
    original's 6) when priced off the extended-history depth pools — richer
    conditional history moves the incumbent window's OWN numbers, but that
    local edge does not replicate once the pre-2003 episodes (1962, 1966,
    1968-70, 1973-74, 1987, 2000-02) join the same bootstrap. More history
    manufactured episodes (refuting the "too few spells" framing) but did
    NOT confirm the edge over the full 75-year span — if anything it
    diluted it. A regen that flips any of this forces the change-control
    conversation."""

    @classmethod
    def setUpClass(cls):
        cls.art = db.load_spx_registered_artifact()

    def test_loads_via_loader(self):
        self.assertIsNotNone(self.art)

    def test_identity_and_config(self):
        a = self.art
        self.assertEqual(a["kind"], "spx_extension")
        self.assertEqual(a["ticker"], "SPX")
        self.assertEqual(a["registered"], "2026-07-19")
        self.assertEqual(a["config"]["stride"], 5)
        self.assertEqual(a["config"]["burn_in_years"], 10)
        self.assertEqual(a["config"]["min_episodes"], 5)
        self.assertEqual(a["consequence"], db.SPX_CONSEQUENCE)
        self.assertTrue(a["series"]["data_gates"]["all_ok"])
        self.assertEqual(a["series"]["meta"]["start"], "1950-01-03")
        self.assertEqual(a["series"]["meta"]["tr_segments"]["sp500tr_from"],
                         "1988-01-04")

    def test_pinned_evals_and_outcomes(self):
        a = self.art
        self.assertEqual(a["evals"]["n"], 3299)
        self.assertEqual(a["evals"]["first"], "1960-01-04")
        self.assertEqual(a["evals"]["last"], "2025-07-14")
        self.assertEqual(a["replay"]["primary"]["outcome"], "not_supported")
        self.assertEqual(a["replay"]["primary"]["n_edge_episodes"], 42)
        self.assertEqual(a["gate"]["selected"], None)
        self.assertEqual(
            [c["primary"]["outcome"] for c in a["gate"]["candidates"]],
            ["not_supported", "not_supported", "not_supported"])
        self.assertEqual(a["sensitivity"]["burn_in_5y"]["primary"]["outcome"],
                         "not_supported")
        self.assertAlmostEqual(a["replay"]["primary"]["stat"],
                               0.06561264095943997, places=6)


if __name__ == "__main__":
    unittest.main()
