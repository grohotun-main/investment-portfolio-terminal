"""Tests for parsers/single_name_hedge.py (spec 2026-07-06, tests 1-12b)."""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from tail_risk import fit_gpd_tail  # noqa: E402
from single_name_hedge import (  # noqa: E402
    REPLAY_WINDOWS,
    FloorPackage,
    KickerPackage,
    PutQuote,
    build_floor_menu,
    buy_price_of,
    crash_mark_table,
    crash_replays,
    filter_stale_dominated,
    horizon_loss_odds,
    kicker_package,
    market_implied_prob,
    normalize_chain,
    payoff_grid,
    solve_floor,
    vol_context,
)


def _row(**kw):
    """Flattened-chain row in the fetch_options_chains._flatten schema."""
    base = {
        "underlying": "TST",
        "contract_ticker": "O:TST270115P00095000",
        "contract_type": "put",
        "strike": 95.0,
        "expiration_date": "2027-01-15",
        "polygon_bid": 3.9,
        "polygon_ask": 4.0,
        "polygon_price": 3.95,
        "polygon_iv": 0.35,
        "polygon_delta": -0.25,
        "polygon_open_interest": 150,
    }
    base.update(kw)
    return base


class TestNormalizeChain(unittest.TestCase):
    def test_filters_and_parses(self):
        rows = [
            _row(),
            _row(contract_type="call"),                     # dropped: not a put
            _row(polygon_bid=None, polygon_ask=None,
                 polygon_price=None),                        # dropped: no price at all
            _row(polygon_bid=0.0, polygon_ask=0.0,
                 polygon_price=0.0),                         # dropped: all zero
            _row(strike=90.0, polygon_ask=None,
                 polygon_price=2.5, polygon_iv=None,
                 polygon_open_interest=None),                # kept: last-only, no IV/OI
        ]
        quotes = normalize_chain(rows)
        self.assertEqual(len(quotes), 2)
        q0 = quotes[0]
        self.assertIsInstance(q0, PutQuote)
        self.assertEqual(q0.expiry, date(2027, 1, 15))
        self.assertEqual(q0.strike, 95.0)
        self.assertEqual(q0.ask, 4.0)
        self.assertEqual(q0.open_interest, 150)
        q1 = quotes[1]
        self.assertIsNone(q1.ask)
        self.assertIsNone(q1.iv)
        self.assertEqual(q1.open_interest, 0)

    def test_nan_hygiene(self):
        # NaN floats (not None) must not raise and must normalize to None
        rows = [_row(polygon_ask=float("nan"), polygon_bid=float("nan"),
                     polygon_price=2.0, polygon_iv=float("nan"))]
        quotes = normalize_chain(rows)
        self.assertEqual(len(quotes), 1)
        self.assertIsNone(quotes[0].ask)
        self.assertIsNone(quotes[0].iv)

    def test_bad_expiry_dropped(self):
        quotes = normalize_chain([_row(expiration_date=None),
                                  _row(expiration_date="not-a-date")])
        self.assertEqual(quotes, [])

    def test_inf_treated_as_missing(self):
        quotes = normalize_chain([_row(polygon_ask=float("inf"),
                                       polygon_price=2.0)])
        self.assertEqual(len(quotes), 1)
        self.assertIsNone(quotes[0].ask)


class TestBuyPrice(unittest.TestCase):
    def test_ask_preferred_live(self):
        q = normalize_chain([_row()])[0]
        self.assertEqual(buy_price_of(q), (4.0, False))

    def test_last_fallback_is_stale(self):
        q = normalize_chain([_row(polygon_ask=None, polygon_price=4.2)])[0]
        self.assertEqual(buy_price_of(q), (4.2, True))

    def test_bid_only_unusable(self):
        q = normalize_chain([_row(polygon_ask=None, polygon_price=None,
                                  polygon_bid=1.0)])[0]
        self.assertEqual(buy_price_of(q), (None, True))


class TestFilterStaleDominated(unittest.TestCase):
    def test_underpriced_high_strike_dropped(self):
        chain = normalize_chain([
            _row(strike=700.0, polygon_ask=190.0),
            _row(strike=720.0, polygon_ask=210.0),
            _row(strike=730.0, polygon_ask=None, polygon_price=158.0),
            _row(strike=740.0, polygon_ask=None, polygon_price=177.0),
            _row(strike=750.0, polygon_ask=236.0),
        ])
        kept, dropped = filter_stale_dominated(chain)
        self.assertEqual(dropped, 2)
        self.assertEqual([q.strike for q in kept],
                         [700.0, 720.0, 750.0])

    def test_monotone_chain_untouched_and_expiry_isolated(self):
        chain = normalize_chain([
            _row(strike=90.0, polygon_ask=3.0),
            _row(strike=95.0, polygon_ask=4.0),
            # different expiry: its own running max, cheap start is fine
            _row(strike=97.0, polygon_ask=1.0,
                 expiration_date="2027-03-19"),
        ])
        kept, dropped = filter_stale_dominated(chain)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 3)

    def test_tolerance_spares_rounding_inversions(self):
        chain = normalize_chain([
            _row(strike=90.0, polygon_ask=10.00),
            _row(strike=91.0, polygon_ask=9.85),   # -1.5% — inside 2% slack
        ])
        kept, dropped = filter_stale_dominated(chain)
        self.assertEqual(dropped, 0)

    def test_bid_only_rows_pass_through(self):
        chain = normalize_chain([
            _row(strike=90.0, polygon_ask=10.0),
            _row(strike=95.0, polygon_ask=None, polygon_price=None,
                 polygon_bid=2.0),
        ])
        kept, dropped = filter_stale_dominated(chain)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)


SELL_BY = date(2027, 1, 15)
TODAY = date(2026, 7, 6)


def _chain_basic():
    """spot=100 test chain. All expiries valid unless stated."""
    return normalize_chain([
        _row(strike=90.0, polygon_ask=3.0, polygon_iv=0.40),
        _row(strike=95.0, polygon_ask=4.0, polygon_iv=0.36),
        _row(strike=97.0, polygon_ask=5.0, polygon_iv=0.35),
        _row(strike=105.0, polygon_ask=6.5, polygon_iv=0.33),
        _row(strike=85.0, polygon_ask=1.5, polygon_iv=0.42),
        _row(strike=80.0, polygon_ask=1.0, polygon_iv=0.45),
    ])


class TestSolveFloor(unittest.TestCase):
    def test_premium_inclusive_pick(self):
        # Naive "strike at -10%" (K=90, ask 3) loses 13% all-in — must NOT
        # be picked. Cheapest premium-inclusive-valid is K=95 @ 4.
        pkg = solve_floor(_chain_basic(), spot=100.0, shares=1800,
                          floor_pct=0.10, sell_by=SELL_BY)
        self.assertIsNotNone(pkg)
        self.assertEqual(pkg.quote.strike, 95.0)
        self.assertEqual(pkg.contracts, 18)
        self.assertEqual(pkg.buy_price, 4.0)
        self.assertEqual(pkg.total_cost, 4.0 * 1800)
        self.assertAlmostEqual(pkg.guaranteed_loss_pct, 0.09)
        self.assertAlmostEqual(pkg.guaranteed_value, (95.0 - 4.0) * 1800)
        self.assertLessEqual(pkg.guaranteed_loss_pct, 0.10)

    def test_expiry_gate(self):
        # A cheaper valid strike expiring BEFORE sell_by must be ignored.
        chain = _chain_basic() + normalize_chain(
            [_row(strike=95.0, polygon_ask=2.0,
                  expiration_date="2026-12-18")])
        pkg = solve_floor(chain, 100.0, 1800, 0.10, SELL_BY)
        self.assertEqual(pkg.buy_price, 4.0)
        self.assertEqual(pkg.quote.expiry, SELL_BY)

    def test_itm_reachable(self):
        # A 2% floor is only reachable ITM here (K=105 @ 6.5 -> loss 1.5%).
        pkg = solve_floor(_chain_basic(), 100.0, 1800, 0.02, SELL_BY)
        self.assertEqual(pkg.quote.strike, 105.0)
        self.assertAlmostEqual(pkg.guaranteed_loss_pct, 0.015)

    def test_infeasible_returns_none(self):
        chain = [q for q in _chain_basic() if q.strike <= 105.0]
        self.assertIsNone(solve_floor(chain, 100.0, 1800, 0.01, SELL_BY))

    def test_menu_monotone_and_keeps_other_floors(self):
        chain = [q for q in _chain_basic() if q.strike <= 105.0]
        menu = build_floor_menu(chain, 100.0, 1800, SELL_BY,
                                [0.01, 0.10, 0.20, 0.30],
                                today=TODAY, r=0.04, q_yield=0.005)
        self.assertIsNone(menu[0])                      # infeasible floor
        costs = [p.total_cost for p in menu[1:]]
        self.assertEqual(costs, sorted(costs, reverse=True))
        self.assertEqual([p.quote.strike for p in menu[1:]],
                         [95.0, 85.0, 80.0])

    def test_stale_quote_basis(self):
        chain = normalize_chain([_row(strike=95.0, polygon_ask=None,
                                      polygon_price=4.2)])
        pkg = solve_floor(chain, 100.0, 1800, 0.10, SELL_BY)
        self.assertTrue(pkg.stale_quote)
        self.assertEqual(pkg.buy_price, 4.2)

    def test_menu_attaches_market_implied(self):
        menu = build_floor_menu(_chain_basic(), 100.0, 1800, SELL_BY, [0.10],
                                today=TODAY, r=0.04, q_yield=0.005)
        p = menu[0].market_implied_prob
        self.assertIsNotNone(p)
        self.assertTrue(0.0 < p < 1.0)

    def test_odd_lot_under_100_shares(self):
        self.assertIsNone(solve_floor(_chain_basic(), 100.0, 50, 0.10,
                                      SELL_BY))


class TestMarketImpliedProb(unittest.TestCase):
    def test_monotone_in_strike(self):
        probs = [market_implied_prob(100.0, k, 0.5, 0.04, 0.01, 0.40)
                 for k in (60.0, 80.0, 100.0, 120.0)]
        self.assertEqual(probs, sorted(probs))

    def test_extremes_and_missing_iv(self):
        self.assertGreater(
            market_implied_prob(100.0, 300.0, 0.5, 0.04, 0.0, 0.40), 0.99)
        self.assertLess(
            market_implied_prob(100.0, 10.0, 0.5, 0.04, 0.0, 0.40), 0.01)
        self.assertIsNone(market_implied_prob(100.0, 90.0, 0.5, 0.04, 0.0, None))
        self.assertIsNone(market_implied_prob(100.0, 90.0, 0.0, 0.04, 0.0, 0.4))


def _core_pkg():
    return solve_floor(_chain_basic(), 100.0, 1800, 0.10, SELL_BY)  # K=95 @ 4


class TestPayoffGrid(unittest.TestCase):
    def test_floor_below_strike_and_cost_above(self):
        core = _core_pkg()
        grid = payoff_grid(100.0, 1800, core, None,
                           s_grid=[20.0, 50.0, 95.0, 120.0])
        below = grid[grid["price"] == 50.0].iloc[0]
        # stock 90,000 + puts 18*100*45 = 81,000 - cost 7,200 = 163,800
        self.assertAlmostEqual(below["hedged"], core.guaranteed_value)
        self.assertAlmostEqual(below["unhedged"], 50.0 * 1800)
        deep = grid[grid["price"] == 20.0].iloc[0]
        self.assertAlmostEqual(deep["hedged"], core.guaranteed_value)
        above = grid[grid["price"] == 120.0].iloc[0]
        self.assertAlmostEqual(above["hedged"],
                               120.0 * 1800 - core.total_cost)
        self.assertNotIn("hedged_kicker", grid.columns)

    def test_kicker_column_adds_far_left(self):
        core = _core_pkg()
        kq = normalize_chain([_row(strike=55.0, polygon_ask=0.40,
                                   polygon_open_interest=100)])[0]
        kick = KickerPackage(quote=kq, contracts=18, buy_price=0.40,
                             stale_quote=False, total_cost=720.0,
                             payout_per_10pct_below_strike=18 * 100 * 100.0 * 0.10)
        grid = payoff_grid(100.0, 1800, core, kick,
                           s_grid=[20.0, 80.0])
        at20 = grid[grid["price"] == 20.0].iloc[0]
        self.assertAlmostEqual(
            at20["hedged_kicker"],
            at20["hedged"] + 18 * 100 * (55.0 - 20.0) - 720.0)
        at80 = grid[grid["price"] == 80.0].iloc[0]
        self.assertAlmostEqual(at80["hedged_kicker"], at80["hedged"] - 720.0)


class TestCrashReplays(unittest.TestCase):
    def test_known_drawdown_and_skip(self):
        idx = pd.bdate_range("2019-06-03", "2020-12-31")
        price = pd.Series(100.0, index=idx)
        # engineer a -40% inside the COVID window
        price.loc["2020-02-19":] = 100.0
        price.loc["2020-03-01":"2020-03-23"] = 60.0
        price.loc["2020-03-24":] = 80.0
        reps = crash_replays(price)
        labels = {r["label"] for r in reps}
        self.assertIn("2020 COVID", labels)
        self.assertNotIn("2000-02 dot-com", labels)   # no data -> skipped
        covid = next(r for r in reps if r["label"] == "2020 COVID")
        self.assertAlmostEqual(covid["drawdown_pct"], -0.40, places=6)

    def test_unsorted_index_tolerated(self):
        idx = pd.bdate_range("2019-06-03", "2020-12-31")
        price = pd.Series(100.0, index=idx)
        price.loc["2020-03-01":"2020-03-23"] = 60.0
        shuffled = price.sample(frac=1.0, random_state=1)
        reps = crash_replays(shuffled)
        covid = next(r for r in reps if r["label"] == "2020 COVID")
        self.assertAlmostEqual(covid["drawdown_pct"], -0.40, places=6)

    def test_partial_window_coverage_skipped(self):
        # history starts mid-COVID-window: a 0% "replay" would be a lie
        idx = pd.bdate_range("2020-03-10", "2021-12-31")
        price = pd.Series(100.0, index=idx)
        reps = crash_replays(price)
        self.assertNotIn("2020 COVID", {r["label"] for r in reps})


class TestKicker(unittest.TestCase):
    def _kchain(self, oi50=100):
        return normalize_chain([
            _row(strike=55.0, polygon_ask=0.40, polygon_open_interest=100),
            _row(strike=50.0, polygon_ask=0.30, polygon_open_interest=oi50),
            _row(strike=60.0, polygon_ask=0.55, polygon_open_interest=200),
        ])

    def test_cheapest_within_otm_and_budget(self):
        k = kicker_package(self._kchain(), 100.0, 1800, SELL_BY,
                           otm_pct=0.45, budget=720.0)
        self.assertEqual(k.quote.strike, 50.0)          # cheapest ask
        self.assertEqual(k.contracts, 24)               # floor(720/30)
        self.assertAlmostEqual(k.total_cost, 720.0)

    def test_oi_gate(self):
        k = kicker_package(self._kchain(oi50=0), 100.0, 1800, SELL_BY,
                           otm_pct=0.45, budget=720.0)
        self.assertEqual(k.quote.strike, 55.0)
        self.assertEqual(k.contracts, 18)               # floor(720/40)

    def test_unaffordable_none(self):
        self.assertIsNone(kicker_package(self._kchain(), 100.0, 1800,
                                         SELL_BY, otm_pct=0.45, budget=20.0))

    def test_otm_threshold(self):
        # otm 0.45 on spot 100 -> strikes must be <= 55; K=60 never eligible
        k = kicker_package(self._kchain(oi50=0), 100.0, 1800, SELL_BY,
                           otm_pct=0.45, budget=720.0)
        self.assertLessEqual(k.quote.strike, 55.0)

    def test_float_floor_budget_not_off_by_one(self):
        chain = normalize_chain([
            _row(strike=50.0, polygon_ask=0.30, polygon_open_interest=10)])
        budget = 24 * 0.30 * 100.0   # 719.9999999999999 in float
        k = kicker_package(chain, 100.0, 1800, SELL_BY,
                           otm_pct=0.45, budget=budget)
        self.assertEqual(k.contracts, 24)

    def test_replay_anchored_pick_beats_deepest(self):
        chain = normalize_chain([
            _row(strike=10.0, polygon_ask=0.05, polygon_open_interest=50),
            _row(strike=25.0, polygon_ask=0.40, polygon_open_interest=50),
            _row(strike=40.0, polygon_ask=1.60, polygon_open_interest=50),
        ])
        # replay price 15: K=10 pays nothing, K=25 pays 10/share on 25
        # contracts (budget 1000 // 40), K=40 pays 25/share on 6 contracts
        k = kicker_package(chain, 100.0, 1800, SELL_BY, otm_pct=0.55,
                           budget=1000.0, replay_price=15.0)
        self.assertEqual(k.quote.strike, 25.0)   # 25*100*10=25k > 6*100*25=15k
        self.assertEqual(k.contracts, 25)

    def test_replay_none_or_uncleared_falls_back_cheapest(self):
        chain = normalize_chain([
            _row(strike=10.0, polygon_ask=0.05, polygon_open_interest=50),
            _row(strike=25.0, polygon_ask=0.40, polygon_open_interest=50),
        ])
        k_none = kicker_package(chain, 100.0, 1800, SELL_BY, otm_pct=0.55,
                                budget=1000.0)
        self.assertEqual(k_none.quote.strike, 10.0)
        k_deep = kicker_package(chain, 100.0, 1800, SELL_BY, otm_pct=0.55,
                                budget=1000.0, replay_price=30.0)
        self.assertEqual(k_deep.quote.strike, 10.0)   # nothing clears 30


class TestHorizonLossOdds(unittest.TestCase):
    def _price(self, n=1600, seed=7, sigma=0.02):
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0004, sigma, n)
        idx = pd.bdate_range("2018-01-02", periods=n)
        return pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)

    def test_shape_and_monotone_in_depth(self):
        odds = horizon_loss_odds(self._price(), horizon_td=126,
                                 floors=[0.10, 0.20, 0.30, 0.50])
        self.assertEqual(set(odds), {0.10, 0.20, 0.30, 0.50})
        probs = [odds[f]["prob"] for f in (0.10, 0.20, 0.30, 0.50)]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for f, d in odds.items():
            self.assertGreaterEqual(d["hi"], d["prob"])
            self.assertLessEqual(d["lo"], d["prob"])
            self.assertIn(d["source"], ("empirical", "gpd", "none"))
            self.assertIn("n_windows", d)
            self.assertIn("n_hits", d)
            self.assertIn("prob_20y", d)
            self.assertIn("confident", d)

    def test_exact_empirical_frequency(self):
        # 6 prices, horizon 2 -> window rets: 0, -0.5, 0, +1.0
        price = pd.Series([100.0, 100.0, 100.0, 50.0, 100.0, 100.0],
                          index=pd.bdate_range("2024-01-01", periods=6))
        odds = horizon_loss_odds(price, horizon_td=2, floors=[0.30])
        d = odds[0.30]
        self.assertEqual(d["n_windows"], 4)
        self.assertEqual(d["n_hits"], 1)
        self.assertAlmostEqual(d["prob"], 0.25)

    def test_gpd_engages_only_when_sparse(self):
        odds = horizon_loss_odds(self._price(), horizon_td=126,
                                 floors=[0.10, 0.60])
        self.assertEqual(odds[0.10]["source"], "empirical")
        self.assertIn(odds[0.60]["source"], ("gpd", "none"))

    def test_20y_variant_equals_full_when_short(self):
        odds = horizon_loss_odds(self._price(n=800), horizon_td=63,
                                 floors=[0.10])
        d = odds[0.10]
        self.assertAlmostEqual(d["prob_20y"], d["prob"])

    def test_empty_floors_returns_empty(self):
        self.assertEqual(horizon_loss_odds(self._price(n=400),
                                           horizon_td=63, floors=[]), {})

    def test_horizon_longer_than_series(self):
        odds = horizon_loss_odds(self._price(n=100), horizon_td=500,
                                 floors=[0.10])
        d = odds[0.10]
        self.assertEqual(d["n_windows"], 0)
        self.assertEqual(d["source"], "none")

    def test_gpd_probability_is_unconditional(self):
        from single_name_hedge import _gpd_exceed_prob
        rng = np.random.default_rng(5)
        wr = pd.Series(rng.normal(0.02, 0.15, 800))
        losses = (-wr[wr < 0]).to_numpy()
        fit = fit_gpd_tail(losses)
        u = float(fit["threshold"])
        p_u = _gpd_exceed_prob(fit, u + 1e-9, len(wr))
        emp_u = float((wr <= -(u + 1e-9)).mean())
        self.assertAlmostEqual(p_u, emp_u, delta=0.01)


class TestVolContext(unittest.TestCase):
    def _price_calm_then_wild(self):
        idx = pd.bdate_range("2019-01-01", periods=900)
        rng = np.random.default_rng(3)
        rets = np.concatenate([rng.normal(0, 0.005, 850),
                               rng.normal(0, 0.06, 50)])
        p = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
        # place the wild tail inside the 2020 COVID replay window
        return p

    def test_rv_percentile_maxed_when_wild_tail(self):
        ctx = vol_context(self._price_calm_then_wild(), iv_today=0.50)
        self.assertGreater(ctx["rv_percentile"], 95.0)
        self.assertGreater(ctx["rv_today"], 0.0)
        self.assertAlmostEqual(ctx["iv_rv_spread"],
                               0.50 - ctx["rv_today"])
        self.assertEqual(ctx["iv_source"], "proxy")
        self.assertIsNone(ctx["iv_percentile"])
        self.assertIsNone(ctx["vix_percentile"])

    def test_true_iv_path_and_anchor_max(self):
        price = self._price_calm_then_wild()
        iv_series = pd.Series(
            np.linspace(0.25, 0.45, 600),
            index=pd.bdate_range("2023-01-02", periods=600))
        ctx = vol_context(price, iv_today=0.60, iv_series=iv_series)
        self.assertEqual(ctx["iv_source"], "true")
        self.assertEqual(ctx["iv_percentile"], 100.0)   # above every point
        self.assertEqual(ctx["iv_history_start"], iv_series.index[0].date())
        self.assertAlmostEqual(ctx["iv_history_high"], float(iv_series.max()))
        p995 = float(iv_series.quantile(0.995))
        rv_term = ctx["crash_window_peaks"]
        rv_anchor = (float(np.median(list(rv_term.values())))
                     if rv_term else None)
        expected = max(p995, rv_anchor) if rv_anchor is not None else p995
        self.assertAlmostEqual(ctx["crash_iv_anchor"], expected)

    def test_thin_iv_series_falls_back_to_proxy(self):
        price = self._price_calm_then_wild()
        thin = pd.Series([0.3] * 100,
                         index=pd.bdate_range("2026-01-01", periods=100))
        ctx = vol_context(price, iv_today=0.40, iv_series=thin)
        self.assertEqual(ctx["iv_source"], "proxy")
        self.assertIsNone(ctx["iv_percentile"])

    def test_dense_two_year_series_is_true(self):
        # live finding: ~75% fill over ~2 years (378 rows, ~704-day span)
        # is the best series the plan tier produces — it must rank as "true"
        iv_series = pd.Series(0.30, index=pd.bdate_range("2024-07-08",
                                                         periods=505))
        iv_series = iv_series.sample(n=378, random_state=1).sort_index()
        ctx = vol_context(self._price_calm_then_wild(), iv_today=0.5,
                          iv_series=iv_series)
        self.assertEqual(ctx["iv_source"], "true")

    def test_vix_percentile_when_supplied(self):
        price = self._price_calm_then_wild()
        vix = pd.Series(np.linspace(12.0, 30.0, 500),
                        index=pd.bdate_range("2024-01-01", periods=500))
        ctx = vol_context(price, iv_today=None, vix=vix)
        self.assertAlmostEqual(ctx["vix_today"], 30.0)
        self.assertEqual(ctx["vix_percentile"], 100.0)
        self.assertIsNone(ctx["iv_rv_spread"])   # no iv_today

    def test_nan_iv_today_treated_as_none(self):
        ctx = vol_context(self._price_calm_then_wild(),
                          iv_today=float("nan"))
        self.assertIsNone(ctx["iv_today"])
        self.assertIsNone(ctx["iv_rv_spread"])

    def test_iv_history_start_is_a_date(self):
        from datetime import date as _date
        iv_series = pd.Series(
            np.linspace(0.25, 0.45, 600),
            index=pd.bdate_range("2023-01-02", periods=600))
        ctx = vol_context(self._price_calm_then_wild(), iv_today=0.5,
                          iv_series=iv_series)
        self.assertIsInstance(ctx["iv_history_start"], _date)


class TestCrashMarkTable(unittest.TestCase):
    def test_monotone_iv_columns_and_floor(self):
        core = _core_pkg()   # K=95 @ 4, expiry 2027-01-15
        tbl = crash_mark_table(core, today=TODAY, spot=100.0, r=0.04,
                               q=0.005, iv_today=0.35, crash_iv_anchor=0.80)
        # 3 drops x 2 timings x 3 IV scenarios
        self.assertEqual(len(tbl), 18)
        self.assertTrue((tbl["excess"] >= -1e-6).all())
        self.assertTrue((tbl["value"] >= tbl["intrinsic"] - 1e-6).all())
        for (_, _), grp in tbl.groupby(["drop_pct", "at_td"]):
            vals = grp.sort_values("iv_used")["value"].to_list()
            self.assertEqual(vals, sorted(vals))
            self.assertEqual(len(set(grp["iv_scenario"])), 3)

    def test_clamp_when_fear_already_maxed(self):
        core = _core_pkg()
        tbl = crash_mark_table(core, today=TODAY, spot=100.0, r=0.04,
                               q=0.005, iv_today=0.90, crash_iv_anchor=0.80)
        self.assertTrue((tbl["iv_used"] == 0.90).all())
        for (_, _), grp in tbl.groupby(["drop_pct", "at_td"]):
            self.assertAlmostEqual(grp["value"].max(), grp["value"].min())

    def test_converges_to_intrinsic_at_expiry(self):
        chain = normalize_chain([_row(strike=95.0, polygon_ask=4.0,
                                      expiration_date=(TODAY + timedelta(days=32)).isoformat())])
        core = solve_floor(chain, 100.0, 1800, 0.10,
                           sell_by=TODAY + timedelta(days=30))
        tbl = crash_mark_table(core, today=TODAY, spot=100.0, r=0.04,
                               q=0.005, iv_today=0.35, crash_iv_anchor=0.80,
                               at_days_list=(21,))
        deep = tbl[tbl["drop_pct"] == 0.40]
        self.assertTrue((deep["excess"] <= 0.02 * deep["intrinsic"]).all())

    def test_nan_anchor_collapses_columns_finite(self):
        # A NaN crash_iv_anchor (e.g. price-history fetch failed) must not
        # poison the lerp via max(x, nan) == x silently faking a clamp; the
        # anchor degrades explicitly to iv_today so every value stays finite
        # and the three IV-scenario columns intentionally coincide.
        core = _core_pkg()
        tbl = crash_mark_table(core, today=TODAY, spot=100.0, r=0.04,
                               q=0.005, iv_today=0.35,
                               crash_iv_anchor=float("nan"))
        self.assertTrue(np.isfinite(tbl["value"].to_numpy()).all())
        self.assertTrue((tbl["iv_used"] == 0.35).all())
        for (_, _), grp in tbl.groupby(["drop_pct", "at_td"]):
            self.assertAlmostEqual(grp["value"].max(), grp["value"].min())

    def test_all_degenerate_returns_empty(self):
        # iv_today=None AND a NaN anchor leaves nothing to mark to model;
        # NaN must not reach binomial_american as sigma (its sigma<=0 guard
        # is False for NaN, producing an all-NaN table) - return an empty
        # frame with the expected columns instead, which the renderer
        # already handles.
        core = _core_pkg()
        tbl = crash_mark_table(core, today=TODAY, spot=100.0, r=0.04,
                               q=0.005, iv_today=None,
                               crash_iv_anchor=float("nan"))
        self.assertEqual(len(tbl), 0)
        self.assertIn("iv_scenario", tbl.columns)


if __name__ == "__main__":
    unittest.main()
