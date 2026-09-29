"""CAPM expected returns + a constrained efficient-frontier tracer.

Pure orchestration. Betas come from factor_regression, Sigma and every solve
from min_variance / risk_parity, so each frontier point is exactly what the
tab's own optimizer would produce under the same constraints at that
risk aversion.

The E[r] estimator is CAPM-implied by design decision (see the 2026-08-15
efficient-frontier spec): on an ~11-name book the standard error of a 10-year
sample mean is about the size of the signal, so a historical-mean frontier
would optimize estimation error. Expected returns here are estimates, not
forecasts, and every consumer must disclose that.

Run tests from phase1_build/ with:
    py -m unittest tests.test_frontier
"""
import numpy as np
import pandas as pd

from factor_regression import per_holding_regressions
from min_variance import build_covariance, suggest_min_variance_grid
from opt_curve import _concentration
from risk_parity import suggest_risk_parity_grid

# Risk-aversion ladder for the frontier sweep. On this book's scale (annual
# vol ~0.15 -> variance ~0.02; mu ~0.08) the variance term dominates at the
# top and the return term at the bottom, so the ladder spans the
# minimum-variance portfolio to the maximum-return corner.
#
# Anchor-plus-band: 10**2.5 (~316) is the old ladder's top value, kept as a
# single high anchor because test_first_point_matches_the_min_variance_
# suggestion needs a large lambda to approximate the mu=None min-variance
# corner. A log-spaced ladder over the FULL range bunches badly on a real
# book: measured live (real data dir, the book's live optimizer defaults)
# every lambda from 316 down to ~7 landed within 0.5% of the anchor's vol --
# 5+ of 12 points on top of each other at the min-variance corner, leaving
# only the low end to describe the arc. The 11 values below the anchor are
# instead placed by measurement (not evenly log-spaced) to keep consecutive
# points spread on the vol axis across the region where the constrained
# solve actually moves, roughly lambda in [0.02, 22]; see tests.test_frontier
# and the O3 smoke-fix report for the measured (lam, vol, exp_return) table.
LAMBDA_LADDER = (
    10 ** 2.5,
    22.0, 8.0, 4.0, 2.0, 1.0, 0.55, 0.3, 0.16, 0.09, 0.045, 0.02,
)

POINT_COLUMNS = ["lam", "vol", "exp_return", "effective_n", "max_weight",
                 "converged", "weights"]

# Beta assumed for holdings with no fitted regression (unpriced or too thin).
# Callers MUST name these holdings in their disclosure.
ASSUMED_BETA = 1.0

_BETA_COL = "beta_mkt_rf"


def capm_expected_returns(daily_prices: pd.DataFrame, ff_daily: pd.DataFrame,
                          symbols, *, rf_annual: float, erp: float,
                          window_days: int | None = 1260,
                          min_obs: int = 126) -> dict:
    """CAPM-implied annual expected returns: E[r_i] = rf + beta_i * ERP.

    Betas are the market-factor loadings from per_holding_regressions over the
    trailing `window_days` of daily returns. A symbol with no price column, or
    too little history to clear `min_obs`, gets ASSUMED_BETA and is listed in
    `assumed` so the UI can disclose it. rf_annual and erp are annual decimals.

    Returns {mu: pd.Series indexed by `symbols`, betas: pd.Series,
             assumed: list[str], error: str|None}. `error` is set only when mu
    cannot be built at all; an all-assumed mu is a legal disclosed result.
    """
    syms = [str(s) for s in symbols]
    blank = {"mu": pd.Series(dtype=float), "betas": pd.Series(dtype=float),
             "assumed": []}
    if not syms:
        return {**blank, "error": "No holdings to price."}
    if ff_daily is None or ff_daily.empty:
        return {**blank, "error": ("Fama-French daily factors missing - run "
                                   "`py parsers/fetch_ff_factors.py --write`.")}
    if not (np.isfinite(float(rf_annual)) and np.isfinite(float(erp))):
        return {**blank, "error": "Risk-free rate / ERP must be finite."}

    priced = ([s for s in syms if s in daily_prices.columns]
              if daily_prices is not None and not daily_prices.empty else [])
    fitted: dict[str, float] = {}
    if priced:
        table, _skipped = per_holding_regressions(
            daily_prices[priced], ff_daily, "CAPM", window_days,
            min_obs=min_obs)
        for _, row in table.iterrows():
            beta = float(row[_BETA_COL])
            if np.isfinite(beta):
                fitted[str(row["symbol"])] = beta

    betas = pd.Series([fitted.get(s, ASSUMED_BETA) for s in syms],
                      index=syms, dtype=float)
    assumed = [s for s in syms if s not in fitted]
    mu = float(rf_annual) + betas * float(erp)
    return {"mu": mu, "betas": betas, "assumed": assumed, "error": None}


def _expected_return(new_pct: dict, mu: pd.Series) -> float:
    """Portfolio E[r] of a full-book percent allocation, young sleeve included.

    The solver only ever sees the mature sleeve, so the young holdings' return
    contribution enters HERE - new_pct carries them at their current weight.
    """
    total = 0.0
    for sym, pct in new_pct.items():
        m = mu.get(sym, np.nan)
        if not np.isfinite(float(m)) or not np.isfinite(float(pct)):
            return float("nan")
        total += float(pct) / 100.0 * float(m)
    return float(total)


def trace_frontier(daily_prices: pd.DataFrame, weights: pd.Series,
                   class_of: dict[str, str], mu: pd.Series, *,
                   name_cap: float, class_floors: dict[str, float],
                   class_caps: dict[str, float] | None = None,
                   lambdas=LAMBDA_LADDER, min_overlap_days: int = 252,
                   covres: dict | None = None, on_point=None) -> dict:
    """Sweep the risk-aversion ladder and return one frontier point per lambda.

    Each point is a constrained min-variance solve under the scalarized
    objective lambda*w'Sigma*w - w'mu, so it obeys exactly the per-name cap,
    class floors and class caps the tab is showing. Points are returned
    DESCENDING in lambda: the first row is the minimum-variance end and the
    last is the maximum-return end, which is also the chart's line order.

    Individually infeasible lambdas land in `skipped` as (lam, 'min_variance',
    message); `error` is set only when Sigma itself cannot be built or mu does
    not cover the book. `on_point(done, total)` fires after each solve.

    Returns {points: DataFrame[lam, vol, exp_return, effective_n, max_weight,
    converged, weights], current: dict|None, markers: list[dict], skipped:
    list[tuple], error: str|None}. weights is the solve's full-book percent
    allocation (new_pct verbatim) so consumers can apply a traced point
    without re-solving. All values are decimals except `weights`
    (percent-scale); the UI formats to percent.
    """
    empty = pd.DataFrame(columns=POINT_COLUMNS)
    blank = {"points": empty, "current": None, "markers": [], "skipped": []}

    mu_s = pd.Series(mu, dtype=float).reindex(weights.index)
    if not np.all(np.isfinite(mu_s.to_numpy(dtype=float))):
        missing = [str(s) for s in weights.index
                   if not np.isfinite(float(mu_s.get(s, np.nan)))]
        return {**blank, "error": ("Expected returns are missing for: "
                                   + ", ".join(missing[:5]) + ".")}

    if covres is None:
        covres = build_covariance(daily_prices, list(weights.index),
                                  min_overlap_days=min_overlap_days)
    if covres["error"]:
        return {**blank, "error": covres["error"]}

    # Current-book star: vol over the modelable sleeve (same basis the
    # optimizers report), expected return over the WHOLE book.
    s_mat = covres["cov"].to_numpy(dtype=float)
    w_cur = (weights.reindex(covres["cov"].index).fillna(0.0)
             .to_numpy(dtype=float))
    cur_en, cur_max = _concentration(
        {s: float(weights[s]) * 100.0 for s in weights.index})
    current = {
        "vol": float(np.sqrt(max(0.0, float(w_cur @ s_mat @ w_cur)))),
        "exp_return": _expected_return(
            {str(s): float(weights[s]) * 100.0 for s in weights.index}, mu_s),
        "effective_n": cur_en, "max_weight": cur_max,
    }

    def _solve(lam):
        return suggest_min_variance_grid(
            daily_prices, weights, class_of, name_cap=name_cap,
            class_floors=class_floors, class_caps=class_caps, mu=mu_s,
            risk_aversion=lam, min_overlap_days=min_overlap_days,
            covres=covres)

    lams = sorted({float(x) for x in lambdas}, reverse=True)
    rows: list[dict] = []
    skipped: list[tuple[float, str, str]] = []
    total = len(lams)
    for done, lam in enumerate(lams, start=1):
        out = _solve(lam)
        if out["kind"] == "error":
            skipped.append((lam, "min_variance", out["message"]))
        else:
            en, mx = _concentration(out["new_pct"])
            rows.append({"lam": lam, "vol": float(out["vol"]),
                         "exp_return": _expected_return(out["new_pct"], mu_s),
                         "effective_n": en, "max_weight": mx,
                         "converged": bool(out["converged"]),
                         "weights": {str(k): float(v)
                                     for k, v in out["new_pct"].items()}})
        if on_point is not None:
            on_point(done, total)

    # Markers: the two portfolios the tab already suggests, priced through mu.
    markers: list[dict] = []
    for key, label, runner in (
        ("min_variance", "Min-variance",
         lambda: suggest_min_variance_grid(
             daily_prices, weights, class_of, name_cap=name_cap,
             class_floors=class_floors, class_caps=class_caps,
             min_overlap_days=min_overlap_days, covres=covres)),
        ("risk_parity", "Equal risk contribution",
         lambda: suggest_risk_parity_grid(
             daily_prices, weights, name_cap=name_cap, class_of=class_of,
             class_caps=class_caps, min_overlap_days=min_overlap_days,
             covres=covres)),
    ):
        out = runner()
        if out["kind"] == "error":
            continue
        markers.append({"key": key, "label": label, "vol": float(out["vol"]),
                        "exp_return": _expected_return(out["new_pct"], mu_s)})

    points = pd.DataFrame(rows, columns=POINT_COLUMNS) if rows else empty
    return {"points": points, "current": current, "markers": markers,
            "skipped": skipped, "error": None}
