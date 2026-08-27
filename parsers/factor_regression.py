"""
Pure-numpy OLS factor regression of the portfolio's monthly TWR on the
Ken French research factors (data: parsers/fetch_ff_factors.py).

Regresses EXCESS return (return_pct − rf, French 1-month T-bill — academic
convention; differs slightly from the DGS3MO series driving the Sharpe
tiles) on each model's factor columns plus an intercept. The intercept IS
the alpha: monthly return left unexplained after factor exposures are
priced. Alpha is annualized ARITHMETICALLY via periods_per_year (×12
monthly, ×252 daily) — the attribution decomposes the window's arithmetic
mean return, not the geometric TWR.

No scipy/statsmodels (project constraint): lstsq + analytic OLS standard
errors σ̂²(XᵀX)⁻¹ and a fixed two-sided-95% t-table for the CI. Plain OLS
SEs (no HAC) — monthly non-overlapping returns carry little autocorrelation.

All functions are pure (no I/O, no Streamlit) for unit-testability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FACTOR_LABELS = {
    "mkt_rf": "Market − RF",
    "smb": "Size (SMB)",
    "hml": "Value (HML)",
    "rmw": "Profitability (RMW)",
    "cma": "Investment (CMA)",
    "mom": "Momentum (Mom)",
}

MODELS: dict[str, tuple[str, ...]] = {
    "CAPM": ("mkt_rf",),
    "FF3": ("mkt_rf", "smb", "hml"),
    "Carhart 4": ("mkt_rf", "smb", "hml", "mom"),
    "FF5": ("mkt_rf", "smb", "hml", "rmw", "cma"),
    "FF5 + Mom": ("mkt_rf", "smb", "hml", "rmw", "cma", "mom"),
}

# Two-sided 95% t critical values by residual dof (standard table knots).
# Between knots the LOWER knot's value is used — conservative (wider CI);
# above 120 dof the normal 1.96. Keeps the CI honest at the ~70-obs monthly
# sample without a scipy dependency.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
         20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000,
         80: 1.990, 100: 1.984, 120: 1.980}


def t_crit_975(dof: int) -> float:
    """Two-sided 95% t critical value: exact at table knots, the nearest
    LOWER knot between them (conservative), 1.96 above 120 dof."""
    if dof <= 0:
        raise ValueError(f"dof must be positive, got {dof}")
    if dof > 120:
        return 1.96
    lower = max(k for k in _T975 if k <= dof)
    return _T975[lower]


@dataclass(frozen=True)
class FactorRegression:
    """One fitted model. Everything pre-computed so the UI only formats."""
    model: str
    factors: tuple[str, ...]
    n: int
    months: tuple[str, ...]            # aligned period labels (YYYY-MM or YYYY-MM-DD), ascending
    betas: dict[str, float]
    se: dict[str, float]
    tstats: dict[str, float]
    alpha_monthly: float
    alpha_t: float
    alpha_annual: float                # alpha_monthly * periods_per_year (arith.)
    alpha_ci_annual: tuple[float, float]
    r2: float
    adj_r2: float
    rf_mean_monthly: float
    factor_means: dict[str, float]     # mean monthly factor value, window
    mean_return_monthly: float         # mean RAW portfolio return, window
    periods_per_year: int = 12         # 12 monthly, 252 daily


def align_returns_with_factors(rets: pd.Series, factors: pd.DataFrame,
                               window: int | None = None) -> pd.DataFrame:
    """Inner-join a return Series with a factor frame on period label.

    rets: decimal returns indexed by period — a DatetimeIndex (daily) or
    anything whose str() form matches the factor frame's key column
    ('month' for the monthly file, 'date' for the daily file). Returns the
    align_twr_with_factors schema — key column is ALWAYS named 'month'
    (run_factor_regression treats it as an opaque label) — ascending,
    NaN rows dropped, trimmed to the trailing `window` rows AFTER
    alignment. Empty/missing input or no overlap -> empty frame.
    """
    cols = ["month", "ret", "rf"] + list(FACTOR_LABELS)
    if (rets is None or len(rets) == 0
            or factors is None or factors.empty):
        return pd.DataFrame(columns=cols)
    if isinstance(rets.index, pd.DatetimeIndex):
        labels = rets.index.strftime("%Y-%m-%d")
    else:
        labels = rets.index.astype(str)
    t = pd.DataFrame({"month": labels,
                      "ret": rets.to_numpy(dtype=float)})
    key = "month" if "month" in factors.columns else "date"
    f = factors.rename(columns={key: "month"}).copy()
    f["month"] = f["month"].astype(str)
    df = (t.merge(f, on="month", how="inner")
           .dropna().sort_values("month").reset_index(drop=True))
    if window is not None:
        df = df.tail(window).reset_index(drop=True)
    return df[cols]


def align_twr_with_factors(twr_portfolio: pd.DataFrame,
                           factors: pd.DataFrame,
                           window_months: int | None = None) -> pd.DataFrame:
    """Inner-join monthly TWR with the factor frame on month.

    Accepts the load_twr frame (month may be a pandas Period) and the
    ff_factors_monthly frame (month is 'YYYY-MM' str). Returns
    DataFrame[month, ret, rf, mkt_rf, smb, hml, rmw, cma, mom], ascending,
    rows with ANY NaN dropped, trimmed to the trailing `window_months`
    AFTER alignment (None = full common history). Empty/missing input or
    no overlap -> empty frame with those columns.
    """
    if (twr_portfolio is None or twr_portfolio.empty
            or factors is None or factors.empty):
        return pd.DataFrame(columns=["month", "ret", "rf"]
                            + list(FACTOR_LABELS))
    rets = pd.Series(twr_portfolio["return_pct"].to_numpy(),
                     index=twr_portfolio["month"].astype(str))
    return align_returns_with_factors(rets, factors, window_months)


def run_factor_regression(aligned: pd.DataFrame,
                          model: str,
                          periods_per_year: int = 12,
                          ) -> FactorRegression | None:
    """OLS of excess return on MODELS[model] factors + intercept.

    periods_per_year: annualization multiplier (12 for monthly series,
    252 for daily). Affects only alpha_annual and alpha_ci_annual; all
    per-period statistics (alpha_monthly, alpha_t, betas, r2) are invariant.

    Returns None (structured skip) when fewer than k+2 aligned months exist
    or the design matrix is rank-deficient — callers render a "not enough
    history" notice instead of numbers. Unknown model -> KeyError
    (programmer error, not data condition).
    """
    fcols = MODELS[model]
    k = len(fcols)
    n = len(aligned)
    if n < k + 2:
        return None
    y = (aligned["ret"] - aligned["rf"]).to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(n)] + [aligned[c].to_numpy(dtype=float) for c in fcols])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - (k + 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = coef / se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    # A numerically-constant y leaves ~1e-35 of float rounding in ss_tot,
    # never exact 0 — require per-month dispersion above ~1e-9 before
    # calling the variance explainable.
    if ss_tot > 1e-18 * n:
        r2 = 1.0 - float(resid @ resid) / ss_tot
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof
    else:
        r2 = adj_r2 = float("nan")
    tc = t_crit_975(dof)
    alpha_m, alpha_se = float(coef[0]), float(se[0])
    return FactorRegression(
        model=model, factors=fcols, n=n,
        months=tuple(aligned["month"]),
        betas={c: float(b) for c, b in zip(fcols, coef[1:])},
        se={c: float(s) for c, s in zip(fcols, se[1:])},
        tstats={c: float(t) for c, t in zip(fcols, tstats[1:])},
        alpha_monthly=alpha_m,
        alpha_t=float(tstats[0]),
        alpha_annual=alpha_m * float(periods_per_year),
        alpha_ci_annual=((alpha_m - tc * alpha_se) * periods_per_year,
                         (alpha_m + tc * alpha_se) * periods_per_year),
        r2=r2, adj_r2=adj_r2,
        rf_mean_monthly=float(aligned["rf"].mean()),
        factor_means={c: float(aligned[c].mean()) for c in fcols},
        mean_return_monthly=float(aligned["ret"].mean()),
        periods_per_year=periods_per_year,
    )


def attribution(res: FactorRegression) -> list[tuple[str, float]]:
    """Annualized (arithmetic, ×periods_per_year) decomposition of the
    window's mean raw monthly return: RF + per-factor β·mean(factor) + alpha.
    Sums exactly to mean_return_monthly × periods_per_year (OLS-with-intercept
    residuals average zero). Order: RF first, factors in model order, alpha
    last — the UI renders this directly as waterfall steps.
    """
    ppy = float(res.periods_per_year)
    rows: list[tuple[str, float]] = [
        ("Risk-free (1-mo T-bill)", res.rf_mean_monthly * ppy)]
    rows += [(FACTOR_LABELS[c], res.betas[c] * res.factor_means[c] * ppy)
             for c in res.factors]
    rows.append(("Alpha (unexplained)", res.alpha_annual))
    return rows


def attribution_timeseries(res: FactorRegression,
                           aligned: pd.DataFrame) -> pd.DataFrame:
    """Per-month decomposition of raw return under res's static betas.

    Columns: month, rf, contrib_<factor> (model order), unexplained.
    unexplained = alpha + residual, computed as the remainder so each
    row sums to aligned['ret'] exactly; column means × periods_per_year
    reproduce attribution(res), and mean(unexplained) × ppy ==
    alpha_annual (OLS-with-intercept residuals average zero).

    Raises ValueError when aligned's month labels differ from res.months
    — the caller must pass the exact frame the regression was fit on
    (silent wrong numbers otherwise; same loud-on-misuse stance as
    run_factor_regression's KeyError on an unknown model).
    """
    if tuple(aligned["month"]) != res.months:
        got = (f"{aligned['month'].iloc[0]}→{aligned['month'].iloc[-1]}"
               if len(aligned) else "empty")
        want = (f"{res.months[0]}→{res.months[-1]}"
                if res.months else "empty")
        raise ValueError(
            f"aligned frame ({len(aligned)} rows, {got}) is not the "
            f"window res was fit on (n={res.n}, {want})")
    out = pd.DataFrame({"month": aligned["month"].to_numpy()})
    out["rf"] = aligned["rf"].to_numpy(dtype=float)
    explained = out["rf"].to_numpy().copy()
    for c in res.factors:
        contrib = res.betas[c] * aligned[c].to_numpy(dtype=float)
        out[f"contrib_{c}"] = contrib
        explained += contrib
    out["unexplained"] = aligned["ret"].to_numpy(dtype=float) - explained
    return out


def rolling_factor_regressions(aligned: pd.DataFrame, model: str,
                               window: int) -> pd.DataFrame:
    """Trailing-window regressions over the aligned frame.

    One row per window END month (ascending): annualized alpha plus
    beta_<factor> for each MODELS[model] column. A window where
    run_factor_regression structurally skips (rank-deficient — n is fixed
    by construction) yields a NaN row so charts show a gap instead of
    silently dropping the month. Fewer than `window` aligned rows -> empty
    frame with the same columns.
    """
    fcols = MODELS[model]
    cols = ["month", "alpha_annual"] + [f"beta_{c}" for c in fcols]
    n = len(aligned)
    if n < window:
        return pd.DataFrame(columns=cols)
    rows = []
    for end in range(window, n + 1):
        sl = aligned.iloc[end - window:end]
        res = run_factor_regression(sl, model)
        row: dict[str, float | str] = {"month": sl["month"].iloc[-1]}
        if res is None:
            row["alpha_annual"] = float("nan")
            row.update({f"beta_{c}": float("nan") for c in fcols})
        else:
            row["alpha_annual"] = res.alpha_annual
            row.update({f"beta_{c}": res.betas[c] for c in fcols})
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def per_holding_regressions(prices: pd.DataFrame, factors: pd.DataFrame,
                            model: str, window: int | None = None,
                            min_obs: int = 126,
                            periods_per_year: int = 252,
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-column factor regressions of daily price returns.

    prices: wide frame (DatetimeIndex × symbol), as load_daily_prices
    returns. Returns (table, skipped): `table` has one row per symbol that
    cleared min_obs AND fit (columns: symbol, n, start, end, alpha_annual,
    alpha_t, r2, beta_<f>..., t_<f>... in MODELS[model] order, input column
    order preserved); `skipped` lists (symbol, n) for the UI caption — a
    symbol lands there on thin history OR a structural regression skip.

    NaN price gaps DROP those return rows (no fillna-0): a holding must
    not look becalmed on days we lack a print. This intentionally differs
    from synthesize_portfolio_returns, where a missing symbol return must
    not zero out the whole portfolio row.
    """
    fcols = MODELS[model]
    tcols = (["symbol", "n", "start", "end", "alpha_annual", "alpha_t",
              "r2"] + [f"beta_{c}" for c in fcols]
             + [f"t_{c}" for c in fcols])
    scols = ["symbol", "n"]
    if (prices is None or prices.empty
            or factors is None or factors.empty):
        return pd.DataFrame(columns=tcols), pd.DataFrame(columns=scols)
    rows: list[dict] = []
    skipped: list[dict] = []
    for sym in prices.columns:
        rets = prices[sym].pct_change(fill_method=None).dropna()
        aligned = align_returns_with_factors(rets, factors, window)
        res = (run_factor_regression(aligned, model,
                                     periods_per_year=periods_per_year)
               if len(aligned) >= min_obs else None)
        if res is None:
            skipped.append({"symbol": sym, "n": len(aligned)})
            continue
        row = {"symbol": sym, "n": res.n, "start": res.months[0],
               "end": res.months[-1], "alpha_annual": res.alpha_annual,
               "alpha_t": res.alpha_t, "r2": res.r2}
        row.update({f"beta_{c}": res.betas[c] for c in fcols})
        row.update({f"t_{c}": res.tstats[c] for c in fcols})
        rows.append(row)
    return (pd.DataFrame(rows, columns=tcols),
            pd.DataFrame(skipped, columns=scols))
