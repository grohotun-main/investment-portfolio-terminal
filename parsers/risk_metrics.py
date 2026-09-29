"""
Pure risk math for the Phase 1 dashboard, extracted from app.py.

All functions are pure (monthly/daily return series in, scalar/Series out).
No file I/O, no Streamlit calls, no module-level side effects — which lets
tests/test_risk_metrics.py exercise them directly without booting Streamlit.

Sections:
  - Return summaries:   _return_stats
  - Risk-adjusted:      compute_sharpe, compute_sortino, compute_calmar,
                        compute_sortino_daily
  - Drawdowns:          compute_drawdown_episodes, window_drawdown_pct
  - Concentration:      compute_concentration
  - Benchmark series:   spy_monthly_returns_aligned, spy_value_at,
                        spy_decline_between, spy_months_underwater_from
  - Covariance:         _ewma_cov, _ledoit_wolf_shrinkage,
                        estimate_covariance
  - Daily ex-ante:      synthesize_portfolio_returns, compute_risk_contributions,
                        compute_downside_risk_contributions,
                        compute_es_contributions
  - Diversification:    compute_dr_time_series, compute_max_dr,
                        classify_dr_regime, compute_dr_ratio_series,
                        compute_dr_regime_thresholds
  - Regime conditioning: classify_market_regime,
                        compute_regime_conditional_dr, interpret_regime_dr
  - Beta / alpha:       _aligned, compute_beta, compute_alpha_annual,
                        compute_up_down_beta, rolling_beta,
                        rolling_alpha_annual, rolling_up_down_beta
  - Tail risk:          compute_var_cvar
  - Period aggregation: aggregate_periodic_returns
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _return_stats(monthly_returns: pd.Series) -> tuple[float, float, float, int]:
    """Return (cagr, ann_vol, ann_downside_vol, n_months) for a monthly series.

    Downside vol uses the textbook Sortino-Bawa definition:
        dvol = sqrt(mean(min(0, r)^2)) * sqrt(12)
    averaged over ALL observations (positives contribute 0, negatives
    contribute r^2). MAR = 0. Matches Excel SORTINO and standard tooling.
    """
    r = monthly_returns.dropna().astype(float)
    n = int(len(r))
    if n < 2:
        return np.nan, np.nan, np.nan, n
    cagr = float((1.0 + r).prod() ** (12.0 / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(12))
    dev_below = np.minimum(r.values, 0.0)
    ann_dvol = float(np.sqrt(float((dev_below ** 2).mean())) * np.sqrt(12))
    return cagr, ann_vol, ann_dvol, n


def _window_rf(rf_input: float | pd.Series, returns: pd.Series) -> float:
    """Resolve a scalar mean RF over the dates of `returns`.

    Accepts either:
      - a `float` (legacy / fallback) — returned as-is, no alignment.
      - a date-indexed `pd.Series` of decimal annualized rates (e.g. the
        FRED DGS3MO series from data/risk_free_rate.csv).

    For the Series case, one RF reading is sampled per return observation
    (forward-filled from the prior business day so month-end and weekend
    return dates inherit the most recent published rate), then averaged.
    A return date that precedes the RF series start back-fills to the
    earliest available rate — gracefully degrades for ancient history.
    Returns NaN when neither input gives any usable observation.

    **Staleness note (Phase 1C audit).** The ffill across the trailing
    edge means a stale RF series silently propagates the most-recent
    published rate forward, with no NaN-poisoning to signal the issue.
    Callers that want to surface stale-RF risk should consult
    `rf_staleness_business_days` and warn the user when the lag exceeds
    a few business days. FRED's DGS3MO settles ~1 biz-day in arrears,
    so a 1-2d lag is normal; 5+ days suggests the fetcher broke.
    """
    if isinstance(rf_input, (int, float, np.floating, np.integer)):
        return float(rf_input)
    if not isinstance(rf_input, pd.Series) or rf_input.empty or returns.empty:
        return float("nan")
    rets_idx = pd.DatetimeIndex(returns.dropna().index)
    rf = rf_input.sort_index().dropna()
    if rf.empty or rets_idx.empty:
        return float("nan")
    union = rf.index.union(rets_idx).sort_values()
    sampled = rf.reindex(union).ffill().bfill().loc[rets_idx].dropna()
    return float(sampled.mean()) if not sampled.empty else float("nan")


def rf_staleness_business_days(
    data_dir: "os.PathLike | str",
    today: pd.Timestamp | None = None,
) -> int | None:
    """Business-day gap between today and the last published RF reading.

    Reads the max date directly from `data_dir/risk_free_rate.csv` and
    returns the number of weekdays the series is behind `today` (or
    `pd.Timestamp.today()` if not supplied). Positive means the RF file
    is stale — Sharpe / Sortino tiles are silently consuming a
    forward-filled old DGS3MO rate. FRED publishes one biz-day in
    arrears, so a 1-day lag is normal; 5+ days suggests the fetcher
    broke or hasn't been run.

    None when the file is missing, unreadable, or empty. Mirrors the
    `bench_tr_staleness_days` shape in parsers/refresh_prices.py.
    """
    from pathlib import Path  # local import keeps the module light
    rf_csv = Path(data_dir) / "risk_free_rate.csv"
    if not rf_csv.exists():
        return None
    try:
        dates = pd.to_datetime(
            pd.read_csv(rf_csv, usecols=["date"])["date"], errors="coerce",
        ).dropna()
    except (ValueError, KeyError):
        return None
    if dates.empty:
        return None
    last = dates.max()
    now = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    # bdate_range covers all weekdays in [start, end] inclusive. Start the
    # day after `last`, end on today's normalized date; len is the count
    # of weekdays strictly between.
    gap = pd.bdate_range(start=last + pd.Timedelta(days=1),
                          end=now.normalize())
    return int(len(gap))


def compute_sharpe(monthly_returns: pd.Series,
                   rf_annual: float | pd.Series) -> float:
    cagr, ann_vol, _, _ = _return_stats(monthly_returns)
    if not (np.isfinite(cagr) and np.isfinite(ann_vol) and ann_vol > 0):
        return np.nan
    rf = _window_rf(rf_annual, monthly_returns)
    if not np.isfinite(rf):
        return np.nan
    return (cagr - rf) / ann_vol


def compute_sortino(monthly_returns: pd.Series,
                    rf_annual: float | pd.Series) -> float:
    """Annualized Sortino: (CAGR − rf) / ann_downside_vol(MAR=0).

    Mixed-MAR convention by design — matches Excel / Bloomberg SORTINO:
    numerator uses the risk-free rate (Sharpe-style), denominator uses
    `sqrt(mean(min(r, 0)²)) × √12` (downside vol relative to MAR = 0,
    not MAR = rf). Strict textbook Sortino requires MAR consistency
    on both sides; the asymmetry here is intentional and locked by
    test (see tests/test_risk_metrics.py:TestSortino). A maintainer
    "fixing" the numerator to (CAGR − 0) would shift the displayed
    Sortino by ~rf / dvol — roughly 0.4-0.7 ratio units at today's
    RF ≈ 3.7% and 5-8% downside vol.
    """
    cagr, _, ann_dvol, _ = _return_stats(monthly_returns)
    if not (np.isfinite(cagr) and np.isfinite(ann_dvol) and ann_dvol > 0):
        return np.nan
    rf = _window_rf(rf_annual, monthly_returns)
    if not np.isfinite(rf):
        return np.nan
    return (cagr - rf) / ann_dvol


def window_drawdown_pct(r_window: pd.Series) -> pd.Series:
    """Drawdown percent series for a return window, anchored at 1.0.

    Anchors the running peak at the pre-window baseline of 1.0 — so a
    worst observation at the start of the window measures a real
    drawdown from "before the window began," not 0% (which is what a
    naive `(1+r).cumprod().cummax()` produces, since cummax sets
    peak = w[0] = 1+r₀).

    Returns a percent series at the same index as r_window. The
    Phase 1C audit caught the unanchored convention on the dashboard's
    Max DD (1Y / 3Y) and Calmar (3Y) tiles; the inline copy in app.py
    is the consumer of this function.
    """
    if r_window.empty:
        return pd.Series([], dtype=float)
    w = (1.0 + r_window).cumprod()
    peak = w.cummax().clip(lower=1.0)
    return (w / peak - 1.0) * 100.0


_CALMAR_MIN_DD_PCT = 0.01  # 1bp — see compute_calmar docstring


def compute_calmar(monthly_returns: pd.Series, dd_pct_window: pd.Series) -> float:
    """CAGR / |Max DD| over the same lookback window. dd_pct_window in percent.

    Returns NaN when CAGR is non-positive — Calmar is defined as "return
    per unit of drawdown risk taken," which loses its interpretive frame
    when the return itself is a loss (a number like -0.5 reads as "you
    lost half your max-DD per year" but the comparison to a positive
    Calmar isn't meaningful). The dashboard's how-to-read prose says
    "higher is better"; a negative Calmar would invite reading a SPY
    Calmar of -0.10 vs a portfolio Calmar of -0.50 as "portfolio
    underperforming by 0.4," which the math doesn't support.

    Also returns NaN when |max DD| is below the 1bp epsilon
    `_CALMAR_MIN_DD_PCT`. Without this guard a window with a single
    -0.001% wobble produces Calmar = CAGR / ~1e-5 ≈ 5,000+, which reads
    as "extraordinary risk-adjusted return" but is just division by
    rounding noise. PR #65 added the symmetric guard against negative
    CAGR; this is the tiny-denominator counterpart. The 1bp threshold
    sits well below the realistic drawdown noise floor for any
    multi-asset portfolio over a typical Calmar window (3Y monthly).
    """
    cagr, _, _, _ = _return_stats(monthly_returns)
    if dd_pct_window.empty:
        return np.nan
    max_dd = float(dd_pct_window.min())
    if not (np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0):
        return np.nan
    if cagr <= 0:
        return np.nan
    if abs(max_dd) < _CALMAR_MIN_DD_PCT:
        return np.nan
    return cagr / abs(max_dd / 100.0)


def compute_sortino_daily(daily_returns: pd.Series,
                          rf_annual: float | pd.Series = 0.0) -> float:
    """Annualized Sortino on a daily return series.

    Textbook Sortino-Bawa with MAR = 0 and √252 annualization. Matches the
    monthly compute_sortino formula (√12) but rescaled to daily resolution.

    **No live UI consumer.** This function was previously rendered on the
    Risk Contribution tab; PR #43 removed it because a daily-resolution
    Sortino on the same window structurally differs from the monthly-
    resolution Sortino on the Risk Overview tab — having both visible
    led to two different "Sortino" numbers labeled the same on adjacent
    tabs (see project_dashboard_design memory). Kept here for
    cross-resolution analysis tools, NOT for dashboard tiles. A future
    contributor adding this back to a tile must also surface the
    resolution distinction so the Risk Overview / Contribution tabs
    don't display the same label on different math.
    """
    r = daily_returns.dropna().astype(float)
    n = int(len(r))
    if n < 2:
        return np.nan
    cagr = float((1.0 + r).prod() ** (252.0 / n) - 1.0)
    dev_below = np.minimum(r.values, 0.0)
    ann_dvol = float(np.sqrt(float((dev_below ** 2).mean())) * np.sqrt(252))
    if not (np.isfinite(cagr) and np.isfinite(ann_dvol) and ann_dvol > 0):
        return np.nan
    rf = _window_rf(rf_annual, daily_returns)
    if not np.isfinite(rf):
        return np.nan
    return (cagr - rf) / ann_dvol


def compute_drawdown_episodes(wealth: pd.Series, dates: pd.Series) -> list[dict]:
    """Peak-to-trough-to-recovery episodes on a wealth index.

    A new peak (wealth >= running peak) closes any open episode at recovery.
    Final episode stays open with recovery_date=None if not recovered yet.
    Depth in percent.
    """
    if len(wealth) < 2:
        return []
    w = wealth.reset_index(drop=True).astype(float)
    d = dates.reset_index(drop=True)
    episodes: list[dict] = []
    peak = float(w.iloc[0])
    peak_idx = 0
    trough = peak
    trough_idx = 0
    in_dd = False
    for i in range(1, len(w)):
        v = float(w.iloc[i])
        if v >= peak:
            if in_dd:
                episodes.append({
                    "peak_date":   d.iloc[peak_idx],
                    "trough_date": d.iloc[trough_idx],
                    "recovery_date": d.iloc[i],
                    "depth_pct":   (trough / peak - 1.0) * 100.0,
                    "peak_to_trough_months": int(trough_idx - peak_idx),
                    "recovery_months":       int(i - trough_idx),
                })
                in_dd = False
            peak = v
            peak_idx = i
            trough = v
            trough_idx = i
        elif v < trough:
            trough = v
            trough_idx = i
            in_dd = True
    if in_dd:
        episodes.append({
            "peak_date":   d.iloc[peak_idx],
            "trough_date": d.iloc[trough_idx],
            "recovery_date": None,
            "depth_pct":   (trough / peak - 1.0) * 100.0,
            "peak_to_trough_months": int(trough_idx - peak_idx),
            "recovery_months":       None,
        })
    return episodes


def compute_concentration(market_values: pd.Series) -> dict:
    """Top-N weights, max weight, and effective number of bets.

    Input is per-position market value (already aggregated by symbol).
    Non-positive values are dropped (short options, etc.).
    """
    mv = market_values[market_values > 0].astype(float)
    total = float(mv.sum())
    if total <= 0 or mv.empty:
        return {"top5_pct": np.nan, "top10_pct": np.nan,
                "max_pct": np.nan, "effective_n": np.nan, "n_positions": 0}
    w = (mv / total).sort_values(ascending=False)
    return {
        "top5_pct":    float(w.head(5).sum() * 100.0),
        "top10_pct":   float(w.head(10).sum() * 100.0),
        "max_pct":     float(w.iloc[0] * 100.0),
        "effective_n": float(1.0 / (w * w).sum()),
        "n_positions": int(len(w)),
    }


def spy_monthly_returns_aligned(twr_portfolio: pd.DataFrame,
                                bench_tr: pd.Series) -> pd.Series:
    """SPY monthly TR returns aligned to the portfolio's statement_date
    boundaries. Index = month_end Timestamp. Months where either boundary
    falls outside the benchmark range (or has no prev_stmt_date) are dropped.

    Mirrors the boundary logic in build_twr_comparison so the SPY series used
    for Risk-tab metrics matches the SPY series used in the Performance vs
    benchmark tab — same windows, same TR convention.
    """
    if twr_portfolio.empty or bench_tr.empty:
        return pd.Series(dtype=float)
    bench_start, bench_end = bench_tr.index.min(), bench_tr.index.max()
    out: list[tuple[pd.Timestamp, float]] = []
    for _, p in twr_portfolio.sort_values("statement_date").iterrows():
        prev = p.get("prev_stmt_date")
        end = p["statement_date"]
        if pd.isna(prev) or pd.isna(end):
            continue
        if prev < bench_start or end > bench_end:
            continue
        ret = float(bench_tr.loc[end] / bench_tr.loc[prev] - 1.0)
        out.append((pd.Timestamp(end).normalize(), ret))
    if not out:
        return pd.Series(dtype=float)
    idx, vals = zip(*out)
    return pd.Series(list(vals), index=pd.DatetimeIndex(idx), name="spy_return")


def spy_value_at(bench_tr: pd.Series,
                 date: pd.Timestamp) -> float:
    """SPY TR value at the given date, using the last known close on or
    before that date (handles non-trading days). NaN if outside range."""
    if bench_tr.empty:
        return np.nan
    ts = pd.Timestamp(date)
    eligible = bench_tr.index <= ts
    if not eligible.any():
        return np.nan
    return float(bench_tr.loc[bench_tr.index[eligible][-1]])


def spy_decline_between(bench_tr: pd.Series,
                        peak_date: pd.Timestamp,
                        trough_date: pd.Timestamp) -> float:
    """SPY's percentage move from peak_date to trough_date (negative if SPY
    fell over the portfolio's drawdown window). Used by the Top-3 drawdown
    episodes table."""
    p = spy_value_at(bench_tr, peak_date)
    t = spy_value_at(bench_tr, trough_date)
    if not (np.isfinite(p) and np.isfinite(t) and p > 0):
        return np.nan
    return float((t / p - 1.0) * 100.0)


def spy_months_underwater_from(bench_tr: pd.Series,
                               peak_date: pd.Timestamp,
                               trough_date: pd.Timestamp | None = None,
                               lookahead_days: int = 90) -> int | None:
    """How many calendar months SPY itself stayed underwater during the
    drawdown episode anchored at `peak_date`.

    SPY's local peak is taken as the MAX value over
    [peak_date, peak_date + lookahead_days] — usually SPY peaked a few
    days/weeks away from the portfolio peak (e.g. SPY's Jan 3 2022 ATH was
    three trading days after a portfolio peak dated Dec 31 2021). If we
    naively use SPY's value AT the portfolio peak date instead, a tiny tick
    up in the following days falsely registers as "recovered in 1 month",
    when SPY actually didn't revisit its early-2022 high until 2024.

    The trough_date is intentionally NOT used to cap the local-peak window:
    on fast drawdowns the SPY local peak can sit *after* the portfolio
    trough (SPY's drawdown lags by a few days), and clipping at trough_date
    would silently miss that peak and undercount SPY's months-underwater.

    **Known limitation (Phase 1D).** The 90-day lookahead is sized for
    typical SPY-vs-portfolio lag (days to weeks). If SPY's true peak lags
    the portfolio peak by more than 90 calendar days (rare — requires
    portfolio-specific assets that lead SPY by that margin), this
    function silently misses it and reports a too-early local peak; no
    diagnostic is emitted. Acceptable for current portfolios but worth
    revisiting if a low-correlation sleeve grows materially.

    Returns None if SPY hasn't recovered to its local peak yet.
    """
    pk = pd.Timestamp(peak_date)
    if bench_tr.empty:
        return None
    window_end = pk + pd.Timedelta(days=lookahead_days)
    # `trough_date` parameter kept for back-compat (now unused).
    _ = trough_date
    window = bench_tr[(bench_tr.index >= pk) & (bench_tr.index <= window_end)]
    if window.empty:
        return None
    spy_peak_date = window.idxmax()
    spy_peak_val  = float(window.max())
    fwd = bench_tr[bench_tr.index > spy_peak_date]
    recovered = fwd[fwd >= spy_peak_val]
    if recovered.empty:
        return None
    recovery_date = recovered.index[0]
    months = (recovery_date.year - spy_peak_date.year) * 12 + \
             (recovery_date.month - spy_peak_date.month)
    return max(int(months), 0)


def synthesize_portfolio_returns(
    weights: pd.Series, daily_prices: pd.DataFrame
) -> pd.Series:
    """Ex-ante daily portfolio returns from current weights × per-asset
    daily returns. Sum is over symbols with both a weight and price data;
    weights are renormalized to that universe.

    Missing per-asset returns on a date contribute zero (the holding
    "moved 0% that day"). For symbols with sparse price coverage this
    biases per-asset vol toward zero and tilts portfolio risk contribution
    toward the symbols with continuous coverage. Use
    `compute_synthesis_gaps(weights, daily_prices)` to surface which
    symbols are losing data this way.

    This is the *static* synthesis — today's weights projected over the
    historical daily-price window. Use it when the question is "what would
    today's portfolio have done in past conditions". For honest historical
    performance (rolling Sharpe / Sortino / vol of the actual portfolio
    the user held over time) use synthesize_portfolio_returns_historical
    instead."""
    common = [s for s in weights.index if s in daily_prices.columns]
    if not common:
        return pd.Series([], dtype=float)
    w = weights[common].astype(float)
    if w.sum() <= 0:
        return pd.Series([], dtype=float)
    w = w / w.sum()
    rets = daily_prices[common].pct_change().fillna(0.0)
    return (rets * w).sum(axis=1).iloc[1:]


def weights_per_snap_monthly(positions: pd.DataFrame, build_weights) -> dict:
    """Build one weight snapshot per CALENDAR MONTH for the historical daily
    synthesis, keyed by the month's latest ``statement_date``.

    A single month-end snapshot is often split across two filing dates — Harbor
    stamps the last business day, Alpine stamps month-end — so keying the
    synthesis by exact statement_date yields two one-broker snapshots per
    dual-date month, each owning a sub-segment of the daily return series with
    only half the portfolio's weights (WSF-1: ~30% of the post-2022 series ran
    on one-broker weights, e.g. March-2026 synth vol 9.71% vs 24.15% unified).
    Coalescing by calendar month folds both brokers — and any carried-forward
    accounts, when ``positions`` is a ``monthly_normalize`` frame — into one
    snapshot per month.

    ``build_weights(month_rows) -> weights Series`` maps a month's position
    rows to a normalized weight Series (return an empty Series to skip the
    month). The result feeds ``synthesize_portfolio_returns_historical``.
    """
    out: dict = {}
    if positions.empty or "statement_date" not in positions.columns:
        return out
    months = positions["statement_date"].dt.to_period("M")
    for _period, snap_m in positions.groupby(months):
        w = build_weights(snap_m)
        if w is not None and not w.empty:
            out[pd.Timestamp(snap_m["statement_date"].max())] = w
    return out


def synthesize_portfolio_returns_historical(
    weights_per_snapshot: dict, daily_prices: pd.DataFrame,
) -> pd.Series:
    """Honest daily portfolio returns using per-statement-date weights.

    `weights_per_snapshot` maps statement_date Timestamp -> weights Series
    (normalized to 1.0 over that snapshot's universe). For each snapshot,
    the daily returns from (statement_date, next_statement_date] are
    computed using that snapshot's weights; the trailing tail after the
    last snapshot uses the latest snapshot's weights.

    Segments are concatenated chronologically. Within each segment, the
    same NaN-as-zero / renormalize-to-available-universe rules as
    `synthesize_portfolio_returns` apply — so per-snapshot symbol coverage
    edges (a holding entering or leaving the portfolio between snapshots)
    are handled cleanly.

    Use this over the static synthesis when you want the portfolio's
    actual realized daily returns — e.g. rolling Sharpe, daily Sortino,
    historical drawdowns. The static synthesis is biased when current
    weights differ materially from past weights (post-rebalance, after
    derisking, after a position split): it projects today's snapshot
    backwards, masking the volatility the portfolio actually experienced.
    """
    if not weights_per_snapshot or daily_prices.empty:
        return pd.Series([], dtype=float)
    stmt_dates = sorted(weights_per_snapshot.keys())
    daily_rets = daily_prices.pct_change()
    segments: list[pd.Series] = []
    for i, stmt_d in enumerate(stmt_dates):
        weights = weights_per_snapshot[stmt_d]
        if weights is None or weights.empty:
            continue
        if i + 1 < len(stmt_dates):
            next_d = stmt_dates[i + 1]
            mask = (daily_rets.index > stmt_d) & (daily_rets.index <= next_d)
        else:
            mask = daily_rets.index > stmt_d
        period_rets = daily_rets.loc[mask]
        if period_rets.empty:
            continue
        common = [s for s in weights.index if s in period_rets.columns]
        if not common:
            continue
        w = weights[common].astype(float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        seg = (period_rets[common].fillna(0.0) * w).sum(axis=1)
        segments.append(seg)
    if not segments:
        return pd.Series([], dtype=float)
    return pd.concat(segments).sort_index()


def compute_synthesis_gaps(
    weights: pd.Series, daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnostic for synthesize_portfolio_returns' NaN-as-zero treatment.

    Returns a DataFrame indexed by symbol with columns:
      weight_pct       — normalized weight in the synthesis universe
      n_days_total     — daily_prices rows available in the window
      n_days_no_price  — rows where price was NaN (silently treated as 0%)
      pct_no_price     — n_days_no_price / n_days_total

    Surface in the UI alongside the Risk tiles so the user can see when
    sparse-coverage symbols are pulling portfolio vol toward zero."""
    common = [s for s in weights.index if s in daily_prices.columns]
    if not common or daily_prices.empty:
        return pd.DataFrame(columns=[
            "weight_pct", "n_days_total", "n_days_no_price", "pct_no_price",
        ])
    w = weights[common].astype(float)
    w = w / w.sum() if w.sum() > 0 else w
    sub = daily_prices[common]
    n_total = len(sub)
    n_nan = sub.isna().sum()
    rows = []
    for sym in common:
        rows.append({
            "symbol": sym,
            "weight_pct": float(w[sym] * 100),
            "n_days_total": int(n_total),
            "n_days_no_price": int(n_nan[sym]),
            "pct_no_price": float(n_nan[sym] / n_total * 100)
                            if n_total else 0.0,
        })
    return (pd.DataFrame(rows).set_index("symbol")
                              .sort_values("pct_no_price", ascending=False))


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------
# Three estimators feed compute_risk_contributions and its downside variant:
#
#   rolling   — sample covariance on the trailing `window` days. Legacy
#               approach; equal weight per observation. Slow to react to
#               regime shifts (a vol spike from 6 months ago weighs the
#               same as one from yesterday) and exhibits "ghost effects"
#               when large returns drop out of the window.
#
#   ewma      — RiskMetrics 1996 exponential weighting with decay factor λ
#               (default 0.94 → half-life ≈ 11 trading days, 1%-of-weight
#               horizon ≈ 75 days). Most recent observation weighted
#               highest; older observations decay geometrically. No
#               demeaning — RiskMetrics assumes daily mean ≈ 0 for
#               equities. Faster regime response (days, not months) with
#               no ghost effects from window boundaries.
#
#   ewma_lw   — EWMA followed by Ledoit-Wolf constant-correlation shrinkage
#               (Ledoit & Wolf 2004, "Honey, I Shrunk the Sample
#               Covariance Matrix"). Stabilizes the matrix when N is large
#               relative to T_effective (under λ=0.94 the effective sample
#               size is ~16-17 observations, which is small for ~25
#               positions). Default. Returns the shrinkage intensity α so
#               the UI can surface how much regularization is being
#               applied.
#
# Input cap: EWMA weights decay below 1% after ~75 observations under
# λ=0.94, so feeding >504 days of history is wasted compute. The cap
# applies to "ewma" / "ewma_lw" only; "rolling" still honors `window`.


def _ewma_cov(rets: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics 1996 EWMA covariance (zero-mean assumption).

    Computes Σ = Σ_t w_t · r_t · r_tᵀ where weights decay exponentially
    backwards in time: the newest observation gets weight (1-λ), the next
    (1-λ)λ, then (1-λ)λ², and so on. Weights are normalized so they sum
    to 1, then absorbed into the returns as √w_t · r_t before the outer
    product — that lets the whole covariance be one matrix multiply
    instead of T outer products.

    No demeaning. RiskMetrics treats daily equity returns as having mean
    ≈ 0 over the relevant horizon, which is a good approximation: the
    daily mean of SPY is ~3-4 bps versus a daily std of ~100 bps.

    Returns daily-scale covariance; the caller annualizes by × 252.
    """
    R = rets.values.astype(float)
    T, N = R.shape
    if T == 0 or N == 0:
        return pd.DataFrame(np.zeros((N, N)),
                            index=rets.columns, columns=rets.columns)
    # Newest observation (index T-1) gets the highest weight.
    weights = lam ** np.arange(T - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    Rw = R * np.sqrt(weights)[:, None]
    cov_mat = Rw.T @ Rw
    return pd.DataFrame(cov_mat, index=rets.columns, columns=rets.columns)


def _ledoit_wolf_shrinkage(
    rets: pd.DataFrame,
    cov_input: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) constant-correlation shrinkage.

    Pulls `cov_input` toward a structured target F whose off-diagonals
    share a single correlation r̄ (the average of off-diagonal sample
    correlations derived from `cov_input`). The shrinkage intensity α*
    is chosen by Ledoit & Wolf's optimal MSE formula:

        α* = max(0, min(κ/T, 1)),  κ = (π̂ − ρ̂) / γ̂

    where π̂ estimates the asymptotic variance of the sample-cov entries,
    ρ̂ the asymptotic covariance between sample-cov and target entries,
    and γ̂ = ‖F − S‖²_F. When κ/T is large (noisy estimate) α* → 1 and
    the result is dominated by the structured target; when κ/T is small
    (well-conditioned estimate) α* → 0 and the result is the input
    covariance.

    Inputs: `rets` is the T × N return frame used to estimate `cov_input`
    (needed for the asymptotic variance computations); `cov_input` is the
    N × N covariance to shrink (sample cov or EWMA cov).

    Returns (S_shrunk, α) where S_shrunk = αF + (1−α)·cov_input.
    """
    S = cov_input.values.astype(float)
    R = rets.values.astype(float)
    T, N = R.shape
    if N < 2 or T < 2:
        return cov_input.copy(), 0.0

    # Detect constant-variance assets (could be e.g. a bond marked at par for
    # an entire window). Pass the input through unchanged rather than silently
    # clipping their variance to 1e-18 — the previous floor was a hidden form
    # of the same "fabricate data to keep the math going" that the no-clipping
    # policy is meant to prevent.
    #
    # Phase 1C audit hardening: the guard was exact `<= 0`, which lets a
    # positive-but-microscopic variance (e.g. 1e-30 from a near-flat asset)
    # through. Downstream, `corr = S / (std·std)` then divides by a
    # microscopic std on that asset's row/col and produces correlations of
    # absurd magnitude that propagate into r_bar and the shrunk matrix.
    # Floor the guard at a small fraction of the median variance.
    diag = np.diag(S).copy()
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
        return cov_input.copy(), 0.0
    eps_var = max(1e-18, 1e-12 * float(np.median(diag)))
    if np.any(diag < eps_var):
        return cov_input.copy(), 0.0

    Y = R - R.mean(axis=0)
    var = diag
    std = np.sqrt(var)

    # Sample correlation derived from cov_input (NOT a separate sample
    # correlation matrix on Y — the shrinkage formula needs consistency
    # between S and the derived correlations).
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    upper = np.triu_indices(N, k=1)
    r_bar = float(corr[upper].mean()) if upper[0].size else 0.0

    # Constant-correlation target: F_ii = S_ii; F_ij = r̄ · √(S_ii · S_jj).
    F = r_bar * np.outer(std, std)
    np.fill_diagonal(F, var)

    # π̂_ij = mean_t (Y_ti · Y_tj − S_ij)². Loop over t to keep memory at
    # O(N²) instead of O(T·N²). N≈25 on this portfolio, so T iterations
    # of an N×N outer product is cheap.
    pi_mat = np.zeros((N, N))
    for t in range(T):
        outer_t = np.outer(Y[t], Y[t])
        pi_mat += (outer_t - S) ** 2
    pi_mat /= T
    pi_hat = float(pi_mat.sum())

    # ρ̂ = Σ_i π̂_ii + Σ_{i ≠ j} (r̄/2) · [√(S_jj/S_ii)·θ_{ii,ij}
    #                                    + √(S_ii/S_jj)·θ_{jj,ij}]
    # where θ_{ii,ij} = mean_t (Y_ti² − S_ii)(Y_ti·Y_tj − S_ij) and
    # likewise for θ_{jj,ij}. The diagonal sum is just π̂_ii.
    rho_hat = float(np.diag(pi_mat).sum())
    Y2 = Y ** 2
    for i in range(N):
        s_ii = var[i]
        for j in range(N):
            if i == j:
                continue
            s_jj = var[j]
            s_ij = S[i, j]
            cross = Y[:, i] * Y[:, j] - s_ij
            theta_ii = float(((Y2[:, i] - s_ii) * cross).mean())
            theta_jj = float(((Y2[:, j] - s_jj) * cross).mean())
            rho_hat += (r_bar / 2.0) * (
                np.sqrt(s_jj / s_ii) * theta_ii
                + np.sqrt(s_ii / s_jj) * theta_jj
            )

    gamma_hat = float(((F - S) ** 2).sum())
    if gamma_hat <= 0:
        # F == S (degenerate: single asset, or all-equal variances+
        # correlations); no useful shrinkage direction.
        return cov_input.copy(), 0.0

    kappa = (pi_hat - rho_hat) / gamma_hat
    alpha = float(max(0.0, min(kappa / T, 1.0)))

    S_shrunk = alpha * F + (1.0 - alpha) * S
    # Phase 1C audit: bail to the unshrunk input if the shrunk matrix
    # contains any NaN / inf entries. The no-clip discipline is a
    # promise to never fabricate values; a NaN-laced output silently
    # poisoning the downstream Risk-Contribution table is the worst
    # form of fabrication.
    if not np.all(np.isfinite(S_shrunk)):
        return cov_input.copy(), 0.0
    return (pd.DataFrame(S_shrunk, index=cov_input.index,
                         columns=cov_input.columns), alpha)


def estimate_covariance(
    rets: pd.DataFrame,
    estimator: str = "ewma_lw",
    window: int = 252,
    lambda_param: float = 0.94,
    cap_days: int = 504,
    annualize_factor: float = 252.0,
) -> dict:
    """Unified covariance estimator with annualization.

    Supported estimators:
      - "rolling"  — sample covariance over the trailing `window` days
      - "ewma"     — RiskMetrics EWMA, λ = `lambda_param` (default 0.94),
                     input capped at `cap_days` (default 504)
      - "ewma_lw"  — EWMA followed by Ledoit-Wolf constant-correlation
                     shrinkage. Default.

    Returns:
      {
        "cov":       pd.DataFrame (N × N, annualized by `annualize_factor`),
        "n_days":    int (observations actually used),
        "alpha":     float | None (LW shrinkage intensity; None unless _lw),
        "estimator": str (echoed),
        "lambda":    float | None (echoed for ewma variants; None otherwise),
      }

    `rets` is the daily return frame; the function applies the right input
    cap per estimator semantics (window for rolling, cap_days for EWMA).
    Empty / degenerate inputs return zero-vol covariance.

    BASIS NOTE: callers pass SIMPLE returns (``pct_change``); the returned
    covariance is on the simple-return basis. If you need correlations
    consistent with this covariance, derive them in place from Σ:
    C[i,j] = Σ[i,j] / sqrt(Σ[i,i] · Σ[j,j]). Do NOT pair the output with
    ``compute_correlation_matrix`` (which uses LOG returns for display
    stability) — mixing bases produces inconsistent marginal-risk
    attribution. The marginal-risk decomposition functions in this module
    (``compute_risk_contributions`` and variants) already use only this
    covariance and never call ``compute_correlation_matrix``; tests in
    ``tests/test_log_vs_simple_returns_boundary.py`` lock that invariant.
    """
    if estimator not in ("rolling", "ewma", "ewma_lw"):
        raise ValueError(f"Unknown estimator: {estimator!r}")

    if estimator == "rolling":
        rets_w = rets.tail(window) if len(rets) > window else rets
        cov_daily = rets_w.cov()
        return {
            "cov":       cov_daily * annualize_factor,
            "n_days":    int(len(rets_w)),
            "alpha":     None,
            "estimator": "rolling",
            "lambda":    None,
        }

    # EWMA variants.
    rets_w = rets.tail(cap_days) if len(rets) > cap_days else rets
    cov_daily = _ewma_cov(rets_w, lam=lambda_param)
    alpha: float | None = None
    if estimator == "ewma_lw":
        cov_daily, alpha = _ledoit_wolf_shrinkage(rets_w, cov_daily)
    return {
        "cov":       cov_daily * annualize_factor,
        "n_days":    int(len(rets_w)),
        "alpha":     alpha,
        "estimator": estimator,
        "lambda":    float(lambda_param),
    }


def series_vol_ann(
    rets: pd.Series,
    estimator: str = "ewma_lw",
    window: int = 252,
    lambda_param: float = 0.94,
    cap_days: int = 504,
) -> float:
    """Annualized volatility of a single return series under `estimator`.

    The one-asset specialization of :func:`estimate_covariance`, so a
    benchmark's vol can be computed with the SAME estimator the portfolio
    tile uses — keeping the colored better/worse delta apples-to-apples
    (audit WSB-2). Returns NaN on an empty series.
    """
    s = pd.Series(rets).dropna()
    if s.empty:
        return float("nan")
    cov = estimate_covariance(
        s.to_frame("_x"), estimator=estimator, window=window,
        lambda_param=lambda_param, cap_days=cap_days,
    )["cov"]
    return float(np.sqrt(cov.iloc[0, 0]))


def compute_risk_contributions(
    weights: pd.Series,
    daily_prices: pd.DataFrame,
    window: int = 252,
    estimator: str = "ewma_lw",
    lambda_param: float = 0.94,
    cap_days: int = 504,
) -> dict:
    """Per-position risk decomposition over a trailing daily window.

    Decomposes the daily-synthesis portfolio's annualized vol into its
    per-symbol component pieces using the standard formulation:

        σ²_p   = wᵀ Σ w
        MCTR_i = (Σ w)_i / σ_p              (vol per unit weight)
        CCTR_i = w_i · MCTR_i               (Σᵢ CCTR_i = σ_p exactly)
        PCTR_i = CCTR_i / σ_p × 100         (sums to 100%)

    All vol numbers are annualized × √252 so they line up with the Vol
    tile in the Risk tab. Σ is produced by `estimate_covariance` —
    estimator defaults to EWMA + Ledoit-Wolf shrinkage (λ=0.94, input
    capped at 504d); pass `estimator="rolling"` to get the legacy 252d
    sample covariance.

    Inputs mirror synthesize_portfolio_returns: weights from
    build_risk_series_bundle (already renormalized, with SGOV/TLH folding
    and cash/options dropped); daily_prices is the full daily price frame.
    Symbols in `weights` but missing from `daily_prices.columns` are
    dropped and weights renormalized to the common universe — same
    contract Pass 2 vol/VaR uses, so PCTR sums match the Vol total under
    the "rolling" estimator. Under EWMA/EWMA-LW the totals diverge from
    the rolling-std Vol tile (different estimator) — the page header
    surfaces the active estimator so the user knows.

    **As-of contract (Phase 1E).** This function reads the LAST
    `window` (or `cap_days` for EWMA) rows of `daily_prices`. It does
    NOT slice by any user-facing "report as of" date — the caller is
    responsible for truncating `daily_prices` if the report should reflect
    a historical regime instead of today's. The current dashboard
    intentionally uses today's daily_prices regardless of the sidebar
    "Holdings as of" selector (Risk Contribution tab caption discloses
    this); any future "render risk as of past date T" mode must pass
    `daily_prices.loc[:T]` rather than the full frame.

    **Window-sensitivity note (Phase 1E).** The displayed PCTR / DR /
    standalone vol all depend on `window` (rolling mode) or `cap_days`
    (EWMA modes). Different choices can shift PCTR by several percentage
    points and DR by 0.1–0.3 ratio units, especially across regime
    transitions where short windows react faster than long ones. The
    defaults (252d / 504d) are conventional 1y / 2y industry choices;
    they are *choices*, not "the right number." Callers showing a single
    estimate should treat it as a point reading under one window, not a
    distribution-free risk forecast.

    Returns a dict with:
      - per_symbol           DataFrame indexed by symbol, sorted by PCTR
                             desc, columns: weight, weight_pct,
                             standalone_vol_ann, n_obs_with_price (real
                             returns across full daily_prices history),
                             n_obs_in_window (real returns within the
                             trailing n_days window the cov actually
                             saw — use this for thin-history detection
                             under EWMA, not n_obs_with_price), mctr_ann,
                             cctr_ann, pctr_pct, diff_pp
      - port_vol_ann         portfolio annualized vol on this window
      - weighted_avg_vol_ann Σ wᵢ × σᵢ — the no-diversification baseline
      - dr                   diversification ratio = weighted_avg / port_vol
                             (1.0 = perfectly correlated, >1 = correlations
                             are reducing risk below the weighted-avg
                             standalone vol)
      - n_days               days of return history used
      - n_symbols            symbols in the decomposition
      - estimator            echoed estimator name
      - alpha                Ledoit-Wolf shrinkage intensity (None unless _lw)
      - lambda               EWMA decay factor (None for rolling)

    Empty / degenerate inputs return a dict with NaN scalars and an empty
    per_symbol DataFrame (callers should guard on n_symbols/n_days).
    """
    empty_df = pd.DataFrame(columns=[
        "weight", "weight_pct", "standalone_vol_ann",
        "n_obs_with_price", "n_obs_in_window",
        "mctr_ann", "cctr_ann", "pctr_pct", "diff_pp",
    ])
    empty = {
        "per_symbol":            empty_df,
        "port_vol_ann":          np.nan,
        "weighted_avg_vol_ann":  np.nan,
        "dr":                    np.nan,
        "n_days":                0,
        "n_symbols":             0,
        "estimator":             estimator,
        "alpha":                 None,
        "lambda":                None,
    }
    if weights is None or weights.empty or daily_prices.empty:
        return empty

    common = [s for s in weights.index if s in daily_prices.columns]
    if not common:
        return empty
    w = weights[common].astype(float)
    if w.sum() <= 0:
        return empty
    w = w / w.sum()

    # Daily returns on the common universe. fillna(0) mirrors
    # synthesize_portfolio_returns — missing-data days contribute 0%
    # rather than dropping the row pairwise. estimate_covariance handles
    # the input cap (window for rolling, cap_days for EWMA). We capture
    # the pre-fillna NaN-count per symbol so per_symbol.standalone_vol_ann
    # can be suppressed downstream for symbols with too-thin coverage
    # (those vols are dominated by the fillna(0) days and aren't
    # statistically meaningful).
    pct_chg = daily_prices[common].pct_change()
    n_obs_per_sym = pct_chg.notna().sum().astype(int)
    rets_all = pct_chg.fillna(0.0).iloc[1:]
    if int(len(rets_all)) < 20:
        return {**empty, "n_days": int(len(rets_all)),
                "n_symbols": int(len(common))}

    cov_info = estimate_covariance(
        rets_all, estimator=estimator, window=window,
        lambda_param=lambda_param, cap_days=cap_days,
        annualize_factor=252.0,
    )
    cov = cov_info["cov"]
    n_days = int(cov_info["n_days"])
    if n_days < 20:
        return {**empty, "n_days": n_days, "n_symbols": int(len(common))}
    w_arr = w.values
    var_p = float(w_arr @ cov.values @ w_arr)
    port_vol_ann = float(np.sqrt(var_p)) if var_p > 0 else 0.0

    # Per-symbol standalone vol on the same window.
    standalone_vol = pd.Series(
        np.sqrt(np.diag(cov.values)), index=cov.index, name="standalone_vol_ann"
    )
    weighted_avg_vol = float((w * standalone_vol).sum())
    # Align the obs count to the cov universe (estimate_covariance can
    # drop columns with degenerate variance; reindex defends against that).
    n_obs_aligned = n_obs_per_sym.reindex(cov.index).fillna(0).astype(int)
    # Phase 1C audit: window-relative n_obs catches symbols where fillna(0)
    # zero-padding dominates the EWMA cov even though total history passes
    # the absolute 60-obs floor. Count NaN-free pct_chg values WITHIN the
    # trailing n_days that estimate_covariance actually saw.
    pct_chg_window = pct_chg.iloc[1:].tail(n_days)
    n_obs_in_window = (
        pct_chg_window.notna().sum()
        .reindex(cov.index).fillna(0).astype(int)
    )

    if port_vol_ann <= 0:
        # Constant-return universe: no risk to decompose. Surface NaN MCTR/
        # PCTR so the UI can show "—" cleanly. Weights still populated.
        per_symbol = pd.DataFrame({
            "weight":             w,
            "weight_pct":         w * 100.0,
            "standalone_vol_ann": standalone_vol,
            "n_obs_with_price":   n_obs_aligned,
            "n_obs_in_window":    n_obs_in_window,
            "mctr_ann":           np.nan,
            "cctr_ann":           0.0,
            "pctr_pct":           np.nan,
            "diff_pp":            np.nan,
        })
        return {
            "per_symbol":           per_symbol,
            "port_vol_ann":         0.0,
            "weighted_avg_vol_ann": weighted_avg_vol,
            "dr":                   np.nan,
            "n_days":               n_days,
            "n_symbols":            int(len(common)),
            "estimator":            cov_info["estimator"],
            "alpha":                cov_info["alpha"],
            "lambda":               cov_info["lambda"],
        }

    mctr = (cov.values @ w_arr) / port_vol_ann
    cctr = w_arr * mctr  # Σ cctr == port_vol_ann by construction
    pctr_pct = cctr / port_vol_ann * 100.0
    diff_pp = pctr_pct - (w_arr * 100.0)

    per_symbol = pd.DataFrame({
        "weight":             w,
        "weight_pct":         w * 100.0,
        "standalone_vol_ann": standalone_vol,
        "n_obs_with_price":   n_obs_aligned,
        "n_obs_in_window":    n_obs_in_window,
        "mctr_ann":           pd.Series(mctr, index=cov.index),
        "cctr_ann":           pd.Series(cctr, index=cov.index),
        "pctr_pct":           pd.Series(pctr_pct, index=cov.index),
        "diff_pp":            pd.Series(diff_pp, index=cov.index),
    }).sort_values("pctr_pct", ascending=False)

    dr = weighted_avg_vol / port_vol_ann if port_vol_ann > 0 else np.nan

    return {
        "per_symbol":           per_symbol,
        "port_vol_ann":         port_vol_ann,
        "weighted_avg_vol_ann": weighted_avg_vol,
        "dr":                   float(dr) if np.isfinite(dr) else np.nan,
        "n_days":               n_days,
        "n_symbols":            int(len(common)),
        "estimator":            cov_info["estimator"],
        "alpha":                cov_info["alpha"],
        "lambda":               cov_info["lambda"],
    }


def compute_downside_risk_contributions(
    weights: pd.Series,
    daily_prices: pd.DataFrame,
    threshold: float = 0.0,
    window: int = 252,
    estimator: str = "ewma_lw",
    lambda_param: float = 0.94,
    cap_days: int = 504,
) -> dict:
    """Per-position risk decomposition restricted to days when the portfolio
    return was below `threshold`.

    Same algebra as compute_risk_contributions, but the covariance matrix is
    estimated only on stress days (port_ret ≤ threshold). Decomposes the
    downside-only portfolio vol into per-position pieces:

        σ²_p,down = wᵀ Σ_down w
        PCTR_down_i = w_i · (Σ_down w)_i / σ_p,down × 100   (sums to 100%)

    Threshold is on raw daily returns (e.g. 0.0 = anything below zero; -0.01
    = anything worse than -1%). Weights and price universe are normalized
    identically to compute_risk_contributions so per-symbol downside PCTR
    is directly comparable to total PCTR (same denominator semantics).

    Estimator semantics:
      - The outer window from which down-days are drawn matches
        compute_risk_contributions (window for rolling, cap_days for EWMA).
      - The cov on the down-day subset is sample covariance — EWMA
        time-weighting doesn't apply cleanly to a scattered subset of
        non-contiguous days. When `estimator="ewma_lw"`, Ledoit-Wolf
        shrinkage is still applied to that sample cov for stability,
        since N positions against typically <50 down days is the regime
        where shrinkage helps most.

    Returns the same dict shape as compute_risk_contributions with the keys
    renamed to *_down (port_vol_ann_down, weighted_avg_vol_ann_down, etc.),
    plus n_down_days and the same estimator/alpha/lambda triple. Empty /
    degenerate inputs return NaN scalars and an empty per_symbol_down
    DataFrame — callers should guard on n_down_days/n_symbols before
    reading numbers.
    """
    empty_df = pd.DataFrame(columns=[
        "weight", "weight_pct", "standalone_vol_ann_down",
        "mctr_ann_down", "cctr_ann_down", "pctr_pct_down",
    ])
    empty = {
        "per_symbol_down":            empty_df,
        "port_vol_ann_down":          np.nan,
        "weighted_avg_vol_ann_down":  np.nan,
        "dr_down":                    np.nan,
        "n_down_days":                0,
        "n_days_window":              0,
        "n_symbols":                  0,
        "threshold":                  float(threshold),
        "estimator":                  estimator,
        "alpha":                      None,
        "lambda":                     None,
    }
    if weights is None or weights.empty or daily_prices.empty:
        return empty
    if estimator not in ("rolling", "ewma", "ewma_lw"):
        raise ValueError(f"Unknown estimator: {estimator!r}")

    common = [s for s in weights.index if s in daily_prices.columns]
    if not common:
        return empty
    w = weights[common].astype(float)
    if w.sum() <= 0:
        return empty
    w = w / w.sum()

    # Outer window: 252d under rolling, 504d under EWMA — matches the
    # window compute_risk_contributions feeds to estimate_covariance.
    outer_window = cap_days if estimator.startswith("ewma") else window
    rets_all = daily_prices[common].pct_change().fillna(0.0).iloc[1:]
    rets = rets_all.tail(outer_window) if len(rets_all) > outer_window else rets_all
    n_days_window = int(len(rets))
    if n_days_window < 20:
        return {**empty, "n_days_window": n_days_window,
                "n_symbols": int(len(common))}

    # Synthesize portfolio return on the same window so we can pick down days.
    port_rets = (rets.values @ w.values)
    mask = port_rets <= threshold
    n_down = int(mask.sum())
    if n_down < 20:
        # Below this, the sample covariance on the down-day subset is too
        # noisy to decompose meaningfully (matches the n>=20 guard in
        # compute_risk_contributions and compute_var_cvar).
        return {**empty, "n_days_window": n_days_window,
                "n_down_days": n_down, "n_symbols": int(len(common))}

    rets_down = rets[mask]
    cov_down_daily = rets_down.cov()
    alpha_down: float | None = None
    if estimator == "ewma_lw":
        cov_down_daily, alpha_down = _ledoit_wolf_shrinkage(
            rets_down, cov_down_daily,
        )
    cov_down = cov_down_daily * 252.0
    lambda_echo = float(lambda_param) if estimator.startswith("ewma") else None
    w_arr = w.values
    var_p_down = float(w_arr @ cov_down.values @ w_arr)
    port_vol_ann_down = float(np.sqrt(var_p_down)) if var_p_down > 0 else 0.0

    standalone_vol_down = pd.Series(
        np.sqrt(np.diag(cov_down.values)), index=cov_down.index,
        name="standalone_vol_ann_down",
    )
    weighted_avg_vol_down = float((w * standalone_vol_down).sum())

    if port_vol_ann_down <= 0:
        per_symbol_down = pd.DataFrame({
            "weight":                  w,
            "weight_pct":              w * 100.0,
            "standalone_vol_ann_down": standalone_vol_down,
            "mctr_ann_down":           np.nan,
            "cctr_ann_down":           0.0,
            "pctr_pct_down":           np.nan,
        })
        return {
            "per_symbol_down":           per_symbol_down,
            "port_vol_ann_down":         0.0,
            "weighted_avg_vol_ann_down": weighted_avg_vol_down,
            "dr_down":                   np.nan,
            "n_down_days":               n_down,
            "n_days_window":             n_days_window,
            "n_symbols":                 int(len(common)),
            "threshold":                 float(threshold),
            "estimator":                 estimator,
            "alpha":                     alpha_down,
            "lambda":                    lambda_echo,
        }

    mctr_down = (cov_down.values @ w_arr) / port_vol_ann_down
    cctr_down = w_arr * mctr_down
    pctr_pct_down = cctr_down / port_vol_ann_down * 100.0

    per_symbol_down = pd.DataFrame({
        "weight":                  w,
        "weight_pct":              w * 100.0,
        "standalone_vol_ann_down": standalone_vol_down,
        "mctr_ann_down":           pd.Series(mctr_down, index=cov_down.index),
        "cctr_ann_down":           pd.Series(cctr_down, index=cov_down.index),
        "pctr_pct_down":           pd.Series(pctr_pct_down, index=cov_down.index),
    }).sort_values("pctr_pct_down", ascending=False)

    dr_down = (weighted_avg_vol_down / port_vol_ann_down
               if port_vol_ann_down > 0 else np.nan)

    return {
        "per_symbol_down":           per_symbol_down,
        "port_vol_ann_down":         port_vol_ann_down,
        "weighted_avg_vol_ann_down": weighted_avg_vol_down,
        "dr_down":                   float(dr_down) if np.isfinite(dr_down) else np.nan,
        "n_down_days":               n_down,
        "n_days_window":             n_days_window,
        "n_symbols":                 int(len(common)),
        "threshold":                 float(threshold),
        "estimator":                 estimator,
        "alpha":                     alpha_down,
        "lambda":                    lambda_echo,
    }


def compute_es_contributions(
    weights: pd.Series,
    daily_prices: pd.DataFrame,
    alpha: float = 0.05,
    window: int = 252,
) -> dict:
    """Per-position contribution to portfolio Expected Shortfall (ES).

    Historical ES at confidence (1-alpha):
        VaR_α  = α-quantile of portfolio returns
        ES_p   = -E[r_p | r_p ≤ VaR_α]

    Euler decomposition (ES is positively homogeneous of degree 1 in w):
        contrib_i = -w_i · E[r_i | r_p ≤ VaR_α]
        Σ_i contrib_i = -E[Σ_i w_i r_i | r_p ≤ VaR_α] = ES_p     (exact sum)
        pctr_es_i = contrib_i / ES_p × 100                       (sums to 100%)

    Daily, decimal return space; ES_p is positive when there is a loss in
    the tail (sign-flipped vs the raw quantile). Inputs and normalization
    mirror compute_risk_contributions (SGOV-folded weights from the bundle,
    daily price universe). Tail days are picked on the synthesized
    portfolio return — same series compute_risk_contributions uses.

    Returns:
      - per_symbol_es     DataFrame indexed by symbol, sorted by pctr_es_pct
                          desc. Columns: weight, weight_pct,
                          tail_mean_ret (mean of asset's return on tail
                          days, decimal), contrib_es (signed positive when
                          loss), pctr_es_pct, n_obs_in_window (real
                          pct_change observations within the trailing
                          window — use this for thin-history detection;
                          parallels compute_risk_contributions per PR #66)
      - port_es           portfolio ES (positive number, daily decimal)
      - var_p             portfolio VaR threshold (negative, daily decimal)
      - n_tail_days       count of days r_p ≤ VaR_p
      - n_days_window     window size actually used
      - n_symbols         number of decomposed symbols
      - alpha             echoed back

    Degenerate inputs return NaN scalars with empty per_symbol_es; callers
    should guard on n_tail_days. The Euler identity requires a strictly
    negative ES — if the empirical tail mean is ≥ 0 (very small window with
    only positive tail days), per_symbol_es returns empty.

    **Window contract (Phase 1C audit).** This function takes ONLY `window`
    — there is no `estimator` parameter and no `cap_days`. The tail is
    always picked from a plain trailing-`window` sample, even when the
    sibling `compute_risk_contributions` / `compute_downside_risk_contributions`
    are called under `estimator="ewma_lw"` (which reads cap_days=504
    internally). The dashboard's Risk-Contribution per-symbol table places
    PCTR / Downside PCTR (504d EWMA under default) alongside ES PCTR
    (252d sample) — by design, the windows are NOT unified. UI captions
    must disclose both. Returns `n_days_window == window` exactly so
    consumers can read the actual window used.

    **Thin-history disclosure (Phase 1D).** The `fillna(0.0)` on pct_change
    treats pre-existence days for newly-added symbols as 0% returns. On tail
    days that predate a symbol's entry, that symbol contributes 0 to ES —
    silently reallocating its tail to longer-history symbols. The
    `n_obs_in_window` column reports the count of real (non-NaN) pct_change
    observations per symbol within the window, so the UI can suppress
    ES PCTR for symbols with thin coverage (mirrors the gate in
    compute_risk_contributions added in PR #66).
    """
    empty_df = pd.DataFrame(columns=[
        "weight", "weight_pct", "tail_mean_ret",
        "contrib_es", "pctr_es_pct", "n_obs_in_window",
    ])
    empty = {
        "per_symbol_es":   empty_df,
        "port_es":         np.nan,
        "var_p":           np.nan,
        "n_tail_days":     0,
        "n_days_window":   0,
        "n_symbols":       0,
        "alpha":           float(alpha),
    }
    if weights is None or weights.empty or daily_prices.empty:
        return empty
    if not (0.0 < alpha < 1.0):
        return empty

    common = [s for s in weights.index if s in daily_prices.columns]
    if not common:
        return empty
    w = weights[common].astype(float)
    if w.sum() <= 0:
        return empty
    w = w / w.sum()

    pct_chg = daily_prices[common].pct_change()
    rets_all = pct_chg.fillna(0.0).iloc[1:]
    rets = rets_all.tail(window) if len(rets_all) > window else rets_all
    n_days_window = int(len(rets))
    # Phase 1D: real-observation count within the same window slice the
    # tail sees. The fillna(0.0) above lets newly-onboarded symbols look
    # "safe" on pre-existence tail days; this count is what the UI uses
    # to suppress ES PCTR for thin-coverage symbols.
    pct_chg_window = pct_chg.iloc[1:].tail(n_days_window) if n_days_window > 0 \
        else pct_chg.iloc[1:]
    n_obs_in_window = pct_chg_window.notna().sum().astype(int)
    if n_days_window < 20:
        return {**empty, "n_days_window": n_days_window,
                "n_symbols": int(len(common))}

    port_rets = pd.Series(rets.values @ w.values, index=rets.index)
    var_p = float(port_rets.quantile(alpha))
    tail_mask = port_rets <= var_p
    n_tail = int(tail_mask.sum())
    if n_tail < 1:
        return {**empty, "n_days_window": n_days_window,
                "n_symbols": int(len(common)), "var_p": var_p}

    # Per-asset tail means in raw return space.
    tail_mean = rets[tail_mask].mean()  # Series indexed by symbol
    # Portfolio ES = -E[r_p | tail] = -Σ w_i × tail_mean_i. Use the
    # synthesized-port mean for numerical consistency (handles any tiny
    # rounding noise vs computing on port_rets directly).
    port_es = float(-(w * tail_mean).sum())
    if port_es <= 0:
        # Tail had non-negative mean (degenerate small-window edge case).
        # Can't normalize percentages meaningfully.
        return {**empty, "n_days_window": n_days_window, "n_tail_days": n_tail,
                "n_symbols": int(len(common)), "var_p": var_p,
                "port_es": port_es}

    contrib_es = -(w * tail_mean)              # signed positive when loss
    pctr_es_pct = contrib_es / port_es * 100.0  # sums to 100%

    per_symbol_es = pd.DataFrame({
        "weight":           w,
        "weight_pct":       w * 100.0,
        "tail_mean_ret":    tail_mean,
        "contrib_es":       contrib_es,
        "pctr_es_pct":      pctr_es_pct,
        "n_obs_in_window":  n_obs_in_window.reindex(w.index).fillna(0).astype(int),
    }).sort_values("pctr_es_pct", ascending=False)

    return {
        "per_symbol_es":   per_symbol_es,
        "port_es":         port_es,
        "var_p":           var_p,
        "n_tail_days":     n_tail,
        "n_days_window":   n_days_window,
        "n_symbols":       int(len(common)),
        "alpha":           float(alpha),
    }


# DR (diversification-ratio) rolling windows in trading days — short /
# medium / long. The single source for both UIs (app.py's sidebar regime
# badge + the terminal Risk-Contribution DR-in-context section).
DR_SHORT_W, DR_MED_W, DR_LONG_W = 21, 63, 252


def compute_dr_time_series(
    weights: pd.Series,
    daily_prices: pd.DataFrame,
    windows: tuple[int, ...] = (21, 63, 252),
) -> pd.DataFrame:
    """Rolling diversification ratio over multiple lookback windows.

    DR = (Σ wᵢ × σᵢ) / σ_p where σ_p² = wᵀ Σ w. By construction
    DR ≥ 1 (Cauchy-Schwarz on the wᵢσᵢ vector against itself); DR = 1
    iff all assets are perfectly correlated. Higher = the actual
    portfolio vol is lower than the weighted-avg standalone vol, i.e.
    diversification is paying off.

    Exploits the identity that the rolling std of the synthesized
    portfolio return equals √(wᵀ Σ_W w) on the same W-day window — so
    each timestamp is one rolling-std op per asset + one rolling-std op
    on port_rets, not a fresh covariance estimation. This makes the
    series fast to compute even on hundreds of symbols × hundreds of
    dates (no caching required at typical universe sizes).

    Weights are held static at their current values across the whole
    series (ex-ante view: "what would the DR have looked like over time
    on today's weights"). Symbols missing from daily_prices.columns are
    dropped and remaining weights renormalized — same contract as
    compute_risk_contributions.

    **Estimator note.** This series uses **rolling sample std** at each
    timestamp (the rolling-window equivalent of `estimator="rolling"`
    in compute_risk_contributions). The right-edge identity
    `DR_W.iloc[-1] == compute_risk_contributions(weights, ...).dr`
    holds only when compute_risk_contributions is called with
    `estimator="rolling"` AND `window=W`. Under the dashboard's default
    `estimator="ewma_lw"`, the tile-side `rc["dr"]` is computed from a
    EWMA + Ledoit-Wolf cov over the trailing cap_days while this
    chart's right edge uses an unweighted sample cov over the trailing
    W days — the two will disagree, typically by a few percent in
    ratio terms. See tests/test_risk_metrics.py for the regression that
    locks both the equality (under rolling) and the divergence
    (under ewma_lw).

    Returns a DataFrame indexed by date with one column per requested
    window, named `dr_{W}d`. Rows where any window lacks min_periods=W
    valid observations contain NaN in that column. Empty inputs return
    an empty frame with the expected columns.
    """
    cols = [f"dr_{W}d" for W in windows]
    empty = pd.DataFrame(columns=cols)
    if weights is None or weights.empty or daily_prices.empty:
        return empty

    common = [s for s in weights.index if s in daily_prices.columns]
    if not common:
        return empty
    w = weights[common].astype(float)
    if w.sum() <= 0:
        return empty
    w = w / w.sum()

    rets = daily_prices[common].pct_change().fillna(0.0).iloc[1:]
    if rets.empty:
        return empty
    # Static-weight synthesized portfolio return — drives the σ_p series.
    port_rets = pd.Series(rets.values @ w.values, index=rets.index)

    out = pd.DataFrame(index=rets.index)
    for W in windows:
        # Per-asset rolling std, weighted-averaged across assets.
        std_i = rets.rolling(window=W, min_periods=W).std(ddof=1)
        weighted_avg_std = (std_i * w).sum(axis=1, min_count=len(w))
        port_std = port_rets.rolling(window=W, min_periods=W).std(ddof=1)
        # Guard div-by-zero on degenerate windows (constant returns).
        dr = weighted_avg_std / port_std.where(port_std > 0)
        out[f"dr_{W}d"] = dr
    return out


def compute_max_dr(
    weights: pd.Series,
    daily_prices: pd.DataFrame,
    window: int = 252,
) -> dict:
    """Closed-form upper bound on the diversification ratio achievable
    over the current asset universe.

    Maximizes DR(w) = (Σ wᵢ σᵢ) / sqrt(wᵀ Σ w) over all real w (no
    non-negativity constraint). The unconstrained maximum is

        max DR = sqrt(1ᵀ R⁻¹ 1)

    where R is the correlation matrix of daily returns over the same
    `window` compute_risk_contributions uses. Achieved by the
    Choueifaty-Coignard maximum-diversification portfolio
        w*_i ∝ (R⁻¹ 1)_i / σ_i
    which CAN include negative weights for assets that aren't
    sufficiently uncorrelated with the rest of the book. The long-only
    max DR is somewhat lower and would need a QP solver — surface this
    closed form as the *ceiling* (a regime-gap benchmark), not as an
    actionable target portfolio. UI should label accordingly.

    Uses np.linalg.pinv on the correlation matrix to absorb singular /
    near-singular R (collinear assets, e.g. SGOV + treasury ladder
    folded onto the same symbol). pinv returns the Moore-Penrose
    pseudo-inverse which is well-defined in the singular case and
    coincides with the regular inverse when R is invertible.

    Returns dict with:
      - max_dr      ceiling (≥ current DR always)
      - n_days      days of history actually used
      - n_symbols   symbols decomposed
      - window      echoed back
    """
    empty = {
        "max_dr":    np.nan,
        "n_days":    0,
        "n_symbols": 0,
        "window":    int(window),
    }
    if weights is None or weights.empty or daily_prices.empty:
        return empty

    common = [s for s in weights.index if s in daily_prices.columns]
    if len(common) < 2:
        # Single-asset universe — DR is identically 1, ceiling = 1.
        return {**empty, "n_symbols": len(common),
                "max_dr": 1.0 if len(common) == 1 else np.nan}

    rets_all = daily_prices[common].pct_change().fillna(0.0).iloc[1:]
    rets = rets_all.tail(window) if len(rets_all) > window else rets_all
    n_days = int(len(rets))
    if n_days < 20:
        return {**empty, "n_days": n_days, "n_symbols": len(common)}

    corr = rets.corr().values
    # pinv handles singular R (perfectly collinear assets after SGOV-fold).
    inv_corr = np.linalg.pinv(corr)
    ones = np.ones(corr.shape[0])
    max_dr_sq = float(ones @ inv_corr @ ones)
    if max_dr_sq <= 0:
        return {**empty, "n_days": n_days, "n_symbols": len(common)}
    return {
        "max_dr":    float(np.sqrt(max_dr_sq)),
        "n_days":    n_days,
        "n_symbols": len(common),
        "window":    int(window),
    }


def compute_max_dr_time_series(
    weights: pd.Series,
    daily_prices: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """Rolling Max-DR ceiling — same Choueifaty-Coignard algebra as
    compute_max_dr, but evaluated on a trailing-`window`-day correlation
    matrix at every date so the ceiling is a true upper bound for the
    DR_W series at each historical timestamp (not a fixed scalar tied
    to today's correlation regime).

    Without this, a single horizontal "Max DR" line plotted alongside
    DR_W misleads: DR_W at past dates can legitimately exceed the
    today-anchored ceiling because they're computed on different
    correlation matrices.

    Returns a Series indexed by date; NaN before the window burn-in
    is satisfied or when the rolling correlation matrix is degenerate.
    """
    if weights is None or weights.empty or daily_prices.empty:
        return pd.Series(dtype=float)

    common = [s for s in weights.index if s in daily_prices.columns]
    if len(common) < 2:
        return pd.Series(dtype=float)

    rets = daily_prices[common].pct_change().fillna(0.0).iloc[1:]
    if rets.empty or len(rets) < window:
        return pd.Series(np.nan, index=rets.index, dtype=float)

    out = pd.Series(np.nan, index=rets.index, dtype=float)
    arr = rets.values
    for i in range(window - 1, len(rets)):
        slc = arr[i - window + 1: i + 1]
        # Drop columns with zero variance — symbols whose price history
        # hadn't started yet at date t had all-zero returns after the
        # .fillna(0.0) above, and np.corrcoef on a zero-variance column
        # yields NaN. The reduced corr matrix gives the ceiling for the
        # ACTIVE universe at t; a zero-variance asset contributes nothing
        # to DR either, so the dominance bound is preserved.
        std = slc.std(axis=0, ddof=1)
        active = std > 0
        if active.sum() < 2:
            continue
        slc_active = slc[:, active]
        corr = np.corrcoef(slc_active, rowvar=False)
        if not np.all(np.isfinite(corr)):
            continue
        try:
            inv_corr = np.linalg.pinv(corr)
        except np.linalg.LinAlgError:
            continue
        ones = np.ones(corr.shape[0])
        max_dr_sq = float(ones @ inv_corr @ ones)
        if max_dr_sq > 0:
            out.iloc[i] = float(np.sqrt(max_dr_sq))
    return out


def classify_dr_regime(
    dr_short: float,
    dr_long: float,
    stress_thr: float = 0.90,
    calm_thr: float = 1.10,
) -> dict:
    """Regime classification from the ratio of short-window DR to
    long-window DR.

    When holdings start moving together (correlations clustering, a
    typical stress signal), DR drops; short-window DR drops faster
    than long-window DR. Conversely, decorrelating idiosyncratic moves
    push short-window DR above the long-window baseline.

    Bands (defaults 0.90 / 1.10):
      - ratio < stress_thr → "Stress"  — corrs clustering, diversification
                                         eroding vs long-term baseline
      - ratio > calm_thr   → "Calm"    — names decorrelating, more idio
                                         move than usual
      - else               → "Normal"

    Returns dict: {label, ratio, dr_short, dr_long}. NaN / non-positive
    inputs return label "—" with NaN ratio (UI can fall back to "—").
    """
    if not (np.isfinite(dr_short) and np.isfinite(dr_long) and dr_long > 0):
        return {"label": "—", "ratio": np.nan,
                "dr_short": dr_short, "dr_long": dr_long}
    ratio = float(dr_short / dr_long)
    if ratio < stress_thr:
        label = "Stress"
    elif ratio > calm_thr:
        label = "Calm"
    else:
        label = "Normal"
    return {"label": label, "ratio": ratio,
            "dr_short": float(dr_short), "dr_long": float(dr_long)}


def compute_dr_ratio_series(
    dr_ts: pd.DataFrame,
    short_col: str = "dr_21d",
    long_col:  str = "dr_252d",
) -> pd.Series:
    """Short-window DR divided by long-window DR over time.

    The regime classifier's input signal: when correlations cluster the
    short-window DR drops faster than the long-window baseline, pulling
    the ratio below 1. When names decorrelate the short DR rises above
    the long baseline, lifting the ratio above 1. Plotting this series
    with shaded threshold bands visualizes the regime signal directly
    instead of inferring it from the gap between two lines.

    NaN observations (where either short or long DR is undefined or
    long DR ≤ 0) are returned as NaN.
    """
    if dr_ts.empty or short_col not in dr_ts.columns or long_col not in dr_ts.columns:
        return pd.Series([], dtype=float, name="dr_ratio")
    short = dr_ts[short_col]
    long_safe = dr_ts[long_col].where(dr_ts[long_col] > 0)
    return (short / long_safe).rename("dr_ratio")


def compute_dr_frames(weights: pd.Series, daily: pd.DataFrame,
                      port_rets: pd.Series) -> dict:
    """Shared DR time-series computation (app.py module-scope block 1788-1828),
    consumed by both UIs — the sidebar regime badge + the Risk-Contribution
    DR-in-context sub-sections (2a tiles/charts, 2b regime conditioning).
    dr_ts / max_dr_ts are clipped to the portfolio's existence; `available` +
    dr_s/dr_l come from the latest non-all-NaN row (the DR-availability gate)."""
    empty = {"dr_ts": pd.DataFrame(), "max_dr_ts": pd.Series(dtype=float),
             "ratio_ts": pd.Series(dtype=float), "available": False,
             "dr_s": float("nan"), "dr_l": float("nan")}
    if weights is None or weights.empty or daily is None or daily.empty:
        return empty
    dr_ts = compute_dr_time_series(weights, daily,
                                   windows=(DR_SHORT_W, DR_MED_W, DR_LONG_W))
    max_dr_ts = compute_max_dr_time_series(weights, daily, window=DR_LONG_W)
    port_start = (port_rets.index.min()
                  if (port_rets is not None and not port_rets.empty) else None)
    if port_start is not None and not dr_ts.empty:
        dr_ts = dr_ts.loc[dr_ts.index >= port_start]
    if port_start is not None and not max_dr_ts.empty:
        max_dr_ts = max_dr_ts.loc[max_dr_ts.index >= port_start]
    if dr_ts.empty:
        return empty
    last = dr_ts.dropna(how="all").tail(1)
    if last.empty:
        return {**empty, "dr_ts": dr_ts, "max_dr_ts": max_dr_ts}
    dr_s = float(last[f"dr_{DR_SHORT_W}d"].iloc[0])
    dr_l = float(last[f"dr_{DR_LONG_W}d"].iloc[0])
    ratio_ts = compute_dr_ratio_series(dr_ts, short_col=f"dr_{DR_SHORT_W}d",
                                       long_col=f"dr_{DR_LONG_W}d")
    available = bool(np.isfinite(classify_dr_regime(dr_s, dr_l).get("ratio", np.nan)))
    return {"dr_ts": dr_ts, "max_dr_ts": max_dr_ts, "ratio_ts": ratio_ts,
            "available": available, "dr_s": dr_s, "dr_l": dr_l}


def compute_dr_regime_thresholds(
    ratio_series: pd.Series,
    method: str = "fixed",
    fixed_stress: float = 0.90,
    fixed_calm:   float = 1.10,
    percentile_stress: float = 0.20,
    percentile_calm:   float = 0.80,
    zscore_threshold:  float = 1.0,
) -> dict:
    """Threshold pair (stress, calm) for the short/long DR ratio.

    Three methods supported:
      - "fixed":      use the round-number defaults 0.90 / 1.10. Out-of-
                      sample by construction — thresholds don't depend on
                      the ratio series. **Default and recommended for
                      regime comparison across portfolios.**
      - "percentile": p20 / p80 of the observed ratio series — "Stress"
                      then literally means the ratio is in the worst 20%
                      of recent history. Self-calibrating to the
                      portfolio's actual operational range.
      - "zscore":     mean ± `zscore_threshold` × std of the ratio. The
                      most adaptive choice; corresponds to "more than 1
                      standard deviation away from the trailing mean."

    **In-sample caveat (Phase 1E audit).** `"percentile"` and `"zscore"`
    are *in-sample* — they fit thresholds on the same series they classify.
    Consequence: percentile mode mechanically produces ~20% stress / ~20%
    calm regardless of the true regime distribution (because the 20th and
    80th percentiles are defined on the data itself); zscore mode is less
    extreme but still self-fitting. Use them for "where does today sit in
    *this portfolio's* operating range" — not for "is this portfolio
    objectively in stress." The UI default is `"fixed"` for this reason.

    Returns a dict with stress_thr, calm_thr, method, plus method-specific
    diagnostics (n_obs, mean, sd, etc.). Empty / degenerate inputs fall
    back to the fixed defaults so the UI can always render *some* bands.
    """
    if method not in ("fixed", "percentile", "zscore"):
        raise ValueError(f"Unknown threshold method: {method!r}")
    if method == "fixed":
        return {
            "stress_thr": float(fixed_stress),
            "calm_thr":   float(fixed_calm),
            "method":     "fixed",
            "n_obs":      int(ratio_series.dropna().shape[0]),
        }
    series = ratio_series.dropna()
    n = int(series.shape[0])
    if n < 10:
        # Not enough data for a meaningful empirical band — fall back to
        # fixed and signal the fallback so the UI can warn.
        return {
            "stress_thr": float(fixed_stress),
            "calm_thr":   float(fixed_calm),
            "method":     "fixed",
            "n_obs":      n,
            "fallback":   f"only {n} obs — too thin for {method}",
        }
    if method == "percentile":
        return {
            "stress_thr": float(series.quantile(percentile_stress)),
            "calm_thr":   float(series.quantile(percentile_calm)),
            "method":     "percentile",
            "n_obs":      n,
            "p_stress":   float(percentile_stress),
            "p_calm":     float(percentile_calm),
        }
    # zscore
    mean = float(series.mean())
    sd = float(series.std(ddof=1))
    # Phase 1C audit: catch degenerate-but-not-thin inputs (n ≥ 10 but
    # zero-variance — e.g. a constant ratio in a flat regime). Without
    # this guard, stress_thr == calm_thr == mean and classify_dr_regime
    # routes everything to "Normal" since both inequalities are strict.
    # Fall back to fixed defaults with a fallback message, matching the
    # n < 10 branch above.
    if not np.isfinite(sd) or sd <= 0:
        return {
            "stress_thr": float(fixed_stress),
            "calm_thr":   float(fixed_calm),
            "method":     "fixed",
            "n_obs":      n,
            "fallback":   "zero-variance ratio series — falling back to fixed",
        }
    return {
        "stress_thr": mean - zscore_threshold * sd,
        "calm_thr":   mean + zscore_threshold * sd,
        "method":     "zscore",
        "n_obs":      n,
        "mean":       mean,
        "sd":         sd,
        "z":          float(zscore_threshold),
    }


SPY_STATES = ("calm", "correction", "stress")
VIX_STATES = ("low", "normal", "high")


def classify_market_regime(
    spy_series: pd.Series,
    vix_series: pd.Series,
    dd_window: int = 21,
    dd_thresholds: tuple[float, float] = (0.03, 0.10),
    vix_z_window: int = 252,
    vix_z_thresholds: tuple[float, float] = (-0.5, 0.5),
) -> pd.DataFrame:
    """Two-axis market regime labels keyed on SPY drawdown × VIX z-score.

    Tags each date with a regime label combining
      - SPY drawdown state from the trailing `dd_window`-day rolling peak:
        "calm"       (dd ≤ dd_thresholds[0]),
        "correction" (dd_thresholds[0] < dd ≤ dd_thresholds[1]),
        "stress"     (dd > dd_thresholds[1]).
      - VIX z-state from the trailing `vix_z_window`-day rolling mean/std
        of the supplied VIX series:
        "low"    (z ≤ vix_z_thresholds[0]),
        "normal" (between),
        "high"   (z > vix_z_thresholds[1]).

    Combined label is "{spy_state}_{vix_state}" — the cross-product of
    nine cells. Used as the conditioning variable for
    `compute_regime_conditional_dr`. Uses CBOE spot ^VIX (not a futures
    ETF proxy like VIXY) — VIXY tracks rolled futures and is
    contango-decayed; z-scoring normalizes the level but leaves
    term-structure distortion in the changes. Spot VIX has no proxy issue
    and 35+ years of free CBOE history.

    SPY and VIX series are aligned on their date-index intersection
    (rolling stats are computed per-series first so each axis uses its
    own full history for burn-in, then joined for the combined label).

    Returns DataFrame indexed by date with columns:
      - spy_drawdown   trailing-window drawdown from peak, in [0, 1]
      - spy_state      one of SPY_STATES, NaN during burn-in
      - vix_z          trailing-window z-score of VIX
      - vix_state      one of VIX_STATES, NaN during burn-in
      - regime         combined "{spy_state}_{vix_state}", NaN on either
                       burn-in, missing input, or non-overlap dates

    Empty inputs return an empty frame with the expected columns.
    """
    cols_out = ["spy_drawdown", "spy_state", "vix_z", "vix_state", "regime"]
    empty = pd.DataFrame(columns=cols_out)
    if spy_series is None or vix_series is None:
        return empty
    spy = spy_series.dropna().sort_index()
    vix = vix_series.dropna().sort_index()
    if spy.empty or vix.empty:
        return empty

    peak = spy.rolling(window=dd_window, min_periods=dd_window).max()
    spy_dd = (peak - spy) / peak.where(peak > 0)
    dd_lo, dd_hi = dd_thresholds
    spy_state = pd.Series(index=spy.index, dtype=object, name="spy_state")
    valid_dd = spy_dd.notna()
    spy_state.loc[valid_dd & (spy_dd <= dd_lo)] = "calm"
    spy_state.loc[valid_dd & (spy_dd > dd_lo) & (spy_dd <= dd_hi)] = "correction"
    spy_state.loc[valid_dd & (spy_dd > dd_hi)] = "stress"

    vix_mean = vix.rolling(window=vix_z_window, min_periods=vix_z_window).mean()
    vix_sd   = vix.rolling(window=vix_z_window, min_periods=vix_z_window).std(ddof=1)
    vix_z = (vix - vix_mean) / vix_sd.where(vix_sd > 0)
    z_lo, z_hi = vix_z_thresholds
    vix_state = pd.Series(index=vix.index, dtype=object, name="vix_state")
    valid_z = vix_z.notna()
    vix_state.loc[valid_z & (vix_z <= z_lo)] = "low"
    vix_state.loc[valid_z & (vix_z > z_lo) & (vix_z <= z_hi)] = "normal"
    vix_state.loc[valid_z & (vix_z > z_hi)] = "high"

    # Outer-join the two axes on date — combined regime is only defined
    # where both states post-burn-in are present on the same date.
    out = pd.DataFrame({
        "spy_drawdown": spy_dd,
        "spy_state":    spy_state,
        "vix_z":        vix_z,
        "vix_state":    vix_state,
    })
    regime = pd.Series(index=out.index, dtype=object, name="regime")
    both = out["spy_state"].notna() & out["vix_state"].notna()
    regime.loc[both] = out.loc[both, "spy_state"] + "_" + out.loc[both, "vix_state"]
    out["regime"] = regime
    return out


def compute_regime_conditional_dr(
    dr_time_series: pd.DataFrame,
    regime_labels: pd.DataFrame,
    dr_ratio_series: pd.Series | None = None,
    min_n_per_cell: int = 20,
    headline_window_col: str = "dr_63d",
    baseline_dr: float | None = None,
) -> dict:
    """Regime-conditional summary of the DR time series.

    For each populated (spy_state, vix_state) cell in the cross-product,
    computes mean / median / std / N of every DR window column in
    `dr_time_series`, plus (optionally) the short/long DR ratio. Cells
    with N below `min_n_per_cell` are flagged as low-confidence — the
    spec is explicit that small-N cells must never be presented as
    confident readings.

    `baseline_dr` is the reference value against which the tail-cell
    delta is measured. When provided (e.g. the trailing-1Y mean of the
    same window — the portfolio's "own baseline"), all delta-vs-baseline
    diagnostics use this number. When None, falls back to the
    unconditional mean over the labelled window. The user-facing
    diagnostic ("does this regime underperform the portfolio's own
    baseline") cares about delta-vs-baseline, not delta-vs-ceiling.

    Diagnostics layered on top of the per-cell summary:
      - `tail_highlight`: the "stress × high" cell DR compared to
        `baseline_dr` (or unconditional mean) — the headline "did the
        hedge work when it mattered" number. Uses `headline_window_col`
        for the comparison (default 63d).
      - `asymmetry`: (calm × low) minus (stress × high) DR on the same
        headline window. Positive = diversification stronger in calm,
        weaker in tails. Negative = stronger in tails (rare, ideal).

    Returns dict:
      - summary: DataFrame, one row per populated cell. Columns:
        spy_state, vix_state, n, low_n,
        {col}_mean, {col}_median, {col}_std for each DR window,
        dr_ratio_mean / _median / _std if ratio supplied,
        {headline}_delta_vs_baseline (cell mean − baseline_dr).
      - overall: dict of unconditional stats over the same date set.
      - baseline_dr: the reference value used for delta computation.
      - tail_highlight: dict {cell_dr, baseline_dr, delta, n, low_n} or
        None if the (stress, high) cell is empty.
      - asymmetry: dict {calm_low_dr, stress_high_dr, delta} or None if
        either anchor cell is empty.
      - n_total: total date count with both DR and regime defined.
      - min_n_per_cell: echoed back for UI display.
      - headline_window_col: echoed back.
    """
    empty_summary = pd.DataFrame(columns=["spy_state", "vix_state", "n", "low_n"])
    if dr_time_series.empty or regime_labels.empty:
        return {
            "summary":            empty_summary,
            "overall":            {},
            "baseline_dr":        baseline_dr,
            "tail_highlight":     None,
            "asymmetry":          None,
            "n_total":            0,
            "min_n_per_cell":     int(min_n_per_cell),
            "headline_window_col": headline_window_col,
        }

    # Align on the intersection of dates with both DR data and regime label.
    dr_cols = [c for c in dr_time_series.columns if c.startswith("dr_")]
    joined = dr_time_series[dr_cols].join(
        regime_labels[["spy_state", "vix_state", "regime"]], how="inner"
    )
    if dr_ratio_series is not None and not dr_ratio_series.empty:
        joined = joined.join(dr_ratio_series.rename("dr_ratio"), how="left")
        ratio_col_present = True
    else:
        ratio_col_present = False

    # Drop rows where regime is undefined (burn-in) or DR headline is NaN.
    joined = joined.dropna(subset=["regime"])
    if joined.empty:
        return {
            "summary":            empty_summary,
            "overall":            {},
            "baseline_dr":        baseline_dr,
            "tail_highlight":     None,
            "asymmetry":          None,
            "n_total":            0,
            "min_n_per_cell":     int(min_n_per_cell),
            "headline_window_col": headline_window_col,
        }

    metric_cols = list(dr_cols) + (["dr_ratio"] if ratio_col_present else [])

    grouped = joined.groupby(["spy_state", "vix_state"], sort=False)
    rows = []
    for (sps, vxs), block in grouped:
        row = {"spy_state": sps, "vix_state": vxs, "n": int(len(block))}
        row["low_n"] = bool(row["n"] < min_n_per_cell)
        for col in metric_cols:
            vals = block[col].dropna()
            if vals.empty:
                row[f"{col}_mean"]   = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_std"]    = np.nan
            else:
                row[f"{col}_mean"]   = float(vals.mean())
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_std"]    = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Unconditional baseline over the same date set.
    overall: dict = {"n": int(len(joined))}
    for col in metric_cols:
        vals = joined[col].dropna()
        if vals.empty:
            overall[f"{col}_mean"]   = np.nan
            overall[f"{col}_median"] = np.nan
            overall[f"{col}_std"]    = np.nan
        else:
            overall[f"{col}_mean"]   = float(vals.mean())
            overall[f"{col}_median"] = float(vals.median())
            overall[f"{col}_std"]    = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan

    headline = headline_window_col if headline_window_col in metric_cols else (
        metric_cols[0] if metric_cols else None
    )

    # Resolved baseline: explicit > unconditional mean over the labelled
    # window. The explicit path is the user-meaningful one (trailing-1Y
    # mean — "what's normal for this portfolio"); unconditional fallback
    # keeps the function self-sufficient when the caller hasn't computed
    # a baseline yet.
    if baseline_dr is not None and np.isfinite(float(baseline_dr)):
        baseline_resolved = float(baseline_dr)
    elif headline is not None:
        _b = overall.get(f"{headline}_mean", np.nan)
        baseline_resolved = float(_b) if np.isfinite(_b) else np.nan
    else:
        baseline_resolved = np.nan

    # Attach per-cell delta-vs-baseline as a summary column for UI
    # display (heatmap z, table column).
    if headline is not None and np.isfinite(baseline_resolved):
        summary[f"{headline}_delta_vs_baseline"] = (
            summary[f"{headline}_mean"] - baseline_resolved
        )

    def _cell_dr(spy_state: str, vix_state: str) -> tuple[float, int] | None:
        if headline is None:
            return None
        match = summary[(summary["spy_state"] == spy_state) &
                        (summary["vix_state"] == vix_state)]
        if match.empty:
            return None
        n = int(match["n"].iloc[0])
        val = match[f"{headline}_mean"].iloc[0]
        if not np.isfinite(val):
            return None
        return float(val), n

    tail_highlight = None
    tail = _cell_dr("stress", "high")
    if tail is not None and headline is not None and np.isfinite(baseline_resolved):
        cell_dr, n = tail
        tail_highlight = {
            "cell":         "stress_high",
            "cell_dr":      cell_dr,
            "baseline_dr":  baseline_resolved,
            "delta":        float(cell_dr - baseline_resolved),
            "n":            n,
            "low_n":        bool(n < min_n_per_cell),
            "window":       headline,
        }

    asymmetry = None
    calm_low = _cell_dr("calm", "low")
    stress_high = _cell_dr("stress", "high")
    if calm_low is not None and stress_high is not None and headline is not None:
        asymmetry = {
            "calm_low_dr":    calm_low[0],
            "stress_high_dr": stress_high[0],
            "delta":          float(calm_low[0] - stress_high[0]),
            "calm_low_n":     calm_low[1],
            "stress_high_n":  stress_high[1],
            "window":         headline,
        }

    return {
        "summary":            summary,
        "overall":            overall,
        "baseline_dr":        baseline_resolved,
        "tail_highlight":     tail_highlight,
        "asymmetry":          asymmetry,
        "n_total":            int(len(joined)),
        "min_n_per_cell":     int(min_n_per_cell),
        "headline_window_col": headline_window_col,
    }


def interpret_regime_dr(
    conditional: dict,
    weakness_thresholds: tuple[float, float] = (0.20, 0.40),
) -> dict:
    """Plain-language read of `compute_regime_conditional_dr` output.

    Classifies the tail cell's delta-vs-baseline into one of:
      - "holds":        delta ≥ 0  — tail cell ≥ baseline (the regime that
                        matters keeps up with the portfolio's own mean).
      - "erodes":       0 > delta > −weakness_thresholds[0]  — mild slip.
      - "weakens":      −weakness_thresholds[0] ≥ delta > −weakness_thresholds[1]
      - "breaks":       delta ≤ −weakness_thresholds[1]  — diversification
                        structure does not hold under stress.
      - "insufficient": tail cell empty or baseline unavailable.

    The baseline is whatever `compute_regime_conditional_dr` resolved
    (explicit `baseline_dr` arg, e.g. trailing-1Y mean — preferred — or
    the unconditional mean as a fallback). The diagnostic question is
    "does this regime underperform the portfolio's own baseline," not
    "is this regime close to a theoretical ceiling."

    Returns dict {character, headline, asymmetry_note, delta, low_n_warning}.
    `headline` is a short prose string ready for UI display.
    """
    tail = conditional.get("tail_highlight")
    asym = conditional.get("asymmetry")
    thr_erodes, thr_weakens = weakness_thresholds

    if tail is None:
        return {
            "character":     "insufficient",
            "headline":      "(stress × high-vol) cell empty — extend window before "
                             "drawing conclusions.",
            "asymmetry_note": "",
            "delta":         np.nan,
            "low_n_warning": True,
        }

    delta = tail["delta"]            # cell_dr − baseline; negative = tail weaker
    cell_dr = tail["cell_dr"]
    baseline_dr = tail["baseline_dr"]
    n = tail["n"]
    low_n = tail["low_n"]

    if delta >= 0:
        character = "holds"
        headline = (
            f"Stress×high-vol cell: DR {cell_dr:.2f} vs baseline {baseline_dr:.2f} "
            f"({delta:+.2f}). Diversification holds when stress hits — the hedge "
            f"sleeve is earning its keep."
        )
    elif -delta < thr_erodes:
        character = "erodes"
        headline = (
            f"Stress×high-vol cell: DR {cell_dr:.2f} vs baseline {baseline_dr:.2f} "
            f"({delta:+.2f}). Mild erosion under stress."
        )
    elif -delta < thr_weakens:
        character = "weakens"
        headline = (
            f"Stress×high-vol cell: DR {cell_dr:.2f} vs baseline {baseline_dr:.2f} "
            f"({delta:+.2f}). Weakens when needed most."
        )
    else:
        character = "breaks"
        headline = (
            f"Stress×high-vol cell: DR {cell_dr:.2f} vs baseline {baseline_dr:.2f} "
            f"({delta:+.2f}). Breaks under stress — the structure isn't holding "
            f"in the regime it's designed for."
        )

    if low_n:
        headline += f" Note: N={n}, treat as preliminary."

    if asym is None:
        asym_note = ""
    else:
        a = asym["delta"]            # calm_low − stress_high
        if a > 0:
            asym_note = (
                f"Calm-low vs stress-high spread: {a:+.2f}. Diversification is "
                f"stronger in calm than in tails — typical."
            )
        elif a < 0:
            asym_note = (
                f"Calm-low vs stress-high spread: {a:+.2f}. Diversification is "
                f"stronger in tails than in calm — the favorable asymmetry."
            )
        else:
            asym_note = "Calm-low and stress-high DR identical."

    return {
        "character":      character,
        "headline":       headline,
        "asymmetry_note": asym_note,
        "delta":          float(delta),
        "low_n_warning":  bool(low_n) or n < 20,
    }


def compute_correlation_matrix(
    daily_prices: pd.DataFrame,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Pearson correlation matrix of daily log returns for `symbols`.

    Use log returns rather than simple returns so highly skewed inputs
    (small T-Bill-ETF moves alongside ±5% equity days) get a more stable
    estimate. NaN-tolerant: any column entirely NaN on the slice is
    dropped before the corr; pairs of columns with no overlap return NaN.

    DO NOT COMBINE WITH ``estimate_covariance`` OUTPUT for marginal-risk
    decomposition. That pipeline (``compute_risk_contributions`` /
    ``compute_downside_risk_contributions`` / ``compute_es_contributions``)
    builds covariance from simple-return ``pct_change``, and mixing
    log-return correlations with simple-return stdevs would give
    inconsistent risk attribution. If you need correlations on the same
    basis as that covariance pipeline, derive them in place from Σ:
    C[i,j] = Σ[i,j] / sqrt(Σ[i,i] · Σ[j,j]). The log-return correlations
    here are for DISPLAY purposes (concentration tab, stress-day
    conditional matrix) where numerical stability on heterogeneous-vol
    mixtures matters more than basis-consistency with the risk-decomp
    pipeline. Tests in ``tests/test_log_vs_simple_returns_boundary.py``
    lock the invariant.
    """
    if daily_prices.empty:
        return pd.DataFrame()
    cols = ([s for s in symbols if s in daily_prices.columns]
            if symbols is not None else list(daily_prices.columns))
    if not cols:
        return pd.DataFrame()
    px = daily_prices[cols].sort_index()
    # log returns avoid tiny-denominator blowups on near-flat series
    rets = np.log(px).diff().dropna(how="all")
    if rets.empty:
        return pd.DataFrame()
    return rets.corr()


def compute_rolling_pair_correlations(
    daily_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    window: int = 90,
) -> pd.DataFrame:
    """Rolling Pearson correlation for the requested (a, b) pairs.

    Returns a DataFrame indexed by date with one column per pair labelled
    "a–b" (em-dash) and values in [-1, 1]. Pairs touching a column missing
    from `daily_prices` are silently skipped (returned DataFrame just
    omits that column).

    Uses LOG returns (see ``compute_correlation_matrix`` for rationale and
    a warning against combining with the simple-return covariance pipeline).
    """
    if daily_prices.empty or window < 5 or not pairs:
        return pd.DataFrame()
    px = daily_prices.sort_index()
    rets = np.log(px).diff()
    out = {}
    for a, b in pairs:
        if a not in rets.columns or b not in rets.columns:
            continue
        ra = rets[a]
        rb = rets[b]
        # pandas rolling .corr handles the pairwise alignment + window
        # arithmetic. min_periods=window forces a full window before
        # emitting a value (no half-cooked early estimates).
        roll = ra.rolling(window=window, min_periods=window).corr(rb)
        out[f"{a}–{b}"] = roll
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1, sort=True).dropna(how="all")


def compute_rolling_avg_pairwise_correlation(
    daily_prices: pd.DataFrame,
    symbols: list[str],
    window: int = 90,
) -> pd.Series:
    """Mean of all unique off-diagonal pairwise correlations over a rolling
    window. Collapses an N×N rolling correlation cube into one line: a
    proxy for "how clustered the universe is right now."

    Excludes self-pairs and double-counts; if fewer than 2 symbols are
    available the result is an empty Series.

    Uses LOG returns (see ``compute_correlation_matrix`` for rationale and
    a warning against combining with the simple-return covariance pipeline).
    """
    if daily_prices.empty or window < 5:
        return pd.Series(dtype=float, name="avg_pair_corr")
    cols = [s for s in symbols if s in daily_prices.columns]
    if len(cols) < 2:
        return pd.Series(dtype=float, name="avg_pair_corr")
    px = daily_prices[cols].sort_index()
    rets = np.log(px).diff()
    # pandas rolling .corr() with no second arg returns a pairwise
    # MultiIndexed frame; pull out the lower-triangle off-diagonal mean.
    n = len(cols)
    triu_idx = np.triu_indices(n, k=1)  # upper triangle, no diagonal
    out = {}
    for end in range(window - 1, len(rets)):
        slc = rets.iloc[end - window + 1 : end + 1]
        if slc.dropna(how="all").shape[0] < window // 2:
            continue
        c = slc.corr().values
        if c.shape != (n, n):
            continue
        vals = c[triu_idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[rets.index[end]] = float(vals.mean())
    if not out:
        return pd.Series(dtype=float, name="avg_pair_corr")
    return pd.Series(out, name="avg_pair_corr").sort_index()


def compute_conditional_correlation_matrix(
    daily_prices: pd.DataFrame,
    symbols: list[str],
    condition_symbol: str = "SPY",
    z_threshold: float = -1.5,
    min_stress_days: int = 15,
    method: str = "spearman",
) -> dict:
    """Rank (Spearman) correlation conditional on stress days for
    `condition_symbol`. Pearson is also supported via `method="pearson"`.

    Stress day = a day where `condition_symbol`'s daily log return is
    ≤ z_threshold standard deviations from its mean, measured on the full
    sample. Default z_threshold = -1.5 picks roughly the worst 7% of SPY
    days. Returns full-sample ρ alongside conditional ρ and their Δ so
    callers can show "where do correlations spike in stress."

    Why Spearman by default: the conditional matrix is computed on a
    small sample (~15–50 stress days). Pearson ρ on tiny samples is
    dominated by single outlier prints — one −60% day in a 20-day window
    can swing ρ by ±0.7 regardless of structural co-movement. Spearman
    converts each day's return into a rank within the sample, so the
    distressed-equity tail still participates in the estimate but with
    bounded leverage. No observation is dropped or clipped; nothing
    about the raw daily_prices is hidden — only the *estimator* on top
    is made outlier-robust. Full-sample matrix uses the same method so
    Δ remains apples-to-apples.

    Returns dict:
      full         — N×N full-sample correlation DataFrame (method)
      conditional  — N×N stress-day correlation DataFrame (method)
      delta        — conditional − full (NaN where either is NaN)
      method       — str, the correlation method used
      threshold    — float, the daily log-return cutoff used
      mean         — float, sample mean of `condition_symbol` log returns
      sd           — float, sample sd of `condition_symbol` log returns
      n_full       — int, # of days in full sample
      n_stress     — int, # of stress days that pass the cutoff
      enough       — bool, n_stress >= min_stress_days

    Empty / underpowered inputs return empty frames with `enough=False`.

    Uses LOG returns (see ``compute_correlation_matrix`` for rationale and
    a warning against combining with the simple-return covariance pipeline).
    """
    empty = pd.DataFrame()
    base = {
        "full": empty, "conditional": empty, "delta": empty,
        "method": method,
        "threshold": float("nan"), "mean": float("nan"), "sd": float("nan"),
        "n_full": 0, "n_stress": 0, "enough": False,
    }
    if daily_prices.empty or not symbols:
        return base
    if condition_symbol not in daily_prices.columns:
        return base
    cols = [s for s in symbols if s in daily_prices.columns]
    if not cols:
        return base
    # Make sure the conditioning symbol is in the returns frame even if
    # the caller didn't list it among `symbols` — but don't add it to the
    # final matrix columns.
    work_cols = list(dict.fromkeys([*cols, condition_symbol]))
    px = daily_prices[work_cols].sort_index()
    rets = np.log(px).diff().dropna(how="all")
    if rets.empty:
        return base
    # SPY's stress days are defined on its raw log-return distribution
    # (not on ranks) — we want "the worst SPY days by actual magnitude,"
    # which is what z-threshold on log returns gives us.
    cond = rets[condition_symbol].dropna()
    if len(cond) < 2:
        return base
    mu = float(cond.mean())
    sigma = float(cond.std(ddof=1))
    if not np.isfinite(sigma) or sigma == 0.0:
        return base
    threshold = mu + z_threshold * sigma  # z_threshold < 0 = stress tail
    stress_idx = cond.index[cond <= threshold]
    n_full = int(len(rets))
    n_stress = int(len(stress_idx))
    # Both full and conditional matrices use `method`. Spearman ranks
    # are computed within each sample independently — so the conditional
    # matrix ranks only the stress days, which is the right behavior:
    # it asks "how do these names co-move on bad days," not "where do
    # bad days fall in the full-sample distribution."
    full_corr = rets[cols].corr(method=method)
    if n_stress < min_stress_days:
        return {
            "full": full_corr, "conditional": empty, "delta": empty,
            "method": method,
            "threshold": float(threshold), "mean": mu, "sd": sigma,
            "n_full": n_full, "n_stress": n_stress, "enough": False,
        }
    stress_rets = rets.loc[stress_idx, cols]
    cond_corr = stress_rets.corr(method=method)
    # Δ frame on the union of indices; NaN where either side is NaN.
    delta = cond_corr.subtract(full_corr, fill_value=np.nan)
    # Diagonal of delta is 0 by construction — but only set where both
    # corr matrices report 1 (i.e. the symbol had any obs on either side).
    for s in cols:
        if (s in cond_corr.index and s in full_corr.index
                and np.isfinite(cond_corr.loc[s, s])
                and np.isfinite(full_corr.loc[s, s])):
            delta.loc[s, s] = 0.0
    return {
        "full": full_corr, "conditional": cond_corr, "delta": delta,
        "method": method,
        "threshold": float(threshold), "mean": mu, "sd": sigma,
        "n_full": n_full, "n_stress": n_stress, "enough": True,
    }


def splice_ticker_history(
    daily_prices: pd.DataFrame,
    ticker_history: dict,
) -> pd.DataFrame:
    """Splice prior-ticker price history under each current-ticker column.

    `ticker_history` maps a current ticker to a list of prior segments:

        {
          "VISN": [{"prior_symbol": "COMM", "effective_date": "2026-01-14"}],
          "META": [{"prior_symbol": "FB",   "effective_date": "2022-06-09"}],
        }

    Each segment says "this ticker used to be `prior_symbol` until
    `effective_date`". Chained renames are supported — list multiple
    segments per current ticker, and each one's date range is
    [previous_effective_date, this_effective_date), with the oldest
    segment filling everything before its effective_date.

    **Chain expressed under one entry — DO NOT split across entries.**
    Phase 1C audit: if a future user wants to splice A → B → C, list
    BOTH segments under the C entry:

        {"C": [{"prior_symbol": "A", "effective_date": "2020-01-01"},
               {"prior_symbol": "B", "effective_date": "2022-01-01"}]}

    Splitting the chain across two separate entries (e.g. one entry
    for B's old name and another for C's old name) is order-dependent:
    out[B] is the running output, so processing C before B uses B's
    UN-extended series, while processing B before C uses the extended
    one. Python dict iteration order is insertion order — same config,
    different result. Tests pin the single-entry pattern.

    Semantics:
      - Current-ticker observations are never overwritten (the splice
        only fills NaN cells).
      - Prior-symbol observations outside their segment's date range are
        ignored — e.g. if BBB was AAA's name 2018-2022 and BBB's column
        in `daily_prices` happens to have post-2022 data (it shouldn't,
        but defensively), those rows do NOT leak into AAA's series.
      - If a prior_symbol is not a column in `daily_prices`, that segment
        is silently skipped (no data to splice in).
      - If a current ticker is not yet a column in `daily_prices`, it is
        created — useful when the user holds a renamed position but the
        fetch run only has the prior_symbol's data on hand.

    Returns a NEW DataFrame; input is not mutated.
    """
    if daily_prices.empty or not ticker_history:
        return daily_prices.copy()

    out = daily_prices.copy()
    for current_ticker, segments in ticker_history.items():
        if not segments:
            continue
        # Walk OLDEST-first so each segment's date range is
        # [prev_effective_date, this_effective_date), and the oldest
        # segment fills everything before its effective_date.
        sorted_segs = sorted(
            segments,
            key=lambda s: pd.Timestamp(s["effective_date"]),
        )
        cur_col = (out[current_ticker].copy()
                   if current_ticker in out.columns
                   else pd.Series(np.nan, index=out.index, dtype=float))
        prev_eff: pd.Timestamp | None = None
        for seg in sorted_segs:
            prior = seg["prior_symbol"]
            this_eff = pd.Timestamp(seg["effective_date"])
            if prior not in out.columns:
                prev_eff = this_eff
                continue
            # Active range for this prior_symbol:
            #   - oldest segment (prev_eff is None): dates < this_eff
            #   - later segments: [prev_eff, this_eff)
            if prev_eff is None:
                mask = out.index < this_eff
            else:
                mask = (out.index >= prev_eff) & (out.index < this_eff)
            write = mask & cur_col.isna()
            cur_col = cur_col.where(~write, out[prior])
            prev_eff = this_eff
        out[current_ticker] = cur_col
    return out


def splice_sgov_with_bil(
    long_prices: pd.DataFrame,
    sgov_col: str = "SGOV",
    bil_col: str = "BIL",
) -> pd.Series:
    """Extend SGOV history backward using BIL as a 1-3 month T-Bill proxy.

    SGOV launched May 2020. BIL has been around since 2007 and tracks the
    same short-T-Bill universe. The splice continues SGOV verbatim where
    it exists, and uses BIL daily-return continuations to back-fill the
    earlier dates. Rebased so the spliced level matches SGOV at SGOV's
    first observation.

    Returns a single Series indexed by date.
    """
    if long_prices.empty or sgov_col not in long_prices.columns:
        return pd.Series(dtype=float, name="SGOV_ext")
    sgov = long_prices[sgov_col].dropna().sort_index()
    if sgov.empty:
        return pd.Series(dtype=float, name="SGOV_ext")
    if bil_col not in long_prices.columns:
        return sgov.rename("SGOV_ext")
    bil = long_prices[bil_col].dropna().sort_index()
    if bil.empty:
        return sgov.rename("SGOV_ext")
    sgov_first = sgov.index.min()
    bil_pre = bil[bil.index < sgov_first]
    if bil_pre.empty:
        return sgov.rename("SGOV_ext")
    # Back-cast BIL onto SGOV's first level by rescaling each pre-launch
    # BIL close as (BIL_t / BIL_first_after_pre) * SGOV_first_level.
    # Simpler equivalent: rescale BIL to match SGOV's first level.
    # Pick the BIL price closest to SGOV's first date for the anchor.
    bil_at_anchor = bil[bil.index <= sgov_first]
    if bil_at_anchor.empty:
        return sgov.rename("SGOV_ext")
    anchor = float(bil_at_anchor.iloc[-1])
    if anchor <= 0:
        return sgov.rename("SGOV_ext")
    scale = float(sgov.iloc[0]) / anchor
    bil_rescaled = bil_pre * scale
    return pd.concat([bil_rescaled, sgov]).sort_index().rename("SGOV_ext")


def extend_sgov_with_bil_panel(daily_prices: pd.DataFrame,
                               long_prices: pd.DataFrame,
                               symbol: str = "SGOV") -> pd.DataFrame:
    """Bridge ``symbol``'s missing history in the daily panel with the
    BIL-spliced long-history series (splice_sgov_with_bil).

    Panel values stay authoritative wherever present — only NaN gaps fill,
    and only dates already in the panel index (no invented rows); every
    other column passes through untouched. No-op when the panel lacks the
    column or the splice comes back empty (e.g. no long-history file — the
    synth fixture). TK 2026-07-17: SGOV should model as T-bills before its
    inception instead of NaN→0% biasing vol/correlations low.
    """
    if daily_prices.empty or symbol not in daily_prices.columns:
        return daily_prices
    ext = splice_sgov_with_bil(long_prices, sgov_col=symbol)
    if ext.empty:
        return daily_prices
    out = daily_prices.copy()
    out[symbol] = out[symbol].combine_first(ext.reindex(out.index))
    return out


def _aligned(p: pd.Series, b: pd.Series, window: int | None = None) -> pd.DataFrame:
    df = pd.concat([p, b], axis=1, keys=["p", "b"], sort=True).dropna()
    if window is not None and len(df) > window:
        df = df.tail(window)
    return df


def compute_beta(p: pd.Series, b: pd.Series, window: int | None = None) -> float:
    df = _aligned(p, b, window)
    if len(df) < 2:
        return np.nan
    var_b = float(df["b"].var())
    # Phase 1C audit: guard against near-zero (not just exactly-zero)
    # benchmark variance. An upstream refactor that filters or
    # arithmetic-shifts the benchmark series can leave a tiny float
    # residual variance like 1e-20 — `cov / 1e-20` would produce
    # absurd β magnitudes silently. compute_var_cvar's 20-obs floor
    # uses ann_vol > 0 elsewhere; mirror that convention here.
    if not np.isfinite(var_b) or var_b <= 1e-16:
        return np.nan
    return float(df["p"].cov(df["b"]) / var_b)


def compute_alpha_annual(p: pd.Series, b: pd.Series, window: int | None = None) -> float:
    """Annualized OLS alpha on RAW returns: (mean_p - β × mean_b) × 252.

    Deliberately not textbook CAPM α. Both inputs are raw returns, not
    excess returns over the risk-free rate. Textbook CAPM regresses
    (p - r_f) on (b - r_f), which gives α = α_raw - r_f · (1 - β); for
    β < 1 with positive r_f, this function therefore overstates true
    CAPM α by r_f · (1 - β) per year.

    The dashboard surfaces this disclosure on the Risk-tab β / α "how
    to read" panel; tile labels read "α (OLS intercept vs SPY)" rather
    than "α (CAPM)" so the caption matches the math. Do NOT silently
    flip this to excess-return regression — that would shift the
    displayed α by ~0.8 pp/yr at today's β / r_f.
    """
    df = _aligned(p, b, window)
    if len(df) < 2:
        return np.nan
    var_b = float(df["b"].var())
    if not np.isfinite(var_b) or var_b <= 1e-16:
        return np.nan
    beta = df["p"].cov(df["b"]) / var_b
    return float((df["p"].mean() - beta * df["b"].mean()) * 252.0)


def compute_up_down_beta(p: pd.Series, b: pd.Series,
                         window: int | None = None) -> tuple[float, float]:
    # Phase 1C audit: near-zero variance guard mirrors compute_beta. Up/Down
    # splits also drop SPY-flat days (`b == 0`), which is fine in practice —
    # SPY total-return rarely flat on a trading day — but worth a note for a
    # future maintainer eyeing the strict inequality.
    df = _aligned(p, b, window)
    if df.empty:
        return np.nan, np.nan
    up = df[df["b"] > 0]
    dn = df[df["b"] < 0]
    up_var = float(up["b"].var()) if len(up) > 2 else 0.0
    dn_var = float(dn["b"].var()) if len(dn) > 2 else 0.0
    up_b = (up["p"].cov(up["b"]) / up_var
            if np.isfinite(up_var) and up_var > 1e-16 else np.nan)
    dn_b = (dn["p"].cov(dn["b"]) / dn_var
            if np.isfinite(dn_var) and dn_var > 1e-16 else np.nan)
    return (float(up_b) if pd.notna(up_b) else np.nan,
            float(dn_b) if pd.notna(dn_b) else np.nan)


def rolling_beta(p: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS β of p on b over a trailing `window` of trading days.

    Returns a Series indexed by date; NaN for the first window-1 dates and
    for any window where b's variance is zero (degenerate)."""
    df = _aligned(p, b)
    cov = df["p"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    return cov / var.where(var > 0)


def rolling_alpha_annual(p: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling annualized α = (mean_p − β·mean_b)·252 over a trailing window."""
    df = _aligned(p, b)
    mp  = df["p"].rolling(window).mean()
    mb  = df["b"].rolling(window).mean()
    cov = df["p"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    beta = cov / var.where(var > 0)
    return (mp - beta * mb) * 252.0


def rolling_up_down_beta(p: pd.Series, b: pd.Series,
                         window: int,
                         min_obs: int = 10,
                         ) -> tuple[pd.Series, pd.Series]:
    """Rolling Up-β and Down-β over a trailing `window`.

    Each side conditions on the sign of b (SPY) inside the window and
    requires at least `min_obs` qualifying days, else NaN."""
    df = _aligned(p, b)
    n = len(df)
    pa, ba = df["p"].values, df["b"].values
    up_out = np.full(n, np.nan)
    dn_out = np.full(n, np.nan)
    for i in range(window - 1, n):
        pw = pa[i - window + 1 : i + 1]
        bw = ba[i - window + 1 : i + 1]
        um = bw > 0
        dm = bw < 0
        if um.sum() >= min_obs:
            bu = bw[um]
            vu = bu.var(ddof=1)
            if vu > 0:
                up_out[i] = np.cov(pw[um], bu, ddof=1)[0, 1] / vu
        if dm.sum() >= min_obs:
            bd = bw[dm]
            vd = bd.var(ddof=1)
            if vd > 0:
                dn_out[i] = np.cov(pw[dm], bd, ddof=1)[0, 1] / vd
    return (pd.Series(up_out, index=df.index, name="up_beta"),
            pd.Series(dn_out, index=df.index, name="dn_beta"))


def compute_var_cvar(returns: pd.Series,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Historical 1-day VaR and CVaR at the given tail probability. Returns
    are decimal (e.g. -0.02 for a -2% day). Both outputs are negative."""
    r = returns.dropna()
    if len(r) < 20:
        return np.nan, np.nan
    var = float(r.quantile(alpha))
    tail = r[r <= var]
    cvar = float(tail.mean()) if not tail.empty else np.nan
    return var, cvar


def aggregate_periodic_returns(returns: pd.Series, dates: pd.Series,
                               freq: str) -> tuple[pd.Series, pd.Series]:
    """Chain monthly returns into longer periods (Q or Y).

    Period return = ∏(1+Rᵢ) − 1 within each period. The plotted date is
    the period START (used as the anchor for plotly's xperiod centering).
    """
    if freq == "M":
        d = pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp()
        return (returns.reset_index(drop=True),
                d.reset_index(drop=True))
    df = pd.DataFrame({
        "r": returns.values,
        "d": pd.to_datetime(dates).values,
    })
    df["period"] = df["d"].dt.to_period(freq)
    out = (df.groupby("period", sort=True)
              .agg(r=("r", lambda x: float((1.0 + x).prod() - 1.0)))
              .reset_index())
    out["d"] = out["period"].dt.to_timestamp()
    return out["r"], out["d"]


def rolling_active_stats(port_m: pd.Series, bench_m: pd.Series,
                         window: int = 12) -> dict:
    """Rolling active-return consistency vs a benchmark (v2-S4).

    hit_rate_pct: share of rolling ``window``-month periods where the
    portfolio's cumulative return beat the benchmark's. tracking_error_pct:
    annualized std of monthly active returns. information_ratio: annualized
    mean active / TE (None when TE is 0). Inner-aligned; honest
    available:false below ``window`` aligned months."""
    p, b = port_m.align(bench_m, join="inner")
    p, b = p.dropna(), b.dropna()
    p, b = p.align(b, join="inner")
    n = int(len(p))
    if n < int(window):
        return {"available": False, "n_months": n}
    hits = 0
    n_win = n - int(window) + 1
    for i in range(n_win):
        pw = float((1 + p.iloc[i:i + window]).prod() - 1)
        bw = float((1 + b.iloc[i:i + window]).prod() - 1)
        if pw > bw:
            hits += 1
    active = p - b
    te = float(active.std(ddof=1)) * (12 ** 0.5)
    ir = (float(active.mean()) * 12 / te) if te > 0 else None
    return {"available": True, "n_windows": n_win,
            "hit_rate_pct": round(hits / n_win * 100, 1),
            "tracking_error_pct": round(te * 100, 2),
            "information_ratio": round(ir, 2) if ir is not None else None}
