# parsers/risk_bundle.py
"""Single source of the Risk-tab series bundle for BOTH UIs (Phase D finale).

The snapshot→weights fold and the risk-series bundle orchestration used to
live twice — ``app.py`` (``build_snapshot_weights`` / ``build_risk_series_bundle``)
and the MERIDIAN terminal (``holdings_service._snapshot_weights`` /
``risk_service._bundle``, plus partial copies in holdings_service and
performance_service). This module is the one implementation both import,
retiring Known Weak Point #4 (Streamlit↔terminal numeric drift) by
construction. Bodies are verbatim moves of the app.py implementations (the
superset side: filter-aware fold, ``weights_mv``, ``filter_state``).

Streamlit-free AND terminal-free by contract: engine-level imports only
(config_local, treasury_proxy, monthly_normalize, risk_metrics).
tests/test_risk_bundle.py pins this with an AST import check.

Inputs are trusted as prepared by each UI: ``twr_portfolio`` carries
``return_pct`` / ``wealth_index`` / ``twr_dd_pct`` + parsed dates (app.py's
load_twr / the terminal's ``_prepare_portfolio_twr``); ``bench_tr`` is the
forward-filled daily SPY total-return value series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config_local as cfg
from monthly_normalize import monthly_normalize, slice_as_of_month
from risk_metrics import (
    compute_synthesis_gaps,
    spy_monthly_returns_aligned,
    synthesize_portfolio_returns_historical,
    weights_per_snap_monthly,
)
from treasury_proxy import treasury_proxy as _treasury_proxy


def build_snapshot_weights(snap: pd.DataFrame,
                           bucket_filter: list[str] | None,
                           class_filter: list[str] | None,
                           daily_prices: pd.DataFrame,
                           ) -> tuple[pd.Series, pd.Series]:
    """(weights, weights_mv) for one positions snapshot — applies the
    Account / Asset-class filter, drops cash + options, then folds:
      * TLH sleeve → SPY (Parametric direct-indexing is one SPY-like
        decision; without the fold the 300+ component names would
        dominate concentration / contribution math).
      * Treasury Ladder rungs → per-rung duration bucket via
        _treasury_proxy: SGOV (<1y), SCHO (1-3y), IEI (3-7y), IEF
        (7-12y), TLT (12y+). (Pre-fold, every rung went to SGOV,
        understating rate exposure ~20x for rungs > 1y.)
      * Uncovered symbols (CUSIP-only bonds, anything missing from
        daily_prices) fall back to SGOV.
    Pass the full bucket/class choice lists to get UNFILTERED weights
    (the Factor tab's contract) — or ``None`` for either filter to skip
    that isin step entirely (byte-equivalent; the terminal's pre-filtered
    and unfiltered callers use this).
    """
    if snap.empty or daily_prices.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    snap_f = snap
    if bucket_filter is not None:
        snap_f = snap_f[snap_f["bucket"].isin(bucket_filter)]
    if class_filter is not None:
        snap_f = snap_f[snap_f["asset_class"].isin(class_filter)]
    is_option_ = snap_f["asset_class"].astype(str).str.startswith("option")
    snap_for_synth = snap_f[
        (snap_f["asset_class"] != "cash") & (~is_option_)
    ].copy()
    if snap_for_synth.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    tlh_mask = snap_for_synth["account_id"] == cfg.TLH_ACCOUNT_ID
    snap_for_synth.loc[tlh_mask, "symbol"] = "SPY"
    tlad_mask = snap_for_synth["bucket"] == "Treasury Ladder"
    if tlad_mask.any():
        as_of = pd.Timestamp(
            snap_for_synth.loc[tlad_mask, "statement_date"].max())
        snap_for_synth.loc[tlad_mask, "symbol"] = snap_for_synth.loc[
            tlad_mask].apply(
                lambda r: _treasury_proxy(r["description"], as_of), axis=1)
    sym_col = snap_for_synth["symbol"]
    uncovered = sym_col.isna() | (~sym_col.isin(daily_prices.columns))
    snap_for_synth.loc[uncovered, "symbol"] = "SGOV"
    wmv = snap_for_synth.groupby("symbol")["market_value"].sum()
    wmv = wmv[wmv > 0]
    if wmv.sum() <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return (wmv / wmv.sum()), wmv


def daily_portfolio_returns(positions: pd.DataFrame,
                            daily_prices: pd.DataFrame,
                            bucket_filter: list[str] | None,
                            class_filter: list[str] | None) -> pd.Series:
    """Daily portfolio returns via HONEST historical synthesis: each calendar
    month contributes one daily segment using the weights the portfolio
    actually held that month (per-month snapshots via monthly_normalize +
    weights_per_snap_monthly, so dual-date statement months collapse to a
    single snapshot — WSF-1). Empty Series when daily prices are absent or
    no month folds to usable weights."""
    if daily_prices.empty:
        return pd.Series(dtype=float)
    weights_per_snap = weights_per_snap_monthly(
        monthly_normalize(positions),
        lambda snap_m: build_snapshot_weights(
            snap_m, bucket_filter, class_filter, daily_prices)[0],
    )
    return (synthesize_portfolio_returns_historical(
                weights_per_snap, daily_prices)
            if weights_per_snap else pd.Series(dtype=float))


def spy_daily_returns(daily_prices: pd.DataFrame,
                      bench_tr: pd.Series) -> pd.Series:
    """Daily SPY returns from the TR value series (reindexed to trading days
    so weekend forward-fills drop out), falling back to price-only SPY when
    the TR series is unavailable."""
    if not daily_prices.empty and bench_tr is not None and not bench_tr.empty:
        spy_tr_trading = bench_tr.reindex(daily_prices.index, method="ffill")
        return spy_tr_trading.pct_change().dropna()
    if not daily_prices.empty and "SPY" in daily_prices.columns:
        return daily_prices["SPY"].pct_change().dropna()
    return pd.Series(dtype=float)


def synthesize_monthly_returns(port_rets: pd.Series,
                               twr_frame: pd.DataFrame) -> pd.Series:
    """Monthly returns compounded from daily returns within each
    (prev_stmt_date → statement_date] window. The first month (no prior
    boundary) emits NaN so the wealth-index baseline aligns with the
    unfiltered TWR path; an empty window emits NaN. Empty Series when either
    input is empty."""
    if twr_frame is None or twr_frame.empty or port_rets.empty:
        return pd.Series(dtype=float)
    synth: list[tuple[pd.Timestamp, float]] = []
    for _, row in twr_frame.sort_values("statement_date").iterrows():
        prev = row.get("prev_stmt_date")
        end = row["statement_date"]
        if pd.isna(end):
            continue
        if pd.isna(prev):
            synth.append((pd.Timestamp(end).normalize(), np.nan))
            continue
        window = port_rets[
            (port_rets.index > pd.Timestamp(prev))
            & (port_rets.index <= pd.Timestamp(end))
        ]
        ret = (float((1.0 + window).prod() - 1.0)
               if not window.empty else np.nan)
        synth.append((pd.Timestamp(end).normalize(), ret))
    if not synth:
        return pd.Series(dtype=float)
    return pd.Series(
        [v for _, v in synth],
        index=pd.DatetimeIndex([d for d, _ in synth]),
        name="monthly",
    )


def build_risk_series_bundle(
    positions: pd.DataFrame,
    positions_monthly: pd.DataFrame,
    latest_dt: pd.Timestamp | None,
    bucket_filter: list[str] | None,
    class_filter: list[str] | None,
    account_active: bool,
    class_active: bool,
    daily_prices: pd.DataFrame,
    bench_tr: pd.Series,
    twr_portfolio: pd.DataFrame,
) -> dict:
    """Build every filtered return series the Risk tabs consume.

    KPI tiles AND the time-series charts read from one bundle so a
    metric/chart pair can't drift out of sync on filter change. Applies
    the Account + Asset-class filters here (the Broker filter is already
    applied upstream). When either filter is active, monthly Pass 1 series
    are synthesized by compounding daily returns within each
    (prev_stmt_date → statement_date) window — statement-based TWR has no
    per-account or per-class slice.

    Returns a dict with:
      - latest_snap      filtered latest snapshot (pre cash/option drop)
      - weights          renormalized symbol weights for daily synthesis
      - weights_mv       pre-normalization $ market value by symbol
      - port_rets        daily portfolio returns (decimal)
      - spy_rets         daily SPY returns (decimal)
      - monthly          monthly portfolio returns indexed by statement_date
      - monthly_source   "twr" when unfiltered, "synthetic" when filtered
      - spy_monthly      SPY monthly returns aligned to statement_date
      - wealth_index     monthly cumulative wealth (matches twr when unfilt.)
      - dd_full_pct      drawdown % at every monthly point (≤ 0)
      - spy_dd_full_pct  same for SPY
      - nav_latest       filtered NAV at latest_dt (for $-drop calcs)
      - synthesis_gaps   per-symbol n_days_no_price coverage diagnostic
      - filter_state     {account_active, class_active}
    """
    # Carry-forward + MTM snapshot (positions_monthly) so accounts lagging
    # the statement frontier stay in the current-state risk universe instead
    # of silently dropping out (WSG-1). slice_as_of_month folds a dual-date
    # frontier month to its canonical date; the historical-series synthesis
    # below still reads the raw `positions` frame (as-reported).
    snap_all = (slice_as_of_month(positions_monthly, latest_dt)
                if latest_dt is not None else positions_monthly.iloc[0:0])
    latest_snap = snap_all[
        snap_all["bucket"].isin(bucket_filter
                                if bucket_filter is not None
                                else snap_all["bucket"].unique())
        & snap_all["asset_class"].isin(class_filter
                                       if class_filter is not None
                                       else snap_all["asset_class"].unique())
    ].copy()
    nav_latest = float(latest_snap["market_value"].sum())

    # Latest-snapshot weights — used by every "current state" tile
    # (concentration, contribution decomposition, ES, beta, etc.).
    weights, weights_mv = build_snapshot_weights(
        snap_all, bucket_filter, class_filter, daily_prices)

    # Coverage-gap diagnostic: synthesize_portfolio_returns treats missing
    # prices as 0% return, which biases portfolio vol toward zero for
    # sparse-coverage symbols; surfaced instead of running silently.
    synthesis_gaps = (compute_synthesis_gaps(weights, daily_prices)
                      if not weights.empty and not daily_prices.empty
                      else pd.DataFrame())

    port_rets = daily_portfolio_returns(
        positions, daily_prices, bucket_filter, class_filter)
    spy_rets = spy_daily_returns(daily_prices, bench_tr)
    # Overlapping window only: SPY history predating the first portfolio
    # return (e.g. a pre-inception crash) must not feed portfolio-vs-SPY
    # stats — the VaR/CVaR/worst-day tiles consumed the full head until
    # the 2026-07 TK feedback round.
    if not port_rets.empty and not spy_rets.empty:
        spy_rets = spy_rets[spy_rets.index >= port_rets.index[0]]

    # Monthly returns — twr_portfolio when unfiltered (byte-identical to the
    # legacy Pass 1 path); synthesized from daily when a filter is active.
    filter_active = account_active or class_active
    if not filter_active and not twr_portfolio.empty:
        rport_sorted = twr_portfolio.sort_values("statement_date") \
                                    .reset_index(drop=True)
        dt_index = pd.DatetimeIndex(rport_sorted["statement_date"].values)
        monthly = pd.Series(
            rport_sorted["return_pct"].astype(float).values,
            index=dt_index, name="monthly",
        )
        wealth_index = pd.Series(
            rport_sorted["wealth_index"].astype(float).values,
            index=dt_index, name="wealth",
        )
        dd_full_pct = pd.Series(
            rport_sorted["twr_dd_pct"].astype(float).values,
            index=dt_index, name="dd_pct",
        )
        monthly_source = "twr"
    else:
        monthly = synthesize_monthly_returns(port_rets, twr_portfolio)
        if monthly.dropna().empty:
            wealth_index = pd.Series(dtype=float, name="wealth")
            dd_full_pct = pd.Series(dtype=float, name="dd_pct")
        else:
            # fillna(0.0) treats NaN months as 0% so the ITD wealth /
            # drawdown paths flat-fill statement gaps rather than dropping
            # them; windowed drawdowns use the raw monthly series and are
            # unaffected.
            wealth_index = ((1.0 + monthly.fillna(0.0)).cumprod()
                            ).rename("wealth")
            dd_full_pct = ((wealth_index / wealth_index.cummax() - 1.0)
                           * 100.0).rename("dd_pct")
        monthly_source = "synthetic"

    # SPY monthly — statement-date aligned, independent of the filter, so
    # SPY metrics read on the same windows as the portfolio metrics.
    spy_monthly = spy_monthly_returns_aligned(twr_portfolio, bench_tr)
    if not spy_monthly.empty:
        spy_wealth = (1.0 + spy_monthly).cumprod()
        spy_dd_full_pct = ((spy_wealth / spy_wealth.cummax() - 1.0) * 100.0)
    else:
        spy_dd_full_pct = pd.Series(dtype=float)

    return {
        "latest_snap":     latest_snap,
        "weights":         weights,
        "weights_mv":      weights_mv,
        "port_rets":       port_rets,
        "spy_rets":        spy_rets,
        "monthly":         monthly,
        "monthly_source":  monthly_source,
        "spy_monthly":     spy_monthly,
        "wealth_index":    wealth_index,
        "dd_full_pct":     dd_full_pct,
        "spy_dd_full_pct": spy_dd_full_pct,
        "nav_latest":      nav_latest,
        "synthesis_gaps":  synthesis_gaps,
        "filter_state": {
            "account_active": account_active,
            "class_active":   class_active,
        },
    }
