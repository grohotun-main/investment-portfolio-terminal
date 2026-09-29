"""Single-name protective-put hedge engine (pure math, no I/O, no network).

Spec: docs/superpowers/specs/2026-07-06-single-name-hedge-report-design.md.
Premiums are per share; one contract wraps 100 shares. A package "floors"
loss at X only when spot - strike + buy_price <= X * spot AND the put's
expiry is at/after the planned sale date (locked decisions 3-5). Values are
intrinsic worst-case by design (decision 9); pre-expiry marks live in
crash_mark_table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dip_analytics import stationary_block_bootstrap_ci
from options_pricer import binomial_american
from tail_risk import fit_gpd_tail

TRADING_DAYS_PER_YEAR = 252
CAL_DAYS_PER_YEAR = 365

# Named crash windows for replays / crash-vol anchoring (market-wide dates;
# the stock's own series is sliced inside them).
REPLAY_WINDOWS: dict[str, tuple[str, str]] = {
    "2000-02 dot-com": ("2000-03-10", "2002-10-09"),
    "2007-09 financial crisis": ("2007-10-09", "2009-03-09"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-14"),
}


@dataclass(frozen=True)
class PutQuote:
    """One normalized chain row (puts only)."""
    contract_ticker: str
    expiry: date
    strike: float
    bid: float | None
    ask: float | None
    last_price: float | None   # Polygon day close
    iv: float | None
    delta: float | None
    open_interest: int


@dataclass(frozen=True)
class FloorPackage:
    floor_pct: float            # e.g. 0.10
    contracts: int              # shares // 100
    quote: PutQuote
    buy_price: float            # per share (ask when live, else day close)
    stale_quote: bool
    total_cost: float           # buy_price * contracts * 100
    cost_pct: float             # total_cost / (spot * shares)
    guaranteed_value: float     # (strike - buy_price) * contracts * 100
    guaranteed_loss_pct: float  # (spot - strike + buy_price) / spot
    market_implied_prob: float | None  # P(S_T < strike), N(-d2) at quote IV
    hist_prob: dict | None      # horizon_loss_odds entry for this floor


@dataclass(frozen=True)
class KickerPackage:
    """Deliberately NOT a FloorPackage — it guarantees nothing."""
    quote: PutQuote
    contracts: int              # may exceed shares // 100 (payout amplifier)
    buy_price: float
    stale_quote: bool
    total_cost: float
    payout_per_10pct_below_strike: float  # contracts * 100 * spot * 0.10


def _clean(x) -> float | None:
    """None for missing/NaN/inf, float otherwise."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def normalize_chain(raw_rows) -> list[PutQuote]:
    """Filter flattened snapshot rows (fetch_options_chains._flatten schema)
    to usable put quotes. Drops calls, unparseable expiries, and rows with no
    price signal at all (bid, ask and day-close absent/zero). Zero-OI rows
    are kept (OI is display/kicker-gating data, not a validity test)."""
    out: list[PutQuote] = []
    for r in raw_rows:
        if r.get("contract_type") != "put":
            continue
        exp_raw = r.get("expiration_date")
        try:
            expiry = date.fromisoformat(str(exp_raw)[:10])
        except (TypeError, ValueError):
            continue
        strike = _clean(r.get("strike"))
        if strike is None or strike <= 0:
            continue
        bid = _clean(r.get("polygon_bid"))
        ask = _clean(r.get("polygon_ask"))
        last = _clean(r.get("polygon_price"))
        if not any(p is not None and p > 0 for p in (bid, ask, last)):
            continue
        oi = _clean(r.get("polygon_open_interest"))
        out.append(PutQuote(
            contract_ticker=str(r.get("contract_ticker") or ""),
            expiry=expiry,
            strike=float(strike),
            bid=bid,
            ask=ask,
            last_price=last,
            iv=_clean(r.get("polygon_iv")),
            delta=_clean(r.get("polygon_delta")),
            open_interest=int(oi) if oi is not None else 0,
        ))
    return out


def buy_price_of(q: PutQuote) -> tuple[float | None, bool]:
    """Per-share price to assume for a BUY, plus a stale flag.

    Ask when live (> 0); else the day close, flagged stale; None when the
    row is bid-only (nothing to buy against — sell-side signal only)."""
    if q.ask is not None and q.ask > 0:
        return q.ask, False
    if q.last_price is not None and q.last_price > 0:
        return q.last_price, True
    return None, True


def filter_stale_dominated(quotes: list[PutQuote], *, tol: float = 0.02
                           ) -> tuple[list[PutQuote], int]:
    """Drop same-expiry quotes whose BUY price sits below the running max of
    lower strikes (a put at a higher strike is worth strictly more, so such
    prints are stale/unexecutable bargains — observed live on thin
    off-hours day-closes). tol is relative slack for rounding-level
    inversions. Quotes with no buy price pass through (they can't be
    bought, so they can't poison the solver). Returns (kept, n_dropped)."""
    by_expiry: dict = {}
    for q in quotes:
        by_expiry.setdefault(q.expiry, []).append(q)
    kept: list[PutQuote] = []
    dropped = 0
    for group in by_expiry.values():
        run_max = 0.0
        for q in sorted(group, key=lambda x: x.strike):
            bp, _ = buy_price_of(q)
            if bp is None:
                kept.append(q)
                continue
            if bp < run_max * (1.0 - tol):
                dropped += 1
                continue
            run_max = max(run_max, bp)
            kept.append(q)
    return kept, dropped


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def market_implied_prob(spot: float, strike: float, T: float, r: float,
                        q: float, iv: float | None) -> float | None:
    """Risk-neutral P(S_T < strike) = N(-d2) at the contract's own IV.
    None when IV is missing/degenerate. Presented in the report as "the odds
    the market charges" — includes a fear premium, not a forecast."""
    if iv is None or not (iv > 0.0) or T <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return None
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return _ncdf(-d2)


def solve_floor(quotes: list[PutQuote], spot: float, shares: int,
                floor_pct: float, sell_by: date) -> FloorPackage | None:
    """Cheapest put whose PREMIUM-INCLUSIVE worst case clears the floor.

    Candidates: expiry >= sell_by (decision 3), any strike incl. ITM
    (decision 4). Feasible when spot - strike + buy_price <= floor_pct*spot.
    Ties break to the lower strike, then the earlier expiry. None when no
    listed strike clears the floor."""
    contracts = shares // 100
    if contracts <= 0 or spot <= 0.0:
        return None
    best_key: tuple | None = None
    best: tuple[PutQuote, float, bool] | None = None
    for q in quotes:
        if q.expiry < sell_by:
            continue
        bp, stale = buy_price_of(q)
        if bp is None:
            continue
        if spot - q.strike + bp > floor_pct * spot + 1e-9:
            continue
        key = (bp, q.strike, q.expiry.toordinal(), stale)
        if best_key is None or key < best_key:
            best_key, best = key, (q, bp, stale)
    if best is None:
        return None
    q, bp, stale = best
    covered = contracts * 100
    total_cost = bp * covered
    return FloorPackage(
        floor_pct=floor_pct,
        contracts=contracts,
        quote=q,
        buy_price=bp,
        stale_quote=stale,
        total_cost=total_cost,
        cost_pct=total_cost / (spot * shares),
        guaranteed_value=(q.strike - bp) * covered,
        guaranteed_loss_pct=(spot - q.strike + bp) / spot,
        market_implied_prob=None,   # attached by build_floor_menu
        hist_prob=None,             # attached by build_floor_menu
    )


def build_floor_menu(quotes: list[PutQuote], spot: float, shares: int,
                     sell_by: date, floors: list[float], *, today: date,
                     r: float, q_yield: float,
                     hist_odds: dict | None = None
                     ) -> list[FloorPackage | None]:
    """One solve per requested floor (fraction, e.g. 0.10). Attaches the
    market-implied and (when supplied) historical odds per package. Entries
    are None for infeasible floors — the report renders those honestly."""
    menu: list[FloorPackage | None] = []
    for f in floors:
        pkg = solve_floor(quotes, spot, shares, f, sell_by)
        if pkg is not None:
            T = max((pkg.quote.expiry - today).days, 0) / CAL_DAYS_PER_YEAR
            mip = market_implied_prob(spot, pkg.quote.strike, T, r, q_yield,
                                      pkg.quote.iv)
            hp = (hist_odds or {}).get(f)
            pkg = replace(pkg, market_implied_prob=mip, hist_prob=hp)
        menu.append(pkg)
    return menu


def kicker_package(quotes: list[PutQuote], spot: float, shares: int,
                   sell_by: date, *, otm_pct: float, budget: float,
                   replay_price: float | None = None
                   ) -> KickerPackage | None:
    """Crash kicker: deep-OTM put (strike <= spot*(1-otm_pct)), expiry >=
    sell_by, open interest > 0. Contracts = floor(budget / cost); None when
    nothing affordable/listed. The kicker is sized to pay the most if the
    stock's OWN worst historical crash repeated from today's price: when
    replay_price (spot marked to that crash's trough) is known and a listed
    strike sits above it, the pick maximizes the payout at replay_price per
    the budget. Cheapest-deepest is only the fallback when no replay level
    is known or listed strikes all sit below it — cheapest-first alone
    degenerates on big run-ups into strikes so deep even the worst replay
    leaves them worthless. A payout amplifier, not coverage — it never
    substitutes for the full-coverage core (decision 5)."""
    if spot <= 0.0 or budget <= 0.0:
        return None
    cutoff = spot * (1.0 - otm_pct)
    cands: list[tuple[PutQuote, float, bool]] = []
    for q in quotes:
        if q.expiry < sell_by or q.strike > cutoff or q.open_interest <= 0:
            continue
        bp, stale = buy_price_of(q)
        if bp is None or bp <= 0:
            continue
        cands.append((q, bp, stale))
    best_key: tuple | None = None
    best: tuple[PutQuote, float, bool] | None = None
    if replay_price is not None:
        for q, bp, stale in cands:
            if q.strike <= replay_price:
                continue
            contracts = int((budget + 1e-9) // (bp * 100.0))
            if contracts < 1:
                continue
            payout = max(q.strike - replay_price, 0.0) * 100.0 * contracts
            key = (-payout, contracts * 100.0 * bp, q.strike)
            if best_key is None or key < best_key:
                best_key, best = key, (q, bp, stale)
    if best is None:
        # fallback: cheapest deep-OTM put (no replay level known/cleared)
        best_key = None
        for q, bp, stale in cands:
            key = (bp, q.strike)
            if best_key is None or key < best_key:
                best_key, best = key, (q, bp, stale)
    if best is None:
        return None
    q, bp, stale = best
    contracts = int((budget + 1e-9) // (bp * 100.0))
    if contracts < 1:
        return None
    return KickerPackage(
        quote=q, contracts=contracts, buy_price=bp, stale_quote=stale,
        total_cost=contracts * 100.0 * bp,
        payout_per_10pct_below_strike=contracts * 100.0 * spot * 0.10,
    )


def payoff_grid(spot: float, shares: int, core: FloorPackage,
                kicker: KickerPackage | None,
                s_grid: list[float] | np.ndarray) -> pd.DataFrame:
    """Position value AT EXPIRY vs terminal price. Intrinsic-only put
    valuation — the deliberate worst case (decision 9): any pre-expiry sale,
    leftover time value or crash-IV spike lands ABOVE these lines."""
    s = np.asarray(list(s_grid), dtype=float)
    unhedged = shares * s
    core_pay = core.contracts * 100.0 * np.maximum(core.quote.strike - s, 0.0)
    hedged = unhedged + core_pay - core.total_cost
    out = pd.DataFrame({"price": s, "unhedged": unhedged, "hedged": hedged})
    if kicker is not None:
        k_pay = kicker.contracts * 100.0 * np.maximum(kicker.quote.strike - s, 0.0)
        out["hedged_kicker"] = hedged + k_pay - kicker.total_cost
    return out


def crash_replays(price: pd.Series,
                  windows: dict[str, tuple[str, str]] = REPLAY_WINDOWS
                  ) -> list[dict]:
    """The stock's OWN peak-to-trough drawdown inside each named window.
    Windows are skipped when they have fewer than 5 observations (ticker
    too young / history not covering them) or when the history's first
    observation lands more than 10 calendar days after the window start
    (a partial peak understates the crash — cummax would anchor at the
    segment's own first value, not the window's true pre-crash high).
    drawdown_pct is negative."""
    out: list[dict] = []
    p = price.dropna().sort_index()
    for label, (start, end) in windows.items():
        seg = p.loc[start:end]
        if len(seg) < 5:
            continue
        if (seg.index[0] - pd.Timestamp(start)).days > 10:
            continue   # history starts mid-window: a partial peak understates the crash
        dd = float((seg / seg.cummax() - 1.0).min())
        out.append({"label": label, "start": start, "end": end,
                    "drawdown_pct": dd})
    return out


_GPD_MIN_EMPIRICAL_HITS = 10   # below this, the GPD tail speaks instead
_TWENTY_YEARS_TD = 20 * TRADING_DAYS_PER_YEAR


def _window_returns(price: pd.Series, horizon_td: int) -> pd.Series:
    p = price.dropna()
    return (p.shift(-horizon_td) / p - 1.0).dropna()


def _gpd_exceed_prob(fit: dict, x: float, n_windows: int) -> float | None:
    """UNCONDITIONAL P(loss magnitude > x) under a POT/GPD fit from
    tail_risk.fit_gpd_tail, scaled per the TOTAL window count (n_windows) so
    it matches the empirical branch's basis. The fit is trained on loss
    windows only, so its own n_total counts losses — dividing by that would
    give P(magnitude > x | loss window) and overstate deep-floor odds.
    None when n_windows is not positive, x is below the fitted threshold, OR
    the fit itself is unusable (fit_gpd_tail returns a NaN sentinel on
    insufficient data — NaN comparisons are False, so finiteness must be
    checked explicitly)."""
    if n_windows <= 0:
        return None
    xi, beta, u = fit["xi"], fit["beta"], fit["threshold"]
    if (not (math.isfinite(u) and math.isfinite(beta) and math.isfinite(xi))
            or beta <= 0 or x <= u):
        return None
    z = (x - u) / beta
    if abs(xi) < 1e-8:
        tail = math.exp(-z)
    else:
        base = 1.0 + xi * z
        if base <= 0.0:
            return 0.0   # beyond the fitted support (finite-endpoint GPD)
        tail = base ** (-1.0 / xi)
    p = float(fit["n_exceedances"]) / float(n_windows) * tail
    return p if math.isfinite(p) else None


def horizon_loss_odds(price: pd.Series, horizon_td: int,
                      floors: list[float]) -> dict:
    """Per floor X: P(point-to-point return over horizon_td trading days
    <= -X), from the stock's own history.

    Empirical overlapping-window frequency when >= _GPD_MIN_EMPIRICAL_HITS
    windows hit; GPD tail (POT on window-loss magnitudes) for the sparse deep
    end; CI via the stationary block bootstrap on DAILY returns with the
    window-frequency statistic; prob_20y = the last-20-years empirical rate
    as a regime robustness check (equals full-sample when history is
    shorter). source in {"empirical","gpd","none"}; "confident" mirrors the
    GPD gate (>= 30 exceedances) and is True on the empirical branch."""
    wr = _window_returns(price, horizon_td)
    daily = price.dropna().pct_change().dropna()
    p20 = price.dropna().iloc[-_TWENTY_YEARS_TD:]
    wr20 = _window_returns(p20, horizon_td)
    losses = (-wr[wr < 0]).to_numpy(dtype=float)
    fit = fit_gpd_tail(losses) if losses.size else None
    out: dict[float, dict] = {}
    for f in floors:
        n = int(len(wr))
        hits = int((wr <= -f).sum())
        emp = hits / n if n else float("nan")
        d: dict = {"n_windows": n, "n_hits": hits,
                   "prob_20y": float((wr20 <= -f).mean()) if len(wr20) else float("nan")}
        if hits >= _GPD_MIN_EMPIRICAL_HITS:
            d.update(prob=emp, source="empirical", confident=True)
        else:
            g = _gpd_exceed_prob(fit, f, n) if fit else None
            if g is not None:
                d.update(prob=g, source="gpd",
                         confident=bool(fit.get("confident", False)))
            else:
                d.update(prob=emp if n else float("nan"), source="none",
                         confident=False)

        def _stat(resampled: pd.Series, _f=f) -> float:
            wealth = (1.0 + pd.Series(resampled).reset_index(drop=True)).cumprod()
            w = (wealth.shift(-horizon_td) / wealth - 1.0).dropna()
            return float((w <= -_f).mean()) if len(w) else float("nan")

        if len(daily) > horizon_td + 10:
            ci = stationary_block_bootstrap_ci(_stat, daily)
            d["lo"], d["hi"] = float(ci["lo"]), float(ci["hi"])
        else:
            d["lo"], d["hi"] = float("nan"), float("nan")
        out[f] = d
    return out


_IV_MIN_ROWS = 250        # ~1 trading year at realistic ~75% ATM-print fill
_IV_MIN_SPAN_DAYS = 540   # ~1.5 calendar years of coverage


def _percentile_of_last(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float((s <= value).mean() * 100.0)


def vol_context(price: pd.Series, iv_today: float | None,
                vix: pd.Series | None = None,
                iv_series: pd.Series | None = None, *,
                rv_window: int = 21,
                crash_windows: dict[str, tuple[str, str]] = REPLAY_WINDOWS
                ) -> dict:
    """IV percentile companions (decisions 10-11). Pure — the IV series is
    fetched by the CLI layer and passed in. iv_source: "true" when a dense
    enough (>= _IV_MIN_ROWS rows) AND long enough (>= _IV_MIN_SPAN_DAYS of
    calendar span) backed-out IV history ranks today's point, else "proxy"
    (RV + VIX percentiles carry the context)."""
    if iv_today is not None and isinstance(iv_today, float) and math.isnan(iv_today):
        iv_today = None
    p = price.dropna()
    rets = p.pct_change()
    rv = rets.rolling(rv_window).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    rv = rv.dropna()
    rv_today = float(rv.iloc[-1]) if len(rv) else float("nan")

    peaks: dict[str, float] = {}
    for label, (start, end) in crash_windows.items():
        seg = rv.loc[start:end]
        if len(seg) >= 5:
            peaks[label] = float(seg.max())
    rv_anchor = float(np.median(list(peaks.values()))) if peaks else (
        float(rv.quantile(0.999)) if len(rv) else float("nan"))

    ivs_all = iv_series.dropna() if iv_series is not None else None
    have_true = (ivs_all is not None and len(ivs_all) >= _IV_MIN_ROWS
                 and (ivs_all.index.max() - ivs_all.index.min()).days
                     >= _IV_MIN_SPAN_DAYS)
    if have_true:
        ivs = ivs_all
        iv_pct = (_percentile_of_last(ivs, float(iv_today))
                  if iv_today is not None else None)
        anchor = max(float(ivs.quantile(0.995)), rv_anchor)
        hist_start = (ivs.index[0].date() if hasattr(ivs.index[0], "date")
                     else pd.to_datetime(ivs.index[0]).date())
        hist_high = float(ivs.max())
        source = "true"
    else:
        iv_pct, anchor, hist_start, hist_high, source = (
            None, rv_anchor, None, None, "proxy")

    vix_today = vix_pct = None
    if vix is not None and len(vix.dropna()):
        v = vix.dropna()
        vix_today = float(v.iloc[-1])
        vix_pct = _percentile_of_last(v, vix_today)

    return {
        "rv_today": rv_today,
        "rv_percentile": _percentile_of_last(rv, rv_today),
        "crash_window_peaks": peaks,
        "iv_percentile": iv_pct,
        "iv_source": source,
        "iv_history_start": hist_start,
        "iv_history_high": hist_high,
        "crash_iv_anchor": anchor,
        "iv_today": iv_today,
        "iv_rv_spread": (float(iv_today) - rv_today
                         if iv_today is not None and not math.isnan(rv_today)
                         else None),
        "vix_today": vix_today,
        "vix_percentile": vix_pct,
    }


_IV_SCENARIOS = ((0.0, "fear unchanged"),
                 (0.5, "halfway to crash level"),
                 (1.0, "at crash level"))


def crash_mark_table(package: FloorPackage, *, today: date, spot: float,
                     r: float, q: float, iv_today: float | None,
                     crash_iv_anchor: float,
                     drop_pcts: tuple[float, ...] = (0.20, 0.40, 0.60),
                     at_days_list: tuple[int, ...] = (21, 63)
                     ) -> pd.DataFrame:
    """Mark-to-model value of the core puts sold INTO a crash (decision 9),
    on the grid drop x elapsed-trading-days x state-anchored IV scenario
    (decision 10): iv(t) = max(iv_today, iv_today + t*(anchor - iv_today)),
    t in {0, 0.5, 1}. Every cell reports the excess over intrinsic. American
    lower bound holds: value >= intrinsic in every cell."""
    if iv_today is None or not math.isfinite(iv_today):
        iv_today = crash_iv_anchor   # degenerate input: all columns coincide
    if not math.isfinite(crash_iv_anchor):
        # NaN anchor (e.g. no usable price history) must not poison the lerp:
        # max(x, nan) silently returns x, faking a "no room to go" clamp.
        crash_iv_anchor = iv_today if iv_today is not None else float("nan")
    if iv_today is None or not math.isfinite(iv_today):
        return pd.DataFrame(columns=["drop_pct", "at_td", "iv_scenario",
                                     "iv_used", "value", "intrinsic",
                                     "excess"])
    k = package.quote.strike
    scale = package.contracts * 100.0
    rows: list[dict] = []
    for at_td in at_days_list:
        cal_elapsed = round(at_td * CAL_DAYS_PER_YEAR / TRADING_DAYS_PER_YEAR)
        t_rem = max((package.quote.expiry - today).days - cal_elapsed, 0) / CAL_DAYS_PER_YEAR
        for drop in drop_pcts:
            s_crash = spot * (1.0 - drop)
            intrinsic = max(k - s_crash, 0.0) * scale
            for t, label in _IV_SCENARIOS:
                iv_used = max(iv_today, iv_today + t * (crash_iv_anchor - iv_today))
                per_share = binomial_american(s_crash, k, t_rem, r, q,
                                              iv_used, "put")["price"]
                value = per_share * scale
                rows.append({"drop_pct": drop, "at_td": at_td,
                             "iv_scenario": label, "iv_used": iv_used,
                             "value": value, "intrinsic": intrinsic,
                             "excess": value - intrinsic})
    return pd.DataFrame(rows)
