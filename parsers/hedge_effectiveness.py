"""Hedge effectiveness back-test for the long-put sleeve.

Given the transaction log + historical option closes pulled by
``fetch_option_history.py``, build a daily sleeve MV time series:

    sleeve_mv(d) = Σ qty(d) × 100 × close(d)
                   for each lot open on day d

Then identify SPY drawdown episodes and quantify how much the sleeve
actually paid out during each.

The back-test answers: when SPY drops X%, what did the put book gain?

Data source
-----------
Polygon ``/v2/aggs/ticker/{O:TICKER}/range/1/day/{from}/{to}`` returns
daily OHLCV per option contract — available on Options Starter ($29/mo)
tier with ≤2 years history. The price field is the actual market close
on each day, so MV is audited rather than model-priced. No IV / pricer
machinery needed.

Limitations
-----------
* Polygon coverage is bar-level; on illiquid days (volume = 0 or
  trivial), the close may be stale. Forward-fill in
  ``lookup_close_at`` handles holes.
* Pre-sleeve days (before the first BUY) yield sleeve_mv = 0.
* Lots that round-trip closed contribute 0 after the close date.
* ``data/option_history.csv`` must be refreshed when new lots open —
  run ``py parsers/fetch_option_history.py --write`` (or click
  "Refresh option history" in the dashboard).

Empirically (2026-05-25): mean abs error vs statement MVs = 2.9%,
mostly attributable to broker mid-vs-close marking conventions.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# Allow both bare imports (when this module is loaded from inside parsers/)
# and qualified imports (when loaded from project root via tests).
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from option_positions import (  # noqa: E402
    _parse_full_option_txn,
    parse_jpm_buy_desc,
    parse_jpm_option_desc,
)

CONTRACT_MULT = 100

# Drawdown episode detection threshold (peak-to-trough decline in pct).
DEFAULT_DRAWDOWN_THRESHOLD_PCT = 3.0


@dataclass
class Lot:
    """One book-level option lot life — opens on first BUY, dies on full
    close or expiry.

    For sleeve aggregation we treat each (opt_type, underlying, expiry,
    strike) key as a single rolling book-level position, aggregating across
    accounts. Account is dropped because JPM occasionally records a BUY in
    one account and the resulting position in another, and because two real
    accounts can hold the same put for the same expiry+strike (e.g. SPY
    575P 12/18/26 is in both JPM and Fidelity). The sleeve back-test is
    about total exposure, not per-account books.

    open_premium is the first BUY's per-share premium (audit-trail only;
    the daily reprice uses historical market close, not the model-priced
    extrapolation from open_premium).
    """
    opt_type: str
    underlying: str
    expiry: date
    strike: float
    open_date: pd.Timestamp           # first BUY settlement
    close_date: Optional[pd.Timestamp]  # date qty first hits 0 (None = open)
    open_qty: float                   # qty after first BUY
    open_premium: float               # per-share premium at open (audit)
    qty_changes: list = field(default_factory=list)
    # qty_changes is a list of (date, signed_delta) — sorted ascending.
    # Used at reprice time to determine qty(d).

    def qty_at(self, d: pd.Timestamp) -> float:
        """Net qty held at end-of-day d."""
        if d < self.open_date:
            return 0.0
        # Past close_date the lot contributes nothing.
        if self.close_date is not None and d > self.close_date:
            return 0.0
        # Past expiry the lot is gone (broker writes off; we keep cleanly to 0).
        if d > pd.Timestamp(self.expiry):
            return 0.0
        q = 0.0
        for chg_date, delta in self.qty_changes:
            if chg_date <= d:
                q += delta
            else:
                break
        return max(q, 0.0)


def _is_put_buy_or_sell(row: pd.Series) -> bool:
    tt = str(row.get("transaction_type") or "").lower()
    if tt not in ("buy", "sell"):
        return False
    desc = row.get("description")
    if not isinstance(desc, str):
        return False
    return bool(re.match(r"^\s*PUT\b", desc, re.IGNORECASE))


def build_strike_resolver(positions: pd.DataFrame) -> dict[tuple, set]:
    """Build a (opt_type, underlying, expiry) -> set of strikes map from
    option_put positions.csv.

    Used to recover strikes for JPM PDF-parsed BUY/SELL transaction
    descriptions that omit the strike (e.g. ``PUT SPY 12/18/26 ETF OPEN
    CONTRACT``). Statement positions DO carry the strike (JPM puts it in
    the position description), so historical sleeve members were always
    on at least one statement before close — we can recover the strike
    by matching on (type, underlying, expiry).

    Account is intentionally dropped from the key: JPM occasionally
    records a BUY in one account and the resulting position in another
    (book-and-allocate), so requiring account match would orphan those
    rows. The sleeve doesn't trade spreads, so (type, ul, expiry) is
    unique to a single strike in practice.

    Test broker accounts (broker.startswith("Test") or
    broker.endswith(" Test")) are excluded to keep reconciliation
    fixtures out of the resolver.

    Returns: dict keyed by 3-tuple, values are sets of strikes (>1
    means ambiguous — caller skips).
    """
    if positions is None or positions.empty:
        return {}
    pos = positions.copy()
    pos = pos[pos["asset_class"].fillna("").str.startswith("option")]
    if pos.empty:
        return {}
    broker = pos["broker"].fillna("").astype(str)
    pos = pos[~broker.str.contains(r"\bTest\b", regex=True)]
    if pos.empty:
        return {}
    resolver: dict[tuple, set] = {}
    for _, r in pos.iterrows():
        p = parse_jpm_option_desc(r.get("description"))
        if p is None or not (p.strike > 0):
            continue
        key = (p.opt_type, p.underlying, p.expiry)
        resolver.setdefault(key, set()).add(float(p.strike))
    return resolver


def _resolve_txn_to_option(
    row: pd.Series, resolver: dict[tuple, set],
) -> Optional[tuple]:
    """Best-effort resolve a BUY/SELL txn to (opt_type, ul, expiry, strike).

    Tries:
      1. _parse_full_option_txn (Fidelity BUY or JPM with inline strike)
      2. parse_jpm_buy_desc (no strike) + resolver lookup by
         (opt_type, ul, expiry)

    Returns None if unresolvable (ambiguous strike, no positions match,
    or unparseable description).
    """
    desc = row.get("description")
    if not isinstance(desc, str):
        return None
    full = _parse_full_option_txn(row)
    if full is not None and full.strike > 0:
        return (full.opt_type, full.underlying, full.expiry, float(full.strike))
    partial = parse_jpm_buy_desc(desc)
    if partial is None:
        return None
    key = (partial.opt_type, partial.underlying, partial.expiry)
    strikes = resolver.get(key)
    if strikes is None or len(strikes) != 1:
        return None
    return (partial.opt_type, partial.underlying, partial.expiry,
            next(iter(strikes)))


def reconstruct_lots(
    transactions: pd.DataFrame,
    positions: Optional[pd.DataFrame] = None,
) -> list[Lot]:
    """Walk all PUT BUY/SELL transactions in chronological order and group
    them into lot lives.

    A "lot" is a (account, opt_type, underlying, expiry, strike) key. The
    first BUY opens the lot. Additional BUYs add qty; SELLs reduce qty.
    The lot closes when net qty returns to 0 (or expiry passes).

    We do NOT split lots that re-open after a full close in the same key —
    in practice this is rare for the sleeve (different strikes/expiries for
    each leg of a roll). If it happens, the lot is treated as one continuous
    line; the cost basis is the average across all BUYs in the chain.

    Returns: list of ``Lot`` objects, one per unique key with at least one BUY.
    """
    if transactions is None or transactions.empty:
        return []

    txn = transactions[transactions.apply(_is_put_buy_or_sell, axis=1)].copy()
    if txn.empty:
        return []
    txn["settlement_date"] = pd.to_datetime(txn["settlement_date"])

    resolver = build_strike_resolver(positions) if positions is not None else {}
    resolved: list[tuple | None] = [
        _resolve_txn_to_option(r, resolver) for _, r in txn.iterrows()
    ]
    txn["_resolved"] = resolved
    txn = txn[txn["_resolved"].notna()].copy()
    txn = txn.sort_values("settlement_date").reset_index(drop=True)

    lots_by_key: dict[tuple, Lot] = {}
    for _, r in txn.iterrows():
        opt_type, underlying, expiry, strike = r["_resolved"]
        key = (opt_type, underlying, expiry, strike)
        qty_raw = float(r.get("quantity") or 0.0)
        if qty_raw == 0:
            continue
        amt = float(r.get("amount") or 0.0)
        tt = str(r.get("transaction_type") or "").lower()
        # Statement-format quantities for option SELL rows are already
        # negative. Normalize to abs() then re-sign by transaction_type so
        # we are robust to either convention (some brokers also emit
        # positive qty + transaction_type=sell).
        qty = abs(qty_raw)
        signed_qty = qty if tt == "buy" else -qty
        sd = pd.Timestamp(r["settlement_date"]).normalize()

        lot = lots_by_key.get(key)
        if lot is None:
            # Must open with a BUY. If we encounter a SELL with no prior
            # BUY for this key (data quirk), skip.
            if tt != "buy":
                continue
            premium_per_share = -amt / (qty * CONTRACT_MULT) if qty > 0 else float("nan")
            lot = Lot(
                opt_type=opt_type,
                underlying=underlying,
                expiry=expiry,
                strike=strike,
                open_date=sd,
                close_date=None,
                open_qty=qty,
                open_premium=premium_per_share,
                qty_changes=[(sd, signed_qty)],
            )
            lots_by_key[key] = lot
        else:
            lot.qty_changes.append((sd, signed_qty))

    # Sort qty_changes per lot; compute close_date when net qty first hits 0.
    out: list[Lot] = []
    for lot in lots_by_key.values():
        lot.qty_changes.sort(key=lambda t: t[0])
        running = 0.0
        close_dt: Optional[pd.Timestamp] = None
        for d, delta in lot.qty_changes:
            running += delta
            if running <= 1e-9 and close_dt is None:
                close_dt = d
            elif running > 1e-9 and close_dt is not None:
                # Re-opened after a close — clear close_date.
                close_dt = None
        lot.close_date = close_dt
        out.append(lot)
    return out


def _build_close_lookup(
    option_history: pd.DataFrame,
) -> dict[tuple, pd.DataFrame]:
    """Return {(opt_type, underlying, expiry_date, strike): DataFrame[date, close]}
    sorted ascending so the lookup can use searchsorted-style fallback.

    The keys match the natural (Lot key) so reprice can index directly.
    """
    if option_history is None or option_history.empty:
        return {}
    h = option_history.copy()
    h["date"] = pd.to_datetime(h["date"])
    if "expiry" in h.columns:
        h["expiry"] = pd.to_datetime(h["expiry"])
    out: dict[tuple, pd.DataFrame] = {}
    for (opt_type, underlying, expiry, strike), grp in h.groupby(
        ["opt_type", "underlying", "expiry", "strike"]
    ):
        # Coerce expiry to date for matching with Lot.expiry (a datetime.date).
        exp_d = pd.Timestamp(expiry).date()
        key = (opt_type, underlying, exp_d, float(strike))
        sub = grp[["date", "close"]].sort_values("date").reset_index(drop=True)
        out[key] = sub
    return out


def _lookup_close(
    close_df: pd.DataFrame, when: pd.Timestamp,
) -> tuple[float, int]:
    """Return (close, days_stale) on `when`.

    Forward-fills across weekends or zero-volume days from the most recent
    earlier close. ``days_stale`` is the integer day-delta from the matched
    bar to ``when`` (0 = exact-date match; 1 = used yesterday's close; 3 =
    used Friday's for Monday after a quiet weekend; etc.).

    Returns ``(NaN, -1)`` when ``when`` is before the contract's first bar.
    The sentinel -1 lets callers distinguish "no data" from "exact match".
    """
    s = close_df[close_df["date"] <= when]
    if s.empty:
        return float("nan"), -1
    matched = s.iloc[-1]
    days = (pd.Timestamp(when) - pd.Timestamp(matched["date"])).days
    return float(matched["close"]), int(days)


def reprice_lots_daily(
    lots: list[Lot],
    option_history: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    trading_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Build per-(lot, date) MV by looking up historical close from
    ``option_history`` (loaded via ``fetch_option_history.load_history_csv``).

    Returns DataFrame [date, opt_type, underlying, expiry, strike, qty,
    close, market_value, days_stale]. One row per (date, lot) where qty > 0.

    ``days_stale`` is the day-delta between ``date`` and the matched
    Polygon bar (0 = exact-date close; >0 = forward-filled across a
    weekend or zero-volume day). Downstream callers use it to flag
    sleeve_mv values that rest on stale close data.

    If ``trading_dates`` is provided, the date grid is restricted to those
    (typically SPY's trading days). Otherwise the grid is the union of all
    dates appearing in the option_history within [start_date, end_date].
    """
    empty_cols = [
        "date", "opt_type", "underlying", "expiry", "strike", "qty",
        "close", "market_value", "days_stale",
    ]
    if not lots or option_history is None or option_history.empty:
        return pd.DataFrame(columns=empty_cols)

    close_lookup = _build_close_lookup(option_history)
    if not close_lookup:
        return pd.DataFrame(columns=empty_cols)

    # Build the date grid.
    if trading_dates is not None:
        dates = pd.DatetimeIndex(trading_dates)
        dates = dates[(dates >= pd.Timestamp(start_date))
                      & (dates <= pd.Timestamp(end_date))]
        date_grid = list(dates)
    else:
        all_dates = pd.concat(
            [df["date"] for df in close_lookup.values()], ignore_index=True
        )
        all_dates = pd.to_datetime(all_dates).sort_values().unique()
        date_grid = [
            d for d in all_dates
            if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)
        ]

    rows: list[dict] = []
    for d in date_grid:
        for lot in lots:
            qty = lot.qty_at(d)
            if qty <= 0:
                continue
            key = (lot.opt_type, lot.underlying, lot.expiry, float(lot.strike))
            close_df = close_lookup.get(key)
            if close_df is None or close_df.empty:
                continue
            close, days_stale = _lookup_close(close_df, d)
            if not math.isfinite(close):
                continue
            mv = qty * CONTRACT_MULT * close
            rows.append({
                "date": d,
                "opt_type": lot.opt_type,
                "underlying": lot.underlying,
                "expiry": pd.Timestamp(lot.expiry),
                "strike": lot.strike,
                "qty": qty,
                "close": close,
                "market_value": mv,
                "days_stale": days_stale,
            })
    return pd.DataFrame(rows)


def build_daily_sleeve_mv(
    transactions: pd.DataFrame,
    option_history: pd.DataFrame,
    *,
    positions: Optional[pd.DataFrame] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    trading_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """End-to-end: reconstruct lots, look up daily closes, aggregate sleeve MV.

    Args:
        transactions: union of statement + interim transactions.
        option_history: DataFrame produced by ``fetch_option_history`` —
            columns include underlying, opt_type, expiry, strike, date, close.
        positions: optional. When supplied, used to recover strikes for JPM
            PDF-parsed BUY/SELL txns that omit the strike.
        start_date: lower bound for the returned series. Default = first
            sleeve BUY date.
        end_date: upper bound. Default = max date in option_history.
        trading_dates: optional DatetimeIndex of trading days to use as the
            date grid. When omitted, the union of dates in option_history is
            used (which is already trading-day-aligned).

    Returns DataFrame [date, sleeve_mv, n_open_lots, n_stale_lots,
    frac_stale_mv, max_days_stale]. The three staleness columns expose
    how much of the day's sleeve MV rests on forward-filled (non-fresh)
    closes — useful for flagging back-test outputs that lean on stale
    Polygon bars (illiquid contracts, weekends, zero-volume days).
    """
    empty_cols = [
        "date", "sleeve_mv", "n_open_lots",
        "n_stale_lots", "frac_stale_mv", "max_days_stale",
    ]
    lots = reconstruct_lots(transactions, positions=positions)
    if not lots:
        return pd.DataFrame(columns=empty_cols)
    if option_history is None or option_history.empty:
        return pd.DataFrame(columns=empty_cols)

    if start_date is None:
        start_date = min(lot.open_date for lot in lots)
    else:
        start_date = pd.Timestamp(start_date)
    if end_date is None:
        end_date = pd.to_datetime(option_history["date"]).max()
    else:
        end_date = pd.Timestamp(end_date)

    per_lot = reprice_lots_daily(
        lots, option_history,
        start_date=start_date, end_date=end_date,
        trading_dates=trading_dates,
    )
    if per_lot.empty:
        return pd.DataFrame(columns=empty_cols)

    agg = per_lot.groupby("date").agg(
        sleeve_mv=("market_value", "sum"),
        n_open_lots=("qty", "count"),
        max_days_stale=("days_stale", "max"),
    ).reset_index()
    # Per-day stale-MV share: sum of MV for rows with days_stale > 0,
    # divided by total sleeve MV that day.
    stale = per_lot[per_lot["days_stale"] > 0].groupby("date").agg(
        n_stale_lots=("qty", "count"),
        stale_mv=("market_value", "sum"),
    ).reset_index()
    agg = agg.merge(stale, on="date", how="left")
    agg["n_stale_lots"] = agg["n_stale_lots"].fillna(0).astype(int)
    agg["stale_mv"] = agg["stale_mv"].fillna(0.0)
    agg["frac_stale_mv"] = (
        agg["stale_mv"] / agg["sleeve_mv"].replace(0, float("nan"))
    ).fillna(0.0)
    agg["max_days_stale"] = agg["max_days_stale"].astype(int)
    agg = agg.drop(columns=["stale_mv"])
    return agg.sort_values("date").reset_index(drop=True)


def find_drawdown_episodes(
    spy_history: pd.DataFrame,
    *,
    threshold_pct: float = DEFAULT_DRAWDOWN_THRESHOLD_PCT,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Identify SPY peak→trough→recover episodes with decline ≥ threshold_pct.

    Episode definition:
    * Peak: a running maximum of SPY close.
    * Trough: the lowest close while continuously below the peak.
    * Recover: first day SPY close >= peak (closes the episode).

    Overlapping nested peaks are collapsed: once a deeper trough appears
    relative to the same standing peak, the episode's trough is updated.
    A new episode only starts after recovery (or end-of-data).

    Args:
        spy_history: [date, close] daily SPY (use total-return series for the
            "cleanest" benchmark, or price series — both work).
        threshold_pct: minimum peak-to-trough decline to include (positive
            number, e.g. 3.0 for ≥ 3% drop).
        start_date, end_date: optional window.

    Returns DataFrame [peak_date, peak_close, trough_date, trough_close,
    decline_pct, recover_date, recovered, duration_days].
    """
    spy = spy_history.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").reset_index(drop=True)
    if start_date is not None:
        spy = spy[spy["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        spy = spy[spy["date"] <= pd.Timestamp(end_date)]
    if spy.empty:
        return pd.DataFrame(columns=[
            "peak_date", "peak_close", "trough_date", "trough_close",
            "decline_pct", "recover_date", "recovered", "duration_days",
        ])
    spy = spy.reset_index(drop=True)

    episodes: list[dict] = []
    peak_close = float(spy.iloc[0]["close"])
    peak_date = spy.iloc[0]["date"]
    in_ep = False
    cur: dict = {}

    for _, row in spy.iloc[1:].iterrows():
        c = float(row["close"])
        d = row["date"]
        if c >= peak_close:
            if in_ep:
                # Close out current episode if it crossed threshold.
                if cur["decline_pct"] <= -threshold_pct:
                    cur["recover_date"] = d
                    cur["recovered"] = True
                    cur["duration_days"] = (d - cur["peak_date"]).days
                    episodes.append(cur)
                in_ep = False
                cur = {}
            peak_close = c
            peak_date = d
        else:
            # Below peak.
            if not in_ep:
                cur = {
                    "peak_date": peak_date,
                    "peak_close": peak_close,
                    "trough_date": d,
                    "trough_close": c,
                    "decline_pct": (c / peak_close - 1) * 100,
                }
                in_ep = True
            else:
                if c < cur["trough_close"]:
                    cur["trough_close"] = c
                    cur["trough_date"] = d
                    cur["decline_pct"] = (c / peak_close - 1) * 100

    # Ongoing (unrecovered) episode at end of data.
    if in_ep and cur["decline_pct"] <= -threshold_pct:
        cur["recover_date"] = None
        cur["recovered"] = False
        cur["duration_days"] = (spy.iloc[-1]["date"] - cur["peak_date"]).days
        episodes.append(cur)

    if not episodes:
        return pd.DataFrame(columns=[
            "peak_date", "peak_close", "trough_date", "trough_close",
            "decline_pct", "recover_date", "recovered", "duration_days",
        ])
    return pd.DataFrame(episodes)


def attach_sleeve_to_episodes(
    episodes: pd.DataFrame,
    daily_sleeve: pd.DataFrame,
) -> pd.DataFrame:
    """For each episode, look up sleeve_mv at peak / trough / recover dates
    and compute the sleeve's $-payoff over the drawdown.

    Adds columns:
        sleeve_mv_peak, sleeve_mv_trough, sleeve_mv_recover,
        sleeve_gain_peak_to_trough, sleeve_gain_pct,
        peak_days_stale, trough_days_stale, recover_days_stale

    The ``*_days_stale`` columns report the max staleness (in days) of the
    underlying Polygon close that drove the sleeve_mv on that date — 0
    means the contract had a fresh close, >0 means we forward-filled.
    None when the date itself has no sleeve row (pre-sleeve / missing).
    Dashboard layer flags episodes whose trough lookup is stale.

    The sleeve_gain values are raw dollar deltas; the dashboard layer is
    responsible for computing any hedge-ratio normalization (e.g. against
    portfolio NAV or SPY-notional protected), because the "right"
    denominator depends on what question the user is asking and may
    include positions not in the back-test (other underlyings, beta
    adjustments, etc).

    Missing-data behavior: if a date has no sleeve row (pre-sleeve era,
    or weekend that doesn't match the SPY date), the corresponding
    sleeve_mv field is NaN and downstream gains return NaN.
    """
    if episodes is None or episodes.empty:
        return episodes
    if daily_sleeve is None or daily_sleeve.empty:
        out = episodes.copy()
        out["sleeve_mv_peak"] = float("nan")
        out["sleeve_mv_trough"] = float("nan")
        out["sleeve_mv_recover"] = float("nan")
        out["sleeve_gain_peak_to_trough"] = float("nan")
        out["sleeve_gain_pct"] = float("nan")
        out["peak_days_stale"] = None
        out["trough_days_stale"] = None
        out["recover_days_stale"] = None
        return out

    ds = daily_sleeve.copy()
    ds["date"] = pd.to_datetime(ds["date"])
    ds = ds.sort_values("date").reset_index(drop=True)
    has_stale_col = "max_days_stale" in ds.columns

    def lookup_row(when) -> tuple[float, Optional[int]]:
        """Returns (sleeve_mv, max_days_stale_on_matched_row)."""
        if when is None or pd.isna(when):
            return float("nan"), None
        ts = pd.Timestamp(when)
        exact = ds[ds["date"] == ts]
        if not exact.empty:
            r = exact.iloc[0]
        else:
            s = ds[ds["date"] <= ts]
            if s.empty:
                return float("nan"), None
            r = s.iloc[-1]
        mv = float(r["sleeve_mv"])
        stale = int(r["max_days_stale"]) if has_stale_col else None
        return mv, stale

    out = episodes.copy().reset_index(drop=True)
    peak_pairs = [lookup_row(r["peak_date"]) for _, r in out.iterrows()]
    trough_pairs = [lookup_row(r["trough_date"]) for _, r in out.iterrows()]
    recover_pairs = [lookup_row(r["recover_date"]) for _, r in out.iterrows()]
    out["sleeve_mv_peak"] = [p[0] for p in peak_pairs]
    out["sleeve_mv_trough"] = [p[0] for p in trough_pairs]
    out["sleeve_mv_recover"] = [p[0] for p in recover_pairs]
    out["peak_days_stale"] = [p[1] for p in peak_pairs]
    out["trough_days_stale"] = [p[1] for p in trough_pairs]
    out["recover_days_stale"] = [p[1] for p in recover_pairs]
    out["sleeve_gain_peak_to_trough"] = (
        out["sleeve_mv_trough"] - out["sleeve_mv_peak"]
    )
    out["sleeve_gain_pct"] = (
        out["sleeve_gain_peak_to_trough"]
        / out["sleeve_mv_peak"].replace(0, float("nan"))
        * 100
    )
    return out


def compare_to_statement_mv(
    daily_sleeve: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-validate: on each statement date, compare daily-repriced sleeve
    MV to the actual statement MV (sum of option_put rows).

    Residual error sources are (a) broker mid-vs-close marking conventions
    (JPM/Fidelity may book MV off the bid or an earlier snapshot than
    Polygon's official close) and (b) low-volume days where the Polygon
    close is stale. Large divergences (>30%) on a fresh-close day indicate
    a strike/expiry mismatch or a broker that revalued mid-month — flag
    in the UI but don't fail.

    Returns DataFrame [date, statement_mv, repriced_mv, error_pct].
    """
    if positions is None or positions.empty or daily_sleeve.empty:
        return pd.DataFrame(columns=[
            "date", "statement_mv", "repriced_mv", "error_pct",
        ])
    pos = positions.copy()
    pos["statement_date"] = pd.to_datetime(pos["statement_date"])
    puts = pos[pos["asset_class"].fillna("").str.startswith("option_put")]
    if puts.empty:
        return pd.DataFrame(columns=[
            "date", "statement_mv", "repriced_mv", "error_pct",
        ])
    stmt_mv = (puts.groupby("statement_date")["market_value"].sum()
                   .reset_index()
                   .rename(columns={"statement_date": "date",
                                     "market_value": "statement_mv"}))
    ds = daily_sleeve.copy()
    ds["date"] = pd.to_datetime(ds["date"])

    rows: list[dict] = []
    for _, r in stmt_mv.iterrows():
        d = r["date"]
        # Most-recent-earlier-or-equal match (statement could be a weekend).
        m = ds[ds["date"] <= d]
        if m.empty:
            continue
        rep_mv = float(m.iloc[-1]["sleeve_mv"])
        stmt = float(r["statement_mv"])
        err = (rep_mv / stmt - 1) * 100 if stmt > 0 else float("nan")
        rows.append({
            "date": d, "statement_mv": stmt,
            "repriced_mv": rep_mv, "error_pct": err,
        })
    return pd.DataFrame(rows)


def find_coverage_gaps(
    lots: list[Lot],
    option_history: pd.DataFrame,
) -> list[dict]:
    """Identify lots whose Polygon history doesn't cover their full life.

    Two failure modes the back-test silently swallows:

    * ``kind="no_history"`` — the contract isn't in option_history at all.
      reprice_lots_daily emits no rows for it, so it contributes 0 to
      sleeve_mv for its entire lifetime. Usually a fetch failure or an
      illiquid OCC ticker Polygon doesn't carry.
    * ``kind="pre_history"`` — the contract IS in option_history, but the
      earliest bar is later than the lot's open_date. The lot contributes
      0 from open_date through (first_bar_date − 1). Typically caused by
      Options Starter's ≤2y history limit when a lot opened before the
      window.

    Both bias the sleeve_gain downward on episodes that straddle the gap
    — the dashboard surfaces them as a coverage-gap banner so the user
    knows the back-test isn't telling the full story.

    Returns a list of dicts:
        {kind, opt_type, underlying, expiry, strike, open_date,
         first_bar_date, gap_days}
    ``first_bar_date`` is None and ``gap_days`` is None for no_history.
    Empty list when every lot has full coverage.
    """
    if not lots or option_history is None or option_history.empty:
        return []
    close_lookup = _build_close_lookup(option_history)
    gaps: list[dict] = []
    for lot in lots:
        key = (lot.opt_type, lot.underlying, lot.expiry, float(lot.strike))
        df = close_lookup.get(key)
        if df is None or df.empty:
            gaps.append({
                "kind": "no_history",
                "opt_type": lot.opt_type,
                "underlying": lot.underlying,
                "expiry": lot.expiry,
                "strike": lot.strike,
                "open_date": lot.open_date,
                "first_bar_date": None,
                "gap_days": None,
            })
            continue
        first_bar = pd.Timestamp(df.iloc[0]["date"])
        if first_bar > lot.open_date:
            gaps.append({
                "kind": "pre_history",
                "opt_type": lot.opt_type,
                "underlying": lot.underlying,
                "expiry": lot.expiry,
                "strike": lot.strike,
                "open_date": lot.open_date,
                "first_bar_date": first_bar,
                "gap_days": (first_bar - lot.open_date).days,
            })
    return gaps
