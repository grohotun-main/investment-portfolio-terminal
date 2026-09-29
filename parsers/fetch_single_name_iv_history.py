"""Back out a daily ATM implied-vol history for ONE underlying from Polygon
historical option closes (spec decision 11, 2026-07-06 hedge report).

Method, per underlying:
  1. Discover each month's listed puts via /v3/reference/options/contracts
     (expired=true); the month's STANDARD expiry = the one carrying the most
     strikes.
  2. For each trading day D in the supplied close series: target expiry =
     the standard expiry with DTE in [DTE_LO, DTE_HI], closest to
     DTE_TARGET; strike = listed strike nearest close(D) (second-nearest
     when that day's bar is missing).
  3. Fetch each needed contract's daily closes ONCE via /v2/aggs (memoized).
  4. Invert close -> IV via options_pricer.implied_vol (european — fast and
     self-consistent across the series; below-intrinsic prints -> NaN ->
     dropped).
  5. Walk selection dates NEWEST -> OLDEST with lazy month discovery; stop
     at a 403 before any row (entitlement wall), after EMPTY_MONTHS_STOP
     consecutive empty months (listings wall), or — once rows exist — after
     BARS_DEAD_DAYS_STOP consecutive selection days with no bar at either
     candidate strike (bars die years before listings do).

Cache: data/iv_history_<TICKER>.csv (incremental — reruns fetch only dates
after the cached max). First backfill takes MINUTES; per-month progress is
printed.

CLI (dry-run by default, mirrors the other fetchers):
  py parsers\\fetch_single_name_iv_history.py AMAT
  py parsers\\fetch_single_name_iv_history.py AMAT --write
(CLI loads spot closes from the dip sidecar via dip_adhoc and r from
data/risk_free_rate.csv; the hedge-report tool calls the library entry
points directly.)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from _config import get_massive_key, get_massive_base
from options_pricer import implied_vol

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DTE_LO, DTE_HI, DTE_TARGET = 20, 70, 35
EMPTY_MONTHS_STOP = 6      # consecutive empty months before declaring the wall
BARS_DEAD_DAYS_STOP = 42   # ~2 trading months of continuous no-prints = the bars entitlement wall
# "ok" needs BOTH density and span; keep equal to single_name_hedge's
# _IV_MIN_ROWS / _IV_MIN_SPAN_DAYS gate (a test pins the two together).
MIN_ROWS = 250             # ~1 trading year at realistic ~75% ATM-print fill
MIN_SPAN_DAYS = 540        # ~1.5 calendar years of coverage
IV_COLS = ["symbol", "date", "expiry", "strike", "option_close",
           "spot_close", "dte", "iv", "fetched_at"]


class EntitlementError(RuntimeError):
    """Polygon 403 — the plan tier lacks this endpoint or depth."""


def default_get_json(url: str, params: dict, timeout: int = 30) -> dict:
    """GET with 2 attempts + backoff (the fetch_risk_free_rate pattern).
    403 raises EntitlementError immediately — retrying it is pointless."""
    last: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 403:
                raise EntitlementError(
                    f"403 NOT_AUTHORIZED: {url.split('?')[0]}")
            resp.raise_for_status()
            return resp.json()
        except EntitlementError:
            raise
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"GET failed after retries: {last}")


def occ_put_ticker(underlying: str, expiry: date, strike: float) -> str:
    """OCC option symbol, e.g. O:AMAT270115P00095000."""
    return (f"O:{underlying.upper()}{expiry.strftime('%y%m%d')}"
            f"P{int(round(strike * 1000)):08d}")


def _month_bounds(y: int, m: int) -> tuple[date, date]:
    first = date(y, m, 1)
    last = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1))
    return first, last - pd.Timedelta(days=1).to_pytimedelta()


def discover_month(underlying: str, y: int, m: int, key: str, base: str,
                   get_json_fn) -> dict[date, list[float]]:
    """All listed put strikes for expiries inside month y-m (incl. expired).
    Returns {expiry: sorted strikes}; empty dict when the month has none."""
    first, last = _month_bounds(y, m)
    url = f"{base}/v3/reference/options/contracts"
    params: dict = {"underlying_ticker": underlying, "contract_type": "put",
                    "expired": "true",
                    "expiration_date.gte": first.isoformat(),
                    "expiration_date.lte": last.isoformat(),
                    "limit": 1000, "apiKey": key}
    exp_strikes: dict[date, set[float]] = {}
    while True:
        payload = get_json_fn(url, params)
        for c in payload.get("results") or []:
            try:
                e = date.fromisoformat(str(c.get("expiration_date"))[:10])
                k = float(c.get("strike_price"))
            except (TypeError, ValueError):
                continue
            exp_strikes.setdefault(e, set()).add(k)
        nxt = payload.get("next_url")
        if not nxt:
            break
        url, params = nxt, {"apiKey": key}
    return {e: sorted(s) for e, s in exp_strikes.items()}


def standard_expiry(exp_strikes: dict) -> tuple[date, list[float]] | None:
    """The month's standard expiry = the expiry carrying the most strikes
    (weeklies list far fewer). Ties break to the later date."""
    if not exp_strikes:
        return None
    e = max(exp_strikes, key=lambda d: (len(exp_strikes[d]), d))
    return e, list(exp_strikes[e])


def contract_closes(underlying: str, expiry: date, strike: float,
                    frm: date, to: date, key: str, base: str,
                    get_json_fn) -> pd.Series:
    """Daily closes for one contract over [frm, to] (one aggs call)."""
    occ = occ_put_ticker(underlying, expiry, strike)
    url = (f"{base}/v2/aggs/ticker/{occ}/range/1/day/"
           f"{frm.isoformat()}/{to.isoformat()}")
    payload = get_json_fn(url, {"adjusted": "true", "limit": 50000,
                                "apiKey": key})
    rows = payload.get("results") or []
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([pd.Timestamp(int(r["t"]), unit="ms").normalize()
                            for r in rows])
    return pd.Series([float(r["c"]) for r in rows], index=idx)


def _pick_expiry(d: date, expiries: list[date]) -> date | None:
    cands = [e for e in expiries if DTE_LO <= (e - d).days <= DTE_HI]
    if not cands:
        return None
    return min(cands, key=lambda e: (abs((e - d).days - DTE_TARGET), e))


def _rate_lookup(r_series: pd.Series):
    """O(log n) as-of lookup for the risk-free series (the naive per-date
    boolean mask materializes the whole index per call - ~700x slower on a
    decades-long DGS3MO series)."""
    r_ff = r_series.dropna().sort_index()
    if r_ff.empty:
        return lambda d: 0.04
    r_dates = r_ff.index.values
    r_vals = r_ff.to_numpy(dtype=float)

    def _r_at(d: date) -> float:
        pos = int(np.searchsorted(r_dates, np.datetime64(d), side="right")) - 1
        return r_vals[pos] if pos >= 0 else r_vals[0]

    return _r_at


def build_iv_history(underlying: str, closes: pd.Series,
                     r_series: pd.Series, q_yield: float, *, key: str,
                     base: str, get_json_fn=default_get_json,
                     start_after: date | None = None,
                     log=print) -> tuple[pd.DataFrame, str]:
    """Build IV rows for every close date (> start_after when given).

    Walks selection dates NEWEST -> OLDEST with lazy month discovery so the
    probe stops at the real wall instead of marching to the beginning of the
    close series: a 403 before any row (entitlement), EMPTY_MONTHS_STOP
    consecutive listing-less months (no options market), or — once at least
    one row exists — BARS_DEAD_DAYS_STOP consecutive selection days whose
    candidate strikes have no bar (bars die years before listings do).
    Returns (DataFrame[IV_COLS] sorted date-ascending, status) where status
    is "ok", or "unavailable" when a 403 hit before ANY row was produced
    (caller decides fallback)."""
    px = closes.dropna().sort_index()
    if start_after is not None:
        px = px[px.index.date > start_after]
    if px.empty:
        return pd.DataFrame(columns=IV_COLS), "ok"
    _r_at = _rate_lookup(r_series)

    monthly: dict[tuple[int, int], tuple[date, list[float]] | None] = {}
    empty_streak = 0
    got_403 = False
    bars_403 = False

    def _discover(yy: int, mm: int) -> tuple[date, list[float]] | None:
        """Standard expiry + strikes of month yy-mm, fetched once (memoized).
        Streaks move only on FRESH discoveries, never on memo hits."""
        nonlocal empty_streak, got_403
        key_m = (yy, mm)
        if key_m not in monthly:
            try:
                found = discover_month(underlying, yy, mm, key, base,
                                       get_json_fn)
            except EntitlementError:
                got_403 = True
                monthly[key_m] = None
                return None
            std = standard_expiry(found)
            monthly[key_m] = std
            if std is None:
                empty_streak += 1
            else:
                empty_streak = 0
                log(f"  {yy}-{mm:02d}: expiry {std[0]} with "
                    f"{len(std[1])} strikes")
        return monthly[key_m]

    bars_memo: dict[tuple[date, float], pd.Series] = {}

    def _bars(e: date, k: float, first_d: date) -> pd.Series:
        nonlocal bars_403
        keyt = (e, k)
        if keyt not in bars_memo:
            try:
                bars_memo[keyt] = contract_closes(
                    underlying, e, k,
                    first_d - pd.Timedelta(days=5).to_pytimedelta(), e,
                    key, base, get_json_fn)
            except EntitlementError:
                bars_403 = True
                bars_memo[keyt] = pd.Series(dtype=float)
        return bars_memo[keyt]

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    dead_streak = 0
    for ts, spot in list(px.items())[::-1]:          # NEWEST -> OLDEST
        if got_403 and not rows:
            break                                    # entitlement wall
        if empty_streak >= EMPTY_MONTHS_STOP:
            break                                    # listings (discovery) wall
        if rows and dead_streak >= BARS_DEAD_DAYS_STOP:
            break                                    # bars (prints) wall
        d = ts.date()
        # candidate months = every month the DTE window [D+LO, D+HI] touches
        lo_d = d + pd.Timedelta(days=DTE_LO).to_pytimedelta()
        hi_d = d + pd.Timedelta(days=DTE_HI).to_pytimedelta()
        expiries: list[date] = []
        strikes_of: dict[date, list[float]] = {}
        yy, mm = lo_d.year, lo_d.month
        while (yy, mm) <= (hi_d.year, hi_d.month):
            std = _discover(yy, mm)
            if std is not None:
                expiries.append(std[0])
                strikes_of[std[0]] = std[1]
            yy, mm = (yy + 1, 1) if mm == 12 else (yy, mm + 1)
        e = _pick_expiry(d, expiries)
        if e is None:
            continue
        ks = sorted(strikes_of[e], key=lambda k: (abs(k - spot), k))
        bar_seen = appended = False
        for k in ks[:2]:   # nearest, then second-nearest fallback
            bar = _bars(e, k, d)
            if ts not in bar.index:
                continue
            bar_seen = True
            oc = float(bar.loc[ts])
            if not oc > 0:
                continue
            t_yrs = (e - d).days / 365.0
            iv = implied_vol(oc, float(spot), float(k), t_yrs,
                             _r_at(d), q_yield, "put")
            if iv == iv:   # not NaN
                rows.append({"symbol": underlying.upper(), "date": d,
                             "expiry": e, "strike": float(k),
                             "option_close": oc, "spot_close": float(spot),
                             "dte": (e - d).days, "iv": float(iv),
                             "fetched_at": fetched_at})
                appended = True
            break          # a bar existed; below-intrinsic NaN is a drop, not a retry
        if appended:
            dead_streak = 0
        elif not bar_seen:
            dead_streak += 1   # only bar-MISSING days count toward the wall

    df = pd.DataFrame(rows, columns=IV_COLS)
    if len(df):
        # rows were appended newest-first; keep the CSV/cache contract ascending
        df = df.sort_values("date").reset_index(drop=True)
    if (got_403 or bars_403) and df.empty:
        return df, "unavailable"
    return df, "ok"


def iv_cache_path(data_dir: Path, underlying: str) -> Path:
    return Path(data_dir) / f"iv_history_{underlying.upper()}.csv"


def load_or_refresh_iv_history(data_dir, underlying: str, closes: pd.Series,
                               r_series: pd.Series, q_yield: float, *,
                               key: str | None = None,
                               base: str | None = None,
                               get_json_fn=default_get_json,
                               log=print) -> dict:
    """Cache -> incremental build -> upsert -> payload.

    Returns {"iv": date-indexed Series, "first_covered": date | None,
    "status": "ok" | "thin" | "unavailable"}. A refresh failure NEVER
    discards an existing cache (stale IV history is still rankable)."""
    key = key or get_massive_key()
    base = base or get_massive_base()
    path = iv_cache_path(data_dir, underlying)
    cached = (pd.read_csv(path, parse_dates=["date", "expiry"])
              if path.exists() else pd.DataFrame(columns=IV_COLS))
    start_after = (cached["date"].max().date()
                   if len(cached) else None)
    try:
        fresh, status = build_iv_history(
            underlying, closes, r_series, q_yield, key=key, base=base,
            get_json_fn=get_json_fn, start_after=start_after, log=log)
    except (RuntimeError, requests.RequestException) as exc:
        log(f"  [!] IV refresh failed ({exc}); serving cache as-is")
        fresh, status = pd.DataFrame(columns=IV_COLS), "ok"

    if status == "unavailable" and len(cached):
        log("[!] options-aggregates entitlement lost (403); serving cached "
            "IV history unchanged")
    if status == "unavailable" and not len(cached):
        return {"iv": pd.Series(dtype=float), "first_covered": None,
                "status": "unavailable"}

    both = pd.concat([cached, fresh], ignore_index=True)
    if len(both):
        both["date"] = pd.to_datetime(both["date"])
        both = (both.drop_duplicates(subset=["symbol", "date"], keep="last")
                    .sort_values("date"))
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        both.to_csv(path, index=False)
    iv = pd.Series(both["iv"].to_numpy(dtype=float),
                   index=pd.DatetimeIndex(both["date"]))
    if len(iv):
        span_days = (iv.index.max() - iv.index.min()).days
        st = ("ok" if (len(iv) >= MIN_ROWS and span_days >= MIN_SPAN_DAYS)
              else "thin")
    else:
        st = "unavailable"
    first = iv.index.min().date() if len(iv) else None
    return {"iv": iv, "first_covered": first, "status": st}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticker")
    ap.add_argument("--write", action="store_true",
                    help="build/refresh data/iv_history_<TICKER>.csv "
                         "(default: dry-run probe of the newest 3 months)")
    args = ap.parse_args()
    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()

    from dip_adhoc import resolve_adhoc  # noqa: E402 (lazy: CLI only)
    from fetch_dip_history import fetch_yahoo, fetch_dividends_yahoo  # noqa: E402
    today = date.today()
    hist = resolve_adhoc(DATA, args.ticker, today, fetch_yahoo,
                         fetch_dividends_yahoo, today, persist=True)
    if hist["status"] not in ("ok", "short"):
        print(f"[!] no price history for {args.ticker}: {hist['msg']}")
        return 1
    rf_path = DATA / "risk_free_rate.csv"
    r_series = (pd.read_csv(rf_path, parse_dates=["date"])
                .set_index("date")["rate_annual"]
                if rf_path.exists() else pd.Series(dtype=float))
    dser = hist["dser"]
    spot = float(hist["price"].iloc[-1])
    q_yield = (float(dser[dser.index >= dser.index.max()
                          - pd.Timedelta(days=365)].sum()) / spot
               if len(dser) else 0.0)
    closes = hist["price"] if args.write else hist["price"].iloc[-90:]
    print(f"IV backfill for {args.ticker.upper()} "
          f"({'full' if args.write else 'dry-run: last ~3 months'}) — "
          f"first runs take MINUTES; progress below.")
    res = load_or_refresh_iv_history(DATA, args.ticker, closes, r_series,
                                     q_yield, key=key, base=base)
    print(f"status={res['status']}  rows={len(res['iv'])}  "
          f"first_covered={res['first_covered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
