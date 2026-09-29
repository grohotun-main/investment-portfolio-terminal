"""Synthetic SPY put grid fetcher for the hedge-exit simulator (Phase F).

The PR #94 back-test only knows about puts the portfolio has actually held
(9 contracts at the time of writing). The exit-rule simulator needs to
test rolling SPY put programs against a synthetic universe of contracts
the portfolio never owned — so we need real Polygon historical closes for
those hypothetical legs.

Strategy
--------
* For each (as_of_date, target_dte, target_moneyness_pct) request, hit
  ``/v3/reference/options/contracts`` to enumerate the SPY put chain that
  existed on ``as_of_date``.
* Pick the contract that minimizes a joint distance:
      |dte_actual − dte_target| + 10 × |moneyness_actual − moneyness_target|
  (Moneyness weighted 10× because being off by 5% on strike matters more
  for hedge effectiveness than being off by 5 days on expiry.)
* Fetch ``/v2/aggs/ticker/O:SPY...`` for each resolved contract's full
  daily history.
* Cache in ``data/option_grid_history.csv`` (gitignored — synthetic data
  isn't committed; user re-runs to rebuild on a fresh checkout).

Falls back to a hand-constructed 3rd-Friday × $5-strike grid if the
reference endpoint 403s on the user's tier.

Run modes
---------
  py parsers/fetch_synthetic_put_grid.py             # dry-run, plan only
  py parsers/fetch_synthetic_put_grid.py --write     # fetch + cache
  py parsers/fetch_synthetic_put_grid.py --write \\
      --start 2024-05-25 --end 2026-05-25 \\
      --dte 90 --moneyness 0.05 --rebalance weekly
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_base, get_massive_key  # noqa: E402
from fetch_option_history import (  # noqa: E402
    build_option_ticker,
    fetch_contract_aggregates,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT_CSV = DATA / "option_grid_history.csv"
SPY_HISTORY_CSV = DATA / "daily_prices.csv"  # checked at runtime; fallback if missing

# Moneyness-vs-DTE distance weights for contract picking. Strike error of
# 1% is treated as roughly equivalent to a 10-day expiry error.
MONEYNESS_WEIGHT = 10.0

# Strike-grid fallback (used when /v3/reference 403s). SPY listed at $5
# strike spacing for OTM puts pre-2020; $1 spacing near-ATM in modern era.
# $5 is a safe coarse grid that always exists.
FALLBACK_STRIKE_STEP = 5.0

# Calendar slot for monthly SPY expiries — 3rd Friday of each month.
# (SPY also has weeklies and EOMs; back-test uses monthlies for stability.)

CSV_COLS = [
    "contract_ticker", "underlying", "opt_type", "expiry", "strike",
    "date", "open", "high", "low", "close", "volume", "fetched_at",
]


@dataclass
class GridTarget:
    """One leg the simulator wants priced.

    Resolved to an OCC ticker at fetch time; the simulator can then read
    its daily history from the cache.
    """
    as_of: date              # day the leg is being opened
    target_dte: int          # desired days-to-expiry at open
    target_moneyness: float  # desired (spot - strike) / spot at open (positive for OTM puts)


def third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of ``year``-``month``."""
    first = date(year, month, 1)
    # weekday(): Mon=0 .. Fri=4. Offset to first Friday, then +14 days.
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def _list_third_fridays(start: date, end: date) -> list[date]:
    """Inclusive list of 3rd-Fridays in [start, end]."""
    out: list[date] = []
    y, m = start.year, start.month
    while True:
        d = third_friday(y, m)
        if d > end:
            break
        if d >= start:
            out.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fetch_spy_put_chain_as_of(
    as_of: date,
    dte_window: tuple[int, int],
    strike_window: tuple[float, float],
    key: str,
    base: str,
) -> tuple[list[dict], bool]:
    """Pull SPY puts from /v3/reference/options/contracts available on ``as_of``.

    Returns (contracts, gated). ``gated=True`` means the endpoint 403'd
    — caller should fall back to the brute-force grid.
    """
    url = f"{base}/v3/reference/options/contracts"
    exp_lo = as_of + timedelta(days=dte_window[0])
    exp_hi = as_of + timedelta(days=dte_window[1])
    params = {
        "underlying_ticker": "SPY",
        "contract_type": "put",
        "as_of": as_of.isoformat(),
        "expiration_date.gte": exp_lo.isoformat(),
        "expiration_date.lte": exp_hi.isoformat(),
        "strike_price.gte": strike_window[0],
        "strike_price.lte": strike_window[1],
        "limit": 1000,
        "apiKey": key,
    }
    rows: list[dict] = []
    while True:
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  [!] network: {e}")
            return rows, False
        if r.status_code == 403:
            return [], True  # tier gated; caller falls back
        if r.status_code != 200:
            print(f"  [!] {r.status_code}: {r.text[:200]}")
            return rows, False
        payload = r.json()
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        params = {"apiKey": key}
    return rows, False


def pick_best_contract(
    chain: list[dict], as_of: date, spot: float,
    target_dte: int, target_moneyness: float,
) -> Optional[dict]:
    """Joint-distance contract picker.

    Returns the chain entry minimizing
        |dte_actual − target_dte| + 10·|moneyness_actual − target_moneyness|
    Skips contracts with missing strike / expiration.
    """
    if not chain or spot <= 0:
        return None
    best, best_d = None, float("inf")
    for c in chain:
        exp_str = c.get("expiration_date")
        K = c.get("strike_price")
        if not exp_str or K is None:
            continue
        try:
            exp_d = date.fromisoformat(exp_str)
        except ValueError:
            continue
        dte = (exp_d - as_of).days
        if dte <= 0:
            continue
        moneyness = (spot - float(K)) / spot  # positive = OTM put
        d = abs(dte - target_dte) + MONEYNESS_WEIGHT * abs(moneyness - target_moneyness) * 100
        if d < best_d:
            best_d = d
            best = c
    return best


def pick_fallback_contract(
    as_of: date, spot: float, target_dte: int, target_moneyness: float,
) -> dict:
    """Brute-force pick when reference endpoint isn't available.

    Pick the 3rd-Friday expiry closest to as_of+target_dte, and round the
    target strike (spot × (1−target_moneyness)) to the nearest
    ``FALLBACK_STRIKE_STEP``.
    """
    target_exp = as_of + timedelta(days=target_dte)
    fridays = _list_third_fridays(target_exp - timedelta(days=45),
                                  target_exp + timedelta(days=45))
    exp = min(fridays, key=lambda d: abs((d - target_exp).days)) if fridays \
        else third_friday(target_exp.year, target_exp.month)
    K_target = spot * (1.0 - target_moneyness)
    K = round(K_target / FALLBACK_STRIKE_STEP) * FALLBACK_STRIKE_STEP
    return {
        "ticker": build_option_ticker("SPY", "put", exp, K),
        "underlying_ticker": "SPY",
        "contract_type": "put",
        "expiration_date": exp.isoformat(),
        "strike_price": K,
    }


def load_cache(path: Path = OUT_CSV) -> pd.DataFrame:
    """Load the synthetic-grid cache. Returns empty frame if missing."""
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLS)
    return pd.read_csv(path, parse_dates=["date", "expiry", "fetched_at"])


def _contract_in_cache(cache: pd.DataFrame, ticker: str) -> bool:
    if cache.empty:
        return False
    return bool((cache["contract_ticker"] == ticker).any())


def fetch_one_contract(
    ticker: str, underlying: str, opt_type: str, expiry: date, strike: float,
    *, key: str, base: str, padding_days: int = 7,
) -> pd.DataFrame:
    """Fetch the full daily history for one contract.

    The window starts at (expiry − 365 − padding) (longest plausible lookback
    for a 365-DTE leg) and ends at min(expiry + padding, today). Bigger
    upfront fetch but means a contract's data is grabbed in one shot and
    never re-fetched.
    """
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    today = date.today()
    from_d = expiry - timedelta(days=365 + padding_days)
    to_d = min(expiry + timedelta(days=padding_days), today)
    if to_d <= from_d:
        return pd.DataFrame(columns=CSV_COLS)
    print(f"  {ticker}: {from_d} → {to_d}", end="")
    bars = fetch_contract_aggregates(ticker, from_d, to_d, key, base)
    if bars.empty:
        print(" (no data)")
        return pd.DataFrame(columns=CSV_COLS)
    bars["contract_ticker"] = ticker
    bars["underlying"] = underlying
    bars["opt_type"] = opt_type
    bars["expiry"] = pd.Timestamp(expiry)
    bars["strike"] = float(strike)
    bars["fetched_at"] = fetched_at
    print(f" — {len(bars)} bars")
    return bars[CSV_COLS]


def load_spy_history() -> pd.DataFrame:
    """Load SPY daily closes from data/daily_prices.csv.

    Returns DataFrame [date, close]. Empty if the file or SPY rows are
    missing — caller decides whether to bail.
    """
    if not SPY_HISTORY_CSV.exists():
        return pd.DataFrame(columns=["date", "close"])
    df = pd.read_csv(SPY_HISTORY_CSV, parse_dates=["date"])
    # daily_prices.csv schema: [symbol, date, close]
    spy = df[df["symbol"] == "SPY"][["date", "close"]].sort_values("date")
    return spy.reset_index(drop=True)


def plan_targets(
    start: date, end: date, dte: int, moneyness: float,
    rebalance: str = "weekly",
) -> list[GridTarget]:
    """Generate the list of (as_of, dte, moneyness) targets the simulator
    will need.

    ``rebalance`` is the cadence at which a new leg is opened — for a
    DTE-roll exit rule at 30 DTE on a 90-DTE leg, this is roughly every
    60 days, but we over-fetch (weekly) to give the simulator flexibility.

    Cadence steps land on a weekday; for ``weekly`` we snap to Mondays so
    the cadence is stable across the window.
    """
    cadence = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30}
    step = cadence.get(rebalance, 7)
    # Snap start to next weekday for daily; next Monday for weekly/biweekly;
    # 1st-of-month for monthly. Keeps the cadence rhythm stable.
    if rebalance == "daily":
        while start.weekday() >= 5:
            start += timedelta(days=1)
    elif rebalance in ("weekly", "biweekly"):
        # Snap to next Monday.
        while start.weekday() != 0:
            start += timedelta(days=1)
    out: list[GridTarget] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(GridTarget(d, dte, moneyness))
        d += timedelta(days=step)
    return out


def resolve_and_fetch(
    targets: list[GridTarget], spy_history: pd.DataFrame,
    *, key: str, base: str, cache: pd.DataFrame,
    strike_band_pct: float = 0.20,
    dte_band_days: tuple[int, int] = (15, 60),
) -> tuple[pd.DataFrame, list[dict]]:
    """For each target, resolve to an OCC contract and fetch its history.

    Returns (updated_cache, plan) where plan is a list of dicts describing
    what was resolved (one per target) — used for the dry-run preview.
    """
    plan: list[dict] = []
    new_rows: list[pd.DataFrame] = []
    seen_tickers: set[str] = set(cache["contract_ticker"].unique()) if not cache.empty else set()
    use_fallback = False  # latched once we get a 403

    spy_idx = spy_history.set_index("date")["close"] if not spy_history.empty else None

    for t in targets:
        spot = None
        if spy_idx is not None:
            try:
                spot = float(spy_idx.asof(pd.Timestamp(t.as_of)))
            except (KeyError, ValueError):
                spot = None
        if spot is None or pd.isna(spot):
            plan.append({"as_of": t.as_of, "status": "no_spot"})
            continue

        if not use_fallback:
            chain, gated = fetch_spy_put_chain_as_of(
                t.as_of,
                (t.target_dte - dte_band_days[0], t.target_dte + dte_band_days[1]),
                (spot * (1 - strike_band_pct - t.target_moneyness),
                 spot * (1 + strike_band_pct - t.target_moneyness)),
                key, base,
            )
            if gated:
                print(f"  [info] /v3/reference returned 403 — switching to fallback strike grid")
                use_fallback = True
                pick = pick_fallback_contract(t.as_of, spot, t.target_dte, t.target_moneyness)
            else:
                best = pick_best_contract(chain, t.as_of, spot, t.target_dte, t.target_moneyness)
                if best is None:
                    plan.append({"as_of": t.as_of, "status": "no_match"})
                    continue
                pick = best
        else:
            pick = pick_fallback_contract(t.as_of, spot, t.target_dte, t.target_moneyness)

        ticker = pick["ticker"]
        K = float(pick["strike_price"])
        exp = date.fromisoformat(pick["expiration_date"])
        actual_dte = (exp - t.as_of).days
        actual_moneyness = (spot - K) / spot

        plan.append({
            "as_of": t.as_of, "spot": spot, "ticker": ticker,
            "expiry": exp, "strike": K,
            "dte": actual_dte, "moneyness": actual_moneyness,
            "status": "cached" if ticker in seen_tickers else "fetch",
        })
        if ticker in seen_tickers:
            continue
        bars = fetch_one_contract(
            ticker, "SPY", "put", exp, K, key=key, base=base,
        )
        if not bars.empty:
            new_rows.append(bars)
            seen_tickers.add(ticker)

    if new_rows:
        updated = pd.concat([cache] + new_rows, ignore_index=True)
        updated = updated.drop_duplicates(["contract_ticker", "date"])
        updated = updated.sort_values(["contract_ticker", "date"]).reset_index(drop=True)
    else:
        updated = cache
    return updated, plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--write", action="store_true",
                   help="Fetch contracts and update data/option_grid_history.csv "
                        "(default: dry-run, prints plan only)")
    p.add_argument("--start", type=str, default=None,
                   help="Back-test window start (ISO date). Default: 2y before today.")
    p.add_argument("--end", type=str, default=None,
                   help="Back-test window end (ISO date). Default: today.")
    p.add_argument("--dte", type=int, default=90,
                   help="Target DTE at leg open (default: 90).")
    p.add_argument("--moneyness", type=float, default=0.05,
                   help="Target moneyness as (spot−K)/spot, positive for OTM (default: 0.05 = 5%% OTM).")
    p.add_argument("--rebalance", type=str, default="weekly",
                   choices=["daily", "weekly", "biweekly", "monthly"],
                   help="Target generation cadence (default: weekly).")
    args = p.parse_args(argv)

    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()

    today = date.today()
    end = date.fromisoformat(args.end) if args.end else today
    start = date.fromisoformat(args.start) if args.start else (end - timedelta(days=730))

    spy = load_spy_history()
    if spy.empty:
        print(f"[!] No SPY history at {SPY_HISTORY_CSV}. Run fetch_daily_prices.py first.")
        return 1

    cache = load_cache()
    print(f"Window: {start} → {end}  |  DTE: {args.dte}d  |  Moneyness: {args.moneyness:.1%}")
    print(f"Cadence: {args.rebalance}  |  Cache rows: {len(cache)}  |  "
          f"Unique cached contracts: {cache['contract_ticker'].nunique() if not cache.empty else 0}")

    targets = plan_targets(start, end, args.dte, args.moneyness, args.rebalance)
    print(f"Targets to resolve: {len(targets)}")
    print()

    if not args.write:
        # Dry-run: resolve plan only (no fetches for new contracts).
        # We still hit the reference endpoint to show what would be picked,
        # but skip the per-contract /v2/aggs fetches.
        print("[dry-run] resolving sample of 5 targets to show planned contracts:")
        print("-" * 60)
        sample = targets[::max(1, len(targets) // 5)][:5]
        _, plan = resolve_and_fetch(
            sample, spy, key=key, base=base, cache=cache,
        )
        for row in plan:
            if row.get("status") == "no_spot":
                print(f"  {row['as_of']}: no SPY spot")
            elif row.get("status") == "no_match":
                print(f"  {row['as_of']}: no chain match")
            else:
                print(f"  {row['as_of']}  spot={row['spot']:.2f}  "
                      f"{row['ticker']}  K={row['strike']:.0f}  "
                      f"dte={row['dte']}d  moneyness={row['moneyness']:.1%}  "
                      f"[{row['status']}]")
        print()
        print(f"[dry-run] Pass --write to fetch full grid ({len(targets)} targets).")
        return 0

    updated, plan = resolve_and_fetch(
        targets, spy, key=key, base=base, cache=cache,
    )
    fetched = sum(1 for r in plan if r.get("status") == "fetch")
    cached = sum(1 for r in plan if r.get("status") == "cached")
    skipped = sum(1 for r in plan if r.get("status") in ("no_spot", "no_match"))
    print()
    print(f"Resolved {len(plan)} targets — fetched: {fetched}, cached: {cached}, skipped: {skipped}")
    print(f"Unique contracts in grid: {updated['contract_ticker'].nunique()}")
    DATA.mkdir(parents=True, exist_ok=True)
    updated.to_csv(OUT_CSV, index=False)
    print(f"[ok] wrote {len(updated)} rows → {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
