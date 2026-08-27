"""Fetch daily OHLCV history for every option contract we have ever held.

Reads transactions (+ interim) and positions to identify every unique
(underlying, opt_type, expiry, strike) lot, then pulls daily aggregates
for each lot's lifetime via the ``/v2/aggs/ticker/{O:...}`` endpoint.
Writes ``data/option_history.csv`` with one row per (contract, date).

This unlocks the hedge-effectiveness back-test: with the close price per
day per contract, sleeve MV on date d is just ``Σ qty × 100 × close(d)``
across open lots, with no IV / pricer dependency. Audited market data,
not a model.

Endpoint: ``/v2/aggs/ticker/O:{TICKER}{YYMMDD}{P|C}{STRIKE*1000:08d}/range/1/day/...``
Empirically works on Options Starter ($29) — confirmed 2026-05-25 across
both currently-open and long-since-closed contracts.

Run:
  py parsers/fetch_option_history.py             # dry-run (no write)
  py parsers/fetch_option_history.py --write     # write CSV
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_base, get_massive_key  # noqa: E402
from hedge_effectiveness import (  # noqa: E402
    build_strike_resolver,
    reconstruct_lots,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT_CSV = DATA / "option_history.csv"

# How many days of padding around the lot's lifetime to fetch. Useful
# for back-test windows that anchor on dates slightly outside the lot's
# active life (e.g. the SPY peak/trough may sit one day outside the
# nearest lot's open_date / close_date).
LOT_PADDING_DAYS = 7

CSV_COLS = [
    "contract_ticker", "underlying", "opt_type", "expiry", "strike",
    "date", "open", "high", "low", "close", "volume", "fetched_at",
]


def build_option_ticker(
    underlying: str, opt_type: str, expiry: date, strike: float,
) -> str:
    """Polygon OCC-style options ticker.

    ``O:{UND}{YYMMDD}{P|C}{STRIKE*1000:08d}``
    Example: SPY $400 PUT expiring 01/15/27 -> ``O:SPY270115P00400000``
    """
    side = "P" if opt_type.lower().startswith("p") else "C"
    ymd = expiry.strftime("%y%m%d")
    strike_int = int(round(float(strike) * 1000))
    return f"O:{underlying.upper()}{ymd}{side}{strike_int:08d}"


def fetch_contract_aggregates(
    ticker: str, from_d: date, to_d: date, key: str, base: str,
) -> pd.DataFrame:
    """Pull /v2/aggs/ticker daily bars for one option contract.

    Returns DataFrame [date, open, high, low, close, volume]. Empty
    DataFrame on no-data / error. Errors are printed but don't raise —
    a single 404 (e.g. contract not on Polygon's tape) shouldn't kill
    the whole fetch.
    """
    url = f"{base}/v2/aggs/ticker/{ticker}/range/1/day/{from_d.isoformat()}/{to_d.isoformat()}"
    params = {"apiKey": key, "adjusted": "true", "sort": "asc",
              "limit": 50000}
    try:
        r = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"  {ticker}: network error: {e}")
        return pd.DataFrame()
    if r.status_code != 200:
        print(f"  {ticker}: HTTP {r.status_code}: {r.text[:200]}")
        return pd.DataFrame()
    data = r.json()
    results = data.get("results") or []
    if not results:
        print(f"  {ticker}: no bars in window")
        return pd.DataFrame()
    rows = []
    for bar in results:
        rows.append({
            "date": pd.Timestamp(bar["t"], unit="ms").normalize(),
            "open": bar.get("o"),
            "high": bar.get("h"),
            "low":  bar.get("l"),
            "close": bar.get("c"),
            "volume": bar.get("v"),
        })
    return pd.DataFrame(rows)


def fetch_history_for_lots(
    lots: list, key: str, base: str, *,
    extra_padding_today: bool = True,
) -> pd.DataFrame:
    """Fetch daily aggs for every lot's lifetime. Returns one
    DataFrame keyed by (contract_ticker, date)."""
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    today = date.today()
    out: list[pd.DataFrame] = []
    print(f"Fetching daily history for {len(lots)} unique contract(s)...")
    for lot in lots:
        ticker = build_option_ticker(
            lot.underlying, lot.opt_type, lot.expiry, lot.strike,
        )
        from_d = lot.open_date.date() - timedelta(days=LOT_PADDING_DAYS)
        # Open lots: pad up to today. Closed lots: only need data through
        # the close date (anything after is qty=0).
        if lot.close_date is None:
            base_end = today
        else:
            base_end = lot.close_date.date()
        to_d = base_end + timedelta(days=LOT_PADDING_DAYS)
        to_d = min(to_d, today)
        if to_d <= from_d:
            print(f"  {ticker}: empty date range, skipping")
            continue
        print(f"  {ticker}: {from_d} -> {to_d}", end="")
        bars = fetch_contract_aggregates(ticker, from_d, to_d, key, base)
        if bars.empty:
            print(" (no data)")
            continue
        bars["contract_ticker"] = ticker
        bars["underlying"] = lot.underlying
        bars["opt_type"] = lot.opt_type
        bars["expiry"] = pd.Timestamp(lot.expiry)
        bars["strike"] = float(lot.strike)
        bars["fetched_at"] = fetched_at
        print(f" — {len(bars)} bars")
        out.append(bars)

    if not out:
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.concat(out, ignore_index=True)
    return df[CSV_COLS].sort_values(
        ["contract_ticker", "date"]
    ).reset_index(drop=True)


def load_history_csv(path: Path = OUT_CSV) -> pd.DataFrame:
    """Load the cached history CSV. Returns empty frame if missing."""
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.read_csv(path, parse_dates=["date", "expiry", "fetched_at"])
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true",
                   help="Write data/option_history.csv (otherwise dry-run)")
    args = p.parse_args(argv)

    # Read txns + interim (caller's mirror of dashboard load_data flow).
    txn_path = DATA / "transactions.csv"
    int_path = DATA / "transactions_interim.csv"
    pos_path = DATA / "positions.csv"
    if not txn_path.exists():
        print(f"Missing {txn_path}", file=sys.stderr)
        return 1
    txn = pd.read_csv(txn_path)
    if int_path.exists():
        interim = pd.read_csv(int_path)
        txn = pd.concat([txn, interim], ignore_index=True)
    pos = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()

    # Reconstruct lots — same code path as the back-test consumes.
    lots = reconstruct_lots(txn, positions=pos)
    if not lots:
        print("No PUT lots found in transactions.", file=sys.stderr)
        return 0
    print(f"Found {len(lots)} unique contracts in sleeve history:")
    for lot in lots:
        cd = lot.close_date.date() if lot.close_date is not None else "open"
        print(f"  {lot.underlying} {lot.opt_type} K=${lot.strike:g} "
              f"exp={lot.expiry}  opened {lot.open_date.date()} -> {cd}")

    key = get_massive_key()
    base = get_massive_base()
    df = fetch_history_for_lots(lots, key, base)
    if df.empty:
        print("No history fetched.")
        return 1
    print(f"\nFetched {len(df)} total bars across "
          f"{df['contract_ticker'].nunique()} contracts.")
    if args.write:
        df.to_csv(OUT_CSV, index=False)
        print(f"Wrote {OUT_CSV}")
    else:
        print("Dry-run — not writing. Pass --write to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
