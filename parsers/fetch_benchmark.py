"""Bulk benchmark fetch — full daily history for one ticker.

Idempotent: overwrites data/benchmark_<ticker>.csv each run.

Run:
  py parsers\\fetch_benchmark.py --write                     # SPY, last 10y -> today
  py parsers\\fetch_benchmark.py --write VOO                 # different ticker
  py parsers\\fetch_benchmark.py --write SPY 2022-01-03      # custom start
  py parsers\\fetch_benchmark.py --write SPY 2022-01-03 2024-12-31  # custom range
  py parsers\\fetch_benchmark.py                             # smoke test, no CSV written
"""
import argparse
import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests

from _config import get_massive_key, get_massive_base
from incremental_fetch import refresh_csv  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_LOOKBACK_DAYS = 365 * 10 - 1  # Stock Developer tier cap; 1-day buffer so the boundary call succeeds


def fetch_daily(ticker: str, start: date, end: date) -> pd.DataFrame:
    key = get_massive_key()
    base = get_massive_base()
    url = f"{base}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    r = requests.get(
        url,
        params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    bars = payload.get("results") or []
    if not bars:
        raise RuntimeError(
            f"No bars returned. status={payload.get('status')} resultsCount={payload.get('resultsCount')}"
        )
    df = pd.DataFrame(bars)
    df["date"] = (
        pd.to_datetime(df["t"], unit="ms")
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    return df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "n_trades"}
    )[["date", "open", "high", "low", "close", "volume", "vwap", "n_trades"]]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write the CSV. Without this flag, runs as a smoke test only.")
    ap.add_argument("--full", action="store_true",
                    help="Ignore on-disk data and re-pull full history.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent per-ticker fetches (default 8). Benchmark is "
                         "a single ticker, so this is effectively always sequential.")
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("start", nargs="?", default=None,
                    help="ISO start date; defaults to MAX_LOOKBACK_DAYS ago")
    ap.add_argument("end", nargs="?", default=None,
                    help="ISO end date; defaults to today")
    args = ap.parse_args(argv[1:])

    ticker = args.ticker.upper()
    arg_start = date.fromisoformat(args.start) if args.start else None
    arg_end = date.fromisoformat(args.end) if args.end else None

    today = date.today()
    # End defaults to today (NOT today-1). A calendar-day buffer here skips
    # across a Monday holiday into the prior Friday's bar, desyncing
    # benchmark_spy.csv from daily_prices.csv and breaking the strict
    # cross-source date-alignment check in refresh_prices.py. Polygon's
    # aggregate endpoint silently omits an unpublished current-day bar
    # (matches behavior of the other three Polygon fetchers).
    end = arg_end or today
    earliest_allowed = today - timedelta(days=MAX_LOOKBACK_DAYS)
    start = arg_start or earliest_allowed
    if start < earliest_allowed:
        print(f"[WARN] requested start {start} predates Stock Developer-tier limit; clamping to {earliest_allowed}")
        start = earliest_allowed

    def _fetch_fn(_tkr, fstart, fend):
        try:
            return fetch_daily(ticker, fstart, fend)
        except RuntimeError:
            # fetch_daily raises when Polygon returns no bars (e.g. the overlap
            # range is a single already-stored day). Treat as "no new data".
            return pd.DataFrame(columns=["date", "open", "high", "low", "close",
                                         "volume", "vwap", "n_trades"])

    out_path = DATA_DIR / f"benchmark_{ticker.lower()}.csv"
    if not args.write:
        preview = _fetch_fn(ticker, start, end)
        print(f"[DRY] {ticker} {len(preview)} bars — would refresh {out_path} "
              f"(use --write)")
        return 0

    DATA_DIR.mkdir(exist_ok=True)
    print(f"[INFO] {ticker} incremental refresh -> {start}..{end}"
          f"{' (FULL)' if args.full else ''}")
    summary = refresh_csv(
        out_path, [ticker], _fetch_fn,
        lookback_start=start, today=end,
        empty_columns=["date", "open", "high", "low", "close",
                       "volume", "vwap", "n_trades"],
        group_col=None, full=args.full, max_workers=args.workers)
    print(f"[OK] wrote {out_path} — "
          + ", ".join(f"{k}={v}" for k, v in summary.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
