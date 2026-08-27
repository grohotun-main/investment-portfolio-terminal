"""
Fetch long-history daily closes for the Big-3 macro tickers used by the
"Major-holding correlations" section of the Risk Contribution tab.

DESIGN: a small companion to fetch_daily_prices.py. Same Polygon endpoint
and same split-adjusted close convention, but a much deeper lookback than
the 3y window the Risk tabs use. Output is read directly by app.py and
spliced (BIL → SGOV pre-2020) before the correlation math runs.

  - Tickers: SPY, SGOV, GLD, BIL
    BIL (1-3 month T-Bill ETF, launched 2007) is the SGOV proxy for the
    pre-May-2020 period — SGOV itself only started trading then, but
    correlation analysis benefits from a longer history.
  - Lookback: 10 years by default (Polygon Stock Developer ceiling).
  - Output: data/long_history_prices.csv with columns symbol, date, close.

Run:
  py parsers/fetch_long_history.py              # dry-run preview
  py parsers/fetch_long_history.py --write      # fetch + emit CSV
  py parsers/fetch_long_history.py --years 5    # shorter window
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from _config import get_massive_key, get_massive_base
from fetch_daily_prices import fetch_daily_history, TICKER_ALIASES
from incremental_fetch import refresh_csv  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_CSV = DATA / "long_history_prices.csv"

DEFAULT_LOOKBACK_YEARS = 10
LOOKBACK_BUFFER_DAYS = 30

# Big-3 + BIL proxy. Keep the list tiny since this file is only consumed by
# the major-holding correlation matrix; per-holding correlations on the
# Top-15 still read from daily_prices.csv (3y window).
TICKERS = ["SPY", "SGOV", "GLD", "BIL"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Fetch and emit CSV (default: dry-run summary).")
    ap.add_argument("--years", type=int, default=DEFAULT_LOOKBACK_YEARS,
                    help=f"Lookback in years (default: {DEFAULT_LOOKBACK_YEARS}).")
    ap.add_argument("--full", action="store_true",
                    help="Ignore on-disk data and re-pull full history.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent per-ticker fetches (default 8). Use 1 for "
                         "fully sequential.")
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=args.years * 365 + LOOKBACK_BUFFER_DAYS)
    print(f"Tickers: {TICKERS}")
    print(f"Window: {start} to {end} ({args.years}y + {LOOKBACK_BUFFER_DAYS}d buffer)")
    print()

    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()

    def _fetch_fn(sym, fstart, fend):
        poly_sym = TICKER_ALIASES.get(sym, sym)
        res = fetch_daily_history(poly_sym, fstart, fend, key, base)
        if "error" in res or not res.get("bars"):
            return pd.DataFrame(columns=["date", "close"])
        return pd.DataFrame(res["bars"], columns=["date", "close"])

    if not args.write:
        # Dry-run: sample one ticker so we don't full-fetch on a preview.
        sample = TICKERS[0]
        preview = _fetch_fn(sample, start, end)
        print(f"DRY-RUN sample {sample}: {len(preview)} bars "
              f"({preview['date'].min() if not preview.empty else '-'} -> "
              f"{preview['date'].max() if not preview.empty else '-'})")
        print(f"(re-run with --write to emit {OUT_CSV})")
        return 0

    print(f"Refreshing {TICKERS} (incremental"
          f"{' — FULL re-pull' if args.full else ''})...")
    summary = refresh_csv(
        OUT_CSV, TICKERS, _fetch_fn,
        lookback_start=start, today=end,
        empty_columns=["symbol", "date", "close"],
        group_col="symbol", full=args.full, max_workers=args.workers)
    print(f"Wrote {OUT_CSV} — "
          + ", ".join(f"{k}={v}" for k, v in summary.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
