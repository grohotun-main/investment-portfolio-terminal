"""
Fetch CBOE spot VIX history for the regime-conditioned diversification module.

DESIGN: VIX (the CBOE volatility index, ^VIX on Yahoo / ticker `I:VIX` on
Polygon) is the right vol regime signal — *not* VIXY or any VIX-futures
ETF. VIXY tracks rolled VIX futures and is contango-decayed; z-scoring
normalizes the level but leaves the term-structure distortion in the
changes. ^VIX has no proxy issue and ~36 years of history.

CBOE publishes the daily history as a free, public CSV — no API key, no
SDK dependency. Endpoint:
    https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
Header: DATE,OPEN,HIGH,LOW,CLOSE  (MM/DD/YYYY dates)
Coverage: 1990-01-02 through prior-day close, daily refresh after market.

Output: data/vix_history.csv with columns date, close (long-form, single
symbol implied — keeps the schema parallel to long_history_prices.csv).

Run:
  py parsers/fetch_vix.py            # dry-run preview
  py parsers/fetch_vix.py --write    # fetch + emit CSV
"""
import argparse
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Routes Python's SSL trust through the Windows cert store so Norton's
# TLS-scanning re-signed certs validate. See parsers/_config.py.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_CSV = DATA / "vix_history.csv"

CBOE_VIX_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
# CBOE returns the bare file with no User-Agent — but some CDNs filter blank
# UAs as bot traffic. Send a generic browser UA to avoid sporadic 403s.
HEADERS = {"User-Agent": "Mozilla/5.0 portfolio_dashboard fetch_vix.py"}


def fetch_vix_history(url: str = CBOE_VIX_URL, timeout: int = 60) -> pd.DataFrame:
    """Download + parse the CBOE VIX history CSV.

    Returns a DataFrame with columns date (datetime64) and close (float),
    sorted ascending by date. Raises requests.RequestException on network
    failure or non-200 status.
    """
    r = requests.get(url, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    # CBOE schema: DATE,OPEN,HIGH,LOW,CLOSE
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return (df[["date", "close"]]
              .sort_values("date")
              .reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Fetch and emit CSV (default: dry-run summary).")
    args = ap.parse_args()

    print(f"Source: {CBOE_VIX_URL}")
    try:
        df = fetch_vix_history()
    except requests.RequestException as e:
        print(f"[!] Fetch failed: {e}")
        return 1

    n = len(df)
    print(f"Rows: {n:,}")
    print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Last close: {df['close'].iloc[-1]:.2f}")

    if not args.write:
        print()
        print(f"(dry-run — re-run with --write to emit {OUT_CSV})")
        return 0

    DATA.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")
    print()
    print(f"Wrote {OUT_CSV} ({n:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
