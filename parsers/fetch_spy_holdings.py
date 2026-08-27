"""Fetch SPDR S&P 500 ETF (SPY) holdings disclosure from SSGA.

Powers the "natural-weight" comparison in the Options Hedging Phase 2
recommender: a portfolio name's MCR % is "excess" when it materially
exceeds the share SPY itself would give that name.

Source
------
SSGA publishes a daily-refreshed XLSX at a stable URL. We cache it to
``data/spy_holdings.csv`` (2 columns: ticker, weight_pct). Refresh
quarterly minimum, monthly nice — see ``--write`` flag.

Run modes:
  py parsers/fetch_spy_holdings.py            # dry-run, prints sample
  py parsers/fetch_spy_holdings.py --write    # write data/spy_holdings.csv
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_base  # noqa: F401  (truststore side-effect)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT_CSV = DATA / "spy_holdings.csv"

SSGA_URL = (
    "https://www.ssga.com/us/en/individual/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)

# SSGA prepends 4 header rows of fund metadata before the column header.
SSGA_HEADER_SKIPROWS = 4

# Tickers that represent cash / FX legs, not equity holdings. Dropped.
_CASH_TICKERS = {"USD", "CASH_USD", "CASH", "-", "—"}


def normalize_holdings_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce the raw SSGA frame to (ticker, weight_pct) rows.

    Drops cash/FX legs. Uppercases tickers. Preserves SSGA's 0-100
    weight scale — does NOT renormalize, because SPY's reported weights
    sum to ~99.something (small cash residual) and the audit comparison
    is most defensible when the published number is preserved.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame({
        "ticker": df["Ticker"].astype(str).str.strip().str.upper(),
        "weight_pct": pd.to_numeric(df["Weight"], errors="coerce"),
    })
    out = out[~out["ticker"].isin(_CASH_TICKERS)]
    out = out[out["weight_pct"].notna()]
    return out.reset_index(drop=True)


def fetch_ssga_holdings_xlsx(url: str = SSGA_URL,
                              timeout: int = 30) -> pd.DataFrame:
    """Pull the SSGA holdings XLSX. Returns the raw frame (un-normalized)."""
    r = requests.get(url, timeout=timeout, headers={
        # SSGA refuses bare requests on some networks; mimic browser UA.
        "User-Agent": "Mozilla/5.0 (Phase2-Hedge-Recommender)",
    })
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content),
                          skiprows=SSGA_HEADER_SKIPROWS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write data/spy_holdings.csv (default: dry-run)")
    args = ap.parse_args()

    try:
        raw = fetch_ssga_holdings_xlsx()
    except requests.RequestException as e:
        print(f"[!] HTTP error fetching SSGA holdings: {e}")
        return 1
    except Exception as e:
        print(f"[!] Could not parse SSGA holdings: {e}")
        return 1

    out = normalize_holdings_frame(raw)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"Holdings fetched: {len(out)} names")
    print()
    print("Top 10 by weight:")
    print("-" * 40)
    print(out.sort_values("weight_pct", ascending=False).head(10).to_string(index=False))

    if args.write:
        out["fetched_at"] = fetched_at
        DATA.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        print(f"\nWrote {OUT_CSV} ({len(out)} rows)")
    else:
        print(f"\n[dry-run] would write {OUT_CSV}; use --write to persist.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
