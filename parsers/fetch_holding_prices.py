"""
Fetch latest daily-close prices from Polygon for every fetchable symbol in the
most recent positions snapshot. Output feeds the mark-to-market step in app.py.

DESIGN: idempotent and transient.
  - Reads `data/positions.csv` (latest statement_date snapshot) + optionally
    `data/transactions_interim.csv` (to capture brand-new symbols bought
    between statements before the synthesizer has run).
  - Classifies each symbol by asset_class:
      equity / etf / mutual_fund / gold / other  -> fetch from Polygon
      cash                                       -> close = $1, status = cash_fixed_1
      option_*                                   -> status = option_no_coverage
      fixed_income                               -> status = bond_no_coverage
      (no symbol)                                -> status = unknown_no_symbol
  - Polygon endpoint: /v2/aggs/ticker/{sym}/range/1/day/{start}/{end} with a
    7-day window so weekends / US holidays still resolve to the most recent
    trading bar. Takes the last bar in the response.
  - Output: `data/prices_latest.csv` with columns
      symbol, as_of_date, close, source, status
    Non-fetchable rows are INCLUDED with close=NaN so the mark-to-market join
    is symmetric and the dashboard can surface "stale price" cues per symbol.

Run modes:
  py parsers/fetch_holding_prices.py            # dry-run, prints samples
  py parsers/fetch_holding_prices.py --write    # emit data/prices_latest.csv
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from _config import get_massive_key, get_massive_base

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POSITIONS_CSV   = DATA / "positions.csv"
INTERIM_TXN_CSV = DATA / "transactions_interim.csv"
OUT_CSV         = DATA / "prices_latest.csv"

LOOKBACK_DAYS = 7  # window covers weekends + US holidays

# Statement tickers that Polygon serves under a different symbol. Class-B
# shares are the common case (statements drop the punctuation).
TICKER_ALIASES = {
    "BRKB": "BRK.B",
    "BFB":  "BF.B",
}


# ---------- symbol collection --------------------------------------------

def _norm_symbol(sym: object) -> str | None:
    if isinstance(sym, str) and sym.strip():
        return sym.strip()
    return None


def _classify(asset_class: str, symbol: str) -> str:
    """Bucket a symboled row. Null-symbol rows are filtered earlier."""
    ac = (asset_class or "").lower()
    if ac == "cash":
        return "cash"
    if ac.startswith("option"):
        return "option"
    # fixed_income WITH a ticker (e.g. SGOV) is a bond ETF — fetchable.
    # Bare-CUSIP Treasuries arrive with symbol=None and are already filtered.
    return "equity"


def collect_symbols() -> pd.DataFrame:
    """Return a deduped DataFrame of (symbol, status_hint) for the latest snapshot.

    Pulls from positions.csv at its max statement_date, plus interim
    transactions if present (brand-new symbols not yet in positions). When the
    same symbol shows up with multiple hints, equity wins (we want to TRY to
    price it; mark-to-market will handle a missing price gracefully).
    """
    positions = pd.read_csv(POSITIONS_CSV, parse_dates=["statement_date"])
    latest_date = positions["statement_date"].max()
    latest = positions[positions["statement_date"] == latest_date].copy()

    rows: list[dict] = []
    for _, r in latest.iterrows():
        sym = _norm_symbol(r.get("symbol"))
        if sym is None:
            # CUSIP-only bonds and other no-symbol rows — leave to the
            # mark-to-market join's "no price match" fallback.
            continue
        hint = _classify(r.get("asset_class"), sym)
        rows.append({"symbol": sym, "status_hint": hint})

    if INTERIM_TXN_CSV.exists():
        interim = pd.read_csv(INTERIM_TXN_CSV)
        for _, r in interim.iterrows():
            sym = _norm_symbol(r.get("symbol"))
            if sym is None:
                continue
            cusip = r.get("cusip")
            if isinstance(cusip, str) and cusip.strip():
                continue  # bond — leave it to statement-price fallback
            rows.append({"symbol": sym, "status_hint": "equity"})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Dedup priority: cash wins (deposit-sweep tickers like QCPCM appear in
    # positions.csv under multiple asset_class rows — one cash row, one TLH
    # holding row — and we know the price is $1, no need to ask Polygon and
    # come back fetch_empty). Equity is next so anything classified as a real
    # holding still goes through the fetch path.
    hint_priority = {"cash": 0, "equity": 1, "option": 2, "bond": 3, "unknown": 4}
    df["_p"] = df["status_hint"].map(hint_priority).fillna(99)
    df = (df.sort_values("_p")
            .drop_duplicates(subset=["symbol"], keep="first")
            .drop(columns=["_p"])
            .sort_values("symbol")
            .reset_index(drop=True))
    return df


# ---------- Polygon fetch ------------------------------------------------

def fetch_latest_close(ticker: str, end: date, key: str, base: str) -> dict | None:
    """Return {'close': float, 'as_of_date': 'YYYY-MM-DD'} or {'error': str}
    or None when the API succeeded but returned no bars."""
    start = end - timedelta(days=LOOKBACK_DAYS)
    url = f"{base}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    try:
        r = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key},
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}
    payload = r.json()
    bars = payload.get("results") or []
    if not bars:
        return None
    last = bars[-1]
    bar_date = (pd.to_datetime(last["t"], unit="ms")
                  .tz_localize("UTC")
                  .tz_convert("America/New_York")
                  .date())
    return {"close": float(last["c"]), "as_of_date": bar_date.isoformat()}


# ---------- driver -------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Write data/prices_latest.csv (default: dry-run)")
    args = ap.parse_args()

    if not POSITIONS_CSV.exists():
        print(f"[!] {POSITIONS_CSV} not found")
        return 1

    syms = collect_symbols()
    if syms.empty:
        print("[!] No symbols found in latest snapshot.")
        return 1

    counts = syms["status_hint"].value_counts().to_dict()
    print(f"Latest-snapshot symbols: {len(syms)} unique  -> "
          f"equity={counts.get('equity', 0)}  "
          f"cash={counts.get('cash', 0)}  "
          f"option={counts.get('option', 0)}  "
          f"bond={counts.get('bond', 0)}  "
          f"unknown={counts.get('unknown', 0)}")
    print()

    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()
    end = date.today()

    out_rows: list[dict] = []
    fetched_ok = 0
    fetched_fail = 0
    print(f"{'Symbol':10s} {'Status':22s} {'As of':12s} {'Close':>14s}")
    print("-" * 62)
    for _, r in syms.iterrows():
        sym = r["symbol"]
        hint = r["status_hint"]

        if hint == "cash":
            row = {"symbol": sym, "as_of_date": end.isoformat(),
                   "close": 1.0, "source": "fixed", "status": "cash_fixed_1"}
        elif hint == "option":
            row = {"symbol": sym, "as_of_date": None, "close": None,
                   "source": "polygon", "status": "option_no_coverage"}
        elif hint == "bond":
            row = {"symbol": sym, "as_of_date": None, "close": None,
                   "source": "polygon", "status": "bond_no_coverage"}
        elif hint == "unknown":
            row = {"symbol": sym, "as_of_date": None, "close": None,
                   "source": "polygon", "status": "unknown_no_symbol"}
        else:  # equity
            poly_sym = TICKER_ALIASES.get(sym, sym)
            res = fetch_latest_close(poly_sym, end, key, base)
            if res is None:
                row = {"symbol": sym, "as_of_date": None, "close": None,
                       "source": "polygon", "status": "fetch_empty"}
                fetched_fail += 1
            elif "error" in res:
                row = {"symbol": sym, "as_of_date": None, "close": None,
                       "source": "polygon",
                       "status": f"fetch_error:{res['error'][:50]}"}
                fetched_fail += 1
            else:
                row = {"symbol": sym, "as_of_date": res["as_of_date"],
                       "close": res["close"], "source": "polygon",
                       "status": "ok"}
                fetched_ok += 1

        out_rows.append(row)
        close_s = (f"${row['close']:>13,.2f}" if row["close"] is not None
                   else f"{'—':>14s}")
        date_s = row["as_of_date"] or "—"
        print(f"{sym:10s} {row['status'][:22]:22s} {date_s:12s} {close_s}")

    out = pd.DataFrame(out_rows).sort_values("symbol").reset_index(drop=True)
    print()
    print(f"Fetched OK: {fetched_ok}   Fetch failures: {fetched_fail}   "
          f"Non-fetchable (cash/option/bond/unknown): "
          f"{len(out) - fetched_ok - fetched_fail}")

    if args.write:
        out.to_csv(OUT_CSV, index=False)
        print(f"\nWrote {OUT_CSV} ({len(out)} rows)")
    else:
        print(f"\n(dry-run — re-run with --write to emit {OUT_CSV})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
