"""
Fetch daily-close history from Polygon for the equity-class symbols in the
latest positions snapshot, plus a fixed set of macro overlay tickers used by
the Risk tab (SPY, GLD, TLT, UUP, VIXY). Output feeds daily-resolution risk
metrics in app.py — beta, vol, VaR, position-level contribution.

DESIGN: idempotent, transient, three-column wide.
  - Reads `data/positions.csv` at its latest statement_date.
  - Includes everything that fetch_holding_prices.py classifies as "equity"
    (equity_etf, equity_stock, fixed_income with a ticker like SGOV).
  - Excludes cash sweep, options, bare-CUSIP bonds, and unknown rows.
  - **Excludes TLH-only tickers** — names that only ever appear in the
    direct-indexing sleeve. app.py folds the TLH sleeve to SPY before any
    risk math, so per-name TLH price history is wasted bandwidth. Tickers
    held in *both* TLH and elsewhere (e.g. AAPL, NVDA when also in the
    individual-stocks sleeve) are kept. See collect_tlh_only_symbols().
  - Always also fetches the macro overlay: SPY, GLD, TLT, UUP, VIXY.
    UUP proxies DXY (USD index); VIXY proxies ^VIX. Both are needed for
    Module 4 macro betas in Pass 4. Pulling them here means one fetch run
    covers Passes 2-4.
  - Polygon endpoint: /v2/aggs/ticker/{sym}/range/1/day/{start}/{end}
    with start = today - LOOKBACK_YEARS years - 30d buffer.
  - Output: `data/daily_prices.csv` with columns symbol, date, close.
    Adjusted closes (Polygon's `adjusted=true` adjusts for SPLITS ONLY —
    NOT dividends, despite some Polygon docs implying otherwise).
    Verified empirically: SPY column here equals benchmark_spy.csv close
    byte-for-byte; the dividend-reinvestment uplift only appears in
    benchmark_spy_tr.csv, which is built by build_benchmark_total_return.py.
    For total-return SPY metrics in the Risk tab, use bench_tr (loaded
    from benchmark_spy_tr.csv); price-only series here is fine for any
    per-symbol vol/beta math where the dividend uplift is small.

Run modes:
  py parsers/fetch_daily_prices.py            # sample 3 tickers, dry-run
  py parsers/fetch_daily_prices.py --write    # fetch all, emit CSV
  py parsers/fetch_daily_prices.py --years 5  # custom lookback
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from _config import get_massive_key, get_massive_base
from incremental_fetch import refresh_csv  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POSITIONS_CSV = DATA / "positions.csv"
OUT_CSV       = DATA / "daily_prices.csv"

DEFAULT_LOOKBACK_YEARS = 7
LOOKBACK_BUFFER_DAYS   = 30  # extra room for trailing-window calculations

# Macro overlay always fetched on top of held names.
#   SPY  — equity benchmark
#   GLD  — gold spot proxy
#   TLT  — long-duration treasury, proxy for 10Y rate move (inverse)
#   UUP  — dollar index proxy (DXY isn't directly tradable on Polygon)
#   VIXY — VIX futures ETF, proxy for ^VIX
#   SCHO, IEI, IEF — duration buckets used by the treasury-ladder fold so a
#       multi-rung ladder isn't squashed to SGOV (0.1y duration) and the Risk
#       tab actually reflects the ladder's rate exposure.
MACRO_OVERLAY = ["SPY", "GLD", "TLT", "UUP", "VIXY",
                 "SCHO", "IEI", "IEF"]

TICKER_ALIASES = {
    "BRKB": "BRK.B",
    "BFB":  "BF.B",
}


_EQUITY_CLASSES = ["equity_etf", "equity_stock",
                   "fixed_income", "gold", "mutual_fund"]


def collect_equity_symbols(latest_only: bool = False) -> list[str]:
    """Return sorted unique equity-class symbols ever held.

    When `latest_only` is True, restrict to the most recent statement_date
    (legacy behavior). When False (default), include every equity-class
    symbol that appears on any statement date — so positions held and then
    sold still get a fetched price history. Their price data is needed for
    any historical analytic that re-includes the holding period.
    """
    positions = pd.read_csv(POSITIONS_CSV, parse_dates=["statement_date"])
    if latest_only:
        latest_date = positions["statement_date"].max()
        positions = positions[positions["statement_date"] == latest_date]
    keep_mask = (
        positions["asset_class"].isin(_EQUITY_CLASSES)
        & positions["symbol"].notna()
        & positions["symbol"].astype(str).str.strip().ne("")
    )
    syms = (positions.loc[keep_mask, "symbol"]
                     .astype(str).str.strip().unique().tolist())
    return sorted(syms)


def collect_tlh_only_symbols() -> set[str]:
    """Return tickers ever held *only* in the TLH sleeve, never elsewhere.

    The dashboard's risk-synthesis layer (build_risk_series_bundle in
    app.py) folds the TLH sleeve to SPY before any DR / covariance /
    contribution math runs — the user does not pick those 300+ names
    individually; the whole sleeve is one direct-indexing decision with
    SPY-like behavior. Fetching per-name daily history for tickers that
    only ever appear in TLH is wasted bandwidth and ~85% of the
    daily_prices.csv row count for nothing.

    Strict membership rule, to be safe: a ticker is "TLH-only" iff every
    historical row for it sits in either
      - account_id == cfg.TLH_ACCOUNT_ID, or
      - asset_class == "tax_loss_harvesting" (the direct-indexing sleeve
        tag applied by the asset reclassifier).
    A ticker held in *both* TLH and an individual sleeve (e.g. AAPL is a
    common S&P member that the user may hold both standalone and inside
    the TLH wrapper) does NOT qualify — its non-TLH appearance keeps it
    in the fetch universe.

    Silent fallback to empty set when config_local / TLH_ACCOUNT_ID is
    absent (the exclusion just becomes a no-op).
    """
    try:
        import sys as _sys
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import config_local as _cfg  # type: ignore
    except ImportError:
        return set()
    tlh_acct = getattr(_cfg, "TLH_ACCOUNT_ID", None)
    if not tlh_acct:
        return set()

    positions = pd.read_csv(POSITIONS_CSV, parse_dates=["statement_date"])
    in_tlh = (
        (positions["account_id"] == tlh_acct)
        | (positions["asset_class"] == "tax_loss_harvesting")
    )
    has_sym = positions["symbol"].notna()
    tlh_syms = set(positions.loc[in_tlh & has_sym, "symbol"]
                            .astype(str).str.strip().unique())
    elsewhere_syms = set(positions.loc[~in_tlh & has_sym, "symbol"]
                                  .astype(str).str.strip().unique())
    return tlh_syms - elsewhere_syms


def collect_prior_symbols() -> list[str]:
    """Return sorted unique prior_symbols from TICKER_HISTORY in config_local.

    Reads the user's corporate-action config so renamed positions (current
    ticker in positions.csv, prior ticker only known via config) still get
    their historical price data fetched. Silent fallback to [] when
    config_local doesn't exist or doesn't define TICKER_HISTORY.
    """
    try:
        # Importing config_local is awkward from a parsers/ script because
        # config_local.py lives at the repo root, not in parsers/. Add the
        # repo root to sys.path temporarily.
        import sys as _sys
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import config_local as _cfg  # type: ignore
    except ImportError:
        return []
    history = getattr(_cfg, "TICKER_HISTORY", {}) or {}
    priors: set[str] = set()
    for segments in history.values():
        for seg in segments or []:
            prior = (seg.get("prior_symbol") or "").strip()
            if prior:
                priors.add(prior)
    return sorted(priors)


def fetch_daily_history(ticker: str, start: date, end: date,
                        key: str, base: str) -> dict:
    """Fetch daily bars. Returns dict with 'bars' list or 'error' string."""
    url = f"{base}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    try:
        r = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc",
                    "limit": 50000, "apiKey": key},
            timeout=60,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}
    payload = r.json()
    bars = payload.get("results") or []
    out = []
    for b in bars:
        d = (pd.to_datetime(b["t"], unit="ms")
               .tz_localize("UTC")
               .tz_convert("America/New_York")
               .date())
        out.append({"date": d.isoformat(), "close": float(b["c"])})
    return {"bars": out}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Fetch ALL symbols and emit data/daily_prices.csv "
                         "(default: sample 3 tickers, dry-run).")
    ap.add_argument("--years", type=int, default=DEFAULT_LOOKBACK_YEARS,
                    help=f"Lookback in years (default: {DEFAULT_LOOKBACK_YEARS}).")
    ap.add_argument("--latest-only", action="store_true",
                    help="Restrict universe to the latest statement snapshot "
                         "only — skips positions that were held in the past "
                         "but sold. Default is to include all ever-held "
                         "symbols so historical analytics have full data.")
    ap.add_argument("--full", action="store_true",
                    help="Ignore on-disk data and re-pull full history for "
                         "every symbol (clean rebuild).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent per-ticker fetches (default 8). Higher is "
                         "faster but risks Polygon rate limits; use 1 for fully "
                         "sequential.")
    args = ap.parse_args()

    if not POSITIONS_CSV.exists():
        print(f"[!] {POSITIONS_CSV} not found")
        return 1

    held_raw = collect_equity_symbols(latest_only=args.latest_only)
    tlh_only = collect_tlh_only_symbols()
    held = [s for s in held_raw if s not in tlh_only]
    overlay = [s for s in MACRO_OVERLAY if s not in held]
    priors  = [s for s in collect_prior_symbols()
               if s not in held and s not in overlay]
    all_syms = held + overlay + priors

    end = date.today()
    start = end - timedelta(days=args.years * 365 + LOOKBACK_BUFFER_DAYS)

    scope = "latest snapshot only" if args.latest_only else "all ever-held"
    print(f"Equity-class symbols ({scope}): {len(held_raw)} raw"
          f" → {len(held)} after dropping {len(tlh_only)} TLH-only "
          f"(folded to SPY by risk synthesis)")
    print(f"Macro overlay (additional): {overlay}")
    if priors:
        print(f"Prior tickers (from TICKER_HISTORY): {priors}")
    print(f"Window: {start} to {end} ({args.years}y + {LOOKBACK_BUFFER_DAYS}d buffer)")
    print()

    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()

    if not args.write:
        # Sample: SPY (always there), and 2 random held names (deterministic
        # via sort) — gives a feel for typical row counts + freshness.
        sample = ["SPY"]
        for s in held:
            if s != "SPY":
                sample.append(s)
            if len(sample) >= 3:
                break
        print(f"DRY-RUN sample fetch on: {sample}")
        print(f"{'Symbol':10s} {'Bars':>6s}  {'First':12s} {'Last':12s} {'Last close':>12s}")
        print("-" * 60)
        for sym in sample:
            poly_sym = TICKER_ALIASES.get(sym, sym)
            res = fetch_daily_history(poly_sym, start, end, key, base)
            if "error" in res:
                print(f"{sym:10s} ERROR: {res['error'][:50]}")
                continue
            bars = res["bars"]
            if not bars:
                print(f"{sym:10s} {0:>6d}  (no data)")
                continue
            print(f"{sym:10s} {len(bars):>6d}  "
                  f"{bars[0]['date']:12s} {bars[-1]['date']:12s} "
                  f"${bars[-1]['close']:>11,.2f}")
        print()
        print(f"(dry-run — re-run with --write to fetch all {len(all_syms)} "
              f"symbols and emit {OUT_CSV})")
        return 0

    # --write: incremental refresh (split-safe append; see incremental_fetch.py)
    def _fetch_fn(sym, fstart, fend):
        poly_sym = TICKER_ALIASES.get(sym, sym)
        res = fetch_daily_history(poly_sym, fstart, fend, key, base)
        if "error" in res or not res.get("bars"):
            return pd.DataFrame(columns=["date", "close"])
        return pd.DataFrame(res["bars"], columns=["date", "close"])

    print(f"Refreshing {len(all_syms)} symbols (incremental"
          f"{' — FULL re-pull' if args.full else ''})...")
    summary = refresh_csv(
        OUT_CSV, all_syms, _fetch_fn,
        lookback_start=start, today=end,
        empty_columns=["symbol", "date", "close"],
        group_col="symbol", full=args.full, max_workers=args.workers)
    print(f"Wrote {OUT_CSV} — "
          + ", ".join(f"{k}={v}" for k, v in summary.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
