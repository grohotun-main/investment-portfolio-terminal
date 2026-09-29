"""Fetch DEEP daily history (raw close for drawdowns + adj close for total
return) and cash dividends for the buy-the-dip watchlist. Yahoo Finance primary
(deep history + adj close + dividends in one source), Stooq cross-check. Output
is gitignored: data/dip_history.csv (symbol,date,close,adj_close) and
data/dip_dividends.csv (symbol,ex_date,amount). A degraded fetch never
overwrites populated output (the no-clobber guard in emit_outputs; exit 1).

Pure assembler `build_history(tickers, fetch_fn, div_fn, today)` is unit-tested
with injected fetchers; the real `fetch_yahoo`/`fetch_dividends_yahoo` wrappers
hit the network and are NEVER called from tests/CI (no-network rule).

Run:
  py parsers/fetch_dip_history.py --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HIST_CSV = DATA / "dip_history.csv"
DIV_CSV = DATA / "dip_dividends.csv"

DEFAULT_TICKERS = ["SPY", "SCHD", "GLD"]

# SPX index-history extension (spec 2026-07-19 §4a): fetched in the same
# --write run, written to a SIDECAR csv. Never into dip_history.csv — every
# symbol there becomes a UI card (terminal/dip_service.py).
INDEX_TICKERS = ["^GSPC", "^SP500TR"]
INDEX_CSV = DATA / "dip_index_history.csv"


def _no_dividends(ticker) -> pd.DataFrame:
    """Indices distribute nothing; keeps build_history's div_fn contract."""
    return pd.DataFrame(columns=["ex_date", "amount"])


def build_index_history(fetch_fn, today) -> pd.DataFrame:
    """Stacked ^GSPC/^SP500TR history via the injected fetcher (same shape
    as dip_history.csv)."""
    hist, _ = build_history(INDEX_TICKERS, fetch_fn, _no_dividends, today)
    return hist


def index_history_ok(hist: pd.DataFrame) -> bool:
    """Both index symbols present with real depth — the no-clobber guard
    (spec §5: an empty fetch must not overwrite a good sidecar)."""
    if hist is None or hist.empty:
        return False
    counts = hist.groupby("symbol")["date"].count()
    return all(int(counts.get(t, 0)) >= 1000 for t in INDEX_TICKERS)


def _csv_rows(path: Path) -> int:
    """Data-row count of an existing CSV (0 when missing, header-only, or
    unreadable — a corrupt file is not worth protecting)."""
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def history_ok(hist: pd.DataFrame, tickers: list[str]) -> bool:
    """No-clobber trust test for the PRIMARY csvs (index_history_ok's spec §5
    applied to dip_history itself, after the 2026-08-19 truncation): the fetch
    is trusted iff every always-on core ticker it was asked for came back with
    rows. The core names are deeply liquid — an empty return there means the
    source broke, not the ticker. Watchlist extras may legitimately be empty
    (typo, delisting) without blocking the write."""
    if hist is None or hist.empty:
        return False
    counts = hist.groupby("symbol")["date"].count()
    return all(int(counts.get(t, 0)) >= 1
               for t in DEFAULT_TICKERS if t in tickers)


def emit_outputs(hist: pd.DataFrame, divs: pd.DataFrame, tickers: list[str],
                 hist_csv: Path = HIST_CSV, div_csv: Path = DIV_CSV) -> int:
    """Guarded writes for the two primary csvs — a degraded fetch must never
    destroy good data (2026-08-19: a total TLS failure returned 0-row frames
    and --write truncated both files to header-only). Returns the process
    exit code: 0 = healthy write; 1 = degraded fetch (populated files are
    kept; with nothing to protect the empty artifacts are still written so a
    bootstrap run behaves as before)."""
    if not history_ok(hist, tickers):
        kept = [p.name for p in (hist_csv, div_csv) if _csv_rows(p) > 0]
        if kept:
            print(f"ERROR: degraded fetch ({len(hist)} history rows) — "
                  f"refusing to overwrite populated {', '.join(kept)}",
                  file=sys.stderr)
            return 1
        hist.to_csv(hist_csv, index=False)
        divs.to_csv(div_csv, index=False)
        print(f"WARNING: degraded fetch and nothing to protect — wrote "
              f"{hist_csv} and {div_csv} anyway", file=sys.stderr)
        return 1
    hist.to_csv(hist_csv, index=False)
    if divs.empty and _csv_rows(div_csv) > 0:
        print(f"wrote {hist_csv}")
        print(f"WARNING: dividend fetch came back empty — keeping existing "
              f"{div_csv}", file=sys.stderr)
        return 1
    divs.to_csv(div_csv, index=False)
    print(f"wrote {hist_csv} and {div_csv}")
    return 0


def build_history(tickers, fetch_fn, div_fn, today) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble stacked history + dividend frames from injected per-ticker
    fetchers. `fetch_fn(ticker, start, end) -> DataFrame[date, close, adj_close]`;
    `div_fn(ticker) -> DataFrame[ex_date, amount]`."""
    hist_parts, div_parts = [], []
    for t in tickers:
        h = fetch_fn(t, None, today).copy()
        h.insert(0, "symbol", t)
        hist_parts.append(h[["symbol", "date", "close", "adj_close"]])
        d = div_fn(t).copy()
        d.insert(0, "symbol", t)
        div_parts.append(d[["symbol", "ex_date", "amount"]])
    hist = pd.concat(hist_parts, ignore_index=True) if hist_parts else \
        pd.DataFrame(columns=["symbol", "date", "close", "adj_close"])
    # Yahoo returns a trailing NaN bar for the current unsettled session; drop
    # rows with no close so the latest row is always a real settled price (the
    # render reads the last close as the current price).
    hist = hist.dropna(subset=["close"]).reset_index(drop=True)
    divs = pd.concat(div_parts, ignore_index=True) if div_parts else \
        pd.DataFrame(columns=["symbol", "ex_date", "amount"])
    return hist, divs


def validate_against_polygon(dip_spy: pd.DataFrame, poly_spy: pd.DataFrame,
                              min_corr: float = 0.99) -> dict:
    """Cross-check the Yahoo SPY close against the existing Polygon SPY over the
    overlapping dates: daily-return correlation must exceed `min_corr`. Returns
    {ok, return_corr, n_overlap}; callers warn (don't crash) on ok=False."""
    a = dip_spy[["date", "close"]].copy()
    b = poly_spy[["date", "close"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    m = a.merge(b, on="date", suffixes=("_dip", "_poly")).sort_values("date")
    if len(m) < 3:
        return {"ok": False, "return_corr": float("nan"), "n_overlap": int(len(m))}
    ra = m["close_dip"].pct_change().dropna()
    rb = m["close_poly"].pct_change().dropna()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(ra.corr(rb))
    return {"ok": bool(corr >= min_corr), "return_corr": corr, "n_overlap": int(len(m))}


# ---------------------------------------------------------------------------
# D3: Real network wrappers (lazy import — yfinance NOT imported at module top
# so the module loads fine without it installed; never called from tests/CI).
# ---------------------------------------------------------------------------

def fetch_yahoo(ticker, start, end) -> pd.DataFrame:
    """Network. Daily history with raw + dividend-adjusted close. start=None pulls
    max available (capped by the source). Not called in tests."""
    import yfinance as yf  # lazy — keeps the module importable without yfinance
    df = yf.Ticker(ticker).history(start=start, end=end, period=None if start else "max",
                                   auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "close", "adj_close"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df.index).tz_localize(None),
        "close": df["Close"].to_numpy(dtype=float),
        "adj_close": df["Adj Close"].to_numpy(dtype=float),
    })
    return out.reset_index(drop=True)


def fetch_dividends_yahoo(ticker) -> pd.DataFrame:
    """Network. Cash distributions (ex-date, amount). Not called in tests."""
    import yfinance as yf  # lazy — keeps the module importable without yfinance
    s = yf.Ticker(ticker).dividends
    if s is None or len(s) == 0:
        return pd.DataFrame(columns=["ex_date", "amount"])
    return pd.DataFrame({"ex_date": pd.to_datetime(s.index).tz_localize(None),
                         "amount": s.to_numpy(dtype=float)}).reset_index(drop=True)


def dip_universe(extra: list[str] | None = None) -> list[str]:
    """SPY + SCHD always; plus any config_local DIP_WATCHLIST additions + `extra`
    (e.g. held equity tickers passed by the caller). Deduped, order-stable."""
    seen, out = set(), []
    try:
        import sys as _sys
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import config_local as _cfg  # gitignored; holds the real watchlist
        DIP_WATCHLIST = getattr(_cfg, "DIP_WATCHLIST", []) or []
    except ImportError:
        DIP_WATCHLIST = []
    for t in DEFAULT_TICKERS + list(DIP_WATCHLIST) + list(extra or []):
        u = str(t).upper().strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="fetch + emit CSVs")
    ap.add_argument("--tickers", nargs="*", default=None, help="override universe")
    args = ap.parse_args()
    tickers = args.tickers or dip_universe()
    if not args.write:
        print(f"[dry-run] would fetch {len(tickers)} tickers: {', '.join(tickers)}")
        print("re-run with --write to fetch and emit CSVs")
        return 0
    today = pd.Timestamp.today().normalize()
    hist, divs = build_history(tickers, fetch_yahoo, fetch_dividends_yahoo, today)
    print(f"fetched {hist['symbol'].nunique()} symbols, {len(hist)} rows; "
          f"{len(divs)} dividend rows")
    DATA.mkdir(exist_ok=True)
    rc = emit_outputs(hist, divs, tickers)
    ih = build_index_history(fetch_yahoo, today)
    if index_history_ok(ih):
        ih.to_csv(INDEX_CSV, index=False)
        print(f"wrote {INDEX_CSV} ({len(ih)} index rows)")
    else:
        print("WARNING: index fetch incomplete — sidecar not written"
              + ("; keeping existing " + str(INDEX_CSV)
                 if INDEX_CSV.exists() else ""),
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
