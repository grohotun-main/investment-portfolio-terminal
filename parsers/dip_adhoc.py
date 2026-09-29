"""Ad-hoc (typed) buy-the-dip ticker support: ticker normalization, per-symbol
slicing shared with the auto cards, sidecar-cache upsert/staleness, an offline
fetcher pair for tests, and resolve_adhoc() (added in a later task) — the
cache->sidecar->fetch orchestration.

Pure functions over pandas; no Streamlit. The live network wrappers live in
fetch_dip_history and are injected, never called from tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_dip_history import build_history  # noqa: E402  (parsers/ on sys.path)

MIN_HISTORY_DAYS = 252  # ~1 trading year; below this, dip stats are unreliable

_HIST_COLS = ["symbol", "date", "close", "adj_close"]
_DIV_COLS = ["symbol", "ex_date", "amount"]


def normalize_ticker(raw) -> str:
    """Upper-cased, stripped ticker; '' for blank/None."""
    return str(raw or "").strip().upper()


def slice_symbol(hist: pd.DataFrame, divs: pd.DataFrame, sym: str):
    """(price, tr, dser) for one symbol: close + adj_close (date-indexed, sorted,
    NaN-dropped) and the dividend amount series (ex_date-indexed). Mirrors the
    inline slice the auto-card loop used, so auto and ad-hoc cards share it."""
    g = hist[hist["symbol"] == sym].set_index("date").sort_index()
    price = g["close"].astype(float).dropna()
    tr = g["adj_close"].astype(float).dropna()
    dser = divs[divs["symbol"] == sym].set_index("ex_date")["amount"].astype(float)
    return price, tr, dser


def adhoc_upsert(existing: pd.DataFrame, ticker: str, fresh: pd.DataFrame) -> pd.DataFrame:
    """Replace `ticker`'s rows in `existing` with `fresh` (both carry a 'symbol'
    column). Order-stable: other tickers keep their order; the upserted rows go
    last. Used for both the history and dividend sidecars."""
    keep = existing[existing["symbol"] != ticker] if not existing.empty else existing
    return pd.concat([keep, fresh], ignore_index=True)


def adhoc_is_stale(existing: pd.DataFrame, ticker: str, ref_date) -> bool:
    """True if `ticker` is absent from `existing` or its newest `date` < ref_date.
    Operates on the history sidecar (it reads the `date` column); the dividend
    sidecar is not staleness-checked. `adhoc_upsert` is the both-sidecars helper."""
    if existing.empty or ticker not in set(existing["symbol"]):
        return True
    last = pd.to_datetime(existing.loc[existing["symbol"] == ticker, "date"]).max()
    return last < pd.Timestamp(ref_date)


def offline_fetchers(source_csv: Path):
    """(price_fn, div_fn) reading a fixture CSV instead of the network — the test
    seam. Price source: `source_csv` (symbol,date,close,adj_close). Dividend
    source: `<source_csv stem>_dividends.csv` if present, else none. Signatures
    match build_history's injected fetchers: price_fn(ticker, start, end),
    div_fn(ticker)."""
    src = pd.read_csv(source_csv, parse_dates=["date"])

    def _price(ticker, start, end):
        g = src[src["symbol"] == ticker]
        return g[["date", "close", "adj_close"]].reset_index(drop=True)

    div_path = source_csv.with_name(source_csv.stem + "_dividends.csv")
    div_src = (pd.read_csv(div_path, parse_dates=["ex_date"]) if div_path.exists()
               else pd.DataFrame(columns=_DIV_COLS))

    def _div(ticker):
        g = div_src[div_src["symbol"] == ticker]
        return g[["ex_date", "amount"]].reset_index(drop=True)

    return _price, _div


def _empty_payload(status: str, msg: str = "") -> dict:
    e = pd.Series(dtype=float)
    return {"status": status, "price": e, "tr": e, "dser": e,
            "asof": None, "n_days": 0, "stale": False, "msg": msg}


def resolve_adhoc(data_dir, ticker, vintage, price_fn, div_fn, today,
                  *, persist: bool) -> dict:
    """Resolve an ad-hoc ticker to a render-ready payload.

    Order: a fresh sidecar (data_dir/dip_adhoc_history.csv) -> fetch via the
    injected (price_fn, div_fn) -> upsert + write the sidecar (only when
    `persist`). Returns {status, price, tr, dser, asof, n_days, stale, msg};
    status in {"ok","empty","short","error"}. On a fetch exception, falls back
    to a (possibly stale) sidecar copy if one exists, else status="error".
    """
    data_dir = Path(data_dir)
    hist_path = data_dir / "dip_adhoc_history.csv"
    div_path = data_dir / "dip_adhoc_dividends.csv"
    side_h = (pd.read_csv(hist_path, parse_dates=["date"]) if hist_path.exists()
              else pd.DataFrame(columns=_HIST_COLS))
    side_d = (pd.read_csv(div_path, parse_dates=["ex_date"]) if div_path.exists()
              else pd.DataFrame(columns=_DIV_COLS))

    def _payload(h, d, *, stale=False) -> dict:
        price, tr, dser = slice_symbol(h, d, ticker)
        n = int(len(price))
        if n == 0:
            p = _empty_payload("empty")
            p["stale"] = stale
            return p
        status = "ok" if n >= MIN_HISTORY_DAYS else "short"
        return {"status": status, "price": price, "tr": tr, "dser": dser,
                "asof": price.index[-1], "n_days": n, "stale": stale, "msg": ""}

    if not adhoc_is_stale(side_h, ticker, vintage):
        return _payload(side_h, side_d)

    try:
        h_t, d_t = build_history([ticker], price_fn, div_fn, today)
    except Exception as exc:  # network / yfinance failure
        if not side_h.empty and ticker in set(side_h["symbol"]):
            return _payload(side_h, side_d, stale=True)
        return _empty_payload("error", str(exc))

    if h_t.empty:
        return _empty_payload("empty")

    if persist:
        new_h = adhoc_upsert(side_h, ticker, h_t)
        new_d = adhoc_upsert(side_d, ticker, d_t)
        data_dir.mkdir(parents=True, exist_ok=True)
        new_h.to_csv(hist_path, index=False)
        new_d.to_csv(div_path, index=False)
        return _payload(new_h, new_d)
    return _payload(h_t, d_t)
