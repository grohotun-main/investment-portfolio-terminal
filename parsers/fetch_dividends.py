"""Fetch dividend history for a ticker from Massive (Polygon).

Used by build_benchmark_total_return.py to compute a dividend-reinvested
total-return index for benchmark comparison.

Idempotent: overwrites data/dividends_<ticker>.csv each run.

Run:  py parsers\\fetch_dividends.py --write            # default SPY, last 10y
  py parsers\\fetch_dividends.py --write VOO
  py parsers\\fetch_dividends.py                        # smoke test, no CSV written
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests

from _config import get_massive_key, get_massive_base
# Shared with the income engine: anything symbol-bearing outside these
# classes is fetchable (Harbor files SGOV under fixed_income and GLD under
# other — class is not a reliable "listed" signal). Polygon returns zero
# rows for unlisted symbols, which the allow_empty path records as a
# header-only known-non-payer file.
from income_analytics import NON_INCOME_CLASSES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Matches fetch_benchmark.py so the dividend window covers the price window.
MAX_LOOKBACK_DAYS = 365 * 10 - 1

POSITIONS_CSV = DATA_DIR / "positions.csv"
# The close matrices whose every column needs a dividend file (total-return
# adjustment, parsers/total_return.py) — held names, benchmarks, proxies,
# renamed-ticker priors alike — and the one long splits file for the same
# universe (header-only when nothing split). Resolved from DATA_DIR at call
# time, never as import-time constants: a test that patches DATA_DIR must
# never write into the real data dir (it did, once).
PRICE_MATRIX_NAMES = ("daily_prices.csv", "long_history_prices.csv")
SPLITS_NAME = "splits.csv"
SPLIT_COLUMNS = ["execution_date", "split_from", "split_to"]
# Incremental refresh (Distributions S3, spec 2026-08-23): a run stamp under
# DATA_DIR decides between the full per-ticker sweep and a market-wide DELTA
# (Polygon's ticker-less list endpoints since the last run, filtered to our
# universe). A full sweep still happens on --full, without a stamp, or every
# FULL_SWEEP_DAYS — the safety net for vendor revisions of old rows.
META_NAME = "dividends_meta.json"
FULL_SWEEP_DAYS = 90
DELTA_OVERLAP_DAYS = 3
_today = date.today          # patchable clock (tests pin the run date)

# Polygon /v3/reference/dividends result columns — used as the header of
# an empty CSV so a confirmed non-payer is distinguishable from
# "never fetched" (see income_analytics.load_div_history).
EMPTY_COLUMNS = ["cash_amount", "currency", "declaration_date",
                 "dividend_type", "ex_dividend_date", "frequency", "id",
                 "pay_date", "record_date", "ticker"]


def collect_dividend_universe() -> list[str]:
    """Unique non-option, non-cash symbols in the latest positions.csv
    month (bare-CUSIP bond rungs have no symbol and drop out naturally).

    Raises ValueError if positions.csv exists but lacks the required columns
    (statement_date, symbol, asset_class) — consistent with other CLI fetchers
    that hard-crash on a malformed positions.csv.
    """
    if not POSITIONS_CSV.exists():
        return []
    pos = pd.read_csv(POSITIONS_CSV,
                      usecols=["statement_date", "symbol", "asset_class"])
    if pos.empty:
        return []
    cur = pos[(pos["statement_date"] == pos["statement_date"].max())
              & ~pos["asset_class"].isin(NON_INCOME_CLASSES)]
    syms = cur["symbol"].dropna().astype(str).str.strip().str.upper()
    return sorted({s for s in syms if s})


def collect_price_universe(data_dir: "Path | None" = None) -> list[str]:
    """Every symbol in ``daily_prices.csv`` / ``long_history_prices.csv``
    under ``data_dir`` (default DATA_DIR) — held names, benchmarks, proxies,
    renamed-ticker priors. The total-return adjustment reads the whole close
    matrix, so every column needs a dividend file — a benchmark left
    price-only would be the one series still sawing down on ex-dates.
    Missing files contribute nothing."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    syms: set[str] = set()
    for name in PRICE_MATRIX_NAMES:
        path = base / name
        if not path.exists():
            continue
        try:
            col = pd.read_csv(path, usecols=["symbol"])["symbol"]
        except (ValueError, pd.errors.EmptyDataError):
            continue
        syms.update(s for s in col.dropna().astype(str).str.strip().str.upper()
                    if s)
    return sorted(syms)


def fetch_splits(ticker: str, since: date, *,
                 allow_empty: bool = True) -> pd.DataFrame:
    """Stock splits for *ticker* from Polygon ``/v3/reference/splits``:
    ``execution_date, split_from, split_to`` (10:1 = from 1 to 10).
    Polygon's dividend ``cash_amount`` is as-declared while its bars are
    split-adjusted, so the total-return adjustment divides pre-split
    dividends by every later ratio — it needs this table."""
    key = get_massive_key()
    base = get_massive_base()
    url = f"{base}/v3/reference/splits"
    params = {"ticker": ticker, "execution_date.gte": since.isoformat(),
              "limit": 1000, "order": "asc", "sort": "execution_date",
              "apiKey": key}
    rows: list[dict] = []
    next_url = None
    while True:
        if next_url:
            r = requests.get(next_url, params={"apiKey": key}, timeout=30)
        else:
            r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
    if not rows:
        if allow_empty:
            return pd.DataFrame(columns=SPLIT_COLUMNS)
        raise RuntimeError(f"No splits returned for {ticker} since {since}")
    df = pd.DataFrame(rows)
    keep = [c for c in SPLIT_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce").dt.date
    return df.dropna(subset=["execution_date"])


def fetch_splits_for_holding(ticker: str, since: date,
                             history: dict) -> pd.DataFrame:
    """Splits for *ticker* plus, re-keyed to *ticker*, the prior symbols'
    splits executed before each rename's effective date — the spliced
    price history carries the prior segment under the current column, so
    its dividends (already merged the same way) need the prior splits."""
    df = fetch_splits(ticker, since, allow_empty=True)
    parts = [df] if not df.empty else []
    for seg in history.get(ticker, []) or []:
        prior_sym = (seg.get("prior_symbol") or "").strip()
        eff = seg.get("effective_date")
        if not prior_sym or not eff:
            continue
        prior = fetch_splits(prior_sym, since, allow_empty=True)
        if prior.empty:
            continue
        cut = pd.Timestamp(eff)
        pre = prior[pd.to_datetime(prior["execution_date"]) < cut]
        if not pre.empty:
            parts.append(pre)
    if not parts:
        return pd.DataFrame(columns=SPLIT_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["execution_date", "split_from", "split_to"])


def _ticker_history() -> dict:
    """TICKER_HISTORY from config_local; {} when absent (same fallback
    idiom as fetch_daily_prices.collect_prior_symbols)."""
    try:
        import sys as _sys
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import config_local as _cfg  # type: ignore
    except ImportError:
        return {}
    return getattr(_cfg, "TICKER_HISTORY", {}) or {}


def merge_spliced(current: pd.DataFrame, prior: pd.DataFrame,
                  effective_date: str) -> pd.DataFrame:
    """Prior-symbol dividends strictly before effective_date + current."""
    if prior is None or prior.empty:
        return current
    cut = pd.Timestamp(effective_date)
    if pd.isna(cut):
        raise ValueError(
            f"merge_spliced: unparseable effective_date {effective_date!r}")
    # ex_dividend_date holds datetime.date objects (fetch_dividends coerces
    # with .dt.date); pandas refuses date-vs-Timestamp comparisons, which
    # failed every renamed ticker whose prior symbol paid (BNY <- BK).
    pre = prior[pd.to_datetime(prior["ex_dividend_date"]) < cut]
    out = pd.concat([pre, current], ignore_index=True)
    # One type on both sides before sorting (a caller may hand Timestamps).
    out["ex_dividend_date"] = pd.to_datetime(out["ex_dividend_date"],
                                             errors="coerce").dt.date
    return (out.sort_values("ex_dividend_date")
            .reset_index(drop=True))


def fetch_for_holding(ticker: str, since: date,
                      history: dict) -> pd.DataFrame:
    """Single-ticker fetch + rename splice (TICKER_HISTORY segments).

    Uses allow_empty=True so genuine non-payers (and non-paying prior
    symbols) return an empty frame instead of raising — the raise-on-empty
    contract is only needed for the single-ticker benchmark path.
    """
    df = fetch_dividends(ticker, since, allow_empty=True)
    for seg in history.get(ticker, []) or []:
        prior_sym = (seg.get("prior_symbol") or "").strip()
        eff = seg.get("effective_date")
        if not prior_sym or not eff:
            continue
        prior = fetch_dividends(prior_sym, since, allow_empty=True)
        df = merge_spliced(df, prior, eff)
    return df


def fetch_dividends(ticker: str, since: date, *,
                    allow_empty: bool = False) -> pd.DataFrame:
    """Fetch dividend history for *ticker* from Polygon.

    Raises RuntimeError when Polygon returns zero rows, UNLESS
    ``allow_empty=True`` — in that case an empty DataFrame with the
    canonical EMPTY_COLUMNS is returned.  The default (raise) is kept so
    the benchmark path still fails loudly on an unexpectedly empty SPY
    response.
    """
    key = get_massive_key()
    base = get_massive_base()
    url = f"{base}/v3/reference/dividends"
    params = {
        "ticker": ticker,
        "ex_dividend_date.gte": since.isoformat(),
        "limit": 1000,
        "order": "asc",
        "sort": "ex_dividend_date",
        "apiKey": key,
    }
    rows: list[dict] = []
    next_url = None
    while True:
        if next_url:
            r = requests.get(next_url, params={"apiKey": key}, timeout=30)
        else:
            r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
    if not rows:
        if allow_empty:
            return pd.DataFrame(columns=EMPTY_COLUMNS)
        raise RuntimeError(f"No dividends returned for {ticker} since {since}")
    df = pd.DataFrame(rows)
    for col in ("ex_dividend_date", "pay_date", "declaration_date", "record_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _paged(url: str, params: dict) -> list[dict]:
    """Polygon list endpoint, following ``next_url`` to the end."""
    key = get_massive_key()
    rows: list[dict] = []
    next_url = None
    while True:
        if next_url:
            r = requests.get(next_url, params={"apiKey": key}, timeout=60)
        else:
            r = requests.get(url, params={**params, "apiKey": key}, timeout=60)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
    return rows


def fetch_dividends_since(since: date) -> pd.DataFrame:
    """MARKET-WIDE dividends with ex-date >= ``since`` (no ticker filter) —
    the incremental delta. Same columns as ``fetch_dividends`` (Polygon's
    result fields incl. ``ticker`` and ``id``); empty frame when none. A
    7-day window is ~6 pages of 1000."""
    rows = _paged(f"{get_massive_base()}/v3/reference/dividends",
                  {"ex_dividend_date.gte": since.isoformat(), "limit": 1000,
                   "order": "asc", "sort": "ex_dividend_date"})
    if not rows:
        return pd.DataFrame(columns=EMPTY_COLUMNS)
    df = pd.DataFrame(rows)
    for col in ("ex_dividend_date", "pay_date", "declaration_date", "record_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def fetch_splits_since(since: date) -> pd.DataFrame:
    """MARKET-WIDE splits executed on/after ``since``:
    ``[ticker, execution_date, split_from, split_to]``."""
    rows = _paged(f"{get_massive_base()}/v3/reference/splits",
                  {"execution_date.gte": since.isoformat(), "limit": 1000,
                   "order": "asc", "sort": "execution_date"})
    cols = ["ticker", *SPLIT_COLUMNS]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]].copy()
    df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce").dt.date
    return df.dropna(subset=["execution_date"])


def _read_meta(data_dir: Path) -> dict:
    p = Path(data_dir) / META_NAME
    if not p.exists():
        return {}
    try:
        # utf-8-sig: a stamp hand-written from PowerShell carries a BOM, which
        # plain json.loads rejects — that would silently force a full sweep.
        return json.loads(p.read_text(encoding="utf-8-sig")) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_meta(data_dir: Path, mode: str, today: date, prev: dict | None) -> None:
    meta = dict(prev or {})
    meta["last_run_asof"] = today.isoformat()
    meta["mode"] = mode
    if mode == "full":
        meta["last_full_run"] = today.isoformat()
    (Path(data_dir) / META_NAME).write_text(json.dumps(meta, indent=2) + "\n",
                                            encoding="utf-8")


def _needs_full(meta: dict, today: date, force: bool) -> bool:
    """Full sweep on --full, without a usable stamp, or every FULL_SWEEP_DAYS."""
    if force or not meta:
        return True
    try:
        last_full = date.fromisoformat(str(meta.get("last_full_run", "")))
        date.fromisoformat(str(meta.get("last_run_asof", "")))
    except ValueError:
        return True
    return (today - last_full).days >= FULL_SWEEP_DAYS


def _row_keys(df: pd.DataFrame) -> pd.Series:
    """Polygon's ``id`` when present, else (ex-date, cash) — the identity a
    delta row replaces in a per-ticker file."""
    ex = pd.to_datetime(df["ex_dividend_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cash = pd.to_numeric(df["cash_amount"], errors="coerce").round(6).astype(str)
    fallback = ex.astype("string") + "|" + cash.astype("string")
    if "id" in df.columns:
        ids = df["id"].astype("string")
        ids = ids.where(ids.notna() & (ids != "") & (ids != "nan"), fallback)
        return ids
    return fallback


def _merge_dividend_rows(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Existing file rows + delta rows: a delta row REPLACES an existing row
    with the same id (a revised amount / date), otherwise appends. Sorted by
    ex-date, ISO date strings throughout (what the files already hold)."""
    new = new.copy()
    new["ex_dividend_date"] = pd.to_datetime(new["ex_dividend_date"],
                                             errors="coerce").dt.strftime("%Y-%m-%d")
    ex = (existing.copy() if existing is not None and not existing.empty
          else pd.DataFrame(columns=EMPTY_COLUMNS))
    if not ex.empty:
        ex["ex_dividend_date"] = pd.to_datetime(ex["ex_dividend_date"],
                                                errors="coerce").dt.strftime("%Y-%m-%d")
    # Row identity = Polygon id (else ex-date|cash). A delta row wins field
    # by field; a field the delta leaves blank keeps the file's value, so a
    # partial payload never erases currency / frequency / pay_date.
    new.index = pd.Index(_row_keys(new).astype(str), name="_k")
    new = new[~new.index.duplicated(keep="last")]
    if ex.empty:
        out = new.reset_index(drop=True)
    else:
        ex.index = pd.Index(_row_keys(ex).astype(str), name="_k")
        ex = ex[~ex.index.duplicated(keep="last")]
        out = new.combine_first(ex).reset_index(drop=True)
    cols = ([c for c in EMPTY_COLUMNS if c in out.columns]
            + [c for c in out.columns if c not in EMPTY_COLUMNS])
    return out[cols].sort_values("ex_dividend_date").reset_index(drop=True)


def _read_div_file(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=EMPTY_COLUMNS)
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EMPTY_COLUMNS)


def _fetch_many(syms: list[str], history: dict, since: date, workers: int,
                failures: list, split_failures: list) -> list[tuple]:
    """Per-ticker dividends + splits for ``syms`` → [(sym, divs|None, splits|None)]."""
    def _one(sym: str):
        try:
            divs = fetch_for_holding(sym, since, history)
        except Exception as exc:  # non-fatal per ticker
            failures.append((sym, repr(exc)))
            return sym, None, None
        try:
            splits = fetch_splits_for_holding(sym, since, history)
        except Exception as exc:  # splits are an overlay — never block dividends
            split_failures.append((sym, repr(exc)))
            splits = None
        return sym, divs, splits

    if workers <= 1 or len(syms) <= 1:
        return [_one(s) for s in syms]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, syms))


def _write_per_ticker(results: list[tuple], write: bool,
                      split_frames: list[pd.DataFrame]) -> int:
    """Write each ticker's dividend file (header-only for a non-payer);
    collect its splits (with a ``symbol`` column). Returns files written."""
    written = 0
    for sym, df, splits in results:
        if df is None:
            continue
        out_csv = DATA_DIR / f"dividends_{sym.lower()}.csv"
        if write:
            if df.empty:
                pd.DataFrame(columns=EMPTY_COLUMNS).to_csv(out_csv, index=False)
            else:
                df.to_csv(out_csv, index=False)
            written += 1
        if splits is not None and not splits.empty:
            s = splits.copy()
            s.insert(0, "symbol", sym)
            split_frames.append(s)
    return written


def _merge_splits(existing: pd.DataFrame, new_frames: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["symbol", *SPLIT_COLUMNS]
    parts = [f for f in [existing, *new_frames] if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat([p[cols] for p in parts], ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["execution_date"] = pd.to_datetime(out["execution_date"],
                                           errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["execution_date"])
    return (out.drop_duplicates(subset=cols)
               .sort_values(["symbol", "execution_date"]).reset_index(drop=True))


def _full_sweep(syms: list[str], history: dict, since: date, *,
                workers: int, write: bool) -> int:
    """Every ticker, full 10y window, splits.csv rebuilt from scratch."""
    failures: list[tuple[str, str]] = []
    split_failures: list[tuple[str, str]] = []
    results = _fetch_many(syms, history, since, workers, failures, split_failures)
    split_frames: list[pd.DataFrame] = []
    written = _write_per_ticker(results, write, split_frames)
    all_splits = _merge_splits(pd.DataFrame(columns=["symbol", *SPLIT_COLUMNS]), split_frames)
    if write:
        # One long file, header-only when nothing split — its presence
        # tells the loader the overlay was fetched, not merely absent.
        all_splits.to_csv(DATA_DIR / SPLITS_NAME, index=False)
    verb = "written" if write else "fetched (dry-run, no --write)"
    for sym, err in failures:
        print(f"  FAIL {sym}: {err}")
    for sym, err in split_failures:
        print(f"  FAIL splits {sym}: {err}")
    # summary last — the app's failure display keeps only the output tail
    print(f"full sweep: {len(syms)} tickers, {written} {verb}, "
          f"{len(failures)} failed; {len(all_splits)} split rows "
          f"({len(split_failures)} split fetches failed)")
    # Return 1 when every ticker failed — a systemic signal (bad key,
    # network down) should not look like success to the caller.
    if failures and len(failures) == len(syms):
        return 1
    return 0


def _delta_run(syms: list[str], history: dict, since: date, meta: dict, *,
               workers: int, write: bool) -> "int | None":
    """Market-wide delta since the last run (minus the overlap), merged into
    the per-ticker files and splits.csv; tickers with no file yet get the
    per-ticker full fetch. Returns None when the delta fetch itself fails —
    the caller falls back to the full sweep."""
    try:
        last_run = date.fromisoformat(str(meta.get("last_run_asof", "")))
    except ValueError:
        return None
    delta_since = last_run - timedelta(days=DELTA_OVERLAP_DAYS)
    try:
        delta = fetch_dividends_since(delta_since)
        sdelta = fetch_splits_since(delta_since)
    except Exception as exc:
        print(f"  FAIL delta fetch since {delta_since}: {exc!r} -- falling back to a full sweep")
        return None
    universe = {s.upper() for s in syms}
    failures: list[tuple[str, str]] = []
    split_failures: list[tuple[str, str]] = []
    split_frames: list[pd.DataFrame] = []
    # 1) tickers without a file (new holding / benchmark): full per-ticker fetch
    missing = [s for s in syms if not (DATA_DIR / f"dividends_{s.lower()}.csv").exists()]
    if missing:
        results = _fetch_many(missing, history, since, workers, failures, split_failures)
        _write_per_ticker(results, write, split_frames)
    # 2) dividend delta -> merge into the files it touches
    touched = added = 0
    if not delta.empty and "ticker" in delta.columns:
        tick = delta["ticker"].astype(str).str.upper()
        ours = delta[tick.isin(universe)]
        for sym, g in ours.groupby(tick[tick.isin(universe)]):
            p = DATA_DIR / f"dividends_{str(sym).lower()}.csv"
            existing = _read_div_file(p)
            merged = _merge_dividend_rows(existing, g.drop(columns=["ticker"], errors="ignore")
                                          .assign(ticker=str(sym)))
            # Write only when the content changed — most delta rows are
            # declared-future events already on file; rewriting identical
            # files churns mtimes and needlessly invalidates the AI caches.
            new_txt = merged.to_csv(index=False).replace("\r\n", "\n")
            old_txt = (p.read_text(encoding="utf-8").replace("\r\n", "\n")
                       if p.exists() else "")
            if new_txt == old_txt:
                continue
            touched += 1
            added += len(merged) - len(existing)
            if write:
                merged.to_csv(p, index=False)    # same writer as the sweep (platform newlines)
    # 3) splits delta -> merge into splits.csv
    existing_splits = (pd.read_csv(DATA_DIR / SPLITS_NAME)
                       if (DATA_DIR / SPLITS_NAME).exists()
                       else pd.DataFrame(columns=["symbol", *SPLIT_COLUMNS]))
    if not sdelta.empty and "ticker" in sdelta.columns:
        ours_s = sdelta[sdelta["ticker"].astype(str).str.upper().isin(universe)]
        if not ours_s.empty:
            split_frames.append(ours_s.rename(columns={"ticker": "symbol"}))
    merged_splits = _merge_splits(existing_splits, split_frames)
    n_split_new = len(merged_splits) - len(existing_splits)
    if write:
        merged_splits.to_csv(DATA_DIR / SPLITS_NAME, index=False)
    for sym, err in failures:
        print(f"  FAIL {sym}: {err}")
    for sym, err in split_failures:
        print(f"  FAIL splits {sym}: {err}")
    print(f"incremental since {delta_since}: {added} dividend rows across "
          f"{touched} tickers, {n_split_new} new split rows; {len(missing)} new "
          f"tickers full-fetched ({len(failures)} failed)"
          + ("" if write else " [dry-run, no --write]"))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write the CSV. Without this flag, runs as a smoke test only.")
    ap.add_argument("--holdings", action="store_true",
                    help="fetch every listed ticker in the latest "
                         "positions month (per-ticker CSVs)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel fetch threads for --holdings")
    ap.add_argument("--full", action="store_true",
                    help="--holdings: force the full per-ticker sweep instead "
                         "of the market-wide delta since the last run")
    ap.add_argument("ticker", nargs="?", default="SPY")
    args = ap.parse_args(argv)

    if args.holdings:
        # Held names ∪ every column of the close matrices (benchmarks,
        # proxies, renamed-ticker priors) — the total-return adjustment
        # needs a dividend file per column, not per holding.
        syms = sorted(set(collect_dividend_universe()) | set(collect_price_universe(DATA_DIR)))
        if not syms:
            print("no dividend universe (positions.csv missing/empty)")
            return 1
        history = _ticker_history()
        today = _today()
        # Same 10y window as single-ticker mode ON PURPOSE:
        # dividends_spy.csv doubles as the benchmark TR build's input.
        since = today - timedelta(days=MAX_LOOKBACK_DAYS)
        meta = _read_meta(DATA_DIR)
        DATA_DIR.mkdir(exist_ok=True)
        if not _needs_full(meta, today, args.full):
            rc = _delta_run(syms, history, since, meta,
                            workers=args.workers, write=args.write)
            if rc is not None:
                if args.write:
                    _write_meta(DATA_DIR, "incremental", today, meta)
                return rc
        rc = _full_sweep(syms, history, since, workers=args.workers, write=args.write)
        if args.write and rc == 0:
            _write_meta(DATA_DIR, "full", today, meta)
        return rc

    ticker = args.ticker.upper()
    since = date.today() - timedelta(days=MAX_LOOKBACK_DAYS)

    print(f"[INFO] dividends for {ticker} since {since}")
    df = fetch_dividends(ticker, since)

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"dividends_{ticker.lower()}.csv"
    if args.write:
        df.to_csv(out, index=False)
        print(f"[OK] {len(df)} dividends "
              f"({df['ex_dividend_date'].min()} -> {df['ex_dividend_date'].max()})")
        print(f"[OK] wrote {out}")
    else:
        print(f"[DRY] {len(df)} dividends "
              f"({df['ex_dividend_date'].min()} -> {df['ex_dividend_date'].max()})")
        print(f"[DRY] would write {out}  (use --write to persist)")
    print(f"\nsum cash per share over window: ${df['cash_amount'].sum():.4f}")
    cols = [c for c in ("ex_dividend_date", "pay_date", "cash_amount", "frequency", "dividend_type") if c in df.columns]
    print(f"\nfirst 3:\n{df[cols].head(3).to_string(index=False)}")
    print(f"\nlast 3:\n{df[cols].tail(3).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
