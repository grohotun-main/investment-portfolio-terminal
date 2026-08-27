"""
Fetch and prepare history for a hypothetical candidate ticker used by the
What-if Risk tab.

DESIGN: companion to fetch_daily_prices.py for one-off historical fetches.
  - The held-universe fetcher serves positions.csv + macro overlay only.
    Candidate tickers being evaluated for addition are NOT in that
    universe, so the What-if tab fetches on demand.
  - Cache: data/whatif_cache/<TICKER>.csv (gitignored under data/). TTL
    of 7 days keeps repeated runs in a session instant while the
    candidate's last bar stays reasonably current.
  - Splice: optional. When the candidate has shorter history than the
    portfolio (e.g., PDBC inception Nov 2014), a user-specified proxy
    ticker back-fills pre-inception dates. Math mirrors
    splice_sgov_with_bil in risk_metrics — proxy rebased to candidate's
    first observation, pre-inception proxy levels concatenated onto
    candidate's verbatim history.
  - All functions pure; the engine module wires them up to the existing
    risk math.
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "data" / "whatif_cache"
DEFAULT_LOOKBACK_YEARS = 10  # Polygon Stock Developer ceiling
CACHE_TTL_DAYS = 7


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.upper().replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.csv"


def _cache_is_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).days < ttl_days


def _read_cache(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return pd.Series(dtype=float, name="close")
    return df.set_index("date")["close"].sort_index().astype(float)


def _write_cache(series: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = series.reset_index()
    df.columns = ["date", "close"]
    df.to_csv(path, index=False)


def _total_return_series(series: pd.Series, actions_provider, start: date,
                         end: date) -> pd.Series:
    """Apply the candidate's distributions + splits (from ``actions_provider``)
    to its close series; a provider failure leaves the series price-only
    (graceful degradation, like a stale cache)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from total_return import total_return_adjust
    name = series.name
    try:
        div_rows, split_rows = actions_provider(str(name), start, end)
    except Exception:
        return series
    if not div_rows:
        return series
    dist = pd.DataFrame(div_rows)
    dist = pd.DataFrame({
        "symbol": str(name),
        "ex_date": pd.to_datetime(dist["ex_dividend_date"], errors="coerce"),
        "cash_amount": pd.to_numeric(dist["cash_amount"], errors="coerce"),
    }).dropna()
    splits = None
    if split_rows:
        sp = pd.DataFrame(split_rows)
        splits = pd.DataFrame({
            "symbol": str(name),
            "execution_date": pd.to_datetime(sp["execution_date"], errors="coerce"),
            "ratio": pd.to_numeric(sp["split_to"], errors="coerce")
                     / pd.to_numeric(sp["split_from"], errors="coerce"),
        }).dropna()
    frame = series.to_frame(name=str(name))
    adjusted = total_return_adjust(frame, dist, splits)
    return adjusted[str(name)].rename(name)


def _polygon_actions_provider() -> Callable[[str, date, date],
                                            tuple[list[dict], list[dict]]]:
    """Production corporate-actions provider: Polygon dividends + splits via
    the fetch_dividends helpers (lazy import — keeps the module importable
    without API keys)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_dividends import fetch_dividends, fetch_splits

    def provider(ticker: str, start: date, end: date) -> tuple[list[dict], list[dict]]:
        divs = fetch_dividends(ticker, start, allow_empty=True)
        splits = fetch_splits(ticker, start, allow_empty=True)
        return (divs.to_dict("records") if not divs.empty else [],
                splits.to_dict("records") if not splits.empty else [])
    return provider


def _polygon_bars_provider() -> Callable[[str, date, date], list[dict]]:
    """Build the production Polygon bars provider.

    Lazy-imports the fetcher + _config so the rest of the module is
    test-importable without API keys.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_daily_prices import fetch_daily_history, TICKER_ALIASES
    from _config import get_massive_key, get_massive_base
    key = get_massive_key()
    base = get_massive_base()

    def provider(ticker: str, start: date, end: date) -> list[dict]:
        poly_sym = TICKER_ALIASES.get(ticker, ticker)
        res = fetch_daily_history(poly_sym, start, end, key, base)
        if "error" in res:
            return []
        return res.get("bars") or []
    return provider


def fetch_candidate_history(
    ticker: str,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    cache_dir: Path | None = None,
    bars_provider: Callable[[str, date, date], list[dict]] | None = None,
    actions_provider: Callable[[str, date, date],
                               tuple[list[dict], list[dict]]] | None = None,
) -> pd.Series:
    """Fetch daily-close history for a candidate ticker — TOTAL-RETURN
    closes (distributions reinvested at the ex-date, split-scaled, last
    level = real close; parsers/total_return.py) so a candidate is never
    the one price-only series inside the optimizer.

    ``actions_provider`` (ticker, start, end) -> (dividend rows
    ``{ex_dividend_date, cash_amount}``, split rows ``{execution_date,
    split_from, split_to}``) is the corporate-actions seam: None with the
    production bars provider → Polygon; None with an injected
    ``bars_provider`` (tests) → price-only, no network.

    Returns a date-indexed, ascending Series of split-adjusted closes.
    Empty Series if the fetch fails AND no stale cache exists.

    Args:
      ticker: candidate (e.g. "PDBC"). Class-B-share aliasing happens
        inside the production provider.
      lookback_years: history window. Default 10 (Polygon Stock
        Developer ceiling).
      cache_dir: per-ticker CSV directory. Default data/whatif_cache/.
      bars_provider: dependency-injection seam for tests. Callable
        (ticker, start, end) -> list of {"date": str, "close": float}.
        None → use the production Polygon provider.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = _cache_path(ticker, cache_dir)
    if _cache_is_fresh(path):
        return _read_cache(path).rename(ticker.upper())

    end = date.today()
    start = end - timedelta(days=lookback_years * 365 + 30)
    if bars_provider is None:
        bars_provider = _polygon_bars_provider()
        if actions_provider is None:
            actions_provider = _polygon_actions_provider()
    bars = bars_provider(ticker.upper(), start, end)
    if not bars:
        if path.exists():
            # Stale cache better than nothing — graceful degradation.
            return _read_cache(path).rename(ticker.upper())
        return pd.Series(dtype=float, name=ticker.upper())
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"])
    series = (df.set_index("date")["close"]
                .sort_index()
                .astype(float)
                .rename(ticker.upper()))
    if actions_provider is not None:
        series = _total_return_series(series, actions_provider, start, end)
    _write_cache(series, path)
    return series


def splice_with_proxy(
    candidate: pd.Series,
    proxy: pd.Series,
) -> pd.Series:
    """Extend `candidate` back in time using `proxy`.

    Math mirrors splice_sgov_with_bil: candidate observations are kept
    verbatim; pre-inception proxy levels are rebased so the join is
    continuous at candidate's first observation. The resulting Series'
    daily returns equal proxy's actual daily returns in the pre-inception
    window — exactly what the downstream return computation needs.

    Empty-input behavior:
      - empty candidate → empty Series
      - empty proxy, or proxy has no pre-candidate data → candidate as-is
      - non-positive / non-finite anchor → candidate as-is
    """
    if candidate is None or candidate.empty:
        return pd.Series(dtype=float, name=getattr(candidate, "name", None))
    cand = candidate.dropna().sort_index()
    name = cand.name
    if cand.empty:
        return pd.Series(dtype=float, name=name)
    if proxy is None or proxy.empty:
        return cand.copy()
    prx = proxy.dropna().sort_index()
    if prx.empty:
        return cand.copy()
    cand_first = cand.index.min()
    prx_pre = prx[prx.index < cand_first]
    if prx_pre.empty:
        return cand.copy()
    anchor_block = prx[prx.index <= cand_first]
    if anchor_block.empty:
        return cand.copy()
    anchor = float(anchor_block.iloc[-1])
    if not np.isfinite(anchor) or anchor <= 0:
        return cand.copy()
    scale = float(cand.iloc[0]) / anchor
    prx_rescaled = prx_pre * scale
    spliced = pd.concat([prx_rescaled, cand]).sort_index()
    # Defensive: if proxy has a bar on candidate's first day, prefer
    # candidate's actual level over the rescaled proxy estimate.
    spliced = spliced[~spliced.index.duplicated(keep="last")]
    return spliced.rename(name)


def build_augmented_price_matrix(
    daily_prices: pd.DataFrame,
    candidate_ticker: str,
    candidate_series: pd.Series,
) -> pd.DataFrame:
    """Return daily_prices with a new candidate column joined onto its date index.

    Pre-existing columns are untouched. If `candidate_ticker` is already
    a column it is overwritten — useful when the same ticker appears in
    the held universe but the What-if tab wants the spliced long-history
    version.

    No forward-fill: candidate dates with no observation produce NaN, and
    downstream return computation drops those rows.
    """
    out = daily_prices.copy() if not daily_prices.empty else pd.DataFrame()
    if candidate_series is None or candidate_series.empty:
        if candidate_ticker not in out.columns:
            out[candidate_ticker] = np.nan
        return out
    s = candidate_series.copy()
    s.index = pd.to_datetime(s.index)
    if out.empty:
        out = pd.DataFrame({candidate_ticker: s})
        out.index.name = "date"
        return out
    out.index = pd.to_datetime(out.index)
    out[candidate_ticker] = s.reindex(out.index)
    return out


def build_multi_augmented_price_matrix(
    daily_prices: pd.DataFrame,
    series_by_ticker: dict[str, pd.Series],
) -> pd.DataFrame:
    """Splice several candidate columns onto `daily_prices` at once.

    Folds ``build_augmented_price_matrix`` once per (ticker, series). Order of
    the resulting new columns follows ``series_by_ticker`` insertion order.
    An empty mapping returns ``daily_prices`` unchanged (a copy is made only
    when there is at least one candidate to splice).
    """
    if not series_by_ticker:
        return daily_prices
    out = daily_prices
    for ticker, series in series_by_ticker.items():
        out = build_augmented_price_matrix(out, ticker, series)
    return out
