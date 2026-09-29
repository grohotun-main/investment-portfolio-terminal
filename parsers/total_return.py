"""Total-return close matrices — distributions reinvested at the ex-date close.

The daily close matrix both UIs build (``pivot(close)`` of ``daily_prices.csv`` /
``long_history_prices.csv``, then the ``TICKER_HISTORY`` splice) is PRICE-only:
a $10/sh return of capital reads as a −49 % day, a monthly-distribution ETF
saws down every month, and every risk / contribution / factor / simulation /
attribution number built from that matrix understates payers while the
portfolio's TWR and the SPY/AGG benchmark legs are total return. This module
turns the matrix into total-return closes using the SAME rule as the SPY leg
(``build_benchmark_total_return.build_tr``): a distribution with cash ``D``
going ex on a bar with close ``P_u`` is reinvested at that close, so the day's
return is ``(P_u + D) / P_{u-1} - 1``.

Shape contract: the adjusted series is rebased so its LAST valid level equals
the actual close — ``adj_t = P_t × Π_{u>t} 1/(1 + D_u/P_u)`` — so the one
level consumer (the Options tab's spot via ``.iloc[-1]``) is untouched while
``pct_change()`` yields total returns. NaN bars stay NaN.

Data: ``data/dividends_<ticker>.csv`` (Polygon ``/v3/reference/dividends``,
written by ``fetch_dividends.py --holdings``; prior-symbol history already
merged) and ``data/splits.csv`` (Polygon ``/v3/reference/splits``). Polygon's
dividend ``cash_amount`` is AS-DECLARED while its bars are split-adjusted, so
a pre-split dividend must be divided by every later split ratio (AVGO paid
$5.25 before its 10:1 and $0.53 after) — ``split_scale_cash``.

No dividend files (the committed synthetic fixture) ⇒ ``apply_total_return``
is an exact no-op, so every golden stays byte-identical.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DIST_GLOB = "dividends_*.csv"
SPLITS_NAME = "splits.csv"
_DIST_COLS = ["symbol", "ex_date", "cash_amount", "pay_date"]
_SPLIT_COLS = ["symbol", "execution_date", "ratio"]
# FINRA due-bill rule: a distribution worth >= 25 % of the security goes ex
# the business day AFTER the payable date, not before the record date. Data
# vendors publish a provisional ex-date for such specials (VISN's $5 on a
# $11.50 stock listed ex 2026-08-17 while the price never moved; its April
# $10 correctly read ex = pay + 1). A large payout must therefore be
# CONFIRMED by the price — at least CONFIRM_FRACTION of its size as a drop —
# on the listed ex-date or on the first bar after the pay date; otherwise it
# is skipped (and retried on the next refresh, when the real ex-date has a
# bar). Ordinary dividends are far below price noise and apply unconditionally.
LARGE_DISTRIBUTION_PCT = 0.25
CONFIRM_FRACTION = 0.5


def _symbol_of(path: Path) -> str:
    return path.stem[len("dividends_"):].strip().upper()


def covered_symbols(data_dir) -> set[str]:
    """Symbols that HAVE a dividend file (a header-only file is a known
    non-payer — covered, just with nothing to apply)."""
    d = Path(data_dir)
    if not d.is_dir():
        return set()
    return {_symbol_of(p) for p in d.glob(DIST_GLOB)}


def load_distributions(data_dir) -> pd.DataFrame:
    """Long frame ``[symbol, ex_date, cash_amount]`` from every
    ``dividends_<ticker>.csv`` in ``data_dir``. Unparseable rows drop; exact
    duplicate rows (same id) collapse; distinct distributions sharing an
    ex-date (a regular + a special) both survive and sum downstream."""
    d = Path(data_dir)
    frames: list[pd.DataFrame] = []
    if d.is_dir():
        for p in sorted(d.glob(DIST_GLOB)):
            try:
                df = pd.read_csv(p)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if df.empty or not {"ex_dividend_date", "cash_amount"}.issubset(df.columns):
                continue
            out = pd.DataFrame({
                "symbol": _symbol_of(p),
                "ex_date": pd.to_datetime(df["ex_dividend_date"], errors="coerce"),
                "cash_amount": pd.to_numeric(df["cash_amount"], errors="coerce"),
                "pay_date": (pd.to_datetime(df["pay_date"], errors="coerce")
                             if "pay_date" in df.columns else pd.NaT),
                "_id": df["id"].astype(str) if "id" in df.columns else "",
            })
            frames.append(out)
    if not frames:
        return pd.DataFrame(columns=_DIST_COLS)
    dist = pd.concat(frames, ignore_index=True).dropna(subset=["ex_date", "cash_amount"])
    dist = dist[dist["cash_amount"] > 0]
    dist = dist.drop_duplicates(subset=["symbol", "ex_date", "cash_amount", "_id"])
    dist = (dist.groupby(["symbol", "ex_date"], as_index=False)
                .agg(cash_amount=("cash_amount", "sum"), pay_date=("pay_date", "max"))
                .sort_values(["symbol", "ex_date"]).reset_index(drop=True))
    return dist[_DIST_COLS]


def _with_pay_date(dist: pd.DataFrame) -> pd.DataFrame:
    """Callers may pass [symbol, ex_date, cash_amount] frames; add a NaT
    pay_date column so the due-bill check has something to read."""
    if "pay_date" not in dist.columns:
        dist = dist.copy()
        dist["pay_date"] = pd.NaT
    return dist


def load_splits(data_dir) -> pd.DataFrame:
    """Long frame ``[symbol, execution_date, ratio]`` (``ratio = split_to /
    split_from``, 10.0 for a 1→10) from ``data_dir/splits.csv``; an empty
    frame with the same columns when the file is absent or header-only."""
    p = Path(data_dir) / SPLITS_NAME
    empty = pd.DataFrame(columns=_SPLIT_COLS)
    if not p.exists():
        return empty
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, OSError):
        return empty
    if df.empty or not {"symbol", "execution_date", "split_from",
                        "split_to"}.issubset(df.columns):
        return empty
    frm = pd.to_numeric(df["split_from"], errors="coerce")
    to = pd.to_numeric(df["split_to"], errors="coerce")
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str).str.strip().str.upper(),
        "execution_date": pd.to_datetime(df["execution_date"], errors="coerce"),
        "ratio": to / frm,
    }).dropna()
    out = out[out["ratio"] > 0]
    return out.sort_values(["symbol", "execution_date"]).reset_index(drop=True)[_SPLIT_COLS]


def split_scale_cash(distributions: pd.DataFrame,
                     splits: pd.DataFrame | None) -> pd.DataFrame:
    """Divide each distribution's cash by the product of the symbol's split
    ratios executed AFTER its ex-date (declared amounts → the split-adjusted
    share basis the price bars use). A reverse split (ratio < 1) scales up."""
    if distributions is None:
        return pd.DataFrame(columns=_DIST_COLS)
    out = distributions.copy()
    if out.empty or splits is None or splits.empty:
        return out
    by_sym = {s: g for s, g in splits.groupby("symbol")}
    scale = []
    for sym, ex in zip(out["symbol"], out["ex_date"]):
        g = by_sym.get(str(sym).upper())
        if g is None:
            scale.append(1.0)
            continue
        later = g[g["execution_date"] > ex]
        scale.append(float(later["ratio"].prod()) if not later.empty else 1.0)
    out["cash_amount"] = out["cash_amount"] / pd.Series(scale, index=out.index, dtype=float)
    return out


def _confirmed_bar(valid: pd.Series, pos: int, cash: float) -> bool:
    """True when bar ``pos`` shows at least CONFIRM_FRACTION of ``cash`` as a
    drop from the prior bar — the price evidence a large payout went ex."""
    prev_close = float(valid.iloc[pos - 1])
    close = float(valid.iloc[pos])
    if not (prev_close > 0 and close > 0) or cash >= prev_close:
        return False
    return close <= prev_close * (1.0 - CONFIRM_FRACTION * cash / prev_close)


def _adjust_column(p: pd.Series, dist: pd.DataFrame,
                   skipped: list) -> tuple[pd.Series, bool]:
    """One symbol: ``dist`` has ex_date / cash_amount / pay_date rows.
    Returns (series, applied). ``skipped`` collects
    (symbol, bar, cash, prev_close, reason) for payouts not applied."""
    valid = p.dropna()
    if len(valid) < 2:
        return p, False
    idx = valid.index
    n = len(idx)
    # Distributions landing on the same bar (several rows, or an ex-date on a
    # non-trading day rolling onto the next bar) accumulate first.
    cash_by_pos: dict[int, float] = {}
    for row in dist.itertuples(index=False):
        cash = float(row.cash_amount)
        pos = int(idx.searchsorted(pd.Timestamp(row.ex_date)))   # first bar >= ex-date
        if pos <= 0 or pos >= n:
            continue                                              # no prior bar / after the window
        prev_close = float(valid.iloc[pos - 1])
        if not (cash > 0 and prev_close > 0) or cash >= prev_close:
            # A payout at or above the prior close is impossible — bad data,
            # never a return. Skip it and say so.
            skipped.append((str(p.name), idx[pos], cash, prev_close, "impossible"))
            continue
        if cash / prev_close >= LARGE_DISTRIBUTION_PCT:
            # Due-bill territory: the listed ex-date may be provisional. Take
            # the first candidate bar the price confirms — the listed ex-date,
            # else the first bar after the pay date — or skip for now.
            candidates = [pos]
            pay = getattr(row, "pay_date", pd.NaT)
            if pd.notna(pay):
                pos2 = int(idx.searchsorted(pd.Timestamp(pay), side="right"))  # first bar > pay date
                if 0 < pos2 < n and pos2 != pos:
                    candidates.append(pos2)
            chosen = next((c for c in candidates if _confirmed_bar(valid, c, cash)), None)
            if chosen is None:
                skipped.append((str(p.name), idx[pos], cash, prev_close, "unconfirmed_large"))
                continue
            pos = chosen
        cash_by_pos[pos] = cash_by_pos.get(pos, 0.0) + cash
    factors = pd.Series(1.0, index=idx)
    applied = False
    for pos, cash in sorted(cash_by_pos.items()):
        close = float(valid.iloc[pos])
        if not close > 0:
            skipped.append((str(p.name), idx[pos], cash, float(valid.iloc[pos - 1]), "impossible"))
            continue
        factors.iloc[pos] = 1.0 / (1.0 + cash / close)   # reinvest at the ex-date close
        applied = True
    if not applied:
        return p, False
    # Π f_u over bars strictly AFTER t: reverse cumprod, shifted one bar back.
    back = factors[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out = p.astype(float)                  # an int-typed close column can't hold the scaled values
    out.loc[idx] = valid.to_numpy(dtype=float) * back.to_numpy(dtype=float)
    return out, True


def total_return_adjust(prices: pd.DataFrame, distributions: pd.DataFrame,
                        splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adjust a ``date × symbol`` close matrix in place-of (a copy is
    returned). ``attrs["total_return"]`` = ``{"adjusted": [symbols with ≥1
    applied distribution], "skipped": [(symbol, bar, cash, prev_close)]}``.
    Symbols with no distributions are returned unchanged."""
    out = prices.copy()
    info: dict = {"adjusted": [], "skipped": []}
    if out.empty or distributions is None or distributions.empty:
        out.attrs["total_return"] = info
        return out
    dist = _with_pay_date(split_scale_cash(distributions, splits))
    dist = dist.dropna(subset=["ex_date", "cash_amount"])
    dist = dist[dist["cash_amount"] > 0]
    groups = {str(s).upper(): g.sort_values("ex_date")[["ex_date", "cash_amount", "pay_date"]]
              for s, g in dist.groupby("symbol")}
    skipped: list = []
    for col in out.columns:
        d = groups.get(str(col).upper())
        if d is None or d.empty:
            continue
        adj, applied = _adjust_column(out[col], d, skipped)
        if applied:
            out[col] = adj
            info["adjusted"].append(str(col))
    info["skipped"] = skipped
    out.attrs["total_return"] = info
    return out


def apply_total_return(prices: pd.DataFrame, data_dir) -> pd.DataFrame:
    """The loaders' entry point: read ``data_dir``'s dividend + split files
    and adjust ``prices``. Adds coverage to ``attrs["total_return"]``:
    ``covered`` (symbols with a dividend file) and ``uncovered`` (matrix
    symbols without one — they stay price-only)."""
    if prices is None or prices.empty:
        return prices
    covered = covered_symbols(data_dir)
    dist = load_distributions(data_dir)
    out = (total_return_adjust(prices, dist, load_splits(data_dir))
           if not dist.empty else prices.copy())
    info = dict(out.attrs.get("total_return") or {"adjusted": [], "skipped": []})
    info["covered"] = len(covered)
    info["uncovered"] = sorted(str(c) for c in out.columns
                               if str(c).upper() not in covered)
    out.attrs["total_return"] = info
    return out
