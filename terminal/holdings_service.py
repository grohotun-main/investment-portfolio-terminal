"""Pure data seam for the MERIDIAN Terminal Holdings tab.

Re-expresses the Streamlit Holdings tab's inline aggregations (``app.py``'s
``_render_holdings_body`` and its loaders) as a pure, importable module that
reuses the ``parsers/`` engine and ``config_local`` maps. It imports **no
Streamlit**: it reads CSVs from a data dir (the same ``APP_DATA_DIR`` the
Streamlit app and the test suite point at) and does the per-snapshot transforms
purely.

Scope of this module (data seam part 1):
  * ``load_frames`` — read + enrich the same frames ``app.py.load_data`` builds.
  * ``_current_snap`` — the latest-as-of marked snapshot slice.
  * ``_alloc_by_class`` / ``_alloc_by_account`` — the two allocation groupbys.
  * ``_top_holdings`` / ``_positions_table`` — the holdings tables.

The KPI tape (TWR / IRR / vol / beta) and the ``build_holdings_view`` assembly
are a separate, later task and are intentionally not built here.

Every value returned by the builders is JSON-native (numpy scalars cast to
float/int, Timestamps to strings) so a later step can ``json.dumps`` the result
without leaks.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

import config_local as cfg
import theme

# parsers/ is a flat module directory on sys.path (the server inserts it; the
# test suite mirrors that). Import the pure engine functions directly.
from twr_aggregate import (portfolio_twr_headline, recompute_portfolio_twr,
                           slice_canonical_twr)
from build_benchmark_total_return import build_blended_tr
from compute_twr import compute_account_irr, compute_portfolio_irr, classify_irr
from mark_to_market import mark_to_market
from monthly_normalize import (
    monthly_normalize,
    month_canonical_dates,
    slice_as_of_month,
)
from asset_reclass import reclass_asset
from synthesize_interim_positions import synthesize_interim_positions
from total_return import apply_total_return
from interim_stub import (InterimStub, chain, compute_interim_stub, stub_flows,
                          to_date_cagr, to_date_span)
from compute_twr import link_returns, annualize  # noqa: F401  (parity import)
from risk_metrics import (compute_beta, extend_sgov_with_bil_panel,
                          splice_ticker_history)
from risk_bundle import (
    build_snapshot_weights,
    daily_portfolio_returns,
    spy_daily_returns,
)

# This codebase ships no demo-broker sidecar: the sidecar overlay is an
# identity pass and the test-broker label set is empty (every broker present
# in the data is treated as real).
TEST_BROKER_LABELS: frozenset = frozenset()


def _overlay_demo(df, _sidecar_path, *_args, **_kwargs):
    return df


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config-driven maps — same source of truth app.py uses.
# --------------------------------------------------------------------------- #
CLASS_COLORS = theme.CLASS_COLORS
BROKER_COLORS = theme.BROKER_COLORS

# Mirror of app.py's CLASS_LABELS (app.py:220-231). Kept verbatim so the two
# UIs label asset classes identically.
CLASS_LABELS = {
    "equity_etf":          "ETFs",
    "equity_stock":        "Individual Stocks",
    "tax_loss_harvesting": "Tax Loss Harvesting",
    "fixed_income":        "Fixed Income",
    "gold":                "Gold",
    "cash":                "Cash & sweep",
    "option_put":          "Options (puts)",
    "option_call":         "Options (calls)",
    "mutual_fund":         "Mutual funds",
    "other":               "Other",
}

ACCOUNT_BUCKETS_FIXED     = cfg.ACCOUNT_BUCKETS_FIXED
FIDELITY_TOD_ACCOUNTS     = cfg.FIDELITY_TOD_ACCOUNTS
FIDELITY_CORE_ETF_SYMBOLS = cfg.FIDELITY_CORE_ETF_SYMBOLS
ACCOUNT_DISPLAY           = cfg.ACCOUNT_DISPLAY
ETF_TICKER_CLASS          = getattr(cfg, "ETF_TICKER_CLASS", {})
_NAME_TO_TICKER           = cfg.NAME_TO_TICKER

_MINUS = "−"  # U+2212 MINUS SIGN — the signed-display minus app.py uses


# --------------------------------------------------------------------------- #
# Enrichment helpers — ported verbatim from app.py so the columns line up
# byte-for-byte with what the Holdings body (and the parity test) expect.
# --------------------------------------------------------------------------- #
def _lookup_ticker(desc: str) -> str | None:
    """Mirror app.py:305-312 — verbal-name prefix → ticker fallback."""
    if not isinstance(desc, str):
        return None
    up = desc.upper()
    for name, tkr in _NAME_TO_TICKER.items():
        if up.startswith(name):
            return tkr
    return None


def _clean_description(s: object) -> str:
    """Mirror app.py:315-325 — strip statement-noise suffixes from a description."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s*-?-?\s*STATEMENT SUMMARY.*$", "", s, flags=re.I)
    s = re.sub(r"\s*Dividend\s*Reinvested.*$", "", s, flags=re.I)
    s = re.sub(r"\s*Dividend\s*--?.*$", "", s, flags=re.I)
    s = re.sub(r"\s*Dividend\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*--?\s*ISIN.*$", "", s, flags=re.I)
    s = re.sub(r"\s+EST\s+YIELD.*$", "", s, flags=re.I)
    s = re.sub(r"\s*--?\s*$", "", s)
    return s.strip(" -")


def _display_symbol(row: pd.Series) -> str:
    """Mirror app.py:437-449 — public ticker, else a UST/CUSIP synthetic label."""
    sym = row.get("symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip()
    desc = row.get("description_clean", "") or ""
    m = re.match(r"UNITED STATES TREAS(?:URY)?[^\d]*(\d{2})/(\d{2})/(\d{4})", desc, re.I)
    if m:
        mm, dd, yyyy = m.groups()
        return f"UST {mm}/{dd}/{yyyy[-2:]}"
    cusip = row.get("cusip")
    if isinstance(cusip, str) and cusip.strip():
        return f"CUSIP {cusip.strip()}"
    return "(unknown)"


def _account_bucket(account_id: str, symbol: str | float) -> str:
    """Mirror app.py:258-273 — fixed bucket, TOD-split, or ``Other (id)``."""
    fixed = ACCOUNT_BUCKETS_FIXED.get(account_id)
    if fixed is not None:
        return fixed
    if account_id in FIDELITY_TOD_ACCOUNTS:
        sym = symbol.strip().upper() if isinstance(symbol, str) else ""
        if sym in FIDELITY_CORE_ETF_SYMBOLS:
            return "Fidelity Core ETFs"
        return "Fidelity Individual Stocks"
    return f"Other ({account_id})"


def _reclass_asset(account_id: str, symbol: str | float, asset_class: str) -> str:
    """Mirror app.py:292-295 — symbol-keyed asset_class overrides."""
    return reclass_asset(account_id, symbol, asset_class,
                         tlh_account_id=cfg.TLH_ACCOUNT_ID,
                         etf_class=ETF_TICKER_CLASS)


# --------------------------------------------------------------------------- #
# collapse_buckets / formatters — ported from app.py (977-1045).
# --------------------------------------------------------------------------- #
def collapse_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Roll TLH and Treasury Ladder positions into one synthetic row each.

    Ported verbatim from app.py:977-1027. The aggregated rows carry
    ``price`/`quantity` = NaN by design — the positions-table ``price_asof``
    rule labels those ``"(aggregated)"``.
    """
    if df.empty:
        return df
    is_tlh = df["asset_class"] == "tax_loss_harvesting"
    is_tlad = df["bucket"] == "JPM Treasury Ladder"
    keep = df[~is_tlh & ~is_tlad].copy()

    extras: list[dict] = []
    if is_tlh.any():
        sub = df[is_tlh]
        extras.append({
            "display_symbol":     "Tax Loss Harvesting (TLH)",
            "description_clean":  f"{len(sub)} positions aggregated",
            "symbol":             "",
            "bucket":             "JPM Tax Loss Harvesting",
            "asset_class":        "tax_loss_harvesting",
            "asset_class_label":  CLASS_LABELS["tax_loss_harvesting"],
            "broker":             "jpm",
            "account_id":         cfg.TLH_ACCOUNT_ID,
            "quantity":           np.nan,
            "price":              np.nan,
            "market_value":       float(sub["market_value"].sum()),
            "cost_basis":         float(sub["cost_basis"].sum()),
            "unrealized_gl":      float(sub["unrealized_gl"].sum()),
        })
    if is_tlad.any():
        sub = df[is_tlad]
        extras.append({
            "display_symbol":     "Treasury Ladder",
            "description_clean":  f"{len(sub)} treasuries aggregated",
            "symbol":             "",
            "bucket":             "JPM Treasury Ladder",
            "asset_class":        "fixed_income",
            "asset_class_label":  CLASS_LABELS["fixed_income"],
            "broker":             "jpm",
            "account_id":         cfg.TREASURY_LADDER_ACCOUNT_ID,
            "quantity":           np.nan,
            "price":              np.nan,
            "market_value":       float(sub["market_value"].sum()),
            "cost_basis":         float(sub["cost_basis"].sum()),
            "unrealized_gl":      float(sub["unrealized_gl"].sum()),
        })
    if extras:
        return pd.concat([keep, pd.DataFrame(extras)], ignore_index=True)
    return keep


def fmt_money(v: float, decimals: int = 0) -> str:
    """Mirror app.py:1033-1038."""
    if pd.isna(v):
        return "-"
    if v < 0:
        return f"-${-v:,.{decimals}f}"
    return f"${v:,.{decimals}f}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    """Mirror app.py:1041-1044."""
    if pd.isna(v):
        return "-"
    return f"{v:.{decimals}f}%"


# --------------------------------------------------------------------------- #
# Frames + loader.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Frames:
    """The enriched, immutable bundle the Holdings view is computed from.

    ``positions`` / ``positions_monthly`` carry the same enrichment columns
    ``app.py.load_data`` produces (the Holdings body and the parity test depend
    on them). The non-positions frames are loaded for the later KPI-tape task
    and may be empty if their CSV is absent.
    """
    positions: pd.DataFrame
    positions_monthly: pd.DataFrame
    transactions: pd.DataFrame
    prices_latest: pd.DataFrame
    prices_as_of: pd.Timestamp | None
    daily_prices: pd.DataFrame
    spy_tr: pd.DataFrame
    twr_portfolio: pd.DataFrame
    irr_table: pd.DataFrame
    available_dates: list  # list[str] "YYYY-MM-DD", newest first
    data_dir: str = ""     # the data dir this bundle was loaded from (for the
                           # SYNTHETIC badge — "synth" in the path → fixture)
    twr_account: pd.DataFrame = field(default_factory=pd.DataFrame)   # per-account TWR (twr_monthly.csv, demo-overlaid)
    summaries: pd.DataFrame = field(default_factory=pd.DataFrame)     # per-account reported totals (summaries.csv)
    broker_scope: tuple[str, ...] | None = None  # None = canonical view; else
                                                 # sorted broker DISPLAY labels
                                                 # (pill/popover casing), NOT
                                                 # raw broker-column values
    agg_tr: pd.DataFrame = field(default_factory=pd.DataFrame)  # AGG TR (60/40 bond leg); empty when the file is absent


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV if present, else an empty frame (mirrors app.py's try/except
    loaders — a missing transient file degrades gracefully)."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _load_prices_latest(data_dir: Path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Mirror app.py.load_prices_latest (452-476): (df, prices_as_of) where
    prices_as_of is the max as_of_date among ``status == 'ok'`` rows; add a
    ``staleness_days`` column."""
    path = data_dir / "prices_latest.csv"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    ok = df[df["status"] == "ok"]
    as_of = pd.Timestamp(ok["as_of_date"].max()) if not ok.empty else None
    if as_of is not None:
        df["staleness_days"] = (as_of - df["as_of_date"]).dt.days
    else:
        df["staleness_days"] = pd.NA
    return df, as_of


def _load_daily_prices(data_dir: Path) -> pd.DataFrame:
    """Mirror app.py.load_daily_prices (489-503): pivoted close matrix with the
    TICKER_HISTORY corporate-action splice applied — a renamed ticker's prior
    symbol's history is grafted under the current ticker before the rename's
    effective date (a transparent no-op for tickers without an entry).

    The splice is REQUIRED, not optional: the Factor tab's per-holding betas +
    monthly-vs-daily cross-check push this matrix through daily regressions, and
    the Risk / Performance synthesis reads it too, so a renamed holding must
    carry its full spliced history to stay 1:1 with Streamlit (which always reads
    the spliced matrix). (Earlier slices' snapshot builders only read the latest
    column, so the splice was a no-op there — but it is not for daily history.)

    Then the TOTAL-RETURN adjustment (``total_return.apply_total_return``):
    distributions from ``dividends_<ticker>.csv`` reinvested at the ex-date
    close, split-scaled, rebased so the last level stays the real close —
    every engine that ``pct_change``s this matrix compares like with like
    against the NAV-based TWR and the SPY/AGG TR legs (spec
    2026-08-22-total-return-basis). A data dir without dividend files is an
    exact no-op."""
    path = data_dir / "daily_prices.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    raw = df.pivot(index="date", columns="symbol", values="close").sort_index()
    if raw.empty:
        return raw
    hist = getattr(cfg, "TICKER_HISTORY", {})
    spliced = splice_ticker_history(raw, hist) if hist else raw
    return apply_total_return(spliced, data_dir)


def _load_long_history_prices(data_dir: Path) -> pd.DataFrame:
    """long_history_prices.csv → pivoted close matrix (mirrors
    riskcontrib_regime._load_long_history: same pivot, same rename splice).
    Feeds the SGOV↔BIL pre-inception bridge; empty frame when absent."""
    path = data_dir / "long_history_prices.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = (pd.read_csv(path, parse_dates=["date"])
           .pivot(index="date", columns="symbol", values="close").sort_index())
    if raw.empty:
        return raw
    spliced = splice_ticker_history(raw, getattr(cfg, "TICKER_HISTORY", {}))
    return apply_total_return(spliced, data_dir)   # same total-return basis as daily


def _load_twr_account(data_dir: Path) -> pd.DataFrame:
    """Per-account TWR (mirror app.py.load_twr's twr_monthly.csv branch,
    723-730): read WITHOUT parse_dates, overlay the data/demo/ sidecar (like
    the other Frames frames) on the raw-string frame, THEN coerce
    month/statement_date on the union so real+demo rows share dtypes."""
    acct = _read_csv(data_dir / "twr_monthly.csv")
    acct = _overlay_demo(acct, data_dir / "demo" / "twr_monthly.csv")
    if not acct.empty:
        acct["month"] = pd.PeriodIndex(acct["month"], freq="M")
        acct["statement_date"] = pd.to_datetime(acct["statement_date"])
    return acct


def _enrich_positions(positions: pd.DataFrame,
                      prices_latest: pd.DataFrame) -> pd.DataFrame:
    """Apply the same enrichment ``app.py.load_data`` does (649-713).

    Marks the latest snapshot to live prices, then derives ``description_clean``,
    ``symbol`` backfill, ``display_symbol``, ``asset_class`` (reclass) +
    ``asset_class_label``, ``bucket`` and ``account_display``. ``broker`` is a
    raw positions.csv column and is left as-is.
    """
    df = positions.copy()
    if not prices_latest.empty:
        df = mark_to_market(df, prices_latest)

    df["description_clean"] = df["description"].apply(_clean_description)

    # Backfill APPLE INC then the verbal-name → ticker map (app.py:694-698).
    mask = df["symbol"].isna() & df["description_clean"].str.startswith("APPLE INC")
    df.loc[mask, "symbol"] = "AAPL"
    mask_missing = df["symbol"].isna()
    df.loc[mask_missing, "symbol"] = (
        df.loc[mask_missing, "description_clean"].apply(_lookup_ticker)
    )

    df["display_symbol"] = df.apply(_display_symbol, axis=1)
    df["asset_class"] = df.apply(
        lambda r: _reclass_asset(r["account_id"], r["symbol"], r["asset_class"]),
        axis=1,
    )
    df["asset_class_label"] = (
        df["asset_class"].map(CLASS_LABELS).fillna(df["asset_class"])
    )
    df["bucket"] = df.apply(
        lambda r: _account_bucket(r["account_id"], r["symbol"]), axis=1,
    )
    df["account_display"] = (
        df["account_id"].map(ACCOUNT_DISPLAY).fillna(df["account_id"])
    )
    return df


# --------------------------------------------------------------------------- #
# Cross-request Frames cache. ``_build_frames`` re-reads + re-enriches the whole
# book (~2s) and every terminal route calls it per request; the AI-box rollout
# added a SECOND concurrent per-tab request, doubling that cost. Cache the
# immutable Frames per data_dir, keyed on a stat-signature of the dir's CSVs so
# any ingest / refresh / month-close (all rewrite CSVs) invalidates it. Frames
# is frozen and every terminal consumer treats it read-only, so the instance is
# SHARED, not copied. Config (TICKER_HISTORY) changes need a server restart
# regardless, which clears this in-process cache.
# --------------------------------------------------------------------------- #
_FRAMES_CACHE: dict = {}
_FRAMES_CACHE_LOCK = threading.Lock()


def _frames_signature(data_dir: Path) -> tuple:
    """(name, mtime_ns, size) for every CSV ``_build_frames`` may read
    (top-level + ``demo/``), sorted. A deliberate superset of the exact file
    list: over-invalidation only costs a needless rebuild, under-invalidation
    would serve stale data."""
    paths = (sorted(data_dir.glob("*.csv"))
             + sorted((data_dir / "demo").glob("*.csv")))
    sig = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:                    # racing delete — skip, don't crash
            continue
        sig.append((p.name, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _clear_frames_cache() -> None:
    """Drop all cached Frames (tests; also a manual invalidation hook)."""
    with _FRAMES_CACHE_LOCK:
        _FRAMES_CACHE.clear()


def load_frames(data_dir: str | Path) -> Frames:
    """Cached wrapper over ``_build_frames`` — see the cache note above. The
    returned Frames is shared and frozen; never mutate it in place."""
    data_dir = Path(data_dir)
    key = str(data_dir)
    sig = _frames_signature(data_dir)
    with _FRAMES_CACHE_LOCK:
        hit = _FRAMES_CACHE.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        # Build under the lock: concurrent misses for one dir coalesce onto a
        # single load (the AI-box path + tab-body path hit the same data_dir);
        # a raising build is not stored, so a bad data dir never caches.
        frames = _build_frames(data_dir)
        _FRAMES_CACHE[key] = (sig, frames)
        return frames


def _build_frames(data_dir: str | Path) -> Frames:
    """Read + enrich the same frames the Streamlit Holdings path uses.

    Mirrors ``app.py.load_data`` (649-713) + ``monthly_normalize_marked`` (842):
    reads the canonical CSVs from ``data_dir``, unions any mid-month
    ``transactions_interim.csv`` and rolls holdings forward via
    ``synthesize_interim_positions`` (so the snapshot advances to
    ``max(interim settlement_date)``), enriches ``positions``, then builds
    ``positions_monthly = mark_to_market(monthly_normalize(positions),
    prices_latest)`` so carried-forward (``_filled``) rows are re-marked.

    The demo-broker sidecar overlay this loader once supported is an identity
    no-op here (``_overlay_demo`` above). ``summaries`` (reported per-account
    totals) is narrowed by broker in ``apply_global_filters`` like every other
    per-account sidecar (the choke-point rule: a service must read these off
    ``Frames``, never re-read the raw CSV from ``data_dir``, or a partial
    broker selection would leak whole-book dollars). The interim roll-forward
    keeps the snapshot/NAV current on live ``data/``; when
    ``transactions_interim.csv`` is absent (the committed fixture has none)
    it is a clean no-op.
    """
    data_dir = Path(data_dir)

    positions = pd.read_csv(data_dir / "positions.csv",
                            parse_dates=["statement_date"])
    positions = _overlay_demo(positions, data_dir / "demo" / "positions.csv",
                              parse_dates=["statement_date"])
    transactions = _read_csv(data_dir / "transactions.csv",
                             parse_dates=["settlement_date"])
    transactions = _overlay_demo(transactions, data_dir / "demo" / "transactions.csv",
                                 parse_dates=["settlement_date"])

    # Union mid-month interim activity if present, then roll the latest
    # statement positions forward with it so holdings, allocations, and the
    # snapshot date reflect a unified date = max(interim settlement_date).
    # Mirrors app.py.load_data (664-682) exactly (same args/order); a missing
    # interim file is a clean no-op (the committed fixture has none).
    interim_path = data_dir / "transactions_interim.csv"
    if interim_path.exists():
        interim = pd.read_csv(interim_path, parse_dates=["settlement_date"])
        if not interim.empty:
            transactions = (
                pd.concat([transactions, interim], ignore_index=True)
                if not transactions.empty else interim
            )
            rolled = synthesize_interim_positions(positions, interim)
            if not rolled.empty:
                positions = pd.concat([positions, rolled], ignore_index=True)

    prices_latest, prices_as_of = _load_prices_latest(data_dir)
    daily_prices = _load_daily_prices(data_dir)
    # SGOV pre-inception bridge (TK 2026-07-17): model SGOV as T-bills where
    # its own bars are missing — BIL-spliced from the long-history file.
    # Fills only NaN gaps on existing panel dates; no-op when the file or
    # columns are absent (the synth fixture).
    daily_prices = extend_sgov_with_bil_panel(
        daily_prices, _load_long_history_prices(data_dir))
    spy_tr = _read_csv(data_dir / "benchmark_spy_tr.csv", parse_dates=["date"])
    agg_tr = _read_csv(data_dir / "benchmark_agg_tr.csv", parse_dates=["date"])
    twr_portfolio = _read_csv(data_dir / "twr_portfolio.csv")
    irr_table = _read_csv(data_dir / "irr_per_account.csv")
    twr_account = _load_twr_account(data_dir)
    # Reported per-account statement totals (mirror app.py.load_summaries,
    # 698-706). No demo sidecar — data/demo/ has no summaries.csv.
    summaries = _read_csv(data_dir / "summaries.csv", parse_dates=["statement_date"])

    positions = _enrich_positions(positions, prices_latest)

    # positions_monthly: normalize to one snapshot per (account, month), then
    # re-mark so carried-forward rows are repriced (app.py.monthly_normalize_marked).
    positions_monthly = monthly_normalize(positions)
    if not prices_latest.empty:
        positions_monthly = mark_to_market(positions_monthly, prices_latest)

    available_dates = [
        pd.Timestamp(d).strftime("%Y-%m-%d") for d in month_canonical_dates(positions)
    ]

    return Frames(
        positions=positions,
        positions_monthly=positions_monthly,
        transactions=transactions,
        prices_latest=prices_latest,
        prices_as_of=prices_as_of,
        daily_prices=daily_prices,
        spy_tr=spy_tr,
        agg_tr=agg_tr,
        twr_portfolio=twr_portfolio,
        irr_table=irr_table,
        available_dates=available_dates,
        data_dir=str(data_dir),
        twr_account=twr_account,
        summaries=summaries,
    )


# --------------------------------------------------------------------------- #
# Snapshot + builders.
# --------------------------------------------------------------------------- #
def interim_stub(frames: Frames) -> "InterimStub | None":
    """The provisional period from the last statement month-end to
    ``prices_as_of`` (spec 2026-08-22), derived from THIS frames object so
    broker/history scoping is inherited. Start NAV is the statement-basis
    snapshot (``market_value_stmt`` where the latest snapshot was re-marked in
    place, else ``market_value``); end NAV is the latest snapshot marked to
    live; flows are the interim FLOW_TYPES rows settled after the statement.
    None when there is no return series, no price date, or the price date is
    not past the last statement (the synth fixture; a freshly closed month)."""
    tp = frames.twr_portfolio
    if (tp is None or tp.empty or "statement_date" not in tp.columns
            or frames.prices_as_of is None):
        return None
    start = pd.Timestamp(pd.to_datetime(tp["statement_date"]).max())
    end = pd.Timestamp(frames.prices_as_of).normalize()
    if pd.isna(start) or end <= start:
        return None
    snap0 = _current_snap(frames, start.strftime("%Y-%m-%d"))
    snap1 = _current_snap(frames)
    if snap0.empty or snap1.empty:
        return None
    # mark_to_market re-marks only rows dated at the frame's latest date: a
    # dual-date frontier month with no interim roll-forward would leave the
    # earlier-dated broker at statement value and dilute the stub while the
    # caption says "marked to live" — no stub rather than a wrong one (review
    # 2026-08-22; the interim roll-forward unifies the dates, so pulling
    # interim transactions restores it).
    if snap1["statement_date"].nunique() > 1:
        return None
    if "market_value_stmt" in snap0.columns:
        mv0 = snap0["market_value_stmt"].where(snap0["market_value_stmt"].notna(),
                                               snap0["market_value"])
    else:
        mv0 = snap0["market_value"]
    return compute_interim_stub(start, end, float(mv0.sum()),
                                float(snap1["market_value"].sum()),
                                stub_flows(frames.transactions, after=start))


def _current_snap(frames: Frames, as_of: str | None = None) -> pd.DataFrame:
    """The marked monthly snapshot for ``as_of`` (default: latest available).

    Mirrors app.py:2260 — ``slice_as_of_month`` resolves by calendar month so a
    dual-date month carries both brokers' rows.
    """
    when = as_of or (frames.available_dates[0] if frames.available_dates else None)
    return slice_as_of_month(frames.positions_monthly, when)


def alloc_by_class_raw(snap: pd.DataFrame) -> pd.DataFrame:
    """Raw allocation-by-asset-class groupby (the shared compute for BOTH UIs):
    market_value summed per asset_class, descending. app.py adds Plotly
    label/colour; the view formatter below adds JSON label/colour/pct/value."""
    return (snap.groupby("asset_class")["market_value"].sum()
            .sort_values(ascending=False).reset_index())


# Cycled for asset classes missing from CLASS_COLORS — a shared "#888" fallback
# would collapse every unmapped class onto one indistinguishable grey ring.
# Hues deliberately absent from CLASS_COLORS (borrowed from risk_service's
# concentration-donut palette).
_FALLBACK_CLASS_COLORS = ["#818CF8", "#2FD79A", "#F472B6", "#FBBF24", "#A78BFA"]


def _alloc_by_class(snap: pd.DataFrame) -> dict:
    """Allocation-by-asset-class view dict (mirror app.py:2348-2367).
    Thin formatter over ``alloc_by_class_raw`` — the numbers live there now."""
    by_class = alloc_by_class_raw(snap)
    total = float(by_class["market_value"].sum())

    slices = []
    n_unmapped = 0
    for _, row in by_class.iterrows():
        cls = str(row["asset_class"])
        mv = float(row["market_value"])
        color = CLASS_COLORS.get(cls)
        if color is None:
            color = _FALLBACK_CLASS_COLORS[n_unmapped % len(_FALLBACK_CLASS_COLORS)]
            n_unmapped += 1
        slices.append({
            "label": CLASS_LABELS.get(cls, cls),
            "class": cls,
            "color": color,
            "pct": (mv / total * 100.0) if total else 0.0,
            "value": fmt_money(mv),
        })
    return {
        "total_label": f"${total / 1e6:.2f}M",
        "n": len(slices),
        "slices": slices,
    }


def alloc_by_account_raw(snap: pd.DataFrame) -> pd.DataFrame:
    """Raw allocation-by-account groupby (the shared compute for BOTH UIs):
    market_value summed per [broker, bucket], descending. This is the
    canonical (descending) order; app.py re-sorts ascending itself for its
    horizontal bar chart."""
    return (snap.groupby(["broker", "bucket"])["market_value"].sum()
            .sort_values(ascending=False).reset_index())


def _alloc_by_account(snap: pd.DataFrame) -> dict:
    """Allocation-by-account view dict (mirror app.py:2372-2376).
    Thin formatter over ``alloc_by_account_raw`` — the numbers live there now.

    Per row: the bucket label, broker, colour (theme.BROKER_COLORS), pct of
    the bucket total, and ``bar`` = pct normalised so the largest row == 100.0
    (the mockup's bars are normalised to the top row, not the raw pct).
    """
    by_acct = alloc_by_account_raw(snap)
    total = float(by_acct["market_value"].sum())
    pcts = [(float(mv) / total * 100.0) if total else 0.0
            for mv in by_acct["market_value"]]
    max_pct = max(pcts) if pcts else 0.0

    rows = []
    for (_, row), pct in zip(by_acct.iterrows(), pcts):
        broker = str(row["broker"])
        color = BROKER_COLORS.get(broker)
        rows.append({
            "label": str(row["bucket"]),
            "broker": broker,
            "color": color,
            "pct": pct,
            "bar": (pct / max_pct * 100.0) if max_pct else 0.0,
        })
    return {"n": len(rows), "rows": rows}


def top_holdings_raw(snap: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Raw top-N holdings groupby (the shared compute for BOTH UIs) — a
    SUPERSET aggregation: ``collapse_buckets`` then groupby
    ``[display_symbol, asset_class]`` summing market_value (``mv``), desc,
    head(top_n); plus ``qty`` (summed quantity), ``n_accts`` (nunique
    account_id), and ``desc`` (shortest description_clean) — columns app.py
    needs for its All-holdings drill-in that the view formatter below ignores.
    """
    collapsed = collapse_buckets(snap)
    if collapsed.empty:
        return collapsed

    return (collapsed.groupby(["display_symbol", "asset_class"])
            .agg(mv=("market_value", "sum"),
                 qty=("quantity", "sum"),
                 n_accts=("account_id", "nunique"),
                 desc=("description_clean",
                       lambda s: min(s, key=len) if len(s) else ""))
            .reset_index().sort_values("mv", ascending=False).head(top_n))


def _top_holdings(snap: pd.DataFrame, total: float, top_n: int = 15) -> dict:
    """Top-N holdings view dict (mirror app.py:2393-2401).
    Thin formatter over ``top_holdings_raw`` — the numbers live there now.
    Uses only the raw frame's ``mv`` column; per row pct (mv/total*100),
    colour, and ``bar`` = pct / max_pct * 100.
    """
    top = top_holdings_raw(snap, top_n=top_n)
    if top.empty:
        return {"top_n": top_n, "rows": []}

    pcts = [(float(mv) / total * 100.0) if total else 0.0 for mv in top["mv"]]
    max_pct = max(pcts) if pcts else 0.0

    rows = []
    for (_, row), pct in zip(top.iterrows(), pcts):
        cls = str(row["asset_class"])
        rows.append({
            "symbol": str(row["display_symbol"]),
            "class": cls,
            "color": CLASS_COLORS.get(cls, "#888"),
            "pct": pct,
            "bar": (pct / max_pct * 100.0) if max_pct else 0.0,
        })
    return {"top_n": top_n, "rows": rows}


def _signed_money(v: float) -> str:
    """Signed dollar string: ``+$1,234`` / ``−$1,234`` (U+2212 minus, matching
    app.py's MERIDIAN chip convention). NaN → ``"—"``."""
    if pd.isna(v):
        return "—"
    if v < 0:
        return f"{_MINUS}${-v:,.0f}"
    return f"+${v:,.0f}"


def _dir(v: float) -> str:
    """``up`` / ``down`` / ``flat`` for a signed value (NaN → ``flat``)."""
    if pd.isna(v) or v == 0:
        return "flat"
    return "up" if v > 0 else "down"


def positions_table_raw(snap: pd.DataFrame, frames: Frames, total: float,
                        as_of_ts) -> pd.DataFrame:
    """Raw All-holdings positions frame (the shared compute for BOTH UIs):
    ``collapse_buckets`` then sort by ``market_value`` desc, with the derived
    numeric columns ``weight_pct`` / ``unrealized_pct`` (NaN when
    cost_basis<=0) and the per-row price-freshness label ``price_asof`` /
    ``price_stmt`` (mirrors ``_price_asof``, app.py:2466-2486) — everything
    app.py's All-holdings block computes BEFORE its own search filter and
    ``st.dataframe`` formatting.

    Computed over the FULL (unfiltered-by-search) universe: every derivation
    here is row-wise over ``total``/``as_of_ts``/``frames`` and does not
    depend on which rows are present, so it is safe to filter (by search
    substring) AFTER calling this — which is exactly what both UIs do. This
    fn takes no ``search`` kwarg by design; search is a formatter/UI concern
    layered on top of these raw numbers, not part of the shared computation.
    """
    as_of_ts = pd.Timestamp(as_of_ts)
    latest_snapshot_ts = (pd.Timestamp(frames.available_dates[0])
                          if frames.available_dates else as_of_ts)
    prices_latest_df = frames.prices_latest
    prices_as_of = frames.prices_as_of

    table = collapse_buckets(snap)
    if table.empty:
        return table

    table = table.sort_values("market_value", ascending=False).copy()
    table["weight_pct"] = (table["market_value"] / total * 100.0
                           if total else 0.0)
    table["unrealized_pct"] = np.where(
        table["cost_basis"].fillna(0) > 0,
        (table["market_value"] - table["cost_basis"])
        / table["cost_basis"] * 100.0,
        np.nan,
    )

    # Per-row price-freshness inputs (app.py:2454-2464).
    is_latest = (as_of_ts == latest_snapshot_ts)
    live_dates: dict[str, pd.Timestamp] = {}
    cash_syms: set[str] = set()
    if is_latest and prices_latest_df is not None and not prices_latest_df.empty:
        ok_rows = prices_latest_df[prices_latest_df["status"] == "ok"]
        live_dates = dict(zip(ok_rows["symbol"], ok_rows["as_of_date"]))
        cash_syms = set(prices_latest_df.loc[
            prices_latest_df["status"] == "cash_fixed_1", "symbol"
        ])
    _STMT_SUFFIX = "  (stmt)"
    stmt_label = as_of_ts.strftime("%b %d, %Y") + _STMT_SUFFIX

    def _price_asof(row) -> tuple[str, bool]:
        """Return (label, is_stmt). Mirrors app.py:2466-2486."""
        if pd.isna(row.get("price")):
            return "(aggregated)", False
        if not is_latest:
            return as_of_ts.strftime("%b %d, %Y"), False
        sym = row.get("symbol")
        if isinstance(sym, str) and sym.strip():
            if sym in live_dates:
                return pd.Timestamp(live_dates[sym]).strftime("%b %d, %Y"), False
            if sym in cash_syms:
                return ((pd.Timestamp(prices_as_of).strftime("%b %d, %Y")
                         if prices_as_of is not None
                         else as_of_ts.strftime("%b %d, %Y")), False)
        if pd.notna(row.get("_filled")) and bool(row.get("_filled")) \
                and pd.notna(row.get("_as_of_date")):
            return (pd.Timestamp(row["_as_of_date"]).strftime("%b %d, %Y")
                    + _STMT_SUFFIX), True
        return stmt_label, True

    labels, is_stmts = [], []
    for _, r in table.iterrows():
        label, is_stmt = _price_asof(r)
        labels.append(label)
        is_stmts.append(is_stmt)
    table["price_asof"] = labels
    table["price_stmt"] = is_stmts
    return table


def _positions_table(snap: pd.DataFrame, frames: Frames, total: float,
                     as_of_ts, search: str = "") -> dict:
    """The All-holdings positions table view dict (mirror app.py:2430-2535).
    Thin formatter over ``positions_table_raw`` — the numbers live there now.
    Applies the optional substring search filter (on
    ``display_symbol|description_clean|bucket``) on top of the raw frame, then
    JSON string-formats each row (money/qty/pct strings, signed G/L, footnote).
    """
    as_of_ts = pd.Timestamp(as_of_ts)
    total_underlying = int(len(snap))

    table = positions_table_raw(snap, frames, total, as_of_ts)
    is_latest = (as_of_ts == (pd.Timestamp(frames.available_dates[0])
                              if frames.available_dates else as_of_ts))

    if not table.empty and search.strip():
        q = search.lower().strip()
        table = table[table.apply(
            lambda r: q in str(r["display_symbol"]).lower()
                      or q in str(r["description_clean"]).lower()
                      or q in str(r["bucket"]).lower(),
            axis=1,
        )]

    rows: list[dict] = []
    for _, r in table.iterrows():
        label, is_stmt = r["price_asof"], bool(r["price_stmt"])
        is_agg = pd.isna(r.get("price"))
        qty = r.get("quantity")
        price = r.get("price")
        mv = float(r["market_value"]) if pd.notna(r["market_value"]) else float("nan")
        cb = r.get("cost_basis")
        ugl = r.get("unrealized_gl")
        ugl_val = float(ugl) if pd.notna(ugl) else float("nan")
        ugl_pct = r.get("unrealized_pct")
        ugl_pct_val = float(ugl_pct) if pd.notna(ugl_pct) else float("nan")
        weight = float(r["weight_pct"]) if pd.notna(r.get("weight_pct")) else 0.0

        rows.append({
            "symbol": str(r["display_symbol"]),
            "description": str(r.get("description_clean", "")),
            "account": str(r.get("bucket", "")),
            "asset_class": str(r.get("asset_class", "")),
            "class_label": str(r.get("asset_class_label", "")),
            "qty": "—" if (is_agg or pd.isna(qty)) else f"{float(qty):,.4f}",
            "price": "—" if (is_agg or pd.isna(price)) else f"${float(price):,.2f}",
            "price_asof": label,
            "price_stmt": bool(is_stmt),
            "market_value": fmt_money(mv),
            "cost_basis": (fmt_money(float(cb)) if pd.notna(cb) else "—"),
            "ugl": _signed_money(ugl_val),
            "ugl_dir": _dir(ugl_val),
            "ugl_pct": (f"{ugl_pct_val:.1f}%" if not pd.isna(ugl_pct_val) else "—"),
            "ugl_pct_dir": _dir(ugl_pct_val),
            "weight_pct": weight,
        })

    n_stale = sum(1 for row in rows if row["price_asof"].endswith("  (stmt)")) if is_latest else 0
    stale_note = (f" {n_stale} row(s) using statement-date price (no live mark)."
                  if n_stale else "")
    footnote = (
        f"Showing {len(rows)} rows of {total_underlying} underlying positions "
        f"on {as_of_ts.strftime('%b %d, %Y')} — Tax Loss Harvesting and "
        f"Treasury Ladder aggregated." + stale_note
    )

    return {
        "shown": len(rows),
        "total_underlying": total_underlying,
        "footnote": footnote,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# KPI tape — mirror app.py render_chrome (9670-9709) + _portfolio_headline.
# --------------------------------------------------------------------------- #
def _spct(v: float) -> str:
    """Signed-percent string mirroring app.py's ``_spct`` (9694-9695).

    ``"—"`` for NaN; otherwise a ``"+"`` prefix for strictly positive values
    (negatives already carry the minus from ``f"{v:.1f}"``)."""
    return "—" if pd.isna(v) else (("+" if v > 0 else "") + f"{v:.1f}%")


def _pnl_color(v: float) -> str | None:
    """``"gain"`` / ``"loss"`` / ``None`` for a signed value (mirrors
    theme.pnl_color's sign split; NaN → ``None`` = neutral)."""
    if pd.isna(v):
        return None
    return "gain" if v >= 0 else "loss"


BENCHMARKS = {"spy": "SPY (S&P 500 TR)", "60_40": "60/40 SPY/AGG blend"}

# Short benchmark labels (table headers, facts descriptors). Single source of
# truth shared by benchmark_service, ai_service, and the AI portfolio route.
BENCH_SHORT = {"spy": "SPY", "60_40": "60/40"}


def resolve_benchmark(requested: str, broker_scope: tuple[str, ...] | None) -> str:
    """Map the request sentinel 'auto' to a concrete benchmark id from the
    broker scope: a JPM-only book resembles a 60/40 mix, so it defaults to the
    blend; every other scope defaults to SPY. Explicit ids pass through."""
    if requested in BENCHMARKS:
        return requested
    return "60_40" if broker_scope == ("JPM",) else "spy"


def _bench_tr_series(frames: Frames, benchmark: str = "spy") -> pd.Series:
    """Forward-filled benchmark TR value series (mirrors app.py.load_benchmark_tr):
    index by date, ffill across non-trading days so the daily-return alignment
    matches the Streamlit Risk bundle. `benchmark="60_40"` blends spy_tr + agg_tr
    (60/40, daily constant-mix); returns an empty Series if a needed frame is
    missing (so the caller can fall back / show an unavailable state)."""
    if benchmark == "60_40":
        tr = build_blended_tr([(0.6, frames.spy_tr), (0.4, frames.agg_tr)])
    else:
        tr = frames.spy_tr
    if tr is None or tr.empty or "date" not in tr.columns or "tr_value" not in tr.columns:
        return pd.Series(dtype=float)
    s = tr.sort_values("date").set_index("date")["tr_value"]
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full).ffill()


def _agg_available(frames: Frames) -> bool:
    """Whether the 60/40 bond leg (AGG TR) data is present. Absence is the only
    real reason the blend degrades to SPY (the file is gitignored, so a fresh
    clone/CI lacks it). Cheaper than building the full blended series just to
    test ``.empty`` — callers that don't otherwise need the series use this to
    decide the SPY fallback without a throwaway ``build_blended_tr``."""
    return frames.agg_tr is not None and not frames.agg_tr.empty


def _snapshot_weights(snap: pd.DataFrame,
                      daily_prices: pd.DataFrame) -> pd.Series:
    """Normalized symbol weights for daily synthesis — thin wrapper over the
    shared ``risk_bundle.build_snapshot_weights`` fold (single source for
    BOTH UIs) for a pre-filtered snapshot: ``None`` filters = no Account /
    Asset-class slice. Returns weights only (the fold also yields the
    pre-normalization $ market values)."""
    return build_snapshot_weights(snap, None, None, daily_prices)[0]


def _daily_return_series(frames: Frames) -> tuple[pd.Series, pd.Series]:
    """(port_rets, spy_rets) daily-return series for the UNFILTERED book —
    thin wrappers over the shared risk_bundle pipeline (per-month honest
    weights → synthesis; SPY from the TR series with price-only fallback).
    Empty Series when daily prices / benchmark are unavailable."""
    daily_prices = frames.daily_prices
    return (daily_portfolio_returns(frames.positions, daily_prices,
                                    None, None),
            spy_daily_returns(daily_prices, _bench_tr_series(frames)))


def _kpi_tape(frames: Frames) -> list[dict]:
    """The six KPI-tape cells, in mockup order, mirroring app.py.render_chrome
    (9670-9709) + ``_portfolio_headline`` (9656-9667) for the UNFILTERED book.

    Each cell is ``{"key","label","value","color","sub"}`` (``color`` ∈
    ``{None,"gain","loss"}``); cell 0 additionally carries a ``"chip"`` =
    the vs-prior-month percent chip. Every value is JSON-native.

    IRR mirrors render_chrome's CANONICAL path exactly: it reads the
    ``PORTFOLIO`` row from ``irr_table`` and shows ``"—"`` when absent. The live
    ``compute_portfolio_irr`` recompute in app.py runs ONLY under a history-start
    cutoff (``start_cutoff_ts is not None``); the default tape — which this seam
    mirrors and the parity gate checks — does NOT recompute, so reading the CSV
    row keeps the two presentation paths in lockstep. Non-real broker scopes are
    the deliberate exception: ``apply_global_filters`` appends a SCOPED
    ``compute_portfolio_irr(..., scoped=True)`` recompute row to ``irr_table``
    for those selections, so this seam reads that recomputed row rather than
    the canonical CSV one — a divergence from app.py by design (spec
    2026-08-07).
    """
    port = frames.twr_portfolio
    snap = _current_snap(frames)
    nav = float(snap["market_value"].sum()) if not snap.empty else float("nan")

    # vs-prior-month chip on cell 0 — % change of the snapshot total vs the
    # prior available month's snapshot total (app.py Holdings-card convention).
    chip = None
    if frames.available_dates:
        prior = [d for d in frames.available_dates
                 if pd.Timestamp(d) < pd.Timestamp(frames.available_dates[0])]
        if prior:
            prior_snap = _current_snap(frames, prior[0])
            prior_total = (float(prior_snap["market_value"].sum())
                           if not prior_snap.empty else 0.0)
            if prior_total > 0 and pd.notna(nav):
                pct = (nav / prior_total - 1.0) * 100.0
                chip = {"text": f"{'+' if pct >= 0 else ''}{pct:.1f}%",
                        "dir": "up" if pct >= 0 else "down"}

    # cum / annualized / max-drawdown / months — the shared headline seam
    # (render_chrome single-source; same formula, so the tape numbers are
    # byte-identical).
    _h = portfolio_twr_headline(port)
    cum, ann, mdd, n = _h.cum, _h.ann, _h.mdd, _h.n
    start_m, mdd_m = _h.start_month, _h.mdd_month
    # Provisional stub period (spec 2026-08-22): cum chains the period, ann
    # is day-count based, both subs say so; mdd/months stay statement-based.
    cum_sub, ann_sub = f"since {start_m}", f"{n} months"
    stub = interim_stub(frames)
    if stub is not None and "return_pct" in port.columns:
        _origin, days = to_date_span(port, stub)
        cum = chain(cum, stub.return_pct)
        ann = to_date_cagr(cum, days)
        cum_sub += f" · to {stub.end_date:%Y-%m-%d} · prov."
        ann_sub += f" + {stub.days}d · prov."

    # IRR — read the PORTFOLIO row (see docstring); NaN → "—".
    irr = float("nan")
    it = frames.irr_table
    if it is not None and not it.empty and "account_id" in it.columns:
        row = it[it["account_id"] == "PORTFOLIO"]
        if len(row):
            irr = float(row.iloc[0]["irr"])

    # Vol · 252d + β from the daily synthesis.
    port_rets, spy_rets = _daily_return_series(frames)
    vol = (float(port_rets.tail(252).std(ddof=1) * np.sqrt(252) * 100.0)
           if len(port_rets) >= 5 else float("nan"))
    beta = compute_beta(port_rets, spy_rets, window=252)

    cell_pv: dict = {
        "key": "portfolio_value", "label": "Portfolio value",
        "value": fmt_money(nav), "color": None, "sub": "marked to live",
        "chip": chip,
    }
    return [
        cell_pv,
        {"key": "cum_twr", "label": "Cumulative TWR",
         "value": _spct(cum * 100.0), "color": _pnl_color(cum),
         "sub": cum_sub},
        {"key": "annualized", "label": "Annualized TWR",
         "value": _spct(ann * 100.0), "color": None, "sub": ann_sub},
        {"key": "irr", "label": "IRR", "value": _spct(irr * 100.0),
         "color": None,
         # Canonical view: the irr_table PORTFOLIO row is whole-book (say why
         # the em-dash when absent — TK 2026-07-19). Non-real broker scopes
         # carry the SCOPED recompute apply_global_filters appended; label
         # the scope, and never claim "whole-book only" there.
         "sub": (
             (f"money-weighted · {' + '.join(frames.broker_scope)}"
              if np.isfinite(irr)
              else "money-weighted · n/a for this selection")
             if frames.broker_scope
             else ("money-weighted" if np.isfinite(irr)
                   else "money-weighted · whole-book only"))},
        {"key": "vol", "label": "Vol · 252d",
         "value": "—" if pd.isna(vol) else f"{vol:.1f}%", "color": None,
         "sub": (f"β {beta:.2f}" if np.isfinite(beta) else "β —")},
        {"key": "max_dd", "label": "Max drawdown", "value": _spct(mdd),
         "color": "loss", "sub": mdd_m},
    ]


# --------------------------------------------------------------------------- #
# Filter-option slugs + the full view assembly.
# --------------------------------------------------------------------------- #
def _slug(s: str) -> str:
    """Opaque, broker-identifier-free slug for a bucket label (security §7.3 —
    account filter ids are slugs, not raw account ids)."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_") or "x"


def _normalize_filter_ids(value: str | list[str] | None) -> list[str]:
    """Normalize a filter param (a scalar 'all'/id, or a list of ids) to a list.
    None, an empty list, or any list containing 'all' collapses to the no-filter
    sentinel ['all'] — byte-equivalent to today's unfiltered path."""
    if value is None:
        return ["all"]
    ids = [value] if isinstance(value, str) else list(value)
    if not ids or "all" in ids:
        return ["all"]
    return ids


def _filter_echo(ids: list[str]) -> str | list[str]:
    """meta.filter echo: the 'all' sentinel string when unfiltered, else the id
    list. Keeps the default (unfiltered) meta byte-identical to the pre-multi
    contract so every golden is unchanged."""
    return "all" if ids == ["all"] else ids


def _filter_meta(account: str | list[str], asset_class: str | list[str],
                 broker: str | list[str] = "all") -> dict:
    """The meta['filter'] echo dict, shared by every filtering service."""
    return {"account": _filter_echo(_normalize_filter_ids(account)),
            "asset_class": _filter_echo(_normalize_filter_ids(asset_class)),
            "broker": _filter_echo(_normalize_filter_ids(broker))}


def _resolve_ids(ids: list[str], by_id: dict) -> list[str] | None:
    """Map normalized filter ids to their underlying values (bucket labels /
    asset-class keys). None for the ['all'] sentinel (no filter); otherwise the
    resolved values, silently dropping unknown ids (real callers pass validated
    ids). Shared by _resolve_filter and build_holdings_view's inline mapping."""
    if ids == ["all"]:
        return None
    return [by_id[i] for i in ids if i in by_id]


def _account_options(snap_all: pd.DataFrame) -> tuple[list[dict], dict]:
    """[{id,label}] account filter options + an id→bucket reverse map, derived
    from the distinct buckets present (sorted, like app.py's bucket_choices)."""
    buckets = sorted(snap_all["bucket"].dropna().astype(str).unique())
    opts, by_id = [], {}
    for b in buckets:
        sid = _slug(b)
        # Disambiguate the rare slug collision so the reverse map stays 1:1.
        base, k = sid, 2
        while sid in by_id:
            sid = f"{base}_{k}"
            k += 1
        opts.append({"id": sid, "label": b})
        by_id[sid] = b
    return opts, by_id


def _broker_options(snap_all: pd.DataFrame) -> tuple[list[dict], dict]:
    """[{id,label}] broker filter options + id→broker reverse map. Real brokers
    first, demo/test brokers (TEST_BROKER_LABELS) last (mirrors app.py's
    broker_choices), slug ids like _account_options. Test options carry
    ``test: True`` (real ones stay {id,label}) so the front-end can label the
    unchecked default state with the real broker names — test brokers are
    opt-in, never part of the default."""
    vals = sorted(snap_all["broker"].dropna().astype(str).unique())
    real = [b for b in vals if b not in TEST_BROKER_LABELS]
    test = [b for b in vals if b in TEST_BROKER_LABELS]
    opts, by_id = [], {}
    for b in real + test:
        sid = _slug(b)
        base, k = sid, 2
        while sid in by_id:
            sid = f"{base}_{k}"
            k += 1
        opt = {"id": sid, "label": b}
        if b in TEST_BROKER_LABELS:
            opt["test"] = True
        opts.append(opt)
        by_id[sid] = b
    return opts, by_id


# Raw broker value -> its DISPLAY casing, mirroring app.js's `_brokerOpts`
# `nice` map / `prettyBroker` fallback (terminal/static/app.js) — the
# terminal's picker/legend already apply this client-side. `_broker_options`'s
# own `label` is deliberately left RAW (it mirrors app.py's `broker_choices`
# AND is baked verbatim into every terminal_*_golden.json via `meta.brokers`;
# recasing it there would churn every tab's golden). This is therefore a
# SEPARATE helper, used only where a raw broker value must become the prose
# form shown in a rendered string (the KPI-tape / Performance headline IRR
# sub's `frames.broker_scope`) rather than an API-facing option list.
def _broker_display_label(raw: str) -> str:
    """Prose casing for a raw broker id, derived by rule so no broker name is
    ever hardcoded: a short all-lower id (<= 3 chars) reads as an initialism
    (``"jpm"`` -> ``"JPM"``); anything else gets its first letter upper-cased
    (``"fidelity"`` -> ``"Fidelity"``), a no-op when already cased."""
    if not raw:
        return raw
    if raw.isascii() and raw.isalpha() and raw.islower() and len(raw) <= 3:
        return raw.upper()
    return raw[:1].upper() + raw[1:]


def data_brokers(frames: "Frames") -> list[str]:
    """Sorted raw broker ids present in the loaded book (test labels
    excluded). The empty/degenerate bundle yields ``[]``."""
    if frames.positions.empty or "broker" not in frames.positions.columns:
        return []
    vals = sorted(frames.positions["broker"].dropna().astype(str).unique())
    return [b for b in vals if b not in TEST_BROKER_LABELS]


def canonical_broker_label(frames: "Frames") -> str:
    """Prose label for the unfiltered book, derived from the data — a
    ``{fidelity, jpm}`` book renders ``"Fidelity + JPM"``. Every caption or
    facts block naming the canonical scope MUST use this rather than a
    literal, so the label follows whatever brokers the data actually holds.
    ``"Portfolio"`` when no broker column is loaded."""
    labels = [_broker_display_label(b) for b in data_brokers(frames)]
    return " + ".join(labels) if labels else "Portfolio"


def _history_start_cutoff(history_start: str | None) -> pd.Timestamp | None:
    """Parse a ``history_start`` option id into a cutoff Timestamp.

    ``"all"`` / ``None`` / unparseable -> ``None`` (no cutoff). ``"{year}+"``
    -> Jan 1 of that year (mirrors app.py:1517-1520)."""
    if not history_start or history_start == "all":
        return None
    try:
        return pd.Timestamp(year=int(str(history_start).rstrip("+")),
                            month=1, day=1)
    except (ValueError, TypeError):
        return None


def _history_start_options(frames: Frames) -> list[dict]:
    """[{id,label}] History-start options from the years present in positions +
    transactions (mirrors app.py:1495-1520). ``[{"all","All history"}]`` plus
    ``"{year}+"`` for every available year EXCEPT the earliest (already covered
    by "All history"). Computed from the FULL (pre-filter) frames, symmetric
    with ``_broker_options`` — the picker offers every year the book spans."""
    yrs_pos = (frames.positions["statement_date"].dt.year.dropna().astype(int)
               if not frames.positions.empty else pd.Series(dtype=int))
    yrs_tx = (frames.transactions["settlement_date"].dt.year.dropna().astype(int)
              if not frames.transactions.empty else pd.Series(dtype=int))
    years = sorted(set(yrs_pos.tolist()) | set(yrs_tx.tolist()))
    opts = [{"id": "all", "label": "All history"}]
    opts += [{"id": f"{y}+", "label": f"{y}+"} for y in years[1:]]
    return opts


def _portfolio_irr_row(pos: pd.DataFrame, txns: pd.DataFrame, *,
                       scoped: bool,
                       start_date: pd.Timestamp | None = None) -> dict | None:
    """Recompute the synthetic PORTFOLIO IRR row for the frames given, or
    None when the recompute is non-finite OR floor-banded (`classify_irr`
    "error" — the -0.9999 corruption signature must never render as a
    plausible-looking -99.99%)."""
    if pos.empty or txns.empty:
        return None
    port_irr = compute_portfolio_irr(
        pos, txns, synthetic_onboarding=cfg.SYNTHETIC_ONBOARDING,
        start_date=start_date, scoped=scoped)
    irr_val = port_irr.get("irr")
    if irr_val is None or not np.isfinite(irr_val):
        return None
    if classify_irr(float(irr_val)) == "error":
        return None
    _s, _e = port_irr["start_date"], port_irr["end_date"]
    _wm = max(1, int((_e.year - _s.year) * 12 + (_e.month - _s.month)))
    return {"account_id": "PORTFOLIO", "start_date": _s, "end_date": _e,
            "window_months": _wm, "terminal_nav": port_irr["terminal_nav"],
            "n_cashflows": port_irr["n_cashflows"],
            "total_deposits": port_irr["total_deposits"],
            "total_withdrawals": port_irr["total_withdrawals"],
            "irr": port_irr["irr"]}


def apply_global_filters(frames: Frames,
                         broker: str | list[str] | None = "all",
                         history_start: str | None = "all") -> Frames:
    """Narrow the whole book by broker (mirror app.py's ``Apply global broker
    filter`` block); also applies the History-start cutoff (mirror app.py's
    ``Apply global history-start cutoff`` block).

    ``broker`` is a scalar/list of broker option ids from ``meta.brokers``, or
    the ``'all'`` no-filter sentinel (which resolves to all REAL brokers, i.e.
    every broker not in ``TEST_BROKER_LABELS``).

    The no-op fast path (returns the SAME ``frames`` object) fires ONLY when the
    selection already covers EVERY broker present in ``frames`` — nothing to
    drop. That is true for the committed fixture (no demo brokers) and for an
    explicit all-brokers selection, but NOT for the real default: on real data
    ``load_frames`` overlays the ``data/demo/`` demo brokers into ``frames``, so
    a real-only selection must actively narrow the book to drop the demo rows
    (else the default view leaks demo accounts/NAV into the real totals).

    Otherwise the book is narrowed to the selection (``positions`` /
    ``transactions`` filtered, ``positions_monthly`` rebuilt so the snapshot /
    allocations reflect the narrowed book). This is the SINGLE choke-point for
    every other per-account/summary sidecar too — ``twr_account`` (filtered to
    the kept accounts) and ``summaries`` (filtered to the selected brokers) are
    narrowed here unconditionally, so a test-only broker selection can never
    leave real per-account dollars in either frame. ``twr_portfolio`` /
    ``irr_table`` are additionally handled by whether the selection is the
    REAL book:

      * Real selection (the real-only default, or an explicit all-real pick):
        only demo rows were dropped, so the cached ``twr_portfolio`` — the real
        portfolio's aggregate series — is KEPT verbatim. ``irr_table`` is still
        filtered to the kept accounts (plus the synthetic ``PORTFOLIO`` row),
        matching app.py's unconditional ``kept_accounts | {"PORTFOLIO"}`` filter
        (app.py:1649-1652) — a no-op unless the cached table carries a stale
        account no longer present in the narrowed book.
      * Non-real selection (demo-only, or a real subset): the cached portfolio
        twr is wrong for it (it pairs cross-account flows across the WHOLE
        book), so RECOMPUTE ``twr_portfolio`` NAV-weighted from the narrowed
        per-account ``twr_account`` via ``recompute_portfolio_twr`` — only the
        kept accounts feed it, so a demo/subset view never leaks the real book.
        ``irr_table`` drops the cached whole-book ``PORTFOLIO`` row (it is never
        a real ``account_id``, so it must be dropped explicitly rather than by
        the accounts filter) then RECOMPUTES a scoped one via
        ``_portfolio_irr_row(..., scoped=True)`` on the narrowed frames —
        appended only when finite and not floor-banded (``classify_irr`` !=
        "error"); ``frames.broker_scope`` carries the sorted broker DISPLAY
        labels (``_broker_display_label``, not the raw column values) for
        this case (``None`` on the real/canonical view).

    Unknown/unresolvable ids (a stale or malformed id) resolve to an empty
    selection, which — like an empty list or the ``'all'`` sentinel — falls
    back to the real-broker view rather than silently narrowing to nothing.

    When ``history_start`` names a year, after the broker block the book is
    truncated to >= Jan 1 of that year — IRR is recomputed FIRST on the
    pre-slice frames (``compute_*_irr`` inject the pre-cutoff NAV as a starting
    deposit) then positions/transactions/twr_account are sliced,
    ``positions_monthly`` rebuilt, and ``twr_portfolio`` recomputed. Mirrors
    app.py:1640-1704 in order.
    """
    snap = _current_snap(frames)
    _, broker_by_id = _broker_options(snap)
    # _broker_options' own `label` stays raw (golden-pinned — see
    # _broker_display_label's docstring); build the DISPLAY map separately so
    # broker_scope (prose-facing) never leaks the raw column casing.
    label_by_raw = {raw: _broker_display_label(raw) for raw in broker_by_id.values()}
    all_brokers = sorted(frames.positions["broker"].dropna().astype(str).unique())
    real_brokers = [b for b in all_brokers if b not in TEST_BROKER_LABELS]

    ids = _normalize_filter_ids(broker)
    sel = (real_brokers if ids == ["all"]
           else [broker_by_id[i] for i in ids if i in broker_by_id])
    if not sel:
        sel = real_brokers
    cutoff = _history_start_cutoff(history_start)
    # No-op ONLY when the broker selection covers every broker present AND
    # there is no history cutoff — otherwise we must fall through to apply the
    # cutoff (which the default broker view still needs).
    if set(sel) == set(all_brokers) and cutoff is None:
        return frames

    pos = frames.positions[frames.positions["broker"].astype(str).isin(sel)].copy()
    txns = (frames.transactions[frames.transactions["broker"].astype(str).isin(sel)].copy()
            if not frames.transactions.empty else frames.transactions)

    pos_monthly = monthly_normalize(pos)
    if not frames.prices_latest.empty:
        pos_monthly = mark_to_market(pos_monthly, frames.prices_latest)

    # The COMPLETE narrowing — every real-$ per-account/summary sidecar goes
    # through here (mirrors app.py's ``kept_accounts | {"PORTFOLIO"}`` logic,
    # app.py:1646-1660), not just the positions/transactions book. This is the
    # single choke-point: a service reading e.g. twr_monthly.csv or
    # summaries.csv fresh from data_dir instead of frames.twr_account /
    # frames.summaries would bypass it and leak real dollars under a
    # test-only broker selection.
    kept_accts = set(pos["account_id"].astype(str).unique())
    is_real = set(sel) == set(real_brokers)
    broker_scope = (None if is_real
                    else tuple(sorted(label_by_raw.get(b, str(b)) for b in sel)))

    # per-account IRR: keep kept accounts + the PORTFOLIO row. On a non-real
    # selection the cached whole-book PORTFOLIO row is wrong for the scope —
    # drop it, then RECOMPUTE a scoped one from the narrowed frames
    # (compute_portfolio_irr(scoped=True): external flows + any internal leg
    # whose pair_id partner left the scope). Deliberate divergence from
    # app.py:1649-1660, which still hides IRR under its broker filter
    # (spec 2026-08-07-broker-scoped-irr-design.md).
    irr = (frames.irr_table[frames.irr_table["account_id"].astype(str).isin(kept_accts | {"PORTFOLIO"})].copy()
           if not frames.irr_table.empty else frames.irr_table)
    if not is_real and not irr.empty:
        irr = irr[irr["account_id"].astype(str) != "PORTFOLIO"].copy()
    if not is_real and cutoff is None:
        # NOTE: the appended row's start_date/end_date are pd.Timestamp
        # objects; every CSV-loaded row in `irr` carries them as strings —
        # this makes irr_table's start_date/end_date a mixed-dtype object
        # column. Harmless today (payload consumers only read account_id/irr
        # off this row), but a future consumer that formats these dates
        # should parse-dates-on-load instead of relying on a shared dtype.
        row = _portfolio_irr_row(pos, txns, scoped=True)
        if row is not None:
            irr = pd.concat([irr, pd.DataFrame([row])], ignore_index=True)
    # per-account TWR: filter to kept accounts (drops demo on the real
    # default; drops real on a test-only selection).
    twr_account = (frames.twr_account[frames.twr_account["account_id"].astype(str).isin(kept_accts)].copy()
                   if not frames.twr_account.empty else frames.twr_account)
    # per-account summaries: filter to the selected brokers (no demo sidecar
    # -> empties out under a test-only selection).
    summaries = (frames.summaries[frames.summaries["broker"].astype(str).isin(sel)].copy()
                 if (not frames.summaries.empty and "broker" in frames.summaries.columns)
                 else frames.summaries)
    # aggregate portfolio TWR: keep the cached real series on the real book;
    # on a non-real selection the cached whole-book series is wrong (it pairs
    # cross-account flows across the WHOLE book, so it can't be subset), so
    # RECOMPUTE it NAV-weighted from the narrowed per-account `twr_account`.
    # Guard the empty case (mirrors app.py:1654 `if not twr_account.empty`): a
    # missing twr_monthly.csv makes `_read_csv` return a COLUMNLESS empty frame,
    # which would KeyError inside the recompute — blank it instead (no
    # per-account TWR to aggregate; still no real-book leak). Only the kept
    # accounts ever feed the recompute, so a demo/subset view never leaks (S2b).
    if is_real:
        twr_portfolio = frames.twr_portfolio
    elif not twr_account.empty:
        twr_portfolio = recompute_portfolio_twr(twr_account)
    else:
        twr_portfolio = pd.DataFrame()

    # --- history-start cutoff (filter-parity S3) ------------------------------
    # Mirror app.py:1640-1704 IN ORDER (the ordering is load-bearing): recompute
    # IRR on the broker-filtered-but-NOT-yet-sliced frames so compute_*_irr can
    # see the pre-cutoff NAV history to inject nav_at_cutoff; slicing first would
    # zero it (silently wrong IRR). daily_prices/bench_tr are NOT clamped here.
    if cutoff is not None:
        if not pos.empty:
            irr = compute_account_irr(
                pos, txns, synthetic_onboarding=cfg.SYNTHETIC_ONBOARDING,
                start_date=cutoff)
            # PORTFOLIO row: canonical book -> whole-book semantics
            # (scoped=False); non-real selection -> scoped recompute. Both
            # share the finite + floor-band guard (the band guard is a
            # deliberate tightening of the old finite-only gate).
            row = _portfolio_irr_row(pos, txns, scoped=(not is_real),
                                     start_date=cutoff)
            if row is not None:
                irr = pd.concat([irr, pd.DataFrame([row])],
                                ignore_index=True)
        else:
            irr = pd.DataFrame()
        # THEN slice positions / transactions / twr_account; rebuild monthly.
        pos = pos[pos["statement_date"] >= cutoff].copy()
        pos_monthly = monthly_normalize(pos)
        if not frames.prices_latest.empty:
            pos_monthly = mark_to_market(pos_monthly, frames.prices_latest)
        if not txns.empty:
            txns = txns[txns["settlement_date"] >= cutoff].copy()
        kept = set(pos["account_id"].astype(str).unique())
        if not twr_account.empty:
            twr_account = twr_account[
                (twr_account["month"].dt.to_timestamp() >= cutoff)
                & twr_account["account_id"].astype(str).isin(kept)].copy()
        # PURE history cutoff on the canonical book: slice the canonical
        # frame — exact (each month's Dietz return is window-independent).
        # The NAV-weighted recompute is the disclosed approximation for
        # account subsets only; using it here shifted the default 2021+
        # view +0.38pp off the canonical series (DA-D-3).
        if is_real and not frames.twr_portfolio.empty:
            twr_portfolio = slice_canonical_twr(frames.twr_portfolio, cutoff)
        elif not twr_account.empty:
            twr_portfolio = recompute_portfolio_twr(twr_account)
        else:
            twr_portfolio = pd.DataFrame()

    return replace(
        frames,
        positions=pos,
        transactions=txns,
        positions_monthly=pos_monthly,
        twr_portfolio=twr_portfolio,
        twr_account=twr_account,
        irr_table=irr,
        summaries=summaries,
        broker_scope=broker_scope,
    )


def _class_options(snap_all: pd.DataFrame) -> tuple[list[dict], dict]:
    """[{id,label}] asset-class filter options (id = the raw asset_class key,
    which is already an opaque non-identifier) + id→asset_class reverse map."""
    classes = sorted(snap_all["asset_class"].dropna().astype(str).unique())
    opts = [{"id": c, "label": CLASS_LABELS.get(c, c)} for c in classes]
    by_id = {c: c for c in classes}
    return opts, by_id


def filter_option_ids(frames: Frames, *,
                      as_of: str | None = None) -> tuple[set[str], set[str]]:
    """(account_ids, class_ids) exactly as a tab view's ``meta.accounts`` /
    ``meta.classes`` would list them — the cheap validation source for the
    server handlers (no throwaway full-view build). Every service derives its
    option lists from ``_account_options`` / ``_class_options`` over
    ``_current_snap``; ``as_of`` matters only for Holdings, whose options come
    from the as-of month snapshot. Pinned per view by
    tests/test_terminal_request_reuse.py."""
    snap_all = _current_snap(frames, as_of)
    acct_opts, _ = _account_options(snap_all)
    class_opts, _ = _class_options(snap_all)
    return ({o["id"] for o in acct_opts}, {o["id"] for o in class_opts})


def _health(frames: Frames) -> dict:
    """``{level,text}`` ingest-health headline via parsers.data_health. Best
    effort: any missing input / load failure degrades to a muted empty strip
    rather than crashing the view (the API stays up even on a thin data dir).

    Delegates the report build to ``health_service.build_health_report_for_frames``
    (lazy import to avoid a load-time cycle — health_service imports this module)
    so the chrome strip and the Data Health tab share one builder and cannot
    drift."""
    try:
        from terminal import health_service
        from data_health import format_health_headline
        report = health_service.build_health_report_for_frames(frames)
        level, text = format_health_headline(report)
        # Map data_health's {green,amber,red,grey} → the front-end's level set.
        level_map = {"green": "success", "amber": "warning",
                     "red": "error", "grey": "muted"}
        return {"level": level_map.get(level, "muted"), "text": text}
    except (FileNotFoundError, KeyError, ImportError, ValueError) as exc:
        # Expected data-absence modes (missing input / thin data dir) degrade to
        # a muted empty strip so the API stays up. A truly unexpected exception
        # type is NOT caught here — it propagates so real bugs stay observable.
        logger.warning("data-health headline unavailable, degrading: %r", exc)
        return {"level": "muted", "text": ""}


def _carried_forward(snap: pd.DataFrame) -> list[dict]:
    """Accounts shown via a carried-forward (``_filled``) row, with the as-of
    date their last-known holdings come from (mirrors app.py:2328-2341)."""
    out: list[dict] = []
    if "_filled" not in snap.columns:
        return out
    carried = snap[snap["_filled"].fillna(False).astype(bool)]
    if carried.empty:
        return out
    for _acct_id, sub in carried.groupby("account_id"):
        disp = (str(sub["account_display"].iloc[0])
                if "account_display" in sub.columns else str(_acct_id))
        asof = pd.Timestamp(sub["_as_of_date"].iloc[0])
        out.append({"account": disp, "as_of": asof.strftime("%b %d, %Y")})
    return out


def _snapshot_cards(snap: pd.DataFrame, snap_all: pd.DataFrame,
                    frames: Frames, as_of_ts: pd.Timestamp,
                    holdings_filtered: bool, *,
                    bucket_sel: list | None = None,
                    class_sel: list | None = None) -> dict:
    """The header cards (mirror app.py:2294-2322, plus the YTD card): portfolio
    value, vs-prior, YTD, accounts, holdings.

    The prior-period and year-end comparisons apply the SAME account / class
    selection as the current snapshot — compared against the unfiltered prior
    book, a one-account filter read "-82.6%" (TK, 2026-08-22). Both are VALUE
    deltas (flows included, like the Streamlit card); the flow-stripped return
    lives on the Performance tab. YTD anchors on the last snapshot dated in
    the prior calendar year (the December statement)."""
    total_filtered = float(snap["market_value"].sum()) if not snap.empty else 0.0
    total_all = float(snap_all["market_value"].sum()) if not snap_all.empty else 0.0

    def _scoped_total(when: pd.Timestamp) -> float:
        past = slice_as_of_month(frames.positions_monthly, when)
        if bucket_sel:
            past = past[past["bucket"].isin(bucket_sel)]
        if class_sel:
            past = past[past["asset_class"].isin(class_sel)]
        return float(past["market_value"].sum()) if not past.empty else 0.0

    def _delta_card(ref_ts: pd.Timestamp | None, missing_label: str) -> dict:
        ref_total = _scoped_total(ref_ts) if ref_ts is not None else 0.0
        if ref_total > 0:
            delta_abs = total_filtered - ref_total
            delta_pct = (total_filtered / ref_total - 1.0) * 100.0
            return {
                "pct": fmt_pct(delta_pct),
                "dir": "up" if delta_abs >= 0 else "down",
                "abs": ("+" if delta_abs >= 0 else "") + fmt_money(delta_abs),
                "prior_label": f"vs {ref_ts.strftime('%b %d, %Y')}",
            }
        return {"pct": "-", "dir": "flat", "abs": "",
                "prior_label": missing_label}

    # available_dates is newest-first, so the first hit is the latest match.
    prior_dates = [d for d in frames.available_dates
                   if pd.Timestamp(d) < as_of_ts]
    prior_ts = pd.Timestamp(prior_dates[0]) if prior_dates else None
    year_end_dates = [d for d in frames.available_dates
                      if pd.Timestamp(d).year < as_of_ts.year]
    year_end_ts = pd.Timestamp(year_end_dates[0]) if year_end_dates else None

    n_brokers = int(snap["broker"].nunique()) if not snap.empty else 0
    accounts_sub = (f"on this date · {n_brokers} broker"
                    + ("s" if n_brokers != 1 else ""))

    return {
        "portfolio_value": fmt_money(total_filtered),
        "portfolio_value_sub": (
            f"of {fmt_money(total_all)} unfiltered · marked to live prices"
            if holdings_filtered else "marked to live prices"),
        "vs_prior": _delta_card(prior_ts, "No earlier snapshot"),
        "ytd": _delta_card(year_end_ts, "No prior year-end snapshot"),
        "accounts": {
            "value": int(snap["bucket"].nunique()) if not snap.empty else 0,
            "sub": accounts_sub,
        },
        "holdings": {
            "symbols": int(snap["display_symbol"].nunique()) if not snap.empty else 0,
            "rows": int(len(snap)),
        },
    }


def build_holdings_view(frames: Frames, *, as_of: str | None = None,
                        account: str | list[str] = "all",
                        asset_class: str | list[str] = "all",
                        top_n: int = 15, search: str = "") -> dict:
    """Assemble the full ``GET /api/holdings`` contract (spec §4.1).

    Pure given ``frames`` + selections. ``as_of=None`` resolves to the latest
    available date. ``account`` / ``asset_class`` are the opaque option ids from
    ``meta.accounts`` / ``meta.classes`` (``"all"`` = no filter). Each accepts a
    scalar id or a list of ids (multi-select) — the resolved buckets / classes are
    unioned and applied to the snapshot, mirroring app.py:2261-2264. Every value
    is JSON-native (``json.dumps`` will not raise).
    """
    as_of = as_of or (frames.available_dates[0] if frames.available_dates else None)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()

    snap_all = _current_snap(frames, as_of)
    acct_opts, acct_by_id = _account_options(snap_all)
    class_opts, class_by_id = _class_options(snap_all)
    broker_opts, _ = _broker_options(snap_all)

    # Resolve the opaque filter ids -> the underlying buckets / asset_classes.
    # account/asset_class accept a scalar id/'all' or a list (multi-select).
    bucket_sel = _resolve_ids(_normalize_filter_ids(account), acct_by_id)
    class_sel = _resolve_ids(_normalize_filter_ids(asset_class), class_by_id)

    snap = snap_all
    if bucket_sel:
        snap = snap[snap["bucket"].isin(bucket_sel)]
    if class_sel:
        snap = snap[snap["asset_class"].isin(class_sel)]
    snap = snap.copy()
    holdings_filtered = bool(bucket_sel) or bool(class_sel)

    total = float(snap["market_value"].sum()) if not snap.empty else 0.0

    meta = {
        "as_of": as_of,
        "as_of_label": as_of_ts.strftime("%b %d, %Y"),
        "available_dates": list(frames.available_dates),
        "accounts": acct_opts,
        "classes": class_opts,
        "brokers": broker_opts,
        "filter": _filter_meta(account, asset_class),
        "synthetic": "synth" in str(frames.data_dir).lower(),
    }

    view = {
        "meta": meta,
        "tape": _kpi_tape(frames),
        "health": _health(frames),
        "carried_forward": _carried_forward(snap),
        "snapshot": _snapshot_cards(snap, snap_all, frames, as_of_ts,
                                    holdings_filtered, bucket_sel=bucket_sel,
                                    class_sel=class_sel),
        "alloc_class": _alloc_by_class(snap),
        "alloc_account": _alloc_by_account(snap),
        "top_holdings": _top_holdings(snap, total, top_n=top_n),
        "positions": _positions_table(snap, frames, total, as_of, search=search),
    }
    return view
