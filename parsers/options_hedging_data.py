"""
Pure data-shaping helpers for the Options Hedging tab.

These functions take the dashboard's raw positions / opt_tbl / priced-
ticker set and produce the inputs the hedge_recommender engine consumes.
No Streamlit, no I/O, no network — all side-effect-free transforms so
the wrapper logic that surfaced 15 bugs in PR #107 can be unit-tested.

See docs/superpowers/specs/2026-05-28-options-hedging-data-helpers-design.md
for the full bug-ledger mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from monthly_normalize import slice_as_of_month


def as_of_holdings(positions: pd.DataFrame,
                   as_of: pd.Timestamp) -> pd.DataFrame:
    """Slice positions to rows in as_of's calendar month, rename
    'symbol' -> 'ticker', and normalize ticker (uppercase, strip, fill
    NaN with empty string so Treasury CUSIPs survive downstream filters
    that key off asset_class)."""
    out = slice_as_of_month(positions, as_of)
    if "symbol" in out.columns:
        out = out.rename(columns={"symbol": "ticker"})
    out["ticker"] = (out["ticker"].fillna("")
                                  .astype(str).str.strip().str.upper())
    return out


_KNOWN_CASH_TICKERS = frozenset({
    "SGOV", "BIL", "SHV", "GOVT", "TLH",
    "SPAXX", "FZDXX", "FDRXX",
})
_KNOWN_COMMODITY_TICKERS = frozenset({"GLD", "SLV", "IAU"})
_CASH_LIKE_ASSET_CLASSES = frozenset({"cash", "fixed_income"})
_OPT_SYMBOL_PATTERN = r"\b(?:PUT|CALL)\b"
_OPT_DESC_PATTERN = r"^\s*(?:PUT|CALL)\b"


def _is_option_row(row: pd.Series) -> bool:
    """Robust 3-way option detection: asset_class prefix OR symbol regex
    OR description regex. Catches positions
    synthesize_interim_positions tags as asset_class='other'."""
    import re
    ac = str(row.get("asset_class", "") or "")
    if ac.startswith("option"):
        return True
    sym = str(row.get("ticker", "") or "")
    if re.search(_OPT_SYMBOL_PATTERN, sym):
        return True
    desc = str(row.get("description", "") or "")
    if re.match(_OPT_DESC_PATTERN, desc):
        return True
    return False


def classify_holding(row: pd.Series) -> str:
    """3-way classifier: 'option' | 'cash' | 'equity'. Hybrid rule —
    known cash tickers + known commodities + asset_class fallback.
    Captures GLD ($288k bug), Treasury CUSIPs (NaN symbol), SGOV-as-
    fixed_income, and option positions misclassified upstream."""
    if _is_option_row(row):
        return "option"
    ticker = str(row.get("ticker", "") or "").strip().upper()
    if ticker in _KNOWN_CASH_TICKERS:
        return "cash"
    if ticker in _KNOWN_COMMODITY_TICKERS:
        return "equity"
    if row.get("asset_class") in _CASH_LIKE_ASSET_CLASSES:
        return "cash"
    return "equity"


def aggregate_by_ticker(holdings: pd.DataFrame) -> pd.DataFrame:
    """Sum market_value across brokers for same ticker. Prior wrapper
    used dict(zip(...)) which silently kept only the last row when a
    ticker appeared in multiple brokers (e.g., SPY in Harbor + Alpine)."""
    if holdings.empty:
        return holdings[["ticker", "market_value"]].copy()
    out = (holdings.groupby("ticker", as_index=False)["market_value"]
                   .sum())
    return out


def filter_to_priced(
    equity: pd.DataFrame,
    priced_tickers: set[str],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Filter equity rows to tickers with price history; return
    (filtered_df, coverage_stats). Stats keys:
        equity_mv_total, equity_mv_priced, coverage_pct,
        n_priced_tickers, n_unpriced."""
    equity_mv_total = float(equity["market_value"].sum()) if not equity.empty else 0.0
    if equity.empty:
        return equity.copy(), {
            "equity_mv_total":   0.0,
            "equity_mv_priced":  0.0,
            "coverage_pct":      0.0,
            "n_priced_tickers":  0,
            "n_unpriced":        0,
        }
    mask = equity["ticker"].isin(priced_tickers)
    filtered = equity[mask].copy()
    equity_mv_priced = float(filtered["market_value"].sum())
    coverage_pct = (equity_mv_priced / equity_mv_total
                    if equity_mv_total > 0 else 0.0)
    n_priced_tickers = int(filtered["ticker"].nunique())
    n_unpriced = int(equity["ticker"].nunique() - n_priced_tickers)
    return filtered, {
        "equity_mv_total":   equity_mv_total,
        "equity_mv_priced":  equity_mv_priced,
        "coverage_pct":      coverage_pct,
        "n_priced_tickers":  n_priced_tickers,
        "n_unpriced":        n_unpriced,
    }


def build_holdings_for_engine(
    equity_priced: pd.DataFrame,
    cash_total: float,
) -> pd.DataFrame:
    """Append a synthetic SGOV row carrying cash_total when cash > 0,
    so the engine's classify_holdings sees the cash-equivalent slice
    without whitelisting every CD / sweep symbol. Negative or zero
    cash leaves equity_priced untouched."""
    equity_rows = equity_priced[["ticker", "market_value"]].copy()
    if cash_total > 0:
        return pd.concat([
            equity_rows,
            pd.DataFrame([{"ticker": "SGOV", "market_value": float(cash_total)}]),
        ], ignore_index=True)
    return equity_rows


def build_existing_options_rows(opt_tbl: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert opt_tbl to the list[dict] the engine consumes. One row
    per broker-lot — DO NOT collapse by (underlying, strike, expiry),
    because the engine's worst_payoffs list is parallel-indexed. Calls
    are skipped (this sleeve is put-only)."""
    if opt_tbl is None or opt_tbl.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in opt_tbl.iterrows():
        if str(row.get("opt_type", "")).lower() != "put":
            continue
        try:
            exp = pd.Timestamp(row["expiry"]).date()
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "underlying":           str(row["underlying"]),
            "strike":               float(row["strike"]),
            "expiry":               exp,
            "contracts":            int(abs(float(row["quantity"]))),
            "cost_basis_per_share": float(row.get("cost_basis_per_share", 0.0)),
            "market_value":         float(row.get("market_value", 0.0)),
        })
    return out


@dataclass(frozen=True)
class OptionsHedgingInputs:
    """Bundle of everything the Options Hedging tab passes to the
    recommender engine, plus what the UI's Section 1 + coverage caption
    need to render."""
    composition_breakdown: dict[str, float]
    holdings_for_engine:   pd.DataFrame
    existing_options:      list[dict[str, Any]]
    coverage_stats:        dict[str, float | int]
    # Priced equity universe the recommender hedges — i.e. the equity
    # tickers WITHOUT the synthetic SGOV cash sentinel that
    # build_holdings_for_engine appends. Lets the UI read the universe
    # directly instead of filtering holdings_for_engine on "SGOV".
    equity_priced_tickers: tuple[str, ...]


def build_options_hedging_inputs(
    positions: pd.DataFrame,
    opt_tbl: pd.DataFrame,
    priced_tickers: set[str],
    as_of: pd.Timestamp,
) -> OptionsHedgingInputs:
    """Run the full data-shaping pipeline. See module docstring for the
    bug-ledger mapping (cfd373c). All inputs and outputs are pure
    pandas; no Streamlit, no I/O."""
    h_full = as_of_holdings(positions, as_of)
    portfolio_value = float(h_full["market_value"].sum())

    classes = h_full.apply(classify_holding, axis=1)
    options_mv = float(h_full.loc[classes == "option", "market_value"].sum())
    equity_rows = h_full.loc[classes == "equity"].copy()
    cash_total  = float(h_full.loc[classes == "cash", "market_value"].sum())
    equity_mv = float(equity_rows["market_value"].sum())

    equity_agg = aggregate_by_ticker(equity_rows)
    equity_priced, coverage_stats = filter_to_priced(equity_agg, priced_tickers)
    equity_priced_tickers = tuple(equity_priced["ticker"].astype(str).tolist())
    holdings_for_engine = build_holdings_for_engine(equity_priced, cash_total)
    existing_options = build_existing_options_rows(opt_tbl)

    composition = {
        "portfolio_value": portfolio_value,
        "equity_mv":       equity_mv,
        "cash_mv":         cash_total,
        "options_mv":      options_mv,
    }
    return OptionsHedgingInputs(
        composition_breakdown=composition,
        holdings_for_engine=holdings_for_engine,
        existing_options=existing_options,
        coverage_stats=coverage_stats,
        equity_priced_tickers=equity_priced_tickers,
    )
