"""Live mark-to-market overlay for the latest-snapshot positions.

Pure function extracted from app.py so it can be unit-tested without
importing the full Streamlit module. The dashboard imports `mark_to_market`
from here; tests do the same.
"""
from __future__ import annotations

import pandas as pd


def mark_to_market(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Overwrite `price` and `market_value` on the latest-snapshot rows using
    prices_latest. Only `ok` and `cash_fixed_1` statuses contribute. Rows
    with no price match keep their statement-date values (graceful fallback
    for bonds, brand-new symbols not in the prices file).

    Historical snapshots (statement_date < latest) are NEVER touched —
    pure-historical views remain pure.

    Option positions are deliberately skipped: their `symbol` is the
    underlying (e.g. ``NVDA`` for an NVDA 115P), but pricing a contract at
    the underlying spot is meaningless — it gives ``qty × spot_price``,
    not ``qty × 100 × option_mid``. The Options Hedging tab uses
    ``data/option_position_snapshot.csv`` for live option marks instead;
    the Holdings tab keeps the statement-date MV for option rows on the
    same principle (the broker's last close is the right baseline until
    a real option-IV refresh lands).

    `unrealized_gl` is recomputed on overwritten rows (mv - cost_basis) so the
    Holdings table's $ and % columns stay consistent on the latest snapshot.

    The pre-mark values are stashed in `market_value_stmt` (first mark wins:
    re-marking an already-marked frame, e.g. positions_monthly rebuilt from
    marked positions, never clobbers it). parsers/data_health.py prefers that
    column so its reconciliation stays on the statement basis the ingest gate
    uses — a live-price move since the statement date must not read as drift.
    """
    if prices.empty or positions.empty:
        return positions
    use = prices[prices["status"].isin(["ok", "cash_fixed_1"])]
    if use.empty:
        return positions
    price_map = dict(zip(use["symbol"], use["close"]))

    latest_date = positions["statement_date"].max()
    df = positions.copy()
    if "market_value_stmt" not in df.columns:
        df["market_value_stmt"] = df["market_value"]
    latest_idx = df.index[df["statement_date"] == latest_date]
    for i in latest_idx:
        # Skip option rows — their `symbol` is the underlying, and applying
        # the underlying close gives a nonsensical MV (qty × spot, not
        # qty × 100 × option_mid).
        asset_class = df.at[i, "asset_class"] if "asset_class" in df.columns else None
        if isinstance(asset_class, str) and asset_class.startswith("option"):
            continue
        sym = df.at[i, "symbol"]
        if not isinstance(sym, str) or not sym.strip():
            continue
        new_price = price_map.get(sym.strip())
        if new_price is None:
            continue
        qty = float(df.at[i, "quantity"]) if pd.notna(df.at[i, "quantity"]) else 0.0
        new_mv = qty * float(new_price)
        df.at[i, "price"] = float(new_price)
        df.at[i, "market_value"] = new_mv
        cb = df.at[i, "cost_basis"]
        if pd.notna(cb):
            df.at[i, "unrealized_gl"] = new_mv - float(cb)
    return df
