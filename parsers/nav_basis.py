"""Canonical NAV basis reconciliation (AUDIT-NAV).

Two legitimate NAV bases coexist on the dashboard and previously disagreed
on screen with no label:
  * canonical / "Portfolio value" — interim-rolled, marked-to-live: current
    economic worth. The Holdings/Income/Options tiles already use this.
  * return-basis — statement carry-forward NAV that anchors the TWR/IRR return
    series (it must link actual statement NAVs, not live marks).
This module computes both and a one-line reconciliation so every tab can cite
its basis instead of silently disagreeing.
"""
import pandas as pd


def canonical_nav(marked_snapshot: pd.DataFrame, exclude_account_ids=()) -> float:
    """Real-only marked NAV at the latest snapshot — the canonical headline
    "Portfolio value". `marked_snapshot` is the monthly-normalized,
    mark-to-market positions frame; sum `market_value` at its max
    `statement_date`, excluding demo/test accounts."""
    if marked_snapshot.empty or "statement_date" not in marked_snapshot.columns:
        return 0.0
    as_of = marked_snapshot["statement_date"].max()
    snap = marked_snapshot[marked_snapshot["statement_date"] == as_of]
    if exclude_account_ids:
        snap = snap[~snap["account_id"].astype(str).isin(set(exclude_account_ids))]
    return float(snap["market_value"].sum())


def return_basis_nav(portfolio_twr: pd.DataFrame) -> float:
    """Statement carry-forward NAV that anchors the return series = the latest
    `nav` in the portfolio TWR frame (parsers/compute_twr.py)."""
    if portfolio_twr.empty or "nav" not in portfolio_twr.columns:
        return 0.0
    return float(portfolio_twr["nav"].iloc[-1])


def nav_reconciliation(canonical: float, return_basis: float) -> dict:
    """Gap + a one-line human caption between the marked and return bases."""
    gap = canonical - return_basis
    return {
        "canonical": canonical,
        "return_basis": return_basis,
        "gap": gap,
        "caption": (
            f"Return-basis NAV ${return_basis:,.0f} anchors the return series; "
            f"current marked value ${canonical:,.0f} differs by ${gap:+,.0f} "
            f"— market moves + interim activity since the last statements."
        ),
    }
