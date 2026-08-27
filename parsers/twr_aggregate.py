"""Portfolio-level TWR re-aggregation from a per-account subset.

Extracted verbatim from app.py (filter-parity Slice 2b) so both the Streamlit
app and the MERIDIAN terminal share ONE definition. Pure pandas — imports no
Streamlit, no config, no terminal/app modules (an import-hygiene test guards
this).

Used when a broker / history filter is non-default: the precomputed
``twr_portfolio.csv`` pairs cross-account flows across the *whole* portfolio,
so a subset can't be derived from it. Instead, weight each account's monthly
TWR by its prior-month NAV and chain.
"""
import pandas as pd
from typing import NamedTuple


class TwrHeadline(NamedTuple):
    cum: float
    ann: float
    mdd: float
    n: int
    start_month: str
    mdd_month: str


def slice_canonical_twr(twr_portfolio: pd.DataFrame,
                        cutoff: "pd.Timestamp") -> pd.DataFrame:
    """Slice the CANONICAL portfolio TWR frame at a history cutoff — EXACT.

    Each month's Modified-Dietz return is independent of the window start,
    so for a pure history-start cutoff on the canonical book the rows at
    ``month >= cutoff`` ARE the canonical series over that window;
    consumers re-chain cumulative/wealth from ``return_pct`` themselves.
    The NAV-weighted ``recompute_portfolio_twr`` below is an APPROXIMATION
    (prior-month-NAV weights; subset-boundary pairing) reserved for
    account-subset scopes where the canonical cross-account pairing
    genuinely cannot be subset — using it for a pure cutoff shifted the
    terminal's default 2021+ view +0.38pp cumulative off the canonical
    series its methodology text claims (DA-D-3)."""
    if (twr_portfolio is None or twr_portfolio.empty
            or "month" not in twr_portfolio.columns):
        return pd.DataFrame()
    months = pd.PeriodIndex(twr_portfolio["month"], freq="M").to_timestamp()
    out = twr_portfolio[months >= pd.Timestamp(cutoff)].copy()
    out = out.reset_index(drop=True)
    if "cum_return" in out.columns and "return_pct" in out.columns:
        # A pre-derived frame (app.py's load_twr shape) carries cumulative
        # columns chained FROM INCEPTION — window-relative by definition,
        # so re-chain them over the slice. (The terminal's raw frame has
        # none of these; its consumers derive their own.)
        wealth = (1.0 + out["return_pct"].fillna(0.0)).cumprod()
        out["cum_return"] = wealth - 1.0
        out["wealth_index"] = wealth
        out["wealth_peak"] = wealth.cummax()
        out["twr_dd_pct"] = (wealth / wealth.cummax() - 1.0) * 100.0
    return out


def recompute_portfolio_twr(twr_account_subset: pd.DataFrame) -> pd.DataFrame:
    """NAV-weighted monthly TWR aggregation across a subset of accounts.

    Used when the broker filter is non-default — the precomputed
    `twr_portfolio.csv` uses cross-account flow pairing across the *whole*
    portfolio, so a subset can't be derived from it. Instead, weight each
    account's monthly TWR by its prior-month NAV and chain.

    Approximate: inter-broker transfers within the filtered set still get
    treated as external at the account level. For Fidelity-only or JPM-only
    views this is fine — those legs are external from the subset's POV anyway.
    """
    sub = twr_account_subset.dropna(subset=["return_pct"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["weight"] = sub["prev_nav"].fillna(0).clip(lower=0)

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].sum()
        ret = (g["return_pct"] * g["weight"]).sum() / w if w > 0 else 0.0
        return pd.Series({"return_pct": ret, "nav": g["nav"].sum()})

    monthly = (sub.groupby("month", as_index=False, group_keys=False)
                  .apply(_agg, include_groups=False))
    monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["statement_date"] = monthly["month"].dt.to_timestamp("M")
    monthly["prev_nav"] = monthly["nav"].shift(1)
    monthly["prev_stmt_date"] = monthly["statement_date"].shift(1)
    monthly["cum_return"] = (1.0 + monthly["return_pct"].fillna(0.0)).cumprod() - 1.0
    wealth = (1.0 + monthly["return_pct"].fillna(0.0)).cumprod()
    monthly["wealth_index"] = wealth
    monthly["wealth_peak"] = wealth.cummax()
    monthly["twr_dd_pct"] = (wealth / wealth.cummax() - 1.0) * 100.0
    return monthly


def portfolio_twr_headline(twr_portfolio: pd.DataFrame) -> TwrHeadline:
    """Cumulative TWR, annualized TWR, max drawdown (%), month count, first
    month, and max-drawdown month — recomputed from a twr_portfolio frame's
    ``return_pct`` + ``month`` columns. ONE source for BOTH UIs' KPI tape +
    Performance headline (render_chrome single-source). Recomputes from
    ``return_pct`` (present on the raw AND the prepared frame) rather than
    reading the prepared ``cum_return``/``twr_dd_pct`` columns, so it works on
    either shape; the formula is identical to those columns' own derivation, so
    the numbers are unchanged. Empty / no-``return_pct`` frame -> all-NaN."""
    if (twr_portfolio is None or twr_portfolio.empty
            or "return_pct" not in twr_portfolio.columns):
        return TwrHeadline(float("nan"), float("nan"), float("nan"), 0, "—", "—")
    rp = twr_portfolio["return_pct"]
    wealth = (1.0 + rp.fillna(0.0)).cumprod()
    cum = float(wealth.iloc[-1] - 1.0)
    n = int(rp.notna().sum())
    ann = (1.0 + cum) ** (12.0 / n) - 1.0 if n > 0 else float("nan")
    dd = (wealth / wealth.cummax() - 1.0) * 100.0
    mdd = float(dd.min())
    start_month = str(twr_portfolio["month"].iloc[0])
    mdd_month = str(twr_portfolio["month"].loc[dd.idxmin()])
    return TwrHeadline(cum, ann, mdd, n, start_month, mdd_month)
