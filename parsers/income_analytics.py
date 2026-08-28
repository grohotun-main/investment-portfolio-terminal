"""Income analytics engine.

Actual income from the transactions frame (dividend / interest /
withholding rows) plus forward income per holding: a dividend channel
from per-ticker dividend-history CSVs (data/dividends_<ticker>.csv,
written by parsers/fetch_dividends.py) and a coupon channel for
fixed-income rows the dividend files can't cover (bare-CUSIP Treasury
rungs, bond funds with statement est. annual income).

Pure pandas — no Streamlit imports.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Income rows. `reinvestment` is excluded on purpose (the buy side of a
# DRIP — counting it would double-count the paired dividend row).
INCOME_TYPES = ("dividend", "interest", "withholding")
# `principal_pmt` rows that are DISTRIBUTIONS on held shares count as
# dividends (Distributions S2, TK 2026-08-22): a return-of-capital payout is
# yield to the holder — its ROC label is tax character (basis reduction,
# handled by the lot engine), not a different kind of income (NEOS-style
# funds distribute mostly as ROC; VISN's $10/sh). Two things share the
# broker label and are NOT yield: cash in lieu of fractional shares
# (mergers / splits) and bond principal paydowns (bare-CUSIP rungs).
DISTRIBUTION_TYPES = ("principal_pmt",)
_NOT_YIELD_RE = re.compile(r"CASH IN LIEU|IN LIEU OF|\bCIL\b", re.I)

# Structurally income-free classes. Everything ELSE with a non-blank
# symbol is a dividend-channel candidate: broker/display classes are
# unreliable (Harbor files SGOV under fixed income; the app reclasses
# TLH-account rows to tax_loss_harvesting and commodity ETFs to gold),
# so candidacy is symbol-driven. Options are excluded by class because a
# leg carries its underlying's symbol (a SPY put says "SPY"); cash rows
# carry sweep tickers (QJERQ, SPAXX) that must not be projected either.
NON_INCOME_CLASSES = ("option_put", "option_call", "cash")

_TS_COLS = ["dividends", "interest", "withholding", "net"]

_FWD_COLS = ["symbol", "quantity", "market_value", "cost_basis", "covered",
             "t12m_per_share", "projected", "yield_mv", "yield_cost",
             "channel"]

# "4.12500%" in a bond description -> the coupon; first match wins.
_COUPON_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# MM/DD/YYYY tokens; the LATEST one in a Treasury description is the
# maturity (the dated date precedes it in Harbor's format).
_MDY_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


def _rollup(projected_12m: float, covered_mv: float,
            covered_cost: float, nav: float,
            projected_on_cost: float = 0.0) -> dict:
    return {
        "projected_12m": projected_12m,
        "yield_on_covered_mv": (projected_12m / covered_mv
                                if covered_mv > 0 else float("nan")),
        "yield_on_covered_cost": (projected_on_cost / covered_cost
                                  if covered_cost > 0 else float("nan")),
        "covered_mv": covered_mv,
        "nav": nav,
        "coverage_pct_nav": (covered_mv / nav if nav > 0
                             else float("nan")),
    }


def _security_distribution_mask(tx: pd.DataFrame) -> pd.Series:
    """True where a row is a distribution on held SHARES: it carries a
    symbol (a bare-CUSIP bond rung does not) and is not cash-in-lieu. A
    frame without a symbol column can't be judged — nothing qualifies."""
    if "symbol" not in tx.columns:
        return pd.Series(False, index=tx.index)
    sym = tx["symbol"].astype("string").str.strip()
    has_symbol = sym.notna() & (sym != "")
    if "description" in tx.columns:
        desc = tx["description"].astype("string").fillna("")
        not_yield = desc.str.contains(_NOT_YIELD_RE, regex=True).fillna(False)
    else:
        not_yield = pd.Series(False, index=tx.index)
    return (has_symbol & ~not_yield).fillna(False).astype(bool)


def _income_rows(tx: pd.DataFrame) -> pd.DataFrame:
    """Income-type rows with a coalesced `when` date and numeric `amount`.

    `when` is the settlement_date, falling back to trade_date when
    settlement is missing/unparseable — interim-CSV and some Harbor income
    rows file a NaT settlement but carry a valid trade date, and dropping
    them silently understated the actuals (WSD-1). Rows with neither
    parseable date are still dropped (genuinely undateable). The single
    coalescing point shared by income_timeseries and trailing_income.
    """
    is_income = tx["transaction_type"].isin(INCOME_TYPES)
    is_dist = tx["transaction_type"].isin(DISTRIBUTION_TYPES) & _security_distribution_mask(tx)
    inc = tx[is_income | is_dist].copy()
    # A qualifying distribution IS a dividend downstream (one income number,
    # not a separate bucket — the tax character is the Tax tab's business).
    inc.loc[inc["transaction_type"].isin(DISTRIBUTION_TYPES), "transaction_type"] = "dividend"
    when = pd.to_datetime(inc["settlement_date"], errors="coerce")
    if "trade_date" in inc.columns:
        when = when.fillna(pd.to_datetime(inc["trade_date"], errors="coerce"))
    inc["when"] = when
    inc = inc[inc["when"].notna()].copy()
    inc["amount"] = pd.to_numeric(inc["amount"], errors="coerce").fillna(0.0)
    return inc


def income_timeseries(tx: pd.DataFrame,
                      by: str | None = None) -> pd.DataFrame:
    """Monthly income components from a transactions frame.

    Returns a frame indexed by month-start Timestamp (plus `by` as a
    second index level when given) with columns dividends / interest /
    withholding / net. Withholding rows are summed as recorded (foreign tax
    negative, small reclaims positive — net of the two); they are NOT forced
    to one sign. Rows are dated by settlement_date, falling back to trade_date
    (see _income_rows).
    """
    inc = _income_rows(tx)
    if inc.empty:
        idx = pd.DatetimeIndex([], name="month")
        if by:
            idx = pd.MultiIndex.from_arrays([idx, pd.Index([], name=by)])
        return pd.DataFrame(columns=_TS_COLS, index=idx)
    inc["month"] = inc["when"].dt.to_period("M").dt.to_timestamp()
    keys = ["month"] + ([by] if by else [])
    out = (inc.pivot_table(index=keys, columns="transaction_type",
                           values="amount", aggfunc="sum", fill_value=0.0)
           .reindex(columns=list(INCOME_TYPES), fill_value=0.0)
           .rename(columns={"dividend": "dividends"}))
    out.columns.name = None
    out["net"] = out[["dividends", "interest", "withholding"]].sum(axis=1)
    return out.sort_index()


def latest_ex_date_through(div_hist: dict, asof) -> "pd.Timestamp | None":
    """Latest ex-dividend date at or before ``asof`` across all tickers.

    Polygon publishes DECLARED FUTURE ex-dates, so a plain
    ``max(ex_dividend_date)`` can land in the future and make a "Dividend
    history through <date>" caption claim history that hasn't happened yet.
    Clamping to ``asof`` keeps the caption truthful. Returns None when no
    ticker has an ex-date on or before ``asof``.

    ``div_hist`` is the {ticker: DataFrame} mapping from load_div_history;
    each frame is expected to carry an ``ex_dividend_date`` column.
    """
    asof = pd.Timestamp(asof)
    best = None
    for h in div_hist.values():
        if h is None or h.empty or "ex_dividend_date" not in h.columns:
            continue
        past = pd.to_datetime(h["ex_dividend_date"], errors="coerce")
        past = past[past <= asof]
        if not past.empty:
            m = past.max()
            best = m if best is None else max(best, m)
    return best


def trailing_income(tx: pd.DataFrame, asof: "date | pd.Timestamp",
                    days: int = 365) -> float:
    """Net actual income over the trailing window (asof - days, asof].

    A true rolling trailing-twelve-months on the coalesced income date
    (see _income_rows), not a count of calendar-month buckets. The
    month-bucket version dropped the prior boundary month wholesale while
    leaning on a partial current month, structurally understating the
    figure (WSG-5). Withholding is negative, so the sum is net.
    """
    inc = _income_rows(tx)
    if inc.empty:
        return 0.0
    asof_ts = pd.Timestamp(asof).normalize()
    start = asof_ts - pd.Timedelta(days=days)
    win = inc[(inc["when"] > start) & (inc["when"] <= asof_ts)]
    return float(win["amount"].sum())


def load_div_history(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every data_dir/dividends_<ticker>.csv into {TICKER: frame}.

    A header-only file yields an empty frame — that encodes "asked
    Polygon, confirmed non-payer" (covered, $0). A zero-byte/corrupt file
    is skipped (its ticker stays uncovered, visible via coverage %).
    """
    out: dict[str, pd.DataFrame] = {}
    if not data_dir.is_dir():
        return out
    for p in sorted(data_dir.glob("dividends_*.csv")):
        ticker = p.stem[len("dividends_"):].strip().upper()
        if not ticker:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue  # corrupt/zero-byte -> uncovered, surfaced by coverage %
        if df.empty:
            out[ticker] = pd.DataFrame(
                columns=["ex_dividend_date", "cash_amount"])
            continue
        df["ex_dividend_date"] = pd.to_datetime(
            df.get("ex_dividend_date"), errors="coerce")
        df["cash_amount"] = pd.to_numeric(
            df.get("cash_amount"), errors="coerce")
        out[ticker] = df.dropna(subset=["ex_dividend_date", "cash_amount"])
    return out


def _t12m_regular(hist: pd.DataFrame, start_ts: pd.Timestamp,
                  asof_ts: pd.Timestamp) -> float:
    """Trailing-window per-share sum with one-time/special rows excluded.

    Polygon marks one-time payouts frequency=0 (a special can still be
    typed "CD") and special cash dividends dividend_type="SC"; neither
    belongs in a forward annualization. Missing columns / NaN markers
    mean regular (legacy CSVs, spliced prior-symbol frames).
    """
    win = hist[(hist["ex_dividend_date"] > start_ts)
               & (hist["ex_dividend_date"] <= asof_ts)]
    if win.empty:
        return 0.0
    drop = pd.Series(False, index=win.index)
    if "frequency" in win.columns:
        drop |= pd.to_numeric(win["frequency"], errors="coerce").eq(0)
    if "dividend_type" in win.columns:
        drop |= (win["dividend_type"].astype(str).str.strip().str.upper()
                 .eq("SC"))
    return float(win.loc[~drop, "cash_amount"].sum())


def _coupon_label(symbol: str, desc: str, cusip: str,
                  coupon_pct: float) -> str:
    """Display key for a coupon-channel row: the symbol when there is
    one, else "UST {coupon}% {MM/YYYY-maturity}", else "BOND {cusip}"."""
    if symbol:
        return symbol
    if "TREASURY" in desc.upper() and np.isfinite(coupon_pct):
        dates = [pd.to_datetime(tok, format="%m/%d/%Y", errors="coerce")
                 for tok in _MDY_RE.findall(desc)]
        dates = [d for d in dates if pd.notna(d)]
        label = f"UST {coupon_pct:g}%"
        if dates:
            label += f" {max(dates):%m/%Y}"
        return label
    if cusip:
        return f"BOND {cusip}"
    return "BOND"


def _collapse_to_sleeve(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """One display row for a sleeve account's per-holding frame.

    projected sums the covered rows (uncovered project NaN, as in the
    rollup); yields re-blend exactly — yield_cost prices only the
    cost-known lots, recovering each row's on-cost income as
    yield_cost x cost_basis (the rollup YoC lesson). quantity is NaN:
    shares and bond face don't add.
    """
    mv = float(frame["market_value"].sum())
    cost = float(frame["cost_basis"].sum())
    projected = float(frame["projected"].fillna(0.0).sum())
    known = (frame["cost_basis"] > 0) & frame["yield_cost"].notna()
    cost_known = float(frame.loc[known, "cost_basis"].sum())
    on_cost = float((frame.loc[known, "yield_cost"]
                     * frame.loc[known, "cost_basis"]).sum())
    return pd.DataFrame([{
        "symbol": label,
        "quantity": np.nan,
        "market_value": mv,
        "cost_basis": cost,
        "covered": True,
        "t12m_per_share": np.nan,
        "projected": projected,
        "yield_mv": projected / mv if mv > 0 else np.nan,
        "yield_cost": on_cost / cost_known if cost_known > 0 else np.nan,
        "channel": "sleeve",
    }])[_FWD_COLS]


def forward_income(positions: pd.DataFrame,
                   div_history: dict[str, pd.DataFrame],
                   asof: date,
                   sleeves: "dict[str, str] | None" = None,
                   ) -> tuple[pd.DataFrame, dict]:
    """Per-holding forward income, dividend channel + coupon channel.

    sleeves (optional) maps account_id -> display label for accounts
    managed as one strategy (the Treasury ladder, the direct-index TLH
    book): each label's holdings collapse into a single row, channel
    "sleeve". Display-only — the rollup dict is always computed over
    the full un-collapsed book, and a symbol held both inside and
    outside a sleeve keeps a per-symbol row for the outside lots.
    Requires an account_id column when non-empty.

    Dividend channel — every row with a non-blank symbol outside
    NON_INCOME_CLASSES, summed per symbol: projected = trailing-12M
    regular per-share dividends (window (asof - 12 months, asof],
    ex-date basis; one-time/special rows excluded — see _t12m_regular)
    x current quantity. covered = a dividend file exists (header-only
    file = known non-payer, still covered).

    Coupon channel — fixed_income rows whose symbol has no dividend
    file (bare-CUSIP Treasury rungs, bond funds): projected = statement
    est_annual_income when present, else face (quantity) x coupon
    parsed from the description; rows with neither stay uncovered.
    Labeled via _coupon_label, grouped per label, covered=True,
    t12m_per_share=NaN. No maturity pro-rating: the ladder is treated
    going-concern (a maturing rung is assumed rolled into similar
    paper). The optional description / est_annual_income / cusip
    columns may be absent — the channel just stays inactive.

    Options and cash stay out of both channels but in the NAV
    denominator. Returns (frame sorted by projected desc — `channel`
    says which model priced each row, rollup dict). Yield-on-cost (per
    row and rollup) prices only the cost-known lots: projected income is
    pro-rated by the costed share of quantity, and cost-less holdings
    (carry-forward rows coerce cost to 0) drop out of both sides of the
    ratio entirely; projected_12m always covers the full covered set.
    """
    if sleeves:
        if "account_id" not in positions.columns:
            raise ValueError("sleeves needs an account_id column on positions")
        acct = positions["account_id"].astype(str)
        _, roll = forward_income(positions, div_history, asof)
        rest_frame, _ = forward_income(
            positions[~acct.isin(sleeves)], div_history, asof)
        parts = [rest_frame]
        labels: dict[str, list[str]] = {}
        for a, lab in sleeves.items():
            labels.setdefault(lab, []).append(a)
        for lab, accounts in labels.items():
            sub = positions[acct.isin(accounts)]
            if sub.empty:
                continue
            sub_frame, _ = forward_income(sub, div_history, asof)
            if sub_frame.empty:
                continue
            parts.append(_collapse_to_sleeve(sub_frame, lab))
        out = pd.concat(parts, ignore_index=True)
        return (out.sort_values("projected", ascending=False,
                                na_position="last").reset_index(drop=True),
                roll)

    pos = positions.copy()
    for col in ("quantity", "market_value", "cost_basis"):
        pos[col] = pd.to_numeric(pos[col], errors="coerce").fillna(0.0)
    nav = float(pos["market_value"].sum())
    # Quantity held in lots whose cost is known. A symbol can mix costed
    # and cost-less lots (Harbor statements omit cost); YoC must price only
    # the costed lots' share of income, not all income over partial cost.
    pos["_qty_costed"] = pos["quantity"].where(pos["cost_basis"] > 0, 0.0)

    # NaN symbols (bare-CUSIP bonds) must become "", not the str "nan".
    sym = pos["symbol"].fillna("").astype(str).str.strip().str.upper()
    cls = pos["asset_class"].fillna("").astype(str)
    desc = (pos["description"].fillna("").astype(str)
            if "description" in pos.columns
            else pd.Series("", index=pos.index, dtype=object))
    cusip = (pos["cusip"].fillna("").astype(str).str.strip()
             if "cusip" in pos.columns
             else pd.Series("", index=pos.index, dtype=object))
    est = (pd.to_numeric(pos["est_annual_income"], errors="coerce")
           if "est_annual_income" in pos.columns
           else pd.Series(np.nan, index=pos.index))

    covered_sym = sym.ne("") & sym.isin(list(div_history))
    coupon_pct = pd.to_numeric(
        desc.str.extract(_COUPON_RE, expand=False), errors="coerce")
    coupon_mask = (cls.eq("fixed_income") & ~covered_sym
                   & ((est > 0) | coupon_pct.notna()))
    div_mask = sym.ne("") & ~cls.isin(NON_INCOME_CLASSES) & ~coupon_mask

    frames = []
    eligible = pos[div_mask].copy()
    if not eligible.empty:
        eligible["symbol"] = sym[div_mask]
        agg = (eligible.groupby("symbol", as_index=False)
               .agg(quantity=("quantity", "sum"),
                    market_value=("market_value", "sum"),
                    cost_basis=("cost_basis", "sum"),
                    _qty_costed=("_qty_costed", "sum")))
        asof_ts = pd.Timestamp(asof)
        start_ts = asof_ts - pd.DateOffset(months=12)
        covered, t12 = [], []
        for s in agg["symbol"]:
            hist = div_history.get(s)
            if hist is None:
                covered.append(False)
                t12.append(np.nan)
                continue
            covered.append(True)
            t12.append(0.0 if hist.empty
                       else _t12m_regular(hist, start_ts, asof_ts))
        agg["covered"] = covered
        agg["t12m_per_share"] = t12
        agg["projected"] = agg["t12m_per_share"] * agg["quantity"]  # NaN for uncovered rows by design (0.0 = known non-payer)
        agg["channel"] = "dividend"
        frames.append(agg)

    cpn = pos[coupon_mask].copy()
    if not cpn.empty:
        cpn["symbol"] = [
            _coupon_label(s, d, c, p)
            for s, d, c, p in zip(sym[coupon_mask], desc[coupon_mask],
                                  cusip[coupon_mask],
                                  coupon_pct[coupon_mask])]
        cpn["projected"] = np.where(
            est[coupon_mask] > 0, est[coupon_mask],
            cpn["quantity"] * coupon_pct[coupon_mask] / 100.0)
        cpn_agg = (cpn.groupby("symbol", as_index=False)
                   .agg(quantity=("quantity", "sum"),
                        market_value=("market_value", "sum"),
                        cost_basis=("cost_basis", "sum"),
                        _qty_costed=("_qty_costed", "sum"),
                        projected=("projected", "sum")))
        cpn_agg["covered"] = True
        cpn_agg["t12m_per_share"] = np.nan
        cpn_agg["channel"] = "coupon"
        frames.append(cpn_agg)

    if not frames:
        return pd.DataFrame(columns=_FWD_COLS), _rollup(0.0, 0.0, 0.0, nav)
    out = pd.concat(frames, ignore_index=True)
    out["yield_mv"] = np.where(out["market_value"] > 0,
                               out["projected"] / out["market_value"],
                               np.nan)
    # Income attributable to cost-known lots: projected scaled by the
    # costed share of the quantity (exact for per-share dividend rates
    # and face-scaled coupons alike). Pricing ALL income over PARTIAL
    # cost would show a 19% YoC on a 4% holding.
    _on_cost = out["projected"] * np.where(
        out["quantity"] > 0, out["_qty_costed"] / out["quantity"], 0.0)
    out["yield_cost"] = np.where(out["cost_basis"] > 0,
                                 _on_cost / out["cost_basis"],
                                 np.nan)

    cov = out[out["covered"]]
    covered_mv = float(cov["market_value"].sum())
    projected_12m = float(cov["projected"].fillna(0.0).sum())
    # YoC over cost-known holdings only — carry-forward rows lack cost
    # basis (coerced 0) and would otherwise inflate the ratio.
    known = cov["cost_basis"] > 0
    covered_cost = float(cov.loc[known, "cost_basis"].sum())
    projected_on_cost = float(_on_cost[cov.index][known].fillna(0.0).sum())
    return (out[_FWD_COLS].sort_values("projected", ascending=False)
            .reset_index(drop=True),
            _rollup(projected_12m, covered_mv, covered_cost, nav,
                    projected_on_cost))
