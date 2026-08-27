# terminal/income_service.py
"""Pure data seam for the MERIDIAN Terminal "Income" tab.

Re-expresses app.py._render_income_body (3485-3741). Every income number lives
in the importable, Streamlit-free parsers/income_analytics.py (income_timeseries
/ trailing_income / forward_income / latest_ex_date_through / load_div_history) —
this module reproduces the SAME inputs the Streamlit body uses (the latest
positions_monthly book, the full transactions frame, the dividend-history
mapping, the same sleeve collapse, and date.today()) and shapes the result into
a JSON-native, allow_nan=False-clean view dict. Numbers match Streamlit 1:1 by
construction.

No query params: income is whole-book (it ignores Account / Asset-class).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import config_local as cfg
import theme

from terminal import holdings_service as hs
from terminal.holdings_service import fmt_money

from income_analytics import (forward_income, income_timeseries,
                              latest_ex_date_through, load_div_history,
                              trailing_income)

ACCOUNT_DISPLAY = hs.ACCOUNT_DISPLAY

# Component series colors (theme tokens, kept literal like the other services'
# CHART_* constants). Withholding is the tax/negative channel → coral.
COMP_COLS = ["dividends", "interest", "withholding"]
COMP_LABELS = ["Dividends", "Interest", "Withholding"]
COMP_COLORS = [theme.ACCENT, theme.GAIN, theme.LOSS]
NET_COLOR = theme.TEXT_PRIMARY  # thick cumulative net line

# Categorical palette cycled for the by-account split (mockup donut hues).
_ACCT_PALETTE = ["#4DA3F5", "#2DD4BF", "#8B7CF6", "#E6B450", "#FB7185",
                 "#38BDF8", "#F59E0B", "#818CF8", "#9AA6B6", "#2FD79A"]


def _acct_color(i: int) -> str:
    return _ACCT_PALETTE[i % len(_ACCT_PALETTE)]


# --------------------------------------------------------------------------- #
# Shared compute seams — app.py and this service both call these (Phase D).
# --------------------------------------------------------------------------- #
def latest_book(pm: pd.DataFrame) -> "tuple[pd.DataFrame, pd.Timestamp | None]":
    """Latest-statement book slice: rows at max(statement_date). Empty-safe
    (book_ts is None when pm is empty). Shared by app.py and build_income_view."""
    if pm.empty:
        return pm, None
    book_ts = pm["statement_date"].max()
    return pm[pm["statement_date"] == book_ts], book_ts


def ytd_income(inc_ts: pd.DataFrame, today_ts: pd.Timestamp) -> float:
    """Calendar-YTD net income: sum of inc_ts['net'] from Jan 1 of today's year."""
    return float(inc_ts.loc[inc_ts.index >= pd.Timestamp(today_ts.year, 1, 1),
                            "net"].sum())


def forward_payers(fwd_df: pd.DataFrame, nav: float) -> pd.DataFrame:
    """Dividend-paying rows (projected > 0), in the engine's sorted order, with a
    pct_mv (market_value / NAV) column. Feeds the top-payers chart and the detail
    table on both UIs."""
    payers = fwd_df[fwd_df["projected"] > 0].copy()
    payers["pct_mv"] = (payers["market_value"] / nav) if nav else float("nan")
    return payers


def components_frame(inc_ts: pd.DataFrame, view: str) -> pd.DataFrame:
    """Component (dividends/interest/withholding) frame at the requested
    granularity. 'yearly' resamples to calendar-year starts; 'monthly' (and the
    cumulative view, which cumsums this) return a copy of inc_ts — callers hold it
    alongside the other views, so it must not alias the shared frame."""
    if view == "yearly":
        return inc_ts.resample("YS").sum() if not inc_ts.empty else inc_ts
    return inc_ts.copy()


def by_account_frame(ts_by: pd.DataFrame, view: str) -> pd.DataFrame:
    """Long [month, account, net] frame with account_id mapped to display names.
    'yearly' rolls months up to calendar-year starts (sum per account); 'monthly'
    (and the cumulative view) return the per-month frame."""
    df = ts_by.reset_index()
    df["account"] = (df["account_id"].map(ACCOUNT_DISPLAY)
                     .fillna(df["account_id"]).astype(str))
    if view == "yearly":
        df["month"] = df["month"].dt.to_period("Y").dt.to_timestamp()
        df = df.groupby(["month", "account"], as_index=False)["net"].sum()
    return df


# --------------------------------------------------------------------------- #
# Section A — Income received (KPIs + chart). Mirror app.py 3505-3594.
# --------------------------------------------------------------------------- #
def _stacked_from_frame(frame: pd.DataFrame, xfmt: str) -> dict:
    """Components stacked-bar view: x = period labels, one series per component."""
    x = [pd.Timestamp(i).strftime(xfmt) for i in frame.index]
    series = [{"name": lab, "color": col,
               "values": [float(v) for v in frame[c].tolist()]}
              for c, lab, col in zip(COMP_COLS, COMP_LABELS, COMP_COLORS)]
    return {"x": x, "series": series, "mode": "stacked"}


def _cumulative_components(inc_ts: pd.DataFrame) -> dict:
    """Cumulative line view: each component's running sum + a thick Net line
    (mirror app.py 3572-3583)."""
    x = [pd.Timestamp(i).strftime("%b %Y") for i in inc_ts.index]
    series = [{"name": lab, "color": col,
               "values": [float(v) for v in inc_ts[c].cumsum().tolist()]}
              for c, lab, col in zip(COMP_COLS, COMP_LABELS, COMP_COLORS)]
    series.append({"name": "Net (cumulative)", "color": NET_COLOR, "width": 4,
                   "values": [float(v) for v in inc_ts["net"].cumsum().tolist()]})
    return {"x": x, "series": series, "mode": "line"}


def _components_views(inc_ts: pd.DataFrame) -> dict:
    return {
        "monthly": _stacked_from_frame(components_frame(inc_ts, "monthly"), "%b %Y"),
        "yearly": _stacked_from_frame(components_frame(inc_ts, "yearly"), "%Y"),
        "cumulative": _cumulative_components(inc_ts),
    }


def _account_stack(df: pd.DataFrame, xfmt: str, mode: str,
                   cumulative: bool = False) -> dict:
    """By-account view from a long [month, account, net] frame. Every account is
    a series aligned to the union of periods (0-filled), so stacked bars and
    cumulative lines render off one shared x. Cumulative cumsums each account's
    0-filled per-period net (mirror app.py 3556-3561's per-account cumsum).

    The 0-fill makes a cumulative account line hold flat across months it had no
    income (vs Streamlit/Plotly, which draws a sloped connector between an
    account's own points). The per-account FINAL cumulative totals are identical
    either way — only the inter-point line shape differs, which the shared
    single-x SVG axis requires and the "numbers, not pixels" mandate permits."""
    if df.empty:
        return {"x": [], "series": [], "mode": mode}
    periods = sorted(df["month"].unique())
    x = [pd.Timestamp(p).strftime(xfmt) for p in periods]
    accounts = sorted(df["account"].astype(str).unique())
    pivot = (df.pivot_table(index="month", columns="account", values="net",
                            aggfunc="sum", fill_value=0.0)
             .reindex(periods, fill_value=0.0))
    series = []
    for i, acct in enumerate(accounts):
        col = pivot[acct] if acct in pivot.columns else pd.Series(0.0, index=periods)
        vals = col.cumsum() if cumulative else col
        series.append({"name": str(acct), "color": _acct_color(i),
                       "values": [float(v) for v in vals.tolist()]})
    return {"x": x, "series": series, "mode": mode}


def _by_account_views(ts_by: pd.DataFrame) -> dict:
    if ts_by.empty:
        return {"monthly": {"x": [], "series": [], "mode": "stacked"},
                "yearly": {"x": [], "series": [], "mode": "stacked"},
                "cumulative": {"x": [], "series": [], "mode": "line"}}
    df = by_account_frame(ts_by, "monthly")
    monthly = _account_stack(df, "%b %Y", "stacked")
    cumulative = _account_stack(df, "%b %Y", "line", cumulative=True)
    dfy = by_account_frame(ts_by, "yearly")
    yearly = _account_stack(dfy, "%Y", "stacked")
    return {"monthly": monthly, "yearly": yearly, "cumulative": cumulative}


def _received(inc_ts: pd.DataFrame, ts_by: pd.DataFrame,
              transactions: pd.DataFrame, projected_12m: float,
              today_ts: pd.Timestamp) -> dict:
    """Section A: actual income KPIs + the component/by-account chart."""
    if inc_ts.empty:
        return {
            "empty": True,
            "empty_message": ("No dividend / interest / withholding "
                              "transactions in the selected scope yet."),
            "kpis": [], "chart": None,
        }

    t12m_actual = trailing_income(transactions, today_ts)
    ytd_actual = ytd_income(inc_ts, today_ts)

    kpis = [
        {"label": "Trailing-12M income (actual)", "value": fmt_money(t12m_actual),
         "sub": "trailing 365 days · net of withholding"},
        {"label": "YTD income (actual)", "value": fmt_money(ytd_actual),
         "sub": f"calendar {today_ts.year}"},
        {"label": "Projected next-12M", "value": fmt_money(projected_12m),
         "sub": "dividends + bond coupons — see Forward income"},
    ]
    return {
        "empty": False, "empty_message": None, "kpis": kpis,
        "chart": {"components": _components_views(inc_ts),
                  "by_account": _by_account_views(ts_by)},
    }


# --------------------------------------------------------------------------- #
# Section B — Forward income (KPIs + top payers + detail). Mirror app.py
# 3596-3676.
# --------------------------------------------------------------------------- #
def _pct(v: float, decimals: int = 2) -> str:
    return "-" if pd.isna(v) else f"{v * 100:.{decimals}f}%"


def _forward(book: pd.DataFrame, div_hist: dict, fwd_df: pd.DataFrame,
             roll: dict, book_ts: pd.Timestamp, today_ts: pd.Timestamp) -> dict:
    if not div_hist:
        return {
            "available": False,
            "unavailable_message": ("No dividend history fetched yet — run "
                                    "`py parsers/fetch_dividends.py --holdings "
                                    "--write`."),
            "kpis": [], "nav_caption": None, "payers_empty": True,
            "payers_message": None, "top_chart": None, "detail": None,
            "history_through_caption": None,
        }

    nav = roll["nav"]
    kpis = [
        {"label": "Projected 12M income",
         "value": fmt_money(roll["projected_12m"]), "sub": None},
        {"label": "Yield (covered MV)",
         "value": _pct(roll["yield_on_covered_mv"]), "sub": None},
        {"label": "Yield on cost", "value": _pct(roll["yield_on_covered_cost"]),
         "sub": "holdings with known cost basis only"},
        {"label": "Coverage (% of NAV)",
         "value": _pct(roll["coverage_pct_nav"], 0),
         "sub": ("options, cash and unfetched tickers are uncovered; bonds "
                 "project via coupons")},
    ]
    nav_caption = (
        f"Yields and coverage are measured against the book's live-marked NAV "
        f"${nav:,.0f} (as of {book_ts.strftime('%b %d, %Y')}) — the marked "
        f"basis the Holdings value uses, not the return-basis NAV the "
        f"Performance tab reconciles.")

    payers = forward_payers(fwd_df, nav)
    if payers.empty:
        top_chart, detail, payers_message = None, None, (
            "No projected dividend income in the covered set.")
    else:
        payers_message = None
        top = payers.head(15)
        top_chart = {"bars": [{"x": str(s), "v": float(v)}
                              for s, v in zip(top["symbol"], top["projected"])]}
        detail = _detail_table(payers)

    ex_max = latest_ex_date_through(div_hist, today_ts)
    history_caption = (
        f"Dividend history through {ex_max.strftime('%b %d, %Y')} — book as of "
        f"{book_ts.strftime('%b %d, %Y')}." if ex_max is not None else None)

    return {
        "available": True, "unavailable_message": None, "kpis": kpis,
        "nav_caption": nav_caption,
        "payers_empty": payers.empty, "payers_message": payers_message,
        "top_chart": top_chart, "detail": detail,
        "history_through_caption": history_caption,
    }


_DETAIL_COLUMNS = ["Symbol", "Projected 12M", "T12M / share", "Shares / face",
                   "Yield", "Yield on cost", "Market value", "% of NAV"]


def _detail_table(payers: pd.DataFrame) -> dict:
    """The forward detail table with Streamlit's exact formatting (app.py
    3643-3668). All payers (not just the top 15), sorted by projected desc.
    pct_mv is precomputed by forward_payers()."""
    def money(v):
        return "-" if pd.isna(v) else f"${v:,.0f}"

    rows = []
    for _, r in payers.iterrows():
        rows.append({
            "Symbol": str(r["symbol"]),
            "Projected 12M": money(r["projected"]),
            # Per-share rates keep cents (SGOV's $3.87 ≠ $4); coupon rows "-".
            "T12M / share": ("-" if pd.isna(r["t12m_per_share"])
                             else f"${r['t12m_per_share']:,.2f}"),
            "Shares / face": ("-" if pd.isna(r["quantity"])
                              else f"{r['quantity']:,.0f}"),
            "Yield": _pct(r["yield_mv"]),
            "Yield on cost": _pct(r["yield_cost"]),
            "Market value": money(r["market_value"]),
            "% of NAV": _pct(r["pct_mv"]),
        })
    return {"columns": _DETAIL_COLUMNS, "rows": rows}


# --------------------------------------------------------------------------- #
# Methodology expander (port app.py 3704-3741 markdown → HTML).
# --------------------------------------------------------------------------- #
_METHODOLOGY = (
    "<ul>"
    "<li><b>Actuals:</b> dividend / interest / withholding rows from the "
    "transaction ledger, grouped by settlement month. Gross income; withholding "
    "(foreign tax) is summed as recorded (mostly small reclaims positive, "
    "occasional tax negative), so the stacked bars read net. Return-of-capital "
    "distributions on held shares (<code>principal_pmt</code> — NEOS-style "
    "funds, special payouts) count as dividends: ROC is tax character, not a "
    "different kind of yield; cash-in-lieu and bond principal are excluded. "
    "<code>reinvestment</code> rows are the buy side of DRIPs and are excluded — "
    "the cash dividend row is already counted.</li>"
    "<li><b>Per-holding actuals are not shown</b>: ~22% of dividend dollars in "
    "the ledger still lack a ticker, so per-name attribution comes from the "
    "forward model instead.</li>"
    "<li><b>Forward income</b> = trailing-12-month per-share cash dividends "
    "(Polygon, ex-date basis) × current shares. One-time / special distributions "
    "(Polygon frequency 0 or type SC) are excluded from the annualization — a "
    "one-off payout is not a run-rate. Reacts to dividend cuts with up to a "
    "year's lag by construction.</li>"
    "<li><b>Bond coupons:</b> fixed-income rows without a dividend file "
    "(bare-CUSIP Treasury rungs, bond funds) project the statement's est. annual "
    "income when present, else face × coupon parsed from the description. "
    "Going-concern ladder: no maturity pro-rating — a maturing rung is assumed "
    "rolled into similar paper.</li>"
    "<li><b>Coverage:</b> any symbol-bearing holding with a fetched history file "
    "— class labels are not trusted (JPM files SGOV under fixed income; "
    "TLH-account names carry a display class) — plus coupon-priced bonds. "
    "Options, cash and never-fetched tickers are <i>uncovered</i> — excluded from "
    "yield math but counted in NAV, so the coverage tile is honest.</li>"
    "<li><b>Renames</b> are spliced at fetch time via TICKER_HISTORY (prior "
    "symbol's dividends before the effective date).</li>"
    "<li>Book = latest month in the (broker-filtered) positions panel, including "
    "interim months when loaded; the Holdings (Account / Asset-class) filter does "
    "not apply here, and coverage / yields use the book's live-marked NAV. A "
    "ticker first bought mid-interim stays uncovered until it appears in a "
    "statement month and the history is refreshed.</li>"
    "</ul>"
)


def forward_rollup(
        pm: pd.DataFrame, div_hist: dict, today_date,
) -> "tuple[pd.DataFrame, dict, pd.Timestamp | None]":
    """(fwd_df, roll, book_ts) for the latest-statement book — the single
    forward-income seam build_income_view and the AI facts reducer share
    (latest-book slice, sleeve collapse, and the empty-book zero rollup
    live here once)."""
    book, book_ts = latest_book(pm)
    if book.empty:
        return book, {"projected_12m": 0.0,
                      "yield_on_covered_mv": float("nan"),
                      "yield_on_covered_cost": float("nan"),
                      "covered_mv": 0.0, "nav": 0.0,
                      "coverage_pct_nav": float("nan")}, book_ts
    fwd_df, roll = forward_income(
        book, div_hist, today_date,
        sleeves={cfg.TREASURY_LADDER_ACCOUNT_ID: "Treasury Ladder",
                 cfg.TLH_ACCOUNT_ID: "Tax Loss Harvesting"})
    return fwd_df, roll, book_ts


# --------------------------------------------------------------------------- #
# View assembly.
# --------------------------------------------------------------------------- #
def build_income_view(frames: hs.Frames, *, asof: "date | None" = None) -> dict:
    """Assemble the GET /api/income contract. Pure given frames.

    Mirrors app.py._render_income_body: the full transactions ledger feeds the
    actual-income series; the latest positions_monthly book + dividend-history
    mapping feed the forward model (sleeve-collapsed for display, full-book for
    the rollup). Whole-book — no Account / Asset-class filter.

    ``asof`` defaults to ``date.today()`` — the same wall-clock the Streamlit
    body uses, so the live API and the AppTest parity gate agree. Tests pin it to
    a fixed date so the golden snapshot stays deterministic (the rendered Income
    surface — trailing-12M, YTD, the whole forward projection — is genuinely
    today-dependent, unlike the other tabs').
    """
    transactions = frames.transactions
    pm = frames.positions_monthly

    inc_ts = income_timeseries(transactions)
    ts_by = income_timeseries(transactions, by="account_id")

    # Latest book = rows at the max statement_date (app.py 3488-3490 — exact
    # equality, not a calendar-month slice). `book` stays local because later
    # sections still receive it (`_forward(book, ...)`).
    book, book_ts = latest_book(pm)

    div_hist = load_div_history(Path(frames.data_dir))
    today_date = asof or date.today()
    today_ts = pd.Timestamp(today_date).normalize()

    # Forward rollup is full-book; the frame is sleeve-collapsed for display
    # (app.py 3495-3498). Computed once — section A's "Projected next-12M" KPI
    # reads the same rollup.
    fwd_df, roll, _bts = forward_rollup(pm, div_hist, today_date)

    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)

    meta = {
        "accounts": acct_opts,
        "classes": class_opts,
        "brokers": broker_opts,
        "available_dates": list(frames.available_dates),
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "filter": {"account": "all", "asset_class": "all", "broker": "all"},
        "book_date": (pd.Timestamp(book_ts).strftime("%b %d, %Y")
                      if book_ts is not None else None),
    }

    return {
        "meta": meta,
        "caption": ("Income is computed on the full broker-filtered book — the "
                    "Holdings (Account / Asset-class) filter does not apply to "
                    "this tab."),
        "received": _received(inc_ts, ts_by, transactions,
                              roll["projected_12m"], today_ts),
        "forward": _forward(book, div_hist, fwd_df, roll, book_ts, today_ts),
        "methodology": _METHODOLOGY,
    }
