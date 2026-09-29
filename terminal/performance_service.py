# terminal/performance_service.py
"""Pure data seam for the MERIDIAN Terminal Performance tab.

Re-expresses app.py._render_performance_body (2539-3119) and its load-time
helpers (load_twr 729-772, monthly_totals 859, filtered_twr_frame 910-953,
account_onboarding_events 956) as a pure, importable module reusing parsers/
and terminal.holdings_service. Imports no Streamlit. Every returned value is
JSON-native so the view can be json.dumps'd without leaks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config_local as cfg
from terminal import holdings_service as hs
from terminal.holdings_service import Frames, fmt_money

from compute_twr import link_returns, annualize  # noqa: F401 (parity import)
from risk_metrics import aggregate_periodic_returns
from risk_bundle import daily_portfolio_returns, synthesize_monthly_returns
from interim_stub import (InterimStub, append_stub, chain, stub_block,  # noqa: F401
                          to_date_cagr, to_date_span)

logger = logging.getLogger(__name__)

ACCOUNT_DISPLAY = cfg.ACCOUNT_DISPLAY
SKIP_FROM_TWR_SUMMARY = cfg.SKIP_FROM_TWR_SUMMARY
SYNTHETIC_ONBOARDING = cfg.SYNTHETIC_ONBOARDING

_TWR_COLS = ["month", "statement_date", "prev_stmt_date", "return_pct",
             "cum_return", "twr_dd_pct"]


# --------------------------------------------------------------------------- #
# TWR frame preparation — mirror app.py.load_twr (729-772).
# --------------------------------------------------------------------------- #
def _prepare_portfolio_twr(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the computed columns load_twr derives onto the raw twr_portfolio
    frame (cum_return / wealth_index / twr_dd_pct + parsed month/dates +
    month_end). Empty frame in => empty frame out."""
    if raw is None or raw.empty or "return_pct" not in raw.columns:
        return pd.DataFrame()
    port = raw.copy()
    port["month"] = pd.PeriodIndex(port["month"], freq="M")
    port["statement_date"] = pd.to_datetime(port["statement_date"])
    if "prev_stmt_date" in port.columns:
        port["prev_stmt_date"] = pd.to_datetime(port["prev_stmt_date"])
    wealth = (1.0 + port["return_pct"].fillna(0.0)).cumprod()
    port["cum_return"] = wealth - 1.0
    port["wealth_index"] = wealth
    port["wealth_peak"] = wealth.cummax()
    port["twr_dd_pct"] = (wealth / wealth.cummax() - 1.0) * 100.0
    port["month_end"] = port["month"].dt.to_timestamp("M").dt.normalize()
    return port


def _load_twr_account(data_dir) -> pd.DataFrame:
    """Mirror load_twr's twr_monthly.csv branch (per-account TWR).

    Not called by ``build_performance_view`` (it reads the demo-overlaid,
    broker-narrowed ``frames.twr_account`` instead — a fresh data_dir read
    here would bypass the ``apply_global_filters`` choke-point and leak real
    per-account $ under a test-only broker selection). Kept as the standalone
    parser ``holdings_service._load_twr_account`` mirrors, and still covered
    directly by tests."""
    path = Path(data_dir) / "twr_monthly.csv"
    if not path.exists():
        return pd.DataFrame()
    acct = pd.read_csv(path)
    if acct.empty:
        return acct
    acct["month"] = pd.PeriodIndex(acct["month"], freq="M")
    acct["statement_date"] = pd.to_datetime(acct["statement_date"])
    return acct


# --------------------------------------------------------------------------- #
# Headline KPIs — mirror app.py 2707-2753 + _portfolio_headline (9656-9667).
# --------------------------------------------------------------------------- #
def headline_raw(twr_view: pd.DataFrame, irr_table: pd.DataFrame,
                 holdings_filter_active: bool, *,
                 stub: "InterimStub | None" = None) -> dict | None:
    """Raw Performance headline numbers (the shared compute for BOTH UIs):
    cumulative/annualized TWR, max drawdown + its month, best/worst month +
    returns, and the portfolio IRR lookup (NaN under a Holdings filter). app.py
    renders these via theme.kpi_card (using only the EXTRAS — it keeps
    _portfolio_headline() for cum/ann/mdd/n, the tape-shared source); the
    _headline formatter below wraps the full dict into cards. None on empty."""
    if twr_view is None or twr_view.empty:
        return None
    port = twr_view
    cum = float(port["cum_return"].iloc[-1])
    n = int(port["return_pct"].notna().sum())
    ann = (1.0 + cum) ** (12.0 / n) - 1.0 if n > 0 else float("nan")
    mdd = float(port["twr_dd_pct"].min())
    has_ret = bool(port["return_pct"].notna().any())
    worst = port.loc[port["return_pct"].idxmin()] if has_ret else None
    best = port.loc[port["return_pct"].idxmax()] if has_ret else None
    if holdings_filter_active:
        irr = float("nan")
    else:
        row = (irr_table[irr_table["account_id"] == "PORTFOLIO"]
               if (irr_table is not None and not irr_table.empty
                   and "account_id" in irr_table.columns) else pd.DataFrame())
        irr = float(row.iloc[0]["irr"]) if len(row) else float("nan")
    out = {
        "cum": cum, "ann": ann, "mdd": mdd, "n": n,
        "max_dd_month": port.loc[port["twr_dd_pct"].idxmin(), "month"],
        "start_stmt_date": port.iloc[0]["statement_date"],
        "worst_ret": float(worst["return_pct"]) if worst is not None else None,
        "worst_month": worst["month"] if worst is not None else None,
        "best_ret": float(best["return_pct"]) if best is not None else None,
        "best_month": best["month"] if best is not None else None,
        "irr": irr,
    }
    if stub is not None:
        # To-date figures (spec 2026-08-22): the statement-basis keys above
        # stay; cum chains the provisional period, ann is day-count based.
        _origin, days = to_date_span(port, stub)
        cum_td = chain(cum, stub.return_pct)
        out.update({"to_date": stub.end_date.strftime("%Y-%m-%d"),
                    "cum_to_date": cum_td, "days_to_date": days,
                    "ann_to_date": to_date_cagr(cum_td, days), "stub": stub})
    return out


def _headline(twr_view: pd.DataFrame, irr_table: pd.DataFrame,
              holdings_filter_active: bool,
              broker_scope: tuple[str, ...] | None = None, *,
              stub: "InterimStub | None" = None) -> list[dict]:
    """Headline cards. Thin formatter over ``headline_raw`` — the numbers live
    there now. Output dict unchanged (golden anchor).

    ``broker_scope``: ``None`` for the canonical whole-book view; a tuple of
    broker labels when the caller has narrowed to a non-real selection, in
    which case the IRR sub-label reads "Money-weighted · {labels}" instead of
    the default text.
    """
    hd = headline_raw(twr_view, irr_table, holdings_filter_active, stub=stub)
    if hd is None:
        return []
    start_lbl = pd.Timestamp(hd["start_stmt_date"]).strftime("%b %Y")
    to_date = "cum_to_date" in hd
    cum_val = hd["cum_to_date"] if to_date else hd["cum"]
    ann_val = hd["ann_to_date"] if to_date else hd["ann"]
    cum_sub = (f"Since {start_lbl} · to {hd['to_date']} · provisional"
               if to_date else f"Since {start_lbl}")
    ann_sub = (f"{hd['n']} months + {hd['stub'].days}d · provisional"
               if to_date else f"{hd['n']} months")
    if holdings_filter_active:
        irr_sub = "Holdings filter active"
    elif broker_scope:
        lbl = " + ".join(broker_scope)
        irr_sub = (f"Money-weighted · {lbl}" if not pd.isna(hd["irr"])
                   else "Money-weighted · n/a for this selection")
    else:
        irr_sub = "Money-weighted"

    def _card(key, label, value, color, sub):
        return {"key": key, "label": label, "value": value,
                "color": color, "sub": sub}

    cards = [
        _card("cum_twr", "Cumulative TWR", hs._spct(cum_val * 100.0),
              hs._pnl_color(cum_val), cum_sub),
        _card("ann_twr", "Annualized TWR", hs._spct(ann_val * 100.0),
              None, ann_sub),
        _card("irr", "IRR (annualized)",
              hs._spct(hd["irr"] * 100.0) if not pd.isna(hd["irr"]) else "—",
              None, irr_sub),
        _card("max_dd", "Max drawdown (TWR)", hs._spct(hd["mdd"]),
              "loss", pd.Timestamp(hd["max_dd_month"].to_timestamp()).strftime("%b %Y")),
    ]
    if hd["worst_ret"] is not None and hd["best_ret"] is not None:
        bw = (f"{hd['worst_ret'] * 100:+.1f}% / {hd['best_ret'] * 100:+.1f}%")
        bw_sub = (f"{hd['worst_month'].strftime('%b %Y')} / "
                  f"{hd['best_month'].strftime('%b %Y')}")
    else:
        bw, bw_sub = "—", ""
    cards.append(_card("best_worst", "Best / worst month", bw, None, bw_sub))
    return cards


# --------------------------------------------------------------------------- #
# External cash flows — mirror app.py 2755-2811.
# --------------------------------------------------------------------------- #
def cashflows_raw(transactions: pd.DataFrame, twr_view: pd.DataFrame,
                  selected_account_ids: list, account_active: bool,
                  class_active: bool) -> dict | None:
    """Raw external-cashflow sums (shared compute for BOTH UIs). None when
    hidden (class filter / no flow_scope / empty transactions). app.py will
    format these with fmt_money + its own captions; _cashflows below wraps
    them with fmt_money + the synthetic note. View dict unchanged (golden)."""
    if (transactions is None or transactions.empty
            or "flow_scope" not in transactions.columns or class_active):
        return None
    tx_scope = (transactions[transactions["account_id"].isin(selected_account_ids)]
                if account_active else transactions)
    ext = tx_scope[tx_scope["flow_scope"] == "external"]
    deposits = float(ext.loc[ext["amount"] > 0, "amount"].sum()) if len(ext) else 0.0
    withdrawals = float(ext.loc[ext["amount"] < 0, "amount"].sum()) if len(ext) else 0.0
    synth_total = (float(twr_view["synthetic_flow"].sum())
                   if "synthetic_flow" in twr_view.columns else 0.0)
    return {"deposits": deposits, "withdrawals": withdrawals,
            "net": deposits + withdrawals, "synth_total": synth_total,
            "n_deposits": int((ext["amount"] > 0).sum()),
            "n_withdrawals": int((ext["amount"] < 0).sum())}


def _cashflows(transactions: pd.DataFrame, twr_view: pd.DataFrame,
               selected_account_ids, account_active: bool,
               class_active: bool) -> dict | None:
    """Mirror app.py 2767-2802. Hidden (None) under a class filter; scoped to
    selected account ids under an account filter. Thin formatter over
    ``cashflows_raw`` — the numbers live there now."""
    raw = cashflows_raw(transactions, twr_view, selected_account_ids,
                        account_active, class_active)
    if raw is None:
        return None
    synth_note = (
        f"Plus {fmt_money(raw['synth_total'])} synthetic onboarding (NAV of accounts "
        f"whose money predates the statement archive — counted as a deposit by "
        f"IRR but not visible in the real-flow tiles above)."
        if raw["synth_total"] > 0 else None)
    return {
        "deposits": {"value": fmt_money(raw["deposits"]), "n": raw["n_deposits"]},
        "withdrawals": {"value": fmt_money(raw["withdrawals"]), "n": raw["n_withdrawals"]},
        "net": {"value": fmt_money(raw["net"])},
        "synthetic_note": synth_note,
    }


# --------------------------------------------------------------------------- #
# Filter resolution — map opaque option ids to bucket/class filters + account
# scope (mirror app.py 1842-1854).
# --------------------------------------------------------------------------- #
def _resolve_filter(frames: Frames, snap_all, account: str | list[str],
                    asset_class: str | list[str]):
    """Map opaque option ids -> (bucket_filter, class_filter, selected_account_ids,
    account_active, class_active). ``account``/``asset_class`` accept a scalar
    id/'all' OR a list of ids (multi-select). bucket/class filters are the full
    choice lists when unfiltered, else the union of the selected buckets/classes.
    ``*_active`` is True only when the resolved set is a PROPER subset of the
    choices (mirror app.py 1860-1861 — selecting every option reads as 'all')."""
    _, acct_by_id = hs._account_options(snap_all)
    _, class_by_id = hs._class_options(snap_all)
    positions = frames.positions
    bucket_choices = sorted(positions["bucket"].dropna().astype(str).unique())
    class_choices = sorted(positions["asset_class"].dropna().astype(str).unique())
    acct_ids = hs._normalize_filter_ids(account)
    class_ids = hs._normalize_filter_ids(asset_class)
    bucket_sel = hs._resolve_ids(acct_ids, acct_by_id)
    class_sel = hs._resolve_ids(class_ids, class_by_id)
    bucket_filter = bucket_sel if bucket_sel else bucket_choices
    class_filter = class_sel if class_sel else class_choices
    if bucket_sel:
        selected_account_ids = (positions.loc[positions["bucket"].isin(bucket_sel),
                                              "account_id"].astype(str).unique().tolist())
    else:
        selected_account_ids = positions["account_id"].astype(str).unique().tolist()
    account_active = set(bucket_filter) != set(bucket_choices)
    class_active = set(class_filter) != set(class_choices)
    return (bucket_filter, class_filter, selected_account_ids,
            account_active, class_active)


# --------------------------------------------------------------------------- #
# Onboarding events + cumulative-TWR / drawdown series — mirror app.py
# 2813-2897 + account_onboarding_events (956-974).
# --------------------------------------------------------------------------- #
def _onboarding_events(positions_monthly: pd.DataFrame) -> pd.DataFrame:
    """Mirror app.py.account_onboarding_events (956-974): first month each
    account appears + its NAV (join_value) + account_display. account_type is
    optional here (only first_month/account_display/join_value are consumed)."""
    if positions_monthly is None or positions_monthly.empty:
        return pd.DataFrame(columns=["account_id", "first_month", "join_value",
                                     "account_display"])
    first = (positions_monthly.groupby("account_id")["month"].min()
             .reset_index().rename(columns={"month": "first_month"}))
    join_value = []
    for _, r in first.iterrows():
        sub = positions_monthly[(positions_monthly["account_id"] == r["account_id"])
                                & (positions_monthly["month"] == r["first_month"])]
        join_value.append(float(sub["market_value"].sum()))
    first["join_value"] = join_value
    first["account_display"] = (first["account_id"].map(ACCOUNT_DISPLAY)
                                .fillna(first["account_id"]))
    return first.sort_values("first_month").reset_index(drop=True)


def events_grouped_raw(events_df: pd.DataFrame, selected_account_ids: list,
                       account_active: bool, class_active: bool) -> pd.DataFrame:
    """events.iloc[1:] grouped by (first_month, account_display) — the shared
    onboarding-marker compute. Empty under a class filter; scoped under an
    account filter. Takes the raw onboarding-events frame so BOTH UIs call it."""
    if class_active or events_df is None or events_df.empty:
        return events_df.iloc[0:0] if events_df is not None else pd.DataFrame()
    ev = events_df
    if account_active:
        ev = ev[ev["account_id"].astype(str).isin(selected_account_ids)]
    return (ev.iloc[1:]
            .groupby(["first_month", "account_display"], as_index=False)
            .agg(join_value=("join_value", "sum"))
            .sort_values(["first_month", "account_display"]))


def _events_grouped(frames: Frames, selected_account_ids,
                    account_active: bool, class_active: bool) -> pd.DataFrame:
    """events.iloc[1:] grouped by (first_month, account_display) (app.py
    2843-2853). Empty under a class filter; scoped under an account filter.
    Thin wrapper over ``events_grouped_raw`` — the shared compute lives there."""
    return events_grouped_raw(_onboarding_events(frames.positions_monthly),
                              selected_account_ids, account_active, class_active)


def _cum_twr_section(twr_view: pd.DataFrame, eg: pd.DataFrame, *,
                     stub: "InterimStub | None" = None) -> dict:
    view = append_stub(twr_view, stub) if stub is not None else twr_view
    pts = []
    for _, r in view.iterrows():
        p = {"x": pd.Timestamp(r["month_end"]).strftime("%Y-%m-%d"),
             "v": float(r["cum_return"] * 100.0)}
        if bool(r.get("provisional", False)):
            p["provisional"] = True          # only the stub row carries the key
        pts.append(p)
    markers = []
    for fm, g in eg.groupby("first_month", sort=True):
        x = fm.to_timestamp("M").strftime("%Y-%m-%d")
        label = "<br>".join(f"+{n}" for n in g["account_display"])
        markers.append({"x": x, "label": label.replace("<br>", " · ")})
    final_cum = float(view["cum_return"].iloc[-1]) if not view.empty else 0.0
    grown = 100_000.0 * (1.0 + final_cum)
    sub = f"$100k grown to {fmt_money(grown)} · dashed lines mark account onboarding"
    if stub is not None:
        sub += f" · last segment provisional (to {stub.end_date:%Y-%m-%d})"
    return {"head": {"title": "Cumulative wealth index (TWR)", "sub": sub},
            "points": pts, "markers": markers}


def _drawdown_section(twr_view: pd.DataFrame) -> dict:
    pts = [{"x": pd.Timestamp(r["month_end"]).strftime("%Y-%m-%d"),
            "dd": float(r["twr_dd_pct"])} for _, r in twr_view.iterrows()]
    trough_v = float(twr_view["twr_dd_pct"].min()) if not twr_view.empty else 0.0
    trough_m = (pd.Timestamp(twr_view.loc[twr_view["twr_dd_pct"].idxmin(), "month_end"])
                .strftime("%b %Y") if not twr_view.empty else "")
    return {"head": {"title": "Underwater plot",
                     "sub": f"deepest {hs._spct(trough_v)} in {trough_m}"},
            "points": pts,
            "trough": {"label": hs._spct(trough_v), "when": trough_m}}


# --------------------------------------------------------------------------- #
# Periodic returns (M/Q/Y) + win-rate — mirror app.py 2899-2934.
# --------------------------------------------------------------------------- #
_GRAN_FREQ = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}
_GRAN_XFMT = {"monthly": "%Y-%m", "quarterly": "%Y-%m", "yearly": "%Y"}


def _periodic(twr_view: pd.DataFrame) -> dict:
    base = twr_view.dropna(subset=["return_pct"]).copy() if not twr_view.empty \
        else twr_view
    out = {}
    # win-rate from the monthly series (app.py mockup caption convention)
    if base is not None and not base.empty:
        wins = int((base["return_pct"] >= 0).sum())
        tot = int(len(base))
        winrate = f"{wins} / {tot} ({round(wins / tot * 100)}%)" if tot else "—"
    else:
        winrate = "—"
    for gran, freq in _GRAN_FREQ.items():
        if base is None or base.empty:
            out[gran] = {"bars": [], "winrate": winrate}
            continue
        if freq == "M":
            agg_ret = base["return_pct"].reset_index(drop=True)
            agg_dt = base["month_end"].reset_index(drop=True)
        else:
            agg_ret, agg_dt = aggregate_periodic_returns(
                base["return_pct"], base["month_end"], freq)
            agg_ret = pd.Series(agg_ret).reset_index(drop=True)
            agg_dt = pd.Series(agg_dt).reset_index(drop=True)
        bars = []
        for d, v in zip(agg_dt, agg_ret):
            if pd.isna(v):  # never ship a NaN v into allow_nan=False JSON
                continue
            bars.append({"x": pd.Timestamp(d).strftime(_GRAN_XFMT[gran]),
                         "v": float(v * 100.0)})
        out[gran] = {"bars": bars, "winrate": winrate}
    return out


# --------------------------------------------------------------------------- #
# Per-account TWR table — mirror app.py 2936-3031.
# --------------------------------------------------------------------------- #
def per_account_raw(twr_account: pd.DataFrame, irr_table: pd.DataFrame,
                    selected_account_ids: list, account_active: bool) -> pd.DataFrame:
    """Raw per-account TWR rows (shared compute for BOTH UIs; mirror the loop in
    the old _per_account). Compounded cum TWR, <12-month-gated annualized + IRR,
    start/end NAV, net flow, synthetic flag. Returns DECIMAL fractions (not
    percents) — each consumer multiplies by 100 and formats. Sorted by cum TWR
    descending (stable, to match the original's tie ordering)."""
    cols = ["account_id", "account_label", "first_month", "last_month",
            "months", "start_nav", "end_nav", "net_flow", "cum_twr",
            "ann_twr", "irr", "is_synthetic"]
    if twr_account is None or twr_account.empty:
        return pd.DataFrame(columns=cols)
    irr_by_acct = ({str(r["account_id"]): float(r["irr"])
                    for _, r in irr_table.iterrows()
                    if r["account_id"] != "PORTFOLIO"}
                   if (irr_table is not None and not irr_table.empty) else {})
    scoped = (twr_account[twr_account["account_id"].astype(str)
                          .isin(selected_account_ids)]
              if account_active else twr_account)
    rows = []
    for acct, g in scoped.groupby("account_id"):
        if acct in SKIP_FROM_TWR_SUMMARY:
            continue
        g = g.sort_values("month")
        valid = g.dropna(subset=["return_pct"])
        if len(valid) == 0:
            continue
        total = float(np.prod(1.0 + valid["return_pct"]) - 1.0)
        n_mo = int(len(valid))
        # A linked TWR below -100% (Dietz small-denominator artifact on a
        # drained account) makes the annualization base negative — a
        # fractional power of it is complex and would 500 the endpoint.
        # Annualized return is undefined there; render as n/a.
        ann = ((1.0 + total) ** (12.0 / n_mo) - 1.0
               if n_mo >= 12 and (1.0 + total) > 0 else np.nan)
        irr_val = irr_by_acct.get(str(acct), np.nan)
        irr_show = irr_val if (n_mo >= 12 and not pd.isna(irr_val)) else np.nan
        label = ACCOUNT_DISPLAY.get(acct, acct)
        is_synth = str(acct) in SYNTHETIC_ONBOARDING
        if is_synth:
            label = f"{label} †"
        rows.append({
            "account_id": str(acct), "account_label": str(label),
            "first_month": str(g.iloc[0]["month"]),
            "last_month": str(g.iloc[-1]["month"]), "months": n_mo,
            "start_nav": float(g.iloc[0]["nav"]),
            "end_nav": float(g.iloc[-1]["nav"]),
            "net_flow": float(g["net_external_flow"].sum()),
            "cum_twr": total, "ann_twr": ann, "irr": irr_show,
            "is_synthetic": is_synth,
        })
    rows.sort(key=lambda r: r["cum_twr"], reverse=True)   # stable, matches original
    return pd.DataFrame(rows, columns=cols)


def _per_account(twr_account: pd.DataFrame, irr_table: pd.DataFrame,
                 selected_account_ids, account_active: bool) -> dict:
    """Mirror app.py 2960-3031. Thin formatter over ``per_account_raw`` — the
    numbers live there now (as decimal fractions); this wrapper multiplies by
    100 and applies the same string formatters the original inlined."""
    raw = per_account_raw(twr_account, irr_table, selected_account_ids, account_active)
    if raw.empty:
        return {"rows": [], "footnote": "", "account_filter_note": account_active}
    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "account": r["account_label"],
            "first": r["first_month"],
            "last": r["last_month"],
            "months": int(r["months"]),
            "start_nav": fmt_money(r["start_nav"]),
            "end_nav": fmt_money(r["end_nav"]),
            "net_flow": hs._signed_money(r["net_flow"]),
            "cum_twr": hs._spct(r["cum_twr"] * 100.0), "cum_dir": hs._dir(r["cum_twr"]),
            "ann_twr": hs._spct(r["ann_twr"] * 100.0) if not pd.isna(r["ann_twr"]) else "—",
            "ann_dir": hs._dir(r["ann_twr"]),
            "irr": hs._spct(r["irr"] * 100.0) if not pd.isna(r["irr"]) else "—",
            "irr_dir": hs._dir(r["irr"]) if not pd.isna(r["irr"]) else "flat",
            "synthetic": bool(r["is_synthetic"]),
        })
    any_synth = bool(raw["is_synthetic"].any())
    footnote = (
        "† Start NAV reflects pre-tracking opening balance — money that predates "
        "the statement archive, seeded as a synthetic onboarding flow at portfolio "
        "rollup time. Not a real deposit; TWR / IRR are computed only from the "
        "post-debut window." if any_synth else "")
    return {"rows": rows, "footnote": footnote,
            "account_filter_note": account_active}


# --------------------------------------------------------------------------- #
# Total NAV by month — mirror app.py 3033-3118 + monthly_totals (859-870).
# --------------------------------------------------------------------------- #
def nav_totals_raw(positions_monthly: pd.DataFrame) -> pd.DataFrame:
    """Raw total-NAV-by-month rows (shared compute for BOTH UIs). Mirror
    app.py.monthly_totals (859-870)."""
    if positions_monthly is None or positions_monthly.empty:
        return pd.DataFrame(columns=["month", "total", "month_end"])
    out = (positions_monthly.groupby("month")
           .agg(total=("market_value", "sum"),
                n_accts=("account_id", "nunique"))
           .reset_index().sort_values("month").reset_index(drop=True))
    out["month_end"] = out["month"].dt.to_timestamp("M").dt.normalize()
    return out


def _nav_section(frames: Frames, eg: pd.DataFrame,
                 bucket_filter, class_filter,
                 holdings_filter_active: bool) -> dict:
    pm = frames.positions_monthly
    if holdings_filter_active:
        pm = pm[pm["bucket"].isin(bucket_filter) & pm["asset_class"].isin(class_filter)]
    ts = nav_totals_raw(pm)
    if ts.empty:
        return {"trio": None, "head": {"title": "Total NAV by month", "sub": ""},
                "points": [], "markers": [], "reconcile_note": ""}
    last = ts.iloc[-1]
    peak = ts.loc[ts["total"].idxmax()]
    trio = {
        "current": {"value": fmt_money(float(last["total"])),
                    "sub": pd.Timestamp(last["month_end"]).strftime("%b %Y") + " · marked to live prices"},
        "peak": {"value": fmt_money(float(peak["total"])),
                 "sub": pd.Timestamp(peak["month_end"]).strftime("%b %Y")},
        "months": {"value": str(len(ts)),
                   "sub": (pd.Timestamp(ts.iloc[0]["month_end"]).strftime("%b %Y")
                           + " to " + pd.Timestamp(last["month_end"]).strftime("%b %Y"))},
    }
    pts = [{"x": pd.Timestamp(r["month_end"]).strftime("%Y-%m-%d"),
            "v": float(r["total"])} for _, r in ts.iterrows()]
    markers = []
    for fm, g in eg.groupby("first_month", sort=True):
        x = fm.to_timestamp("M").strftime("%Y-%m-%d")
        lines = [f"+{r['account_display']} (+${r['join_value'] / 1000:,.0f}K)"
                 for _, r in g.iterrows()]
        markers.append({"x": x, "label": " · ".join(lines)})
    reconcile = (
        "Statement return-basis NAV anchors TWR / IRR; the marked "
        f"\"Portfolio value\" of {fmt_money(float(last['total']))} reflects live "
        "prices and reconciles on this tab. The two figures differ because "
        "statements lag live marks by the settlement window."
        if not holdings_filter_active else "")
    return {"trio": trio, "head": {"title": "Total NAV by month",
            "sub": "dollar NAV summed across accounts — onboarding steps are visible-but-harmless"},
            "points": pts, "markers": markers, "reconcile_note": reconcile}


# --------------------------------------------------------------------------- #
# Filtered-TWR synthesis — the shared risk_bundle pipeline (single source
# with app.py's filtered_twr_frame path since Phase D).
# --------------------------------------------------------------------------- #
def _filtered_daily_port_rets(frames: Frames, bucket_filter,
                              class_filter) -> pd.Series:
    """Filtered daily portfolio returns — the shared risk_bundle pipeline
    with the Account/Asset-class filter applied inside the per-month fold
    (single source with app.py's bundle since Phase D)."""
    return daily_portfolio_returns(frames.positions, frames.daily_prices,
                                   bucket_filter, class_filter)


def _filtered_twr_view(frames: Frames, port_canonical: pd.DataFrame,
                       bucket_filter, class_filter) -> pd.DataFrame:
    """Mirror filtered_twr_frame (910-953); the monthly compounding now
    delegates to ``risk_bundle.synthesize_monthly_returns`` (single source
    since Phase D), then builds a twr_portfolio-shaped frame."""
    empty = pd.DataFrame(columns=_TWR_COLS + ["wealth_index", "month_end"])
    if port_canonical is None or port_canonical.empty:
        return empty
    port_rets = _filtered_daily_port_rets(frames, bucket_filter, class_filter)
    if port_rets.empty:
        return empty
    monthly = synthesize_monthly_returns(port_rets, port_canonical).dropna()
    if monthly.empty:
        return empty
    out = pd.DataFrame({"statement_date": pd.to_datetime(monthly.index),
                        "return_pct": monthly.values.astype(float)}) \
        .sort_values("statement_date").reset_index(drop=True)
    out["prev_stmt_date"] = out["statement_date"].shift(1)
    out["month"] = out["statement_date"].dt.to_period("M")
    out["month_end"] = out["month"].dt.to_timestamp("M").dt.normalize()
    wealth = (1.0 + out["return_pct"].fillna(0.0)).cumprod()
    out["cum_return"] = wealth - 1.0
    out["wealth_index"] = wealth
    out["twr_dd_pct"] = (wealth / wealth.cummax() - 1.0) * 100.0
    return out


# --------------------------------------------------------------------------- #
# View assembly (grows over subsequent tasks).
# --------------------------------------------------------------------------- #
def twr_view_for(frames: Frames, account: str | list[str] = "all",
                 asset_class: str | list[str] = "all", snap_all=None):
    """Resolved Holdings filter + the (possibly synthesized) monthly TWR
    view — the single seam build_performance_view and the AI facts reducer
    share, so the box narrates exactly the tab's series under any filter
    (box==tab by construction). ``snap_all`` lets a caller that already
    holds the current snapshot pass it in (build_performance_view does);
    default recomputes it. Returns (port_view, bucket_filter,
    class_filter, selected_account_ids, account_active, class_active)."""
    port = _prepare_portfolio_twr(frames.twr_portfolio)
    if snap_all is None:
        snap_all = hs._current_snap(frames)
    (bucket_filter, class_filter, selected_account_ids,
     account_active, class_active) = _resolve_filter(frames, snap_all,
                                                     account, asset_class)
    # Filtered Performance series are synthesized from daily returns within
    # each statement-date window (mirror filtered_twr_frame + the risk
    # bundle); the unfiltered path stays byte-identical to the canonical
    # twr_portfolio frame.
    if account_active or class_active:
        port_view = _filtered_twr_view(frames, port, bucket_filter,
                                       class_filter)
    else:
        port_view = port
    return (port_view, bucket_filter, class_filter, selected_account_ids,
            account_active, class_active)


def build_performance_view(frames: Frames, *, account: str | list[str] = "all",
                           asset_class: str | list[str] = "all") -> dict:
    """Assemble the GET /api/performance contract. Pure given frames +
    selections. Grows task-by-task; Task 1 returns meta + headline."""
    # Read off Frames (loaded + demo-overlaid once in load_frames, narrowed by
    # apply_global_filters) — NOT a fresh data_dir read, which would bypass the
    # broker choke-point and leak real per-account $ under a test-only
    # selection. _load_twr_account (above) stays as the standalone parser it
    # mirrors, still covered directly by tests.
    twr_account = frames.twr_account

    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)
    (port_view, bucket_filter, class_filter, selected_account_ids,
     account_active, class_active) = twr_view_for(frames, account, asset_class,
                                                  snap_all=snap_all)
    holdings_filter_active = account_active or class_active
    # Onboarding events grouped once — reused by the NAV section in a later task.
    eg = _events_grouped(frames, selected_account_ids,
                         account_active, class_active)
    as_of_label = (pd.Timestamp(frames.available_dates[0]).strftime("%b %d, %Y")
                   if frames.available_dates else "")

    meta = {
        "as_of_label": as_of_label,
        "accounts": acct_opts,
        "classes": class_opts,
        "brokers": broker_opts,
        "filter": hs._filter_meta(account, asset_class),
        "holdings_filter_active": holdings_filter_active,
        "account_filter_active": account_active,
        "class_filter_active": class_active,
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "empty": bool(port_view.empty),
    }

    # Filter disclosure — the synthesized-TWR caveat shown when a Holdings
    # filter narrows the Performance series (combined-statement note is Task 8).
    disclosures = {"holdings_filter": None, "combined_statement": None}
    if holdings_filter_active:
        bits = []
        if account_active:
            bits.append("Account filtered")
        if class_active:
            bits.append("Asset-class filtered")
        disclosures["holdings_filter"] = {"text":
            ("Holdings filter active (" + " · ".join(bits) + "). Monthly TWR is "
             "synthesized from daily returns within each statement-date window "
             "using the filtered weight set — numbers can shift slightly from the "
             "unfiltered view. IRR is hidden under this filter.")}

    # Combined-statement coverage note (unfiltered only — mirror app.py
    # 2605-2614 gating). A month with no separate statement because a broker
    # rolled it into a multi-month combined PDF carries n_accounts_filled>0 and
    # n_accounts_missing==0; that bookkeeping isn't carried on the Holdings-
    # filtered path, so the note only fires unfiltered.
    if (not holdings_filter_active and not port_view.empty
            and "n_accounts_filled" in frames.twr_portfolio.columns):
        filled = frames.twr_portfolio["n_accounts_filled"].fillna(0)
        missing = (frames.twr_portfolio["n_accounts_missing"].fillna(0)
                   if "n_accounts_missing" in frames.twr_portfolio.columns
                   else pd.Series(0, index=frames.twr_portfolio.index))
        n_combined = int(((filled > 0) & (missing == 0)).sum())
        if n_combined:
            disclosures["combined_statement"] = {"text":
                (f"{n_combined} month(s) had no separate statement because a "
                 "broker rolled them into a multi-month combined PDF. The "
                 "forward-filled NAV is biased toward the prior month — a broker "
                 "quirk, not a missing file.")}

    # Provisional stub period (spec 2026-08-22): whole-book / broker-scoped
    # views only — a Holdings-filtered series is daily-synthesized, so the
    # snapshot basis does not apply there.
    stub = None if holdings_filter_active else hs.interim_stub(frames)
    return {
        "meta": meta,
        "disclosures": disclosures,
        "headline": _headline(port_view, frames.irr_table,
                              holdings_filter_active,
                              broker_scope=frames.broker_scope, stub=stub),
        "cashflows": _cashflows(frames.transactions, port_view, selected_account_ids,
                                account_active, class_active),
        "cum_twr": _cum_twr_section(port_view, eg, stub=stub),
        # Additive: the key exists only when a stub does (golden stays byte-identical).
        **({"stub": stub_block(stub, hs._spct)} if stub is not None else {}),
        "drawdown": _drawdown_section(port_view),
        "periodic": _periodic(port_view),
        "per_account": _per_account(twr_account, frames.irr_table,
                                    selected_account_ids, account_active),
        "nav": _nav_section(frames, eg, bucket_filter, class_filter,
                            holdings_filter_active),
    }
