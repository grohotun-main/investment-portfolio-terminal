# terminal/options_service.py
"""Pure data seam for the MERIDIAN Terminal "Options Hedging" tab — read half.

Re-expresses app.py._render_options_body (7984-8518, minus the Refresh buttons —
the terminal drives those from the ACTION layer instead, see the "options" group
in terminal/actions_service.py) as a Streamlit-free module. It reads the option
CSVs from frames.data_dir and calls the SAME engines app.py does
(option_positions / options_pricer / iv_rank) — zero new math. Per-row
Greeks/economics are computed only to roll up into the aggregate tiles +
weighted IV + the IV-percentile gauge; the read half renders no per-position
table (see the spec's Update note).

Whole-book: no Account / Asset-class filter (the tab ignores them).
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from terminal import holdings_service as hs

from option_positions import (build_option_position_table, option_book_aggregates,
                              greek_dollar_columns)
from options_pricer import binomial_american
from iv_rank import (book_iv_percentile, book_iv_percentile_series,
                     format_weighted_iv_caption)
from monthly_normalize import slice_as_of_month

from options_hedging_data import build_options_hedging_inputs
from risk_metrics import compute_risk_contributions
from crash_betas import compute_crash_betas
from hedge_recommender import (build_hedge_basket, identify_excess_mcr_names,
                               compute_roll_schedule, EXIT_RULE_MULTIPLIER,
                               TENOR_DAYS, CRASH_WINDOWS)
from hedge_signals import (build_hedge_signals, hedge_signal_universe,
                           signals_to_table_rows, format_signal_headline)
from fetch_targeted_chain import fetch_targeted_contracts
from terminal import risksim_service as rss

IV_PCT_WINDOW_DAYS = 252
IV_PCT_SPAN_DAYS = 252
# Dividend-yield convention, verbatim from app.py:8195.
Q_BY_TICKER = {"SPY": 0.015, "QQQ": 0.005}

RC_WINDOW = 252  # app.py:149 (module-scope; the Options tab reuses it)
REC_TARGETS = {"A": [0.05, 0.10, 0.15, 0.20, 0.25], "B": [0.005, 0.010, 0.015]}
TARGET_LABELS = {
    "A": {0.05: "5%", 0.10: "10%", 0.15: "15%", 0.20: "20%", 0.25: "25%"},
    "B": {0.005: "0.5%", 0.010: "1.0%", 0.015: "1.5%"}}
MODE_LABELS = {"A": "A — Cap drawdown", "B": "B — Tail hedge"}
DEFAULT_TARGET = 0.10

CAPTION = (  # app.py 7992-7998 verbatim
    "Tracks long-option protection in the portfolio. Parses each option "
    "position's strike/expiry from broker statements, prices Greeks via "
    "the same LR-American pricer used by `stress_hedge`, and shows "
    "scenario P&L for tail-event shocks. Inherits the global Broker "
    "filter; ignores Account / Asset-class filters (every long option "
    "shows regardless of the active class slice)."
)


def _jnum(v):
    """JSON-safe float: NaN/inf -> None so json.dumps(allow_nan=False) is clean."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _staleness_chip(label: str, fetched_at, missing_hint: str,
                    now: pd.Timestamp) -> str:
    """Port of app.py:8099-8116, but the not-loaded hint names the CLI parser
    rather than a button — the same CLI-hint idiom the dip / factor / income /
    riskcontrib / risksim services use. The terminal's own Refresh buttons live
    in the ACTION layer ("options" group, terminal/actions_service.py)."""
    if fetched_at is None:
        return f"⚪ {label}: not loaded — {missing_hint}"
    age = now - pd.Timestamp(fetched_at)
    age_hrs = age.total_seconds() / 3600
    if age_hrs < 24:
        age_str, icon = f"{int(age.total_seconds() / 60)} min ago", "\U0001f7e2"
    elif age_hrs < 24 * 7:
        age_str, icon = f"{int(age_hrs)} h ago", "\U0001f7e1"
    else:
        age_str, icon = f"{int(age_hrs / 24)} d ago", "\U0001f534"
    return (f"{icon} {label}: "
            f"{pd.Timestamp(fetched_at).strftime('%b %d, %Y %H:%M UTC')} "
            f"({age_str})")


def _rf_rate(data_dir: Path) -> float:
    rf = _read_csv(data_dir / "risk_free_rate.csv", parse_dates=["date"])
    if rf.empty or "rate_annual" not in rf.columns:
        return 0.04
    try:
        return float(rf.dropna(subset=["rate_annual"]).sort_values("date")
                     .iloc[-1]["rate_annual"])
    except Exception:
        return 0.04


def _merge_snapshot(opt_tbl: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Port of app.py:8148-8181 — merge Polygon snapshot fields onto opt_tbl by
    (underlying, opt_type, strike, expiry)."""
    merge_cols = ["spot", "premium_mid", "polygon_iv", "polygon_delta",
                  "polygon_vega", "polygon_bid", "polygon_ask",
                  "polygon_open_interest"]
    if not snap.empty:
        snap = snap.copy()
        snap["expiry"] = pd.to_datetime(snap["expiry"]).dt.date
        snap_keys = snap.set_index(["underlying", "opt_type", "strike", "expiry"])
        opt_tbl["expiry"] = pd.to_datetime(opt_tbl["expiry"]).dt.date
        cols = list(merge_cols)
        if "atm_iv" in snap.columns:
            cols.append("atm_iv")
        for col in cols:
            opt_tbl[col] = opt_tbl.apply(
                lambda r: snap_keys.at[
                    (r["underlying"], r["opt_type"], r["strike"], r["expiry"]),
                    col,
                ] if (r["underlying"], r["opt_type"], r["strike"],
                      r["expiry"]) in snap_keys.index else np.nan,
                axis=1,
            )
        if "atm_iv" not in snap.columns:
            opt_tbl["atm_iv"] = np.nan
    else:
        for col in (*merge_cols, "atm_iv"):
            opt_tbl[col] = np.nan
    return opt_tbl


def _assemble_opt_tbl(frames: hs.Frames, *, today: date) -> pd.DataFrame:
    """Build the merged, Greek-priced, economics-enriched option table — the same
    frame app.py builds at 8128-8330. Returns empty (post-unparsed-filter) if there
    are no parseable options."""
    data_dir = Path(frames.data_dir)
    snap = _read_csv(data_dir / "option_position_snapshot.csv",
                     parse_dates=["expiry", "fetched_at"])

    opt_tbl = build_option_position_table(frames.positions, frames.transactions)
    if opt_tbl.empty:
        return opt_tbl
    opt_tbl = opt_tbl[opt_tbl["source"] != "unparsed"].copy()
    if opt_tbl.empty:
        return opt_tbl

    opt_tbl = _merge_snapshot(opt_tbl, snap)

    r_rate = _rf_rate(data_dir)
    opt_tbl["dte"] = opt_tbl["expiry"].apply(lambda e: max(0, (e - today).days))
    opt_tbl["T_years"] = opt_tbl["dte"] / 365.0
    opt_tbl["q"] = opt_tbl["underlying"].map(Q_BY_TICKER).fillna(0.0)

    def _greeks(row: pd.Series) -> dict:
        if not (np.isfinite(row.get("spot", np.nan))
                and np.isfinite(row.get("polygon_iv", np.nan))
                and row.get("T_years", 0) > 0):
            return {"delta": np.nan, "gamma": np.nan, "vega": np.nan,
                    "theta": np.nan, "price": np.nan}
        try:
            res = binomial_american(
                spot=float(row["spot"]), strike=float(row["strike"]),
                T=float(row["T_years"]), r=r_rate, q=float(row["q"]),
                sigma=float(row["polygon_iv"]), opt=row["opt_type"], method="lr")
            return {"delta": res["delta"], "gamma": res["gamma"],
                    "vega": res["vega"], "theta": res["theta"],
                    "price": res["price"]}
        except Exception:
            return {"delta": np.nan, "gamma": np.nan, "vega": np.nan,
                    "theta": np.nan, "price": np.nan}

    g = opt_tbl.apply(_greeks, axis=1, result_type="expand")
    opt_tbl[["model_delta", "model_gamma", "model_vega",
             "model_theta", "model_price"]] = g

    # Per-row economics + Greek-dollar display units (app.py 8237-8330). These feed
    # only the aggregates below — no per-position table is rendered.
    opt_tbl["unrealized_pnl"] = opt_tbl["market_value"] - opt_tbl["cost_basis_total"]
    opt_tbl = greek_dollar_columns(opt_tbl)
    return opt_tbl


def build_options_view(frames: hs.Frames, *, today: "date | None" = None,
                       now: "pd.Timestamp | None" = None) -> dict:
    """Assemble the GET /api/options read-half contract. Pure given frames.

    ``today`` (DTE / Greeks / aggregates / IV as-of) defaults to date.today();
    ``now`` (staleness age) defaults to pd.Timestamp.now(tz="UTC"). The live route
    passes neither (real clock); tests pin both for a deterministic golden.
    """
    today = today or date.today()
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    data_dir = Path(frames.data_dir)

    # Staleness — read the two CSVs and stamp chips regardless of whether there are
    # any option positions (app.py renders the chips before the empty-return).
    snap = _read_csv(data_dir / "option_position_snapshot.csv",
                     parse_dates=["expiry", "fetched_at"])
    atm = _read_csv(data_dir / "atm_iv_history.csv",
                    parse_dates=["date", "fetched_at"])
    snap_fetched = (snap["fetched_at"].max()
                    if not snap.empty and "fetched_at" in snap.columns else None)
    atm_fetched = (atm["fetched_at"].max()
                   if not atm.empty and "fetched_at" in atm.columns else None)
    staleness = {
        "snapshot": {
            "fetched_at": (pd.Timestamp(snap_fetched).isoformat()
                           if snap_fetched is not None else None),
            "chip": _staleness_chip(
                "Snapshot", snap_fetched,
                "run `parsers/fetch_option_position_iv.py --write`", now)},
        "atm_iv": {
            "fetched_at": (pd.Timestamp(atm_fetched).isoformat()
                           if atm_fetched is not None else None),
            "chip": _staleness_chip(
                "ATM IV history", atm_fetched,
                "run `parsers/fetch_atm_iv_history.py --write`", now)},
    }

    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)
    meta = {
        "title": "Options Hedging",
        "caption": CAPTION,
        "accounts": acct_opts,
        "classes": class_opts,
        "brokers": broker_opts,
        "available_dates": list(frames.available_dates),
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "filter": {"account": "all", "asset_class": "all", "broker": "all"},
    }
    r_rate = _rf_rate(data_dir)
    footer = {"r_rate_pct": _jnum(r_rate * 100.0),
              "q_note": "1.5% SPY, 0% else"}

    # Empty state — no parseable options. Chips + footer still render.
    opt_raw = build_option_position_table(frames.positions, frames.transactions)
    if opt_raw.empty:
        return {"meta": meta, "empty": True,
                "empty_message":
                    "No option positions found in the latest statement period.",
                "staleness": staleness, "aggregates": None,
                "iv_percentile": None, "footer": footer}
    if opt_raw[opt_raw["source"] != "unparsed"].empty:
        return {"meta": meta, "empty": True,
                "empty_message":
                    "No parseable option positions for the selected broker(s).",
                "staleness": staleness, "aggregates": None,
                "iv_percentile": None, "footer": footer}

    opt_tbl = _assemble_opt_tbl(frames, today=today)

    # --- Aggregate tiles (app.py 8278-8437) ---
    agg = option_book_aggregates(opt_tbl, today)
    if int(agg.get("n_live") or 0) == 0:
        # Every parseable contract is closed (qty 0) or expired — an interim
        # roll that netted the whole book, or a statement still listing
        # closed positions. A grid of $0 tiles reads as "existing puts with
        # -100% P&L" (TK, Aug 2026); say what the book actually is.
        n_gone = int(agg.get("n_excluded") or 0)
        return {"meta": meta, "empty": True,
                "empty_message":
                    ("No open option positions — "
                     f"{n_gone} closed or expired since the last statement."),
                "staleness": staleness, "aggregates": None,
                "iv_percentile": None, "footer": footer}
    as_of_ts = (pd.Timestamp(frames.available_dates[0])
                if frames.available_dates else pd.Timestamp(today))
    snap_nav = slice_as_of_month(frames.positions_monthly, as_of_ts)
    port_nav = (float(snap_nav["market_value"].sum())
                if not snap_nav.empty else float("nan"))

    total_gamma = float(opt_tbl["gamma_dollar_per_1pct"].sum(skipna=True))
    total_vega = float(opt_tbl["vega_dollar_per_volpt"].sum(skipna=True))
    total_theta = float(opt_tbl["theta_dollar_per_day"].sum(skipna=True))

    iv_w = opt_tbl["polygon_iv"].notna() & (opt_tbl["market_value"] > 0)
    if iv_w.any():
        mv_w = float(opt_tbl.loc[iv_w, "market_value"].sum())
        weighted_iv = (float((opt_tbl.loc[iv_w, "polygon_iv"]
                              * opt_tbl.loc[iv_w, "market_value"]).sum() / mv_w)
                       if mv_w > 0 else float("nan"))
    else:
        weighted_iv = float("nan")
    greeks_missing = (not iv_w.any() and total_gamma == 0.0
                      and total_vega == 0.0 and total_theta == 0.0)

    notional = agg["notional_protected"]
    pct_nav = (notional / port_nav * 100.0
               if (np.isfinite(port_nav) and port_nav > 0) else float("nan"))
    cost = agg["cost_basis"]
    pnl = agg["unrealized_pnl"]
    aggregates = {
        "notional_protected": _jnum(notional),
        "notional_pct_nav": _jnum(pct_nav),
        "premium_at_risk": _jnum(agg["premium_at_risk"]),
        "cost_basis": _jnum(cost),
        "n_excluded": int(agg["n_excluded"]),
        "unrealized_pnl": _jnum(pnl),
        "pnl_pct_cost": _jnum(pnl / cost * 100.0) if cost > 0 else None,
        "weighted_dte": _jnum(agg["weighted_dte"]),
        "gamma_dollar": _jnum(total_gamma),
        "vega_dollar": _jnum(total_vega),
        "theta_dollar": _jnum(total_theta),
        "weighted_iv": _jnum(weighted_iv),
        "greeks_missing": bool(greeks_missing),
    }

    # --- IV-percentile gauge + sparkline (app.py 8439-8512) ---
    book_iv_pct = None
    _opt_mv = {}
    if not atm.empty:
        _opt_mv = (opt_tbl.loc[opt_tbl["market_value"] > 0]
                   .groupby("underlying")["market_value"].sum().to_dict())
        if _opt_mv:
            book_iv_pct = book_iv_percentile(
                atm, _opt_mv, as_of=today, window_days=IV_PCT_WINDOW_DAYS)

    caption = None
    if math.isfinite(weighted_iv) and book_iv_pct is not None:
        caption = format_weighted_iv_caption(
            weighted_iv, book_iv_pct, window_days=IV_PCT_WINDOW_DAYS)

    series = []
    last_pct = None
    if book_iv_pct is not None and not math.isnan(book_iv_pct.percentile):
        last_pct = _jnum(book_iv_pct.percentile)
        sdf = book_iv_percentile_series(
            atm, _opt_mv, as_of=today, window_days=IV_PCT_WINDOW_DAYS,
            span_days=IV_PCT_SPAN_DAYS)
        if len(sdf) >= 2:
            # House chart-point shape {"x", "v"} — attachAxes date ticks and
            # the crosshair read p.x (a "date" key rendered an axis-less chart).
            series = [{"x": pd.Timestamp(d).strftime("%Y-%m-%d"),
                       "v": _jnum(p)}
                      for d, p in zip(sdf["date"], sdf["percentile"])]

    iv_percentile = {
        "caption": caption,
        "window_days": IV_PCT_WINDOW_DAYS,
        "last_percentile": last_pct,
        "series": series,
        "bands": {"y_lo": 0, "y_hi": 100, "cheap_thr": 20, "rich_thr": 80},
    }

    return {"meta": meta, "empty": False, "empty_message": None,
            "staleness": staleness, "aggregates": aggregates,
            "iv_percentile": iv_percentile, "footer": footer}


def _iso(d):
    return d.isoformat() if d is not None else None


def _scenario_to_dict(s) -> dict:
    return {"portfolio_drawdown": _jnum(s.portfolio_drawdown),
            "implied_spy_drop": _jnum(s.implied_spy_drop),
            "unhedged_pnl": _jnum(s.unhedged_pnl),
            "existing_payoff": _jnum(s.existing_payoff),
            "existing_pnl": _jnum(s.existing_pnl),
            "existing_pnl_pct": _jnum(s.existing_pnl_pct),
            "new_payoff": _jnum(s.new_payoff),
            "combined_pnl": _jnum(s.combined_pnl),
            "combined_pnl_pct": _jnum(s.combined_pnl_pct)}


def _existing_put_to_dict(ep, today) -> dict:
    roll_by, roll_into = compute_roll_schedule(ep.expiry, today)
    return {"ticker": ep.ticker, "strike": _jnum(ep.strike),
            "expiry": _iso(ep.expiry), "contracts": int(ep.contracts),
            "cost_basis": _jnum(ep.cost_basis),
            "current_value": _jnum(ep.current_value),
            "worst_case_payoff": _jnum(ep.worst_case_payoff),
            "roll_by": _iso(roll_by), "roll_into": _iso(roll_into),
            "sell_at": _jnum(EXIT_RULE_MULTIPLIER * ep.cost_basis)}


def _new_put_to_dict(leg) -> dict:
    return {"ticker": leg.ticker, "role": leg.role, "strike": _jnum(leg.strike),
            "strike_pct_otm": _jnum(leg.strike_pct_otm), "expiry": _iso(leg.expiry),
            "contracts": int(leg.contracts),
            "premium_per_share": _jnum(leg.premium_per_share),
            "position_cost": _jnum(leg.position_cost),
            "annualized_drag_pct": _jnum(leg.annualized_drag_pct),
            "sell_at": _jnum(EXIT_RULE_MULTIPLIER * leg.position_cost)}


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_cap(v) -> str:
    return f"{v * 100:.1f}%" if not _isnan(v) else "—"


def _fmt_worst(v) -> str:
    # Signed worst-case net P&L: a loss reads "a loss of X%", a gain
    # "a gain of +X%" (the honest-gain-not-loss over-cover case).
    if _isnan(v):
        return "—"
    return (f"a loss of {abs(v) * 100:.1f}%" if v < 0
            else f"a gain of +{v * 100:.1f}%")


def headline_decision(*, mode, current_cap_pct, target_cap_pct,
                      combined_cap_pct, has_new_puts) -> dict:
    """The 5-way hedge-headline branch DECISION, shared by both UIs (which each
    render their own HTML/markdown wording). Returns {level, case, over_covers}.
    Single-sources the thresholds (+0.001 / -0.005 / the NaN gate) so they can't
    drift. case in {unavailable, already_covered, mode_a, mode_b_ladder,
    budget_too_small}."""
    if _isnan(current_cap_pct) or _isnan(combined_cap_pct):
        return {"level": "warn", "case": "unavailable", "over_covers": False}
    if mode == "A" and (not has_new_puts
                        and current_cap_pct <= target_cap_pct + 0.001):
        return {"level": "success", "case": "already_covered", "over_covers": False}
    if mode == "A":
        return {"level": "note", "case": "mode_a",
                "over_covers": combined_cap_pct < target_cap_pct - 0.005}
    if has_new_puts:
        return {"level": "note", "case": "mode_b_ladder", "over_covers": False}
    return {"level": "info", "case": "budget_too_small", "over_covers": False}


def _build_headline(*, mode, current_cap_pct, target_cap_pct, combined_cap_pct,
                    current_worst_pnl_pct, combined_worst_pnl_pct,
                    has_new_puts, target) -> dict:
    """Server-side port of the app.py 8861-8943 headline branching. Pure string
    assembly over already-computed fields — no math. Returns {level, html}; the
    front-end maps level -> a .callout-* class."""
    d = headline_decision(mode=mode, current_cap_pct=current_cap_pct,
                          target_cap_pct=target_cap_pct,
                          combined_cap_pct=combined_cap_pct,
                          has_new_puts=has_new_puts)
    case = d["case"]
    if case == "unavailable":
        return {"level": d["level"], "html": (
            "<strong>Scenarios unavailable.</strong> Couldn't compute the "
            "implied SPY drop — usually means crash-beta history is missing for "
            "held tickers, or daily-prices need a refresh. The table above shows "
            "which scenarios failed.")}
    if case == "already_covered":
        return {"level": d["level"], "html": (
            "<strong>Already covered — buy nothing.</strong> With your current "
            f"puts the worst of the 5 scenarios is "
            f"<strong>{_fmt_worst(current_worst_pnl_pct)}</strong> of the "
            f"portfolio (target was a max loss of "
            f"<strong>{_fmt_cap(target_cap_pct)}</strong>).")}
    if case == "mode_a":
        why = ("<p>The new puts take you <strong>past</strong> the cap on "
               "purpose: single-name puts are sized to neutralize your "
               "concentrated (excess-MCR) names in full, not trimmed to the cap "
               "— so the basket over-covers, and the worst modeled scenario can "
               "even turn convex-positive (a net gain).</p>") if d["over_covers"] else ""
        return {"level": d["level"], "html": (
            f"<p><strong>With your existing puts, the worst of the 5 scenarios "
            f"is {_fmt_worst(current_worst_pnl_pct)}</strong> of the portfolio</p>"
            f"<p><strong>Your target cap:</strong> a max loss of "
            f"{_fmt_cap(target_cap_pct)}</p>"
            f"<p><strong>After buying the recommended new puts, the worst of the "
            f"5 scenarios is {_fmt_worst(combined_worst_pnl_pct)}</strong></p>"
            f"{why}")}
    if case == "mode_b_ladder":
        return {"level": d["level"], "html": (
            f"<p><strong>Existing puts</strong>: the worst of the 5 scenarios is "
            f"<strong>{_fmt_worst(current_worst_pnl_pct)}</strong> of the "
            f"portfolio</p>"
            f"<p>This is a <strong>{target * 100:.2f}%/yr</strong> deep-OTM "
            f"tail-hedge ladder — a standing premium, <strong>not</strong> a "
            f"loss cap.</p>"
            f"<p>Adding the recommended puts on top brings the worst case to "
            f"<strong>{_fmt_worst(combined_worst_pnl_pct)}</strong>.</p>")}
    return {"level": d["level"], "html": (
        f"<strong>Budget too small to ladder in a put.</strong> A "
        f"<strong>{target * 100:.2f}%/yr</strong> budget doesn't cover even one "
        f"whole 20%-OTM contract here — raise the budget, or check that the "
        f"option chain refreshed.")}


def _scenario_notes(rec) -> list:
    """Data-driven captions under the scenarios table (app.py 8842-8859):
    excluded-no-history (any mode) then the Mode-A cap-precision note. Plain
    strings — the front-end renders them via textContent."""
    notes = []
    excluded = rec.diagnostics.get("scenario_excluded_no_history", []) or []
    if excluded:
        names = ", ".join(f"{t} ({w * 100:.0f}% of equity)" for t, w in excluded)
        notes.append(
            f"⚠ No crash-window history for {names} — listed after the most "
            "recent crash window, so no crash beta exists. Excluded from the "
            "implied-SPY-drop math (remaining equity renormalized); scenario P&L "
            "omits this name's market value.")
    if rec.mode == "A" and rec.cap_precision_note:
        notes.append(rec.cap_precision_note)
    return notes


def _rec_to_dict(rec, today) -> dict:
    excluded = rec.diagnostics.get("scenario_excluded_no_history", []) or []
    return {"mode": rec.mode, "target": _jnum(rec.target),
            "current_cap_pct": _jnum(rec.current_cap_pct),
            "target_cap_pct": _jnum(rec.target_cap_pct),
            "combined_cap_pct": _jnum(rec.combined_cap_pct),
            "current_worst_pnl_pct": _jnum(rec.current_worst_pnl_pct),
            "combined_worst_pnl_pct": _jnum(rec.combined_worst_pnl_pct),
            "scenarios": [_scenario_to_dict(s) for s in rec.scenarios],
            "existing_puts": [_existing_put_to_dict(ep, today)
                              for ep in rec.existing_puts],
            "new_puts": [_new_put_to_dict(l) for l in rec.new_puts],
            "total_new_premium": _jnum(rec.total_new_premium),
            "total_new_drag_pct": _jnum(rec.total_new_drag_pct),
            "total_combined_drag_pct": _jnum(rec.total_combined_drag_pct),
            "cap_precision_note": (rec.cap_precision_note or None),
            "diagnostics": {"scenario_excluded_no_history":
                            [[t, _jnum(w)] for t, w in excluded]},
            "headline": _build_headline(
                mode=rec.mode, current_cap_pct=rec.current_cap_pct,
                target_cap_pct=rec.target_cap_pct,
                combined_cap_pct=rec.combined_cap_pct,
                current_worst_pnl_pct=rec.current_worst_pnl_pct,
                combined_worst_pnl_pct=rec.combined_worst_pnl_pct,
                has_new_puts=bool(rec.new_puts), target=rec.target),
            "scenario_notes": _scenario_notes(rec)}


def _fetch_chain(data_dir: Path, targets: list) -> pd.DataFrame:
    """Offline seam (app.py 8678-8695): a committed hedge_chain_fixture.csv in the
    data dir short-circuits the live Polygon fetch; else resolve the key + fetch."""
    fixture = data_dir / "hedge_chain_fixture.csv"
    if fixture.exists():
        return pd.read_csv(fixture)
    rss._ensure_massive_key()
    return fetch_targeted_contracts(targets)


def build_chain_targets(daily_prices, universe, excess_names, *,
                        mode: str, target: float, today) -> dict:
    """Assemble the option-chain fetch targets (shared by both UIs). Pure:
    strike_depth = target (Mode A) or 0.20 (Mode B); spot = last non-NaN price
    per name; one OTM put target per (SPY + excess names). Returns
    {spot_prices, strike_depth, target_expiry, chain_targets}. `universe` is
    passed in (the caller already built it for crash_betas)."""
    strike_depth = target if mode == "A" else 0.20
    target_expiry = today + pd.Timedelta(days=TENOR_DAYS).to_pytimedelta()
    spot_prices = {t: float(daily_prices[t].dropna().iloc[-1]) for t in universe
                   if t in daily_prices.columns and not daily_prices[t].dropna().empty}
    chain_targets = [(t, spot_prices[t] * (1.0 - strike_depth), target_expiry, "put")
                     for t in ["SPY", *excess_names] if t in spot_prices]
    return {"spot_prices": spot_prices, "strike_depth": strike_depth,
            "target_expiry": target_expiry, "chain_targets": chain_targets}


def parse_chain_premiums(chain_df) -> dict:
    """chain_df rows -> {underlying: {strike, premium, expiry}} (shared). Skips
    rows with no polygon_price or an unparseable expiry."""
    out = {}
    for _, row in chain_df.iterrows():
        pxv = row.get("polygon_price")
        if pxv is None or pd.isna(pxv):
            continue
        try:
            expd = date.fromisoformat(str(row["expiration_date"])[:10])
        except (TypeError, ValueError):
            continue
        out[row["request_underlying"]] = {
            "strike": float(row["request_strike"]),
            "premium": float(row["polygon_price"]), "expiry": expd}
    return out


def _recommend_inputs(frames: hs.Frames, *, mode: str, target: float,
                      today: date) -> dict:
    """Assemble the dependency-injected inputs build_hedge_basket needs — the exact
    frames app.py builds at 8605-8724. Exposed so the parity test can recompute."""
    data_dir = Path(frames.data_dir)
    as_of_ts = (pd.Timestamp(frames.available_dates[0])
                if frames.available_dates else pd.Timestamp(today))
    opt_tbl = _assemble_opt_tbl(frames, today=today)
    daily = frames.daily_prices
    priced = set(daily.columns) if daily is not None and not daily.empty else set()
    hedging = build_options_hedging_inputs(
        positions=frames.positions, opt_tbl=opt_tbl,
        priced_tickers=priced, as_of=as_of_ts)

    bundle = rss._bundle_for(frames, "all", "all")
    weights = bundle["weights"]
    try:
        per_symbol_mcr = compute_risk_contributions(
            weights, daily, window=RC_WINDOW, estimator="ewma_lw")["per_symbol"]
    except Exception:  # noqa: BLE001
        per_symbol_mcr = pd.DataFrame(columns=["pctr_pct"])

    spy_path = data_dir / "spy_holdings.csv"
    spy_missing = not spy_path.exists()
    spy_holdings = (pd.read_csv(spy_path) if not spy_missing
                    else pd.DataFrame(columns=["ticker", "weight_pct"]))

    held = list(hedging.equity_priced_tickers)
    universe = sorted({*held, "SPY"})
    crash_betas = compute_crash_betas(daily, tickers=universe, spy_ticker="SPY",
                                      windows=CRASH_WINDOWS)
    try:
        excess = identify_excess_mcr_names(per_symbol_mcr, spy_holdings)
    except Exception:  # noqa: BLE001
        excess = []
    ct = build_chain_targets(daily, universe, excess, mode=mode, target=target,
                             today=today)
    spot_prices = ct["spot_prices"]
    chain_targets = ct["chain_targets"]
    chain_df, chain_error = pd.DataFrame(), None
    if chain_targets:
        try:
            chain_df = _fetch_chain(data_dir, chain_targets)
        except Exception as e:  # noqa: BLE001
            chain_error = str(e)
    chain_premiums = parse_chain_premiums(chain_df)

    return {"hedging": hedging, "weights": weights,
            "per_symbol_mcr": per_symbol_mcr, "spy_holdings": spy_holdings,
            "spy_missing": spy_missing, "crash_betas": crash_betas,
            "chain_premiums": chain_premiums, "chain_error": chain_error,
            "spot_prices": spot_prices, "holdings": hedging.holdings_for_engine,
            "existing_options": hedging.existing_options}


def build_recommend_view(frames: hs.Frames, *, mode: str, target: float,
                         today: "date | None" = None) -> dict:
    """GET /api/options/recommend contract for one (mode, target). Pure given
    frames. Mirrors app.py 8531-8784: composition + the recommendation +
    hedge signals. Domain failures come back as warnings/chain_error, not raises."""
    today = today or date.today()
    data_dir = Path(frames.data_dir)
    meta = {"mode": mode, "target": _jnum(target),
            "mode_label": MODE_LABELS.get(mode, mode),
            "target_label": TARGET_LABELS.get(mode, {}).get(target, ""),
            "as_of": today.isoformat(),
            "defaults": {"mode": "A", "target": DEFAULT_TARGET}}

    inp = _recommend_inputs(frames, mode=mode, target=target, today=today)
    hedging = inp["hedging"]
    comp = hedging.composition_breakdown
    pv = comp["portfolio_value"]
    composition = {
        "portfolio_value": _jnum(pv),
        "equity_mv": _jnum(comp["equity_mv"]),
        "equity_pct": _jnum(comp["equity_mv"] / max(pv, 1e-9) * 100.0),
        "cash_mv": _jnum(comp["cash_mv"]),
        "cash_pct": _jnum(comp["cash_mv"] / max(pv, 1e-9) * 100.0),
        "options_mv": _jnum(comp["options_mv"]),
        "options_pct": _jnum(comp["options_mv"] / max(pv, 1e-9) * 100.0)}

    warnings: list[str] = []
    if inp["spy_missing"]:
        warnings.append("SPY holdings not cached — run "
                        "`py parsers/fetch_spy_holdings.py --write` to enable "
                        "excess-MCR identification.")
    cs = hedging.coverage_stats
    coverage_caption = None
    if cs["coverage_pct"] < 1.0 and cs["equity_mv_total"] > 0:
        coverage_caption = (
            f"Hedging math covers {cs['coverage_pct']*100:.1f}% of equity MV "
            f"({cs['n_priced_tickers']} priced tickers; {cs['n_unpriced']} held "
            "names without price history are excluded).")

    recommendation, hedge_signals = None, {"level": "grey", "headline": "", "rows": []}
    weights = inp["weights"]
    if frames.daily_prices is None or frames.daily_prices.empty:
        warnings.append("Need daily_prices to compute MCR-based recommendations.")
    elif weights.empty:
        warnings.append("Filtered universe has no positions to hedge.")
    else:
        try:
            rec = build_hedge_basket(
                mode=mode, target=target, holdings=inp["holdings"],
                existing_options=inp["existing_options"],
                per_symbol_mcr=inp["per_symbol_mcr"],
                spy_holdings=inp["spy_holdings"],
                chain_premiums=inp["chain_premiums"],
                crash_betas=inp["crash_betas"], today=today,
                spot_prices=inp["spot_prices"])
            recommendation = _rec_to_dict(rec, today)
            atm = _read_csv(data_dir / "atm_iv_history.csv",
                            parse_dates=["date", "fetched_at"])
            sigs = build_hedge_signals(
                universe=hedge_signal_universe(rec.diagnostics),
                iv_history=atm, as_of=today)
            level, headline = format_signal_headline(sigs)
            hedge_signals = {"level": level, "headline": headline,
                             "rows": signals_to_table_rows(sigs)}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Could not build hedge basket: {e}")

    return {"meta": meta, "composition": composition,
            "coverage_caption": coverage_caption, "warnings": warnings,
            "chain_error": inp["chain_error"], "recommendation": recommendation,
            "hedge_signals": hedge_signals}
