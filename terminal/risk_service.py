# terminal/risk_service.py
"""Pure data seam for the MERIDIAN Terminal "Risk Overview" tab.

Re-expresses app.py._render_risk_body (4188-5507). Every number lives in the
importable, Streamlit-free engines (parsers/risk_metrics.py + the shared
parsers/risk_bundle.py bundle); this module prepares the terminal's inputs and
shapes the engine output into a JSON-native, allow_nan=False-clean view dict.
Numbers match Streamlit 1:1 by construction — both UIs consume the same
engine.

Filters: like Performance, the Account / Asset-class selects narrow the book and
the monthly Pass-1 metrics are then synthesized from daily returns within each
statement-date window (statement-based TWR has no per-account / per-class slice).
Unfiltered, the monthly series is byte-identical to twr_portfolio.

The bundle comes from the shared engine ``parsers/risk_bundle.py`` (single
source for BOTH UIs since Phase D); this module only prepares the terminal's
inputs and shapes the view.

Charts: the line/area series (rolling Sharpe, underwater drawdown, rolling vol,
rolling beta, rolling alpha) ship as common-index point sets the front-end's
drawOverlayChart aligns by position; the distribution histogram + the return
scatter ship as raw bin/point data for two small bespoke SVG drawers. Axis ticks
and hover readouts are derived client-side from those point sets (app.js
attachAxes / attachCrosshair), so this module emits no axis-label or hover field.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import theme
from terminal import holdings_service as hs
from terminal.holdings_service import Frames, fmt_money
from terminal.performance_service import _prepare_portfolio_twr, _resolve_filter

from risk_metrics import (
    compute_alpha_annual,
    compute_beta,
    compute_calmar,
    compute_concentration,
    compute_drawdown_episodes,
    compute_sharpe,
    compute_sortino,
    compute_up_down_beta,
    compute_var_cvar,
    rolling_alpha_annual,
    rolling_beta,
    rolling_up_down_beta,
    spy_decline_between,
    spy_months_underwater_from,
    window_drawdown_pct,
    _window_rf,
)
from risk_bundle import build_risk_series_bundle

# Fallback annualized RF when data/risk_free_rate.csv is absent (app.py:1196).
RF_FALLBACK_ANNUAL = 0.043

# Direction-of-goodness flags (app.py:1060-1064).
HIGHER_BETTER = "higher_better"
LOWER_BETTER = "lower_better"
LESS_NEG_BETTER = "less_negative_better"

# Series colors (theme tokens, resolved to hex for the front-end drawers).
C_PORT = theme.CHART_PORTFOLIO      # azure portfolio line
C_BENCH = theme.CHART_BENCH         # grey SPY line
C_DD = theme.CHART_DRAWDOWN         # coral drawdown area
C_UP = theme.CLASS_COLORS["equity_stock"]   # up-beta
C_ALPHA = theme.CLASS_COLORS["mutual_fund"]  # alpha line
# Donut palette for the per-ticker concentration ring (no natural class map —
# cycled by slice index; "Other" always grey).
_DONUT_PALETTE = ["#4DA3F5", "#2DD4BF", "#818CF8", "#E6B450", "#FB7185",
                  "#38BDF8", "#2FD79A", "#A78BFA", "#F472B6", "#FBBF24"]
_OTHER_COLOR = theme.TEXT_MUTED


# --------------------------------------------------------------------------- #
# Formatters (assume a finite float; non-finite handled by _compare_tile).
# --------------------------------------------------------------------------- #
def _ratio(v: float) -> str:
    return f"{v:.2f}"


def _pct1(v: float) -> str:
    return f"{v:.1f}%"


def _pct2(v: float) -> str:
    return f"{v:.2f}%"


def _fin(v) -> bool:
    return v is not None and math.isfinite(float(v))


def _jnum(v):
    """JSON-safe float: NaN/inf -> None (the allow_nan=False route forbids NaN)."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _signal(port: float, spy, direction: str, threshold_pct: float = 2.0) -> str:
    """Port-vs-SPY quality signal -> 'up' / 'down' / 'flat' (app.py
    _quality_signal 1067-1095). 'up' = better (value rendered green)."""
    if spy is None or not (_fin(port) and _fin(spy)):
        return "flat"
    diff = float(port) - float(spy)
    rel = max(abs(float(spy)) * threshold_pct / 100.0, threshold_pct / 100.0 * 0.5)
    if abs(diff) < rel:
        return "flat"
    if direction in (HIGHER_BETTER, LESS_NEG_BETTER):
        return "up" if diff > 0 else "down"
    if direction == LOWER_BETTER:
        return "up" if diff < 0 else "down"
    return "flat"


def _compare_tile(label: str, port, spy, direction: str, fmt, sub: str = "") -> dict:
    """A render_compare column (app.py 1098-1137): port value (colored by the
    SPY comparison) + a 'SPY: X' line + sub-caption."""
    pv = float(port) if port is not None else float("nan")
    value = fmt(pv) if _fin(pv) else "—"
    spy_str = (fmt(float(spy)) if _fin(spy) else "—") if spy is not None else None
    return {"label": label, "value": value, "spy": spy_str,
            "dir": _signal(pv, spy, direction), "sub": sub}


def _metric_tile(label: str, value: str, sub: str = "") -> dict:
    """A plain st.metric column (the Beta tiles — no SPY comparison)."""
    return {"label": label, "value": value, "spy": None, "dir": "flat", "sub": sub}


def _load_rf(data_dir) -> pd.Series:
    """FRED DGS3MO daily series (app.py.load_risk_free_rate 602-621). Empty
    Series when the file is absent -> callers fall back to RF_FALLBACK_ANNUAL."""
    path = Path(data_dir) / "risk_free_rate.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="rate_annual")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["rate_annual"].sort_index()


# --------------------------------------------------------------------------- #
# Bundle — delegates to the shared engine (parsers/risk_bundle.py, the single
# source for BOTH UIs since Phase D). This wrapper only does the terminal's
# input prep (Frames unpack, twr column derivation, bench-TR reconstruction)
# and echoes bench_tr / latest_dt back into the dict for downstream sections.
# --------------------------------------------------------------------------- #
def _bundle(frames: Frames, bucket_filter, class_filter,
            account_active: bool, class_active: bool,
            benchmark: str = "spy") -> dict:
    positions = frames.positions
    twr = _prepare_portfolio_twr(frames.twr_portfolio)
    bench_tr = hs._bench_tr_series(frames, benchmark)
    latest_dt = (positions["statement_date"].max()
                 if not positions.empty else None)
    b = build_risk_series_bundle(
        positions=positions,
        positions_monthly=frames.positions_monthly,
        latest_dt=latest_dt,
        bucket_filter=bucket_filter,
        class_filter=class_filter,
        account_active=account_active,
        class_active=class_active,
        daily_prices=frames.daily_prices,
        bench_tr=bench_tr,
        twr_portfolio=twr,
    )
    return {**b, "bench_tr": bench_tr, "latest_dt": latest_dt}


# --------------------------------------------------------------------------- #
# Series helpers.
# --------------------------------------------------------------------------- #
def _overlay(series_map) -> list:
    """Put N (name,color,dash,width,Series) onto a COMMON sorted index (union of
    all) so drawOverlayChart — which aligns purely by position — stays correct;
    missing/NaN points serialize as null (the drawer breaks the line there)."""
    idx = None
    for *_rest, s in series_map:
        idx = s.index if idx is None else idx.union(s.index)
    if idx is None or len(idx) == 0:
        return []
    idx = idx.sort_values()
    out = []
    for name, color, dash, width, s in series_map:
        vals = s.reindex(idx).values
        pts = [{"x": pd.Timestamp(d).strftime("%Y-%m-%d"), "v": _jnum(v)}
               for d, v in zip(idx, vals)]
        out.append({"name": name, "color": color, "dash": bool(dash),
                    "width": width, "points": pts})
    return out


def _rolling_sharpe_daily(rets: pd.Series, rf_input, window: int = 252) -> pd.Series:
    """app.py 4348-4367."""
    if rets is None or len(rets) < window + 1:
        return pd.Series(dtype=float)
    log_r = np.log1p(rets)
    ann_ret = np.exp(log_r.rolling(window).sum()) - 1.0
    ann_vol = rets.rolling(window).std(ddof=1) * np.sqrt(252)
    if isinstance(rf_input, pd.Series) and not rf_input.empty:
        union = rf_input.index.union(rets.index).sort_values()
        rf_daily = (rf_input.sort_index().reindex(union).ffill().bfill().loc[rets.index])
        rf_roll = rf_daily.rolling(window).mean()
        return ((ann_ret - rf_roll) / ann_vol).dropna()
    return ((ann_ret - float(rf_input)) / ann_vol).dropna()


def _rolling_vol(rets: pd.Series, window: int = 60) -> pd.Series:
    """app.py 4858-4861."""
    if rets is None or len(rets) < window + 1:
        return pd.Series(dtype=float)
    return rets.rolling(window).std(ddof=1) * np.sqrt(252) * 100.0


# --------------------------------------------------------------------------- #
# Section 1 — Risk-adjusted return (app.py 4287-4443).
# --------------------------------------------------------------------------- #
def _risk_adjusted(b: dict, rf) -> dict:
    monthly = b["monthly"]
    r_all = monthly.dropna()
    r_1y, r_3y = r_all.tail(12), r_all.tail(36)
    dd_1y, dd_3y = window_drawdown_pct(r_1y), window_drawdown_pct(r_3y)

    sharpe_1y = compute_sharpe(r_1y, rf)
    sharpe_3y = compute_sharpe(r_3y, rf)
    sortino_1y = compute_sortino(r_1y, rf)
    sortino_3y = compute_sortino(r_3y, rf)
    calmar_3y = compute_calmar(r_3y, dd_3y)

    spy_m = b["spy_monthly"]
    spy_1y, spy_3y = spy_m.tail(12), spy_m.tail(36)
    spy_dd_3y = window_drawdown_pct(spy_3y)
    spy_sharpe_1y = compute_sharpe(spy_1y, rf) if not spy_1y.empty else np.nan
    spy_sharpe_3y = compute_sharpe(spy_3y, rf) if not spy_3y.empty else np.nan
    spy_sortino_1y = compute_sortino(spy_1y, rf) if not spy_1y.empty else np.nan
    spy_sortino_3y = compute_sortino(spy_3y, rf) if not spy_3y.empty else np.nan
    spy_calmar_3y = compute_calmar(spy_3y, spy_dd_3y) if not spy_3y.empty else np.nan

    rf_1y_mean_pct = (_window_rf(rf, r_1y) * 100.0 if not r_1y.empty else float("nan"))
    rf_label = "DGS3MO" if isinstance(rf, pd.Series) and not rf.empty else "fallback"
    calmar_sub = (f"CAGR / |Max DD| ({min(len(r_3y), 36)} mo)"
                  if len(r_3y) < 36 else "CAGR / |Max DD|")

    tiles = [
        _compare_tile("Sharpe (1Y)", sharpe_1y, spy_sharpe_1y, HIGHER_BETTER, _ratio,
                      f"RF = {rf_1y_mean_pct:.1f}% avg ({rf_label})"),
        _compare_tile("Sharpe (3Y)", sharpe_3y, spy_sharpe_3y, HIGHER_BETTER, _ratio,
                      f"{min(len(r_3y), 36)} months"),
        _compare_tile("Sortino (1Y)", sortino_1y, spy_sortino_1y, HIGHER_BETTER, _ratio,
                      "Downside vol only"),
        _compare_tile("Sortino (3Y)", sortino_3y, spy_sortino_3y, HIGHER_BETTER, _ratio,
                      "Downside vol only"),
        _compare_tile("Calmar (3Y)", calmar_3y, spy_calmar_3y, HIGHER_BETTER, _ratio,
                      calmar_sub),
    ]

    # Rolling 1Y Sharpe chart (app.py 4406-4443).
    port_roll = _rolling_sharpe_daily(b["port_rets"], rf)
    spy_roll = _rolling_sharpe_daily(b["spy_rets"], rf)
    if not port_roll.empty and not spy_roll.empty:
        spy_roll = spy_roll.loc[spy_roll.index >= port_roll.index[0]]
        chart = {"available": True, "series": _overlay([
            ("Portfolio", C_PORT, False, 2.2, port_roll),
            ("SPY", C_BENCH, True, 1.6, spy_roll),
        ])}
    else:
        chart = {"available": False, "series": [], "message": (
            f"Rolling 1Y Sharpe chart needs ≥1 year of daily history "
            f"(have {len(b['port_rets'])} days for portfolio, "
            f"{len(b['spy_rets'])} for SPY).") if len(b["port_rets"]) > 0 else None}
    return {"tiles": tiles, "rolling_sharpe": chart}


# --------------------------------------------------------------------------- #
# Section 2 — Drawdown (app.py 4445-4645).
# --------------------------------------------------------------------------- #
def _tint_cls(p, s, fewer_better: bool):
    """Green/red cell-tint hints for the episodes table (app.py _style_dd_row).
    Returns (port_cls, spy_cls) in {'good','bad',''}."""
    if not (_fin(p) and _fin(s)) or p == s:
        return "", ""
    better_is_port = (p < s) if fewer_better else (p > s)
    return ("good", "bad") if better_is_port else ("bad", "good")


def _drawdown(b: dict) -> dict:
    monthly = b["monthly"]
    r_all = monthly.dropna()
    r_1y, r_3y = r_all.tail(12), r_all.tail(36)
    dd_1y, dd_3y = window_drawdown_pct(r_1y), window_drawdown_pct(r_3y)
    ann_vol_3y = (float(r_3y.std(ddof=1) * np.sqrt(12) * 100.0)
                  if len(r_3y) >= 2 else np.nan)

    dd_full = b["dd_full_pct"]
    cur_dd = float(dd_full.iloc[-1]) if not dd_full.empty else np.nan
    cur_nav = b["nav_latest"]
    cur_dd_dollar = (cur_nav * (cur_dd / 100.0) / (1.0 + cur_dd / 100.0)
                     if _fin(cur_dd) and cur_dd < 0 else 0.0)
    max_dd_itd = float(dd_full.min()) if not dd_full.empty else np.nan
    if not dd_full.empty:
        max_dd_itd_dt = (pd.Timestamp(dd_full.idxmin()).to_period("M")
                         .to_timestamp("M").normalize())
    else:
        max_dd_itd_dt = pd.NaT
    max_dd_1y = float(dd_1y.min()) if not dd_1y.empty else np.nan
    max_dd_3y = float(dd_3y.min()) if not dd_3y.empty else np.nan

    spy_dd_full = b["spy_dd_full_pct"]
    spy_m = b["spy_monthly"]
    spy_1y, spy_3y = spy_m.tail(12), spy_m.tail(36)
    spy_dd_1y, spy_dd_3y = window_drawdown_pct(spy_1y), window_drawdown_pct(spy_3y)
    spy_cur_dd = float(spy_dd_full.iloc[-1]) if not spy_dd_full.empty else np.nan
    spy_max_dd_1y = float(spy_dd_1y.min()) if not spy_dd_1y.empty else np.nan
    spy_max_dd_3y = float(spy_dd_3y.min()) if not spy_dd_3y.empty else np.nan
    spy_max_dd_itd = float(spy_dd_full.min()) if not spy_dd_full.empty else np.nan
    spy_ann_vol_3y = (float(spy_3y.std(ddof=1) * np.sqrt(12) * 100.0)
                      if len(spy_3y) >= 2 else np.nan)

    # Tile 1 — Current DD has the "At peak" special case (app.py 4480-4492).
    if cur_dd < -0.05:
        t1 = _compare_tile("Current DD", cur_dd, spy_cur_dd, LESS_NEG_BETTER, _pct2,
                           f"{fmt_money(abs(cur_dd_dollar))} below peak")
    else:
        t1 = {"label": "Current DD", "value": "At peak",
              "spy": (_pct2(spy_cur_dd) if _fin(spy_cur_dd) else None),
              "dir": "flat", "sub": "Wealth index at all-time high"}
    tiles = [
        t1,
        _compare_tile("Max DD (1Y)", max_dd_1y, spy_max_dd_1y, LESS_NEG_BETTER, _pct1,
                      f"Last {len(r_1y)} months"),
        _compare_tile("Max DD (3Y)", max_dd_3y, spy_max_dd_3y, LESS_NEG_BETTER, _pct1,
                      "Calmar denominator"),
        _compare_tile("Max DD (ITD)", max_dd_itd, spy_max_dd_itd, LESS_NEG_BETTER, _pct1,
                      max_dd_itd_dt.strftime("%b %Y") if pd.notna(max_dd_itd_dt) else "—"),
        _compare_tile("Annualized vol (3Y)", ann_vol_3y, spy_ann_vol_3y, LOWER_BETTER, _pct1,
                      "Monthly std × √12"),
    ]

    # Top-3 episodes (depth <= -2%) — app.py 4507-4592.
    bench_tr = b["bench_tr"]
    wi = b["wealth_index"]
    episodes_block = {"available": False, "rows": [],
                      "message": "No drawdown episode deeper than -2% in history."}
    top3_peaks = []
    if not wi.empty:
        ep_dates = pd.Series(pd.DatetimeIndex(wi.index).to_period("M")
                             .to_timestamp("M").normalize())
        episodes = compute_drawdown_episodes(wi.reset_index(drop=True), ep_dates)
        episodes = [e for e in episodes if e["depth_pct"] <= -2.0]
        episodes.sort(key=lambda e: e["depth_pct"])
        top3 = episodes[:3]
        rows = []
        for e in top3:
            port_under = (e["peak_to_trough_months"]
                          + (e["recovery_months"] if e["recovery_months"] is not None else 0))
            port_under_disp = (f"{port_under}" if e["recovery_months"] is not None
                               else f"{port_under}+ ongoing")
            spy_dec = spy_decline_between(bench_tr, e["peak_date"], e["trough_date"])
            spy_under = spy_months_underwater_from(bench_tr, e["peak_date"],
                                                   trough_date=e["trough_date"])
            spy_under_disp = f"{spy_under}" if spy_under is not None else "ongoing"
            p_dec = float(e["depth_pct"])
            s_dec = float(spy_dec) if np.isfinite(spy_dec) else np.nan
            p_uw = float(port_under)
            s_uw = float(spy_under) if spy_under is not None else np.nan
            dec_p_cls, dec_s_cls = _tint_cls(p_dec, s_dec, fewer_better=False)
            uw_p_cls, uw_s_cls = _tint_cls(p_uw, s_uw, fewer_better=True)
            rows.append({
                "peak": e["peak_date"].strftime("%b %Y"),
                "trough": e["trough_date"].strftime("%b %Y"),
                "recovery": (e["recovery_date"].strftime("%b %Y")
                             if e["recovery_date"] is not None else "Ongoing"),
                "port_decline": _pct1(p_dec), "port_decline_cls": dec_p_cls,
                "spy_decline": (_pct1(s_dec) if np.isfinite(s_dec) else "—"),
                "spy_decline_cls": dec_s_cls,
                "port_uw": port_under_disp, "port_uw_cls": uw_p_cls,
                "spy_uw": spy_under_disp, "spy_uw_cls": uw_s_cls,
            })
            top3_peaks.append(e["peak_date"].strftime("%Y-%m-%d"))
        if rows:
            episodes_block = {"available": True, "rows": rows, "message": None}

    underwater = {"available": False, "series": []}
    if not dd_full.empty:
        underwater = {"available": True, "markers": top3_peaks, "series": _overlay([
            ("Portfolio", C_DD, False, 2.0, dd_full),
            ("SPY", C_BENCH, True, 1.4, spy_dd_full),
        ])}
    return {"tiles": tiles, "episodes": episodes_block, "underwater": underwater}


# --------------------------------------------------------------------------- #
# Section 3 — Concentration / Effective N (app.py 4647-4781).
# --------------------------------------------------------------------------- #
def _fmt_amt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    return f"${v / 1_000:.0f}K"


def _concentration(b: dict) -> dict:
    snap = b["latest_snap"].copy()
    if snap.empty:
        return {"available": False, "tiles": [], "donut": None}
    is_option = snap["asset_class"].astype(str).str.startswith("option")
    snap = snap[(snap["asset_class"] != "cash") & (~is_option)].copy()
    tlh_mask = snap["asset_class"] == "tax_loss_harvesting"
    tlad_mask = snap["bucket"] == "JPM Treasury Ladder"
    snap.loc[tlh_mask, "symbol"] = "SPY"
    snap.loc[tlad_mask, "symbol"] = "SGOV"
    snap = snap.dropna(subset=["symbol"])
    by_ticker = snap.groupby("symbol")["market_value"].sum()
    overrides = {}
    if tlh_mask.any():
        overrides["SPY"] = "SPY (incl. TLH)"
    if tlad_mask.any():
        overrides["SGOV"] = "SGOV (incl. treasury ladder)"

    conc = compute_concentration(by_ticker)
    tiles = []
    if conc["n_positions"] > 0:
        tiles = [
            {"label": "Effective N", "value": f"{conc['effective_n']:.1f}",
             "sub": "1 / Σ wᵢ² — lower = more concentrated"},
            {"label": "Max weight", "value": f"{conc['max_pct']:.1f}%", "sub": ""},
            {"label": "Top-5 weight", "value": f"{conc['top5_pct']:.1f}%", "sub": ""},
        ]

    donut = None
    pos = by_ticker[by_ticker > 0]
    if not pos.empty:
        total_mv = float(pos.sum())
        positions_desc = pos.sort_values(ascending=False)
        pct = positions_desc / total_mv * 100.0
        small_mask = pct < 2.0
        n_total = int(len(positions_desc))
        n_small = int(small_mask.sum())
        if small_mask.any():
            big = positions_desc[~small_mask]
            positions_desc = pd.concat([big, pd.Series({"Other": float(positions_desc[small_mask].sum())})])
        slices = []
        for i, (sym, mv) in enumerate(positions_desc.items()):
            label = "Other" if sym == "Other" else overrides.get(sym, sym)
            color = _OTHER_COLOR if sym == "Other" else _DONUT_PALETTE[i % len(_DONUT_PALETTE)]
            slices.append({"label": label, "pct": float(mv / total_mv * 100.0),
                           "value": _fmt_amt(mv), "color": color})
        other_note = (f" · {n_small} positions < 2% rolled into Other"
                      if small_mask.any() else "")
        donut = {"slices": slices, "total_label": f"${total_mv / 1e6:.2f}M",
                 "n_names": n_total,
                 "head": f"Positions by weight ({n_total} names · total "
                         f"{_fmt_amt(total_mv)}{other_note})"}
    return {"available": bool(tiles or donut), "tiles": tiles, "donut": donut}


# --------------------------------------------------------------------------- #
# Section 4 — Volatility & tail risk, daily (app.py 4783-5142).
# --------------------------------------------------------------------------- #
SPARSE_BIN_DATES_MAX = 3   # hist bins holding <= this many days list their dates


def _sparse_bin_dates(pct: pd.Series, edges: np.ndarray) -> dict:
    """{bin_index_str: [iso dates]} for histogram bins holding 1..MAX
    observations, so the hover tip can answer "when was that tail day".
    Bin membership replicates np.histogram: half-open [e_i, e_i+1) with the
    last bin closed on the right — the date lists always sum to the counts."""
    if pct.empty:
        return {}
    nbins = len(edges) - 1
    idx = np.searchsorted(edges, pct.values, side="right") - 1
    idx[pct.values >= edges[-1]] = nbins - 1
    idx = np.clip(idx, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    out: dict = {}
    for b in np.nonzero((counts > 0) & (counts <= SPARSE_BIN_DATES_MAX))[0]:
        out[str(int(b))] = [str(pd.Timestamp(t).date())
                            for t in pct.index[idx == int(b)]]
    return out


def _daily_vol(b: dict) -> dict:
    port_rets, spy_rets = b["port_rets"], b["spy_rets"]
    r_60, r_252 = port_rets.tail(60), port_rets.tail(252)
    vol_60 = float(r_60.std(ddof=1) * np.sqrt(252) * 100.0) if len(r_60) >= 5 else np.nan
    vol_252 = float(r_252.std(ddof=1) * np.sqrt(252) * 100.0) if len(r_252) >= 5 else np.nan
    var_95, cvar_95 = compute_var_cvar(port_rets, alpha=0.05)
    worst_day = float(port_rets.min()) if not port_rets.empty else np.nan
    worst_day_dt = port_rets.idxmin().date() if not port_rets.empty else None

    spy_60, spy_252 = spy_rets.tail(60), spy_rets.tail(252)
    spy_vol_60 = float(spy_60.std(ddof=1) * np.sqrt(252) * 100.0) if len(spy_60) >= 5 else np.nan
    spy_vol_252 = float(spy_252.std(ddof=1) * np.sqrt(252) * 100.0) if len(spy_252) >= 5 else np.nan
    spy_var_95, spy_cvar_95 = (compute_var_cvar(spy_rets, alpha=0.05)
                               if not spy_rets.empty else (np.nan, np.nan))
    spy_worst_day = float(spy_rets.min()) if not spy_rets.empty else np.nan
    spy_worst_day_dt = spy_rets.idxmin().date() if not spy_rets.empty else None

    def _x100(v):
        return v * 100.0 if _fin(v) else np.nan
    port_wd = worst_day_dt.strftime("%b %d, %Y") if worst_day_dt else "—"
    spy_wd = spy_worst_day_dt.strftime("%b %d, %Y") if spy_worst_day_dt else "—"
    tiles = [
        _compare_tile("Vol 60d", vol_60, spy_vol_60, LOWER_BETTER, _pct1,
                      "Annualized, daily std × √252"),
        _compare_tile("Vol 252d", vol_252, spy_vol_252, LOWER_BETTER, _pct1, "Trailing 1Y"),
        _compare_tile("VaR 95% (1d)", _x100(var_95), _x100(spy_var_95), LESS_NEG_BETTER, _pct2,
                      "Historical, overlapping daily window"),
        _compare_tile("CVaR 95% (1d)", _x100(cvar_95), _x100(spy_cvar_95), LESS_NEG_BETTER, _pct2,
                      "Avg loss in the 5% tail"),
        _compare_tile("Worst day", _x100(worst_day), _x100(spy_worst_day), LESS_NEG_BETTER, _pct2,
                      f"Port: {port_wd} · SPY: {spy_wd}"),
    ]

    # Distribution histogram (app.py 4902-5112) — shared bin edges across both.
    dist = {"available": False}
    rets_pct = (port_rets.dropna() * 100.0)
    hist_n = len(rets_pct)
    if hist_n >= 30:
        spy_pct = pd.Series(dtype=float)
        if not spy_rets.empty:
            shared = rets_pct.index.intersection(spy_rets.index)
            if not shared.empty:
                spy_pct = (spy_rets.loc[shared] * 100.0).dropna()
        mu = float(rets_pct.mean())
        sigma = float(rets_pct.std(ddof=1))
        spy_mu = float(spy_pct.mean()) if not spy_pct.empty else np.nan
        nbins = 50
        x_min, x_max = float(rets_pct.min()), float(rets_pct.max())
        if not spy_pct.empty:
            x_min = min(x_min, float(spy_pct.min()))
            x_max = max(x_max, float(spy_pct.max()))
        bin_w = (x_max - x_min) / nbins if x_max > x_min else 1.0
        edges = np.array([x_min + i * bin_w for i in range(nbins + 1)])
        port_counts = [int(c) for c in np.histogram(rets_pct.values, bins=edges)[0]]
        spy_counts = ([int(c) for c in np.histogram(spy_pct.values, bins=edges)[0]]
                      if not spy_pct.empty else [])
        xline = np.linspace(x_min, x_max, 200)
        pdf = ((1.0 / (sigma * math.sqrt(2.0 * math.pi)))
               * np.exp(-0.5 * ((xline - mu) / sigma) ** 2)) if sigma > 0 else np.zeros_like(xline)
        pdf_scaled = np.maximum(pdf * hist_n * bin_w, 0.5)
        markers = []
        if _fin(mu):
            markers.append({"x": mu, "label": f"Port μ {mu:+.1f}%", "color": C_PORT})
        if _fin(spy_mu):
            markers.append({"x": spy_mu, "label": f"SPY μ {spy_mu:+.1f}%", "color": C_BENCH})
        if _fin(var_95):
            markers.append({"x": var_95 * 100.0, "label": f"VaR 95 {var_95 * 100.0:.1f}%",
                            "color": C_BENCH})
        if _fin(cvar_95):
            markers.append({"x": cvar_95 * 100.0, "label": f"CVaR 95 {cvar_95 * 100.0:.1f}%",
                            "color": C_DD})
        dist = {"available": True, "n_days": hist_n, "x_min": x_min, "x_max": x_max,
                "bin_w": bin_w, "port_counts": port_counts, "spy_counts": spy_counts,
                "port_color": C_PORT, "spy_color": C_BENCH,
                "fit": [{"x": float(xx), "y": float(yy)} for xx, yy in zip(xline, pdf_scaled)],
                "markers": markers,
                "port_dates": _sparse_bin_dates(rets_pct, edges),
                "spy_dates": _sparse_bin_dates(spy_pct, edges)}

    # Rolling 60d vol chart (app.py 5114-5142).
    vol_port = _rolling_vol(port_rets).dropna()
    vol_spy = _rolling_vol(spy_rets).dropna()
    if not vol_port.empty and not vol_spy.empty:
        vol_spy = vol_spy.loc[vol_spy.index >= vol_port.index[0]]
    rolling_vol = {"available": bool(len(vol_port) >= 1),
                   "series": _overlay([
                       ("Portfolio", C_PORT, False, 2.2, vol_port),
                       ("SPY", C_BENCH, True, 1.6, vol_spy),
                   ]) if len(vol_port) >= 1 else []}
    return {"available": True, "tiles": tiles, "distribution": dist,
            "rolling_vol": rolling_vol}


# --------------------------------------------------------------------------- #
# Section 5 — Beta to SPY, daily (app.py 5144-5507).
# --------------------------------------------------------------------------- #
def _ols_pct(d: pd.DataFrame):
    if len(d) < 3 or d["b"].var() <= 0:
        return float("nan"), float("nan")
    slope = float(d["p"].cov(d["b"]) / d["b"].var())
    intercept = float(d["p"].mean() - slope * d["b"].mean())
    return slope, intercept


def _beta(b: dict) -> dict:
    port_rets, spy_rets = b["port_rets"], b["spy_rets"]
    if spy_rets.empty:
        return {"available": False,
                "message": "SPY not in daily_prices.csv — beta unavailable."}
    beta_60 = compute_beta(port_rets, spy_rets, window=60)
    beta_252 = compute_beta(port_rets, spy_rets, window=252)
    up_b, dn_b = compute_up_down_beta(port_rets, spy_rets, window=252)
    alpha_252 = compute_alpha_annual(port_rets, spy_rets, window=252)
    tiles = [
        _metric_tile("β to SPY (60d)", f"{beta_60:.2f}" if _fin(beta_60) else "—",
                     "Recent regime"),
        _metric_tile("β to SPY (252d)", f"{beta_252:.2f}" if _fin(beta_252) else "—",
                     "Trailing 1Y"),
        _metric_tile("Up-β (SPY > 0)", f"{up_b:.2f}" if _fin(up_b) else "—",
                     "Up-days only, 1Y"),
        _metric_tile("Down-β (SPY < 0)", f"{dn_b:.2f}" if _fin(dn_b) else "—",
                     "Hedge truth-teller, down days only, 1Y"),
        _metric_tile("α (OLS intercept vs SPY)",
                     _pct2(alpha_252 * 100.0) if _fin(alpha_252) else "—",
                     "Annualized, 1Y · raw returns (not CAPM)"),
    ]

    # Scatter (app.py 5207-5296) — last 1Y, up/down OLS + β=1 diagonal.
    scatter = {"available": False}
    df_b = (pd.concat([port_rets, spy_rets], axis=1, keys=["p", "b"], sort=True)
            .dropna().tail(252))
    if len(df_b) >= 30:
        d = df_b * 100.0
        up_s, up_i = _ols_pct(d[d["b"] > 0])
        dn_s, dn_i = _ols_pct(d[d["b"] < 0])
        x_min, x_max = float(d["b"].min()), float(d["b"].max())
        def _line(slope, inter, x0, x1):
            if not _fin(slope):
                return None
            return {"x0": x0, "y0": slope * x0 + inter, "x1": x1, "y1": slope * x1 + inter}
        scatter = {
            "available": True, "n": int(len(df_b)),
            "points": [{"bx": float(bx), "py": float(py)}
                       for bx, py in zip(d["b"].values, d["p"].values)],
            "up_line": _line(up_s, up_i, 0.0, max(x_max, 0.0)),
            "dn_line": _line(dn_s, dn_i, min(x_min, 0.0), 0.0),
            "diag": {"x0": x_min, "y0": x_min, "x1": x_max, "y1": x_max},
            "up_label": f"Up-β = {up_s:.2f}" if _fin(up_s) else None,
            "dn_label": f"Down-β = {dn_s:.2f}" if _fin(dn_s) else None,
            "up_color": C_UP, "dn_color": C_DD, "port_color": C_PORT,
        }

    # Rolling β + α charts (app.py 5298-5451).
    rolling_beta_chart = {"available": False, "series": []}
    rolling_alpha_chart = {"available": False, "series": []}
    df_full = (pd.concat([port_rets, spy_rets], axis=1, keys=["p", "b"], sort=True)
               .dropna())
    if len(df_full) >= 252:
        p, bb = df_full["p"], df_full["b"]
        rb_60 = rolling_beta(p, bb, 60).dropna()
        rb_252 = rolling_beta(p, bb, 252).dropna()
        ra_252 = (rolling_alpha_annual(p, bb, 252) * 100.0).dropna()
        up_252, dn_252 = rolling_up_down_beta(p, bb, 252)
        rolling_beta_chart = {"available": True, "baseline": 1.0, "series": _overlay([
            ("β 252d (all days)", C_PORT, False, 2.2, rb_252),
            ("β 60d", C_UP, True, 1.3, rb_60),
            ("Up-β 252d", C_UP, False, 1.6, up_252.dropna()),
            ("Down-β 252d", C_DD, False, 1.6, dn_252.dropna()),
        ])}
        rolling_alpha_chart = {"available": True, "baseline": 0.0, "series": _overlay([
            ("α 252d (annualized)", C_ALPHA, False, 2.0, ra_252),
        ])}
    return {"available": True, "tiles": tiles, "scatter": scatter,
            "rolling_beta": rolling_beta_chart, "rolling_alpha": rolling_alpha_chart}


# --------------------------------------------------------------------------- #
# Static prose (app.py captions / how-to-read / quadrant table).
# --------------------------------------------------------------------------- #
_CAPTION = ("Module 3 — drawdown, risk-adjusted return, concentration, vol/VaR, "
            "beta. Monthly TWR resolution (Pass 1). Daily metrics (60d vol, VaR, "
            "beta) need daily prices. Per-position risk contribution lives on its "
            "own tab.")
_CONC_CAPTION = ("Latest MTM snapshot. Same ticker across multiple accounts is "
                 "consolidated. Cash and options excluded — concentration risk is "
                 "about non-cash single-name exposure. TLH sleeve mapped to SPY; "
                 "treasury ladder mapped to SGOV (economic exposure, not legal "
                 "positions).")
_QUADRANT_HTML = (
    "<p><b>Reading α and β together.</b> The green quadrant (β &lt; 1, α &gt; 0) "
    "is the regime you want — de-risked and still outperforming a β-scaled "
    "passive; the red quadrant (β &gt; 1, α &lt; 0) is the one to fix — amplified "
    "market exposure that also underperforms its own benchmark.</p>"
    "<table class='quadrant'><thead><tr><th></th><th>α &gt; 0</th><th>α &lt; 0</th></tr></thead>"
    "<tbody>"
    "<tr><th>β &gt; 1</th>"
    "<td class='q-neut'>Leveraged market exposure <em>and</em> outperforming what "
    "that leverage would predict. Acceptable if you wanted the risk.</td>"
    "<td class='q-bad'><strong>Worst case.</strong> Amplified market exposure that "
    "<em>also</em> underperforms its own benchmark — two drags stacking.</td></tr>"
    "<tr><th>β &lt; 1</th>"
    "<td class='q-good'><strong>Best case.</strong> De-risked <em>and</em> still "
    "outperforming a β-scaled passive. Best risk-adjusted quadrant.</td>"
    "<td class='q-neut'>De-risked but lagging even the lower benchmark — a small "
    "drag (≈100-150 bps/yr) is often just cash + fees; larger gaps are real "
    "underperformance.</td></tr>"
    "</tbody></table>")


# --------------------------------------------------------------------------- #
# View assembly.
# --------------------------------------------------------------------------- #
def build_risk_view(frames: Frames, *, account: str | list[str] = "all",
                    asset_class: str | list[str] = "all") -> dict:
    """Assemble the GET /api/risk contract. Pure given frames + selections.

    Top-level state mirrors app.py's guard: an empty twr_portfolio -> 'no_twr'
    (the Risk tab's only hard early-return). The Risk tab always uses the latest
    snapshot's weights regardless of the Holdings 'as of' selector — there is no
    as_of param here, only the Account / Asset-class filters (like Performance)."""
    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)
    (bucket_filter, class_filter, _sel_ids,
     account_active, class_active) = _resolve_filter(frames, snap_all, account, asset_class)

    risk_latest_dt = (frames.positions["statement_date"].max()
                      if not frames.positions.empty else None)
    meta = {
        "accounts": acct_opts, "classes": class_opts, "brokers": broker_opts,
        "filter": hs._filter_meta(account, asset_class),
        "account_filter_active": account_active,
        "class_filter_active": class_active,
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "risk_latest_label": (pd.Timestamp(risk_latest_dt).strftime("%b %d, %Y")
                              if risk_latest_dt is not None else ""),
    }

    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {"meta": meta, "caption": _CAPTION,
                "state": {"available": False, "unavailable": "no_twr",
                          "unavailable_message": "Need twr_portfolio.csv for risk metrics."},
                "filter_note": None, "coverage_gaps": None,
                "risk_adjusted": None, "drawdown": None, "concentration": None,
                "daily": None, "beta": None, "quadrant_html": _QUADRANT_HTML,
                "conc_caption": _CONC_CAPTION}

    rf_series = _load_rf(frames.data_dir)
    rf = rf_series if not rf_series.empty else RF_FALLBACK_ANNUAL
    b = _bundle(frames, bucket_filter, class_filter, account_active, class_active)

    # Filtered-source disclosure (app.py 4233-4242).
    filter_note = None
    if b["monthly_source"] == "synthetic":
        filter_note = ("Account / Asset-class filter active. Monthly Pass 1 metrics "
                       "(Sharpe, Sortino, Calmar, monthly drawdown) are synthesized "
                       "from daily returns within each statement-date window — "
                       "numbers can shift slightly from the unfiltered view. Daily "
                       "metrics (Vol, VaR, β) update on the filtered weight set.")

    # Coverage-gap diagnostic (app.py 4244-4285): symbols with > 5% missing days.
    coverage_gaps = None
    sgaps = b["synthesis_gaps"]
    if not sgaps.empty:
        bad = sgaps[sgaps["pct_no_price"] > 5.0]
        if not bad.empty:
            rows = [{"symbol": str(sym),
                     "weight_pct": f"{r['weight_pct']:.2f}%",
                     "n_days_total": int(r["n_days_total"]),
                     "n_days_no_price": int(r["n_days_no_price"]),
                     "pct_no_price": f"{r['pct_no_price']:.1f}%"}
                    for sym, r in bad.iterrows()]
            coverage_gaps = {"n": int(len(bad)),
                             "weight_total": float(bad["weight_pct"].sum()), "rows": rows}

    daily_empty = frames.daily_prices.empty
    daily_block = (None if daily_empty else _daily_vol(b))
    beta_block = (None if daily_empty else _beta(b))

    return {
        "meta": meta, "caption": _CAPTION, "conc_caption": _CONC_CAPTION,
        "state": {"available": True, "unavailable": None, "unavailable_message": None},
        "filter_note": filter_note, "coverage_gaps": coverage_gaps,
        "daily_available": not daily_empty,
        "daily_unavailable_message": (
            "Daily-resolution metrics (vol / VaR / beta) need data/daily_prices.csv."
            if daily_empty else None),
        "risk_adjusted": _risk_adjusted(b, rf),
        "drawdown": _drawdown(b),
        "concentration": _concentration(b),
        "daily": daily_block,
        "beta": beta_block,
        "quadrant_html": _QUADRANT_HTML,
    }
