# terminal/riskcontrib_regime.py
"""Pure builder for the Risk Contribution tab's "Diversification under market
regimes" sub-section (Slice 2b; app.py 6312-6588).

The 3x3 VIX x SPY regime heatmap (cell DR vs the portfolio's trailing-1Y
baseline), the character callout, the 3 KPI tiles, and the cell-detail table.
Reuses riskcontrib_dr.compute_dr_frames for the DR series and the importable
regime engine chain (classify_market_regime -> compute_regime_conditional_dr ->
interpret_regime_dr). 1:1 with app.py on identical weights+daily+vix+long_history.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import config_local as cfg
import theme
from total_return import apply_total_return
from risk_metrics import (
    SPY_STATES,
    VIX_STATES,
    classify_market_regime,
    compute_regime_conditional_dr,
    interpret_regime_dr,
    splice_ticker_history,
)
from terminal.riskcontrib_dr import DR_LONG_W, DR_MED_W, compute_dr_frames

_HEADLINE_COL = f"dr_{DR_MED_W}d"   # 63d — the regime conditioning window
_CALLOUT_LEVEL = {"holds": "ok", "erodes": "info", "weakens": "warn",
                  "breaks": "error", "insufficient": "info"}

_CAPTION_HTML = (  # app.py 6323-6327
    "DR conditioned on market state. <b>Does diversification hold when stress "
    "hits?</b> The bottom-right cell is the test.")

_HOWTO_HTML = (  # app.py 6562-6588
    "<p>Higher DR (greener) = more diversification in that regime. "
    "<b>Diagnostic is the tail cell's delta vs the portfolio's own baseline, "
    "not the absolute number.</b> Baseline is the trailing-1Y mean of the "
    "headline window — anchors <i>what's normal for this portfolio</i> rather "
    "than a theoretical ceiling.</p>"
    "<p><b>Characters</b> (auto-classified from the (stress × high-vol) cell's "
    "delta vs baseline)</p>"
    "<ul><li><b>Holds</b> — tail cell ≥ baseline</li>"
    "<li><b>Erodes</b> — 0.0 to −0.2 below</li>"
    "<li><b>Weakens</b> — −0.2 to −0.4 below</li>"
    "<li><b>Breaks</b> — more than −0.4 below</li></ul>"
    "<p><b>Regime axes</b></p>"
    "<ul><li><b>SPY drawdown</b> (trailing 21d): calm &lt; 3%, correction 3–10%, "
    "stress &gt; 10%. Captures sustained equity pain, not single bad days.</li>"
    "<li><b>VIX z-score</b> (CBOE spot ^VIX against trailing 252d mean/std): "
    "low z ≤ −0.5, normal between, high z &gt; +0.5. Spot index, not VIXY — no "
    "contango distortion.</li></ul>"
    "<p><b>Sparsity note</b> — if the tail cell has N&lt;20, it's likely one "
    "episode rather than a regime. Extend the window (deeper daily_prices.csv "
    "fetch) before drawing conclusions. This is a diagnostic, not a signal "
    "generator.</p>")

_MSG_NO_INPUTS = ("Need data/vix_history.csv (run `py parsers/fetch_vix.py "
                  "--write`) and SPY in either long_history_prices.csv or "
                  "daily_prices.csv for regime conditioning.")
_MSG_NO_OVERLAP = ("Regime labels and DR series have no overlap yet. Run `py "
                   "parsers/fetch_daily_prices.py --years 5 --write` to deepen "
                   "the DR window.")

_ZMIN, _ZMAX = -0.4, 0.4
_DIVERGING = theme.HEATMAP_DIVERGING


def _jnum(v):
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _load_vix(data_dir) -> pd.Series:
    path = Path(data_dir) / "vix_history.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="VIX")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["close"].rename("VIX").sort_index()


def _load_long_history(data_dir) -> pd.DataFrame:
    path = Path(data_dir) / "long_history_prices.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = (pd.read_csv(path, parse_dates=["date"])
           .pivot(index="date", columns="symbol", values="close").sort_index())
    if raw.empty:
        return raw
    spliced = splice_ticker_history(raw, getattr(cfg, "TICKER_HISTORY", {}))
    # Same total-return basis as every other close-matrix loader (spec
    # 2026-08-22-total-return-basis). The TR switch originally missed this
    # loader, leaving the Big-3 correlations + DR-regime conditioning
    # price-only while Streamlit ran TR; parity with
    # holdings_service._load_long_history_prices is test-locked.
    return apply_total_return(spliced, data_dir)


def _hexlerp(c1: str, c2: str, t: float) -> str:
    a, b = c1.lstrip("#"), c2.lstrip("#")
    r = round(int(a[0:2], 16) + (int(b[0:2], 16) - int(a[0:2], 16)) * t)
    g = round(int(a[2:4], 16) + (int(b[2:4], 16) - int(a[2:4], 16)) * t)
    bl = round(int(a[4:6], 16) + (int(b[4:6], 16) - int(a[4:6], 16)) * t)
    return f"#{r:02X}{g:02X}{bl:02X}"


def _diverging_color(z: float) -> str:
    """HEATMAP_DIVERGING interpolated at z, clamped to [-0.4, 0.4] — mirrors
    Plotly's zmin/zmax linear mapping (app.py 6466)."""
    if z is None or not np.isfinite(float(z)):
        z = 0.0
    z = max(_ZMIN, min(_ZMAX, float(z)))
    t = (z - _ZMIN) / (_ZMAX - _ZMIN)
    for i in range(len(_DIVERGING) - 1):
        p0, c0 = _DIVERGING[i]
        p1, c1 = _DIVERGING[i + 1]
        if t <= p1:
            local = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return _hexlerp(c0, c1, local)
    return _DIVERGING[-1][1]


def _cell(row, headline_col: str) -> dict:
    """One heatmap cell from a summary row (Series) or None (empty cell)."""
    if row is None:
        return {"mean": None, "delta": None, "n": 0, "low_n": False,
                "present": False, "text_html": "no obs", "color": None}
    mean = row.get(f"{headline_col}_mean", np.nan)
    n = int(row["n"])
    low_n = bool(row["low_n"])
    if not np.isfinite(mean):
        return {"mean": None, "delta": None, "n": n, "low_n": low_n,
                "present": False, "text_html": "no obs", "color": None}
    delta = row.get(f"{headline_col}_delta_vs_baseline", np.nan)
    flag = "*" if low_n else ""
    dtxt = f"{delta:+.2f}" if np.isfinite(delta) else "—"
    z = float(delta) if np.isfinite(delta) else 0.0
    return {"mean": _jnum(mean), "delta": _jnum(delta), "n": n, "low_n": low_n,
            "present": True,
            "text_html": f"<b>{mean:.2f}{flag}</b><br>Δ {dtxt}<br>N={n}",
            "color": _diverging_color(z)}


def _detail_table(summary: pd.DataFrame) -> dict:
    """The cell-detail expander (app.py 6531-6558): summary sorted by state with
    every mean/median/std/delta column formatted."""
    if summary.empty:
        return {"columns": [], "rows": []}
    disp = summary.copy()
    disp["_so"] = disp["spy_state"].map({s: i for i, s in enumerate(SPY_STATES)})
    disp["_vo"] = disp["vix_state"].map({s: i for i, s in enumerate(VIX_STATES)})
    disp = disp.sort_values(["_so", "_vo"]).drop(columns=["_so", "_vo"])

    def _fmt(col, v):
        if col == "low_n":
            return "preliminary" if v else ""
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "—"
            return f"{v:+.3f}" if col.endswith("_delta_vs_baseline") else f"{v:.3f}"
        return str(v)

    columns = list(disp.columns)
    rows = [[_fmt(c, r[c]) for c in columns] for _, r in disp.iterrows()]
    return {"columns": columns, "rows": rows}


def build_dr_regime(weights: pd.Series, daily: pd.DataFrame, port_rets: pd.Series,
                    vix: pd.Series, long_history: pd.DataFrame) -> dict:
    """The dr_regime block of GET /api/riskcontrib. Pure; 1:1 with app.py
    6312-6588."""
    base = {"available": False, "reason": None, "message": None,
            "caption_html": _CAPTION_HTML, "howto_html": _HOWTO_HTML,
            "window_caption_html": None, "character": None, "heatmap": None,
            "tiles": [], "detail": None}

    f = compute_dr_frames(weights, daily, port_rets)
    if not f["available"]:
        return {**base, "reason": "dr_unavailable"}
    dr_ts, ratio_ts = f["dr_ts"], f["ratio_ts"]

    if not long_history.empty and "SPY" in long_history.columns:
        spy = long_history["SPY"]
    elif "SPY" in daily.columns:
        spy = daily["SPY"]
    else:
        spy = pd.Series(dtype=float)
    if vix is None or vix.empty or spy.empty:
        return {**base, "reason": "no_inputs", "message": _MSG_NO_INPUTS}

    labels = classify_market_regime(spy_series=spy, vix_series=vix)
    hw = _HEADLINE_COL
    tail = (dr_ts[hw].dropna().tail(DR_LONG_W)
            if hw in dr_ts.columns else pd.Series(dtype=float))
    baseline = float(tail.mean()) if not tail.empty else None
    cond = compute_regime_conditional_dr(
        dr_ts, labels, dr_ratio_series=ratio_ts, min_n_per_cell=20,
        headline_window_col=hw, baseline_dr=baseline)
    interp = interpret_regime_dr(cond)

    if cond["n_total"] == 0:
        return {**base, "reason": "no_overlap", "message": _MSG_NO_OVERLAP}

    summary = cond["summary"]
    by_cell = {(r["spy_state"], r["vix_state"]): r for _, r in summary.iterrows()}
    cells = [[_cell(by_cell.get((sp, vx)), hw) for vx in VIX_STATES]
             for sp in SPY_STATES]

    regime_dates = labels.dropna(subset=["regime"]).index
    overlap = regime_dates.intersection(dr_ts[hw].dropna().index)
    w_lo = (overlap.min().date() if len(overlap) else regime_dates.min().date())
    w_hi = (overlap.max().date() if len(overlap) else regime_dates.max().date())
    b_str = (f"{baseline:.2f}"
             if (baseline is not None and np.isfinite(baseline)) else "—")
    min_n = cond["min_n_per_cell"]
    window_caption = (
        f"Window: <b>{w_lo} → {w_hi}</b> · <b>{cond['n_total']:,}</b> labelled "
        f"days · baseline (trailing-1Y mean DR_{DR_MED_W}d) = <b>{b_str}</b>. "
        f"Regime axes: SPY 21d drawdown × CBOE spot ^VIX z-score (trailing "
        f"{DR_LONG_W}d). Cells with N&lt;{min_n} marked <b>*</b> — preliminary.")

    character = {"level": _CALLOUT_LEVEL.get(interp["character"], "info"),
                 "headline": interp["headline"],
                 "asymmetry_note": interp["asymmetry_note"] or None}

    th, asy = cond["tail_highlight"], cond["asymmetry"]
    if th is not None:
        prelim = " — preliminary" if th["low_n"] else ""
        tail_tile = {"label": f"Tail cell DR ({DR_MED_W}d)",
                     "value": f"{th['cell_dr']:.2f}",
                     "sub": f"{th['delta']:+.2f} vs baseline {th['baseline_dr']:.2f} · "
                            f"N={th['n']}{prelim}"}
    else:
        tail_tile = {"label": f"Tail cell DR ({DR_MED_W}d)", "value": "—",
                     "sub": "No observations in (stress × high-vol) cell."}
    base_tile = {"label": f"Baseline DR ({DR_MED_W}d, trailing-1Y)", "value": b_str,
                 "sub": "Trailing-1Y mean of the headline window — what's normal "
                        "for this portfolio."}
    if asy is not None:
        asy_tile = {"label": "Calm-low − stress-high", "value": f"{asy['delta']:+.2f}",
                    "sub": "Positive = stronger in calm than tails (typical); "
                           "negative = stronger in tails (rare, ideal)."}
    else:
        asy_tile = {"label": "Calm-low − stress-high", "value": "—",
                    "sub": "Either anchor cell empty."}

    return {**base, "available": True,
            "window_caption_html": window_caption, "character": character,
            "heatmap": {"rows": list(SPY_STATES), "cols": list(VIX_STATES),
                        "cells": cells,
                        "legend": {"lo": _ZMIN, "hi": _ZMAX, "title": "Δ vs baseline"}},
            "tiles": [tail_tile, base_tile, asy_tile],
            "detail": _detail_table(summary)}
