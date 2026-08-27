# terminal/riskcontrib_corr.py
"""Pure builder for the Risk Contribution tab's "Major-holding correlations"
sub-section (Slice 3a; app.py 6590-6872).

Big-3 (SPY/SGOV/GLD) static heatmap on the long-history file (BIL splice for
SGOV's pre-launch stretch, 3y daily fallback) + a rolling-90d pair-correlation
chart with a view-window toggle; the Top-15-by-PCTR static heatmap + a rolling
avg-pairwise line, computed PER ESTIMATOR (the Streamlit Top-15 views hang off
the selected estimator's rc). Reuses the importable correlation engine and the
regime slice's hex-lerp. 1:1 with app.py on identical inputs; the stress-Δ
matrices (6874-6999) are Slice 3b.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import theme
from risk_metrics import (
    compute_conditional_correlation_matrix,
    compute_correlation_matrix,
    compute_rolling_avg_pairwise_correlation,
    compute_rolling_pair_correlations,
    splice_sgov_with_bil,
)
from terminal.riskcontrib_regime import _hexlerp

_CORR = theme.HEATMAP_CORR   # teal (ρ≈0, diversifier) → coral (ρ→1, clustered)
_B3 = ["SPY", "SGOV", "GLD"]


def _jnum(v):
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _interp_scale(scale, t: float) -> str:
    """Interpolate a Plotly-style [[pos, '#hex'], …] scale at t in [0,1]."""
    t = max(0.0, min(1.0, t))
    for i in range(len(scale) - 1):
        p0, c0 = scale[i]
        p1, c1 = scale[i + 1]
        if t <= p1:
            local = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return _hexlerp(c0, c1, local)
    return scale[-1][1]


# Piecewise value→ramp anchors. Real correlation matrices on this book live in
# roughly [-0.2, 1.0] (rarely below -0.6), so a linear [-1, 1] mapping washes
# every cell into the middle of the palette. The stretch hands the crowded
# 0.4..1.0 band over half the ramp while keeping the scale FIXED (matrices stay
# comparable across estimators/dates — deliberately not per-matrix adaptive).
_CORR_STOPS = ((-0.6, 0.0), (-0.2, 0.12), (0.0, 0.22),
               (0.4, 0.45), (0.7, 0.70), (1.0, 1.0))


def _corr_t(rho: float) -> float:
    """Piecewise-linear ramp position in [0,1] for rho, clamped to the anchor
    span (values beyond ±end saturate)."""
    v = max(_CORR_STOPS[0][0], min(_CORR_STOPS[-1][0], float(rho)))
    for (v0, t0), (v1, t1) in zip(_CORR_STOPS, _CORR_STOPS[1:]):
        if v <= v1:
            return t0 if v1 == v0 else t0 + (t1 - t0) * (v - v0) / (v1 - v0)
    return 1.0


def _corr_color(rho):
    """theme.HEATMAP_CORR interpolated at the stretched ramp position.
    None for NaN/None."""
    if rho is None or not np.isfinite(float(rho)):
        return None
    return _interp_scale(_CORR, _corr_t(float(rho)))


def _corr_gradient() -> str:
    """CSS left→right gradient of the correlation palette for the legend bar,
    with each anchor placed at its VALUE-linear position so the bar truthfully
    shows the stretch (compressed negatives, expanded high-ρ band)."""
    lo, hi = _CORR_STOPS[0][0], _CORR_STOPS[-1][0]
    stops = ", ".join(
        f"{_interp_scale(_CORR, t)} {100 * (v - lo) / (hi - lo):.0f}%"
        for v, t in _CORR_STOPS)
    return f"linear-gradient(to right, {stops})"


_DELTA_VMAX = 0.5
_DIVERGING = theme.HEATMAP_DIVERGING   # coral (t=0) → amber (0.5) → teal (t=1)


def _delta_color(delta, vmax: float = _DELTA_VMAX):
    """Stress-Δ color: negative Δ (correlations FELL in stress = good) → teal,
    ≈0 → amber, positive Δ (SPIKED = bad) → coral. Clamped to ±vmax. None for
    NaN/None. Sign-flipped mapping over HEATMAP_DIVERGING (t=0 coral, t=1 teal)."""
    if delta is None or not np.isfinite(float(delta)):
        return None
    d = max(-vmax, min(vmax, float(delta)))
    t = (vmax - d) / (2.0 * vmax)   # d=+vmax → t=0 (coral); d=-vmax → t=1 (teal)
    return _interp_scale(_DIVERGING, t)


def _delta_gradient() -> str:
    """Legend bar left(lo=-vmax=teal) → right(hi=+vmax=coral): reversed DIVERGING."""
    stops = ", ".join(c for _, c in reversed(_DIVERGING))
    return f"linear-gradient(to right, {stops})"


_MAJOR_CAPTION_HTML = (  # app.py 6598-6604
    "How co-aligned the portfolio's biggest pieces are. Two views: the Big-3 "
    "(SPY / SGOV / GLD) on the longest available history — BIL (T-Bill ETF, "
    "launched 2007) stands in for SGOV's pre-2020 stretch — and the Top-15 "
    "weighted positions on the trailing 3y window. Cell colors use a stretched "
    "fixed scale (anchors −0.6 / 0 / 0.4 / 1.0) so differences in the crowded "
    "high-ρ band stay visible; values beyond the ends saturate.")

_ROLL_PAIR_COLORS = {
    "SPY–SGOV": theme.CLASS_COLORS["equity_etf"],
    "SPY–GLD": theme.CLASS_COLORS["gold"],
    "SGOV–GLD": theme.CLASS_COLORS["tax_loss_harvesting"],
}

_MSG_NO_LONG = ("Long-history file missing — run `py parsers/fetch_long_history.py "
                "--write` to enable extended Big-3 correlations. Falling back to the "
                "3y daily-prices window.")
_MSG_NO_OVERLAP_B3 = "Need ≥90 days of overlap for Big-3 correlations."


def _corr_heatmap(corr_df: pd.DataFrame, order: list[str], zmin: float, zmax: float,
                  *, color_fn=None, gradient: str | None = None,
                  value_label: str = "ρ") -> dict | None:
    """Build the drawHeatmap payload (N×N) from a correlation/Δ DataFrame in the
    given symbol order. Cells carry server-computed color + text; NaN pairs are
    present=False. `color_fn(value)->hex|None` + `gradient` + `value_label`
    default to the ρ (HEATMAP_CORR) styling used by the static matrices."""
    if corr_df is None or corr_df.empty:
        return None
    order = [s for s in order if s in corr_df.columns]
    if len(order) < 2:
        return None
    if color_fn is None:
        color_fn = _corr_color
    if gradient is None:
        gradient = _corr_gradient()
    m = corr_df.reindex(index=order, columns=order)
    cells = []
    for rsym in order:
        row = []
        for csym in order:
            v = m.loc[rsym, csym]
            fin = v is not None and np.isfinite(v)
            row.append({"present": bool(fin),
                        "color": color_fn(float(v)) if fin else None,
                        "text_html": (f"{float(v):.2f}" if fin else "")})
        cells.append(row)
    return {"rows": order, "cols": order, "cells": cells,
            "legend": {"title": value_label, "lo": zmin, "hi": zmax,
                       "gradient": gradient}}


def _rolling_block(roll: pd.DataFrame) -> dict | None:
    """Big-3 rolling-corr: the full series + the view-window anchors (app.py
    6687-6754). Client slices by `start`; drawOverlayChart is index-based."""
    if roll is None or roll.empty:
        return None
    last = roll.index.max()
    first = roll.index.min()

    def _iso(ts):
        return pd.Timestamp(ts).date().isoformat()

    raw = [("All", None),
           ("5y", last - pd.DateOffset(years=5)),
           ("3y", last - pd.DateOffset(years=3)),
           ("2y", last - pd.DateOffset(years=2)),
           ("1y", last - pd.DateOffset(years=1)),
           ("2025+", pd.Timestamp("2025-01-01")),
           ("YTD", pd.Timestamp(last.year, 1, 1))]
    window_options = [{"id": lab, "label": lab,
                       "start": (None if start is None else _iso(max(start, first)))}
                      for lab, start in raw]
    series = []
    for col in roll.columns:
        pts = [{"t": _iso(ix), "v": _jnum(v)} for ix, v in roll[col].items()]
        series.append({"name": str(col),
                       "color": _ROLL_PAIR_COLORS.get(str(col)),
                       "points": pts})
    return {"title": "Rolling 90-day correlation", "y_label": "ρ (rolling 90d)",
            "series": series, "window_options": window_options, "default": "All"}


def _has_big3_cols(long_history: pd.DataFrame) -> bool:
    """Whether long_history carries the SPY/SGOV/GLD trio. app.py gates the Big-3
    static (6651-6653) and stress (6915-6917) blocks on column-PRESENCE, not on
    the spliced frame being non-empty — the two differ only for an all-NaN trio,
    but parity requires the column check (all-NaN → insufficient_overlap /
    insufficient_stress, NOT the no-long-history fallback / no_inputs)."""
    return (not long_history.empty
            and {"SPY", "GLD"}.issubset(long_history.columns)
            and "SGOV" in long_history.columns)


def _big3_frame(long_history: pd.DataFrame) -> pd.DataFrame:
    """SGOV/BIL-spliced SPY/SGOV/GLD frame; empty when the trio isn't present.
    Shared by the static (_big3) and stress (_big3_stress) builders so both use
    the identical window."""
    if not _has_big3_cols(long_history):
        return pd.DataFrame()
    sgov_ext = splice_sgov_with_bil(long_history)
    return pd.concat({"SPY": long_history["SPY"].dropna(),
                      "SGOV": sgov_ext.dropna(),
                      "GLD": long_history["GLD"].dropna()},
                     axis=1).dropna(how="all")


def _big3(long_history: pd.DataFrame, daily: pd.DataFrame) -> dict:
    out = {"available": False, "fallback": False, "reason": None, "message": None,
           "caption_html": None, "heatmap": None, "rolling": None}
    if _has_big3_cols(long_history):
        big3 = _big3_frame(long_history)
        if big3.shape[0] >= 90 and big3.shape[1] == 3:
            sgov_native_first = long_history["SGOV"].dropna().index.min()
            proxy = (f" · BIL proxy used for SGOV before "
                     f"{sgov_native_first.strftime('%b %Y')}"
                     if pd.notna(sgov_native_first) else "")
            out["caption_html"] = (
                f"Window: <b>{big3.index.min().strftime('%b %Y')} – "
                f"{big3.index.max().strftime('%b %Y')}</b> "
                f"({len(big3):,} trading days){proxy}.")
            out["heatmap"] = _corr_heatmap(
                compute_correlation_matrix(big3, _B3), _B3, -0.6, 1.0)
            out["rolling"] = _rolling_block(compute_rolling_pair_correlations(
                big3, pairs=[("SPY", "SGOV"), ("SPY", "GLD"), ("SGOV", "GLD")],
                window=90))
            out["available"] = out["heatmap"] is not None
            return out
        out["reason"] = "insufficient_overlap"
        out["message"] = _MSG_NO_OVERLAP_B3
        return out
    # fallback: long_history missing/incomplete -> info + optional 3y daily heatmap
    out["fallback"] = True
    out["reason"] = "no_long_history"
    out["message"] = _MSG_NO_LONG
    if all(s in daily.columns for s in _B3):
        out["heatmap"] = _corr_heatmap(
            compute_correlation_matrix(daily, _B3), _B3, -0.6, 1.0)
        out["caption_html"] = (f"Pearson correlation, daily log returns "
                               f"({len(daily)} trading days).")  # match app.py 6770 number format (no thousands sep)
        out["available"] = out["heatmap"] is not None
    return out


_T15_LEGEND_HTML = (  # app.py 6804-6813
    "<b>Color legend</b> — green = pair moves nearly independently (ρ ≈ 0, a "
    "genuine diversifier and the uncommon-good case in a real portfolio); "
    "yellow → orange = increasingly correlated; deep red = ρ ≈ 1 (diagonal and "
    "tight clusters). Scale anchored at ρ = −0.3 (anything more negative clips); "
    "negative pairwise correlation among holdings is rare in practice.")

_T15_AVG_HOWTO_HTML = (  # app.py 6848-6868
    "<p>Each point is the <b>average of all unique pairwise correlations</b> "
    "among the Top-15 positions over the trailing 90 trading days — one scalar "
    "for \"how clustered the portfolio's biggest names are right now.\"</p>"
    "<ul><li><b>Higher (toward +1)</b> = names moving together; idiosyncratic "
    "diversification eroding — a market-wide shock drags the whole top-15 down "
    "at once. Stress regimes push avg ρ well above its calm baseline.</li>"
    "<li><b>Lower (toward 0)</b> = names moving independently; single-name moves "
    "cancel, so portfolio vol sits well below the weighted-average standalone "
    "vol (this is what drives the Diversification Ratio above 1).</li>"
    "<li><b>What to watch</b> — <i>changes</i>, not the level. A line drifting "
    "up over months warns the book is becoming more single-factor (typically "
    "more SPY-like) even if the dollar weights haven't moved.</li></ul>")

_MSG_INSUFFICIENT_NAMES = ("Not enough Top-15 holdings present in daily prices "
                           "for a correlation matrix.")


def _top15_in_prices(per_symbol: pd.DataFrame, daily: pd.DataFrame) -> list[str]:
    """Top-15-by-PCTR symbols present in daily columns, PCTR order. Shared by the
    static (_top15_for) and stress (_top15_stress) Top-15 builders."""
    return [s for s in (str(x) for x in per_symbol.head(15).index)
            if s in daily.columns]


def _top15_for(per_symbol: pd.DataFrame, n_days, daily: pd.DataFrame,
               port_rets: pd.Series) -> dict:
    """Top-15-by-PCTR correlation heatmap + rolling avg-pairwise line for ONE
    estimator (app.py 6775-6869). `per_symbol` is already PCTR-sorted."""
    out = {"available": False, "reason": None, "message": None,
           "caption_html": None, "heatmap": None, "legend_caption_html": None,
           "avg_roll": None}
    top15_in_prices = _top15_in_prices(per_symbol, daily)
    if len(top15_in_prices) < 3:
        out["reason"] = "insufficient_names"
        out["message"] = _MSG_INSUFFICIENT_NAMES
        return out
    n = max(int(n_days or 0), 1)
    out["caption_html"] = (
        f"Ordered by Total PCTR. {len(top15_in_prices)} of 15 in the "
        f"daily-prices universe — TLH/treasury folds to SPY/SGOV already "
        f"applied, options excluded. Window: trailing <b>{n}</b> trading days.")
    corr = compute_correlation_matrix(daily.tail(n), top15_in_prices)
    out["heatmap"] = _corr_heatmap(corr, top15_in_prices, -0.6, 1.0)
    out["legend_caption_html"] = _T15_LEGEND_HTML

    avg = compute_rolling_avg_pairwise_correlation(daily, top15_in_prices, window=90)
    if not avg.empty and port_rets is not None and not port_rets.empty:
        avg = avg.loc[avg.index >= port_rets.index.min()]
    if not avg.empty:
        pts = [{"t": pd.Timestamp(ix).date().isoformat(), "v": _jnum(v)}
               for ix, v in avg.items()]
        out["avg_roll"] = {
            "title": "Rolling 90-day average pairwise correlation — Top 15",
            "y_label": "ρ (rolling 90d, avg)", "baseline": 0,
            "series": [{"name": "Avg pairwise ρ",
                        "color": theme.CLASS_COLORS["mutual_fund"], "points": pts}],
            "howto_html": _T15_AVG_HOWTO_HTML}
    out["available"] = out["heatmap"] is not None
    return out


_STRESS_SECTION_CAPTION_HTML = (  # app.py 6884-6893 (color words adapted to the terminal palette)
    "Stress day = a day where SPY's daily log return is at least 1.5 σ below its "
    "mean (measured on the full sample). On a normal distribution that's roughly "
    "the worst 7% of days. The matrix below is <b>Δρ = stress-day Spearman ρ − "
    "full-sample Spearman ρ</b> for each pair (coral = correlation <i>spiked</i> "
    "in stress, teal = it <i>fell</i>, amber = unchanged). Per-name Δ down on the "
    "table above is the marginal view; this is the cross-name view.")

_WHY_SPEARMAN_HTML = (  # app.py 6894-6906
    "<b>Why Spearman here, Pearson everywhere else?</b> The conditional ρ on "
    "~15–50 stress days is dominated by single outlier prints under Pearson — one "
    "distressed-equity −60% day can swing a cell by ±0.7 regardless of structural "
    "co-movement. Spearman converts each day's return to its rank within the "
    "sample, so the outlier day still participates but with bounded leverage. No "
    "data is dropped or clipped — the full-sample Δ baseline uses Spearman too so "
    "the subtraction stays apples-to-apples. The static Top-15 matrix above runs "
    "on ~250+ days, where Pearson is fine.")

_STRESS_HOWTO_HTML = (  # app.py 6979-6990 (color words adapted)
    "<p>Each cell is (conditional ρ − full-sample ρ) for that pair. <b>Coral "
    "blocks</b> = pairs whose co-movement <i>spikes</i> on SPY's worst days — "
    "clusters of coral indicate names that crash together (typically: equity-beta "
    "names you'd hoped were diversified). <b>Teal blocks</b> = pairs that "
    "<i>decouple</i> in stress — usually flight-to-quality assets (treasuries, "
    "long-dollar) or genuinely uncorrelated factors. <b>Amber / pale cells</b> = "
    "unchanged from baseline.</p>")

_MSG_INSUFFICIENT_NAMES_STRESS = ("Not enough Top-15 holdings present in daily "
                                  "prices for a stress correlation matrix.")


def _stress_delta_heatmap(delta_df, order):
    return _corr_heatmap(delta_df, order, -_DELTA_VMAX, _DELTA_VMAX,
                         color_fn=_delta_color, gradient=_delta_gradient(),
                         value_label="Δρ")


def _big3_stress(long_history: pd.DataFrame) -> dict:
    """app.py 6913-6948. No long_history trio -> message stays None (app.py's
    block has no else, so nothing renders below the subheader)."""
    out = {"available": False, "reason": None, "message": None,
           "caption_html": None, "heatmap": None}
    if not _has_big3_cols(long_history):
        out["reason"] = "no_inputs"
        return out
    frame = _big3_frame(long_history)
    cond = compute_conditional_correlation_matrix(
        frame, _B3, condition_symbol="SPY", z_threshold=-1.5)
    if not cond["enough"]:
        out["reason"] = "insufficient_stress"
        out["message"] = (f"Need ≥15 stress days to estimate conditional "
                          f"correlations; have {cond['n_stress']}.")
        return out
    out["caption_html"] = (
        f"Full sample: <b>{cond['n_full']:,}</b> trading days · Stress days at ≤ "
        f"−1.5σ: <b>{cond['n_stress']:,}</b> (threshold daily log return ≤ "
        f"{cond['threshold'] * 100:+.2f}%).")
    out["heatmap"] = _stress_delta_heatmap(cond["delta"], _B3)
    out["available"] = out["heatmap"] is not None
    return out


def _top15_stress(per_symbol: pd.DataFrame, n_days, daily: pd.DataFrame) -> dict:
    """app.py 6950-6999. Same Top-15 membership/order + trailing-n_days window as
    the static Top-15 (per estimator)."""
    out = {"available": False, "reason": None, "message": None,
           "caption_html": None, "heatmap": None, "howto_html": _STRESS_HOWTO_HTML}
    order = _top15_in_prices(per_symbol, daily)
    if len(order) < 3:
        out["reason"] = "insufficient_names"
        out["message"] = _MSG_INSUFFICIENT_NAMES_STRESS
        return out
    n = max(int(n_days or 0), 1)
    cond = compute_conditional_correlation_matrix(
        daily.tail(n), order, condition_symbol="SPY", z_threshold=-1.5)
    if not cond["enough"]:
        out["reason"] = "insufficient_stress"
        out["message"] = (f"Need ≥15 stress days to estimate conditional "
                          f"correlations on the trailing-{n}d window; have "
                          f"{cond['n_stress']}.")
        return out
    out["caption_html"] = (
        f"Window: trailing <b>{n}</b> trading days · Stress days at ≤ −1.5σ: "
        f"<b>{cond['n_stress']:,}</b> (threshold daily log return ≤ "
        f"{cond['threshold'] * 100:+.2f}%).")
    out["heatmap"] = _stress_delta_heatmap(cond["delta"], order)
    out["available"] = out["heatmap"] is not None
    return out


def build_stress_correlations(vol_blocks: dict, daily: pd.DataFrame,
                              long_history: pd.DataFrame) -> dict:
    """The `correlations.stress` block (app.py 6874-6999). Big-3 once; Top-15 per
    estimator (like the static Top-15)."""
    top15 = {est_id: _top15_stress(rc["per_symbol"], rc.get("n_days"), daily)
             for est_id, rc in vol_blocks.items()}
    return {"caption_html": _STRESS_SECTION_CAPTION_HTML,
            "why_spearman_html": _WHY_SPEARMAN_HTML,
            "big3": _big3_stress(long_history),
            "top15": top15}


def build_major_correlations(vol_blocks: dict, daily: pd.DataFrame,
                             long_history: pd.DataFrame,
                             port_rets: pd.Series) -> dict:
    """The `correlations` block of GET /api/riskcontrib. Pure; 1:1 with app.py
    6590-6872. Top-15 is keyed per estimator; Big-3 is computed once."""
    top15 = {est_id: _top15_for(rc["per_symbol"], rc.get("n_days"), daily, port_rets)
             for est_id, rc in vol_blocks.items()}
    return {"major": {"caption_html": _MAJOR_CAPTION_HTML,
                      "big3": _big3(long_history, daily),
                      "top15": top15},
            "stress": build_stress_correlations(vol_blocks, daily, long_history)}
