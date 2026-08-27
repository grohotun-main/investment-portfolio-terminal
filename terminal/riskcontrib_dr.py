# terminal/riskcontrib_dr.py
"""Pure builder for the Risk Contribution tab's "Diversification ratio in
context" sub-section (Slice 2a; app.py 5947-6310).

Shapes the gauge / DR-across-windows / ratio-with-bands visuals on top of the
shared ``risk_metrics.compute_dr_frames`` orchestration (single source for both
UIs since Phase D; re-exported here so ``riskcontrib_regime`` + the tests import
it unchanged), fed the already-parity-covered risk_service bundle weights +
daily_prices. The threshold method
(fixed/percentile/zscore) is precomputed for all three so the front-end swaps
client-side with zero refetch: only the gauge zones, the ratio bands, and the
regime label depend on it — the tiles, the DR multi-line chart, and Max DR are
method-independent.

PARITY: weights + daily are passed to the engines UNMODIFIED, exactly as app.py
does. The DR universe is weights ∩ daily.columns, so phantom daily-only columns
(e.g. a config_local TICKER_HISTORY splice) are excluded automatically — do not
add universe filtering (it would diverge from Streamlit).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import theme
from terminal.risk_service import _overlay   # common-index series serializer

from risk_metrics import (
    DR_LONG_W,
    DR_MED_W,
    DR_SHORT_W,
    classify_dr_regime,
    compute_dr_frames,
    compute_dr_ratio_series,
    compute_dr_regime_thresholds,
    compute_max_dr,
)
_METHODS = ("fixed", "percentile", "zscore")
_METHOD_LABELS = [
    ("fixed", "Fixed (0.90 / 1.10)"),
    ("percentile", "Percentile (20th / 80th)"),
    ("zscore", "Z-score (mean ± 1σ)"),
]
_DOT = {"Stress": "\U0001F534", "Normal": "⚪", "Calm": "\U0001F7E2"}

# Series colours (theme tokens → hex, matching app.py 6138/6147/6156/6183).
_C_SHORT = theme.CHART_BENCH                  # grey  #9AA6B6
_C_MED = theme.CLASS_COLORS["equity_etf"]     # azure #4DA3F5 (dotted)
_C_LONG = theme.CHART_PORTFOLIO               # azure #4DA3F5 (solid, thicker)
_C_CEIL = theme.GAIN                          # teal  #2FD79A (dashed)

_CONTROL_HELP = (
    "Fixed = round-number cuts, comparable across portfolios; Percentile / "
    "Z-score = fit on this portfolio's own ratio history (in-sample). Details "
    "under “How this works.”")

_LEDE_HTML = (
    "Rolling diversification ratio on short (21d) / medium (63d) / long (252d) "
    "windows — the regime signal is the <b>short/long ratio</b>.")

_CAPTION_HTML = (  # app.py 5948-5963 (DR_*_W substituted)
    "DR across short (21d), medium (63d), and long (252d) windows on today's "
    "weights, computed with <b>rolling sample std</b> (not the tile's active "
    "estimator). The regime signal is the <b>short/long DR ratio</b> — when "
    "correlations cluster, short drops faster than long and the ratio falls. The "
    "ceiling is the unconstrained Choueifaty-Coignard maximum on a rolling 252d "
    "correlation matrix (allows negative weights — long-only ceiling is "
    "materially lower). <b>It is a true upper bound only for DR 252d</b>; "
    "short/medium DR are computed on different correlation matrices and can "
    "legitimately sit above this line. Right edge of DR 252d may differ from the "
    "Diversification ratio tile above when the tile's estimator isn't "
    "<code>rolling</code>.")

_HOWTO_HTML = (  # app.py 6270-6309
    "<p><b>Diversification ratio</b> (DR = Σ wᵢσᵢ / σ_p) "
    "measures how much the portfolio's actual vol is reduced by imperfect "
    "cross-asset correlations. DR ≥ 1 always (Cauchy-Schwarz); = 1 when "
    "assets are perfectly correlated; higher means more diversification "
    "benefit.</p>"
    "<p><b>Windows</b> — short (≈1 month) captures the current regime; "
    "long (≈1 year) is the baseline. The medium window (~quarter) is the "
    "intermediate signal.</p>"
    "<p><b>The dial above</b> plots the <b>short/long DR ratio</b> — the "
    "current regime expressed as a single number.</p>"
    "<ul>"
    "<li>\U0001F534 <b>left zone</b> — short DR is well below the long-run "
    "DR, meaning correlations are clustering and diversification is eroding "
    "faster than usual.</li>"
    "<li>⚪ <b>middle zone</b> — normal regime; short and long DR are "
    "roughly in line.</li>"
    "<li>\U0001F7E2 <b>right zone</b> — short DR is well above long-run DR; "
    "names are decorrelating and idiosyncratic moves dominate.</li>"
    "</ul>"
    "<p><b>Threshold methods</b> (controls where the colored zones start)</p>"
    "<ul>"
    "<li><b>Fixed</b> — round-number defaults 0.90 / 1.10. Same regime cuts "
    "across portfolios; easy to compare.</li>"
    "<li><b>Percentile</b> — 20th / 80th percentile of the <i>observed</i> "
    "ratio history. Self-calibrating; \"Stress\" literally means \"ratio is in "
    "the worst 20% of recent observations.\" Best when this portfolio's "
    "operational range differs from the round-number defaults.</li>"
    "<li><b>Z-score</b> — mean ± 1σ of the ratio's distribution. "
    "Adapts to the ratio's own variability; useful when the ratio is unusually "
    "stable (tight band) or unusually noisy (wide band).</li>"
    "</ul>"
    "<p><i>In-sample caveat</i> — Percentile and Z-score thresholds are fit on "
    "the same ratio series they classify: percentile mode mechanically labels "
    "~20% of observations Stress and ~20% Calm by construction. Use those "
    "modes for \"where does today sit in this portfolio's operating range,\" "
    "not for objective regime calls; Fixed is the out-of-sample yardstick.</p>"
    "<p><b>Trailing-1Y mean reference</b> — the dotted line on the main DR "
    "chart is the trailing 1-year mean of DR_long, anchoring \"normal for this "
    "portfolio.\"</p>"
    "<p><b>Max DR (theoretical)</b> = √(1ᵀ R⁻¹ 1) — the "
    "unconstrained Choueifaty-Coignard ceiling on the current asset universe. "
    "<i>Allows negative weights</i> — the long-only ceiling is materially "
    "lower (would need a QP solver). Use the gap as a structural diversification "
    "signal, not a target.</p>")

_UNAVAILABLE_MSG = ("Need ≥252 trading days of daily history on the current "
                    "universe for a DR time series.")


def _jnum(v):
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _ratio_str(v) -> str:
    return (f"{float(v):.2f}×"
            if (v is not None and np.isfinite(float(v))) else "—")


def _control() -> dict:
    return {"label": "Regime thresholds", "default": "fixed", "help": _CONTROL_HELP,
            "options": [{"id": m, "label": lbl} for m, lbl in _METHOD_LABELS]}


def _unavailable() -> dict:
    return {"available": False, "message": _UNAVAILABLE_MSG,
            "lede_html": _LEDE_HTML,
            "caption_html": _CAPTION_HTML, "howto_html": _HOWTO_HTML,
            "tiles": [], "dr_chart": {"series": [], "baseline": None, "title": ""},
            "ratio_series": [], "control": _control(), "thresholds": {}}


def _threshold_block(method: str, ratio_ts: pd.Series, dr_s: float, dr_l: float) -> dict:
    thr = compute_dr_regime_thresholds(ratio_ts, method=method)
    stress, calm = float(thr["stress_thr"]), float(thr["calm_thr"])
    eff = thr["method"]                       # effective method (may flip to fixed)
    reg = classify_dr_regime(dr_s, dr_l, stress_thr=stress, calm_thr=calm)
    ratio_now = reg["ratio"]
    dot = _DOT.get(reg["label"], "⚫")
    obs = ratio_ts.dropna()
    if not obs.empty:
        g_lo = min(0.6, float(obs.min()) - 0.05, stress - 0.05)
        g_hi = max(1.4, float(obs.max()) + 0.05, calm + 0.05)
        y_lo = min(float(obs.min()) - 0.03, stress - 0.05)
        y_hi = max(float(obs.max()) + 0.03, calm + 0.05)
    else:
        g_lo, g_hi = 0.6, 1.4
        y_lo, y_hi = stress - 0.10, calm + 0.10
    title_html = (f"<b>{dot} {reg['label']}</b> · Short ({DR_SHORT_W}d) / "
                  f"Long ({DR_LONG_W}d) DR ratio")
    cap = (
        f"Dial reads <b>{ratio_now:.2f}</b> · "
        f"\U0001F534 below {stress:.2f} = correlations clustering "
        f"(diversification eroding faster than the trailing-year baseline) · "
        f"⚪ {stress:.2f} – {calm:.2f} = normal · "
        f"\U0001F7E2 above {calm:.2f} = names decorrelating (idiosyncratic moves "
        f"dominate). Thresholds set by <i>{eff}</i> method."
        if np.isfinite(ratio_now) else "")
    return {
        "method_label": eff, "stress_thr": _jnum(stress), "calm_thr": _jnum(calm),
        "fallback": thr.get("fallback"),
        "regime": {"label": reg["label"], "dot": dot, "ratio": _jnum(ratio_now)},
        "gauge": {"lo": _jnum(g_lo), "hi": _jnum(g_hi), "value": _jnum(ratio_now),
                  "stress_thr": _jnum(stress), "calm_thr": _jnum(calm),
                  "title_html": title_html, "caption_html": cap},
        "bands": {"y_lo": _jnum(y_lo), "y_hi": _jnum(y_hi),
                  "stress_thr": _jnum(stress), "calm_thr": _jnum(calm),
                  "title": f"Short / Long DR ratio with {eff} regime bands"},
    }


def build_dr_in_context(weights: pd.Series, daily: pd.DataFrame,
                        port_rets: pd.Series) -> dict:
    """The dr_in_context block of GET /api/riskcontrib. Pure; 1:1 with app.py
    5947-6310 on identical weights + daily inputs."""
    f = compute_dr_frames(weights, daily, port_rets)
    if not f["available"]:
        return _unavailable()
    dr_ts, max_dr_ts, ratio_ts = f["dr_ts"], f["max_dr_ts"], f["ratio_ts"]
    dr_s, dr_l = f["dr_s"], f["dr_l"]

    med = dr_ts[f"dr_{DR_MED_W}d"].dropna()
    dr_m = float(med.iloc[-1]) if not med.empty else np.nan
    mx = compute_max_dr(weights, daily, window=DR_LONG_W)

    tiles = [
        {"label": f"DR {DR_SHORT_W}d (short)", "value": _ratio_str(dr_s),
         "sub": "Recent regime"},
        {"label": f"DR {DR_MED_W}d (medium)", "value": _ratio_str(dr_m),
         "sub": "Trailing quarter"},
        {"label": f"DR {DR_LONG_W}d (long)", "value": _ratio_str(dr_l),
         "sub": "Trailing year — baseline"},
        {"label": "Max DR (theoretical)", "value": _ratio_str(mx["max_dr"]),
         "sub": f"Closed-form ceiling · {mx['n_symbols']} symbols · "
                f"unconstrained"},
    ]

    long_recent = dr_ts[f"dr_{DR_LONG_W}d"].dropna().tail(DR_LONG_W)
    one_y_mean = float(long_recent.mean()) if not long_recent.empty else np.nan
    chart_series = [
        (f"DR {DR_SHORT_W}d (short)", _C_SHORT, False, 1.5, dr_ts[f"dr_{DR_SHORT_W}d"]),
        (f"DR {DR_MED_W}d (medium)", _C_MED, True, 1.6, dr_ts[f"dr_{DR_MED_W}d"]),
        (f"DR {DR_LONG_W}d (long)", _C_LONG, False, 2.2, dr_ts[f"dr_{DR_LONG_W}d"]),
    ]
    if not max_dr_ts.empty and bool(max_dr_ts.notna().any()):
        chart_series.append((f"Max DR ({DR_LONG_W}d, rolling ceiling)", _C_CEIL,
                             True, 1.4, max_dr_ts))
    dr_chart = {"series": _overlay(chart_series), "baseline": _jnum(one_y_mean),
                "title": (f"DR across windows (short / medium / long) + 1Y mean + "
                          f"rolling {DR_LONG_W}d ceiling")}

    ratio_series = [{"x": pd.Timestamp(d).strftime("%Y-%m-%d"), "v": _jnum(v)}
                    for d, v in ratio_ts.items()]

    return {
        "available": True, "message": None,
        "lede_html": _LEDE_HTML,
        "caption_html": _CAPTION_HTML, "howto_html": _HOWTO_HTML,
        "tiles": tiles, "dr_chart": dr_chart, "ratio_series": ratio_series,
        "control": _control(),
        "thresholds": {m: _threshold_block(m, ratio_ts, dr_s, dr_l) for m in _METHODS},
    }
