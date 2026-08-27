# terminal/factor_service.py
"""Pure data seam for the MERIDIAN Terminal "Factor analysis" tab.

Re-expresses app.py._render_factor_body (3745-4184). Every factor number lives
in the importable, Streamlit-free parsers/factor_regression.py (+
synthesize_portfolio_returns from risk_metrics) — this module reproduces the
SAME inputs the Streamlit body uses (the monthly portfolio TWR, the Ken French
monthly + daily factor files, the unfiltered latest-snapshot weights, and the
daily price matrix) and shapes the engine output into a JSON-native,
allow_nan=False-clean view dict. Numbers match Streamlit 1:1 by construction.

Whole-book: the regressions always run on the full broker-filtered book — the
Account / Asset-class selects never apply — so GET /api/factor takes no params
(like /api/health, /api/income), no empty-param-422 surface.

Precompute-all: every window x model block is computed here; the front-end swaps
Window / Model / Attribution-view / Rolling-window entirely client-side with no
refetch. The tab is fully date-stable (window = trailing-from-last-aligned-month;
weights = latest statement snapshot; per-holding window_days count back from the
last daily-price date), so the golden needs no asof thread (unlike Income).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import theme
from terminal import holdings_service as hs

from factor_regression import (FACTOR_LABELS, MODELS,
                               align_returns_with_factors,
                               align_twr_with_factors, attribution,
                               attribution_timeseries, per_holding_regressions,
                               rolling_factor_regressions,
                               run_factor_regression)
from monthly_normalize import slice_as_of_month
from risk_metrics import synthesize_portfolio_returns

# --- Control vocabulary (mirror app.py's selectbox / radio options) --------- #
WINDOWS = ["Full history", "Last 60 months", "Last 36 months"]
WINDOW_MONTHS = {"Full history": None, "Last 60 months": 60, "Last 36 months": 36}
WINDOW_DAYS = {"Full history": None, "Last 60 months": 1260, "Last 36 months": 756}
MODEL_NAMES = list(MODELS)                 # CAPM, FF3, Carhart 4, FF5, FF5 + Mom
DEFAULT_MODEL = MODEL_NAMES[-1]            # app.py radio index=len(MODELS)-1
ROLL_WINDOWS = [24, 36, 60]
DEFAULT_ROLL = 36                          # app.py selectbox index=1
ATTR_VIEWS = ["Cumulative", "Monthly"]
DEFAULT_ATTR_VIEW = "Cumulative"           # app.py radio first option

# --- Series colors (theme tokens; literal hues for the per-factor palette) -- #
FACTOR_COLORS = {
    "rf": theme.TEXT_MUTED,
    "mkt_rf": theme.ACCENT,
    "smb": theme.GAIN,
    "hml": "#8B7CF6",
    "rmw": "#E6B450",
    "cma": "#FB7185",
    "mom": "#38BDF8",
}
UNEXPLAINED_COLOR = theme.TEXT_SECONDARY   # dashed in the cumulative view
TOTAL_COLOR = theme.TEXT_PRIMARY           # the thick Total (arith.) line
ALPHA_COLOR = "#E6B450"                     # dashed alpha line in the rolling chart


def _jnum(v) -> float | None:
    """JSON-safe float for raw numeric fields: NaN / inf -> None so the
    allow_nan=False route never sees an invalid token. (Formatted display
    strings keep their literal 'nan' for byte-parity with st.metric.)"""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _load_ff(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly, daily) Ken French factor frames — pure read, replicating
    app.py.load_ff_factors / load_ff_factors_daily (empty frame if absent)."""
    d = Path(data_dir)
    try:
        ff_m = pd.read_csv(d / "ff_factors_monthly.csv", dtype={"month": str})
    except FileNotFoundError:
        ff_m = pd.DataFrame()
    try:
        ff_d = pd.read_csv(d / "ff_factors_daily.csv", dtype={"date": str})
    except FileNotFoundError:
        ff_d = pd.DataFrame()
    return ff_m, ff_d


def _factor_weights(frames: hs.Frames) -> pd.Series:
    """The unfiltered current-holdings weight Series (app.py:2128-2132): the
    latest statement snapshot folded TLH->SPY / Treasury-rung->duration ETF /
    uncovered->SGOV via the already-parity-covered hs._snapshot_weights."""
    pos = frames.positions
    if pos.empty or frames.daily_prices.empty:
        return pd.Series(dtype=float)
    risk_latest_dt = pos["statement_date"].max()
    snap = slice_as_of_month(frames.positions_monthly, risk_latest_dt)
    return hs._snapshot_weights(snap, frames.daily_prices)


# --------------------------------------------------------------------------- #
# Orchestration seams shared with app.py's tab body (Phase D consolidation).
# --------------------------------------------------------------------------- #
def factor_results(twr: pd.DataFrame, ff_monthly: pd.DataFrame,
                   window_label: str) -> tuple[pd.DataFrame, dict]:
    """Align monthly TWR with the factor file over the window, then run every
    model. Returns (aligned, {model: FactorResult|None}); results is {} when the
    aligned window is empty. Shared by app.py's tab body and _window_block so the
    regression inputs + per-model fits are computed one way."""
    aligned = align_twr_with_factors(twr, ff_monthly, WINDOW_MONTHS[window_label])
    results = ({m: run_factor_regression(aligned, m) for m in MODEL_NAMES}
               if not aligned.empty else {})
    return aligned, results


def per_holding_result(weights: pd.Series, daily_prices: pd.DataFrame,
                       ff_daily: pd.DataFrame, model: str,
                       window_days: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-holding daily factor regression: roster (weights ∩ price columns) →
    per_holding_regressions → weight column → sort by weight desc. Returns
    (disp, skipped); disp is empty when no holding clears the fit cut. Callers
    guarantee ff_daily / daily_prices / weights are non-empty (each renders its
    own not-available message for those)."""
    roster = [s for s in weights.index if s in daily_prices.columns]
    ph_table, ph_skipped = per_holding_regressions(
        daily_prices[roster], ff_daily, model, window_days)
    if ph_table.empty:
        return ph_table, ph_skipped
    disp = ph_table.copy()
    disp.insert(1, "weight", disp["symbol"].map(weights))
    disp = disp.sort_values("weight", ascending=False)
    return disp, ph_skipped


def cross_check_daily(weights: pd.Series, daily_prices: pd.DataFrame,
                      ff_daily: pd.DataFrame, model: str,
                      window_days: int | None) -> tuple[object | None, int]:
    """Daily synthetic-portfolio regression for the monthly↔daily cross-check:
    synthesize current-weight returns → align to daily factors → regress at
    ppy=252. Returns (res_d, n_aligned_d); res_d is None on too few aligned days
    (n lets the caller word the message). Callers guarantee ff_daily / weights /
    daily_prices non-empty and the monthly result present."""
    port_daily = synthesize_portfolio_returns(weights, daily_prices)
    aligned_d = align_returns_with_factors(port_daily, ff_daily, window_days)
    res_d = run_factor_regression(aligned_d, model, periods_per_year=252)
    return res_d, len(aligned_d)


# --------------------------------------------------------------------------- #
# Alpha-by-model strip (app.py 3805-3829).
# --------------------------------------------------------------------------- #
def _strip_entry(model: str, res) -> dict:
    if res is None:
        return {"model": model, "available": False, "value": "—", "delta": None,
                "help": ("Not enough aligned months for this model on the "
                         "selected window.")}
    lo, hi = res.alpha_ci_annual
    return {
        "model": model, "available": True,
        "value": f"{res.alpha_annual * 100:+.1f}%",
        "delta": f"±{(hi - lo) / 2 * 100:.1f} pp",
        "help": (f"t = {res.alpha_t:.2f} · R² = {res.r2:.2f} · 95% CI "
                 f"{lo * 100:+.1f}% … {hi * 100:+.1f}%"),
    }


# --------------------------------------------------------------------------- #
# Detail: beta table (app.py 3863-3876).
# --------------------------------------------------------------------------- #
_BETA_COLUMNS = ["Factor", "β", "SE", "t", "Significant (|t|>2)"]


def _beta_table(res) -> tuple[dict, list]:
    rows, numeric = [], []
    for c in res.factors:
        b, s, t = res.betas[c], res.se[c], res.tstats[c]
        rows.append({
            "Factor": FACTOR_LABELS[c], "β": f"{b:+.2f}", "SE": f"{s:.2f}",
            "t": f"{t:+.2f}", "Significant (|t|>2)": "✓" if abs(t) > 2 else "",
        })
        numeric.append({"factor": c, "beta": _jnum(b), "se": _jnum(s),
                        "t": _jnum(t)})
    return {"columns": _BETA_COLUMNS, "rows": rows}, numeric


# --------------------------------------------------------------------------- #
# Attribution waterfall (app.py 3878-3904). attribution() labels differ from
# the over-time chart's labels — keep each section's labels verbatim.
# --------------------------------------------------------------------------- #
def _waterfall(res) -> dict:
    contribs = attribution(res)                 # [(label, annual_decimal), ...]
    total = res.mean_return_monthly * 12.0
    items = [{"label": lab, "value_pp": _jnum(v * 100), "text": f"{v * 100:+.1f}"}
             for lab, v in contribs]
    return {
        "items": items,
        "total_label": "Mean return (arith.)",
        "total_pp": _jnum(total * 100), "total_text": f"{total * 100:+.1f}",
        "caption": (f"Decomposes the window's arithmetic mean monthly return ×12 "
                    f"({total * 100:+.1f}%). This will NOT match the geometric "
                    "annualized TWR on the Performance tab — arithmetic means "
                    "ignore compounding. Expected, not a bug."),
    }


# --------------------------------------------------------------------------- #
# Attribution over time (app.py 3905-3961). One per-period series set (raw
# decimals); the client ×100 + (for Cumulative) cumsums and derives the Total.
# --------------------------------------------------------------------------- #
def _attribution(res, aligned: pd.DataFrame) -> dict:
    ats = attribution_timeseries(res, aligned)
    x = [str(m) for m in ats["month"].tolist()]
    series = [{"key": "rf", "name": "Risk-free", "color": FACTOR_COLORS["rf"],
               "values": [_jnum(v) for v in ats["rf"].tolist()]}]
    for c in res.factors:
        series.append({"key": f"contrib_{c}", "name": FACTOR_LABELS[c],
                       "color": FACTOR_COLORS[c],
                       "values": [_jnum(v) for v in ats[f"contrib_{c}"].tolist()]})
    series.append({"key": "unexplained", "name": "Unexplained (α + residual)",
                   "color": UNEXPLAINED_COLOR, "dash": True,
                   "values": [_jnum(v) for v in ats["unexplained"].tolist()]})
    return {
        "x": x, "series": series,
        "total_label": "Total (arith.)", "total_color": TOTAL_COLOR,
        "caption": ("Static betas from the regression above — the same "
                    "constant-exposure assumption as the waterfall (its bars are "
                    "these components' means ×12). Components sum exactly to each "
                    "month's return; unexplained (α + residual) trends at slope α "
                    "in the cumulative view — steady climb = persistent alpha, "
                    "one step = one lucky stretch. Arithmetic running sum: will "
                    "NOT match the geometric cumulative TWR on the Performance "
                    "tab. Expected, not a bug."),
    }


# --------------------------------------------------------------------------- #
# Rolling factor betas (app.py 3962-4007). Precompute all 3 roll windows.
# --------------------------------------------------------------------------- #
def _rolling(aligned: pd.DataFrame, model: str, factors) -> dict:
    k = len(factors)
    by_roll = {}
    for w in ROLL_WINDOWS:
        if len(aligned) < w + 2:
            by_roll[str(w)] = {
                "available": False,
                "message": (f"Need at least {w + 2} aligned months for a {w}m "
                            f"rolling window — have {len(aligned)}."),
            }
            continue
        roll = rolling_factor_regressions(aligned, model, w)
        rx = [str(m) for m in roll["month"].tolist()]
        rseries = [{"name": f"β {FACTOR_LABELS[c]}", "color": FACTOR_COLORS[c],
                    "values": [_jnum(v) for v in roll[f"beta_{c}"].tolist()]}
                   for c in factors]
        rseries.append({"name": "α (annualized)", "color": ALPHA_COLOR,
                        "dash": True,
                        "values": [_jnum(v) for v in roll["alpha_annual"].tolist()]})
        by_roll[str(w)] = {
            "available": True,
            "low_obs_warning": (
                f"{w} months across {k} factors is fewer than 10 observations "
                "per parameter — treat the rolling lines as indicative, not "
                "precise." if w / k < 10 else None),
            "x": rx, "series": rseries,
        }
    return {
        "by_roll": by_roll,
        "caption": ("Plain OLS per window, window end-dated. α is the dashed "
                    "line, annualized arithmetically (×12) like the strip above; "
                    "a gap means a window the regression structurally skipped."),
    }


# --------------------------------------------------------------------------- #
# Per-holding daily betas (app.py 4014-4086).
# --------------------------------------------------------------------------- #
def _per_holding(weights: pd.Series, daily_prices: pd.DataFrame,
                 ff_daily: pd.DataFrame, model: str, window_days: int | None,
                 window_label: str) -> dict:
    if ff_daily.empty:
        return {"available": False,
                "message": ("Daily factor file not fetched yet — run "
                            "`py parsers/fetch_ff_factors.py --write` (the "
                            "fetcher now writes monthly + daily files).")}
    if daily_prices.empty or weights.empty:
        return {"available": False,
                "message": ("Needs daily prices and a holdings snapshot (see "
                            "the Risk tab).")}
    disp, ph_skipped = per_holding_result(weights, daily_prices, ff_daily, model,
                                          window_days)
    if disp.empty:
        return {"available": False,
                "message": ("No holding clears the cut on this window — each "
                            "needs 126+ aligned trading days and a "
                            "non-degenerate fit.")}
    fcols = list(MODELS[model])
    cols = (["Symbol", "Weight", "Days (n)"]
            + [f"β {FACTOR_LABELS[c]}" for c in fcols]
            + ["α (ann.)", "α t", "R²", "Significant (|t|>2)"])
    rows = []
    for _, r in disp.iterrows():
        sig = ", ".join(FACTOR_LABELS[c].split(" ")[0] for c in fcols
                        if abs(r[f"t_{c}"]) > 2) or "—"
        row = {"Symbol": str(r["symbol"]), "Weight": f"{r['weight']:.1%}",
               "Days (n)": f"{int(r['n'])}"}
        for c in fcols:
            row[f"β {FACTOR_LABELS[c]}"] = f"{r[f'beta_{c}']:+.2f}"
        row["α (ann.)"] = f"{r['alpha_annual']:+.1%}"
        row["α t"] = f"{r['alpha_t']:+.1f}"
        row["R²"] = f"{r['r2']:.2f}"
        row["Significant (|t|>2)"] = sig
        rows.append(row)
    out = {
        "available": True, "table": {"columns": cols, "rows": rows},
        "skipped_caption": None,
        "caption": (
            f"Daily total-return closes × French daily factors, window = "
            f"{window_label.lower()}. Plain-OLS t-stats are optimistic at daily "
            "frequency (volatility clustering). At |t|>2, each cell has a "
            "~1-in-20 chance of printing significant under the null. "
            "Distributions are reinvested at the ex-date close (per-ticker "
            "dividend files); a name without a dividend file stays price-only "
            "and its α is understated by its distribution yield. Roster uses "
            "the Risk-tab modelable-book folding (TLH→SPY, Treasury rungs→"
            "duration ETFs, uncovered→SGOV); the Holdings filter does not apply."),
    }
    if not ph_skipped.empty:
        out["skipped_caption"] = (
            "Skipped (too few aligned days or a degenerate fit): "
            + ", ".join(f"{r.symbol} ({r.n}d)"
                        for r in ph_skipped.itertuples()))
    return out


# --------------------------------------------------------------------------- #
# Cross-check: monthly TWR vs daily synthetic (app.py 4088-4146).
# --------------------------------------------------------------------------- #
_CROSS_COLUMNS = ["Metric", "Monthly (real TWR)",
                  "Daily (current wts, synthetic)", "Δ"]


def _cross_check(res_m, weights: pd.Series, daily_prices: pd.DataFrame,
                 ff_daily: pd.DataFrame, model: str,
                 window_days: int | None) -> dict:
    if ff_daily.empty or weights.empty or daily_prices.empty:
        return {"available": False,
                "message": ("Needs the daily factor file and a current-weights "
                            "snapshot (see the section above).")}
    if res_m is None:
        return {"available": False,
                "message": ("Monthly regression unavailable on this window — "
                            "pick a smaller model or a longer window.")}
    res_d, n_aligned_d = cross_check_daily(weights, daily_prices, ff_daily, model,
                                           window_days)
    if res_d is None:
        return {"available": False,
                "message": (f"Too few aligned daily observations "
                            f"({n_aligned_d}) for {model}.")}
    rows = [{"Metric": f"β {FACTOR_LABELS[c]}",
             "Monthly (real TWR)": f"{res_m.betas[c]:+.2f}",
             "Daily (current wts, synthetic)": f"{res_d.betas[c]:+.2f}",
             "Δ": f"{res_d.betas[c] - res_m.betas[c]:+.2f}"}
            for c in res_m.factors]
    rows += [
        {"Metric": "α (ann.)",
         "Monthly (real TWR)": f"{res_m.alpha_annual:+.1%}",
         "Daily (current wts, synthetic)": f"{res_d.alpha_annual:+.1%}", "Δ": "—"},
        {"Metric": "R²", "Monthly (real TWR)": f"{res_m.r2:.2f}",
         "Daily (current wts, synthetic)": f"{res_d.r2:.2f}", "Δ": "—"},
        {"Metric": "n", "Monthly (real TWR)": f"{res_m.n}",
         "Daily (current wts, synthetic)": f"{res_d.n}", "Δ": "—"},
    ]
    return {
        "available": True, "table": {"columns": _CROSS_COLUMNS, "rows": rows},
        "caption": ("Same model, two lenses. Monthly = the real portfolio's "
                    "reconciled TWR (what was actually held, incl. trades/options/"
                    "cash). Daily = today's weights frozen and projected over "
                    "history (total-return synthesis, the Risk-tab convention). "
                    "βs should roughly agree when current holdings resemble the "
                    "held history; αs are NOT comparable — the daily column "
                    "freezes today's weights and ignores every timing/trading "
                    "effect the real TWR lived through."),
    }


# --------------------------------------------------------------------------- #
# Block assembly.
# --------------------------------------------------------------------------- #
def _model_block(model: str, aligned: pd.DataFrame, results: dict,
                 weights: pd.Series, daily_prices: pd.DataFrame,
                 ff_daily: pd.DataFrame, window_days: int | None,
                 window_label: str, factor_through: str | None) -> dict:
    """One window x model block. The detail/beta/waterfall/attribution/rolling
    sections gate on the monthly fit; per-holding + cross-check render regardless
    (app.py renders them outside the res-None branch)."""
    res = results[model]
    k = len(MODELS[model])
    block: dict = {}
    if res is None:
        block.update({
            "available": False,
            "too_few_message": (
                f"Only {len(aligned)} aligned month(s) — too few for {model} "
                f"({k} factors need ≥ {k + 2}). Pick a smaller model or a longer "
                "window."),
            "low_obs_warning": None, "metrics": None, "window_caption": None,
            "beta_table": None, "beta_numeric": None, "waterfall": None,
            "attribution": None, "rolling": None,
        })
    else:
        bt, bn = _beta_table(res)
        block.update({
            "available": True, "too_few_message": None,
            "low_obs_warning": (
                f"{res.n} months across {k} factors is fewer than 10 "
                "observations per parameter — treat betas as indicative, not "
                "precise." if res.n / k < 10 else None),
            "metrics": {"n": f"{res.n}", "r2": f"{res.r2:.2f}",
                        "adj_r2": f"{res.adj_r2:.2f}"},
            "window_caption": (
                f"Aligned window {res.months[0]} → {res.months[-1]} · factor "
                f"file through {factor_through} (French publishes with a ~1-2 "
                "month lag; the regression trims to common months)."),
            "beta_table": bt, "beta_numeric": bn,
            "waterfall": _waterfall(res),
            "attribution": _attribution(res, aligned),
            "rolling": _rolling(aligned, model, list(res.factors)),
        })
    block["per_holding"] = _per_holding(weights, daily_prices, ff_daily, model,
                                        window_days, window_label)
    block["cross_check"] = _cross_check(res, weights, daily_prices, ff_daily,
                                        model, window_days)
    return block


def _window_block(window_label: str, twr: pd.DataFrame, ff_monthly: pd.DataFrame,
                  ff_daily: pd.DataFrame, weights: pd.Series,
                  daily_prices: pd.DataFrame, factor_through: str | None) -> dict:
    wdays = WINDOW_DAYS[window_label]
    aligned, results = factor_results(twr, ff_monthly, window_label)
    if aligned.empty:
        return {"aligned_empty": True,
                "aligned_message": ("No overlapping months between the TWR series "
                                    "and the factor file — refresh the factor "
                                    "data."),
                "strip": [], "models": {}}
    strip = [_strip_entry(m, results[m]) for m in MODEL_NAMES]
    models = {m: _model_block(m, aligned, results, weights, daily_prices,
                              ff_daily, wdays, window_label, factor_through)
              for m in MODEL_NAMES}
    return {"aligned_empty": False, "aligned_message": None,
            "strip": strip, "models": models}


def _control_vocab() -> dict:
    return {
        "windows": WINDOWS, "default_window": WINDOWS[0],
        "models": MODEL_NAMES, "default_model": DEFAULT_MODEL,
        "roll_windows": ROLL_WINDOWS, "default_roll": DEFAULT_ROLL,
        "attr_views": ATTR_VIEWS, "default_attr_view": DEFAULT_ATTR_VIEW,
    }


_CAPTION = ("Monthly portfolio TWR regressed on the Ken French research "
            "factors. Always the full real portfolio — Account / Asset-class "
            "filters never apply to this tab.")

# Methodology & caveats expander (app.py 4148-4184 markdown -> HTML).
_METHODOLOGY = (
    "<ul>"
    "<li><b>Regression:</b> OLS of monthly <b>excess</b> TWR "
    "(<code>return_pct − RF</code>) on the model's factors, plain OLS standard "
    "errors, 95% CI via a t-table (no scipy). RF here is the French <b>1-month "
    "T-bill</b>, not the DGS3MO series the Sharpe tiles use — conventions differ "
    "by a few bps.</li>"
    "<li><b>Data:</b> Ken French Data Library (keyless). The 5-factor 2×3 file "
    "supplies Mkt-RF/SMB/HML/RMW/CMA/RF for every model (its SMB differs "
    "microscopically from the 3-factor file's); the momentum file supplies "
    "Mom.</li>"
    "<li><b>Coverage:</b> factors are US-equity long-short academic portfolios. "
    "Cash, options, and any non-US sleeve load into alpha/residual by "
    "construction.</li>"
    "<li><b>Sample size:</b> ~74 monthly observations on full history — CIs on "
    "5-6 factor models are wide; treat the alpha point estimate with respect for "
    "its error bar.</li>"
    "<li><b>The headline question:</b> if CAPM alpha is large but FF5+Mom alpha "
    "is near zero with the CI straddling zero, the 'alpha' was factor exposure "
    "(size/value/momentum tilts), not stock-picking skill.</li>"
    "<li><b>Daily sections:</b> per-holding betas and the cross-check use the "
    "French DAILY files and total-return closes (distributions reinvested at "
    "the ex-date close; a name without a dividend file stays price-only) at "
    "ppy=252. Plain OLS at daily frequency understates SEs under volatility "
    "clustering — read those t-stats as optimistic.</li>"
    "<li><b>Roster:</b> Risk-tab modelable-book folding (TLH sleeve→SPY, "
    "Treasury rungs→duration ETFs, uncovered→SGOV); Holdings filter "
    "intentionally ignored.</li>"
    "<li><b>Attribution over time:</b> the waterfall's monthly time series under "
    "the same static full-window betas — ret_t = RF_t + Σ β·f_t + (α + "
    "residual_t). Cumulative view links arithmetically (running sum of monthly "
    "contributions), so it will not match the geometric cumulative TWR.</li>"
    "</ul>"
)


def build_factor_view(frames: hs.Frames) -> dict:
    """Assemble the GET /api/factor contract. Pure given frames; no params,
    no asof (date-stable). Top-level empty states mirror app.py's guards:
    twr.empty -> 'no_twr'; ff_monthly.empty -> 'no_factors'."""
    twr = frames.twr_portfolio
    ff_monthly, ff_daily = _load_ff(frames.data_dir)
    daily_prices = frames.daily_prices
    weights = _factor_weights(frames)

    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)
    factor_through = (str(ff_monthly["month"].max())
                      if not ff_monthly.empty else None)

    meta = {
        "accounts": acct_opts, "classes": class_opts, "brokers": broker_opts,
        "available_dates": list(frames.available_dates),
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "filter": {"account": "all", "asset_class": "all", "broker": "all"},
        "factor_file_through": factor_through,
    }

    if twr is None or twr.empty:
        state = {"available": False, "unavailable": "no_twr",
                 "unavailable_message": ("No portfolio TWR series yet — ingest "
                                         "statements first."),
                 **_control_vocab()}
        return {"meta": meta, "caption": _CAPTION, "state": state,
                "by_window": {}, "methodology": _METHODOLOGY}
    if ff_monthly.empty:
        state = {"available": False, "unavailable": "no_factors",
                 "unavailable_message": ("Factor data not fetched yet. Run "
                                         "`py parsers/fetch_ff_factors.py "
                                         "--write` (Ken French Data Library — "
                                         "free, keyless)."),
                 **_control_vocab()}
        return {"meta": meta, "caption": _CAPTION, "state": state,
                "by_window": {}, "methodology": _METHODOLOGY}

    by_window = {w: _window_block(w, twr, ff_monthly, ff_daily, weights,
                                  daily_prices, factor_through)
                 for w in WINDOWS}
    state = {"available": True, "unavailable": None, "unavailable_message": None,
             **_control_vocab()}
    return {"meta": meta, "caption": _CAPTION, "state": state,
            "by_window": by_window, "methodology": _METHODOLOGY}
