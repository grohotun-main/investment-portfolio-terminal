# terminal/riskcontrib_service.py
"""Pure data seam for the MERIDIAN Terminal "Risk Contribution" tab, Slice 1.

Re-expresses the core decomposition of app.py._render_riskcontrib_body
(5515-5936 + 7001-7287): the per-position vol / downside / ES risk contributions,
the portfolio risk panel (with benchmark compare), and the top-contributors block.
The risk bundle is reused from risk_service (already parity-covered); the
decomposition engines are importable from risk_metrics / treasury_proxy.
Precompute-all: every estimator×ES×threshold block + per-symbol benchmark compare
ship in one body; the front-end swaps controls client-side with zero refetch.
Slices 2 (DR-in-context) and 3 (correlations) follow.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

from terminal import holdings_service as hs
from terminal import risk_service as rs
from terminal.performance_service import _resolve_filter

from risk_metrics import (
    compute_risk_contributions,
    compute_downside_risk_contributions,
    compute_es_contributions,
    compute_var_cvar,
    series_vol_ann,
)
from treasury_proxy import treasury_proxy_breakdown

from terminal.riskcontrib_dr import build_dr_in_context
from terminal.riskcontrib_regime import build_dr_regime, _load_vix, _load_long_history
from terminal.riskcontrib_corr import build_major_correlations

RC_WINDOW = 252

ESTIMATORS = [
    {"id": "ewma_lw", "label": "EWMA + Ledoit-Wolf (default)"},
    {"id": "ewma",    "label": "EWMA (no shrinkage)"},
    {"id": "rolling", "label": "Rolling 252d (legacy)"},
]
ES_LEVELS = [  # id is the str(alpha); label mirrors app.py 5626
    {"id": "0.05",  "alpha": 0.05,  "label": "95% (α=5%)"},
    {"id": "0.025", "alpha": 0.025, "label": "97.5% (α=2.5%)"},
    {"id": "0.01",  "alpha": 0.01,  "label": "99% (α=1%)"},
]
THRESHOLDS = [  # id is the str(threshold); label mirrors app.py 5644
    {"id": "0.0",    "thr": 0.0,    "label": "0% (Sortino MAR)"},
    {"id": "-0.005", "thr": -0.005, "label": "−0.5% (mild stress)"},
    {"id": "-0.01",  "thr": -0.01,  "label": "−1% (sharp stress)"},
]

_INFO_HTML = (  # app.py 5517-5524, bold preserved
    "<b>This page is a regime monitor, not a crash forecaster.</b> It tells you "
    "which positions drive risk <i>in the current regime</i> — based on the "
    "trailing 252 trading days. Expected Shortfall values here are "
    "<i>average-bad-day</i> losses for this window, not tail-event predictions.")
_CAPTION_HTML = (  # app.py 5525-5534
    "Breaks down portfolio risk by position, asking three different questions: "
    "which positions contribute most to (1) <b>overall volatility</b>, "
    "(2) <b>losses on bad days</b>, and (3) <b>losses in the worst tail</b>. "
    "Uses the same weight universe as the Risk Overview tab — TLH sleeve folds "
    "to SPY, treasury rungs map to duration-matched Treasury ETFs (uncovered → "
    "SGOV; see the per-rung mix below), cash and options excluded. Inherits the "
    "Account / Asset-class filter.")

_EST_DISPLAY = {"ewma_lw": "EWMA + Ledoit-Wolf", "ewma": "EWMA",
                "rolling": "Rolling 252d (sample covariance)"}


def _jnum(v):
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _fin(v) -> bool:
    return v is not None and math.isfinite(float(v))


def _treasury_ladder(latest_snap: pd.DataFrame) -> dict:
    """app.py 5667-5685."""
    if latest_snap is None or latest_snap.empty:
        return {"present": False, "caption": None}
    lad = latest_snap[latest_snap["bucket"] == "Treasury Ladder"]
    if lad.empty:
        return {"present": False, "caption": None}
    as_of = pd.Timestamp(lad["statement_date"].max())
    counts, unparsed = treasury_proxy_breakdown(lad["description"], as_of)
    mix = ", ".join(f"{sym}×{n}" for sym, n in counts.most_common())
    cap = f"Treasury ladder ({len(lad)} rungs) → duration proxies: {mix}."
    if unparsed:
        cap += (f" ⚠️ {unparsed} rung(s) exposed no parseable maturity and "
                f"defaulted to SGOV — duration understated; check the statement format.")
    return {"present": True, "caption": cap}


def _controls_static() -> dict:
    """Static control skeleton (no per-option captions — used for unavailable states
    where we don't have a bundle to compute stress-day counts)."""
    return {
        "estimators": [{"id": e["id"], "label": e["label"]} for e in ESTIMATORS],
        "es_levels": [{"id": e["id"], "label": e["label"], "caption": ""} for e in ES_LEVELS],
        "thresholds": [{"id": t["id"], "label": t["label"], "caption": ""} for t in THRESHOLDS],
        "benchmarks": [{"id": "SPY", "label": "SPY (S&P 500 TR)"}],
    }


def _build_controls(b, vol_blocks) -> dict:
    """Build the full controls dict with per-option captions (app.py 5636-5665)."""
    port_tail = b["port_rets"].tail(252) if not b["port_rets"].empty else pd.Series(dtype=float)
    es_caps = []
    for a in ES_LEVELS:
        n_tail = int(np.round(252 * a["alpha"]))
        es_caps.append({"id": a["id"], "label": a["label"],
                        "caption": (f"Looks at the worst {a['alpha'] * 100:.1f}% of the "
                                    f"trailing 252 trading days — about {n_tail} tail-day(s) "
                                    f"— and averages the losses there.")})
    thr_caps = []
    for t in THRESHOLDS:
        n_stress = int((port_tail <= t["thr"]).sum())
        thr_caps.append({"id": t["id"], "label": t["label"],
                         "caption": (f"Days where the portfolio returned less than "
                                     f"{t['thr'] * 100:+.1f}% count as 'stress days'. The "
                                     f"trailing-252-day window has {n_stress} stress day(s) "
                                     f"out of {len(port_tail)}.")})
    return {
        "estimators": [{"id": e["id"], "label": e["label"]} for e in ESTIMATORS],
        "es_levels": es_caps,
        "thresholds": thr_caps,
        "benchmarks": [{"id": "SPY", "label": "SPY (S&P 500 TR)"}],
    }


# --------------------------------------------------------------------------- #
# Combo assembly helpers.
# --------------------------------------------------------------------------- #

def _delta_cls(v) -> str:
    """app.py _color_delta_pp 7208-7222 → a CSS class name."""
    if v is None or not math.isfinite(float(v)):
        return ""
    v = float(v)
    if v >= 5.0:  return "loss-strong"
    if v >= 2.0:  return "loss-mild"
    if v <= -5.0: return "gain-strong"
    if v <= -2.0: return "gain-mild"
    return ""


_MIN_VOL_OBS_ABS = 60
_MIN_VOL_OBS_FRAC = 0.5


def _table_rows(rc, rd, re_) -> list:
    """app.py 7119-7206: join downside+ES onto the vol per_symbol, apply the
    thin-history gates, head(15), format every cell with cls/bold hints."""
    per = rc["per_symbol"]
    n_days_window = max(int(rc.get("n_days") or 0), 1)
    min_window_obs = int(_MIN_VOL_OBS_FRAC * n_days_window)
    j = per[["weight_pct", "standalone_vol_ann", "pctr_pct", "diff_pp"]].copy()
    for col in ("n_obs_with_price", "n_obs_in_window"):
        j[col] = per[col] if col in per.columns else np.nan
    thin = (j["n_obs_with_price"] < _MIN_VOL_OBS_ABS) | (j["n_obs_in_window"] < min_window_obs)
    j.loc[thin, "standalone_vol_ann"] = np.nan

    if not rd["per_symbol_down"].empty:
        j = j.join(rd["per_symbol_down"][["pctr_pct_down"]], how="left")
    else:
        j["pctr_pct_down"] = np.nan

    if not re_["per_symbol_es"].empty:
        es_n_days = max(int(re_.get("n_days_window") or 0), 1)
        es_min_obs = int(_MIN_VOL_OBS_FRAC * es_n_days)
        ecols = ["pctr_es_pct"] + (["n_obs_in_window"]
                                   if "n_obs_in_window" in re_["per_symbol_es"].columns else [])
        ep = re_["per_symbol_es"][ecols].copy()
        if "n_obs_in_window" in ep.columns:
            thin_es = ((ep["n_obs_in_window"] < _MIN_VOL_OBS_ABS)
                       | (ep["n_obs_in_window"] < es_min_obs))
            ep.loc[thin_es, "pctr_es_pct"] = np.nan
        j = j.join(ep[["pctr_es_pct"]], how="left")
    else:
        j["pctr_es_pct"] = np.nan

    j["delta_down_pp"] = j["pctr_pct_down"] - j["pctr_pct"]
    top = j.head(15)
    rows = []
    for sym, r in top.iterrows():
        pctr = float(r["pctr_pct"])
        down = r["pctr_pct_down"]; es = r["pctr_es_pct"]
        dd = r["delta_down_pp"]; esd = (float(es) - pctr) if pd.notna(es) else np.nan
        sv = r["standalone_vol_ann"]
        rows.append({
            "symbol": str(sym),
            "weight": f"{float(r['weight_pct']):.2f}%",
            "pctr": f"{pctr:.2f}%",
            "risk_delta": {"text": f"{float(r['diff_pp']):+.2f}", "cls": _delta_cls(r["diff_pp"])},
            "downside_pctr": (f"{float(down):.2f}%" if pd.notna(down) else "—"),
            "delta_down": {"text": (f"{float(dd):+.2f}" if pd.notna(dd) else "—"),
                           "cls": _delta_cls(float(dd) if pd.notna(dd) else None)},
            "es_pctr": {"text": (f"{float(es):.2f}%" if pd.notna(es) else "—"),
                        "cls": _delta_cls(float(es) if pd.notna(es) else None),
                        "bold": bool(pd.notna(es) and (float(es) - pctr) >= 5.0)},
            "es_delta": {"text": (f"{float(esd):+.2f}" if pd.notna(esd) else "—"),
                         "cls": _delta_cls(float(esd) if pd.notna(esd) else None)},
            "standalone_vol": (f"{float(sv) * 100.0:.1f}%" if pd.notna(sv) else "—"),
        })
    return rows


def _estimator_strip(rc) -> str:
    """app.py 5717-5732."""
    s = f"<b>Covariance estimator:</b> {_EST_DISPLAY.get(rc['estimator'], rc['estimator'])}"
    if rc.get("lambda") is not None:
        s += f" · <b>decay λ</b> = {rc['lambda']:.2f} (higher = slower decay, more memory)"
    if rc.get("alpha") is not None:
        s += (f" · <b>shrinkage intensity α</b> = {rc['alpha']:.2f} "
              f"(0 = no shrinkage, 1 = full pull toward the target)")
    s += f" · <b>input</b> = {rc['n_days']} trading day(s)"
    return s


def _window_caption(rc, rd, re_, cov_estimator, thr_label) -> str:
    """app.py 7264-7286 (two branches)."""
    thr_short = thr_label.split(" ")[0]
    if (not rd["per_symbol_down"].empty) and _fin(rd.get("dr_down")):
        return (
            f"<i>Windows: <b>{rc['n_days']}</b> trading days for PCTR / Downside "
            f"PCTR ({cov_estimator}); <b>{re_['n_days_window']}</b> trading days for "
            f"ES PCTR (sample-historical, no estimator).  Universe: "
            f"<b>{rc['n_symbols']}</b> symbols · PCTR sums to 100% by construction.  "
            f"<b>Downside:</b> {rd['n_down_days']} stress day(s) ≤ {thr_short} — "
            f"downside volatility {rd['port_vol_ann_down'] * 100:.2f}% (annualized), "
            f"downside Diversification Ratio {rd['dr_down']:.2f}×.  "
            f"<b>Expected Shortfall:</b> {re_['n_tail_days']} tail day(s) at "
            f"{re_['alpha'] * 100:.1f}% confidence.</i>")
    return (
        f"<i>Windows: <b>{rc['n_days']}</b> trading days for PCTR ({cov_estimator}); "
        f"<b>{re_['n_days_window']}</b> trading days for ES PCTR (sample-historical).  "
        f"Universe: <b>{rc['n_symbols']}</b> symbols · PCTR sums to 100% by "
        f"construction.</i>")


_ROW_CUE_HTML = (  # app.py 7254-7262
    "<b>Row cues</b> — 🔴 orange = risk driver / stress amplifier (positive Δ ≥ 5pp "
    "= strong, ≥ 2pp = mild) · 🟢 green = diversifier (negative Δ ≤ −5pp = strong, "
    "≤ −2pp = mild). <b>Bold ES PCTR</b> = hidden tail risk (ES contribution ≥ 5pp "
    "above total PCTR).")


def _pairbar_rows(per: pd.DataFrame) -> pd.DataFrame:
    """Rows for the Weight-vs-PCTR pairbar: top-10 by PCTR ∪ top-10 by dollar
    weight, kept in ``per``'s PCTR-descending order. The union keeps big-but-
    quiet holdings (the T-bill sleeve: near-zero PCTR, large weight) on the
    chart — the weight/risk gap IS the chart's story (TK 2026-07-19)."""
    keep = set(per.head(10).index) | set(
        per.sort_values("weight_pct", ascending=False).head(10).index)
    return per.loc[[s for s in per.index if s in keep]]


def _combo_block(rc, rd, re_, *, cov_estimator, alpha, alpha_label, thr_label) -> dict:
    per = rc["per_symbol"]
    # estimator strip + sample warnings
    warns = []
    if rd["n_down_days"] < 20:
        warns.append({"kind": "downside", "message": (
            f"Downside covariance: only {rd['n_down_days']} day(s) below "
            f"{thr_label.split(' ')[0]} on the {rd['n_days_window']}-day window — "
            f"need at least 20 for a stable estimate. Downside columns will show '—'.")})
    if re_["n_tail_days"] < 5:
        warns.append({"kind": "es", "message": (
            f"Expected Shortfall: only {re_['n_tail_days']} tail day(s) at "
            f"{alpha_label.split(' ')[0]} confidence — too thin to trust. ES columns "
            f"will show '—'.")})
    # portfolio tiles
    port_es = re_["port_es"]
    es_ok = (port_es is not None and np.isfinite(port_es) and port_es > 0)
    es_cap = ("—" if not (re_["var_p"] is not None and np.isfinite(re_["var_p"])
                          and re_["n_tail_days"] > 0)
              else (f"Average loss on the {re_['n_tail_days']} worst day(s) in the "
                    f"window. Value at Risk (VaR) at the same confidence: "
                    f"{re_['var_p'] * 100:.2f}%."))
    portfolio = {
        "vol": {"value": f"{rc['port_vol_ann'] * 100:.2f}%",
                "raw": _jnum(rc["port_vol_ann"]),
                "caption": f"How wide swings are on average. Based on {rc['n_days']}-day covariance."},
        "es": {"value": (f"{port_es * 100:.2f}%" if es_ok else "—"),
               "raw": _jnum(port_es if es_ok else None), "caption": es_cap},
    }
    # top-contributor tiles (app.py 7012-7033)
    top_sym = str(per.index[0])
    top_pctr = float(per["pctr_pct"].iloc[0]); top_w = float(per["weight_pct"].iloc[0])
    top3 = float(per["pctr_pct"].head(3).sum())
    top5 = float(per["pctr_pct"].head(5).sum()); top5w = float(per["weight_pct"].head(5).sum())
    dr = rc["dr"]
    top_tiles = [
        {"label": "Top risk contributor", "value": top_sym,
         "sub": f"{top_pctr:.1f}% of risk · {top_w:.1f}% of weight"},
        {"label": "Top 3 — risk share", "value": f"{top3:.1f}%",
         "sub": "Share of total portfolio vol in top 3 names"},
        {"label": "Top 5 — risk / weight", "value": f"{top5:.1f}% / {top5w:.1f}%",
         "sub": "PCTR / dollar weight — gap = how much of the risk budget lives in "
                "fewer names than the $ budget"},
        {"label": "Diversification ratio",
         "value": (f"{dr:.2f}×" if np.isfinite(dr) else "—"),
         "sub": f"{rc['weighted_avg_vol_ann'] * 100:.1f}% weighted-avg standalone vol → "
                f"{rc['port_vol_ann'] * 100:.1f}% port vol · estimator: {cov_estimator}"},
    ]
    pairbar = _pairbar_rows(per)
    return {
        "estimator_strip": _estimator_strip(rc),
        "sample_warnings": warns,
        "portfolio": portfolio,
        "top_tiles": top_tiles,
        "weight_vs_pctr": {
            "symbols": [str(s) for s in pairbar.index],
            "weight": [_jnum(v) for v in pairbar["weight_pct"].values],
            "pctr": [_jnum(v) for v in pairbar["pctr_pct"].values]},
        "table": {"rows": _table_rows(rc, rd, re_),
                  "row_cue_caption": _ROW_CUE_HTML,
                  "window_caption": _window_caption(rc, rd, re_, cov_estimator, thr_label)},
    }


# --------------------------------------------------------------------------- #
# Benchmarks.
# --------------------------------------------------------------------------- #

def _cmp(port, bench) -> dict:
    """_bench_line_html 5862-5895: LOWER_BETTER, neutral 0.10 pp. value/delta in %."""
    if not (_fin(port) and _fin(bench)):
        return {"value": None, "delta": None, "dir": "flat"}
    d = (float(port) - float(bench)) * 100.0
    if abs(d) < 0.10:
        direction = "flat"
    else:
        direction = "up" if d < 0 else "down"   # lower is better
    return {"value": f"{float(bench) * 100:.2f}%", "delta": f"{d:+.2f} pp", "dir": direction}


def _benchmarks(b, frames, vol_blocks, es_blocks) -> tuple[dict, list]:
    """Return ({sym: {label, vol:{est:cmp}, es:{est:{alpha:cmp}}}}, controls_benchmark_list)."""
    daily = frames.daily_prices
    spy = b["spy_rets"]
    series = {"SPY": ("SPY (S&P 500 TR)", spy)}
    has_tlt = "TLT" in daily.columns
    if has_tlt:
        spy_d = spy if not spy.empty else daily["SPY"].pct_change().dropna()
        tlt_d = daily["TLT"].pct_change().dropna()
        aligned = pd.concat([spy_d, tlt_d], axis=1, keys=["SPY", "TLT"], sort=True).dropna()
        series["60_40"] = ("60/40 SPY/TLT blend", 0.6 * aligned["SPY"] + 0.4 * aligned["TLT"])
    for sym in sorted(daily.columns):
        rets = daily[sym].pct_change().dropna()
        if rets.empty:
            # Skip a symbol with no usable price history — e.g. a phantom
            # all-NaN column injected by a TICKER_HISTORY rename whose symbols
            # aren't present in this data dir. config_local.py (which holds
            # TICKER_HISTORY) is gitignored, so without this the fixture's
            # benchmark universe would differ from CI. A priceless benchmark is
            # a dead dropdown entry anyway (its comparison line never renders).
            continue
        series.setdefault(str(sym), (str(sym), rets))

    out = {}
    ctrl = []
    for sid, (label, rets) in series.items():
        ctrl.append({"id": sid, "label": label})
        vol = {}
        for e in ESTIMATORS:
            rc = vol_blocks[e["id"]]
            n_days = int(rc.get("n_days") or 0)
            if rets.empty or n_days < 20:
                bench_vol = float("nan")
            else:
                lam = rc["lambda"] if rc.get("lambda") is not None else 0.94
                bench_vol = series_vol_ann(rets, estimator=e["id"], window=RC_WINDOW,
                                           lambda_param=lam)
            vol[e["id"]] = _cmp(rc["port_vol_ann"], bench_vol)
        # Bench ES is keyed [estimator][alpha]: the tail length is the SELECTED
        # estimator's rc["n_days"] (app.py:5844 `bench_rets.tail(rc["n_days"])`),
        # and n_days differs by estimator on real data (rolling caps at 252, EWMA
        # at 504). So bench ES = f(symbol, estimator, alpha) for 1:1 parity.
        es = {}
        for e in ESTIMATORS:
            rc = vol_blocks[e["id"]]
            n_days = int(rc.get("n_days") or 0)
            es_e = {}
            for a in ES_LEVELS:
                re_ = es_blocks[a["id"]]
                if rets.empty or n_days < 20:
                    bench_es = float("nan")
                else:
                    _var_b, _cvar_b = compute_var_cvar(rets.tail(n_days), alpha=a["alpha"])
                    bench_es = -_cvar_b if np.isfinite(_cvar_b) else float("nan")
                port_es = re_["port_es"]
                es_e[a["id"]] = _cmp(port_es if (port_es is not None and np.isfinite(port_es)) else None,
                                     bench_es)
            es[e["id"]] = es_e
        out[sid] = {"label": label, "vol": vol, "es": es}
    return out, ctrl


# --------------------------------------------------------------------------- #
# View assembly.
# --------------------------------------------------------------------------- #

def build_riskcontrib_view(frames, *, account: str | list[str] = "all",
                           asset_class: str | list[str] = "all") -> dict:
    """Assemble the GET /api/riskcontrib contract. Pure given frames + selections."""
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
    # Filter-active disclosure (app.py 5551-5563). Set before any guard so all
    # states (including unavailable) carry the filter_note.
    filter_note = None
    if account_active or class_active:
        bits = [x for x, on in (("Account", account_active), ("Asset-class", class_active)) if on]
        filter_note = (
            f"{' + '.join(bits)} filter active. Risk contributions reflect the "
            f"filtered subset only — positions in excluded buckets are not part of "
            f"this decomposition, so the displayed vol / DR / ES are not the "
            f"full-portfolio numbers.")

    base = {"meta": meta, "info_html": _INFO_HTML, "caption_html": _CAPTION_HTML,
            "filter_note": filter_note, "treasury_ladder": {"present": False, "caption": None},
            "controls": _controls_static(),
            "combos": {}, "benchmarks": {}, "dr_in_context": None, "dr_regime": None,
            "correlations": None}

    def _state(unavail, msg):
        return {**base, "state": {"available": False, "unavailable": unavail,
                                  "unavailable_message": msg}}

    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return _state("no_twr", "Need twr_portfolio.csv for risk metrics.")
    if frames.daily_prices.empty:
        return _state("daily_empty",
                      "Daily prices required — risk decomposition uses the covariance "
                      "matrix on a 252d trailing window. Run "
                      "`py parsers/fetch_daily_prices.py --write`.")

    b = rs._bundle(frames, bucket_filter, class_filter, account_active, class_active)
    if b["weights"].empty:
        return _state("weights_empty", "Filtered universe has no positions to decompose.")
    base["treasury_ladder"] = _treasury_ladder(b["latest_snap"])

    # Default-estimator guard (app.py 5691-5697). Estimators read the same daily
    # history (rolling sees 252d, EWMA up to 504d), so they are thin-together —
    # drive the top-level state off the default ewma_lw rc.
    rc0 = compute_risk_contributions(b["weights"], frames.daily_prices,
                                     window=RC_WINDOW, estimator="ewma_lw")
    if rc0["per_symbol"].empty:
        return _state("decomp_thin",
                      f"Need ≥20 trading days of daily history for a meaningful "
                      f"decomposition; have {rc0['n_days']} on the filtered universe.")
    if rc0["port_vol_ann"] <= 0:
        return _state("port_vol_zero",
                      "Portfolio vol on this window is zero — nothing to decompose.")

    daily = frames.daily_prices
    w = b["weights"]
    vol_blocks = {e["id"]: (rc0 if e["id"] == "ewma_lw" else
                            compute_risk_contributions(w, daily, window=RC_WINDOW,
                                                       estimator=e["id"]))
                  for e in ESTIMATORS}
    down_blocks = {(e["id"], t["id"]): compute_downside_risk_contributions(
                       w, daily, threshold=t["thr"], window=RC_WINDOW, estimator=e["id"])
                   for e in ESTIMATORS for t in THRESHOLDS}
    es_blocks = {a["id"]: compute_es_contributions(w, daily, alpha=a["alpha"], window=RC_WINDOW)
                 for a in ES_LEVELS}

    combos = {}
    for e in ESTIMATORS:
        for a in ES_LEVELS:
            for t in THRESHOLDS:
                combos[f"{e['id']}|{a['id']}|{t['id']}"] = _combo_block(
                    vol_blocks[e["id"]], down_blocks[(e["id"], t["id"])], es_blocks[a["id"]],
                    cov_estimator=e["id"], alpha=a["alpha"],
                    alpha_label=a["label"], thr_label=t["label"])

    benchmarks, bench_ctrl = _benchmarks(b, frames, vol_blocks, es_blocks)
    controls = _build_controls(b, vol_blocks)
    controls["benchmarks"] = bench_ctrl

    dr_in_context = build_dr_in_context(b["weights"], daily, b["port_rets"])
    vix = _load_vix(frames.data_dir)
    long_history = _load_long_history(frames.data_dir)
    dr_regime = build_dr_regime(b["weights"], daily, b["port_rets"], vix, long_history)
    correlations = build_major_correlations(vol_blocks, daily, long_history, b["port_rets"])
    return {**base, "controls": controls, "combos": combos, "benchmarks": benchmarks,
            "dr_in_context": dr_in_context, "dr_regime": dr_regime,
            "correlations": correlations,
            "state": {"available": True, "unavailable": None, "unavailable_message": None}}
