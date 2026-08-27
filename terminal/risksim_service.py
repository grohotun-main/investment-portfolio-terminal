# terminal/risksim_service.py
"""Pure data seam for the MERIDIAN Terminal "Risk Simulation" tab (Slices 1-4).

Ports the full app.py._render_whatif_body (7304-7975): the reweight -> simulate
loop, the optimizer box, the cap-curve trace, and the not-held candidate-ticker
path. The math is importable: risk_service._bundle re-derives the same
`risk_bundle` the Streamlit body consumes (current weights + bench_tr);
parsers.whatif_engine.compute_before_after computes before/after risk;
parsers.min_variance / parsers.risk_parity provide the Suggest optimizers;
parsers.opt_curve.trace_cap_curve sweeps the cap ladder. This module constructs
the WhatIfScenario (with an optional not-held candidate ticker), builds the
optimizer inputs, and shapes the engine output to JSON. Numbers match Streamlit
1:1 by construction.

Entry points:
  build_risksim_view(frames, *, account, asset_class) -> the GET /api/risksim
    seed (meta + guards + the current-weights grid + the optimizer block).
  run_simulation(frames, new_weights, *, account, asset_class, candidates,
    bundle) -> the POST /api/risksim/simulate result (coverage + weight bars +
    before/after headline + detail panels), or an {error} when the scenario
    can't run. Up to MAX_CANDIDATES not-held tickers (each with grid weight)
    are fetched and modeled -> one MCR verdict per candidate.
  run_optimize(frames, *, optimizer, cap_pct, floors, caps, account, asset_class) ->
    the POST /api/risksim/optimize result ({kind, message, new_pct}) — min-
    variance / risk-parity Suggest weights for the grid.
  run_trace(frames, *, cap_pct, floors, caps, account, asset_class) -> the POST
    /api/risksim/trace result (vol vs Effective N per cap across both optimizers).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from terminal import holdings_service as hs
from terminal.holdings_service import Frames
from terminal.performance_service import _resolve_filter
from terminal.risk_service import _bundle, _load_rf, _overlay
from terminal.riskcontrib_corr import _corr_heatmap

from min_variance import (anchored_defaults, suggest_min_variance_grid,
                          to_floor_bucket)
from risk_parity import suggest_risk_parity_grid
from whatif_engine import WhatIfCandidate, WhatIfScenario, compute_before_after
from opt_curve import trace_cap_curve
from whatif_data import (fetch_candidate_history, splice_with_proxy,
                         build_multi_augmented_price_matrix)
from frontier import LAMBDA_LADDER, capm_expected_returns, trace_frontier
from terminal import factor_service as _fs
from terminal import risk_service as _rsvc

# ---- Frontier result memo (for the tab-scoped AI summary box) -------------- #
# The frontier is an ~80s optimizer sweep; the AI reducer must not re-run it.
# The POST handler memoizes each result keyed by frontier_sig(...) and the AI
# reducer reads it. Per-process (single-worker uvicorn), like the AI _JOBS map;
# a restart empties it (a cold miss -> the box shows "re-run to summarize").
_FRONTIER_MEMO: dict = {}
_FRONTIER_MEMO_MAX = 6
_FRONTIER_MEMO_LOCK = threading.Lock()


def frontier_sig(*, data_version, broker, history_start, account,
                 asset_class, cap_pct, floors, caps, erp_pct) -> str:
    """Reproducible signature of a frontier's full identity (hashlib over
    canonical JSON, NOT the salted builtin hash). The POST handler computes it;
    the FE echoes it back on the AI request; tests recompute it."""
    obj = {"dv": data_version,
           "broker": sorted(broker or ["all"]),
           "history_start": history_start or "all",
           "account": sorted(account or ["all"]),
           "asset_class": sorted(asset_class or ["all"]),
           "cap_pct": round(float(cap_pct), 6),
           "floors": {str(k): round(float(v), 6) for k, v in (floors or {}).items()},
           "caps": {str(k): round(float(v), 6) for k, v in (caps or {}).items()},
           "erp_pct": round(float(erp_pct), 6)}
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def frontier_memo_put(sig, payload, account, asset_class) -> None:
    with _FRONTIER_MEMO_LOCK:
        _FRONTIER_MEMO[sig] = {"payload": payload, "account": account,
                               "asset_class": asset_class}
        while len(_FRONTIER_MEMO) > _FRONTIER_MEMO_MAX:
            _FRONTIER_MEMO.pop(next(iter(_FRONTIER_MEMO)))


def frontier_memo_get(sig):
    if not sig:
        return None
    with _FRONTIER_MEMO_LOCK:
        return _FRONTIER_MEMO.get(sig)


# ---- Simulation result memo (for the tab-scoped AI summary box) ----------- #
# The what-if reweight's numeric before/after facts, memoized by simulate_sig
# so the AI reducer narrates the exact run without re-simulating. Per-process
# (single-worker uvicorn), like _FRONTIER_MEMO; a restart empties it (cold
# miss -> the box shows the stale line). Mirrors the frontier memo trio.
_SIMULATE_MEMO: dict = {}
_SIMULATE_MEMO_MAX = 6
_SIMULATE_MEMO_LOCK = threading.Lock()


def simulate_sig(*, data_version, broker, history_start, account, asset_class,
                 weights, candidates=None) -> str:
    """Reproducible signature of a simulation's full identity (canonical-JSON
    sha1, like frontier_sig). Any input that changes the result changes the sig
    -> a fresh narration; an identical re-run hits the memo (no re-narrate).
    ``candidates`` is the list form (``{"ticker", "proxy"}`` dicts); hashed as a
    sorted (order-insensitive) list so candidate order in the request never
    matters, only the set."""
    cand = [
        {"ticker": str(c.get("ticker") or "").strip().upper(),
         "proxy": str(c.get("proxy") or "").strip().upper()}
        for c in (candidates or []) if str(c.get("ticker") or "").strip()]
    obj = {"dv": data_version,
           "broker": sorted(broker or ["all"]),
           "history_start": history_start or "all",
           "account": sorted(account or ["all"]),
           "asset_class": sorted(asset_class or ["all"]),
           "weights": {str(k): round(float(v), 6)
                       for k, v in sorted((weights or {}).items())},
           "candidates": sorted(cand, key=lambda d: (d["ticker"], d["proxy"]))}
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def simulate_memo_put(sig, facts, account, asset_class) -> None:
    with _SIMULATE_MEMO_LOCK:
        _SIMULATE_MEMO[sig] = {"facts": facts, "account": account,
                               "asset_class": asset_class}
        while len(_SIMULATE_MEMO) > _SIMULATE_MEMO_MAX:
            _SIMULATE_MEMO.pop(next(iter(_SIMULATE_MEMO)))


def simulate_memo_get(sig):
    if not sig:
        return None
    with _SIMULATE_MEMO_LOCK:
        return _SIMULATE_MEMO.get(sig)


_CAPTION_HTML = (
    "Reweight your portfolio and see how the risk changes. Edit any "
    "<b>New %</b> cell up or down, or set a holding to 0 to drop it — then Run "
    "to compare current vs simulated on the same EWMA + Ledoit-Wolf covariance "
    "engine as the Risk Contribution tab. The optimizer console below suggests "
    "constrained allocations and sweeps the cap and return trade-offs. "
    "Inherits the Account / Asset-class filters.")
_GRID_CAPTION_HTML = (
    "Edit any <b>New %</b> cell to move weight between holdings, or set one to 0 "
    "to drop it. Simulated weights must total 100% before you can Run.")
_OPT_CAPTION_HTML = (
    "Two one-click weightings — covariance only, no expected-return guesses. "
    "Min-variance: the lowest-volatility mix subject to the per-name cap and "
    "per-class floors. Risk-parity: equal risk contribution from every holding "
    "— inherently diversifying, subject to the same per-name cap. Fills the grid "
    "below; review and Run. In-sample optimization on the trailing-window "
    "covariance — no transaction costs, and the estimates carry sampling error, "
    "so treat as direction, not precision. Holdings too young to model are "
    "carried at their current weight and excluded from the vol estimate.")

# The fixed cap ladder (percent) swept by the Trace, unioned with the live cap
# (app.py 7450-7452). 13 caps × 2 optimizers = ≤ 28 quick solves over one Σ build.
_TRACE_LADDER_PCT = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
                     60.0, 75.0, 100.0]
_TRACE_SERIES = [("min_variance", "Min-variance"), ("risk_parity", "Risk-parity")]
_TRACE_CAPTION_BASE = (
    "Each marker is one optimizer run at one per-name cap; the line runs from the "
    "tightest cap to the loosest. Σ-only estimates — same numbers as the "
    "Suggest banners; fill the grid and Run for the full before/after. Min-variance "
    "traced at floors: {floors}; risk-parity has no floors. Young holdings held at "
    "current weight.{class_caps}{risk_budgets}{skipped}")

_FRONTIER_BETA_YEARS = 5
_FRONTIER_WINDOW_DAYS = 1260
_FRONTIER_CAPTION_BASE = (
    "Each marker is one constrained solve at a different risk aversion; the "
    "line runs from the minimum-variance end to the maximum-return end, under "
    "your live constraints (per-name cap {cap:.0f}%, floors: {floors})."
    "{class_caps} Expected returns are CAPM-implied <b>estimates, not "
    "forecasts</b>: E[r] = rf + β × ERP with rf = {rf:.1f}% ({rf_src}), "
    "ERP = {erp:.1f}%, β fitted on {years} years of daily returns vs the "
    "Fama-French market factor.{assumed} Young holdings held at current "
    "weight. ★ = current book.{erc_note}{skipped}")


def _resolve(frames: Frames, account: str | list[str], asset_class: str | list[str]):
    """Filter resolution shared by seed + run (mirrors risk_service.build_risk_view)."""
    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    (bucket_filter, class_filter, _ids,
     account_active, class_active) = _resolve_filter(frames, snap_all, account, asset_class)
    return (acct_opts, class_opts, bucket_filter, class_filter,
            account_active, class_active)


def _bundle_for(frames: Frames, account: str | list[str], asset_class: str | list[str]) -> dict:
    """The risk bundle for the requested filter (current weights + bench_tr)."""
    (_a, _c, bucket_filter, class_filter,
     account_active, class_active) = _resolve(frames, account, asset_class)
    return _bundle(frames, bucket_filter, class_filter, account_active, class_active)


def _meta(frames: Frames, account: str | list[str], asset_class: str | list[str]) -> dict:
    acct_opts, class_opts, *_ = _resolve(frames, account, asset_class)
    broker_opts, _ = hs._broker_options(hs._current_snap(frames))
    return {"accounts": acct_opts, "classes": class_opts, "brokers": broker_opts,
            "filter": hs._filter_meta(account, asset_class),
            "synthetic": "synth" in str(frames.data_dir).lower()}


def _mv_class_of(bundle: dict) -> dict:
    """Floor-bucket per held symbol, exactly as app.py 7381-7384: a symbol absent
    from the latest snapshot defaults to 'fixed_income' before to_floor_bucket."""
    snap_cls = (bundle["latest_snap"].groupby("symbol")["asset_class"]
                .first().to_dict())
    return {str(s): to_floor_bucket(snap_cls.get(s, "fixed_income"))
            for s in bundle["weights"].index}


def _optimizer_seed(bundle: dict) -> dict:
    """GET-seed optimizer block: caption + anchored cap/floor defaults + the
    sorted non-'other' floor buckets (app.py 7380-7398)."""
    class_of = _mv_class_of(bundle)
    defs = anchored_defaults(bundle["weights"], class_of)
    buckets = sorted({b for b in class_of.values() if b != "other"})
    return {
        "caption_html": _OPT_CAPTION_HTML,
        "cap_default_pct": float(defs["cap_default"]) * 100.0,
        "buckets": [
            {"key": b, "label": b.replace("_", " ").title(),
             "floor_default_pct": (float(defs["equity_floor_default"]) * 100.0
                                   if b == "equity" else 0.0)}
            for b in buckets
        ],
    }


def build_risksim_view(frames: Frames, *, account: str | list[str] = "all",
                       asset_class: str | list[str] = "all",
                       bundle: dict | None = None) -> dict:
    """GET /api/risksim contract: meta + the three load guards + the seed grid.
    ``bundle``: optional precomputed ``_bundle_for(frames, account, asset_class)``
    for the SAME filters (the request-scoped reuse seam); None computes it here."""
    meta = _meta(frames, account, asset_class)
    base = {"meta": meta, "caption_html": _CAPTION_HTML}

    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {**base, "state": _unavail("Need twr_portfolio.csv for risk metrics."),
                "grid": None, "optimizer": None}
    if bundle is None:
        bundle = _bundle_for(frames, account, asset_class)
    if bundle["weights"].empty:
        return {**base, "state": _unavail(
            "Filtered universe has no positions to model against."),
                "grid": None, "optimizer": None}
    if frames.daily_prices.empty:
        return {**base, "state": _unavail(
            "Daily prices required for Risk Simulation math. Run "
            "`py parsers/fetch_daily_prices.py --write`."),
                "grid": None, "optimizer": None}

    rows = [{"ticker": str(t), "now_pct": float(w) * 100.0}
            for t, w in bundle["weights"].items()]
    return {**base, "state": {"available": True, "unavailable": None,
                              "unavailable_message": None},
            "grid": {"rows": rows, "caption_html": _GRID_CAPTION_HTML},
            "optimizer": _optimizer_seed(bundle)}


def _unavail(msg: str) -> dict:
    return {"available": False, "unavailable": "guard", "unavailable_message": msg}


# --------------------------------------------------------------------------- #
# Formatters (mirror app.py _fmt_pct/_fmt_num/_delta_* and _conc_* exactly).
# --------------------------------------------------------------------------- #
def _fin(v) -> bool:
    return v is not None and np.isfinite(float(v))


def _fmt_pct(v, p: int = 2) -> str:
    return "—" if not _fin(v) else f"{float(v) * 100:.{p}f}%"


def _fmt_num(v, p: int = 2) -> str:
    return "—" if not _fin(v) else f"{float(v):.{p}f}"


def _fmt_conc_pct(v, p: int = 1) -> str:
    return "—" if not _fin(v) else f"{float(v):.{p}f}%"


def _d_pct(v) -> str:
    return "" if not _fin(v) else f"{float(v) * 100:+.2f}pp"


def _d_num(v) -> str:
    return "" if not _fin(v) else f"{float(v):+.2f}"


def _d_pp(v, p: int = 1) -> str:
    return "" if not _fin(v) else f"{float(v):+.{p}f}pp"


def _dir(delta, higher_better: bool) -> str:
    """'up' (green) when the change is an improvement, 'down' (red) when worse."""
    if not _fin(delta) or abs(float(delta)) < 1e-12:
        return "flat"
    improved = (float(delta) > 0) if higher_better else (float(delta) < 0)
    return "up" if improved else "down"


def _tile(label: str, pair: dict, val_fmt, delta_fmt, higher_better: bool,
          prefix: str = "Δ ") -> dict:
    """A renderRiskTiles tile: after value colored by direction-of-goodness, Δ in sub."""
    after = pair.get("after")
    delta = pair.get("delta")
    dstr = delta_fmt(delta)
    return {"label": label, "value": val_fmt(after),
            "dir": _dir(delta, higher_better),
            "sub": (prefix + dstr) if dstr else "—"}


def _headline(head: dict, conc: dict) -> dict:
    """The three before/after tile groups (app.py 7793-7841). One MCR tile per
    entry in head['mcr_candidates'] (up to MAX_CANDIDATES) whose verdict is a
    real read (not 'unknown') is appended to the diversification group."""
    div_tiles = [
        _tile("Diversification Ratio", head["dr"], _fmt_num, _d_num, higher_better=True),
        _tile("Down-β vs SPY", head["down_beta"], _fmt_num, _d_num, higher_better=False),
    ]
    for mc in head.get("mcr_candidates", []):
        if mc["verdict"] == "unknown":
            continue
        verdict = mc["verdict"]
        direction = {"diversifying": "up", "neutral": "flat",
                     "risk_adding": "down"}.get(verdict, "flat")
        div_tiles.append({
            "label": f"MCR({mc['ticker']})",
            "value": _fmt_pct(mc["mcr"]),
            "dir": direction,
            "sub": f"{verdict} · vs σ_p {_fmt_pct(head['vol']['after'])}"})
    return {
        "risk": [
            _tile("Portfolio vol (ann.)", head["vol"], _fmt_pct, _d_pct, higher_better=False),
            _tile("Sharpe", head["sharpe"], _fmt_num, _d_num, higher_better=True),
            _tile("Max drawdown", head["max_dd"], _fmt_pct, _d_pct, higher_better=True),
        ],
        "diversification": div_tiles,
        "concentration": [
            _tile("Effective N", conc["effective_n"], _fmt_num, _d_num, higher_better=True),
            _tile("Top-5 weight", conc["top5_pct"], _fmt_conc_pct, _d_pp, higher_better=False),
            _tile("Max weight", conc["max_pct"], _fmt_conc_pct, _d_pp, higher_better=False),
        ],
        "caption_html": (
            "Cards show the <b>after</b> value with Δ vs before; green = improved. "
            "Lower is better for vol / down-β / concentration; higher Effective N "
            "and Diversification Ratio are better. Sharpe uses a 3-month-Treasury "
            "RF baseline. Sortino, VaR/CVaR, correlations and stressed DR are in "
            "the detail panels below."),
    }


def _coverage_html(cov: dict) -> str | None:
    """app.py 7707-7723, extended to N candidates. Pure reweight: overlap window
    only (unchanged). With candidates: one 'candidate TICKER from …' bit per
    ``cov['candidates']`` entry with a known inception, then the overlap window,
    then one 'spliced TICKER<->PROXY' bit per spliced entry.
    ``compute_before_after`` always populates ``cov['candidates']`` (empty list
    for a pure reweight), so this reads only that key."""
    bits = []
    cands = cov.get("candidates") or []
    for c in cands:
        if c.get("inception") is not None:
            bits.append(f"candidate <b>{c['ticker']}</b> from <b>"
                        f"{pd.Timestamp(c['inception']).strftime('%Y-%m-%d')}</b>")

    start, end, days = cov.get("overlap_start"), cov.get("overlap_end"), cov.get("overlap_days")
    if start is not None and end is not None:
        bits.append(f"overlap <b>{pd.Timestamp(start).strftime('%Y-%m-%d')}</b> → "
                    f"<b>{pd.Timestamp(end).strftime('%Y-%m-%d')}</b> ({int(days)} trading days)")

    for c in cands:
        if c.get("spliced"):
            bits.append(f"spliced <b>{c['ticker']}</b>↔<b>{c.get('proxy')}</b>")
    return ("📅 Coverage — " + " · ".join(bits)) if bits else None


def _weight_bars(cur: pd.Series, new: pd.Series) -> dict:
    """Current vs simulated weight % per ticker (app.py 7735-7761), sorted by
    current weight desc; tickers with either side > 0. drawGroupedBars shape."""
    idx = [t for t in cur.index.union(new.index)]
    cur_al = cur.reindex(idx).fillna(0.0)
    new_al = new.reindex(idx).fillna(0.0)
    keep = [t for t in idx if cur_al[t] > 1e-9 or new_al[t] > 1e-9]
    keep.sort(key=lambda t: float(cur_al[t]), reverse=True)
    port = [{"x": str(t), "v": float(cur_al[t]) * 100.0} for t in keep]
    bench = [{"x": str(t), "v": float(new_al[t]) * 100.0} for t in keep]
    return {"port": port, "bench": bench}


def _bt_row(metric: str, pair: dict, val_fmt, delta_fmt) -> dict:
    """A before/after/Δ table row (formatted strings)."""
    return {"metric": metric,
            "before": val_fmt(pair.get("before")), "after": val_fmt(pair.get("after")),
            "delta": (delta_fmt(pair.get("delta")) or "—")}


def _corr_heat(df) -> dict | None:
    """Reuse the riskcontrib N×N heatmap builder over the matrix's own symbols,
    symmetric [-1, 1] ρ domain. None when < 2 symbols."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return _corr_heatmap(df, [str(c) for c in df.columns], -0.6, 1.0)


def _risk_contrib_rows(per) -> dict | None:
    """The after-state per-position risk contribution table (app.py 7896-7906)."""
    if not isinstance(per, pd.DataFrame) or per.empty:
        return None
    cols = [c for c in ("weight_pct", "standalone_vol_ann", "mctr_ann",
                        "cctr_ann", "pctr_pct") if c in per.columns]
    rows = []
    for sym, r in per.iterrows():
        def g(c, scale=1.0):
            v = r.get(c)
            return _fmt_conc_pct(float(v) * scale) if (c in cols and _fin(v)) else "—"
        rows.append({"symbol": str(sym),
                     "weight_pct": g("weight_pct"),
                     "standalone_vol_ann": g("standalone_vol_ann", 100.0),
                     "mctr_ann": _fmt_pct(r.get("mctr_ann")) if _fin(r.get("mctr_ann")) else "—",
                     "cctr_ann": _fmt_pct(r.get("cctr_ann")) if _fin(r.get("cctr_ann")) else "—",
                     "pctr_pct": g("pctr_pct")})
    return {"rows": rows}


_DR_NOTE_HTML = (
    "<p>DR is the no-diversification baseline divided by realized portfolio vol; "
    "&gt;1 means correlations are reducing total risk below the weighted-average "
    "standalone vol.</p>")


def _mcr_detail_html(head: dict) -> str | None:
    """The MCR sub-block for the diversification detail panel (app.py 7865-7879),
    extended to N candidates. One <p> per head['mcr_candidates'] entry with a
    real verdict; None (key omitted) when none do."""
    vol_after = head["vol"]["after"]
    parts = []
    for mc in head.get("mcr_candidates", []):
        if mc["verdict"] == "unknown":
            continue
        mcr = mc["mcr"]
        ratio = (mcr / vol_after if (_fin(mcr) and _fin(vol_after) and vol_after > 0)
                 else float("nan"))
        parts.append(
            f"<p><b>MCR({mc['ticker']})</b> = {_fmt_pct(mcr)} (vol per unit weight, "
            f"annualized; from the after-state covariance) · σ_p (after) = "
            f"{_fmt_pct(vol_after)} · ratio = {_fmt_num(ratio)} → verdict "
            f"<b>{mc['verdict']}</b></p>")
    return "".join(parts) if parts else None


def _detail(head: dict, det: dict) -> dict:
    """The four detail panels (app.py 7852-7974). The MCR sub-block reads
    candidate data straight from ``head['mcr_candidates']``."""
    vol_table = {"rows": [
        _bt_row("Portfolio vol (ann.)", head["vol"], _fmt_pct, _d_pct),
        _bt_row("Sharpe", head["sharpe"], _fmt_num, _d_num),
        _bt_row("Sortino", head["sortino"], _fmt_num, _d_num),
    ]}

    diversification = {
        "dr_note_html": _DR_NOTE_HTML,
        "corr_before": _corr_heat(det.get("corr_matrix_before")),
        "corr_after": _corr_heat(det.get("corr_matrix_after")),
        "risk_contrib": _risk_contrib_rows(det.get("risk_contribution_after")),
    }
    mcr_html = _mcr_detail_html(head)
    if mcr_html is not None:
        diversification["mcr_html"] = mcr_html

    ddb = det.get("drawdown_curve_before")
    dda = det.get("drawdown_curve_after")
    drawdown = None
    if (isinstance(ddb, pd.Series) and isinstance(dda, pd.Series)
            and not ddb.empty and not dda.empty):
        drawdown = {"series": _overlay([
            ("Before", "#8794A9", True, 1.6, ddb * 100.0),
            ("After", "#4DA3F5", False, 2.2, dda * 100.0),
        ])}
    tail = {"drawdown": drawdown, "tail_table": {"rows": [
        _bt_row("Max drawdown", head["max_dd"], _fmt_pct, _d_pct),
        _bt_row("VaR 95% (daily)", head["var95"], _fmt_pct, _d_pct),
        _bt_row("CVaR 95% (daily)", head["cvar95"], _fmt_pct, _d_pct),
    ]}}

    mb, ma = det.get("stress_meta_before", {}), det.get("stress_meta_after", {})
    stress = {
        "caption_html": (
            f"Stress days (SPY log-return z ≤ −1.50): before = "
            f"{mb.get('n_stress', 0)}/{mb.get('n_full', 0)}, after = "
            f"{ma.get('n_stress', 0)}/{ma.get('n_full', 0)}. Spearman correlation "
            "is used inside the stress-day matrix to stay robust on small samples."),
        "scorr_before": _corr_heat(det.get("stressed_corr_matrix_before")),
        "scorr_after": _corr_heat(det.get("stressed_corr_matrix_after")),
        "stress_table": {"rows": [
            _bt_row("Conditional avg corr", head["stressed_corr_avg"], _fmt_num, _d_num),
            _bt_row("Down-β vs SPY", head["down_beta"], _fmt_num, _d_num),
            _bt_row("Stressed DR", head["stressed_dr"], _fmt_num, _d_num),
        ]},
    }
    return {"vol_table": vol_table, "diversification": diversification,
            "tail": tail, "stress": stress}


def _bta(pair: dict, scale: float = 1.0) -> dict:
    """A {before, after, delta} fact from a raw result pair. `scale`=100 turns a
    fraction into percent; delta scales the same. Non-finite -> null (reuses the
    module's _fin; rss must not import ai, so ai._num is out of reach here)."""
    def _n(v):
        return round(float(v) * scale, 4) if _fin(v) else None
    return {"before": _n(pair.get("before")), "after": _n(pair.get("after")),
            "delta": _n(pair.get("delta"))}


def _simulate_ai_facts(result: dict, cur: pd.Series, new: pd.Series) -> dict:
    """Compact, unit-normalized, NaN-safe before/after facts for the AI box.
    Unit rules are colocated with the display formatters on purpose (see the
    plan's unit table): vol/max_dd/VaR/CVaR/mcr are fractions (x100); top5/max
    weight are already percent; the rest are plain numbers.
    ``facts['candidates']`` is built straight from ``head['mcr_candidates']``."""
    head, conc = result["headline"], result["concentration"]
    facts = {
        "vol_pct": _bta(head["vol"], 100.0),
        "sharpe": _bta(head["sharpe"]),
        "sortino": _bta(head["sortino"]),
        "max_dd_pct": _bta(head["max_dd"], 100.0),
        "var95_pct": _bta(head["var95"], 100.0),
        "cvar95_pct": _bta(head["cvar95"], 100.0),
        "dr": _bta(head["dr"]),
        "down_beta": _bta(head["down_beta"]),
        "stressed_corr_avg": _bta(head["stressed_corr_avg"]),
        "stressed_dr": _bta(head["stressed_dr"]),
        "effective_n": _bta(conc["effective_n"]),
        "top5_pct": _bta(conc["top5_pct"]),      # already percent — no scale
        "max_pct": _bta(conc["max_pct"]),        # already percent — no scale
    }
    # Top-6 real weight moves (|delta| > 0.01pp), largest first, ticker-tiebroken
    # for cross-platform-stable ordering. cur/new are FRACTIONS -> x100.
    idx = cur.index.union(new.index)
    cur_al = cur.reindex(idx).fillna(0.0)
    new_al = new.reindex(idx).fillna(0.0)
    moves = []
    for t in idx:
        b, a = float(cur_al[t]) * 100.0, float(new_al[t]) * 100.0
        if abs(a - b) > 0.01:
            moves.append({"ticker": str(t), "before_pct": round(b, 2),
                          "after_pct": round(a, 2), "delta_pp": round(a - b, 2)})
    moves.sort(key=lambda m: (-abs(m["delta_pp"]), m["ticker"]))
    facts["weight_moves"] = moves[:6]
    facts["candidates"] = [
        {"ticker": str(mc["ticker"]),
         "mcr_pct": (round(float(mc["mcr"]) * 100.0, 4) if _fin(mc["mcr"]) else None),
         "verdict": mc["verdict"]}
        for mc in head.get("mcr_candidates", [])
        if mc["verdict"] != "unknown"
    ]
    return facts


# Offline seam + live-key shim for the candidate fetch (Slice 4). The service is
# pure compute EXCEPT the candidate path, which fetches Polygon history on Run.
# A committed data-dir sidecar short-circuits the live fetch for tests (the
# dip_service dip_adhoc_source.csv precedent); the live path resolves the key the
# way Streamlit does (env -> Windows User registry), terminal-local, no app.py import.
_CAND_SIDECAR = "whatif_candidate_source.csv"
# Isolated cache for the offline path so tests never write the tracked fixture tree
# or the real data/whatif_cache. Memoizes the immutable sidecar (deterministic);
# if you EDIT the sidecar, delete this dir to drop the memoized copy.
_CAND_TEST_CACHE = Path(tempfile.gettempdir()) / "meridian_whatif_test_cache"


def _ensure_massive_key() -> None:
    """Resolve MASSIVE_API_KEY into os.environ the way app.py does: process env
    first, else the Windows User-scope registry. No-op off win32 / if already set.
    parsers._config.get_massive_key then finds it. Terminal-local (no app.py import)."""
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if key and key != "PASTE_YOUR_KEY_HERE":
        return
    if sys.platform != "win32":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
            v, _ = winreg.QueryValueEx(k, "MASSIVE_API_KEY")
        if str(v).strip():
            os.environ["MASSIVE_API_KEY"] = str(v).strip()
    except Exception:
        pass


def _offline_candidate_provider(src: Path):
    """A bars_provider reading a long `symbol,date,close` sidecar (filters by the
    requested ticker), matching fetch_candidate_history's injected-provider contract:
    (ticker, start, end) -> [{"date": str, "close": float}, ...]."""
    df = pd.read_csv(src)

    def provider(ticker: str, start: date, end: date) -> list[dict]:
        g = df[df["symbol"].astype(str).str.upper() == str(ticker).upper()]
        return [{"date": str(d), "close": float(c)}
                for d, c in zip(g["date"], g["close"])]
    return provider


def _candidate_provider(data_dir):
    """(bars_provider, cache_dir) for the candidate fetch. Sidecar present ->
    (offline reader, isolated temp cache). Else -> (None, None): the live Polygon
    provider + the real 7-day data/whatif_cache, after ensuring the API key."""
    src = Path(data_dir) / _CAND_SIDECAR
    if src.exists():
        return _offline_candidate_provider(src), _CAND_TEST_CACHE
    _ensure_massive_key()
    return None, None


def run_simulation(frames: Frames, new_weights: dict, *,
                   account: str | list[str] = "all",
                   asset_class: str | list[str] = "all",
                   candidates: list[dict] | None = None,
                   bundle: dict | None = None) -> dict:
    """POST /api/risksim/simulate core. `new_weights` maps ticker -> New % (the
    grid). `candidates` is the up-to-``MAX_CANDIDATES`` list form: each dict is
    ``{"ticker", "proxy"}`` (a blank/omitted proxy = no splice for that name).
    Each accepted candidate is fetched (cache-first, Polygon via the
    _candidate_provider seam), optionally spliced with its proxy, and modeled
    as a new name -> its own MCR verdict. Domain failures (held / empty fetch /
    short overlap) come back as {error}; a soft proxy-empty degradation for a
    candidate adds a `note` (present only then, so the pure-reweight shape is
    untouched) instead of failing the whole run.

    ``bundle``: optional precomputed ``_bundle_for(frames, account, asset_class)``
    for the SAME filters; None computes it here."""
    blank = {"error": None, "coverage_html": None, "weight_bars": None,
             "headline": None, "detail": None}

    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {**blank, "error": "Need twr_portfolio.csv for risk metrics."}
    if bundle is None:
        bundle = _bundle_for(frames, account, asset_class)
    cur = bundle["weights"]
    if cur.empty:
        return {**blank, "error": "Filtered universe has no positions to model against."}
    if frames.daily_prices.empty:
        return {**blank, "error": "Daily prices required for Risk Simulation math."}

    new = pd.Series({str(t): float(v) / 100.0 for t, v in new_weights.items()},
                    dtype=float)
    new = new[new > 1e-9]
    held = {str(s).upper() for s in cur.index}

    specs = candidates or []
    provider, cache_dir = _candidate_provider(frames.data_dir)
    eng_cands, hist_by_ticker, notes = [], {}, []
    for c in specs:
        tkr = str(c.get("ticker") or "").strip().upper()
        cproxy = str(c.get("proxy") or "").strip().upper()
        if not tkr:
            continue
        if cproxy and cproxy == tkr:
            return {**blank, "error": f"Proxy for {tkr} must differ from the candidate ticker."}
        if tkr in held:
            return {**blank, "error": f"{tkr} is already held — reweight it in the grid."}
        if tkr not in new.index:
            return {**blank, "error": f"{tkr} must carry a weight in the grid to be added."}
        try:
            hist = fetch_candidate_history(tkr, cache_dir=cache_dir, bars_provider=provider)
        except Exception as e:
            return {**blank, "error": f"Polygon fetch failed for {tkr}: {e}"}
        if hist.empty:
            return {**blank, "error": (f"No history returned for {tkr} — check the ticker "
                                       "and that the Polygon key is configured.")}
        use_proxy = None
        if cproxy and cproxy != tkr:
            try:
                ph = fetch_candidate_history(cproxy, cache_dir=cache_dir, bars_provider=provider)
            except Exception as e:
                return {**blank, "error": f"Polygon fetch failed for proxy {cproxy}: {e}"}
            if ph.empty:
                notes.append(f"Proxy {cproxy} for {tkr} returned no history — "
                            "running without splice.")
            else:
                hist = splice_with_proxy(hist, ph)
                use_proxy = cproxy
        hist_by_ticker[tkr] = hist
        eng_cands.append(WhatIfCandidate(tkr, use_proxy))

    try:
        scenario = WhatIfScenario(current_weights=cur, new_weights=new,
                                  candidates=tuple(eng_cands))
        scenario.validate()
    except (ValueError, TypeError) as e:
        return {**blank, "error": f"Scenario validation failed: {e}"}

    result = compute_before_after(
        scenario, frames.daily_prices, hist_by_ticker,
        bench_tr=bundle["bench_tr"], rf_series=_load_rf(frames.data_dir),
        history_start=None)

    note = " ".join(notes) if notes else None
    cov_html = _coverage_html(result["coverage"])
    if result["error"]:
        out = {**blank, "error": result["error"], "coverage_html": cov_html}
        if note:
            out["note"] = note
        return out

    out = {"error": None, "coverage_html": cov_html,
           "weight_bars": _weight_bars(cur, new),
           "headline": _headline(result["headline"], result["concentration"]),
           "detail": _detail(result["headline"], result["detail"]),
           "ai_facts": _simulate_ai_facts(result, cur, new)}
    if note:
        out["note"] = note
    return out


def _augment_with_candidates(frames: Frames, bundle: dict, candidates: list[dict],
                             *, min_overlap_days: int = 252,
                             provider_override: tuple | None = None) -> tuple:
    """Splice user candidate tickers into the optimizer universe (Approach A).

    Returns ``(aug_prices, aug_weights, aug_class_of, warnings)``. Each accepted
    candidate is made investable at **0% weight** (the optimizer sizes it) and
    gets its user-picked floor bucket in ``class_of``. A candidate that is blank,
    already held, a duplicate of an earlier slot, fails to fetch, or has
    ``< min_overlap_days + 1`` days of history after any proxy splice is dropped
    and named in ``warnings`` (an excluded young name would be held at 0% by
    ``build_covariance`` anyway — dropping it up front keeps the message clear).

    ``provider_override``: a ``(bars_provider, cache_dir)`` tuple for tests; None
    resolves the live/offline seam via ``_candidate_provider(frames.data_dir)``.
    """
    weights = bundle["weights"]
    class_of = dict(_mv_class_of(bundle))
    warnings: list[str] = []
    if not candidates:
        return frames.daily_prices, weights, class_of, warnings

    provider, cache_dir = (provider_override if provider_override is not None
                           else _candidate_provider(frames.data_dir))

    held = {str(s).upper() for s in weights.index}
    series_by_ticker: dict[str, pd.Series] = {}
    zeros: dict[str, float] = {}
    seen: set[str] = set()
    for c in candidates:
        tkr = str(c.get("ticker") or "").strip().upper()
        bucket = str(c.get("asset_class") or "other").strip().lower()
        proxy = str(c.get("proxy") or "").strip().upper()
        if not tkr:
            warnings.append("A blank candidate slot was skipped.")
            continue
        if tkr in held:
            warnings.append(f"{tkr} is already held — skipped (reweight it in the grid).")
            continue
        if tkr in seen:
            warnings.append(f"{tkr} entered more than once — skipped the duplicate.")
            continue
        seen.add(tkr)
        try:
            hist = fetch_candidate_history(tkr, cache_dir=cache_dir, bars_provider=provider)
        except Exception as e:
            warnings.append(f"{tkr}: fetch failed ({e}) — skipped.")
            continue
        if hist.empty:
            warnings.append(f"{tkr}: no history returned — skipped.")
            continue
        if proxy and proxy != tkr:
            try:
                prx = fetch_candidate_history(proxy, cache_dir=cache_dir, bars_provider=provider)
            except Exception:
                prx = pd.Series(dtype=float)
            if not prx.empty:
                hist = splice_with_proxy(hist, prx)
            else:
                warnings.append(f"{tkr}: proxy {proxy} returned no history — used unspliced history.")
        n_obs = int(hist.notna().sum())
        if n_obs < min_overlap_days + 1:
            warnings.append(
                f"{tkr}: only {n_obs} days of history (need >= {min_overlap_days + 1}) "
                "— skipped. Add a proxy to extend it.")
            continue
        series_by_ticker[tkr] = hist
        zeros[tkr] = 0.0
        class_of[tkr] = bucket

    if not series_by_ticker:
        return frames.daily_prices, weights, class_of, warnings
    aug_prices = build_multi_augmented_price_matrix(frames.daily_prices, series_by_ticker)
    aug_weights = pd.concat([weights, pd.Series(zeros, dtype=float)])
    return aug_prices, aug_weights, class_of, warnings


def run_optimize(frames: Frames, *, optimizer: str, cap_pct: float,
                 floors: dict, caps: dict | None = None,
                 budgets: dict | None = None,
                 account: str | list[str] = "all",
                 asset_class: str | list[str] = "all",
                 candidates: list[dict] | None = None,
                 bundle: dict | None = None) -> dict:
    """POST /api/risksim/optimize core. Dispatch to the min-variance / risk-parity
    Suggest engine and shape {kind, message, new_pct}. Well-formed but infeasible
    inputs (cap too low, all-young, non-PD Σ) come back as kind='error' — never a
    raise. `floors` maps floor-bucket -> percent; risk-parity ignores it.
    `caps` maps floor-bucket -> percent class cap (absent / 100 = off);
    both optimizers honor it (min-variance via Dykstra half-spaces, risk-parity
    via group pinning). `budgets` maps bucket -> percent risk budget (ENTERED
    buckets only; 0/absent = unset); risk-parity only — the min-variance
    objective has no use for risk budgets.
    ``bundle``: optional precomputed ``_bundle_for(frames, account, asset_class)``
    for the SAME filters; None computes it here. ``candidates``: optional
    list of ``{ticker, asset_class, proxy}`` dicts spliced into the optimizer
    universe as investable names at 0% weight via ``_augment_with_candidates``;
    when truthy, the returned dict gains a ``warnings`` key."""
    blank = {"kind": "error", "message": "", "new_pct": None}
    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {**blank, "message": "Need twr_portfolio.csv for risk metrics."}
    if bundle is None:
        bundle = _bundle_for(frames, account, asset_class)
    weights = bundle["weights"]
    if weights.empty:
        return {**blank, "message": "Filtered universe has no positions to model against."}
    if frames.daily_prices.empty:
        return {**blank, "message": "Daily prices required for Risk Simulation math."}

    class_of = _mv_class_of(bundle)
    daily_prices = frames.daily_prices
    warnings: list[str] = []
    if candidates:
        daily_prices, weights, class_of, warnings = _augment_with_candidates(
            frames, bundle, candidates)

    cap = float(cap_pct) / 100.0
    buckets = {b for b in class_of.values() if b != "other"}
    class_caps = {b: float((caps or {}).get(b, 100.0)) / 100.0
                  for b in buckets}
    class_risk_budgets = {b: float(v) / 100.0
                          for b, v in (budgets or {}).items()
                          if float(v) > 0.0}
    if optimizer == "min_variance":
        class_floors = {b: float(floors.get(b, 0.0)) / 100.0 for b in buckets}
        out = suggest_min_variance_grid(daily_prices, weights, class_of,
                                        name_cap=cap, class_floors=class_floors,
                                        class_caps=class_caps)
    elif optimizer == "risk_parity":
        out = suggest_risk_parity_grid(daily_prices, weights,
                                       name_cap=cap, class_of=class_of,
                                       class_caps=class_caps,
                                       class_risk_budgets=class_risk_budgets)
    else:
        return {**blank, "message": f"Unknown optimizer: {optimizer}"}

    new_pct = out["new_pct"]
    if new_pct is not None:
        new_pct = {str(k): float(v) for k, v in new_pct.items()}
    result = {"kind": out["kind"], "message": out["message"], "new_pct": new_pct}
    if candidates:
        result["warnings"] = warnings
    return result


def _floors_caps_fragments(floors: dict, caps: dict | None,
                           buckets: list[str]) -> tuple[str, str]:
    """(floors_txt, caps_txt) shared verbatim by the trace and frontier
    captions. floors_txt lists every bucket's floor percent ("none" when
    there are no buckets); caps_txt is the finished ' Class caps applied:
    …' sentence ('' when no cap is below 100) — one wording for both
    captions (O4b-1 unification)."""
    floors_txt = ", ".join(f"{b.replace('_', ' ')} {float(floors.get(b, 0.0)):.0f}%"
                           for b in buckets) or "none"
    active = {b: float((caps or {}).get(b, 100.0)) for b in buckets
              if float((caps or {}).get(b, 100.0)) < 100.0}
    caps_txt = ("" if not active else
                " Class caps applied: " + ", ".join(
                    f"{b.replace('_', ' ')} {v:.0f}%"
                    for b, v in sorted(active.items())) + ".")
    return floors_txt, caps_txt


_SWEEP_BUSY_MSG = "A sweep is already running — wait for it to finish."
_SWEEP_LOCK = threading.Lock()
_SWEEP_PROGRESS: dict | None = None


def _sweep_begin(op: str, total: int) -> bool:
    """Claim the single sweep slot. False (caller must bail with
    _SWEEP_BUSY_MSG) when another sweep holds it — uvicorn runs sync routes
    in a threadpool, so overlapping POSTs are real; the FE lockout (O4b-2)
    is UX, this lock is the guarantee."""
    global _SWEEP_PROGRESS
    with _SWEEP_LOCK:
        if _SWEEP_PROGRESS is not None:
            return False
        _SWEEP_PROGRESS = {"op": op, "done": 0, "total": int(total),
                           "started": time.time()}
        return True


def _sweep_tick(done: int, total: int) -> None:
    with _SWEEP_LOCK:
        if _SWEEP_PROGRESS is not None:
            _SWEEP_PROGRESS["done"] = int(done)
            _SWEEP_PROGRESS["total"] = int(total)


def _sweep_end() -> None:
    global _SWEEP_PROGRESS
    with _SWEEP_LOCK:
        _SWEEP_PROGRESS = None


def sweep_progress() -> dict:
    """GET /api/risksim/progress payload. Reads only the slot — no frames."""
    with _SWEEP_LOCK:
        if _SWEEP_PROGRESS is None:
            return {"running": False}
        return {"running": True, "op": _SWEEP_PROGRESS["op"],
                "done": _SWEEP_PROGRESS["done"],
                "total": _SWEEP_PROGRESS["total"]}


def _group_skip_reasons(skipped: list[tuple]) -> list[dict]:
    """(cap|lam, optimizer, message) tuples -> [{reason, n}] in
    first-occurrence order. The engines preserve every distinct reason;
    pre-O4b the service surfaced only skipped[0] and only when the sweep
    came back entirely empty."""
    groups: dict[str, int] = {}
    for _x, _opt, msg in skipped:
        groups[str(msg)] = groups.get(str(msg), 0) + 1
    return [{"reason": r, "n": n} for r, n in groups.items()]


def run_trace(frames: Frames, *, cap_pct: float, floors: dict,
              caps: dict | None = None, budgets: dict | None = None,
              account: str | list[str] = "all",
              asset_class: str | list[str] = "all",
              candidates: list[dict] | None = None,
              bundle: dict | None = None) -> dict:
    """POST /api/risksim/trace core. Sweep the fixed cap ladder (∪ cap_pct) across
    BOTH optimizers via opt_curve.trace_cap_curve and shape (vol, Effective N) per
    (cap, optimizer) for the concentration-tradeoff chart. Well-formed inputs that
    can't build Σ come back error-set (never a raise); individually infeasible caps
    raise skipped_n and drop those points. `floors` maps floor-bucket -> percent;
    risk-parity ignores it.
    `caps` maps floor-bucket -> percent class cap (absent / 100 = off);
    both optimizers honor it (min-variance via Dykstra half-spaces, risk-parity
    via group pinning). `budgets` maps bucket -> percent risk budget (ENTERED
    buckets only; 0/absent = unset); risk-parity only — the min-variance
    objective has no use for risk budgets.
    Each point also carries ``weights_pct`` (the solve's full-book percent
    allocation, verbatim — apply it without re-solving). ``skipped_reasons``
    groups infeasible-cap skips into ``[{reason, n}]`` (first-occurrence
    order), designed for reuse by the frontier via ``_group_skip_reasons``.
    ``bundle``: optional precomputed ``_bundle_for(frames, account, asset_class)``
    for the SAME filters; None computes it here. ``candidates``: optional
    what-if tickers folded into the sweep universe via
    ``_augment_with_candidates``; when truthy, the payload gains a
    ``warnings`` key and accepted candidates appear in every point's
    ``weights_pct``.

    Runs inside the single-flight sweep slot (``_sweep_begin``/``_sweep_tick``/
    ``_sweep_end``); a refused overlap returns ``error=_SWEEP_BUSY_MSG`` with
    an empty series."""
    blank = {"error": None, "series": [], "current": None,
             "skipped_n": 0, "skipped_reasons": [], "empty_message": None,
             "caption_html": ""}
    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {**blank, "error": "Need twr_portfolio.csv for risk metrics."}
    if bundle is None:
        bundle = _bundle_for(frames, account, asset_class)
    weights = bundle["weights"]
    if weights.empty:
        return {**blank, "error": "Filtered universe has no positions to model against."}
    if frames.daily_prices.empty:
        return {**blank, "error": "Daily prices required for Risk Simulation math."}

    class_of = _mv_class_of(bundle)
    daily_prices = frames.daily_prices
    warnings: list[str] = []
    if candidates:
        daily_prices, weights, class_of, warnings = _augment_with_candidates(
            frames, bundle, candidates)
    buckets = sorted({b for b in class_of.values() if b != "other"})
    class_floors = {b: float(floors.get(b, 0.0)) / 100.0 for b in buckets}
    class_caps = {b: float((caps or {}).get(b, 100.0)) / 100.0 for b in buckets}
    class_risk_budgets = {b: float(v) / 100.0
                          for b, v in (budgets or {}).items()
                          if float(v) > 0.0}
    ladder = sorted({*_TRACE_LADDER_PCT, float(cap_pct)})
    if not _sweep_begin("trace", len(ladder) * 2):
        return {**blank, "error": _SWEEP_BUSY_MSG}
    try:
        trace = trace_cap_curve(daily_prices, weights, class_of,
                                caps=[c / 100.0 for c in ladder],
                                class_floors=class_floors, class_caps=class_caps,
                                class_risk_budgets=class_risk_budgets,
                                on_point=_sweep_tick)
    finally:
        _sweep_end()
    if trace["error"]:
        return {**blank, "error": trace["error"]}

    pts = trace["points"]
    series = []
    for key, label in _TRACE_SERIES:
        d = pts[pts["optimizer"] == key].sort_values("cap")
        if d.empty:
            continue
        series.append({"key": key, "label": label, "points": [
            {"cap": float(r["cap"]), "vol": float(r["vol"]),
             "effective_n": float(r["effective_n"]),
             "max_weight": float(r["max_weight"]),
             "converged": bool(r["converged"]),
             "weights_pct": {str(k): float(v)
                             for k, v in r["weights"].items()}}
            for _, r in d.iterrows()]})

    cur = trace["current"]
    current = None
    if cur is not None and all(_fin(cur.get(k))
                               for k in ("vol", "effective_n", "max_weight")):
        current = {k: float(cur[k]) for k in ("vol", "effective_n", "max_weight")}

    skipped = trace["skipped"]
    skipped_reasons = _group_skip_reasons(skipped)
    reasons_txt = "; ".join(f"{g['reason'].rstrip('.')} ({g['n']})"
                            for g in skipped_reasons)
    empty_message = (reasons_txt or None) if not series else None
    floors_txt, caps_txt = _floors_caps_fragments(floors, caps, buckets)
    budgets_txt = ("" if not class_risk_budgets else
                   " Risk budgets applied: " + ", ".join(
                       f"{b.replace('_', ' ')} {v * 100:.0f}%"
                       for b, v in sorted(class_risk_budgets.items()))
                   + ".")
    skip_txt = (f" {len(skipped)} infeasible cap point(s) skipped — "
                f"{reasons_txt}." if skipped else "")
    caption = _TRACE_CAPTION_BASE.format(floors=floors_txt,
                                         class_caps=caps_txt,
                                         risk_budgets=budgets_txt,
                                         skipped=skip_txt)
    result = {"error": None, "series": series, "current": current,
              "skipped_n": len(skipped), "skipped_reasons": skipped_reasons,
              "empty_message": empty_message, "caption_html": caption}
    if candidates:
        result["warnings"] = warnings
    return result


def run_frontier(frames: Frames, *, cap_pct: float, floors: dict,
                 caps: dict | None = None, erp_pct: float = 4.5,
                 account: str | list[str] = "all",
                 asset_class: str | list[str] = "all",
                 candidates: list[dict] | None = None,
                 bundle: dict | None = None) -> dict:
    """POST /api/risksim/frontier core. Build CAPM expected returns, sweep the
    risk-aversion ladder under the live constraints via frontier.trace_frontier,
    and shape (vol, E[r]) points for the frontier chart.

    Returns the run_trace shape (single-element ``series`` so the chart
    primitive is shared) plus ``markers``. Each point also carries ``lam``
    (the solve's risk aversion, descending — ladder order) and
    ``weights_pct`` (the solve's full-book percent allocation, verbatim —
    apply it without re-solving). ``skipped_reasons`` groups infeasible-lambda
    skips into ``[{reason, n}]`` (first-occurrence order) via the shared
    ``_group_skip_reasons`` helper. Well-formed inputs that can't build Σ or
    μ come back error-set (never a raise); individually infeasible lambdas
    raise skipped_n. `floors` / `caps` map floor-bucket -> percent; `erp_pct`
    is the equity risk premium in percent. No risk budgets: the frontier is
    min-variance-family only. Factor and risk-free files are read from
    ``frames.data_dir``. `candidates` splices user tickers into the universe
    via ``_augment_with_candidates`` before CAPM μ and the solve, so accepted
    names get priced and appear in every point's ``weights_pct``; when
    truthy the payload also gains ``warnings`` (per-candidate skip reasons).

    Runs inside the single-flight sweep slot (``_sweep_begin``/``_sweep_tick``/
    ``_sweep_end``); a refused overlap returns ``error=_SWEEP_BUSY_MSG`` with
    an empty series. Progress ticks only the lambda-ladder solves — the two
    marker solves ``trace_frontier`` runs afterward (min-variance, equal-risk-
    contribution) aren't ticked, so ``sweep_progress()`` reports total/total
    through them.
    """
    blank = {"error": None, "series": [], "current": None, "markers": [],
             "capm": None, "skipped_n": 0, "skipped_reasons": [],
             "empty_message": None, "caption_html": ""}
    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return {**blank, "error": "Need twr_portfolio.csv for risk metrics."}
    if bundle is None:
        bundle = _bundle_for(frames, account, asset_class)
    weights = bundle["weights"]
    if weights.empty:
        return {**blank,
                "error": "Filtered universe has no positions to model against."}
    if frames.daily_prices.empty:
        return {**blank,
                "error": "Daily prices required for Risk Simulation math."}

    daily_prices = frames.daily_prices
    warnings: list[str] = []
    class_of = _mv_class_of(bundle)
    if candidates:
        daily_prices, weights, class_of, warnings = _augment_with_candidates(
            frames, bundle, candidates)

    _ff_monthly, ff_daily = _fs._load_ff(frames.data_dir)
    rf_series = _rsvc._load_rf(frames.data_dir)
    if rf_series.empty:
        rf_annual, rf_src = _rsvc.RF_FALLBACK_ANNUAL, "fallback"
    else:
        rf_annual, rf_src = float(rf_series.iloc[-1]), "FRED DGS3MO"
    if not _fin(rf_annual):
        rf_annual, rf_src = _rsvc.RF_FALLBACK_ANNUAL, "fallback"
    erp = float(erp_pct) / 100.0

    mu_res = capm_expected_returns(daily_prices, ff_daily,
                                   [str(s) for s in weights.index],
                                   rf_annual=rf_annual, erp=erp,
                                   window_days=_FRONTIER_WINDOW_DAYS)
    if mu_res["error"]:
        return {**blank, "error": mu_res["error"]}
    mu = pd.Series(mu_res["mu"].to_numpy(dtype=float), index=weights.index)

    buckets = sorted({b for b in class_of.values() if b != "other"})
    class_floors = {b: float(floors.get(b, 0.0)) / 100.0 for b in buckets}
    class_caps = {b: float((caps or {}).get(b, 100.0)) / 100.0
                  for b in buckets}
    if not _sweep_begin("frontier", len({float(x) for x in LAMBDA_LADDER})):
        return {**blank, "error": _SWEEP_BUSY_MSG}
    try:
        tr = trace_frontier(daily_prices, weights, class_of, mu,
                            name_cap=float(cap_pct) / 100.0,
                            class_floors=class_floors, class_caps=class_caps,
                            on_point=_sweep_tick)
    finally:
        _sweep_end()
    if tr["error"]:
        return {**blank, "error": tr["error"]}

    keys = ("vol", "exp_return", "effective_n", "max_weight")
    points = [{**{k: float(r[k]) for k in keys},
               "lam": float(r["lam"]),
               "converged": bool(r["converged"]),
               "weights_pct": {str(k): float(v)
                               for k, v in r["weights"].items()}}
              for _, r in tr["points"].iterrows()
              if all(_fin(r[k]) for k in keys)]
    series = ([{"key": "frontier", "label": "Efficient frontier",
                "points": points}] if points else [])

    cur = tr["current"]
    current = None
    if cur is not None and all(_fin(cur.get(k)) for k in keys):
        current = {k: float(cur[k]) for k in keys}
    markers = [{"key": m["key"], "label": m["label"], "vol": float(m["vol"]),
                "exp_return": float(m["exp_return"])}
               for m in tr["markers"]
               if _fin(m.get("vol")) and _fin(m.get("exp_return"))]

    skipped = tr["skipped"]
    skipped_reasons = _group_skip_reasons(skipped)
    reasons_txt = "; ".join(f"{g['reason'].rstrip('.')} ({g['n']})"
                            for g in skipped_reasons)
    empty_message = (reasons_txt or None) if not series else None
    floors_txt, caps_txt = _floors_caps_fragments(floors, caps, buckets)
    assumed = mu_res["assumed"]
    assumed_txt = ("" if not assumed else
                   " β = 1.0 assumed (no fitted regression) for: "
                   + ", ".join(assumed) + ".")
    skip_txt = (f" {len(skipped)} infeasible point(s) skipped — "
                f"{reasons_txt}." if skipped else "")
    floors_active = any(_fin(v) and float(v) > 0.0
                        for v in class_floors.values())
    erc_present = any(m["key"] == "risk_parity" for m in markers)
    erc_note = ("" if not (floors_active and erc_present) else
               " The equal-risk-contribution marker is traced without class "
               "floors — like the tab's risk-parity optimizer — so it can "
               "sit outside the frontier's feasible set.")
    caption = _FRONTIER_CAPTION_BASE.format(
        cap=float(cap_pct), floors=floors_txt, class_caps=caps_txt,
        rf=rf_annual * 100.0, rf_src=rf_src, erp=float(erp_pct),
        years=_FRONTIER_BETA_YEARS, assumed=assumed_txt, erc_note=erc_note,
        skipped=skip_txt)
    capm = {"rf_pct": round(rf_annual * 100.0, 4), "rf_src": rf_src,
            "erp_pct": float(erp_pct), "beta_years": int(_FRONTIER_BETA_YEARS),
            "assumed_beta_names": [str(x) for x in assumed]}
    result = {"error": None, "series": series, "current": current,
              "markers": markers, "capm": capm, "skipped_n": len(skipped),
              "skipped_reasons": skipped_reasons,
              "empty_message": empty_message, "caption_html": caption}
    if candidates:
        result["warnings"] = warnings
    return result


def floor_buckets_for(frames: Frames, bundle: dict) -> set[str]:
    """The optimizer seed's floor-bucket keys exactly as GET /api/risksim lists
    them — including the load-guard corners where the seed block is absent
    (empty set, so any posted floor key 422s, matching the old view-derived
    validation in the optimize/trace handlers)."""
    if frames.twr_portfolio is None or frames.twr_portfolio.empty:
        return set()
    if bundle["weights"].empty or frames.daily_prices.empty:
        return set()
    return {b for b in _mv_class_of(bundle).values() if b != "other"}
