"""Walk-forward referee for the Buy-the-Dip verdict.

Per evaluation date t, replay the EXACT shipped verdict pipeline
(dip_analytics.dip_verdict_block) on data truncated to <= t, then join
realized forward total returns from t. Aggregations report per-band outcome
tables (the referee) and the pre-registered primary metric.

Spec: docs/superpowers/specs/2026-07-14-dip-verdict-backtest-design.md.
Pure numpy/pandas; no network; deterministic (fixed seeds).

CLI:
  py parsers/dip_backtest.py --ticker SPY [--stride 5] [--burn-in-years 10]
                             [--data-dir DIR] [--write]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dip_analytics as da                          # noqa: E402
import dip_ladder as dl                             # noqa: E402
import dip_extend as de                             # noqa: E402
from risk_metrics import compute_drawdown_episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

EDGE_BANDS = frozenset({"strong", "neutral"})    # "edge-claimed" per spec §4
REALIZED_HORIZONS = (21, 63, 126, 252)
PRIMARY_HORIZON = da.VERDICT_HORIZON             # 252
MIN_EDGE_EPISODES = 5                            # primary episode gate (§2)
DEPTH_TK_RULE = 85.0                             # secondary: TK's rule row
BAND_ORDER = ("strong", "neutral", "weak", "inconclusive", "shallow")
_AGG_COLS = ("n_days", "n_episodes", "med_252", "hit_252", "omega_252",
             "med_126", "med_63", "med_21")

REGISTERED_PATH = ROOT / "parsers" / "dip_backtest_registered.json"
REGISTERED_DATE = "2026-07-14"   # the S4 registration date; changes only with
                                 # a new registration (a change-control event)


def load_dip_series(data_dir: Path | str, ticker: str) -> tuple[pd.Series, pd.Series]:
    """(price=close, tr=adj_close) for `ticker` from dip_history.csv
    (long format: symbol,date,close,adj_close)."""
    csv = Path(data_dir) / "dip_history.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run `py parsers/fetch_dip_history.py --write`")
    df = pd.read_csv(csv, parse_dates=["date"])
    sub = df[df["symbol"].astype(str).str.upper() == ticker.upper()]
    if sub.empty:
        raise ValueError(f"ticker {ticker!r} not in {csv}")
    sub = sub.sort_values("date").set_index("date")
    return sub["close"].astype(float), sub["adj_close"].astype(float)


def episode_spans(price: pd.Series) -> list[tuple[int, pd.Timestamp, object]]:
    """(episode_no, peak_date, recovery_date_or_None) per drawdown spell."""
    eps = compute_drawdown_episodes(price.reset_index(drop=True),
                                    pd.Series(price.index))
    return [(k, e["peak_date"], e["recovery_date"]) for k, e in enumerate(eps)]


def episode_id_for(date: pd.Timestamp, spans) -> float:
    """Episode number whose spell contains `date` (peak < date <= recovery;
    open episodes unbounded right). NaN at/above peaks. Clustering key ONLY —
    never feeds the verdict (spec §4)."""
    for no, peak, rec in spans:
        if date > peak and (rec is None or date <= rec):
            return float(no)
    return float("nan")


def walk_forward(price: pd.Series, tr: pd.Series, *, stride: int = 5,
                 burn_in_years: int = 10,
                 horizons: tuple = REALIZED_HORIZONS) -> pd.DataFrame:
    """One row per evaluation date: the verdict computed on data <= t only,
    plus realized forward total returns from t. Frontier-censored at
    max(horizons); evaluations start after `burn_in_years` of history and are
    anchored at the first eligible day."""
    if not price.index.equals(tr.index):
        raise ValueError("price and tr must share one index")
    n = len(price)
    censor = max(horizons)
    first_ok = price.index[0] + pd.DateOffset(years=burn_in_years)
    eligible = [i for i in range(n)
                if price.index[i] >= first_ok and i + censor < n]
    if not eligible:
        raise ValueError(f"insufficient history: {n} days for "
                         f"burn_in_years={burn_in_years} + censor {censor}")
    spans = episode_spans(price)
    tr_arr = tr.to_numpy(dtype=float)
    rows = []
    for i in eligible[::stride]:
        t = price.index[i]
        blk = da.dip_verdict_block(price.iloc[:i + 1], tr.iloc[:i + 1],
                                   horizons=(da.VERDICT_HORIZON,))
        v = blk["verdict"]
        row = {"date": t, "band": v["band"],
               "depth_pctile": float(blk["state"]["pct_history_shallower"]),
               "omega": v["omega"], "edge": v["edge"],
               "edge_ci_lo": float(v["edge_ci"]["lo"]),
               "rr_pct": blk["rr_pct"], "n_cond": int(v["n"]),
               "episode_id": episode_id_for(t, spans)}
        for h in horizons:
            row[f"fwd_{h}"] = tr_arr[i + h] / tr_arr[i] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _agg(rows: pd.DataFrame) -> dict:
    """One aggregation record: day/episode counts + realized forward stats
    (the 252d anchor gets hit-rate and Omega; medians at every horizon)."""
    out = {"n_days": int(len(rows)),
           "n_episodes": int(rows["episode_id"].dropna().nunique())}
    for h in (252, 126, 63, 21):
        r = rows[f"fwd_{h}"].to_numpy(dtype=float)
        out[f"med_{h}"] = float(np.median(r)) if r.size else float("nan")
    r252 = rows["fwd_252"].to_numpy(dtype=float)
    out["hit_252"] = float((r252 > 0).mean()) if r252.size else float("nan")
    out["omega_252"] = da.omega_ratio(r252)
    return out


def _frame(recs: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(recs, orient="index")[list(_AGG_COLS)]


def referee_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per walk-forward band (fixed order) plus the 'all' baseline row."""
    recs = {band: _agg(rows[rows["band"] == band]) for band in BAND_ORDER}
    recs["all"] = _agg(rows)
    return _frame(recs)


def tk_rule_row(rows: pd.DataFrame) -> pd.DataFrame:
    """Secondary (descriptive): depth percentile >= 85 AND edge-claimed."""
    mask = (rows["depth_pctile"] >= DEPTH_TK_RULE) & rows["band"].isin(EDGE_BANDS)
    return _frame({"tk_rule": _agg(rows[mask])})


def depth_decile_curve(rows: pd.DataFrame) -> pd.DataFrame:
    """Secondary (descriptive): realized outcomes by depth-percentile decile;
    the top bucket includes 100.0."""
    recs = {}
    for lo in range(0, 100, 10):
        hi = lo + 10
        top = 100.0 + 1e-9 if hi == 100 else float(hi)
        m = (rows["depth_pctile"] >= lo) & (rows["depth_pctile"] < top)
        recs[f"{lo}-{hi}"] = _agg(rows[m])
    return _frame(recs)


def primary_metric(rows: pd.DataFrame, *, n_boot: int = da.BOOTSTRAP_N,
                   seed: int = da.BOOTSTRAP_SEED, ci: float = da.BOOTSTRAP_CI,
                   stride: int = 5,
                   min_episodes: int = MIN_EDGE_EPISODES) -> dict:
    """THE pre-registered primary (spec §2): S = Omega(fwd_252 | edge-claimed)
    − Omega(fwd_252 | all), stationary-block-bootstrapped over the day-ordered
    rows. Expected block = ceil(252 / stride) ROWS (≈ one horizon of
    overlapping-window dependence — deliberately not the engine's daily-tuned
    21). Outcome: 'validated' (episodes >= min AND ci_lo > 0),
    'not_supported' (episodes >= min, CI fails to clear), else 'inconclusive'."""
    r = rows.sort_values("date")
    fwd = r["fwd_252"].to_numpy(dtype=float)
    edge = r["band"].isin(EDGE_BANDS).to_numpy()
    n = len(r)
    n_ep = int(r.loc[r["band"].isin(EDGE_BANDS), "episode_id"]
                .dropna().nunique())
    block_rows = int(math.ceil(PRIMARY_HORIZON / max(1, stride)))

    def _stat(f: np.ndarray, e: np.ndarray) -> float:
        if not e.any():
            return float("nan")
        return da.omega_ratio(f[e]) - da.omega_ratio(f)

    point = _stat(fwd, edge) if n else float("nan")
    lo = hi = float("nan")
    if n >= 2:
        rng = np.random.default_rng(seed)
        p_new = 1.0 / block_rows
        stats = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = da._stationary_bootstrap_indices(rng, n, p_new)
            stats[b] = _stat(fwd[idx], edge[idx])
        finite = stats[np.isfinite(stats)]
        if finite.size:
            q = (1.0 - ci) / 2.0
            lo = float(np.quantile(finite, q))
            hi = float(np.quantile(finite, 1.0 - q))
    if n_ep < min_episodes:
        outcome = "inconclusive"
    elif np.isfinite(lo) and lo > 0.0:
        outcome = "validated"
    else:
        outcome = "not_supported"
    return {"stat": point, "ci_lo": lo, "ci_hi": hi,
            "n_edge_days": int(edge.sum()), "n_edge_episodes": n_ep,
            "block_rows": block_rows, "outcome": outcome}


def _json_safe_agg(rec: dict) -> dict:
    """JSON-safe copy of an _agg record: non-finite floats -> None, plus
    omega_252_inf=True when Omega is +inf (100% hit -> no losing outcomes;
    the UI renders that as an infinity glyph, distinct from missing)."""
    out = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
           for k, v in rec.items()}
    om = rec.get("omega_252")
    if isinstance(om, float) and math.isinf(om) and om > 0:
        out["omega_252_inf"] = True
    return out


def build_registered_artifact(rows: pd.DataFrame, *, ticker: str, stride: int,
                              burn_in_years: int) -> dict:
    """The committed registered-run record (spec 2026-07-16 §4): referee
    per-band table + TK-rule row + primary metric + provenance. Pure function
    of the walk-forward rows and run params — no wall clock; the registration
    date is the module constant REGISTERED_DATE."""
    ref = {band: _json_safe_agg(_agg(rows[rows["band"] == band]))
           for band in BAND_ORDER}
    ref["all"] = _json_safe_agg(_agg(rows))
    pm = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
          for k, v in primary_metric(rows, stride=stride).items()}
    tk = rows[(rows["depth_pctile"] >= DEPTH_TK_RULE)
              & rows["band"].isin(EDGE_BANDS)]
    return {
        "schema": 1,
        "ticker": ticker.upper(),
        "registered": REGISTERED_DATE,
        "config": {"stride": stride, "burn_in_years": burn_in_years,
                   "censor": max(REALIZED_HORIZONS),
                   "horizon": PRIMARY_HORIZON},
        "evals": {"n": int(len(rows)),
                  "first": str(rows["date"].iloc[0].date()),
                  "last": str(rows["date"].iloc[-1].date())},
        "primary": pm,
        "referee": ref,
        "tk_rule": _json_safe_agg(_agg(tk)),
    }


def load_registered_artifact(path: "Path | None" = None) -> "dict | None":
    """Committed registered-run record, or None on missing / corrupt /
    schema-drifted file. Never raises — a broken artifact degrades the dip
    card to its pre-referee shape, it must not 500 the tab."""
    p = Path(path) if path is not None else REGISTERED_PATH
    try:
        art = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(art, dict) or art.get("schema") != 1:
        return None
    if not {"ticker", "registered", "config", "evals",
            "primary", "referee", "tk_rule"} <= art.keys():
        return None
    ref = art["referee"]
    if not isinstance(ref, dict) or not (set(BAND_ORDER) | {"all"}) <= set(ref):
        return None
    return art


# ---------------------------------------------------------------------------
# Rotation-ladder replay (spec 2026-07-18) — the SECOND registered claim.

LADDER_REGISTERED_DATE = "2026-07-18"
LADDER_REGISTERED_PATH = ROOT / "parsers" / "dip_ladder_registered.json"
LADDER_MIN_TRANCHES = 3


def load_cash_rets(data_dir: Path | str) -> pd.Series:
    """Daily cash-leg returns for the ladder backtest: the Fama-French daily
    risk-free rate (decimal), committed at data_dir/ff_factors_daily.csv.
    BIL/SGOV CLOSES are the wrong instrument here — T-bill ETFs pay their
    yield via distributions, so a close-only series is nearly flat (spec
    Update 2026-07-18)."""
    p = Path(data_dir) / "ff_factors_daily.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run `py parsers/fetch_ff_factors.py --write`")
    df = pd.read_csv(p, parse_dates=["date"])
    return df.set_index("date")["rf"].astype(float).sort_index()


def ladder_evals(rows: pd.DataFrame) -> pd.DataFrame:
    """Eval-day signals for simulate_ladder: band + the ★-rule flag under
    the REGISTERED definition (depth >= DEPTH_TK_RULE and edge-claimed)."""
    return pd.DataFrame({
        "date": rows["date"],
        "band": rows["band"],
        "tk_rule": ((rows["depth_pctile"] >= DEPTH_TK_RULE)
                    & rows["band"].isin(EDGE_BANDS)),
    })


def ladder_backtest(rows: pd.DataFrame, price: pd.Series, tr: pd.Series,
                    cash_rets: pd.Series, *, fractions: dict | None = None,
                    n_boot: int = da.BOOTSTRAP_N,
                    seed: int = da.BOOTSTRAP_SEED,
                    ci: float = da.BOOTSTRAP_CI, stride: int = 5) -> dict:
    """Replay the ladder over the walk-forward eval sequence and score the
    PRE-REGISTERED primary: S = Omega(fwd-252 ladder-wealth returns at eval
    days) − Omega(same for a daily-rebalanced constant-mix baseline at the
    ladder's realized average equity exposure, same cash leg), stationary-
    block-bootstrapped over day-ordered rows (block = ceil(252/stride) rows,
    the verdict referee's convention, same seed). Outcome: 'inconclusive'
    when n_tranches < LADDER_MIN_TRANCHES; 'validated' iff ci_lo > 0; else
    'not_supported'. Registered BEFORE the real run; committed either way."""
    sim = dl.simulate_ladder(price, tr, cash_rets, ladder_evals(rows),
                             fractions=fractions)
    w = float(sim["summary"]["avg_equity_exposure"])
    eq_ret = tr.pct_change().fillna(0.0)
    cash_r = (cash_rets.reindex(tr.index).fillna(0.0)
              if cash_rets is not None and not cash_rets.empty
              else pd.Series(0.0, index=tr.index))
    base_wealth = (1.0 + w * eq_ret + (1.0 - w) * cash_r).cumprod()
    lw = sim["wealth"]
    pos = {d: i for i, d in enumerate(tr.index)}
    H = PRIMARY_HORIZON
    fl, fb = [], []
    for d in rows.sort_values("date")["date"]:
        i = pos.get(pd.Timestamp(d))
        if i is None or i + H >= len(tr.index):
            continue
        fl.append(float(lw.iloc[i + H] / lw.iloc[i] - 1.0))
        fb.append(float(base_wealth.iloc[i + H] / base_wealth.iloc[i] - 1.0))
    fl_a, fb_a = np.asarray(fl, dtype=float), np.asarray(fb, dtype=float)
    n = len(fl_a)
    block_rows = int(math.ceil(H / max(1, stride)))

    def _stat(a: np.ndarray, b: np.ndarray) -> float:
        return da.omega_ratio(a) - da.omega_ratio(b)

    point = _stat(fl_a, fb_a) if n else float("nan")
    lo = hi = float("nan")
    if n >= 2:
        rng = np.random.default_rng(seed)
        p_new = 1.0 / block_rows
        stats = np.empty(n_boot, dtype=float)
        for b_i in range(n_boot):
            idx = da._stationary_bootstrap_indices(rng, n, p_new)
            stats[b_i] = _stat(fl_a[idx], fb_a[idx])
        finite = stats[np.isfinite(stats)]
        if finite.size:
            q = (1.0 - ci) / 2.0
            lo = float(np.quantile(finite, q))
            hi = float(np.quantile(finite, 1.0 - q))
    nt = int(sim["summary"]["n_tranches"])
    if nt < LADDER_MIN_TRANCHES:
        outcome = "inconclusive"
    elif np.isfinite(lo) and lo > 0.0:
        outcome = "validated"
    else:
        outcome = "not_supported"
    return {"stat": point, "ci_lo": lo, "ci_hi": hi, "n_evals": n,
            "n_tranches": nt,
            "skipped_deploys": int(sim["summary"]["skipped_deploys"]),
            "avg_equity_exposure": w,
            "final_wealth": float(sim["summary"]["final_wealth"]),
            "baseline_final_wealth": float(base_wealth.iloc[-1]),
            "block_rows": block_rows, "outcome": outcome,
            "tranches": sim["tranches"]}


def _json_num(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def build_ladder_registered_artifact(rows: pd.DataFrame, price: pd.Series,
                                     tr: pd.Series, cash_rets: pd.Series, *,
                                     ticker: str, stride: int,
                                     burn_in_years: int,
                                     fractions: dict | None = None) -> dict:
    """The committed ladder-run record — config + fractions (part of the
    registered claim) + primary + honest outcome. Pure function; the
    registration date is the module constant LADDER_REGISTERED_DATE."""
    fr = dict(dl.LADDER_FRACTIONS if fractions is None else fractions)
    res = ladder_backtest(rows, price, tr, cash_rets, fractions=fr,
                          stride=stride)
    return {
        "schema": 1,
        "kind": "ladder",
        "ticker": ticker.upper(),
        "registered": LADDER_REGISTERED_DATE,
        "config": {"stride": stride, "burn_in_years": burn_in_years,
                   "censor": max(REALIZED_HORIZONS),
                   "horizon": PRIMARY_HORIZON,
                   "cash_leg": "ff_rf_daily",
                   "exit": "recovery_to_anchor_peak",
                   "min_tranches": LADDER_MIN_TRANCHES},
        "fractions": fr,
        "evals": {"n": int(len(rows)),
                  "first": str(rows["date"].iloc[0].date()),
                  "last": str(rows["date"].iloc[-1].date())},
        "ladder": {k: _json_num(res[k]) for k in
                   ("n_tranches", "skipped_deploys", "avg_equity_exposure",
                    "final_wealth", "baseline_final_wealth")},
        "primary": {k: _json_num(res[k]) for k in
                    ("stat", "ci_lo", "ci_hi", "n_evals", "block_rows",
                     "outcome")},
    }


def load_ladder_registered_artifact(path: "Path | None" = None) -> "dict | None":
    """Committed ladder-run record, or None on missing / corrupt / drifted
    file. Never raises — same degrade discipline as the verdict artifact."""
    p = Path(path) if path is not None else LADDER_REGISTERED_PATH
    try:
        art = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(art, dict) or art.get("schema") != 1:
        return None
    if art.get("kind") != "ladder":
        return None
    if not {"ticker", "registered", "config", "fractions", "evals",
            "ladder", "primary"} <= art.keys():
        return None
    return art


# ---------------------------------------------------------------------------
# Strong-gate recalibration experiment (spec 2026-07-18) — the THIRD
# registered claim. Candidate gates re-split strong/neutral INSIDE the
# validated edge set; edge membership (strong ∪ neutral) is gate-invariant.

GATE_REGISTERED_DATE = "2026-07-18"
GATE_REGISTERED_PATH = ROOT / "parsers" / "dip_gate_registered.json"

# Ordered by adoption preference (spec §2): first candidate whose strong
# band validates out-of-sample ships. Params mirror dip_buy_verdict's
# strong clause: `strong_rr` = the reward/risk depth-rank threshold;
# `requires_ci` = whether the 90% edge-CI must clear zero (True) or the
# point edge estimate suffices (False).
GATE_CANDIDATES = (
    {"id": "rr0", "strong_rr": 0.0, "requires_ci": True, "preference": 1},
    {"id": "rr50", "strong_rr": 0.5, "requires_ci": True, "preference": 2},
    {"id": "pt67", "strong_rr": 0.67, "requires_ci": False, "preference": 3},
)


def gate_relabel(rows: pd.DataFrame, *, strong_rr: float,
                 requires_ci: bool) -> pd.Series:
    """Candidate band per eval row, derived ONLY from stored walk-forward
    fields. Exact relabeling of dip_buy_verdict's cascade: banding there is
    pure post-processing of (omega, edge_ci, edge, rr) which are computed
    before the band — so a candidate gate never needs a re-replay. Bands
    upstream of the strong condition (shallow/inconclusive/weak) are
    unchanged; the strong/neutral split is recomputed. NaN rank or NaN CI
    can never claim strong (comparison semantics match the verdict)."""
    band = rows["band"].astype(str)
    omega = rows["omega"].to_numpy(dtype=float)
    edge = rows["edge"].to_numpy(dtype=float)
    ci_lo = rows["edge_ci_lo"].to_numpy(dtype=float)
    rr = rows["rr_pct"].to_numpy(dtype=float)

    rr_ok = np.isfinite(rr) & (rr >= strong_rr)
    inf_o = np.isinf(omega)
    if requires_ci:
        gate_ok = np.isfinite(ci_lo) & (ci_lo > 0.0)
    else:
        gate_ok = edge > 0.0                      # NaN edge -> False
    strong = (inf_o & rr_ok) | (~inf_o & gate_ok & (omega > 1.0) & rr_ok)
    neutral = inf_o | ((edge > 0.0) & (omega > 1.0))
    relab = np.where(strong, "strong", np.where(neutral, "neutral", "weak"))

    out = band.copy()
    is_edge = band.isin(EDGE_BANDS).to_numpy()
    out[is_edge] = relab[is_edge]
    return out


def strong_primary_metric(rows: pd.DataFrame, bands: pd.Series, *,
                          n_boot: int = da.BOOTSTRAP_N,
                          seed: int = da.BOOTSTRAP_SEED,
                          ci: float = da.BOOTSTRAP_CI, stride: int = 5,
                          min_episodes: int = MIN_EDGE_EPISODES) -> dict:
    """Per-candidate registered primary (spec 2026-07-18 §4):
    S = Omega(fwd_252 | strong-under-candidate) − Omega(fwd_252 | all),
    stationary-block-bootstrapped over day-ordered rows resampling
    (fwd, mask) pairs jointly — primary_metric's exact conventions
    (block = ceil(252/stride) rows, same seed). `bands` is the candidate
    band per row (gate_relabel output), index-aligned to `rows`.

    Outcome: 'inconclusive' below `min_episodes` distinct strong episodes;
    'validated' when the CI lower bound clears 0 OR the strong set has no
    losing outcomes at all (omega_inf=True — the CI machinery cannot
    express an all-win subset; >= min_episodes distinct episodes is the
    guard); else 'not_supported'."""
    r = rows.assign(_cband=bands.astype(str)).sort_values("date")
    fwd = r["fwd_252"].to_numpy(dtype=float)
    sm = (r["_cband"] == "strong").to_numpy()
    n = len(r)
    n_ep = int(r.loc[r["_cband"] == "strong", "episode_id"]
                .dropna().nunique())
    block_rows = int(math.ceil(PRIMARY_HORIZON / max(1, stride)))

    def _stat(f: np.ndarray, e: np.ndarray) -> float:
        if not e.any():
            return float("nan")
        return da.omega_ratio(f[e]) - da.omega_ratio(f)

    point = _stat(fwd, sm) if n else float("nan")
    lo = hi = float("nan")
    if n >= 2:
        rng = np.random.default_rng(seed)
        p_new = 1.0 / block_rows
        stats = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = da._stationary_bootstrap_indices(rng, n, p_new)
            stats[b] = _stat(fwd[idx], sm[idx])
        finite = stats[np.isfinite(stats)]
        if finite.size:
            q = (1.0 - ci) / 2.0
            lo = float(np.quantile(finite, q))
            hi = float(np.quantile(finite, 1.0 - q))
    om_s = da.omega_ratio(fwd[sm]) if sm.any() else float("nan")
    omega_inf = bool(math.isinf(om_s) and om_s > 0)
    if n_ep < min_episodes:
        outcome = "inconclusive"
    elif (np.isfinite(lo) and lo > 0.0) or omega_inf:
        outcome = "validated"
    else:
        outcome = "not_supported"
    return {"stat": point, "ci_lo": lo, "ci_hi": hi,
            "n_strong_days": int(sm.sum()), "n_strong_episodes": n_ep,
            "block_rows": block_rows, "omega_inf": omega_inf,
            "outcome": outcome}


GATE_SELECTION_RULE = ("first candidate by preference order whose strong "
                       "band validates (ci_lo > 0 or omega_inf, >= "
                       f"{MIN_EDGE_EPISODES} episodes); none -> incumbent "
                       "gate stays")


def _gate_record(rows: pd.DataFrame, bands: pd.Series, *, stride: int) -> dict:
    """Primary + descriptive aggregates for one gate's band assignment."""
    pm = {k: _json_num(v)
          for k, v in strong_primary_metric(rows, bands, stride=stride).items()}
    b = bands.astype(str).to_numpy()
    return {"primary": pm,
            "descriptive": {
                "strong": _json_safe_agg(_agg(rows[b == "strong"])),
                "residual_neutral": _json_safe_agg(_agg(rows[b == "neutral"])),
            }}


def build_gate_registered_artifact(rows: pd.DataFrame, *, ticker: str,
                                   stride: int, burn_in_years: int) -> dict:
    """The committed gate-experiment record (spec 2026-07-18): incumbent
    descriptive record + the three pre-declared candidates in adoption-
    preference order + the selection. Pure function of the walk-forward
    rows and run params; the registration date is the module constant.

    Integrity guard: the incumbent relabel must reproduce the stored bands
    exactly — if it cannot, the relabel seam has drifted from the shipped
    verdict cascade and no artifact may be built from these rows."""
    incumbent = {"strong_rr": da.VERDICT_STRONG_RR, "requires_ci": True}
    inc_bands = gate_relabel(rows, **incumbent)
    if not (inc_bands == rows["band"]).all():
        raise ValueError("incumbent relabel drifted from stored bands — "
                         "gate_relabel no longer mirrors dip_buy_verdict")
    cands, selected = [], None
    for c in GATE_CANDIDATES:
        bands = gate_relabel(rows, strong_rr=c["strong_rr"],
                             requires_ci=c["requires_ci"])
        rec = _gate_record(rows, bands, stride=stride)
        cands.append({"id": c["id"], "strong_rr": c["strong_rr"],
                      "requires_ci": c["requires_ci"],
                      "preference": c["preference"], **rec})
        if selected is None and rec["primary"]["outcome"] == "validated":
            selected = c["id"]
    return {
        "schema": 1,
        "kind": "gate",
        "ticker": ticker.upper(),
        "registered": GATE_REGISTERED_DATE,
        "config": {"stride": stride, "burn_in_years": burn_in_years,
                   "censor": max(REALIZED_HORIZONS),
                   "horizon": PRIMARY_HORIZON,
                   "incumbent": incumbent,
                   "min_episodes": MIN_EDGE_EPISODES},
        "evals": {"n": int(len(rows)),
                  "first": str(rows["date"].iloc[0].date()),
                  "last": str(rows["date"].iloc[-1].date())},
        "incumbent": _gate_record(rows, inc_bands, stride=stride),
        "candidates": cands,
        "selection_rule": GATE_SELECTION_RULE,
        "selected": selected,
    }


def load_gate_registered_artifact(path: "Path | None" = None) -> "dict | None":
    """Committed gate-experiment record, or None on missing / corrupt /
    drifted file. Never raises — same degrade discipline as the other
    registered artifacts."""
    p = Path(path) if path is not None else GATE_REGISTERED_PATH
    try:
        art = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(art, dict) or art.get("schema") != 1:
        return None
    if art.get("kind") != "gate":
        return None
    if not {"ticker", "registered", "config", "evals", "incumbent",
            "candidates", "selection_rule", "selected"} <= art.keys():
        return None
    if not isinstance(art["candidates"], list) or len(art["candidates"]) != 3:
        return None
    return art


# ---------------------------------------------------------------------------
# Rotation-ladder v2 re-score (spec 2026-07-18 v2 metric) — the FOURTH
# registered claim. Same mechanism and fractions as v1; a primary that can
# see a low-exposure overlay (terminal-wealth log-ratio — every day
# contributes finitely, no inf−inf). v1 code paths and artifact untouched.

LADDER_V2_REGISTERED_DATE = "2026-07-19"
LADDER_V2_REGISTERED_PATH = ROOT / "parsers" / "dip_ladder_v2_registered.json"
LADDER_V2_BLOCK_DAYS = 126            # deployment-spell dependence scale
LADDER_V2_SENS_BLOCKS = (21, 63, 252)  # recorded, NON-gating


def _wealth_log_diffs(lw: pd.Series, bw: pd.Series) -> np.ndarray:
    """Paired daily log-return differences. Day 0 is measured from the 1.0
    capital start (both strategies begin with 1.0 but their day-0 returns
    differ by the cash-leg weight), so sum(d) = ln(W_L(T)/W_B(T)) exactly."""
    lv = np.log(lw.to_numpy(dtype=float))
    bv = np.log(bw.to_numpy(dtype=float))
    return np.diff(lv, prepend=0.0) - np.diff(bv, prepend=0.0)


def ladder_wealth_metric(lw: pd.Series, bw: pd.Series, *, n_tranches: int,
                         block_days: int = LADDER_V2_BLOCK_DAYS,
                         n_boot: int = da.BOOTSTRAP_N,
                         seed: int = da.BOOTSTRAP_SEED,
                         ci: float = da.BOOTSTRAP_CI,
                         min_tranches: int = LADDER_MIN_TRANCHES) -> dict:
    """V2 pre-registered primary (spec 2026-07-18 v2 metric):
    S = ln(final ladder wealth / final baseline wealth), computed as the
    sum of paired daily log-return differences; stationary block bootstrap
    on that daily series (mean block `block_days`, seed 0, 90% CI).
    Outcome: 'inconclusive' below `min_tranches`; 'validated' iff
    ci_lo > 0; else 'not_supported'."""
    d = _wealth_log_diffs(lw, bw)
    n = d.size
    point = float(d.sum()) if n else float("nan")
    lo = hi = float("nan")
    if n >= 2:
        rng = np.random.default_rng(seed)
        p_new = 1.0 / block_days
        stats = np.empty(n_boot, dtype=float)
        for b_i in range(n_boot):
            idx = da._stationary_bootstrap_indices(rng, n, p_new)
            stats[b_i] = float(d[idx].sum())
        finite = stats[np.isfinite(stats)]
        if finite.size:
            q = (1.0 - ci) / 2.0
            lo = float(np.quantile(finite, q))
            hi = float(np.quantile(finite, 1.0 - q))
    if n_tranches < min_tranches:
        outcome = "inconclusive"
    elif np.isfinite(lo) and lo > 0.0:
        outcome = "validated"
    else:
        outcome = "not_supported"
    return {"stat": point, "ci_lo": lo, "ci_hi": hi, "n_days": int(n),
            "block_days": int(block_days), "outcome": outcome}


def ladder_deployed_mask(dates, tranches: list) -> np.ndarray:
    """Deployment flag per date: any tranche holds equity at that day's
    CLOSE — entry_date <= d < exit_date (the engine exits before it
    deploys, so proceeds sit in cash at the exit day's close); open
    tranches (exit None) deploy through the tail."""
    dd = pd.DatetimeIndex(pd.to_datetime(dates))
    out = np.zeros(len(dd), dtype=bool)
    for t in tranches:
        m = np.asarray(dd >= pd.Timestamp(t["entry_date"]))
        x = t.get("exit_date")
        if x is not None:
            m &= np.asarray(dd < pd.Timestamp(x))
        out |= m
    return out


def ladder_deployed_omega(rows: pd.DataFrame, lw: pd.Series, bw: pd.Series,
                          tranches: list, *, stride: int = 5,
                          n_boot: int = da.BOOTSTRAP_N,
                          seed: int = da.BOOTSTRAP_SEED,
                          ci: float = da.BOOTSTRAP_CI) -> dict:
    """NON-GATING descriptive (spec 2026-07-18 v2 §4): S = Omega(fwd-252
    ladder-wealth returns | deployed eval days) − Omega(same, baseline),
    block bootstrap over the restricted day-ordered rows (block =
    ceil(252/stride) rows, house seed). Recorded beside the primary with
    its weaknesses named: it conditions on the strategy's own state and
    discards the cash-drag information. Never gates the outcome."""
    if not lw.index.equals(bw.index):
        raise ValueError("wealth series must share one index")
    r = rows.sort_values("date")
    mask = ladder_deployed_mask(r["date"], tranches)
    pos = {d: i for i, d in enumerate(lw.index)}
    H = PRIMARY_HORIZON
    fl, fb = [], []
    for d in r.loc[mask, "date"]:
        i = pos.get(pd.Timestamp(d))
        if i is None or i + H >= len(lw.index):
            continue
        fl.append(float(lw.iloc[i + H] / lw.iloc[i] - 1.0))
        fb.append(float(bw.iloc[i + H] / bw.iloc[i] - 1.0))
    fl_a, fb_a = np.asarray(fl, dtype=float), np.asarray(fb, dtype=float)
    n = len(fl_a)
    block_rows = int(math.ceil(H / max(1, stride)))

    def _stat(a: np.ndarray, b: np.ndarray) -> float:
        return da.omega_ratio(a) - da.omega_ratio(b)

    point = _stat(fl_a, fb_a) if n else float("nan")
    lo = hi = float("nan")
    if n >= 2:
        rng = np.random.default_rng(seed)
        p_new = 1.0 / block_rows
        stats = np.empty(n_boot, dtype=float)
        for b_i in range(n_boot):
            idx = da._stationary_bootstrap_indices(rng, n, p_new)
            stats[b_i] = _stat(fl_a[idx], fb_a[idx])
        finite = stats[np.isfinite(stats)]
        if finite.size:
            q = (1.0 - ci) / 2.0
            lo = float(np.quantile(finite, q))
            hi = float(np.quantile(finite, 1.0 - q))
    return {"stat": point, "ci_lo": lo, "ci_hi": hi, "n_rows": n,
            "block_rows": block_rows}


LADDER_V2_PRIOR_OBSERVATION = (
    "The v1 registration (2026-07-18, all-windows Omega primary) read "
    "not_supported while final wealth showed 5.77x vs 3.78x; this v2 "
    "metric was chosen after observing that run, to fix the inf-Omega "
    "degeneracy. The same-day gate experiment selected no new gate, so "
    "the deploy pattern and this stat's point value were fully known at "
    "v2 registration time; the CI and the outcome rule's result were not "
    "computed until after the registration committed.")


def build_ladder_v2_registered_artifact(rows: pd.DataFrame, price: pd.Series,
                                        tr: pd.Series, cash_rets: pd.Series,
                                        *, ticker: str, stride: int,
                                        burn_in_years: int,
                                        fractions: dict | None = None) -> dict:
    """The committed v2 ladder-run record (spec 2026-07-18 v2 metric):
    config incl. the gate in force + the metric declaration, fractions,
    ladder summary, the prior-observation (spoilage) disclosure, the v2
    primary, block-length sensitivity (non-gating) and the deployed-only
    Omega descriptive. Pure function; v1 plumbing untouched — the baseline
    construction is deliberately duplicated, not refactored, so the frozen
    v1 claim's code is never edited."""
    fr = dict(dl.LADDER_FRACTIONS if fractions is None else fractions)
    sim = dl.simulate_ladder(price, tr, cash_rets, ladder_evals(rows),
                             fractions=fr)
    w = float(sim["summary"]["avg_equity_exposure"])
    eq_ret = tr.pct_change().fillna(0.0)
    cash_r = (cash_rets.reindex(tr.index).fillna(0.0)
              if cash_rets is not None and not cash_rets.empty
              else pd.Series(0.0, index=tr.index))
    bw = (1.0 + w * eq_ret + (1.0 - w) * cash_r).cumprod()
    lw = sim["wealth"]
    nt = int(sim["summary"]["n_tranches"])
    pm = ladder_wealth_metric(lw, bw, n_tranches=nt)
    sens = {}
    for blk in LADDER_V2_SENS_BLOCKS:
        s = ladder_wealth_metric(lw, bw, n_tranches=nt, block_days=blk)
        sens[f"block_{blk}"] = {"ci_lo": _json_num(s["ci_lo"]),
                                "ci_hi": _json_num(s["ci_hi"])}
    dep = ladder_deployed_omega(rows, lw, bw, sim["tranches"], stride=stride)
    return {
        "schema": 1,
        "kind": "ladder_v2",
        "ticker": ticker.upper(),
        "registered": LADDER_V2_REGISTERED_DATE,
        "config": {"stride": stride, "burn_in_years": burn_in_years,
                   "censor": max(REALIZED_HORIZONS),
                   "horizon": PRIMARY_HORIZON,
                   "cash_leg": "ff_rf_daily",
                   "exit": "recovery_to_anchor_peak",
                   "min_tranches": LADDER_MIN_TRANCHES,
                   "gate": {"strong_rr": da.VERDICT_STRONG_RR,
                            "requires_ci": True},
                   "metric": {"primary": "terminal_wealth_log_ratio",
                              "block_days": LADDER_V2_BLOCK_DAYS}},
        "fractions": fr,
        "evals": {"n": int(len(rows)),
                  "first": str(rows["date"].iloc[0].date()),
                  "last": str(rows["date"].iloc[-1].date())},
        "ladder": {"n_tranches": nt,
                   "skipped_deploys": int(sim["summary"]["skipped_deploys"]),
                   "avg_equity_exposure": _json_num(w),
                   "final_wealth": _json_num(
                       float(sim["summary"]["final_wealth"])),
                   "baseline_final_wealth": _json_num(float(bw.iloc[-1]))},
        "prior_observation": LADDER_V2_PRIOR_OBSERVATION,
        "primary": {k: _json_num(v) for k, v in pm.items()},
        "sensitivity": sens,
        "descriptive": {"deployed_omega":
                        {k: _json_num(v) for k, v in dep.items()}},
    }


def load_ladder_v2_registered_artifact(path: "Path | None" = None) \
        -> "dict | None":
    """Committed v2 ladder-run record, or None on missing / corrupt /
    drifted file (a v1 `kind: ladder` file must NOT load as v2). Never
    raises — same degrade discipline as the other registered artifacts."""
    p = Path(path) if path is not None else LADDER_V2_REGISTERED_PATH
    try:
        art = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(art, dict) or art.get("schema") != 1:
        return None
    if art.get("kind") != "ladder_v2":
        return None
    if not {"ticker", "registered", "config", "fractions", "evals",
            "ladder", "prior_observation", "primary", "sensitivity",
            "descriptive"} <= art.keys():
        return None
    return art


# ---------------------------------------------------------------------------
# SPX index-history extension experiment (spec 2026-07-19) — the FIFTH
# registered artifact: the incumbent verdict replay + the #274 gate
# candidates re-scored on a 1950+ spliced index series. Consequence:
# artifact only.

SPX_REGISTERED_DATE = "2026-07-19"
SPX_REGISTERED_PATH = ROOT / "parsers" / "dip_spx_registered.json"
INCUMBENT_WINDOW = ("2003-01-29", "2025-06-23")   # the S4 eval window
SPX_CONSEQUENCE = ("artifact only — no UI or gate change ships from this "
                   "registration (spec 2026-07-19 locked decision 3)")


def _json_safe_gates(gates: dict) -> dict:
    return {k: ({kk: _json_num(vv) for kk, vv in v.items()}
                if isinstance(v, dict) else v)
            for k, v in gates.items()}


def build_spx_registered_artifact(rows: pd.DataFrame,
                                  rows_burn5: pd.DataFrame, *, stride: int,
                                  burn_in_years: int, series_meta: dict,
                                  data_gates: dict) -> dict:
    """The committed SPX-extension record (spec 2026-07-19 §4f): series
    provenance + data gates, the replay claim (primary + referee table +
    TK-rule row + incumbent-window descriptive), the burn-in-5y
    sensitivity, and the gate claim (#274 candidates re-scored). Pure
    function of the walk-forward rows and params; the registration date is
    the module constant. Same incumbent-relabel integrity guard as the gate
    artifact."""
    incumbent = {"strong_rr": da.VERDICT_STRONG_RR, "requires_ci": True}
    inc_bands = gate_relabel(rows, **incumbent)
    if not (inc_bands == rows["band"]).all():
        raise ValueError("incumbent relabel drifted from stored bands — "
                         "gate_relabel no longer mirrors dip_buy_verdict")
    ref = {band: _json_safe_agg(_agg(rows[rows["band"] == band]))
           for band in BAND_ORDER}
    ref["all"] = _json_safe_agg(_agg(rows))
    pm = {k: _json_num(v)
          for k, v in primary_metric(rows, stride=stride).items()}
    lo_d = pd.Timestamp(INCUMBENT_WINDOW[0])
    hi_d = pd.Timestamp(INCUMBENT_WINDOW[1])
    riw = rows[(rows["date"] >= lo_d) & (rows["date"] <= hi_d)]
    pm_iw = ({k: _json_num(v)
              for k, v in primary_metric(riw, stride=stride).items()}
             if len(riw) else None)
    pm5 = {k: _json_num(v)
           for k, v in primary_metric(rows_burn5, stride=stride).items()}
    cands, selected = [], None
    for c in GATE_CANDIDATES:
        bands = gate_relabel(rows, strong_rr=c["strong_rr"],
                             requires_ci=c["requires_ci"])
        rec = _gate_record(rows, bands, stride=stride)
        cands.append({"id": c["id"], "strong_rr": c["strong_rr"],
                      "requires_ci": c["requires_ci"],
                      "preference": c["preference"], **rec})
        if selected is None and rec["primary"]["outcome"] == "validated":
            selected = c["id"]
    tk = rows[(rows["depth_pctile"] >= DEPTH_TK_RULE)
              & rows["band"].isin(EDGE_BANDS)]
    return {
        "schema": 1,
        "kind": "spx_extension",
        "ticker": "SPX",
        "registered": SPX_REGISTERED_DATE,
        "series": {"meta": series_meta,
                   "data_gates": _json_safe_gates(data_gates)},
        "config": {"stride": stride, "burn_in_years": burn_in_years,
                   "censor": max(REALIZED_HORIZONS),
                   "horizon": PRIMARY_HORIZON,
                   "incumbent": incumbent,
                   "min_episodes": MIN_EDGE_EPISODES,
                   "incumbent_window": list(INCUMBENT_WINDOW)},
        "evals": {"n": int(len(rows)),
                  "first": str(rows["date"].iloc[0].date()),
                  "last": str(rows["date"].iloc[-1].date())},
        "replay": {"primary": pm, "referee": ref,
                   "tk_rule": _json_safe_agg(_agg(tk)),
                   "incumbent_window": pm_iw},
        "sensitivity": {"burn_in_5y": {
            "evals": {"n": int(len(rows_burn5)),
                      "first": str(rows_burn5["date"].iloc[0].date()),
                      "last": str(rows_burn5["date"].iloc[-1].date())},
            "primary": pm5}},
        "gate": {"incumbent": _gate_record(rows, inc_bands, stride=stride),
                 "candidates": cands,
                 "selection_rule": GATE_SELECTION_RULE,
                 "selected": selected},
        "consequence": SPX_CONSEQUENCE,
    }


def load_spx_registered_artifact(path: "Path | None" = None) -> "dict | None":
    """Committed SPX-extension record, or None on missing / corrupt /
    drifted file. Never raises — same degrade discipline as the other
    registered artifacts."""
    p = Path(path) if path is not None else SPX_REGISTERED_PATH
    try:
        art = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(art, dict) or art.get("schema") != 1:
        return None
    if art.get("kind") != "spx_extension":
        return None
    if not {"ticker", "registered", "series", "config", "evals", "replay",
            "gate", "sensitivity", "consequence"} <= art.keys():
        return None
    g = art["gate"]
    if not isinstance(g, dict) or not isinstance(g.get("candidates"), list) \
            or len(g["candidates"]) != 3:
        return None
    return art


def _print_block(title: str, frame: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    print(frame.to_string(float_format=lambda v: f"{v: .4f}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Walk-forward referee for the Buy-the-Dip verdict")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--stride", type=int, default=5,
                    help="evaluate every Nth eligible trading day (default 5)")
    ap.add_argument("--burn-in-years", type=int, default=10,
                    help="history required before the first evaluation")
    ap.add_argument("--data-dir", default=None,
                    help="defaults to $APP_DATA_DIR or <repo>/data")
    ap.add_argument("--write", action="store_true",
                    help="dump eval rows to <data-dir>/dip_backtest_<T>.csv")
    ap.add_argument("--write-registered", action="store_true",
                    help="write the registered-run artifact (committed at "
                         "parsers/dip_backtest_registered.json) — a "
                         "change-control event, see spec 2026-07-16")
    ap.add_argument("--write-ladder-registered", action="store_true",
                    help="replay the rotation ladder and write its "
                         "registered artifact (parsers/"
                         "dip_ladder_registered.json) — SPY-only, a "
                         "change-control event, see spec 2026-07-18")
    ap.add_argument("--write-gate-registered", action="store_true",
                    help="score the strong-gate candidates and write the "
                         "registered artifact (parsers/"
                         "dip_gate_registered.json) — SPY-only, a "
                         "change-control event, see spec 2026-07-18 "
                         "strong-gate recalibration")
    ap.add_argument("--write-ladder-v2-registered", action="store_true",
                    help="re-score the rotation ladder with the v2 "
                         "terminal-wealth log-ratio primary and write "
                         "parsers/dip_ladder_v2_registered.json — "
                         "SPY-only, a change-control event, see spec "
                         "2026-07-18 ladder v2 metric")
    ap.add_argument("--extended-spx", action="store_true",
                    help="run on the SPX extended index series (SPY + "
                         "^GSPC/^SP500TR/synthetic splice, spec 2026-07-19) "
                         "— requires --ticker SPX")
    ap.add_argument("--write-spx-registered", action="store_true",
                    help="write the SPX-extension registered artifact "
                         "(parsers/dip_spx_registered.json) — requires "
                         "--extended-spx; a change-control event, see spec "
                         "2026-07-19")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir or os.environ.get("APP_DATA_DIR")
                    or (ROOT / "data"))
    t = args.ticker.upper()
    # Every registered-writer refusal is an ARGUMENT check: it runs here,
    # before any data load / walk-forward / bootstrap, so a mistaken run
    # fails in milliseconds instead of after a full referee replay. Order is
    # the historical precedence (spx-mode checks first).
    if args.write_spx_registered and not args.extended_spx:
        print("ERROR: --write-spx-registered requires --extended-spx "
              "(spec 2026-07-19 §4d)", file=sys.stderr)
        return 1
    if args.extended_spx and t != "SPX":
        print("ERROR: --extended-spx requires --ticker SPX (spec "
              "2026-07-19 §4d)", file=sys.stderr)
        return 1
    if args.write_registered and (t != "SPY" or args.extended_spx):
        print("ERROR: the registered verdict artifact is SPY-only "
              "(spec 2026-07-16); --extended-spx is refused here so a "
              "non-SPY run cannot overwrite the incumbent",
              file=sys.stderr)
        return 1
    if args.write_ladder_registered and t != "SPY":
        print("ERROR: the registered ladder call is SPY-only (spec "
              "2026-07-18 §6)", file=sys.stderr)
        return 1
    if args.write_gate_registered and t != "SPY":
        print("ERROR: the registered gate experiment is SPY-only (spec "
              "2026-07-18 strong-gate recalibration §8)", file=sys.stderr)
        return 1
    if args.write_ladder_v2_registered and t != "SPY":
        print("ERROR: the registered ladder v2 re-score is SPY-only "
              "(spec 2026-07-18 ladder v2 metric §8)", file=sys.stderr)
        return 1
    ext = None
    try:
        if args.extended_spx:
            ext = de.load_extended_spx(data_dir)
            price, tr = ext["price"], ext["tr"]
        else:
            price, tr = load_dip_series(data_dir, args.ticker)
        rows = walk_forward(price, tr, stride=args.stride,
                            burn_in_years=args.burn_in_years)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"dip-backtest {t}: {len(rows)} evaluations "
          f"{rows['date'].iloc[0].date()} -> {rows['date'].iloc[-1].date()} "
          f"(stride {args.stride}, burn-in {args.burn_in_years}y, "
          f"censor {max(REALIZED_HORIZONS)}d)")
    _print_block("referee table (per walk-forward band)", referee_table(rows))
    _print_block("TK rule (depth >= 85 pctile AND edge-claimed; descriptive)",
                 tk_rule_row(rows))
    _print_block("depth-decile curve (descriptive)", depth_decile_curve(rows))
    pm = primary_metric(rows, stride=args.stride)
    if args.extended_spx:
        tag = " [SPX extended-index experiment — spec 2026-07-19]"
    else:
        tag = "" if t == "SPY" else \
            " [descriptive-only: the registered call is SPY]"
    print(f"\n== primary metric{tag} ==")
    for k, v in pm.items():
        print(f"  {k}: {v}")
    if args.write:
        out_csv = Path(data_dir) / f"dip_backtest_{t}.csv"
        rows.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
    if args.write_registered:
        art = build_registered_artifact(rows, ticker=t, stride=args.stride,
                                        burn_in_years=args.burn_in_years)
        REGISTERED_PATH.write_bytes(
            (json.dumps(art, indent=2) + "\n").encode("ascii"))
        print(f"\nwrote {REGISTERED_PATH}")
    if args.write_ladder_registered:
        try:
            cash = load_cash_rets(data_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        lart = build_ladder_registered_artifact(
            rows, price, tr, cash, ticker=t, stride=args.stride,
            burn_in_years=args.burn_in_years)
        print("\n== rotation ladder (registered replay) ==")
        for k, v in {**lart["ladder"], **lart["primary"]}.items():
            print(f"  {k}: {v}")
        LADDER_REGISTERED_PATH.write_bytes(
            (json.dumps(lart, indent=2) + "\n").encode("ascii"))
        print(f"\nwrote {LADDER_REGISTERED_PATH}")
    if args.write_gate_registered:
        gart = build_gate_registered_artifact(rows, ticker=t,
                                              stride=args.stride,
                                              burn_in_years=args.burn_in_years)
        print("\n== strong-gate experiment (registered run) ==")
        for c in gart["candidates"]:
            pm = c["primary"]
            print(f"  {c['id']} (rr>={c['strong_rr']}, "
                  f"{'CI' if c['requires_ci'] else 'point'}): "
                  f"days={pm['n_strong_days']} eps={pm['n_strong_episodes']} "
                  f"stat={pm['stat']} ci=[{pm['ci_lo']}, {pm['ci_hi']}] "
                  f"omega_inf={pm['omega_inf']} -> {pm['outcome']}")
        print(f"  selected: {gart['selected']}")
        GATE_REGISTERED_PATH.write_bytes(
            (json.dumps(gart, indent=2) + "\n").encode("ascii"))
        print(f"\nwrote {GATE_REGISTERED_PATH}")
    if args.write_ladder_v2_registered:
        try:
            cash = load_cash_rets(data_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        v2 = build_ladder_v2_registered_artifact(
            rows, price, tr, cash, ticker=t, stride=args.stride,
            burn_in_years=args.burn_in_years)
        print("\n== rotation ladder v2 (registered re-score) ==")
        for k, v in {**v2["ladder"], **v2["primary"]}.items():
            print(f"  {k}: {v}")
        for blk, s in v2["sensitivity"].items():
            print(f"  sensitivity {blk}: ci=[{s['ci_lo']}, {s['ci_hi']}]")
        dep = v2["descriptive"]["deployed_omega"]
        print(f"  deployed_omega: stat={dep['stat']} "
              f"ci=[{dep['ci_lo']}, {dep['ci_hi']}] n_rows={dep['n_rows']}")
        LADDER_V2_REGISTERED_PATH.write_bytes(
            (json.dumps(v2, indent=2) + "\n").encode("ascii"))
        print(f"\nwrote {LADDER_V2_REGISTERED_PATH}")
    if args.write_spx_registered:
        gates = de.run_data_gates(ext["components"])
        print("\n== SPX data-quality gates (spec 2026-07-19 §4c) ==")
        for k, v in gates.items():
            print(f"  {k}: {v}")
        if not gates["all_ok"]:
            print("ERROR: data-quality gates failed — the registered run "
                  "is BLOCKED until the data layer is fixed (spec §4c)",
                  file=sys.stderr)
            return 1
        rows5 = walk_forward(price, tr, stride=args.stride, burn_in_years=5)
        art = build_spx_registered_artifact(
            rows, rows5, stride=args.stride,
            burn_in_years=args.burn_in_years, series_meta=ext["meta"],
            data_gates=gates)
        print("\n== SPX extension (registered run) ==")
        print(f"  replay: {art['replay']['primary']}")
        for c in art["gate"]["candidates"]:
            pm_c = c["primary"]
            print(f"  {c['id']}: days={pm_c['n_strong_days']} "
                  f"eps={pm_c['n_strong_episodes']} stat={pm_c['stat']} "
                  f"ci=[{pm_c['ci_lo']}, {pm_c['ci_hi']}] "
                  f"-> {pm_c['outcome']}")
        print(f"  selected: {art['gate']['selected']}")
        SPX_REGISTERED_PATH.write_bytes(
            (json.dumps(art, indent=2) + "\n").encode("ascii"))
        print(f"\nwrote {SPX_REGISTERED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
