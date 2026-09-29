"""Cap-sweep tracer: vol vs concentration across both optimizers.

Sweeps a ladder of per-name caps, runs the existing min-variance and
risk-parity suggest functions at each cap, and returns one
(vol, effective-N) point per (cap, optimizer) so the UI can chart the
de-concentration tradeoff. Pure orchestration — all solver math lives in
min_variance / risk_parity, so every point matches what that optimizer's
Suggest banner would say at that cap.

Run tests from phase1_build/ with:
    py -m unittest tests.test_opt_curve
"""
import numpy as np
import pandas as pd

from min_variance import build_covariance, suggest_min_variance_grid
from risk_parity import suggest_risk_parity_grid

POINT_COLUMNS = ["cap", "optimizer", "vol", "effective_n", "max_weight",
                 "converged", "weights"]


def _concentration(new_pct: dict) -> tuple[float, float]:
    """(effective_n, max_weight) of a full-book percent allocation.

    Single shared copy (frontier.py imports it). NaN-hard: degenerate or
    non-finite allocations fail soft to (nan, nan), never raise.
    """
    w = np.array(list(new_pct.values()), dtype=float) / 100.0
    s = float(w.sum())
    if w.size == 0 or not np.isfinite(s) or s <= 0.0:
        return float("nan"), float("nan")
    w = w / s
    return float(1.0 / (w * w).sum()), float(w.max())


def trace_cap_curve(daily_prices: pd.DataFrame, weights: pd.Series,
                    class_of: dict[str, str], *,
                    caps: list[float], class_floors: dict[str, float],
                    class_caps: dict[str, float] | None = None,
                    class_risk_budgets: dict[str, float] | None = None,
                    min_overlap_days: int = 252, on_point=None) -> dict:
    """Run both optimizers at every cap in `caps` (decimals in (0, 1]).

    Duplicates in `caps` are collapsed and the sweep runs ascending.
    Min-variance is traced with `class_floors`; risk-parity is floor-less
    (as shipped). Both optimizers are traced with `class_caps` (bucket ->
    max class weight); risk-parity enforces them by group pinning (O1b).
    class_risk_budgets threads to risk-parity only (budgets are meaningless
    for the min-variance objective). Individually infeasible caps land in
    `skipped` as (cap, optimizer, message) — `error` is set only when Σ
    itself cannot be built. `on_point(done, total)` is called after each
    solve (UI progress hook). All outputs are decimals; the UI formats to %.

    Returns {points: DataFrame[cap, optimizer, vol, effective_n,
    max_weight, converged, weights], current: dict|None, skipped:
    list[tuple], error: str|None}. weights is the solve's full-book percent
    allocation (new_pct verbatim) so consumers can apply a traced point
    without re-solving.
    """
    empty = pd.DataFrame(columns=POINT_COLUMNS)
    covres = build_covariance(daily_prices, list(weights.index),
                              min_overlap_days=min_overlap_days)
    if covres["error"]:
        return {"points": empty, "current": None, "skipped": [],
                "error": covres["error"]}

    # Current-book star (same formulas the suggests use for cur_vol).
    s_mat = covres["cov"].to_numpy(dtype=float)
    w_cur = (weights.reindex(covres["cov"].index).fillna(0.0)
             .to_numpy(dtype=float))
    cur_vol = float(np.sqrt(max(0.0, float(w_cur @ s_mat @ w_cur))))
    cur_en, cur_max = _concentration(
        {s: float(weights[s]) * 100.0 for s in weights.index})
    current = {"vol": cur_vol, "effective_n": cur_en,
               "max_weight": cur_max}

    runs = [
        ("min_variance", lambda cap: suggest_min_variance_grid(
            daily_prices, weights, class_of, name_cap=cap,
            class_floors=class_floors, class_caps=class_caps,
            min_overlap_days=min_overlap_days, covres=covres)),
        ("risk_parity", lambda cap: suggest_risk_parity_grid(
            daily_prices, weights, name_cap=cap, class_of=class_of,
            class_caps=class_caps, class_risk_budgets=class_risk_budgets,
            min_overlap_days=min_overlap_days, covres=covres)),
    ]
    caps_sorted = sorted({float(c) for c in caps})
    rows: list[dict] = []
    skipped: list[tuple[float, str, str]] = []
    total = len(caps_sorted) * len(runs)
    done = 0
    for cap in caps_sorted:
        for opt_name, run in runs:
            out = run(cap)
            done += 1
            if out["kind"] == "error":
                skipped.append((cap, opt_name, out["message"]))
            else:
                en, mx = _concentration(out["new_pct"])
                rows.append({"cap": cap, "optimizer": opt_name,
                             "vol": float(out["vol"]),
                             "effective_n": en, "max_weight": mx,
                             "converged": bool(out["converged"]),
                             "weights": {str(k): float(v)
                                         for k, v in out["new_pct"].items()}})
            if on_point is not None:
                on_point(done, total)
    points = (pd.DataFrame(rows, columns=POINT_COLUMNS) if rows else empty)
    return {"points": points, "current": current, "skipped": skipped,
            "error": None}
