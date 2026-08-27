"""Robustness analyses for the Phase F exit-rule back-test.

Two layers of analysis on top of ``hedge_exit_simulator.simulate_program``:

1. **Parameter sensitivity sweep** — vary one rule-tuning knob at a time
   (DTE-close threshold for dte_roll, recovery_frac for monetize,
   profit-take multiplier for profit_take_3x) and chart drag vs payoff.
   Output: per-rule curves so the user can see where the Pareto-optimal
   parameter lives instead of guessing one number.

2. **Walk-forward window panel** — slice the back-test history into N
   overlapping 1-year windows shifted by ~2 months; re-run each rule on
   each window. Output: distribution of (drag, payoff) per rule across
   windows. With only ~2y of real data and 5 drawdowns total, a single
   point estimate misrepresents uncertainty; a small panel of 1y windows
   makes the regime-dependence visible (does the rule still rank #1 in
   every window, or only when the Feb-Apr 2025 drop is included?).

Why not also a full Monte-Carlo bootstrap with option repricing
--------------------------------------------------------------
True path-resampling bootstrap requires regenerating option prices on
each resampled SPY path, which means coupling to the Phase A pricer
(BSM/LR) + an IV process. That's a bigger lift; defer until walk-forward
either fails to give a clear picture or the user explicitly asks for it.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from hedge_effectiveness import find_drawdown_episodes
from hedge_exit_simulator import (
    EXIT_RULES,
    HedgePolicy,
    compare_runs,
    simulate_program,
)


# --- Parameter sweep --------------------------------------------------------

def sweep_parameter(
    rule_name: str, param_name: str, values: list,
    *, policy: HedgePolicy,
    spy_history: pd.DataFrame, option_grid: pd.DataFrame,
    start: date, end: date, dd_threshold_pct: float = 3.0,
    other_kwargs: Optional[dict] = None,
    iv_rank_series: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Re-run the simulator for ``rule_name`` across ``values`` of one
    parameter, holding everything else fixed.

    Returns a DataFrame with columns:
        param_value, drag_pct, sum_payoff_pct, mean_payoff_pct,
        payoff_per_dollar_drag, n_trades

    Use to chart drag vs payoff as a function of one rule knob.
    """
    if rule_name not in EXIT_RULES:
        raise ValueError(f"Unknown rule {rule_name!r}")
    base_kwargs = dict(other_kwargs or {})
    episodes = find_drawdown_episodes(
        spy_history, threshold_pct=dd_threshold_pct,
        start_date=start, end_date=end,
    )

    rows: list[dict] = []
    for v in values:
        kwargs = dict(base_kwargs)
        kwargs[param_name] = v
        ledger, legs = simulate_program(
            policy, rule_name, spy_history, option_grid,
            start=start, end=end, rule_kwargs=kwargs,
            iv_rank_series=iv_rank_series,
        )
        cmp = compare_runs({rule_name: (ledger, legs)}, episodes, policy)
        r = cmp.iloc[0]
        rows.append({
            "param_value": v,
            "drag_pct": r["annualized_drag_pct"],
            "sum_payoff_pct": r["total_episode_payoff_pct"],
            "mean_payoff_pct": r["mean_episode_payoff_pct"],
            "payoff_per_dollar_drag": r["payoff_per_dollar_drag"],
            "n_trades": r["n_trades"],
        })
    return pd.DataFrame(rows)


# --- Walk-forward windows ---------------------------------------------------

def walk_forward_windows(
    start: date, end: date,
    *, window_days: int = 365, stride_days: int = 60,
) -> list[tuple[date, date]]:
    """Generate overlapping (window_start, window_end) pairs.

    Each window is ``window_days`` long; consecutive windows are offset
    by ``stride_days``. Last window ends ≤ ``end``.

    Example: start=2024-05-25, end=2026-05-25, window=365, stride=60
        → windows starting Jun-2024, Aug-2024, Oct-2024, …, Apr-2025
          (=11 windows, each 1 year long).
    """
    out: list[tuple[date, date]] = []
    cur = start
    while True:
        w_end = cur + timedelta(days=window_days)
        if w_end > end:
            break
        out.append((cur, w_end))
        cur = cur + timedelta(days=stride_days)
    return out


def run_walk_forward(
    rule_name: str, policy: HedgePolicy,
    spy_history: pd.DataFrame, option_grid: pd.DataFrame,
    *, start: date, end: date,
    window_days: int = 365, stride_days: int = 60,
    dd_threshold_pct: float = 3.0,
    rule_kwargs: Optional[dict] = None,
    iv_rank_series: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run one rule across overlapping 1-year windows.

    Returns DataFrame with one row per window, columns:
        window_start, window_end, drag_pct, sum_payoff_pct,
        mean_payoff_pct, payoff_per_dollar_drag, n_trades, n_episodes
    """
    windows = walk_forward_windows(
        start, end, window_days=window_days, stride_days=stride_days,
    )
    rows: list[dict] = []
    for w_start, w_end in windows:
        eps = find_drawdown_episodes(
            spy_history, threshold_pct=dd_threshold_pct,
            start_date=w_start, end_date=w_end,
        )
        ledger, legs = simulate_program(
            policy, rule_name, spy_history, option_grid,
            start=w_start, end=w_end, rule_kwargs=rule_kwargs,
            iv_rank_series=iv_rank_series,
        )
        cmp = compare_runs({rule_name: (ledger, legs)}, eps, policy)
        r = cmp.iloc[0]
        rows.append({
            "window_start": w_start, "window_end": w_end,
            "drag_pct": r["annualized_drag_pct"],
            "sum_payoff_pct": r["total_episode_payoff_pct"],
            "mean_payoff_pct": r["mean_episode_payoff_pct"],
            "payoff_per_dollar_drag": r["payoff_per_dollar_drag"],
            "n_trades": r["n_trades"],
            "n_episodes": int(len(eps)),
        })
    return pd.DataFrame(rows)


def walk_forward_compare_all(
    policy: HedgePolicy,
    spy_history: pd.DataFrame, option_grid: pd.DataFrame,
    *, start: date, end: date,
    window_days: int = 365, stride_days: int = 60,
    dd_threshold_pct: float = 3.0,
    rule_kwargs_by_rule: Optional[dict[str, dict]] = None,
    iv_rank_series: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convenience: walk-forward all 4 rules, return a long-format frame
    [rule, window_start, …] suitable for groupby aggregation.
    """
    rule_kwargs_by_rule = rule_kwargs_by_rule or {}
    out = []
    for rule_name in EXIT_RULES:
        sub = run_walk_forward(
            rule_name, policy, spy_history, option_grid,
            start=start, end=end,
            window_days=window_days, stride_days=stride_days,
            dd_threshold_pct=dd_threshold_pct,
            rule_kwargs=rule_kwargs_by_rule.get(rule_name),
            iv_rank_series=iv_rank_series,
        )
        sub.insert(0, "rule", rule_name)
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pick_optimal_param(
    sweep_df: pd.DataFrame,
    *, criterion: str = "payoff_per_dollar_drag",
    require_positive_drag: bool = True,
) -> dict:
    """Pick the param_value that maximizes ``criterion`` across a sweep.

    Use to replace fiat-chosen knob values (e.g. profit_take's 3× default)
    with the empirically best value over the back-test window.

    Arguments
    ---------
    sweep_df : DataFrame produced by :func:`sweep_parameter`.
    criterion : column to maximize. Default ``payoff_per_dollar_drag`` —
        the comparison-table headline ratio.
    require_positive_drag : drop rows where the program net-printed money
        (drag ≤ 0) before picking, since pay/$drag flips sign there and
        becomes meaningless as an objective. Falls back to all rows if
        every row has non-positive drag.

    Returns
    -------
    {
        "optimal_value": param value at the winning row,
        "optimal_row": full sweep row (pd.Series),
        "n_candidates": number of rows considered after the drag filter,
    }
    """
    if sweep_df.empty:
        raise ValueError("Cannot pick optimum from empty sweep")
    if criterion not in sweep_df.columns:
        raise ValueError(
            f"criterion {criterion!r} not in sweep columns "
            f"{list(sweep_df.columns)}"
        )
    candidates = sweep_df
    if require_positive_drag and "drag_pct" in sweep_df.columns:
        positive = sweep_df[sweep_df["drag_pct"] > 0]
        if not positive.empty:
            candidates = positive
    candidates = candidates.dropna(subset=[criterion])
    if candidates.empty:
        raise ValueError(
            f"No valid rows for criterion {criterion!r} after filters"
        )
    best_idx = candidates[criterion].idxmax()
    return {
        "optimal_value": sweep_df.loc[best_idx, "param_value"],
        "optimal_row": sweep_df.loc[best_idx],
        "n_candidates": int(len(candidates)),
    }


def summarize_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a walk-forward-all-rules frame into per-rule median +
    10/90 percentile across windows.
    """
    if df.empty:
        return df
    g = df.groupby("rule")
    rows: list[dict] = []
    for rule, sub in g:
        rows.append({
            "rule": rule,
            "n_windows": len(sub),
            "drag_median": float(sub["drag_pct"].median()),
            "drag_p10": float(sub["drag_pct"].quantile(0.10)),
            "drag_p90": float(sub["drag_pct"].quantile(0.90)),
            "payoff_median": float(sub["sum_payoff_pct"].median()),
            "payoff_p10": float(sub["sum_payoff_pct"].quantile(0.10)),
            "payoff_p90": float(sub["sum_payoff_pct"].quantile(0.90)),
            "pay_per_drag_median": float(sub["payoff_per_dollar_drag"].median()),
        })
    return pd.DataFrame(rows)
