"""Robustness runner: parameter sweep + walk-forward windows for Phase F.

Two analyses on the back-test:

* ``--sweep`` runs a single-rule sensitivity sweep across one parameter,
  prints a per-value table (drag, payoff, pay/$drag). Defaults are sensible
  ranges for each tunable parameter.

* ``--walk-forward`` slices history into overlapping 1-year windows and
  re-runs all 4 rules on each, printing per-rule (median, p10, p90)
  across windows. With ~2y of data and 60-day stride, that's ~11 windows
  per rule — small but enough to see ranking stability.

Run:
  py parsers/run_hedge_robustness.py --sweep dte_roll dte_threshold "15,30,45,60"
  py parsers/run_hedge_robustness.py --sweep monetize recovery_frac "0.20,0.33,0.50,0.66"
  py parsers/run_hedge_robustness.py --sweep profit_take_3x mult "1.5,2.0,3.0,5.0"
  py parsers/run_hedge_robustness.py --walk-forward
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from hedge_exit_simulator import HedgePolicy  # noqa: E402
from hedge_robustness import (  # noqa: E402
    summarize_walk_forward,
    sweep_parameter,
    walk_forward_compare_all,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"


def _load_data():
    spy = pd.read_csv(DATA / "daily_prices.csv", parse_dates=["date"])
    spy = spy[spy["symbol"] == "SPY"][["date", "close"]].sort_values("date").reset_index(drop=True)
    grid = pd.read_csv(DATA / "option_grid_history.csv",
                       parse_dates=["date", "expiry", "fetched_at"])
    return spy, grid


def _print_sweep(rule: str, param: str, df: pd.DataFrame) -> None:
    fmtd = df.copy()
    for c in ("drag_pct", "sum_payoff_pct", "mean_payoff_pct"):
        fmtd[c] = fmtd[c].map(lambda x: f"{x:+.2f}")
    fmtd["payoff_per_dollar_drag"] = fmtd["payoff_per_dollar_drag"].map(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a"
    )
    fmtd.columns = [param, "drag%/yr", "sum_pay%", "mean_pay%", "pay/$drag", "n_trades"]
    print(f"\n=== {rule}: sensitivity to {param} ===")
    print(fmtd.to_string(index=False))


def _print_walk_forward(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    # Per-window detail (compact).
    if not df.empty:
        det = df.copy()
        det["window"] = det["window_start"].astype(str) + " → " + det["window_end"].astype(str)
        det = det[["rule", "window", "drag_pct", "sum_payoff_pct",
                   "payoff_per_dollar_drag", "n_episodes"]]
        det["drag_pct"] = det["drag_pct"].map(lambda x: f"{x:+.2f}")
        det["sum_payoff_pct"] = det["sum_payoff_pct"].map(lambda x: f"{x:+.2f}")
        det["payoff_per_dollar_drag"] = det["payoff_per_dollar_drag"].map(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a"
        )
        det.columns = ["rule", "window", "drag%/yr", "sum_pay%", "pay/$drag", "#eps"]
        print("\n=== Walk-forward: per-window per-rule ===")
        print(det.to_string(index=False))

    # Summary.
    print("\n=== Walk-forward: per-rule across all windows ===")
    s = summary.copy()
    for c in ("drag_median", "drag_p10", "drag_p90",
              "payoff_median", "payoff_p10", "payoff_p90",
              "pay_per_drag_median"):
        s[c] = s[c].map(lambda x: f"{x:+.2f}")
    s.columns = ["rule", "n_win",
                 "drag_med", "drag_p10", "drag_p90",
                 "pay_med", "pay_p10", "pay_p90", "pay/$drag_med"]
    print(s.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--dte", type=int, default=90)
    p.add_argument("--moneyness", type=float, default=0.05)
    p.add_argument("--notional", type=float, default=2_980_000.0)
    p.add_argument("--dd-threshold", type=float, default=3.0)
    p.add_argument("--sweep", nargs=3, metavar=("RULE", "PARAM", "VALUES"),
                   help="Sweep PARAM across comma-separated VALUES on RULE")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run all 4 rules across overlapping 1y windows")
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--stride-days", type=int, default=60)
    p.add_argument("--quiet", action="store_true",
                   help="Skip per-window detail in walk-forward output")
    args = p.parse_args(argv)

    spy, grid = _load_data()
    if spy.empty or grid.empty:
        print("[!] missing daily_prices.csv or option_grid_history.csv")
        return 1
    today = date.today()
    end = date.fromisoformat(args.end) if args.end else today
    start = date.fromisoformat(args.start) if args.start else (end - timedelta(days=730))
    policy = HedgePolicy(
        target_dte=args.dte, target_moneyness=args.moneyness,
        notional_protected=args.notional,
    )

    print(f"Window  : {start} → {end}")
    print(f"Policy  : SPY puts, ~{args.dte}d DTE, ~{args.moneyness:.1%} OTM, "
          f"${args.notional:,.0f} notional")

    if args.sweep:
        rule, param, vals_str = args.sweep
        # Parse values: try float first, then int.
        raw_vals = [v.strip() for v in vals_str.split(",") if v.strip()]
        try:
            vals = [int(v) for v in raw_vals]
        except ValueError:
            vals = [float(v) for v in raw_vals]
        df = sweep_parameter(
            rule, param, vals,
            policy=policy, spy_history=spy, option_grid=grid,
            start=start, end=end, dd_threshold_pct=args.dd_threshold,
        )
        _print_sweep(rule, param, df)

    if args.walk_forward:
        df = walk_forward_compare_all(
            policy, spy, grid, start=start, end=end,
            window_days=args.window_days, stride_days=args.stride_days,
            dd_threshold_pct=args.dd_threshold,
        )
        summary = summarize_walk_forward(df)
        if args.quiet:
            _print_walk_forward(pd.DataFrame(), summary)
        else:
            _print_walk_forward(df, summary)

    if not (args.sweep or args.walk_forward):
        print("\n[info] Specify --sweep RULE PARAM VALUES or --walk-forward.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
