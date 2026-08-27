"""Comparison runner for the Phase F exit-rule back-test.

Loads SPY history + the synthetic put grid (fetched by
``fetch_synthetic_put_grid.py``), runs all 4 exit rules over a back-test
window, finds SPY drawdown episodes, and prints a side-by-side table:

    rule | drag (%/yr) | mean episode payoff (%) | n trades | Pareto?

Use this to validate the simulator end-to-end before wiring it into the
Streamlit tab.

Run:
  py parsers/run_exit_rule_comparison.py                       # 2y window, default policy
  py parsers/run_exit_rule_comparison.py --start 2024-05-25 \
      --end 2026-05-25 --dte 90 --moneyness 0.05 \
      --notional 500000 --dd-threshold 3.0
  py parsers/run_exit_rule_comparison.py --by-episode          # also print per-episode breakdown
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from hedge_effectiveness import find_drawdown_episodes  # noqa: E402
from hedge_exit_simulator import (  # noqa: E402
    EXIT_RULES,
    HedgePolicy,
    compare_runs,
    episode_payoffs,
    pareto_frontier_mask,
    simulate_program,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
SPY_CSV = DATA / "daily_prices.csv"
GRID_CSV = DATA / "option_grid_history.csv"


def _load_spy() -> pd.DataFrame:
    df = pd.read_csv(SPY_CSV, parse_dates=["date"])
    spy = df[df["symbol"] == "SPY"][["date", "close"]].sort_values("date")
    return spy.reset_index(drop=True)


def _load_grid() -> pd.DataFrame:
    if not GRID_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(GRID_CSV, parse_dates=["date", "expiry", "fetched_at"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=str, default=None,
                   help="Back-test start (default: 2y ago).")
    p.add_argument("--end", type=str, default=None,
                   help="Back-test end (default: today).")
    p.add_argument("--dte", type=int, default=90,
                   help="Target DTE at open (default: 90).")
    p.add_argument("--moneyness", type=float, default=0.05,
                   help="Target moneyness (default: 0.05 = 5%% OTM).")
    p.add_argument("--notional", type=float, default=500_000.0,
                   help="$ notional protected (default: 500_000).")
    p.add_argument("--dd-threshold", type=float, default=3.0,
                   help="Min SPY decline pct to flag as episode (default: 3.0).")
    p.add_argument("--by-episode", action="store_true",
                   help="Also print a per-episode × per-rule sleeve-payoff table.")
    p.add_argument("--rules", type=str, default="all",
                   help="Comma-separated rule names, or 'all' (default).")
    args = p.parse_args(argv)

    spy = _load_spy()
    if spy.empty:
        print(f"[!] No SPY history at {SPY_CSV}.")
        return 1
    grid = _load_grid()
    if grid.empty:
        print(f"[!] No grid at {GRID_CSV}. Run "
              "fetch_synthetic_put_grid.py --write first.")
        return 1

    today = date.today()
    end = date.fromisoformat(args.end) if args.end else today
    start = date.fromisoformat(args.start) if args.start else (end - timedelta(days=730))

    if args.rules == "all":
        rule_names = list(EXIT_RULES.keys())
    else:
        rule_names = [r.strip() for r in args.rules.split(",") if r.strip()]
        for r in rule_names:
            if r not in EXIT_RULES:
                print(f"[!] Unknown rule {r!r}. Available: {sorted(EXIT_RULES.keys())}")
                return 1

    policy = HedgePolicy(
        target_dte=args.dte, target_moneyness=args.moneyness,
        notional_protected=args.notional,
    )

    print(f"Back-test window  : {start} → {end}")
    print(f"Policy            : SPY puts, ~{args.dte}d DTE, "
          f"~{args.moneyness:.1%} OTM, ${args.notional:,.0f} notional")
    print(f"Grid cache        : {len(grid)} rows, "
          f"{grid['contract_ticker'].nunique()} unique contracts")
    print(f"SPY rows in window: "
          f"{((spy['date'] >= pd.Timestamp(start)) & (spy['date'] <= pd.Timestamp(end))).sum()}")
    print(f"Rules             : {', '.join(rule_names)}")
    print()

    # 1) Run each rule.
    runs: dict[str, tuple[pd.DataFrame, list]] = {}
    for rule in rule_names:
        print(f"  running {rule}...", end=" ", flush=True)
        ledger, legs = simulate_program(
            policy, rule, spy, grid, start=start, end=end,
        )
        runs[rule] = (ledger, legs)
        n_open = sum(1 for l in legs if l.is_open())
        print(f"{len(legs)} legs ({n_open} still open at end)")

    # 2) Find drawdown episodes.
    episodes = find_drawdown_episodes(
        spy, threshold_pct=args.dd_threshold,
        start_date=start, end_date=end,
    )
    print(f"\nSPY drawdown episodes ≥{args.dd_threshold}%: {len(episodes)}")
    if not episodes.empty:
        for _, ep in episodes.iterrows():
            rec = f", recovered {ep['recover_date'].date()}" if pd.notna(ep["recover_date"]) else " [ONGOING]"
            print(f"  {ep['peak_date'].date()} → {ep['trough_date'].date()}  "
                  f"{ep['decline_pct']:+.1f}%{rec}")

    # 3) Side-by-side comparison.
    cmp = compare_runs(runs, episodes, policy)
    pareto = pareto_frontier_mask(cmp)
    cmp["pareto"] = pareto.values

    print("\n" + "=" * 80)
    print("Rule comparison")
    print("=" * 80)
    cols = [
        "rule", "annualized_drag_pct", "mean_episode_payoff_pct",
        "median_episode_payoff_pct", "total_episode_payoff_pct",
        "payoff_per_dollar_drag", "n_trades", "pareto",
    ]
    fmtd = cmp[cols].copy()
    fmtd.columns = ["rule", "drag%/yr", "mean_pay%", "med_pay%",
                    "sum_pay%", "pay/$drag", "n_trades", "Pareto?"]
    for c in ("drag%/yr", "mean_pay%", "med_pay%", "sum_pay%"):
        fmtd[c] = fmtd[c].map(lambda x: f"{x:+.2f}")
    fmtd["pay/$drag"] = fmtd["pay/$drag"].map(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a"
    )
    fmtd["Pareto?"] = fmtd["Pareto?"].map(lambda b: "yes" if b else "")
    print(fmtd.to_string(index=False))
    print()
    print("drag%/yr      : Annualized premium burn as % of notional protected (positive = drag)")
    print("mean/med_pay% : Mean/median per-episode sleeve gain as % of notional (positive = paid off)")
    print("sum_pay%      : Sum across episodes")
    print("pay/$drag     : $ sleeve gain per $1 of drag (>1 means hedge paid for itself)")
    print("Pareto?       : on the (low drag, high payoff) frontier — no other rule dominates")

    # 4) Optional per-episode breakdown.
    if args.by_episode and not episodes.empty:
        print("\n" + "=" * 80)
        print("Per-episode payoff (% of notional protected)")
        print("=" * 80)
        ep_table = episodes[[
            "peak_date", "trough_date", "decline_pct",
        ]].copy().reset_index(drop=True)
        ep_table["peak"] = ep_table["peak_date"].dt.strftime("%Y-%m-%d")
        ep_table["trough"] = ep_table["trough_date"].dt.strftime("%Y-%m-%d")
        ep_table["dd%"] = ep_table["decline_pct"].map(lambda x: f"{x:+.1f}")
        out = ep_table[["peak", "trough", "dd%"]].copy()
        for rule in rule_names:
            ledger, _ = runs[rule]
            ep = episode_payoffs(ledger, episodes)
            gains_pct = (
                ep["sleeve_gain_peak_to_trough"].astype("float64")
                / policy.notional_protected * 100
            )
            out[rule] = gains_pct.map(
                lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a"
            ).values
        print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
