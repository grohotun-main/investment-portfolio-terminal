"""Run the Phase F stress scenarios against the recommended hedge program.

Prints a per-scenario × per-rule table showing leg P&L at shock day and at
post-shock recovery, plus which rule would fire.

Run:
  py parsers/run_hedge_stress.py
  py parsers/run_hedge_stress.py --notional 2980000 --moneyness 0.05
  py parsers/run_hedge_stress.py --sigma 0.18    # higher-vol assumption
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from hedge_exit_simulator import HedgePolicy  # noqa: E402
from hedge_stress_scenarios import stress_test_program  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"


def _latest_spot() -> float:
    """Latest SPY close from daily_prices.csv."""
    df = pd.read_csv(DATA / "daily_prices.csv", parse_dates=["date"])
    spy = df[df["symbol"] == "SPY"].sort_values("date")
    return float(spy.iloc[-1]["close"])


def _vix_implied_sigma() -> float:
    """Latest VIX → SPY ~30-DTE ATM IV proxy. Add ~1% for 90-DTE term."""
    df = pd.read_csv(DATA / "vix_history.csv", parse_dates=["date"])
    df = df.sort_values("date")
    vix = float(df.iloc[-1]["close"])
    return vix / 100.0


def _pivot_fmt(df: pd.DataFrame, value: str, obs: str,
               fmt) -> pd.DataFrame:
    sub = df[df["observation"] == obs]
    piv = sub.pivot(index="scenario", columns="rule", values=value)
    return piv.map(fmt)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--today", default=None, help="ISO date for 'today' (default: latest SPY bar).")
    p.add_argument("--spot", type=float, default=None, help="Override SPY spot.")
    p.add_argument("--sigma", type=float, default=None, help="Override SPY 90-DTE ATM IV (default: VIX/100).")
    p.add_argument("--dte", type=int, default=90)
    p.add_argument("--moneyness", type=float, default=0.05)
    p.add_argument("--notional", type=float, default=2_980_000.0)
    p.add_argument("--recovery-days", type=int, default=30)
    p.add_argument("--recovery-frac", type=float, default=0.50)
    args = p.parse_args(argv)

    spot = args.spot if args.spot is not None else _latest_spot()
    sigma = args.sigma if args.sigma is not None else _vix_implied_sigma()
    today = date.fromisoformat(args.today) if args.today else date.today()

    policy = HedgePolicy(
        target_dte=args.dte, target_moneyness=args.moneyness,
        notional_protected=args.notional,
    )

    df = stress_test_program(
        policy, rule_kwargs_by_rule=None,
        today=today, spot=spot, sigma_atm=sigma,
        recovery_days=args.recovery_days,
        recovery_frac=args.recovery_frac,
    )

    print(f"Today          : {today}")
    print(f"SPY spot       : ${spot:.2f}")
    print(f"ATM IV (90-DTE): {sigma:.1%}")
    print(f"Policy         : SPY puts, ~{args.dte}d DTE, ~{args.moneyness:.1%} OTM, ${args.notional:,.0f} notional")
    print(f"Recommended leg: K=${df.attrs['leg_strike']:.0f}  "
          f"exp={df.attrs['leg_expiry']}  "
          f"contracts={df.attrs['leg_contracts']}  "
          f"premium=${df.attrs['leg_premium']:.2f}  "
          f"cost=${df.attrs['leg_cost_basis']:,.0f}")
    print()

    print("=" * 80)
    print(f"SHOCK DAY (t=0) — instantaneous spot × spot_mult, IV × vol_mult")
    print("=" * 80)
    print("Payoff as % of notional protected:")
    print(_pivot_fmt(df, "pnl_pct_notional", "shock_day",
                    lambda x: f"{x:+.2f}").to_string())
    print()
    print("Would each rule fire on this day?")
    print(_pivot_fmt(df, "rule_fires", "shock_day",
                    lambda b: "FIRE" if b else "hold").to_string())
    print()

    print("=" * 80)
    print(f"RECOVERY DAY (t+{args.recovery_days}d) — spot recovers {args.recovery_frac:.0%} of shock, IV back to normal")
    print("=" * 80)
    print("Payoff as % of notional protected:")
    print(_pivot_fmt(df, "pnl_pct_notional", "recovery_day",
                    lambda x: f"{x:+.2f}").to_string())
    print()
    print("Would each rule fire on this day?")
    print(_pivot_fmt(df, "rule_fires", "recovery_day",
                    lambda b: "FIRE" if b else "hold").to_string())
    print()
    print(
        "Read: 'FIRE' on shock day = rule closes leg at the shock-day MV, "
        "capturing the peak gain. 'FIRE' on recovery day = rule closes at "
        "a partially-decayed MV. 'hold' means the leg stays open."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
