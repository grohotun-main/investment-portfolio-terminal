"""Stress-shock scenarios for the Phase F exit-rule simulator.

Walk-forward back-tests (F.7) can only tell you about regimes you've already
seen. Stress scenarios cover the regimes you haven't — synthetic spot+IV
shocks at 4 canonical tail magnitudes (Mild, Liberation-Day, Moderate,
COVID), reprice the recommended hedge program's "next leg", and report:

  * Sleeve MV at shock + would-fire of each exit rule on shock day
  * Sleeve MV 30 days post-shock under a 50%-recovery assumption + would-fire
  * Net P&L (current premium − recovery-point MV) per scenario per rule

Reuses ``parsers.stress_hedge`` canonical Scenario list and the Phase A
LR American pricer for option valuation.

Why "next leg" instead of the current sleeve
--------------------------------------------
The Phase F simulator opens one leg at a time per the policy. Stress
testing TODAY against the user-recommended policy means asking "if I
were to enter today's recommended leg and a tail event happened, what
would each rule do?" That's the actionable forward-looking question.
Stress against the literal current sleeve is what PR #79's
``run_stress_hedge`` already covers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "parsers") not in sys.path:
    sys.path.insert(0, str(ROOT / "parsers"))

from hedge_exit_simulator import (  # noqa: E402
    CONTRACT_MULT,
    EXIT_RULES,
    HedgePolicy,
    Leg,
    SimState,
    contracts_for_notional,
)
from options_pricer import binomial_american  # noqa: E402
from stress_hedge import DEFAULT_SCENARIOS, Scenario  # noqa: E402


@dataclass
class StressOutcome:
    """One scenario × one rule × one observation point (shock or recovery)."""
    scenario: str
    rule: str
    observation: str         # "shock_day" | "recovery_day"
    days_from_today: int     # 0 for shock, +N for recovery
    spy_spot: float
    leg_mv_per_share: float
    leg_mv_total: float
    rule_fires: bool
    pnl_if_close_now: float  # $ — closes leg at current MV vs original premium
    pnl_pct_notional: float  # pnl as % of notional protected


def _build_recommended_leg(
    policy: HedgePolicy, spot: float, sigma_atm: float,
    today: date, r: float = 0.045, q: float = 0.013,
) -> tuple[Leg, float]:
    """Build the leg the policy would open today, return (leg, premium_today).

    Uses LR-priced premium based on ATM IV — same convention as PR #79.
    For 5%-OTM puts, actual market premium typically runs ~10-20% higher
    due to skew; we accept that mild understatement as the trade-off for
    not having a live IV surface.
    """
    K = round(spot * (1 - policy.target_moneyness) / 5.0) * 5.0
    expiry = today + timedelta(days=policy.target_dte)
    T = policy.target_dte / 365.25
    res = binomial_american(spot=spot, strike=K, T=T, r=r, q=q,
                            sigma=sigma_atm, opt="put", method="lr")
    premium = float(res["price"])
    contracts = contracts_for_notional(policy.notional_protected, K)
    leg = Leg(
        open_date=pd.Timestamp(today), ticker=f"SPY_{today.isoformat()}_K{int(K)}_P",
        underlying="SPY", expiry=expiry, strike=K,
        contracts=contracts, premium_paid=premium,
    )
    return leg, premium


def _rule_predicate(rule_name: str, leg: Leg, state: SimState,
                    rule_kwargs: Optional[dict] = None) -> bool:
    """Evaluate exit-rule predicate; return True if it would fire."""
    fn = EXIT_RULES[rule_name]
    if rule_kwargs:
        result = fn(leg, state, **rule_kwargs)
    else:
        result = fn(leg, state)
    return result is not None


def _reprice_leg(leg: Leg, spot: float, sigma: float, today: date,
                 r: float = 0.045, q: float = 0.013) -> float:
    """LR-reprice leg under (spot, sigma) at the time delta."""
    T = max(((leg.expiry - today).days) / 365.25, 1.0 / 365.25)
    res = binomial_american(spot=spot, strike=leg.strike, T=T, r=r, q=q,
                            sigma=sigma, opt="put", method="lr")
    return float(res["price"])


def stress_test_program(
    policy: HedgePolicy, rule_kwargs_by_rule: Optional[dict[str, dict]],
    *, today: date, spot: float, sigma_atm: float,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    recovery_days: int = 30,
    recovery_frac: float = 0.50,
    r: float = 0.045, q: float = 0.013,
    shock_iv_rank: float = 100.0,    # post-shock vol is empirically extreme
    recovery_iv_rank: float = 50.0,  # vol has decayed by recovery point
) -> pd.DataFrame:
    """Apply ``scenarios`` to the policy's "next leg" and evaluate each
    exit rule at the shock day and at the recovery day.

    Returns DataFrame with one row per (scenario × rule × observation),
    columns matching StressOutcome.

    Observation points
    ------------------
    * **shock_day** — t=0 instant shock. Spot becomes spot×spot_mult, IV
      becomes sigma_atm×vol_mult, DTE unchanged. Reprice + check rule.
    * **recovery_day** — t+``recovery_days`` later. Spot recovers
      ``recovery_frac`` of the shock toward original spot. IV decays
      linearly back to sigma_atm. DTE reduced by recovery_days.
    """
    rule_kwargs_by_rule = rule_kwargs_by_rule or {}
    leg, premium = _build_recommended_leg(
        policy, spot, sigma_atm, today, r=r, q=q,
    )
    notional = policy.notional_protected
    rows: list[dict] = []

    for sc in scenarios:
        # Shock day: spot × spot_mult, sigma × vol_mult, DTE unchanged.
        shock_spot = spot * sc.spot_mult
        shock_sigma = sigma_atm * sc.vol_mult
        leg_mv_shock = _reprice_leg(leg, shock_spot, shock_sigma, today, r=r, q=q)

        # Recovery day: spot recovers `recovery_frac` of the shock back toward spot.
        recovery_date = today + timedelta(days=recovery_days)
        recovery_spot = shock_spot + recovery_frac * (spot - shock_spot)
        # IV decays linearly from shock_sigma back to sigma_atm over recovery_days.
        recovery_sigma = sigma_atm  # full decay assumption
        leg_mv_recovery = _reprice_leg(leg, recovery_spot, recovery_sigma,
                                       recovery_date, r=r, q=q)

        # Build SimState for each observation point. peak/trough reflect
        # the post-shock world: peak stays at today's spot (pre-shock high),
        # trough is at the shock low.
        state_shock = SimState(
            today=pd.Timestamp(today), spot=shock_spot,
            peak_spot=spot, trough_spot=shock_spot,
            leg_mv_per_share=leg_mv_shock,
            leg_total_mv=leg_mv_shock * leg.contracts * CONTRACT_MULT,
            flatted=False, flatted_at_peak=None,
            iv_rank_today=shock_iv_rank,
        )
        state_recovery = SimState(
            today=pd.Timestamp(recovery_date), spot=recovery_spot,
            peak_spot=spot, trough_spot=shock_spot,
            leg_mv_per_share=leg_mv_recovery,
            leg_total_mv=leg_mv_recovery * leg.contracts * CONTRACT_MULT,
            flatted=False, flatted_at_peak=None,
            iv_rank_today=recovery_iv_rank,
        )

        for rule_name in EXIT_RULES:
            kwargs = rule_kwargs_by_rule.get(rule_name)
            fires_shock = _rule_predicate(rule_name, leg, state_shock, kwargs)
            fires_recovery = _rule_predicate(rule_name, leg, state_recovery, kwargs)

            for obs, st, mv_ps, fires, dfd in (
                ("shock_day", state_shock, leg_mv_shock, fires_shock, 0),
                ("recovery_day", state_recovery, leg_mv_recovery, fires_recovery,
                 recovery_days),
            ):
                mv_total = mv_ps * leg.contracts * CONTRACT_MULT
                pnl_close = mv_total - leg.cost_basis_total()
                rows.append({
                    "scenario": sc.name,
                    "rule": rule_name,
                    "observation": obs,
                    "days_from_today": dfd,
                    "spy_spot": st.spot,
                    "leg_mv_per_share": mv_ps,
                    "leg_mv_total": mv_total,
                    "rule_fires": fires,
                    "pnl_if_close_now": pnl_close,
                    "pnl_pct_notional": (pnl_close / notional * 100) if notional > 0 else 0.0,
                })

    out = pd.DataFrame(rows)
    out.attrs["leg_premium"] = premium
    out.attrs["leg_cost_basis"] = leg.cost_basis_total()
    out.attrs["leg_strike"] = leg.strike
    out.attrs["leg_contracts"] = leg.contracts
    out.attrs["leg_expiry"] = leg.expiry
    out.attrs["spot_today"] = spot
    out.attrs["sigma_atm_today"] = sigma_atm
    return out
