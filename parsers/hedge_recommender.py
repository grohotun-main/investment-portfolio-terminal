"""Hedge basket recommender for the Options Hedging Phase 2 tab.

Single entry point: ``build_hedge_basket(...)``. Given current
holdings, existing put positions, daily prices, and the active
mode/target, returns the structure the UI renders top-to-bottom.

Composition:
  parsers/hedge_recommender.py  (this module)  --> parsers/hedge_scenarios.py
                                                --> parsers/crash_betas.py
                                                --> parsers/fetch_targeted_chain.py
                                                --> parsers/option_positions.py
                                                --> parsers/risk_metrics.py (compute_risk_contributions)
                                                --> parsers/fetch_spy_holdings.py

This module owns: identification of excess-MCR names, sizing math
(Mode A and Mode B), assembly of the recommendation structure
(scenarios table + headline readout + existing/new/combined blocks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locked constants (per spec section "Locked defaults"). Do NOT expose to UI.
# ---------------------------------------------------------------------------
TENOR_DAYS = 180
CRASH_WINDOWS: list[tuple[str, str]] = [
    ("2020-02-20", "2020-03-23"),  # COVID
    ("2022-01-01", "2022-06-30"),  # 2022 H1
    ("2025-04-01", "2025-04-30"),  # April 2025
    ("2026-01-29", "2026-03-30"),  # Iran war
]
SCENARIO_DRAWDOWNS: list[float] = [-0.05, -0.10, -0.15, -0.20, -0.25]

DEFAULT_MODE: Literal["A", "B"] = "A"
DEFAULT_CAP_PCT = 0.10
DEFAULT_BUDGET_PCT = 0.010
TAIL_STRIKE_OTM_PCT = 0.20
EXCESS_MCR_THRESHOLD_MULT = 1.5

# Locked after a 5-crash back-test re-validation (2018 Q4, 2020 COVID,
# 2022 H1, April 2025, Iran war 2026) on real Polygon option closes back
# to 2016. Pareto winner: mult=3 (pay/$drag = 3.96, drag = 2.16%/yr).
# Supersedes the earlier mult=5 lock, which was a 2-year-sample artifact
# (pay/$drag looked like 21 off a single profit-take fire).
EXIT_RULE_MULTIPLIER = 3

CONTRACT_MULT = 100


# ---------------------------------------------------------------------------
# Data classes returned by the engine.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PutLeg:
    """One row of the new-puts-to-add recommendation."""
    ticker: str
    role: str                       # "Systematic" | "Idiosyncratic (excess concentration)"
    strike: float
    strike_pct_otm: float
    expiry: date
    contracts: int
    premium_per_share: float
    position_cost: float            # contracts * premium * CONTRACT_MULT
    annualized_drag_pct: float      # (cost * (365/TENOR)) / portfolio_equity_value
    liquidity_ok: bool = True       # False -> falls back to SPY systematic absorption


@dataclass(frozen=True)
class ExistingPut:
    """One row of the read-only existing-puts block."""
    ticker: str
    strike: float
    expiry: date
    contracts: int
    cost_basis: float
    current_value: float
    worst_case_payoff: float


@dataclass(frozen=True)
class ScenarioRow:
    """One row of the 5-row scenarios table."""
    portfolio_drawdown: float       # e.g. -0.10
    implied_spy_drop: float
    unhedged_pnl: float
    existing_payoff: float
    existing_pnl: float             # unhedged + existing payoff (existing puts only, $)
    existing_pnl_pct: float         # existing_pnl / portfolio_value
    new_payoff: float
    combined_pnl: float             # unhedged + existing + new (in $)
    combined_pnl_pct: float         # combined_pnl / portfolio_value


@dataclass
class HedgeRecommendation:
    """Top-level structure consumed by the UI."""
    mode: Literal["A", "B"]
    target: float                   # cap_pct or budget_pct
    current_cap_pct: float          # worst-case LOSS magnitude, existing puts only (0 if no loss)
    target_cap_pct: float           # the cap the user is targeting (Mode A) or
                                    # worst-case loss after target basket (Mode B)
    combined_cap_pct: float         # worst-case LOSS magnitude, existing + new puts (0 if no loss)
    current_worst_pnl_pct: float    # SIGNED worst-case net P&L %, existing only (gain > 0)
    combined_worst_pnl_pct: float   # SIGNED worst-case net P&L %, existing + new (gain > 0)
    scenarios: list[ScenarioRow]
    existing_puts: list[ExistingPut]
    new_puts: list[PutLeg]
    total_new_premium: float
    total_new_drag_pct: float
    total_combined_drag_pct: float
    cap_precision_note: str = ""    # Mode A only -- describes any slip
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Identification: who is "excess MCR"?
# ---------------------------------------------------------------------------
def identify_excess_mcr_names(per_symbol: pd.DataFrame,
                                spy_holdings: pd.DataFrame,
                                *,
                                threshold_mult: float = EXCESS_MCR_THRESHOLD_MULT,
                                ) -> list[str]:
    """Find names whose portfolio MCR % exceeds threshold_mult x
    their natural weight in SPY.

    Args:
        per_symbol: output of ``compute_risk_contributions``; indexed
            by ticker, has column "pctr_pct".
        spy_holdings: output of ``fetch_spy_holdings.normalize_holdings_frame``;
            columns "ticker", "weight_pct".
        threshold_mult: a name is "excess" when MCR% > threshold_mult *
            natural_weight_pct. Default 1.5 keeps the recommendation
            narrow (only material concentrations).

    Returns:
        list of tickers (the equity-slice names where idiosyncratic
        hedge is recommended).
    """
    if per_symbol.empty:
        return []
    spy_map = dict(zip(spy_holdings["ticker"].astype(str).str.upper(),
                        spy_holdings["weight_pct"].astype(float)))
    excess: list[str] = []
    for ticker, row in per_symbol.iterrows():
        mcr = float(row.get("pctr_pct", 0.0))
        natural = float(spy_map.get(str(ticker).upper(), 0.0))
        if mcr > threshold_mult * natural:
            excess.append(str(ticker))
    return excess


def compute_excess_share(per_symbol: pd.DataFrame,
                          spy_holdings: pd.DataFrame,
                          excess_names: Iterable[str],
                          ) -> dict[str, float]:
    """Per-name fraction of MCR that's idiosyncratic.

    excess_share[name] = max(0, mcr% - natural_weight%) / mcr%

    Used to scale the per-name dollar loss when sizing single-name puts:
    the idiosyncratic portion is what the single-name hedge covers; the
    systematic portion is absorbed by the SPY systematic hedge.
    """
    spy_map = dict(zip(spy_holdings["ticker"].astype(str).str.upper(),
                        spy_holdings["weight_pct"].astype(float)))
    out: dict[str, float] = {}
    for ticker in excess_names:
        if ticker not in per_symbol.index:
            continue
        mcr = float(per_symbol.loc[ticker, "pctr_pct"])
        if mcr <= 0:
            out[ticker] = 0.0
            continue
        natural = float(spy_map.get(str(ticker).upper(), 0.0))
        out[ticker] = max(0.0, (mcr - natural) / mcr)
    return out


# ---------------------------------------------------------------------------
# Mode A sizing -- cap drawdown at target.
# ---------------------------------------------------------------------------
def _intrinsic_per_contract(*, strike: float, underlying_at_scenario: float
                             ) -> float:
    """Per-contract intrinsic put payoff at a stress underlying price.

    Propagates NaN when ``underlying_at_scenario`` is NaN. ``max(0.0,
    nan)`` returns 0.0 in CPython (NaN comparisons evaluate False), which
    would silently turn a missing-spot condition into a zero payoff and
    hide upstream cascade bugs — the prior worst-case-payoff symptom.
    """
    if not math.isfinite(underlying_at_scenario):
        return float("nan")
    return max(0.0, strike - underlying_at_scenario) * CONTRACT_MULT


def size_mode_a_basket(*,
                        portfolio_value: float,
                        equity_weight: float,
                        spy_drop_worst: float,
                        cap_pct: float,
                        excess_names: Iterable[str],
                        per_name_drop_worst: Mapping[str, float],
                        per_name_position_mv: Mapping[str, float],
                        per_name_spot: Mapping[str, float],
                        per_name_excess_share: Mapping[str, float],
                        chain_premiums: Mapping[str, dict],
                        existing_payoff_worst: float,
                        ) -> list[PutLeg]:
    """Mode A sizing: cap drawdown at cap_pct.

    Algorithm (per spec section "Mode A"):
      1. For each excess-MCR name, size single-name puts to cover the
         idiosyncratic portion of that name's worst-case loss above the
         per-position cap.
      2. Net unhedged worst-case loss - existing payoff - sum single-name
         payoffs gives residual SPY systematic gap.
      3. Size SPY puts to fill the residual gap.

    All quantities rounded UP via math.ceil (cap is honored at the cost
    of slight overhedge, per spec's open-question resolution).

    Args:
        portfolio_value: total portfolio MV in $.
        equity_weight: fraction of portfolio in equity slice.
        spy_drop_worst: implied SPY drop in worst scenario (negative).
        cap_pct: target cap as fraction (e.g. 0.10).
        excess_names: list of tickers that get idiosyncratic single-name hedges.
        per_name_drop_worst: ticker -> drop% in worst scenario (negative).
        per_name_position_mv: ticker -> position $ MV.
        per_name_spot: ticker -> current spot price.
        per_name_excess_share: ticker -> fraction of name's MCR that's idiosyncratic.
        chain_premiums: ticker -> {"strike": float, "premium": float, "expiry": date, ...}.
        existing_payoff_worst: $ payoff of existing puts in worst scenario.

    Returns list of PutLeg objects (may be empty if existing puts cover).
    """
    legs: list[PutLeg] = []
    # Materialize excess_names so we don't re-iterate a one-shot iterable below.
    excess_names = list(excess_names)

    # --- 1. Idiosyncratic single-name puts ---
    single_name_payoff_at_worst = 0.0
    for ticker in excess_names:
        if ticker not in chain_premiums:
            continue
        if ticker not in per_name_spot or ticker not in per_name_drop_worst:
            continue
        spot = float(per_name_spot[ticker])
        drop = float(per_name_drop_worst[ticker])  # negative
        position_mv = float(per_name_position_mv[ticker])
        excess_share = float(per_name_excess_share.get(ticker, 0.0))
        # Idiosyncratic loss to absorb in worst scenario.
        idio_loss = abs(position_mv * drop * excess_share)
        if not math.isfinite(idio_loss) or idio_loss <= 0:
            # NaN = a no-crash-history name (its drop is undefined — can't size
            # a crash hedge without a crash beta); zero/neg = nothing to
            # absorb. Skip either way rather than feeding NaN into math.ceil
            # below (which raises "cannot convert float NaN to integer" and
            # takes down the whole Options tab). Disclosed via diagnostics.
            continue
        strike = float(chain_premiums[ticker]["strike"])
        premium = float(chain_premiums[ticker]["premium"])
        expiry = chain_premiums[ticker].get("expiry") or date.today()
        underlying_at_worst = spot * (1.0 + drop)
        intrinsic = _intrinsic_per_contract(strike=strike,
                                              underlying_at_scenario=underlying_at_worst)
        if intrinsic <= 0:
            # Strike is too low to pay in the worst scenario; skip this
            # name (its loss will fall through to SPY systematic).
            continue
        contracts = max(1, math.ceil(idio_loss / intrinsic))
        single_name_payoff_at_worst += contracts * intrinsic
        position_cost = contracts * premium * CONTRACT_MULT
        portfolio_equity_value = portfolio_value * equity_weight
        drag_pct = (position_cost * (365.0 / TENOR_DAYS)
                    / max(portfolio_equity_value, 1e-9))
        strike_pct_otm = (spot - strike) / spot if spot > 0 else 0.0
        legs.append(PutLeg(
            ticker=ticker,
            role="Idiosyncratic (excess concentration)",
            strike=strike,
            strike_pct_otm=strike_pct_otm,
            expiry=expiry,
            contracts=contracts,
            premium_per_share=premium,
            position_cost=position_cost,
            annualized_drag_pct=drag_pct,
            liquidity_ok=True,
        ))

    # --- 2. SPY systematic put to fill residual gap ---
    # Total unhedged worst-case loss = -sum name_drop * position_mv
    # (drops are negative, so the sign flip makes loss positive).
    # Skip names whose worst-case drop is non-finite (no-crash-history names —
    # NaN beta → NaN drop). Their loss is undefined, so they're excluded from
    # the systematic sizing too (and disclosed via diagnostics); including the
    # NaN would make residual_gap NaN and silently drop the SPY leg.
    unhedged_loss_worst = -sum(
        float(per_name_drop_worst.get(t, 0.0)) * float(mv)
        for t, mv in per_name_position_mv.items()
        if math.isfinite(float(per_name_drop_worst.get(t, 0.0)))
    )
    cap_dollar = abs(portfolio_value * cap_pct)
    residual_gap = unhedged_loss_worst - existing_payoff_worst - single_name_payoff_at_worst - cap_dollar
    if math.isfinite(residual_gap) and residual_gap > 0 and "SPY" in chain_premiums:
        if "SPY" not in per_name_spot:
            # Orchestrator should always populate SPY spot. Skip rather than
            # reconstruct from strike — that would silently bake in a wrong
            # spot under any caller that didn't follow Mode A's strike convention.
            return legs
        spy_spot = float(per_name_spot["SPY"])
        spy_drop = float(per_name_drop_worst.get("SPY", spy_drop_worst))
        spy_strike = float(chain_premiums["SPY"]["strike"])
        spy_premium = float(chain_premiums["SPY"]["premium"])
        spy_expiry = chain_premiums["SPY"].get("expiry") or date.today()
        underlying_at_worst = spy_spot * (1.0 + spy_drop)
        intrinsic = _intrinsic_per_contract(strike=spy_strike,
                                              underlying_at_scenario=underlying_at_worst)
        if intrinsic > 0:
            contracts = max(1, math.ceil(residual_gap / intrinsic))
            position_cost = contracts * spy_premium * CONTRACT_MULT
            portfolio_equity_value = portfolio_value * equity_weight
            drag_pct = (position_cost * (365.0 / TENOR_DAYS)
                        / max(portfolio_equity_value, 1e-9))
            strike_pct_otm = (spy_spot - spy_strike) / spy_spot if spy_spot > 0 else 0.0
            legs.append(PutLeg(
                ticker="SPY",
                role="Systematic",
                strike=spy_strike,
                strike_pct_otm=strike_pct_otm,
                expiry=spy_expiry,
                contracts=contracts,
                premium_per_share=spy_premium,
                position_cost=position_cost,
                annualized_drag_pct=drag_pct,
                liquidity_ok=True,
            ))
    return legs


# ---------------------------------------------------------------------------
# Mode B sizing — fixed 20% OTM strikes, fixed annual premium budget.
# ---------------------------------------------------------------------------
def size_mode_b_basket(*,
                        portfolio_value: float,
                        equity_weight: float,
                        budget_pct: float,
                        excess_names: Iterable[str],
                        per_name_spot: Mapping[str, float],
                        per_name_excess_mcr_pct: Mapping[str, float],
                        spy_systematic_mcr_pct: float,
                        chain_premiums: Mapping[str, dict],
                        ) -> list[PutLeg]:
    """Mode B sizing: fixed 20% OTM strikes, fixed annual premium budget.

    Budget is split between SPY (carries the systematic component) and
    excess-MCR names (idiosyncratic) in proportion to each's contribution
    to total excess risk.

    Per-leg contracts = floor(budget_for_leg / (premium x 100)). Floor
    (not ceil) because the budget is a hard cap, not a target.
    """
    # Coerce to list so excess_names can be iterated twice safely.
    excess_names = list(excess_names)

    total_risk = float(spy_systematic_mcr_pct) + sum(
        float(per_name_excess_mcr_pct.get(t, 0.0)) for t in excess_names
    )
    if total_risk <= 0:
        return []

    annual_budget = portfolio_value * budget_pct
    legs: list[PutLeg] = []
    portfolio_equity_value = portfolio_value * equity_weight

    # --- SPY leg ---
    spy_share = float(spy_systematic_mcr_pct) / total_risk
    spy_budget = annual_budget * spy_share
    if "SPY" in chain_premiums and spy_budget > 0:
        if "SPY" not in per_name_spot:
            pass  # skip SPY leg — spot missing; idiosyncratic legs may still proceed
        else:
            spy_spot = float(per_name_spot["SPY"])
            strike = float(chain_premiums["SPY"]["strike"])
            premium = float(chain_premiums["SPY"]["premium"])
            expiry = chain_premiums["SPY"].get("expiry") or date.today()
            per_contract_cost = premium * CONTRACT_MULT
            if per_contract_cost > 0:
                contracts = max(0, math.floor(spy_budget / per_contract_cost))
                if contracts > 0:
                    position_cost = contracts * per_contract_cost
                    drag_pct = (position_cost * (365.0 / TENOR_DAYS)
                                / max(portfolio_equity_value, 1e-9))
                    strike_pct_otm = (spy_spot - strike) / spy_spot if spy_spot > 0 else TAIL_STRIKE_OTM_PCT
                    legs.append(PutLeg(
                        ticker="SPY", role="Systematic",
                        strike=strike, strike_pct_otm=strike_pct_otm,
                        expiry=expiry, contracts=contracts,
                        premium_per_share=premium,
                        position_cost=position_cost,
                        annualized_drag_pct=drag_pct,
                        liquidity_ok=True,
                    ))

    # --- Per-name idiosyncratic legs ---
    for ticker in excess_names:
        if ticker not in chain_premiums:
            continue
        if ticker not in per_name_spot:
            continue
        share = float(per_name_excess_mcr_pct.get(ticker, 0.0)) / total_risk
        leg_budget = annual_budget * share
        spot = float(per_name_spot[ticker])
        strike = float(chain_premiums[ticker]["strike"])
        premium = float(chain_premiums[ticker]["premium"])
        expiry = chain_premiums[ticker].get("expiry") or date.today()
        per_contract_cost = premium * CONTRACT_MULT
        if per_contract_cost <= 0 or leg_budget <= 0:
            continue
        contracts = max(0, math.floor(leg_budget / per_contract_cost))
        if contracts == 0:
            continue
        position_cost = contracts * per_contract_cost
        drag_pct = (position_cost * (365.0 / TENOR_DAYS)
                    / max(portfolio_equity_value, 1e-9))
        strike_pct_otm = (spot - strike) / spot if spot > 0 else TAIL_STRIKE_OTM_PCT
        legs.append(PutLeg(
            ticker=ticker,
            role="Idiosyncratic (excess concentration)",
            strike=strike, strike_pct_otm=strike_pct_otm,
            expiry=expiry, contracts=contracts,
            premium_per_share=premium,
            position_cost=position_cost,
            annualized_drag_pct=drag_pct,
            liquidity_ok=True,
        ))
    return legs


# ---------------------------------------------------------------------------
# Existing-puts payoff + table
# ---------------------------------------------------------------------------
def evaluate_existing_puts_payoff(positions: list[dict],
                                   per_scenario_spot: Mapping[float, Mapping[str, float]],
                                   ) -> dict[float, float]:
    """For each scenario drawdown level, sum intrinsic payoff of all existing puts.

    Args:
        positions: list of dicts with keys
            {underlying, strike, expiry, contracts}.
        per_scenario_spot: scenario_drawdown -> {ticker: stress_spot}.

    Returns:
        scenario_drawdown -> total $ intrinsic payoff (sum across all positions).
    """
    out: dict[float, float] = {}
    for scenario, spot_map in per_scenario_spot.items():
        total = 0.0
        for pos in positions:
            ticker = pos["underlying"]
            if ticker not in spot_map:
                continue
            stress_spot = float(spot_map[ticker])
            if not math.isfinite(stress_spot):
                # Missing stressed spot (e.g. a no-crash-history underlying) ->
                # this leg's payoff is unknowable. Propagate NaN rather than
                # letting max(0.0, nan) == 0.0 silently zero the put's
                # protection, mirroring _intrinsic_per_contract above.
                intrinsic_per_share = float("nan")
            else:
                intrinsic_per_share = max(0.0, float(pos["strike"]) - stress_spot)
            total += intrinsic_per_share * float(pos["contracts"]) * CONTRACT_MULT
        out[scenario] = total
    return out


def existing_puts_to_table(positions: list[dict],
                             worst_payoffs: list[float],
                             ) -> list[ExistingPut]:
    """Build the read-only existing-puts block rows.

    ``worst_payoffs`` is a parallel list — one entry per position, in the
    same order. We use a parallel list instead of a dict keyed on
    (underlying, strike, expiry) because two positions in different
    brokers can share that key (e.g. user holds NVDA $115 puts in both
    Harbor and Alpine) — a dict would silently collide and both rows
    would display the same number.
    """
    if len(worst_payoffs) != len(positions):
        raise ValueError(
            f"worst_payoffs length ({len(worst_payoffs)}) must match "
            f"positions length ({len(positions)})"
        )
    out: list[ExistingPut] = []
    for pos, worst in zip(positions, worst_payoffs):
        cost_basis = (float(pos.get("cost_basis_per_share", 0.0))
                      * float(pos["contracts"]) * CONTRACT_MULT)
        out.append(ExistingPut(
            ticker=str(pos["underlying"]),
            strike=float(pos["strike"]),
            expiry=pos["expiry"],
            contracts=int(pos["contracts"]),
            cost_basis=cost_basis,
            current_value=float(pos.get("market_value", 0.0)),
            worst_case_payoff=worst,
        ))
    return out


def compute_roll_schedule(expiry: date, today: date, *,
                          dte_trigger_days: int = 90,
                          roll_tenor_days: int = TENOR_DAYS,
                          ) -> tuple[date, date]:
    """Roll schedule for one held put (time-based trigger only).

    Returns ``(roll_by, roll_into)``:

      roll_by   = ``expiry - dte_trigger_days`` — the documented time
                  trigger ("roll at 90 DTE"). May be in the past for a put
                  already inside the window.
      roll_into = (roll_by, or ``today`` if roll_by is already past)
                  ``+ roll_tenor_days`` — the approximate expiry to roll
                  into; in practice snap to the nearest listed monthly.

    The ±10% SPY-drift trigger is price-based, not date-based, so it is not
    expressed here — the UI surfaces it as a separate condition.
    """
    roll_by = expiry - timedelta(days=dte_trigger_days)
    roll_anchor = roll_by if roll_by > today else today
    roll_into = roll_anchor + timedelta(days=roll_tenor_days)
    return roll_by, roll_into


# ---------------------------------------------------------------------------
# Worst-case aggregation (signed) + loss cap
# ---------------------------------------------------------------------------
def _worst_case_pnl_pct(scenarios: Iterable["ScenarioRow"],
                        portfolio_value: float,
                        *, include_new: bool) -> float:
    """Most-NEGATIVE signed net P&L % across scenarios with a finite implied
    SPY drop. A loss is negative, a gain positive. ``include_new`` toggles the
    recommended-puts payoff (existing-only vs combined). NaN when no scenario
    is valid.

    Replaces an earlier ``max(abs(...))`` aggregation that stripped the sign and
    so reported a deep-scenario convex GAIN (the over-covering basket) as a
    "worst-case loss". The honest worst case is the minimum signed outcome.
    """
    pnls: list[float] = []
    for s in scenarios:
        if not math.isfinite(s.implied_spy_drop):
            continue
        net = s.unhedged_pnl + s.existing_payoff
        if include_new:
            net += s.new_payoff
        pnls.append(net / max(portfolio_value, 1e-9))
    return min(pnls) if pnls else float("nan")


def _loss_cap_pct(worst_pnl_pct: float) -> float:
    """Positive worst-case LOSS magnitude from a signed worst-case P&L %.

    0.0 when the worst case is a gain (the hedges over-cover every scenario);
    NaN propagates. Keeps the downstream cap comparisons (already-covered,
    over-covers) operating on a positive "max loss" while the signed field
    drives honest gain/loss wording in the UI.
    """
    if not math.isfinite(worst_pnl_pct):
        return float("nan")
    return max(0.0, -worst_pnl_pct)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def build_hedge_basket(*,
                        mode: Literal["A", "B"],
                        target: float,
                        holdings: pd.DataFrame,
                        existing_options: list[dict],
                        per_symbol_mcr: pd.DataFrame,
                        spy_holdings: pd.DataFrame,
                        chain_premiums: Mapping[str, dict],
                        crash_betas: Mapping[str, float],
                        today: date,
                        spot_prices: Mapping[str, float],
                        ) -> HedgeRecommendation:
    """Top-level entry: assemble the full HedgeRecommendation.

    All heavy dependencies are *injected* (chain_premiums, crash_betas,
    per_symbol_mcr, spy_holdings) so the engine itself
    is testable without network or cache. The Streamlit caller is
    responsible for producing these inputs.

    Algorithm:
      1. Classify holdings into equity vs cash-equivalent slices.
      2. Identify excess-MCR names.
      3. For each of the 5 SCENARIO_DRAWDOWNS, invert to per-name drops.
      4. Compute unhedged P&L per scenario.
      5. Compute existing-puts payoff per scenario.
      6. Size target basket via Mode A or Mode B.
      7. Net target - existing -> new_puts (the delta).
      8. Compute scenario row for hedged P&L.
      9. Build headline 3 caps (current / target / combined).
      10. Assemble HedgeRecommendation.
    """
    # Local import to keep this module free of a circular-import risk if
    # hedge_scenarios ever needs anything from hedge_recommender.
    from hedge_scenarios import (
        classify_holdings, HoldingClass,
        invert_portfolio_drawdown, per_name_scenario_drops,
        equity_names_without_crash_history,
    )

    # --- 1. Classify holdings, compute portfolio splits ---
    # Collapse multi-row tickers (e.g. SPY held in both Harbor and Alpine,
    # or a snapshot frame that wasn't yet filtered to one statement_date)
    # by summing market_value. Without this, set_index("ticker") produces
    # a duplicate index that inflates weighted_beta inside
    # invert_portfolio_drawdown, and dict(zip(...)) silently keeps only
    # the last row in per_name_position_mv.
    h = (holdings.copy()
                  .groupby("ticker", as_index=False)["market_value"].sum())
    h["_class"] = h["ticker"].apply(classify_holdings)
    portfolio_value = float(h["market_value"].sum())
    equity_value = float(h[h["_class"] == HoldingClass.EQUITY]["market_value"].sum())
    equity_weight = equity_value / max(portfolio_value, 1e-9)
    equity_h = h[h["_class"] == HoldingClass.EQUITY].copy()
    if equity_h["market_value"].sum() > 0:
        equity_h["_w_eq"] = equity_h["market_value"] / equity_h["market_value"].sum()
    else:
        equity_h["_w_eq"] = 0.0
    weights_in_equity = pd.Series(equity_h.set_index("ticker")["_w_eq"])
    per_name_position_mv = dict(zip(h["ticker"], h["market_value"].astype(float)))

    # --- 2. Excess-MCR names ---
    excess_names = identify_excess_mcr_names(per_symbol_mcr, spy_holdings)
    excess_share = compute_excess_share(per_symbol_mcr, spy_holdings, excess_names)

    # MCR breakdown (mode-independent — feeds Mode-B sizing AND the
    # hedge-signal panel via diagnostics). per_name_mcr_pct: each excess
    # name's MCR %. spy_systematic_mcr_pct: the residual broad-market MCR
    # after excess + cash names are removed (100% - excess - cash).
    per_name_mcr_pct = {t: float(per_symbol_mcr.loc[t, "pctr_pct"])
                        for t in excess_names if t in per_symbol_mcr.index}
    cash_mcr_sum = sum(
        float(per_symbol_mcr.loc[t, "pctr_pct"])
        for t in per_symbol_mcr.index
        if classify_holdings(str(t)) == HoldingClass.CASH_EQUIVALENT
    )
    spy_systematic_mcr_pct = max(
        0.0, 100.0 - sum(per_name_mcr_pct.values()) - cash_mcr_sum)

    # Equity names the inverter excludes for lack of a usable crash beta
    # (no crash-window history — e.g. a listing newer than the most recent
    # crash window). Surfaced to the UI so the renormalization is disclosed.
    scenario_excluded_no_history = equity_names_without_crash_history(
        weights_in_equity=weights_in_equity, crash_betas=crash_betas,
    )

    # --- 3. Per-scenario drops ---
    per_scenario_spy: dict[float, float] = {}
    per_scenario_name_drops: dict[float, dict[str, float]] = {}
    per_scenario_spot: dict[float, dict[str, float]] = {}
    for d in SCENARIO_DRAWDOWNS:
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=d, equity_weight=equity_weight,
            weights_in_equity=weights_in_equity, crash_betas=crash_betas,
        )
        per_scenario_spy[d] = spy_drop
        if np.isnan(spy_drop):
            per_scenario_name_drops[d] = {}
            per_scenario_spot[d] = {}
            continue
        drops = per_name_scenario_drops(spy_drop=spy_drop, crash_betas=crash_betas)
        per_scenario_name_drops[d] = drops
        per_scenario_spot[d] = {
            t: float(spot_prices.get(t, 0.0)) * (1.0 + drop)
            for t, drop in drops.items()
        }

    # --- 4. Unhedged P&L per scenario ---
    def _unhedged(scenario: float) -> float:
        # Skip names whose scenario drop is non-finite — these are the
        # no-crash-history names invert_portfolio_drawdown already excluded
        # (NaN beta → NaN drop). Including them would NaN-out the whole sum
        # and blank the table for a single brand-new position. Their
        # exclusion is disclosed via diagnostics["scenario_excluded_no_history"].
        # Names absent from `drops` (cash, untracked) contribute 0 as before.
        drops = per_scenario_name_drops.get(scenario, {})
        total = 0.0
        for t, mv in per_name_position_mv.items():
            drop = drops.get(t, 0.0)
            if np.isfinite(drop):
                total += drop * mv
        return total

    # --- 5. Existing puts payoff per scenario ---
    existing_payoffs = evaluate_existing_puts_payoff(
        existing_options, per_scenario_spot,
    )
    worst = SCENARIO_DRAWDOWNS[-1]
    existing_payoff_worst = existing_payoffs.get(worst, 0.0)

    # --- 6. Size target basket ---
    if mode == "A":
        target_basket = size_mode_a_basket(
            portfolio_value=portfolio_value,
            equity_weight=equity_weight,
            spy_drop_worst=per_scenario_spy.get(worst, -0.25),
            cap_pct=target,
            excess_names=excess_names,
            per_name_drop_worst=per_scenario_name_drops.get(worst, {}),
            per_name_position_mv=per_name_position_mv,
            per_name_spot={t: float(spot_prices.get(t, 0.0))
                            for t in spot_prices},
            per_name_excess_share=excess_share,
            chain_premiums=chain_premiums,
            existing_payoff_worst=existing_payoff_worst,
        )
    else:
        target_basket = size_mode_b_basket(
            portfolio_value=portfolio_value,
            equity_weight=equity_weight,
            budget_pct=target,
            excess_names=excess_names,
            per_name_spot={t: float(spot_prices.get(t, 0.0))
                            for t in spot_prices},
            per_name_excess_mcr_pct=per_name_mcr_pct,
            spy_systematic_mcr_pct=spy_systematic_mcr_pct,
            chain_premiums=chain_premiums,
        )

    # --- 7. Subtract existing -> delta = new_puts ---
    # Credit accounting: existing-put payoff at the worst scenario is
    # pooled per-ticker first (same-ticker credit covers same-ticker
    # need preferentially), then any unused credit spills into a
    # portfolio-wide pool that can absorb other tickers' legs. This
    # handles the "already covered" case where a large SPY put pays
    # so much it eliminates the need for any new leg, on any ticker.
    # Missing-spot guard: if the worst-scenario spot map is missing a
    # ticker (i.e. upstream NaN cascade left per_scenario_spot empty), we
    # propagate NaN rather than defaulting to a 0 underlying — which would
    # silently produce a full-notional "underlying went to $0" payoff and
    # make the engine declare "Already covered".
    def _stressed_spot_at_worst(ticker: str) -> float:
        spot_map = per_scenario_spot.get(worst, {})
        if ticker not in spot_map:
            return float("nan")
        return float(spot_map[ticker])

    def _leg_intrinsic_at_worst(leg: PutLeg) -> float:
        underlying_at_worst = _stressed_spot_at_worst(leg.ticker)
        return _intrinsic_per_contract(strike=leg.strike,
                                         underlying_at_scenario=underlying_at_worst
                                         ) * leg.contracts

    existing_by_ticker_worst: dict[str, float] = {}
    for pos in existing_options:
        t = pos["underlying"]
        intrinsic = _intrinsic_per_contract(
            strike=float(pos["strike"]),
            underlying_at_scenario=_stressed_spot_at_worst(t),
        ) * float(pos["contracts"])
        existing_by_ticker_worst[t] = existing_by_ticker_worst.get(t, 0.0) + intrinsic

    # Mode B (tail-hedge budget) recommends the FULL budget basket — it's a
    # standing premium you ladder in regardless of existing coverage, so it
    # is NOT netted against existing puts (existing coverage still shows up
    # in the scenario rows + caps). Mode A (cap drawdown) nets below: credit
    # existing-put payoff against each target leg so we recommend only the
    # shortfall needed to reach the cap.
    if mode == "B":
        new_puts: list[PutLeg] = list(target_basket)
    else:
        new_puts = []
        for leg in target_basket:
            leg_required = _leg_intrinsic_at_worst(leg)
            # First try same-ticker credit.
            same_ticker_credit = existing_by_ticker_worst.get(leg.ticker, 0.0)
            if same_ticker_credit >= leg_required:
                existing_by_ticker_worst[leg.ticker] = same_ticker_credit - leg_required
                continue
            # Spill: borrow from any other ticker's remaining credit.
            residual = leg_required - same_ticker_credit
            existing_by_ticker_worst[leg.ticker] = 0.0
            cross_credit_pool = sum(v for k, v in existing_by_ticker_worst.items()
                                    if k != leg.ticker)
            # All-or-nothing gate: a cross-ticker pool that only partially
            # covers this leg is NOT applied. The "Already covered — buy
            # nothing" intent requires FULL coverage; partial pool stays in
            # reserve. Don't 'fix' this to a proportional draw-down — that
            # would understate the new-puts recommendation.
            if cross_credit_pool >= residual:
                # Draw down the pool proportionally so the bookkeeping stays
                # tractable for any subsequent legs.
                for k in list(existing_by_ticker_worst.keys()):
                    if k == leg.ticker:
                        continue
                    share = (existing_by_ticker_worst[k] / cross_credit_pool
                             if cross_credit_pool > 0 else 0.0)
                    existing_by_ticker_worst[k] = existing_by_ticker_worst[k] - share * residual
                continue
            new_puts.append(leg)

    # --- 8. Scenario rows ---
    new_payoffs: dict[float, float] = {}
    for d in SCENARIO_DRAWDOWNS:
        spot_map = per_scenario_spot.get(d, {})
        if not spot_map:
            # Empty spot map -> scenario couldn't be computed. Propagate
            # NaN (not 0) so the UI shows "—" instead of "$0".
            new_payoffs[d] = float("nan")
            continue
        total = 0.0
        for leg in new_puts:
            if leg.ticker not in spot_map:
                continue
            total += _intrinsic_per_contract(
                strike=leg.strike,
                underlying_at_scenario=spot_map[leg.ticker],
            ) * leg.contracts
        new_payoffs[d] = total

    scenarios: list[ScenarioRow] = []
    for d in SCENARIO_DRAWDOWNS:
        spy_drop = per_scenario_spy.get(d, float("nan"))
        if not np.isfinite(spy_drop):
            # When the inverter failed (e.g. empty weights_in_equity or
            # missing crash_betas) all downstream $ figures are undefined.
            # NaN keeps the missing-data condition visible in the UI.
            scenarios.append(ScenarioRow(
                portfolio_drawdown=d,
                implied_spy_drop=spy_drop,
                unhedged_pnl=float("nan"),
                existing_payoff=float("nan"),
                existing_pnl=float("nan"),
                existing_pnl_pct=float("nan"),
                new_payoff=float("nan"),
                combined_pnl=float("nan"),
                combined_pnl_pct=float("nan"),
            ))
            continue
        unh = _unhedged(d)
        ex = existing_payoffs.get(d, 0.0)
        nw = new_payoffs.get(d, 0.0)
        existing_pnl = unh + ex
        combined = unh + ex + nw
        scenarios.append(ScenarioRow(
            portfolio_drawdown=d,
            implied_spy_drop=spy_drop,
            unhedged_pnl=unh,
            existing_payoff=ex,
            existing_pnl=existing_pnl,
            existing_pnl_pct=existing_pnl / max(portfolio_value, 1e-9),
            new_payoff=nw,
            combined_pnl=combined,
            combined_pnl_pct=combined / max(portfolio_value, 1e-9),
        ))

    # --- 9. Headline caps ---
    # "Worst across 5 scenarios" = the most-NEGATIVE *signed* net P&L %. Earlier
    # this took max(abs(...)), which stripped the sign and grabbed the deepest
    # scenario — where an over-covering basket is a convex GAIN — then labelled
    # that gain a "worst-case loss" (the 23.6% bug). _worst_case_pnl_pct returns
    # the honest signed worst case (gain > 0); _loss_cap_pct converts it to the
    # positive loss magnitude the headline comparisons need (0 when the worst
    # case is itself a gain). Both NaN when no scenario is computable, so the UI
    # still shows "scenarios unavailable" instead of a gaslighting "covered".
    current_worst_pnl_pct = _worst_case_pnl_pct(
        scenarios, portfolio_value, include_new=False)
    combined_worst_pnl_pct = _worst_case_pnl_pct(
        scenarios, portfolio_value, include_new=True)
    current_cap_pct = _loss_cap_pct(current_worst_pnl_pct)
    combined_cap_pct = _loss_cap_pct(combined_worst_pnl_pct)
    target_cap_pct = target if mode == "A" else combined_cap_pct

    # --- 10. Drag aggregates + assemble ---
    total_new_premium = sum(l.position_cost for l in new_puts)
    portfolio_equity_value = max(portfolio_value * equity_weight, 1e-9)
    total_new_drag_pct = (total_new_premium * (365.0 / TENOR_DAYS)
                          / portfolio_equity_value)
    total_existing_premium = sum(
        float(p.get("cost_basis_per_share", 0.0)) * float(p["contracts"])
        * CONTRACT_MULT for p in existing_options
    )
    total_combined_drag_pct = (
        (total_new_premium + total_existing_premium) * (365.0 / TENOR_DAYS)
        / portfolio_equity_value
    )

    # Existing puts -> display rows. Worst-case payoff is computed
    # per-position (parallel list) rather than per (underlying, strike,
    # expiry) key — otherwise two NVDA $115 12/18/26 lots held in
    # different brokers would collide in the dict and both rows would
    # show the same payoff. Uses _intrinsic_per_contract's NaN-on-missing-
    # spot guard so the UI shows "—" instead of a bogus full-notional
    # value when the upstream cascade fails.
    worst_payoffs: list[float] = [
        _intrinsic_per_contract(
            strike=float(pos["strike"]),
            underlying_at_scenario=_stressed_spot_at_worst(pos["underlying"]),
        ) * float(pos["contracts"])
        for pos in existing_options
    ]
    existing_table = existing_puts_to_table(existing_options, worst_payoffs)

    # Cap-precision note (Mode A only). Suppressed when no scenario was
    # computable — the prior bug printed "within target across all 5
    # scenarios" with cap_observed = 0.0 (because all scenarios were NaN
    # and skipped in the cap aggregation). Worst case = the scenario with the
    # most-negative *signed* combined %, so a deep-scenario convex gain is
    # never mistaken for the worst loss.
    note = ""
    valid_scenarios = [s for s in scenarios if np.isfinite(s.implied_spy_drop)]
    if mode == "A" and valid_scenarios:
        worst_s = min(valid_scenarios, key=lambda s: s.combined_pnl_pct)
        worst_d = worst_s.portfolio_drawdown
        worst_pct = worst_s.combined_pnl_pct  # signed; gain > 0
        if worst_pct >= 0:
            note = (f"After the recommended puts, even the worst modeled "
                    f"scenario nets a gain (+{worst_pct*100:.1f}% in the "
                    f"{worst_d*100:.0f}% portfolio scenario). The basket "
                    f"over-covers because single-name puts are sized to "
                    f"neutralize your concentrated names in full, not trimmed "
                    f"to the cap.")
        else:
            cap_observed = -worst_pct  # positive loss magnitude
            slip_pp = (cap_observed - target) * 100.0
            if cap_observed > target + 0.001:
                note = (f"After the recommended puts, worst-case loss is "
                        f"{cap_observed*100:.1f}% in the {worst_d*100:.0f}% "
                        f"portfolio scenario — {slip_pp:+.1f}pp above the "
                        f"{target*100:.0f}% target. The gap is whole-contract "
                        f"discreteness (you can't buy fractional puts); it "
                        f"shrinks in deeper scenarios.")
            else:
                note = (f"After the recommended puts, worst-case loss is capped "
                        f"at {cap_observed*100:.1f}% — inside the "
                        f"{target*100:.0f}% target. The basket over-covers "
                        f"because single-name puts are sized to neutralize your "
                        f"concentrated names in full, not trimmed to the cap.")

    return HedgeRecommendation(
        mode=mode,
        target=target,
        current_cap_pct=current_cap_pct,
        target_cap_pct=target_cap_pct,
        combined_cap_pct=combined_cap_pct,
        current_worst_pnl_pct=current_worst_pnl_pct,
        combined_worst_pnl_pct=combined_worst_pnl_pct,
        scenarios=scenarios,
        existing_puts=existing_table,
        new_puts=new_puts,
        total_new_premium=total_new_premium,
        total_new_drag_pct=total_new_drag_pct,
        total_combined_drag_pct=total_combined_drag_pct,
        cap_precision_note=note,
        diagnostics={
            "portfolio_value":      portfolio_value,
            "equity_weight":        equity_weight,
            "excess_names":         excess_names,
            "per_name_mcr_pct":     per_name_mcr_pct,
            "spy_systematic_mcr_pct": spy_systematic_mcr_pct,
            "spy_drop_per_scenario": per_scenario_spy,
            "scenario_excluded_no_history": scenario_excluded_no_history,
        },
    )
