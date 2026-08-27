"""Stress-scenario hedge calculator — Phase B of the hedging engine.

Given a candidate long-option hedge `(ticker, strike, expiry, n_contracts)`
plus today's spot + IV, reprice the contract at hand-picked tail scenarios
and report upfront cost, scenario payoff, cost-of-insurance, and the
breakeven move.

This is intentionally NOT Monte Carlo. The buying use case is unpredictable
tail events (Liberation Day, COVID); the decision is "which strike and
expiry for this kind of shock", not "what's my P&L distribution". Named
scenarios with explicit (spot, vol) multipliers make that decision legible
in a way that a distribution does not. Full MC stays a Phase B.5 option
(would need Polygon Developer for historical IV regimes).

Conventions
-----------
* Long positions only — you are buying protection.
* **Instant shock** model: the scenario applies to `(spot, sigma)` at t=0,
  DTE unchanged. Models a "what's my position worth right now if this
  happens today" view. A days-elapsed mode could be added later but it
  mixes theta into the scenario interpretation, which is a separate
  decision.
* Uses LR American (`options_pricer.binomial_american(method="lr")`).
  Do not switch to CRR — CRR vega oscillates non-monotonically across N,
  which would create artifacts when sweeping σ.
* Risk-free rate `r` and dividend yield `q` are held flat across
  scenarios. Stressing rates / yields is a separate extension.

Breakeven definitions — two numbers:

  * **At-expiry breakeven** (`breakeven_spot`): spot at which option
    *intrinsic* at expiry equals the upfront premium paid. For a put:
    `K - premium`. Conservative "have I made my money back if held to
    expiry" number.
  * **MTM breakeven** (`mtm_breakeven_spot`): spot at which today's
    *option value* (American, LR-priced at today's T, σ, r, q) equals
    the upfront premium paid. Closer to today's spot than at-expiry
    breakeven because time value is still in the option. Answers
    "where does spot have to be right now for me to be flat if I
    sold the position today?" — relevant for monetize-early decisions.
    Solved via Newton-Raphson on spot using delta; None if no solution
    within sane bracket (e.g. premium > strike for a put).

Wing-skew handling
------------------
Two `vol_baseline` modes:

  * `"contract"` (default, backwards-compat): shocks multiply the
    contract's own IV directly. For deep-OTM puts whose IV is already
    skew-elevated (e.g. 38% on a 30%-OTM SPY put while ATM ~16%), a 5x
    vol_mult gives stressed IV ~190% — above 2020 actuals (SPY ATM
    peaked ~80%). Treat far-OTM headline P&L as an upper bound.
  * `"atm"`: shocks multiply the *ATM* IV for the same expiry, then
    re-apply the absolute vol-point skew (sticky-strike parallel shift).
    Example: ATM=16%, contract=38%, skew=22 vol pts. COVID-style
    5x → ATM_new=80%, contract_new=80+22=102%. Aligns with how
    equity-vol surfaces actually move in stress.
    Requires `sigma_atm` (today's same-expiry ATM IV) — pass explicitly
    or let the CLI extract it from the fetched chain.

Absolute skew is the conservative middle ground: proportional skew
(multiply ratios) would just reproduce the contract-mode result, while
flat skew (assume zero) would ignore the smile entirely.

Units: dollars throughout. `premium_per_share` is the per-share price
(NOT per-contract); multiply by 100 to get per-contract cost.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "parsers") not in sys.path:
    sys.path.insert(0, str(ROOT / "parsers"))

from options_pricer import binomial_american  # noqa: E402

OptType = Literal["call", "put"]

CONTRACT_MULT = 100  # US listed equity options: 100 shares per contract


@dataclass(frozen=True)
class Scenario:
    """A named (spot, vol) shock. spot_mult=0.85 means -15% spot;
    vol_mult=3.0 means triple today's IV."""
    name: str
    spot_mult: float
    vol_mult: float


# Canonical tail scenarios. Magnitudes anchored to real history:
#   * "Mild correction":      -5% / 1.5x IV — routine pullback (Aug-2024)
#   * "Liberation-Day-style": -10% / 2x   — Apr-2025 tariff shock
#   * "Moderate crash":       -15% / 3x   — Q4-2018, Aug-2015 China
#   * "COVID-style":          -35% / 5x   — Feb-Mar 2020
# Override via the `scenarios=` arg to evaluate_hedge.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Mild correction",      0.95, 1.5),
    Scenario("Liberation-Day-style", 0.90, 2.0),
    Scenario("Moderate crash",       0.85, 3.0),
    Scenario("COVID-style",          0.65, 5.0),
)


@dataclass(frozen=True)
class Hedge:
    """A candidate long-option hedge. premium_per_share is what you'd pay
    today (the ask, the mid, or what you actually paid if you already own
    it). All P&L is computed relative to this number."""
    ticker: str
    option_type: OptType
    strike: float
    expiry: date
    n_contracts: int
    premium_per_share: float


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    shocked_spot: float
    shocked_vol: float
    repriced_per_share: float        # LR-priced option value under the shock
    pnl_per_share: float             # = repriced_per_share - premium_per_share
    pnl_total: float                 # = n_contracts * 100 * pnl_per_share
    pnl_pct_of_notional: float       # = pnl_total / notional_protected


@dataclass(frozen=True)
class HedgeEvaluation:
    hedge: Hedge
    spot_today: float
    sigma_today: float
    r: float
    q: float
    T_years: float
    base_per_share: float            # reprice at today's (spot, sigma) — sanity check vs premium
    upfront_cost: float              # n_contracts * 100 * premium_per_share
    notional_protected: float        # n_contracts * 100 * spot_today
    cost_of_insurance_pct: float     # upfront_cost / notional_protected
    breakeven_spot: float            # at-expiry intrinsic = premium
    breakeven_decline_pct: float     # (spot_today - breakeven_spot) / spot_today
    mtm_breakeven_spot: float | None         # today's spot at which mark-to-market = premium; None if no bracket
    mtm_breakeven_decline_pct: float | None  # (spot_today - mtm_breakeven_spot) / spot_today; None if above
    scenarios: tuple[ScenarioResult, ...]
    vol_baseline: Literal["contract", "atm"] = "contract"
    sigma_atm: float | None = None   # ATM IV used in "atm" mode; None in "contract" mode


def _solve_mtm_breakeven_spot(
    strike: float, opt: OptType, T: float, r: float, q: float, sigma: float,
    target_premium: float, n_steps: int, spot_hint: float,
) -> float | None:
    """Solve for spot S where binomial_american(S, ...) = target_premium.
    Newton-Raphson using delta with a bisection fallback.

    Returns None when no positive solution exists in the search bracket —
    typically when `target_premium` is above the option's maximum value
    (e.g. premium > strike for a put). Also returns None at T=0 because
    intrinsic-only breakeven is already captured by `breakeven_spot`.
    """
    if T <= 0 or target_premium <= 0 or sigma <= 0 or strike <= 0:
        return None

    def price_and_delta(s: float) -> tuple[float, float]:
        if s <= 0:
            s = 1e-3
        res = binomial_american(spot=s, strike=strike, T=T, r=r, q=q,
                                sigma=sigma, opt=opt, n_steps=n_steps,
                                method="lr")
        return res["price"], res["delta"]

    s = max(1e-3, spot_hint)
    for _ in range(20):
        v, d = price_and_delta(s)
        diff = v - target_premium
        if abs(diff) < 1e-4:
            return s
        if abs(d) < 1e-10:
            break  # delta degenerate — fall to bisection
        s_new = s - diff / d
        if s_new <= 0 or s_new > strike * 50.0:
            break  # diverged — fall to bisection
        s = s_new

    # Bisection fallback. value(S) is monotonic in S: decreasing for puts,
    # increasing for calls. Use a wide bracket and a sign-change check.
    lo, hi = 1e-3, max(strike, target_premium) * 50.0
    v_lo, _ = price_and_delta(lo)
    v_hi, _ = price_and_delta(hi)
    f_lo, f_hi = v_lo - target_premium, v_hi - target_premium
    if f_lo * f_hi > 0:
        return None  # no sign change → no solution in bracket
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v_mid, _ = price_and_delta(mid)
        f_mid = v_mid - target_premium
        if abs(f_mid) < 1e-4:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def evaluate_hedge(
    hedge: Hedge,
    spot_today: float,
    sigma_today: float,
    r: float,
    q: float,
    today: date | None = None,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    n_steps: int = 200,
    vol_baseline: Literal["contract", "atm"] = "contract",
    sigma_atm: float | None = None,
) -> HedgeEvaluation:
    """Evaluate `hedge` against today's market and a sequence of stress
    scenarios. Returns a fully-populated HedgeEvaluation.

    Parameters
    ----------
    spot_today, sigma_today
        Current underlying spot and implied vol (decimal, e.g. 0.20).
    r, q
        Continuously-compounded risk-free rate and continuous dividend
        yield (both decimal). The caller resolves these (e.g. with
        implied_dividend.solve_q for q, FRED 3-mo T-bill for r).
    today
        Defaults to date.today(). DTE = (hedge.expiry - today).days.
    scenarios
        Defaults to DEFAULT_SCENARIOS. Pass your own sequence to override.
    n_steps
        LR tree step count. Default 200 matches the pricer's default; you
        can bump for high-precision sweeps.
    vol_baseline
        "contract" (default) shocks the contract's own IV directly.
        "atm" shocks the ATM IV and re-applies an absolute-vol skew —
        see module docstring for the math. Requires `sigma_atm`.
    sigma_atm
        Today's at-the-money IV for the same expiry (decimal). Required
        when vol_baseline="atm"; ignored otherwise.
    """
    if vol_baseline == "atm":
        if sigma_atm is None:
            raise ValueError(
                'vol_baseline="atm" requires sigma_atm (today\'s ATM IV '
                "for the same expiry). Pass it explicitly or use "
                'vol_baseline="contract".'
            )
        if sigma_atm <= 0:
            raise ValueError(f"sigma_atm must be positive, got {sigma_atm}")
    elif vol_baseline != "contract":
        raise ValueError(
            f'vol_baseline must be "contract" or "atm", got {vol_baseline!r}'
        )
    today_d = today or date.today()
    days_to_expiry = (hedge.expiry - today_d).days
    if days_to_expiry < 0:
        raise ValueError(
            f"hedge.expiry {hedge.expiry} is before today {today_d}"
        )
    T = days_to_expiry / 365.0

    notional = hedge.n_contracts * CONTRACT_MULT * spot_today
    upfront  = hedge.n_contracts * CONTRACT_MULT * hedge.premium_per_share
    cost_pct = upfront / notional if notional > 0 else float("nan")

    if hedge.option_type == "put":
        breakeven_spot = hedge.strike - hedge.premium_per_share
    else:
        breakeven_spot = hedge.strike + hedge.premium_per_share
    breakeven_decline_pct = (
        (spot_today - breakeven_spot) / spot_today if spot_today > 0 else float("nan")
    )

    base = binomial_american(
        spot=spot_today, strike=hedge.strike, T=T, r=r, q=q,
        sigma=sigma_today, opt=hedge.option_type, n_steps=n_steps, method="lr",
    )

    mtm_be = _solve_mtm_breakeven_spot(
        strike=hedge.strike, opt=hedge.option_type, T=T, r=r, q=q,
        sigma=sigma_today, target_premium=hedge.premium_per_share,
        n_steps=n_steps, spot_hint=spot_today,
    )
    mtm_be_decline = (
        (spot_today - mtm_be) / spot_today
        if (mtm_be is not None and spot_today > 0) else None
    )

    # Absolute (vol-point) skew vs ATM. Only used in "atm" mode.
    skew = (sigma_today - sigma_atm) if vol_baseline == "atm" else 0.0

    results: list[ScenarioResult] = []
    for sc in scenarios:
        s_shock = spot_today * sc.spot_mult
        if vol_baseline == "atm":
            # ATM IV stressed multiplicatively, contract IV = stressed ATM + skew.
            # Floor at a tiny positive value so the pricer never sees sigma<=0
            # (would happen only if the user passes a sigma_atm > sigma_today
            # combined with a vol_mult < 1, i.e. a vol-collapse scenario on an
            # ITM contract — exotic, but still want graceful behavior).
            sig_shock = max(1e-4, sigma_atm * sc.vol_mult + skew)
        else:
            sig_shock = sigma_today * sc.vol_mult
        priced = binomial_american(
            spot=s_shock, strike=hedge.strike, T=T, r=r, q=q,
            sigma=sig_shock, opt=hedge.option_type, n_steps=n_steps, method="lr",
        )
        v = priced["price"]
        pnl_per_share = v - hedge.premium_per_share
        pnl_total     = hedge.n_contracts * CONTRACT_MULT * pnl_per_share
        pnl_pct = pnl_total / notional if notional > 0 else float("nan")
        results.append(ScenarioResult(
            scenario=sc,
            shocked_spot=s_shock,
            shocked_vol=sig_shock,
            repriced_per_share=v,
            pnl_per_share=pnl_per_share,
            pnl_total=pnl_total,
            pnl_pct_of_notional=pnl_pct,
        ))

    return HedgeEvaluation(
        hedge=hedge,
        spot_today=spot_today,
        sigma_today=sigma_today,
        r=r, q=q, T_years=T,
        base_per_share=base["price"],
        upfront_cost=upfront,
        notional_protected=notional,
        cost_of_insurance_pct=cost_pct,
        breakeven_spot=breakeven_spot,
        breakeven_decline_pct=breakeven_decline_pct,
        mtm_breakeven_spot=mtm_be,
        mtm_breakeven_decline_pct=mtm_be_decline,
        scenarios=tuple(results),
        vol_baseline=vol_baseline,
        sigma_atm=sigma_atm if vol_baseline == "atm" else None,
    )
