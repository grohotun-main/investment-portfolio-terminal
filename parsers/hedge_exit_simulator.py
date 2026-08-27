"""Exit-rule back-test simulator for a rolling SPY put hedge program.

Closes the Phase F gap from the original options ask: which exit rule
actually achieves a max-loss target with what drag? PR #94's back-test
only reprices puts the portfolio has actually held; this module simulates
mechanically-rolled programs against a synthetic put universe (built by
``fetch_synthetic_put_grid.py``) so we can compare candidate exit rules.

Design
------
A simulation has three moving parts:

1. ``HedgePolicy`` — the parametric program (target DTE, target moneyness,
   notional protected). The simulator opens legs that match these targets.
2. ``ExitRule`` — a predicate that decides, on each day, whether a leg
   should close. Three implementations cover the philosophy space:
     * ``dte_roll``        — close at DTE ≤ threshold, immediately roll
     * ``monetize_recovery`` — close when SPY rallies X% off trough toward
                                  prior peak, then stay flat until SPY drops
                                  Y% from a new peak (re-entry trigger)
     * ``profit_take_mult`` — close when sleeve MV ≥ N × premium paid
3. Entry policy — for DTE-roll/profit-take rules, a new leg opens
   immediately when no leg is held. For monetize_recovery, re-entry is
   gated on a new drawdown trigger.

The simulator walks forward day-by-day, looks up close prices from the
cache, and records a daily ledger of sleeve MV, premium flows, and
realized P&L. Output feeds the metric/Pareto stage.

Why not Monte Carlo here
------------------------
This module runs deterministic back-tests on real historical data. The
bootstrap layer (separate module, coming next) wraps this simulator,
resamples (SPY return, IV-change) blocks, and re-runs N times to
generate confidence intervals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd

CONTRACT_MULT = 100  # standard equity-options multiplier

# Default exit-rule parameters. Overridable per-run via simulate_program kwargs.
DEFAULT_DTE_ROLL_THRESHOLD = 30
DEFAULT_MONETIZE_RECOVERY_FRAC = 0.50  # close when SPY recovers ≥50% off trough
DEFAULT_MONETIZE_REENTRY_DRAWDOWN = 0.05  # re-enter after SPY drops ≥5% off new peak
# Guards added after the sample-run investigation surfaced May-2025 churn — 3
# legs opened+closed within 12 days each, all at a loss totaling -$110k. Cause:
# with peak anchored at a prior cycle's high, small dips/bounces easily hit the
# 50%-recovery trigger before the leg has appreciated. These two guards block
# monetize from firing on (a) immature legs (b) loss-position legs.
DEFAULT_MONETIZE_MIN_HOLD_DAYS = 14
DEFAULT_MONETIZE_MIN_PROFIT_MULT = 1.0  # leg MV must be ≥ premium paid
DEFAULT_PROFIT_TAKE_MULT = 3.0


# --- Data classes -----------------------------------------------------------

@dataclass
class HedgePolicy:
    """Parametric definition of the rolling hedge program."""
    target_dte: int = 90
    target_moneyness: float = 0.05     # (spot − strike) / spot, positive for OTM
    notional_protected: float = 100_000.0  # $ value of SPY exposure to hedge


@dataclass
class Leg:
    """One open or closed option position in the sleeve."""
    open_date: pd.Timestamp
    ticker: str
    underlying: str
    expiry: date
    strike: float
    contracts: int                      # whole contracts (multiplied by 100 for $ math)
    premium_paid: float                 # per share at open ($/share, positive)
    close_date: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None  # per share at close ($/share)
    close_reason: Optional[str] = None   # "expiry" | "dte_roll" | "monetize" | "profit_take"

    def is_open(self) -> bool:
        return self.close_date is None

    def cost_basis_total(self) -> float:
        return self.premium_paid * self.contracts * CONTRACT_MULT

    def realized_proceeds(self) -> float:
        if self.close_price is None:
            return 0.0
        return self.close_price * self.contracts * CONTRACT_MULT

    def realized_pnl(self) -> float:
        return self.realized_proceeds() - self.cost_basis_total()


@dataclass
class SimState:
    """Day-level state passed to exit-rule predicates."""
    today: pd.Timestamp
    spot: float                          # SPY close on ``today``
    peak_spot: float                     # running maximum spot since back-test start (or last reset)
    trough_spot: float                   # running minimum spot since peak (= spot if rallying)
    leg_mv_per_share: float              # current option close per share
    leg_total_mv: float                  # leg.contracts × 100 × close
    flatted: bool = False                # True if monetize rule exited and not yet re-entered
    flatted_at_peak: Optional[float] = None  # peak_spot at the moment of flattening
    iv_rank_today: float = float("nan")  # supplied per-day when an IV-rank-based rule is active


# --- Exit rules -------------------------------------------------------------
#
# Each rule returns the close_reason string when the leg should close on
# ``today``, or None to leave it open.

def rule_dte_roll(
    leg: Leg, state: SimState, *, dte_threshold: int = DEFAULT_DTE_ROLL_THRESHOLD,
) -> Optional[str]:
    """Close when DTE drops at/below the threshold."""
    dte = (leg.expiry - state.today.date()).days
    if dte <= 0:
        return "expiry"
    if dte <= dte_threshold:
        return "dte_roll"
    return None


def rule_monetize_recovery(
    leg: Leg, state: SimState,
    *, recovery_frac: float = DEFAULT_MONETIZE_RECOVERY_FRAC,
    min_hold_days: int = DEFAULT_MONETIZE_MIN_HOLD_DAYS,
    min_profit_mult: float = DEFAULT_MONETIZE_MIN_PROFIT_MULT,
) -> Optional[str]:
    """Close when SPY rallies recovery_frac × drawdown back toward the peak,
    subject to two guards that block premature/loss-taking closes.

    Example: peak=600, trough=540 (10% DD). recovery_frac=0.5 → close when
    SPY ≥ 540 + 0.5×(600−540) = 570 (=5% off peak).

    Guards (added after May-2025 churn investigation):
      * min_hold_days: leg must have been open ≥ this many days before
        monetize can fire. Blocks short-life chop legs from being closed
        on micro-recoveries before they appreciate.
      * min_profit_mult: leg current MV per share must be ≥ premium_paid ×
        this multiplier. Blocks "monetize at a loss" — only fire when the
        leg has actually appreciated.

    Falls back to expiry close if the trigger never fires.
    """
    if state.today.date() >= leg.expiry:
        return "expiry"
    if state.peak_spot <= 0 or state.trough_spot >= state.peak_spot:
        return None
    drawdown = state.peak_spot - state.trough_spot
    # Only fire if we actually saw a meaningful drawdown — otherwise this
    # would close on the first up day. Require ≥2% DD from peak to engage.
    if drawdown / state.peak_spot < 0.02:
        return None
    recovery_level = state.trough_spot + recovery_frac * drawdown
    if state.spot < recovery_level:
        return None
    # Guard A: leg must be sufficiently mature.
    days_held = (state.today - leg.open_date).days
    if days_held < min_hold_days:
        return None
    # Guard B: leg must currently be in profit.
    if state.leg_mv_per_share < leg.premium_paid * min_profit_mult:
        return None
    return "monetize"


def rule_profit_take(
    leg: Leg, state: SimState, *, mult: float = DEFAULT_PROFIT_TAKE_MULT,
) -> Optional[str]:
    """Close when current leg MV ≥ mult × cost basis."""
    if state.today.date() >= leg.expiry:
        return "expiry"
    if leg.premium_paid <= 0:
        return None
    if state.leg_mv_per_share >= mult * leg.premium_paid:
        return "profit_take"
    return None


DEFAULT_EMPIRICAL_PCT_RHIGH = 80.0
DEFAULT_EMPIRICAL_PCT_RLOW = 30.0


def rule_empirical_percentile(
    leg: Leg, state: SimState,
    *, r_high: float = DEFAULT_EMPIRICAL_PCT_RHIGH,
    r_low: float = DEFAULT_EMPIRICAL_PCT_RLOW,
) -> Optional[str]:
    """Close when SPY ATM IV percentile (rank-order in trailing 252d) is
    empirically rich. Re-entry after a flatted period is handled by
    ``simulate_program`` reading ``state.flatted`` and ``r_low``.

    Safeties:
      * Expiry close fires regardless of rank (matches profit_take pattern).
      * NaN rank (no history) does not fire — leg stays open.
    """
    if state.today.date() >= leg.expiry:
        return "expiry"
    if state.iv_rank_today is None or math.isnan(state.iv_rank_today):
        return None
    if state.iv_rank_today >= r_high:
        return "empirical_pct"
    return None


EXIT_RULES: dict[str, Callable[[Leg, SimState], Optional[str]]] = {
    "dte_roll":        rule_dte_roll,
    "monetize":        rule_monetize_recovery,
    "profit_take_3x":  rule_profit_take,
    "empirical_pct":   rule_empirical_percentile,
}


# --- Cache lookup helpers ---------------------------------------------------

def build_grid_index(option_grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index the grid CSV by contract_ticker for O(1) per-leg reprice lookup."""
    if option_grid is None or option_grid.empty:
        return {}
    df = option_grid.copy()
    df["date"] = pd.to_datetime(df["date"])
    out: dict[str, pd.DataFrame] = {}
    for tk, grp in df.groupby("contract_ticker"):
        out[tk] = grp[["date", "close"]].sort_values("date").reset_index(drop=True)
    return out


def lookup_close(grid_index: dict[str, pd.DataFrame], ticker: str,
                 when: pd.Timestamp) -> Optional[float]:
    """Look up the contract's close on ``when``, forward-filling across
    weekends / zero-volume days. Returns None when no bar exists ≤ when.
    """
    df = grid_index.get(ticker)
    if df is None or df.empty:
        return None
    s = df[df["date"] <= when]
    if s.empty:
        return None
    return float(s.iloc[-1]["close"])


def resolve_from_cache(
    option_grid: pd.DataFrame, as_of: date, spot: float,
    target_dte: int, target_moneyness: float,
    *,
    grid_index: Optional[dict[str, pd.DataFrame]] = None,
    dte_window: tuple[int, int] = (15, 60),
    moneyness_window: float = 0.20,
) -> Optional[dict]:
    """Find the cached contract closest to (target_dte, target_moneyness)
    on ``as_of``, restricted to contracts that have a close on ``as_of``.

    Returns dict with ticker / strike / expiry, or None if no contract
    in the cache matches.
    """
    if option_grid is None or option_grid.empty:
        return None
    if grid_index is None:
        grid_index = build_grid_index(option_grid)
    as_of_ts = pd.Timestamp(as_of)
    # All cached contracts (one row per unique ticker).
    meta = (option_grid[["contract_ticker", "underlying", "opt_type",
                         "expiry", "strike"]]
            .drop_duplicates()
            .reset_index(drop=True))
    meta["expiry"] = pd.to_datetime(meta["expiry"])

    best, best_d = None, float("inf")
    for _, r in meta.iterrows():
        exp_d = r["expiry"].date()
        dte = (exp_d - as_of).days
        if dte <= 0:
            continue
        if dte < target_dte - dte_window[0] or dte > target_dte + dte_window[1]:
            continue
        K = float(r["strike"])
        moneyness = (spot - K) / spot
        if abs(moneyness - target_moneyness) > moneyness_window:
            continue
        # Require the contract has a close on or before as_of (no future-pricing).
        if lookup_close(grid_index, r["contract_ticker"], as_of_ts) is None:
            continue
        d = abs(dte - target_dte) + 10.0 * abs(moneyness - target_moneyness) * 100
        if d < best_d:
            best_d = d
            best = {
                "ticker": r["contract_ticker"],
                "underlying": r["underlying"],
                "opt_type": r["opt_type"],
                "expiry": exp_d,
                "strike": K,
            }
    return best


# --- Sizing -----------------------------------------------------------------

def contracts_for_notional(
    notional_protected: float, strike: float, *, mult: int = CONTRACT_MULT,
) -> int:
    """Number of put contracts required to cover ``notional_protected`` $.

    Sizing convention: 1 put contract protects ``strike × mult`` of
    underlying-equivalent notional (max payoff at strike=0). For a SPY put
    struck at 540, 1 contract protects $54,000.

    Returns at least 1 contract.
    """
    per_contract = strike * mult
    if per_contract <= 0:
        return 1
    return max(1, int(round(notional_protected / per_contract)))


# --- Main simulator ---------------------------------------------------------

def simulate_program(
    policy: HedgePolicy,
    exit_rule_name: str,
    spy_history: pd.DataFrame,
    option_grid: pd.DataFrame,
    *,
    start: date,
    end: date,
    monetize_reentry_drawdown: float = DEFAULT_MONETIZE_REENTRY_DRAWDOWN,
    rule_kwargs: Optional[dict] = None,
    iv_rank_series: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[Leg]]:
    """Walk [start, end] day-by-day, simulating one leg at a time.

    Returns
    -------
    (ledger, legs):
        ledger: DataFrame [date, spy_close, sleeve_mv, premium_paid_today,
                premium_received_today, cum_premium_paid, cum_premium_received,
                realized_pnl_to_date, n_open_legs, leg_ticker, action]
        legs: list of all Leg objects (open + closed) produced by the run

    The simulator holds at most ONE leg at a time for v1 simplicity. A
    "rolling ladder" of multiple overlapping legs is a Phase F.1
    extension — gets us most of the way there without the bookkeeping
    complexity of overlapping positions.
    """
    if exit_rule_name not in EXIT_RULES:
        raise ValueError(
            f"Unknown exit_rule_name {exit_rule_name!r}. "
            f"Available: {sorted(EXIT_RULES.keys())}"
        )
    rule_fn = EXIT_RULES[exit_rule_name]
    rule_kwargs = rule_kwargs or {}

    spy = spy_history.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy[(spy["date"] >= pd.Timestamp(start)) & (spy["date"] <= pd.Timestamp(end))]
    spy = spy.sort_values("date").reset_index(drop=True)
    if spy.empty:
        raise ValueError(f"No SPY history in [{start}, {end}].")

    grid_index = build_grid_index(option_grid)

    # Daily IV-rank lookup (forward-filled). Only consulted by rank-based rules.
    if iv_rank_series is not None and not iv_rank_series.empty:
        rk = iv_rank_series.copy()
        rk["date"] = pd.to_datetime(rk["date"])
        rk = rk.sort_values("date").set_index("date")
    else:
        rk = None

    # State.
    open_leg: Optional[Leg] = None
    legs: list[Leg] = []
    peak_spot = float(spy.iloc[0]["close"])
    trough_spot = peak_spot
    flatted = False
    flatted_at_peak: Optional[float] = None
    cum_premium_paid = 0.0
    cum_premium_received = 0.0
    realized_pnl = 0.0
    ledger_rows: list[dict] = []

    for _, day_row in spy.iterrows():
        today = day_row["date"]
        spot = float(day_row["close"])

        # Update peak / trough.
        if spot > peak_spot:
            peak_spot = spot
            trough_spot = spot
            # If we were flatted, check re-entry trigger from the new peak.
            # We don't re-enter on a new peak day itself — we wait for a drawdown.
        elif spot < trough_spot:
            trough_spot = spot

        # Compute today's IV rank (forward-filled). Used by rank-based rules
        # for both the entry/close decision and the re-entry trigger below.
        iv_rank_today = float("nan")
        if rk is not None and not rk.empty:
            past = rk[rk.index <= today]
            if not past.empty:
                iv_rank_today = float(past.iloc[-1]["rank"])

        # Re-entry checks (per-rule).
        if flatted and exit_rule_name == "monetize":
            # Re-enter when SPY drops by monetize_reentry_drawdown off the new peak.
            if (peak_spot - spot) / peak_spot >= monetize_reentry_drawdown:
                flatted = False
                flatted_at_peak = None
                # Also reset trough to today's spot so the next "recovery" calculation
                # is anchored on this drawdown's start, not the prior one.
                trough_spot = spot
        elif flatted and exit_rule_name == "empirical_pct":
            # Re-enter when IV rank drops back to/under r_low.
            r_low = (rule_kwargs or {}).get("r_low", DEFAULT_EMPIRICAL_PCT_RLOW)
            if not math.isnan(iv_rank_today) and iv_rank_today <= r_low:
                flatted = False

        premium_paid_today = 0.0
        premium_received_today = 0.0
        action = ""

        # 1) Reprice open leg if any. Close it if exit rule fires.
        if open_leg is not None:
            leg_mv_per_share = lookup_close(grid_index, open_leg.ticker, today)
            if leg_mv_per_share is None:
                # No data on this day — keep leg open, carry yesterday's MV as 0
                # for ledger purposes. Should be rare for SPY puts.
                leg_total_mv = 0.0
                leg_mv_per_share = 0.0
            else:
                leg_total_mv = leg_mv_per_share * open_leg.contracts * CONTRACT_MULT

            state = SimState(
                today=today, spot=spot, peak_spot=peak_spot, trough_spot=trough_spot,
                leg_mv_per_share=leg_mv_per_share, leg_total_mv=leg_total_mv,
                flatted=flatted, flatted_at_peak=flatted_at_peak,
                iv_rank_today=iv_rank_today,
            )
            reason = rule_fn(open_leg, state, **rule_kwargs) if rule_kwargs \
                else rule_fn(open_leg, state)
            if reason is not None:
                # Close it. At expiry, intrinsic value = max(strike − spot, 0).
                if reason == "expiry":
                    close_per_share = max(open_leg.strike - spot, 0.0)
                else:
                    close_per_share = leg_mv_per_share
                open_leg.close_date = today
                open_leg.close_price = close_per_share
                open_leg.close_reason = reason
                proceeds = close_per_share * open_leg.contracts * CONTRACT_MULT
                premium_received_today += proceeds
                cum_premium_received += proceeds
                realized_pnl += open_leg.realized_pnl()
                action = f"close[{reason}]:{open_leg.ticker}"
                # Flag flatted state for any rule that monetizes vol/spot richness.
                if exit_rule_name == "monetize" and reason == "monetize":
                    flatted = True
                    flatted_at_peak = peak_spot
                elif exit_rule_name == "empirical_pct" and reason == "empirical_pct":
                    flatted = True
                    flatted_at_peak = None  # unused for this rule
                open_leg = None  # cleared; entry pass below may re-open

        # 2) Entry pass — open a new leg unless we're in "flatted" mode.
        if open_leg is None and not flatted:
            pick = resolve_from_cache(
                option_grid, today.date(), spot,
                policy.target_dte, policy.target_moneyness,
                grid_index=grid_index,
            )
            if pick is not None:
                premium_per_share = lookup_close(grid_index, pick["ticker"], today)
                if premium_per_share is not None and premium_per_share > 0:
                    contracts = contracts_for_notional(
                        policy.notional_protected, pick["strike"],
                    )
                    open_leg = Leg(
                        open_date=today,
                        ticker=pick["ticker"],
                        underlying=pick["underlying"],
                        expiry=pick["expiry"],
                        strike=pick["strike"],
                        contracts=contracts,
                        premium_paid=premium_per_share,
                    )
                    legs.append(open_leg)
                    cost = open_leg.cost_basis_total()
                    premium_paid_today += cost
                    cum_premium_paid += cost
                    if action:
                        action = action + "; "
                    action = action + f"open:{pick['ticker']}@{premium_per_share:.2f}"

        # 3) Ledger row.
        sleeve_mv = 0.0
        leg_ticker = ""
        if open_leg is not None:
            mv_ps = lookup_close(grid_index, open_leg.ticker, today)
            if mv_ps is not None:
                sleeve_mv = mv_ps * open_leg.contracts * CONTRACT_MULT
            leg_ticker = open_leg.ticker
        ledger_rows.append({
            "date": today,
            "spy_close": spot,
            "sleeve_mv": sleeve_mv,
            "premium_paid_today": premium_paid_today,
            "premium_received_today": premium_received_today,
            "cum_premium_paid": cum_premium_paid,
            "cum_premium_received": cum_premium_received,
            "realized_pnl_to_date": realized_pnl,
            "n_open_legs": 1 if open_leg is not None else 0,
            "leg_ticker": leg_ticker,
            "action": action,
            "peak_spot": peak_spot,
            "trough_spot": trough_spot,
            "flatted": flatted,
        })

    ledger = pd.DataFrame(ledger_rows)
    return ledger, legs


# --- Per-run metrics --------------------------------------------------------

def summarize_run(ledger: pd.DataFrame, legs: list[Leg],
                  policy: HedgePolicy) -> dict:
    """Compute headline metrics from a single back-test run.

    Returns dict with:
      total_premium_paid, total_premium_received, net_cost,
      annualized_drag_pct, n_trades, avg_dte_at_open,
      avg_moneyness_at_open, sleeve_mv_final.
    """
    total_paid = float(ledger["premium_paid_today"].sum())
    total_recv = float(ledger["premium_received_today"].sum())
    net_cost = total_paid - total_recv  # positive = drag
    days = (ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days
    years = days / 365.25 if days > 0 else 1.0
    drag_pct = (net_cost / policy.notional_protected) / years * 100 if policy.notional_protected > 0 else 0.0
    return {
        "total_premium_paid": total_paid,
        "total_premium_received": total_recv,
        "net_cost": net_cost,
        "annualized_drag_pct": drag_pct,
        "n_trades": len(legs),
        "n_open_at_end": int(ledger["n_open_legs"].iloc[-1]),
        "sleeve_mv_final": float(ledger["sleeve_mv"].iloc[-1]),
        "window_days": int(days),
    }


# --- Multi-rule comparison + Pareto ----------------------------------------

def episode_payoffs(
    ledger: pd.DataFrame, episodes: pd.DataFrame,
) -> pd.DataFrame:
    """For each SPY drawdown episode, look up sleeve cumulative-realized-pnl
    + sleeve_mv at peak, trough, recover. Sleeve "gain" during the episode is

        (sleeve_mv_trough + cum_realized_at_trough) − (sleeve_mv_peak + cum_realized_at_peak)

    i.e. the change in sleeve VALUE (open MV + realized P&L) from peak to
    trough. This correctly attributes:
      * Open-leg MV gains if the leg is still held into the trough
      * Realized P&L from any close that happened mid-episode (e.g. monetize)

    Returns episodes_df with new columns: peak_sleeve_value, trough_sleeve_value,
    recover_sleeve_value, sleeve_gain_peak_to_trough, sleeve_gain_pct
    (gain as % of peak SPY × notional_protected — i.e. payoff vs. the
    notional being hedged).

    Missing-data behavior: lookup falls back to the most recent earlier
    ledger row, matching the back-test's forward-fill convention.
    """
    if episodes is None or episodes.empty or ledger.empty:
        return episodes.copy() if episodes is not None else pd.DataFrame()
    led = ledger.copy()
    led["date"] = pd.to_datetime(led["date"])
    led = led.sort_values("date").reset_index(drop=True)
    led["sleeve_value"] = led["sleeve_mv"] + led["realized_pnl_to_date"]

    def lookup(when) -> Optional[float]:
        if when is None or pd.isna(when):
            return None
        ts = pd.Timestamp(when)
        m = led[led["date"] <= ts]
        if m.empty:
            return None
        return float(m.iloc[-1]["sleeve_value"])

    out = episodes.copy().reset_index(drop=True)
    out["peak_sleeve_value"] = [lookup(r["peak_date"]) for _, r in out.iterrows()]
    out["trough_sleeve_value"] = [lookup(r["trough_date"]) for _, r in out.iterrows()]
    out["recover_sleeve_value"] = [
        lookup(r["recover_date"]) if pd.notna(r["recover_date"]) else None
        for _, r in out.iterrows()
    ]
    out["sleeve_gain_peak_to_trough"] = (
        pd.Series(out["trough_sleeve_value"], dtype="float64")
        - pd.Series(out["peak_sleeve_value"], dtype="float64")
    )
    return out


def compare_runs(
    runs: dict[str, tuple[pd.DataFrame, list[Leg]]],
    episodes: pd.DataFrame, policy: HedgePolicy,
) -> pd.DataFrame:
    """Side-by-side comparison of multiple exit rules.

    Args:
        runs: {rule_name: (ledger, legs)} keyed by exit-rule string.
        episodes: SPY drawdown episodes (output of
            ``hedge_effectiveness.find_drawdown_episodes``) — used to
            compute per-episode sleeve payoff.
        policy: the (shared) HedgePolicy used for all runs.

    Returns DataFrame with one row per rule, columns:
        rule, total_premium_paid, total_premium_received, net_cost,
        annualized_drag_pct, n_trades,
        mean_episode_payoff, median_episode_payoff, total_episode_payoff,
        payoff_per_dollar_drag, n_episodes_covered.
    """
    rows: list[dict] = []
    for rule_name, (ledger, legs) in runs.items():
        s = summarize_run(ledger, legs, policy)
        ep = episode_payoffs(ledger, episodes)
        gains = pd.Series(ep["sleeve_gain_peak_to_trough"], dtype="float64").dropna() \
            if not ep.empty else pd.Series(dtype="float64")
        # Convert per-episode $ to % of notional protected.
        gain_pct = gains / policy.notional_protected * 100 if policy.notional_protected > 0 \
            else gains
        mean_pay = float(gain_pct.mean()) if not gain_pct.empty else 0.0
        med_pay = float(gain_pct.median()) if not gain_pct.empty else 0.0
        total_pay = float(gain_pct.sum()) if not gain_pct.empty else 0.0
        # Payoff-per-dollar-drag = sum of payoff $ / total premium burn $.
        # Positive ratio means we got back $X for every $1 of drag.
        payoff_per_drag = (float(gains.sum()) / s["net_cost"]) if s["net_cost"] > 0 else float("nan")
        rows.append({
            "rule": rule_name,
            "total_premium_paid": s["total_premium_paid"],
            "total_premium_received": s["total_premium_received"],
            "net_cost": s["net_cost"],
            "annualized_drag_pct": s["annualized_drag_pct"],
            "n_trades": s["n_trades"],
            "mean_episode_payoff_pct": mean_pay,
            "median_episode_payoff_pct": med_pay,
            "total_episode_payoff_pct": total_pay,
            "payoff_per_dollar_drag": payoff_per_drag,
            "n_episodes_covered": int(gain_pct.notna().sum()) if not gain_pct.empty else 0,
        })
    return pd.DataFrame(rows)


def pareto_frontier_mask(
    df: pd.DataFrame,
    drag_col: str = "annualized_drag_pct",
    payoff_col: str = "total_episode_payoff_pct",
) -> pd.Series:
    """Mark which rules sit on the Pareto frontier (low drag, high payoff).

    A rule is Pareto-optimal if no other rule has BOTH:
      * drag_col strictly lower (less drag is better), AND
      * payoff_col strictly higher (more payoff is better).

    Returns boolean Series aligned with df.index.
    """
    if df.empty:
        return pd.Series(dtype=bool)
    mask: list[bool] = []
    for i, r in df.iterrows():
        dominated = False
        for j, r2 in df.iterrows():
            if i == j:
                continue
            if (r2[drag_col] < r[drag_col]) and (r2[payoff_col] > r[payoff_col]):
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)
