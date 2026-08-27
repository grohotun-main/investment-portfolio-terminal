"""Options pricer — Black-Scholes (European) + Cox-Ross-Rubinstein binomial
tree (American) — plus an implied-volatility solver.

This module is pure-Python (numpy + stdlib math; no scipy, no network). It is
the math engine for Phase A of the hedging work. Phase B will use it inside a
Monte-Carlo simulator; Phase C will surface it on a Streamlit Hedging tab.

All pricing functions return a dict with the same keys:
    {"price", "delta", "gamma", "vega", "theta", "rho"}

Conventions
-----------
    spot     dollars
    strike   dollars
    T        time to expiry, in YEARS (e.g. 30 calendar days = 30/365)
    r        continuously-compounded risk-free rate, decimal (0.05 = 5%)
    q        continuous dividend yield, decimal
    sigma    annualized volatility, decimal (0.20 = 20%)
    opt      "call" or "put"
    exercise "european" or "american"

Greek units (RAW — natural mathematical units, no market-display rescaling):
    delta    dV/dS                 per $1 of spot
    gamma    d2V/dS2               per $1 of spot, per $1
    vega     dV/dsigma             per 1.00 vol unit  (per "percent point" = vega * 0.01)
    theta    -dV/d(time-to-exp)    per YEAR           (per calendar day = theta / 365)
    rho      dV/dr                 per 1.00 rate unit (per "percent point" = rho * 0.01)

Polygon's snapshot Greeks use market display units (vega per 1% vol, theta per
calendar day). The verification script handles that conversion — keep this
module in textbook units so the math stays auditable.

Formulas: Hull, "Options, Futures, and Other Derivatives", ch. 14-19.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

OptType = Literal["call", "put"]
ExStyle = Literal["european", "american"]

DEFAULT_N_STEPS = 200       # CRR steps — ~5e-3 abs error on ATM puts vs n=2000
DEFAULT_IV_TOL  = 1e-6
MAX_IV_ITERS    = 60


# ---------- normal CDF / PDF (avoid scipy) -------------------------------

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------- expiry / zero-vol degenerate cases --------------------------

def _at_expiry(spot: float, strike: float, opt: OptType) -> dict[str, float]:
    intrinsic = max(spot - strike, 0.0) if opt == "call" else max(strike - spot, 0.0)
    if opt == "call":
        delta = 1.0 if spot > strike else 0.0
    else:
        delta = -1.0 if spot < strike else 0.0
    return {"price": intrinsic, "delta": delta, "gamma": 0.0,
            "vega": 0.0, "theta": 0.0, "rho": 0.0}


def _zero_vol(spot: float, strike: float, T: float, r: float, q: float,
              opt: OptType) -> dict[str, float]:
    fwd = spot * math.exp(-q * T)
    disc_K = strike * math.exp(-r * T)
    val = max(fwd - disc_K, 0.0) if opt == "call" else max(disc_K - fwd, 0.0)
    return {"price": val, "delta": 0.0, "gamma": 0.0,
            "vega": 0.0, "theta": 0.0, "rho": 0.0}


# ---------- Black-Scholes (European, closed-form) ------------------------

def black_scholes(spot: float, strike: float, T: float, r: float, q: float,
                  sigma: float, opt: OptType) -> dict[str, float]:
    """European Black-Scholes: price + delta, gamma, vega, theta, rho."""
    if T <= 0.0:
        return _at_expiry(spot, strike, opt)
    if sigma <= 0.0:
        return _zero_vol(spot, strike, T, r, q, opt)

    sqrtT  = math.sqrt(T)
    d1     = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2     = d1 - sigma * sqrtT
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    nd1    = _npdf(d1)

    if opt == "call":
        Nd1, Nd2 = _ncdf(d1), _ncdf(d2)
        price = spot * disc_q * Nd1 - strike * disc_r * Nd2
        delta = disc_q * Nd1
        theta = (-spot * nd1 * sigma * disc_q / (2.0 * sqrtT)
                 - r * strike * disc_r * Nd2
                 + q * spot * disc_q * Nd1)
        rho   = strike * T * disc_r * Nd2
    elif opt == "put":
        Nmd1, Nmd2 = _ncdf(-d1), _ncdf(-d2)
        price = strike * disc_r * Nmd2 - spot * disc_q * Nmd1
        delta = -disc_q * Nmd1
        theta = (-spot * nd1 * sigma * disc_q / (2.0 * sqrtT)
                 + r * strike * disc_r * Nmd2
                 - q * spot * disc_q * Nmd1)
        rho   = -strike * T * disc_r * Nmd2
    else:
        raise ValueError(f"opt must be 'call' or 'put', got {opt!r}")

    gamma = disc_q * nd1 / (spot * sigma * sqrtT)
    vega  = spot * disc_q * nd1 * sqrtT

    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


# ---------- Leisen-Reimer binomial tree (American, sharper Greeks) ------

def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method 2 inversion of the standard-normal CDF for a
    binomial tree with `n` steps. This is the trick that lets the LR tree
    align its central terminal node with the strike — the source of LR's
    O(1/n²) Greek convergence (vs CRR's O(1/n) with oscillation).
    """
    if z == 0.0:
        return 0.5
    sign  = 1.0 if z > 0.0 else -1.0
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    expo  = -((z / denom) ** 2) * (n + 1.0 / 6.0)
    h_sq  = max(0.25 - 0.25 * math.exp(expo), 0.0)
    return 0.5 + sign * math.sqrt(h_sq)


def _lr_tree_params(spot: float, strike: float, T: float, r: float, q: float,
                    sigma: float, n_steps: int
                    ) -> tuple[int, float, float, float, float, float] | None:
    """Return (n_eff, dt, u, d, p, p_prime) for an LR tree. n_eff is n_steps
    bumped to the next odd integer. Returns None if numerics degenerate."""
    n_eff = n_steps if n_steps % 2 == 1 else n_steps + 1
    dt    = T / n_eff
    sqrtT = math.sqrt(T)
    d1    = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2    = d1 - sigma * sqrtT
    p_prime = _peizer_pratt(d1, n_eff)
    p       = _peizer_pratt(d2, n_eff)
    if not (0.0 < p < 1.0 and 0.0 < p_prime < 1.0):
        return None
    drift = math.exp((r - q) * dt)
    u = drift * (p_prime / p)
    d = (drift - p * u) / (1.0 - p)
    if u <= 0.0 or d <= 0.0:
        return None
    return n_eff, dt, u, d, p, p_prime


def _lr_price(spot: float, strike: float, T: float, r: float, q: float,
              sigma: float, opt: OptType, n_steps: int,
              want_tree_greeks: bool = False
              ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Price an American option via Leisen-Reimer binomial. Same return
    shape as _crr_price."""
    params = _lr_tree_params(spot, strike, T, r, q, sigma, n_steps)
    if params is None:
        bs = black_scholes(spot, strike, T, r, q, sigma, opt)
        return bs["price"], None, None
    n_eff, dt, u, d, p, _ = params
    disc = math.exp(-r * dt)

    j  = np.arange(n_eff + 1, dtype=np.float64)
    ST = spot * (u ** (n_eff - j)) * (d ** j)
    V  = np.maximum(ST - strike, 0.0) if opt == "call" else np.maximum(strike - ST, 0.0)

    V_at_step1: np.ndarray | None = None
    V_at_step2: np.ndarray | None = None

    for k in range(n_eff - 1, -1, -1):
        V_cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        i      = np.arange(k + 1, dtype=np.float64)
        S_k    = spot * (u ** (k - i)) * (d ** i)
        ex     = np.maximum(S_k - strike, 0.0) if opt == "call" else np.maximum(strike - S_k, 0.0)
        V      = np.maximum(V_cont, ex)
        if want_tree_greeks:
            if k == 2:
                V_at_step2 = V.copy()
            elif k == 1:
                V_at_step1 = V.copy()

    return float(V[0]), V_at_step1, V_at_step2


# ---------- CRR binomial tree (American, legacy) ------------------------

def _crr_price(spot: float, strike: float, T: float, r: float, q: float,
               sigma: float, opt: OptType, n_steps: int,
               want_tree_greeks: bool = False
               ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Price an American option via CRR. If want_tree_greeks, also return the
    step-1 and step-2 value vectors (used for delta/gamma/theta extraction)."""
    dt   = T / n_steps
    u    = math.exp(sigma * math.sqrt(dt))
    d    = 1.0 / u
    disc = math.exp(-r * dt)
    p    = (math.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        # Numerical edge — fall back to European BS (no early-exercise premium)
        # Caller will get a slightly wrong price but won't blow up.
        bs = black_scholes(spot, strike, T, r, q, sigma, opt)
        return bs["price"], None, None

    j  = np.arange(n_steps + 1, dtype=np.float64)
    ST = spot * (u ** (n_steps - j)) * (d ** j)
    V  = np.maximum(ST - strike, 0.0) if opt == "call" else np.maximum(strike - ST, 0.0)

    V_at_step1: np.ndarray | None = None
    V_at_step2: np.ndarray | None = None

    for k in range(n_steps - 1, -1, -1):
        V_cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        i     = np.arange(k + 1, dtype=np.float64)
        S_k   = spot * (u ** (k - i)) * (d ** i)
        if opt == "call":
            ex = np.maximum(S_k - strike, 0.0)
        else:
            ex = np.maximum(strike - S_k, 0.0)
        V = np.maximum(V_cont, ex)
        if want_tree_greeks:
            if k == 2:
                V_at_step2 = V.copy()
            elif k == 1:
                V_at_step1 = V.copy()

    return float(V[0]), V_at_step1, V_at_step2


def binomial_american(spot: float, strike: float, T: float, r: float, q: float,
                      sigma: float, opt: OptType,
                      n_steps: int = DEFAULT_N_STEPS,
                      method: Literal["lr", "crr"] = "lr") -> dict[str, float]:
    """American option pricing via binomial tree.

    method:
      "lr"  Leisen-Reimer (default). Peizer-Pratt inversion aligns the central
            terminal node with the strike, giving O(1/n²) Greek convergence
            without the CRR oscillation. Recommended for OTM Greek work
            (hedge sizing, IV surfaces).
      "crr" Cox-Ross-Rubinstein. Symmetric tree (u·d = 1). Cheaper per node
            but Greeks away from ATM converge slowly — vega on 5-20% OTM puts
            drifts 15-20% even at n=200.

    Delta and gamma come from tree nodes (formula generalizes to non-symmetric
    trees). Theta: CRR exploits u·d=1 to read it off V2[1]; LR uses a 1-day
    time bump (extra reprice). Vega and rho: central finite differences.
    """
    if T <= 0.0:
        return _at_expiry(spot, strike, opt)
    if sigma <= 0.0:
        return _zero_vol(spot, strike, T, r, q, opt)
    if n_steps < 3:
        raise ValueError("n_steps must be >= 3 to extract tree Greeks")

    price_fn = _lr_price if method == "lr" else _crr_price

    price, V1, V2 = price_fn(spot, strike, T, r, q, sigma, opt, n_steps,
                             want_tree_greeks=True)

    if V1 is None or V2 is None:
        # Numerical degeneracy — return BS Greeks
        return black_scholes(spot, strike, T, r, q, sigma, opt)

    if method == "lr":
        params = _lr_tree_params(spot, strike, T, r, q, sigma, n_steps)
        # params can't be None here — _lr_price would have returned (V1=None)
        _, dt, u, d, _, _ = params  # type: ignore[misc]
    else:
        dt = T / n_steps
        u  = math.exp(sigma * math.sqrt(dt))
        d  = 1.0 / u

    # Delta from step-1 nodes: V1[0] at S*u, V1[1] at S*d. Works for any tree.
    S_u, S_d = spot * u, spot * d
    delta = (V1[0] - V1[1]) / (S_u - S_d)

    # Gamma from step-2 nodes (S*u², S*u*d, S*d²). The S_ud term equals spot
    # exactly when u·d = 1 (CRR), so this also reduces to the textbook CRR
    # formula in that case.
    S_uu, S_ud, S_dd = spot * u * u, spot * u * d, spot * d * d
    delta_hi = (V2[0] - V2[1]) / (S_uu - S_ud)
    delta_lo = (V2[1] - V2[2]) / (S_ud - S_dd)
    gamma    = (delta_hi - delta_lo) / (0.5 * (S_uu - S_dd))

    # Theta — convention: dV/d(calendar_time) = -dV/d(time-to-expiry), so
    # long options have theta < 0.
    if method == "crr":
        # V2[1] is at S=spot for CRR; pure theta from the central node
        theta = (V2[1] - price) / (2.0 * dt)
    else:
        # LR: V2[1] is at S=spot*u*d ≠ spot. Use a 1-day time bump instead.
        eps_T = 1.0 / 365.0
        if T > eps_T:
            p_minus, _, _ = price_fn(spot, strike, T - eps_T, r, q, sigma, opt, n_steps)
            theta = -(price - p_minus) / eps_T
        else:
            theta = black_scholes(spot, strike, T, r, q, sigma, opt)["theta"]

    # Vega and rho via central FD (using the chosen tree)
    eps_sigma = 1.0e-3
    p_up_s, _, _ = price_fn(spot, strike, T, r, q, sigma + eps_sigma, opt, n_steps)
    p_dn_s, _, _ = price_fn(spot, strike, T, r, q, sigma - eps_sigma, opt, n_steps)
    vega = (p_up_s - p_dn_s) / (2.0 * eps_sigma)

    eps_r = 1.0e-4
    p_up_r, _, _ = price_fn(spot, strike, T, r + eps_r, q, sigma, opt, n_steps)
    p_dn_r, _, _ = price_fn(spot, strike, T, r - eps_r, q, sigma, opt, n_steps)
    rho = (p_up_r - p_dn_r) / (2.0 * eps_r)

    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


# ---------- Dispatcher ---------------------------------------------------

def price_and_greeks(spot: float, strike: float, T: float, r: float, q: float,
                     sigma: float, opt: OptType,
                     exercise: ExStyle = "european",
                     n_steps: int = DEFAULT_N_STEPS) -> dict[str, float]:
    """Dispatch to Black-Scholes (european) or CRR binomial (american)."""
    if exercise == "european":
        return black_scholes(spot, strike, T, r, q, sigma, opt)
    if exercise == "american":
        return binomial_american(spot, strike, T, r, q, sigma, opt, n_steps)
    raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")


# ---------- Implied-vol solver ------------------------------------------

def implied_vol(market_price: float, spot: float, strike: float, T: float,
                r: float, q: float, opt: OptType,
                exercise: ExStyle = "european",
                initial_guess: float = 0.20,
                tol: float = DEFAULT_IV_TOL,
                max_iters: int = MAX_IV_ITERS,
                n_steps: int = DEFAULT_N_STEPS) -> float:
    """Back out implied volatility from a market price. Newton-Raphson with
    bisection fallback. Returns NaN if no convergence or if market_price is
    below intrinsic (arbitrage violation in the input data)."""
    if T <= 0.0 or market_price <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return float("nan")

    # Reject below-intrinsic-forward inputs (Polygon prints these occasionally
    # for illiquid contracts).
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if opt == "call":
        intrinsic_fwd = max(spot * disc_q - strike * disc_r, 0.0)
    else:
        intrinsic_fwd = max(strike * disc_r - spot * disc_q, 0.0)
    if market_price < intrinsic_fwd - 1e-6:
        return float("nan")

    def price_at(sig: float) -> dict[str, float]:
        return price_and_greeks(spot, strike, T, r, q, sig, opt, exercise, n_steps)

    sigma = max(initial_guess, 1e-4)
    for _ in range(max_iters):
        res  = price_at(sigma)
        diff = res["price"] - market_price
        if abs(diff) < tol:
            return sigma
        vega = res["vega"]
        if vega < 1e-10:
            break
        step = diff / vega
        sigma_new = sigma - step
        if sigma_new <= 1e-6 or sigma_new > 10.0:
            break  # leave Newton; bisection picks up below
        sigma = sigma_new

    # Bisection fallback. Bracket: σ ∈ (1e-4, 5.0) covers everything from
    # near-deterministic to triple-digit IV (which Polygon sometimes prints for
    # deep ITM calls — see probe output).
    lo, hi = 1.0e-4, 5.0
    f_lo = price_at(lo)["price"] - market_price
    f_hi = price_at(hi)["price"] - market_price
    if f_lo * f_hi > 0:
        return float("nan")  # no sign change — can't bracket
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = price_at(mid)["price"] - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float("nan")
