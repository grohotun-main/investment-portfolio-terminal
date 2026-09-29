"""Constant-maturity ATM IV interpolation (pure, no I/O).

The IV-percentile gauge compares today's ATM IV against its own trailing
year. But the front-month series measures a *moving* maturity — DTE
sawtooths ~35→1 and jumps each monthly roll — so the percentile compares
readings taken at different points on the vol term structure. The ruler
keeps changing length.

This module pins the maturity: given one day's option term structure
(several `(dte_days, iv)` points), it returns the ATM IV at a fixed
`target_days` horizon, interpolated in TOTAL VARIANCE.

Why total variance: variance is additive in time, so w = sigma^2 * T is
the quantity that interpolates linearly across maturities. Interpolating
the vol directly (sigma) is wrong whenever the term structure is sloped.

Quality flag (honest about how the number was produced — no fabrication):
  * "interp" — target bracketed by two expiries (or hit exactly).
  * "approx" — all expiries on one side; nearest single expiry used.
  * "none"   — no usable points; returns NaN (the gauge NaN-skips).
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import pandas as pd

QUALITY_INTERP = "interp"
QUALITY_APPROX = "approx"
QUALITY_NONE = "none"

DAYS_PER_YEAR = 365.0

# Schema of the derived constant-maturity history (data/atm_iv_history.csv).
# `atm_iv` holds the constant-maturity value, so the gauge consumer
# (iv_rank.book_iv_percentile) reads the same date/underlying/atm_iv contract
# it always has — now on an honest, fixed-maturity ruler.
CM_CSV_COLS = ["date", "underlying", "atm_iv", "quality",
               "target_days", "fetched_at"]


def _clean(points: Iterable[Tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only points with positive DTE and finite, positive IV, sorted
    by DTE ascending. A 0-DTE or non-finite point can't carry term-
    structure information, so it's dropped before bracketing."""
    out: list[tuple[float, float]] = []
    for dte, iv in points:
        d = float(dte)
        v = float(iv)
        if d > 0.0 and math.isfinite(v) and v > 0.0:
            out.append((d, v))
    out.sort(key=lambda p: p[0])
    return out


def _interp_total_variance(lo: tuple[float, float], hi: tuple[float, float],
                           target_days: float) -> float:
    """Total-variance interpolation between two `(dte, iv)` points to
    `target_days`. w = sigma^2 * T is linear in DTE; convert back to a vol
    at the target maturity."""
    dte_lo, iv_lo = lo
    dte_hi, iv_hi = hi
    t_lo = dte_lo / DAYS_PER_YEAR
    t_hi = dte_hi / DAYS_PER_YEAR
    w_lo = iv_lo * iv_lo * t_lo
    w_hi = iv_hi * iv_hi * t_hi
    frac = (target_days - dte_lo) / (dte_hi - dte_lo)
    w_t = w_lo + frac * (w_hi - w_lo)
    t_t = target_days / DAYS_PER_YEAR
    return math.sqrt(w_t / t_t)


def constant_maturity_iv(points: Iterable[Tuple[float, float]],
                         target_days: float = 90.0) -> tuple[float, str]:
    """ATM IV at a fixed `target_days` horizon from one day's term structure.

    `points` is an iterable of `(dte_days, iv)` for the same date/underlying.
    Returns `(iv, quality)`:
      * bracketed (or exact hit)  → total-variance interp, "interp".
      * one-sided                 → nearest single expiry, "approx".
      * no usable points          → (NaN, "none").
    """
    clean = _clean(points)
    if not clean:
        return float("nan"), QUALITY_NONE

    # Exact hit — a real CM reading; avoids a zero-width bracket below.
    for dte, iv in clean:
        if dte == target_days:
            return iv, QUALITY_INTERP

    below = [p for p in clean if p[0] < target_days]
    above = [p for p in clean if p[0] > target_days]

    if below and above:
        lo = below[-1]   # largest DTE below target
        hi = above[0]    # smallest DTE above target
        return _interp_total_variance(lo, hi, target_days), QUALITY_INTERP

    # One-sided: nearest single expiry to the target (no extrapolation).
    nearest = below[-1] if below else above[0]
    return nearest[1], QUALITY_APPROX


def derive_cm_history(term_frame: pd.DataFrame,
                      target_days: float = 90.0) -> pd.DataFrame:
    """Project a raw term-structure history into a constant-maturity history.

    `term_frame` is long-format with several `(expiry)` rows per
    `(date, underlying)` — columns `date, underlying, dte_days, atm_iv,
    fetched_at` (extra columns ignored). Returns one row per
    `(date, underlying)` with `atm_iv` = the `target_days` constant-maturity
    value, plus `quality` and `target_days`, sorted by `(underlying, date)`.

    Dead days (every close failed to invert → all-NaN) are EMITTED with
    NaN `atm_iv` and quality "none" rather than dropped: the record stays
    honest, and the percentile gauge already NaN-skips them. `fetched_at`
    carries the newest stamp touching that day.
    """
    if term_frame is None or term_frame.empty:
        return pd.DataFrame(columns=CM_CSV_COLS)

    rows: list[dict] = []
    for (d, u), grp in term_frame.groupby(["date", "underlying"], sort=False):
        points = list(zip(grp["dte_days"], grp["atm_iv"]))
        iv, quality = constant_maturity_iv(points, target_days=target_days)
        fetched_at = (grp["fetched_at"].max()
                      if "fetched_at" in grp.columns else "")
        rows.append({
            "date":        d,
            "underlying":  u,
            "atm_iv":      iv,
            "quality":     quality,
            "target_days": int(target_days),
            "fetched_at":  fetched_at,
        })
    out = pd.DataFrame(rows, columns=CM_CSV_COLS)
    return out.sort_values(["underlying", "date"]).reset_index(drop=True)
