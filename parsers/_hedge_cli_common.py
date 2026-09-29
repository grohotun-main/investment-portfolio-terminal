"""Shared CLI helpers for hedge scripts.

Used by:
  * `run_stress_hedge.py` — single-candidate stress eval
  * `compare_hedges.py`   — multi-candidate side-by-side compare

Pure-math hedge logic lives in `stress_hedge.py`. This module is the
*IO / market-data lookup* layer: load risk-free rate from CSV, fetch
option chains from Polygon, pick premium / ATM IV from a chain row.

Underscored module name signals "internal helper, not user API" — same
convention as `_config.py`.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RF_CSV = DATA / "risk_free_rate.csv"

# HTTP statuses worth retrying — transient server / throttle conditions.
# Everything else (401/403 entitlement, 404 bad ticker) fails immediately:
# retrying can't fix it. Mirrors fetch_atm_iv_history._RETRYABLE_STATUS.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _get_json(url: str, params: dict, *, get=requests.get, sleep=time.sleep,
              max_retries: int = 5, base_delay: float = 2.0,
              timeout: int = 30) -> dict:
    """GET `url` and return parsed JSON, retrying transient failures with
    exponential backoff. Raises the underlying ``requests`` error on a
    non-transient status (4xx) or after exhausting retries.

    The Polygon snapshot endpoint occasionally read-times-out under load; a
    bare `requests.get` with no retry let a single slow response abort the
    whole Option IV refresh. `get` and `sleep` are injectable so the backoff
    is unit-testable without real HTTP or wall-clock delay. Honors a
    `Retry-After` header when present. Mirrors `fetch_atm_iv_history._get_json`.
    """
    attempt = 0
    while True:
        try:
            resp = get(url, params=params, timeout=timeout)
        except requests.RequestException:
            # Network-level failure (connection reset, read timeout) —
            # transient; back off and retry until the budget is exhausted.
            if attempt >= max_retries:
                raise
            sleep(base_delay * (2 ** attempt))
            attempt += 1
            continue

        if resp.status_code == 200:
            return resp.json() or {}

        if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            delay = (float(retry_after) if retry_after
                     else base_delay * (2 ** attempt))
            sleep(delay)
            attempt += 1
            continue

        # Non-transient (4xx entitlement / bad ticker), or retryable but out
        # of retries: fail loud, same contract as the old raise_for_status().
        raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")


def load_risk_free_rate() -> float:
    """Latest 3-mo T-bill rate (decimal) from data/risk_free_rate.csv."""
    df = pd.read_csv(RF_CSV, parse_dates=["date"])
    df = df.dropna(subset=["rate_annual"])
    latest = df.sort_values("date").iloc[-1]
    return float(latest["rate_annual"])


def fetch_expiry_chain(underlying: str, expiry: date, key: str, base: str,
                       *, get=requests.get, sleep=time.sleep
                       ) -> tuple[pd.DataFrame, float | None]:
    """Pull all contracts for one underlying / one expiry. Returns
    (chain_df in fetch_options_chains.py layout, underlying_spot).

    Each page is fetched through `_get_json`, which retries transient Polygon
    read-timeouts / throttling — so a single slow snapshot response no longer
    aborts the caller (the Option IV refresh). `get`/`sleep` injectable for
    tests."""
    url = f"{base}/v3/snapshot/options/{underlying}"
    params = {
        "expiration_date": expiry.isoformat(),
        "limit": 250,
        "apiKey": key,
    }
    rows: list[dict] = []
    spot: float | None = None
    while True:
        payload = _get_json(url, params, get=get, sleep=sleep)
        for c in payload.get("results") or []:
            det = c.get("details") or {}
            gk  = c.get("greeks") or {}
            day = c.get("day") or {}
            lq  = c.get("last_quote") or {}
            ua  = c.get("underlying_asset") or {}
            if spot is None:
                spot = ua.get("price")
            exp_str = det.get("expiration_date")
            try:
                dte = (date.fromisoformat(exp_str) - date.today()).days if exp_str else None
            except (TypeError, ValueError):
                dte = None
            rows.append({
                "underlying": underlying,
                "contract_ticker": det.get("ticker"),
                "contract_type": det.get("contract_type"),
                "strike": det.get("strike_price"),
                "expiration_date": exp_str,
                "dte": dte,
                "polygon_price": day.get("close"),
                "polygon_iv": c.get("implied_volatility"),
                "polygon_delta": gk.get("delta"),
                "polygon_vega": gk.get("vega"),
                "polygon_open_interest": c.get("open_interest"),
                "polygon_bid": lq.get("bid"),
                "polygon_ask": lq.get("ask"),
                "underlying_price": ua.get("price"),
            })
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        params = {"apiKey": key}
    return pd.DataFrame(rows), spot


def list_nearby_expiries(underlying: str, target: date, opt_type: str,
                         key: str, base: str, window_days: int = 30
                         ) -> list[str]:
    """Return sorted unique expiration dates for `opt_type` contracts on
    `underlying` within ±window_days of `target`. Used to give the user
    actionable suggestions when their --expiry doesn't match an actual
    listing (weekends, holidays, non-listed weeklies)."""
    url = f"{base}/v3/snapshot/options/{underlying}"
    params = {
        "expiration_date.gte": (target - timedelta(days=window_days)).isoformat(),
        "expiration_date.lte": (target + timedelta(days=window_days)).isoformat(),
        "contract_type": opt_type,
        "limit": 250,
        "apiKey": key,
    }
    expiries: set[str] = set()
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        for c in payload.get("results") or []:
            exp = (c.get("details") or {}).get("expiration_date")
            if exp:
                expiries.add(exp)
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        params = {"apiKey": key}
    return sorted(expiries)


def pick_premium(row: pd.Series) -> float | None:
    """Mid if both bid+ask present and >0; else ask; else last close."""
    bid = row.get("polygon_bid")
    ask = row.get("polygon_ask")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (float(bid) + float(ask)) / 2.0
    if ask is not None and ask > 0:
        return float(ask)
    px = row.get("polygon_price")
    if px is not None and px > 0:
        return float(px)
    return None


def pick_atm_iv(chain: pd.DataFrame, opt_type: str,
                spot: float) -> tuple[float | None, float | None]:
    """Pick the IV of the same-type contract whose strike is closest to spot.
    Returns (iv, strike) — both None if no candidate has a populated IV.

    Same-type because put/call IVs at the same K can differ slightly from
    quoting noise even when PCP says they should match. For ATM-baseline
    stress we want the side that matches what we're pricing."""
    cands = chain[(chain["contract_type"] == opt_type)
                  & chain["polygon_iv"].notna()
                  & chain["strike"].notna()].copy()
    if cands.empty:
        return None, None
    cands["abs_dist"] = (cands["strike"].astype(float) - spot).abs()
    pick = cands.sort_values(["abs_dist", "strike"]).iloc[0]
    return float(pick["polygon_iv"]), float(pick["strike"])
