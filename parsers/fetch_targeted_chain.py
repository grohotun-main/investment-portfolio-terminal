"""Pull a small set of targeted option contracts from Polygon.

Inputs: list of (underlying, target_strike, target_expiry, contract_type)
tuples. For each input, hits ``/v3/snapshot/options/{underlying}`` with
a DTE band around the target expiry, then picks the contract whose
(expiry, strike) is closest to the target.

Why this exists: the Phase 2 recommender wants live premium/IV at
specific strikes (e.g. SPY at 90% spot for a 10% cap). The bulk
``fetch_options_chains.py`` would over-fetch; the snapshot endpoint is
designed for this — one request per (underlying, expiry-window) gets
us everything we need at moderate cost.

Run:
  py parsers/fetch_targeted_chain.py            # dry-run on a fixed sample
  py parsers/fetch_targeted_chain.py --write    # writes data/hedge_chain_snapshot.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_key, get_massive_base  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT_CSV = DATA / "hedge_chain_snapshot.csv"

DTE_BAND_DAYS = 30  # ± around target expiry


def _fetch_chain_window(underlying: str, target_expiry: date,
                         key: str, base: str,
                         band_days: int = DTE_BAND_DAYS,
                         ) -> list[dict]:
    """Pull all contracts for `underlying` in [target ± band_days]."""
    lo = (target_expiry - timedelta(days=band_days)).isoformat()
    hi = (target_expiry + timedelta(days=band_days)).isoformat()
    params = {
        "expiration_date.gte": lo,
        "expiration_date.lte": hi,
        "limit": 250,
        "apiKey": key,
    }
    url = f"{base}/v3/snapshot/options/{underlying}"
    out: list[dict] = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        out.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        params = {"apiKey": key}
    return out


def pick_nearest_contract(chain: list[dict],
                           target_strike: float,
                           target_expiry: date,
                           contract_type: str = "put",
                           ) -> dict | None:
    """Pick the contract closest to (target_strike, target_expiry).

    Tie-break order:
      1. contract_type must match.
      2. Closest expiry (|expiry - target| in days), then
      3. Closest strike, then
      4. Prefer open_interest > 0 over 0 (liquidity).

    Returns a flat dict {strike, expiration_date, contract_type,
    contract_ticker, polygon_price, polygon_iv, polygon_delta,
    polygon_gamma, polygon_vega, polygon_theta, open_interest, bid, ask}
    or None if no qualifying contract.
    """
    pool = []
    for c in chain:
        det = c.get("details") or {}
        if (det.get("contract_type") or "").lower() != contract_type:
            continue
        K = det.get("strike_price")
        exp = det.get("expiration_date")
        if K is None or exp is None:
            continue
        try:
            exp_d = date.fromisoformat(exp)
        except (TypeError, ValueError):
            continue
        pool.append((c, det, abs((exp_d - target_expiry).days),
                     abs(float(K) - target_strike), int(c.get("open_interest") or 0)))
    if not pool:
        return None
    # Sort: nearest expiry, nearest strike, then OI desc (boolean trick: -OI).
    pool.sort(key=lambda x: (x[2], x[3], -x[4]))
    c, det, _exp_d, _k_d, oi = pool[0]
    gk = c.get("greeks") or {}
    day = c.get("day") or {}
    lq = c.get("last_quote") or {}
    return {
        "strike":          det.get("strike_price"),
        "expiration_date": det.get("expiration_date"),
        "contract_type":   det.get("contract_type"),
        "contract_ticker": det.get("ticker"),
        "polygon_price":   day.get("close"),
        "polygon_iv":      c.get("implied_volatility"),
        "polygon_delta":   gk.get("delta"),
        "polygon_gamma":   gk.get("gamma"),
        "polygon_vega":    gk.get("vega"),
        "polygon_theta":   gk.get("theta"),
        "open_interest":   oi,
        "polygon_bid":     lq.get("bid"),
        "polygon_ask":     lq.get("ask"),
    }


def fetch_targeted_contracts(targets: Iterable[tuple[str, float, date, str]],
                               *, key: str | None = None,
                               base: str | None = None,
                               ) -> pd.DataFrame:
    """Fetch one matched contract per (underlying, strike, expiry, type) tuple.

    Returns a DataFrame with one row per target, including request keys
    so the caller can join.
    """
    if key is None:
        key = get_massive_key()
    if base is None:
        base = get_massive_base()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    for underlying, strike, expiry, ctype in targets:
        chain = _fetch_chain_window(underlying, expiry, key=key, base=base)
        picked = pick_nearest_contract(chain, strike, expiry, ctype)
        rows.append({
            "request_underlying":    underlying,
            "request_strike":        strike,
            "request_expiry":        expiry.isoformat(),
            "request_contract_type": ctype,
            "fetched_at":            fetched_at,
            **(picked or {
                "strike": None, "expiration_date": None,
                "contract_type": None, "contract_ticker": None,
                "polygon_price": None, "polygon_iv": None,
                "polygon_delta": None, "polygon_gamma": None,
                "polygon_vega": None, "polygon_theta": None,
                "open_interest": 0, "polygon_bid": None, "polygon_ask": None,
            }),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write data/hedge_chain_snapshot.csv (default: dry-run)")
    args = ap.parse_args()

    # Dry-run sample: SPY at 90% of an arbitrary spot, ~180 DTE.
    today = date.today()
    target_expiry = today + timedelta(days=180)
    sample_targets = [
        ("SPY", 500.0, target_expiry, "put"),
        ("NVDA", 100.0, target_expiry, "put"),
    ]
    try:
        df = fetch_targeted_contracts(sample_targets)
    except (requests.RequestException, RuntimeError) as e:
        print(f"[!] Fetch failed: {e}")
        return 1

    cols = ["request_underlying", "request_strike", "request_expiry",
            "strike", "expiration_date", "polygon_price", "polygon_iv",
            "open_interest"]
    print("Sample (1 row per target):")
    print(df[cols].to_string(index=False))

    if args.write:
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    else:
        print(f"\n[dry-run] would write {OUT_CSV}; use --write to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
