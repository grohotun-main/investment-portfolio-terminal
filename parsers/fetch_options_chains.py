"""Fetch a sample of options-chain snapshots from Polygon for the hedging
engine's verification step.

DESIGN: idempotent, dry-run-by-default, mirrors fetch_holding_prices.py shape.
  - Pulls /v3/snapshot/options/{underlying} for a small universe (SPY, XSP,
    QQQ, NVDA, AAPL, VIX).
  - For each underlying: filters to expiries in a hedging-relevant window
    (default 20-90 DTE) and strikes within ±N% of spot. This keeps the
    sample CSV at a few hundred rows instead of 10k+.
  - Polygon's snapshot returns the EOD close (day.close), Polygon's
    pre-computed Greeks + IV, open interest, and live bid/ask. All are
    captured so the verification script can compare against our pricer.

Output: `data/options_chains_sample.csv` with one row per contract.

Run modes:
  py parsers/fetch_options_chains.py            # dry-run, prints samples
  py parsers/fetch_options_chains.py --write    # emit data/options_chains_sample.csv

Polygon endpoint requires Options Starter tier or above (snapshot is gated
behind Starter; this script will 403 on Basic).
"""
import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from _config import get_massive_key, get_massive_base

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_CSV = DATA / "options_chains_sample.csv"

# Per-underlying fetch knobs. dte_min/max set the expiry window in calendar
# days; strike_band is the half-width of the strike filter as a fraction of
# spot (0.15 = ±15%). VIX uses a wider band because its strikes are sparse
# and the range needs to cover both calm-vol (~14) and stress-vol (~40+)
# regions of the chain.
UNIVERSE: dict[str, dict] = {
    "SPY":  {"dte_min": 20, "dte_max": 90, "strike_band": 0.15},
    "QQQ":  {"dte_min": 20, "dte_max": 90, "strike_band": 0.15},
    "NVDA": {"dte_min": 20, "dte_max": 90, "strike_band": 0.25},
    "AAPL": {"dte_min": 20, "dte_max": 90, "strike_band": 0.25},
    # XSP and VIX deferred — empirical coverage on Options Starter ($29),
    # reprobed 2026-05-25:
    #   XSP: only contract metadata + open_interest=0. All value fields
    #     (underlying spot, day.close, Greeks, IV, bid/ask) come back null.
    #     No path on this tier.
    #   VIX: day.close + day.volume + open_interest ARE populated. Missing:
    #     VIX-spot in underlying_asset, Greeks, IV, bid/ask.
    # The Indices product (massive.com/indices) does NOT fix this — its
    # /v3/snapshot/indices returns 403 NOT_AUTHORIZED on our tier and only
    # sells spot/aggregates on the indices themselves, not options-on-
    # indices Greeks. (And VIX spot is already free via fetch_vix.py.)
    # The real blocker for VIX-call hedging is forward-based pricing math,
    # not data: VIX itself is non-tradable, so long-dated VIX options price
    # off the VIX-futures curve (contango), not VIX-spot. Adding that is a
    # Phase C pricer change, not a fetch-script change.
}

CSV_COLS = [
    "underlying", "contract_ticker", "contract_type", "exercise_style",
    "strike", "expiration_date", "dte",
    "polygon_price", "polygon_volume", "polygon_vwap",
    "polygon_iv", "polygon_delta", "polygon_gamma", "polygon_vega",
    "polygon_theta", "polygon_open_interest",
    "polygon_bid", "polygon_ask",
    "underlying_price", "snapshot_timeframe", "fetched_at",
]


def _fetch_chain(underlying: str, dte_min: int, dte_max: int,
                 key: str, base: str) -> tuple[list[dict], float | None, str | None]:
    """Pull all contracts in the DTE window. Paginates until exhausted.
    Returns (rows, underlying_price, snapshot_timeframe)."""
    today = date.today()
    params = {
        "expiration_date.gte": (today + timedelta(days=dte_min)).isoformat(),
        "expiration_date.lte": (today + timedelta(days=dte_max)).isoformat(),
        "limit": 250,
        "apiKey": key,
    }
    url = f"{base}/v3/snapshot/options/{underlying}"
    rows: list[dict] = []
    ul_price = None
    timeframe = None

    while True:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [!] {underlying}: HTTP error: {e}")
            return rows, ul_price, timeframe
        payload = r.json()
        results = payload.get("results") or []
        rows.extend(results)
        # Capture spot/timeframe from the first response that has them.
        if ul_price is None and results:
            ua = results[0].get("underlying_asset") or {}
            ul_price = ua.get("price")
            timeframe = ua.get("timeframe")
        next_url = payload.get("next_url")
        if not next_url:
            break
        # The next_url already encodes pagination cursor; just append the key.
        # Clear params so we don't double-encode the cursor params.
        url = next_url
        params = {"apiKey": key}
    return rows, ul_price, timeframe


def _flatten(underlying: str, contract: dict, ul_price: float | None,
             timeframe: str | None, fetched_at: str) -> dict:
    det = contract.get("details") or {}
    gk  = contract.get("greeks") or {}
    day = contract.get("day") or {}
    lq  = contract.get("last_quote") or {}
    exp_str = det.get("expiration_date")
    try:
        dte = (date.fromisoformat(exp_str) - date.today()).days if exp_str else None
    except (TypeError, ValueError):
        dte = None
    return {
        "underlying": underlying,
        "contract_ticker": det.get("ticker"),
        "contract_type": det.get("contract_type"),
        "exercise_style": det.get("exercise_style"),
        "strike": det.get("strike_price"),
        "expiration_date": exp_str,
        "dte": dte,
        "polygon_price": day.get("close"),
        "polygon_volume": day.get("volume"),
        "polygon_vwap": day.get("vwap"),
        "polygon_iv": contract.get("implied_volatility"),
        "polygon_delta": gk.get("delta"),
        "polygon_gamma": gk.get("gamma"),
        "polygon_vega": gk.get("vega"),
        "polygon_theta": gk.get("theta"),
        "polygon_open_interest": contract.get("open_interest"),
        "polygon_bid": lq.get("bid"),
        "polygon_ask": lq.get("ask"),
        "underlying_price": ul_price,
        "snapshot_timeframe": timeframe,
        "fetched_at": fetched_at,
    }


def _filter_by_strike(rows: list[dict], spot: float, band: float) -> list[dict]:
    """Keep only contracts whose strike falls within ±band fraction of spot.
    No-op if spot is missing (returns rows unchanged so the user can still
    inspect what came back)."""
    if spot is None or spot <= 0:
        return rows
    lo, hi = spot * (1 - band), spot * (1 + band)
    out = []
    for r in rows:
        K = (r.get("details") or {}).get("strike_price")
        if K is None:
            continue
        if lo <= float(K) <= hi:
            out.append(r)
    return out


def fetch_chain(underlying: str, dte_min: int, dte_max: int, *,
                key: str | None = None, base: str | None = None
                ) -> tuple[list[dict], float | None]:
    """Public seam (2026-07-06 hedge-report spec): flattened snapshot rows +
    spot for one underlying over a DTE window. Wraps the private
    _fetch_chain/_flatten pair unchanged — promote-on-second-consumer."""
    key = key or get_massive_key()
    base = base or get_massive_base()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw, spot, tf = _fetch_chain(underlying, dte_min, dte_max, key, base)
    return [_flatten(underlying, c, spot, tf, fetched_at) for c in raw], spot


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Write data/options_chains_sample.csv (default: dry-run)")
    args = ap.parse_args()

    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    all_rows: list[dict] = []
    print(f"{'Underlying':10s} {'Spot':>10s} {'Raw':>6s} {'Filtered':>9s}  Status")
    print("-" * 60)
    for ticker, cfg in UNIVERSE.items():
        rows, spot, tf = _fetch_chain(ticker, cfg["dte_min"], cfg["dte_max"], key, base)
        kept = _filter_by_strike(rows, spot, cfg["strike_band"])
        flat = [_flatten(ticker, c, spot, tf, fetched_at) for c in kept]
        all_rows.extend(flat)
        spot_str = f"{spot:.2f}" if spot is not None else "n/a"
        status = "OK" if rows else "NO DATA"
        print(f"{ticker:10s} {spot_str:>10s} {len(rows):>6d} {len(kept):>9d}  {status}")

    if not all_rows:
        print("[!] No rows pulled. Either bad credentials or the tier is too thin.")
        return 1

    df = pd.DataFrame(all_rows, columns=CSV_COLS)

    # Sample preview (per the user's "show samples before bulk runs" feedback)
    print()
    print(f"Total rows: {len(df)}")
    print()
    print("Sample (3 contracts per underlying):")
    print("-" * 60)
    sample = df.groupby("underlying", group_keys=False).head(3)
    cols_to_show = ["underlying", "contract_type", "strike", "expiration_date",
                    "polygon_price", "polygon_iv", "polygon_delta",
                    "polygon_open_interest"]
    print(sample[cols_to_show].to_string(index=False))

    # Surface populated-field counts so we know if Greeks/IV are present
    print()
    print("Field population (out of {}):".format(len(df)))
    for col in ("polygon_iv", "polygon_delta", "polygon_gamma", "polygon_vega",
                "polygon_theta", "polygon_open_interest", "polygon_bid",
                "polygon_ask", "polygon_price"):
        n = int(df[col].notna().sum())
        print(f"  {col:25s} {n:>5d}  ({100*n/len(df):5.1f}%)")

    if not args.write:
        print()
        print("[dry-run] add --write to emit", OUT_CSV.relative_to(ROOT))
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print()
    print(f"[ok] wrote {len(df)} rows -> {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
