"""Fetch live Polygon snapshots for the option contracts we currently hold.

Reads the parsed position table from `option_positions.build_option_position_table`
(latest statement date), pulls the matching Polygon snapshot for each unique
(underlying, expiry) combo, and writes per-position rows to
`data/option_position_snapshot.csv`.

The dashboard's Options Hedging tab reads this CSV to populate Greeks /
premium / IV without hitting Polygon on every Streamlit rerun.

Endpoint: ``/v3/snapshot/options/{underlying}`` filtered by expiration_date.
One API call per (underlying, expiry) combo regardless of how many strikes
we hold at that expiry — strikes are filtered client-side after the pull.

Run:
  py parsers/fetch_option_position_iv.py              # dry-run
  py parsers/fetch_option_position_iv.py --write      # write CSV
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_base, get_massive_key  # noqa: E402
from _hedge_cli_common import (  # noqa: E402
    fetch_expiry_chain, pick_atm_iv, pick_premium,
)
from option_positions import build_option_position_table  # noqa: E402
from synthesize_interim_positions import synthesize_interim_positions  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT_CSV = DATA / "option_position_snapshot.csv"

CSV_COLS = [
    "underlying", "opt_type", "strike", "expiry",
    "spot", "premium_mid", "polygon_iv",
    "atm_iv", "atm_strike",
    "polygon_delta", "polygon_gamma", "polygon_vega", "polygon_theta",
    "polygon_bid", "polygon_ask", "polygon_open_interest",
    "polygon_price", "polygon_volume",
    "contract_ticker", "fetched_at",
]


def _match_row(chain: pd.DataFrame, strike: float, opt_type: str
               ) -> pd.Series | None:
    """Find the chain row matching (strike, opt_type)."""
    if chain.empty:
        return None
    m = (chain["contract_type"] == opt_type) & (chain["strike"].astype(float) == float(strike))
    hits = chain[m]
    if hits.empty:
        return None
    return hits.iloc[0]


def fetch_snapshot_for_positions(
    parsed_positions: pd.DataFrame, key: str, base: str,
) -> pd.DataFrame:
    """Hit Polygon once per (underlying, expiry), match each position by
    (strike, opt_type), and return a per-position DataFrame.

    Positions with NaN strike / expiry (parser-source 'unparsed') are
    silently skipped — they're surfaced separately in the dashboard.
    """
    have_keys = (
        parsed_positions["underlying"].notna()
        & parsed_positions["expiry"].notna()
        & parsed_positions["strike"].notna()
    )
    work = parsed_positions[have_keys].copy()
    if work.empty:
        return pd.DataFrame(columns=CSV_COLS)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Dedupe by (underlying, expiry) so we make one call per chain pull,
    # not one per strike. Multiple accounts holding the same contract reuse
    # the same Polygon response.
    unique_combos = work[["underlying", "expiry"]].drop_duplicates()
    print(f"Polygon: fetching {len(unique_combos)} chain(s) for "
          f"{len(work)} held position(s)")

    chain_cache: dict[tuple, tuple[pd.DataFrame, float | None]] = {}
    # ATM IV cache: keyed by (underlying, expiry, opt_type). Put-ATM and
    # call-ATM at the same strike can differ from quoting noise; the stress
    # block wants the side that matches what we're pricing.
    atm_cache: dict[tuple, tuple[float | None, float | None]] = {}
    out_rows: list[dict] = []

    for _, combo in unique_combos.iterrows():
        underlying = combo["underlying"]
        # combo["expiry"] is a date or pd.Timestamp; coerce to date
        exp_val = combo["expiry"]
        if hasattr(exp_val, "date"):
            exp = exp_val.date()
        elif isinstance(exp_val, date):
            exp = exp_val
        else:
            exp = pd.to_datetime(exp_val).date()
        chain, spot = fetch_expiry_chain(underlying, exp, key, base)
        chain_cache[(underlying, exp)] = (chain, spot)

        # Compute ATM IV for each opt_type held against this (underlying, expiry).
        # No extra HTTP — same chain we just pulled. Surfaces in the per-chain
        # log so the user sees what the stress block will use.
        held_types = work[
            (work["underlying"] == underlying)
            & (work["expiry"] == combo["expiry"])
        ]["opt_type"].unique()
        atm_bits: list[str] = []
        for opt_type in held_types:
            if spot is not None and not chain.empty:
                iv, k = pick_atm_iv(chain, opt_type, float(spot))
            else:
                iv, k = None, None
            atm_cache[(underlying, exp, opt_type)] = (iv, k)
            if iv is not None:
                atm_bits.append(
                    f"ATM-{opt_type[0].upper()} {iv*100:.1f}%@${k:.0f}"
                )
        atm_suffix = f", {' / '.join(atm_bits)}" if atm_bits else ""
        print(f"  {underlying} {exp}: {len(chain):>3d} contracts, "
              f"spot={spot if spot is not None else 'n/a'}{atm_suffix}")

    # Match each parsed position against the cached chain.
    for _, pos in work.iterrows():
        exp_val = pos["expiry"]
        if hasattr(exp_val, "date"):
            exp = exp_val.date()
        elif isinstance(exp_val, date):
            exp = exp_val
        else:
            exp = pd.to_datetime(exp_val).date()
        chain, spot = chain_cache.get((pos["underlying"], exp), (pd.DataFrame(), None))
        row = _match_row(chain, float(pos["strike"]), pos["opt_type"])
        if row is None:
            print(f"  [!] no Polygon match for {pos['underlying']} "
                  f"{pos['opt_type']} K={pos['strike']} exp {exp}")
            continue

        mid = pick_premium(row)
        atm_iv, atm_strike = atm_cache.get(
            (pos["underlying"], exp, pos["opt_type"]), (None, None)
        )
        out_rows.append({
            "underlying":      pos["underlying"],
            "opt_type":        pos["opt_type"],
            "strike":          float(pos["strike"]),
            "expiry":          exp.isoformat(),
            "spot":            spot,
            "premium_mid":     mid,
            "polygon_iv":      row.get("polygon_iv"),
            "atm_iv":          atm_iv,
            "atm_strike":      atm_strike,
            "polygon_delta":   row.get("polygon_delta"),
            "polygon_gamma":   None,   # _hedge_cli_common doesn't pull gamma
            "polygon_vega":    row.get("polygon_vega"),
            "polygon_theta":   None,   # _hedge_cli_common doesn't pull theta
            "polygon_bid":     row.get("polygon_bid"),
            "polygon_ask":     row.get("polygon_ask"),
            "polygon_open_interest": row.get("polygon_open_interest"),
            "polygon_price":   row.get("polygon_price"),
            "polygon_volume":  None,
            "contract_ticker": row.get("contract_ticker"),
            "fetched_at":      fetched_at,
        })

    return pd.DataFrame(out_rows, columns=CSV_COLS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Write data/option_position_snapshot.csv (default: dry-run)")
    ap.add_argument("--as-of", default=None,
                    help="Snapshot statement date to use (YYYY-MM-DD). "
                         "Default: latest in positions.csv.")
    args = ap.parse_args(argv)

    try:
        key = get_massive_key()
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    base = get_massive_base()

    positions = pd.read_csv(DATA / "positions.csv",
                            parse_dates=["statement_date"])
    transactions = pd.read_csv(DATA / "transactions.csv",
                               parse_dates=["settlement_date"])
    # Mirror the dashboard's load_data flow so we see post-statement opens
    # captured by interim CSV ingest. Without this, options bought between
    # monthly statements are invisible to the IV snapshot.
    interim_path = DATA / "transactions_interim.csv"
    if interim_path.exists():
        interim = pd.read_csv(interim_path, parse_dates=["settlement_date"])
        if not interim.empty:
            transactions = pd.concat(
                [transactions, interim], ignore_index=True
            )
            rolled = synthesize_interim_positions(positions, interim)
            if not rolled.empty:
                positions = pd.concat(
                    [positions, rolled], ignore_index=True
                )
    as_of = (datetime.strptime(args.as_of, "%Y-%m-%d").date()
             if args.as_of else None)
    parsed = build_option_position_table(positions, transactions, as_of=as_of)
    if parsed.empty:
        print("No option positions in the latest snapshot.")
        return 0
    # Drop TEST-* test fixture accounts and unparsed rows.
    real = parsed[
        (~parsed["account_id"].astype(str).str.startswith("TEST"))
        & (parsed["source"] != "unparsed")
    ].copy()
    if real.empty:
        print("No parseable real-account option positions.")
        return 0

    # Multiple accounts may hold the same contract — dedupe before fetching.
    unique_positions = real.drop_duplicates(
        subset=["underlying", "opt_type", "strike", "expiry"]
    )
    print(f"{len(real)} held position(s), "
          f"{len(unique_positions)} unique contract(s) "
          f"across {real['account_id'].nunique()} account(s)")

    snap = fetch_snapshot_for_positions(unique_positions, key, base)
    if snap.empty:
        print("[!] No snapshots fetched.")
        return 1

    print()
    print(snap[["underlying", "opt_type", "strike", "expiry",
                "spot", "premium_mid", "polygon_iv", "atm_iv",
                "polygon_delta"]].to_string(index=False))

    if not args.write:
        print()
        print(f"[dry-run] add --write to emit {OUT_CSV.relative_to(ROOT)}")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    snap.to_csv(OUT_CSV, index=False)
    print()
    print(f"[ok] wrote {len(snap)} rows -> {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
