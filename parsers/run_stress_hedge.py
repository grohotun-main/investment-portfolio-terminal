"""CLI: stress-test a candidate hedge against canonical tail scenarios.

Fetches live underlying spot + IV + premium from Polygon snapshot, resolves
dividend yield via implied_dividend.solve_q, loads the latest 3-mo T-bill
for r, then runs stress_hedge.evaluate_hedge and prints a formatted table.

Example (live fetch):
  py parsers/run_stress_hedge.py --ticker SPY --type put --strike 540 \\
      --expiry 2026-08-15 --n 10

All of --spot, --sigma, --premium can be overridden — useful for what-if
sizing without round-tripping Polygon, or for offline replay.

Example (fully offline):
  py parsers/run_stress_hedge.py --ticker SPY --type put --strike 540 \\
      --expiry 2026-08-15 --n 10 --spot 495 --sigma 0.22 --premium 8.50 \\
      --no-fetch

Example (ATM-baseline mode — fixes wing-skew bias on far-OTM contracts):
  py parsers/run_stress_hedge.py --ticker SPY --type put --strike 525 \\
      --expiry 2026-08-21 --n 10 --vol-baseline atm
  # CLI extracts same-expiry ATM IV from the fetched chain automatically.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_base, get_massive_key  # noqa: E402
from _hedge_cli_common import (  # noqa: E402
    fetch_expiry_chain,
    list_nearby_expiries,
    load_risk_free_rate,
    pick_atm_iv,
    pick_premium,
)
from implied_dividend import solve_q  # noqa: E402
from stress_hedge import CONTRACT_MULT, Hedge, evaluate_hedge  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _print_eval(ev) -> None:
    h = ev.hedge
    dte = (h.expiry - date.today()).days
    n_shares = h.n_contracts * CONTRACT_MULT
    print()
    print(f"{h.ticker} {h.strike:g} {h.option_type.upper()} "
          f"expiring {h.expiry.isoformat()} ({dte} DTE)")
    print(f"n={h.n_contracts} contracts ({n_shares:,} share equivalents)")
    print()
    print("Today:")
    print(f"  spot:               {ev.spot_today:>10.2f}")
    print(f"  IV:                 {ev.sigma_today*100:>9.2f}%")
    print(f"  r:                  {ev.r*100:>9.2f}%      "
          f"q: {ev.q*100:+.3f}%")
    print(f"  premium / share:    {h.premium_per_share:>10.2f}    "
          f"upfront cost: ${ev.upfront_cost:>12,.2f}")
    print(f"  base reprice:       {ev.base_per_share:>10.2f}    "
          f"(vs premium {h.premium_per_share - ev.base_per_share:+.2f})")
    print(f"  notional protected: ${ev.notional_protected:>12,.2f}")
    print(f"  cost-of-insurance:  {ev.cost_of_insurance_pct*100:>9.2f}%")
    print(f"  breakeven at expiry: {ev.breakeven_spot:>9.2f}  "
          f"(decline {ev.breakeven_decline_pct*100:+.2f}%)")
    if ev.mtm_breakeven_spot is None:
        print(f"  breakeven (mark-to-market today): n/a  "
              f"(T=0 or premium above max value)")
    else:
        print(f"  breakeven (mark-to-market today): {ev.mtm_breakeven_spot:>9.2f}  "
              f"(decline {ev.mtm_breakeven_decline_pct*100:+.2f}%)")
    if ev.vol_baseline == "atm" and ev.sigma_atm is not None:
        skew_pts = (ev.sigma_today - ev.sigma_atm) * 100
        print(f"  ATM IV (baseline):  {ev.sigma_atm*100:>9.2f}%      "
              f"skew: {skew_pts:+.2f} vol pts")
        print(f"  mode: atm  (vol shocks scale ATM IV, skew re-added)")
    elif ev.sigma_today > 0.25:
        # Contract-mode warning — atm mode handles wing skew correctly so
        # no warning is needed in that path.
        print()
        print(f"  [note] contract IV {ev.sigma_today*100:.1f}% is skew-elevated "
              f"(typical SPY ATM 14-18%).")
        print(f"         vol_mult shocks compound on this baseline, so far-OTM "
              f"scenario P&L is biased high. Pass --vol-baseline atm to fix.")
    print()
    print(f"{'Scenario':<22s} {'Spot':>8s} {'IV':>7s}   "
          f"{'Option $':>10s} {'Δ P&L $':>14s} {'% notional':>11s}")
    print("-" * 80)
    for s in ev.scenarios:
        print(f"{s.scenario.name:<22s} "
              f"{s.shocked_spot:>8.2f} "
              f"{s.shocked_vol*100:>6.1f}%   "
              f"{s.repriced_per_share:>10.2f} "
              f"{s.pnl_total:>+14,.2f} "
              f"{s.pnl_pct_of_notional*100:>+10.2f}%")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ticker", required=True, help="Underlying symbol, e.g. SPY")
    ap.add_argument("--type", choices=["put", "call"], required=True)
    ap.add_argument("--strike", type=float, required=True)
    ap.add_argument("--expiry", required=True,
                    help="Contract expiry, ISO format YYYY-MM-DD")
    ap.add_argument("--n", type=int, required=True, help="Number of contracts")
    ap.add_argument("--spot", type=float, default=None,
                    help="Override underlying spot (default: live)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Override IV as decimal (e.g. 0.22). Default: live polygon_iv.")
    ap.add_argument("--premium", type=float, default=None,
                    help="Override premium per share. Default: mid of bid/ask.")
    ap.add_argument("--q", type=float, default=None,
                    help="Override dividend yield as decimal. Default: solve_q on the expiry chain.")
    ap.add_argument("--r", type=float, default=None,
                    help="Override risk-free rate. Default: latest from data/risk_free_rate.csv")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Skip Polygon — requires --spot, --sigma, --premium (and --q recommended).")
    ap.add_argument("--n-steps", type=int, default=200,
                    help="LR tree step count (default: 200)")
    ap.add_argument("--vol-baseline", choices=["contract", "atm"],
                    default="contract",
                    help="contract (default): vol shocks scale the contract's "
                         "own IV. atm: shocks scale same-expiry ATM IV, with "
                         "absolute vol-point skew re-added. atm mode fixes "
                         "the wing-skew bias for deep-OTM contracts.")
    ap.add_argument("--sigma-atm", type=float, default=None,
                    help="Override same-expiry ATM IV (decimal). Only used "
                         "when --vol-baseline atm. Default: extract from "
                         "fetched chain (closest-strike same-type contract).")
    args = ap.parse_args()

    try:
        expiry = date.fromisoformat(args.expiry)
    except ValueError:
        print(f"[!] --expiry must be ISO YYYY-MM-DD, got {args.expiry!r}")
        return 2

    spot      = args.spot
    sigma     = args.sigma
    premium   = args.premium
    q         = args.q
    r         = args.r if args.r is not None else load_risk_free_rate()
    sigma_atm = args.sigma_atm

    if not args.no_fetch:
        try:
            key  = get_massive_key()
            base = get_massive_base()
        except RuntimeError as e:
            print(f"[!] {e}")
            return 1
        print(f"Fetching {args.ticker} chain for expiry {expiry.isoformat()}...",
              flush=True)
        chain, fetched_spot = fetch_expiry_chain(args.ticker, expiry, key, base)
        if chain.empty:
            print(f"[!] No contracts returned for {args.ticker} {args.type} "
                  f"expiry {expiry} (often: weekend / non-listed weekly).")
            nearby = list_nearby_expiries(args.ticker, expiry, args.type,
                                          key, base, window_days=30)
            if nearby:
                print(f"  Available {args.type} expiries within ±30d:")
                for e in nearby:
                    print(f"    {e}")
            return 1

        # Locate our specific contract
        match = chain[(chain["contract_type"] == args.type)
                      & (chain["strike"].astype(float) == float(args.strike))]
        if match.empty:
            print(f"[!] Contract not found in chain: {args.type} "
                  f"strike={args.strike}. Available strikes for {args.type}:")
            avail = sorted(chain[chain["contract_type"] == args.type]
                           ["strike"].dropna().unique())
            print(" ", ", ".join(f"{s:g}" for s in avail[:20]),
                  "..." if len(avail) > 20 else "")
            return 1
        row = match.iloc[0]

        if spot is None:
            spot = float(fetched_spot) if fetched_spot is not None else \
                   float(row.get("underlying_price") or 0) or None
        if sigma is None and row.get("polygon_iv") is not None:
            sigma = float(row["polygon_iv"])
        if premium is None:
            premium = pick_premium(row)
        if q is None:
            q_res = solve_q(chain, args.ticker, r)
            q = q_res["q"]
            print(f"  q resolved via {q_res['method']} "
                  f"(n_strikes={q_res['n_strikes']}): {q*100:+.3f}%")
        if args.vol_baseline == "atm" and sigma_atm is None:
            atm_iv, atm_k = pick_atm_iv(chain, args.type, spot)
            if atm_iv is None:
                print(f"[!] vol_baseline atm requested but no same-type "
                      f"contract in the fetched chain had a populated IV. "
                      f"Pass --sigma-atm explicitly.")
                return 1
            sigma_atm = atm_iv
            print(f"  ATM IV resolved from chain (closest {args.type} "
                  f"K={atm_k:g} to spot {spot:.2f}): {sigma_atm*100:.2f}%")

    missing = [n for n, v in (("--spot", spot), ("--sigma", sigma),
                              ("--premium", premium)) if v is None]
    if missing:
        print(f"[!] Missing required values: {', '.join(missing)}. "
              f"Pass them explicitly or drop --no-fetch.")
        return 1
    if args.vol_baseline == "atm" and sigma_atm is None:
        print("[!] --vol-baseline atm requires --sigma-atm in --no-fetch mode.")
        return 1
    if q is None:
        q = 0.0
        print("  [warn] q unresolved — defaulting to 0%")

    hedge = Hedge(
        ticker=args.ticker,
        option_type=args.type,
        strike=args.strike,
        expiry=expiry,
        n_contracts=args.n,
        premium_per_share=premium,
    )
    ev = evaluate_hedge(hedge, spot_today=spot, sigma_today=sigma,
                        r=r, q=q, n_steps=args.n_steps,
                        vol_baseline=args.vol_baseline, sigma_atm=sigma_atm)
    _print_eval(ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
