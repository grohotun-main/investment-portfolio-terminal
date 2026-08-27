"""CLI: compare 2-N candidate hedges side-by-side across stress scenarios.

The buy-decision shape: same ticker, several (strike, expiry, type)
candidates, identical scenarios — read down each candidate column to
score it; read across each scenario row to compare candidates head-to-head.

Each candidate uses --n contracts (default 1 — multiply mentally). Live
fetch (default): one Polygon chain pull per unique expiry, then q is
solved on that chain and ATM IV is extracted from it (for
--vol-baseline atm).

Candidate spec: ``K:expiry:type[:premium[:sigma[:atm]]]``.  The trailing
fields are optional overrides; in ``--no-fetch`` mode they're required
(premium + sigma always, atm only when ``--vol-baseline atm``).

Example (3 candidates on SPY, mixed strikes + expiries, live fetch):
  py parsers/compare_hedges.py --ticker SPY \\
      --candidate 525:2026-08-21:put \\
      --candidate 540:2026-08-21:put \\
      --candidate 540:2026-09-19:put \\
      --vol-baseline atm

Example (replay / fully offline — all data baked into the specs):
  py parsers/compare_hedges.py --ticker SPY --no-fetch --spot 745.64 \\
      --candidate 525:2026-08-21:put:1.16:0.3829:0.1556 \\
      --candidate 540:2026-08-21:put:1.32:0.3647:0.1556 \\
      --vol-baseline atm
"""
import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

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
from stress_hedge import (  # noqa: E402
    CONTRACT_MULT,
    DEFAULT_SCENARIOS,
    Hedge,
    HedgeEvaluation,
    evaluate_hedge,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class CandidateSpec:
    """Parsed candidate specification. Required fields (strike, expiry,
    option_type) come from positional CLI input; optional fields are
    overrides used in --no-fetch mode or to override live-fetched values.
    """
    strike: float
    expiry: date
    option_type: str        # "put" | "call"
    premium: float | None = None
    sigma: float | None = None
    sigma_atm: float | None = None


def parse_candidate_spec(spec: str) -> CandidateSpec:
    """Parse ``K:expiry:type[:premium[:sigma[:atm]]]`` → CandidateSpec.

    The leading three parts are required (strike positive, expiry ISO
    date, type put|call). The trailing three are optional overrides.
    Raises ValueError with a specific message on any parse failure so
    the CLI can surface it to the user.
    """
    parts = spec.split(":")
    if not (3 <= len(parts) <= 6):
        raise ValueError(
            f"candidate spec must be 'K:expiry:type[:premium[:sigma[:atm]]]',"
            f" got {spec!r} ({len(parts)} colon-separated parts, expected 3-6)"
        )
    k_str, exp_str, type_str = parts[:3]
    try:
        strike = float(k_str)
    except ValueError as e:
        raise ValueError(f"strike must be a number in {spec!r}, got {k_str!r}") from e
    if strike <= 0:
        raise ValueError(f"strike must be positive in {spec!r}, got {strike}")
    try:
        expiry = date.fromisoformat(exp_str)
    except ValueError as e:
        raise ValueError(
            f"expiry must be ISO YYYY-MM-DD in {spec!r}, got {exp_str!r}"
        ) from e
    type_norm = type_str.strip().lower()
    if type_norm not in ("put", "call"):
        raise ValueError(
            f"type must be 'put' or 'call' in {spec!r}, got {type_str!r}"
        )

    def _opt_pos(name: str, idx: int, *, positive: bool = True) -> float | None:
        if idx >= len(parts):
            return None
        raw = parts[idx].strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError as e:
            raise ValueError(
                f"{name} must be a number in {spec!r}, got {parts[idx]!r}"
            ) from e
        if positive and val <= 0:
            raise ValueError(
                f"{name} must be positive in {spec!r}, got {val}"
            )
        return val

    premium  = _opt_pos("premium", 3)
    sigma    = _opt_pos("sigma",   4)
    atm      = _opt_pos("atm",     5)
    return CandidateSpec(strike=strike, expiry=expiry, option_type=type_norm,
                         premium=premium, sigma=sigma, sigma_atm=atm)


def _candidate_label(idx: int) -> str:
    """A, B, C, ... AA, AB, ... AZ, BA, ... (Excel-column style)."""
    label = ""
    n = idx
    while True:
        label = chr(ord("A") + n % 26) + label
        n = n // 26 - 1
        if n < 0:
            return label


def format_compare_table(
    ticker: str,
    spot: float,
    r: float,
    evals: Sequence[HedgeEvaluation],
    q_by_expiry: dict[date, float],
    atm_iv_by_expiry: dict[date, float] | None = None,
    today: date | None = None,
) -> str:
    """Render the multi-candidate comparison table as a single string.

    Pure formatter — no IO, no pricer. Tests verify it on hand-built
    HedgeEvaluation objects. The CLI's main() wires up data fetch and
    calls evaluate_hedge per candidate to produce `evals`.
    """
    today_d = today or date.today()
    lines: list[str] = []

    # Header line: market frame
    lines.append(f"{ticker}  spot {spot:.2f}  r {r*100:.2f}%")
    lines.append("")

    # Per-expiry q + ATM IV
    expiries_in_order = list(OrderedDict.fromkeys(ev.hedge.expiry for ev in evals))
    if len(expiries_in_order) > 1:
        lines.append("Per-expiry market frame:")
    else:
        lines.append("Expiry frame:")
    for exp in expiries_in_order:
        q = q_by_expiry.get(exp, 0.0)
        atm = (atm_iv_by_expiry or {}).get(exp)
        atm_str = f"  ATM IV {atm*100:5.2f}%" if atm is not None else ""
        lines.append(f"  {exp.isoformat()} : q {q*100:+6.3f}%{atm_str}")
    lines.append("")

    # Candidate spec block
    n_first = evals[0].hedge.n_contracts
    same_n = all(ev.hedge.n_contracts == n_first for ev in evals)
    n_label = f"n={n_first} contracts each" if same_n else "n varies per candidate"
    lines.append(f"Candidates ({n_label}):")
    for i, ev in enumerate(evals):
        h = ev.hedge
        label = _candidate_label(i)
        dte = (h.expiry - today_d).days
        atm_part = ""
        if ev.vol_baseline == "atm" and ev.sigma_atm is not None:
            skew_pts = (ev.sigma_today - ev.sigma_atm) * 100
            atm_part = f"  (ATM {ev.sigma_atm*100:5.2f}%  skew {skew_pts:+5.2f}pp)"
        lines.append(
            f"  {label}) K={h.strike:g} {h.option_type:<4s} exp {h.expiry.isoformat()}  "
            f"{dte:>3d} DTE  IV {ev.sigma_today*100:5.2f}%{atm_part}"
        )
        if ev.mtm_breakeven_spot is None:
            be_mtm_part = "  BE@mtm n/a"
        else:
            be_mtm_part = (
                f"  BE@mtm {ev.mtm_breakeven_spot:7.2f} "
                f"({ev.mtm_breakeven_decline_pct*100:+6.2f}%)"
            )
        lines.append(
            f"        prem {h.premium_per_share:7.2f}  "
            f"upfront ${ev.upfront_cost:>10,.2f}  "
            f"notional ${ev.notional_protected:>12,.2f}  "
            f"COI {ev.cost_of_insurance_pct*100:5.2f}%  "
            f"BE@exp {ev.breakeven_spot:7.2f} "
            f"({ev.breakeven_decline_pct*100:+6.2f}%){be_mtm_part}"
        )

    # Mode line
    modes = {ev.vol_baseline for ev in evals}
    if len(modes) == 1:
        lines.append(f"\nMode: vol_baseline={next(iter(modes))}")
    else:
        lines.append(f"\nMode: mixed vol_baseline ({sorted(modes)})")

    # Stress P&L table — one row per scenario, three sub-cells per candidate:
    # P&L $ (14 wide) + % notional (8 wide) + marker " *" or "  " (2 wide) = 24.
    scenario_col_w = 22
    cell_w = 24
    header = " " * scenario_col_w
    for i, ev in enumerate(evals):
        h = ev.hedge
        tag = f"{_candidate_label(i)}) {h.strike:g}{h.option_type[0].upper()}/{h.expiry.strftime('%b%d')}"
        header += f"{tag:>{cell_w}s}"
    lines.append("")
    lines.append("Stress P&L (Δ$ total | % notional, * = highest in scenario):")
    lines.append(header)
    sub = " " * scenario_col_w
    for _ in evals:
        sub += f"{'P&L $':>14s}{'% not':>8s}{'':>2s}"
    lines.append(sub)
    lines.append("-" * (scenario_col_w + cell_w * len(evals)))

    # Scenario names must align across candidates. All evals should share
    # the same scenario sequence (we pass identical scenarios to each
    # evaluate_hedge call). We index by position into evals[0].scenarios.
    n_scenarios = len(evals[0].scenarios)
    if any(len(ev.scenarios) != n_scenarios for ev in evals):
        raise ValueError(
            "all evals must share the same scenario sequence — "
            f"got lengths {[len(ev.scenarios) for ev in evals]}"
        )
    for s_idx in range(n_scenarios):
        name = evals[0].scenarios[s_idx].scenario.name
        row = f"{name:<{scenario_col_w}s}"
        # Winner = highest pnl_total for this scenario across candidates.
        # Ties: first occurrence wins (max returns the first max).
        winner_idx = max(range(len(evals)),
                         key=lambda i: evals[i].scenarios[s_idx].pnl_total)
        for cand_idx, ev in enumerate(evals):
            s = ev.scenarios[s_idx]
            pnl_s   = f"{s.pnl_total:>+13,.0f}"
            pct_s   = f"{s.pnl_pct_of_notional*100:>+7.2f}%"
            marker  = " *" if cand_idx == winner_idx else "  "
            row += f"{pnl_s:>14s}{pct_s:>8s}{marker}"
        lines.append(row)

    return "\n".join(lines)


SORT_CHOICES = ("original", "premium", "coi", "breakeven-decline", "covid-pnl")


def _sort_key(ev: HedgeEvaluation, by: str) -> float:
    """Return the value to sort by. All sorts are ascending."""
    if by == "premium":
        return ev.hedge.premium_per_share
    if by == "coi":
        return ev.cost_of_insurance_pct
    if by == "breakeven-decline":
        return ev.breakeven_decline_pct
    if by == "covid-pnl":
        # COVID is the last (most severe) of DEFAULT_SCENARIOS; sort by
        # P&L total so highest-paying candidate ends up last.
        return ev.scenarios[-1].pnl_total
    raise ValueError(f"unknown sort key {by!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ticker", required=True,
                    help="Underlying symbol shared by all candidates (e.g. SPY)")
    ap.add_argument("--candidate", required=True, action="append",
                    metavar="K:EXPIRY:TYPE[:premium[:sigma[:atm]]]",
                    help="Candidate hedge spec. Required: strike:YYYY-MM-DD:put|call. "
                         "Optional trailing fields (premium, sigma, atm) override "
                         "live values; required in --no-fetch mode. "
                         "Pass --candidate multiple times for 2-N candidates.")
    ap.add_argument("--n", type=int, default=1,
                    help="Contracts per candidate (default 1 — multiply mentally)")
    ap.add_argument("--r", type=float, default=None,
                    help="Override risk-free rate. Default: latest from "
                         "data/risk_free_rate.csv")
    ap.add_argument("--q-override", type=float, default=None,
                    help="Override dividend yield (single value applied to all "
                         "expiries). Default: solve_q per expiry chain.")
    ap.add_argument("--n-steps", type=int, default=200,
                    help="LR tree step count (default: 200)")
    ap.add_argument("--vol-baseline", choices=["contract", "atm"],
                    default="contract",
                    help="contract (default): scenario vol shocks scale each "
                         "contract's own IV. atm: shocks scale same-expiry ATM "
                         "IV with absolute skew re-added (fixes wing-skew bias).")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Skip Polygon entirely. Requires --spot and each "
                         "--candidate to carry premium + sigma (and atm if "
                         "--vol-baseline atm).")
    ap.add_argument("--spot", type=float, default=None,
                    help="Override / required-for-no-fetch underlying spot.")
    ap.add_argument("--sort-by", choices=SORT_CHOICES, default="original",
                    help="Sort candidates before display. covid-pnl puts the "
                         "highest-paying-in-the-worst-scenario candidate last.")
    args = ap.parse_args()

    if args.n <= 0:
        print(f"[!] --n must be positive, got {args.n}")
        return 2
    if len(args.candidate) < 2:
        print("[!] need at least 2 --candidate flags for a side-by-side "
              "compare (use run_stress_hedge.py for a single candidate)")
        return 2

    # Parse candidate specs upfront — fail fast on bad input.
    try:
        specs = [parse_candidate_spec(s) for s in args.candidate]
    except ValueError as e:
        print(f"[!] {e}")
        return 2

    expiries = sorted({s.expiry for s in specs})
    if len(expiries) > 1:
        print(f"[note] {len(expiries)} different expiries — theta decay differs "
              f"per candidate. Per-candidate COI is comparable, but absolute "
              f"P&L is not strictly apples-to-apples (closer-dated options "
              f"lose extrinsic value faster).")
        print()

    r = args.r if args.r is not None else load_risk_free_rate()

    if args.no_fetch:
        spot, q_by_expiry, atm_iv_by_expiry = _resolve_offline(args, specs, r)
        chains: dict[date, pd.DataFrame] = {}
    else:
        chains, spot = _fetch_chains(args.ticker, expiries, specs)
        if spot is None:
            return 1
        q_by_expiry, atm_iv_by_expiry = _resolve_live(
            args, specs, chains, spot, r,
        )

    if spot is None or q_by_expiry is None:
        return 1

    # Build Hedge per candidate and evaluate. In live mode, locate the
    # contract in its chain; in offline mode, premium + sigma come from spec.
    evals: list[HedgeEvaluation] = []
    for spec in specs:
        sigma   = spec.sigma
        premium = spec.premium
        if not args.no_fetch:
            chain = chains[spec.expiry]
            match = chain[(chain["contract_type"] == spec.option_type)
                          & (chain["strike"].astype(float) == float(spec.strike))]
            if match.empty:
                print(f"[!] Contract not found in {spec.expiry} chain: "
                      f"{spec.option_type} strike={spec.strike}. "
                      f"Available {spec.option_type} strikes:")
                avail = sorted(chain[chain["contract_type"] == spec.option_type]
                               ["strike"].dropna().unique())
                print(" ", ", ".join(f"{s:g}" for s in avail[:25]),
                      "..." if len(avail) > 25 else "")
                return 1
            row = match.iloc[0]
            if sigma is None:
                sigma = row.get("polygon_iv")
            if premium is None:
                premium = pick_premium(row)
            if sigma is None:
                print(f"[!] No IV for {spec.option_type} K={spec.strike} "
                      f"exp {spec.expiry}. Try a more liquid strike or pass "
                      f"sigma in the spec.")
                return 1
            if premium is None:
                print(f"[!] No premium (bid/ask/close) for {spec.option_type} "
                      f"K={spec.strike} exp {spec.expiry}. Try a more liquid "
                      f"strike or pass premium in the spec.")
                return 1

        hedge = Hedge(ticker=args.ticker, option_type=spec.option_type,
                      strike=spec.strike, expiry=spec.expiry,
                      n_contracts=args.n, premium_per_share=float(premium))
        sigma_atm = (spec.sigma_atm if spec.sigma_atm is not None
                     else atm_iv_by_expiry.get(spec.expiry)) \
                    if args.vol_baseline == "atm" else None
        ev = evaluate_hedge(
            hedge, spot_today=spot, sigma_today=float(sigma),
            r=r, q=q_by_expiry[spec.expiry], n_steps=args.n_steps,
            vol_baseline=args.vol_baseline, sigma_atm=sigma_atm,
        )
        evals.append(ev)

    if args.sort_by != "original":
        evals = sorted(evals, key=lambda ev: _sort_key(ev, args.sort_by))

    print()
    print(format_compare_table(
        ticker=args.ticker, spot=spot, r=r, evals=evals,
        q_by_expiry=q_by_expiry,
        atm_iv_by_expiry=atm_iv_by_expiry if args.vol_baseline == "atm" else None,
    ))
    print()
    return 0


def _fetch_chains(ticker: str, expiries: list[date],
                  specs: list[CandidateSpec],
                  ) -> tuple[dict[date, pd.DataFrame], float | None]:
    """Pull one chain per unique expiry. Returns (chains, observed_spot).
    Spot is None on fetch/credential failure; the caller propagates exit 1."""
    try:
        key  = get_massive_key()
        base = get_massive_base()
    except RuntimeError as e:
        print(f"[!] {e}")
        return {}, None

    chains: dict[date, pd.DataFrame] = {}
    spot_observed: float | None = None
    for exp in expiries:
        print(f"Fetching {ticker} chain for {exp.isoformat()}...", flush=True)
        chain, sp = fetch_expiry_chain(ticker, exp, key, base)
        if chain.empty:
            print(f"[!] No contracts for {ticker} expiry {exp} "
                  f"(weekend / non-listed weekly).")
            opt_type = next(s.option_type for s in specs if s.expiry == exp)
            nearby = list_nearby_expiries(ticker, exp, opt_type,
                                          key, base, window_days=30)
            if nearby:
                print(f"  Available {opt_type} expiries within ±30d:")
                for e in nearby:
                    print(f"    {e}")
            return chains, None
        chains[exp] = chain
        if spot_observed is None and sp is not None:
            spot_observed = float(sp)
    if spot_observed is None:
        print("[!] Could not resolve underlying spot from any fetched chain.")
    return chains, spot_observed


def _resolve_live(args, specs: list[CandidateSpec],
                  chains: dict[date, pd.DataFrame], spot: float, r: float,
                  ) -> tuple[dict[date, float], dict[date, float]]:
    """In live mode, resolve q per expiry (PCP-median / fallback) and ATM IV
    per expiry (closest-strike). Returns (q_by_expiry, atm_iv_by_expiry).
    atm_iv_by_expiry is empty when vol_baseline != atm."""
    q_by_expiry: dict[date, float] = {}
    atm_iv_by_expiry: dict[date, float] = {}
    for exp, chain in chains.items():
        if args.q_override is not None:
            q_by_expiry[exp] = args.q_override
        else:
            q_res = solve_q(chain, args.ticker, r)
            q_by_expiry[exp] = q_res["q"]
            print(f"  {exp.isoformat()} q via {q_res['method']} "
                  f"(n_strikes={q_res['n_strikes']}): "
                  f"{q_by_expiry[exp]*100:+.3f}%")
        if args.vol_baseline == "atm":
            opt_type = next(s.option_type for s in specs if s.expiry == exp)
            # Spec-level atm override takes precedence; only fetch if absent.
            spec_atm = next((s.sigma_atm for s in specs
                             if s.expiry == exp and s.sigma_atm is not None),
                            None)
            if spec_atm is not None:
                atm_iv_by_expiry[exp] = spec_atm
                continue
            atm_iv, atm_k = pick_atm_iv(chain, opt_type, spot)
            if atm_iv is None:
                print(f"[!] vol_baseline atm requested but no same-type "
                      f"contract in the {exp} chain had a populated IV. "
                      f"Pass atm in the candidate spec, or drop "
                      f"--vol-baseline atm.")
                raise SystemExit(1)
            atm_iv_by_expiry[exp] = atm_iv
            print(f"  {exp.isoformat()} ATM IV (closest {opt_type} "
                  f"K={atm_k:g} to spot {spot:.2f}): {atm_iv*100:.2f}%")
    return q_by_expiry, atm_iv_by_expiry


def _resolve_offline(args, specs: list[CandidateSpec], r: float
                     ) -> tuple[float | None,
                                dict[date, float] | None,
                                dict[date, float]]:
    """In --no-fetch mode, validate that every spec carries the inputs we'd
    otherwise fetch (premium, sigma, and atm when applicable). Returns
    (spot, q_by_expiry, atm_iv_by_expiry) or (None, None, {}) on validation
    failure (with a printed error)."""
    if args.spot is None:
        print("[!] --no-fetch requires --spot")
        return None, None, {}

    missing: list[str] = []
    for s in specs:
        tag = f"{s.option_type} K={s.strike:g} exp {s.expiry.isoformat()}"
        if s.premium is None:
            missing.append(f"{tag}: premium")
        if s.sigma is None:
            missing.append(f"{tag}: sigma")
        if args.vol_baseline == "atm" and s.sigma_atm is None:
            missing.append(f"{tag}: atm")
    if missing:
        print("[!] --no-fetch missing required spec overrides:")
        for m in missing:
            print(f"    {m}")
        return None, None, {}

    if args.q_override is None:
        print("[!] --no-fetch requires --q-override (no chain to solve q on)")
        return None, None, {}

    expiries = sorted({s.expiry for s in specs})
    q_by_expiry = {exp: args.q_override for exp in expiries}
    atm_iv_by_expiry: dict[date, float] = {}
    if args.vol_baseline == "atm":
        for exp in expiries:
            # All specs at this expiry must agree on atm (we asserted above
            # that every spec has sigma_atm set, so this is safe).
            atms = {s.sigma_atm for s in specs if s.expiry == exp}
            if len(atms) > 1:
                print(f"[!] --no-fetch atm values disagree at {exp}: {atms}")
                return None, None, {}
            atm_iv_by_expiry[exp] = atms.pop()
    return args.spot, q_by_expiry, atm_iv_by_expiry


if __name__ == "__main__":
    raise SystemExit(main())
