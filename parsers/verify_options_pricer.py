"""Verify parsers/options_pricer.py against Polygon's reported Greeks.

LOGIC
-----
For each contract in data/options_chains_sample.csv (must be fresh — re-run
fetch_options_chains.py --write first):

  1. T = (expiration_date - today) / 365.
  2. r = latest 3-month T-bill rate from data/risk_free_rate.csv.
  3. q = per-(underlying, expiry) dividend yield from implied_dividend.solve_q
     — median of PCP-implied q across many liquid near-ATM strikes, with a
     hardcoded trailing-yield fallback. Phase A's single-strike solver
     produced negative q for QQQ/NVDA/AAPL; this fixes it.
  4. sigma = Polygon's reported implied_volatility.
  5. Our Greeks via binomial_american(spot, K, T, r, q, sigma, opt, n=100).
     n=100 (vs the module's default 200) trades ~5e-4 abs accuracy for 2x
     speed — the tests already lock in n=200 convergence.
  6. Compare to Polygon's reported delta/gamma/vega/theta after unit
     conversion (our vega is per 1.00 vol unit, Polygon's is per 1% point;
     our theta is per year, Polygon's is per calendar day).

OUTPUTS
-------
Prints per-underlying summary stats:
  - median / 95th-percentile absolute error per Greek
  - sign agreement (% of contracts where our_X and polygon_X have same sign)

No file is written. If a CSV of per-contract errors is wanted, pipe stdout
or extend with --write.

Run:
  py parsers/verify_options_pricer.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from implied_dividend import solve_q  # noqa: E402
from options_pricer import binomial_american, implied_vol  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
CHAIN_CSV = DATA / "options_chains_sample.csv"
RF_CSV    = DATA / "risk_free_rate.csv"

N_STEPS_VERIFY = 100  # accuracy/speed tradeoff for the verification pass

# Acceptance thresholds. These are loose because Polygon's r and q estimates
# differ slightly from ours — the goal is to catch outright bugs, not 5th-
# decimal precision.
ACCEPT = {
    "delta": 0.02,   # absolute
    "gamma": 0.005,
    "vega":  0.05,   # in "per 1%" units
    "theta": 0.10,   # in "per day" units
}


def load_risk_free_rate() -> float:
    df = pd.read_csv(RF_CSV, parse_dates=["date"])
    df = df.dropna(subset=["rate_annual"])
    latest = df.sort_values("date").iloc[-1]
    return float(latest["rate_annual"])


def summarize(res: pd.DataFrame) -> None:
    """Print per-underlying error stats and the count outside threshold."""
    if res.empty:
        print("[!] No comparable contracts found.")
        return
    print()
    print(f"Compared {len(res)} contracts across {res['underlying'].nunique()} underlyings.")
    print()
    print(f"{'Underlying':<10s} {'n':>5s}  "
          f"{'Δ med|p95':>14s}  {'Γ med|p95':>14s}  "
          f"{'ν med|p95':>14s}  {'Θ med|p95':>14s}")
    print("-" * 80)
    for ul, g in res.groupby("underlying"):
        line = f"{ul:<10s} {len(g):>5d}  "
        for greek in ("delta", "gamma", "vega", "theta"):
            med = g[f"abs_err_{greek}"].median()
            p95 = g[f"abs_err_{greek}"].quantile(0.95)
            line += f"{med:>6.4f}|{p95:>6.4f}  "
        print(line)

    print()
    print("Fraction of contracts within acceptance threshold:")
    for greek, thr in ACCEPT.items():
        col = f"abs_err_{greek}"
        within = (res[col] <= thr).mean()
        print(f"  |Δ{greek}| ≤ {thr:.3f}:  {within*100:5.1f}%  "
              f"(worst: {res[col].max():.4f})")

    print()
    print("Sign agreement (% same-sign as Polygon):")
    for greek in ("delta", "vega", "theta"):
        ours_col = f"our_{greek}_compare"
        pol_col  = f"polygon_{greek}"
        agree = ((np.sign(res[ours_col]) == np.sign(res[pol_col]))
                 | (res[ours_col].abs() < 1e-9)).mean()
        print(f"  {greek}:  {agree*100:5.1f}%")

    # Also: implied-vol round-trip — back IV out of Polygon's market price
    # using OUR pricer, compare to Polygon's IV.
    iv_compare = res.dropna(subset=["our_iv"])
    if not iv_compare.empty:
        err = (iv_compare["our_iv"] - iv_compare["polygon_iv"]).abs()
        print()
        print(f"Implied-vol round-trip ({len(iv_compare)} contracts where we recovered IV):")
        print(f"  median |Δσ|: {err.median():.5f}")
        print(f"  95th  |Δσ|: {err.quantile(0.95):.5f}")
        print(f"  max   |Δσ|: {err.max():.5f}")


def main() -> int:
    if not CHAIN_CSV.exists():
        print(f"[!] {CHAIN_CSV} not found. Run:")
        print(f"      py parsers/fetch_options_chains.py --write")
        return 1

    df = pd.read_csv(CHAIN_CSV, parse_dates=["expiration_date", "fetched_at"])
    print(f"Loaded {len(df)} contracts ({df['underlying'].nunique()} underlyings)")
    r = load_risk_free_rate()
    print(f"Risk-free rate (latest from FRED): {r*100:.2f}%")

    # Resolve q per (underlying, expiry) via the multi-tier solver.
    q_table: dict[tuple, dict] = {}
    for (ul, exp), g in df.groupby(["underlying", "expiration_date"]):
        q_table[(ul, exp)] = solve_q(g, ul, r)

    print()
    print("Dividend yield resolution by underlying (median across expiries):")
    for ul in sorted(df["underlying"].unique()):
        entries = [v for (u, _), v in q_table.items() if u == ul]
        if not entries:
            continue
        qs = [e["q"] for e in entries]
        from collections import Counter
        methods = Counter(e["method"] for e in entries)
        method_str = ", ".join(f"{m}={n}" for m, n in methods.most_common())
        print(f"  {ul}: q={np.median(qs)*100:+.3f}%  "
              f"(expiries={len(entries)}, methods: {method_str})")

    # Iterate contracts, compute our Greeks + IV
    work = df.dropna(subset=["polygon_iv", "polygon_delta",
                             "polygon_price", "underlying_price"]).copy()
    print()
    print(f"Comparable contracts (have IV + Greeks + price): {len(work)}")
    print(f"Pricing each with binomial_american(n_steps={N_STEPS_VERIFY})...")

    results: list[dict] = []
    t0 = time.time()
    for i, (_, row) in enumerate(work.iterrows()):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(work) - i) / rate
            print(f"  {i:>5d}/{len(work)}  ({rate:.0f} contracts/s, ETA {eta:.0f}s)")

        ul, exp = row["underlying"], row["expiration_date"]
        q = q_table[(ul, exp)]["q"]
        T = float(row["dte"]) / 365.0
        if T <= 0:
            continue
        try:
            ours = binomial_american(
                spot=float(row["underlying_price"]),
                strike=float(row["strike"]),
                T=T, r=r, q=q,
                sigma=float(row["polygon_iv"]),
                opt=row["contract_type"],
                n_steps=N_STEPS_VERIFY,
            )
        except Exception as e:
            continue

        # IV round-trip: back σ out from Polygon's market price using our pricer
        try:
            our_iv = implied_vol(
                market_price=float(row["polygon_price"]),
                spot=float(row["underlying_price"]),
                strike=float(row["strike"]),
                T=T, r=r, q=q,
                opt=row["contract_type"],
                exercise="american",
                n_steps=N_STEPS_VERIFY,
                initial_guess=float(row["polygon_iv"]),
            )
        except Exception:
            our_iv = float("nan")

        results.append({
            "underlying": ul,
            "contract_ticker": row["contract_ticker"],
            "type": row["contract_type"],
            "strike": float(row["strike"]),
            "dte": int(row["dte"]),
            "spot": float(row["underlying_price"]),
            # Greeks in Polygon's display units for direct comparison
            "our_delta_compare":  ours["delta"],
            "polygon_delta":      float(row["polygon_delta"]),
            "our_gamma_compare":  ours["gamma"],
            "polygon_gamma":      float(row["polygon_gamma"]),
            "our_vega_compare":   ours["vega"] * 0.01,   # per 1%
            "polygon_vega":       float(row["polygon_vega"]),
            "our_theta_compare":  ours["theta"] / 365.0, # per day
            "polygon_theta":      float(row["polygon_theta"]),
            "our_iv":             our_iv,
            "polygon_iv":         float(row["polygon_iv"]),
            "our_price":          ours["price"],
            "polygon_price":      float(row["polygon_price"]),
        })

    res = pd.DataFrame(results)
    for g in ("delta", "gamma", "vega", "theta"):
        res[f"err_{g}"]     = res[f"our_{g}_compare"] - res[f"polygon_{g}"]
        res[f"abs_err_{g}"] = res[f"err_{g}"].abs()

    summarize(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
