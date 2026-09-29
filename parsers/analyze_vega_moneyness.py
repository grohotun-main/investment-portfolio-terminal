"""One-shot analysis: bucket vega errors by moneyness band.

Purpose: the headline "vega within 0.05/pct: 72.8%" from verify_options_pricer
averages across all moneyness levels. For tail-risk hedging with 10-20% OTM
puts (60-180 DTE), we need to know where the failures actually live. This
script re-runs the comparison and buckets by moneyness × DTE.

Run:
    py parsers/analyze_vega_moneyness.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from implied_dividend import solve_q  # noqa: E402
from options_pricer import binomial_american, black_scholes  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
CHAIN_CSV = DATA / "options_chains_sample.csv"
RF_CSV    = DATA / "risk_free_rate.csv"

N_STEPS = int(__import__("os").environ.get("VEGA_ANALYZE_N", "200"))
VEGA_THR = 0.05  # per 1%-point


def otm_pct(row) -> float:
    """Unified OTM-ness: positive = OTM, negative = ITM. Works for both
    calls and puts.
    """
    S = float(row["underlying_price"])
    K = float(row["strike"])
    if row["contract_type"] == "put":
        return (S - K) / S
    return (K - S) / S


def moneyness_bin(otm: float) -> str:
    if otm < -0.05:        return "ITM (>5%)"
    if otm < -0.02:        return "ITM 2-5%"
    if abs(otm) <= 0.02:   return "ATM ±2%"
    if otm <= 0.05:        return "OTM 2-5%"
    if otm <= 0.10:        return "OTM 5-10%"
    if otm <= 0.20:        return "OTM 10-20%"
    if otm <= 0.30:        return "OTM 20-30%"
    return "OTM 30%+"


BIN_ORDER = [
    "ITM (>5%)", "ITM 2-5%", "ATM ±2%", "OTM 2-5%", "OTM 5-10%",
    "OTM 10-20%", "OTM 20-30%", "OTM 30%+",
]


def dte_bin(dte: int) -> str:
    if dte <= 30:  return "0-30 DTE"
    if dte <= 60:  return "30-60 DTE"
    if dte <= 180: return "60-180 DTE"
    return "180+ DTE"


DTE_ORDER = ["0-30 DTE", "30-60 DTE", "60-180 DTE", "180+ DTE"]


def load_risk_free_rate() -> float:
    df = pd.read_csv(RF_CSV, parse_dates=["date"])
    df = df.dropna(subset=["rate_annual"])
    return float(df.sort_values("date").iloc[-1]["rate_annual"])


def main() -> int:
    df = pd.read_csv(CHAIN_CSV, parse_dates=["expiration_date", "fetched_at"])
    print(f"Loaded {len(df)} contracts ({df['underlying'].nunique()} underlyings)")
    r = load_risk_free_rate()
    print(f"Risk-free: {r*100:.2f}%")

    q_table: dict[tuple, float] = {}
    for (ul, exp), g in df.groupby(["underlying", "expiration_date"]):
        q_table[(ul, exp)] = solve_q(g, ul, r)["q"]

    work = df.dropna(subset=["polygon_iv", "polygon_vega",
                             "polygon_price", "underlying_price"]).copy()
    print(f"Comparable contracts: {len(work)}")
    print(f"Re-pricing with binomial_american(n_steps={N_STEPS})...")

    rows: list[dict] = []
    t0 = time.time()
    for i, (_, row) in enumerate(work.iterrows()):
        if i % 500 == 0 and i > 0:
            rate = i / (time.time() - t0)
            eta  = (len(work) - i) / rate
            print(f"  {i:>5d}/{len(work)}  ({rate:.0f}/s, ETA {eta:.0f}s)")

        ul, exp = row["underlying"], row["expiration_date"]
        q = q_table[(ul, exp)]
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
                n_steps=N_STEPS,
            )
        except Exception:
            continue

        # BS European vega — what polygon SHOULD report if they're using
        # their own IV. If polygon_vega disagrees with this, polygon's vega
        # field is inconsistent with polygon's IV field.
        bs_vega_per_pct = black_scholes(
            float(row["underlying_price"]), float(row["strike"]),
            T, r, q, float(row["polygon_iv"]), row["contract_type"]
        )["vega"] * 0.01

        otm = otm_pct(row)
        our_vega_per_pct = ours["vega"] * 0.01
        polygon_vega = float(row["polygon_vega"])
        rows.append({
            "underlying": ul,
            "type": row["contract_type"],
            "dte": int(row["dte"]),
            "dte_bin": dte_bin(int(row["dte"])),
            "otm_pct": otm,
            "moneyness_bin": moneyness_bin(otm),
            "our_vega": our_vega_per_pct,
            "polygon_vega": polygon_vega,
            "bs_vega": bs_vega_per_pct,
            "abs_err": abs(our_vega_per_pct - polygon_vega),
            "polygon_vs_bs": abs(polygon_vega - bs_vega_per_pct),
            "polygon_consistent": abs(polygon_vega - bs_vega_per_pct) <= 0.05,
        })

    res = pd.DataFrame(rows)
    print(f"\nPriced {len(res)} contracts in {time.time()-t0:.1f}s\n")

    # Polygon self-consistency
    consistent_pct = res["polygon_consistent"].mean() * 100
    print(f"Polygon vega self-consistency (within 0.05 of BS-from-polyIV): "
          f"{consistent_pct:.1f}%")
    print(f"  → {(100-consistent_pct):.1f}% of contracts have polygon's vega disagreeing")
    print(f"    with polygon's own IV by more than 0.05.\n")

    # By moneyness alone
    print("=" * 78)
    print("VEGA ACCURACY BY MONEYNESS")
    print("=" * 78)
    print(f"{'Bin':<14s} {'n':>5s}  {'median ν':>9s}  {'med |err|':>10s}"
          f"  {'p95 |err|':>10s}  {'% within 0.05':>14s}")
    print("-" * 78)
    for b in BIN_ORDER:
        g = res[res["moneyness_bin"] == b]
        if g.empty:
            continue
        med_vega   = g["polygon_vega"].median()
        med_err    = g["abs_err"].median()
        p95_err    = g["abs_err"].quantile(0.95)
        within     = (g["abs_err"] <= VEGA_THR).mean() * 100
        print(f"{b:<14s} {len(g):>5d}  {med_vega:>9.4f}  {med_err:>10.4f}"
              f"  {p95_err:>10.4f}  {within:>13.1f}%")

    # Moneyness × DTE: focus on hedging-relevant cells
    print()
    print("=" * 78)
    print("VEGA % WITHIN 0.05 — MONEYNESS × DTE (n in parens)")
    print("=" * 78)
    header = f"{'Bin':<14s}  " + "  ".join(f"{d:>14s}" for d in DTE_ORDER)
    print(header)
    print("-" * len(header))
    for b in BIN_ORDER:
        line = f"{b:<14s}  "
        for d in DTE_ORDER:
            g = res[(res["moneyness_bin"] == b) & (res["dte_bin"] == d)]
            if g.empty:
                line += f"{'—':>14s}  "
            else:
                within = (g["abs_err"] <= VEGA_THR).mean() * 100
                line += f"{within:>7.1f}% ({len(g):>3d})  "
        print(line)

    # Hedging-relevant subset: 5-20% OTM puts, 30-180 DTE
    print()
    print("=" * 78)
    print("HEDGING ZOOM: 5-20% OTM PUTS, 30-180 DTE")
    print("=" * 78)
    hedge = res[
        (res["type"] == "put") &
        (res["moneyness_bin"].isin(["OTM 5-10%", "OTM 10-20%"])) &
        (res["dte_bin"].isin(["30-60 DTE", "60-180 DTE"]))
    ]
    if hedge.empty:
        print("  (no contracts in this band — sample lacks tail-hedge candidates)")
    else:
        print(f"  ALL contracts in band (n={len(hedge)}):")
        print(f"    median polygon vega: {hedge['polygon_vega'].median():.4f}")
        print(f"    median abs err:      {hedge['abs_err'].median():.4f}")
        print(f"    p95 abs err:         {hedge['abs_err'].quantile(0.95):.4f}")
        print(f"    within 0.05:         {(hedge['abs_err'] <= VEGA_THR).mean()*100:.1f}%")

        hedge_clean = hedge[hedge["polygon_consistent"]]
        print(f"\n  EXCLUDING polygon-inconsistent contracts (n={len(hedge_clean)} of {len(hedge)}):")
        print(f"    median abs err:      {hedge_clean['abs_err'].median():.4f}")
        print(f"    p95 abs err:         {hedge_clean['abs_err'].quantile(0.95):.4f}")
        print(f"    within 0.05:         {(hedge_clean['abs_err'] <= VEGA_THR).mean()*100:.1f}%")
        print(f"    within 0.02:         {(hedge_clean['abs_err'] <= 0.02).mean()*100:.1f}%")

        # Our vega vs BS (should match very closely since both use same σ)
        our_vs_bs = (hedge_clean["our_vega"] - hedge_clean["bs_vega"]).abs()
        print(f"\n  Our LR American vs BS European vega (same band, clean):")
        print(f"    median |our - bs|:  {our_vs_bs.median():.4f}  "
              f"(this is the American premium)")
        print(f"    p95  |our - bs|:    {our_vs_bs.quantile(0.95):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
