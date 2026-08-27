"""Build a dividend-reinvested total-return series for a benchmark ticker.

Inputs (idempotent CSV-in/CSV-out, no API):
  data/benchmark_<ticker>.csv     prices, from fetch_benchmark.py
  data/dividends_<ticker>.csv     dividends, from fetch_dividends.py

Output:
  data/benchmark_<ticker>_tr.csv  one row per trading day with:
      date           trading day
      close          raw close
      shares         accumulated shares (starts at 1.0, grows via reinvestment)
      tr_value       shares * close (daily portfolio value)
      tr_index       tr_value rebased to 100 at the first row
      daily_return   tr_value pct change vs prior trading day

Reinvestment rule: on a ticker's ex-dividend day, the holder receives
shares * cash_amount, immediately reinvested at that day's close — the
standard total-return index construction.

Run:  py parsers\\build_benchmark_total_return.py --write [TICKER]
  py parsers\\build_benchmark_total_return.py             # smoke test, no CSV written
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_tr(ticker: str) -> pd.DataFrame:
    prices_path = DATA_DIR / f"benchmark_{ticker.lower()}.csv"
    divs_path = DATA_DIR / f"dividends_{ticker.lower()}.csv"
    prices = pd.read_csv(prices_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    divs = pd.read_csv(divs_path, parse_dates=["ex_dividend_date"]).sort_values("ex_dividend_date").reset_index(drop=True)

    # An empty dividends.csv (header-only) leaves ex_dividend_date as object
    # dtype since parse_dates can't infer from no rows; .dt would then raise.
    if divs.empty:
        div_map: dict = {}
    else:
        div_map = divs.groupby(divs["ex_dividend_date"].dt.date)["cash_amount"].sum().to_dict()

    shares = 1.0
    rows = []
    for _, r in prices.iterrows():
        d = r["date"].date()
        close = float(r["close"])
        if d in div_map:
            cash = shares * float(div_map[d])
            shares += cash / close
        rows.append({"date": d, "close": close, "shares": shares, "tr_value": shares * close})
    df = pd.DataFrame(rows)
    df["tr_index"] = df["tr_value"] / df["tr_value"].iloc[0] * 100.0
    df["daily_return"] = df["tr_value"].pct_change()
    return df


def build_blended_tr(components: list[tuple[float, pd.DataFrame]]) -> pd.DataFrame:
    """Daily constant-mix total-return index from weighted component TR frames.

    Each component is (weight, tr_df) where tr_df has `date` + `tr_value`.
    Rebalanced to the weights every trading day: r_blend[t] = Σ wᵢ·rᵢ[t],
    compounded into a tr_value/tr_index rebased to 100. Inner-aligned on the
    dates common to all components. Empty frame if any component is empty or
    the aligned intersection has < 2 rows.
    """
    weights = [w for w, _ in components]
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"blend weights must sum to 1.0, got {sum(weights)}")
    series = []
    for _, df in components:
        if df is None or df.empty or "date" not in df.columns or "tr_value" not in df.columns:
            return pd.DataFrame(columns=["date", "tr_value", "tr_index", "daily_return"])
        s = df.sort_values("date").set_index("date")["tr_value"].astype(float)
        series.append(s)
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) < 2:
        return pd.DataFrame(columns=["date", "tr_value", "tr_index", "daily_return"])
    daily = [s.reindex(idx).pct_change() for s in series]
    blend_ret = sum(w * r for (w, _), r in zip(components, daily))
    blend_ret.iloc[0] = 0.0
    tr_index = (1.0 + blend_ret).cumprod() * 100.0
    out = pd.DataFrame({
        "date": [d.date() if hasattr(d, "date") else d for d in idx],
        "tr_value": (tr_index * 1000.0).to_numpy(),   # arbitrary base; only ratios matter
        "tr_index": tr_index.to_numpy(),
        "daily_return": blend_ret.to_numpy(),
    })
    # Chained assignment (out["daily_return"].iloc[0] = ...) is a no-op under
    # pandas Copy-on-Write — use .loc on the frame directly.
    out.loc[out.index[0], "daily_return"] = float("nan")   # first bar has no prior return
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Write the CSV. Without this flag, runs as a smoke test only.")
    ap.add_argument("ticker", nargs="?", default="SPY")
    args = ap.parse_args(argv[1:])

    ticker = args.ticker.upper()
    df = build_tr(ticker)
    out = DATA_DIR / f"benchmark_{ticker.lower()}_tr.csv"
    if args.write:
        df.to_csv(out, index=False)

    prices = pd.read_csv(DATA_DIR / f"benchmark_{ticker.lower()}.csv", parse_dates=["date"]).sort_values("date")
    p0, p1 = float(prices["close"].iloc[0]), float(prices["close"].iloc[-1])
    tr0, tr1 = float(df["tr_value"].iloc[0]), float(df["tr_value"].iloc[-1])
    # 365-day basis matches Excel XIRR and compute_twr.py's xirr() — using
    # 365.25 here would drift the displayed CAGR vs the persisted IRR by
    # ~0.07pp/yr over a 6y window.
    years = (pd.to_datetime(df["date"].iloc[-1]) - pd.to_datetime(df["date"].iloc[0])).days / 365.0

    price_total = (p1 / p0 - 1.0) * 100.0
    price_cagr = ((p1 / p0) ** (1.0 / years) - 1.0) * 100.0
    tr_total = (tr1 / tr0 - 1.0) * 100.0
    tr_cagr = ((tr1 / tr0) ** (1.0 / years) - 1.0) * 100.0

    tag = "[OK]" if args.write else "[DRY]"
    suffix = "" if args.write else "  (use --write to persist)"
    print(f"{tag} {len(df)} bars  window: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}  ({years:.2f}y)")
    print(f"{tag} {'wrote' if args.write else 'would write'} {out}{suffix}")
    print()
    print(f"  Price-only total: {price_total:+.2f}%   CAGR: {price_cagr:+.2f}%")
    print(f"  Total-return    : {tr_total:+.2f}%   CAGR: {tr_cagr:+.2f}%")
    print(f"  Dividend uplift : {tr_cagr - price_cagr:+.2f} pp/yr")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
