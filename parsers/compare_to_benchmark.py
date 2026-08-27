"""Compare portfolio TWR and IRR to a benchmark's total-return series.

Aligns the benchmark on portfolio statement dates so the comparison is
month-end-to-month-end, same chaining as compute_twr.py.

A portfolio month is included only when BOTH endpoints (prev_stmt_date
and statement_date) fall inside the benchmark's data window — partial
periods would be unfair on either side.

TWR comparison: month-to-month return chaining over the matching window.

IRR comparison: builds a windowed cashflow stream
  [-window_start_NAV, every external txn flow at its real date,
   synthetic onboarding flows whose debut falls in window,
   +terminal_NAV]
and computes xirr() over it. Then builds a parallel SPY counterfactual:
identical cashflows but each one buys SPY at that date's total-return
value, and the terminal is the simulated SPY portfolio NAV — i.e. "what
if every deposit had gone into SPY instead?".

Inputs (no API):
  data/twr_portfolio.csv          (compute_twr.py output)
  data/transactions.csv           (for IRR cashflow detail)
  data/positions.csv              (for synthetic onboarding NAV lookup)
  data/benchmark_<ticker>_tr.csv  (build_benchmark_total_return.py output)

Outputs (CLI artifacts only — the dashboard computes the comparison LIVE via
build_twr_comparison / build_irr_comparison and never reads these CSVs, so a
stale file on disk does not affect any displayed number):
  data/comparison_<ticker>.csv          monthly TWR side-by-side rows
  data/comparison_<ticker>_summary.csv  one-row summary KPIs

Run:  py parsers\\compare_to_benchmark.py [TICKER]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Reuse xirr + monthly_navs from the existing TWR module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_twr import xirr, monthly_navs  # noqa: E402

# config_local lives at project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import config_local as cfg
except ImportError as e:
    raise RuntimeError(
        "config_local.py not found. Copy config_example.py to config_local.py."
    ) from e

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEFAULT_SYNTHETIC_ONBOARDING = cfg.SYNTHETIC_ONBOARDING


def benchmark_value_lookup(tr: pd.DataFrame) -> pd.Series:
    """TR value indexed by calendar day, forward-filled across non-trading days
    (so a Memorial-Day statement_date resolves to the prior Friday's close)."""
    s = tr.sort_values("date").set_index("date")["tr_value"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full).ffill()


def compute_twr_comparison(ticker: str, port: pd.DataFrame, tr_lookup: pd.Series):
    bench_start = tr_lookup.index.min()
    bench_end = tr_lookup.index.max()
    rows = []
    for _, p in port.iterrows():
        prev = p["prev_stmt_date"]
        end = p["statement_date"]
        if pd.isna(prev) or prev < bench_start or end > bench_end:
            continue
        rows.append({
            "month": p["month"],
            "statement_date": end.date(),
            "prev_stmt_date": prev.date(),
            "port_return": float(p["return_pct"]),
            "bench_return": float(tr_lookup.loc[end] / tr_lookup.loc[prev] - 1.0),
            "port_nav": float(p["nav"]),
            "bench_tr_value": float(tr_lookup.loc[end]),
        })
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp, None
    comp["spread"] = comp["port_return"] - comp["bench_return"]
    valid = comp.dropna(subset=["port_return", "bench_return"])
    port_cum = float(np.prod(1.0 + valid["port_return"]) - 1.0)
    bench_cum = float(np.prod(1.0 + valid["bench_return"]) - 1.0)
    years = len(valid) / 12.0
    summary = {
        "n_months": int(len(valid)),
        "years": years,
        "window_start": valid.iloc[0]["statement_date"],
        "window_end": valid.iloc[-1]["statement_date"],
        "port_twr_cum": port_cum,
        "bench_twr_cum": bench_cum,
        "port_twr_ann": (1.0 + port_cum) ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
        "bench_twr_ann": (1.0 + bench_cum) ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
    }
    return comp, summary


def compute_irr_comparison(port: pd.DataFrame, tr_lookup: pd.Series,
                           transactions: pd.DataFrame, positions: pd.DataFrame,
                           synthetic_onboarding: dict[str, str]):
    bench_start = tr_lookup.index.min()
    bench_end = tr_lookup.index.max()
    eligible = port[
        port["prev_stmt_date"].notna()
        & (port["prev_stmt_date"] >= bench_start)
        & (port["statement_date"] <= bench_end)
    ].sort_values("statement_date")
    if eligible.empty:
        return None

    window_start = pd.Timestamp(eligible.iloc[0]["prev_stmt_date"])
    window_start_nav = float(eligible.iloc[0]["prev_nav"])
    window_end = pd.Timestamp(eligible.iloc[-1]["statement_date"])
    window_end_nav = float(eligible.iloc[-1]["nav"])

    txn = transactions[transactions.get("flow_scope", "") == "external"].copy()
    txn_in = txn[
        (txn["settlement_date"] > window_start)
        & (txn["settlement_date"] <= window_end)
    ].sort_values("settlement_date")

    navs = monthly_navs(positions)
    synth_flows = []
    for acct, ym in (synthetic_onboarding or {}).items():
        debut = navs[(navs["account_id"] == acct) & (navs["month"].astype(str) == ym)]
        if not len(debut):
            continue
        debut_date = pd.Timestamp(debut.iloc[0]["statement_date"])
        if window_start < debut_date <= window_end:
            synth_flows.append({"date": debut_date, "amount": float(debut.iloc[0]["nav"])})

    # Shared cashflow stream (investor POV: deposits negative, withdrawals positive)
    cf_base: list[float] = [-window_start_nav]
    dt_base: list[pd.Timestamp] = [window_start]
    for _, f in txn_in.iterrows():
        cf_base.append(-float(f["amount"]))
        dt_base.append(pd.Timestamp(f["settlement_date"]))
    for s in synth_flows:
        cf_base.append(-float(s["amount"]))
        dt_base.append(s["date"])

    # Portfolio terminal
    cf_port = cf_base + [window_end_nav]
    dt_port = dt_base + [window_end]

    # SPY counterfactual terminal: same flows, each buys SPY shares at that date's TR value
    spy_shares = window_start_nav / float(tr_lookup.loc[window_start])
    for _, f in txn_in.iterrows():
        amt = float(f["amount"])
        d = pd.Timestamp(f["settlement_date"])
        spy_shares += amt / float(tr_lookup.loc[d])
    for s in synth_flows:
        spy_shares += float(s["amount"]) / float(tr_lookup.loc[s["date"]])
    spy_terminal = spy_shares * float(tr_lookup.loc[window_end])
    cf_spy = cf_base + [spy_terminal]
    dt_spy = dt_base + [window_end]

    return {
        "window_start": window_start,
        "window_end": window_end,
        "window_start_nav": window_start_nav,
        "window_end_nav": window_end_nav,
        "spy_terminal_nav": spy_terminal,
        "n_cashflows": len(cf_port),
        "n_real_flows": len(txn_in),
        "n_synth_flows": len(synth_flows),
        "total_deposits": float(sum(-c for c in cf_base if c < 0)),
        "total_withdrawals": float(sum(c for c in cf_base if c > 0)),
        "irr_port": xirr(cf_port, dt_port),
        "irr_spy": xirr(cf_spy, dt_spy),
    }


# ---------------------------------------------------------------------------
# Dashboard-side variants (used live, not from the CLI). Sibling to
# compute_twr_comparison / compute_irr_comparison above but with slightly
# different output shapes — these emit the wealth/drawdown columns and
# win-rate counts the Performance-vs-Benchmark tab plots, and the IRR
# variant accepts an `allowed_accounts` filter that mirrors the dashboard's
# Broker filter. The two pairs are intentionally kept distinct (different
# semantics, different consumers); they share the no-partial-period
# window-overlap rule and the SPY-counterfactual cashflow construction.
# ---------------------------------------------------------------------------


def build_twr_comparison(port_twr: pd.DataFrame,
                          tr_lookup: pd.Series,
                          base_amount: float = 100_000.0) -> dict | None:
    """Match portfolio monthly TWR against benchmark TR over the overlapping
    window. Returns dict with:
      - comp: row per included month (port_return, bench_return, spread,
              port_wealth, bench_wealth, port_dd_pct, bench_dd_pct)
      - summary: one-shot scalars (cum, ann, win-rate, max DD, final wealth)
    Or None if there is no overlap.
    """
    if port_twr.empty or tr_lookup.empty:
        return None
    bench_start = tr_lookup.index.min()
    bench_end = tr_lookup.index.max()
    rows: list[dict] = []
    for _, p in port_twr.sort_values("statement_date").iterrows():
        prev = p.get("prev_stmt_date")
        end = p["statement_date"]
        ret = p["return_pct"]
        if pd.isna(prev) or pd.isna(ret):
            continue
        if prev < bench_start or end > bench_end:
            continue
        rows.append({
            "month": p["month"],
            "statement_date": end,
            "prev_stmt_date": prev,
            "port_return": float(ret),
            "bench_return": float(tr_lookup.loc[end] / tr_lookup.loc[prev] - 1.0),
        })
    if not rows:
        return None
    comp = pd.DataFrame(rows)
    comp["spread"] = comp["port_return"] - comp["bench_return"]
    comp["port_wealth"] = base_amount * (1.0 + comp["port_return"]).cumprod()
    comp["bench_wealth"] = base_amount * (1.0 + comp["bench_return"]).cumprod()
    comp["port_dd_pct"] = (comp["port_wealth"] / comp["port_wealth"].cummax() - 1.0) * 100
    comp["bench_dd_pct"] = (comp["bench_wealth"] / comp["bench_wealth"].cummax() - 1.0) * 100

    n_months = len(comp)
    years = n_months / 12.0
    port_cum = float((1.0 + comp["port_return"]).prod() - 1.0)
    bench_cum = float((1.0 + comp["bench_return"]).prod() - 1.0)
    summary = {
        "n_months": n_months,
        "years": years,
        "window_start": comp.iloc[0]["prev_stmt_date"],
        "window_end": comp.iloc[-1]["statement_date"],
        "port_twr_cum": port_cum,
        "bench_twr_cum": bench_cum,
        "port_twr_ann": (1.0 + port_cum) ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
        "bench_twr_ann": (1.0 + bench_cum) ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
        "port_wealth_final": float(comp.iloc[-1]["port_wealth"]),
        "bench_wealth_final": float(comp.iloc[-1]["bench_wealth"]),
        "port_max_dd": float(comp["port_dd_pct"].min()),
        "port_max_dd_date": comp.loc[comp["port_dd_pct"].idxmin(), "statement_date"],
        "bench_max_dd": float(comp["bench_dd_pct"].min()),
        "bench_max_dd_date": comp.loc[comp["bench_dd_pct"].idxmin(), "statement_date"],
        "win_months": int((comp["spread"] > 0).sum()),
        "loss_months": int((comp["spread"] < 0).sum()),
        "tie_months": int((comp["spread"] == 0).sum()),
        "base_amount": base_amount,
    }
    return {"comp": comp, "summary": summary}


def build_irr_comparison(port_twr: pd.DataFrame,
                          transactions: pd.DataFrame,
                          positions: pd.DataFrame,
                          tr_lookup: pd.Series,
                          synthetic_onboarding: dict[str, str],
                          allowed_accounts: set[str] | None = None) -> dict | None:
    """Build windowed IRR comparison: portfolio cashflows vs SPY counterfactual.

    `allowed_accounts` restricts both the synthetic-onboarding accounts and
    the external-transaction set so the IRR is consistent with whatever
    broker subset `port_twr` was filtered to. When None, no extra filter is
    applied (caller is expected to have already filtered transactions).
    """
    if port_twr.empty or tr_lookup.empty:
        return None
    bench_start = tr_lookup.index.min()
    bench_end = tr_lookup.index.max()
    eligible = port_twr[
        port_twr["prev_stmt_date"].notna()
        & (port_twr["return_pct"].notna())
        & (port_twr["prev_stmt_date"] >= bench_start)
        & (port_twr["statement_date"] <= bench_end)
    ].sort_values("statement_date")
    if eligible.empty:
        return None

    window_start = pd.Timestamp(eligible.iloc[0]["prev_stmt_date"])
    window_start_nav = float(eligible.iloc[0]["prev_nav"])
    window_end = pd.Timestamp(eligible.iloc[-1]["statement_date"])
    window_end_nav = float(eligible.iloc[-1]["nav"])
    if not (window_start_nav > 0 and window_end_nav > 0):
        return None

    txn = transactions.copy() if not transactions.empty else pd.DataFrame()
    if not txn.empty and "flow_scope" in txn.columns:
        txn = txn[txn["flow_scope"] == "external"]
        if allowed_accounts is not None:
            txn = txn[txn["account_id"].isin(allowed_accounts)]
        txn_in = txn[
            (txn["settlement_date"] > window_start)
            & (txn["settlement_date"] <= window_end)
        ].sort_values("settlement_date")
    else:
        txn_in = pd.DataFrame()

    # Synthetic onboarding: only include accounts that exist in this subset
    # AND whose debut falls inside the matching window.
    synth_flows: list[dict] = []
    pos = positions.copy()
    pos["month"] = pos["statement_date"].dt.to_period("M")
    for acct, ym in synthetic_onboarding.items():
        if allowed_accounts is not None and acct not in allowed_accounts:
            continue
        debut = pos[(pos["account_id"] == acct) & (pos["month"].astype(str) == ym)]
        if debut.empty:
            continue
        debut_date = pd.Timestamp(debut["statement_date"].max())
        debut_nav = float(debut[debut["statement_date"] == debut_date]["market_value"].sum())
        if window_start < debut_date <= window_end and debut_nav > 0:
            synth_flows.append({"date": debut_date, "amount": debut_nav})

    cf_base: list[float] = [-window_start_nav]
    dt_base: list[pd.Timestamp] = [window_start]
    if not txn_in.empty:
        for _, f in txn_in.iterrows():
            cf_base.append(-float(f["amount"]))
            dt_base.append(pd.Timestamp(f["settlement_date"]))
    for sf in synth_flows:
        cf_base.append(-float(sf["amount"]))
        dt_base.append(sf["date"])

    cf_port = cf_base + [window_end_nav]
    dt_port = dt_base + [window_end]

    spy_shares = window_start_nav / float(tr_lookup.loc[window_start])
    if not txn_in.empty:
        for _, f in txn_in.iterrows():
            d = pd.Timestamp(f["settlement_date"])
            spy_shares += float(f["amount"]) / float(tr_lookup.loc[d])
    for sf in synth_flows:
        spy_shares += float(sf["amount"]) / float(tr_lookup.loc[sf["date"]])
    spy_terminal = spy_shares * float(tr_lookup.loc[window_end])
    cf_spy = cf_base + [spy_terminal]
    dt_spy = dt_base + [window_end]

    return {
        "window_start": window_start,
        "window_end": window_end,
        "window_start_nav": window_start_nav,
        "window_end_nav": window_end_nav,
        "spy_terminal_nav": spy_terminal,
        "n_real_flows": int(len(txn_in)),
        "n_synth_flows": len(synth_flows),
        "total_deposits": float(sum(-c for c in cf_base if c < 0)),
        "total_withdrawals": float(sum(c for c in cf_base if c > 0)),
        "irr_port": xirr(cf_port, dt_port),
        "irr_bench": xirr(cf_spy, dt_spy),
    }


def main(argv: list[str]) -> int:
    ticker = (argv[1] if len(argv) > 1 else "SPY").upper()

    port = pd.read_csv(DATA_DIR / "twr_portfolio.csv",
                       parse_dates=["statement_date", "prev_stmt_date"])
    tr = pd.read_csv(DATA_DIR / f"benchmark_{ticker.lower()}_tr.csv",
                     parse_dates=["date"])
    tr_lookup = benchmark_value_lookup(tr)

    comp, twr_sum = compute_twr_comparison(ticker, port, tr_lookup)
    if twr_sum is None:
        print("[WARN] no overlapping months — check benchmark date range")
        return 1
    comp.to_csv(DATA_DIR / f"comparison_{ticker.lower()}.csv", index=False)
    print(f"[OK] wrote data/comparison_{ticker.lower()}.csv  ({len(comp)} months)")

    txn = pd.read_csv(DATA_DIR / "transactions.csv",
                      parse_dates=["settlement_date", "trade_date"])
    pos = pd.read_csv(DATA_DIR / "positions.csv", parse_dates=["statement_date"])
    irr = compute_irr_comparison(port, tr_lookup, txn, pos, DEFAULT_SYNTHETIC_ONBOARDING)

    # ----- Output -----
    print()
    print("=" * 72)
    print(f"TWR comparison  (window: {twr_sum['window_start']} -> {twr_sum['window_end']}, "
          f"{twr_sum['n_months']} months ~ {twr_sum['years']:.2f}y)")
    print("=" * 72)
    print(f"  Portfolio cumulative TWR : {twr_sum['port_twr_cum']*100:+8.2f}%   "
          f"annualized: {twr_sum['port_twr_ann']*100:+6.2f}%")
    print(f"  {ticker} TR cumulative      : {twr_sum['bench_twr_cum']*100:+8.2f}%   "
          f"annualized: {twr_sum['bench_twr_ann']*100:+6.2f}%")
    print(f"  Spread                   : {(twr_sum['port_twr_cum']-twr_sum['bench_twr_cum'])*100:+8.2f}%   "
          f"annualized: {(twr_sum['port_twr_ann']-twr_sum['bench_twr_ann'])*100:+6.2f} pp/yr")

    wins = (comp["spread"] > 0).sum()
    losses = (comp["spread"] < 0).sum()
    print(f"  Win-rate                 : portfolio {wins} months  vs  {ticker} {losses} months")

    print()
    print("=" * 72)
    print(f"IRR comparison  (money-weighted, same window)")
    print("=" * 72)
    if irr is None:
        print("[WARN] could not build IRR comparison")
    else:
        print(f"  Window-start NAV ({irr['window_start'].date()}):   "
              f"${irr['window_start_nav']:>14,.0f}")
        print(f"  Real external txn flows in window:  {irr['n_real_flows']}")
        print(f"  Synthetic-onboarding flows:         {irr['n_synth_flows']}")
        print(f"  Total deposits  (incl synthetic):   ${irr['total_deposits']:>14,.0f}")
        print(f"  Total withdrawals:                  ${irr['total_withdrawals']:>14,.0f}")
        print(f"  Portfolio terminal NAV ({irr['window_end'].date()}): "
              f"${irr['window_end_nav']:>14,.0f}")
        print(f"  SPY counterfactual terminal:        ${irr['spy_terminal_nav']:>14,.0f}")
        print(f"  Terminal delta (Port - SPY):        "
              f"${irr['window_end_nav'] - irr['spy_terminal_nav']:>+14,.0f}")
        print()
        print(f"  Portfolio IRR (windowed) : {irr['irr_port']*100:+6.2f}%")
        print(f"  {ticker} counterfactual IRR : {irr['irr_spy']*100:+6.2f}%")
        print(f"  Spread                  : "
              f"{(irr['irr_port'] - irr['irr_spy'])*100:+6.2f} pp/yr")

    # ----- Persist summary KPIs for dashboard -----
    summary_row = {
        "ticker": ticker,
        "window_start": twr_sum["window_start"],
        "window_end": twr_sum["window_end"],
        "n_months": twr_sum["n_months"],
        "years": twr_sum["years"],
        "port_twr_cum": twr_sum["port_twr_cum"],
        "bench_twr_cum": twr_sum["bench_twr_cum"],
        "port_twr_ann": twr_sum["port_twr_ann"],
        "bench_twr_ann": twr_sum["bench_twr_ann"],
        "port_twr_win_months": int(wins),
        "bench_twr_win_months": int(losses),
    }
    if irr is not None:
        summary_row.update({
            "irr_port": irr["irr_port"],
            "irr_bench": irr["irr_spy"],
            "irr_window_start_nav": irr["window_start_nav"],
            "irr_window_end_nav": irr["window_end_nav"],
            "irr_bench_terminal_nav": irr["spy_terminal_nav"],
            "irr_total_deposits": irr["total_deposits"],
            "irr_total_withdrawals": irr["total_withdrawals"],
            "irr_n_real_flows": irr["n_real_flows"],
            "irr_n_synth_flows": irr["n_synth_flows"],
        })
    pd.DataFrame([summary_row]).to_csv(
        DATA_DIR / f"comparison_{ticker.lower()}_summary.csv", index=False
    )
    print()
    print(f"[OK] wrote data/comparison_{ticker.lower()}_summary.csv")
    print()
    print(f"  Top 3 months portfolio beat {ticker}:")
    for _, r in comp.nlargest(3, "spread").iterrows():
        print(f"    {r['month']}  port {r['port_return']*100:+6.2f}%  {ticker.lower()} "
              f"{r['bench_return']*100:+6.2f}%  spread {r['spread']*100:+6.2f}pp")
    print(f"  Top 3 months {ticker} beat portfolio:")
    for _, r in comp.nsmallest(3, "spread").iterrows():
        print(f"    {r['month']}  port {r['port_return']*100:+6.2f}%  {ticker.lower()} "
              f"{r['bench_return']*100:+6.2f}%  spread {r['spread']*100:+6.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
