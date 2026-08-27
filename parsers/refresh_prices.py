"""
One-command refresh of every external data CSV the dashboard reads: four
Polygon-sourced price files plus the three free / sparse-cadence CSVs (FRED
risk-free rate, CBOE VIX history, SPY dividend ex-dates), and the bench TR
build step that combines two of those inputs.

Why this exists
---------------
There are four Polygon fetchers, each landing a different file:

    parsers/fetch_holding_prices.py   ->  data/prices_latest.csv
    parsers/fetch_daily_prices.py     ->  data/daily_prices.csv
    parsers/fetch_long_history.py     ->  data/long_history_prices.csv
    parsers/fetch_benchmark.py        ->  data/benchmark_spy.csv

Running any single one in isolation is a footgun: the dashboard mixes them
(live MTM overlay reads prices_latest; risk math reads daily_prices;
Big-3 correlation matrix reads long_history; β / α / SPY comparison reads
benchmark_spy_tr). When one is fresh and the others stale, the "as of"
date the user sees in different panels disagrees. That class of bug was
flagged by the May 2026 audit (#20, #23) and the Phase 1B audit caught
that fetch_benchmark / fetch_dividends / build_benchmark_total_return
were absent from this script (5-day SPY TR staleness silently biased β/α
via reindex+ffill).

This script runs all Polygon fetches first, prints each file's as-of
date when done, and warns if they don't agree on the most recent trading
day.

Four "extra" CSV updaters run in a second pass:

    parsers/fetch_risk_free_rate.py   ->  data/risk_free_rate.csv     (FRED DGS3MO)
    parsers/fetch_vix.py              ->  data/vix_history.csv        (CBOE ^VIX)
    parsers/fetch_dividends.py        ->  data/dividends_spy.csv      (SPY ex-divs)
    parsers/fetch_ff_factors.py       ->  data/ff_factors_monthly.csv  (Ken French factors)

FRED + CBOE settle one business day in arrears (their CSV dates frequently
lag Polygon's intraday-close by a day during pre-close runs). Dividends
are emitted sparsely (~4 SPY ex-dates / year), so the dividend file's
last date is the most recent ex-date, not "today." All four are reported
alongside the Polygon dates but NOT required to match the Polygon trading
day. French factor data publishes monthly with a multi-week lag, so its
file's last month trails by design.

A post-step transforms the bench inputs into a total-return series:

    parsers/build_benchmark_total_return.py  -> data/benchmark_spy_tr.csv

Run:
    py parsers/refresh_prices.py              # all of the above, --write
    py parsers/refresh_prices.py --dry-run    # smoke-test, no CSV writes
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# (script_relative_to_root, output_csv, date_column, optional_status_filter,
#  extra_argv)
# `status_filter` excludes rows whose `status` column is in the set — used for
# prices_latest, where cash sweep rows always carry today's calendar date and
# would mask a stale equity-side fetch. `extra_argv` is per-entry positional
# args (the AGG benchmark leg reruns the same scripts with an explicit
# ticker).
#
# All Polygon fetchers default their end-date to today (Polygon silently
# omits an unpublished current-day bar). They are therefore expected to land
# on the same trading day in the strict cross-source date-alignment check
# below.
FETCHERS = [
    ("parsers/fetch_holding_prices.py", DATA / "prices_latest.csv",        "as_of_date", {"cash_fixed_1"}, []),
    ("parsers/fetch_daily_prices.py",   DATA / "daily_prices.csv",         "date",       set(), []),
    ("parsers/fetch_long_history.py",   DATA / "long_history_prices.csv",  "date",       set(), []),
    ("parsers/fetch_benchmark.py",      DATA / "benchmark_spy.csv",        "date",       set(), []),
    # The 60/40 blend's AGG leg. It was orchestration-orphaned (nothing
    # refreshed it, no guard watched it), so it went stale and the blend's
    # inner-join silently truncated every JPM-scoped daily benchmark series
    # while risk_bundle ffilled 0% benchmark days into β/α (DA-B-2).
    ("parsers/fetch_benchmark.py",      DATA / "benchmark_agg.csv",        "date",       set(), ["AGG"]),
]

# Non-strict CSV updaters run in a second pass — excluded from the cross-source
# date-alignment check below. FRED + CBOE settle one biz-day in arrears;
# fetch_dividends emits one row per ex-date (~quarterly for SPY/AGG), so its
# file's last date is the most recent dividend, not "today." Ken French
# factors publish monthly with a multi-week lag, so the file's last month
# trails the current trading day by design.
EXTRA_FETCHERS = [
    ("parsers/fetch_risk_free_rate.py", DATA / "risk_free_rate.csv",       "date",             set(), []),
    ("parsers/fetch_vix.py",            DATA / "vix_history.csv",          "date",             set(), []),
    ("parsers/fetch_dividends.py",      DATA / "dividends_spy.csv",        "ex_dividend_date", set(), []),
    # AGG ∉ daily_prices, so the --holdings universe sweep above never
    # covers its dividend file — the AGG TR build needs an explicit pull.
    ("parsers/fetch_dividends.py",      DATA / "dividends_agg.csv",        "ex_dividend_date", set(), ["AGG"]),
    ("parsers/fetch_ff_factors.py",     DATA / "ff_factors_monthly.csv",   "month",            set(), []),
]

# Post-fetch transform steps. These consume one or more freshly-written CSVs
# and emit a derived file. They're sequenced after FETCHERS+EXTRA_FETCHERS so
# their inputs are current. The bench TR build was missing from the
# orchestration before the Phase 1B audit (5-day TR staleness silently
# biased β/α via reindex+ffill into the daily_prices calendar).
POST_STEPS = [
    ("parsers/build_benchmark_total_return.py", DATA / "benchmark_spy_tr.csv", "date", set(), []),
    ("parsers/build_benchmark_total_return.py", DATA / "benchmark_agg_tr.csv", "date", set(), ["AGG"]),
]

# Fetchers that support incremental --full (re-pull full history). The others
# ignore the flag.
INCREMENTAL_FETCHERS = frozenset({
    "parsers/fetch_daily_prices.py",
    "parsers/fetch_long_history.py",
    "parsers/fetch_benchmark.py",
    "parsers/fetch_dividends.py",     # market-wide delta since the last run; --full = sweep
})


def _extra_flags(script_rel: str, full: bool,
                 extra_argv: list[str] | None = None) -> list[str]:
    flags = ["--full"] if (full and script_rel in INCREMENTAL_FETCHERS) else []
    if script_rel == "parsers/fetch_dividends.py" and not extra_argv:
        # The whole close-matrix universe + splits.csv, every refresh — the
        # total-return adjustment (parsers/total_return.py) reads them; SPY's
        # file (the benchmark TR input) is part of that universe. An entry
        # with an explicit ticker (the AGG leg) is a single-ticker pull, not
        # a second sweep.
        flags.append("--holdings")
    return flags


def _max_date(csv: Path, col: str, exclude_status: set[str]) -> str | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if exclude_status and "status" in df.columns:
        df = df[~df["status"].isin(exclude_status)]
    series = pd.to_datetime(df[col], errors="coerce").dropna()
    return series.max().date().isoformat() if len(series) else None


def bench_tr_staleness_days(data_dir: Path, ticker: str = "SPY") -> int | None:
    """Trading-day gap between benchmark_<ticker>_tr.csv and daily_prices.csv.

    Returns the number of trading days the bench TR file is behind the
    daily-prices file. Positive means TR is stale and the dashboard's
    `bench_tr.reindex(daily_prices.index, method='ffill')` is silently
    forward-filling the benchmark across the gap into β/α/spread math —
    for the AGG leg it also truncates the 60/40 blend's inner join, which
    is what silently hid the vs-Benchmark provisional segment (DA-C-10).

    None when either file is missing or unreadable.
    Zero or negative when TR is current (negative shouldn't happen in
    practice since TR is derived from bench prices; treat as "current").

    The unit is *trading days* — derived from daily_prices' own index so
    weekends and exchange holidays don't get counted as staleness.
    """
    bench_csv = data_dir / f"benchmark_{ticker.lower()}_tr.csv"
    daily_csv = data_dir / "daily_prices.csv"
    if not bench_csv.exists() or not daily_csv.exists():
        return None
    try:
        bench_last = pd.to_datetime(
            pd.read_csv(bench_csv, usecols=["date"])["date"]
        ).max()
        daily_dates = pd.to_datetime(
            pd.read_csv(daily_csv, usecols=["date"])["date"]
        )
    except (ValueError, KeyError):
        return None
    if pd.isna(bench_last) or daily_dates.empty:
        return None
    unique_trading_days = daily_dates.drop_duplicates().sort_values()
    later = unique_trading_days[unique_trading_days > bench_last]
    return int(len(later))


def classify_exit(strict_failures: list[str], soft_failures: list[str],
                  date_mismatch: bool) -> int:
    """Refresh exit-code policy.

    Fatal (nonzero): a cross-source date mismatch among the strict Polygon
    fetchers (2), or a failure of a strict fetcher / the derived TR build (1).
    Soft failures — the non-strict free sources (FRED RF, CBOE VIX, SPY
    dividends) — only WARN and never set a nonzero code: a flaky or timed-out
    free feed must not sink the whole refresh (it previously made the
    'Refresh all data' button report failure on a FRED ReadTimeout). The
    affected CSV simply keeps its prior value until the next successful run.
    """
    if date_mismatch:
        return 2
    if strict_failures:
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run each fetcher without --write (smoke test only)")
    ap.add_argument("--full", action="store_true",
                    help="Force a complete re-pull on the incremental fetchers "
                         "(daily prices, long history, benchmark).")
    args = ap.parse_args()

    write_flag = [] if args.dry_run else ["--write"]
    soft_scripts = {script for (script, *_rest) in EXTRA_FETCHERS}
    strict_failures: list[str] = []
    soft_failures: list[str] = []
    for script_rel, _out_csv, _, _, extra in (*FETCHERS, *EXTRA_FETCHERS,
                                              *POST_STEPS):
        label = f"{script_rel} {' '.join(extra)}".rstrip()
        print(f"\n=== {label} ===")
        result = subprocess.run(
            [sys.executable, str(ROOT / script_rel), *extra, *write_flag,
             *_extra_flags(script_rel, args.full, extra)],
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            bucket = soft_failures if script_rel in soft_scripts else strict_failures
            bucket.append(script_rel)
            print(f"[!] {script_rel} exited {result.returncode}")

    print("\n" + "=" * 62)
    print("As-of date per output CSV (cash rows excluded for prices_latest):")
    dates: dict[str, str | None] = {}
    for _, out_csv, col, exclude, _extra in FETCHERS:
        d = _max_date(out_csv, col, exclude)
        dates[out_csv.name] = d
        print(f"  {out_csv.name:30s}  {d or '(not written)'}")
    for _, out_csv, col, exclude, _extra in EXTRA_FETCHERS:
        d = _max_date(out_csv, col, exclude)
        print(f"  {out_csv.name:30s}  {d or '(not written)'}  (sparse / biz-day lag OK)")
    for _, out_csv, col, exclude, _extra in POST_STEPS:
        d = _max_date(out_csv, col, exclude)
        print(f"  {out_csv.name:30s}  {d or '(not written)'}  (derived from above)")

    real_dates = [d for d in dates.values() if d]
    date_mismatch = bool(real_dates) and len(set(real_dates)) > 1
    if date_mismatch:
        print(f"\n[!] Date mismatch: {sorted(set(real_dates))}")
        print("    One or more fetchers landed an older trading day.")

    # Non-strict free sources (FRED / VIX / dividends) only WARN — a flaky or
    # timed-out free feed must not fail the whole refresh.
    if soft_failures:
        print(f"\n[warn] {len(soft_failures)} non-strict free-source fetcher(s) "
              f"failed — refresh still OK, those CSVs keep their prior values: "
              f"{soft_failures}")
    if strict_failures:
        print(f"\n[!] {len(strict_failures)} strict fetcher(s) failed: "
              f"{strict_failures}")

    rc = classify_exit(strict_failures, soft_failures, date_mismatch)
    if rc == 0:
        print(f"\n[OK] All {len(FETCHERS)} Polygon CSVs land on the same trading day.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
