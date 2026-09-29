"""Fetch the ATM IV term structure per sleeve underlying, then derive a
constant-maturity series for the IV-percentile gauge.

Two artifacts:
  * ``data/atm_iv_term_history.csv`` — RAW: the ATM put IV across the next
    ~5 listed monthlies per (underlying, day). Expensive, permanent on
    disk, fetched incrementally.
  * ``data/atm_iv_history.csv`` — DERIVED: one constant-maturity (90-day)
    ATM IV per (underlying, day), interpolated in total variance from the
    term file. Cheap; fully re-derived each run. The gauge consumes this
    (columns date / underlying / atm_iv unchanged from before).

Why constant-maturity: front-month IV measures a *moving* maturity (DTE
sawtooths ~35→1 and jumps each roll), so a trailing-window percentile
compares readings taken at different points on the term structure. Pinning
the maturity makes the percentile apples-to-apples. See iv_constant_maturity.

Why this exists: Polygon's snapshot endpoint gives live IV but no history.
We invert IV ourselves from Polygon's audited daily option closes using
`options_pricer.implied_vol`, against each monthly 3rd Friday and the
$5-rounded ATM strike (reliable strike grid on both SPY and NVDA).

~5 HTTP requests per (underlying, day). Idempotent: incremental refresh
fetches only missing days; re-runs never re-pull the deep backfill.

Endpoint: ``/v2/aggs/ticker/O:{TICKER}{YYMMDD}P{STRIKE*1000:08d}/range/1/day/{d}/{d}``

Run:
  py parsers/fetch_atm_iv_history.py             # dry-run
  py parsers/fetch_atm_iv_history.py --write     # write CSV
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_base, get_massive_key  # noqa: E402
from fetch_option_history import build_option_ticker  # noqa: E402
from iv_constant_maturity import (  # noqa: E402
    CM_CSV_COLS, derive_cm_history,
)
from options_pricer import implied_vol  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
# Raw term-structure history (expensive, permanent on disk, incremental).
TERM_CSV = DATA / "atm_iv_term_history.csv"
# Derived constant-maturity history (cheap full re-derive each run; this is
# the file the dashboard's IV-percentile gauge consumes).
OUT_CSV = DATA / "atm_iv_history.csv"

TERM_CSV_COLS = ["date", "underlying", "expiry", "dte_days",
                 "atm_strike", "spot", "close", "atm_iv", "fetched_at"]

# Maturities fetched per (day, underlying). 5 listed monthlies always
# straddle the 90-day target — 4 can fall short just before a February
# expiry (short-month spacing). See next_n_monthly_expiries.
N_EXPIRIES = 5
# Constant-maturity horizon the gauge measures. Tunable; the raw term file
# stores enough structure to re-derive 30d / 60d later without re-fetching.
TARGET_DAYS = 90
# Concurrent option-close fetches. Each call is independent I/O (~0.5s incl.
# Norton TLS re-sign), so a thread pool turns a ~2h sequential backfill into
# ~10-15 min. Kept modest so we stay well under Polygon's rate ceiling; the
# _get_json retry layer absorbs the occasional 429.
MAX_WORKERS = 8

LOOKBACK_DAYS = 252
# Deepest start the Options Advanced tier reliably serves (SPY/NVDA spot in
# daily_prices begins 2016-05; option closes resolve from ~2016). Used by
# --max-depth for the one-time deep grab.
MAX_DEPTH_START = date(2016, 1, 1)


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    # weekday(): Mon=0 .. Fri=4 .. Sun=6
    offset_to_first_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=offset_to_first_friday + 14)


def _is_3rdfriday_holiday(d: date) -> bool:
    """True if d is a US market holiday that can land on a 3rd Friday.

    Only Good Friday (Easter - 2) and Juneteenth (Jun 19 since 2021)
    can ever land on a 3rd Friday — other federal holidays fall on
    Mondays / Thursdays / days outside the 15–21 range.
    """
    if d == easter(d.year) - timedelta(days=2):  # Good Friday
        return True
    if d.year >= 2021 and d.month == 6 and d.day == 19:  # Juneteenth
        return True
    return False


def listed_monthly_expiry(year: int, month: int) -> date:
    """Standard listed monthly options expiry for a given month.

    Nominally the 3rd Friday; rolls to Thursday when the 3rd Friday is
    a US market holiday (Good Friday or Juneteenth).
    """
    friday = _third_friday(year, month)
    if _is_3rdfriday_holiday(friday):
        return friday - timedelta(days=1)
    return friday


def front_month_expiry(as_of: date) -> date:
    """First standard monthly options expiry strictly after as_of.

    Holiday-aware: when the 3rd Friday is Good Friday or Juneteenth,
    the listed expiry is Thursday. Strictly-after semantics: on a
    listed-expiry day we already roll to next month, to avoid the
    DTE≈0 IV noise spike that distorts the front-month measurement.
    """
    candidate = listed_monthly_expiry(as_of.year, as_of.month)
    if candidate > as_of:
        return candidate
    next_month = as_of.month + 1
    next_year = as_of.year
    if next_month > 12:
        next_month = 1
        next_year += 1
    return listed_monthly_expiry(next_year, next_month)


def next_n_monthly_expiries(as_of: date, n: int) -> list[date]:
    """The next `n` listed monthly expiries strictly after `as_of`, ascending.

    Generalizes `front_month_expiry` (which is exactly `[0]` of this list) for
    the constant-maturity derive step, which needs several expiries per day to
    bracket the target horizon. Holiday-aware throughout — each element comes
    from `listed_monthly_expiry`, so Good-Friday / Juneteenth Thursday shifts
    are preserved at every step.

    n=5 is the practical floor for a 90-day target: 4 monthlies can fall short
    of 90 DTE just before a February expiry (short-month spacing), which would
    force a lower-quality one-sided reading exactly at the target.
    """
    expiries: list[date] = []
    cursor = as_of
    for _ in range(n):
        nxt = front_month_expiry(cursor)
        expiries.append(nxt)
        cursor = nxt  # strictly-after semantics roll us past nxt next loop
    return expiries


def atm_strike(spot: float) -> float:
    """Round spot to nearest $5 — listed strike on SPY and NVDA in the
    near-money range. Midpoint (X.5×5) rounds up, picking the
    higher strike (conservative for put-side ATM measurement).
    """
    return float(math.floor(spot / 5.0 + 0.5) * 5)


def invert_atm_iv(*, close: float, spot: float, strike: float,
                  dte_days: int, r: float, q: float) -> float:
    """Invert IV from an option close, returning NaN on bad input.

    Put-side convention: the sleeve is all puts, and put-call parity
    makes put / call ATM IV match to within quoting noise.
    """
    if dte_days <= 0 or close <= 0.0:
        return float("nan")
    return implied_vol(
        market_price=close, spot=spot, strike=strike,
        T=dte_days / 365.0, r=r, q=q, opt="put",
    )


class FetchError(RuntimeError):
    """A network/HTTP failure that must surface, NOT be swallowed as
    no-data. Conflating the two silently truncated the backfill (SPY
    skipped entirely, NVDA's last year dropped). Callers let this
    propagate so a partial run fails loud instead of writing junk."""


# HTTP statuses worth retrying — transient server/throttle conditions.
# Everything else (401/403 entitlement, 404 bad ticker) fails immediately:
# retrying can't fix it, and masking it is what hid the tier wall.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _get_json(url: str, params: dict, *, get=requests.get, sleep=time.sleep,
              max_retries: int = 5, base_delay: float = 2.0,
              timeout: int = 30) -> dict:
    """GET `url` and return parsed JSON, retrying transient failures with
    exponential backoff. Raises `FetchError` on a non-transient status or
    after exhausting retries.

    `get` and `sleep` are injectable so the retry logic is unit-testable
    without real HTTP or wall-clock delay. Honors a `Retry-After` header
    when the server sends one.
    """
    attempt = 0
    while True:
        try:
            resp = get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            # Network-level failure (connection reset, timeout) — transient.
            if attempt >= max_retries:
                raise FetchError(
                    f"GET failed after {attempt} retries: {exc}") from exc
            sleep(base_delay * (2 ** attempt))
            attempt += 1
            continue

        if resp.status_code == 200:
            return resp.json() or {}

        if resp.status_code not in _RETRYABLE_STATUS:
            # Non-transient (401/403 entitlement, 404 bad ticker, …): no
            # amount of retrying helps. Fail LOUD — this is the wall that
            # previously got swallowed as "no data".
            raise FetchError(
                f"HTTP {resp.status_code} for {url} (non-retryable)")

        if attempt >= max_retries:
            raise FetchError(
                f"HTTP {resp.status_code} for {url} after "
                f"{attempt} retries (gave up)")

        # Transient: honor Retry-After if present, else exponential backoff.
        retry_after = resp.headers.get("Retry-After")
        delay = (float(retry_after) if retry_after
                 else base_delay * (2 ** attempt))
        sleep(delay)
        attempt += 1


def fetch_option_close(
    ticker: str, day: date, key: str, base: str, *,
    get=requests.get, sleep=time.sleep,
) -> float | None:
    """One-day daily bar for an option contract. Returns the close, or
    None ONLY when the contract genuinely didn't trade that day (HTTP 200,
    empty results). Raises `FetchError` on any HTTP/network failure — the
    caller must not confuse a fetch failure with a real data gap.
    """
    url = (f"{base}/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{day.isoformat()}/{day.isoformat()}")
    params = {"apiKey": key, "adjusted": "true", "limit": 1}
    payload = _get_json(url, params, get=get, sleep=sleep)
    results = payload.get("results") or []
    if not results:
        return None
    c = results[0].get("c")
    return float(c) if c is not None else None


def load_spot_history(underlyings: list[str]) -> pd.DataFrame:
    """Pull spot history for the requested underlyings from
    data/daily_prices.csv. Returns long-form [date, symbol, close].

    NOTE: daily_prices.csv is split-ADJUSTED. Safe for SPY (never split)
    but wrong for building historical option tickers on names that have
    split (e.g. NVDA) — use `fetch_unadjusted_spot_range` for the deep
    backfill so strikes match the contracts actually listed at the time.
    """
    prices = pd.read_csv(DATA / "daily_prices.csv", parse_dates=["date"])
    mask = prices["symbol"].isin(underlyings)
    return prices.loc[mask, ["date", "symbol", "close"]].copy()


def fetch_unadjusted_spot_range(
    underlying: str, start: date, end: date, key: str, base: str, *,
    get=requests.get, sleep=time.sleep,
) -> pd.DataFrame:
    """Fetch UNADJUSTED daily closes for one underlying over [start, end]
    in a single Polygon stock-aggs call. Returns long-form
    [date, symbol, close] on trading days only.

    Unadjusted is the whole point: a historical option contract was
    listed against the unadjusted price, so the ATM strike must be
    derived from unadjusted spot or the ticker won't resolve across a
    split boundary.

    Raises `FetchError` on HTTP/network failure. A 200 with no bars is
    genuine no-data → empty frame. (Previously an HTTP error returned an
    empty frame too, which silently skipped the entire underlying — that
    is how SPY vanished from the deep backfill.)
    """
    url = (f"{base}/v2/aggs/ticker/{underlying}/range/1/day/"
           f"{start.isoformat()}/{end.isoformat()}")
    params = {"apiKey": key, "adjusted": "false", "limit": 50000}
    payload = _get_json(url, params, get=get, sleep=sleep, timeout=60)
    results = payload.get("results") or []
    rows = []
    for bar in results:
        ts_ms = bar.get("t")
        c = bar.get("c")
        if ts_ms is None or c is None:
            continue
        d = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).date()
        rows.append({"date": pd.Timestamp(d), "symbol": underlying,
                     "close": float(c)})
    return pd.DataFrame(rows, columns=["date", "symbol", "close"])


class RegressionError(RuntimeError):
    """The freshly-derived dataset would lose data already on disk (an
    underlying disappeared, or a per-underlying row count shrank). Raised
    before overwrite so a partial/failed run can never clobber a complete
    history."""


def assert_not_regressive(existing: pd.DataFrame,
                          new: pd.DataFrame) -> None:
    """Raise `RegressionError` if `new` drops an underlying present in
    `existing` or has fewer rows for any shared underlying. The last line
    of defense behind the fail-loud fetch layer: even if a fetch failure
    slipped through, we refuse to shrink the on-disk gauge data.

    An empty/absent `existing` (first write) always passes.
    """
    if existing is None or existing.empty:
        return
    old_counts = existing.groupby("underlying").size().to_dict()
    new_counts = (new.groupby("underlying").size().to_dict()
                  if new is not None and not new.empty else {})
    problems = []
    for u, n_old in old_counts.items():
        n_new = new_counts.get(u, 0)
        if n_new == 0:
            problems.append(f"{u}: present on disk ({n_old} rows), absent in new")
        elif n_new < n_old:
            problems.append(f"{u}: {n_old} rows on disk → {n_new} in new (shrank)")
    if problems:
        raise RegressionError(
            "Refusing to overwrite — new dataset regresses:\n  "
            + "\n  ".join(problems))


def merge_history(existing: pd.DataFrame,
                  new: pd.DataFrame) -> pd.DataFrame:
    """Union two term-IV-history frames, deduped on (date, underlying,
    expiry) with the NEW row winning. The term schema carries several
    expiry rows per (date, underlying), so `expiry` is part of the key —
    distinct expiries on the same day must all survive. Lets a re-run
    append only fresh days without clobbering the deep backfill already on
    disk. Empty `existing` returns `new` verbatim."""
    if existing is None or existing.empty:
        return new.copy()
    if new is None or new.empty:
        return existing.copy()
    combined = pd.concat([existing, new], ignore_index=True)
    combined["date"] = combined["date"].astype(str)
    combined = combined.drop_duplicates(
        subset=["date", "underlying", "expiry"], keep="last",
    ).reset_index(drop=True)
    return combined.sort_values(
        ["underlying", "date", "expiry"]
    ).reset_index(drop=True)


def missing_days(existing: pd.DataFrame, candidates: list[date],
                 underlying: str) -> list[date]:
    """Candidate days not already present for `underlying` in `existing`.
    Drives incremental fetching so we never re-pull a day we have."""
    if existing is None or existing.empty or "underlying" not in existing.columns:
        return list(candidates)
    have = existing[existing["underlying"] == underlying]
    if have.empty:
        return list(candidates)
    present = set(pd.to_datetime(have["date"]).dt.date)
    return [d for d in candidates if d not in present]


def load_risk_free(default: float = 0.04) -> pd.Series:
    """Pull risk-free rate history. Returns a date-indexed Series.
    Falls back to the default constant if the file is missing."""
    path = DATA / "risk_free_rate.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    rfr = pd.read_csv(path, parse_dates=["date"])
    rate_col = "rate" if "rate" in rfr.columns else rfr.columns[-1]
    s = rfr.set_index("date")[rate_col].astype(float)
    if s.max() > 1.0:
        s = s / 100.0  # CSV may be in %, normalize to fraction
    return s


@dataclass(frozen=True)
class WorkItem:
    """One option-close fetch to perform: the ATM put for `underlying` on
    `day` at `expiry`/`strike`, plus the spot and rate needed to invert IV.
    Produced purely by `plan_work_items`; consumed by the parallel fetcher."""
    underlying: str
    day: date
    expiry: date
    strike: float
    spot: float
    r: float
    q: float


def plan_work_items(
    spot_by_underlying: dict[str, pd.DataFrame],
    *, existing: pd.DataFrame | None, rfr: pd.Series,
    n_expiries: int = N_EXPIRIES, q: float = 0.015,
) -> list[WorkItem]:
    """Build the flat fetch work-list (pure — no I/O).

    For each underlying's spot frame, skip days already cached (incremental
    refresh, see `missing_days`) and emit one `WorkItem` per (day, expiry)
    across the next `n_expiries` monthlies. Splitting this out keeps the
    expensive concurrency layer thin and the day/expiry/strike logic
    unit-testable without touching the network.
    """
    items: list[WorkItem] = []
    for underlying, u_spot in spot_by_underlying.items():
        if u_spot is None or u_spot.empty:
            continue
        u_spot = u_spot.sort_values("date")
        candidates = [row["date"].date() for _, row in u_spot.iterrows()]
        want = set(missing_days(existing, candidates, underlying))
        for _, row in u_spot.iterrows():
            d = row["date"].date()
            if d not in want:
                continue
            s = float(row["close"])
            k = atm_strike(s)
            r = (float(rfr.get(pd.Timestamp(d), 0.04))
                 if rfr is not None and not rfr.empty else 0.04)
            for expiry in next_n_monthly_expiries(d, n_expiries):
                items.append(WorkItem(
                    underlying=underlying, day=d, expiry=expiry,
                    strike=k, spot=s, r=r, q=q,
                ))
    return items


def _fetch_one_work_item(item: WorkItem, key: str, base: str, *,
                         get=requests.get, sleep=time.sleep,
                         fetched_at: str = "") -> dict | None:
    """Fetch + invert a single WorkItem. Returns a term row dict, or None
    when the contract genuinely didn't trade / IV won't invert. Raises
    `FetchError` on HTTP failure (propagated by the executor to abort the
    run) — the no-data vs failure distinction from #117 is preserved."""
    ticker = build_option_ticker(item.underlying, "put", item.expiry,
                                 item.strike)
    close = fetch_option_close(ticker, item.day, key, base,
                               get=get, sleep=sleep)
    if close is None:
        return None
    dte = (item.expiry - item.day).days
    iv = invert_atm_iv(close=close, spot=item.spot, strike=item.strike,
                       dte_days=dte, r=item.r, q=item.q)
    if not math.isfinite(iv):
        return None
    return {
        "date":       item.day.isoformat(),
        "underlying": item.underlying,
        "expiry":     item.expiry.isoformat(),
        "dte_days":   dte,
        "atm_strike": item.strike,
        "spot":       item.spot,
        "close":      close,
        "atm_iv":     iv,
        "fetched_at": fetched_at,
    }


def _fetch_work_items_parallel(
    items: list[WorkItem], key: str, base: str, *,
    max_workers: int = MAX_WORKERS, get=requests.get, sleep=time.sleep,
    fetched_at: str = "", progress=None,
) -> list[dict]:
    """Fetch every WorkItem concurrently and return the term rows.

    Fail-loud is preserved under concurrency: a `FetchError` from ANY worker
    propagates out of `.result()` and aborts the whole run (remaining
    futures are cancelled) — so a transient wall can never silently produce
    a partial dataset, exactly the #117 failure. None results (genuine
    no-data) are simply dropped. `progress(done, total)` is called after
    each completion for live status. Order-independent.
    """
    rows: list[dict] = []
    if not items:
        return rows
    total = len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_fetch_one_work_item, it, key, base,
                        get=get, sleep=sleep, fetched_at=fetched_at)
            for it in items
        ]
        try:
            for fut in as_completed(futures):
                row = fut.result()  # re-raises FetchError → aborts
                done += 1
                if row is not None:
                    rows.append(row)
                if progress is not None:
                    progress(done, total)
        except FetchError:
            for f in futures:
                f.cancel()
            raise
    return rows


def _print_progress(done: int, total: int) -> None:
    """Carriage-return progress line for the live fetch (every ~2%)."""
    step = max(1, total // 50)
    if done % step == 0 or done == total:
        pct = 100.0 * done / total
        print(f"\r  fetched {done}/{total} ({pct:.0f}%)",
              end="" if done < total else "\n", flush=True)


def build_history(
    underlyings: list[str], key: str, base: str,
    *, start_date: date, end_date: date | None = None,
    existing: pd.DataFrame | None = None,
    q: float = 0.015, use_unadjusted: bool = True,
    max_workers: int = MAX_WORKERS, get=requests.get, sleep=time.sleep,
) -> pd.DataFrame:
    """Fetch newly-needed ATM IV term rows over [start_date, end_date].
    Returns only NEW rows — the caller merges with `existing`.

    Three steps: (1) pull each underlying's unadjusted spot range, (2) plan
    the flat work-list skipping cached days (`plan_work_items`, pure), (3)
    fetch the option closes concurrently (`_fetch_work_items_parallel`,
    fail-loud). `use_unadjusted=True` derives strikes from unadjusted spot
    so historical tickers resolve across split boundaries (NVDA 4:1 2021,
    10:1 2024); SPY never split.
    """
    end = end_date or date.today()
    rfr = load_risk_free()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Fetching ATM IV history: {underlyings} × "
          f"{start_date.isoformat()}..{end.isoformat()} "
          f"(unadjusted={use_unadjusted}, workers={max_workers})")

    spot_by_underlying: dict[str, pd.DataFrame] = {}
    for underlying in underlyings:
        if use_unadjusted:
            u_spot = fetch_unadjusted_spot_range(
                underlying, start_date, end, key, base, get=get, sleep=sleep,
            )
        else:
            allspot = load_spot_history([underlying])
            u_spot = allspot[
                (allspot["date"].dt.date >= start_date)
                & (allspot["date"].dt.date <= end)
            ]
        spot_by_underlying[underlying] = u_spot

    items = plan_work_items(spot_by_underlying, existing=existing, rfr=rfr,
                            n_expiries=N_EXPIRIES, q=q)
    for underlying, u_spot in spot_by_underlying.items():
        n_days = len({it.day for it in items if it.underlying == underlying})
        print(f"  {underlying}: {len(u_spot)} trading days available, "
              f"{n_days} to fetch (rest already cached)")
    print(f"  {len(items)} option-close fetches across {max_workers} workers")

    rows = _fetch_work_items_parallel(
        items, key, base, max_workers=max_workers, get=get, sleep=sleep,
        fetched_at=fetched_at, progress=_print_progress,
    )
    return pd.DataFrame(rows, columns=TERM_CSV_COLS)


def discover_sleeve_underlyings() -> list[str]:
    """Read data/option_position_snapshot.csv to find the underlyings
    we currently hedge with. Empty list if the snapshot is missing."""
    path = DATA / "option_position_snapshot.csv"
    if not path.exists():
        return []
    snap = pd.read_csv(path)
    return sorted(snap["underlying"].dropna().unique().tolist())


def _load_existing() -> pd.DataFrame:
    """Load the on-disk raw term history if present, else an empty frame.
    This is the expensive artifact incremental refresh protects; the
    derived CM file is rebuilt from it each run."""
    if TERM_CSV.exists():
        try:
            return pd.read_csv(TERM_CSV)
        except Exception:  # noqa: BLE001
            return pd.DataFrame(columns=TERM_CSV_COLS)
    return pd.DataFrame(columns=TERM_CSV_COLS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write data/atm_iv_history.csv (default: dry-run).")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                    help=f"Trailing-window start = today - N days "
                         f"(default: {LOOKBACK_DAYS}). Ignored if "
                         f"--start-date or --max-depth is given.")
    ap.add_argument("--start-date", type=str, default=None,
                    help="Explicit backfill start YYYY-MM-DD. Overrides "
                         "--lookback-days.")
    ap.add_argument("--max-depth", action="store_true",
                    help=f"Backfill from {MAX_DEPTH_START.isoformat()} — the "
                         "deepest history the Options Advanced tier serves. "
                         "Grab this while the tier is active.")
    ap.add_argument("--no-incremental", action="store_true",
                    help="Re-fetch every day in range, ignoring cached rows "
                         "(still merges, fresh rows win).")
    ap.add_argument("--adjusted", action="store_true",
                    help="Use split-ADJUSTED spot from daily_prices.csv "
                         "instead of unadjusted Polygon bars. Wrong for "
                         "split names; for debugging only.")
    ap.add_argument("--underlying", action="append", default=None,
                    help="Override sleeve discovery. Repeat for multi.")
    ap.add_argument("--target-days", type=int, default=TARGET_DAYS,
                    help=f"Constant-maturity horizon for the derived gauge "
                         f"series (default: {TARGET_DAYS}). Re-derives from "
                         f"the cached term file — no re-fetch needed.")
    ap.add_argument("--force", action="store_true",
                    help="Bypass the regression guard and overwrite even if "
                         "the new dataset has fewer rows / loses an "
                         "underlying. Use only for an intentional shrink.")
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                    help=f"Concurrent option-close fetches (default "
                         f"{MAX_WORKERS}). Higher is faster but risks "
                         f"Polygon rate limits; the retry layer absorbs "
                         f"occasional 429s. Use 1 for fully sequential.")
    args = ap.parse_args(argv)

    underlyings = args.underlying or discover_sleeve_underlyings()
    if not underlyings:
        print("[!] No sleeve underlyings. Pass --underlying SYM.")
        return 1

    # Start-date precedence: --start-date > --max-depth > --lookback-days.
    if args.start_date:
        start_date = date.fromisoformat(args.start_date)
    elif args.max_depth:
        start_date = MAX_DEPTH_START
    else:
        start_date = date.today() - timedelta(days=args.lookback_days)

    existing = pd.DataFrame(columns=TERM_CSV_COLS)
    if not args.no_incremental:
        existing = _load_existing()
        if not existing.empty:
            print(f"Existing term history: {len(existing)} rows "
                  f"(incremental — only missing days will be fetched).")

    key = get_massive_key()
    base = get_massive_base()
    new = build_history(
        underlyings, key, base,
        start_date=start_date, existing=existing,
        use_unadjusted=not args.adjusted, max_workers=args.max_workers,
    )
    merged = merge_history(existing, new)
    if merged.empty:
        print("[!] No term IV history (existing empty and none fetched).")
        return 1

    # Derive the constant-maturity gauge series from the full term frame.
    cm = derive_cm_history(merged, target_days=args.target_days)
    print(f"\n+{len(new)} new term row(s); {len(merged)} term rows total → "
          f"{len(cm)} CM-{args.target_days} rows across "
          f"{merged['underlying'].nunique()} underlying(s).")
    print(cm.groupby("underlying")["atm_iv"].agg(["count", "min", "max"]))
    qcounts = cm["quality"].value_counts().to_dict()
    print(f"CM quality: {qcounts}")

    if args.write:
        DATA.mkdir(parents=True, exist_ok=True)
        # Write the RAW term file first, unconditionally: it only ever grows
        # (merge_history unions), so it can't regress, and it is the
        # expensive artifact that makes a re-run resumable. Never gate it
        # behind the CM guard — a guard trip must not discard fetched data.
        merged.to_csv(TERM_CSV, index=False)
        print(f"[ok] wrote {TERM_CSV.relative_to(ROOT)} ({len(merged)} rows)")

        # Guard only the DERIVED CM file: never let a partial/failed run
        # shrink the on-disk gauge series. Fetch now fails loud on HTTP
        # errors; this catches any regression that still slips past. The
        # term file above is already safe, so --force here re-derives
        # instantly with no re-fetch.
        if not args.force and OUT_CSV.exists():
            try:
                prior_cm = pd.read_csv(OUT_CSV)
            except Exception:  # noqa: BLE001
                prior_cm = pd.DataFrame(columns=CM_CSV_COLS)
            # Only compare against a prior CM series (same schema). A legacy
            # front-month file has no `quality` column — skip the guard for
            # that one-time schema transition rather than false-trip on
            # apples-to-oranges row counts.
            if "quality" in prior_cm.columns:
                try:
                    assert_not_regressive(prior_cm, cm)
                except RegressionError as exc:
                    print(f"[!] {exc}")
                    print("[!] Term file saved; refusing to overwrite the CM "
                          "gauge. Re-run to completion, or pass --force for "
                          "an intentional shrink.")
                    return 1
        cm.to_csv(OUT_CSV, index=False)
        print(f"[ok] wrote {OUT_CSV.relative_to(ROOT)} ({len(cm)} rows)")
    else:
        print("[dry-run] add --write to persist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
