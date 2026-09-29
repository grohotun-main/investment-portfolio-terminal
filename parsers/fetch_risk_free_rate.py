"""
Fetch the FRED DGS3MO series (3-month constant-maturity Treasury yield) as
the time-varying risk-free rate driving Sharpe / Sortino numerators.

DESIGN: DGS3MO is the Federal Reserve's daily-published, annualized,
constant-maturity 3M Treasury yield — the canonical academic risk-free rate.
History reaches back to 1982-01-04, which covers every dated portfolio
return we ever compute. FRED missing-data is the literal string `.` on
holidays / when the auction was deferred; those rows are dropped on read
and the dashboard forward-fills inside load_risk_free_rate().

Endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO
Header: observation_date,DGS3MO  (YYYY-MM-DD dates, percent values)
No API key, free, public.

Output: data/risk_free_rate.csv with columns date, rate_annual.
`rate_annual` is the decimal annualized yield (0.0425 for 4.25%) — the
unit consumers in risk_metrics.py expect, no conversion at the call site.

Run:
  py parsers/fetch_risk_free_rate.py            # dry-run preview
  py parsers/fetch_risk_free_rate.py --write    # fetch + emit CSV
"""
import argparse
import json
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from _config import get_fred_api_key

# Routes Python's SSL trust through the Windows cert store so Norton's
# TLS-scanning re-signed certs validate. See parsers/_config.py.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_CSV = DATA / "risk_free_rate.csv"

FRED_DGS3MO_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
)
# FRED's official API host. Reachable on networks where the fredgraph host
# above read-times-out (Norton TLS interception); requires a free API key
# (get_fred_api_key). Returns the full DGS3MO observation history as JSON.
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
HEADERS = {"User-Agent": "Mozilla/5.0 portfolio_dashboard fetch_risk_free_rate.py"}

# FRED's fredgraph endpoint sporadically closes the TCP connection mid-handshake
# on this network (RemoteDisconnected) and, when the host / Norton TLS path is
# blocked, can hang until the read timeout. urllib3's Retry only kicks in on
# HTTP status codes, not connection-level failures, so we add a manual retry
# loop — but keep the budget SMALL (one retry, short read timeout) so a blocked
# FRED fails in tens of seconds, not minutes. The refresh orchestrator treats a
# FRED failure as non-fatal, so failing fast here beats hanging the whole
# "Refresh all data" run (it used to burn ~5 min: 5 attempts x a 60s timeout).
MAX_ATTEMPTS = 2
BACKOFF_BASE = 1.5


def _get_with_retry(url: str, *, params: dict | None = None,
                    timeout: int = 20) -> requests.Response:
    """GET `url` with a bounded retry loop, returning the Response. Retries
    only connection-level failures (ConnectionError / Timeout) up to
    MAX_ATTEMPTS, then raises the last one; an HTTP error status raises
    immediately (a bad key / 5xx isn't fixed by retrying). Calls requests.get /
    time.sleep at module scope so tests can patch them.
    """
    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_BASE ** attempt
                print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed "
                      f"({type(e).__name__}); retrying in {wait:.1f}s")
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _to_rate_frame(date_series, value_series) -> pd.DataFrame:
    """Shape raw date/value columns into [date, rate_annual] (decimal). FRED
    marks missing observations with a literal '.' -> NaN via to_numeric ->
    dropped (holidays / deferred auctions are forward-filled at consumption in
    load_risk_free_rate()). Shared by both parsers so the API and graph paths
    yield identical frames.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(date_series),
        "rate_annual": pd.to_numeric(value_series, errors="coerce") / 100.0,
    })
    return (df.dropna(subset=["rate_annual"])
              .sort_values("date").reset_index(drop=True))


def _parse_graph_csv(text: str) -> pd.DataFrame:
    """Parse the fredgraph CSV (observation_date,DGS3MO; older dumps DATE,DGS3MO)
    into [date, rate_annual]."""
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    return _to_rate_frame(df[df.columns[0]], df[df.columns[1]])


def _parse_api_json(text: str) -> pd.DataFrame:
    """Parse FRED API /series/observations JSON into [date, rate_annual]."""
    obs = json.loads(text).get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", "rate_annual"])
    df = pd.DataFrame(obs)
    return _to_rate_frame(df["date"], df["value"])


def fetch_dgs3mo(api_key: str = "", timeout: int = 20) -> pd.DataFrame:
    """Download + parse the FRED DGS3MO series -> DataFrame[date, rate_annual]
    (decimal — 0.0425 for 4.25% — FRED '.' rows dropped, ascending by date).

    When `api_key` is set, try FRED's API host (api.stlouisfed.org) FIRST — it
    stays reachable on networks where the keyless fredgraph host
    (fred.stlouisfed.org) read-times-out (Norton TLS interception) — then fall
    back to the graph CSV. With no key, use only the graph CSV (prior behavior).
    Raises requests.RequestException only if EVERY source fails; the per-source
    retry budget is bounded (see _get_with_retry) so a fully-blocked FRED still
    fails in tens of seconds, not minutes.
    """
    sources = []
    if api_key:
        sources.append((
            "FRED API (api.stlouisfed.org)", FRED_API_URL,
            {"series_id": "DGS3MO", "api_key": api_key, "file_type": "json"},
            _parse_api_json,
        ))
    sources.append((
        "FRED graph CSV (fred.stlouisfed.org)", FRED_DGS3MO_URL, None,
        _parse_graph_csv,
    ))

    last_err: Exception | None = None
    for label, url, params, parse in sources:
        try:
            return parse(_get_with_retry(url, params=params, timeout=timeout).text)
        except requests.RequestException as e:
            last_err = e
            print(f"  source unreachable: {label} ({type(e).__name__})")
    raise last_err  # type: ignore[misc]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Fetch and emit CSV (default: dry-run summary).")
    args = ap.parse_args()

    api_key = get_fred_api_key()
    if api_key:
        print(f"Source: FRED API ({FRED_API_URL}) — key set; "
              f"graph CSV fallback")
    else:
        print(f"Source: {FRED_DGS3MO_URL} "
              f"(set FRED_API_KEY to use the reachable API host)")
    try:
        df = fetch_dgs3mo(api_key=api_key)
    except requests.RequestException as e:
        print(f"[!] Fetch failed after trying all sources: {e}")
        return 1

    n = len(df)
    print(f"Rows: {n:,}")
    print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")
    last = df["rate_annual"].iloc[-1] * 100.0
    mean_12mo = df["rate_annual"].tail(252).mean() * 100.0
    mean_36mo = df["rate_annual"].tail(756).mean() * 100.0
    print(f"Most recent: {last:.2f}%")
    print(f"Trailing 12mo mean (252 biz days): {mean_12mo:.2f}%")
    print(f"Trailing 36mo mean (756 biz days): {mean_36mo:.2f}%")

    if not args.write:
        print()
        print(f"(dry-run — re-run with --write to emit {OUT_CSV})")
        return 0

    DATA.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")
    print()
    print(f"Wrote {OUT_CSV} ({n:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
