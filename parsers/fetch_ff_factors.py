"""
Fetch the Ken French research-factor series for the Factor Analysis tab:
market excess return, size, value, profitability, investment (the 5-factor
2x3 file) plus momentum, and the 1-month T-bill RF — both monthly and daily.

DESIGN: the French Data Library is the canonical academic factor source —
free, keyless, updated ~monthly with a few weeks' publication lag. Each
download is a zip holding ONE CSV laid out as: text preamble, a monthly
block (rows keyed YYYYMM), an annual block (rows keyed YYYY), a copyright
footer. (Daily files are the same layout minus the annual block; rows are
keyed YYYYMMDD.) Only the data rows are wanted; a line-level regex selects
them robustly without depending on section ordering. Missing-data sentinels
(-99.99 / -999) -> NaN (rows dropped at join). Percent -> decimal on parse.

The FF5 file's SMB (not the FF3 file's) is used for every model the
dashboard offers — the two constructions differ microscopically and using
one file keeps the fetch to two small zips per frequency.

Endpoints (keyless, public):
    F-F_Research_Data_5_Factors_2x3_CSV.zip         (Mkt-RF SMB HML RMW CMA RF, monthly)
    F-F_Momentum_Factor_CSV.zip                     (Mom, monthly)
    F-F_Research_Data_5_Factors_2x3_daily_CSV.zip   (Mkt-RF SMB HML RMW CMA RF, daily)
    F-F_Momentum_Factor_daily_CSV.zip               (Mom, daily)

Output:
    data/ff_factors_monthly.csv  — month (YYYY-MM), mkt_rf, smb, hml, rmw, cma, mom, rf
    data/ff_factors_daily.csv    — date (YYYY-MM-DD), mkt_rf, smb, hml, rmw, cma, mom, rf
    All values decimal.

Run:
  py parsers/fetch_ff_factors.py            # dry-run preview
  py parsers/fetch_ff_factors.py --write    # fetch + emit both CSVs
"""
import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

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
OUT_CSV = DATA / "ff_factors_monthly.csv"
OUT_CSV_DAILY = DATA / "ff_factors_daily.csv"

FRENCH_BASE = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp")
FF5_URL = f"{FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = f"{FRENCH_BASE}/F-F_Momentum_Factor_CSV.zip"
FF5_DAILY_URL = f"{FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_DAILY_URL = f"{FRENCH_BASE}/F-F_Momentum_Factor_daily_CSV.zip"
HEADERS = {"User-Agent": "Mozilla/5.0 portfolio_dashboard fetch_ff_factors.py"}

# Same bounded-retry rationale as fetch_risk_free_rate.py: the refresh
# orchestrator treats this source as non-fatal, so failing fast beats
# hanging the whole "Refresh all data" run.
MAX_ATTEMPTS = 2
BACKOFF_BASE = 1.5

_MONTHLY_ROW = re.compile(r"^\s*(\d{6})\s*,")
_DAILY_ROW = re.compile(r"^\s*(\d{8})\s*,")
_MISSING_SENTINELS = (-99.99, -999.0)

EXPECTED_COLUMNS = ["month", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
EXPECTED_COLUMNS_DAILY = ["date", "mkt_rf", "smb", "hml", "rmw", "cma",
                          "mom", "rf"]


def parse_french_csv(text: str, date_digits: int = 6) -> pd.DataFrame:
    """Parse one French-library CSV into a decimal frame.

    date_digits=6 (default): monthly file — keeps `^\\s*\\d{6}\\s*,` rows,
    key column `month` ('YYYY-MM'). date_digits=8: daily file — keeps
    `^\\s*\\d{8}\\s*,` rows, key column `date` ('YYYY-MM-DD'). The two
    patterns cannot cross-match (a YYYYMMDD row has no comma after 6
    digits; a YYYYMM row has no 8 digits), so a frequency mix-up raises
    the no-rows ValueError instead of mis-parsing. The column header is
    the nearest preceding non-blank line, normalized (strip, lower,
    '-' -> '_'). Percent values /100; -99.99/-999 sentinels -> NaN.
    Raises ValueError when no rows are found (layout change canary).
    """
    if date_digits == 6:
        row_re, key = _MONTHLY_ROW, "month"
    elif date_digits == 8:
        row_re, key = _DAILY_ROW, "date"
    else:
        raise ValueError(f"date_digits must be 6 or 8, got {date_digits}")
    header_cols: list[str] | None = None
    rows: list[list[str]] = []
    last_nonblank = ""
    for ln in text.splitlines():
        m = row_re.match(ln)
        if m:
            if header_cols is None:
                header_cols = [c.strip().lower().replace("-", "_")
                               for c in last_nonblank.split(",")][1:]
            ym = m.group(1)
            if date_digits == 6:
                keyval = f"{ym[:4]}-{ym[4:]}"
            else:
                keyval = f"{ym[:4]}-{ym[4:6]}-{ym[6:]}"
            parts = [p.strip() for p in ln.split(",")][1:]
            rows.append([keyval] + parts[:len(header_cols)])
        elif ln.strip():
            last_nonblank = ln
    if header_cols is None or not rows:
        raise ValueError(
            "no data rows found — French CSV layout changed?")
    df = pd.DataFrame(rows, columns=[key] + header_cols)
    for c in header_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        v = v.mask(v.isin(_MISSING_SENTINELS))
        df[c] = v / 100.0
    return df


def _get_with_retry(url: str, *, timeout: int = 30) -> requests.Response:
    """GET `url` with a bounded retry loop (connection-level failures only,
    MAX_ATTEMPTS total); an HTTP error status raises immediately. Calls
    requests.get / time.sleep at module scope so tests can patch them."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
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


def _download_csv_text(url: str, timeout: int) -> str:
    """Download a French-library zip and return its single CSV member as
    text. latin-1 decode: the files are ASCII-with-occasional-Latin-1 and
    must never crash on an odd byte in the preamble."""
    r = _get_with_retry(url, timeout=timeout)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    if len(names) != 1:
        raise ValueError(f"expected exactly 1 zip member at {url}, "
                         f"got {names}")
    return zf.read(names[0]).decode("latin-1")


def fetch_ff_factors(timeout: int = 30) -> pd.DataFrame:
    """Download + join the 5-factor and momentum monthly files.

    Returns DataFrame[month, mkt_rf, smb, hml, rmw, cma, mom, rf] (decimal),
    ascending by month, restricted to months where EVERY column is present
    (inner join + sentinel-NaN rows dropped). The 5-factor file starts
    1963-07 and momentum 1927-01, so the result effectively starts 1963-07 —
    decades more than any portfolio window needs.
    """
    ff5 = parse_french_csv(_download_csv_text(FF5_URL, timeout))
    mom = parse_french_csv(_download_csv_text(MOM_URL, timeout))
    need5 = {"mkt_rf", "smb", "hml", "rmw", "cma", "rf"}
    if not need5.issubset(ff5.columns) or "mom" not in mom.columns:
        raise ValueError(
            f"unexpected French columns: ff5={list(ff5.columns)} "
            f"mom={list(mom.columns)}")
    df = ff5.merge(mom[["month", "mom"]], on="month", how="inner")
    return (df[EXPECTED_COLUMNS].dropna()
              .sort_values("month").reset_index(drop=True))


def fetch_ff_factors_daily(timeout: int = 30) -> pd.DataFrame:
    """Download + join the 5-factor and momentum DAILY files.

    Returns DataFrame[date 'YYYY-MM-DD', mkt_rf, smb, hml, rmw, cma, mom,
    rf] (decimal), ascending, restricted to dates where every column is
    present. ~15k rows from 1963-07 (~1 MB on disk) — trivial, so no
    trimming; the regression aligns to the portfolio's own price window.
    """
    ff5 = parse_french_csv(_download_csv_text(FF5_DAILY_URL, timeout),
                           date_digits=8)
    mom = parse_french_csv(_download_csv_text(MOM_DAILY_URL, timeout),
                           date_digits=8)
    need5 = {"mkt_rf", "smb", "hml", "rmw", "cma", "rf"}
    if not need5.issubset(ff5.columns) or "mom" not in mom.columns:
        raise ValueError(
            f"unexpected French daily columns: ff5={list(ff5.columns)} "
            f"mom={list(mom.columns)}")
    df = ff5.merge(mom[["date", "mom"]], on="date", how="inner")
    return (df[EXPECTED_COLUMNS_DAILY].dropna()
              .sort_values("date").reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--write", action="store_true",
                    help="Fetch and emit CSV (default: dry-run summary).")
    args = ap.parse_args()

    print(f"Sources: {FF5_URL}\n         {MOM_URL}\n"
          f"         {FF5_DAILY_URL}\n         {MOM_DAILY_URL}")
    # Both frequencies share one exit code: the refresh orchestrator treats
    # this script as a single non-fatal unit, so a daily-endpoint outage also
    # skips the (atomic) monthly write. Acceptable — French data is slow-
    # moving and the prior CSVs are retained; revisit only if the daily file
    # gets its own freshness gate.
    try:
        df = fetch_ff_factors()
        df_daily = fetch_ff_factors_daily()
    except (requests.RequestException, ValueError, zipfile.BadZipFile) as e:
        print(f"[!] Fetch failed: {e}")
        return 1

    n, nd = len(df), len(df_daily)
    print(f"Monthly rows: {n:,} "
          f"({df['month'].iloc[0]} -> {df['month'].iloc[-1]})")
    print(f"Daily rows:   {nd:,} "
          f"({df_daily['date'].iloc[0]} -> {df_daily['date'].iloc[-1]})")
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print("Latest month:")
        print(df.tail(1).to_string(index=False))
        print("Latest day:")
        print(df_daily.tail(1).to_string(index=False))

    if not args.write:
        print()
        print(f"(dry-run — re-run with --write to emit {OUT_CSV} "
              f"and {OUT_CSV_DAILY})")
        return 0

    DATA.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    df_daily.to_csv(OUT_CSV_DAILY, index=False)
    print()
    print(f"Wrote {OUT_CSV} ({n:,} rows)")
    print(f"Wrote {OUT_CSV_DAILY} ({nd:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
