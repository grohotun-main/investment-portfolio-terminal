"""Holdings reconciliation guard — pure core.

Compares each account's EXTRACTED position value (summed market_value, from the
holdings parsers) against the statement's REPORTED account total, in two bands,
so a silently-wrong extraction is caught at ingest while legitimate residuals
stay quiet.

Why band-based and not exact-match
----------------------------------
Extracted Σ(market_value) does NOT equal the reported "Total Account Value" even
when extraction is perfect — the reported total includes accrued income and
occasionally unpriced positions (e.g. the JPM May -$2,602 residual = accrued
income + an unpriced call). On top of that, the Parametric TLH account (300+
direct-index lots) carries genuine ~0.2-0.5% lot-rounding noise. The +$234K JPM
May phantom, by contrast, was +15%. So:

  * WATCH  |diff%| > 0.30%                         — surfaced, not blocking
  * ERROR  |diff%| > 2% AND |diff$| > $10,000       — the bug net (blocks at ingest)
  * a per-account allowlist tolerance reclassifies known noise as "known" until
    it grows past the tolerance, at which point the normal bands resume (so a
    creeping drift re-fires instead of being permanently muted).

This module is pure (no I/O); the ingest gate and the standalone trace CLI build
on it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Band thresholds (defaults; tunable by callers / CLI).
WATCH_PCT = 0.30      # |diff%| above this is at least WATCH
ERROR_PCT = 2.0       # ERROR needs |diff%| above this ...
ERROR_USD = 10_000.0  # ... AND |diff$| above this (the $-floor keeps small
                      #     accounts' percentage noise out of the block band)

# Reconciliation key is (broker, account_id, month).
Key = tuple


class ReconRow(NamedTuple):
    broker: str
    account_id: str
    month: str
    extracted: float
    reported: float
    diff_usd: float
    diff_pct: float
    band: str          # "ok" | "known" | "watch" | "error"


def classify(extracted: float, reported: float, *,
             watch_pct: float = WATCH_PCT,
             error_pct: float = ERROR_PCT,
             error_usd: float = ERROR_USD,
             tol_pct: Optional[float] = None) -> str:
    """The band for one account's (extracted vs reported) totals.

    `tol_pct` is the account's allowlisted known-noise tolerance; a drift within
    it is "known" (info), above it the normal bands resume. A reported total of
    0 can't be reconciled by ratio, so it is "error" when anything was extracted
    and "ok" when nothing was (no ZeroDivision either way).
    """
    diff = extracted - reported
    if reported <= 0:
        return "error" if abs(diff) > 0 else "ok"

    abs_pct = abs(100.0 * diff / reported)
    abs_usd = abs(diff)

    if tol_pct is not None and abs_pct <= tol_pct:
        return "known"
    if abs_pct > error_pct and abs_usd > error_usd:
        return "error"
    if abs_pct > watch_pct:
        return "watch"
    return "ok"


def reconcile(extracted_by_key: dict, reported_by_key: dict,
              allowlist: dict) -> list[ReconRow]:
    """One ReconRow per account the statement REPORTS a total for.

    Driven by the reported totals (statement ground truth): an account reported
    but not extracted defaults to extracted=0 and so surfaces as a large drift,
    rather than being silently omitted. `allowlist[account_id]["max_pct"]`
    supplies the per-account tolerance.
    """
    rows: list[ReconRow] = []
    for key, reported in reported_by_key.items():
        broker, account_id, month = key
        extracted = extracted_by_key.get(key, 0.0)

        entry = allowlist.get(account_id)
        tol = entry.get("max_pct") if entry else None

        band = classify(extracted, reported, tol_pct=tol)
        diff_usd = extracted - reported
        diff_pct = (100.0 * diff_usd / reported) if reported else float("nan")
        rows.append(ReconRow(broker, account_id, month,
                             extracted, reported, diff_usd, diff_pct, band))
    return rows


def format_table(rows: list[ReconRow]) -> str:
    """Aligned per-account reconciliation table — printed on every ingest; it
    is also the standalone drift-trace output."""
    if not rows:
        return "(no accounts to reconcile)"

    header = (f"{'broker':<9} {'account':<12} {'month':<8} "
              f"{'extracted':>16} {'reported':>16} {'diff $':>14} "
              f"{'diff %':>9}  band")
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda x: (x.broker, x.account_id, x.month)):
        lines.append(
            f"{r.broker:<9} {r.account_id:<12} {r.month:<8} "
            f"{r.extracted:>16,.2f} {r.reported:>16,.2f} {r.diff_usd:>+14,.2f} "
            f"{r.diff_pct:>+8.2f}%  {r.band}"
        )
    return "\n".join(lines)


class LaggingRow(NamedTuple):
    broker: str
    account_id: str
    last_month: str       # the account's newest statement month, "YYYY-MM"
    broker_latest: str    # its broker's newest statement month after this ingest


def lagging_accounts(latest_month_by_account: dict,
                     suppress: frozenset = frozenset()) -> list[LaggingRow]:
    """Accounts whose newest statement month trails their OWN broker's newest.

    `latest_month_by_account` maps (broker, account_id) -> "YYYY-MM": the newest
    month each account has a real statement for, across existing positions and
    this ingest's freshly-parsed rows. Per broker, the frontier is the max month
    over its accounts; an account lags when its month is strictly behind that
    frontier — its broker advanced to a newer statement and this account did
    not (no statement downloaded for it).

    Advisory: a carried-forward laggard keeps the value correct; this just
    surfaces it at ingest so the missing statement gets fetched. Brokers are
    scored independently (one broker's frontier never flags another's accounts;
    a broker that hasn't advanced this run is never a laggard). `suppress` (a
    set of account_ids) silences genuinely-closed accounts.
    """
    by_broker: dict = {}
    for (broker, acct), month in latest_month_by_account.items():
        by_broker.setdefault(broker, {})[acct] = month

    rows: list[LaggingRow] = []
    for broker, accts in by_broker.items():
        frontier = max(accts.values())
        for acct, month in accts.items():
            if month < frontier and acct not in suppress:
                rows.append(LaggingRow(broker, acct, month, frontier))
    return rows


def format_lagging(rows: list[LaggingRow]) -> str:
    """The completeness line(s) printed right after the reconciliation table on
    every ingest (advisory — never blocks)."""
    if not rows:
        return ("Holdings completeness — all accounts current to their "
                "broker's latest statement.")
    lines = ["Holdings completeness — accounts behind their broker's "
             "latest statement:"]
    for r in sorted(rows, key=lambda x: (x.broker, x.account_id)):
        lines.append(f"  {r.broker:<9} {r.account_id:<12} last {r.last_month}  "
                     f"({r.broker} now at {r.broker_latest})")
    noun = "statement" if len(rows) == 1 else "statements"
    lines.append(f"  -> {len(rows)} carried forward; download the missing "
                 f"{noun} and re-run ingest to refresh.")
    return "\n".join(lines)


def load_allowlist() -> dict:
    """Per-account known-noise tolerances for the guard, from config_local.py
    (gitignored; schema in config_example.py). Optional: returns {} when
    config_local or the HOLDINGS_RECON_ALLOWLIST constant is absent, so every
    account falls back to the default bands."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import config_local as cfg
    except ImportError:
        return {}
    return dict(getattr(cfg, "HOLDINGS_RECON_ALLOWLIST", {}))


SUMMARIES_COLUMNS = ["statement_date", "broker", "account_id",
                     "reported_total", "source_file"]


def upsert_summaries(records, path) -> int:
    """Advance summaries.csv with reported-total records (each a dict with
    SUMMARIES_COLUMNS keys), making live a file that has been Phase-0-frozen.

    Upserts by (broker, account_id, month): a key present in `records` is
    refreshed and new keys are added, while every other (broker, account,
    month) already on file is preserved — so a corrected re-ingest replaces
    rather than duplicates. Returns the row count written.
    """
    import pandas as pd
    path = Path(path)
    new_df = pd.DataFrame(list(records), columns=SUMMARIES_COLUMNS)
    existing = pd.read_csv(path, dtype=str) if path.exists() else None

    def _key(df):
        month = df["statement_date"].astype(str).str.slice(0, 7)
        return (df["broker"].astype(str) + "|"
                + df["account_id"].astype(str) + "|" + month)

    if new_df.empty:
        out = existing if existing is not None else new_df
    elif existing is None or existing.empty:
        out = new_df
    else:
        superseded = set(_key(new_df))
        kept = existing[~_key(existing).isin(superseded)]
        out = pd.concat([kept, new_df], ignore_index=True)

    out.to_csv(path, index=False)
    return len(out)
