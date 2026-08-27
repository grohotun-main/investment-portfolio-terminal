"""Data-health adapter — verify this month's statement ingest.

Pure (no Streamlit, no I/O beyond the frames passed in). Surfaces the ingest
gate's own reconciliation (parsers/reconcile_holdings.py) in the dashboard so
the Data Health tab and the gate agree by construction.

Reads only: positions[statement_date, broker, account_id, market_value
(+ market_value_stmt, preferred when present — the pre-mark statement value
mark_to_market stashes, so a live re-mark can't read as drift)] and
summaries[statement_date, broker, account_id, reported_total]. `today` and
`allowlist` are passed in so this
stays pure. See docs/superpowers/specs/2026-06-05-data-health-ingest-panel-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

import pandas as pd

# parsers/ is on sys.path (inserted by app.py / the tests); reconcile_holdings
# is a sibling module — its `reconcile`/`lagging_accounts` ARE the ingest gate.
from reconcile_holdings import lagging_accounts, reconcile  # noqa: E402


@dataclass(frozen=True)
class AccountHealth:
    account_id: str
    label: str
    broker: str
    state: Literal["verified", "missing", "carried"]
    lagging: bool                     # broker advanced to M but this acct didn't
    band: Optional[str]               # ok|known|watch|error; None when no recon
    extracted: Optional[float]
    reported: Optional[float]
    diff_usd: Optional[float]
    diff_pct: Optional[float]
    last_verified_month: str          # "YYYY-MM"
    days_since: Optional[int]


@dataclass(frozen=True)
class HealthReport:
    as_of_month: str
    recon_available: bool
    accounts: list[AccountHealth]
    n_ok: int
    n_known: int
    n_watch: int
    n_error: int                      # includes "missing"
    n_carried: int
    worst_level: Literal["green", "amber", "red", "grey"]
    unreconciled_months: list[str] = field(default_factory=list)


def _pretty_month(ym: str) -> str:
    if not ym or ym == "—":
        return "—"
    try:
        return pd.to_datetime(ym + "-01").strftime("%b %Y")
    except (ValueError, TypeError):
        return str(ym)


def _priority(a: AccountHealth) -> int:
    """Sort key: error/missing first, then watch, then carried, then ok/known."""
    if a.state == "missing" or (a.state == "verified" and a.band == "error"):
        return 0
    if a.state == "verified" and a.band == "watch":
        return 1
    if a.state == "carried":
        return 2
    return 3


def build_health_report(positions: pd.DataFrame, summaries: pd.DataFrame,
                        *, today: date,
                        label_by_account: Optional[dict] = None,
                        allowlist: Optional[dict] = None,
                        suppress: frozenset = frozenset()) -> HealthReport:
    label_by_account = label_by_account or {}
    allowlist = allowlist or {}

    # No reported totals at all -> recon unavailable (grey; absence ≠ failure).
    if summaries is None or len(summaries) == 0:
        return HealthReport("", False, [], 0, 0, 0, 0, 0, "grey")

    s = summaries.copy()
    s["month"] = pd.to_datetime(s["statement_date"]).dt.strftime("%Y-%m")
    s["reported_total"] = pd.to_numeric(s["reported_total"], errors="coerce")
    as_of_month = s["month"].max()

    # Statements are ground truth for coverage/lag and which month we verify.
    latest_month_by_account: dict = {}
    last_stmt_date: dict = {}
    for (broker, acct), grp in s.groupby(["broker", "account_id"]):
        m = grp["month"].max()
        latest_month_by_account[(broker, acct)] = m
        last_stmt_date[(broker, acct)] = pd.to_datetime(
            grp.loc[grp["month"] == m, "statement_date"]).max()

    cur = s[s["month"] == as_of_month]
    reported_by_key = {(r.broker, r.account_id, as_of_month): float(r.reported_total)
                       for r in cur.itertuples()}

    # Extracted = Σ market_value over the account's positions in month M — on
    # the STATEMENT basis. Both UIs pass frames whose latest snapshot was
    # re-marked to live prices at load (parsers/mark_to_market.py, which
    # stashes the pre-mark values in `market_value_stmt`); prefer the stash so
    # a real market move since the statement date cannot read as
    # reconciliation drift. The ingest gate sums raw parsed rows, so this is
    # what keeps the tab agreeing with it.
    p = positions.copy()
    if "market_value_stmt" in p.columns:
        stmt = pd.to_numeric(p["market_value_stmt"], errors="coerce")
        p["market_value"] = stmt.fillna(p["market_value"])
    p["month"] = pd.to_datetime(p["statement_date"]).dt.strftime("%Y-%m")
    cur_pos = p[p["month"] == as_of_month]
    extracted_by_key: dict = {}
    for (broker, acct), grp in cur_pos.groupby(["broker", "account_id"]):
        extracted_by_key[(broker, acct, as_of_month)] = float(grp["market_value"].sum())

    recon_rows = reconcile(extracted_by_key, reported_by_key, allowlist)
    recon_by_acct = {(r.broker, r.account_id): r for r in recon_rows}

    lagging = {(r.broker, r.account_id)
               for r in lagging_accounts(latest_month_by_account, frozenset(suppress))}

    # Universe = accounts that actually have statements (from summaries). The
    # demo-broker overlay app.py injects into positions/accounts has no
    # statements, so it is excluded here rather than mis-flagged "carried".
    roster = set(latest_month_by_account.keys())

    out: list = []
    n_ok = n_known = n_watch = n_error = n_carried = 0
    for (broker, acct) in roster:
        label = label_by_account.get(acct, acct)
        stmt_dt = last_stmt_date.get((broker, acct))
        days = None
        if stmt_dt is not None and not pd.isna(stmt_dt):
            days = (today - stmt_dt.date()).days
        rr = recon_by_acct.get((broker, acct))
        if rr is not None:
            state = ("verified" if (broker, acct, as_of_month) in extracted_by_key
                     else "missing")
            out.append(AccountHealth(
                acct, label, broker, state, False, rr.band,
                rr.extracted, rr.reported, rr.diff_usd, rr.diff_pct,
                as_of_month, days))
            if state == "missing" or rr.band == "error":
                n_error += 1
            elif rr.band == "watch":
                n_watch += 1
            elif rr.band == "known":
                n_known += 1
            else:
                n_ok += 1
        else:
            lv = latest_month_by_account.get((broker, acct)) or "—"
            out.append(AccountHealth(
                acct, label, broker, "carried",
                (broker, acct) in lagging, None,
                None, None, None, None, lv, days))
            n_carried += 1

    out.sort(key=lambda a: (_priority(a), a.broker, a.account_id))

    # Holdings for a month newer than the reconciled month, restricted to real
    # accounts (those that appear in summaries) so the demo-broker overlay and
    # other non-statement rows can't trigger it. These are loaded but have no
    # reported totals yet -> unverified.
    real_keys = set(latest_month_by_account.keys())
    newer = p[p["month"] > as_of_month]
    unreconciled_months = sorted({
        m for (b, a, m) in zip(newer["broker"], newer["account_id"], newer["month"])
        if (b, a) in real_keys
    })

    if n_error:
        worst = "red"
    elif n_watch or n_carried or unreconciled_months:
        worst = "amber"
    else:
        worst = "green"

    return HealthReport(as_of_month, True, out, n_ok, n_known, n_watch,
                        n_error, n_carried, worst, unreconciled_months)


def format_health_headline(report: HealthReport) -> tuple[str, str]:
    """(level, text) for the strip. level in {green, amber, red, grey}."""
    if not report.recon_available:
        return ("grey", "Recon unavailable — no reported totals loaded.")
    if report.n_error:
        return ("red", f"✗ {report.n_error} account(s) off >2% / >$10k or "
                       f"missing — open the Data Health tab.")
    if report.unreconciled_months:
        months = ", ".join(_pretty_month(m) for m in report.unreconciled_months)
        return ("amber", f"⚠ {months} holdings loaded but not yet reconciled "
                         f"(no reported totals); reconciled through "
                         f"{_pretty_month(report.as_of_month)}.")
    if report.n_watch:
        return ("amber", f"⚠ {report.n_watch} account(s) within the watch band "
                         f"(>0.30%) — open the Data Health tab.")
    if report.n_carried:
        carried = [a for a in report.accounts if a.state == "carried"]
        who = carried[0]
        extra = f" (+{len(carried) - 1} more)" if len(carried) > 1 else ""
        lv = _pretty_month(who.last_verified_month)
        return ("amber", f"⚠ {report.n_carried} account(s) carried forward — "
                         f"{who.label} last verified {lv}{extra}.")
    return ("green", f"✓ All {len(report.accounts)} account(s) reconcile · "
                     f"current to {_pretty_month(report.as_of_month)}.")


def health_rows_to_table(report: HealthReport) -> list[dict]:
    """Render rows for the tab table (keeps the chip/'—' branching tested)."""
    rows = []
    for a in report.accounts:
        if a.state == "carried":
            verdict = "carried (lagging)" if a.lagging else "carried"
            rows.append({
                "Account": a.label, "Broker": a.broker,
                "State": "Carried forward",
                "Last verified": _pretty_month(a.last_verified_month),
                "Extracted": "—", "Reported": "—", "Δ$": "—", "Δ%": "—",
                "Verdict": verdict,
            })
        else:
            rows.append({
                "Account": a.label, "Broker": a.broker,
                "State": "Missing" if a.state == "missing" else "Verified",
                "Last verified": _pretty_month(a.last_verified_month),
                "Extracted": f"{a.extracted:,.2f}",
                "Reported": f"{a.reported:,.2f}",
                "Δ$": f"{a.diff_usd:+,.2f}",
                "Δ%": f"{a.diff_pct:+.2f}%",
                "Verdict": a.band,
            })
    return rows
