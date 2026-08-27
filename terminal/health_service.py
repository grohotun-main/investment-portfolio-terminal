# terminal/health_service.py
"""Pure data seam for the MERIDIAN Terminal "Data health" tab.

Re-expresses app.py._render_health_body (9194-9218). The whole surface comes
from the importable, Streamlit-free parsers/data_health.py — build_health_report
-> format_health_headline / health_rows_to_table — so the numbers match the
Streamlit tab 1:1 by construction. The report-building boilerplate lives here
(build_health_report_for_frames) and holdings_service._health delegates to it,
so the chrome verdict strip and this tab cannot drift.

No query params: the report is whole-book (it ignores Account / Asset-class).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from data_health import (HealthReport, build_health_report,
                         format_health_headline, health_rows_to_table)
from reconcile_holdings import load_allowlist

from terminal import holdings_service as hs

# data_health's {green,amber,red,grey} -> the front-end callout level set
# (identical map holdings_service._health uses for the chrome strip).
_LEVEL_MAP = {"green": "success", "amber": "warning",
              "red": "error", "grey": "muted"}


def build_health_report_for_frames(frames: hs.Frames, *,
                                   today: date | None = None) -> HealthReport:
    """Build the ingest HealthReport from a loaded Frames bundle, exactly as
    app.py:1484 does (summaries.csv + allowlist + account labels). Shared by the
    chrome verdict strip (holdings_service._health) and this tab so they agree
    by construction. `today` feeds only days_since, which neither UI renders, so
    the rendered surface is deterministic.

    Reads ``frames.summaries`` (loaded once in ``load_frames``, narrowed by
    ``apply_global_filters``) rather than re-reading summaries.csv from
    ``frames.data_dir`` — a fresh read would bypass the broker choke-point and
    leak real per-account reported totals under a test-only broker selection.
    `today` is the AI-facts test seam (callers that pass nothing keep the
    wall clock)."""
    return build_health_report(
        frames.positions, frames.summaries,
        today=today or pd.Timestamp.today().normalize().date(),
        label_by_account=hs.ACCOUNT_DISPLAY,
        allowlist=load_allowlist(),
    )


def build_health_view(frames: hs.Frames) -> dict:
    """Assemble the GET /api/health contract. Pure given frames.

    Keys: meta, headline {level,text}, summary {counts+worst}, table (the 9-col
    reconciliation rows verbatim from health_rows_to_table), message (the two
    empty states; None when there is a table).
    """
    report = build_health_report_for_frames(frames)
    level, text = format_health_headline(report)
    table = health_rows_to_table(report)

    # Filter options drive the (inert, but always-visible) shared chrome selects
    # so a ?tab=health-first load doesn't leave them empty (same as the other
    # tabs). The tab itself ignores them.
    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)

    meta = {
        "as_of_month": report.as_of_month,
        "recon_available": report.recon_available,
        "accounts": acct_opts,
        "classes": class_opts,
        "brokers": broker_opts,
        "available_dates": list(frames.available_dates),
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "filter": {"account": "all", "asset_class": "all", "broker": "all"},
    }
    summary = {
        "n_ok": report.n_ok, "n_known": report.n_known,
        "n_watch": report.n_watch, "n_error": report.n_error,
        "n_carried": report.n_carried, "worst_level": report.worst_level,
    }

    # Empty-state message mirrors app.py 9216-9217 EXACTLY: Streamlit emits a
    # body message ONLY in the reconciled-but-empty case (`elif
    # health_report.recon_available: st.info(...)`). When recon is UNAVAILABLE
    # it shows no body message — the grey headline already states it, so adding
    # one here would double the headline and diverge from the rendered body.
    message = None
    if report.recon_available and not table:
        message = {"text": "No accounts to reconcile for the latest statement "
                           "month.", "level": "info"}

    return {
        "meta": meta,
        "headline": {"level": _LEVEL_MAP.get(level, "muted"), "text": text},
        "summary": summary,
        "table": table,
        "message": message,
    }
