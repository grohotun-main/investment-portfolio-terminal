"""Global chrome view for the terminal (QA-polish S7): the four sidebar
staleness warnings, the Data-sources panel, the regime badge, and the footer —
the module-scope chrome app.py renders around its tabs (app.py:1131-1208
warnings, 1436-1488 data sources, 1660-1684 regime badge, 9100-9111 footer).
Wraps the same engines and files; zero new math.

``today`` is injectable so the day-count fields golden-test deterministically
(the route always uses the real clock).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# parsers/ is a flat module directory on sys.path (the risk_bundle convention).
from refresh_prices import bench_tr_staleness_days
from risk_metrics import (
    DR_LONG_W,
    DR_SHORT_W,
    classify_dr_regime,
    compute_dr_frames,
    rf_staleness_business_days,
)

from terminal import holdings_service as hs
from terminal import risk_service as rs
from terminal.holdings_service import Frames
from terminal.performance_service import _resolve_filter

_PRICE_STALE_DAYS = 5          # mirror app.py's _PRICE_STALE_DAYS
_DOT = {"Stress": "🔴", "Normal": "⚪", "Calm": "🟢"}


def _load_prices_latest(data_dir: Path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Mirror app.py.load_prices_latest (app.py:355-378) without the st cache:
    (df + staleness_days column, prices_as_of = max ok as_of_date)."""
    path = Path(data_dir) / "prices_latest.csv"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    ok = df[df["status"] == "ok"]
    as_of = pd.Timestamp(ok["as_of_date"].max()) if not ok.empty else None
    if as_of is not None:
        df["staleness_days"] = (as_of - df["as_of_date"]).dt.days
    else:
        df["staleness_days"] = float("nan")
    return df, as_of


def _warnings_and_sources(frames: Frames, data_dir: Path,
                          today: pd.Timestamp) -> tuple[str, list, dict]:
    prices_df, prices_as_of = _load_prices_latest(data_dir)

    if prices_as_of is not None:
        prices_caption = (f"Prices as of {prices_as_of.strftime('%b %d, %Y')} — "
                          "live marks for ticker'd holdings; no-ticker rungs "
                          "(the treasury ladder) stay at their statement price.")
    else:
        prices_caption = ("Prices as of statement date — run Refresh market "
                          "data for live marks.")

    warnings: list[dict] = []
    latest = (pd.Timestamp(frames.available_dates[0])
              if frames.available_dates else None)
    if latest is not None:
        lag = int((today - latest.normalize()).days)
        if lag > 7:
            warnings.append({"icon": "⏱", "text":
                f"Holdings {lag} days stale — share counts last updated "
                f"{latest.strftime('%b %d')}. Drop a fresh broker CSV in "
                f"Interim Transactions/ (or run Pull interim transactions)."})

    stale_rows: list[dict] = []
    if not prices_df.empty and "staleness_days" in prices_df.columns:
        stale = prices_df[
            (prices_df["status"] == "ok")
            & (prices_df["staleness_days"].fillna(0) > _PRICE_STALE_DAYS)
        ]
        if not stale.empty:
            worst = int(stale["staleness_days"].max())
            warnings.append({"icon": "📉", "text":
                f"{len(stale)} symbol(s) with stale prices — up to {worst}d "
                f"behind the rest of the holding universe. MTM marks at "
                f"last-known close; the list is under Data sources."})
            s = (stale[["symbol", "as_of_date", "close", "staleness_days"]]
                 .sort_values("staleness_days", ascending=False))
            stale_rows = [{
                "symbol": str(r.symbol),
                "last_bar": r.as_of_date.strftime("%Y-%m-%d"),
                "last_close": f"${r.close:,.2f}",
                "days": int(r.staleness_days),
            } for r in s.itertuples()]

    # Both benchmark TR legs: SPY, and the 60/40 blend's AGG leg — the AGG
    # file used to sit outside every guard, so its staleness silently
    # truncated JPM-scoped benchmark series (DA-B-2). A data dir without
    # the AGG file (fixture) probes None and stays silent.
    for _tick in ("SPY", "AGG"):
        try:
            bench_lag = bench_tr_staleness_days(Path(data_dir), _tick)
        except Exception:                 # probe, not a gate (file may be absent)
            bench_lag = None
        if bench_lag and bench_lag >= 2:
            warnings.append({"icon": "📈", "text":
                f"{_tick} total-return series is {int(bench_lag)} trading "
                f"day(s) behind daily_prices — β/α and the bench spread "
                f"silently ffill the gap. Run Refresh market data."})

    try:
        # today MUST thread through — the probe defaults to the real clock,
        # which baked the golden-generation date into the golden and broke
        # the CI re-run one business day later (main run 2026-07-13).
        rf_lag = rf_staleness_business_days(Path(data_dir), today=today)
    except Exception:
        rf_lag = None
    if rf_lag and rf_lag >= 5:
        warnings.append({"icon": "💰", "text":
            f"Risk-free rate (FRED DGS3MO) is {int(rf_lag)} business day(s) "
            f"behind today — Sharpe/Sortino use a forward-filled stale RF. "
            f"Run Refresh market data."})

    # Data-sources caption (app.py:1447-1461). Counts are post-broker-filter
    # here (the terminal seam narrows frames before this runs) — app.py's sit
    # above its broker rebind, a documented micro-divergence.
    positions, transactions = frames.positions, frames.transactions
    prices_line = ""
    if not prices_df.empty:
        n_ok = int((prices_df["status"] == "ok").sum())
        n_total = len(prices_df)
        n_stale = n_total - n_ok - int((prices_df["status"] == "cash_fixed_1").sum())
        asof_txt = prices_as_of.strftime("%b %d, %Y") if prices_as_of else "—"
        prices_line = (f" Prices: {n_ok}/{n_total} symbols marked from Massive "
                       f"as of {asof_txt}"
                       + (f", {n_stale} fallback to statement price." if n_stale else "."))
    caption = (
        f"Positions: {len(positions):,} rows over "
        f"{positions['statement_date'].nunique()} statement-dates, "
        f"{frames.positions_monthly['month'].nunique() if not frames.positions_monthly.empty else 0} normalized months. "
        f"Transactions: {len(transactions):,} rows (Fidelity + JPM). "
        f"Accounts: {positions['bucket'].nunique()} sub-accounts. "
        f"Latest data: positions through "
        f"{positions['statement_date'].max().strftime('%b %d, %Y')}"
        + (f", transactions through "
           f"{transactions['settlement_date'].max().strftime('%b %d, %Y')}"
           if not transactions.empty else "")
        + prices_line
    ) if not positions.empty else "No positions loaded."

    return prices_caption, warnings, {"caption": caption, "stale_rows": stale_rows}


def _regime(frames: Frames) -> dict:
    """The sidebar regime badge (app.py:1660-1684): whole-book DR short/long
    ratio via the shared risk bundle + compute_dr_frames, fixed 0.90/1.10
    bands (classify_dr_regime defaults, like app.py's badge)."""
    unavailable = {"available": False}
    try:
        # Resolve the "all" selection the way every risk route does — the
        # bundle ALWAYS applies its bucket/class lists, so "all" means the
        # full id sets, not empty lists (the #232 boundary-widening contract).
        snap_all = hs._current_snap(frames)
        (bucket_filter, class_filter, _ids,
         account_active, class_active) = _resolve_filter(frames, snap_all, "all", "all")
        b = rs._bundle(frames, bucket_filter, class_filter,
                       account_active, class_active)
    except Exception:
        return unavailable
    weights, daily = b["weights"], frames.daily_prices
    if weights.empty or daily.empty:
        return unavailable
    f = compute_dr_frames(weights, daily, b["port_rets"])
    if not f.get("available"):
        return unavailable
    reg = classify_dr_regime(f["dr_s"], f["dr_l"])
    if not math.isfinite(reg.get("ratio", float("nan"))):
        return unavailable
    return {
        "available": True,
        "dot": _DOT.get(reg["label"], "⚫"),
        "label": reg["label"],
        "line": (f"DR {DR_SHORT_W}d / {DR_LONG_W}d = "
                 f"{reg['dr_short']:.2f}× / {reg['dr_long']:.2f}× "
                 f"(ratio {reg['ratio']:.2f})"),
        "help": ("Ratio = short-window DR ÷ long-window DR (both ≥ 1 "
                 "long-only, rolling sample std). Below 1 means "
                 "diversification has compressed recently vs the long "
                 "baseline; above 1 means it's loosening. 🔴 Stress = "
                 "correlations clustering · ⚪ Normal · 🟢 Calm = "
                 "decorrelating. Bands at 0.90 / 1.10."),
    }


def _footer(frames: Frames) -> str:
    """app.py:9100-9111 verbatim."""
    t = frames.transactions
    if t.empty:
        return ("Phase 0 reconciliation: 97% across 167 account-statements. "
                "Phase 1 transactions: not loaded.")
    return ("Phase 0 reconciliation: 97% across 167 account-statements. "
            f"Phase 1 transactions: Fidelity + JPM merged, {len(t):,} rows "
            f"({t['settlement_date'].min().strftime('%b %Y')} - "
            f"{t['settlement_date'].max().strftime('%b %Y')}). "
            "SPY and 60/40 SPY/TLT benchmarks integrated (Massive/Polygon). "
            "Next up: AGG blend for fixed-income context.")


def build_chrome_view(frames: Frames, data_dir, today=None) -> dict:
    today_ts = (pd.Timestamp(today) if today is not None
                else pd.Timestamp.today()).normalize()
    prices_caption, warnings, sources = _warnings_and_sources(
        frames, Path(data_dir), today_ts)
    return {
        "prices_caption": prices_caption,
        "warnings": warnings,
        "data_sources": sources,
        "regime": _regime(frames),
        "footer": _footer(frames),
        # The persistent KPI strip is broker/history-scoped global chrome, so
        # it ships here too — the Holdings payload alone can't repaint it when
        # a filter changes while another tab is active.
        "tape": hs._kpi_tape(frames),
    }
