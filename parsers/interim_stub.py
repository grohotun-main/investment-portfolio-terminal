"""Provisional interim stub period for the return series (spec 2026-08-22).

One Modified-Dietz period from the last statement month-end to the latest
price date, on ONE NAV basis (the marked monthly snapshot at both ends), with
the interim transactions' recognised external flows time-weighted. Never
persisted: consumers that opt in (Performance headline + wealth chart,
vs-Benchmark table/summary/growth, KPI tape, AI facts) append it at request
time via ``append_stub``; ``twr_portfolio`` itself — and every monthly-period
statistic built on it — stays statement-anchored. Flows settled after
``end_date`` get weight 0 (their cash is already inside ``nav_end``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from compute_twr import FLOW_TYPES, modified_dietz_period


@dataclass(frozen=True)
class InterimStub:
    start_date: pd.Timestamp      # last statement month-end
    end_date: pd.Timestamp        # latest price date
    nav_start: float              # snapshot at start_date (statement basis)
    nav_end: float                # latest snapshot, marked to live
    net_flow: float               # signed sum of recognised flows
    n_flows: int
    return_pct: float
    days: int
    flows_through: pd.Timestamp | None

    def as_facts(self) -> dict:
        """Dollar-free, deny-regex-clean block for the AI facts."""
        sign = "in" if self.net_flow > 0 else "out" if self.net_flow < 0 else "none"
        return {"start": self.start_date.strftime("%Y-%m-%d"),
                "end": self.end_date.strftime("%Y-%m-%d"),
                "stub_return_pct": round(self.return_pct * 100.0, 2),
                "stub_days": int(self.days),
                "n_flows": int(self.n_flows),
                "net_flow_sign": sign,
                "flows_through": (self.flows_through.strftime("%Y-%m-%d")
                                  if self.flows_through is not None else None),
                "provisional": True}


def stub_flows(transactions: pd.DataFrame | None, *,
               after: "pd.Timestamp | str") -> pd.DataFrame:
    """Recognised external flows settled after ``after``: FLOW_TYPES rows
    (amounts signed as stored), minus ``flow_scope == 'internal'`` legs and
    rows with no parseable amount. Interim rows carry no flow_scope, so the
    TYPE filter — not compute_twr's scope filter — is the gate here."""
    empty = pd.DataFrame(columns=["settlement_date", "amount"])
    if (transactions is None or transactions.empty
            or "settlement_date" not in transactions.columns
            or "transaction_type" not in transactions.columns):
        return empty
    sd = transactions["settlement_date"]
    if not is_datetime64_any_dtype(sd):
        sd = pd.to_datetime(sd, errors="coerce")
    # Narrow on the native column first (the statement frame is ~10k rows,
    # the interim slice ~100), then type/scope/amount on the survivors.
    t = transactions.loc[sd > pd.Timestamp(after)]
    t = t[t["transaction_type"].isin(FLOW_TYPES)]
    if "flow_scope" in t.columns:
        t = t[t["flow_scope"].fillna("").astype(str) != "internal"]
    amt = pd.to_numeric(t["amount"], errors="coerce")
    t = t.loc[amt.notna()]
    out = pd.DataFrame({"settlement_date": pd.to_datetime(t["settlement_date"]),
                        "amount": amt.loc[t.index].astype(float)})
    return out.reset_index(drop=True)


def compute_interim_stub(start_date: "pd.Timestamp | str", end_date: "pd.Timestamp | str",
                         nav_start: float, nav_end: float,
                         flows: pd.DataFrame | None) -> InterimStub | None:
    """None when the period is empty/inverted, a NAV end is non-finite or
    non-positive, or Dietz returns NaN. No clipping of the result."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or end <= start:
        return None
    try:
        v0, v1 = float(nav_start), float(nav_end)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(v0) and np.isfinite(v1)) or v0 <= 0 or v1 <= 0:
        return None
    fl = flows if flows is not None else pd.DataFrame(columns=["settlement_date", "amount"])
    r = modified_dietz_period(v0, v1, fl, start, end)
    if r is None or not np.isfinite(r):
        return None
    return InterimStub(start_date=start, end_date=end, nav_start=v0, nav_end=v1,
                       net_flow=float(fl["amount"].sum()) if len(fl) else 0.0,
                       n_flows=int(len(fl)), return_pct=float(r),
                       days=int((end - start).days),
                       flows_through=(pd.Timestamp(fl["settlement_date"].max())
                                      if len(fl) else None))


def append_stub(twr: pd.DataFrame | None, stub: InterimStub | None) -> pd.DataFrame | None:
    """``twr`` (raw or prepared twr_portfolio shape) plus one provisional row;
    a boolean ``provisional`` column is always added (False on existing rows).
    Derived columns present on the input (cum_return / wealth_index /
    wealth_peak / twr_dd_pct / month_end) are recomputed so the new row chains.
    Date columns keep their dtype (datetime or ISO string)."""
    if twr is None:
        return None
    out = twr.copy()
    if "provisional" not in out.columns:
        out["provisional"] = False
    if stub is None:
        return out

    dt_cols = {c for c in out.columns if is_datetime64_any_dtype(out[c])}

    def _date(col, ts):
        return ts if col in dt_cols else ts.strftime("%Y-%m-%d")

    # Empty cells typed per column (NaT for datetimes) so concat keeps dtypes.
    row = {c: (pd.NaT if c in dt_cols else np.nan) for c in out.columns}
    if "month" in out.columns:
        row["month"] = (pd.Period(stub.end_date, freq="M")
                        if isinstance(out["month"].dtype, pd.PeriodDtype)
                        else stub.end_date.strftime("%Y-%m"))
    row.update({"statement_date": _date("statement_date", stub.end_date),
                "prev_stmt_date": _date("prev_stmt_date", stub.start_date),
                "month_end": _date("month_end", stub.end_date.normalize()),
                "nav": stub.nav_end, "prev_nav": stub.nav_start,
                "net_external_flow": stub.net_flow, "return_pct": stub.return_pct,
                "n_flows": stub.n_flows, "is_real_statement": False,
                "provisional": True})
    row = {k: v for k, v in row.items() if k in out.columns}
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    if "cum_return" in out.columns:
        wealth = (1.0 + out["return_pct"].fillna(0.0)).cumprod()
        out["cum_return"] = wealth - 1.0
        for col, val in (("wealth_index", wealth), ("wealth_peak", wealth.cummax()),
                         ("twr_dd_pct", (wealth / wealth.cummax() - 1.0) * 100.0)):
            if col in out.columns:
                out[col] = val
    return out


def stub_block(stub: InterimStub | None, fmt_pct: "Callable[[float], str]") -> dict | None:
    """Payload block for the tabs: ``as_facts()`` plus a one-line caption.
    ``fmt_pct`` formats a percent-scale float (the services' ``_spct``)."""
    if stub is None:
        return None
    d = stub.as_facts()
    through = (f"interim through {stub.flows_through:%Y-%m-%d} · "
               if stub.flows_through is not None else "")
    d["caption"] = (f"{stub.start_date:%b %d} → {stub.end_date:%b %d, %Y} provisional: "
                    f"{fmt_pct(stub.return_pct * 100.0)} · marked to live · {through}"
                    "unaudited until the next statement lands")
    return d


def to_date_cagr(cum: float, days: int) -> float:
    """Day-count annualisation for a to-date cumulative return."""
    if (days is None or days <= 0 or cum is None
            or not np.isfinite(cum) or cum <= -1.0):
        return float("nan")
    return float((1.0 + cum) ** (365.0 / float(days)) - 1.0)


def chain(cum: float, r: float) -> float:
    """Cumulative ``cum`` extended by one more period return ``r``."""
    if cum is None or r is None or not (np.isfinite(cum) and np.isfinite(r)):
        return float("nan")
    return float((1.0 + cum) * (1.0 + r) - 1.0)


def to_date_span(port: pd.DataFrame, stub: InterimStub) -> tuple[pd.Timestamp, int]:
    """``(origin, days)`` for annualising a to-date cumulative: ``origin`` is
    the return series' true start — the first return-bearing row's
    ``prev_stmt_date``, else the month-end BEFORE that row's month (scoped
    frames rebuilt by ``recompute_portfolio_twr`` carry a return on row 0
    with a NaT ``prev_stmt_date``; the canonical frame has a NaN-return row 0
    whose successor carries the date). ``days`` = ``stub.end_date - origin``."""
    first = port.loc[port["return_pct"].notna()] if "return_pct" in port.columns else port
    row = first.iloc[0] if len(first) else port.iloc[0]
    prev0 = row["prev_stmt_date"] if "prev_stmt_date" in row.index else None
    if prev0 is not None and pd.notna(prev0):
        origin = pd.Timestamp(prev0).normalize()
    else:
        month = row["month"] if "month" in row.index else None
        try:
            per = month if isinstance(month, pd.Period) else pd.Period(str(month), freq="M")
            origin = (per - 1).to_timestamp(how="end").normalize()
        except (ValueError, TypeError):
            origin = (pd.Timestamp(row["statement_date"]) - pd.offsets.MonthEnd(1)).normalize()
    return origin, int((pd.Timestamp(stub.end_date) - origin).days)


def bench_stub_return(tr: pd.Series | None, stub: InterimStub | None) -> float | None:
    """Benchmark TR return over the stub's dates, or None when the series is
    absent or does not cover BOTH ends (one gate for every consumer)."""
    if stub is None or tr is None or len(tr) == 0:
        return None
    if not (tr.index.min() <= stub.start_date and stub.end_date <= tr.index.max()):
        return None
    try:
        return float(tr.loc[stub.end_date] / tr.loc[stub.start_date] - 1.0)
    except KeyError:
        return None


def ytd_to_date(cum_p: float, cum_b: float, stub: InterimStub,
                bench_stub: float) -> tuple[float, float]:
    """To-date YTD pair. Statement YTD is anchored to the LAST STATEMENT's
    calendar year; when the stub ends in a later year (a December close with
    January prices) the to-date YTD is the stub alone on both sides, never
    the prior full year chained with January."""
    if stub.end_date.year > stub.start_date.year:
        return float(stub.return_pct), float(bench_stub)
    return chain(cum_p, stub.return_pct), chain(cum_b, bench_stub)
