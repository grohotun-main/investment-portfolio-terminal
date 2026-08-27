"""Cost-of-protection time series for the long-put hedge sleeve.

Builds a month-grain trajectory of cumulative protection cost:

    cost_to_date(t) = gross_premium_paid(<=t)
                    - gross_proceeds_received(<=t)
                    - sleeve_mv(t)

Positive values mean hedging has lost money to date (the typical state
on a calm tape; theta bleeds the open sleeve, realized losses bleed from
closures).

Statement-date quirk: JPM books on the last *business* day, Fidelity on
the last *calendar* day. For some months both fall in the same calendar
month with different statement_date values. We bin by month_end and take
the latest statement_date per broker per month, then sum across brokers
to get the true month-end sleeve MV.

Optional `snapshot_today_mv` lets the caller anchor the chart's final
point on a live Polygon mid (post-statement interim BUYs + decay since
last statement), which can differ from the latest statement MV by tens
of percent on a 30-day-old statement.

Optional `history_start` rebases the cost series so it starts at 0 on
the cutoff. Pre-cutoff rows are dropped; an anchor row at the cutoff
date is inserted so the chart begins at the user's selected start year.
The caller chart then reads as "what has hedging cost me SINCE cutoff".
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


_PUT_DESC = re.compile(r"\bPUT\b")


def _put_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Filter to PUT transactions (BUY / SELL / expire / transfer)."""
    if transactions.empty:
        return transactions.iloc[0:0]
    is_put = transactions["description"].fillna("").str.contains(
        _PUT_DESC, regex=True
    )
    return transactions[is_put].copy()


def _monthly_sleeve_mv(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-month put sleeve MV using latest-per-broker statement aggregation.

    Returns DataFrame [date, sleeve_mv] with one row per calendar month
    appearing in `positions`. The `date` is the actual latest statement
    date within that month across all brokers — NOT the month-end calendar
    date. This matters when:
      - JPM books on the last business day (e.g. 5/30) and Fidelity on
        the last calendar day (5/31): row dated 5/31 (the later one).
      - synthesize_interim_positions has rolled positions forward to a
        non-month-end date (e.g. 5/15): row dated 5/15, not 5/31. Critical
        for the today-vs-latest comparison in the snapshot append logic.
    Months with no put rows get sleeve_mv = 0 (but still appear so the
    chart's x-axis is contiguous).
    """
    if positions.empty:
        return pd.DataFrame(columns=["date", "sleeve_mv"])
    p = positions.copy()
    p["month_end"] = p["statement_date"] + pd.offsets.MonthEnd(0)
    # Latest statement_date per (broker, month_end). Some months JPM and
    # Fidelity both report at month-end, others split (e.g. JPM 5/30 vs
    # Fidelity 5/31, or both synth-rolled to 5/15) — keep the latest per
    # broker within each month.
    latest = (p.groupby(["broker", "month_end"])["statement_date"]
               .max().reset_index()
               .rename(columns={"statement_date": "_latest"}))
    p = p.merge(latest, on=["broker", "month_end"], how="left")
    p = p[p["statement_date"] == p["_latest"]]
    puts = p[p["asset_class"].fillna("").str.contains("option_put")]
    # MV summed per month_end; row date is the latest actual statement
    # date across brokers within that month.
    mv = (puts.groupby("month_end")["market_value"].sum()
              .reset_index()
              .rename(columns={"market_value": "sleeve_mv"}))
    anchor_date = (p.groupby("month_end")["statement_date"].max()
                    .reset_index()
                    .rename(columns={"statement_date": "date"}))
    all_months = pd.DataFrame({"month_end": sorted(p["month_end"].unique())})
    out = (all_months
           .merge(mv, on="month_end", how="left")
           .merge(anchor_date, on="month_end", how="left")
           .fillna({"sleeve_mv": 0.0})
           .drop(columns=["month_end"]))
    return out[["date", "sleeve_mv"]]


def build_protection_cost_timeline(
    transactions: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    snapshot_today_mv: Optional[float] = None,
    today: Optional[pd.Timestamp] = None,
    history_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Build the cost-of-protection trajectory.

    Args:
        transactions: caller-filtered (broker + interim-union as needed).
            All PUT-bearing rows here are included in the cumulative cash
            calc — filter to non-TEST brokers upstream.
        positions: caller-filtered. Statement-date positions across the
            real brokers; the per-broker latest-statement rule is applied
            internally.
        snapshot_today_mv: live Polygon snapshot MV of the sleeve. When
            supplied along with `today`, the function appends a final row
            dated `today` whose sleeve_mv is this value (rather than the
            latest statement MV).
        today: pd.Timestamp anchor for the final snapshot row. Required
            if snapshot_today_mv is set; ignored otherwise.
        history_start: cutoff date for the returned series. Pre-cutoff
            rows are dropped; cost_to_date is rebased to 0 on cutoff so
            the chart shows post-cutoff incremental cost. An anchor row
            is inserted at `history_start` with cost_to_date = 0.

    Returns:
        DataFrame [date, gross_paid, gross_received, sleeve_mv,
        cost_to_date]. Empty when no PUT activity exists. Always sorted
        ascending by date. Monetary columns are in dollars; cost_to_date
        is positive when the sleeve has lost money to date.
    """
    puts_t = _put_transactions(transactions)
    mv_m = _monthly_sleeve_mv(positions)
    empty = pd.DataFrame(columns=[
        "date", "gross_paid", "gross_received", "sleeve_mv", "cost_to_date"
    ])
    if puts_t.empty or mv_m.empty:
        return empty

    rows = []
    for _, mrow in mv_m.iterrows():
        d = mrow["date"]
        slc = puts_t[puts_t["settlement_date"] <= d]
        paid = float(-slc.loc[slc["amount"] < 0, "amount"].sum())
        recv = float(slc.loc[slc["amount"] > 0, "amount"].sum())
        mv = float(mrow["sleeve_mv"])
        rows.append({
            "date": d, "gross_paid": paid, "gross_received": recv,
            "sleeve_mv": mv, "cost_to_date": paid - recv - mv,
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Drop pre-activity months (no puts traded yet) — these would all show
    # cost_to_date = 0 and just stretch the x-axis pointlessly.
    df = df[df["gross_paid"] > 0].reset_index(drop=True)
    if df.empty:
        return empty

    # Snapshot-driven final point. Only added if `today` strictly exceeds
    # the latest statement month_end — otherwise the live snapshot would
    # contradict the statement on the same date.
    if (snapshot_today_mv is not None and today is not None):
        today_ts = pd.Timestamp(today).normalize()
        last_m = df["date"].max()
        if today_ts > last_m:
            slc = puts_t[puts_t["settlement_date"] <= today_ts]
            paid = float(-slc.loc[slc["amount"] < 0, "amount"].sum())
            recv = float(slc.loc[slc["amount"] > 0, "amount"].sum())
            mv = float(snapshot_today_mv)
            df = pd.concat([df, pd.DataFrame([{
                "date": today_ts, "gross_paid": paid, "gross_received": recv,
                "sleeve_mv": mv, "cost_to_date": paid - recv - mv,
            }])], ignore_index=True)

    # History-start rebase: shift series so cost_to_date == 0 on cutoff.
    if history_start is not None:
        hs = pd.Timestamp(history_start).normalize()
        pre = df[df["date"] <= hs]
        baseline = float(pre.iloc[-1]["cost_to_date"]) if not pre.empty else 0.0
        df = df[df["date"] >= hs].copy()
        df["cost_to_date"] = df["cost_to_date"] - baseline
        # Anchor row at cutoff so the chart starts there. Carry forward
        # the gross_paid / gross_received / sleeve_mv from the pre-cutoff
        # state so tooltips read sensibly.
        if df.empty or df.iloc[0]["date"] > hs:
            if not pre.empty:
                anchor = pre.iloc[-1].copy()
                anchor["date"] = hs
                anchor["cost_to_date"] = 0.0
            else:
                anchor = pd.Series({
                    "date": hs, "gross_paid": 0.0, "gross_received": 0.0,
                    "sleeve_mv": 0.0, "cost_to_date": 0.0,
                })
            df = pd.concat([pd.DataFrame([anchor]), df], ignore_index=True)

    return df.reset_index(drop=True)
