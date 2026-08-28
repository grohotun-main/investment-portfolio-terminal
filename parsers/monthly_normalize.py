"""Normalize a positions frame to one snapshot per (account, calendar month).

Pure function extracted from app.py so it can be unit-tested without importing
the full Streamlit module (same pattern as parsers/mark_to_market.py). The
dashboard imports `monthly_normalize` from here and wraps it with
`st.cache_data`; tests import it directly.
"""
from __future__ import annotations

import pandas as pd


def monthly_normalize(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize positions to one snapshot per (account, calendar month).

    Harbor reports on the last business day, Alpine on the last calendar day,
    so the same month can have two statement_dates. For each (account, month)
    keep the position-set whose statement_date is latest in that month.

    Then forward-fill any month an account is missing, up to the portfolio's
    global latest month:

      - INTERNAL gap — a month between an account's first and last real
        statement (e.g. a Alpine TOD account skips a calendar month).
      - TRAILING gap — a month AFTER an account's last real statement, when
        that account lags the newest broker statement (e.g. Alpine issued
        May statements for most accounts but not for a lagging one). Without
        this the lagging account silently drops out of the latest snapshot
        (~$89K vanishing from Holdings/NAV), instead of showing last-known.

    Without filling, the portfolio NAV chart drops for the missing month
    because the absent account is treated as $0. Filled rows carry the most
    recent known positions verbatim and are tagged `_filled=True`;
    `_as_of_date` records the original statement_date they were carried from
    (preserved through multi-month chains) so the Holdings tab can badge them
    "as of <month>".

    Invariant: the global latest month and every per-month canonical fill date
    are computed *within the frame as passed in*. Callers that pre-filter
    (broker / history-start) must filter BEFORE calling, so each trailing gap
    `<= global_max` still has a real statement in-frame to source its canonical
    date from — otherwise a gap could fall back to a calendar month-end that
    isn't a real statement_date the Holdings "as of" selector offers.
    """
    df = positions.copy()
    df["month"] = df["statement_date"].dt.to_period("M")

    latest_per = (df.groupby(["account_id", "month"])["statement_date"]
                  .max().reset_index()
                  .rename(columns={"statement_date": "_keep_date"}))
    df = df.merge(latest_per, on=["account_id", "month"])
    df = df[df["statement_date"] == df["_keep_date"]].drop(columns=["_keep_date"])
    df["_filled"] = False
    df["_as_of_date"] = df["statement_date"]

    if df.empty:
        return df

    all_months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    global_max = df["month"].max()
    # Canonical snapshot date for each month = the latest real statement_date
    # any account carries that month. Filled rows adopt it so they line up with
    # the Holdings "as of" selector (which only offers real statement dates)
    # and reach mark-to-market (which marks statement_date == max). Months with
    # no real statement anywhere fall back to the calendar month-end.
    month_dates = df.groupby("month")["statement_date"].max()

    def _fill_date(m: pd.Period) -> pd.Timestamp:
        if m in month_dates.index:
            return month_dates[m]
        return m.to_timestamp("M").normalize()

    fills: list[pd.DataFrame] = []
    for acct, g in df.groupby("account_id"):
        present = set(g["month"].unique())
        a_min = g["month"].min()
        # Internal gaps (first < m < last real) AND trailing gaps (m after the
        # last real statement, up to the global latest month). Months before an
        # account's debut are left alone — it did not exist yet.
        gaps = sorted(m for m in all_months
                      if a_min < m <= global_max and m not in present)
        if not gaps:
            continue
        # Ascending so each gap carries from the month immediately before it —
        # which may itself be a fill we just produced (a multi-month lag). The
        # original real rows live in `g`; chained fills in `carried`.
        carried: dict[pd.Period, pd.DataFrame] = {}
        for gap_m in gaps:
            prior_month = max(m for m in present if m < gap_m)
            src = carried.get(prior_month)
            prior_rows = (src if src is not None
                          else g[g["month"] == prior_month]).copy()
            prior_rows["month"] = gap_m
            prior_rows["statement_date"] = _fill_date(gap_m)
            prior_rows["_filled"] = True
            # _as_of_date is NOT overwritten — it propagates the original real
            # statement_date through the chain.
            carried[gap_m] = prior_rows
            fills.append(prior_rows)
            present.add(gap_m)
    if fills:
        df = pd.concat([df, *fills], ignore_index=True)
    return df.sort_values(["account_id", "month"]).reset_index(drop=True)


def month_canonical_dates(positions: pd.DataFrame) -> list[pd.Timestamp]:
    """One canonical statement_date per calendar month, newest first.

    The canonical date is the month's LATEST statement_date across all
    accounts — the same date `monthly_normalize` stamps on filled rows
    (``groupby('month')['statement_date'].max()``) and the date
    mark-to-market marks. Feeds the Holdings "as of" picker so a dual-date
    month (Harbor last-biz-day vs Alpine month-end) offers ONE entry whose
    month-slice is the full portfolio, not one broker's partial slice (WSF-2).

    Empty / column-less frame -> empty list.
    """
    if positions.empty or "statement_date" not in positions.columns:
        return []
    dates = pd.to_datetime(positions["statement_date"])
    per_month = dates.groupby(dates.dt.to_period("M")).max()
    return sorted((pd.Timestamp(d).normalize() for d in per_month),
                  reverse=True)


def slice_as_of_month(df: pd.DataFrame, as_of, *,
                      date_col: str = "statement_date") -> pd.DataFrame:
    """Rows whose ``date_col`` falls in the same calendar month as ``as_of``.

    Drop-in replacement for the exact ``df[df[date_col] == as_of]`` filters so
    dual-date months return BOTH brokers' rows. Returns a copy (call sites
    mutate the result). Empty / column-less frame -> empty copy.
    ``as_of=None`` / ``pd.NaT`` returns an empty frame (matches the prior
    ``df[df[date_col] == NaT]`` no-op when the Holdings picker has no selection).
    """
    if df.empty or date_col not in df.columns:
        return df.copy()
    month_ts = pd.Timestamp(as_of)
    if pd.isna(month_ts):
        # No month selected (None/NaT) — empty slice, matching the prior
        # `df[df[date_col] == NaT]` behavior so an empty picker / empty
        # positions frame degrades gracefully instead of raising.
        return df.iloc[0:0].copy()
    month = month_ts.to_period("M")
    in_month = pd.to_datetime(df[date_col]).dt.to_period("M") == month
    return df[in_month].copy()
