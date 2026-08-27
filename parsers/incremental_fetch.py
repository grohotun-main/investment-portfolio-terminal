"""Incremental price fetch — refresh only the missing tail, split-safe.

Pure engine (`incremental_refresh`) + a thin CSV I/O wrapper (`refresh_csv`)
used by fetch_daily_prices / fetch_long_history / fetch_benchmark so a routine
refresh re-fetches only a short trailing window instead of re-pulling years of
history. Two wrinkles drive the design: (1) a stock split makes Polygon
(adjusted=true) retroactively re-adjust the WHOLE series, and (2) the most
recent bars are mutable (an unsettled intraday close keeps moving between
pulls). So each run re-fetches from a SETTLED anchor (the latest stored bar at
least `settle_buffer_days` old) to today, replaces that window (correcting any
unsettled recent bars), keeps older rows, and split-checks by comparing the
anchor's close to disk -- a split re-adjusts the anchor too, so a mismatch
there means split -> full re-pull.
See docs/superpowers/specs/2026-06-05-incremental-price-fetch-design.md.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pandas as pd


class RegressionError(RuntimeError):
    """The merged result would drop an entity or shrink its row count vs disk.
    Raised before any write so a half-failed fetch can't clobber good history."""


def _iso(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def assert_not_regressive(existing: pd.DataFrame | None, merged: pd.DataFrame | None,
                          group_col: str | None) -> None:
    if existing is None or existing.empty:
        return
    if group_col is None:
        if len(merged) < len(existing):
            raise RegressionError(
                f"Refusing to overwrite — row count shrank "
                f"{len(existing)} -> {len(merged)}.")
        return
    old = existing.groupby(group_col).size().to_dict()
    new = (merged.groupby(group_col).size().to_dict()
           if merged is not None and not merged.empty else {})
    problems = [f"{e}: {n} -> {new.get(e, 0)}"
                for e, n in old.items() if new.get(e, 0) < n]
    if problems:
        raise RegressionError(
            "Refusing to overwrite — new dataset regresses:\n  " + "\n  ".join(problems))


def incremental_refresh(existing, entities, fetch_fn, *, lookback_start, today,
                        date_col="date", close_col="close",
                        group_col="symbol", full=False, rel_tol=1e-4,
                        settle_buffer_days=7, max_workers=8):
    """Return (merged_df, summary). Pure: no I/O; `today` + `fetch_fn` injected.

    fetch_fn(ticker, start, end) -> DataFrame with at least [date_col, close_col]
    (ALL its columns are preserved on write); returns an empty frame on
    error/no-data. summary counts entities per mode:
    full / append / current / resplit / kept.

    Per entity: re-fetch a trailing window from the SETTLED anchor (the latest
    stored bar at least `settle_buffer_days` old) to today, replace that window
    (so mutable/unsettled recent bars are corrected), keep older rows, and
    split-check on the anchor's close. No settled anchor (empty disk, --full, or
    a history younger than the buffer) -> full re-pull.

    The per-entity fetches run concurrently across `max_workers` threads (each
    entity is independent — it reads only its own slice of `existing` and the
    final merge sorts, so completion order is irrelevant). `fetch_fn` must
    therefore be safe to call from multiple threads; the bundled equity/benchmark
    closures capture immutable key/base strings and build fresh frames, so they
    are. `max_workers <= 1` or a single entity runs the old sequential path
    verbatim. An unexpected raise from any worker propagates out (after in-flight
    fetches finish — `cancel()` only drops not-yet-started ones), so the run
    aborts before any write rather than emitting a partial frame; a normal fetch
    error is still the empty-frame "kept" case.
    """
    if existing is None:
        existing = pd.DataFrame()
    entities = list(entities)
    settle_cutoff = today - timedelta(days=settle_buffer_days)
    summary = {"full": 0, "append": 0, "current": 0, "resplit": 0, "kept": 0}
    pieces: list[pd.DataFrame] = []

    def _ex_for(ent):
        # group_col=None -> single series: the whole frame IS the entity.
        if existing.empty or group_col is None:
            return existing
        return existing[existing[group_col] == ent]

    def _close_at(df, d):
        if df is None or df.empty:
            return None
        mask = pd.to_datetime(df[date_col]).dt.date == d
        return float(df.loc[mask, close_col].iloc[-1]) if mask.any() else None

    def _settled_anchor(df):
        # latest stored date old enough (>= settle_buffer_days) to be settled.
        if df is None or df.empty:
            return None
        dts = pd.to_datetime(df[date_col]).dt.date
        eligible = dts[dts <= settle_cutoff]
        return eligible.max() if not eligible.empty else None

    def _tag(df, ent):
        if group_col is not None and group_col not in df.columns:
            return df.assign(**{group_col: ent})
        return df

    def _full_pull(ent, ex):
        fresh = fetch_fn(ent, lookback_start, today)
        if fresh is None or fresh.empty:
            if ex is not None and not ex.empty:
                return ex, "kept"
            return None, None
        return _tag(fresh, ent), "full"

    def _resplit(ent, ex):
        refetch = fetch_fn(ent, lookback_start, today)
        if refetch is None or refetch.empty:
            return ex, "kept"
        return _tag(refetch, ent), "resplit"

    def _process_entity(ent):
        """One entity's (piece, mode). Reads only `existing` (its own slice) and
        calls `fetch_fn`; mutates no shared state -> safe to run concurrently.
        Returns (None, None) when a brand-new entity's fetch comes back empty."""
        ex = _ex_for(ent)
        anchor = None if (full or ex is None or ex.empty) else _settled_anchor(ex)
        if anchor is None:                                  # no settled anchor
            return _full_pull(ent, ex)
        fresh = fetch_fn(ent, anchor, today)                # window from anchor
        if fresh is None or fresh.empty:
            return ex, "kept"
        disk_c, fresh_c = _close_at(ex, anchor), _close_at(fresh, anchor)
        if disk_c is None or fresh_c is None:               # can't verify anchor
            return _resplit(ent, ex)
        if abs(fresh_c - disk_c) > rel_tol * abs(disk_c):   # split re-adjusted it
            return _resplit(ent, ex)
        # No split: replace the [anchor, today] window with fresh (correcting any
        # unsettled recent bars) and keep older disk rows verbatim.
        ex_dt = pd.to_datetime(ex[date_col]).dt.date
        fresh_dt = pd.to_datetime(fresh[date_col]).dt.date
        kept_old = ex[ex_dt < anchor]
        window = _tag(fresh[fresh_dt >= anchor], ent)
        piece = pd.concat([kept_old, window], ignore_index=True)
        mode = "append" if fresh_dt.max() > ex_dt.max() else "current"
        return piece, mode

    # Sequential for a single entity or max_workers<=1 (byte-for-byte the old
    # path; also the only way group_col=None is ever used). Else a bounded thread
    # pool overlaps the per-entity network waits.
    if max_workers <= 1 or len(entities) <= 1:
        results = [_process_entity(ent) for ent in entities]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_entity, ent) for ent in entities]
            try:
                results = [f.result() for f in as_completed(futures)]
            except Exception:
                for f in futures:
                    f.cancel()
                raise

    for piece, mode in results:
        if piece is not None:
            pieces.append(piece)
        if mode is not None:
            summary[mode] += 1

    if group_col is not None and not existing.empty:        # carry forward
        for ent in set(existing[group_col].unique()) - set(entities):
            pieces.append(existing[existing[group_col] == ent])

    merged = pd.concat(pieces, ignore_index=True) if pieces else existing.copy()
    if not merged.empty:
        merged[date_col] = _iso(merged[date_col])
        key = [date_col] if group_col is None else [group_col, date_col]
        merged = (merged.drop_duplicates(subset=key, keep="last")
                        .sort_values(key)
                        .reset_index(drop=True))
    assert_not_regressive(existing, merged, group_col)
    return merged, summary


def refresh_csv(out_csv, entities, fetch_fn, *, lookback_start, today,
                empty_columns, group_col="symbol", full=False,
                date_col="date", close_col="close", settle_buffer_days=7,
                max_workers=8):
    """Read out_csv (empty frame if absent) -> incremental_refresh -> write.
    Returns the per-mode summary dict."""
    out_csv = Path(out_csv)
    existing = (pd.read_csv(out_csv) if out_csv.exists()
                else pd.DataFrame(columns=list(empty_columns)))
    merged, summary = incremental_refresh(
        existing, entities, fetch_fn, lookback_start=lookback_start, today=today,
        date_col=date_col, close_col=close_col, group_col=group_col, full=full,
        settle_buffer_days=settle_buffer_days, max_workers=max_workers)
    merged.to_csv(out_csv, index=False)
    return summary
