"""Tax S3 engine: TLH harvest scanner + cross-account wash-sale guard.

Pure functions over the gated lot ledger (data/lots.csv), the canonical
transactions frame and the shared prices_latest marks. Nothing here writes
under data/; the CLI (Task 9) prints a report and exits.

Semantics (spec §4, locked): a replacement acquisition is a
buy/reinvestment row of the SAME instrument key — resolved through the
ledger's own crosswalk, never raw symbol equality — in ANY account, IRAs
included (an IRA replacement disallows the loss permanently). Option rows
never match (options live outside the lot ledger). The window is 30
calendar days each side, inclusive. Candidates are taxable-account,
reconstructed-evidence, priced open loss lots only.

Design: docs/superpowers/specs/2026-07-27-tax-s3-tlh-wash-sale-design.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# Only touch the console encoding / sys.path when run as a script (py
# parsers/tax_scanner.py). Importing this module as part of the parsers
# package (e.g. terminal/tax_service.py, inside the terminal server's own
# process) must have no side effect on that process's stdout/stderr.
if __package__ in (None, ""):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsers.lot_engine import (  # noqa: E402
    build_key_resolvers, classify_term, is_option_row,
    load_corporate_identity, resolve_instrument_key,
)

WINDOW_DAYS = 30
ACQUISITION_TYPES = ("buy", "reinvestment")

_ACQ_COLUMNS = ["account_id", "instrument_key", "wash_date", "quantity",
                "transaction_type", "source_row"]


def account_types(positions: pd.DataFrame) -> dict[str, str]:
    """account_id -> account_type, latest statement's label winning — the
    same latest-wins canon the lot engine derives (slice 1)."""
    if positions is None or positions.empty \
            or "account_type" not in positions.columns:
        return {}
    posf = positions[positions["account_type"].notna()].copy()
    posf["_sdate"] = pd.to_datetime(posf["statement_date"], errors="coerce")
    posf = posf.sort_values("_sdate")
    return {str(acct): str(grp["account_type"].iloc[-1])
            for acct, grp in posf.groupby("account_id", sort=False)}


def taxable_of(acct_type: str | None) -> bool | None:
    """True taxable / False IRA / None unknown (callers exclude fail-closed)."""
    if not acct_type:
        return None
    return "ira" not in str(acct_type).lower()


def price_map(prices_latest: pd.DataFrame | None) -> dict[str, float]:
    """symbol -> live price from the shared prices_latest frame ('ok' rows
    at their close; cash sweeps fixed at 1.0), same marks Holdings uses."""
    if prices_latest is None or prices_latest.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in prices_latest.iterrows():
        status = str(row.get("status") or "")
        sym = str(row.get("symbol") or "")
        if not sym:
            continue
        if status == "ok" and pd.notna(row.get("close")):
            out[sym] = float(row["close"])
        elif status == "cash_fixed_1":
            out[sym] = 1.0
    return out


def tx_dates(tx: pd.DataFrame) -> pd.Series:
    """The ledger's one date per transaction row: trade date, falling back
    to settlement — the same `_date` rule parsers/lot_engine.py replays on,
    so the detector and the replay date rows identically.

    A column that isn't there degrades to all-NaT rather than raising:
    `DataFrame.get` returns None for a missing column and
    `pd.to_datetime(None)` is the NaT SCALAR, whose `.fillna` does not
    exist. That threw an AttributeError one frame below every caller —
    tolerable in the CLI, a 500 on the terminal's Tax tab.
    """
    def col(name: str) -> pd.Series:
        if name not in getattr(tx, "columns", ()):
            return pd.Series(pd.NaT, index=tx.index, dtype="datetime64[ns]")
        return pd.to_datetime(tx[name], errors="coerce")

    return col("trade_date").fillna(col("settlement_date"))


def tx_frontier_of(tx: pd.DataFrame):
    """Latest observed transaction date, or None when there is none.

    This is how far forward the wash guard can see at all. Month-end
    statements put it BEHIND today, so a backward wash window routinely
    reaches past it — every consumer must say so rather than let a "clear"
    read as "safe" (spec Update item 2).
    """
    if tx is None or len(tx) == 0:
        return None
    frontier = tx_dates(tx).max()
    return None if pd.isna(frontier) else frontier


def keyed_acquisitions(tx: pd.DataFrame,
                       resolver: Optional[dict[str, str]] = None,
                       cusip_resolver: Optional[dict[str, str]] = None,
                       fold: Optional[dict[str, str]] = None,
                       *, types: tuple[str, ...] = ACQUISITION_TYPES
                       ) -> pd.DataFrame:
    """buy/reinvestment rows keyed in the ledger's instrument-key space.

    `types` (default ACQUISITION_TYPES) lets a caller key other row types
    — the chat wash calendar keys sells this way — in the same
    instrument-key space, with the same wash_date / source_row rules.

    wash_date prefers trade_date, falls back to settlement — the ledger's
    own `_date` rule, so the detector and the replay date rows identically.
    Option rows are dropped: options live outside the lot ledger, so an
    option acquisition must never block an equity harvest.

    source_row carries the row's position in the ORIGINAL `tx` frame —
    the same positional index parsers/lot_engine.py stamps onto a lot's
    own source_row — so a candidate lot's opening purchase can be matched
    and excluded from its own blocking set by row identity, never by a
    date/quantity heuristic.
    """
    if tx is None or tx.empty:
        return pd.DataFrame(columns=_ACQ_COLUMNS)
    acq = tx[tx["transaction_type"].astype(str).isin(types)]
    if acq.empty:
        return pd.DataFrame(columns=_ACQ_COLUMNS)
    wash_date = tx_dates(acq)
    rows = []
    for idx, row in acq.iterrows():
        if is_option_row(row.get("description")):
            continue
        if pd.isna(wash_date.loc[idx]):
            continue
        key, _source = resolve_instrument_key(row, resolver or {},
                                              cusip_resolver or {}, fold)
        rows.append({"account_id": str(row.get("account_id")),
                     "instrument_key": key,
                     "wash_date": wash_date.loc[idx],
                     "quantity": row.get("quantity"),
                     "transaction_type": str(row.get("transaction_type")),
                     "source_row": idx})
    return pd.DataFrame(rows, columns=_ACQ_COLUMNS)


def replacement_buys(acquisitions: pd.DataFrame, instrument_key: str,
                     center_date, *, window_days: int = WINDOW_DAYS,
                     sides: str = "both") -> pd.DataFrame:
    """Acquisitions of `instrument_key` within the wash window of
    `center_date`, ANY account. sides="backward" looks only at
    [center-window, center] — the scanner's view judging a sale placed
    today (tomorrow's buys don't exist yet); "both" is the two-sided
    historical window wash_check uses."""
    if acquisitions is None or acquisitions.empty:
        return pd.DataFrame(columns=_ACQ_COLUMNS)
    center = pd.Timestamp(center_date)
    lo = center - pd.Timedelta(days=window_days)
    hi = (center if sides == "backward"
          else center + pd.Timedelta(days=window_days))
    hit = acquisitions[
        (acquisitions["instrument_key"] == instrument_key)
        & (acquisitions["wash_date"] >= lo)
        & (acquisitions["wash_date"] <= hi)]
    return hit.sort_values("wash_date").reset_index(drop=True)


CANDIDATE_COLUMNS = ["account_id", "instrument_key", "symbol",
                     "acquired_date", "term", "quantity_remaining",
                     "basis_remaining", "price", "market_value",
                     "unrealized_gl", "wash_status", "blocking_buys",
                     "window_ends"]


def scan_harvest_candidates(lots: pd.DataFrame, tx: pd.DataFrame,
                            prices: dict[str, float], *, as_of,
                            account_type_of: dict[str, str],
                            resolver: Optional[dict[str, str]] = None,
                            cusip_resolver: Optional[dict[str, str]] = None,
                            fold: Optional[dict[str, str]] = None,
                            window_days: int = WINDOW_DAYS,
                            tx_frontier: Optional[
                                date | pd.Timestamp | str] = None) -> dict:
    """Open-lot harvest candidates with the cross-account wash guard.

    {"candidates": DataFrame[CANDIDATE_COLUMNS] deepest loss first,
     "summary": {...}} — the summary carries every exclusion count so the
    report can say what it is silent on instead of reading complete.
    Candidates: taxable accounts (fail-closed), reconstructed evidence,
    priced, unrealized loss only. Blockers: ANY account, backward window.
    A lot's own opening acquisition never blocks itself (matched on its
    source_row, not a date/quantity heuristic) — the buy that created the
    lot is not a replacement purchase for it; a genuinely distinct second
    acquisition of the same instrument inside the window still blocks.

    tx_frontier (date/Timestamp/str, or None when unknown) is the
    transactions ledger's latest observed date. Month-end statements mean
    the backward wash window often reaches past it, so the summary also
    reports how much of that window is actually observed
    (window_days_total/_observed/_pct, tx_frontier) — additive only, never
    changes wash_status: a "clear" verdict must not be misread as more
    than "no blocking buy found in what we can see".
    """
    as_of = pd.Timestamp(as_of)
    window_days_total = window_days + 1
    frontier_ts = None if tx_frontier is None else pd.Timestamp(tx_frontier)
    if frontier_ts is not None and pd.isna(frontier_ts):
        frontier_ts = None
    if frontier_ts is None:
        window_days_observed = None
        window_observed_pct = None
        tx_frontier_str = None
    else:
        window_start = as_of - pd.Timedelta(days=window_days)
        raw_observed = (frontier_ts - window_start).days + 1
        window_days_observed = min(max(raw_observed, 0), window_days_total)
        window_observed_pct = round(
            window_days_observed / window_days_total * 100, 1)
        tx_frontier_str = str(frontier_ts.date())
    summary = {"as_of": str(as_of.date()), "window_days": window_days,
               "lots_seen": 0 if lots is None else int(len(lots)),
               "excluded_non_taxable_accounts": 0,
               "excluded_ira_accounts": 0,
               "excluded_unknown_accounts": 0,
               "excluded_printed_evidence": 0,
               "excluded_unpriced": 0, "excluded_no_basis": 0,
               "excluded_gain_or_flat": 0,
               "excluded_no_shares_remaining": 0,
               "candidates": 0, "blocked": 0, "ira_blocked": 0,
               "total_unrealized_loss": 0.0,
               "window_days_total": window_days_total,
               "window_days_observed": window_days_observed,
               "window_observed_pct": window_observed_pct,
               "tx_frontier": tx_frontier_str}
    empty = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if lots is None or lots.empty:
        return {"candidates": empty, "summary": summary}
    acq = keyed_acquisitions(tx, resolver, cusip_resolver, fold)
    out = []
    for _, lot in lots.iterrows():
        acct = str(lot.get("account_id"))
        taxable = taxable_of(account_type_of.get(acct))
        if taxable is not True:
            # proven IRA vs cannot-prove-taxable are different facts (the
            # next slice's honesty strip needs the IRA count on its own);
            # excluded_non_taxable_accounts stays as their combined total
            summary["excluded_non_taxable_accounts"] += 1
            if taxable is False:
                summary["excluded_ira_accounts"] += 1
            else:
                summary["excluded_unknown_accounts"] += 1
            continue
        if str(lot.get("basis_evidence")) != "reconstructed":
            summary["excluded_printed_evidence"] += 1
            continue
        qty = pd.to_numeric(lot.get("quantity_remaining"), errors="coerce")
        if pd.isna(qty) or float(qty) <= 0:
            # "no shares left to sell" is a different fact from "at or
            # above cost" (excluded_gain_or_flat, below) — a lot must
            # never be counted as a foregone gain when it simply has
            # nothing left
            summary["excluded_no_shares_remaining"] += 1
            continue
        qty = float(qty)
        raw_sym = lot.get("symbol")
        sym = str(raw_sym).strip() if pd.notna(raw_sym) else ""
        sym = sym or str(lot.get("instrument_key"))
        price = prices.get(sym)
        if price is None:
            summary["excluded_unpriced"] += 1
            continue
        basis = pd.to_numeric(lot.get("basis_remaining"), errors="coerce")
        if pd.isna(basis):
            summary["excluded_no_basis"] += 1
            continue
        basis = float(basis)
        mv = qty * float(price)
        unrl = mv - basis
        if unrl >= 0:
            summary["excluded_gain_or_flat"] += 1
            continue
        acquired = pd.to_datetime(lot.get("acquired_date"), errors="coerce")
        key = str(lot.get("instrument_key"))
        hits = replacement_buys(acq, key, as_of, window_days=window_days,
                                sides="backward")
        # the buy that OPENED this lot is the acquisition of the shares
        # being sold, never a replacement for itself; exclude it from its
        # own blocking set by transactions row identity (source_row may
        # be NaN on a synthesized opening lot -- nothing to exclude then)
        own_row = pd.to_numeric(lot.get("source_row"), errors="coerce")
        if pd.notna(own_row) and not hits.empty:
            hits = hits[pd.to_numeric(hits["source_row"], errors="coerce")
                       != own_row]
        blocking = [
            {"account_id": str(h["account_id"]),
             "date": str(pd.Timestamp(h["wash_date"]).date()),
             "quantity": (None if pd.isna(h["quantity"])
                          else float(h["quantity"])),
             "transaction_type": h["transaction_type"],
             # unknown-type accounts block like taxable ones; only a
             # PROVEN IRA earns the permanent-disallowance flag
             "is_ira": taxable_of(
                 account_type_of.get(str(h["account_id"]))) is False}
            for _, h in hits.iterrows()]
        window_ends = ((max(pd.Timestamp(b["date"]) for b in blocking)
                        + pd.Timedelta(days=window_days + 1))
                       if blocking
                       else as_of + pd.Timedelta(days=window_days + 1))
        out.append({"account_id": acct, "instrument_key": key, "symbol": sym,
                    "acquired_date": (None if pd.isna(acquired)
                                      else str(acquired.date())),
                    "term": classify_term(acquired, as_of),
                    "quantity_remaining": qty, "basis_remaining": basis,
                    "price": float(price), "market_value": mv,
                    "unrealized_gl": unrl,
                    "wash_status": "blocked" if blocking else "clear",
                    "blocking_buys": blocking,
                    "window_ends": str(window_ends.date())})
    cand = pd.DataFrame(out, columns=CANDIDATE_COLUMNS)
    if not cand.empty:
        cand = cand.sort_values("unrealized_gl").reset_index(drop=True)
        summary["candidates"] = int(len(cand))
        summary["blocked"] = int((cand["wash_status"] == "blocked").sum())
        summary["ira_blocked"] = int(sum(
            any(b["is_ira"] for b in bl) for bl in cand["blocking_buys"]))
        summary["total_unrealized_loss"] = round(
            float(cand["unrealized_gl"].sum()), 2)
    return {"candidates": cand, "summary": summary}


WASH_CHECK_COLUMNS = ["account_id", "instrument_key", "close_date",
                      "realized_gl", "printed_wash", "detector_wash",
                      "cross_account", "bucket", "source_row"]


def wash_check(realizations: pd.DataFrame, transactions: pd.DataFrame,
               resolver: Optional[dict[str, str]] = None,
               cusip_resolver: Optional[dict[str, str]] = None,
               fold: Optional[dict[str, str]] = None,
               *, window_days: int = WINDOW_DAYS
               ) -> Optional[pd.DataFrame]:
    """Judge the detector against Harbor's printed W flag, one row per
    historical Harbor loss sell (realizations reference their sell row via
    source_row — the relief_check join). None means the tax_flag column
    itself is absent (a pre-S3 book: the report says so instead of scoring
    air) — an empty-but-columned book (nothing to judge) returns an empty
    frame instead, never collapsed onto the same None.

    Buckets: agree_wash / agree_clean / broker_only (the falsifier — broker
    printed W, detector found no window acquisition; every one is listed on
    the report) / detector_only (expected nonzero, NOT a defect: the
    detector is deliberately over-inclusive — it lists any window
    acquisition, while the broker nets share quantities and sees only its
    own accounts). `cross_account` marks whether any matched acquisition
    sat in a DIFFERENT account from the sell — the cross-account slice of
    detector_only is exactly what no single broker statement can see; the
    rest is same-account, a case the broker saw and still didn't flag.

    Denominator: Harbor rows only (spec §5.3) — every other broker's loss
    sells (Alpine prints no wash flag) sit outside the judged universe.
    That exclusion count, plus the judged sells whose forward half-window
    (close_date + window_days) runs past the transactions frontier — an
    `agree_clean` verdict there is under-informed, since a wash-triggering
    buy after the frontier would be invisible — ride on the returned
    frame's `.attrs` (`excluded_other_broker`, `forward_unobserved`,
    `tx_frontier`, `broker_frontiers`) rather than a column: all describe
    the whole judged run, not any one row. The frontier itself is derived
    the same way `main()` derives it: trade_date falling back to
    settlement_date, max over the WHOLE transactions frame (not just Harbor
    rows) — i.e. the single LATEST-reporting broker's own frontier. A
    replacement buy can sit in any broker's book, so the true limit on
    "could a buy be invisible" is the EARLIEST broker's frontier, not this
    max: `forward_unobserved` is therefore a LOWER bound whenever brokers
    disagree. `broker_frontiers` (broker -> that broker's own latest
    observed date) is exactly that per-broker breakdown, for a caller that
    wants to disclose or size the gap.
    """
    if transactions is None or "tax_flag" not in transactions.columns:
        return None
    dates = tx_dates(transactions)
    frontier = dates.max()
    tx_frontier_str = None if pd.isna(frontier) else str(frontier.date())
    per_broker = dates.groupby(transactions["broker"].astype(str)).max()
    broker_frontiers = {b: str(d.date()) for b, d in per_broker.items()
                        if pd.notna(d)}

    def _attach(frame: pd.DataFrame, *, excluded_other_broker: int = 0,
               forward_unobserved: int = 0) -> pd.DataFrame:
        frame.attrs["excluded_other_broker"] = excluded_other_broker
        frame.attrs["forward_unobserved"] = forward_unobserved
        frame.attrs["tx_frontier"] = tx_frontier_str
        frame.attrs["broker_frontiers"] = broker_frontiers
        return frame

    empty = pd.DataFrame(columns=WASH_CHECK_COLUMNS)
    if realizations is None or realizations.empty:
        return _attach(empty)
    sold = realizations[
        (realizations["close_reason"] == "sell")
        & (pd.to_numeric(realizations["realized_gl"], errors="coerce") < 0)]
    if sold.empty:
        return _attach(empty)
    acq = keyed_acquisitions(transactions, resolver, cusip_resolver, fold)
    broker = transactions["broker"].astype(str)
    flags = transactions["tax_flag"]
    rows = []
    excluded_other_broker = 0
    forward_unobserved = 0
    for src, grp in sold.groupby("source_row"):
        if src not in transactions.index:
            continue
        if broker.loc[src] != "harbor":
            excluded_other_broker += 1
            continue
        close = pd.to_datetime(grp["close_date"].iloc[0], errors="coerce")
        key = str(grp["instrument_key"].iloc[0])
        flag = flags.loc[src]
        printed = isinstance(flag, str) and "W" in flag
        hits = replacement_buys(acq, key, close, window_days=window_days,
                                sides="both")
        detector = bool(len(hits)) and pd.notna(close)
        own_account = str(grp["account_id"].iloc[0])
        cross_account = bool(len(hits)) and any(
            str(a) != own_account for a in hits["account_id"])
        bucket = ("agree_wash" if printed and detector else
                  "agree_clean" if not printed and not detector else
                  "broker_only" if printed else "detector_only")
        if (pd.isna(close) or pd.isna(frontier)
                or close + pd.Timedelta(days=window_days) > frontier):
            forward_unobserved += 1
        rows.append({"account_id": grp["account_id"].iloc[0],
                     "instrument_key": key,
                     "close_date": (None if pd.isna(close)
                                    else str(close.date())),
                     "realized_gl": round(float(pd.to_numeric(
                         grp["realized_gl"], errors="coerce").sum()), 2),
                     "printed_wash": printed, "detector_wash": detector,
                     "cross_account": cross_account,
                     "bucket": bucket, "source_row": src})
    return _attach(pd.DataFrame(rows, columns=WASH_CHECK_COLUMNS),
                   excluded_other_broker=excluded_other_broker,
                   forward_unobserved=forward_unobserved)


def _data_dir() -> Path:
    return Path(os.environ.get("APP_DATA_DIR")
                or Path(__file__).resolve().parents[1] / "data")


def _lots_freshness_warning(meta: Optional[dict], *,
                            tx_rows: int) -> Optional[str]:
    """None when data/lots.csv's `source_row` values can still be trusted
    to index the SAME rows in the currently loaded transactions.csv; a
    printed warning string otherwise.

    scan_harvest_candidates excludes a lot's own opening purchase from its
    blocking set by matching `lot["source_row"]` (baked into lots.csv at
    build time) against transactions.csv's own positional row index. But
    `parsers/combine_txns.py` re-sorts by (settlement_date, broker,
    account_id) and `reset_index(drop=True)`s on every rebuild, so a stale
    lots.csv's source_row values can silently point at a DIFFERENT row
    once transactions.csv is rebuilt — excluding the wrong buy and turning
    a genuine replacement purchase into a false "clear".

    This is a coarse, row-COUNT check only — the same corner
    `terminal/tax_service.py`'s own `freshness()` cuts (a same-size
    re-ingest reads fresh even though rows moved is a known gap, not a
    claim of exactness). `parsers/` must never import `terminal/`, so this
    is a standalone copy of just that check's transactions-row half.
    """
    if meta is None:
        return ("data/lots_meta.json not found — the source_row coupling "
                "between data/lots.csv and the currently loaded "
                "transactions.csv is unverified (a stale lots.csv could "
                "silently misjoin). Refresh with "
                "`py parsers/build_lots.py --write` to generate it.")
    recorded = (meta.get("inputs") or {}).get("transactions_rows")
    if recorded is None:
        return ("data/lots_meta.json has no recorded transactions row "
                "count — the source_row coupling between data/lots.csv "
                "and the currently loaded transactions.csv is unverified. "
                "Refresh with `py parsers/build_lots.py --write`.")
    if recorded != tx_rows:
        return (f"data/lots.csv looks STALE: it was built from "
                f"{recorded} transactions row(s), but transactions.csv "
                f"now has {tx_rows}. combine_txns.py re-sorts and resets "
                "row positions on every rebuild, so lots.csv's source_row "
                "values may no longer point at the same transactions "
                "rows — a wash-check 'clear' verdict here could be "
                "silently wrong. Refresh with "
                "`py parsers/build_lots.py --write`.")
    return None


def _fmt_blocker(b: dict) -> str:
    """One blocking-buy line: account, date, its OWN quantity (a
    fractional-share reinvestment must not veto a large harvest with no
    visible scale), type, and the IRA marker."""
    qty = b.get("quantity")
    qty_str = "?" if qty is None else f"{qty:.4f}"
    ira = " [IRA]" if b["is_ira"] else ""
    return (f"{b['account_id']} {b['date']} qty {qty_str} "
            f"{b['transaction_type']}{ira}")


def _print_report(result: dict, tx_frontier) -> None:
    s = result["summary"]
    cand = result["candidates"]
    print("== Harvest candidates ==")
    print(f"as of {s['as_of']}  window +/-{s['window_days']}d  "
          f"lots seen: {s['lots_seen']}")
    print(f"candidates: {s['candidates']} (blocked {s['blocked']}, "
          f"IRA-blocked {s['ira_blocked']})  "
          f"total unrealized loss: {s['total_unrealized_loss']:+.2f}")
    print(f"excluded — printed-evidence lots: "
          f"{s['excluded_printed_evidence']}; "
          f"IRA accounts: {s['excluded_ira_accounts']}; "
          f"unknown-type accounts: {s['excluded_unknown_accounts']}; "
          f"unpriced: {s['excluded_unpriced']}; "
          f"no-basis: {s['excluded_no_basis']}; "
          f"no shares remaining: {s['excluded_no_shares_remaining']}; "
          f"at/above cost: {s['excluded_gain_or_flat']}")
    print(f"guard sees statement history through {tx_frontier} — trades "
          "after that date are invisible to the wash window.")
    if s["window_observed_pct"] is None:
        # the LEAST informed case — unknown must never read as more
        # complete than a known partial observation, so it gets the
        # caveat too, not a pass into silence
        print("window observability unknown — the transactions ledger's "
              "latest date could not be determined, so how much of the "
              "backward wash window this scan actually covers is unknown.")
        print("'clear' means no blocking buy was found in an "
              "unknown-sized slice of the window — it does not mean no "
              "blocking buy exists.")
    elif s["window_observed_pct"] < 100:
        unobserved = s["window_days_total"] - s["window_days_observed"]
        print(f"window observed: {s['window_days_observed']} of "
              f"{s['window_days_total']} backward-window days "
              f"({s['window_observed_pct']:.1f}%) — the remaining "
              f"{unobserved} days are past the statement frontier, "
              "invisible to this scan.")
        print("'clear' means no blocking buy in the observed part of the "
              "window — it does not mean no blocking buy exists.")
    if s["ira_blocked"]:
        print("[IRA] = replacement bought inside an IRA: that loss is "
              "permanently disallowed, not deferred onto any replacement "
              "lot's basis.")
    for _, r in cand.iterrows():
        blocks = "; ".join(
            _fmt_blocker(b) for b in r["blocking_buys"]) or "-"
        # status-keyed: "clear ... window ends X" reads as "clear UNTIL
        # X" (backwards). A blocked row's block clears on window_ends; a
        # clear row stays clear only absent a rebuy before window_ends --
        # window_ends is the first SAFE day, so "before" (not "by", which
        # reads as on-or-before) is the exact word. The clear-case note
        # also drops its own leading "clear": the status column just
        # printed that word, so repeating it here would stutter.
        window_note = (f"clears {r['window_ends']}"
                       if r["wash_status"] == "blocked"
                       else f"unless rebought before {r['window_ends']}")
        print(f"  {r['account_id']} {r['instrument_key']:<8} {r['term']:<8}"
              f" qty {r['quantity_remaining']:>12.4f}  "
              f"unrl {r['unrealized_gl']:>+14.2f}  {r['wash_status']:<8}"
              f" {window_note}  blocks: {blocks}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="TLH harvest scan + wash guard (report only, no writes)")
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--as-of", default=None,
                    help="YYYY-MM-DD; default = local today")
    args = ap.parse_args(argv)
    data = _data_dir()
    lots_path = data / "lots.csv"
    if not lots_path.exists():
        print(f"no {lots_path} — run `py parsers/build_lots.py --write` on a "
              "gate-passing book first.")
        return 1
    lots = pd.read_csv(lots_path)
    tx = pd.read_csv(data / "transactions.csv")
    positions = pd.read_csv(data / "positions.csv")

    meta_path = data / "lots_meta.json"
    meta = None
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = loaded if isinstance(loaded, dict) else None
        except (ValueError, OSError, UnicodeDecodeError):
            meta = None
    freshness_warning = _lots_freshness_warning(meta, tx_rows=len(tx))
    if freshness_warning:
        print(f"WARNING: {freshness_warning}\n")

    prices_path = data / "prices_latest.csv"
    prices_latest = (pd.read_csv(prices_path) if prices_path.exists()
                     else pd.DataFrame())
    fold, _splits = load_corporate_identity()
    resolver, cusip_resolver = build_key_resolvers(tx, positions, fold)
    frontier = tx_frontier_of(tx)
    # display-only string ("unknown" or a date) for the report header --
    # deliberately NOT named tx_frontier: that name belongs to the kwarg
    # below, which wants a Timestamp/None. `tx_frontier=tx_frontier`
    # would pass the string "unknown" into pd.Timestamp(...) and crash.
    tx_frontier_display = ("unknown" if frontier is None
                           else str(frontier.date()))
    as_of = args.as_of or date.today().isoformat()
    result = scan_harvest_candidates(
        lots, tx, price_map(prices_latest), as_of=as_of,
        account_type_of=account_types(positions), resolver=resolver,
        cusip_resolver=cusip_resolver, fold=fold,
        window_days=args.window_days, tx_frontier=frontier)
    _print_report(result, tx_frontier_display)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
