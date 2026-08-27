# terminal/tax_service.py
"""Pure data seam for the MERIDIAN Terminal "Tax" tab (tax S2b + S3b).

First consumer surface of the gated lot ledger (data/lots.csv, slice 10)
and its build metadata (data/lots_meta.json, slice 2a). Two views over one
payload — **Open lots** and **Harvest** (TLH candidates + the cross-account
wash guard, S3b); one fetch, client toggle. Realized YTD (round 2) rides
the same payload: the build-time ``realized_ytd`` block the meta carries
is aggregated here over the taxable, broker-scoped accounts — this module
never recomputes a realized figure.

All harvest math is `parsers.tax_scanner.scan_harvest_candidates`; nothing
here computes a wash verdict. What this module owns is the honesty around
it: a `clear` chip means "no blocking buy in the OBSERVED part of the
window", and the window is routinely mostly unobserved because the ledger
is built from month-end statements.

Carry evidence, consumers decide (locked campaign decision): every lot row
ships its ``basis_evidence`` + ``band`` verbatim, and the header strip
states the gate numbers honestly — the ledger is accurate on what it
reconstructs and silent on the rest. Never present it as "lots solved".

Taxable accounts only (locked: IRAs in the ledger, OUT of tax views):
``account_type`` derives latest-wins from positions, any type containing
"ira" is excluded, and an account whose type cannot be derived at all is
excluded FAIL-CLOSED and counted — a tax view must never include an
account it cannot prove taxable.

The only computation here is ``quantity_remaining * price -
basis_remaining`` per lot and an LT/ST label from ``acquired_date``.
Pricing is two rungs, each shared with an existing surface rather than
invented here: (1) the same ``prices_latest`` frame the Holdings tab
marks to market with, looked up by symbol falling back to
``instrument_key`` — the scanner's own rule (a crosswalk-resolved lot
carries its ticker in the KEY with an empty symbol column; Open lots
and Harvest must price identically); (2) the latest statement month's
own marks (``market_value / quantity`` per instrument from
positions.csv) for instruments the live feed does not carry — the
treasury ladder's cusip-keyed notes — flagged ``price_source:
"statement"`` with their as-of month, the Holdings tab's ``price_stmt``
semantics. A lot neither rung can price carries null market value /
unrealized — never a fabricated 0 — and the totals say so instead of
silently pairing an all-lots basis with a priced-only market value.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from parsers import asset_reclass
from parsers.asset_reclass import reclass_asset
from parsers.income_analytics import DISTRIBUTION_TYPES, income_timeseries
from parsers.lot_engine import (build_key_resolvers, classify_term,
                                days_to_long_term, load_corporate_identity)
from parsers.tax_estimate import (TAX_YEAR, estimate_year_tax,
                                  is_treasury_income)
from parsers.tax_scanner import (
    WINDOW_DAYS, account_types, keyed_acquisitions, price_map,
    replacement_buys, scan_harvest_candidates, taxable_of,
    tx_frontier_of,
)
from terminal import holdings_service as hs

# The tab owns NO term rule of its own. It had one — a `days > 365` count —
# which disagreed with the ledger's calendar-anniversary rule on exactly the
# dates a leap day falls inside the span: 366 days elapse by the first
# anniversary, so the count said "long" while the IRS rule ("more than one
# year") says a sale ON the anniversary is still short. Two views of one lot
# must never disagree about its term, so `classify_term` is the single rule
# and this is a re-export, not a reimplementation. Dates are LOCAL (the DTE
# lesson: UTC "today" is tomorrow in the local evening). `price_map` is
# likewise the scanner's, so Open lots and Harvest mark at one price.
term_of = classify_term

LOT_FIELDS = ("lot_id", "account_id", "account_label", "symbol", "instrument_key",
              "type", "is_tlh", "origin", "acquired_date", "term",
              "quantity_remaining", "basis_remaining", "price",
              "price_source", "price_asof", "market_value", "unrealized_gl",
              "basis_evidence", "band")


def _tlh_account_id() -> str:
    """The tax-loss-harvest account id (config_local via holdings_service;
    "" when unset). The Type FILTER carves that account's lots into their
    own bucket (TK, round 3, filter-only): the instrument-truth ``type``
    field stays untouched — a TLH AAPL lot is still a stock — and the
    account fact ships as the ``is_tlh`` BOOLEAN, so no account
    identifier ever reaches the payload or the committed golden."""
    return str(getattr(getattr(hs, "cfg", None), "TLH_ACCOUNT_ID", "")
               or "")


def stmt_price_map(raw_positions: pd.DataFrame | None
                   ) -> tuple[dict[str, float], str]:
    """(key -> unit price, as-of "YYYY-MM") from the LATEST positions
    month — the statement's own marks, for instruments the live feed
    does not carry (cusip-keyed treasuries; anything outside the fetch
    universe).

    Keyed by BOTH the row's symbol and its cusip, so symbol-keyed and
    cusip-keyed lots resolve through one map. Unit price is
    sum(market_value) / sum(quantity) per key over that month — some
    brokers print one positions row per LOT, so per-key aggregation is
    load-bearing, not cosmetic. A key whose quantity is ~0 or whose
    market value is missing yields no price — never a fabricated one.
    """
    if raw_positions is None or raw_positions.empty \
            or "statement_date" not in raw_positions.columns:
        return {}, ""
    dates = pd.to_datetime(raw_positions["statement_date"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return {}, ""
    month = latest.strftime("%Y-%m")
    rows = raw_positions[dates == latest]
    acc: dict[str, list[float]] = {}
    for _, r in rows.iterrows():
        qty = pd.to_numeric(r.get("quantity"), errors="coerce")
        mv = pd.to_numeric(r.get("market_value"), errors="coerce")
        if pd.isna(qty) or pd.isna(mv):
            continue
        for key in (str(r.get("symbol") or "").strip(),
                    str(r.get("cusip") or "").strip()):
            if key:
                tot = acc.setdefault(key, [0.0, 0.0])
                tot[0] += float(mv)
                tot[1] += float(qty)
    out = {k: mv_sum / qty_sum for k, (mv_sum, qty_sum) in acc.items()
           if abs(qty_sum) > 1e-9}
    return out, month


# asset_class -> instrument type, AFTER the same reclass the Holdings tab
# applies. positions.csv carries the broker's RAW tag, and brokers misfile
# ETFs (Fidelity's 2026 format lists SPY/SGOV under Common Stock — PR #131),
# so raw asset_class alone types a real book's ETF lots "stock"/"other"
# (2026-07-31 live smoke: the ETFs view was empty on 678 real lots). The
# corrections are all existing, derived sources — parsers.asset_reclass
# (its account-scoped tax_loss_harvesting display rule disabled via a
# sentinel: type is an instrument fact, not an account fact) plus the
# config ETF sets (ETF_TICKER_CLASS / FIDELITY_CORE_ETF_SYMBOLS) already
# bound by holdings_service. No ticker lists in code (the Sage rule).
_CLASS_TYPE = {"equity_stock": "stock", "equity_etf": "etf"}
_ETF_CLASSES = frozenset({"equity_etf", "gold"})
_NEVER_ACCOUNT = "\x00type-map"  # matches no real account id


def _type_map(raw_positions: pd.DataFrame | None, *,
              etf_class: dict[str, str] | None = None,
              core_etf_symbols: frozenset[str] | set[str] = frozenset()
              ) -> dict[str, str]:
    """key -> "stock" | "etf" | "other" from positions' asset_class run
    through ``reclass_asset``, keyed by BOTH symbol and cusip
    (stmt_price_map's dual keying) so symbol-keyed and cusip-keyed lots
    classify through one map.

    Latest-wins PER KEY over the WHOLE positions history — each key takes
    its asset_class at its own max statement_date, not the newest month's
    row set: a stranded instrument absent from the newest statement must
    still classify from its last-seen month. Consumers look up the row's
    symbol falling back to instrument_key — the SAME rule pricing uses,
    so two views of one lot can never classify differently — and default
    "other" for a key positions has never carried.

    "etf" means: reclassed class in ``_ETF_CLASSES`` (equity_etf, or the
    commodity bucket — GLD-family tickers are ETPs), OR the ticker sits in
    the user's own ETF config (an ``etf_class`` entry mapping it anywhere
    but back to equity_stock — e.g. a bond ETF mapped fixed_income for
    risk bucketing is still an ETF to the tax view — or membership in
    ``core_etf_symbols``). Config maps default empty, so tests and CI
    stay config_local-independent.
    """
    etf_class = etf_class or {}
    core = {str(s).upper() for s in (core_etf_symbols or ())}
    if raw_positions is None or raw_positions.empty \
            or "asset_class" not in raw_positions.columns:
        return {}
    # Option legs are not type contenders for their key: a covered call's
    # positions row carries the underlying's ticker as its symbol, so at a
    # same-date tie the winner would otherwise be positions.csv row order
    # (an ingest artifact). Same hazard asset_reclass documents around its
    # equity_stock-only override guard.
    cls_str = raw_positions["asset_class"].astype(str)
    is_option_row = cls_str.str.startswith("option_")
    if "symbol" in raw_positions.columns:
        sym_str = raw_positions["symbol"].astype(str).str.strip().str.upper()
        is_option_row = is_option_row | sym_str.str.match(
            asset_reclass._DISPLAY_OPT_RE)
    base = raw_positions.loc[~is_option_row]
    if base.empty:
        return {}
    dates = pd.to_datetime(base.get("statement_date"), errors="coerce")
    parts = []
    for col in ("symbol", "cusip"):
        if col not in base.columns:
            continue
        keys = base[col]
        part = pd.DataFrame({"_key": keys.astype(str).str.strip(),
                             "_cls": base["asset_class"],
                             "_sdate": dates})
        parts.append(part[keys.notna() & (part["_key"] != "")])
    if not parts:
        return {}
    ordered = pd.concat(parts, ignore_index=True)
    # Same-date ties resolve by class preference (equity_etf beats
    # equity_stock beats the rest), never by row order: ascending rank
    # sorts the preferred class last at its date, and the last write per
    # key wins.
    ordered["_pref"] = ordered["_cls"].map(
        {"equity_etf": 2, "equity_stock": 1}).fillna(0)
    ordered = ordered.sort_values(["_sdate", "_pref"],
                                  na_position="first", kind="stable")

    def type_of(key: str, cls) -> str:
        raw = str(cls) if pd.notna(cls) else ""
        final = reclass_asset(_NEVER_ACCOUNT, key, raw,
                              tlh_account_id="\x00never",
                              etf_class=etf_class)
        key_u = key.upper()
        if (final in _ETF_CLASSES or key_u in core
                or etf_class.get(key_u) not in (None, "equity_stock")):
            return "etf"
        return _CLASS_TYPE.get(final, "other")

    # dict built in ascending date order: the last write per key wins,
    # which IS the key's own latest row (ties -> later positions row)
    return {str(k): type_of(str(k), cls)
            for k, cls in zip(ordered["_key"], ordered["_cls"])}


def freshness(meta: dict | None, *, tx_rows: int,
              positions_max_month: str) -> tuple[bool, str | None]:
    """(stale, reason) — the content-based staleness contract of slice 2a.

    Compares the live book's input frontier against what the build saw.
    Row-count equality is deliberately coarse (a same-size re-ingest reads
    fresh); the positions frontier catches the close-shaped changes."""
    if not meta:
        return True, ("no build metadata — refresh with "
                      "py parsers/build_lots.py --write")
    inputs = meta.get("inputs") or {}
    if inputs.get("positions_max_month") != positions_max_month:
        return True, (f"positions frontier moved to {positions_max_month} "
                      f"(ledger built at "
                      f"{inputs.get('positions_max_month')}) — the close "
                      f"refreshes it, or run build_lots --write")
    if inputs.get("transactions_rows") != tx_rows:
        return True, (f"transactions changed ({tx_rows} rows vs "
                      f"{inputs.get('transactions_rows')} at build) — the "
                      f"close refreshes it, or run build_lots --write")
    return False, None


def _round2(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _sum_or_none(values: list) -> float | None:
    """Sum of the non-null values, or None when there are none — a group
    with zero priced lots must never read as $0.00 (null-never-zero)."""
    vals = [v for v in values if v is not None]
    return _round2(sum(vals)) if vals else None


HARVEST_FIELDS = ("account_id", "account_label", "symbol", "instrument_key",
                  "type", "is_tlh", "acquired_date", "term",
                  "quantity_remaining", "basis_remaining", "price",
                  "market_value", "unrealized_gl", "wash_status",
                  "blocking_buys", "is_ira_blocked", "window_ends")


def _harvest(lots, raw_tx, raw_positions, prices, *, asof, types, labels,
             keep_account, stale: bool, tmap: dict[str, str],
             tlh_id: str = "") -> dict:
    """The `harvest` key: TLH candidates + the cross-account wash guard.

    All the math is `parsers.tax_scanner.scan_harvest_candidates`; this is
    shaping only. Two deliberate scoping rules, both mirroring what the
    open-lots side above already does:

    * the scan sees EVERY lot, so its exclusion counts (IRA, unknown-type,
      printed-evidence, unpriced, gain) stay whole-book tax facts rather
      than shrinking with the broker picker — the same reason open lots
      counts its exclusions before narrowing;
    * the resulting candidates are then narrowed by that picker, and the
      counts that describe the RENDERED rows are recomputed over the
      narrowed set, so the strip can never describe rows the table isn't
      showing.

    Blockers are never narrowed: a replacement buy in any account at any
    broker blocks, which is the whole point of a cross-account guard.
    """
    fold, _splits = load_corporate_identity()
    resolver, cusip_resolver = build_key_resolvers(raw_tx, raw_positions,
                                                   fold)
    result = scan_harvest_candidates(
        lots, raw_tx, prices, as_of=asof, account_type_of=types,
        resolver=resolver, cusip_resolver=cusip_resolver, fold=fold,
        tx_frontier=tx_frontier_of(raw_tx))
    summary = dict(result["summary"])

    rows: list[dict] = []
    for _, c in result["candidates"].iterrows():
        acct = str(c["account_id"])
        if not keep_account(acct):
            continue
        blocking = list(c["blocking_buys"])
        sym = str(c["symbol"]) if pd.notna(c["symbol"]) else ""
        rows.append({
            "account_id": acct,
            "account_label": str(labels.get(acct, acct)),
            "symbol": str(c["symbol"]),
            "instrument_key": str(c["instrument_key"]),
            # the open-lots loop's identical sym-or-ikey lookup — two
            # views of one lot must never classify differently
            "type": tmap.get(sym or str(c["instrument_key"]), "other"),
            "is_tlh": bool(tlh_id) and acct == tlh_id,
            # A synthesized opening lot has NO printed acquisition date
            # (the slice-1 SHORTFALL rule never guesses one). The scanner
            # emits Python None, but building a DataFrame from those dicts
            # turns None into NaN in this column, and a NaN reaching
            # `json.dumps(..., allow_nan=False)` is a 500 on the whole tab
            # — Open lots included. Coerce back at the JSON boundary.
            "acquired_date": (None if pd.isna(c["acquired_date"])
                              else str(c["acquired_date"])),
            "term": str(c["term"]),
            "quantity_remaining": float(c["quantity_remaining"]),
            "basis_remaining": _round2(float(c["basis_remaining"])),
            "price": _round2(float(c["price"])),
            "market_value": _round2(float(c["market_value"])),
            "unrealized_gl": _round2(float(c["unrealized_gl"])),
            "wash_status": str(c["wash_status"]),
            "blocking_buys": blocking,
            "is_ira_blocked": any(b["is_ira"] for b in blocking),
            "window_ends": str(c["window_ends"]),
        })
    summary["candidates"] = len(rows)
    summary["blocked"] = sum(1 for r in rows if r["wash_status"] == "blocked")
    summary["ira_blocked"] = sum(1 for r in rows if r["is_ira_blocked"])
    summary["total_unrealized_loss"] = _sum_or_none(
        [r["unrealized_gl"] for r in rows]) or 0.0

    semantics = {
        "window_days": summary.get("window_days"),
        "as_of": summary.get("as_of"),
        "tx_frontier": summary.get("tx_frontier"),
        "window_days_total": summary.get("window_days_total"),
        "window_days_observed": summary.get("window_days_observed"),
        "window_observed_pct": summary.get("window_observed_pct"),
        # A "clear" verdict is bounded by what the ledger can SEE. Month-end
        # statements leave the frontier behind today, so most of a backward
        # window is routinely unobserved (spec Update item 2) — never let
        # `clear` be read as "safe to harvest".
        "clear_means": ("no blocking buy in the OBSERVED part of the "
                        "window — later trades are not in the ledger yet"),
        # A stale ledger is worse than a stale display here: lots.csv's
        # source_row values index transactions.csv POSITIONALLY, and
        # combine_txns re-sorts and resets those positions on every
        # rebuild, so a stale pair can exclude the wrong buy and turn a
        # genuine replacement purchase into a false "clear".
        "stale_note": (("the ledger is stale, so a lot's own opening buy may "
                        "be matched against the wrong transactions row — a "
                        "'clear' here can be wrong until it is rebuilt")
                       if stale else None),
    }
    return {"candidates": rows, "summary": summary, "semantics": semantics}


def _harvest_or_unavailable(*args, **kwargs) -> dict:
    """`_harvest`, degraded to a NAMED failure instead of a 500.

    Open lots reads two columns of transactions.csv; the harvest scan reads
    the whole frame and joins it to the ledger, so a malformed or
    half-written transactions.csv can raise here for shapes the tab used to
    tolerate. A tax view that renders nothing is worse than one that
    renders Open lots and says why Harvest is missing — but the reason is
    always carried, never swallowed: no silent empty candidate list, which
    would read as "nothing to harvest".
    """
    try:
        return _harvest(*args, **kwargs)
    except (KeyError, ValueError, TypeError, AttributeError,
            IndexError) as exc:
        return {"candidates": [], "summary": {}, "semantics": {},
                "unavailable": (f"the harvest scan could not read this "
                                f"book's transactions "
                                f"({type(exc).__name__}: {exc}) — open lots "
                                f"above are unaffected")}


# Cusip-shaped instrument keys (bills, notes) carry 5+ digit runs; they are
# not tickers and would trip the AI scrub's account-mask guard.
_TICKER_DIGIT_RUN_RE = re.compile(r"\d{5,}")

# A ticker-shaped instrument key: the lot engine's key falls back symbol ->
# cusip -> description slug, and only ticker-shaped keys may be named to the
# chat (a cash-sweep slug or a cusip is never a ticker).
_TICKER_SHAPE_RE = re.compile(r"^[A-Z][A-Z0-9./\-]{0,9}$")


def _ticker_like(key) -> bool:
    return (isinstance(key, str) and bool(_TICKER_SHAPE_RE.match(key))
            and not _TICKER_DIGIT_RUN_RE.search(key))


def _scrub_safe_label(acct: str, labels: dict[str, str]) -> str:
    label = str(labels.get(acct) or acct)
    return "unlabeled account" if _TICKER_DIGIT_RUN_RE.search(label) else label


def _omit_empty(row: dict) -> dict:
    """Compaction (#380 final review, −15.5 KB on the real book): drop
    None / 0 / False / empty-list values from an emitted row — absent
    means zero or none, and the emitting block's note says so."""
    return {k: v for k, v in row.items()
            if not (v is None or v is False or v == 0 or v == [])}


def wash_calendar(tx: pd.DataFrame, *, asof: date, labels: dict[str, str],
                  resolver: dict[str, str], cusip_resolver: dict[str, str],
                  fold: dict[str, str] | None,
                  window_days: int = WINDOW_DAYS,
                  sleeve_accounts: frozenset = frozenset()) -> dict:
    """Per-ticker wash-window facts from the transaction ledger ALONE — no
    lots, so it stays fresh while the ledger waits for a close.

    For every instrument with a buy/reinvestment or a sell inside the
    trailing `window_days` ending at `asof`: the last acquisition and the
    last day a loss sale is still inside `replacement_buys`' backward window
    of it (`wash_if_sold_before` = last acquisition + window_days); the last
    sell and the symmetric forward window (`wash_if_rebought_before` = last
    sell + window_days); counts; the account labels involved; and the number
    of in-window sells the broker printed a wash flag on (`tax_flag == "W"`
    — interim rows carry no flag). Option rows never enter (the scanner's
    own `is_option_row` exclusion), and only ticker-shaped keys are named:
    a cusip or an unresolved description slug (a cash sweep, an unresolved
    reinvestment) is dropped and counted in `unkeyed_rows_omitted`.
    Whether a past sale realized a loss is NOT stated — that is the lot
    ledger's call. Dollar-free by construction: tickers, dates, counts,
    labels."""
    if tx is None:
        tx = pd.DataFrame()
    # Positional self-consistency: `flagged` is indexed by tx.index and the
    # keyed rows carry that index as `source_row`; a duplicate-labelled frame
    # (a concat without ignore_index) would make the reindex ambiguous.
    tx = tx.reset_index(drop=True)
    asof_ts = pd.Timestamp(asof).normalize()
    lo = asof_ts - pd.Timedelta(days=window_days)

    def _window(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df[(df["wash_date"] >= lo) & (df["wash_date"] <= asof_ts)]

    acq = _window(keyed_acquisitions(tx, resolver, cusip_resolver, fold))
    sells = _window(keyed_acquisitions(tx, resolver, cusip_resolver, fold,
                                       types=("sell",)))
    if "tax_flag" in getattr(tx, "columns", ()):
        flagged = tx["tax_flag"].astype(str).str.strip().str.upper().eq("W")
    else:
        flagged = pd.Series(False, index=getattr(tx, "index", []))

    def _day(ts) -> str:
        return str(pd.Timestamp(ts).date())

    tickers: list[dict] = []
    keys = sorted(set(acq["instrument_key"]) | set(sells["instrument_key"]))
    unkeyed_rows = 0
    for key in keys:
        if not _ticker_like(key):
            unkeyed_rows += int((acq["instrument_key"] == key).sum()
                                + (sells["instrument_key"] == key).sum())
            continue
        a = acq[acq["instrument_key"] == key]
        s = sells[sells["instrument_key"] == key]
        ids = set(pd.concat([a["account_id"], s["account_id"]]).astype(str))
        accounts = sorted({_scrub_safe_label(x, labels) for x in ids})
        row = {"ticker": key, "accounts": accounts,
               "acquisitions_in_window": int(len(a)),
               "last_acquisition": None, "wash_if_sold_before": None,
               "sells_in_window": int(len(s)),
               "last_sell": None, "wash_if_rebought_before": None,
               "broker_flagged_wash_sells": 0,
               # all in-window activity sits in the direct-index sleeve —
               # such names are not lot-actionable (#380 review rider)
               "sleeve_only": bool(sleeve_accounts) and ids <= sleeve_accounts}
        if len(a):
            last = a["wash_date"].max()
            row["last_acquisition"] = _day(last)
            row["wash_if_sold_before"] = _day(
                last + pd.Timedelta(days=window_days))
        if len(s):
            last = s["wash_date"].max()
            row["last_sell"] = _day(last)
            row["wash_if_rebought_before"] = _day(
                last + pd.Timedelta(days=window_days))
            row["broker_flagged_wash_sells"] = int(
                flagged.reindex(s["source_row"]).fillna(False).sum())
        tickers.append(_omit_empty(row))
    frontier = tx_frontier_of(tx)
    return {
        "as_of": str(asof_ts.date()), "window_days": int(window_days),
        "tx_frontier": None if frontier is None else _day(frontier),
        "unkeyed_rows_omitted": int(unkeyed_rows),
        "note": ("whole book — wash rules cross accounts and brokers; "
                 "trailing window of window_days ending at as_of over "
                 "the transaction rows supplied (fresh through tx_frontier). "
                 "wash_if_sold_before: a loss sale on or before this date is "
                 "washed by the latest purchase — provided shares other "
                 "than the ones being sold were acquired inside the "
                 "window; a single lot round-tripped within the window is "
                 "not washed by its own purchase. wash_if_rebought_before: "
                 "a repurchase on or before this date washes a loss "
                 "realized on the latest sale. Whether a sale realized a "
                 "loss is not stated here; broker_flagged_wash_sells "
                 "counts statement sells the broker printed a wash flag "
                 "on (interim rows carry no flags). Option trades are not "
                 "counted. Rows whose instrument has no ticker symbol "
                 "(cash sweeps, unresolved reinvestments, bills) are not "
                 "listed; unkeyed_rows_omitted counts them. Keys absent "
                 "from a row mean zero or none. sleeve_only marks names "
                 "whose in-window activity is all inside the direct-index "
                 "sleeve account."),
        "tickers": tickers,
    }


def _picker_options(frames) -> dict:
    """The global Account/Asset-class option lists every whole-book tab's
    meta carries (health/income/factor do the same) so a ?tab=tax-first
    load doesn't leave the pickers empty for the session."""
    try:
        snap_all = hs._current_snap(frames)
        acct_opts, _ = hs._account_options(snap_all)
        class_opts, _ = hs._class_options(snap_all)
        return {"accounts": acct_opts, "classes": class_opts}
    except Exception:
        return {"accounts": [], "classes": []}


def _error_view(reason: str, asof: date, options: dict | None = None) -> dict:
    meta = {"asof": asof.isoformat(), "filter": {}}
    meta.update(options or {"accounts": [], "classes": []})
    return {"kind": "error", "reason": reason, "meta": meta}


def _silent_note(meta: dict | None) -> str:
    """The honesty strip's sentence, built from the gate's own numbers —
    accuracy is claimed ONLY over the reconstructed share; the remainder is
    silence, not error."""
    gate = (meta or {}).get("gate") or {}
    acc, cov = gate.get("accuracy_pct"), gate.get("coverage_pct")
    if acc is None or cov is None:
        return ("basis accuracy unavailable — no build metadata; evidence "
                "bands per lot still apply")
    return (f"every lot is shown. Basis provenance: {acc}% accurate on "
            f"the {cov}% of joined basis the ledger reconstructs itself; "
            f"the remaining {round(100.0 - cov, 2)}% carries the broker's "
            f"own printed figures (instrument totals exact, per-lot "
            f"slicing approximate — why TLH excludes those instruments)")


# The degrade path names the fix and fabricates nothing: a pre-round-2
# meta simply has no realized block, and a fabricated 0.00 would read as
# "nothing realized this year" — a claim the build never made.
_REALIZED_UNAVAILABLE = ("realized figures appear at the next ledger build "
                         "— py parsers/build_lots.py --write (the month "
                         "close runs it automatically)")

# Meta slot -> display term. Option confirms carry their own printed ST/LT
# tag, so they fold into short/long and can never touch "unknown" — that
# bucket means "the LEDGER could not date the acquisition".
_REALIZED_FOLD = {"short": "short", "long": "long", "unknown": "unknown",
                  "options_short": "short", "options_long": "long"}

# Rank exists so sort order isn't alphabetical on the term string
# (long<short); id only tie-breaks accounts sharing a label (golden-pinned).
_TERM_RANK = {"short": 0, "long": 1, "unknown": 2}


def _realized_ytd(meta: dict | None, types: dict, keep_account,
                  labels: dict) -> dict:
    """``_realized_ytd_inner`` degraded to a NAMED failure, never a 500
    and never zeros — the harvest key's `_harvest_or_unavailable`
    contract, applied to the realized block: a structurally malformed
    lots_meta.json (a list where by_account should be, a non-numeric
    money value) must cost this tab the realized tiles WITH a reason,
    not Open lots and Harvest with a traceback.
    """
    try:
        return _realized_ytd_inner(meta, types, keep_account, labels)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        return {"unavailable":
                (f"the realized_ytd block in lots_meta.json could not be "
                 f"read ({type(exc).__name__}: {exc}) — rebuild with "
                 f"py parsers/build_lots.py --write")}


def _realized_ytd_inner(meta: dict | None, types: dict,
                        keep_account, labels: dict) -> dict:
    """``summary.realized_ytd`` — lots_meta.json's build-time by_account
    block (spec §3a) aggregated over accounts that are BOTH provably
    taxable (the open-lots loop's own fail-closed predicate: an IRA's or
    unknown-type account's realized never lands in a tax view) AND inside
    the global broker narrowing (``keep_account``).

    Ledger terms and option slots fold per ``_REALIZED_FOLD``; a term
    bucket is emitted only if it saw closes or nonzero money, so a
    never-touched term never renders as a hollow $0.00. An EMPTY
    by_account aggregates to true zeros with ``by_term`` {} — that is a
    statement about the year, NOT the degrade path (``unavailable`` is a
    statement about the build). Money re-rounds at this boundary because
    accumulating cent-clean per-slot figures reintroduces float dust;
    slot values are cent-clean by construction (net derives from rounded
    components at build), so every layer cross-foots exactly.

    Disclosure counters, never silent drops (the swallowed-error rule):
    ``options_uncovered`` (option closes whose confirm printed no usable
    figure), ``unrecognized_slots`` (slot names this fold does not know,
    or slot values that are not dicts — a newer meta read by older code),
    ``broker_unresolved`` (ledger rows the build dropped as
    provenance-unresolvable, carried from the block's own notes).

    Round 4: the fold runs per-account first and emits ``by_account``
    rows (one per (account, term) with activity) before accumulating
    into the global buckets — one emission path, so the view cannot
    disagree with the tiles.
    """
    block = (meta or {}).get("realized_ytd")
    if not isinstance(block, dict):
        return {"unavailable": _REALIZED_UNAVAILABLE}
    folded = {t: {"gains": 0.0, "losses": 0.0, "net": 0.0, "closes": 0}
              for t in ("short", "long", "unknown")}
    rows: list[dict] = []
    uncovered = 0
    unrecognized = 0
    for acct, slots in (block.get("by_account") or {}).items():
        acct = str(acct)
        if taxable_of(types.get(acct)) is not True or not keep_account(acct):
            continue
        mine = {t: {"gains": 0.0, "losses": 0.0, "net": 0.0, "closes": 0}
                for t in ("short", "long", "unknown")}
        for name, slot in (slots or {}).items():
            term = _REALIZED_FOLD.get(name)
            if term is None or not isinstance(slot, dict):
                unrecognized += 1
                continue
            b = mine[term]
            b["gains"] += float(slot.get("gains") or 0.0)
            b["losses"] += float(slot.get("losses") or 0.0)
            b["net"] += float(slot.get("net") or 0.0)
            b["closes"] += int(slot.get("closes") or 0)
            if name.startswith("options_"):
                uncovered += int(slot.get("uncovered") or 0)
        for t, b in mine.items():
            g = folded[t]
            g["gains"] += b["gains"]
            g["losses"] += b["losses"]
            g["net"] += b["net"]
            g["closes"] += b["closes"]
            if b["closes"] or b["gains"] or b["losses"] or b["net"]:
                rows.append({"account_id": acct,
                             "account_label": str(labels.get(acct, acct)),
                             "term": t,
                             "closes": int(b["closes"]),
                             "gains": _round2(b["gains"]),
                             "losses": _round2(b["losses"]),
                             "net": _round2(b["net"])})
    rows.sort(key=lambda r: (r["account_label"], _TERM_RANK[r["term"]],
                             r["account_id"]))
    by_term = {t: {"gains": _round2(b["gains"]),
                   "losses": _round2(b["losses"]),
                   "net": _round2(b["net"])}
               for t, b in folded.items()
               if b["closes"] or b["gains"] or b["losses"] or b["net"]}
    return {"year": block.get("year"),
            "gains": _round2(sum(v["gains"] for v in by_term.values())),
            "losses": _round2(sum(v["losses"] for v in by_term.values())),
            "net": _round2(sum(v["net"] for v in by_term.values())),
            "by_term": by_term,
            "by_account": rows,
            "options_in": True,
            "options_uncovered": uncovered,
            "unrecognized_slots": unrecognized,
            "broker_unresolved": int((block.get("notes") or {})
                                     .get("broker_unresolved") or 0),
            "unavailable": None}


def build_tax_view(frames, data_dir: str | Path, *,
                   asof: date | None = None,
                   broker: list[str] | None = None) -> dict:
    """The /api/tax view dict (JSON-native, allow_nan=False-clean).

    ``frames`` supplies the shared prices; lots.csv / lots_meta.json /
    the raw positions+transactions (staleness inputs) are read from
    ``data_dir`` — the same APP_DATA_DIR seam every service uses.
    ``broker`` is the GLOBAL picker's selection (["all"] or broker ids);
    it narrows to that broker's taxable accounts. Fail-closed exclusions
    (IRA / unknown type) are counted BEFORE the broker narrowing — they
    are tax-ineligibility facts, not filter state.
    """
    asof = asof or date.today()
    data = Path(data_dir)
    options = _picker_options(frames)

    lots_path = data / "lots.csv"
    if not lots_path.exists():
        return _error_view(
            "data/lots.csv not found — the lot ledger has not been built "
            "here yet. The month close writes it, or run "
            "py parsers/build_lots.py --write.", asof, options)
    lots = pd.read_csv(lots_path)

    # A corrupt/unreadable lots_meta.json must degrade like a missing one
    # (stale-flagged lots), never 500 the tab.
    meta_path = data / "lots_meta.json"
    meta, meta_unreadable = None, False
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("lots_meta.json is not an object")
        except (ValueError, OSError, UnicodeDecodeError):
            meta, meta_unreadable = None, True

    # Staleness inputs, read RAW (frames overlays interim activity, so its
    # row count would false-flag the comparison the meta writer pinned).
    # The whole transactions frame, not just the row count: the harvest
    # scan's wash guard reads it, and its `source_row` self-exclusion is
    # defined against THIS frame's positional index — the same one the
    # staleness check guards.
    try:
        raw_positions = pd.read_csv(data / "positions.csv")
        raw_tx = pd.read_csv(data / "transactions.csv")
        tx_rows = len(raw_tx)
    except FileNotFoundError as exc:
        return _error_view(
            f"the lot ledger's input files are incomplete here "
            f"({Path(str(exc.filename or exc)).name or exc}) — lots.csv "
            f"exists but its staleness inputs do not.", asof, options)
    latest = pd.to_datetime(raw_positions["statement_date"],
                            errors="coerce").max()
    positions_max_month = latest.strftime("%Y-%m") if pd.notna(latest) else ""
    stale, stale_reason = freshness(meta, tx_rows=tx_rows,
                                    positions_max_month=positions_max_month)
    if meta_unreadable:
        stale_reason = ("unreadable build metadata (lots_meta.json) — "
                        "refresh with py parsers/build_lots.py --write")

    types = account_types(raw_positions)
    prices = price_map(getattr(frames, "prices_latest", None))
    stmt_prices, stmt_asof = stmt_price_map(raw_positions)
    tmap = _type_map(
        raw_positions,
        etf_class=dict(getattr(hs, "ETF_TICKER_CLASS", {}) or {}),
        core_etf_symbols=frozenset(
            getattr(hs, "FIDELITY_CORE_ETF_SYMBOLS", None) or ()))
    tlh_id = _tlh_account_id()
    labels = getattr(hs, "ACCOUNT_DISPLAY", {}) or {}
    # The route validates broker OPTION IDS (slugs); positions carries raw
    # broker LABELS. Resolve ids -> labels through the same by_id map
    # apply_global_filters uses — comparing ids against labels only works
    # while every label happens to equal its own slug. Unresolvable ids
    # fall back to unfiltered, the house rule (the route 422s unknowns
    # before they get here).
    broker_sel = None
    if broker and "all" not in broker:
        _, by_id = hs._broker_options(hs._current_snap(frames))
        resolved = {by_id[i] for i in broker if i in by_id}
        broker_sel = resolved or None
    acct_broker = ({str(a): str(b) for a, b in
                    raw_positions.groupby("account_id")["broker"].first()
                    .items()}
                   if "broker" in raw_positions.columns else {})

    def keep_account(acct: str) -> bool:
        """The global broker picker's narrowing, as one predicate — open
        lots and harvest must never disagree about which accounts show."""
        return (broker_sel is None
                or acct_broker.get(acct) in broker_sel)

    rows: list[dict] = []
    excluded = {"ira_accounts": set(), "unknown_accounts": set()}
    for _idx, lot in lots.iterrows():
        acct = str(lot["account_id"])
        taxable = taxable_of(types.get(acct))
        if taxable is None:
            excluded["unknown_accounts"].add(acct)
            continue
        if taxable is False:
            excluded["ira_accounts"].add(acct)
            continue
        if not keep_account(acct):
            continue
        sym = str(lot["symbol"]) if pd.notna(lot["symbol"]) else ""
        ikey = str(lot["instrument_key"])
        qty = float(lot["quantity_remaining"])
        basis = (float(lot["basis_remaining"])
                 if pd.notna(lot["basis_remaining"]) else None)
        # symbol falling back to instrument_key is the scanner's own
        # lookup (parsers/tax_scanner.py): a crosswalk-resolved lot
        # carries its ticker in the KEY with an empty symbol column.
        # Open lots priced by bare symbol while Harvest priced by the
        # fallback left ~$450k of TOD basis "unpriced" here (TK
        # feedback 2026-07-30) — the S3b rule again: two views of one
        # lot must never disagree.
        price = prices.get(sym or ikey)
        price_source = "live" if price is not None else None
        if price is None:
            price = stmt_prices.get(sym or ikey)
            if price is not None:
                price_source = "statement"
        mv = _round2(qty * price) if price is not None else None
        unreal = (_round2(mv - basis)
                  if mv is not None and basis is not None else None)
        acquired = (pd.Timestamp(lot["acquired_date"])
                    if pd.notna(lot["acquired_date"]) else None)
        rows.append({
            # the lots.csv positional row index: the sim-selection key.
            # Stable for a given build (any rebuild reshuffles — sim legs
            # re-validate server-side on every call, so a stale id can
            # only produce a NAMED rejection, never a wrong number).
            "lot_id": int(_idx),
            "account_id": acct,
            "account_label": str(labels.get(acct, acct)),
            "symbol": sym,
            "instrument_key": str(lot["instrument_key"]),
            # classified by the SAME sym-or-ikey lookup pricing uses just
            # above — a crosswalk-resolved lot (empty symbol, ticker in
            # the key) must not price as SPY yet classify as "other"
            "type": tmap.get(sym or ikey, "other"),
            "is_tlh": bool(tlh_id) and acct == tlh_id,
            "origin": str(lot["origin"]),
            "acquired_date": (acquired.strftime("%Y-%m-%d")
                              if acquired is not None else None),
            "term": term_of(acquired, asof),
            "days_to_long_term": days_to_long_term(acquired, asof),
            "quantity_remaining": float(qty),
            "basis_remaining": _round2(basis),
            "price": _round2(price),
            "price_source": price_source,
            "price_asof": (stmt_asof if price_source == "statement"
                           else None),
            "market_value": mv,
            "unrealized_gl": unreal,
            "basis_evidence": str(lot["basis_evidence"]),
            "band": str(lot["band"]),
        })
    rows.sort(key=lambda r: (r["account_id"], r["symbol"],
                             r["acquired_date"] or "9999-99-99"))

    accounts: list[dict] = []
    for acct in sorted({r["account_id"] for r in rows}):
        mine = [r for r in rows if r["account_id"] == acct]
        accounts.append({
            "id": acct,
            "label": str(labels.get(acct, acct)),
            "lots": len(mine),
            "basis": _round2(sum(r["basis_remaining"] or 0.0 for r in mine)),
            # null-never-zero: an account with no priced lots has no market
            # value to report, not a $0.00 one
            "market_value": _sum_or_none([r["market_value"] for r in mine]),
            "unrealized_gl": _sum_or_none([r["unrealized_gl"]
                                           for r in mine]),
        })

    priced = [r for r in rows if r["market_value"] is not None]
    by_evidence: dict[str, int] = {}
    by_band: dict[str, int] = {}
    for r in rows:
        by_evidence[r["basis_evidence"]] = \
            by_evidence.get(r["basis_evidence"], 0) + 1
        by_band[r["band"]] = by_band.get(r["band"], 0) + 1

    gate = (meta or {}).get("gate")
    view_meta = {
        "built_at": (meta or {}).get("built_at"),
        "open_lots": (meta or {}).get("open_lots"),
        "gate": gate,
        "joined_bands": (meta or {}).get("joined_bands"),
        "stale": stale,
        "stale_reason": stale_reason,
        "asof": asof.isoformat(),
        "filter": {},
    }
    view_meta.update(options)
    return {
        "kind": "tax",
        "meta": view_meta,
        "summary": {
            "accounts": accounts,
            # basis/market_value/unrealized_gl and priced_basis all
            # describe the PRICED universe, so the triplet is internally
            # consistent (mv - priced_basis == unrealized_gl); the
            # unpriced remainder is stated beside it, never silently
            # deflating market value against an all-lots basis (TK
            # feedback 2026-07-30: the mixed pair read as a large loss)
            "totals": {
                "basis": _round2(sum(r["basis_remaining"] or 0.0
                                     for r in rows)),
                "priced_basis": _round2(sum(r["basis_remaining"] or 0.0
                                            for r in priced)),
                "unpriced_basis": _round2(sum(r["basis_remaining"] or 0.0
                                              for r in rows
                                              if r["market_value"] is None)),
                "market_value": _sum_or_none([r["market_value"]
                                              for r in priced]),
                "unrealized_gl": _sum_or_none([r["unrealized_gl"]
                                               for r in priced]),
                "priced_lots": len(priced),
                "stmt_priced_lots": sum(
                    1 for r in priced
                    if r["price_source"] == "statement"),
                "unpriced_lots": len(rows) - len(priced),
            },
            "by_evidence": by_evidence,
            "by_band": by_band,
            "excluded": {"ira_accounts": len(excluded["ira_accounts"]),
                         "unknown_accounts":
                             len(excluded["unknown_accounts"])},
            "realized_ytd": _realized_ytd(meta, types, keep_account, labels),
            "silent_share_note": _silent_note(meta),
        },
        "lots": rows,
        "harvest": _harvest_or_unavailable(
            lots, raw_tx, raw_positions, prices, asof=asof, types=types,
            labels=labels, keep_account=keep_account, stale=stale,
            tmap=tmap, tlh_id=tlh_id),
    }


# ---------------------------------------------------------------------------
# Year tax estimate (+ sell simulator) — spec 2026-08-06.

_PROFILE_FIELDS = ("filing_status", "w2_income", "state", "deduction",
                   "carryforward_loss", "qualified_dividend_pct",
                   "unknown_term_assumption")
_PROFILE_REQUIRED = ("filing_status", "w2_income", "state")
_PROFILE_DEFAULTS = {"deduction": "standard", "carryforward_loss": 0.0,
                     "qualified_dividend_pct": 1.0,
                     "unknown_term_assumption": "long"}


def _config_profile() -> dict:
    """config_local.TAX_PROFILE via the hs.cfg seam (tests patch HERE,
    so the dev box's real profile can never leak into a test run)."""
    return dict(getattr(getattr(hs, "cfg", None), "TAX_PROFILE", None)
                or {})


def _estimate_error(reason: str) -> dict:
    return {"kind": "error", "reason": reason}


def _year_income(raw_tx: pd.DataFrame, types: dict, year: int) -> dict:
    """Taxable-account calendar-year income actuals + the CA-exempt
    treasury share of interest. Same fail-closed account predicate as
    the open-lots loop."""
    tx = raw_tx[raw_tx["account_id"].astype(str)
                .map(lambda a: taxable_of(types.get(a)) is True)].copy()
    # Distribution rows (principal_pmt) are yield on the Income tab but
    # return of capital to the IRS — taxed once, via the basis reduction
    # the lot engine already applies. income_timeseries folds them into
    # `dividends`, so the estimate must feed it the income rows only, or
    # the same dollars are taxed as dividends now AND as capital gain
    # later off the lowered basis (DA-F-1).
    ts = income_timeseries(
        tx[~tx["transaction_type"].astype(str).isin(DISTRIBUTION_TYPES)])
    if ts.empty:
        sums = {"dividends": 0.0, "interest": 0.0, "withholding": 0.0}
        frontier = None
    else:
        mine = ts[ts.index.year == year]
        sums = {c: float(mine[c].sum()) for c in
                ("dividends", "interest", "withholding")}
        frontier = tx_frontier_of(tx)
    treasury = 0.0
    if not tx.empty:
        inte = tx[tx["transaction_type"].astype(str) == "interest"].copy()
        when = pd.to_datetime(inte["settlement_date"], errors="coerce")
        if "trade_date" in inte.columns:
            when = when.fillna(pd.to_datetime(inte["trade_date"],
                                              errors="coerce"))
        inte = inte[when.dt.year == year]
        for _, r in inte.iterrows():
            if is_treasury_income(str(r.get("symbol", "") or ""),
                                  str(r.get("description", "") or "")):
                # NaN is truthy, so `... or 0.0` never fires for a
                # blank/garbled amount — it would sum NaN into treasury
                # forever and 500 the whole estimate at the
                # allow_nan=False boundary. Skip unparsable rows instead.
                v = pd.to_numeric(r.get("amount"), errors="coerce")
                if pd.notna(v):
                    treasury += float(v)
    return {"dividends": round(sums["dividends"], 2),
            "interest": round(sums["interest"], 2),
            "treasury_interest": round(treasury, 2),
            "withholding": round(sums["withholding"], 2),
            "through": (frontier.strftime("%Y-%m-%d")
                       if frontier is not None else "")}


def _recent_acquisition_observed(acquisitions: pd.DataFrame,
                                 instrument_key: str, own_source_row,
                                 asof: date) -> bool:
    """Factual wash signal for a simulated loss sale: a replacement
    ACQUISITION (buy or reinvestment — parsers.tax_scanner's
    ACQUISITION_TYPES) of the same RESOLVED instrument key, OTHER THAN
    this lot's own opening purchase, in any account, inside the
    trailing 30-day backward OBSERVABLE window.

    Delegates to the harvest scanner's own `replacement_buys` — the
    identical window rule the harvest view judges real candidates with,
    restated for a hypothetical sell — plus its self-exclusion rule
    (parsers.tax_scanner.scan_harvest_candidates, ~line 303-310): the
    buy that OPENED this lot is the acquisition of the shares being
    sold, never a replacement for itself, so it is dropped from the hit
    set by transactions row identity (`own_source_row`, lots.csv's own
    `source_row` column for this lot — NaN on a synthesized opening lot
    means nothing to exclude). Without this, EVERY loss lot bought
    within the window would flag on its own purchase — exactly the
    recently-bought-underwater lots a user is most likely to simulate
    selling.

    `acquisitions` must already be `keyed_acquisitions(...)` output, so
    matching runs on the ledger's own crosswalk identity — the same key
    space `instrument_key` is drawn from and lots.csv itself is keyed
    in — never a raw printed symbol compare: a crosswalk-resolved lot
    (empty symbol, ticker in instrument_key) whose replacement
    acquisition ALSO prints no symbol (a DRIP/transfer row, the exact
    shape the resolver exists for) still matches. Zero-flag is NOT
    safety: most of the window is unobservable, and a PARTIAL sale can
    still wash against the retained shares of this same purchase (see
    the assumptions list `build_tax_estimate` returns)."""
    hits = replacement_buys(acquisitions, instrument_key, asof,
                            window_days=WINDOW_DAYS, sides="backward")
    if hits.empty:
        return False
    own_row = pd.to_numeric(own_source_row, errors="coerce")
    if pd.notna(own_row):
        hits = hits[pd.to_numeric(hits["source_row"], errors="coerce")
                   != own_row]
    return not hits.empty


def _sim_legs(sim, view_lots: list[dict], acquisitions: pd.DataFrame,
              source_row_by_lot_id: dict, asof: date
              ) -> tuple[list[dict], list[str]]:
    """Validate ALL-OR-NOTHING (spec §5.1): any bad leg empties the
    accepted list; every failure is named. No partial estimates.
    `source_row_by_lot_id` (lots.csv's own `source_row` column, keyed by
    the same positional index `lot_id` is) lets the wash check exclude
    a lot's own opening purchase from its own replacement-acquisition
    hits — see `_recent_acquisition_observed`."""
    by_id = {r["lot_id"]: r for r in view_lots}
    legs, rejected = [], []
    for s in (sim or []):
        lid, qty = s.get("lot_id"), s.get("qty")
        try:
            lid = int(lid)
            qty = float(qty)
        except (TypeError, ValueError):
            rejected.append(f"leg {s!r}: lot_id/qty not numeric")
            continue
        row = by_id.get(lid)
        if row is None:
            rejected.append(f"unknown lot_id {lid}")
            continue
        if not qty > 0:
            rejected.append(f"lot {lid}: qty must be > 0")
            continue
        remaining = float(row["quantity_remaining"])
        if qty > remaining + 1e-9:
            rejected.append(f"lot {lid}: qty {qty:g} exceeds "
                            f"remaining {remaining:g}")
            continue
        if row["price"] is None or row["basis_remaining"] is None:
            rejected.append(f"lot {lid}: unpriced (no live or statement "
                            f"mark) — cannot simulate")
            continue
        if row["term"] not in ("short", "long"):
            rejected.append(f"lot {lid}: term unknown — cannot simulate")
            continue
        frac = qty / remaining
        proceeds = qty * float(row["price"])
        basis_part = float(row["basis_remaining"]) * frac
        gl = round(proceeds - basis_part, 2)
        key = row["symbol"] or row["instrument_key"]
        legs.append({
            "lot_id": lid,
            "symbol": key,
            "account_label": row["account_label"],
            "qty": qty,
            "proceeds": round(proceeds, 2),
            "basis_part": round(basis_part, 2),
            "gl": gl,
            "term": row["term"],
            "wash_observed": bool(gl < 0 and _recent_acquisition_observed(
                acquisitions, row["instrument_key"],
                source_row_by_lot_id.get(lid), asof)),
        })
    if rejected:
        return [], rejected
    return legs, []


def build_tax_estimate(frames, data_dir: str | Path, *,
                       overrides: dict | None = None,
                       sim: list[dict] | None = None,
                       asof: date | None = None) -> dict:
    """POST /api/tax/estimate payload. Wraps estimate_year_tax; ALL
    numbers come from the engine. Whole taxable book by design
    (broker=None): calendar-year taxes do not narrow by picker."""
    asof = asof or date.today()
    view = build_tax_view(frames, data_dir, asof=asof, broker=None)
    if view.get("kind") == "error":
        return _estimate_error(view.get("reason", "tax view unavailable"))
    rz = (view.get("summary") or {}).get("realized_ytd") or {}
    if rz.get("unavailable"):
        return _estimate_error(
            "realized_ytd unavailable — " + str(rz["unavailable"]))

    profile = {**_PROFILE_DEFAULTS, **_config_profile(),
               **{k: v for k, v in (overrides or {}).items()
                  if v is not None}}
    missing = [k for k in _PROFILE_REQUIRED if not str(
        profile.get(k, "")).strip()]
    if missing:
        return _estimate_error(
            "tax profile not configured (missing: " + ", ".join(missing)
            + ") — set TAX_PROFILE in config_local.py (schema in "
            "config_example.py) or use the override form")
    profile = {k: profile.get(k) for k in _PROFILE_FIELDS}

    year = int(rz.get("year") or TAX_YEAR)
    bt = rz.get("by_term") or {}
    realized = {t: float((bt.get(t) or {}).get("net") or 0.0)
                for t in ("short", "long", "unknown")}

    data = Path(data_dir)
    raw_tx = pd.read_csv(data / "transactions.csv")
    raw_positions = pd.read_csv(data / "positions.csv")
    types = account_types(raw_positions)
    income = _year_income(raw_tx, types, year)

    # The harvest scan's own crosswalk (S3b): resolves a replacement
    # acquisition's instrument key the same way lots.csv itself is keyed,
    # so wash matching never depends on a printed symbol being present.
    fold, _splits = load_corporate_identity()
    resolver, cusip_resolver = build_key_resolvers(raw_tx, raw_positions,
                                                    fold)
    acquisitions = keyed_acquisitions(raw_tx, resolver, cusip_resolver, fold)
    # lot_id IS lots.csv's own positional row index (Task 2's invariant),
    # so this dict lets the wash check exclude a lot's own opening buy
    # from its own hits without adding source_row to the view payload.
    lots_df = pd.read_csv(data / "lots.csv")
    source_row_by_lot_id = lots_df["source_row"].to_dict()
    legs, rejected = _sim_legs(sim, view.get("lots") or [], acquisitions,
                               source_row_by_lot_id, asof)
    engine_income = {k: income[k] for k in
                     ("dividends", "interest", "treasury_interest",
                      "withholding")}
    try:
        baseline = estimate_year_tax(profile, realized, engine_income)
        with_sim = (estimate_year_tax(
            profile, realized, engine_income,
            [{"gl": leg["gl"], "term": leg["term"]} for leg in legs])
            if legs else None)
    except (ValueError, TypeError) as exc:
        # ValueError: unsupported filing_status/state (the engine's own
        # validation). TypeError: a hand-edited config_local TAX_PROFILE
        # with a None field (e.g. w2_income) passes the missing-fields
        # check (str(None) is a non-blank string) but blows up float(None)
        # inside the engine — same named-degrade path, not a 500.
        return _estimate_error(str(exc))

    meta = view.get("meta") or {}
    assumptions = list(baseline["assumptions"]) + [
        "whole taxable book (broker filter not applied)",
        "realized YTD excludes Fidelity option confirms",
        "wash flags cover only the observable window — zero flags is "
        "not safety",
        "wash flags cover replacement purchases OTHER than the lot's own "
        "opening buy — a partial sale can still wash against the shares "
        "you keep from that same purchase",
        f"income actuals through {income['through'] or 'n/a'}",
    ]
    if year != TAX_YEAR:
        assumptions.append(f"realized year {year} estimated on "
                           f"{TAX_YEAR} tables")
    return {
        "kind": "estimate",
        "year": year,
        "baseline": baseline,
        "with_sim": with_sim,
        "sim_legs": legs,
        "sim_rejected": rejected,
        "profile_used": profile,
        "income": income,
        "provenance": {
            "lots_stale": bool(meta.get("stale")),
            "stale_reason": str(meta.get("stale_reason") or ""),
            "income_through": income["through"],
            "table_note": "fed 2026 (Rev. Proc. 2025-32) · CA 2025 "
                          "schedule",
        },
        "assumptions": assumptions,
    }
