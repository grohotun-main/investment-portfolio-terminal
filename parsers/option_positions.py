"""Extract strike / expiry / opt_type from option position rows.

Positions CSV stores option holdings with a free-text `description` field
whose format depends on the broker. Strike + expiry data live there for JPM
but not Fidelity — Fidelity positions only carry the option_type + name,
which means the (strike, expiry) has to come from the most recent matching
BUY transaction. Two broker formats handled:

* **JPM position description** —  ``PUT NVDA 12/18/26 135 NVIDIA CORPORATION ...``
  Order is fixed: type, ticker, MM/DD/YY expiry, strike. Parsed directly.

* **Fidelity position description** —  ``PUT NVIDIA CORPORATION`` (just
  type + issuer name; no strike/expiry).  Strike + expiry are recovered
  from the most recent matching BUY transaction whose description carries
  the canonical Fidelity option format:
  ``PUT (NVDA) NVIDIA CORPORATION ... You Bought DEC 18 26 $135 (100 SHS) ...``

Cost-basis per share is computed from the most recent matching BUY
transaction in the same account: ``-amount / (quantity * 100)``.

This module is pure-Python (re + datetime + pandas). No options pricer,
no network — those layers consume the table this module builds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from monthly_normalize import slice_as_of_month

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONTRACT_MULT = 100

OptType = Literal["call", "put"]

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# JPM position description: "PUT NVDA 12/18/26 135 NVIDIA CORPORATION ..."
# Strike is the FIRST bare number after the date. Allow integer or decimal,
# but not embedded in another token (so "10:1 STOCK SPLIT" doesn't trip).
_JPM_RE = re.compile(
    r"^\s*(PUT|CALL)\s+([A-Z][A-Z0-9.]*)\s+"
    r"(\d{2}/\d{2}/\d{2})\s+"
    r"(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

# JPM BUY transaction: "PUT SPY 12/18/26 ETF UNSOLICITED OPEN CONTRACT..."
# Same prefix as the position description but no strike (JPM omits it on the
# trade row). Used only by the cost-basis lookup, which matches on
# (type, underlying, expiry) and pulls strike from the position side.
_JPM_BUY_RE = re.compile(
    r"^\s*(PUT|CALL)\s+([A-Z][A-Z0-9.]*)\s+(\d{2}/\d{2}/\d{2})\b",
    re.IGNORECASE,
)

# Fidelity BUY transaction: "PUT (NVDA) NVIDIA CORPORATION ... You Bought
#   DEC 18 26 $135 (100 SHS) OPENING ..."
_FID_TYPE_RE = re.compile(
    r"^\s*(PUT|CALL)\s+\(([A-Z][A-Z0-9.]*)\)",
    re.IGNORECASE,
)
# Tolerate "DEC 18 26", "DEC 18 2026", "DEC 18, 2026"
_FID_DATE_RE = re.compile(
    r"\b([A-Z]{3})\s+(\d{1,2}),?\s+(\d{2,4})\b",
    re.IGNORECASE,
)
_FID_STRIKE_RE = re.compile(r"\$(\d[\d,]*(?:\.\d+)?)")


@dataclass(frozen=True)
class ParsedOption:
    opt_type: str          # "put" | "call"
    underlying: str        # ticker
    expiry: date
    strike: float


def parse_jpm_option_desc(desc: str | None) -> ParsedOption | None:
    """Parse a JPM-style option position description.

    Returns ParsedOption on success, None if the description doesn't match
    the expected shape (caller's choice whether that's a soft skip or a
    surfaced warning).
    """
    if not isinstance(desc, str):
        return None
    m = _JPM_RE.match(desc)
    if not m:
        return None
    opt_type = m.group(1).lower()
    try:
        expiry = datetime.strptime(m.group(3), "%m/%d/%y").date()
        strike = float(m.group(4))
    except ValueError:
        return None
    if strike <= 0:
        return None
    return ParsedOption(opt_type=opt_type, underlying=m.group(2).upper(),
                        expiry=expiry, strike=strike)


def parse_fidelity_buy_desc(desc: str | None) -> ParsedOption | None:
    """Parse a Fidelity BUY transaction description.

    Note: this parses BUY *transactions*, not position rows — Fidelity
    position descriptions don't carry strike/expiry at all. The caller is
    expected to walk back from a position row to a matching BUY txn.
    """
    if not isinstance(desc, str):
        return None
    tm = _FID_TYPE_RE.match(desc)
    if not tm:
        return None
    opt_type = tm.group(1).lower()
    underlying = tm.group(2).upper()
    dm = _FID_DATE_RE.search(desc)
    if not dm:
        return None
    mon = _MONTHS.get(dm.group(1).upper())
    if mon is None:
        return None
    try:
        day = int(dm.group(2))
        year = int(dm.group(3))
        if year < 100:
            year += 2000
        expiry = date(year, mon, day)
    except (ValueError, TypeError):
        return None
    sm = _FID_STRIKE_RE.search(desc)
    if not sm:
        return None
    try:
        strike = float(sm.group(1).replace(",", ""))
    except ValueError:
        return None
    if strike <= 0:
        return None
    return ParsedOption(opt_type=opt_type, underlying=underlying,
                        expiry=expiry, strike=strike)


def _is_option_row(row: pd.Series) -> bool:
    ac = str(row.get("asset_class") or "")
    return ac.startswith("option")


def _is_option_buy(row: pd.Series) -> bool:
    if str(row.get("transaction_type") or "").lower() != "buy":
        return False
    desc = row.get("description")
    if not isinstance(desc, str):
        return False
    return bool(re.match(r"^\s*(PUT|CALL)\b", desc, re.IGNORECASE))


def _is_option_sell(row: pd.Series) -> bool:
    if str(row.get("transaction_type") or "").lower() != "sell":
        return False
    desc = row.get("description")
    if not isinstance(desc, str):
        return False
    return bool(re.match(r"^\s*(PUT|CALL)\b", desc, re.IGNORECASE))


def _parse_full_option_txn(row: pd.Series) -> ParsedOption | None:
    """Parse a BUY/SELL transaction desc into a full ParsedOption (strike included).

    Tries the JPM position-style parser first (which interim CSV BUYs honor:
    ``PUT SPY 11/20/26 640 ...``) then the Fidelity BUY parser. Returns None
    when strike is missing from the description (legacy PDF-parsed JPM rows
    of the form ``PUT SPY 12/18/26 ETF OPEN CONTRACT``) — those are unusable
    for post-statement synthesis because we can't group without a strike.
    """
    desc = row.get("description")
    p = parse_jpm_option_desc(desc)
    if p is not None:
        return p
    return parse_fidelity_buy_desc(desc)


def parse_jpm_buy_desc(desc: str | None) -> ParsedOption | None:
    """Parse a JPM BUY transaction description.

    JPM BUY rows carry type + underlying + expiry but no strike (the
    statement omits it on the trade row). Returned ParsedOption has
    ``strike=NaN`` so callers know it must be paired with a position
    row to recover the full quintuple. Cost-basis matching folds these
    in by (account, type, underlying, expiry).
    """
    if not isinstance(desc, str):
        return None
    m = _JPM_BUY_RE.match(desc)
    if not m:
        return None
    try:
        expiry = datetime.strptime(m.group(3), "%m/%d/%y").date()
    except ValueError:
        return None
    return ParsedOption(opt_type=m.group(1).lower(),
                        underlying=m.group(2).upper(),
                        expiry=expiry,
                        strike=float("nan"))


def _parse_buy(row: pd.Series) -> ParsedOption | None:
    desc = row.get("description")
    # Try Fidelity first (it carries strike); fall back to JPM (no strike).
    fid = parse_fidelity_buy_desc(desc)
    if fid is not None:
        return fid
    return parse_jpm_buy_desc(desc)


def build_option_position_table(
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return one row per option position on `as_of` (or latest statement
    date if None).

    Columns:
      account_id, statement_date, asset_class, symbol, description,
      opt_type, underlying, expiry, strike, quantity,
      market_value, premium_per_share_mv, cost_basis_per_share,
      cost_basis_total, source ("jpm" | "fidelity" | "post_statement"
      | "unparsed")

    `premium_per_share_mv` is back-derived from market_value:
    ``market_value / (quantity * 100)``. Useful for sanity-checking
    against Polygon mid.

    Positions that can't be parsed (e.g. Fidelity rows with no matching
    BUY transaction) come back with source="unparsed" and NaN for
    opt_type/underlying/expiry/strike so callers can decide whether to
    drop or surface them.

    Post-statement opens: any option BUY whose settlement_date is
    AFTER the latest statement is synthesized into an additional row
    (source="post_statement"), with quantity = net of in-window
    BUY+SELL pairs. Round-trip closes net to zero and are skipped.
    Adds to an existing position pool quantity and cost basis instead
    of duplicating. Market value is bootstrapped from the latest BUY
    price; the IV-fetch step refreshes it with Polygon mids.

    Note on synth-rolled positions: when ``synthesize_interim_positions``
    has been applied upstream, the latest option statement_date in
    ``positions`` is the synth roll date (= ``max(interim_settlement_date)``),
    not the real statement date — and the rolled rows already carry every
    interim open / close / expiry. The function uses that synth date as
    ``as_of_ts`` for the snap and SKIPS post-statement synthesis (re-adding
    the in-window BUYs on top of the rolled rows double-counted them).
    Detection: ``as_of_ts == max(txn.settlement_date)`` (see
    ``_detect_real_statement_cutoff``). The Fidelity cost-basis walk-back
    still sees every BUY on or before ``as_of_ts``.
    """
    opt_pos = positions[positions.apply(_is_option_row, axis=1)].copy()
    if not opt_pos.empty:
        opt_pos["statement_date"] = pd.to_datetime(opt_pos["statement_date"])

    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
    elif not opt_pos.empty:
        as_of_ts = opt_pos["statement_date"].max()
    elif not positions.empty and "statement_date" in positions.columns:
        as_of_ts = pd.to_datetime(positions["statement_date"]).max()
    else:
        as_of_ts = pd.Timestamp.now().normalize()

    txn_cutoff = _detect_real_statement_cutoff(opt_pos, transactions, as_of_ts)

    snap = (slice_as_of_month(opt_pos, as_of_ts)
            if not opt_pos.empty else opt_pos)

    # Pre-parse all option BUY transactions on or before as_of_ts so we can
    # match Fidelity positions back to the trade that opened them.
    txn_buys = transactions[transactions.apply(_is_option_buy, axis=1)].copy()
    if not txn_buys.empty:
        txn_buys["settlement_date"] = pd.to_datetime(txn_buys["settlement_date"])
        txn_buys = txn_buys[txn_buys["settlement_date"] <= as_of_ts].copy()
        parsed: list[ParsedOption | None] = [
            _parse_buy(r) for _, r in txn_buys.iterrows()
        ]
        txn_buys["_parsed"] = parsed
        txn_buys = txn_buys[txn_buys["_parsed"].notna()].copy()

    rows: list[dict] = []
    for _, pos in snap.iterrows():
        parsed = parse_jpm_option_desc(pos.get("description"))
        source = "jpm" if parsed else None

        if parsed is None:
            # Fidelity-style: walk back to the most recent matching BUY.
            parsed, source = _find_fidelity_match(pos, txn_buys)

        mv = float(pos.get("market_value") or 0.0)
        qty = float(pos.get("quantity") or 0.0)
        # Broker-stated statement cost basis is authoritative when present.
        # The txn-pool fallback has two blind spots a roll exposes: sells are
        # never netted, and JPM strike-less buys pool across every strike
        # sharing (acct, type, underlying, expiry) — see RollCostBasisTests.
        try:
            stmt_cb = float(pos.get("cost_basis"))
        except (TypeError, ValueError):
            stmt_cb = float("nan")
        if stmt_cb == stmt_cb and stmt_cb > 0:      # finite and positive
            cost_basis_total = stmt_cb
            cost_basis_per_share = (stmt_cb / (qty * CONTRACT_MULT)
                                    if qty > 0 else float("nan"))
        else:
            cost_basis_per_share, cost_basis_total = _cost_basis(
                pos, parsed, txn_buys
            )
        premium_mv = (mv / (qty * CONTRACT_MULT)
                      if qty > 0 and CONTRACT_MULT > 0 else float("nan"))
        rows.append({
            "account_id":    pos.get("account_id"),
            "statement_date": as_of_ts,
            "asset_class":   pos.get("asset_class"),
            "symbol":        pos.get("symbol"),
            "description":   pos.get("description"),
            "opt_type":      parsed.opt_type   if parsed else None,
            "underlying":    parsed.underlying if parsed else None,
            "expiry":        parsed.expiry     if parsed else None,
            "strike":        parsed.strike     if parsed else float("nan"),
            "quantity":      qty,
            "market_value":  mv,
            "premium_per_share_mv":  premium_mv,
            "cost_basis_per_share":  cost_basis_per_share,
            "cost_basis_total":      cost_basis_total,
            "source":        source or "unparsed",
        })

    if txn_cutoff >= as_of_ts:
        rows.extend(_synthesize_post_statement_opens(
            transactions, txn_cutoff, as_of_ts, rows
        ))
    # else: synth-rolled snapshot (txn_cutoff stepped back to the real
    # statement date) — the roll-forward already booked every interim open /
    # close / expiry into `positions`, so re-synthesizing the in-window BUYs
    # here double-counted them (Aug 2026: 2 SNDK calls became 4). The rolled
    # rows are authoritative; post-statement synthesis is the NON-rolled
    # path's job (a statement snapshot read without the roll).

    if not rows:
        return _empty_position_table()
    return pd.DataFrame(rows)


def greek_dollar_columns(opt_tbl: pd.DataFrame) -> pd.DataFrame:
    """Add the 3 per-row Greek-dollar display columns (gamma$/vega$/theta$) to an
    option table that already carries model_gamma/model_vega/model_theta, spot,
    quantity. The single source of the dollar-Greek formulas for BOTH app.py's
    read-half opt_tbl build and options_service._assemble_opt_tbl. (unrealized_pnl
    is NOT set here — each UI sets it separately.) Mutates + returns opt_tbl."""
    opt_tbl["gamma_dollar_per_1pct"] = (
        opt_tbl["quantity"] * 100.0
        * opt_tbl["model_gamma"] * (opt_tbl["spot"] ** 2) * 0.01)
    opt_tbl["vega_dollar_per_volpt"] = (
        opt_tbl["quantity"] * 100.0 * opt_tbl["model_vega"] * 0.01)
    opt_tbl["theta_dollar_per_day"] = (
        opt_tbl["quantity"] * 100.0 * opt_tbl["model_theta"] / 365.0)
    return opt_tbl


def option_book_aggregates(
    opt_tbl: pd.DataFrame,
    today: date | pd.Timestamp,
) -> dict:
    """Aggregate-exposure tiles over the *live* long-option book.

    The live book excludes positions that are no longer current holdings:

    * **expired** contracts (``expiry < today``), and
    * **zero-quantity** rows (closed positions still listed on the last
      statement).

    "Premium at risk" and unrealized P&L use the live Polygon mid
    (``quantity * 100 * premium_mid``) when a mid is available, falling
    back to the statement-reported ``market_value`` when it is NaN/absent.
    Both behaviours were flagged by the 2026-06 audit (WS-E): the tiles
    previously summed statement ``market_value`` over *every* row,
    including expired/closed ones — overstating the unrealized loss and
    understating premium-at-risk versus today's mid.

    Returns a dict:
      ``notional_protected`` — Σ puts' ``qty × strike × 100`` (live book)
      ``premium_at_risk``    — Σ live market value (live book)
      ``cost_basis``         — Σ ``cost_basis_total`` (live book)
      ``unrealized_pnl``     — ``premium_at_risk − cost_basis``
      ``weighted_dte``       — MV-weighted days-to-expiry (NaN if no MV)
      ``n_live`` / ``n_excluded`` — live vs dropped row counts
    """
    empty = {
        "notional_protected": 0.0, "premium_at_risk": 0.0,
        "cost_basis": 0.0, "unrealized_pnl": 0.0,
        "weighted_dte": float("nan"), "n_live": 0, "n_excluded": 0,
    }
    if opt_tbl is None or opt_tbl.empty:
        return empty

    today_ts = pd.Timestamp(today).normalize()
    expiry = pd.to_datetime(opt_tbl["expiry"], errors="coerce").dt.normalize()
    qty = pd.to_numeric(opt_tbl["quantity"], errors="coerce").fillna(0.0)

    # Live = open (non-zero qty) AND not expired (expiry on/after today).
    # A NaT expiry is treated as not-live (malformed / not a current contract).
    live_mask = (qty != 0) & expiry.notna() & (expiry >= today_ts)
    n_excluded = int((~live_mask).sum())
    if not live_mask.any():
        return {**empty, "n_excluded": n_excluded}

    live = opt_tbl[live_mask]
    live_qty = qty[live_mask]
    mv_stmt = pd.to_numeric(live["market_value"], errors="coerce").fillna(0.0)
    if "premium_mid" in live.columns:
        mid = pd.to_numeric(live["premium_mid"], errors="coerce")
    else:
        mid = pd.Series(float("nan"), index=live.index)
    # Live MV: quantity × 100 × Polygon mid where present, else statement MV.
    live_mv = (live_qty * CONTRACT_MULT * mid).where(mid.notna(), mv_stmt)
    total_mv = float(live_mv.sum())

    strike = pd.to_numeric(live["strike"], errors="coerce").fillna(0.0)
    is_put = live["opt_type"].astype("string").str.lower().eq("put").fillna(False)
    notional = (live_qty * strike * CONTRACT_MULT).where(is_put, 0.0)

    dte = (expiry[live_mask] - today_ts).dt.days.clip(lower=0)
    cost_basis = float(
        pd.to_numeric(live["cost_basis_total"], errors="coerce")
        .fillna(0.0).sum()
    )
    weighted_dte = (
        float((dte * live_mv).sum() / total_mv) if total_mv > 0
        else float("nan")
    )
    return {
        "notional_protected": float(notional.sum()),
        "premium_at_risk": total_mv,
        "cost_basis": cost_basis,
        "unrealized_pnl": total_mv - cost_basis,
        "weighted_dte": weighted_dte,
        "n_live": int(live_mask.sum()),
        "n_excluded": n_excluded,
    }


def _detect_real_statement_cutoff(
    opt_pos: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of_ts: pd.Timestamp,
) -> pd.Timestamp:
    """Return the txn-cutoff date for post-statement synthesis.

    When the upstream caller has run ``synthesize_interim_positions``, the
    ``as_of_ts`` we picked is the synth roll date (= max interim settlement
    date), not a real statement date. Filtering ``txn.settlement_date >
    as_of_ts`` then excludes the very interim BUYs we want to detect.

    Heuristic: if ``as_of_ts`` equals ``max(transactions.settlement_date)``,
    assume synth-rolled and step back to the prior unique option
    statement_date. Otherwise return ``as_of_ts`` unchanged.

    Detection is robust because synth-rolling always uses the max interim
    settlement date, which is also the max settlement date in the
    ``transactions`` frame passed to us (callers union interim into
    transactions before calling this function). When no synth has been
    applied, the latest option statement is older than the latest txn,
    so detection naturally no-ops.
    """
    if opt_pos.empty or transactions is None or transactions.empty:
        return as_of_ts
    if "settlement_date" not in transactions.columns:
        return as_of_ts
    max_txn = pd.to_datetime(transactions["settlement_date"]).max()
    if as_of_ts != max_txn:
        return as_of_ts
    earlier = opt_pos.loc[
        opt_pos["statement_date"] < as_of_ts, "statement_date"
    ]
    if earlier.empty:
        return as_of_ts
    return earlier.max()


def _synthesize_post_statement_opens(
    transactions: pd.DataFrame,
    txn_cutoff: pd.Timestamp,
    snap_date: pd.Timestamp,
    existing_rows: list[dict],
) -> list[dict]:
    """Build option position rows from BUY/SELL txns dated AFTER txn_cutoff.

    Two distinct dates because the upstream synth roll-forward decouples
    the *display* date from the *real* statement date (see
    ``_detect_real_statement_cutoff``):

      * ``txn_cutoff`` — last real statement date. Transactions strictly
        after this are eligible for synthesis.
      * ``snap_date`` — display date for emitted rows. Equals
        ``as_of_ts`` so the synthesized rows line up with the statement
        carry-forwards in the same snapshot.

    Groups in-window option transactions by
    ``(account_id, opt_type, underlying, expiry, strike)`` and emits one
    row per group with positive net quantity. When a group matches a row
    already produced from the statement snapshot, mutates that row in
    place (pooled qty + pooled cost basis) instead of duplicating.

    Limitations:
      * Skips JPM rows whose broker desc omits the strike — those
        legacy-PDF BUY lines (``PUT SPY 12/18/26 ETF OPEN CONTRACT``)
        carry no key we can group on. In practice interim CSV BUYs
        carry the strike, so the gap only affects historical recovery.
      * Net qty < 0 (in-window short) is dropped; this parser doesn't
        synthesize naked-short positions out of nowhere.
    """
    if transactions is None or transactions.empty:
        return []
    if "settlement_date" not in transactions.columns:
        return []

    txns = transactions.copy()
    txns["settlement_date"] = pd.to_datetime(txns["settlement_date"])
    txns = txns[txns["settlement_date"] > txn_cutoff]
    if txns.empty:
        return []

    is_opt = txns.apply(
        lambda r: _is_option_buy(r) or _is_option_sell(r), axis=1
    )
    txns = txns[is_opt].copy()
    if txns.empty:
        return []

    txns["_parsed"] = [_parse_full_option_txn(r) for _, r in txns.iterrows()]
    txns = txns[txns["_parsed"].notna()].copy()
    if txns.empty:
        return []

    groups: dict[tuple, dict] = {}
    for _, r in txns.iterrows():
        p: ParsedOption = r["_parsed"]
        key = (r.get("account_id"), p.opt_type, p.underlying,
               p.expiry, p.strike)
        g = groups.setdefault(key, {
            "qty_buy": 0.0, "qty_sell": 0.0,
            "amt_buy": 0.0,
            "last_buy_price": float("nan"),
            "last_buy_date": pd.NaT,
            "last_buy_desc": None,
            "asset_class": ("option_put" if p.opt_type == "put"
                            else "option_call"),
        })
        qty = float(r.get("quantity") or 0.0)
        amt = r.get("amount")
        amt_f = float(amt) if pd.notna(amt) else 0.0
        if str(r.get("transaction_type") or "").lower() == "buy":
            g["qty_buy"] += qty
            g["amt_buy"] += amt_f
            sd = r["settlement_date"]
            if pd.isna(g["last_buy_date"]) or sd >= g["last_buy_date"]:
                g["last_buy_date"] = sd
                price = r.get("price")
                g["last_buy_price"] = (float(price) if pd.notna(price)
                                       else float("nan"))
                g["last_buy_desc"] = r.get("description")
        else:
            g["qty_sell"] += qty

    existing_index: dict[tuple, dict] = {}
    for r in existing_rows:
        if r.get("opt_type") is None or r.get("underlying") is None:
            continue
        key = (r.get("account_id"), r["opt_type"], r["underlying"],
               r.get("expiry"), r.get("strike"))
        existing_index[key] = r

    out: list[dict] = []
    for key, g in groups.items():
        account, opt_type, underlying, expiry, strike = key
        net_qty = g["qty_buy"] + g["qty_sell"]
        if net_qty <= 0:
            continue

        if g["qty_buy"] > 0:
            per_share = -g["amt_buy"] / (g["qty_buy"] * CONTRACT_MULT)
        else:
            per_share = float("nan")
        total_cost = (per_share * net_qty * CONTRACT_MULT
                      if pd.notna(per_share) else float("nan"))

        last_price = g["last_buy_price"]
        mv = (net_qty * CONTRACT_MULT * last_price
              if pd.notna(last_price) else 0.0)
        premium_mv = last_price if pd.notna(last_price) else float("nan")

        existing = existing_index.get(key)
        if existing is not None:
            old_qty = float(existing.get("quantity") or 0.0)
            new_qty = old_qty + net_qty
            old_total = existing.get("cost_basis_total")
            old_total_f = (float(old_total) if pd.notna(old_total) else 0.0)
            new_total = (old_total_f + total_cost
                         if pd.notna(total_cost) else float("nan"))
            existing["quantity"] = new_qty
            existing["cost_basis_total"] = new_total
            existing["cost_basis_per_share"] = (
                new_total / (new_qty * CONTRACT_MULT)
                if new_qty > 0 and pd.notna(new_total) else float("nan")
            )
            old_mv = float(existing.get("market_value") or 0.0)
            unit_mv = (old_mv / old_qty) if old_qty > 0 else 0.0
            existing["market_value"] = unit_mv * new_qty
            existing["premium_per_share_mv"] = (
                unit_mv / CONTRACT_MULT
                if unit_mv > 0 else float("nan")
            )
            continue

        out.append({
            "account_id":    account,
            "statement_date": snap_date,
            "asset_class":   g["asset_class"],
            "symbol":        underlying,
            "description":   g["last_buy_desc"],
            "opt_type":      opt_type,
            "underlying":    underlying,
            "expiry":        expiry,
            "strike":        strike,
            "quantity":      net_qty,
            "market_value":  mv,
            "premium_per_share_mv":  premium_mv,
            "cost_basis_per_share":  per_share,
            "cost_basis_total":      total_cost,
            "source":        "post_statement",
        })
    return out


def _empty_position_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "account_id", "statement_date", "asset_class", "symbol", "description",
        "opt_type", "underlying", "expiry", "strike", "quantity",
        "market_value", "premium_per_share_mv",
        "cost_basis_per_share", "cost_basis_total", "source",
    ])


def _find_fidelity_match(
    pos: pd.Series, txn_buys: pd.DataFrame,
) -> tuple[ParsedOption | None, str | None]:
    """Find the BUY transaction that opened a Fidelity position.

    Matches on (account_id, opt_type, underlying). When multiple buys
    match (e.g. partial closes followed by re-opens), prefers the most
    recent. Returns (parsed, "fidelity") on success, (None, None) when
    no match found.
    """
    if txn_buys is None or txn_buys.empty:
        return None, None
    pos_symbol = str(pos.get("symbol") or "").upper()
    pos_desc = str(pos.get("description") or "").upper()
    pos_acct = pos.get("account_id")
    # Position description is "PUT NVIDIA CORPORATION" — opt_type is the
    # leading word. Use it for matching even when strike/expiry are absent.
    type_m = re.match(r"^\s*(PUT|CALL)\b", pos_desc)
    if not type_m:
        return None, None
    pos_opt_type = type_m.group(1).lower()

    # Filter buys to those in the same account whose parsed underlying
    # matches the position's symbol (and parsed opt_type matches too).
    cands = txn_buys[txn_buys["account_id"] == pos_acct]
    if cands.empty:
        return None, None
    # Some Fidelity buy rows have NaN symbol — match on parsed underlying.
    cands = cands[cands["_parsed"].apply(
        lambda p: p is not None and p.underlying == pos_symbol
                  and p.opt_type == pos_opt_type
    )]
    if cands.empty:
        return None, None
    pick = cands.sort_values("settlement_date").iloc[-1]
    return pick["_parsed"], "fidelity"


def _cost_basis(
    pos: pd.Series, parsed: ParsedOption | None, txn_buys: pd.DataFrame,
) -> tuple[float, float]:
    """Recover cost basis from BUY transactions matching this position.

    Returns (per_share, total). NaN/NaN when no matching buy is found.
    The match is on (account_id, opt_type, underlying, expiry); strike is
    additionally required when the buy-side parse carries one (Fidelity).
    JPM buys omit strike from their description so strike is unmatched on
    that side.

    FALLBACK ONLY — the statement-stated ``cost_basis`` on the position row
    takes precedence in build_option_position_table. This pool cannot net
    sells (rolled-away lots keep counting) and, for strike-less JPM buys,
    charges the FULL pool to every row sharing the (acct, type, underlying,
    expiry) key — a roll into two strikes double-counts. Both limitations
    are pinned by RollCostBasisTests.
    """
    nan = float("nan")
    if parsed is None or txn_buys is None or txn_buys.empty:
        return nan, nan

    def _matches(p: ParsedOption | None) -> bool:
        if p is None:
            return False
        if (p.opt_type != parsed.opt_type
                or p.underlying != parsed.underlying
                or p.expiry != parsed.expiry):
            return False
        # Strike must match when the buy side parsed one (Fidelity).
        # JPM buys parse to NaN strike; accept any.
        import math as _m
        if not _m.isnan(p.strike) and p.strike != parsed.strike:
            return False
        return True

    cands = txn_buys[
        (txn_buys["account_id"] == pos.get("account_id"))
        & txn_buys["_parsed"].apply(_matches)
    ]
    if cands.empty:
        return nan, nan
    # Sum cost across all matching buys (handles partial re-adds). amount
    # is negative for buys; flip sign so cost is positive.
    amts = pd.to_numeric(cands["amount"], errors="coerce")
    qtys = pd.to_numeric(cands["quantity"], errors="coerce")
    if amts.isna().all() or qtys.isna().all():
        return nan, nan
    total_cost = float(-amts.fillna(0).sum())
    total_contracts = float(qtys.fillna(0).sum())
    if total_contracts <= 0:
        return nan, nan
    per_share = total_cost / (total_contracts * CONTRACT_MULT)
    return per_share, total_cost


def main(argv: list[str] | None = None) -> int:
    """Diagnostic CLI: dump the parsed table for the latest snapshot."""
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--as-of", default=None,
                    help="Statement date to extract (YYYY-MM-DD). "
                         "Default: latest in positions.csv.")
    args = ap.parse_args(argv)

    positions = pd.read_csv(DATA / "positions.csv",
                            parse_dates=["statement_date"])
    transactions = pd.read_csv(DATA / "transactions.csv",
                               parse_dates=["settlement_date"])
    interim_path = DATA / "transactions_interim.csv"
    if interim_path.exists():
        interim = pd.read_csv(interim_path, parse_dates=["settlement_date"])
        if not interim.empty:
            transactions = pd.concat(
                [transactions, interim], ignore_index=True
            )

    as_of = (datetime.strptime(args.as_of, "%Y-%m-%d").date()
             if args.as_of else None)
    tbl = build_option_position_table(positions, transactions, as_of=as_of)

    if tbl.empty:
        print("No option positions found.")
        return 0

    show = tbl[["account_id", "statement_date", "underlying", "opt_type",
                "strike", "expiry", "quantity", "market_value",
                "cost_basis_per_share", "premium_per_share_mv", "source"]]
    show["statement_date"] = show["statement_date"].dt.date
    print(show.to_string(index=False))
    print()
    n_parsed = int((tbl["source"] != "unparsed").sum())
    print(f"Parsed: {n_parsed}/{len(tbl)} ({tbl['source'].value_counts().to_dict()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
