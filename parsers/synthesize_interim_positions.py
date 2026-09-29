"""
Synthesize end-of-period positions by rolling the latest statement positions
forward with interim CSV transactions.

DESIGN: idempotent and transient (mirrors parsers/ingest_csv_activity.py).
  - Reads `data/positions.csv` and `data/transactions_interim.csv` each run.
  - Per (account, key), aggregates interim transactions into delta quantity +
    delta cash and applies to that account's OWN latest statement positions.
    `key` is `symbol` for stocks (e.g. "T"), `cusip` for bonds (where symbol
    is NaN). New (account, key) combinations get fresh rows. Basing each
    account on its own latest statement — not the broker's global latest — is
    load-bearing: an account lagging the newest broker statement (e.g. a
    Alpine sleeve with no May statement while the others have one) must
    still roll forward, or its interim trades attach to nothing and the whole
    account collapses to the net of those trades.
  - Option legs arrive as contract symbols that never match the statement's
    bare-underlying option rows by key — Alpine OCC ("-SPY261218P575") or
    Harbor display format ("SPY DEC 26 PUT 650.00"). They're mapped onto the
    statement row for the SAME CONTRACT (underlying + put/call + expiry +
    strike, read from whichever side spells it out; a close with no exact
    match falls back to the first free same-underlying row): a close (net
    qty < 0) reduces that row and scales its cost basis, an add-on buy pools
    into it, and an open with no base row books a fresh option_put/option_call
    row at premium. Quantity is authoritative for option legs whatever the
    ingest type — the `other`-typed expirations / journals through which
    contracts leave an account with no cash (Alpine "EXPIRED ...", Harbor
    "Journal") close them too. (Aug 2026: Harbor legs used to fall into the
    equity path as "brand-new symbols", booking negative phantom rows while
    the statement puts were carried forward; expired calls lingered at cost.)
  - Cash positions (asset_class == "cash") absorb the net per-account cash
    impact of every interim transaction. An account that takes in interim cash
    but holds no cash position gets a synthesized $1-NAV cash row so the money
    isn't silently dropped.
  - `statement_date` on rolled rows = unified snapshot = max(interim
    settlement_date across all brokers). A broker whose latest interim row
    pre-dates this snapshot is assumed to have had no further activity
    (the user downloaded the CSV today; the gap means a quiet stretch).
  - market_value = new_quantity * existing price (no mark-to-market guess).
    For brand-new symbols, market_value = new_quantity * latest interim
    transaction price.
  - cost_basis: unchanged for existing rows; for new rows it's the sum of
    |amount| across the buying transactions.
  - asset_class for brand-new rows is inferred (no statement section header to
    copy): inherit the security's class from the book's own statement history
    by symbol/cusip key; else a Treasury description -> fixed_income; else a
    plain ticker -> equity_stock (display reclass still maps known ETFs and
    commodities on top). Unrecognized cusip-only rows keep the honest
    "other" placeholder.

Per-transaction rules (applied to position quantity and account cash):
  buy / sell / reinvestment / stock_split:  quantity += tx.quantity (signed)
  merger / redemption (cash-out shape):     quantity += tx.quantity; a leg
                                            printed under a cusip the statement
                                            never carried for the security (the
                                            EA cash merger, Aug 2026) closes the
                                            symbol-keyed row its description
                                            names (strict unique name match,
                                            qty-sane) instead of spawning a
                                            negative phantom row
  other (option leg with a quantity):       quantity += tx.quantity (expiry /
                                            journal — no cash)
  dividend / interest / other / etc.:       no quantity change
  exchange:                                 skipped entirely (no qty, no cash)
  other (security exchange leg): a Alpine fund-exchange leg mis-typed `other`
    but carrying a real quantity + non-zero amount + security key (not an option)
    is applied as a buy/sell — quantity participates and cash is taken from the
    trade DIRECTION (buy=cash out), since the broker's `amount` sign on these
    legs is unreliable (WSD-4). Carnival NaN-amount renames, $0-amount lending/
    collateral placeholders, internal-transfer-tagged rows, and cash journals are
    excluded by `_is_security_exchange_leg` and stay no-op. Balanced round-trips
    (securities-lending in/out legs) DO match but net to 0 shares / 0 cash.
  Every type EXCEPT `exchange` contributes its `amount` to the per-account
  cash delta — including `other`, which carries internal-pair cash transfers
  (Harbor Journal / Alpine transfer pairs) that net to zero portfolio-wide.

Run modes:
  python parsers/synthesize_interim_positions.py             # dry-run + samples
  python parsers/synthesize_interim_positions.py --write     # emit positions_interim.csv
"""
import re
import sys
import argparse
from pathlib import Path

import pandas as pd

from lot_engine import normalize_security_name, security_name_from_description

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
POSITIONS_CSV = DATA_DIR / "positions.csv"
INTERIM_TXN_CSV = DATA_DIR / "transactions_interim.csv"
OUT_CSV = DATA_DIR / "positions_interim.csv"

# Transaction types that move the security's quantity.
QTY_TYPES = {"buy", "sell", "reinvestment", "stock_split"}
# Transaction types skipped entirely (no cash, no quantity).
# `exchange` is a corp-action share-for-share swap where the "FROM" leg may not
# be held under a matching identifier at Apr 30, and we have no live price for
# the new symbol — better to omit than to fabricate. `other` is NOT skipped:
# it covers internal-pair cash transfers (Harbor Journal, Alpine transfer pairs)
# where we want the cash side, but its few corp-action members (Carnival paired
# stock) have amount=NaN so they no-op naturally on the cash side and we leave
# their quantity alone (Carnival share-class rename has no economic effect).
SKIP_TYPES = {"exchange"}

# Corporate-action out-legs that remove shares FOR CASH — a cash merger
# ("<NAME> ... CMR $<px>P/S") or a redemption / maturity. They join the
# quantity path only in the cash-out shape (quantity < 0, amount > 0); a
# share-for-share leg typed merger or a leg with no amount stays
# quantity-inert. Without this, a row typed `redemption` would be carried
# forward AND land its face in cash — a double count.
CORP_OUT_TYPES = {"merger", "redemption"}

# Name-rescue rule (ported from the lot engine's corporate-action key rescue,
# spec 2026-07-30 §3.3): the run of leading tokens a row's security name
# shares with a statement description decides, strict unique maximum.
_RESCUE_MIN_RUN = 2
_RESCUE_SINGLE_TOKEN_CHARS = 6
_QTY_EPS = 1e-6


def _key_for(symbol: object, cusip: object) -> str | None:
    """Identify a position by symbol-or-cusip. Bonds typically have NaN symbol."""
    if isinstance(symbol, str) and symbol.strip():
        return f"SYM:{symbol.strip()}"
    if isinstance(cusip, str) and cusip.strip():
        return f"CUSIP:{cusip.strip()}"
    return None


# OCC-style option symbol, e.g. "-SPY261218P575" => underlying SPY, put.
# The leading "-" and short (non-zero-padded) strike are Alpine activity-CSV
# quirks; the date is always 6 digits and the type is the C/P before the strike.
_OCC_RE = re.compile(r"^-?([A-Za-z.]+)(\d{6})([CP])(\d+(?:\.\d+)?)$")
# Harbor display-format option symbol (the interim CSV's Ticker column), e.g.
# "SPY DEC 26 PUT 650.00" — expiry MONTH only; the day rides in the description.
_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_DISPLAY_LEG_RE = re.compile(
    r"^([A-Z][A-Z0-9.]*)\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+"
    r"(\d{2})\s+(CALL|PUT)\s+([\d.,]+)$")
# The full contract as brokers spell it in a description — Harbor position and
# activity rows ("PUT SPY 12/18/26 650 STATE STREET ...") and the OCC symbol
# Alpine appends to its statement rows ("... (100 SHS) (NVDA261218P115)").
_Harbor_DESC_RE = re.compile(
    r"^\s*(PUT|CALL)\s+([A-Z][A-Z0-9.]*)\s+(\d{2})/(\d{2})/(\d{2})\s+"
    r"(\d+(?:\.\d+)?)\b", re.I)
_FID_DESC_OCC_RE = re.compile(r"\(([A-Z][A-Z.]*)(\d{6})([CP])(\d+(?:\.\d+)?)\)")

# Treasury description prefix (both "TREASURY" and the statement's truncated
# "TREAS" spellings) — the same signal _display_symbol keys its UST label on.
_UST_DESC_RE = re.compile(r"UNITED STATES TREAS", re.I)


def _inherited_class_map(positions: pd.DataFrame) -> dict[str, str]:
    """key -> asset_class of the most recent statement row holding that key.

    Option classes never inherit (statement option rows carry the bare
    underlying as `symbol`, so a bare-symbol interim buy keyed "SYM:SPY" is
    the underlying, not the derivative); "other" carries no information.
    """
    df = positions[positions["asset_class"].notna()]
    cls = df["asset_class"].astype(str)
    df = df[~cls.str.startswith("option") & (cls != "other")]
    out: dict[str, str] = {}
    for r in df.sort_values("statement_date").to_dict("records"):
        key = _key_for(r.get("symbol"), r.get("cusip"))
        if key is not None:
            out[key] = str(r["asset_class"])
    return out


def _new_row_asset_class(key: str, tx: dict, inherited: dict[str, str]) -> str:
    """Asset class for a brand-new interim position (no statement row to copy).

    First match wins: inherit the security's class from the book's own
    statement history; a Treasury description -> fixed_income; a plain ticker
    -> equity_stock (the display reclass layer still maps known ETFs /
    commodity tickers on top); else the honest "other" placeholder until the
    next statement lands. Option legs never reach here — both symbol shapes
    route through the contract-aware option path (`_parse_option_leg`).
    """
    hit = inherited.get(key)
    if hit is not None:
        return hit
    desc = tx.get("description")
    if isinstance(desc, str) and _UST_DESC_RE.search(desc):
        return "fixed_income"
    if key.startswith("SYM:"):
        return "equity_stock"
    return "other"


def _parse_option_leg(symbol: object) -> tuple[str, str] | None:
    """Parse an interim option-leg symbol into (underlying, "put"|"call").

    Statement option rows carry the bare underlying as `symbol` (e.g. "SPY")
    with asset_class option_put/option_call, but interim activity rows carry
    a CONTRACT symbol — Alpine OCC ("-SPY261218P575") or Harbor display format
    ("SPY DEC 26 PUT 650.00") — so they never match by `_key_for`. Returns
    None for anything that isn't an option contract symbol (plain tickers,
    bonds)."""
    if not isinstance(symbol, str):
        return None
    s = symbol.strip()
    m = _OCC_RE.match(s)
    if m:
        return m.group(1).upper(), ("call" if m.group(3).upper() == "C" else "put")
    m = _DISPLAY_LEG_RE.match(s.upper())
    if m:
        return m.group(1), ("call" if m.group(4) == "CALL" else "put")
    return None


def _contract_key(symbol: object, description: object) -> tuple | None:
    """(underlying, type, year, month, day|None, strike) identifying ONE
    contract, read from whichever of symbol / description spells it out:
    a Harbor description ("PUT SPY 12/18/26 650 ..."), the OCC symbol Alpine
    appends to statement descriptions ("(NVDA261218P115)") or carries as the
    activity symbol ("-NVDA261218P115"), else the Harbor display symbol — whose
    expiry is month-only, so ``day`` is None there."""
    desc = description if isinstance(description, str) else ""
    m = _Harbor_DESC_RE.match(desc)
    if m:
        return (m.group(2).upper(),
                "put" if m.group(1).upper() == "PUT" else "call",
                2000 + int(m.group(5)), int(m.group(3)), int(m.group(4)),
                float(m.group(6)))
    sym = symbol.strip() if isinstance(symbol, str) else ""
    m = _FID_DESC_OCC_RE.search(desc) or _OCC_RE.match(sym)
    if m:
        ymd = m.group(2)
        return (m.group(1).upper(),
                "call" if m.group(3).upper() == "C" else "put",
                2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]),
                float(m.group(4)))
    m = _DISPLAY_LEG_RE.match(sym.upper())
    if m:
        return (m.group(1), "call" if m.group(4) == "CALL" else "put",
                2000 + int(m.group(3)), _MONTHS[m.group(2)], None,
                float(m.group(5).replace(",", "")))
    return None


def _same_contract(a: tuple | None, b: tuple | None) -> bool:
    """Exact match on underlying / type / year / month / strike; the expiry
    day is compared only when both sides know it."""
    if a is None or b is None:
        return False
    if a[:4] != b[:4] or abs(a[5] - b[5]) > 1e-9:
        return False
    return a[4] is None or b[4] is None or a[4] == b[4]


def _is_security_exchange_leg(r: dict) -> bool:
    """True for an `other`-typed row that is really a security buy/sell leg
    mis-typed at ingest (e.g. a Alpine fund exchange): qty-bearing, a real
    NON-ZERO cash amount, a security key, not an option, not an internal
    transfer. The action that would name it ("EXCHANGE"/"TRANSFERRED"/...) is
    lost at ingest, so this is a structural heuristic, not an exact classifier:

      * Excluded: Carnival share-class renames (amount=NaN), cash journals
        (no security key), $0-amount lending/collateral placeholders
        (e.g. "COLLATERAL DELV..."), and ingest-tagged internal transfers.
      * Balanced `other` round-trips (securities-lending in/out legs) DO match
        but net to 0 shares / 0 cash — harmless.
      * KNOWN LIMITATION: an UNPAIRED external in-kind share transfer (shares
        arrive, no cash) would be booked here as a cash purchase (a phantom
        outflow). This is transient — it self-heals when the next statement
        lands and replaces the synthesized snapshot — and cannot be told apart
        from a real purchase per-row, so it is accepted, not silently hidden."""
    if r.get("transaction_type") != "other":
        return False
    if str(r.get("flow_scope") or "").strip().lower() == "internal":
        return False
    q, a = r.get("quantity"), r.get("amount")
    if pd.isna(q) or q == 0 or pd.isna(a) or a == 0:
        return False
    if _key_for(r.get("symbol"), r.get("cusip")) is None:
        return False
    return _parse_option_leg(r.get("symbol")) is None


def _is_corporate_out_leg(r: dict) -> bool:
    """True for a merger / redemption row in the cash-out shape
    (shares leave: quantity < 0; cash arrives: amount > 0)."""
    if r.get("transaction_type") not in CORP_OUT_TYPES:
        return False
    q, a = r.get("quantity"), r.get("amount")
    if pd.isna(q) or pd.isna(a):
        return False
    return float(q) < 0 and float(a) > 0


# Mirrors the leading-run scoring of lot_engine's _name_match (the rule's source of truth).
def _leading_overlap(a: list[str], b: list[str]) -> int:
    """Length of the common leading token run."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _rescue_by_name(records: list[dict], account: str, description: object,
                    required_qty: float, claimed: set[int],
                    blockers: tuple[str, ...] | list[str] = ()) -> tuple:
    """Verdict for the ONE statement row in `account` that a
    corporate-action description names: ("match", idx) | ("claimed", idx)
    | ("blocker", desc) | ("refuse", None). (Spec Update 2026-08-23.)

    Brokers print a cash-merger / redemption out-leg under the identifier
    the action left behind — a cusip the statement never printed for the
    equity (Harbor equities are symbol-keyed on statements) — so the exact
    key misses. Both sides print the security NAME: match on the run of
    leading tokens (row side: the text before the first cusip-shaped
    token; statement side: the whole description), strict unique maximum
    with a run of >= 2 tokens, or exactly 1 token of >= 6 characters
    (issuer words; short generic leaders cannot rescue). The name contest
    decides the target and no runner-up is ever promoted (a silent
    wrong-position close is the one outcome worse than the visible
    fresh-row artifact the caller falls back to). The four verdicts:

    - ("blocker", desc): exactly ONE in-window symbol-keyed position with
      no statement row out-names the statement winner (STRICT run
      inequality — the merged shares are more likely the fresh ones); the
      caller nets into that fresh position, quantity-gated there. A tie
      with the winner, or two strictly-better blockers, refuses — which
      holding was merged is not decisive.
    - ("claimed", idx): the unique winner was already moved by an earlier
      exact-key delta; the caller accumulates onto the EMITTED row,
      quantity-gated there (the base quantity here is stale).
    - ("match", idx): the unique winner, unclaimed, holding the shares.
    - ("refuse", None): everything else — the caller keeps today's
      NAV-neutral fresh row.

    Only SYMBOL-keyed, non-option, non-cash rows are candidates: two
    different cusips are two different instruments, never a drift to
    bridge.
    """
    row_tokens = security_name_from_description(description).split()
    if not row_tokens:
        return ("refuse", None)
    scores: dict[int, int] = {}
    for i, rec in enumerate(records):
        if rec.get("account_id") != account:
            continue
        aclass = str(rec.get("asset_class") or "")
        if aclass.startswith("option") or aclass == "cash":
            continue
        rec_key = _key_for(rec.get("symbol"), rec.get("cusip"))
        if rec_key is None or not rec_key.startswith("SYM:"):
            continue
        run = _leading_overlap(
            row_tokens, normalize_security_name(rec.get("description")).split())
        if run <= 0:
            continue
        scores[i] = run
    if not scores:
        return ("refuse", None)
    best = max(scores.values())
    winners = [i for i, s in scores.items() if s == best]
    if len(winners) != 1:
        return ("refuse", None)
    if not (best >= _RESCUE_MIN_RUN or (
            best == 1 and len(row_tokens[0]) >= _RESCUE_SINGLE_TOKEN_CHARS)):
        return ("refuse", None)
    blocker_runs = [(b, _leading_overlap(
        row_tokens, normalize_security_name(b).split())) for b in blockers]
    better = [b for b, r in blocker_runs if r > best]
    if len(better) == 1:
        # The stronger name wins even over a claimed winner: the merged
        # shares are more likely the fresh in-window position's.
        return ("blocker", better[0])
    if better or any(r == best for _, r in blocker_runs):
        # Two strictly-better blockers, or a tie with the winner: which
        # holding was merged is not decisive (final-review veto survives).
        return ("refuse", None)
    if winners[0] in claimed:
        # Already moved by an earlier delta — the caller applies this leg
        # ON TOP of the emitted row (never from the stale base here).
        return ("claimed", winners[0])
    held = records[winners[0]].get("quantity")
    held = float(held) if pd.notna(held) else 0.0
    if required_qty > held + _QTY_EPS:
        # The named row cannot supply the shares: refuse — no sibling is
        # promoted (stricter than the lot engine's filter-first rule, on
        # purpose).
        return ("refuse", None)
    return ("match", winners[0])


def synthesize_interim_positions(
    positions: pd.DataFrame, interim: pd.DataFrame
) -> pd.DataFrame:
    """Roll the latest statement positions forward with interim transactions.

    Returns a DataFrame with the same columns as `positions`, containing one
    row per (broker, account, key) that exists at the unified snapshot date.
    Existing-symbol rows reflect aggregated interim deltas; brand-new symbol
    rows are fresh. Cash positions absorb the net per-account cash impact.
    Empty DataFrame if `interim` is empty.
    """
    if interim.empty:
        return positions.iloc[0:0].copy()

    snapshot_date = pd.Timestamp(interim["settlement_date"].max()).normalize()
    # Security-class inheritance for brand-new rows, from the WHOLE book's
    # statement history (class is a property of the security, not the account).
    inherited_class = _inherited_class_map(positions)

    out_rows: list[dict] = []
    for broker, bro_pos_all in positions.groupby("broker"):
        # Base each account on ITS OWN latest statement, not the broker's
        # global latest. An account lagging the newest broker statement (e.g.
        # Alpine issued May statements for most accounts but never one for
        # an individual-stocks sleeve) must still roll its last-known
        # holdings forward. Filtering to the broker-global max date instead
        # drops the lagging account's base, so its interim trades attach to
        # nothing and the whole account collapses to the net of those trades
        # (the lagging-sleeve disappearance, Jun 2026).
        acct_latest_date = bro_pos_all.groupby("account_id")["statement_date"].transform("max")
        bro_latest = bro_pos_all[bro_pos_all["statement_date"] == acct_latest_date].copy()
        # Per-account base statement date — interim rows at/before this are
        # already reflected in the statement (see the stale-activity guard in
        # the transaction loop below).
        acct_base_date = bro_pos_all.groupby("account_id")["statement_date"].max()
        # An account whose latest statement is at/after the interim snapshot
        # is superseded: rolling it "forward" would emit a backdated copy of
        # the statement book at snapshot_date, and every month-sliced consumer
        # (composition, option table, health extracted totals) would then
        # count the account twice (the Jun 2026 double-count).
        superseded = frozenset(
            acct_base_date[acct_base_date >= snapshot_date].index
        )
        if superseded:
            bro_latest = bro_latest[
                ~bro_latest["account_id"].isin(superseded)
            ].copy()
        bro_interim = interim[interim["broker"] == broker].copy()
        if bro_interim.empty:
            # Re-label all latest positions to the snapshot date so they appear
            # in the unified view; nothing changed for this broker.
            relabeled = bro_latest.copy()
            relabeled["statement_date"] = snapshot_date
            out_rows.extend(relabeled.to_dict("records"))
            continue

        bro_latest["_key"] = bro_latest.apply(
            lambda r: _key_for(r["symbol"], r["cusip"]), axis=1
        )
        # Index by ROW POSITION (not key) so SPY-equity vs SPY-option in the
        # same account never collide. pos_index_map collects candidate row
        # positions per (account, key); the resolver below prefers non-option
        # rows when an interim trade key matches both an underlying and a
        # derivative on it.
        pos_index_map: dict[tuple[str, str], list[int]] = {}
        for i, r in enumerate(bro_latest.to_dict("records")):
            if r["_key"] is not None:
                pos_index_map.setdefault((r["account_id"], r["_key"]), []).append(i)
        bro_records = bro_latest.to_dict("records")

        def _resolve_position_index(account: str, key: str) -> int | None:
            candidates = pos_index_map.get((account, key), [])
            if not candidates:
                return None
            # Prefer rows whose asset_class doesn't start with "option" —
            # interim trades target the underlying, not the derivative.
            non_option = [
                i for i in candidates
                if not str(bro_records[i].get("asset_class", "")).startswith("option")
            ]
            return non_option[0] if non_option else candidates[0]

        # Per-account cash delta accumulator.
        cash_delta_per_account: dict[str, float] = {}
        qty_delta: dict[tuple[str, str], float] = {}
        latest_tx_for_new: dict[tuple[str, str], dict] = {}
        cost_basis_for_new: dict[tuple[str, str], float] = {}
        # (account, key) -> the OUT-LEG's own description, for cusip-keyed
        # corporate-action legs (shares leaving) — eligible for the name
        # rescue when the exact key finds no statement row.
        # (two out-legs under one cusip key: the last in file order wins —
        # both name the same security)
        rescue_eligible: dict[tuple[str, str], str] = {}
        # Option legs, keyed by (account, OCC symbol) — handled apart from the
        # equity path because their symbols don't match statement option rows.
        opt_delta: dict[tuple[str, str], float] = {}
        opt_meta: dict[tuple[str, str], tuple[str, str]] = {}
        opt_tx: dict[tuple[str, str], dict] = {}
        opt_contract: dict[tuple[str, str], tuple] = {}   # best-known contract key
        opt_cost: dict[tuple[str, str], float] = {}       # Σ|amount| of the buys

        for r in bro_interim.to_dict("records"):
            ttype = r["transaction_type"]
            if ttype in SKIP_TYPES:
                continue
            account = r["account_id"]
            if account in superseded:
                # The statement covers the whole interim window; even undated
                # rows (which the stale-activity guard below keeps) must not
                # spawn rows for a superseded account.
                continue
            # Skip activity at/before this account's base statement — it's
            # already reflected there. The interim CSV is a rolling window that
            # reaches back before the latest statement (e.g. a dividend-
            # reinvestment on a since-sold position); re-applying those rows
            # resurrects phantom holdings and double-counts cash. settlement_date
            # is authoritative; fall back to trade_date when the broker leaves it
            # blank (Alpine DRIP rows). Unknown date, or an account with no
            # base statement -> keep (don't drop genuinely new activity).
            base_d = acct_base_date.get(account)
            eff_d = r.get("settlement_date")
            if pd.isna(eff_d):
                eff_d = r.get("trade_date")
            if (base_d is not None and pd.notna(eff_d)
                    and pd.Timestamp(eff_d) <= pd.Timestamp(base_d)):
                continue
            key = _key_for(r["symbol"], r["cusip"])

            is_leg = _is_security_exchange_leg(r)
            is_leg_buy = is_leg and float(r["quantity"]) > 0
            is_corp_out = _is_corporate_out_leg(r)

            amt = r.get("amount")
            if pd.notna(amt):
                if is_leg:
                    # Buy (+qty) -> cash out; sell (-qty) -> cash in. Use the
                    # trade direction, not the mis-signed `amount`.
                    contribution = (-abs(float(amt)) if is_leg_buy
                                    else abs(float(amt)))
                else:
                    contribution = float(amt)
                cash_delta_per_account[account] = (
                    cash_delta_per_account.get(account, 0.0) + contribution
                )

            # Option leg: route to the contract-aware apply step (below)
            # rather than the bare-symbol equity path, so it maps onto the
            # right statement row instead of fabricating a phantom row keyed
            # on the contract symbol (the Aug 2026 negative Harbor puts).
            leg = _parse_option_leg(r["symbol"])
            if leg is not None:
                qd = r.get("quantity")
                # Quantity is authoritative for option legs whatever the
                # ingest type: buys/sells, AND the `other`-typed expirations /
                # journals through which contracts leave an account with no
                # cash (Alpine "EXPIRED ..." qty<0 amount 0, Harbor "Journal"
                # qty<0 amount NaN). Ignoring those left expired calls on the
                # book at cost.
                if ((ttype in QTY_TYPES or ttype == "other")
                        and pd.notna(qd) and qd != 0):
                    okey = (account, r["symbol"].strip())
                    opt_delta[okey] = opt_delta.get(okey, 0.0) + float(qd)
                    opt_meta[okey] = leg
                    ck = _contract_key(r.get("symbol"), r.get("description"))
                    prev_ck = opt_contract.get(okey)
                    if ck is not None and (prev_ck is None or prev_ck[4] is None):
                        opt_contract[okey] = ck
                    if float(qd) > 0 and pd.notna(amt):
                        opt_cost[okey] = opt_cost.get(okey, 0.0) + abs(float(amt))
                    prev = opt_tx.get(okey)
                    if prev is None or r["settlement_date"] >= prev["settlement_date"]:
                        opt_tx[okey] = r
                continue

            if (ttype in QTY_TYPES or is_leg or is_corp_out) and key is not None:
                qd = r.get("quantity")
                if pd.notna(qd) and qd != 0:
                    qty_delta[(account, key)] = (
                        qty_delta.get((account, key), 0.0) + float(qd)
                    )
                    if (key.startswith("CUSIP:") and float(qd) < 0
                            and (is_leg or is_corp_out)):
                        desc = r.get("description")
                        rescue_eligible[(account, key)] = (
                            desc if isinstance(desc, str) else "")
                    prev = latest_tx_for_new.get((account, key))
                    if prev is None or r["settlement_date"] >= prev["settlement_date"]:
                        latest_tx_for_new[(account, key)] = r
                    if ((ttype in {"buy", "reinvestment"} or (is_leg and qd > 0))
                            and pd.notna(amt)):
                        cost_basis_for_new[(account, key)] = (
                            cost_basis_for_new.get((account, key), 0.0) + abs(float(amt))
                        )

        # Apply quantity deltas. Track row indices touched (not keys), so
        # collision rows still get carried forward by the catch-all loop.
        # In-window symbol-keyed positions with no statement row (fresh buys
        # the statement never saw), per account: not rescue targets (they are
        # not statement rows) but BLOCKERS of a rescue whose name they match
        # at least as well — the merged shares may be theirs.
        fresh_names: dict[str, list[str]] = {}
        # description -> the ONE fresh (account, key) it identifies, for
        # the blocker-net path; a description two fresh keys share maps to
        # None (ambiguous — the net falls back to the fresh-row artifact).
        fresh_key_by_desc: dict[tuple[str, str], tuple[str, str] | None] = {}
        for (acct, k), tx in latest_tx_for_new.items():
            if k.startswith("SYM:") and _resolve_position_index(acct, k) is None:
                d = tx.get("description")
                if isinstance(d, str) and d.strip():
                    fresh_names.setdefault(acct, []).append(d)
                    dk = (acct, d)
                    fresh_key_by_desc[dk] = (None if dk in fresh_key_by_desc
                                             else (acct, k))
        touched_indices: set[int] = set()
        emitted_at: dict[int, int] = {}                 # bro_records idx -> out_rows pos
        fresh_at: dict[tuple[str, str], int] = {}       # (account, key) -> out_rows pos

        def _land_on_emitted(pos, dq) -> bool:
            """Apply dq onto an already-emitted out_rows row when it can
            supply the shares (spec Update 2026-08-23 A/B); False -> the
            caller falls back to today's fresh-row artifact."""
            if pos is None:
                return False
            row = out_rows[pos]
            old_qty = float(row["quantity"])
            if abs(dq) > old_qty + _QTY_EPS:
                return False
            old_mv = float(row["market_value"])
            unit_value = (old_mv / old_qty) if old_qty else 0.0
            row["quantity"] = old_qty + dq
            row["market_value"] = row["quantity"] * unit_value
            return True

        def _rescue_candidate(item: tuple[tuple[str, str], float]) -> bool:
            (acct, k), d = item
            return (d < 0 and (acct, k) in rescue_eligible
                    and _resolve_position_index(acct, k) is None)

        # Exact-key deltas apply first; rescue candidates run LAST (stable
        # sort) so a rescue never targets a row an exact-key delta already
        # moved — the same row would otherwise be emitted twice from its
        # stale base (review catch 2026-08-22). A rescue refused for that
        # reason falls through to the fresh-row path like any other.
        for (account, key), dq in sorted(qty_delta.items(),
                                         key=_rescue_candidate):
            idx = _resolve_position_index(account, key)
            if idx is None and _rescue_candidate(((account, key), dq)):
                # Cusip-keyed out-leg with no statement row under that cusip:
                # close the symbol-keyed row the description names. Verdicts
                # (spec Update 2026-08-23): "match" closes the untouched
                # statement row; "claimed" accumulates onto the row an
                # exact-key delta already emitted; "blocker" nets into the
                # fresh in-window position that out-names the statement
                # winner. Each landing is share-gated; anything ungated or
                # refused falls through to today's fresh-row path (the
                # visible netting artifact beats a silent skip, which would
                # double count the proceeds already landing in cash).
                verdict, payload = _rescue_by_name(
                    bro_records, account, rescue_eligible[(account, key)],
                    abs(dq), touched_indices,
                    blockers=fresh_names.get(account, ()))
                if verdict == "match":
                    idx = payload
                elif verdict == "claimed":
                    if _land_on_emitted(emitted_at.get(payload), dq):
                        continue
                elif verdict == "blocker":
                    fk = fresh_key_by_desc.get((account, payload))
                    if fk is not None and _land_on_emitted(
                            fresh_at.get(fk), dq):
                        continue
            if idx is not None:
                touched_indices.add(idx)
                existing = bro_records[idx]
                old_qty = float(existing["quantity"])
                old_mv = float(existing["market_value"])
                unit_value = (old_mv / old_qty) if old_qty else 0.0
                new_qty = old_qty + dq
                row = dict(existing)
                row["quantity"] = new_qty
                row["market_value"] = new_qty * unit_value
                row["statement_date"] = snapshot_date
                row.pop("_key", None)
                out_rows.append(row)
                emitted_at[idx] = len(out_rows) - 1
            else:
                # Brand-new symbol: build a fresh row from the latest tx.
                tx = latest_tx_for_new[(account, key)]
                tx_qty = float(tx["quantity"]) if pd.notna(tx["quantity"]) else 0.0
                tx_amt = float(tx["amount"]) if pd.notna(tx["amount"]) else 0.0
                unit_value = (abs(tx_amt) / abs(tx_qty)) if tx_qty else 0.0
                price = float(tx["price"]) if pd.notna(tx["price"]) else 0.0
                template = bro_latest[bro_latest["account_id"] == account].head(1)
                account_type = (
                    template["account_type"].iloc[0] if not template.empty else ""
                )
                new_row = {
                    "statement_date":     snapshot_date,
                    "broker":             broker,
                    "account_id":         account,
                    "account_type":       account_type,
                    "symbol":             tx["symbol"],
                    "cusip":              tx["cusip"],
                    "description":        tx["description"],
                    "asset_class":        _new_row_asset_class(key, tx, inherited_class),
                    "quantity":           dq,
                    "price":              price,
                    "market_value":       dq * unit_value,
                    "cost_basis":         cost_basis_for_new.get((account, key), 0.0),
                    "unrealized_gl":      0.0,
                    "est_annual_income":  0.0,
                    "currency":           "USD",
                    "source_file":        "synthesize_interim_positions",
                }
                out_rows.append(new_row)
                fresh_at[(account, key)] = len(out_rows) - 1

        # Apply option-leg deltas onto the statement row for the SAME CONTRACT
        # (underlying + put/call + expiry + strike, from whichever side spells
        # it out). A close (net qty < 0) reduces that row and scales its cost
        # basis, an add-on buy pools into it, and an open with no base row
        # books a fresh option row at premium. Either way the leg is a real
        # option_put/option_call row, never an "other" row. A close with no
        # exact match falls back to the first free same-underlying+type row
        # (statement rows can aggregate by underlying, losing the strike).
        claimed_opt: set[int] = set()
        for okey, dq in opt_delta.items():
            if abs(dq) < 1e-9:
                continue          # round-tripped inside the window: nothing to book
            account = okey[0]
            underlying, otype = opt_meta[okey]
            aclass = f"option_{otype}"
            tx = opt_tx[okey]
            tq = float(tx["quantity"]) if pd.notna(tx["quantity"]) else 0.0
            ta = float(tx["amount"]) if pd.notna(tx["amount"]) else 0.0
            premium = (abs(ta) / abs(tq)) if tq else 0.0
            contract = opt_contract.get(okey)
            candidates = [
                i for i, rec in enumerate(bro_records)
                if (i not in claimed_opt and i not in touched_indices
                    and rec["account_id"] == account
                    and str(rec.get("asset_class", "")) == aclass
                    and isinstance(rec.get("symbol"), str)
                    and rec["symbol"].strip() == underlying)]
            base_idx = next(
                (i for i in candidates
                 if _same_contract(contract, _contract_key(
                     bro_records[i].get("symbol"),
                     bro_records[i].get("description")))),
                None)
            if base_idx is None and dq < 0 and candidates:
                base_idx = candidates[0]
            if base_idx is not None:
                claimed_opt.add(base_idx)
                touched_indices.add(base_idx)
                existing = bro_records[base_idx]
                old_qty = float(existing["quantity"])
                old_mv = float(existing["market_value"])
                unit_value = (old_mv / old_qty) if old_qty else premium
                new_qty = old_qty + dq
                row = dict(existing)
                row["quantity"] = new_qty
                row["market_value"] = new_qty * unit_value
                old_cb = existing.get("cost_basis")
                old_cb = float(old_cb) if pd.notna(old_cb) else 0.0
                if dq < 0:
                    # (Partial) close: the statement cost basis follows the
                    # contracts that remain — the Options tab treats a positive
                    # statement cost_basis as authoritative, so a closed
                    # contract's cost must not linger as an unrealized loss.
                    row["cost_basis"] = (old_cb * (new_qty / old_qty)
                                         if old_qty else 0.0)
                else:
                    row["cost_basis"] = old_cb + opt_cost.get(okey, 0.0)
                row["statement_date"] = snapshot_date
                row.pop("_key", None)
                out_rows.append(row)
            else:
                # Open a new contract (or close one not on the statement) — its
                # own option row, marked at the trade premium.
                template = bro_latest[bro_latest["account_id"] == account].head(1)
                account_type = (
                    template["account_type"].iloc[0] if not template.empty else ""
                )
                out_rows.append({
                    "statement_date":     snapshot_date,
                    "broker":             broker,
                    "account_id":         account,
                    "account_type":       account_type,
                    "symbol":             underlying,
                    "cusip":              None,
                    "description":        tx.get("description") or aclass,
                    "asset_class":        aclass,
                    "quantity":           dq,
                    "price":              float(tx["price"]) if pd.notna(tx["price"]) else 0.0,
                    "market_value":       dq * premium,
                    "cost_basis":         opt_cost.get(okey, 0.0) if dq > 0 else 0.0,
                    "unrealized_gl":      0.0,
                    "est_annual_income":  0.0,
                    "currency":           "USD",
                    "source_file":        "synthesize_interim_positions",
                })

        # Cash positions: apply per-account net cash delta. quantity ==
        # market_value for all observed cash symbols, so move them together.
        accounts_with_cash_row: set[str] = set()
        for i, r in enumerate(bro_records):
            if r["asset_class"] != "cash":
                continue
            accounts_with_cash_row.add(r["account_id"])
            touched_indices.add(i)
            delta = cash_delta_per_account.get(r["account_id"], 0.0)
            new_qty = float(r["quantity"]) + delta
            row = dict(r)
            row["quantity"] = new_qty
            row["market_value"] = new_qty
            row["statement_date"] = snapshot_date
            row.pop("_key", None)
            out_rows.append(row)

        # An account can take in interim cash (dividend, transfer, sale
        # proceeds) without holding a cash position at its last statement —
        # e.g. a sleeve that sweeps to a linked account. With no cash row to
        # absorb it the delta is silently dropped; synthesize a $1-NAV cash
        # row so the money is preserved.
        for account, delta in cash_delta_per_account.items():
            if account in accounts_with_cash_row or abs(delta) < 0.005:
                continue
            template = bro_latest[bro_latest["account_id"] == account].head(1)
            account_type = (
                template["account_type"].iloc[0] if not template.empty else ""
            )
            out_rows.append({
                "statement_date":     snapshot_date,
                "broker":             broker,
                "account_id":         account,
                "account_type":       account_type,
                "symbol":             "CASH",
                "cusip":              None,
                "description":        "Interim cash (synthesized)",
                "asset_class":        "cash",
                "quantity":           delta,
                "price":              1.0,
                "market_value":       delta,
                "cost_basis":         delta,
                "unrealized_gl":      0.0,
                "est_annual_income":  0.0,
                "currency":           "USD",
                "source_file":        "synthesize_interim_positions",
            })

        # Carry forward every untouched row by index — this preserves
        # collision-victim rows (e.g. SPY option_put when SPY ETF is in the
        # same account) and rows whose symbol+cusip are both NaN.
        for i, r in enumerate(bro_records):
            if i in touched_indices:
                continue
            row = dict(r)
            row["statement_date"] = snapshot_date
            row.pop("_key", None)
            out_rows.append(row)

    if not out_rows:
        # Every account superseded (fully stale interim file) — same shape as
        # the empty-interim early return.
        return positions.iloc[0:0].copy()
    out = pd.DataFrame(out_rows)
    # Match the column order of positions.csv exactly.
    out = out[list(positions.columns)]
    return out


def _print_samples(positions: pd.DataFrame, interim: pd.DataFrame,
                   rolled: pd.DataFrame) -> None:
    """Print before/after for the largest position changes (auto-discovered)."""
    snapshot_date = pd.Timestamp(interim["settlement_date"].max()).normalize()
    latest_per_broker = positions.groupby("broker")["statement_date"].max()
    print(f"\nSnapshot date (unified): {snapshot_date.date()}")
    print(f"Per-broker latest statement: "
          f"{ {k: v.date().isoformat() for k, v in latest_per_broker.items()} }")
    print(f"Per-broker latest interim:   "
          f"{ {k: pd.Timestamp(v).date().isoformat() for k, v in interim.groupby('broker')['settlement_date'].max().items()} }")
    print()

    # Build (account, key) -> (old_qty, old_mv, new_qty, new_mv) for every
    # row in rolled, then pick the N with the largest absolute MV delta.
    latest_per_acct = positions.groupby("account_id")["statement_date"].max()
    old_view = positions[
        positions.apply(
            lambda r: r["statement_date"] == latest_per_acct.get(r["account_id"]),
            axis=1,
        )
    ].copy()
    old_view["_key"] = old_view.apply(lambda r: _key_for(r["symbol"], r["cusip"]), axis=1)
    new_view = rolled.copy()
    new_view["_key"] = new_view.apply(lambda r: _key_for(r["symbol"], r["cusip"]), axis=1)
    old_idx = {(r["account_id"], r["_key"]): r for _, r in old_view.iterrows()}
    new_idx = {(r["account_id"], r["_key"]): r for _, r in new_view.iterrows()}
    deltas = []
    for k in set(old_idx) | set(new_idx):
        old_qty = float(old_idx[k]["quantity"]) if k in old_idx else 0.0
        old_mv  = float(old_idx[k]["market_value"]) if k in old_idx else 0.0
        new_qty = float(new_idx[k]["quantity"]) if k in new_idx else 0.0
        new_mv  = float(new_idx[k]["market_value"]) if k in new_idx else 0.0
        deltas.append((k[0], k[1], old_qty, new_qty, old_mv, new_mv, abs(new_mv - old_mv)))
    deltas.sort(key=lambda d: d[6], reverse=True)
    top_n = deltas[:8]

    print(f"{'Account / key':52s} {'Old qty':>14s} -> {'New qty':>14s}  "
          f"{'Old MV':>14s} -> {'New MV':>14s}")
    print("-" * 130)
    for account, key, old_qty, new_qty, old_mv, new_mv, _ in top_n:
        print(f"{(account + ' ' + key)[:52]:52s} {old_qty:>14,.4f} -> {new_qty:>14,.4f}  "
              f"${old_mv:>13,.2f} -> ${new_mv:>13,.2f}")

    # Aggregate NAV summary by broker.
    latest = pd.concat([
        positions[positions["broker"] == b][
            positions[positions["broker"] == b]["statement_date"]
            == positions[positions["broker"] == b]["statement_date"].max()
        ]
        for b in positions["broker"].unique()
    ])
    print()
    print(f"{'Broker':12s} {'Prior NAV':>16s}  {'Rolled NAV':>16s}  {'Delta':>14s}")
    print("-" * 70)
    for broker in sorted(positions["broker"].unique()):
        old_nav = float(latest[latest["broker"] == broker]["market_value"].sum())
        new_nav = float(rolled[rolled["broker"] == broker]["market_value"].sum())
        print(f"{broker:12s} ${old_nav:>15,.2f}  ${new_nav:>15,.2f}  ${new_nav-old_nav:>+13,.2f}")
    total_old = float(latest["market_value"].sum())
    total_new = float(rolled["market_value"].sum())
    print(f"{'TOTAL':12s} ${total_old:>15,.2f}  ${total_new:>15,.2f}  ${total_new-total_old:>+13,.2f}")
    print()
    print(f"Rolled rows: {len(rolled)}  (vs prior snapshot: {len(latest)})")
    new_keys = set(
        (r["account_id"], _key_for(r["symbol"], r["cusip"]))
        for r in rolled.to_dict("records")
    )
    old_keys = set(
        (r["account_id"], _key_for(r["symbol"], r["cusip"]))
        for r in latest.to_dict("records")
    )
    print(f"  Brand-new (account, key) since prior snapshot: {len(new_keys - old_keys)}")
    print(f"  Disappeared from prior snapshot:               {len(old_keys - new_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Write data/positions_interim.csv; default is dry-run.")
    args = parser.parse_args()

    if not INTERIM_TXN_CSV.exists():
        print(f"No interim transactions at {INTERIM_TXN_CSV}; nothing to synthesize.")
        if args.write and OUT_CSV.exists():
            OUT_CSV.unlink()
            print(f"Removed stale {OUT_CSV.name}.")
        return

    positions = pd.read_csv(POSITIONS_CSV, parse_dates=["statement_date"])
    interim = pd.read_csv(INTERIM_TXN_CSV, parse_dates=["settlement_date"])
    rolled = synthesize_interim_positions(positions, interim)

    _print_samples(positions, interim, rolled)

    if args.write:
        rolled.to_csv(OUT_CSV, index=False)
        print(f"\nWrote {OUT_CSV} ({len(rolled)} rows)")
    else:
        print(f"\n(dry-run — re-run with --write to emit {OUT_CSV})")


if __name__ == "__main__":
    main()
