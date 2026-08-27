"""Per-lot tax ledger reconstruction from the transaction history.

Pure functions, no I/O. Chronological FIFO replay of transactions.csv rows
into open lots, realizations, and an exceptions audit trail; plus banded
reconciliation of reconstructed basis against broker-reported cost_basis.
Design: docs/superpowers/specs/2026-07-23-tax-lot-engine-design.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TX_REQUIRED = ["trade_date", "settlement_date", "account_id",
               "transaction_type", "symbol", "cusip", "description",
               "quantity", "price", "amount", "pair_id"]
POS_REQUIRED = ["statement_date", "account_id", "symbol", "cusip",
                "description", "asset_class", "quantity", "cost_basis"]

OPEN_COLUMNS = ["account_id", "instrument_key", "key_source", "symbol",
                "open_date", "acquired_date", "origin", "quantity_open",
                "quantity_remaining", "basis_open", "basis_remaining",
                "source_row", "maturity", "basis_evidence", "wash_adjustment"]
REALIZATION_COLUMNS = ["account_id", "instrument_key", "open_date",
                       "acquired_date", "close_date", "close_reason",
                       "quantity_closed", "basis_closed", "proceeds",
                       "realized_gl", "holding_days", "term",
                       "closing_method", "source_row", "open_source_row",
                       "basis_source", "disallowed_wash"]
EXCEPTION_COLUMNS = ["account_id", "instrument_key", "trade_date",
                     "transaction_type", "reason", "quantity", "amount",
                     "source_row"]

LONG_TERM_DAYS = 365          # informational only; term classification uses calendar anniversaries


def classify_term(acquired_date, as_of) -> str:
    """'long' | 'short' | 'unknown' — the ledger's one term rule.

    Long only STRICTLY AFTER the calendar anniversary (IRS "more than one
    year"): a leap-year span is 366 days, so a day-count threshold misfires
    exactly on anniversary day. Missing acquired stays "unknown" — never
    guessed (the slice-1 SHORTFALL rule).
    """
    if acquired_date is None or pd.isna(acquired_date):
        return "unknown"
    return ("long" if pd.Timestamp(as_of)
            > pd.Timestamp(acquired_date) + pd.DateOffset(years=1)
            else "short")


def days_to_long_term(acquired_date, as_of) -> int | None:
    """Days from as_of until this lot's holding period passes one year.

    None exactly when classify_term is not "short" — i.e. the lot is
    already long-term or has no acquired date. Shares the calendar
    anniversary with classify_term (long only STRICTLY AFTER it), so the
    first long-term-qualifying day is the anniversary + 1 day.
    """
    # No acquired date, or no reference date, => no countdown. The caller
    # always passes a concrete as_of (build_tax_view defaults date.today());
    # a NaT as_of deliberately yields None here (we do not mirror
    # classify_term's "short", and we do not change classify_term).
    if acquired_date is None or pd.isna(acquired_date) or pd.isna(as_of):
        return None
    first_lt_day = (pd.Timestamp(acquired_date)
                    + pd.DateOffset(years=1) + pd.Timedelta(days=1))
    days = (first_lt_day - pd.Timestamp(as_of)).days
    return days if days >= 1 else None


_QTY_EPS = 1e-6

# Relief conventions. FIFO is the ledger default and the fallback; the others
# are honoured when a broker prints them on the sell row. MLMG (maximum loss,
# minimum gain) and PRO (pro rata) are recognised but not executable here,
# and SPEC is Fidelity's specific-share flag with no lot detail printed
# ("refer to confirm for Lot detail") — all three relieve FIFO and log why.
RELIEF_METHODS = {"FIFO", "LIFO", "HC", "LC", "LTHC", "VSP",
                  "MLMG", "PRO", "SPEC"}
_RELIEF_FALLBACK_REASON = {"MLMG": "relief_unsupported",
                           "PRO": "relief_unsupported",
                           "SPEC": "relief_lot_unspecified"}
# JPM names the lot(s) a specific-match sell closed, inline in the row text:
#   "... MARKET IN THIS SECURITY VS 081925 1 @614.9 ROME: ..."
# and chains them when one sell matched several lots:
#   "... VS 021925 1 @591.42 VS 031125 1 @563.17 VS 031925 1 @559.34 ..."
# Each hint is <acquisition date MMDDYY> <quantity> @<unit cost>. This text is
# the broker stating which lots it relieved, so it drives the relief even
# where the Closing Method column is blank — which is exactly what the
# multi-lot rows print.
RE_VSP_HINT = re.compile(
    r"\bVS\s+(\d{6})\s+([\d,]+(?:\.\d+)?)\s*@\s*([\d.]+)")

MM_DESC_PAT = re.compile(
    r"^CASH\b|MONEY MARKET|SWEEP|\bDEPOSIT\b|\bCORE\b|MMF|GOVT MMKT|TREASURY FUND",
    re.I)
# JPM bond buys print "<MM/DD/YYYY> MATURITY DATE" in the confirm text.
RE_MATURITY = re.compile(r"\b(\d{2}/\d{2}/\d{4})\s+MATURITY\b")
# Interest rows print the maturity as "DUE MM/DD/YYYY" even when the buy
# confirm's truncated text never says MATURITY (treasury notes).
RE_DUE = re.compile(r"\bDUE\s+(\d{2}/\d{2}/\d{4})\b")


def _maturity_of(description) -> pd.Timestamp:
    """The maturity date a bond confirm prints, NaT when it prints none."""
    match = RE_MATURITY.search(_clean(description))
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%m/%d/%Y", errors="coerce")
_SLUG_PAT = re.compile(r"[^A-Z0-9]+")


def _clean(value) -> str:
    """A stripped string, or '' for NaN/None/'nan'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def instrument_key(symbol, cusip, description,
                   fold: Optional[dict[str, str]] = None) -> tuple[str, str]:
    """(key, key_source) with symbol > cusip > description-slug precedence.

    `fold` maps prior identifiers (renamed tickers, post-action cusips) onto
    their canonical symbol — see symbol_fold. It relabels the key only; the
    source still says which column supplied it.
    """
    sym = _clean(symbol)
    if sym:
        key = sym.upper()
    else:
        cus = _clean(cusip)
        if cus:
            key = cus.upper()
        else:
            slug = _SLUG_PAT.sub(" ", _clean(description).upper()).strip()
            return (fold.get(slug, slug) if fold else slug), "desc"
        return (fold.get(key, key) if fold else key), "cusip"
    return (fold.get(key, key) if fold else key), "symbol"


def symbol_fold(ticker_history: Optional[dict] = None,
                corporate_actions: Optional[dict] = None) -> dict[str, str]:
    """{prior identifier: canonical symbol} from the two config surfaces.

    TICKER_HISTORY entries ({NEW: [{"prior_symbol": OLD, ...}]}) contribute
    OLD -> NEW; chains (A renamed to B renamed to C) resolve to the terminal
    symbol. CORPORATE_ACTIONS events contribute their optional post-action
    "cusips" aliases -> the (terminal) ticker. A cycle is a config error and
    raises rather than folding anything silently.

    The fold is deliberately date-free: a key labels one continuing
    instrument, and effective_date stays price-splicing metadata. A recycled
    ticker (old symbol later reused by a different issuer) would wrongly
    merge — entries are owner-curated per rename, like the price splice.
    """
    fold: dict[str, str] = {}
    for new, entries in (ticker_history or {}).items():
        for entry in entries or []:
            prior = _clean(entry.get("prior_symbol")).upper()
            if prior:
                fold[prior] = _clean(new).upper()

    def _terminal(symbol: str) -> str:
        seen = {symbol}
        while symbol in fold:
            symbol = fold[symbol]
            if symbol in seen:
                raise ValueError(
                    f"TICKER_HISTORY rename cycle involving {symbol!r}")
            seen.add(symbol)
        return symbol

    fold = {prior: _terminal(prior) for prior in fold}
    for ticker, events in (corporate_actions or {}).items():
        target = _clean(ticker).upper()
        target = fold.get(target, target)
        for event in events or []:
            for cusip in event.get("cusips", []) or []:
                cus = _clean(cusip).upper()
                if cus:
                    fold[cus] = target
    return fold


def corporate_split_events(corporate_actions: Optional[dict],
                           fold: Optional[dict[str, str]] = None
                           ) -> list[dict]:
    """[{key, date, ratio}] replay events from CORPORATE_ACTIONS config.

    Only kind == "split" exists today; anything else is a loud config error
    rather than a silently ignored entry. Ratio multiplies open share
    counts, so it must be positive and finite.
    """
    events: list[dict] = []
    for ticker, actions in (corporate_actions or {}).items():
        key = _clean(ticker).upper()
        if fold:
            key = fold.get(key, key)
        for action in actions or []:
            kind = _clean(action.get("kind"))
            if kind != "split":
                raise ValueError(
                    f"CORPORATE_ACTIONS[{ticker!r}]: unknown kind {kind!r}")
            date = pd.to_datetime(action.get("effective_date"),
                                  errors="coerce")
            if pd.isna(date):
                raise ValueError(
                    f"CORPORATE_ACTIONS[{ticker!r}]: bad effective_date "
                    f"{action.get('effective_date')!r}")
            ratio = pd.to_numeric(action.get("ratio"), errors="coerce")
            if pd.isna(ratio) or not np.isfinite(ratio) or ratio <= 0:
                raise ValueError(
                    f"CORPORATE_ACTIONS[{ticker!r}]: bad ratio "
                    f"{action.get('ratio')!r}")
            events.append({"key": key, "date": date, "ratio": float(ratio)})
    return events


# Fidelity prints a security's NAME + cusip on confirms but its NAME + ticker
# in holdings (its holdings parser writes cusip="" literally), so transactions
# key by cusip while positions key by symbol and NOTHING joins. Both facts are
# printed by the same broker in the same statement, so the crosswalk is
# learned from the data rather than guessed: positions supply name -> symbol,
# and a transaction's name is the text before its cusip token. Anything
# ambiguous refuses to resolve (audit WSF-7: an honest unresolved key beats a
# confident wrong one).
_NAME_STRIP = re.compile(r"[^A-Z0-9 ]+")
_CUSIP_TOKEN = re.compile(r"\b[0-9A-Z]{8}\d\b")
_MIN_NAME_CHARS = 6


def normalize_security_name(text) -> str:
    """Uppercased, punctuation collapsed to single spaces, trimmed."""
    return re.sub(r"\s+", " ",
                  _NAME_STRIP.sub(" ", _clean(text).upper())).strip()


def security_name_from_description(description) -> str:
    """The security name a description leads with, normalized.

    Fidelity confirms read "<NAME> <CUSIP> You Bought ...", so the name is
    whatever precedes the first cusip-shaped token; without one, the whole
    description is the name.
    """
    desc = _clean(description)
    match = _CUSIP_TOKEN.search(desc)
    return normalize_security_name(desc[:match.start()] if match else desc)


def build_name_resolver(positions: Optional[pd.DataFrame],
                        fold: Optional[dict[str, str]] = None
                        ) -> dict[str, str]:
    """{normalized security name: symbol} learned from positions rows.

    A name carrying more than one symbol across the book is dropped, never
    resolved to one of them. Symbols fold to their canonical identity first,
    so a renamed ticker's months do not read as two symbols for one name —
    unfolded, every rename would drop its own name from the resolver.
    """
    if positions is None or positions.empty:
        return {}
    learned: dict[str, set[str]] = {}
    for _, row in positions.iterrows():
        symbol = _clean(row["symbol"]).upper()
        if fold:
            symbol = fold.get(symbol, symbol)
        name = normalize_security_name(row["description"])
        if symbol and len(name) >= _MIN_NAME_CHARS:
            learned.setdefault(name, set()).add(symbol)
    return {name: next(iter(syms)) for name, syms in learned.items()
            if len(syms) == 1}


def build_cusip_resolver(transactions: Optional[pd.DataFrame],
                         names: dict[str, str],
                         fold: Optional[dict[str, str]] = None
                         ) -> dict[str, str]:
    """{cusip: symbol} for cusips a NAME match has already proved.

    Fidelity prints the cusip inside the description only on trade confirms.
    Its dividend-reinvestment and account-transfer rows carry the cusip in the
    column but not in the description, so the name rule cannot speak for them
    and they would strand under the raw cusip while the same instrument's buys
    resolve — one instrument under two keys in one account, which the ledger
    cannot detect and which silently stops in-kind transfers from moving lots.

    A cusip IS the instrument identifier, so once any row proves cusip ->
    symbol by name, sibling rows carrying that cusip inherit it. A cusip with
    contradictory proofs is dropped entirely: option confirms occasionally
    carry a neighbouring row's cusip, and nothing may be inferred from a
    contradicted pair.
    """
    if transactions is None or transactions.empty or not names:
        return {}
    proven: dict[str, set[str]] = {}
    for _, row in transactions.iterrows():
        cusip = _clean(row["cusip"]).upper()
        if not cusip or _clean(row["symbol"]):
            continue
        key, source = resolve_instrument_key(row, names, fold=fold)
        if source == "resolved":
            proven.setdefault(cusip, set()).add(key)
    return {cusip: next(iter(syms)) for cusip, syms in proven.items()
            if len(syms) == 1}


def build_key_resolvers(transactions: Optional[pd.DataFrame],
                        positions: Optional[pd.DataFrame],
                        fold: Optional[dict[str, str]] = None
                        ) -> tuple[dict[str, str], dict[str, str]]:
    """(name resolver, cusip resolver) for one book.

    Option rows are excluded from the cusip-proving pass — their descriptions
    name a derivative, not the underlying, and their cusip column is
    occasionally mis-attributed from an adjacent row.
    """
    names = build_name_resolver(positions, fold)
    frame = transactions
    if frame is not None and not frame.empty:
        frame = frame[~frame["description"].apply(is_option_row)]
    return names, build_cusip_resolver(frame, names, fold)


def resolve_instrument_key(row, resolver: dict[str, str],
                           cusip_resolver: Optional[dict[str, str]] = None,
                           fold: Optional[dict[str, str]] = None
                           ) -> tuple[str, str]:
    """(key, key_source) for a TRANSACTION row, canonicalized onto symbols.

    A symbol the statement printed always wins — the resolver can only speak
    for rows that print no symbol. A candidate is a learned name that
    **extends or equals** the row's name; they must agree unanimously or
    nothing resolves.

    The direction is deliberately one-way. Holdings names are the richer of
    the two ("… TRUST UNIT DEPOSITARY RECEIPT" vs the confirm's "… TRUST
    UNIT"), so extension in that direction is the normal shape. Allowing the
    reverse — a row name extending a learned name — merges *different*
    securities that share an issuer prefix: a contingent-value right
    ("HOLOGIC INCORPO … CVR") and a cash-merger stub both collapse onto the
    common stock's ticker. Extra tokens on the row side change the
    instrument, so they must defeat the match, not widen it.
    """
    key, source = instrument_key(row["symbol"], row["cusip"],
                                 row["description"], fold)
    if source == "symbol" or not (resolver or cusip_resolver):
        return key, source
    name = security_name_from_description(row["description"])
    if resolver and len(name) >= _MIN_NAME_CHARS:
        hits = {symbol for learned, symbol in resolver.items()
                if learned.startswith(name)}
        if len(hits) == 1:
            return next(iter(hits)), "resolved"
    # rows whose description omits the cusip (DRIPs, account transfers) can
    # still inherit a cusip->symbol pair another row proved by name
    if cusip_resolver and source == "cusip":
        symbol = cusip_resolver.get(key)
        if symbol:
            return symbol, "resolved"
    return key, source


def bond_principal_basis(quantity, price, amount) -> float:
    """Cost basis for one lot-opening row: bond principal, else the amount.

    A bond trade's amount is principal PLUS accrued interest paid to the
    seller, and accrued interest is not basis (it washes against the first
    coupon). Bond quantity is face value and price is quoted per 100 face, so
    principal is qty*price/100 — two orders of magnitude away from the equity
    reading (qty*price), which is what makes identifying the convention from
    the arithmetic safe. Accrued interest cannot exceed roughly one coupon
    period, so a genuine bond principal lands far inside the 10% band while
    an equity's never does.
    """
    # coerce first: callers building frames from dicts can pass a non-numeric
    # price, and this is the only site that reads it
    quantity = pd.to_numeric(quantity, errors="coerce")
    price = pd.to_numeric(price, errors="coerce")
    gross = abs(pd.to_numeric(amount, errors="coerce"))
    if pd.isna(gross) or gross <= 0 or pd.isna(quantity) or pd.isna(price):
        return gross
    if price <= 0:
        return gross
    principal = abs(quantity) * price / 100.0
    equity = abs(quantity) * price
    if (abs(principal - gross) < abs(equity - gross)
            and abs(principal - gross) <= 0.10 * gross):
        return principal
    return gross


def cash_keys_from_positions(positions: Optional[pd.DataFrame],
                             fold: Optional[dict[str, str]] = None
                             ) -> set[str]:
    """Instrument keys whose positions rows are classed as cash."""
    if positions is None or positions.empty:
        return set()
    cash = positions[positions["asset_class"].astype(str) == "cash"]
    return {instrument_key(r["symbol"], r["cusip"], r["description"], fold)[0]
            for _, r in cash.iterrows()}


def is_cash_like(description, key: str, cash_keys: set[str],
                 key_source: str) -> bool:
    """Cash-class instruments carry no lots (fixed $1 NAV: basis == value).

    Symbol/cusip-keyed rows are excluded only via positions-declared cash
    keys (or the literal symbol CASH); the description pattern applies to
    desc-keyed rows alone, so a real holding whose NAME contains a pattern
    word can never be silently dropped.
    """
    if key in cash_keys or key == "CASH":
        return True
    # "resolved" rows were desc- or cusip-keyed before the crosswalk spoke for
    # them, so they stay eligible for the description heuristic — otherwise
    # resolution would silently re-admit sweep rows the pre-resolver engine
    # dropped.
    return (key_source in ("desc", "resolved")
            and bool(MM_DESC_PAT.search(_clean(description))))


# Options are out of scope for the lot ledger (their basis lives in the
# option machinery), but both brokers key option confirms AND option
# positions by the UNDERLYING's symbol — so without an explicit guard they
# flow into the underlying's equity lots. Shapes: JPM confirms/positions
# "CALL TKR 01/17/25 …", Fidelity "PUT (TKR) ISSUER …". Anchored so issuer
# names beginning with CALL…/PUT… and mid-description event words never
# match.
_OPTION_DESC_PAT = re.compile(
    r"^(CALL|PUT)\s+(?:\(|\S{1,10}\s+\d{1,2}/\d{1,2}/\d{2,4}\b)")


def is_option_row(description) -> bool:
    """True when a transaction row's description has an option-confirm shape."""
    return bool(_OPTION_DESC_PAT.match(_clean(description)))


def is_option_position(row) -> bool:
    """True for option positions rows: option asset_class, or option desc."""
    if _clean(row.get("asset_class") if hasattr(row, "get")
              else row["asset_class"]).lower().startswith("option"):
        return True
    return is_option_row(row["description"])


def relief_method(row) -> str:
    """The relief convention to apply to a sell row.

    The broker's printed method when we recognise it, FIFO otherwise — a
    transactions frame without the column (an older CSV, or any caller that
    never had one) replays exactly as it did before slice 5.
    """
    raw = ""
    if hasattr(row, "get"):
        raw = _clean(row.get("closing_method")).upper()
    return raw if raw in RELIEF_METHODS else "FIFO"


def vsp_hints(description) -> list[tuple[pd.Timestamp, float, float]]:
    """(acquired_date, quantity, unit cost) for every lot a sell row names.

    Fragments the statement wrapped into unreadability (a hint whose quantity
    column was lost mid-wrap, an unparseable date) are skipped rather than
    guessed at — a partial hint identifies no lot.
    """
    out = []
    for raw_date, raw_qty, raw_price in RE_VSP_HINT.findall(_clean(description)):
        try:
            date = pd.to_datetime(raw_date, format="%m%d%y")
            out.append((date, float(raw_qty.replace(",", "")),
                        float(raw_price)))
        except (ValueError, TypeError):
            continue
    return out


def _cost_per_share(lot: "_Lot") -> float:
    return (lot.basis_remaining / lot.quantity_remaining
            if lot.quantity_remaining > _QTY_EPS else 0.0)


def _lot_date(lot: "_Lot") -> pd.Timestamp:
    """Acquisition date, falling back to the open date when unknown."""
    return lot.acquired_date if pd.notna(lot.acquired_date) else lot.open_date


def _is_long_term(lot: "_Lot", close_date: pd.Timestamp) -> bool:
    acquired = _lot_date(lot)
    return bool(pd.notna(acquired)
                and close_date > acquired + pd.DateOffset(years=1))


def _order_lots(lots: list, method: str, close_date: pd.Timestamp) -> list:
    """Lots in the order `method` relieves them.

    Sorting is stable, so equal keys keep the underlying FIFO order and the
    ledger stays deterministic. Lots a sell named explicitly are handled by
    `_close_lots`'s plan, ahead of this ordering.
    """
    if method == "LIFO":
        ordered = sorted(lots, key=lambda l: -_lot_date(l).toordinal())
    elif method == "HC":
        ordered = sorted(lots, key=lambda l: -_cost_per_share(l))
    elif method == "LC":
        ordered = sorted(lots, key=_cost_per_share)
    elif method == "LTHC":
        ordered = sorted(lots, key=lambda l: (
            0 if _is_long_term(l, close_date) else 1, -_cost_per_share(l)))
    else:                      # FIFO and every fallback
        ordered = list(lots)
    return ordered


@dataclass
class _Lot:
    account_id: str
    instrument_key: str
    key_source: str
    symbol: str
    open_date: pd.Timestamp
    acquired_date: pd.Timestamp      # NaT when genuinely unknown
    origin: str
    quantity_open: float
    quantity_remaining: float
    basis_open: float
    basis_remaining: float
    source_row: int
    # Bond maturity parsed from the opening row's "MM/DD/YYYY MATURITY"
    # text, NaT otherwise. Lets a redemption row that prints no identifier
    # at all (JPM bills) find the instrument it redeems by date + face.
    maturity: pd.Timestamp = pd.NaT
    # True once a printed figure sized a close on this lot — the basis it
    # carries is then the broker's arithmetic, not our reconstruction.
    printed_relief: bool = False
    # Dollars of a wash-sale disallowed loss folded into this lot's
    # basis_remaining (0.0 when none). Set by apply_wash_folds; rides on
    # lots.csv as provenance for the in-place basis step-up. In OPEN_COLUMNS,
    # so open_row() emits it via getattr.
    wash_adjustment: float = 0.0

    def open_row(self) -> dict:
        row = {c: getattr(self, c) for c in OPEN_COLUMNS
               if c != "basis_evidence"}
        row["basis_evidence"] = ("printed" if self.printed_relief
                                 else "reconstructed")
        return row


@dataclass
class LotLedgerResult:
    open_lots: pd.DataFrame
    realizations: pd.DataFrame
    exceptions: pd.DataFrame
    # Bond-maturity evidence the replay pulled from maturity_by_key (a
    # parsed maturity on the row itself always wins and never counts here).
    # Mirrors the deleted enrich_bond_maturities' (n_lots, n_instruments)
    # shape. Defaulted so callers built before this slice are unaffected.
    n_maturity_enriched: int = 0
    maturity_enriched_keys: set[str] = field(default_factory=set)


class _Replay:
    """Chronological replay state. Tasks 2-5 extend the dispatch table."""

    def __init__(self, cash_keys: set[str],
                 resolver: Optional[dict[str, str]] = None,
                 cusip_resolver: Optional[dict[str, str]] = None,
                 fold: Optional[dict[str, str]] = None,
                 names_by_key: Optional[dict[str, set[str]]] = None,
                 maturity_by_key: Optional[dict[str, pd.Timestamp]] = None):
        self.resolver = resolver or {}
        self.cusip_resolver = cusip_resolver or {}
        self.fold = fold or {}
        self.names_by_key = names_by_key or {}
        self.cash_keys = cash_keys
        self.maturity_by_key = maturity_by_key or {}
        self.lots: dict[tuple[str, str], list[_Lot]] = {}
        self.realizations: list[dict] = []
        self.exceptions: list[dict] = []
        # (account_id, instrument_key) pairs whose basis was ever sized by a
        # printed figure. Tracked per PAIR, not per lot: a lot that printed
        # relief closed out entirely is gone from open_lots, and reading only
        # the survivors would let a printed-touched instrument report itself
        # as reconstructed — the exact leak the strict anchor must not have.
        self.printed_relief_pairs: set[tuple[str, str]] = set()
        # counters for maturity_by_key enrichments applied in handle_open —
        # exposed on the result so build_lots can print them without a
        # separate post-replay pass
        self.n_maturity_enriched = 0
        self.maturity_enriched_keys: set[str] = set()

    # -- plumbing -----------------------------------------------------------
    def _key(self, row) -> tuple[str, str]:
        """Canonical (key, key_source) for a transaction row."""
        return resolve_instrument_key(row, self.resolver, self.cusip_resolver,
                                      self.fold)

    # -- corporate-action key rescue (spec §3.3) ---------------------------
    def _holders(self, account_id: str) -> dict[str, list["_Lot"]]:
        """Instrument keys with open lots in this account, at replay time."""
        return {k: fifo for (a, k), fifo in self.lots.items()
                if a == account_id
                and any(l.quantity_remaining > _QTY_EPS for l in fifo)}

    @staticmethod
    def _leading_overlap(row_tokens: list[str], name: str) -> int:
        name_tokens = name.split()
        n = 0
        for a, b in zip(row_tokens, name_tokens):
            if a != b:
                break
            n += 1
        return n

    def _name_match(self, account_id: str, description,
                    required_qty: Optional[float] = None,
                    exclude: Optional[str] = None) -> Optional[str]:
        """The unique lot-holding instrument this description names, or None.

        Match on the run of leading tokens shared with a learned positions
        name: a strict-maximum winner with a run of >=2 tokens, or exactly 1
        token of >=6 characters (issuer words like KELLANOVA; short generic
        leaders cannot rescue). `required_qty` refuses a rescue that would
        close more shares than the candidate holds — a share-bearing event
        naming the right instrument cannot exceed its position.
        """
        row_tokens = security_name_from_description(description).split()
        if not row_tokens:
            return None
        scores: dict[str, int] = {}
        for key, fifo in self._holders(account_id).items():
            if key == exclude:
                continue
            names = self.names_by_key.get(key) or ()
            score = max((self._leading_overlap(row_tokens, n)
                         for n in names), default=0)
            if score > 0:
                if required_qty is not None:
                    held = sum(l.quantity_remaining for l in fifo)
                    if required_qty > held + _QTY_EPS:
                        continue
                scores[key] = score
        if not scores:
            return None
        best = max(scores.values())
        winners = [k for k, s in scores.items() if s == best]
        if len(winners) != 1:
            return None
        if best >= 2 or (best == 1 and len(row_tokens[0]) >= 6):
            return winners[0]
        return None

    def _corporate_action_key(self, row: pd.Series) -> tuple[str, bool]:
        """(key to act on, whether to act) for a corporate-action row.

        Brokers print these rows under whatever identifier the action left
        behind — the successor's symbol on an exchange out-leg, a cusip the
        lots never carried, or no identifier at all — so the printed key
        routinely drifts from where the lots live. Sells are never routed
        here; this applies only to merger/redemption/stock_split/exchange/
        principal_pmt rows.
        """
        key, _source = self._key(row)
        account = row["account_id"]
        holders = self._holders(account)
        row_tokens = security_name_from_description(row["description"]).split()
        if key in holders:
            names = self.names_by_key.get(key)
            if names and row_tokens and max(
                    (self._leading_overlap(row_tokens, n) for n in names),
                    default=0) == 0:
                # the description contradicts every learned name for the
                # printed key: a mis-attributed symbol (the CVNA split row
                # printed under DVN). Acting on the printed key would mutate
                # the wrong instrument's lots.
                alt = self._name_match(account, row["description"],
                                       exclude=key)
                if alt:
                    return alt, True
                self._log_exception(row, key, "corporate_action_key_mismatch",
                                    quantity=row["quantity"],
                                    amount=row["amount"])
                return key, False
            return key, True
        cus = _clean(row["cusip"]).upper()
        cus = self.fold.get(cus, cus)
        if cus and cus != key and cus in holders:
            return cus, True
        if key in self.names_by_key:
            # The printed key holds no lots here but IS a real positions
            # instrument (a spinoff child like the HONA receipt, whose
            # description leads with the PARENT's issuer word). That is not
            # identifier drift — believe the broker: the name rescue below
            # would re-key the row onto the parent and mutate the WRONG
            # instrument's lots, the mirror image of the veto above. Scope:
            # names_by_key is book-wide, not per-account; a key held only in
            # another account still counts as "real", and acting on it here
            # yields at worst a benign *_without_position flag.
            return key, True
        qty = row["quantity"]
        required = abs(qty) if pd.notna(qty) and qty < 0 else None
        alt = self._name_match(account, row["description"],
                               required_qty=required)
        if alt:
            return alt, True
        if str(row["transaction_type"]) == "redemption":
            return self._redemption_maturity_key(row, key)
        return key, True

    def _redemption_maturity_key(self, row: pd.Series,
                                 key: str) -> tuple[str, bool]:
        """Match an identifier-less redemption to lots by maturity + face."""
        qty = row["quantity"]
        face = abs(qty) if pd.notna(qty) else np.nan
        matches = []
        for cand, fifo in self._holders(row["account_id"]).items():
            if not any(pd.notna(l.maturity) and l.maturity == row["_date"]
                       for l in fifo):
                continue
            held = sum(l.quantity_remaining for l in fifo)
            if pd.notna(face) and abs(held - face) <= _QTY_EPS:
                matches.append(cand)
        if len(matches) == 1:
            return matches[0], True
        self._log_exception(row, key, "redemption_unmatched",
                            quantity=row["quantity"], amount=row["amount"])
        return key, False

    def _fifo(self, account_id: str, key: str) -> list[_Lot]:
        return self.lots.setdefault((account_id, key), [])

    def _log_exception(self, row: pd.Series, key: str, reason: str,
                       quantity: float = np.nan, amount: float = np.nan):
        self.exceptions.append({
            "account_id": row["account_id"], "instrument_key": key,
            "trade_date": row["_date"], "transaction_type": row["transaction_type"],
            "reason": reason, "quantity": quantity, "amount": amount,
            "source_row": row["_row"]})

    def _close(self, lot: _Lot, quantity: float, basis: float, proceeds: float,
               close_date: pd.Timestamp, reason: str, realized: bool = True,
               method: str = "FIFO", source_row=np.nan,
               basis_source: str = "reconstructed",
               lot_basis: Optional[float] = None):
        held = ((close_date - lot.acquired_date).days
                if pd.notna(lot.acquired_date) else np.nan)
        term = classify_term(lot.acquired_date, close_date)
        self.realizations.append({
            "account_id": lot.account_id, "instrument_key": lot.instrument_key,
            "open_date": lot.open_date, "acquired_date": lot.acquired_date,
            "close_date": close_date, "close_reason": reason,
            "quantity_closed": quantity, "basis_closed": basis,
            "proceeds": proceeds if realized else np.nan,
            "realized_gl": (proceeds - basis) if realized else 0.0,
            "holding_days": held, "term": term,
            "closing_method": method, "source_row": source_row,
            "open_source_row": lot.source_row,
            "basis_source": basis_source,
            "disallowed_wash": 0.0})
        lot.quantity_remaining -= quantity
        # `lot_basis` splits "basis the sell relieved" from "basis this lot
        # gave up" — they differ only when the relieved figure was drawn
        # across the instrument's lots rather than from the closing shares.
        lot.basis_remaining -= basis if lot_basis is None else lot_basis

    def _close_lots(self, account_id: str, key: str, quantity: float,
                    total_proceeds: float, close_date: pd.Timestamp,
                    reason: str, realized: bool = True,
                    method: str = "FIFO", plan=None,
                    source_row=np.nan, printed_cost=None,
                    lot_unknowable: bool = False) -> float:
        """Close `quantity` shares under `method`; returns the UNCOVERED rest.

        `plan` is an ordered list of (lot, max quantity) the broker named
        explicitly (a specific match); it is honoured first, and anything left
        over falls through to `method`'s ordering.

        Proceeds are allocated pro-rata by the share of the REQUESTED quantity
        actually closed from each lot, so an underflow leaves the uncovered
        remainder's proceeds unallocated (they ride on the exception row).
        Closed-out lots also return the closed (lot, qty, basis) list via
        self._last_closed for handlers that re-open lots (transfers/exchanges).
        """
        lots = self._fifo(account_id, key)
        # ---- decide the takes, mutating nothing --------------------------
        remaining = quantity
        reserved: dict[int, float] = {}
        takes: list[tuple[_Lot, float]] = []

        def offer(lot: _Lot, cap: float):
            nonlocal remaining
            if remaining <= _QTY_EPS:
                return
            free = lot.quantity_remaining - reserved.get(id(lot), 0.0)
            take = min(free, remaining, cap)
            if take <= _QTY_EPS:
                return
            reserved[id(lot)] = reserved.get(id(lot), 0.0) + take
            takes.append((lot, take))
            remaining -= take

        for lot, cap in (plan or []):
            offer(lot, cap)
        for lot in _order_lots(lots, method, close_date):
            offer(lot, float("inf"))

        # ---- assign basis, pro-rata on each lot's ORIGINAL remaining, so
        # two takes from one lot still sum to its proportional share -------
        basis_by_take = [
            lot.basis_remaining * take / lot.quantity_remaining
            if lot.quantity_remaining > _QTY_EPS else 0.0
            for lot, take in takes]

        # ---- two cases where the ledger cannot reconstruct the basis and the
        # broker printed what it relieved. (1) A synthesized opening lot pools
        # an unknown set of real lots at their average cost, so FIFO inside it
        # IS average cost and can never reproduce oldest-lot relief. (2) A sell
        # the broker relieved by specific share without naming the lot is not
        # executable at all. In both, the printed figure is the only
        # non-guessing basis to draw. Known lots on ordinary sells are never
        # overridden. -------------------------------------------------------
        pool_idx = [i for i, (lot, _take) in enumerate(takes)
                    if lot.origin == "opening"]
        # which override actually ran, or None. NOT the same as the
        # `lot_unknowable` request: an unknowable sell whose instrument holds
        # a lot of unknown basis falls through to the pool path, and labelling
        # that relief `printed_unknowable` would misreport where it came from.
        printed_kind: Optional[str] = None
        self._last_printed_shortfall = 0.0
        # instrument basis the unknowable path must leave behind, or None
        unknowable_target: Optional[float] = None
        have_printed = printed_cost is not None and pd.notna(printed_cost)
        # An instrument holding a lot of unknown basis has no total to spread:
        # re-deriving every lot from it would turn one NaN into all of them.
        # `basis_unknown` already says the honest thing about such a pair.
        known_basis = all(pd.notna(lot.basis_remaining) for lot in lots)

        if lot_unknowable and have_printed and takes and known_basis:
            # The broker relieved a lot it did not name — very often not the
            # one FIFO would pick, since specific-share identification exists
            # precisely to close something other than the oldest shares. With
            # no lot identified, the only fact available is at the INSTRUMENT
            # level: these shares left, and this much basis left with them.
            # So the sell removes exactly the printed basis from the
            # instrument and the remainder is spread over the surviving
            # shares. Tying the relief to the FIFO-selected lot instead would
            # strand the rest of that lot's basis on zero shares and destroy
            # it when the lot leaves the queue.
            available = sum(lot.basis_remaining for lot in lots)
            wanted = float(printed_cost)
            total = min(max(wanted, 0.0), available)
            # a printed cost the instrument cannot supply means the LOTS are
            # wrong; that is a finding, not something to absorb silently
            self._last_printed_shortfall = max(0.0, wanted - total)
            printed_kind = "printed_unknowable"
            unknowable_target = available - total
            shares = sum(take for _lot, take in takes)
            basis_by_take = [total * take / shares for _lot, take in takes]
        elif pool_idx and have_printed:
            pool_set = set(pool_idx)
            known = sum(b for i, b in enumerate(basis_by_take)
                        if i not in pool_set)
            pool_shares = sum(takes[i][1] for i in pool_idx)
            available = sum(takes[i][0].basis_remaining for i in pool_idx)
            wanted = float(printed_cost) - known
            pool_total = min(max(wanted, 0.0), available)
            self._last_printed_shortfall = max(0.0, wanted - pool_total)
            if pool_shares > _QTY_EPS:
                printed_kind = "printed_pool"
                for i in pool_idx:
                    basis_by_take[i] = min(
                        pool_total * takes[i][1] / pool_shares,
                        takes[i][0].basis_remaining)

        # ---- redemption at maturity relieves at face: amortized cost equals
        # face there exactly (either amortization method), matching the
        # broker's printed closing figure — the discount was booked as
        # interest income along the way, never capital gain. The lot itself
        # still decrements its reconstructed share via the lot_basis channel,
        # so remaining basis cannot go negative. 5 calendar days of
        # settlement grace; pre-maturity redemptions are untouched.
        face_lot_basis: dict[int, float] = {}
        if reason == "redemption" and printed_kind is None:
            for i, (lot, take) in enumerate(takes):
                if (pd.notna(lot.maturity)
                        and lot.quantity_open > 0
                        and pd.notna(lot.basis_open)
                        and _BOND_PRICE_PER_FACE_MIN
                        <= lot.basis_open / lot.quantity_open
                        <= _BOND_PRICE_PER_FACE_MAX
                        and close_date >= lot.maturity
                        - pd.Timedelta(days=5)):
                    face_lot_basis[i] = basis_by_take[i]
                    basis_by_take[i] = take

        # ---- apply --------------------------------------------------------
        self._last_closed: list[tuple[_Lot, float, float]] = []
        self._last_basis_residue = 0.0
        for i, ((lot, take), basis) in enumerate(zip(takes, basis_by_take)):
            proceeds = total_proceeds * take / quantity
            if printed_kind and (printed_kind == "printed_unknowable"
                                 or lot.origin == "opening"):
                basis_source = printed_kind
                lot.printed_relief = True
                self.printed_relief_pairs.add((account_id, key))
            elif i in face_lot_basis:
                basis_source = "amortized_face"
            else:
                basis_source = "reconstructed"
            self._close(lot, take, basis, proceeds, close_date, reason,
                        realized=realized, method=method,
                        source_row=source_row, basis_source=basis_source,
                        lot_basis=(0.0 if unknowable_target is not None
                                   else face_lot_basis.get(i)))
            self._last_closed.append((lot, take, basis))
        if unknowable_target is not None:
            # Re-spread what is left over the shares still held. The lots of a
            # printed-relieved instrument carry an average cost from here on,
            # which is the honest consequence of the broker not naming what it
            # closed; only the instrument total is meaningful, and
            # `basis_evidence` is what says so.
            survivors = [l for l in lots if l.quantity_remaining > _QTY_EPS]
            held = sum(l.quantity_remaining for l in survivors)
            if held > _QTY_EPS:
                for l in lots:
                    l.printed_relief = True
                    l.basis_remaining = (
                        unknowable_target * l.quantity_remaining / held
                        if l.quantity_remaining > _QTY_EPS else 0.0)
            else:
                # nothing is held any more, so nothing can carry basis: what
                # is left is reconstruction error the printed figure exposed
                self._last_basis_residue = unknowable_target
                for l in lots:
                    l.basis_remaining = 0.0
        # exhausted lots leave the queue; mutate in place — this list object
        # is the one held in self.lots
        lots[:] = [l for l in lots if l.quantity_remaining > _QTY_EPS]
        return remaining

    # -- handlers -----------------------------------------------------------
    def handle_open(self, row: pd.Series, key: str, source: str,
                    origin: str):
        quantity, amount = row["quantity"], row["amount"]
        if pd.isna(quantity) or pd.isna(amount) or quantity <= 0:
            self._log_exception(row, key, "missing_fields",
                                quantity=quantity, amount=amount)
            return
        basis = bond_principal_basis(quantity, row["price"], amount)
        # A maturity parsed off THIS row always wins; the pre-collected map
        # (collect_bond_maturities, evidence drawn from the instrument's
        # other rows — e.g. an interest row's DUE date on a note whose own
        # buy confirm prints no MATURITY text) only fills in when the row
        # itself is silent.
        maturity = _maturity_of(row["description"])
        if pd.isna(maturity):
            maturity = self.maturity_by_key.get(key, pd.NaT)
            if pd.notna(maturity):
                self.n_maturity_enriched += 1
                self.maturity_enriched_keys.add(key)
        self._fifo(row["account_id"], key).append(_Lot(
            account_id=row["account_id"], instrument_key=key,
            key_source=source, symbol=self._lot_symbol(row, key, source),
            open_date=row["_date"], acquired_date=row["_date"], origin=origin,
            quantity_open=quantity, quantity_remaining=quantity,
            basis_open=basis, basis_remaining=basis,
            source_row=row["_row"], maturity=maturity))

    def _lot_symbol(self, row: pd.Series, key: str, source: str) -> str:
        """The symbol a lot carries: canonical when the fold relabeled it.

        A BK-keyed buy folded onto BNY must not keep symbol BK — the Tax
        tab's price lookup is symbol-first, and the printed symbol no longer
        trades. Unfolded rows keep the printed text untouched.
        """
        raw = _clean(row["symbol"])
        if source == "symbol" and raw and raw.upper() != key:
            return key
        return raw

    def _vsp_plan(self, account_id: str, key: str, row: pd.Series
                  ) -> tuple[list, int]:
        """([(lot, quantity)] the row named, count of hints that matched none).

        Matching is exact: same acquisition date, and a cost per share within
        a cent of the printed one. A near miss is not a match — silently
        relieving the wrong lot is worse than a logged fallback. A lot is
        never named twice, so a chain of hints consumes distinct lots.
        """
        hints = vsp_hints(row["description"])
        if not hints:
            return [], 0
        plan, used, unmatched = [], set(), 0
        for hint_date, hint_qty, hint_price in hints:
            for lot in self._fifo(account_id, key):
                if id(lot) in used or lot.quantity_remaining <= _QTY_EPS:
                    continue
                if pd.isna(lot.acquired_date) or lot.acquired_date != hint_date:
                    continue
                if abs(_cost_per_share(lot) - hint_price) > 0.01:
                    continue
                plan.append((lot, hint_qty))
                used.add(id(lot))
                break
            else:
                unmatched += 1
        return plan, unmatched

    def handle_sell(self, row: pd.Series, key: str):
        quantity, amount = row["quantity"], row["amount"]
        if pd.isna(quantity) or pd.isna(amount):
            self._log_exception(row, key, "missing_fields",
                                quantity=quantity, amount=amount)
            return
        printed = relief_method(row)
        # The named-lot text outranks the Closing Method column: a multi-lot
        # specific match prints every lot it closed but leaves that column
        # blank, so the column alone would miss the broker's own statement.
        plan, unmatched = self._vsp_plan(row["account_id"], key, row)
        method = "VSP" if plan else ("FIFO" if printed == "VSP" else printed)
        if unmatched or (printed == "VSP" and not plan):
            self._log_exception(row, key, "vsp_lot_unmatched",
                                quantity=abs(quantity), amount=abs(amount))
        fallback = _RELIEF_FALLBACK_REASON.get(method)
        # SPEC states specific-share relief and names no lot, so the ledger
        # cannot execute it and the printed cost is the only non-guessing
        # basis. MLMG/PRO stay plain FIFO fallbacks: they are executable in
        # principle, just not implemented here, so overriding their basis
        # would assert knowledge we do not have.
        lot_unknowable = fallback == "relief_lot_unspecified"
        if fallback:
            self._log_exception(row, key, fallback, quantity=abs(quantity),
                                amount=abs(amount))
            method = "FIFO"
        uncovered = self._close_lots(row["account_id"], key, abs(quantity),
                                     abs(amount), row["_date"], "sell",
                                     method=method, plan=plan,
                                     source_row=row["_row"],
                                     printed_cost=row.get("closing_cost"),
                                     lot_unknowable=lot_unknowable)
        if getattr(self, "_last_printed_shortfall", 0.0) > 0.005:
            self._log_exception(row, key, "printed_basis_exceeds_lots",
                                quantity=abs(quantity),
                                amount=self._last_printed_shortfall)
        if abs(getattr(self, "_last_basis_residue", 0.0)) > 0.01:
            self._log_exception(row, key, "printed_relief_basis_residue",
                                quantity=abs(quantity),
                                amount=self._last_basis_residue)
        if uncovered > _QTY_EPS:
            self._log_exception(row, key, "sell_underflow",
                                quantity=uncovered, amount=abs(amount))

    def handle_split(self, row: pd.Series, key: str):
        quantity = row["quantity"]
        if pd.isna(quantity) or quantity <= 0:
            self._log_exception(row, key, "missing_fields", quantity=quantity)
            return
        fifo = self._fifo(row["account_id"], key)
        held = sum(lot.quantity_remaining for lot in fifo)
        if held <= _QTY_EPS:
            self._log_exception(row, key, "split_without_position",
                                quantity=quantity)
            return
        factor = 1.0 + quantity / held
        for lot in fifo:
            lot.quantity_remaining *= factor

    def handle_merger(self, row: pd.Series, key: str):
        quantity, amount = row["quantity"], row["amount"]
        cash_shape = (pd.notna(quantity) and quantity < 0
                      and pd.notna(amount) and amount > 0)
        if not cash_shape:
            self._log_exception(row, key, "merger_unrecognized_shape",
                                quantity=quantity, amount=amount)
            return
        fifo = self._fifo(row["account_id"], key)
        held = sum(lot.quantity_remaining for lot in fifo)
        if held <= _QTY_EPS:
            self._log_exception(row, key, "merger_without_position",
                                quantity=quantity, amount=amount)
            return
        self._close_lots(row["account_id"], key, held, abs(amount),
                         row["_date"], "merger_cash",
                         source_row=row["_row"])

    def handle_redemption(self, row: pd.Series, key: str):
        quantity, amount = row["quantity"], row["amount"]
        if pd.isna(quantity) or pd.isna(amount):
            self._log_exception(row, key, "missing_fields",
                                quantity=quantity, amount=amount)
            return
        uncovered = self._close_lots(row["account_id"], key, abs(quantity),
                                     abs(amount), row["_date"], "redemption",
                                     source_row=row["_row"])
        if uncovered > _QTY_EPS:
            self._log_exception(row, key, "sell_underflow",
                                quantity=uncovered, amount=abs(amount))

    def handle_principal(self, row: pd.Series, key: str):
        """A principal payment / return of capital reduces basis, not shares.

        Allocated per share (the printed rate is per-unit); clamped at zero
        with the excess logged — a distribution beyond basis is a gain event
        this ledger does not fabricate. Matches the broker's own arithmetic:
        the position's reported cost basis drops by exactly the amount.
        """
        amount = row["amount"]
        if pd.isna(amount) or amount <= 0:
            self._log_exception(row, key, "missing_fields",
                                quantity=row["quantity"], amount=amount)
            return
        fifo = self._fifo(row["account_id"], key)
        held = sum(lot.quantity_remaining for lot in fifo)
        if held <= _QTY_EPS:
            self._log_exception(row, key, "principal_without_position",
                                quantity=row["quantity"], amount=amount)
            return
        if any(pd.isna(lot.basis_remaining) for lot in fifo):
            # an unknown-basis lot cannot absorb its share; reducing only the
            # known lots would misstate them
            self._log_exception(row, key, "principal_without_position",
                                quantity=row["quantity"], amount=amount)
            return
        excess = 0.0
        for lot in fifo:
            share = float(amount) * lot.quantity_remaining / held
            reduced = lot.basis_remaining - share
            if reduced < 0:
                excess += -reduced
                reduced = 0.0
            lot.basis_remaining = reduced
        if excess > 0.005:
            self._log_exception(row, key, "principal_exceeds_basis",
                                quantity=row["quantity"], amount=excess)

    def handle_corporate_split(self, key: str, ratio: float):
        """Apply a config-stated split: open share counts multiply by ratio.

        Basis is untouched (a split moves no money); every account holding
        the key splits — the action is issuer-level.
        """
        for (account, k), fifo in self.lots.items():
            if k != key:
                continue
            for lot in fifo:
                if lot.quantity_remaining > _QTY_EPS:
                    lot.quantity_remaining *= ratio

    def _move(self, out_row: pd.Series, in_row: pd.Series, out_key: str,
              close_reason: str, origin: str):
        """Close out-side lots (zero realized) and re-open them on the
        in-side, carrying basis and acquired_date."""
        in_key, in_source = self._key(in_row)
        out_qty = abs(out_row["quantity"])
        in_qty = abs(in_row["quantity"])
        uncovered = self._close_lots(out_row["account_id"], out_key, out_qty,
                                     0.0, out_row["_date"], close_reason,
                                     realized=False,
                                     source_row=out_row["_row"])
        if uncovered > _QTY_EPS:
            self._log_exception(out_row, out_key, "sell_underflow",
                                quantity=uncovered)
        moved_qty = out_qty - uncovered
        if moved_qty <= _QTY_EPS:
            return
        for lot, take, basis in self._last_closed:
            self._fifo(in_row["account_id"], in_key).append(_Lot(
                account_id=in_row["account_id"], instrument_key=in_key,
                key_source=in_source,
                symbol=self._lot_symbol(in_row, in_key, in_source),
                open_date=in_row["_date"], acquired_date=lot.acquired_date,
                origin=origin,
                quantity_open=in_qty * take / out_qty,
                quantity_remaining=in_qty * take / out_qty,
                basis_open=basis, basis_remaining=basis,
                source_row=in_row["_row"]))

    def handle_pair_event(self, event: dict):
        if "row" in event:  # unpaired
            row = event["row"]
            qty = row["quantity"]
            if (event["kind"] == "exchange" and event.get("lone")
                    and pd.notna(qty) and -1.0 < qty < 0):
                # a singleton SUB-SHARE out-leg is a cash-in-lieu fractional
                # (a CIL is definitionally the sub-share remainder of a
                # ratio conversion): close it UNREALIZED — the consideration
                # rides elsewhere, and inventing a realized loss from the
                # leg's printed 0.00 would poison realized views. Whole-
                # share lone legs and multi-leg refused groups are NOT
                # closed: theirs is a failed pairing whose basis should
                # have carried, and stranding keeps that visible.
                key, apply = self._corporate_action_key(row)
                if not apply:
                    return
                uncovered = self._close_lots(
                    row["account_id"], key, abs(qty), 0.0, row["_date"],
                    "exchange_out", realized=False, source_row=row["_row"])
                if uncovered > _QTY_EPS:
                    self._log_exception(row, key, "sell_underflow",
                                        quantity=uncovered)
                return
            key, _ = self._key(row)
            reason = ("exchange_unpaired" if event["kind"] == "exchange"
                      else "transfer_unmatched")
            self._log_exception(row, key, reason, quantity=row["quantity"],
                                amount=row["amount"])
            return
        out_row, in_row = event["out"], event["in"]
        if event["kind"] == "exchange":
            # exchange out-legs print the successor's symbol or a cusip the
            # lots never carried — route through the corporate-action rescue
            out_key, apply = self._corporate_action_key(out_row)
            if not apply:
                return
            self._move(out_row, in_row, out_key, "exchange_out", "exchange_in")
        else:
            out_key, _ = self._key(out_row)
            self._move(out_row, in_row, out_key, "transfer_out", "transfer_in")

    def handle_opening(self, account_id: str, date: pd.Timestamp,
                       rows: pd.DataFrame):
        """Synthesize opening lots for first-statement shortfalls (spec §4.4).

        Brokers may report one positions row per lot, so rows aggregate per
        instrument before the shortfall computation — mirroring
        reconcile_lots. Any missing cost_basis in the group marks the whole
        instrument's opening basis unknown (never guessed).
        """
        agg: dict[str, dict] = {}
        for _, row in rows.iterrows():
            key, source = instrument_key(row["symbol"], row["cusip"],
                                         row["description"], self.fold)
            if (str(row["asset_class"]) == "cash"
                    or is_cash_like(row["description"], key, self.cash_keys,
                                    source)):
                continue
            qty = row["quantity"]
            if pd.isna(qty) or qty <= _QTY_EPS:
                continue
            entry = agg.setdefault(key, {
                "source": source, "symbol": _clean(row["symbol"]),
                "quantity": 0.0, "basis": 0.0, "basis_missing": False})
            entry["quantity"] += float(qty)
            if pd.isna(row["cost_basis"]):
                entry["basis_missing"] = True
            else:
                entry["basis"] += float(row["cost_basis"])
        for key, entry in agg.items():
            fifo = self._fifo(account_id, key)
            recon_qty = sum(lot.quantity_remaining for lot in fifo)
            shortfall = entry["quantity"] - recon_qty
            if shortfall <= _QTY_EPS:
                continue
            if entry["basis_missing"]:
                basis = np.nan
                self.exceptions.append({
                    "account_id": account_id, "instrument_key": key,
                    "trade_date": date, "transaction_type": "opening",
                    "reason": "opening_basis_missing",
                    "quantity": shortfall, "amount": np.nan,
                    "source_row": -1})
            else:
                recon_basis = sum(lot.basis_remaining for lot in fifo)
                basis = max(entry["basis"] - recon_basis, 0.0)
            fifo.append(_Lot(
                account_id=account_id, instrument_key=key,
                key_source=entry["source"], symbol=entry["symbol"],
                open_date=date, acquired_date=pd.NaT, origin="opening",
                quantity_open=shortfall, quantity_remaining=shortfall,
                basis_open=basis, basis_remaining=basis, source_row=-1))

    # -- output -------------------------------------------------------------
    def result(self) -> LotLedgerResult:
        open_rows = [lot.open_row()
                     for fifo in self.lots.values() for lot in fifo
                     if lot.quantity_remaining > _QTY_EPS]
        # Evidence is a property of the INSTRUMENT's basis history, not of the
        # lots that happen to have survived: printed relief may have closed
        # out the very lot it sized, leaving only untouched lots behind.
        for row in open_rows:
            if (row["account_id"], row["instrument_key"]) in \
                    self.printed_relief_pairs:
                row["basis_evidence"] = "printed"
        return LotLedgerResult(
            open_lots=pd.DataFrame(open_rows, columns=OPEN_COLUMNS),
            realizations=pd.DataFrame(self.realizations,
                                      columns=REALIZATION_COLUMNS),
            exceptions=pd.DataFrame(self.exceptions,
                                    columns=EXCEPTION_COLUMNS),
            n_maturity_enriched=self.n_maturity_enriched,
            maturity_enriched_keys=self.maturity_enriched_keys)


# Lot-affecting transaction types with a single-row handler. Types absent
# here and from _PAIRED_TYPES (Task 4) have no lot impact and are skipped.
# cash_in_lieu: cash for a fractional share the ledger never held as a lot
# (the one lot-fragment case, a sub-share exchange out-leg, closes via that
# leg) — skip, don't log an unhandled_event.
_NO_LOT_IMPACT = {"dividend", "interest", "withholding", "contribution",
                  "option_expire", "cash_in_lieu"}
_DEFERRED: set[str] = set()
# Rows brokers print under whatever identifier the action left behind — the
# successor's symbol, a cusip the lots never carried, or nothing at all —
# so their keys route through the rescue/veto in _corporate_action_key.
# Sells are deliberately NOT in this set.
_CORPORATE_TYPES = {"merger", "redemption", "stock_split", "principal_pmt"}


def _prepare(transactions: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TX_REQUIRED if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions frame missing columns: {missing}")
    frame = transactions.copy()
    frame["_row"] = frame.index
    trade = pd.to_datetime(frame["trade_date"], errors="coerce")
    settle = pd.to_datetime(frame["settlement_date"], errors="coerce")
    frame["_date"] = trade.fillna(settle)
    frame = frame[frame["_date"].notna()]
    if frame.empty:
        return frame
    return frame.sort_values(["_date", "_row"], kind="stable")


_PAIRED_TYPES = {"exchange", "transfer_in", "transfer_out"}


def _pair_events(frame: pd.DataFrame,
                 resolver: Optional[dict[str, str]] = None,
                 cusip_resolver: Optional[dict[str, str]] = None,
                 fold: Optional[dict[str, str]] = None
                 ) -> tuple[pd.DataFrame, list[dict]]:
    """Extract exchange/transfer rows into paired move events.

    Returns (frame_without_those_rows, events) where each event is
    {"kind": "exchange"|"transfer", "date", "out": row, "in": row} or an
    unpaired {"kind": ..., "row": row} the replay must log as an exception.
    In-kind rows are the quantity-bearing ones; cash transfers (no quantity)
    are dropped here — they have no lot impact.
    """
    mask = frame["transaction_type"].isin(_PAIRED_TYPES)
    rest, sub = frame[~mask], frame[mask]
    events: list[dict] = []

    ex = sub[sub["transaction_type"] == "exchange"]
    for (_, _), grp in ex.groupby(["account_id", "_date"], sort=False):
        legs = [r for _, r in grp.iterrows()]
        outs_e = [r for r in legs
                  if pd.notna(r["quantity"]) and r["quantity"] < 0]
        ins_e = [r for r in legs
                 if pd.notna(r["quantity"]) and r["quantity"] > 0]
        degenerate = [r for r in legs
                      if pd.isna(r["quantity"]) or r["quantity"] == 0]
        for row in degenerate:
            events.append({"kind": "exchange", "row": row})
        if len(outs_e) == 1 and len(ins_e) == 1:
            events.append({"kind": "exchange", "date": outs_e[0]["_date"],
                           "out": outs_e[0], "in": ins_e[0]})
        else:
            # one-sided or multi-leg group: refuse to guess a pairing.
            # A SINGLETON leg is marked lone — there was nothing to pair
            # with, so an out-side singleton is a plain share removal (a
            # cash-in-lieu fractional) rather than a failed pairing.
            for row in outs_e + ins_e:
                events.append({"kind": "exchange", "row": row,
                               "lone": len(legs) == 1})

    tr = sub[sub["transaction_type"].isin(["transfer_in", "transfer_out"])]
    inkind = tr[tr["quantity"].notna()]
    outs = [r for _, r in inkind.iterrows()
            if r["transaction_type"] == "transfer_out"]
    ins = [r for _, r in inkind.iterrows()
           if r["transaction_type"] == "transfer_in"]

    def _shape(row) -> tuple:
        return (resolve_instrument_key(row, resolver or {},
                                       cusip_resolver or {}, fold)[0],
                row["_date"], abs(row["quantity"]))

    used_in: set[int] = set()
    for out_row in outs:
        pair = _clean(out_row["pair_id"])
        match = None
        if pair:
            for i, in_row in enumerate(ins):
                if i not in used_in and _clean(in_row["pair_id"]) == pair:
                    match = i
                    break
        else:
            shape_hits = [i for i, in_row in enumerate(ins)
                          if i not in used_in
                          and not _clean(in_row["pair_id"])
                          and _shape(in_row) == _shape(out_row)]
            rival_outs = [o for o in outs
                          if not _clean(o["pair_id"])
                          and _shape(o) == _shape(out_row)]
            if len(shape_hits) == 1 and len(rival_outs) == 1:
                match = shape_hits[0]
        if match is None:
            events.append({"kind": "transfer", "row": out_row})
        else:
            used_in.add(match)
            events.append({"kind": "transfer", "date": out_row["_date"],
                           "out": out_row, "in": ins[match]})
    for i, in_row in enumerate(ins):
        if i not in used_in:
            events.append({"kind": "transfer", "row": in_row})
    return rest, events


def _names_by_key(positions: Optional[pd.DataFrame],
                  fold: Optional[dict[str, str]] = None
                  ) -> dict[str, set[str]]:
    """{instrument key: normalized positions names} for the rescue matcher."""
    if positions is None or positions.empty:
        return {}
    names: dict[str, set[str]] = {}
    for _, row in positions.iterrows():
        key, _source = instrument_key(row["symbol"], row["cusip"],
                                      row["description"], fold)
        name = normalize_security_name(row["description"])
        if key and name:
            names.setdefault(key, set()).add(name)
    return names


def build_lot_ledger(transactions: pd.DataFrame,
                     *,
                     opening_positions: Optional[pd.DataFrame] = None,
                     fold: Optional[dict[str, str]] = None,
                     splits: Optional[list[dict]] = None,
                     maturity_by_key: Optional[dict[str, pd.Timestamp]] = None
                     ) -> LotLedgerResult:
    """Replay transactions chronologically into a per-lot FIFO ledger.

    `opening_positions` (the positions.csv frame) enables cash-class
    exclusion and first-statement opening-lot synthesis (Task 5); None runs
    a pure-transactions replay (unit tests). `fold` (symbol_fold) relabels
    prior identifiers onto their canonical symbol; `splits`
    (corporate_split_events) injects config-stated share multiplications.
    `maturity_by_key` (collect_bond_maturities, run PRE-replay by the
    caller) supplies a maturity for opening lots whose own row prints none —
    a maturity parsed off the row always wins over the map. All default to
    None so callers without config replay exactly as before.
    """
    frame = _prepare(transactions)
    if opening_positions is not None and not opening_positions.empty:
        missing = [c for c in POS_REQUIRED
                   if c not in opening_positions.columns]
        if missing:
            raise ValueError(f"positions frame missing columns: {missing}")
    # Option rows leave the replay before pairing so option in-kind
    # transfers / exchange legs are never matched or moved either.
    option_rows = frame.iloc[0:0]
    if not frame.empty:
        opt_mask = frame["description"].apply(is_option_row)
        option_rows, frame = frame[opt_mask], frame[~opt_mask]
    resolver = build_name_resolver(opening_positions, fold)
    cusip_resolver = build_cusip_resolver(frame, resolver, fold)
    frame, pair_events = _pair_events(frame, resolver, cusip_resolver, fold)
    replay = _Replay(cash_keys_from_positions(opening_positions, fold),
                     resolver, cusip_resolver, fold=fold,
                     names_by_key=_names_by_key(opening_positions, fold),
                     maturity_by_key=maturity_by_key)
    for _, row in option_rows.iterrows():
        key, _src = resolve_instrument_key(row, resolver, cusip_resolver,
                                           fold)
        replay._log_exception(row, key, "option_excluded",
                              quantity=row["quantity"], amount=row["amount"])
    events: list[tuple] = [(row["_date"], 0, row["_row"], "tx", row)
                           for _, row in frame.iterrows()]
    for ev in pair_events:
        if "row" in ev:
            date, seq = ev["row"]["_date"], ev["row"]["_row"]
        else:
            date, seq = ev["date"], min(ev["out"]["_row"], ev["in"]["_row"])
        events.append((date, 0, seq, "pair", ev))
    # config-stated splits sort BEFORE same-day transactions (priority -1):
    # the action is effective at the open, so a same-day sell quotes
    # post-split share counts
    for split in splits or []:
        events.append((split["date"], -1, len(events), "corp_split",
                       (split["key"], split["ratio"])))
    if opening_positions is not None and not opening_positions.empty:
        posf = opening_positions.copy()
        posf = posf[~posf.apply(is_option_position, axis=1)]
        posf["_sdate"] = pd.to_datetime(posf["statement_date"],
                                        errors="coerce")
        posf = posf[posf["_sdate"].notna()]
        for account_id, grp in posf.groupby("account_id", sort=False):
            first = grp["_sdate"].min()
            events.append((first, 2, len(events), "opening",
                           (account_id, first, grp[grp["_sdate"] == first])))
    for _, _, _, kind, payload in sorted(events, key=lambda e: (e[0], e[1], e[2])):
        if kind == "pair":
            replay.handle_pair_event(payload)
        elif kind == "opening":
            replay.handle_opening(*payload)
        elif kind == "corp_split":
            replay.handle_corporate_split(*payload)
        else:
            _dispatch_row(replay, payload)
    return replay.result()


def _dispatch_row(replay: _Replay, row: pd.Series):
    kind = str(row["transaction_type"])
    key, source = replay._key(row)
    if kind in _NO_LOT_IMPACT:
        return
    if is_cash_like(row["description"], key, replay.cash_keys, source):
        return
    if kind in _DEFERRED:
        replay._log_exception(row, key, "unhandled_event",
                              quantity=row["quantity"], amount=row["amount"])
    elif kind in ("buy", "reinvestment"):
        replay.handle_open(row, key, source, origin=kind)
    elif kind == "sell":
        replay.handle_sell(row, key)
    elif kind in _CORPORATE_TYPES:
        key, apply = replay._corporate_action_key(row)
        if not apply:
            return
        if kind == "stock_split":
            replay.handle_split(row, key)
        elif kind == "merger":
            replay.handle_merger(row, key)
        elif kind == "redemption":
            replay.handle_redemption(row, key)
        else:
            replay.handle_principal(row, key)
    else:
        replay._log_exception(row, key, "unhandled_event",
                              quantity=row["quantity"], amount=row["amount"])


# --------------------------------------------------------------------------
# Reconciliation (spec §5): reconstructed basis vs broker-reported cost_basis
# --------------------------------------------------------------------------

RECON_COLUMNS = ["account_id", "instrument_key", "month", "reconstructed",
                 "reported", "diff_usd", "diff_pct", "band",
                 "reconstructed_qty", "reported_qty", "qty_diff",
                 "basis_evidence"]

# Share-count agreement tolerance: one thousandth of a share, which is the
# last digit brokers report on fractional positions. A relative term was
# tried and removed — float64 noise on these sums is ~1e-9 even at bond face
# scale (tens of thousands), six orders below this floor, so a relative term
# is inert at every real magnitude and only risks silencing whole-share
# corporate-action gaps.
_QTY_RECON_EPS = 0.001


def quantity_mismatch(reconstructed_qty: float, reported_qty: float) -> bool:
    """True when a reconstructed share count disagrees with the reported one.

    Basis is only meaningful once quantity agrees: a percentage difference on
    a position whose share count is wrong measures nothing, and a basis that
    happens to agree anyway is a coincidence, not a reconciliation.

    A NaN on either side is not a mismatch — it is an absent statement, and
    guessing zero for it would print a share count the broker never gave.
    """
    if pd.isna(reconstructed_qty) or pd.isna(reported_qty):
        return False
    return abs(float(reconstructed_qty) - float(reported_qty)) > _QTY_RECON_EPS

LOT_OK_USD = 1.0        # absolute forgiveness for rounding dust
LOT_WATCH_PCT = 0.30    # |diff%| above this is at least watch
LOT_ERROR_PCT = 2.0     # error needs |diff%| above this ...
LOT_ERROR_USD = 1_000.0  # ... AND |diff$| above this (per-instrument floor)

# Bond price shape: quantity IS face dollars, so clean cost per face dollar
# is price/100. Real treasuries sit ~0.9-1.05; the band [0.5, 2.0] admits
# deep discounts and premiums while a share-priced instrument (basis/qty =
# share price) almost always falls outside. Shared by the accretion
# envelope AND redemption face-relief — the two mechanisms assume the same
# premise, so they must enforce the same gate. Residual: an instrument
# genuinely priced $0.50-$2.00/share with bond-shaped maturity text could
# still pass; the report names every credited/relieved row, which is the
# operating mitigation.
_BOND_PRICE_PER_FACE_MIN = 0.5
_BOND_PRICE_PER_FACE_MAX = 2.0

# Wash-sale replacement window (days each side). Kept numerically identical
# to parsers.tax_scanner.WINDOW_DAYS — imported there would be a cycle; a
# test pins the two together instead.
_WASH_WINDOW_DAYS = 30


def _accretion_window(group: pd.DataFrame,
                      as_of) -> Optional[tuple[float, float]]:
    """[floor, ceiling] a bond group's reported basis may honestly occupy.

    Straight-line bound between clean purchase cost and face: the chord
    provably contains the broker's constant-yield amortization curve
    (convexity, shared endpoints), so no coupon parse or yield solve is
    needed. Returns None — meaning "no envelope, band normally" — unless
    EVERY lot in the group carries a maturity and clean numbers.
    Quantity is face dollars for bonds (price per 100), so face == qty.
    """
    if "maturity" not in group.columns:
        return None
    if group.empty or pd.isna(pd.Timestamp(as_of)):
        return None
    mat = pd.to_datetime(group["maturity"], errors="coerce")
    if mat.isna().any():
        return None
    opened = pd.to_datetime(group["open_date"], errors="coerce")
    as_of = pd.Timestamp(as_of)
    lo = hi = 0.0
    for m, od, q_open, q_rem, b_open in zip(
            mat, opened, group["quantity_open"],
            group["quantity_remaining"], group["basis_open"]):
        if (pd.isna(od) or pd.isna(b_open) or b_open <= 0
                or pd.isna(q_open) or q_open <= 0
                or pd.isna(q_rem) or q_rem < 0):
            return None
        term_days = (m - od).days
        frac = 1.0 if term_days <= 0 else min(
            max((as_of - od).days / term_days, 0.0), 1.0)
        face, clean = float(q_open), float(b_open)
        # Enforce the bond premise instead of assuming it: quantity IS face
        # dollars, so clean cost per face dollar is the price/100 — real
        # bonds sit inside _BOND_PRICE_PER_FACE_MIN/_MAX. An equity lot
        # (basis/qty = share price) falls outside that band immediately, so
        # a coincidental DUE date in an equity's text cannot pull it into
        # the envelope. Shared with redemption face-relief's gate.
        if not (_BOND_PRICE_PER_FACE_MIN <= clean / face
                <= _BOND_PRICE_PER_FACE_MAX):
            return None
        accreted = clean + (face - clean) * frac
        scale = float(q_rem) / float(q_open)
        lo += min(clean, accreted) * scale
        hi += max(clean, accreted) * scale
    return lo, hi


def _wash_fold_plan(realizations: Optional[pd.DataFrame], pair,
                    group: pd.DataFrame,
                    window_days: int = _WASH_WINDOW_DAYS) -> dict:
    """Itemized wash fold: which realized loss to disallow per sell, and how
    many basis dollars to add to each still-open replacement lot.

    Detection contract is identical to the band's: same-account realized LOSS
    sells; still-open, distinct-lot replacements inside the window;
    replacement capacity is `quantity_remaining` consumed FIFO-by-acquisition
    across sells chronologically (one replacement lot cannot back two washes).

    Returns {"disallowed_by_sell": {realization_index: dollars},
             "added_by_lot": {open_lot_index: dollars}, "total": float}.
    A sell's disallowed dollars are spread across the replacement lots it
    consumed in proportion to shares taken, so sum(added_by_lot) ==
    sum(disallowed_by_sell) == total. apply_wash_folds applies the two maps;
    _wash_consistent_delta returns total for the reconcile band."""
    empty = {"disallowed_by_sell": {}, "added_by_lot": {}, "total": 0.0}
    need = {"account_id", "instrument_key", "close_reason", "realized_gl",
            "close_date", "quantity_closed", "open_source_row"}
    if (realizations is None or realizations.empty
            or not need.issubset(realizations.columns) or group.empty):
        return empty
    rz = realizations[
        (realizations["account_id"] == pair[0])
        & (realizations["instrument_key"] == pair[1])
        & (realizations["close_reason"] == "sell")
        & (pd.to_numeric(realizations["realized_gl"], errors="coerce") < 0)]
    if rz.empty:
        return empty
    rz = rz.sort_values("close_date")
    opened = pd.to_datetime(group["open_date"], errors="coerce")
    capacity = {i: (float(q) if pd.notna(q) and q > 0 else 0.0)
                for i, q in group["quantity_remaining"].items()}
    disallowed_by_sell: dict = {}
    added_by_lot: dict = {}
    total = 0.0
    for r in rz.itertuples(index=True):
        close = pd.to_datetime(r.close_date, errors="coerce")
        if pd.isna(close) or pd.isna(r.quantity_closed):
            continue
        qty_sold = float(r.quantity_closed)
        if qty_sold <= 0 or pd.isna(r.realized_gl):
            continue
        in_window = ((opened >= close - pd.Timedelta(days=window_days))
                     & (opened <= close + pd.Timedelta(days=window_days))
                     & (group["source_row"] != r.open_source_row))
        idx = sorted(group.loc[in_window].index, key=lambda i: opened.loc[i])
        repl_qty = sum(capacity[i] for i in idx)
        if repl_qty <= 0:
            continue
        claim = min(repl_qty, qty_sold)
        disallowed = abs(float(r.realized_gl)) * (claim / qty_sold)
        disallowed_by_sell[r.Index] = disallowed
        total += disallowed
        remaining = claim
        for i in idx:
            if remaining <= 0:
                break
            take = min(capacity[i], remaining)
            capacity[i] -= take
            remaining -= take
            added_by_lot[i] = added_by_lot.get(i, 0.0) + disallowed * (
                take / claim)
    return {"disallowed_by_sell": disallowed_by_sell,
            "added_by_lot": added_by_lot, "total": total}


def _wash_consistent_delta(realizations: Optional[pd.DataFrame], pair,
                           group: pd.DataFrame,
                           window_days: int = _WASH_WINDOW_DAYS) -> float:
    """Disallowed-loss total the broker would have folded into the pair's
    still-open replacement basis. Sum of the fold plan's per-sell disallowed
    dollars (see _wash_fold_plan for the full matching contract); used by
    reconcile_lots to band `wash_consistent`."""
    return _wash_fold_plan(realizations, pair, group, window_days)["total"]


def apply_wash_folds(ledger, recon: pd.DataFrame) -> None:
    """Apply the printed-basis-confirmed wash folds in `recon` to `ledger`
    IN PLACE: defer each disallowed loss out of realized_gl and into the
    still-open replacement lot's basis_remaining, stamping the provenance
    columns (disallowed_wash on the realization, wash_adjustment on the lot).

    Conservative gate: only pairs reconcile_lots banded `wash_consistent`
    (the broker's reported basis confirms the delta) are folded. Call with
    the reconcile that detected the wash, then REGENERATE reconcile — the
    folded pairs reconcile `ok`, and a second call on a fresh reconcile is a
    no-op (idempotent at the build level). basis_remaining / realized_gl
    accumulate (+=); the two provenance columns are set absolutely, so this
    must be driven from a freshly-detected reconcile, never a stale one.
    Self-guarded: each loop skips rows whose provenance stamp is already
    nonzero, so a repeat call is inert even on a STALE reconcile."""
    if recon is None or recon.empty or "band" not in recon.columns:
        return
    ol, rz = ledger.open_lots, ledger.realizations
    if ol is None or ol.empty:
        return
    for row in recon[recon["band"] == "wash_consistent"].itertuples(
            index=False):
        pair = (row.account_id, row.instrument_key)
        grp = ol[(ol["account_id"] == pair[0])
                 & (ol["instrument_key"] == pair[1])]
        plan = _wash_fold_plan(rz, pair, grp)
        for lot_idx, added in plan["added_by_lot"].items():
            # already folded (stamp set) -> skip, so a re-run on a stale recon
            # never double-folds a partial-capacity residual
            if float(ol.at[lot_idx, "wash_adjustment"]) != 0.0:
                continue
            ol.at[lot_idx, "basis_remaining"] = (
                float(ol.at[lot_idx, "basis_remaining"]) + added)
            ol.at[lot_idx, "wash_adjustment"] = added
        for rz_idx, disallowed in plan["disallowed_by_sell"].items():
            if float(rz.at[rz_idx, "disallowed_wash"]) != 0.0:
                continue
            rz.at[rz_idx, "realized_gl"] = (
                float(rz.at[rz_idx, "realized_gl"]) + disallowed)
            rz.at[rz_idx, "disallowed_wash"] = disallowed


def classify_basis(reconstructed: float, reported: float, *,
                   tol_pct: Optional[float] = None,
                   tolerances: Optional[dict] = None) -> str:
    """Band for one instrument's reconstructed-vs-reported basis.

    Mirrors reconcile_holdings.classify: allowlisted tolerance -> known;
    reported <= 0 can't ratio (error iff anything was reconstructed);
    |diff| <= ok_usd is rounding dust -> ok.
    """
    tol = {"ok_usd": LOT_OK_USD, "watch_pct": LOT_WATCH_PCT,
           "error_pct": LOT_ERROR_PCT, "error_usd": LOT_ERROR_USD}
    tol.update(tolerances or {})
    diff = reconstructed - reported
    if abs(diff) <= tol["ok_usd"]:
        return "ok"
    if reported <= 0:
        return "error"
    abs_pct = abs(100.0 * diff / reported)
    if tol_pct is not None and abs_pct <= tol_pct:
        return "known"
    if abs_pct > tol["error_pct"] and abs(diff) > tol["error_usd"]:
        return "error"
    if abs_pct > tol["watch_pct"]:
        return "watch"
    return "ok"


def collect_bond_maturities(tx: pd.DataFrame,
                            resolver: Optional[dict] = None,
                            cusip_resolver: Optional[dict] = None,
                            fold: Optional[dict[str, str]] = None,
                            ) -> dict[str, pd.Timestamp]:
    """{instrument key: maturity date} evidence from statement text.

    Evidence = every distinct date matched by RE_MATURITY or RE_DUE across
    the instrument's transaction descriptions (rows resolved to keys with
    resolve_instrument_key). A key enriches ONLY when the evidence is
    exactly one distinct date; zero or conflicting dates omit the key
    entirely (fail-normal) — this exactly-one-date rule is the binding
    guard against a coincidental or mismatched DUE/MATURITY token.

    Runs PRE-replay: the caller (build_lots) computes this once and passes
    it into build_lot_ledger(maturity_by_key=...), so a note-shaped opener
    that prints no MATURITY text of its own can still have its lot's
    maturity filled from an interest row's DUE date elsewhere in the same
    instrument's history — a post-replay pass over open_lots can never see
    that, because a redemption occurring earlier in the same replay would
    already have needed the maturity to face-relieve.

    `resolver`/`cusip_resolver` are necessarily the CLI's build_key_resolvers
    pass over the raw transactions frame, not the ledger's internal
    _prepare-based resolution during replay — the two can occasionally
    disagree on a key. That asymmetry is fail-normal: a mismatched key just
    means this function's evidence enriches nothing for that row, never a
    wrong enrichment.
    """
    dates_by_key: dict[str, set[pd.Timestamp]] = {}
    if tx is None or tx.empty:
        return {}
    for _, row in tx.iterrows():
        desc = row["description"]
        if pd.isna(desc):
            continue
        # cheap regex test before the (comparatively expensive) key
        # resolution — the vast majority of rows carry no maturity text at
        # all, and resolving a key nobody will use is wasted work
        found = RE_MATURITY.findall(desc) + RE_DUE.findall(desc)
        if not found:
            continue
        key, _src = resolve_instrument_key(row, resolver or {},
                                           cusip_resolver or {}, fold)
        for s in found:
            ts = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
            # A regex-shaped-but-invalid calendar date (e.g. "13/45/2026")
            # coerces to NaT and is dropped here rather than raising deep
            # inside the replay — a garbled date must not kill the whole
            # build_lots run.
            if pd.notna(ts):
                dates_by_key.setdefault(key, set()).add(ts)
    return {key: next(iter(dates)) for key, dates in dates_by_key.items()
            if len(dates) == 1}


def reconcile_lots(open_lots: pd.DataFrame, positions_month: pd.DataFrame, *,
                   tolerances: Optional[dict] = None,
                   allowlist: Optional[dict] = None,
                   fold: Optional[dict[str, str]] = None,
                   realizations: Optional[pd.DataFrame] = None
                   ) -> pd.DataFrame:
    """Per (account, instrument): lots' basis_remaining vs reported cost_basis.

    `positions_month` is ONE statement month's positions rows (the caller
    filters); non-cash rows only are compared. Lots with no positions row are
    `unjoinable`; positions rows with no lots are `uncovered`. A comparison
    whose lots include an unknown (NaN) basis is banded `basis_unknown` and
    never ok/watch/error.
    """
    allow = allowlist or {}
    month = ""
    as_of_ts = pd.NaT
    posf = positions_month
    if not posf.empty:
        sdt = pd.to_datetime(posf["statement_date"], errors="coerce")
        month = str(sdt.max().date()) if sdt.notna().any() else ""
        as_of_ts = sdt.max() if sdt.notna().any() else pd.NaT
        posf = posf[(posf["asset_class"].astype(str) != "cash")
                    & ~posf.apply(is_option_position, axis=1)]

    recon: dict[tuple[str, str], float] = {}
    recon_qty: dict[tuple[str, str], float] = {}
    basis_unknown: set[tuple[str, str]] = set()
    evidence: dict[tuple[str, str], str] = {}
    if not open_lots.empty:
        by_pair = open_lots.groupby(["account_id", "instrument_key"],
                                    sort=False)
        grouped = by_pair["basis_remaining"]
        recon = {k: float(v) for k, v in grouped.sum().items()}
        recon_qty = {k: float(v) for k, v
                     in by_pair["quantity_remaining"].sum().items()}
        nan_flags = grouped.apply(lambda s: bool(s.isna().any()))
        basis_unknown = {k for k, v in nan_flags.items() if v}
        # "printed" wins over "reconstructed" — the conservative direction:
        # an instrument is only unaided if NOTHING about its basis was drawn
        # from a figure the broker printed. Absent column = pre-slice-9
        # callers and hand-built frames, which are unaided by construction.
        if "basis_evidence" in open_lots.columns:
            evidence = {k: ("printed" if (v == "printed").any()
                            else "reconstructed")
                        for k, v in by_pair["basis_evidence"]}

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    reported_by_pair: dict[tuple[str, str], float] = {}
    reported_qty_by_pair: dict[tuple[str, str], float] = {}
    if not posf.empty:
        pos_keys = posf.apply(lambda r: instrument_key(
            r["symbol"], r["cusip"], r["description"], fold)[0], axis=1)
        # same filtered frame the basis side aggregates, so cash/option rows
        # can never be counted on one side only
        grouped_pos = posf.assign(_key=pos_keys).groupby(
            ["account_id", "_key"], sort=False)
        # min_count=1 so an all-NaN basis group stays NaN instead of summing
        # to 0.0 — a cost basis the broker never stated. Carry-forward months
        # (MTM repriced, basis deliberately not carried) are exactly that
        # case, and banding them `error` against a fabricated zero accounted
        # for 12 of the 14 error bands in the slice-5 verdict.
        reported_by_pair = {
            k: float(v) for k, v
            in grouped_pos["cost_basis"].sum(min_count=1).items()}
        # min_count=1 so an all-NaN quantity group stays NaN instead of
        # summing to 0.0 — a fabricated share count the broker never stated
        reported_qty_by_pair = {
            k: float(v) for k, v
            in grouped_pos["quantity"].sum(min_count=1).items()}
    for pair, reported in reported_by_pair.items():
        seen.add(pair)
        rec_qty = recon_qty.get(pair, 0.0)
        rep_qty = reported_qty_by_pair.get(pair, np.nan)
        if pair in basis_unknown:
            reconstructed, band = np.nan, "basis_unknown"
        elif pair in recon:
            reconstructed = recon[pair]
            # quantity precedence: a basis percentage on a position whose
            # share count disagrees measures nothing, and the allowlist
            # tolerates basis only
            if quantity_mismatch(rec_qty, rep_qty):
                band = "qty_mismatch"
            elif pd.isna(reported):
                # the broker printed no cost basis for this pair this month;
                # ranked BELOW the quantity band deliberately, because a
                # carry-forward row still reports quantity and that
                # comparison stays valid
                band = "reported_unknown"
            else:
                band = classify_basis(reconstructed, reported,
                                      tol_pct=allow.get(pair),
                                      tolerances=tolerances)
                if band in ("watch", "error"):
                    grp = open_lots[
                        (open_lots["account_id"] == pair[0])
                        & (open_lots["instrument_key"] == pair[1])]
                    window = _accretion_window(grp, as_of_ts)
                    if window is not None and (
                            window[0] - LOT_OK_USD <= reported
                            <= window[1] + LOT_OK_USD):
                        band = "accretion_ok"
                    else:
                        delta = _wash_consistent_delta(
                            realizations, pair, grp)
                        if delta > 0.0 and abs(
                                (reported - reconstructed) - delta
                                ) <= LOT_OK_USD:
                            band = "wash_consistent"
        else:
            reconstructed, band = 0.0, "uncovered"
        diff = reconstructed - reported
        rows.append({"account_id": pair[0], "instrument_key": pair[1],
                     "month": month, "reconstructed": reconstructed,
                     "reported": reported, "diff_usd": diff,
                     "diff_pct": (100.0 * diff / reported if reported > 0
                                  else np.nan),
                     "band": band,
                     "reconstructed_qty": rec_qty, "reported_qty": rep_qty,
                     "qty_diff": rec_qty - rep_qty,
                     "basis_evidence": evidence.get(pair, "reconstructed")})
    for pair, reconstructed in recon.items():
        if pair in seen:
            continue
        if pair in basis_unknown:
            reconstructed = np.nan
        rows.append({"account_id": pair[0], "instrument_key": pair[1],
                     "month": month, "reconstructed": reconstructed,
                     "reported": np.nan, "diff_usd": np.nan,
                     "diff_pct": np.nan, "band": "unjoinable",
                     "reconstructed_qty": recon_qty.get(pair, np.nan),
                     "reported_qty": np.nan, "qty_diff": np.nan,
                     "basis_evidence": evidence.get(pair, "reconstructed")})
    return pd.DataFrame(rows, columns=RECON_COLUMNS)


LOT_COLUMNS = OPEN_COLUMNS + ["band"]


def lot_rows(open_lots: pd.DataFrame, recon: pd.DataFrame) -> pd.DataFrame:
    """The lots.csv frame: every open lot, stamped with its instrument's band.

    `basis_evidence` already rides on the lot (slice 9); the band is a property
    of the (account, instrument) comparison, so it is joined on here rather
    than recomputed. Every open lot's pair HAS a reconciliation row by
    construction — a band where positions cover it, `unjoinable` where they do
    not — so a missing one means the caller paired frames from different runs,
    and writing a blank band would ship an unlabelled row.
    """
    if open_lots.empty:
        return pd.DataFrame(columns=LOT_COLUMNS)
    bands = {(r["account_id"], r["instrument_key"]): r["band"]
             for _, r in recon.iterrows()}
    out = open_lots.copy()
    keys = list(zip(out["account_id"], out["instrument_key"]))
    missing = sorted({k for k in keys if k not in bands})
    if missing:
        raise ValueError(
            f"open lots with no reconciliation row: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}")
    out["band"] = [bands[k] for k in keys]
    return out[LOT_COLUMNS]


# --------------------------------------------------------------------------
# Relief check (slice 5): reconstructed relieved basis vs the printed one
# --------------------------------------------------------------------------

RELIEF_COLUMNS = ["broker", "account_id", "instrument_key", "source_row",
                  "close_date", "printed_method", "closing_method",
                  "reconstructed_basis", "printed_basis", "diff", "matched",
                  "status"]
# Flat five cents. Brokers round each lot's relieved basis to the cent, so a
# sell spanning a handful of lots can differ by a few of them. No relative
# term: slice 4 shipped one that could never bind, and the fix was to delete
# it. If a real run shows one is needed, add it WITH a test that fails when
# the term is removed.
_RELIEF_EPS = 0.05


def relief_check(realizations: pd.DataFrame, transactions: pd.DataFrame,
                 exceptions: Optional[pd.DataFrame] = None,
                 resolver: Optional[dict[str, str]] = None,
                 cusip_resolver: Optional[dict[str, str]] = None,
                 fold: Optional[dict[str, str]] = None
                 ) -> pd.DataFrame:
    """Compare each printed-cost sell's reconstructed relieved basis against
    the cost basis the broker printed on the same row.

    `transactions` must be the SAME frame (same index) that built the ledger —
    realizations reference their source row by that index. Pass the same
    resolvers the ledger used, or Fidelity rows are labelled by raw cusip here
    and by resolved symbol in the reconciliation, and the two sections of the
    report name the same instrument differently.

    Sells that cannot be judged stay in the frame with a `status` saying why
    (option, underflowed, no lots closed), so the denominator can never shrink
    silently.
    """
    if (transactions is None or transactions.empty
            or "closing_cost" not in transactions.columns):
        return pd.DataFrame(columns=RELIEF_COLUMNS)
    cost = pd.to_numeric(transactions["closing_cost"], errors="coerce")
    sells = transactions[
        (transactions["transaction_type"].astype(str) == "sell")
        & cost.notna()]
    if sells.empty:
        return pd.DataFrame(columns=RELIEF_COLUMNS)

    closed_basis: dict = {}
    applied: dict = {}
    drawn_status: dict = {}
    if (realizations is not None and not realizations.empty
            and "source_row" in realizations.columns):
        sold = realizations[realizations["close_reason"] == "sell"]
        if not sold.empty:
            closed_basis = sold.groupby("source_row")["basis_closed"].sum(
                min_count=1).to_dict()
            applied = sold.groupby(
                "source_row")["closing_method"].first().to_dict()
            # sells whose relief was DRAWN FROM the printed figure — an
            # unknown pool's composition or an unnamed specific-share lot.
            # Scoring either would be the check marking its own homework; the
            # status says which, so the two never merge in the report.
            if "basis_source" in sold.columns:
                printed_rows = sold[sold["basis_source"].isin(
                    ("printed_pool", "printed_unknowable"))]
                drawn_status = dict(zip(printed_rows["source_row"],
                                        printed_rows["basis_source"]))
    underflowed: set = set()
    if (exceptions is not None and not exceptions.empty
            and "source_row" in exceptions.columns):
        underflowed = set(exceptions.loc[
            exceptions["reason"] == "sell_underflow", "source_row"])

    rows = []
    for idx, row in sells.iterrows():
        printed = float(cost.loc[idx])
        key, _source = resolve_instrument_key(row, resolver or {},
                                              cusip_resolver or {}, fold)
        if is_option_row(row.get("description")):
            status, reconstructed = "excluded_option", np.nan
        elif idx in underflowed:
            status, reconstructed = "excluded_underflow", np.nan
        elif idx in drawn_status:
            status, reconstructed = drawn_status[idx], float(closed_basis[idx])
        elif idx not in closed_basis or pd.isna(closed_basis[idx]):
            status, reconstructed = "excluded_no_lots", np.nan
        else:
            status, reconstructed = "compared", float(closed_basis[idx])
        diff = reconstructed - printed if status == "compared" else np.nan
        rows.append({
            "broker": _clean(row.get("broker")),
            "account_id": row.get("account_id"),
            "instrument_key": key, "source_row": idx,
            "close_date": _clean(row.get("trade_date"))
            or _clean(row.get("settlement_date")),
            # what the broker printed vs what the ledger could execute: a
            # specific-share sell relieved FIFO must not read as a FIFO sell,
            # or the report hides the whole point of the fallback
            "printed_method": relief_method(row),
            "closing_method": applied.get(idx, relief_method(row)),
            "reconstructed_basis": reconstructed,
            "printed_basis": printed,
            "diff": diff,
            "matched": bool(status == "compared" and abs(diff) <= _RELIEF_EPS),
            "status": status})
    return pd.DataFrame(rows, columns=RELIEF_COLUMNS)


def _load_config_dict(name: str) -> dict:
    """Optional dict constant from gitignored config_local.py ({} fallback)."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import config_local as cfg
    except ImportError:
        return {}
    return dict(getattr(cfg, name, {}))


def load_lot_tolerances() -> dict:
    """LOT_RECON_TOLERANCES from config_local (schema in config_example)."""
    return _load_config_dict("LOT_RECON_TOLERANCES")


def load_lot_allowlist() -> dict:
    """LOT_RECON_ALLOWLIST from config_local (schema in config_example)."""
    return _load_config_dict("LOT_RECON_ALLOWLIST")


def load_ticker_history() -> dict:
    """TICKER_HISTORY from config_local (schema in config_example)."""
    return _load_config_dict("TICKER_HISTORY")


def load_corporate_actions() -> dict:
    """CORPORATE_ACTIONS from config_local (schema in config_example)."""
    return _load_config_dict("CORPORATE_ACTIONS")


def load_corporate_identity() -> tuple[dict[str, str], list[dict]]:
    """(fold, split events) from config — the one loader entry points call."""
    actions = load_corporate_actions()
    fold = symbol_fold(load_ticker_history(), actions)
    return fold, corporate_split_events(actions, fold)


def average_cost_remaining_basis(rows: pd.DataFrame) -> float:
    """Remaining basis for ONE instrument under average-cost relief.

    Diagnostic only (spec §6 report section 4): buys/reinvestments pool;
    sells remove the pool-average basis; splits add shares. Transfers,
    mergers and exchanges are ignored — instruments involved in those are
    outside this diagnostic's competence and stay FIFO-judged.
    """
    qty = basis = 0.0
    frame = _prepare(rows)
    for _, row in frame.iterrows():
        kind = str(row["transaction_type"])
        quantity, amount = row["quantity"], row["amount"]
        if kind in ("buy", "reinvestment"):
            if pd.notna(quantity) and pd.notna(amount) and quantity > 0:
                qty += quantity
                # same basis convention the FIFO ledger opens lots with, or a
                # bond's accrued interest would make the two conventions
                # incomparable
                basis += bond_principal_basis(quantity, row["price"], amount)
        elif kind == "sell":
            if pd.notna(quantity) and qty > _QTY_EPS:
                take = min(abs(quantity), qty)
                basis -= basis * take / qty
                qty -= take
        elif kind == "stock_split":
            if pd.notna(quantity) and quantity > 0:
                qty += quantity
    return basis
