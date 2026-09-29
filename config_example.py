"""Template for config_local.py.

How to use:
    1. Copy this file to config_local.py (kept out of git).
    2. Replace every placeholder with your real broker account IDs / tickers.
    3. Run the dashboard normally; app.py imports `config_local` at startup.

Keeping the real values in config_local.py rather than source means the
repository can be public-shared (or shared with collaborators) without
disclosing account identifiers or specific holdings.
"""
from __future__ import annotations

# Account-ID -> bucket label. Multiple IDs can collapse to one bucket.
ACCOUNT_BUCKETS_FIXED = {
    "AAA-XXXXX": "Broker A Account 1",
    "BBB-XXXXX": "Broker A Account 2",
}

# Accounts split by symbol downstream (e.g. transfer-on-death accounts whose
# holdings span multiple display buckets).
TOD_SPLIT_ACCOUNTS = set()                  # e.g. {"X00-000000"}

# Tickers treated as the "core ETF" bucket when a TOD account is split.
TOD_CORE_ETF_SYMBOLS = {"SPY"}            # public tickers; placeholder

# Ticker -> asset_class override for funds the broker may file under the wrong
# statement section. A statement format with no dedicated "Exchange Traded
# Products" section lists ETFs under "Stocks / Common Stock", and the holdings
# pipeline then tags them "equity_stock". This map restores the right class
# at display time (the shared reclass), mirroring the built-in GLD->gold
# commodity override. Applied to every broker and idempotent — when a later
# statement files the security correctly the override returns the same class.
#
# Schema: ticker -> one of "equity_etf", "fixed_income", "equity_stock".
# Commodity ETFs (GLD, IAU, SLV, ...) are handled in parsers/asset_reclass.py
# and do NOT need an entry here.
ETF_TICKER_CLASS = {
    # "SPY":  "equity_etf",      # S&P 500 ETF the broker filed under Common Stock
    # "SGOV": "fixed_income",    # 0-3mo Treasury ETF mis-filed as a stock
}

# Per-account display label.
ACCOUNT_DISPLAY = {}                           # account_id -> display string

# Account-holder surname, used ONLY as a line anchor by the Alpine holdings
# parser to locate the "NAME - ACCOUNT TYPE" statement header and read the
# account type. It is account-holder PII, so the real value lives in
# config_local.py; the parser falls back to a placeholder when this is unset.
ACCOUNT_HOLDER_SURNAME = "SMITH"               # your statements' surname

# Special-case account-IDs referenced inline in app.py.
TLH_ACCOUNT_ID             = ""                # tax-loss-harvest account
TREASURY_LADDER_ACCOUNT_ID = ""
SKIP_FROM_TWR_SUMMARY      = set()             # tiny / noise accounts

# Synthetic onboarding flow — accounts whose money predates the statement
# archive (used to keep cumulative TWR / IRR finite).
SYNTHETIC_ONBOARDING = {}                      # {"AAA-XXXXX": "YYYY-MM"}
PRE_TRACKING_DEBUTS_RAW = []                   # [("AAA-XXXXX", "YYYY-MM")]

# Mapping from PDF "description name" prefix to ticker.
NAME_TO_TICKER = {
    "EXAMPLE COMPANY NAME": "TKR",
}

# Corporate-action history — RENAMES ONLY.
#
# This config handles only one kind of corporate action: a clean ticker
# rename where one company keeps trading as itself under a new symbol
# (e.g. FB → META, COMM → VISN). It does NOT handle spinoffs, mergers,
# acquisitions, cash buyouts, or stock-for-stock deals — those require
# cost-basis allocation math in the transaction pipeline and are out of
# scope here. Misconfiguring a merger as a rename will silently splice
# two unrelated price histories into one column and produce nonsense
# correlations / vol on that name. See the "Spinoffs, mergers..." note
# at the bottom of this block for the full list of unsupported events.
#
# Broker statements typically retag historical positions with the company's
# CURRENT ticker after a rename. But our daily-price feed only has data
# under whatever symbol was active at each point in time. Without a splice,
# a renamed ticker shows up in the dashboard with a short price history
# (artificially low vol, near-zero correlations from fillna-on-missing).
#
# Schema: current_ticker -> list of prior-ticker segments. Each segment is
#   {"prior_symbol": str, "effective_date": "YYYY-MM-DD"}
# meaning "this ticker used to be `prior_symbol` until `effective_date`"
# (i.e. effective_date is the first trading day the NEW ticker is active).
#
# Multi-segment chains are supported. Example: CCC was BBB until 2022-01-03
# and was AAA before that:
#   "CCC": [
#       {"prior_symbol": "BBB", "effective_date": "2022-01-03"},
#       {"prior_symbol": "AAA", "effective_date": "2020-01-02"},
#   ]
# Order in the list does not matter; the splice sorts segments
# chronologically and fills date ranges [prev_effective, this_effective).
#
# When you add a new entry, also re-run:
#   py parsers/fetch_daily_prices.py --write
# so the prior_symbol's history is fetched into data/daily_prices.csv.
# Then restart Streamlit so the dashboard picks up the new config.
#
# The LOT LEDGER (parsers/lot_engine.py, via build_lots/realized_check/
# tax_scanner) also consumes this map: prior symbols FOLD onto the current
# ticker so a lot bought under the old name is closed by sells printed under
# the new one, and joins its positions row. The fold is date-free — a key
# labels one continuing instrument — so an OLD symbol later recycled by a
# DIFFERENT issuer would wrongly merge; entries are owner-curated per
# rename, exactly like the price splice.
#
# Spinoffs, mergers, and cash buyouts are NOT handled by this config — the
# lot engine reads those events from the statement rows themselves
# (merger/redemption/exchange transactions). Splits a broker never prints as
# activity go in CORPORATE_ACTIONS below.
TICKER_HISTORY = {
    # Example — uncomment and edit:
    # "VISN": [{"prior_symbol": "COMM", "effective_date": "2026-01-14"}],
}

# Corporate actions the statements do NOT print as activity rows, applied to
# the lot ledger at replay time (parsers/lot_engine.py). Today one kind:
#
#   "split" — multiply open share counts by `ratio` on `effective_date`
#   (basis unchanged). Alpine prints no split activity at all, so without
#   an entry a split's post-ratio sells underflow and the pre-ratio lot
#   strands. `ratio` is owner-stated, never inferred from quantity gaps.
#   Optional "cusips": identifier aliases the action introduced (a
#   reissued "COM NEW" cusip) — rows keyed by those cusips fold onto the
#   ticker so post-action sells reach the same lots.
#
# Schema: ticker -> list of events, each
#   {"kind": "split", "effective_date": "YYYY-MM-DD", "ratio": float,
#    "cusips": [str, ...]}   # cusips optional
CORPORATE_ACTIONS = {
    # Example — a 10:1 forward split with a reissued cusip:
    # "SMCI": [{"kind": "split", "effective_date": "2024-10-01",
    #           "ratio": 10.0, "cusips": ["86800U302"]}],
}

# Per-account known-noise tolerances for the holdings reconciliation guard
# (parsers/reconcile_holdings.py). At statement ingest the guard compares each
# account's summed extracted position value against the statement's reported
# account total. A drift WITHIN an account's max_pct is classified "known"
# (logged, not flagged); ABOVE it the normal bands resume (WATCH > 0.30%, ERROR
# > 2% AND > $10k), so a *growing* drift still surfaces and an ERROR still
# blocks the write.
#
# Use this only for accounts with a legitimate, recurring residual between
# summed position market value and the reported total — e.g. a direct-indexing
# account's lot-rounding, or a bond-heavy account whose reported total includes
# accrued income / occasionally unpriced positions. Optional: omit the constant
# entirely and every account just uses the default bands.
#
# Schema: account_id -> {"max_pct": float, "reason": str}
# Note: a max_pct at or above the ERROR threshold (2%) would suppress the block
# for that account — keep tolerances well below it.
HOLDINGS_RECON_ALLOWLIST = {
    # "AAA-XXXXX": {"max_pct": 0.6, "reason": "direct-index lot-rounding"},
}

# Accounts that legitimately have no statement for the month being closed
# (closed accounts, TOD/transfer shells that stopped issuing). month_close.py
# exempts these from its carried check; every OTHER carried account BLOCKS the
# close, because a statement that never arrived means the month is incomplete.
# Keep this list short and re-verify it at each close -- a permanently-exempt
# account is one nobody is checking. Real ids live in config_local.py
# (gitignored); absent key -> {} -> every carried account blocks (the safe
# default).
MONTH_CLOSE_EXPECTED_CARRIED = {
    # "AAA-XXXXX": {"reason": "TOD shell, no statements since Apr 2026"},
}

# Buy-the-Dip tab: extra tickers to watch beyond SPY + SCHD and your held
# equities. Symbols are upper-cased and de-duplicated. Real list lives in
# config_local.py (gitignored).
DIP_WATCHLIST = []          # e.g. ["QQQ", "SCHG"]

# Tax-lot reconciliation band overrides (parsers/lot_engine.py). Optional
# keys: ok_usd, watch_pct, error_pct, error_usd (defaults in lot_engine).
LOT_RECON_TOLERANCES = {}

# Known per-instrument basis noise for the lot reconciliation:
# (account_id, instrument_key) -> tol_pct. Deltas within tol_pct band as
# "known" instead of watch/error.
LOT_RECON_ALLOWLIST = {}

# ---------------------------------------------------------------------------
# Tax estimate profile (Tax Management tab, /api/tax/estimate).
# Personal inputs live ONLY in config_local.py (gitignored). Empty dict =
# the tab shows a named "not configured" note; the UI override form can
# also supply these per-session. Schema:
# TAX_PROFILE = {
#     "filing_status": "single",          # or "married_joint"
#     "w2_income": 0.0,                    # calendar-year wages, approx ok
#     "state": "CA",                       # only CA supported
#     "deduction": "standard",             # or a dollar amount (itemized)
#     "carryforward_loss": 0.0,            # prior-year capital loss, >= 0
#     "qualified_dividend_pct": 1.0,       # 0.0-1.0 share of dividends
#     "unknown_term_assumption": "long",   # or "short"
# }
TAX_PROFILE: dict = {}
