"""
Duration-matched ETF proxy mapping for bare-CUSIP US Treasury rungs.

A user-held Treasury (e.g. ``"UNITED STATES TREASURY 01/31/2028 JJ 31"``)
is mapped to a duration-matched Treasury ETF (SGOV / SCHO / IEI / IEF /
TLT) so the risk-tab series uses the ETF's smooth total-return path
rather than the bond's clean-price-only NAV (which jumps on coupon dates
because accrued interest only lands when the coupon is paid). This keeps
β / vol / VaR / drawdown on the bond sleeve realistic.

A NOTE ON NAV: bare-CUSIP bond rows on the Holdings tab show
``market_value = qty × clean_price``. Accrued interest between coupon
dates is NOT added to the per-bond row — it lands as an ``INTEREST``
cash flow in the transactions stream when the coupon is paid. At
statement date the account NAV is correct (clean × qty + accumulated
cash from prior coupons); only the intra-month per-bond display is
technically understated by the accrued amount. On the user's current
17-rung ladder the systematic understatement is ~$4k across ~$400k face
≈ 0.05% of total portfolio NAV — invisible to allocation decisions, so
no accrual is added at the bond row.

The May 2026 audit (PR #26) replaced an earlier all-rungs-to-SGOV
mapping that underestimated duration on intermediate-maturity rungs by
~20× (a 7-year Treasury was being treated as if it had SGOV's ~0.1y
duration). This module is the consolidated home for that mapping and
the regression test in tests/test_treasury_proxy.py locks the per-bucket
boundaries that the May fix established.
"""
from __future__ import annotations

import collections
import re
from collections.abc import Iterable

import pandas as pd

# A JPM Treasury rung's maturity is the LAST mm/dd/yyyy date appearing
# after the word "TREASURY" in the description. Two statement formats exist:
#   old:  "UNITED STATES TREASURY 01/31/2028 JJ 31"          (single date)
#   new:  "UNITED STATES TREASURY NOTE DATED DATE 01/31/2021 01/31/2028"
#         ("DATED DATE <issue> <maturity>" — issue date first, maturity last)
# Taking the LAST date after TREASURY handles both; the old regex anchored
# the date immediately after TREASURY, hit the word NOTE under the new
# format, and silently fell back to SGOV for every rung (WSB-1, ~20x
# duration understatement). Anchoring on TREASURY keeps genuinely
# non-Treasury descriptions on the SGOV fallback. Tolerates extra
# whitespace and M/D/YYYY shorthand.
_TREASURY_RE = re.compile(r"TREASURY", re.IGNORECASE)
_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")


def _maturity_token(description: str) -> str | None:
    """Return the LAST mm/dd/yyyy token after "TREASURY", or ``None``.

    ``None`` means the description is not a parseable Treasury rung — no
    TREASURY anchor, or no date token after it — and the caller should fall
    back to SGOV.
    """
    tm = _TREASURY_RE.search(description)
    if not tm:
        return None
    dates = _DATE_RE.findall(description, tm.end())
    return dates[-1] if dates else None

# Duration-matched proxy for a JPM Treasury rung. The thresholds reflect
# the ETFs' modified duration (rough): SGOV ~0.1y, SCHO ~1.8y, IEI ~4.5y,
# IEF ~7.5y, TLT ~17y. A 100bp rate move on an in-bucket rung therefore
# shows up at roughly the right magnitude in vol / VaR / DR tiles.
#
# Boundary convention is CLOSED-OPEN on the right edge: a rung with
# years-to-maturity EXACTLY at a threshold falls through to the NEXT
# (longer-duration) bucket. So 3.0y → IEI (not SCHO); 7.0y → IEF (not
# IEI). The regression test asserts this explicitly so a future tweak to
# `<=` would be caught.
_TREASURY_PROXY_BUCKETS: list[tuple[float, str]] = [
    (1.0,  "SGOV"),   # < 1y  → 0-3mo Treasury bill ETF
    (3.0,  "SCHO"),   # 1-3y  → 1-3y Treasury ETF
    (7.0,  "IEI"),    # 3-7y  → 3-7y Treasury ETF
    (12.0, "IEF"),    # 7-12y → 7-10y Treasury ETF (close enough)
]
_TREASURY_PROXY_LONG = "TLT"  # ≥ 12y → 20+y Treasury ETF


def treasury_proxy(description: str | float, as_of: pd.Timestamp) -> str:
    """Return the duration-matched Treasury ETF symbol for a JPM Treasury
    rung described by ``description``, evaluated as of ``as_of``.

    Falls back to SGOV when the maturity date can't be parsed — preserves
    the prior behavior on rows that don't expose a maturity in their
    description (TIPS, STRIPS, agency notes that don't follow the JPM
    "TREASURY MM/DD/YYYY" convention).
    """
    if not isinstance(description, str):
        return "SGOV"
    tok = _maturity_token(description)
    if tok is None:
        return "SGOV"
    try:
        maturity = pd.Timestamp(tok)
    except (ValueError, TypeError):
        return "SGOV"
    years = (maturity - as_of).days / 365.25
    if years <= 0:
        return "SGOV"
    for threshold, proxy in _TREASURY_PROXY_BUCKETS:
        if years < threshold:
            return proxy
    return _TREASURY_PROXY_LONG


def treasury_proxy_breakdown(
    descriptions: Iterable, as_of: pd.Timestamp
) -> tuple[collections.Counter, int]:
    """Summarize the duration-proxy assignment for a set of ladder-rung
    ``descriptions``, evaluated as of ``as_of``.

    Returns ``(counts, n_unparsed)`` where ``counts`` maps proxy symbol →
    number of rungs, and ``n_unparsed`` counts rungs that look like
    Treasuries (contain "TREASURY") yet expose no parseable maturity and so
    silently fall back to SGOV — the WSB-1 failure mode. A money-market
    fund or other non-Treasury row that maps to SGOV is NOT counted as
    unparsed. Drives the Risk-tab diagnostic caption so a future
    statement-format drift surfaces instead of mis-bucketing the ladder.
    """
    counts: collections.Counter = collections.Counter()
    n_unparsed = 0
    for d in descriptions:
        counts[treasury_proxy(d, as_of)] += 1
        if (isinstance(d, str) and _TREASURY_RE.search(d)
                and _maturity_token(d) is None):
            n_unparsed += 1
    return counts, n_unparsed
