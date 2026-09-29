"""Calendar-year tax estimate attributable to the portfolio.

Pure module: no I/O, no config imports. The service assembles inputs;
this file owns every tax number. Headline definition (spec §2): tax
attributable to the portfolio = bracket-walk(W-2 + portfolio) minus
bracket-walk(W-2 alone), so wages fill the lower brackets and portfolio
income is taxed at the rates it actually lands on.

Table provenance (update yearly, one file):
- Federal 2026: IRS Rev. Proc. 2025-32 (brackets, standard deduction,
  LT thresholds).
- CA: FTB 2025 schedule (the 2026 indexation publishes fall 2026 —
  the staleness is stated in the assumptions list; ~3%/yr, well inside
  the tab's +/-10% claim). MHST threshold is statutory, unindexed.
- NIIT thresholds are statutory and unindexed.
"""
from __future__ import annotations

import re

TAX_YEAR = 2026

# (rate, upper_edge_of_taxable_income); None = unbounded top band.
FED_BRACKETS = {
    "single": [(0.10, 12_400), (0.12, 50_400), (0.22, 105_700),
               (0.24, 201_775), (0.32, 256_225), (0.35, 640_600),
               (0.37, None)],
    "married_joint": [(0.10, 24_800), (0.12, 100_800), (0.22, 211_400),
                      (0.24, 403_550), (0.32, 512_450), (0.35, 768_700),
                      (0.37, None)],
}
FED_LT_THRESHOLDS = {
    "single": [(0.00, 49_450), (0.15, 545_500), (0.20, None)],
    "married_joint": [(0.00, 98_900), (0.15, 613_700), (0.20, None)],
}
FED_STD_DEDUCTION = {"single": 16_100.0, "married_joint": 32_200.0}

CA_BRACKETS = {
    "single": [(0.01, 11_079), (0.02, 26_264), (0.04, 41_452),
               (0.06, 57_542), (0.08, 72_724), (0.093, 371_479),
               (0.103, 445_771), (0.113, 742_953), (0.123, None)],
    "married_joint": [(0.01, 22_158), (0.02, 52_528), (0.04, 82_904),
                      (0.06, 115_084), (0.08, 145_448), (0.093, 742_958),
                      (0.103, 891_542), (0.113, 1_485_906),
                      (0.123, None)],
}
CA_STD_DEDUCTION = {"single": 5_706.0, "married_joint": 11_412.0}
CA_MHST_RATE, CA_MHST_THRESHOLD = 0.01, 1_000_000.0   # all statuses

NIIT_RATE = 0.038
NIIT_THRESHOLD = {"single": 200_000.0, "married_joint": 250_000.0}

CAPITAL_LOSS_OFFSET_CAP = 3_000.0

# CUSIP shape for US Treasury issues (bills 912797..., notes 91282C...):
# a structural rule, not an instrument list (CLT-PI-061).
_TREASURY_CUSIP_RE = re.compile(r"^912[0-9A-Z]{6}$")
_TREASURY_DESC_RE = re.compile(
    r"\b(UNITED STATES TREASURY|US TREASURY|U S TREASURY)\b")


def is_treasury_income(symbol: str, description: str) -> bool:
    """US-govt-obligation income row (CA-exempt interest)."""
    sym = (symbol or "").strip().upper()
    if _TREASURY_CUSIP_RE.match(sym):
        return True
    return bool(_TREASURY_DESC_RE.search((description or "").upper()))


def _walk(table, amount: float) -> float:
    """Total tax on `amount` of taxable income under a bracket table."""
    tax, lo = 0.0, 0.0
    if amount <= 0:
        return 0.0
    for rate, edge in table:
        hi = amount if edge is None else min(amount, float(edge))
        if hi > lo:
            tax += rate * (hi - lo)
        if edge is None or amount <= edge:
            break
        lo = float(edge)
    return tax


def _marginal(table, amount: float) -> float:
    """The rate the NEXT dollar above `amount` pays."""
    for rate, edge in table:
        if edge is None or amount < edge:
            return rate
    return table[-1][0]


def _layer(table, base: float, size: float) -> float:
    """Tax on a layer of `size` stacked on top of `base` taxable income."""
    if size <= 0:
        return 0.0
    return _walk(table, base + size) - _walk(table, base)


def _mhst(taxable: float) -> float:
    return CA_MHST_RATE * max(0.0, taxable - CA_MHST_THRESHOLD)


def _net_terms(realized: dict, sim_legs, assumption: str,
               carryforward_loss: float) -> dict:
    """Spec §4.3.1 — cross-netting, the $3,000 ordinary offset, and the
    carryforward remainder. Unknown-term gains join the assumed term;
    the prior-year carryforward is long-term by stated assumption."""
    sim_st = sum(float(s["gl"]) for s in (sim_legs or [])
                 if s["term"] == "short")
    sim_lt = sum(float(s["gl"]) for s in (sim_legs or [])
                 if s["term"] == "long")
    unknown = float(realized.get("unknown", 0.0))
    st = float(realized.get("short", 0.0)) + sim_st
    lt = (float(realized.get("long", 0.0)) + sim_lt
          - max(0.0, carryforward_loss))
    if assumption == "short":
        st += unknown
    else:
        lt += unknown
    if st < 0.0 < lt:
        absorb = min(-st, lt)
        st += absorb
        lt -= absorb
    elif lt < 0.0 < st:
        absorb = min(-lt, st)
        lt += absorb
        st -= absorb
    combined = st + lt
    offset = carry_out = 0.0
    if combined < 0.0:
        offset = max(combined, -CAPITAL_LOSS_OFFSET_CAP)
        carry_out = combined - offset
    return {"st_net": round(st, 2), "lt_net": round(lt, 2),
            "st_taxable": max(st, 0.0), "lt_taxable": max(lt, 0.0),
            "ordinary_offset": round(offset, 2),
            "carryforward_out": round(carry_out, 2)}


def _core(fs: str, w2: float, ded: float, ca_ded: float, qdp: float,
          realized: dict, income: dict, sim_legs, assumption: str,
          carryforward_loss: float) -> dict:
    net = _net_terms(realized, sim_legs, assumption, carryforward_loss)
    interest = float(income.get("interest", 0.0))
    treasury = float(income.get("treasury_interest", 0.0))
    divs = float(income.get("dividends", 0.0))
    ord_inv = (interest + divs * (1.0 - qdp) + net["st_taxable"]
               + net["ordinary_offset"])
    pref = net["lt_taxable"] + divs * qdp

    fed_base = max(0.0, w2 - ded)
    fed_with = max(0.0, w2 + ord_inv - ded)
    fed_ord = _walk(FED_BRACKETS[fs], fed_with) - _walk(FED_BRACKETS[fs],
                                                        fed_base)
    fed_pref = _layer(FED_LT_THRESHOLDS[fs], fed_with, pref)

    # CA: everything ordinary (no LT preference, qualified divs taxed),
    # treasury interest exempt.
    ca_inv = ((interest - treasury) + divs + net["st_taxable"]
              + net["lt_taxable"] + net["ordinary_offset"])
    ca_base = max(0.0, w2 - ca_ded)
    ca_with = max(0.0, w2 + ca_inv - ca_ded)
    ca = (_walk(CA_BRACKETS[fs], ca_with) - _walk(CA_BRACKETS[fs], ca_base)
          + _mhst(ca_with) - _mhst(ca_base))

    nii = (interest + divs + net["st_taxable"] + net["lt_taxable"]
           + net["ordinary_offset"])
    magi = w2 + nii
    niit = NIIT_RATE * max(0.0, min(nii, max(0.0,
                                             magi - NIIT_THRESHOLD[fs])))
    ftc = max(0.0, -float(income.get("withholding", 0.0)))
    total = fed_ord + fed_pref + ca + niit - ftc
    return {"net": net, "fed_ord": fed_ord, "fed_pref": fed_pref,
            "ca": ca, "niit": niit, "ftc": ftc, "total": total,
            "fed_with": fed_with, "pref": pref, "ca_with": ca_with}


def estimate_year_tax(profile: dict, realized: dict, income: dict,
                      sim_legs: list[dict] | None = None) -> dict:
    """Spec §4 contract. Whole-dollar ints out; ValueError on an
    unsupported filing_status/state (the service maps it to a named
    degrade, never a 500)."""
    fs = str(profile.get("filing_status", "")).strip()
    if fs not in FED_BRACKETS:
        raise ValueError(f"unsupported filing_status: {fs!r} "
                         f"(supported: {sorted(FED_BRACKETS)})")
    state = str(profile.get("state", "")).strip().upper()
    if state != "CA":
        raise ValueError(f"unsupported state: {state!r} (only CA)")
    w2 = float(profile.get("w2_income", 0.0))
    ded_in = profile.get("deduction", "standard")
    if isinstance(ded_in, str) and ded_in.strip().lower() == "standard":
        ded, ca_ded = FED_STD_DEDUCTION[fs], CA_STD_DEDUCTION[fs]
        ded_note = "standard deduction (federal and CA amounts)"
    else:
        ded = ca_ded = float(ded_in)
        ded_note = ("itemized deduction applied to BOTH jurisdictions "
                    "(approximation)")
    qdp = min(1.0, max(0.0, float(
        profile.get("qualified_dividend_pct", 1.0))))
    cf = max(0.0, float(profile.get("carryforward_loss", 0.0)))
    assumption = str(profile.get("unknown_term_assumption", "long"))
    if assumption not in ("long", "short"):
        assumption = "long"
    other = "short" if assumption == "long" else "long"

    main = _core(fs, w2, ded, ca_ded, qdp, realized, income, sim_legs,
                 assumption, cf)
    alt = _core(fs, w2, ded, ca_ded, qdp, realized, income, sim_legs,
                other, cf)

    assumptions = [
        "tax attributable to the portfolio: bracket-walk(W-2 + "
        "portfolio) minus bracket-walk(W-2 alone)",
        ded_note,
        f"unknown-term realized gains treated as {assumption}-term",
        "prior-year carryforward treated as long-term",
        f"qualified dividend share: {qdp:.0%} of dividends",
        "MAGI approximated as W-2 + net investment income",
        "foreign withholding credited dollar-for-dollar (approximation)",
        "federal tables: 2026 (Rev. Proc. 2025-32); CA tables: 2025 "
        "schedule (2026 indexation pending)",
        "AMT and quarterly payment timing are out of scope",
    ]
    # Compose totals from rounded leaves (not independently rounded sums)
    # so display layers cross-foot exactly: total = federal + state + niit
    # - ftc, federal = federal_ordinary + federal_preferential. This
    # invariant holds because sums derive from rounded components, never a
    # second independent rounding (same lesson in realized_check.py).
    main_fed_ord = round(main["fed_ord"])
    main_fed_pref = round(main["fed_pref"])
    main_state = round(main["ca"])
    main_niit = round(main["niit"])
    main_ftc = round(main["ftc"])
    main_federal = main_fed_ord + main_fed_pref
    main_total = main_federal + main_state + main_niit - main_ftc

    # Compute alt pass's leaves identically so swing_if_other preserves
    # symmetry: swing = alt_total - main_total, where both totals are
    # composed from their own leaves.
    alt_fed_ord = round(alt["fed_ord"])
    alt_fed_pref = round(alt["fed_pref"])
    alt_state = round(alt["ca"])
    alt_niit = round(alt["niit"])
    alt_ftc = round(alt["ftc"])
    alt_federal = alt_fed_ord + alt_fed_pref
    alt_total = alt_federal + alt_state + alt_niit - alt_ftc

    return {
        "year": TAX_YEAR,
        "total": main_total,
        "federal": main_federal,
        "federal_ordinary": main_fed_ord,
        "federal_preferential": main_fed_pref,
        "state": main_state,
        "niit": main_niit,
        "ftc": main_ftc,
        "netting": {k: main["net"][k] for k in
                    ("st_net", "lt_net", "ordinary_offset",
                     "carryforward_out")},
        "marginal": {
            "fed_ordinary": _marginal(FED_BRACKETS[fs], main["fed_with"]),
            "fed_preferential": _marginal(FED_LT_THRESHOLDS[fs],
                                          main["fed_with"] + main["pref"]),
            "ca": (_marginal(CA_BRACKETS[fs], main["ca_with"])
                   + (CA_MHST_RATE
                      if main["ca_with"] > CA_MHST_THRESHOLD else 0.0)),
        },
        "unknown_term": {
            "assumption": assumption,
            "amount": round(float(realized.get("unknown", 0.0)), 2),
            "swing_if_other": alt_total - main_total,
        },
        "assumptions": assumptions,
    }
