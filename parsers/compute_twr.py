"""
Compute monthly modified-Dietz returns per account from positions + transactions.

Modified Dietz formula for one period:
    R = (V_end - V_begin - sum(F)) / (V_begin + sum(F_t × w_t))

where F is each external cash flow (transfer_in/out, contribution) during the
period and w_t = (T - t)/T is its time-weight (T = period length in days,
t = days from period start to the flow).

Two TWR series are produced (uses the `flow_scope` column added by
`pair_internal_transfers.py`):

* **Per-account TWR**: a flow is external if it leaves/arrives at the
  specific account, regardless of where the other side is. So we count
  `flow_scope ∈ {external, internal}` and exclude only `sweep`
  (within-account FCASH ↔ money-market reshuffles).
* **Portfolio-level TWR**: only true external flows count. Internal
  cross-account transfers (`flow_scope == "internal"`) wash to zero
  portfolio-wide and are excluded.

Linking monthly returns: R_total = prod(1 + R_i) - 1
Annualized: R_ann = (1 + R_total)^(12/n_months) - 1

Excluded transaction types from external flows (they are INTERNAL to the
account and net to zero in NAV change):
  buy, sell, reinvestment, dividend, interest, redemption, principal_pmt,
  withholding, merger, stock_split, exchange, option_expire, other

Note (historical): in early versions the JPM Cash Flow Summary parser was
not yet in place, so JPM cross-account journals showed up as un-classified
gaps in the originating account's portfolio-level TWR. The CFS parser now
captures them.
"""
import argparse
import math
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import NamedTuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
POSITIONS_CSV = DATA_DIR / "positions.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"

# Account-shaped constants (synthetic-onboarding map) live in config_local.py
# (gitignored). config_example.py ships as the template.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import config_local as cfg
except ImportError as e:
    raise RuntimeError(
        "config_local.py not found. Copy config_example.py to config_local.py "
        "and fill in your synthetic-onboarding map."
    ) from e

# Flow-type filter (legacy): per-account view treats all transfer_in/out/
# contribution rows as external to that account.
FLOW_TYPES = {"transfer_in", "transfer_out", "contribution"}

# Per-account TWR includes flow_scope ∈ {external, internal}:
#   external = bank wire / EFT / outside the portfolio
#   internal = transferred between *my* accounts (it's still external to
#              this specific account even if it nets to zero portfolio-wide)
# Sweeps are excluded because they're not real cash leaving/arriving — just
# FCASH ↔ money-market reshuffles within one account.
ACCOUNT_FLOW_SCOPES = {"external", "internal"}

# Portfolio-level TWR sees only `external` — internal transfers wash to zero.
PORTFOLIO_FLOW_SCOPES = {"external"}


FIDELITY_COVERAGE_CSV = DATA_DIR / "fidelity_statement_periods.csv"


def _load_fidelity_coverage() -> pd.DataFrame:
    """Read fidelity_statement_periods.csv if present. Empty frame if not.

    Sidecar written by fidelity_txn_parser.py with columns:
        broker, account_id, period_start, period_end, source_file
    """
    if not FIDELITY_COVERAGE_CSV.exists():
        return pd.DataFrame(columns=[
            "broker", "account_id", "period_start", "period_end", "source_file",
        ])
    df = pd.read_csv(FIDELITY_COVERAGE_CSV,
                     parse_dates=["period_start", "period_end"])
    return df


def _account_covered_in_month(account_id: str, month: pd.Period,
                              coverage: pd.DataFrame) -> bool:
    """True iff some Fidelity statement period for `account_id` spans
    `month` end-to-end. A normal single-month statement that *is* the
    month also returns True, but the caller only consults this for
    forward-filled months (which by definition have no positions row),
    so single-month coverage there means the parser saw a statement
    the user didn't ingest into positions.csv — still legitimate."""
    if coverage.empty:
        return False
    rows = coverage[coverage["account_id"] == account_id]
    if rows.empty:
        return False
    month_start = month.to_timestamp(how="start")
    month_end = month.to_timestamp(how="end").normalize()
    spans = (rows["period_start"] <= month_start) & (rows["period_end"] >= month_end)
    return bool(spans.any())


def monthly_navs(positions: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with (account_id, month, statement_date, nav,
    is_real_statement).

    Forward-fills missing monthly statements from an account's first real
    statement through the global latest month — both INTERIOR gaps (a month
    skipped between two real statements) and TRAILING gaps (an account lagging
    the statement frontier). Without the trailing fill, an account that hasn't
    statemented for the live month silently drops out of the portfolio NAV sum
    and books its balance as a phantom loss (WSA-1); carrying it forward (a
    flat 0% month) is the accepted policy and matches the holdings-side
    monthly_normalize. Months before an account's first statement stay absent.

    `is_real_statement` is False on forward-filled rows so downstream
    consumers can surface coverage gaps instead of silently displaying 0%
    return for a gap month.
    """
    p = positions.copy()
    p["month"] = p["statement_date"].dt.to_period("M")
    agg = p.groupby(["account_id", "month"]).agg(
        statement_date=("statement_date", "max"),
        nav=("market_value", "sum"),
    ).reset_index()

    if agg.empty:
        agg["is_real_statement"] = pd.Series(dtype=bool)
        return agg
    agg["is_real_statement"] = True
    all_months = pd.period_range(agg["month"].min(), agg["month"].max(), freq="M")
    global_max = agg["month"].max()
    fills: list[dict] = []
    for acct, g in agg.groupby("account_id"):
        present = set(g["month"])
        a_min = g["month"].min()
        # Interior gaps (a_min < m < account's last real) AND trailing gaps (m
        # after the account's last real statement, up to the global latest
        # month). The trailing fill carries an account lagging the statement
        # frontier so it doesn't drop out of the portfolio NAV sum (WSA-1);
        # mirrors the holdings-side monthly_normalize. Pre-debut months
        # (m <= a_min) stay absent.
        gaps = sorted(m for m in all_months
                      if a_min < m <= global_max and m not in present)
        # `present` holds only REAL statement months. Don't mutate it as we
        # fill gaps — every gap month forward-fills from the most recent real
        # statement, which is the same value the prior gap fill would carry
        # transitively. Adding gap months back into `present` would let a
        # subsequent multi-month gap pick a gap-filled month as `prior_month`
        # and then IndexError on `g[g["month"] == prior_month].iloc[0]`
        # because `g` only contains real statements.
        for gap_m in gaps:
            prior_month = max(m for m in present if m < gap_m)
            prior = g[g["month"] == prior_month].iloc[0]
            fills.append({
                "account_id": acct,
                "month": gap_m,
                "statement_date": gap_m.to_timestamp("M").normalize(),
                "nav": float(prior["nav"]),
                "is_real_statement": False,
            })
    if fills:
        agg = pd.concat([agg, pd.DataFrame(fills)], ignore_index=True)
    return agg.sort_values(["account_id", "month"]).reset_index(drop=True)


def _filter_flows(transactions: pd.DataFrame,
                  scopes: set[str]) -> pd.DataFrame:
    """Return only flow rows whose `flow_scope` is in `scopes`.

    Falls back to the legacy transaction-type filter if `flow_scope` is
    missing (e.g. running against a pre-pairing transactions.csv).
    """
    t = transactions.copy()
    if "flow_scope" in t.columns:
        t = t[t["flow_scope"].isin(scopes)]
    else:
        t = t[t["transaction_type"].isin(FLOW_TYPES)]
    # A flow row with no parseable dollar amount (NaN) carries a direction but
    # no magnitude. modified_dietz's weighted-flow loop does
    # `weighted_flows += amount * w` → NaN, which blanks the whole month's
    # return and silently drops it from the linked cumulative TWR. Drop such
    # rows here so both the per-account and portfolio TWR paths inherit one
    # guard (the IRR cashflow builders apply the same filter inline).
    t = t[t["amount"].notna()]
    t["month"] = t["settlement_date"].dt.to_period("M")
    return t


def external_flows_by_month(transactions: pd.DataFrame) -> pd.DataFrame:
    """Account-level external flows (incl. internal cross-account transfers)."""
    return _filter_flows(transactions, ACCOUNT_FLOW_SCOPES)


def portfolio_flows_by_month(transactions: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-level external flows (excludes internal cross-account pairs)."""
    return _filter_flows(transactions, PORTFOLIO_FLOW_SCOPES)


def modified_dietz_period(v_begin: float, v_end: float, flows: pd.DataFrame,
                          period_start: pd.Timestamp, period_end: pd.Timestamp) -> float:
    """Compute modified-Dietz return for one (account, month).

    flows: dataframe filtered to this account and this month, with columns
           settlement_date, amount (sign per the type → transfer_in positive,
           transfer_out negative, contribution positive).
    """
    if v_begin is None or pd.isna(v_begin):
        # No prior NAV — can't compute return. Caller handles.
        return np.nan
    if v_begin == 0 and len(flows) == 0:
        return np.nan
    T = (period_end - period_start).days
    if T <= 0:
        return np.nan
    sum_flows = flows["amount"].sum() if len(flows) else 0.0
    weighted_flows = 0.0
    for _, f in flows.iterrows():
        t = max(0, (f["settlement_date"] - period_start).days)
        w = max(0.0, min(1.0, (T - t) / T))
        weighted_flows += f["amount"] * w
    denom = v_begin + weighted_flows
    if denom == 0:
        return np.nan
    return (v_end - v_begin - sum_flows) / denom


def compute_monthly_twr(positions: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame keyed by (account_id, month) with columns:
       statement_date, nav, prev_nav, net_external_flow, return_pct."""
    navs = monthly_navs(positions)
    flows = external_flows_by_month(transactions)

    rows = []
    for acct, grp in navs.sort_values(["account_id", "month"]).groupby("account_id"):
        grp = grp.sort_values("month").reset_index(drop=True)
        for i in range(len(grp)):
            month = grp.loc[i, "month"]
            stmt_date = grp.loc[i, "statement_date"]
            nav = grp.loc[i, "nav"]
            is_real = bool(grp.loc[i, "is_real_statement"])
            if i == 0:
                # First month for this account — no prior NAV, so return_pct
                # stays NaN. But net_external_flow gets the debut-month flow
                # sum so per-account NetFlow totals include onboarding
                # deposits. Bounded to the debut month to exclude any stray
                # pre-tracking rows.
                f_first = flows[
                    (flows["account_id"] == acct)
                    & (flows["month"] == month)
                ]
                sum_flow_first = float(f_first["amount"].sum()) if len(f_first) else 0.0
                rows.append({
                    "account_id": acct, "month": month,
                    "statement_date": stmt_date, "nav": nav,
                    "prev_nav": None, "prev_stmt_date": None,
                    "net_external_flow": sum_flow_first, "return_pct": np.nan,
                    "n_flows": len(f_first),
                    "is_real_statement": is_real,
                })
                continue
            prev_stmt_date = grp.loc[i-1, "statement_date"]
            prev_nav = grp.loc[i-1, "nav"]
            # Period: from prev_stmt_date (exclusive) to stmt_date (inclusive)
            # We include flows where prev_stmt_date < settle_date <= stmt_date
            f_sub = flows[
                (flows["account_id"] == acct) &
                (flows["settlement_date"] > prev_stmt_date) &
                (flows["settlement_date"] <= stmt_date)
            ]
            sum_flow = f_sub["amount"].sum() if len(f_sub) else 0.0
            r = modified_dietz_period(prev_nav, nav, f_sub, prev_stmt_date, stmt_date)
            rows.append({
                "account_id": acct, "month": month,
                "statement_date": stmt_date, "nav": nav,
                "prev_nav": prev_nav, "prev_stmt_date": prev_stmt_date,
                "net_external_flow": sum_flow, "return_pct": r,
                "n_flows": len(f_sub),
                "is_real_statement": is_real,
            })
    return pd.DataFrame(rows)


def compute_portfolio_twr(positions: pd.DataFrame,
                          transactions: pd.DataFrame,
                          synthetic_onboarding: dict[str, str] | None = None,
                          ) -> pd.DataFrame:
    """Portfolio-level monthly TWR: NAV summed across all accounts each month,
    with only `external` flows counting as outside-the-portfolio flows.

    synthetic_onboarding: optional ``{account_id: "YYYY-MM"}`` map of accounts
    whose tracking debut should be treated as a synthetic external inflow.
    Used for accounts whose money predates the tracking window (so there's no
    real `transfer_in` we could ever capture). The synthetic flow amount =
    the account's NAV in its debut month, dated at month-end.

    Default sourced from config_local.SYNTHETIC_ONBOARDING.

    DON'T add synthetic flows for accounts seeded from another tracked
    account — those self-cancel at portfolio level (a carve-out lowers the
    donor's NAV by the same amount it raises the recipient's).

    Returns DataFrame keyed by month with columns:
        month, statement_date, nav, prev_nav, net_external_flow, return_pct,
        n_flows, n_accounts_active, new_accounts_in_month, synthetic_flow,
        n_accounts_filled, filled_accounts,
        n_accounts_missing, missing_accounts, combined_statement_accounts

    INVARIANT FOR CONSUMERS: ``net_external_flow`` on the portfolio rollup
    rolls together the real external-flow sum AND any ``synthetic_flow`` for
    that month — modified-Dietz needs the synthetic onboarding counted as an
    inflow so the debut month's return doesn't show as +500% from
    "magically appearing" money. Anyone summing ``net_external_flow`` as
    "total real deposits ever" MUST subtract ``synthetic_flow`` first, or
    use ``transactions[flow_scope == "external"]`` as the source-of-truth
    for real wires (see app.py:2171-2206 for the recommended pattern).
    The per-account ``compute_monthly_twr`` output does NOT inject synthetic
    flows — only the portfolio rollup does.
    """
    if synthetic_onboarding is None:
        synthetic_onboarding = cfg.SYNTHETIC_ONBOARDING

    navs = monthly_navs(positions)
    flows = portfolio_flows_by_month(transactions)

    # Build synthetic-flow records: one row per (account, debut_month).
    synth_rows = []
    for acct, ym in synthetic_onboarding.items():
        debut_navs = navs[(navs["account_id"] == acct)
                          & (navs["month"].astype(str) == ym)]
        if not len(debut_navs):
            continue
        debut_nav = float(debut_navs.iloc[0]["nav"])
        debut_stmt = debut_navs.iloc[0]["statement_date"]
        synth_rows.append({
            "month": pd.Period(ym, freq="M"),
            "statement_date": debut_stmt,
            "amount": debut_nav,
            "account_id": acct,
        })
    synth = pd.DataFrame(synth_rows) if synth_rows else pd.DataFrame(
        columns=["month", "statement_date", "amount", "account_id"])

    # Portfolio-level NAV per month (sum across accounts).
    nav_by_month = navs.groupby("month", as_index=False).agg(
        statement_date=("statement_date", "max"),
        nav=("nav", "sum"),
        n_accounts=("account_id", "nunique"),
    ).sort_values("month").reset_index(drop=True)

    # Coverage tripwire: per month, which accounts were forward-filled
    # because no statement landed at that month-end. Two flavors:
    #   - "combined"  — Fidelity issued one PDF covering Feb 1 → Mar 31
    #                   (so Mar 31 positions row exists; Feb 28 doesn't).
    #                   The parser writes fidelity_statement_periods.csv;
    #                   we cross-reference that here to recognize the
    #                   month was intentionally rolled into a later
    #                   statement rather than silently dropped.
    #   - "missing"   — no statement at all. The genuine gap. The
    #                   dashboard warning only fires for this set.
    filled = navs[~navs["is_real_statement"]]
    coverage = _load_fidelity_coverage()
    filled_by_month: dict[pd.Period, str] = {}
    missing_by_month: dict[pd.Period, str] = {}
    combined_by_month: dict[pd.Period, str] = {}
    missing_count_by_month: dict[pd.Period, int] = {}
    filled_count_by_month: dict[pd.Period, int] = {}
    for m, grp in filled.groupby("month"):
        accts = sorted(grp["account_id"].astype(str).unique())
        covered = [a for a in accts if _account_covered_in_month(a, m, coverage)]
        missing = [a for a in accts if a not in covered]
        filled_by_month[m] = ",".join(accts)
        combined_by_month[m] = ",".join(covered)
        missing_by_month[m] = ",".join(missing)
        filled_count_by_month[m] = len(accts)
        missing_count_by_month[m] = len(missing)

    # Track which months an account first appears in (debut month).
    first_month_by_account = navs.groupby("account_id")["month"].min().reset_index()
    first_month_by_account.columns = ["account_id", "first_month"]
    new_accounts_per_month = (
        first_month_by_account.groupby("first_month")["account_id"]
        .apply(lambda s: ",".join(sorted(s)))
        .to_dict()
    )

    rows = []
    for i in range(len(nav_by_month)):
        month = nav_by_month.loc[i, "month"]
        stmt_date = nav_by_month.loc[i, "statement_date"]
        nav = nav_by_month.loc[i, "nav"]
        n_acct = nav_by_month.loc[i, "n_accounts"]
        new_accts = new_accounts_per_month.get(month, "")
        # Any synthetic onboarding flow for this month?
        synth_this_month = synth[synth["month"] == month] if len(synth) else \
            pd.DataFrame(columns=["amount", "settlement_date"])
        synth_amount = float(synth_this_month["amount"].sum()) \
            if len(synth_this_month) else 0.0

        n_filled = int(filled_count_by_month.get(month, 0))
        filled_accts = filled_by_month.get(month, "")
        n_missing = int(missing_count_by_month.get(month, 0))
        missing_accts = missing_by_month.get(month, "")
        combined_accts = combined_by_month.get(month, "")
        if i == 0:
            # First portfolio month — no prior NAV, return_pct stays NaN.
            # Same display-fix as compute_monthly_twr: capture debut-month
            # real flows (e.g. the debut account's onboarding wires) plus any
            # synthetic onboarding amount so the NetFlow total isn't
            # under-reported.
            f_first = flows[flows["month"] == month].copy()
            if synth_amount != 0.0:
                extra = synth_this_month.rename(
                    columns={"statement_date": "settlement_date"})[
                        ["settlement_date", "amount"]].copy()
                f_first = pd.concat([f_first, extra], ignore_index=True)
            sum_flow_first = float(f_first["amount"].sum()) if len(f_first) else 0.0
            rows.append({
                "month": month, "statement_date": stmt_date, "nav": nav,
                "prev_nav": None, "prev_stmt_date": None,
                "net_external_flow": sum_flow_first, "return_pct": np.nan,
                "n_flows": len(f_first), "n_accounts_active": n_acct,
                "new_accounts_in_month": new_accts,
                "synthetic_flow": synth_amount,
                "n_accounts_filled": n_filled,
                "filled_accounts": filled_accts,
                "n_accounts_missing": n_missing,
                "missing_accounts": missing_accts,
                "combined_statement_accounts": combined_accts,
            })
            continue
        prev_stmt_date = nav_by_month.loc[i-1, "statement_date"]
        prev_nav = nav_by_month.loc[i-1, "nav"]
        f_sub = flows[
            (flows["settlement_date"] > prev_stmt_date)
            & (flows["settlement_date"] <= stmt_date)
        ].copy()
        # Append synthetic flows as additional rows in the per-period flow
        # dataframe — modified_dietz_period reads `amount` and
        # `settlement_date`.
        if synth_amount != 0.0:
            extra = synth_this_month.rename(
                columns={"statement_date": "settlement_date"})[
                    ["settlement_date", "amount"]].copy()
            f_sub = pd.concat([f_sub, extra], ignore_index=True)
        sum_flow = f_sub["amount"].sum() if len(f_sub) else 0.0
        r = modified_dietz_period(prev_nav, nav, f_sub, prev_stmt_date, stmt_date)
        rows.append({
            "month": month, "statement_date": stmt_date, "nav": nav,
            "prev_nav": prev_nav, "prev_stmt_date": prev_stmt_date,
            "net_external_flow": sum_flow, "return_pct": r,
            "n_flows": len(f_sub), "n_accounts_active": n_acct,
            "new_accounts_in_month": new_accts,
            "synthetic_flow": synth_amount,
            "n_accounts_filled": n_filled,
            "filled_accounts": filled_accts,
            "n_accounts_missing": n_missing,
            "missing_accounts": missing_accts,
            "combined_statement_accounts": combined_accts,
        })
    return pd.DataFrame(rows)


def link_returns(returns: pd.Series) -> float:
    """Chain monthly returns to a total cumulative return."""
    valid = returns.dropna()
    if len(valid) == 0:
        return np.nan
    return float(np.prod(1.0 + valid) - 1.0)


def annualize(total_return: float, n_months: int) -> float:
    if pd.isna(total_return) or n_months <= 0:
        return np.nan
    years = n_months / 12.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


# ============================================================================
# IRR (money-weighted return) — companion to TWR
# ============================================================================
# TWR strips out cash-flow timing to measure pure investment skill; IRR is the
# rate the investor actually earned given when they put money in/out. We carry
# both. Per-account IRR uses ACCOUNT_FLOW_SCOPES (same as per-account TWR);
# portfolio IRR uses PORTFOLIO_FLOW_SCOPES + the synthetic-onboarding entries
# for accounts whose money predates the statement archive.

# Sign convention (investor POV):
#   transfer_in / contribution (amount > 0, money INTO account) → cashflow < 0
#   transfer_out (amount < 0, money OUT of account)              → cashflow > 0
#   terminal NAV at end                                          → cashflow > 0
#   synthetic onboarding (amount > 0 in TWR, money phantom-IN)   → cashflow < 0
# Uniformly: cf = -amount; terminal added as +terminal_nav.


def xirr(cashflows: list[float], dates: list[pd.Timestamp],
         guess: float = 0.1) -> float:
    """Solve sum(cf_i / (1+r)^((d_i - d_0)/365)) = 0 for r.

    Returns the annualized rate r. Newton's method from ``guess`` (default
    0.1), with a bisection fallback over the rate bracket [-0.9999, 10.0] if
    Newton fails to converge (typical when cashflow timing is degenerate).

    Root selection: a conventional series (one sign change — deposits then a
    terminal redemption) has a unique root, returned exactly. A sign-
    alternating series can in principle admit multiple NPV roots; this returns
    the FIRST root the solver lands on (Newton's result, else the bisection
    root within the bracket), and NaN when no sign change brackets a root in
    [-0.9999, 10.0]. The portfolio IRR gate's watch bands (>10.0 gain /
    <-0.90 loss) flag any pathological value that slips through.
    """
    if len(cashflows) < 2:
        return float("nan")
    cf = np.asarray(cashflows, dtype=float)
    if not np.all(np.isfinite(cf)):
        # A non-finite cashflow (e.g. an in-kind transfer the parser could not
        # price → NaN amount) makes NPV non-finite at every rate. np.sign(nan)
        # defeats the bisection same-sign guard (nan == nan is False) and
        # silently collapses the solver onto its -0.9999 lower bound. Refuse:
        # an undefined input has an undefined IRR, not a -99.99% return.
        return float("nan")
    if not (np.any(cf > 0) and np.any(cf < 0)):
        # Need both signs (a deposit and a redemption/terminal NAV).
        return float("nan")
    d0 = min(dates)
    t = np.array([(d - d0).days / 365.0 for d in dates], dtype=float)

    def npv(r: float) -> float:
        return float(np.sum(cf / np.power(1.0 + r, t)))

    def dnpv(r: float) -> float:
        return float(np.sum(-t * cf / np.power(1.0 + r, t + 1.0)))

    # Newton's method
    r = guess
    for _ in range(100):
        try:
            f = npv(r)
            if abs(f) < 1e-9:
                return r
            df = dnpv(r)
            if df == 0:
                break
            step = f / df
            # Damp huge steps that would push r below -1 (invalid).
            r_new = r - step
            if r_new <= -0.9999:
                r_new = (r - 0.9999) / 2.0
            if abs(r_new - r) < 1e-10:
                return r_new
            r = r_new
        except (OverflowError, ZeroDivisionError):
            break

    # Bisection fallback over a wide rate range.
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9 or (hi - lo) < 1e-10:
            return mid
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (lo + hi) / 2.0


def _account_cashflows(account_id: str,
                       positions: pd.DataFrame,
                       transactions: pd.DataFrame,
                       scopes: set[str],
                       synthetic_onboarding: dict[str, str] | None = None,
                       start_date: pd.Timestamp | None = None,
                       ) -> tuple[list[float], list[pd.Timestamp], float]:
    """Build (cashflows, dates, terminal_nav) for one account.

    Cashflows are signed from the investor's POV (deposits negative, withdrawals
    positive). The terminal NAV at the account's last statement_date is appended
    as a positive cashflow (treats the position as if liquidated then). If the
    account has a synthetic-onboarding entry, that gets added as a negative
    cashflow at the debut month's statement_date.

    When ``start_date`` is provided, the IRR window is truncated: flows before
    start_date are dropped, synthetic_onboarding is bypassed, and the NAV held
    at the last statement strictly before start_date is injected as a synthetic
    deposit (negative cashflow) on start_date. This makes IRR-since-cutoff
    well-defined — otherwise the algorithm would treat the cutoff-date NAV as
    a free gift and return an inflated rate.
    """
    nav = monthly_navs(positions)
    nav_acct = nav[nav["account_id"] == account_id].sort_values("month")
    if nav_acct.empty:
        return [], [], 0.0

    nav_at_cutoff = 0.0
    if start_date is not None:
        pre_cutoff = nav_acct[nav_acct["statement_date"] < start_date]
        if not pre_cutoff.empty:
            nav_at_cutoff = float(pre_cutoff.iloc[-1]["nav"])
        nav_acct = nav_acct[nav_acct["statement_date"] >= start_date]
        if nav_acct.empty:
            return [], [], 0.0

    terminal_nav = float(nav_acct.iloc[-1]["nav"])
    terminal_date = nav_acct.iloc[-1]["statement_date"]
    debut_stmt = nav_acct.iloc[0]["statement_date"]
    debut_month = nav_acct.iloc[0]["month"]

    flows = transactions.copy()
    if "flow_scope" in flows.columns:
        flows = flows[flows["flow_scope"].isin(scopes)]
    else:
        flows = flows[flows["transaction_type"].isin(FLOW_TYPES)]
    flows = flows[flows["account_id"] == account_id]
    if start_date is not None:
        flows = flows[flows["settlement_date"] >= start_date]
    # A flow row with no parseable dollar amount (NaN) carries a direction but
    # no magnitude — it cannot be a cashflow. Drop it rather than append NaN,
    # which poisons every NPV and floors xirr to -0.9999.
    flows = flows[flows["amount"].notna()]

    cf_list: list[float] = []
    dt_list: list[pd.Timestamp] = []

    if start_date is not None and nav_at_cutoff > 0:
        cf_list.append(-nav_at_cutoff)
        dt_list.append(pd.Timestamp(start_date))

    for _, row in flows.iterrows():
        cf_list.append(-float(row["amount"]))
        dt_list.append(row["settlement_date"])

    # synthetic_onboarding still applies under a cutoff when the account had
    # no pre-cutoff NAV — i.e. it materialized inside the truncated window. In
    # that case nav_at_cutoff is 0 and the synthetic deposit is the only thing
    # seeding the cashflow series. Suppress only when nav_at_cutoff > 0
    # (else we'd double-count: nav_at_cutoff already captures the pre-cutoff
    # value the synthetic entry was modeling).
    if (synthetic_onboarding and account_id in synthetic_onboarding
            and (start_date is None or nav_at_cutoff == 0.0)):
        ym = synthetic_onboarding[account_id]
        if str(debut_month) == ym:
            cf_list.append(-float(nav_acct.iloc[0]["nav"]))
            dt_list.append(debut_stmt)

    cf_list.append(terminal_nav)
    dt_list.append(terminal_date)
    return cf_list, dt_list, terminal_nav


def compute_account_irr(positions: pd.DataFrame,
                        transactions: pd.DataFrame,
                        synthetic_onboarding: dict[str, str] | None = None,
                        start_date: pd.Timestamp | None = None,
                        ) -> pd.DataFrame:
    """One row per account: account_id, start_date, end_date, terminal_nav,
    n_cashflows, total_deposits, total_withdrawals, irr (annualized).

    `synthetic_onboarding` defaults to config_local.SYNTHETIC_ONBOARDING —
    same as portfolio TWR. Without it, accounts whose deposit history
    predates the statement archive have undefined IRR: the cashflow series
    shows only outflows + a still-large terminal NAV → infinite return.

    `start_date` (optional) truncates each account's window to
    [start_date, terminal]. NAV-at-cutoff is injected as a synthetic deposit;
    synthetic_onboarding is bypassed.
    """
    if synthetic_onboarding is None:
        synthetic_onboarding = cfg.SYNTHETIC_ONBOARDING
    nav = monthly_navs(positions)
    rows = []
    for acct in sorted(nav["account_id"].unique()):
        cf, dt, term_nav = _account_cashflows(
            acct, positions, transactions, ACCOUNT_FLOW_SCOPES,
            synthetic_onboarding=synthetic_onboarding,
            start_date=start_date,
        )
        if not cf:
            continue
        deposits = float(sum(-c for c in cf if c < 0))
        withdrawals = float(sum(c for c in cf[:-1] if c > 0))
        irr = xirr(cf, dt)
        # Window months — flag short-window IRRs as noisy. The dashboard
        # already suppresses annualized values for n_months < 12 in the
        # Per-account table; expose the count here so downstream users
        # inspecting the CSV directly can apply the same filter.
        start = min(dt)
        end = max(dt)
        window_months = max(1, int(
            (end.year - start.year) * 12 + (end.month - start.month)
        ))
        rows.append({
            "account_id": acct,
            "start_date": start,
            "end_date": end,
            "window_months": window_months,
            "terminal_nav": term_nav,
            "n_cashflows": len(cf),
            "total_deposits": deposits,
            "total_withdrawals": withdrawals,
            "irr": irr,
        })
    return pd.DataFrame(rows)


def compute_portfolio_irr(positions: pd.DataFrame,
                          transactions: pd.DataFrame,
                          synthetic_onboarding: dict[str, str] | None = None,
                          start_date: pd.Timestamp | None = None,
                          scoped: bool = False,
                          ) -> dict:
    """Single portfolio-level IRR. Uses PORTFOLIO_FLOW_SCOPES (external only —
    internal cross-account transfers wash to $0 portfolio-wide) plus the
    synthetic-onboarding amount for any account whose money predates the
    statement archive. Terminal NAV = sum of all per-account terminal NAVs.

    When `start_date` is provided, the IRR is computed over the truncated
    window: external flows before start_date are dropped, synthetic_onboarding
    is bypassed, and the portfolio NAV held at the last statement strictly
    before start_date (summed across all real accounts) is injected as a
    single synthetic deposit on start_date.

    When `scoped=True` the transactions frame is treated as an account SCOPE
    (e.g. one broker's accounts, pre-narrowed by the caller) rather than the
    whole book: internal-transfer legs are kept per `pair_id` GROUP, not
    per-row. `_make_pair_id` hashes `date|amount|sorted(accounts)` with no
    occurrence counter, so two same-day same-|amount| transfers between the
    same two accounts legitimately share one `pair_id` — a group can hold
    more than one pair's worth of legs. A group whose in-scope legs net to
    ~$0 (tolerance $0.005) is fully paired inside the scope and washes
    exactly as the whole-book path assumes; any other group's non-zero net
    is money crossing the scope boundary, so ALL of its legs are kept (they
    sum to the net, sign already correct: cf = -amount). Null-pair-id
    internal rows are included defensively (their wash partner is unproven
    in scope). No-op when the `flow_scope` or `pair_id` column is absent.
    Default False keeps every existing caller — including the ingest-time
    canonical row — byte-identical.
    """
    if synthetic_onboarding is None:
        synthetic_onboarding = cfg.SYNTHETIC_ONBOARDING
    nav = monthly_navs(positions)
    if nav.empty:
        return {"irr": float("nan"), "n_cashflows": 0,
                "terminal_nav": 0.0, "start_date": None, "end_date": None}

    # Terminal NAV = sum of latest NAV per account (using each account's last
    # statement_date, which the forward-fill in monthly_navs has already aligned).
    last_per_acct = nav.sort_values("month").groupby("account_id").tail(1)
    terminal_nav = float(last_per_acct["nav"].sum())
    terminal_date = last_per_acct["statement_date"].max()

    flows = transactions.copy()
    if "flow_scope" in flows.columns:
        keep = flows["flow_scope"].isin(PORTFOLIO_FLOW_SCOPES)
        if scoped and "pair_id" in flows.columns:
            internal = flows["flow_scope"] == "internal"
            if internal.any():
                # A pair_id GROUP (not a count): _make_pair_id hashes
                # date|amount|accounts, so two same-day same-amount transfers
                # between the same accounts legitimately share one id. A group
                # whose in-scope legs net to ~$0 is fully paired inside the
                # scope and washes; any non-zero net is money crossing the
                # scope boundary — keep ALL its legs (they sum to the net).
                net = (flows.loc[internal].groupby("pair_id")["amount"]
                       .transform("sum"))
                unbalanced = pd.Series(False, index=flows.index)
                unbalanced.loc[internal] = net.abs().gt(0.005)
                keep = keep | (internal & (flows["pair_id"].isna()
                                           | unbalanced))
        flows = flows[keep]
    else:
        flows = flows[flows["transaction_type"].isin(FLOW_TYPES)]
    if start_date is not None:
        flows = flows[flows["settlement_date"] >= start_date]
    # Drop flows with no parseable amount (NaN) — an unpriced in-kind journal
    # has a direction but no magnitude; see _account_cashflows.
    flows = flows[flows["amount"].notna()]

    cf_list: list[float] = []
    dt_list: list[pd.Timestamp] = []

    if start_date is not None:
        pre_cutoff = nav[nav["statement_date"] < start_date]
        portfolio_nav_at_cutoff = 0.0
        if not pre_cutoff.empty:
            last_pre = (pre_cutoff.sort_values("month")
                        .groupby("account_id").tail(1))
            portfolio_nav_at_cutoff = float(last_pre["nav"].sum())
        if portfolio_nav_at_cutoff > 0:
            cf_list.append(-portfolio_nav_at_cutoff)
            dt_list.append(pd.Timestamp(start_date))

    for _, row in flows.iterrows():
        cf_list.append(-float(row["amount"]))
        dt_list.append(row["settlement_date"])

    # Synthetic onboarding still applies under a cutoff for accounts whose
    # debut is post-cutoff (no pre-cutoff NAV from `nav` to capture). When the
    # account existed pre-cutoff, portfolio_nav_at_cutoff already sums in
    # its NAV; including the synthetic entry there would double-count.
    for acct, ym in synthetic_onboarding.items():
        debut_rows = nav[(nav["account_id"] == acct)
                         & (nav["month"].astype(str) == ym)]
        if not len(debut_rows):
            continue
        debut_stmt = debut_rows.iloc[0]["statement_date"]
        if start_date is not None and pd.Timestamp(debut_stmt) < start_date:
            # Pre-cutoff debut — captured by portfolio_nav_at_cutoff.
            continue
        cf_list.append(-float(debut_rows.iloc[0]["nav"]))
        dt_list.append(debut_stmt)

    cf_list.append(terminal_nav)
    dt_list.append(terminal_date)

    if len(cf_list) < 2:
        return {"irr": float("nan"), "n_cashflows": len(cf_list),
                "terminal_nav": terminal_nav,
                "start_date": dt_list[0] if dt_list else None,
                "end_date": dt_list[-1] if dt_list else None,
                "total_deposits": 0.0, "total_withdrawals": 0.0}

    irr = xirr(cf_list, dt_list)
    return {
        "irr": irr,
        "n_cashflows": len(cf_list),
        "terminal_nav": terminal_nav,
        "start_date": min(dt_list),
        "end_date": max(dt_list),
        "total_deposits": float(sum(-c for c in cf_list if c < 0)),
        "total_withdrawals": float(sum(c for c in cf_list[:-1] if c > 0)),
    }


# --------------------------------------------------------------------------
# IRR sanity gate
# --------------------------------------------------------------------------
# Bands each account's computed IRR so derived-metric corruption fails loud at
# ingest instead of silently reaching the dashboard — the same role
# reconcile_holdings (PR #129) plays for NAV. PR #147's NaN-amount in-kind flows
# floored the portfolio + two JPM accounts' IRR at xirr's -0.9999 bisection
# bound (shown as -99.99%); that was fixed at the source, but nothing validated
# the IRR step the way NAV is validated pre-write. This is that missing gate.
#
# Only the -0.9999 floor blocks: it is a *finite, fake* return the dashboard
# renders as a plausible "-99.99%" — the silent-corruption case the gate exists
# to stop. A non-finite (NaN) IRR is instead *undefined* and renders as "n/a"
# (honest, not misleading) and is legitimately produced by accounts with too
# few cashflows (e.g. a live single-cashflow account)
# — so it is surfaced as "watch", never blocked, or every month with such an
# account would falsely abort ingest. A genuinely large negative return (the
# a live account at -99.75%) sits clear of the tight floor tolerance and is
# likewise "watch", not "error".
IRR_FLOOR = -0.9999       # xirr's bisection lower bound; an IRR pinned here is
IRR_FLOOR_TOL = 1e-6      #   the solver hitting the wall, not a -99.99% return
IRR_WATCH_LOSS = -0.90    # a finite IRR at/below this is a very large loss, or
IRR_WATCH_GAIN = 10.0     #   at/above IRR_WATCH_GAIN implausibly high: advisory


class IrrSanityRow(NamedTuple):
    account_id: str
    irr: float
    band: str          # "ok" | "watch" | "error"


def classify_irr(irr: float, *,
                 floor: float = IRR_FLOOR,
                 floor_tol: float = IRR_FLOOR_TOL,
                 watch_loss: float = IRR_WATCH_LOSS,
                 watch_gain: float = IRR_WATCH_GAIN) -> str:
    """The sanity band for one account's computed annualized IRR.

    "error" — the corruption signature PR #147 produced: an IRR pinned at xirr's
        -0.9999 bisection floor (within `floor_tol`), which the dashboard
        renders as a plausible-looking but *fake* "-99.99%" return. Blocks the
        write.
    "watch" — surfaced, never blocks. Either (a) a non-finite IRR (NaN / ±inf):
        an *undefined* IRR shown as "n/a" — honest, not a fake return, and
        legitimately produced by a single-cashflow new account or a fully
        -withdrawn sleeve; or (b) a finite but extreme value: a loss at or below
        `watch_loss`, or a gain at or above `watch_gain`. This band keeps a
        legitimately large negative return (and an honest n/a) out of "error".
    "ok" — everything else.
    """
    try:
        irr = float(irr)
    except (TypeError, ValueError):
        return "error"
    if not math.isfinite(irr):
        return "watch"
    if irr <= floor + floor_tol:
        return "error"
    if irr <= watch_loss or irr >= watch_gain:
        return "watch"
    return "ok"


def check_irr_sanity(irr_df: pd.DataFrame) -> list[IrrSanityRow]:
    """One IrrSanityRow per account in the computed IRR table, banded by
    `classify_irr`. Row order is preserved (per-account rows then the PORTFOLIO
    row, as written to irr_per_account.csv)."""
    rows: list[IrrSanityRow] = []
    for _, r in irr_df.iterrows():
        try:
            irr = float(r["irr"])
        except (TypeError, ValueError):
            irr = float("nan")
        rows.append(IrrSanityRow(str(r["account_id"]), irr, classify_irr(irr)))
    return rows


def format_irr_sanity(rows: list[IrrSanityRow]) -> str:
    """Aligned per-account IRR sanity table, mirroring
    reconcile_holdings.format_table."""
    if not rows:
        return "(no accounts to check)"
    header = f"{'account':<25} {'IRR':>12}  band"
    lines = [header, "-" * len(header)]
    for r in rows:
        irr_str = f"{r.irr * 100:>+10.2f}%" if math.isfinite(r.irr) else "       n/a"
        lines.append(f"{r.account_id:<25} {irr_str}  {r.band}")
    return "\n".join(lines)


def run_irr_gate(irr_out: pd.DataFrame, *, force: bool = False) -> int:
    """Print the IRR sanity table and return an exit code: 3 if any account's
    IRR is in the "error" band (pinned at the -0.9999 floor) and not `force`,
    else 0.

    The caller writes irr_per_account.csv only on a 0, so a corrupt IRR never
    overwrites the prior-good file — it carries forward, exactly as a
    reconciliation-blocked holding does in upsert_holdings. A non-zero return
    aborts the ingest pipeline (ingest_statements stops on the first non-zero
    step)."""
    rows = check_irr_sanity(irr_out)
    print("\nIRR sanity check — computed IRR per account:")
    print(format_irr_sanity(rows))
    bad = [r for r in rows if r.band == "error"]
    if not bad:
        return 0
    names = ", ".join(r.account_id for r in bad)
    if force:
        print(f"\n[FORCED] writing {len(bad)} corrupt IRR row(s) anyway "
              f"({names}).", file=sys.stderr)
        return 0
    print(
        f"\n[BLOCKED] {len(bad)} account(s) have a -99.99%-floored IRR "
        f"({names}) — the signature of an unpriced / NaN-amount flow poisoning "
        f"the solver (see PR #147), not a real return. irr_per_account.csv was "
        f"NOT overwritten and keeps its prior values; the dashboard carries the "
        f"last good IRR forward. Investigate the flagged account's flows, then "
        f"re-run; use --force to write anyway.",
        file=sys.stderr,
    )
    return 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute per-account + portfolio TWR and IRR from "
                    "positions.csv + transactions.csv.")
    ap.add_argument(
        "--force", action="store_true",
        help="Write irr_per_account.csv even when the IRR sanity gate flags an "
             "IRR pinned at the -0.9999 floor (the PR #147 corruption "
             "signature). Default blocks the write and exits non-zero so the "
             "ingest pipeline aborts.")
    args = ap.parse_args(argv)

    pos = pd.read_csv(POSITIONS_CSV, parse_dates=["statement_date"])
    txn = pd.read_csv(TRANSACTIONS_CSV, parse_dates=["settlement_date"])
    print(f"Loaded: {len(pos)} positions, {len(txn)} transactions")
    if "flow_scope" in txn.columns:
        print(f"  flow_scope distribution: "
              f"{txn['flow_scope'].value_counts(dropna=False).to_dict()}")
    else:
        print("  WARNING: flow_scope column missing — falling back to legacy "
              "transaction-type filter (run pair_internal_transfers.py first "
              "to get a clean portfolio TWR).")

    # ============================ Per-account TWR ============================
    twr = compute_monthly_twr(pos, txn)
    print(f"\nMonthly TWR table: {len(twr)} (account, month) rows")
    print(f"By account:")
    print(twr.groupby("account_id").size().to_string())

    print("\n=== Per-account total cumulative TWR ===")
    print(f"{'Account':25s} {'#Mo':>5s} {'StartNAV':>14s} {'EndNAV':>14s} "
          f"{'NetFlow':>14s} {'TotalRet':>10s} {'Ann':>10s}")
    for acct, g in twr.groupby("account_id"):
        g = g.sort_values("month")
        valid = g.dropna(subset=["return_pct"])
        if len(valid) == 0:
            continue
        total = link_returns(valid["return_pct"])
        n_months = len(valid)
        ann = annualize(total, n_months)
        start_nav = g.iloc[0]["nav"]
        end_nav = g.iloc[-1]["nav"]
        net_flow = g["net_external_flow"].sum()
        print(f"{acct:25s} {n_months:>5d} ${start_nav:>13,.0f} ${end_nav:>13,.0f} "
              f"${net_flow:>13,.0f}  {total*100:>+8.1f}%  {ann*100:>+8.1f}%")

    # Pure-period detail: the account with a strictly-earliest start month,
    # restricted to the window before any other account joined. Skipped when
    # the earliest-start month is shared.
    starts = twr.groupby("account_id")["month"].min()
    earliest_start = starts.min()
    tied = starts[starts == earliest_start].index.tolist()
    if len(tied) != 1:
        print(f"\n=== Pure period skipped (multiple accounts tied for "
              f"earliest start: {tied}) ===")
    else:
        first_acct = tied[0]
        print(f"\n=== {first_acct} pure period ===")
        sub = twr[twr["account_id"] == first_acct].copy()
        sub["month_str"] = sub["month"].astype(str)
        later = twr[~twr["account_id"].isin(tied)]
        cutoff = (later["month"].astype(str).min()
                  if not later.empty else sub["month_str"].max() + "x")
        pure = sub[sub["month_str"] < cutoff].copy()
        if not pure.empty:
            pure_valid = pure.dropna(subset=["return_pct"])
            pure_total = link_returns(pure_valid["return_pct"])
            pure_ann = annualize(pure_total, len(pure_valid))
            pure_flow = pure["net_external_flow"].sum()
            start = pure.iloc[0]["nav"]
            end = pure.iloc[-1]["nav"]
            print(f"  Start NAV: ${start:,.2f}")
            print(f"  End NAV:   ${end:,.2f}")
            print(f"  Net external flow:    ${pure_flow:,.2f}")
            print(f"  Months tracked:       {len(pure_valid)}")
            print(f"  Cumulative TWR:       {pure_total*100:+.2f}%")
            print(f"  Annualized TWR:       {pure_ann*100:+.2f}%")

            print("\n  Top 5 monthly gains:")
            for _, r in pure.dropna(subset=["return_pct"]).nlargest(5, "return_pct").iterrows():
                print(f"    {r['month']}  {r['return_pct']*100:+.2f}%  "
                      f"(flow: ${r['net_external_flow']:+,.0f})")
            print("  Top 5 monthly losses:")
            for _, r in pure.dropna(subset=["return_pct"]).nsmallest(5, "return_pct").iterrows():
                print(f"    {r['month']}  {r['return_pct']*100:+.2f}%  "
                      f"(flow: ${r['net_external_flow']:+,.0f})")
        else:
            print("  (no pure-period months — first_acct shares start with later accounts)")

    out = DATA_DIR / "twr_monthly.csv"
    twr_out = twr.copy()
    twr_out["month"] = twr_out["month"].astype(str)
    twr_out.to_csv(out, index=False)
    print(f"\nWrote: {out}")

    # ============================ Portfolio TWR =============================
    print("\n" + "=" * 72)
    print("PORTFOLIO-LEVEL TWR")
    print("=" * 72)
    port = compute_portfolio_twr(pos, txn)
    print(f"Months: {len(port)}")

    port_valid = port.dropna(subset=["return_pct"]).copy()
    port_total = link_returns(port_valid["return_pct"])
    port_ann = annualize(port_total, len(port_valid))
    start_nav = port.iloc[0]["nav"]
    end_nav = port.iloc[-1]["nav"]
    net_flow = port["net_external_flow"].sum(skipna=True)

    print(f"\n  Start NAV (first month): ${start_nav:>14,.2f}  "
          f"({port.iloc[0]['statement_date'].date()})")
    print(f"  End NAV (last month):    ${end_nav:>14,.2f}  "
          f"({port.iloc[-1]['statement_date'].date()})")
    print(f"  Net external flow:       ${net_flow:>14,.2f}  "
          f"(true new money into the portfolio)")
    print(f"  NAV change net of flows: ${end_nav - start_nav - net_flow:>14,.2f}")
    print(f"  Months tracked:          {len(port_valid)}")
    print(f"  Cumulative portfolio TWR: {port_total*100:+.2f}%")
    print(f"  Annualized portfolio TWR: {port_ann*100:+.2f}%")

    # Flag suspect debut months (new account joined with material un-captured
    # inflow). "Material" = the new accounts' combined debut NAV exceeds 5%
    # of the prior portfolio NAV AND wasn't matched by a synthetic or
    # external flow in the same month.
    navs_local = monthly_navs(pos)

    def _new_account_debut_nav(row) -> float:
        if not row["new_accounts_in_month"]:
            return 0.0
        accts = row["new_accounts_in_month"].split(",")
        return float(
            navs_local[(navs_local["account_id"].isin(accts))
                       & (navs_local["month"] == row["month"])]["nav"].sum()
        )
    port_check = port.copy()
    port_check["debut_nav"] = port_check.apply(_new_account_debut_nav, axis=1)
    # Note: net_external_flow already includes synthetic_flow (compute_portfolio_twr
    # adds synthetic rows into the period's flow set before summing). So we use
    # net_external_flow directly as the "accounted for" denominator.
    port_check["accounted_for"] = port_check["net_external_flow"].fillna(0.0)
    suspect = port_check[
        (port_check["new_accounts_in_month"] != "")
        & (port_check["new_accounts_in_month"].notna())
        & (port_check["return_pct"].abs() > 0.05)
        & (port_check["debut_nav"] > 0.05 * port_check["prev_nav"].fillna(np.inf))
        & ((port_check["debut_nav"] - port_check["accounted_for"]).abs()
           > 0.5 * port_check["debut_nav"])
    ]
    if len(suspect):
        print("\n  Suspect months (new accounts joined + outsized return + "
              "material un-captured inflow):")
        for _, r in suspect.iterrows():
            print(f"    {r['month']}  ret={r['return_pct']*100:>+7.2f}%  "
                  f"flow=${r['net_external_flow']:>+12,.0f}  "
                  f"synth=${r['synthetic_flow']:>+12,.0f}  "
                  f"debut_nav=${r['debut_nav']:>12,.0f}  "
                  f"new={r['new_accounts_in_month']}")
    else:
        print("\n  No suspect months — all account onboardings appear to be "
              "either captured flows, synthetic onboarding flows, or "
              "in-kind cross-account transfers that self-cancel.")

    port_out = DATA_DIR / "twr_portfolio.csv"
    port_out_df = port.copy()
    port_out_df["month"] = port_out_df["month"].astype(str)
    port_out_df.to_csv(port_out, index=False)
    print(f"\nWrote: {port_out}")

    # ============================ IRR (money-weighted) ======================
    print("\n" + "=" * 72)
    print("IRR (money-weighted return) — complement to TWR")
    print("=" * 72)

    irr_acct = compute_account_irr(pos, txn)
    print(f"\n{'Account':25s} {'#CF':>5s} {'Deposits':>14s} {'Withdrawals':>14s} "
          f"{'TermNAV':>14s} {'IRR':>10s}")
    for _, r in irr_acct.sort_values("irr", ascending=False).iterrows():
        irr_str = f"{r['irr']*100:>+8.2f}%" if not pd.isna(r["irr"]) else "       n/a"
        print(f"{r['account_id']:25s} {r['n_cashflows']:>5d} "
              f"${r['total_deposits']:>13,.0f} ${r['total_withdrawals']:>13,.0f} "
              f"${r['terminal_nav']:>13,.0f}  {irr_str}")

    port_irr = compute_portfolio_irr(pos, txn)
    irr_str = (f"{port_irr['irr']*100:>+8.2f}%"
               if not pd.isna(port_irr["irr"]) else "       n/a")
    print(f"\n{'PORTFOLIO (external only)':25s} {port_irr['n_cashflows']:>5d} "
          f"${port_irr['total_deposits']:>13,.0f} "
          f"${port_irr['total_withdrawals']:>13,.0f} "
          f"${port_irr['terminal_nav']:>13,.0f}  {irr_str}")

    # Save: per-account rows + one PORTFOLIO row.
    irr_out = irr_acct.copy()
    irr_out["start_date"] = pd.to_datetime(irr_out["start_date"]).dt.strftime("%Y-%m-%d")
    irr_out["end_date"] = pd.to_datetime(irr_out["end_date"]).dt.strftime("%Y-%m-%d")
    port_row = pd.DataFrame([{
        "account_id": "PORTFOLIO",
        "start_date": pd.Timestamp(port_irr["start_date"]).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(port_irr["end_date"]).strftime("%Y-%m-%d"),
        "terminal_nav": port_irr["terminal_nav"],
        "n_cashflows": port_irr["n_cashflows"],
        "total_deposits": port_irr["total_deposits"],
        "total_withdrawals": port_irr["total_withdrawals"],
        "irr": port_irr["irr"],
    }])
    irr_out = pd.concat([irr_out, port_row], ignore_index=True)

    # ---- IRR sanity gate: corruption fails loud, never reaches the dashboard
    code = run_irr_gate(irr_out, force=args.force)
    if code != 0:
        return code

    irr_csv = DATA_DIR / "irr_per_account.csv"
    irr_out.to_csv(irr_csv, index=False)
    print(f"\nWrote: {irr_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
