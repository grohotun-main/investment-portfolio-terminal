"""
Pair internal cross-account transfers in transactions.csv.

Why this exists
---------------
Per-account TWR is fine as-is: every transfer_in/transfer_out is correctly
"external" relative to the specific account, regardless of where the money
came from or went. But portfolio-level TWR needs to wash internal transfers
to zero — when money leaves one of my accounts and lands in another, that's
a re-shuffle of existing money, not new investment activity.

Detection rule (Alpine only — Harbor doesn't put the counterparty in the
description; that work waits on the Cash Flow Summary parser):

  1. The description contains an account-number-shaped token that ISN'T
     the account_id of the row itself. Pattern: ``[A-Z0-9]{3}-\d{5,6}``,
     optionally followed by ``-1`` / ``-2`` (a Alpine sub-account suffix
     for TOD vs IRA legs).
  2. There exists a matching row in the *other* account on the same or
     adjacent settle_date with the same |amount| and opposite sign.

We also catch a second class of internal flow that the Phase-0 parser
mis-labelled as transfer_in/out: same-account sweep movements between
the FCASH core position and a money-market mutual fund (FGMM, SPRXX,
FZFXX). These have no counterparty account in the description but appear
on the same day in the same account with descriptions like
``CASH Transferred ... FCASH IS LIQUID``. Sweeps are flagged with
sweep_internal=True; they're stripped from BOTH account-level and
portfolio-level external-flow sums.

Output
------
Adds two columns to transactions.csv (in place):
  - ``flow_scope``: one of {"external", "internal", "sweep", ""}
      external = bank wire / EFT / outside the portfolio
      internal = paired cross-account transfer between my accounts
      sweep    = same-account FCASH ↔ money-market reshuffle
      "" (empty) = row isn't a flow (buy / sell / dividend / etc.)
  - ``pair_id``: a UUID-style hash that ties the two sides of an internal
      pair together (null for external and sweep). Used by the TWR
      computation as a sanity check.

Run:
    python3 parsers/pair_internal_transfers.py
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
POSITIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "positions.csv"

# config_local lives at the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import config_local as cfg
except ImportError as e:
    raise RuntimeError(
        "config_local.py not found. Copy config_example.py to config_local.py."
    ) from e

# In-kind detection thresholds. A month's NAV-change-net-of-flows that's
# bigger than EITHER of these is treated as suspect (not market movement).
IN_KIND_ABS_THRESHOLD = 50_000.0   # $50K minimum to flag
IN_KIND_PCT_THRESHOLD = 0.15       # 15% of prior NAV

# Counterparty account number: 3 alnum chars + dash + 5-6 digits, optionally
# followed by "-1" / "-2" sub-account suffix. Anchored on word boundaries so
# stray digit strings in the description (timestamps, wire IDs) don't match.
COUNTERPARTY_RE = re.compile(r"\b([A-Z0-9]{3}-\d{5,6})(?:-\d)?\b")

# Description tokens that mark a same-account sweep movement.
SWEEP_PATTERNS = [
    re.compile(r"CASH\s+Transferred", re.I),       # "CASH Transferred ..."
    re.compile(r"FCASH\s+IS\s+LIQUI", re.I),       # FCASH sweep core
    re.compile(r"ALPINE\s+GOVERNMENT\s+MONEY",   # FGMM (sweep MMF)
               re.I),
    re.compile(r"MARGIN\s+TO\s+CASH", re.I),       # same-account margin<->cash
                                                   # journal (WSF-8)
]

FLOW_TYPES = {"transfer_in", "transfer_out", "contribution"}


def _find_counterparty(desc: str, self_account: str) -> str | None:
    """Return the OTHER account number found in the description, or None."""
    if not isinstance(desc, str):
        return None
    for m in COUNTERPARTY_RE.finditer(desc):
        cand = m.group(1)
        if cand != self_account:
            return cand
    return None


def _looks_like_sweep(desc: str) -> bool:
    if not isinstance(desc, str):
        return False
    return any(p.search(desc) for p in SWEEP_PATTERNS)


def _make_pair_id(date: pd.Timestamp, amount_abs: float,
                  acct_a: str, acct_b: str) -> str:
    """Stable 12-char hash from the canonical key of a pair."""
    key = f"{date.date().isoformat()}|{amount_abs:.2f}|" + \
          "|".join(sorted([acct_a, acct_b]))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def pair_transfers(df: pd.DataFrame, date_window: int = 5) -> pd.DataFrame:
    """
    Mutate df in place: add `flow_scope` and `pair_id` columns.

    `date_window`: how many days apart the two sides of a pair may be (in
    practice Alpine settles both sides on the same calendar day, but
    cross-broker journals can lag by 1-2 days — we leave headroom).
    """
    df = df.copy()
    df["flow_scope"] = ""
    df["pair_id"] = pd.Series([None] * len(df), dtype=object)

    is_flow = df["transaction_type"].isin(FLOW_TYPES)
    flow_idx = df.index[is_flow].tolist()

    # First pass: detect sweeps (no counterparty, sweep tokens, same-account
    # pair within 1 day).
    sweep_marked = 0
    for i in flow_idx:
        row = df.loc[i]
        if _find_counterparty(row["description"], row["account_id"]):
            continue  # has a counterparty → handled as internal
        if _looks_like_sweep(row["description"]):
            df.at[i, "flow_scope"] = "sweep"
            sweep_marked += 1

    # Second pass: pair internal cross-account transfers.
    # Build a candidate map: for each row that *names* a counterparty, look
    # for the mirror row in the named account.
    internal_pairs = 0
    consumed: set[int] = set()
    for i in flow_idx:
        if i in consumed or df.at[i, "flow_scope"] == "sweep":
            continue
        row = df.loc[i]
        cp = _find_counterparty(row["description"], row["account_id"])
        if cp is None:
            continue
        amt = row["amount"]
        if pd.isna(amt):
            continue
        # Mirror search: same |amount|, opposite sign, in counterparty
        # account, within date window.
        date = row["settlement_date"]
        candidates = df[
            (df["account_id"] == cp)
            & df["transaction_type"].isin(FLOW_TYPES)
            & (df["settlement_date"] >= date - pd.Timedelta(days=date_window))
            & (df["settlement_date"] <= date + pd.Timedelta(days=date_window))
        ].copy()
        if candidates.empty:
            continue
        # Find exact (or near-exact, within 1 cent) amount match with
        # opposite sign.
        candidates["amount_diff"] = (candidates["amount"] + amt).abs()
        candidates = candidates[candidates["amount_diff"] <= 0.01]
        # Prefer un-consumed and same-day matches; otherwise nearest date.
        candidates = candidates[~candidates.index.isin(consumed)]
        if candidates.empty:
            continue
        # Sort by date proximity then index
        candidates["date_diff"] = (candidates["settlement_date"] - date).abs()
        candidates = candidates.sort_values(["date_diff", "amount_diff"])
        j = candidates.index[0]

        pid = _make_pair_id(date, abs(amt), row["account_id"], cp)
        df.at[i, "flow_scope"] = "internal"
        df.at[j, "flow_scope"] = "internal"
        df.at[i, "pair_id"] = pid
        df.at[j, "pair_id"] = pid
        consumed.add(i)
        consumed.add(j)
        internal_pairs += 1

    # Third pass: everything still un-flagged but is a flow → "external"
    leftover = is_flow & (df["flow_scope"] == "")
    df.loc[leftover, "flow_scope"] = "external"

    # Fourth pass: same-broker same-day +/- pairing. Harbor Cash Flow Summary
    # entries don't name a counterparty in the description, so pass 2 misses
    # them. Match positive vs negative flows on the same day, within the same
    # broker, across distinct accounts. Exact-amount matches first; residuals
    # are paired by splitting rows so partial inter-account moves with an
    # external residual (the leftover that's a real bank wire) are handled
    # cleanly. Partial matches split the larger flow at the smaller flow's
    # amount; the remainder stays external.
    df, new_rows, sameday_pairs = _pair_within_broker_same_day(df, broker="harbor")
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    # Recompute leftover-external count post-pairing.
    final_external = ((df["transaction_type"].isin(FLOW_TYPES))
                       & (df["flow_scope"] == "external")).sum()

    print(f"  Marked {sweep_marked} sweeps")
    print(f"  Paired {internal_pairs} internal cross-account transfers "
          f"(counterparty-named, {internal_pairs*2} rows)")
    print(f"  Paired {sameday_pairs} same-day intra-broker transfers "
          f"(Harbor Cash Flow Summary)")
    print(f"  Remaining external flows: {int(final_external)}")
    return df


def _pair_within_broker_same_day(
    df: pd.DataFrame, broker: str
) -> tuple[pd.DataFrame, list[dict], int]:
    """Pair same-day +/- external flows within a single broker.

    Two-pass within each (broker, date) group:
      1. Exact-amount opposite-sign matches across distinct accounts.
      2. Residuals: greedy partial pairing with row splitting. If one side's
         row exceeds the paired amount, the row is split into a paired
         (internal) and unpaired (external) portion.

    Returns: (df with in-place updates, list of new rows to append, pair count).
    """
    df = df.copy()
    new_rows: list[dict] = []
    pair_count = 0

    flow_mask = (df["broker"] == broker) & (df["flow_scope"] == "external")
    if not flow_mask.any():
        return df, new_rows, 0

    for date, idx_arr in df[flow_mask].groupby("settlement_date").groups.items():
        idx_list = list(idx_arr)
        if len(idx_list) < 2:
            continue

        # Snapshot the queue (idx -> mutable amount). Refresh from df to pick
        # up any splits that happened earlier in this same date group.
        pos: list[dict] = []
        neg: list[dict] = []
        for i in idx_list:
            if df.at[i, "flow_scope"] != "external":
                continue
            amt = df.at[i, "amount"]
            if pd.isna(amt) or amt == 0:
                continue
            entry = {"idx": i, "amount": float(amt),
                     "account": df.at[i, "account_id"]}
            (pos if amt > 0 else neg).append(entry)
        if not pos or not neg:
            continue

        # --- Pass A: exact-amount cross-account match ---
        used: set = set()
        for p in pos:
            if p["idx"] in used:
                continue
            for n in neg:
                if n["idx"] in used:
                    continue
                if p["account"] == n["account"]:
                    continue
                if abs(p["amount"] + n["amount"]) <= 0.01:
                    pid = _make_pair_id(date, abs(p["amount"]),
                                         p["account"], n["account"])
                    df.at[p["idx"], "flow_scope"] = "internal"
                    df.at[n["idx"], "flow_scope"] = "internal"
                    df.at[p["idx"], "pair_id"] = pid
                    df.at[n["idx"], "pair_id"] = pid
                    used.add(p["idx"])
                    used.add(n["idx"])
                    pair_count += 1
                    break

        # --- Pass B: greedy partial pairing with row splitting ---
        pos_q = [p for p in pos if p["idx"] not in used]
        neg_q = [n for n in neg if n["idx"] not in used]
        pos_q.sort(key=lambda x: -x["amount"])
        neg_q.sort(key=lambda x: x["amount"])  # most negative first

        while pos_q and neg_q:
            p, n = pos_q[0], neg_q[0]
            if p["account"] == n["account"]:
                # Try a different counterparty on either side.
                swapped = False
                if len(pos_q) > 1 and pos_q[1]["account"] != n["account"]:
                    pos_q[0], pos_q[1] = pos_q[1], pos_q[0]
                    swapped = True
                elif len(neg_q) > 1 and neg_q[1]["account"] != p["account"]:
                    neg_q[0], neg_q[1] = neg_q[1], neg_q[0]
                    swapped = True
                if not swapped:
                    break
                continue

            pair_amt = min(p["amount"], abs(n["amount"]))
            if pair_amt <= 0.01:
                break
            pid = _make_pair_id(date, pair_amt, p["account"], n["account"])

            # Positive side
            if p["amount"] - pair_amt > 0.01:
                base = df.loc[p["idx"]].to_dict()
                base["amount"] = pair_amt
                base["flow_scope"] = "internal"
                base["pair_id"] = pid
                new_rows.append(base)
                df.at[p["idx"], "amount"] = p["amount"] - pair_amt
                p["amount"] -= pair_amt
            else:
                df.at[p["idx"], "flow_scope"] = "internal"
                df.at[p["idx"], "pair_id"] = pid
                pos_q.pop(0)

            # Negative side
            if abs(n["amount"]) - pair_amt > 0.01:
                base = df.loc[n["idx"]].to_dict()
                base["amount"] = -pair_amt
                base["flow_scope"] = "internal"
                base["pair_id"] = pid
                new_rows.append(base)
                df.at[n["idx"], "amount"] = n["amount"] + pair_amt
                n["amount"] += pair_amt
            else:
                df.at[n["idx"], "flow_scope"] = "internal"
                df.at[n["idx"], "pair_id"] = pid
                neg_q.pop(0)

            pair_count += 1

    return df, new_rows, pair_count


def synthesize_in_kind_flows(
    txn: pd.DataFrame, positions: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Detect & inject internal flow pairs for in-kind cross-account transfers.

    Some securities moves between my accounts (in-kind journals at Harbor or
    Alpine) leave no `transfer_in` / `transfer_out`
    row in `transactions.csv` — only a NAV shift in `positions.csv`. Without
    a synthetic flow, the donor account's TWR shows a fake huge loss and the
    receiver's TWR shows a fake huge gain.

    Heuristic
    ---------
    For each calendar month and each tracked account, compute
        unexplained = NAV_change - recorded_flow_sum
    (For an account's first month, prev_nav is 0; unexplained = NAV - flows.)
    Flag (account, month) as suspect if |unexplained| exceeds BOTH the
    absolute and the percent-of-prev-NAV thresholds. Within each suspect
    month, pair the largest unexplained inflow against the largest
    unexplained outflow; synthesize an internal pair flow for
    min(|inflow|, |outflow|). Residuals are left alone (attributed to
    market movement or to one-sided synthetic onboarding handled elsewhere).
    """
    # Strip any synthetic rows from prior runs so re-running is idempotent
    # (the detector compares NAV deltas vs ALL flows including prior synths).
    if "source_file" in txn.columns:
        txn = txn[txn["source_file"] != "synthetic_in_kind"].copy()

    p = positions.copy()
    p["month"] = p["statement_date"].dt.to_period("M")
    latest = (p.groupby(["account_id", "month"])["statement_date"]
              .max().reset_index().rename(columns={"statement_date": "_keep"}))
    p = p.merge(latest, on=["account_id", "month"])
    p = p[p["statement_date"] == p["_keep"]]
    nav = (p.groupby(["account_id", "broker", "month"])
           .agg(market_value=("market_value", "sum"),
                statement_date=("statement_date", "max"))
           .reset_index())

    t = txn.copy()
    t["month"] = t["settlement_date"].dt.to_period("M")
    flows_by_month = (t[t["transaction_type"].isin(FLOW_TYPES)]
                      .groupby(["account_id", "month"])["amount"]
                      .sum().reset_index().rename(columns={"amount": "flow_sum"}))
    nav = nav.merge(flows_by_month, on=["account_id", "month"], how="left")
    nav["flow_sum"] = nav["flow_sum"].fillna(0.0)
    nav = nav.sort_values(["account_id", "month"]).reset_index(drop=True)
    nav["prev_nav"] = nav.groupby("account_id")["market_value"].shift(1)
    nav["first_month"] = nav["prev_nav"].isna()
    # For first months: unexplained = debut NAV - flows received this month.
    # For subsequent months: unexplained = delta - flows.
    nav["unexplained"] = (
        nav["market_value"]
        - nav["prev_nav"].fillna(0)
        - nav["flow_sum"]
    )
    # Skip pre-tracking-window debuts handled by synthetic_onboarding
    # elsewhere (compute_twr.py adds them at portfolio level). Sourced from
    # config_local so identifiers aren't baked into source.
    pre_tracking_debuts = {(acct, pd.Period(ym, freq="M"))
                           for acct, ym in cfg.PRE_TRACKING_DEBUTS_RAW}
    nav["pre_track"] = nav.apply(
        lambda r: (r["account_id"], r["month"]) in pre_tracking_debuts, axis=1)

    threshold_abs = IN_KIND_ABS_THRESHOLD
    threshold_pct = IN_KIND_PCT_THRESHOLD
    nav["suspect"] = (
        ~nav["pre_track"]
        & (nav["unexplained"].abs() > threshold_abs)
        & (nav["first_month"]
           | (nav["unexplained"].abs() > threshold_pct * nav["prev_nav"].fillna(0)))
    )

    suspects = nav[nav["suspect"]].copy()
    if suspects.empty:
        return txn, 0

    new_rows: list[dict] = []
    paired_count = 0
    for month, grp in suspects.groupby("month"):
        inflows = grp[grp["unexplained"] > 0].sort_values(
            "unexplained", ascending=False).copy()
        outflows = grp[grp["unexplained"] < 0].sort_values(
            "unexplained", ascending=True).copy()  # most negative first
        while len(inflows) and len(outflows):
            recv = inflows.iloc[0]
            send = outflows.iloc[0]
            amt = min(float(recv["unexplained"]), float(-send["unexplained"]))
            if amt < threshold_abs:
                break
            # Settle date = first day of the month. In-kind securities are
            # present for the whole month gaining/losing market value; if we
            # dated this at month-end, modified Dietz would give the flow
            # weight ~0 and treat the entire NAV jump as a "return" on the
            # small prior NAV (a new sleeve going $1K -> $128K would read as
            # +12,000% if the inflow lands on the last day). Dating at day 1
            # makes
            # the flow weight ~1, which is the correct economic model.
            settle = pd.Timestamp(recv["statement_date"]).replace(day=1)
            pid = _make_pair_id(settle, amt,
                                 recv["account_id"], send["account_id"])
            base_recv = {
                "settlement_date": settle,
                "trade_date": settle,
                "broker": recv["broker"],
                "account_id": recv["account_id"],
                "transaction_type": "transfer_in",
                "symbol": pd.NA,
                "cusip": pd.NA,
                "description": (
                    f"Synthetic in-kind transfer from {send['account_id']} "
                    f"(detected from NAV delta)"
                ),
                "quantity": pd.NA,
                "price": pd.NA,
                "amount": amt,
                "source_file": "synthetic_in_kind",
                "flow_scope": "internal",
                "pair_id": pid,
            }
            base_send = {
                **base_recv,
                "broker": send["broker"],
                "account_id": send["account_id"],
                "transaction_type": "transfer_out",
                "description": (
                    f"Synthetic in-kind transfer to {recv['account_id']} "
                    f"(detected from NAV delta)"
                ),
                "amount": -amt,
            }
            new_rows.append(base_recv)
            new_rows.append(base_send)
            paired_count += 1
            # Update residuals
            new_recv_unexp = float(recv["unexplained"]) - amt
            new_send_unexp = float(send["unexplained"]) + amt
            inflows = inflows.iloc[1:].copy()
            outflows = outflows.iloc[1:].copy()
            if new_recv_unexp > threshold_abs:
                row = recv.copy()
                row["unexplained"] = new_recv_unexp
                inflows = pd.concat(
                    [pd.DataFrame([row]), inflows], ignore_index=True
                ).sort_values("unexplained", ascending=False)
            if new_send_unexp < -threshold_abs:
                row = send.copy()
                row["unexplained"] = new_send_unexp
                outflows = pd.concat(
                    [pd.DataFrame([row]), outflows], ignore_index=True
                ).sort_values("unexplained", ascending=True)

    if not new_rows:
        return txn, 0

    print("  Synthetic in-kind transfer pairs (NAV delta inference):")
    for r in new_rows[::2]:
        sib = next(s for s in new_rows if s["pair_id"] == r["pair_id"]
                   and s["account_id"] != r["account_id"])
        print(f"    {r['settlement_date'].date()}  "
              f"${abs(r['amount']):>10,.2f}  "
              f"{sib['account_id']} -> {r['account_id']}")

    augmented = pd.concat([txn, pd.DataFrame(new_rows)], ignore_index=True)
    return augmented, paired_count


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["settlement_date"])
    print(f"Loaded {len(df)} transactions from {CSV_PATH.name}")

    df2 = pair_transfers(df)

    # In-kind pass — needs positions, runs after explicit pairing
    pos = pd.read_csv(POSITIONS_PATH, parse_dates=["statement_date"])
    df2, in_kind_pairs = synthesize_in_kind_flows(df2, pos)
    if in_kind_pairs:
        print(f"  Synthesized {in_kind_pairs} in-kind transfer pair(s) "
              f"({in_kind_pairs * 2} rows added)")

    # Summary by scope
    print("\nFlow-scope distribution:")
    print(df2["flow_scope"].value_counts(dropna=False).to_string())

    # Sanity: internal pairs should sum to zero per pair_id
    print("\n=== Internal pair sanity check ===")
    pairs = df2[df2["flow_scope"] == "internal"].groupby("pair_id").agg(
        n=("amount", "size"),
        net=("amount", "sum"),
        accts=("account_id", lambda s: sorted(set(s))),
        date=("settlement_date", "min"),
        abs_amt=("amount", lambda s: s.abs().iloc[0]),
    )
    print(f"  {len(pairs)} pairs, max |net|=${pairs['net'].abs().max():.2f} "
          f"(should be 0.00)")
    df2.to_csv(CSV_PATH, index=False)
    if len(pairs) <= 30:
        for pid, p in pairs.iterrows():
            print(f"    {pid}  {p['date'].date()}  ${p['abs_amt']:>10,.2f}  "
                  f"{' <-> '.join(p['accts'])}   net=${p['net']:.2f}")
    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
