"""Dry-powder rotation ladder — the dip-verdict sizing state machine.

Spec: docs/superpowers/specs/2026-07-18-dip-rotation-ladder-design.md
(TK-approved 2026-07-18). Start 100% cash; when an eval day enters a rung
(verdict band or the ★ TK-rule flag), deploy that rung's pre-committed
fraction of the REMAINING cash into the ticker at that day's total-return
level; each tranche exits back to cash on the first day the PRICE regains
the drawdown-anchor peak it dipped from. One tranche per rung entry — a
rung re-arms only when its tranche exits. On a single eval day the band
tranche deploys before the ★-rule tranche (deterministic order).

The fractions are part of the REGISTERED claim (the walk-forward ladder
backtest pins them); changing them is a new registered run, not a knob.

Pure functions over date-indexed Series — no I/O, no network. The
walk-forward harness (dip_backtest) and any UI consume this module.
"""
from __future__ import annotations

import pandas as pd

# Engine band ids ("neutral"/"strong" — the UI's Buy/Strong buy) plus the
# ★-rule flag. Fractions are OF REMAINING powder.
LADDER_FRACTIONS = {"neutral": 0.25, "strong": 0.50, "tk_rule": 1.0}

_EPS = 1e-12


def simulate_ladder(price: pd.Series, tr: pd.Series, cash_rets: pd.Series,
                    evals: pd.DataFrame, *,
                    fractions: dict | None = None) -> dict:
    """Run the ladder over ``price``/``tr`` (shared index) with eval-day
    signals.

    ``evals``: DataFrame with columns ``date``, ``band`` and optionally
    ``tk_rule`` (bool). Dates outside the price index are ignored (the
    harness derives eval days from the same index). ``cash_rets``: daily
    decimal returns for the cash leg; dates missing from it carry flat.

    Returns ``{"wealth": Series (daily, starts from 1.0 capital),
    "tranches": list[dict], "summary": dict}``. A never-recovered tranche
    stays open (``exit_date`` None) and is marked at the final TR.
    """
    if not price.index.equals(tr.index):
        raise ValueError("price and tr must share one index")
    fr = dict(LADDER_FRACTIONS if fractions is None else fractions)

    cash_r = (cash_rets.reindex(price.index).fillna(0.0)
              if cash_rets is not None and not cash_rets.empty
              else pd.Series(0.0, index=price.index))
    peak = price.cummax()

    by_date: dict = {}
    if evals is not None and len(evals):
        for _, r in evals.iterrows():
            by_date[pd.Timestamp(r["date"])] = {
                "band": r["band"], "tk_rule": bool(r.get("tk_rule", False))}

    cash = 1.0
    armed = {k: True for k in fr}
    open_tranches: list[dict] = []
    tranches: list[dict] = []
    skipped = 0
    wealth = []
    equity_share_sum = 0.0

    for d in price.index:
        cash *= 1.0 + float(cash_r.loc[d])

        # Exits first — a recovery day frees powder (and re-arms the rung)
        # before any same-day deployment is considered.
        still_open = []
        for t in open_tranches:
            if float(price.loc[d]) >= t["anchor_peak"] - _EPS:
                proceeds = t["units"] * float(tr.loc[d])
                cash += proceeds
                t["exit_date"] = d
                t["round_trip_return"] = (float(tr.loc[d]) / t["entry_tr"]
                                          - 1.0)
                armed[t["rung"]] = True
                tranches.append(t)
            else:
                still_open.append(t)
        open_tranches = still_open

        sig = by_date.get(d)
        if sig is not None:
            rungs = []
            if sig["band"] in fr:
                rungs.append(sig["band"])
            if sig["tk_rule"] and "tk_rule" in fr:
                rungs.append("tk_rule")
            for rung in rungs:                      # band first, then ★-rule
                if not armed.get(rung, False):
                    continue
                if cash <= _EPS:
                    skipped += 1
                    continue
                amt = fr[rung] * cash
                units = amt / float(tr.loc[d])
                cash -= amt
                armed[rung] = False
                open_tranches.append({
                    "rung": rung, "band": sig["band"],
                    "entry_date": d, "entry_tr": float(tr.loc[d]),
                    "anchor_peak": float(peak.loc[d]),
                    "deployed": amt, "units": units,
                    "exit_date": None, "round_trip_return": None,
                })

        eq = sum(t["units"] * float(tr.loc[d]) for t in open_tranches)
        w = cash + eq
        wealth.append(w)
        equity_share_sum += (eq / w) if w > 0 else 0.0

    tranches.extend(open_tranches)                  # never-recovered tail
    wealth_s = pd.Series(wealth, index=price.index)
    n_days = len(price.index)
    summary = {
        "n_tranches": len(tranches),
        "skipped_deploys": skipped,
        "avg_equity_exposure": (equity_share_sum / n_days) if n_days else 0.0,
        "final_wealth": float(wealth_s.iloc[-1]) if n_days else 1.0,
    }
    return {"wealth": wealth_s, "tranches": tranches, "summary": summary}
