"""Extended S&P dip series (synthetic id SPX): SPY spliced onto index
history back to 1950 — price via ^GSPC, total return via ^SP500TR (1988+)
and a dividend-accrual synthetic segment before 1988.

Spec: docs/superpowers/specs/2026-07-19-dip-spx-history-extension-design.md.
Feeds ONLY the registered referee experiment (dip_backtest --extended-spx);
no UI surface reads this module.

Yield table provenance: parsers/spx_dividend_yield_monthly.csv is derived
ONE TIME from the public Shiller "Irrational Exuberance" monthly dataset
(ie_data.xls, Yale): yield = D (trailing-12m dividend rate) / P (monthly
average price). Derivation script: scratch/tools/derive_spx_yield_table.py
(scratch/ is gitignored — deliberately uncommitted; this docstring is the
provenance record). Derived 2026-07-19. The synthetic span ends just past
the ^SP500TR start (SYNTH_ANCHOR_PAD_DAYS) and only bars strictly before
that start survive the splice — on real data only yield months ≤ 1988-01
can influence the output; later rows exist for the synthetic-vs-^SP500TR
tracking gate.

Pure functions over injected Series; file IO only in load_extended_spx
(Task 4). No network anywhere in this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from whatif_data import splice_with_proxy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YIELD_CSV = Path(__file__).resolve().parent / "spx_dividend_yield_monthly.csv"

SPX_ID = "SPX"
SPX_START = "1950-01-01"          # locked decision 4 (spec §2)
SYNTH_ANCHOR_PAD_DAYS = 60        # synthetic levels run to sptr_start + pad:
                                  # enough bars past the ^SP500TR start to
                                  # anchor the splice; only bars BEFORE the
                                  # start survive into the series
TRADING_DAYS = 252.0
MIN_OVERLAP_CORR = 0.99           # spec §4c gate 1


def load_yield_table(path: "Path | None" = None) -> pd.Series:
    """Committed monthly trailing-12m dividend yield, indexed by 'YYYY-MM'."""
    p = Path(path) if path is not None else YIELD_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — the committed yield table is missing "
            "(spec 2026-07-19 §4a)")
    df = pd.read_csv(p, dtype={"month": str})
    if not {"month", "yield"} <= set(df.columns):
        raise ValueError(f"{p}: expected columns month,yield")
    s = df.set_index("month")["yield"].astype(float).sort_index()
    if s.index.has_duplicates:
        raise ValueError(f"{p}: duplicate months")
    return s


def synthetic_tr_levels(gspc_close: pd.Series,
                        yields_by_month: pd.Series) -> pd.Series:
    """Synthetic total-return LEVELS from index price + dividend accrual:
    tr_ret_d = price_ret_d + yield_annual(month(d)) / 252 (spec §4b).
    Level base 1.0 — splices rebase, only returns matter downstream. A month
    missing from the table raises (no silent zero-dividend days)."""
    px = gspc_close.dropna().sort_index() if gspc_close is not None \
        else pd.Series(dtype=float)
    if px.empty:
        raise ValueError("synthetic_tr_levels: empty gspc_close")
    months = px.index.strftime("%Y-%m")
    missing = sorted(set(months) - set(yields_by_month.index))
    if missing:
        raise ValueError(
            "synthetic_tr_levels: yield table missing months "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    y = yields_by_month.reindex(months).to_numpy(dtype=float)
    ret = px.pct_change().fillna(0.0).to_numpy(dtype=float) + y / TRADING_DAYS
    return pd.Series(np.cumprod(1.0 + ret), index=px.index,
                     name="SPX_SYNTH_TR")


def build_extended_spx(spy_close: pd.Series, spy_adj: pd.Series,
                       gspc_close: pd.Series, sptr_close: pd.Series,
                       yields_by_month: pd.Series) -> dict:
    """The SPX pair (spec §4b): price = SPY⊕^GSPC (2 segments), tr =
    SPY⊕^SP500TR⊕synthetic (3 segments), every join via splice_with_proxy
    (segment returns verbatim, levels rebased). Returns
    {"price": Series, "tr": Series, "meta": dict} on ONE shared index.
    Joins that fail to extend RAISE — never a silently shorter series."""
    for name, s in (("spy_close", spy_close), ("spy_adj", spy_adj),
                    ("gspc_close", gspc_close), ("sptr_close", sptr_close)):
        if s is None or s.dropna().empty:
            raise ValueError(f"build_extended_spx: empty component {name}")
    spy_c = spy_close.dropna().sort_index()
    spy_a = spy_adj.dropna().sort_index()
    gspc = gspc_close.dropna().sort_index()
    gspc = gspc[gspc.index >= pd.Timestamp(SPX_START)]
    sptr = sptr_close.dropna().sort_index()
    if gspc.empty or gspc.index.min() >= spy_c.index.min():
        raise ValueError("build_extended_spx: ^GSPC does not extend before "
                         "SPY — price join failed")
    if sptr.index.min() >= spy_a.index.min():
        raise ValueError("build_extended_spx: ^SP500TR does not extend "
                         "before SPY — TR join failed")

    price = splice_with_proxy(spy_c, gspc).rename(SPX_ID)
    if price.index.min() != gspc.index.min():
        raise ValueError("build_extended_spx: price splice did not reach "
                         "the index start")

    tr_8893 = splice_with_proxy(spy_a, sptr)
    synth_end = sptr.index.min() + pd.Timedelta(days=SYNTH_ANCHOR_PAD_DAYS)
    synth = synthetic_tr_levels(gspc[gspc.index <= synth_end],
                                yields_by_month)
    if synth.index.min() >= tr_8893.index.min():
        raise ValueError("build_extended_spx: synthetic segment does not "
                         "extend before ^SP500TR — TR join failed")
    tr = splice_with_proxy(tr_8893, synth).rename(SPX_ID)

    # Price calendar is the master; stray holiday mismatches (^SP500TR vs
    # ^GSPC sessions) forward-fill INSIDE a segment. A leading gap means a
    # broken join — raise, never degrade (spec §5).
    tr = tr.reindex(price.index).ffill()
    if tr.isna().any():
        raise ValueError("build_extended_spx: TR has leading gaps after "
                         "reindex to the price index")
    meta = {
        "id": SPX_ID,
        "start": str(price.index.min().date()),
        "end": str(price.index.max().date()),
        "price_segments": {"gspc_until": str(spy_c.index.min().date()),
                           "spy_from": str(spy_c.index.min().date())},
        "tr_segments": {"synthetic_until": str(sptr.index.min().date()),
                        "sp500tr_from": str(sptr.index.min().date()),
                        "spy_from": str(spy_a.index.min().date())},
        "components": {"spy_last": str(spy_c.index.max().date()),
                       "gspc_last": str(gspc.index.max().date()),
                       "sptr_last": str(sptr.index.max().date())},
        "yield_span": [str(yields_by_month.index.min()),
                       str(yields_by_month.index.max())],
    }
    return {"price": price, "tr": tr, "meta": meta}


def overlap_return_corr(a: pd.Series, b: pd.Series) -> dict:
    """Daily-return correlation over the shared span (spec §4c gate 1;
    validate_against_polygon precedent)."""
    ra = a.dropna().sort_index().pct_change()
    rb = b.dropna().sort_index().pct_change()
    j = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(j) < 3:
        return {"corr": float("nan"), "n_overlap": int(len(j))}
    return {"corr": float(j.iloc[:, 0].corr(j.iloc[:, 1])),
            "n_overlap": int(len(j))}


def tracking_measurements(synth_levels: pd.Series,
                          actual_levels: pd.Series) -> dict:
    """Synthetic-vs-actual TR construction error over the shared span
    (spec §4c gate 2): |annualized-return gap| and p95 of |rolling-252d
    cum-return difference|, both in bps."""
    j = pd.concat([synth_levels.dropna().sort_index(),
                   actual_levels.dropna().sort_index()],
                  axis=1, join="inner").dropna()
    n = len(j)
    if n < 253:
        return {"ann_diff_bps": float("nan"),
                "p95_roll252_bps": float("nan"), "n_overlap": int(n)}
    rs = j.iloc[:, 0].pct_change().dropna()
    ra = j.iloc[:, 1].pct_change().dropna()
    g_s = float((1.0 + rs).prod() ** (TRADING_DAYS / len(rs)) - 1.0)
    g_a = float((1.0 + ra).prod() ** (TRADING_DAYS / len(ra)) - 1.0)
    roll_s = (1.0 + rs).rolling(252).apply(np.prod, raw=True)
    roll_a = (1.0 + ra).rolling(252).apply(np.prod, raw=True)
    d = (roll_s - roll_a).abs().dropna()
    return {"ann_diff_bps": abs(g_s - g_a) * 1e4,
            "p95_roll252_bps": float(np.quantile(d.to_numpy(), 0.95)) * 1e4,
            "n_overlap": int(n)}


def load_extended_spx(data_dir: "Path | str",
                      yield_path: "Path | None" = None) -> dict:
    """Build the SPX pair from the data dir (spec §4b): SPY from
    dip_history.csv, indices from the dip_index_history.csv sidecar, yields
    from the committed table. Returns build_extended_spx's dict plus
    "components" (the raw input Series, for the data gates)."""
    d = Path(data_dir)
    hist_csv = d / "dip_history.csv"
    idx_csv = d / "dip_index_history.csv"
    for p in (hist_csv, idx_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run "
                "`py parsers/fetch_dip_history.py --write`")
    hist = pd.read_csv(hist_csv, parse_dates=["date"])
    idx = pd.read_csv(idx_csv, parse_dates=["date"])

    def _sym(df, path, sym, col):
        sub = df[df["symbol"].astype(str).str.upper() == sym.upper()]
        if sub.empty:
            raise ValueError(
                f"{sym} not in {path} — refetch with "
                "`py parsers/fetch_dip_history.py --write`")
        return sub.sort_values("date").set_index("date")[col].astype(float)

    comp = {"spy_close": _sym(hist, hist_csv, "SPY", "close"),
            "spy_adj": _sym(hist, hist_csv, "SPY", "adj_close"),
            "gspc_close": _sym(idx, idx_csv, "^GSPC", "close"),
            "sptr_close": _sym(idx, idx_csv, "^SP500TR", "close"),
            "yields": load_yield_table(yield_path)}
    built = build_extended_spx(comp["spy_close"], comp["spy_adj"],
                               comp["gspc_close"], comp["sptr_close"],
                               comp["yields"])
    return {**built, "components": comp}


# Gate-1 split-window constants (spec Update 2026-07-20, TK decision):
# full-window corr is a gross-misalignment tripwire; the strict 0.99
# floor (MIN_OVERLAP_CORR) binds the post-decimalization window only —
# US equities quoted in 1/8-1/16 fractions until Apr 2001, and that tick
# noise decorrelates 1990s daily returns without any level drift.
MIN_OVERLAP_CORR_FULL = 0.98
MODERN_CORR_START = "2002-01-01"
# Frozen at registration (spec §4c; freeze rule: measured * 1.5, rounded up
# to 5/25 bps). Measured on the real 1988+ overlap: see the spec Update
# section. A regen of these bounds is a change-control event.
TRACKING_MAX_ANN_BPS = 10.0
TRACKING_MAX_P95_BPS = 50.0


def _corr_gate(a: pd.Series, b: pd.Series) -> dict:
    """Split-window corr gate (spec Update 2026-07-20): full-window floor
    MIN_OVERLAP_CORR_FULL catches gross misalignment; the 2002+ window
    must clear the strict MIN_OVERLAP_CORR."""
    m = pd.Timestamp(MODERN_CORR_START)
    full = overlap_return_corr(a, b)
    modern = overlap_return_corr(a[a.index >= m], b[b.index >= m])
    return {"corr": full["corr"], "n_overlap": full["n_overlap"],
            "corr_2002p": modern["corr"],
            "n_overlap_2002p": modern["n_overlap"],
            "ok": bool(np.isfinite(full["corr"])
                       and full["corr"] >= MIN_OVERLAP_CORR_FULL
                       and np.isfinite(modern["corr"])
                       and modern["corr"] >= MIN_OVERLAP_CORR)}


def run_data_gates(components: dict) -> dict:
    """All three §4c gates on the raw components (gate 1 in its re-shaped
    split-window form, spec Update 2026-07-20). Pass/fail only — the
    registered run is BLOCKED (CLI exit 1) unless all_ok."""
    c = components
    g1 = _corr_gate(c["gspc_close"], c["spy_close"])
    g2 = _corr_gate(c["sptr_close"], c["spy_adj"])
    sptr = c["sptr_close"].dropna().sort_index()
    start = sptr.index.min()
    last_m = c["yields"].index.max()
    end = pd.Timestamp(last_m) + pd.offsets.MonthEnd(0)
    g = c["gspc_close"].dropna().sort_index()
    g = g[(g.index >= start) & (g.index <= end)]
    tm = tracking_measurements(synthetic_tr_levels(g, c["yields"]), sptr)
    tm["ok"] = bool(np.isfinite(tm["ann_diff_bps"])
                    and tm["ann_diff_bps"] <= TRACKING_MAX_ANN_BPS
                    and np.isfinite(tm["p95_roll252_bps"])
                    and tm["p95_roll252_bps"] <= TRACKING_MAX_P95_BPS)
    return {"corr_gspc_vs_spy": g1, "corr_sptr_vs_spy": g2, "tracking": tm,
            "all_ok": bool(g1["ok"] and g2["ok"] and tm["ok"])}
