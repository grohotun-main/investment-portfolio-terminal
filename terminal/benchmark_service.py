# terminal/benchmark_service.py
"""Pure data seam for the MERIDIAN Terminal "Performance vs Benchmark" tab.

Re-expresses app.py._render_benchmark_body (3122-3483). The comparison math
lives in parsers/compare_to_benchmark.py (build_twr_comparison /
build_irr_comparison) — importable, no Streamlit — so this module only
reproduces the *inputs* (the Holdings-filtered TWR view + the resolved
benchmark's TR lookup — SPY or the 60/40 SPY/AGG blend) and shapes the
result into a JSON-native, allow_nan=False-clean view dict.
Reuses holdings_service + performance_service helpers (already parity-gated by
test_terminal_performance.py), so the two UIs cannot drift on those numbers.
"""
from __future__ import annotations

import pandas as pd

from terminal import holdings_service as hs
from terminal import performance_service as ps
from terminal.holdings_service import fmt_money, fmt_pct

from compare_to_benchmark import build_twr_comparison, build_irr_comparison
from risk_metrics import aggregate_periodic_returns

from period_returns import window_returns
from interim_stub import (InterimStub, bench_stub_return, chain, stub_block,
                          to_date_cagr, to_date_span, ytd_to_date)
from period_returns import WINDOWS

BENCH_TICKER = "SPY"   # legacy default; per-call label comes from hs.BENCH_SHORT
SYNTHETIC_ONBOARDING = ps.SYNTHETIC_ONBOARDING

# Lowercase granularity keys mirror the JSON contract the JS segmented control
# consumes (same convention as performance_service._periodic).
_GRAN_FREQ = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}
_GRAN_XFMT = {"monthly": "%Y-%m", "quarterly": "%Y-%m", "yearly": "%Y"}

# Chart colors (mirror app.py's theme.CHART_PORTFOLIO / CHART_BENCH on the
# terminal's dark palette — kept literal here, like holdings/performance JS).
CHART_PORT = "#4DA3F5"
CHART_BENCH = "#8794A9"
CHART_PROVISIONAL = "#E0A030"   # dashed amber tail: the provisional stub period


# --------------------------------------------------------------------------- #
# Caption / label helpers — mirror app.py 3129-3158. The terminal HAS a
# broker selector (#323); ``broker_scope`` is None on the canonical real
# book and a tuple of broker display labels when narrowed, so the captions
# name the actual scope instead of hardcoding the whole book (DA-D-3).
# --------------------------------------------------------------------------- #
def _subset_label(acct_opts: list[dict], class_opts: list[dict],
                  account: str | list[str], asset_class: str | list[str],
                  broker_scope: tuple[str, ...] | None = None,
                  canonical: str = "Portfolio") -> str:
    base = (f"Combined portfolio ({canonical})" if not broker_scope
            else f"Broker scope: {' + '.join(broker_scope)}")
    bits = []
    acct_ids = hs._normalize_filter_ids(account)
    class_ids = hs._normalize_filter_ids(asset_class)
    if acct_ids != ["all"]:
        labels = [next((o["label"] for o in acct_opts if o["id"] == i), i)
                  for i in acct_ids]
        bits.append("Account: " + ", ".join(labels))
    if class_ids != ["all"]:
        labels = [next((o["label"] for o in class_opts if o["id"] == i), i)
                  for i in class_ids]
        bits.append("Asset-class: " + ", ".join(labels))
    return base + (" · " + " · ".join(bits) if bits else "")


def _filter_caption(holdings_filter_active: bool, short: str,
                    broker_scope: tuple[str, ...] | None = None,
                    canonical: str = "Portfolio") -> str:
    if holdings_filter_active:
        return (f"Account / Asset-class filters in the sidebar select the subset "
                f"compared against {short} total return.")
    if broker_scope:
        return (f"{' + '.join(broker_scope)} only vs {short} total return — "
                f"use the Account / Asset-class filters to compare a subset.")
    return (f"Full combined portfolio ({canonical}) vs {short} total "
            f"return — use the Account / Asset-class filters to compare a subset.")


def _methodology(s: dict, holdings_filter_active: bool, short: str,
                 broker_scope: tuple[str, ...] | None = None) -> str:
    """Port app.py's 'How this works' body to an HTML string the front-end
    injects via innerHTML. Three tails, mirroring app.py's own branches:
    Holdings-filter slice / broker subset (the NAV-weighted recompute,
    honestly disclosed — the terminal HAS a broker selector since #323) /
    full portfolio (the canonical series — true even under a history
    cutoff now that pure cutoffs slice the canonical frame, DA-D-3)."""
    start = pd.Timestamp(s["window_start"]).strftime("%b %d, %Y")
    end = pd.Timestamp(s["window_end"]).strftime("%b %d, %Y")
    if holdings_filter_active:
        tail = (
            "<b>Holdings-filter slice.</b> TWR is synthesized from daily returns "
            "within each statement-date window using the filter's weight set — the "
            "same series the Risk tab consumes, so Pass-1 numbers across tabs stay "
            "aligned. "
            "IRR is hidden: external cashflows aren't tagged by asset-class, and "
            "per-bucket attribution of cross-bucket transfers has no honest "
            "definition under this slice.")
    elif broker_scope:
        tail = (
            f"<b>Broker subset ({' + '.join(broker_scope)}).</b> TWR is a "
            "NAV-weighted chain of the selected accounts' monthly returns (an "
            "approximation — internal transfers within the subset aren't "
            "paired). IRR is built from the subset's external transactions "
            "only (internal moves between selected accounts wash; transfers "
            "to accounts outside the subset are not currently reclassified).")
    else:
        tail = (
            "<b>Full portfolio.</b> Both TWR and IRR use the canonical compute_twr.py "
            "output with cross-account internal-transfer pairing applied "
            "portfolio-wide.")
    body = (
        f"<b>Window.</b> Comparison only includes calendar months where both "
        f"endpoints fall inside the {short} TR series. The current overlap "
        f"is <b>{int(s['n_months'])} months ({s['years']:.2f}y)</b>, {start} → "
        f"{end}.<br><br>"
        f"<b>Growth of $100,000.</b> Both lines start at $100,000 on the "
        f"window-start date and chain monthly returns ($100K × ∏(1+Rᵢ)). "
        f"{short} TR includes dividend reinvestment.<br><br>"
        f"<b>Drawdown.</b> Peak-to-trough percentage decline of the wealth index, "
        f"computed independently for each line.<br><br>"
        f"<b>TWR.</b> Time-weighted return chains monthly Modified-Dietz returns "
        f"and isolates investment performance from contribution timing.<br><br>"
        f"<b>IRR.</b> Money-weighted: same external cashflows, but the "
        f"counterfactual buys {short} on each flow's date and rolls "
        f"dividends — answers \"what if every contribution had gone into "
        f"{short} instead?\".<br><br>"
        f"{tail}"
    )
    if short == "60/40":
        # spec §6 disclosure: the two "60/40"s in this app are NOT the same
        # series — the Risk tabs' comparator (riskcontrib_service._benchmarks)
        # is 60/40 SPY/TLT price-return; this one is total-return SPY/AGG.
        # Gated so the SPY-path string above stays byte-identical (golden-pinned).
        body += (
            "<br><br><b>60/40 construction.</b> The 60/40 benchmark here is "
            "60% SPY + 40% AGG total return, rebalanced daily (constant-mix). "
            "This differs from the Risk tabs' \"60/40 SPY/TLT\" comparator, "
            "which is a price-return volatility/ES benchmark built for a "
            "different purpose — the two \"60/40\"s are not the same series."
        )
    return body


# --------------------------------------------------------------------------- #
# KPI row — mirror app.py 3256-3306. Headline value strings are formatted to
# byte-match the Streamlit st.metric values (fmt_pct decimals=2, fmt_money).
# --------------------------------------------------------------------------- #
def _headline(s: dict, irr_cmp: dict | None,
              holdings_filter_active: bool, short: str) -> list[dict]:
    twr_spread_pp = (s["port_twr_ann"] - s["bench_twr_ann"]) * 100
    port_ann = s["port_twr_ann"] * 100
    bench_ann = s["bench_twr_ann"] * 100
    wins = int(s["win_months"])
    losses = int(s["loss_months"])
    win_rate_pct = wins / max(1, wins + losses) * 100.0
    port_final = float(s["port_wealth_final"])
    bench_final = float(s["bench_wealth_final"])

    cards: list[dict] = []
    cards.append({
        "key": "twr", "label": "Portfolio TWR ann.",
        "value": fmt_pct(port_ann, 2),
        "color": hs._pnl_color(s["port_twr_ann"]),
        "delta": f"{twr_spread_pp:+.2f}% / yr vs {short}",
        "delta_dir": hs._dir(twr_spread_pp),
        "sub": (f"{short} TR ann.: {bench_ann:+.2f}%"
                + (f" · to {s['to_date']} · provisional" if s.get("to_date") else "")),
    })

    if irr_cmp is not None and not pd.isna(irr_cmp["irr_port"]):
        irr_spread_pp = (irr_cmp["irr_port"] - irr_cmp["irr_bench"]) * 100
        cards.append({
            "key": "irr", "label": "Portfolio IRR (windowed)",
            "value": fmt_pct(irr_cmp["irr_port"] * 100, 2),
            "color": hs._pnl_color(irr_cmp["irr_port"]),
            "delta": f"{irr_spread_pp:+.2f}% / yr vs {short}",
            "delta_dir": hs._dir(irr_spread_pp),
            "sub": f"{short} counterfactual IRR: {irr_cmp['irr_bench'] * 100:+.2f}%",
        })
    else:
        sub = ("Holdings filter active" if holdings_filter_active
               else "Insufficient cashflow data for this subset")
        cards.append({
            "key": "irr", "label": "Portfolio IRR (windowed)",
            "value": "—", "color": None, "delta": None, "delta_dir": "flat",
            "sub": sub,
        })

    cards.append({
        "key": "winrate", "label": "Monthly win-rate",
        "value": f"{wins}/{wins + losses}  ({win_rate_pct:.0f}%)",
        "color": None, "delta": None, "delta_dir": "flat",
        "sub": (f"Months portfolio's return exceeded {short}'s — includes "
                f"down months (e.g. portfolio -5% vs {short} -10% = win, "
                "since the portfolio lost less)."),
    })

    cards.append({
        "key": "wealth", "label": "$100K → portfolio",
        "value": fmt_money(port_final),
        "color": None,
        "delta": f"{fmt_money(port_final - bench_final)} vs {short}",
        "delta_dir": "up" if port_final >= bench_final else "down",
        # app.py colors this card with delta_color=("normal" if ahead else
        # "inverse"), which renders GREEN in BOTH cases (only the arrow flips).
        # Mirror that: arrow follows the sign (delta_dir), color stays gain.
        "delta_color": "up",
        "sub": f"$100K → {short}: {fmt_money(bench_final)}",
    })
    return cards


# --------------------------------------------------------------------------- #
# Growth-of-$100k overlay — mirror app.py 3308-3345. Both lines carry a leading
# base point at the window start ($100k), so the JS doesn't need the window date.
# --------------------------------------------------------------------------- #
def _growth(comp: pd.DataFrame, s: dict, short: str, *,
            stub: "InterimStub | None" = None,
            bench_stub: float | None = None) -> dict:
    base = float(s["base_amount"])
    ws = pd.Timestamp(s["window_start"])
    base_x = ws.strftime("%Y-%m-%d")
    port_pts = [{"x": base_x, "v": base}]
    bench_pts = [{"x": base_x, "v": base}]
    for _, r in comp.iterrows():
        x = pd.Timestamp(r["statement_date"]).strftime("%Y-%m-%d")
        port_pts.append({"x": x, "v": float(r["port_wealth"])})
        bench_pts.append({"x": x, "v": float(r["bench_wealth"])})
    if stub is not None and bench_stub is not None:
        # Provisional point on BOTH series (index-aligned — the overlay chart
        # positions points by index); the FE draws the last segment dashed.
        x1 = stub.end_date.strftime("%Y-%m-%d")
        port_pts.append({"x": x1, "v": port_pts[-1]["v"] * (1.0 + stub.return_pct),
                         "provisional": True})
        bench_pts.append({"x": x1, "v": bench_pts[-1]["v"] * (1.0 + bench_stub),
                          "provisional": True})
    return {
        "head": {"title": f"Growth of $100,000 — portfolio vs {short}",
                 "sub": (f"Both lines start at $100,000 on "
                         f"{ws.strftime('%b %d, %Y')} and chain each month's "
                         "return.")},
        "base": base,
        "series": [
            {"name": "Portfolio", "key": "port", "color": CHART_PORT,
             "dash": False, "points": port_pts},
            {"name": f"{short} (total return)", "key": "bench",
             "color": CHART_BENCH, "dash": True, "points": bench_pts},
        ],
    }


# --------------------------------------------------------------------------- #
# Drawdown overlay + trio — mirror app.py 3347-3402.
# --------------------------------------------------------------------------- #
def _drawdown(comp: pd.DataFrame, s: dict, short: str) -> dict:
    port_pts = [{"x": pd.Timestamp(r["statement_date"]).strftime("%Y-%m-%d"),
                 "dd": float(r["port_dd_pct"])} for _, r in comp.iterrows()]
    bench_pts = [{"x": pd.Timestamp(r["statement_date"]).strftime("%Y-%m-%d"),
                  "dd": float(r["bench_dd_pct"])} for _, r in comp.iterrows()]
    dd_spread = s["port_max_dd"] - s["bench_max_dd"]
    # Positive spread = portfolio's worst was less severe than the benchmark's.
    better = "shallower" if dd_spread > 0 else "deeper"
    return {
        "head": {"title": f"Drawdown — portfolio vs {short}",
                 "sub": ("Peak-to-trough decline of each wealth index. Negative = "
                         "below prior all-time-high. Lower troughs = deeper losses "
                         "to recover from.")},
        "trio": {
            "port": {"value": f"{s['port_max_dd']:+.1f}%",
                     "sub": pd.Timestamp(s["port_max_dd_date"]).strftime("%b %Y")},
            "bench": {"value": f"{s['bench_max_dd']:+.1f}%",
                      "sub": pd.Timestamp(s["bench_max_dd_date"]).strftime("%b %Y")},
            "spread": {"value": f"{abs(dd_spread):.1f}% {better}",
                       "sub": "Difference between the two worst-drawdown numbers"},
        },
        "series": [
            {"name": "Portfolio", "key": "port", "color": CHART_PORT,
             "dash": False, "points": port_pts},
            {"name": short, "key": "bench", "color": CHART_BENCH,
             "dash": True, "points": bench_pts},
        ],
    }


# --------------------------------------------------------------------------- #
# Periodic grouped + spread bars (M/Q/Y) — mirror app.py 3404-3481. All three
# granularities are precomputed so the JS segmented control switches without a
# refetch (same approach as performance_service._periodic).
# --------------------------------------------------------------------------- #
def _periodic(comp: pd.DataFrame) -> dict:
    # No win-rate here: app.py's benchmark periodic section shows none (win-rate
    # is a headline KPI only). Each granularity carries port/bench/spread bars.
    out: dict = {}
    for gran, freq in _GRAN_FREQ.items():
        if freq == "M":
            port_r = comp["port_return"].reset_index(drop=True)
            bench_r = comp["bench_return"].reset_index(drop=True)
            dts = comp["statement_date"].reset_index(drop=True)
        else:
            port_r, dts = aggregate_periodic_returns(
                comp["port_return"], comp["statement_date"], freq)
            bench_r, _ = aggregate_periodic_returns(
                comp["bench_return"], comp["statement_date"], freq)
            port_r = pd.Series(port_r).reset_index(drop=True)
            bench_r = pd.Series(bench_r).reset_index(drop=True)
            dts = pd.Series(dts).reset_index(drop=True)

        port_bars, bench_bars, spread_bars = [], [], []
        for d, pv, bv in zip(dts, port_r, bench_r):
            if pd.isna(pv) or pd.isna(bv):  # never ship NaN into allow_nan=False
                continue
            x = pd.Timestamp(d).strftime(_GRAN_XFMT[gran])
            port_bars.append({"x": x, "v": float(pv * 100.0)})
            bench_bars.append({"x": x, "v": float(bv * 100.0)})
            spread_bars.append({"x": x, "v": float((pv - bv) * 100.0)})
        out[gran] = {"port": port_bars, "bench": bench_bars,
                     "spread": spread_bars}
    return out


def _window_origin(comp: pd.DataFrame, key: str) -> pd.Timestamp:
    """Calendar start of a trailing window (the prev_stmt_date of its first
    row) — the same cutoffs period_returns.window_returns applies."""
    c = comp.sort_values("statement_date")
    end = pd.Timestamp(c.iloc[-1]["statement_date"])
    spec = next(sp for k, _l, sp, _a in WINDOWS if k == key)
    if spec == "ytd":
        win = c[pd.to_datetime(c["statement_date"]) > pd.Timestamp(year=end.year - 1, month=12, day=31)]
    elif spec is None:
        win = c
    else:
        win = c[pd.to_datetime(c["statement_date"]) > end - pd.DateOffset(months=spec)]
    return pd.Timestamp(win.iloc[0]["prev_stmt_date"])


def _returns_table(comp: pd.DataFrame, short: str, *,
                   stub: "InterimStub | None" = None,
                   bench_stub: float | None = None,
                   caption: str | None = None) -> dict:
    """Trailing-window portfolio-vs-benchmark returns (YTD/1Y/3Y/5Y/ITD).
    Raw decimals; the JS formats. Windows the overlap can't fill are
    available:false (never a short slice mislabeled as a long window).
    ``comp`` is the STATEMENT-only comparison; with a stub, each available
    row additionally carries the to-date figures — statement window chained
    with the provisional period, annualised rows re-annualised over the
    window's CALENDAR days to the price date (the headline's basis), YTD
    per ``ytd_to_date`` — the statement keys and vol columns are untouched."""
    rows = window_returns(comp)
    as_of = pd.Timestamp(comp.iloc[-1]["statement_date"]).strftime("%Y-%m-%d")
    out = {"as_of": as_of, "bench_label": short, "rows": rows}
    if stub is None or bench_stub is None:
        if caption:
            # No stub to chain, but the caller has something to say about
            # WHY (DA-C-10's "provisional segment unavailable" note) — the
            # FE renders returns_table.caption whenever present.
            out["caption"] = caption
        return out
    for r in rows:
        if not r["available"]:
            continue
        n = int(r["n_months"])
        if r["key"] == "ytd":
            p_td, b_td = ytd_to_date(r["port"], r["bench"], stub, bench_stub)
        elif r["annualized"]:
            cum_p = (1.0 + r["port"]) ** (n / 12.0) - 1.0
            cum_b = (1.0 + r["bench"]) ** (n / 12.0) - 1.0
            days = int((stub.end_date - _window_origin(comp, r["key"])).days)
            p_td = to_date_cagr(chain(cum_p, stub.return_pct), days)
            b_td = to_date_cagr(chain(cum_b, bench_stub), days)
        else:
            p_td, b_td = chain(r["port"], stub.return_pct), chain(r["bench"], bench_stub)
        r["port_to_date"], r["bench_to_date"] = p_td, b_td
        r["spread_to_date"] = p_td - b_td
        r["to_date"] = stub.end_date.strftime("%Y-%m-%d")
        r["provisional"] = True
    out["caption"] = (caption or stub_block(stub, hs._spct)["caption"]) + " · vol columns stay statement-based"
    return out


# --------------------------------------------------------------------------- #
# View assembly.
# --------------------------------------------------------------------------- #
def build_benchmark_view(frames: hs.Frames, *, account: str | list[str] = "all",
                         asset_class: str | list[str] = "all",
                         benchmark: str = "auto") -> dict:
    """Assemble the GET /api/benchmark contract. Pure given frames + selections.

    Mirrors app.py._render_benchmark_body: the Holdings-filter-scoped portfolio
    TWR view is compared against the resolved benchmark (SPY or the 60/40
    SPY/AGG blend) via the importable build_twr_comparison / build_irr_comparison
    engine functions. IRR is hidden under any Holdings filter (cashflows aren't
    class-tagged).
    """
    port = ps._prepare_portfolio_twr(frames.twr_portfolio)
    resolved = hs.resolve_benchmark(benchmark, frames.broker_scope,
                                    frames=frames)
    unavailable_fallback = False
    tr_lookup = hs._bench_tr_series(frames, resolved)
    if resolved == "60_40" and (tr_lookup is None or tr_lookup.empty):
        # AGG TR absent (e.g. not fetched yet) — degrade to SPY, keep state ok.
        resolved, unavailable_fallback = "spy", True
        tr_lookup = hs._bench_tr_series(frames, "spy")
    short = hs.BENCH_SHORT[resolved]

    snap_all = hs._current_snap(frames)
    acct_opts, _ = hs._account_options(snap_all)
    class_opts, _ = hs._class_options(snap_all)
    broker_opts, _ = hs._broker_options(snap_all)
    (bucket_filter, class_filter, selected_account_ids,
     account_active, class_active) = ps._resolve_filter(
         frames, snap_all, account, asset_class)
    holdings_filter_active = account_active or class_active

    # Unfiltered: byte-identical to the canonical twr_portfolio frame (carries
    # nav/prev_nav, so IRR works). Filtered: daily-synthesized (no nav columns,
    # IRR hidden) — same series Performance + Risk consume.
    port_view = (ps._filtered_twr_view(frames, port, bucket_filter, class_filter)
                 if holdings_filter_active else port)

    # meta is built first (and always carries accounts/classes) so the endpoint's
    # 422 filter-validation works even in a non-"ok" early-return state.
    meta = {
        "ticker": short,
        "benchmark": {"id": resolved, "label": hs.BENCHMARKS[resolved],
                      "short": short, "requested": benchmark,
                      "unavailable_fallback": unavailable_fallback},
        "subset_label": _subset_label(acct_opts, class_opts, account, asset_class,
                                      frames.broker_scope,
                                      canonical=hs.canonical_broker_label(frames)),
        "filter_caption": _filter_caption(holdings_filter_active, short,
                                          frames.broker_scope,
                                          canonical=hs.canonical_broker_label(frames)),
        "accounts": acct_opts,
        "classes": class_opts,
        "brokers": broker_opts,
        "filter": hs._filter_meta(account, asset_class),
        "holdings_filter_active": holdings_filter_active,
        "account_filter_active": account_active,
        "class_filter_active": class_active,
        "synthetic": "synth" in str(frames.data_dir).lower(),
        "state": "ok",
        "message_level": None,
        "window": None,
    }

    def _early(state: str, message: str, level: str) -> dict:
        meta["state"] = state
        meta["message_level"] = level
        return {"meta": meta, "message": message,
                "disclosures": {"methodology": ""},
                "headline": [], "growth": None, "drawdown": None,
                "periodic": None, "returns_table": None}

    # Two distinct empty states (mirror app.py 3161-3187 — do NOT merge them).
    if port_view is None or port_view.empty:
        if holdings_filter_active:
            return _early(
                "empty_filtered",
                "No return series for the current Account / Asset-class filter — "
                "the selected slice has no priceable holdings (e.g. cash-only or "
                "options-only, which are excluded from return synthesis). Clear "
                f"the filter to compare against {short}.", "info")
        return _early(
            "no_twr",
            "TWR data not found. Run `python3 parsers/compute_twr.py` to generate "
            "`data/twr_portfolio.csv`, then reload.", "error")
    if tr_lookup is None or tr_lookup.empty:
        return _early(
            "no_bench",
            f"{BENCH_TICKER} TR series not loaded or no overlapping months. Run "
            f"`python parsers/build_benchmark_total_return.py {BENCH_TICKER}` (and "
            "`compute_twr.py` first if needed).", "error")

    twr_cmp = build_twr_comparison(port_view, tr_lookup, base_amount=100_000.0)
    if twr_cmp is None:
        return _early(
            "no_overlap",
            f"No overlapping months between portfolio TWR and {short}.",
            "error")

    comp = twr_cmp["comp"]          # statement-only: drawdown/periodic/table vols
    s = twr_cmp["summary"]

    # Provisional stub period (spec 2026-08-22): whole-book / broker-scoped
    # views only, and only when the benchmark series covers the stub dates.
    # The to-date headline/window scalars are one-line chains of the
    # statement summary; counts, drawdowns and vols stay statement-based.
    stub = None if holdings_filter_active else hs.interim_stub(frames)
    bench_stub = bench_stub_return(tr_lookup, stub)
    stub_gap_note = None
    if stub is not None and bench_stub is None:
        # DA-C-10: the provisional segment used to vanish silently when the
        # benchmark series stops short of the stub window (a stale AGG leg
        # truncates the 60/40 blend's inner join) while the Performance tab
        # kept chaining its own stub — an unexplained cross-tab
        # contradiction. Name the reason instead of hiding the column.
        bench_end = (pd.Timestamp(tr_lookup.index.max()).strftime("%Y-%m-%d")
                     if tr_lookup is not None and len(tr_lookup) else "n/a")
        stub_gap_note = (f"Provisional segment unavailable: the {short} "
                         f"series ends {bench_end}, before the provisional "
                         f"period ends — run Refresh market data.")
    if bench_stub is None:
        stub = None
    s_head, stub_blk = s, None
    if stub is not None:
        days = int((stub.end_date - pd.Timestamp(s["window_start"])).days)
        p_cum = chain(s["port_twr_cum"], stub.return_pct)
        b_cum = chain(s["bench_twr_cum"], bench_stub)
        s_head = {**s,
                  "years": days / 365.0,
                  "port_twr_cum": p_cum, "bench_twr_cum": b_cum,
                  "port_twr_ann": to_date_cagr(p_cum, days),
                  "bench_twr_ann": to_date_cagr(b_cum, days),
                  "port_wealth_final": float(s["port_wealth_final"]) * (1.0 + stub.return_pct),
                  "bench_wealth_final": float(s["bench_wealth_final"]) * (1.0 + bench_stub),
                  "window_end": stub.end_date,
                  "to_date": stub.end_date.strftime("%Y-%m-%d")}
        stub_blk = stub_block(stub, hs._spct)

    # IRR vs SPY is undefined under a Holdings filter (transactions aren't
    # class-tagged) — mirror app.py 3202-3209.
    if holdings_filter_active:
        irr_cmp = None
    else:
        # frames.twr_account (loaded once + demo-overlaid in load_frames,
        # narrowed by apply_global_filters) — NOT a fresh data_dir read, which
        # would bypass the broker choke-point and leak real per-account $
        # under a test-only selection.
        twr_account = frames.twr_account
        allowed = (set(twr_account["account_id"].unique())
                   if not twr_account.empty else None)
        irr_cmp = build_irr_comparison(
            port_view, frames.transactions, frames.positions, tr_lookup,
            SYNTHETIC_ONBOARDING, allowed)

    meta["window"] = {
        "n_months": int(s["n_months"]),
        "years": float(s_head["years"]),
        "start": pd.Timestamp(s["window_start"]).strftime("%b %d, %Y"),
        "end": (pd.Timestamp(s_head["window_end"]).strftime("%b %d, %Y")
                + (" (provisional)" if stub is not None else "")),
    }

    return {
        "meta": meta,
        "message": "",
        "disclosures": {"methodology": _methodology(s_head, holdings_filter_active,
                                                    short, frames.broker_scope)},
        "headline": _headline(s_head, irr_cmp, holdings_filter_active, short),
        "growth": _growth(comp, s_head, short, stub=stub, bench_stub=bench_stub),
        "drawdown": _drawdown(comp, s, short),
        "periodic": _periodic(comp),
        "returns_table": _returns_table(comp, short, stub=stub, bench_stub=bench_stub,
                                        caption=(stub_blk["caption"] if stub_blk
                                                 else stub_gap_note)),
        # Additive: the key exists only when a stub does (golden stays byte-identical).
        **({"stub": _stub_payload(stub_blk, s_head, bench_stub)} if stub is not None else {}),
    }


def _stub_payload(blk: dict, s_head: dict, bench_stub: float) -> dict:
    """The shared stub block plus the to-date comparison scalars the tab
    shows (decimals; the JS formats)."""
    d = dict(blk)
    d.update({"bench_stub_return": float(bench_stub),
              "years_to_date": float(s_head["years"]),
              "port_twr_cum_to_date": float(s_head["port_twr_cum"]),
              "bench_twr_cum_to_date": float(s_head["bench_twr_cum"]),
              "port_twr_ann_to_date": float(s_head["port_twr_ann"]),
              "bench_twr_ann_to_date": float(s_head["bench_twr_ann"])})
    return d
