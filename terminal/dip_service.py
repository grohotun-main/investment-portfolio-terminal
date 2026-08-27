# terminal/dip_service.py
"""Pure data seam for the MERIDIAN Terminal "Buy the Dip" tab (static half).

Re-expresses app.py._render_dip_body / _render_dip_card (the SPY/SCHD/GLD auto
cards + turbulence banner + legend). Every dip number lives in the importable,
Streamlit-free engine (dip_analytics / turbulence / tail_risk / dip_adhoc), so
the view matches Streamlit 1:1 by construction. Whole-book, no params, fully
date-stable (everything keys off the data's latest date, not today), so the
golden needs no asof thread. The ad-hoc live-ticker lookup is a separate
(PR B) surface and is intentionally NOT built here.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from terminal import holdings_service as hs

import theme
import dip_adhoc
import dip_analytics
import dip_backtest
import tail_risk
import turbulence
from fetch_dip_history import fetch_yahoo, fetch_dividends_yahoo  # live ad-hoc fetch

HORIZONS = (21, 63, 126, 252)
HLABELS_LONG = {21: "1 month", 63: "3 months", 126: "6 months", 252: "12 months"}
MACRO = ["SPY", "GLD", "TLT", "UUP", "VIXY"]
_TRIO = ("SPY", "SCHD", "GLD")

# rgb triples for the tint backgrounds — mirror app.py 9232-9233 exactly.
_GRN = f"{int(theme.GAIN[1:3], 16)},{int(theme.GAIN[3:5], 16)},{int(theme.GAIN[5:7], 16)}"
_RED = f"{int(theme.LOSS[1:3], 16)},{int(theme.LOSS[3:5], 16)},{int(theme.LOSS[5:7], 16)}"


def _jnum(v) -> float | None:
    """JSON-safe float for RAW numeric fields: NaN/inf -> None so allow_nan=False
    never sees an invalid token. Display strings keep literal formatting for
    byte-parity with Streamlit."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


# --- display formatters (mirror app.py 9236-9246) -------------------------- #
def _pct(x):  return f"{x * 100:.1f}%"
def _pct0(x): return f"{x * 100:.0f}%"
def _pctn(x): return _pct(x) if np.isfinite(x) else "—"
def _fmt_mo(d): return f"{d / 21.0:.0f} mo" if np.isfinite(d) else "—"


def _dual(reg, full, fmt):
    """Today's-regime value; append the all-conditions value in parens only when
    it differs (mirror app.py 9242-9246)."""
    a, b = fmt(reg), fmt(full)
    return a if a == b else f"{a} ({b})"


def _md_strong(s: str) -> str:
    """Convert `**x**` -> `<strong>x</strong>` so the engine's markdown captions
    (e.g. dip_analytics.time_underwater_caption) render as HTML in the terminal
    instead of leaking literal asterisks (Streamlit's st.caption parses the
    markdown; the front-end renders this via innerHTML). The engine text is
    trusted (no user input), so emitting HTML here is safe — same posture as the
    server-authored _LEGEND_HTML / factor_service._METHODOLOGY."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


# --- tint backgrounds (mirror app.py 9249-9268 byte-for-byte) -------------- #
def _bg_return(v):
    if not np.isfinite(v):
        return ""
    if v >= 0:
        return f"background-color: rgba({_GRN},{min(0.30, 0.06 + 1.6 * v):.3f})"
    return f"background-color: rgba({_RED},{min(0.30, 0.06 + 1.6 * abs(v)):.3f})"


def _bg_down(v):
    if not np.isfinite(v):
        return ""
    return f"background-color: rgba({_RED},{min(0.30, 0.04 + 0.55 * v):.3f})"


def _bg_loss(v):
    if not np.isfinite(v):
        return ""
    if v >= 0:
        return f"background-color: rgba({_GRN},{min(0.30, 0.06 + 1.6 * v):.3f})"
    return f"background-color: rgba({_RED},{min(0.32, 0.05 + 0.55 * abs(v)):.3f})"


def _load_dip_csvs(data_dir):
    """Read dip_history.csv / dip_dividends.csv directly (app.py's loaders are
    st.cache_data-bound, not importable). Same parse_dates as app.py 542-556."""
    data_dir = Path(data_dir)
    hp = data_dir / "dip_history.csv"
    dp = data_dir / "dip_dividends.csv"
    hist = (pd.read_csv(hp, parse_dates=["date"]) if hp.exists()
            else pd.DataFrame(columns=["symbol", "date", "close", "adj_close"]))
    divs = (pd.read_csv(dp, parse_dates=["ex_date"]) if dp.exists()
            else pd.DataFrame(columns=["symbol", "ex_date", "amount"]))
    return hist, divs


CAPTION = (
    "Is the current drawdown unusual, how much sharper could it get if you buy "
    "now, and what dividend yield do you lock in? Drawdown/percentile on price; "
    "forward returns on total return; EVT tail for the rare further-fall. "
    "Small samples carry wide error bars — episode counts are shown so you can judge."
)

# 'How to read this tab' content — copied VERBATIM from the st.markdown(...) body
# at app.py 9558-9590 so the legend text matches Streamlit byte-for-byte.
LEGEND_BODY = (
    "**Why two numbers everywhere?** Every dip stat is conditioned on the asset's "
    "own *volatility regime today* (calm 🟢 vs stressed 🟠 tape). The **main number** "
    "is the history that matches today's regime; the number in **(parentheses)** is "
    "the all-regime reference — shown only when it differs (so an unchanged value "
    "isn't repeated). When today's regime has too few comparable dips, the row "
    "falls back to all-regime history and says so.\n\n"
    "**The forward-return table** — every cell is a total return measured **from your "
    "buy price, at that horizon** (not a drawdown to the low along the way):\n"
    "- **If You Hold** — how long you hold after buying a dip at least this deep.\n"
    "- **Typical Median Forward Return** — the median outcome at that horizon.\n"
    "- **Chance You're Still Down** — how often you'd be below your buy price at the "
    "horizon.\n"
    "- **If Still Down, Median Loss** — among only the times it ended lower, the "
    "median loss.\n"
    "- **10th-percentile loss** — the worst 1-in-10 outcome; 90% of the time the "
    "loss is smaller than this.\n"
    "- **Worst Case Ever** — the worst that horizon ever *ended* (not its lowest "
    "point along the way — the further-fall line below each table covers the path).\n"
    "- **History (Days)** — how many comparable days are behind the row.\n\n"
    "**Verdict callout (top of each card)** — a plain read of the reward/risk "
    "edge: 🟢 strong buy, 🟡 buy, 🔴 don't buy, ⚪ no signal/not enough data. "
    "It uses the "
    "**Omega ratio** (upside dollars per dollar of downside at 12 months; above "
    "1 is favorable), compares it to the asset's normal Omega (the edge), and "
    "resamples history for a confidence interval — so a verdict is only "
    "“strong buy” when the edge holds up statistically, and says "
    "“not enough data” when there are too few comparable dips.\n"
    "- **Time underwater** — after buying a dip this deep, how long it "
    "historically took to get back to your buy price (break-even).\n\n"
    "**This tab is only informative when the dip is deep.** A shallow dip (a low "
    "“Deeper than …%”) makes the comparison set ≈ all days, so the table ≈ the "
    "ticker's ordinary forward returns. The buy-the-dip edge appears when "
    "**“Deeper than” is high.**"
)

# Server-authored HTML rendering of LEGEND_BODY (same wording/emoji, markdown
# hand-converted to HTML) — served as legend.body and rendered via innerHTML by
# the front-end. Streamlit's st.markdown parses LEGEND_BODY's `**bold**` / bullets
# for free; the vanilla terminal does NOT, so passing the raw markdown to
# textContent leaked literal `**` and `- ` markers. This mirrors the established
# server-HTML-prose pattern (factor_service._METHODOLOGY rendered via innerHTML).
# Trusted content (no user input). Keep LEGEND_BODY above as the source of truth
# for the wording.
_LEGEND_HTML = (
    "<p><strong>Why two numbers everywhere?</strong> Every dip stat is conditioned "
    "on the asset's own <em>volatility regime today</em> (calm 🟢 vs stressed 🟠 "
    "tape). The <strong>main number</strong> is the history that matches today's "
    "regime; the number in <strong>(parentheses)</strong> is the all-regime "
    "reference — shown only when it differs (so an unchanged value isn't repeated). "
    "When today's regime has too few comparable dips, the row falls back to "
    "all-regime history and says so.</p>"
    "<p><strong>The forward-return table</strong> — every cell is a total return "
    "measured <strong>from your buy price, at that horizon</strong> (not a drawdown "
    "to the low along the way):</p>"
    "<ul>"
    "<li><strong>If You Hold</strong> — how long you hold after buying a dip at "
    "least this deep.</li>"
    "<li><strong>Typical Median Forward Return</strong> — the median outcome at "
    "that horizon.</li>"
    "<li><strong>Chance You're Still Down</strong> — how often you'd be below your "
    "buy price at the horizon.</li>"
    "<li><strong>If Still Down, Median Loss</strong> — among only the times it "
    "ended lower, the median loss.</li>"
    "<li><strong>10th-percentile loss</strong> — the worst 1-in-10 outcome; 90% of "
    "the time the loss is smaller than this.</li>"
    "<li><strong>Worst Case Ever</strong> — the worst that horizon ever "
    "<em>ended</em> (not its lowest point along the way — the further-fall line "
    "below each table covers the path).</li>"
    "<li><strong>History (Days)</strong> — how many comparable days are behind the "
    "row.</li>"
    "</ul>"
    "<p><strong>Verdict callout (top of each card)</strong> — a plain read of the "
    "reward/risk edge: 🟢 strong buy, 🟡 buy, 🔴 don't buy, ⚪ no signal/not "
    "enough data. It "
    "uses the <strong>Omega ratio</strong> (upside dollars per dollar of downside "
    "at 12 months; above 1 is favorable), compares it to the asset's normal Omega "
    "(the edge), and resamples history for a confidence interval — so a verdict is "
    "only “strong buy” when the edge holds up statistically, and says “not "
    "enough data” when there are too few comparable dips.</p>"
    "<ul>"
    "<li><strong>Time underwater</strong> — after buying a dip this deep, how long "
    "it historically took to get back to your buy price (break-even).</li>"
    "</ul>"
    "<p><strong>This tab is only informative when the dip is deep.</strong> A "
    "shallow dip (a low “Deeper than …%”) makes the comparison set ≈ all days, so "
    "the table ≈ the ticker's ordinary forward returns. The buy-the-dip edge "
    "appears when <strong>“Deeper than” is high.</strong></p>"
)


def build_dip_view(frames) -> dict:
    """Note: the cards/turbulence are whole-market and never read
    ``frames.positions``, but ``meta`` still carries ``brokers`` + a
    ``filter.broker`` echo (default ``"all"``) so the shared chrome Broker pill
    is populated on a direct ?tab=dip landing, matching the other whole-book
    tabs (spec: filter-parity S2a)."""
    broker_opts, _ = hs._broker_options(hs._current_snap(frames))
    filter_meta = {"account": "all", "asset_class": "all", "broker": "all"}

    hist, divs = _load_dip_csvs(frames.data_dir)
    if hist.empty:
        return {"meta": {"vintage": None, "symbols": [], "brokers": broker_opts,
                         "filter": filter_meta},
                "caption": CAPTION,
                "turbulence": None, "legend": _legend(),
                "empty": {"message": ("No dip history yet. Run "
                                      "`py parsers/fetch_dip_history.py --write` "
                                      "to populate `data/dip_history.csv`.")},
                "cards": []}

    # BUILD order: trio-first-alpha — the SAME order app.py renders in. Card
    # numbers are build-order-independent (bootstrap is per-call seeded;
    # verified leaf-for-leaf 2026-07-19), but the parity test pairs
    # Streamlit's positional metric trios to cards assuming this order, so
    # keep it and apply any display ordering AFTER the builds.
    watch = sorted(hist["symbol"].unique(),
                   key=lambda s: (s not in _TRIO, s))  # trio first, then alpha
    vintage = str(pd.to_datetime(hist["date"]).max().date())

    cards = []
    for sym in watch:
        price, tr, dser = dip_adhoc.slice_symbol(hist, divs, sym)
        if len(price) < dip_adhoc.MIN_HISTORY_DAYS:
            continue
        cards.append(_build_card(sym, price, tr, dser))
    # DISPLAY order (TK 2026-07-19): trio in ITS OWN order — SPY, SCHD, GLD —
    # then extras alphabetical. Applied to the finished cards so the build
    # order above (and everything keyed to it) stays put.
    cards.sort(key=lambda c: (c["symbol"] not in _TRIO,
                              _TRIO.index(c["symbol"]) if c["symbol"] in _TRIO
                              else 0, c["symbol"]))

    return {"meta": {"vintage": vintage, "symbols": [c["symbol"] for c in cards],
                     "brokers": broker_opts, "filter": filter_meta},
            "caption": CAPTION, "turbulence": _turbulence(frames),
            "legend": _legend(), "empty": None, "cards": cards}


def _legend() -> dict:
    """Static 'How to read this tab' content. `body` is server-authored HTML
    (_LEGEND_HTML, the HTML rendering of LEGEND_BODY) so the front-end can render
    it via innerHTML — textContent would leak the raw `**`/`- ` markdown markers."""
    return {"title": "How to read this tab — the two-number convention & columns",
            "body": _LEGEND_HTML}


_TB_LABEL = {"calm": "🟢 calm", "elevated": "🟡 elevated",
             "abnormal": "🔴 abnormal regime"}


def turbulence_snapshot(daily_prices) -> "dict | None":
    """Macro-turbulence regime from the daily-price matrix: MACRO columns →
    daily returns → turbulence_now. Returns {regime, percentile, n}, or None when
    too few macro columns / < 31 return rows. Shared by app.py's banner and
    _turbulence (the banner formats it; _turbulence adds the label + _jnum)."""
    dp = daily_prices
    cols = [c for c in MACRO if not dp.empty and c in dp.columns]
    rets = (dp[cols].sort_index().pct_change(fill_method=None).dropna()
            if cols else pd.DataFrame())
    if not (rets.shape[1] >= 2 and len(rets) > 30):
        return None
    tnow = turbulence.turbulence_now(rets)
    return {"regime": tnow["regime"], "percentile": tnow["percentile"],
            "n": int(len(rets))}


def _turbulence(frames) -> dict | None:
    snap = turbulence_snapshot(frames.daily_prices)
    if snap is None:
        return None
    return {"regime": snap["regime"],
            "label": _TB_LABEL.get(snap["regime"], snap["regime"]),
            "percentile": _jnum(snap["percentile"]), "n": snap["n"]}


def _verdict_block(sym, verdict, state, history_years) -> dict:
    """Band -> callout level + a terminal-native sentence built from the SAME
    engine numbers the Streamlit card uses (band/omega/baseline/CI/n). Wording is
    terminal-native by design (the app.py f-strings are not importable); the
    parity gate pins the numbers, not the prose (TK: '1:1 = numbers, not pixels').
    ``history_years`` drives the referee-grounded thin-history disclosure
    (dip_analytics.history_depth_caveat) appended to edge-claiming bands."""
    band = verdict["band"]
    om = verdict["omega"]
    base = verdict["baseline_omega"]
    lo, hi = verdict["omega_ci"]["lo"], verdict["omega_ci"]["hi"]
    ci = (f" (90% CI {lo:.1f}–{hi:.1f})"
          if np.isfinite(lo) and np.isfinite(hi) else "")
    vs = f" vs {base:.1f} normally" if np.isfinite(base) else ""
    omt = "∞" if np.isinf(om) else (f"{om:.1f}" if np.isfinite(om) else "n/a")
    # tiny/no in-sample losses -> the "$X per $1" framing reads absurd (app.py:9339-9340)
    if band == "strong" and (np.isinf(om) or om >= 10.0):
        mag = ("had no losing 12-month outcomes in-sample" if np.isinf(om)
               else f"saw gains far outweigh the rare losses (Omega {om:.0f})")
        level, text = "success", (
            f"🟢 Strong buy. After dips this deep, {sym} {mag}{vs}, "
            f"and this depth ranks in {sym}'s top third for reward-to-risk. "
            f"Caveat: in-sample history, not a forecast.")
    elif band == "strong":
        level, text = "success", (
            f"🟢 Strong buy. Over 12 months, dips this deep gave "
            f"${om:.1f} of upside per $1 of downside (Omega {om:.1f}{vs}), and the "
            f"edge holds under resampling{ci}. This depth ranks in {sym}'s top "
            f"third for reward-to-risk. Caveat: in-sample history, not a forecast.")
    elif band == "neutral":
        level, text = "info", (
            f"🟡 Buy. Reward-to-risk is favorable (Omega "
            f"{omt}{vs}){ci}, but not decisively — read the table and the "
            f"further-fall risk below before adding.")
    elif band == "weak":
        level, text = "warning", (
            f"🔴 Don't buy. Downside outweighs upside at this depth "
            f"(Omega {omt}{vs}){ci}. History hasn't rewarded buying dips this "
            f"deep in {sym}.")
    elif band == "shallow":
        level, text = "info", (
            f"⚪ No signal — dip too shallow. At deeper-than-"
            f"{state['pct_history_shallower']:.0f}% the comparison set is ≈ all "
            f"days, so these are {sym}'s ordinary forward returns, not a dip "
            f"signal. The edge appears on deeper pullbacks.")
    else:  # inconclusive
        level, text = "info", (
            f"⚪ Not enough data — only {verdict['n']} 12-month outcomes this "
            f"deep{f' (Omega {omt})' if omt != 'n/a' else ''} — too few to "
            f"call. Lean on the table, not a verdict.")
    cav = dip_analytics.history_depth_caveat(sym, band, history_years)
    if cav:
        text = f"{text} {cav}"
    return {"band": band, "level": level, "text": text,
            "omega": _jnum(om), "baseline_omega": _jnum(base),
            "ci_lo": _jnum(lo), "ci_hi": _jnum(hi), "n": int(verdict["n"])}


_REFEREE_COLS = ["Verdict Band", "Days", "Episodes", "Median 12m", "Hit 12m",
                 "Omega 12m"]
_BAND_LABEL = {"strong": "🟢 Strong buy", "neutral": "🟡 Buy",
               "weak": "🔴 Don't buy", "inconclusive": "⚪ Not enough data",
               "shallow": "⚪ No signal (too shallow)"}


@functools.lru_cache(maxsize=1)
def _registered_artifact() -> "dict | None":
    """Committed registered-referee record; file changes only via a PR +
    restart (config lifecycle), so one read per process. Read-only — never
    mutate the cached dict."""
    return dip_backtest.load_registered_artifact()


def _referee_row(label: str, rec: dict) -> dict:
    def num(v, fmt):
        return "—" if v is None else fmt(v)
    omega = ("∞" if rec.get("omega_252_inf")
             else "—" if rec.get("omega_252") is None
             else f"{rec['omega_252']:.1f}")
    return {"Verdict Band": label,
            "Days": f"{rec['n_days']:,}",
            "Episodes": f"{rec['n_episodes']:,}",
            "Median 12m": num(rec.get("med_252"), _pct),
            "Hit 12m": num(rec.get("hit_252"), _pct),
            "Omega 12m": omega}


def _referee_block(sym: str) -> "dict | None":
    """The registered walk-forward referee's per-band realized table —
    attached ONLY to the artifact's own ticker (SPY). Static registered
    evidence (spec 2026-07-16); everything renders from the committed
    artifact, nothing is recomputed per request."""
    art = _registered_artifact()
    if art is None or art.get("ticker") != sym:
        return None
    rows = [_referee_row(_BAND_LABEL[b], art["referee"][b])
            for b in dip_backtest.BAND_ORDER]
    rows.append(_referee_row("— All days", art["referee"]["all"]))
    # "(TK rule)" dropped 2026-07-19 — the label now says what the row IS
    # (artifact key stays tk_rule; display-only rename).
    rows.append(_referee_row("★ Deeper than 85% + edge-claimed",
                             art["tk_rule"]))
    pm, ev, cfg = art["primary"], art["evals"], art["config"]
    stat = f"{pm['stat']:+.1f}" if pm.get("stat") is not None else "n/a"
    ci = (f"{pm['ci_lo']:+.1f} to {pm['ci_hi']:+.1f}"
          if pm.get("ci_lo") is not None and pm.get("ci_hi") is not None
          else "n/a")
    caption = (
        f"Out-of-sample track record of this verdict (walk-forward referee, "
        f"registered {art['registered']}): each row replays the verdict on "
        f"data available that day only — {art['ticker']}, {ev['n']:,} "
        f"evaluations {ev['first']} → {ev['last']}, stride {cfg['stride']}, "
        f"{cfg['burn_in_years']}y burn-in. Registered primary: "
        f"{pm['outcome']} — edge-claimed minus all-days 12m Omega {stat} "
        f"(90% CI {ci}), {pm['n_edge_episodes']} edge episodes. "
        f"{art['ticker']} only; other tickers have no registered run.")
    return {"columns": _REFEREE_COLS, "rows": rows, "caption": caption}


@dataclass
class DipCardData:
    """Raw per-card engine outputs — the shared card-builder orchestration. Both
    _build_card (terminal JSON) and app.py._render_dip_card (Streamlit) render
    from this; the pure verdict-inputs (ent/reg_ent/losses/…) stay internal."""
    state: dict
    n_ep: int
    recov: dict
    ymet: dict
    today_regime: str
    fwd_full: dict
    fwd_reg: dict
    ff_full: dict
    ff_reg: dict
    use_reg_ff: bool
    ff_head: dict
    fit: dict
    evt95: float
    evt99: float
    verdict: dict
    rec_reg: dict
    rec_full: dict
    use_rec: bool
    rec_head: dict
    history_years: float


def dip_card_data(sym, price, tr, dser) -> DipCardData:
    """One dip card's full engine orchestration (drawdown → regime entries →
    forward stats ×2 → further-fall ×2 → GPD/EVT tail → dip_buy_verdict →
    recovery). Pure; caller guarantees len(price) >= dip_adhoc.MIN_HISTORY_DAYS.
    The verdict half is dip_analytics.dip_verdict_block (shared with the
    walk-forward referee); this function adds the card-only extras (episode
    count, recovery rate, yield, EVT tail, recovery times)."""
    da = dip_analytics
    blk = da.dip_verdict_block(price, tr, horizons=HORIZONS)
    state = blk["state"]
    n_ep = da.episodes_reaching(price, state["current_dd"])
    recov = da.recovery_rate(price)
    ymet = (da.yield_percentile(price, dser) if not dser.empty
            else {"current_yield": float("nan"), "percentile": float("nan")})
    in_reg = blk["in_reg"]
    use_reg_ff = blk["use_reg_ff"]

    losses = da._further_fall_losses(
        price, state["current_dd"], in_regime=in_reg if use_reg_ff else None)
    fit = (tail_risk.fit_gpd_tail(losses) if len(losses)
           else {"confident": False, "n_exceedances": 0})
    evt95 = tail_risk.tail_loss_quantile(fit, 0.05) if fit.get("confident") else float("nan")
    evt99 = tail_risk.tail_loss_quantile(fit, 0.01) if fit.get("confident") else float("nan")

    rec_reg = da.conditional_recovery_time(price, state["current_dd"], in_regime=in_reg)
    rec_full = da.conditional_recovery_time(price, state["current_dd"])
    use_rec = rec_reg["n_complete"] >= da.REGIME_MIN_N
    rec_head = rec_reg if use_rec else rec_full

    return DipCardData(
        state=state, n_ep=n_ep, recov=recov, ymet=ymet,
        today_regime=blk["today_regime"],
        fwd_full=blk["fwd_full"], fwd_reg=blk["fwd_reg"],
        ff_full=blk["ff_full"], ff_reg=blk["ff_reg"],
        use_reg_ff=use_reg_ff, ff_head=blk["ff_head"],
        fit=fit, evt95=evt95, evt99=evt99, verdict=blk["verdict"],
        rec_reg=rec_reg, rec_full=rec_full, use_rec=use_rec,
        rec_head=rec_head,
        history_years=blk["history_years"])


def _build_card(sym, price, tr, dser) -> dict:
    d = dip_card_data(sym, price, tr, dser)
    asof = price.index[-1].date()
    # Total-return drawdown (adj_close) beside the price drawdown (TK
    # 2026-07-19): for dividend payers the price peak goes stale while TR sits
    # at highs (SCHD ~3.4%/yr yield), so the price-basis number alone can read
    # "deep dip" at a TR all-time high. Display-only — every signal still
    # keys off the price series.
    tr_dd = (float(tr.iloc[-1] / tr.max() - 1.0)
             if len(tr) and np.isfinite(tr.max()) and tr.max() > 0
             else float("nan"))
    card = {
        "symbol": sym,
        "today_regime": d.today_regime,
        "regime_chip": {"calm": "🟢 calm tape", "stressed": "🟠 stressed tape"}.get(
            d.today_regime, f"{d.today_regime} tape"),
        "verdict": _verdict_block(sym, d.verdict, d.state, d.history_years),
        "kpis": _kpis(d.state, d.n_ep, d.ymet, asof, tr_dd),
        "bridge_text": _bridge(sym, d.state),
        "forward_table": _forward_table(sym, d.fwd_full, d.fwd_reg, d.today_regime),
        "further_fall": {"text": _further_fall_text(
            sym, d.today_regime, d.use_reg_ff, d.ff_reg, d.ff_full, d.fit,
            d.evt95, d.evt99)},
        "time_underwater": _md_strong(_time_underwater(
            sym, d.verdict["band"], d.state, d.use_rec, d.rec_reg, d.rec_full,
            d.rec_head)),
        "track_record": _track_record(d.recov),
        "underwater": [{"x": str(ts.date()), "v": _jnum(v)}
                       for ts, v in (dip_analytics.underwater(price) * 100.0).items()],
    }
    ref = _referee_block(sym)
    if ref:
        card["referee"] = ref
    return card


def _kpis(state, n_ep, ymet, asof, tr_dd=float("nan")) -> dict:
    yp = ymet["percentile"]
    yhelp = (f"{yp:.0f}th pct of its own yield range — high = cheap. Assumes the "
             "distribution holds (crises cut dividends)."
             if np.isfinite(yp) else "No dividend history to compute yield percentile.")
    # TR-basis sub-line follows the house _dual taste: shown only when it
    # differs from the price number at display precision (else the price
    # figure IS the TR figure and repeating it would just be noise).
    dd_str = f"{state['current_dd'] * 100:.1f}%"
    tr_str = f"{tr_dd * 100:.1f}%" if np.isfinite(tr_dd) else None
    dd_sub = (f"{tr_str} with dividends reinvested"
              if tr_str is not None and tr_str != dd_str else None)
    return {
        "current_dd": {
            "value": dd_str,
            "sub": dd_sub,
            "help": (f"as of {asof} close; "
                     f"{state['frac_below_52w_high'] * 100:.1f}% below 52-wk high"
                     + (f"; total-return basis: {tr_str}" if tr_str else ""))},
        "deeper_than": {
            "value": f"{state['pct_history_shallower']:.0f}% of history",
            "help": (f"sharper only {state['pct_history_deeper']:.0f}% of the time · "
                     f"{n_ep} distinct episodes this deep over "
                     f"{state['n_days']} days")},
        "locked_yield": {
            "value": (f"{ymet['current_yield'] * 100:.2f}%"
                      if np.isfinite(ymet["current_yield"]) else "n/a"),
            "help": yhelp},
    }


def _bridge(sym, state):
    shallower = state["pct_history_shallower"]
    word = ("unusually deep" if shallower >= 80 else
            "on the deeper side" if shallower >= 55 else
            "fairly typical" if shallower >= 33 else "on the milder side")
    return (f"In plain terms: {sym} is {_pct(-state['current_dd'])} below its prior "
            f"peak — a dip that's {word} (deeper than {shallower:.0f}% of its own "
            "history). Here's what buying dips at least this deep has led to "
            "historically:")


_FWD_COLS = ["If You Hold", "Typical Median Forward Return", "Chance You're Still Down",
             "If Still Down, Median Loss", "10th-percentile loss", "Worst Case Ever",
             "History (Days)"]


def _forward_table(sym, fwd_full, fwd_reg, today_regime) -> dict:
    rows, tints, numeric = [], [], []
    for h in HORIZONS:
        fn = fwd_full[h]["n"]
        if fn == 0:
            continue
        use = fwd_reg[h]["n"] >= dip_analytics.REGIME_MIN_N
        src = fwd_reg[h] if use else fwd_full[h]
        full = fwd_full[h]
        pdown, pdown_full = 1.0 - src["hit_rate"], 1.0 - full["hit_rate"]

        def cell(key, fmt=_pct):
            return _dual(src[key], full[key], fmt) if use else fmt(src[key])

        rows.append({
            "If You Hold": HLABELS_LONG[h],
            "Typical Median Forward Return": cell("median"),
            "Chance You're Still Down": (_dual(pdown, pdown_full, _pct0) if use
                                         else _pct0(pdown)),
            "If Still Down, Median Loss": cell("cond_loss", _pctn),
            "10th-percentile loss": cell("p10"),
            "Worst Case Ever": cell("worst"),
            "History (Days)": f"{src['n']:,} ({fn:,})" if use else f"{fn:,}",
        })
        tints.append({
            "Typical Median Forward Return": _bg_return(src["median"]),
            "Chance You're Still Down": _bg_down(pdown),
            "If Still Down, Median Loss": _bg_loss(src["cond_loss"]),
            "10th-percentile loss": _bg_loss(src["p10"]),
            "Worst Case Ever": _bg_loss(src["worst"]),
        })
        numeric.append({
            "Typical Median Forward Return": _jnum(src["median"]),
            "Chance You're Still Down": _jnum(pdown),
            "If Still Down, Median Loss": _jnum(src["cond_loss"]),
            "10th-percentile loss": _jnum(src["p10"]),
            "Worst Case Ever": _jnum(src["worst"]),
        })
    caption = (
        "Every number is a return from your buy price (a dip at least this deep), "
        "measured AT that horizon — not a drawdown to the low along the way. "
        "'Chance You're Still Down' + 'If Still Down, Median Loss' are a pair; "
        "'10th-percentile loss' is the worst 1-in-10 outcome (you did better 90% of "
        "the time); 'Worst Case Ever' is the worst that horizon ever ended — the "
        "path can dip deeper, see the further-fall line below. Main number = "
        f"today's {today_regime}-tape history; parentheses = all conditions (shown "
        "only when they differ). Green = gains, red = losses; deeper shade = bigger "
        f"move. Rows fall back to all conditions under {dip_analytics.REGIME_MIN_N} "
        f"{today_regime}-tape dips.")
    return {"columns": _FWD_COLS, "rows": rows, "tints": tints,
            "numeric": numeric, "caption": caption}


def _further_fall_text(sym, today_regime, use_reg_ff, ff_reg, ff_full, fit, evt95, evt99):
    evt_clause = (
        f"A tail model puts the rare extreme near {_pct(-evt95)} in a 1-in-20 dip "
        f"and {_pct(-evt99)} in a 1-in-100 dip."
        if fit.get("confident")
        else f"(Too few extreme cases ({fit.get('n_exceedances', 0)}) for a reliable "
             "tail estimate — leaning on the history above.)")
    ff_head = ff_reg if use_reg_ff else ff_full
    if ff_head["n_complete"] == 0:
        return (f"If you buy now: no dip this deep has fully recovered in {sym}'s "
                f"history yet — all {ff_head['n_censored']} are still underwater. "
                "Lean on the worst-case column and the tail estimate.")
    if use_reg_ff:
        ref = ff_full["quantiles"]
        return (f"If you buy now, how much further could it fall first? In past "
                f"{today_regime}-tape dips this deep, {sym} typically slipped "
                f"another {_dual(ff_reg['quantiles'][0.5], ref[0.5], _pct)} before "
                f"bottoming. In a rough 1-in-7 dip it fell "
                f"{_dual(ff_reg['quantiles'][0.85], ref[0.85], _pct)} more; in a "
                f"1-in-20 dip {_dual(ff_reg['quantiles'][0.95], ref[0.95], _pct)} "
                f"more (the 2008/2020-class crashes). " + evt_clause +
                f" Based on {ff_reg['n_complete']} comparable dips"
                + (f", {ff_reg['n_censored']} of which never recovered."
                   if ff_reg['n_censored'] else "."))
    q = ff_full["quantiles"]
    return (f"If you buy now, how much further could it fall first? Across all "
            f"market conditions (too few {today_regime}-tape dips this deep to judge "
            f"that regime alone), {sym} typically slipped another {_pct(q[0.5])} "
            f"before bottoming. In a rough 1-in-7 dip it fell {_pct(q[0.85])} more; "
            f"in a 1-in-20 dip {_pct(q[0.95])} more. " + evt_clause +
            f" Based on {ff_full['n_complete']} comparable dips"
            + (f", {ff_full['n_censored']} of which never recovered."
               if ff_full['n_censored'] else "."))


def _time_underwater(sym, band, state, use_rec, rec_reg, rec_full, rec_head):
    med = (_dual(rec_reg["median_days"], rec_full["median_days"], _fmt_mo)
           if use_rec else _fmt_mo(rec_head["median_days"]))
    return dip_analytics.time_underwater_caption(
        sym, band, state["current_dd"], median_text=med,
        p90_text=_fmt_mo(rec_head["p90_days"]),
        n_complete=rec_head["n_complete"], n_censored=rec_head["n_censored"])


def _track_record(recov):
    pct = (f" ({recov['recovery_rate'] * 100:.0f}%)."
           if np.isfinite(recov["recovery_rate"]) else ".")
    return (f"Track record: {recov['recovered']} of {recov['n_episodes']} dips of "
            f"5%+ eventually climbed back to their old high" + pct +
            " Broad indexes recover; single stocks sometimes don't — that's why "
            "dip-buying sticks to SPY/SCHD.")


# --- ad-hoc typed-ticker lookup (PR-B) ------------------------------------- #
# Mirrors app.py's `🔍 Check any ticker` form (9592-9625), reusing the same
# engine (dip_adhoc.resolve_adhoc) and the same _build_card the auto cards use,
# so the typed card matches Streamlit 1:1. The status prose is terminal-native
# plain text (no markdown — rendered via textContent). _adhoc_fetchers is a
# module-level function so tests can monkeypatch it (the error/stale seam).
def _adhoc_fetchers(data_dir):
    """(price_fn, div_fn, persist) for an ad-hoc fetch — terminal copy of
    app.py._adhoc_fetchers. Offline fixture reader when data_dir/dip_adhoc_source.csv
    exists (the test seam), else the live Yahoo wrappers with sidecar persistence."""
    src = Path(data_dir) / "dip_adhoc_source.csv"
    if src.exists():
        price_fn, div_fn = dip_adhoc.offline_fetchers(src)
        return price_fn, div_fn, False
    return fetch_yahoo, fetch_dividends_yahoo, True


def resolve_adhoc_card(data_dir, sym, vintage) -> dict:
    """Ad-hoc typed-ticker resolution: pick fetchers (offline seam vs live Yahoo
    via _adhoc_fetchers) then dip_adhoc.resolve_adhoc. Returns the raw resolve
    payload ({status, price, tr, dser, asof, stale, n_days, msg}). Shared by
    app.py._load_adhoc_dip and build_dip_lookup. Calls _adhoc_fetchers by its
    module name so the test monkeypatch seam still applies."""
    price_fn, div_fn, persist = _adhoc_fetchers(data_dir)
    return dip_adhoc.resolve_adhoc(
        data_dir, sym, vintage, price_fn, div_fn,
        pd.Timestamp.today().normalize(), persist=persist)


def _lookup_note(status, sym, *, asof=None, n_days=0, msg="") -> str:
    """Plain-text per-state caption (no markdown markers — the front-end renders
    notes via textContent). Wording mirrors app.py 9605-9624."""
    if status == "ok":
        return f"Live — {sym} as of {asof}."
    if status == "stale":
        return (f"⚠️ Live fetch failed — showing the last cached {sym} "
                f"(as of {asof}).")
    if status == "short":
        return (f"Only {n_days} trading days of history for {sym} — need "
                f"~{dip_adhoc.MIN_HISTORY_DAYS} (about a year) for dip stats.")
    if status == "empty":
        return f"No data for {sym} — check the symbol."
    if status == "already":
        return f"{sym} is already shown below."
    return f"Couldn't reach the data source for {sym}. Try again. ({msg})"


def build_dip_lookup(frames, ticker) -> dict:
    """Resolve a typed ad-hoc ticker to a JSON-native render payload.

    {ticker, status, asof, note, card, n_days}. `card` (same shape as a
    build_dip_view auto card) is present only for ok/stale. The watch set is the
    full dip_history symbol universe (matching app.py:9593) — the server is the
    authoritative `already` guard; the client guard is a best-effort optimization.
    """
    sym = dip_adhoc.normalize_ticker(ticker)
    data_dir = frames.data_dir
    hist, _divs = _load_dip_csvs(data_dir)
    vintage = (str(pd.to_datetime(hist["date"]).max().date())
               if not hist.empty else None)
    watch = set(hist["symbol"].unique()) if not hist.empty else set()

    base = {"ticker": sym, "status": "empty", "asof": None,
            "note": "", "card": None, "n_days": 0}

    if not sym:  # belt-and-suspenders; the route allowlist forbids blank
        return {**base, "note": _lookup_note("empty", sym)}
    if sym in watch:
        return {**base, "status": "already", "note": _lookup_note("already", sym)}

    res = resolve_adhoc_card(data_dir, sym, vintage)
    status = res["status"]

    if status == "ok":
        out_status = "stale" if res["stale"] else "ok"
        asof = str(res["asof"].date())
        return {**base, "status": out_status, "asof": asof,
                "note": _lookup_note(out_status, sym, asof=asof),
                "card": _build_card(sym, res["price"], res["tr"], res["dser"])}
    if status == "short":
        n = int(res["n_days"])
        return {**base, "status": "short", "n_days": n,
                "note": _lookup_note("short", sym, n_days=n)}
    if status == "empty":
        return {**base, "note": _lookup_note("empty", sym)}
    return {**base, "status": "error",
            "note": _lookup_note("error", sym, msg=res.get("msg", ""))}
