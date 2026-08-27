"""Single-name protective-put hedge report -> one offline HTML file.

Usage (PowerShell, repo root):
  py tools\\hedge_report.py --ticker AMAT --shares 1800 --sell-by 2027-01-15
Optional: --floors 10,20,30,40,50  --kicker-otm 45  --kicker-budget-pct 10
          --no-iv-history  --out PATH

Spec: docs/superpowers/specs/2026-07-06-single-name-hedge-report-design.md.
Layering: gather_inputs() does ALL network/disk; build_report_data() and
render_html() are pure (offline-testable)."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _config import get_massive_key, get_massive_base          # noqa: E402
from dip_adhoc import resolve_adhoc                            # noqa: E402
from fetch_dip_history import fetch_yahoo, fetch_dividends_yahoo  # noqa: E402
from fetch_options_chains import fetch_chain                   # noqa: E402
from fetch_single_name_iv_history import load_or_refresh_iv_history  # noqa: E402
from options_pricer import implied_vol                         # noqa: E402
import single_name_hedge as snh                                # noqa: E402

DATA = ROOT / "data"
KEY_HINT = ("$env:MASSIVE_API_KEY = [System.Environment]::"
            "GetEnvironmentVariable('MASSIVE_API_KEY','User')")


def resolve_api_key() -> str:
    """env -> .env (get_massive_key) -> HKCU registry (the app.py
    _resolve_massive_api_key fallback, so a plain CLI run just works)."""
    try:
        return get_massive_key()
    except RuntimeError:
        pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "MASSIVE_API_KEY")
        if val:
            os.environ["MASSIVE_API_KEY"] = str(val)
            return str(val)
    except (OSError, ImportError):
        pass
    print("[!] MASSIVE_API_KEY not found (env, .env, or User registry).\n"
          f"    In PowerShell run:  {KEY_HINT}")
    raise SystemExit(2)


def assert_sale_covering_chain(rows: list[dict], ticker: str,
                               sell_by: date) -> None:
    """Spec error-table row: exit 3, listing the expiries that WERE found,
    when the chain is empty or nothing expires on/after the sale date —
    an all-infeasible 'report' would be a confusing artifact, not an answer."""
    if not rows:
        print(f"[!] empty chain for {ticker} — check the ticker / market "
              f"hours / plan tier")
        raise SystemExit(3)
    expiries = sorted({str(r.get("expiration_date"))[:10] for r in rows
                       if r.get("expiration_date")})
    if not any(e >= sell_by.isoformat() for e in expiries):
        print(f"[!] no listed expiry on/after {sell_by} for {ticker}; "
              f"found: {', '.join(expiries)}")
        raise SystemExit(3)


def _trailing_yield(dser: pd.Series, spot: float) -> float:
    if dser is None or not len(dser) or not spot > 0:
        return 0.0
    cutoff = dser.index.max() - pd.Timedelta(days=365)
    return float(dser[dser.index >= cutoff].sum()) / spot


def _self_consistent_iv_today(quotes, spot, today, r, q_yield) -> float | None:
    """Invert TODAY's ~35-DTE nearest-ATM put the same way the history was
    built (daily close first — the history inverts closes; ask only as
    fallback), so the percentile compares like with like (decision 11)."""
    cands = [q for q in quotes
             if 20 <= (q.expiry - today).days <= 70]
    if not cands:
        return None
    q = min(cands, key=lambda x: (abs((x.expiry - today).days - 35),
                                  abs(x.strike - spot)))
    px = (q.last_price if (q.last_price is not None and q.last_price > 0)
          else (q.ask if (q.ask is not None and q.ask > 0) else None))
    if px is None:
        return None
    t = (q.expiry - today).days / 365.0
    iv = implied_vol(px, spot, q.strike, t, r, q_yield, "put")
    return float(iv) if iv == iv else None


def _pkg_dict(p: snh.FloorPackage) -> dict:
    return {"floor_pct": p.floor_pct, "contracts": p.contracts,
            "ticker": p.quote.contract_ticker,
            "expiry": p.quote.expiry.isoformat(), "strike": p.quote.strike,
            "bid": p.quote.bid, "ask": p.quote.ask,
            "last": p.quote.last_price, "iv": p.quote.iv,
            "open_interest": p.quote.open_interest,
            "buy_price": p.buy_price, "stale_quote": p.stale_quote,
            "total_cost": p.total_cost, "cost_pct": p.cost_pct,
            "guaranteed_value": p.guaranteed_value,
            "guaranteed_loss_pct": p.guaranteed_loss_pct,
            "market_implied_prob": p.market_implied_prob,
            "hist_prob": p.hist_prob}


def build_report_data(*, ticker: str, shares: int, sell_by: date,
                      today: date, floors: list[float], kicker_otm: float,
                      kicker_budget_pct: float, chain_rows: list[dict],
                      spot: float | None, hist: dict, rf: float,
                      vix: pd.Series | None, iv_payload: dict | None,
                      iv_solve_rows: list[dict] | None = None) -> dict:
    """Pure composition: raw inputs -> everything the renderer needs."""
    warnings: list[str] = []
    quotes = snh.normalize_chain(chain_rows)
    quotes, n_stale_dropped = snh.filter_stale_dominated(quotes)
    if n_stale_dropped:
        warnings.append(f"{n_stale_dropped} stale price print(s) excluded "
                        f"(a put priced below a lower strike in the same "
                        f"expiry is an unexecutable bargain).")
    price = hist.get("price", pd.Series(dtype=float))
    if spot is None or not spot > 0:
        if len(price):
            spot = float(price.iloc[-1])
            warnings.append("Live spot unavailable — using the last close "
                            "from history; regenerate before trading.")
        else:
            raise SystemExit("[!] no spot price available at all")
    if shares % 100:
        warnings.append(f"{shares % 100} shares are an odd lot standard "
                        f"contracts cannot cover — the floor protects "
                        f"{(shares // 100) * 100} shares.")
    if hist.get("stale"):
        warnings.append("Price history is cached as of "
                        f"{pd.Timestamp(hist.get('asof')).date()}.")

    q_yield = _trailing_yield(hist.get("dser"), spot)
    horizon_td = max(round((sell_by - today).days * 252 / 365), 21)
    odds = (snh.horizon_loss_odds(price, horizon_td, floors)
            if hist.get("status") in ("ok", "short") and len(price) else None)
    if odds is None:
        warnings.append("No usable price history — historical odds omitted; "
                        "the floor math is unaffected.")

    menu_pkgs = snh.build_floor_menu(quotes, spot, shares, sell_by, floors,
                                     today=today, r=rf, q_yield=q_yield,
                                     hist_odds=odds)
    if any(p is not None and p.stale_quote for p in menu_pkgs):
        warnings.append("Some quotes had no live ask — day-close prices "
                        "used and flagged; regenerate during market hours.")

    ill = None   # illustration package: feasible floor nearest 0.20
    feas = [p for p in menu_pkgs if p is not None]
    if feas:
        ill = min(feas, key=lambda p: abs(p.floor_pct - 0.20))
    # Kicker sizing anchor: the stock's own worst historical crash replayed
    # from today's price (cheapest-deepest degenerates on big run-ups).
    replays = snh.crash_replays(price) if len(price) else []
    worst_dd = min((r["drawdown_pct"] for r in replays), default=None)
    replay_price = spot * (1.0 + worst_dd) if worst_dd is not None else None
    kicker = None
    if ill is not None:
        kicker = snh.kicker_package(quotes, spot, shares, sell_by,
                                    otm_pct=kicker_otm,
                                    budget=kicker_budget_pct * ill.total_cost,
                                    replay_price=replay_price)

    iv_series = iv_payload["iv"] if (iv_payload
                                     and iv_payload.get("status") in ("ok", "thin")
                                     and len(iv_payload.get("iv", []))) else None
    if iv_payload and iv_payload.get("status") == "unavailable":
        warnings.append("Historical option data unavailable for the IV "
                        "percentile — showing labeled proxies instead.")
    if iv_solve_rows is not None:
        solve_quotes, _ = snh.filter_stale_dominated(
            snh.normalize_chain(iv_solve_rows))
    else:
        solve_quotes = quotes   # already filtered above
    iv_today = _self_consistent_iv_today(solve_quotes, spot, today, rf,
                                         q_yield)
    vol = snh.vol_context(price, iv_today, vix=vix, iv_series=iv_series)
    if vol["iv_source"] == "proxy" and iv_series is not None:
        warnings.append("Backed-out IV history is too thin to rank against "
                        "— realized-vol percentile carries the context.")

    crash_marks = pd.DataFrame()
    iv_for_marks = iv_today
    if iv_for_marks is None and ill is not None:
        iv_for_marks = ill.quote.iv   # package's own IV beats the anchor as "today"
    if ill is not None:
        crash_marks = snh.crash_mark_table(
            ill, today=today, spot=spot, r=rf, q=q_yield,
            iv_today=iv_for_marks, crash_iv_anchor=vol["crash_iv_anchor"])

    s_grid = np.linspace(0.05 * spot, 1.30 * spot, 126)
    tight = feas[0] if feas else None
    loose = feas[-1] if feas else None
    payoff = {"replays": replays,
              "tight": None, "loose": None, "ill_kicker": None}
    if tight is not None:
        payoff["tight"] = snh.payoff_grid(spot, shares, tight, None, s_grid)
        payoff["tight_floor"] = tight.floor_pct
    if loose is not None and loose is not tight:
        payoff["loose"] = snh.payoff_grid(spot, shares, loose, None, s_grid)
        payoff["loose_floor"] = loose.floor_pct
    if ill is not None and kicker is not None:
        payoff["ill_kicker"] = snh.payoff_grid(spot, shares, ill, kicker, s_grid)
        payoff["ill_floor"] = ill.floor_pct

    return {
        "meta": {"ticker": ticker.upper(), "shares": shares,
                 "spot": spot, "value": spot * shares,
                 "sell_by": sell_by.isoformat(), "today": today.isoformat(),
                 "rf": rf, "q_yield": q_yield, "horizon_td": horizon_td},
        "menu": [(_pkg_dict(p) if p is not None else None) for p in menu_pkgs],
        "floors": floors,
        "odds": odds,
        "vol": vol,
        "kicker": (None if kicker is None else {
            "ticker": kicker.quote.contract_ticker,
            "expiry": kicker.quote.expiry.isoformat(),
            "strike": kicker.quote.strike, "contracts": kicker.contracts,
            "buy_price": kicker.buy_price, "total_cost": kicker.total_cost,
            "stale_quote": kicker.stale_quote,
            "open_interest": kicker.quote.open_interest}),
        "crash_marks": crash_marks,
        "payoff": payoff,
        "warnings": warnings,
    }


_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;max-width:960px;margin:2rem auto;
     padding:0 1rem;color:#1a1a1a;line-height:1.45}
h1{font-size:1.5rem} h2{font-size:1.15rem;margin-top:2.2rem;border-bottom:2px solid #eee;
   padding-bottom:.25rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92rem}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}
th{background:#f5f5f7} td:first-child,th:first-child{text-align:left}
.caption{color:#555;font-size:.85rem;margin:.2rem 0 .8rem}
.badge{display:inline-block;background:#eef;border:1px solid #99c;border-radius:4px;
       padding:0 .4rem;font-size:.8rem;margin-left:.4rem}
.warn{background:#fff7e6;border:1px solid #e6c200;border-radius:6px;
      padding:.6rem .8rem;margin:.5rem 0;font-size:.9rem}
.box{background:#f5f9ff;border:1px solid #bcd;border-radius:6px;
     padding:.8rem 1rem;margin:.8rem 0}
.infeasible{color:#a00}
"""


def _usd(x) -> str:
    return f"${x:,.0f}" if x is not None else "—"


def _pctf(x, dp=1) -> str:
    return f"{100 * x:.{dp}f}%" if x is not None else "—"


def _dash(x, fmt="{:.2f}") -> str:
    if x is None:
        return "—"
    try:
        if x != x:   # NaN
            return "—"
    except TypeError:
        pass
    return fmt.format(x)


def _odds_cell(hp: dict | None) -> str:
    if not hp or hp.get("prob") != hp.get("prob"):
        return "—"
    src = {"empirical": "measured", "gpd": "tail model", "none": "few cases"}
    s = f"{100 * hp['prob']:.1f}%"
    if hp.get("lo") == hp.get("lo"):
        s += f" ({100 * hp['lo']:.0f}–{100 * hp['hi']:.0f}%)"
    return s + f" <span class='badge'>{src.get(hp.get('source'), '?')}</span>"


def _menu_html(data: dict) -> str:
    rows = []
    for f, p in zip(data["floors"], data["menu"]):
        if p is None:
            rows.append(f"<tr><td>lose ≤ {_pctf(f, 0)}</td>"
                        f"<td colspan='7' class='infeasible'>not achievable "
                        f"with listed strikes</td></tr>")
            continue
        rows.append(
            f"<tr><td>lose ≤ {_pctf(f, 0)}</td>"
            f"<td>{p['contracts']} puts @ {p['strike']:g}, exp {p['expiry']}</td>"
            f"<td>{_usd(p['total_cost'])}</td><td>{_pctf(p['cost_pct'])}</td>"
            f"<td>{_usd(p['guaranteed_value'])}</td>"
            f"<td>{_pctf(p['guaranteed_loss_pct'])}</td>"
            f"<td>{_odds_cell(p['hist_prob'])}</td>"
            f"<td>{_pctf(p['market_implied_prob'])}</td></tr>")
    return (
        "<table><tr><th>Goal</th><th>What to buy</th><th>Cost</th>"
        "<th>Cost %</th><th>You keep at least</th><th>True worst case</th>"
        "<th>Odds it pays (this stock's history)</th>"
        "<th>Odds the market charges</th></tr>" + "".join(rows) + "</table>"
        "<p class='caption'>How to read a row: pay the Cost today; if the "
        "stock then falls ANY amount below the strike by the sale date, you "
        "still walk away with at least “You keep at least”. The worst case "
        "already includes the cost of the insurance itself. If the stock "
        "doesn't fall, the cost is all you lose — like an unused insurance "
        "policy. “Odds the market charges” are implied by option prices and "
        "include a fear premium — usually higher than real-world frequency. "
        "History odds are measured at the floor level; the market column "
        "prices finishing below the strike — a slightly different bar — so "
        "the two can differ beyond the fear premium.</p>")


def _tickets_html(data: dict) -> str:
    rows = []
    for p in data["menu"]:
        if p is None:
            continue
        stale = " <span class='badge'>day-close quote</span>" if p["stale_quote"] else ""
        rows.append(
            f"<tr><td>{p['ticker']}{stale}</td><td>{_pctf(p['floor_pct'], 0)}</td>"
            f"<td>{_dash(p['bid'])}</td><td>{_dash(p['ask'])}</td>"
            f"<td>{_dash(p['last'])}</td><td>{_dash(p['iv'], '{:.0%}')}</td>"
            f"<td>{p['open_interest']}</td><td>{p['contracts']}</td></tr>")
    return (
        "<table><tr><th>Contract</th><th>Floor</th><th>Bid</th><th>Ask</th>"
        "<th>Last</th><th>IV</th><th>Open interest</th><th>Buy</th></tr>"
        + "".join(rows) + "</table>"
        "<p class='caption'>Use a LIMIT order near the bid/ask midpoint — "
        "18 contracts is small enough to fill without moving the market. "
        "Every IV here should be read against the percentile context above.</p>")


def _vol_html(data: dict) -> str:
    v = data["vol"]
    if v["iv_source"] == "true" and v["iv_percentile"] is not None:
        head = (f"<p><b>Put insurance on this stock is pricier today than "
                f"{v['iv_percentile']:.0f}% of all days since "
                f"{v['iv_history_start']}</b> (true option-market history; "
                f"series peak {_dash(v['iv_history_high'], '{:.0%}')}).</p>")
    elif v["iv_source"] == "true":
        head = ("<p><b>Today's at-the-money point couldn't be solved from "
                "the chain</b> — percentile headline omitted; the option-"
                "history and realized-vol context below still stand.</p>")
    else:
        head = ("<p><b>No usable option-price history for a true IV "
                "percentile</b> <span class='badge'>proxy</span> — the "
                "stock's own price-swing history carries the context "
                "below.</p>")
    lines = []
    if v["rv_percentile"] == v["rv_percentile"]:   # non-NaN (short history)
        lines.append(
            f"Actual choppiness (21-day realized vol "
            f"{_dash(v['rv_today'], '{:.0%}')}) is higher than "
            f"{v['rv_percentile']:.0f}% of its own history.")
    if v["iv_rv_spread"] is not None:
        lines.append(f"The market charges about "
                     f"{100 * v['iv_rv_spread']:.0f} points over that — "
                     f"the insurance markup.")
    if v["vix_percentile"] is not None:
        lines.append(f"Market-wide fear (VIX {v['vix_today']:.0f}) sits at "
                     f"the {v['vix_percentile']:.0f}th percentile.")
    verdict = ("Context, not a prediction: the menu prices above already "
               "embed today's IV. A high percentile means today is an "
               "expensive day to buy — if the purchase isn't urgent, "
               "spreading it over a few days smooths that.")
    return (head + "<ul>" + "".join(f"<li>{ln}</li>" for ln in lines)
            + f"</ul><p class='caption'>{verdict}</p>")


def _payoff_html(data: dict) -> str:
    pay = data["payoff"]
    if pay.get("tight") is None:
        return "<p class='warn'>No feasible floor — no payoff chart.</p>"
    m = data["meta"]
    fig = go.Figure()
    g = pay["tight"]
    fig.add_scatter(x=g["price"], y=g["unhedged"], name="No insurance",
                    line={"color": "#888"})
    fig.add_scatter(x=g["price"], y=g["hedged"],
                    name=f"Floor {_pctf(pay['tight_floor'], 0)}",
                    line={"color": "#c62828"})
    if pay.get("loose") is not None:
        gl = pay["loose"]
        fig.add_scatter(x=gl["price"], y=gl["hedged"],
                        name=f"Floor {_pctf(pay['loose_floor'], 0)}",
                        line={"color": "#1565c0"})
    if pay.get("ill_kicker") is not None:
        gk = pay["ill_kicker"]
        fig.add_scatter(x=gk["price"], y=gk["hedged_kicker"],
                        name=f"Floor {_pctf(pay['ill_floor'], 0)} + kicker",
                        line={"color": "#2e7d32", "dash": "dash"})
    for rep in pay["replays"]:
        px = m["spot"] * (1.0 + rep["drawdown_pct"])
        fig.add_vline(x=px, line_dash="dot", line_color="#999")
        fig.add_annotation(x=px, y=1.0, yref="paper", showarrow=False,
                           textangle=-90, font={"size": 10},
                           text=f"{rep['label']}: {_pctf(rep['drawdown_pct'], 0)}")
    fig.update_layout(template="plotly_white", height=460,
                      margin={"l": 60, "r": 20, "t": 30, "b": 40},
                      xaxis_title=f"{m['ticker']} price at the sale date",
                      yaxis_title="What the position is worth",
                      legend={"orientation": "h", "y": -0.18})
    chart = fig.to_html(full_html=False, include_plotlyjs="inline",
                        config={"displayModeBar": False})
    return chart + (
        "<p class='caption'>These lines are the WORST case (puts valued at "
        "expiry, cost included). Selling earlier, leftover time value, or a "
        "fear spike can only land you ABOVE them. Dotted verticals replay "
        "this stock's own past crashes from today's price.</p>")


def _marks_html(data: dict) -> str:
    cm = data["crash_marks"]
    if cm is None or len(cm) == 0:
        return "<p class='warn'>No feasible core package — no early-crash table.</p>"
    scen = [label for _, label in snh._IV_SCENARIOS]
    rows = []
    for (drop, at_td), grp in cm.groupby(["drop_pct", "at_td"]):
        by = {r["iv_scenario"]: r for _, r in grp.iterrows()}
        cells = "".join(
            f"<td>{_usd(by[s]['value'])}<br>"
            f"<span class='caption'>+{_usd(by[s]['excess'])} vs worst case"
            f"</span></td>" for s in scen)
        months_txt = (f"~{max(1, round(at_td / 21))} month"
                      + ("s" if round(at_td / 21) > 1 else ""))
        rows.append(f"<tr><td>stock −{drop:.0%}, "
                    f"{months_txt} from now</td>"
                    f"{cells}</tr>")
    return (
        "<table><tr><th>Scenario</th><th>Fear unchanged</th>"
        "<th>Halfway to crash level</th><th>At crash level</th></tr>"
        + "".join(rows) + "</table>"
        "<p class='caption'>What your puts would be WORTH if sold during the "
        "crash, before expiry — they keep time value, and fear makes puts "
        "expensive (you'd be selling insurance back when everyone wants "
        "it). If fear is already at crash levels today, the three columns "
        "collapse together: no more room to go.</p>"
        "<div class='box'><b>In a crash, sell the puts — don't exercise "
        "them.</b> Exercising sells your shares at the strike THIS year, "
        "which triggers the very tax event the January plan avoids. Selling "
        "the puts banks the payout, keeps the shares, and keeps your "
        "timing.</div>")


def _kicker_html(data: dict) -> str:
    k = data["kicker"]
    if k is None:
        return ("<p>No affordable deep-out-of-the-money contract with open "
                "interest was listed — skip the kicker.</p>")
    return (
        f"<p>Optional lottery-ticket layer: <b>{k['contracts']} more puts</b> "
        f"at strike {k['strike']:g} (exp {k['expiry']}, {k['ticker']}), "
        f"costing {_usd(k['total_cost'])} total. "
        f"It guarantees nothing — but it is sized to pay the most if this "
        f"stock's own worst historical crash repeated from today's price, "
        f"and its cost is capped at what you paid. That is the 90/10 idea: "
        f"most of the budget buys the certainty above; a small slice buys "
        f"the far tail.</p>")


def _how_sure_html(data: dict) -> str:
    return (
        "<ol>"
        "<li><b>The floor is contract arithmetic, not a model.</b> At the "
        "sale date it is exact; before then it can only be better (an "
        "American put is never worth less than what exercising it pays, and "
        "the time decay is already prepaid inside the cost).</li>"
        "<li><b>Costs are live quotes.</b> They move every day — regenerate "
        "this report the day you trade.</li>"
        "<li><b>The odds are estimates.</b> Ranges shown are statistical "
        "confidence intervals from this stock's own history; the market-"
        "implied numbers embed a fear premium. Where history is thin, the "
        "table says so.</li></ol>")


_TAX = ("<p>Puts on stock already held long-term (more than a year) do not "
        "restart that status and are not a constructive sale — but tax "
        "straddle rules can change how a LOSS on the puts is treated, and "
        "state/AMT details vary. Confirm the plan with a tax professional "
        "before trading. This report is analysis, not tax or investment "
        "advice.</p>")

_GLOSSARY = [
    ("Put", "a contract that lets you sell 100 shares at a fixed price "
            "(the strike) until it expires — price insurance"),
    ("Strike", "the guaranteed sale price per share — your deductible level"),
    ("Premium", "what the insurance costs you today, per share"),
    ("Expiry", "the policy's end date — pick it AFTER your planned sale"),
    ("Contract", "one option covers exactly 100 shares"),
    ("IV (implied volatility)", "the market's fear gauge baked into the "
            "price — high IV = expensive insurance"),
    ("Open interest", "how many contracts exist — a liquidity hint"),
    ("OCC ticker", "the exchange's exact name for one contract — what you "
            "type into the broker"),
]


def render_html(data: dict) -> str:
    m = data["meta"]
    sale_month = date.fromisoformat(m["sell_by"]).strftime("%B %Y")
    warn = "".join(f"<div class='warn'>{w}</div>" for w in data["warnings"])
    gloss = "".join(f"<tr><td>{t}</td><td style='text-align:left'>{d}</td></tr>"
                    for t, d in _GLOSSARY)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{m['ticker']} protective-put plan — {m['today']}</title>
<style>{_CSS}</style></head><body>
<h1>{m['ticker']}: sell in {sale_month} without riding a crash down</h1>
<div class='warn'>Prices move; regenerate this report the day you trade.
Generated {m['today']} — quotes are that day's.</div>
{warn}
<h2>1. Your position today</h2>
<p>{m['shares']:,} shares × {_usd(m['spot'])} = <b>{_usd(m['value'])}</b>,
planned sale by <b>{m['sell_by']}</b>. The question this report answers:
what exactly do you buy so that, no matter what happens before then, you
cannot lose more than a limit you choose?</p>
<h2>2. The insurance menu</h2>{_menu_html(data)}
<h2>3. Is today a pricey day to buy insurance?</h2>{_vol_html(data)}
<h2>4. The trade tickets</h2>{_tickets_html(data)}
<h2>5. Pictures: what you end up with</h2>{_payoff_html(data)}
<h2>6. If the crash comes early — what the puts would actually be worth</h2>
{_marks_html(data)}
<h2>7. The crash kicker (optional add-on)</h2>{_kicker_html(data)}
<h2>8. How sure are these numbers?</h2>{_how_sure_html(data)}
<h2>9. Taxes (one paragraph, not advice)</h2>{_TAX}
<h2>10. Glossary</h2><table>{gloss}</table>
</body></html>"""


def gather_inputs(args) -> dict:
    """ALL network/disk in one place (untested by unit tests; exercised by
    the live run)."""
    key = resolve_api_key()
    base = get_massive_base()
    today = date.today()
    days_to_sale = (args.sell_by - today).days
    if days_to_sale <= 0:
        raise SystemExit("[!] --sell-by must be in the future")
    rows, spot = fetch_chain(args.ticker,
                             max(7, days_to_sale - 45), days_to_sale + 150,
                             key=key, base=base)
    assert_sale_covering_chain(rows, args.ticker, args.sell_by)
    iv_solve_rows = None
    if max(7, days_to_sale - 45) > 70:
        # the sale-covering window can't contain a ~35-DTE contract; fetch a
        # small dedicated window so the IV percentile compares like with like
        iv_solve_rows, _ = fetch_chain(args.ticker, 20, 70, key=key, base=base)
    hist = resolve_adhoc(DATA, args.ticker, today, fetch_yahoo,
                         fetch_dividends_yahoo, today, persist=True)
    rf_path = DATA / "risk_free_rate.csv"
    if rf_path.exists():
        rf_df = pd.read_csv(rf_path, parse_dates=["date"])
        r_series = rf_df.set_index("date")["rate_annual"]
        rf = float(r_series.iloc[-1])
    else:
        r_series, rf = pd.Series(dtype=float), 0.04
    vix_path = DATA / "vix_history.csv"
    vix = None
    if vix_path.exists():
        vdf = pd.read_csv(vix_path, parse_dates=["date"])
        vix = vdf.set_index("date").iloc[:, 0]
    iv_payload = None
    if not args.no_iv_history and hist.get("status") in ("ok", "short"):
        spot_eff = spot if spot else float(hist["price"].iloc[-1])
        print("[iv] building/refreshing the backed-out IV history — the "
              "FIRST run takes minutes; per-month progress below.")
        iv_payload = load_or_refresh_iv_history(
            DATA, args.ticker, hist["price"], r_series,
            _trailing_yield(hist.get("dser"), spot_eff), key=key, base=base)
        print(f"[iv] status={iv_payload['status']} "
              f"rows={len(iv_payload['iv'])} "
              f"first_covered={iv_payload['first_covered']}")
    return {"chain_rows": rows, "spot": spot, "hist": hist, "rf": rf,
            "vix": vix, "iv_payload": iv_payload,
            "iv_solve_rows": iv_solve_rows, "today": today}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--shares", required=True, type=int)
    ap.add_argument("--sell-by", required=True, dest="sell_by",
                    type=date.fromisoformat)
    ap.add_argument("--floors", default="10,20,30,40,50")
    ap.add_argument("--kicker-otm", default=45.0, type=float)
    ap.add_argument("--kicker-budget-pct", default=10.0, type=float)
    ap.add_argument("--no-iv-history", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    inputs = gather_inputs(args)
    data = build_report_data(
        ticker=args.ticker, shares=args.shares, sell_by=args.sell_by,
        today=inputs["today"],
        floors=[float(f) / 100.0 for f in args.floors.split(",")],
        kicker_otm=args.kicker_otm / 100.0,
        kicker_budget_pct=args.kicker_budget_pct / 100.0,
        chain_rows=inputs["chain_rows"], spot=inputs["spot"],
        hist=inputs["hist"], rf=inputs["rf"], vix=inputs["vix"],
        iv_payload=inputs["iv_payload"],
        iv_solve_rows=inputs["iv_solve_rows"])
    html = render_html(data)
    out = Path(args.out) if args.out else (
        DATA / f"hedge_report_{args.ticker.upper()}_"
               f"{inputs['today']:%Y%m%d}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
