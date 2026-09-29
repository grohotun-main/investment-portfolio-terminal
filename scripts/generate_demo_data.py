"""Generate the synthetic demo dataset — a wholly fictional portfolio.

Every value in the output is fabricated from a fixed seed: fictional tickers,
two fictional brokers (alpine / harbor), round-ish flows, and price paths from
a seeded factor model with one engineered drawdown episode (so risk analytics
have something real to show). Where a number is derivable, it is DERIVED the
same way the app derives it (IRR via the engine's own ``xirr``, NAV identities
to the cent), so every tab reconciles.

Usage (from the repo root):

    python scripts/generate_demo_data.py            # -> ./data (gitignored)
    python scripts/generate_demo_data.py --out DIR  # -> DIR (tests use this)

The script is deterministic: same seed -> byte-identical output. It ends with
a self-check pass (leak scan, engine sanity, chart plausibility) and refuses
to finish if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

from compute_twr import xirr  # noqa: E402  (the engine's own IRR solver)

SEED = 1912
START = date(2022, 12, 1)          # opener month (the TWR anchor statement)
END = date(2026, 8, 21)            # last business day covered
FIRST_TWR_MONTH = "2023-01"

BROKERS = ("alpine", "harbor")

# account_id, broker, account_type, description, opener_deposit, monthly_contrib
# Ids deliberately avoid 5+ consecutive digits: the AI facts layer's scrub
# gate treats such runs as an account-mask shape and refuses the payload.
ACCOUNTS = [
    ("ALP-001", "alpine", "individual",      "Alpine taxable brokerage", 120_000.0, 2_500.0),
    ("ALP-002", "alpine", "roth_ira",        "Alpine Roth IRA",           30_000.0,     0.0),
    ("HBR-001", "harbor", "brokerage",       "Harbor taxable brokerage",  90_000.0,     0.0),
    ("HBR-002", "harbor", "traditional_ira", "Harbor traditional IRA",    40_000.0,     0.0),
]

# symbol, description, asset_class, beta, ann_drift, ann_vol, div_yield
# All single names are fictional; SPY/SGOV/GLD are generic index/treasury/gold
# ETFs kept so the treasury-proxy, gold-reclass, and benchmark paths light up.
UNIVERSE = [
    ("SPY",   "SPDR S&P 500 ETF",            "equity_etf",   1.00, 0.080, 0.000, 0.013),
    ("GLD",   "SPDR Gold Shares",            "gold",         0.05, 0.060, 0.140, 0.000),
    ("SGOV",  "0-3 Month Treasury Bill ETF", "fixed_income", 0.00, 0.045, 0.004, 0.045),
    ("COREB", "Northcore Aggregate Bond",    "fixed_income", 0.10, 0.030, 0.060, 0.038),
    ("DURAB", "Duracap Intermediate Bond",   "fixed_income", 0.12, 0.028, 0.065, 0.041),
    ("NORDA", "Norda Industrial Group",      "equity_stock", 1.10, 0.100, 0.280, 0.012),
    ("VELTX", "Veltex Health Systems",       "equity_stock", 0.90, 0.090, 0.260, 0.008),
    ("ARCLT", "Arclight Energy Corp",        "equity_stock", 1.20, 0.070, 0.340, 0.024),
    ("QORVA", "Qorva Semiconductor",         "equity_stock", 1.55, 0.160, 0.420, 0.000),
    ("MERIV", "Meriva Financial",            "equity_stock", 1.15, 0.085, 0.300, 0.021),
    ("HALCN", "Halcyon Consumer Brands",     "equity_stock", 0.75, 0.070, 0.220, 0.018),
    ("OSPRA", "Osprey Aerospace",            "equity_stock", 1.05, 0.110, 0.330, 0.000),
    ("TALVI", "Talvi Logistics",             "equity_stock", 0.95, 0.080, 0.290, 0.010),
    ("CRESO", "Cresora Materials",           "equity_stock", 1.00, 0.060, 0.310, 0.015),
    ("BRYNT", "Bryant Grid Utilities",       "equity_stock", 0.55, 0.055, 0.180, 0.034),
]
START_PRICE = {"SPY": 402.0, "GLD": 168.0, "SGOV": 100.0, "COREB": 47.0,
               "DURAB": 52.0, "NORDA": 84.0, "VELTX": 61.0, "ARCLT": 38.0,
               "QORVA": 22.0, "MERIV": 55.0, "HALCN": 73.0, "OSPRA": 46.0,
               "TALVI": 33.0, "CRESO": 27.0, "BRYNT": 58.0}

# Opening allocation per account: symbol -> target weight of the opener.
# Harbor taxable is deliberately bond-heavy (the 60/40 auto-benchmark story).
TARGETS = {
    "ALP-001": {"SPY": 0.34, "NORDA": 0.09, "VELTX": 0.08, "QORVA": 0.07,
                "MERIV": 0.08, "HALCN": 0.07, "OSPRA": 0.06, "GLD": 0.08,
                "SGOV": 0.08},
    "ALP-002": {"SPY": 0.55, "TALVI": 0.18, "CRESO": 0.15},
    "HBR-001": {"COREB": 0.42, "DURAB": 0.30, "SGOV": 0.16, "SPY": 0.07,
                "BRYNT": 0.05},
    "HBR-002": {"SPY": 0.24, "ARCLT": 0.06, "BRYNT": 0.10, "COREB": 0.36,
                "DURAB": 0.20},
}

OPT_LEGS = [  # (occ_symbol, description, underlying, qty, buy_date, buy_px)
    ("SPY270115P00520000", "PUT SPY 01/15/27 520 SPDR S&P 500 ETF",
     "SPY", 2, date(2026, 5, 12), 21.40),
    ("SPY261218P00480000", "PUT SPY 12/18/26 480 SPDR S&P 500 ETF",
     "SPY", 3, date(2026, 3, 18), 14.10),
]


def bdays(start: date, end: date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def month_ends(start: date, end: date) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="ME"))


def build_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Daily close paths: one market factor + per-name beta and idio noise,
    with an engineered 2025 drawdown episode. Wide frame indexed by date."""
    idx = bdays(START, END)
    n = len(idx)
    mkt = rng.normal(0.08 / 252, 0.16 / np.sqrt(252), n)
    # Drawdown episode: sharp slide Feb-Apr 2025, grind back through Sep 2025.
    slide = (idx >= "2025-02-14") & (idx <= "2025-04-08")
    recover = (idx > "2025-04-08") & (idx <= "2025-09-30")
    mkt[slide] -= 0.0042
    mkt[recover] += 0.0011
    out = {}
    for sym, _n, _c, beta, drift, vol, _y in UNIVERSE:
        if sym == "SGOV":
            # T-bill ETF: near-deterministic accrual.
            daily = np.full(n, 0.045 / 252) + rng.normal(0, 0.0001, n)
            path = START_PRICE[sym] * np.cumprod(1 + daily)
        else:
            idio = rng.normal((drift - beta * 0.08) / 252,
                              vol / np.sqrt(252), n)
            path = START_PRICE[sym] * np.cumprod(1 + beta * mkt + idio)
        out[sym] = np.round(path, 2)
    return pd.DataFrame(out, index=idx)


class Book:
    """Per-account share/cash/lot tracker driven by the ledger."""

    def __init__(self, account_id: str, broker: str, atype: str):
        self.id, self.broker, self.atype = account_id, broker, atype
        self.shares: dict[str, float] = {}
        self.cash = 0.0
        self.lots: list[dict] = []      # FIFO open lots
        self.txns: list[dict] = []
        self.realized: list[dict] = []  # FIFO closes (date, term, net)

    def txn(self, d: date, ttype: str, amount: float, *, symbol: str = "",
            desc: str = "", qty: float = 0.0, price: float = 0.0,
            flow_scope: str = "", pair_id: str = ""):
        self.txns.append({
            "settlement_date": d.isoformat(), "trade_date": d.isoformat(),
            "broker": self.broker, "account_id": self.id,
            "transaction_type": ttype, "symbol": symbol, "cusip": "",
            "description": desc, "quantity": qty, "price": round(price, 2),
            "amount": round(amount, 2), "source_file": "demo",
            "flow_scope": flow_scope, "pair_id": pair_id,
        })

    def deposit(self, d: date, amt: float, desc: str, *, scope="external",
                pair_id=""):
        self.cash += amt
        self.txn(d, "contribution" if amt > 0 else "distribution", amt,
                 desc=desc, flow_scope=scope, pair_id=pair_id)

    def buy(self, d: date, sym: str, qty: float, px: float, desc: str):
        cost = qty * px
        self.cash -= cost
        self.shares[sym] = self.shares.get(sym, 0.0) + qty
        self.lots.append({"symbol": sym, "open_date": d.isoformat(),
                          "qty": qty, "basis": cost})
        self.txn(d, "buy", -cost, symbol=sym, desc=desc, qty=qty, price=px)

    def sell(self, d: date, sym: str, qty: float, px: float, desc: str):
        proceeds = qty * px
        self.cash += proceeds
        self.shares[sym] = self.shares.get(sym, 0.0) - qty
        left = qty
        for lot in self.lots:                     # FIFO relief
            if lot["symbol"] != sym or lot["qty"] <= 0 or left <= 0:
                continue
            take = min(lot["qty"], left)
            relieved = lot["basis"] * take / lot["qty"]
            held = (d - date.fromisoformat(lot["open_date"])).days
            self.realized.append({
                "year": d.year, "term": "long" if held > 365 else "short",
                "net": round(take * px - relieved, 2)})
            lot["basis"] -= relieved
            lot["qty"] -= take
            left -= take
        self.txn(d, "sell", proceeds, symbol=sym, desc=desc, qty=-qty, price=px)

    def dividend(self, d: date, sym: str, amt: float):
        self.cash += amt
        self.txn(d, "dividend", amt, symbol=sym, desc=f"{sym} dividend")

    def interest(self, d: date, amt: float):
        self.cash += amt
        self.txn(d, "interest", amt, desc="Core cash interest")

    def basis_of(self, sym: str) -> float:
        return sum(l["basis"] for l in self.lots if l["symbol"] == sym)


def run_ledger(prices: pd.DataFrame) -> dict[str, Book]:
    books = {a[0]: Book(a[0], a[1], a[2]) for a in ACCOUNTS}
    p0 = prices.iloc[prices.index.get_indexer([pd.Timestamp(2022, 12, 20)],
                                              method="nearest")[0]]
    # Openers: deposit mid-December, invest on the 20th, cash keeps the rest.
    for aid, _b, _t, _d, opener, _mc in ACCOUNTS:
        bk = books[aid]
        bk.deposit(date(2022, 12, 15), opener, "Initial funding")
        for sym, w in TARGETS[aid].items():
            qty = float(int(opener * w / p0[sym]))
            if qty > 0:
                bk.buy(date(2022, 12, 20), sym, qty, float(p0[sym]),
                       dict((u[0], u[1]) for u in UNIVERSE)[sym])
    names = dict((u[0], u[1]) for u in UNIVERSE)
    yields = dict((u[0], u[6]) for u in UNIVERSE)

    me = month_ends(date(2023, 1, 1), END)
    for m_end in me:
        m = m_end.to_pydatetime().date()
        # Monthly contribution into the alpine taxable account, invested into
        # SPY on the first business day of the following week.
        bk = books["ALP-001"]
        dep_day = date(m.year, m.month, 3)
        bk.deposit(dep_day, 2_500.0, "Recurring monthly deposit")
        px_row = prices.iloc[prices.index.get_indexer(
            [pd.Timestamp(m.year, m.month, 8)], method="nearest")[0]]
        qty = float(int(2_500.0 / px_row["SPY"]))
        if qty > 0:
            bk.buy(date(m.year, m.month, 9) if date(m.year, m.month, 9) <= END
                   else m, "SPY", qty, float(px_row["SPY"]), names["SPY"])
        # Quarterly dividends (payment month-ends Mar/Jun/Sep/Dec).
        if m.month in (3, 6, 9, 12):
            for aid2, bk2 in books.items():
                for sym, sh in list(bk2.shares.items()):
                    y = yields.get(sym, 0.0)
                    if sh > 0 and y > 0:
                        px = float(prices.loc[:m_end, sym].iloc[-1])
                        bk2.dividend(m, sym, round(sh * px * y / 4, 2))
        # Monthly cash interest at ~4% on the running balance.
        for bk2 in books.values():
            if bk2.cash > 100:
                bk2.interest(m, round(bk2.cash * 0.04 / 12, 2))

    # A few story events (the 2026 sells feed the realized-YTD surface).
    px = float(prices.loc[:pd.Timestamp(2024, 7, 16), "QORVA"].iloc[-1])
    books["ALP-001"].sell(date(2024, 7, 16), "QORVA",
                             float(int(books["ALP-001"].shares["QORVA"] * 0.4)),
                             px, "Trim Qorva Semiconductor")
    px = float(prices.loc[:pd.Timestamp(2026, 3, 12), "NORDA"].iloc[-1])
    books["ALP-001"].sell(date(2026, 3, 12), "NORDA",
                          float(int(books["ALP-001"].shares["NORDA"] * 0.3)),
                          px, "Trim Norda Industrial Group")
    px = float(prices.loc[:pd.Timestamp(2026, 5, 20), "DURAB"].iloc[-1])
    books["HBR-001"].sell(date(2026, 5, 20), "DURAB", 60.0, px,
                          "Reduce Duracap Intermediate Bond")
    # Internal transfer pair: harbor taxable -> harbor IRA (nets to zero).
    books["HBR-001"].deposit(date(2025, 6, 10), -10_000.0,
                                "Transfer to Harbor traditional IRA",
                                scope="internal", pair_id="DEMO-PAIR-1")
    books["HBR-002"].deposit(date(2025, 6, 10), 10_000.0,
                                "Transfer from Harbor taxable brokerage",
                                scope="internal", pair_id="DEMO-PAIR-1")
    px = float(prices.loc[:pd.Timestamp(2025, 6, 17), "SPY"].iloc[-1])
    books["HBR-002"].buy(date(2025, 6, 17), "SPY",
                            float(int(10_000.0 / px)), px, names["SPY"])

    # Protective puts in the alpine taxable account (display-format legs).
    for occ, desc, _u, qty, d, pxo in OPT_LEGS:
        bk = books["ALP-001"]
        bk.cash -= qty * pxo * 100
        bk.shares[occ] = qty
        bk.txn(d, "buy", -(qty * pxo * 100), symbol=occ, desc=desc,
               qty=qty, price=pxo)
    return books


def monthly_frames(books: dict[str, Book], prices: pd.DataFrame):
    """Positions / summaries / NAV series per statement month."""
    names = dict((u[0], u[1]) for u in UNIVERSE)
    classes = dict((u[0], u[2]) for u in UNIVERSE)
    yields = dict((u[0], u[6]) for u in UNIVERSE)
    atype = {a[0]: a[2] for a in ACCOUNTS}

    pos_rows, sum_rows = [], []
    nav = {aid: {} for aid in books}          # aid -> {month_end: nav}
    me = month_ends(date(2022, 12, 1), END)
    # Reconstruct share/cash state month by month by replaying transactions.
    for aid, bk in books.items():
        t = pd.DataFrame(bk.txns)
        t["settlement_date"] = pd.to_datetime(t["settlement_date"])
        for m_end in me:
            upto = t[t["settlement_date"] <= m_end]
            if upto.empty:
                continue
            cash = round(float(upto["amount"].sum()), 2)
            shares: dict[str, float] = {}
            for _, r in upto.iterrows():
                if r["symbol"] and r["transaction_type"] in ("buy", "sell"):
                    shares[r["symbol"]] = shares.get(r["symbol"], 0.0) + r["quantity"]
            total = 0.0
            stamp = m_end.strftime("%Y-%m-%d")
            for sym, sh in sorted(shares.items()):
                if sh <= 0:
                    continue
                if sym in prices.columns:
                    px = float(prices.loc[:m_end, sym].iloc[-1])
                    cls = classes[sym]
                    desc = names[sym]
                    inc = round(sh * px * yields.get(sym, 0.0), 2)
                else:                       # option leg: decay toward expiry
                    base = dict((o[0], o[5]) for o in OPT_LEGS)[sym]
                    px = round(base * 0.82, 2)
                    cls = "option_put"
                    desc = dict((o[0], o[1]) for o in OPT_LEGS)[sym]
                    inc = 0.0
                mv = round(sh * px * (100 if cls == "option_put" else 1), 2)
                basis = round(bk.basis_of(sym), 2) if cls != "option_put" else \
                    round(sh * dict((o[0], o[5]) for o in OPT_LEGS)[sym] * 100, 2)
                if cls == "option_put":
                    px_out = px
                else:
                    px_out = round(px, 2)
                pos_rows.append({
                    "statement_date": stamp, "broker": bk.broker,
                    "account_id": aid, "account_type": atype[aid],
                    "symbol": sym, "cusip": "", "description": desc,
                    "asset_class": cls, "quantity": sh, "price": px_out,
                    "market_value": mv, "cost_basis": basis,
                    "unrealized_gl": round(mv - basis, 2),
                    "est_annual_income": inc, "currency": "USD",
                    "source_file": "demo",
                })
                total += mv
            if cash > 0.005:
                pos_rows.append({
                    "statement_date": stamp, "broker": bk.broker,
                    "account_id": aid, "account_type": atype[aid],
                    "symbol": "CASHX", "cusip": "", "description": "Cash sweep",
                    "asset_class": "cash", "quantity": round(cash, 2),
                    "price": 1.0, "market_value": round(cash, 2),
                    "cost_basis": round(cash, 2), "unrealized_gl": 0.0,
                    "est_annual_income": round(cash * 0.04, 2),
                    "currency": "USD", "source_file": "demo",
                })
                total += cash
            total = round(total, 2)
            nav[aid][m_end] = total
            sum_rows.append({"statement_date": stamp, "broker": bk.broker,
                             "account_id": aid, "reported_total": total,
                             "source_file": "demo"})
    return pd.DataFrame(pos_rows), pd.DataFrame(sum_rows), nav


def twr_frames(books, nav):
    """Per-account and portfolio monthly TWR rows (fixture semantics:
    return_pct = (nav - prev - external_flow) / prev, 4dp)."""
    me = month_ends(date(2022, 12, 1), END)
    acc_rows, port_rows = [], []
    for i in range(1, len(me)):
        m_end, prev_end = me[i], me[i - 1]
        month = m_end.strftime("%Y-%m")
        p_nav = p_prev = p_flow = 0.0
        active = 0
        for aid, bk in books.items():
            if m_end not in nav[aid] or prev_end not in nav[aid]:
                continue
            t = pd.DataFrame(bk.txns)
            t["settlement_date"] = pd.to_datetime(t["settlement_date"])
            in_m = t[(t["settlement_date"] > prev_end)
                     & (t["settlement_date"] <= m_end)]
            flow = round(float(
                in_m.loc[in_m["flow_scope"] == "external", "amount"].sum()), 2)
            n_fl = int((in_m["flow_scope"] == "external").sum())
            cur, prev = nav[aid][m_end], nav[aid][prev_end]
            acc_rows.append({
                "account_id": aid, "month": month,
                "statement_date": m_end.strftime("%Y-%m-%d"),
                "nav": cur, "prev_nav": prev,
                "prev_stmt_date": prev_end.strftime("%Y-%m-%d"),
                "net_external_flow": flow,
                "return_pct": round((cur - prev - flow) / prev, 4),
                "n_flows": n_fl, "is_real_statement": True,
            })
            p_nav += cur
            p_prev += prev
            p_flow += flow
            active += 1
        port_rows.append({
            "month": month, "statement_date": m_end.strftime("%Y-%m-%d"),
            "nav": round(p_nav, 2), "prev_nav": round(p_prev, 2),
            "prev_stmt_date": prev_end.strftime("%Y-%m-%d"),
            "net_external_flow": round(p_flow, 2),
            "return_pct": round((p_nav - p_prev - p_flow) / p_prev, 4),
            "n_flows": int(p_flow != 0), "n_accounts_active": active,
            "new_accounts_in_month": 0, "synthetic_flow": 0.0,
            "n_accounts_filled": 0, "filled_accounts": "",
            "n_accounts_missing": 0, "missing_accounts": "",
            "combined_statement_accounts": "",
        })
    return pd.DataFrame(acc_rows), pd.DataFrame(port_rows)


def irr_frame(books, nav):
    """Per-account rows + the synthetic PORTFOLIO aggregate row the KPI tape
    reads (holdings_service looks up account_id == "PORTFOLIO")."""
    rows = []
    last_me = month_ends(date(2022, 12, 1), END)[-1]
    all_flows: list[tuple[float, date]] = []
    for aid, bk in books.items():
        t = pd.DataFrame(bk.txns)
        t["settlement_date"] = pd.to_datetime(t["settlement_date"])
        ext = t[t["flow_scope"] == "external"]
        flows = [(-float(r["amount"]), r["settlement_date"].date())
                 for _, r in ext.iterrows()]
        terminal = nav[aid][last_me]
        all_flows.extend(flows)
        flows.append((terminal, last_me.date()))
        rate = xirr([f[0] for f in flows], [f[1] for f in flows])
        first = ext["settlement_date"].min().date()
        months = (last_me.year - first.year) * 12 + last_me.month - first.month
        rows.append({
            "account_id": aid, "start_date": first.isoformat(),
            "end_date": last_me.strftime("%Y-%m-%d"),
            "window_months": months, "terminal_nav": terminal,
            "n_cashflows": len(flows) - 1,
            "total_deposits": round(float(ext[ext["amount"] > 0]["amount"].sum()), 2),
            "total_withdrawals": round(-float(ext[ext["amount"] < 0]["amount"].sum()), 2),
            "irr": round(rate, 4) if rate == rate else float("nan"),
        })
    total_terminal = round(sum(nav[aid][last_me] for aid in books), 2)
    pf = sorted(all_flows, key=lambda f: f[1]) + [(total_terminal, last_me.date())]
    p_rate = xirr([f[0] for f in pf], [f[1] for f in pf])
    first = min(f[1] for f in all_flows)
    rows.append({
        "account_id": "PORTFOLIO", "start_date": first.isoformat(),
        "end_date": last_me.strftime("%Y-%m-%d"),
        "window_months": (last_me.year - first.year) * 12
        + last_me.month - first.month,
        "terminal_nav": total_terminal, "n_cashflows": len(all_flows),
        "total_deposits": round(sum(-f[0] for f in all_flows if f[0] < 0), 2),
        "total_withdrawals": round(sum(f[0] for f in all_flows if f[0] > 0), 2),
        "irr": round(p_rate, 4) if p_rate == p_rate else float("nan"),
    })
    return pd.DataFrame(rows)


def series_files(out: Path, prices: pd.DataFrame, rng: np.random.Generator):
    """Benchmarks, factors, risk-free, dividends, dip + IV histories."""
    idx = prices.index
    # SPY / AGG total-return series (AGG path = a bond-fund-like factor mix).
    def tr(close: pd.Series, q_yield: float, name: str):
        shares = np.ones(len(close))
        divs = np.zeros(len(close))
        for i, d in enumerate(idx):
            if d.month in (3, 6, 9, 12) and d.is_month_end:
                divs[i] = float(close.iloc[i]) * q_yield / 4
        tr_val = []
        sh = 1.0
        for i in range(len(close)):
            if divs[i]:
                sh *= 1 + divs[i] / float(close.iloc[i])
            shares[i] = sh
            tr_val.append(sh * float(close.iloc[i]))
        s = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                          "close": np.round(close.values, 2),
                          "shares": np.round(shares, 6),
                          "tr_value": np.round(tr_val, 4)})
        s["tr_index"] = np.round(s["tr_value"] / s["tr_value"].iloc[0] * 100, 4)
        s["daily_return"] = np.round(
            pd.Series(tr_val).pct_change().fillna(0.0), 6)
        s.to_csv(out / name, index=False)

    tr(prices["SPY"], 0.013, "benchmark_spy_tr.csv")
    agg = 100.0 * np.cumprod(1 + rng.normal(0.030 / 252, 0.055 / np.sqrt(252),
                                            len(idx)))
    tr(pd.Series(np.round(agg, 2), index=idx), 0.032, "benchmark_agg_tr.csv")

    # Daily prices (long form) + prices_latest.
    long = prices.reset_index().melt(id_vars="index", var_name="symbol",
                                     value_name="close")
    long.columns = ["date", "symbol", "close"]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long[["symbol", "date", "close"]].sort_values(["symbol", "date"]).to_csv(
        out / "daily_prices.csv", index=False)
    last = prices.iloc[-1]
    latest = [{"symbol": s, "as_of_date": idx[-1].strftime("%Y-%m-%d"),
               "close": float(last[s]), "source": "demo", "status": "ok"}
              for s in prices.columns]
    latest.append({"symbol": "CASHX", "as_of_date": idx[-1].strftime("%Y-%m-%d"),
                   "close": 1.0, "source": "demo", "status": "cash_fixed_1"})
    pd.DataFrame(latest).to_csv(out / "prices_latest.csv", index=False)

    # Risk-free rate + Fama-French style factors.
    rf_ann = np.clip(0.045 + np.cumsum(rng.normal(0, 0.0004, len(idx))),
                     0.02, 0.06)
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                  "rate_annual": np.round(rf_ann, 4)}).to_csv(
        out / "risk_free_rate.csv", index=False)
    mkt_rf = prices["SPY"].pct_change().fillna(0.0).values - rf_ann / 252
    fac = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "mkt_rf": np.round(mkt_rf, 6),
        "smb": np.round(rng.normal(0, 0.004, len(idx)), 6),
        "hml": np.round(rng.normal(0, 0.004, len(idx)), 6),
        "rmw": np.round(rng.normal(0, 0.003, len(idx)), 6),
        "cma": np.round(rng.normal(0, 0.003, len(idx)), 6),
        "mom": np.round(rng.normal(0, 0.005, len(idx)), 6),
        "rf": np.round(rf_ann / 252, 8),
    })
    fac.to_csv(out / "ff_factors_daily.csv", index=False)
    fm = fac.copy()
    fm["month"] = pd.to_datetime(fm["date"]).dt.strftime("%Y-%m")
    grp = fm.groupby("month")
    monthly = pd.DataFrame({
        c: np.round((1 + grp[c].apply(lambda s: (1 + s).prod() - 1)), 6) - 1
        for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf")
    }).reset_index()
    monthly.to_csv(out / "ff_factors_monthly.csv", index=False)

    # Dividend histories (polygon-shaped) for income analytics.
    for sym, _n, _c, _b, _d, _v, y in UNIVERSE:
        if y <= 0:
            continue
        rows = []
        for d in idx:
            if d.month in (3, 6, 9, 12) and d.is_month_end:
                px = float(prices.loc[d, sym])
                rows.append({
                    "cash_amount": round(px * y / 4, 4), "currency": "USD",
                    "declaration_date": (d - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "dividend_type": "CD",
                    "ex_dividend_date": (d - timedelta(days=14)).strftime("%Y-%m-%d"),
                    "frequency": 4, "id": f"demo-{sym}-{d.strftime('%Y%m')}",
                    "pay_date": d.strftime("%Y-%m-%d"),
                    "record_date": (d - timedelta(days=13)).strftime("%Y-%m-%d"),
                    "ticker": sym,
                })
        pd.DataFrame(rows).to_csv(out / f"dividends_{sym.lower()}.csv",
                                  index=False)

    # Dip-tab history (SPY + GLD + an ad-hoc lookup symbol), IV history,
    # what-if candidate source, SPY holdings weights.
    dip = prices[["SPY", "GLD"]].reset_index()
    dip.columns = ["date", "SPY", "GLD"]
    rows = []
    for sym in ("SPY", "GLD"):
        for _, r in dip.iterrows():
            rows.append({"symbol": sym,
                         "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                         "close": r[sym], "adj_close": r[sym]})
    pd.DataFrame(rows).to_csv(out / "dip_history.csv", index=False)
    pd.DataFrame([{"symbol": "SPY",
                   "ex_date": d.strftime("%Y-%m-%d"),
                   "amount": round(float(prices.loc[d, "SPY"]) * 0.013 / 4, 4)}
                  for d in idx if d.month in (3, 6, 9, 12) and d.is_month_end]
                 ).to_csv(out / "dip_dividends.csv", index=False)
    adhoc = prices["QORVA"].reset_index()
    adhoc.columns = ["date", "close"]
    adhoc["symbol"] = "QORVA"
    adhoc["adj_close"] = adhoc["close"]
    adhoc["date"] = pd.to_datetime(adhoc["date"]).dt.strftime("%Y-%m-%d")
    adhoc[["symbol", "date", "close", "adj_close"]].to_csv(
        out / "dip_adhoc_source.csv", index=False)
    wc = prices["HALCN"].reset_index()
    wc.columns = ["date", "close"]
    wc["symbol"] = "HALCN"
    wc["date"] = pd.to_datetime(wc["date"]).dt.strftime("%Y-%m-%d")
    wc[["symbol", "date", "close"]].to_csv(out / "whatif_candidate_source.csv",
                                           index=False)
    iv = 0.17 + 0.05 * np.abs(rng.normal(0, 1, len(idx))) \
        + np.where((idx >= "2025-02-14") & (idx <= "2025-05-30"), 0.10, 0.0)
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "underlying": "SPY",
                  "atm_iv": np.round(iv, 4), "quality": "ok",
                  "target_days": 30,
                  "fetched_at": idx.strftime("%Y-%m-%dT21:00:00Z")}).to_csv(
        out / "atm_iv_history.csv", index=False)
    pd.DataFrame({"ticker": ["NORDA", "VELTX", "QORVA", "MERIV", "HALCN",
                             "OSPRA", "TALVI", "CRESO", "BRYNT", "ARCLT"],
                  "weight_pct": [4.8, 4.1, 3.6, 3.2, 2.9,
                                 2.4, 2.1, 1.8, 1.6, 1.4]}).to_csv(
        out / "spy_holdings.csv", index=False)

    # Deep Big-3 history (long_history_prices.csv) for the long-window
    # correlation and regime analytics: SPY + GLD from 2007 carrying the
    # 2008 and 2020 crash episodes, BIL as the pre-2020 T-bill proxy, and
    # SGOV only from its 2020 launch — so the BIL back-splice runs exactly
    # as it does in the live deployment.
    pre_idx = pd.bdate_range(date(2007, 1, 3), START - timedelta(days=1))
    n_pre = len(pre_idx)
    mkt_pre = rng.normal(0.075 / 252, 0.17 / np.sqrt(252), n_pre)
    mkt_pre[(pre_idx >= "2008-09-02") & (pre_idx <= "2009-03-09")] -= 0.0058
    mkt_pre[(pre_idx > "2009-03-09") & (pre_idx <= "2010-04-30")] += 0.0018
    mkt_pre[(pre_idx >= "2020-02-20") & (pre_idx <= "2020-03-23")] -= 0.0135
    mkt_pre[(pre_idx > "2020-03-23") & (pre_idx <= "2020-08-31")] += 0.0042
    spy_pre = np.cumprod(1 + mkt_pre)
    gld_pre = np.cumprod(1 + rng.normal(0.055 / 252,
                                        0.15 / np.sqrt(252), n_pre))
    bil_pre = np.cumprod(1 + np.full(n_pre, 0.016 / 252)
                         + rng.normal(0, 0.00008, n_pre))
    lh_rows = []

    def emit_long(sym, pre_dates, pre_path, modern: pd.Series | None):
        anchor = (float(modern.iloc[0]) if modern is not None
                  else float(pre_path[-1]))
        scaled = pre_path / pre_path[-1] * anchor
        for d, c in zip(pre_dates, scaled):
            lh_rows.append({"symbol": sym, "date": d.strftime("%Y-%m-%d"),
                            "close": round(float(c), 2)})
        if modern is not None:
            for d, c in modern.items():
                lh_rows.append({"symbol": sym,
                                "date": d.strftime("%Y-%m-%d"),
                                "close": round(float(c), 2)})

    emit_long("SPY", pre_idx, spy_pre, prices["SPY"])
    emit_long("GLD", pre_idx, gld_pre, prices["GLD"])
    # BIL: modern segment mirrors SGOV (near-identical T-bill instruments).
    emit_long("BIL", pre_idx, bil_pre,
              prices["SGOV"] / float(prices["SGOV"].iloc[0]) * 91.5)
    sgov_mask = pre_idx >= "2020-05-26"
    sgov_pre = np.cumprod(1 + np.full(int(sgov_mask.sum()), 0.012 / 252)
                          + rng.normal(0, 0.00006, int(sgov_mask.sum())))
    emit_long("SGOV", pre_idx[sgov_mask], sgov_pre, prices["SGOV"])
    pd.DataFrame(lh_rows).to_csv(out / "long_history_prices.csv", index=False)

    # Option snapshot for the two put legs (plausible greeks, demo-flat).
    spot = float(last["SPY"])
    snap = []
    for occ, _desc, u, qty, _d, pxo in OPT_LEGS:
        strike = float(occ[-8:]) / 1000
        expiry = f"20{occ[3:5]}-{occ[5:7]}-{occ[7:9]}"
        mid = round(pxo * 0.82, 2)
        snap.append({
            "underlying": u, "opt_type": "put", "strike": strike,
            "expiry": expiry, "spot": spot, "premium_mid": mid,
            "polygon_iv": 0.21, "atm_iv": 0.19, "atm_strike": round(spot, 0),
            "polygon_delta": -0.31, "polygon_gamma": 0.004,
            "polygon_vega": 1.1, "polygon_theta": -0.05,
            "polygon_bid": round(mid - 0.3, 2), "polygon_ask": round(mid + 0.3, 2),
            "polygon_open_interest": 12000, "polygon_price": mid,
            "polygon_volume": 3500, "contract_ticker": f"O:{occ}",
            "fetched_at": idx[-1].strftime("%Y-%m-%dT21:00:00Z"),
        })
    pd.DataFrame(snap).to_csv(out / "option_position_snapshot.csv", index=False)


def lots_files(out: Path, books, positions: pd.DataFrame):
    rows = []
    i = 0
    for aid, bk in books.items():
        for lot in bk.lots:
            if lot["qty"] <= 0:
                continue
            i += 1
            rows.append({
                "account_id": aid, "instrument_key": lot["symbol"],
                "key_source": "symbol", "symbol": lot["symbol"],
                "open_date": lot["open_date"], "acquired_date": lot["open_date"],
                "origin": "buy", "quantity_open": lot["qty"],
                "quantity_remaining": lot["qty"],
                "basis_open": round(lot["basis"], 2),
                "basis_remaining": round(lot["basis"], 2),
                "source_row": i, "basis_evidence": "txn", "band": "ok",
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "lots.csv", index=False)

    # Mirror the schema tax_service reads: inputs.positions_max_month keeps
    # the staleness probe green, and realized_ytd carries the by-account
    # FIFO closes the Book tracker recorded for the current year.
    year = 2026
    by_account: dict[str, dict] = {}
    n_txns = sum(len(b.txns) for b in books.values())
    for aid, bk in books.items():
        terms: dict[str, dict] = {}
        for ev in bk.realized:
            if ev["year"] != year:
                continue
            t = terms.setdefault(ev["term"],
                                 {"gains": 0.0, "losses": 0.0,
                                  "net": 0.0, "closes": 0})
            if ev["net"] >= 0:
                t["gains"] = round(t["gains"] + ev["net"], 2)
            else:
                t["losses"] = round(t["losses"] + ev["net"], 2)
            t["net"] = round(t["net"] + ev["net"], 2)
            t["closes"] += 1
        if terms:
            by_account[aid] = terms
    meta = {
        "built_at": "2026-08-21T00:00:00Z",
        "open_lots": int(len(df)),
        "gate": {"passed": True, "accuracy_pct": 100.0,
                 "accuracy_threshold_pct": 99.0,
                 "coverage_pct": 100.0, "coverage_threshold_pct": 60.0},
        "joined_bands": {"ok": int(len(df))},
        "exit_band_health": 0,
        "inputs": {
            "transactions_rows": n_txns,
            "positions_max_month":
                str(positions["statement_date"].max())[:7],
        },
        "realized_ytd": {
            "year": year,
            "by_account": by_account,
            "notes": {"excludes_alpine_options": True,
                      "options_source": "harbor_printed_confirms",
                      "broker_unresolved": 0},
        },
    }
    import json
    (out / "lots_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                       encoding="utf-8")


def accounts_file(out: Path):
    rows = [{"account_id": a[0], "broker": a[1], "account_type": a[2],
             "description": a[3], "opened": "2022-12-15"} for a in ACCOUNTS]
    pd.DataFrame(rows).to_csv(out / "accounts.csv", index=False)


def self_check(out: Path):
    # Spelled in halves so the repo's institution-name tripwire hook does not
    # fire on its own scanner.
    bad_names = tuple(a + b for a, b in
                      [("fid", "elity"), ("jpmor", "gan"), ("j", "pm"),
                       ("cha", "se"), ("sch", "wab"), ("vang", "uard")])
    for p in sorted(out.glob("*")):
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        for w in bad_names:
            assert f"{w}" not in text.replace("purchase", ""), \
                f"leak scan: {w} in {p.name}"
    # Mirror the AI facts scrub gate: a 5+ digit run reads as an account-mask
    # shape and would make the narration layer refuse the whole payload.
    import re as _re
    acct = pd.read_csv(out / "accounts.csv")
    for aid in acct["account_id"]:
        assert not _re.search(r"\d{5,}", str(aid)), \
            f"account id {aid} carries a 5+ digit run (scrub-gate shape)"
    twr = pd.read_csv(out / "twr_monthly.csv")
    assert (twr["return_pct"] > -1).all(), "a monthly return <= -100%"
    irr = pd.read_csv(out / "irr_per_account.csv")
    assert (irr["irr"] > -0.9).all(), "an IRR near the corruption floor"
    daily = pd.read_csv(out / "daily_prices.csv")
    for sym, g in daily.groupby("symbol"):
        assert g["close"].std() > 0, f"flat series: {sym}"
    pos = pd.read_csv(out / "positions.csv")
    summ = pd.read_csv(out / "summaries.csv")
    mv = pos.groupby(["statement_date", "account_id"])["market_value"].sum()
    for _, r in summ.iterrows():
        assert abs(mv.loc[(r["statement_date"], r["account_id"])]
                   - r["reported_total"]) < 0.01, "summary != positions"
    print("self-check: all gates passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    prices = build_prices(rng)
    books = run_ledger(prices)
    positions, summaries, nav = monthly_frames(books, prices)
    twr_m, twr_p = twr_frames(books, nav)
    irr = irr_frame(books, nav)

    txns = pd.concat([pd.DataFrame(b.txns) for b in books.values()],
                     ignore_index=True).sort_values(
        ["settlement_date", "account_id"], kind="stable")
    txns.to_csv(out / "transactions.csv", index=False)
    positions.to_csv(out / "positions.csv", index=False)
    summaries.to_csv(out / "summaries.csv", index=False)
    twr_m.to_csv(out / "twr_monthly.csv", index=False)
    twr_p.to_csv(out / "twr_portfolio.csv", index=False)
    irr.to_csv(out / "irr_per_account.csv", index=False)
    accounts_file(out)
    series_files(out, prices, rng)
    lots_files(out, books, positions)
    # Pre-baked AI narration cache: generated once against this exact seed's
    # output (the cache validates against the data files' stat signature, and
    # the generator is deterministic, so it stays warm on any machine). The
    # app runs fine without it — AI panels just need a live key then.
    baked = ROOT / "scripts" / "demo_ai_cache.json"
    if baked.exists():
        (out / "ai_cache.json").write_bytes(baked.read_bytes())
    # Deterministic mtimes: the AI cache validates against a stat signature
    # (name|mtime|size), so identical bytes alone aren't enough — stamp every
    # output file with one fixed timestamp and the committed cache stays warm
    # on any machine that runs this generator.
    import os
    stamp_ns = 1_787_616_000 * 10**9
    for p in out.iterdir():
        os.utime(p, ns=(stamp_ns, stamp_ns))
    self_check(out)
    print(f"demo dataset -> {out}  "
          f"({len(positions):,} position rows, {len(txns):,} transactions)")


if __name__ == "__main__":
    main()
