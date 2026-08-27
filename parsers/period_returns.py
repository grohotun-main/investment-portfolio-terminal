"""Trailing-window return decomposition for portfolio-vs-benchmark tables.

Given the per-month aligned frame from build_twr_comparison (`statement_date`,
`port_return`, `bench_return` — monthly decimals), compute cumulative or
annualized portfolio and benchmark returns over the standard trailing windows.
A fixed-length window whose data does not reach back to its start is reported
`available: False` rather than as a misleadingly short slice.
"""
from __future__ import annotations

import pandas as pd

# (key, label, spec, annualized). spec: "ytd" | trailing-months int | None (ITD).
WINDOWS = [
    ("ytd", "YTD", "ytd", False),
    ("1y", "1Y", 12, False),
    ("3y", "3Y", 36, True),
    ("5y", "5Y", 60, True),
    ("itd", "Since inception", None, True),
]


def _cum(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def _annualize(cum: float, n_months: int) -> float:
    if n_months <= 0:
        return float("nan")
    return (1.0 + cum) ** (12.0 / n_months) - 1.0


def _vol(returns: pd.Series, n: int):
    """Annualized vol from monthly returns; None (never NaN) when < 2 obs."""
    if n < 2:
        return None
    v = float(returns.std(ddof=1)) * (12.0 ** 0.5)
    return None if pd.isna(v) else v


def _row(key, label, ann, available, port, bench, n, requested,
         port_vol=None, bench_vol=None):
    spread = None if (port is None or bench is None) else port - bench
    return {"key": key, "label": label, "annualized": ann, "available": available,
            "port": port, "bench": bench, "spread": spread,
            "port_vol": port_vol, "bench_vol": bench_vol,
            "n_months": int(n), "requested_months": requested}


def window_returns(comp: pd.DataFrame, as_of=None) -> list[dict]:
    if comp is None or comp.empty:
        return [_row(k, lbl, ann, False, None, None, 0,
                     (spec if isinstance(spec, int) else None))
                for (k, lbl, spec, ann) in WINDOWS]
    c = comp.sort_values("statement_date").reset_index(drop=True).copy()
    c["statement_date"] = pd.to_datetime(c["statement_date"])
    end = pd.Timestamp(as_of) if as_of is not None else c["statement_date"].iloc[-1]
    earliest = c["statement_date"].iloc[0]

    out = []
    for key, label, spec, ann in WINDOWS:
        if spec == "ytd":
            cutoff = pd.Timestamp(year=end.year - 1, month=12, day=31)
            win = c[(c["statement_date"] > cutoff) & (c["statement_date"] <= end)]
            requested, available = None, not win.empty
        elif spec is None:
            win = c[c["statement_date"] <= end]
            requested, available = None, not win.empty
        else:
            cutoff = end - pd.DateOffset(months=spec)
            win = c[(c["statement_date"] > cutoff) & (c["statement_date"] <= end)]
            requested = spec
            available = bool(earliest <= cutoff) and not win.empty
        n = len(win)
        if not available or n == 0:
            out.append(_row(key, label, ann, False, None, None, n, requested))
            continue
        port_cum, bench_cum = _cum(win["port_return"]), _cum(win["bench_return"])
        if ann:
            port_v, bench_v = _annualize(port_cum, n), _annualize(bench_cum, n)
        else:
            port_v, bench_v = port_cum, bench_cum
        out.append(_row(key, label, ann, True, port_v, bench_v, n, requested,
                        port_vol=_vol(win["port_return"], n),
                        bench_vol=_vol(win["bench_return"], n)))
    return out
