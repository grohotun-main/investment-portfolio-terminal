"""Background parser-job runner for the terminal's action buttons (QA-polish S6).

Wraps ``parsers.parser_jobs`` (the same helpers app.py's sidebar buttons use —
single-sourced there this slice). Commands are verbatim from app.py; every one
writes only gitignored ``data/`` files, and the server is loopback-only, so the
trust domain matches Streamlit's buttons.

Single-flight by design: one job at a time (the parsers contend on the same
CSVs). State is a module-level snapshot the front-end polls; the job itself
runs on a daemon thread so the event loop never blocks.
"""
from __future__ import annotations

import copy
import sys
import threading
import time

# parsers/ is a flat module directory on sys.path (same convention as every
# terminal service's engine imports, e.g. holdings_service's risk_bundle).
from parser_jobs import resolve_massive_api_key, run_parser_sequence

_PY = sys.executable

# Step dicts are the run_parser_sequence contract: {label, cmd, timeout, needs_key}.
_STEP_MARKET = {"label": "Market data", "needs_key": True, "timeout": 1800,
                "cmd": [_PY, "parsers/refresh_prices.py"]}
_STEP_OPTION_IV = {"label": "Option IV", "needs_key": True, "timeout": 120,
                   "cmd": [_PY, "parsers/fetch_option_position_iv.py", "--write"]}
_STEP_ATM_IV = {"label": "ATM IV history", "needs_key": True, "timeout": 600,
                "cmd": [_PY, "parsers/fetch_atm_iv_history.py", "--write"]}
_STEP_DIP = {"label": "Dip history", "needs_key": False, "timeout": 600,
             "cmd": [_PY, "parsers/fetch_dip_history.py", "--write"]}

ACTIONS: dict[str, dict] = {
    "refresh_all": {
        "label": "Refresh all data", "group": "rail",
        "help": "Market data (Polygon + FRED + VIX), dip history (Yahoo), "
                "option IV, ATM IV history — back-to-back. ~10-15 min on a "
                "full pull (market data is the long pole).",
        "steps": [_STEP_MARKET, _STEP_DIP,
                  _STEP_OPTION_IV, _STEP_ATM_IV],
    },
    "market_data": {
        "label": "Refresh market data", "group": "rail",
        "help": "All 3 Polygon fetchers + FRED risk-free rate + CBOE VIX. "
                "~2-3 min.",
        "steps": [{**_STEP_MARKET, "timeout": 900}],
    },
    "dividends": {
        "label": "Refresh dividend history", "group": "income",
        "help": "parsers/fetch_dividends.py --holdings --write (Polygon).",
        "steps": [{"label": "Dividend history", "needs_key": True, "timeout": 600,
                   "cmd": [_PY, "parsers/fetch_dividends.py", "--holdings", "--write"]}],
    },
    "factor_data": {
        "label": "Fetch factor data", "group": "factor",
        "help": "parsers/fetch_ff_factors.py --write (Ken French Data Library "
                "— free, keyless).",
        "steps": [{"label": "FF factor fetch", "needs_key": False, "timeout": 120,
                   "cmd": [_PY, "parsers/fetch_ff_factors.py", "--write"]}],
    },
    "option_iv": {
        "label": "Refresh option IV", "group": "options",
        "help": "parsers/fetch_option_position_iv.py --write (Polygon).",
        "steps": [_STEP_OPTION_IV],
    },
    "dip_history": {
        "label": "Refresh dip history", "group": "dip",
        "help": "parsers/fetch_dip_history.py --write (Yahoo — keyless). Deep "
                "SPY/SCHD/GLD + watchlist history for the Buy-the-Dip tab; "
                "also runs as part of Refresh all data.",
        "steps": [_STEP_DIP],
    },
    "atm_iv": {
        "label": "Refresh ATM IV history", "group": "options",
        "help": "parsers/fetch_atm_iv_history.py --write (Polygon).",
        "steps": [_STEP_ATM_IV],
    },
}

# Test seam: when set, every action runs THESE steps instead of its own —
# keeps the suite offline (no real parser is ever shelled by a test).
_TEST_STEPS: list[dict] | None = None

_lock = threading.Lock()
_state: dict = {
    "running": False, "action": None, "label": None,
    "steps": [], "tail": "", "ok": None,
    "started_at": None, "finished_at": None,
}


def status() -> dict:
    """A deep-copied snapshot of the job state (the poll target)."""
    with _lock:
        return copy.deepcopy(_state)


def actions_view() -> dict:
    """The button roster + current state for GET /api/actions."""
    acts = [{
        "id": aid, "label": a["label"], "help": a["help"], "group": a["group"],
        "needs_key": any(s.get("needs_key") for s in a["steps"]),
    } for aid, a in ACTIONS.items()]
    return {"actions": acts, "status": status()}


def start(action_id: str) -> tuple[bool, str]:
    """Start an action on a daemon thread. (False, why) when unknown or busy."""
    action = ACTIONS.get(action_id)
    if action is None:
        return False, "unknown action"
    steps = _TEST_STEPS if _TEST_STEPS is not None else action["steps"]
    with _lock:
        if _state["running"]:
            return False, "another action is already running"
        _state.update({
            "running": True, "action": action_id, "label": action["label"],
            "steps": [{"label": s["label"], "status": "pending"} for s in steps],
            "tail": "", "ok": None,
            "started_at": time.time(), "finished_at": None,
        })
    threading.Thread(target=_run, args=(steps,), daemon=True).start()
    return True, "started"


def _run(steps: list[dict]) -> None:
    api_key = resolve_massive_api_key()

    def on_step(phase: str, label: str, icon: str | None = None) -> None:
        with _lock:
            for row in _state["steps"]:
                if row["label"] == label:
                    row["status"] = "running" if phase == "start" else icon
                    break

    try:
        all_ok, _results, combined = run_parser_sequence(
            steps, api_key, on_step=on_step)
    except Exception as exc:  # never leave the lock-out stuck on a crash
        all_ok, combined = False, f"job runner crashed: {exc!r}"
    with _lock:
        _state.update({"running": False, "ok": all_ok, "tail": combined,
                       "finished_at": time.time()})
