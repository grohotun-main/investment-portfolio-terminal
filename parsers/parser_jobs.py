"""Parser-job helpers behind the terminal's action endpoints (refresh
buttons): resolve the market-data API key, run a parser script as a
subprocess, and chain scripts into sequences with per-step status.

UI-framework-free by construction: subprocess/os/sys only. Every command
these helpers run writes only gitignored ``data/`` files.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from _config import load_env  # noqa: E402  (parsers/ sibling, like risk_bundle)

# Subprocess cwd: the repo root (this module lives one level down in parsers/).
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_massive_api_key() -> str:
    """Return MASSIVE_API_KEY from the process env, falling back to the
    Windows User-scope registry on win32. Empty string if not found.

    The server process may not inherit the key when launched from a
    PowerShell that started before the key was set. Reading from the User
    scope at click-time lets refresh buttons work without a server restart.
    """
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if key and key != "PASTE_YOUR_KEY_HERE":
        return key
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[System.Environment]::GetEnvironmentVariable("
                 "'MASSIVE_API_KEY','User')"],
                capture_output=True, text=True, timeout=5,
            )
            return (r.stdout or "").strip()
        except Exception:
            return ""
    return ""


def run_parser_subprocess(label: str, cmd: list[str],
                          env_extra: dict | None = None,
                          timeout: int = 900) -> tuple[bool, str]:
    """Run a parser script from the project root and return (ok, tail).

    `tail` is the last ~15 lines of combined stdout/stderr — enough to
    surface the "Wrote N rows" / "All three CSVs land on..." summary lines
    in the UI without dumping the full fetch log.

    `.env` values are merged into the subprocess env as a fallback for
    INTERIM_TXN_DIR (which `ingest_csv_activity.py` reads from `os.environ`
    directly, not via `_config.load_env`). Process env wins over `.env`.
    """
    env = os.environ.copy()
    for k, v in load_env().items():
        env.setdefault(k, v)
    if env_extra:
        env.update(env_extra)
    env.setdefault("PYTHONUTF8", "1")
    try:
        r = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            env=env, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"{label} timed out after {timeout}s."
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-15:])
    return r.returncode == 0, tail


def run_parser_sequence(steps: list[dict], api_key: str,
                        on_step=None
                        ) -> tuple[bool, list[tuple[str, str]], str]:
    """Run several refresh parsers back-to-back (the "Refresh all" shape).

    Each step is a dict: {"label", "cmd", "timeout", "needs_key"}. These are
    independent data refreshes (interim txns, prices, option IV, ATM IV), so a
    failure does NOT abort the rest — every step runs and its outcome is
    reported. A Polygon-hitting step is skipped (not run) when no API key is
    available, and counts against overall success.

    `on_step`, if given, is called for live progress: `on_step("start", label)`
    just before a step runs and `on_step("done", label, icon)` after it, where
    icon is "✅"|"❌"|"⏭️". This lets the caller surface which step is running
    (the steps run synchronously, so without it the UI looks frozen for the
    ~10-15 min the market-data pull takes).

    Returns (all_ok, results, combined_tail) where results is
    [(label, "✅"|"❌"|"⏭️"), ...] for a one-line summary chip and combined_tail
    concatenates each step's stdout tail for the "Last refresh details" view.
    """
    def _notify(phase: str, label: str, icon: str | None = None) -> None:
        if on_step is not None:
            on_step(phase, label, icon)

    results: list[tuple[str, str]] = []
    tails: list[str] = []
    all_ok = True
    for step in steps:
        label = step["label"]
        _notify("start", label)
        if step.get("needs_key") and not api_key:
            results.append((label, "⏭️"))
            tails.append(f"=== {label} ===\nSkipped — MASSIVE_API_KEY not set.")
            all_ok = False
            _notify("done", label, "⏭️")
            continue
        env_extra = ({"MASSIVE_API_KEY": api_key}
                     if step.get("needs_key") else None)
        ok, tail = run_parser_subprocess(label, step["cmd"],
                                         env_extra=env_extra,
                                         timeout=step.get("timeout", 300))
        icon = "✅" if ok else "❌"
        results.append((label, icon))
        tails.append(f"=== {label} ===\n{tail}")
        all_ok = all_ok and ok
        _notify("done", label, icon)
    return all_ok, results, "\n\n".join(tails)
