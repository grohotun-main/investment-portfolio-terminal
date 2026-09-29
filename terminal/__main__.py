"""Loopback launcher: ``python -m terminal``.

Binds ``127.0.0.1:8502`` ONLY (Streamlit keeps 8501), prints the URL, tries to
open the browser, then runs uvicorn. Loopback is the auth boundary (spec §7.1) —
never bind ``0.0.0.0`` here.
"""
from __future__ import annotations

import sys
import webbrowser

import uvicorn

HOST, PORT = "127.0.0.1", 8502  # loopback ONLY; Streamlit keeps 8501


def main() -> None:
    # Windows consoles default to cp1252; reconfigure to UTF-8 so any non-ASCII
    # output never crashes startup (the PYTHONUTF8 caveat in CLAUDE.md). Keep the
    # banner ASCII-only regardless, as belt-and-suspenders.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    url = f"http://{HOST}:{PORT}"
    print(f"MERIDIAN Terminal -> {url}  (loopback only; Ctrl-C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run("terminal.server:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
