"""MERIDIAN Portfolio Terminal — post-Streamlit UI (FastAPI + vanilla front-end).

Loopback-only. Reuses the parsers/ engine through ``terminal.holdings_service``,
which re-expresses the Streamlit Holdings tab's inline aggregations as a pure,
importable data seam. The Streamlit app (``app.py``) is independent and
unaffected by this package.
"""

# parsers/ is a flat module directory (holdings_service does ``from
# mark_to_market import ...`` etc.). app.py puts it on sys.path at startup and
# the test suite does the same; do it here too so any entry point (``python -m
# terminal``, ``uvicorn terminal.server:app``) imports cleanly without a manual
# PYTHONPATH. Mirrors app.py's sys.path.insert.
import sys as _sys
from pathlib import Path as _Path

_PARSERS = _Path(__file__).resolve().parent.parent / "parsers"
if _PARSERS.is_dir() and str(_PARSERS) not in _sys.path:
    _sys.path.insert(0, str(_PARSERS))
