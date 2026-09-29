"""Tiny env loader — no python-dotenv dependency.

Resolution order for MASSIVE_API_KEY:
  1. os.environ  — preferred. Set via PowerShell:
         setx MASSIVE_API_KEY "your_key_here"
     then open a NEW PowerShell window so the value is loaded.
     This path keeps the key out of any file Claude Code is watching.
  2. .env file at phase1_build/.env  (fallback)

This module also makes this machine's TLS-intercepting environment safe at
import time, in two ways:

1. It injects the Windows certificate store into Python's SSL trust path.
   Reason: Norton's TLS-scanning layer re-signs every outbound HTTPS
   connection with the "Norton Web/Mail Shield Root" cert, which lives in the
   Windows store but not in certifi's bundle. Without this, api.polygon.io
   calls fail with CERTIFICATE_VERIFY_FAILED.
2. It strips a hostile ``SSLKEYLOGFILE`` (a Windows device-namespace path
   injected by a TLS-monitoring agent) that otherwise makes every TLS
   connection abort with PermissionError. See ``_neutralize_hostile_keylog``.
"""
import os
import sys
from pathlib import Path


def _neutralize_hostile_keylog() -> None:
    r"""Strip an ``SSLKEYLOGFILE`` that points into the Windows device
    namespace (``\\.\...``) from the process environment.

    A TLS-monitoring agent on this machine injects a per-session
    ``SSLKEYLOGFILE`` naming one of its own named pipes (observed:
    ``\\.\nllMonFltProxy\<hex>``, ``\\.\nlaKdbg_engine_ipc_<hex>``) into the
    processes it spawns. Python's ``ssl.create_default_context()`` honours that
    variable and opens the path to log TLS session keys. The pipe is
    session-scoped, so it goes stale — and once it does, EVERY SSL context
    creation raises ``PermissionError(13)`` / ``FileNotFoundError`` and aborts
    the connection. On this box that silently breaks two things:
      * the Polygon / requests fetchers ("Refresh all data" -> "Connection
        aborted"), and
      * the terminal's ``anthropic.Anthropic()`` construction (every
        ``/api/ai/*`` route -> 500, so the AI Analysis tab never loads).

    ``ssl`` re-reads the variable on every context creation, so popping it from
    ``os.environ`` here — at import, before any HTTP client is built — is a
    permanent, process-wide fix. A device-namespace path is never a keylog
    file a developer set on purpose, so a legitimate file-path
    ``SSLKEYLOGFILE`` is deliberately left untouched.
    """
    v = os.environ.get("SSLKEYLOGFILE", "")
    if v and v.replace("/", "\\").startswith("\\\\.\\"):
        os.environ.pop("SSLKEYLOGFILE", None)


# Import-time side effect: run BEFORE any SSL context is created below.
_neutralize_hostile_keylog()

try:
    # Side-effect at import time: route Python's SSL trust through the
    # Windows certificate store so requests trusts the Norton "Web/Mail
    # Shield Root" CA. Do NOT remove — without it api.polygon.io fails with
    # CERTIFICATE_VERIFY_FAILED on this machine. See module docstring.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # Optional — needed only on machines with TLS-intercepting AV.

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_PATH) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _get(key: str, default: str = "") -> str:
    v = os.environ.get(key, "").strip()
    if v:
        return v
    return load_env().get(key, default).strip()


def get_massive_key() -> str:
    key = _get("MASSIVE_API_KEY")
    if not key or key == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            "MASSIVE_API_KEY not set. Two options:\n"
            '  (a) Recommended:  setx MASSIVE_API_KEY "your_key_here"\n'
            "      then open a NEW PowerShell window before re-running.\n"
            f"  (b) Edit {ENV_PATH} and paste your key after MASSIVE_API_KEY=\n"
        )
    return key


def get_massive_base() -> str:
    return _get("MASSIVE_API_BASE", "https://api.polygon.io") or "https://api.polygon.io"


def _user_scope_env(key: str) -> str:
    """Windows User-scope env var read straight from the registry (empty
    off-win32 or on failure). A server launched from a shell that predates a
    ``setx`` inherits a stale environment; this fallback keeps key resolution
    working without restarting the machine's shells (same idea as
    parsers/parser_jobs.resolve_massive_api_key, via winreg instead of a
    PowerShell subprocess)."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            v, _ = winreg.QueryValueEx(h, key)
            return str(v).strip()
    except OSError:
        return ""


def get_anthropic_key() -> str:
    """Anthropic API key for the terminal's AI narration layer, or "" when
    unset (OPTIONAL — never raises; AI panels render a quiet off-state
    without it). Resolution: os.environ, then .env, then the Windows
    User-scope registry (so a terminal launched from a pre-setx shell still
    finds the key)."""
    key = _get("ANTHROPIC_API_KEY")
    if not key:
        key = _user_scope_env("ANTHROPIC_API_KEY")
    if key == "PASTE_YOUR_KEY_HERE":
        return ""
    return key


def get_fred_api_key() -> str:
    """FRED API key, or "" when unset (OPTIONAL — unlike the Massive key this
    never raises). When set, fetch_risk_free_rate.py prefers FRED's API host
    (api.stlouisfed.org), which stays reachable on networks where the keyless
    fredgraph host (fred.stlouisfed.org) is blocked by TLS interception; when
    unset it falls back to the keyless graph CSV. Free key:
    https://fredaccount.stlouisfed.org/apikeys
    """
    return _get("FRED_API_KEY")
