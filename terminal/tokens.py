"""CSS custom-property block generated from ``theme.py``.

``theme.py`` is the single source of truth for the MERIDIAN palette the
Streamlit app uses. ``root_css`` emits a ``:root { --token: value; … }`` block
built from those same constants so the vanilla terminal front-end references
``var(--token)`` and cannot drift from Streamlit on colour / typography.
"""
from __future__ import annotations

import theme

# Token name → theme constant. Kept explicit (not introspected) so the mapping
# is auditable and a renamed theme constant fails loudly at import time.
_TOKENS = {
    "--bg":         theme.CANVAS,
    "--bg-rail":    theme.PANEL,
    "--card":       theme.CARD,
    "--hover":      theme.HOVER,
    "--border":     theme.BORDER,
    "--accent":     theme.ACCENT,
    "--gain":       theme.GAIN,
    "--loss":       theme.LOSS,
    "--text":       theme.TEXT_PRIMARY,
    "--text-muted": theme.TEXT_MUTED,
    "--text-2":     theme.TEXT_SECONDARY,
    "--font-sans":  theme.FONT_SANS,
    "--font-mono":  theme.FONT_MONO,
}


def root_css() -> str:
    """Return a ``:root{…}`` CSS block of the MERIDIAN design tokens.

    Served at ``GET /tokens.css`` and linked first by ``index.html`` so every
    later rule can reference ``var(--token)``.
    """
    body = "".join(f"{name}:{value};" for name, value in _TOKENS.items())
    return f":root{{{body}}}"
