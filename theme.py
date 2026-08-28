"""Design tokens — single source of truth for the terminal's palette.

The MERIDIAN terminal is server-rendered data + a vanilla-JS front-end: chart
colors are decided service-side and shipped in payloads, so the palette lives
here in Python and `terminal/static/app.css` mirrors the same values for the
static chrome. Presentation only — no data or numbers depend on this module.
"""
from __future__ import annotations

# --- Surfaces (cool blue-grey ramp; flat fills + hairlines, no gradients) ---
CANVAS = "#07090D"   # app canvas
PANEL = "#0A0E15"    # nav rail
CARD = "#151C27"     # every panel / KPI / table card
HOVER = "#1C2532"    # inputs, hover fills, bar tracks (elevated)
BORDER = "#28323F"   # row + card hairlines

# --- Text ladder -----------------------------------------------------------
TEXT_PRIMARY = "#EDF2F9"
TEXT_SECONDARY = "#A4B0C0"
TEXT_MUTED = "#7C8A9C"

# --- Accent + semantic -----------------------------------------------------
ACCENT = "#4DA3F5"   # azure — active nav, primary, key chart line, focus
GAIN = "#2FD79A"     # teal-green (colorblind-separated from LOSS by hue+warmth)
LOSS = "#FB6F63"     # warm coral
NEUTRAL = TEXT_SECONDARY  # benchmark / no-change

# --- Semantic chart roles --------------------------------------------------
CHART_PORTFOLIO = ACCENT
CHART_BENCH = TEXT_SECONDARY      # benchmark goes gray
CHART_DRAWDOWN = LOSS

# --- Diverging heatmap scales (teal→amber→coral; colorblind-kinder than a
#     green→red ramp). DIVERGING: low end = bad (coral) → high end = good
#     (teal); for signed Δ-vs-baseline matrices. CORR: low end = good (teal,
#     ρ≈0 diversifier) → high end = bad (coral, ρ→1); stop positions match
#     the asymmetric [-0.3, 1.0] correlation domain used on the Risk tab.
HEATMAP_DIVERGING = [
    [0.00, "#FB6F63"],   # coral — well below baseline
    [0.25, "#E68A5C"],
    [0.50, "#E6B450"],   # amber — baseline
    [0.75, "#7FCFA0"],
    [1.00, "#2FD79A"],   # teal — well above baseline
]
HEATMAP_CORR = [
    [0.0,   "#2FD79A"],   # teal — ρ≈0 (genuine diversifier)
    [0.231, "#7FCFA0"],
    [0.385, "#C2D98C"],
    [0.538, "#E6CD8E"],
    [0.692, "#E6B450"],   # amber
    [0.846, "#F08A5C"],
    [1.0,   "#FB6F63"],   # coral — ρ→1 (clustered)
]

# --- Asset-class palette (lightened for legibility on dark) ----------------
CLASS_COLORS = {
    "equity_etf":          "#4DA3F5",   # azure
    "equity_stock":        "#2DD4BF",   # teal
    "tax_loss_harvesting": "#38BDF8",   # cyan (kept distinct from equity_stock)
    "fixed_income":        "#8B7CF6",   # violet
    "gold":                "#E6B450",   # amber
    "cash":                "#9AA6B6",   # cool grey
    "option_put":          "#FB7185",   # rose
    "option_call":         "#F59E0B",   # amber-orange
    "mutual_fund":         "#818CF8",   # indigo
    "other":               "#79859A",   # cool slate
}
# Keyed by the broker ids that appear in the data's `broker` column.
BROKER_COLORS = {
    "alpine": "#4DA3F5", "harbor": "#2DD4BF",
}

# --- Typography ------------------------------------------------------------
FONT_SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'Cascadia Code', Consolas, monospace"


def tint(hex_color: str, alpha: float = 0.18) -> str:
    """`rgba(r,g,b,alpha)` string from a `#rrggbb` token."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def pnl_color(x) -> str:
    """GAIN for x>0, LOSS for x<0, NEUTRAL for 0 / NaN / non-numeric."""
    try:
        if x > 0:
            return GAIN
        if x < 0:
            return LOSS
    except (TypeError, ValueError):
        pass
    return NEUTRAL
