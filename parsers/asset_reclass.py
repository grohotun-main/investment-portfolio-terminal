"""Asset-class overrides applied after the statement parser's section-header
classification.

Pure function (no I/O, no config import) so it's unit-testable in isolation;
app.py binds the config-dependent arguments (the TLH account id and the
user's ETF override map) and calls it per position. See
tests/test_asset_reclass.py.

Why this layer exists
---------------------
The broker statement's section header is not a reliable source of truth for a
security's asset class:

  * Commodity tickers (GLD, IAU, ...) are tagged inconsistently across sources
    (Harbor "other", Alpine "fixed_income", test brokers "gold"). Without an
    override the Risk-tab Class filter misclassifies them and HHI / sector
    weights are wrong.
  * Some statement formats have no dedicated "Exchange Traded Products"
    section and list ETFs (SPY, SGOV, GLD) under "Stocks / Common Stock", so
    the holdings pipeline tags them ``equity_stock``. The caller-supplied
    ``etf_class`` map (ticker -> canonical class) restores the right class
    regardless of which section a given month's statement filed the security
    under.
"""
from __future__ import annotations

import re

# Broker DISPLAY-format option symbol, e.g. "SPY DEC 26 PUT 650.00" — the form
# interim Harbor option activity uses (OCC is "-SPY261218P575"). Captures the
# CALL/PUT side; the underlying / strike are not needed for classification.
_DISPLAY_OPT_RE = re.compile(
    r"^[A-Z][A-Z.]*\s+"
    r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+"
    r"\d{2}\s+(CALL|PUT)\s+[\d.,]+$")


def _display_option_class(sym: str) -> str | None:
    """``option_call`` / ``option_put`` for a broker DISPLAY-format option
    symbol, else None.

    Interim Harbor option legs arrive in this display format rather than OCC, so
    the synthesizer's OCC regex misses them and books the leg
    ``asset_class='other'`` — which would let it slip into the income
    dividend-channel universe (audit WSD-3). ``sym`` is expected upper-cased."""
    m = _DISPLAY_OPT_RE.match(sym)
    if not m:
        return None
    return "option_call" if m.group(1) == "CALL" else "option_put"


# Commodity tickers -> canonical class, overriding however the broker tagged
# them. Generic public tickers (not holdings-specific), so they live here
# rather than in config_local.
COMMODITY_TICKER_CLASS = {
    "GLD":  "gold",      # SPDR Gold Trust
    "IAU":  "gold",      # iShares Gold Trust
    "GLDM": "gold",      # SPDR Gold MiniShares
    "SLV":  "gold",      # iShares Silver Trust (folded into gold bucket — same
    "SIVR": "gold",      #   risk regime, no separate UI bucket yet)
    "GDX":  "gold",      # VanEck Gold Miners ETF
    "GDXJ": "gold",      # VanEck Junior Gold Miners ETF
}


def reclass_asset(account_id: str, symbol: str | float, asset_class: str, *,
                  tlh_account_id: str, etf_class: dict[str, str]) -> str:
    """Return the display asset class for one position.

    Rules, first match wins:
      1. Everything in the Tax Loss Harvesting account is ``tax_loss_harvesting``.
      2. A broker DISPLAY-format option leg currently booked ``other`` (interim
         Harbor activity, e.g. "SPY DEC 26 PUT 650.00") -> ``option_put`` /
         ``option_call`` so the income engine's class exclusion catches it.
      3. Commodity tickers (GLD, IAU, ...) -> their canonical class regardless
         of the broker tag.
      3. Phase-0 ``commodity_etf`` rows collapse into ``gold``.
      4. A ticker in ``etf_class`` that the broker tagged ``equity_stock``
         (the mis-file we're fixing) -> its mapped class. Guarded on
         ``equity_stock`` so an option leg on the same underlying (a SPY put
         carries symbol "SPY") is never clobbered into an ETF.
      5. Otherwise keep the broker's tag.
    """
    if account_id == tlh_account_id:
        return "tax_loss_harvesting"
    sym = symbol.strip().upper() if isinstance(symbol, str) else ""
    if asset_class == "other":
        opt = _display_option_class(sym)
        if opt is not None:
            return opt
    override = COMMODITY_TICKER_CLASS.get(sym)
    if override is not None:
        return override
    if asset_class == "commodity_etf":
        return "gold"
    if asset_class == "equity_stock":
        etf_override = etf_class.get(sym)
        if etf_override is not None:
            return etf_override
    return asset_class
