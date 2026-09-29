"""
Tests for parsers/treasury_proxy.py — duration-bucket mapping for the Harbor
bare-CUSIP Treasury ladder.

This is the regression spot-check that PR #26 (the May 2026 audit fix for
the ~20× duration underestimate on intermediate-maturity rungs) was
missing. Each bucket gets at least one in-bucket case plus a boundary
case so a future tweak that drifts the thresholds or flips the
closed-open boundary convention breaks here loudly.
"""
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))
sys.path.insert(0, str(ROOT))

from treasury_proxy import treasury_proxy, treasury_proxy_breakdown  # noqa: E402


class TestTreasuryProxyBuckets(unittest.TestCase):
    """as_of anchored at the user's most recent statement date so the
    live ladder is the implicit test universe — the in-bucket maturities
    below match actual CUSIPs in account 100-00003."""

    as_of = pd.Timestamp("2026-04-30")

    def _proxy(self, mat: str) -> str:
        return treasury_proxy(
            f"UNITED STATES TREASURY {mat} JJ 31", self.as_of
        )

    def test_under_one_year_maps_to_sgov(self) -> None:
        self.assertEqual(self._proxy("07/15/2026"), "SGOV")   # 0.21y
        self.assertEqual(self._proxy("09/30/2026"), "SGOV")   # 0.42y
        self.assertEqual(self._proxy("02/15/2027"), "SGOV")   # 0.79y

    def test_one_to_three_years_maps_to_scho(self) -> None:
        self.assertEqual(self._proxy("05/31/2027"), "SCHO")   # 1.08y
        self.assertEqual(self._proxy("01/31/2029"), "SCHO")   # 2.75y

    def test_boundary_three_years_maps_to_iei_not_scho(self) -> None:
        # Exactly 3.00y to maturity. Threshold for SCHO is `years < 3.0`,
        # so 3.00y falls through to IEI. Locks in the closed-open boundary
        # convention so a future change to `<=` is caught.
        self.assertEqual(self._proxy("04/30/2029"), "IEI")

    def test_three_to_seven_years_maps_to_iei(self) -> None:
        self.assertEqual(self._proxy("04/30/2031"), "IEI")    # 5.0y

    def test_seven_to_twelve_years_maps_to_ief(self) -> None:
        self.assertEqual(self._proxy("04/30/2034"), "IEF")    # 8.0y

    def test_over_twelve_years_maps_to_tlt(self) -> None:
        self.assertEqual(self._proxy("04/30/2040"), "TLT")    # 14.0y

    def test_matured_or_past_falls_back_to_sgov(self) -> None:
        # Negative years-to-maturity → SGOV fallback (a settled-but-not-
        # yet-removed row should park at the front of the curve, not get
        # extrapolated as ultra-long).
        self.assertEqual(self._proxy("01/01/2020"), "SGOV")

    def test_unparseable_description_falls_back_to_sgov(self) -> None:
        # Descriptions that don't expose a maturity (TIPS, STRIPS, agency
        # notes that don't follow the Harbor "TREASURY MM/DD/YYYY" convention)
        # quietly fall back to SGOV.
        self.assertEqual(treasury_proxy("FOO BAR", self.as_of), "SGOV")
        self.assertEqual(
            treasury_proxy("UNITED STATES TREASURY", self.as_of), "SGOV"
        )

    def test_non_string_input_falls_back_to_sgov(self) -> None:
        # NaN / numeric description (malformed positions row) → SGOV.
        self.assertEqual(treasury_proxy(math.nan, self.as_of), "SGOV")
        self.assertEqual(treasury_proxy(0.0, self.as_of), "SGOV")


class TestTreasuryProxyNewStatementFormat(unittest.TestCase):
    """Newer statement formats print
    ``UNITED STATES TREASURY NOTE DATED DATE <issue> <maturity>`` — the
    maturity is the LAST date in the description, not the token
    immediately after TREASURY. The prior regex anchored the date right
    after TREASURY, found the word NOTE instead, and silently fell back to
    SGOV for every rung (~20x duration understatement; every ladder rung
    mis-bucketed, SCHO/IEI rows missing from the Risk Contribution
    decomposition). The fixtures below reproduce that description
    geometry."""

    # Live snapshot date the repro ran at.
    as_of = pd.Timestamp("2026-05-31")

    def test_new_format_uses_last_date_as_maturity(self) -> None:
        # Issue date 01/31/2021 (~5y past), maturity 01/31/2028 (~1.7y).
        # Parsing the FIRST date would yield SGOV (matured); SCHO proves the
        # maturity (the LAST date) drives the bucket.
        self.assertEqual(
            treasury_proxy(
                "UNITED STATES TREASURY NOTE DATED DATE 01/31/2021 01/31/2028",
                self.as_of),
            "SCHO")

    def test_new_format_intermediate_rung_maps_to_iei(self) -> None:
        # Maturity 05/31/2029 ≈ 3.0y → IEI (3-7y). Restores one of the
        # SCHO/IEI rows the audit found missing.
        self.assertEqual(
            treasury_proxy(
                "UNITED STATES TREASURY NOTE DATED DATE 05/31/2022 05/31/2029",
                self.as_of),
            "IEI")

    def test_new_format_short_rung_maps_to_sgov(self) -> None:
        # Maturity 07/15/2026 ≈ 0.12y → SGOV legitimately (front of curve),
        # not via a parse failure.
        self.assertEqual(
            treasury_proxy(
                "UNITED STATES TREASURY NOTE DATED DATE 07/15/2023 07/15/2026",
                self.as_of),
            "SGOV")

    def test_treasury_without_maturity_still_falls_back_to_sgov(self) -> None:
        # Contains TREASURY but no parseable date token → SGOV (the genuine
        # fallback path, preserved).
        self.assertEqual(
            treasury_proxy("UNITED STATES TREASURY NOTE DATED DATE", self.as_of),
            "SGOV")


class TestTreasuryProxyBreakdown(unittest.TestCase):
    """The diagnostic the Risk Contribution tab renders so a future
    statement-format drift (the WSB-1 failure) is visible, not silent."""

    as_of = pd.Timestamp("2026-05-31")

    def test_breakdown_counts_proxies_and_flags_unparsed(self) -> None:
        descs = [
            # SCHO (1.7y), IEI (3.0y), SGOV (0.1y, legitimately short)
            "UNITED STATES TREASURY NOTE DATED DATE 01/31/2021 01/31/2028",
            "UNITED STATES TREASURY NOTE DATED DATE 05/31/2022 05/31/2029",
            "UNITED STATES TREASURY NOTE DATED DATE 07/15/2023 07/15/2026",
            # Money-market fund — not a Treasury note, SGOV, NOT a parse fail.
            "HarborORGAN TR II U S GOVT MONEY MARKET FD CL IM",
            # Looks like a Treasury but exposes no maturity → SGOV + unparsed.
            "UNITED STATES TREASURY NOTE DATED DATE",
        ]
        counts, n_unparsed = treasury_proxy_breakdown(descs, self.as_of)
        self.assertEqual(dict(counts), {"SCHO": 1, "IEI": 1, "SGOV": 3})
        self.assertEqual(n_unparsed, 1)

    def test_breakdown_clean_ladder_has_no_unparsed(self) -> None:
        descs = [
            "UNITED STATES TREASURY NOTE DATED DATE 01/31/2021 01/31/2028",
            "UNITED STATES TREASURY NOTE DATED DATE 05/31/2022 05/31/2029",
        ]
        counts, n_unparsed = treasury_proxy_breakdown(descs, self.as_of)
        self.assertEqual(n_unparsed, 0)
        self.assertEqual(dict(counts), {"SCHO": 1, "IEI": 1})

    def test_breakdown_ignores_non_string_rows(self) -> None:
        # A NaN description (malformed row) maps to SGOV but is NOT counted
        # as an unparsed Treasury — only TREASURY-bearing rows flag.
        counts, n_unparsed = treasury_proxy_breakdown(
            [math.nan, "UNITED STATES TREASURY NOTE DATED DATE 01/31/2021 01/31/2028"],
            self.as_of)
        self.assertEqual(n_unparsed, 0)
        self.assertEqual(dict(counts), {"SGOV": 1, "SCHO": 1})


if __name__ == "__main__":
    unittest.main()
