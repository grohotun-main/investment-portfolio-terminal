"""
A3 — Phase 1A audit: locks the boundary between
  (a) display-side correlation functions, which use LOG returns for
      numerical stability on heterogeneous-vol mixtures (e.g. SGOV at
      ±0.01% sitting next to TLT at ±2%), and
  (b) the marginal-risk decomposition pipeline, which uses SIMPLE-return
      pct_change covariance for the marginal contribution algebra.

Mixing the two bases — using log-return correlations alongside simple-
return stdevs in the same risk decomposition — would produce inconsistent
marginal-risk attribution. The two are correctly kept separate today (the
log-return correlation functions are display-only; the risk-decomp
pipeline builds covariance independently and never reaches for them).
This test locks that invariant by source inspection so a future change
that introduces a cross-call breaks here loudly.
"""
import inspect
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))
sys.path.insert(0, str(ROOT))

import risk_metrics as rm  # noqa: E402


_LOG_RETURN_CORR_FN_NAMES = (
    "compute_correlation_matrix",
    "compute_rolling_pair_correlations",
    "compute_rolling_avg_pairwise_correlation",
    "compute_conditional_correlation_matrix",
)


class TestLogVsSimpleReturnsBoundary(unittest.TestCase):
    def _assert_does_not_call_log_corr(self, fn) -> None:
        # Match a function CALL — name followed by `(` after optional
        # whitespace — not a mere mention in a docstring or comment. This
        # keeps the cross-reference notes in docstrings legal while still
        # catching an actual cross-pipeline call.
        src = inspect.getsource(fn)
        for name in _LOG_RETURN_CORR_FN_NAMES:
            pattern = rf"\b{re.escape(name)}\s*\("
            match = re.search(pattern, src)
            self.assertIsNone(
                match,
                f"{fn.__name__} must not call {name} — those functions "
                "compute LOG-return correlations for display stability, "
                "while marginal-risk math is on SIMPLE-return covariance. "
                "Mixing bases produces inconsistent risk attribution. If "
                "you need correlations on the covariance basis, derive "
                "them in place: C[i,j] = Σ[i,j] / sqrt(Σ[i,i] · Σ[j,j])."
            )

    def test_compute_risk_contributions_does_not_call_log_corr(self) -> None:
        self._assert_does_not_call_log_corr(rm.compute_risk_contributions)

    def test_compute_downside_risk_contributions_does_not_call_log_corr(self) -> None:
        self._assert_does_not_call_log_corr(rm.compute_downside_risk_contributions)

    def test_compute_es_contributions_does_not_call_log_corr(self) -> None:
        self._assert_does_not_call_log_corr(rm.compute_es_contributions)

    def test_estimate_covariance_does_not_call_log_corr(self) -> None:
        # The covariance estimator itself feeds the risk-contribution
        # pipeline; if it ever started returning log-corr-derived values
        # the downstream math would silently drift. Lock the same invariant.
        self._assert_does_not_call_log_corr(rm.estimate_covariance)


if __name__ == "__main__":
    unittest.main()
