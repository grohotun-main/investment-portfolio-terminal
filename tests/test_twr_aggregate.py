"""Unit tests for the extracted portfolio-TWR re-aggregation helper.

`recompute_portfolio_twr` was lifted verbatim out of app.py (filter-parity
Slice 2b) into parsers/twr_aggregate.py so BOTH the Streamlit app and the
MERIDIAN terminal share ONE definition. It is a pure pandas re-aggregation of
an already-computed per-account monthly-TWR subset — used when a non-default
broker / history filter means the cached whole-book twr_portfolio.csv (which
pairs cross-account flows across the whole book) can't simply be subset.

These tests pin the aggregation math (deterministic, hand-checked), the
empty/NaN edge cases, and the module's import hygiene (no Streamlit/terminal/
app imports — the demo_overlay / risk_bundle precedent).
"""
import ast
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from parsers.twr_aggregate import recompute_portfolio_twr  # noqa: E402


class TestRecomputePortfolioTwr(unittest.TestCase):
    def _two_account_two_month(self) -> pd.DataFrame:
        # Two accounts (A, B), two months. NAV-weighted by prev_nav.
        #   2024-01: (0.10*100 + 0.20*100)/200 = 0.15 ; nav = 110+120 = 230
        #   2024-02: (0.05*110 + -0.10*120)/230 = -6.5/230 = -0.02826086957
        #   cum after 2 mo: 1.15 * (1 - 6.5/230) - 1 = 0.1175 (exact)
        return pd.DataFrame([
            {"account_id": "A", "month": "2024-01", "return_pct": 0.10,
             "prev_nav": 100.0, "nav": 110.0},
            {"account_id": "B", "month": "2024-01", "return_pct": 0.20,
             "prev_nav": 100.0, "nav": 120.0},
            {"account_id": "A", "month": "2024-02", "return_pct": 0.05,
             "prev_nav": 110.0, "nav": 115.5},
            {"account_id": "B", "month": "2024-02", "return_pct": -0.10,
             "prev_nav": 120.0, "nav": 108.0},
        ])

    def test_nav_weighted_aggregation(self):
        out = recompute_portfolio_twr(self._two_account_two_month())
        self.assertEqual(len(out), 2)
        first, last = out.iloc[0], out.iloc[1]
        self.assertAlmostEqual(first["return_pct"], 0.15, places=9)
        self.assertAlmostEqual(first["nav"], 230.0, places=9)
        self.assertAlmostEqual(last["return_pct"], -6.5 / 230.0, places=9)
        self.assertAlmostEqual(last["nav"], 223.5, places=9)

    def test_cumulative_and_wealth_columns(self):
        out = recompute_portfolio_twr(self._two_account_two_month())
        last = out.iloc[-1]
        self.assertAlmostEqual(last["cum_return"], 0.1175, places=9)
        self.assertAlmostEqual(last["wealth_index"], 1.1175, places=9)
        self.assertAlmostEqual(last["wealth_peak"], 1.15, places=9)     # Jan peak
        self.assertAlmostEqual(last["twr_dd_pct"], (1.1175 / 1.15 - 1.0) * 100.0,
                               places=6)
        # contract: the columns downstream consumers read exist
        for col in ("month", "statement_date", "return_pct", "cum_return",
                    "wealth_index", "wealth_peak", "twr_dd_pct"):
            self.assertIn(col, out.columns)

    def test_empty_input_returns_empty(self):
        out = recompute_portfolio_twr(pd.DataFrame(
            columns=["account_id", "month", "return_pct", "prev_nav", "nav"]))
        self.assertTrue(out.empty)

    def test_all_nan_returns_dropped_to_empty(self):
        df = self._two_account_two_month()
        df["return_pct"] = float("nan")
        out = recompute_portfolio_twr(df)
        self.assertTrue(out.empty)   # dropna(subset=["return_pct"]) removes all


class TestImportHygiene(unittest.TestCase):
    def test_no_streamlit_terminal_or_app_imports(self):
        src = (ROOT / "parsers" / "twr_aggregate.py").read_text(encoding="utf-8")
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        forbidden = {"streamlit", "terminal", "app"}
        self.assertEqual(mods & forbidden, set(),
                         f"twr_aggregate must not import {forbidden}; got "
                         f"{mods & forbidden}")


class TestPortfolioTwrHeadline(unittest.TestCase):
    def _frame(self):
        # returns [+20%, -10%] over two months.
        #   wealth: 1.20, 1.08 -> cum = 0.08 ; n = 2
        #   ann = 1.08**(12/2) - 1 = 1.08**6 - 1
        #   dd:  0, (1.08/1.20-1)*100 = -10.0 -> mdd = -10.0 at 2024-02
        import pandas as pd
        return pd.DataFrame({"month": ["2024-01", "2024-02"],
                             "return_pct": [0.20, -0.10]})

    def test_headline_values(self):
        from parsers.twr_aggregate import portfolio_twr_headline
        h = portfolio_twr_headline(self._frame())
        self.assertAlmostEqual(h.cum, 0.08, places=9)
        self.assertEqual(h.n, 2)
        self.assertAlmostEqual(h.ann, 1.08 ** (12 / 2) - 1, places=9)
        self.assertAlmostEqual(h.mdd, -10.0, places=9)
        self.assertEqual(h.start_month, "2024-01")
        self.assertEqual(h.mdd_month, "2024-02")

    def test_empty_frame(self):
        import pandas as pd
        from parsers.twr_aggregate import portfolio_twr_headline
        h = portfolio_twr_headline(pd.DataFrame())
        self.assertTrue(pd.isna(h.cum) and pd.isna(h.ann) and pd.isna(h.mdd))
        self.assertEqual(h.n, 0)
        self.assertEqual((h.start_month, h.mdd_month), ("—", "—"))

    def test_missing_return_pct_column(self):
        import pandas as pd
        from parsers.twr_aggregate import portfolio_twr_headline
        h = portfolio_twr_headline(pd.DataFrame({"month": ["2024-01"]}))
        self.assertEqual(h.n, 0)
        self.assertTrue(pd.isna(h.cum))


if __name__ == "__main__":
    unittest.main()
