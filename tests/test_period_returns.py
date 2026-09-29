import unittest
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "parsers"))
from period_returns import window_returns, WINDOWS


def _comp(n_months, port_m=0.01, bench_m=0.005, end="2026-04-30"):
    dates = pd.date_range(end=end, periods=n_months, freq="ME")
    return pd.DataFrame({"statement_date": dates,
                         "port_return": [port_m] * n_months,
                         "bench_return": [bench_m] * n_months})


class TestWindowReturns(unittest.TestCase):
    def test_returns_one_row_per_window(self):
        rows = window_returns(_comp(28))
        self.assertEqual([r["key"] for r in rows], [w[0] for w in WINDOWS])

    def test_itd_cumulative_matches_compounding(self):
        rows = {r["key"]: r for r in window_returns(_comp(24, 0.01, 0.005))}
        itd = rows["itd"]
        self.assertTrue(itd["available"])
        self.assertTrue(itd["annualized"])
        # 24 months of +1%/mo -> cum (1.01^24-1); annualized == monthly comp to yr
        cum = 1.01 ** 24 - 1.0
        self.assertAlmostEqual(itd["port"], (1.0 + cum) ** (12.0 / 24) - 1.0, places=9)

    def test_1y_is_cumulative_last_12(self):
        rows = {r["key"]: r for r in window_returns(_comp(28, 0.01, 0.005))}
        oney = rows["1y"]
        self.assertTrue(oney["available"])
        self.assertFalse(oney["annualized"])
        self.assertAlmostEqual(oney["port"], 1.01 ** 12 - 1.0, places=9)
        self.assertAlmostEqual(oney["spread"], oney["port"] - oney["bench"], places=12)

    def test_5y_unavailable_when_only_28_months(self):
        rows = {r["key"]: r for r in window_returns(_comp(28))}
        self.assertFalse(rows["5y"]["available"])
        self.assertIsNone(rows["5y"]["port"])
        self.assertEqual(rows["5y"]["requested_months"], 60)
        self.assertFalse(rows["3y"]["available"])

    def test_3y_available_when_40_months(self):
        rows = {r["key"]: r for r in window_returns(_comp(40))}
        self.assertTrue(rows["3y"]["available"])
        self.assertFalse(rows["5y"]["available"])

    def test_ytd_counts_current_year_months(self):
        # end 2026-04-30 -> YTD = Jan..Apr 2026 = 4 months
        rows = {r["key"]: r for r in window_returns(_comp(28, 0.01, 0.005))}
        self.assertTrue(rows["ytd"]["available"])
        self.assertEqual(rows["ytd"]["n_months"], 4)
        self.assertAlmostEqual(rows["ytd"]["port"], 1.01 ** 4 - 1.0, places=9)

    def test_as_of_bounds_window_above(self):
        # comp extends to 2026-04-30 (28 months); as_of pins well before
        # that. Every window must exclude months AFTER as_of, not just
        # honor the lower cutoff.
        port_m, bench_m = 0.01, 0.005
        comp = _comp(28, port_m, bench_m)
        as_of = pd.Timestamp("2025-06-30")
        rows = {r["key"]: r for r in window_returns(comp, as_of="2025-06-30")}

        expected_itd_n = int((comp["statement_date"] <= as_of).sum())
        self.assertEqual(rows["itd"]["n_months"], expected_itd_n)

        oney = rows["1y"]
        self.assertTrue(oney["available"])
        self.assertAlmostEqual(oney["port"], (1.0 + port_m) ** 12 - 1.0, places=9)

        expected_ytd_n = int(((comp["statement_date"].dt.year == 2025) &
                               (comp["statement_date"] <= as_of)).sum())
        self.assertEqual(expected_ytd_n, 6)
        self.assertEqual(rows["ytd"]["n_months"], expected_ytd_n)

    def test_empty_comp_all_unavailable(self):
        rows = window_returns(pd.DataFrame(columns=["statement_date", "port_return", "bench_return"]))
        self.assertTrue(all(not r["available"] for r in rows))

    def test_vol_annualized_from_monthly_std(self):
        dates = pd.date_range(end="2026-04-30", periods=24, freq="ME")
        pr = [0.02, -0.01] * 12                      # alternating -> nonzero std
        comp = pd.DataFrame({"statement_date": dates, "port_return": pr,
                             "bench_return": [0.005] * 24})
        rows = {r["key"]: r for r in window_returns(comp)}
        expected = float(pd.Series(pr).std(ddof=1)) * (12 ** 0.5)
        self.assertAlmostEqual(rows["itd"]["port_vol"], expected, places=9)
        self.assertAlmostEqual(rows["itd"]["bench_vol"], 0.0, places=12)  # constant

    def test_vol_none_when_under_two_obs(self):
        dates = pd.date_range(end="2026-01-31", periods=13, freq="ME")   # end Jan
        comp = pd.DataFrame({"statement_date": dates, "port_return": [0.01] * 13,
                             "bench_return": [0.005] * 13})
        rows = {r["key"]: r for r in window_returns(comp)}
        self.assertEqual(rows["ytd"]["n_months"], 1)                     # Jan only
        self.assertIsNone(rows["ytd"]["port_vol"])
        self.assertIsNone(rows["ytd"]["bench_vol"])

    def test_vol_none_on_unavailable_window(self):
        rows = {r["key"]: r for r in window_returns(_comp(28))}
        self.assertFalse(rows["5y"]["available"])
        self.assertIsNone(rows["5y"]["port_vol"])
        self.assertIsNone(rows["5y"]["bench_vol"])


if __name__ == "__main__":
    unittest.main()
