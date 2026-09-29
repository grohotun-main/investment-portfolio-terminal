"""Tests for parsers/fetch_ff_factors.py — the Ken French factor fetcher.

The French library serves zip files each holding ONE CSV whose layout is:
text preamble, a monthly block (rows keyed YYYYMM), an annual block (rows
keyed YYYY), and a copyright footer. Only the monthly rows are data we
want; -99.99 / -999 are missing-data sentinels. Values are percent.
"""
import inspect
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import fetch_ff_factors as ff  # noqa: E402

# Replicates the real F-F_Research_Data_5_Factors_2x3 layout: preamble,
# header, monthly rows, annual section, footer. One sentinel cell (-99.99).
FF5_TEXT = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202604 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
202601,2.50,-1.10,0.80,0.30,-0.20,0.40
202602,-3.10,0.90,1.20,-0.50,0.10,0.41
202603,1.75,0.20,-0.60,0.15,0.05,0.39
202604,0.95,-99.99,0.30,0.10,0.00,0.40

 Annual Factors: January-December

,Mkt-RF,SMB,HML,RMW,CMA,RF
2025,12.30,-2.10,3.40,1.10,0.50,5.10

Copyright 2026 Kenneth R. French
"""

# Momentum file: the REAL file's header is ",Mom   " with trailing spaces —
# built via concatenation so editors can't silently strip them — and starts
# 1927 in reality, so its month range OVERLAPS but is not equal to the FF5
# file's.
MOM_TEXT = (
    "Missing data are indicated by -99.99 or -999.\n"
    "\n"
    ",Mom   \n"
    "202512,1.10\n"
    "202601,0.60\n"
    "202602,-0.40\n"
    "202603,2.10\n"
    "202604,0.85\n"
    "\n"
    " Annual Factors: January-December \n"
    "\n"
    ",Mom\n"
    "2025,4.20\n"
    "\n"
    "Copyright 2026 Kenneth R. French\n"
)


def _zip_bytes(member_name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member_name, text)
    return buf.getvalue()


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class TestParseFrenchCsv(unittest.TestCase):
    def test_monthly_rows_only(self) -> None:
        # Preamble, annual section, and footer must all be excluded.
        df = ff.parse_french_csv(FF5_TEXT)
        self.assertEqual(list(df["month"]),
                         ["2026-01", "2026-02", "2026-03", "2026-04"])

    def test_percent_to_decimal(self) -> None:
        df = ff.parse_french_csv(FF5_TEXT)
        row = df[df["month"] == "2026-01"].iloc[0]
        self.assertAlmostEqual(row["mkt_rf"], 0.0250, places=6)
        self.assertAlmostEqual(row["rf"], 0.0040, places=6)

    def test_columns_normalized(self) -> None:
        # Mkt-RF -> mkt_rf; the momentum header's trailing spaces stripped.
        df5 = ff.parse_french_csv(FF5_TEXT)
        self.assertEqual(list(df5.columns),
                         ["month", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
        dfm = ff.parse_french_csv(MOM_TEXT)
        self.assertEqual(list(dfm.columns), ["month", "mom"])

    def test_missing_sentinel_becomes_nan(self) -> None:
        df = ff.parse_french_csv(FF5_TEXT)
        row = df[df["month"] == "2026-04"].iloc[0]
        self.assertTrue(row[["smb"]].isna().all())

    def test_no_monthly_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            ff.parse_french_csv("just a preamble\nno data here\n")

    def test_header_trailing_whitespace_stripped(self) -> None:
        # The real momentum file's header is ",Mom   " — guard that the
        # fixture really carries the trailing spaces AND that parsing
        # strips them.
        header_line = MOM_TEXT.splitlines()[2]
        self.assertEqual(header_line, ",Mom   ")
        self.assertEqual(list(ff.parse_french_csv(MOM_TEXT).columns),
                         ["month", "mom"])


class TestFetchAndJoin(unittest.TestCase):
    def _patched_fetch(self):
        """Run fetch_ff_factors with requests.get returning in-memory zips."""
        responses = {
            ff.FF5_URL: _FakeResp(_zip_bytes("ff5.csv", FF5_TEXT)),
            ff.MOM_URL: _FakeResp(_zip_bytes("mom.csv", MOM_TEXT)),
        }
        calls: list[str] = []

        def fake_get(url, **_k):
            calls.append(url)
            return responses[url]

        with mock.patch.object(ff.requests, "get", side_effect=fake_get):
            df = ff.fetch_ff_factors()
        return df, calls

    def test_join_is_month_intersection_with_sentinels_dropped(self) -> None:
        # FF5 has 2026-01..04, Mom has 2025-12..2026-04. Intersection is
        # 2026-01..04, minus 2026-04 whose FF5 smb is the -99.99 sentinel.
        df, _ = self._patched_fetch()
        self.assertEqual(list(df["month"]), ["2026-01", "2026-02", "2026-03"])

    def test_output_column_order(self) -> None:
        df, _ = self._patched_fetch()
        self.assertEqual(list(df.columns), ff.EXPECTED_COLUMNS)

    def test_both_urls_fetched_once(self) -> None:
        _, calls = self._patched_fetch()
        self.assertEqual(sorted(calls), sorted([ff.FF5_URL, ff.MOM_URL]))

    def test_momentum_values_joined(self) -> None:
        df, _ = self._patched_fetch()
        row = df[df["month"] == "2026-03"].iloc[0]
        self.assertAlmostEqual(row["mom"], 0.0210, places=6)

    def test_multi_member_zip_raises(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.csv", FF5_TEXT)
            zf.writestr("b.csv", FF5_TEXT)
        with mock.patch.object(ff.requests, "get",
                               return_value=_FakeResp(buf.getvalue())):
            with self.assertRaises(ValueError):
                ff.fetch_ff_factors()


class TestFailFast(unittest.TestCase):
    """Same fail-fast contract as the FRED fetcher: a blocked host must not
    hang the 'Refresh all data' run."""

    def test_retry_budget_is_small(self) -> None:
        self.assertLessEqual(ff.MAX_ATTEMPTS, 3)

    def test_default_read_timeout_is_short(self) -> None:
        default_timeout = inspect.signature(
            ff.fetch_ff_factors).parameters["timeout"].default
        self.assertLessEqual(default_timeout, 30)

    def test_gives_up_after_exactly_max_attempts(self) -> None:
        calls = {"n": 0}

        def fake_get(*_a, **_k):
            calls["n"] += 1
            raise requests.Timeout("read timed out")

        with mock.patch.object(ff.requests, "get", side_effect=fake_get), \
                mock.patch.object(ff.time, "sleep", lambda *_a, **_k: None):
            with self.assertRaises(requests.RequestException):
                ff.fetch_ff_factors()
        # First URL exhausts the budget and the error propagates.
        self.assertEqual(calls["n"], ff.MAX_ATTEMPTS)


# Daily-file replicas: 8-digit YYYYMMDD rows, NO annual section. Momentum
# daily header carries trailing spaces like the monthly one — built via
# concatenation so editors can't strip them (assertion below guards it).
FF5_DAILY_TEXT = (
    "This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202604 "
    "CRSP database.\n"
    "\n"
    ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
    "20260105,0.55,-0.20,0.10,0.05,-0.02,0.018\n"
    "20260106,-0.80,0.15,0.30,-0.10,0.04,0.018\n"
    "20260107,0.25,0.05,-0.15,0.02,0.01,0.018\n"
    "20260108,0.40,-99.99,0.05,0.01,0.00,0.018\n"
    "\n"
    "Copyright 2026 Kenneth R. French\n"
)
MOM_DAILY_TEXT = (
    "Missing data are indicated by -99.99 or -999.\n"
    "\n"
    ",Mom   \n"
    "20260102,0.30\n"
    "20260105,0.12\n"
    "20260106,-0.08\n"
    "20260107,0.45\n"
    "20260108,0.20\n"
    "\n"
    "Copyright 2026 Kenneth R. French\n"
)


class TestParseFrenchCsvDaily(unittest.TestCase):
    def test_daily_rows_parsed_with_date_column(self) -> None:
        df = ff.parse_french_csv(FF5_DAILY_TEXT, date_digits=8)
        self.assertEqual(list(df.columns),
                         ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
        self.assertEqual(list(df["date"]),
                         ["2026-01-05", "2026-01-06",
                          "2026-01-07", "2026-01-08"])

    def test_daily_percent_to_decimal_and_sentinel(self) -> None:
        df = ff.parse_french_csv(FF5_DAILY_TEXT, date_digits=8)
        self.assertAlmostEqual(
            df[df["date"] == "2026-01-05"].iloc[0]["mkt_rf"], 0.0055,
            places=8)
        self.assertTrue(
            df[df["date"] == "2026-01-08"]["smb"].isna().all())

    def test_daily_regex_does_not_match_monthly_rows_and_vice_versa(self) -> None:
        # 6-digit parse on daily text finds nothing (YYYYMMDD rows have no
        # comma after 6 digits) and must raise the layout canary; same for
        # 8-digit parse on monthly text.
        with self.assertRaises(ValueError):
            ff.parse_french_csv(FF5_DAILY_TEXT, date_digits=6)
        with self.assertRaises(ValueError):
            ff.parse_french_csv(FF5_TEXT, date_digits=8)

    def test_mom_daily_header_trailing_spaces(self) -> None:
        self.assertEqual(MOM_DAILY_TEXT.splitlines()[2], ",Mom   ")
        df = ff.parse_french_csv(MOM_DAILY_TEXT, date_digits=8)
        self.assertEqual(list(df.columns), ["date", "mom"])

    def test_monthly_default_unchanged(self) -> None:
        df = ff.parse_french_csv(FF5_TEXT)
        self.assertEqual(list(df["month"]),
                         ["2026-01", "2026-02", "2026-03", "2026-04"])


class TestFetchDailyAndJoin(unittest.TestCase):
    def _patched_fetch(self):
        responses = {
            ff.FF5_DAILY_URL: _FakeResp(
                _zip_bytes("ff5d.csv", FF5_DAILY_TEXT)),
            ff.MOM_DAILY_URL: _FakeResp(
                _zip_bytes("momd.csv", MOM_DAILY_TEXT)),
        }
        with mock.patch.object(ff.requests, "get",
                               side_effect=lambda url, **_k: responses[url]):
            return ff.fetch_ff_factors_daily()

    def test_join_is_date_intersection_with_sentinels_dropped(self) -> None:
        # FF5 daily has 01-05..01-08, Mom daily 01-02..01-08. Intersection
        # 01-05..01-08, minus 01-08 whose smb is the sentinel.
        df = self._patched_fetch()
        self.assertEqual(list(df["date"]),
                         ["2026-01-05", "2026-01-06", "2026-01-07"])

    def test_output_column_order(self) -> None:
        df = self._patched_fetch()
        self.assertEqual(list(df.columns), ff.EXPECTED_COLUMNS_DAILY)

    def test_momentum_joined_decimal(self) -> None:
        df = self._patched_fetch()
        self.assertAlmostEqual(
            df[df["date"] == "2026-01-07"].iloc[0]["mom"], 0.0045, places=8)


class TestMainWritesBothFiles(unittest.TestCase):
    def _run_main(self, tmp: Path, fail_daily: bool = False) -> int:
        responses = {
            ff.FF5_URL: _FakeResp(_zip_bytes("ff5.csv", FF5_TEXT)),
            ff.MOM_URL: _FakeResp(_zip_bytes("mom.csv", MOM_TEXT)),
            ff.FF5_DAILY_URL: _FakeResp(
                _zip_bytes("ff5d.csv", FF5_DAILY_TEXT)),
            ff.MOM_DAILY_URL: _FakeResp(
                _zip_bytes("momd.csv", MOM_DAILY_TEXT)),
        }

        def fake_get(url, **_k):
            if fail_daily and url == ff.FF5_DAILY_URL:
                raise requests.ConnectionError("daily endpoint down")
            return responses[url]

        with mock.patch.object(ff.requests, "get", side_effect=fake_get), \
                mock.patch.object(ff.time, "sleep", lambda *_a, **_k: None), \
                mock.patch.object(ff, "DATA", tmp), \
                mock.patch.object(ff, "OUT_CSV",
                                  tmp / "ff_factors_monthly.csv"), \
                mock.patch.object(ff, "OUT_CSV_DAILY",
                                  tmp / "ff_factors_daily.csv"), \
                mock.patch.object(sys, "argv",
                                  ["fetch_ff_factors.py", "--write"]):
            return ff.main()

    def test_write_emits_both_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rc = self._run_main(tmp)
            self.assertEqual(rc, 0)
            self.assertTrue((tmp / "ff_factors_monthly.csv").exists())
            self.assertTrue((tmp / "ff_factors_daily.csv").exists())

    def test_any_fetch_failure_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rc = self._run_main(tmp, fail_daily=True)
            self.assertEqual(rc, 1)
            self.assertFalse((tmp / "ff_factors_monthly.csv").exists())
            self.assertFalse((tmp / "ff_factors_daily.csv").exists())


if __name__ == "__main__":
    unittest.main()
