"""Tests for the --holdings extension of parsers/fetch_dividends.py.

Offline: positions universe via a temp CSV + patch.object; network via a
patched fetch_dividends().
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

# _config lives in parsers/ — add it so `from _config import ...` resolves
# at module load time (same pattern as test_fetch_daily_prices.py).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "parsers"))

import parsers.fetch_dividends as fd  # noqa: E402


def _write_positions(tmp: Path, rows) -> Path:
    """rows: (statement_date, symbol, asset_class) tuples."""
    df = pd.DataFrame([{
        "statement_date": r[0], "symbol": r[1], "asset_class": r[2],
    } for r in rows])
    p = tmp / "positions.csv"
    df.to_csv(p, index=False)
    return p


def _hist(pairs):
    df = pd.DataFrame(list(pairs),
                      columns=["ex_dividend_date", "cash_amount"])
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"])
    return df


class TestCollectDividendUniverse(unittest.TestCase):
    def test_latest_month_symbols_except_options_cash(self) -> None:
        # Symbol-driven: fixed_income ETFs (SGOV at Harbor) belong in the
        # universe; options and cash sweeps never do.
        with TemporaryDirectory() as td:
            csv = _write_positions(Path(td), [
                ("2026-04-30", "OLD", "equity_stock"),    # stale month
                ("2026-05-31", "SPY", "equity_etf"),
                ("2026-05-31", "spy", "equity_etf"),      # dupe, case
                ("2026-05-31", "AAA", "equity_stock"),
                ("2026-05-31", "SGOV", "fixed_income"),   # Harbor-classed ETF
                ("2026-05-31", "GLD", "other"),           # Harbor-classed gold
                ("2026-05-31", "SPY", "option_put"),      # leg: excluded
                ("2026-05-31", "QJERQ", "cash"),          # sweep: excluded
                ("2026-05-31", "", "equity_stock"),       # blank symbol
                ("2026-05-31", "MF1", "mutual_fund"),
            ])
            with patch.object(fd, "POSITIONS_CSV", csv):
                syms = fd.collect_dividend_universe()
        self.assertEqual(syms, ["AAA", "GLD", "MF1", "SGOV", "SPY"])

    def test_missing_positions_csv_returns_empty(self) -> None:
        with patch.object(fd, "POSITIONS_CSV",
                          Path("Z:/nope/positions.csv")):
            self.assertEqual(fd.collect_dividend_universe(), [])


class TestMergeSpliced(unittest.TestCase):
    def test_prior_rows_before_effective_date_only(self) -> None:
        cur = _hist([("2026-03-20", 1.0)])
        prior = _hist([("2025-12-19", 0.9),   # before cut -> kept
                       ("2026-02-10", 0.95)])  # on/after cut -> dropped
        out = fd.merge_spliced(cur, prior, "2026-01-15")
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out["cash_amount"].sum()), 1.9,
                               places=6)
        # sorted ascending by ex date
        self.assertTrue(out["ex_dividend_date"].is_monotonic_increasing)

    def test_prior_with_date_objects_splices(self) -> None:
        # fetch_dividends() coerces ex_dividend_date to datetime.date; the
        # splice must compare those against the Timestamp cut (BNY <- BK
        # failed with "Cannot compare Timestamp with datetime.date").
        import datetime as _dt
        prior = pd.DataFrame({"ex_dividend_date": [_dt.date(2025, 12, 19),
                                                   _dt.date(2026, 2, 10)],
                              "cash_amount": [0.9, 0.95]})
        current = _hist([("2026-03-20", 1.5)])
        out = fd.merge_spliced(current, prior, "2026-01-15")
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out["cash_amount"].sum()), 2.4, places=6)

    def test_empty_prior_returns_current(self) -> None:
        cur = _hist([("2026-03-20", 1.0)])
        out = fd.merge_spliced(cur, _hist([]), "2026-01-15")
        self.assertEqual(len(out), 1)

    def test_current_empty_prior_rows_survive(self) -> None:
        # renamed ticker that hasn't paid under the new symbol yet
        cur = _hist([])
        prior = _hist([("2025-12-19", 0.9)])
        out = fd.merge_spliced(cur, prior, "2026-01-15")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out["cash_amount"].iloc[0]), 0.9,
                               places=6)

    def test_blank_effective_date_raises(self) -> None:
        cur = _hist([("2026-03-20", 1.0)])
        prior = _hist([("2025-12-19", 0.9)])
        with self.assertRaises(ValueError):
            fd.merge_spliced(cur, prior, "")


class TestHoldingsMode(unittest.TestCase):
    def _run(self, td: Path, fake_fetch, argv=None):
        csv = _write_positions(td, [
            ("2026-05-31", "SPY", "equity_etf"),
            ("2026-05-31", "AAA", "equity_stock"),
        ])
        with patch.object(fd, "POSITIONS_CSV", csv), \
             patch.object(fd, "DATA_DIR", td), \
             patch.object(fd, "fetch_dividends", side_effect=fake_fetch), \
             patch.object(fd, "fetch_splits",
                          side_effect=lambda t, s, *, allow_empty=True: pd.DataFrame(
                              columns=fd.SPLIT_COLUMNS)), \
             patch.object(fd, "_ticker_history", return_value={}):
            rc = fd.main(argv or ["--holdings", "--write", "--workers", "1"])
        return rc

    def test_writes_per_ticker_files_including_nonpayer_header(self) -> None:
        # AAA is a genuine non-payer: fetch_dividends returns empty frame
        # (allow_empty=True path). Must be written as a header-only CSV.
        def fake_fetch(ticker, since, *, allow_empty=False):
            if ticker == "SPY":
                return _hist([("2026-03-20", 1.5)])
            return pd.DataFrame()  # AAA: no dividends ever
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            rc = self._run(td, fake_fetch)
            self.assertEqual(rc, 0)
            spy = pd.read_csv(td / "dividends_spy.csv")
            self.assertEqual(len(spy), 1)
            aaa = pd.read_csv(td / "dividends_aaa.csv")
            self.assertEqual(len(aaa), 0)
            self.assertIn("ex_dividend_date", aaa.columns)
            self.assertIn("cash_amount", aaa.columns)

    def test_single_ticker_failure_is_nonfatal(self) -> None:
        # A real network/auth error must still be non-fatal-per-ticker.
        def fake_fetch(ticker, since, *, allow_empty=False):
            if ticker == "AAA":
                raise RuntimeError("boom")
            return _hist([("2026-03-20", 1.5)])
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            rc = self._run(td, fake_fetch)
            self.assertEqual(rc, 0)
            self.assertTrue((td / "dividends_spy.csv").exists())
            self.assertFalse((td / "dividends_aaa.csv").exists())

    def test_no_write_flag_writes_nothing(self) -> None:
        def fake_fetch(ticker, since, *, allow_empty=False):
            return _hist([("2026-03-20", 1.5)])
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            rc = self._run(td, fake_fetch,
                           argv=["--holdings", "--workers", "1"])
            self.assertEqual(rc, 0)
            self.assertEqual(list(td.glob("dividends_*.csv")), [])

    def test_splice_applied_via_ticker_history(self) -> None:
        def fake_fetch(ticker, since, *, allow_empty=False):
            if ticker == "SPY":
                return _hist([("2026-03-20", 1.5)])
            if ticker == "OLDSPY":
                return _hist([("2025-12-19", 0.9),    # pre-rename: kept
                              ("2026-02-10", 0.95)])  # post-cut: dropped
            return pd.DataFrame()
        history = {"SPY": [{"prior_symbol": "OLDSPY",
                            "effective_date": "2026-01-15"}]}
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            csv = _write_positions(td, [
                ("2026-05-31", "SPY", "equity_etf"),
            ])
            with patch.object(fd, "POSITIONS_CSV", csv), \
                 patch.object(fd, "DATA_DIR", td), \
                 patch.object(fd, "fetch_dividends",
                              side_effect=fake_fetch), \
                 patch.object(fd, "fetch_splits",
                              side_effect=lambda t, s, *, allow_empty=True:
                              pd.DataFrame(columns=fd.SPLIT_COLUMNS)), \
                 patch.object(fd, "_ticker_history",
                              return_value=history):
                rc = fd.main(["--holdings", "--write", "--workers", "1"])
            self.assertEqual(rc, 0)
            spy = pd.read_csv(td / "dividends_spy.csv")
            # splits.csv lands in the PATCHED data dir, never the real one
            self.assertTrue((td / "splits.csv").exists())
        self.assertEqual(len(spy), 2)  # 0.9 spliced in, 0.95 cut, 1.5 kept
        self.assertAlmostEqual(float(spy["cash_amount"].sum()), 2.4,
                               places=6)

    def test_all_tickers_failed_returns_error(self) -> None:
        def fake_fetch(ticker, since, *, allow_empty=False):
            raise RuntimeError("auth down")
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            rc = self._run(td, fake_fetch)
            self.assertEqual(rc, 1)
            self.assertEqual(list(td.glob("dividends_*.csv")), [])

    def test_empty_universe_returns_error(self) -> None:
        # Only cash sweeps and bare-CUSIP rungs (blank symbol) -> nothing
        # fetchable.
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            csv = _write_positions(td, [
                ("2026-05-31", "QJERQ", "cash"),
                ("2026-05-31", "", "fixed_income"),
            ])
            with patch.object(fd, "POSITIONS_CSV", csv), \
                 patch.object(fd, "DATA_DIR", td):
                rc = fd.main(["--holdings", "--write"])
            self.assertEqual(rc, 1)


# ---------------------------------------------------------------------------
# Mocked-HTTP tests for the allow_empty seam in fetch_dividends()
# ---------------------------------------------------------------------------

_ONE_ROW_PAYLOAD = (
    '{"results":[{"cash_amount":1.5,"currency":"USD",'
    '"declaration_date":"2026-03-01","dividend_type":"CD",'
    '"ex_dividend_date":"2026-03-20","frequency":4,'
    '"id":"D1","pay_date":"2026-03-25","record_date":"2026-03-21",'
    '"ticker":"SPY"}],'
    '"next_url":null,"status":"OK"}'
)

_EMPTY_PAYLOAD = '{"results":[],"next_url":null,"status":"OK"}'


class _Resp:
    """Minimal stand-in for a requests.Response."""
    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        pass

    def json(self):
        import json
        return json.loads(self._text)


class TestFetchDividendsAllowEmpty(unittest.TestCase):
    """Pin the allow_empty contract against mocked HTTP."""

    def _patch_http(self, payload: str):
        """Context manager that patches fd.requests.get + key/base helpers."""
        from unittest.mock import patch as _patch, MagicMock
        from datetime import date as _date

        def fake_get(url, params=None, timeout=None, **kw):
            return _Resp(payload)

        return (
            patch.object(fd.requests, "get", side_effect=fake_get),
            patch.object(fd, "get_massive_key", return_value="testkey"),
            patch.object(fd, "get_massive_base",
                         return_value="https://api.polygon.io"),
        )

    def test_empty_results_allow_empty_true_returns_empty_frame(self) -> None:
        """Non-payer: empty results + allow_empty=True → header-only frame."""
        from datetime import date
        patches = self._patch_http(_EMPTY_PAYLOAD)
        with patches[0], patches[1], patches[2]:
            df = fd.fetch_dividends("VRTX", date(2016, 6, 13),
                                    allow_empty=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        for col in fd.EMPTY_COLUMNS:
            self.assertIn(col, df.columns,
                          f"EMPTY_COLUMNS column {col!r} missing from result")

    def test_empty_results_default_raises(self) -> None:
        """Benchmark contract: empty results with default arg → RuntimeError."""
        from datetime import date
        patches = self._patch_http(_EMPTY_PAYLOAD)
        with patches[0], patches[1], patches[2]:
            with self.assertRaises(RuntimeError):
                fd.fetch_dividends("SPY", date(2016, 6, 13))

    def test_nonempty_results_parsed_correctly(self) -> None:
        """One-row response: columns present, ex_dividend_date is a date."""
        from datetime import date
        patches = self._patch_http(_ONE_ROW_PAYLOAD)
        with patches[0], patches[1], patches[2]:
            df = fd.fetch_dividends("SPY", date(2016, 6, 13))
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df["cash_amount"].iloc[0]), 1.5)
        # Date coercion: ex_dividend_date must be a datetime.date (not str)
        import datetime
        self.assertIsInstance(df["ex_dividend_date"].iloc[0], datetime.date)


def _splits(rows) -> pd.DataFrame:
    """rows: (execution_date, split_from, split_to)."""
    return pd.DataFrame(list(rows), columns=["execution_date", "split_from", "split_to"])


class TestPriceUniverseAndSplits(unittest.TestCase):
    """Total-return basis (spec 2026-08-22): --holdings must cover every
    column of the close matrices, not just held names, and write the
    splits overlay the adjustment needs (Polygon dividends are as-declared,
    bars are split-adjusted)."""

    def _env(self, td: Path, fake_fetch, fake_splits, history=None):
        csv = _write_positions(td, [("2026-05-31", "SPY", "equity_etf"),
                                    ("2026-05-31", "AAA", "equity_stock")])
        return [patch.object(fd, "POSITIONS_CSV", csv),
                patch.object(fd, "DATA_DIR", td),
                patch.object(fd, "fetch_dividends", side_effect=fake_fetch),
                patch.object(fd, "fetch_splits", side_effect=fake_splits),
                patch.object(fd, "_ticker_history", return_value=history or {})]

    def test_price_universe_reads_both_matrices(self) -> None:
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            pd.DataFrame({"symbol": ["QQQ", "qqq", "SPY"], "date": ["2026-01-02"] * 3,
                          "close": [1.0, 1.0, 1.0]}).to_csv(td / "daily_prices.csv", index=False)
            pd.DataFrame({"symbol": ["TLT"], "date": ["2026-01-02"],
                          "close": [1.0]}).to_csv(td / "long_history_prices.csv", index=False)
            self.assertEqual(fd.collect_price_universe(td), ["QQQ", "SPY", "TLT"])
            self.assertEqual(fd.collect_price_universe(td / "nope"), [])

    def test_holdings_universe_unions_positions_and_price_files(self) -> None:
        seen: list[str] = []

        def fake_fetch(ticker, since, *, allow_empty=False):
            seen.append(ticker)
            return pd.DataFrame()

        def fake_splits(ticker, since, *, allow_empty=True):
            return pd.DataFrame(columns=fd.SPLIT_COLUMNS)

        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            pd.DataFrame({"symbol": ["QQQ"], "date": ["2026-01-02"],
                          "close": [1.0]}).to_csv(td / "daily_prices.csv", index=False)
            patches = self._env(td, fake_fetch, fake_splits)
            for p in patches:
                p.start()
            try:
                rc = fd.main(["--holdings", "--write", "--workers", "1"])
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(seen), ["AAA", "QQQ", "SPY"])
            self.assertTrue((td / "dividends_qqq.csv").exists())
            # nothing split -> header-only splits.csv still written
            s = pd.read_csv(td / "splits.csv")
            self.assertEqual(len(s), 0)
            self.assertEqual(list(s.columns), ["symbol", "execution_date", "split_from", "split_to"])

    def test_splits_written_with_prior_symbol_rekeyed(self) -> None:
        def fake_fetch(ticker, since, *, allow_empty=False):
            return _hist([("2026-03-20", 1.5)]) if ticker == "SPY" else pd.DataFrame()

        def fake_splits(ticker, since, *, allow_empty=True):
            if ticker == "OLDSPY":
                return _splits([("2025-06-02", 1, 2),     # pre-rename: re-keyed to SPY
                                ("2026-02-02", 1, 3)])    # post-rename: dropped
            if ticker == "AAA":
                return _splits([("2024-07-15", 1, 10)])
            return pd.DataFrame(columns=fd.SPLIT_COLUMNS)

        history = {"SPY": [{"prior_symbol": "OLDSPY", "effective_date": "2026-01-15"}]}
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            patches = self._env(td, fake_fetch, fake_splits, history)
            for p in patches:
                p.start()
            try:
                rc = fd.main(["--holdings", "--write", "--workers", "1"])
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(rc, 0)
            s = pd.read_csv(td / "splits.csv")
        rows = {(r.symbol, str(r.execution_date), int(r.split_from), int(r.split_to))
                for r in s.itertuples()}
        self.assertEqual(rows, {("AAA", "2024-07-15", 1, 10),
                                ("SPY", "2025-06-02", 1, 2)})

    def test_split_fetch_failure_is_nonfatal(self) -> None:
        def fake_fetch(ticker, since, *, allow_empty=False):
            return _hist([("2026-03-20", 1.5)])

        def fake_splits(ticker, since, *, allow_empty=True):
            raise RuntimeError("splits endpoint down")

        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            patches = self._env(td, fake_fetch, fake_splits)
            for p in patches:
                p.start()
            try:
                rc = fd.main(["--holdings", "--write", "--workers", "1"])
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(rc, 0)
            self.assertTrue((td / "dividends_spy.csv").exists())
            self.assertEqual(len(pd.read_csv(td / "splits.csv")), 0)


def _delta(rows) -> pd.DataFrame:
    """Market-wide delta rows: (ticker, id, ex_dividend_date, cash_amount)."""
    df = pd.DataFrame(list(rows), columns=["ticker", "id", "ex_dividend_date", "cash_amount"])
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"]).dt.date
    return df


class TestIncrementalMode(unittest.TestCase):
    """Distributions S3 (spec 2026-08-23): --holdings runs a market-wide
    DELTA (ticker-less Polygon queries since the last run) when a fresh
    stamp exists, a full per-ticker sweep otherwise / on --full / every
    FULL_SWEEP_DAYS; tickers without a file are full-fetched inside a delta
    run; rows merge by Polygon id."""

    TODAY = pd.Timestamp("2026-08-23").date()

    def _patches(self, td: Path, *, fake_fetch, fake_splits, fake_since, fake_splits_since,
                 history=None, positions=(("2026-05-31", "SPY", "equity_etf"),
                                          ("2026-05-31", "AAA", "equity_stock"))):
        csv = _write_positions(td, list(positions))
        return [patch.object(fd, "POSITIONS_CSV", csv),
                patch.object(fd, "DATA_DIR", td),
                patch.object(fd, "_today", lambda: self.TODAY),
                patch.object(fd, "fetch_dividends", side_effect=fake_fetch),
                patch.object(fd, "fetch_splits", side_effect=fake_splits),
                patch.object(fd, "fetch_dividends_since", side_effect=fake_since),
                patch.object(fd, "fetch_splits_since", side_effect=fake_splits_since),
                patch.object(fd, "_ticker_history", return_value=history or {})]

    def _run(self, td: Path, argv=None, **kw) -> int:
        patches = self._patches(td, **kw)
        for p in patches:
            p.start()
        try:
            return fd.main(argv or ["--holdings", "--write", "--workers", "1"])
        finally:
            for p in patches:
                p.stop()

    @staticmethod
    def _meta(td: Path, last_run: str, last_full: str) -> None:
        import json
        (td / fd.META_NAME).write_text(json.dumps({
            "last_run_asof": last_run, "last_full_run": last_full, "mode": "full"}),
            encoding="utf-8")

    @staticmethod
    def _write_div_file(td: Path, sym: str, rows) -> Path:
        """rows: (id, ex_dividend_date, cash_amount); [] -> header-only."""
        p = td / f"dividends_{sym.lower()}.csv"
        if rows:
            pd.DataFrame([{"cash_amount": c, "currency": "USD", "declaration_date": "",
                           "dividend_type": "CD", "ex_dividend_date": ex, "frequency": 4,
                           "id": i, "pay_date": ex, "record_date": ex, "ticker": sym}
                          for i, ex, c in rows]).to_csv(p, index=False)
        else:
            pd.DataFrame(columns=fd.EMPTY_COLUMNS).to_csv(p, index=False)
        return p

    def test_no_stamp_runs_full_and_writes_stamp(self) -> None:
        import json
        fetched, since_calls = [], []

        def fake_fetch(ticker, since, *, allow_empty=False):
            fetched.append(ticker)
            return _hist([("2026-03-20", 1.5)])

        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            rc = self._run(td, fake_fetch=fake_fetch,
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=lambda s: since_calls.append(s) or pd.DataFrame(columns=fd.EMPTY_COLUMNS),
                           fake_splits_since=lambda s: pd.DataFrame(columns=["ticker", *fd.SPLIT_COLUMNS]))
            self.assertEqual(rc, 0)
            meta = json.loads((td / fd.META_NAME).read_text(encoding="utf-8"))
        self.assertEqual(sorted(fetched), ["AAA", "SPY"])
        self.assertEqual(since_calls, [])
        self.assertEqual(meta["mode"], "full")
        self.assertEqual(meta["last_full_run"], "2026-08-23")
        self.assertEqual(meta["last_run_asof"], "2026-08-23")

    def test_fresh_stamp_runs_delta_and_merges_by_id(self) -> None:
        import json
        fetched, since_calls = [], []

        def fake_fetch(ticker, since, *, allow_empty=False):
            fetched.append(ticker)
            return pd.DataFrame()

        def fake_since(since):
            since_calls.append(since)
            return _delta([("SPY", "a2", "2026-06-20", 1.6),      # new event
                           ("SPY", "a1", "2026-03-20", 1.55),     # revised amount, same id
                           ("ZZZ", "z1", "2026-06-20", 9.9)])     # not in the universe

        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            self._write_div_file(td, "SPY", [("a1", "2026-03-20", 1.5)])
            aaa = self._write_div_file(td, "AAA", [])
            aaa_bytes = aaa.read_bytes()
            self._meta(td, "2026-08-16", "2026-08-10")
            rc = self._run(td, fake_fetch=fake_fetch,
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=fake_since,
                           fake_splits_since=lambda s: pd.DataFrame(columns=["ticker", *fd.SPLIT_COLUMNS]))
            self.assertEqual(rc, 0)
            spy = pd.read_csv(td / "dividends_spy.csv")
            self.assertEqual(aaa.read_bytes(), aaa_bytes)            # untouched
            self.assertFalse((td / "dividends_zzz.csv").exists())
            meta = json.loads((td / fd.META_NAME).read_text(encoding="utf-8"))
        self.assertEqual(fetched, [])                                  # no per-ticker calls
        self.assertEqual(since_calls, [pd.Timestamp("2026-08-13").date()])   # last run - 3d overlap
        self.assertEqual(len(spy), 2)
        by_id = spy.set_index("id")
        self.assertAlmostEqual(float(by_id.loc["a1", "cash_amount"]), 1.55, places=6)
        self.assertAlmostEqual(float(by_id.loc["a2", "cash_amount"]), 1.6, places=6)
        self.assertEqual(spy["ex_dividend_date"].tolist(), ["2026-03-20", "2026-06-20"])
        self.assertEqual(meta["mode"], "incremental")
        self.assertEqual(meta["last_run_asof"], "2026-08-23")
        self.assertEqual(meta["last_full_run"], "2026-08-10")

    def test_missing_file_is_full_fetched_inside_a_delta_run(self) -> None:
        fetched = []

        def fake_fetch(ticker, since, *, allow_empty=False):
            fetched.append(ticker)
            return _hist([("2026-03-20", 0.4)])

        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            self._write_div_file(td, "SPY", [("a1", "2026-03-20", 1.5)])
            self._meta(td, "2026-08-16", "2026-08-10")
            rc = self._run(td, fake_fetch=fake_fetch,
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=lambda s: pd.DataFrame(columns=fd.EMPTY_COLUMNS),
                           fake_splits_since=lambda s: pd.DataFrame(columns=["ticker", *fd.SPLIT_COLUMNS]))
            self.assertEqual(rc, 0)
            self.assertTrue((td / "dividends_aaa.csv").exists())
            self.assertEqual(len(pd.read_csv(td / "dividends_aaa.csv")), 1)
        self.assertEqual(fetched, ["AAA"])

    def test_full_flag_and_stale_sweep_force_full(self) -> None:
        def fake_fetch(ticker, since, *, allow_empty=False):
            return _hist([("2026-03-20", 1.5)])

        def boom(since):
            raise AssertionError("delta must not run in full mode")

        common = dict(fake_fetch=fake_fetch,
                      fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                      fake_since=boom,
                      fake_splits_since=boom)
        with TemporaryDirectory() as td_s:           # stale sweep: 91 days since the last full
            td = Path(td_s)
            self._write_div_file(td, "SPY", []); self._write_div_file(td, "AAA", [])
            self._meta(td, "2026-08-16", "2026-05-24")
            self.assertEqual(self._run(td, **common), 0)
        with TemporaryDirectory() as td_s:           # explicit --full on a fresh stamp
            td = Path(td_s)
            self._write_div_file(td, "SPY", []); self._write_div_file(td, "AAA", [])
            self._meta(td, "2026-08-16", "2026-08-10")
            self.assertEqual(self._run(td, argv=["--holdings", "--write", "--workers", "1", "--full"],
                                       **common), 0)

    def test_splits_delta_merged_and_deduped(self) -> None:
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            self._write_div_file(td, "SPY", []); self._write_div_file(td, "AAA", [])
            pd.DataFrame([{"symbol": "SPY", "execution_date": "2024-01-01",
                           "split_from": 1, "split_to": 2}]).to_csv(td / "splits.csv", index=False)
            self._meta(td, "2026-08-16", "2026-08-10")

            def fake_splits_since(since):
                return pd.DataFrame([
                    {"ticker": "SPY", "execution_date": pd.Timestamp("2024-01-01").date(), "split_from": 1, "split_to": 2},   # dup
                    {"ticker": "AAA", "execution_date": pd.Timestamp("2026-08-20").date(), "split_from": 1, "split_to": 10},
                    {"ticker": "ZZZ", "execution_date": pd.Timestamp("2026-08-20").date(), "split_from": 1, "split_to": 3},
                ])

            rc = self._run(td, fake_fetch=lambda t, s, *, allow_empty=False: pd.DataFrame(),
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=lambda s: pd.DataFrame(columns=fd.EMPTY_COLUMNS),
                           fake_splits_since=fake_splits_since)
            self.assertEqual(rc, 0)
            s = pd.read_csv(td / "splits.csv")
        rows = {(r.symbol, str(r.execution_date), int(r.split_from), int(r.split_to)) for r in s.itertuples()}
        self.assertEqual(rows, {("SPY", "2024-01-01", 1, 2), ("AAA", "2026-08-20", 1, 10)})

    def test_identical_delta_rows_do_not_rewrite_the_file(self) -> None:
        # Declared-future events are already on file on the next run; their
        # re-delivery must not touch the file (mtime = AI cache key input).
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            spy = self._write_div_file(td, "SPY", [("a1", "2026-09-18", 1.7)])
            self._write_div_file(td, "AAA", [])
            before = (spy.read_bytes(), spy.stat().st_mtime_ns)
            self._meta(td, "2026-08-16", "2026-08-10")
            rc = self._run(td, fake_fetch=lambda t, s, *, allow_empty=False: pd.DataFrame(),
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=lambda s: _delta([("SPY", "a1", "2026-09-18", 1.7)]),
                           fake_splits_since=lambda s: pd.DataFrame(columns=["ticker", *fd.SPLIT_COLUMNS]))
            self.assertEqual(rc, 0)
            self.assertEqual((spy.read_bytes(), spy.stat().st_mtime_ns), before)

    def test_stamp_with_utf8_bom_still_reads(self) -> None:
        # A stamp written from PowerShell (Out-File -Encoding utf8) carries a
        # BOM; it must still select the delta, not silently force a sweep.
        since_calls = []
        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            self._write_div_file(td, "SPY", []); self._write_div_file(td, "AAA", [])
            (td / fd.META_NAME).write_bytes(
                b"\xef\xbb\xbf" + b'{"last_run_asof": "2026-08-16", "last_full_run": "2026-08-10", "mode": "full"}')
            rc = self._run(td, fake_fetch=lambda t, s, *, allow_empty=False: pd.DataFrame(),
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=lambda s: since_calls.append(s) or pd.DataFrame(columns=fd.EMPTY_COLUMNS),
                           fake_splits_since=lambda s: pd.DataFrame(columns=["ticker", *fd.SPLIT_COLUMNS]))
            self.assertEqual(rc, 0)
        self.assertEqual(len(since_calls), 1)

    def test_delta_failure_falls_back_to_full(self) -> None:
        import json
        fetched = []

        def fake_fetch(ticker, since, *, allow_empty=False):
            fetched.append(ticker)
            return pd.DataFrame()

        def boom(since):
            raise RuntimeError("polygon 503")

        with TemporaryDirectory() as td_s:
            td = Path(td_s)
            self._write_div_file(td, "SPY", []); self._write_div_file(td, "AAA", [])
            self._meta(td, "2026-08-16", "2026-08-10")
            rc = self._run(td, fake_fetch=fake_fetch,
                           fake_splits=lambda t, s, *, allow_empty=True: pd.DataFrame(columns=fd.SPLIT_COLUMNS),
                           fake_since=boom,
                           fake_splits_since=lambda s: pd.DataFrame(columns=["ticker", *fd.SPLIT_COLUMNS]))
            self.assertEqual(rc, 0)
            meta = json.loads((td / fd.META_NAME).read_text(encoding="utf-8"))
        self.assertEqual(sorted(fetched), ["AAA", "SPY"])
        self.assertEqual(meta["mode"], "full")


if __name__ == "__main__":
    unittest.main()
