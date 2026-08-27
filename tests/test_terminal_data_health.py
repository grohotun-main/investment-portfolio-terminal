# tests/test_terminal_data_health.py
import dataclasses
import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import health_service as hes
from data_health import (build_health_report, format_health_headline,
                         health_rows_to_table)
from reconcile_holdings import load_allowlist


def _recompute_report(frames):
    """Rebuild the HealthReport exactly as app.py:1484 does — the parity ref."""
    s = pd.read_csv(FIXTURE / "summaries.csv", parse_dates=["statement_date"])
    return build_health_report(
        frames.positions, s,
        today=pd.Timestamp.today().normalize().date(),
        label_by_account=hs.ACCOUNT_DISPLAY,
        allowlist=load_allowlist())


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = hes.build_health_view(cls.frames)

    def test_contract_keys(self):
        self.assertEqual(set(self.view.keys()),
                         {"meta", "headline", "summary", "table", "message"})

    def test_meta_keys(self):
        for k in ("as_of_month", "recon_available", "accounts", "classes",
                  "available_dates", "synthetic", "filter"):
            self.assertIn(k, self.view["meta"])

    def test_meta_filter_options_have_id_label(self):
        for opt in self.view["meta"]["accounts"] + self.view["meta"]["classes"]:
            self.assertIn("id", opt)
            self.assertIn("label", opt)

    def test_full_view_jsonable_no_nan(self):
        json.dumps(self.view, allow_nan=False)


class TestHeadlineAndTableParity(unittest.TestCase):
    """The terminal view is byte-identical to the pure engine outputs on the
    SAME report app.py builds — the '1:1 numbers' guarantee at the data layer."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = hes.build_health_view(cls.frames)
        cls.report = _recompute_report(cls.frames)

    def test_headline_text_matches_engine(self):
        _level, text = format_health_headline(self.report)
        self.assertEqual(self.view["headline"]["text"], text)

    def test_headline_level_mapped(self):
        level, _ = format_health_headline(self.report)
        mapped = {"green": "success", "amber": "warning",
                  "red": "error", "grey": "muted"}[level]
        self.assertEqual(self.view["headline"]["level"], mapped)

    def test_table_matches_engine(self):
        self.assertEqual(self.view["table"], health_rows_to_table(self.report))

    def test_summary_counts_match_report(self):
        s = self.view["summary"]
        self.assertEqual(
            (s["n_ok"], s["n_known"], s["n_watch"], s["n_error"], s["n_carried"]),
            (self.report.n_ok, self.report.n_known, self.report.n_watch,
             self.report.n_error, self.report.n_carried))

    def test_fixture_is_amber_carried(self):
        # Pins the known fixture state so a fixture change is caught loudly.
        self.assertEqual(self.view["headline"]["level"], "warning")
        self.assertIn("carried forward", self.view["headline"]["text"])
        self.assertEqual(len(self.view["table"]), 3)
        self.assertEqual(self.view["summary"]["n_ok"], 2)
        self.assertEqual(self.view["summary"]["n_carried"], 1)


class TestEmptyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)

    def test_recon_unavailable_when_no_summaries(self):
        # An empty frames.summaries: recon unavailable -> grey/muted headline,
        # empty table, and NO body message (Streamlit's _render_health_body
        # emits the info message only in the reconciled-but-empty branch; the
        # grey headline already states the unavailable case).
        # build_health_report_for_frames reads frames.summaries (loaded once
        # in load_frames, narrowed by apply_global_filters — the S2a
        # choke-point), NOT a fresh frames.data_dir read, so the "absent"
        # case is simulated by blanking that field directly rather than by
        # pointing data_dir at a directory with no summaries.csv.
        frames2 = dataclasses.replace(self.frames, summaries=pd.DataFrame())
        view = hes.build_health_view(frames2)
        self.assertFalse(view["meta"]["recon_available"])
        self.assertEqual(view["headline"]["level"], "muted")
        self.assertEqual(view["table"], [])
        self.assertIsNone(view["message"])

    def test_ok_state_has_no_message(self):
        view = hes.build_health_view(self.frames)
        # The fixture reconciles (amber/carried), table non-empty -> no message.
        self.assertIsNone(view["message"])


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_health_ok(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("headline", body)
        self.assertIn("table", body)

    def test_unknown_params_ignored_not_500(self):
        # The route declares no filter params; an unexpected query param is
        # ignored by FastAPI (200), never a 500.
        r = self.client.get("/api/health", params={"account": "nope"})
        self.assertEqual(r.status_code, 200)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/health")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestParity(unittest.TestCase):
    """Cross-check the terminal view against what the Streamlit Data Health tab
    actually renders: the color-coded headline alert + the reconciliation
    dataframe. Activates the tab via session_state['active_tab']. Slow (boots
    Streamlit) — intentional."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from streamlit.testing.v1 import AppTest
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = hes.build_health_view(cls.frames)
        cls.report = _recompute_report(cls.frames)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        at.session_state["active_tab"] = "Data Health"
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_headline_alert_parity(self):
        # amber -> st.warning; the terminal headline.text must appear among the
        # rendered alerts (the chrome strip + the tab both emit it). Streamlit
        # auto-extracts a leading emoji from the message into the alert's `.icon`
        # and reports the remainder as `.value`, so the engine string (which
        # leads with the band glyph) is reconstituted as `icon + " " + value`
        # before comparison — otherwise the leading glyph would never match.
        level, text = format_health_headline(self.report)
        bucket = {"green": self.at.success, "amber": self.at.warning,
                  "red": self.at.error, "grey": self.at.info}[level]
        full = [(a.icon + " " + a.value) if a.icon else a.value for a in bucket]
        self.assertIn(text, full,
                      "terminal headline text not found among Streamlit alerts")
        # the terminal's data-layer headline text is the same engine string
        self.assertEqual(self.view["headline"]["text"], text)

    def test_table_parity(self):
        self.assertGreaterEqual(len(self.at.dataframe), 1,
                                "Data Health tab rendered no dataframe")
        got = self.at.dataframe[0].value.reset_index(drop=True)
        exp = pd.DataFrame(self.view["table"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, exp, check_dtype=False)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_data_health_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = hes.build_health_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(view, expected)


if __name__ == "__main__":
    unittest.main()
