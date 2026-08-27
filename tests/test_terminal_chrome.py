"""Terminal global chrome (QA-polish S7): staleness warnings, data sources,
regime badge, footer. ``today`` is pinned so day-count fields are
deterministic; the golden is byte-comparable on the synth fixture.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "synth_data"
TODAY = "2026-05-20"   # fixture positions end 2026-04-30 -> holdings-stale (20d) fires

from terminal import chrome_service as cs  # noqa: E402
from terminal import holdings_service as hs  # noqa: E402


class TestView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = cs.build_chrome_view(cls.frames, FIXTURE, today=TODAY)

    def test_shape(self):
        for k in ("prices_caption", "warnings", "data_sources", "regime",
                  "footer", "tape"):
            self.assertIn(k, self.view)
        for w in self.view["warnings"]:
            self.assertIn("icon", w)
            self.assertIn("text", w)
        self.assertIn("caption", self.view["data_sources"])
        self.assertIn("stale_rows", self.view["data_sources"])
        self.assertIn("available", self.view["regime"])

    def test_holdings_stale_warning_fires_on_old_fixture(self):
        texts = " ".join(w["text"] for w in self.view["warnings"])
        self.assertIn("days stale", texts)

    def test_footer_mirrors_app(self):
        self.assertTrue(self.view["footer"].startswith("Phase 0 reconciliation"))
        if not self.frames.transactions.empty:
            self.assertIn("rows", self.view["footer"])

    def test_tape_matches_holdings_tape(self):
        # One tape, two carriers: the persistent KPI strip is broker/history-
        # scoped global chrome, so /api/chrome must ship the SAME cells the
        # Holdings payload carries — otherwise a filter change on a
        # non-Holdings tab leaves the strip stale until Holdings refetches.
        hview = hs.build_holdings_view(self.frames)
        self.assertEqual(self.view["tape"], hview["tape"])
        self.assertEqual([c["key"] for c in self.view["tape"]],
                         ["portfolio_value", "cum_twr", "annualized",
                          "irr", "vol", "max_dd"])

    def test_tape_matches_holdings_tape_under_broker_filter(self):
        # The one-tape rule (#323) must hold on a filtered scope too: chrome
        # and Holdings read the same _kpi_tape(frames), so a broker-narrowed
        # frames object yields identical cells — including the scoped IRR sub.
        opts = hs._broker_options(hs._current_snap(self.frames))[0]
        one = [o["id"] for o in opts if o["id"] != "all"][:1]
        if not one:
            self.skipTest("no broker options on fixture")
        f2 = hs.apply_global_filters(self.frames, one, "all")
        cview = cs.build_chrome_view(f2, FIXTURE, today=TODAY)
        hview = hs.build_holdings_view(f2)
        self.assertEqual(cview["tape"], hview["tape"])

    def test_json_native(self):
        json.dumps(self.view, allow_nan=False)   # raises on NaN / numpy leaks

    def test_today_pins_determinism(self):
        again = cs.build_chrome_view(self.frames, FIXTURE, today=TODAY)
        self.assertEqual(self.view, again)

    def test_today_reaches_the_rf_probe(self):
        # Regression (main CI 2026-07-13): the RF staleness probe defaulted to
        # the REAL clock, so the golden baked in its generation date and broke
        # one business day later. A different pinned today must change the
        # rf-warning day count — proving the pin threads all the way through.
        def _rf_days(view):
            for w in view["warnings"]:
                if "Risk-free rate" in w["text"]:
                    return w["text"]
            return None
        later = cs.build_chrome_view(self.frames, FIXTURE, today="2026-06-22")
        a, b = _rf_days(self.view), _rf_days(later)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a, b)


class TestGolden(unittest.TestCase):
    GOLDEN = REPO / "tests" / "fixtures" / "terminal_chrome_golden.json"

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = cs.build_chrome_view(frames, FIXTURE, today=TODAY)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(view, expected)


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

    def test_chrome_ok(self):
        r = self.client.get("/api/chrome")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for k in ("prices_caption", "warnings", "data_sources", "regime", "footer"):
            self.assertIn(k, body)

    def test_unknown_broker_422(self):
        r = self.client.get("/api/chrome", params={"broker": "nope"})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
