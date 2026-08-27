"""Terminal action layer (QA-polish S6): registry shape, job lifecycle,
single-flight, and the route contract.

Offline by construction: every start() in this file goes through the
``actions_service._TEST_STEPS`` seam (`python -c` stubs) — no real parser is
ever shelled by the suite.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "synth_data"

from terminal import actions_service as acts  # noqa: E402

_STUB_OK = [{"label": "stub", "needs_key": False, "timeout": 30,
             "cmd": [sys.executable, "-c", "print('stub-tail-7')"]}]
_STUB_SLOW = [{"label": "slow", "needs_key": False, "timeout": 30,
               "cmd": [sys.executable, "-c", "import time; time.sleep(0.8)"]}]


def _wait_idle(timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not acts.status()["running"]:
            return
        time.sleep(0.05)
    raise AssertionError("action still running after %ss" % timeout)


class _SeamTest(unittest.TestCase):
    def tearDown(self):
        _wait_idle()
        acts._TEST_STEPS = None


class TestRegistry(unittest.TestCase):
    def test_nine_actions_with_real_scripts(self):
        self.assertEqual(len(acts.ACTIONS), 9)
        for aid, a in acts.ACTIONS.items():
            self.assertTrue(a["label"] and a["help"] and a["group"], aid)
            for step in a["steps"]:
                self.assertEqual(step["cmd"][0], sys.executable, aid)
                self.assertTrue((REPO / step["cmd"][1]).exists(),
                                f"{aid}: {step['cmd'][1]} missing")

    def test_needs_key_flags(self):
        keyed = {aid for aid, a in acts.ACTIONS.items()
                 if any(s.get("needs_key") for s in a["steps"])}
        self.assertEqual(keyed, {"refresh_all", "market_data", "dividends",
                                 "option_iv", "atm_iv"})

    def test_refresh_all_is_the_five_step_sequence(self):
        labels = [s["label"] for s in acts.ACTIONS["refresh_all"]["steps"]]
        self.assertEqual(labels, ["Interim transactions", "Market data",
                                  "Dip history", "Option IV",
                                  "ATM IV history"])


class TestLifecycle(_SeamTest):
    def test_start_runs_and_finishes_ok(self):
        acts._TEST_STEPS = _STUB_OK
        ok, why = acts.start("interim")
        self.assertTrue(ok, why)
        _wait_idle()
        st = acts.status()
        self.assertIs(st["ok"], True)
        self.assertEqual(st["steps"], [{"label": "stub", "status": "✅"}])
        self.assertIn("stub-tail-7", st["tail"])
        self.assertIsNotNone(st["finished_at"])

    def test_unknown_action_refused(self):
        ok, why = acts.start("nope")
        self.assertFalse(ok)
        self.assertIn("unknown", why)

    def test_single_flight(self):
        acts._TEST_STEPS = _STUB_SLOW
        ok, _ = acts.start("interim")
        self.assertTrue(ok)
        ok2, why2 = acts.start("market_data")
        self.assertFalse(ok2)
        self.assertIn("already running", why2)
        _wait_idle()


class TestRoutes(_SeamTest):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_roster_shape(self):
        r = self.client.get("/api/actions")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["actions"]), 9)
        for a in body["actions"]:
            for k in ("id", "label", "help", "group", "needs_key"):
                self.assertIn(k, a)
        self.assertIn("running", body["status"])

    def test_unknown_action_422(self):
        r = self.client.post("/api/actions/nope")
        self.assertEqual(r.status_code, 422)

    def test_start_and_status_roundtrip(self):
        acts._TEST_STEPS = _STUB_OK
        r = self.client.post("/api/actions/interim")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"started": True})
        _wait_idle()
        st = self.client.get("/api/actions/status").json()
        self.assertIs(st["ok"], True)
        self.assertIn("stub-tail-7", st["tail"])

    def test_busy_is_409(self):
        acts._TEST_STEPS = _STUB_SLOW
        r1 = self.client.post("/api/actions/interim")
        self.assertEqual(r1.status_code, 200)
        r2 = self.client.post("/api/actions/market_data")
        self.assertEqual(r2.status_code, 409)
        _wait_idle()


if __name__ == "__main__":
    unittest.main()
