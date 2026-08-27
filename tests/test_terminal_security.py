"""Loopback trust-boundary guard for the terminal server.

The API binds 127.0.0.1 only, but loopback does not protect against code running
in the user's own browser. ``_loopback_guard`` closes two gaps:

  * DNS rebinding — a page on attacker.com rebinds it to 127.0.0.1 and reads the
    API cross-origin. The request still carries ``Host: attacker.com``, so a
    loopback-only Host allowlist rejects it.
  * CSRF — a cross-origin page POSTs to a state-changing endpoint (starting a
    parser job). The request carries a foreign ``Origin``, so it is rejected
    before the route runs.

"testserver" is the in-process TestClient authority (unreachable from a browser)
and is therefore allowed, which is why the other terminal tests keep working.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

# An id that is never in actions_service.ACTIONS, so POST /api/actions/<id>
# answers 422 IF (and only if) the request reaches the route past the guard.
UNKNOWN_ACTION = "definitely-not-a-real-action"


class TestLoopbackGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.app = app
        cls.client = TestClient(app)                       # Host: testserver

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    # --- Host header: DNS-rebinding defense --------------------------------
    def test_testclient_host_allowed(self):
        # The in-process test authority is on the allowlist, so ordinary
        # terminal tests are unaffected by the guard.
        self.assertEqual(self.client.get("/tokens.css").status_code, 200)

    def test_explicit_loopback_host_allowed(self):
        from fastapi.testclient import TestClient
        c = TestClient(self.app, base_url="http://127.0.0.1:8502")
        self.assertEqual(c.get("/tokens.css").status_code, 200)

    def test_foreign_host_rejected(self):
        # DNS-rebinding: the browser connects to 127.0.0.1 but sends the
        # attacker's own hostname in Host.
        from fastapi.testclient import TestClient
        c = TestClient(self.app, base_url="http://attacker.example")
        self.assertEqual(c.get("/tokens.css").status_code, 403)

    # --- Origin header: CSRF defense ---------------------------------------
    def test_loopback_origin_allowed(self):
        r = self.client.get("/tokens.css",
                            headers={"origin": "http://127.0.0.1:8502"})
        self.assertEqual(r.status_code, 200)

    def test_cross_origin_get_rejected(self):
        r = self.client.get("/tokens.css",
                            headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_cross_origin_post_blocked_before_route(self):
        # A foreign Origin is rejected by the guard (403); it must NOT reach the
        # action route, which would answer 422 for the unknown id (and, for a
        # real id, would start a parser job).
        r = self.client.post(f"/api/actions/{UNKNOWN_ACTION}",
                            headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_same_origin_post_reaches_route(self):
        # Same-origin POST passes the guard and reaches the route, which rejects
        # the unknown id with 422 — proving the guard is not blocking legit use.
        r = self.client.post(f"/api/actions/{UNKNOWN_ACTION}",
                            headers={"origin": "http://127.0.0.1:8502"})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
