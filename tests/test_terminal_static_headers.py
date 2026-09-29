"""Cache-header contract for the terminal's static assets (QA-polish S1).

Static responses must carry ``Cache-Control: no-cache`` so browsers revalidate
``/app.js`` on every load (the StaticFiles ETag still yields 304s). Without any
Cache-Control header, browsers heuristic-cache the bundle and serve a stale
front-end after a merge. API responses are exempt — the middleware is scoped to
``/`` + ``*.js`` / ``*.css`` / ``*.html``.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"


class TestStaticNoCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_static_assets_no_cache(self):
        for path in ("/", "/app.js", "/app.css", "/tokens.css"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            self.assertEqual(r.headers.get("cache-control"), "no-cache", path)

    def test_api_responses_not_forced(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertNotEqual(r.headers.get("cache-control"), "no-cache")
