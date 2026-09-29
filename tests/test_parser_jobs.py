"""Unit tests for parsers/parser_jobs.py (QA-polish S6 extraction).

Offline by construction: the only subprocesses launched are `python -c` stubs.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parsers"))

import parser_jobs  # noqa: E402


class TestResolveKey(unittest.TestCase):
    def test_env_hit_wins(self):
        old = os.environ.get("MASSIVE_API_KEY")
        os.environ["MASSIVE_API_KEY"] = "k-test-123"
        try:
            self.assertEqual(parser_jobs.resolve_massive_api_key(), "k-test-123")
        finally:
            if old is None:
                os.environ.pop("MASSIVE_API_KEY", None)
            else:
                os.environ["MASSIVE_API_KEY"] = old


class TestRunSubprocess(unittest.TestCase):
    def test_happy_path_tail(self):
        ok, tail = parser_jobs.run_parser_subprocess(
            "stub", [sys.executable, "-c", "print('tail-marker-42')"], timeout=60)
        self.assertTrue(ok)
        self.assertIn("tail-marker-42", tail)

    def test_nonzero_exit_is_not_ok(self):
        ok, tail = parser_jobs.run_parser_subprocess(
            "stub", [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
            timeout=60)
        self.assertFalse(ok)
        self.assertIn("boom", tail)

    def test_timeout_reported(self):
        ok, tail = parser_jobs.run_parser_subprocess(
            "sleepy", [sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
        self.assertFalse(ok)
        self.assertIn("timed out", tail)


class TestRunSequence(unittest.TestCase):
    def test_keyed_step_skipped_without_key(self):
        steps = [{"label": "K", "needs_key": True, "timeout": 30,
                  "cmd": [sys.executable, "-c", "print('never-runs')"]}]
        all_ok, results, tail = parser_jobs.run_parser_sequence(steps, api_key="")
        self.assertFalse(all_ok)
        self.assertEqual(results, [("K", "⏭️")])
        self.assertIn("Skipped", tail)
        self.assertNotIn("never-runs", tail)

    def test_failure_does_not_abort_rest(self):
        steps = [
            {"label": "bad", "needs_key": False, "timeout": 30,
             "cmd": [sys.executable, "-c", "import sys; sys.exit(1)"]},
            {"label": "good", "needs_key": False, "timeout": 30,
             "cmd": [sys.executable, "-c", "print('second-ran')"]},
        ]
        events = []
        all_ok, results, tail = parser_jobs.run_parser_sequence(
            steps, api_key="", on_step=lambda *a: events.append(a))
        self.assertFalse(all_ok)
        self.assertEqual([icon for _, icon in results], ["❌", "✅"])
        self.assertIn("second-ran", tail)
        self.assertEqual(events[0], ("start", "bad", None))
        self.assertEqual(events[-1], ("done", "good", "✅"))


if __name__ == "__main__":
    unittest.main()
