"""Tests for parsers/_config.py's hostile-SSLKEYLOGFILE neutralization.

A TLS-monitoring agent on the dev box injects a per-session ``SSLKEYLOGFILE``
naming one of its own named pipes (``\\.\nllMonFltProxy\<hex>``). Once that
pipe goes stale, Python's ``ssl.create_default_context()`` aborts every TLS
connection with PermissionError — breaking the Polygon fetchers AND the
terminal's ``anthropic.Anthropic()`` construction (every /api/ai/* -> 500).
``_config`` pops such a device-namespace path at import; a legitimate file
path is left alone.
"""
import os
import ssl
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import _config as cfg  # noqa: E402


class TestNeutralizeHostileKeylog(unittest.TestCase):
    def _run_with(self, value):
        """Set SSLKEYLOGFILE (or unset when None), run the guard, return the
        resulting os.environ value (None if absent)."""
        env = {k: v for k, v in os.environ.items() if k != "SSLKEYLOGFILE"}
        if value is not None:
            env["SSLKEYLOGFILE"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            cfg._neutralize_hostile_keylog()
            return os.environ.get("SSLKEYLOGFILE")

    def test_strips_device_namespace_pipe(self):
        # The exact injection shapes seen on the dev box.
        for hostile in (r"\\.\nllMonFltProxy\3ca0c265230a30c0",
                        r"\\.\nlaKdbg_engine_ipc_bd3a",
                        "//./nllMonFltProxy/deadbeef"):  # forward-slash variant
            self.assertIsNone(self._run_with(hostile),
                              f"should strip device path {hostile!r}")

    def test_preserves_legit_file_path(self):
        legit = r"C:\Users\dev\tls-keys.log"
        self.assertEqual(self._run_with(legit), legit)

    def test_unset_is_a_noop(self):
        self.assertIsNone(self._run_with(None))

    @unittest.skipUnless(sys.platform == "win32",
                         "device-namespace path semantics are Windows-only; "
                         "on POSIX '\\\\.\\x' is a valid filename and would be "
                         "created rather than rejected")
    def test_ssl_context_survives_stale_device_pipe(self):
        """End-to-end: with the hostile value present a default context blows
        up; after the guard runs, context creation succeeds again."""
        env = {k: v for k, v in os.environ.items() if k != "SSLKEYLOGFILE"}
        # A device path that cannot be opened reproduces the live failure.
        env["SSLKEYLOGFILE"] = r"\\.\nlaKdbg_engine_ipc_stale"
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises((PermissionError, FileNotFoundError, OSError)):
                ssl.create_default_context()
            cfg._neutralize_hostile_keylog()
            ssl.create_default_context()  # must not raise now


if __name__ == "__main__":
    unittest.main()
