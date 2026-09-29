"""Tests for parsers/cache_keys.file_signature.

The helper feeds a file-state tuple into @st.cache_data loader keys so an
out-of-band re-ingest that rewrites a data file invalidates the cache
without a manual clear or server restart (audit WSC-1). The contract that
matters: stable when the file is unchanged (so the cache still hits every
rerun), but different the moment the file changes or (dis)appears.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from cache_keys import file_signature  # noqa: E402


class TestFileSignature(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_stable_when_unchanged(self) -> None:
        p = self.dir / "a.csv"
        p.write_text("x,y\n1,2\n")
        self.assertEqual(file_signature(p), file_signature(p))

    def test_changes_when_content_changes(self) -> None:
        p = self.dir / "a.csv"
        p.write_text("x,y\n1,2\n")
        before = file_signature(p)
        # Longer content -> size differs, so the signature changes even when
        # the filesystem mtime resolution is too coarse to register.
        p.write_text("x,y\n1,2\n3,4\n5,6\n")
        self.assertNotEqual(before, file_signature(p))

    def test_missing_file_is_sentinel_not_error(self) -> None:
        self.assertEqual(file_signature(self.dir / "nope.csv"), ((0.0, 0),))

    def test_present_differs_from_absent(self) -> None:
        p = self.dir / "a.csv"
        self.assertEqual(file_signature(p), ((0.0, 0),))  # absent
        p.write_text("data\n")
        self.assertNotEqual(file_signature(p), ((0.0, 0),))

    def test_multiple_paths_keep_order(self) -> None:
        a = self.dir / "a.csv"
        a.write_text("aa\n")
        b = self.dir / "b.csv"
        b.write_text("bbbb\n")
        sig = file_signature(a, b)
        self.assertEqual(len(sig), 2)
        self.assertEqual(sig, (file_signature(a)[0], file_signature(b)[0]))


if __name__ == "__main__":
    unittest.main()
