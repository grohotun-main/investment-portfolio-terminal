"""Gates for scripts/generate_demo_data.py: deterministic output, the same
column contract as the committed fixture, and a bundle the services load.

The generator's own self-check (leak scan, engine sanity, chart plausibility)
runs inside every invocation — a non-zero exit here means a gate fired.
"""
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synth_data"
SCRIPT = ROOT / "scripts" / "generate_demo_data.py"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))


def _generate(out: Path) -> None:
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"generator failed:\n{r.stdout}\n{r.stderr}"


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "demo"
        _generate(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deterministic(self):
        with TemporaryDirectory() as t2:
            out2 = Path(t2) / "demo"
            _generate(out2)
            names1 = sorted(p.name for p in self.out.iterdir())
            names2 = sorted(p.name for p in out2.iterdir())
            self.assertEqual(names1, names2)
            for name in names1:
                self.assertEqual((self.out / name).read_bytes(),
                                 (out2 / name).read_bytes(),
                                 f"non-deterministic output: {name}")

    def test_column_contract_matches_fixture(self):
        # Every file the committed fixture defines, the demo emits with the
        # same header (the services' schema contract).
        for fx in sorted(FIXTURE.glob("*.csv")):
            demo = self.out / fx.name
            if not demo.exists():
                # per-symbol dividend files differ by universe; the hedge
                # chain is a test-only offline seam, not app data.
                self.assertTrue(fx.name.startswith("dividends_")
                                or fx.name == "hedge_chain_fixture.csv",
                                f"missing from demo output: {fx.name}")
                continue
            self.assertEqual(
                demo.read_text(encoding="utf-8").splitlines()[0],
                fx.read_text(encoding="utf-8").splitlines()[0],
                f"header drift: {fx.name}")

    def test_loads_and_builds(self):
        from terminal import holdings_service as hs
        frames = hs.load_frames(str(self.out))
        self.assertEqual(hs.canonical_broker_label(frames), "Alpine + Harbor")
        view = hs.build_holdings_view(frames)
        self.assertEqual([o["id"] for o in view["meta"]["brokers"]],
                         ["alpine", "harbor"])

    def test_portfolio_irr_row_present_and_sane(self):
        irr = pd.read_csv(self.out / "irr_per_account.csv")
        row = irr[irr["account_id"] == "PORTFOLIO"]
        self.assertEqual(len(row), 1)
        self.assertGreater(float(row["irr"].iloc[0]), -0.9)


if __name__ == "__main__":
    unittest.main()
