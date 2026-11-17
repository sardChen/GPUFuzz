import random
import unittest

from common.targets import load_target
from fuzzer.esd_model import build_model
from fuzzer.generator import TestGenerator
from oracle import ConsistencyOracle
from tests.fixtures.driver import FixtureDriver, expected_obligation


class TestOracleRegression(unittest.TestCase):
    """Clean traces must be silent; faulty traces must report the right rule."""

    @classmethod
    def setUpClass(cls):
        cls.spec = load_target("ventus")
        cls.model, _ = build_model("ventus", use_rtl=False)
        cls.gen = TestGenerator(cls.spec, cls.model, random.Random(1))
        cls.oracle = ConsistencyOracle(cls.spec, cls.model)

    def _check(self, rel, clean):
        tc = self.gen.synthesize(rel)
        drv = FixtureDriver(self.spec, clean=clean)
        return tc, self.oracle.check(drv.execute(tc).trace, tc)

    def test_every_relation_detects_its_obligation(self):
        for rel in self.model:
            if rel.target != "ventus":
                continue
            tc, clean = self._check(rel, True)
            self.assertTrue(clean.clean, "%s: false positive %s"
                            % (rel.shape, [v.oid for v in clean.violations]))
            tc, faulty = self._check(rel, False)
            want = expected_obligation(tc)
            got = {v.oid for v in faulty.violations}
            self.assertIn(want, got, "%s: expected %s, got %s" % (rel.shape, want, got))

    def test_progress_family(self):
        rel = self.model.get("R-VENTUS-TX_PROGRESS")
        tc, res = self._check(rel, False)
        self.assertTrue(any(v.family == "progress" for v in res.violations))


if __name__ == "__main__":
    unittest.main()
