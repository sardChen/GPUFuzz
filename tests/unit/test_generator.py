import random
import unittest

from common.targets import TARGETS, load_target
from fuzzer.esd_model import build_model
from fuzzer.generator import TestGenerator
from fuzzer.mutator import Mutator


class TestGeneration(unittest.TestCase):
    def test_all_relations_generate_valid_tests(self):
        total = 0
        for name in TARGETS:
            spec = load_target(name)
            model, _ = build_model(name, use_rtl=False)
            gen = TestGenerator(spec, model, random.Random(7))
            for rel in model:
                tc = gen.synthesize(rel)
                self.assertEqual(tc.program.validate(spec.caps), [],
                                 "%s/%s invalid program" % (name, rel.shape))
                self.assertEqual(tc.config.validate(spec.caps), [],
                                 "%s/%s invalid configuration" % (name, rel.shape))
                self.assertEqual(tc.program.instrs[-1].op, "exit")
                self.assertTrue(tc.protected, "%s/%s has no protected slots" % (name, rel.shape))
                total += 1
        self.assertGreaterEqual(total, 60)

    def test_mutation_preserves_protected_slots_and_validity(self):
        spec = load_target("ventus")
        model, _ = build_model("ventus", use_rtl=False)
        gen = TestGenerator(spec, model, random.Random(3))
        mut = Mutator(spec, random.Random(5))
        for rel in model:
            tc = gen.synthesize(rel)
            for _ in range(3):
                child = mut.mutate(tc, rel)
                if child is None:
                    continue
                self.assertEqual(child.program.validate(spec.caps), [])
                self.assertEqual(child.config.validate(spec.caps), [])
                slots = {i.slot for i in child.program.instrs}
                for slot in tc.protected:
                    self.assertIn(slot, slots, "protected slot %s removed" % slot)


if __name__ == "__main__":
    unittest.main()
