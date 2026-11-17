import unittest

from common.isa.emit import AssemblyEmitter, emit_program, load_profile
from common.ir import Instr, MemRef, Program
from common.targets import load_target
from fuzzer.esd_model import build_model
from fuzzer.generator import TestGenerator
import random


class TestEmitter(unittest.TestCase):
    def test_direct_encodings(self):
        p = Program()
        p.emit("mov_imm", dst="r13", imm=0x1234)
        p.emit("add", dst="r14", srcs=("r13", "r13"))
        p.emit("st", srcs=("r14",), mem=MemRef(base="r13", offset=8))
        p.emit("exit")
        res = emit_program(p, "ventus")
        self.assertIn("li x13, 4660", res.asm)
        self.assertIn("add x14, x13, x13", res.asm)
        self.assertIn("sw x14, 8(x13)", res.asm)
        self.assertTrue(res.ok, res.unsupported)

    def test_macros_for_special_ops(self):
        p = Program()
        p.emit("tid", dst="r1")
        p.emit("sat_add", dst="r13", srcs=("r1", "r1"))
        p.emit("exit")
        res = emit_program(p, "ventus")
        self.assertIn("GPU_GET_TID x1", res.asm)
        self.assertIn("GPU_SAT_ADD x13, x1, x1", res.asm)
        prof = load_profile("ventus")
        header = res.macro_header(prof)
        self.assertIn(".macro GPU_SAT_ADD", header)
        self.assertIn(".endm", header)

    def test_missing_macros_are_reported_not_guessed(self):
        p = Program()
        p.emit("tex", dst="r13", srcs=("r1",))
        p.emit("exit")
        res = emit_program(p, "nyuzi")           # nyuzi profile has no macros
        self.assertFalse(res.ok)
        self.assertTrue(any("GPU_TEX" in u for u in res.unsupported))

    def test_every_target_emits_something(self):
        for target in ("vortex", "ventus", "verigpu", "miaow", "nyuzi", "smol-gpu"):
            spec = load_target(target)
            model, _ = build_model(target, use_rtl=False)
            gen = TestGenerator(spec, model, random.Random(2))
            tc = gen.synthesize(next(iter(model)))
            res = emit_program(tc.program, target)
            self.assertIn(".text", res.asm)
            self.assertIn("gpufuzz_entry", res.asm)


if __name__ == "__main__":
    unittest.main()
