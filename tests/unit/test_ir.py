import unittest

from common.config import Config, EventCondition
from common.ir import Instr, MemRef, Program


def _prog() -> Program:
    p = Program(nregs=32, gmem_bytes=0x1000)
    p.emit("tid", dst="r1")
    p.emit("mov_imm", dst="r13", imm=0x1000)
    p.emit("st", srcs=("r13",), mem=MemRef(base="r13", offset=0))
    p.emit("label", label="L")
    p.emit("add", dst="r14", srcs=("r13",), imm=1)
    p.emit("br", srcs=("r14",), label="L2", aux={"sense": "nz"})
    p.emit("label", label="L2")
    p.emit("exit")
    return p


class TestProgram(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_prog().validate(), [])

    def test_missing_exit(self):
        p = _prog()
        p.instrs = p.instrs[:-1]
        self.assertTrue(any("exit" in x for x in p.validate()))

    def test_backward_branch_is_rejected(self):
        p = Program()
        p.emit("label", label="A")
        p.emit("jmp", label="A")
        p.emit("exit")
        self.assertTrue(any("backward" in x for x in p.validate()))

    def test_unknown_register(self):
        p = Program(nregs=8)
        p.emit("add", dst="r13", srcs=("r1",))
        p.emit("exit")
        self.assertTrue(any("beyond nregs" in x for x in p.validate()))

    def test_capability_check(self):
        p = _prog()
        p.emit("fmul", dst="r14", srcs=("r13", "r13"))
        p.instrs.insert(-1, p.instrs.pop())
        problems = p.validate({"features": ["mul"], "max_regs": 32, "smem_bytes": 0})
        self.assertTrue(any("features" in x for x in problems))

    def test_roundtrip(self):
        p = _prog()
        self.assertEqual(p.json_hash() if hasattr(p, "json_hash") else p.hash(),
                         Program.from_json(p.to_json()).hash())


class TestConfig(unittest.TestCase):
    def test_validate_warps(self):
        cfg = Config(grid=(1, 1, 1), block=(64, 1, 1), warps_per_block=2)
        caps = {"lanes": 32, "warps_per_block": 4, "max_blocks": 16,
                "max_regs": 32, "smem_bytes": 0, "inject": []}
        self.assertEqual(cfg.validate(caps), [])

    def test_event_condition_needs_injection(self):
        cfg = Config(conditions=[EventCondition("cache_flush", "when_dirty")])
        caps = {"lanes": 32, "warps_per_block": 4, "max_blocks": 16,
                "max_regs": 32, "smem_bytes": 0, "inject": ["mem_backpressure"]}
        self.assertTrue(any("cache_flush" in p for p in cfg.validate(caps)))


if __name__ == "__main__":
    unittest.main()
