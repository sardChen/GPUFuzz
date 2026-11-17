import json
import os
import tempfile
import unittest

from adapters.trace.adapter import SignalExpr, TraceAdapter


MAP = {
    "signals": [{"id": "wb_valid", "path": "m.wb_valid", "width": 1},
                {"id": "wb_ready", "path": "m.wb_ready", "width": 1},
                {"id": "wb_reg", "path": "m.wb_reg", "width": 8},
                {"id": "flush_valid", "path": "m.flush_valid", "width": 1},
                {"id": "flush_addr", "path": "m.flush_addr", "width": 32}],
    "events": [
        {"type": "wb", "when": "wb_valid & wb_ready",
         "obj": {"kind": "reg", "from": "wb_reg", "fmt": "reg:%d"}},
        {"type": "flush", "when": "flush_valid",
         "obj": {"kind": "line", "from": "flush_addr", "fmt": "line:0x%x"}},
    ],
    "states": [{"sid": "dcache.dirty", "path": "m.dirty", "obj_fmt": "line:0x%x"}],
}


class TestSignalExpr(unittest.TestCase):
    def test_evaluate(self):
        e = SignalExpr("a & b | c", {"a", "b", "c"})
        self.assertEqual(e.evaluate({"a": 1, "b": 0, "c": 0}), 0)
        self.assertEqual(e.evaluate({"a": 1, "b": 1, "c": 0}), 1)

    def test_unknown_signal(self):
        with self.assertRaises(ValueError):
            SignalExpr("a & zzz", {"a"})

    def test_code_execution_is_rejected(self):
        for bad in ("__import__('os')", "a.foo", "a[0]", "(lambda: 1)()"):
            with self.assertRaises(ValueError):
                SignalExpr(bad, {"a"})


class TestTraceAdapter(unittest.TestCase):
    def _write(self, records):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in records:
            fh.write(json.dumps(r) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_rising_edge_events(self):
        path = self._write([
            {"t": "hdr", "target": "ventus", "top": "Ventus", "unresolved": []},
            {"t": "sig", "c": 1, "v": {"wb_valid": 1, "wb_ready": 1, "wb_reg": 5}},
            {"t": "sig", "c": 2, "v": {"wb_valid": 0}},
            {"t": "sig", "c": 3, "v": {"flush_valid": 1, "flush_addr": 0x40}},
            {"t": "sig", "c": 4, "v": {"flush_valid": 0}},
            {"t": "st", "c": 5, "s": "dcache.dirty", "o": "line:0x40", "v": 1},
            {"t": "arch", "mem": {"4096": 1}, "regs": {}, "status": {"exit": 0}},
            {"t": "end", "c": 5, "timeout": 0, "exit": 0},
        ])
        tr = TraceAdapter("ventus", MAP).parse(path, "tc1")
        wbs = [e for e in tr.events if e.a == "wb"]
        self.assertEqual(len(wbs), 1)                 # one pulse, one event
        self.assertEqual(wbs[0].u, "reg:5")
        self.assertEqual(wbs[0].cycle, 1)
        flushes = [e for e in tr.events if e.a == "flush"]
        self.assertEqual(flushes[0].u, "line:0x40")
        self.assertEqual(tr.states[0].sid, "dcache.dirty")
        self.assertEqual(tr.arch.mem, {4096: 1})
        self.assertFalse(tr.timeout)

    def test_timeout_and_arch(self):
        path = self._write([
            {"t": "hdr", "target": "ventus", "unresolved": ["x.y"]},
            {"t": "arch", "mem": {}, "regs": {}, "status": {"exit": 1}},
            {"t": "end", "c": 999, "timeout": 1, "exit": 1},
        ])
        tr = TraceAdapter("ventus", MAP).parse(path, "tc2")
        self.assertTrue(tr.timeout)
        self.assertEqual(tr.cycles, 999)
        self.assertEqual(tr.meta.get("unresolved"), ["x.y"])


if __name__ == "__main__":
    unittest.main()
