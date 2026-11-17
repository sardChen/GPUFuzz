"""Verilator driver: assemble a test case, run the simulator, adapt the trace."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from adapters.trace.adapter import TraceAdapter
from adapters.verilator import build as builder
from adapters.verilator import gen as harness_gen
from common.driver import Driver
from common.isa.emit import write_assembly
from common.targets import TargetSpec
from common.types import RunResult, TestCase, Trace
from common.util import Log, ensure_dir, read_json, write_json


class VerilatorDriver(Driver):
    """Runs test cases on a Verilator-built model of one target.

    Layout::

        <workdir>/<target>/build/     generated sim_main.cpp, obj_dir, simulator
        <workdir>/<target>/kernels/   assembled kernel images (content addressed)
        <workdir>/<target>/traces/    one JSONL trace per execution
    """

    name = "verilator"

    def __init__(self, target: TargetSpec, workdir: str = "work", rtl: Optional[str] = None,
                 sim: Optional[str] = None, auto_build: bool = False,
                 dry_run: bool = False, code_base: int = 0x0,
                 timeout_s: float = 600.0, log: Optional[Log] = None, **kw: Any) -> None:
        super().__init__(target, **kw)
        self.workdir = os.path.abspath(workdir)
        self.rtl = rtl
        self.sim_override = sim
        self.auto_build = auto_build
        self.dry_run = dry_run
        self.code_base = code_base
        self.timeout_s = timeout_s
        self.log = log or Log("info")
        self.root = os.path.join(self.workdir, target.name)
        self.build_dir = os.path.join(self.root, "build")
        self.kernel_dir = ensure_dir(os.path.join(self.root, "kernels"))
        self.trace_dir = ensure_dir(os.path.join(self.root, "traces"))
        self.adapter = TraceAdapter(target.name, target.signal_map(), log=self.log)
        self.sim = sim or ""
        self._prepared = False
        self._prep_error = ""

    # ------------------------------------------------------------------ #
    def prepare(self) -> Dict[str, Any]:
        if self._prepared:
            return {"ok": bool(self.sim), "sim": self.sim, "error": self._prep_error}
        self._prepared = True
        if self.sim_override:
            self.sim = self.sim_override
            return {"ok": True, "sim": self.sim}
        existing = builder.load_build(self.build_dir) or {}
        if existing.get("ok") and existing.get("sim") and os.path.exists(existing["sim"]):
            self.sim = existing["sim"]
            return {"ok": True, "sim": self.sim, "cached": True}
        if not self.auto_build:
            self._prep_error = ("simulator not built; run `python3 cli.py build --target %s "
                                "--rtl <checkout>` or pass --auto-build" % self.target.name)
            return {"ok": False, "error": self._prep_error}
        res = builder.build(self.target.name, self.rtl, self.build_dir,
                            dry_run=self.dry_run, log=self.log)
        self.sim = res.sim
        self._prep_error = res.error
        return res.to_json()

    # ------------------------------------------------------------------ #
    def execute(self, tc: TestCase) -> RunResult:
        t0 = time.time()
        if not self.sim:
            info = self.prepare()
            if not self.sim:
                return RunResult(ok=False, driver=self.name,
                                 error=info.get("error") or "no simulator")
        try:
            kernel = self._kernel(tc)
        except Exception as exc:
            return RunResult(ok=False, driver=self.name, error="kernel build failed: %s" % exc)
        trace_path = os.path.join(self.trace_dir, "%s.jsonl" % _safe(tc.id))
        cfg = tc.config
        cmd = [self.sim, "--kernel", kernel, "--trace", trace_path,
               "--test-id", tc.id, "--max-cycles", str(cfg.max_cycles),
               "--code-base", hex(self.code_base)]
        for addr, val in cfg.gmem_init:
            cmd += ["--gmem", "%d:%d" % (addr & 0xFFFFFFFF, val & 0xFFFFFFFF)]
        for launch in cfg.launches:
            cmd += ["--launch-arg", str(launch.arg)]
        for cond in cfg.conditions:
            params = ",".join("%s=%s" % (k, v) for k, v in sorted(cond.params.items()))
            cmd += ["--event", "%s:%s:%s" % (cond.ev, cond.anchor, params)]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return RunResult(ok=False, driver=self.name,
                             error="simulator timed out after %.0fs" % self.timeout_s,
                             wall_s=time.time() - t0)
        except FileNotFoundError as exc:
            return RunResult(ok=False, driver=self.name, error=str(exc))
        if not os.path.exists(trace_path):
            return RunResult(ok=False, driver=self.name,
                             error="no trace produced (rc=%d): %s"
                                   % (proc.returncode, (proc.stderr or "")[-400:]),
                             wall_s=time.time() - t0)
        try:
            trace = self.adapter.parse(trace_path, tc.id)
        except Exception as exc:
            return RunResult(ok=False, driver=self.name, error="trace parse failed: %s" % exc,
                             wall_s=time.time() - t0)
        unresolved = trace.meta.get("unresolved") or []
        if unresolved:
            self.log.warn("%s: %d unresolved signals in the signal map (%s)"
                          % (self.target.name, len(unresolved), ", ".join(unresolved[:3])))
        return RunResult(ok=True, trace=trace, driver=self.name, wall_s=time.time() - t0,
                         artifact_dir=self.trace_dir,
                         raw={"rc": proc.returncode, "cmd": cmd, "trace": trace_path})

    # ------------------------------------------------------------------ #
    def _kernel(self, tc: TestCase) -> str:
        key = hashlib.sha256(tc.program.hash().encode()).hexdigest()[:16]
        outdir = ensure_dir(os.path.join(self.kernel_dir, key))
        binary = os.path.join(outdir, "kernel.bin")
        if os.path.exists(binary):
            return binary
        paths = write_assembly(tc.program, self.target.name, outdir, "kernel")
        if paths.get("unsupported"):
            raise RuntimeError("unsupported operations: %s" % paths["unsupported"])
        tool = (paths.get("toolchain") or {})
        cc = tool.get("cc") or "riscv64-unknown-elf-gcc"
        objcopy = tool.get("objcopy") or "riscv64-unknown-elf-objcopy"
        if not shutil.which(cc):
            raise RuntimeError("toolchain %r not found (see common/isa/%s.json)"
                               % (cc, self.target.name))
        elf = os.path.join(outdir, "kernel.elf")
        cmd = [cc] + list(tool.get("flags") or []) + ["-o", elf,
                                                      paths["startup"], paths["asm"]]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
        if proc.returncode != 0:
            raise RuntimeError("assembler failed: %s" % (proc.stderr or "")[-500:])
        if not shutil.which(objcopy):
            raise RuntimeError("objcopy %r not found" % objcopy)
        cmd2 = [objcopy] + list(tool.get("objcopy_flags") or ["-O", "binary"]) + [elf, binary]
        proc = subprocess.run(cmd2, capture_output=True, text=True, cwd=outdir)
        if proc.returncode != 0:
            raise RuntimeError("objcopy failed: %s" % (proc.stderr or "")[-500:])
        write_json(os.path.join(outdir, "kernel.json"),
                   {"entry": paths.get("entry"), "macros": paths.get("macros"),
                    "unsupported": paths.get("unsupported")})
        return binary

    def describe(self) -> str:
        return "verilator(sim=%s)" % (self.sim or "<not built>")


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)[:96]
