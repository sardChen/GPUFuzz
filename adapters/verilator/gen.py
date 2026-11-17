"""Generate the Verilator testbench from a recipe + signal map.

Everything target specific is data:

* ``adapters/verilator/recipes/<target>.json`` -- top module, clock/reset ports,
  source list, Verilator flags, memory pin map, toolchain
* ``adapters/signals/<target>.json`` -- RTL signal paths for states/events, the
  event derivation rules the Python trace adapter uses, and the injection method
  for each controllable event

The generated ``sim_main.cpp`` instantiates the model, resolves the signals by
hierarchical path, drives one test case, and writes a JSONL trace.  Nothing in the
C++ is hand edited, and ``--dry-run`` writes the sources without invoking
Verilator, so the generation itself is testable on any machine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.targets import TargetSpec, load_target
from common.util import ensure_dir, read_json, write_json

MEM_KINDS = {"none": 0, "simple": 1, "axil": 2, "axi4": 3}

# canonical pin names used by the generated main
PIN_ORDER = [
    "REQ_VALID", "REQ_READY", "REQ_WE", "REQ_ADDR", "REQ_WDATA", "RSP_VALID", "RSP_RDATA",
    "AWVALID", "AWREADY", "AWADDR", "WVALID", "WREADY", "WDATA", "WSTRB",
    "BVALID", "BREADY", "BRESP", "ARVALID", "ARREADY", "ARADDR",
    "RVALID", "RREADY", "RDATA", "RLAST",
]


@dataclass
class GenResult:
    outdir: str
    cpp: str
    build_cmd: List[str] = field(default_factory=list)
    unresolved_roles: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"outdir": self.outdir, "cpp": self.cpp, "build_cmd": self.build_cmd,
                "unresolved": self.unresolved_roles, "notes": self.notes}


def _cstr(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


class HarnessGenerator:
    def __init__(self, spec: TargetSpec) -> None:
        self.spec = spec
        self.recipe = spec.recipe()
        self.sigmap = spec.signal_map()
        if not self.recipe:
            raise FileNotFoundError("missing Verilator recipe: %s" % spec.recipe_path)
        if not self.sigmap:
            raise FileNotFoundError("missing signal map: %s" % spec.signals_path)

    # ------------------------------------------------------------------ #
    def generate(self, outdir: str) -> GenResult:
        ensure_dir(outdir)
        res = GenResult(outdir=outdir, cpp=os.path.join(outdir, "sim_main.cpp"))

        signals = list(self.sigmap.get("signals") or [])
        # pins become addressable signals too (they are read and written)
        pins = (self.recipe.get("memory") or {}).get("pins") or {}
        pin_sig: Dict[str, int] = {}
        for name in PIN_ORDER:
            path = pins.get(name.lower()) or pins.get(name)
            if not path:
                continue
            pin_sig[name] = len(signals)
            signals.append({"id": "pin_%s" % name.lower(), "path": path,
                            "width": 64, "kind": "signal"})

        # every declared signal is emitted, even when its path is still unresolved:
        # the harness then reports it in the trace header instead of silently
        # shifting every dependent object index.
        sig_rows = []
        sig_index: Dict[str, int] = {}
        for s in signals:
            paths = s.get("paths")
            path = s.get("path") or (paths or {}).get("template", "").replace("%d", "0")
            sig_index[s.get("id")] = len(sig_index)
            sig_rows.append('    {"%s", "%s", %d},'
                            % (_cstr(s.get("id", "")), _cstr(path), int(s.get("width", 1))))

        state_rows = []
        for st in self.sigmap.get("states") or []:
            obj_sig = sig_index.get(st.get("obj_from", ""), -1)
            state_rows.append('    {"%s", "%s", "%s", %d, %d},'
                              % (_cstr(st.get("sid", "")), _cstr(st.get("path", "")),
                                 _cstr(st.get("obj_fmt", "%s")), obj_sig,
                                 int(st.get("width", 1))))
        if not state_rows:
            res.notes.append("signal map declares no state elements: the interaction "
                             "detector will report 'not observable'")

        inj_rows = []
        for ev, spec in (self.sigmap.get("inject") or {}).items():
            method = spec.get("method", "drive")
            path = spec.get("path") or "|".join(spec.get("pins") or [])
            inj_rows.append('    {"%s", "%s", "%s", %d, %d},'
                            % (_cstr(ev), _cstr(method), _cstr(path),
                               int(spec.get("value", 0)), int(spec.get("cycles", 1))))

        memory = self.recipe.get("memory") or {}
        kind = MEM_KINDS.get(memory.get("kind", "none"), 0)
        pins_init = []
        enum_init = []
        for i, name in enumerate(PIN_ORDER):
            idx = pin_sig.get(name, -1)
            enum_init.append("    PIN_%s = %d," % (name, idx))
            pins_init.append('    "%s",' % name)
        base = int(memory.get("base", 0x1000))
        size = int(memory.get("size", 1 << 20))

        unresolved_roles = self._unresolved_progress()
        res.unresolved_roles = unresolved_roles

        subs = {
            "{{TOP}}": self.recipe.get("top", "Top"),
            "{{TARGET}}": self.spec.name,
            "{{SIGNALS}}": "\n".join(sig_rows),
            "{{STATES}}": "\n".join(state_rows),
            "{{INJECTS}}": "\n".join(inj_rows),
            "{{CLOCK}}": self.recipe.get("clock", {}).get("path", "clock"),
            "{{RESET}}": self.recipe.get("reset", {}).get("path", "reset"),
            "{{RESET_LEVEL}}": "1" if self.recipe.get("reset", {}).get("polarity", "high") == "high" else "0",
            "{{RESET_CYCLES}}": str(int(self.recipe.get("reset", {}).get("cycles", 20))),
            "{{DONE}}": self.recipe.get("done", ""),
            "{{RETIRE}}": self.recipe.get("progress", {}).get("retire", ""),
            "{{STORE}}": self.recipe.get("progress", {}).get("store", ""),
            "{{MEM_KIND}}": str(kind),
            "{{MEM_BASE}}": hex(base),
            "{{MEM_SIZE}}": hex(size),
            "{{PINS}}": "\n".join(pins_init),
            "{{PIN_ENUM}}": "\n".join(enum_init),
            "{{CLOCK_MEMBER}}": self.recipe.get("clock", {}).get("port", "clock"),
            "{{RESET_MEMBER}}": self.recipe.get("reset", {}).get("port", "reset"),
        }

        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "runtime", "sim_main.cpp.in"), "r", encoding="utf-8") as fh:
            text = fh.read()
        for k, v in subs.items():
            text = text.replace(k, v)
        with open(res.cpp, "w", encoding="utf-8") as fh:
            fh.write(text)

        # copy the runtime header next to the generated main
        with open(os.path.join(here, "runtime", "gpufuzz_rt.h"), "r", encoding="utf-8") as fh:
            rt = fh.read()
        with open(os.path.join(outdir, "gpufuzz_rt.h"), "w", encoding="utf-8") as fh:
            fh.write(rt)

        res.build_cmd = self.build_command(outdir)
        write_json(os.path.join(outdir, "gen.json"),
                   {"recipe": self.recipe.get("target"), "signals": len(sig_rows),
                    "states": len(state_rows), "injects": len(inj_rows),
                    "memory_kind": memory.get("kind", "none"),
                    "build_cmd": res.build_cmd, "unresolved": unresolved_roles})
        if not self.recipe.get("done"):
            res.notes.append("recipe has no 'done' signal: every run will hit max_cycles")
        return res

    # ------------------------------------------------------------------ #
    def build_command(self, outdir: str) -> List[str]:
        r = self.recipe
        cmd = ["verilator", "--cc", "--build", "--top-module", r.get("top", "Top"),
               "--Mdir", os.path.join(outdir, "obj_dir"),
               "-Wno-fatal", "-Wno-UNOPTFLAT", "--public-flat-rw", "--trace"]
        for inc in r.get("include_dirs") or ():
            cmd += ["-I" + inc]
        cmd += list(r.get("verilator_flags") or ())
        cmd += list(r.get("sources") or ())
        cmd += ["--exe", os.path.join(outdir, "sim_main.cpp")]
        cmd += ["-CFLAGS", "-I" + outdir]
        cmd += ["-o", "gpufuzz_sim"]
        return cmd

    def _unresolved_progress(self) -> List[str]:
        need = ["done"]
        out = []
        for k in need:
            if not self.recipe.get(k):
                out.append(k)
        return out


def generate(spec: TargetSpec, outdir: str) -> GenResult:
    return HarnessGenerator(spec).generate(outdir)


def generate_all(outdir: str, targets: Optional[List[str]] = None) -> Dict[str, Any]:
    from common.targets import TARGETS
    res = {}
    for name in targets or list(TARGETS):
        spec = load_target(name)
        try:
            res[name] = generate(spec, os.path.join(outdir, name)).to_json()
        except Exception as exc:
            res[name] = {"error": str(exc)}
    return res
