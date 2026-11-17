"""Target descriptions: the six open-source GPGPUs from the paper.

A target spec is *data* (``specs/<target>.json``) plus a few accessors.  It is
the only place that knows a design's capabilities; the fuzzer, the oracle and
the adapters all consume it, so adding a seventh GPU means adding one JSON file,
one signal map and one build recipe.

Capabilities are deliberately expressed in the IR's vocabulary (``features``) so
that generated programs can be validated before they ever reach a simulator.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .util import read_json

TARGETS = ("vortex", "ventus", "verigpu", "miaow", "nyuzi", "smol-gpu")

# Table 1 of the paper.
TABLE1 = {
    "vortex": dict(isa="RISC-V + SIMT", arch="Multi-core SIMT", hdl="SystemVerilog"),
    "ventus": dict(isa="RISC-V + RVV", arch="Vector/SIMT", hdl="Chisel"),
    "verigpu": dict(isa="RISC-V-like", arch="GPU prototype", hdl="SystemVerilog"),
    "miaow": dict(isa="AMD Southern Islands", arch="Compute Unit", hdl="Verilog"),
    "nyuzi": dict(isa="Custom ISA", arch="Vector GPU", hdl="SystemVerilog"),
    "smol-gpu": dict(isa="RV32I-like", arch="Multi-warp SIMT", hdl="SystemVerilog"),
}


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def specs_dir() -> str:
    return os.path.join(repo_root(), "specs")


@dataclass
class TargetSpec:
    name: str
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- basics ------------------------------------------------------------ #
    @property
    def full_name(self) -> str:
        return self.raw.get("full_name", self.name)

    @property
    def isa(self) -> str:
        return self.raw.get("isa") or TABLE1.get(self.name, {}).get("isa", "?")

    @property
    def arch(self) -> str:
        return self.raw.get("arch") or TABLE1.get(self.name, {}).get("arch", "?")

    @property
    def hdl(self) -> str:
        return self.raw.get("hdl") or TABLE1.get(self.name, {}).get("hdl", "?")

    @property
    def caps(self) -> Dict[str, Any]:
        c = dict(self.raw.get("caps") or {})
        c.setdefault("lanes", 32)
        c.setdefault("warps_per_block", 32)
        c.setdefault("max_blocks", 1024)
        c.setdefault("max_regs", 32)
        c.setdefault("smem_bytes", 0)
        c.setdefault("features", [])
        c.setdefault("inject", [])
        return c

    # -- paths ------------------------------------------------------------- #
    def path(self, key: str, default: str = "") -> str:
        p = self.raw.get(key, default)
        if not p:
            return ""
        return p if os.path.isabs(p) else os.path.join(repo_root(), p)

    @property
    def signals_path(self) -> str:
        return self.path("signals", os.path.join("adapters", "signals", self.name + ".json"))

    @property
    def recipe_path(self) -> str:
        return self.path("recipe", os.path.join("adapters", "verilator", "recipes", self.name + ".json"))

    @property
    def isa_profile_path(self) -> str:
        return self.path("isa_profile", os.path.join("common", "isa", self.name + ".json"))

    def signal_map(self) -> Dict[str, Any]:
        return read_json(self.signals_path, {}) or {}

    def recipe(self) -> Dict[str, Any]:
        return read_json(self.recipe_path, {}) or {}

    # -- bug bookkeeping (Table 2) ---------------------------------------- #
    @property
    def bugs(self) -> Dict[str, Any]:
        return dict(self.raw.get("bugs") or {})

    def bug_for_shape(self, shape: str) -> str:
        for bid, meta in self.bugs.items():
            if meta.get("shape") == shape:
                return bid
        return ""

    def describe(self) -> str:
        c = self.caps
        return ("%s (%s, %s, %s): lanes=%d warps/block=%d smem=%dB features=%s"
                % (self.full_name, self.isa, self.arch, self.hdl, c["lanes"],
                   c["warps_per_block"], c["smem_bytes"], ",".join(c["features"])))


def load_target(name: str) -> TargetSpec:
    if name not in TARGETS:
        raise ValueError("unknown target %r (known: %s)" % (name, ", ".join(TARGETS)))
    path = os.path.join(specs_dir(), name + ".json")
    raw = read_json(path)
    if raw is None:
        raise FileNotFoundError("missing target spec: %s" % path)
    return TargetSpec(name=name, raw=raw)


def load_all(names: Optional[List[str]] = None) -> Dict[str, TargetSpec]:
    return {n: load_target(n) for n in (names or list(TARGETS))}
