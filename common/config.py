"""Execution configuration ``C = <lambda, Gamma>``.

``lambda`` holds kernel launch parameters, memory initialisation and resource
settings.  ``Gamma = <eta_1 ... eta_m>`` holds target-supported execution event
conditions, each ``eta_j = <e_j, a_j, phi_j>``: an event, its activation anchor
and the condition governing its occurrence (Section 4.3).

Anchors are *execution relative* ("after the first store retires", "after N
commits"), never fixed cycle numbers, so the same configuration keeps working
when the DUT's timing changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .util import sha

# Event kinds the adapters know how to inject.  A target advertises which of
# them it supports in specs/<target>.json ("inject": [...]).
INJECTABLE = (
    "cache_flush",         # flush a line / the whole DCache
    "cache_invalidate",    # invalidate a line (Nyuzi L1 semantics)
    "mem_backpressure",    # hold mem_ready low / stall downstream memory
    "launch_queue",        # enqueue launches while another kernel runs
    "tex_reconfig",        # reprogram texture/mipmap state
    "dma_copy",            # copy engine command (VeriGPU controller)
    "warp_oversubscribe",  # request more warps than hardware slots
    "rollback_stress",     # force divergence + reconvergence
)

ANCHORS = (
    "kernel_start",
    "after_retire",
    "after_store",
    "after_issue",
    "when_dirty",
    "when_pending_wb",
    "kernel_end",
)


@dataclass
class EventCondition:
    """``eta = <e, a, phi>``."""

    ev: str
    anchor: str = "kernel_start"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ev not in INJECTABLE:
            raise ValueError("unknown injectable event %r" % (self.ev,))
        if self.anchor not in ANCHORS:
            raise ValueError("unknown anchor %r" % (self.anchor,))

    def describe(self) -> str:
        p = ",".join("%s=%s" % kv for kv in sorted(self.params.items()))
        return "%s@%s(%s)" % (self.ev, self.anchor, p)

    def to_json(self) -> Dict[str, Any]:
        return {"ev": self.ev, "anchor": self.anchor, "params": dict(self.params)}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "EventCondition":
        return EventCondition(ev=d["ev"], anchor=d.get("anchor", "kernel_start"),
                              params=dict(d.get("params") or {}))


@dataclass
class Launch:
    """One kernel launch in the (possibly queued) launch stream."""

    grid: Tuple[int, int, int] = (1, 1, 1)
    block: Tuple[int, int, int] = (32, 1, 1)
    arg: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {"grid": list(self.grid), "block": list(self.block), "arg": self.arg}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Launch":
        g = tuple(int(x) for x in (d.get("grid") or (1, 1, 1)))
        b = tuple(int(x) for x in (d.get("block") or (32, 1, 1)))
        return Launch(grid=g, block=b, arg=int(d.get("arg", 0)))


@dataclass
class Config:
    """``C = <lambda, Gamma>``."""

    # lambda: launch parameters and resources
    grid: Tuple[int, int, int] = (1, 1, 1)
    block: Tuple[int, int, int] = (32, 1, 1)
    warps_per_block: int = 1
    regs_per_thread: int = 32
    smem_bytes: int = 0
    gmem_base: int = 0x1000
    gmem_init: List[Tuple[int, int]] = field(default_factory=list)
    launches: List[Launch] = field(default_factory=list)

    # Gamma: execution event conditions
    conditions: List[EventCondition] = field(default_factory=list)

    # execution limits
    max_cycles: int = 300000
    progress_timeout: int = 40000
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.launches:
            self.launches = [Launch(grid=self.grid, block=self.block)]

    # -- derived ----------------------------------------------------------- #
    @property
    def threads_per_block(self) -> int:
        b = self.block
        return max(1, b[0]) * max(1, b[1]) * max(1, b[2])

    @property
    def blocks(self) -> int:
        g = self.grid
        return max(1, g[0]) * max(1, g[1]) * max(1, g[2])

    def conditions_of(self, ev: str) -> List[EventCondition]:
        return [c for c in self.conditions if c.ev == ev]

    def has(self, ev: str) -> bool:
        return any(c.ev == ev for c in self.conditions)

    def describe(self) -> str:
        conds = "; ".join(c.describe() for c in self.conditions) or "-"
        return ("grid=%s block=%s warps=%d regs=%d smem=%dB launches=%d | %s"
                % (self.grid, self.block, self.warps_per_block, self.regs_per_thread,
                   self.smem_bytes, len(self.launches), conds))

    # -- validation -------------------------------------------------------- #
    def validate(self, caps: Dict[str, Any]) -> List[str]:
        problems: List[str] = []
        lanes = int(caps.get("lanes", 32))
        tpb = self.threads_per_block
        if tpb % lanes:
            problems.append("threads/block=%d is not a multiple of lanes=%d" % (tpb, lanes))
        warps = tpb // lanes
        if warps > int(caps.get("warps_per_block", 32)):
            problems.append("block needs %d warps > target limit %d"
                            % (warps, caps.get("warps_per_block", 32)))
        if self.warps_per_block != warps:
            problems.append("warps_per_block=%d inconsistent with block=%s (implies %d)"
                            % (self.warps_per_block, self.block, warps))
        if self.blocks > int(caps.get("max_blocks", 1024)):
            problems.append("grid=%s exceeds max_blocks=%s" % (self.grid, caps.get("max_blocks")))
        if self.smem_bytes > int(caps.get("smem_bytes", 0)):
            problems.append("smem_bytes=%d exceeds target limit %d"
                            % (self.smem_bytes, caps.get("smem_bytes", 0)))
        if self.regs_per_thread > int(caps.get("max_regs", 32)):
            problems.append("regs_per_thread=%d exceeds max_regs=%d"
                            % (self.regs_per_thread, caps.get("max_regs", 32)))
        injectable = set(caps.get("inject") or ())
        for c in self.conditions:
            if c.ev not in injectable:
                problems.append("target cannot inject event %r" % (c.ev,))
        # sequential launches need no injection; a queued launch *condition* does,
        # and that is already covered by the per-condition check above
        return problems

    def hash(self) -> str:
        return sha(self.to_json(), n=16)

    # -- serialization ----------------------------------------------------- #
    def to_json(self) -> Dict[str, Any]:
        return {
            "grid": list(self.grid), "block": list(self.block),
            "warps_per_block": self.warps_per_block,
            "regs_per_thread": self.regs_per_thread,
            "smem_bytes": self.smem_bytes, "gmem_base": self.gmem_base,
            "gmem_init": [[int(a), int(v)] for a, v in self.gmem_init],
            "launches": [l.to_json() for l in self.launches],
            "conditions": [c.to_json() for c in self.conditions],
            "max_cycles": self.max_cycles, "progress_timeout": self.progress_timeout,
            "extra": self.extra,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Config":
        return Config(
            grid=tuple(int(x) for x in (d.get("grid") or (1, 1, 1))),
            block=tuple(int(x) for x in (d.get("block") or (32, 1, 1))),
            warps_per_block=int(d.get("warps_per_block", 1)),
            regs_per_thread=int(d.get("regs_per_thread", 32)),
            smem_bytes=int(d.get("smem_bytes", 0)),
            gmem_base=int(d.get("gmem_base", 0x1000)),
            gmem_init=[(int(a), int(v)) for a, v in (d.get("gmem_init") or [])],
            launches=[Launch.from_json(x) for x in (d.get("launches") or [])],
            conditions=[EventCondition.from_json(x) for x in (d.get("conditions") or [])],
            max_cycles=int(d.get("max_cycles", 300000)),
            progress_timeout=int(d.get("progress_timeout", 40000)),
            extra=dict(d.get("extra") or {}),
        )

    def clone(self) -> "Config":
        return Config.from_json(self.to_json())
