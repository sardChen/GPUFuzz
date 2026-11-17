"""Generation context handed to shape templates."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.config import Config, EventCondition, Launch
from common.targets import TargetSpec
from common.types import Relation
from fuzzer.builder import GenPlan, ProgramBuilder
from .spec import ShapeSpec


@dataclass
class GenCtx:
    """Everything a shape template needs to synthesize ``T = <K, C>``."""

    target: TargetSpec
    rel: Relation
    shape: ShapeSpec
    rng: random.Random
    params: Dict[str, Any] = field(default_factory=dict)
    lanes: int = 32
    origin: str = "seed"

    # -- knobs ------------------------------------------------------------- #
    def param(self, name: str, default: Any) -> Any:
        return self.params.get(name, default)

    def pick(self, name: str, default: Any, choices: Sequence[Any]) -> Any:
        """Use the provided knob, or draw one deterministically from ``choices``."""
        if name in self.params:
            val = self.params[name]
            return val if val in choices else default
        return self.rng.choice(list(choices))

    def injectable(self, ev: str) -> bool:
        return ev in set(self.target.caps.get("inject") or ())

    def conds(self, items) -> List[EventCondition]:
        """Keep only the event conditions this target can actually inject."""
        out = []
        for c in items:
            if c is None:
                continue
            if self.injectable(c.ev):
                out.append(c)
            else:
                self.params.setdefault("_skipped_conditions", []).append(c.ev)
        return out

    def pick_warps(self, name: str, default: int, choices) -> int:
        """``pick`` restricted to the target's warps-per-block limit."""
        limit = int(self.target.caps.get("warps_per_block", 32))
        usable = [w for w in choices if int(w) <= limit]
        if not usable:
            return min(default, limit)
        if name in self.params and int(self.params[name]) in usable:
            return int(self.params[name])
        return int(self.rng.choice(usable))

    # -- builders ---------------------------------------------------------- #
    def prog(self, smem: int = 0, name: str = "") -> ProgramBuilder:
        return ProgramBuilder(nregs=int(self.target.caps.get("max_regs", 32)),
                              gmem_base=0x1000, smem_bytes=smem,
                              name=name or ("%s-%s" % (self.target.name, self.shape.shape)))

    def config(self, p: ProgramBuilder, warps: int = 1, blocks: int = 1,
               conds: Optional[List[EventCondition]] = None,
               init: Optional[Sequence[Tuple[int, int]]] = None,
               launches: Optional[List[Launch]] = None,
               max_cycles: int = 200000, progress_timeout: int = 40000,
               extra: Optional[Dict[str, Any]] = None) -> Config:
        lanes = int(self.target.caps.get("lanes", 32))
        block = (lanes * warps, 1, 1)
        cfg = Config(
            grid=(blocks, 1, 1), block=block, warps_per_block=warps,
            regs_per_thread=int(self.target.caps.get("max_regs", 32)),
            smem_bytes=p.p.smem_bytes,
            gmem_base=p.gmem_base,
            gmem_init=list(init or ()),
            launches=list(launches or [Launch(grid=(blocks, 1, 1), block=block)]),
            conditions=list(conds or ()),
            max_cycles=max_cycles, progress_timeout=progress_timeout,
            extra=dict(extra or {}),
        )
        return cfg

    def plan(self, p: ProgramBuilder, cfg: Config, notes: str = "",
             **knobs: Any) -> GenPlan:
        # every test case must terminate: guarantee the terminal exit here so a
        # template can never hand an un-terminated program to a DUT
        if not p.p.instrs or p.p.instrs[-1].op != "exit":
            p.finish()
        return GenPlan(program=p.p, config=cfg, protected=tuple(p.protected),
                       knobs=dict(knobs), notes=notes)
