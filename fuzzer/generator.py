"""Test case synthesis: ``T = <K, C>`` for a selected ESD relation.

The generator coordinates the SIMT program generator and the execution
configuration synthesizer (Figure 5, steps 1-2).  Every synthesized case is
validated against the target's capabilities *before* it can enter the seed pool:
SIMT execution constraints, memory access validity and injectable event support
are hard requirements, not warnings.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.config import Config, EventCondition, Launch
from common.esd.model import ESDModel
from common.ir import Program
from common.targets import TargetSpec
from common.types import Relation, TestCase
from common.util import Log, rng_for, sha, slug
from .builder import GenPlan
from .shapes import catalog, templates
from .shapes.genctx import GenCtx


class GenerationError(RuntimeError):
    pass


@dataclass
class GenStats:
    synthesized: int = 0
    rejected: int = 0
    repaired: int = 0
    per_shape: Dict[str, int] = field(default_factory=dict)
    problems: Dict[str, int] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"synthesized": self.synthesized, "rejected": self.rejected,
                "repaired": self.repaired, "per_shape": self.per_shape,
                "problems": self.problems}


class TestGenerator:
    def __init__(self, target: TargetSpec, model: ESDModel,
                 rng: Optional[random.Random] = None,
                 log: Optional[Log] = None) -> None:
        self.target = target
        self.model = model
        self.rng = rng or random.Random(0xC0FFEE)
        self.log = log or Log("warn")
        self.stats = GenStats()

    # ------------------------------------------------------------------ #
    def shape_of(self, rel: Relation) -> catalog.ShapeSpec:
        sh = catalog.SHAPES.get(rel.shape)
        if sh is None:
            raise GenerationError("relation %s has no shape spec (%r)"
                                  % (rel.rid, rel.shape))
        return sh

    def synthesize(self, rel: Relation, params: Optional[Dict[str, Any]] = None,
                   salt: Any = 0) -> TestCase:
        sh = self.shape_of(rel)
        last: Optional[List[str]] = None
        for attempt in range(4):
            g = GenCtx(target=self.target, rel=rel, shape=sh,
                       rng=rng_for(rel.rid, salt, attempt, params),
                       params=dict(params or {}), lanes=int(self.target.caps["lanes"]))
            plan = templates.generate(sh.gen, g)
            problems = self._validate(plan)
            if not problems:
                tc = self._make_testcase(rel, sh, plan, salt, attempt)
                self.stats.synthesized += 1
                self.stats.per_shape[sh.shape] = self.stats.per_shape.get(sh.shape, 0) + 1
                if attempt:
                    self.stats.repaired += 1
                return tc
            last = problems
            for p in problems:
                key = p.split(":")[0][:60]
                self.stats.problems[key] = self.stats.problems.get(key, 0) + 1
            # repair attempt: clamp resources / drop optional structure
            if not self._repair(plan, problems):
                break
        self.stats.rejected += 1
        raise GenerationError("cannot synthesize %s for %s: %s"
                              % (rel.rid, self.target.name, "; ".join(last or [])))

    # ------------------------------------------------------------------ #
    def _validate(self, plan: GenPlan) -> List[str]:
        caps = self.target.caps
        problems = plan.program.validate(caps)
        problems += plan.config.validate(caps)
        if plan.program.smem_bytes > 0 and int(caps.get("smem_bytes", 0)) == 0:
            problems.append("program uses shared memory but target has none")
        return problems

    def _repair(self, plan: GenPlan, problems: Sequence[str]) -> bool:
        """Best-effort repair of over-sized resources.  Returns True if changed."""
        changed = False
        caps = self.target.caps
        text = " ".join(problems)
        cfg: Config = plan.config
        if "warps" in text or "block" in text:
            max_warps = int(caps.get("warps_per_block", 32))
            if cfg.warps_per_block > max_warps:
                cfg.warps_per_block = max_warps
                cfg.block = (self.target.caps["lanes"] * max_warps, 1, 1)
                cfg.launches = [Launch(grid=cfg.grid, block=cfg.block)]
                changed = True
        if "max_blocks" in text:
            cfg.grid = (min(cfg.grid[0], int(caps.get("max_blocks", 1024))), 1, 1)
            changed = True
        if "smem" in text and plan.program.smem_bytes > 0:
            plan.program.smem_bytes = 0
            cfg.smem_bytes = 0
            for ins in plan.program.instrs:
                if ins.mem and ins.mem.space == "shared":
                    ins.mem = None  # pragma: no cover - shapes gate on smem
            changed = True
        return changed

    def _make_testcase(self, rel: Relation, sh: catalog.ShapeSpec,
                       plan: GenPlan, salt: Any, attempt: int) -> TestCase:
        tid = "%s-%s" % (slug(rel.shape, 24), plan.program.hash()[:10])
        return TestCase(
            id=tid, target=self.target.name, program=plan.program, config=plan.config,
            rid=rel.rid, protected=plan.protected,
            provenance={
                "shape": sh.shape, "generator": sh.gen, "spec_ref": sh.spec_ref,
                "knobs": plan.knobs, "notes": plan.notes, "salt": salt,
                "attempt": attempt, "omega": rel.omega.oid,
            },
            generation=0, parent=None)

    # ------------------------------------------------------------------ #
    def initial_seeds(self, per_relation: int = 2, shapes: Optional[Sequence[str]] = None,
                      limit: Optional[int] = None) -> List[TestCase]:
        """Seed pool for ``initialSeeds(M)``: every relation gets test cases."""
        seeds: List[TestCase] = []
        rels = [r for r in self.model if r.target == self.target.name]
        if shapes:
            rels = [r for r in rels if r.shape in set(shapes)]
        for rel in rels:
            for k in range(per_relation):
                try:
                    seeds.append(self.synthesize(rel, salt=k))
                except GenerationError as exc:
                    self.log.debug(str(exc))
                if limit and len(seeds) >= limit:
                    return seeds
        return seeds
