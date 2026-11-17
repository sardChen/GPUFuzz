"""Dependency preserving mutation (Section 4.3).

The mutator perturbs the program ``K`` and the configuration ``C`` while keeping
the conditions required to reach the target interaction state.  Two mechanisms
implement "preserving":

* **protected slots** -- the generator tags the instructions that establish the
  trigger condition (``TestCase.protected``); structural mutations never delete
  or rewrite those instructions;
* **rebinding** -- when a program mutation moves the trigger instruction, event
  anchors are re-bound to the new execution position instead of staying at a
  stale offset.

Every mutant is validated before it is returned; an invalid mutation yields
``None`` and the caller keeps the parent seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.config import Config, EventCondition, Launch
from common.ir import Instr, MemRef, Program, is_reg, reg_index
from common.targets import TargetSpec
from common.types import Relation, TestCase
from common.util import Log

# mutation kinds grouped by what they touch
PROGRAM_MUTATIONS = ("insert_filler", "delete_insn", "duplicate_insn", "change_imm",
                     "change_reg", "swap_insn", "extend_window", "latency_shift")
CONFIG_MUTATIONS = ("cfg_warps", "cfg_blocks", "cfg_backpressure", "cfg_event_param",
                    "cfg_init", "cfg_cycles", "cfg_anchor")
ALL_MUTATIONS = PROGRAM_MUTATIONS + CONFIG_MUTATIONS + ("rebind",)

NO_DELETE = ("label", "exit", "br", "jmp", "jmp_reg")


@dataclass
class MutationRecord:
    kinds: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    reverted: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {"kinds": self.kinds, "detail": self.detail, "reverted": self.reverted}


class Mutator:
    def __init__(self, target: TargetSpec, rng: Optional[random.Random] = None,
                 log: Optional[Log] = None) -> None:
        self.target = target
        self.rng = rng or random.Random(0xBEEF)
        self.log = log or Log("warn")

    # ------------------------------------------------------------------ #
    def mutate(self, tc: TestCase, rel: Relation, rounds: int = 1) -> Optional[TestCase]:
        """Return a mutated copy of ``tc`` or None if no valid mutant was found."""
        best: Optional[TestCase] = None
        for attempt in range(4):
            child = TestCase.from_json(tc.to_json())
            child.parent = tc.id
            child.generation = tc.generation + 1
            rec = MutationRecord()
            for _ in range(max(1, rounds)):
                kind = self._choose_kind(rel, child)
                fn = getattr(self, "_m_" + kind)
                try:
                    changed = fn(child, rel, rec)
                except Exception as exc:                     # pragma: no cover
                    self.log.debug("mutation %s failed: %s" % (kind, exc))
                    changed = False
                if changed:
                    rec.kinds.append(kind)
            child.provenance = dict(child.provenance)
            child.provenance["mutations"] = rec.to_json()
            if not rec.kinds:
                continue
            problems = self._validate(child)
            if problems:
                rec.reverted += 1
                continue
            self._rebind(child)
            child.id = "%s-m%d%s" % (tc.id.split("-m")[0], child.generation,
                                     child.program.hash()[:6])
            best = child
            break
        return best

    def _validate(self, tc: TestCase) -> List[str]:
        caps = self.target.caps
        problems = tc.program.validate(caps)
        problems += tc.config.validate(caps)
        return problems

    def _choose_kind(self, rel: Relation, tc: TestCase) -> str:
        kinds = [k for k in ALL_MUTATIONS if k != "rebind"]
        shape = rel.shape
        # weights: events/timing knobs of the targeted relation get more mass
        weights = []
        for k in kinds:
            w = 1.0
            if k in ("cfg_backpressure", "cfg_anchor") and any(
                    c.ev == "mem_backpressure" for c in tc.config.conditions):
                w = 3.0
            if k == "extend_window" and rel.omega.family == "progress":
                w = 2.5
            if k == "cfg_warps" and rel.ctx.get("same_warp"):
                w = 2.0
            if k == "latency_shift" and shape in ("SB_ISSUE_WB", "WB_ARB", "ROLLBACK_WB",
                                                  "SB_CLEAR_CONDITION"):
                w = 3.0
            if k == "change_reg" and rel.objvar.kind == "reg":
                w = 2.0
            weights.append(w)
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for k, w in zip(kinds, weights):
            acc += w
            if r <= acc:
                return k
        return kinds[-1]

    # ------------------------------------------------------------------ #
    # program mutations
    # ------------------------------------------------------------------ #
    def _free_reg(self, p: Program) -> str:
        used = set(reg_index(r) for r in
                   [x for i in p.instrs for x in i.regs_read() + i.regs_written()])
        for i in range(min(p.nregs - 1, 31), 12, -1):
            if i not in used:
                return "r%d" % i
        return "r27"

    def _protected_slots(self, tc: TestCase) -> set:
        return set(tc.protected)

    def _m_insert_filler(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        idx = self.rng.randrange(0, len(p.instrs))
        reg = self._free_reg(p)
        imm = self.rng.choice((1, 2, 3, 7, 0x11, 0x40))
        p.instrs.insert(idx, Instr(op="add", dst=reg, srcs=("r0",), imm=imm,
                                   slot="mut.filler"))
        rec.detail["insert_at"] = idx
        return True

    def _m_delete_insn(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        prot = self._protected_slots(tc)
        cands = [i for i, ins in enumerate(p.instrs)
                 if ins.slot not in prot and ins.op not in NO_DELETE]
        if len(cands) < 2:
            return False
        idx = self.rng.choice(cands)
        removed = p.instrs.pop(idx)
        rec.detail["deleted"] = removed.text()
        return True

    def _m_duplicate_insn(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        prot = self._protected_slots(tc)
        cands = [i for i, ins in enumerate(p.instrs)
                 if ins.slot not in prot and ins.op not in ("label", "exit", "br", "jmp",
                                                            "jmp_reg", "bar", "fence")]
        if not cands:
            return False
        idx = self.rng.choice(cands)
        p.instrs.insert(idx + 1, Instr.from_json(p.instrs[idx].to_json()))
        rec.detail["duplicated"] = p.instrs[idx].text()
        return True

    def _m_change_imm(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        prot = self._protected_slots(tc)
        cands = [i for i, ins in enumerate(p.instrs)
                 if ins.slot not in prot and ins.imm and ins.op not in NO_DELETE]
        if not cands:
            return False
        idx = self.rng.choice(cands)
        ins = p.instrs[idx]
        old = ins.imm
        if ins.op in ("mov_imm",) or (ins.mem is not None):
            ins.imm = self.rng.choice((0, 4, 8, 16, 64, 0xFF, 0x1000))
        else:
            ins.imm = self.rng.choice((1, 2, 3, 5, 9, 17))
        rec.detail["imm"] = [old, ins.imm]
        return True

    def _m_change_reg(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        prot = self._protected_slots(tc)
        dead = [i for i, ins in enumerate(p.instrs)
                if ins.dst and ins.slot not in prot and self._is_dead(p, i)]
        if not dead:
            return False
        idx = self.rng.choice(dead)
        new = self._free_reg(p)
        rec.detail["reg"] = [p.instrs[idx].dst, new]
        p.instrs[idx].dst = new
        return True

    @staticmethod
    def _is_dead(p: Program, idx: int) -> bool:
        dst = p.instrs[idx].dst
        if not dst:
            return False
        for ins in p.instrs[idx + 1:]:
            if dst in ins.regs_read():
                return False
            if dst in ins.regs_written():
                return True
        return True

    def _m_swap_insn(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        prot = self._protected_slots(tc)
        for _ in range(8):
            i = self.rng.randrange(0, max(1, len(p.instrs) - 1))
            a, b = p.instrs[i], p.instrs[i + 1]
            if a.op in ("label", "exit", "br", "bar") or b.op in ("label", "exit", "bar"):
                continue
            if a.slot in prot or b.slot in prot:
                continue
            if set(a.regs_written()) & (set(b.regs_read()) | set(b.regs_written())):
                continue
            p.instrs[i], p.instrs[i + 1] = b, a
            rec.detail["swap"] = [a.text(), b.text()]
            return True
        return False

    def _m_extend_window(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        p = tc.program
        prot = self._protected_slots(tc)
        positions = [i for i, ins in enumerate(p.instrs) if ins.slot in prot]
        if not positions:
            return False
        idx = min(len(p.instrs) - 1, max(positions) + 1)
        n = self.rng.choice((4, 8, 16, 32))
        p.instrs.insert(idx, Instr(op="spin", imm=n, slot="mut.extend"))
        rec.detail["spin"] = n
        return True

    def _m_latency_shift(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        """Lengthen/shorten a producer so its events move relative to each other."""
        p = tc.program
        long_ops = [i for i, ins in enumerate(p.instrs)
                    if ins.op in ("mul", "div", "fmul", "fdiv", "rem", "sat_add")]
        short_ops = [i for i, ins in enumerate(p.instrs) if ins.op == "add"]
        if long_ops and self.rng.random() < 0.5:
            idx = self.rng.choice(long_ops)
            ins = p.instrs[idx]
            alt = "div" if ins.op in ("mul", "rem") else "mul"
            if alt in ("div", "mul") and "int_div" not in self.target.caps["features"] and alt == "div":
                alt = "mul"
            rec.detail["latency"] = [ins.op, alt]
            ins.op = alt
            return True
        if short_ops:
            idx = self.rng.choice(short_ops)
            if p.instrs[idx].slot in self._protected_slots(tc):
                return False
            if "mul" not in self.target.caps["features"]:
                return False
            rec.detail["latency"] = ["add", "mul"]
            p.instrs[idx].op = "mul"
            return True
        return False

    # ------------------------------------------------------------------ #
    # configuration mutations
    # ------------------------------------------------------------------ #
    def _m_cfg_warps(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        caps = self.target.caps
        lanes = int(caps["lanes"])
        max_warps = int(caps["warps_per_block"])
        warps = tc.config.warps_per_block
        choices = [w for w in (1, 2, 4, 8, 16, 32) if w <= max_warps and w != warps]
        if not choices:
            return False
        new = self.rng.choice(choices)
        tc.config.warps_per_block = new
        tc.config.block = (lanes * new, 1, 1)
        tc.config.launches = [Launch(grid=tc.config.grid, block=tc.config.block, arg=l.arg)
                              for l in tc.config.launches]
        rec.detail["warps"] = [warps, new]
        return True

    def _m_cfg_blocks(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        maxb = min(16, int(self.target.caps.get("max_blocks", 64)))
        cur = tc.config.grid[0]
        choices = [b for b in (1, 2, 4, 8, 16) if b <= maxb and b != cur]
        if not choices:
            return False
        new = self.rng.choice(choices)
        tc.config.grid = (new, 1, 1)
        tc.config.launches = [Launch(grid=(new, 1, 1), block=l.block, arg=l.arg)
                              for l in tc.config.launches]
        rec.detail["blocks"] = [cur, new]
        return True

    def _m_cfg_backpressure(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        conds = [c for c in tc.config.conditions if c.ev == "mem_backpressure"]
        if not conds:
            return False
        c = self.rng.choice(conds)
        key = self.rng.choice(("hold", "start"))
        old = int(c.params.get(key, 0))
        if key == "hold":
            new = max(1, old + self.rng.choice((-8, -4, 4, 8, 16, 32)))
            new = min(new, 512)
        else:
            new = max(0, old + self.rng.choice((-2, -1, 1, 2, 4)))
        c.params[key] = new
        rec.detail["bp"] = [key, old, new]
        return True

    def _m_cfg_event_param(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        conds = tc.config.conditions
        if not conds:
            return False
        c = self.rng.choice(conds)
        keys = [k for k in c.params if isinstance(c.params[k], int)]
        if not keys:
            return False
        k = self.rng.choice(keys)
        old = c.params[k]
        delta = self.rng.choice((-2, -1, 1, 2, 4, 8))
        c.params[k] = max(0, old + delta)
        rec.detail["event_param"] = [c.ev, k, old, c.params[k]]
        return True

    def _m_cfg_init(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        cfg = tc.config
        if cfg.gmem_init:
            i = self.rng.randrange(len(cfg.gmem_init))
            addr, val = cfg.gmem_init[i]
            newval = self.rng.choice((0, 0xFFFFFFFF, 0xA5A5A5A5, 0x5A5A5A5A,
                                      val ^ 0xFFFF, val + 1))
            cfg.gmem_init[i] = (addr, newval)
            rec.detail["init"] = [addr, val, newval]
        else:
            addr = cfg.gmem_base + self.rng.choice((0, 64, 128))
            cfg.gmem_init.append((addr, self.rng.getrandbits(32)))
            rec.detail["init"] = [addr]
        return True

    def _m_cfg_cycles(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        old = tc.config.max_cycles
        tc.config.max_cycles = int(max(20000, old * self.rng.choice((0.5, 0.75, 1.5, 2.0))))
        rec.detail["max_cycles"] = [old, tc.config.max_cycles]
        return True

    def _m_cfg_anchor(self, tc: TestCase, rel: Relation, rec: MutationRecord) -> bool:
        conds = tc.config.conditions
        if not conds:
            return False
        c = self.rng.choice(conds)
        anchors = [a for a in ("kernel_start", "after_retire", "after_store",
                               "after_issue", "when_dirty", "when_pending_wb")
                   if a != c.anchor]
        c.anchor = self.rng.choice(anchors)
        rec.detail["anchor"] = [c.ev, c.anchor]
        return True

    # ------------------------------------------------------------------ #
    # rebinding
    # ------------------------------------------------------------------ #
    def _rebind(self, tc: TestCase) -> None:
        """Re-bind execution-relative anchors after the program shape changed.

        The paper's rule: if a program mutation changes when the target state
        becomes active, the event conditions must be rebound to the *new*
        execution position instead of a stale offset.
        """
        p = tc.program
        prot = self._protected_slots(tc)
        trigger_idx = [i for i, ins in enumerate(p.instrs) if ins.slot in prot]
        if not trigger_idx:
            return
        first_trigger = min(trigger_idx)
        first_store = next((i for i, ins in enumerate(p.instrs) if ins.op == "st"), None)
        for c in tc.config.conditions:
            if c.ev == "mem_backpressure" and c.anchor == "after_store" and first_store is not None:
                c.params["start"] = max(0, min(int(c.params.get("start", 0)),
                                               len(p.instrs) - first_store))
            if c.anchor in ("after_retire", "after_issue") and "after" in c.params:
                c.params["after"] = max(1, len(p.instrs) - first_trigger)
        tc.provenance["rebound_trigger_index"] = first_trigger
