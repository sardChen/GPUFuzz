"""Functional consistency detector (Section 4.4).

Differential GPU testing: compare architectural effects against the reference
model, never cycle by cycle.  Addresses whose values derive from
architecturally-unknown sources (texture data) are skipped; registers are only
compared when the target's trace adapter actually dumps them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.types import ArchEffects, OMEGA, Relation, TestCase
from common.util import Log
from oracle.reference import ReferenceModel, ReferenceResult
from oracle.status_collector import Observation
from oracle.violation import Violation

MAX_DETAIL = 8


class FunctionalConsistencyDetector:
    def __init__(self, reference: ReferenceModel, log: Optional[Log] = None) -> None:
        self.reference = reference
        self.log = log or Log("warn")
        self.last_result: Optional[ReferenceResult] = None

    # ------------------------------------------------------------------ #
    def check(self, obs: Observation, tc: TestCase,
              rel: Optional[Relation] = None,
              allow_missing: bool = True) -> List[Violation]:
        ref = self.reference.run(tc)
        self.last_result = ref
        exp, got = ref.arch, obs.arch

        mismatches: List[Dict[str, Any]] = []
        missing: List[int] = []

        if not got.mem and not got.regs:
            # nothing was dumped: the oracle cannot say anything about A
            return []

        lo, hi = self._dump_range(obs, tc)
        for addr in sorted(exp.mem):
            if not (lo <= addr <= hi):
                continue
            if addr in ref.unpredictable:
                continue
            if self._is_param_block(addr, tc):
                continue
            e_val = exp.mem[addr]
            if addr not in got.mem:
                if allow_missing:
                    missing.append(addr)
                continue
            if int(got.mem[addr]) != int(e_val):
                mismatches.append({"addr": addr, "expected": e_val, "observed": int(got.mem[addr])})

        reg_mismatches: List[Dict[str, Any]] = []
        if got.regs:
            for key, e_val in sorted(exp.regs.items()):
                if key not in got.regs:
                    continue
                if int(got.regs[key]) != int(e_val):
                    reg_mismatches.append({"reg": key, "expected": e_val,
                                           "observed": int(got.regs[key])})

        status_bad = False
        if "exit" in exp.status and "exit" in got.status:
            status_bad = int(exp.status["exit"]) != int(got.status["exit"])

        if not (mismatches or missing or reg_mismatches or status_bad):
            return []

        oid = "OM_ARCH_MISMATCH"
        if rel is not None and rel.omega.kind == "functional":
            oid = rel.omega.oid
        first_cycle = 0
        msg_bits = []
        if mismatches:
            m = mismatches[0]
            msg_bits.append("memory[%#x] = %#x, expected %#x" % (m["addr"], m["observed"], m["expected"]))
        if missing:
            msg_bits.append("%d expected word(s) never written (e.g. %#x)" % (len(missing), missing[0]))
        if reg_mismatches:
            r = reg_mismatches[0]
            msg_bits.append("%s = %#x, expected %#x" % (r["reg"], r["observed"], r["expected"]))
        if status_bad:
            msg_bits.append("exit status %s, expected %s" % (got.status.get("exit"), exp.status.get("exit")))

        v = Violation(
            target=tc.target, rid=rel.rid if rel else "", oid=oid,
            message="architectural mismatch: " + "; ".join(msg_bits),
            family=OMEGA.get(oid, {}).get("family", "state_transition"),
            kind="functional", detector="functional", obj="mem:0", cycle=first_cycle,
            shape=rel.shape if rel else "",
            evidence={
                "mismatches": mismatches[:MAX_DETAIL],
                "missing_writes": [hex(a) for a in missing[:MAX_DETAIL]],
                "register_mismatches": reg_mismatches[:MAX_DETAIL],
                "n_mismatches": len(mismatches) + len(reg_mismatches),
                "n_missing": len(missing),
                "expected_exit": exp.status.get("exit"),
                "observed_exit": got.status.get("exit"),
                "unpredictable_addrs": [hex(a) for a in sorted(ref.unpredictable)[:4]],
            },
            testcase_id=tc.id)
        return [v]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dump_range(obs: Observation, tc: TestCase) -> Tuple[int, int]:
        rng = obs.trace.meta.get("mem_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            return int(rng[0]), int(rng[1])
        if obs.arch.mem:
            lo = min(obs.arch.mem)
            hi = max(obs.arch.mem)
            return lo, max(hi, lo + 0x1000)
        base = tc.config.gmem_base
        return base - 0x100, base + 0x10000

    @staticmethod
    def _is_param_block(addr: int, tc: TestCase) -> bool:
        base = tc.config.gmem_base + 0x7F0
        return base <= addr < base + 0x10
