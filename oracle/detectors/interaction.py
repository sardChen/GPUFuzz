"""State interaction consistency detector (Section 4.4).

For every *activated* relation with a justified obligation, the detector runs the
matching rule from :mod:`oracle.rules`.  Relations whose obligation is ``⊥`` are
skipped: they guide exploration but are excluded from correctness checking.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from common.esd.match import Binding
from common.types import Relation, TestCase
from common.util import Log
from oracle.rules import REGISTRY
from oracle.status_collector import Observation
from oracle.violation import Violation


class StateInteractionConsistencyDetector:
    def __init__(self, target: str, log: Optional[Log] = None) -> None:
        self.target = target
        self.log = log or Log("warn")
        self.skipped: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def check(self, obs: Observation, tc: TestCase,
              activations: Sequence[Binding]) -> List[Violation]:
        out: List[Violation] = []
        covered_oids: set = set()
        for b in activations:
            rel = b.rel
            if rel.target != self.target:
                continue
            if not rel.checkable:
                self.skipped[rel.rid] = "obligation is ⊥ (exploration only)"
                continue
            if rel.omega.kind == "functional":
                # architectural obligations belong to the functional detector
                continue
            fn = REGISTRY.get(rel.omega.oid)
            if fn is None:
                self.skipped[rel.rid] = "no checker for %s" % rel.omega.oid
                continue
            found = fn(b, obs, tc)
            for v in found:
                v.testcase_id = tc.id
            out.extend(found)
            covered_oids.add(rel.omega.oid)
        out.extend(self._global_rules(obs, tc, activations))
        return out

    # ------------------------------------------------------------------ #
    def _global_rules(self, obs: Observation, tc: TestCase,
                      activations: Sequence[Binding]) -> List[Violation]:
        """Rules that must fire even when the relation never activated.

        A hang cannot satisfy ``mem_req << kernel_end`` (there is no kernel_end),
        so forward progress needs an entry point that does not depend on
        activation.
        """
        if not obs.trace.timeout:
            return []
        rel = self._progress_relation(tc, activations)
        if rel is None:
            return []
        b = Binding(rel=rel, obj="kernel:0", events={})
        from oracle.rules.checkers import check_no_hang
        found = check_no_hang(b, obs, tc)
        for v in found:
            v.testcase_id = tc.id
        return found

    def _progress_relation(self, tc: TestCase,
                           activations: Sequence[Binding]) -> Optional[Relation]:
        for b in activations:
            if b.rel.omega.oid in ("OM_PROGRESS", "OM_NO_HANG", "OM_DIV_COMPLETION"):
                return b.rel
        for b in activations:
            if b.rel.omega.family == "progress":
                return b.rel
        return None
