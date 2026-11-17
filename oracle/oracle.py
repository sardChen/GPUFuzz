"""The consistency oracle: one entry point for both detectors.

    Oracle.check(trace, testcase) -> OracleResult

The oracle is stateless with respect to the campaign; the fuzzer passes the
activations it already computed so the model is matched once per execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from common.esd.match import Binding, activate
from common.esd.model import ESDModel
from common.targets import TargetSpec
from common.types import Relation, TestCase, Trace
from common.util import Log
from oracle.detectors.functional import FunctionalConsistencyDetector
from oracle.detectors.interaction import StateInteractionConsistencyDetector
from oracle.reference import ReferenceModel
from oracle.status_collector import Observation, RuntimeStatusCollector
from oracle.violation import Violation


@dataclass
class OracleResult:
    violations: List[Violation] = field(default_factory=list)
    activations: List[Binding] = field(default_factory=list)
    observability: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    checked_relations: int = 0
    functional_checked: bool = False
    interaction_checked: bool = False

    @property
    def clean(self) -> bool:
        return not self.violations

    def families(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.violations:
            out[v.family] = out.get(v.family, 0) + 1
        return out

    def describe(self) -> str:
        if not self.violations:
            return "no violation (%d relations checked)" % self.checked_relations
        return "%d violation(s): %s" % (
            len(self.violations), ", ".join(sorted({v.oid for v in self.violations})))


class ConsistencyOracle:
    def __init__(self, target: TargetSpec, model: ESDModel,
                 log: Optional[Log] = None) -> None:
        self.target = target
        self.model = model
        self.log = log or Log("warn")
        self.collector = RuntimeStatusCollector(target.name, log=self.log)
        self.reference = ReferenceModel(target)
        self.functional = FunctionalConsistencyDetector(self.reference, log=self.log)
        self.interaction = StateInteractionConsistencyDetector(target.name, log=self.log)

    # ------------------------------------------------------------------ #
    def check(self, trace: Trace, tc: TestCase,
              activations: Optional[Sequence[Binding]] = None) -> OracleResult:
        rels = [r for r in self.model if r.target == self.target.name]
        obs = self.collector.collect(trace, rels)
        acts = list(activations) if activations is not None else [
            a for r in rels for a in activate(r, trace, max_matches=8)]

        result = OracleResult(activations=list(acts),
                              observability=self.collector.report())
        if not obs.events:
            result.notes.append("trace has no events: no detector can run")
            return result

        # targeted relation: the one the test case was synthesized for
        targeted = next((r for r in rels if r.rid == tc.rid), None)

        # The caller may have passed the activations of the relation it selected
        # for mutation, which is not necessarily the relation the test case was
        # built for.  Always include the targeted relation, otherwise its
        # specific obligation would be silently skipped and only the generic
        # architectural symptom would be reported.
        if targeted is not None and not any(b.rel.rid == targeted.rid for b in acts):
            acts = list(acts) + activate(targeted, trace, max_matches=8)
            result.activations = list(acts)

        # ---- architectural consistency ---------------------------------- #
        fv = self.functional.check(obs, tc, rel=targeted,
                                   allow_missing=not trace.timeout)
        result.functional_checked = True
        result.violations.extend(fv)

        # ---- state interaction consistency ------------------------------ #
        iv = self.interaction.check(obs, tc, acts)
        result.interaction_checked = True
        result.violations.extend(iv)
        result.checked_relations = sum(1 for b in acts if b.rel.checkable)

        if trace.timeout:
            result.notes.append("execution hit the cycle limit")
        if not obs.arch.mem and not obs.arch.regs:
            result.notes.append("no architectural dump: functional check skipped")

        # de-duplicate within one execution (same rule, same object class)
        seen: Dict[str, Violation] = {}
        for v in result.violations:
            seen.setdefault(v.signature(), v)
        result.violations = list(seen.values())

        # Some rules report a *symptom* rather than a root cause: an
        # architectural mismatch (OM_ARCH_MISMATCH) or a kernel that never
        # terminates (OM_NO_HANG).  When the same execution already violated a
        # more specific obligation, keep that one and fold the symptom into its
        # evidence, so one root cause yields one bug report.
        GENERIC = ("OM_ARCH_MISMATCH", "OM_NO_HANG")
        specific = [v for v in result.violations if v.oid not in GENERIC]
        generic = [v for v in result.violations if v.oid in GENERIC]
        if specific and generic:
            keep = specific[0]
            keep.evidence = dict(keep.evidence)
            keep.evidence["symptoms"] = [{"oid": g.oid, "message": g.message,
                                          "evidence": g.evidence} for g in generic]
            result.violations = specific
        return result

    # ------------------------------------------------------------------ #
    def describe_coverage_of_checking(self) -> str:
        oid = sorted({r.omega.oid for r in self.model
                      if r.target == self.target.name and r.checkable})
        return "%s: %d checkable obligations (%s)" % (self.target.name, len(oid), ", ".join(oid))
