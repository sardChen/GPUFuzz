"""Violation records and their de-duplication signatures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.types import OMEGA
from common.util import sha


@dataclass
class Violation:
    """One reported inconsistency (a bug candidate)."""

    target: str
    rid: str
    oid: str
    message: str
    family: str = "state_transition"
    kind: str = "interaction"
    detector: str = "interaction"          # interaction | functional | progress
    obj: str = ""
    cycle: int = 0
    shape: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    testcase_id: str = ""

    def __post_init__(self) -> None:
        info = OMEGA.get(self.oid, {})
        if not self.family:
            self.family = info.get("family", "state_transition")
        if not self.kind:
            self.kind = info.get("kind", "interaction")

    # ------------------------------------------------------------------ #
    def signature(self) -> str:
        """Root-cause-ish key used for de-duplication.

        Rule id + object *class* + shape: two reports that violate the same rule
        on the same kind of hardware object are treated as one root cause, even
        when they were found by different test cases at different addresses.
        """
        obj_class = self.obj.split(":", 1)[0] if ":" in self.obj else self.obj
        return "%s|%s|%s|%s" % (self.target, self.oid, obj_class, self.shape)

    def uid(self) -> str:
        return sha(self.signature(), self.detector, n=10)

    def describe(self) -> str:
        loc = (" %s@%d" % (self.obj, self.cycle)) if self.obj or self.cycle else ""
        return "[%s/%s] %s%s: %s" % (self.detector, self.family, self.oid, loc, self.message)

    def to_json(self) -> Dict[str, Any]:
        return {
            "target": self.target, "rid": self.rid, "oid": self.oid,
            "family": self.family, "kind": self.kind, "detector": self.detector,
            "obj": self.obj, "cycle": self.cycle, "shape": self.shape,
            "message": self.message, "evidence": self.evidence,
            "testcase_id": self.testcase_id, "uid": self.uid(),
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Violation":
        return Violation(target=d.get("target", ""), rid=d.get("rid", ""),
                         oid=d.get("oid", ""), message=d.get("message", ""),
                         family=d.get("family", "state_transition"),
                         kind=d.get("kind", "interaction"),
                         detector=d.get("detector", "interaction"),
                         obj=d.get("obj", ""), cycle=int(d.get("cycle", 0)),
                         shape=d.get("shape", ""), evidence=dict(d.get("evidence") or {}),
                         testcase_id=d.get("testcase_id", ""))


def dedup(violations: List[Violation]) -> List[Violation]:
    """Keep one report per signature, merging the evidence of duplicates."""
    out: Dict[str, Violation] = {}
    for v in violations:
        sig = v.signature()
        if sig not in out:
            copy = Violation.from_json(v.to_json())
            copy.evidence = dict(copy.evidence)
            copy.evidence["occurrences"] = 1
            copy.evidence["testcases"] = [v.testcase_id] if v.testcase_id else []
            out[sig] = copy
        else:
            keep = out[sig]
            keep.evidence["occurrences"] = int(keep.evidence.get("occurrences", 1)) + 1
            tcs = keep.evidence.setdefault("testcases", [])
            if v.testcase_id and v.testcase_id not in tcs:
                tcs.append(v.testcase_id)
    return sorted(out.values(), key=lambda v: (v.target, v.oid, v.obj))
