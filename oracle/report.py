"""Bug reports: bookkeeping, de-duplication and on-disk artifacts."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from common.types import TestCase
from common.util import append_jsonl, ensure_dir, read_json, table, write_json
from oracle.reproduce import ReproResult
from oracle.violation import Violation, dedup


@dataclass
class BugReport:
    report_id: str
    signature: str
    target: str
    rid: str
    oid: str
    family: str
    kind: str
    shape: str
    message: str
    detector: str
    occurrences: int = 1
    testcases: List[str] = field(default_factory=list)
    repro: Optional[ReproResult] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    violation: Optional[Violation] = None

    # -- status ------------------------------------------------------- #
    @property
    def status(self) -> str:
        if self.repro is None:
            return "unverified"
        if self.repro.reproduced:
            return "reproduced" if self.repro.deterministic else "reproduced-nondeterministic"
        return "not-reproduced"

    def to_json(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id, "signature": self.signature,
            "target": self.target, "rid": self.rid, "oid": self.oid,
            "family": self.family, "kind": self.kind, "shape": self.shape,
            "message": self.message, "detector": self.detector,
            "occurrences": self.occurrences, "testcases": self.testcases,
            "status": self.status,
            "repro": self.repro.to_json() if self.repro else None,
            "artifacts": self.artifacts,
            "violation": self.violation.to_json() if self.violation else None,
        }


class Reporter:
    def __init__(self, outdir: str, log: Any = None) -> None:
        self.outdir = ensure_dir(outdir)
        self.reports_dir = ensure_dir(os.path.join(outdir, "reports"))
        self.tc_dir = ensure_dir(os.path.join(outdir, "testcases"))
        self.log = log
        self._by_sig: Dict[str, BugReport] = {}
        self._order: List[str] = []

    # ------------------------------------------------------------------ #
    def add(self, v: Violation, tc: Optional[TestCase],
            repro: Optional[ReproResult] = None) -> BugReport:
        sig = v.signature()
        rep = self._by_sig.get(sig)
        if rep is None:
            rid = "BUG-%s-%02d" % (v.target.upper(), len(self._order) + 1)
            rep = BugReport(report_id=rid, signature=sig, target=v.target, rid=v.rid,
                            oid=v.oid, family=v.family, kind=v.kind, shape=v.shape,
                            message=v.message, detector=v.detector, violation=v)
            self._by_sig[sig] = rep
            self._order.append(sig)
        else:
            rep.occurrences += 1
        if tc and tc.id not in rep.testcases:
            rep.testcases.append(tc.id)
        if repro is not None:
            rep.repro = repro
            if repro.minimized is not None:
                rep.testcases.append(repro.minimized.id)
        if tc is not None:
            rep.artifacts = self._save(tc, v, repro, rep)
        return rep

    # ------------------------------------------------------------------ #
    def _save(self, tc: TestCase, v: Violation, repro: Optional[ReproResult],
              rep: BugReport) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        p = os.path.join(self.tc_dir, "%s.json" % tc.id)
        write_json(p, tc.to_json())
        paths["testcase"] = p
        vp = os.path.join(self.reports_dir, "%s.violation.json" % rep.report_id)
        write_json(vp, v.to_json())
        paths["violation"] = vp
        if repro is not None:
            rp = os.path.join(self.reports_dir, "%s.repro.json" % rep.report_id)
            write_json(rp, repro.to_json())
            paths["repro"] = rp
            if repro.minimized is not None:
                mp = os.path.join(self.tc_dir, "%s.json" % repro.minimized.id)
                write_json(mp, repro.minimized.to_json())
                paths["minimized"] = mp
        return paths

    # ------------------------------------------------------------------ #
    def reports(self) -> List[BugReport]:
        return [self._by_sig[s] for s in self._order]

    def table(self) -> str:
        rows = []
        for r in self.reports():
            rows.append([r.report_id, r.target, r.oid, r.family, r.shape,
                         str(r.occurrences), r.status])
        return table(rows, ["id", "target", "obligation", "family", "shape", "hits", "status"])

    def summary(self) -> Dict[str, Any]:
        reps = self.reports()
        by_target: Dict[str, int] = {}
        by_family: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for r in reps:
            by_target[r.target] = by_target.get(r.target, 0) + 1
            by_family[r.family] = by_family.get(r.family, 0) + 1
            by_status[r.status] = by_status.get(r.status, 0) + 1
        return {"total": len(reps), "by_target": by_target, "by_family": by_family,
                "by_status": by_status,
                "true_positives": by_status.get("reproduced", 0),
                "unverified": by_status.get("unverified", 0)}

    def write(self, extra: Optional[Dict[str, Any]] = None) -> str:
        data = {"summary": self.summary(), "reports": [r.to_json() for r in self.reports()],
                "extra": extra or {}}
        path = os.path.join(self.outdir, "bugs.json")
        write_json(path, data)
        txt = os.path.join(self.outdir, "bugs.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(self.table() + "\n\n")
            for r in self.reports():
                fh.write("%s  %s\n" % (r.report_id, r.message))
                fh.write("    rule=%s shape=%s testcases=%s status=%s\n"
                         % (r.oid, r.shape, ",".join(r.testcases[:4]), r.status))
        return path
