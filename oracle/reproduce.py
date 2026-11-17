"""Bug reproduction and minimization (Section 4.4, *Bug Reproduction and
De-duplication*).

A candidate is replayed from reset; only candidates whose violated rule
reappears deterministically are kept.  Reproduced candidates are then minimized
by removing instructions while the same rule keeps firing, which is what makes a
report actionable for a hardware engineer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.esd.model import ESDModel
from common.ir import Instr
from common.types import TestCase, Trace, RunResult
from common.util import Log, write_json
from oracle.oracle import ConsistencyOracle, OracleResult


@dataclass
class ReproResult:
    signature: str = ""
    reproduced: bool = False
    runs: int = 0
    hits: int = 0
    deterministic: bool = False
    minimized: Optional[TestCase] = None
    removed: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"signature": self.signature, "reproduced": self.reproduced,
                "runs": self.runs, "hits": self.hits,
                "deterministic": self.deterministic, "removed": self.removed,
                "minimized": self.minimized.to_json() if self.minimized else None,
                "detail": self.detail}


class BugReproducer:
    def __init__(self, driver: Any, oracle: ConsistencyOracle, model: ESDModel,
                 log: Optional[Log] = None) -> None:
        self.driver = driver
        self.oracle = oracle
        self.model = model
        self.log = log or Log("warn")

    # ------------------------------------------------------------------ #
    def _run(self, tc: TestCase) -> Tuple[RunResult, OracleResult]:
        res = self.driver.execute(tc)
        if not res.ok or res.trace is None:
            return res, OracleResult(notes=["driver failed: " + res.error])
        out = self.oracle.check(res.trace, tc)
        return res, out

    # ------------------------------------------------------------------ #
    def verify(self, tc: TestCase, signature: str, runs: int = 3,
               minimize: bool = True) -> ReproResult:
        rep = ReproResult(signature=signature, runs=runs)
        for i in range(runs):
            res, out = self._run(tc)
            if not res.ok:
                rep.detail.setdefault("errors", []).append(res.error)
                continue
            if any(v.signature() == signature for v in out.violations):
                rep.hits += 1
        rep.reproduced = rep.hits >= max(2, runs - 1)
        rep.deterministic = rep.hits == runs
        if rep.reproduced and minimize:
            mini, removed = self.minimize(tc, signature)
            rep.minimized = mini
            rep.removed = removed
        return rep

    # ------------------------------------------------------------------ #
    def minimize(self, tc: TestCase, signature: str,
                 max_rounds: int = 40) -> Tuple[Optional[TestCase], int]:
        """Delta-debug the program while the same rule keeps firing."""
        best = TestCase.from_json(tc.to_json())
        best.provenance = dict(best.provenance)
        best.provenance["minimized_from"] = tc.id
        protected = set(tc.protected)
        total_removed = 0

        for _round in range(max_rounds):
            candidate_idx = [
                i for i, ins in enumerate(best.program.instrs)
                if ins.slot not in protected and ins.op not in ("label", "exit")
            ]
            if not candidate_idx:
                break
            chunk = max(1, len(candidate_idx) // 2)
            removed_any = False
            for start in range(0, len(candidate_idx), chunk):
                trial = TestCase.from_json(best.to_json())
                drop = set(candidate_idx[start:start + chunk])
                trial.program.instrs = [ins for i, ins in enumerate(trial.program.instrs)
                                        if i not in drop]
                if not trial.program.instrs:
                    continue
                if trial.program.validate(self.driver.target.caps if hasattr(self.driver, "target")
                                          else None):
                    continue
                res, out = self._run(trial)
                if not res.ok:
                    continue
                if any(v.signature() == signature for v in out.violations):
                    best = trial
                    total_removed += len(drop)
                    removed_any = True
                    break
            if not removed_any:
                break
        best.id = tc.id + "-min"
        return best, total_removed
