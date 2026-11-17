"""Fixture driver: executes a test case on the synthetic trace synthesizer.

Two modes:

* ``faults=[]``  -- a **clean** trace.  The oracle must report nothing.  This is
  the false-positive regression test.
* ``faults=[...]`` -- the same trace with one bug's signature applied.  The
  oracle must report the corresponding obligation.

This is the only backend that runs without RTL; it produces no coverage and no
timing data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from common.driver import Driver
from common.targets import TargetSpec
from common.types import RunResult, TestCase
from tests.fixtures import faults as fault_lib
from tests.fixtures.synth import SynthCtx, TraceSynthesizer, finish


def default_faults(tc: TestCase) -> List[str]:
    shape = (tc.provenance or {}).get("shape", "")
    entry = fault_lib.SHAPE_FAULT.get(shape)
    return [entry[0]] if entry else []


def expected_obligation(tc: TestCase) -> Optional[str]:
    shape = (tc.provenance or {}).get("shape", "")
    entry = fault_lib.SHAPE_FAULT.get(shape)
    return entry[1] if entry else None


class FixtureDriver(Driver):
    name = "fixture"

    def __init__(self, target: TargetSpec, faults: Optional[Sequence[str]] = None,
                 clean: bool = False, latency_scale: float = 1.0,
                 keep_regs: bool = False, **kw: Any) -> None:
        super().__init__(target, **kw)
        self.faults = list(faults) if faults is not None else None
        self.clean = clean
        self.latency_scale = latency_scale
        self.keep_regs = keep_regs
        self.synth = TraceSynthesizer(target, latency_scale=latency_scale)

    def execute(self, tc: TestCase) -> RunResult:
        try:
            ctx = self.synth.build(tc)
        except Exception as exc:                       # pragma: no cover
            return RunResult(ok=False, driver=self.name,
                             error="synthesis failed: %r" % (exc,))
        applied: List[str] = []
        if not self.clean:
            applied = list(self.faults) if self.faults is not None else default_faults(tc)
            if applied:
                fault_lib.apply(ctx, applied)
        if not self.keep_regs:
            ctx.arch.regs = {}
        trace = finish(ctx)
        trace.meta["faults"] = applied
        return RunResult(ok=True, trace=trace, driver=self.name,
                         raw={"faults": applied})
