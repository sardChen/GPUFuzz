"""Trace replay driver: feed recorded traces back through the fuzzer/oracle.

Useful for debugging a campaign without rebuilding RTL: put ``<testcase id>.jsonl``
files in a directory and this driver replays them.  It never invents a trace for a
test case it has no recording for -- missing ids are reported as failures.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from adapters.trace.adapter import TraceAdapter
from common.driver import Driver
from common.targets import TargetSpec
from common.types import RunResult, TestCase
from common.util import Log


class TraceReplayDriver(Driver):
    name = "trace"

    def __init__(self, target: TargetSpec, tracdir: str = "traces",
                 log: Optional[Log] = None, **kw: Any) -> None:
        super().__init__(target, **kw)
        self.dir = os.path.abspath(tracdir)
        self.log = log or Log("warn")
        self.adapter = TraceAdapter(target.name, target.signal_map(), log=self.log)

    def execute(self, tc: TestCase) -> RunResult:
        path = os.path.join(self.dir, tc.id + ".jsonl")
        if not os.path.exists(path):
            return RunResult(ok=False, driver=self.name,
                             error="no recorded trace for %s in %s" % (tc.id, self.dir))
        trace = self.adapter.parse(path, tc.id)
        return RunResult(ok=True, trace=trace, driver=self.name, artifact_dir=self.dir,
                         raw={"trace": path})
