"""Driver interface: how the fuzzer executes one test case on a DUT.

The Verilator driver in :mod:`adapters.verilator.driver` is the real
implementation; the fixture driver used by the self tests lives in
``tests/fixtures``.  Both satisfy this interface, so the fuzzing loop is identical
whether it talks to RTL or to a recorded trace.
"""
from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Optional

from common.targets import TargetSpec
from common.types import RunResult, TestCase


class Driver(abc.ABC):
    name: str = "abstract"

    def __init__(self, target: TargetSpec, **kw: Any) -> None:
        self.target = target
        self.kw = kw

    @abc.abstractmethod
    def execute(self, tc: TestCase) -> RunResult:
        """Run ``tc`` from reset and return the trace (or an error)."""

    def prepare(self) -> Dict[str, Any]:
        """Optional one-off setup (elaborate/compile).  Returns a status dict."""
        return {"ok": True}

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return self.name


class NullDriver(Driver):
    """Fails every execution with an explanation instead of pretending to run."""

    name = "null"

    def __init__(self, target: TargetSpec, reason: str = "no driver configured", **kw: Any) -> None:
        super().__init__(target, **kw)
        self.reason = reason

    def execute(self, tc: TestCase) -> RunResult:
        return RunResult(ok=False, driver=self.name, error=self.reason)


_REGISTRY: Dict[str, Callable[..., Driver]] = {}


def register_driver(kind: str, factory: Callable[..., Driver]) -> None:
    _REGISTRY[kind] = factory


def available() -> Dict[str, Callable[..., Driver]]:
    _ensure_builtin()
    return dict(_REGISTRY)


def create_driver(kind: str, target: TargetSpec, **kw: Any) -> Driver:
    _ensure_builtin()
    if kind not in _REGISTRY:
        raise ValueError("unknown driver %r (available: %s)"
                         % (kind, ", ".join(sorted(_REGISTRY))))
    return _REGISTRY[kind](target, **kw)


_BUILTINS_LOADED = False


def _ensure_builtin() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        from adapters.verilator.driver import VerilatorDriver
        register_driver("verilator", VerilatorDriver)
    except Exception:                                   # pragma: no cover
        pass
    try:
        from adapters.trace.driver import TraceReplayDriver
        register_driver("trace", TraceReplayDriver)
    except Exception:                                   # pragma: no cover
        pass
    try:
        from tests.fixtures.driver import FixtureDriver
        register_driver("fixture", FixtureDriver)
    except Exception:                                   # pragma: no cover
        pass
