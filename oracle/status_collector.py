"""Runtime status collector: build ``O = <A, E, S>`` from a DUT trace.

The collector is also the place where *observability* is decided.  A relation is
only checked when the trace actually carries the signals it needs; otherwise the
detector stays silent instead of inventing a violation.  That is what keeps the
oracle's false positive rate low when instrumentation is incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common.types import (ArchEffects, Event, Relation, StateObs, Trace,
                          obj_kind, obj_value)
from common.util import Log


@dataclass
class Observation:
    """``O = <A, E, S>`` with the indexes the rules need."""

    trace: Trace
    events: List[Event] = field(default_factory=list)
    states: List[StateObs] = field(default_factory=list)
    arch: ArchEffects = field(default_factory=ArchEffects)

    # derived indexes
    _by_obj: Dict[str, List[Event]] = field(default_factory=dict, repr=False)
    _by_state: Dict[Tuple[str, str], List[StateObs]] = field(default_factory=dict, repr=False)
    _state_ids: Set[str] = field(default_factory=set, repr=False)
    _event_types: Set[str] = field(default_factory=set, repr=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, trace: Trace) -> "Observation":
        events = sorted(trace.events, key=lambda e: (e.cycle, e.a, e.u))
        # drop exact duplicates produced by overlapping instrumentation taps
        seen = set()
        uniq: List[Event] = []
        for e in events:
            warp = e.ctx.env.get("warp") if e.ctx else None
            k = (e.a, e.u, e.o, e.cycle, warp)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(e)
        states = sorted(trace.states, key=lambda s: (s.cycle, s.sid, s.obj))
        obs = cls(trace=trace, events=uniq, states=states, arch=trace.arch)
        for e in uniq:
            obs._by_obj.setdefault(e.u, []).append(e)
        for s in states:
            obs._by_state.setdefault((s.sid, s.obj), []).append(s)
            obs._state_ids.add(s.sid)
        obs._event_types = {e.a for e in uniq}
        return obs

    # ------------------------------------------------------------------ #
    @property
    def target(self) -> str:
        return self.trace.target

    @property
    def cycles(self) -> int:
        return self.trace.cycles

    def events_of(self, types: Iterable[str], obj: Optional[str] = None,
                  min_cycle: int = -1, max_cycle: int = 1 << 62) -> List[Event]:
        ts = types if isinstance(types, (set, frozenset)) else set(types)
        pool = self._by_obj.get(obj, []) if obj is not None else self.events
        return [e for e in pool if e.a in ts and min_cycle <= e.cycle <= max_cycle]

    def states_of(self, sid: str, obj: Optional[str] = None,
                  min_cycle: int = -1, max_cycle: int = 1 << 62) -> List[StateObs]:
        if obj is not None:
            pool = self._by_state.get((sid, obj), [])
        else:
            pool = [s for s in self.states if s.sid == sid]
        return [s for s in pool if min_cycle <= s.cycle <= max_cycle]

    def has_state(self, sid: str) -> bool:
        return sid in self._state_ids

    def has_events(self, types: Iterable[str]) -> bool:
        return bool(set(types) & self._event_types)

    def objects_of_kind(self, kind: str) -> List[str]:
        out: List[str] = []
        for e in self.events:
            if obj_kind(e.u) == kind and e.u not in out:
                out.append(e.u)
        for s in self.states:
            if obj_kind(s.obj) == kind and s.obj not in out:
                out.append(s.obj)
        return out

    # ------------------------------------------------------------------ #
    def series(self, sid: str, obj: str, field: Optional[str] = None,
               collapse: bool = True) -> List[Tuple[int, Any]]:
        """Value series of one state element.

        With ``collapse`` (the default) several samples of the same element in the
        same cycle are reduced to the last one: what matters is the value the
        element holds *at the end of* the cycle, not the transient order of the
        taps that recorded it.  A design that momentarily clears a bit and then
        re-sets it within the same cycle is correct; one that leaves it cleared
        is not.
        """
        out: List[Tuple[int, Any]] = []
        for s in self.states_of(sid, obj):
            val = s.val
            if field and isinstance(val, dict):
                val = val.get(field)
            if collapse and out and out[-1][0] == s.cycle:
                out[-1] = (s.cycle, val)
                continue
            out.append((s.cycle, val))
        return out

    def transitions(self, sid: str, obj: str, field: Optional[str] = None
                    ) -> List[Tuple[int, Any, Any]]:
        """[(cycle, old, new)] for a state element's end-of-cycle value series."""
        ser = self.series(sid, obj, field, collapse=True)
        out = []
        for i in range(1, len(ser)):
            if ser[i][1] != ser[i - 1][1]:
                out.append((ser[i][0], ser[i - 1][1], ser[i][1]))
        return out

    def first(self, types: Iterable[str], obj: Optional[str] = None,
              after: int = -1) -> Optional[Event]:
        evs = self.events_of(types, obj, min_cycle=after + 1)
        return evs[0] if evs else None

    def count(self, types: Iterable[str], obj: Optional[str] = None) -> int:
        return len(self.events_of(types, obj))

    def kernel_ended(self) -> bool:
        return bool(self.events_of(("kernel_end",))) or bool(self.arch.status.get("exit") is not None
                                                             and not self.trace.timeout)

    def summary(self) -> Dict[str, Any]:
        return {
            "events": len(self.events), "states": len(self.states),
            "state_ids": sorted(self._state_ids), "event_types": sorted(self._event_types),
            "cycles": self.cycles, "timeout": self.trace.timeout,
            "arch_regs": len(self.arch.regs), "arch_mem": len(self.arch.mem),
        }


class RuntimeStatusCollector:
    """Validates and indexes the trace that both detectors consume."""

    def __init__(self, target: str, log: Optional[Log] = None) -> None:
        self.target = target
        self.log = log or Log("warn")
        self.observability: Dict[str, List[str]] = {}

    def collect(self, trace: Trace, relations: Optional[Iterable[Relation]] = None) -> Observation:
        obs = Observation.build(trace)
        if trace.target and trace.target != self.target:
            self.log.warn("trace target %r does not match oracle target %r"
                          % (trace.target, self.target))
        for rel in relations or ():
            missing = []
            for sid in rel.sids:
                if not obs.has_state(sid):
                    missing.append("state:" + sid)
            for role in rel.roles:
                if not obs.has_events(role.types):
                    missing.append("event:" + "|".join(role.types))
            if missing:
                self.observability[rel.rid] = missing
        return obs

    def report(self) -> Dict[str, List[str]]:
        return dict(self.observability)
