"""Activation of ESD relations against a runtime observation.

Section 4.2 defines

    Active(r, O)  <=>  Related_O(c, e, s)  AND  Time_O(e) |= theta

``Related_O`` requires that the observed events influence the associated state
*for the same hardware object* -- merely co-occurring events do not activate a
relation.  ``Time_O |= theta`` is checked on the events' relative timestamps, so
the same relation stays valid across different DUT timings.

The functions here are shared by the fuzzer (to derive state interaction
feedback ``X``) and by the oracle (to decide which obligations are in scope).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common.types import Event, Relation, StateObs, Trace, obj_kind

MAX_CANDIDATES_PER_ROLE = 6
MAX_COMBINATIONS = 400


@dataclass
class Binding:
    """One concrete activation of a relation: concrete object + events."""

    rel: Relation
    obj: str
    events: Dict[str, Optional[Event]]
    states: List[StateObs] = field(default_factory=list)
    guard_hits: Dict[str, int] = field(default_factory=dict)

    def ev(self, role: str) -> Optional[Event]:
        return self.events.get(role)

    @property
    def lo(self) -> int:
        return min(e.cycle for e in self.events.values() if e is not None)

    @property
    def hi(self) -> int:
        return max(e.cycle for e in self.events.values() if e is not None)

    def state_values(self, sid: str) -> List[StateObs]:
        return [s for s in self.states if s.sid == sid]

    def describe(self) -> str:
        evs = ", ".join("%s=%s@%d" % (r, e.a, e.cycle if e else -1)
                        for r, e in sorted(self.events.items()))
        return "%s obj=%s {%s} window=[%d,%d]" % (self.rel.rid, self.obj, evs, self.lo, self.hi)

    def key(self) -> str:
        return "%s|%s|%s" % (self.rel.rid, self.obj,
                             ",".join("%s@%d" % (r, e.cycle if e else -1)
                                      for r, e in sorted(self.events.items())))


def _role_candidates(rel: Relation, role_name: str, trace: Trace, obj: str) -> List[Event]:
    role = rel.role(role_name)
    out: List[Event] = []
    types = set(role.types)
    for ev in trace.events:
        if ev.a not in types:
            continue
        if role.obj is not None and ev.u != obj:
            continue
        if role.outcomes and ev.o not in role.outcomes:
            continue
        out.append(ev)
    out.sort(key=lambda e: e.cycle)
    return out[:MAX_CANDIDATES_PER_ROLE]


def _check_atoms(rel: Relation, events: Dict[str, Event]) -> bool:
    for atom in rel.theta.atoms:
        a = events.get(atom.lhs)
        b = events.get(atom.rhs)
        if a is None or b is None:
            # a missing *optional* role cannot be ordered against anything; the
            # obligation itself has to report the absence
            return True
        if atom.kind == "before":
            ok = a.cycle <= b.cycle
        elif atom.kind == "after":
            ok = a.cycle >= b.cycle
        elif atom.kind == "same_cycle":
            ok = a.cycle == b.cycle
        elif atom.kind == "coincide":
            ok = abs(a.cycle - b.cycle) <= max(0, atom.param)
        elif atom.kind == "overlap":
            ok = a.cycle <= b.cycle <= a.cycle + max(0, atom.param)
        elif atom.kind == "within":
            ok = 0 <= (b.cycle - a.cycle) <= max(0, atom.param)
        elif atom.kind == "not_before":
            ok = not (a.cycle < b.cycle)
        else:  # pragma: no cover - guarded by Atom validation
            ok = True
        if not ok:
            return False
    return True


def _ctx_ok(rel: Relation, events: Dict[str, Event], trace: Trace) -> bool:
    ctx = rel.ctx or {}
    if "min_warps" in ctx:
        warps = int(trace.meta.get("warps", 0) or 0)
        if warps and warps < int(ctx["min_warps"]):
            return False
    if "min_blocks" in ctx:
        blocks = int(trace.meta.get("blocks", 0) or 0)
        if blocks and blocks < int(ctx["min_blocks"]):
            return False
    present = [e for e in events.values() if e is not None]
    if ctx.get("same_warp"):
        warps = {e.ctx.env.get("warp") for e in present if e.ctx}
        warps.discard(None)
        if len(warps) > 1:
            return False
    if "kernel_index" in ctx:
        idxs = {e.ctx.env.get("kernel") for e in present if e.ctx}
        if idxs and int(ctx["kernel_index"]) not in {int(i) for i in idxs if i is not None}:
            return False
    return True


def _collect_states(rel: Relation, obj: str, lo: int, hi: int, trace: Trace,
                    extra_objs: Set[str]) -> List[StateObs]:
    sids = set(rel.sids)
    slack = 64
    out = []
    for s in trace.states:
        if s.sid not in sids:
            continue
        if s.obj != obj and s.obj not in extra_objs:
            continue
        if lo - slack <= s.cycle <= hi + slack:
            out.append(s)
    out.sort(key=lambda s: s.cycle)
    return out


def match_relation(rel: Relation, trace: Trace,
                   max_matches: int = 16) -> List[Binding]:
    """Return every activation of ``rel`` in ``trace`` (bounded)."""
    if not trace.events:
        return []

    role_names = [r.role for r in rel.roles]
    objvar_kind = rel.objvar.kind

    # candidate objects: anything of the right kind seen in events or states
    objs: List[str] = []
    for ev in trace.events:
        if obj_kind(ev.u) == objvar_kind and ev.u not in objs:
            objs.append(ev.u)
    for st in trace.states:
        if obj_kind(st.obj) == objvar_kind and st.obj not in objs:
            objs.append(st.obj)
    if not objs:
        return []

    matches: List[Binding] = []
    for obj in objs:
        cand = {r: _role_candidates(rel, r, trace, obj) for r in role_names}
        missing_required = [r for r in role_names
                            if not cand[r] and not rel.role(r).optional]
        if missing_required:
            continue
        for r in role_names:
            if not cand[r]:
                cand[r] = [None]        # optional role: absent
        # bound the search; roles with the fewest candidates first
        ordered = sorted(role_names, key=lambda r: len(cand[r]))
        combos = itertools.product(*[cand[r] for r in ordered])
        tried = 0
        for combo in combos:
            tried += 1
            if tried > MAX_COMBINATIONS:
                break
            events = dict(zip(ordered, combo))
            bound = [e for e in events.values() if e is not None]
            # two roles must never bind the same event instance
            if len({id(e) for e in bound}) != len(bound):
                continue
            if not bound:
                continue
            if not _check_atoms(rel, events):
                continue
            if not _ctx_ok(rel, events, trace):
                continue
            lo = min(e.cycle for e in bound)
            hi = max(e.cycle for e in bound)
            extra = {e.u for e in bound}
            states = _collect_states(rel, obj, lo, hi, trace, extra)
            matches.append(Binding(rel=rel, obj=obj, events=events, states=states))
            if len(matches) >= max_matches:
                return matches
    return matches


def is_related(rel: Relation, binding: Binding) -> bool:
    """``Related_O(c, e, s)``: the events must touch the same object.

    ``obj == ev.u`` for every bound role already enforces this in
    :func:`match_relation`; this hook keeps the predicate explicit and lets a
    target adapter add object-equivalence classes (e.g. MIAOW wavefronts that
    alias after rescheduling).
    """
    for ev in binding.events.values():
        if ev is None:
            continue
        if ev.u and ev.u != binding.obj and obj_kind(ev.u) == rel.objvar.kind:
            return False
    return True


def activate(rel: Relation, trace: Trace, max_matches: int = 16) -> List[Binding]:
    """``Active(r, O)``: related and temporally consistent activations."""
    return [b for b in match_relation(rel, trace, max_matches) if is_related(rel, b)]


def first_activation(rel: Relation, trace: Trace) -> Optional[Binding]:
    acts = activate(rel, trace, max_matches=1)
    return acts[0] if acts else None


def interaction_key(rel: Relation, b: Binding) -> str:
    """Stable key of a state interaction feedback item ``<context, interaction, state>``."""
    first = next((e for e in b.events.values() if e is not None), None)
    ctx = "|".join("%s=%s" % (k, first.ctx.env.get(k))
                   for k in ("warp",) if first and first.ctx) or "-"
    inter = ",".join("%s@%s" % (r, e.o if e else "absent") for r, e in sorted(b.events.items()))
    state_after = "|".join(
        "%s=%s" % (s.sid, _short(s.val)) for s in b.states[-3:]) or "nostate"
    return "%s|%s|%s|%s|%s" % (rel.target, rel.rid, rel.shape, ctx, inter + ">" + state_after)


def _short(val: Any) -> str:
    s = str(val)
    return s if len(s) <= 24 else s[:21] + "..."


def context_of(b: Binding) -> Dict[str, Any]:
    """``chi`` of the activation, as recorded in the feedback item."""
    out: Dict[str, Any] = {}
    for role in sorted(b.events):
        ev = b.events[role]
        if ev is not None and ev.ctx:
            for k, v in ev.ctx.env.items():
                out.setdefault(k, v)
            out.setdefault("phase", ev.ctx.phase)
            break
    out.setdefault("obj", b.obj)
    return out
