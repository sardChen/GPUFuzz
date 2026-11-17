"""The Execution State Dependency Model ``M = (R, Freq)``.

``R`` is the relation set; ``Freq`` counts observed state interactions
``x = <context, interaction, state>``.  The model is built statically
(:mod:`fuzzer.esd_model`), refined online (Section 4.2, *online runtime
refinement*) and consumed by both the fuzzer (relation selection, feedback,
seed prioritisation) and the oracle (which obligations are in scope).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common.types import (BOT, Obligation, Relation, RelationSet, StateObs, Trace,
                     EventRole, Theta, Atom)
from common.util import read_json, table, weighted_choice, write_json
from .match import Binding, activate, interaction_key

# guard keys recorded per relation
GUARD_KEYS = ("overlap", "ordered", "same_cycle", "windows", "outcomes")


@dataclass
class RefinementReport:
    attempted: List[str] = field(default_factory=list)
    activated: List[str] = field(default_factory=list)
    new_relations: List[str] = field(default_factory=list)
    guards: Dict[str, List[str]] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"attempted": self.attempted, "activated": self.activated,
                "new_relations": self.new_relations, "guards": self.guards}


class ESDModel:
    """Relation set + interaction frequency map, with the refinement machinery."""

    def __init__(self, relations: Optional[Iterable[Relation]] = None,
                 target: str = "", freq: Optional[Dict[str, int]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        self.relations = RelationSet(relations)
        self.target = target
        self.freq: Dict[str, int] = dict(freq or {})
        self.meta: Dict[str, Any] = dict(meta or {})
        self._provisional = 0

    # ------------------------------------------------------------------ #
    # relation access
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.relations)

    def __iter__(self):
        return iter(self.relations)

    def get(self, rid: str) -> Relation:
        return self.relations.get(rid)

    def checkable(self) -> List[Relation]:
        return [r for r in self.relations if r.checkable]

    def exploratory(self) -> List[Relation]:
        return [r for r in self.relations if not r.checkable]

    def add(self, rel: Relation) -> None:
        self.relations.add(rel)

    # ------------------------------------------------------------------ #
    # Algorithm 1: selectTarget(M, Freq)
    # ------------------------------------------------------------------ #
    def relation_weight(self, rel: Relation) -> float:
        """Exploration weight for ``selectTarget``.

        Paper text: selection considers the runtime confidence kappa(r) and the
        occurrence frequency of previously observed interactions; repeated
        unsuccessful attempts reduce exploration priority, and provisional
        relations stay eligible.

        weight = (eps + kappa) * (1 + unseen_bonus) / (1 + freq(r))
        """
        kappa = rel.stats.kappa
        freq = self.freq_of_relation(rel)
        unseen = 1.0 if freq == 0 else 0.0
        provisional = 0.0 if rel.checkable else 0.15   # keep ⊥ relations alive
        w = (0.10 + kappa) * (1.0 + unseen + provisional) / (1.0 + freq)
        if rel.stats.n_try > 12 and rel.stats.n_hit == 0:
            w *= 0.35                                  # repeated failures
        return max(1e-4, w)

    def freq_of_relation(self, rel: Relation) -> int:
        prefix = "%s|%s|%s|" % (rel.target, rel.rid, rel.shape)
        return sum(v for k, v in self.freq.items() if k.startswith(prefix))

    def select_target(self, rng: random.Random, target: Optional[str] = None,
                      shape: Optional[str] = None) -> Relation:
        cands = [r for r in self.relations
                 if (target is None or r.target == target)
                 and (shape is None or r.shape == shape)]
        if not cands:
            raise ValueError("no relation available for target=%s shape=%s" % (target, shape))
        weights = [self.relation_weight(r) for r in cands]
        return weighted_choice(rng, cands, weights)

    # ------------------------------------------------------------------ #
    # feedback
    # ------------------------------------------------------------------ #
    def activations(self, trace: Trace, rel: Optional[Relation] = None,
                    max_matches: int = 8) -> List[Binding]:
        rels = [rel] if rel is not None else [r for r in self.relations
                                             if r.target in ("", trace.target)]
        out: List[Binding] = []
        for r in rels:
            out.extend(activate(r, trace, max_matches=max_matches))
        return out

    def feedback_items(self, trace: Trace, rel: Optional[Relation] = None) -> List[str]:
        """``X(T')``: state interaction feedback keys ``<context, interaction, state>``."""
        keys = []
        for b in self.activations(trace, rel):
            keys.append(interaction_key(b.rel, b))
        return keys

    def interaction_points(self, trace: Trace) -> Set[str]:
        """Finer grained interaction points, used for campaign bookkeeping."""
        pts: Set[str] = set()
        for b in self.activations(trace):
            for s in b.states:
                pts.add("%s/%s/%s" % (b.rel.shape, s.sid, _canon(s.val)))
            if not b.states:
                pts.add("%s/%s" % (b.rel.shape, "no-state"))
        return pts

    def update_freq(self, items: Iterable[str]) -> int:
        n_new = 0
        for x in items:
            if self.freq.get(x, 0) == 0:
                n_new += 1
            self.freq[x] = self.freq.get(x, 0) + 1
        return n_new

    def rarity(self, items: Sequence[str]) -> float:
        """sum 1/(1+Freq[x]) over the feedback items of one execution."""
        return sum(1.0 / (1.0 + self.freq.get(x, 0)) for x in items)

    # ------------------------------------------------------------------ #
    # Section 4.2: online runtime refinement
    # ------------------------------------------------------------------ #
    def refine(self, trace: Trace, attempted_rid: Optional[str],
               activated: Sequence[Binding], clean: bool = True,
               max_provisional: int = 24) -> RefinementReport:
        """Update confidence/guards, and add provisional relations.

        ``clean`` must be False when a detector reported a violation: the paper
        refines only on executions without reported consistency violations, so
        that faulty behaviour is never learned as correct.
        """
        rep = RefinementReport()
        if attempted_rid and attempted_rid in self.relations:
            rel = self.get(attempted_rid)
            rel.stats.n_try += 1
            rep.attempted.append(rel.rid)
            if activated:
                rel.stats.n_hit += 1
                rel.stats.activations += len(activated)
                rel.stats.last_seen = trace.cycles
                rep.activated.append(rel.rid)
                rep.guards[rel.rid] = self._record_guards(rel, activated)
        if not clean:
            return rep
        if len([r for r in self.relations if r.source == "runtime"]) >= max_provisional:
            return rep
        for rel in self._unmodeled(trace, activated):
            try:
                self.add(rel)
                rep.new_relations.append(rel.rid)
            except ValueError:
                continue
        return rep

    def penalty(self, rid: str) -> None:
        """A reported violation means the relation fired; keep it hot."""
        if rid in self.relations:
            self.get(rid).stats.violations += 1

    def _record_guards(self, rel: Relation, acts: Sequence[Binding]) -> List[str]:
        out: List[str] = []
        for b in acts:
            for atom in rel.theta.atoms:
                a, c = b.events.get(atom.lhs), b.events.get(atom.rhs)
                if not a or not c:
                    continue
                if a.cycle == c.cycle:
                    key = "same_cycle"
                elif a.cycle < c.cycle:
                    key = "ordered"
                else:
                    key = "overlap"
                out.append(key)
                rel.stats.guards[key] = rel.stats.guards.get(key, 0) + 1
            for e in b.events.values():
                key = "outcome:" + e.o
                out.append(key)
                rel.stats.guards[key] = rel.stats.guards.get(key, 0) + 1
        return out

    def _unmodeled(self, trace: Trace, activated: Sequence[Binding]) -> List[Relation]:
        """Provisional relations for dependencies the static model missed.

        Heuristic (documented, conservative): two *different* event families that
        (a) share an object of the same kind, (b) occur within a small window,
        and (c) update a state id that no existing relation covers for that
        object, yield a provisional relation with ``omega = ⊥``.
        """
        known: Set[Tuple[str, str, str]] = set()
        for r in self.relations:
            for role in r.roles:
                for t in role.types:
                    for s in r.sids:
                        known.add((t, r.objvar.kind, s))
        out: List[Relation] = []
        by_obj: Dict[str, List[Any]] = {}
        for ev in trace.events:
            by_obj.setdefault(ev.u, []).append(ev)
        seen: Set[Tuple[str, str, str]] = set()
        for obj, evs in by_obj.items():
            if len(evs) < 2:
                continue
            evs.sort(key=lambda e: e.cycle)
            for i in range(len(evs) - 1):
                a, b = evs[i], evs[i + 1]
                gap = b.cycle - a.cycle
                if gap > 16 or a.a == b.a:
                    continue
                states = [s for s in trace.states if s.obj == obj and a.cycle <= s.cycle <= b.cycle + 8]
                if not states:
                    continue
                sid = states[0].sid
                kind = obj.split(":", 1)[0]
                fingerprint = (a.a, kind, sid)
                if fingerprint in known or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                rid = "R-RT-%s-%03d" % (trace.target, len(seen))
                try:
                    rel = Relation(
                        rid=rid, target=trace.target, shape="RUNTIME_%s_%s" % (a.a.upper(), b.a.upper()),
                        ctx={"same_warp": True}, objvar=_objvar_for(obj),
                        roles=(EventRole("first", (a.a,), _objvar_for(obj)),
                               EventRole("second", (b.a,), _objvar_for(obj))),
                        sids=(sid,), theta=Theta((Atom("before", "first", "second", 16),)),
                        omega=BOT, source="runtime",
                        evidence={"discovered_at": [a.cycle, b.cycle], "object": obj},
                    )
                    out.append(rel)
                except (ValueError, KeyError):
                    continue
        return out

    # ------------------------------------------------------------------ #
    # persistence / reporting
    # ------------------------------------------------------------------ #
    def to_json(self) -> Dict[str, Any]:
        return {"target": self.target, "meta": self.meta,
                "relations": [r.to_json() for r in self.relations],
                "freq": self.freq}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ESDModel":
        rs = [Relation.from_json(r) for r in d.get("relations", [])]
        return cls(rs, target=d.get("target", ""), freq=d.get("freq"),
                   meta=d.get("meta"))

    def save(self, path: str) -> None:
        write_json(path, self.to_json())

    @classmethod
    def load(cls, path: str) -> "ESDModel":
        d = read_json(path)
        if d is None:
            raise FileNotFoundError("no ESD model at %s" % path)
        return cls.from_json(d)

    def summary_rows(self) -> List[List[str]]:
        rows = []
        for r in sorted(self.relations, key=lambda x: (x.target, x.rid)):
            rows.append([r.target, r.rid, r.shape, ",".join(r.sids),
                         r.omega.oid if r.checkable else "⊥", r.source,
                         "%d/%d" % (r.stats.n_hit, r.stats.n_try),
                         "%.2f" % r.stats.kappa, r.bug_hint])
        return rows

    def summary(self) -> str:
        rows = self.summary_rows()
        head = ["target", "rid", "shape", "states", "obligation", "src", "hit/try", "kappa", "bug"]
        return table(rows, head)

    def stats(self) -> Dict[str, Any]:
        per_target: Dict[str, Dict[str, int]] = {}
        for r in self.relations:
            d = per_target.setdefault(r.target, {"relations": 0, "checkable": 0,
                                                 "provisional": 0, "hit": 0})
            d["relations"] += 1
            d["checkable"] += 1 if r.checkable else 0
            d["provisional"] += 1 if r.source == "runtime" else 0
            d["hit"] += r.stats.n_hit
        return {"relations": len(self.relations), "freq_items": len(self.freq),
                "per_target": per_target}


def _canon(val: Any) -> str:
    if isinstance(val, dict):
        return ",".join("%s=%s" % (k, val[k]) for k in sorted(val))
    return str(val)


def _objvar_for(obj: str) -> Any:
    from common.types import ObjVar
    from common.types import OBJ_KINDS
    kind = obj.split(":", 1)[0] or "unit"
    return ObjVar("o", kind if kind in OBJ_KINDS else "unit")
