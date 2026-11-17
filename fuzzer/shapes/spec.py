"""Shape specifications: the catalog entries that turn into ESD relations.

A *shape* is the target independent description of one execution state
dependency: which events are involved, which state cluster they touch, the
temporal constraint between them, and the consistency obligation that must hold.
The catalog is the result of the paper's "Step 1: GPU semantic analysis" (event
and obligation extraction from ISA/memory/synchronisation semantics); each target
spec selects the shapes its design actually implements.

Shapes are also the unit of test-case synthesis: each shape names a generator
template that emits the setup / trigger / observation program.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.types import (Atom, BOT, EventRole, Obligation, ObjVar, Relation,
                             RelationSet, Theta)


@dataclass(frozen=True)
class RoleSpec:
    role: str
    types: Tuple[str, ...]
    same_obj: bool = True
    outcomes: Tuple[str, ...] = ()
    optional: bool = False


@dataclass
class ShapeSpec:
    shape: str
    kind: str                       # interaction | functional
    omega: str                      # obligation id, "⊥" when undetermined
    obj_kind: str
    sids: Tuple[str, ...]
    roles: Tuple[RoleSpec, ...]
    atoms: Tuple[Tuple[Any, ...], ...]   # (kind, lhs, rhs[, param])
    ctx: Dict[str, Any] = field(default_factory=dict)
    targets: Tuple[str, ...] = ("*",)
    gen: str = ""
    spec_ref: str = ""
    knobs: Tuple[str, ...] = ()
    desc: str = ""
    bug_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    # -- relation factory -------------------------------------------------- #
    def applies_to(self, target: str) -> bool:
        return "*" in self.targets or target in self.targets

    def objvar(self) -> ObjVar:
        name = {"line": "L", "txn": "T", "reg": "R", "slot": "S", "warp": "W",
                "cfg": "C", "tex": "X", "wave": "V", "buff": "B", "unit": "U",
                "mem": "M", "kernel": "K", "block": "K", "lane": "N", "pc": "P"}.get(
            self.obj_kind, "O")
        # a few shapes need a second, differently-typed object (e.g. wave+reg)
        return ObjVar(name, self.obj_kind)

    def to_relation(self, target: str, rid: Optional[str] = None,
                    evidence: Optional[Dict[str, Any]] = None) -> Relation:
        ov = self.objvar()
        roles = tuple(EventRole(r.role, r.types, ov if r.same_obj else None, r.outcomes,
                                optional=r.optional)
                      for r in self.roles)
        atoms = tuple(Atom(a[0], a[1], a[2], a[3] if len(a) > 3 else 0)
                      for a in self.atoms)
        rel = Relation(
            rid=rid or "R-%s-%s" % (target.upper(), self.shape),
            target=target,
            shape=self.shape,
            ctx=dict(self.ctx),
            objvar=ov,
            roles=roles,
            sids=tuple(self.sids),
            theta=Theta(atoms),
            omega=Obligation(self.omega) if self.omega not in ("⊥", "BOT") else BOT,
            source="semantic",
            evidence=dict(evidence or {}),
            bug_hint=",".join(self.bug_ids.get(target, ())),
        )
        rel.evidence.setdefault("shape_kind", self.kind)
        rel.evidence.setdefault("spec_ref", self.spec_ref)
        rel.evidence.setdefault("desc", self.desc)
        return rel

    def to_json(self) -> Dict[str, Any]:
        return {
            "shape": self.shape, "kind": self.kind, "omega": self.omega,
            "obj_kind": self.obj_kind, "sids": list(self.sids),
            "roles": [{"role": r.role, "types": list(r.types), "same_obj": r.same_obj,
                       "outcomes": list(r.outcomes), "optional": r.optional}
                      for r in self.roles],
            "atoms": [list(a) for a in self.atoms], "ctx": self.ctx,
            "targets": list(self.targets), "gen": self.gen,
            "spec_ref": self.spec_ref, "knobs": list(self.knobs),
            "desc": self.desc,
            "bug_ids": {k: list(v) for k, v in self.bug_ids.items()},
        }
