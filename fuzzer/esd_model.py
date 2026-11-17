"""ESDModel construction: the three static steps of Section 4.2.

Step 1 (semantic analysis) comes from the shape catalog: it encodes the GPU
semantics, instruction semantics, memory protocol, and the derived obligations,
independent of any RTL implementation.

Step 2 (RTL dependency tracing) consumes a *signal map* (``adapters/signals/
<target>.json``) that links semantic events and states to elaborated RTL signals,
including the update enables observed while tracing the design.  A state whose
value/update condition is influenced by two or more event families is a
candidate interaction state; states that no catalog relation covers yield
provisional (``omega = ⊥``) relations.

Step 3 (relation assembly) produces the ``R`` set: contexts, event roles, state
cluster, temporal constraint and obligation, plus the evidence needed to report
*why* a relation exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from common.esd.model import ESDModel
from common.targets import TargetSpec, load_target
from common.types import (BOT, Atom, EventRole, ObjVar, Relation, Theta,
                            STATE_SET)
from common.util import Log, read_json
from .shapes.catalog import SHAPES, for_target, validate_catalog


@dataclass
class ConstructionReport:
    target: str
    n_shapes: int = 0
    n_relations: int = 0
    n_checkable: int = 0
    n_rtl_only: int = 0
    unobservable: List[str] = field(default_factory=list)
    missing_states: Dict[str, List[str]] = field(default_factory=dict)
    missing_events: Dict[str, List[str]] = field(default_factory=dict)
    rtl_states: int = 0
    n_resolved: int = 0
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "target": self.target, "shapes": self.n_shapes,
            "relations": self.n_relations, "checkable": self.n_checkable,
            "rtl_only": self.n_rtl_only, "unobservable": self.unobservable,
            "missing_states": self.missing_states,
            "missing_events": self.missing_events, "rtl_states": self.rtl_states,
            "resolved_signals": self.n_resolved, "note": self.note,
        }

    def describe(self) -> str:
        parts = ["%s: %d relations (%d checkable, %d rtl-only)"
                 % (self.target, self.n_relations, self.n_checkable, self.n_rtl_only)]
        if self.unobservable:
            parts.append("%d not observable in the signal map: %s"
                         % (len(self.unobservable), ", ".join(self.unobservable[:4])
                            + ("..." if len(self.unobservable) > 4 else "")))
        if self.note:
            parts.append(self.note)
        return "; ".join(parts)


def _state_index(sigmap: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """state id -> declared RTL elements (``signals[].kind == state`` + ``states[]``)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for s in sigmap.get("signals") or ():
        st = s.get("state")
        if s.get("kind") == "state" and st:
            out.setdefault(st, []).append(s)
    for s in sigmap.get("states") or ():
        st = s.get("sid")
        if st:
            out.setdefault(st, []).append(dict(s, kind="state"))
    return out


def _event_types(sigmap: Dict[str, Any]) -> Set[str]:
    return {e.get("type") for e in (sigmap.get("events") or ()) if e.get("type")}


def _writers(sigmap: Dict[str, Any]) -> Dict[str, Set[str]]:
    """state id -> set of event families that update it (RTL tracing result)."""
    out: Dict[str, Set[str]] = {}
    for s in list(sigmap.get("signals") or ()) + list(sigmap.get("states") or ()):
        st = s.get("state") or s.get("sid")
        if st:
            out.setdefault(st, set()).update(s.get("written_by") or ())
    return out


class ESDModelBuilder:
    """Builds ``M`` for one target."""

    def __init__(self, log: Optional[Log] = None) -> None:
        self.log = log or Log("warn")

    # -- public ------------------------------------------------------------ #
    def build(self, target: TargetSpec, use_rtl: bool = True,
              rtl_override: Optional[str] = None) -> Tuple[ESDModel, ConstructionReport]:
        problems = validate_catalog()
        if problems:
            raise ValueError("shape catalog is inconsistent: " + "; ".join(problems))

        shapes = for_target(target.name)
        rels: List[Relation] = []
        for sh in shapes:
            rel = sh.to_relation(target.name, evidence={
                "spec_ref": sh.spec_ref,
                "desc": sh.desc,
                "shape_kind": sh.kind,
                "knobs": list(sh.knobs),
                "generator": sh.gen,
            })
            rels.append(rel)

        rep = ConstructionReport(target=target.name, n_shapes=len(shapes))
        sigmap: Dict[str, Any] = {}
        if use_rtl:
            path = rtl_override or target.signals_path
            sigmap = read_json(path, {}) or {}
            if not sigmap:
                rep.note = ("no signal map at %s -- RTL tracing skipped, relations "
                            "carry no signal evidence" % path)
                self.log.warn(rep.note)
            else:
                self._attach_rtl(rels, sigmap, rep)
                rels.extend(self._rtl_only(rels, sigmap, rep))
        else:
            rep.note = "RTL tracing disabled (--no-rtl)"

        for rel in rels:
            self._assemble(rel, sigmap, rep)
        model = ESDModel(rels, target=target.name,
                         meta={"built_from": "catalog+rtl" if sigmap else "catalog",
                               "signal_map": sigmap.get("target", "")})
        rep.n_relations = len(model)
        rep.n_checkable = len(model.checkable())
        return model, rep

    # -- step 2 ------------------------------------------------------------ #
    def _attach_rtl(self, rels: List[Relation], sigmap: Dict[str, Any],
                    rep: ConstructionReport) -> None:
        states = _state_index(sigmap)
        events = _event_types(sigmap)
        writers = _writers(sigmap)
        rep.rtl_states = len(states)
        resolved = 0
        for rel in rels:
            paths: List[str] = []
            missing_s: List[str] = []
            for sid in rel.sids:
                entries = states.get(sid) or []
                if not entries:
                    missing_s.append(sid)
                    continue
                for e in entries:
                    p = e.get("path") or ""
                    if p:
                        resolved += 1
                    paths.append(p or ("<%s: unresolved>" % sid))
                rel.stats.guards.setdefault("rtl_signal:" + sid, len(entries))
            missing_e: List[str] = []
            for role in rel.roles:
                if not (set(role.types) & events):
                    missing_e.extend(sorted(set(role.types) - events))
            rel.evidence["signals"] = paths
            rel.evidence["declared"] = not missing_s and not missing_e
            rel.evidence["rtl_writers"] = {s: sorted(writers.get(s, ())) for s in rel.sids}
            if missing_s:
                rep.missing_states[rel.rid] = missing_s
            if missing_e:
                rep.missing_events[rel.rid] = sorted(set(missing_e))
            if not rel.evidence["declared"]:
                rep.unobservable.append(rel.rid)
        rep.n_resolved = resolved

    def _rtl_only(self, rels: List[Relation], sigmap: Dict[str, Any],
                  rep: ConstructionReport) -> List[Relation]:
        """Provisional relations for interaction states the catalog misses."""
        covered = {s for r in rels for s in r.sids}
        writers = _writers(sigmap)
        out: List[Relation] = []
        for sid, fams in sorted(writers.items()):
            if len(fams) < 2 or sid in covered:
                continue
            fam_list = sorted(fams)
            first, second = fam_list[0], fam_list[1]
            try:
                rel = Relation(
                    rid="R-%s-RTL-%s" % (rep.target.upper(), sid.replace(".", "_").upper()),
                    target=rep.target, shape="RTL_%s" % sid.replace(".", "_").upper(),
                    ctx={}, objvar=ObjVar("O", "buff"),
                    roles=(EventRole("first", (first,), None),
                           EventRole("second", (second,), None)),
                    sids=(sid,), theta=Theta((Atom("before", "first", "second", 64),)),
                    omega=BOT, source="rtl",
                    evidence={"reason": "state influenced by %s" % ", ".join(fam_list),
                              "rtl_writers": fam_list,
                              "signals": [s.get("path") for s in
                                          (sigmap.get("signals") or ())
                                          if s.get("state") == sid]},
                )
            except ValueError:
                continue
            out.append(rel)
        rep.n_rtl_only = len(out)
        return out

    # -- step 3 ------------------------------------------------------------ #
    def _assemble(self, rel: Relation, sigmap: Dict[str, Any],
                  rep: ConstructionReport) -> None:
        rel.evidence.setdefault("assembled", True)
        rel.evidence.setdefault("target_isa", sigmap.get("isa", ""))
        if rel.ctx.get("same_warp") is None and rel.objvar.kind in ("reg", "warp", "lane"):
            rel.ctx.setdefault("same_warp", True)
        rel.evidence.setdefault("theta_text", rel.theta.describe())
        if rel.checkable:
            rel.evidence.setdefault("omega_text", rel.omega.describe())


def build_model(target: str, use_rtl: bool = True,
                log: Optional[Log] = None) -> Tuple[ESDModel, ConstructionReport]:
    spec = load_target(target)
    return ESDModelBuilder(log).build(spec, use_rtl=use_rtl)


def build_all(use_rtl: bool = True, log: Optional[Log] = None
              ) -> Tuple[Dict[str, ESDModel], List[ConstructionReport]]:
    from common.targets import TARGETS
    models: Dict[str, ESDModel] = {}
    reps: List[ConstructionReport] = []
    for name in TARGETS:
        m, r = build_model(name, use_rtl=use_rtl, log=log)
        models[name] = m
        reps.append(r)
    return models, reps
