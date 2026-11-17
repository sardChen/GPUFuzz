"""Execution trace adapter: raw harness JSONL -> canonical ``O = <A, E, S>``.

The harness only samples signals and state elements; *what a sample means* is
declared in the signal map as small expressions, e.g.

    {"type": "wb", "when": "wb_valid & wb_ready", "obj": {"kind": "reg", "from": "wb_reg"}}

This module compiles those expressions in a restricted AST (no calls, no
attributes, only bitwise/comparison operators over signal ids) and turns rising
edges into canonical events.  Keeping the derivation on the Python side means a
target's signal names never leak into the fuzzer or the oracle.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from common.types import ArchEffects, Ctx, Event, StateObs, Trace
from common.util import Log, read_json, read_jsonl

_ALLOWED_BINOPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
                   ast.Add, ast.Sub, ast.Mult)
_ALLOWED_UNARY = (ast.Not, ast.Invert, ast.USub, ast.UAdd)
_ALLOWED_CMP = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class ExpressionError(ValueError):
    pass


class SignalExpr:
    """A compiled ``when`` expression over signal ids."""

    def __init__(self, text: str, signal_ids: Set[str]) -> None:
        self.text = text
        self.names: Set[str] = set()
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError("bad expression %r: %s" % (text, exc))
        self._code = compile(tree, "<signal-expr>", "eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in signal_ids:
                    raise ExpressionError("expression %r references unknown signal %r"
                                          % (text, node.id))
                self.names.add(node.id)
            elif isinstance(node, (ast.BinOp,)) and not isinstance(node.op, _ALLOWED_BINOPS):
                raise ExpressionError("operator %s not allowed in %r"
                                      % (type(node.op).__name__, text))
            elif isinstance(node, ast.UnaryOp) and not isinstance(node.op, _ALLOWED_UNARY):
                raise ExpressionError("unary operator not allowed in %r" % (text,))
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    if not isinstance(op, _ALLOWED_CMP):
                        raise ExpressionError("comparison not allowed in %r" % (text,))
            elif isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda)):
                raise ExpressionError("expression %r is not a signal expression" % (text,))

    def evaluate(self, values: Dict[str, int]) -> int:
        env = {n: int(values.get(n, 0)) for n in self.names}
        return int(eval(self._code, {"__builtins__": {}}, env))  # noqa: S307 - restricted AST


@dataclass
class DerivedEvent:
    type: str
    expr: SignalExpr
    obj_kind: str = ""
    obj_from: str = ""
    obj_fmt: str = ""
    outcome: str = "ok"
    prev: int = 0

    def fire(self, values: Dict[str, int]) -> Optional[Tuple[str, str]]:
        cur = self.expr.evaluate(values)
        rising = bool(cur) and not self.prev
        self.prev = cur
        if not rising:
            return None
        if not self.obj_from:
            return self.obj_fmt or ("%s:0" % (self.obj_kind or "unit")), self.outcome
        val = int(values.get(self.obj_from, 0))
        try:
            obj = self.obj_fmt % val
        except TypeError:
            obj = "%s:%d" % (self.obj_kind or "unit", val)
        return obj, self.outcome


class TraceAdapter:
    """Converts the harness JSONL into a :class:`common.types.Trace`."""

    def __init__(self, target: str, sigmap: Dict[str, Any],
                 log: Optional[Log] = None) -> None:
        self.target = target
        self.map = sigmap or {}
        self.log = log or Log("warn")
        ids = {s.get("id") for s in (self.map.get("signals") or ()) if s.get("id")}
        self.signal_ids: Set[str] = {i for i in ids if i}
        self.derived: List[DerivedEvent] = []
        self.derivation_errors: List[str] = []
        for spec in self.map.get("events") or ():
            try:
                expr = SignalExpr(spec.get("when", "0"), self.signal_ids)
            except ExpressionError as exc:
                self.derivation_errors.append(str(exc))
                continue
            obj = spec.get("obj") or {}
            self.derived.append(DerivedEvent(
                type=spec.get("type", ""), expr=expr,
                obj_kind=obj.get("kind", ""), obj_from=obj.get("from", ""),
                obj_fmt=obj.get("fmt", ""), outcome=spec.get("outcome", "ok")))

    # ------------------------------------------------------------------ #
    def parse(self, path: str, testcase_id: str = "") -> Trace:
        events: List[Event] = []
        states: List[StateObs] = []
        arch = ArchEffects()
        cycles = 0
        timeout = False
        exit_code = 0
        header: Dict[str, Any] = {}
        values: Dict[str, int] = {}
        meta: Dict[str, Any] = {}

        for rec in read_jsonl(path):
            t = rec.get("t")
            if t == "hdr":
                header = rec
                meta["unresolved"] = rec.get("unresolved") or []
                meta["top"] = rec.get("top", "")
            elif t == "sig":
                cycle = int(rec.get("c", 0))
                changed = rec.get("v") or {}
                for k, v in changed.items():
                    values[k] = int(v)
                for d in self.derived:
                    if not (d.expr.names & set(changed)):
                        continue
                    fired = d.fire(values)
                    if fired:
                        obj, outcome = fired
                        events.append(Event(a=d.type, u=obj, o=outcome, cycle=cycle,
                                            ctx=Ctx(env={}, phase="outstanding", obj=obj)))
            elif t == "st":
                states.append(StateObs.from_json(rec))
            elif t == "ev":
                try:
                    events.append(Event(a=rec["a"], u=rec.get("u", ""), o=rec.get("o", "ok"),
                                        cycle=int(rec.get("c", 0)), raw=rec.get("raw")))
                except ValueError as exc:
                    self.log.debug("dropping event: %s" % exc)
            elif t == "arch":
                arch = ArchEffects.from_json(rec)
                if "c" in rec:
                    cycles = int(rec["c"])
            elif t == "end":
                cycles = int(rec.get("c", cycles))
                timeout = bool(rec.get("timeout", 0))
                exit_code = int(rec.get("exit", 0))
        if header:
            meta["run_cycles"] = cycles
        return Trace(target=self.target, testcase_id=testcase_id, events=events,
                     states=states, arch=arch, cov=[], cycles=cycles,
                     exit_code=exit_code, timeout=timeout, meta=meta)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        return ("%s: %d derived event rules, %d signals%s"
                % (self.target, len(self.derived), len(self.signal_ids),
                   "" if not self.derivation_errors else
                   " (%d bad rules)" % len(self.derivation_errors)))


def load_adapter(target: str, signals_path: str, log: Optional[Log] = None) -> TraceAdapter:
    sigmap = read_json(signals_path, {}) or {}
    return TraceAdapter(target, sigmap, log=log)
