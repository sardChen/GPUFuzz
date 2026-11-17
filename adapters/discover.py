"""Resolve the signal-map probes against a real RTL checkout.

The signal maps in ``adapters/signals/`` carry *probes* (regex fragments) instead
of guessed hierarchical paths.  This tool walks the RTL sources, finds candidate
declarations, infers the enclosing module, and writes the dotted path that
Verilator's scope lookup expects: ``<scope>.<module>.<signal>``.

Usage::

    python3 cli.py discover --target ventus --rtl ~/gpu/ventus --write

``--write`` rewrites ``adapters/signals/<target>.json`` in place (a ``.bak`` copy
is kept) and records the source location of every resolved signal as evidence.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.targets import TargetSpec, load_target
from common.util import Log, table, write_json

SRC_EXT = (".sv", ".v", ".svh", ".vh", ".scala")
MODULE_RE = re.compile(r"^\s*(?:module|class)\s+([A-Za-z_][\w$]*)")
SIGNAL_RE = re.compile(r"^\s*(?:logic|reg|wire|input|output|val)\b[^;=]*?"
                       r"\b([A-Za-z_][\w$]*)\s*(?:[;=,\[]|$)")


@dataclass
class Candidate:
    name: str
    module: str
    file: str
    line: int
    confidence: str = "medium"

    @property
    def path(self) -> str:
        return "%s.%s" % (self.module, self.name)


@dataclass
class Resolution:
    key: str                  # signal / state id
    probe: str
    candidates: List[Candidate] = field(default_factory=list)
    chosen: str = ""
    location: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"key": self.key, "probe": self.probe, "chosen": self.chosen,
                "location": self.location,
                "candidates": [c.__dict__ for c in self.candidates[:8]]}


class SignalDiscovery:
    def __init__(self, spec: TargetSpec, rtl_dir: str, log: Optional[Log] = None) -> None:
        self.spec = spec
        self.rtl = os.path.abspath(rtl_dir)
        self.log = log or Log("info")
        self.records: List[Tuple[str, str, int, str]] = []   # file, module, line, name

    # ------------------------------------------------------------------ #
    def scan(self) -> int:
        for dirpath, _dirs, files in os.walk(self.rtl):
            for fn in files:
                if not fn.endswith(SRC_EXT):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        module = ""
                        for i, line in enumerate(fh, 1):
                            m = MODULE_RE.match(line)
                            if m:
                                module = m.group(1)
                                continue
                            for found in SIGNAL_RE.finditer(line):
                                self.records.append((os.path.relpath(path, self.rtl),
                                                     module, i, found.group(1)))
                except OSError:
                    continue
        return len(self.records)

    def resolve(self, probe: str, key: str) -> Resolution:
        res = Resolution(key=key, probe=probe)
        try:
            rx = re.compile(probe)
        except re.error:
            rx = re.compile(re.escape(probe))
        seen = set()
        for file, module, line, name in self.records:
            if not (rx.search(name) or rx.search("%s.%s" % (module, name))):
                continue
            sig = (module, name)
            if sig in seen:
                continue
            seen.add(sig)
            conf = "high" if rx.fullmatch(name) else "medium"
            res.candidates.append(Candidate(name=name, module=module, file=file,
                                            line=line, confidence=conf))
        res.candidates.sort(key=lambda c: (0 if c.confidence == "high" else 1,
                                           len(c.name), c.file))
        if res.candidates:
            best = res.candidates[0]
            res.chosen = best.path
            res.location = "%s:%d" % (best.file, best.line)
        return res

    # ------------------------------------------------------------------ #
    def run(self, write: bool = False) -> Dict[str, Any]:
        n = self.scan()
        sigmap = self.spec.signal_map()
        scope = sigmap.get("scope") or ("TOP." + str(sigmap.get("top", "")))
        out: Dict[str, Any] = {"target": self.spec.name, "rtl": self.rtl,
                               "declarations": n, "signals": {}, "states": {},
                               "events": {}, "unresolved": []}
        for entry in sigmap.get("signals") or ():
            if entry.get("path"):
                out["signals"][entry["id"]] = {"path": entry["path"], "kept": True}
                continue
            res = self.resolve(entry.get("probe", entry.get("id", "")), entry.get("id", ""))
            if res.chosen:
                entry["path"] = "%s.%s" % (scope, res.chosen)
                entry["confidence"] = res.candidates[0].confidence
                entry["evidence"] = res.location
                out["signals"][entry["id"]] = {"path": entry["path"],
                                               "evidence": res.location,
                                               "candidates": len(res.candidates)}
            else:
                out["unresolved"].append("signal:" + entry.get("id", ""))
        for entry in sigmap.get("states") or ():
            if entry.get("path"):
                continue
            res = self.resolve(entry.get("probe", entry.get("sid", "")), entry.get("sid", ""))
            if res.chosen:
                entry["path"] = "%s.%s" % (scope, res.chosen)
                entry["confidence"] = res.candidates[0].confidence
                entry["evidence"] = res.location
                out["states"][entry["sid"]] = {"path": entry["path"],
                                               "evidence": res.location}
            else:
                out["unresolved"].append("state:" + entry.get("sid", ""))
        # memory pins live in the recipe, not the signal map
        recipe = self.spec.recipe()
        if recipe:
            pins = (recipe.get("memory") or {}).get("pins") or {}
            resolved_pins = 0
            for name, path in list(pins.items()):
                if path:
                    resolved_pins += 1
                    continue
                res = self.resolve(name, name)
                if res.chosen:
                    pins[name] = "%s.%s" % (scope, res.chosen)
                    resolved_pins += 1
                    out.setdefault("pins", {})[name] = {"path": pins[name],
                                                        "evidence": res.location}
            out["pins_resolved"] = "%d/%d" % (resolved_pins, len(pins))
            if write and resolved_pins:
                recipe["memory"]["pins"] = pins
                write_json(self.spec.recipe_path, recipe)

        for ev in sigmap.get("events") or ():
            obj_from = (ev.get("obj") or {}).get("from")
            if obj_from and obj_from not in {s.get("id") for s in sigmap.get("signals") or ()}:
                out["unresolved"].append("event-obj:%s" % obj_from)

        if write:
            path = self.spec.signals_path
            backup = path + ".bak"
            if os.path.exists(path) and not os.path.exists(backup):
                with open(path, "r", encoding="utf-8") as src, open(backup, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
            sigmap["confidence"] = "resolved" if not out["unresolved"] else "partial"
            write_json(path, sigmap)
            out["written"] = path
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def summary_rows(result: Dict[str, Any]) -> List[List[str]]:
        rows = []
        for kind in ("signals", "states"):
            for key, val in sorted((result.get(kind) or {}).items()):
                rows.append([kind, key, val.get("path", ""), val.get("evidence", "")])
        return rows

    @staticmethod
    def render(result: Dict[str, Any]) -> str:
        rows = SignalDiscovery.summary_rows(result)
        txt = table(rows, ["kind", "element", "path", "evidence"])
        if result.get("unresolved"):
            txt += "\n\nunresolved: " + ", ".join(result["unresolved"])
        return txt


def discover(target: str, rtl: str, write: bool = False,
             log: Optional[Log] = None) -> Dict[str, Any]:
    spec = load_target(target)
    return SignalDiscovery(spec, rtl, log).run(write=write)
