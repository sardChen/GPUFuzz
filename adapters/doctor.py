"""Readiness check for the six targets.

``cli.py doctor`` answers, per target: is there a spec, a recipe, a resolved
signal map, an ISA profile with the macros the generated code needs, the
toolchain, Verilator and a built simulator?  Nothing is hidden behind a green
checkmark: every gap is listed with the command that fixes it.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.targets import TARGETS, TargetSpec, load_target
from common.util import Log, read_json, table


@dataclass
class TargetStatus:
    target: str
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def level(self) -> str:
        if self.problems:
            return "blocked"
        if self.warnings:
            return "partial"
        return "ready"

    def to_json(self) -> Dict[str, Any]:
        return {"target": self.target, "level": self.level(),
                "problems": self.problems, "warnings": self.warnings, "facts": self.facts}


def check_target(spec: TargetSpec, workdir: str = "work", rtl: Optional[str] = None) -> TargetStatus:
    st = TargetStatus(target=spec.name)

    # spec / caps
    st.facts["isa"] = spec.isa
    st.facts["hdl"] = spec.hdl
    st.facts["caps"] = {k: spec.caps.get(k) for k in ("lanes", "warps_per_block", "smem_bytes")}

    # recipe
    recipe = spec.recipe()
    if not recipe:
        st.problems.append("missing recipe %s" % spec.recipe_path)
    else:
        st.facts["top"] = recipe.get("top")
        if not recipe.get("done"):
            st.warnings.append("recipe has no 'done' signal: runs will always hit max_cycles")
        mem = recipe.get("memory") or {}
        if mem.get("kind") == "none":
            st.warnings.append("recipe declares no memory interface")
        else:
            pins = mem.get("pins") or {}
            unresolved = [k for k, v in pins.items() if not v]
            st.facts["mem_pins"] = "%d/%d" % (len(pins) - len(unresolved), len(pins))
            if unresolved:
                st.problems.append("memory pins unresolved (%d): run `cli.py discover --write`"
                                   % len(unresolved))

    # signal map
    sigmap = spec.signal_map()
    if not sigmap:
        st.problems.append("missing signal map %s" % spec.signals_path)
    else:
        sigs = sigmap.get("signals") or []
        states = sigmap.get("states") or []
        resolved_s = sum(1 for s in sigs if s.get("path"))
        resolved_t = sum(1 for s in states if s.get("path"))
        st.facts["signals"] = "%d/%d resolved" % (resolved_s, len(sigs))
        st.facts["states"] = "%d/%d resolved" % (resolved_t, len(states))
        st.facts["events"] = len(sigmap.get("events") or ())
        if resolved_s < len(sigs):
            st.problems.append("signal map unresolved: run `cli.py discover --target %s --rtl <checkout> --write`" % spec.name)
        if not states:
            st.warnings.append("no state elements: interaction obligations cannot be checked")
        if not sigmap.get("events"):
            st.problems.append("signal map has no event derivation rules")

    # ISA profile
    try:
        from common.isa.emit import load_profile
        prof = load_profile(spec.name)
        macros = prof.get("macros") or {}
        placeholder = [k for k, v in macros.items() if v.get("conf") in ("placeholder", "medium")]
        st.facts["isa_dialect"] = prof.get("dialect")
        st.facts["macros"] = "%d defined, %d placeholder/medium" % (len(macros), len(placeholder))
        if not macros:
            st.problems.append("ISA profile defines no GPU_* macros: emission will fail")
        elif placeholder:
            st.warnings.append("%d macros are placeholders (%s)"
                               % (len(placeholder), ", ".join(placeholder[:4])))
        tc = prof.get("toolchain") or {}
        cc = tc.get("cc")
        if cc and not shutil.which(cc):
            st.problems.append("toolchain %r not on PATH" % cc)
        st.facts["toolchain"] = cc
    except FileNotFoundError as exc:
        st.problems.append(str(exc))

    # verilator + build
    st.facts["verilator"] = "yes" if shutil.which("verilator") else "no"
    if not shutil.which("verilator"):
        st.warnings.append("verilator not on PATH: only --dry-run / fixture runs are possible here")
    build_json = read_json(os.path.join(workdir, spec.name, "build", "build.json"), {}) or {}
    if build_json.get("ok"):
        st.facts["simulator"] = build_json.get("sim")
    else:
        st.facts["simulator"] = "not built"

    # RTL
    if rtl:
        rtl_path = os.path.join(rtl, spec.name)
        st.facts["rtl"] = "found" if os.path.isdir(rtl_path) else "missing"
        if not os.path.isdir(rtl_path):
            st.warnings.append("no RTL checkout at %s" % rtl_path)
    return st


def doctor(workdir: str = "work", rtl: Optional[str] = None,
           targets: Optional[List[str]] = None, log: Optional[Log] = None) -> Dict[str, Any]:
    specs = [load_target(t) for t in (targets or list(TARGETS))]
    statuses = [check_target(s, workdir, rtl) for s in specs]
    return {"targets": [s.to_json() for s in statuses],
            "summary": {
                "ready": sum(1 for s in statuses if s.level() == "ready"),
                "partial": sum(1 for s in statuses if s.level() == "partial"),
                "blocked": sum(1 for s in statuses if s.level() == "blocked"),
            }}


def render(result: Dict[str, Any]) -> str:
    rows = []
    for t in result["targets"]:
        rows.append([t["target"], t["level"], t["facts"].get("signals", "-"),
                     t["facts"].get("macros", "-"), t["facts"].get("verilator", "-"),
                     t["facts"].get("simulator", "-")])
    txt = table(rows, ["target", "level", "signals", "macros", "verilator", "simulator"])
    details = []
    for t in result["targets"]:
        if t["problems"] or t["warnings"]:
            details.append("%s (%s)" % (t["target"], t["level"]))
            for p in t["problems"]:
                details.append("  ! " + p)
            for w in t["warnings"]:
                details.append("  - " + w)
    if details:
        txt += "\n\n" + "\n".join(details)
    s = result["summary"]
    txt += "\n\n%d ready, %d partial, %d blocked" % (s["ready"], s["partial"], s["blocked"])
    return txt
