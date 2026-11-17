"""Build the Verilator simulator for one target from its recipe."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adapters.verilator import gen as harness_gen
from common.targets import TargetSpec, load_target
from common.util import Log, ensure_dir, read_json, write_json


@dataclass
class BuildResult:
    ok: bool = False
    outdir: str = ""
    sim: str = ""
    cmd: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    log: str = ""
    error: str = ""
    dry_run: bool = False
    sources: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"ok": self.ok, "outdir": self.outdir, "sim": self.sim,
                "cmd": self.cmd, "elapsed": round(self.elapsed, 3),
                "error": self.error, "dry_run": self.dry_run,
                "sources": len(self.sources), "log_tail": self.log[-2000:]}


def expand_sources(recipe: Dict[str, Any], rtl_dir: Optional[str]) -> List[str]:
    out: List[str] = []
    for pattern in recipe.get("sources") or ():
        if os.path.isabs(pattern):
            out.extend(sorted(glob.glob(pattern, recursive=True)))
        elif rtl_dir:
            out.extend(sorted(glob.glob(os.path.join(rtl_dir, pattern), recursive=True)))
    seen, uniq = set(), []
    for p in out:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            uniq.append(p)
    return uniq


def check_tools(recipe: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "verilator": bool(shutil.which("verilator")),
        "make": bool(shutil.which("make")),
    }


def build(target: str, rtl_dir: Optional[str], outdir: str, dry_run: bool = False,
          log: Optional[Log] = None, timeout: float = 1800.0) -> BuildResult:
    log = log or Log("info")
    spec = load_target(target)
    recipe = spec.recipe()
    if not recipe:
        return BuildResult(error="missing recipe: %s" % spec.recipe_path)

    res = BuildResult(outdir=outdir, dry_run=dry_run)
    t0 = time.time()
    try:
        g = harness_gen.generate(spec, outdir)
    except Exception as exc:
        res.error = "harness generation failed: %s" % exc
        return res
    res.cmd = list(g.build_cmd)

    srcs = expand_sources(recipe, rtl_dir)
    res.sources = srcs
    if not srcs:
        if dry_run:
            res.log = ("dry run without RTL: harness generated, sources unresolved "
                       "(patterns: %s)" % ", ".join(recipe.get("sources") or ()))
        else:
            res.error = ("no RTL sources matched the recipe patterns; pass --rtl <checkout> "
                         "(patterns: %s)" % ", ".join(recipe.get("sources") or ()))
            return res
    # replace the recipe globs with the resolved file list
    res.cmd = [c for c in res.cmd if c not in (recipe.get("sources") or [])]
    insert_at = res.cmd.index("--exe") if "--exe" in res.cmd else len(res.cmd)
    res.cmd[insert_at:insert_at] = srcs or list(recipe.get("sources") or ())

    tools = check_tools(recipe)
    if not dry_run and not tools["verilator"]:
        res.error = "verilator not found on PATH"
        return res
    if dry_run:
        res.ok = True
        res.error = ""
        res.elapsed = time.time() - t0
        write_json(os.path.join(outdir, "build.json"), res.to_json())
        return res

    ensure_dir(outdir)
    log.info("building %s: %d sources -> %s" % (target, len(srcs), outdir))
    try:
        proc = subprocess.run(res.cmd, cwd=outdir, capture_output=True, text=True,
                              timeout=timeout)
        res.log = (proc.stdout or "") + (proc.stderr or "")
        res.ok = proc.returncode == 0
        if not res.ok:
            res.error = "verilator returned %d" % proc.returncode
    except subprocess.TimeoutExpired:
        res.error = "build timed out after %.0fs" % timeout
    except FileNotFoundError as exc:
        res.error = str(exc)
    sim = os.path.join(outdir, "obj_dir", "gpufuzz_sim")
    if not os.path.exists(sim):
        sim = os.path.join(outdir, "obj_dir", "V" + str(recipe.get("top")) + ".exe")
    res.sim = sim if os.path.exists(sim) else ""
    if res.ok and not res.sim:
        res.ok = False
        res.error = "build succeeded but the simulator binary was not found"
    res.elapsed = time.time() - t0
    write_json(os.path.join(outdir, "build.json"), res.to_json())
    if res.ok:
        log.info("built %s in %.1fs -> %s" % (target, res.elapsed, res.sim))
    else:
        log.error("build failed: %s" % res.error)
    return res


def load_build(outdir: str) -> Optional[Dict[str, Any]]:
    return read_json(os.path.join(outdir, "build.json"))
