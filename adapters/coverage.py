"""Coverage post-processing (RQ2 of the paper).

Two things are needed to report *state interaction RTL coverage*:

1. a coverage database from the simulator (Verilator's ``coverage.dat``), and
2. a set of *marked regions*: source locations identified by static analysis as
   interaction sensitive (warp scheduling, scoreboard updates, memory request /
   response handling, cache management, synchronisation logic).

Coverage items outside the marked regions are dropped, exactly as described in
Section 6.2, so different tools are compared on the same RTL regions.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common.util import table

_COV_LINE = re.compile(r"^C\s+'([^']+)'\s+(\d+)\s+(\d+)\s*$")
_REGION_LINE = re.compile(r"^\s*([^\s:]+)\s*:\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


@dataclass
class RegionSet:
    """Marked interaction-sensitive source regions."""

    regions: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "RegionSet":
        rs = cls()
        if not path or not os.path.exists(path):
            return rs
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                m = _REGION_LINE.match(line)
                if not m:
                    continue
                fname, start, end = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
                rs.regions.setdefault(fname, []).append((start, end))
        return rs

    def contains(self, fname: str, line: int) -> bool:
        for base, ranges in self.regions.items():
            if not (fname == base or fname.endswith(base)):
                continue
            for lo, hi in ranges:
                if lo <= line <= hi:
                    return True
        return False

    def __len__(self) -> int:
        return sum(len(v) for v in self.regions.values())

    def describe(self) -> str:
        return "%d files / %d ranges" % (len(self.regions), len(self))


@dataclass
class CoveragePoint:
    file: str
    line: int
    count: int

    @property
    def key(self) -> str:
        return "%s:%d" % (os.path.basename(self.file), self.line)


def parse_coverage_dat(path: str) -> List[CoveragePoint]:
    """Parse Verilator's ``coverage.dat`` (``C 'file' line count``)."""
    out: List[CoveragePoint] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _COV_LINE.match(line.strip())
            if not m:
                continue
            out.append(CoveragePoint(file=m.group(1), line=int(m.group(2)),
                                     count=int(m.group(3))))
    return out


def interaction_coverage(cov_files: Sequence[str], regions: RegionSet,
                         hit_only: bool = True) -> Dict[str, object]:
    """Covered interaction-sensitive points, merged over several databases."""
    points: Dict[str, CoveragePoint] = {}
    all_points = 0
    for path in cov_files:
        for p in parse_coverage_dat(path):
            all_points += 1
            if hit_only and p.count <= 0:
                continue
            if not regions.contains(p.file, p.line):
                continue
            key = p.key
            prev = points.get(key)
            if prev is None:
                points[key] = p
            else:
                prev.count += p.count
    covered = {k: v.count for k, v in points.items() if v.count > 0}
    return {"interaction_points": len(covered), "points": sorted(covered),
            "all_coverage_items": all_points,
            "region_ranges": len(regions)}


def compare(campaigns: Dict[str, Dict[str, object]]) -> str:
    """Render a Table-4 style comparison from several campaigns."""
    rows = []
    for name, data in sorted(campaigns.items()):
        rows.append([name, str(data.get("interaction_points", 0)),
                     str(data.get("bugs", 0)), str(data.get("iterations", 0))])
    return table(rows, ["tool", "interaction coverage", "bugs", "iterations"])


def mark_regions_from_source(root: str, patterns: Sequence[str],
                             out_path: str) -> int:
    """Produce a region file by grepping interaction-sensitive constructs.

    This is the scriptable half of the static marking step: give it the RTL
    checkout and regular expressions (e.g. ``way_dirty``, ``mshr``,
    ``scoreboard``) and it writes ``file:line`` ranges around each match.
    """
    import fnmatch
    hits = 0
    lines_out: List[str] = ["# interaction-sensitive regions marked by mark_regions_from_source"]
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith((".sv", ".v", ".svh", ".vh", ".scala")):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.readlines()
            except OSError:
                continue
            for pat in patterns:
                rx = re.compile(pat)
                for i, line in enumerate(content, 1):
                    if rx.search(line):
                        lines_out.append("%s:%d-%d" % (rel, i, min(i + 24, len(content))))
                        hits += 1
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(set(lines_out))) + "\n")
    return hits
