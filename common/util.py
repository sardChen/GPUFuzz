"""Small, dependency-free helpers shared by the fuzzer, the oracle and the adapters.

The whole project is stdlib-only so that it runs on any machine that has a
``python3`` (>= 3.8).  Nothing here is GPU specific.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# identifiers / hashing
# --------------------------------------------------------------------------- #


def sha(*parts: Any, n: int = 12) -> str:
    """Short, stable hash over arbitrary json-able parts (deterministic)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def slug(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^0-9a-zA-Z_.-]+", "-", text.strip()).strip("-").lower()
    return s[:maxlen] or "x"


class Counter:
    """Deterministic monotone id source (per process, seeded by name)."""

    def __init__(self, prefix: str = "id", start: int = 0) -> None:
        self.prefix = prefix
        self.n = start

    def __call__(self) -> str:
        self.n += 1
        return "%s-%06d" % (self.prefix, self.n)


# --------------------------------------------------------------------------- #
# json io
# --------------------------------------------------------------------------- #


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("%s:%d: bad jsonl: %s" % (path, lineno, exc))


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


def append_jsonl(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_json"):
        return o.to_json()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def ensure_dir(path: str) -> str:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:  # pragma: no cover - windows cross-drive
        return path


# --------------------------------------------------------------------------- #
# deterministic randomness
# --------------------------------------------------------------------------- #


def rng_for(*parts: Any) -> random.Random:
    """A ``random.Random`` seeded from the stable hash of ``parts``."""
    return random.Random(int(sha(*parts, n=16), 16))


def weighted_choice(rng: random.Random, items: Sequence[Any], weights: Sequence[float]) -> Any:
    if not items:
        raise ValueError("weighted_choice on empty sequence")
    total = float(sum(weights))
    if total <= 0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


# --------------------------------------------------------------------------- #
# time / budget
# --------------------------------------------------------------------------- #


class Budget:
    """Wall-clock or iteration budget for a fuzzing campaign.

    ``limit`` is either seconds (time budget) or a count (iteration budget),
    depending on ``mode``.  ``remaining()`` is what the loop checks.
    """

    def __init__(self, limit: float, mode: str = "iters") -> None:
        self.limit = float(limit)
        self.mode = mode
        self.start = time.time()
        self.used = 0.0

    def tick(self, n: float = 1.0) -> None:
        self.used += n

    def remaining(self) -> float:
        if self.mode == "time":
            return self.limit - (time.time() - self.start)
        return self.limit - self.used

    def exhausted(self) -> bool:
        return self.remaining() <= 0

    def elapsed(self) -> float:
        return time.time() - self.start


@contextlib.contextmanager
def timer() -> Iterator[List[float]]:
    box = [0.0]
    t0 = time.time()
    try:
        yield box
    finally:
        box[0] = time.time() - t0


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


class Log:
    """Line oriented logger with a level and an optional file sink."""

    LEVELS = {"quiet": 0, "error": 1, "warn": 2, "info": 3, "debug": 4, "trace": 5}

    def __init__(self, level: str = "info", path: Optional[str] = None) -> None:
        self.level = self.LEVELS.get(level, 3)
        self.fh = None
        if path:
            ensure_dir(os.path.dirname(path) or ".")
            self.fh = open(path, "a", encoding="utf-8")

    def _emit(self, lvl: str, msg: str) -> None:
        if self.LEVELS[lvl] > self.level:
            return
        line = "[%s] %s %s" % (lvl.upper()[:4], time.strftime("%H:%M:%S"), msg)
        print(line, file=sys.stderr)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def error(self, msg: str) -> None:
        self._emit("error", msg)

    def warn(self, msg: str) -> None:
        self._emit("warn", msg)

    def info(self, msg: str) -> None:
        self._emit("info", msg)

    def debug(self, msg: str) -> None:
        self._emit("debug", msg)

    def trace(self, msg: str) -> None:
        self._emit("trace", msg)

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #


def hexint(x: int) -> str:
    return "0x%x" % (x & 0xFFFFFFFFFFFFFFFF)


def bit(x: int, i: int) -> int:
    return (x >> i) & 1


def popcount(x: int) -> int:
    return bin(x & 0xFFFFFFFFFFFFFFFF).count("1")


def indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.splitlines())


def table(rows: List[List[str]], header: Optional[List[str]] = None) -> str:
    """Render a plain text table (used by reports and the CLI)."""
    all_rows = ([header] if header else []) + rows
    if not all_rows:
        return ""
    ncol = max(len(r) for r in all_rows)
    widths = [
        max(len(str(r[i])) if i < len(r) else 0 for r in all_rows) for i in range(ncol)
    ]
    out = []
    for idx, r in enumerate(all_rows):
        cells = [str(r[i]).ljust(widths[i]) if i < len(r) else " " * widths[i] for i in range(ncol)]
        out.append("  ".join(cells).rstrip())
        if header and idx == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / float(len(sa | sb))
