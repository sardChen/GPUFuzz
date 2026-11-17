"""A small SIMT program IR.

Every test case is ``T = <K, C>`` where ``K`` is a valid SIMT program.  The IR
below is the single representation that:

* the generator synthesizes (setup / trigger / observation sequences),
* the mutator rewrites (dependency-preserving mutations),
* the reference model interprets (golden architectural effects), and
* the per-target emitters translate into assembly of the target ISA.

It is deliberately tiny: straight-line code, forward branches only (so every
program terminates), one hardwired zero register ``r0``, and explicit memory
spaces.  Warp divergence comes from data-dependent forward branches, which is
what the targets' SIMT front ends have to handle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .util import sha

# --------------------------------------------------------------------------- #
# opcode set
# --------------------------------------------------------------------------- #

# integer ALU
ALU_OPS = ("mov", "mov_imm", "add", "sub", "mul", "mad", "div", "rem",
           "and", "or", "xor", "shl", "shr", "sra", "slt", "sltu",
           "min", "max", "neg", "abs", "sat_add", "sat_sub", "mulh")
# floating point (IEEE-754 binary32 kept in the low 32 bits of a register)
FP_OPS = ("fadd", "fsub", "fmul", "fdiv", "fsqrt", "f2i", "i2f",
          "fcvt_rtz", "fcvt_rup", "fcvt_rdn", "fcvt_rne", "fneg", "fabs")
# memory
MEM_OPS = ("ld", "st")
# control / special
CTRL_OPS = ("label", "br", "jmp", "jmp_reg", "la", "bar", "fence", "exit", "nop", "spin")
# target specific units
SPECIAL_OPS = ("tex", "cache_op", "tid", "ntid", "ctaid", "nctaid", "laneid",
               "warpid", "clock", "setcfg", "getcfg", "memcpy_len")

OPS = frozenset(ALU_OPS + FP_OPS + MEM_OPS + CTRL_OPS + SPECIAL_OPS)

# op -> feature tag used for target capability checks
OP_FEATURE = {
    "div": "int_div", "rem": "int_div", "mulh": "int_div", "mul": "mul", "mad": "mul",
    "sat_add": "sat", "sat_sub": "sat",
    "fadd": "fp32", "fsub": "fp32", "fmul": "fp32", "fdiv": "fp32", "fsqrt": "fp32",
    "f2i": "fp32", "i2f": "fp32", "fcvt_rtz": "fp32", "fcvt_rup": "fp32",
    "fcvt_rdn": "fp32", "fcvt_rne": "fp32", "fneg": "fp32", "fabs": "fp32",
    "tex": "texture", "cache_op": "cache_control", "setcfg": "cfg_regs",
    "getcfg": "cfg_regs", "memcpy_len": "dma", "bar": "barrier",
}

# memory spaces
SPACES = ("global", "shared", "local")


@dataclass(frozen=True)
class MemRef:
    """A memory operand: ``space[base + offset]`` with a fixed access width."""

    space: str = "global"
    base: Optional[str] = None      # register holding the base address
    offset: int = 0                 # static byte offset
    width: int = 4                  # 1, 2 or 4 bytes

    def __post_init__(self) -> None:
        if self.space not in SPACES:
            raise ValueError("bad memory space %r" % (self.space,))
        if self.width not in (1, 2, 4):
            raise ValueError("bad access width %r" % (self.width,))

    def to_json(self) -> Dict[str, Any]:
        return {"space": self.space, "base": self.base, "offset": self.offset,
                "width": self.width}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "MemRef":
        return MemRef(space=d.get("space", "global"), base=d.get("base"),
                      offset=int(d.get("offset", 0)), width=int(d.get("width", 4)))


@dataclass
class Instr:
    """One IR instruction.

    ``srcs`` holds register names (``"r5"``) or integer immediates.  ``slot`` is
    a provenance tag assigned by the generator templates; the dependency
    preserving mutator protects slots listed in ``TestCase.protected``.
    """

    op: str
    dst: Optional[str] = None
    srcs: Tuple[Any, ...] = ()
    imm: int = 0
    mem: Optional[MemRef] = None
    label: Optional[str] = None
    aux: Dict[str, Any] = field(default_factory=dict)
    slot: str = ""

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError("unknown op %r" % (self.op,))
        if self.dst is not None and not is_reg(self.dst):
            raise ValueError("bad dst register %r" % (self.dst,))
        for s in self.srcs:
            if isinstance(s, str) and not is_reg(s):
                raise ValueError("bad src register %r" % (s,))

    # -- helpers ----------------------------------------------------------- #
    def regs_read(self) -> List[str]:
        out = [s for s in self.srcs if isinstance(s, str) and is_reg(s)]
        if self.mem and self.mem.base:
            out.append(self.mem.base)
        return out

    def regs_written(self) -> List[str]:
        if self.op in ("st", "br", "jmp", "jmp_reg", "label", "bar", "fence", "exit",
                       "nop", "spin", "cache_op", "setcfg", "memcpy_len"):
            return []
        return [self.dst] if self.dst else []

    def feature(self) -> Optional[str]:
        return OP_FEATURE.get(self.op)

    def text(self) -> str:
        """Human readable form (also used by emitters for comments)."""
        if self.op == "label":
            return "%s:" % (self.label,)
        if self.op == "la":
            return "la %s, %s" % (self.dst, self.label)
        if self.op == "jmp_reg":
            return "jmp_reg %s" % (self.srcs[0] if self.srcs else "r0")
        if self.op == "br":
            return "br %s, %s, %s" % (self.srcs[0] if self.srcs else "r0",
                                      self.label, self.aux.get("sense", "nz"))
        if self.op in ("ld", "st"):
            m = self.mem or MemRef()
            base = m.base or "0"
            expr = "%s[%s%+#d]" % (m.space, base, m.offset) if m.offset else "%s[%s]" % (m.space, base)
            if self.op == "ld":
                return "ld %s, %s" % (self.dst, expr)
            return "st %s, %s" % (self.srcs[0] if self.srcs else "r0", expr)
        args = [str(s) for s in self.srcs]
        if self.mem:
            args.append(str(self.mem))
        if self.imm:
            args.append("#%d" % (self.imm & 0xFFFFFFFF))
        if self.aux:
            args.append("{%s}" % ",".join("%s=%s" % kv for kv in sorted(self.aux.items())))
        head = "%s %s" % (self.op, self.dst) if self.dst else self.op
        return (head + (", " + ", ".join(args) if args else "")).strip()

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op": self.op}
        if self.dst:
            d["dst"] = self.dst
        if self.srcs:
            d["srcs"] = list(self.srcs)
        if self.imm:
            d["imm"] = self.imm
        if self.mem:
            d["mem"] = self.mem.to_json()
        if self.label:
            d["label"] = self.label
        if self.aux:
            d["aux"] = self.aux
        if self.slot:
            d["slot"] = self.slot
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Instr":
        return Instr(op=d["op"], dst=d.get("dst"), srcs=tuple(d.get("srcs") or ()),
                     imm=int(d.get("imm", 0)),
                     mem=MemRef.from_json(d["mem"]) if d.get("mem") else None,
                     label=d.get("label"), aux=dict(d.get("aux") or {}),
                     slot=d.get("slot", ""))


REG_RE = None


def is_reg(s: Any) -> bool:
    if not isinstance(s, str) or len(s) < 2 or s[0] != "r":
        return False
    return s[1:].isdigit()


def reg_index(name: str) -> int:
    return int(name[1:])


def reg_name(i: int) -> str:
    return "r%d" % i


# --------------------------------------------------------------------------- #
# program
# --------------------------------------------------------------------------- #


class ProgramError(ValueError):
    pass


@dataclass
class Program:
    """``K``: a SIMT program."""

    instrs: List[Instr] = field(default_factory=list)
    nregs: int = 32
    smem_bytes: int = 0
    gmem_bytes: int = 4096
    name: str = ""

    # -- construction ------------------------------------------------------ #
    def emit(self, op: str, **kw: Any) -> Instr:
        ins = Instr(op=op, **kw)
        self.instrs.append(ins)
        return ins

    def extend(self, ins: Iterable[Instr]) -> None:
        self.instrs.extend(ins)

    def __len__(self) -> int:
        return len(self.instrs)

    def __iter__(self) -> Iterable[Instr]:
        return iter(self.instrs)

    # -- analysis ---------------------------------------------------------- #
    def opcodes(self) -> Set[str]:
        return {i.op for i in self.instrs}

    def features(self) -> Set[str]:
        out: Set[str] = set()
        for i in self.instrs:
            f = i.feature()
            if f:
                out.add(f)
        return out

    def labels(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for idx, ins in enumerate(self.instrs):
            if ins.op == "label":
                if ins.label in out:
                    raise ProgramError("duplicate label %r" % (ins.label,))
                out[ins.label] = idx
        return out

    def uses_regs(self) -> List[int]:
        used: Set[int] = {0}
        for ins in self.instrs:
            for r in ins.regs_read() + ins.regs_written():
                used.add(reg_index(r))
        return sorted(used)

    def text(self) -> str:
        lines = []
        for idx, ins in enumerate(self.instrs):
            lines.append("%4d  %s%s" % (idx, ins.text(),
                                        ("   ;" + ins.slot) if ins.slot else ""))
        return "\n".join(lines)

    def hash(self) -> str:
        return sha([i.to_json() for i in self.instrs], self.nregs, self.smem_bytes, n=16)

    # -- validation -------------------------------------------------------- #
    def validate(self, caps: Optional[Dict[str, Any]] = None) -> List[str]:
        """Return a list of problems; an empty list means the program is valid.

        Checks structural well-formedness, then target capabilities when a
        capability dict is supplied (``{"features": [...], "max_regs": 32}``).
        """
        problems: List[str] = []
        if not self.instrs:
            return ["empty program"]

        lbls = self.labels()
        for idx, ins in enumerate(self.instrs):
            for r in ins.regs_read() + ins.regs_written():
                if reg_index(r) >= self.nregs:
                    problems.append("insn %d: register %s beyond nregs=%d" % (idx, r, self.nregs))
            if ins.op == "br":
                if ins.label not in lbls:
                    problems.append("insn %d: branch to unknown label %r" % (idx, ins.label))
                elif lbls[ins.label] <= idx:
                    problems.append("insn %d: backward branch (non terminating)" % idx)
            if ins.op == "jmp":
                if ins.label not in lbls:
                    problems.append("insn %d: jump to unknown label %r" % (idx, ins.label))
                elif lbls[ins.label] <= idx:
                    problems.append("insn %d: backward jump (non terminating)" % idx)
            if ins.op in ("ld", "st"):
                m = ins.mem or MemRef()
                if m.space == "shared" and self.smem_bytes <= 0:
                    problems.append("insn %d: shared access but smem_bytes=0" % idx)
                if m.space == "shared" and m.offset >= self.smem_bytes:
                    problems.append("insn %d: shared offset %d beyond %d"
                                    % (idx, m.offset, self.smem_bytes))
        if self.instrs[-1].op not in ("exit",):
            problems.append("program does not end with exit")

        # Termination is guaranteed by construction: the final instruction is
        # ``exit`` and no branch/jump may target an earlier instruction.  The
        # mutator preserves both invariants, so the DUT can never be sent a
        # non-terminating program.
        if caps:
            feats = self.features()
            supported = set(caps.get("features") or ())
            missing = sorted(feats - supported)
            if missing:
                problems.append("target lacks features: %s" % ",".join(missing))
            max_regs = int(caps.get("max_regs", 32))
            if self.nregs > max_regs:
                problems.append("nregs=%d exceeds target max_regs=%d" % (self.nregs, max_regs))
            max_smem = int(caps.get("smem_bytes", 0))
            if self.smem_bytes > max_smem:
                problems.append("smem_bytes=%d exceeds target limit %d" % (self.smem_bytes, max_smem))
        return problems

    # -- serialization ----------------------------------------------------- #
    def to_json(self) -> Dict[str, Any]:
        return {"instrs": [i.to_json() for i in self.instrs], "nregs": self.nregs,
                "smem_bytes": self.smem_bytes, "gmem_bytes": self.gmem_bytes,
                "name": self.name}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Program":
        return Program(instrs=[Instr.from_json(i) for i in d.get("instrs", [])],
                       nregs=int(d.get("nregs", 32)),
                       smem_bytes=int(d.get("smem_bytes", 0)),
                       gmem_bytes=int(d.get("gmem_bytes", 4096)),
                       name=d.get("name", ""))

    def clone(self) -> "Program":
        return Program.from_json(self.to_json())
