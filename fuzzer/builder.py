"""Program builder used by the shape templates.

Templates describe *what* has to happen (setup / trigger / observation); the
builder takes care of the mechanical parts: register allocation, constant
materialisation, address formation and the observation window, and it tags
instructions with *slots* so the dependency-preserving mutator knows which
instructions belong to a protected trigger condition.
"""
from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from common.ir import Instr, MemRef, Program, is_reg, reg_index

# memory layout inside the test's global window (relative to config.gmem_base)
OBS_OFFSET = 0x800          # observation buffer: 64 B per thread
OBS_STRIDE = 64
OBS_SHIFT = 6               # log2(OBS_STRIDE)
DATA_OFFSET = 0x0           # data area used by store/load sequences
LINE_SIZE = 64              # cache line granularity used by the templates

ZERO_REG = "r0"
# Registers the shape templates address by name; the builder never hands them out
# as scratch, so a materialised constant can never be clobbered by a template.
# Mapping used by the assembler backends (see common/isa/emit.py):
#   r0=x0 (zero)  r1=x1 (thread id)  r2=x2 (stack pointer, reserved for macros)
#   r5..r12=x5..x12 (templates)      r13..r31=x13..x31 (scratch)
# Macro contract: a GPU_* macro may use x2/sp for spilling but must not clobber
# any x1 or x5..x31 register.
RESERVED_REGS = (5, 6, 7, 8, 9, 10, 11, 12)
SCRATCH_BASE = 13
SCRATCH_LAST = 31          # the whole 13..31 range is available to the kernel


class ProgramBuilder:
    def __init__(self, nregs: int = 32, gmem_base: int = 0x1000,
                 smem_bytes: int = 0, name: str = "") -> None:
        self.p = Program(nregs=nregs, smem_bytes=smem_bytes, name=name,
                         gmem_bytes=0x10000)
        self.gmem_base = gmem_base
        self._used: set = set(RESERVED_REGS)
        self._consts: Dict[int, str] = {}
        self._slot = ""
        self.protected: List[str] = []
        self._obs_base: Optional[str] = None
        self._dbase: Optional[str] = None
        self._tid: Optional[str] = None
        self._slot_seq = 0

    # -- emitting ---------------------------------------------------------- #
    def emit(self, op: str, **kw: Any) -> Instr:
        kw.setdefault("slot", self._slot)
        ins = Instr(op=op, **kw)
        self.p.instrs.append(ins)
        for r in ins.regs_written():
            self._used.add(reg_index(r))
        return ins

    @contextlib.contextmanager
    def slot(self, name: str, protect: bool = False) -> Iterator[None]:
        """Tag emitted instructions with a provenance slot."""
        prev = self._slot
        self._slot_seq += 1
        self._slot = "%s#%d" % (name, self._slot_seq)
        if protect and self._slot not in self.protected:
            self.protected.append(self._slot)
        try:
            yield
        finally:
            self._slot = prev

    # -- registers --------------------------------------------------------- #
    def reg(self, avoid: Iterable[str] = ()) -> str:
        """Allocate a scratch register from the non-reserved pool."""
        avoid_idx = {reg_index(r) for r in avoid if is_reg(r)}
        for i in range(SCRATCH_BASE, min(self.p.nregs, SCRATCH_LAST + 1)):
            if i in self._used or i in avoid_idx:
                continue
            self._used.add(i)
            return "r%d" % i
        for i in range(4, SCRATCH_BASE):
            if i in self._used or i in avoid_idx:
                continue
            self._used.add(i)
            return "r%d" % i
        raise RuntimeError("out of registers (%d reserved)" % (self.p.nregs,))

    def free(self, *regs: str) -> None:
        for r in regs:
            if is_reg(r) and reg_index(r) >= SCRATCH_BASE:
                self._used.discard(reg_index(r))

    def imm(self, value: int, dst: Optional[str] = None) -> str:
        """Return a register that holds ``value`` (materialised once)."""
        value &= 0xFFFFFFFF
        if dst:
            self.emit("mov_imm", dst=dst, imm=value)
            return dst
        if value in self._consts:
            return self._consts[value]
        r = self.reg()
        self.emit("mov_imm", dst=r, imm=value)
        self._consts[value] = r
        return r

    # -- standard prologue ------------------------------------------------- #
    @property
    def tid(self) -> str:
        if self._tid is None:
            self._tid = "r1"
            self.emit("tid", dst="r1", aux={"dim": 0})
        return self._tid

    @property
    def obs_base(self) -> str:
        """Register holding ``gmem_base + OBS_OFFSET + 4*tid``."""
        if self._obs_base is not None:
            return self._obs_base
        base = self.reg()
        off = self.reg()
        self.emit("mov_imm", dst=base, imm=self.gmem_base + OBS_OFFSET)
        self.emit("shl", dst=off, srcs=(self.tid, self.imm(OBS_SHIFT)))
        self.emit("add", dst=base, srcs=(base, off))
        self.free(off)
        self._obs_base = base
        return base

    @property
    def data_base(self) -> str:
        """Register holding ``gmem_base + DATA_OFFSET`` (cached)."""
        if self._dbase is None:
            r = self.reg()
            self.emit("mov_imm", dst=r, imm=self.gmem_base + DATA_OFFSET)
            self._dbase = r
        return self._dbase

    def prologue(self) -> None:
        _ = self.tid  # r1 = tid.x
        _ = self.data_base

    # -- memory ------------------------------------------------------------ #
    def store(self, value: str, base: str, offset: int, space: str = "global",
              width: int = 4) -> None:
        self.emit("st", srcs=(value,), mem=MemRef(space=space, base=base,
                                                  offset=offset, width=width))

    def load(self, dst: str, base: str, offset: int, space: str = "global",
             width: int = 4) -> None:
        self.emit("ld", dst=dst, mem=MemRef(space=space, base=base,
                                            offset=offset, width=width))

    def observe(self, value: str, index: int) -> None:
        """Store ``value`` into this thread's observation slot ``index``."""
        self.emit("st", srcs=(value,),
                  mem=MemRef(space="global", base=self.obs_base,
                             offset=index * 4, width=4), slot=self._slot or "obs")

    def barrier(self) -> None:
        self.emit("bar", imm=0)

    def fence(self) -> None:
        self.emit("fence")

    def exit(self) -> None:
        self.emit("exit")

    def finish(self) -> None:
        self.p.instrs.append(Instr(op="exit", slot="finish"))

    # -- addresses --------------------------------------------------------- #
    def addr_of(self, base: str, offset: int) -> str:
        """Materialise ``base + offset`` into a fresh register."""
        if offset == 0:
            return base
        r = self.reg(avoid=(base,))
        self.emit("add", dst=r, srcs=(base, self.imm(offset)))
        return r

    # -- branching --------------------------------------------------------- #
    def label(self, name: str) -> str:
        self.emit("label", label=name)
        return name

    def br(self, cond: str, label: str, sense: str = "nz") -> None:
        self.emit("br", srcs=(cond,), label=label, aux={"sense": sense})

    # -- helpers used by several templates --------------------------------- #
    def set_pred(self, dst: str, cond: str) -> None:
        """``dst = (cond != 0)`` -- used to turn predicates into data."""
        self.emit("sltu", dst=dst, srcs=(ZERO_REG, cond))

    def make_predicate(self, lhs: str, rhs: str, op: str = "slt") -> str:
        dst = self.reg(avoid=(lhs, rhs))
        self.emit(op, dst=dst, srcs=(lhs, rhs))
        return dst


@dataclass
class GenPlan:
    """What a shape template produces."""

    program: Program
    config: Any
    protected: Tuple[str, ...] = ()
    knobs: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> str:
        return ("%d insns (%s), %s, protected=%s"
                % (len(self.program), ",".join(sorted(self.program.opcodes())),
                   self.config.describe(), ",".join(self.protected) or "-"))
