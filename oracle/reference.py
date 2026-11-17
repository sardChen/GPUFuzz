"""Reference model: a lane-parallel golden interpreter over the SIMT IR.

This is the "available reference model" of Section 4.4 -- it executes the same
``K`` that was handed to the DUT and produces the expected architectural effects
``A``.  Like any differential GPU oracle it compares *architectural effects*, not
cycles, so legal variations in warp scheduling and memory ordering are tolerated
by construction.

Notes on the modelled semantics (documented so the oracle's expectations are not
mistaken for RTL behaviour):

* registers are 32 bit, ``r0`` is hardwired to zero, arithmetic wraps;
* every kernel launch starts with an architecturally zeroed register file
  (this is the property bug #8 violates);
* the driver writes a launch parameter block at ``gmem_base + 0x7F0``:
  ``[0]`` launch index, ``[1]`` grid blocks, ``[2]`` threads per block;
* barriers do not constrain per-lane results, so the golden model executes lanes
  independently;
* ``fcvt_<rm>`` saturates on overflow: directed modes (rtz/rdn) return the
  largest finite magnitude, rne/rup return infinity (RVV/Vortex conversion
  family);
* ``tex`` results are architectural unknowns: their taint is propagated so the
  functional detector does not compare addresses derived from texture data.  The
  texture dependency itself is checked by ``OM_TEX_OFFSET_FRESH`` from the
  observed mipmap state, which is implementation independent.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from common.config import Config
from common.ir import Instr, MemRef, Program
from common.targets import TargetSpec
from common.types import ArchEffects, TestCase

MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000
F32_MAX_BITS = 0x7F7FFFFF
F32_INF = 0x7F800000
F32_QNAN = 0x7FC00000
TEX_OFFSET_DEFAULT = 0x0

# launch parameter block offset (relative to config.gmem_base)
LAUNCH_PARAM_OFFSET = 0x7F0


def u32(x: int) -> int:
    return x & MASK32


def s32(x: int) -> int:
    x &= MASK32
    return x - 0x100000000 if x & SIGN32 else x


def f32_from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def bits_from_f32(v: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:                      # pragma: no cover
        return F32_INF if v > 0 else (F32_INF | SIGN32)


def _is_nan_bits(b: int) -> bool:
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def _canon_nan(b: int) -> int:
    return F32_QNAN if (b & 0x80000000) == 0 else (F32_QNAN | SIGN32)


def _round_f32(x: float) -> float:
    return f32_from_bits(bits_from_f32(x))


# --------------------------------------------------------------------------- #
# IEEE-754 binary32 helpers
# --------------------------------------------------------------------------- #


def fp_binop(op: str, abits: int, bbits: int) -> int:
    a, b = f32_from_bits(abits), f32_from_bits(bbits)
    if _is_nan_bits(abits) or _is_nan_bits(bbits):
        # 0 * inf and inf * 0 are invalid, not "NaN propagates"
        if op in ("mul", "div") and ((math.isinf(a) and b == 0.0) or (math.isinf(b) and a == 0.0)):
            return F32_QNAN
        return _canon_nan(abits if _is_nan_bits(abits) else bbits)
    if op == "add":
        r = a + b
        if math.isinf(a) and math.isinf(b) and (a > 0) != (b > 0):
            return F32_QNAN            # inf + -inf
    elif op == "sub":
        r = a - b
        if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
            return F32_QNAN
    elif op == "mul":
        r = a * b
        if (math.isinf(a) and b == 0.0) or (math.isinf(b) and a == 0.0):
            return F32_QNAN
    elif op == "div":
        if b == 0.0:
            if a == 0.0:
                return F32_QNAN
            return F32_INF | (SIGN32 if (math.copysign(1.0, a) != math.copysign(1.0, b)) else 0)
        r = a / b
    elif op == "sqrt":
        if a < 0.0:
            return F32_QNAN
        r = math.sqrt(a)
    else:                                            # pragma: no cover
        raise ValueError("bad fp op %r" % (op,))
    try:
        return bits_from_f32(r)
    except OverflowError:                            # pragma: no cover
        return F32_INF | (SIGN32 if r < 0 else 0)


def fp_cvt_saturating(src_bits: int, mode: str) -> int:
    """``fcvt_<mode>``: integer valued f32 with saturating overflow handling."""
    x = f32_from_bits(src_bits)
    if math.isnan(x):
        return F32_QNAN
    if math.isinf(x):
        if mode in ("rtz", "rdn"):
            return F32_MAX_BITS | (SIGN32 if x < 0 else 0)
        return F32_INF | (SIGN32 if x < 0 else 0)
    if abs(x) < 2.0 ** 24:
        if mode == "rtz":
            r = math.trunc(x)
        elif mode == "rup":
            r = math.ceil(x)
        elif mode == "rdn":
            r = math.floor(x)
        else:
            r = round(x)                     # banker's rounding ~ round-to-nearest-even
        return bits_from_f32(float(r))
    # already integral at this magnitude; only overflow can still matter
    if abs(x) > 3.4028235677973366e38:
        if mode in ("rtz", "rdn"):
            return F32_MAX_BITS | (SIGN32 if x < 0 else 0)
        return F32_INF | (SIGN32 if x < 0 else 0)
    return src_bits


def sat_add(a: int, b: int) -> int:
    sa, sb = s32(a), s32(b)
    r = sa + sb
    if r > 0x7FFFFFFF:
        return 0x7FFFFFFF
    if r < -0x80000000:
        return 0x80000000
    return u32(r)


def sat_sub(a: int, b: int) -> int:
    return sat_add(a, u32(-s32(b)))


# --------------------------------------------------------------------------- #
# golden execution
# --------------------------------------------------------------------------- #


@dataclass
class ReferenceResult:
    arch: ArchEffects
    unpredictable: Set[int] = field(default_factory=set)   # addresses with tainted data
    executions: int = 0
    steps: int = 0
    notes: List[str] = field(default_factory=list)


class ReferenceModel:
    """Executes ``T = <K, C>`` and returns the expected architectural effects."""

    def __init__(self, target: TargetSpec, max_steps_per_lane: int = 200000) -> None:
        self.target = target
        self.max_steps = max_steps_per_lane

    # ------------------------------------------------------------------ #
    def run(self, tc: TestCase) -> ReferenceResult:
        prog: Program = tc.program
        cfg: Config = tc.config
        lanes = int(self.target.caps["lanes"])
        warps = max(1, cfg.threads_per_block // lanes)
        blocks = cfg.blocks

        gmem: Dict[int, int] = {}
        for addr, val in cfg.gmem_init:
            gmem[addr & MASK32] = u32(val)
        arch = ArchEffects(status={"exit": 0, "launches": len(cfg.launches),
                                   "blocks": blocks, "warps": warps})
        unpredictable: Set[int] = set()
        labels = prog.labels()
        steps_total = 0
        executions = 0

        for lidx, launch in enumerate(cfg.launches):
            base = cfg.gmem_base
            gmem[base + LAUNCH_PARAM_OFFSET] = lidx & MASK32
            gmem[base + LAUNCH_PARAM_OFFSET + 4] = blocks & MASK32
            gmem[base + LAUNCH_PARAM_OFFSET + 8] = cfg.threads_per_block & MASK32
            for blk in range(blocks):
                smem: Dict[int, int] = {}
                for warp in range(warps):
                    for lane in range(lanes):
                        regs, pc, taint, steps = self._run_lane(
                            prog, cfg, gmem, smem, labels, blk, warp, lane, lidx,
                            blocks)
                        steps_total += steps
                        executions += 1
                        for i, val in enumerate(regs):
                            arch.regs["w%d.l%d.r%d" % (warp, lane, i)] = u32(val)
                        arch.pc["w%d.l%d" % (warp, lane)] = pc
                # merge memory of the block into the archival image
                for addr in sorted(gmem):
                    arch.mem[addr] = gmem[addr]
        # only keep the memory window the test can legitimately touch
        lo = cfg.gmem_base - 0x100
        hi = cfg.gmem_base + max(prog.gmem_bytes, 0x2000)
        arch.mem = {a: v for a, v in arch.mem.items() if lo <= a < hi}
        return ReferenceResult(arch=arch, unpredictable=unpredictable,
                               executions=executions, steps=steps_total)

    # ------------------------------------------------------------------ #
    def dynamic_trace(self, tc: TestCase, warp: int = 0, lane: int = 0,
                      block: int = 0) -> List[Tuple[int, Instr, List[int]]]:
        """Executed ``(pc, insn, register file snapshot)`` -- used by the trace
        fixtures and by the minimizer, never by the oracle itself."""
        prog, cfg = tc.program, tc.config
        labels = prog.labels()
        lanes = int(self.target.caps["lanes"])
        warps = max(1, cfg.threads_per_block // lanes)
        gmem = {a & MASK32: u32(v) for a, v in cfg.gmem_init}
        gmem[cfg.gmem_base + LAUNCH_PARAM_OFFSET] = 0
        gmem[cfg.gmem_base + LAUNCH_PARAM_OFFSET + 4] = blocks_of(cfg)
        gmem[cfg.gmem_base + LAUNCH_PARAM_OFFSET + 8] = cfg.threads_per_block
        dyn: List[Tuple[int, Instr, List[int]]] = []
        self._run_lane(prog, cfg, gmem, {}, labels, block, warp, lane, 0,
                       blocks_of(cfg), dynamic=dyn)
        return dyn

    def _run_lane(self, prog: Program, cfg: Config, gmem: Dict[int, int],
                  smem: Dict[int, int], labels: Dict[str, int], blk: int,
                  warp: int, lane: int, lidx: int, blocks: int,
                  dynamic: Optional[List[Tuple[int, Instr, List[int]]]] = None
                  ) -> Tuple[List[int], int, List[bool], int]:
        regs = [0] * max(prog.nregs, 32)
        taint = [False] * len(regs)
        pc = 0
        steps = 0
        tid = warp * int(self.target.caps["lanes"]) + lane
        code_base = 0x0
        while 0 <= pc < len(prog.instrs):
            steps += 1
            if steps > self.max_steps:              # pragma: no cover - defensive
                break
            ins = prog.instrs[pc]
            if dynamic is not None:
                dynamic.append((pc, ins, list(regs)))
            nxt = pc + 1
            op = ins.op

            def src(i: int = 0) -> int:
                v = ins.srcs[i] if i < len(ins.srcs) else 0
                return regs[_ri(v)] if isinstance(v, str) else u32(int(v))

            def tnt(i: int = 0) -> bool:
                v = ins.srcs[i] if i < len(ins.srcs) else 0
                return taint[_ri(v)] if isinstance(v, str) else False

            def write(dst: Optional[str], val: int, tainted: bool = False) -> None:
                if not dst:
                    return
                i = _ri(dst)
                if i == 0:
                    return                              # r0 is hardwired zero
                regs[i] = u32(val)
                taint[i] = tainted

            if op == "mov":
                write(ins.dst, src(0), tnt(0))
            elif op == "mov_imm":
                write(ins.dst, ins.imm)
            elif op in ("add", "sub", "and", "or", "xor", "shl", "shr", "sra",
                        "slt", "sltu", "min", "max", "mul", "mulh", "mad"):
                a, b = src(0), src(1)
                if op == "add":
                    r = a + b
                elif op == "sub":
                    r = a - b
                elif op == "and":
                    r = a & b
                elif op == "or":
                    r = a | b
                elif op == "xor":
                    r = a ^ b
                elif op == "shl":
                    r = a << (b & 31)
                elif op == "shr":
                    r = a >> (b & 31)
                elif op == "sra":
                    r = s32(a) >> (b & 31)
                elif op == "slt":
                    r = 1 if s32(a) < s32(b) else 0
                elif op == "sltu":
                    r = 1 if u32(a) < u32(b) else 0
                elif op == "min":
                    r = s32(a) if s32(a) < s32(b) else s32(b)
                elif op == "max":
                    r = s32(a) if s32(a) > s32(b) else s32(b)
                elif op == "mul":
                    r = s32(a) * s32(b)
                elif op == "mulh":
                    r = (s32(a) * s32(b)) >> 32
                else:  # mad: dst = a*b + imm
                    r = s32(a) * s32(b) + ins.imm
                write(ins.dst, r, tnt(0) or tnt(1))
            elif op == "div":
                a, b = s32(src(0)), s32(src(1))
                if b == 0:
                    r = MASK32 if a >= 0 else 1        # all ones / min-int convention
                else:
                    q = abs(a) // abs(b)
                    r = -q if (a < 0) != (b < 0) else q
                write(ins.dst, r, tnt(0) or tnt(1))
            elif op == "rem":
                a, b = s32(src(0)), s32(src(1))
                if b == 0:
                    r = a
                else:
                    q = abs(a) // abs(b)
                    q = -q if (a < 0) != (b < 0) else q
                    r = a - q * b
                write(ins.dst, r, tnt(0) or tnt(1))
            elif op == "neg":
                write(ins.dst, -s32(src(0)), tnt(0))
            elif op == "abs":
                write(ins.dst, abs(s32(src(0))), tnt(0))
            elif op == "sat_add":
                write(ins.dst, sat_add(src(0), src(1)), tnt(0) or tnt(1))
            elif op == "sat_sub":
                write(ins.dst, sat_sub(src(0), src(1)), tnt(0) or tnt(1))
            elif op in ("fadd", "fsub", "fmul", "fdiv"):
                opmap = {"fadd": "add", "fsub": "sub", "fmul": "mul", "fdiv": "div"}
                write(ins.dst, fp_binop(opmap[op], src(0), src(1)), tnt(0) or tnt(1))
            elif op == "fsqrt":
                write(ins.dst, fp_binop("sqrt", src(0), 0), tnt(0))
            elif op in ("f2i",):
                v = f32_from_bits(src(0))
                write(ins.dst, 0x7FFFFFFF if (math.isnan(v) or v > 2 ** 31 - 1)
                      else (0x80000000 if v < -2 ** 31 else int(v)))
            elif op == "i2f":
                write(ins.dst, bits_from_f32(float(s32(src(0)))))
            elif op.startswith("fcvt_"):
                write(ins.dst, fp_cvt_saturating(src(0), op.split("_", 1)[1]))
            elif op == "fneg":
                write(ins.dst, src(0) ^ SIGN32)
            elif op == "fabs":
                write(ins.dst, src(0) & ~SIGN32)
            elif op in ("ld", "st"):
                m = ins.mem or MemRef()
                base = regs[_ri(m.base)] if m.base else 0
                addr = u32(base + m.offset)
                if m.space == "shared":
                    addr &= 0xFFFF
                    mem = smem
                else:
                    mem = gmem
                if op == "ld":
                    if m.width == 4:
                        write(ins.dst, mem.get(addr & ~3, 0))
                    elif m.width == 2:
                        w = mem.get(addr & ~3, 0)
                        write(ins.dst, (w >> (8 * (addr & 2))) & 0xFFFF)
                    else:
                        w = mem.get(addr & ~3, 0)
                        write(ins.dst, (w >> (8 * (addr & 3))) & 0xFF)
                else:
                    val = src(0)
                    if m.width == 4:
                        mem[addr & ~3] = u32(val)
                    elif m.width == 2:
                        old = mem.get(addr & ~3, 0)
                        sh = 8 * (addr & 2)
                        mem[addr & ~3] = (old & ~(0xFFFF << sh)) | ((val & 0xFFFF) << sh)
                    else:
                        old = mem.get(addr & ~3, 0)
                        sh = 8 * (addr & 3)
                        mem[addr & ~3] = (old & ~(0xFF << sh)) | ((val & 0xFF) << sh)
            elif op == "tex":
                # architectural unknown: deterministic stand-in, tainted
                coord = src(0)
                level = int(ins.aux.get("level", 0))
                write(ins.dst, (coord * 2654435761 + (level << 8)) & MASK32, True)
            elif op in ("tid", "laneid"):
                write(ins.dst, tid if op == "tid" else lane)
            elif op == "ntid":
                write(ins.dst, cfg.threads_per_block)
            elif op == "ctaid":
                write(ins.dst, blk)
            elif op == "nctaid":
                write(ins.dst, blocks)
            elif op == "warpid":
                write(ins.dst, warp)
            elif op == "clock":
                write(ins.dst, 0)
            elif op == "getcfg":
                name = ins.aux.get("name", "")
                val = {"launch_id": lidx, "queue_arg": launch_arg(cfg, lidx),
                       "nctaid": blocks_of(cfg)}.get(name, 0)
                write(ins.dst, int(val))
            elif op in ("setcfg", "cache_op", "memcpy_len", "nop", "spin", "fence", "bar"):
                pass
            elif op == "la":
                lbl = ins.label
                write(ins.dst, (code_base + labels.get(lbl, 0) * 4) if lbl in labels else 0)
            elif op == "br":
                cond = src(0)
                sense = ins.aux.get("sense", "nz")
                taken = (cond != 0) if sense == "nz" else (cond == 0)
                if taken and ins.label in labels:
                    nxt = labels[ins.label]
            elif op == "jmp":
                nxt = labels.get(ins.label, nxt)
            elif op == "jmp_reg":
                target = src(0) & ~1
                if target % 4 == 0 and 0 <= target // 4 < len(prog.instrs):
                    nxt = target // 4
            elif op == "exit":
                break
            elif op == "label":
                pass
            else:                                       # pragma: no cover
                raise ValueError("reference model: unknown op %r" % (op,))
            pc = nxt if 0 <= nxt <= len(prog.instrs) else len(prog.instrs)
        return regs, pc, taint, steps


def blocks_of(cfg: Config) -> int:
    g = cfg.grid
    return max(1, g[0]) * max(1, g[1]) * max(1, g[2])


def launch_arg(cfg: Config, lidx: int) -> int:
    if 0 <= lidx < len(cfg.launches):
        return int(cfg.launches[lidx].arg)
    return 0


def _ri(name: str) -> int:
    try:
        return int(name[1:])
    except (TypeError, ValueError):                     # pragma: no cover
        return 0
