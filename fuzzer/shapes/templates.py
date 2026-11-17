"""Shape generators: synthesize ``T = <K, C>`` for one ESD relation.

Every template follows the three-sequence structure of Section 4.3:

    setup    establish the execution state the relation needs
    trigger  activate the related events under the temporal constraint theta
    observe  expose the resulting behaviour

The generator only *arranges* the conditions; it never asserts the outcome.  What
counts as correct is decided by the oracle from the obligation omega, not from
the RTL implementation.

All functions take a :class:`GenCtx` and return a :class:`GenPlan`.  Knobs read
from ``g.params`` are the same knobs the mutator perturbs, so a seed and its
mutants stay comparable.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from common.config import Config, EventCondition, Launch
from common.ir import OP_FEATURE
from fuzzer.builder import GenPlan, ProgramBuilder
from .genctx import GenCtx

F32_INF = 0x7F800000
F32_NAN = 0x7FC00000
F32_MAX = 0x7F7FFFFF
F32_ONE = 0x3F800000
F32_HALF = 0x3F000000
F32_BIG = 0x7F000000          # 1.7e38, close to the f32 maximum
F32_NEG = 0xBF800000          # -1.0

# offset (relative to config.gmem_base) of the driver-written launch parameter
# block: [0] launch index, [1] grid blocks, [2] threads per block
LAUNCH_PARAM_OFFSET = 0x7F0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _supported(g: GenCtx, op: str) -> bool:
    feat = OP_FEATURE.get(op)
    return feat is None or feat in set(g.target.caps.get("features") or ())


def _pick_op(g: GenCtx, name: str, choices: Sequence[str], default: str) -> str:
    want = g.param(name, None)
    if isinstance(want, str) and want in choices and _supported(g, want):
        return want
    for op in choices:
        if _supported(g, op):
            return op
    return default


def _obs(p: ProgramBuilder, reg: str, idx: int) -> None:
    p.observe(reg, idx)


def _f32(g: GenCtx, p: ProgramBuilder, bits: int) -> str:
    return p.imm(bits)


# --------------------------------------------------------------------------- #
# 1. scoreboard: issue of a new writer overlapping the writeback of the old one
#    (Ventus bug #1)
# --------------------------------------------------------------------------- #


def gen_sb_issue_wb(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    reg = g.pick("target_reg", "r5", ("r5", "r6", "r7"))
    nfill = int(g.param("filler", g.rng.choice((0, 1, 2, 3, 4))))
    ncons = int(g.param("consumers", g.rng.choice((1, 2))))
    lat = _pick_op(g, "latency_op", ("div", "mul", "fmul"), "mul")
    a = p.imm(int(g.param("lhs", g.rng.choice((3, 5, 7)))))
    b = p.imm(int(g.param("rhs", g.rng.choice((3, 5, 9)))))
    if lat in ("fmul", "fdiv"):
        a, b = _f32(g, p, F32_ONE), _f32(g, p, F32_HALF)

    with p.slot("trigger.old_writer", protect=True):
        p.emit(lat, dst=reg, srcs=(a, b))          # A: long latency, writes reg
    fillers = []
    for i in range(nfill):
        r = p.reg(avoid=(reg,))
        p.emit("add", dst=r, srcs=(b, p.imm(0x11 + i)))
        fillers.append(r)
    with p.slot("trigger.new_writer", protect=True):
        p.emit("add", dst=reg, srcs=(b, p.imm(7)))  # B: WAW on reg, independent of A
    consumers = []
    with p.slot("trigger.consumer", protect=True):
        for i in range(ncons):
            r = p.reg(avoid=(reg,))
            p.emit("add", dst=r, srcs=(reg, p.imm(1 + i)))
            consumers.append(r)
    for i, r in enumerate(consumers):
        _obs(p, r, i)
    _obs(p, reg, len(consumers))
    for r in fillers:
        p.free(r)
    p.finish()
    cfg = g.config(p, warps=int(g.pick_warps("warps", 1, (1, 2, 4))), blocks=1,
                   extra={"align": nfill, "latency_op": lat})
    return g.plan(p, cfg, "issue(B->%s) must coincide with wb(A->%s)" % (reg, reg),
                  alignment=nfill, target_reg=reg)


def gen_sb_clear_condition(g: GenCtx) -> GenPlan:
    plan = gen_sb_issue_wb(g)
    plan.notes = "scoreboard busy clear condition on %s" % plan.knobs.get("target_reg")
    return plan


# --------------------------------------------------------------------------- #
# 2. DCache flush over a dirty line under memory backpressure (Ventus bug #3)
# --------------------------------------------------------------------------- #


def gen_dcache_flush_dirty(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    off = int(g.pick("line_off", 0, (0, 64, 128, 192)))
    hold = int(g.pick("bp_hold", 24, (8, 16, 24, 32, 48)))
    start = int(g.pick("bp_start", 0, (0, 2, 4, 8)))
    value = int(g.param("value", 0xA5A5A5A5))
    with p.slot("setup.dirty_line", protect=True):
        rv = p.imm(value)
        p.store(rv, p.data_base, off)
    with p.slot("setup.second_line", protect=False):
        p.store(p.imm(value ^ 0x5A5A5A5A), p.data_base, off + 64)
    with p.slot("observe.reload", protect=True):
        p.load("r9", p.data_base, off)
        _obs(p, "r9", 0)
    conds = [
        EventCondition("cache_flush", anchor="when_dirty",
                       params={"line": off // 64, "hold": 1}),
        EventCondition("mem_backpressure", anchor="after_store",
                       params={"start": start, "hold": hold}),
    ]
    cfg = g.config(p, warps=int(g.pick_warps("warps", 1, (1, 2))), blocks=1, conds=g.conds(conds),
                   extra={"flush_line": off // 64})
    return g.plan(p, cfg, "flush must not clear dirty before the writeback is secured",
                  line=off // 64, bp_hold=hold)


# --------------------------------------------------------------------------- #
# 3. shared-memory lanes coexisting with global accesses (Ventus bug #2)
# --------------------------------------------------------------------------- #


def gen_lsu_lane_route(g: GenCtx) -> GenPlan:
    smem = 1024
    p = g.prog(smem=smem)
    p.prologue()
    goff = int(g.pick("goff", 0, (0, 64, 128)))
    order = str(g.pick("order", "shared_first", ("shared_first", "global_first")))
    lane = p.reg()
    p.emit("laneid", dst=lane)
    sh = p.reg()
    p.emit("shl", dst=sh, srcs=(lane, p.imm(2)))
    v = p.reg()
    p.emit("add", dst=v, srcs=(p.tid, p.imm(1)))

    def do_shared() -> None:
        with p.slot("setup.shared_store", protect=True):
            p.store(v, sh, 0, space="shared")

    def do_global() -> None:
        with p.slot("trigger.global_store", protect=True):
            p.store(v, p.data_base, goff)

    if order == "shared_first":
        do_shared()
        do_global()
    else:
        do_global()
        do_shared()
    with p.slot("observe.readback", protect=True):
        p.load("r9", sh, 0, space="shared")
        p.load("r10", p.data_base, goff)
        _obs(p, "r9", 0)
        _obs(p, "r10", 1)
    cfg = g.config(p, warps=int(g.pick_warps("warps", 2, (1, 2, 4))), blocks=1)
    return g.plan(p, cfg, "shared lanes must not be routed through DCache",
                  order=order, goff=goff)


# --------------------------------------------------------------------------- #
# 4. zero length copy desynchronising the controller (VeriGPU bug #6)
# --------------------------------------------------------------------------- #


def gen_dma_zero_len(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    off = int(g.pick("dma_dst", 0x40, (0x40, 0x80, 0xC0)))
    with p.slot("setup.store", protect=True):
        p.store(p.imm(0x1234ABCD), p.data_base, 0)
    conds = [
        EventCondition("dma_copy", anchor="after_retire", params={"len": 0, "dst": off, "seq": 1}),
        EventCondition("dma_copy", anchor="after_retire", params={"len": 16, "src": 0, "dst": off, "seq": 2}),
    ]
    with p.slot("observe.after_copy", protect=True):
        p.load("r9", p.data_base, off)
        _obs(p, "r9", 0)
    cfg = g.config(p, warps=1, blocks=1, conds=g.conds(conds), extra={"dma_seq": 2})
    return g.plan(p, cfg, "a zero length copy must keep host/controller in sync", dma_dst=off)


# --------------------------------------------------------------------------- #
# 5. division whose destination is the zero register (VeriGPU bug #7)
# --------------------------------------------------------------------------- #


def gen_div_r0(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    a = p.imm(int(g.param("dividend", 0x7F3)))
    b = p.imm(int(g.param("divisor", 7)))
    with p.slot("trigger.div_to_zero_reg", protect=True):
        p.emit("div", dst="r0", srcs=(a, b))
    with p.slot("trigger.div_normal", protect=True):
        p.emit("div", dst="r8", srcs=(a, b))
    _obs(p, "r8", 0)
    cfg = g.config(p, warps=int(g.pick_warps("warps", 1, (1, 2))), blocks=1)
    return g.plan(p, cfg, "a division writing r0 must still complete")


# --------------------------------------------------------------------------- #
# 6. stale register file across kernel launches (VeriGPU bug #8)
# --------------------------------------------------------------------------- #


def gen_kernel_init_regs(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    first = p.reg()
    # the driver writes the launch index into the parameter block before each
    # launch; every target can read it, no cfg-register feature required
    p.load(first, p.data_base, LAUNCH_PARAM_OFFSET)
    is_first = p.reg(avoid=(first,))
    p.emit("sltu", dst=is_first, srcs=(first, p.imm(1)))
    keep = g.pick("keep_reg", "r9", ("r9", "r11", "r12"))
    with p.slot("trigger.launch_path", protect=True):
        p.br(is_first, "L_first", sense="nz")
        _obs(p, keep, 0)                       # second launch: must observe a fresh 0
        p.emit("jmp", label="L_end")
        p.label("L_first")
        p.emit("mov_imm", dst=keep, imm=int(g.param("stale_value", 0x1234)))
        _obs(p, keep, 1)
    p.label("L_end")
    p.finish()
    launches = [Launch(grid=(1, 1, 1), block=(g.lanes, 1, 1), arg=0),
                Launch(grid=(1, 1, 1), block=(g.lanes, 1, 1), arg=1)]
    cfg = g.config(p, warps=1, blocks=1, launches=launches,
                   extra={"launch_args": [0, 1]})
    cfg.launches = launches
    return g.plan(p, cfg, "kernel start must clear every architectural register", keep_reg=keep)


# --------------------------------------------------------------------------- #
# 7. truncated global address (VeriGPU bug #9)
# --------------------------------------------------------------------------- #


def gen_mem_addr_trunc(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    hi = int(g.param("hi_off", 0x3000))
    alias = (0x1000 + hi) & 0x3FFF
    base = p.imm(0x1000 + hi)
    val = 0xCAFEBABE
    with p.slot("trigger.high_store", protect=True):
        p.store(p.imm(val), base, 0)
    with p.slot("observe.high_load", protect=True):
        p.load("r9", base, 0)
        _obs(p, "r9", 0)
    init = [(alias, 0xBAD0BAD0)]
    cfg = g.config(p, warps=1, blocks=1, init=init, extra={"hi_off": hi})
    return g.plan(p, cfg, "stores to high addresses must not alias low memory", hi_off=hi)


# --------------------------------------------------------------------------- #
# 8. JALR target without clearing the LSB (VeriGPU bug #10)
# --------------------------------------------------------------------------- #


def gen_jalr_align(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    tgt = p.reg()
    with p.slot("trigger.jalr", protect=True):
        p.emit("la", dst=tgt, label="L_taken")
        p.emit("add", dst=tgt, srcs=(tgt, p.imm(1)))     # deliberately odd target
        p.emit("jmp_reg", srcs=(tgt,))
        p.emit("jmp", label="L_end")
        p.label("L_taken")
        _obs(p, p.imm(0x600D), 0)
    p.label("L_end")
    p.finish()
    cfg = g.config(p, warps=1, blocks=1)
    return g.plan(p, cfg, "JALR must clear the least significant target bit")


# --------------------------------------------------------------------------- #
# 9. floating point special values (VeriGPU bugs #11/#13)
# --------------------------------------------------------------------------- #


def gen_fp_nan_inf(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    op = _pick_op(g, "fp_op", ("fadd", "fmul", "fdiv"), "fadd")
    a = _f32(g, p, F32_NAN)
    b = _f32(g, p, F32_ONE)
    with p.slot("trigger.fp_special", protect=True):
        p.emit(op, dst="r7", srcs=(a, b))
        a2 = _f32(g, p, F32_INF)
        p.emit(op, dst="r8", srcs=(a2, a2))
    _obs(p, "r7", 0)
    _obs(p, "r8", 1)
    cfg = g.config(p, warps=1, blocks=1)
    return g.plan(p, cfg, "NaN/inf inputs must propagate NaN", fp_op=op)


def gen_fp_zero_inf(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    zero = _f32(g, p, 0)
    inf = _f32(g, p, F32_INF)
    with p.slot("trigger.zero_times_inf", protect=True):
        p.emit("fmul", dst="r7", srcs=(zero, inf))
    _obs(p, "r7", 0)
    cfg = g.config(p, warps=1, blocks=1)
    return g.plan(p, cfg, "0 * inf is NaN, not 0")


def gen_fp_overflow_pattern(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    big = _f32(g, p, F32_BIG)
    with p.slot("trigger.overflow", protect=True):
        p.emit("fmul", dst="r7", srcs=(big, big))
        p.emit("fadd", dst="r8", srcs=("r7", big))
    _obs(p, "r7", 0)
    _obs(p, "r8", 1)
    cfg = g.config(p, warps=1, blocks=1)
    return g.plan(p, cfg, "overflow must round, not produce a NaN pattern")


def gen_fp_round_overflow(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    mode = str(g.pick("round", "rtz", ("rtz", "rup", "rdn", "rne")))
    big = _f32(g, p, F32_BIG)
    with p.slot("trigger.conv_overflow", protect=True):
        p.emit("fmul", dst="r7", srcs=(big, big))       # +inf
        p.emit("fcvt_" + mode, dst="r8", srcs=("r7",))  # conversion of inf
    _obs(p, "r8", 0)
    cfg = g.config(p, warps=1, blocks=1, extra={"round": mode})
    return g.plan(p, cfg, "directed rounding on overflow", round=mode)


# --------------------------------------------------------------------------- #
# 10. saturating arithmetic decode entry (Ventus bug #4)
# --------------------------------------------------------------------------- #


def gen_sat_decode(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    op = str(g.pick("sat_op", "sat_add", ("sat_add", "sat_sub")))
    hi = p.imm(0x7FFFFFFF)
    one = p.imm(1)
    preset = p.imm(0x5)
    with p.slot("setup.preset", protect=True):
        p.emit("mov", dst="r7", srcs=(preset,))
    with p.slot("trigger.saturating", protect=True):
        if op == "sat_add":
            p.emit("sat_add", dst="r7", srcs=(hi, one))
            p.emit("sat_sub", dst="r8", srcs=(p.imm(0x80000000), one))
        else:
            p.emit("sat_sub", dst="r7", srcs=(p.imm(0x80000000), one))
            p.emit("sat_add", dst="r8", srcs=(hi, one))
    _obs(p, "r7", 0)
    _obs(p, "r8", 1)
    cfg = g.config(p, warps=1, blocks=1)
    return g.plan(p, cfg, "saturating arithmetic must write its result", sat_op=op)


# --------------------------------------------------------------------------- #
# 11. signed compare/branch (VeriGPU bug #5)
# --------------------------------------------------------------------------- #


def gen_alu_signed(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    neg = p.imm(int(g.param("neg_val", 0xFFFFFFFE)) & 0xFFFFFFFF)
    pos = p.imm(int(g.param("pos_val", 1)))
    with p.slot("trigger.signed_compare", protect=True):
        p.emit("slt", dst="r7", srcs=(neg, pos))     # signed: -2 < 1 -> 1
        p.emit("sltu", dst="r8", srcs=(neg, pos))    # unsigned: 0
        p.emit("slt", dst="r9", srcs=(pos, neg))
    with p.slot("trigger.signed_branch", protect=True):
        p.br("r7", "L_less", sense="nz")
        p.emit("mov_imm", dst="r10", imm=0)
        p.emit("jmp", label="L_end")
        p.label("L_less")
        p.emit("mov_imm", dst="r10", imm=1)
    p.label("L_end")
    _obs(p, "r7", 0)
    _obs(p, "r8", 1)
    _obs(p, "r10", 2)
    p.finish()
    cfg = g.config(p, warps=1, blocks=1)
    return g.plan(p, cfg, "signed compare and branch semantics", neg=hex(0xFFFFFFFE))


# --------------------------------------------------------------------------- #
# 12. multi block dispatch (smol-gpu bug #14)
# --------------------------------------------------------------------------- #


def gen_dispatch_blocks(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    blocks = int(g.pick("blocks", 4, (2, 4, 8, 16)))
    bid = p.reg()
    p.emit("ctaid", dst=bid)
    slot = p.reg()
    p.emit("shl", dst=slot, srcs=(bid, p.imm(3)))
    addr = p.reg()
    p.emit("mov_imm", dst=addr, imm=0x1000 + 0x400)
    p.emit("add", dst=addr, srcs=(addr, slot))
    with p.slot("trigger.block_store", protect=True):
        p.store(bid, addr, 0)
    _obs(p, bid, 0)
    cfg = g.config(p, warps=1, blocks=blocks,
                   conds=g.conds([EventCondition("launch_queue", anchor="kernel_start",
                                                 params={"blocks": blocks})]),
                   extra={"blocks": blocks})
    return g.plan(p, cfg, "every configured block must be dispatched", blocks=blocks)


# --------------------------------------------------------------------------- #
# 13. warp slot exhaustion (smol-gpu bug #15)
# --------------------------------------------------------------------------- #


def gen_warp_slot_alloc(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    warps = int(g.pick_warps("warps", 8, (4, 8, 16)))
    wid = p.reg()
    p.emit("warpid", dst=wid)
    addr = p.reg()
    p.emit("mov_imm", dst=addr, imm=0x1000 + 0x400)
    slot = p.reg()
    p.emit("shl", dst=slot, srcs=(wid, p.imm(2)))
    p.emit("add", dst=addr, srcs=(addr, slot))
    with p.slot("trigger.warp_store", protect=True):
        p.store(wid, addr, 0)
    _obs(p, wid, 0)
    caps = g.target.caps
    cfg = g.config(p, warps=warps, blocks=2,
                   conds=g.conds([EventCondition("warp_oversubscribe", anchor="kernel_start",
                                                 params={"warps": warps})]),
                   extra={"warps": warps, "slots": caps.get("warp_slots", 4)})
    return g.plan(p, cfg, "warp allocation must not reuse an occupied slot", warps=warps)


# --------------------------------------------------------------------------- #
# 14. queued launch configuration (Vortex bug #17)
# --------------------------------------------------------------------------- #


def gen_launch_queue_cfg(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    arg = p.reg()
    p.emit("getcfg", dst=arg, aux={"name": "queue_arg"})
    with p.slot("observe.queue_arg", protect=True):
        _obs(p, arg, 0)
    launches = [Launch(grid=(1, 1, 1), block=(g.lanes, 1, 1), arg=1),
                Launch(grid=(1, 1, 1), block=(g.lanes, 1, 1), arg=2)]
    cfg = g.config(p, warps=1, blocks=1, launches=launches,
                   conds=g.conds([EventCondition("launch_queue", anchor="kernel_start",
                                                 params={"depth": 2})]),
                   extra={"launch_args": [1, 2]})
    cfg.launches = launches
    return g.plan(p, cfg, "queued launches keep their captured configuration")


# --------------------------------------------------------------------------- #
# 15. texture mipmap offsets (Vortex bug #18)
# --------------------------------------------------------------------------- #


def gen_tex_mipmap(g: GenCtx) -> GenPlan:
    smem = 0
    p = g.prog(smem=smem)
    p.prologue()
    coord = p.imm(int(g.param("coord", 16)))
    level = int(g.pick("level", 1, (0, 1, 2)))
    with p.slot("trigger.tex_first", protect=True):
        p.emit("tex", dst="r7", srcs=(coord,), aux={"level": level})
        _obs(p, "r7", 0)
    conds = [EventCondition("tex_reconfig", anchor="after_retire",
                            params={"level": level + 1, "offset": 0x40})]
    with p.slot("trigger.tex_second", protect=True):
        p.emit("tex", dst="r8", srcs=(coord,), aux={"level": level})
        _obs(p, "r8", 1)
    cfg = g.config(p, warps=1, blocks=1, conds=g.conds(conds),
                   extra={"tex_level": level, "tex_coord": int(g.param("coord", 16))})
    return g.plan(p, cfg, "texture reconfiguration must refresh active offsets", level=level)


# --------------------------------------------------------------------------- #
# 16. delayed LSU response after rescheduling (MIAOW bug #19)
# --------------------------------------------------------------------------- #


def gen_wave_response_assoc(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    lat = int(g.pick("mem_latency", 32, (16, 24, 32, 48)))
    off1 = int(g.pick("off1", 0, (0, 64)))
    off2 = off1 + 64
    with p.slot("trigger.first_load", protect=True):
        p.load("r7", p.data_base, off1)
    with p.slot("trigger.second_load", protect=True):
        p.load("r8", p.data_base, off2)
    _obs(p, "r7", 0)
    _obs(p, "r8", 1)
    conds = [EventCondition("mem_backpressure", anchor="after_issue",
                            params={"start": 0, "hold": lat, "latency": lat})]
    init = [(0x1000 + off1, 0x11111111), (0x1000 + off2, 0x22222222)]
    cfg = g.config(p, warps=int(g.pick_warps("warps", 4, (2, 4, 8))), blocks=1, conds=g.conds(conds),
                   init=init, extra={"mem_latency": lat})
    return g.plan(p, cfg, "a delayed response belongs to its own wavefront", latency=lat)


# --------------------------------------------------------------------------- #
# 17. writeback arbitration (MIAOW bug #20)
# --------------------------------------------------------------------------- #


def gen_wb_arb(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    nfill = int(g.param("filler", g.rng.choice((0, 1, 2))))
    a = _f32(g, p, F32_ONE)
    b = _f32(g, p, F32_HALF)
    with p.slot("trigger.fpu_wb", protect=True):
        p.emit("fmul", dst="r7", srcs=(a, b))
    for i in range(nfill):
        r = p.reg(avoid=("r7", "r8"))
        p.emit("add", dst=r, srcs=(p.imm(3), p.imm(i)))
    with p.slot("trigger.lsu_wb", protect=True):
        p.load("r8", p.data_base, 0)
    _obs(p, "r7", 0)
    _obs(p, "r8", 1)
    init = [(0x1000, 0x5A5A5A5A)]
    cfg = g.config(p, warps=1, blocks=1, init=init, extra={"align": nfill})
    return g.plan(p, cfg, "simultaneous writebacks must both land", align=nfill)


# --------------------------------------------------------------------------- #
# 18. invalidation overlapping an outstanding fill (Nyuzi bug #21)
# --------------------------------------------------------------------------- #


def gen_l1_invalidate_fill(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    off = int(g.pick("line_off", 0, (0, 64, 128)))
    delay = int(g.pick("inval_after", 1, (1, 2, 3)))
    with p.slot("trigger.miss", protect=True):
        p.load("r7", p.data_base, off)
        _obs(p, "r7", 0)
    conds = [EventCondition("cache_invalidate", anchor="after_issue",
                            params={"line": off // 64, "after": delay})]
    with p.slot("observe.reload", protect=True):
        p.load("r8", p.data_base, off)
        _obs(p, "r8", 1)
    init = [(0x1000 + off, 0x0BADF00D)]
    cfg = g.config(p, warps=1, blocks=1, conds=g.conds(conds), init=init,
                   extra={"line": off // 64})
    return g.plan(p, cfg, "an invalidated line must not be re-installed by an older fill",
                  line=off // 64)


# --------------------------------------------------------------------------- #
# 19. rollback / squash overlapping a writeback (Nyuzi bug #22)
# --------------------------------------------------------------------------- #


def gen_rollback_wb(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    parity = p.reg()
    p.emit("and", dst=parity, srcs=(p.tid, p.imm(1)))
    op = _pick_op(g, "latency_op", ("mul", "div"), "mul")
    a = p.imm(3)
    b = p.imm(5)
    with p.slot("trigger.divergent_paths", protect=True):
        p.br(parity, "L_odd", sense="nz")
        p.emit(op, dst="r7", srcs=(a, b))
        p.emit("jmp", label="L_end")
        p.label("L_odd")
        p.emit("add", dst="r7", srcs=(a, b))
    p.label("L_end")
    _obs(p, "r7", 0)
    conds = [EventCondition("rollback_stress", anchor="kernel_start",
                            params={"mode": "divergence", "warps": 2})]
    cfg = g.config(p, warps=int(g.pick_warps("warps", 2, (2, 4))), blocks=1, conds=g.conds(conds),
                   extra={"divergence": True})
    return g.plan(p, cfg, "a squashed operation must not write back", latency_op=op)


# --------------------------------------------------------------------------- #
# 20. barrier release order (generic, all targets)
# --------------------------------------------------------------------------- #


def gen_barrier_order(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    warps = int(g.pick_warps("warps", 4, (2, 4, 8)))
    wid = p.reg()
    p.emit("warpid", dst=wid)
    addr = p.reg()
    p.emit("mov_imm", dst=addr, imm=0x1000 + 0x400)
    slot = p.reg()
    p.emit("shl", dst=slot, srcs=(wid, p.imm(2)))
    p.emit("add", dst=addr, srcs=(addr, slot))
    with p.slot("trigger.barrier_before", protect=True):
        p.barrier()
        p.store(wid, addr, 0)
    with p.slot("trigger.barrier_after", protect=True):
        p.barrier()
        p.load("r9", addr, 0)
        _obs(p, "r9", 0)
    cfg = g.config(p, warps=warps, blocks=1, extra={"warps": warps})
    return g.plan(p, cfg, "a barrier releases only after every arrival", warps=warps)


# --------------------------------------------------------------------------- #
# 21. transaction progress under released backpressure (generic)
# --------------------------------------------------------------------------- #


def gen_txn_progress(g: GenCtx) -> GenPlan:
    p = g.prog()
    p.prologue()
    hold = int(g.pick("bp_hold", 32, (16, 32, 64)))
    nops = int(g.pick("nops", 2, (1, 2, 4)))
    for i in range(nops):
        with p.slot("trigger.mem_stress", protect=True):
            p.store(p.imm(0x100 + i), p.data_base, i * 64)
            p.load("r%d" % (8 + i), p.data_base, i * 64)
            _obs(p, "r%d" % (8 + i), i)
    conds = [EventCondition("mem_backpressure", anchor="kernel_start",
                            params={"start": 0, "hold": hold})]
    cfg = g.config(p, warps=int(g.pick_warps("warps", 1, (1, 2))), blocks=1, conds=g.conds(conds),
                   progress_timeout=int(g.param("progress_timeout", 40000)))
    return g.plan(p, cfg, "every outstanding transaction completes", bp_hold=hold)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

GENERATORS: Dict[str, Callable[[GenCtx], GenPlan]] = {
    "sb_issue_wb": gen_sb_issue_wb,
    "sb_clear_condition": gen_sb_clear_condition,
    "dcache_flush_dirty": gen_dcache_flush_dirty,
    "lsu_lane_route": gen_lsu_lane_route,
    "dma_zero_len": gen_dma_zero_len,
    "div_r0": gen_div_r0,
    "kernel_init_regs": gen_kernel_init_regs,
    "mem_addr_trunc": gen_mem_addr_trunc,
    "jalr_align": gen_jalr_align,
    "fp_nan_inf": gen_fp_nan_inf,
    "fp_zero_inf": gen_fp_zero_inf,
    "fp_overflow_pattern": gen_fp_overflow_pattern,
    "fp_round_overflow": gen_fp_round_overflow,
    "sat_decode": gen_sat_decode,
    "alu_signed": gen_alu_signed,
    "dispatch_blocks": gen_dispatch_blocks,
    "warp_slot_alloc": gen_warp_slot_alloc,
    "launch_queue_cfg": gen_launch_queue_cfg,
    "tex_mipmap": gen_tex_mipmap,
    "wave_response_assoc": gen_wave_response_assoc,
    "wb_arb": gen_wb_arb,
    "l1_invalidate_fill": gen_l1_invalidate_fill,
    "rollback_wb": gen_rollback_wb,
    "barrier_order": gen_barrier_order,
    "txn_progress": gen_txn_progress,
}


def generate(gen_name: str, g: GenCtx) -> GenPlan:
    fn = GENERATORS.get(gen_name)
    if fn is None:
        raise KeyError("no generator named %r" % (gen_name,))
    plan = fn(g)
    plan.knobs.setdefault("shape", g.shape.shape)
    return plan
