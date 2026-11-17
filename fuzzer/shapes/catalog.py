"""The shape catalog: 25 execution state dependencies over the six GPGPUs.

Each entry is the output of the paper's *Step 1: GPU semantic analysis* for one
kind of state interaction.  ``targets`` restricts a shape to designs that
implement the relevant unit; ``bug_ids`` links the shape to the paper's Table 2
so a campaign can report which relation detected which bug.

Mapping to Table 2 (all 22 bugs are covered):

    #1  SB_ISSUE_WB            #2  LSU_LANE_ROUTE         #3  DCACHE_FLUSH_DIRTY
    #4  SAT_DECODE            #5  ALU_SIGNED             #6  DMA_ZERO_LEN
    #7  DIV_R0                #8  KERNEL_INIT_REGS       #9  MEM_ADDR_TRUNC
    #10 JALR_ALIGN            #11 FP_NAN_INF              #12 FP_OVERFLOW_PATTERN
    #13 FP_ZERO_INF           #14 DISPATCH_BLOCKS        #15 WARP_SLOT_ALLOC
    #16 FP_ROUND_OVERFLOW     #17 LAUNCH_QUEUE_CFG       #18 TEX_MIPMAP
    #19 WAVE_RESPONSE_ASSOC   #20 WB_ARB                 #21 L1_INVALIDATE_FILL
    #22 ROLLBACK_WB
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .spec import RoleSpec, ShapeSpec
from .templates import GENERATORS

ALL = ("*",)
NON_VERIGPU = ("vortex", "ventus", "miaow", "nyuzi", "smol-gpu")
WITH_SCOREBOARD = ("vortex", "ventus", "miaow", "nyuzi", "smol-gpu")
WITH_DCACHE = ("vortex", "ventus", "nyuzi")


def _specs() -> List[ShapeSpec]:
    S: List[ShapeSpec] = []

    # ---------------- scoreboard -------------------------------------- #
    S.append(ShapeSpec(
        shape="SB_ISSUE_WB", kind="interaction", omega="OM_SCOREBOARD_NEW_WRITER",
        obj_kind="reg", sids=("scoreboard.busy", "rf.write"),
        roles=(RoleSpec("old_wb", ("wb",)), RoleSpec("new_issue", ("issue",))),
        atoms=(("overlap", "old_wb", "new_issue", 2),),
        targets=WITH_SCOREBOARD, gen="sb_issue_wb",
        spec_ref="Section 2, Section 4.2 (issue/writeback overlap); Fig. 9",
        knobs=("filler", "target_reg", "latency_op", "warps", "consumers"),
        desc="An older instruction writes back to R while a new instruction that "
             "targets R issues in the same window; the new dependency must survive.",
        bug_ids={"ventus": ("#1",)}))

    S.append(ShapeSpec(
        shape="SB_CLEAR_CONDITION", kind="interaction", omega="OM_SB_CLEAR_CONDITION",
        obj_kind="reg", sids=("scoreboard.busy",),
        roles=(RoleSpec("issue", ("issue",)), RoleSpec("wb", ("wb",), optional=True)),
        atoms=(("overlap", "issue", "wb", 1),),
        targets=WITH_SCOREBOARD, gen="sb_clear_condition",
        spec_ref="Section 4.2 (state transition rules)",
        knobs=("filler", "target_reg", "warps"),
        desc="A busy bit may only be cleared by the writeback of the pending writer."))

    # ---------------- caches / memory --------------------------------- #
    S.append(ShapeSpec(
        shape="DCACHE_FLUSH_DIRTY", kind="interaction", omega="OM_DIRTY_SECURED",
        obj_kind="line", sids=("dcache.dirty", "wb.pending"),
        roles=(RoleSpec("store", ("store",)), RoleSpec("flush", ("flush",)),
               RoleSpec("wb_pending", ("wb_pending_set",), optional=True)),
        atoms=(("before", "store", "flush"), ("overlap", "flush", "wb_pending", 16)),
        targets=WITH_DCACHE, gen="dcache_flush_dirty",
        spec_ref="Section 3, Fig. 3, Fig. 4", knobs=("line_off", "bp_hold", "bp_start", "warps"),
        desc="Clearing the dirty bit of a line under memory backpressure requires "
             "the writeback to be accepted or safely buffered first.",
        bug_ids={"ventus": ("#3",)}))

    S.append(ShapeSpec(
        shape="LSU_LANE_ROUTE", kind="interaction", omega="OM_LANE_ROUTE",
        obj_kind="warp", sids=("lane.route", "dcache.valid"),
        roles=(RoleSpec("shared", ("lane_route",), outcomes=("shared",)),
               RoleSpec("global", ("lane_route",), outcomes=("global",))),
        atoms=(("coincide", "shared", "global", 4),),
        targets=("ventus", "vortex"), gen="lsu_lane_route",
        spec_ref="Section 4.2 (per-lane routing); Table 2 #2",
        knobs=("warps", "order", "goff"),
        desc="Shared-memory lanes must be routed by the LSU even when global "
             "accesses coexist in the same warp.",
        bug_ids={"ventus": ("#2",)}))

    S.append(ShapeSpec(
        shape="L1_INVALIDATE_FILL", kind="interaction", omega="OM_L1_INVALIDATE_ORDER",
        obj_kind="line", sids=("l1.fill", "dcache.valid"),
        roles=(RoleSpec("fill", ("fill_start",)), RoleSpec("invalidate", ("invalidate",)),
               RoleSpec("install", ("fill_install",), optional=True)),
        atoms=(("before", "fill", "invalidate"), ("before", "invalidate", "install")),
        targets=("nyuzi",), gen="l1_invalidate_fill",
        spec_ref="Table 2 #21", knobs=("line_off", "inval_after"),
        desc="An invalidation overlapping an outstanding fill must not be undone "
             "by the late installation of that fill.",
        bug_ids={"nyuzi": ("#21",)}))

    S.append(ShapeSpec(
        shape="WAVE_RESPONSE_ASSOC", kind="interaction", omega="OM_WAVE_ASSOC",
        obj_kind="txn", sids=("txn.state", "rf.write"),
        roles=(RoleSpec("req", ("mem_req",)), RoleSpec("rsp", ("mem_rsp",), optional=True)),
        atoms=(("before", "req", "rsp"), ("within", "req", "rsp", 512)),
        targets=("miaow",), gen="wave_response_assoc",
        spec_ref="Table 2 #19", knobs=("mem_latency", "warps", "off1"),
        desc="A delayed LSU response must be associated with the wavefront that "
             "issued the request, even after rescheduling.",
        bug_ids={"miaow": ("#19",)}))

    S.append(ShapeSpec(
        shape="WB_ARB", kind="interaction", omega="OM_WB_ARB_NO_LOSS",
        obj_kind="unit", sids=("rf.arb", "rf.write"),
        roles=(RoleSpec("first", ("wb_arb",)), RoleSpec("second", ("wb_arb",), optional=True)),
        atoms=(("same_cycle", "first", "second"),),
        targets=("miaow",), gen="wb_arb",
        spec_ref="Table 2 #20", knobs=("filler",),
        desc="Simultaneous writebacks from distinct execution units must both "
             "reach the register file.",
        bug_ids={"miaow": ("#20",)}))

    S.append(ShapeSpec(
        shape="DMA_ZERO_LEN", kind="interaction", omega="OM_MEMCPY_LEN",
        obj_kind="unit", sids=("txn.state",),
        roles=(RoleSpec("zero_copy", ("dma_copy",), outcomes=("zero_length",)),
               RoleSpec("next_cmd", ("dma_copy",), optional=True)),
        atoms=(("before", "zero_copy", "next_cmd"),),
        targets=("verigpu",), gen="dma_zero_len",
        spec_ref="Table 2 #6", knobs=("dma_dst",),
        desc="A zero-length copy must not desynchronise the host/controller "
             "command stream.",
        bug_ids={"verigpu": ("#6",)}))

    # ---------------- pipelines / configuration ----------------------- #
    S.append(ShapeSpec(
        shape="ROLLBACK_WB", kind="interaction", omega="OM_SQUASH_NO_WB",
        obj_kind="reg", sids=("pipe.valid", "rf.write"),
        roles=(RoleSpec("squash", ("squash",)), RoleSpec("wb", ("wb",), optional=True)),
        atoms=(("overlap", "squash", "wb", 2),),
        targets=("nyuzi",), gen="rollback_wb",
        spec_ref="Table 2 #22", knobs=("warps", "latency_op"),
        desc="A squashed operation must not update the register file.",
        bug_ids={"nyuzi": ("#22",)}))

    S.append(ShapeSpec(
        shape="DIV_R0", kind="interaction", omega="OM_DIV_COMPLETION",
        obj_kind="reg", sids=("div.state", "scoreboard.busy"),
        roles=(RoleSpec("start", ("div_start",)),
               RoleSpec("end", ("kernel_end",), same_obj=False, optional=True)),
        atoms=(("before", "start", "end"),),
        targets=("verigpu", "vortex"), gen="div_r0",
        spec_ref="Table 2 #7", knobs=("warps",),
        desc="A division whose destination is the zero register must still "
             "produce a completion.",
        bug_ids={"verigpu": ("#7",)}))

    S.append(ShapeSpec(
        shape="KERNEL_INIT_REGS", kind="interaction", omega="OM_KERNEL_STATE_FRESH",
        obj_kind="reg", sids=("rf.write",),
        roles=(RoleSpec("launch", ("kernel_start",), same_obj=False),
               RoleSpec("use", ("issue",), optional=True)),
        atoms=(("before", "launch", "use"),),
        targets=("verigpu", "ventus", "smol-gpu", "vortex"), gen="kernel_init_regs",
        spec_ref="Table 2 #8", knobs=("keep_reg", "stale_value"),
        desc="Kernel initialisation must clear every architectural register, not "
             "only r0.",
        bug_ids={"verigpu": ("#8",)}))

    S.append(ShapeSpec(
        shape="LAUNCH_QUEUE_CFG", kind="interaction", omega="OM_QUEUE_CFG_STABLE",
        obj_kind="cfg", sids=("cfg.reg",),
        roles=(RoleSpec("push", ("queue_push",)), RoleSpec("write", ("cfg_write",), optional=True),
               RoleSpec("pop", ("queue_pop",), optional=True)),
        atoms=(("before", "push", "write"), ("before", "write", "pop")),
        targets=("vortex",), gen="launch_queue_cfg",
        spec_ref="Table 2 #17", knobs=("warps",),
        desc="A queued launch keeps the configuration captured at enqueue time.",
        bug_ids={"vortex": ("#17",)}))

    S.append(ShapeSpec(
        shape="DISPATCH_BLOCKS", kind="interaction", omega="OM_DISPATCH_BLOCK_COUNT",
        obj_kind="kernel", sids=("dispatch.count",),
        roles=(RoleSpec("launch", ("kernel_start",)),
               RoleSpec("dispatch", ("dispatch",), same_obj=False, optional=True)),
        atoms=(("before", "launch", "dispatch"),),
        targets=("smol-gpu",), gen="dispatch_blocks",
        spec_ref="Table 2 #14", knobs=("blocks",),
        desc="The dispatcher must use the block count that arrives with the "
             "launch, not a value captured earlier.",
        bug_ids={"smol-gpu": ("#14",)}))

    S.append(ShapeSpec(
        shape="WARP_SLOT_ALLOC", kind="interaction", omega="OM_WARP_SLOT_CAPACITY",
        obj_kind="slot", sids=("warp.slot",),
        roles=(RoleSpec("first", ("alloc",)), RoleSpec("second", ("alloc",))),
        atoms=(("overlap", "first", "second", 4),),
        targets=("smol-gpu",), gen="warp_slot_alloc",
        spec_ref="Table 2 #15", knobs=("warps",),
        desc="Warp allocation must never reuse an occupied slot when the slot "
             "table is exhausted.",
        bug_ids={"smol-gpu": ("#15",)}))

    S.append(ShapeSpec(
        shape="TEX_MIPMAP", kind="interaction", omega="OM_TEX_OFFSET_FRESH",
        obj_kind="tex", sids=("tex.mipmap",),
        roles=(RoleSpec("config", ("tex_config",)), RoleSpec("access", ("tex_access",), optional=True)),
        atoms=(("before", "config", "access"),),
        targets=("vortex",), gen="tex_mipmap",
        spec_ref="Table 2 #18", knobs=("level", "coord"),
        desc="A texture reconfiguration must replace the active mipmap offsets.",
        bug_ids={"vortex": ("#18",)}))

    # ---------------- generic obligations over all designs ------------- #
    S.append(ShapeSpec(
        shape="TX_PROGRESS", kind="interaction", omega="OM_PROGRESS",
        obj_kind="txn", sids=("txn.state",),
        roles=(RoleSpec("req", ("mem_req",)),
               RoleSpec("end", ("kernel_end",), same_obj=False, optional=True)),
        atoms=(("before", "req", "end"),),
        targets=ALL, gen="txn_progress",
        spec_ref="Section 4.4 (forward progress rules)", knobs=("bp_hold", "nops"),
        desc="Every outstanding transaction must reach a completion state after "
             "controllable backpressure is released."))

    S.append(ShapeSpec(
        shape="BARRIER_ORDER", kind="interaction", omega="OM_BARRIER_ORDER",
        obj_kind="barrier", sids=("barrier.count",),
        roles=(RoleSpec("arrive", ("barrier_arrive",)),
               RoleSpec("release", ("barrier_release",), optional=True)),
        atoms=(("before", "arrive", "release"),),
        targets=ALL, gen="barrier_order",
        spec_ref="Section 2, Section 4.4", knobs=("warps",),
        desc="A barrier releases a warp only after every arrival of the epoch."))

    # ---------------- functional shapes -------------------------------- #
    S.append(ShapeSpec(
        shape="SAT_DECODE", kind="functional", omega="OM_SAT_DECODE",
        obj_kind="reg", sids=("dec.table",),
        roles=(RoleSpec("decode", ("decode",)), RoleSpec("commit", ("commit",))),
        atoms=(("before", "decode", "commit"),),
        targets=("ventus",), gen="sat_decode",
        spec_ref="Table 2 #4", knobs=("sat_op",),
        desc="Saturating arithmetic instructions must write their result.",
        bug_ids={"ventus": ("#4",)}))

    S.append(ShapeSpec(
        shape="ALU_SIGNED", kind="functional", omega="OM_SIGNED_SEMANTICS",
        obj_kind="reg", sids=("alu.flag",),
        roles=(RoleSpec("compare", ("issue",)), RoleSpec("commit", ("commit",))),
        atoms=(("before", "compare", "commit"),),
        targets=("verigpu", "smol-gpu"), gen="alu_signed",
        spec_ref="Table 2 #5", knobs=("neg_val", "pos_val"),
        desc="Signed comparisons and branches use signed arithmetic.",
        bug_ids={"verigpu": ("#5",)}))

    S.append(ShapeSpec(
        shape="MEM_ADDR_TRUNC", kind="functional", omega="OM_ADDR_NO_TRUNC",
        obj_kind="mem", sids=("mem.addr",),
        roles=(RoleSpec("store", ("store",)), RoleSpec("load", ("load",))),
        atoms=(("before", "store", "load"),),
        targets=("verigpu", "vortex"), gen="mem_addr_trunc",
        spec_ref="Table 2 #9", knobs=("hi_off",),
        desc="Global memory addresses are not truncated in the store datapath.",
        bug_ids={"verigpu": ("#9",)}))

    S.append(ShapeSpec(
        shape="JALR_ALIGN", kind="functional", omega="OM_JALR_ALIGN",
        obj_kind="pc", sids=("pc.value",),
        roles=(RoleSpec("jump", ("issue",)), RoleSpec("target", ("commit",))),
        atoms=(("before", "jump", "target"),),
        targets=("verigpu", "vortex", "smol-gpu"), gen="jalr_align",
        spec_ref="Table 2 #10", knobs=(),
        desc="JALR clears the least significant bit of the computed target.",
        bug_ids={"verigpu": ("#10",)}))

    S.append(ShapeSpec(
        shape="FP_NAN_INF", kind="functional", omega="OM_FP_SPECIAL",
        obj_kind="reg", sids=("fpu.result",),
        roles=(RoleSpec("op", ("issue",)), RoleSpec("commit", ("commit",))),
        atoms=(("before", "op", "commit"),),
        targets=("verigpu", "miaow", "nyuzi", "vortex"), gen="fp_nan_inf",
        spec_ref="Table 2 #11", knobs=("fp_op",),
        desc="NaN and opposing-infinity inputs propagate as IEEE-754 requires.",
        bug_ids={"verigpu": ("#11",)}))

    S.append(ShapeSpec(
        shape="FP_ZERO_INF", kind="functional", omega="OM_FP_SPECIAL",
        obj_kind="reg", sids=("fpu.result",),
        roles=(RoleSpec("op", ("issue",)), RoleSpec("commit", ("commit",))),
        atoms=(("before", "op", "commit"),),
        targets=("verigpu", "miaow", "nyuzi", "vortex"), gen="fp_zero_inf",
        spec_ref="Table 2 #13", knobs=(),
        desc="0 x infinity is NaN, not zero.",
        bug_ids={"verigpu": ("#13",)}))

    S.append(ShapeSpec(
        shape="FP_OVERFLOW_PATTERN", kind="functional", omega="OM_FP_SPECIAL",
        obj_kind="reg", sids=("fpu.result",),
        roles=(RoleSpec("op", ("issue",)), RoleSpec("commit", ("commit",))),
        atoms=(("before", "op", "commit"),),
        targets=("verigpu", "miaow"), gen="fp_overflow_pattern",
        spec_ref="Table 2 #12", knobs=(),
        desc="Overflow produces the rounded overflow result, not a NaN pattern.",
        bug_ids={"verigpu": ("#12",)}))

    S.append(ShapeSpec(
        shape="FP_ROUND_OVERFLOW", kind="functional", omega="OM_FP_ROUND_OVERFLOW",
        obj_kind="reg", sids=("fpu.result",),
        roles=(RoleSpec("op", ("issue",)), RoleSpec("commit", ("commit",))),
        atoms=(("before", "op", "commit"),),
        targets=("vortex", "verigpu"), gen="fp_round_overflow",
        spec_ref="Table 2 #16", knobs=("round",),
        desc="Conversions honour the directed rounding mode on overflow.",
        bug_ids={"vortex": ("#16",)}))
    return S


SHAPES: Dict[str, ShapeSpec] = {s.shape: s for s in _specs()}

# sanity: every shape names an existing generator and a known obligation
def validate_catalog() -> List[str]:
    from common.types import OMEGA
    problems = []
    for name, s in SHAPES.items():
        if s.gen not in GENERATORS:
            problems.append("%s: unknown generator %r" % (name, s.gen))
        if s.omega not in OMEGA and s.omega not in ("⊥", "BOT"):
            problems.append("%s: unknown obligation %r" % (name, s.omega))
        if s.kind not in ("interaction", "functional"):
            problems.append("%s: bad kind %r" % (name, s.kind))
        if not s.roles:
            problems.append("%s: no roles" % name)
    return problems


def for_target(target: str) -> List[ShapeSpec]:
    return [s for s in SHAPES.values() if s.applies_to(target)]


def bug_coverage() -> Dict[str, List[str]]:
    """Which shapes claim which bug id, per target."""
    out: Dict[str, List[str]] = {}
    for s in SHAPES.values():
        for target, ids in s.bug_ids.items():
            for b in ids:
                out.setdefault(target, []).append("%s:%s" % (b, s.shape))
    return out
