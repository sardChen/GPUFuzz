"""Trace level emulation of the 22 bugs from Table 2 plus a few generic faults.

Each function takes a :class:`SynthCtx` produced by the clean synthesizer and
edits the event/state/architectural records so that they carry the *signature* of
one real bug: the obligation is violated, and (for the functional bugs) the
architectural dump shows the wrong value.

These are stimuli for the oracle's regression tests, not hardware models.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from common.types import Event, StateObs, obj_ref
from tests.fixtures.synth import SynthCtx

F32_INF = 0x7F800000
F32_NAN = 0x7FC00000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _obs_addrs(ctx: SynthCtx) -> List[int]:
    return ctx.obs_addrs()


def _set_obs(ctx: SynthCtx, index: int, value: int) -> bool:
    addrs = _obs_addrs(ctx)
    if not addrs:
        return False
    idx = index if index >= 0 else len(addrs) + index
    idx = max(0, min(idx, len(addrs) - 1))
    ctx.arch.mem[addrs[idx]] = value & 0xFFFFFFFF
    return True


def _perturb_all_obs(ctx: SynthCtx, value: int = 0) -> bool:
    addrs = _obs_addrs(ctx)
    for a in addrs:
        ctx.arch.mem[a] = value & 0xFFFFFFFF
    return bool(addrs)


def _states_at(ctx: SynthCtx, sid: str, cycle: int) -> List[StateObs]:
    return [s for s in ctx.states if s.sid == sid and s.cycle == cycle]


def _first(ctx: SynthCtx, etype: str) -> Optional[Event]:
    evs = [e for e in ctx.events if e.a == etype]
    return evs[0] if evs else None


def _drop_events(ctx: SynthCtx, pred: Callable[[Event], bool]) -> int:
    before = len(ctx.events)
    ctx.events = [e for e in ctx.events if not pred(e)]
    return before - len(ctx.events)


def _knob(ctx: SynthCtx, name: str, default: Any = None) -> Any:
    return (ctx.tc.provenance.get("knobs") or {}).get(name, default)


# --------------------------------------------------------------------------- #
# interaction faults
# --------------------------------------------------------------------------- #


def scoreboard_early_clear(ctx: SynthCtx) -> None:
    """#1: an older writeback clears the busy bit of a newly issued writer.

    The two events have to overlap: the older writer's writeback is moved onto the
    cycle in which the new writer issues, and the clear is applied *after* the
    set, so the end-of-cycle value is "not busy" while the new writer is pending.
    The architectural result stays correct -- this is a silent bug.
    """
    by_reg: Dict[str, List[Event]] = {}
    for e in ctx.events:
        if e.a == "issue":
            by_reg.setdefault(e.u, []).append(e)
    for reg, issues in sorted(by_reg.items()):
        if len(issues) < 2:
            continue
        issues.sort(key=lambda e: e.cycle)
        first, second = issues[0], issues[1]
        old_wb = next((e for e in ctx.events
                       if e.a == "wb" and e.u == reg and e.cycle > first.cycle), None)
        if old_wb is None:
            continue
        old_wb.cycle = second.cycle          # the writeback now coincides with the issue
        ctx.st("scoreboard.busy", reg, second.cycle, 0, "scoreboard.busy")
        return


def scoreboard_clear_no_wb(ctx: SynthCtx) -> None:
    """A busy bit cleared with no writeback of that register at all."""
    serial = {}
    for s in sorted(ctx.states, key=lambda x: x.cycle):
        if s.sid != "scoreboard.busy":
            continue
        prev = serial.get(s.obj)
        if prev == 1 and s.val == 0:
            _drop_events(ctx, lambda e, r=s.obj, c=s.cycle:
                         e.a == "wb" and e.u == r and abs(e.cycle - c) <= 1)
            return
        serial[s.obj] = s.val


def dirty_clear_early(ctx: SynthCtx) -> None:
    """#3: dirty cleared before the writeback is accepted; data is lost."""
    flush = _first(ctx, "flush")
    if flush is None:
        return
    line = flush.u
    # the clear happens at flush time, not at acceptance
    for s in ctx.states:
        if s.sid == "dcache.dirty" and s.obj == line and s.val == 0:
            s.cycle = flush.cycle
            break
    _drop_events(ctx, lambda e: e.a in ("mem_accept",) and e.u == line)
    for s in ctx.states:
        if s.sid == "wb.pending" and s.obj == line and s.val == 1:
            s.val = 0
    ctx.ev("mem_drop", line, flush.cycle + 2, "lost", phase="stalled")
    # the reload now observes the stale memory value
    _set_obs(ctx, 0, 0)


def lane_misroute(ctx: SynthCtx) -> None:
    """#2: shared lanes are routed through the DCache."""
    shared = [e for e in ctx.events if e.a == "lane_route" and e.o == "shared"]
    for e in shared[:2]:
        ctx.ev("lane_route", e.u, e.cycle + 1, "misdirected",
               raw={"space": "shared", "route": "dcache"})
    for s in ctx.states:
        if s.sid == "lane.route":
            s.val = {"space": "shared", "route": "dcache"}
            break
    ctx.st("lane.route", obj_ref("warp", 0), 30, {"space": "shared", "route": "dcache"},
           "lsu.lane_route")
    _set_obs(ctx, 0, 0)


def memcpy_desync(ctx: SynthCtx) -> None:
    """#6: a zero length copy desynchronises the command stream."""
    cmds = [e for e in ctx.events if e.a == "dma_copy"]
    if not cmds:
        return
    cmds[0].o = "zero_length"
    if len(cmds) > 1:
        cmds[1].o = "misapplied"
    _set_obs(ctx, -1, 0)


def kernel_init_stale(ctx: SynthCtx) -> None:
    """#8: a register keeps the value of the previous launch."""
    reg = _knob(ctx, "keep_reg", "r9")
    stale = int(_knob(ctx, "stale_value", 0x1234))
    # the initialisation write for this register is missing
    _drop_events(ctx, lambda e, r=reg: e.a == "reg_init" and e.u == obj_ref("reg", r))
    # architectural dump must show the stale value for the second launch
    for key in list(ctx.arch.regs):
        if key.endswith("." + reg):
            ctx.arch.regs[key] = stale
    _set_obs(ctx, 0, stale)


def warp_slot_collision(ctx: SynthCtx) -> None:
    """#15: an occupied warp slot is handed to another warp."""
    cycle = 30
    # the same slot is handed out twice: warp 3 reuses the slot owned by warp 1
    ctx.st("warp.slot", obj_ref("slot", 0), cycle, {"owner": 1}, "sched.slot_owner")
    ctx.st("warp.slot", obj_ref("slot", 0), cycle + 1, {"owner": 1}, "sched.slot_owner")
    for w in range(min(ctx.warps, 4)):
        ctx.ev("alloc", obj_ref("slot", 0), cycle + w, "ok", raw={"warp": w})
    _set_obs(ctx, 0, 0)


def dispatch_zero(ctx: SynthCtx) -> None:
    """#14: no blocks are dispatched; the kernel completes without work."""
    _drop_events(ctx, lambda e: e.a == "dispatch")
    ctx.st("dispatch.count", "kernel:0", 8, 0, "dispatch.block_count")
    _perturb_all_obs(ctx, 0)


def queue_cfg_overwrite(ctx: SynthCtx) -> None:
    """#17: a queued launch sees the configuration of the later launch."""
    args = [l.arg for l in ctx.tc.config.launches]
    last = args[-1] if args else 0
    for e in ctx.events:
        if e.a in ("cfg_write", "cfg_read"):
            e.raw = dict(e.raw or {})
            e.raw["queue_arg"] = last
    ctx.st("cfg.reg", "cfg:kernel_q", 12, {"queue_arg": last}, "cfg.queue_arg")
    _perturb_all_obs(ctx, last)


def tex_stale(ctx: SynthCtx) -> None:
    """#18: an access after reconfiguration reuses the stale mipmap offset."""
    cmds = [e for e in ctx.events if e.a == "tex_config"]
    if not cmds:
        return
    cfg_c = cmds[0]
    new_off = (cfg_c.raw or {}).get("offset", 0x40)
    for s in ctx.states:
        if s.sid == "tex.mipmap" and s.cycle >= cfg_c.cycle:
            s.val = {"level": (s.val or {}).get("level", 0), "offset": new_off - 0x20}
            break
    addrs = _obs_addrs(ctx)
    if len(addrs) > 1:
        ctx.arch.mem[addrs[-1]] = ctx.arch.mem.get(addrs[0], 0)


def wave_mismatch(ctx: SynthCtx) -> None:
    """#19: a delayed response is delivered to the wrong wavefront."""
    rsp = _first(ctx, "mem_rsp")
    if rsp is None:
        return
    rsp.raw = dict(rsp.raw or {}, wave=2, req_wave=0, delayed=True)
    rsp.o = "wrong_wave"
    _set_obs(ctx, 0, 0)


def wb_drop(ctx: SynthCtx) -> None:
    """#20: simultaneous writebacks are mis-arbitrated, one update is lost."""
    arbs = [e for e in ctx.events if e.a == "wb_arb"]
    if arbs:
        # two units write back in the same cycle; the arbitration drops one
        victim = arbs[0]
        ctx.ev("wb_arb", victim.u, victim.cycle, "dropped",
               raw={"reg": (victim.raw or {}).get("reg"), "unit": "lsu", "conflict": True})
        victim.o = "ok"
    ctx.st("rf.arb", "unit:rf", arbs[-1].cycle if arbs else 40,
           {"conflict": 1, "dropped": 1, "settled": 1}, "rf.arb_conflict")
    _set_obs(ctx, -1, 0)


def l1_stale(ctx: SynthCtx) -> None:
    """#21: an invalidation is undone by the late installation of a fill."""
    inst = _first(ctx, "fill_install")
    if inst is not None:
        inst.o = "stale_reinstall"
    for s in ctx.states:
        if s.sid == "l1.fill":
            s.val = {"state": "reinstalled"}
    _set_obs(ctx, -1, 0)


def squash_wb(ctx: SynthCtx) -> None:
    """#22: a squashed operation still writes back."""
    sq = _first(ctx, "squash")
    if sq is None:
        return
    ctx.ev("wb", sq.u, sq.cycle + 1, "squashed", phase="completed",
           raw={"squashed": True})
    ctx.st("scoreboard.busy", sq.u, sq.cycle, 1, "scoreboard.busy")
    ctx.st("pipe.valid", sq.u, sq.cycle, 0, "pipe.valid")
    _set_obs(ctx, 0, 0)


def barrier_early(ctx: SynthCtx) -> None:
    """A barrier releases before every arrival."""
    arr = _first(ctx, "barrier_arrive")
    if arr is None:
        return
    _drop_events(ctx, lambda e: e.a == "barrier_release" and e.cycle <= arr.cycle + 1)
    ctx.ev("barrier_release", "barrier:0", arr.cycle + 1, "ok")
    _set_obs(ctx, 0, 0)


def txn_lost(ctx: SynthCtx) -> None:
    """A request is dropped under backpressure and never completes."""
    req = _first(ctx, "mem_req")
    if req is None:
        return
    _drop_events(ctx, lambda e: e.a in ("mem_rsp", "mem_accept") and e.u == req.u)
    ctx.ev("mem_drop", req.u, req.cycle + 3, "lost", phase="stalled")
    ctx.cycles = max(ctx.cycles, req.cycle + int(ctx.tc.config.progress_timeout) + 100)


def kernel_hang(ctx: SynthCtx) -> None:
    """The kernel never reaches its exit condition."""
    _drop_events(ctx, lambda e: e.a == "kernel_end")
    ctx.timeout = True
    ctx.cycles = max(ctx.cycles, ctx.tc.config.max_cycles)


def div_r0_hang(ctx: SynthCtx) -> None:
    """#7: a division into r0 never completes, the core waits forever."""
    for e in list(ctx.events):
        if e.a == "div_done" and e.u.endswith(".r0"):
            ctx.events.remove(e)
            break
    _drop_events(ctx, lambda e: e.a == "kernel_end")
    ctx.timeout = True
    ctx.cycles = max(ctx.cycles, ctx.tc.config.max_cycles)


# --------------------------------------------------------------------------- #
# functional faults
# --------------------------------------------------------------------------- #


def signed_swap(ctx: SynthCtx) -> None:
    """#5: signed compare behaves like an unsigned compare."""
    addrs = _obs_addrs(ctx)
    if not addrs:
        return
    ctx.arch.mem[addrs[0]] = 0          # slt(-2, 1) -> 0
    if len(addrs) > 2:
        ctx.arch.mem[addrs[2]] = 1      # the branch takes the wrong path


def addr_trunc(ctx: SynthCtx) -> None:
    """#9: the high address aliases a lower one."""
    _set_obs(ctx, 0, 0xBAD0BAD0)


def jalr_misaligned(ctx: SynthCtx) -> None:
    """#10: JALR does not clear the LSB, the target block never runs."""
    _set_obs(ctx, 0, 0)


def fp_result(ctx: SynthCtx, value: int) -> None:
    for a in _obs_addrs(ctx):
        ctx.arch.mem[a] = value


def fp_nan_zero(ctx: SynthCtx) -> None:
    """#11/#13: NaN-producing operations return zero."""
    fp_result(ctx, 0)


def fp_overflow_nan(ctx: SynthCtx) -> None:
    """#12: overflow produces a NaN pattern."""
    fp_result(ctx, F32_NAN)


def fp_round_inf(ctx: SynthCtx) -> None:
    """#16: directed rounding on overflow returns infinity."""
    fp_result(ctx, F32_INF)


def sat_missing(ctx: SynthCtx) -> None:
    """#4: the saturating instruction executes without writing its result."""
    addrs = _obs_addrs(ctx)
    if not addrs:
        return
    ctx.arch.mem[addrs[0]] = 0x5        # the pre-existing value survives
    if len(addrs) > 1:
        ctx.arch.mem[addrs[1]] = 0x5


# --------------------------------------------------------------------------- #
# registry + shape mapping
# --------------------------------------------------------------------------- #

FAULTS: Dict[str, Callable[[SynthCtx], None]] = {
    "scoreboard_early_clear": scoreboard_early_clear,
    "scoreboard_clear_no_wb": scoreboard_clear_no_wb,
    "dirty_clear_early": dirty_clear_early,
    "lane_misroute": lane_misroute,
    "memcpy_desync": memcpy_desync,
    "kernel_init_stale": kernel_init_stale,
    "warp_slot_collision": warp_slot_collision,
    "dispatch_zero": dispatch_zero,
    "queue_cfg_overwrite": queue_cfg_overwrite,
    "tex_stale": tex_stale,
    "wave_mismatch": wave_mismatch,
    "wb_drop": wb_drop,
    "l1_stale": l1_stale,
    "squash_wb": squash_wb,
    "barrier_early": barrier_early,
    "txn_lost": txn_lost,
    "kernel_hang": kernel_hang,
    "div_r0_hang": div_r0_hang,
    "signed_swap": signed_swap,
    "addr_trunc": addr_trunc,
    "jalr_misaligned": jalr_misaligned,
    "fp_nan_zero": fp_nan_zero,
    "fp_overflow_nan": fp_overflow_nan,
    "fp_round_inf": fp_round_inf,
    "sat_missing": sat_missing,
}

# shape -> (fault name, obligation the oracle must report)
SHAPE_FAULT = {
    "SB_ISSUE_WB": ("scoreboard_early_clear", "OM_SCOREBOARD_NEW_WRITER"),
    "SB_CLEAR_CONDITION": ("scoreboard_clear_no_wb", "OM_SB_CLEAR_CONDITION"),
    "DCACHE_FLUSH_DIRTY": ("dirty_clear_early", "OM_DIRTY_SECURED"),
    "LSU_LANE_ROUTE": ("lane_misroute", "OM_LANE_ROUTE"),
    "DMA_ZERO_LEN": ("memcpy_desync", "OM_MEMCPY_LEN"),
    "DIV_R0": ("div_r0_hang", "OM_DIV_COMPLETION"),
    "KERNEL_INIT_REGS": ("kernel_init_stale", "OM_KERNEL_STATE_FRESH"),
    "WARP_SLOT_ALLOC": ("warp_slot_collision", "OM_WARP_SLOT_CAPACITY"),
    "DISPATCH_BLOCKS": ("dispatch_zero", "OM_DISPATCH_BLOCK_COUNT"),
    "LAUNCH_QUEUE_CFG": ("queue_cfg_overwrite", "OM_QUEUE_CFG_STABLE"),
    "TEX_MIPMAP": ("tex_stale", "OM_TEX_OFFSET_FRESH"),
    "WAVE_RESPONSE_ASSOC": ("wave_mismatch", "OM_WAVE_ASSOC"),
    "WB_ARB": ("wb_drop", "OM_WB_ARB_NO_LOSS"),
    "L1_INVALIDATE_FILL": ("l1_stale", "OM_L1_INVALIDATE_ORDER"),
    "ROLLBACK_WB": ("squash_wb", "OM_SQUASH_NO_WB"),
    "BARRIER_ORDER": ("barrier_early", "OM_BARRIER_ORDER"),
    "TX_PROGRESS": ("txn_lost", "OM_PROGRESS"),
    "MEM_ADDR_TRUNC": ("addr_trunc", "OM_ADDR_NO_TRUNC"),
    "JALR_ALIGN": ("jalr_misaligned", "OM_JALR_ALIGN"),
    "ALU_SIGNED": ("signed_swap", "OM_SIGNED_SEMANTICS"),
    "SAT_DECODE": ("sat_missing", "OM_SAT_DECODE"),
    "FP_NAN_INF": ("fp_nan_zero", "OM_FP_SPECIAL"),
    "FP_ZERO_INF": ("fp_nan_zero", "OM_FP_SPECIAL"),
    "FP_OVERFLOW_PATTERN": ("fp_overflow_nan", "OM_FP_SPECIAL"),
    "FP_ROUND_OVERFLOW": ("fp_round_inf", "OM_FP_ROUND_OVERFLOW"),
}


def apply(ctx: SynthCtx, names: Sequence[str]) -> None:
    for n in names:
        fn = FAULTS.get(n)
        if fn is None:
            raise KeyError("unknown fault %r" % (n,))
        fn(ctx)
    ctx.faults = list(names)                 # type: ignore[attr-defined]
