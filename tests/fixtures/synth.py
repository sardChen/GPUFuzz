"""Synthetic trace synthesis for oracle self tests.

**This is not a GPU model.**  It is a stimulus generator that turns one test case
into a plausible, self-consistent DUT trace (``O = <A, E, S>``) built from the
reference execution, so that the oracle's rules can be regression tested without
RTL.  Fault variants apply the trace-level signature of a known bug (Table 2) to
the same trace, which is exactly what the rules have to catch.

Nothing measured with this module is a performance or coverage result; the README
says so explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.config import Config
from common.ir import Instr, MemRef, Program
from common.targets import TargetSpec
from common.types import (ArchEffects, Ctx, Event, StateObs, TestCase, Trace,
                          obj_ref)
from oracle.reference import ReferenceModel

LATENCY = {
    "add": 2, "sub": 2, "and": 2, "or": 2, "xor": 2, "mov": 1, "mov_imm": 1,
    "shl": 2, "shr": 2, "sra": 2, "slt": 2, "sltu": 2, "min": 2, "max": 2,
    "neg": 2, "abs": 2, "mul": 4, "mulh": 4, "mad": 5, "div": 12, "rem": 14,
    "sat_add": 2, "sat_sub": 2,
    "fadd": 5, "fsub": 5, "fmul": 5, "fdiv": 14, "fsqrt": 20,
    "f2i": 6, "i2f": 6, "fcvt_rtz": 6, "fcvt_rup": 6, "fcvt_rdn": 6,
    "fcvt_rne": 6, "fneg": 2, "fabs": 2,
    "ld": 8, "st": 3, "tex": 10, "bar": 2, "fence": 1, "exit": 1, "nop": 1,
    "spin": 1, "la": 1, "jmp": 1, "jmp_reg": 1, "br": 1, "label": 1,
    "tid": 1, "ntid": 1, "ctaid": 1, "nctaid": 1, "laneid": 1, "warpid": 1,
    "clock": 1, "setcfg": 1, "getcfg": 1, "cache_op": 2, "memcpy_len": 4,
}

LINE = 64


@dataclass
class SynthCtx:
    tc: TestCase
    target: TargetSpec
    events: List[Event] = field(default_factory=list)
    states: List[StateObs] = field(default_factory=list)
    arch: ArchEffects = field(default_factory=ArchEffects)
    lines: Dict[str, int] = field(default_factory=dict)      # line -> dirty cycle
    regs: Dict[str, int] = field(default_factory=dict)       # reg -> issue cycle
    warp_cycles: Dict[int, List[Tuple[int, Instr, int]]] = field(default_factory=dict)
    timeout: bool = False
    cycles: int = 0
    notes: List[str] = field(default_factory=list)
    warps: int = 1
    blocks: int = 1

    # -- helpers used by the fault variants ---------------------------- #
    def ev(self, a: str, u: str, cycle: int, o: str = "ok", raw: Optional[Dict[str, Any]] = None,
           phase: str = "outstanding", warp: int = 0) -> Event:
        e = Event(a=a, u=u, o=o, cycle=cycle,
                  ctx=Ctx(env={"warp": warp}, phase=phase, obj=u), raw=raw)
        self.events.append(e)
        return e

    def st(self, sid: str, obj: str, cycle: int, val: Any, sig: str = "") -> StateObs:
        s = StateObs(sid=sid, obj=obj, cycle=cycle, val=val, sig=sig)
        self.states.append(s)
        return s

    def obs_addrs(self) -> List[int]:
        base = self.tc.config.gmem_base + 0x800
        return sorted(a for a in self.arch.mem if a >= base and a < base + 0x10000)

    def by_type(self, a: str) -> List[Event]:
        return [e for e in self.events if e.a == a]


# --------------------------------------------------------------------------- #
# clean synthesis
# --------------------------------------------------------------------------- #


class TraceSynthesizer:
    def __init__(self, target: TargetSpec, latency_scale: float = 1.0) -> None:
        self.target = target
        self.ref = ReferenceModel(target)
        self.scale = latency_scale

    # ------------------------------------------------------------------ #
    def build(self, tc: TestCase, faults: Sequence[str] = ()) -> SynthCtx:
        cfg = tc.config
        lanes = int(self.target.caps["lanes"])
        warps = max(1, min(cfg.threads_per_block // max(1, lanes), 8))
        blocks = min(cfg.blocks, 16)
        ref = self.ref.run(tc)
        arch = ArchEffects.from_json(ref.arch.to_json())

        ctx = SynthCtx(tc=tc, target=self.target, arch=arch, warps=warps, blocks=blocks)
        ctx.ev("kernel_start", "kernel:0", 4, "ok", phase="issued")
        for r in range(min(int(tc.program.nregs), 32)):
            ctx.ev("reg_init", obj_ref("reg", "r%d" % r), 5, "ok", phase="issued")
        for b in range(blocks):
            ctx.ev("dispatch", obj_ref("block", b), 6 + b, "ok", phase="issued")

        cfg_conds = {c.ev: c for c in cfg.conditions}
        dirty_cycle: Dict[str, int] = {}
        store_events: List[Tuple[int, str]] = []
        barrier_times: Dict[int, List[int]] = {}
        first_store = None
        txn_seq = 0
        t0 = 20

        for w in range(warps):
            t = t0 + w * 2
            dyn = self.ref.dynamic_trace(tc, warp=w, lane=0, block=0)
            timeline: List[Tuple[int, Instr, int]] = []
            for pc, ins, regs in dyn:
                lat = int(LATENCY.get(ins.op, 2) * self.scale)
                issue_c = t
                wb_c = t + lat
                txn = "txn:%d" % txn_seq
                txn_seq += 1 if ins.op in ("ld", "st") else 0
                self._emit_insn(ctx, ins, w, issue_c, wb_c, cfg, txn)
                timeline.append((issue_c, ins, wb_c))
                if ins.op == "st":
                    line = self._line_of(ins, cfg)
                    dirty_cycle[line] = max(dirty_cycle.get(line, 0), issue_c + 2)
                    store_events.append((issue_c + 2, line))
                    if first_store is None:
                        first_store = issue_c + 2
                if ins.op == "bar":
                    barrier_times.setdefault(w, []).append(issue_c)
                t += 1
            ctx.warp_cycles[w] = timeline

        end_cycle = max((max((c for c, _, _ in tl), default=0) for tl in
                         ctx.warp_cycles.values()), default=t0) + 30
        ctx.cycles = end_cycle + 10

        # ---- flush / writeback / backpressure sequence ----------------- #
        flush = cfg_conds.get("cache_flush")
        bp = cfg_conds.get("mem_backpressure")
        if flush and store_events and not first_store is None:
            line = store_events[0][1]
            f_cycle = dirty_cycle.get(line, first_store + 4) + 2
            ctx.ev("flush", line, f_cycle, "ok", phase="issued")
            ctx.ev("flush_select", line, f_cycle, "ok", phase="issued")
            ctx.st("dcache.dirty", line, dirty_cycle.get(line, first_store), 1, "dcache.way_dirty")
            ctx.ev("wb_pending_set", line, f_cycle + 1, "ok", phase="outstanding")
            ctx.st("wb.pending", line, f_cycle + 1, 1, "dcache.wb_pending")
            hold = int((bp.params.get("hold", 0) if bp else 0))
            accept_c = f_cycle + 2 + hold
            if bp:
                start = f_cycle + int(bp.params.get("start", 0))
                for k in range(0, max(1, hold), 4):
                    ctx.ev("mem_stall", line, start + k, "stalled", phase="stalled")
            ctx.ev("mem_req", line, f_cycle + 2, "ok", phase="outstanding")
            ctx.ev("mem_accept", line, accept_c, "accepted", phase="completed")
            ctx.st("wb.pending", line, accept_c, 0, "dcache.wb_pending")
            ctx.st("dcache.dirty", line, accept_c, 0, "dcache.way_dirty")

        inval = cfg_conds.get("cache_invalidate")
        if inval and store_events:
            line = store_events[0][1]
            inv_c = dirty_cycle.get(line, first_store or t0) + int(inval.params.get("after", 1)) + 2
            ctx.ev("fill_start", line, max(1, inv_c - 1), "ok")
            ctx.ev("invalidate", line, inv_c, "ok")
            ctx.ev("fill_install", line, inv_c + 3, "ok")
            ctx.st("l1.fill", line, inv_c + 3, {"state": "valid", "data": "fresh"}, "l1.fill_state")

        # txns that were never accepted (progress baseline is clean)
        for e in ctx.by_type("mem_req"):
            ctx.ev("mem_rsp", e.u, e.cycle + 6, "ok", phase="completed")

        # barriers: release once every warp arrived
        if barrier_times:
            # barrier epochs are serialised: epoch k+1 starts after epoch k released
            n_epochs = max(len(v) for v in barrier_times.values())
            base = min(min(v) for v in barrier_times.values() if v)
            stride = 4 * max(1, warps)
            for k in range(n_epochs):
                arrivals = [base + k * stride + w * 2
                            for w in sorted(barrier_times) if k < len(barrier_times[w])]
                if not arrivals:
                    continue
                for w, a in enumerate(sorted(arrivals)):
                    ctx.ev("barrier_arrive", "barrier:0", a, "ok", warp=w)
                ctx.ev("barrier_release", "barrier:0", max(arrivals) + 1, "ok")
                ctx.st("barrier.count", "barrier:0", max(arrivals), len(arrivals),
                       "sync.arrivals")

        # ---- configuration / texture / dma / queue events -------------- #
        for cond in cfg.conditions:
            if cond.ev == "tex_reconfig":
                tex_acc = [e for e in ctx.events if e.a == "tex_access"]
                tcfg = (tex_acc[0].cycle + 2) if tex_acc else (end_cycle - 20)
                ctx.ev("tex_config", obj_ref("tex", 0), tcfg, "ok",
                       raw={"level": cond.params.get("level", 0),
                            "offset": cond.params.get("offset", 0)})
                ctx.st("tex.mipmap", obj_ref("tex", 0), tcfg,
                       {"level": cond.params.get("level", 0),
                        "offset": cond.params.get("offset", 0)}, "tex.mip_offset")
                # accesses after the reconfiguration observe the new offset
                for st in ctx.states:
                    if st.sid == "tex.mipmap" and st.cycle > tcfg and isinstance(st.val, dict):
                        st.val = {"level": st.val.get("level", 0),
                                  "offset": cond.params.get("offset", 0)}
            if cond.ev == "dma_copy":
                ctx.ev("dma_copy", "unit:dma", end_cycle - 15,
                       "zero_length" if int(cond.params.get("len", 0)) == 0 else "ok",
                       raw={"len": cond.params.get("len", 0),
                            "dst": cond.params.get("dst", 0)})
            if cond.ev == "launch_queue":
                ctx.ev("queue_push", "cfg:kernel_q", 8, "ok")
                ctx.ev("cfg_write", "cfg:kernel_q", 9, "ok",
                       raw={"queue_arg": tc.config.launches[-1].arg if tc.config.launches else 0})
                ctx.ev("queue_pop", "cfg:kernel_q", 10, "ok")

        # ---- kernel end ------------------------------------------------- #
        if not ctx.timeout:
            ctx.ev("kernel_end", "kernel:0", end_cycle, "ok", phase="completed")

        # ---- architectural dump for every warp -------------------------- #
        ctx.cycles = max(ctx.cycles, end_cycle + 20)
        return ctx

    # ------------------------------------------------------------------ #
    def _emit_insn(self, ctx: SynthCtx, ins: Instr, warp: int, issue_c: int,
                   wb_c: int, cfg: Config, txn: str = "") -> None:
        reg = ins.dst
        if reg and ins.op not in ("st", "br", "jmp", "jmp_reg", "bar", "fence",
                                  "label", "exit", "cache_op"):
            obj = obj_ref("reg", "w%d.%s" % (warp, reg))
            ctx.ev("issue", obj, issue_c, "issued", phase="issued", warp=warp)
            # writeback first, then a possible new issue in the same cycle:
            # a correct scoreboard keeps the new dependency
            ctx.ev("wb", obj, wb_c, "ok", phase="completed", warp=warp)
            ctx.st("scoreboard.busy", obj, issue_c, 1, "scoreboard.busy")
            ctx.st("scoreboard.busy", obj, wb_c, 0, "scoreboard.busy")
            ctx.ev("commit", obj, wb_c, "ok", phase="completed", warp=warp)
            if ins.op in ("fmul", "fdiv", "fadd", "fsub"):
                ctx.ev("wb_arb", "unit:rf", wb_c, "ok", raw={"reg": reg, "unit": "fpu"})
            if ins.op in ("ld",):
                ctx.ev("wb_arb", "unit:rf", wb_c, "ok", raw={"reg": reg, "unit": "lsu"})
        if ins.op in ("div", "rem"):
            dobj = obj_ref("reg", "w%d.%s" % (warp, ins.dst or "r0"))
            ctx.ev("div_start", dobj, issue_c, "ok",
                   raw={"divisor_reg": ins.srcs[1] if len(ins.srcs) > 1 else None})
            ctx.ev("div_done", dobj, wb_c, "ok")
        if ins.op in ("ld", "st"):
            line = self._line_of(ins, cfg)
            txn = txn or line
            ctx.ev("mem_req", txn, issue_c, "ok", phase="outstanding", warp=warp,
                   raw={"line": line})
            ctx.ev("mem_rsp", txn, wb_c, "ok", phase="completed", warp=warp,
                   raw={"line": line})
            if ins.op == "st":
                ctx.ev("store", line, issue_c, "ok", phase="issued", warp=warp)
                ctx.ev("lane_route", obj_ref("warp", warp),
                       issue_c, (ins.mem.space if ins.mem else "global"),
                       raw={"space": (ins.mem.space if ins.mem else "global")})
                ctx.ev("dirty_set", line, issue_c + 1, "ok", phase="outstanding")
            else:
                ctx.ev("load", line, issue_c, "ok", phase="issued", warp=warp)
        if ins.op == "tex":
            ctx.ev("tex_access", obj_ref("tex", 0), wb_c, "ok",
                   raw={"level": (ins.aux or {}).get("level", 0)})
            ctx.st("tex.mipmap", obj_ref("tex", 0), wb_c, {"level": (ins.aux or {}).get("level", 0),
                                                          "offset": 0}, "tex.mip_offset")
        if ins.op == "br":
            # reconvergence squash of the divergent path; the register it would
            # have written is tracked as its own object
            ctx.ev("squash", obj_ref("reg", "w%d.diverge" % warp), issue_c, "ok",
                   phase="issued")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _line_of(ins: Instr, cfg: Config) -> str:
        m = ins.mem or MemRef()
        addr = cfg.gmem_base + (m.offset if m.offset else 0)
        if m.space == "shared":
            return obj_ref("line", "smem+%#x" % (m.offset // LINE * LINE))
        return obj_ref("line", "0x%x" % (addr // LINE * LINE))


def finish(ctx: SynthCtx) -> Trace:
    """Sort the trace and attach the remaining metadata."""
    events = sorted(ctx.events, key=lambda e: e.cycle)
    states = sorted(ctx.states, key=lambda s: (s.cycle, s.sid, s.obj))
    meta = {
        "warps": ctx.warps, "blocks": ctx.blocks,
        "mem_range": [ctx.tc.config.gmem_base - 0x100,
                      ctx.tc.config.gmem_base + 0x10000],
        "synthetic": True,
        "faults": list(getattr(ctx, "faults", ()) or ()),
    }
    return Trace(target=ctx.target.name, testcase_id=ctx.tc.id, events=events,
                 states=states, arch=ctx.arch, cov=[], cycles=ctx.cycles,
                 exit_code=0, timeout=ctx.timeout, meta=meta)
