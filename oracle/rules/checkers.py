"""Obligation checkers: one function per omega in the model.

Every checker receives a concrete activation (the bound events and the observed
state window), the whole observation ``O`` and the test case, and returns the
violations of that obligation.  Two rules govern all of them:

* **no evidence, no verdict** -- a checker that cannot see the signals it needs
  returns nothing (the report says "not observable" instead of "bug");
* **semantics, not implementation** -- the expectation comes from the ISA, the
  bus protocol or the design invariant recorded in the obligation, never from
  "what the RTL currently does".

The three families of Section 4.4 map to:

    state_transition  values updated without their required conditions
    lifecycle         pending transactions released before completion
    progress          obligations that never resolve
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common.esd.match import Binding
from common.ir import Program, reg_index
from common.ir import is_reg
from common.types import Event, OMEGA, StateObs, TestCase, obj_kind
from oracle.status_collector import Observation
from oracle.violation import Violation

MAX_EVIDENCE = 6


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _viol(b: Binding, obs: Observation, tc: TestCase, message: str,
          obj: Optional[str] = None, cycle: int = 0,
          evidence: Optional[Dict[str, Any]] = None,
          family: Optional[str] = None, detector: str = "interaction",
          oid: Optional[str] = None) -> Violation:
    oid = oid or b.rel.omega.oid
    info = OMEGA.get(oid, {})
    return Violation(
        target=b.rel.target, rid=b.rel.rid, oid=oid, message=message,
        family=family or info.get("family", "state_transition"),
        kind=info.get("kind", "interaction"), detector=detector,
        obj=obj or b.obj, cycle=cycle, shape=b.rel.shape,
        evidence=dict(evidence or {}), testcase_id=tc.id)


def _bit_series(obs: Observation, sid: str, obj: str) -> List[Tuple[int, Any]]:
    return obs.series(sid, obj)


def _clear_transitions(obs: Observation, sid: str, obj: str) -> List[Tuple[int, Any, Any]]:
    out = []
    for c, old, new in obs.transitions(sid, obj):
        if old in (1, True) and new in (0, False):
            out.append((c, old, new))
    return out


def _regs_read_before_write(prog: Program) -> List[str]:
    written: Set[int] = {0}
    out: List[str] = []
    for ins in prog.instrs:
        for r in ins.regs_read():
            if is_reg(r):
                i = reg_index(r)
                if i not in written and r not in out:
                    out.append(r)
        for r in ins.regs_written():
            written.add(reg_index(r))
    return out


def _expected_blocks(tc: TestCase) -> int:
    g = tc.config.grid
    return max(1, g[0]) * max(1, g[1]) * max(1, g[2])


def _expected_warps(tc: TestCase) -> int:
    return max(1, int(tc.config.warps_per_block))


# --------------------------------------------------------------------------- #
# state transition rules
# --------------------------------------------------------------------------- #


def check_scoreboard_new_writer(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_SCOREBOARD_NEW_WRITER: an older writeback must not clear a new writer."""
    reg = b.obj
    issue = b.ev("new_issue")
    old_wb = b.ev("old_wb")
    if issue is None or not obs.has_state("scoreboard.busy"):
        return []

    series = obs.series("scoreboard.busy", reg)          # end-of-cycle values
    at_issue = None
    for c, val in series:
        if c == issue.cycle:
            at_issue = val
    out: List[Violation] = []

    # (a) the dependency established by the issue must exist at the end of that cycle
    if at_issue in (0, False):
        own_wb_same = [e for e in obs.events_of(("wb",), reg)
                       if e.cycle <= issue.cycle and e is not old_wb]
        if not own_wb_same:
            out.append(_viol(
                b, obs, tc,
                "scoreboard entry %s is not busy at the end of cycle %d even though a "
                "writer was just issued to it" % (reg, issue.cycle),
                obj=reg, cycle=issue.cycle,
                evidence={"issue_cycle": issue.cycle,
                          "old_wb_cycle": old_wb.cycle if old_wb else None,
                          "end_of_cycle_value": at_issue,
                          "rule": "a newly issued writer must stay busy"}))

    # (b) a clear in the window before the new writer's own writeback
    later_wb = [e for e in obs.events_of(("wb",), reg)
                if e.cycle > issue.cycle and e is not old_wb]
    own_wb = later_wb[0].cycle if later_wb else None
    for c, old_val, new_val in _clear_transitions(obs, "scoreboard.busy", reg):
        if c <= issue.cycle:
            continue
        if own_wb is not None and c >= own_wb:
            continue
        out.append(_viol(
            b, obs, tc,
            "scoreboard busy bit of %s cleared at cycle %d while the writer issued at "
            "cycle %d is still pending%s" % (reg, c, issue.cycle,
                                             "" if own_wb is None else
                                             " (own writeback at %d)" % own_wb),
            obj=reg, cycle=c,
            evidence={"issue_cycle": issue.cycle, "clear_cycle": c,
                      "old_wb_cycle": old_wb.cycle if old_wb else None,
                      "new_wb_cycle": own_wb}))
    return out[:MAX_EVIDENCE]


def check_scoreboard_clear_condition(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_SB_CLEAR_CONDITION: a busy bit clears only on its own writeback."""
    reg = b.obj
    if not obs.has_state("scoreboard.busy"):
        return []
    out = []
    for c, _old, _new in _clear_transitions(obs, "scoreboard.busy", reg):
        wb_same = [e for e in obs.events_of(("wb",), reg) if abs(e.cycle - c) <= 1]
        if wb_same:
            continue
        out.append(_viol(
            b, obs, tc,
            "scoreboard busy bit of %s cleared at cycle %d without a writeback of that "
            "register" % (reg, c), obj=reg, cycle=c,
            evidence={"clear_cycle": c, "nearest_wb":
                      min([e.cycle for e in obs.events_of(("wb",), reg)] or [-1], key=lambda x: abs(x - c))}))
        if len(out) >= MAX_EVIDENCE:
            break
    return out


def check_dirty_secured(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_DIRTY_SECURED: DirtyClear(L) requires WritebackAccepted(L) or SafelyBuffered(L)."""
    line = b.obj
    if not obs.has_state("dcache.dirty"):
        return []
    clears = _clear_transitions(obs, "dcache.dirty", line)
    if not clears:
        return []
    accepts = obs.events_of(("mem_accept",)) + obs.events_of(("mem_fire",))
    out = []
    for c, _o, _n in clears:
        accepted_before = [e for e in accepts if e.cycle <= c]
        if accepted_before:
            continue
        pending_after = [s for s in obs.states_of("wb.pending", line)
                         if s.cycle >= c and bool(s.val) in (True, 1)]
        buffered = bool(pending_after) and any(
            e.cycle > c for e in accepts)
        if buffered:
            continue
        stalls = [e for e in obs.events_of(("mem_stall",)) if abs(e.cycle - c) <= 8]
        out.append(_viol(
            b, obs, tc,
            "dirty state of %s cleared at cycle %d before the writeback was accepted or "
            "safely buffered%s" % (line, c,
                                   " (memory backpressure present)" if stalls else ""),
            obj=line, cycle=c,
            evidence={"clear_cycle": c,
                      "next_accept": (min([e.cycle for e in accepts if e.cycle > c],
                                          default=None)),
                      "backpressure_events": len(stalls),
                      "pending_samples": [{"cycle": s.cycle, "val": s.val}
                                          for s in pending_after[:4]]}))
        if len(out) >= MAX_EVIDENCE:
            break
    return out


def check_lane_route(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_LANE_ROUTE: shared-memory lanes are never routed through the DCache."""
    out = []
    for e in obs.events_of(("lane_route",)):
        if e.o in ("misdirected", "via_dcache", "dcache"):
            out.append(_viol(b, obs, tc,
                             "shared-memory access %s was routed through the DCache at "
                             "cycle %d" % (e.u, e.cycle), obj=e.u, cycle=e.cycle,
                             evidence={"outcome": e.o, "raw": e.raw or {}}))
    if obs.has_state("lane.route"):
        for s in obs.states_of("lane.route", b.obj):
            val = s.val if isinstance(s.val, dict) else {"route": s.val}
            if str(val.get("route", "")).lower() in ("dcache", "cache") and \
                    str(val.get("space", "shared")).lower() == "shared":
                out.append(_viol(b, obs, tc,
                                 "lane routing state reports a shared access on the DCache "
                                 "path at cycle %d" % s.cycle, obj=s.obj, cycle=s.cycle,
                                 evidence={"value": s.val, "signal": s.sig}))
    return out[:MAX_EVIDENCE]


def check_dispatch_blocks(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_DISPATCH_BLOCK_COUNT: every configured block is dispatched."""
    expected = _expected_blocks(tc)
    dispatched = len({e.u for e in obs.events_of(("dispatch",))})
    final = None
    if obs.has_state("dispatch.count"):
        ser = obs.series("dispatch.count", "kernel:0") or obs.series("dispatch.count", b.obj)
        if not ser:
            for s in obs.states_of("dispatch.count"):
                ser = [(s.cycle, s.val)]
        if ser:
            final = ser[-1][1]
    if final is not None and isinstance(final, int):
        dispatched = max(dispatched, final)
    if dispatched >= expected:
        return []
    if not obs.events_of(("dispatch",)) and not obs.has_state("dispatch.count"):
        return []
    return [_viol(b, obs, tc,
                  "only %d of %d blocks were dispatched before the kernel finished"
                  % (dispatched, expected), obj="kernel:0", cycle=obs.cycles,
                  evidence={"expected_blocks": expected, "dispatched": dispatched,
                            "kernel_end": bool(obs.events_of(("kernel_end",)))})]


def check_warp_slot_capacity(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_WARP_SLOT_CAPACITY: an occupied slot is never handed to another warp."""
    if not obs.has_state("warp.slot"):
        return []
    out = []
    for slot in obs.objects_of_kind("slot"):
        owner: Dict[Any, int] = {}
        for s in obs.states_of("warp.slot", slot):
            val = s.val if isinstance(s.val, dict) else {"owner": s.val}
            cur = val.get("owner")
            if cur in (None, 0):
                owner.clear()
                continue
            if cur in owner:
                out.append(_viol(
                    b, obs, tc,
                    "slot %s still owned by warp %s when it was allocated again at cycle %d"
                    % (slot, cur, s.cycle), obj=slot, cycle=s.cycle,
                    evidence={"warp": cur, "value": s.val, "signal": s.sig}))
                return out
            owner[cur] = s.cycle
    return out


def check_kernel_state_fresh(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_KERNEL_STATE_FRESH: every register read before it is written starts at zero."""
    starts = obs.events_of(("kernel_start",))
    if len(starts) < 2 and len(tc.config.launches) < 2:
        return []
    needed = [r for r in _regs_read_before_write(tc.program) if reg_index(r) != 0]
    if not needed:
        return []
    inits = {e.u for e in obs.events_of(("reg_init",))}
    if not inits:
        # fall back to the architectural dump: a stale value is visible directly
        stale = []
        for lidx in range(1, len(tc.config.launches)):
            for reg in needed:
                for key, val in obs.arch.regs.items():
                    if key.endswith("." + reg) and val != 0:
                        stale.append((key, val))
                        break
        if not stale:
            return []
        return [_viol(b, obs, tc,
                      "register %s still holds %#x at the start of a later launch "
                      "(kernel initialisation is incomplete)" % (stale[0][0], stale[0][1]),
                      obj="reg:" + stale[0][0],
                      cycle=starts[1].cycle if len(starts) > 1 else 0,
                      evidence={"stale_registers": stale[:MAX_EVIDENCE],
                                "read_before_write": needed})]
    missing = [r for r in needed if ("reg:" + r) not in inits and r not in inits]
    if not missing:
        return []
    return [_viol(b, obs, tc,
                  "kernel initialisation did not clear %s before the kernel reads it"
                  % ", ".join(missing[:4]), obj="reg:" + missing[0],
                  cycle=starts[-1].cycle if starts else 0,
                  evidence={"missing": missing, "initialised": sorted(inits)[:8]})]


def check_queue_cfg_stable(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_QUEUE_CFG_STABLE: a queued launch uses the configuration captured at enqueue."""
    expected = [launch.arg for launch in tc.config.launches]
    observed: List[Tuple[int, Any]] = []
    for s in obs.states_of("cfg.reg"):
        val = s.val if isinstance(s.val, dict) else {"queue_arg": s.val}
        if "queue_arg" in val:
            observed.append((s.cycle, val["queue_arg"]))
    for e in obs.events_of(("cfg_read",)):
        if e.raw and "value" in e.raw:
            observed.append((e.cycle, e.raw["value"]))
    if not observed or len(expected) < 2:
        return []
    out = []
    for idx, (cycle, got) in enumerate(observed[:len(expected)]):
        if int(got) != int(expected[idx]):
            out.append(_viol(
                b, obs, tc,
                "launch %d observed configuration %s but was queued with %s"
                % (idx, got, expected[idx]), obj="cfg:kernel_q", cycle=cycle,
                evidence={"launch_index": idx, "observed": got, "expected": expected[idx],
                          "all_expected": expected}))
    return out


def check_tex_offset_fresh(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_TEX_OFFSET_FRESH: an access after a reconfiguration uses the new offset."""
    if not obs.has_state("tex.mipmap"):
        return []
    reconfigs = obs.events_of(("tex_config",))
    if not reconfigs:
        return []
    out = []
    for cfg_ev in reconfigs:
        new_offset = (cfg_ev.raw or {}).get("offset")
        if new_offset is None:
            continue
        for s in obs.states_of("tex.mipmap", b.obj, min_cycle=cfg_ev.cycle):
            val = s.val if isinstance(s.val, dict) else {"offset": s.val}
            off = val.get("offset")
            if off is not None and int(off) != int(new_offset):
                out.append(_viol(
                    b, obs, tc,
                    "texture %s used mipmap offset %s at cycle %d after being "
                    "reconfigured to %s" % (b.obj, off, s.cycle, new_offset),
                    obj=b.obj, cycle=s.cycle,
                    evidence={"configured": new_offset, "observed": off,
                              "reconfig_cycle": cfg_ev.cycle}))
                break
        if out:
            break
    return out


def check_wave_assoc(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_WAVE_ASSOC: a response belongs to the wavefront that issued the request."""
    out = []
    for e in obs.events_of(("mem_rsp",)):
        raw = e.raw or {}
        if e.o in ("mismatch", "wrong_wave", "misassociated"):
            out.append(_viol(b, obs, tc,
                             "memory response at cycle %d was delivered to the wrong "
                             "wavefront" % e.cycle, obj=e.u, cycle=e.cycle,
                             evidence={"outcome": e.o, "raw": raw}))
            continue
        if "wave" in raw and "req_wave" in raw and raw["wave"] != raw["req_wave"]:
            out.append(_viol(
                b, obs, tc,
                "response for %s went to wavefront %s instead of %s at cycle %d"
                % (e.u, raw["wave"], raw["req_wave"], e.cycle), obj=e.u, cycle=e.cycle,
                evidence={"response_wave": raw["wave"], "request_wave": raw["req_wave"]}))
    return out[:MAX_EVIDENCE]


def check_wb_arb(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_WB_ARB_NO_LOSS: simultaneous writebacks both reach the register file."""
    out = []
    for e in obs.events_of(("wb_arb",)):
        if e.o in ("dropped", "lost"):
            out.append(_viol(b, obs, tc,
                             "writeback %s was dropped by arbitration at cycle %d"
                             % (e.u, e.cycle), obj=e.u, cycle=e.cycle,
                             evidence={"outcome": e.o, "raw": e.raw or {}}))
    if obs.has_state("rf.arb"):
        for s in obs.states_of("rf.arb"):
            val = s.val if isinstance(s.val, dict) else {"conflict": s.val}
            if val.get("dropped") or (val.get("conflict") and val.get("settled") == 1):
                out.append(_viol(b, obs, tc,
                                 "register file arbitration reported a dropped update at "
                                 "cycle %d" % s.cycle, obj=s.obj, cycle=s.cycle,
                                 evidence={"value": s.val, "signal": s.sig}))
                break
    return out[:MAX_EVIDENCE]


def check_squash_no_wb(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_SQUASH_NO_WB: a squashed operation does not update the register file."""
    out = []
    for e in obs.events_of(("wb",)):
        raw = e.raw or {}
        if raw.get("squashed") or e.o == "squashed":
            out.append(_viol(b, obs, tc,
                             "squashed operation still wrote %s back at cycle %d"
                             % (e.u, e.cycle), obj=e.u, cycle=e.cycle,
                             evidence={"raw": raw, "outcome": e.o}))
    if obs.has_state("pipe.valid") and obs.has_state("rf.write"):
        for sq in obs.events_of(("squash",)):
            for w in obs.events_of(("wb",), sq.u, min_cycle=sq.cycle, max_cycle=sq.cycle + 2):
                wv = obs.states_of("pipe.valid", sq.u, min_cycle=sq.cycle, max_cycle=sq.cycle)
                if any(bool(s.val) in (False, 0) for s in wv):
                    out.append(_viol(
                        b, obs, tc,
                        "writeback of %s at cycle %d overlapped the squash of the same "
                        "operation" % (sq.u, w.cycle), obj=sq.u, cycle=w.cycle,
                        evidence={"squash_cycle": sq.cycle, "wb_cycle": w.cycle}))
    return out[:MAX_EVIDENCE]


def check_barrier_order(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_BARRIER_ORDER: release only after every arrival of the epoch."""
    expected = _expected_warps(tc)
    out = []
    arrivals = 0
    for e in obs.events:
        if e.a == "barrier_arrive":
            arrivals += 1
        elif e.a == "barrier_release":
            if arrivals < expected:
                out.append(_viol(
                    b, obs, tc,
                    "barrier released at cycle %d after %d of %d warp arrivals"
                    % (e.cycle, arrivals, expected), obj=e.u, cycle=e.cycle,
                    evidence={"arrivals": arrivals, "expected": expected}))
            arrivals = 0
    return out[:MAX_EVIDENCE]


# --------------------------------------------------------------------------- #
# lifecycle rules
# --------------------------------------------------------------------------- #


def check_txn_no_loss(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_TXN_NO_LOSS: a request stays valid until accepted or safely buffered."""
    out = []
    for e in obs.events_of(("mem_req",)):
        if e.o == "lost":
            out.append(_viol(b, obs, tc,
                             "request %s disappeared before acceptance at cycle %d"
                             % (e.u, e.cycle), obj=e.u, cycle=e.cycle,
                             evidence={"raw": e.raw or {}}))
    for e in obs.events_of(("mem_drop",)):
        out.append(_viol(b, obs, tc,
                         "memory transaction %s dropped at cycle %d while downstream "
                         "backpressure was asserted" % (e.u, e.cycle),
                         obj=e.u, cycle=e.cycle, evidence={"raw": e.raw or {}}))
    return out[:MAX_EVIDENCE]


def check_l1_invalidate_order(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_L1_INVALIDATE_ORDER: an older fill never resurrects an invalidated line."""
    out = []
    for e in obs.events_of(("fill_install",)):
        if e.o in ("stale_reinstall", "after_invalidate", "stale"):
            out.append(_viol(
                b, obs, tc,
                "fill of %s was installed at cycle %d after the line had been invalidated"
                % (e.u, e.cycle), obj=e.u, cycle=e.cycle,
                evidence={"outcome": e.o, "raw": e.raw or {}}))
    if obs.has_state("l1.fill"):
        for s in obs.states_of("l1.fill", b.obj):
            val = s.val if isinstance(s.val, dict) else {"state": s.val}
            if str(val.get("state", "")).lower() in ("stale", "reinstalled"):
                out.append(_viol(b, obs, tc,
                                 "L1 line %s holds stale data installed after invalidation "
                                 "at cycle %d" % (s.obj, s.cycle), obj=s.obj, cycle=s.cycle,
                                 evidence={"value": s.val, "signal": s.sig}))
    return out[:MAX_EVIDENCE]


def check_memcpy_len(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_MEMCPY_LEN: zero length copies keep the command stream aligned."""
    cmds = [e for e in obs.events_of(("dma_copy",))]
    if not cmds:
        return []
    expected = [c.params for c in tc.config.conditions_of("dma_copy")]
    out = []
    for e in cmds:
        if e.o in ("misapplied", "desync", "misread"):
            out.append(_viol(b, obs, tc,
                             "copy command at cycle %d was misapplied after a zero-length "
                             "copy (controller/command stream desynchronised)" % e.cycle,
                             obj=e.u, cycle=e.cycle, evidence={"raw": e.raw or {}}))
    got = [int((e.raw or {}).get("len", -1)) for e in cmds]
    want = [int(p.get("len", -1)) for p in expected]
    if expected and got != want:
        out.append(_viol(
            b, obs, tc,
            "controller executed command lengths %s, host issued %s" % (got[:6], want[:6]),
            obj="unit:dma", cycle=cmds[0].cycle,
            evidence={"executed": got[:6], "issued": want[:6]}))
    return out[:MAX_EVIDENCE]


# --------------------------------------------------------------------------- #
# progress rules
# --------------------------------------------------------------------------- #


def check_div_completion(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_DIV_COMPLETION: every started division produces a completion."""
    starts = obs.events_of(("div_start",))
    if not starts:
        return []
    out = []
    for st in starts:
        done = obs.events_of(("div_done",), st.u, min_cycle=st.cycle)
        if done:
            continue
        if not (obs.events_of(("kernel_end",)) or obs.trace.timeout):
            continue
        out.append(_viol(
            b, obs, tc,
            "division into %s started at cycle %d never reported completion"
            % (st.u, st.cycle), obj=st.u, cycle=st.cycle,
            evidence={"start_cycle": st.cycle, "kernel_end":
                      (obs.events_of(("kernel_end",)) or [None])[0].cycle
                      if obs.events_of(("kernel_end",)) else None,
                      "timeout": obs.trace.timeout}))
    return out[:MAX_EVIDENCE]


def check_progress(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_PROGRESS: outstanding transactions complete once backpressure is released."""
    timeout = int(tc.config.progress_timeout or 40000)
    out = []
    for e in obs.events_of(("mem_req",)):
        if e.o in ("accepted", "completed"):
            continue
        responses = obs.events_of(("mem_rsp", "mem_accept", "mem_fire"), e.u, min_cycle=e.cycle)
        if responses:
            continue
        if not (obs.events_of(("kernel_end",)) or obs.trace.timeout):
            continue
        if (obs.cycles - e.cycle) < timeout and not obs.trace.timeout:
            continue
        out.append(_viol(
            b, obs, tc,
            "transaction %s issued at cycle %d never completed (%d cycles unresolved)"
            % (e.u, e.cycle, obs.cycles - e.cycle), obj=e.u, cycle=e.cycle,
            evidence={"issue_cycle": e.cycle, "end_cycle": obs.cycles,
                      "progress_timeout": timeout, "timeout_flag": obs.trace.timeout},
            family="progress", detector="progress"))
    return out[:MAX_EVIDENCE]


def check_no_hang(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_NO_HANG: a launched kernel reaches its exit condition."""
    if not obs.trace.timeout:
        return []
    return [_viol(b, obs, tc,
                  "kernel did not terminate within %d cycles" % obs.cycles,
                  obj="kernel:0", cycle=obs.cycles,
                  evidence={"cycles": obs.cycles, "max_cycles": tc.config.max_cycles},
                  family="progress", detector="progress", oid="OM_NO_HANG")]


def check_txn_ownership(b: Binding, obs: Observation, tc: TestCase) -> List[Violation]:
    """OM_TXN_OWNERSHIP: a response is consumed by the transaction that owns it."""
    out = []
    for e in obs.events_of(("mem_rsp",)):
        raw = e.raw or {}
        if e.o in ("mismatch", "wrong_txn") or ("txn" in raw and "req_txn" in raw
                                                and raw["txn"] != raw["req_txn"]):
            out.append(_viol(b, obs, tc,
                             "response for %s was consumed by the wrong transaction at "
                             "cycle %d" % (e.u, e.cycle), obj=e.u, cycle=e.cycle,
                             evidence={"raw": raw}))
    return out[:MAX_EVIDENCE]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

REGISTRY = {
    "OM_SCOREBOARD_NEW_WRITER": check_scoreboard_new_writer,
    "OM_SB_CLEAR_CONDITION": check_scoreboard_clear_condition,
    "OM_DIRTY_SECURED": check_dirty_secured,
    "OM_LANE_ROUTE": check_lane_route,
    "OM_TXN_NO_LOSS": check_txn_no_loss,
    "OM_TXN_OWNERSHIP": check_txn_ownership,
    "OM_L1_INVALIDATE_ORDER": check_l1_invalidate_order,
    "OM_SQUASH_NO_WB": check_squash_no_wb,
    "OM_BARRIER_ORDER": check_barrier_order,
    "OM_KERNEL_STATE_FRESH": check_kernel_state_fresh,
    "OM_QUEUE_CFG_STABLE": check_queue_cfg_stable,
    "OM_TEX_OFFSET_FRESH": check_tex_offset_fresh,
    "OM_WAVE_ASSOC": check_wave_assoc,
    "OM_WB_ARB_NO_LOSS": check_wb_arb,
    "OM_DISPATCH_BLOCK_COUNT": check_dispatch_blocks,
    "OM_WARP_SLOT_CAPACITY": check_warp_slot_capacity,
    "OM_MEMCPY_LEN": check_memcpy_len,
    "OM_DIV_COMPLETION": check_div_completion,
    "OM_PROGRESS": check_progress,
    "OM_NO_HANG": check_no_hang,
}
