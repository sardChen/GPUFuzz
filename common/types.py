"""Core data model of GPUFuzz: execution contexts, events, states, temporal
constraints, consistency obligations, ESD relations and runtime traces.

The names follow Section 4.2 of the paper:

    c = <chi, rho, nu>          execution context
    e = <a, u, o_e>             execution event
    s                           interaction state (RTL element or cluster)
    theta                       temporal constraint
    omega                       consistency obligation
    r = <c, e, s, theta, omega> ESD relation

Everything in this module is pure data + cheap helpers; matching relations
against a runtime observation lives in :mod:`common.esd.match`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .util import sha

# --------------------------------------------------------------------------- #
# event vocabulary
# --------------------------------------------------------------------------- #
# Canonical event types.  Target signal maps translate raw RTL pulses onto this
# vocabulary (see adapters/signals/*.json), so the model and the oracle stay
# independent of a specific design's signal names.

EV_TYPES = (
    # instruction flow
    "issue", "wb", "commit", "squash", "rollback", "decode", "stall",
    # memory path
    "mem_req", "mem_rsp", "mem_accept", "mem_stall", "mem_fire", "mem_drop",
    "load", "store", "lane_route", "addr_trunc",
    # cache / transaction lifecycle
    "dirty_set", "dirty_clear", "flush", "flush_select", "fill_start",
    "fill_install", "invalidate", "wb_pending_set", "wb_pending_clear",
    "wb_arb", "wb_lost",
    # scheduling / resources
    "sched", "alloc", "free", "dispatch", "kernel_start", "kernel_end",
    "barrier_arrive", "barrier_release", "div_start", "div_done",
    # configuration / special units
    "cfg_write", "cfg_read", "queue_push", "queue_pop", "tex_config",
    "tex_access", "launch", "dma_copy", "reg_init",
)
EV_SET = frozenset(EV_TYPES)

# Outcomes attached to e = <a, u, o_e>.
EV_OUTCOMES = ("ok", "accepted", "rejected", "stalled", "lost", "buffered",
               "completed", "premature", "stale", "squashed", "dropped")

# Canonical interaction-state identifiers (s).  A relation clusters one or more
# of these; the cluster is what the oracle watches.
STATE_IDS = (
    "scoreboard.busy",      # per-register pending-write bits
    "rf.write",             # register-file write port activity
    "dcache.dirty",         # dirty bits per line
    "dcache.valid",         # valid bits per line
    "dcache.tag",           # tag array
    "wb.pending",           # pending writeback transaction
    "mshr.fill",            # outstanding fill
    "txn.state",            # generic transaction state
    "l1.fill",
    "warp.slot",            # warp allocation table
    "sched.state",          # warp scheduler state
    "cfg.reg",              # GPU configuration registers
    "tex.mipmap",           # texture mipmap offset registers
    "rf.arb",               # writeback arbitration
    "dispatch.count",       # block dispatch counters
    "barrier.count",        # barrier arrival counters
    "pipe.valid",           # pipeline valid bits
    "div.state",            # divider FSM
    "lane.route",           # per-lane memory routing
    "mem.addr",             # address datapath
    "alu.flag",             # ALU comparison flags
    "fpu.result",           # floating point result register
    "dec.table",            # decode table entry
    "pc.value",             # program counter / branch target
)
STATE_SET = frozenset(STATE_IDS)

# Object-reference kinds.  Object references are strings "kind:value", which is
# what lets relations talk about "the same cache line" or "the same txn".
OBJ_KINDS = ("reg", "lane", "line", "txn", "slot", "warp", "block", "kernel",
             "cfg", "tex", "wave", "mem", "pc", "unit", "buff", "barrier")

PHASES = ("issued", "outstanding", "completed", "stalled")


def obj_ref(kind: str, value: Any) -> str:
    return "%s:%s" % (kind, value)


def obj_kind(ref: str) -> str:
    return ref.split(":", 1)[0] if ":" in ref else ""


def obj_value(ref: str) -> str:
    return ref.split(":", 1)[1] if ":" in ref else ref


# --------------------------------------------------------------------------- #
# c = <chi, rho, nu>
# --------------------------------------------------------------------------- #


@dataclass
class Ctx:
    """Execution context ``c = <chi, rho, nu>``.

    ``env``   (chi) execution environment: kernel/block/warp/lane/config keys.
    ``phase`` (rho) execution phase: issued/outstanding/completed/stalled.
    ``obj``   (nu)  the affected hardware object, as an object reference.
    """

    env: Dict[str, Any] = field(default_factory=dict)
    phase: str = "outstanding"
    obj: str = ""

    def key(self) -> str:
        parts = ["%s=%s" % (k, self.env[k]) for k in sorted(self.env)]
        return "%s|%s|%s" % (self.phase, self.obj, ",".join(parts))

    def matches(self, other: "Ctx") -> bool:
        if self.phase and other.phase and self.phase != other.phase:
            return False
        if self.obj and other.obj and self.obj != other.obj:
            return False
        for k, v in self.env.items():
            if k in other.env and other.env[k] != v:
                return False
        return True

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"phase": self.phase}
        if self.obj:
            out["obj"] = self.obj
        if self.env:
            out["env"] = self.env
        return out

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Ctx":
        return Ctx(env=dict(d.get("env") or {}), phase=d.get("phase", ""),
                   obj=d.get("obj", ""))


# --------------------------------------------------------------------------- #
# e = <a, u, o_e>
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    a: str                      # event type (see EV_TYPES)
    u: str                      # object reference
    o: str = "ok"               # outcome
    cycle: int = 0              # timestamp, from the DUT trace
    ctx: Optional[Ctx] = None
    raw: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.a not in EV_SET:
            raise ValueError("unknown event type %r (see common.types.EV_TYPES)" % (self.a,))

    def key(self) -> str:
        return "%s@%s|%s|%s" % (self.a, self.cycle, self.u, self.o)

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"a": self.a, "u": self.u, "o": self.o, "c": self.cycle}
        if self.ctx:
            d["ctx"] = self.ctx.to_json()
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Event":
        return Event(a=d.get("a") or d.get("ev"), u=d.get("u", ""),
                     o=d.get("o", "ok"), cycle=int(d.get("c", d.get("cycle", 0))),
                     ctx=Ctx.from_json(d["ctx"]) if d.get("ctx") else None,
                     raw=d.get("raw"))


# --------------------------------------------------------------------------- #
# s
# --------------------------------------------------------------------------- #


@dataclass
class StateObs:
    """One observed value of an interaction state element at a cycle."""

    sid: str                    # one of STATE_IDS
    obj: str                    # object reference (line/txn/reg/slot/...)
    cycle: int
    val: Any                    # scalar or dict of named fields
    sig: str = ""               # originating RTL signal (for reporting)
    note: str = ""

    def __post_init__(self) -> None:
        if self.sid not in STATE_SET:
            raise ValueError("unknown state id %r (see common.types.STATE_IDS)" % (self.sid,))

    def key(self) -> str:
        return "%s|%s|%s|%s" % (self.sid, self.obj, self.cycle, json.dumps(self.val, sort_keys=True, default=str))

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"s": self.sid, "o": self.obj, "c": self.cycle, "v": self.val}
        if self.sig:
            d["sig"] = self.sig
        if self.note:
            d["n"] = self.note
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "StateObs":
        return StateObs(sid=d.get("s"), obj=d.get("o", ""), cycle=int(d.get("c", 0)),
                        val=d.get("v"), sig=d.get("sig", ""), note=d.get("n", ""))


# --------------------------------------------------------------------------- #
# theta
# --------------------------------------------------------------------------- #

ATOM_KINDS = ("before", "after", "overlap", "within", "not_before", "same_cycle",
              "coincide")


@dataclass(frozen=True)
class Atom:
    """One temporal atom between two event roles.

    before      : lhs.cycle <= rhs.cycle
    after       : lhs.cycle >= rhs.cycle
    overlap     : lhs.cycle <= rhs.cycle <= lhs.cycle + param
    within      : rhs.cycle - lhs.cycle <= param
    not_before  : lhs.cycle >  rhs.cycle is forbidden (lhs must not precede)
    same_cycle  : lhs.cycle == rhs.cycle
    coincide    : |lhs.cycle - rhs.cycle| <= param  (order insensitive overlap)
    """

    kind: str
    lhs: str
    rhs: str
    param: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ATOM_KINDS:
            raise ValueError("unknown temporal atom %r" % (self.kind,))

    def describe(self) -> str:
        sym = {"before": "<<", "after": ">>", "overlap": "||", "within": "<=%d" % self.param,
               "not_before": "!<<", "same_cycle": "==",
               "coincide": "~%d" % self.param}[self.kind]
        return "%s %s %s" % (self.lhs, sym, self.rhs)

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "lhs": self.lhs, "rhs": self.rhs, "param": self.param}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Atom":
        return Atom(kind=d["kind"], lhs=d["lhs"], rhs=d["rhs"], param=int(d.get("param", 0)))


@dataclass(frozen=True)
class Theta:
    atoms: Tuple[Atom, ...] = ()

    def describe(self) -> str:
        if not self.atoms:
            return "(no temporal constraint)"
        return ", ".join(a.describe() for a in self.atoms)

    def to_json(self) -> List[Dict[str, Any]]:
        return [a.to_json() for a in self.atoms]

    @staticmethod
    def from_json(d: Any) -> "Theta":
        if d is None:
            return Theta()
        return Theta(tuple(Atom.from_json(a) for a in d))

    def roles(self) -> List[str]:
        out: List[str] = []
        for a in self.atoms:
            for r in (a.lhs, a.rhs):
                if r not in out:
                    out.append(r)
        return out


# --------------------------------------------------------------------------- #
# omega
# --------------------------------------------------------------------------- #

# Obligation families follow Section 4.4: state-transition rules, transaction
# lifecycle rules and forward-progress rules.  "functional" obligations are
# checked by the functional detector instead (ISA/architectural semantics).

OMEGA: Dict[str, Dict[str, Any]] = {
    # ---- state transition (interaction) ---------------------------------- #
    "OM_SCOREBOARD_NEW_WRITER": dict(
        family="state_transition", kind="interaction",
        text="A newly issued writer must stay busy when an older writeback "
             "clears the same scoreboard entry."),
    "OM_SB_CLEAR_CONDITION": dict(
        family="state_transition", kind="interaction",
        text="A busy bit may be cleared only if the writeback that clears it "
             "belongs to the currently pending writer."),
    "OM_LANE_ROUTE": dict(
        family="state_transition", kind="interaction",
        text="Shared-memory lanes must be routed by the LSU, never through DCache."),
    "OM_KERNEL_STATE_FRESH": dict(
        family="state_transition", kind="interaction",
        text="Kernel initialization clears every architectural register, not just r0."),
    "OM_QUEUE_CFG_STABLE": dict(
        family="state_transition", kind="interaction",
        text="Queued launches keep the configuration captured at enqueue time."),
    "OM_TEX_OFFSET_FRESH": dict(
        family="state_transition", kind="interaction",
        text="A texture reconfiguration replaces active mipmap offsets."),
    "OM_WAVE_ASSOC": dict(
        family="state_transition", kind="interaction",
        text="A delayed LSU response is associated with its own wavefront."),
    "OM_WB_ARB_NO_LOSS": dict(
        family="state_transition", kind="interaction",
        text="Concurrent writebacks from distinct units must not drop an update."),
    "OM_DISPATCH_BLOCK_COUNT": dict(
        family="state_transition", kind="interaction",
        text="The dispatcher uses the block count that arrives with the launch."),
    "OM_WARP_SLOT_CAPACITY": dict(
        family="state_transition", kind="interaction",
        text="Warp allocation never reuses an occupied slot."),
    "OM_MEMCPY_LEN": dict(
        family="state_transition", kind="interaction",
        text="Zero-length copies keep host and controller command streams aligned."),
    # ---- transaction lifecycle ------------------------------------------- #
    "OM_DIRTY_SECURED": dict(
        family="lifecycle", kind="interaction",
        text="DirtyClear(L) requires WritebackAccepted(L) or SafelyBuffered(L)."),
    "OM_TXN_NO_LOSS": dict(
        family="lifecycle", kind="interaction",
        text="An outstanding request stays valid until accepted or safely buffered."),
    "OM_TXN_OWNERSHIP": dict(
        family="lifecycle", kind="interaction",
        text="A transaction response is consumed by the transaction that owns it."),
    "OM_L1_INVALIDATE_ORDER": dict(
        family="lifecycle", kind="interaction",
        text="An invalidated line must not be re-installed by an older fill."),
    "OM_SQUASH_NO_WB": dict(
        family="lifecycle", kind="interaction",
        text="A squashed operation must not update the register file."),
    "OM_BARRIER_ORDER": dict(
        family="lifecycle", kind="interaction",
        text="A barrier releases a warp only after all arrivals of the epoch."),
    # ---- forward progress ------------------------------------------------ #
    "OM_PROGRESS": dict(
        family="progress", kind="interaction",
        text="Every pending transaction completes after backpressure is released."),
    "OM_DIV_COMPLETION": dict(
        family="progress", kind="interaction",
        text="A division always produces a writeback completion."),
    "OM_NO_HANG": dict(
        family="progress", kind="interaction",
        text="A launched kernel reaches its exit condition."),
    # ---- functional (architectural) -------------------------------------- #
    "OM_SIGNED_SEMANTICS": dict(
        family="state_transition", kind="functional",
        text="Signed compare/branch use signed arithmetic."),
    "OM_FP_SPECIAL": dict(
        family="state_transition", kind="functional",
        text="NaN/infinity/zero special cases follow IEEE-754."),
    "OM_FP_ROUND_OVERFLOW": dict(
        family="state_transition", kind="functional",
        text="Overflow results follow the active rounding mode."),
    "OM_SAT_DECODE": dict(
        family="state_transition", kind="functional",
        text="Every decoded instruction writes its architectural result."),
    "OM_ADDR_NO_TRUNC": dict(
        family="state_transition", kind="functional",
        text="Addresses are not truncated in the memory datapath."),
    "OM_JALR_ALIGN": dict(
        family="state_transition", kind="functional",
        text="JALR clears the least significant target address bit."),
    "OM_ARCH_MISMATCH": dict(
        family="state_transition", kind="functional",
        text="Architectural effects match the reference model."),
}

OMEGA_IDS = frozenset(OMEGA)


@dataclass(frozen=True)
class Obligation:
    """``omega`` of a relation.  ``oid == "⊥"`` means undetermined.

    Relations with undetermined obligations are still generated/mutated (they
    guide exploration) but excluded from correctness checking, per Section 4.2.
    """

    oid: str
    params: Tuple[Tuple[str, Any], ...] = ()
    justified: bool = True
    text: str = ""

    def __post_init__(self) -> None:
        if self.oid not in OMEGA and self.oid not in ("⊥", "BOT", None):
            raise ValueError("unknown obligation %r" % (self.oid,))

    @property
    def undetermined(self) -> bool:
        return self.oid in ("⊥", "BOT", None) or not self.justified

    @property
    def kind(self) -> str:
        return OMEGA.get(self.oid, {}).get("kind", "interaction")

    @property
    def family(self) -> str:
        return OMEGA.get(self.oid, {}).get("family", "state_transition")

    def param(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)

    def describe(self) -> str:
        if self.undetermined:
            return "⊥ (undetermined)"
        txt = OMEGA[self.oid]["text"]
        if self.params:
            txt += " [" + ", ".join("%s=%s" % (k, v) for k, v in self.params) + "]"
        return txt

    def to_json(self) -> Dict[str, Any]:
        return {"oid": self.oid, "params": [list(p) for p in self.params],
                "justified": self.justified}

    @staticmethod
    def from_json(d: Any) -> "Obligation":
        if not d:
            return Obligation("⊥", justified=False)
        if isinstance(d, str):
            return Obligation(d)
        return Obligation(oid=d.get("oid", "⊥"),
                          params=tuple(tuple(p) for p in d.get("params", ())),
                          justified=bool(d.get("justified", True)))


BOT = Obligation("⊥", justified=False)


# --------------------------------------------------------------------------- #
# r = <c, e, s, theta, omega>
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjVar:
    """A relation-local object variable, e.g. ``L`` of kind ``line``."""

    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in OBJ_KINDS:
            raise ValueError("unknown object kind %r" % (self.kind,))

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class EventRole:
    """One member of the related event set ``e``.

    ``role`` is the symbol used by ``theta`` atoms; ``types`` are the canonical
    event types that may bind to it; ``obj`` is the object it must share (None
    means "any object, not correlated").
    """

    role: str
    types: Tuple[str, ...]
    obj: Optional[ObjVar] = None
    outcomes: Tuple[str, ...] = ()
    # Optional roles express "expected completion" obligations: the relation is
    # activated by the trigger role alone, and the checker reports the missing
    # event instead of the relation never activating (e.g. no dispatch at all).
    optional: bool = False

    def __post_init__(self) -> None:
        for t in self.types:
            if t not in EV_SET:
                raise ValueError("role %s: unknown event type %r" % (self.role, t))

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "types": list(self.types)}
        if self.optional:
            d["optional"] = True
        if self.obj:
            d["obj"] = self.obj.name
        if self.outcomes:
            d["outcomes"] = list(self.outcomes)
        return d


@dataclass
class RelStats:
    n_hit: int = 0
    n_try: int = 0
    activations: int = 0
    violations: int = 0
    last_seen: int = 0
    guards: Dict[str, int] = field(default_factory=dict)

    @property
    def kappa(self) -> float:
        """Activation confidence, eq. in Section 4.2: (n_hit+1)/(n_try+2)."""
        return (self.n_hit + 1.0) / (self.n_try + 2.0)

    def to_json(self) -> Dict[str, Any]:
        return {"n_hit": self.n_hit, "n_try": self.n_try,
                "activations": self.activations, "violations": self.violations,
                "last_seen": self.last_seen, "guards": self.guards,
                "kappa": round(self.kappa, 4)}


@dataclass
class Relation:
    rid: str
    target: str
    shape: str
    ctx: Dict[str, Any]                 # chi constraints that select the context
    objvar: ObjVar
    roles: Tuple[EventRole, ...]
    sids: Tuple[str, ...]
    theta: Theta
    omega: Obligation
    source: str = "semantic"            # semantic | rtl | runtime
    evidence: Dict[str, Any] = field(default_factory=dict)
    stats: RelStats = field(default_factory=RelStats)
    bug_hint: str = ""                  # e.g. "#3" from the paper's Table 2

    def __post_init__(self) -> None:
        for s in self.sids:
            if s not in STATE_SET:
                raise ValueError("%s: unknown state id %r" % (self.rid, s))
        role_names = {r.role for r in self.roles}
        for atom in self.theta.atoms:
            for side in (atom.lhs, atom.rhs):
                if side not in role_names:
                    raise ValueError("%s: theta references unknown role %r" % (self.rid, side))
        if self.omega.oid not in ("⊥", "BOT") and not self.omega.justified:
            raise ValueError("%s: unjustified obligation must be ⊥" % self.rid)

    # -- convenience ------------------------------------------------------- #
    @property
    def checkable(self) -> bool:
        return not self.omega.undetermined

    def role(self, name: str) -> EventRole:
        for r in self.roles:
            if r.role == name:
                return r
        raise KeyError(name)

    def describe(self) -> str:
        evs = ", ".join("%s<%s>" % (r.role, "|".join(r.types)) for r in self.roles)
        return ("%s [%s] c=(%s) e={%s} s={%s} theta=(%s) omega=%s"
                % (self.rid, self.target, ",".join("%s=%s" % kv for kv in sorted(self.ctx.items())),
                   evs, ",".join(self.sids), self.theta.describe(), self.omega.oid))

    def to_json(self) -> Dict[str, Any]:
        return {
            "rid": self.rid, "target": self.target, "shape": self.shape,
            "ctx": self.ctx, "objvar": self.objvar.to_json(),
            "roles": [r.to_json() for r in self.roles], "sids": list(self.sids),
            "theta": self.theta.to_json(), "omega": self.omega.to_json(),
            "source": self.source, "evidence": self.evidence,
            "stats": self.stats.to_json(), "bug_hint": self.bug_hint,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Relation":
        objv = d["objvar"]
        objvar = ObjVar(objv["name"], objv["kind"])
        roles = []
        for r in d["roles"]:
            roles.append(EventRole(role=r["role"], types=tuple(r["types"]),
                                   obj=objvar if r.get("obj") == objvar.name else None,
                                   outcomes=tuple(r.get("outcomes", ())),
                                   optional=bool(r.get("optional", False))))
        rel = Relation(
            rid=d["rid"], target=d["target"], shape=d["shape"], ctx=dict(d.get("ctx") or {}),
            objvar=objvar, roles=tuple(roles), sids=tuple(d["sids"]),
            theta=Theta.from_json(d.get("theta")), omega=Obligation.from_json(d.get("omega")),
            source=d.get("source", "semantic"), evidence=dict(d.get("evidence") or {}),
            bug_hint=d.get("bug_hint", ""))
        st = d.get("stats") or {}
        rel.stats = RelStats(n_hit=st.get("n_hit", 0), n_try=st.get("n_try", 0),
                             activations=st.get("activations", 0),
                             violations=st.get("violations", 0),
                             last_seen=st.get("last_seen", 0),
                             guards=dict(st.get("guards") or {}))
        return rel


class RelationSet:
    """``R``: the relation set of an ESDModel, with a stable iteration order."""

    def __init__(self, relations: Optional[Iterable[Relation]] = None) -> None:
        self._by_id: Dict[str, Relation] = {}
        for r in relations or ():
            self.add(r)

    def add(self, rel: Relation) -> None:
        if rel.rid in self._by_id:
            raise ValueError("duplicate relation id %r" % rel.rid)
        self._by_id[rel.rid] = rel

    def get(self, rid: str) -> Relation:
        return self._by_id[rid]

    def __contains__(self, rid: str) -> bool:
        return rid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def ids(self) -> List[str]:
        return list(self._by_id)

    def filter(self, target: Optional[str] = None, shape: Optional[str] = None,
               checkable: Optional[bool] = None,
               source: Optional[str] = None) -> List[Relation]:
        out = []
        for r in self:
            if target and r.target != target:
                continue
            if shape and r.shape != shape:
                continue
            if checkable is not None and r.checkable != checkable:
                continue
            if source and r.source != source:
                continue
            out.append(r)
        return out

    def to_json(self) -> Dict[str, Any]:
        return {"relations": [r.to_json() for r in self]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "RelationSet":
        return RelationSet(Relation.from_json(r) for r in d.get("relations", []))


# --------------------------------------------------------------------------- #
# runtime trace  O = <A, E, S>
# --------------------------------------------------------------------------- #


@dataclass
class ArchEffects:
    """``A``: architectural effects (instruction results, regs, memory)."""

    regs: Dict[str, int] = field(default_factory=dict)      # "w0.l3.r5" -> value
    mem: Dict[int, int] = field(default_factory=dict)       # addr -> word
    status: Dict[str, Any] = field(default_factory=dict)    # exit code, retires, ...
    pc: Dict[str, int] = field(default_factory=dict)        # "w0" -> final pc

    def to_json(self) -> Dict[str, Any]:
        return {
            "regs": {k: int(v) for k, v in self.regs.items()},
            "mem": {str(k): int(v) for k, v in self.mem.items()},
            "status": self.status,
            "pc": {k: int(v) for k, v in self.pc.items()},
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "ArchEffects":
        return ArchEffects(
            regs={k: int(v) for k, v in (d.get("regs") or {}).items()},
            mem={int(k): int(v) for k, v in (d.get("mem") or {}).items()},
            status=dict(d.get("status") or {}),
            pc={k: int(v) for k, v in (d.get("pc") or {}).items()},
        )


@dataclass
class Trace:
    """One execution of one test case, as produced by a DUT run."""

    target: str
    testcase_id: str
    events: List[Event] = field(default_factory=list)
    states: List[StateObs] = field(default_factory=list)
    arch: ArchEffects = field(default_factory=ArchEffects)
    cov: List[str] = field(default_factory=list)            # coverage point ids
    cycles: int = 0
    exit_code: int = 0
    timeout: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def events_of(self, types: Iterable[str]) -> List[Event]:
        ts = set(types)
        return [e for e in self.events if e.a in ts]

    def states_of(self, sid: str, obj: Optional[str] = None) -> List[StateObs]:
        return [s for s in self.states
                if s.sid == sid and (obj is None or s.obj == obj)]

    def objects(self, kind: str) -> List[str]:
        seen: List[str] = []
        for src in list(self.events) + list(self.states):
            u = getattr(src, "u", None) or getattr(src, "obj", "")
            if obj_kind(u) == kind and u not in seen:
                seen.append(u)
        return seen

    def window(self, lo: int, hi: int) -> List[StateObs]:
        return [s for s in self.states if lo <= s.cycle <= hi]

    def fingerprint(self) -> str:
        return sha([e.key() for e in self.events], self.cycles, self.exit_code, n=16)

    def to_json(self) -> Dict[str, Any]:
        return {
            "target": self.target, "testcase_id": self.testcase_id,
            "events": [e.to_json() for e in self.events],
            "states": [s.to_json() for s in self.states],
            "arch": self.arch.to_json(), "cov": list(self.cov),
            "cycles": self.cycles, "exit_code": self.exit_code,
            "timeout": self.timeout, "meta": self.meta,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Trace":
        return Trace(
            target=d["target"], testcase_id=d.get("testcase_id", "?"),
            events=[Event.from_json(e) for e in d.get("events", [])],
            states=[StateObs.from_json(s) for s in d.get("states", [])],
            arch=ArchEffects.from_json(d.get("arch") or {}),
            cov=list(d.get("cov") or []), cycles=int(d.get("cycles", 0)),
            exit_code=int(d.get("exit_code", 0)), timeout=bool(d.get("timeout", False)),
            meta=dict(d.get("meta") or {}))


# --------------------------------------------------------------------------- #
# test cases  T = <K, C>
# --------------------------------------------------------------------------- #


@dataclass
class TestCase:
    id: str
    target: str
    program: Any                        # common.ir.Program
    config: Any                         # common.config.Config
    rid: Optional[str] = None           # relation it was synthesized for
    protected: Tuple[str, ...] = ()     # slots that mutation must preserve
    provenance: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id, "target": self.target, "rid": self.rid,
            "program": self.program.to_json(), "config": self.config.to_json(),
            "protected": list(self.protected), "provenance": self.provenance,
            "generation": self.generation, "parent": self.parent,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "TestCase":
        from .ir import Program
        from .config import Config
        return TestCase(
            id=d["id"], target=d["target"], program=Program.from_json(d["program"]),
            config=Config.from_json(d["config"]), rid=d.get("rid"),
            protected=tuple(d.get("protected") or ()),
            provenance=dict(d.get("provenance") or {}),
            generation=int(d.get("generation", 0)), parent=d.get("parent"))


@dataclass
class RunResult:
    """Result of handing one test case to a DUT driver."""

    ok: bool
    trace: Optional[Trace] = None
    driver: str = "none"
    error: str = ""
    wall_s: float = 0.0
    artifact_dir: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
