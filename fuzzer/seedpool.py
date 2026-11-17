"""Seed pool and the priority tuple of Algorithm 1.

    priority(T') = < 1[Delta != empty], N_new, sum_x 1/(1+Freq[x]), N_cov >

The tuple is compared lexicographically: consistency violations first, then new
interactions, then rare interactions, then newly covered RTL points.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.types import Relation, TestCase


@dataclass(frozen=True)
class Priority:
    violation: bool = False
    new_interactions: int = 0
    rarity: float = 0.0
    new_cov: int = 0

    def key(self) -> Tuple[int, int, float, int]:
        return (1 if self.violation else 0, int(self.new_interactions),
                round(float(self.rarity), 6), int(self.new_cov))

    def describe(self) -> str:
        return "<Δ=%d, new=%d, rare=%.2f, cov=%d>" % (
            1 if self.violation else 0, self.new_interactions, self.rarity, self.new_cov)


@dataclass
class SeedEntry:
    tc: TestCase
    priority: Priority
    order: int
    picks: int = 0
    hits: int = 0

    @property
    def rid(self) -> str:
        return self.tc.rid or ""

    @property
    def shape(self) -> str:
        return self.tc.provenance.get("shape", "")


class SeedPool:
    """Bounded pool of interesting test cases, selected per target relation."""

    def __init__(self, cap: int = 2048) -> None:
        self.cap = cap
        self._entries: List[SeedEntry] = []
        self._counter = 0
        self.enqueued = 0
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    # ------------------------------------------------------------------ #
    def enqueue(self, tc: TestCase, priority: Priority) -> SeedEntry:
        self._counter += 1
        entry = SeedEntry(tc=tc, priority=priority, order=self._counter)
        self._entries.append(entry)
        self.enqueued += 1
        if len(self._entries) > self.cap:
            self._evict()
        return entry

    def _evict(self) -> None:
        """Drop the least interesting seed: low priority, then least recently used."""
        victim = min(range(len(self._entries)),
                     key=lambda i: (self._entries[i].priority.key(),
                                    -self._entries[i].picks,
                                    -self._entries[i].order))
        if victim == len(self._entries) - 1:
            self._entries.pop()
        else:
            self._entries.pop(victim)
        self.dropped += 1

    # ------------------------------------------------------------------ #
    def select(self, rel: Relation, rng: random.Random) -> TestCase:
        """Pick a seed that reaches (or approaches) the relation's context."""
        same_rid = [e for e in self._entries if e.rid == rel.rid]
        same_shape = [e for e in self._entries if e.shape == rel.shape]
        cands = same_rid or same_shape or list(self._entries)
        if not cands:                                   # pragma: no cover
            raise LookupError("seed pool is empty")
        # favour high priority and rarely picked seeds
        weights = []
        for e in cands:
            v, new, rare, cov = e.priority.key()
            w = (1.0 + v) * (1.0 + new) * (1.0 + rare) * (1.0 + 0.1 * cov)
            w /= (1.0 + e.picks)
            weights.append(max(w, 1e-6))
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        chosen = cands[-1]
        for e, w in zip(cands, weights):
            acc += w
            if r <= acc:
                chosen = e
                break
        chosen.picks += 1
        return chosen.tc

    def mark_hit(self, tc: TestCase) -> None:
        for e in self._entries:
            if e.tc.id == tc.id:
                e.hits += 1
                return

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        if not self._entries:
            return {"size": 0, "enqueued": self.enqueued, "dropped": self.dropped}
        with_violation = sum(1 for e in self._entries if e.priority.violation)
        by_rid: Dict[str, int] = {}
        for e in self._entries:
            by_rid[e.rid] = by_rid.get(e.rid, 0) + 1
        return {
            "size": len(self._entries), "enqueued": self.enqueued,
            "dropped": self.dropped, "violating": with_violation,
            "distinct_relations": len(by_rid),
            "top_priority": max(e.priority.key() for e in self._entries),
        }


def is_interesting(p: Priority) -> bool:
    """Section 4.3: retain cases with violations, new or rare interactions, new cover."""
    return p.violation or p.new_interactions > 0 or p.rarity >= 1.0 or p.new_cov > 0
