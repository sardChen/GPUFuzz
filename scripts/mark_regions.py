#!/usr/bin/env python3
"""Generate an interaction-sensitive region file from an RTL checkout.

    python3 scripts/mark_regions.py --rtl ~/gpu/ventus --out specs/regions/ventus.txt

The default patterns are the components named in Section 2/6.2 of the paper
(scoreboard, scheduler, cache flush/dirty/writeback, memory handshake,
synchronisation).  Pass --pattern to add your own.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.coverage import mark_regions_from_source

DEFAULT_PATTERNS = [
    r"\bscoreboard\b", r"\bbusy\b", r"\bscheduler\b", r"\bwarp_(alloc|slot|sched)\b",
    r"\bdirty\b", r"\bflush\b", r"\bmshr\b", r"\bfill_(valid|install|done)\b",
    r"\b(aw|ar|r|w|b)(valid|ready)\b", r"\bbarrier\b", r"\bsync\b",
    r"\b(invalidate|inv_valid)\b", r"\bwb_(valid|pending|arb)\b",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", action="append", default=[])
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args(argv)
    patterns = DEFAULT_PATTERNS + args.pattern
    n = mark_regions_from_source(args.rtl, patterns, args.out)
    print("%d marked regions -> %s" % (n, args.out))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
