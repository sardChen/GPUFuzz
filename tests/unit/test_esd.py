import unittest

from common.esd.match import activate, match_relation
from common.types import (Atom, BOT, Ctx, Event, EventRole, ObjVar, Obligation,
                          Relation, StateObs, TestCase, Theta, Trace)


def _rel(**kw):
    objv = ObjVar("L", "line")
    base = dict(rid="R1", target="ventus", shape="T", ctx={}, objvar=objv,
                roles=(EventRole("store", ("store",), objv),
                       EventRole("flush", ("flush",), objv)),
                sids=("dcache.dirty",),
                theta=Theta((Atom("before", "store", "flush"),)),
                omega=Obligation("OM_DIRTY_SECURED"))
    base.update(kw)
    return Relation(**base)


def _trace(store_c=10, flush_c=20, extra=None):
    evs = [Event("store", "line:0x40", cycle=store_c),
           Event("flush", "line:0x40", cycle=flush_c)]
    evs += extra or []
    sts = [StateObs("dcache.dirty", "line:0x40", store_c, 1)]
    return Trace(target="ventus", testcase_id="t", events=evs, states=sts, cycles=100)


class TestMatching(unittest.TestCase):
    def test_activation(self):
        acts = activate(_rel(), _trace())
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0].obj, "line:0x40")

    def test_order_matters(self):
        self.assertEqual(activate(_rel(), _trace(store_c=30, flush_c=20)), [])

    def test_different_objects_do_not_activate(self):
        tr = _trace()
        tr.events[1].u = "line:0x80"
        self.assertEqual(activate(_rel(), tr), [])

    def test_same_cycle_overlap(self):
        rel = _rel(theta=Theta((Atom("same_cycle", "store", "flush"),)))
        self.assertEqual(len(activate(rel, _trace(store_c=20, flush_c=20))), 1)

    def test_optional_role(self):
        objv = ObjVar("L", "line")
        rel = _rel(roles=(EventRole("store", ("store",), objv),
                          EventRole("flush", ("flush",), objv),
                          EventRole("wb", ("wb_pending_set",), objv, optional=True)),
                   theta=Theta((Atom("before", "store", "flush"),)))
        acts = activate(rel, _trace())
        self.assertEqual(len(acts), 1)
        self.assertIsNone(acts[0].events["wb"])

    def test_unjustified_relation_is_not_checkable(self):
        rel = _rel(omega=BOT)
        self.assertFalse(rel.checkable)


class TestRelationValidation(unittest.TestCase):
    def test_unknown_theta_role(self):
        with self.assertRaises(ValueError):
            _rel(theta=Theta((Atom("before", "store", "nope"),)))

    def test_unknown_state(self):
        with self.assertRaises(ValueError):
            _rel(sids=("nope.state",))


if __name__ == "__main__":
    unittest.main()
