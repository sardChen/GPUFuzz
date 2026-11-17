#!/usr/bin/env python3
"""GPUFuzz command line interface.

    python3 cli.py model      --target ventus          # build the ESDModel
    python3 cli.py generate   --target ventus          # synthesize test cases (.S + .json)
    python3 cli.py asm        --target ventus          # show one generated program
    python3 cli.py build      --target ventus --rtl ~/gpu/ventus
    python3 cli.py discover   --target ventus --rtl ~/gpu/ventus --write
    python3 cli.py doctor     --rtl ~/gpu
    python3 cli.py fuzz       --target ventus --budget 200 --driver verilator
    python3 cli.py oracle     --target ventus --trace t.jsonl --testcase tc.json
    python3 cli.py selftest
    python3 cli.py coverage   --cov coverage.dat --regions specs/regions/ventus.txt
    python3 cli.py report     --run runs/ventus

The only backend that needs no RTL is ``--driver fixture`` (synthetic traces, for
oracle regression tests).  Real campaigns use ``--driver verilator``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.driver import create_driver
from common.targets import TARGETS, load_target
from common.types import TestCase
from common.util import Log, ensure_dir, read_json, table, write_json
from fuzzer.esd_model import build_model


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _log(args) -> Log:
    return Log(getattr(args, "log_level", "info"))


def _model_for(target: str, use_rtl: bool, log: Log):
    model, report = build_model(target, use_rtl=use_rtl, log=log)
    return model, report


def _driver_for(args, spec, log: Log):
    kind = getattr(args, "driver", "fixture")
    if kind == "verilator":
        return create_driver("verilator", spec, workdir=args.work, rtl=args.rtl,
                             sim=getattr(args, "sim", None),
                             auto_build=getattr(args, "auto_build", False),
                             dry_run=getattr(args, "dry_run", False), log=log)
    if kind == "fixture":
        from tests.fixtures.driver import FixtureDriver
        return FixtureDriver(spec, clean=getattr(args, "clean", False), 
                             keep_regs=getattr(args, "keep_regs", False))
    raise SystemExit("unknown driver %r" % kind)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_model(args) -> int:
    log = _log(args)
    targets = [args.target] if args.target else list(TARGETS)
    rows = []
    for target in targets:
        model, rep = _model_for(target, not args.no_rtl, log)
        rows.append([target, str(len(model)), str(len(model.checkable())),
                     str(sum(1 for r in model if r.source == "runtime")),
                     str(len(rep.unobservable))])
        if args.out:
            path = os.path.join(ensure_dir(args.out), "esdmodel-%s.json" % target)
            model.save(path)
            rows[-1].append(path)
    print(table(rows, ["target", "relations", "checkable", "provisional", "unobservable"]))
    if args.verbose:
        for target in targets:
            model, _ = _model_for(target, not args.no_rtl, log)
            print("\n== %s ==" % target)
            print(model.summary())
    return 0


def cmd_generate(args) -> int:
    log = _log(args)
    from fuzzer.generator import GenerationError, TestGenerator
    from common.isa.emit import write_assembly

    model, rep = _model_for(args.target, not args.no_rtl, log)
    spec = load_target(args.target)
    gen = TestGenerator(spec, model, random.Random(args.seed), log=log)
    outdir = ensure_dir(args.out or os.path.join("generated", args.target))
    seeds = gen.initial_seeds(per_relation=args.per_relation, shapes=args.shape)
    rows = []
    for tc in seeds:
        path = os.path.join(outdir, tc.id + ".json")
        write_json(path, tc.to_json())
        emitted = ""
        if args.emit_asm:
            try:
                paths = write_assembly(tc.program, args.target, os.path.join(outdir, tc.id))
                emitted = paths["asm"]
                if paths.get("unsupported"):
                    emitted += " (unsupported: %s)" % paths["unsupported"]
            except Exception as exc:
                emitted = "asm failed: %s" % exc
        rows.append([tc.id, tc.rid or "", str(len(tc.program)),
                     tc.config.describe()[:46], os.path.basename(path), emitted])
    print(table(rows, ["testcase", "relation", "insns", "configuration", "json", "asm"]))
    print("\n%d test cases -> %s" % (len(seeds), outdir))
    if gen.stats.problems:
        print("generation problems: %s" % json.dumps(gen.stats.problems, sort_keys=True))
    return 0


def cmd_asm(args) -> int:
    log = _log(args)
    from fuzzer.generator import TestGenerator
    from common.isa.emit import write_assembly, emit_program

    model, _ = _model_for(args.target, not args.no_rtl, log)
    spec = load_target(args.target)
    gen = TestGenerator(spec, model, random.Random(args.seed), log=log)
    rels = [r for r in model if r.target == args.target]
    if args.shape:
        rels = [r for r in rels if r.shape == args.shape]
    if args.rid:
        rels = [r for r in rels if r.rid == args.rid]
    if not rels:
        raise SystemExit("no relation matches")
    tc = gen.synthesize(rels[0])
    print("# relation: %s" % rels[0].describe())
    print("# program:")
    print(tc.program.text())
    print()
    res = emit_program(tc.program, args.target)
    print(res.asm)
    if res.unsupported:
        print("# UNSUPPORTED: %s" % "; ".join(res.unsupported))
    if args.out:
        paths = write_assembly(tc.program, args.target, args.out)
        print("# written: %s" % paths["asm"])
    return 0


def cmd_build(args) -> int:
    log = _log(args)
    if not args.rtl:
        raise SystemExit("build needs --rtl <checkout>")
    from adapters.verilator import build as builder
    outdir = os.path.join(args.work, args.target, "build")
    res = builder.build(args.target, args.rtl, outdir, dry_run=args.dry_run, log=log)
    print(json.dumps(res.to_json(), indent=2)[:4000])
    return 0 if res.ok else 1


def cmd_discover(args) -> int:
    log = _log(args)
    if not args.rtl:
        raise SystemExit("discover needs --rtl <checkout>")
    from adapters import discover
    res = discover.discover(args.target, args.rtl, write=args.write, log=log)
    print(discover.SignalDiscovery.render(res))
    return 0 if not res.get("unresolved") else 1


def cmd_doctor(args) -> int:
    from adapters import doctor
    res = doctor.doctor(workdir=args.work, rtl=args.rtl,
                        targets=[args.target] if args.target else None)
    print(doctor.render(res))
    if args.json:
        print(json.dumps(res, indent=2))
    # non-strict by default so `doctor` can sit in a pipeline; use --strict in CI
    return 1 if (args.strict and res["summary"]["blocked"]) else 0


def cmd_fuzz(args) -> int:
    log = _log(args)
    from fuzzer.loop import Campaign, CampaignConfig
    model, rep = _model_for(args.target, not args.no_rtl, log)
    spec = load_target(args.target)
    driver = _driver_for(args, spec, log)
    if args.budget_mode == "iters" and args.iterations:
        budget = args.iterations
    else:
        budget = args.budget
    cfg = CampaignConfig(target=args.target, budget=budget, budget_mode=args.budget_mode,
                         outdir=args.out, reproduce=not args.no_repro,
                         repro_runs=args.repro_runs, minimize=not args.no_minimize,
                         seed=args.seed, log_every=args.log_every,
                         shapes=args.shape, seeds_per_relation=args.per_relation)
    campaign = Campaign(spec, model, driver, cfg, log=log)
    res = campaign.run()
    print(json.dumps(res.summary(), indent=2))
    if res.bugs:
        print()
        print(campaign.reporter.table())
    if res.notes:
        print("\nnotes:\n  " + "\n  ".join(res.notes))
    return 0 if res.iterations > 0 else 1


def cmd_oracle(args) -> int:
    log = _log(args)
    from adapters.trace.adapter import load_adapter
    from oracle.oracle import ConsistencyOracle
    from common.types import Trace

    spec = load_target(args.target)
    model, _ = _model_for(args.target, not args.no_rtl, log)
    tc = TestCase.from_json(read_json(args.testcase))
    trace = _load_trace(args.trace, args.target, spec, tc.id, log)
    oracle = ConsistencyOracle(spec, model, log=log)
    result = oracle.check(trace, tc)
    print(result.describe())
    for v in result.violations:
        print("  " + v.describe())
        print("    evidence: " + json.dumps(v.evidence, default=str)[:600])
    if args.json:
        print(json.dumps([v.to_json() for v in result.violations], indent=2))
    print("\nobservation: " + json.dumps(oracle.collector.report(), indent=2)[:1200])
    return 0 if result.clean else 2


def _load_trace(path: str, target: str, spec, tc_id: str, log: Log):
    """Accept either a raw harness JSONL trace or a serialized Trace JSON."""
    from adapters.trace.adapter import load_adapter
    from common.types import Trace
    try:
        whole = read_json(path)
    except Exception:
        whole = None
    if isinstance(whole, dict) and "events" in whole and "target" in whole:
        return Trace.from_json(whole)
    return load_adapter(target, spec.signals_path, log=log).parse(path, tc_id)


def cmd_dump(args) -> int:
    """Write a recorded trace + test case pair (fixture backend) for inspection."""
    log = _log(args)
    from fuzzer.generator import TestGenerator
    from tests.fixtures.driver import FixtureDriver

    spec = load_target(args.target)
    model, _ = _model_for(args.target, not args.no_rtl, log)
    gen = TestGenerator(spec, model, random.Random(args.seed), log=log)
    rels = [r for r in model if r.target == args.target]
    if args.shape:
        rels = [r for r in rels if r.shape == args.shape]
    if not rels:
        raise SystemExit("no relation matches")
    tc = gen.synthesize(rels[0])
    drv = FixtureDriver(spec, clean=args.clean)
    res = drv.execute(tc)
    if not res.ok:
        raise SystemExit("fixture driver failed: %s" % res.error)
    outdir = ensure_dir(args.out)
    tpath = os.path.join(outdir, tc.id + ".trace.json")
    tcpath = os.path.join(outdir, tc.id + ".testcase.json")
    write_json(tpath, res.trace.to_json())
    write_json(tcpath, tc.to_json())
    print("trace:    %s" % tpath)
    print("testcase: %s" % tcpath)
    print("next:     python3 cli.py oracle --target %s --trace %s --testcase %s"
          % (args.target, tpath, tcpath))
    return 0


def cmd_selftest(args) -> int:
    log = Log("warn")
    from oracle import ConsistencyOracle
    from fuzzer.generator import TestGenerator
    from tests.fixtures.driver import FixtureDriver, expected_obligation

    targets = [args.target] if args.target else list(TARGETS)
    failures: List[str] = []
    total = 0
    rows = []
    for target in targets:
        spec = load_target(target)
        model, _ = _model_for(target, False, log)
        gen = TestGenerator(spec, model, random.Random(1), log=log)
        oracle = ConsistencyOracle(spec, model, log=log)
        clean = FixtureDriver(spec, clean=True)
        faulty = FixtureDriver(spec)
        ok = 0
        rels = [r for r in model if r.target == target]
        for rel in rels:
            total += 1
            tc = gen.synthesize(rel)
            o_clean = oracle.check(clean.execute(tc).trace, tc)
            o_fault = oracle.check(faulty.execute(tc).trace, tc)
            want = expected_obligation(tc)
            got = {v.oid for v in o_fault.violations}
            if o_clean.violations:
                failures.append("%s/%s: %d false positive(s) on a clean trace"
                                % (target, rel.shape, len(o_clean.violations)))
            elif want and want not in got:
                failures.append("%s/%s: expected %s, got %s"
                                % (target, rel.shape, want, sorted(got) or "nothing"))
            else:
                ok += 1
        rows.append([target, str(len(rels)), str(ok)])

    # end-to-end campaign smoke test on the fixture backend
    from fuzzer.loop import Campaign, CampaignConfig
    smoke: List[List[str]] = []
    for target in targets[: args.smoke_targets]:
        spec = load_target(target)
        model, _ = _model_for(target, False, log)
        cfg = CampaignConfig(target=target, budget=args.smoke_iters, outdir=args.out,
                             log_every=10 ** 9, seed=3, repro_runs=2, minimize=True)
        res = Campaign(spec, model, FixtureDriver(spec), cfg, log=log).run()
        smoke.append([target, str(res.iterations), str(len(res.bugs)),
                      str(res.interactions), "%.1fs" % res.elapsed])
        if res.iterations == 0:
            failures.append("%s: campaign did not run" % target)

    print(table(rows, ["target", "relations", "oracle checks passed"]))
    print()
    print(table(smoke, ["target", "iterations", "bug candidates", "interactions", "time"]))
    print()
    if failures:
        print("%d FAILURE(S):" % len(failures))
        for f in failures[:20]:
            print("  - " + f)
        return 1
    print("selftest OK: %d relations, %d targets" % (total, len(targets)))
    return 0


def cmd_coverage(args) -> int:
    from adapters import coverage
    regions = coverage.RegionSet.load(args.regions)
    res = coverage.interaction_coverage(args.cov, regions, hit_only=not args.include_zero)
    print(json.dumps({k: v for k, v in res.items() if k != "points"}, indent=2))
    if args.list_points:
        for p in res["points"][: args.limit]:
            print(p)
    return 0


def cmd_report(args) -> int:
    from oracle.report import Reporter
    path = os.path.join(args.run, "bugs.json")
    data = read_json(path)
    if data is None:
        print("no report at %s" % path)
        return 1
    print(json.dumps(data.get("summary"), indent=2))
    rows = []
    for r in data.get("reports", []):
        rows.append([r["report_id"], r["target"], r["oid"], r["family"],
                     str(r["occurrences"]), r["status"]])
    print(table(rows, ["id", "target", "obligation", "family", "hits", "status"]))
    if args.messages:
        for r in data.get("reports", []):
            print("\n%s: %s" % (r["report_id"], r["message"]))
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    # accept the global options on either side of the subcommand; the copies on
    # the subparsers use SUPPRESS so an explicit prefix value is not overridden
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=argparse.SUPPRESS,
                        choices=["quiet", "error", "warn", "info", "debug", "trace"])
    common.add_argument("--work", default=argparse.SUPPRESS,
                        help="build/run working directory")
    common.add_argument("--rtl", default=argparse.SUPPRESS,
                        help="root directory containing the six RTL checkouts")

    p = argparse.ArgumentParser(prog="gpufuzz", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="info",
                   choices=["quiet", "error", "warn", "info", "debug", "trace"])
    p.add_argument("--work", default="work", help="build/run working directory")
    p.add_argument("--rtl", default=None, help="root directory containing the six RTL checkouts")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("model", help="build and show the ESDModel", parents=[common])
    m.add_argument("--target")
    m.add_argument("--no-rtl", action="store_true")
    m.add_argument("--out")
    m.add_argument("--verbose", action="store_true")
    m.set_defaults(func=cmd_model)

    g = sub.add_parser("generate", parents=[common], help="synthesize seed test cases")
    g.add_argument("--target", required=True)
    g.add_argument("--out")
    g.add_argument("--per-relation", type=int, default=1)
    g.add_argument("--shape", action="append")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-rtl", action="store_true")
    g.add_argument("--emit-asm", action="store_true")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("asm", parents=[common], help="show the generated program for one relation")
    a.add_argument("--target", required=True)
    a.add_argument("--shape")
    a.add_argument("--rid")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out")
    a.add_argument("--no-rtl", action="store_true")
    a.set_defaults(func=cmd_asm)

    b = sub.add_parser("build", parents=[common], help="build the Verilator simulator for a target")
    b.add_argument("--target", required=True)
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("discover", parents=[common], help="resolve signal-map probes against RTL")
    d.add_argument("--target", required=True)
    d.add_argument("--write", action="store_true")
    d.set_defaults(func=cmd_discover)

    doc = sub.add_parser("doctor", parents=[common], help="print the per-target readiness matrix")
    doc.add_argument("--target")
    doc.add_argument("--json", action="store_true")
    doc.add_argument("--strict", action="store_true",
                     help="exit non-zero when a target is blocked")
    doc.set_defaults(func=cmd_doctor)

    f = sub.add_parser("fuzz", parents=[common], help="run a fuzzing campaign")
    f.add_argument("--target", required=True)
    f.add_argument("--driver", default="fixture", choices=["verilator", "fixture"])
    f.add_argument("--budget", type=float, default=200)
    f.add_argument("--budget-mode", default="iters", choices=["iters", "time"])
    f.add_argument("--iterations", type=int, default=0)
    f.add_argument("--out", default="runs")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--per-relation", type=int, default=1)
    f.add_argument("--shape", action="append")
    f.add_argument("--no-rtl", action="store_true")
    f.add_argument("--no-repro", action="store_true")
    f.add_argument("--no-minimize", action="store_true")
    f.add_argument("--repro-runs", type=int, default=3)
    f.add_argument("--log-every", type=int, default=25)
    f.add_argument("--auto-build", action="store_true")
    f.add_argument("--sim")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--clean", action="store_true",
                   help="fixture driver: emit clean traces (false positive check)")
    f.add_argument("--keep-regs", action="store_true",
                   help="fixture driver: keep the register dump")
    f.set_defaults(func=cmd_fuzz)

    o = sub.add_parser("oracle", parents=[common], help="run the oracle on a recorded trace")
    o.add_argument("--target", required=True)
    o.add_argument("--trace", required=True)
    o.add_argument("--testcase", required=True)
    o.add_argument("--json", action="store_true")
    o.add_argument("--no-rtl", action="store_true")
    o.set_defaults(func=cmd_oracle)

    du = sub.add_parser("dump", parents=[common], help="write a recorded trace + testcase pair (fixture backend)")
    du.add_argument("--target", required=True)
    du.add_argument("--shape")
    du.add_argument("--out", default="runs/dump")
    du.add_argument("--seed", type=int, default=0)
    du.add_argument("--clean", action="store_true")
    du.add_argument("--no-rtl", action="store_true")
    du.set_defaults(func=cmd_dump)

    s = sub.add_parser("selftest", parents=[common], help="oracle regression + campaign smoke test")
    s.add_argument("--target")
    s.add_argument("--out", default="runs/selftest")
    s.add_argument("--smoke-iters", type=int, default=20)
    s.add_argument("--smoke-targets", type=int, default=2)
    s.set_defaults(func=cmd_selftest)

    c = sub.add_parser("coverage", parents=[common], help="interaction coverage from Verilator coverage.dat")
    c.add_argument("--cov", action="append", required=True)
    c.add_argument("--regions", required=True)
    c.add_argument("--include-zero", action="store_true")
    c.add_argument("--list-points", action="store_true")
    c.add_argument("--limit", type=int, default=40)
    c.set_defaults(func=cmd_coverage)

    r = sub.add_parser("report", parents=[common], help="render a campaign report")
    r.add_argument("--run", required=True)
    r.add_argument("--messages", action="store_true")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:                             # pragma: no cover
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "log_level", "info") in ("debug", "trace"):
            traceback.print_exc()
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
