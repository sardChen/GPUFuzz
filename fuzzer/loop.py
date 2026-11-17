"""The fuzzing campaign: Algorithm 1 of the paper, line by line.

    1  B <- empty; Freq <- empty
    2  seedPool <- initialSeeds(M)
    3  while testing budget remains do
    4    r <- selectTarget(M, Freq)
    5    T <- selectSeed(seedPool, r)
    6    T' <- dependencyMutate(T, r)
    7    O, Ncov <- GPU.execute(T')
    8    X <- interactionFeedback(O, M)
    9    Delta <- checkConsistency(O, T', M)
    10   p <- calculatePriority(X, Ncov, Delta, Freq)
    11   if interesting(X, Ncov, Delta, Freq) then seedPool.enqueue(T', p)
    12   Freq.update(X)
    13   if Delta != empty then B.add(<T', Delta>)
    14   if Delta == empty then M.refine(O, X)

The campaign owns no target specific code: the driver supplies ``O`` (a trace),
the oracle supplies ``Delta``, the model supplies ``X``.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from common.driver import Driver
from common.esd.model import ESDModel, RefinementReport
from common.targets import TargetSpec
from common.types import RunResult, TestCase, Trace
from common.util import Budget, Log, ensure_dir, write_json
from fuzzer.generator import GenerationError, TestGenerator
from fuzzer.mutator import Mutator
from fuzzer.seedpool import Priority, SeedPool, is_interesting
from oracle.oracle import ConsistencyOracle, OracleResult
from oracle.report import BugReport, Reporter
from oracle.reproduce import BugReproducer


@dataclass
class CampaignConfig:
    target: str
    budget: int = 200
    budget_mode: str = "iters"          # iters | time
    seeds_per_relation: int = 1
    shapes: Optional[List[str]] = None
    mutate_rounds: int = 1
    outdir: str = "."
    reproduce: bool = True
    repro_runs: int = 3
    minimize: bool = True
    seed: int = 0
    log_every: int = 25
    max_failures: int = 5


@dataclass
class CampaignResult:
    target: str
    iterations: int = 0
    seeds: int = 0
    elapsed: float = 0.0
    bugs: List[BugReport] = field(default_factory=list)
    pool: Dict[str, Any] = field(default_factory=dict)
    interactions: int = 0
    coverage: int = 0
    freq_items: int = 0
    model_stats: Dict[str, Any] = field(default_factory=dict)
    failures: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "target": self.target, "iterations": self.iterations, "seeds": self.seeds,
            "elapsed_s": round(self.elapsed, 3),
            "iterations_per_s": round(self.iterations / self.elapsed, 3) if self.elapsed else 0,
            "bugs": len(self.bugs), "interactions": self.interactions,
            "coverage_points": self.coverage, "freq_items": self.freq_items,
            "exhausted_failures": self.failures, "notes": self.notes,
            "pool": self.pool, "model": self.model_stats,
        }


class Campaign:
    def __init__(self, spec: TargetSpec, model: ESDModel, driver: Driver,
                 cfg: CampaignConfig, log: Optional[Log] = None) -> None:
        self.spec = spec
        self.model = model
        self.driver = driver
        self.cfg = cfg
        self.log = log or Log("info")
        self.rng = random.Random(cfg.seed or 0xC0FFEE)
        self.generator = TestGenerator(spec, model, random.Random(self.rng.getrandbits(32)),
                                       log=self.log)
        self.mutator = Mutator(spec, random.Random(self.rng.getrandbits(32)), log=self.log)
        self.pool = SeedPool()
        self.oracle = ConsistencyOracle(spec, model, log=self.log)
        self.reporter = Reporter(ensure_dir(os.path.join(cfg.outdir, cfg.target)), log=self.log)
        self.reproducer = BugReproducer(driver, self.oracle, model, log=self.log)
        self.budget = Budget(cfg.budget, cfg.budget_mode)
        self.covered: set = set()
        self.interaction_points: set = set()
        self.failures = 0
        self.notes: List[str] = []

    # ------------------------------------------------------------------ #
    def initial_seeds(self) -> List[TestCase]:
        return self.generator.initial_seeds(per_relation=self.cfg.seeds_per_relation,
                                            shapes=self.cfg.shapes)

    def run(self) -> CampaignResult:
        t0 = time.time()
        res = CampaignResult(target=self.spec.name)
        seeds = self.initial_seeds()
        res.seeds = len(seeds)
        for tc in seeds:
            self.pool.enqueue(tc, Priority())
        if not seeds:
            res.notes.append("no seed could be synthesized for this target")
            res.elapsed = time.time() - t0
            return res

        it = 0
        while not self.budget.exhausted():
            it += 1
            self.budget.tick(1)
            r = self.model.select_target(self.rng, target=self.spec.name)
            tc = self.pool.select(r, self.rng)
            mutant = self.mutator.mutate(tc, r, rounds=self.cfg.mutate_rounds) or tc

            run: RunResult = self.driver.execute(mutant)
            if not run.ok or run.trace is None:
                self.failures += 1
                if self.failures % 10 == 1:
                    self.log.warn("driver failure: %s" % (run.error or "unknown"))
                if self.failures >= self.cfg.max_failures and it <= self.cfg.max_failures + 1:
                    res.notes.append(
                        "driver %r failed %d times (%s); is the DUT built? "
                        "see `cli.py doctor`" % (self.driver.name, self.failures, run.error))
                    break
                continue
            trace: Trace = run.trace

            acts = self.model.activations(trace, r)
            x = self.model.feedback_items(trace, r)
            delta = self.oracle.check(trace, mutant, activations=acts)

            n_new = sum(1 for k in x if self.model.freq.get(k, 0) == 0)
            new_cov = len(set(trace.cov) - self.covered)
            self.covered |= set(trace.cov)
            self.interaction_points |= self.model.interaction_points(trace)
            prio = Priority(violation=bool(delta.violations), new_interactions=n_new,
                            rarity=self.model.rarity(x), new_cov=new_cov)

            if is_interesting(prio):
                self.pool.enqueue(mutant, prio)
            self.model.update_freq(x)

            if delta.violations:
                self._record(mutant, delta)
            else:
                self.model.refine(trace, r.rid, acts, clean=True)

            if it % self.cfg.log_every == 0:
                self.log.info("%s it=%d pool=%d bugs=%d interactions=%d"
                              % (self.spec.name, it, len(self.pool),
                                 len(self.reporter.reports()), len(self.interaction_points)))

        res.iterations = it
        res.elapsed = time.time() - t0
        res.bugs = self.reporter.reports()
        res.pool = self.pool.stats()
        res.interactions = len(self.interaction_points)
        res.coverage = len(self.covered)
        res.freq_items = len(self.model.freq)
        res.model_stats = self.model.stats()
        res.failures = self.failures
        res.notes.extend(self.notes)
        self.reporter.write({"campaign": res.summary(),
                             "driver": self.driver.describe(),
                             "budget": self.cfg.budget})
        return res

    # ------------------------------------------------------------------ #
    def _record(self, tc: TestCase, delta: OracleResult) -> None:
        known = {r.signature: r for r in self.reporter.reports()}
        for v in delta.violations:
            repro = None
            already = known.get(v.signature())
            if self.cfg.reproduce and not (already and already.repro is not None
                                           and already.repro.reproduced):
                repro = self.reproducer.verify(tc, v.signature(),
                                               runs=self.cfg.repro_runs,
                                               minimize=self.cfg.minimize)
            rep = self.reporter.add(v, tc, repro)
            self.model.penalty(v.rid)
            self.log.info("bug candidate %s: %s" % (rep.report_id, v.describe()[:110]))
