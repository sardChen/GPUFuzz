# GPUFuzz

Implementation of the paper **GPUFuzz: Uncovering State-Interaction
Inconsistencies in GPGPU Hardware**, split along the Design section:

| folder | role | paper |
|---|---|---|
| [`fuzzer/`](fuzzer) | builds the ESDModel, synthesizes `T=<K,C>`, dependency-preserving mutation, feedback-driven seed scheduling | §4.2 §4.3 / Alg. 1 |
| [`oracle/`](oracle) | collects `O=<A,E,S>`, functional + state-interaction consistency checking, reproduction/minimization/de-duplication | §4.4 / Fig. 7 |

Around them: `common/` (IR, configuration, `c/e/s/θ/ω`), `adapters/` (Verilator
harness generation, build/run drivers, trace adapter, coverage) and `specs/`
(the six GPGPUs: Vortex, Ventus, VeriGPU, MIAOW, Nyuzi, smol-gpu).

25 shapes → 62 relations; all 22 bugs of Table 2 are covered.

---

## Quick start (no RTL needed)

Python 3.8+, standard library only.

```bash
cd gpufuzz

python3 cli.py model                 # ESDModel for the six targets
python3 cli.py generate --target ventus --emit-asm --out generated/ventus
python3 cli.py asm --target ventus --shape DCACHE_FLUSH_DIRTY
python3 cli.py selftest              # oracle regression: clean traces silent, fault traces detected
python3 -m unittest discover -s tests/unit -t .    # unit tests
```

`selftest` is the useful one: for all 62 relations it checks "no defect → zero
false positives, defect → the matching obligation fires".

## Running on real RTL

Requires Verilator ≥ 5 and the target toolchain (see `common/isa/<target>.json`).

```bash
python3 cli.py discover --target ventus --rtl ~/gpu/ventus --write   # resolve signal paths
python3 cli.py doctor   --rtl ~/gpu                                  # what is still missing
python3 cli.py build    --target ventus --rtl ~/gpu/ventus           # build the generated harness
python3 cli.py fuzz     --target ventus --driver verilator --budget 2000 --out runs
python3 cli.py report   --run runs/ventus --messages
```

Coverage (interaction-sensitive regions only, as in §6.2):

```bash
python3 scripts/mark_regions.py --rtl ~/gpu/ventus --out specs/regions/ventus.txt
python3 cli.py coverage --cov work/ventus/build/coverage.dat --regions specs/regions/ventus.txt
```

To see how the oracle judges one test case:

```bash
python3 cli.py dump   --target ventus --shape SB_ISSUE_WB --out runs/dump
python3 cli.py oracle --target ventus --trace runs/dump/*.trace.json \
                      --testcase runs/dump/*.testcase.json
```

## Layout

```
cli.py              entry point for every command
fuzzer/             ESDModel construction, shape templates, generation, mutation, seed pool, campaign loop
oracle/             status collector, two detectors, 20 interaction rules, reference model, reproduction, reports
common/             IR, execution configuration, ESD types and matching, target specs, assembly emission
adapters/           Verilator adaptation layer (recipe + signal map -> generated harness), trace adapter, coverage
specs/              per-target caps and the Table 2 bug list
tests/              synthetic trace fixtures + unit tests
```

Useful make targets: `make demo`, `make selftest`, `make doctor`, `make test`.

## Limitations

- **No RTL is bundled.** The six checkouts, Verilator and the toolchains are
  external; `cli.py doctor` lists exactly what is missing.
- **Signal paths ship empty.** Hierarchical names cannot be derived from the
  paper — run `discover` on your checkout first (it writes a `.bak` and records
  the source location of every resolved signal).
- **27 of the 35 `GPU_*` macros per RISC-V target are `placeholder`/`medium`.**
  CSR numbers, texture/barrier/exit encodings and memory pins are design
  specific; replace them following the comments.
- **Nyuzi and MIAOW are still `dialect: opaque`** — their assembler dialects must
  be supplied; `doctor` reports this as a blocking problem.
- **`--driver fixture` is not a DUT.** It synthesizes traces to regression-test
  the oracle and must not be used for bug counts or coverage claims.
- `specs/*.json` caps are the projects' documented defaults — adjust them to your
  elaboration; the `verify` field lists the parameters to check.

## Paper mapping

`§4.2` → `common/types.py`, `common/esd/`, `fuzzer/esd_model.py`; `§4.3` →
`fuzzer/generator.py`, `mutator.py`, `seedpool.py`, `loop.py`; `§4.4` →
`oracle/`; `§5 Fig. 8` → `adapters/`; `Table 1/2` → `specs/*.json`.
