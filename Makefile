# GPUFuzz: ESDModel-guided GPGPU fuzzing with a Verilator adaptation layer
#
# Everything here is stdlib Python.  The targets themselves (RTL + Verilator +
# toolchains) are external; see README.md for what runs where.

PY      ?= python3
WORK    ?= work
RUNS    ?= runs
TARGET  ?= ventus
RTL     ?= $(HOME)/gpu
BUDGET  ?= 200

.PHONY: help model gen asm doctor build discover fuzz fuzz-fixture selftest test \
        dump demo clean all targets

help:
	@echo "GPUFuzz -- make targets"
	@echo "  make model                  build and show the ESDModel for all six GPGPUs"
	@echo "  make gen                    synthesize seed test cases (Ventus) + target assembly"
	@echo "  make asm                    show one generated program and its assembly"
	@echo "  make doctor                 per-target readiness matrix"
	@echo "  make test                   unit tests (no RTL needed)"
	@echo "  make selftest               oracle regression + campaign smoke test (no RTL)"
	@echo "  make demo                   the whole offline flow in one go"
	@echo "  make dump                   write a trace/testcase pair for inspection"
	@echo "  make build     RTL=<dir>    build the Verilator simulator for TARGET"
	@echo "  make discover  RTL=<dir>    resolve signal-map probes against a checkout"
	@echo "  make fuzz      RTL=<dir>    real campaign on RTL (needs build first)"
	@echo "  make fuzz-fixture           campaign on synthetic traces (no RTL)"
	@echo "  make clean                  remove build/run artifacts"

targets:
	@$(PY) cli.py model

model:
	$(PY) cli.py model --verbose

gen:
	$(PY) cli.py generate --target $(TARGET) --emit-asm --out generated/$(TARGET)

asm:
	$(PY) cli.py asm --target $(TARGET)

doctor:
	$(PY) cli.py doctor --rtl $(RTL)

test:
	$(PY) -m unittest discover -s tests/unit -t . -v

selftest:
	$(PY) cli.py selftest --out $(RUNS)/selftest

dump:
	$(PY) cli.py dump --target $(TARGET) --out $(RUNS)/dump
	@echo "now: $(PY) cli.py oracle --target $(TARGET) --trace $(RUNS)/dump/*.trace.json --testcase $(RUNS)/dump/*.testcase.json"

demo: test
	@$(PY) cli.py model
	@echo
	@$(PY) cli.py generate --target ventus --emit-asm --out generated/ventus >/dev/null
	@echo "generated seed pool -> generated/ventus/"
	@$(PY) cli.py selftest --out $(RUNS)/selftest
	@echo
	@$(PY) cli.py dump --target ventus --out $(RUNS)/demo
	@$(PY) cli.py oracle --target ventus --trace $(RUNS)/demo/*.trace.json \
	    --testcase $(RUNS)/demo/*.testcase.json | head -3
	@echo
	@$(PY) cli.py doctor || true

build:
	$(PY) cli.py build --target $(TARGET) --rtl $(RTL)

discover:
	$(PY) cli.py discover --target $(TARGET) --rtl $(RTL) --write

fuzz: build
	$(PY) cli.py fuzz --target $(TARGET) --driver verilator --auto-build \
	    --budget $(BUDGET) --out $(RUNS) --rtl $(RTL)

fuzz-fixture:
	$(PY) cli.py fuzz --target $(TARGET) --driver fixture --budget $(BUDGET) --out $(RUNS)

clean:
	rm -rf $(WORK) $(RUNS) generated __pycache__ */__pycache__ */*/__pycache__ \
	       */*/*/__pycache__
