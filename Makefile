# MR Gym developer commands. `make` or `make help` lists them.
PYTHON ?= python
MRGYM ?= mrgym
IDS = $(shell $(MRGYM) list --json | $(PYTHON) -c "import json, sys; print(' '.join(r['id'] for r in json.load(sys.stdin)))")

export OMP_NUM_THREADS ?= 1

.DEFAULT_GOAL := help
.PHONY: help setup test lint generate check validate figures conformance audit mutation handmutants golden

help:
	@echo "make setup      install the package (editable) and the developer tools"
	@echo "make test       run the tests"
	@echo "make lint       ruff check"
	@echo "make generate   regenerate the environment cards under envs/ and the README tables"
	@echo "make check      fail if a generated file or the results table is stale, or if an id fails its gate"
	@echo "make validate   physics checks against analytic formulas, as a table"
	@echo "make figures    redraw the figures (needs matplotlib)"
	@echo "make conformance  every validator, then the 11-theme scorecard (exit 1 below 90 %); DOCKER=1 / ONLINE=1 add what needs them"
	@echo "make audit      every check behind the tests of the reward and the scanner, one report (audit_report.json);"
	@echo "                needs: pip install -r tools/ci/requirements-audit.txt. About 25 minutes on 2 cores."
	@echo "make mutation   the mutation gate alone (score >= 85 %, under 10 minutes)"
	@echo "make handmutants  the 16 review mutants and 3 constant mutants, each against the three test groups"
	@echo "make golden     check the golden files (regenerate: python tools/audit/golden.py --write)"

setup:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install "pytest>=8" "ruff>=0.15" "matplotlib>=3.7"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check

generate:
	$(PYTHON) tools/generate.py

check:
	$(PYTHON) tools/generate.py --check
	$(MRGYM) results --check
	@for id in $(IDS); do echo "mrgym check $$id"; $(MRGYM) check $$id || exit 1; done

validate:
	$(MRGYM) validate

figures:
	$(PYTHON) tools/figures/make_all.py

# Each framework's validator runs in its own environment; CONF_GYM, CONF_VF, CONF_INSPECT, CONF_OPENENV and CONF_HARBOR
# override the default throw-away `uv run --isolated` environments (see conformance/README.md).
conformance:
	$(PYTHON) conformance/run.py $(if $(DOCKER),--docker) $(if $(ONLINE),--online)

# ---- test strength (tools/audit/). `make audit` exits 0 only if every row of the report passes.
audit:
	$(PYTHON) tools/audit/report.py

mutation:
	$(PYTHON) tools/audit/mutation.py

handmutants:
	$(PYTHON) tools/audit/hand_mutants.py

golden:
	$(PYTHON) tools/audit/golden.py --check
