# MR Gym developer commands. `make` or `make help` lists them.
PYTHON ?= python
MRGYM ?= mrgym
IDS = Scanner/Explore-v0 Scanner/NoWrap-v0 Scanner/T1Transmit-v0

export OMP_NUM_THREADS ?= 1

.DEFAULT_GOAL := help
.PHONY: help setup test lint generate check validate figures

help:
	@echo "make setup      install the package (editable) and the developer tools"
	@echo "make test       run the tests"
	@echo "make lint       ruff check"
	@echo "make generate   regenerate the environment cards under envs/ and the README tables"
	@echo "make check      fail if a generated file or the results table is stale, or if an id fails its gate"
	@echo "make validate   physics checks against analytic formulas, as a table"
	@echo "make figures    redraw the figures (needs matplotlib)"

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
