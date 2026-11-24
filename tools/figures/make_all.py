"""Regenerate every figure under ``assets/`` from runs of the engine in this repository.

    python tools/figures/make_all.py                  all figures (about half a minute on one core)
    python tools/figures/make_all.py hero loop        only the named ones

Needs matplotlib (``pip install -e ".[viz]"``). ``make figures`` calls this script.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES = ("hero", "episode", "gallery_sequences", "gallery_physics", "loop", "hidden_vs_received")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"figures to build (default: all of {', '.join(FIGURES)})")
    a = ap.parse_args(argv)
    unknown = [n for n in a.names if n not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    t0 = time.time()
    for name in a.names or FIGURES:
        t = time.time()
        status = importlib.import_module(name).main([])
        if status:
            return int(status)
        print(f"  {name}: {time.time() - t:.1f} s")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
