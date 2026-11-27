"""Write or check the golden files under ``tests/golden/`` (what they hold: ``tests/golden_cases.py``).

    python tools/audit/golden.py --check     compare the tree with the files; exit 1 on any difference
    python tools/audit/golden.py --write     regenerate both files

``--write`` after an engine-version bump is the one command that brings the golden data up to date. It refuses to
change a number of ``engine_digests.json`` while ``ENGINE_VERSION`` is the one recorded in the file: scan data that
moves under an unchanged engine version is the defect the file exists to catch (bump the version in
``src/mrgym/scanner/session.py`` first). Adding cases under the same engine is allowed. It also refuses to write from
a stack other than the recorded one unless ``--new-reference-stack`` is given.

Task snapshots (instances, statements, cards, constants, reference episodes) are rewritten by ``--write`` and the
changed ids are listed: a changed snapshot of a released id needs a new ``-vN`` (CONTRIBUTING.md).

Two more files are covered. ``volume_digests.json`` (the volume worlds) is rewritten by ``--write`` like the task
snapshots. ``compat_2d.json`` (the 2D baseline recorded on ``v1``) is compared in both modes and never written here:
it moves only by ``python tests/compat_cases.py --write [--rebase]``.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import golden_cases as G  # noqa: E402


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def changed(old: dict, new: dict) -> tuple[list, list, list]:
    """(keys whose value differs, keys only in new, keys only in old) of two nested dicts."""
    a, b = _flat(old), _flat(new)
    return (sorted(k for k in a if k in b and a[k] != b[k]), sorted(set(b) - set(a)), sorted(set(a) - set(b)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--new-reference-stack", action="store_true", help="record the digests of this numpy / scipy / machine")
    a = ap.parse_args(argv)
    if a.write == a.check:
        ap.error("give exactly one of --write, --check")

    old_e, old_t = G.load(G.ENGINE_FILE), G.load(G.TASK_FILE)
    new_e, new_t = G.engine_digests(), G.task_snapshots()
    status = 0

    # ---- engine digests
    numbers = lambda d: {k: d[k] for k in ("images", "noisy", "prices", "array", "core") if k in d} if d else {}       # noqa: E731
    diff, added, removed = changed(numbers(old_e), numbers(new_e))
    same_engine = bool(old_e) and old_e["engine"] == new_e["engine"]
    same_stack = bool(old_e) and old_e["stack"] == new_e["stack"]
    print(f"engine {new_e['engine']} (file: {old_e['engine'] if old_e else 'none'}), stack {new_e['stack']['numpy']} / "
          f"{new_e['stack']['scipy']} / {new_e['stack']['machine']}"
          f"{'' if same_stack or not old_e else ' (NOT the recorded stack)'}")
    print(f"engine_digests: {len(diff)} changed, {len(added)} new, {len(removed)} gone")
    for k in diff[:12]:
        print("  changed:", k)
    if a.check:
        if not old_e or diff or added or removed or not same_engine:
            status = 1
            print("engine_digests.json is not current" + ("" if same_engine else ": the engine version moved; run --write"))
    else:
        if old_e and not same_stack and not a.new_reference_stack:
            print("refused: this is not the stack the file was recorded on (give --new-reference-stack to move it)")
            return 1
        if same_engine and same_stack and (diff or removed):
            print(f"refused: scan data or prices changed while ENGINE_VERSION is still {new_e['engine']}. Bump it in "
                  "src/mrgym/scanner/session.py (and add an 'Engine' line to CHANGELOG.md), then run --write again.")
            return 1
        G.dump(G.ENGINE_FILE, new_e)
        print("wrote", os.path.relpath(G.ENGINE_FILE, ROOT))

    # ---- task snapshots
    diff, added, removed = changed(old_t or {}, new_t)
    ids = sorted({k.split(".")[1] for k in diff + added + removed if k.startswith("ids.")})
    print(f"task_snapshots: {len(diff)} changed, {len(added)} new, {len(removed)} gone" + (f"; ids touched: {', '.join(ids)}" if ids else ""))
    for k in diff[:20]:
        print("  changed:", k)
    if a.check:
        if diff or added or removed:
            status = 1
            print("task_snapshots.json is not current")
    else:
        G.dump(G.TASK_FILE, new_t)
        print("wrote", os.path.relpath(G.TASK_FILE, ROOT))

    # ---- volume worlds (an additive file: no scan, no id depends on it yet)
    old_v, new_v = G.load(G.VOLUME_FILE), G.volume_digests()
    diff, added, removed = changed((old_v or {}).get("worlds", {}), new_v["worlds"])
    print(f"volume_digests: {len(diff)} changed, {len(added)} new, {len(removed)} gone")
    for k in diff[:12]:
        print("  changed:", k)
    if a.check:
        if not old_v or diff or added or removed:
            status = 1
            print("volume_digests.json is not current")
    elif old_v and old_v["stack"] != new_v["stack"] and not a.new_reference_stack:
        print("refused: volume_digests.json was recorded on another stack (give --new-reference-stack to move it)")
        status = 1
    else:
        G.dump(G.VOLUME_FILE, new_v)
        print("wrote", os.path.relpath(G.VOLUME_FILE, ROOT))

    # ---- the 2D baseline: checked in both modes, never written here (python tests/compat_cases.py --write [--rebase])
    import compat_cases as C
    base = C.load()
    diff, added, removed = C.compare(base or {}, C.compute())
    print(f"compat_2d: {len(diff)} changed, {len(added)} new (not part of the baseline), {len(removed)} gone")
    for k in (diff + removed)[:12]:
        print("  changed:" if k in diff else "  gone:", k)
    if not base or diff or removed:
        status = 1
        print("a 2D exam is not what the baseline of tests/golden/compat_2d.json holds: see tests/compat_cases.py")
    return status


if __name__ == "__main__":
    sys.exit(main())
