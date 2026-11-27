"""The 2D baseline: what a 2D exam was before the third dimension arrived (``tests/golden/compat_2d.json``).

Volume worlds, receive arrays and the parameters that come with them (slices, orientation, acceleration, ...) are
added NEXT TO the 2D scanner. The rule of that work: for a 2D exam nothing moves by one bit.
This module computes what "nothing" covers, and ``tests/test_compat_2d_unchanged.py`` compares it with the values
stored on branch ``v1`` before the first 3D package (the ``recorded`` entry names the commit):

    worlds        every 2D world kind: sha256 of the class weights and of each field, of the tissue values, the system
                  state, the regions and the console card (and an 8 x 8 block fingerprint for other numerical stacks)
    exams         per kind: the protocol schema (full and brief) and the scanner card the agent reads
    scans         per kind and console sequence: the whole ``plan`` output, the header line, the noise-free image
    refusals      the ``plan`` output for every parameter of real consoles that a 2D exam refuses (``NOT_SIMULATED``):
                  for a 2D exam ``slices``, ``slice_gap_mm``, ``acceleration`` ... keep their refusal text
    golden        every entry of ``tests/golden/engine_digests.json`` (images, noisy, prices) as it was
    fingerprints  the fingerprint of every environment id as it was (``src/mrgym/compat/fingerprints.json``)

The file is a BASELINE, not a generated file that follows the tree: ``tools/audit/golden.py --write`` does not rewrite
it. It moves in two cases only, both deliberate, and both by this command:

    python tests/compat_cases.py --write            add entries; refused if a stored value would change
    python tests/compat_cases.py --write --rebase   a new baseline (after an engine-version bump, or when an id was
                                                    re-fingerprinted on purpose); the changed entries are listed

Entries may be added by later packages (a new id, a new golden case, a new key of a world); a stored entry must stay
equal. ``tools/audit/golden.py --check`` runs the same comparison (``make golden``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILE = os.path.join(HERE, "golden", "compat_2d.json")
SEED = "000102030405060708090a0b0c0d0e0f"
GRID = 320                                  # master grid of the cases (the engine is the same code at any grid)
KINDS_2D = ("brain", "abdomen", "qa", "acr", "nist_t1")          # the world kinds of v1, spelled out: a new kind is not a 2D kind
FIELD_T = 3.0                               # every kind exists at 3 T (the NIST plate at 3 T only)
PROTOCOLS = dict(                           # one protocol per console sequence (those of tests/golden_cases.py when this was recorded)
    spgr=dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, n_echoes=2, ro_polarity=-1),
    afi=dict(sequence="afi", tr_ms=25.0, flip_deg=55.0, matrix=64),
    se=dict(sequence="se", tr_ms=500.0, matrix=64, pe_dir="x", centre_x_mm=12.0, phase_fov=0.75),
    ir=dict(sequence="ir", tr_ms=1500.0, ti_ms=400.0, matrix=64, fat_sat=True),
    bssfp=dict(sequence="bssfp", tr_ms=8.0, flip_deg=30.0, matrix=64, phase_oversampling=0.25),
    epi=dict(sequence="epi", tr_ms=200.0, matrix=64, partial_fourier=0.75),
)
CARD_CFG = dict(budget_s=600.0, max_requests=24, task=dict(regions=[]))


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else list(o))


def sha(obj) -> str:
    """sha256 of a JSON value (canonical form) or of bytes."""
    return hashlib.sha256(obj if isinstance(obj, bytes) else canon(obj).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- one world kind
_ENGINE: dict = {}


def _engine(kind):
    from mrgym.scanner import kspace
    from mrgym.scanner import world as W
    if kind not in _ENGINE:
        _ENGINE.clear()                                                 # one world at a time
        _ENGINE[kind] = kspace.Engine(W.make_world(SEED, dict(kind=kind, b0_t=FIELD_T, n=GRID)))
    return _ENGINE[kind]


def world_case(kind) -> dict:
    """Digest of everything a 2D world holds. Arrays as the bytes of their own dtype; the rest as canonical JSON."""
    from mrgym.scanner import session
    w = _engine(kind).world
    pd = np.array([w["tissues"][k]["pd"] for k in w["names"]], np.float32)
    out = dict(shape=list(w["weights"].shape), weights=sha(np.ascontiguousarray(w["weights"]).tobytes()),
               fingerprint=session.fingerprint(np.tensordot(pd, w["weights"], 1)[None]))
    for k in sorted(w["fields"]):
        v = w["fields"][k]
        out["field." + k] = sha(np.ascontiguousarray(v).tobytes()) if isinstance(v, np.ndarray) else sha(v)
    for k in sorted(w["masks"]):
        out["mask." + k] = sha(np.ascontiguousarray(w["masks"][k]).tobytes())
    for k in ("names", "tissues", "system", "regions", "console", "size_mm", "effects", "air", "kind", "b0_t", "n", "extent_mm", "schema"):
        out[k] = sha(w.get(k))
    return out


def exam_case(kind) -> dict:
    from mrgym.scanner import console, session
    exam = _engine(kind).world["console"]
    return dict(schema=sha(console.protocol_schema(exam)), schema_brief=sha(console.protocol_schema(exam, brief=True)),
                scanner_card=sha(session.scanner_card(CARD_CFG, exam)))


def scan_case(kind, sequence) -> dict:
    """``plan`` (the whole output), the header line and the noise-free image of one protocol on one kind."""
    from mrgym.scanner import console, session
    eng = _engine(kind)
    pl = console.plan(PROTOCOLS[sequence], eng.world["console"])
    images = eng.scan(PROTOCOLS[sequence])["images"]
    mag, ph = np.abs(images).astype(np.float32), np.angle(images).astype(np.float32)
    return dict(plan=sha(pl), header=pl["header"], shape=list(images.shape), image=sha(mag.tobytes() + ph.tobytes()),
                fingerprint=session.fingerprint(mag))


def refusals() -> dict:
    """What a 2D exam answers to the parameters this scanner does not simulate: the whole ``plan`` output."""
    from mrgym.scanner import console
    return {name: sha(console.plan(dict(PROTOCOLS["se"], **{name: 2}))) for name in sorted(console.NOT_SIMULATED)}


def stored_golden() -> dict:
    """Every entry of the stored engine digests, hashed: ``images/<key>``, ``noisy/<key>``, ``prices/<key>``."""
    with open(os.path.join(HERE, "golden", "engine_digests.json"), encoding="utf-8") as f:
        g = json.load(f)
    return {f"{part}/{k}": sha(v)[:32] for part in ("images", "noisy", "prices") for k, v in g[part].items()}


def stored_fingerprints() -> dict:
    with open(os.path.join(ROOT, "src", "mrgym", "compat", "fingerprints.json"), encoding="utf-8") as f:
        return {env_id: row["fingerprint"] for env_id, row in json.load(f)["ids"].items()}


def scan_keys() -> list[str]:
    return [f"{kind}/{seq}" for kind in KINDS_2D for seq in PROTOCOLS]


def compute() -> dict:
    """Everything, from the tree that is imported (``mrgym.__file__``) and the stored files next to this module."""
    import golden_cases as G
    import mrgym
    out = dict(engine=mrgym.ENGINE_VERSION, stack=G.stack(), worlds={}, exams={}, scans={})
    for kind in KINDS_2D:                                               # grouped: one world in memory at a time
        out["worlds"][kind], out["exams"][kind] = world_case(kind), exam_case(kind)
        for seq in PROTOCOLS:
            out["scans"][f"{kind}/{seq}"] = scan_case(kind, seq)
    out.update(refusals=refusals(), golden=stored_golden(), fingerprints=stored_fingerprints())
    return out


def load() -> dict | None:
    try:
        with open(FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def compare(old: dict, new: dict) -> tuple[list, list, list]:
    """(entries whose value differs, entries only in new, entries only in old), ``recorded`` and ``stack`` left out."""
    a, b = (_flat({k: v for k, v in d.items() if k not in ("recorded", "stack")}) for d in (old, new))
    return sorted(k for k in a if k in b and a[k] != b[k]), sorted(set(b) - set(a)), sorted(set(a) - set(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write or check tests/golden/compat_2d.json (the 2D baseline)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rebase", action="store_true", help="with --write: store a new baseline although stored values change")
    a = ap.parse_args(argv)
    if a.write == a.check:
        ap.error("give exactly one of --write, --check")
    import golden_cases as G
    import mrgym
    old, new = load(), compute()
    diff, added, gone = compare(old or {}, new)
    print(f"compat_2d: {len(diff)} changed, {len(added)} new, {len(gone)} gone (tree: {os.path.dirname(mrgym.__file__)})")
    for k in (diff + gone)[:20]:
        print("  changed:" if k in diff else "  gone:", k)
    if a.check:                                                         # new entries (a new id, a new golden case) are not part of the baseline
        return 1 if diff or gone or not old else 0
    if old and (diff or gone) and not a.rebase:
        print("refused: stored values of the 2D baseline would change. If that is intended (a new engine version, an id "
              "re-fingerprinted on purpose) give --rebase and say so in the commit; otherwise the tree changed a 2D exam.")
        return 1
    if old and old["stack"] != new["stack"] and not a.rebase:
        print("refused: this is not the numerical stack the baseline was recorded on (give --rebase to move it)")
        return 1
    try:
        tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(mrgym.__file__))))
        commit = subprocess.run(["git", "-C", tree, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", tree, "status", "--porcelain", "--", "src"], capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", False
    keep = old["recorded"] if old and not (diff or gone) else dict(commit=commit, source_tree_clean=not dirty, engine=new["engine"])
    G.dump(FILE, dict(new, recorded=keep))
    print("wrote", os.path.relpath(FILE, ROOT), "recorded on", keep)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, HERE)
    sys.exit(main())
