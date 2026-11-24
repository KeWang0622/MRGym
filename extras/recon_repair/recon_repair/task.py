"""Task instances of the recon-repair family: sampling, the admission gate, and writing a task directory.

An instance is fully determined by (FAMILY, VERSION, split, level, seed). Nothing is stored between the generator and
the task directory; a directory can always be rebuilt from its key.
"""
import io
import json
import os
import shutil
import tempfile
import warnings

import numpy as np

from . import alt, sim
from . import cartesian as cart

FAMILY, VERSION = "recon-repair", "0.2"
ORGANISATION = "mrgym-extras"
NETWORK_MODE = "no-network"     # the reference solution can be regenerated from FORMAT.md by anyone who can install this folder
SPLITS = {"train": 11, "dev": 12, "heldout_mechanism": 13}        # seed namespaces; no two splits share a stream
LEVEL_FAULTS = {1: 1, 2: 2, 3: 3}
N_PUBLIC, N_HIDDEN = 2, 3
MARGIN = 2.0          # a fault must push the error to at least MARGIN x the pass threshold on every scan
ORACLE_MAX = 0.08     # the reference itself must be this close to the truth


def threshold(e_ref):
    return max(1.6 * e_ref, e_ref + 0.015)


def run_source(src, npz):
    """Run reconstruction source text on one scan (dict of arrays) in this process. Returns the image or None."""
    work = tempfile.mkdtemp(prefix="recon_repair_gate_")
    try:
        np.savez(os.path.join(work, "scan.npz"), **npz)
        ns = {"__name__": "recon"}
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                exec(compile(src, "recon.py", "exec"), ns)
                ns["main"](os.path.join(work, "scan.npz"), os.path.join(work, "image.npy"))
                return np.load(os.path.join(work, "image.npy"))
            except Exception:
                return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def error(img, truth):
    if img is None or img.shape != truth.shape or not np.all(np.isfinite(img)):
        return float("inf")
    return sim.nrmse(img, truth)


def sample(rng, level, split):
    """Choose mechanisms, one variant each, a dialect that makes them expressible, and the scans."""
    held = list(cart.HELD_OUT_GROUPS)
    seen = [g for g in cart.GROUPS if g not in held]
    n = LEVEL_FAULTS[level]
    if split == "heldout_mechanism":                      # at least one mechanism never seen in training
        first = [held[int(rng.integers(len(held)))]]
        rest = [g for g in cart.GROUPS if g != first[0]]
        groups = first + [rest[i] for i in rng.permutation(len(rest))[:n - 1]]
    else:
        groups = [seen[i] for i in rng.permutation(len(seen))[:n]]
    variants = []
    for g in groups:
        options = sorted(v for v, (vg, _, _) in cart.VARIANTS.items() if vg == g)
        variants.append(options[int(rng.integers(len(options)))])
    require, need = {}, set()
    for v in variants:
        need |= cart.VARIANTS[v][1]
        for key, allowed in cart.VARIANTS[v][2].items():
            require[key] = [a for a in require.get(key, allowed) if a in allowed]
    dialect = cart.sample_dialect(rng, require)
    scans = [cart.sample_scan(rng, frozenset(need)) for _ in range(N_PUBLIC + N_HIDDEN)]
    return variants, dialect, scans


def gate(variants, dialect, sims):
    """The admission checks. Returns (ok, report). ``sims`` is a list of (npz dict, truth)."""
    reference = cart.recon_source(dialect)
    e_ref = [error(run_source(reference, z), t) for z, t in sims]
    thr = [threshold(e) for e in e_ref]
    report = dict(error_reference=e_ref, threshold=thr)
    if max(e_ref) > ORACLE_MAX:
        return False, dict(report, reason="reference too far from the truth")
    e_alt = [error(alt.reconstruct(z, dialect), t) for z, t in sims]
    report["error_second_implementation"] = e_alt
    if any(e > th for e, th in zip(e_alt, thr)):
        return False, dict(report, reason="second correct implementation rejected")
    given = cart.recon_source(dialect, variants)
    e_given = [error(run_source(given, z), t) for z, t in sims]
    report["error_as_given"] = e_given
    if any(e < MARGIN * th for e, th in zip(e_given, thr)):
        return False, dict(report, reason="reconstruction as given is not clearly wrong on every scan")
    report["error_one_fault_left"] = {}
    if len(variants) > 1:                                # repairing all faults but one must still fail everywhere
        for v in variants:
            e_one = [error(run_source(cart.recon_source(dialect, [v]), z), t) for z, t in sims]
            report["error_one_fault_left"][v] = e_one
            if any(e < MARGIN * th for e, th in zip(e_one, thr)):
                return False, dict(report, reason=f"fault {v} alone is not clearly wrong on every scan")
    return True, report


def build(seed, level=1, split="train", max_attempts=40):
    """Return an admitted instance as a dict, resampling until the gate passes."""
    rejected = []
    for attempt in range(max_attempts):
        rng = np.random.default_rng([SPLITS[split], level, seed, attempt])
        variants, dialect, scans = sample(rng, level, split)
        sims = [cart.simulate(s, dialect) for s in scans]
        ok, report = gate(variants, dialect, sims)
        if ok:
            return dict(family=FAMILY, version=VERSION, split=split, level=level, seed=seed, attempt=attempt,
                        variants=variants, groups=[cart.VARIANTS[v][0] for v in variants], dialect=dialect,
                        scans=scans, sims=sims, gate=report, rejected=rejected)
        rejected.append(dict(attempt=attempt, variants=variants, reason=report["reason"]))
    raise RuntimeError(f"no admissible instance for seed {seed} level {level} split {split}: {rejected}")


# ------------------------------------------------------------------------------------------------ writing

INSTRUCTION = """# Repair a reconstruction

`/app/recon.py` reconstructs 2D multi-coil Cartesian MRI scans written by one scanner. The scanner's raw data format
is described in `/app/FORMAT.md`. The script runs, but the images it produces from this scanner's data are wrong.

`/app/data/` holds two scans from the scanner, `scan_1.npz` and `scan_2.npz`.{scout}

Repair `/app/recon.py` so that

    python recon.py <scan.npz> <image.npy>

writes the correct image for any scan from this scanner: a 2D float array indexed [y, x], with the shape, orientation
and centring defined in `FORMAT.md`, showing the magnitude of the object with the coil sensitivities divided out.
The overall scale of the image is free.

How your work is checked: your `recon.py` is run on other scans from the same scanner that you have not seen. They
show different objects and use other matrix sizes and other values of the header and per-record fields, within what
`FORMAT.md` allows. Each image is compared with the true object by the relative root-mean-square error of the magnitude
image over the whole field of view, after the best global scale. The error must be close to what a correct
reconstruction of that scan achieves (the limit is not disclosed; a correct reconstruction passes with margin, a
reconstruction with any fault left does not). Every scan must pass.

Rules
- `FORMAT.md` is correct and the data follow it. The fault, or faults, are in `recon.py`.
- The deliverable is the single file `/app/recon.py`. It may use only the Python standard library, numpy and scipy,
  and must finish within 120 seconds per scan.
- Do not hard-code anything about the two example scans.
"""
SCOUT = (" Next to each is `scout_1.npy` / `scout_2.npy`: a coarse localizer image of the same slice ([y, x], centred "
         "on the same prescribed centre, band-limited to a quarter of the matrix), which you can use to judge your result.")

DOCKERFILE = """FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 && useradd -m agent
WORKDIR /app
COPY app/ /app/
RUN chown -R agent:agent /app
"""
TEST_SH = """#!/bin/bash
# Runs in the separate verifier sandbox. /app/recon.py is the artifact copied from the agent's sandbox.
set -uo pipefail
chmod -R go-rwx /tests            # ground truth is readable by root only; the submission runs as "nobody"
mkdir -p /logs/verifier
python /tests/verify.py --submission /app --hidden /tests/hidden \\
    --out /logs/verifier/reward.json --details /logs/verifier/details.json
if [ ! -s /logs/verifier/reward.json ]; then echo '{"reward": 0.0}' > /logs/verifier/reward.json; fi
"""
SOLVE_SH = """#!/bin/bash
set -euo pipefail
cp "$(dirname "$0")/recon.py" /app/recon.py
"""


def task_name(inst):
    return f"{FAMILY}-{inst['split'].replace('_', '-')}-l{inst['level']}-s{inst['seed']:05d}"


def write(inst, out_root):
    """Write the instance as a Harbor-format task directory. Returns its path."""
    root = os.path.join(out_root, task_name(inst))
    shutil.rmtree(root, ignore_errors=True)
    app = os.path.join(root, "environment", "app")
    hidden = os.path.join(root, "tests", "hidden")
    for p in (os.path.join(app, "data"), hidden, os.path.join(root, "solution")):
        os.makedirs(p)
    d, g = inst["dialect"], inst["gate"]
    with_scout = inst["level"] < 3
    cases = []
    for i, (npz, truth) in enumerate(inst["sims"]):
        if i < N_PUBLIC:
            np.savez(os.path.join(app, "data", f"scan_{i + 1}.npz"), **npz)
            if with_scout:
                np.save(os.path.join(app, "data", f"scout_{i + 1}.npy"), cart.scout(truth))
        else:
            j = i - N_PUBLIC + 1
            np.savez(os.path.join(hidden, f"scan_{j}.npz"), **npz)
            np.save(os.path.join(hidden, f"truth_{j}.npy"), truth)
            cases.append(dict(scan=f"scan_{j}.npz", truth=f"truth_{j}.npy", threshold=g["threshold"][i],
                              error_reference=g["error_reference"][i], error_as_given=min(g["error_as_given"][i], 10.0)))
    _text(os.path.join(hidden, "meta.json"), json.dumps(dict(cases=cases), indent=1))
    _text(os.path.join(app, "recon.py"), cart.recon_source(d, inst["variants"]))
    _text(os.path.join(app, "FORMAT.md"), cart.format_doc(d))
    _text(os.path.join(root, "instruction.md"), INSTRUCTION.format(scout=SCOUT if with_scout else ""))
    _text(os.path.join(root, "environment", "Dockerfile"), DOCKERFILE)
    _text(os.path.join(root, "solution", "recon.py"), cart.recon_source(d))
    _text(os.path.join(root, "solution", "solve.sh"), SOLVE_SH, 0o755)
    _text(os.path.join(root, "tests", "test.sh"), TEST_SH, 0o755)
    shutil.copy(os.path.join(os.path.dirname(__file__), "verify.py"),
                os.path.join(root, "tests", "verify.py"))
    _text(os.path.join(root, "task.toml"), f'''schema_version = "1.4"
artifacts = ["/app/recon.py"]

[task]
name = "{ORGANISATION}/{task_name(inst)}"

[metadata]
family = "{FAMILY}"
family_version = "{VERSION}"
split = "{inst['split']}"
level = {inst['level']}
seed = {inst['seed']}
category = "mri-reconstruction"

[verifier]
environment_mode = "separate"
timeout_sec = 900.0
user = "root"

[agent]
timeout_sec = 3600.0
user = "agent"

[environment]
network_mode = "{NETWORK_MODE}"
cpus = 2
memory_mb = 4096
''')
    # answer key: lives beside the task, never inside environment/
    key = {k: inst[k] for k in ("family", "version", "split", "level", "seed", "attempt", "variants", "groups",
                                "dialect", "gate", "rejected")}
    _text(os.path.join(root, "answer_key.json"), json.dumps(key, indent=1, default=float))
    return root


def _text(path, text, mode=0o644):
    with io.open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, mode)
