"""Checks for the recon-repair family. Run: python3 tests/test_recon_repair.py  (no test framework needed)."""
import hashlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrfrl.recon_repair import alt, cartesian as cart, task  # noqa: E402


def tree_hash(root):
    h = hashlib.sha256()
    for base, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            h.update(os.path.relpath(os.path.join(base, name), root).encode())
            if name.endswith(".npz"):                     # zip containers carry timestamps; hash the arrays
                with np.load(os.path.join(base, name)) as z:
                    for key in sorted(z.files):
                        h.update(key.encode() + z[key].tobytes())
            else:
                h.update(open(os.path.join(base, name), "rb").read())
    return h.hexdigest()


def test_every_variant_changes_the_source():
    for v, (_, _, req) in cart.VARIANTS.items():
        d = cart.sample_dialect(np.random.default_rng(1), req)
        assert cart.recon_source(d, [v]) != cart.recon_source(d), v
        compile(cart.recon_source(d, [v]), "recon.py", "exec")


def test_format_doc_names_every_field():
    for seed in range(20):
        d = cart.sample_dialect(np.random.default_rng(seed))
        doc = cart.format_doc(d)
        for name in d["names"].values():
            assert f"`{name}" in doc, (seed, name)


def test_instances_are_reproducible():
    a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
    for out in (a, b):
        task.write(task.build(7, 2, "train"), out)
    assert tree_hash(a) == tree_hash(b)


def test_splits_keep_mechanisms_apart():
    for seed in range(15):
        rng = np.random.default_rng([task.SPLITS["train"], 3, seed, 0])
        variants, _, _ = task.sample(rng, 3, "train")
        assert not {cart.VARIANTS[v][0] for v in variants} & set(cart.HELD_OUT_GROUPS)
        rng = np.random.default_rng([task.SPLITS["heldout_mechanism"], 1, seed, 0])
        variants, _, _ = task.sample(rng, 1, "heldout_mechanism")
        assert cart.VARIANTS[variants[0]][0] in cart.HELD_OUT_GROUPS


def test_reference_and_second_implementation_agree_with_truth():
    worst_ref, worst_alt = 0.0, 0.0
    for seed in range(12):
        rng = np.random.default_rng([5, seed])
        d = cart.sample_dialect(rng)
        npz, truth = cart.simulate(cart.sample_scan(rng), d)
        e_ref = task.error(task.run_source(cart.recon_source(d), npz), truth)
        e_alt = task.error(alt.reconstruct(npz, d), truth)
        assert e_ref <= task.ORACLE_MAX and e_alt <= task.threshold(e_ref), (seed, e_ref, e_alt)
        worst_ref, worst_alt = max(worst_ref, e_ref), max(worst_alt, e_alt / task.threshold(e_ref))
    print(f"  worst reference error {worst_ref:.3f}; second implementation at most {worst_alt:.2f} x threshold")


def test_answer_key_is_not_in_the_agent_image():
    out = tempfile.mkdtemp()
    root = task.write(task.build(3, 3, "train"), out)
    seen = {os.path.relpath(os.path.join(b, f), root) for b, _, fs in os.walk(os.path.join(root, "environment")) for f in fs}
    assert seen == {"environment/Dockerfile", "environment/app/recon.py", "environment/app/FORMAT.md",
                    "environment/app/data/scan_1.npz", "environment/app/data/scan_2.npz"}, seen   # level 3: no scout


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
