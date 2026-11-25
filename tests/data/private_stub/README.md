# Stub of a private family package

A private package adds task families to MR Gym without touching the public tree: it declares an entry point in the
group `mrgym.envs` whose value is a module with `SPECS` (the same list a built-in family keeps in its `spec.py`).
`import mrgym` registers every such module; a plug-in that fails to load is reported as a warning and skipped.

    [project.entry-points."mrgym.envs"]
    private_stub = "mrgym_private_stub.spec"

This directory is what `pip install` of such a package leaves on the path: the package and its `*.dist-info` with
`entry_points.txt`. `tests/test_keys.py` puts it on `PYTHONPATH` and checks that `mrgym list` shows the id. The family
itself is a thin copy of `Scanner/NullTissue` whose `make_instance` takes the split (the hook for held-out ranges).
