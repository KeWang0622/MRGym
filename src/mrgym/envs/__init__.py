"""Built-in environments. Importing this module registers their ids.

A task family is a sub-package with a ``spec.py`` that holds ``SPECS``: a list of keyword dicts for ``register``. The
sub-packages are found by listing this directory, so adding a family touches no shared file.

Other packages add families through the entry-point group ``mrgym.envs``: each entry point names a module with ``SPECS``
(a private package registers its families this way, without a change to this tree). A plug-in that cannot be loaded is
reported as a warning and skipped; it never breaks ``import mrgym``.
"""
import importlib
import pkgutil
import warnings

from ..registry import register, retire

register("Scanner/Explore-v0", "mrgym.envs.explore:ExploreEnv", "mrgym.envs.explore:card", kind="sandbox",
         levels=(1,), default_level=1, max_steps=1024, reward_range=(0.0, 0.0), reward_threshold=0.0)

# Retired ids (docs/VERSIONING.md): package 0.1.0 is the last one that serves them. Their graded reward, their
# per-instance admission and their stored results belong to engine 0.1.0; the successors are families on the pass-only
# contract of ``envs/family.py``, registered by the ``spec.py`` of their packages.
retire("Scanner/NoWrap-v0", "0.1.0", successor="Scanner/NoWrap-v1")
retire("Scanner/T1Transmit-v0", "0.1.0", successor="Scanner/T1Transmit-v1")

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__) if m.ispkg):
    try:
        _spec = importlib.import_module(f"{__name__}.{_m}.spec")
    except ModuleNotFoundError as _e:
        if _e.name != f"{__name__}.{_m}.spec":
            raise
        continue
    for _kw in _spec.SPECS:
        register(**_kw)


def load_entry_points(group: str = "mrgym.envs") -> list:
    """Register the families of installed plug-in packages. Returns the ids that were added."""
    from importlib import metadata
    added = []
    for ep in sorted(metadata.entry_points(group=group), key=lambda e: e.name):
        try:
            for kw in ep.load().SPECS:
                added.append(register(**kw).id)
        except Exception as e:  # noqa: BLE001  a broken plug-in must not take the package down
            warnings.warn(f"mrgym: the environment plug-in {ep.name!r} ({ep.value}) was not registered: {type(e).__name__}: {e}",
                          RuntimeWarning, stacklevel=2)
    return added


load_entry_points()
