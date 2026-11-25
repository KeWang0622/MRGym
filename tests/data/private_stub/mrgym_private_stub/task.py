"""The stub family: ``Scanner/NullTissue`` under another id, with a ``make_instance`` that takes the split."""
from mrgym.envs.null_tissue import task as _base
from mrgym.envs.null_tissue.task import *  # noqa: F401, F403

FAMILY = "private_stub"
LEVELS = {lv: _base.LEVELS[lv] for lv in (1, 2)}


def make_instance(seed, level, task_id=None, split="train"):
    """``split`` is where a family draws its held-out ranges; the stub only records it in the hidden config."""
    cfg = _base.make_instance(seed, level, task_id)
    cfg["split"] = split
    return cfg


def card():
    return dict(_base.card(), id="Scanner/PrivateStub-v0", title="Stub of a private family (test data)", kind="candidate")
