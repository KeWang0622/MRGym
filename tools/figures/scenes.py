"""The phantoms and the scan call the gallery figures share.

The worlds are the ones ``mrgym scan --seed 1`` images, so a reader can reproduce any panel from the command line:

    mrgym scan --phantom brain --lesion ms sequence=se tr_ms=3000 te_ms=90 matrix=192 pe_dir=x phase_fov=0.75
"""
from __future__ import annotations

import functools

import numpy as np

from mrgym.env import derive_seed
from mrgym.scanner import kspace
from mrgym.scanner import session as core
from mrgym.scanner import world as W

EXPLORE_ID, USER_SEED = "Scanner/Explore-v0", 1
SPECS = dict(
    brain=dict(kind="brain", b0_t=3.0, lesion="ms"),
    abdomen=dict(kind="abdomen", b0_t=3.0, lesion="liver"),
)


@functools.cache
def engine(name: str) -> kspace.Engine:
    """The k-space engine on one of the two gallery phantoms (3 T)."""
    seed = derive_seed(EXPLORE_ID, 1, "train", USER_SEED)
    return kspace.Engine(W.make_world(seed, SPECS[name]))


def scan(name: str, protocol: dict, noise: int = 0) -> dict:
    """One scan of a gallery phantom: dict(image, fov_yx_mm, header, realised, scan_s). Fixed noise stream."""
    out = engine(name).scan(protocol, np.random.default_rng(noise))
    rp = out["plan"]["realised"]
    return dict(image=np.abs(out["images"][0]), fov_yx_mm=core.fov_yx(rp), header=out["header"], realised=rp,
                scan_s=out["plan"]["scan_s"], protocol=protocol)
