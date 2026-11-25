"""Checkpoint callback: copy every finished ``global_step_N`` folder off the node as soon as verl has written it.

    trainer.checkpoint_callback_class=mrgym_verl.checkpoint.SyncCheckpoint

A preemptible node loses its disk, so the copy is synchronous: ``on_save`` returns after the upload and only then
writes ``<dest>/LATEST`` with the step number (``mrgym_verl.store.publish_checkpoint``). The launch script reads
``LATEST`` on a fresh disk, pulls that folder into ``trainer.default_local_dir`` and lets
``trainer.resume_mode=auto`` pick it up. The curriculum's state file is copied into the checkpoint folder first, so
a checkpoint carries the sampler it was trained with.

Destination: ``MRGYM_CKPT_DEST`` = ``s3://bucket/prefix`` (boto3, the node's role) or a directory. Unset: no copy.
"""
from __future__ import annotations

import os
import time

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback

from . import instances, store


class SyncCheckpoint(CheckpointCallback):
    def on_save(self, trainer, global_step: int, checkpoint_dir: str, async_save: bool = False, **kwargs) -> None:
        dest = os.environ.get("MRGYM_CKPT_DEST")
        if not dest:
            return
        if async_save:                       # worker-side writes may still be in flight: not durable, do not publish
            print(f"[mrgym] step {global_step}: asynchronous save, not copied", flush=True)
            return
        t0 = time.time()
        instances.snapshot_state(checkpoint_dir)
        latest = store.publish_checkpoint(checkpoint_dir, dest, int(global_step))
        print(f"[mrgym] step {global_step}: checkpoint copied to {dest} ({latest['files']} file(s), {time.time() - t0:.1f} s)", flush=True)
