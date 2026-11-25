"""MR Gym on verl: an in-process agent loop, instance rows, a curriculum hook, checkpoint mirroring, and a GPU-free
simulation of the whole run. See ../README.md.

Nothing here is imported by the ``mrgym`` package. ``episode``, ``instances``, ``store``, ``stoprules``, ``calsum``,
``readout``, ``mixstub`` and ``sim`` need numpy and ``mrgym`` only; ``agent_loop``, ``checkpoint`` and ``dryrun``
import verl.
"""
PINNED_VERL = "8357f1e0a7ce4c06dc970c2c637d49b33815f5cc"       # verl main, 2026-11-24 (0.10.0.dev0)
AGENT_NAME = "mrgym_agent"
SIGNALS = ("pass", "progress")                                  # "pass" is the environment's paid reward; "progress" is not
