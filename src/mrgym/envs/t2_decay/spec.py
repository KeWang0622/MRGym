"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine. The family is a measurement task by construction at levels 1 and 2 (the reference plays a fixed list
of scans); the gate table reports the fixed list (``open_loop``) next to the reference at every level.
"""
SPECS = [dict(id="Scanner/T2Decay-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.t2_decay.task:card",
              kind="admitted", family="mrgym.envs.t2_decay", levels=(1, 2, 3), default_level=2, max_steps=48)]
