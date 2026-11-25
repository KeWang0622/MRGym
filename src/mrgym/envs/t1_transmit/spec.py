"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "practice". The family is a measurement task: one fixed protocol with a fixed estimator (the ``open_loop``
control) is built to pass wherever the reference passes, so no decision of the agent is tested. The stored gate result
(``mrgym gate <id> --n 300 --write``) reports that row next to the reference; ``mrgym check`` accepts the practice
status when the row is within 25 points of the reference at every level, or when the gate rejects a level.
``Scanner/T1Transmit-v0`` is retired (``mrgym.envs``).
"""
SPECS = [dict(id="Scanner/T1Transmit-v1", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.t1_transmit.task:card",
              kind="practice", family="mrgym.envs.t1_transmit", levels=(1, 2, 3), default_level=2, max_steps=64)]
