"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine. The family is a measurement task: the reference plays one fixed scan with a fixed estimator, and the
gate table reports that under ``open_loop``.
"""
SPECS = [dict(id="Scanner/FatFraction-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.fat_fraction.task:card",
              kind="candidate", family="mrgym.envs.fat_fraction", levels=(1, 2, 3), default_level=2, max_steps=48)]
