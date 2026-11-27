"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine; then "admitted". A family whose gate rejects a level is "practice".
"""
SPECS = [dict(id="Scanner/FrequencyAdjust-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.freq_adjust.task:card",
              kind="candidate", family="mrgym.envs.freq_adjust", levels=(1, 2, 3), default_level=2, max_steps=40)]
