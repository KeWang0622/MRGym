"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine; then "admitted". A family whose gate rejects a level is "practice".
"""
SPECS = [dict(id="Scanner/SNRSpec-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.snr_spec.task:card",
              kind="admitted", family="mrgym.envs.snr_spec", levels=(1, 2, 3), default_level=2, max_steps=48)]
