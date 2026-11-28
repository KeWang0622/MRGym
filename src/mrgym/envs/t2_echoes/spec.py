"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine. The console of this family lists one sequence beyond the six: ``mese`` (world spec ``console_sequences``).
"""
SPECS = [dict(id="Scanner/T2Echoes-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.t2_echoes.task:card",
              kind="candidate", family="mrgym.envs.t2_echoes", levels=(1, 2, 3), default_level=2, max_steps=40)]
