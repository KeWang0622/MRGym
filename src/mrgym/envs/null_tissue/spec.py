"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine; then "admitted". A family whose gate rejects a level is "practice".

Level 0 is a practice level (``task.LEVELS[0]["kind"] == "practice"``): it is registered and playable, the gate does
not run it by default and the admitted claim does not cover it.
"""
SPECS = [dict(id="Scanner/NullTissue-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.null_tissue.task:card",
              kind="admitted", family="mrgym.envs.null_tissue", levels=(0, 1, 2, 3), default_level=2, max_steps=48)]
