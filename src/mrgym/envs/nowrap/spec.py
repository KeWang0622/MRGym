"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every claim level on
the current engine; then "admitted". A family whose gate rejects a claim level is "practice".

Levels 0 and 1 are practice levels (``task.LEVELS[lv]["kind"] == "practice"``): registered and playable, not run by
the gate by default and not covered by the claim. ``Scanner/NoWrap-v0`` is retired (``mrgym.envs``).
"""
SPECS = [dict(id="Scanner/NoWrap-v1", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.nowrap.task:card",
              kind="candidate", family="mrgym.envs.nowrap", levels=(0, 1, 2, 3), default_level=2, max_steps=40)]
