"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every level on the
current engine; then "admitted". A family whose gate rejects a level is "practice".

``not_ready``: since engine 0.3.0 the id does not examine what it was built for (module text of ``task.py``, "Engine
0.3.0"; ``tests/test_fat_suppress.py::test_on_this_engine_fat_saturation_passes_in_no_region``). Its thresholds are part
of the id and stay; the successor is ``Scanner/FatSuppress-v1`` with thresholds taken from the corrected physics, which
is not built.
"""
SPECS = [dict(id="Scanner/FatSuppress-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym.envs.fat_suppress.task:card",
              kind="candidate", family="mrgym.envs.fat_suppress", levels=(1, 2, 3), default_level=2, max_steps=64,
              not_ready="spectral fat saturation leaves the fat lines outside the band of its pulse, more than the task's threshold allows, "
                        "so inversion recovery is the answer in every region and the choice the task examines does not arise; "
                        "it needs other thresholds, which is a new id (Scanner/FatSuppress-v1, not built)")]
