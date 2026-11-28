"""Registration of the family (read by ``mrgym.envs`` at import; keep this module free of heavy imports).

``kind``: "candidate" until the stored gate result (``mrgym gate <id> --n 300 --write``) admits every claim level on
the current engine; then "admitted". A family whose gate rejects a claim level is "practice".
``Scanner/RFPulse-v0`` stays registered and served as it was (docs/VERSIONING.md: which ids retire is decided at a release).
"""
SPECS = [dict(id="Scanner/RFPulse-v1", entry_point="mrgym.envs.design_tools_v1:DesignFamilyEnvV1", card="mrgym.envs.rf_pulse_v1.task:card",
              kind="admitted", family="mrgym.envs.rf_pulse_v1", levels=(1, 2, 3), default_level=2, max_steps=40)]
