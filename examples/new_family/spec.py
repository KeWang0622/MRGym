"""Registration. ``kind`` stays "candidate" until ``mrgym gate <id> --n 300 --write`` has stored a result."""
SPECS = [dict(id="Example/T2Vials-v0", entry_point="mrgym.envs.family:FamilyEnv", card=f"{__package__}.task:card",
              kind="candidate", family=__package__, levels=(1,), default_level=1, max_steps=24)]
