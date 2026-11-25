"""What the entry point names: ``SPECS``, keyword dicts for ``mrgym.register`` (no heavy imports here)."""
SPECS = [dict(id="Scanner/PrivateStub-v0", entry_point="mrgym.envs.family:FamilyEnv", card="mrgym_private_stub.task:card",
              kind="candidate", family="mrgym_private_stub", levels=(1, 2), default_level=1, max_steps=48)]
