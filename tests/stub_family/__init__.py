"""A task family that costs microseconds, for the tests of ``mrgym.gate`` and ``envs/family.py``.

It is registered by the ``stub_env`` fixture of ``tests/conftest.py`` as ``Test/Stub-v0`` and never ships. The hidden
world is a small QA phantom (96 x 96 master grid); the "measurement" is one cheap scan; the deliverable is the number on
the task card. What a policy passes is therefore a known function of the seed, which is what the tests compute.
"""
