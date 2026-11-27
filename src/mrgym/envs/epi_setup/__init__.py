"""``Scanner/EPISetup``: a single-shot echo-planar protocol that meets its limits on displacement, ghost, wrap-around and
signal-to-noise ratio in the least scan time.

    spec.py       registration (found by ``mrgym.envs``)
    task.py       instances, the task text, the deliverable, the verifier, the card
    metrics.py    what the verifier measures: twins of the delivered scan with one mechanism switched off
    oracle.py     the fewest averages with which any allowed protocol meets every requirement, on the hidden world
    controls.py   the reference policy, the policies that must fail, the ablations, the reported controls, tuning
"""
