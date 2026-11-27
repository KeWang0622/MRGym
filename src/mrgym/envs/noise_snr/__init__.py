"""``Scanner/NoiseSNR``: the signal-to-noise ratio of a stated protocol, measured the way the standards measure it.

    spec.py       registration (found by ``mrgym.envs``)
    task.py       instances, the task text, the deliverable, the verifier, the card
    controls.py   the reference policy (two acquisitions and their difference), the shortcuts that must fail, the
                  background-method ablation, tuning

One test of the quality-control battery, on its own: ``Scanner/PhantomQA`` measures geometry, uniformity and ghosting
and names the fault; this family asks for the one number PhantomQA does not ask for, the noise.
"""
