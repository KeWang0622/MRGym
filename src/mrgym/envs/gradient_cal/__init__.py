"""``Scanner/GradientCal``: the amplitude calibration of the two in-plane gradients, from a phantom of known size.

    spec.py       registration (found by ``mrgym.envs``)
    task.py       instances, the task text, the deliverable, the verifier, the card
    controls.py   the reference policy (sub-pixel edges on the phase-encode axis of two scans), the shortcuts that must
                  fail, the ablation without the non-linearity step, tuning

One test of the quality-control battery, on its own: ``Scanner/PhantomQA`` reads two diameters to 0.6 mm and says
which fault it is; this family delivers the calibration itself, to 0.3 %.
"""
