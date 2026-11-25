"""``Scanner/PhantomQA``: phantom QA on the ACR large phantom, with the diagnosis of the system fault.

    spec.py       registration (found by ``mrgym.envs``)
    task.py       instances, the task text, the deliverable, the verifier, the card
    controls.py   the reference policy, the policies that must fail, the ablations, the open-loop control, tuning
    acr_fault_matrix.json   the stored fault x test matrix of ``mrgym qa acr --matrix --write`` (``mrgym.qa``)
    gate_dev.json           a development run of the gate (its n is in the file); the lead's stored gate replaces it
"""
