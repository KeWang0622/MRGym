"""``Scanner/ProtocolToSpec``: adapt the standard protocol of a two-image examination until four coupled requirements hold.

    spec.py       registration (found by ``mrgym.envs``)
    task.py       instances, the task text, the deliverable, the verifier, the card
    oracle.py     the feasible protocols on the hidden world, by exhaustive search; independent of ``controls``
    controls.py   the reference, the policies that must fail, the ablations, ``open_loop`` and ``characterise_all``

The family is built to be a decision task by measurement: on a console where every scan is priced, the budget pays for
two scans; the first shows which requirement fails and the second has to be chosen accordingly. Whether a level is a
measurement task or a decision task is read from its gate result (``task.task_kind``), never asserted.
"""
