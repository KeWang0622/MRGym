"""Register the example family from this folder, play its reference policy once, then run a small gate.

    python examples/new_family/play.py            # about one minute on two cores
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # examples/: this folder is the package new_family

import mrgym  # noqa: E402
from mrgym import gate, policies  # noqa: E402
from new_family import spec  # noqa: E402

if __name__ == "__main__":
    env_id = spec.SPECS[0]["id"]
    if env_id not in mrgym.registry:
        mrgym.register(**spec.SPECS[0])
    env = mrgym.make(env_id, level=1, seed=0)
    print(env.reset()[0].text, "\n")
    step = policies.run(env, "reference", seed=0)
    print(f"reference policy: reward {step.reward:.0f}, {step.info['budget']['used_s']:.0f} of {step.info['budget']['budget_s']:.0f} s of scan time\n")
    print(gate.text_table(gate.run(env_id, n=int(os.environ.get("N", 60)), workers=2)))
