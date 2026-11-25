"""docs/TRAINING.md and the pre-registered plan hold no typed table: what is computed is compared with the code."""
import importlib.util
import json
import os

import pytest

from mrgym import learn
from mrgym.curriculum import Curriculum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def _read(*parts):
    path = os.path.join(DOCS, *parts)
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_band_table_of_the_training_page_is_the_computed_one():
    doc = _read("TRAINING.md")
    start, end = "<!-- band-table: mrgym.learn.band_table() -->", "<!-- /band-table -->"
    assert doc[doc.index(start) + len(start):doc.index(end)].strip() == learn.band_table()
    assert f"at least {100 * learn.MIXED_MIN:.0f} %" in doc and "groups of 8" in doc and learn.GROUP_SIZE == 8


def test_compute_table_of_the_plan_is_the_computed_one():
    _read("experiments", "PREREG-learnability.md")
    path = os.path.join(DOCS, "experiments", "learnability", "compute.py")
    spec = importlib.util.spec_from_file_location("prereg_compute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0
    plan = json.loads(_read("experiments", "learnability", "plan.json"))
    full, core = module.rows(plan)
    assert full["total"]["low"] < full["total"]["high"] and core["total"]["high"] < full["total"]["high"]
    assert full["parts"][2]["episodes"] == 5 * 120 * 16 * 8             # five RL arms


def test_the_plan_file_agrees_with_the_trainer_settings_and_the_package():
    plan = json.loads(_read("experiments", "learnability", "plan.json"))
    env = dict(line.split("#")[0].strip().split("=", 1) for line in _read("experiments", "learnability", "overrides.env").splitlines()
               if "=" in line and not line.startswith("#"))
    a = plan["algorithm"]
    assert (int(env["GROUP_SIZE"]), int(env["PROMPTS_PER_STEP"]), int(env["TOTAL_STEPS"]), int(env["MAX_TURNS"])) == \
        (a["group_size"], a["groups_per_step"], a["steps"], a["max_turns"])
    assert float(env["LR"]) == a["lr"] and int(env["LORA_RANK"]) == plan["model"]["lora_rank"] and env["KL_COEF"] == "0"
    assert env["MRGYM_ENV"] == plan["environment"]["train"] and env["MRGYM_LEVELS"] == ",".join(map(str, plan["environment"]["train_levels"]))
    lo, hi = (int(x) for x in env["TRAIN_SEEDS"].split("-"))
    assert lo == plan["curriculum"]["first_seed"] and hi - lo + 1 == a["groups_per_step"] * a["steps"]
    assert a["group_size"] == learn.GROUP_SIZE and plan["status"].startswith("not launched")
    c = plan["curriculum"]
    cur = Curriculum([(plan["environment"]["train"], lv) for lv in plan["environment"]["train_levels"]], mode=c["mode"], alpha=c["alpha"],
                     tau=c["tau"], floor=c["floor"], first_seed=c["first_seed"])
    assert cur.sample(1)[0]["seed"] == c["first_seed"] > 9063            # training seeds lie above the calibration and dev seeds
    import mrgym
    spec = mrgym.spec(plan["environment"]["train"])
    assert set(plan["environment"]["train_levels"]) | {plan["environment"]["never_trained_level"]} == set(spec.levels)
    doc = _read("experiments", "PREREG-learnability.md")
    assert "not launched" in doc and "mrgym check Scanner/NullTissue-v0 --learnability" in doc
