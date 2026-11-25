"""Synthetic episode records of schema 2 for the tests of the results pipeline: a run folder that meets the published
design (300 instances of ``Scanner/NullTissue-v0`` on the ``report`` instances, two models with two episodes per
instance, every control policy once), written without playing an episode. Outcomes come from a fixed generator, so
the folder is the same on every machine. Nothing here is a result of any model."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from mrgym import manifest, record, results
from mrgym.agents import summarize

ENV = "Scanner/NullTissue-v0"
MODELS = {"model-a": 0.55, "model-b": 0.40}
SCRIPTED = {"reference": 0.99, "prior": 0.0, "fixed_recipe": 0.03, "brute_force": 0.02, "wrong_units": 0.0, "textbook": 0.30}
PLAN_UTC, RUN_UTC = "2026-12-01T00:00:00Z", "2026-12-02T08:00:00Z"
CONVERSATION = dict(
    schema=record.TRANSCRIPT_SCHEMA, canary=record.CANARY, harness_id="synthetic", complete=True,
    tools=[{"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}}],
    messages=[{"role": "system", "content": "synthetic session note"}, {"role": "user", "content": "synthetic task"},
              {"role": "assistant", "content": None, "turn": 1, "reasoning": "synthetic reasoning",
               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "submit", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "deliverable received"}],
    turns=[dict(turn=1, reasoning="synthetic reasoning", usage=dict(prompt_tokens=900, completion_tokens=40), seconds=1.5, attempts=1)])
CONVERSATION_TEXT = json.dumps(CONVERSATION, indent=1) + "\n"
CONVERSATION_SHA = hashlib.sha256(CONVERSATION_TEXT.encode()).hexdigest()


def instances(per_level: int = 100, levels=(1, 2, 3), prefix: str = "report"):
    return [(lv, f"{prefix}:{i}") for lv in levels for i in range(per_level)]


def _record(agent, kind, level, seed, episode, passed, plan_sha, utc=RUN_UTC, command="mrgym run-agent --synthetic"):
    model = kind != "scripted"
    rec = dict(
        schema=record.EPISODE_SCHEMA, agent=agent, agent_kind=kind, model=agent if model else None,
        params_sent=dict(temperature=1.0, max_tokens=4096) if model else {}, params_dropped=[],
        settings=dict(max_turns=20, calc=True, temperature=1.0, max_tokens=4096) if model else None,
        env_id=ENV, level=level, seed=seed, split="train", episode=episode, task_id=f"nulltissue-{level}-{seed}",
        engine="mrgym-engine-test", package_version="0.0.0", status="completed", stop_reason="submitted",
        reward=float(passed), passed=bool(passed), error=0.01 if passed else 0.4, gates={}, checks=dict(deliverable_valid=True, measured=True, nulled_R1=bool(passed)),
        scans=4, scan_s_used=120.0, scan_s_budget=200.0, turns=1 if model else None, tool_calls=1, invalid_calls=0,
        tokens_in=900 if model else None, tokens_out=40 if model else None, tokens_estimated=False, wall_s=2.0, utc=utc,
        harness_id="synthetic", harness=dict(kind="chat-tools" if model else "scripted"), canary=record.CANARY, command=command,
        plan_sha256=plan_sha, group=dict(env_id=ENV, level=level, split="train", task_id=f"nulltissue-{level}-{seed}"),
        max_turns=20 if model else None, adapter_sha256=None, provider="synthetic.invalid" if model else None,
        sampler=dict(requested=dict(temperature=1.0, max_tokens=4096), sent=dict(temperature=1.0, max_tokens=4096), dropped=[]) if model else None,
        deliverable={"ti_ms": {"R1": 400.0}}, outcome="pass" if passed else "fail:nulled_R1",
        ladder=dict(submitted=True, valid=True, measured=True, passed=bool(passed)), distance=dict(error=0.01 if passed else 0.4),
        transcript=dict(file=record.TRANSCRIPT_FILE, schema=record.TRANSCRIPT_SCHEMA, sha256=CONVERSATION_SHA,
                        bytes=len(CONVERSATION_TEXT), complete=True, messages=4, assistant_turns=1, tool_results=1, reasoning_chars=19) if model else None)
    rec["_dir"] = f"episodes/{'policy-' if kind == 'scripted' else ''}{agent}/Scanner-NullTissue-v0/L{level}/s{seed.replace(':', '_')}/e{episode}"
    return record.upgrade(rec)


def records(per_level: int = 100, episodes: int = 2, plan_sha=None, models=None, seed: int = 20261125) -> list[dict]:
    """Completed records of the published design. Every instance has a difficulty shared by all agents, so episodes of
    one instance are correlated and the two models are correlated."""
    rng = np.random.default_rng(seed)
    out = []
    inst = instances(per_level)
    shift = rng.normal(0.0, 0.8, len(inst))
    for agent, p in (MODELS if models is None else models).items():
        logit = np.log(p / (1 - p)) + shift
        for (lv, s), z in zip(inst, logit):
            for e in range(episodes):
                out.append(_record(agent, "model", lv, s, e, rng.random() < 1 / (1 + np.exp(-z)), plan_sha))
    for agent, p in SCRIPTED.items():
        for lv, s in inst:
            out.append(_record(agent, "scripted", lv, s, 0, rng.random() < p, plan_sha))
    return out


def plan(models=None) -> dict:
    return results.new_plan(list(MODELS if models is None else models), [ENV], registered_utc=PLAN_UTC)


def plan_with_hash(models=None) -> dict:
    p = plan(models)
    return dict(p, sha256=hashlib.sha256((json.dumps(p, indent=1) + "\n").encode()).hexdigest())


def oracle(recs) -> dict:
    return {ENV: dict(kind="difference", engine="mrgym-engine-test", instances={summarize.instance_key(r): 1e-4 for r in recs})}


def summary(recs=None, the_plan="default", with_oracle=True) -> dict:
    """``results.json`` of the synthetic run, built in memory."""
    p = plan_with_hash() if the_plan == "default" else the_plan
    recs = records(plan_sha=None if p is None else p["sha256"]) if recs is None else recs
    return summarize.summarise(recs, p, None, oracle(recs) if with_oracle else None)


def write(directory: str, recs=None, with_plan: bool = True, with_oracle: bool = True) -> list[dict]:
    """Write the synthetic run as a results folder: records, conversations, the plan and the oracle values."""
    os.makedirs(directory, exist_ok=True)
    sha = None
    if with_plan:
        text = json.dumps(plan(), indent=1) + "\n"
        with open(os.path.join(directory, summarize.PLAN_FILE), "w", encoding="utf-8") as f:
            f.write(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
    recs = records(plan_sha=sha) if recs is None else recs
    for r in recs:
        d = os.path.join(directory, r["_dir"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in r.items() if k not in ("_dir", "record_schema", "tier")}, f)
        if r["agent_kind"] != "scripted":
            with open(os.path.join(d, record.TRANSCRIPT_FILE), "w", encoding="utf-8") as f:
                f.write(CONVERSATION_TEXT)
    if with_oracle:
        with open(os.path.join(directory, summarize.ORACLE_FILE), "w", encoding="utf-8") as f:
            json.dump(oracle(recs), f)
    return recs


def report_hash() -> str:
    return manifest.build("report", [ENV])["environments"][ENV]["seed_set_sha256"]


# ------------------------------------------------------------------ fixtures of the reporting rules
def compact(s: dict) -> dict:
    """A results.json without the parts the rules do not read (keeps the fixture files small)."""
    s = {k: v for k, v in s.items() if k not in ("reliability", "model_settings")}
    s["rows"] = [{k: v for k, v in r.items() if k not in ("by_level", "stop_reasons")} for r in s["rows"]]
    return s


def _row(s: dict, agent: str) -> dict:
    return next(r for r in s["rows"] if r["agent"] == agent)


def lint_fixtures() -> dict:
    """{file name: text}: one valid results.json, one valid page, and one file per reporting rule with exactly one
    kind of violation. ``tests/data/lint/`` holds these files; a test checks that they are what this function gives."""
    base = compact(summary())
    out = {"valid.json": json.dumps(base, indent=1) + "\n"}

    def variant(name, change):
        s = json.loads(json.dumps(base))
        change(s)
        out[name] = json.dumps(s, indent=1) + "\n"

    def pilot(s):
        _row(s, "model-b").update(tier="pilot", tier_reason="< 100 instances", instances=60)

    def no_plan(s):
        s.update(plan=None, plan_sha256=None)

    variant("R02_unit_is_the_instance.json", lambda s: _row(s, "model-a").update(ci_method="wilson"))
    variant("R04_seed_sets_differ.json", lambda s: _row(s, "model-b").update(seed_set_sha256="0" * 64))
    variant("R05_no_command.json", lambda s: _row(s, "model-a").update(command=None))
    variant("R07_pilot_in_comparison.json", pilot)
    variant("R08_no_ablation_row.json", lambda s: s["rows"].remove(_row(s, "textbook")))
    variant("R10_infra_errors.json", lambda s: _row(s, "model-a").update(infra_errors=30, infra_error_rate=0.0476))
    variant("R11_no_plan.json", no_plan)
    variant("R12_no_cost.json", lambda s: _row(s, "model-a").update(tokens_in_total=None, tokens_out_total=None))
    variant("R13_no_conversation.json", lambda s: _row(s, "model-a").update(records_missing={"transcript": 4}))
    variant("D01_too_few_instances.json", lambda s: _row(s, "model-a").update(instances=240))
    page = "# A page\n\nThe table is generated.\n\n"
    out["valid.md"] = page + ("The reference passes <!-- STAT:row:Scanner/NullTissue-v0:reference -->298/300<!-- /STAT -->. "
                              "A level is admitted when the reference passes at least 90 % of the instances.\n"
                              "model-a is ahead of model-b <!-- CMP:NullTissue-v0:model-a-vs-model-b -->.\n")
    out["R01_rate_typed_by_hand.md"] = page + "The reference policy passes 24/24 episodes on the second level.\n"
    out["R03_claim_without_test.md"] = page + "Level 2 is harder than level 1.\n"
    out["R06_certainty_without_bound.md"] = page + "The reference always passes.\n"
    out["R09_selected_subset.md"] = page + ("On admitted instances the reference passes "
                                            "<!-- STAT:row:Scanner/NullTissue-v0:reference -->298/300<!-- /STAT -->.\n")
    out["M1_no_model_has_been_run.md"] = page + "These rows are scripted control policies only. No language model has been run yet.\n"
    out["M4_same_instances.md"] = page + "Rows of kind model are language models, and every agent plays the same instances.\n"
    return out
