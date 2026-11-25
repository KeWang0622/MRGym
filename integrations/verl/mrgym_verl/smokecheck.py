"""One-line verdict on a trainer run from its console log and the episode log. Used by the node scripts after every
trainer run, so that a failure names its cause instead of a stack trace.

    python -m mrgym_verl.smokecheck --log run_a.log --episodes records/episodes.jsonl --expect-step 2 [--phase train]

Exit 0 and ``OK ...`` or exit 1 and ``FAIL <cause>``. Causes, in the order they are tested:

    OOM on load            an out-of-memory error and no episode was played: the weights or the rollout replicas did not fit
    OOM on first backward  an out-of-memory error after episodes were played and before any actor update was logged
    OOM later              an out-of-memory error after a completed update
    context overflow       at least ``--cut-max`` of the episodes ended because a reply was cut at the reply limit or
                           the stream no longer fitted the response length
    parser mismatch        more than half of the episodes made no tool call (the trainer's tool parser does not read
                           the model's format), or more than half of all calls were refused for their shape
    no step                the log has no ``step:<expect-step>`` line
    step labels            training episodes carry the sampler's step count instead of the trainer's (the trainer did
                           not pass ``global_steps`` to the agent loop): records would be mislabelled after a resume
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from . import stoprules

OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|Cannot allocate memory|No available memory for the cache blocks")


def check(log_text: str, rows: list[dict], expect_step: int | None = None, phase: str = "train", cut_max: float = 0.20) -> tuple[bool, str, dict]:
    eps = [r for r in rows if r.get("phase") == phase] or rows
    steps = sorted({int(x) for x in re.findall(r"step:(\d+) - ", log_text)})
    updated = "actor/grad_norm" in log_text
    seconds = [float(x) for x in re.findall(r"timing_s/step:([0-9.]+)", log_text)]
    facts = dict(episodes=len(eps), steps_logged=steps, seconds_per_step=seconds,
                 pass_rate=None if not eps else round(float(np.mean([bool(r.get("passed")) for r in eps])), 4),
                 no_tool_call_share=None if not eps else round(float(np.mean([not r.get("tool_calls") for r in eps])), 4),
                 interface_tax=None if not eps else round(float(np.mean([r.get("interface_tax") or 0.0 for r in eps])), 4),
                 length_cut_share=None if not eps else round(float(np.mean([r.get("aborted") == "length" or bool(r.get("cut_replies")) for r in eps])), 4),
                 tokens_out_mean=None if not eps else int(np.mean([r.get("tokens_out") or 0 for r in eps])),
                 context_tokens_max=max((r.get("context_tokens") or 0 for r in eps), default=0),
                 step_sources=sorted({str(r.get("step_source")) for r in eps}))
    m = OOM.search(log_text)
    if m:
        where = "on load: no episode was played, so the weights or the rollout replicas did not fit" if not rows else \
            "on the first backward: episodes were played and no actor update was logged; lower TOKENS_PER_GPU, MAX_RESPONSE_LEN or GPU_MEM_UTIL" \
            if not updated else f"after a completed update (last step logged: {steps[-1] if steps else None})"
        return False, f"OOM {where} ({m.group(0)})", facts
    if eps and facts["length_cut_share"] >= cut_max:
        return False, (f"context overflow: {100 * facts['length_cut_share']:.0f} % of the episodes were cut at the reply limit or by the response "
                       f"length (longest stream {facts['context_tokens_max']} tokens); raise MAX_RESPONSE_LEN / MAX_TOKENS_PER_TURN"), facts
    if eps and facts["no_tool_call_share"] > 0.5:
        return False, (f"tool-call parser mismatch: {100 * facts['no_tool_call_share']:.0f} % of the episodes made no tool call that the "
                       "trainer's parser could read (TOOL_FORMAT against the model's chat template)"), facts
    if eps and facts["interface_tax"] > 0.5:
        return False, f"tool-call parser mismatch: {100 * facts['interface_tax']:.0f} % of all calls were refused for their shape or vocabulary", facts
    if expect_step is not None and expect_step not in steps:
        return False, f"the log has no step {expect_step} (steps logged: {steps})", facts
    if phase == "train" and eps and facts["step_sources"] != ["trainer"]:
        return False, f"step labels: episodes carry step_source {facts['step_sources']}, not the trainer's global step", facts
    return True, "ok", facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--expect-step", type=int, default=None)
    ap.add_argument("--phase", default="train")
    ap.add_argument("--cut-max", type=float, default=0.20)
    a = ap.parse_args(argv)
    try:
        with open(a.log, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    ok, reason, facts = check(text, stoprules.load(a.episodes), a.expect_step, a.phase, a.cut_max)
    print(("OK " if ok else "FAIL ") + reason + " | " + json.dumps(facts))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
