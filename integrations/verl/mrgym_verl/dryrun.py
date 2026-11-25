"""CPU dry run of the verl adapter: everything of the trainer path that needs no GPU, with verl's own code.

    python -m mrgym_verl.dryrun --tokenizer <dir with the model's tokenizer files> --model-type qwen3_5 \
        --overrides overrides.txt --cells "Scanner/NoWrap-v1:2 Scanner/SNRSpec-v0:1" --seeds 100-101 --out out/dry

What runs
  1. Hydra composes verl's ``ppo_trainer`` config with the overrides of ``overrides.sh`` (a wrong key fails here).
  2. ``MRGymAgentLoop`` is built the way verl's worker builds it (``hydra.utils.instantiate`` on the entry of
     ``agent_loops.yaml``) with the real tokenizer and verl's Continuous Token builder for the model family.
  3. Episodes: every cell x seed x scripted behaviour, played on one event loop. The "server" is ``ScriptedServer``:
     it returns the token ids of a scripted assistant turn in the model's tool-call format with made-up log-probs.
     Tool parsing, ``env.step``, context tokens, masks, rewards, records and the episode log are the real code path.
     Behaviours: every listed control policy of the environment (the calls it makes on the same instance), and
     ``text_only``, ``bad_json`` (cut-off arguments, then the reference), ``cut`` (a first reply cut by the token
     limit), ``context_cap`` (played with a response length too small for the episode).
  4. Checks per trajectory: lengths agree; the mask is 1 exactly on the scripted tokens; log-probs are 0 off the mask;
     the reward equals the pass of the same calls played straight on the environment; the token stream equals a
     from-scratch chat-template rendering of the final conversation; the stored record lacks no required field.
  5. One training row per (cell, seed) goes through the curriculum hook instead of naming its instance.
  6. verl's GRPO advantage per group, and (``--grad-step``) verl's PPO policy loss on a tiny random causal LM with a
     LoRA adapter: one optimizer step, adapter saved, reloaded, logits compared. The sequences are cropped and their
     token ids mapped to a compact vocabulary for that step (a 1-layer model on CPU, a few megabytes); it shows that
     the tensors fit verl's loss, nothing about learning.
  7. The checkpoint callback copies a fake ``global_step_N`` folder and writes ``LATEST``.
  8. ``--ray``: the curriculum hook as a named Ray actor, asked from two worker processes.
  9. Token lengths with this tokenizer: first prompt, tool schemas, each observation, per cell (for ``max_model_len``).

What does not run: vLLM, the FSDP actor, weight sync into the rollout engine, verl's TransferQueue and trainer loop.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import time

import numpy as np
import torch

import mrgym
from mrgym import policies
from mrgym import record as mrecord
from mrgym.agents.runner import parse_seeds, policy_recipe

from . import AGENT_NAME, instances, stoprules
from .episode import Episode

FAILURES = ("text_only", "bad_json", "cut", "context_cap")
_CALLS: dict = {}


# ------------------------------------------------------------------ a scripted "model"
def policy_calls(env_id: str, level: int, seed: int, policy: str, split: str = "train") -> list[tuple[str, dict]]:
    key = (env_id, level, seed, policy, split)
    if key not in _CALLS:
        env = mrgym.make(env_id, level=level, seed=seed, split=split)
        policies.run(env, policy, recipe=policy_recipe(env.spec, level))
        _CALLS[key] = [(e["call"]["name"], json.loads(json.dumps(e["call"]["arguments"], default=float))) for e in env.transcript()[1:]]
        env.close()
    return _CALLS[key]


def script(env_id: str, level: int, seed: int, behaviour: str, split: str = "train") -> list[tuple[str, list]]:
    """[(text before the calls, [(name, arguments or a raw string)])], one entry per reply."""
    if behaviour == "text_only":
        return [("The answer follows from the task statement alone.", [])] * 4
    ref = policy_calls(env_id, level, seed, "reference", split)
    if behaviour == "bad_json":
        return [("Scanning.", [(ref[0][0], '{"sequence": "spgr", "tr_ms": 20, ')])] + [(f"Step {i + 2}.", [c]) for i, c in enumerate(ref)]
    if behaviour in ("cut", "context_cap"):
        return [(f"Step {i + 1}: {c[0]}.", [c]) for i, c in enumerate(ref)]
    return [(f"Step {i + 1}: {c[0]}.", [c]) for i, c in enumerate(policy_calls(env_id, level, seed, behaviour, split))]


def render_call(name: str, arguments, fmt: str) -> str:
    if fmt == "hermes":
        if isinstance(arguments, str):
            return '<tool_call>\n{"name": "' + name + '", "arguments": ' + arguments + "\n</tool_call>"
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"
    if isinstance(arguments, str):                                       # cut-off JSON as the text of the first parameter
        params = f"<parameter=protocol>\n{arguments}\n</parameter>\n"
    else:
        params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in arguments.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def render_turn(turn: tuple[str, list], fmt: str) -> str:
    text, calls = turn
    calls_text = "\n".join(render_call(n, a, fmt) for n, a in calls)
    sep = "\n\n" if fmt == "qwen3_coder" else "\n"
    return (text + (sep if calls_text else "") if text else "") + calls_text


class ScriptedServer:
    """Stands in for verl's ``LLMServerClient``: ``generate(request_id, prompt_ids=, sampling_params=)``."""

    def __init__(self, tokenizer, fmt: str):
        from verl.workers.rollout.replica import TokenOutput
        self.TokenOutput, self.tok, self.fmt = TokenOutput, tokenizer, fmt
        self.turn = collections.Counter()
        self.eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.sampled: dict = collections.defaultdict(list)

    async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
        env_id, level, seed, behaviour, split = sampling_params["_script"]
        turns = script(env_id, level, seed, behaviour, split)
        k = self.turn[request_id]
        self.turn[request_id] += 1
        text = render_turn(turns[min(k, len(turns) - 1)], self.fmt)
        if self.tok.decode(prompt_ids[-6:]).endswith("<think>\n"):       # reasoning is on: the reply starts inside the block
            text = "I will use the scanner.\n</think>\n\n" + text
        ids = self.tok.encode(text, add_special_tokens=False) + [self.eos]
        stop = "completed"
        if behaviour == "cut" and k == 0:                               # the token limit cuts the first reply before any call
            ids, stop = self.tok.encode("Let me think about the geometry of the object and the field of view", add_special_tokens=False), "length"
        ids = ids[: sampling_params["max_tokens"]]
        self.sampled[request_id].append(ids)
        await asyncio.sleep(0)
        return self.TokenOutput(token_ids=ids, log_probs=[-0.25] * len(ids), stop_reason=stop,
                                extra_fields={"global_steps": 0, "min_global_steps": 0, "max_global_steps": 0})


def replay(job, fmt: str, max_reply_tokens: int | None = None) -> tuple[dict, list, list]:
    """The same scripted turns played straight on the environment, without a tokenizer or a trainer."""
    env_id, level, seed, behaviour, split = job
    ep = Episode(env_id, level, seed, split, max_reply_tokens=max_reply_tokens)
    _, tools = ep.start()
    turns, k, done, n = script(env_id, level, seed, behaviour, split), 0, False, 0
    if behaviour == "cut":
        ep.abort("length")
        done = True
    while not done:
        text, calls = turns[min(k, len(turns) - 1)]
        cs = []
        for name, args in calls:
            if isinstance(args, str):
                if fmt == "hermes":                                       # the hermes parser drops a call it cannot read
                    continue
                args = {"protocol": args}                                # the XML parser passes the parameter text on
            n += 1
            cs.append({"id": f"call_{n}", "name": name, "arguments": json.dumps(args)})
        _, done = ep.assistant_turn(text, cs)
        k += 1
    out, messages = ep.result(), list(ep.messages)
    ep.close()
    return out, messages, tools


def compose(overrides: list[str]):
    import verl
    from hydra import compose as hcompose
    from hydra import initialize_config_dir
    cfg_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return hcompose(config_name="ppo_trainer", overrides=overrides)


# ------------------------------------------------------------------ the run
async def main_async(a) -> dict:
    import hydra
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap, _agent_loop_registry
    from verl.trainer.ppo import core_algos
    from verl.utils.dataset.rl_dataset import RLHFDataset

    from .agent_loop_view import template_view

    report: dict = {"args": vars(a), "checks": {}, "versions": {}}
    import transformers
    import verl
    report["versions"] = dict(verl=getattr(verl, "__version__", None), torch=torch.__version__, transformers=transformers.__version__,
                              mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID)
    overrides = [ln.strip() for ln in open(a.overrides, encoding="utf-8") if ln.strip()]
    cfg = compose(overrides)
    report["checks"]["hydra_compose"] = f"ok, {len(overrides)} overrides"
    rollout = cfg.actor_rollout_ref.rollout
    fmt = rollout.multi_turn.format
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for entry in OmegaConf.load(rollout.agent.agent_loop_config_path):
        _agent_loop_registry[entry.name] = entry
    server = ScriptedServer(tok, fmt)
    os.environ["MRGYM_RECORD_DIR"] = os.path.join(a.out, "records")
    os.environ.setdefault("RUN_NAME", "dryrun")

    def make_loop(response_length=None):
        c = cfg
        if response_length:
            c = OmegaConf.merge(cfg, OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": response_length}},
                                                       "data": {"max_response_length": response_length}}))
        return hydra.utils.instantiate(
            config=_agent_loop_registry[AGENT_NAME], trainer_config=DictConfigWrap(config=c), server_manager=server, tokenizer=tok,
            processor=None, hf_model_type=a.model_type, dataset_cls=RLHFDataset, data_config=DictConfigWrap(c.data), tools=ToolListWrap([]))

    loop0 = make_loop()
    report["builder"] = type(loop0.continuous_token_builder).__name__
    report["loop"] = dict(signal=loop0.signal, max_tokens_per_turn=loop0.max_tokens_per_turn, tool_call_limit=loop0.tool_call_limit,
                          response_length=loop0.response_length, tool_format=fmt)

    cells, seeds = instances.parse_arms(a.cells), parse_seeds(a.seeds)
    jobs = []
    for env_id, level in cells:
        names = [p for p in a.policies.split(",") if p in policies.policy_names(mrgym.spec(env_id))] + list(FAILURES)
        jobs += [(env_id, level, s, b, "train") for s in seeds for b in names]
        jobs += [(env_id, level, seeds[0], "reference", "dev")]
    sampling = dict(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, logprobs=True)

    gate = asyncio.Semaphore(4)                                          # a loop object per episode costs memory: four at a time

    async def one(i, job):
        env_id, level, seed, behaviour, split = job
        async with gate:
            loop = make_loop(a.small_response if behaviour == "context_cap" else None)
            out = await loop.run({**sampling, "_script": job}, raw_prompt=[{"role": "user", "content": "placeholder"}],
                                 uid=f"{env_id}|{level}|{seed}|{split}", session_id=i, global_steps=1,
                                 extra_info=dict(phase="train" if split == "train" else "dev", env_id=env_id, level=level, seed=seed, split=split))
        return job, out, None

    t0 = time.time()
    results = await asyncio.gather(*[one(i, j) for i, j in enumerate(jobs)])
    report["episodes"], report["wall_s"] = len(results), round(time.time() - t0, 2)

    # ---- per-trajectory checks
    rows, bad = [], collections.Counter()
    for job, out, _loop in results:
        n_p, n_r = len(out.prompt_ids), len(out.response_ids)
        mask, lp = np.array(out.response_mask), np.array(out.response_logprobs if out.response_logprobs else [0.0] * n_r)
        direct, _, _ = replay(job, fmt, loop0.max_tokens_per_turn)
        extra = out.extra_fields
        ok_len = len(mask) == n_r == len(lp)
        ok_lp = bool(np.all(lp[mask == 0] == 0.0) and np.all(lp[mask == 1] == -0.25))
        if job[3] == "context_cap":                                      # the trainer cut it: no pass, labelled as such
            ok_reward = out.reward_score == 0.0 and extra["reward_extra_info"]["context_cap"] == 1.0 and extra["mrgym_outcome"] == "no_submission_length"
        else:
            ok_reward = out.reward_score == direct["signal"] == float(direct["passed"]) and extra["mrgym_outcome"] == direct["outcome"]
        bad["length"] += not ok_len
        bad["logprob_alignment"] += not ok_lp
        bad["reward_vs_direct"] += not ok_reward
        rows.append(dict(cell=f"{job[0]}/L{job[1]}", seed=job[2], behaviour=job[3], split=job[4], prompt_tokens=n_p, response_tokens=n_r,
                         loss_tokens=int(mask.sum()), reward=out.reward_score, outcome=extra["mrgym_outcome"], stop=extra["mrgym_stop_reason"],
                         turns=int(extra["reward_extra_info"]["turns"]), direct_outcome=direct["outcome"], ok=ok_len and ok_lp and ok_reward))
    report["checks"]["trajectory_failures"] = dict(bad)
    report["rows"] = rows

    # ---- sampled tokens are kept verbatim and are exactly the masked ones
    all_masked = sorted(tuple(t for t, m in zip(out.response_ids, out.response_mask) if m) for _, out, _ in results)
    all_sampled = sorted(tuple(sum(v, [])) for v in server.sampled.values())
    report["checks"]["masked_tokens_equal_sampled_tokens"] = all_sampled == all_masked

    # ---- incremental stream against a from-scratch rendering of the final conversation
    same = diff = 0
    first_diff = None
    kwargs = dict(cfg.data.get("apply_chat_template_kwargs", {}) or {})
    strip = lambda t: re.sub(r"<think>.*?</think>\s*", "<think></think>", t, flags=re.DOTALL)   # noqa: E731
    for job, out, _ in results:
        if job[3] in ("cut", "context_cap"):
            continue
        _, messages, tools = replay(job, fmt, loop0.max_tokens_per_turn)
        while messages and messages[-1]["role"] != "assistant":          # the stream ends with the last reply of the model
            messages.pop()
        full = tok.apply_chat_template(template_view(messages), tools=tools, tokenize=False, add_generation_prompt=False, **kwargs)
        stream = tok.decode(out.prompt_ids + out.response_ids)
        if strip(full).strip() == strip(stream).strip():
            same += 1
        else:
            diff += 1
            if first_diff is None:
                x, y = strip(full), strip(stream)
                i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                first_diff = dict(job=list(job), at_char=i, template=x[max(0, i - 160): i + 200], stream=y[max(0, i - 160): i + 200])
    report["checks"]["stream_vs_full_template"] = dict(identical=same, different=diff, first_difference=first_diff)

    # ---- records and the episode log
    missing, n_rec = collections.Counter(), 0
    for root, _, files in os.walk(os.path.join(a.out, "records")):
        if "record.json" in files:
            with open(os.path.join(root, "record.json"), encoding="utf-8") as f:
                rec = json.load(f)
            n_rec += 1
            for m in mrecord.missing(rec):
                missing[m] += 1
            problem = mrecord.transcript_problem(rec, root)
            if problem:
                missing["transcript: " + problem] += 1
            for k in ("signal_mode", "signal", "paid_reward", "steps", "phase", "step", "uid", "session", "context_tokens"):
                if k not in rec.get("training", {}):
                    missing["training." + k] += 1
    log = stoprules.load(os.path.join(a.out, "records", "episodes.jsonl"))
    report["checks"]["records"] = dict(written=n_rec, expected=len(results), log_lines=len(log), missing_fields=dict(missing))

    # ---- the curriculum hook behind a training row (this process; --ray checks the actor)
    hook_cfg = os.path.join(a.out, "curriculum.json")
    instances.write_config(hook_cfg, cells, group_size=2, groups_per_step=len(cells), state_path=os.path.join(a.out, "curriculum_state.json"))
    os.environ[instances.ENV_VAR] = hook_cfg
    cur = []
    for g in range(len(cells)):
        for s in range(2):
            loop = make_loop()
            hook = instances.client()
            env_id, level, seed, _ = await hook.assign(f"g{g}")
            out = await loop.run({**sampling, "_script": (env_id, level, seed, "reference" if s else "prior", "train")},
                                 raw_prompt=[{"role": "user", "content": "placeholder"}], uid=f"g{g}", session_id=s, global_steps=2,
                                 extra_info=dict(phase="train", split="train", curriculum=True, index=g))
            cur.append((g, out.extra_fields["mrgym_env_id"], out.extra_fields["mrgym_level"], out.extra_fields["mrgym_seed"], out.reward_score))
    info = await instances.client().info()
    seeds_per_group = {g: {c[3] for c in cur if c[0] == g} for g in range(len(cells))}
    report["checks"]["curriculum_rows"] = dict(
        one_instance_per_group=all(len(v) == 1 for v in seeds_per_group.values()),
        distinct_seeds=len({next(iter(v)) for v in seeds_per_group.values()}) == len(cells),
        sampler_steps=info["steps"], seeds_issued=info["issued"], state_file=os.path.exists(os.path.join(a.out, "curriculum_state.json")))

    # ---- training samples: tensors, padded batch, GRPO advantage per group (verl's own function)
    train = [(j, o) for j, o, _ in results if j[4] == "train"]
    dicts = [o.as_dict() for _, o in train]
    width = max(d["responses"].numel() for d in dicts)
    resp_mask, rm = torch.zeros(len(dicts), width, dtype=torch.int64), torch.zeros(len(dicts), width, dtype=torch.float32)
    for i, d in enumerate(dicts):
        n = d["responses"].numel()
        resp_mask[i, :n] = d["response_mask"]
        rm[i, :n] = d["rm_scores"]
    index = np.array([f"{j[0]}|{j[1]}|{j[2]}" for j, _ in train], dtype=object)
    adv_cfg = OmegaConf.create({"norm_adv_by_std_in_grpo": bool(cfg.algorithm.norm_adv_by_std_in_grpo)})
    adv, _ = core_algos.get_adv_estimator_fn("grpo")(token_level_rewards=rm, response_mask=resp_mask, index=index, config=adv_cfg)
    groups = collections.defaultdict(list)
    for i, (j, o) in enumerate(train):
        a_i = float(adv[i][resp_mask[i] > 0][0]) if resp_mask[i].sum() else 0.0
        groups[f"{j[0]}/L{j[1]}/s{j[2]}"].append((j[3], o.reward_score, round(a_i, 3)))
    report["groups"] = dict(groups)
    report["checks"]["groups_with_reward_spread"] = f"{sum(len({r for _, r, _ in g}) > 1 for g in groups.values())}/{len(groups)}"
    report["checks"]["advantage_zero_off_mask"] = bool(torch.all(adv[resp_mask == 0] == 0))
    report["tensor_keys"] = sorted(dicts[0])

    # ---- token lengths with this tokenizer
    report["tokens"] = token_table(tok, cells, seeds, kwargs, template_view, loop0.max_tokens_per_turn)

    if a.grad_step:
        report["checks"]["grad_step"] = grad_step(dicts, adv, resp_mask, cfg, len(tok), a)

    # ---- checkpoint callback: copy + LATEST
    from .checkpoint import SyncCheckpoint
    from .store import latest_checkpoint
    src = os.path.join(a.out, "fake_ckpt", "global_step_5")
    os.makedirs(os.path.join(src, "actor"), exist_ok=True)
    with open(os.path.join(src, "actor", "adapter_model.safetensors"), "wb") as f:
        f.write(b"x" * 16)
    os.environ["MRGYM_CKPT_DEST"] = os.path.join(a.out, "remote")
    SyncCheckpoint(cfg).on_save(trainer=None, global_step=5, checkpoint_dir=src)
    report["checks"]["checkpoint_callback"] = dict(latest_checkpoint(os.environ["MRGYM_CKPT_DEST"]),
                                                   curriculum_state_in_checkpoint=os.path.exists(os.path.join(a.out, "remote", "global_step_5", "curriculum_state.json")))
    if a.ray:
        report["checks"]["ray_actor"] = ray_check(hook_cfg)
    return report


def token_table(tok, cells, seeds, template_kwargs, template_view, max_reply_tokens: int | None = 16384) -> dict:
    """Tokens of the parts of an episode that do not depend on the model, per cell: the first prompt (system + task +
    tool schemas), the tool schemas alone, each observation of the reference policy, the whole reference episode
    with empty replies around its calls, and the first prompt plus a median observation for every allowed tool call
    (what the context holds besides the model's own text when an episode uses its whole tool-call limit)."""
    out = {}
    for env_id, level in cells:
        first, tools_n, obs, calls, whole = [], [], [], [], []
        for seed in seeds:
            ep = Episode(env_id, level, seed, max_reply_tokens=max_reply_tokens)     # the task text names the reply limit
            messages, tools = ep.start()
            first.append(len(tok.encode(tok.apply_chat_template(template_view(messages), tools=tools, tokenize=False, add_generation_prompt=True,
                                                                **template_kwargs), add_special_tokens=False)))
            tools_n.append(len(tok.encode(json.dumps(tools), add_special_tokens=False)))
            for i, (name, args) in enumerate(policy_calls(env_id, level, seed, "reference")):
                appended, _ = ep.assistant_turn("", [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}])
                obs += [len(tok.encode(m["content"] or "", add_special_tokens=False)) for m in appended if m["role"] == "tool"]
            calls.append(ep.tool_calls)
            whole.append(len(tok.encode(tok.apply_chat_template(template_view(ep.messages), tools=tools, tokenize=False, add_generation_prompt=False,
                                                                **template_kwargs), add_special_tokens=False)))
            ep.close()
        out[f"{env_id}/L{level}"] = dict(
            first_prompt_max=max(first), tool_schemas=max(tools_n), observation_median=int(np.median(obs)), observation_max=max(obs),
            reference_calls_max=max(calls), tool_call_limit=mrgym.spec(env_id).max_steps, reference_episode_without_model_text_max=max(whole),
            observations_if_every_call_used=max(first) + mrgym.spec(env_id).max_steps * int(np.median(obs)))
    return out


def grad_step(dicts, adv, resp_mask, cfg, vocab: int, a) -> dict:
    """verl's PPO policy loss on a tiny random causal LM with a LoRA adapter: one step, save, reload, compare."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
    from verl.trainer.ppo import core_algos
    torch.manual_seed(0)
    crop_p, crop_r = 64, 256
    keep = [i for i, d in enumerate(dicts) if d["responses"].numel() > 0][:8]
    ids, mask, advs = [], [], []
    for i in keep:
        d = dicts[i]
        p, r = d["prompts"][-crop_p:], d["responses"][:crop_r]
        ids.append(torch.cat([p, r]))
        pad = crop_p + crop_r - ids[-1].numel()
        ids[-1] = torch.nn.functional.pad(ids[-1], (0, pad))
        n = r.numel()
        mask.append(torch.nn.functional.pad(resp_mask[i, :n].float(), (0, crop_r - n)))
        advs.append(torch.nn.functional.pad(adv[i, :n].float(), (0, crop_r - n)))
    ids, mask, advs = torch.stack(ids), torch.stack(mask), torch.stack(advs)
    used, ids = torch.unique(ids, return_inverse=True)                     # a compact vocabulary: the logits of a 248k-entry
    vocab = int(used.numel())                                               # vocabulary would take gigabytes even for a toy model
    config = GPT2Config(vocab_size=vocab, n_positions=crop_p + crop_r, n_embd=32, n_layer=1, n_head=2, resid_pdrop=0.0, embd_pdrop=0.0,
                        attn_pdrop=0.0)
    model = get_peft_model(GPT2LMHeadModel(config), LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], fan_in_fan_out=True))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    def log_prob():
        logits = model(input_ids=ids).logits[:, crop_p - 1:-1]              # position t predicts response token t
        return torch.log_softmax(logits, -1).gather(-1, ids[:, crop_p:].unsqueeze(-1)).squeeze(-1)

    from verl.utils.config import omega_conf_to_dataclass
    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)       # the dataclass verl's actor hands to its loss
    loss_fn = core_algos.get_policy_loss_fn("vanilla")
    old_lp = log_prob().detach()        # the sampler's log-probs are those of the current weights (bypass mode); the scripted -0.25 are not
    loss, metrics = loss_fn(old_log_prob=old_lp, log_prob=log_prob(), advantages=advs, response_mask=mask,
                            loss_agg_mode=actor_cfg.loss_agg_mode, config=actor_cfg)
    opt.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
    opt.step()
    after = log_prob().detach()
    folder = os.path.join(a.out, "tiny_ckpt", "global_step_1", "actor", "lora_adapter")
    model.save_pretrained(folder)
    torch.manual_seed(0)
    again = PeftModel.from_pretrained(GPT2LMHeadModel(config), folder)
    again.base_model.model.load_state_dict({k: v for k, v in model.base_model.model.state_dict().items() if "lora_" not in k}, strict=False)
    model = again
    restored = log_prob().detach()
    return dict(sequences=len(keep), cropped_to=[crop_p, crop_r], compact_vocabulary=vocab, loss=float(loss), grad_norm=grad_norm, loss_tokens=int(mask.sum()),
                finite=bool(torch.isfinite(loss)) and grad_norm > 0, step_changed_log_probs=float((after - old_lp).abs().max()), trainable_params=sum(p.numel() for p in opt.param_groups[0]["params"]),
                restore_max_abs_diff=float((after - restored).abs().max()), adapter_files=sorted(os.listdir(folder)),
                metric_keys=sorted(metrics)[:6])


def ray_check(hook_cfg: str) -> dict:
    """The hook as a named actor: two worker processes ask for the same group and for another one."""
    import ray
    ray.init(num_cpus=2, include_dashboard=False, namespace=instances.ACTOR_NAMESPACE, log_to_driver=False,
             object_store_memory=80 * 1024 * 1024, _system_config={"automatic_object_spilling_enabled": False})
    try:
        @ray.remote(num_cpus=1)
        def ask(uid: str, cfg: str):
            import asyncio as aio
            import os as _os

            from mrgym_verl import instances as inst
            got = aio.run(inst.client(cfg).assign(uid))
            aio.run(inst.client(cfg).report(uid, 1.0, True))
            return _os.getpid(), got

        a1, a2, b1 = ray.get([ask.remote("ray-A", hook_cfg), ask.remote("ray-A", hook_cfg), ask.remote("ray-B", hook_cfg)])
        info = ray.get(ray.get_actor(instances.ACTOR_NAME, namespace=instances.ACTOR_NAMESPACE).info.remote())
        return dict(worker_pids=sorted({a1[0], a2[0], b1[0]}), driver_pid=os.getpid(), same_group_same_instance=a1[1][:3] == a2[1][:3],
                    other_group_other_seed=a1[1][2] != b1[1][2], issued=info["issued"], restored=info["restored"], ray=ray.__version__)
    finally:
        ray.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model-type", required=True, help="config.json model_type, e.g. qwen3_5")
    ap.add_argument("--overrides", required=True, help="one Hydra override per line (launch.sh writes overrides.txt)")
    ap.add_argument("--cells", default="Scanner/NoWrap-v1:2")
    ap.add_argument("--seeds", default="100-101")
    ap.add_argument("--policies", default="reference,prior", help="control policies played as scripted behaviours")
    ap.add_argument("--small-response", type=int, default=600, help="response length of the context_cap behaviour")
    ap.add_argument("--grad-step", action="store_true")
    ap.add_argument("--ray", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-only", action="store_true", help="print the token table of --cells with this tokenizer and exit (no verl needed)")
    ap.add_argument("--chat-kwargs", default="", help='with --tokens-only: chat-template switches, "k=v k=v"')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.tokens_only:
        from transformers import AutoTokenizer

        from .agent_loop_view import template_view
        kw = {k: (v == "True" if v in ("True", "False") else v) for k, v in (x.split("=", 1) for x in a.chat_kwargs.split())}
        table = token_table(AutoTokenizer.from_pretrained(a.tokenizer), instances.parse_arms(a.cells), parse_seeds(a.seeds), kw, template_view)
        with open(os.path.join(a.out, "token_table.json"), "w", encoding="utf-8") as f:
            json.dump(table, f, indent=1)
        print("| Cell | First prompt | of which tool schemas | Observation median / max | Reference episode without model text | Tool-call limit | First prompt + a median observation per allowed call |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for cell, t in table.items():
            print(f"| {cell} | {t['first_prompt_max']:,} | {t['tool_schemas']:,} | {t['observation_median']:,} / {t['observation_max']:,} | "
                  f"{t['reference_episode_without_model_text_max']:,} | {t['tool_call_limit']} | {t['observations_if_every_call_used']:,} |")
        return 0
    report = asyncio.run(main_async(a))
    with open(os.path.join(a.out, "dryrun_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    c = report["checks"]
    ok = (not any(c["trajectory_failures"].values()) and c["masked_tokens_equal_sampled_tokens"] and c["advantage_zero_off_mask"]
          and c["records"]["written"] == c["records"]["expected"] and not c["records"]["missing_fields"]
          and c["curriculum_rows"]["one_instance_per_group"] and c["curriculum_rows"]["distinct_seeds"]
          and c["checkpoint_callback"]["global_step"] == 5 and c["stream_vs_full_template"]["different"] == 0
          and (not a.grad_step or (c["grad_step"]["finite"] and c["grad_step"]["restore_max_abs_diff"] < 1e-5))
          and (not a.ray or (c["ray_actor"]["same_group_same_instance"] and c["ray_actor"]["other_group_other_seed"])))
    print(json.dumps({k: report[k] for k in ("versions", "builder", "loop", "episodes", "wall_s", "tokens", "checks")}, indent=1, default=str))
    print("DRYRUN PASS" if ok else "DRYRUN FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
