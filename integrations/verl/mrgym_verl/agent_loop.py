"""verl agent loop for MR Gym: one in-process ``mrgym`` episode per trajectory, token-in / token-out.

Registered as ``mrgym_agent`` (``config/agent_loops.yaml``). The dataset row names the instance in ``extra_info``
(``mrgym_verl.instances``): a fixed one (``env_id``, ``level``, ``seed``, ``split``) for evaluation rows, or
``curriculum: true`` for a training row, whose instance the curriculum hook issues per group (verl gives the ``n``
rollouts of one prompt the same ``uid``). The row's ``prompt`` is a placeholder; the first messages come from
``env.reset()`` through ``mrgym_verl.episode.Episode``, so they are those of ``mrgym run-agent``.

What verl gets back (``AgentLoopOutput``)
    prompt_ids         the chat template applied once to [system, user] with ``tools=env.tools("openai")``
    response_ids       everything after it: sampled assistant tokens as generated, then tool results and harness notes
                       rendered by verl's Continuous Token builder, then the next sampled turn, ...
    response_mask      1 on sampled tokens, 0 elsewhere
    response_logprobs  sampler log-probs on sampled tokens (``rollout.calculate_log_probs=True``)
    reward_score       the trainer's signal: the environment's paid reward (1 for a pass, else 0) with ``signal: pass``;
                       ``info["score"]["progress"]`` with ``signal: progress`` (not the paid reward; see README)
    extra_fields       ``reward_extra_info``: numeric per-episode values for verl's logs (``acc`` = pass, ``paid_reward``,
                       ``progress``, ``submitted``, ``cut``, ``context_cap``, turns, calls, tokens); flat ``mrgym_*`` fields

Limits. There is no turn cap. An episode ends where the environment ends it (the terminal tool, its tool-call limit,
its rejected-call limit), after three replies without a tool call, or - the two ends only a token stream has - when
a reply is cut at ``max_tokens_per_turn`` before its end-of-turn token or the response no longer fits
``data.max_response_length``. Both of the latter are told to the environment (outcome ``no_submission_length``),
counted (``cut``, ``context_cap``) and watched by the stop rules. ``max_tokens_per_turn`` is stated in the task text
(the environment's ``harness.max_reply_tokens``).

Every finished episode is written as a record in the runner's schema with the whole conversation
(``MRGYM_RECORD_DIR``; nothing is written when it is unset) and as one line of ``episodes.jsonl``.

Options of the loop's entry in ``agent_loops.yaml``: ``signal`` (``pass`` | ``progress``), ``max_tokens_per_turn``,
``tool_call_limit`` (null: the environment's own), ``calc``. The tool-call format is the trainer's
(``actor_rollout_ref.rollout.multi_turn.format``: ``qwen3_coder`` for Qwen3.5 / Qwen3.8, ``hermes`` for Qwen3).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

from . import AGENT_NAME, instances, stoprules
from .agent_loop_view import template_view
from .episode import Episode, episode_dir, free_dir

logger = logging.getLogger(__name__)

NUMERIC_KEYS = ("progress", "turns", "tool_calls", "invalid_calls", "interface_tax", "tokens_in", "tokens_out")
FLAT_KEYS = ("outcome", "stop_reason", "env_id", "level", "seed", "split")


class _Prop:
    """Duck type of verl's ``OpenAIFunctionPropertySchema`` that keeps the raw JSON Schema of the property.

    verl's pydantic tool schema rejects a property without ``type`` (an ``anyOf``). The ``qwen3_coder`` parser only
    calls ``model_dump()`` and reads ``type`` to decide how to decode the parameter text; a property without a type
    is reported as ``object`` so that its text is decoded as JSON.
    """

    def __init__(self, schema: dict):
        self._schema = dict(schema)
        self._schema.setdefault("type", "object")

    def model_dump(self, **_):
        return dict(self._schema)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def parser_tools(openai_tools: list[dict]) -> list:
    """OpenAI tool dicts -> the attribute shape verl's tool parsers read."""
    out = []
    for t in openai_tools:
        f = t["function"]
        props = {k: _Prop(v) for k, v in (f.get("parameters") or {}).get("properties", {}).items()}
        out.append(_Obj(type="function", function=_Obj(name=f["name"], parameters=_Obj(properties=props))))
    return out


@register(AGENT_NAME)
class MRGymAgentLoop(AgentLoopBase):
    merges_context_messages = True      # tool results are appended to sampled tokens: verl checks the builder first

    def __init__(self, *args, tools=None, signal: str = "pass", max_tokens_per_turn: int = 16384,
                 tool_call_limit: int | None = None, calc: bool = True, name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)               # ``tools`` is verl's global tool list; not used here
        mt = self.rollout_config.multi_turn
        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        self.response_length = int(self.rollout_config.response_length)
        self.signal, self.max_tokens_per_turn, self.calc_tool = signal, int(max_tokens_per_turn), bool(calc)
        self.tool_call_limit = None if tool_call_limit in (None, "", 0) else int(tool_call_limit)
        self.record_dir = os.environ.get("MRGYM_RECORD_DIR") or None
        self.run_name = os.environ.get("RUN_NAME") or "run"
        self.model_name = os.environ.get("MRGYM_MODEL_NAME") or "policy"

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        info = dict(kwargs.get("extra_info") or {})
        uid, session = str(kwargs.get("uid") or uuid4().hex), int(kwargs.get("session_id") or 0)
        phase = str(info.get("phase") or "train")
        hook, hook_step = None, None
        if info.get("curriculum"):
            hook = instances.client()
            env_id, level, seed, hook_step = await hook.assign(uid)
            split = "train"
        else:
            env_id, level, seed, split = str(info["env_id"]), int(info["level"]), int(info["seed"]), str(info.get("split") or "train")
        step = kwargs.get("global_steps")
        step_source = "trainer"
        if step is None or int(step) < 0:                                # the trainer gave none: the sampler's own count
            step, step_source = (hook_step if hook_step is not None else -1), "curriculum"
        step = int(step)

        ep = Episode(env_id, level, seed, split, tool_call_limit=self.tool_call_limit, calc=self.calc_tool,
                     max_reply_tokens=self.max_tokens_per_turn, signal=self.signal)
        try:
            return await self._play(ep, sampling_params, dict(phase=phase, step=step, step_source=step_source, uid=uid, session=session), hook)
        finally:
            ep.close()

    async def _play(self, ep: Episode, sampling_params: dict, where: dict, hook) -> AgentLoopOutput:
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex                                         # sticky session: prefix cache on one replica
        messages, tools = await asyncio.to_thread(ep.start)             # world generation is CPU work
        p_tools = parser_tools(tools)

        ids = self._cap_text_prompt_length(await self.ct_build_initial_tokens(template_view(messages), tools=tools))
        n_prompt = len(ids)
        mask: list[int] = []
        logps: list[float] | None = None
        sp = dict(sampling_params)
        if self.tool_parser.stop_token_ids:
            sp["stop_token_ids"] = list(set((sp.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
        server_extra: dict[str, Any] = {}
        call_no, context_cap, done = 0, False, False

        while not done:
            room = self.response_length - len(mask)
            if room <= 16:                                              # no room for another reply
                ep.abort("length")
                context_cap = True
                break
            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=ids,
                    sampling_params={**sp, "max_tokens": min(self.max_tokens_per_turn, room)})
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
            if not server_extra:
                server_extra = dict(out.extra_fields or {})
            elif (out.extra_fields or {}).get("max_global_steps"):
                server_extra["max_global_steps"] = out.extra_fields["max_global_steps"]
            if logps is None and out.log_probs:
                logps = [0.0] * len(mask)
            n_request = len(ids)
            merged, mask, logps = await self.ct_merge_assistant_token(
                ids, out.token_ids, mask, logps, assistant_logprobs=out.log_probs if out.log_probs else None)
            ids = merged.token_ids

            content, parsed = await self.tool_parser.extract_tool_calls(out.token_ids, p_tools)
            calls = []
            for c in parsed:
                call_no += 1
                calls.append({"id": c.tool_call_id or f"call_{call_no}", "name": c.name, "arguments": c.arguments})
            if out.stop_reason == "length" and not calls:
                # A reply cut by the token cap has no end-of-turn token; appending context after it would train on a
                # sequence the template never produces. The episode ends here, reward 0, outcome no_submission_length.
                ep.abort("length", reply=dict(raw_text=self.tokenizer.decode(out.token_ids), prompt_tokens=n_request,
                                              completion_tokens=len(out.token_ids)))
                break

            n_before = len(ep.messages)
            raw_text = self.tokenizer.decode(out.token_ids) if self.record_dir else None
            with simple_timer("tool_calls", metrics):
                _, done = await asyncio.to_thread(
                    ep.assistant_turn, content, calls, finish="length" if out.stop_reason == "length" else None,
                    prompt_tokens=n_request, completion_tokens=len(out.token_ids), raw_text=raw_text)
            if done:
                break
            # The harness reply (tool results, then at most one note) becomes context tokens with mask 0.
            view = template_view(ep.messages)
            merged, new_mask, new_logps = await self.ct_merge_context_msg(
                view[: n_before + 1], view, ids, mask, logps, tools=tools)
            if len(new_mask) >= self.response_length:                   # the observation does not fit
                ep.abort("length")
                context_cap = True
                break
            ids, mask, logps = merged.token_ids, new_mask, new_logps

        result = await asyncio.to_thread(ep.result)
        response_ids = ids[n_prompt:]
        if len(response_ids) != len(mask):
            raise RuntimeError(f"token / mask length mismatch: {len(response_ids)} vs {len(mask)}")
        result["context_tokens"] = len(ids)
        if hook is not None:
            await hook.report(where["uid"], float(result["signal"]), bool(result["passed"]))
        if self.record_dir:
            await asyncio.to_thread(self._write, ep, result, where, sampling_params, context_cap)
        extra = dict(server_extra)
        extra["reward_extra_info"] = {
            "acc": float(result["passed"]), "paid_reward": float(result["reward"]), "submitted": float(result["submitted"]),
            "cut": float(ep.aborted == "length" and not context_cap), "context_cap": float(context_cap),
            "context_tokens": float(len(ids)), **{k: float(result[k] or 0.0) for k in NUMERIC_KEYS}}
        extra.update({f"mrgym_{k}": result[k] for k in FLAT_KEYS})
        extra.update({"turn_scores": [], "tool_rewards": []})          # same schema as verl's own loops
        return AgentLoopOutput(
            prompt_ids=ids[:n_prompt],
            response_ids=response_ids[: self.response_length],
            response_mask=mask[: self.response_length],
            response_logprobs=logps[: self.response_length] if logps else None,
            reward_score=float(result["signal"]),
            num_turns=len(ep.messages),
            metrics=metrics,
            extra_fields=extra,
        )

    def _write(self, ep: Episode, result: dict, where: dict, sampling_params: dict, context_cap: bool) -> None:
        sampler = {k: sampling_params.get(k) for k in ("temperature", "top_p", "top_k") if k in sampling_params}
        sampler["max_tokens"] = self.max_tokens_per_turn
        out = os.path.join(self.record_dir, where["phase"], f"step_{where['step']:06d}")
        d = free_dir(episode_dir(out, self.run_name, ep.env_id, ep.level, ep.seed, where["session"]), where["uid"])
        training = dict(where, attempt=os.environ.get("MRGYM_ATTEMPT", "0"), context_cap=context_cap, context_tokens=result["context_tokens"],
                        response_length=self.response_length)
        ep.write_record(d, agent=self.run_name, model=self.model_name, sampler=sampler, episode=where["session"], training=training,
                        base_url="inproc://verl-rollout", command=f"python -m verl.trainer.main_ppo (integrations/verl/launch.sh, run {self.run_name})")
        stoprules.append(os.path.join(self.record_dir, "episodes.jsonl"),
                         stoprules.line(result, phase=where["phase"], step=where["step"], step_source=where["step_source"], uid=where["uid"],
                                        session=where["session"]))
