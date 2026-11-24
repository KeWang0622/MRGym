"""The white-box loop a trainer runs: messages in, tool calls out, tool messages back (Chat Completions shapes).

``rollout(env, policy_fn)`` drives one episode with any callable ``policy_fn(messages, tools) -> assistant message``
(a model client, or a scripted stand-in in tests). It returns the message list and the final step: exactly what
ART's ``rollout()`` or TRL's ``environment_factory`` loop needs, with nothing MR-specific in it.
"""


def rollout(env, policy_fn, seed=None, system=None):
    obs, info = env.reset(seed=seed)
    tools = env.tools("openai")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": obs.text}]
    step = None
    while True:
        msg = policy_fn(messages, tools)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:                                           # the model stopped without submitting
            return messages, step, dict(reward=0.0, reason="no_tool_call", info=info)
        for call in calls:
            step = env.step(call)
            messages += step.observation.to_openai(call.get("id"))
            if step.done:
                return messages, step, dict(reward=step.reward, reason="truncated" if step.truncated else "submitted", info=info)
