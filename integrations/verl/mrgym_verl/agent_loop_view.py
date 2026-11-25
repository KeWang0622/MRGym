"""The conversation as chat templates read it. Kept apart from ``agent_loop`` so that it imports without verl."""
from __future__ import annotations

import json


def template_view(messages: list[dict]) -> list[dict]:
    """The conversation in the form chat templates expect: tool-call arguments as objects, content never None.

    ``mrgym`` keeps arguments as the JSON string the model wrote; the Qwen3.5 / Qwen3.8 templates iterate over them
    as a mapping. Only the builder's delta rendering reads this view: sampled tokens are kept as generated.
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c["function"])
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out
