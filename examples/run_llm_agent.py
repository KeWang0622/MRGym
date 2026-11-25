"""Run one language model on MR Gym through any OpenAI-compatible chat-completions endpoint.

    export OPENAI_BASE_URL=https://your-endpoint/v1     # the URL that serves /chat/completions
    export OPENAI_API_KEY=...                           # not needed for most local endpoints
    export MRGYM_MODEL=your-model-id
    python examples/run_llm_agent.py

The model plays seeds 0-2 of Scanner/NoWrap-v1, levels 2 and 3. The scripted reference policy and the no-scan
answer play the same six instances, so the table shows the model between the two. The environment runs in this
process; the model only writes tool calls. The command line does the same:

    mrgym run-agent --env Scanner/NoWrap-v1 --seeds 0-2 --model "$MRGYM_MODEL" --base-url "$OPENAI_BASE_URL" \
        --policy reference,prior --out runs/llm_example
    mrgym results --dir runs/llm_example

Add --dry-run to print what the model would receive without calling anything.
"""
import os

from mrgym import results
from mrgym.agents import runner

OUT = "runs/llm_example"

if __name__ == "__main__":
    run = runner.run(
        OUT, ["Scanner/NoWrap-v1"], levels=[2, 3], seeds=range(3),
        models=[os.environ["MRGYM_MODEL"]], policies=["reference", "prior"],
        base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ.get("OPENAI_API_KEY"),
        max_turns=20, workers=2)
    results.write(OUT)                                   # results.json and the table in README.md, from the records
    print(results.table_markdown(OUT))
    print(f"{run['started']} episodes played, {run['infra_errors']} lost to endpoint failures; records under {OUT}/episodes")
