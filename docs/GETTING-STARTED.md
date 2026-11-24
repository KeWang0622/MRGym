# Getting started

Python 3.10 or later, numpy and scipy. No GPU and no network.

```bash
git clone https://github.com/KeWang0622/MRGym.git
cd MRGym
pip install -e .
mrgym --version
mrgym list
```

`mrgym --version` prints the package version and the scanner engine. A number measured on one engine is comparable
only with numbers from the same engine and the same environment version (`-v0`).

## 1. Watch a scripted episode

```bash
mrgym demo
mrgym demo --id Scanner/T1Transmit-v0 --level 2 --seed 1
```

The transcript shows every tool call and the console's reply. The policy is scripted and sees only tool results.

## 2. Operate the scanner yourself

```bash
mrgym scan sequence=se tr_ms=500 te_ms=15 matrix=256 --out mrgym_out/brain_t1w.png
mrgym play --phantom qa
```

`mrgym play` reads `plan`, `scan`, `view`, `info`, `budget` and `quit` from the keyboard and writes a PNG per scan.
The [scanner manual](scanner.md) lists every parameter.

## 3. Run an episode from Python

```python
import mrgym

env = mrgym.make("Scanner/NoWrap-v0", level=1, seed=0)     # obs="vision" returns PNG views
obs, info = env.reset()                                    # obs.text is the task statement
tools = env.tools("openai")                                # function schemas: "openai", "anthropic" or "mcp"
obs, reward, terminated, truncated, info = env.step({"name": "plan", "arguments": {"protocol": {
    "sequence": "spgr", "tr_ms": 10, "flip_deg": 10, "fov_mm": 288, "matrix": 160}}})
```

- `step` accepts a plain `{"name", "arguments"}` call, an OpenAI tool call, an Anthropic `tool_use` block and an MCP
  `tools/call` request. It never raises for something the agent did: a malformed or refused call comes back as an
  observation with `is_error` and costs nothing.
- `obs.text` is complete for a text-only model. `obs.data` is the same content as JSON. `obs.images` is filled only
  by `view` in `obs="vision"`. `obs.to_openai(id)`, `obs.to_anthropic(id)` and `obs.to_mcp()` give the wire forms.
- The reward is 0 until the terminal tool. `info["score"]` holds the gates and the error; the agent sees the score
  only with `reveal_score=True`.
- `seed` is an integer, a string, or 32 hex digits. Integers are mapped to 128-bit world seeds with a public key, so
  training worlds are reproducible by anyone. `split="heldout"` needs `MRGYM_SECRET` in the environment; without it
  every world is public.

## 4. Connect a language model

`mrgym run-agent` drives any OpenAI-compatible chat endpoint through the tools and writes one record per episode.
`--dry-run` prints what the model would receive and calls nothing.

```bash
mrgym run-agent --env Scanner/NoWrap-v0 --levels 1,2 --seeds 0-23 --model MODEL --base-url URL --out results --dry-run
mrgym results --check
```

[results/README.md](../results/README.md) gives the full command and the table it feeds.

## 5. Record and view

```bash
mrgym record Scanner/NoWrap-v0 --level 2 --seed 0 --policy reference --out runs/demo
mrgym view runs/demo/episode_bundle.json --out runs/demo/viewer.html
```

The bundle holds every call, every reply, the images, the scanner log and the score. The hidden object is stored
under its own key and the viewer shows it only at the last step.

## 6. Agents with code execution: the HTTP service

In process, the hidden world is out of reach of a tool-calling model but not of code running in the same process.
An agent with a shell needs the scanner behind a process boundary:

```bash
mrgym serve --id Scanner/NoWrap-v0 --level 1 --seed 0 --port 8000 --log-dir runs/scanner_log
python src/mrgym/scanner/client.py --url http://127.0.0.1:8000 info
```

`src/mrgym/scanner/client.py` is one file with no import from the package; copy it into the agent's workspace. The
service has four routes (`info`, `budget`, `plan`, `scan`) and none that resets the session or returns the truth.
The verifier replays the hash-chained log with `mrgym.scanner.session.replay(config, events)` and scores the
deliverable with the environment's `task.score`. The log written by the service and the log of an in-process episode
with the same calls are identical link by link; `tests/test_scanner.py` asserts it.

## 7. Check what you changed

```bash
pytest -q
mrgym validate
python tools/generate.py --check
```
