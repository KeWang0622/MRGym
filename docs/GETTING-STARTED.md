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
mrgym demo --id Scanner/T1Transmit-v1 --level 2 --seed 1
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

env = mrgym.make("Scanner/NoWrap-v1", level=1, seed=0)     # obs="vision" returns PNG views
obs, info = env.reset()                                    # obs.text is the task statement
tools = env.tools("openai")                                # function schemas: "openai", "anthropic" or "mcp"
obs, reward, terminated, truncated, info = env.step({"name": "plan", "arguments": {
    "sequence": "spgr", "tr_ms": 10, "flip_deg": 10, "fov_mm": 288, "matrix": 160}})
```

- `step` accepts a plain `{"name", "arguments"}` call, an OpenAI tool call, an Anthropic `tool_use` block and an MCP
  `tools/call` request. It never raises for something the agent did: a malformed or refused call comes back as an
  observation with `is_error`, costs no scan time and does not end the episode. `obs.data["error"]` names the type
  of the error, the parameter, the nearest valid value and a `fix`.
- The limits of an episode (scan time, tool calls, rejected calls) are in the task text, in `obs.status`, in the
  `status` tool and in `info["limits"]`: the same numbers. The status line under every result prints the tool calls
  used of the limit, and the scan time left and the rejected calls whenever they changed. `max_steps=N` in
  `mrgym.make` sets the tool-call limit.
- `obs.text` is complete for a text-only model. `obs.data` is the same content as JSON. `obs.images` is filled only
  by `view` in `obs="vision"`. `obs.to_openai(id)`, `obs.to_anthropic(id)` and `obs.to_mcp()` give the wire forms.
- The reward is 0 until the terminal tool. `info["score"]` holds the gates and the error; the agent sees the score
  only with `reveal_score=True`.
- `seed` is an integer, a string, or 32 hex digits. Integers are mapped to 128-bit world seeds with a public key, so
  training worlds are reproducible by anyone, and a secret in the environment does not change them. The public splits
  are `train`, `dev`, `tune` and `gate`; a secret split (`test-iid`, ..., `heldout`) needs `MRGYM_SECRET_FILE` (a key
  from `mrgym epoch keygen`) and refuses to run without it. Floats, booleans and unknown split names are errors.

## 4. Connect a language model

`mrgym run-agent` drives any OpenAI-compatible chat endpoint through the tools and writes one record per episode.
`--dry-run` prints what the model would receive and calls nothing.

```bash
mrgym run-agent --env Scanner/NoWrap-v1 --levels 2,3 --seeds 0-23 --model MODEL --base-url URL --out results --dry-run
mrgym results --check
```

[results/README.md](../results/README.md) gives the full command and the table it feeds.

## 5. Record and view

```bash
mrgym record Scanner/NoWrap-v1 --level 2 --seed 0 --policy reference --out runs/demo
mrgym view runs/demo/episode_bundle.json --out runs/demo/viewer.html
```

The bundle holds every call, every reply, the images, the scanner log and the score. The hidden object is stored
under its own key and the viewer shows it only at the last step.

## 6. Agents with code execution: the HTTP service

In process, the hidden world is out of reach of a tool-calling model but not of code running in the same process.
An agent with a shell needs the scanner behind a process boundary:

```bash
mkdir -p runs && echo '{"env_id": "Scanner/NullTissue-v0", "level": 1, "split": "train", "seed": 0}' > runs/episode.json
chmod 600 runs/episode.json
mrgym serve --episode-file runs/episode.json --public-worlds-ok --port 8000 \
            --log-dir runs/scanner_log --state-dir runs/scanner_state --ready-file runs/ready.json &
# the handle of the episode is in runs/ready.json; its agent token in runs/scanner_state/<handle>.token
export MRSCAN_URL=http://127.0.0.1:8000 MRSCAN_TOKEN_FILE=runs/scanner_state/<handle>.token
python src/mrgym/scanner/client.py task
python src/mrgym/scanner/client.py scan --protocol '{"sequence": "ir", "tr_ms": 1500, "ti_ms": 400, "matrix": 64, "partial_fourier": 0.6}'
python src/mrgym/scanner/client.py submit --answer '{"ti_ms": {"R1": 412.0, "R2": 655.0}}'
mrgym verify --episode-file runs/episode.json --log-dir runs/scanner_log/<handle> \
             --head runs/scanner_state/<handle>.head --answer answer.json
```

`src/mrgym/scanner/client.py` is one file with no import from the package; copy it into the agent's workspace. The
agent's side of the service is `info`, `task`, `budget`, `plan`, `scan` and `submit`, each behind the episode's token;
no route resets the session or returns the truth, and `submit` answers `{"received": true}` only. The seed is never
on a command line: the episode comes from a file of mode 0600 (or from the operator plane, see
`python -m mrgym.scanner.service --help`). `--public-worlds-ok` is for demos: the train worlds can be rebuilt by
anyone who installs the package, so an evaluation of a code-executing agent uses a secret split
(`mrgym epoch keygen --out secret.key`, `MRGYM_SECRET_FILE=secret.key`, `"split": "test-iid"`).

`mrgym verify` replays the log against its head and scores the deliverable with the environment's own code. The head
is the service's commitment to the whole log (length, last MAC, scan seconds); it lives in the state directory, not
with the log, and a log that is shorter than its head, edited or taken from another episode does not verify. The
records written by the service and those of an in-process episode with the same calls have the same content chain,
link by link; `tests/test_scanner.py` asserts it. The threat model is in [SECURITY.md](../SECURITY.md).

## 7. Check what you changed

```bash
pytest -q
mrgym validate
python tools/generate.py --check
```
