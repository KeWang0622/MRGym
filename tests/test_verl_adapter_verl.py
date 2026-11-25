"""The parts of the verl integration that import verl: the agent loop, the checkpoint callback, the CPU dry run.

Skipped where verl (and a tokenizer of the model family) is not installed. To run it:

    uv venv VENV && uv pip install --python VENV/bin/python "verl @ git+https://github.com/verl-project/verl@8357f1e" pytest
    MRGYM_VERL_TOKENIZER=<folder with a Qwen3.5 / Qwen3.8 tokenizer and chat template> \
        PYTHONPATH=src VENV/bin/python -m pytest tests/test_verl_adapter_verl.py

What is shown (``mrgym_verl.dryrun``, with verl's own classes and the real tokenizer, no GPU): the overrides of
``overrides.sh`` compose against verl's config tree; scripted episodes through ``MRGymAgentLoop`` give token streams
equal to a from-scratch chat-template rendering, masks exactly on the sampled tokens, the environment's pass as the
reward, complete records; a cut reply and a full context end as ``no_submission_length``; the curriculum hook serves
training rows; verl's GRPO advantage and PPO loss accept the tensors; the checkpoint callback publishes ``LATEST``.
"""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("verl", reason="verl is not installed (see the module docstring)")
pytest.importorskip("transformers")
TOKENIZER = os.environ.get("MRGYM_VERL_TOKENIZER")
if not TOKENIZER or not os.path.exists(os.path.join(TOKENIZER, "tokenizer_config.json")):
    pytest.skip("MRGYM_VERL_TOKENIZER does not name a tokenizer folder", allow_module_level=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERL = os.path.join(ROOT, "integrations", "verl")


def test_dryrun_passes(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), VERL]), OMP_NUM_THREADS="1", MODEL_PATH=TOKENIZER,
               RUN_NAME="t", DATA_DIR=str(tmp_path / "data"), CKPT_DIR=str(tmp_path / "ckpt"), RECORD_DIR=str(tmp_path / "records"),
               CHAT_KWARGS="enable_thinking=False", TRAIN_ARMS="Scanner/NoWrap-v1:2", MRGYM_PRINT_ONLY="1", PYTHON=sys.executable)
    p = subprocess.run(["bash", os.path.join(VERL, "launch.sh")], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "verl.trainer.main_ppo" in p.stdout, p.stderr[-2000:]
    overrides = str(tmp_path / "overrides.txt")
    with open(overrides, "w") as f:
        f.write(subprocess.run(["bash", "-c", f'source "{VERL}/overrides.sh"; printf "%s\\n" "${{OVERRIDES[@]}}"'], env=env, capture_output=True,
                               text=True, check=True).stdout)
    with open(os.path.join(TOKENIZER, "config.json")) as f:
        model_type = json.load(f)["model_type"]
    p = subprocess.run([sys.executable, "-m", "mrgym_verl.dryrun", "--tokenizer", TOKENIZER, "--model-type", model_type, "--overrides", overrides,
                        "--cells", "Scanner/NoWrap-v1:2", "--seeds", "100", "--grad-step", "--out", str(tmp_path / "out")],
                       env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and p.stdout.rstrip().endswith("DRYRUN PASS"), p.stdout[-3000:] + p.stderr[-2000:]
    with open(tmp_path / "out" / "dryrun_report.json") as f:
        report = json.load(f)
    c = report["checks"]
    assert c["stream_vs_full_template"]["different"] == 0 and c["masked_tokens_equal_sampled_tokens"] and not c["records"]["missing_fields"]
    by = {r["behaviour"]: r for r in report["rows"] if r["split"] == "train"}
    assert by["reference"]["reward"] == 1.0 and by["prior"]["reward"] == 0.0
    assert by["cut"]["outcome"] == by["context_cap"]["outcome"] == "no_submission_length"
    assert report["loop"]["signal"] == "pass" and report["loop"]["tool_call_limit"] is None
