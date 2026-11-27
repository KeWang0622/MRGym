# Roadmap

What exists in the research preview, and what is still research. The second list holds plans, not promises: an
item may change or be dropped, and none of it is available until it appears in the first list.

## Available in the research preview

| Part | Where |
|---|---|
| Virtual scanner: k-space engine, six sequences (`spgr`, `afi`, `se`, `ir`, `bssfp`, `epi`), three phantoms, console with timing, cost and limits | `src/mrgym/scanner/`, [scanner.md](scanner.md) |
| Scan log with a keyed chain, a head held outside the log directory, and replay | `src/mrgym/scanner/session.py` |
| Environment surface: `make`, `reset`, `tools`, `step` | `src/mrgym/env.py`, `src/mrgym/registry.py` |
| `Scanner/Explore-v0`: the scanner with nothing asked, no reward | `mrgym play`, `mrgym scan` |
| `Scanner/NoWrap-v1`: scored; status as printed by `mrgym list` | its card under `envs/` |
| `Scanner/T1Transmit-v1`: scored practice (a measurement task: a fixed protocol passes) | its card under `envs/` |
| The gate: control policies on fresh instances, one verdict per level | `mrgym gate <id>`, `mrgym check <id>` |
| Physics checks against analytic formulas | `mrgym validate` |
| Recorder and one-file viewer | `mrgym record`, `mrgym view` |
| Model runner for an OpenAI-compatible endpoint | `mrgym run-agent` |
| HTTP scanner service: agent plane behind a token, operator plane on a unix socket | `mrgym serve` |
| OpenEnv adapter, written and not run | `integrations/openenv/` |

## Remaining research work

Environments

- A diagnosis environment: find and characterise a lesion within a scan-time budget, with the decision scored on
  the hidden object. It is not built.
- More environments whose right answer depends on the instance. One admitted environment is a thin basis.

Scanner

- Fast spin echo, with its echo-train contrast and blurring, and a price that makes it a real choice against
  spin echo. Built as a sequence of a console that lists it (`fse`, with `mese`, `megre` and `dwse`: served from the
  signal core, [physics/core.md](physics/core.md), section 11); no environment offers it to an agent yet.
- Multi-slice acquisition and its pricing: slices per repetition, cross-talk, coverage against time.
- Slice profile. Excitation is a hard pulse today and the slice is ideal.
- Receive arrays: several coils, correlated noise, parallel imaging and its noise penalty.
- The approach to steady state, motion and flow are not simulated.
- Comparison with an external simulator or a physical scanner, and review by MR physicists.

Evidence

- Model results: language models run on the admitted environment under a fixed protocol, every started episode
  counted, with the recorded episodes published.
- Training evidence: whether training on these environments improves an agent on the environments, and whether
  that transfers to held-out evaluation. Nothing has been trained on MR Gym yet.

Distribution

- Harbor task directories and a verifiers package for the scanner environments. Neither is written.
- A run of the OpenEnv adapter, and a published Space.
- Upload to PyPI. The package installs from a checkout today.
