# MR Gym

A virtual MRI scanner for AI agents: an RL environment in which an agent runs priced acquisitions on a hidden
system + tissue world and is scored on what it delivers.

**Status: early and being rebuilt.** The scanner core works; the environment surface, the environments on top of it and the
documentation are being redone. Interfaces and names will change. There are no model results and no admitted physics
environment yet.

## What is in the tree today

| Part | State |
|---|---|
| `mrfrl/scanner/` | Scanner session: free `plan`, priced `scan`, scan-time budget, hash-chained log that a verifier replays |
| `mrfrl/phys/` | Hidden world and signal physics: closed forms and an EPG steady state for RF-spoiled gradient echo, with a consistency suite |
| `mrfrl/t1_transmit/` | A first measurement task with control policies. Its own admission rule rejects it: one fixed recipe solves every instance |
| `mrfrl/recon_repair/` | Repair of a faulty reconstruction, graded on hidden scans. A code-repair task in an MR setting, not a scanner environment |

## Run

Python ≥ 3.10, numpy, scipy.

```sh
python3 tests/test_scanner_core.py
python3 tests/test_recon_repair.py
python3 -m mrfrl.t1_transmit selfcheck --n 10   # control policies and the admission table
```

## Known limits

- No `make` / `reset` / `step` entry point yet; an episode needs internal imports.
- `recon_repair` can be solved by anyone who has this repository's generator: run it only with the network closed.
  The local verifier separates the submission from the ground truth by directory only; that is not a security boundary.
- The engine uses hard pulses without a slice profile, serves the steady state without simulating the transient, forms images in
  the image domain, and has one receive channel with white noise.
- Nothing here has been validated against a scanner or an external simulator.

## Related

[MRFoundry](https://github.com/KeWang0622/MRFoundry) · [MRI Agent Bench](https://github.com/KeWang0622/MRIAgentBench)
(held-out evaluation; none of its tasks, generators or reference solutions are used here).

Licence: Apache-2.0.
