# Roadmap

What exists, what exists only as a library, and what is not built. The lists after the first are plans, not
promises: an item may change or be dropped, and none of it is available until it appears in the first list.
[The map](tasks.md) holds the same information item by item and is generated from the code.

## Available

| Part | Where |
|---|---|
| Virtual scanner on engine 0.3.0: a k-space engine, a console with timing, cost and limits, a scan log that replays | `src/mrgym/scanner/`, [scanner.md](scanner.md) |
| Six sequences on every console: `spgr`, `afi`, `se`, `ir`, `bssfp`, `epi` | [scanner.md](scanner.md) |
| Sequences formed by the signal core, on consoles that list them: `fse`, `mese`, `megre`, `dwse`, `satrec`, prepared `fse`; a task names the ones its console offers | [physics/core.md](physics/core.md) |
| Task families with a score, grouped by subject and kind of task; each is checked on 300 fresh objects per level before it is called ready | the table in the [README](../README.md), `mrgym list`, the cards under `envs/` |
| A design task: the agent designs an RF pulse with `design` and `simulate`, images its profile on a scanner with hidden hardware errors, and is held to a proven lower bound on pulse duration | `Scanner/RFPulse-v1`, [physics/design.md](physics/design.md) |
| The check a task must pass: scripted policies on fresh instances, one verdict per level | `mrgym gate <id>`, `mrgym check <id>`, [ADDING_A_TASK.md](ADDING_A_TASK.md) |
| Physics checks against closed forms and independent computations, one table per package | `mrgym validate`, `python -m mrgym.core.validate`, `python -m mrgym.design.validate`, [physics/validation.md](physics/validation.md) |
| A comparison with public phantom data measured on real scanners, with the differences listed | `python tools/measured/run_all.py`, [physics/measured.md](physics/measured.md) |
| Training stack and one result: a 27B model trained by reinforcement learning on one task, measured on engine 0.2.0 | [RECIPE.md](RECIPE.md), [experiments/RESULTS-27b.md](experiments/RESULTS-27b.md) |
| Recorder, one-file viewer, model runner for an OpenAI-compatible endpoint, HTTP scanner service | `mrgym record`, `mrgym view`, `mrgym run-agent`, `mrgym serve` |

## Built as a library, not yet reached by a scan or a task

Each computes and checks its physics; the scanner does not use it yet.

| Library | What it computes | What is missing |
|---|---|---|
| `mrgym.spins` | coupled spin systems by density matrix; PRESS, STEAM, semi-LASER, MEGA; a metabolite basis; quantification | a voxel that holds metabolites, a console sequence |
| `mrgym.em` | B1+ and B1-, SAR and RF shimming from Maxwell's equations on the slice (two dimensions) | the scanner's fields are still declared patterns |
| `mrgym.scanner.array`, `recon` | receive arrays, coil combination, SENSE, g-factor | console options; GRAPPA |
| `mrgym.scanner.volume`, `fields3d` | a seeded 3D world with exact planes; the susceptibility field by the dipole kernel; shimming to second order | a slice engine that scans it; multi-slice; 3D encoding |
| `mrgym.core` beyond what is served | DESS, STEAM, a gradient echo with a real spoiler, magnetisation transfer (a two-pool tissue table exists) | serving them within about a second per scan |

## Known and not yet corrected

- Seven diffusion coefficients of the tissue table lie outside their sources ([physics/limits.md](physics/limits.md), block 3a).
- Differences from measurement that need an engine change: no temperature (the fat-water shift is the body's),
  receiver noise too high for a head array, console timing slower than product sequences, an ACR phantom that is too
  uniform and always centred ([physics/measured.md](physics/measured.md), known differences).
- `Scanner/FatSuppress-v0` is not ready on engine 0.3.0: on the corrected physics spectral saturation cannot meet
  its threshold. A successor with other thresholds is not built.
- The core-served echo trains cost 1 to 5 CPU seconds per scan; the six console sequences cost 0.2 to 0.4.
- Transverse magnetisation carried from one repetition to the next is dephased, not formed.
- The corrected proof code behind the duration bound of `Scanner/RFPulse-v1` was reviewed once, before its last
  three fixes, and not again.

## Not built

Tasks

- A task in which the next scan must depend on the previous result. Every family except one level of
  `Scanner/RFPulse-v1` is a measurement task: a fixed list of scans does as well as the scripted solution.
- A request task: an objective and requirements, no method named.
- Design tasks beyond the RF pulse: the flip angles of an echo train, a diffusion-encoding waveform, a whole sequence.
- A lesion-contrast task. One was built and is not registered: on this engine a constant protocol passes a quarter
  to a half of its instances.
- More tasks on the core-served sequences. `Scanner/T2Echoes-v0` is the only one. A diffusion-coefficient task and an
  echo-train protocol task were built and are not registered: the first reads 20 to 66 % high with its stated method
  since magnetisation is carried across TR, the second depends on a table the core has since rewritten.

Scanner

- Flow, perfusion, contrast agents, chemical exchange in images, spectroscopy in the scanner, susceptibility mapping
  as a task, functional contrast.
- Eddy currents, nerve stimulation and acoustic noise as limits of the console.
- Field strengths other than 1.5 and 3 T.

Evidence

- Training on engine 0.3.0. The one training result was measured on engine 0.2.0 and on one task.
- Any model run on the thirteen single-skill tasks, on the design task or on the core-served sequences.
- Review by MR physicists outside the project.

Distribution

- A release tag and an upload to PyPI. The package installs from a checkout.
- A run of the OpenEnv adapter and of the Harbor export on the current ids.
