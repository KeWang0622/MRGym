# Changelog

MR Gym versions anything that can move a reward. A change that cannot alter any reward is a patch. A change that
needs stored episodes to be re-scored is a minor version. A change that needs episodes to be re-run is a major
version.

Three versions travel with every number:

| Version | Where | It changes when |
|---|---|---|
| Package | `mrgym.__version__`, `pyproject.toml` | the rule above |
| Engine | `mrgym.ENGINE_ID`, printed by `mrgym --version` | the same seed and protocol can give other scan data. Bumped by hand; `tests/test_engine_version.py` pins it with reference numbers. Recorded logs replay only on the engine that wrote them. |
| Environment | the `-vN` suffix of the id | a reward of that environment can change. The old id is never reused. |

## Unreleased

### The interface and the task texts say what is true (before the first release of interface v1)

Found by reading stored conversations of an untrained model (run c04, thinking off): places where an answer or a task
text made the reader believe something false, or did not say what a scan would pay for. No tolerance, verifier,
level or hint tier changed, and no text states a method. The scanner's log, every reward and every scripted policy are
unchanged (`tools/audit/golden.py`: the chain head, reward, steps and scan seconds of every reference episode are the
ones stored before; only statements and the hash of the observation texts moved). `Observation.text`, two tool
descriptions and the task texts changed, so a model result measured before this entry is not comparable with one
after it. Pinned by `tests/test_interface_clarity.py`.

Task texts
- Every family whose deliverable is evaluated on a scan the task names (`NoWrap`, `NullTissue`, `FatSuppress`,
  `BandFree`, `SNRSpec`, `ProtocolToSpec`, `ArtefactClinic`) states, in the same two sentences
  (`family.OWN_SCANS`, `family.NOT_CHARGED`): the agent's own scans may use any sequence and any parameters the
  console accepts, only the deliverable is evaluated, and the delivered scan is not charged to the scan-time budget
  and not limited by it. They replace "Use the scan-time budget for whatever you need to measure first".
  `PhantomQA` and `T1Transmit` do not carry them: there the deliverable is a measurement on the agent's own scans.
- `Scanner/PhantomQA-v0`: a failed test has exactly one fault (true by construction); the background rectangles have
  their 63 mm side tangential (top, bottom: along x; left, right: along y); the 1 cm^2 circle lies entirely inside
  the 200 cm^2 circle.
- `Scanner/ArtefactClinic-v0`: a field left out of the delivered protocol takes the console default.

Answers
- `plan`: the first line prices the protocol as a scan now and says what it leaves (`as a scan now: 35.00 s, leaves
  115.00 s`; `as a scan now: needs 33.50 s, 30.00 s left`), without a verdict on "the budget": a plan is also how a
  deliverable is looked at, and a deliverable is not scanned. On a console that reuses adjustments
  (`Scanner/ProtocolToSpec-v0`) a scan priced with the short adjustments also shows its price as a scan of its own,
  the one the verifier charges a delivered image. The image is printed as (y, x) with the phase-encode axis named,
  the order of `scan` and `view`. The minimums are the ones the console enforces (`se`: `tr_ms 50`, not the 12.14 the
  timing would allow). A phase side that the console moved to a whole, even number of pixels is said (`phase_fov
  0.511364 -> 0.5 (44 px, an even number)`). A line names the marked regions in which no pixel of the planned image
  lies, and those its edge cuts (`data`: `regions_without_pixels`, `regions_partly_outside`, `scan_s_own_scan`, each
  only when it applies).
- `profile`: a `width_mm` that reaches beyond the image is cut there and the note says so, with the centre the line
  then has. `samples` above 128 is cut to 128 with a note (it was refused); the description says "default one per
  pixel up to 128, the most per call".
- `view` (digit map): the y of every row after it and one line with the x of the columns, the centres of the blocks.
  The map is unchanged and `data["digit_map"]` holds it without the numbers. Cost: 150 characters at 16 rows, 225
  at 32, 300 at 48; the size limits of `view` in the tests moved from 600 / 1400 to 700 / 1550 for it.
- The budget refusal states the numbers and that any accepted protocol that fits can be scanned; it no longer
  proposes fewer lines, fewer averages or a shorter `tr_ms`.
- Phase of a read-out: the `scan` description, `scanner_info` `ir`, `outputs` and `sequences` say that a phase value
  carries the receive coil's phase, that its own sign says nothing, and that a sign is read against a read-out of the
  same region whose sign is known (before: "the sign is in the phase image").
- Datasheet, `slice_profile`: with `excitation: nonselective` every voxel sees the same actual flip angle (the
  nominal angle times the local transmit factor), without a slice profile.

### What an agent reads got shorter (interface v1, before its first release)

An episode is a token stream a trainer has to hold in one piece, and most of it was the environment's own text.
The scanner's log, every reward and every scripted policy are unchanged, and so is the `Observation.data` they read
(`plan`, `scan`, `read_regions`, `profile`, the instrument card of `scanner_info`). `Observation.text`, the tool
definitions and the task texts are not, so a model result measured on the text before this entry is not comparable
with one measured after it; where `data` mirrors a rendering it follows it (the rows `calc` shows, the index of
topic `physics`, the default `digit_map` of `view`, the messages of errors). No model baseline was published on the
old text. `tools/obs_budget.py` replays recorded conversations on the tree it is run with and prints where the tokens
go, before and after.

Tool definitions
- `plan` documents each protocol parameter in one line (what it is, its fixed range in words, the sequences that
  have it, the default). `scan` and a protocol delivered as one argument (`Scanner/ArtefactClinic-v0`) list the same
  parameters by name and type, without descriptions (`ToolSpec.see`). No `minimum` / `maximum` keyword is left on a
  protocol parameter: the ranges are stated in the description and checked as before, for `plan`, `scan` and a
  delivered protocol alike (`interface._range_problems`).
- `plan` takes `verbose`. `scanner_info` takes `parameter` next to `topic`, and its topics are `overview`, `task`,
  `presets`, `sequences`, the six sequences, `geometry`, `cost`, `outputs`, `physics`, `encoding`, `slice_profile`,
  `fat`, `motion`, `hidden`. `scan_index` is optional in `read_regions`, `view` and `profile` (the latest scan).
  `view` draws 16 blocks along the longer side by default (32 before) and at most 48 (64 before: 4,160 digits).
- Shorter descriptions throughout; "Free" for "No scan time"; `submit` ends with one sentence instead of two.

Answers
- `calc` shows `print(...)` and bare expressions only; before, every assignment was echoed as well.
- `scanner_info`: `physics` is an index of the mechanisms; each mechanism and each sequence is a topic of its own (a
  sequence: its signal model, the parameters that are its own, the shared ones with their defaults in one line, one
  example call). `sequences` names the six sequences instead of printing six tables. `overview` is the card in short
  with the list of topics. `task` and `presets` are printed compactly. Nothing of the datasheet is out of reach.
- `plan`: two lines. Cost and fit; then what the request does not say (pixels and field of view of the image,
  phase-encode lines, echo time, minimum TE / TR, SAR, fat shift, relative SNR, and the gradient use from 70 % of
  its limit). What the console changed against the request is said (`note:` for a raised `te_ms`, `realised` for a
  time moved to the raster). The header line, every realised parameter and the timing are behind `verbose`.
- `scan`: index, image, cost and echo time in one line, then the read-outs. The header line of a stored scan is
  printed by `status`. Read-out tables: one space between columns, no `image` column for a single image.
- `profile`: two header lines, then rows of `position value`: the position to a tenth of a millimetre for samples
  of half a millimetre or more, every value at the resolution of four significant digits of the largest one.
- The status line under every answer is the tool calls used of the limit; the scan time left and the rejected calls
  are added when they changed (and at reset, and under `status`). The `charged:` line under a refusal is gone: the
  status line says `(nothing charged)`. `Observation.status` and `data["charged"]` hold every number as before.
- Errors: `error <type> · fix: {...} · <parameter>: <message>` on one line (the patch first); lists of every
  parameter of a tool are replaced by a pointer or a suggestion; console range messages state the range only.
- The Limits block of the task text is shorter. Task texts: the sentence "You operate a virtual MR scanner through
  tool calls." is gone, region tables are `id: y, x, radius`; `Scanner/T1Transmit-v1` points at topic
  `slice_profile`.

What is accepted that was refused
- `calc` reads a Python subset: comments (`#`, a leading `//`), text as labels, `print`, tuples, comparisons,
  `and` / `or` / `not`, `a if c else b`, slices, dictionaries, comprehensions, `lambda`, `def`, `for` / `while` / `if`,
  `import math` / `import numpy as np` (no-ops), `math.` / `np.` prefixes, and more functions (`ceil`, `floor`,
  `round`, `len`, `range`, `zip`, `sorted`, `linspace`, `polyfit`, `interp`, ...). It shows `print(...)` and every
  bare expression; a text that asks for nothing gets the names it assigned. Names stay for the rest of the episode.
  Most of the interface tax of a model that does not reason aloud was `calc` text the old grammar refused
  (`mrgym iface-tax --replay` on the corpus below prints both numbers).
- `calc` runs the programs a model writes (`src/mrgym/calc.py`; `python tools/audit/calc_corpus.py <corpus>` replays
  recorded calls and prints what is refused before and after, by class, with the wall times):
  - the syntax tree is turned into closures once and those are run; nothing is handed to `eval` / `exec` / `compile`
    and no attribute of a Python object is looked up by a name of the text (`tests/test_calc_security.py`);
  - bounds are counts, never a clock: `calc.MAX_STEPS` steps per call (a step is an evaluated node, or an element a
    call makes or hands to a function), `MAX_ITEMS` elements per list / array, `MAX_TEXT` characters per text,
    `MAX_KEPT` numbers between calls, `MEMORY_BOUND_MB` per call. A call that is stopped says how far its loops were;
  - methods of lists, text, dictionaries and arrays; `%`, `.format` and f-string formats (`nan` and `inf` are
    printed, not refused); `*` unpacking, `:=`, `try` / `except` / `raise` / `assert`, closures, `global`, complex
    numbers, `&` / `|` between tests of arrays, a `range` of any length in a loop; `np.linalg.solve` / `lstsq` /
    `inv` / `det`, `polyval`, `meshgrid`, `column_stack` / `vstack` / `hstack`, `statistics`, `itertools`, `bisect`,
    and on a function of the text `scipy.optimize.brentq` / `minimize_scalar` / `minimize` / `least_squares` /
    `curve_fit` / `fsolve`;
  - a refusal says in one line what to write instead, with the line it happened on.
  Answers that changed: `a + b` joins two lists and `a += b` extends one, as in Python, and a list written in place
  times a whole number repeats (`[0] * n`; before, all of these were element-wise); `a < b` orders two lists. A list
  from a name or a call times a number is scaled as before (`x * 2`), and `array([...])` is element-wise throughout. `round` of a number rounds as Python does (numpy's rule before: they differ in
  the last digit of some ties). `interp` reads points whose `xp` falls throughout from the other end.
- Canonical form: C14 (`regions` / `circles` on `plan` are dropped), C15 (the fields of a nested deliverable written
  next to it are moved into it, when each name has one home).
- Read-outs: with no `regions` argument, a circle that carries the id of a marked region is read in its place, and
  circles that fill a call alone are read without the marked regions (both were refusals).
- `profile`: a range that reaches beyond the image is read up to its edge and reported; `start_mm` and `stop_mm`
  are taken in either order.

Interface tax (`mrgym iface-tax`): a second replay corpus, `tests/data/c03_interface_calls.jsonl` (every call of 368
episodes of an open 27B model with reasoning off, seven environments, each episode replayed on its own instance);
`tools/obs_budget.py --corpus` writes such a file from a run folder. The test asks for at most 5 % of calls and of
turns on it, and for no call that was accepted when recorded to be refused now.

Smaller things: `mrgym check` accepts the undescribed parameters of a tool that names the tool documenting them
(`ToolSpec.see`); `Env._charged_line` is a hook (the base class still prints the `charged:` line); the refusal of a
missing required parameter no longer fails on a parameter without a description; `tests/test_observation_budget.py`
holds a ceiling per environment id for the characters of tool definitions plus task text.


Engine: `mrgym-engine-0.2.0`. Every protocol gives other data than on 0.1.0; logs recorded on 0.1.0 replay only
there, and every stored control-policy result has to be measured again.

Environments: one break to `-v1`.

- `Scanner/NoWrap-v1` replaces `Scanner/NoWrap-v0`: a family on `envs/family.py` with a pass-only reward. The field
  of view can be moved and turned, so the deliverable gains `centre_y_mm`, `centre_x_mm` and `rotation_deg`; the pass
  rule is "whole object inside, no wrap-around, pixel as required, scan time at most 1.10 x the optimum", with the
  optimum from an exhaustive search on the hidden world (`envs/nowrap/oracle.py`). Levels 0 and 1 are practice levels
  (centred, upright object; the procedure is stated), levels 2 (off-centre) and 3 (off-centre and turned) carry the
  claim. The object is an elongated disc (new world-spec key `qa.body_aspect`).
- `Scanner/T1Transmit-v1` replaces `Scanner/T1Transmit-v0`: the same contract, a fixed tolerance (each region within
  a factor 1.1 of its true T1) instead of one calibrated on the reference, and a reference that is correct on engine
  0.2.0: non-selective 3D where the budget has room for it, otherwise 2D with the slice profile in the model.
  Registered as practice: a fixed protocol passes.
- Removed: `Scanner/NoWrap-v0` and `Scanner/T1Transmit-v0` (retired: `mrgym.make` raises `RetiredEnv` naming
  package 0.1.0), their environment classes, their stored results of engine 0.1.0, and the per-instance admission:
  `mrgym.admission`, `controls.admit`, `info["admitted"]`, the `admitted` field of new episode records, the
  `(function, budget scale)` policy table and the admission flag of `mrgym check`. The gate is `mrgym gate <id>`: one
  verdict per level. `task_id_for` lives in `mrgym.keys`.
- `mrgym check <id>` compares the kind of an id with its stored gate: a candidate needs none; an admitted id needs one
  from the current engine that admits every level of the claim. `mrgym list` prints kind, levels of the claim and the
  retired ids; `envs/index.json` is an object with `environments` and `retired`.
- `Scanner/BandFree-v0` and `Scanner/FatSuppress-v0` (candidates, new on engine 0.2.0). BandFree: two balanced SSFP
  images, each with its own marked regions; per image the receiver bandwidth and one phase cycle, or two when one
  acquisition cannot be band-free; the target is a 2D (slice-selective) image, whose bands are narrower than the
  hard-pulse formula says; at level 3 one marked region holds oil. FatSuppress: per fat-containing region, spectral
  fat saturation or inversion recovery with its inversion time. The hidden world of both is `make_world(seed,
  world_spec)` and nothing else (no hook completes it afterwards).
- World-spec key `fatsat_bw_hz` (a number or a range, at 3 T): the bandwidth of the fat-saturation pulse of the
  session, drawn from a sub-stream of its own. Absent: 1.3 ppm as before, and the world is bit-identical.

- Slice-selective excitation is the default (`excitation: selective`): a windowed-sinc pulse (`rf_pulse`: fast,
  normal, sharp), the signal integrated over the slice profile for every sequence, the pulse priced in minimum
  TE / TR, RF power and a thinnest slice (`SLICE_TOO_THIN`). `excitation: nonselective` serves the hard-pulse
  signal of engine 0.1.0 at the price of a 3D scan (acquisition time times the number of partitions).
- Field of view: `rotation_deg` turns it in the plane; the world is re-evaluated on the turned grid, nothing is
  interpolated. Region read-outs stay in patient coordinates.
- Periodic motion as a declared model (`world_spec["motion"]`): ghosts along phase encoding.
- Hidden system state `fatsat_offset_hz` and protocol parameter `fatsat_offset_hz`; the T1 of oil vials of the QA
  phantom is drawn per world. World-spec keys `edge_mm` and `gnl` for sensitivity studies.
- `mrgym validate`: 63 checks (38 before), new group `rf`; `mrgym validate --group <name>`.
- `mrgym validate --perturb <id>`: the reference policy of an environment re-scored under perturbed physics, stored
  per environment and rendered as the "Validity" section of its card.
- External validation ladder: `python tools/external/run_all.py --check`, `docs/physics/external.md`.
- Worlds built without the new spec keys are bit-identical to those of 0.1.0 (`tests/test_engine_0_2.py`).
- Minimum TR of the default gradient echo rose by about 1.1 ms (2 ms selective pulse and its slice gradient):
  protocols at `tr_ms` 10 with the default spoiler are refused; the localisers of `Scanner/NoWrap-v0` and one
  scan of `Scanner/T1Transmit-v0` were moved to 11 ms for that reason only.

Keys, splits, identifiers, log integrity, the service and the red-team suite. No reward of a stored instance changes;
the `info` keys, the log format and the `serve` / `verify` commands do.

- Splits are a registry (`mrgym.keys.SPLITS`). Public: `train`, `dev`, `tune`, `gate`. Secret: `test-iid`,
  `test-range`, `test-combo`, `test-world`, `test-text`, `ood`, `test-family`, `test-mechanism`, `heldout`, `semi`.
  An unknown split is an error. A secret in the environment no longer changes the train split.
- The secret is at least 32 bytes as hex, from `MRGYM_SECRET_FILE` (mode 0600) or `MRGYM_SECRET`; sub-keys by HKDF.
  Worlds of a secret split made under the old derivation (`HMAC(MRGYM_SECRET, ...)`) are not reproduced.
- Seeds: an int (not a bool) or a printable string. Floats, booleans, bytes and containers raise `mrgym.Error`.
- The episode id is random and new on every reset. The task id is trainer-side only and left the instrument card;
  on a secret split it is keyed.
- Scanner log version 3: a MAC per record under a key of the episode, a seal record, and a head kept outside the
  log directory and returned as `info["log_head"]`. `replay` needs the head; a restart needs the receipt
  (`resume_from`), and a log that does not match it raises `LogTampered` (the service exits with status 3).
- `mrgym serve`: agent routes behind a bearer token, `POST /v1/submit`, `GET /v1/task`, an operator plane on a unix
  socket. `--seed`, `--id`, `--level` and `--config-out` are gone; the episode comes from `--episode-file`,
  `--episode-stdin` or the operator plane. `mrgym verify` takes `--episode-file` and `--head`.
- New: `mrgym epoch keygen | new | run | reveal | verify`, `mrgym redteam <id>`, the entry-point group `mrgym.envs`
  for families from other packages, `integrations/container/` (written and linted, not run), a canary string in
  exported records.
- The gate plays on the public split `gate` and tunes on `tune` (before: string seeds on `train`).

## 0.1.0.dev0

Research preview. Not released: no tag, no upload.

Engine: `mrgym-engine-0.1.0`.

The repository was rebuilt around the scanner. Nothing before this entry was released, so there is no upgrade
path to describe.

- Package renamed to `mrgym`: one distribution, one import name, one command.
- Scanner engine that forms images in k-space, with six sequences: `spgr`, `afi`, `se`, `ir`, `bssfp`, `epi`.
- Three phantoms: brain, abdomen, QA.
- Console: every plan and scan returns the realised protocol, timing, scan time, limits and a header line.
- Scan log: append-only and hash-chained. Replay is bit-exact on the same numerical stack and within a stated
  tolerance on another one; the replay report says which.
- Explicit engine version in place of a hash of source files.
- Environment surface: `mrgym.make`, `reset`, `tools`, `step`, with tool schemas in OpenAI, Anthropic and MCP form.
- Registry of environment ids of the form `Namespace/Name-vN`, each with a kind: sandbox, practice or admitted.
- Three environments:
  - `Scanner/Explore-v0`: sandbox, no reward.
  - `Scanner/NoWrap-v0`: scored; listed as admitted by its control-policy gate. Counts are on its card.
  - `Scanner/T1Transmit-v0`: scored practice; its own admission gate rejects it.
- `mrgym check <id>`: the gate for publishing an id. With `--admission` it reruns the control policies.
- `mrgym validate`: physics checks against analytic formulas, printed as a table.
- Recorder and viewer: `mrgym record` saves an episode as one bundle, `mrgym view` writes a one-file HTML viewer.
- Model runner: `mrgym run-agent` plays an environment with a language model through an OpenAI-compatible endpoint.
- The code-repair task set `recon_repair` moved to `extras/recon_repair/`. It is not part of the package and is
  not counted in any result.
- Tests that keep MRI Agent Bench material out of the tree (`tests/test_no_benchmark_content.py`).
