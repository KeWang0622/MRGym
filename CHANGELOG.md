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

### The engine is compared with measured data; two defects it exposed outside the engine core are fixed

No scan data, no engine id and no environment id changes (`tools/audit/golden.py --check`: 0 changed;
`python -m mrgym.compat.fingerprint --check`).

- **New page [docs/physics/measured.md](docs/physics/measured.md)**, generated from a stored table that
  `python tools/measured/run_all.py --write` computes and `--check` verifies: public phantom sessions on real
  scanners next to the engine's own model of the same protocol, one row per quantity with the measurement, its
  spread, the engine, the difference and a tolerance or the words "declared difference". The originals are fetched by
  `--fetch` into a folder outside the repository; the repository keeps a few kB per rung under
  `docs/physics/measured/`, each with licence, citation and the checksum of the original.
- **Fixed: QA regions placed from the isocentre.** `qa.ghosting` and `qa.artifact_line` put their background
  regions round the phantom as it is found in the image (`qa.follow`). On a real image whose phantom lay 21 mm off
  centre the ghosting ratio was 0.18 instead of 0.0016. A phantom within 3.5 mm of the isocentre keeps the regions
  of the guidance, so every number of the package's own worlds is unchanged (the stored fault x test matrix was
  re-run: 26 rows, no difference).
- **Added: the published tables of the NIST system phantom** (`anatomy.NIST_S1`, `NIST_S2`: T1 and T2 of both plates
  at 1.5 and 3 T) and the T2 plate `anatomy.nist_t2`. `NIST_T1` and the world `nist_t1` are not changed.
- The map: "Agreement with measurement" (C33) is formed in part; its evidence is the rows of the new table.
- The sentence that nothing here has been compared with a scanner is replaced, where this change may touch it
  (`docs/physics/limits.md`, `external.md`, `docs/DESIGN.md`, the map), by what has and has not been compared.
### Receive arrays and parallel imaging, as a library

No reward, no scan data and no id changes; the engine version is unchanged (a world without the new key gives the
bytes of before: `tools/audit/golden.py --check` reports 0 changed).

- **`mrgym.scanner.array`.** A world spec may carry `array` (elements on a ring, a row or two rows; per-element
  Biot-Savart loop maps; a noise covariance; a dead, noisy or mis-scaled element). `ArrayEngine` forms the
  multi-channel k-space of a protocol through `Engine.amplitudes` and `Engine.kspace`: per coil (the definition) or
  fast (the object once, the maps in image space), with correlated noise scaled as the single-channel noise is.
  Maps and covariance can be handed in from outside (`array.given`).
- **`mrgym.scanner.recon`.** Noise whitening; root sum of squares, optimal (Roemer) and adaptive (Walsh)
  combination; map estimation from a prescan or from calibration lines; sampling masks; SENSE with optional
  regularisation, its g-factor map and noise in SNR units; conjugate-gradient SENSE; pseudo multiple replicas. GRAPPA
  is a seam.
- **Checks.** `python -c "from mrgym.scanner import recon; recon.validate()"` ([docs/physics/array.md](docs/physics/array.md));
  `python tools/audit/array_rederive.py`; an additive section `array` in `tests/golden/engine_digests.json`.
- **Not yet.** No console option, tool or task uses it: the scanner an agent talks to still has one receive channel.

### The repository is laid out for someone who wants to use the environment

No reward, no scan data and no id changes.

- **Front page.** `README.md` is short: what it is, install and use, the tasks, how to add one, whether training works,
  the docs. What it held before is on [docs/TOUR.md](docs/TOUR.md) (demo transcript, inventory, figures, timing, limits,
  repository map); the full environment table is [envs/README.md](envs/README.md), the training table
  [docs/RECIPE.md](docs/RECIPE.md). New page: [docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md), with a runnable skeleton in
  `examples/new_family/`.
- **Moved.** `DATASHEET.md` to `docs/DATASHEET.md`; `conformance/` to `tools/conformance/` (`make conformance` is
  unchanged); `results/README.md`, `results/results.json`, `results/run_config.json` and `results/manifests/` to
  `docs/results/`.
- **Removed: `results/episodes/`.** The 1,400 raw episode records of the pilot runs (package 0.1.0, engine
  `mrgym-engine-0.1.0`, the retired ids `Scanner/NoWrap-v0` and `Scanner/T1Transmit-v0`, schema 1 without
  conversations). Commit `294ae25` is the last one that holds them. Their rows stay in `docs/results/`.
  `mrgym results --check --dir docs/results` now checks that archive for what it is: the table against `results.json`,
  with the statement that nothing was compared with records.
- **Removed: `datasets/`.** It was the output of `mrgym export harbor`, which writes it again in half a minute. Each
  export draws new instance seeds, so the folder is not reproduced byte for byte; the 21 instances that were
  committed are in commit `294ae25`. `datasets/` is now ignored by git.
- **Removed: `extras/recon_repair/`.** A code-repair task set that was not part of `mrgym`, did not import it and
  was used by nothing here. Commit `294ae25` is the last one that holds it.
- `python tools/check_links.py` checks every relative link of the Markdown pages; `make check` runs it.

### The map: which MRI physics is covered, and which task asks for which piece

`src/mrgym/physics_map.py` is one authored table: four subjects (signal and contrast; spatial encoding and
reconstruction; the system; image quality and artefacts), five kinds of task (measure, set, trade, request, design),
the items of MRI physics that textbooks and curricula name, each with the status of the engine and one row per scored
family: its kind, the item it isolates, the items it uses. The statuses: formed (the scanner serves it, and a check
that exists shows it), partial, computed (the package computes it and checks it, for example in `mrgym.design`, but
no console sequence or task reaches it), absent. An item that is not fully formed names the piece of engine work it
waits for, one of seven (a 3D world, a receive array, spin systems, a field model, the core in the console, realistic
anatomy, measured data), or none where ordinary work on the present engine is enough. Generated from the table: `docs/tasks.md`, `mrgym map`,
one line under the title of every environment card, and the task table of the README, which is now grouped by subject
with the kind of task beside each id and a sentence with the counts of the map. `mrgym list` groups the ids by
subject and prints the kind of task beside each; `mrgym list --flat` prints the list as before. No reward, statement,
tool or fingerprint changes: the map is read by no environment, and a family can be filed elsewhere without a new id.
A new family needs one line in `FAMILIES` (`mrgym map --assert`, CONTRIBUTING.md).

### Four families name the requirement a failed delivery misses (the label changes, no reward does)

`info["outcome"]` of a delivered, measured answer that fails was `check_failed` on `Scanner/SNRSpec-v0`,
`Scanner/BandFree-v0`, `Scanner/ArtefactClinic-v0` and `Scanner/ProtocolToSpec-v0`. It is now the first requirement of
the pass rule that the delivery misses, in the order in which the verifier declares its checks (`task.REQUIREMENTS`,
`task.OUTCOMES`, `task.outcome`). The label names the requirement, never the image, finding or region:

| Family | Labels, worst first |
|---|---|
| SNRSpec | `snr_too_low`, `scan_too_long` |
| BandFree | `band_on_region`, `second_acquisition_not_needed`, `bandwidth_too_high` |
| ArtefactClinic | `diagnosis_wrong`, `remedy_leaves_artefact`, `locked_parameter_changed`, `parameter_outside_window`, `pixel_too_large`, `scan_too_long`, `snr_too_low` |
| ProtocolToSpec | `t1_contrast_too_low`, `t2_contrast_too_low`, `snr_too_low`, `scan_too_long` |

Unchanged: `passed`, `checks`, the reward, `progress` and every tolerance. The label is a function of the stored
`checks`, so a record written before this entry is relabelled without a re-run:
`python tools/audit/policy_identity.py relabel RUN/episodes`. Counts of `check_failed` on these four ids are not
comparable across this entry; counts of every other label are. `tools/audit/policy_identity.py play` / `compare`
plays every control policy on two trees and compares reward, checks and deliverable episode by episode.

### Two single-skill families of spatial encoding: `Scanner/ShiftFree-v0`, `Scanner/GhostAside-v0` (candidates)

New ids; no existing id, text or reward changes. Both are `kind="candidate"`: no gate result is stored for them.

- `Scanner/ShiftFree-v0` (`envs/shift_free/`): the readout as a time axis. Fat is displaced along the frequency-encode
  axis by its frequency difference over the bandwidth per pixel. Three marked regions lie between fat discs at hidden
  distances; per region the bandwidth, the readout direction and (level 3) the phase-encode axis of its scan are
  delivered: fat off the region at the highest signal-to-noise ratio that allows.
- `Scanner/GhostAside-v0` (`envs/ghost_aside/`): the line-to-line time axis. A tube whose signal pulsates leaves
  ghosts along the phase-encode axis at (phase field of view) x dist(rate x TR x averages). Three tubes pulsate at
  hidden rates; per tube the phase-encode axis and the repetition time of the fastest scan without a ghost on its
  two regions are delivered.

Their size ceilings in `tests/test_observation_budget.py` are first values (16300 characters each).

### Two candidate families on spatial encoding: `Scanner/Resolve-v0` and `Scanner/FoldAway-v0`

Both are single-skill families on the spin echo, registered as `candidate` (no stored gate yet), with levels 1 to 3
(level 1 states the method). `Scanner/Resolve-v0`: three pairs of small features at hidden spacings; deliver, per
pair, the geometry of the fastest scan of the whole object that separates the pair (a dip of one half between the two
peaks, read between the pixels). `Scanner/FoldAway-v0`: an object larger than the affordable field of view and four
marked regions; deliver, per region, the fastest geometry in which no aliased signal lands on the region (checked
against a twin scan without wrap-around). No existing id, reward or engine number changes.

### Two single-skill candidate families: `Scanner/ErnstFlip-v0` and `Scanner/T1Vial-v0`

Both are new ids (`kind="candidate"`: no stored gate yet); no reward of an existing id changes.

- `Scanner/ErnstFlip-v0`: for five marked vials of hidden T1, the nominal flip angle at which a 2D spoiled gradient echo
  at a stated repetition time gives each its highest signal. Pass: at the delivered angle every vial gives at least
  99.5 % of the largest signal it can give at any flip angle. The budget pays for two gradient-echo scans. Level 1
  states the method (two scans, a printed table of the 2D signal ratio), level 2 withholds it, level 3 adds a hidden
  transmit scale. The textbook angle arccos(exp(-TR / T1)) of the true T1 fails at every level (the slice profile).
- `Scanner/T1Vial-v0`: T1 of three marked vials on a system with a calibrated transmitter, each within a factor 1.10.
  Level 1 states inversion recovery with three inversion times and the signed three-parameter fit, level 2 has 60 % of
  the scan time, level 3 has vials from 60 to 2000 ms that differ by a factor of ten or more.

### A refused call is charged by its class: this changes when an episode ends

A call refused with class `shape` or `vocabulary` (for how it is written) is a rejected call, and a call refused with
class `task` (for what it asks) is a tool call, whatever its type; before, every `invalid_value` refusal was a
rejected call, so a calculator program that ran and failed, a value out of its range or off its grid, a missing
parameter, another unit and a read-out the scanner cannot take counted toward the eight rejected calls that end an
episode, against the rule the task text states. This changes when an episode ends (such refusals no longer bring the
rejected-call limit closer, and they now use up tool calls), so results measured before and after this entry are not
comparable on any family whose episodes meet these refusals, above all the families that use the calculator.

Unchanged: every limit (tool calls, 8 rejected calls, 3 refused deliverables), the text of every refusal, the charge
of scan time (none for any refusal), and `invalid_deliverable`, `episode_over` and `internal_error`, which stay
rejected calls. What the calculator cannot read (a syntax error, Python it does not have) stays class `shape`: a
rejected call. The rule is `mrgym.interface.counts_as_tool_call`; the table of `mrgym.interface`, `mrgym.env`,
docs/TRAINING.md and the Limits block of every task text say it in the same words (`interface.TOOL_CALLS_TOLD`,
`interface.REJECTED_CALLS_TOLD`; the Limits sentences on tool calls and rejected calls and one sentence of the
datasheet changed). `tools/audit/rejected_replay.py RUN/episodes` reads the stored episodes of a run under both
rules. Pinned by `tests/test_rejected_accounting.py`.

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

### Volume worlds: the hidden object in three dimensions (no scan of them yet)

No reward, no scan data and no id changes: a 2D exam is bit for bit what it was.

- **Two world kinds, `head3d` and `qa3d`** (`mrgym.scanner.volume`). A world is a seeded recipe of analytic shapes, not
  a voxel array: a head (scalp, fat, skull, CSF, cortex, white matter, ventricles, deep nuclei, air cavities, eyes;
  lesions, straight vessels with a hidden direction and speed, placed inserts and a pose as options) and a cylinder
  test object (the vial section of the 2D `qa` phantom of the same seed, a ball, a pair of crossed wedges, a block
  with two crossed ramps). It is evaluated at points, on a plane of any position and angle on the engine's grid, and on
  a coarse lattice; regions are balls and cylinders with their truth. What it holds, the coordinate convention and the
  measured numbers: [docs/physics/volume.md](docs/physics/volume.md).
- **Nothing images a volume world yet.** It has no fields and the engine takes 2D worlds only; the map of MR physics
  is unchanged.
- **A baseline for the work that follows** (`tests/test_compat_2d_unchanged.py`, `tests/golden/compat_2d.json`): for
  every 2D world kind and console sequence the world, the protocol schema, the scanner card, the `plan` output, the
  header and the noise-free image, the refusal of the parameters a 2D exam does not simulate, the golden engine digests
  and the id fingerprints, as recorded on the tree before this change. `make golden` checks it and the new
  `tests/golden/volume_digests.json`.

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
