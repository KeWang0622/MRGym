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

### The signal core behind the console: a sequence registry, and four sequences served from block lists

No engine bump, no environment id touched, no fingerprint moved. The rule (docs/VERSIONING.md): the engine id changes
when the same seed and protocol can give other data. Every protocol a console accepted before gives the bytes it
gave (`tests/test_compat_2d_unchanged.py`; `python tools/audit/golden.py --check`: 0 changed in images, noisy scans,
prices, array, task snapshots and the 2D baseline; `python -m mrgym.compat.fingerprint --check`: 25 ids current).
What is new is reachable only with a console card that lists it (`exam["sequences"]`, world spec
`console_sequences`), which no world of before has: new protocols on new consoles, added to the golden file as the
section `core` (104 entries).

- **The seam** (commit "The seam"). `console.REGISTRY`: one entry per sequence with its parameters, timing rule
  (which returns the RF pulses of a cycle and the cycles per average), sampling plan, where its amplitude comes from
  and its block list. `console.plan` is three stages; `Engine.clean_kspace` forms a scan by the entry. The six keep
  their closed forms and their code path. `mrgym validate` has a fifth group, `core`: rows J1 to J6 solve the
  registry's block list of each of the six with `mrgym.core` and compare with what the engine serves (shadow, not
  serving); the rows Q of `python -m mrgym.core.validate` take their block lists from the registry.
- **Image formation with an amplitude that changes from line to line** (`scanner/served.py::form_lines`): exact as a
  definition (the weight of line p is the amplitude of the sample that fills it), fast by one low-rank factorisation
  of (class, transmit node) x (sample of the cycle, readout sample), transformed along the phase-encode axis first.
  The truncation is bounded per k-space sample by 0.05 standard deviations of the receiver noise at 64 averages.
- **Four sequences on a console that lists them**: `fse` (echo-train length, echo spacing, effective echo time,
  centric or linear order, one refocusing angle), `mese` (one image per echo), `megre` (2 to 16 gradient echoes,
  monopolar or bipolar), `dwse` (b-value from the gradient limits; signal x exp(-b D)). Parameters, refusals with
  nearest valid values, scan time, RF power and header lines in `console.py`; the manual in
  `scanner/datasheet_core.md` ([docs/scanner-core.md](docs/scanner-core.md)); 34 validation rows (J7 to J40) against
  closed forms, the isochromat ensemble of the core, the phase graph of the design layer, a direct Fourier sum, the
  integral of the gradient waveform and the engine's own `spgr` and `se`.
- **An echo train reduces exactly to its refocusing flip angle** (docs/physics/core.md, section 11): the echoes and
  the periodic Mz of every slice column follow in closed form from three single shots on Chebyshev points of the
  cosine of that angle. This replaced the solve of every column, which is slower and, on the engine's slice columns
  and transmit nodes, was off by up to 6e-3 for long trains.
- **Tissue data, additive.** `tissue.ADC_SOURCES`: where each diffusion coefficient stands against a fetched abstract
  (9 of 25) or is declared (16); `tissue.water_adc_um2_ms` for phantom liquids. No number of the table changed: the
  spoiling model of `spgr` reads the same column. Found: the liver's 1.10 um^2/ms lies below its source (1.34 to 1.77).
- **Measured data.** Rung M3b of `tools/measured/run_all.py` runs the console's `mese` on the nine tubes of
  Ben-Eliezer 2015: with the published ratio of the two slices the measured T2 error (+50.5, +55.3, +82.6 %) lies
  between the two pulse shapes (+45.2 / +53.6, +51.1 / +61.6, +77.6 / +90.4 %); as served (refocusing slice 1.5 times
  the excitation slice) it is 6.0 to 18.4 points lower, now known difference 8.
- **Declared, with sizes** (docs/physics/limits.md, blocks 3a, 8, 8a): an ideal crusher ends every repetition of
  `fse`, `mese` and `dwse` (it leaves out 7e-3 of an echo at T1 900 ms, 0.16 for a fluid at TR 1500 ms); one
  refocusing angle per train; at most 32 echoes (`megre`: 16) and matrix 256 for the trains, for computing time (1.3 to
  6 s per scan on a busy machine, against 0.4 to 0.65 s for `se`); no fat saturation, no motion; isotropic free diffusion.
- **Not in this entry.** The tool layer of the environments (`envs/scanner_env.py`) does not offer the four sequences:
  no task can reach them yet. `satrec`, a generalised `ir`, a prepared gradient-echo train, magnetisation transfer,
  DESS, a real spoiler and STEAM are not served (core.md, section 11, says what each needs).

### Engine 0.3.0 joined with what was built beside it

Engine 0.3.0 (the entry below) was made on a tree that did not yet hold eleven task families, the signal core, the
spin systems, the RF field model, the receive arrays, the volume worlds, the fields in 3D and the comparison with
measured data. This entry is what it took to make each of them true on the corrected engine. No environment id, no
pass rule, no tolerance and no budget was changed; the engine is the one of the entry below, bit for bit
(`tests/golden/engine_digests.json`: every image, noisy scan and price of that entry is unchanged).

What compared itself with engine 0.2.0 and moved

- **The signal core** (`mrgym.core`; `python -m mrgym.core.validate`, 85 rows). Three things followed the engine, and
  in all three the side that moved is the core. (1) The slice: `Selective` and `operators.slice_columns` take one
  slice thickness as the width at half maximum of the small-flip-angle profile (`operators.slice_calibration`, found
  by bisection on the transform of the design layer's pulse; row C2 holds it equal to the engine's `rf.fwhm_factor`).
  Before, twelve of its 81 rows failed on engine 0.3.0; of the others only Q14 has another number.
  (2) `library.served` passes `phase_cycle_deg` of `bssfp` on as the advance of the RF phase; it had followed the sign
  of engine 0.2.0, which the core's own row Q10b had reported as wrong. Q10 and Q10b now compare at the same
  off-resonance, Q10b with what `Engine.amplitudes` serves. (3) New: the fat-saturation pulse as an event of a block
  list (`sat=(q, delay_ms)` of `spgr`, `epi`, `se`, `ir` and of `served`). Rows Q16 to Q16d reproduce the engine's
  closed forms, its phase graph and the amplitude maps it serves line by line: a second derivation of the saturated
  steady state by separate code.
- **The spin systems** (`mrgym.spins`). The typed copy of the engine's fat table (`metabolites.ENGINE_FAT_PEAKS`) has
  the main line at 0.694 and names the source the engine names; rows M1 and M1b pass again; the digest of the reviewed
  metabolite table is pinned anew for the six `triglyceride` rows, the only ones that moved.
- **The receive arrays** (`mrgym.scanner.array`). The fast formation treated fat as one species with one spectrum.
  Under fat saturation the engine now gives every line of fat its own amplitude map, and the fast formation was 26 %
  off the per-coil formation for a fat-saturated scan. Each line is now formed, weighted with the coil maps and given
  its own frequency by itself (`Engine._species`): 0.07 % for that scan, the level of a scan without saturation
  (`tests/test_array.py`, three fat-saturated cases). Every other row of `array_checks` passes with numbers that
  differ in the last digits (the slice of the default pulse is 0.6 % thinner).
- **The RF field model, the fields in 3D, the volume worlds, the design layer**: nothing to change. Their tables pass
  (`em` 82, `fields3d` 54, `design` 81 rows; the volume page is written again and differs in the engine it names and
  in its timings only).
- **The comparison with measured data** (`python tools/measured/run_all.py --write`, with `--fetch` from a local copy
  of the originals for rung M5): 33 of 33 rows with a tolerance pass, none of the 24 declared differences grew. Row
  M0.2 (T1 of sphere 5 of the NIST plate) is closed by the engine: it is now a row with tolerance 0 and no longer a
  declared difference. Row M4.1 shows 69.4 % for the main fat line. Known difference 11 now holds what is left: the
  model's glycerol peak at 4.31 ppm against the 4.2 ppm of the liver table. The SNR rows move by the 0.6 % of the
  slice calibration (M6.8: the Siemens scanners up to 3.7 times the engine, 3.8 before). The fat rows use neither a
  selective pulse nor fat saturation: M4.3, M8.1 and M8.3 print the numbers of before, M8.2 differs in one last
  digit. The stored
  fault x test matrix of the ACR tests (26 rows), which the regions that follow the phantom could have moved, was
  computed again on the joined tree: only its timings differ.
- **The 2D baseline** (`tests/golden/compat_2d.json`) was recorded again on engine 0.3.0
  (`python tests/compat_cases.py --write --rebase`). It exists to show that volume and array packages change no 2D
  exam, and it holds images, so it cannot outlive an engine version: a new engine version is the only legitimate
  reason for it to move. Before it was moved, the worlds, schemas, scanner cards, plans, headers, images and refusals
  it covers (314 entries) were computed on the tree of engine 0.3.0 without those packages and on the joined tree:
  equal bit for bit.
- **Tests that carried a second calculation**: `tests/test_t2_decay.py` (its own slice integral takes the calibrated
  thickness, found by its own bisection), `tests/test_fat_fraction.py` (the six peaks typed from the cited model; the
  echoes are compared on the fully sampled scan, and what homodyne detection does to a nearly cancelled
  opposed-phase echo of the stated scan is a test of its own), `tests/test_core_evaluators.py`.

The engine's own record

- A sixth corrected error had no line in the entry below: echo-planar partial Fourier with `pe_polarity` -1 acquired
  one line more than the realised protocol counts. It now has its test
  (`tests/test_engine_0_3.py::test_epi_partial_fourier_acquires_the_lines_the_protocol_counts`, which fails on 0.2.0).
- `docs/physics/limits.md`, block 0, lists the seven reported errors of engine 0.2.0: six corrected, each with the
  test that pins it, and one declared with its measured size (the columns of a `bssfp` slice, row R15).

Task families

- Statements made true. `Scanner/T1Vial-v0` said that partial Fourier lines are zero-filled; it now says what the
  homodyne image does to a read-out (the error of the stated method with `partial_fourier` 0.55 is 2.3 % on the
  instance of the test, 13 % on engine 0.2.0; the stated method itself is fully sampled and its numbers are those of
  before). The practice level of `Scanner/NullTissue-v0` proposes and acquires the fully sampled probe of the
  reference policy instead of the partial Fourier probe (`task.FIRST_PROBE`). `Scanner/EPISetup-v0` says why partial
  Fourier and fat saturation are fixed in it, and the module text of `Scanner/Resolve-v0` no longer says that partial
  Fourier makes the resolution anisotropic.
- `Scanner/FatSuppress-v0` is registered as not ready on this engine (`EnvSpec.not_ready`, a new field: the status
  line, the card and the task table of the front page say so and why). Its thresholds are part of the id. A
  successor `Scanner/FatSuppress-v1` with thresholds from the corrected physics is proposed, not built.
- Development gates on engine 0.3.0 (`mrgym gate <id> --n 60`, split `gate`, seeds 0 to 59), against the same gate
  on 0.2.0. Passes of 60 per level, before -> after:

  | Family, levels 1 / 2 / 3 | Reference | Most passes of a policy that must fail | Fixed scan list (open loop) | Verdict of the development gate |
  |---|---|---|---|---|
  | `T2Decay` | 60 / 60 / 60 -> 60 / 60 / 59 | 0 / 1 / 0 -> 0 / 1 / 0 | 60 / 60 / 60 -> 60 / 60 / 59 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `FatFraction` | 60 / 60 / 60 -> 60 / 60 / 60 | 1 / 0 / 0 -> 1 / 0 / 0 | 60 / 60 / 60 -> 60 / 60 / 60 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `ShiftFree` | 60 / 60 / 60 -> 60 / 60 / 60 | 1 / 1 / 0 -> 1 / 1 / 0 | 60 / 60 / 60 -> 60 / 60 / 60 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `GhostAside` | 60 / 60 / 60 -> 60 / 60 / 60 | 0 / 1 / 0 -> 0 / 1 / 0 | 60 / 60 / 60 -> 60 / 60 / 60 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `Resolve` | 60 / 59 / 59 -> 59 / 58 / 60 | 0 / 0 / 0 -> 0 / 0 / 0 | 60 / 59 / 59 -> 59 / 58 / 60 | admitted / admitted / practice -> admitted / practice / practice |
  | `FoldAway` | 60 / 59 / 60 -> 60 / 60 / 60 | 1 / 0 / 0 -> 1 / 0 / 0 | 60 / 59 / 60 -> 60 / 60 / 60 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `NoiseSNR` | 60 / 60 / 60 -> 60 / 60 / 60 | 1 / 0 / 1 -> 1 / 0 / 1 | 60 / 60 / 60 -> 60 / 60 / 60 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `GradientCal` | 60 / 60 / 60 -> 60 / 60 / 60 | 1 / 2 / 0 -> 1 / 2 / 0 | 60 / 60 / 60 -> 60 / 60 / 60 | admitted / practice / admitted -> admitted / practice / admitted |
  | `ErnstFlip` | 60 / 60 / 59 -> 60 / 60 / 59 | 1 / 1 / 0 -> 1 / 1 / 0 | 60 / 60 / 58 -> 60 / 60 / 59 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `T1Vial` | 60 / 59 / 60 -> 60 / 58 / 60 | 0 / 0 / 0 -> 0 / 0 / 0 | 60 / 59 / 60 -> 60 / 58 / 60 | admitted / admitted / admitted -> admitted / practice / admitted |
  | `EPISetup` | 59 / 60 / 59 -> 59 / 60 / 57 | 1 / 1 / 1 -> 1 / 1 / 1 | 59 / 60 / 56 -> 59 / 59 / 53 | practice / admitted / admitted -> practice / admitted / practice |
  | `FrequencyAdjust` | 60 / 60 / 60 -> 60 / 60 / 60 | 1 / 0 / 1 -> 1 / 0 / 1 | 60 / 60 / 60 -> 60 / 60 / 60 | admitted / admitted / admitted -> admitted / admitted / admitted |
  | `TransmitAdjust` | 60 / 60 / 60 -> 60 / 60 / 60 | 1 / 0 / 1 -> 1 / 0 / 1 | 58 / 59 / 57 -> 60 / 60 / 58 | admitted / admitted / admitted -> admitted / admitted / admitted |

  How to read the differences. The noise of a scan in an episode is drawn from a stream keyed on the realised
  protocol (`session._noise_rng`), and the realised protocol gained fields with engine 0.3.0: every scan of every
  episode has other noise than on 0.2.0, for the same seed. A reference that passes all but a few instances
  therefore moves by a few instances, and a development verdict that stood two instances above the bound can fall
  below it. `Scanner/EPISetup-v0`, level 3: 59 -> 57 on the same 60 worlds; one instance fails on both engines, two
  more on 0.3.0 on the fat rule (in one the reference misses the oil disc, which it finds from pixels that stand
  five noise standard deviations high). On the seeds 0 to 179 the reference passes 175 of 180 on engine 0.3.0 and
  174 of 180 on 0.2.0. `Scanner/T1Vial-v0`, level 2: 59 -> 58, one more instance beyond the tolerance; on the seeds
  0 to 359 the reference passes 352 of 360 on 0.3.0 and 351 of 360 on 0.2.0. `Scanner/Resolve-v0`: 60 / 59 / 59 ->
  59 / 58 / 60, one instance up or down at every level. The development verdicts of three levels (EPISetup 3,
  T1Vial 2, Resolve 2) fall to practice at n = 60, each with a lower bound just under the 90 % the gate asks for
  (87.6, 89.9 and 89.9 %); where it was measured on more seeds the rate is the one of before, and the gate of 300
  instances decides.
  `Scanner/T2Decay-v0` and `Scanner/FatFraction-v0` scan with `partial_fourier` 0.75, as their level 1 states and
  as their budgets were set. They pass as before, with larger errors, which is the price partial Fourier now has
  (95th percentile of the error of the reference: T2Decay 0.8 / 4.2 / 4.6 % -> 0.9 / 6.3 / 6.0 % of T2, tolerance
  10 %; FatFraction 1.7 / 1.8 / 1.8 -> 1.7 / 2.1 / 1.9 points, tolerance 3 points). No reference policy, oracle,
  tolerance or budget of the eleven families was changed, and none uses fat saturation, another pulse than the
  default or echo-planar partial Fourier in a scored scan.
- Interface sizes: the ceilings of the eleven families are raised by 550 characters, as those of the other ids were
  in the entry below (`tests/test_observation_budget.py`).

What a reader sees

- No task is marked ready: every stored check (n = 300) was made on engine 0.2.0. The paragraph under the task table
  of the front page says so, and `tests/test_front_page.py` holds the table to it.
- The training result (README, `docs/RECIPE.md`, `docs/experiments/RESULTS-27b.md`) names the engine it was measured
  on, 0.2.0, and says that the runs have not been repeated on 0.3.0. The sentence is written by
  `docs/experiments/27b/report.py` from the commits of the runs.
- The map (`mrgym map`): shimming is served in part (the two in-plane terms of a scan, row K34; the 3D library
  computes more), the centre frequency and fat saturation are formed, the transmit factor of a scan is evidence of
  the transmit item (K35, K36) and priced (C9). Fat saturation, the `bssfp` phase cycle and the slice also cite the
  rows of the signal core.
- `mrgym check`: a hidden value is looked for as a number, not as a run of digits. The 150 ms of the
  `Scanner/NoiseSNR-v0` phantom was "found" in a read-out of 0.0121506732 once the scan data of engine 0.3.0 held
  those digits (`check.served`, `tests/test_env.py`).

Golden data and fingerprints, against the files of engine 0.2.0

- `tests/golden/engine_digests.json`: against the file of the engine entry below (engine 0.3.0 before the merge) nothing
  changed: 54 images, 8 noisy scans, 30 prices. Against engine 0.2.0, as that entry says: 41 of the 46 images that
  existed (every slice-selective scan; the five that stayed are the nonselective, fully sampled scans without fat
  saturation), all 8 noisy scans, no price, 8 new images. The section of the receive arrays is new on 0.3.0: its 3
  image entries moved against 0.2.0 (slice-selective scans), its maps, covariance and g map did not.
- `tests/golden/task_snapshots.json` and `src/mrgym/compat/fingerprints.json`: the eleven families are new on engine
  0.3.0 (engine id, scoring source, transcript; `Scanner/EPISetup-v0` and `Scanner/ShiftFree-v0` also in their
  statements, which print a signal-to-noise floor computed on the hidden world; `Scanner/FatFraction-v0` in the table
  of fat peaks; `Scanner/T1Vial-v0` in the sentence about partial Fourier). Of the ids of the engine entry only
  `Scanner/NullTissue-v0` moved (the probe of its practice level). The two retired ids are frozen.
- `tests/golden/volume_digests.json`: unchanged (no scan is in it).
- `tests/golden/compat_2d.json`, recorded again: of its worlds one entry moved (the tissues of `nist_t1`: T1 of
  sphere 5); every schema, scanner card, plan and refusal (four new parameters); all 30 images.

### Engine 0.3.0: five behaviours that contradicted textbook practice are corrected; four adjustment parameters

Engine: `mrgym-engine-0.3.0`. An audit of the engine against textbook physics reproduced four wrong behaviours on
0.2.0; a fifth (the sign of the `bssfp` phase cycle) was found by a review of the design layer the same day. Each is
now a failing-then-passing test with the textbook number (`tests/test_engine_0_3.py`) and one or more rows of
`mrgym validate` against a separately written calculation (`docs/physics/validation.md`).

What changed in the physics

- **Partial Fourier costs SNR.** On 0.2.0 the lines that were not acquired were zero-filled: the scan got shorter,
  the noise of a pixel fell, and the header's relative SNR rose (5.66 / 6.53 / 7.54 for `partial_fourier` 1 / 0.75 /
  0.55 at 64 lines). The console now synthesises the missing lines by homodyne detection (Noll, Nishimura &
  Macovski 1991, step weighting; `kspace.Engine.recon`): the lines without a conjugate partner weigh 2, the image is
  the component in phase with a low-resolution image of the symmetric lines. Resolution is the nominal one where
  the phase is smooth; the noise of a pixel is that of a full scan times sqrt((3 L - 2 l - 3) / L) for l of L lines
  (`console.noise_lines`, `Engine.noise_sd`), and the relative SNR falls by that factor (rows K15, K17, K29, K30).
  The phase of such an image is the low-resolution phase and its background is not Rayleigh distributed. A fully
  sampled scan is reconstructed as before.
- **Fat saturation acts on the lines inside its band.** On 0.2.0 a saturation pulse on the fat line removed the
  whole six-line spectrum. Each line is now saturated by its own place in the band of the pulse (centre
  `fatsat_offset_hz`, hidden width, edges 8 Hz soft), water as one line: a band of 1.3 ppm leaves 21.9 % of the fat
  signal, and the two lines next to water go only with water (rows K28, K31).
- **Fat saturation drives a steady state.** On 0.2.0 the saturated signal was the unsaturated one times a constant
  at every TR and flip angle. The pulse is now an event in every repetition, `fatsat_delay_ms` (7 ms with the
  default pulse) before the excitation: the longitudinal magnetisation of a saturated line recovers during that
  time and its periodic state depends on TR, T1 and the flip angle; closed form under ideal spoiling
  (`signal.mz_saturated`: `se`, `ir`, `epi`), the longitudinal states of every order in the RF-spoiled steady state
  of `spgr` (`epg`), through the slice profile for selective scans (rows S16, S17, R14, K32).
- **The slice gradient is calibrated per pulse.** On 0.2.0 one "slice thickness" was the nominal bandwidth of the
  pulse, so `rf_pulse: fast` excited 1.20 (1 degree) to 1.40 (90 degrees) times `slice_mm`. The gradient is now
  `rf.fwhm_factor` (1.197 / 0.994 / 1.000 for `fast` / `normal` / `sharp`) times the nominal one, so that the full
  width at half maximum of the small-flip-angle profile is `slice_mm` for every pulse (row R13); the pulses still
  differ in their transition band and in how the profile broadens with the flip angle. The thinnest selective
  slice is 1.46 mm (`fast` 1.76 mm, `sharp` 1.47 mm; 1.47 mm for all three before). Through-slice dephasing
  (`rf.dephasing_factor`) follows the calibrated profile.
- **The RF phase advance of `bssfp` moves the bands up in frequency.** `phase_cycle_deg` is the advance of the RF
  phase per TR, counted the way a faster spin precesses (`docs/physics/conventions.md`, section 5). The steady state
  depends on how far a spin runs ahead of the pulse, 2 pi df TR - phase cycle; 0.2.0 computed it with
  2 pi df TR + phase cycle, so a phase cycle of 90 degrees at TR 5 ms put the dark band at -50 Hz where a
  pulse-by-pulse Bloch iteration has it at +50 Hz. 0 and 180 degrees are the same in both. Rows S18 and K11 (which
  now has phase cycles of 90 and 300 degrees; with 180 degrees alone it could not see the sign).
  `Scanner/BandFree-v0` used the old sign consistently in its verifier, its oracle and the band model of its
  controls; all three follow the engine now, and the constant protocols of `fixed_recipe` are the mirror images
  (360 degrees minus the old phase cycle) of the tuned ones. A deliverable written for 0.2.0 scores on 0.3.0 as its
  mirror image did.

New protocol parameters (every sequence; defaults change nothing; they cost no scan time): `cf_shift_hz` (-500 to 500
Hz, added to the centre frequency of the scanner: displacement, echo phase, bSSFP bands and the fat-saturation band
move together), `tx_scale` (0.5 to 2, a factor on the transmit amplitude of the excitation and refocusing pulses; RF
power x `tx_scale`^2, peak B1 x `tx_scale`), `shim_y_hz_per_mm` and `shim_x_hz_per_mm` (-2 to 2 Hz/mm, linear field
terms in patient coordinates). Each compensates a hidden state that existed already (`cf_offset_hz`, `tx_gain`, the
B0 field): rows K33 to K36 and C9. `plan` prints the ones that differ from their defaults, the header carries them
after `ADJ`, the scanner log stores them with the realised protocol. `scanner_info` has the topic `adjustments`. The
word `shim` is still refused as NOT_SIMULATED, and the refusal now names the parameters.

Two tables pinned to the sources they cite (found by a comparison with public measured data; in the same engine bump,
`tests/test_cited_tables.py`)

- `tissue.FAT_PEAKS` is the six-peak fat model that Hernando et al. 2017 list (their shifts, minus the 0.1 ppm they add
  for a phantom at room temperature), not Table 3 of Hamilton et al. 2011, which engine 0.2.0 named as its source. The
  citation is corrected and the amplitude of the main peak is the model's 0.694 (0.693 before; the other five were
  right). The model puts the glycerol peak at 4.31 ppm where Hamilton's table has 4.2 ppm; its amplitudes are within
  0.8 points of the table's. This is stated, not changed.
- `anatomy.NIST_T1`: T1 of sphere 5 is 496.7 ms, the value of Table S1 of Stupic et al. 2021 (496.0 before, the
  number in the analysis code of the ISMRM reproducibility challenge, from which the list had been taken). The list
  now cites the table.

What is not comparable across the bump

- Every slice-selective scan serves other numbers than on 0.2.0 (the default pulse by 0.6 %, `fast` by 16 %), every
  partial Fourier scan, every scan with fat saturation, and every `bssfp` scan with a phase cycle other than 0 or
  180 degrees. Scans with `excitation: nonselective`, full sampling and no fat saturation are bit for bit those of
  0.2.0 (and of 0.1.0: `tests/test_engine_version.py`), `bssfp` at those two phase cycles included.
- Logs recorded on 0.2.0 replay only there. Every stored gate result (`src/mrgym/envs/results/*.gate.json`, n = 300)
  was measured on 0.2.0 and is stale: the seven ids registered as admitted are registered as candidates until their
  gates are measured again, `mrgym list` and the cards say that the gate has not been run on this engine and no
  longer repeat the old verdict (`gate.status_line`), and `mrgym check` asks for nothing of a candidate.
- Model results measured on 0.2.0 are not comparable with results on 0.3.0 for any id: the tool schemas gained four
  parameters and one topic, and the data of every default scan moved.
- `Scanner/FatSuppress-v0`: with the thresholds of its task statement (at most 5 % of the fat signal left, fat
  saturation "enough" below 3 %) spectral fat saturation now passes in no region, because the lines outside the
  band keep 9 to 31 % of the signal; the best answer is inversion recovery in every region and the second pass rule
  cannot fail. The reference, the oracle and the scoring table are correct for the corrected physics; the claim of
  the family (a choice between two methods) is not supported on this engine and needs new thresholds under a new id.
- `Scanner/ProtocolToSpec-v0`: the reference read the noise from the background of its partial Fourier scouts by the
  Rayleigh mean, which a homodyne image does not have; it now uses the mean of one Gaussian component.
- `Scanner/NullTissue-v0`: the control policies probed with matrix 64 and `partial_fourier` 0.6, the cheapest probe of
  0.2.0 (0.6 of the time and less noise). On 0.3.0 that probe has 1.3 times the noise of the full scan and a biased
  signed mean where a region is near its null (`docs/physics/limits.md`, block 5a): with it the reference passed
  57 and 55 of 60 instances of levels 2 and 3 (`mrgym gate Scanner/NullTissue-v0 --n 60`; 60 and 59 on 0.2.0). The
  probes are now fully sampled at matrix 32 (`controls.PROBE`): 60 and 59 of 60. No rule, tolerance or budget was
  changed. The practice level still proposes the old probe in its task text (`task.FIRST_PROBE`): not changed here.

Interface sizes (`tests/test_observation_budget.py`, `tests/test_interface.py`): the ceiling of every id is raised by
550 characters (ArtefactClinic 700) for the four parameters and the topic, `plan` 4000 to 4380, `scan` 2450 to 2600,
the `physics` index 700 to 750 characters.
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
