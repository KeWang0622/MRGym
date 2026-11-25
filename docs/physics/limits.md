# Limits: what is simplified

The virtual scanner computes a stated set of equations. This page lists what those equations leave out, in the
order a pulse-sequence physicist would probe. Each block has five lines: what is simplified, what a real scanner
does, the size here, what it means for a task on this scanner, and whether a change is planned.

Rules of this page:

- A size is a number only if a command in this repository prints it. The command stands next to the number.
  Otherwise the size reads "not measured".
- "Planned" refers to [ROADMAP.md](../ROADMAP.md). Its entries are plans, not promises.
- `mrgym validate` checks the scanner against formulas and against calculations written separately by the same
  authors ([validation.md](validation.md)); [external.md](external.md) compares it with another group's code and
  with published operating points. No image here was compared with a measurement on a real scanner.

**Out of scope.** MR Gym has never been compared with data from an MR scanner. It computes a stated model. Not
modelled, and not testable with `mrgym validate --perturb`: magnetisation transfer and exchange; restricted or
anisotropic diffusion; flow, perfusion, physiological motion (one periodic line-modulation model exists, block 7);
eddy currents, concomitant fields, gradient impulse response; SAR in W/kg, local SAR, nerve stimulation; B0 drift
and heating; anything through the slice beyond independent columns (one 2D world: no slice positioning, no
multi-slice cross-talk, no slab profile or slab wrap in the 3D mode); parallel transmit; vendor pulses (the
excitation is a windowed sinc), vendor filters and distortion correction; coil loading and receive arrays. A
policy learned here is evidence about reasoning with the modelled physics, not a protocol for a patient.

Commands used below:

```bash
python examples/textbook/01_ernst_angle.py                 # served steady state against the Ernst formula
python examples/textbook/07_gibbs_ringing.py               # ringing against pixel size
python examples/textbook/08_noise.py                       # noise statistics
python examples/textbook/12_approach_to_steady_state.py    # first sampled line against the periodic state
python examples/textbook/13_slice_profile_vfa.py           # two-angle T1 on a 2D slice against 3D
mrgym validate --group rf                                  # the slice-selective pulse and the slice integral (rows R1 to R12)
python tools/external/run_all.py                           # the external ladder (rows L1.1 to L3.10)
mrgym timing                                               # seconds of computation per scan
mrgym scan sequence=se tr_ms=3000 te_ms=90 matrix=256      # prints the header line with TA and SAR
python -c "from mrgym.scanner import world; print(world.DEFAULT_SPEC, world.B1_SHAPE, world.GRID)"
python -c "from mrgym.scanner import session; print(session.REPLAY_RTOL, session.FINGERPRINT_BLOCKS)"
```

`mrgym scan` also writes a PNG under `mrgym_out/`.

---

## 1. Slice selection: independent columns, one pulse shape

- **Simplified.** A selective scan (`excitation: selective`, the default) excites the slice with a Hann-windowed
  sinc of 128 steps (`rf_pulse`: time-bandwidth product 2, 4 or 8). The rotation at every height through the slice
  is the exact product of the 128 steps; nothing relaxes while the pulse plays. The signal of a voxel is the
  integral, over independent columns through the slice, of the sequence's signal at the local flip angle. The
  object and every field are taken as constant through the slice, except for one linear term: the through-slice
  gradient of B0, which multiplies gradient echoes by the small-tip dephasing factor of the pulse. `bssfp` treats
  the rotation of each column as a rotation about the pulse axis. `excitation: nonselective` serves one flip angle
  per voxel (hard pulses) and is priced as a 3D scan; its slab profile and slab wrap are not modelled.
- **A real scanner** plays vendor pulses with other profiles, on an object that changes
  through the slice, and its neighbouring slices disturb each other.
- **Size here.** What the profile does: `python examples/textbook/13_slice_profile_vfa.py` prints the two-angle T1
  of a 2D slice with the nominal angles as -45 to -76 % (time-bandwidth 2), -37 to -70 % (4) and -19 to -53 % (8)
  of the truth; published for uncorrected 2D scans: 40 to 80 % (Svedin & Parker 2017), -63 % (Dieringer 2014).
  Numerics of the integral: rows R4 and R5 of `mrgym validate --group rf` (7.9e-05 relative for TR of 20 ms and
  more, 6.1e-04 at TR 5 ms, where the wings of the profile outweigh its centre). Relaxation during a 2 ms pulse:
  0.05 % of the signal in the case of row L1.3 of `python tools/external/run_all.py`. Other pulse shapes:
  `mrgym validate --perturb <id>` re-scores an environment with time-bandwidth 2 and 8, a Hamming window and a
  rectangular profile. The difference to a vendor pulse is not measured.
- **For a task.** Flip-angle methods on a 2D scan are biased, as on a scanner, and the agent has to correct for the
  profile or pay for 3D. The correction learned here is for this pulse family.
- **Planned.** Multi-slice acquisition is on the roadmap; other pulse designs are not.

## 2. Steady state served, transient not simulated

- **Simplified.** The console charges dummy time: 4 s for `spgr`, `afi` and `bssfp`, one cycle for `se` and `ir`,
  two shots for `epi`. Every phase-encode line is then served at the periodic steady state.
- **A real scanner** starts sampling while the magnetisation still approaches the steady state. The first lines
  carry more signal, and the error depends on the order of the lines.
- **Size here.** `python examples/textbook/12_approach_to_steady_state.py` computes the missing transient with the
  Bloch chain of `mrgym.scanner.bloch` (ideal spoiling, start from equilibrium). Worst case over TR 8, 15, 30 ms
  and flip angles 1 to 30 degrees, at 3 T:

  | Material | T1 (ms) | First sampled line above the periodic state | Centre of a 128-line scan |
  |---|---:|---:|---:|
  | white matter | 838 | +0.06 % | +0.02 % |
  | grey matter | 1607 | +1.21 % | +0.52 % |
  | blood | 1932 | +2.22 % | +1.06 % |
  | longest-T1 vial of the QA phantom | 2500 | +4.60 % | +2.46 % |
  | cerebrospinal fluid | 4300 | +15.50 % | +9.72 % |

  The worst case is TR 15 ms at 5 degrees in every row. The same script gives the spin-echo case. With exact 90
  and 180 degree pulses one dummy cycle reaches the periodic state. With a transmit factor of 0.8 or 1.2 the first
  line of white matter at TR 500 ms is 21.8 % below or 13.2 % above it, and each further cycle multiplies the
  error by -0.14 or +0.14. The transients of the RF-spoiled steady state and of `bssfp` are not measured.
- **For a task.** Low-flip-angle data of long-T1 material are cleaner here than on a scanner. A variable-flip-angle
  protocol needs no preparation time beyond what the console charges.
- **Planned.** Listed on the roadmap as not simulated; no date.

## 3. Spoiling and diffusion model of `spgr` and `afi`

- **Simplified.** The steady state is an extended-phase-graph solution with instantaneous pulses, T1, T2,
  quadratic RF phase cycling and one unit of spoiler dephasing per TR; with a selective pulse it is solved for
  every column through the slice at that column's flip angle (block 1). Diffusion enters with the b-factors of a gradient that is
  constant over the whole TR, for isotropic free diffusion in a single pool. Readout and phase-encode moments add
  no diffusion weighting. There is no magnetisation transfer and no exchange.
- **A real scanner** plays the spoiler as a lobe. Tissue has restricted, anisotropic diffusion and a bound pool.
- **Size here.** At the default moment the lobe lasts 3.39 ms of a 15 ms TR
  (`python examples/textbook/01_ernst_angle.py`). The same script prints served signal over the Ernst formula
  for six tissues at 3 T, flip 5 to 60 degrees, TR 8 to 30 ms, RF increment 117 degrees: 0.965 to 1.293 at
  100 mT·ms/m, 0.878 to 1.001 at 300, 0.944 to 1.001 at 600. The error of the diffusion model itself is not
  measured: rows S5 to S8 of `mrgym validate` check the solver, not the model.
- **For a task.** The deviation from the Ernst formula is the model's answer. Its direction and rough size are
  credible; its exact value for a given tissue is not established.
- **Planned.** Not on the roadmap.

## 4. Single slice

- **Simplified.** One axial slice through the isocentre.
- **A real scanner** interleaves slices within a TR. That brings cross-talk between neighbours, magnetisation
  transfer from off-resonant pulses, slice timing and a trade of coverage against time.
- **Size here.** Not measured.
- **For a task.** There is no coverage to plan and no slice gap to choose. Scan time is the time of one slice.
- **Planned.** On the roadmap ("Multi-slice acquisition and its pricing").

## 5. One receive channel

- **Simplified.** One channel with a smooth complex receive profile. Noise is white, Gaussian and added per
  k-space sample. There is no coil array, no parallel imaging, no g-factor and no noise correlation.
- **A real scanner** combines many coils. Background noise of a sum-of-squares image is non-central chi, and
  accelerated scans have position-dependent noise.
- **Size here.** The background magnitude is Rayleigh: mean 1.253 sigma, standard deviation 0.656 sigma
  (`python examples/textbook/08_noise.py`; row K16).
- **For a task.** Noise can be read from any empty region and is the same everywhere. Acceleration is not a way
  to buy time.
- **Planned.** On the roadmap ("Receive arrays").

## 6. Piecewise-constant tissue, schematic anatomy

- **Simplified.** A tissue class has one value of proton density, T1, T2, T2' and diffusion coefficient per
  session. The head, the abdomen and the QA phantom are drawn from implicit functions: layers, slits for sulci,
  ellipses for ventricles, nuclei, organs and vessels. Boundaries are 0.5 mm soft. The object lives on a master
  grid of 640 points, 0.4 mm for the head and the QA phantom and 0.66 mm for the abdomen; the smallest pixel is two
  grid steps.
- **A real scanner** images tissue with texture, gradual transitions and structure down to the cell.
- **Size here.** `python examples/textbook/07_gibbs_ringing.py`: the overshoot at a sharp edge is 8.93 % at 4 mm
  pixels, 8.64 % at 2 mm and 7.61 % at 1 mm, against 8.95 % for a continuous step. With the 0.5 mm soft edge of the
  shipped phantoms it is 8.38 %, 6.67 % and 2.13 %. Realism of the anatomy is not measured.
- **For a task.** The spread of a region read-out is noise plus field shading, never biology. Small pixels show
  less ringing than a real edge would.
- **Planned.** Not on the roadmap.

## 7. Motion is a line modulation of one tissue class

- **Simplified.** A world may hold movers (`world.make_motion`; no shipped environment has one by default): a
  tissue class whose signal is multiplied by `1 + depth x cos(2 pi f t + phase)` (pulsatile flow), or whose
  position is displaced by `amp x cos(2 pi f t + phase)` along one patient axis (respiration). Each phase-encode
  line of that class is modulated with the time at which the line is acquired, averaged over the averages of the
  line. There is no spin history, no inflow, no flow void, nothing through the plane; the place a displaced class
  leaves is not filled; the phase of the motion is a fixed function of the protocol, so a repeated scan shows the
  same ghost. Without movers nothing moves: blood is a static tissue class.
- **A real scanner** shows ghosts along phase encoding from periodic motion, blurring from irregular motion, flow
  voids and inflow enhancement. In the abdomen these set the protocol.
- **Size here.** Rows K25 to K27 of `mrgym validate`: copies at `lines x frac(f x TR x averages)` pixels within
  half a pixel, their amplitude within 5 % of `(depth / 2) x |sin(pi f TR averages) / (averages x sin(pi f TR))|`,
  a frozen displacement within 0.05 mm. Agreement with a moving object in a scanner is not measured.
- **For a task.** Ghost position and amplitude respond to TR, averages and the phase-encode direction as the
  formulas say, including the fact that averaging is not a 1 / sqrt(averages) remedy. Anything that needs spin
  history (inflow, saturation of moving blood) is absent.
- **Planned.** Not on the roadmap.

## 8. No fast spin echo

- **Simplified.** `se` and `ir` acquire one phase-encode line per TR.
- **A real scanner** uses echo trains for T2-weighted and FLAIR imaging and pays with blurring and RF power.
- **Size here.** `mrgym scan sequence=se tr_ms=3000 te_ms=90 matrix=256` prints TA 12:54.0 and SAR 1 %.
  `mrgym scan sequence=ir tr_ms=9000 ti_ms=2500 te_ms=90 matrix=256` prints TA 38:36.0 and SAR 2 %. The RF power
  limit comes into reach only for short TR and large flip angles:
  `mrgym scan sequence=bssfp tr_ms=4.5 flip_deg=90 matrix=64 bandwidth_hz_per_px=1000 rf_pulse=fast` is refused
  at 3 T with 153 % of the limit and prints, with `--field 1.5`, SAR 38 %.
- **For a task.** Scan time, not RF power, limits T2-weighted and FLAIR protocols. The trade of echo-train length
  against blurring and power does not exist.
- **Planned.** On the roadmap ("Fast spin echo").

## 8a. Spin echoes: crushed means gone

- **Simplified.** In `se` and `ir` every transverse component that the crushers dephase is lost for good, and the
  longitudinal magnetisation of a column evolves as one number. A refocusing pulse that is not 180 degrees (a
  transmit error, or the edge of its slice profile) in fact stores part of the dephased magnetisation along z as a
  fine grating, and identical crushers recall part of it in the next repetition: a stimulated echo on top of the
  spin echo.
- **A real scanner** has this pathway whenever TR is not much longer than T1, and sequences reduce it with crusher
  schemes and phase cycling.
- **Size here.** Row L1.4 of `python tools/external/run_all.py`: with the grating kept, the independent integrator
  gives -2 to +39 % of the served signal in its five cases (TR 300 to 2000 ms, T1 500 to 1200 ms, B1 80 to
  115 %). The served model itself agrees with the integrator to the tolerance of that row.
- **For a task.** Spin-echo signal against TR and B1 is that of a sequence whose crushers work perfectly between
  repetitions. An absolute spin-echo signal at short TR should not be carried over to a scanner.
- **Planned.** Not on the roadmap.

## 9. Idealised EPI

- **Simplified.** Single-shot gradient-echo EPI with flat-top sampling only. Decay and off-resonance advance from
  line to line, not within a line. Partial Fourier lines are zero-filled. The ghost comes from one constant delay
  between gradient and acquisition. The signal amplitude is the Ernst steady state at `tr_ms`. Fat saturation is
  one efficiency per session inside a fixed band.
- **A real scanner** samples on ramps, corrects odd and even echoes with a measured phase, has eddy currents and
  usually halves the distortion with parallel imaging.
- **Size here.** `mrgym scan sequence=epi tr_ms=3000` prints ESP 0.91 and WFS 37.9 px (phase) at 3 T: fat would
  move by 37.9 pixels without saturation. Agreement of the distortion with its formula is row K12 and
  `python examples/textbook/10_epi_distortion.py`. The difference from a real EPI readout is not measured.
- **For a task.** Distortion is that of an unaccelerated readout. It can only be reduced by a shorter echo
  spacing, fewer lines or a blip reversal.
- **Planned.** Not on the roadmap.

## 10. Simplified fields

- **Simplified.** Head: the field of three air spheres below the slice. Abdomen: gas spheres in the slice plane.
  Both get a first-order shim over the brain or the abdominal cavity and a smooth random term. QA phantom: a linear
  plus saddle field. The shim and the frequency adjustment are fixed for the session. The transmit pattern is a closed form
  whose amplitude depends on the field strength. The receive profile is unrelated to the transmit pattern. There
  is no gradient non-linearity (the world-spec key `gnl` warps positions for sensitivity studies only; no shipped
  environment sets it), no concomitant field, no eddy current and no drift. A turned or shifted field of view sees
  the same fields: they are functions of the patient position.
- **A real scanner** has a field shaped by the whole anatomy, lets the operator re-shim, and warps images away
  from the isocentre.
- **Size here.** The amplitudes are constants of the package, printed by the `world` command above: transmit
  pattern 3 % (1.5 T) and 11 % (3 T) for the head, 7 % and 26 % for the abdomen. Each is varied by up to 20 % per
  session (`world.anatomy_fields`).
  Agreement with in-vivo field maps is not measured.
- **For a task.** Off-resonance can be mapped but not corrected at the source. Geometry is exact once
  off-resonance is accounted for.
- **Planned.** Not on the roadmap.

## 11. bSSFP: steady state only

- **Simplified.** The closed-form steady state with the local off-resonance. No transient and no catalyst. Inside
  the steady state fat is a single line at -3.4 ppm; the six peaks act only during the readout.
- **A real scanner** needs a preparation, shows transient oscillations, and has finite pulses.
- **Size here.** Not measured. Band positions against their formula: row K11 and
  `python examples/textbook/09_bssfp_bands.py`.
- **For a task.** Bands sit where the formula puts them. Signal right after the start of a scan is not available.
- **Planned.** Not on the roadmap.

## 12. RF power is not SAR; no nerve stimulation limit

- **Simplified.** The console adds B1 squared times duration over the pulses of one cycle, divides by the cycle
  time and compares with a public limit on B1rms: 3.0 µT at 3 T, 6.0 µT at 1.5 T. A selective pulse counts with
  the energy of its shape (1.80, 3.11 or 7.16 times a hard pulse of the same duration and angle; row C4) and is
  lengthened if its peak would exceed 20 µT, a choice of this package; a nonselective pulse is a hard pulse.
  Everything is computed from the nominal flip angle. The header calls the result SAR. There is no body model, no
  averaging window and no local SAR. Gradients are checked against an amplitude limit and played at a fixed
  slew rate; there is no stimulation model.
- **A real scanner** predicts whole-body and local SAR from the patient and the coil under IEC 60601-2-33, and
  limits dB/dt.
- **Size here.** Not measured against any SAR model. The transmit-gain error of a session (3 to 12 %, printed by
  the `world` command above) changes the delivered power and is ignored by the console.
- **For a task.** The percentage is a consistent price for RF, not a safety figure. It cannot be converted to
  W/kg.
- **Planned.** Not on the roadmap.

## 13. Console timing is a model

- **Simplified.** Minimum TE and TR come from a short list of lobes: prephaser, readout ramp, spoiler, crushers,
  and for a selective pulse the ramp of the slice gradient and its rephasing lobe (which overlaps the prephaser).
  Phase encoding takes no time of its own. The slice gradient of the refocusing pulse and the partition encoding
  of the 3D mode take none. The readout may use the full gradient amplitude; the other lobes, and the slice
  gradient, use 80 % of amplitude and slew rate.
- **A real scanner** fits a full waveform to its hardware and adds vendor-specific margins.
- **Size here.** Not measured against a vendor console.
- **For a task.** Minimum times order protocols correctly. They should not be read as the limits of a product
  sequence.
- **Planned.** Not on the roadmap.

## 14. Tissue values are not citation-grade

- **Simplified.** `src/mrgym/scanner/tissue.py` tags every number: [A] stated in the abstract of the cited paper,
  [T] attributed to a table of the paper and not re-checked against its full text, [C] chosen by the authors
  inside a plausible range. Proton densities, most T2* values and all lesion values are [T] or [C]. Each session
  scales T1, T2 and T2* of every class by a random factor with 5 % spread (`anatomy._jitter`). QA-phantom vials are drawn from
  ranges and are test-object values; an oil vial is the subcutaneous-fat row with its T1 scaled by a factor drawn
  within +-20 % per vial (`anatomy.QA_SPEC["oil_t1_jitter"]`), so that the inversion time that nulls oil differs
  between worlds.
- **A real scanner** measures what the subject has.
- **Size here.** Not measured. [references.md](references.md) states how far each source was checked.
- **For a task.** Values are of the right size. Do not quote them as literature values, and do not expect a
  tissue to be recognisable from its numbers alone.
- **Planned.** Not on the roadmap.

## 15. Noise level and system errors are choices

- **Simplified.** The receiver noise density, the transmit-gain error (3 to 12 %, either sign), the
  centre-frequency error (up to 15 Hz at 3 T), the inversion efficiency (0.92 to 0.98), the fat-saturation
  efficiency (0.88 to 0.97), the offset of the fat-saturation pulse from the fat line (0 by default) and the
  residual gradient delay (up to 0.4 µs) are drawn per session from ranges set in `world.DEFAULT_SPEC`. An environment may set other ranges in its world specification.
- **A real scanner** has errors that depend on the coil, the load and the adjustment procedure.
- **Size here.** The ranges are printed by the `world` command above. They are not fitted to any scanner.
- **For a task.** The errors have realistic signs and mechanisms. Their sizes say nothing about a particular
  system.
- **Planned.** Not on the roadmap.

## 16. Replay across machines

- **Simplified.** A scan is a deterministic function of the seed, the protocol and the scan index. The log stores
  a SHA-256 digest of the float32 images and the mean magnitude of 8 x 8 blocks of every image. A replay on the
  same numerical stack reproduces the digest ("bit-exact"). On another machine, BLAS or FFT build the last bits can
  differ; the replay then accepts block means within a relative tolerance of 1e-4 of the largest block
  ("tolerance") and reports which mode it used (`src/mrgym/scanner/session.py`).
- **A real scanner** never repeats a scan exactly.
- **Size here.** The two constants are printed by the `session` command above. The size of the difference
  between two machines is not measured. Logs replay only on the engine version that wrote them: engine 0.2.0
  (slice-selective excitation by default) serves other numbers than 0.1.0 for every protocol. The hard-pulse
  signal of 0.1.0 is `excitation: nonselective` (row R6), at the price of a 3D scan.
- **For a task.** A verifier on another machine can confirm a log. In tolerance mode it confirms 64 block means
  per image, not every pixel.
- **Planned.** Not on the roadmap.

## 17. In-process environments are not a security boundary

- **Simplified.** `mrgym.make(...)` runs the scanner inside the caller's Python process. A model that acts through
  tool calls sees observations only. Code in the same process can read the seed with `env.episode_config()` and
  rebuild the hidden world; the textbook examples do exactly that.
- **A real scanner** hides the patient by being a separate machine.
- **Size here.** Not applicable.
- **For a task.** A reward earned in process is valid for a tool-calling model. For an agent that runs code, use
  the HTTP service (`mrgym serve`, one process per episode) and keep the episode configuration with the operator.
- **Planned.** Harbor task directories are on the roadmap; they are not written.

## 18. What the admission gate shows

- **Simplified.** A level of an environment is admitted when, on fresh instances, a reference policy passes, every
  listed shortcut policy fails and every listed ablation loses (`src/mrgym/gate.py`, `mrgym gate <id>`).
- **A real examination** has no such gate.
- **Size here.** The stored results are in the environment cards under `envs/`.
- **For a task.** The gate shows that the listed shortcut policies fail. It does not prove that no shortcut
  exists.
- **Planned.** More environments are on the roadmap.

## 19. Which simplification changes a scored answer

- **Simplified.** The engine is one set of choices: a pulse shape, a noise level, field sizes, edge widths, tissue
  values, an ideal receiver. What is right in an environment could depend on any of them.
- **A real scanner** differs from all of these choices at once.
- **Size here.** `mrgym validate --perturb <id>` replays the reference policy of an environment with one choice
  changed per row, scores it against the changed world, and stores the table (the Validity section of the
  environment's card). For `Scanner/NullTissue-v0`, level 2, on 60 paired instances
  (`mrgym validate --perturb Scanner/NullTissue-v0 --level 2 --n 30 --draws 2 --write`, engine `mrgym-engine-0.2.0`): unperturbed the reference passes 60 of 60.
  The pass rate falls (class flips): `magnitude_only` (0/60), `motion_ghost` (15/60).
  The right answer moves and the reference follows it by measuring: `relax_shift` (60/60), `inv_weak` (60/60).
  Robust: `rf_fast` (60/60), `rf_sharp` (60/60), `rf_hamming` (60/60), `rf_rect` (60/60), `ideal_spoiling` (60/60), `fat_singlepeak` (60/60), `noise_x0.5` (60/60), `noise_x2` (60/60), `tx_gain_x2` (60/60), `b1_strong` (60/60), `b0_x5` (60/60), `cf_x5` (60/60), `edge_sharp` (60/60), `edge_soft` (60/60), `adc_low` (60/60), `fatsat_weak` (60/60), `fatsat_off` (60/60), `grad_delay_x5` (60/60), `gnl` (60/60).
  Noise-like: `rx_surface` (57/60).
  Undecided at this number of instances: none.
  Not run for want of the physics: `transient`, `array8_sos`, `spoiler_lobe`, `afi_phase_per_pulse`, `fov_free`. The other environments have no stored table.
- **For a task.** A row that flips without being followed says that the answer of that environment depends on a
  choice of the engine; its card has to say whether the choice is modelled, randomised or declared. A table on one
  engine is not a comparison with a scanner.
- **Planned.** The rows that are not run need the approach to steady state and a receive array; neither is on the
  roadmap with a date.
