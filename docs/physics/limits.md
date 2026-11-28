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
  with published operating points; [measured.md](measured.md) compares it with measured data.

**Compared with measurement, and not.** [measured.md](measured.md) puts public phantom sessions on real scanners
next to the engine's own model of the same protocol: the ACR series measured with the QA functions of this package,
inversion-recovery T1 and spin-echo T2 of the NIST plates, the six echoes of a fat-water phantom, a gel sphere, and
published phantom measurements. What is compared there are statistics of images, signals of regions and fitted
values, row by row, with the differences that were found
([Known differences from measurement](measured.md#known-differences-from-measurement)). What has not been done: no
image of the six console sequences has been compared pixel by pixel with a scan of the same object, and no tissue
in a body has been compared at all.

**Out of scope.** MR Gym computes a stated model. Not
modelled, and not testable with `mrgym validate --perturb`: magnetisation transfer and exchange (the tissue table
holds two-pool numbers of seven tissues, `tissue.mt_table`, and the signal core computes a saturated bound pool; no
console sequence saturates one, block 8b); restricted or
anisotropic diffusion (free isotropic diffusion weighting exists on a console that lists `dwse`, block 3a); flow,
perfusion, physiological motion (one periodic line-modulation model exists, block 7);
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
mrgym validate --group rf                                  # the slice-selective pulse and the slice integral (rows R1 to R15)
mrgym validate --group encoding                            # image formation: partial Fourier (K29, K30), fat saturation (K28, K31, K32)
mrgym validate --group core                                # the block-list core behind the console: shadow of the six (J1 to J6), fse, mese, megre, dwse (J7 to J40)
python tools/external/run_all.py                           # the external ladder (rows L1.1 to L3.10)
python tools/measured/run_all.py                           # the comparison with measured data (rungs M0 to M9; minutes)
mrgym timing                                               # seconds of computation per scan
mrgym scan sequence=se tr_ms=3000 te_ms=90 matrix=256      # prints the header line with TA and SAR
python -c "from mrgym.scanner import world; print(world.DEFAULT_SPEC, world.B1_SHAPE, world.GRID)"
python -c "from mrgym.scanner import session; print(session.REPLAY_RTOL, session.FINGERPRINT_BLOCKS)"
```

`mrgym scan` also writes a PNG under `mrgym_out/`.

---

## 0. Errors of engine 0.2.0: six fixed in `mrgym-engine-0.3.0`, one declared

Seven behaviours of engine 0.2.0 were reported as wrong, not as simplifications: four by an audit against textbook
physics on 2026-11-26, two by the signal core ([core.md](core.md)), which re-derives every served sequence from the
Bloch equation, on 2026-11-27 (the sign of the `bssfp` phase cycle, which a review of the design layer found the same
day, and the columns of a `bssfp` slice), and one by the work on the fix itself (the line count of echo-planar partial
Fourier). Engine 0.3.0 corrects six. Each of the six has a test that states the right behaviour with a number and fails
on 0.2.0 (`tests/test_engine_0_3.py`), and all but the last have rows of `mrgym validate` against separately written
calculations. What is still simplified about each is in the blocks named in the last column.

| What engine 0.2.0 did | What engine 0.3.0 does | Checked by | Still simplified |
|---|---|---|---|
| Partial Fourier shortened the scan and raised the SNR of a region (the missing lines were zero-filled). | The missing lines are synthesised by homodyne detection: the resolution stays, the noise of a pixel rises, and the relative SNR of the header falls with it. | `tests/test_engine_0_3.py::test_p1_partial_fourier_saves_time_and_costs_snr_in_the_header`, `::test_p1_measured_snr_falls_with_partial_fourier_as_the_header_says`; rows K15, K17, K29, K30, K37 | block 5a |
| A perfect fat-saturation pulse removed all six fat lines. | The pulse saturates each line by its place in the band: the lines outside the band stay. | `tests/test_engine_0_3.py::test_p2_a_perfect_fat_saturation_pulse_leaves_the_lines_outside_its_band`; rows K28, K31 | block 9a |
| The fat-saturation factor was the same at every TR and flip angle. | The pulse is played before every excitation; a saturated line recovers until the excitation and has its own steady state. | `tests/test_engine_0_3.py::test_p3_the_fat_saturation_factor_depends_on_tr_and_flip_angle`; rows S16, S17, R14, K32; `tools/external/run_all.py` row L1.8 | block 9a |
| `rf_pulse=fast` excited a slice 1.2 to 1.4 times the prescribed `slice_mm`. | The slice gradient is calibrated per pulse: the width at half maximum of the small-flip-angle profile is `slice_mm` for all three. | `tests/test_engine_0_3.py::test_p4_every_pulse_excites_the_prescribed_thickness`; row R13; row L1.7 | block 1 |
| `phase_cycle_deg` of `bssfp` moved the dark bands down in frequency (to -50 Hz for 90 degrees at TR 5 ms). | The bands sit where the spin keeps step with the RF phase, at (m + `phase_cycle_deg` / 360) / TR: +50 Hz in that case. 0 and 180 degrees are unchanged. | `tests/test_engine_0_3.py::test_b1_the_dark_band_of_bssfp_moves_up_in_frequency_with_the_rf_phase_advance`; rows S18, K11; row L1.5 | block 11 |
| Echo-planar partial Fourier with `pe_polarity` -1 acquired one line more than the header and the scan time counted (the line without a conjugate partner on the lattice). | That line is not acquired with either polarity: the number of acquired lines is the `lines` of the realised protocol. | `tests/test_engine_0_3.py::test_epi_partial_fourier_acquires_the_lines_the_protocol_counts` | block 5a |

The seventh is not corrected. It is a simplification with a measured size, declared in block 11:

| What the engine does (0.2.0 and 0.3.0) | Size on engine 0.3.0 | Checked by | Declared in |
|---|---|---|---|
| `bssfp` on a selected slice turns each column through the slice about the pulse axis: the small off-resonance that a position gains from the pulse itself is left out. | Against isochromats turned by their whole balanced pulse: 0.24 % of the largest signal over the half of the period round the middle of the pass band, up to 5.2 % of it within 90 degrees of the null. The signal core, with the rotation of every column, finds 1.2 % of the slice integral in its three cases. | row R15; row Q11b of [core.md](core.md) | block 11 |

Results measured on engine 0.2.0 are not comparable with results on 0.3.0 (CHANGELOG.md, "Engine 0.3.0"). The
training result on the front page was measured on `Scanner/NoWrap-v1` and `Scanner/NullTissue-v0` on engine 0.2.0.
The pass rules of both ids are evaluated on scans without partial Fourier, without fat saturation and with the
default pulse, so none of the six enters their verdicts except through the calibration of the default pulse (every
slice-selective signal x 1 / 0.994). What does change for an agent on `Scanner/NullTissue-v0` is the noise of its
own probes when it uses partial Fourier, and the signed value of such a probe near a null (block 5a); the reference
policy of that id probes fully sampled since 0.3.0, and so does the probe its practice level proposes. The stored
episodes of that result do not record the protocols of the scans, so how many of them used partial Fourier, fat
saturation or another pulse was not counted.

What a comparison with measured data found is listed separately, each with its number and with what would close it:
[Known differences from measurement](measured.md#known-differences-from-measurement). One of them was a defect of
the QA measurements and is fixed: the background regions of the ghosting test were placed from the isocentre and
landed on a phantom that lay off centre. Two were wrong table values and are fixed with engine 0.3.0 (the amplitude
of the main fat line, T1 of sphere 5 of the NIST plate: `tests/test_cited_tables.py`).

---

## 1. Slice selection: independent columns, one pulse shape

- **Simplified.** A selective scan (`excitation: selective`, the default) excites the slice with a Hann-windowed
  sinc of 128 steps (`rf_pulse`: time-bandwidth product 2, 4 or 8). The slice gradient is calibrated per pulse so
  that the profile of a small flip angle is `slice_mm` wide at half its maximum (row R13); at large flip angles the
  profile is wider, and that is served. The rotation at every height through the slice
  is the exact product of the 128 steps; nothing relaxes while the pulse plays. Fat is excited in the same slice
  as water: the displacement of the fat slice by the chemical shift is not simulated (it would change nothing in an
  object that is constant through the slice). The signal of a voxel is the
  integral, over independent columns through the slice, of the sequence's signal at the local flip angle. The
  object and every field are taken as constant through the slice, except for one linear term: the through-slice
  gradient of B0, which multiplies gradient echoes by the small-tip dephasing factor of the pulse. `bssfp` treats
  the rotation of each column as a rotation about the pulse axis (block 11). The signal is tabulated over the flip
  angle at the slice centre and interpolated: on 5 to 16 Chebyshev nodes that span the transmit field of the object,
  and for the RF-spoiled steady state of `spgr` the difference to the ideal-spoiling form on 16 nodes.
  `excitation: nonselective` serves one flip angle
  per voxel (hard pulses) and is priced as a 3D scan; its slab profile and slab wrap are not modelled.
- **A real scanner** plays vendor pulses with other profiles, on an object that changes
  through the slice, and its neighbouring slices disturb each other.
- **Size here.** What the profile does: `python examples/textbook/13_slice_profile_vfa.py` prints the two-angle T1
  of a 2D slice with the nominal angles as -45 to -76 % (time-bandwidth 2), -37 to -70 % (4) and -19 to -53 % (8)
  of the truth; published for uncorrected 2D scans: 40 to 80 % (Svedin & Parker 2017), -63 % (Dieringer 2014).
  Numerics of the integral: rows R4 and R5 of `mrgym validate --group rf` (7.9e-05 relative for TR of 20 ms and
  more, 6.1e-04 at TR 5 ms, where the wings of the profile outweigh its centre). Interpolation over the flip
  angle: rows R8 and R10 (2.1e-04 relative for the RF-spoiled `spgr` signal against the phase graph solved at the
  flip angle of every position; 1.2e-07 for the nodes over the object). Relaxation during a 2 ms pulse:
  0.05 % of the signal in the case of row L1.3 of `python tools/external/run_all.py`. Other pulse shapes:
  `mrgym validate --perturb <id>` re-scores an environment with time-bandwidth 2 and 8, a Hamming window and a
  rectangular profile. The difference to a vendor pulse is not measured.
- **For a task.** Flip-angle methods on a 2D scan are biased, as on a scanner, and the agent has to correct for the
  profile or pay for 3D. The correction learned here is for this pulse family.
- **Planned.** Multi-slice acquisition is on the roadmap; other pulse designs are not.

## 2. Steady state served, transient not simulated

- **Simplified.** The approach to the steady state is not simulated. The console charges dummy time: 4 s for `spgr`,
  `afi` and `bssfp`, one cycle for `se` and `ir`, two shots for `epi`. Every phase-encode line is then served at the
  periodic steady state.
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
  The sequences served from the block-list core (blocks 3a and 8) are served at the periodic state of the
  REPETITION too, after one dummy repetition; inside a repetition the echoes of a train are computed one after the
  other, so the change of the signal along an echo train is in the data.
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
  `megre` (a console that lists it) is this steady state read at 2 to 16 echo times: row J25 of
  `mrgym validate --group core` holds its amplitudes within 1e-6 of those of `spgr`, row J26 its images. A bipolar
  readout alternates the direction of the displacement along the readout and nothing else: there is no phase error
  between odd and even echoes (the gradient delay of the system acts on `epi` only).
- **For a task.** The deviation from the Ernst formula is the model's answer. Its direction and rough size are
  credible; its exact value for a given tissue is not established.
- **Planned.** Not on the roadmap.

## 3a. Diffusion weighting (`dwse`): one free, isotropic coefficient per class

- **Simplified.** On a console that lists `dwse`, a pair of trapezoid lobes either side of the refocusing pulse
  multiplies the spin echo of every tissue class by exp(-b D), D being the one diffusion coefficient of the class
  (`tissue.py`, the column the spoiling model of block 3 reads). The lobes are sized from the console's gradient
  limits and lengthen the minimum echo time; the b-value the console states is the integral of the waveform it
  plays. Diffusion is free and isotropic, so `diffusion_dir` does not change the data. Left out: the tensor,
  kurtosis and the perfusion fraction; the b-value of the imaging gradients and their cross terms with the lobes;
  eddy currents and the heating of the gradient coil; a shorter echo time by lobes that are not placed next to the
  refocusing pulse (0.1 ms at the default geometry, 6.5 ms at 50 Hz per pixel); the loss of signal that motion gives a line-by-line
  diffusion-weighted spin echo (a moving class is modulated line by line, block 7; it is not dephased by the lobes). The lobes
  are the crushers of the refocusing pulse and are taken to dephase by whole cycles inside a voxel whatever their size; with
  b = 0 the console's own crusher lobes play (1 ms, 23.5 mT ms/m, 0.15 s/mm^2 for the echo). A spoiler lobe of the same
  area ends the repetition on an axis perpendicular to the lobes; the gratings along z that survive the repetition are
  carried into the next one, the transverse magnetisation is not (block 8a). The readout is one line per repetition, not the echo-planar readout that diffusion imaging uses
  on a scanner.
- **A real scanner** measures an apparent coefficient that depends on direction, diffusion time and b-value, and
  reads it with a single-shot echo-planar train.
- **Size here.** `mrgym validate --group core`: the stated b-value against the integral of the waveform and
  against the request (J31, 1e-6), against the trapezoid closed form (J32) and against the sum the core
  attenuates with (J33); the coefficient read back from two images of a disc (J35, 1e-9); b = 0 against `se`
  (J15). The minimum echo time at the default geometry is 8.38, 29.44, 47.32, 58.56, 72.74, 82.7 ms for b = 0,
  100, 500, 1000, 2000, 3000 s/mm^2 (J36). A coefficient from b = 0 and b = 1000 at a TR of 20 s is the material's
  to 1e-15 once the weighting of the crushers is counted; at a TR of 3 s it is off by 1.6 % (J35b): the b = 0 scan holds
  the stimulated echo of its last repetition and the b = 1000 scan does not. Of the 25 coefficients of the tissue table
  23 lie against a source that was opened (abstract or full text) and 2 are declared; 7 of the 23 lie OUTSIDE their
  source and are listed with the value it supports (`tissue.ADC_OUTSIDE_SOURCE`): liver 1.10 against 1.34 to 1.77,
  blood 2.50 against 0.94 to 1.66, fat 0.05 against 0.012 to 0.042, spinal cord 0.90 against 0.69 to 0.84, cortical bone
  0.10 against 0.063 to 0.093. They keep the numbers of engine 0.3.0, because a corrected number moves the data of
  `spgr` and `afi` on every world that holds the class: an engine version
  (`python -c "from mrgym.scanner import tissue; [print(r) for r in tissue.adc_report()]"`).
- **For a task.** The trade of b-value against echo time, T2 shine-through and the noise floor at high b behave as
  the formulas say. A coefficient measured here is the table's number, not a tissue's.
- **Planned.** The tensor comes with the 3D world; no date.

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
- **Planned.** On the roadmap ("Receive arrays"). The first part exists as a library that no tool and no task
  uses yet: a world can carry an array of receive elements, and its multi-channel data, coil combinations, SENSE
  and g-factor maps are formed and checked in [array.md](array.md)
  (`python -c "from mrgym.scanner import recon; recon.validate()"`). The scanner an agent talks to is unchanged
  until the console offers acceleration and coil combination.

## 5a. Partial Fourier is reconstructed by homodyne detection

- **Simplified.** With `partial_fourier` below 1 the console synthesises the lines it did not acquire: the lines
  that exist on one side of the centre only weigh 2, and the image is the component of that transform in phase
  with a low-resolution image made from the lines acquired on both sides (Hann-tapered; where that image has no
  signal its phase is continued from a coarser one). The step weighting of Noll 1991 is used, with no transition
  band. The phase of the returned image is the low-resolution phase. The one line without a conjugate partner on
  the lattice is not acquired.
- **A real scanner** offers several reconstructions (zero filling, homodyne detection with a ramp, iterative
  projection) and chooses per sequence.
- **Size here.** Rows K29 and K30 of `mrgym validate`: for a real object the image differs from the fully sampled
  one by the bound of K29, and the header's relative SNR is that of the fully sampled scan times
  sqrt(lines / sum of the squared weights). The noise of a pixel is that of row K15. What an object with a phase
  that varies within the low-resolution image loses is not measured.
- **For a task.** Partial Fourier trades scan time for SNR at the nominal resolution, as on a console. The
  background of such an image is not Rayleigh distributed, so a noise estimate from the background does not hold;
  with few symmetric lines (0.55 to 0.6) gradient echoes at long echo times lose signal where the field changes
  quickly. Where a region has almost no signal (a material near its null in an inversion-recovery image) the
  low-resolution phase there is that of the noise and of the surroundings, and the complex mean of the region is
  biased towards it; the size is not printed by a command of this repository. The reference policy of
  `Scanner/NullTissue-v0` therefore probes fully sampled (`envs/null_tissue/controls.py`).
- **Planned.** Other reconstructions are not on the roadmap.

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
  The sequences served from the block-list core have the same model (`served.form`, `tests/test_served_motion.py`):
  `mese`, `megre`, `dwse` and `satrec` read one line per repetition, as `se` does. An echo train (`fse`) reads
  several lines after one excitation: a line is modulated at the time of the echo that fills it, shot index x TR +
  echo time, so the lines of one shot are an echo spacing apart and neighbouring lines of one band a repetition
  apart. The echoes themselves are those of the object at rest: motion during a train does not change its
  amplitudes, and nothing is dephased by moving through a crusher or a diffusion lobe.
- **For a task.** Ghost position and amplitude respond to TR, averages and the phase-encode direction as the
  formulas say, including the fact that averaging is not a 1 / sqrt(averages) remedy. Anything that needs spin
  history (inflow, saturation of moving blood) is absent.
- **Planned.** Not on the roadmap.

## 8. Echo trains (`fse`, `mese`): only on a console that lists them, and idealised

- **Simplified.** A console offers the six sequences of before unless its card lists more (`console.offered`;
  world spec `console_sequences`). One environment lists one (`Scanner/T2Echoes-v0`: `mese`); in every other one `se` and `ir` still acquire one
  phase-encode line per TR and the words of real consoles for an echo train are refused as before. A console that
  lists `fse` or `mese` serves them from the block-list core ([core.md](core.md), section 11): 90 degree
  excitation, a train of refocusing pulses of ONE nominal angle, every pulse the rotation of the console's windowed
  sinc at each slice column, the refocusing pulse on `refocus_slice_ratio` times the slice (1.5 by default: the
  pulse pair of `se` and `ir`, and one manufacturer's default for fast spin echo; product sequences use 1.0 to 1.6,
  [references.md](references.md)). Crusher lobes of 1 ms at the derated limits (23.5 mT ms/m, one cycle per
  1.0 mm) play either side of every refocusing pulse and one more ends the repetition as its spoiler, on the same
  axis; spins diffuse across them with the coefficient of their class. The stimulated echoes of imperfect
  refocusing are in the echoes, and so are the gratings along z that survive the repetition (block 8a). Idealised:
  pulses are instantaneous and without relaxation; a voxel holds whole cycles of the crushers; the imaging gradients
  add no dephasing and no diffusion weighting; what is transverse at the end of a repetition is dephased for good
  (block 8a); the amplitude is that of spins on resonance (off-resonance moves the image, block 10, not the echo
  train); fat is one species with its T1 and T2 (no J coupling, so fat is not brighter than in `se`); there is no
  magnetisation transfer, no eddy-current phase error between odd and even echoes, no fat saturation; a moving class
  is modulated line by line and its echoes are those of the object at rest (block 7). The amplitude changes from
  line to line; the image is formed from a low-rank factorisation of that
  dependence whose truncation error is bounded by 0.05 standard deviations of the receiver noise of one k-space
  sample at 64 averages (in the image, where the errors of the samples add coherently: measured 0.016 of the noise
  of a pixel at 64 averages, row J14). The transmit scale of a voxel enters by interpolation over Chebyshev nodes. Limits of the
  simulation, refused with a reason: at most 32 echoes, matrix at most 256 (`mrgym timing --cpu`;
  [core.md](core.md), section 11).
- **A real scanner** uses echo trains for T2-weighted and FLAIR imaging and pays with blurring and RF power; its
  trains have variable refocusing angles, and fat is bright in them.
- **Size here.** `mrgym validate --group core`. Exact pulses against the closed form with the diffusion weighting of
  the crushers (J7, 1e-12); the slice against the rotation of every column (J8), against the isochromat ensemble
  stepped to its steady state at TR 1500, 330 and 200 ms (J9, 4e-10) and against the phase graph of the design layer
  with the gratings kept (J20, 5e-8); interpolation over the transmit scale (J10, 2e-6 of the largest echo) and the
  integral through the slice (J11, 1e-7; J23c); k-space against the Fourier sum line by line (J12) and the truncation
  against its bound (J13, J14); T2 blurring within 5 % of 2 ETL ESP / (pi T2) pixels and within 1 % of the
  transform of the modulation (J16, J17); the effective echo time (J18); scan time and RF power (J37, J38). A train
  of 16 pulses of 180 degrees every 140 ms is refused at 3 T for RF power (J40). Against a published measurement of
  the T2 error of a multi-echo spin echo: [measured.md](measured.md), rung M3b: with the protocol of the paper
  (`refocus_slice_ratio` 1.2) the measured error lies between the console's two pulse shapes.
- **For a task.** On a console without the sequences: scan time, not RF power, limits T2-weighted protocols, and
  the trade of echo-train length against blurring and power does not exist. On a console with them it exists, with
  the idealisations above; a T2 fitted to `mese` echoes carries the stimulated-echo bias of this slice profile,
  not of a product pulse, and the signal of a fluid at a TR of a few seconds is that of a static fluid whose
  coherences survive the repetition, which pulsation destroys in a body.
- **Planned.** Variable refocusing angles through the design layer (the block list takes one angle per echo
  already); no date.

## 8a. Spin echoes: crushed means gone

- **Simplified.** The stimulated echo of a spin echo is not simulated. In `se` and `ir` every transverse component
  that the crushers dephase is lost for good, and the longitudinal magnetisation of a column evolves as one number. A refocusing pulse that is not 180 degrees (a
  transmit error, or the edge of its slice profile) in fact stores part of the dephased magnetisation along z as a
  fine grating, and identical crushers recall part of it in the next repetition: a stimulated echo on top of the
  spin echo.
- **A real scanner** has this pathway whenever TR is not much longer than T1, and sequences reduce it with crusher
  schemes and phase cycling.
- **Size here.** Row L1.4 of `python tools/external/run_all.py`: with the grating kept, the independent integrator
  gives -2 to +39 % of the served signal in its five cases (TR 300 to 2000 ms, T1 500 to 1200 ms, B1 80 to
  115 %). The served model itself agrees with the integrator to the tolerance of that row.
  `fse`, `mese` and `dwse` do not have this simplification for the gratings: they play crusher lobes and serve the
  periodic state of the whole repetition with the gratings along z that reach the next excitation (`periodic`,
  method "carry"; `served.train_response`). Against the same train behind an ideal crusher the largest change of an
  echo is, in per cent of the largest echo (8 echoes of 10 ms, 180 degrees, row J24): at TR 500 ms white matter 3.3,
  grey matter 6.3, cerebrospinal fluid 18, fat 1.2, muscle 3.3, blood 11, liver 1.6; at 1500 ms 0.70, 2.3, 9.2,
  0.07, 1.1, 4.0, 0.29; at 4000 ms 0.03, 0.37, 2.9, 0.00, 0.16, 0.70, 0.01. The served formulation equals the
  isochromat ensemble stepped to its steady state to 1e-10 with 45 orders carried (J9) and the design layer's phase
  graph to 5e-8 (J20).
  The carried state is cut: a grating is kept when relaxation and diffusion across the rest of the repetition leave
  at least 1e-3 of it, up to 16 orders of the crusher. In water-like tissue diffusion sets the cut and it costs 9e-8
  (J23b). Where the limit of 16 orders sets it (no diffusion, fat) it costs 4e-4 of the largest echo for fat in a
  train of 16 echoes at TR 500 ms, 1e-5 at 1500 ms (J24b), 5e-4 for a material without diffusion at TR 1500 ms and T1
  900 ms (J23); over the protocols the console accepts (8 and 24 echoes, refocusing 60 to 180 degrees) at most 1e-3
  when the train is repeated without a pause, 6e-4 with 100 ms between the train and the next excitation, 2e-4 with
  400 ms, 4e-5 with a second (J24c).
  NOT carried: the transverse magnetisation that is left when the last echo has been read. It crosses the spoiler
  and the rest of the repetition, and in a voxel whose spins stayed in phase with each other it would return in the
  next train. How much depends on the phase it gathers on the way: on the off-resonance of the voxel within a
  fraction of 1 / TR, and on the spread of off-resonance inside the voxel (T2'). The amplitude of these sequences
  holds neither, so the magnetisation is taken as dephased, which is what static dephasing does to tissue and
  pulsation to a fluid in a body. Size for a static voxel exactly on resonance (row J24d; 8 echoes, hard pulses at
  90 % B1; in brackets how far the echo swings with the off-resonance): cerebrospinal fluid 89 (69) % of the largest
  echo at TR 500 ms, 24 (23) % at 1500 ms, 4.6 (4.6) % at 4000 ms; blood 8.0, 0.2, 0.0 %; grey matter 0.3 % and
  white matter 0.1 % at 500 ms. The core computes it (`periodic`, method carry with `carry_transverse`; core rows
  N14c); serving it needs the average over the off-resonance inside the voxel and a table over the off-resonance
  of the voxel, which are not built. The signal of a FLUID in an echo train at a TR of a few seconds or less is
  therefore that of a fluid without transverse coherence across repetitions.
- **For a task.** The signal of `se` and `ir` against TR and B1 is that of a sequence whose crushers work
  perfectly between repetitions. An absolute spin-echo signal at short TR should not be carried over to a scanner.
  `dwse` with b = 0 is NOT `se`: it differs by the stimulated echo (7 % of the largest pixel at TR 900 ms in the
  phantom of row J15b).
- **Planned.** `se` and `ir` on the block-list core is one engine version with a table of differences; no date.

## 8b. Prepared sequences (`satrec`, `fse` with `prep`): the preparation is an event on Mz

- **Simplified.** On a console that lists them. `satrec`: a saturation pulse, a saturation time, one gradient echo
  with ideal spoiling before and after. `fse` with `prep`: an inversion or a saturation `ti_ms` before the
  excitation. The preparation multiplies the longitudinal magnetisation of every spin by one number (inversion:
  minus the inversion efficiency of the system, as in `ir`; saturation: 1 minus the saturation efficiency, complete
  unless the world states another), the gratings along z included, and an ideal spoiler removes what is transverse
  at that moment. It acts on the whole object: it is not slice-selective, there is no profile of the preparation
  pulse and no dependence of its efficiency on T2, B1 or off-resonance (known difference 13 of
  [measured.md](measured.md) for the inversion). No magnetisation-prepared gradient-echo TRAIN is served (the core
  computes one, `library.mprage`; a train whose amplitude changes from line to line needs the line formation for a
  gradient echo, which is not built). No T2 preparation and no magnetisation-transfer pulse are served.
- **A real scanner** uses adiabatic or composite pulses whose efficiency depends on the tissue and the fields, and
  prepares slabs.
- **Size here.** `mrgym validate --group core`: `satrec` against sin a (1 - exp(-TS / T1)) and the fixed point of
  an incomplete saturation (J41, 1e-12), through the slice against the isochromat ensemble (J42), T1 from a series of
  images (J43, 1e-9); `fse` with a preparation against the closed form with incomplete recovery (J44, 1e-12), the
  null of a fluid (J45, 0.01 ms), through the slice against the ensemble (J46); time and RF power (J47).
- **For a task.** T1 by saturation recovery needs no knowledge of the flip angle or the slice profile here, as the
  formula says. The null of an inversion-prepared train moves with TR as the formula says; its sharpness is that
  of a perfect, uniform inversion.
- **Planned.** The prepared gradient-echo train and the magnetisation-transfer pulse; no date.

## 9. Idealised EPI

- **Simplified.** Single-shot gradient-echo EPI with flat-top sampling only. Decay and off-resonance advance from
  line to line, not within a line. Partial Fourier lines are synthesised (block 5a). The ghost comes from one
  constant delay between gradient and acquisition. The signal amplitude is the Ernst steady state at `tr_ms`, with
  the saturation of block 9a when fat saturation is on.
- **A real scanner** samples on ramps, corrects odd and even echoes with a measured phase, has eddy currents and
  usually halves the distortion with parallel imaging.
- **Size here.** `mrgym scan sequence=epi tr_ms=3000` prints ESP 0.91 and WFS 37.9 px (phase) at 3 T: fat would
  move by 37.9 pixels without saturation. Agreement of the distortion with its formula is row K12 and
  `python examples/textbook/10_epi_distortion.py`. The difference from a real EPI readout is not measured.
- **For a task.** Distortion is that of an unaccelerated readout. It can only be reduced by a shorter echo
  spacing, fewer lines or a blip reversal.
- **Planned.** Not on the roadmap.

## 9a. Fat saturation: a band with one efficiency, acting on the longitudinal magnetisation

- **Simplified.** The saturation pulse multiplies the longitudinal magnetisation of every resonance line by
  1 - efficiency x band, 7 ms before the centre of a 2 ms excitation (6.5 ms and 8 ms for the 1 ms and 4 ms
  pulses), in every repetition. The band has one width per session and edges 8 Hz soft, steeper than an 8 ms pulse
  can make; the efficiency is one number per session and does not follow the transmit field or `tx_scale`. Fat
  has six lines with one T1 and one T2; water is one line. Each line reaches its own steady state: exactly under
  ideal spoiling (`se`, `ir`, `epi`); in the RF-spoiled steady state of `spgr` the pulse multiplies the
  longitudinal states of every order and leaves the transverse states as they are, so what a real pulse turns
  from the transverse plane into z and back, and the dephasing by its own spoiler, are missing. The fat slice is
  not displaced against the water slice (block 1). There is no magnetisation transfer from the pulse.
- **A real scanner** plays a pulse whose band has edges of the order of 1 / duration, whose flip angle follows the
  transmit field, and whose lines relax with their own T1; its spoiler adds to the spoiling of the sequence.
- **Size here.** Rows S16, S17, R14, K28, K31 and K32 of `mrgym validate` check that the engine computes this
  model (against a matrix Bloch chain, an isochromat ensemble by time stepping, and the six-line sum). The
  difference to a real pulse is not measured.
- **For a task.** What fat saturation leaves depends on TR, T1 and the flip angle, and never falls below the
  share of the fat spectrum outside the band: a ratio measured at one repetition time is not the ratio at another.
  The size of the residual in the RF-spoiled sequence is the model's answer; its exact value should not be
  carried over to a scanner.
- **Planned.** Not on the roadmap.

## 10. Simplified fields

- **Simplified.** Head: the field of three air spheres below the slice. Abdomen: gas spheres in the slice plane.
  Both get a first-order shim over the brain or the abdominal cavity and a smooth random term. QA phantom: a linear
  plus saddle field. The scanner makes no adjustment of its own during a session: the hidden shim, frequency error
  and transmit gain stay as they are. A scan can add a frequency shift (`cf_shift_hz`), two in-plane linear field
  terms (`shim_y_hz_per_mm`, `shim_x_hz_per_mm`) and a transmit factor (`tx_scale`); there is no shim term through
  the slice and none above first order, the terms are exact (no shim coil imperfection, no eddy currents from
  switching them), and they cost no time. The transmit pattern is a closed form
  whose amplitude depends on the field strength. The receive profile is unrelated to the transmit pattern. Gradient
  non-linearity is one radial warp of positions (the world-spec key `gnl`), used by level 3 of
  `Scanner/GradientCal-v0` and off everywhere else. There is no concomitant field, no eddy current and no drift. A turned or shifted field of view sees
  the same fields: they are functions of the patient position.
- **A real scanner** has a field shaped by the whole anatomy, lets the operator re-shim, and warps images away
  from the isocentre.
- **Size here.** The amplitudes are constants of the package, printed by the `world` command above: transmit
  pattern 3 % (1.5 T) and 11 % (3 T) for the head, 7 % and 26 % for the abdomen. Each is varied by up to 20 % per
  session (`world.anatomy_fields`).
  Agreement with in-vivo field maps is not measured.
- **For a task.** Off-resonance can be mapped, and its constant and in-plane linear parts can be corrected at the
  source; a saddle term, the field of an air sphere and the gradient through the slice cannot. Geometry is exact
  once off-resonance is accounted for. Rows K33 to K36 and C9 of `mrgym validate` check the adjustments.
- **Planned.** Not on the roadmap.

## 11. bSSFP: steady state only

- **Simplified.** The closed-form steady state with the local off-resonance. No transient and no catalyst. Inside
  the steady state fat is a single line at -3.4 ppm; the six peaks act only during the readout. With a selective
  pulse every column through the slice is solved as a hard pulse of its own flip angle about the pulse axis. Away
  from the slice centre the whole balanced pulse of a column (prephasing lobe, pulse, rephasing lobe) is a rotation
  about a tilted axis: it adds a precession about z that differs from column to column, and that is left out.
- **A real scanner** needs a preparation, shows transient oscillations, and has finite pulses.
- **Size here.** Transient: not measured. Band positions against their formula, for phase cycles of 180, 90 and 300
  degrees: row K11 and `python examples/textbook/09_bssfp_bands.py`. The columns: row R15 of
  `mrgym validate --group rf` compares the served slice integral with isochromats that are turned by their whole
  balanced pulse, pulse by pulse (`fast`, `normal` and `sharp` at 60, 40 and 50 degrees): the two differ by 0.24 % of
  the largest signal over the half of the period around the middle of the pass band and by up to 5.2 % of it within
  90 degrees of the null. The served band has the right position and nearly the right shoulders; its floor is not
  the floor of a real slice.
- **For a task.** Bands sit where the formula puts them. Signal right after the start of a scan is not available.
- **Planned.** Not on the roadmap.

## 12. RF power is not SAR; no nerve stimulation limit

- **Simplified.** The console adds B1 squared times duration over the pulses of one cycle, divides by the cycle
  time and compares with a public limit on B1rms: 3.0 µT at 3 T, 6.0 µT at 1.5 T. A selective pulse counts with
  the energy of its shape (1.80, 3.11 or 7.16 times a hard pulse of the same duration and angle; row C4) and is
  lengthened if its peak would exceed 20 µT, a choice of this package; a nonselective pulse is a hard pulse.
  Everything is computed from the nominal flip angle times `tx_scale`. The header calls the result SAR: it is
  B1rms squared as a percentage of a limit, not a specific absorption rate. There is no body model, no
  averaging window and no local SAR. Gradients are checked against an amplitude limit and played at a fixed
  slew rate: no protocol is refused for its slew rate or its dB/dt, and there is no stimulation model.
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
- **Size here.** Not measured. [references.md](references.md) states how far each source was checked. The diffusion
  coefficients have their own table of sources (`tissue.ADC_SOURCES`; block 3a).
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
  (slice-selective excitation by default) serves other numbers than 0.1.0 for every protocol, and engine 0.3.0
  (calibrated slice gradient, homodyne partial Fourier, fat saturation per line with its steady state) other
  numbers than 0.2.0 for every selective scan. The hard-pulse signal of 0.1.0 is `excitation: nonselective`
  without fat saturation and with full Fourier sampling (row R6), at the price of a 3D scan.
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
