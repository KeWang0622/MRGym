# Conventions

Units, signs, axes and definitions of the virtual scanner, each with the place in the code that fixes it.
Read this page before comparing an MR Gym image, phase or field map with another tool.

Where a statement is checked by a command, the command is named: a row of `mrgym validate` (S, R, K, C numbers) or a
script in [examples/textbook/](../../examples/textbook/). Simplifications are in [limits.md](limits.md), papers in
[references.md](references.md).

## 1. Units

| Quantity | What the agent sends and reads | Inside the package | DICOM | BIDS | Pulseq file |
|---|---|---|---|---|---|
| Time (`tr_ms`, `te_ms`, `ti_ms`, `echo_spacing_ms`) | ms | ms in `console` and `signal`; s for sample times in `kspace` | ms | s | s |
| Scan time, budget | s | s | | | |
| Flip angle (`flip_deg`) | degrees, nominal | rad, actual = nominal x B1 | degrees | degrees | RF amplitude in Hz |
| Off-resonance, fat frequency | Hz | Hz | | Hz | Hz |
| Receiver bandwidth | Hz per pixel | Hz per pixel; sampling rate = 2 x matrix x bandwidth | Hz per pixel | | dwell time in s |
| Position, field of view, slice | mm from the isocentre | mm; k-space in cycles per mm | mm | mm | m |
| Gradient amplitude, slew rate | percent of the limit | T/m, T/m/s | | | Hz/m |
| Spoiler moment (`spoiler_mt_ms_per_m`) | mT·ms/m | T·s/m; rad/m after multiplying by 2 pi gamma | | mT·s/m | 1/m |
| B1 | percent of the limit on B1rms squared | µT | | | Hz |
| Relaxation times | ms in every task that asks for one | ms | | s in quantitative maps | |
| Diffusion coefficient | hidden | µm²/ms, equal to 1e-9 m²/s | | | |
| Phase | rad, in (-pi, pi] | rad | | | rad |

Two traps follow from the table. A relaxation time delivered in seconds is wrong by a factor of 1,000. A spoiler
moment copied from a BIDS sidecar is 1,000 times smaller than the MR Gym number.

Parameter names carry their unit as a suffix. The console rejects unknown names (`console.validate`).

## 2. Constants

| Constant | Value | Where |
|---|---|---|
| Proton gyromagnetic ratio, gamma / 2 pi | 42.577478518 MHz/T | `console.GAMMA_HZ_PER_T`, `signal.GAMMA_MHZ_PER_T`, `world.GAMMA_HZ_PER_T` |
| Main fat peak relative to water | -3.4 ppm | `tissue.FAT_PPM` |
| Fat spectrum | six peaks, amplitudes sum to 0.999 | `tissue.FAT_PEAKS` |
| Susceptibility difference air - tissue | 9.4 ppm | `world.DCHI_AIR` |
| Field strengths | 1.5 T and 3 T | `world.B1RMS_LIMIT_UT` |

Other tools use slightly different values of gamma. Compare the value before exchanging gradient moments or
frequencies in ppm.

## 3. Axes and image orientation

- The slice is axial and passes through the isocentre. B0 points along z, through the slice.
- Arrays are `[image, y, x]`: row = y, column = x.
- +x is the patient's left. +y is posterior. On a displayed image +x points right and +y points down, the
  radiological view of an axial slice (`anatomy.py`, module docstring).
- Pixel `i` of an `n`-point axis is centred at `centre + (i - n // 2) x FOV / n` (`session.pixel_positions`). With
  the default centre the isocentre is pixel `n // 2`, and the image covers `-FOV / 2` to `FOV / 2` minus one pixel.
  Checked by K19.
- `centre_y_mm`, `centre_x_mm` move the centre of the field of view. The band filter of the readout and the fold
  of the phase direction follow it: a spin at `y` appears at `((y - centre + FOV / 2) mod FOV) - FOV / 2` from the
  centre. Checked by K21, K22.
- `rotation_deg` = r turns the field of view about its centre: the image rows run from +y towards +x. The pixel
  at `(v, u)` mm from the centre along rows and columns shows the patient position
  `centre + (v cos r - u sin r, v sin r + u cos r)` (`session.pixel_grid`); the object appears turned by -r. The
  scanner images the world re-evaluated on the turned grid (`world.turned`), so a turned image has the edges and
  the sampling of an upright one. `pe_dir` names the image axis (rows or columns), not the patient axis. Checked
  by K23, K24.
- Regions are circles in patient coordinates, whatever the field of view. A region read-out uses the pixels whose
  centre lies inside the circle (`session.roi_readouts`).
- There is no patient coordinate system and no exporter. If you write DICOM yourself, the axes above are
  x = L, y = P of the LPS frame.

## 4. Readout and phase encoding

- `pe_dir` = `y` (default): phase encoding along y, readout along x. `pe_dir` = `x` swaps them. The header prints
  `A>>P` for y and `R>>L` for x.
- The pixel is square. The phase direction has `matrix x phase_fov` pixels, rounded to an even number.
- The readout is sampled at twice the matrix and band-filtered, so the readout direction does not fold. The phase
  direction folds. Checked by K6, K7 and example 06.
- For every sequence except `epi` the order of the phase-encode lines has no effect on the data of a static
  object, because every line is served at the steady state. `epi` with `pe_polarity` = +1 acquires the lines in
  order of increasing k.
- When something in the world moves periodically, time enters. Line by line: lines in order of increasing k, the
  averages of a line in succession, one cycle (`tr_ms`; `afi`: TR1 + TR2) apart, so line `j` of average `a` is
  acquired at `(j x averages + a) x cycle`. `epi`: line `j` of a shot at `j x echo_spacing_ms`, one shot per
  average, `tr_ms` apart. The phase of the motion at the first line is a fixed hash of the mover and the realised
  protocol: the same protocol gives the same ghost, another protocol another phase (`kspace.Engine._motion`).
  Copies of a pulsing source sit at `+- lines x frac(f x cycle x averages)` pixels along phase. Checked by K25 to K27.

### Which way things move

The signal equation is `S(k) = sum over spins of m(r) exp(-2 pi i k·r)`, with k in cycles per mm (`kspace.py`,
module docstring). `ro_polarity` = +1 samples k along the readout in increasing order in time. Off-resonance adds
the phase `exp(+2 pi i Δf t)`. Together:

| What | Displacement | Axis |
|---|---|---|
| Spin off resonance by Δf, line-by-line sequences | `-ro_polarity x Δf / bandwidth_hz_per_px` pixels | readout |
| Fat, main peak (Δf < 0) | towards +readout for `ro_polarity` = +1, by 3.4 ppm x gamma x B0 / bandwidth pixels | readout |
| Spin off resonance by Δf, `epi` | `-pe_polarity x Δf x echo spacing x lines` pixels | phase |
| Fat in `epi` without fat saturation | towards +phase for `pe_polarity` = +1 | phase |
| Narrow-band interference at frequency f | a line at `-ro_polarity x f / bandwidth` pixels from the centre | readout |

Checked by K3, K4, K5, K12, K14 and examples 05 and 10.

Three notes for a reader who compares with a scanner.

1. The datasheet calls `ro_polarity` the sign of the readout gradient. With one sign convention for all phases
   (faster spins gain positive phase), sampling k in increasing order under the kernel `exp(-2 pi i k x)` is what a
   negative gradient does. Polarity +1 here therefore displaces like a negative readout gradient on a scanner that
   is described with a single sign. Compare directions of displacement, not gradient signs.
2. The header figure `WFS` is the shift of the main fat peak. The image holds six fat peaks. Their centre of mass
   moves by the amplitude-weighted mean of -3.02 ppm, about 11 % less than the header says. Example 05 prints both.
3. In `epi` nothing evolves within a readout line, so off-resonance does not displace along the readout.

## 5. Phase and sign of the signal

- Off-resonance Δf is positive for a spin that precesses faster than the receiver reference. Such a spin gains
  positive phase: a gradient echo carries `exp(+2 pi i Δf TE)` (`kspace.Engine.kspace`). Checked by K20 and
  example 04.
- **KomaMRI and MRzero from version 1.0 use the opposite sign, `exp(-2 pi i Δf t)`.** Conjugate a phase image or
  negate a field map before comparing with either. This sentence rests on reading those packages; no command in
  this repository checks it.
- Δf is the sum of the local field offset and the error of the frequency adjustment. Fat adds its chemical shift.
- The signal is demodulated with the phase of the excitation pulse. On resonance at TE = 0 it is real and
  positive, before the receive profile multiplies it (`epg.py`, module docstring).
- Spin echoes (`se`, `ir`) carry no off-resonance phase at the echo centre. Their phase is the receive phase.
- `ir` is signed. Negative magnetisation shows as a phase jump of pi relative to a long TI. Example 02 uses it.
- `bssfp` carries `exp(+i θ_free / 2)` at TE = TR / 2, with θ_free = 2 pi Δf TR, on top of the phase of the steady
  state (`signal.bssfp`).
- `phase_cycle_deg` of `bssfp` is the advance of the RF phase from one pulse to the next, counted in the same sense
  as the phase of a faster spin: pulse `j` turns the magnetisation about the axis at the angle `j x phase_cycle_deg`
  from x. The steady state depends on how far the spin runs ahead of the pulse per TR,
  `2 pi Δf TR - phase_cycle_deg`, and is lowest where that is a whole number of turns: the dark bands sit at
  Δf = (m + `phase_cycle_deg` / 360) / TR, so a phase cycle of 90 degrees puts one at +1 / (4 TR). Engine 0.2.0 had
  the opposite sign (bands at (m - `phase_cycle_deg` / 360) / TR); 0 and 180 degrees are the same in both. Checked
  by S18 and K11 against a pulse-by-pulse simulation.
- RF spoiling: pulse `j` has the phase `Φ j (j + 1) / 2`, Φ = `rf_phase_increment_deg`. `afi` counts raster
  positions of length TR1, also those without a pulse (`epg.py`).
- Every image is multiplied by the receive profile, a smooth complex function of position.

## 6. Fourier transform and scale of the reconstruction

Acquired k-space is an array `[lines_full, 2 x matrix]`. Sample `(p, q)` sits at
`k = ((p - lines_full // 2) / FOV_pe', (q - matrix) / (2 x FOV))`, where `FOV_pe'` includes the phase oversampling.
The reconstruction (`kspace.Engine.recon`) is

    image = fftshift(ifft2(ifftshift(S))) x (number of samples) / (2 x FOV x FOV_pe')

which is the sum of `S(k) exp(+2 pi i k·r)` times the k-space cell area. Then the central `matrix_pe x matrix`
pixels are kept. Lines not acquired with partial Fourier are zeros. There is no filter, no apodisation and no
homodyne step.

Scale. `S` is the sum of magnetisation times master-grid cell area times slice thickness. A pixel value is
therefore

    (transverse magnetisation per unit M0) x (proton density) x slice_mm x (receive profile)

and does not change with matrix, field of view or bandwidth. Doubling `slice_mm` doubles the signal. The datasheet
tells the agent only that the scale is arbitrary and constant within a session.

Noise is added to k-space, per sample, before the reconstruction (`kspace.Engine.acquire_kspace`). Real and
imaginary parts each have the standard deviation `sigma0 x sqrt(2 x matrix x bandwidth / averages)`. The image
noise then follows as `sigma0 x sqrt(bandwidth x lines / averages) / (dx x dy x lines_full)`. Checked by K15 to
K17 and example 08.

## 7. TE, TI and TR

All times are between centres of pulses and echoes (`console.py`, timing model). Requested times are rounded up to
the 0.01 ms raster; use the realised values that `plan` and `scan` return.

| Sequence | TR (`tr_ms`) | TE (`te_ms`) | Other |
|---|---|---|---|
| `spgr` | excitation to excitation | centre of the excitation to centre of the readout; echo `i` at TE + `i` x `echo_spacing_ms` | |
| `afi` | TR1, the short interval; TR2 = `tr_ratio` x TR1; one cycle = TR1 + TR2 | as `spgr`, in both intervals | image 0 is acquired in TR1, image 1 in TR2 |
| `se` | excitation to excitation | centre of the 90 degree pulse to the echo; the 180 degree pulse sits at TE / 2 | one phase-encode line per TR |
| `ir` | inversion to inversion | as `se`, counted from the excitation | TI = centre of the inversion to centre of the excitation |
| `bssfp` | excitation to excitation | TR / 2, not adjustable | `phase_cycle_deg` = RF phase advance per TR |
| `epi` | excitation to excitation, one image per TR | centre of the excitation to the line through k = 0 | `echo_spacing_ms` = time between successive lines |

In the signal models every pulse is instantaneous in time: nothing relaxes while it plays, and "centre" is the
instant of the pulse. Pulse durations enter the console's minimum times and the RF power, and with a selective
pulse the time-bandwidth product shapes the slice profile (section 8a).

## 8. B1

B1 is the ratio of actual to nominal flip angle at a position. It is the product of a smooth spatial pattern and
one transmit-gain error per session (`kspace.Engine.b1`, `world.b1_at`). It scales every excitation and
refocusing pulse: `se` and `ir` play `90 x B1` and `180 x B1` degrees. The adiabatic inversion and the fat
saturation do not scale with B1; each has one efficiency per session. The console computes RF power from the
nominal flip angle.

## 8a. Slice selection

- `excitation` = `selective` (default): the excitation is a Hann-windowed sinc of 128 equal time steps whose
  amplitudes sum to the flip angle at the slice centre (`rf.sinc_pulse`). `rf_pulse` names duration and
  time-bandwidth product: `fast` 1 ms / 2, `normal` 2 ms / 4, `sharp` 4 ms / 8 (`rf.PULSES`). The bandwidth
  `tbw / duration` spans one slice thickness: "thickness" is the width of the excited band, close to the full
  width at half maximum of the profile for `normal` and `sharp`, and narrower than it for `fast`.
- The rotation at height z is the product of the 128 small rotations about the pulse axis, each between two
  half-steps of precession about z (`2 pi x tbw x z / thickness / 128` per step), followed by a rephasing lobe of
  half the selection area (`rf.cayley_klein`; Pauly 1991). Cayley-Klein parameters: a rotation by φ about the unit
  axis n has `a = cos(φ/2) - i n_z sin(φ/2)`, `b = -i (n_x + i n_y) sin(φ/2)`; from equilibrium `Mxy = 2 conj(a) b`,
  flip angle `2 asin|b|`. Checked by R1, R2.
- The signal of a voxel is the integral over z of the sequence's signal at the local flip angle, times the
  transverse phase factor, per unit thickness: independent columns, no exchange between them. Quadrature:
  `6 x tbw` points per thickness out to `0.5 + 12 / tbw` thicknesses from the centre (6.5, 3.5 and 2 thicknesses for
  `fast`, `normal` and `sharp`; `rf.quadrature`), the two sides of the slice added (`kspace.slice_table`). Checked by R3 to R5, R7, R10.
  A rectangular profile would give the one-angle formula; `flip_deg` x B1 is the angle at the slice centre.
- `se`, `ir`: the 180 degree pulse is the same shape, 1.5 times as long, on a slice 1.5 times as thick
  (`rf.REFOCUS_WIDTH`), with crushers. A column contributes `-b² conj(Mxy)` of its transverse magnetisation and
  keeps `(1 - 2|b|²)` of its longitudinal magnetisation, b being the Cayley-Klein parameter of the refocusing
  pulse at that height. Transverse magnetisation that the crushers dephase is taken as lost for good: a
  longitudinal grating stored by an imperfect refocusing pulse is not recalled in the next repetition. The echo
  is reported positive for ideal pulses. Checked by R9; the size of the grating term is in
  [external.md](external.md), row L1.4. The inversion pulse of `ir` is not slice-selective.
- `bssfp`: the closed form at the flip angle of every column, summed; the transverse phase of the profile is kept
  as a weight, the rotation of each column is taken about the pulse axis.
- Peak B1 = `rf.peak_factor(tbw)` x the amplitude of a hard pulse of the same duration and angle; energy =
  `rf.energy_factor(tbw)` x that of the hard pulse. Slice gradient = `tbw / (duration x gamma x thickness)`.
  Checked by C4 to C6.
- `excitation` = `nonselective`: hard pulses, one flip angle per voxel, the formulas of the datasheet. The scan
  is priced as a 3D acquisition of `ceil(160 mm / slice_mm)` partitions (`console.SYSTEM["slab_mm"]`), the noise
  is that of the 3D reconstruction (1 / sqrt(partitions) of the 2D noise), and the image is one partition. This
  is the signal model of engine 0.1.0. Checked by R6, C7, C8.

## 9. T2*

`1 / T2* = 1 / T2 + 1 / T2'` per tissue class (`signal.t2star_ms`). T2' is the reversible part and gives a
mono-exponential (Lorentzian) decay.

- Gradient echoes (`spgr`, `afi`, `epi`): `exp(-(TE + τ) / T2*)`, with τ the time of a sample relative to the echo
  centre. The decay continues through the readout and blurs. Checked by K9, K10.
- Spin echoes (`se`, `ir`) and `bssfp`: `exp(-(TE + τ) / T2) x exp(-|τ| / T2')`. Static dephasing is refocused at
  the echo centre.
- A field that varies inside a pixel dephases further, because the phase is applied on the master grid.
- Gradient echoes also lose signal where the field has the through-slice gradient `g` in Hz/mm. With
  `c = g x slice_mm x TE` cycles of phase across the thickness, a selective pulse leaves
  `sinc(c) x (1 + cos(2 pi c / tbw)) / 2` and nothing beyond `c = tbw / 2`: for a small flip angle the profile is
  the Fourier transform of the pulse, so the factor is the pulse shape itself (`rf.dephasing_factor`). It can be
  negative: the phase of the signal then jumps by pi. The hard pulse of the 3D mode keeps the factor of engine
  0.1.0, `exp(-(pi c)² / (4 ln 2))`, a Gaussian profile whose full width at half maximum is `slice_mm`
  (`kspace.Engine.kspace`). The QA phantom has `g` = 0. Checked by R11.
- Fat saturation: the pulse is centred on the main fat line (-3.4 ppm) plus the system's hidden offset
  (`world["system"]["fatsat_offset_hz"]`) plus the protocol's `fatsat_offset_hz`. A spin is saturated by the
  session's efficiency when its own frequency lies within half the pulse bandwidth (1.3 ppm in all) of that
  centre (`kspace.Engine.amplitudes`). Checked by K28.

## 10. Parameter names against DICOM and BIDS

The keywords are quoted from the DICOM data dictionary and from the BIDS specification. No command in this
repository checks them against the standards, and the package ships no exporter.

| MR Gym | DICOM keyword (tag), unit | BIDS field, unit | Note |
|---|---|---|---|
| `tr_ms` | RepetitionTime (0018,0080), ms | RepetitionTimeExcitation, s; RepetitionTimePreparation, s for `ir` | `afi`: two RepetitionTimeExcitation values |
| `te_ms` | EchoTime (0018,0081), ms | EchoTime, s | multi-echo `spgr`: one EchoTime per image |
| `ti_ms` | InversionTime (0018,0082), ms | InversionTime, s | same definition: centre to centre |
| `flip_deg` | FlipAngle (0018,1314), degrees | FlipAngle, degrees | nominal |
| `bandwidth_hz_per_px` | PixelBandwidth (0018,0095), Hz per pixel | | |
| `averages` | NumberOfAverages (0018,0083) | | complex averages |
| `matrix`, `phase_fov` | AcquisitionMatrix (0018,1310); PercentPhaseFieldOfView (0018,0094) = 100 x `phase_fov` | | |
| `fov_mm` / `matrix` | PixelSpacing (0028,0030), mm | | square pixels |
| `slice_mm` | SliceThickness (0018,0050), mm | SliceThickness, mm | |
| `pe_dir` | InPlanePhaseEncodingDirection (0018,1312): COL for `y`, ROW for `x` | PhaseEncodingDirection | the BIDS sign depends on the orientation you export |
| `partial_fourier` | PercentSampling (0018,0093) = 100 x `partial_fourier` | PartialFourier | the missing lines are synthesised by homodyne detection |
| `echo_spacing_ms`, `epi` | | EffectiveEchoSpacing, s | equal to the echo spacing: no parallel imaging |
| `spoiler_mt_ms_per_m` | | SpoilingGradientMoment, mT·s/m | divide by 1,000 |
| `rf_phase_increment_deg` | | SpoilingRFPhaseIncrement, degrees | |
| field strength | MagneticFieldStrength (0018,0087), T | MagneticFieldStrength, T | |
| header `WFS` | | WaterFatShift, pixels | main fat peak |
| header `SAR` | none: SAR (0018,1316) is W/kg | | percent of a limit on B1rms squared; see limits.md |
| `excitation` | MRAcquisitionType (0018,0023): 2D for `selective`, 3D for `nonselective` | MRAcquisitionType | |
| `centre_y_mm`, `centre_x_mm`, `rotation_deg` | ImagePositionPatient (0020,0032), ImageOrientationPatient (0020,0037) | | not exported; x = L, y = P |
| `phase_oversampling`, `ro_polarity`, `pe_polarity`, `phase_cycle_deg`, `tr_ratio`, `rf_pulse`, `fatsat_offset_hz` | no standard attribute | no field | |

The sequences map onto BIDS quantitative-MRI collections: `spgr` at several flip angles is VFA, multi-echo `spgr`
is MEGRE, `afi` is TB1AFI, an `ir` series is IRT1.
