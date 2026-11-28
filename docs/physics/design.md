# The design layer: simulating a designed pulse, gradient program or sequence

`mrgym.design` is a library. It simulates an RF pulse, a gradient program or a block list that somebody designed,
and checks it against hardware limits. It serves no task and no tool yet, it holds no state, and it does not change
the scanner engine: `mrgym.scanner` is imported only for its limits (`gradients.Hardware.console`) and, in
`mrgym.design.validate`, to compare the two.

Read [conventions.md](conventions.md) first; this page states only what the design layer adds. Papers named here are
in the module docstrings. The block list of `design.sequence` has a fuller relative in `mrgym.core`: three evaluators
(isochromat ensemble, phase graph, periodic solve), two pools and a library of sequences ([core.md](core.md)).

## 1. What each module computes

| Module | Function | Computes | Units |
|---|---|---|---|
| `rotation` | `rotate` | Cayley-Klein parameters (a, b) of any complex B1(t) under any gradient G(t) on one to three axes, for arrays of positions, off-resonances and B1 scales. No relaxation. Two models of a raster sample: `split` (a hard sub-pulse between two halves of the precession, the engine's model) and `piecewise` (one rotation about the effective field) | B1 in µT, G in mT/m, time in ms, position in mm, frequency in Hz |
| | `bloch` | magnetisation after the pulse from any start, with T1 and T2 during the pulse | per unit M0 |
| | `apply`, `compose`, `inverse`, `matrix`, `flip_angle`, `z_rotation` | algebra of (a, b) | |
| `pulses` | `sinc` | windowed sinc from flip angle, duration, time-bandwidth product, window | |
| | `slr`, `slr_filter`, `forward_slr`, `inverse_slr`, `min_phase_a` | Shinnar-Le Roux design with an equiripple or least-squares linear-phase filter: excitation, refocusing, inversion, saturation, for profile ripples given in units of M0 | |
| | `sech`, `sech_mz` | hyperbolic-secant adiabatic inversion from peak B1, β, µ; the exact Mz of the untruncated pulse | β in rad/ms |
| | `control_points` | a pulse from at most 64 complex control points; rule: point k of K at time k T / (K − 1), real and imaginary part interpolated linearly or by a natural cubic spline, evaluated at the centres of the raster samples | µT |
| | `verse` | variable-rate rescaling of a pulse and its gradient under a peak-B1 limit, a gradient limit and, optionally, a slew limit | |
| | `Pulse.record` | duration, peak B1, energy ∫\|B1\|² dt, B1rms over a stated TR, flip angle on resonance, bandwidth, slice gradient, rephasing area | µT² ms for the energy |
| `profile` | `fwhm`, `passband_ripple`, `stopband_ripple`, `transition_width`, `out_of_slice_fraction`, `phase_figures`, `flip_distribution`, `figures` | figures of merit of a sampled slice profile | mm, units of M0, rad, degrees |
| | `standard_mask`, `check_mask` | pass or fail against a list of bands, with the first violated band named | |
| | `jensen_integral`, `pulse_energy_floor_ut2_ms`, `duration_bound_ms`, `variable_rate_floor_ms` | lower bounds on RF energy and duration (section 4) | |
| `sequence` | `rf`, `rf_rotation`, `rf_pulse`, `delay`, `gradient`, `spoil`, `readout`, `run` | echo signals of a typed block list by a time-stepping extended phase graph, for one or many columns (positions through a slice, tissues), with diffusion per block | dephasing in whole cycles per voxel, time in ms |
| `gradients` | `program`, `rasterise` | a program of at most 32 vertices (ms, mT/m) per axis, joined by straight lines, on the raster | |
| | `moments`, `k_trajectory`, `b_value`, `b_tensor`, `heating` | moments 0 to 4 about a stated time, k(t), b-value and b-tensor with refocusing pulses changing the sign, ∫G² dt; exact for the rasterised waveform | mT/m ms^(n+1), cycles/mm, s/mm², (mT/m)² ms |
| | `check`, `check_pulse_gradient`, `pns_percent` | amplitude and slew per raster sample, per axis and as a vector, a stimulation figure, raster and end conditions, with the first offending sample of every broken rule | mT/m, T/m/s, percent |
| | `trapezoid`, `optimal_lobe` | the shortest triangle or trapezoid of an area (closed form); the shortest waveform on the raster with given moments (linear programme with a certificate) | |
| `store` | `DesignStore.put`, `get`, `summary` | immutable records under the first 12 hexadecimal digits of the SHA-256 of the design's canonical serialisation; a summary of at most 160 words and 880 characters (about 220 tokens at four characters per token; a proxy, no tokenizer is run) | |

## 2. Conventions

- **Rotation.** Right-handed, by the angle φ about the unit axis n: `a = cos(φ/2) − i n_z sin(φ/2)`,
  `b = −i (n_x + i n_y) sin(φ/2)`. A real positive B1 turns +z towards −y, so `Mxy = −i sin(flip)`; a B1 of phase p
  turns about (cos p, sin p, 0). This is the convention of `scanner/rf.py`.
- **Precession and the sign of a gradient.** A spin with off-resonance Δf at position r under the gradient G gains
  the phase `+2π (γ G·r + Δf) t`: a positive frequency gains positive phase, as in conventions.md, section 5. The
  sign of a gradient is the sign of the frequency it gives a spin at positive r, and `k(t) = γ ∫ G dt` enters the
  phase as `+2π k·r`.
- **Time zero.** A sample covers its own interval; the rotation runs from the start of the first sample to the end
  of the last. Nothing is referred to the pulse centre. A rephasing lobe is a rotation about z by its area. Gradient
  programs start at t = 0 and moments are taken about a time the caller states.
- **Sequences.** A pulse of flip angle α and phase p gives `F_0 = −i exp(i p) sin α` from equilibrium. `readout`
  multiplies by +i by default, so that the signal of a pulse about x is real and positive, as the engine reports it.
- **How this was checked against the engine.** Rows D4 and P1 compare (a, b) and the sub-pulse weights with
  `rf.cayley_klein` and `rf.sinc_pulse` for every console pulse; D5 and D5b compare peak B1, energy, slice gradient
  and the RF power figure with `console.py` (from engine 0.3.0 the console calibrates its slice gradient: the
  designed `sinc` that equals a console pulse of `slice_mm` has the thickness `slice_mm / rf.fwhm_factor(tbw)`); E7, E8 compare the phase graph with `epg.steady_state`; E6b and E9
  compare with `signal.bssfp` and `signal.ir_gre`; E11 compares a slice-selective pulse, column by column, with
  `kspace.slice_signal`; G1 compares the trapezoid with `console._lobe_ms`. The signs themselves are pinned by D1
  (closed form with phase and off-resonance) and by `tests/test_design_rotation.py`.

## 3. Formed physics and declared models

Formed (the equations follow from the Bloch equations and Fourier analysis, and the table below checks the code
against them): the RF rotation; relaxation during a pulse by symmetric operator splitting (half of the precession
and of the relaxation, the RF rotation, the other half; second order in the raster, rows D10); the Shinnar-Le Roux
transform and its inverse; the hyperbolic-secant solution; the extended phase graph with T1, T2 and free Gaussian
diffusion; gradient moments, k-space, b-tensor; amplitude and slew of a waveform; the closed-form trapezoid; the
linear programme for a lobe.

Declared (a choice or a fit, stated where it is used):

- **Stimulation figure** (`gradients.pns_percent`): the rectangular strength-duration law that IEC 60601-2-33 uses
  for whole-body gradients, as reproduced in the open literature; the standard itself was not read. The threshold
  constants, the distance from the isocentre at which dB/dt is taken, the limit and the root-sum-of-squares over
  axes are fields of `gradients.Hardware`. It is a price for fast switching. It is not a safety statement.
- **Heating figure** (`gradients.heating`): ∫G² dt, proportional to the resistive loss of an ideal coil. No duty-cycle
  limit is attached to it.
- **D∞** (`pulses.dinf`): an empirical fit for the transition width of an equiripple filter. `slr` uses it as a
  starting value and then adjusts the transition until the filter meets both ripples (row P4).
- **Hardware limits** (`gradients.Hardware`): numbers of a record. `Hardware.console()` reads the virtual
  scanner's; per-axis or vector judgement is the field `rule`.
- **Ideal spoiler** (`sequence.spoil`): sets all transverse states to zero. A real spoiler is a `gradient` block.

## 4. The bounds

`profile.duration_bound_ms` is the larger of two floors for a pulse that turns every spin of a frequency band by at
least a given angle under a peak-B1 limit. The hard-pulse floor: a product of rotations cannot turn a vector
further than the sum of their angles, so the duration is at least `flip / (2π γ B1max)`. The energy floor: in the
hard-pulse model on a uniform raster under a constant gradient, Jensen's formula for the Shinnar-Le Roux polynomial
A bounds the sum of `−ln cos(θ_j / 2)` from below by an integral over the profile magnitude, and the energy is at
most `B1max² × duration`. The derivation and every assumption are in the docstring. The bound uses the pass band
only; it is not tight (row F5 gives the closest any generator output comes), and it does not hold for a
variable-rate pulse, for which only the hard-pulse floor and `profile.variable_rate_floor_ms` apply. A design that
beats a floor means the floor or the simulator is wrong; rows F4 to F7 test this on every generator.

### What was tried for a tighter bound, and why the first design task compares with a reference

`Scanner/RFPulse-v0` asks for the shortest pulse that meets a flip-angle mask under limits. Its pass rule needs a
number to compare the duration with. Three candidates for a bound were derived for it; none is usable, and the task
compares with a stated reference search instead (`envs/rf_pulse/model.py::search`) and says so.

| Candidate | Statement | Status | Size |
|---|---|---|---|
| hard-pulse floor | flip angle ≤ 2π γ × transmit scale × B1 limit × duration | theorem: any gradient, any off-resonance, the amplifier law (it only lowers B1), the piecewise model (a sample turns +z by at most its nominal angle) | 3 to 18 times below the reference search on about 70 instances of the three levels: `floor_ms` of every score |
| slope of the profile | for a train of n hard sub-pulses under a constant gradient, \|dθ/dψ\| ≤ n − 1 (ψ the precession per sample): the Bernstein-Szegő inequality on 2\|B\|² − 1 = −cos θ, a trigonometric polynomial of degree n − 1 bounded by 1. A transition from θ₀ − tolerance to the stop-band angle over the width W × thickness then needs duration ≥ (θ₀ − tol − out) / (2π γ G_max W × thickness) | proven for the hard-pulse model on a uniform raster (also for a variable-rate pulse on resonance, whose samples have equal gradient area); not for the piecewise model, off resonance, or a delayed gradient | looser than the hard-pulse floor: 0.06 ms against 0.10 ms for 30° over 5 mm, W = 0.2 (the fastest growth ignores the ripples) |
| the mask as a linear programme | \|B\|² is linear in the autocorrelation of the sub-pulses, so "is there an n-sample pulse that meets the mask" is a linear programme | a certificate for ONE sample spacing. The design chooses its gradient, hence its spacing in position, and a finer spacing is at least as capable (the polynomial in x^m is a polynomial in x); the infimum over spacings is a problem over band-limited functions, for which no finite certificate was found | not built |

So the pass rule has a reference in it: "at most `factor` times the shortest design the reference search finds on
the hidden scanner". The search is over one generator (equiripple Shinnar-Le Roux, constant gradient); a design of
another kind may be shorter, passes, and is recorded with `merit_reference` > 1. The hard-pulse floor stays in the
rule as the line below which a delivered design is an engine defect.

### The transmit chain of a scanner (`hardware`)

`hardware.Chain` holds what a scanner does to a designed pulse and the ideal simulator does not: a transmit scale,
the amplifier law `a / (1 + (a / A)^6)^(1/6)` (Rapp's model of a solid-state amplifier with smoothness 3: a
declared choice, not a measurement of any amplifier), a delay of the slice-select gradient against the RF, an
off-resonance. `hardware.rotate` is the rotation of the played waveform in the piecewise model; the verifier, the
scan `rfscan` and the `simulate` tool all call it, with the hidden chain, the hidden chain at every voxel, and the
chain the agent assumes. `tests/test_rf_pulse.py` holds it against an integration of the Bloch equations by a
general-purpose solver (2e-7 on Mxy and Mz, with every element of the chain switched on and for a variable-rate
pulse) and the amplifier law and the delayed gradient against closed forms. The delay is given per RF sample as the
mean of the delayed gradient over that sample: exact in area, first order within the sample.

`pulses.slr_flip` states the two ripples of a Shinnar-Le Roux excitation on the flip angle (degrees) instead of on
|Mxy|, which does not move with the flip angle at 90°: |b| = sin(θ / 2) is one-to-one with θ, and the two
tolerances are handed to the filter design through the relations of a saturation pulse, whose profile is cos θ.

## 5. Validation

`python -m mrgym.design.validate` runs the rows below and exits with status 1 if one fails;
`tests/test_design_validate.py` asserts that all pass and that this table is the one the command prints
(`--markdown`). Errors below 1e-3 are printed with one significant digit: they are rounding noise and differ between
machines. The references were written by the same authors as the code: the rows show that the code computes the
stated equations. They do not show that the equations describe a scanner.

<!-- validation table: begin -->
### RF rotation

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| D1 | hard pulse: (a, b), Mxy, Mz for 300 random flip angles (0 to 360 deg), phases and off-resonances | a = cos(t/2) exp(-i psi/2), b = -i exp(i p) sin(t/2); Mxy = -i exp(i (p + psi/2)) sin t, Mz = cos t | 1.7e-15 | ≤ 1.0e-13 | pass |
| D2 | rectangular pulse off resonance (model piecewise), 200 random amplitudes, offsets and B1 scales | one rotation by |w_eff| T about the effective field (w1 cos p, w1 sin p, dw); Mz = 1 - 2 (w1/w_eff)^2 sin^2(w_eff T/2) | 3.5e-15 | ≤ 1.0e-12 | pass |
| D3 | the same pulse in the model split (hard sub-pulses): order of convergence with the raster, 50 to 800 samples | second order: the error falls by 4 when the raster is halved | 5.9e-05 in the order | ≤ 0.02 | pass |
| D3b | error of the model split for that pulse at 100 samples (10 us raster) | the effective-field closed form | 1.3e-04 | ≤ 0.001 | pass |
| D4 | agreement with the engine: (a, b) of `scanner/rf.py::cayley_klein` for the pulses fast / normal / sharp, Hann and Hamming, 5 to 180 deg, as excitation (with the rephasing lobe) and as the refocusing pulse on the 1.5 x slice | the engine | 4.7e-14 | ≤ 1.0e-12 | pass |
| D5 | peak B1, energy and slice gradient of the three console pulses | `rf.peak_factor`, `rf.energy_factor`, `console._b1_ut`, G = `rf.fwhm_factor` tbw / (duration gamma thickness) | 4.4e-16 relative | ≤ 1.0e-12 | pass |
| D5b | RF power figure and slice gradient of a console protocol (spgr, 30 deg, TR 20 ms, 5 mm) | `console.plan`: 100 B1rms^2 / (3 uT)^2 printed to 0.01 %, gradient printed to 0.001 mT/m | 0.8966 of the printed rounding | ≤ 1 | pass |
| D5c | gyromagnetic ratio of this layer | `console.GAMMA_HZ_PER_T` | 0 | ≤ 0 | pass |
| D6 | slice profile of a 1 degree windowed sinc (time-bandwidth 2, 4, 8; Hann, Hamming, Blackman) | small-tip approximation: Mxy = -i x the Fourier sum of the sub-pulse angles with the precession that follows each | 5.5e-05 of the flip angle | ≤ 1.0e-04 | pass |
| D7 | rotation matrix of a 180 degree sinc at 37 positions | separately written cascade of real 3 x 3 rotations (Rodrigues formula), sample by sample | 1.1e-14 | ≤ 1.0e-12 | pass |
| D7b | refocusing between equal crushers: echo and longitudinal magnetisation, averaged over 32 crusher phases of that cascade | echo = -b^2 conj(Mxy), Mz' = (1 - 2 |b|^2) Mz | 5.6e-15 | ≤ 1.0e-12 | pass |
| D8 | unitarity before the final normalisation: 512 random samples, 2,000 isochromats, both models, one and three gradient axes | |a|^2 + |b|^2 = 1 | 1.7e-13 | ≤ 1.0e-12 | pass |
| D8b | unitarity of the returned (normalised) parameters, 4,096 samples | |a|^2 + |b|^2 = 1 | 6.7e-16 | ≤ 1.0e-14 | pass |
| D9 | time reversal, 200 random samples of unequal duration under two gradient axes, 50 isochromats, both models | the reversed, conjugated pulse gives (a, -conj(b)); the reversed pulse with B1, gradient and offset negated undoes the rotation | 2.1e-15 | ≤ 1.0e-12 | pass |
| D10 | relaxation during a 4 ms complex pulse (model split): 4 isochromats, T2 4 to 60 ms, T1 300 to 2000 ms, 400 samples (10 us raster) | `scipy.integrate.solve_ivp` (DOP853, rtol 1e-12) on the Bloch equations with the continuous waveform | 3.7e-05 | ≤ 1.0e-04 | pass |
| D10o | the same (model split): order of convergence with the raster, 200 to 1,600 samples | second order (symmetric splitting, samples at interval centres) | 2.8e-04 in the order | ≤ 0.02 | pass |
| D10b | relaxation during a 4 ms complex pulse (model piecewise): 4 isochromats, T2 4 to 60 ms, T1 300 to 2000 ms, 400 samples (10 us raster) | `scipy.integrate.solve_ivp` (DOP853, rtol 1e-12) on the Bloch equations with the continuous waveform | 7.1e-05 | ≤ 1.0e-04 | pass |
| D10bo | the same (model piecewise): order of convergence with the raster, 200 to 1,600 samples | second order (symmetric splitting, samples at interval centres) | 2.3e-05 in the order | ≤ 0.02 | pass |
| D11 | `bloch` without relaxation, from a tilted start | `apply(rotate(...))` | 5.6e-15 | ≤ 1.0e-12 | pass |
| D12 | a time-varying gradient along an oblique direction on three axes | the same waveform on one axis at the projected positions | 4.7e-16 | ≤ 1.0e-12 | pass |
| D13 | time of one rotation: 512 samples x 2,000 positions (best of 3) | requirement | 0.009009 s | ≤ 0.1 | pass |

### Pulse generators

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| P1 | `sinc` with 128 samples: sub-pulse weights and flip angle on resonance | `scanner/rf.py::sinc_pulse` | 1.7e-15 | ≤ 1.0e-12 | pass |
| P2 | the polynomials A, B of `forward_slr` for 96 random complex sub-pulses | `rotation.rotate` on 721 precession angles per sample: a = exp(-i n psi/2) A, b = exp(-i (n-1) psi/2) B | 2.0e-14 | ≤ 1.0e-12 | pass |
| P2b | `inverse_slr` of those polynomials | the sub-pulses they were made from | 3.6e-16 | ≤ 1.0e-12 | pass |
| P3 | inverse transform of the designs: b of the forward-simulated `slr` pulse (excitation 90 / 30, refocusing 180 / 150, inversion, saturation; equiripple and least squares; 150 to 256 samples) | the designed polynomial B on 721 points of the unit circle | 7.2e-15 | ≤ 1.0e-09 | pass |
| P3b | the minimum-phase A of those designs: -sum ln cos(theta_j / 2) of the pulse | Jensen's formula: -(1/4 pi) x integral of ln(1 - |B|^2), equality for a minimum-phase A | 4.6e-13 relative | ≤ 1.0e-06 | pass |
| P4 | ripples of the simulated profiles of the equiripple designs, in the slice and outside, relative to the ripples asked for (1 % of M0 each) | the specification: |Mxy| (excitation), Mz (inversion, saturation), |b|^2 (refocusing) | 1.002 of the requested ripple | ≤ 1.02 | pass |
| P5 | filter ripples for profile ripples of 1 %: `polynomial_ripples` (solved numerically) | the relations of Pauly et al. 1991, which are first order in the ripples | 0.0354 relative | ≤ 0.04 | pass |
| P6 | Mz after a hyperbolic-secant pulse (mu 3, truncated at 6e-7 of the peak, 4,000 samples), 4 amplitudes x 61 offsets | exact solution for the untruncated pulse (Silver, Joseph, Hoult 1985) | 8.3e-06 | ≤ 5.0e-05 | pass |
| P6o | the same: order of convergence from 1,000 to 4,000 samples | second order | 0.1287 in the order | ≤ 0.3 | pass |
| P7 | adiabatic threshold: 10 ms hyperbolic secant (mu 5, truncated at 1 %), B1 from the threshold of the exact solution for Mz = -0.99 to 2.5 x that | Mz stays below -0.98 (exact solution of the untruncated pulse: below -0.99) | -0.9879 | ≤ -0.98 | pass |
| P7b | the same pulse below the threshold (0.2 to 1 x): Mz against B1 scale | exact solution of the untruncated pulse; the truncation at 1 % of the peak is the difference | 0.004731 | ≤ 0.006 | pass |
| P7c | a hard 180 degree pulse over the same factor 2 in B1 | Mz = cos(pi x scale): -1 and +1 | 0 | ≤ 1.0e-12 | pass |
| P8 | control points: two points (linear rule at the sample centres); one point (hard pulse): flip, energy, B1rms | the straight line; flip = 2 pi gamma B1 T, energy = B1^2 T, B1rms = B1 sqrt(T / TR) | 3.6e-15 | ≤ 1.0e-12 | pass |
| P8b | nine control points on a smooth curve: error of the cubic rule relative to the error of the linear rule | a spline through the points is closer to the curve than straight lines | 0.1027 | ≤ 0.5 | pass |
| P9 | variable-rate rescaling of a 90 degree sinc to a peak of 6 uT (gradient up to 20 mT/m, with and without a slew limit of 120 T/m/s): (a, b) on resonance at 481 positions | the pulse before rescaling | 1.3e-14 | ≤ 1.0e-12 | pass |
| P9b | the same pulses 200 Hz off resonance: largest change of Mxy | off resonance the rotation is NOT preserved (a change above 0.05 of M0 is expected) | 0.2467 | ≥ 0.05 | pass |
| P9c | peak B1, gradient and slew of the rescaled pulses relative to the limits given | at most 1 | 1 | ≤ 1 | pass |
| P9d | duration of the rescaled pulse when only the peak B1 binds | sum |theta_j| / (2 pi gamma B1max) (`profile.variable_rate_floor_ms`) | 2.2e-16 relative | ≤ 1.0e-12 | pass |

### Profile figures and bounds

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| F1 | width at half maximum, 10-90 % transition width, out-of-slice fraction, pass-band and stop-band ripple of synthetic profiles (Gaussian, trapezoid with added ripples) on 24,001 positions | their analytic values | 1.2e-05 relative | ≤ 0.001 | pass |
| F2 | phase figures of a uniform slice with a linear phase: slope, rms phase, residual after the line, refocused fraction | slope; slope d / sqrt(12); 0; sin(slope d / 2) / (slope d / 2) | 4.2e-04 relative | ≤ 0.001 | pass |
| F3 | profile mask: a profile inside the mask passes; a planted stop-band or pass-band excess fails with that band named first and the margin -0.03; a band without samples fails | by construction | 0 | ≤ 0 | pass |
| F4 | RF energy of 66 generator outputs (sinc, Shinnar-Le Roux, hyperbolic secant, control points) relative to the energy floor of their own |B|: smallest ratio, minus 1 | no pulse is below the floor (Jensen's formula); the ratio is 1 for a minimum-phase A and small sub-pulse angles | 9.6e-08 | ≥ 0 | pass |
| F5 | duration of the same outputs relative to `duration_bound_ms` at their own specification (smallest flip angle over 20, 50 and 80 % of the bandwidth, their own peak B1): smallest ratio | no design is shorter than the bound (closest: sinc rect tbw 2 10 deg) | 1.755 | ≥ 1 | pass |
| F6 | a second pulse with the same |b| at 801 positions but the maximum-phase A: difference of |b| | equal by construction (the floor depends on |B| only) | 1.2e-13 | ≤ 1.0e-09 | pass |
| F6b | its energy relative to the floor (the minimum-phase pulse: 1.000031) | above 1: the floor separates the two | 1284 | ≥ 1.05 | pass |
| F7 | a hard pulse at the peak limit against the bound for its flip angle | the hard-pulse floor is attained: ratio 1 | 6.7e-16 relative | ≤ 1.0e-09 | pass |

### Sequences (extended phase graph)

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| E1 | spin echo (90 about y, 180 about x; echo times 10 to 90 ms; with and without crushers; 37 Hz off resonance) | exp(-TE / T2), no off-resonance phase at the echo | 6.7e-17 | ≤ 1.0e-13 | pass |
| E2 | stimulated echo of three pulses with a mixing-time crusher | (1/2) sin a1 sin a2 sin a3 exp(-2 tau / T2) exp(-TM / T1) | 1.7e-18 | ≤ 1.0e-13 | pass |
| E3 | Carr-Purcell-Meiboom-Gill train with refocusing angles 180 to 60 deg: echoes 1 and 2 | sin^2(a/2) E2 and sin^4(a/2) E2^2 + (1/2) sin^2(a) E2 E1 (direct and stimulated pathway; E = exp(-spacing / T)) | 1.1e-16 | ≤ 1.0e-13 | pass |
| E4 | 14 echoes of trains with 120 and 75 deg refocusing: the CPMG phase, the non-CPMG phase, a 40 deg phase error | separately written ensemble of 64 isochromats with real 3 x 3 rotations (the recursion the phase graph replaces) | 3.6e-16 | ≤ 1.0e-12 | pass |
| E4b | trains of 400 refocusing pulses of 60, 120, 150 deg without relaxation, mean of the last 100 echoes: all pulses equal, and with a first pulse of 90 + a/2 deg | pseudo steady state: sin(a / 2) without and 2 sin(a / 2) / (1 + sin(a / 2)) with the first pulse | 9.4e-07 | ≤ 5.0e-04 | pass |
| E4c | those two closed forms | separately written isochromat picture: every isochromat's magnetisation at the first echo projected on the rotation axis of its echo period, averaged over 720 dephasing angles | 1.4e-15 | ≤ 1.0e-09 | pass |
| E5 | spoiled gradient echo with an ideal spoiler, iterated for 12 T1: 4 flip angles and repetition times | Ernst formula sin a (1 - E1) / (1 - E1 cos a) | 2.4e-11 | ≤ 1.0e-09 | pass |
| E6 | balanced SSFP with 180 deg phase cycling, iterated for 25 T1: on resonance, in the pass band, near a null | periodic state of the 3 x 3 Bloch matrices (Freeman & Hill 1971), solved as a linear system | 2.8e-13 | ≤ 1.0e-09 | pass |
| E6b | the same states | the engine's closed form `scanner/signal.py::bssfp` | 2.8e-13 | ≤ 1.0e-09 | pass |
| E7 | RF-spoiled gradient echo (117 deg increment, one cycle of dephasing per TR), 1,500 repetitions | the engine's steady state `scanner/epg.py::steady_state` (a banded linear solve, no time stepping) | 2.3e-12 | ≤ 1.0e-09 | pass |
| E7b | its distance from the Ernst value (15 deg, TR 8 ms, T1 1000 ms, T2 80 ms) | RF spoiling approaches ideal spoiling: within 5 % | 0.03498 relative | ≤ 0.05 | pass |
| E8 | the same with diffusion (bd = 0.002 per repetition) | the engine's steady state `scanner/epg.py::steady_state` (a banded linear solve, no time stepping) | 2.0e-12 | ≤ 1.0e-09 | pass |
| E9 | inversion recovery from equilibrium (4 inversion times, one at the null) and in the steady state with a 30 deg readout | 1 - 2 exp(-TI / T1); the engine's `signal.ir_gre` | 1.4e-17 | ≤ 1.0e-12 | pass |
| E10 | pulsed-gradient spin echo in the phase graph (4 cycles per 2 mm voxel, 12 ms lobes 40 ms apart) | exp(-b D) with b from `gradients.b_value` of the same lobes, and b = q^2 (Delta - delta / 3) (Stejskal & Tanner 1965) | 5.2e-11 | ≤ 1.0e-06 | pass |
| E10b | `bd_of_b` for that lobe | bd = D kappa^2 t | 0 relative | ≤ 1.0e-12 | pass |
| E11 | spoiled gradient echo of a slice-selective pulse (fast, normal, sharp): one column per position of the engine's quadrature, each with its own (a, b), summed | the engine's slice integral `scanner/kspace.py::slice_signal` (ideal spoiling) | 3.3e-14 relative | ≤ 1.0e-09 | pass |
| E12 | the whole state after three pulses with positive and negative dephasing, read back as the magnetisation of single isochromats | the isochromat ensemble | 1.2e-15 | ≤ 1.0e-12 | pass |
| E13 | time of a train of 400 refocusing pulses with 64 states (best of 3) | requirement | 0.02882 s | ≤ 0.1 | pass |

### Gradient programs and limits

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| G1 | time-optimal triangle / trapezoid of 5 areas at the console's derated limits: duration and area | `scanner/console.py::_lobe_ms`; 2 sqrt(A / S) or A / G + G / S | 2.2e-16 relative | ≤ 1.0e-12 | pass |
| G2 | the trapezoid on the 10 us raster, 40 random areas: area, and amplitude and slew recomputed from the rasterised waveform | the requested area; within 40 mT/m and 150 T/m/s | 2.0e-15 relative | ≤ 1.0e-12 | pass |
| G2b | its duration beyond the closed form | at most 3 raster steps, never shorter | 1.365 raster steps | ≤ 3 | pass |
| G3 | moments 0 to 2 of a trapezoid about a later time, its integral of G^2 and its final k | m0 = G (flat + ramp); m1 = m0 x distance; m2 = m0 x distance^2 + m0 ((flat + ramp)^2 + ramp^2) / 12; G^2 (flat + 2 ramp / 3); gamma m0 | 1.6e-15 relative | ≤ 1.0e-12 | pass |
| G4 | binomial lobe trains about an echo 3 ms later: 1:-2:1 (m0, m1) and 1:-3:3:-1 (m0, m1, m2) | zero (velocity and acceleration compensation); the next moment is not zero | 4.5e-15 of one lobe's area | ≤ 1.0e-12 | pass |
| G4b | the first moment that is left: m2 of 1:-2:1 and m3 of 1:-3:3:-1 | not zero | 47.04 mT/m ms^(n+1) | ≥ 1 | pass |
| G5 | b-value of a trapezoid pair (40 mT/m, 20 ms, 40 ms apart, 0.4 ms ramps): two equal lobes around a refocusing pulse, and two opposed lobes without one | gamma^2 G^2 [delta^2 (Delta - delta/3) + ramp^3/30 - delta ramp^2/6] = 1526.73 s/mm^2 (Stejskal & Tanner 1965 with ramps) | 3.5e-13 relative | ≤ 1.0e-12 | pass |
| G5b | rectangular lobes (ramps of 1e-9 ms): an adjacent bipolar pair, and the velocity-compensated double pair + - - + | (2/3) gamma^2 G^2 delta^3 and twice that; first moment of the double pair zero | 2.0e-10 relative | ≤ 1.0e-08 | pass |
| G6 | b-tensor of a random three-axis program (16 vertices per axis, 60 ms, two refocusing pulses, one of them between raster times) | numerical quadrature on 600,001 points (`scipy.integrate`: cumulative trapezoid, then Simpson) | 6.0e-14 of the trace | ≤ 1.0e-06 | pass |
| G6b | b-tensor of one waveform played along the direction (0.6, 0, 0.8) | b u u^T | 3.8e-14 relative | ≤ 1.0e-12 | pass |
| G7 | limit check of a two-axis program: per-axis rule names the slew of y at 0.5 ms (280 T/m/s); the vector rule also names the amplitude (41.04 mT/m); a program that does not return to zero is named | by construction | 0 | ≤ 0 | pass |
| G8 | stimulation figure (declared model) of trapezoid ramps and of a full reversal | 100 x slew x radius / (rheobase (1 + chronaxie / ramp time)); a reversal is one ramp of twice the time | 3.3e-15 relative | ≤ 1.0e-09 | pass |
| G9 | shortest lobe by linear programming: area only (5 areas, triangle and trapezoid regime) and zero area with a first moment (4 values, both regimes) | closed forms rounded up to the raster: 2 sqrt(A / S), A / G + G / S; two opposed triangles 4 sqrt(a / S) with 2 a^(3/2) / sqrt(S) = m1, or two opposed trapezoids at Gmax | 0 raster steps | ≤ 0 | pass |
| G9b | the same results: shorter than the closed form of the continuous problem | never (a waveform on the raster is a waveform) | 0 raster steps | ≤ 1.0e-09 | pass |
| G9c | the same results: moments recomputed from the returned samples; amplitude and slew rechecked | the targets; the limits | 1.9e-15 relative | ≤ 1.0e-09 | pass |
| G9d | flow-compensated lobe (area 5 mT/m ms, first moment zero at an echo 1 ms after it): residual moments; one raster step shorter is infeasible; it is longer than the lobe without compensation | the targets; the certificate | 6.1e-16 | ≤ 1.0e-09 | pass |

<!-- validation table: end -->

## 6. Limits

- **B1 inside a pulse** is one scale per isochromat. No spatial pattern of amplitude or phase beyond that scale, one
  transmit channel.
- **Gradients are ideal.** No delay, no eddy currents, no concomitant field, no amplifier bandwidth: the waveform
  that is checked is the waveform that is simulated. A variable-rate pulse is therefore better here than on a system
  whose gradient cannot follow it.
- **Profiles are one-dimensional**, along the direction of the gradient; a three-axis gradient is supported by
  `rotation.rotate`, the figures of merit are not.
- **One pool.** No magnetisation transfer, no exchange, no flow; diffusion is free and Gaussian; T2' is not in the
  phase graph.
- **Pulses in a sequence are instantaneous.** `sequence.rf_pulse` takes the rotation of a designed pulse; its
  duration is not added to the timing, and relaxation during it is available only through `rotation.bloch`.
- **Rephasing lobes** are rotations by their area: no time passes in them.
- **Equiripple Shinnar-Le Roux designs on a fine raster** can carry their largest B1 in the first and last samples;
  the record shows the peak, and the least-squares filter does not have them.
- **Variable-rate rescaling with a slew limit** always meets the limits but is the fastest schedule only while every
  rate stays above a bound stated in the docstring; it does not design the ramps around the pulse.
- **Design ids** are stable against rounding noise (12 digits of the largest entry of every array), not against a
  value that sits on a rounding boundary; ids were not compared between machines.
- **The limits check** does not rotate logical axes to physical ones, and the slice-select gradient of a pulse is
  checked without the ramps around the pulse.
- **The equiripple generator on the 10 µs raster** designs its filter with one tap per raster sample: a 5 ms pulse is
  a 500-tap filter, which takes 0.1 to 0.3 s, and its first and last sample keep a fixed flip angle, so the peak B1
  of a long pulse stops falling with its duration (about 6.5 µT per degree of that sample).
- **The transmit chain** (`hardware`) is a declared model: one scale, one memoryless amplitude law, one delay. It has
  no phase distortion, no memory, no eddy currents, and no relaxation during the pulse.
- **Not done**: minimum-phase and root-flipped designs, multiband pulses, a proven duration bound tighter than the
  hard-pulse floor, a time-optimal variable-rate design with gradient ramps, export to a sequence file.
