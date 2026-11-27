# Coupled spins: the density matrix and spectroscopy

`mrgym.spins` is a library. It simulates coupled proton spin systems with the density matrix and builds
spectroscopy on it: pulse-acquire, spin echo, PRESS, STEAM, semi-LASER, MEGA editing, 2D J-resolved and COSY, a
basis of metabolites, and the fit of a spectrum with that basis. It serves no task and no tool yet, and it does not
change the scanner engine. It uses `mrgym.design` for the rotation of a single spin and for pulse waveforms.

Read [conventions.md](conventions.md) and [design.md](design.md) first; this page states what the spin layer adds.

## 1. Why the Bloch equations are not enough

The Bloch equations follow one magnetisation vector per isochromat. Two protons joined by a scalar coupling J do not
have separate vectors: the state of the pair holds terms such as 2 Ix Sz (magnetisation of I that is antiphase with
respect to the state of S), zero- and double-quantum coherence, and longitudinal two-spin order. None of them is a
magnetisation, all of them turn into observable signal later, and a pulse on S changes what I does afterwards. The
echo time dependence of lactate, the loss of lactate in PRESS at 3 T, the mixing-time dependence of STEAM and the
whole of spectral editing are of this kind.

The state is the density matrix ρ, for n protons a 2^n × 2^n matrix. Thermal equilibrium, in the high-temperature
limit and without the unit matrix, is ρ = Fz = Σ Iz: one unit of z magnetisation per proton. It evolves by the
Liouville-von Neumann equation, dρ/dt = −i [H, ρ], that is ρ → U ρ U⁺ with U = exp(−i H t) for a constant H.
Every expectation value is a trace: the transverse magnetisation is Tr(ρ F+).

## 2. The Hamiltonian

In the frame of the reference frequency, in Hz:

    H = Σ_i ν_i I_iz + Σ_(i<j) J_ij I_i · I_j ,        ν_i = (δ_i − δ_ref) × 1e-6 × γ B0

The first term is the chemical shift, proportional to the field. The second is the scalar coupling, independent of
the field. `coupling="strong"` (the default) keeps the whole product I_i · I_j; `coupling="weak"` keeps I_iz I_jz
only, which is the limit |ν_i − ν_j| ≫ |J| and is there for tests against product-operator formulas. H commutes with
the total Fz; it is diagonalised block by block in the total magnetic quantum number M, and all propagation is done
in its eigenbasis, where a delay is a multiplication of every matrix element by a phase.

**Equivalent groups.** The three protons of a methyl group are simulated as one total spin: a quartet (j = 3/2) and
two doublets (j = 1/2), one small problem each (`operators`, module docstring; rows O2, O3, O7, M3). The coupling
inside the group is a constant in every such block and appears in no signal.

**Size.** A network of n single protons costs matrices of 2^n. The limit is 2^8; the largest entry of the basis
has six protons (64). Rows M4 to M4d give the times.

## 3. What a sequence element does to coherences

The matrix element between two levels has the coherence order p = M_r − M_s. z magnetisation and zero-quantum
coherence have p = 0, the observable single-quantum coherence p = ±1.

| Element | Block | What it does |
|---|---|---|
| Delay | `delay(ms)` | every element gains the phase of its two energies; the order does not change. Chemical shift and coupling act together |
| Hard pulse | `rf(flip, phase)` | rotates every proton about (cos φ, sin φ, 0); mixes all orders. With `band_ppm` it acts only on the sites in a band: an ideal frequency-selective pulse |
| Shaped pulse | `rf_pulse(b1, dt, g, ...)` | a B1(t) waveform of `mrgym.design.pulses`, every site at its own frequency; under its slice gradient on an ensemble of positions |
| Ideal crusher | `select(orders)`, `spoil()` | keeps the listed orders. `spoil()` keeps p = 0: z magnetisation and zero-quantum coherence, which a gradient does not dephase in a homonuclear system |
| Crusher pair around a pulse | `pathway="refocus"` or `"keep"` | equal crushers keep p → −p (refocusing); opposite crushers keep p → p (an editing pulse leaves the observed spin alone) |
| Gradient | `gradient(cycles, axis)`, `average(axis)` | explicit dephasing: an ensemble of positions over which the gradient winds whole turns; every later block acts on every member |
| Acquisition | `acquire(phase)` | the signal from that instant on, as a list of lines (frequency, complex amplitude, relaxation group) |

The projection on coherence orders and the explicit ensemble give the same signal (rows E8, E8b, S5e, S8b).

**Shaped pulses, two models.** `full` integrates the waveform under the whole Hamiltonian, one hard sub-pulse per
raster sample between two halves of the free evolution, as `mrgym.design.rotation.rotate(model="split")` does for
one spin; second order in the raster (rows E3, E3o). `product` turns every site by its own Bloch rotation from
`mrgym.design.rotation.rotate` and lets the coupling act as if the pulse were instantaneous at its centre. It is
exact without coupling (row E3b) and costs no matrix product per sample; with coupling its error is the coupling
evolution that the pulse interrupts (rows E3c, S2b, S6b). The sequence builders use `product` unless told
otherwise.

**Slice-selective refocusing and chemical-shift displacement.** A pulse designed with a thickness is played under
its gradient on positions along its own axis. A site whose frequency differs from the carrier by Δν is refocused in
a slice moved by −thickness × Δν / bandwidth (row S4). In PRESS this makes four compartments for a coupled pair: both
partners refocused by both pulses, by one, by the other, by neither. The coupling evolves differently in each, and
the sum is what the voxel gives (rows S2, S3). The ensemble is summed with the weight spacing / thickness, so a
signal is relative to an ideal voxel of the nominal size, and the sample extends `extent` thicknesses (default 2.5)
along each axis. The result depends on that number: an equiripple pulse refocuses its stop-band ripple of everything
in that range, so a larger sample gives more signal (`tests/test_spins_sequences.py` pins two values).

**Relaxation** is not part of the propagation. A line carries the relaxation group of its site; `Lines.fid` applies
exp(−t / T2) of that group during the acquisition, and `sequences.relaxation_factors` the factors
exp(−(time transverse) / T2), exp(−(mixing time) / T1) and the recovery between repetitions (`sequences.recovery`:
Mz inverted at each refocusing pulse and set to zero by the third pulse of STEAM; rows E9 to E9c against the Bloch
equations). This is exact for uncoupled spins with ideal pulses and for a network whose coherences all decay at one
rate. Field inhomogeneity is a line shape: a Lorentzian and a Gaussian width in `Lines.fid`.

## 4. Conventions

They are those of `mrgym.design` and of the engine; rows E2, E3b, M1, M1b compare numbers, not statements.

- **Rotation.** Right-handed, U = exp(−i φ n·I). For one proton this is the spinor matrix
  `[[a, −conj(b)], [b, conj(a)]]` with the Cayley-Klein parameters of `mrgym.design.rotation`. A pulse about +x
  takes +z towards −y: Mxy = −i sin(flip).
- **Precession.** A larger chemical shift is a higher frequency, and a positive frequency gains positive phase:
  Mxy → Mxy exp(+2π i ν t). The propagator of a delay is exp(−2π i H t).
- **Detection.** s(t) = i exp(−i receiver phase) Tr(ρ(t) F+) / 2^(n−2). F+ reads the elements of order −1, the I⁻
  part of ρ. The factor i makes the signal of a pulse about +x real and positive, as the engine reports it; one
  proton at equilibrium gives a line of amplitude 1.
- **Phases of the sequences.** Excitation about +x, refocusing about +y: an echo has the sign of the free induction
  decay.
- **Carrier.** A shaped pulse may be played at its own frequency (`carrier_ppm`); its phase is referred to the centre
  of the pulse. Times in a sequence are between pulse centres.
- **Reference.** Frequencies are relative to `ref_ppm`, by default 4.7 ppm, where the engine places water.
- **Spectrum** (`evolve.spectrum`). S(f) = dwell × Σ w_k s_k exp(−2π i f t_k), the first sample with weight 1/2. A
  line A exp(2π i f0 t − t/T2) appears at +f0 with a positive real part of height A T2, full width 1/(π T2) and
  area A/2 (rows E5, E5b). The axis ascends in Hz or ppm; plot it reversed. Phasing multiplies by
  exp(−i (φ0 + 2π f τ)): it removes a constant phase φ0 and a delay τ of the first sample.
- **Units.** Time in ms, frequency in Hz, B1 in µT, gradient in mT/m, position in mm; γ is the value of the engine.

## 5. The sequences

Every builder of `sequences` returns the block list, the time spent transverse and stored along z, and the times
of the refocusing pulses.

| Builder | Parameters | Pulses and pathway |
|---|---|---|
| `pulse_acquire` | flip angle | 90x, acquire |
| `spin_echo` | TE; optional slice-selective refocusing pulse | 90x – TE/2 – 180y – TE/2 |
| `press` | TE, or TE1 and TE2; refocusing pulse, carrier, positions | 90x – TE1/2 – 180y – (TE1+TE2)/2 – 180y – TE2/2 |
| `steam` | TE, TM | 90x – TE/2 – 90x – TM – 90(−x) – TE/2; orders +1, 0, −1 |
| `slaser` | TE; adiabatic pulse of `mrgym.design.pulses.sech` | 90x and two pairs of full passages at TE × (1, 3, 5, 7)/8 |
| `mega_press` | TE, TE1, editing frequency; ideal or Gaussian editing pulse | PRESS with two selective 180s, TE/2 apart, around the second refocusing pulse |
| `jres` | first TE, increments | PRESS at incremented echo times |
| `cosy` | increments | 90x – t1 – 90x – acquire, with or without gradient selection |

Pulses from protocol numbers: `refocusing_pulse(bandwidth, thickness)` (Shinnar-Le Roux, duration = time-bandwidth
product / bandwidth), `adiabatic_pulse(bandwidth, duration, thickness)` (hyperbolic secant, peak B1 10 % above
γ B1 = bandwidth / 2) and `gaussian(flip, duration)` (editing pulse; its inversion bandwidth is computed).

The excitation is ideal in all of them. In `slaser` the ensemble of an axis stays open between the two pulses of
its pair, because the phase that one adiabatic pulse leaves across the slice is removed only by the second
(row S6).

**STEAM and the zero-quantum coherence.** For a weakly coupled pair A, X with ideal pulses the signal of A is

    s_A = (1/2) cos²θ + (1/4) sin²θ [−(1 − cos ε) + i sin ε] cos(2π (ν_A − ν_X) TM),    θ = π J TE/2,  ε = 2π (ν_A − ν_X) TE/2

Product operators: the first TE/2 turns −Iy into −Iy cos θ + 2 Ix Sz sin θ (and the shifts); the second pulse makes
−Iz cos θ cos α_I and −2 Ix Sy sin θ cos α_I from them (α the precession angle in TE/2, crusher included), and the
same for X; the zero-quantum part of 2 Ix Sy passes the crusher of TM together with Iz and oscillates there at
ν_A − ν_X; the third pulse turns it into 2 Ix Sz, which the coupling turns into Iy during the last TE/2. Averaging
over the crusher phase leaves the terms above. Row S5c compares the three coefficients with the simulation, row
S5e the whole sequence with an explicit ensemble for three strongly coupled protons.

## 6. The metabolite basis

`metabolites.BASIS`; the table is generated by `metabolites.table_markdown()` and a test compares the two. Tags:
[F] read in a fetched document, which for the brain metabolites is a software copy of the published table and not
the paper; [E] the value of this package's engine; [M] recalled, no source fetched.

<!-- metabolite table: begin -->
| Metabolite | Network | Site: shift in ppm (protons) | Couplings in Hz | Source | |
|---|---|---|---|---|---|
| water | water | water 4.7 (2) | none | engine convention (4.7 ppm); spant 4.65, FID-A 4.7 | [E] |
| lactate | lactate | CH3 1.3142 (3), CH 4.0974 | CH3-CH 6.933 | Govindaraju 2000, as copied in spant (and equal in VeSPA) | [F] |
| alanine | alanine | CH3 1.4667 (3), CH 3.7746 | CH3-CH 7.234 | Govindaraju 2000, as copied in spant (and equal in VeSPA) | [F] |
| naa | naa_acetyl | naa_acetyl 2.008 (3) | none | Govindaraju 2000, as copied in spant (and equal in VeSPA), the entry with the amide proton | [F] |
| naa | naa_aspartyl | CH 4.3817, CH2a 2.6727, CH2b 2.4863, NH 7.8205 | CH-CH2a 3.861, CH-CH2b 9.821, CH-NH 6.4, CH2a-CH2b -15.592 | Govindaraju 2000, as copied in spant (and equal in VeSPA), the entry with the amide proton | [F] |
| creatine | cr_ch3 | cr_ch3 3.027 (3) | none | Govindaraju 2000, as copied in spant (and equal in VeSPA); NH at 6.649 ppm left out | [F] |
| creatine | cr_ch2 | cr_ch2 3.913 (2) | none | Govindaraju 2000, as copied in spant (and equal in VeSPA); NH at 6.649 ppm left out | [F] |
| choline | cho_ch3 | cho_ch3 3.185 (9) | none | Govindaraju 2015 (corrigendum), as copied in spant; couplings to 14N (0.57, 2.572, 2.681 Hz) left out | [F] |
| choline | cho_ch2 | 1CH2a 4.054, 1CH2b 4.054, 2CH2a 3.501, 2CH2b 3.501 | 1CH2a-1CH2b -14.1, 1CH2a-2CH2a 3.14, 1CH2a-2CH2b 6.979, 1CH2b-2CH2a 7.011, 1CH2b-2CH2b 3.168, 2CH2a-2CH2b -14.07 | Govindaraju 2015 (corrigendum), as copied in spant; couplings to 14N (0.57, 2.572, 2.681 Hz) left out | [F] |
| gaba | gaba | 4CH2a 3.0128, 4CH2b 3.0128, 3CH2a 1.889, 3CH2b 1.889, 2CH2a 2.284, 2CH2b 2.284 | 4CH2a-4CH2b -12.021, 4CH2a-3CH2a 5.372, 4CH2a-3CH2b 7.127, 4CH2b-3CH2a 10.578, 4CH2b-3CH2b 6.982, 3CH2a-3CH2b -13.121, 3CH2a-2CH2a 7.755, 3CH2a-2CH2b 7.432, 3CH2b-2CH2a 6.173, 3CH2b-2CH2b 7.933, 2CH2a-2CH2b -10.744 | Govindaraju 2015 (corrigendum), as copied in spant (and equal in VeSPA) | [F] |
| gaba_near2013 | gaba_near2013 | 2CH2a 2.284, 2CH2b 2.284, 3CH2a 1.888, 3CH2b 1.888, 4CH2a 3.013, 4CH2b 3.013 | 2CH2a-2CH2b -15.938, 2CH2a-3CH2a 7.678, 2CH2a-3CH2b 6.98, 2CH2b-3CH2a 6.98, 2CH2b-3CH2b 7.678, 3CH2a-3CH2b -15, 3CH2a-4CH2a 8.51, 3CH2a-4CH2b 6.503, 3CH2b-4CH2a 6.503, 3CH2b-4CH2b 8.51, 4CH2a-4CH2b -14.062 | Near 2013, Table 1; equal in spant; an alternative set; not part of the default basis | [F] |
| glutamate | glutamate | 2CH 3.7433, 3CH2a 2.0375, 3CH2b 2.12, 4CH2a 2.3378, 4CH2b 2.352 | 2CH-3CH2a 7.331, 2CH-3CH2b 4.651, 3CH2a-3CH2b -14.849, 3CH2a-4CH2a 6.413, 3CH2a-4CH2b 8.406, 3CH2b-4CH2a 8.478, 3CH2b-4CH2b 6.875, 4CH2a-4CH2b -15.915 | Govindaraju 2000, as copied in spant (and equal in VeSPA) | [F] |
| glutamine | glutamine | 2CH 3.753, 3CH2a 2.129, 3CH2b 2.109, 4CH2a 2.432, 4CH2b 2.454 | 2CH-3CH2a 5.847, 2CH-3CH2b 6.5, 3CH2a-3CH2b -14.504, 3CH2a-4CH2a 9.165, 3CH2a-4CH2b 6.347, 3CH2b-4CH2a 6.324, 3CH2b-4CH2b 9.209, 4CH2a-4CH2b -15.371 | Govindaraju 2000, as copied in spant (and equal in VeSPA); NH2 at 6.816 and 7.529 ppm left out | [F] |
| myo_inositol | myo_inositol | 1CH 3.5217, 2CH 4.0538, 3CH 3.5217, 4CH 3.6144, 5CH 3.269, 6CH 3.6144 | 1CH-2CH 2.889, 1CH-6CH 9.998, 2CH-3CH 3.006, 3CH-4CH 9.997, 4CH-5CH 9.485, 5CH-6CH 9.482 | Govindaraju 2000, as copied in spant (and equal in VeSPA) | [F] |
| lipid_pair | lipid_pair | CH3 0.9 (3), CH2 1.3 (2) | CH3-CH2 7 | shifts of the methyl and methylene peaks of Hamilton 2011 (Table 1, read); the coupling of 7 Hz is recalled, not read; a two-group model: the methylene is coupled to the methyl only, not to the rest of the chain | [M] |
| triglyceride | fat_1 x 0.08709 | fat_1 0.9 | none | mrgym.scanner.tissue.FAT_PEAKS (the engine attributes them to Hamilton 2011; the table read for this release has 4.20 ppm and 8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the engine has 4.31 ppm and 8.7 / 69.3 / 12.8 / 0.4 / 3.9 / 4.8 %); uncoupled lines whose amplitudes sum to 1, as in the engine | [E] |
| triglyceride | fat_2 x 0.6937 | fat_2 1.3 | none | mrgym.scanner.tissue.FAT_PEAKS (the engine attributes them to Hamilton 2011; the table read for this release has 4.20 ppm and 8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the engine has 4.31 ppm and 8.7 / 69.3 / 12.8 / 0.4 / 3.9 / 4.8 %); uncoupled lines whose amplitudes sum to 1, as in the engine | [E] |
| triglyceride | fat_3 x 0.1281 | fat_3 2.1 | none | mrgym.scanner.tissue.FAT_PEAKS (the engine attributes them to Hamilton 2011; the table read for this release has 4.20 ppm and 8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the engine has 4.31 ppm and 8.7 / 69.3 / 12.8 / 0.4 / 3.9 / 4.8 %); uncoupled lines whose amplitudes sum to 1, as in the engine | [E] |
| triglyceride | fat_4 x 0.004004 | fat_4 2.76 | none | mrgym.scanner.tissue.FAT_PEAKS (the engine attributes them to Hamilton 2011; the table read for this release has 4.20 ppm and 8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the engine has 4.31 ppm and 8.7 / 69.3 / 12.8 / 0.4 / 3.9 / 4.8 %); uncoupled lines whose amplitudes sum to 1, as in the engine | [E] |
| triglyceride | fat_5 x 0.03904 | fat_5 4.31 | none | mrgym.scanner.tissue.FAT_PEAKS (the engine attributes them to Hamilton 2011; the table read for this release has 4.20 ppm and 8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the engine has 4.31 ppm and 8.7 / 69.3 / 12.8 / 0.4 / 3.9 / 4.8 %); uncoupled lines whose amplitudes sum to 1, as in the engine | [E] |
| triglyceride | fat_6 x 0.04805 | fat_6 5.3 | none | mrgym.scanner.tissue.FAT_PEAKS (the engine attributes them to Hamilton 2011; the table read for this release has 4.20 ppm and 8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the engine has 4.31 ppm and 8.7 / 69.3 / 12.8 / 0.4 / 3.9 / 4.8 %); uncoupled lines whose amplitudes sum to 1, as in the engine | [E] |
<!-- metabolite table: end -->

**Not verified.**

<!-- not verified: begin -->
- lipid_pair: the coupling of 7.0 Hz between methyl and methylene [M]
- every brain metabolite: read in software copies (spant, VeSPA) of Govindaraju 2000 and of the 2015 corrigendum, not in the papers
- gaba: which of the two sets of four vicinal couplings belongs to the 3.01-1.89 ppm pair and which to the 1.89-2.28 ppm pair is taken from spant and VeSPA; the corrigendum was not read
<!-- not verified: end -->

Water relaxation times and water contents (`metabolites.WATER_3T`) are those quoted by Harris et al. 2015 and Near
et al. 2021. Relaxation times of metabolites are inputs of a simulation; the basis holds none.

## 7. Quantification

`quantify.fit` is linear-combination modelling in its smallest form: real amplitudes of the basis decays, one common
frequency shift, one common extra Lorentzian width, one zero-order phase, by variable projection. `quantify.crb`
gives the Cramér-Rao bounds of that model (rows Q2, Q3). `quantify.water_referenced` is the formula of Near et al.
2021, equations 4 and 5, for the molal concentration in tissue (row Q4; row Q4b gives the size of the error when the
recovery of a PRESS repetition is corrected with 1 − exp(−TR/T1)). `quantify.snr` is the peak height over the
standard deviation of the noise in the frequency domain and `quantify.linewidth_hz` the full width at half maximum
of a singlet, as Kreis et al. 2020 define them (rows Q5, Q5b).

## 8. Validation

`python -m mrgym.spins.validate` runs the rows below and exits with status 1 if one fails;
`tests/test_spins_validate.py` asserts that all pass and that this table is the one the command prints
(`--markdown`). Errors below 1e-3 are printed with one significant digit: they are rounding noise and differ
between machines, and so do the times. The references were written by the same authors as the code: the rows show
that the code computes the stated equations. Rows that quote a published number say which, and where it was read.

<!-- validation table: begin -->
### Operators and Hamiltonian

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| O1 | spin matrices for j = 1/2 to 5 | [Jx, Jy] = i Jz and cyclic; Jx^2 + Jy^2 + Jz^2 = j (j + 1) | 4.0e-15 | ≤ 1.0e-12 | pass |
| O2 | total-spin decomposition of 1 to 12 equivalent protons | sum over j of g(n, j) (2j + 1) = 2^n | 0 | ≤ 0 | pass |
| O3 | rotation matrices of a group of total spin 1/2 to 9/2 for 40 random rotations given as Cayley-Klein parameters of `mrgym.design.rotation` | `scipy.linalg.expm` of -i angle n.J | 3.2e-15 | ≤ 1.0e-12 | pass |
| O4 | Hamiltonian of 2, 4 and 6 random strongly coupled protons rebuilt from its eigenvalues and eigenvectors; Fz of the eigenvectors | sum nu Iz + sum J I.I from Pauli matrices; Fz diagonal with a definite M per eigenvector | 4.0e-15 | ≤ 1.0e-12 | pass |
| O5 | AX spectrum (weak coupling, 300 Hz apart, J = 7 Hz): four line positions (Hz) and amplitudes | nu_A +- J/2, nu_X +- J/2; amplitude 1/2 each | 0 | ≤ 1.0e-10 | pass |
| O5b | AB spectrum (strong coupling; shift difference d from 300 to 3 Hz, J from 7 to 15 Hz and -12 Hz): positions and amplitudes | centre +- (C +- |J|)/2, C = sqrt(d^2 + J^2); inner lines (1 + |J|/C)/2, outer (1 - |J|/C)/2: the roof effect | 5.7e-14 | ≤ 1.0e-10 | pass |
| O6 | A3X spectrum (lactate shifts and coupling, weak coupling) at 3 T: positions (Hz) and amplitudes of the six lines | doublet nu_A +- J/2 of 3/2 each; quartet nu_X + (-3, -1, 1, 3) J/2 of 1/8, 3/8, 3/8, 1/8 | 5.7e-14 | ≤ 1.0e-10 | pass |
| O6b | the same with strong coupling: total amplitude of the lines within 20 Hz of the doublet and of the quartet | 3 and 1 protons to within 1e-6: at 3 T (J / shift difference = 0.02) the coupling moves intensity inside each multiplet, and between the two only in a high order | 5.4e-08 | ≤ 1.0e-06 | pass |
| O7 | equivalent groups as total spins (A3X, A3M2X, A2M2X2, A3M2X2, random shifts, strong coupling), free induction decay after PRESS and a further pulse; largest matrix 36 instead of 128 | every proton as its own site, with a coupling of -12.4 Hz inside the groups (it must not matter) | 7.6e-14 | ≤ 1.0e-11 | pass |
| O8 | two protons 1.5 ppm apart with J = 9 Hz at 1.5, 3 and 7 T: the two splittings and the distance of the doublet centres | splittings J at every field; distance sqrt(d^2 + J^2) with d = 1.5 ppm x gamma B0 (95.8, 191.6, 447.1 Hz) | 1.1e-13 Hz | ≤ 1.0e-10 | pass |

### Propagation and detection

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| E1 | propagators of a hard pulse and of a slice-selective sinc (both models) for 2, 3 and 5 coupled protons; a state carried through pulse, delay, pulse, delay | U U^+ = 1; trace, trace of rho^2 and Hermiticity conserved | 2.6e-13 | ≤ 1.0e-11 | pass |
| E2 | three uncoupled protons (1.3, 3.0, 4.7 ppm) from a tilted start through a slice-selective sinc, a Shinnar-Le Roux refocusing pulse and a hyperbolic secant at 13 positions, both pulse models: Mxy and Mz of every site | `mrgym.design.rotation.rotate` and `apply` at the site's frequency (the Bloch rotation) | 4.1e-13 | ≤ 1.0e-11 | pass |
| E3 | a 12 ms Gaussian 180 near two strongly coupled protons (32 Hz apart, J = 11 Hz), carrier 300 Hz below the reference, model full, 400 samples: the propagator | `scipy.linalg.expm` of the whole Hamiltonian for every sample, Pauli matrices in the product basis | 8.4e-06 | ≤ 1.0e-04 | pass |
| E3o | the same: order of convergence from 50 to 400 samples | second order (hard-pulse splitting) | 1.6e-05 in the order | ≤ 0.05 | pass |
| E3b | the two pulse models for that pulse without coupling | equal: the model product is exact for uncoupled spins | 1.8e-13 | ≤ 1.0e-11 | pass |
| E3c | the model product for that pulse with the coupling (J x T = 0.13) | the model full; the difference is the coupling evolution that the pulse interrupts, a fraction of J x T | 0.008421 | ≤ 0.02 | pass |
| E4 | free induction decay (24 samples) of four strongly coupled protons after three pulses, from the line list | i Tr(rho(t) F+) / 2^(n-2), the state stepped with `scipy.linalg.expm` in the product basis | 1.3e-14 | ≤ 1.0e-11 | pass |
| E5 | spectrum of one decaying line (T2 = 80 ms, on a grid point, 16,384 samples of 0.25 ms): height, full width at half maximum, area of the real part, position | height A T2, width 1 / (pi T2), area A / 2 (relative errors); position (in points) | 9.9e-04 | ≤ 0.002 | pass |
| E5b | the sampled spectrum at every frequency, relative to the peak | the sum of the sampled decay in closed form: A dwell (1/2) coth(z / 2), z = dwell (1 / T2 + 2 pi i (f - f0)) | 2.1e-15 | ≤ 1.0e-10 | pass |
| E5c | the sampled spectrum against the continuous one, relative to the peak | A / (1 / T2 + 2 pi i (f - f0)); the difference is the sampling, A dwell z / 12 to first order (1.0e-03 at the edge of the band) | 0.001022 | ≤ 0.002 | pass |
| E6 | spin echo of an AX pair (weak coupling, J = 7.3 Hz), echo times 10 to 200 ms: signal of A with a 180 on both spins and with a 180 on A only | cos(pi J TE) (the chemical shift is refocused, the coupling is not); 1 (the coupling is refocused too) | 1.3e-14 | ≤ 1.0e-11 | pass |
| E7 | INEPT on an AX pair (weak coupling): 90x(I) - tau - 180x(I, S) - tau, then 90y(I) 90x(S); coefficients of Iy, 2 Ix Sz, Sz and 2 Iz Sy for four tau, one of them 1 / (4J) | product operators (Sorensen et al. 1983): Iz -> Iy cos(2 pi J tau) - 2 Ix Sz sin(2 pi J tau), Sz -> -Sz; then 2 Ix Sz -> 2 Iz Sy: complete transfer to antiphase S at tau = 1 / (4J) | 1.2e-14 | ≤ 1.0e-11 | pass |
| E8 | three strongly coupled protons: a 140 degree pulse between equal crushers and a selective 160 degree pulse between opposite crushers, as projections on coherence orders (p -> -p, p -> p) | explicit ensembles of 16 gradient phases, summed at the end | 5.8e-16 | ≤ 1.0e-11 | pass |
| E8b | the same sequence | 256 isochromats with z rotations for the gradients, Pauli matrices and `scipy.linalg.expm` | 7.4e-16 | ≤ 1.0e-11 | pass |
| E9 | longitudinal magnetisation before the excitation in the steady state: 90 degree pulse; spin echo; 30 degree pulse | 1 - E1; 1 - 2 exp(-(TR - TE/2) / T1) + E1; (1 - E1) / (1 - E1 cos 30) | 1.1e-16 | ≤ 1.0e-13 | pass |
| E9b | an uncoupled proton with T1 = 1350 ms, T2 = 110 ms in the steady state of spin echo (TE 40, TR 900 ms), PRESS (20 + 40, 1200), semi-LASER (64, 1500) and MEGA-PRESS without editing (68, 1100): signal from the relaxation factors of the sequence | the Bloch equations stepped over 40 repetitions with real 3 x 3 rotations, refocusing pulses at times written by hand | 1.5e-16 | ≤ 1.0e-12 | pass |
| E9c | the same proton in the steady state of STEAM (TE 20, TM 30, TR 2000 ms and TE 30, TM 100, TR 1000 ms) | the Bloch equations for 8 crusher phases of TE, transverse magnetisation removed in TM: the third pulse leaves no z magnetisation, so the recovery runs for TR - TE/2 - TM | 1.5e-16 | ≤ 1.0e-12 | pass |
| E9d | the same against 64 isochromats whose TM crusher is a phase that repeats in every repetition, relative | not equal: what is transverse in TM returns to z at the third pulse and comes back as signal one repetition later. The ideal spoiler of this package leaves that pathway out | 0.003157 | ≤ 0.01 | pass |

### Sequences

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| S1 | lactate methyl doublet at TE = 1 / J = 144.2 ms, ideal pulses, weak coupling: spin echo, symmetric and asymmetric PRESS | -3: the doublet is inverted (cos(pi J TE) = -1) | 1.9e-14 | ≤ 1.0e-10 | pass |
| S1b | the same with strong coupling at 3 T (symmetric PRESS) | -3 to within 1 % (J / shift difference = 0.02) | 0.00189 | ≤ 0.01 | pass |
| S2 | lactate at 3 T, spin echo of 144 ms with one slice-selective refocusing pulse of 874 Hz (time-bandwidth 16, model full): relative loss of the doublet, 0.809 | 2 x shift difference / bandwidth = 2 x 356 / 874 = 0.814 for a rectangular profile; Lange et al. 2006 (AJNR 27:895, appendix) state 81.2 % for 355 Hz and 874 Hz and measured 72.2 % | 0.004616 | ≤ 0.03 | pass |
| S2b | the same: model product against model full, relative to the ideal signal | equal up to the coupling evolution during the 18 ms pulse | 0.002252 | ≤ 0.03 | pass |
| S2c | the same loss with a 4.6 ms pulse of the same bandwidth (time-bandwidth 4: wide transition bands), 0.807 | the same 2 x shift difference / bandwidth: transition bands that are symmetric about the slice edge do not change the loss to first order. The 72.2 % that Lange et al. measured is 9 points below their formula and below this simulation; the width of the transition bands does not explain it | 0.006837 | ≤ 0.03 | pass |
| S3 | lactate at 3 T, PRESS with two slice-selective Shinnar-Le Roux refocusing pulses (time-bandwidth 12; 1.0, 1.2, 2.0 kHz; TE1 / TE2 = 72 / 72, 30 / 114, 144 / 144, 40 / 60 ms), model full: methyl signal per proton | four compartments with rectangular profiles: (1-f)^2 cos(pi J TE) + f (1-f) (cos(pi J TE1) + cos(pi J TE2)) + f^2, f = shift difference / bandwidth | 0.007167 | ≤ 0.015 | pass |
| S4 | centre of the refocused slice (midpoint of the half-height edges of |b|^2) of a site 3 ppm below, 1 ppm below and 2 ppm above the carrier, 1 kHz pulse at 3 T, in slice thicknesses | -(frequency offset) / bandwidth: a site at a higher frequency is selected at a more negative position (0.38 of the slice for 3 ppm at 127.7 MHz and 1 kHz; Oz et al. 2021 give 35 % per 3 ppm for 1.06 kHz, which is 123 MHz) | 2.9e-07 | ≤ 0.001 | pass |
| S5 | STEAM (TE 20 ms, TM 30 ms) of an uncoupled proton 64 Hz off resonance with T1 = 1200 ms, T2 = 90 ms | (1/2) exp(-TE / T2) exp(-TM / T1): half of a spin echo | 9.8e-17 | ≤ 1.0e-12 | pass |
| S5b | STEAM (TE 40 ms) of an AX pair (weak coupling, J = 7 Hz, 2.8 ppm apart at 3 T), 41 mixing times from 5 to 25 ms: signal of A | a constant plus one oscillation at the zero-quantum frequency nu_A - nu_X = -357.7 Hz (residual of that fit) | 2.3e-16 | ≤ 1.0e-10 | pass |
| S5c | the same: the constant and the coefficients of the cosine and the sine | (1/2) cos^2(t) and (1/4) sin^2(t) (-(1 - cos e) + i sin e), 0, with t = pi J TE/2 and e = 2 pi (nu_A - nu_X) TE/2 (product operators; derivation in docs/physics/spins.md) | 3.4e-16 | ≤ 1.0e-10 | pass |
| S5d | the same: size of the oscillation relative to the constant at TE = 40 ms | not zero: the mixing time changes the signal of a coupled spin | 0.1024 | ≥ 0.05 | pass |
| S5e | STEAM (TE 30 ms, TM 13 ms) of three strongly coupled protons: free induction decay | 64 isochromats with z rotations for the crushers of TE and TM, Pauli matrices and `scipy.linalg.expm` | 1.9e-15 | ≤ 1.0e-11 | pass |
| S6 | semi-LASER (TE 144 ms) with two pairs of hyperbolic-secant pulses (1999 Hz, 8 ms, 25 uT), 64 positions per axis, on a singlet at the carrier: signal per proton, 0.869 | (integral of |b|^4 over the slice / thickness)^2 with b from `mrgym.design.rotation.rotate` on 6,001 positions: the phase that one adiabatic pulse leaves is removed by the second; the signal is below 1 because the profile of this pulse has wide transitions | 1.2e-09 | ≤ 0.002 | pass |
| S6b | lactate methyl under that sequence: model product against model full, per proton | equal up to the coupling evolution during the four 8 ms pulses | 0.01361 | ≤ 0.03 | pass |
| S6c | the same (model full) relative to the singlet: -0.605; PRESS with 1.2 kHz pulses keeps -0.406 | compartments with rectangular profiles, two pairs: -(1 - 2f) = -0.644 with f = shift difference / bandwidth = 0.178; the profile of the pulse is not rectangular, hence the tolerance | 0.03908 | ≤ 0.06 | pass |
| S7 | MEGA-PRESS of an A2M2 system (weak coupling, J = 7.35 Hz) at TE = 1 / (2J) = 68 ms, ideal inversion of M: signal of A with and without the editing pulses, and (on - off) / 2 per proton | on: 2 (triplet refocused); off: 0 (outer lines inverted, centre line unchanged); editing efficiency 1/2, the value assumed for GABA (Mullins et al. 2014; Harris et al. 2015) | 6.8e-15 | ≤ 1.0e-10 | pass |
| S7b | GABA (the basis entry) at 3 T, MEGA-PRESS TE 68 ms, editing at 1.9 ppm: (on - off) / 2 of the 3.01 ppm multiplet per proton, with an ideal inversion (0.489) and with a 14 ms Gaussian of 109 Hz, off at 7.46 ppm (0.483) | 0.5 (the efficiency assumed in quantification: Mullins et al. 2014, eq. 1; Harris et al. 2015); the multiplet is not an ideal triplet, so the value is near but not at 0.5 | 0.01715 | ≤ 0.03 | pass |
| S7c | shape of that difference spectrum (ideal inversion, 2 Hz lines): two maxima 14.7 Hz apart, the value at the centre relative to them | a pseudo-doublet: the outer lines of the multiplet are edited, the centre is low but not absent (Near et al. 2013); distance of the maxima between 12 and 18 Hz (the sums of the two vicinal couplings are 12.5 and 17.6 Hz) | -0.3048 | ≤ 0 | pass |
| S7d | lactate, MEGA-PRESS at TE = 1 / J with an ideal inversion of the methine proton at 4.1 ppm (weak coupling): methyl signal on and off | +3 and -3: the whole doublet is in the difference (efficiency 1) | 1.0e-14 | ≤ 1.0e-10 | pass |
| S8 | 2D J-resolved signal of an AX pair (weak coupling), 16 echo times x 32 samples | sum over the four lines of (1/2) exp(2 pi i (nu +- J/2) t2) exp(+- i pi J t1): a line at nu +- J/2 in F2 sits at +- J/2 in F1 | 2.6e-14 | ≤ 1.0e-11 | pass |
| S8b | gradient-selected COSY of two strongly coupled protons, 6 x 8 samples | 8 isochromats with equal crushers around the second pulse, Pauli matrices and `scipy.linalg.expm` | 1.4e-15 | ≤ 1.0e-11 | pass |
| S8c | the same experiment on two uncoupled protons | -(1/2) sum over the sites of exp(2 pi i nu (t2 - t1)): diagonal peaks only, no cross peak without coupling | 3.2e-15 | ≤ 1.0e-12 | pass |
| S8d | magnitude of the 2D spectrum (64 x 64, 60 ms decay) at (F1 of A, F2 of B) relative to the diagonal peak of A: 0.90 with J = 9 Hz, 0.04 without (leakage of the diagonal); the difference | a cross peak appears with the coupling: at least 0.3 | 0.8601 | ≥ 0.3 | pass |

### Metabolite basis

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| M1 | the six peaks of the entry `triglyceride` and the gyromagnetic ratio | `mrgym.scanner.tissue.FAT_PEAKS`, `console.GAMMA_HZ_PER_T` | 0 | ≤ 0 | pass |
| M1b | free induction decay of `triglyceride` after a 90 degree pulse at 1.5 and 3 T, reference on water | the engine's `scanner/signal.py::fat_spectrum` | 8.7e-15 | ≤ 1.0e-12 | pass |
| M2 | all 13 entries after a 90 degree pulse at 3 T: sum of the line amplitudes | the number of protons of the entry | 2.5e-13 | ≤ 1.0e-10 | pass |
| M2b | the same: first moment of the spectrum, sum of amplitude x frequency | sum of the site frequencies (a coupling moves no centre of gravity: Tr(F- [H_J, F+]) = 0) | 1.3e-10 Hz | ≤ 1.0e-08 | pass |
| M3 | lactate, alanine and the lipid pair under PRESS: methyl and methylene groups as total spins | every proton as its own site | 1.6e-13 | ≤ 1.0e-11 | pass |
| M4 | time of the PRESS basis (TE 30 ms, 12 entries, 2,048 points, 3 T), ideal refocusing pulses, one core (best of 3) | requirement | 0.1035 s | ≤ 2 | pass |
| M4b | the same with slice-selective refocusing pulses (1.2 kHz, 32 positions per pulse, model product) | requirement | 0.7088 s | ≤ 2 | pass |
| M4c | time of one network of seven strongly coupled protons (matrices of 128, about 3,000 lines) under PRESS, 2,048 points: ideal refocusing pulses | requirement | 0.1371 s | ≤ 2 | pass |
| M4d | time of one network of seven strongly coupled protons (matrices of 128, about 3,000 lines) under PRESS, 2,048 points: the slice-selective pulses of M4b, model product | requirement | 1.327 s | ≤ 10 | pass |

### Quantification

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| Q1 | fit of a noise-free mixture of six basis spectra (PRESS, 3 T) with a shift of 2.3 Hz, 1.7 Hz of extra width and a phase of 0.4 rad | the amplitudes, shift, width and phase that were put in | 1.6e-14 | ≤ 1.0e-06 | pass |
| Q1b | the same, fitted in the frequency domain between 0.8 and 4.2 ppm | the amplitudes that were put in | 3.6e-15 | ≤ 1.0e-06 | pass |
| Q2 | Cramer-Rao bounds of the nine parameters of that fit | inverse Fisher matrix from central differences of the model | 1.2e-10 relative | ≤ 1.0e-06 | pass |
| Q3 | 300 noisy copies of that mixture (seeded; noise 0.6 per sample, first sample of NAA / noise = 111): standard deviation of every fitted amplitude relative to its Cramer-Rao bound (the bounds are 0.2 to 2.3 % of the amplitudes) | 1 (a least-squares fit attains the bound at this noise); the tolerance is 3.7 standard errors of a standard deviation from 300 trials | 0.06595 | ≤ 0.15 | pass |
| Q3b | the same: bias of every amplitude in units of its bound | 0 within 3.5 standard errors of the mean (0.2 bounds) | 0.1091 | ≤ 0.2 | pass |
| Q4 | water-referenced concentration of creatine (methyl singlet, PRESS TE 30 ms, TR 2 s) in a voxel of 55 % grey matter, 35 % white matter, 10 % CSF. The signals are counted by hand: protons x mmol per kg of water x kg of water in the compartment x exp(-TE/T2) (1 - exp(-TR/T1)), water at 55,510 mmol/kg in all three compartments, creatine in the tissue only | the molal concentration that was put in, 10.3 mmol per kg of tissue water (formula of Near et al. 2021, eqs. 4 and 5; water content and relaxation times as quoted by Harris et al. 2015) | 4.4e-16 relative | ≤ 1.0e-12 | pass |
| Q4b | the same when the signals are built with the recovery of a PRESS repetition (two inversions after the excitation) and the formula corrects with 1 - exp(-TR / T1) | an error of the order of TE / TR, stated here so that a task can account for it | 6.1e-04 relative | ≤ 0.01 | pass |
| Q5 | signal-to-noise ratio of a singlet in the frequency domain (peak height / standard deviation of the noise, Kreis et al. 2020), measured 283 | A T2 / (sigma x dwell x sqrt(n - 3/4)) = 281; the tolerance covers the noise on the peak and on the estimate of the noise | 0.004969 relative | ≤ 0.05 | pass |
| Q5b | full width at half maximum of that singlet with 3 Hz of extra Lorentzian width, measured on the real spectrum (4 x zero filling) | 1 / (pi T2) + 3 Hz = 8.31 Hz | 1.8e-04 Hz | ≤ 0.02 | pass |
<!-- validation table: end -->

Literature behind the rows, read in open copies: Lange et al., AJNR 27:895 (2006), appendix (lactate loss);
Mullins et al., NeuroImage 86:43 (2014), doi:10.1016/j.neuroimage.2012.12.004, and Harris et al., J Magn Reson
Imaging 42:1431 (2015), doi:10.1002/jmri.24903 (editing efficiency 0.5; water relaxation); Near et al., Magn Reson
Med 70:1183 (2013), doi:10.1002/mrm.24572 (pseudo-doublet); Öz et al., NMR Biomed 34:e4236 (2021),
doi:10.1002/nbm.4236 (displacement); Kreis et al., NMR Biomed 34:e4347 (2021), doi:10.1002/nbm.4347
(signal-to-noise ratio, line width); Near et al., NMR Biomed 34:e4257 (2021), doi:10.1002/nbm.4257 (water
referencing); Hamilton et al., NMR Biomed 24:784 (2011), doi:10.1002/nbm.1622 (fat spectrum).

Not read: Sørensen et al., Prog NMR Spectrosc 16:163 (product operators; the rules used are written out in the
rows), Bottomley 1987 (PRESS), Frahm et al. 1987 (STEAM), Mescher et al. 1998 (MEGA), Scheenen et al. 2008
(semi-LASER), Yablonskiy et al. 1998 (four compartments), Cavassila et al. 2001 (Cramér-Rao bounds), Levitt, *Spin
Dynamics*, de Graaf, *In Vivo NMR Spectroscopy*. The sequences and formulas attributed to them here are what the
authors of this package know of them; they are checked by the rows above, not against those texts.

## 9. Where this meets the Bloch and phase-graph layer

The blocks are those of `mrgym.design.sequence` with the same meaning: `rf`, `rf_pulse`, `delay`, `gradient` in
whole cycles, `spoil`, and `acquire` for `readout`. A block list of that layer can be played here block by block;
what the spin layer adds is the system the blocks act on (a density matrix per network instead of F and Z states)
and the pathway argument of a pulse. For uncoupled spins the two layers compute the same magnetisation (row E2).

## 10. What is not built yet

- **Cross-relaxation and the nuclear Overhauser effect, cross-correlation, different relaxation rates of antiphase
  and multiple-quantum terms.** Relaxation is a factor per line. Zero-quantum coherence in a mixing time is given
  T1; nothing relaxes during a shaped pulse; the recovery of a group that editing pulses invert is computed as if
  they were not there.
- **Signal that returns after more than one repetition.** Crushers are ideal within a repetition and everything
  transverse is gone before the next excitation (row E9d gives the size of what this leaves out in STEAM).
- **Chemical exchange** (the amide and amine protons, water exchange, saturation transfer).
- **Macromolecules and a baseline.** Neither a measured macromolecule spectrum nor a parametrised one is in the basis
  or in the fit.
- **Other nuclei.** No 31P, 13C or 14N; the couplings of choline protons to 14N are left out.
- **Field inhomogeneity as a distribution over the voxel.** It is a line shape; a field map of the scanner world is
  not used.
- **Eddy currents**, frequency drift, motion, water suppression and its side bands, outer-volume suppression.
- **Slice-selective excitation in the sequences.** The excitation is ideal; `evolve.rf_design` plays a designed
  excitation pulse, and no builder uses it yet. Positions are relative to the voxel centre; no off-centre voxel.
- **B1 inhomogeneity.** `rf_pulse` takes one B1 scale; no spatial pattern.
- **Shaped pulses in STEAM, and gradient-modulated adiabatic pulses** for semi-LASER.
- **A voxel in the scanner world, a console sequence, a tool**: the layer is not connected to the scanner.
