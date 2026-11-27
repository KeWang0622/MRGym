# The signal core: a sequence as a list of blocks, evaluated from the Bloch equation

`mrgym.core` computes the signal of a pulse sequence that is written as a list of blocks. It is a library next to the
scanner engine. `mrgym.scanner` still computes each of its six sequences from that sequence's own closed-form steady
state; the core reproduces those six from block lists (section 9, group "engine"; in shadow at the console, section
11) and holds eleven more sequences. Four of them (`fse`, `mese`, `megre`, `dwse`) are served by a console whose card
lists them: section 11 says how. The core holds no state and serves no task.

Read [conventions.md](conventions.md) and [design.md](design.md) first; the sign conventions here are theirs
(section 8).

## 1. From the Bloch equation to two operators

The magnetisation M of a group of spins that see the same field (an isochromat) obeys

    dM/dt = γ M × B − (Mx, My, 0) / T2 − (0, 0, Mz − M0) / T1.

In the frame that rotates at the transmitter frequency two cases cover a sequence.

- **Between pulses** the field is along z: the off-resonance Δf of the spin plus what the gradients add at its
  position. The transverse part Mxy = Mx + i My turns by 2π (Δf t + k · r), k = γ ∫ G dt, and decays with T2; Mz
  recovers towards M0 with T1. Both are exact for any gradient waveform.
- **During a pulse** B1 is so much larger than the relaxation rates that relaxation is neglected, and the pulse
  is a rotation. A hard pulse of flip angle α and phase p turns about the axis (cos p, sin p, 0). A slice-selective
  pulse is a train of small hard pulses with precession under the slice gradient between them; their product is
  one rotation per position through the slice, held as two complex numbers (a, b) with |a|² + |b|² = 1
  (Cayley-Klein parameters; `design.rotation`). A rotation acts on (Mxy, Mz) as

      Mxy' = conj(a)² Mxy − b² conj(Mxy) + 2 conj(a) b Mz,     Mz' = (|a|² − |b|²) Mz − 2 Re(a b conj(Mxy)).

A sequence is therefore a product of rotations and of decay-with-precession steps. `core.bloch` does exactly this
for an ensemble of isochromats, with real 3 × 3 matrices, and averages them at each sample. It is slow and it is the
reference for everything else.

## 2. Extended phase graphs: why gradients shift states

A voxel contains spins at every position. A gradient lobe gives the spin at x the phase 2π n x, where x is the
position in units of the length over which the lobe dephases by one cycle and n the number of cycles. After any
number of pulses and lobes the magnetisation of the voxel is periodic in x, so it is a Fourier series:

    Mxy(x) = Σ_k F_k exp(2π i k x),      Mz(x) = Σ_k Z_k exp(2π i k x),      Z_−k = conj(Z_k).

The coefficients F_k and Z_k are the states of the extended phase graph. Each operator of section 1 becomes simple.

- A gradient of n cycles multiplies Mxy by exp(2π i n x): F_k moves to F_(k+n). Z does not move.
- Relaxation and off-resonance act on every position alike: F_k is multiplied by exp(−t / T2) exp(2π i Δf t), Z_k by
  exp(−t / T1), and Z_0 gains (1 − exp(−t / T1)) M0.
- A rotation acts at every position with the same (a, b). Since conj(Mxy) has the coefficients conj(F_−k), the
  rotation mixes exactly three numbers per order: F_k, conj(F_−k) and Z_k.
- The receiver sees the mean of Mxy over the voxel. For whole cycles that is F_0: an echo is a state that has
  come back to order 0. For a fraction of a cycle the mean is Σ_k F_k sinc(k).
- Free diffusion blurs the pattern along x. A blur multiplies a Fourier component by exp(−D (2π k / L)² t), L being
  the length of one cycle. While a lobe is being played the order changes from k to k + n; with s(t) the fraction
  of the lobe's area played so far, the transverse state decays by
  exp(−D (2π / L)² t (k² + 2 m1 k n + m2 n²)), m1 and m2 the time averages of s and s². A constant gradient has
  m1 = 1/2, m2 = 1/3 (Weigel et al. 2010); `blocks.lobe_shape` gives them for a trapezoid. The exponent is D times
  the b-value of the block for that state (`blocks.b_increment`), so the Stejskal-Tanner attenuation needs no
  separate formula.

`core.epg` steps a block list on these states for all tissues, slice positions and B1 scales at once.

## 3. What each block does

A block is a frozen record of numbers; a sequence is a tuple of blocks plus the length of one cycle of dephasing
(`blocks.sequence`). It can be compared, hashed and written to JSON (`blocks.to_data`, `from_data`, `digest`).

| Block | Isochromat | States |
|---|---|---|
| `rf(flip, phase)` | rotation about (cos p, sin p, 0) by flip × B1 scale | F_k' = conj(a)² F_k − b² conj(F_−k) + 2 conj(a) b Z_k; Z_k' = (\|a\|² − \|b\|²) Z_k − a b conj(F_−k) − conj(a b) F_k |
| `rf(..., selective=Selective(tbw))` | the rotation of the console's windowed sinc at the spin's position through the slice | the same with (a, b) of the column |
| `rf(ck=...)` (`rf_designed`) | the rotation of a pulse made with `mrgym.design`, per slice column | the same |
| `rf(..., crushed=True)` | the pulse between a perfect crusher pair: Mxy' = −b² conj(Mxy), Mz' = (\|a\|² − \|b\|²) Mz | F_k' = −b² conj(F_−k), Z_k' = (\|a\|² − \|b\|²) Z_k |
| `rf(energy_ut2_ms=E, offset_hz=Δ)` | two pools only: Mz of the bound pool × exp(−π ω1² τ g(Δ)), ω1² τ = (2πγ)² E | the same factor on every Z_k of the bound pool |
| `delay(ms)` | precession by Δf, decay with T2, recovery with T1; with two pools the exchange between them | F_k × exp(−t / T2) exp(2π i Δf t); Z_k × exp(−t / T1); Z_0 + (1 − exp(−t / T1)) M0; diffusion of every order |
| `gradient(cycles, ms, shape)` | phase 2π n x; with a duration also a delay | the delay, the diffusion factor of section 2, then F_k → F_(k+n) |
| `spoil()` | Mxy = 0 | F_k = 0 for every k |
| `crush()` | Mxy = 0 and Mz replaced by its mean over the voxel | F_k = 0, Z_k = 0 for k ≠ 0 |
| `mz(factor)` | Mz × factor (inversion of efficiency e: −e; saturation: 1 − e) | Z_k × factor |
| `adc(label, phase, static_ms)` | mean of Mxy over the voxel × exp(−i phase) | Σ_k F_k sinc(k) × exp(−i phase) × exp(−\|static\| / T2') |
| `repeat(body, times, ...)` | the body `times` times; the RF phase of repetition j grows by Δ j + φ j (j + 1) / 2; named tables give a flip angle or a phase per repetition | |

`spoil`, `crush`, `mz` and the crushed pulse are idealisations. They exist so that a block list can state the model
the engine serves today. A real spoiler or crusher is a `gradient`, and the difference is then computed, not assumed
(rows N5, N6 and N13).

T2' is the decay from a static spread of off-resonance inside the voxel. `adc(static_ms=t)` multiplies the sample by
exp(−|t| / T2'), t being the net time for which the sampled pathway has dephased (TE at a gradient echo, 0 at a spin
echo). This is exact when every pathway that reaches the sample has that static time. Otherwise give the spins an
ensemble of off-resonances (`operators.lorentzian_spread`, row X8).

Two pools are an option of the spins (`operators.spins(..., pool="mt")` or `pool="cx"`): free water and a bound pool
without transverse magnetisation that exchange longitudinal magnetisation (magnetisation transfer), or two pools
with transverse magnetisation and a frequency shift (chemical exchange, the Bloch-McConnell equations). The
exchange during a delay is the exact exponential of the 2 × 2 rate matrix.

## 4. The steady state is a fixed point

One period of a periodic sequence maps the state s to M s + c: every block is linear in the magnetisation, and
only the recovery of Mz adds a constant. The periodic state solves (1 − M) s = c. `core.periodic` assembles M as a
sparse matrix on the states of all columns and solves it directly; no repetition is stepped and convergence is not
a question. With conj(F_−k) as an unknown of its own every operator is linear over the complex numbers.

Two points need care.

- **RF spoiling.** With a quadratic phase φ j (j + 1) / 2 the pulses differ from one repetition to the next, so
  the state is not periodic in the laboratory frame. In a frame that turns with the pulse phase and is translated
  along the spoiler axis by φ / n of a cycle per repetition (n = net dephasing per repetition) it is. The change of
  frame after each repetition is: F_k × exp(−i (Δ + φ)) exp(i k φ / n), Z_k × exp(i k φ / n). This holds when every pulse
  of the repetition sits at zero accumulated dephasing and n ≠ 0; `blocks.compile_ops(frame=True)` refuses anything
  else. A linear phase cycle (Δ only) needs no condition. Row X3 compares the two frames. A sample taken while the
  repetition is dephased by C cycles keeps a phase j φ C / n in repetition j: its magnitude is periodic, its phase is
  not.
- **Truncation.** Between two pulses a state is either transverse for the whole interval or longitudinal, and
  only a transverse state changes its order. An interval of duration d with g cycles of dephasing therefore costs
  exp(−d / T2) per g orders, and orders above −ln(tol) T2 g / d are dropped (`operators.auto_kmax`). For one cycle
  per TR this is the rule of `scanner/epg.py`.

When a period ends with an ideal crusher the state at that point is Mz alone. The period is then stepped twice,
from Mz = 0 and from Mz = 1; Mz at the end and every sample are linear in the starting Mz, which gives the fixed
point and the samples (`method="reduced"`, used for echo trains and prepared sequences). States above half the
reach of the gradients cannot return to order 0 before the crusher and are not carried; nothing is lost by that
(row X5).

`periodic.approach` returns the transient: the samples of every period stepped from equilibrium, next to the
periodic ones.

## 5. Three evaluators of one block list

| Module | Method | Use |
|---|---|---|
| `core.bloch` | isochromat ensemble, real 3 × 3 rotations, optional ensemble of off-resonances, optional Bloch-Torrey integration on the grid of isochromats | the reference |
| `core.epg` | states stepped in time, all columns at once; diffusion; two pools | transients, single shots, echo trains |
| `core.periodic` | fixed point of one period | steady states |

They share the numbers of the operators (`core.operators`) and nothing else. Section 9, group "agreement",
compares them on seeded random block lists to 1e-10.

## 6. Sequences as data

`core.library` holds functions that return a block list from protocol-level parameters.

| Function | What it is | Model |
|---|---|---|
| `spgr` | RF-spoiled gradient echo | `spoiling="rf"`: one cycle of dephasing spread over the TR, quadratic RF phase, sample right after the pulse (the engine's `spgr`); `"ideal"`: ideal spoiler (the Ernst formula) |
| `afi` | dual-TR gradient echo | pulses on positions 0 and 1 of a raster of TR1, a table of flip angles |
| `epi` | one excitation per TR | ideal spoiler |
| `se`, `ir` | spin echo; with an inversion | crushed refocusing pulse and ideal spoiler (the engine's model), or real crushers |
| `bssfp` | balanced SSFP | sample at TR / 2, linear phase cycle; a selective pulse has half its rewinder before and half after |
| `fse`, `mese` | echo train; multi-echo spin echo | constant or variable refocusing angles, real or ideal crushers, ideal crusher at the end of the TR |
| `megre` | several gradient echoes | ideal spoiler |
| `spgr_lobe` | RF-spoiled gradient echo | a real spoiler lobe after the sample, any phase increment |
| `dess` | free induction decay and echo of one steady state | one cycle per TR |
| `mprage` | inversion, a train of small pulses, recovery | ideal spoiler after each pulse |
| `t2prep`, `satrec` | T2 preparation; saturation recovery | exact preparation pulses; a saturation efficiency |
| `dwse`, `steam` | Stejskal-Tanner spin echo; stimulated echo | trapezoid lobes with their shape moments; `diffusion_waveform` gives the same program to `design.gradients.b_value` |
| `mt_spgr` | gradient echo with a saturation pulse | two pools; the pulse acts on the bound pool only |

    from mrgym.core import blocks as B, library, operators, periodic, epg

    seq = library.fse(tr_ms=3000, esp_ms=8, etl=16, refocus_deg=150, selective=B.Selective(tbw=4))
    zeta, weights = operators.slice_columns(4.0)                       # positions through the slice
    spins = operators.grid([dict(t1_ms=838, t2_ms=75), dict(t1_ms=4300, t2_ms=2000)], zeta)
    echoes = periodic.steady_state(seq, spins).signal                   # [echo, column]
    per_tissue = (echoes.reshape(16, 2, -1) * weights).sum(-1)          # integral over the slice
    first_tr = epg.run(seq, spins).signal                               # the first TR from equilibrium

A new sequence is a new function of this kind. `library.served` maps a realised protocol of the console to the
block list of the sequence it names. `spgr`, `epi`, `se` and `ir` take `sat=(q, delay_ms)`: the fat-saturation pulse
of the console for one resonance line, an event of every repetition that multiplies the longitudinal magnetisation
by q that long before the excitation (section 9, rows Q16 to Q16d).

## 7. What the core leaves out

- **Geometry through the slice** beyond independent columns: a column is a position with its own rotation. There
  is no object structure along z and no neighbouring slice.
- **Motion of spins** other than free diffusion. No flow, no bulk motion. A flow block would move magnetisation
  between columns; it does not exist.
- **Relaxation during a pulse**, and the duration of a pulse in the timing. `mrgym.design.rotation.bloch` has
  relaxation during a pulse; its result is not a rotation and is not used here.
- **J coupling** and every other spin system with more than one line per pool.
- **Restricted or anisotropic diffusion.** One free, isotropic coefficient per tissue; the bound pool does not
  diffuse.
- **The direct effect of an off-resonance saturation pulse on free water.** `rf(offset_hz=...)` saturates the bound
  pool only.
- **A spectrally selective saturation pulse as a pulse.** The fat-saturation pulse of the console is one event,
  the longitudinal magnetisation of one resonance line times q (`sat=(q, delay_ms)`, rows Q16 to Q16d): the
  engine's model. Which share of the band a line receives, and what the pulse does to transverse magnetisation, are
  not computed here.
- **Images.** The core returns amplitudes; k-space, noise and reconstruction stay in `scanner/kspace.py`.
- **Agreement with a scanner.** Section 9 shows that the code computes the stated equations.

## 8. Conventions

- **Rotation.** Right-handed by the angle φ about the unit axis n: a = cos(φ/2) − i n_z sin(φ/2),
  b = −i (n_x + i n_y) sin(φ/2). A hard pulse of flip angle α and phase p from equilibrium gives
  Mxy = −i exp(i p) sin α (design.md, section 2).
- **Precession.** A positive frequency gains positive phase (conventions.md, section 5). A gradient block of +n
  cycles gives the spin at +x the phase +2π n x and moves F_k to F_(k+n).
- **Sample.** `adc` multiplies by exp(−i receiver phase) and by a demodulation factor, +i by default, so that
  the signal of a pulse about x is real and positive, as the engine reports it.
- **Units.** ms, degrees, Hz, µT, cycles; diffusion coefficient in µm²/ms; the length of a cycle in mm; b in s/mm².
- **Time.** A pulse takes no time. A block's duration is the duration of its free evolution.
- **Spin echoes.** Excitation about x, refocusing about y: the echo is positive.

Rows C1 to C3 compare these with `mrgym.design` and with `scanner/rf.py`.

## 9. Validation

`python -m mrgym.core.validate` runs the rows below and exits with status 1 if one fails;
`tests/test_core_validate.py` asserts that all pass and that this table is the one the command prints
(`--markdown`). Errors below 1e-3 are printed with one significant digit: they are rounding noise and differ between
machines. The references were written by the same authors as the code: the rows show that the code computes the
stated equations. They do not show that the equations describe a scanner. Where a closed form is attributed to a
paper that was not opened for this work, the row says so and names the independent computation that checks it.

<!-- validation table: begin -->
### The evaluators against each other, and the conventions

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| C1 | a hard pulse of flip angle a and phase p from equilibrium, then free precession at df (40 random cases; both evaluators; the Cayley-Klein parameters against `design.rotation.axis_angle`) | Mxy = -i exp(i p) sin a, then x exp(+2 pi i df t); the default demodulation +i makes a pulse about x real and positive | 7.4e-16 | ≤ 1.0e-12 | pass |
| C2 | slice columns: the calibration of the slice gradient (nominal bandwidths across the width at half maximum of the small-flip-angle profile, by bisection on the transform of the design layer's pulse), positions and weights of the integral through the slice; the rotation of the windowed sinc at every column (fast, normal, sharp; Hann, Hamming; 5 to 180 deg; excitation and the refocusing pulse on the 1.5 x slice), computed with `design.rotation.rotate` | `scanner/rf.py::fwhm_factor`, `quadrature` and `cayley_klein` (rows R1 to R3 and R13 of `mrgym validate` check that cascade and that calibration) | 1.0e-13 | ≤ 1.0e-12 | pass |
| C3 | 12 random block lists of `mrgym.design.sequence` (30 blocks: pulses, delays, gradients, ideal spoilers), converted with `blocks.from_design` | `design.sequence.run`, the time-stepping phase graph of the design layer (separate code) | 2.8e-16 | ≤ 1.0e-12 | pass |
| X1 | extended phase graph, 40 random block lists of 24 blocks x 5 columns (hard and crushed pulses, delays, gradients of -2 to 3 cycles, ideal spoilers and crushers, Mz preparations; random T1, T2, off-resonance, B1, M0): every sample | isochromat ensemble (`core.bloch`: real 3 x 3 rotations, as many isochromats as the gradients can reach orders + 1) | 9.9e-16 | ≤ 1.0e-10 | pass |
| X1b | the same lists: the final state read back as the magnetisation of every isochromat | the isochromats of the ensemble | 1.4e-15 | ≤ 1.0e-10 | pass |
| X2 | fractional dephasing (quarter cycles) on two gradient axes, 10 random lists: the final state as the magnetisation of 12 x 12 isochromats | the isochromat ensemble | 1.3e-15 | ≤ 1.0e-10 | pass |
| X2b | the same: the sample sum_k F_k sinc(k) against the mean over 24, 48, 96 isochromats per axis: order of convergence | second order in the spacing of the isochromats (midpoint rule) | 0.002667 in the order | ≤ 0.1 | pass |
| X2c | its error at 96 isochromats per axis | the mean over the ensemble | 8.3e-05 | ≤ 0.002 | pass |
| X3 | RF phase increments in the pulse frame (state turned and translated after every repetition): 20 random repeats of 25 repetitions with linear and quadratic increments, flip-angle tables, two pulses per repetition, gradients of both signs and a sample away from zero dephasing, inside a longer list: every sample and the final state | the same lists with the increments added to the pulse and receiver phases (laboratory frame) | 3.1e-13 | ≤ 1.0e-10 | pass |
| X4 | the periodic state as a fixed point (sparse linear solve on the phase-graph states, all columns at once): 24 random periods x 4 columns: balanced with a linear phase cycle, and unbalanced with quadratic RF spoiling; flip-angle tables of 1 to 3 repetitions, two pulses per repetition, 14 orders kept | the phase graph stepped through 260 periods from equilibrium (T1 25 to 60 ms, so that it has converged), same truncation | 7.8e-13 | ≤ 1.0e-10 | pass |
| X5 | a period that ends with an ideal crusher: the fixed point of Mz from two runs of the period that carry only the orders up to half its reach (`reduced`), 12 random lists of 10 blocks: samples and the state at the start of the period | the matrix method on all states the gradients can reach, and the phase graph stepped through 250 periods | 6.7e-16 | ≤ 1.0e-10 | pass |
| X6 | two pools, magnetisation transfer (bound pool saturated by every pulse and by off-resonance pulses, exchange during every delay): 8 random lists x 3 columns in the phase graph, and their periodic state by the matrix method | isochromat ensemble with `scipy.linalg.expm` of the exchange generator; the phase graph stepped through 300 periods | 4.1e-12 | ≤ 1.0e-10 | pass |
| X6b | two pools, chemical exchange (both pools transverse, a frequency shift between them): 8 random lists | the same two references | 2.3e-15 | ≤ 1.0e-10 | pass |
| X7 | free diffusion in the phase graph (attenuation of every state by its own order during every block; four pulses, gradients of both signs, D 1 and 3 um^2/ms, 60 um per cycle): order of convergence of the reference towards it, 8 to 64 substeps per block | Bloch-Torrey equation on the grid of isochromats: a Gaussian blur in the Fourier domain of the grid, the gradient advanced between half steps (second order); no closed-form factor is used | 0.002182 in the order | ≤ 0.1 | pass |
| X7b | its distance at 64 substeps | the same reference | 2.0e-05 | ≤ 5.0e-05 | pass |
| X7c | size of the effect in that test: largest change of a sample by diffusion | it must be large for the test to mean something | 0.5378 | ≥ 0.02 | pass |
| X8 | T2' of a gradient echo as the factor exp(-static time / T2') of the adc (T2' 30 ms, echoes at 12 and 24 ms) | an ensemble of 8,000 static off-resonances with a Lorentzian distribution (`operators.lorentzian_spread`); it converges as 1 / n | 0.001049 | ≤ 0.002 | pass |
| X8b | the same at a spin echo (static time 0) | the ensemble of 501 off-resonances: every isochromat is refocused | 2.2e-16 | ≤ 1.0e-12 | pass |
| X9 | every library sequence written to JSON-ready data and read back (17 sequences) | the same sequence: equal, same hash, same content digest | 0 | ≤ 0 | pass |

### The six served sequences against the engine

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| Q1 | spgr with ideal spoiling (the engine's tier 1 and `epi`), hard pulse: the grid of row S1 (3,456 values: 1 to 90 deg, TR 4 to 15,000 ms, T1 250 to 2,800 ms, B1 0.6 to 1.4) and the Ernst angle of row S4 | `scanner/signal.py::spgr` (the Ernst formula). Both sides are exact | 7.5e-14 relative | ≤ 1.0e-12 | pass |
| Q2 | spgr with RF spoiling and diffusion (tier 2), hard pulse, 252 steady states: the cases of rows S5 and S6, the 100 cases of row S7 (increment 0) and six tissues at 3 T x 3 flip angles x TR 8 and 30 ms x increments 117 and 50 deg x spoiler moments 60 and 300 mT ms/m | `scanner/epg.py::steady_state` at the same truncation (and for increment 0 the closed form of Hanicke & Vogel 2003). Both sides solve the same linear system | 1.9e-13 relative | ≤ 1.0e-09 | pass |
| Q2b | the limits of row S8: T2 << TR, TR >> T2, complete diffusion spoiling | the Ernst formula | 2.9e-14 relative | ≤ 1.0e-06 | pass |
| Q2c | row S9: distance from the Ernst formula at TR 5 ms, 30 deg, T1 1500 ms, T2 200 ms | incomplete spoiling: at least 3 % (the engine's row gives the same number) | 0.07756 relative | ≥ 0.03 | pass |
| Q3 | spgr through the slice (the cases of row R8, with diffusion): one hard pulse of the column's flip angle per column, weighted with the transverse phase of the column (`Selective(model='flip')`, the engine's model) | the engine's sum over the same columns with `epg.steady_state` at every column. Both sides are exact for that model | 2.2e-14 relative | ≤ 1.0e-09 | pass |
| Q3b | the same cases as the engine SERVES them (`kspace.slice_signal`) | the engine interpolates the phase graph over the flip angle on 16 Chebyshev nodes (its row R8 allows 5e-4); the core solves every column: the core is the exact side | 2.1e-04 relative | ≤ 5.0e-04 | pass |
| Q3c | the cases of row R7 with the ROTATION of every column (Cayley-Klein parameters of the selective pulse, `Selective(model='rotation')`) | the engine's separately written isochromat ensemble through the slice (`scanner/bloch.py::slice_spgr_isochromats`, 3 x 3 cascades). The rotation model is the exact one; the ensemble has 192 spoiler phases and twice the columns | 8.8e-06 relative | ≤ 2.0e-04 | pass |
| Q3d | the engine's flip model against the rotation model in those cases | equal: the part of a column's rotation that turns about z is the same in every repetition, and with a net spoiler moment per TR a constant turn about z is a translation along the spoiler axis. The flip model is exact for spgr and afi (it is 9e-06 from the ensemble) | 3.8e-15 relative | ≤ 1.0e-12 | pass |
| Q4 | spgr as `Engine.amplitudes` serves it on a phantom world with a strong B1 pattern, 3 voxels of each of 7 materials: hard pulse (the protocol of row S15) and the default selective pulse (row R10) | the engine interpolates over the flip angle (5 to 16 Chebyshev nodes; its rows allow 1e-4); the core solves every voxel and column: the core is the exact side for the engine's model | 1.3e-07 relative | ≤ 1.0e-04 | pass |
| Q5 | afi with ideal spoiling, hard pulse: the grid of row S2 (12,960 values) | `scanner/signal.py::afi` (Yarnykh 2007). Both sides are exact | 1.8e-13 relative | ≤ 1.0e-12 | pass |
| Q5b | afi with RF spoiling and diffusion: the dual-TR cases of rows S5 and S6 and six tissues x 2 flip angles x TR1 10 and 20 ms x TR2 / TR1 2 and 5 (a table of flip angles on a raster of TR1: pulse, pulse, n - 1 empty positions) | `scanner/epg.py::steady_state` with its pulse pattern. Both sides are exact | 1.5e-15 relative | ≤ 1.0e-09 | pass |
| Q5c | afi as `Engine.amplitudes` serves it (the protocol of row S15, hard pulse) | the engine interpolates over the flip angle; the core is the exact side | 2.9e-09 relative | ≤ 1.0e-04 | pass |
| Q6 | epi (one excitation per TR, ideal spoiler): 3 flip angles x 3 TR x the T1 and B1 grid; and as `Engine.amplitudes` serves it, hard and selective, fat saturation off | the Ernst formula; the engine. Exact on both sides for the hard pulse; the selective pulse is interpolated by the engine over the B1 range | 4.3e-15 relative | ≤ 1.0e-06 | pass |
| Q7 | se, hard pulse: 400 random (T1 200 to 4,500 ms, pulses at 60 to 120 % of 90 and 180 deg) x 4 (TR, TE) (row S10); the refocusing efficiency with REAL crushers, one echo from fresh magnetisation (row S12) | `scanner/signal.py::se` x exp(-TE / T2) (the engine applies the decay outside the amplitude); sin^2(beta / 2). Both sides are exact for the crushed model | 2.2e-15 relative | ≤ 1.0e-12 | pass |
| Q8 | ir, hard pulse: the same 400 voxels x 4 (TR, TE, TI, inversion efficiency 0.8 to 1) (row S11; absolute, the signal crosses zero) | `scanner/signal.py::ir_se` x exp(-TE / T2). Both sides are exact | 2.8e-16 | ≤ 1.0e-12 | pass |
| Q8b | the null of ir at TR 60 s for T1 400, 1000, 2500 ms (row S13): distance of the zero crossing from T1 ln 2 | TI = T1 ln 2 for TR >> T1 (at T1 2500 ms, exp(-TR / T1) = 4e-11 is still there) | 5.5e-11 relative | ≤ 1.0e-09 | pass |
| Q9 | se and ir through the slice with the rotation of every column, refocusing on the 1.5 x slice (the cases of row R9, B1 80 to 115 %) | `scanner/kspace.py::slice_signal`. Both sides are exact for the crushed model: without transverse magnetisation before a pulse, flip model and rotation model coincide | 2.0e-14 relative | ≤ 1.0e-12 | pass |
| Q9b | se and ir as `Engine.amplitudes` serves them: selective (rows R10) and hard (the protocol of row R6) | the engine interpolates the slice integral over the B1 range (5 to 16 nodes); the core is the exact side | 1.2e-09 relative | ≤ 1.0e-06 | pass |
| Q10 | bssfp, hard pulse: 200 random (T1, T2, TR, flip angle, off-resonance over three bands) of row S14 with the 180 deg cycle, and phase cycles of 0, 90, 137, 270 deg | `scanner/signal.py::bssfp` at the precession per TR seen from the pulse, 2 pi df TR - `phase_cycle_deg`, x exp(i pi df TR) x exp(-TR / 2 T2): the RF phase ADVANCES by `phase_cycle_deg` per TR, as the engine counts it since 0.3.0. Both sides are exact | 1.4e-12 relative | ≤ 1.0e-10 | pass |
| Q10b | the sign of the phase cycle in what the engine serves: `Engine.amplitudes` of a nonselective bssfp with phase cycles of 90, 137, 270 deg on a phantom world with a field (6 voxels of each of 7 materials) | `library.served` with the RF phase advancing by `phase_cycle_deg` per TR (conventions.md, section 5). Engine 0.2.0 put the bands at the opposite off-resonance; this row found it, and with 0.3.0 the two agree at the same off-resonance | 1.6e-14 relative | ≤ 1.0e-06 | pass |
| Q11 | bssfp through the slice, flip model: 3 cases off resonance; and as `Engine.amplitudes` serves it (row R10) | the engine's sum of its closed form over the columns (exact on both sides); the served value interpolates sin and cos of the flip angle over the B1 range (the core is the exact side) | 1.7e-14 relative | ≤ 1.0e-06 | pass |
| Q11b | the engine's idealisation for bssfp in those 3 cases: flip model against the rotation of every column with a balanced slice gradient (half the rewinding area before the pulse, half after) | in a balanced sequence the transverse magnetisation survives, and the rotation of an off-centre column has a component about z that acts as an off-resonance of that column. Size of the difference in the slice integral (the rotation model is the exact one) | 0.01183 relative | ≤ 0.3 | pass |
| Q12 | inversion recovery with one gradient-echo pulse per inversion: the grid of row S3 (inversion efficiency 0.8 to 1) | `scanner/signal.py::ir_gre`. Both sides are exact | 4.4e-16 | ≤ 1.0e-12 | pass |
| Q13 | the quadrature through the slice (6 x tbw points per thickness out to 0.5 + 12 / tbw thicknesses), ideal spoiling, TR 20 and 2,000 ms: a subset of row R4 (3 pulses x 4 flip angles x 3 T1) | the core on 8,001 columns over +- 12 thicknesses (the bound of row R4) | 8.5e-05 relative | ≤ 2.0e-04 | pass |
| Q13b | the same at TR 5 ms (row R5) | the same reference (the bound of row R5) | 6.1e-04 relative | ≤ 0.001 | pass |
| Q14 | signal left by a field gradient through the slice (0.25 to 1.7 cycles across the thickness at the echo, 2 deg): the columns given their own off-resonance in the block list (row R11) | `scanner/rf.py::dephasing_factor`: sinc(cycles) x window(cycles / tbw), the small-tip form the engine applies (the bound of row R11) | 0.001154 | ≤ 0.002 | pass |
| Q16 | a saturation pulse in every repetition (Mz x q, 4.5 to 12 ms before the excitation; q 0 to 1), hard pulse, ideal spoiling: spgr and epi (144 cases x 4 T1), se (36 x 4), ir (24 x 4; absolute, the signal crosses zero) | `scanner/signal.py::spgr`, `se`, `ir_se` with `sat` (the closed form (beta0 + q beta1) / (1 - q alpha), engine 0.3.0). Both sides are exact. A fully saturated line of T1 343 ms (fat at 1.5 T), the pulse 7 ms before the excitation, keeps 0.024, 0.131, 0.023, 0.020 of its signal at TR / flip angle 30 ms / 10 deg, 30 / 60, 500 / 60, 4000 / 90: not one factor | 8.1e-15 relative | ≤ 1.0e-12 | pass |
| Q16b | the same pulse in the RF-spoiled steady state of spgr, acting on the longitudinal states of every order (6 cases, 0 to 100 % saturated, no diffusion) | `scanner/epg.py::spgr` with `sat`, the engine's phase graph (separate code; with diffusion the two models differ in where the spoiler plays and are not compared) | 2.2e-15 relative | ≤ 1.0e-10 | pass |
| Q16c | the same through the slice with the rotation of every column: spgr and epi (ideal spoiling), se, ir; fast, normal and sharp pulses (6 cases) | `scanner/kspace.py::slice_signal` with `sat`. Both sides are exact | 1.4e-14 relative | ≤ 1.0e-12 | pass |
| Q16d | fat saturation as the engine SERVES it, line by line (`Engine._species`): spgr (hard and selective, RF spoiled), se, ir, epi on an abdomen world at 1.5 T, 2 voxels of each of 3 fat tissues x 6 lines; the lines keep 5 to 100 % of their longitudinal magnetisation at the pulse | `library.served` with the saturation of each line as an event (the share of the band a line receives is the engine's, `kspace.sat_band`). The engine interpolates over the flip angle and over q; the core solves every line, voxel and column. For the RF-spoiled spgr the core plays the spoiler before the saturation pulse and the engine spreads it over the whole TR: with the diffusion of fat the two models differ by the size printed here; se, ir and epi agree to the engine's interpolation (1e-8) | 1.2e-05 relative | ≤ 1.0e-04 | pass |
| Q15 | two-angle variable-flip-angle T1 on a slice with the nominal angles in the fit (row R12): the two slice signals | `kspace.slice_signal` with ideal spoiling; the bias they give is -37 to -76 % (published: 40 to 80 % underestimation, Svedin & Parker 2017) | 2.9e-14 relative | ≤ 1.0e-12 | pass |

### Sequences beyond the six

| # | Quantity | Closed form or independent method | Largest error | Tolerance | |
|---|---|---|---|---|---|
| N1 | fse / CPMG train: 16 echoes with 180 deg pulses; echoes 1 and 2 with 150, 120, 60 deg | exp(-n ESP / T2); sin^2(b/2) E2 and sin^4(b/2) E2^2 + (1/2) sin^2(b) E2 E1 (spin echo and stimulated echo; Hennig 1988) | 2.2e-16 | ≤ 1.0e-12 | pass |
| N1b | 14 echoes of 6 trains with variable refocusing angles (40 to 180 deg), imperfect excitation, random T1, T2, off-resonance and B1 | isochromat ensemble (the recursion that the phase graph replaces) | 8.9e-16 | ≤ 1.0e-10 | pass |
| N1c | fse with an 80 deg excitation and 8 refocusing angles from 170 to 90 deg, repeated every 900 ms (T1 600 to 2,500 ms): the periodic state from two runs of one period | the matrix method; the phase graph stepped through 40 TR; the isochromat ensemble stepped through 14 TR | 3.3e-16 | ≤ 1.0e-10 | pass |
| N2 | pseudo steady state of long trains without relaxation (60, 120, 150 deg; 400 echoes, mean of the last 100; 120 orders kept): equal pulses, and a first pulse of 90 + a/2 deg | sin(a/2), and 2 sin(a/2) / (1 + sin(a/2)) with the first pulse (derived for the design layer, rows E4b, E4c; not taken from a paper) | 4.2e-05 | ≤ 5.0e-04 | pass |
| N2b | those two closed forms | re-derived here: each isochromat keeps the component of its first-echo magnetisation along the rotation axis of its echo period, the unit vector along (sin(a/2), 0, cos(a/2) sin p) (from the quaternion of Rz(p) Rx(a) Rz(p)); mean over 4,096 dephasing angles p | 1.1e-16 | ≤ 1.0e-09 | pass |
| N3 | mese, 16 echoes, 150 deg: with ideal crushers (spin-echo pathway only) the echoes and the fitted T2; with 180 deg the fitted T2 | (sin^2(b/2) E2)^n, so 1 / T2fit = 1 / T2 - ln(sin^2(b/2)) / ESP; T2 itself | 1.8e-15 | ≤ 1.0e-10 | pass |
| N3b | mese with real crushers, 150 deg: bias of a mono-exponential T2 fit over all 16 echoes (T2 40, 80, 300 ms; without the first echo the bias is still 4.0 %) | stimulated echoes: the first echo is low and later echoes are fed from Mz, which decays with T1, so T2 is overestimated by several per cent and dropping the first echo does not remove it: the train has to be fitted with its own model. The echoes themselves equal the isochromat ensemble (row N1b) | 0.0473 relative | ≥ 0.03 | pass |
| N4 | megre, 4 echoes: the periodic state with T2' and off-resonance | Ernst formula x exp(-TE / T2*) x exp(2 pi i df TE), 1 / T2* = 1 / T2 + 1 / T2' (the engine's `signal.spgr` and decay model) | 2.6e-16 | ≤ 1.0e-12 | pass |
| N4b | megre without T2': the same echoes | isochromat ensemble stepped through 744 TR from equilibrium | 3.7e-16 | ≤ 1.0e-10 | pass |
| N5 | spgr_lobe (pulse, TE, sample, a spoiler lobe of 2 ms, rest of TR; no diffusion): 3 cases | `scanner/epg.py::steady_state` x exp(-TE / T2): without diffusion only the dephasing per TR matters, not when it is played | 5.5e-16 | ≤ 1.0e-09 | pass |
| N5b | the same 3 cases | the engine's separately written isochromat ensemble (`scanner/bloch.py::rf_spoiled_isochromats`, 1,024 spins, time stepping) x exp(-TE / T2) | 3.2e-11 | ≤ 1.0e-09 | pass |
| N5c | increment 0 with a spoiler of 2 cycles | tan(a/2) (1 - (E1 - cos a) r) x exp(-TE / T2) (Hanicke & Vogel 2003): the gradient-spoiled steady state does not depend on the number of whole cycles | 7.2e-16 | ≤ 1.0e-09 | pass |
| N6 | RF spoiling against the increment (30 deg, TR 10 ms, T1 1000 ms, T2 80 ms): signal / Ernst is 2.78 at 0 deg, 1.011 at 117 deg, 1.98 at 180 deg; measured: the distance at 117 deg | 117 deg is close to the Ernst signal (Zur 1991; Preibisch & Deichmann 2009) | 0.01091 relative | ≤ 0.05 | pass |
| N6b | the same: the smaller distance from Ernst of the increments 0 and 180 deg | increments of 0 and 180 deg do not spoil: the signal is far from the Ernst value | 0.981 relative | ≥ 0.2 | pass |
| N6c | the same three increments with strong diffusion across the spoiler (D 2.5 um^2/ms, 0.2 um per cycle) | diffusion removes every coherence between repetitions: the Ernst signal whatever the increment | 9.8e-15 relative | ≤ 1.0e-06 | pass |
| N7 | dess, 3 cases: the free induction decay right after the pulse and the echo right before the next pulse | tan(a/2) (1 - (E1 - cos a) r) and -tan(a/2) (1 - (1 - E1 cos a) r), r = sqrt((1 - E2^2) / ((1 - E1 cos a)^2 - E2^2 (E1 - cos a)^2)) (Hanicke & Vogel 2003; paper not opened for the second: this computation finds that it is the echo at the moment of the next pulse, with the opposite sign of the free induction decay, and row N7b confirms the number) | 4.6e-16 | ≤ 1.0e-10 | pass |
| N7b | the same two samples | isochromat ensemble of 128 spins stepped through 12 T1 from equilibrium (states above order 128 alias; they have decayed) | 9.3e-09 | ≤ 1.0e-06 | pass |
| N8 | mprage (inversion of efficiency 0.9 to 1, 8 to 128 pulses, recovery): every line of the train in the periodic state | Look-Locker recursion in closed form: Mz_i = Mss + (Mz_0 - Mss)(E cos a)^i, Mss = (1 - E) / (1 - E cos a), closed over the cycle (derived here) | 1.9e-16 | ≤ 1.0e-12 | pass |
| N8b | its one-pulse limit; and 8 pulses for T1 120 and 200 ms | `scanner/signal.py::ir_gre`; isochromat ensemble stepped through 60 cycles | 1.1e-16 | ≤ 1.0e-12 | pass |
| N9 | t2prep (90 - 180 x 1, 2, 4 - 90 back, exact pulses, 90 deg readout): the periodic signal | (1 - exp(-(TR - TEprep) / T1)) exp(-TEprep / T2): Mz that regrows during the preparation ends transverse and is spoiled | 2.2e-16 | ≤ 1.0e-12 | pass |
| N9b | t2prep with B1 at 70 to 130 %, off-resonance up to 30 Hz, a 70 deg readout | isochromat ensemble stepped through 40 TR | 4.4e-16 | ≤ 1.0e-10 | pass |
| N10 | satrec, saturation efficiency 0.6 to 1: the periodic signal | sin a (1 - exp(-TS / T1)) for a complete saturation, whatever TR and the flip angle; the closed fixed point of the two-interval recursion otherwise (derived here) | 5.6e-17 | ≤ 1.0e-12 | pass |
| N10b | the same sequences at T1 / 8 | isochromat ensemble stepped through 60 TR | 2.8e-17 | ≤ 1.0e-12 | pass |
| N11 | dwse, 4 gradient programs (rectangular, trapezoid, triangle; b 41 to 1551 s/mm^2) x 3 tissues: the echo | exp(-TE / T2) exp(-b D) with b from `mrgym.design.gradients.b_value` of the same program (Stejskal & Tanner 1965) | 2.5e-16 | ≤ 1.0e-08 | pass |
| N11b | the b-value of those programs: `gradients.b_value`, and the sum of `blocks.b_increment` over the blocks | gamma^2 G^2 (delta^2 (Delta - delta / 3) + eps^3 / 30 - delta eps^2 / 6), the trapezoid form (textbook form, source not opened; `gradients.b_value` is the exact integral of the waveform and agrees with it) | 8.9e-16 relative | ≤ 1.0e-07 | pass |
| N11c | the first program against the Bloch-Torrey grid integration at 64 substeps per block (order 2.00) | no closed-form diffusion factor is used by the reference | 9.2e-06 | ≤ 1.0e-05 | pass |
| N12 | steam with diffusion lobes, 3 sets of flip angles, TM 100 to 900 ms, 3 tissues | (1/2) sin a1 sin a2 sin a3 exp(-TE / T2) exp(-TM / T1) exp(-b D), b from `gradients.b_value` (the grating is stored along z during TM and keeps diffusing) | 1.1e-16 | ≤ 1.0e-08 | pass |
| N12b | the same sequences without diffusion | isochromat ensemble on two gradient axes | 3.3e-16 | ≤ 1.0e-12 | pass |
| N13 | se and ir through the slice with REAL crushers and an ideal spoiler after the echo, so that the grating stored along z by an imperfect refocusing pulse comes back as a stimulated echo in the next TR (5 cases, B1 80 to 115 %): change against the served model, -2.40 to +38.70 % | docs/physics/limits.md, block 8a: -2 to +39 %, from a third, separately written integrator (`tools/external/independent_bloch.py`, row L1.4); measured: distance of the rounded ends of the range | 0 points | ≤ 0 | pass |
| M1 | absorption lines of the bound pool (Lorentzian, Gaussian, super-Lorentzian): integral over the angular frequency | 1 (the formulas are attributed to Morrison & Henkelman 1995, paper not opened; this checks their normalisation only) | 7.1e-05 | ≤ 0.002 | pass |
| M2 | two pools under continuous off-resonance irradiation (1 to 10 kHz, 3 to 10 uT), as a train of saturation pulses every 4, 2, 1 ms extrapolated to zero spacing (at 1 ms the distance is 3e-05): Mz of free water in the steady state | (Rb kf / Ra + W + Rb + kr) / (kf / Ra (Rb + W) + W + Rb + kr), the closed form of Henkelman et al. 1993 without its direct-saturation term (paper not opened; it equals the linear solve of the two rate equations, which is in this row) | 1.0e-08 | ≤ 1.0e-07 | pass |
| M2b | the same: order of convergence with the spacing of the pulses, 4 to 1 ms | second order (the saturation is split symmetrically around the delay) | 6.3e-04 in the order | ≤ 0.1 | pass |
| M3 | mt_spgr (saturation pulse, 2 ms, excitation of 6, 15, 30 deg, TR 30 ms; bound fraction 13.9 %): the periodic signal | fixed point of the 2 x 2 chain [saturate bound pool, exchange and recover, excite free pool, exchange and recover] with `scipy.linalg.expm` (the matrix form used for pulsed MT, e.g. Yarnykh 2012; written here independently) | 3.7e-16 | ≤ 1.0e-12 | pass |
| M3b | the same at T1 / 20; and with a vanishing bound pool | isochromat ensemble stepped through 120 TR; the Ernst signal | 1.4e-12 | ≤ 1.0e-08 | pass |
| M4 | MT saturation delta from the signals with and without the pulse at 6, 15, 30 deg: 7.52 to 8.44 %; measured: its spread over the flip angles | delta = (A a / S - 1) R1 TR - a^2 / 2 is nearly independent of the flip angle (Helms et al. 2008, as restated by Tabelow et al. 2019); an approximation, so a band, not an equality | 0.1174 relative | ≤ 0.2 | pass |
<!-- validation table: end -->

### What the comparison with the engine showed

- For the hard pulse, and for `se` and `ir` through the slice, block list and engine agree to rounding.
- Where the engine interpolates over the flip angle (`spgr`, `afi` and every selective pulse as served by
  `Engine.amplitudes`), it differs from the core by what rows Q3b, Q4, Q5c and Q9b measure, inside the bounds of the
  engine's own interpolation rows. The core solves every column, so it is the exact side for the same model.
- For `spgr` and `afi` the engine treats a slice column as a hard pulse of the column's flip angle and uses the
  column's transverse phase as a weight. This equals the full rotation of the column (row Q3d): the part of the
  rotation about z repeats in every TR, and with a net spoiler moment a constant turn about z is a translation.
- For `bssfp` through the slice the same treatment is an approximation. With a balanced slice gradient the
  rotation of an off-centre column has a component about z, which shifts the off-resonance response of that
  column. Row Q11b measures the change of the slice integral in three cases. The engine keeps this simplification
  and declares it ([limits.md](limits.md), blocks 0 and 11; its own row R15 measures it over the whole band).
- `phase_cycle_deg` of `bssfp`. Engine 0.2.0 added the cycle to the precession per TR, which is the steady state of
  an RF phase that decreases by the cycle per TR, while conventions.md calls it the RF phase advance; for values
  other than 0 and 180 degrees the bands sat on the other side. The comparison of this page found that. Engine 0.3.0
  subtracts the cycle, and `library.served` now passes `phase_cycle_deg` on as the advance it is: rows Q10 and Q10b
  compare at the same off-resonance. The side that moved is the core's block list, which had followed the engine.
- The slice. Engine 0.3.0 calibrates the slice gradient of every pulse so that the profile of a small flip angle is
  one slice thickness wide at half its maximum; 0.2.0 let the gradient span the nominal bandwidth of the pulse.
  The core's `Selective` block and `slice_columns` follow (`operators.slice_calibration`, computed here from the
  pulse of the design layer; row C2 holds it equal to the engine's number). The side that moved is the core.
- Fat saturation. Engine 0.3.0 plays the saturation pulse in every repetition and saturates each line of fat by its
  place in the band. As a block list that is one event, Mz x q, a delay before the excitation, per line
  (`library.served(..., sat=(q, delay))`): rows Q16 to Q16d reproduce the engine's closed forms, its phase graph and
  the amplitude maps it serves line by line. The share of the band a line receives is taken from the engine; the
  transverse effect of the pulse is in neither.

## 10. Speed

`python -m mrgym.core.validate --speed` prints this table: the amplitudes of 6 tissues × 85 slice columns (510
columns), one process, one thread, best of two runs. The numbers are from the machine the table was made on and
change with the machine. "Stepped" starts from equilibrium; the last column is the distance of its last samples from
the periodic ones relative to the largest sample (a stepped run of a long-T1 tissue has not arrived).

| Sequence | Columns | Repetitions stepped | Stepped (s) | Periodic solve (s) | Stepped value against the periodic one |
|---|---:|---:|---:|---:|---:|
| spgr, RF-spoiled, TR 10 ms; stepping keeps all 500 orders | 510 | 500 | 12.130 | 0.924 | 3e-02 |
| the same; stepping keeps 64 orders | 510 | 500 | 1.200 | 0.951 | 3e-02 |
| spgr, ideal spoiling | 510 | 500 | 0.040 | 0.003 | 3e-02 |
| afi, TR 20 + 100 ms | 510 | 600 | 1.566 | 3.499 | 2e-03 |
| epi, TR 2000 ms | 510 | 10 | 0.003 | 0.003 | 4e-04 |
| se, TR 500 ms | 510 | 10 | 0.007 | 0.006 | 4e-04 |
| ir, TR 2000 ms | 510 | 10 | 0.007 | 0.006 | 3e-05 |
| bssfp, TR 5 ms | 510 | 1000 | 0.103 | 0.003 | 7e-02 |
| fse, 16 echoes | 510 | 4 | 0.043 | 0.021 | 1e-03 |
| fse, 32 echoes | 510 | 4 | 0.151 | 0.045 | 9e-04 |
| fse, 64 echoes | 510 | 4 | 0.797 | 0.147 | 9e-04 |
| fse, 128 echoes | 510 | 4 | 3.799 | 0.761 | 8e-04 |
| fse, 128 echoes, 32 orders kept | 510 | 4 | 0.501 | 0.239 | 8e-04 |
| mese, 16 echoes | 510 | 4 | 0.040 | 0.021 | 3e-03 |
| megre, 4 echoes | 510 | 200 | 0.043 | 0.005 | 1e-02 |
| spgr_lobe, TR 10 ms | 510 | 500 | 1.087 | 1.263 | 1e-01 |
| dess, TR 15 ms | 510 | 400 | 0.579 | 0.849 | 4e-02 |
| mprage, 128 pulses | 510 | 4 | 0.035 | 0.019 | 1e-03 |
| t2prep | 510 | 10 | 0.002 | 0.001 | 1e-16 |
| satrec | 510 | 10 | 0.001 | 0.002 | 0e+00 |
| dwse | 510 | 4 | 0.002 | 0.001 | 9e-17 |
| steam | 510 | 4 | 0.005 | 0.002 | 1e-16 |
| mt_spgr, two pools | 510 | 300 | 0.122 | 0.021 | 5e-11 |
| spgr, RF-spoiled, on 16 flip angles per tissue (hard pulse) | 96 | 500 | 0.256 | 0.165 | 3e-02 |
| afi on 16 flip angles per tissue (hard pulse) | 96 | 600 | 0.316 | 0.734 | 2e-03 |
| for scale, the engine today: `kspace.slice_signal`, spgr tier 2, the same 6 tissues (16 flip angles per tissue, first call) | 96 | – | – | 0.217 | – |

| Echoes | Refocusing angle (deg) | Orders kept | Change of the slice-integrated echoes, relative to the first echo |
|---:|---:|---:|---:|
| 64 | 150 | 16 | 4e-03 |
| 64 | 150 | 32 | 2e-03 |
| 64 | 150 | 64 | 0e+00 |
| 128 | 150 | 16 | 2e-03 |
| 128 | 150 | 32 | 3e-03 |
| 128 | 150 | 64 | 2e-03 |
| 128 | 120 | 16 | 7e-03 |
| 128 | 120 | 32 | 5e-03 |
| 128 | 120 | 64 | 3e-03 |

Reading it against a budget of about a second per scan, for the amplitudes alone:

- Everything without a gradient between its pulses (`spgr` with ideal spoiling, `epi`, `se`, `ir`, `bssfp`, `megre`,
  `mprage`, `t2prep`, `satrec`, `dwse`, `steam`, `mt_spgr`) takes a few hundredths of a second.
- Echo trains: the cost grows roughly with the square of the echo-train length, because the number of states
  grows with it. 16, 32 and 64 echoes fit easily; 128 echoes take most of a second for 510 columns. Dropping orders
  is not a remedy to rely on: the second table gives what it changes.
- RF-spoiled sequences with a spoiler (`spgr`, `spgr_lobe`, `dess`) take about a second for 510 columns and `afi`
  about three. For these the engine's shortcut is exact (previous section): solve a few flip angles per tissue and
  interpolate. On 16 flip angles per tissue the core takes about as long as the engine does today.
- Stepping a short-TR gradient echo to its steady state does not fit: about ten seconds for 500 TR with every
  order kept, and after 500 TR the long-T1 tissues have not arrived. The periodic solve is what makes these
  sequences affordable.

The cost of forming an image from amplitudes that differ from line to line: section 11.

## 11. How the console uses the core

**One registry.** `scanner/console.py::REGISTRY` holds one entry per sequence (`console.Sequence`): the parameters
with their ranges, the timing rule (a function of the plan context that returns echo and repetition times, the RF
pulses of a cycle for the RF power, and the cycles per average), the sampling plan (`console.Sampling`: which image
is made from which sample of the cycle, which sample fills which phase-encode line), where the amplitude comes from
(`serves`: "closed form" or "core"), and the block list of a realised protocol (`Sequence.blocks`, which is
`core.library.served`: one source for what is served and for what is compared). `console.plan` is three stages
(`_prescribe`, the entry's timing rule, `_conclude`); `Engine.clean_kspace` forms the k-space of a scan by the
entry (`kspace.FORMATION`).

**The six stay where they were.** Their entries say `serves = "closed form"`: the code path and every byte of a scan
are those of engine 0.3.0 (`tests/test_compat_2d_unchanged.py`, the golden digests and the id fingerprints are
untouched). Their block lists run in shadow: rows J1 to J6 of `mrgym validate --group core` solve the registry's
block list of each with `core.periodic` and compare with `Engine.amplitudes`.

**A console offers what its card lists.** `console.offered(exam)`: the six, and the sequences named by
`exam["sequences"]` (world spec `console_sequences`, a list or "all"). A card without the key is the console of
before: its schema, its refusals and its scanner card do not change, so no environment id moves. The tool layer of
the environments (`envs/scanner_env.py`: help topics, examples) does not know the four yet; `Engine.scan`, the
session (`ScannerSession`, log and replay) and the console do. Their manual is the supplement
[scanner-core.md](../scanner-core.md) (`scanner/datasheet_core.md`).

**Amplitudes** (`scanner/served.py::table`), per tissue class on Chebyshev nodes of the transmit scale:

| Sequence | Block list | How it is solved | Check |
|---|---|---|---|
| `fse`, `mese` | `library.fse` / `mese`: real crushers of one cycle either side of every refocusing pulse, an ideal crusher at the end of the repetition | reduced exactly to the refocusing flip angle (below): three single shots of `core.epg` on (echoes + 3) Chebyshev points of its cosine | J7 to J11, J19 to J24 |
| `dwse` | `library.dwse`: the lobes as gradient blocks with their shape moments; b = 0: `library.se` | `core.periodic` on every slice column | J31 to J36, J15 |
| `megre` | `library.spgr` (the steady state of the engine's `spgr`; the echoes are read by the readout stage) | `core.periodic`: ideal spoiling in every slice column, the departure from it on 16 nodes of the flip angle | J25 to J30 |

**An echo train reduces to one angle.** In a train with equal delays and equal crushers either side of every
refocusing pulse, a refocusing rotation (a, b) acts on the sampled echoes as the hard pulse of flip angle
θ = 2 asin|b| about the transverse axis of phase arg(b) + 90°: the phase of a is a turn about z that is the same
before and after the pulse, a precession, and every pathway that is sampled at order 0 has refocused it. The states
split into two families that the pulses never mix, since a pulse couples F_k, conj(F_−k) and Z_k of one |k| and a
crusher moves every F by one: the family that starts from the transverse magnetisation after the excitation (odd
orders at the pulses: the echoes and the gratings Z_odd) and the family that starts from the longitudinal
magnetisation (even orders at the pulses: free induction decays that are never at order 0 when an echo is sampled,
and Mz itself). With m0 and c0 the transverse and the longitudinal magnetisation right after the excitation per unit
of Mz before it, in the frame in which the refocusing axis is y,

    echo_n = Mz (u A_n(θ) + v B_n(θ)),  u = −Im m0, v = Re m0;      Mz of the next repetition = c0 P(θ) Mz + R(θ),

and A_n, B_n, P, R are polynomials of degree ≤ n in cos θ. Three single shots (excitation about x, about y, none) on
n + 3 Chebyshev points therefore give them without interpolation error, the periodic Mz is R / (1 − c0 P), and the
echoes of every slice column and transmit scale follow from the two rotations of that column. Row J8 compares with
the periodic solve of every column (1.8e-15), row J9 with the isochromat ensemble, row J20 with the phase graph of
the design layer. An independent review re-derived the statement and confirmed it with its own isochromat ensemble
for general rotations (3e-15); it does not hold when the refocusing rotation changes along the train, and such a
train (one angle per echo: the seam for the design layer) is solved on every column instead. What the reduction
bought: the integral through the slice and the dependence on the transmit scale oscillate with the echo number; on
the engine's own slice columns a train of 32 echoes was off by 6e-3, and 9 transmit nodes by 5e-4. With the
reduction 4 times the columns and up to 72 nodes cost nothing (rows J10, J11).

**Image formation with an amplitude that changes from line to line** (`served.form_lines`). With a(c, j, s) the
table (class c, node j, sample s), s(p) the sample that fills line p, g_c(x) the evolution of class c along the
readout, one factorisation a(c, j, s) g_c(x) = Σ_t U[c, j, t] H[t, s, x] + R of a small matrix gives
S(p, x) = Σ_t H[t, s(p), x] K[image of U_t](p, x): one coefficient image per term, transformed along the
phase-encode axis first. The rank is the smallest for which
κ max_(s,x) Σ_c mass_c Λ_c max_j |R[c, j, s, x]| ≤ 0.05 standard deviations of the receiver noise of one k-space
sample at 64 averages (κ: gain of the displacement operator; mass_c: the largest sample class c can make; Λ_c: the
Lebesgue constant of the interpolation over the transmit scale). Row J12: every sample against the Fourier sum line
by line (7e-15); J13: the truncation error is 0.13 to 0.28 of its bound; J14: in the image, where the errors of the
samples add coherently, a pixel is off by 0.016 of its noise at 64 averages; J16, J17: the T2 blurring of `fse` is
the point-spread function it should be.

**Seconds per scan.** `mrgym timing --cpu` (processor seconds of one process, brain world at 3 T, noise-free data
not cached). Measured on a machine that ran some forty other processes on a quota of 32 cores: read the rows
against `se`, not as absolute numbers.

<!-- timing of the core sequences: begin -->
| Sequence | matrix 64 | matrix 128 | matrix 256 |
|---|---:|---:|---:|
| spgr | 0.47 | 0.36 | 0.49 |
| spgr, 4 echoes | 1.31 | 1.44 | 1.53 |
| afi | 1.23 | 0.85 | 1.01 |
| se | 0.41 | 0.64 | 0.65 |
| ir | 0.44 | 0.54 | 0.60 |
| bssfp | 0.80 | 0.95 | 0.86 |
| epi | 1.17 | 1.46 | refused |
| fse, 8 echoes | 1.26 | 1.27 | 1.94 |
| fse, 16 echoes | 1.52 | 1.76 | 2.54 |
| fse, 32 echoes | 2.56 | 2.50 | 3.00 |
| mese, 8 echoes | 1.41 | 1.38 | 2.13 |
| mese, 32 echoes | 2.73 | 3.12 | 5.33 |
| megre, 4 echoes | 1.29 | 1.33 | 1.58 |
| megre, 16 echoes | 4.07 | 4.53 | 5.96 |
| dwse, b 0 | 0.44 | 0.47 | 0.57 |
| dwse, b 1000 | 0.40 | 0.42 | 0.49 |
<!-- timing of the core sequences: end -->

(Median of 2. A second run of the same command gave `fse, 16 echoes` 1.81 to 2.63, `mese, 32 echoes` 3.24 to 5.37
and `se` 0.40 to 0.61: the load of the machine moves a number by a third. Not re-timed on a quiet machine.) A scan of `dwse` costs what `se` costs;
`megre` costs what `spgr` costs per echo; an echo train costs 3 to 4 times a spin echo at 16 echoes and 5 to 8 times
at 32, most of it in the transforms of the 15 to 45 coefficient images.

Limits the console sets for these sequences, each refused with its reason: `fse` and `mese` at most 32 echoes and
matrix 256; `megre` at most 16 echoes.

**Not served, and what each would need.**

| Sequence of the library | Why not in this package | What it needs |
|---|---|---|
| `mprage` (inversion-prepared gradient-echo train), `satrec`, `t2prep` | not built: time | a timing rule and a sampling plan (lines per shot, order); the amplitude is one number per line and class with ideal spoiling, so `form_lines` takes it as it is |
| `ir` with an inversion that depends on T2 (known difference 13 of [measured.md](measured.md)) | the core has no relaxation during a pulse | a block that carries the result of `design.rotation.bloch`, which is not a rotation |
| `mt_spgr` | each tissue of the world is one pool | a bound fraction, an exchange rate and a bound-pool T2 per tissue class, with sources; the saturation pulse in the timing and the RF power |
| `dess`, `spgr_lobe` (a real spoiler) | 0.85 and 1.3 s per 510 columns in the periodic solve | the flip-angle nodes of `megre` (exact for these sequences too); for `dess` the readout evolution of the second echo |
| `steam` | three selective pulses per column: no reduction to one angle | cheap per column; a timing rule (TE, TM) and crushers on a second axis |
| gratings that survive the repetition in `fse`, `mese`, `dwse` (rows J23, J24: 7e-3 at T1 900 ms, 0.16 for a fluid at TR 1500 ms) | the reduction rests on the ideal crusher | the odd-order gratings as part of the periodic state: a linear system of about 2 n unknowns per column |
| the six on the core | every served number would move at 1e-12 or more | one engine version with a table of differences |
