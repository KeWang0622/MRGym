# Receive arrays: several coils, their noise, and parallel imaging

`mrgym.scanner.array` gives a world several receive elements and forms the multi-channel k-space of a scan.
`mrgym.scanner.recon` holds the reconstruction mathematics. Both are libraries: the scanner an agent uses still
has one receive channel ([limits.md](limits.md), block 5), no tool and no task uses them yet, and a world without
the key `array` is the world of before, byte for byte. Read [conventions.md](conventions.md) first.

## 1. What a sensitivity map is

A receive coil is a loop of wire. Precessing magnetisation induces a voltage in it. How much a spin at position r
contributes is given by reciprocity: it is the field that a unit current in the same loop would produce at r. Only
the part of that field that is transverse to B0 and rotates with the spins counts. With B0 along z this is

    S(r) = (Bx(r) - i By(r)) / 2          the sensitivity map: one complex number per position and coil

Its magnitude says how strongly the coil hears that position, its phase with what delay. The image of coil c is the
object times S_c. A small loop next to the head hears the tissue beside it well and the far side hardly at all.

## 2. The noise covariance

Most of the noise comes from the object itself: thermal currents in conductive tissue induce a random voltage in
every coil. Two coils that look at the same tissue hear partly the same noise. All of it is described by one
matrix,

    psi_ij = E[n_i conj(n_j)]             diagonal: the noise power of each channel; off-diagonal: what two share

Here `psi` is dimensionless: 1 on the diagonal means the noise density of the world's single channel, so that every
noise law of the scanner (bandwidth, averages, matrix) holds per channel (`ArrayEngine.sample_sd`, row N3).
Measured as on a scanner, from samples without signal: `recon.noise_covariance` (row N1).

## 3. Why the combination matters

Each coil gives its own image. Adding them with the right weights gives the best single image; adding them with
the wrong ones wastes signal-to-noise ratio.

- **Optimal** (`recon.roemer`): weight every coil with its own sensitivity and with the inverse of the noise
  covariance, `rho = S^H psi^-1 d / (S^H psi^-1 S)`. No linear combination has less noise (rows C1, C3, C7). It needs
  the maps.
- **Root sum of squares** (`recon.sos`): no maps needed. It uses the data as their own map. Close to optimal where
  the signal is strong and the channels' noise is alike; below one good channel when one channel is noisy (rows C2,
  C4, C5). After whitening the channels (`recon.prewhiten`) it equals the optimal one at high SNR (row C6). Its
  background is not zero: the magnitude of N channels of pure noise follows a chi distribution (rows N4, N5).
- **Adaptive** (`recon.adaptive`): estimates the weights from the image itself, block by block (rows C11, C12).

Two channels show the size of the effect in closed form (rows C9, C10): with equal sensitivity and noise
correlation rho the optimal combination gains sqrt(2 / (1 + rho)) over one channel; a second channel that hears no
signal but shares noise still gains 1 / sqrt(1 - |rho|^2), because its noise can be subtracted.

## 4. SENSE in four lines, and what g is

A scan that acquires only every R-th phase-encode line is R times shorter and folds the image: pixel p of the folded
image of coil c is the sum of R pixels a field of view / R apart, each weighted with that coil's sensitivity there.

    a = A rho                                            A[c, m] = S_c(p_m): coils by folded pixels
    rho = U a,   U = (A^H psi^-1 A)^-1 A^H psi^-1          the best linear unbiased estimate
    cov(rho) = (A^H psi^-1 A)^-1                          its noise
    SNR_R(p) = SNR_full(p) / (g(p) sqrt(R)),   g(p) = sqrt([(A^H psi^-1 A)^-1]_pp [A^H psi^-1 A]_pp) >= 1

The factor sqrt(R) is the price of fewer samples. The factor g is the price of telling the folded pixels apart with
coils. It is 1 where the coils see the folded pixels very differently and large where they see them alike. For two
folded pixels it is 1 / sqrt(1 - |c|^2), with c the cosine between the two pixels' sensitivity vectors (row P3).
g depends on the coil layout, on the phase-encode direction and on R (rows P31 to P35), and on every element
working (rows P36 to P40). `recon.sense_plan` returns the unfolding matrices, the g map and the noise of every pixel;
`ArrayEngine.noise_sd_map` is the same in the scanner's units, for a verifier (rows P8, P13, P18, P23).

The maps a user has are estimates (`recon.estimate_maps`): from a low-resolution prescan divided by the body-coil
image, or from calibration lines in the middle of the accelerated scan. What they cost is reported in rows P25 to
P30. Even the true maps leave a residual, because a map varies inside a pixel (rows P9, P14, P19, P24); only data
that are the maps times an object on the pixels unfold to round-off (row P1). Regularisation (`reg`) trades noise
for bias (rows P41 to P44). `recon.sense_cg` solves the same problem for any set of lines. GRAPPA is not built
(`recon.grappa` is its seam).

## 5. What is declared here, and what will replace it

| | Declared now (`array.make_array`) | Replaced by |
|---|---|---|
| Maps | circular loops on a ring, a row or two opposite rows around the object; the exact Biot-Savart field of a loop with steady current (`array.loop_field`, elliptic integrals), which is the quasi-static limit: no wavelength effects, no conductor other than the loop, no shield | B1- computed on a grid by a field solver, handed in with `array.given(maps=...)` |
| Noise covariance | the overlap of the loops' magnetic fields over the object, reduced by a decoupling factor (`array.overlap_psi`) | the loss integral of the electric fields over the object, handed in with `array.given(psi=...)` |
| Element gains and cable phases | a small spread drawn per world | measured or computed values |
| Faults (`array.with_faults`) | a dead element (no signal, its own noise), a noisy element, a gain error | |
| Size of the SNR | the optimal combination has the sensitivity of a receive profile of 1 at the isocentre (`centre_snr`) | follows from the computed fields |

Layout, gains and faults are drawn from a random stream of their own and are hidden. An agent is told the number of
channels (`array.public`).

## 6. How the multi-channel data are formed

- **Per coil** (`ArrayEngine.per_coil`): the engine images the object once per element, with the element's map in
  place of the receive profile on the master grid. This is the definition, and it costs one scan per element.
- **Fast** (`ArrayEngine.fast`): the engine forms the object's k-space once, a little beyond the acquired band; the
  maps multiply its image; the result goes back to k-space. Fat is formed without its chemical shift and receives
  its spectrum after the coil has weighted it, so every fat peak is weighted where the fat is. Maps are taken at
  the position of the spins (the off-resonance displacement along the readout is undone) and never outside the
  object.

The two differ where a map varies inside the blur of one acquired sample: rows F4, F5. The difference grows as
loops shrink towards a few pixels (rows F6 to F8) and with the displacement of the spins; for echo-planar scans and
moving objects the per-coil formation is used. `ArrayEngine.formation_error` measures it for a protocol.
Noise is drawn channel by channel in the order of the single-channel receiver, so one uniform coil reproduces the
single-channel scan bit for bit (row F1).

## 7. Validation

`python -c "from mrgym.scanner import recon; recon.validate()"` prints the table below (about a minute on one core);
`python -m mrgym.scanner.array_checks --markdown` writes it. Rows marked "reported" state a size without a
tolerance. An independent re-derivation with dense matrices, quadrature and Monte Carlo is
`python tools/audit/array_rederive.py`.

<!-- validation table: begin -->
### Formation of multi-channel data

| # | Quantity | Closed form or independent method | Found | Tolerance | |
|---|---|---|---|---|---|
| F1 | one uniform coil, full sampling, formed per coil, with noise (two echoes): the combined image against the engine's single-channel image of the same world and noise stream | equal bit for bit | 0 | ≤ 0 | pass |
| F2 | the same through the SENSE unfolding at R = 1 | equal | 0 | ≤ 1.0e-14 | pass |
| F3 | the same by the fast formation: rms difference / rms of the noise-free image (a uniform map: what is left is the engine's own compression of the time functions, SVD_TOL = 1e-4, which the split into water and fat changes) | the per-coil formation | 2.0e-05 | ≤ 2.0e-04 | pass |
| F4 | fast against per-coil formation, 8 loops on a ring, spgr, matrix 96, 300 Hz per pixel: rms difference of the k-space / its rms | per coil: the map in place of the receive profile on the master grid | 9.8e-04 | ≤ 0.005 | pass |
| F5 | fast against per-coil formation, 8 loops on a ring, spgr, matrix 64, 300 Hz per pixel: rms difference of the k-space / its rms | per coil: the map in place of the receive profile on the master grid | 0.001484 | ≤ 0.005 | pass |
| F6 | the same for one loop of radius 60 mm (16 pixels) 20 mm behind the head, matrix 64: the difference grows as the map varies more inside the blur of a sample | reported | 0.00125 | reported |  |
| F7 | the same for one loop of radius 30 mm (8 pixels) 20 mm behind the head, matrix 64: the difference grows as the map varies more inside the blur of a sample | reported | 0.003527 | reported |  |
| F8 | the same for one loop of radius 15 mm (4 pixels) 20 mm behind the head, matrix 64: the difference grows as the map varies more inside the blur of a sample | reported | 0.009302 | reported |  |
| F9 | the positions used for the maps on a turned and warped grid (rotation 30 degrees, gradient non-linearity 0.03) against the master grid of that world | world.grid | 0 mm | ≤ 1.0e-09 | pass |

### Noise

| # | Quantity | Closed form or independent method | Found | Tolerance | |
|---|---|---|---|---|---|
| N1 | covariance of the samples of 1000 noise scans (8 channels, 2048 samples each) / (2 sd^2), sd from the single-channel noise law: largest deviation from psi in units of its sampling error | psi of the array; sampling error sqrt(psi_ii psi_jj / N) | 2.081 standard errors | ≤ 4.5 | pass |
| N2 | the same: largest relative deviation of an element | reported | 0.001454 | reported |  |
| N3 | noise of a coil image of a channel with psi_ii = 1 (ArrayEngine.image_sd) | Engine.noise_sd of the same protocol | 0 | ≤ 1.0e-12 | pass |
| N4 | background of the root sum of squares of 8 whitened channels (368640 pixels of noise): mean | chi distribution with 16 degrees of freedom: sqrt(2) Gamma(8.5) / Gamma(8) = 3.9380 sd | 4.0e-04 | ≤ 0.003 | pass |
| N5 | the same: standard deviation | sqrt(16 - mean^2) = 0.7014 sd | 0.001965 | ≤ 0.005 | pass |
| N6 | the same without whitening (channels correlated as psi says), mean / the chi mean for the mean channel variance: the formula needs white channels | reported | 0.9989 | reported |  |
| N7 | background magnitude of one channel: mean and standard deviation (the larger deviation) | Rayleigh: 1.2533 sd, 0.6551 sd | 7.9e-04 | ≤ 0.005 | pass |

### Coil combination

| # | Quantity | Closed form or independent method | Found | Tolerance | |
|---|---|---|---|---|---|
| C1 | SNR of the optimal combination / SNR of the root sum of squares, smallest value over the object (closed forms, 8 loops) | at least 1 (Cauchy-Schwarz) | 1.003 | ≥ 1 | pass |
| C2 | SNR of the root sum of squares / SNR of the optimal combination, mean over the object | reported | 0.9792 | reported |  |
| C3 | SNR of the optimal combination / SNR of the best single channel, smallest value over the object | at least 1 (the single channel is one of the linear combinations) | 1.101 | ≥ 1 | pass |
| C4 | share of the object where the root sum of squares has at least the SNR of the best single channel | reported (no theorem: see C5) | 1 | reported |  |
| C5 | the same share when one channel has 16 times the noise variance, without whitening | reported | 0.6358 | reported |  |
| C6 | root sum of squares of whitened channels: SNR / SNR of the optimal combination, largest deviation (closed forms, noisy channel included) | equal | 1.1e-15 | ≤ 1.0e-12 | pass |
| C7 | noise of the optimal combination by 300 replicas / its closed form, mean over the object | sd of a coil image / sqrt(S^H psi^-1 S) | 7.5e-04 | ≤ 0.01 | pass |
| C8 | SNR of the root sum of squares by replicas of a scan with 20 times the signal / its closed form, mean over the bright half of the object | S^H S / sqrt(S^H psi S) x magnetisation / sd | 0.001354 | ≤ 0.02 | pass |
| C9 | two channels with equal sensitivity and noise correlation rho (0, 0.3, -0.5, 0.8): gain of the optimal combination over one channel | sqrt(2 / (1 + rho)) | 2.2e-16 | ≤ 1.0e-12 | pass |
| C10 | two channels, the second sees no signal but shares noise (rho = 0.3, 0.6 e^(1.1 i), -0.9): gain of the optimal combination over the first channel alone | 1 / sqrt(1 - \|rho\|^2) | 2.2e-16 | ≤ 1.0e-12 | pass |
| C11 | adaptive combination (Walsh, 5 x 5 pixels, no maps) of the noise-free scan: magnitude / the optimal combination in the same units, median deviation over the object | equal where the block sees one sensitivity direction | 1.1e-04 | ≤ 0.02 | pass |
| C12 | noise of the magnitude of the adaptive combination of a noisy scan by 60 replicas, in units of the whitened channel noise, mean over the bright half of the object | reported; 1 for fixed unit weights | 0.9964 | reported |  |

### SENSE and the g-factor

| # | Quantity | Closed form or independent method | Found | Tolerance | |
|---|---|---|---|---|---|
| P1 | residual of the unfolding when the data are the maps times an object on the pixels (R = 2, 3, 4, every offset of the lattice): largest error / largest value | round-off | 1.9e-14 | ≤ 1.0e-10 | pass |
| P2 | the same by conjugate gradients on R = 3 with 12 calibration lines | the object | 1.6e-11 | ≤ 1.0e-06 | pass |
| P3 | g of two pixels folded onto each other, 6 random channels and a random psi | 1 / sqrt(1 - \|c\|^2), c = the cosine between the two pixels' sensitivity vectors in the metric psi^-1 | 0 | ≤ 1.0e-10 | pass |
| P4 | two channels, each seeing one of the two folded pixels, noise correlation rho (0.3, 0.6 e^(1.1 i), -0.9): g | 1 / sqrt(1 - \|rho\|^2) | 2.2e-16 | ≤ 1.0e-12 | pass |
| P5 | the prototype array (given from outside), matrix 96, R = 2: mean g over the object, closed form | the value recorded for the prototype: 1.135 | 1.9e-04 | ≤ 0.002 | pass |
| P6 | the prototype array (given from outside), R = 2: g measured by 300 replicas (noise of SENSE / noise of the full optimal image / sqrt(R)), mean over the object, / the mean closed form | 1 | 4.9e-04 | ≤ 0.01 | pass |
| P7 | the prototype array (given from outside), R = 2: SNR_full / SNR_R / (g sqrt(R)), mean over the object | 1 | 7.2e-04 | ≤ 0.01 | pass |
| P8 | the prototype array (given from outside), R = 2: the verifier's noise map (ArrayEngine.noise_sd_map) / the noise by replicas, mean over the object | sd_full(p) g(p) sqrt(R) | 7.9e-04 | ≤ 0.01 | pass |
| P9 | the prototype array (given from outside), R = 2: residual aliasing of the engine's data with the true maps at the pixel centres, median of \|SENSE - full\| / \|full\| over the object (a map varies inside a pixel; the data know, the pixel value does not) | reported | 0.006179 | reported |  |
| P10 | the prototype array (given from outside), matrix 96, R = 3: mean g over the object, closed form | the value recorded for the prototype: 1.944 | 1.8e-05 | ≤ 0.002 | pass |
| P11 | the prototype array (given from outside), R = 3: g measured by 300 replicas (noise of SENSE / noise of the full optimal image / sqrt(R)), mean over the object, / the mean closed form | 1 | 5.0e-04 | ≤ 0.01 | pass |
| P12 | the prototype array (given from outside), R = 3: SNR_full / SNR_R / (g sqrt(R)), mean over the object | 1 | 9.8e-04 | ≤ 0.01 | pass |
| P13 | the prototype array (given from outside), R = 3: the verifier's noise map (ArrayEngine.noise_sd_map) / the noise by replicas, mean over the object | sd_full(p) g(p) sqrt(R) | 8.9e-04 | ≤ 0.01 | pass |
| P14 | the prototype array (given from outside), R = 3: residual aliasing of the engine's data with the true maps at the pixel centres, median of \|SENSE - full\| / \|full\| over the object (a map varies inside a pixel; the data know, the pixel value does not) | reported | 0.01525 | reported |  |
| P15 | 8 loops on a ring, matrix 96, R = 2: mean g over the object, closed form | reported | 1.012 | reported |  |
| P16 | 8 loops on a ring, R = 2: g measured by 300 replicas (noise of SENSE / noise of the full optimal image / sqrt(R)), mean over the object, / the mean closed form | 1 | 0.001656 | ≤ 0.01 | pass |
| P17 | 8 loops on a ring, R = 2: SNR_full / SNR_R / (g sqrt(R)), mean over the object | 1 | 0.001473 | ≤ 0.01 | pass |
| P18 | 8 loops on a ring, R = 2: the verifier's noise map (ArrayEngine.noise_sd_map) / the noise by replicas, mean over the object | sd_full(p) g(p) sqrt(R) | 1.3e-04 | ≤ 0.01 | pass |
| P19 | 8 loops on a ring, R = 2: residual aliasing of the engine's data with the true maps at the pixel centres, median of \|SENSE - full\| / \|full\| over the object (a map varies inside a pixel; the data know, the pixel value does not) | reported | 0.006022 | reported |  |
| P20 | 8 loops on a ring, matrix 96, R = 3: mean g over the object, closed form | reported | 1.246 | reported |  |
| P21 | 8 loops on a ring, R = 3: g measured by 300 replicas (noise of SENSE / noise of the full optimal image / sqrt(R)), mean over the object, / the mean closed form | 1 | 0.001311 | ≤ 0.01 | pass |
| P22 | 8 loops on a ring, R = 3: SNR_full / SNR_R / (g sqrt(R)), mean over the object | 1 | 1.9e-04 | ≤ 0.01 | pass |
| P23 | 8 loops on a ring, R = 3: the verifier's noise map (ArrayEngine.noise_sd_map) / the noise by replicas, mean over the object | sd_full(p) g(p) sqrt(R) | 4.5e-04 | ≤ 0.01 | pass |
| P24 | 8 loops on a ring, R = 3: residual aliasing of the engine's data with the true maps at the pixel centres, median of \|SENSE - full\| / \|full\| over the object (a map varies inside a pixel; the data know, the pixel value does not) | reported | 0.0174 | reported |  |
| P25 | 8 loops on a ring, maps estimated from a prescan (matrix 64, 600 Hz per pixel) with the body coil as reference, full sampling: median of \|combined image - body-coil image\| / \|body-coil image\| over the object (the error of the maps alone) | reported | 0.01815 | reported |  |
| P26 | the same maps, R = 2: median of \|SENSE - body-coil image\| / \|body-coil image\| over the object | reported | 0.0215 | reported |  |
| P27 | the same maps, R = 2: the part that is residual aliasing: median of \|SENSE - full scan combined with the same maps\| / the latter | reported | 0.006308 | reported |  |
| P28 | the same maps, R = 3: median of \|SENSE - body-coil image\| / \|body-coil image\| over the object | reported | 0.03281 | reported |  |
| P29 | the same maps, R = 3: the part that is residual aliasing: median of \|SENSE - full scan combined with the same maps\| / the latter | reported | 0.01856 | reported |  |
| P30 | 8 loops on a ring, R = 2 with 24 calibration lines, maps estimated from those lines (no body coil): median of \|SENSE - full scan combined with the same maps\| / the latter over the object | reported | 0.01047 | reported |  |
| P31 | a row of 4 loops behind the head (they differ along x only), R = 2: mean g over the object, phase direction along y | reported | 1.938 | reported |  |
| P32 | the same, phase direction along x | reported | 1.521 | reported |  |
| P33 | the row of 4 loops, R = 2: mean g with the phase direction along y / along x | above 1: folding along the axis on which the loops do not differ costs more | 1.274 | ≥ 1.1 | pass |
| P34 | the row of 4 loops, R = 3: mean g with the phase direction along y / along x | above 1: folding along the axis on which the loops do not differ costs more | 1.64 | ≥ 1.1 | pass |
| P35 | the ring of 8 loops, R = 3: the same ratio | reported | 0.9731 | reported |  |
| P36 | one of the 8 loops dead, R = 3: noise of the unfolded image / noise with all loops, smallest value over the object | at least 1 (a channel less cannot lower the noise) | 1 | ≥ 1 | pass |
| P37 | the same: at the pixel where g rises most, one of the three pixels folded onto each other has the dead loop as its nearest: best rank of the dead loop by distance among them | 1: the loop that told that pixel from the others is gone | 1 | ≤ 1 | pass |
| P38 | the same: g at that pixel with all loops | reported | 1.354 | reported |  |
| P39 | the same: g at that pixel with the dead loop | reported | 3.828 | reported |  |
| P40 | the same: mean g over the object with the dead loop / with all loops | reported | 1.075 | reported |  |
| P41 | R = 4: mean g over the object | reported | 2.153 | reported |  |
| P42 | R = 4, Tikhonov regularisation 0.003: mean noise amplification over the object / unregularised | reported: below 1, paid with bias | 0.8977 | reported |  |
| P43 | R = 4: median residual \|SENSE - full\| / \|full\| over the object, unregularised | reported | 0.03469 | reported |  |
| P44 | the same with the regularisation | reported | 0.05102 | reported |  |

### Cost

| # | Quantity | Closed form or independent method | Found | Tolerance | |
|---|---|---|---|---|---|
| T1 | whole scan of a new contrast, 8 channels, R = 2, matrix 96 (fast formation, noise, SENSE with its unfolding matrices and g map) / the single-channel scan of a new contrast, medians of 3 on one core | design target: 1.5 | 1.376 | ≤ 1.5 | pass |
| T2 | the same: the 8-channel scan | reported | 0.5071 s | reported |  |
| T3 | the same: the single-channel scan | reported | 0.3686 s | reported |  |
| T4 | peak resident memory of this process so far (worlds on the 640 grid, per-coil references included) | reported; limit 3000 | 583.1 MB | reported |  |
<!-- validation table: end -->

## 8. Sources

How far each was checked: **read** = the full text was fetched and the passage read; **abstract** = only the abstract
was fetched; **not compared with the paper** = the formula is the one the paper is known for, written down by the
authors of this package and supported only by the rows named. Records: [references.md](references.md).

| Source | Used for | Checked |
|---|---|---|
| Roemer 1990 | optimal combination, noise correlation, overlap of neighbours | abstract; the formulas were not compared with the paper: rows C1, C3, C7, C9, C10 |
| Pruessmann 1999 | unfolding matrix, g-factor, SNR law, maps by division by a reference | read (equations 2, 23, 24; "Sensitivity assessment") |
| Pruessmann 2001 | conjugate-gradient SENSE with intensity correction | not compared with the paper: row P2 and the re-derivation |
| Griswold 2002 | GRAPPA (the seam only) | read |
| Walsh 2000 | adaptive combination | read (the submitted manuscript) |
| Kellman & McVeigh 2005 | SNR units, whitening, noise scaling of the transform, background bias per channel count | read; its printed bias values are checked in the re-derivation |
| Robson 2008 | pseudo multiple replicas, the estimate of psi | read (equations 10 to 12) |
| Constantinides 1997 | chi statistics of the root sum of squares | read (equations 2, 4); its erratum was not obtained |
| Breuer 2005, Breuer 2009 | controlled aliasing; g of GRAPPA (neither is built) | read; abstract |
