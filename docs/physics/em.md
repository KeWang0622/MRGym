# Radio-frequency fields on the slice, from Maxwell's equations

`mrgym.em` computes the transmit field B1+, the receive field B1- and the absorbed power (SAR) of a slice from the
permittivity and conductivity of its tissues and the currents of a coil. It is a library with tests. It serves no
task and no tool, and it does not change the scanner: the engine still images with the declared patterns of
`scanner/world.py` (`B1_SHAPE`, `fields["b1"]`, `fields["rx"]`) and still prices RF by B1rms² (limits.md, sections 10
and 12). Section 9 says what connecting the two would take.

Read [conventions.md](conventions.md) first. Results of this package appear only inside four generated blocks
(tissue table, demonstration, validation table, cost estimate), each printed by a command of
`python -m mrgym.em.validate` and refreshed with `--write-doc`; numbers elsewhere are definitions or published
values with their source.

## 1. The model, stated once

Time-harmonic Maxwell equations in two dimensions on the axial slice. Everything is invariant along z: an
infinitely long body in an infinitely long coil whose conductors run along z. Then the electric field has only a z
component and the magnetic field lies in the plane (the TM-z problem), and with the time dependence exp(+iωt)

    ∇²Ez + k0² εc Ez = iωµ0 Jz,      εc = εr − iσ/(ωε0),      Bx = (i/ω) ∂Ez/∂y,   By = −(i/ω) ∂Ez/∂x.

One scalar equation per frequency. It contains the wavelength in tissue, the conduction currents that shield and
heat, the interference of several sources, and nothing else. It is the right problem for the rungs of a birdcage
or the elements of a transmit array seen in an axial slice. It leaves out the end rings, the finite length of body
and coil, and every field with a z component of B or an in-plane component of E. What that costs is in section 7.

| Module | Computes |
|---|---|
| `units` | constants, the phasor convention, wavenumber, wavelength, penetration depth |
| `properties` | εr, σ of the engine's tissue classes at any frequency (four-term Cole-Cole model), density; the maps of a world |
| `analytic` | closed forms: a cylinder of concentric lossy layers in a birdcage current sheet (quadrature, linear) or among line currents, open or shielded: Ez, Bx, By, B1+, B1-, absorbed, delivered and radiated power |
| `solver` | finite-difference frequency-domain solve for any εr(y, x), σ(y, x): one sparse factorisation, any number of channels; absorbing layer or shield; powers, point SAR, averaged SAR, the quadratic forms of power and local SAR; the problem of an engine world (`WorldModel`) |
| `coils` | birdcage (one mode, or two ports), transmit arrays of line or loop elements; receive maps by reciprocity; noise resistance matrix, SNR of the best combination, g-factor |
| `shim` | channel weights for a uniform \|B1+\| in a region, phase-only or magnitude-and-phase, under limits on power and local SAR; the trade-off curve |
| `ept` | conductivity and permittivity back from the complex B1+ map (Helmholtz relation); the phase-only, magnitude-only and transceive-phase approximations |
| `validate` | the tables of this page |

## 2. Conventions

- **SI units** throughout: m, A, V/m, T, S/m, kg/m³, W. Positions `(y, x)` in metres, arrays `[y, x]`, angles from
  +x towards +y: the engine's axes (conventions.md, section 3) in metres instead of millimetres.
- **Phasors.** A field is Re{F exp(+iωt)}; F is a peak amplitude, so a time average is Re{F G*}/2. An outgoing
  wave is exp(−ikr), a lossy medium has Im k < 0.
- **Rotating components.** B1+ = (Bx + iBy)/2 is the amplitude of the part of the field that rotates from +x
  towards +y; B1- = (Bx − iBy)*/2 that of the part rotating the other way (row A8). The engine's magnetisation
  precesses from +x towards +y (a positive frequency gains positive phase) and nutates right-handed about the field,
  dM/dt = γ B × M ([design.md](design.md), section 2). Row A8b integrates that equation in the laboratory frame:
  B1+ excites, with the flip angle γ|B1+|T and the phase convention of design.md, and B1- does not. A proton turns
  left-handed about B0; the two pictures are mirror images, so these formulas describe a proton with B0 along −z,
  and with B0 along +z the patterns of B1+ and B1- change places (Vaidya et al. 2016 [F], restating Hoult 2000
  [M]). Nothing the engine reports depends on the choice.
- **Reception.** The signal a coil receives from a unit magnetisation is proportional to conj(B1-) = (Bx − iBy)/2
  of the field that coil makes with unit current (`coils.receive_maps`).
- **Per unit.** Channel maps are per ampere of drive. Powers are per metre of length along z (W/m); the noise
  resistance is per metre (Ω/m). SAR = σ|Ez|²/(2ρ) in W/kg does not depend on the length. With weights scaled so
  that |B1+| is the B1rms of a sequence, powers and SAR are time averages.
- **Quadrature.** Rung k at angle φk carries I exp(−iφk). An ideal sheet K0 exp(−iφ) makes B1+ = µ0K0/2 in an
  empty coil at low frequency; N rungs make µ0NI/(4πR) on the axis (rows A2, A3, S6).

## 3. B1+ and B1-, and why they differ at high field

At low frequency a quadrature coil makes a field that rotates in one sense everywhere: B1- is zero and the
transmit pattern is flat. Two things change with frequency. The wavelength in tissue shrinks to the size of the
head (rows A11, A11b), so the fields of the rungs no longer add in phase away from the axis; and the conduction
currents they induce make fields of their own. In a homogeneous cylinder under the quadrature sheet the result is
B1+ ∝ J0(kρ) and |B1-| ∝ |J2(kρ)| (row A4): the transmit field is brightest on the axis, and a counter-rotating
field grows from zero on the axis outwards. The transmit pattern of a coil and its receive pattern are then
different functions of position; for a body without circular symmetry neither is the mirror image of the other.
The demonstration in section 6 gives the size in the brain world.

## 4. Where SAR comes from

The body absorbs σ|E|²/2 per unit volume. E is induced by the changing B (Faraday), so at low frequency
|E| = ωB1+ρ around the axis of a quadrature coil and the absorbed power for a fixed B1+ grows as ω² (closed form
πσω²|B1+|²a⁴/4 per metre for a cylinder of radius a; rows A10, S12). The law holds within 15 % at 64 MHz in a
head-sized cylinder and fails at 298 MHz (rows A10b, A10c). Three quantities are returned:

- **point SAR** σ|Ez|²/(2ρ) per pixel (`Solver.point_sar`);
- **averaged SAR**, the stand-in for the 10 g average: power absorbed in a square of side 21.5 mm (the side of a
  10 g cube at 1000 kg/m³) divided by the mass in that square; pixels cut by its edge count with the share inside.
  The fields do not change along z, so averaging over the cube equals averaging over its cross-section. The square
  has a fixed size; it is not grown until it holds 10 g, and it has a value only where it holds at least half of
  the mass of a full square (`Solver.averaged_sar`; rows S13 to S13c). This is a definition of this package, not
  that of a standard;
- **mean SAR of the slice**: absorbed power per metre over mass per metre.

For channel weights w all three are quadratic forms, SAR_p = Re(wᴴ Q_p w) (`Solver.sar_matrices`; row H1). A limit
on local SAR has to be imposed at every pixel where the map has a value (`Solver.sar_support`), not on the body
mask of a world, which leaves out the partly filled pixels of the surface. Limits
of IEC 60601-2-33 as restated in the open literature (a review in PubMed Central, PMC9849420 [F]; the standard was
not read): whole body 2 and 4 W/kg (normal and first-level mode), head 3.2 W/kg, local 10 g in head and trunk 10
and 20 W/kg, extremities 20 and 40 W/kg, averaged over 6 minutes, any 10 s at most twice that
(`shim.IEC_LIMITS_W_PER_KG`). Nothing here is a safety statement: section 7.

## 5. Tissue values

Four-term Cole-Cole model with the parameters of Gabriel and Gabriel's 1996 compilation (tags in
`properties.py`): F = parameter row fetched and checked against the IFAC-CNR calculator at the three frequencies
(rows P1, P2); f = row fetched, no calculator value to check against; C = our choice, no source. Densities: IT'IS
database V5.0. These are values of excised tissue as compiled in 1996, not of a patient.

<!-- tissue table: begin -->
| Engine class | Row of the dispersion model | Tag | 64 MHz: permittivity, S/m | 128 MHz | 298 MHz | Wavelength at 298 MHz, mm | Density, kg/m³ | Note |
|---|---|---|---|---|---|---|---|---|
| `wm` | Brain (White Matter) | F | 67.8, 0.292 | 52.5, 0.342 | 43.8, 0.413 | 147 | 1041 |  |
| `gm` | Brain (Grey Matter) | F | 97.4, 0.511 | 73.5, 0.587 | 60.1, 0.691 | 123 | 1044.5 |  |
| `deep_gm` | Brain (Grey Matter) | C | 97.4, 0.511 | 73.5, 0.587 | 60.1, 0.691 | 123 | 1044.5 | deep grey matter takes the grey-matter row |
| `csf` | Cerebro Spinal Fluid | F | 97.3, 2.066 | 84.0, 2.143 | 72.8, 2.224 | 95 | 1007 |  |
| `scalp_fat` | Fat (Not Infiltrated) | F | 6.5, 0.035 | 5.9, 0.037 | 5.6, 0.040 | 415 | 911 | subcutaneous fat; the choice between the two fat rows is ours |
| `skull` | Bone (Cortical) | F | 16.7, 0.060 | 14.7, 0.067 | 13.4, 0.082 | 270 | 1908 |  |
| `diploe` | Bone (Cancellous) | F | 30.9, 0.161 | 26.3, 0.180 | 23.2, 0.215 | 202 | 1178.3 |  |
| `skin` | Skin (Dry) | F | 92.2, 0.436 | 65.4, 0.523 | 49.9, 0.640 | 134 | 1109 |  |
| `muscle` | Muscle | F | 72.2, 0.688 | 63.5, 0.719 | 58.2, 0.770 | 123 | 1090.4 |  |
| `blood` | Blood | F | 86.4, 1.207 | 73.2, 1.249 | 65.7, 1.315 | 110 | 1049.75 |  |
| `lesion_ms` | Brain (Grey Matter) | C | 97.4, 0.511 | 73.5, 0.587 | 60.1, 0.691 | 123 | 1044.5 | no dispersion model of a plaque was read: the grey-matter row (more water than white matter) |
| `tumour` | Brain (Grey Matter) | C | 97.4, 0.511 | 73.5, 0.587 | 60.1, 0.691 | 123 | 1044.5 | as lesion_ms |
| `oedema` | Brain (Grey Matter) | C | 97.4, 0.511 | 73.5, 0.587 | 60.1, 0.691 | 123 | 1044.5 | as lesion_ms |
| `liver` | Liver | F | 80.6, 0.448 | 64.3, 0.511 | 53.6, 0.609 | 131 | 1078.75 |  |
| `spleen` | Spleen | F | 110.6, 0.744 | 82.9, 0.835 | 66.6, 0.968 | 114 | 1089 |  |
| `kidney_cortex` | Kidney | F | 118.6, 0.741 | 89.6, 0.852 | 70.6, 1.021 | 111 | 1066.25 | one kidney row for cortex and medulla |
| `kidney_medulla` | Kidney | F | 118.6, 0.741 | 89.6, 0.852 | 70.6, 1.021 | 111 | 1066.25 | one kidney row for cortex and medulla |
| `pancreas` | Thyroid | F | 73.9, 0.778 | 66.8, 0.804 | 62.5, 0.851 | 119 | 1086.5 | the calculator's pancreas is its gland row, which is the thyroid row |
| `abd_muscle` | Muscle | F | 72.2, 0.688 | 63.5, 0.719 | 58.2, 0.770 | 123 | 1090.4 |  |
| `marrow` | Bone Marrow (Infiltrated) | f | 16.4, 0.154 | 13.5, 0.162 | 12.1, 0.175 | 268 | 1028.5 | red marrow; density of the IT'IS row Bone Marrow (Red) |
| `subcut_fat` | Fat (Not Infiltrated) | F | 6.5, 0.035 | 5.9, 0.037 | 5.6, 0.040 | 415 | 911 | the choice between the two fat rows is ours |
| `visceral_fat` | Fat (Average Infiltrated) | f | 13.6, 0.066 | 12.4, 0.070 | 11.7, 0.076 | 288 | 911 | the choice between the two fat rows is ours |
| `bowel_fluid` | Vitreous Humor | C | 69.1, 1.503 | 69.1, 1.505 | 69.0, 1.518 | 105 | 1004.5 | a body fluid: the calculator's body-fluid row is the vitreous-humor row |
| `cord` | Nerve | F | 55.1, 0.312 | 44.1, 0.354 | 37.0, 0.418 | 157 | 1075 | the calculator's spinal cord is the nerve row |
| `liver_lesion` | Vitreous Humor | C | 69.1, 1.503 | 69.1, 1.505 | 69.0, 1.518 | 105 | 1004.5 | cyst-like: the body-fluid row |
| `acr_fill` (test objects) | declared | C | 78.0, 0.900 | 78.0, 0.900 | 78.0, 0.900 | 108 | 1000 | see `properties.PHANTOM` |
| `phantom_fill` (test objects) | declared | C | 78.0, 0.500 | 78.0, 0.500 | 78.0, 0.500 | 112 | 1000 | see `properties.PHANTOM` |
| `oil` (test objects) | declared | C | 3.0, 0.000 | 3.0, 0.000 | 3.0, 0.000 | 581 | 920 | see `properties.PHANTOM` |
<!-- tissue table: end -->

Not sourced: the lesions (`lesion_ms`, `tumour`, `oedema`, `liver_lesion`) and `bowel_fluid` take the row of
another tissue; `deep_gm` takes grey matter; the three liquids of the test objects are declared (no measurement of
the ACR filling at these frequencies was found).

## 6. Demonstration: the brain world in a quadrature birdcage

A 16-rung birdcage of radius 14 cm without shield around the brain world (fixed seed), solved on the master grid
coarsened by 5 (2 mm) and enlarged to hold the coil. `python -m mrgym.em.validate --demo`.

<!-- demonstration: begin -->
| \|B1+\| over the brain, relative to the isocentre | Coefficient of variation | Centre over rim | Smallest | Largest | Mean \|B1-\| over mean \|B1+\| | Peak point SAR over head mean | Peak averaged SAR over head mean | Head-mean SAR at 1 µT, W/kg | B1+ at which the head mean is 3.2 W/kg, µT | Power of linear over quadrature drive, field along y, along x |
|---|---|---|---|---|---|---|---|---|---|---|
| computed, 64 MHz | 0.0436 | 1.12 | 0.824 | 1.03 | 0.1 | 8.36 | 2.81 | 0.0986 | 5.7 | 1.45, 2.49 |
| computed, 128 MHz | 0.109 | 1.34 | 0.619 | 1.03 | 0.252 | 7.68 | 2.58 | 0.341 | 3.06 | 1.5, 2.37 |
| computed, 298 MHz | 0.249 | 1.65 | 0.272 | 1.01 | 0.707 | 5.45 | 2.59 | 0.741 | 2.08 | 1.55, 2.24 |
| declared pattern of the engine, 3 T (`world.B1_SHAPE`) | 0.0354 | 1.11 | 0.872 | 1 | – | – | – | – | – | – |
| computed at 128 MHz over declared, pixel by pixel in the brain | 0.0828 | – | 0.697 | 1.06 | – | – | – | – | – | – |

Derived from the table (and from the published numbers named in section 7):

- Coefficient of variation, declared over computed at 128 MHz: 0.33. Correlation of the two patterns over the brain: 0.82.
- Mean |B1-| over mean |B1+| in the brain at 64, 128, 298 MHz: 0.10, 0.25, 0.71.
- Coefficient of variation at 298 MHz over the published 0.138 of a three-dimensional head model: 1.81.
- Peak averaged SAR over mean SAR, computed over published (3.4, 3.2, 4.1): 0.83, 0.81, 0.63 at 64, 128 and 298 (published: 300) MHz.
- B1+ at which the mean SAR of the slice is 3.2 W/kg, over the engine's public B1rms limit: 0.95 at 1.5 T (6 µT), 1.02 at 3 T (3 µT).
- Mean SAR for a fixed B1+ at the isocentre, 128 over 64 MHz: 3.45 (the square of the frequency ratio: 4.00); 298 over 128 MHz: 2.18 (the square: 5.42).
- Power of linear drive over quadrature drive, mean of the two field directions: 1.97, 1.93, 1.90 (a circular body: 2).
<!-- demonstration: end -->

How the declared pattern differs from the computed one at 3 T. Both brighten the centre. The declared pattern is
several times flatter (first bullet) and is a paraboloid of the brain's own ellipse with a small random tilt; the
computed one falls off faster towards the rim, follows the conductivity of the ventricles and the CSF rim, and
comes with a counter-rotating field (second bullet) that the declared receive profile knows nothing of. The
declared receive profile is unrelated to the transmit pattern; the computed one is tied to it by reciprocity.
Against a published three-dimensional result the two-dimensional model overstates the non-uniformity (third
bullet), so the truth at 3 T probably lies between the two rows.

Three readings of the other columns. The mean SAR of the slice reaches 3.2 W/kg at a B1+ close to the engine's
public B1rms limits (fifth bullet); those limits are declared numbers, so this is the agreement of two models, not
a validation of either. Linear drive costs more or less than twice the power of quadrature in this elliptical
head, depending on the direction of the field; the mean of the two directions is close to the factor 2 of a
circular body (rows A9, S9; Glover et al. 1985 [F] report measured power ratios of 0.5 to 0.7 for circular against
linear polarisation). The frequency-squared law already overestimates the step from 64 to 128 MHz, and the step
to 298 MHz by far more (sixth bullet).

## 7. What the two-dimensional assumption costs

Compared with published three-dimensional head models:

| Quantity | Here (brain world) | Published, three dimensions |
|---|---|---|
| Relative standard deviation of \|B1+\| over an axial slice at 300 MHz, birdcage-like drive | demonstration, row 298 MHz and third bullet: the model overstates it | 0.138 (16 elements, axial slice), 0.202 (whole brain): Mao et al. 2006 [F] |
| Peak averaged SAR over mean SAR at 64, 128, 300 MHz | demonstration, fourth bullet: the model is lower, most at 300 MHz, where it misses the published rise | 3.4, 3.2, 4.1 (10 g over head average, TEM-like coil): Wang et al. 2007 [F] |
| Uniformity after RF shimming at 300 MHz, relative to the birdcage-like mode | row H3e (8 line elements in a shielded coil) | 0.060 after shimming from 0.138 with 16 elements: Mao et al. 2006 [F] |
| B1 non-uniformity at 7 T against 4 T | not compared: the definition behind the published numbers was not read | "≈42 % compared to ≈23 %": Vaughan et al. 2001 [S] |
| Mean SAR of the head for a stated B1+ | demonstration, column "head-mean SAR at 1 µT"; not compared | no number was obtained |

Why: a real coil has end rings whose currents close the rungs; their fields have z components and their electric
field is not along z. A head ends; fields enter it from above as well, which fills in the minima a cylinder shows.
A two-dimensional head model in a birdcage has a precedent in the literature (Jin and Chen 1997 [S]).
Local SAR near the end rings, the neck and the shoulders does not exist here. No number of this package is a
safety figure for a person.

Numerical accuracy inside the model, on a 2 mm grid (section 8): rows S3 and S3c give the error of Ez and of B1+
away from tissue boundaries, both second order in the grid (rows S4, S4b). In the one-pixel band along a boundary
the centred difference that turns Ez into B is first order (rows S3d, S4c): what is returned there is the field
averaged over one pixel to either side. The absorbing layer is validated from 30 MHz with 16 cells and from 10 MHz
with 24; below that the solver refuses it and a shield has to be used (rows S1, S1b).

## 8. Validation

`python -m mrgym.em.validate` runs the rows below and exits with status 1 if one fails;
`tests/test_em_validate.py` asserts that all pass and that this table is the one the command prints. Errors below
1e-3 are printed with one significant digit. The series solution and the finite-difference solver share no code
beyond the constants; each is also checked on its own (the series against Hankel functions, magnetostatics, its own
power balance and numerical derivatives; the solver against reciprocity and the discrete Poynting theorem). The
references were written by the same authors as the code. A reference that says "by construction" marks a property
the code enforces; such a row fails only if the enforcement breaks. The rows show that the code solves the stated
two-dimensional equations; they do not show that those describe a head in a coil.

<!-- validation table: begin -->
### Tissue properties

| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |
|---|---|---|---|---|---|
| P1 | four-term Cole-Cole model: permittivity and conductivity of 15 tissue rows at 64, 128 and 298 MHz | the values returned by the IFAC-CNR calculator for the same rows (six digits) | 2.2e-05 relative | ≤ 1.0e-04 | pass |
| P2 | wavelength 2 pi / Re k = c / (f Re sqrt(eps_complex)) and penetration depth -1 / Im k of the same rows | the wavelength and penetration depth returned by that calculator | 1.7e-05 relative | ≤ 1.0e-04 | pass |
| P3 | every tissue class of the engine (25) has a row and a density; permittivity above 1, conductivity above 0, density 800 to 2000 kg/m^3 at the three frequencies | by construction | 0 | ≤ 0 | pass |
| P4 | property maps of a hand-made world of four pixels (grey matter, CSF, half of each, air), and an override of one conductivity | the two records, their mean, and permittivity 1 without conductivity or mass; the overriding value | 0 | ≤ 1.0e-12 | pass |

### Closed-form solutions

| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |
|---|---|---|---|---|---|
| A1 | Ez of one line current in an empty coil: Bessel series to order 60, inside 0.71 of the source radius, 64 / 128 / 298 MHz | -(w mu0 I / 4) H0^(2)(k0 \|r - r'\|) (Graf's addition theorem) | 3.5e-09 relative | ≤ 1.0e-08 | pass |
| A2 | empty quadrature current sheet at 100 kHz, open and shielded: \|B1+\| over the bore and \|B1-\| | magnetostatics: B1+ = mu0 K0 / 2, uniform, times 1 - (R / S)^2 inside a shield; B1- = 0 | 3.8e-07 relative | ≤ 1.0e-06 | pass |
| A3 | birdcage of 16 rungs at 100 kHz: B1+ inside 0.64 of the coil radius, and on the axis | direct sum of the fields mu0 I / (2 pi r) of 16 straight wires; on the axis mu0 N I / (4 pi R) | 3.8e-07 relative | ≤ 1.0e-06 | pass |
| A4 | homogeneous brain-like cylinder (radius 9 cm) under the quadrature sheet, 64 / 128 / 298 MHz: B1+ and \|B1-\| relative to B1+ on the axis | J0(k rho) and \|J2(k rho)\| | 1.6e-15 | ≤ 1.0e-12 | pass |
| A5 | three layers of one material; two layers of free space | the homogeneous cylinder; the empty coil | 8.0e-16 relative | ≤ 1.0e-10 | pass |
| A6 | power balance of the series solution: two-layer cylinders, 16 rungs, quadrature and linear, open and shielded, 64 / 128 / 298 MHz | delivered by the sources = absorbed (Gauss-Legendre quadrature of sigma \|Ez\|^2 / 2) + radiated (far field) | 2.0e-14 relative | ≤ 1.0e-09 | pass |
| A7 | Helmholtz equation in each layer and in the gap (CSF / brain cylinder, linear drive, 120 random points per frequency): laplace(Ez) + k^2 Ez | zero; the Laplacian by fourth-order differences of the series on a 0.25 mm stencil | 3.7e-08 relative | ≤ 1.0e-05 | pass |
| A7b | Bx, By, B1+ and B1- of the series at the same points | Bx = (i / w) dEz/dy, By = -(i / w) dEz/dx by fourth-order differences of Ez; B1+ = (Bx + i By) / 2, B1- = (Bx - i By)* / 2 | 1.6e-07 relative | ≤ 1.0e-06 | pass |
| A8 | the rotating components of a linearly driven coil at two off-centre points: the laboratory field Re{B exp(i w t)} averaged over one period in the frame turning from +x towards +y, and in the frame turning the other way | B1+ = (Bx + i By) / 2 and B1- = (Bx - i By)* / 2 as returned (the definitions restated from Hoult 2000) | 1.9e-16 relative | ≤ 1.0e-12 | pass |
| A8b | the Bloch equation dM/dt = gamma B x M integrated in the laboratory frame (`scipy.integrate.solve_ivp`, DOP853) for three elliptically polarised fields at the precession frequency, nutation 2,000 times slower than precession, 40 degrees: Mxy in the rotating frame and Mz; and for fields with B1- only | the convention of design.md: Mxy = -i exp(i p) sin(flip), Mz = cos(flip) with flip = gamma \|B1+\| T and p the phase of B1+; no excitation by B1- (both up to a ripple of the order of the ratio of the two rates, 5e-4) | 7.5e-04 | ≤ 0.001 | pass |
| A10 | power absorbed per unit B1+^2 on the axis in a cylinder (radius 9 cm, permittivity 60, 0.6 S/m) at 1 and 2 MHz | quasi-static closed form pi sigma w^2 \|B1+\|^2 a^4 / 4 (the field E = w B1+ rho of a uniform rotating B1+): power for a fixed B1+ grows as the square of the frequency | 1.3e-04 relative | ≤ 0.001 | pass |
| A10b | the same ratio to the quasi-static law at 64 MHz | the law still holds within 15 % | 0.1218 relative | ≤ 0.15 | pass |
| A10c | the same ratio at 298 MHz | the law fails: with the wavelength in the body the field on the axis is no longer the field everywhere, and the power is below half of the law's | 0.1092 | ≤ 0.5 | pass |
| A11 | wavelength c / (f sqrt(eps)) in water (permittivity 78) at 94, 125 and 138 MHz | published: 36, 27 and 24 cm (Collins et al. 2005, J Magn Reson Imaging 21:192; two digits) | 0.0249 relative | ≤ 0.03 | pass |
| A11b | wavelength at 300 MHz in the mean of the grey and white matter rows | published: "approximately 13.4 cm" in brain tissue (Collins and Wang 2011, Magn Reson Med; PubMed Central PMC3078983) | 0.009919 relative | ≤ 0.03 | pass |
| A11c | half a wavelength at 64 MHz in a medium of permittivity 100 | published: "about 25 cm" (Glover et al. 1985, J Magn Reson 64:255) | 0.06315 relative | ≤ 0.1 | pass |

### Finite-difference solver

| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |
|---|---|---|---|---|---|
| S1 | absorbing layer of 16 cells (2 mm grid): Ez of a line current in free space, inside 10 cm, at 64 / 128 / 298 MHz | -(w mu0 I / 4) H0^(2)(k0 \|r - r'\|) | 1.0e-04 relative | ≤ 3.0e-04 | pass |
| S1b | a layer of 24 cells at 10 and 20 MHz, the range in which the solver asks for the thicker layer | the same | 6.4e-05 relative | ≤ 3.0e-04 | pass |
| S2 | a layer of 8 cells at 64 / 128 / 298 MHz: the error a thinner layer leaves | the same | 0.004261 relative | ≤ 0.01 | pass |
| S3 | homogeneous, fat / muscle and CSF / brain cylinders in an open 16-rung birdcage, 2 mm grid, 64 / 128 / 298 MHz: Ez inside the cylinder | the Bessel series for the same 16 line currents | 0.002309 of the largest \|Ez\| | ≤ 0.004 | pass |
| S3b | the same cases: B1+ and B1-, rms error over the cylinder | the series | 0.007321 of the rms | ≤ 0.01 | pass |
| S3c | the same cases: B1+ more than 1.5 pixels away from a layer boundary | the series | 0.001448 of the largest \|B1+\| | ≤ 0.0025 | pass |
| S3d | the same cases: B1+ in the pixels next to a layer boundary, where the curvature of Ez jumps and the centred difference is first order | the series | 0.02368 of the largest \|B1+\| | ≤ 0.03 | pass |
| S3e | the same cases: absorbed power and power delivered by the rungs | the series | 0.001862 relative | ≤ 0.003 | pass |
| S4 | order of convergence of Ez from 4 to 2 to 1 mm at 298 MHz (the three cylinders in the shielded coil): the smallest of the three | second order | 1.795 | ≥ 1.7 | pass |
| S4b | the same: B1+ away from layer boundaries | second order | 1.821 | ≥ 1.7 | pass |
| S4c | the same: B1+ in the pixels next to a layer boundary, distance of the order from 1 | first order | 0.1478 | ≤ 0.35 | pass |
| S4d | Ez error on the 1 mm grid in those cases | the series | 6.2e-04 of the largest \|Ez\| | ≤ 0.001 | pass |
| S4e | the open coil with its absorbing layer, fat / muscle cylinder, from 4 to 2 mm: order of convergence of Ez | second order | 1.908 | ≥ 1.7 | pass |
| S5 | homogeneous cylinder in a shielded birdcage (shield radius 19 cm), 2 mm grid, 64 / 128 / 298 MHz: Ez, B1+ away from the boundary, absorbed power | the series with Ez = 0 on the shield | 0.001067 relative | ≤ 0.004 | pass |
| S6 | quasi-static limit: empty shielded birdcage of 16 rungs at 100 kHz, 2 mm grid: \|B1+\| within 4 cm of the axis, and \|B1-\| there | magnetostatics: uniform, mu0 N I / (4 pi R) (1 - (R / S)^2); no B1- | 3.3e-06 relative | ≤ 1.0e-04 | pass |
| S6b | the shield, a circle on a square grid: B1+ of the empty shielded birdcage at 128 MHz within 9 cm, from 4 to 2 to 1 mm: order of convergence | second order (the series with Ez = 0 on the shield) | 1.819 | ≥ 1.7 | pass |
| S6c | the same: error on the 2 mm grid | the series | 1.8e-04 relative | ≤ 5.0e-04 | pass |
| S7 | power balance at 298 MHz (fat / muscle cylinder, open coil): delivered by the rungs - absorbed - flux through the faces of a box of 32 cm | zero: the discrete form of Poynting's theorem holds for the difference equations | 2.3e-15 of the delivered power | ≤ 1.0e-09 | pass |
| S7b | the same flux through a box of 30 cm | the same flux: nothing is absorbed or delivered between the two boxes | 1.6e-15 relative | ≤ 1.0e-09 | pass |
| S7c | radiated power: the flux through the box | the series: far-field sum over the azimuthal orders | 3.2e-05 relative | ≤ 0.001 | pass |
| S7d | the same body in the shielded coil: delivered - absorbed | zero: nothing leaves a shield | 1.1e-16 relative | ≤ 1.0e-09 | pass |
| S8 | reciprocity in the brain world at 298 MHz: Ez at point j from a line current at point i, 12 random points inside and outside the head, and the asymmetry of the matrix | Ez at i from the same current at j; a symmetric matrix | 3.7e-16 relative | ≤ 1.0e-10 | pass |
| S9 | linear drive with twice the rung current of the quadrature drive (fat / muscle cylinder, 128 MHz, 2 mm grid): B1+ on the axis and absorbed power | the same B1+ for twice the power: for a body with circular symmetry the orders +1 and -1 do not mix, at any frequency (Glover et al. 1985 report up to 50 % less power with circular polarisation) | 7.8e-07 relative | ≤ 1.0e-05 | pass |
| S10 | central brightening: \|B1+\| relative to the axis over a head-sized brain-like cylinder (radius 9 cm) in the 16-rung birdcage, 2 mm grid, 64 / 128 / 298 MHz | \|J0(k rho)\|: the pattern of the ideal quadrature sheet | 0.001023 | ≤ 0.005 | pass |
| S10b | its size at 298 MHz: coefficient of variation of \|B1+\| over the cylinder, as a factor from the published value (the larger of ratio and inverse ratio) | published: relative standard deviation 0.138 over an axial slice of a three-dimensional head model at 300 MHz, 16 elements driven as a birdcage (Mao et al. 2006, Magn Reson Med 56:918): the same order of size, within a factor 2.5 | 1.698 | ≤ 2.5 | pass |
| S10c | coefficient of variation at 64 MHz over that at 298 MHz | near absence at 1.5 T: below 0.4 | 0.3126 | ≤ 0.4 | pass |
| S11 | wavelength in a lossless water cylinder (permittivity 78) at 298 MHz, 1 mm grid, shielded coil: radius of the first null of B1+, as a wavelength 2 pi x radius / 2.4048 | lambda = c / (f sqrt(eps)) = 113.9 mm (B1+ is J0(k rho), first zero at k rho = 2.4048) | 7.4e-06 relative | ≤ 0.002 | pass |
| S12 | absorbed power per unit B1+^2 on the axis at 1 and 2 MHz (shielded coil, 2 mm grid): the ratio of the two, and the value at 1 MHz | 4 (the square of the frequency); pi sigma w^2 a^4 / 4 | 1.5e-04 relative | ≤ 0.002 | pass |
| S13 | SAR in the brain world at 128 MHz: the average over the square of 21.5 mm (11 pixels across, the two at the ends counted with 0.886) at 200 random pixels of the head; the point SAR integrated over the mass | a loop over the pixels of each square: weighted power in the square / weighted mass in the square; the absorbed power | 3.6e-16 relative | ≤ 1.0e-12 | pass |
| S13b | the averaged SAR of a uniform field in a uniform medium, away from the edge of the grid | sigma \|E\|^2 / (2 rho) | 2.2e-16 relative | ≤ 1.0e-12 | pass |
| S13c | peak averaged SAR over mean SAR of the brain world at 128 MHz on grids of 1.6, 2 and 3.2 mm: largest difference from the 1.6 mm value | the definition does not depend on the pixel: within 3 % | 0.02011 relative | ≤ 0.03 | pass |
| S14 | one factorisation and 8 channel solves at 298 MHz on the brain world (41,616 unknowns, 2 mm; best of 3, one thread) | requirement | 0.3254 s | ≤ 2 | pass |

### Coils, reciprocity and reception

| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |
|---|---|---|---|---|---|
| C1 | the two linear ports of a birdcage combined with weights (1, -i) and (1, +i) in the brain world at 128 MHz: Ez, B1+, B1- | the quadrature and the reverse mode solved directly (B1- of a combination is the conjugate of the combined conj(B1-)) | 4.3e-14 relative | ≤ 1.0e-10 | pass |
| C1b | mean \|B1+\| over the brain: reverse mode over quadrature mode | the reverse mode turns against the magnetisation: below 0.5 at 3 T | 0.2505 | ≤ 0.5 | pass |
| C2 | 8 line elements and 8 loop elements in an empty shielded coil at 64 MHz, with the weights of the birdcage-like mode and with the opposite sense: B1+ at the isocentre of the opposite sense, and B1- of the birdcage-like mode, over B1+ of the birdcage-like mode | zero: on the axis of an empty coil each sense makes one rotating component only | 7.3e-16 | ≤ 1.0e-09 | pass |
| C3 | 8 loops in the brain world at 128 MHz: Re(w^H R w) / 2 for 5 random weight vectors | the power absorbed by the field of the combined currents | 6.7e-16 relative | ≤ 1.0e-10 | pass |
| C4 | signal-to-noise ratio of the best combination over the brain, for the 8 loops and for 8 channels that are random complex mixtures of them, solved again from their own currents | equal: the optimum does not depend on the choice of channels (this fails if the noise matrix is conjugated) | 2.1e-11 relative | ≤ 1.0e-08 | pass |
| C4b | the same with the transposed (conjugated) noise matrix of the mixed channels | not equal: a change above 1 % shows that row C4 can tell the two apart | 493.6 relative | ≥ 0.01 | pass |
| C5 | 40 pixels of the brain: the combination with weights psi^-1 s against the optimum; 5 random combinations each, excess over the optimum | equal; never above (Roemer et al. 1990) | 1.0e-12 relative | ≤ 1.0e-10 | pass |
| C6 | g-factor: two channels and two folded positions with unit noise (50 random cases); six channels with a random noise matrix; one position | 1 / sin(angle between the two sensitivity vectors); the same after whitening the channels; 1 | 5.3e-14 relative | ≤ 1.0e-09 | pass |
| C6b | g-factor map of the 8 loops over the brain for twofold and threefold undersampling along y | explicit unfolding: noise of each unfolded pixel through the unfolding matrix, over the noise of the best combination of the fully sampled pixel | 4.1e-12 relative | ≤ 1.0e-08 | pass |
| C6c | smallest g-factor of that explicit computation, minus 1 | unfolding never lowers the noise: not below zero | -4.1e-12 | ≥ -1.0e-10 | pass |

### RF shimming

| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |
|---|---|---|---|---|---|
| H1 | brain world at 298 MHz, 8 line elements in a shielded coil, 3 random weight vectors: Re(w^H Q_p w) for point SAR at every pixel that holds tissue and for the averaged SAR at every pixel where it has a value; Re(w^H R w) / 2; the two maps outside those pixels | the SAR maps and the absorbed power of the combined field; zero outside | 1.2e-15 relative | ≤ 1.0e-10 | pass |
| H2 | gradient of the objective with respect to the real and imaginary parts of the weights | centred differences | 5.4e-10 relative | ≤ 1.0e-06 | pass |
| H3 | 8 problems (magnitude-and-phase and phase-only; no limit, a power limit, a limit on the averaged local SAR, both): objective of what the optimiser delivers, before the final scaling and the fallback, minus the objective of the birdcage-like mode at its best feasible scale; the largest of the eight | below zero: shimming is at least as good as the mode it starts from, on its own objective (the returned result cannot be worse by construction: it falls back on the start) | -0.02621 | ≤ 0 | pass |
| H3b | the same problems: how often the result had to fall back on the start | never | 0 | ≤ 0 | pass |
| H3c | the same problems: largest excess over a limit (power; averaged local SAR at every pixel where it has a value) in what the optimiser delivers, before the final scaling removes it | none beyond the tolerance of the optimiser | 1.8e-15 relative | ≤ 1.0e-06 | pass |
| H3d | the same problems: first-order optimality at the result (what is left of the gradient of the objective after the active limits, for a step of relative size 1; the objective itself is 0.01 to 0.2 there) | zero at a local optimum | 4.6e-08 | ≤ 1.0e-05 | pass |
| H3e | coefficient of variation of \|B1+\| over the brain at 298 MHz without limits: magnitude-and-phase shim (best of 9 starts) over birdcage-like mode | smaller: below 0.7 | 0.5123 | ≤ 0.7 | pass |
| H4 | two channels, phase only: objective of the result over the smallest objective on a scan of 7,201 relative phases, each at its best scale, minus 1; one channel under a SAR limit: the result against the scaled start | the scan (best of the two starts 0 and pi); with one channel only the scale is free | 0 relative | ≤ 1.0e-05 | pass |
| H5 | trade-off of uniformity against the limit on averaged local SAR (6 limits from 0.2 to 1.5 of the SAR of the birdcage-like mode): largest rise of the objective when the limit is relaxed | none, by construction: each limit also starts from the solution of the tighter one, which stays allowed | -3.4e-04 | ≤ 0 | pass |
| H5b | the same curve: objective at the tightest limit over objective at the loosest | a real trade-off: above 1.5 | 7.581 | ≥ 1.5 | pass |
| H5c | the 5 limits of the curve that lie below the local SAR of the shim without limits: peak SAR of the result over the limit, smallest | the limit is reached: a result that stays below it gives uniformity away | 1 | ≥ 0.999 | pass |

### Electrical properties tomography

| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |
|---|---|---|---|---|---|
| E1 | permittivity and conductivity of the homogeneous cylinder from the B1+ of the Bessel series (2 mm grid, 5 x 5 window), more than 12 mm inside the surface, 64 / 128 / 298 MHz | the values the field was computed with | 0.009711 relative | ≤ 0.02 | pass |
| E1b | the same from the B1+ of the finite-difference solver | the same | 0.009515 relative | ≤ 0.02 | pass |
| E2 | where it fails: median error of the conductivity in the pixels at the surface of the cylinder, smallest over the three frequencies | large: the Helmholtz relation does not hold across a boundary (above 50 %) | 5.227 relative | ≥ 0.5 | pass |
| E3 | transceive-phase assumption on the centred cylinder under the ideal current sheet at 298 MHz (transmit in quadrature, receive in the reverse mode): difference between the transmit phase and the receive phase over the cylinder | zero: the two fields are mirror images, so half the transceive phase is the transmit phase | 1.2e-16 rad | ≤ 1.0e-09 | pass |
| E3a | the same with the solver and a birdcage of 16 rungs, B1+ taken as \|B1+\| exp(i transceive phase / 2): permittivity and conductivity | the values of the cylinder (16 rungs are close to the sheet) | 0.009802 relative | ≤ 0.02 | pass |
| E3a2 | the same with a birdcage of 4 rungs | the assumption fails with few rungs: above 10 % | 0.8417 relative | ≥ 0.1 | pass |
| E3b | the same assumption with a linearly driven sheet used for both: median error of the conductivity | the assumption fails (transmit and receive phase differ): above 10 % | 0.9492 relative | ≥ 0.1 | pass |
| E3c | phase-only conductivity laplace(phi) / (w mu0) at 298 MHz on the same cylinder: median error | an approximation that drops the term 2 grad(ln \|B1+\|) . grad(phi): above 10 % here | 1.394 relative | ≥ 0.1 | pass |
| E4 | Laplacian of a cubic polynomial with the 3 x 3, 5 x 5 and 7 x 7 least-squares kernels | exact | 5.3e-13 | ≤ 1.0e-10 | pass |
<!-- validation table: end -->

## 9. What is not built, and what the next step needs

- **The scanner does not use these fields.** Putting them in place of the declared patterns means: `fields["b1"]`
  = |B1+| over a reference value and `fields["rx"]` = conj(B1-) of the receiving coil times the phase of B1+, on the
  master grid (`WorldModel.to_world`, with `order=3` for these smooth maps), built once per world in
  `world.anatomy_fields`, `qa_fields`, `acr_fields`;
  evaluated again on a turned grid (`world.turned`), which here means interpolating the solved maps instead of
  re-evaluating a closed form; worlds at 7 T need tissue rows (`tissue.table`) the engine does not have. Every
  stored result that depends on a field would change: golden files, the fingerprints, the gates of every family
  that reads B1.
- **The third dimension.** End rings, z-directed fields, a finite head. The next step is a three-dimensional
  time-domain (Yee) solver on a coarse grid, run once per world and cached. An estimate from the measured time of
  one numpy update step on one thread, 15 RF periods to the steady state, a box of 0.4 × 0.4 × 0.5 m
  (`python -m mrgym.em.validate --cost3d`; the assumptions are in the docstring of `validate.cost_3d`, and no such
  solver exists yet):

<!-- cost: begin -->
```text
cell   cells      MHz   steps to 15 periods   s per step   s per channel   MB (6 fields + 6 coefficients, single precision)
5 mm   640,000     64                24,341       0.0152             370   31
5 mm   640,000    128                12,171       0.0152             185   31
5 mm   640,000    298                 5,228       0.0152              79   31
8 mm   155,000     64                15,213       0.0024              37   7
8 mm   155,000    128                 7,607       0.0024              18   7
8 mm   155,000    298                 3,268       0.0024               8   7
```
<!-- cost: end -->

It also needs anatomy through the slice, which the engine's worlds do not have, an absorbing boundary in three
dimensions, and a reference to validate against (the Mie series of a layered sphere takes the place of the
cylinder). Further:

- **Real coil geometry**: conductors of finite width, capacitors, tuning and matching, coupling between elements,
  the losses of the coil itself. Here a channel is a set of ideal currents; power is what the body absorbs and what
  radiates.
- **Implants and their heating**: needs sub-millimetre conductors and the tangential E along them; the
  two-dimensional E is along z only, which is the worst case for a wire along z and blind to any other.
- **Temperature**: the bio-heat equation with perfusion on top of the SAR map.
- **Virtual observation points** (Eichfelder and Gebhardt 2011 [S]): the shim uses every pixel's matrix with
  cutting planes, which is exact and fast at this size; a compression is needed only for much larger problems.
- **Parallel imaging**: `coils.g_factor` and the noise matrix are the ingredients; nothing reconstructs an
  undersampled image. **Spatially selective parallel transmission** (Katscher et al. 2003 [S], Zhu 2004 [S]): the
  per-channel maps are its input; only static shimming is built.
- **EPT with noise and in three dimensions**: the Helmholtz relation of `ept` has no missing term here because
  nothing varies along z. The transceive-phase assumption is exact only for a body with circular symmetry under an
  ideal current sheet; a birdcage of 16 rungs is close to it, one of 4 rungs is not, a linear coil is not (rows E3
  to E3b).
- **Elements with a net current.** A single rung (`coils.array(element="line")`) returns its current through the
  shield. Without a shield the common mode of such an array is a radiating line current that no real coil has: use
  line elements inside a shield and loops in an open coil.
- **Sub-pixel mixtures.** A pixel takes the mean permittivity and conductivity of what it holds, the mixing rule
  for a field along z when the pixel is much smaller than the wavelength; the error is second order in their ratio.

## 10. Sources, and how far each was read

[F] fetched and read for this page; [S] abstract or search result only; [M] recalled by the authors, not fetched.

| Source | Used for | Tag |
|---|---|---|
| C. Gabriel, S. Gabriel, Compilation of the dielectric properties of body tissues at RF and microwave frequencies, report AL/OE-TR-1996-0037 (1996), appendix C, HTML edition of IFAC-CNR | Cole-Cole parameters | F |
| IFAC-CNR calculator of the dielectric properties of body tissues (webnir.eu) | permittivity, conductivity, wavelength, penetration depth at 64, 128, 298 MHz (rows P1, P2) | F |
| S. Gabriel, R. W. Lau, C. Gabriel, Phys Med Biol 41:2271 (1996) | the model, as the paper of record | M |
| IT'IS Foundation, tissue properties database V5.0, doi:10.13099/VIP21000-05-0 | densities | F |
| M. V. Vaidya et al., Concepts Magn Reson Part B (2016) | definitions of B1+ and B1-, reception ∝ conj(B1-), the direction of B0 | F |
| D. I. Hoult, Concepts Magn Reson 12:173 (2000) | the origin of those definitions | M |
| C. M. Collins, Z. Wang, Magn Reson Med (2011), PubMed Central PMC3078983 | wavelength in brain at 300 MHz (row A11b); SAR ∝ f² at low field; the loss integral between two coils | F |
| G. H. Glover et al., J Magn Reson 64:255 (1985) | the lossy cylinder in a transverse RF field; factor 2 between linear and circular polarisation (rows A9, S9); half wavelength at 64 MHz (row A11c) | F |
| C. M. Collins et al., J Magn Reson Imaging 21:192 (2005) | wavelengths in water (row A11) | F |
| W. Mao, M. B. Smith, C. M. Collins, Magn Reson Med 56:918 (2006) | non-uniformity at 300 MHz before and after RF shimming (row S10b, section 7) | F |
| Z. Wang et al., J Magn Reson Imaging 26:437 (2007) | peak local over average SAR in the head (section 7) | F |
| A review of the IEC limits in PubMed Central, PMC9849420 | SAR limits (section 4) | F |
| A methodological review of electrical properties tomography in PubMed Central, PMC7910937; PMC5675043 | Helmholtz formulas with exp(+iωt), the phase-only and magnitude-only forms, failure at boundaries, the transceive-phase assumption | F |
| A paper on local SAR supervision in PubMed Central, PMC11604848 | the local-SAR quadratic form | F |
| J. T. Vaughan et al., Magn Reson Med 46:24 (2001) | B1 non-uniformity at 7 T and 4 T (section 7) | S |
| J. Jin, J. Chen, Magn Reson Med 38:953 (1997) | a two-dimensional head model in a birdcage | S |
| U. Katscher et al., IEEE Trans Med Imaging 28:1365 (2009); T. Voigt et al., Magn Reson Med 66:456 (2011) | electrical properties tomography | S |
| U. Katscher et al., Magn Reson Med 49:144 (2003); Y. Zhu, Magn Reson Med 51:775 (2004); K. Setsompop et al., Magn Reson Med 59:908 (2008) | parallel transmission, magnitude least squares | S |
| G. Eichfelder, M. Gebhardt, Magn Reson Med 66:1468 (2011) | virtual observation points | S |
| P. B. Roemer et al., Magn Reson Med 16:192 (1990); K. P. Pruessmann et al., Magn Reson Med 42:952 (1999) | SNR of the best combination; the g-factor formula | M |
| R. F. Harrington, Time-Harmonic Electromagnetic Fields (1961) | field of a line current, Bessel series of a cylinder | M |
| P. S. Tofts, J Magn Reson B 104:143 (1994); J. R. Keltner et al., Magn Reson Med 22:467 (1991); P. A. Bottomley, E. R. Andrew, Phys Med Biol 23:630 (1978); Q. X. Yang et al., Magn Reson Med 47:982 (2002); P.-F. Van de Moortele et al., Magn Reson Med 54:1503 (2005) | named as related analytic and experimental work; no number of this page rests on them | S |
