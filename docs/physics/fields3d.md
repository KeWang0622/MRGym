# Fields in three dimensions

`mrgym.scanner.fields3d` computes the static field that an object induces by its own magnetic susceptibility, shims
it, adds small inclusions in closed form, continues the declared transmit and receive shapes along z, and holds the
forward and the reference inverse step of susceptibility mapping.

**Status: a library.** No scan, no world and no task of this package uses it. The engine's field is still the
declared one of [limits.md](limits.md) section 10. The table in section 9 is printed by

    python -c "from mrgym.scanner import fields3d; fields3d.validate()"
    python -m mrgym.scanner.fields3d_validate --markdown      # the table below
    python -m mrgym.scanner.fields3d_validate --head          # the head figures of section 10

and `tests/test_fields3d.py` asserts that every row passes and that this page shows the rows the command prints.
Figures in the prose are named by their row (F1 to F45); none is typed here.

Source tags: **[F]** full text fetched and read, **[S]** abstract or bibliographic record only. Section 13 lists
the sources. No formula or value on this page rests on a source that was not fetched.

## 1. The model

Matter of volume susceptibility chi (SI, |chi| much smaller than 1) in a field B0 along the unit vector b carries the
magnetisation M = chi B0 / mu0. To first order in chi the field it adds along b is a convolution of chi with the
field of a point dipole, and a product in k-space:

    delta(k) = D(k) chi(k),      D(k) = 1/3 - (k . b)^2 / |k|^2,      delta = (B_nucleus - B0) . b / B0

This is the form of Marques & Bowtell 2005 [F] (their Eq. 6) and of Wang & Liu 2015 [F] ("D(k) = (1/3 − kz²/k²)
[k ≠ 0]"); Koch 2009 [F] attributes it to Salomir 2003 [S] and to Marques & Bowtell. First order is enough "if the
susceptibility range does not exceed 10−4 (or 100 ppm)" (Salomir 2003, abstract): air, tissue, bone and titanium
qualify, steel does not.

## 2. Conventions

| Item | Convention | Where |
|---|---|---|
| Sign | delta > 0 is a spin that precesses faster: positive off-resonance, positive phase at a gradient echo, as in [conventions.md](conventions.md) section 5. Other texts differ: Shmueli 2009 [F] writes "φ = −γ * B0 * TE * FT⁻¹[…]", the consensus paper of 2024 [F] "φB(TE) = γ·TE·ΔB0" and warns that "vendors may use the opposite sign convention" | `Field3D.hz`, `to_phase` |
| Units | chi and delta in ppm relative to soft tissue (water); Hz = ppm x 1e-6 x gamma x B0 at the world's field strength | `Field3D.hz_per_ppm` |
| Air | +9.4 ppm relative to tissue, the engine's constant. Schenck 1996 [F] lists water "−9.05×10⁻⁶" and air "0.36" separately; Koch 2009 uses −9.2 and 0.3 | `world.DCHI_AIR` |
| k = 0 | D has no limit at k = 0: its value depends on the direction of approach, that is on the shape of the object at infinity. "The value of the Fourier transform of the field at k = 0, which effectively describes the average value of the field, is not well-defined" (Marques & Bowtell). It is set to zero: before the image term of section 3 the mean over the transformed box is zero. Two consequences: a constant added to chi and to its surroundings changes nothing (F13), and the constant of the result has no meaning. A scanner removes it with the frequency adjustment, this module with the zeroth shim term | `dipole_kernel` |
| Sphere of Lorentz | The 1/3. A nucleus sees the macroscopic field minus (2/3) chi(r) B0, because the matter within a small sphere around it adds nothing on average. Included by default. `lorentz=False` gives the macroscopic field, D = 1 - (k . b)^2 / k^2; the two differ by (2/3)(chi - mean chi) exactly (a test) | `dipole_field(lorentz=)` |
| Axes | Lattices are arrays `[z, y, x]`; `spacing` and `Lattice.origin` follow the array (dz, dy, dx). Vectors (`b0_dir`, a frame's origin, row and column, centres of inclusions) are patient components (y, x, z) in mm from the isocentre, the order of the engine's grids and of `world["air"]`. Index i of an axis of `Lattice.centred` sits at (i - n // 2) x spacing, the engine's rule | `Lattice` |
| B0 | any direction; the engine's is (0, 0, 1), a lattice axis, whatever the slice orientation. Along a lattice axis D is the kernel of section 1. Oblique to the lattice, the mixed products k_i k_j of (k . b)^2 are replaced by sin(2 pi k_i h_i) sin(2 pi k_j h_j) / (4 pi^2 h_i h_j), the symbol of two central differences: the plain product is odd in each component, so it jumps at the edge of the zone of a periodic lattice, and the field then rings at the cell scale with ten times the error (F16b, F16c). Turning B0 equals turning the object (F14 for quarter turns, F15 and F16 for other angles) | `dipole_kernel` |

**Does the sphere of Lorentz matter for an image?** It is a local term, constant within one material. For one
material in air it is a constant, and the frequency adjustment removes it: the image of a homogeneous object does
not depend on it. Between two materials it changes the step of the field at their interface, by two thirds of
their susceptibility difference, and that is seen (fat against water, bone against tissue). It is kept for that
reason. The classical closed forms are printed without it: Schenck 1996 gives "ΔBz=2Δχ/3Bo inside the sphere",
"ΔBz=ΔχBo inside the cylinder" along B0 and "ΔBz=Δχ/2Bo" across it. With the term these are 0, 1/3 and -1/6 of the
step; row F5 and a test compare both forms.

## 3. The lattice

**Cell means.** The class weights of a lattice must be volume fractions of the cells, not samples at the cell
centres: a staircase misplaces the surface by up to half a cell. `volume_fraction(sdf, lattice)` forms them from
a signed-distance function by sampling the surface cells. With cell means the dipole moment of an object is exact
and the error falls with the square of the cell size (F21).

**Padding and images.** The transform makes the lattice periodic: "the calculated field is actually that which
would be generated by an infinite cubic grid of spheres" (Marques & Bowtell). They recommend a buffer whose size
"depends on the accuracy desired"; Bouwman & Bakker 2012 [S] state that "padding with a factor of two gives
accurate results" and replace it by a second, coarse transform. Here the lattice is padded by 1.5 (`pad`) and the
leading field of the images is taken off in closed form (`images=True`). To second order in the position the
images of a source q add

    (1/2) sum_src q(r') (r - r')^T H (r - r'),      H_ij = sum over n != 0 of d_i d_j f(R_n),   f = dipole field

where H is a lattice sum that converges absolutely (`image_hessian`). For a cubic cell and B0 along z this is the
formula 6.216 (a / L)^3 (r / L)^2 (3 cos^2 t - 1) for a sphere of radius a, which a separate derivation obtained
from a real-space sum of dipoles (F18, F19, F20c). The term is a second-order harmonic, so a first-order shim does
not remove it: for a head on a padded lattice it is the row F20d. A lattice that is meant to be periodic (an
infinite cylinder along a lattice axis) sets `images=False`.

**An infinite cylinder on a periodic lattice** differs from its closed form by a constant: minus the area fraction
of the cylinder in the cell times its inside value, exactly, for a square cell and any angle to B0 (F6 to F9). The
reason is the k = 0 convention: the closed form is zero far away, the lattice has zero mean. What is left is the
field of the image cylinders, 1.505 x (area fraction) x sin^2 t (rho / L)^2 cos 2 phi (F10, F11). In a cell that
is not square the constant also depends on the direction of B0 in the cell, and the square-cell rule is wrong by a
factor of three in the example of F11b.

**A body that leaves the lattice** (a neck) would end in a flat cut, which is an end face with the full surface
charge (F23). `extrude=("z-",)` repeats the face plane through half of the padding on that side (F22). The image
term is not applied to an extruded object.

## 4. Surfaces

The field is discontinuous wherever chi is. Across an interface with unit normal n it steps by

    (1/3 - (n . b)^2) x (step of chi)

which follows from the two boundary conditions (normal B and tangential H continuous) and holds on any curved
surface. At an air surface normal to B0 that is two thirds of the step from tissue to air (in Hz: the scale line
under the table of section 10). A lattice holds a mixture of both sides in its surface cells; interpolating it
puts a ramp of that size into the first millimetres of tissue. Koch 2009 says of the transform method that it "falls apart directly at susceptibility interfaces and
their immediate proximity"; Marques & Bowtell found deviations "within a distance of approximately two voxel
spacings from the boundary".

**Method.** For a susceptibility that varies only along n the dipole convolution is exact and local: delta =
kappa chi with kappa = 1/3 - (n . b)^2. The lattice field is therefore split into that local part and a smooth
remainder, the remainder is interpolated, and the local part is put back with the susceptibility of the material
that gives the signal:

    delta(r) = I{delta_lattice - kappa chi_lattice}(r) + I{kappa}(r) chi_seen(r)

I is trilinear interpolation, n the direction of the gradient of chi_lattice (central differences, averaged over
3 x 3 x 3 cells). `chi_seen` is the susceptibility of the tissue at the point. A caller that has the class weights
of the image plane passes it (`seen_chi`), and every interface, also bone against tissue, is then as sharp as the
plane's own grid. Without it the lattice's tissue-only mean is used, continued two cells into the air; interfaces
between tissues are then smeared over one cell, as on the plain lattice.

**What it costs.** The split is exact for a plane interface and for any layering parallel to it (two tests). On a
curved surface the remainder is of the order (cell size / radius of curvature): rows F24 to F26 give it for an
air sphere the size of a sinus and for a tissue sphere, against the closed form, at arbitrary positions within
1 mm of the surface and further away.

**The guards.** Rows F24b and F25b repeat the same points with plain interpolation and pass only if that error is
large. Row F32 is the image-level check that found the problem in the first place: one slice, imaged by the engine
from master grids of 0.8 and 0.4 mm; F32b repeats it with plain interpolation and passes only if the difference
exceeds the acceptance of F32. `tests/test_fields3d.py::test_near_surface_field_fails_with_plain_interpolation`
asserts both sides.

**Not done.** A masked-mean continuation of the field (keep the cells that are tissue with all neighbours, extend
outwards) was measured during the build: it is good where the field is flat (inside a convex tissue body) and no
better than plain interpolation above a small cavity, where the field rises steeply towards the surface.

## 5. Image planes

`sample(field, frame, grid, fov, thickness, chi_seen=None)` returns the two entries of `world["fields"]` that the
engine reads: `b0_hz` on the master grid of a plane, after the stored shim, and `dbdz_hz_per_mm`, the gradient
along the plane's normal. A frame is a `volume.Frame`, or a tuple (origin, row, column[, normal]) in patient
components; the grid rule is the engine's. `plane(name, pos_mm)` gives the three standard planes with the normals
of `volume.Frame.plane` (+z, +y, +x). The gradient is the secant across the slice,
(f(+thickness / 2) - f(-thickness / 2)) / thickness: the linear term that `rf.dephasing_factor` assumes, taken
over the range it is applied to. `offsets=` also returns the field on a stack of planes through the slice, for a
phase per plane.

Interpolation is trilinear. Its error bound is h^2 / 8 times the second derivative per axis; rows F27 to F31 give
the measured error against the closed form on axial, coronal and sagittal planes and for B0 along y, and the last
row of section 10 compares the 2 mm lattice with a 1 mm lattice on a slice of the head.

## 6. Shim

The basis is the nine solid harmonics up to second order, in mm^order, in the frame of the magnet (z along B0):

    1;   x, y, z;   z^2 - (x^2 + y^2) / 2,  z x,  z y,  x^2 - y^2,  x y

These are the Cartesian forms of Table 1 of Koch 2009 [F], except that the author manuscript prints the last one as
"2xy". Each has zero Laplacian (F34). `shim_fit` minimises the sum of squares over the cells of a volume of
interest with a limit on each coefficient (bounded least squares on the QR factor); it reports the channels at
their limit and the largest violation of the optimality conditions (F35, F36). The coefficients are absolute: the
fit is made on the field without shim, so fitting twice does not shim twice. A pure first- or second-order field
is removed to rounding (F33).

`SHIM_LIMITS_UT` holds DECLARED limits in microtesla per metre^order. They are of the order of a whole-body
system and are not a product's data; no source was found for them.

**Against the engine.** `world.anatomy_fields` fits a constant and the two in-plane gradients over the brain of
the slice and subtracts the mean through-slice gradient. That is a first-order least-squares shim over a thin slab.
On the engine's own closed-form field this module returns the same four coefficients to rounding (F37). On an
object that does not change along z, a first-order fit over the lattice volume and the engine's fit on one slice
of it are the same problem on two samplings of the region; F38 gives the difference.

## 7. Small structures

A sphere of fewer than two cells of radius is not resolved by the lattice (F39 gives the error at 1, 2 and 4
cells). The rule (`lattice_or_closed_form`, `MIN_LATTICE_RADIUS_CELLS`): a structure of at least two cells of
radius enters the lattice; a smaller one is left out of the lattice and added as a closed-form inclusion
(`Field3D(inclusions=[...])`: a sphere, or an infinite cylinder for a vessel), never both. The closed forms are
exact for any radius and are evaluated at the points themselves, so their own surface is sharp. This is what
`world.dipole_hz` does today for the declared air spheres; outside the sphere the two functions agree to rounding
(a test). Inside, this module returns 0 and the engine a value that no tissue sees.

Superposition holds because the model is first order (F13).

## 8. Transmit and receive profiles, susceptibility mapping

**Profiles.** `declared_profiles(kind, b0_t, size_mm, rng, rng_z)` continues the declared shapes of
`world.anatomy_fields` ("brain": central brightening; "abdomen": quadrupolar shading) along z. They are DECLARED
shapes, like the in-plane ones: polynomials of the position, not solutions of Maxwell's equations. The in-plane
parameters are drawn in the engine's order, so the same generator gives the 2D world's `b1` and `rx` on the plane
z = 0 to rounding (F41, and a test for the abdomen). Along z the radial terms count z as a third coordinate and the
tilts get a z component from a separate generator. The interface is two methods, `b1(y, x, z)` and
`rx(y, x, z)`. A computed map (`mrgym.em` solves for B1 on its own grid) can replace the declared object through an
adapter with the same two methods; that adapter is not written. The test objects (`qa`, `acr`,
`nist_t1`) are not continued here.

**Phase.** `to_phase(field_hz, te_ms)` returns 2 pi f TE per echo, wrapped into (-pi, pi] (F44).

**Reference inversion.** `tkd` divides by D where |D| exceeds a threshold and by the signed threshold elsewhere,
the form of Wang & Liu 2015, Eq. 13: "X_TKD(k) = sgn(D(k)) / max(|D(k)|, λ) · B(k)". It "leads to an underestimation
of the susceptibility values" (the same paper). The size of the underestimate follows from the angular mean of D
times its truncated inverse (`tkd_factor`, derived here; no source read in full states a number):

    factor(t) = 1 - (u2 - u1) + (1 / t) x integral from u1 to u2 of abs(1/3 - u^2) du,     u1,2 = sqrt(1/3 -+ t)

F42 reproduces the specification's figure on its lattice, F43 compares the lattice with the formula. Not built:
background-field removal, masks, any regularised inversion.

## 9. Validation table

Errors of a field are in units of the susceptibility step unless a unit is printed. The measured column is from
the machine that generated this page.

<!-- validation table: begin -->
### The dipole kernel against closed forms

| # | Quantity | Closed form or independent method | Error | Tolerance | |
|---|---|---|---|---|---|
| F1 | sphere, radius 24 mm (12 cells): field outside, from 3 mm off the surface to 3 radii, rms error | (1/3)(a/r)^3 (3 cos^2 t - 1); the specification reached 0.0004 (0.00022 padded) | 1.8e-04 | ≤ 2.5e-04 | pass |
| F2 | the same sphere: field inside, more than 3 mm from the surface, standard deviation (mean 1e-08) | 0 with the sphere of Lorentz: uniform | 0.001464 | ≤ 0.002 | pass |
| F3 | the specification's own lattice (edge of half-width 1 mm sampled at the cell centres, no padding): larger of the outside rms error / 0.0004 and the inside standard deviation / 0.0008 | the specification's figures 0.00038 and 0.00073 | 0.9568 of the bound | ≤ 1 | pass |
| F4 | the same sphere at four lattice points: on the B0 axis and on the equator at 1.5 and 2 radii | 0.19753, 0.08333 (along B0); -0.09877, -0.04167 (across) | 1.0e-04 | ≤ 0.001 | pass |
| F5 | the same sphere without the sphere of Lorentz: mean inside | 2/3 (Schenck: B inside = B0 (1 + 2 chi / 3)), less 2/3 of the volume fraction of the box, the mean that k = 0 removes | 6.1e-08 | ≤ 2.0e-04 | pass |
| F6 | infinite cylinder, radius 24 mm, parallel to B0, cross-section of the box 512 mm: larger of the error of the mean inside and the rms error outside (3 mm off the surface to 2.2 radii) | inside (3 cos^2 t - 1) / 6 = +0.33333, outside (1/2) sin^2 t (a/rho)^2 cos 2 phi, both plus the offset -(area fraction) x inside = -0.00230 | 1.5e-08 | ≤ 0.001 | pass |
| F7 | infinite cylinder, radius 24 mm, at 30 degrees to B0, cross-section of the box 512 mm: larger of the error of the mean inside and the rms error outside (3 mm off the surface to 2.2 radii) | inside (3 cos^2 t - 1) / 6 = +0.20833, outside (1/2) sin^2 t (a/rho)^2 cos 2 phi, both plus the offset -(area fraction) x inside = -0.00144 | 1.6e-04 | ≤ 0.001 | pass |
| F8 | infinite cylinder, radius 24 mm, at the magic angle, 54.74 degrees, cross-section of the box 512 mm: larger of the error of the mean inside and the rms error outside (3 mm off the surface to 2.2 radii) | inside (3 cos^2 t - 1) / 6 = +0.00000, outside (1/2) sin^2 t (a/rho)^2 cos 2 phi, both plus the offset -(area fraction) x inside = -0.00000 | 4.3e-04 | ≤ 0.001 | pass |
| F9 | infinite cylinder, radius 24 mm, perpendicular to B0, cross-section of the box 512 mm: larger of the error of the mean inside and the rms error outside (3 mm off the surface to 2.2 radii) | inside (3 cos^2 t - 1) / 6 = -0.16667, outside (1/2) sin^2 t (a/rho)^2 cos 2 phi, both plus the offset -(area fraction) x inside = +0.00115 | 6.4e-04 | ≤ 0.001 | pass |
| F10 | the same cylinder across B0 in an unpadded box (256 mm): the lattice minus the closed form outside is an offset plus the field of the images; fitted amplitude of (rho / L)^2 cos 2 phi over the area fraction | offset -(area fraction) x inside, exact for a square cell; images 1.505 x area fraction x sin^2 t (rho / L)^2 cos 2 phi (lattice sum of line dipoles, derived separately) | 0.04452 relative | ≤ 0.05 | pass |
| F11 | the same case after both terms: rms of what is left outside | 0 | 6.4e-04 | ≤ 0.001 | pass |
| F11b | the same cylinder across B0 in a cell of 320 x 512 mm, B0 along the short and along the long edge: constant inside against its formula (it is +0.00556 and -0.00188; the square-cell rule would give +0.00184 for both) | (area fraction) x [-(3 cos^2 t - 1) / 6 + (1/2) sin^2 t cos 2 alpha (s - 1)], s = (pi L2 / L1) [1/3 - 2 sum_n sinh^-2(pi n L2 / L1)], alpha the angle of B0 from the edge L1 (derived separately; s = 1 for a square) | 2.1e-08 | ≤ 1.0e-05 | pass |
| F12 | spherical shell, radii 16 and 24 mm (4 cells thick): larger of the rms error in the cavity (r < 13 mm), in the shell (19 to 21 mm) and outside (27 to 72 mm) | cavity 0; shell -(1/3)(a1/r)^3 (3 cos^2 t - 1); outside (1/3)((a2^3 - a1^3)/r^3)(3 cos^2 t - 1) | 0.003096 | ≤ 0.006 | pass |
| F13 | superposition and linearity: field of 2 chi1 - 3 chi2 against 2 field1 - 3 field2; and a constant added to chi and to its surroundings (double precision) | equal; no change (only differences of chi matter) | 4.4e-16 | ≤ 1.0e-12 | pass |
| F14 | B0 along x, and along y, against the object turned by 90 degrees with B0 along z (an object without symmetry, double precision) | equal: turning B0 is turning the object | 1.7e-16 | ≤ 1.0e-12 | pass |
| F15 | prolate spheroid (semi-axes 36 and 18 mm) at 30, 54.74 and 90 degrees to B0: mean inside, once with B0 turned and once with the object turned | 1/3 - N_c cos^2 t - N_a sin^2 t, N_c = 0.17356 (demagnetising factors) | 3.2e-05 | ≤ 0.0015 | pass |
| F16 | the same three angles: B0 turned against the object turned | equal | 3.8e-05 | ≤ 0.0015 | pass |
| F16b | B0 oblique to the lattice (along the body diagonal, a face diagonal and a general direction), the sphere off the lattice centre: larger of the rms error outside (3 mm off the surface to 3 radii, constant removed) / 0.0003 and the standard deviation inside / 0.002 | the closed form with that direction: as good as along an axis (F1, F2) | 0.6361 of the bound | ≤ 1 | pass |
| F16c | the same three directions with the textbook kernel, the plain product k_i k_j in the mixed terms: rms error outside | this row fails if the plain product is good enough: it jumps at the edge of the zone and the field rings | 0.001725 | ≥ 0.001 | pass |
| F17 | the sphere on cells of 3 x 2 x 1.5 mm (z, y, x): rms error outside, 3 mm off the surface to 3 radii | the closed form | 4.6e-04 | ≤ 6.0e-04 | pass |

### The periodic lattice: padding, spacing, cut faces

| # | Quantity | Closed form or independent method | Error | Tolerance | |
|---|---|---|---|---|---|
| F18 | padding: the sphere transformed at 128^3 (box 256 mm) against 320^3 (640 mm), within 3 radii: largest difference 8.1e-04; rms of (difference - image formula) over rms of the difference | images of a cubic cell: 6.216 (a / L)^3 (r / L)^2 (3 cos^2 t - 1) (lattice sum of dipoles, derived separately), for both box lengths | 0.0369 relative | ≤ 0.1 | pass |
| F19 | padding: the sphere transformed at 192^3 (box 384 mm) against 320^3 (640 mm), within 3 radii: largest difference 9.6e-05; rms of (difference - image formula) over rms of the difference | images of a cubic cell: 6.216 (a / L)^3 (r / L)^2 (3 cos^2 t - 1) (lattice sum of dipoles, derived separately), for both box lengths | 0.02219 relative | ≤ 0.25 | pass |
| F20 | padding 1.5 (the default) without the image term: largest image error of the sphere within 3 radii | bound 2 x 6.216 (a / L)^3 (3 a / L)^2 = 1.1e-4 at L = 384 mm | 9.6e-05 | ≤ 1.1e-04 | pass |
| F20b | the image term taken off (the default), a sphere off the centre of the lattice, against the transform at 320^3, within 3 radii, constant removed: largest difference without padding (4.2e-05 in place of 8e-4), with padding 1.5, and with padding 1.5 and B0 along the body diagonal; the larger of the last two | 0: the quadratic term of the images is (1/2) sum_src q (r - r')^T H (r - r'), H the lattice sum of second derivatives of the dipole field (``image_hessian``) | 8.6e-06 | ≤ 2.0e-05 | pass |
| F20c | that lattice sum for a cubic cell, B0 along z: H_zz L^5 against 4 x 6.216 x 3 / (4 pi) | 5.936 (the image formula of F18, derived separately) | 2.7e-04 relative | ≤ 0.002 | pass |
| F20d | a homogeneous ellipsoidal head (180 x 150 x 220 mm) in air on the specification's lattice, padding 1.5, at 3 T: standard deviation of the field over the brain; without the image term it is 2.16 Hz | 0: the field inside an ellipsoid is uniform | 0.07059 Hz | ≤ 0.3 | pass |
| F21 | convergence with the cell size: rms error of the sphere outside (8 mm off the surface to 2.5 radii), box 240 mm, at 4, 2, 1 mm = 2.3e-04, 4.9e-05, 1.4e-05; order | second order (cell means carry the exact dipole moment; what is left is the shape within a cell) | 2.001 | ≥ 1.5 | pass |
| F22 | a rod along B0 (radius 16 mm) that leaves the lattice through its lower face, continued through the padding (``extrude``): field along its axis from 37 mm above the face to 6 mm below its upper end, relative to the upper point, against a rod that ends 96 mm below the face | (c1 + c2) / 2 - 2/3, c = d / sqrt(d^2 + a^2) for the distances d to the two ends | 9.5e-04 | ≤ 0.003 | pass |
| F23 | the same rod cut at the face: largest error of that profile against the rod that goes on for ever; with the continuation it is 0.003 | a flat cut is an end face with the full surface charge: the closed form gives 0.032 | 0.03204 | ≥ 0.025 | pass |

### Surfaces and image planes

| # | Quantity | Closed form or independent method | Error | Tolerance | |
|---|---|---|---|---|---|
| F24 | an air sphere of 12 mm in tissue (a sinus): points in the tissue within 1 mm of the surface, rms error of the field at arbitrary positions (9.4 ppm at 3 T: x 1200 Hz) | the closed form on the tissue side | 0.02073 | ≤ 0.03 | pass |
| F24b | the same points with plain interpolation of the lattice field (no step taken out) | the error the specification found: the step smeared into the tissue; this row fails if it is small | 0.1056 | ≥ 0.08 | pass |
| F25 | a tissue sphere of 24 mm in air: points inside within 1 mm of the surface, rms error of the field at arbitrary positions (9.4 ppm at 3 T: x 1200 Hz) | the closed form on the tissue side | 0.007355 | ≤ 0.012 | pass |
| F25b | the same points with plain interpolation of the lattice field (no step taken out) | the error the specification found: the step smeared into the tissue; this row fails if it is small | 0.0973 | ≥ 0.08 | pass |
| F26 | the air sphere of 12 mm: points in the tissue 2 to 8 mm from the surface, rms error | the closed form | 0.001588 | ≤ 0.0025 | pass |
| F27 | ``sample`` on an axial plane 8 mm above an air sphere of 12 mm at 3 T (peak 173 Hz), grid of 0.4 mm from the 2 mm lattice: largest error of the field | the closed form; trilinear interpolation alone allows h^2 / 8 x the second derivative, 9 Hz at the peak | 2.546 Hz | ≤ 10 | pass |
| F28 | the same plane: through-plane gradient over a slice of 5 mm (peak 27 Hz/mm), largest error | the secant of the closed form across the slice | 0.9685 Hz/mm | ≤ 2 | pass |
| F29 | ``sample`` on a coronal plane 20 mm from the centre of the sphere: largest error | the closed form with the same direction of B0 | 1.273 Hz | ≤ 10 | pass |
| F30 | ``sample`` on a sagittal plane 20 mm from the centre of the sphere: largest error | the closed form with the same direction of B0 | 1.273 Hz | ≤ 10 | pass |
| F31 | ``sample`` on an axial plane, B0 along y, 20 mm from the centre of the sphere: largest error | the closed form with the same direction of B0 | 1.273 Hz | ≤ 10 | pass |
| F32 | one slice imaged by the engine from master grids of 0.8 and 0.4 mm (spoiled gradient echo, TE 6 ms, matrix 64; a tissue ball with an air ball 6 mm under the slice): rms difference of the two images | the specification's acceptance: at most 1 % (its first version: 1.7 % to 3.4 %) | 6.7e-04 relative | ≤ 0.01 | pass |
| F32b | the same pair of images with plain interpolation of the lattice field | above the 1 % of the acceptance: this row fails if the check above cannot see the smeared step | 0.02616 relative | ≥ 0.01 | pass |

### Shim

| # | Quantity | Closed form or independent method | Error | Tolerance | |
|---|---|---|---|---|---|
| F33 | a pure first-order field and a pure second-order field over an ellipsoidal volume off the isocentre: largest residual and largest error of a coefficient, relative | removed to rounding | 1.0e-15 | ≤ 1.0e-11 | pass |
| F34 | Laplacian of the nine basis functions (second differences at 200 points) | 0: a field made by currents outside the volume is harmonic | 3.6e-08 mm^-2 x term | ≤ 1.0e-06 | pass |
| F35 | a field that needs 2.5 times the limit of the Z2 channel: the bounded fit puts Z2 at its limit and no other channel; largest violation of the optimality conditions (the gradient is zero for a free channel and points outwards at a limit) | 0 (Karush-Kuhn-Tucker) | 2.2e-16 relative | ≤ 1.0e-09 | pass |
| F36 | the same fit: smallest sum of squares among 200 feasible points drawn around the solution, over that of the solution | at least 1 | 1.011 ratio | ≥ 1 | pass |
| F37 | first order against the engine: the fitted shim of a brain world (``world.anatomy_fields``: constant, y, x over the brain, and the mean through-slice gradient) against this module's fit of the same closed-form field on the planes z = -1, 0, 1 mm: largest relative difference of the four coefficients | equal: the engine's shim is a first-order fit over a thin slab | 3.2e-15 relative | ≤ 1.0e-09 | pass |
| F38 | an object that does not change along z (an elliptic cylinder with an air cylinder, B0 across them; field over the region 45.9 Hz sd): first-order shim over the volume on the 2 mm lattice against the engine's fit on the slice's 0.4 mm grid: rms difference of the two shimmed fields over the sd after shim (43.8 Hz); the shim's gradient along z is 1e-15 Hz/mm | the same least-squares problem on two samplings of one region | 0.004308 relative | ≤ 0.02 | pass |

### Inclusions, profiles, susceptibility mapping, time

| # | Quantity | Closed form or independent method | Error | Tolerance | |
|---|---|---|---|---|---|
| F39 | a sphere of radius 2 cells on the lattice (cell means): rms error outside, from one cell off the surface to three radii further, over the rms of the field there (1 cell: 0.070; 4 cells: 0.019) | the rule: at least 2 cells of radius on the lattice (``MIN_LATTICE_RADIUS_CELLS``), smaller in closed form | 0.02948 relative | ≤ 0.04 | pass |
| F40 | a closed-form inclusion (a sphere of 1.5 mm, 3 ppm) added to a lattice field: the field with it minus the field without, at 5000 points around and inside it | the exact dipole outside, 0 inside: added once, in ppm | 8.7e-19 ppm | ≤ 1.0e-05 | pass |
| F41 | declared transmit and receive profiles on the plane z = 0 against the engine's brain world of the same seed (B1+ ratio, complex receive profile) | equal: the 3D shapes continue the 2D ones | 1.8e-15 | ≤ 1.0e-12 | pass |
| F42 | truncated k-space division at threshold 0.1 on the specification's sphere: mean recovered susceptibility inside against the specification's 0.905 | 0.905 (true value 1: the underestimate) | 4.7e-04 | ≤ 0.003 | pass |
| F43 | the same inversion at thresholds 0.1 and 0.2 on cell means: recovered susceptibility weighted with the sphere against the angular mean of D x its truncated inverse (0.9129, 0.8224) | ``tkd_factor``: 1 - (u2 - u1) + (1 / t) x integral of abs(1/3 - u^2) between u1,2 = sqrt(1/3 -+ t); the lattice is lower by the k = 0 term (the volume fraction, 0.003) and by the few directions a 128-cell box has at small k | 0.008214 | ≤ 0.01 | pass |
| F44 | multi-echo phase: slope over the echo times / 2 pi against the field; wrapped phase in (-pi, pi]; a positive field gives a positive phase | phase = 2 pi f TE, the sign of the engine | 5.7e-14 Hz or rad | ≤ 1.0e-09 | pass |
| F45 | time to build the field of a world on the specification's lattice (136 x 128 x 128 cells of 2 mm, transformed at 216 x 192 x 192), one thread; one plane on the 0.4 mm grid then takes 0.12 s | at most 0.8 s | 0.4385 s | ≤ 0.8 | pass |
<!-- validation table: end -->

## 10. A head, against the engine's declared field and against published field maps

The head is three nested ellipsoids (soft tissue, a skull shell, soft tissue) with the three air spheres that the
engine's `brain` world of one seed declares (a frontal sinus and two ear regions below the slice), cut out of it.
Susceptibilities: air +9.4 ppm, skull -2.1 ppm (Koch 2009 uses "bone (χ≈−11.3 ppm)" against tissue at −9.2;
Schenck 1996 quotes two values for cortical bone, "−8.86×10⁻⁶" and "−12.82×10⁻⁶", on either side of water). The
brain is the inner ellipsoid above a floor 10 mm below the slice.

<!-- head figures: begin -->
Head: extent 179 mm (y) x 140 mm (x) x 220 mm (z); frontal air sphere of radius 12.0 mm, its top 14.7 mm below the slice z = 0; seed `0123456789abcdef0123456789abcdef`.

| Quantity at 3 T | Dipole lattice: head of nested ellipsoids with the three air spheres | The three declared spheres alone (the engine today) |
|---|---|---|
| Field 4 / 8 / 12 / 20 mm above the top of the frontal sphere, before shim (lattice: relative to the brain mean) | +343 / +174 / +100 / +39 Hz | +338 / +173 / +100 / +42 Hz (that sphere's closed form) |
| The same points after a first-order shim over the whole brain; after a second-order shim | +339 / +170 / +96 / +36 Hz; +335 / +166 / +91 / +30 Hz | - |
| Whole brain before shim: standard deviation; range about the mean | 8.9 Hz; -11 to +214 Hz | one slice only |
| Whole brain after a first-order shim: standard deviation; largest; share of cells beyond 20 Hz | 8.8 Hz; 210 Hz; 2.2 % | - |
| Whole brain after a second-order shim (channels at a limit of `SHIM_LIMITS_UT`: none) | 8.4 Hz; 205 Hz; 2.0 % | no second order |
| Slice z = 0, the brain of both models, before shim: standard deviation | 13.0 Hz | 12.3 Hz |
| The same slice after the engine's shim (constant, y, x fitted on the slice): standard deviation; largest; share beyond 20 Hz | 12.6 Hz; 85 Hz; 7.5 % | 12.0 Hz; 78 Hz (with the engine's smooth random term: 11.8 Hz; 78 Hz; 6.9 %) |
| The two shimmed slice fields against each other: rms difference; correlation | 0.7 Hz; 1.000 | - |
| Through-slice gradient on the slice, mean removed: standard deviation | 1.09 Hz/mm | 1.04 Hz/mm |
| Whole brain, the same head without air: field of the outer shape alone (an ellipsoid: uniform in theory, so this is the error of the lattice); added by the skull shell (standard deviations) | 0.06 Hz; 0.08 Hz | absent |
| The 2 mm lattice against a 1 mm lattice on the slice z = 0 (brain, constant removed): rms; largest | 0.08 Hz; 1.7 Hz | - |

The seeded 3D head (`head3d` of `volume.py`, seed `0123456789abcdef0123456789abcdef`, 3 T, 2 mm lattice of cell means); matter covers 28 % of the lower face of its lattice:

| Whole brain, standard deviation (largest) | before shim | after a first-order shim | after a second-order shim |
|---|---|---|---|
| neck cut at the lower face of the lattice | 25.5 Hz | 10.3 Hz (153 Hz) | 7.0 Hz (168 Hz) |
| neck continued through half the padding (`extrude`), padding 1.5 | 26.8 Hz | 17.6 Hz (114 Hz) | 6.3 Hz (160 Hz) |
| the same, padding 2 (the continuation is longer) | 32.8 Hz | 15.1 Hz (119 Hz) | 6.1 Hz (157 Hz) |

The first two fields differ over the brain by 10.4 Hz (standard deviation, before shim).

Scale: 1 ppm is 127.7 Hz at 3 T; the largest step at an air surface, (2/3) x 9.4 ppm, is 800 Hz. Published whole-brain standard deviations at 7 T, scaled to 3 T by the field strength: 51.9 Hz x 3 / 7 = 22.2 Hz (Stockmann & Wald 2018, one subject, first and second order); 29.72 Hz x 3 / 7 = 12.7 Hz (the same paper, eight subjects (Kim 2016)); 43 Hz x 3 / 7 = 18.4 Hz (Juchem 2011, orders 0 to 3, computed).
<!-- head figures: end -->

**Against the engine today.** On the slice the two fields are the same field: the rows "against each other" give
their difference. That is expected. The field inside a homogeneous ellipsoid is uniform, so an ellipsoidal head
adds nothing, and the dipole lattice then computes what the three closed-form spheres already state. The lattice
differs from the declared field in what it can take as input: cavities that are not spheres, a head that is not an
ellipsoid (face, neck, shoulders), layers of bone and fat, and any slice position and orientation, where the
declared field exists on one axial slice.

**The seeded 3D head** has these: the second table gives its field over the brain. Two things differ from the
ellipsoid head. A shim now gains, because the shape of the head below the brain adds a smooth term. And the result
depends on what is assumed below the lattice: the neck leaves through the lower face, and cutting it there,
continuing it, or continuing it further changes the field over the brain by more than any numerical error on this
page (the line under the table, and the first-order column). Most of that difference is smooth and a second-order
shim removes it. What lies below the head (shoulders, chest) is not modelled; it is the largest uncertainty of the
field of a head here.

**Against field maps in vivo.** Published magnitudes, each at its own field strength:

| Source | Statement |
|---|---|
| Koch 2009 [F] | "Within both the human and rodent brains at B0 field strengths ≥3T it is not uncommon to encounter B0 offsets over 200 Hz." At 4 T, after shimming up to second order: "Field inhomogeneity of up to 150 Hz can readily be encountered a few centimeters into the brain." |
| Stockmann & Wald 2018 [F] | 7 T: "(γ/2π)ΔB0 = 300 Hz (∼ 1 ppm), near the most inhomogeneous B0 area of the prefrontal cortex"; a whole-brain standard deviation of "51.9 Hz" after first- and second-order shim in one subject and "29.72 Hz" in a group of eight; "approximately twice as large at 7 T compared to 3 T" |
| Juchem 2011 [F] | 7 T, static shim up to third order (computed): standard deviation "43 Hz"; "the induced magnetic field distortions scale linearly with the scanner B0 field strength" |
| Truong 2014 [F] | 3 T, after second-order shim of the whole brain: root-mean-square error in the anterior half of the brain 48.4 Hz ("from 48.4 to 19.2 Hz" with local coils) |

Read against the first table and the scale line above. The published peak of about 1 ppm near the frontal sinus is
what this head shows at the middle two of the four heights above the air sphere; at the lowest height it is larger
than the figures quoted, because a real sinus is separated from the brain by bone and is not a sphere of this
size. The whole-brain standard deviation after a second-order shim is below the in-vivo figures scaled to 3 T, for
the ellipsoid head and for the seeded head alike, and a second-order shim gains little on the ellipsoid head, where
in vivo it gains visibly (Koch 2009, Fig. 8, qualitative): that head has three compact sources and no smooth
large-scale term. No whole-brain figure at 3 T after a first- or second-order shim was found in a source read in
full.

## 11. What the slice engine calls

    L = vol.lattice(2.0, supersample=2)                                        # cell means, all classes of vol.matter
    lattice = fields3d.Lattice(L.weights.shape[1:], L.spacing_mm, L.origin_mm)
    chi = [vol.susceptibility_ppm[k] for k in vol.matter]
    field = fields3d.from_weights(L.weights, chi, lattice, b0_t, extrude=("z-",))   # once per world
    field.shim(vol.organ_mask(L.weights), order=1)                             # once per world (or per task)
    seen = fields3d.seen_chi(plane_weights, [vol.susceptibility_ppm[k] for k in vol.names])   # per plane (vol.eval_plane)
    f = fields3d.sample(field, frame, grid, fov, thickness, chi_seen=seen)     # per plane
    world["fields"]["b0_hz"], world["fields"]["dbdz_hz_per_mm"] = f["b0_hz"], f["dbdz_hz_per_mm"]
    p = fields3d.declared_profiles("brain", b0_t, size_mm, rng, rng_z).on_plane(frame, grid, fov)   # b1, rx

Time: F45. A `Field3D` holds six float32 lattices of the size of the lattice.

## 12. Limits and what was not verified

- First order in chi; isotropic susceptibility; no chemical exchange, no microstructure. The field of the magnet
  itself and of the gradient coils is not here.
- The surface split is a plane approximation; its error near a curved surface is the rows F24 to F26, on spheres
  only. Edges, thin plates and tubes of a few cells were not measured.
- Interfaces between tissues are sharp only when the caller passes `chi_seen`.
- The lattice form of the mixed kernel terms for an oblique B0 is a choice (the simplest odd periodic function with
  the right limit). It was checked on a sphere for three directions and on a spheroid for three angles.
- The image term is the leading one. It was checked on a sphere and on an ellipsoid; an object that fills most of
  its box needs more padding.
- The continuation through a face (`extrude`) was checked on a straight rod along B0 only. Its length is half the
  padding on that side, so the field of an extruded head depends on the padding (section 10); the image term is
  not applied to it.
- The image term was checked for B0 along a cell axis and along the body diagonal of a cubic cell (F20b), not for a
  cell with three different edges.
- The declared shim limits, the skull susceptibility and the z dependence of the transmit and receive shapes are
  choices of this package.
- The head of section 10 is one seed and is schematic. No comparison with a measured field map was made.
- The closed forms were re-derived in a separate script that shares no code with this package (surface-charge
  quadrature and a real-space dipole sum, no k-space kernel); `tests/test_fields3d.py` holds its values.

## 13. Sources

| Source | Tag | Used for |
|---|---|---|
| R. Salomir, B. D. de Senneville, C. T. W. Moonen. A fast calculation method for magnetic field inhomogeneity due to an arbitrary distribution of bulk susceptibility. *Concepts in Magnetic Resonance Part B* 19B(1):26-34, 2003. doi:10.1002/cmr.b.10083 | S | the first-order range; attribution of the kernel |
| J. P. Marques, R. Bowtell. Application of a Fourier-based method for rapid calculation of field inhomogeneity due to spatial variation of magnetic susceptibility. *Concepts in Magnetic Resonance Part B* 25B(1):65-78, 2005. doi:10.1002/cmr.b.20034 | F | the kernel, k = 0, the sphere of Lorentz, the buffer, where the errors sit |
| J. F. Schenck. The role of magnetic susceptibility in magnetic resonance imaging: MRI magnetic compatibility of the first and second kinds. *Medical Physics* 23(6):815-850, 1996. doi:10.1118/1.597854 | F (scanned copy; pages 815 to 840, 842, 843) | susceptibility values; closed forms of sphere and cylinder without the Lorentz term |
| J. G. Bouwman, C. J. G. Bakker. Alias subtraction more efficient than conventional zero-padding in the Fourier-based calculation of the susceptibility induced perturbation of the magnetic field in MR. *Magnetic Resonance in Medicine* 68(2):621-630, 2012. doi:10.1002/mrm.24343 | S | padding by two |
| K. M. Koch, D. L. Rothman, R. A. de Graaf. Optimization of static magnetic field homogeneity in the human and animal brain in vivo. *Progress in Nuclear Magnetic Resonance Spectroscopy* 54(2):69-96, 2009. doi:10.1016/j.pnmrs.2008.04.001 | F (author manuscript) | shim functions, least squares with limits, susceptibilities, in-vivo magnitudes |
| Y. Wang, T. Liu. Quantitative susceptibility mapping (QSM): decoding MRI data for a tissue magnetic biomarker. *Magnetic Resonance in Medicine* 73(1):82-101, 2015. doi:10.1002/mrm.25358 | F | kernel, truncated division, its underestimate |
| K. Shmueli et al. Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data. *Magnetic Resonance in Medicine* 62(6):1510-1522, 2009. doi:10.1002/mrm.22135 | F (author manuscript) | truncation of the inverse filter; sign of the phase |
| QSM Consensus Organization Committee. Recommended implementation of quantitative susceptibility mapping for clinical research in the brain: a consensus of the ISMRM Electro-Magnetic Tissue Properties Study Group. *Magnetic Resonance in Medicine* 91(5):1834-1862, 2024. doi:10.1002/mrm.30006 | F (author manuscript) | sign conventions |
| J. P. Stockmann, L. L. Wald. In vivo B0 field shimming methods for MRI at 7 T. *NeuroImage* 168:71-87, 2018. doi:10.1016/j.neuroimage.2017.06.013 | F | in-vivo magnitudes at 7 T |
| C. Juchem et al. Dynamic multi-coil shimming of the human brain at 7 Tesla. *Journal of Magnetic Resonance* 212(2):280-288, 2011. doi:10.1016/j.jmr.2011.07.005 | F | in-vivo magnitudes at 7 T, scaling with B0 |
| T.-K. Truong, D. Darnell, A. W. Song. Integrated RF/shim coil array for parallel reception and localized B0 shimming in the human brain. *NeuroImage* 103:235-240, 2014. doi:10.1016/j.neuroimage.2014.09.052 | F | residual at 3 T after second-order shim |
| J. A. Osborn. Demagnetizing factors of the general ellipsoid. *Physical Review* 67(11-12):351-357, 1945. doi:10.1103/PhysRev.67.351 | S (record only) | the integral for the demagnetising factors is the one this paper is known for; it is checked numerically (F15) and against the separate derivation, not against the paper |
