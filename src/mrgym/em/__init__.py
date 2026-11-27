"""Radio-frequency fields on the slice, from Maxwell's equations.

Time-harmonic Maxwell equations in two dimensions on the axial slice, everything invariant along z (an infinitely
long body in an infinitely long coil): E along z, B in the plane. This gives the transmit field B1+, the receive
field B1-, wavelength effects, conductive losses and SAR from the permittivity and conductivity of the tissues. It
leaves out the end rings, the finite length of body and coil and every z-directed field. A library: no task, no
tool, no state, and the scanner engine (``mrgym.scanner``) is not changed; it is read for its worlds only.

``units``       constants, the phasor convention exp(+i w t), wavenumber and wavelength
``properties``  permittivity, conductivity and density of the engine's tissue classes against frequency
``analytic``    closed forms: a layered lossy cylinder in a birdcage current sheet or among line currents
``solver``      finite-difference frequency-domain solver on the pixel grid; powers, SAR maps, a world's problem
``coils``       birdcage and transmit arrays as line currents; reception by reciprocity, noise matrix, g-factor
``shim``        RF shimming under limits on power and local SAR; the trade-off curve
``ept``         conductivity and permittivity back from B1+ (Helmholtz relation), and where that fails
``validate``    every function against a closed form or an independent computation: ``python -m mrgym.em.validate``

Conventions and the validation table: docs/physics/em.md.
"""
from . import analytic, coils, ept, properties, shim, solver, units
from .solver import Solver, WorldModel

__all__ = ["Solver", "WorldModel", "analytic", "coils", "ept", "properties", "shim", "solver", "units"]
