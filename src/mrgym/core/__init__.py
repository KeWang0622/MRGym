"""One signal core: any sequence written as a list of blocks, evaluated from first principles.

``blocks``     the vocabulary: rf, delay, gradient, spoil, crush, mz, adc, repeat; plain data
``operators``  the spins, the conventions and the numbers of every operator (shared by the evaluators)
``bloch``      an isochromat ensemble integrating the Bloch equation: slow, the reference
``epg``        extended phase graphs stepped in time, for all columns at once; diffusion, two pools
``periodic``   the steady state of a periodic block list as the fixed point of one period, and the approach to it
``library``    sequences as data: the six the scanner serves and eleven it does not
``validate``   every number against a closed form or an independent computation: ``python -m mrgym.core.validate``

Nothing here feeds the scanner yet: ``mrgym.scanner`` is imported by ``validate`` only, to compare.
Principles, conventions and the validation table: docs/physics/core.md.
"""
from . import bloch, blocks, epg, library, operators, periodic
from .blocks import Sequence
from .operators import Spins, grid, spins

__all__ = ["Sequence", "Spins", "blocks", "bloch", "epg", "grid", "library", "operators", "periodic", "spins"]
