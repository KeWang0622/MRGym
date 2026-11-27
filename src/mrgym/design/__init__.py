"""Pulse-sequence design, stage 1: the physics layer.

Simulates a designed RF pulse, gradient program or sequence and checks it against hardware limits. A library: no
task, no tool, no state. The engine (``mrgym.scanner``) is not changed and is imported only for its limits
(``gradients.Hardware.console``) and, in ``validate``, for agreement checks.

``rotation``   one general RF rotation: any B1(t) under any gradient, off-resonance, B1 scale, relaxation
``pulses``     generators: windowed sinc, Shinnar-Le Roux, hyperbolic secant, control points, variable rate
``profile``    figures of merit of a slice profile, a profile mask, lower bounds on RF energy and duration
``sequence``   time-stepping extended phase graph over a typed block list
``gradients``  gradient programs: moments, k-space, b-value, limits per raster sample, the shortest lobe
``store``      immutable design records under content-derived ids
``validate``   every function against a closed form or an independent computation: ``python -m mrgym.design.validate``

Units and conventions: ``units``, ``rotation`` (module docstrings) and docs/physics/design.md.
"""
from . import gradients, profile, pulses, rotation, sequence, store, units
from .gradients import Hardware
from .pulses import Pulse
from .store import DesignStore

__all__ = ["DesignStore", "Hardware", "Pulse", "gradients", "profile", "pulses", "rotation", "sequence", "store", "units"]
