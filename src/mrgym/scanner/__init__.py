"""The virtual scanner.

``world``    the hidden world: anatomy (``anatomy``), tissue table (``tissue``), fields and system state
``console``  protocol in; realised protocol, timing, scan time, limits and header out (never signal)
``signal``   closed-form signal models; ``epg`` the RF-spoiled steady state with finite T2 and diffusion
``rf``       the slice-selective pulse: its rotation at every height through the slice, its energy and peak
``kspace``   the engine: world x realised protocol -> k-space -> images
``session``  one episode: budget, hash-chained log, replay
``service``  HTTP front end, one process per episode; ``client`` a single-file agent-side client
"""
from .kspace import Engine
from .session import ENGINE_ID, ENGINE_VERSION, LocalScanner, ScannerError, ScannerSession, replay
from .world import make_world

__all__ = ["ENGINE_ID", "ENGINE_VERSION", "Engine", "LocalScanner", "ScannerError", "ScannerSession", "make_world", "replay"]
