"""Signal physics of the virtual scanner (server side only; never shipped to the agent).

Tiers on one world definition (``world.py``):
  tier 1  ``closed``  textbook closed forms under their stated idealisations (ideal spoiling, hard pulses)
  tier 2  ``epg``     extended phase graph steady state of RF-spoiled gradient echo with finite T2
  check   ``bloch``   independent matrix / isochromat Bloch calculations used only by the consistency suite

Units inside ``phys``: times in ms, angles in rad unless a name ends in ``_deg``, frequencies in Hz, positions in
units of the field of view (see ``mrfrl.sim``). Signals are transverse magnetisation per unit M0 right after the
excitation pulse (TE = 0), demodulated with the phase of that pulse, real and positive for an on-resonance spin.
"""
