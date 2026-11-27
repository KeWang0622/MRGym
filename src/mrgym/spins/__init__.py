"""Coupled proton spin systems: a density-matrix simulator and the spectroscopy built on it.

A library: it serves no task and no tool yet, holds no state and does not change the scanner engine.

    operators    spin systems, equivalent groups as total spins, the Hamiltonian in its eigenbasis
    evolve       blocks (rf, rf_pulse, delay, select, gradient, average, acquire), propagation, lines, spectrum
    sequences    pulse-acquire, spin echo, PRESS, STEAM, semi-LASER, MEGA-PRESS, J-resolved, COSY; relaxation factors
    metabolites  a basis of spin systems with the source of every row
    quantify     linear-combination fit, Cramer-Rao bounds, water-referenced concentration, line width, SNR
    validate     ``python -m mrgym.spins.validate``: every function against a closed form or an independent computation

Example: the PRESS spectrum of lactate at 3 T, echo time 144 ms.

    from mrgym.spins import evolve, metabolites, sequences
    lines = sequences.simulate(metabolites.get("lactate"), sequences.press(144.0), b0_t=3.0)
    ppm, spec = evolve.spectrum(lines.fid(2048, 0.5, t2_ms=200.0), 0.5, b0_t=3.0)

Units and conventions: the module docstrings of ``operators`` and ``evolve``, and docs/physics/spins.md.
"""
