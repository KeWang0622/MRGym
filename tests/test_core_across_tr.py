"""The periodic state of a spin-echo repetition with what survives to the next excitation.

Textbook statements, each against a closed form or the isochromat ensemble:
  1. A 90 - 180 spin echo with exact pulses repeated every TR: (1 - 2 exp(-(TR - TE/2)/T1) + exp(-TR/T1)) exp(-TE/T2)
     when T2 << TR - TE (nothing transverse reaches the next excitation), with real crushers and a real spoiler.
  2. A CPMG train of n exact 180 degree pulses at a TR that is short against T1: echo j = Mz exp(-j ESP / T2) with
     Mz = 1 - 2 sum_j (-1)^(n-j) exp(-(TR - (j - 1/2) ESP)/T1) - (-1)^n exp(-TR/T1): every 180 inverts what has regrown
     (with u = 1 - Mz: free evolution multiplies u by exp(-t/T1), an inversion makes it 2 - u, and u = 1 after the 90).
  3. Refocusing below 180 degrees stores a grating along z; identical crushers recall it in the NEXT repetition. The
     periodic state with the gratings carried equals the isochromat ensemble stepped to its steady state, and it is not
     the state behind an ideal crusher.
  4. For TR >> T1 the two agree: the gratings have relaxed.
  5. A fluid (T2 comparable to TR): transverse magnetisation reaches the next excitation through a real spoiler.
"""
import math

import numpy as np
import pytest

from mrgym.core import bloch, epg, library, operators, periodic
from mrgym.core import blocks as B


def _steady_ensemble(seq_of, sp, reps, n_sub=None):
    """Samples of the last repetition of ``reps`` repetitions stepped from equilibrium by the isochromat ensemble."""
    res = bloch.run(seq_of(reps), sp, n_sub=n_sub)
    n = len(res.labels) // reps
    return res.signal[-n:]


def test_spin_echo_with_incomplete_recovery_closed_form():
    t1, t2, tr, te = 900.0, 40.0, 700.0, 30.0
    sp = operators.spins(t1, t2)
    seq = library.se(tr, te, crusher_cycles=1, tail=library.Tail(cycles=1, ms=1.0, after_ms=2.0))
    got = periodic.steady_state(seq, sp, method="carry", carry=(3,)).signal[0, 0]
    want = (1 - 2 * math.exp(-(tr - te / 2) / t1) + math.exp(-tr / t1)) * math.exp(-te / t2)
    assert abs(got - want) < 1e-6 * want                 # exp(-(TR - TE) / T2) = 5e-8 is what T2 << TR leaves out
    full = periodic.steady_state(seq, sp, method="carry", carry=(3,), carry_transverse=True).signal[0, 0]
    assert abs(full - got) < 1e-6


def test_cpmg_train_at_short_tr_closed_form():
    t1, t2, esp, n, tr = 1200.0, 25.0, 8.0, 6, 400.0
    sp = operators.spins(t1, t2)
    seq = library.fse(tr, esp, n, 180.0, tail=library.Tail(cycles=1, ms=1.0, after_ms=1.5))
    got = periodic.steady_state(seq, sp, method="carry", carry=(4,)).signal[:, 0]
    mz = 1 - 2 * sum((-1) ** (n - j) * math.exp(-(tr - (j - 0.5) * esp) / t1) for j in range(1, n + 1)) - (-1) ** n * math.exp(-tr / t1)
    want = mz * np.exp(-np.arange(1, n + 1) * esp / t2)
    assert np.abs(got - want).max() < 1e-9


@pytest.mark.parametrize("refocus,b1", [(150.0, 1.0), (120.0, 0.9), (180.0, 0.8)])
def test_gratings_carried_equal_the_ensemble_and_differ_from_the_ideal_crusher(refocus, b1):
    t1, t2, esp, n, tr = 60.0, 30.0, 6.0, 4, 70.0       # T1 short: the ensemble arrives in 40 repetitions
    sp = operators.spins(t1, t2, b1=b1)
    tail = library.Tail(cycles=1, ms=1.0, after_ms=1.0)
    make = lambda reps: library.fse(tr, esp, n, refocus, excitation_deg=80.0, tail=tail, repetitions=reps)      # noqa: E731
    want = _steady_ensemble(make, sp, 40, n_sub=192)[:, 0]
    got = periodic.steady_state(make(1), sp, method="carry", carry=(28,), carry_transverse=True).signal[:, 0]           # 12 orders: 2e-7; 20: 9e-11; 28: 5e-14
    assert np.abs(got - want).max() < 1e-9
    ideal = periodic.steady_state(library.fse(tr, esp, n, refocus, excitation_deg=80.0), sp).signal[:, 0]
    assert np.abs(ideal - want).max() > 1e-3            # what the ideal crusher leaves out is not small here


def test_long_tr_is_the_ideal_crusher():
    t1, t2, esp, n = 300.0, 60.0, 8.0, 5
    sp = operators.spins(t1, t2, b1=0.85)
    tail = library.Tail(cycles=1, ms=1.0, after_ms=1.0)
    for tr, bound in ((12 * t1, 1e-4), (30 * t1, 1e-11)):
        got = periodic.steady_state(library.fse(tr, esp, n, 140.0, tail=tail), sp, method="carry", carry=(10,), carry_transverse=True).signal
        ideal = periodic.steady_state(library.fse(tr, esp, n, 140.0), sp).signal
        assert np.abs(got - ideal).max() < bound


def test_fluid_transverse_magnetisation_reaches_the_next_excitation():
    t1, t2, esp, n, tr = 90.0, 80.0, 6.0, 3, 60.0
    sp = operators.spins(t1, t2, b1=0.9)
    tail = library.Tail(cycles=1, ms=1.0, after_ms=1.0)
    make = lambda reps: library.fse(tr, esp, n, 160.0, tail=tail, repetitions=reps)       # noqa: E731
    want = _steady_ensemble(make, sp, 60, n_sub=256)[:, 0]
    got = periodic.steady_state(make(1), sp, method="carry", carry=(32,), carry_transverse=True).signal[:, 0]
    assert np.abs(got - want).max() < 1e-9
    z_only = periodic.steady_state(make(1), sp, method="carry", carry=(32,), carry_transverse=False).signal[:, 0]
    assert np.abs(z_only - want).max() > 1e-3           # the gratings alone are not the whole of it


def test_carry_converges_to_the_matrix_solve_of_all_states():
    sp = operators.grid([dict(t1_ms=400.0, t2_ms=70.0, adc_um2_ms=1.0), dict(t1_ms=2500.0, t2_ms=900.0, adc_um2_ms=3.0)], (0.0,), (0.8, 1.0))
    tail = library.Tail(cycles=1, ms=1.0, after_ms=1.0)
    seq = library.fse(500.0, 10.0, 4, 150.0, crusher_ms=1.0, pulse_ms=3.0, tail=tail, cycle_mm=1.0)
    full = periodic.steady_state(seq, sp, method="matrix", kmax=60).signal
    err = [np.abs(periodic.steady_state(seq, sp, method="carry", carry=(k,), carry_transverse=True).signal - full).max() for k in (2, 6, 14)]
    assert err[0] > err[1] > err[2] and err[2] < 1e-9


def test_two_axes_spoiler_perpendicular_to_the_diffusion_lobes():
    sp = operators.grid([dict(t1_ms=80.0, t2_ms=60.0, adc_um2_ms=2.0)], (0.0,), (0.85,))
    tail = library.Tail(cycles=1, ms=1.0, after_ms=1.0, axis=1)
    make = lambda reps: library.dwse(55.0, 20.0, 20.0, 3.0, 9.0, 0.5, 1.0, repetitions=reps, tail=tail, tail_cycle_mm=1.0)    # noqa: E731
    seq = make(1)
    assert len(seq.cycle_mm) == 2
    full = periodic.steady_state(seq, sp, method="matrix", kmax=(8, 8)).signal
    got = periodic.steady_state(seq, sp, method="carry", carry=(4, 5), carry_transverse=True).signal
    assert np.abs(got - full).max() < 1e-7
    assert np.abs(periodic.steady_state(make(1), sp, method="carry", carry=(0, 0)).signal - full).max() > 1e-4


def test_prune_to_a_box_keeps_what_can_return_to_it():
    sp = operators.spins(700.0, 90.0, b1=0.9)
    seq = library.fse(300.0, 8.0, 6, 150.0, tail=library.Tail(cycles=1, ms=1.0, after_ms=1.0))
    prog = B.compile_ops(seq)
    a = epg.run(prog, sp, keep_state=True)
    b = epg.run(prog, sp, keep_state=True, prune=(3,))
    assert np.abs(a.signal - b.signal).max() < 1e-15
    k = a.state.kmax[0]
    box = slice(k - 3, k + 4)
    kb = b.state.kmax[0]
    assert np.abs(a.state.f[box] - b.state.f[slice(kb - 3, kb + 4)]).max() < 1e-15
    assert np.abs(a.state.z[box] - b.state.z[slice(kb - 3, kb + 4)]).max() < 1e-15
