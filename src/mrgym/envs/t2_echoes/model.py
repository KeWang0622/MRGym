"""The echo-train model of ``Scanner/T2Echoes``: the echoes of a multi-echo spin echo with one refocusing angle.

A train of refocusing pulses of angle theta, ``esp`` apart, with equal crushers either side of every pulse (the
Carr-Purcell-Meiboom-Gill train of the console). Between two pulses the magnetisation of a voxel is a sum of states
that the crushers have dephased by a whole number k of cycles: transverse ones dephasing (F+_k), transverse ones
rephasing (F-_k), and longitudinal gratings (Z_k). One echo period is

    half an echo spacing     F x exp(-esp / (2 T2)),  Z x exp(-esp / (2 T1))
    a crusher                F+_k -> F+_(k+1),  F-_k -> F-_(k-1);  F-_1 becomes the state without dephasing, F_0
    the pulse                F+' = cos^2(theta/2) F+ + sin^2(theta/2) F- + sin(theta) Z
                             F-' = sin^2(theta/2) F+ + cos^2(theta/2) F- - sin(theta) Z
                             Z'  = -sin(theta)/2 F+ + sin(theta)/2 F- + cos(theta) Z
    a crusher, half an echo spacing; the echo is F_0.

(All states are real: the pulses are 90 degrees out of phase with the excitation. The recovery of Z_0 never reaches an
echo of the same train: it starts a family of states that is at an even number of cycles when an echo is read.) With
theta = 180 degrees this is exp(-n esp / T2). Below 180 degrees part of the magnetisation waits along z, where it
decays with T1, and comes back later: the first echo is low, the later ones are high, and a line through the
logarithm of the echoes gives a T2 that is too long.

``train`` is the recursion in numpy; ``CALC_TEXT`` is the same recursion as a text the ``calc`` tool reads (the
statement of level 1 prints it; a test runs it through ``calc`` and holds the two equal).

What the model leaves out: on a 2D slice the refocusing angle is not one number. It falls from its value at the slice
centre to zero across the edge of the slice profile, and the excitation leaves a transverse phase through the slice.
Fitted with one free angle to the echoes of the console's ``sharp`` pulse, the model gives T2 within -2 to +6 % for a
transmit scale that is off by 8 to 22 %, and within -8 to +11 % at 25 to 45 % off (``task.BIAS``;
``tests/test_t2_echoes.py`` measures it on the engine); the fitted angle is an effective one (about 145 degrees at a
calibrated transmitter), not the transmit scale. For T2 below about 35 ms the fit reads 5 to 17 % long: too few echoes.
"""
from __future__ import annotations

import numpy as np

T1_ASSUMED_MS = 1000.0                      # the model's T1: the echoes depend on it weakly (the time the stored magnetisation waits)


def train(n: int, esp_ms: float, t2_ms: float, theta_deg: float, t1_ms: float = T1_ASSUMED_MS) -> np.ndarray:
    """Echoes 1 .. n of the train, per unit of transverse magnetisation after the excitation."""
    th = np.radians(theta_deg)
    c2, s2, s, c = np.cos(th / 2) ** 2, np.sin(th / 2) ** 2, np.sin(th), np.cos(th)
    e2, e1 = np.exp(-esp_ms / (2 * t2_ms)), np.exp(-esp_ms / (2 * t1_ms))
    fp, fm, z = np.zeros(n + 2), np.zeros(n + 2), np.zeros(n + 2)
    fp[0] = fm[0] = 1.0
    out = np.zeros(n)
    for i in range(n):
        fp, fm, z = fp * e2, fm * e2, z * e1
        fp, fm = np.concatenate([[fm[1]], fp[:-1]]), np.concatenate([[fm[1]], fm[2:], [0.0]])
        fp, fm, z = c2 * fp + s2 * fm + s * z, s2 * fp + c2 * fm - s * z, -0.5 * s * fp + 0.5 * s * fm + c * z
        fp, fm = np.concatenate([[fm[1]], fp[:-1]]), np.concatenate([[fm[1]], fm[2:], [0.0]])
        fp, fm, z = fp * e2, fm * e2, z * e1
        out[i] = fp[0]
    return out


# The same recursion for the ``calc`` tool (Python syntax, numpy names). ``train(n, esp, t2, theta)`` returns the n echoes.
CALC_TEXT = f"""def train(n, esp, t2, theta, t1={T1_ASSUMED_MS:g}):
    th = radians(theta)
    c2 = cos(th / 2) ** 2; s2 = sin(th / 2) ** 2; s = sin(th); c = cos(th)
    e2 = exp(-esp / (2 * t2)); e1 = exp(-esp / (2 * t1))
    fp = concatenate([[1.0], zeros(n + 1)]); fm = fp; z = zeros(n + 2)
    out = []
    for i in range(n):
        fp = fp * e2; fm = fm * e2; z = z * e1
        fp, fm = concatenate([[fm[1]], fp[:-1]]), concatenate([[fm[1]], fm[2:], [0.0]])
        fp, fm, z = c2 * fp + s2 * fm + s * z, s2 * fp + c2 * fm - s * z, -0.5 * s * fp + 0.5 * s * fm + c * z
        fp, fm = concatenate([[fm[1]], fp[:-1]]), concatenate([[fm[1]], fm[2:], [0.0]])
        fp = fp * e2; fm = fm * e2; z = z * e1
        out = out + [fp[0]]
    return array(out)"""
