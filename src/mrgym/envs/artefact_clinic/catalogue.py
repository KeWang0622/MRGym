"""The catalogue of the clinic: one entry per cause. Adding a cause is adding one ``Cause`` to ``CATALOGUE``.

An entry holds everything the family needs to know about a cause:

    label        the name an agent delivers
    mechanism    the switch of its counterfactual twin (``metrics``: an engine effect name, or fatphase / wrap /
                 truncation)
    group        causes of one group share a console parameter family; at levels 1 and 2 exactly one cause of a
                 group is armed by the complaint protocol (``None``: armed in every complaint protocol)
    arm, disarm  the complaint generator: how the complaint protocol is drawn so that the mechanism can (cannot)
                 reach a marked region. Where it does reach one is decided by the hidden object, never by the sheet
    remedy       the reference remedies on a protocol, at the dose the catalogue states: ((console parameter,
                 function), ...) in order of preference; the first whose parameter the examination has not locked
                 is the one the generator verifies and the reference policy applies
    present, absent, clean
                 thresholds of the target-region error E (see ``metrics``): a region carries the cause when the
                 mechanism alone makes E >= ``present``; it is free of it below ``absent`` (regions in between are
                 never served); a delivered protocol is clean when E <= ``clean``
    formula, remedies, experiment
                 the textbook row (governing formula; remedies with what each costs; the differential experiment)
    responds     other experiments (named by mechanism) that always change a finding with this cause
    erratic      experiments whose effect on a finding with this cause is not determined (a ghost copy has another
                 phase in every scan: it may or may not change under an experiment that does not move it)

A level serves the causes it lists (``task.LEVELS[lv]["causes"]``); nothing else in the family names a cause.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

from ...scanner import console, rf

GAMMA_HZ_PER_T = console.GAMMA_HZ_PER_T
FAT_PPM = console.FAT_PPM
WFS_ARMED_PX = (2.0, 3.0)          # water-fat shift of a complaint protocol that arms the displacement, pixels
WFS_QUIET_PX = (0.15, 0.25)        # ... of one that does not
WFS_CLEAN_PX = 0.20                # dose of the bandwidth remedy: at most this water-fat shift
MATRIX_REMEDY = 2                  # dose of the resolution remedy: pixel size divided by this
GRADIENT_USE = 0.95                # the bandwidth remedy stays below this fraction of the gradient limit
MOTION_AVERAGES_DIVISOR = 2        # textbook dose of the ghost remedy: the averages divided by this (the copy moves with TR x averages)
DROPOUT_RESIDUAL = 0.04            # dose of the slice remedy: this share of through-slice signal loss may remain (clean is 0.05)
SLICE_MIN_MM = 1.5                 # the console refuses thinner slices (SLICE_TOO_THIN)
GHOST_CLEAR_MM = 18.5              # a ghost copy is off a finding when its centre is this far from the finding's along the phase axis


def fat_hz(field_t: float) -> float:
    """Water-fat frequency difference (main methylene peak), Hz."""
    return FAT_PPM * 1e-6 * GAMMA_HZ_PER_T * field_t


def fat_period_ms(field_t: float) -> float:
    return 1e3 / fat_hz(field_t)


def echo_class(te_ms: float, field_t: float) -> str:
    """"in" when water and the main fat peak are within a quarter cycle of in phase at ``te_ms``, else "opposed"."""
    frac = (te_ms / fat_period_ms(field_t)) % 1.0
    return "in" if min(frac, 1.0 - frac) < 0.25 else "opposed"


def bandwidth_cap(matrix: int, fov_mm: float) -> float:
    """Largest bandwidth per pixel (multiple of 5 Hz) that keeps the readout gradient under ``GRADIENT_USE`` of the limit."""
    g = GRADIENT_USE * console.SYSTEM["g_max_mt_m"] * 1e-3
    return min(console.SEQUENCES["spgr"]["bandwidth_hz_per_px"][1], math.floor(g * GAMMA_HZ_PER_T * fov_mm * 1e-3 / matrix / 5.0) * 5.0)


# ------------------------------------------------------------------------------------------------ complaint generators
def _bw(field_t, wfs_px, matrix, fov_mm):
    return float(min(bandwidth_cap(matrix, fov_mm), max(50.0, round(fat_hz(field_t) / wfs_px / 5.0) * 5.0)))


def _arm_shift(p, ctx, rng):
    p["bandwidth_hz_per_px"] = _bw(ctx["field_t"], rng.uniform(*WFS_ARMED_PX), p["matrix"], p["fov_mm"])


def _quiet_shift(p, ctx, rng):
    p["bandwidth_hz_per_px"] = _bw(ctx["field_t"], rng.uniform(*WFS_QUIET_PX), p["matrix"], p["fov_mm"])


def _arm_phase(p, ctx, rng):
    ctx["echo"] = "opposed"


def _quiet_phase(p, ctx, rng):
    ctx["echo"] = "in"


def _arm_nothing(p, ctx, rng):
    pass


# ------------------------------------------------------------------------------------------------ remedies
def _remedy_shift(p, ctx):
    p["bandwidth_hz_per_px"] = max(p["bandwidth_hz_per_px"], math.ceil(fat_hz(ctx["field_t"]) / WFS_CLEAN_PX / 5.0) * 5.0)


def _remedy_polarity(p, ctx):
    p["ro_polarity"] = -p["ro_polarity"]


def _remedy_phase(p, ctx):
    ctx["echo"] = "in"


def _remedy_truncation(p, ctx):
    p["matrix"] = int(p["matrix"] * MATRIX_REMEDY)


def _remedy_wrap(p, ctx):
    p["phase_fov"] = 1.0


def _remedy_motion(p, ctx):
    p["averages"] = max(1, int(p["averages"]) // MOTION_AVERAGES_DIVISOR)


def _cycles(value: float, of, lo: float = 0.0, hi: float = 1.6) -> float:
    """The number of turns c in [lo, hi] at which the decreasing function ``of`` passes ``value`` (bisection)."""
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if of(mid) > value else (lo, mid)
    return 0.5 * (lo + hi)


def slice_dose(slice_mm: float, signal_ratio: float, te_ratio: float = 1.0, tbw: float | None = None, window: str = "hann") -> float:
    """Slice thickness that leaves ``DROPOUT_RESIDUAL`` of through-slice dephasing, from one experiment: the signal
    per unit thickness of a slice half as thick over that of the full slice (``signal_ratio`` >= 1).

    The loss factor is D(c), c = G x thickness x TE turns of phase across the slice: the pulse shape of the console
    (``rf.dephasing_factor``; ``tbw`` None: the Gaussian of a hard pulse, exp(-(pi c)^2 / (4 ln 2))). The experiment
    gives c on the complaint protocol from D(c / 2) / D(c) = ratio; the dose is the thickness at which
    D(c x thickness / slice x ``te_ratio``) = 1 - residual, ``te_ratio`` = echo time of the remedy over that of the
    complaint (the loss changes with the echo time too). Rounded down to 0.1 mm, never below ``SLICE_MIN_MM``."""
    if tbw is None:
        factor = lambda c: math.exp(-(math.pi * c) ** 2 / (4 * math.log(2)))                      # noqa: E731
    else:
        factor = lambda c: float(rf.dephasing_factor(np.array([c]), tbw, window)[0])               # noqa: E731
    top = 1.6 if tbw is None else min(1.6, 0.45 * tbw)
    c0 = _cycles(1.0 / max(signal_ratio, 1.0 + 1e-6), lambda c: factor(c) / factor(0.5 * c), hi=top)
    keep = _cycles(1.0 - DROPOUT_RESIDUAL, factor, hi=top)
    dose = slice_mm * keep / max(c0 * te_ratio, 1e-9)
    return float(max(SLICE_MIN_MM, min(slice_mm, math.floor(10.0 * dose + 1e-9) / 10.0)))


def _remedy_dropout(p, ctx):
    """The textbook half slice. Where the experiment was run, ``generator.assemble`` replaces it by the measured
    dose (``slice_dose``) once the echo time of the remedy is known."""
    p["slice_mm"] = float(max(SLICE_MIN_MM, 0.5 * p["slice_mm"]))


def ghost_copies(pe_source_mm: float, cycles_per_tr: float, averages: int, fov_pe_mm: float) -> tuple:
    """Phase-encode positions (mm) of the two first-order copies of a pulsing structure at ``pe_source_mm``: the
    source -+ frac(f x TR x averages) x the encoded phase field of view, folded into that field of view."""
    d = (cycles_per_tr * averages) % 1.0 * fov_pe_mm
    return tuple((pe_source_mm + s * d + fov_pe_mm / 2.0) % fov_pe_mm - fov_pe_mm / 2.0 for s in (1.0, -1.0))


def ghost_averages(ghosts, findings_pe, fov_pe_mm: float, most: int, clear: float = GHOST_CLEAR_MM):
    """Dose of the ghost remedy from a measured displacement: the largest number of averages <= ``most`` at which
    no copy of any tracked structure lies within ``clear`` mm of a finding of its column. ``ghosts``: what the
    experiment measured per ghost finding, dicts with ``pe_source_mm``, ``cycles_per_tr`` and ``column`` (indices
    into ``findings_pe`` of the findings on its phase-encode line); ``findings_pe``: phase-encode coordinates of the
    findings. None when no number of averages clears every finding."""
    for n in range(int(most), 0, -1):
        if all(abs(c - findings_pe[i]) >= clear for g in ghosts for i in g["column"]
               for c in ghost_copies(g["pe_source_mm"], g["cycles_per_tr"], n, fov_pe_mm)):
            return n
    return None


@dataclasses.dataclass(frozen=True)
class Cause:
    label: str
    mechanism: str
    group: str | None
    arm: Callable
    disarm: Callable
    remedy: tuple
    formula: str
    remedies: tuple                    # ((what, price), ...), the first is the reference remedy
    experiment: str
    present: float = 0.10
    absent: float = 0.03
    clean: float = 0.05
    responds: tuple = ()               # other experiments (named by mechanism) that also change a finding with this cause
    erratic: tuple = ()                # experiments that may or may not change a finding with this cause
    look: tuple = ("edge",)            # how a finding with this cause looks in the complaint image (``generator.look``)
    world: dict | None = None          # what the hidden world needs for this mechanism (keys of the world spec)
    enabled: bool = True               # False: the entry is complete but no level serves it (see ``DISABLED``)


_ENTRIES = (
    Cause("chemical_shift_displacement", "chemshift", "fat", _arm_shift, _quiet_shift,
          (("bandwidth_hz_per_px", _remedy_shift), ("ro_polarity", _remedy_polarity)),
          "fat appears -polarity x 3.4 ppm x gamma x B0 / bandwidth_hz_per_px pixels along the readout",
          (("raise the bandwidth until the shift is at most 0.2 pixel", "SNR falls with the square root of the bandwidth"),
           ("reverse the readout polarity", "free; moves the shift to the other side of the fat"),
           ("swap the phase-encode direction", "free; moves the shift to the other axis and the wrap-around with it"),
           ("fat saturation", "10 ms per repetition; fat is lost from the image")),
          "reverse the readout polarity: the bright and the dark side exchange; raise the bandwidth: the shift shrinks; "
          "change TE: nothing"),
    Cause("opposed_phase_cancellation", "fatphase", "fat", _arm_phase, _quiet_phase, (("te_ms", _remedy_phase),),
          "a pixel that holds water w and fat f shows |w + f c(TE)|, c = sum_p a_p exp(2 pi i f_p TE); minima near "
          "(k + 1/2) / (3.4 ppm x gamma x B0): 2.3, 6.8 ms at 1.5 T; 1.1, 3.4 ms at 3 T; gradient echo only",
          (("move TE to the nearest in-phase time", "later: T2* loss and a longer minimum TR; earlier: needs more bandwidth"),
           ("spin echo", "scan time"), ("smaller pixels", "the line gets thinner, not weaker; scan time and SNR")),
          "TE by half a fat period: the line goes; reversed polarity or other bandwidth: nothing; the line follows "
          "the border on every side; a material that holds water and fat is dark as a whole and brightens",
          look=("edge", "dark")),
    Cause("truncation_ringing", "truncation", None, _arm_nothing, _arm_nothing, (("matrix", _remedy_truncation),),
          "sinc point-spread function of the sampled k-space: 9 % overshoot of a step, ripple period 2 pixels",
          (("double the matrix at the same field of view", "scan time with the lines, SNR with the voxel"),
           ("smaller field of view", "wrap-around")),
          "matrix x 2: the ripple spacing halves (a real rim keeps its width in mm); bandwidth, TE, polarity: nothing",
          absent=0.03, clean=0.03),
    Cause("wrap_around", "wrap", None, _arm_nothing, _arm_nothing, (("phase_fov", _remedy_wrap),),
          "shown position = position modulo the phase field of view: what lies beyond it folds in from the other side",
          (("phase field of view to the full field of view", "scan time with the lines"),
           ("phase oversampling", "scan time with the lines"),
           ("swap the phase-encode direction", "free if the object is shorter along the other axis; moves the fat shift")),
          "more phase field of view or oversampling: the copy leaves; TR, averages, bandwidth, TE: nothing", look=("bright",)),
    Cause("periodic_motion_ghost", "motion", None, _arm_nothing, _arm_nothing, (("averages", _remedy_motion),),
          "a structure whose signal pulses with frequency f appears again at +- frac(f x TR x averages) x the encoded "
          "phase field of view along the phase-encode axis, whatever the direction of the motion",
          (("other averages (or another TR) so that no copy lies on a finding: the number follows from the measured "
            "displacement of the copy", "SNR with the square root of the averages"),
           ("swap the phase-encode direction", "moves the copy to the other axis, and the wrap-around with it")),
          "other averages or TR: the copy moves (a folded copy does not); more phase oversampling: it moves too, with "
          "the encoded field of view; bandwidth, TE, slice: the copy stays where it is, with another phase in every scan",
          responds=("wrap",), erratic=("chemshift", "fatphase", "truncation", "dropout"), look=("bright",),
          world=dict(motion=dict(kind="pulse", f_hz=(0.9, 1.5), depth=(0.35, 0.6)))),
    Cause("susceptibility_signal_loss", "dropout", None, _arm_nothing, _arm_nothing, (("slice_mm", _remedy_dropout),),
          "gradient echo: signal x the dephasing factor of G x dz x TE cycles across the slice profile (hard pulse: "
          "exp(-(pi G dz TE)^2 / (4 ln 2))), G the through-slice field gradient in Hz/mm, dz the slice",
          (("thinner slice, as thin as the measured loss requires", "SNR with the slice thickness"),
           ("shorter TE", "needs more bandwidth: SNR"), ("spin echo", "scan time")),
          "half the slice: the signal per mm rises where the loss is (a real dark material keeps its ratio); other TE: "
          "the loss changes with TE squared; bandwidth, matrix, phase field of view: nothing",
          responds=("fatphase",), look=("dark",), world=dict(dbdz=dict(n=(3, 4), peak_hz_per_mm=(4.0, 8.0), sigma_mm=(10.0, 16.0)))),
)
CATALOGUE = {c.label: c for c in _ENTRIES if c.enabled}
# Complete entries that no level serves (``enabled`` False). None at present: the motion ghost, rare where findings
# are searched on a drawn object, is served on constructed sites (``generator.build_constructed``).
DISABLED = {c.label: c for c in _ENTRIES if not c.enabled}

# Artefacts this engine cannot form are not listed here and are not faked: fat-water swap (needs a Dixon
# reconstruction), slice cross-talk (needs several slices), partial volume through the slice (needs a third dimension).
PENDING = ()

MECHANISMS = tuple(c.mechanism for c in CATALOGUE.values())
LABELS = tuple(CATALOGUE)
BY_MECHANISM = {c.mechanism: c for c in CATALOGUE.values()}
BASE = tuple(c.label for c in CATALOGUE.values() if c.world is None)   # the causes that need nothing from the world beyond the test object


def mechanisms(labels) -> tuple:
    return tuple(CATALOGUE[lab].mechanism for lab in labels)


def decode(flagged: set, labels) -> list:
    """The causes of a finding from the experiments that changed it (named by mechanism). A cause is present when
    its own experiment changed the finding, unless that change is explained by another present cause whose
    ``responds`` or ``erratic`` lists the experiment (a ghost also moves under phase oversampling; signal loss also
    changes with TE)."""
    own = [CATALOGUE[lab] for lab in labels if CATALOGUE[lab].mechanism in flagged]
    return sorted(c.label for c in own if not any(c.mechanism in d.responds + d.erratic for d in own if d is not c))


def expected(causes, experiment: str):
    """Whether an experiment (named by mechanism) changes a finding that carries ``causes``: True, False, or None
    where it is not determined (an experiment that is ``erratic`` for a cause and that no cause responds to)."""
    if any(CATALOGUE[lab].mechanism == experiment or experiment in CATALOGUE[lab].responds for lab in causes):
        return True
    return None if any(experiment in CATALOGUE[lab].erratic for lab in causes) else False


def minimal_battery(labels, options=None) -> tuple:
    """The experiments (named by mechanism) that diagnose every finding of a level without looking: the smallest set
    under which any two cause sets the level serves change a finding under different experiments. ``options``: the
    cause sets served (default: every single cause, the pair of the fat causes, and none). Two cause sets are told
    apart by an experiment whose effect is determined for both and differs. An experiment is in the set exactly
    when dropping it leaves two cause sets indistinguishable; an experiment that only confirms what another one
    shows (the reversed readout) is never in it."""
    if options is None:
        fat = tuple(sorted(lab for lab in labels if CATALOGUE[lab].group == "fat"))
        options = [(lab,) for lab in labels] + ([fat] if len(fat) > 1 else []) + [()]
    options = sorted({tuple(sorted(o)) for o in options})
    names = list(mechanisms(labels))

    def separates(subset):
        rows = [[expected(o, e) for e in subset] for o in options]
        return all(any(a is not None and b is not None and a != b for a, b in zip(rows[i], rows[j]))
                   for i in range(len(rows)) for j in range(i))
    return tuple(e for e in names if not separates([x for x in names if x != e]))
