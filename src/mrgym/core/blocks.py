"""The vocabulary of the signal core: a sequence is a list of typed blocks, plain data.

Every block is a frozen dataclass of numbers, strings and tuples: it can be compared, hashed, written to JSON
(``to_data`` / ``from_data``) and given a content digest (``digest``). Nothing in a block depends on a tissue.

    rf(flip_deg, phase_deg)            an instantaneous pulse. Hard by default (one flip angle x the B1 scale of the
                                       spin); ``selective=Selective(tbw, ...)`` makes it the console's windowed sinc with
                                       one rotation per slice column; ``ck=...`` gives the Cayley-Klein parameters of a
                                       designed pulse per slice column (``rf_designed``). ``crushed=True`` is the ideal
                                       refocusing pulse between a perfect crusher pair. ``energy_ut2_ms`` and
                                       ``offset_hz`` say how it saturates a bound pool (magnetisation transfer).
    delay(ms)                          free evolution: T1 with regrowth, T2, precession by the off-resonance, exchange
    gradient(cycles, ms, shape)        dephasing by ``cycles`` (per axis, whole or fractional) of the length
                                       ``Sequence.cycle_mm``; with a duration it is also a delay, and the block then
                                       attenuates every state by diffusion (``b_increment`` gives its b-value)
    spoil()                            ideal spoiler: every transverse state is set to zero, longitudinal ones are kept
    crush()                            ideal crusher: transverse states AND the longitudinal gratings are set to zero
    mz(factor)                         ideal preparation of the longitudinal magnetisation: Mz -> factor x Mz
                                       (inversion of efficiency e: -e; saturation of efficiency e: 1 - e)
    adc(label, phase_deg, static_ms)   one sample of the voxel signal, labelled with the echo or line it feeds
    repeat(body, times, ...)           the body ``times`` times, with RF phase increments per repetition (linear:
                                       phase cycling; quadratic: RF spoiling) and named tables of per-repetition
                                       flip angles or phases

Conventions are those of docs/physics/design.md and are stated once, in ``operators``.

``compile_ops`` turns a block list into the flat list of primitive operations that the three evaluators (``bloch``,
``epg``, ``periodic``) run. Two frames are offered for the RF phase increments of a ``repeat``:
  lab frame      the phase of repetition j, Delta j + phi j (j + 1) / 2, is added to every RF pulse and to the
                 receiver phase of every ``adc`` of the body
  pulse frame    pulses and receiver keep their own phases and after every repetition the whole state is turned by
                 -(Delta + phi) about z and translated by phi / n of a cycle (n = net dephasing of the body). The
                 state at the start of a repetition is then the same in every repetition once the sequence is
                 periodic, which is what ``periodic`` solves for. Valid when every RF pulse of the body sits at zero
                 accumulated dephasing and, for phi != 0, the body has a net dephasing of whole cycles; anything else
                 is refused. Derivation: with S_j the state before repetition j in the laboratory, G_j = D(j phi / n)
                 Rz(-phase_j) S_j, where Rz turns the transverse plane and D(s) multiplies order k by exp(i k s) (a
                 translation). A gradient of c cycles commutes with D up to a turn about z by c s, and a pulse commutes
                 with D but not with a turn about z: so the pulses must sit where the accumulated c is zero, and the
                 turns left over at the end of the repetition add up to the constant -(Delta + phi).
                 A sample taken at a dephasing C != 0 of its repetition j carries the phase j phi C / n in the laboratory;
                 the compiler puts it into the receiver phase, so both frames return the same samples (its magnitude
                 is periodic, its phase is not).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from fractions import Fraction

import numpy as np

WINDOWS = ("hann", "hamming")
MODELS = ("rotation", "flip")
LINEAR = (0.5, 1.0 / 3.0)            # shape moments of a constant gradient: k grows linearly during the block


# ---------------------------------------------------------------------------------------------- the blocks
@dataclasses.dataclass(frozen=True)
class Selective:
    """The console's slice-selective pulse: a windowed sinc of ``samples`` hard sub-pulses with time-bandwidth product
    ``tbw``, played on a slice ``width`` times the reference slice (positions of the spins are in reference slices),
    followed by a rephasing lobe of ``rephase`` times the selection area and preceded by a lobe of ``prephase`` times
    it (a balanced sequence plays half before and half after). ``model``: "rotation" keeps the rotation of every
    column; "flip" replaces it by a hard pulse of the column's flip angle about the pulse axis (the engine's model:
    the transverse phase of the column is then a weight of its signal, ``operators.column_phase``)."""
    tbw: float
    window: str = "hann"
    rephase: float = 0.5
    width: float = 1.0
    samples: int = 128
    model: str = "rotation"
    prephase: float = 0.0

    def __post_init__(self):
        if not self.tbw > 0 or self.window not in WINDOWS or self.model not in MODELS or not self.width > 0 or self.samples < 1:
            raise ValueError(f"Selective: tbw > 0, window in {WINDOWS}, model in {MODELS}, width > 0, samples >= 1")


@dataclasses.dataclass(frozen=True)
class RF:
    flip_deg: float = 0.0
    phase_deg: float = 0.0
    crushed: bool = False
    selective: Selective | None = None
    ck: tuple | None = None                 # ((re a, im a, re b, im b), ...) per slice column: a designed pulse
    energy_ut2_ms: float = 0.0              # integral of |B1|^2 dt: saturates the bound pool
    offset_hz: float = 0.0                  # frequency offset of the pulse; != 0: a saturation pulse, no rotation
    flip_table: str | None = None           # name of a table of the enclosing repeat: flip angle of repetition j
    phase_table: str | None = None          # the same for an extra phase
    kind: str = dataclasses.field(default="rf", init=False)


@dataclasses.dataclass(frozen=True)
class Delay:
    ms: float
    kind: str = dataclasses.field(default="delay", init=False)


@dataclasses.dataclass(frozen=True)
class Gradient:
    cycles: tuple                           # dephasing per axis, in cycles of ``Sequence.cycle_mm``
    ms: float = 0.0
    shape: tuple = LINEAR                   # (m1, m2): time averages of s and s^2, s(t) = fraction of the area played
    kind: str = dataclasses.field(default="gradient", init=False)


@dataclasses.dataclass(frozen=True)
class Spoil:
    kind: str = dataclasses.field(default="spoil", init=False)


@dataclasses.dataclass(frozen=True)
class Crush:
    kind: str = dataclasses.field(default="crush", init=False)


@dataclasses.dataclass(frozen=True)
class Mz:
    factor: float
    kind: str = dataclasses.field(default="mz", init=False)


@dataclasses.dataclass(frozen=True)
class ADC:
    label: object = None
    phase_deg: float = 0.0
    static_ms: float = 0.0                  # net time of static (T2') dephasing of the sampled pathway
    kind: str = dataclasses.field(default="adc", init=False)


@dataclasses.dataclass(frozen=True)
class Repeat:
    body: tuple
    times: int
    phase_inc_deg: float = 0.0              # Delta: linear increment of the RF phase per repetition
    spoil_inc_deg: float = 0.0              # phi: quadratic increment (RF spoiling), phase_j = phi j (j + 1) / 2
    tables: tuple = ()                      # ((name, (values, ...)), ...): value of repetition j = values[j % len]
    kind: str = dataclasses.field(default="repeat", init=False)


BLOCKS = dict(rf=RF, delay=Delay, gradient=Gradient, spoil=Spoil, crush=Crush, mz=Mz, adc=ADC, repeat=Repeat)


@dataclasses.dataclass(frozen=True)
class Sequence:
    """A block list with what the blocks cannot say themselves: the length of one cycle of dephasing per axis
    (``cycle_mm``; needed for diffusion only), and a name. Iterating a sequence iterates its blocks."""
    blocks: tuple
    cycle_mm: tuple | None = None
    name: str = ""

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)


# ---------------------------------------------------------------------------------------------- constructors
def _num(x, what):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite")
    return x


def rf(flip_deg=0.0, phase_deg=0.0, *, crushed=False, selective=None, ck=None, energy_ut2_ms=0.0, offset_hz=0.0,
       flip_table=None, phase_table=None):
    """An instantaneous RF pulse (see the module docstring)."""
    if selective is not None and ck is not None:
        raise ValueError("an rf block is hard, selective or given by ck, not two of them")
    if ck is not None:
        ck = tuple(tuple(float(v) for v in row) for row in ck)
        if not ck or any(len(row) != 4 or abs(row[0] ** 2 + row[1] ** 2 + row[2] ** 2 + row[3] ** 2 - 1) > 1e-9 for row in ck):
            raise ValueError("ck is a list of (re a, im a, re b, im b) with |a|^2 + |b|^2 = 1")
    if energy_ut2_ms < 0:
        raise ValueError("energy_ut2_ms is not negative")
    if offset_hz and (flip_deg or ck is not None or selective is not None):
        raise ValueError("a pulse with a frequency offset is a saturation pulse: it has no flip angle of its own")
    return RF(_num(flip_deg, "flip_deg"), _num(phase_deg, "phase_deg"), bool(crushed), selective, ck, _num(energy_ut2_ms, "energy_ut2_ms"),
              _num(offset_hz, "offset_hz"), flip_table, phase_table)


def rf_designed(pulse, pos_mm, phase_deg=0.0, b1_scale=1.0, df_hz=0.0, crushed=False):
    """The rotation of a designed ``mrgym.design.pulses.Pulse`` at the positions ``pos_mm`` (one per slice column,
    index ``Spins.column``), after its rephasing lobe, as an rf block. The Cayley-Klein parameters are data: they do
    not scale with the B1 of the spins; pass ``b1_scale`` here."""
    a, b = pulse.rotate(np.asarray(pos_mm, float), df_hz, b1_scale)
    energy = float(pulse.energy_ut2_ms) * float(b1_scale) ** 2
    return rf(0.0, phase_deg, crushed=crushed, ck=np.stack([a.real, a.imag, b.real, b.imag], 1), energy_ut2_ms=energy)


def delay(ms):
    if not _num(ms, "ms") >= 0:
        raise ValueError("a delay is not negative")
    return Delay(float(ms))


def gradient(cycles, ms=0.0, shape=LINEAR):
    """Dephasing by ``cycles`` (a number: first axis; or a tuple with one entry per axis). ``shape`` = (m1, m2), the
    time averages of s(t) and s(t)^2 with s the fraction of the area played up to t: (1/2, 1/3) for a constant
    gradient; ``lobe_shape`` computes it for a waveform. It matters for diffusion only."""
    cyc = tuple(_num(c, "cycles") for c in (cycles if isinstance(cycles, (tuple, list)) else (cycles,)))
    if not 1 <= len(cyc) <= 3:
        raise ValueError("dephasing is given on one to three axes")
    m1, m2 = (float(v) for v in shape)
    if not _num(ms, "ms") >= 0 or not (0 <= m1 <= 1 and 0 <= m2 <= 1 and m2 >= m1 * m1 - 1e-12):
        raise ValueError("gradient: ms >= 0 and shape moments 0 <= m1^2 <= m2 <= 1")
    return Gradient(cyc, float(ms), (m1, m2))


def spoil():
    return Spoil()


def crush():
    return Crush()


def mz(factor):
    return Mz(_num(factor, "factor"))


def adc(label=None, phase_deg=0.0, static_ms=0.0):
    """One sample. ``static_ms``: the net time during which the sampled pathway has dephased in the static field
    inhomogeneity of the voxel (TE for a gradient echo, 0 at a spin echo); the sample is multiplied by
    exp(-|static_ms| / T2'). The general case is an ensemble of off-resonances (``operators.lorentzian_spread``)."""
    return ADC(label, _num(phase_deg, "phase_deg"), _num(static_ms, "static_ms"))


def repeat(body, times, phase_inc_deg=0.0, spoil_inc_deg=0.0, tables=None):
    """``body`` repeated ``times`` times. ``tables``: {name: values}; an rf block of the body with ``flip_table=name``
    takes values[j % len(values)] as its flip angle in repetition j (``phase_table``: added to its phase)."""
    body = tuple(body)
    if int(times) != times or times < 0:
        raise ValueError("times is a whole number >= 0")
    tab = tuple(sorted((str(k), tuple(float(x) for x in v)) for k, v in dict(tables or {}).items()))
    if any(not v for _, v in tab):
        raise ValueError("a table has at least one value")
    names = {k for k, _ in tab}
    for blk in body:
        if blk.kind == "rf" and any(t is not None and t not in names for t in (blk.flip_table, blk.phase_table)):
            raise ValueError("an rf block names a table that the repeat does not hold")
    return Repeat(body, int(times), _num(phase_inc_deg, "phase_inc_deg"), _num(spoil_inc_deg, "spoil_inc_deg"), tab)


def sequence(blocks, cycle_mm=None, name=""):
    blocks = tuple(blocks.blocks if isinstance(blocks, Sequence) else blocks)
    _check(blocks)
    if cycle_mm is not None:
        cycle_mm = tuple(float(c) for c in (cycle_mm if isinstance(cycle_mm, (tuple, list)) else (cycle_mm,)))
        if any(not c > 0 for c in cycle_mm):
            raise ValueError("cycle_mm is positive")
    return Sequence(blocks, cycle_mm, str(name))


def _check(blocks):
    for blk in blocks:
        if type(blk) not in BLOCKS.values():
            raise ValueError("a sequence is a list of blocks made by rf, delay, gradient, spoil, crush, mz, adc, repeat")
        if blk.kind == "repeat":
            _check(blk.body)


def lobe_shape(t_ms, g):
    """(area in mT/m ms, duration in ms, (m1, m2)) of a one-axis waveform given at increasing times and joined by
    straight lines: s(t) = running area / total area, m1 = mean of s, m2 = mean of s^2 over the duration. Exact for
    the piecewise-linear waveform (Simpson's rule on segments of s, which is quadratic in t; s^2 by 3-point Gauss)."""
    t, g = np.asarray(t_ms, float), np.asarray(g, float)
    h = np.diff(t)
    if t.ndim != 1 or t.size < 2 or g.shape != t.shape or np.any(h <= 0):
        raise ValueError("a waveform is increasing times and as many amplitudes")
    seg = 0.5 * (g[:-1] + g[1:]) * h
    area, start = float(seg.sum()), np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    if area == 0:
        raise ValueError("the lobe has no area")
    m1 = m2 = 0.0
    for x, w in ((0.5 - math.sqrt(0.15), 5 / 18), (0.5, 8 / 18), (0.5 + math.sqrt(0.15), 5 / 18)):
        s = (start + g[:-1] * x * h + (g[1:] - g[:-1]) * (x * h) ** 2 / (2 * h)) / area
        m1 += float(np.sum(w * h * s))
        m2 += float(np.sum(w * h * s * s))
    dur = float(t[-1] - t[0])
    return area, dur, (m1 / dur, m2 / dur)


def b_increment(cycles_in, cycles, ms, cycle_mm, shape=LINEAR):
    """b-value (s/mm^2) that a block of duration ``ms`` adds to a state that enters it with the dephasing
    ``cycles_in`` and leaves with ``cycles_in + cycles`` (numbers, or per axis):
        b = sum over axes of kappa^2 t (k^2 + 2 m1 k n + m2 n^2),   kappa = 2 pi / cycle_mm,
    the integral of (2 pi k(t))^2 dt along the block. A delay is the case n = 0."""
    k, n, c = (np.atleast_1d(np.asarray(v, float)) for v in (cycles_in, cycles, cycle_mm))
    m1, m2 = shape
    return float(np.sum((2 * math.pi / c) ** 2 * (k * k + 2 * m1 * k * n + m2 * n * n)) * ms * 1e-3)


# ---------------------------------------------------------------------------------------------- serialisation
def to_data(obj):
    """Blocks, sequences and lists of them as JSON-ready data."""
    if isinstance(obj, Sequence):
        return dict(kind="sequence", name=obj.name, cycle_mm=None if obj.cycle_mm is None else list(obj.cycle_mm),
                    blocks=[to_data(b) for b in obj.blocks])
    if isinstance(obj, (list, tuple)):
        return [to_data(b) for b in obj]
    out = dict(kind=obj.kind)
    for f in dataclasses.fields(obj):
        if f.name == "kind":
            continue
        v = getattr(obj, f.name)
        if f.name == "body":
            v = [to_data(b) for b in v]
        elif f.name == "selective" and v is not None:
            v = dataclasses.asdict(v)
        elif f.name == "tables":
            v = {k: list(vals) for k, vals in v}
        elif f.name == "ck" and v is not None:
            v = [list(row) for row in v]
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def from_data(data):
    """The inverse of ``to_data``."""
    if isinstance(data, list):
        return tuple(from_data(d) for d in data)
    d = dict(data)
    kind = d.pop("kind")
    if kind == "sequence":
        return sequence(from_data(d["blocks"]), d.get("cycle_mm"), d.get("name", ""))
    if kind == "rf":
        sel = d.pop("selective", None)
        flip, phase = d.pop("flip_deg", 0.0), d.pop("phase_deg", 0.0)
        return rf(flip, phase, selective=None if sel is None else Selective(**sel), **d)
    if kind == "repeat":
        return repeat(from_data(d["body"]), d["times"], d.get("phase_inc_deg", 0.0), d.get("spoil_inc_deg", 0.0), d.get("tables"))
    if kind == "gradient":
        return gradient(tuple(d["cycles"]), d.get("ms", 0.0), tuple(d.get("shape", LINEAR)))
    if kind == "adc":
        label = d.get("label")
        return adc(tuple(label) if isinstance(label, list) else label, d.get("phase_deg", 0.0), d.get("static_ms", 0.0))
    if kind in ("spoil", "crush"):
        return BLOCKS[kind]()
    if kind == "delay":
        return delay(d["ms"])
    if kind == "mz":
        return mz(d["factor"])
    raise ValueError(f"unknown block kind {kind!r}")


def digest(obj):
    """First 16 hexadecimal digits of the SHA-256 of the canonical JSON of a block list or sequence."""
    text = json.dumps(to_data(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def from_design(blocks):
    """A block list of ``mrgym.design.sequence`` as core blocks (diffusion ``bd`` of a design block is not carried:
    in the core it follows from the tissue and ``cycle_mm``). Pulses given by Cayley-Klein parameters must be
    scalars or one value per slice column."""
    out = []
    for blk in blocks:
        if blk.kind == "rf":
            if blk.scales:
                flips = np.atleast_1d(np.degrees(blk.flip_rad))
                if flips.size != 1:
                    raise ValueError("per-column flip angles of a design block: use the B1 scale of the spins instead")
                out.append(rf(float(flips[0]), math.degrees(blk.phase_rad)))
            else:
                a, b = np.atleast_1d(blk.a), np.atleast_1d(blk.b)
                a, b = np.broadcast_arrays(a, b)
                out.append(rf(ck=np.stack([a.real, a.imag, b.real, b.imag], 1)))
        elif blk.kind == "delay":
            out.append(delay(blk.ms))
        elif blk.kind == "gradient":
            out.append(gradient(blk.cycles, blk.ms))
        elif blk.kind == "spoil":
            out.append(spoil())
        else:
            out.append(adc(blk.index, math.degrees(blk.phase_rad)))
    return tuple(out)


# ---------------------------------------------------------------------------------------------- compilation
@dataclasses.dataclass(frozen=True)
class Op:
    """One primitive operation of a compiled sequence. ``n``: dephasing in lattice units per axis; ``block``: the rf
    block (for its kind of rotation); ``turn`` / ``shift``: the frame change after a repetition (rad; rad per cycle)."""
    kind: str                               # rf, free, spoil, crush, mz, adc, frame
    block: RF | None = None
    flip_deg: float = 0.0
    phase_deg: float = 0.0
    ms: float = 0.0
    n: tuple = ()
    shape: tuple = LINEAR
    factor: float = 1.0
    label: object = None
    static_ms: float = 0.0
    turn: float = 0.0
    shift: tuple = ()
    at_zero: bool = True                    # adc: the accumulated dephasing of its repetition is zero


@dataclasses.dataclass(frozen=True)
class Program:
    ops: tuple
    unit: tuple                             # cycles per lattice step, per axis
    reach: tuple                            # largest lattice index the gradients can reach, per axis
    cycle_mm: tuple | None
    labels: tuple                           # labels of the adc operations, in order
    frame: bool
    period_ms: float                        # duration of the compiled list

    @property
    def axes(self):
        return len(self.unit)


def _blocks_of(seq):
    if isinstance(seq, Sequence):
        return seq.blocks, seq.cycle_mm
    blocks = tuple(seq)
    _check(blocks)
    return blocks, None


def _all_cycles(blocks):
    for blk in blocks:
        if blk.kind == "gradient":
            yield blk.cycles
        elif blk.kind == "repeat":
            yield from _all_cycles(blk.body)


def lattice(seq):
    """(axes, unit per axis): the coarsest lattice that holds every dephasing of the sequence. Unit 1 when all
    dephasings of an axis are whole numbers of cycles; otherwise their greatest common divisor (as fractions with
    denominators up to 10^6)."""
    blocks, _ = _blocks_of(seq)
    cyc = list(_all_cycles(blocks))
    axes = max([len(c) for c in cyc], default=1)
    unit = []
    for i in range(axes):
        fr = [Fraction(c[i]).limit_denominator(10 ** 6) for c in cyc if len(c) > i and c[i] != 0]
        g = Fraction(1)
        if fr and any(f.denominator != 1 for f in fr):
            g = fr[0]
            for f in fr[1:]:
                g = Fraction(math.gcd(g.numerator * f.denominator, f.numerator * g.denominator), g.denominator * f.denominator)
            g = abs(g)
        unit.append(g)
    return axes, tuple(unit)


def _net(blocks, axes):
    """Net dephasing (cycles per axis) of a block list, and whether every rf sits at zero accumulated dephasing."""
    acc, ok = [0.0] * axes, [True]

    def walk(blks):
        for blk in blks:
            if blk.kind == "gradient":
                for i, c in enumerate(blk.cycles):
                    acc[i] += c
            elif blk.kind == "rf":
                ok[0] = ok[0] and all(abs(a) < 1e-9 for a in acc)
            elif blk.kind == "repeat":
                for _ in range(blk.times):
                    walk(blk.body)

    walk(blocks)
    return acc, ok[0]


def compile_ops(seq, frame=False, repetitions=None):
    """The flat list of operations of a block list (``Program``). ``frame``: the pulse frame for the RF phase
    increments of every repeat (module docstring). ``repetitions``: when the sequence is ONE repeat, the number of its
    repetitions to compile instead of ``times`` (``periodic`` compiles one period this way; the state then stays in
    the pulse frame at the end instead of being turned back)."""
    blocks, cycle_mm = _blocks_of(seq)
    axes, unit = lattice(blocks)
    if cycle_mm is not None:
        if len(cycle_mm) == 1 and axes > 1:
            cycle_mm = cycle_mm * axes
        if len(cycle_mm) < axes:
            raise ValueError(f"the sequence dephases on {axes} axes; cycle_mm has {len(cycle_mm)}")
        cycle_mm = tuple(cycle_mm[:axes])
    if repetitions is not None:
        if len(blocks) != 1 or blocks[0].kind != "repeat":
            raise ValueError("repetitions applies to a sequence that is one repeat block")
        blocks = (dataclasses.replace(blocks[0], times=int(repetitions)),)
    ops, labels = [], []
    levels = []                                             # pulse frames in force: [repetition, shift, dephasing since its start]

    def lat(cycles):
        out = []
        for i in range(axes):
            c = cycles[i] if i < len(cycles) else 0.0
            n = Fraction(c).limit_denominator(10 ** 6) / unit[i]
            if n.denominator != 1:
                raise ValueError("a dephasing is not on the lattice of the sequence")
            out.append(int(n))
        return tuple(out)

    def emit(blks, phase, tables, rep, path, top=False):
        for blk in blks:
            k = blk.kind
            if k == "rf":
                flip = blk.flip_deg if blk.flip_table is None else tables[blk.flip_table][rep % len(tables[blk.flip_table])]
                extra = 0.0 if blk.phase_table is None else tables[blk.phase_table][rep % len(tables[blk.phase_table])]
                ops.append(Op("rf", block=blk, flip_deg=flip, phase_deg=blk.phase_deg + extra + phase))
            elif k == "delay":
                if blk.ms > 0:
                    ops.append(Op("free", ms=blk.ms, n=(0,) * axes))
            elif k == "gradient":
                n = lat(blk.cycles)
                for level in levels:
                    for i, c in enumerate(blk.cycles):
                        level[2][i] += c
                if blk.ms > 0 or any(n):
                    ops.append(Op("free", ms=blk.ms, n=n, shape=blk.shape))
            elif k in ("spoil", "crush"):
                ops.append(Op(k))
            elif k == "mz":
                ops.append(Op("mz", factor=blk.factor))
            elif k == "adc":
                label = blk.label if not path else (blk.label if isinstance(blk.label, tuple) else (blk.label,)) + path
                labels.append(label)
                # in a translated frame a sample taken at the dephasing C of its repetition j differs from the laboratory
                # sample by the phase j (shift . C); it is put into the receiver phase so that both frames give the same
                turn = sum(j * sum(sh * c for sh, c in zip(shift, acc)) for j, shift, acc in levels)
                ops.append(Op("adc", label=label, phase_deg=blk.phase_deg + phase - math.degrees(turn), static_ms=blk.static_ms,
                              at_zero=all(abs(c) < 1e-9 for _, _, acc in levels for c in acc)))
            else:
                tab = dict(tables, **dict(blk.tables))
                net, rf_ok = _net(blk.body, axes)
                moving = frame and bool(blk.phase_inc_deg or blk.spoil_inc_deg)
                shift = (0.0,) * axes
                if frame and blk.spoil_inc_deg:
                    norm = sum(v * v for v in net)
                    if not rf_ok or norm < 1e-18 or any(u != 1 for u in unit):
                        raise ValueError("pulse frame: an RF-spoiled repeat needs every rf at zero accumulated dephasing, a net "
                                         "dephasing per repetition and whole cycles")
                    shift = tuple(math.radians(blk.spoil_inc_deg) * v / norm for v in net)
                for j in range(blk.times):
                    if moving:
                        levels.append([j, shift, [0.0] * axes])
                    ph = phase if frame else phase + blk.phase_inc_deg * j + blk.spoil_inc_deg * j * (j + 1) / 2
                    emit(blk.body, ph, tab, j, path + (j,))
                    if moving:
                        levels.pop()
                        ops.append(Op("frame", turn=-math.radians(blk.phase_inc_deg + blk.spoil_inc_deg), shift=shift))
                if moving and not (top and repetitions is not None):          # back to the frame of the enclosing list
                    back = math.radians(blk.phase_inc_deg * blk.times + blk.spoil_inc_deg * blk.times * (blk.times + 1) / 2)
                    ops.append(Op("frame", turn=back, shift=tuple(-blk.times * v for v in shift)))

    emit(blocks, 0.0, {}, 0, (), top=True)
    reach = tuple(sum(abs(op.n[i]) for op in ops if op.kind == "free") for i in range(axes))
    return Program(tuple(ops), tuple(float(u) for u in unit), reach, cycle_mm, tuple(labels), bool(frame),
                   float(sum(op.ms for op in ops)))


__all__ = ["ADC", "BLOCKS", "Crush", "Delay", "Gradient", "LINEAR", "Mz", "Op", "Program", "RF", "Repeat", "Selective", "Sequence",
           "Spoil", "adc", "b_increment", "compile_ops", "crush", "delay", "digest", "from_data", "from_design", "gradient", "lattice",
           "lobe_shape", "mz", "repeat", "rf", "rf_designed", "sequence", "spoil", "to_data"]
