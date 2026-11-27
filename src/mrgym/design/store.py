"""Immutable design records.

A design is a pulse (``pulses.Pulse``), a gradient program (``gradients.program``) or a block list
(``sequence`` blocks). ``DesignStore.put`` stores it under a short id derived from its content: the first 12
hexadecimal digits of the SHA-256 of a canonical serialisation (JSON with sorted keys and no spaces; every number
written with 12 significant digits and every array first rounded to 12 digits of its largest entry, so that the last
bits of a floating-point result, and rounding noise around zero, do not change the id; complex numbers as
[real, imaginary]). The same design gets the same id, and an id never changes its content. Rounding cannot be made
safe against every perturbation: a value that sits on a rounding boundary can still fall either way.

A record holds the kind, the design itself, the generator call that made it, its small record (duration, peak,
energy, limit check) and nothing else: no time stamp, no author, no file path. ``summary(id)`` is the text an agent
reads; it is kept to at most 160 words and 880 characters, which is about 220 tokens at four characters per token.
Both counts are proxies (``count_words`` splits on white space); no tokenizer is run here.

No global state: a store is an object. Nothing is written unless a directory is passed in, and then only
``<directory>/<id>.json``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from . import gradients, pulses, sequence

KINDS = ("pulse", "gradient", "sequence")
ID_LENGTH = 12
MAX_WORDS, MAX_CHARS = 160, 880


def canonical(obj):
    """JSON-ready form of a design: dictionaries with string keys, lists, strings, booleans, None and numbers, the
    numbers rounded to 12 significant digits (a float that is a whole number becomes an int)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return repr(x)
        x = float(f"{x:.12g}")
        return int(x) if x == int(x) and abs(x) < 1e15 else x
    if isinstance(obj, (complex, np.complexfloating)):
        return [canonical(obj.real), canonical(obj.imag)]
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return canonical(obj.item())
        if obj.dtype.kind in "fc" and obj.size:                 # an array is rounded on the scale of its largest entry,
            peak = float(np.abs(obj).max())                     # so that rounding noise around zero does not reach the id
            if peak > 0 and math.isfinite(peak):
                quantum = 10.0 ** (math.floor(math.log10(peak)) - 11)
                obj = np.round(obj / quantum) * quantum
        return [canonical(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def serialise(obj):
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def count_words(text):
    """Number of white-space separated words: the proxy this module uses for the length of a summary."""
    return len(text.split())


# ---------------------------------------------------------------------------------------------- what is stored
def _pulse_design(p):
    return dict(b1_ut=p.b1_ut, dt_ms=p.dt_ms, g_mt_m=p.g_mt_m, kind=p.kind, rephase_mt_ms_m=p.rephase_mt_ms_m,
                bandwidth_hz=p.bandwidth_hz, thickness_mm=p.thickness_mm)


def _block_design(blk):
    out = dict(kind=blk.kind)
    if blk.kind == "rf":
        if blk.scales:
            out.update(flip_deg=np.degrees(blk.flip_rad), phase_deg=math.degrees(blk.phase_rad))
        else:
            out.update(a=np.asarray(blk.a), b=np.asarray(blk.b))
            if blk.weights is not None:
                out.update(weights=np.asarray(blk.weights))
    elif blk.kind == "delay":
        out.update(ms=blk.ms, bd=blk.bd)
    elif blk.kind == "gradient":
        out.update(cycles=blk.cycles, ms=blk.ms, bd=blk.bd)
    elif blk.kind == "readout":
        out.update(index=blk.index, phase_deg=math.degrees(blk.phase_rad))
    return out


def describe(design, tr_ms=None, hardware=None):
    """(kind, design as plain data, generator call, record) of a pulse, a gradient program or a block list."""
    hw = hardware or gradients.Hardware()
    if isinstance(design, pulses.Pulse):
        tr = design.duration_ms if tr_ms is None else tr_ms
        rec = design.record(tr)
        lim = gradients.check_pulse_gradient(design, hw)
        rec["limits"] = dict(ok=lim["ok"], violations=[v["code"] for v in lim["violations"]], slew_t_m_s=lim["slew_t_m_s"],
                             b1_peak_limit_ut=hw.b1_peak_ut, g_max_mt_m=hw.g_max_mt_m, slew_limit_t_m_s=hw.slew_t_m_s)
        return "pulse", _pulse_design(design), dict(design.call), rec
    if isinstance(design, dict):
        prog = gradients.program(**design)
        t, g = gradients.rasterise(prog, hw.raster_ms)
        chk = gradients.check(t, g, hw)
        heat = chk["heating_mt2_m2_ms"]
        rec = dict(duration_ms=chk["duration_ms"], peak_mt_m=max(chk["peak_mt_m"].values()), slew_t_m_s=max(chk["slew_t_m_s"].values()),
                   energy_mt2_m2_ms=round(sum(heat.values()), 6), pns_percent=chk["pns_combined_percent"],
                   limits=dict(ok=chk["ok"], violations=[v["code"] for v in chk["violations"]], rule=chk["rule"],
                               g_max_mt_m=hw.g_max_mt_m, slew_limit_t_m_s=hw.slew_t_m_s, pns_limit_percent=hw.pns_limit_percent))
        return "gradient", {a: [list(v) for v in vs] for a, vs in prog.items()}, dict(generator="program"), rec
    if isinstance(design, (list, tuple)) and design and all(isinstance(b, sequence.Block) for b in design):
        blocks = [_block_design(b) for b in design]
        flips = [np.abs(np.degrees(b.flip_rad)) if b.scales else np.degrees(2 * np.arcsin(np.clip(np.abs(b.b), 0, 1)))
                 for b in design if b.kind == "rf"]              # nominal angle, or the flip angle of a given rotation
        rec = dict(blocks=len(blocks), duration_ms=round(sum(b.ms for b in design), 6), rf_pulses=len(flips),
                   readouts=sum(b.kind == "readout" for b in design),
                   peak_flip_deg=round(max([float(np.max(f)) for f in flips], default=0.0), 4),
                   energy_deg2=round(sum(float(np.mean(f ** 2)) for f in flips), 4),
                   net_dephasing_cycles=int(sum(b.cycles for b in design)), limits=dict(ok=True, violations=[]))
        return "sequence", blocks, dict(generator="blocks"), rec
    raise TypeError("a design is a pulses.Pulse, a gradient program (dict of vertex lists) or a list of sequence blocks")


class DesignStore:
    """Content-addressed store of designs. ``directory``: where records are also written as JSON (optional)."""

    def __init__(self, directory=None, hardware=None):
        self._records = {}
        self._hardware = hardware or gradients.Hardware()
        self._directory = None
        if directory is not None:
            self._directory = os.path.abspath(directory)
            os.makedirs(self._directory, exist_ok=True)
            for name in sorted(os.listdir(self._directory)):
                if name.endswith(".json") and len(name) == ID_LENGTH + 5:
                    with open(os.path.join(self._directory, name), encoding="utf-8") as f:
                        text = f.read()
                    rec = json.loads(text)
                    if self._id(rec["kind"], rec["design"]) != name[:-5]:
                        raise ValueError(f"record {name} does not match its id: the file was changed")
                    self._records[name[:-5]] = text

    @staticmethod
    def _id(kind, design):
        return hashlib.sha256(serialise(dict(kind=kind, design=design)).encode()).hexdigest()[:ID_LENGTH]

    def put(self, design, tr_ms=None):
        """Store a design; returns its id. Storing the same design again returns the same id and changes nothing.
        ``tr_ms``: the repetition over which a pulse's B1rms is stated (default: its own duration)."""
        kind, data, call, rec = describe(design, tr_ms, self._hardware)
        data = canonical(data)
        ident = self._id(kind, data)
        text = serialise(dict(id=ident, kind=kind, design=data, call=call, record=rec))
        if ident in self._records:
            return ident                                        # immutable: the first record stays
        self._records[ident] = text
        if self._directory is not None:
            path = os.path.join(self._directory, ident + ".json")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        return ident

    def __contains__(self, ident):
        return ident in self._records

    def __len__(self):
        return len(self._records)

    def ids(self):
        return sorted(self._records)

    def get(self, ident):
        """A fresh copy of the record ``dict(id, kind, design, call, record)``; changing it does not change the store."""
        if ident not in self._records:
            raise KeyError(f"no design {ident!r}")
        return json.loads(self._records[ident])

    def pulse(self, ident):
        """The stored pulse as a ``pulses.Pulse`` (samples as stored: 12 significant digits)."""
        rec = self.get(ident)
        if rec["kind"] != "pulse":
            raise ValueError(f"design {ident} is a {rec['kind']}, not a pulse")
        d = rec["design"]
        return pulses.Pulse(np.array([complex(re, im) for re, im in d["b1_ut"]]), np.array(d["dt_ms"], float), np.array(d["g_mt_m"], float),
                            d["kind"], d["rephase_mt_ms_m"], d["bandwidth_hz"], d["thickness_mm"], rec["call"])

    def summary(self, ident):
        """What an agent reads about a design: at most 160 words and 880 characters."""
        rec = self.get(ident)
        r, call = rec["record"], rec["call"]
        lim = r["limits"]
        verdict = "within limits" if lim["ok"] else "VIOLATES " + ", ".join(lim["violations"])
        if rec["kind"] == "pulse":
            args = _call_text(call)
            text = (f"design {ident}: {r['kind']} pulse from {args}. "
                    f"{r['samples']} samples, {r['duration_ms']:g} ms, flip {r['flip_deg']:g} deg on resonance, "
                    f"peak B1 {r['peak_b1_ut']:.4g} uT (limit {lim['b1_peak_limit_ut']:g}), energy {r['energy_ut2_ms']:.4g} uT2*ms, "
                    f"B1rms {r['b1rms_ut']:.4g} uT over TR {r['tr_ms']:g} ms. ")
            if r["thickness_mm"] is not None:
                band = "not stated" if r["bandwidth_hz"] is None else f"{r['bandwidth_hz']:.5g} Hz"
                text += (f"Slice {r['thickness_mm']:g} mm, bandwidth {band}, gradient {r['gradient_mt_m']:.4g} mT/m "
                         f"(limit {lim['g_max_mt_m']:g}), slew {lim['slew_t_m_s']:.4g} T/m/s (limit {lim['slew_limit_t_m_s']:g}), "
                         f"rephasing area {r['rephase_mt_ms_m']:.4g} mT*ms/m. ")
            else:
                text += "No slice-select gradient. "
        elif rec["kind"] == "gradient":
            axes = ", ".join(f"{a} {len(v)} vertices" for a, v in rec["design"].items())
            text = (f"design {ident}: gradient program ({axes}). {r['duration_ms']:g} ms, peak {r['peak_mt_m']:.4g} mT/m "
                    f"(limit {lim['g_max_mt_m']:g}, rule {lim['rule']}), slew {r['slew_t_m_s']:.4g} T/m/s (limit {lim['slew_limit_t_m_s']:g}), "
                    f"stimulation {r['pns_percent']:.3g} % of the declared threshold (limit {lim['pns_limit_percent']:g}), "
                    f"integral of G2 {r['energy_mt2_m2_ms']:.4g} (mT/m)2*ms. ")
        else:
            text = (f"design {ident}: sequence of {r['blocks']} blocks. {r['rf_pulses']} RF pulses (largest flip {r['peak_flip_deg']:g} deg, "
                    f"sum of flip2 {r['energy_deg2']:.4g} deg2), {r['readouts']} readouts, {r['duration_ms']:g} ms, "
                    f"net dephasing {r['net_dephasing_cycles']} cycles. ")
        tail = f"Limits: {verdict}."
        if count_words(text + tail) > MAX_WORDS or len(text + tail) > MAX_CHARS:      # the verdict is never cut
            tail = "... " + tail
            words = text[:MAX_CHARS - len(tail) - 1].rsplit(" ", 1)[0].split()
            text = " ".join(words[:MAX_WORDS - count_words(tail)]) + " "
        return text + tail


def _call_text(call, limit=260):
    """The generator call as text, long lists shortened."""
    def short(v):
        if isinstance(v, dict):
            return short(v.get("generator", "..."))
        if isinstance(v, list):
            return f"[{len(v)} values]"
        return f"{v:g}" if isinstance(v, float) else str(v)

    name = call.get("generator", "unknown")
    if name == "verse":
        inner = _call_text(call.get("of", {}), 120)
        return _shorten(f"verse({inner}, b1_peak_ut={short(call['b1_peak_ut'])}, g_max_mt_m={short(call['g_max_mt_m'])}, "
                        f"slew_t_m_s={short(call['slew_t_m_s'])})", limit)
    args = ", ".join(f"{k}={short(v)}" for k, v in call.items() if k not in ("generator", "design") and v is not None)
    return _shorten(f"{name}({args})", limit)


def _shorten(text, limit):
    return text if len(text) <= limit else text[:limit - 4] + "...)"
