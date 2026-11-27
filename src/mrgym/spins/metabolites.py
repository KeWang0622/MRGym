"""A basis of proton spin systems: chemical shifts (ppm) and scalar couplings (Hz) with the source of every row.

How far each number was checked is tagged, as in ``mrgym.scanner.tissue``:
  [F]  read in a document that was fetched for this release. For the brain metabolites that document is a software
       copy of the published table, NOT the paper: ``spant`` (R package, file R/mol_parameters.R, fetched
       2026-11-26), whose entries name Govindaraju et al. 2000 or its 2015 corrigendum as their source. The
       couplings and shifts of lactate, alanine, NAA, glutamate, glutamine, myo-inositol and creatine were also
       compared with a second copy (VeSPA) and are equal there. The paper and the corrigendum themselves were
       not read (both behind a paywall).
  [E]  the value of this package's engine (``mrgym.scanner.tissue``); a test asserts that the two are equal.
  [M]  recalled by the authors; no source was fetched. Listed under "not verified" in docs/physics/spins.md.

Sources
  Govindaraju 2000   Govindaraju V, Young K, Maudsley AA. Proton NMR chemical shifts and coupling constants for
                     brain metabolites. NMR Biomed 13:129-153.
  Govindaraju 2015   Govind V, Young K, Maudsley AA. Corrigendum to the above. NMR Biomed 28:923-924 (GABA, choline).
  Near 2013          Near J, Evans CJ, Puts NAJ, Barker PB, Edden RAE. J-difference editing of GABA: simulated and
                     experimental multiplet patterns. Magn Reson Med 70:1183-1191, Table 1 (read in the open copy).
  Hamilton 2011      Hamilton G et al. In vivo characterization of the liver fat 1H MR spectrum. NMR Biomed 24:784.

Left out of the table: exchangeable protons other than the amide of NAA (the NH of creatine at 6.65 ppm, the NH2
of glutamine), the couplings of choline protons to 14N, and couplings inside a methyl group (they change no
spectrum). Relaxation times of metabolites are not in the table: they are inputs of a simulation.

Water is placed at 4.7 ppm, the value of the engine; the copies of the literature table use 4.65 or 4.7 ppm.
"""
from __future__ import annotations

import dataclasses

from .operators import SpinSystem

WATER_PPM = 4.7

# water relaxation at 3 T and relative water content, as quoted by Harris, Puts & Edden 2015 (J Magn Reson Imaging
# 42:1431) and Near et al. 2021 (NMR Biomed 34:e4257); both read in their open copies  [F]
WATER_3T = dict(gm=dict(water=0.78, t1_ms=1331.0, t2_ms=110.0), wm=dict(water=0.65, t1_ms=832.0, t2_ms=79.2),
                csf=dict(water=0.97, t1_ms=3817.0, t2_ms=503.0))

# the engine's six-peak fat spectrum: ppm relative to water and relative amplitude (mrgym.scanner.tissue.FAT_PEAKS)  [E]
ENGINE_FAT_PEAKS = ((-3.80, 0.087), (-3.40, 0.694), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048))


@dataclasses.dataclass(frozen=True)
class Metabolite:
    name: str
    full_name: str
    systems: tuple
    source: str
    tag: str
    note: str = ""

    @property
    def protons(self):
        """Protons behind the signal (scale included): the amplitude of the pulse-acquire signal."""
        return sum(s.n_spins * s.scale for s in self.systems)

    @property
    def groups(self):
        return tuple(dict.fromkeys(g for s in self.systems for g in s.groups))


def _singlet(name, ppm, count, group=None, scale=1.0):
    return SpinSystem(name, (ppm,), (), counts=(count,), labels=(name,), groups=(group or name,), scale=scale)


def _network(name, shifts, labels, couplings, counts=None):
    return SpinSystem(name, tuple(shifts), tuple(couplings), counts=counts, labels=tuple(labels))


_G00 = "Govindaraju 2000, as copied in spant (and equal in VeSPA)"

BASIS = (
    Metabolite("water", "water", (_singlet("water", WATER_PPM, 2),), "engine convention (4.7 ppm); spant 4.65, FID-A 4.7", "E"),
    Metabolite("lactate", "lactate", (_network("lactate", (1.3142, 4.0974), ("CH3", "CH"), ((0, 1, 6.933),), (3, 1)),), _G00, "F"),
    Metabolite("alanine", "alanine", (_network("alanine", (1.4667, 3.7746), ("CH3", "CH"), ((0, 1, 7.234),), (3, 1)),), _G00, "F"),
    Metabolite("naa", "N-acetylaspartate",
               (_singlet("naa_acetyl", 2.008, 3),
                _network("naa_aspartyl", (4.3817, 2.6727, 2.4863, 7.8205), ("CH", "CH2a", "CH2b", "NH"),
                         ((0, 1, 3.861), (0, 2, 9.821), (1, 2, -15.592), (0, 3, 6.4)))),
               _G00 + ", the entry with the amide proton", "F"),
    Metabolite("creatine", "creatine", (_singlet("cr_ch3", 3.027, 3), _singlet("cr_ch2", 3.913, 2)), _G00, "F",
               "NH at 6.649 ppm left out"),
    Metabolite("choline", "choline",
               (_singlet("cho_ch3", 3.185, 9),
                _network("cho_ch2", (4.054, 4.054, 3.501, 3.501), ("1CH2a", "1CH2b", "2CH2a", "2CH2b"),
                         ((0, 1, -14.1), (0, 2, 3.14), (0, 3, 6.979), (1, 2, 7.011), (1, 3, 3.168), (2, 3, -14.07)))),
               "Govindaraju 2015 (corrigendum), as copied in spant", "F", "couplings to 14N (0.57, 2.572, 2.681 Hz) left out"),
    Metabolite("gaba", "gamma-aminobutyric acid",
               (_network("gaba", (3.0128, 3.0128, 1.889, 1.889, 2.284, 2.284), ("4CH2a", "4CH2b", "3CH2a", "3CH2b", "2CH2a", "2CH2b"),
                         ((0, 1, -12.021), (0, 2, 5.372), (0, 3, 7.127), (1, 2, 10.578), (1, 3, 6.982), (2, 3, -13.121),
                          (2, 4, 7.755), (2, 5, 7.432), (3, 4, 6.173), (3, 5, 7.933), (4, 5, -10.744))),),
               "Govindaraju 2015 (corrigendum), as copied in spant (and equal in VeSPA)", "F"),
    Metabolite("gaba_near2013", "gamma-aminobutyric acid (Near 2013)",
               (_network("gaba_near2013", (2.284, 2.284, 1.888, 1.888, 3.013, 3.013), ("2CH2a", "2CH2b", "3CH2a", "3CH2b", "4CH2a", "4CH2b"),
                         ((0, 1, -15.938), (0, 2, 7.678), (0, 3, 6.980), (1, 2, 6.980), (1, 3, 7.678), (2, 3, -15.000),
                          (2, 4, 8.510), (2, 5, 6.503), (3, 4, 6.503), (3, 5, 8.510), (4, 5, -14.062))),),
               "Near 2013, Table 1; equal in spant", "F", "an alternative set; not part of the default basis"),
    Metabolite("glutamate", "glutamate",
               (_network("glutamate", (3.7433, 2.0375, 2.1200, 2.3378, 2.352), ("2CH", "3CH2a", "3CH2b", "4CH2a", "4CH2b"),
                         ((0, 1, 7.331), (0, 2, 4.651), (1, 2, -14.849), (1, 3, 6.413), (1, 4, 8.406), (2, 3, 8.478), (2, 4, 6.875), (3, 4, -15.915))),),
               _G00, "F"),
    Metabolite("glutamine", "glutamine",
               (_network("glutamine", (3.753, 2.129, 2.109, 2.432, 2.454), ("2CH", "3CH2a", "3CH2b", "4CH2a", "4CH2b"),
                         ((0, 1, 5.847), (0, 2, 6.5), (1, 2, -14.504), (1, 3, 9.165), (1, 4, 6.347), (2, 3, 6.324), (2, 4, 9.209), (3, 4, -15.371))),),
               _G00, "F", "NH2 at 6.816 and 7.529 ppm left out"),
    Metabolite("myo_inositol", "myo-inositol",
               (_network("myo_inositol", (3.5217, 4.0538, 3.5217, 3.6144, 3.269, 3.6144), ("1CH", "2CH", "3CH", "4CH", "5CH", "6CH"),
                         ((0, 1, 2.889), (0, 5, 9.998), (1, 2, 3.006), (2, 3, 9.997), (3, 4, 9.485), (4, 5, 9.482))),),
               _G00, "F"),
    Metabolite("lipid_pair", "lipid chain end: methyl coupled to its methylene",
               (_network("lipid_pair", (0.90, 1.30), ("CH3", "CH2"), ((0, 1, 7.0),), (3, 2)),),
               "shifts of the methyl and methylene peaks of Hamilton 2011 (Table 1, read); the coupling of 7 Hz is recalled, not read", "M",
               "a two-group model: the methylene is coupled to the methyl only, not to the rest of the chain"),
    Metabolite("triglyceride", "six-peak triglyceride spectrum of the engine",
               tuple(_singlet(f"fat_{k + 1}", round(WATER_PPM + ppm, 4), 1, "triglyceride", amp / sum(a for _, a in ENGINE_FAT_PEAKS))
                     for k, (ppm, amp) in enumerate(ENGINE_FAT_PEAKS)),
               "mrgym.scanner.tissue.FAT_PEAKS: the six-peak model that Hernando 2017 lists, without its room-temperature shift (the engine "
               "cites it as that since 0.3.0); the liver table of Hamilton 2011, read for this release, has 4.20 ppm and "
               "8.8 / 70.0 / 12.0 / 0.6 / 3.9 / 4.7 % where the model has 4.31 ppm and 8.7 / 69.4 / 12.8 / 0.4 / 3.9 / 4.8 %", "E",
               "uncoupled lines whose amplitudes sum to 1, as in the engine"),
)
_BY_NAME = {m.name: m for m in BASIS}
DEFAULT = tuple(m.name for m in BASIS if m.name != "gaba_near2013")
NOT_VERIFIED = (
    "lipid_pair: the coupling of 7.0 Hz between methyl and methylene [M]",
    "every brain metabolite: read in software copies (spant, VeSPA) of Govindaraju 2000 and of the 2015 corrigendum, not in the papers",
    "gaba: which of the two sets of four vicinal couplings belongs to the 3.01-1.89 ppm pair and which to the 1.89-2.28 ppm pair is "
    "taken from spant and VeSPA; the corrigendum was not read",
)


def get(name):
    if name not in _BY_NAME:
        raise KeyError(f"unknown metabolite {name!r}; the basis holds {', '.join(_BY_NAME)}")
    return _BY_NAME[name]


def basis(names=None):
    """The metabolites of ``names`` (default: the default basis, every entry but the alternative GABA set)."""
    return tuple(get(n) for n in (names or DEFAULT))


def table_markdown():
    """The metabolite table of docs/physics/spins.md, generated from ``BASIS``."""
    out = ["| Metabolite | Network | Site: shift in ppm (protons) | Couplings in Hz | Source | |", "|---|---|---|---|---|---|"]
    for m in BASIS:
        for s in m.systems:
            sites = ", ".join(f"{lab} {ppm:g}" + (f" ({c})" if c > 1 else "") for lab, ppm, c in zip(s.labels, s.shifts_ppm, s.counts))
            js = ", ".join(f"{s.labels[i]}-{s.labels[j]} {v:g}" for i, j, v in s.j_hz) or "none"
            scale = f" x {s.scale:.4g}" if s.scale != 1 else ""
            out.append(f"| {m.name} | {s.name}{scale} | {sites} | {js} | {m.source}{'; ' + m.note if m.note else ''} | [{m.tag}] |")
    return "\n".join(out)
