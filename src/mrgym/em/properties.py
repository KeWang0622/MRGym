"""Permittivity, conductivity and density of the engine's tissue classes at radio frequencies.

Dispersion model (Gabriel S, Lau RW, Gabriel C 1996, Phys Med Biol 41:2271, "III. Parametric models for the
dielectric spectrum of tissues"): four Cole-Cole terms and a static conductivity,

    eps(w) = ef + sum_{n=1..4} del_n / (1 + (i w tau_n)^(1 - alf_n)) + sig / (i w eps0),

whose real part is the relative permittivity and whose imaginary part is -sigma / (w eps0).

How far each number was checked:
  [F]  the parameter row was fetched and read for this module: Appendix C (summary table) of C. Gabriel and
       S. Gabriel, "Compilation of the dielectric properties of body tissues at RF and microwave frequencies",
       report AL/OE-TR-1996-0037 (Brooks AFB 1996), in the HTML edition of IFAC-CNR (archived copy of
       niremf.ifac.cnr.it/docs/DIELECTRIC/AppendixC.html, and the live table at webnir.eu, "Model parameters").
       ``CHECK_VALUES`` holds the permittivity, conductivity, wavelength and penetration depth that the IFAC-CNR
       calculator (webnir.eu, "All tissues, single frequency") returned at 64, 128 and 298 MHz for the same rows;
       rows P1 and P2 of ``validate`` compare this package's evaluation with them. The calculator evaluates the same
       model, so the comparison checks the transcription of the parameters and the code, not the measurements. The
       1996 journal paper itself was not read.
  [f]  the row was fetched as above, but the calculator has no such row, so there is no tabulated check value
  [D]  densities: IT'IS Foundation tissue properties database V5.0 (doi:10.13099/VIP21000-05-0), column "Density
       (kg/m3)", average; fetched and read
  [C]  chosen by the authors of this package: the class is mapped to the row of another tissue, or the value is a
       declared estimate. There is no source for it.
The values are those of excised tissue near body temperature as compiled in 1996; later work revises several of them
(the IT'IS database lists lower conductivities for some tissues at low frequency). They are not patient values.
"""
from __future__ import annotations

import math

import numpy as np

from .units import EPS0

# row: ef, del1, tau1 [ps], alf1, del2, tau2 [ns], alf2, sig [S/m], del3, tau3 [us], alf3, del4, tau4 [ms], alf4
COLE_COLE = {
    "Brain (Grey Matter)":       (4.0, 45.0, 7.958, 0.10, 400.0, 15.915, 0.15, 0.02, 2.0e5, 106.103, 0.22, 4.5e7, 5.305, 0.00),
    "Brain (White Matter)":      (4.0, 32.0, 7.958, 0.10, 100.0, 7.958, 0.10, 0.02, 4.0e4, 53.052, 0.30, 3.5e7, 7.958, 0.02),
    "Cerebro Spinal Fluid":      (4.0, 65.0, 7.958, 0.10, 40.0, 1.592, 0.00, 2.0, 0.0, 159.155, 0.00, 0.0, 15.915, 0.00),
    "Fat (Not Infiltrated)":     (2.5, 3.0, 7.958, 0.20, 15.0, 15.915, 0.10, 0.01, 3.3e4, 159.155, 0.05, 1.0e7, 7.958, 0.01),
    "Fat (Average Infiltrated)": (2.5, 9.0, 7.958, 0.20, 35.0, 15.915, 0.10, 0.035, 3.3e4, 159.155, 0.05, 1.0e7, 15.915, 0.01),
    "Bone (Cortical)":           (2.5, 10.0, 13.263, 0.20, 180.0, 79.577, 0.20, 0.02, 5.0e3, 159.155, 0.20, 1.0e5, 15.915, 0.00),
    "Bone (Cancellous)":         (2.5, 18.0, 13.263, 0.22, 300.0, 79.577, 0.25, 0.07, 2.0e4, 159.155, 0.20, 2.0e7, 15.915, 0.00),
    "Bone Marrow (Infiltrated)": (2.5, 9.0, 14.469, 0.20, 80.0, 15.915, 0.10, 0.1, 1.0e4, 1591.549, 0.10, 2.0e6, 15.915, 0.10),
    # the archived table prints sig of dry skin as 0.000; 0.0002 is the value of the live table
    "Skin (Dry)":                (4.0, 32.0, 7.234, 0.00, 1100.0, 32.481, 0.20, 0.0002, 0.0, 159.155, 0.20, 0.0, 15.915, 0.20),
    "Muscle":                    (4.0, 50.0, 7.234, 0.10, 7000.0, 353.678, 0.10, 0.2, 1.2e6, 318.310, 0.10, 2.5e7, 2.274, 0.00),
    "Blood":                     (4.0, 56.0, 8.377, 0.10, 5200.0, 132.629, 0.10, 0.7, 0.0, 159.155, 0.20, 0.0, 15.915, 0.00),
    "Liver":                     (4.0, 39.0, 8.842, 0.10, 6000.0, 530.516, 0.20, 0.02, 5.0e4, 22.736, 0.20, 3.0e7, 15.915, 0.05),
    "Spleen":                    (4.0, 48.0, 7.958, 0.10, 2500.0, 63.662, 0.15, 0.03, 2.0e5, 265.258, 0.25, 5.0e7, 6.366, 0.00),
    "Kidney":                    (4.0, 47.0, 7.958, 0.10, 3500.0, 198.944, 0.22, 0.05, 2.5e5, 79.577, 0.22, 3.0e7, 4.547, 0.00),
    "Nerve":                     (4.0, 26.0, 7.958, 0.10, 500.0, 106.103, 0.15, 0.006, 7.0e4, 15.915, 0.20, 4.0e7, 15.915, 0.00),
    "Thyroid":                   (4.0, 55.0, 7.958, 0.10, 2500.0, 159.155, 0.10, 0.5, 1.0e5, 159.155, 0.20, 4.0e7, 15.915, 0.00),
    "Vitreous Humor":            (4.0, 65.0, 7.234, 0.00, 30.0, 159.155, 0.10, 1.5, 0.0, 159.155, 0.00, 0.0, 15.915, 0.00),
}

# What the IFAC-CNR calculator returned for these rows  [F]:
# {row: {MHz: (relative permittivity, conductivity S/m, wavelength m, penetration depth m)}}, six significant digits
CHECK_VALUES = {
    "Brain (Grey Matter)":     {64: (97.4294, 0.510857, 0.402513, 0.120935), 128: (73.5204, 0.586717, 0.244219, 0.0867748), 298: (60.0902, 0.691433, 0.123256, 0.0626666)},
    "Brain (White Matter)":    {64: (67.8358, 0.291498, 0.501975, 0.169947), 128: (52.5338, 0.342144, 0.297779, 0.122039), 298: (43.8209, 0.412692, 0.146572, 0.0882914)},
    "Cerebro Spinal Fluid":    {64: (97.3124, 2.06597, 0.252986, 0.0475786), 128: (84.0406, 2.14301, 0.166342, 0.0348799), 298: (72.7841, 2.22393, 0.0947626, 0.0253418)},
    "Fat (Not Infiltrated)":   {64: (6.50636, 0.0352754, 1.54606, 0.455966), 128: (5.92154, 0.03687, 0.892036, 0.378047), 298: (5.6354, 0.0395386, 0.414967, 0.325506)},
    "Bone (Cortical)":         {64: (16.6812, 0.0595246, 1.04355, 0.400334), 128: (14.7171, 0.0673513, 0.583606, 0.316326), 298: (13.4463, 0.0824911, 0.269913, 0.239863)},
    "Bone (Cancellous)":       {64: (30.8685, 0.160846, 0.716077, 0.215904), 128: (26.2837, 0.179877, 0.418176, 0.165298), 298: (23.181, 0.215176, 0.201711, 0.123047)},
    "Skin (Dry)":              {64: (92.1679, 0.43574, 0.422901, 0.134947), 128: (65.437, 0.522692, 0.258824, 0.0919076), 298: (49.9015, 0.640377, 0.133834, 0.0623148)},
    "Muscle":                  {64: (72.2347, 0.688202, 0.396897, 0.0910411), 128: (63.4948, 0.719224, 0.244985, 0.0705662), 298: (58.2291, 0.769995, 0.123499, 0.0561623)},
    "Blood":                   {64: (86.4441, 1.20666, 0.317183, 0.0649734), 128: (73.159, 1.24862, 0.204183, 0.0487701), 298: (65.6889, 1.31544, 0.109538, 0.0370647)},
    "Liver":                   {64: (80.5595, 0.447974, 0.436846, 0.127072), 128: (64.2507, 0.510886, 0.261401, 0.0931045), 298: (53.5713, 0.608918, 0.130682, 0.0671153)},
    "Spleen":                  {64: (110.559, 0.743898, 0.355653, 0.0939921), 128: (82.8917, 0.835168, 0.22008, 0.0676469), 298: (66.577, 0.967836, 0.11423, 0.0483075)},
    "Kidney":                  {64: (118.556, 0.7413, 0.350044, 0.0958329), 128: (89.6168, 0.852295, 0.21419, 0.06811), 298: (70.5905, 1.02087, 0.111006, 0.047128)},
    "Nerve":                   {64: (55.0621, 0.312167, 0.52603, 0.151438), 128: (44.0653, 0.353794, 0.315132, 0.111521), 298: (36.9512, 0.417769, 0.157423, 0.0812062)},
    "Thyroid":                 {64: (73.9472, 0.778298, 0.379498, 0.0841929), 128: (66.7839, 0.804159, 0.235402, 0.0656823), 298: (62.4717, 0.850726, 0.118842, 0.0528245)},
    "Vitreous Humor":          {64: (69.1264, 1.50315, 0.297171, 0.0556701), 128: (69.0619, 1.50536, 0.194016, 0.042572), 298: (69.0169, 1.5175, 0.104982, 0.0335235)},
}

# engine tissue class -> (row of COLE_COLE, density kg/m^3, tag of the row or of the mapping, note)
CLASS_ROWS = {
    # ---- head
    "wm":             ("Brain (White Matter)", 1041.0, "F", ""),
    "gm":             ("Brain (Grey Matter)", 1044.5, "F", ""),
    "deep_gm":        ("Brain (Grey Matter)", 1044.5, "C", "deep grey matter takes the grey-matter row"),
    "csf":            ("Cerebro Spinal Fluid", 1007.0, "F", ""),
    "scalp_fat":      ("Fat (Not Infiltrated)", 911.0, "F", "subcutaneous fat; the choice between the two fat rows is ours"),
    "skull":          ("Bone (Cortical)", 1908.0, "F", ""),
    "diploe":         ("Bone (Cancellous)", 1178.3, "F", ""),
    "skin":           ("Skin (Dry)", 1109.0, "F", ""),
    "muscle":         ("Muscle", 1090.4, "F", ""),
    "blood":          ("Blood", 1049.75, "F", ""),
    "lesion_ms":      ("Brain (Grey Matter)", 1044.5, "C", "no dispersion model of a plaque was read: the grey-matter row (more water than white matter)"),
    "tumour":         ("Brain (Grey Matter)", 1044.5, "C", "as lesion_ms"),
    "oedema":         ("Brain (Grey Matter)", 1044.5, "C", "as lesion_ms"),
    # ---- abdomen
    "liver":          ("Liver", 1078.75, "F", ""),
    "spleen":         ("Spleen", 1089.0, "F", ""),
    "kidney_cortex":  ("Kidney", 1066.25, "F", "one kidney row for cortex and medulla"),
    "kidney_medulla": ("Kidney", 1066.25, "F", "one kidney row for cortex and medulla"),
    "pancreas":       ("Thyroid", 1086.5, "F", "the calculator's pancreas is its gland row, which is the thyroid row"),
    "abd_muscle":     ("Muscle", 1090.4, "F", ""),
    "marrow":         ("Bone Marrow (Infiltrated)", 1028.5, "f", "red marrow; density of the IT'IS row Bone Marrow (Red)"),
    "subcut_fat":     ("Fat (Not Infiltrated)", 911.0, "F", "the choice between the two fat rows is ours"),
    "visceral_fat":   ("Fat (Average Infiltrated)", 911.0, "f", "the choice between the two fat rows is ours"),
    "bowel_fluid":    ("Vitreous Humor", 1004.5, "C", "a body fluid: the calculator's body-fluid row is the vitreous-humor row"),
    "cord":           ("Nerve", 1075.0, "F", "the calculator's spinal cord is the nerve row"),
    "liver_lesion":   ("Vitreous Humor", 1004.5, "C", "cyst-like: the body-fluid row"),
}

# Test objects: declared values, no source [C]. Water has no dispersion below 1 GHz (its relaxation is near 19 GHz);
# 78 is its static permittivity near room temperature. The conductivities are estimates: 0.9 S/m for the ACR filling
# (10 mM NiCl2 + 75 mM NaCl, from the molar conductivities of the two salts near 20 C), 0.5 S/m for the doped
# solutions of the other test objects. Oil: permittivity 3, no conductivity.
PHANTOM = {
    "acr_fill":     (78.0, 0.9, 1000.0),
    "phantom_fill": (78.0, 0.5, 1000.0),
    "oil":          (3.0, 0.0, 920.0),
}


def cole_cole(row, f_hz):
    """(relative permittivity, conductivity in S/m) of a row of ``COLE_COLE`` at the frequency ``f_hz`` (Hz)."""
    ef, d1, t1, a1, d2, t2, a2, sig, d3, t3, a3, d4, t4, a4 = COLE_COLE[row]
    w = 2.0 * math.pi * float(f_hz)
    eps = complex(ef) + sig / (1j * w * EPS0)
    for d, tau, alf in ((d1, t1 * 1e-12, a1), (d2, t2 * 1e-9, a2), (d3, t3 * 1e-6, a3), (d4, t4 * 1e-3, a4)):
        eps += d / (1.0 + (1j * w * tau) ** (1.0 - alf))
    return float(eps.real), float(-eps.imag * w * EPS0)


def tissue(name, f_hz, fat=False, kind=None):
    """Properties of one engine tissue class: dict(eps_r, sigma [S/m], density [kg/m^3], row, tag).

    Classes of the test objects (``body``, ``vial*``, ``insert*``, ``sphere*``, ``fill``) take the declared values of
    ``PHANTOM``: the ACR filling for ``kind="acr"``, oil for a class whose tissue record says ``fat``."""
    if name in CLASS_ROWS:
        row, density, tag, note = CLASS_ROWS[name]
        eps_r, sigma = cole_cole(row, f_hz)
        return dict(eps_r=eps_r, sigma=sigma, density=density, row=row, tag=tag, note=note)
    if name in PHANTOM:
        key = name
    elif name == "fill" or name == "body" or name.startswith(("vial", "insert", "sphere")):
        key = "oil" if fat else ("acr_fill" if kind == "acr" else "phantom_fill")
    else:
        raise KeyError(f"no dielectric properties for the tissue class {name!r}")
    eps_r, sigma, density = PHANTOM[key]
    return dict(eps_r=eps_r, sigma=sigma, density=density, row=key, tag="C", note="declared, see PHANTOM")


def table(f_hz, names=None):
    """Properties of every engine tissue class (or of ``names``) at one frequency: {class: record of ``tissue``}."""
    return {k: tissue(k, f_hz) for k in (names or CLASS_ROWS)}


def world_maps(world, f_hz, overrides=None):
    """Relative permittivity, conductivity (S/m) and density (kg/m^3) of a world on its master grid.

    A pixel shared by several classes takes the weighted mean of each (air fills what the weights leave: permittivity
    1, no conductivity, no mass). For E along z this is the mixing rule of the long-wavelength limit (``solver``, module
    docstring).
    ``overrides``: {class: dict(eps_r=, sigma=, density=)} replaces values, for a body of hidden properties."""
    wts = np.asarray(world["weights"], float)
    air = np.clip(1.0 - wts.sum(axis=0), 0.0, 1.0)
    eps, sig, rho = air.copy(), np.zeros(air.shape), np.zeros(air.shape)
    for k, name in enumerate(world["names"]):
        rec = tissue(name, f_hz, fat=bool(world["tissues"].get(name, {}).get("fat", False)), kind=world.get("kind"))
        rec = dict(rec, **((overrides or {}).get(name) or {}))
        eps += wts[k] * rec["eps_r"]
        sig += wts[k] * rec["sigma"]
        rho += wts[k] * rec["density"]
    return eps, sig, rho
