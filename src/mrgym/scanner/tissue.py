"""Tissue table of the hidden world: PD, T1, T2, T2*, ADC per tissue at 1.5 T and 3 T, with a source tag per row.

How far each number was checked is tagged:
  [A]  the value is stated in the abstract of the cited paper
  [T]  the value is attributed to a table of the cited paper and was not re-checked against the full text for this
       release; treat it as "of the right size", not as a citation-grade number
  [C]  chosen by the authors of this package inside a plausible range; there is no source
A number without a tag in its row is [C]; that includes most proton densities. The diffusion coefficients have their
own table of sources below (``ADC_SOURCES``): nine of them lie against a fetched abstract or a formula, sixteen are [C].
Times in ms. PD relative to water = 1. T2' is derived: 1/T2' = 1/T2* - 1/T2. Fat rows carry the chemical shift.

Sources (full records in docs/physics/references.md)
  Wright 2008        MAGMA 21:121, doi:10.1007/s10334-008-0104-8    WM / GM / putamen / caudate T1 at 1.5 and 3 T      [A]
  Rooney 2007        MRM 57:308, doi:10.1002/mrm.21122              CSF T1 4.3 +- 0.2 s, independent of field          [A]
  Wansapura 1999     JMRI 9:531                                     3 T T2 of GM and WM: the abstract is reported to print
                                                                    grey 80 / white 110 ms, the opposite of the values
                                                                    used here, so the T2 of WM and GM below are      [C]
  Peters 2007        MRI 25:748, doi:10.1016/j.mri.2007.02.014      T2* of GM and WM at 1.5 and 3 T                    [T]
  Stanisz 2005       MRM 54:507, doi:10.1002/mrm.20605              muscle, blood, cord T1 / T2 at 1.5 and 3 T         [T]
  de Bazelaire 2004  Radiology 230:652, doi:10.1148/radiol.2303021331   abdomen: 3 T values [A], 1.5 T values [T]
  Hernando 2017      MRM 77:1516, doi:10.1002/mrm.26228 (PMC4835219)  six-peak fat model: shifts and amplitudes, read in
                                                                    the full text (Methods, image reconstruction)      [F]
  Hamilton 2011      NMR Biomed 24:784, doi:10.1002/nbm.1622        the measured liver fat spectrum (Table 3) that such
                                                                    models describe; NOT the list used here (below)
  [F]  the value is stated in the full text of the cited paper, which was read for it
"""

FAT_PPM = -3.4              # main methylene peak (1.3 ppm) relative to water (4.7 ppm)
# Six-peak fat model, ppm relative to water and relative amplitude. Source: Hernando 2017, which lists the model it
# reconstructs with as shifts of -3.90, -3.50, -2.70, -2.04, -0.49, 0.50 ppm with amplitudes 0.087, 0.694, 0.128, 0.004,
# 0.039, 0.048, after "a common frequency shift of 0.1 ppm" for a phantom at room temperature (3.50 ppm between water
# and the main peak). The list here is that model without the room-temperature shift: 3.40 ppm, the body's.
# It is not Table 3 of Hamilton 2011, although engine 0.2.0 and before labelled it so (and carried 0.693 for the main
# peak). That table has 8.8, 70.0, 12.0, 0.6, 3.9 and 4.7 % at 0.9, 1.3, 2.1, 2.75, 4.2 and 5.3 ppm with water at 4.7:
# the amplitudes are within 0.8 points of the model's, five positions within 0.01 ppm, and the glycerol peak, which
# the table has at 4.2 ppm, sits at 4.31 ppm in the model (``tests/test_cited_tables.py`` holds both lists).
FAT_PEAKS = ((-3.80, 0.087), (-3.40, 0.694), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048))

#            name            PD     T1@1.5  T1@3   T2@1.5 T2@3  T2*@1.5 T2*@3  ADC   fat
_T = [
    # ---- head
    ("wm",           0.69,   646,   838,    85,    75,    66,    53,   0.75, 0),   # T1 Wright [A]; T2 [C] (see Wansapura above); T2* Peters [T]; PD [T]
    ("gm",           0.82,  1197,  1607,   100,    95,    84,    66,   0.85, 0),   # same sources
    ("deep_gm",      0.80,  1090,  1360,    85,    70,    60,    42,   0.80, 0),   # putamen / caudate T1 Wright [A]; T2, T2* [C] (iron)
    ("csf",          1.00,  4300,  4300,  2000,  2000,  1000,   800,   3.00, 0),   # T1 Rooney [A]; T2 [T] 1.5-2.5 s
    ("scalp_fat",    0.95,   343,   382,    58,    68,    40,    35,   0.05, 1),   # subcutaneous fat, de Bazelaire: 3 T [A], 1.5 T [T]
    ("skull",        0.08,   400,   450,    10,     8,     2,   1.5,   0.10, 0),   # cortical bone: T2 < 1 ms, practically no signal [C]
    ("diploe",       0.45,   549,   586,    49,    49,    20,    12,   0.20, 1),   # fatty marrow, de Bazelaire L4 marrow: 3 T [A], 1.5 T [T]
    ("skin",         0.65,   900,  1100,    35,    30,    25,    20,   0.80, 0),   # [C]
    ("muscle",       0.80,  1008,  1412,    44,    50,    30,    25,   1.40, 0),   # Stanisz [T]
    ("lesion_ms",    0.88,  1100,  1500,   160,   150,   110,    90,   1.10, 0),   # demyelinating plaque: longer T1 and T2 than WM [C]
    ("tumour",       0.88,  1500,  2000,   140,   130,   100,    80,   1.20, 0),   # [C]
    ("oedema",       0.86,  1300,  1700,   190,   170,   130,   100,   1.50, 0),   # vasogenic oedema [C]
    ("blood",        0.85,  1441,  1932,   290,   275,   150,    60,   2.50, 0),   # Stanisz [T]: an in-vitro value; in vivo is reported lower (not adopted here); no flow modelled
    # ---- abdomen (de Bazelaire 2004: 3 T [A], 1.5 T [T]; T2* [C])
    ("liver",        0.75,   586,   809,    46,    34,    28,    18,   1.10, 0),
    ("spleen",       0.78,  1057,  1328,    79,    61,    50,    35,   0.90, 0),
    ("kidney_cortex", 0.80,  966,  1142,    87,    76,    60,    50,   2.20, 0),
    ("kidney_medulla", 0.84, 1412, 1545,    85,    81,    60,    45,   2.00, 0),
    ("pancreas",     0.75,   584,   725,    46,    43,    30,    25,   1.40, 0),
    ("abd_muscle",   0.80,   856,   898,    27,    29,    22,    20,   1.40, 0),
    ("marrow",       0.60,   549,   586,    49,    49,    15,    10,   0.30, 1),
    ("subcut_fat",   0.95,   343,   382,    58,    68,    40,    35,   0.05, 1),
    ("visceral_fat", 0.92,   343,   382,    58,    68,    35,    30,   0.05, 1),
    ("bowel_fluid",  0.95,  2500,  3000,   600,   500,   200,   120,   2.50, 0),   # [C]
    ("cord",         0.75,   745,   993,    74,    78,    55,    45,   0.90, 0),   # Stanisz [T]
    ("liver_lesion", 0.85,  1200,  1500,   120,   100,    70,    50,   1.60, 0),   # [C] (cyst-like / metastasis-like)
]
NAMES = [r[0] for r in _T]

# ---------------------------------------------------------------------------------------------- diffusion coefficients
# The ADC column of ``_T`` (um^2/ms = 1e-3 mm^2/s) is older than this table and is read by every sequence in which
# spins diffuse: the RF-spoiled steady state of ``spgr`` and ``afi`` (attenuation across the spoiler), and ``dwse``
# (signal x exp(-b ADC)). A tissue has one coefficient, so ``dwse`` reads the same column; its numbers are NOT changed
# here (a changed number would move the data of ``spgr``). What follows is additive: where each number stands against
# a source, and the self-diffusion of water with temperature for phantom liquids. One free, isotropic coefficient
# per class: no tensor, no kurtosis, no perfusion fraction (docs/physics/limits.md, block 3).
#
# name: (tag, source, (low, high) of the source in um^2/ms or None, what the source states)
#   [A]  stated in the abstract of the cited paper, which was fetched for this release (Europe PMC, 2026-11-27); the
#        interval is mean +- 1 SD of the abstract, or its stated range
#   [R]  computed from a formula recalled from the cited paper, which was NOT opened; see ``water_adc_um2_ms``
#   [C]  chosen by the authors of this package inside a plausible range; there is no source
_HELENIUS = "Helenius 2002 (AJNR 23:194, PMID 11847041; 80 healthy adults, 1.5 T, b = 0 and 1000 s/mm^2)"
_KILICKESMEZ = "Kilickesmez 2008 (Diagn Interv Radiol 14:83, PMID 18553281; 50 normal abdomens, b = 0, 500, 1000 s/mm^2)"
ADC_SOURCES = dict(
    wm=("A", _HELENIUS, (0.62, 0.79), "white matter 0.70 +- 0.03, range 0.62 to 0.79"),
    gm=("A", _HELENIUS, (0.78, 1.09), "cortical grey matter 0.89 +- 0.04, range 0.78 to 1.09"),
    deep_gm=("A", _HELENIUS, (0.64, 0.83), "deep grey matter 0.75 +- 0.03, range 0.64 to 0.83"),
    csf=("R", "Holz 2000: free water at 37 C", (2.9, 3.2), "3.04 for free water at 37 C by the formula of ``water_adc_um2_ms``; CSF is taken as free water"),
    liver=("A", _KILICKESMEZ, (1.14, 1.98), "1.34 +- 0.20 to 1.77 +- 0.21 over four segments"),
    spleen=("A", _KILICKESMEZ, (0.90, 1.66), "1.28 +- 0.38"),
    kidney_cortex=("A", _KILICKESMEZ, (1.86, 2.30), "2.08 +- 0.22"),
    kidney_medulla=("A", _KILICKESMEZ, (1.76, 2.12), "1.94 +- 0.18"),
    pancreas=("A", _KILICKESMEZ, (1.21, 1.97), "1.59 +- 0.38 (tail) to 1.68 +- 0.26 (body)"),
    bowel_fluid=("C", None, None, "a fluid between tissue and free water"),
    scalp_fat=("C", None, None, "lipid protons diffuse about two orders of magnitude more slowly than water"),
    subcut_fat=("C", None, None, "as scalp_fat"), visceral_fat=("C", None, None, "as scalp_fat"),
    diploe=("C", None, None, "fatty marrow"), marrow=("C", None, None, "fatty marrow"),
    skull=("C", None, None, "cortical bone has practically no signal at any b-value"),
    skin=("C", None, None, None), muscle=("C", None, None, None), abd_muscle=("C", None, None, None),
    blood=("C", None, None, "no flow is modelled; the number stands for stationary blood"), cord=("C", None, None, None),
    lesion_ms=("C", None, None, "a demyelinating plaque: above normal white matter"),
    tumour=("C", None, None, None), oedema=("C", None, None, "vasogenic oedema: well above white matter"),
    liver_lesion=("C", None, None, "cyst-like"),
)
# Phantom liquids (``anatomy.py``: the fills, vials and spheres of the QA, ACR and NIST worlds) carry 2.0 um^2/ms or a
# draw around it: [C] declared, and 1 % below free water at 20 C by the formula below; the doping of a phantom solution
# changes relaxation, not diffusion, at these concentrations (an assumption, not sourced).


def water_adc_um2_ms(temp_c):
    """Self-diffusion coefficient of water (um^2/ms) at ``temp_c`` degrees Celsius: D = D0 (T / Ts - 1)^gamma with
    D0 = 1.635e-8 m^2/s, Ts = 215.05 K, gamma = 2.063, T in kelvin (the fit of Holz, Heil & Sacco 2000, Phys Chem Chem
    Phys 2:4740, stated for 0 to 100 C). Tag [R]: the constants are recalled; the paper was not opened and its
    abstract carries no numbers. What was checked: the formula gives 2.299 at 25 C, the value attributed to Mills
    1973 (J Phys Chem 77:685; also not opened), 2.02 at 20 C and 3.04 at 37 C."""
    t = float(temp_c) + 273.15
    if not 273.15 <= t <= 373.15:
        raise ValueError("the fit is stated for 0 to 100 degrees Celsius")
    return 1.635e-8 * (t / 215.05 - 1.0) ** 2.063 * 1e9


def adc_report(b0_t=3.0):
    """Every tissue with its coefficient against its source: list of dict(name, adc_um2_ms, tag, source, low, high,
    inside (None without an interval), note)."""
    tab, out = table(b0_t), []
    for name in NAMES:
        tag, source, span, note = ADC_SOURCES[name]
        adc = tab[name]["adc_um2_ms"]
        out.append(dict(name=name, adc_um2_ms=adc, tag=tag, source=source, low=None if span is None else span[0], high=None if span is None else span[1],
                        inside=None if span is None else bool(span[0] <= adc <= span[1]), note=note))
    return out


def table(b0_t):
    """Tissue dictionary at 1.5 T or 3 T: name -> dict(pd, t1_ms, t2_ms, t2s_ms, t2p_ms, adc_um2_ms, fat)."""
    if b0_t not in (1.5, 3.0):
        raise ValueError("tissue table holds 1.5 T and 3 T")
    i = 0 if b0_t == 1.5 else 1
    out = {}
    for name, pd, t1a, t1b, t2a, t2b, sa, sb, adc, fat in _T:
        t1, t2, t2s = (t1a, t1b)[i], (t2a, t2b)[i], (sa, sb)[i]
        out[name] = dict(pd=pd, t1_ms=float(t1), t2_ms=float(t2), t2s_ms=float(t2s),
                         t2p_ms=1.0 / (1.0 / t2s - 1.0 / t2), adc_um2_ms=adc, fat=bool(fat))
    return out
