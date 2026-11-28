"""Tissue table of the hidden world: PD, T1, T2, T2*, ADC per tissue at 1.5 T and 3 T, with a source tag per row.

How far each number was checked is tagged:
  [A]  the value is stated in the abstract of the cited paper
  [T]  the value is attributed to a table of the cited paper and was not re-checked against the full text for this
       release; treat it as "of the right size", not as a citation-grade number
  [C]  chosen by the authors of this package inside a plausible range; there is no source
A number without a tag in its row is [C]; that includes most proton densities. The diffusion coefficients have their
own table of sources below (``ADC_SOURCES``): 23 of them lie against a source that was opened, 2 are declared, and 7
of the 23 lie OUTSIDE their source and are marked so (``ADC_OUTSIDE_SOURCE``).
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
    ("liver",        0.75,   586,   809,    46,    34,    28,    18,   1.10, 0),   # ADC 1.10 lies BELOW its source (1.34 to 1.77): ``ADC_OUTSIDE_SOURCE``
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
# The ADC column of ``_T`` (um^2/ms = 1e-3 mm^2/s) is read by every sequence in which spins diffuse: the RF-spoiled
# steady state of ``spgr``, ``afi`` and ``megre`` (attenuation across the spoiler), ``dwse`` (signal x exp(-b ADC)) and
# the crushers and gratings of ``fse``, ``mese`` and ``dwse``. A tissue has one free, isotropic coefficient: no tensor,
# no kurtosis, no perfusion fraction (docs/physics/limits.md, block 3a).
#
# name: (tag, source, (low, high) of the source in um^2/ms or None, what the source states)
#   [A]  stated in the abstract of the cited paper, which was opened for this table (Europe PMC / PubMed, 2026-11-27)
#   [F]  stated in the full text of the cited paper, which was read for it (same date)
#   [C]  chosen by the authors of this package inside a plausible range; there is no source
# The interval is the range the source states, or its mean +- 1 SD (over its regions where it has several).
# A coefficient outside its interval is NOT silently kept: ``ADC_OUTSIDE_SOURCE`` lists it with the value the source
# supports, and ``adc_report`` prints ``inside: False``. The liver was known; the other six were found when every row
# was opened. All seven keep the number of engine 0.3.0 in THIS engine, because a changed number moves the data of
# ``spgr`` and ``afi`` on every world that holds the class (abdomen; brain; the oil vials of the QA phantom, which are
# the fat row): that is an engine version (docs/VERSIONING.md), and every stored admission gate is tied to the engine
# it was measured on. CHANGELOG.md says where the corrected table and its engine version are prepared.
_HELENIUS = "Helenius 2002 (AJNR 23:194, PMID 11847041; 80 healthy adults, 1.5 T, b = 0 and 1000 s/mm^2)"
_KILICKESMEZ = "Kilickesmez 2008 (Diagn Interv Radiol 14:83, PMID 18553281; 50 normal abdomens, b = 0, 500, 1000 s/mm^2)"
_KONO = "Kono 2001 (AJNR 22:1081, PMID 11415902; 1.5 T, b = 0 to 1000 s/mm^2)"
_FAT = ("A", "Lehnert 2004 (Magn Reson Imaging 22:39, PMID 14972393); Steidle 2011 (Magn Reson Med 65:1118, PMID 21413077)", (0.012, 0.042),
        "subcutaneous fat 0.0121 to 0.0205 (Lehnert), up to 0.042 (Steidle): the 0.05 here is above both")
_MARROW = ("A", "Dietrich 2017 (NMR Biomed 30:e3333, PMID 26114411; review of 24 studies)", (0.2, 0.6),
           "normal vertebral bone marrow typically between 0.2 and 0.6")
ADC_SOURCES = dict(
    wm=("A", _HELENIUS, (0.62, 0.79), "white matter 0.70 +- 0.03, range 0.62 to 0.79"),
    gm=("A", _HELENIUS, (0.78, 1.09), "cortical grey matter 0.89 +- 0.04, range 0.78 to 1.09"),
    deep_gm=("A", _HELENIUS, (0.64, 0.83), "deep grey matter (in the full text: basal ganglia) 0.75 +- 0.03, range 0.64 to 0.83"),
    csf=("F", _HELENIUS + ", Table 1; Annet 2002 (Neuroradiology 44:326, PMID 11914809), abstract", (2.62, 3.33),
         "lateral ventricle, middle part 3.02 +- 0.16, range 2.62 to 3.33 (Helenius); 2.98 to 3.37, mean 3.185 (Annet)"),
    liver=("A", _KILICKESMEZ + "; Bruegel 2008 (Eur Radiol 18:477, PMID 17960390); Yoshikawa 2006 (AJR 187:1521, PMID 17114546)", (1.14, 1.98),
           "1.34 +- 0.20 (right posterior segment) to 1.77 +- 0.21 (left lateral) over four segments (Kilickesmez); 1.24 at b = 50, 300, 600 "
           "(Bruegel); 1.50 to 1.56 at b = 0, 600 (Yoshikawa): the 1.10 here is below all three. The value the sources support is the "
           "right posterior segment, the one least moved by the heart"),
    spleen=("A", _KILICKESMEZ, (0.90, 1.66), "1.28 +- 0.38"),
    kidney_cortex=("A", _KILICKESMEZ, (1.86, 2.30), "2.08 +- 0.22"),
    kidney_medulla=("A", _KILICKESMEZ, (1.76, 2.12), "1.94 +- 0.18"),
    pancreas=("A", _KILICKESMEZ, (1.21, 1.97), "1.59 +- 0.38 (tail) to 1.68 +- 0.26 (body)"),
    muscle=("A", "Zaraiskaya 2006 (J Magn Reson Imaging 24:402, PMID 16823776; calf, 3 T)", (1.31, 1.41), "controls 1.31 to 1.41"),
    abd_muscle=("F", "Weedall 2024 (BJR Open 6:tzae002, PMC10873271; lumbar muscles, 3 T, b = 0 and 450 s/mm^2)", (1.29, 1.94),
                "mean diffusivity 1.46 +- 0.17 (quadratus lumborum, older group) to 1.73 +- 0.21 (multifidus, younger group): the 1.40 here "
                "is below every mean of the source and inside one of its intervals; the abdominal wall itself has no source"),
    blood=("A", "Funck 2018 (Magn Reson Med 79:2752, PMID 28940621; ex vivo, body temperature)", (0.94, 1.66),
           "1.30 +- 0.18 (monopolar encoding, range 0.94 to 1.52), 1.54 +- 0.12 (flow compensated, 1.26 to 1.66): the 2.50 here is above both"),
    cord=("A", "Mamata 2005 (J Magn Reson Imaging 22:38, PMID 15971186; 1.5 T)", (0.69, 0.84),
          "0.81 +- 0.03 (C2 to C3), 0.75 +- 0.06 (C4 to C7): the 0.90 here is above both"),
    skin=("F", "Skwierawska 2024 (Diagnostics 14:934, PMC11083106; b = 50 and 750 or 800 s/mm^2)", (0.23, 0.84),
          "healthy skin 0.48 +- 0.02, range 0.23 to 0.84 (the authors call the value biased by low signal): a weak source"),
    marrow=_MARROW, diploe=_MARROW,
    scalp_fat=_FAT, subcut_fat=_FAT, visceral_fat=_FAT,
    skull=("A", "Fernandez-Seara 2002 (Biophys J 82:522, PMID 11751339; rabbit tibia, 40 C, exchange with heavy water)", (0.063, 0.093),
           "0.078 +- 0.015: the 0.10 here is above it (cortical bone has practically no signal at any b-value)"),
    lesion_ms=("A", "Roychowdhury 2000 (AJNR 21:869, PMID 10815662)", (0.85, 1.56), "non-enhancing lesions 1.3 +- 0.26, ring-enhancing 1.2 +- 0.35"),
    tumour=("A", _KONO, (0.88, 1.41), "grade II astrocytoma 1.14 +- 0.18, range 0.88 to 1.41 (glioblastoma 0.82 +- 0.13: the class here is a low-grade tumour)"),
    oedema=("F", _KONO, (1.06, 2.09), "peritumoral oedema of glioblastoma 1.42 +- 0.32, range 1.06 to 2.09 (metastasis 1.37 +- 0.41)"),
    liver_lesion=("C", None, None, "cyst-like or metastasis-like: 1.60 is the value of no lesion type (Bruegel 2008, abstract: metastases 1.22, haemangiomas "
                                   "1.92, cysts 3.02)"),
    bowel_fluid=("C", None, None, "a fluid between tissue and free water (3.04 at 37 C)"),
)
# name: the value its source supports (the mean, or the middle of the stated range), for the engine version that corrects it
ADC_OUTSIDE_SOURCE = dict(liver=1.34, blood=1.30, cord=0.81, scalp_fat=0.02, subcut_fat=0.02, visceral_fat=0.02, skull=0.08)
# Phantom liquids (``anatomy.py``: the fills, vials and spheres of the QA, ACR and NIST worlds) carry 2.0 um^2/ms or a
# draw around it: [C] declared, and 1 % below free water at 20 C by the formula below; the doping of a phantom solution
# changes relaxation, not diffusion, at these concentrations (an assumption, not sourced).


# ---------------------------------------------------------------------------------------------- magnetisation transfer
# Two pools per tissue: free water, and protons bound to macromolecules that exchange longitudinal magnetisation with it
# (the binary spin-bath model of Henkelman 1993; ``mrgym.core.operators``, pool "mt"). An ADDITIVE table: ``table``
# does not return these numbers and no sequence of engine 0.3.0 reads them.
#
# Source of every row: Stanisz et al. 2005 (Magn Reson Med 54:507, doi:10.1002/mrm.20605), Table 2, read in the full
# text (tag [F]; an archived copy of the publisher's page, 2026-11-27). In vitro, 37 C, continuous-wave saturation.
#   M0B   "the fraction of magnetization that resides in the semisolid pool", per cent. The paper does not define it as a
#         ratio. It is used here as M0b / M0a (relative to free water, the F of Sled & Pike 2001): the paper's 1.5 T cells
#         for white and grey matter are Sled & Pike's F (0.152, 0.072), and their kf / F (4.6 / 0.152 = 30, 2.4 / 0.072 = 33)
#         are the paper's R. Another paper reprints the liver cell as a fraction of ALL protons (Yarnykh 2015): for 5 to 15
#         per cent the two readings differ by 5 to 15 per cent of the value.
#   R     the exchange rate constant of the model, 1/s; the rate from free water to the bound pool is kf = R x M0B.
#   T2B   of the bound pool, microseconds; its absorption line is super-Lorentzian in tissue and Lorentzian in blood.
#   The 3 T values were measured by the paper. Its 1.5 T values are literature values it lists ("we did not measure MT
#   at 1.5 T"); a cell it leaves empty is taken from 3 T and marked. T1 of the bound pool is not stated in the paper: 1 s,
#   the convention of the field (Sled & Pike 2001: "set it arbitrarily to 1 s-1").
# name: {field: (M0B per cent, R 1/s, T2B us, lineshape, which cells are borrowed from 3 T)}
_STANISZ = "Stanisz 2005 (Magn Reson Med 54:507, Table 2)"
_MT = dict(
    wm={3.0: (13.9, 23.0, 10.0, "superlorentzian", ()), 1.5: (15.2, 30.0, 11.3, "superlorentzian", ())},
    gm={3.0: (5.0, 40.0, 9.1, "superlorentzian", ()), 1.5: (7.2, 33.0, 11.1, "superlorentzian", ())},
    muscle={3.0: (7.4, 66.0, 8.7, "superlorentzian", ()), 1.5: (6.9, 70.0, 8.2, "superlorentzian", ())},
    liver={3.0: (6.9, 51.0, 7.7, "superlorentzian", ()), 1.5: (6.9, 53.0, 7.8, "superlorentzian", ("f_b",))},
    cord={3.0: (12.6, 26.0, 10.5, "superlorentzian", ())},
    blood={3.0: (2.8, 35.0, 280.0, "lorentzian", ()), 1.5: (3.3, 40.0, 340.0, "lorentzian", ())},
    cartilage={3.0: (17.1, 57.0, 8.3, "superlorentzian", ())},         # at 0 degrees to the field (55 degrees: 18.2, 60, 8.3); no class of the tissue table
)
# Classes modelled WITHOUT a bound pool: fluids and fat. "In CSF, as expected, there is no measurable exchange of
# magnetization" (Morrison & Henkelman 1995, abstract); "the MT effect in CSF and fat is negligible (< 3.5 %)" (Weigel
# 2010, full text). Every other class of the table has no MT numbers: declared, not zero.
MT_NONE = ("csf", "bowel_fluid", "scalp_fat", "subcut_fat", "visceral_fat")
MT_T1B_MS = 1000.0


def mt_table(b0_t):
    """Two-pool numbers at 1.5 T or 3 T: name -> dict(f_b: bound pool relative to free water; k_ab_per_s: rate from
    free water to the bound pool (R x f_b); t2b_ms; t1b_ms; lineshape; tag; source; borrowed: the cells taken from
    3 T). The keys are those of ``mrgym.core.operators.spins(pool="mt")``. A tissue the source did not measure at the
    field is left out."""
    if b0_t not in (1.5, 3.0):
        raise ValueError("the two-pool table holds 1.5 T and 3 T")
    out = {}
    for name, rows in _MT.items():
        if b0_t in rows:
            m0b, rate, t2b_us, line, borrowed = rows[b0_t]
            out[name] = dict(f_b=m0b / 100.0, k_ab_per_s=rate * m0b / 100.0, t2b_ms=t2b_us * 1e-3, t1b_ms=MT_T1B_MS, lineshape=line, tag="F",
                             source=_STANISZ + (" at 3 T (measured)" if b0_t == 3.0 else " at 1.5 T (literature values listed there)"), borrowed=borrowed)
    return out


def water_adc_um2_ms(temp_c):
    """Self-diffusion coefficient of water (um^2/ms) at ``temp_c`` degrees Celsius: D = D0 (T / Ts - 1)^gamma with
    D0 = 1.635e-8 m^2/s, Ts = 215.05 K, gamma = 2.063, T in kelvin (the fit of Holz, Heil & Sacco 2000, Phys Chem Chem
    Phys 2:4740, stated for 0 to 100 C). The paper itself was not opened; the three constants were read where another
    paper quotes them (Fardis 2022, Int J Mol Sci 23:14432, PMC9697084, full text). The formula gives 2.299 at 25 C,
    the value attributed to Mills 1973 (J Phys Chem 77:685; not opened), 2.02 at 20 C and 3.04 at 37 C; a measured
    value of cerebrospinal fluid in vivo is 3.02 +- 0.16 (``ADC_SOURCES``)."""
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
