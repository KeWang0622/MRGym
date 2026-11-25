"""Tissue table of the hidden world: PD, T1, T2, T2*, ADC per tissue at 1.5 T and 3 T, with a source tag per row.

How far each number was checked is tagged:
  [A]  the value is stated in the abstract of the cited paper
  [T]  the value is attributed to a table of the cited paper and was not re-checked against the full text for this
       release; treat it as "of the right size", not as a citation-grade number
  [C]  chosen by the authors of this package inside a plausible range; there is no source
A number without a tag in its row is [C]; that includes every diffusion coefficient and most proton densities.
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
  Hamilton 2011      NMR Biomed 24:784, doi:10.1002/nbm.1622        six-peak fat spectrum (amplitudes [T])
"""

FAT_PPM = -3.4              # main methylene peak (1.3 ppm) relative to water (4.7 ppm)  [T]
# six-peak fat spectrum relative to water, ppm and relative amplitude (Hamilton 2011, liver fat)  [T]
FAT_PEAKS = ((-3.80, 0.087), (-3.40, 0.693), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048))

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
