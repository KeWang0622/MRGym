"""Two tables of the hidden world against the sources they cite. The source values are written out here, from the
documents, with where they stand; the tables in the package must equal them. A change of either side fails.

  tissue.FAT_PEAKS      the six-peak fat model of Hernando et al. 2017 (Magn Reson Med 77:1516, PMC4835219), Methods,
                        "Image reconstruction": "a six-peak fat model [...] with frequency shifts relative to the water
                        peak (ppm): -3.90, -3.50, -2.70, -2.04, -0.49, 0.50, respectively, and relative amplitudes:
                        0.087, 0.694, 0.128, 0.004, 0.039, 0.048", after "a common frequency shift of 0.1 ppm" for a
                        phantom at about 22 degrees C. The package holds the model without that shift.
  anatomy.NIST_T1       T1 of the NiCl2 array of the ISMRM / NIST system phantom at 3 T and 20 degrees C: Table S1 of the
                        supporting information of Stupic et al. 2021 (Magn Reson Med 86:1194, PMC8252537), column
                        "NMR measured T1 @ 3.0 T (ms)", spheres Ni-1 to Ni-14.

Engine 0.2.0 and before had 0.693 for the main fat peak and called the list Hamilton 2011, and had 496.0 ms for sphere 5
(the number of the reproducibility challenge's analysis code; the table has 496.7).
"""
import pytest

from mrgym.scanner import anatomy, tissue

# Hernando 2017, as printed (phantom at room temperature: every fat peak 0.1 ppm further from water than in the body)
HERNANDO_2017_PPM = (-3.90, -3.50, -2.70, -2.04, -0.49, 0.50)
HERNANDO_2017_AMPLITUDE = (0.087, 0.694, 0.128, 0.004, 0.039, 0.048)
HERNANDO_2017_ROOM_TEMPERATURE_SHIFT_PPM = 0.1
# Hamilton et al. 2011 (NMR Biomed 24:784, PMC3860876), Table 3: observed ppm and percent of the total fat signal; water 4.7 ppm
HAMILTON_2011_TABLE_3 = ((0.9, 8.8), (1.3, 70.0), (2.1, 12.0), (2.75, 0.6), (4.2, 3.9), (5.3, 4.7))
HAMILTON_2011_WATER_PPM = 4.7
# Stupic 2021, supporting Table S1, "NMR measured T1 @ 3.0 T (ms)", Ni-1 to Ni-14
STUPIC_2021_S1_T1_3T_MS = (1989.0, 1454.0, 984.1, 706.0, 496.7, 351.5, 247.13, 175.3, 125.9, 89.0, 62.7, 44.53, 30.84, 21.719)
STUPIC_2021_S1_T2_3T_MS_NI12 = 31.86


def test_the_fat_model_is_the_six_peak_model_of_its_cited_source():
    ppm, amplitude = zip(*tissue.FAT_PEAKS)
    assert amplitude == HERNANDO_2017_AMPLITUDE and sum(amplitude) == pytest.approx(1.0, abs=1e-12)
    assert ppm == pytest.approx([p + HERNANDO_2017_ROOM_TEMPERATURE_SHIFT_PPM for p in HERNANDO_2017_PPM], abs=1e-12)
    assert tissue.FAT_PPM == ppm[amplitude.index(max(amplitude))] == -3.4            # the main peak: the shift the console prints
    assert "Hernando 2017" in tissue.__doc__ and "NOT the list used here" in tissue.__doc__


def test_the_fat_model_against_the_measured_liver_spectrum_it_is_often_named_after():
    """The model is not Table 3 of Hamilton 2011: stated, with the size of the difference. Amplitudes within 0.8 points;
    five positions within 0.01 ppm; the glycerol peak 0.11 ppm from the table's 4.2 ppm (at 4.31 ppm)."""
    table = sorted((ppm - HAMILTON_2011_WATER_PPM, percent / 100.0) for ppm, percent in HAMILTON_2011_TABLE_3)
    model = sorted(tissue.FAT_PEAKS)
    assert sum(p for _, p in HAMILTON_2011_TABLE_3) == pytest.approx(100.0)
    assert max(abs(a - b) for (_, a), (_, b) in zip(model, table)) == pytest.approx(0.008, abs=1e-9)
    off = [round(abs(m - t), 2) for (m, _), (t, _) in zip(model, table)]
    assert off == [0.0, 0.0, 0.0, 0.01, 0.11, 0.0]
    glycerol = model[4][0] + HAMILTON_2011_WATER_PPM
    assert glycerol == pytest.approx(4.31) and table[4][0] + HAMILTON_2011_WATER_PPM == pytest.approx(4.2)


def test_the_nist_t1_plate_holds_the_values_of_its_cited_table():
    assert anatomy.NIST_T1["t1_ms"] == STUPIC_2021_S1_T1_3T_MS and anatomy.NIST_T1["field_t"] == 3.0
    assert anatomy.NIST_T1["t2_ms_sphere_12"] == STUPIC_2021_S1_T2_3T_MS_NI12
    world = anatomy.nist_t1(__import__("numpy").random.default_rng(0), n=64)
    assert [world["tissues"][f"sphere{i + 1}"]["t1_ms"] for i in range(14)] == list(STUPIC_2021_S1_T1_3T_MS)
