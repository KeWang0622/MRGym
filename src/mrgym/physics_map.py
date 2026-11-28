"""The map: which MRI physics the scanner covers, and which task asks for which piece of it.

    mrgym map                  the grid of subjects x kinds of task, and the counts
    mrgym map --items          every item with its status and its tasks
    mrgym map --assert         exit 1 unless ``validate()`` finds no problem
    mrgym map --json
    mrgym map --write docs/tasks.md

This module is the one authored table. ``docs/tasks.md``, ``mrgym map``, the grouping of ``mrgym list``, the line under
the title of every environment card and the task table of the README are generated from it; nothing is written twice.

Two axes.
    SUBJECTS   what one needs to know. Four, each with a test that decides where an item is filed.
    KINDS      what one does with it: measure, set, trade, request, design.

Three tables.
    ITEMS      the pieces of MRI physics and applied MR physics that textbooks and curricula name, in teaching order.
               Each has an id (subject letter + number; an id is never reused or renumbered, a new item takes the
               next number), a block inside its subject, the items it rests on (``rests_on``), and the status of the engine:
                   "formed"        the scanner serves it: an agent meets it at the console, and a check that exists
                                   shows it (``evidence``)
                   "partial"       the scanner serves part of it; ``reason`` says what is missing
                   "computed"      the package computes it and checks it against a reference (``evidence``), but the
                                   scanner does not serve it: no console sequence, parameter or task reaches it.
                                   ``reason`` says what is computed and what is missing
                   "absent"        neither served nor computed with a check; ``reason`` says what is missing
               An item has the first of these that is true, so one that the scanner serves in part stays "partial"
               when a validated package (``EVIDENCE_PACKAGES``: the design layer, the signal core, the spin systems,
               the RF field model, the receive arrays, the fields in 3D) also computes more of it. A task can isolate
               or use only what the scanner serves ("formed", "partial"). There is no status for "cannot be formed":
               what is missing is engine work, and ``needs`` names it.
               ``needs`` is the piece of engine work an item that is not fully formed waits for, one of ``ENABLERS``
               (a 3D world, a receive array, spin systems, a field model, the core in the console, realistic anatomy,
               measured data), or "" when ordinary work on the present engine is enough.
               ``evidence`` names checks that exist: a row of ``mrgym validate`` ("K6"), a row of the validation
               table of a package, written with its prefix in ``EVIDENCE_PACKAGES`` ("design:E3", "core:N1",
               "spins:S7", "em:S10", "array:P3", "fields3d:F33"), a test ("tests/test_x.py::test_name"), a textbook
               example that asserts its result
               ("examples/textbook/03_t2_spin_echo.py", run by tests/test_textbook_examples.py), a command
               ("mrgym qa acr --fault receive_coil") or the rows of a rung of the comparison with measured data
               ("measured:M6": the stored table docs/physics/measured/results.json holds rows for that rung;
               tests/test_measured.py asserts that the table is current).
               ``outlines`` is the number of the ten teaching and examination outlines (``OUTLINES``) that name the
               item, read at the level of their headings; None where no single line of them corresponds. It ranks what
               is missing (``missing()``); it is not printed on the page. The items themselves follow those outlines
               and the tables of contents of ``TEXTBOOKS``.
    FAMILIES   every scored task family that is registered: its kind, the ONE item it isolates (the subject and the
               block of a family are those of that item), the items it uses besides, and the families whose skill it
               builds on (declared from the cards, not measured). Keyed by the family name without namespace and
               version, so ``NoWrap-v2`` inherits the row of ``NoWrap``.
    PLANNED    the same rows for families that are being built. They are shown as "in construction". When a family is
               registered, move its line from ``PLANNED`` to ``FAMILIES``; ``validate()`` asks for it.

Nothing here can move a reward: the module is not imported by any environment and is not part of the fingerprint of
an id (``mrgym.compat.fingerprint``). A family can be filed elsewhere without a new ``-vN``.

Beside ``mrgym.coverage``: that module follows ONE outline (the technologists' examination) line by line and reads the
``COMPETENCIES`` each family declares in its task module. It stays as it is, as an audit against that outline. The
map stands above it: it is ordered by physics, covers what the outline leaves out (signal, sequences, reconstruction,
the system), and is the page a newcomer reads first.
"""
from __future__ import annotations

import collections
import functools
import json
import os
import re

Subject = collections.namedtuple("Subject", "letter title line test")
Kind = collections.namedtuple("Kind", "number name delivers example")
Item = collections.namedtuple("Item", "id name block rests_on status reason evidence outlines needs")
Family = collections.namedtuple("Family", "kind isolates uses builds_on")
_VALIDATE_ROW = re.compile(r"^[A-Z]+\d+$")                    # a row of ``mrgym validate``
_ROW_ID = r"[A-Z]+[0-9]+[a-z]?[0-9]?"                         # the id of a row of a package: letters, a number, at most one small letter and a digit ("E3a2")
_FIRST_ARGUMENT = rf'_row\("({_ROW_ID})"'                     # ids written as ``_row("E3", ...)``
_ALSO_IN_A_LOOP = rf'(?:_row\(|, )"({_ROW_ID})"[,)]'          # the same, and ids that a loop takes from a tuple: ``..., "M4c"),``
# A third way, for a module whose source does not write every id out: the receive arrays number the rows of a group with
# a counter (``f"P{len(rows) + 1}"``), the fields in 3D build some from a loop variable (``id_ + "b"``). No pattern finds
# those in the source, so the ids are read from the table that the command writes into the page of the package, between
# the two marks below; the test of the package holds that table equal to what the command prints. Without the checkout
# there is no page: such a row then counts as not looked up, like a test or an example.
_ON_THE_PAGE = rf"(?m)^\| ({_ROW_ID}) \|"
_TABLE_MARKS = ("<!-- validation table: begin -->", "<!-- validation table: end -->")
# The packages that have a validation table of their own. An item cites a row of one as "<prefix>:<row id>".
#   prefix: (the module that holds the rows,
#            how the ids of its rows are found: a pattern with one group, applied to the source of that module
#            (``_ON_THE_PAGE``: to the table on the page),
#            the page that shows the table, relative to docs/,
#            the command a reader runs)
# Everything else is built from this table: the pattern of such a row (``package_row``), whether a row exists
# (``package_ids``), the cell of an item and the sentence above the checks on the page (``_evidence_text``,
# ``_row_sources``). A new validated package is one line here, added when its module is merged and not before:
# ``validate()`` refuses a line whose module cannot be imported, whose pattern finds no id or whose page is missing.
EVIDENCE_PACKAGES = {
    "design": ("mrgym.design.validate", _FIRST_ARGUMENT, "physics/design.md", "python -m mrgym.design.validate"),
    "core": ("mrgym.core.validate", _FIRST_ARGUMENT, "physics/core.md", "python -m mrgym.core.validate"),
    "spins": ("mrgym.spins.validate", _ALSO_IN_A_LOOP, "physics/spins.md", "python -m mrgym.spins.validate"),
    "em": ("mrgym.em.validate", _FIRST_ARGUMENT, "physics/em.md", "python -m mrgym.em.validate"),
    "array": ("mrgym.scanner.array_checks", _ON_THE_PAGE, "physics/array.md", "python -m mrgym.scanner.array_checks"),
    "fields3d": ("mrgym.scanner.fields3d_validate", _ON_THE_PAGE, "physics/fields3d.md", "python -m mrgym.scanner.fields3d_validate"),
}
_MEASURED_RUNG = re.compile(r"^measured:(M\d+[a-z]?(?:/M\d+)?)$")   # the rows of a rung of ``python tools/measured/run_all.py`` (docs/physics/measured.md)

SUBJECTS = (
    Subject("A", "Signal and contrast", "what signal a tissue gives", "it can be asked about one voxel"),
    Subject("B", "Spatial encoding and reconstruction", "where the signal lands in the image",
            "it can be asked on an object of uniform contrast"),
    Subject("C", "The system", "what the machine adds: field, RF, gradients, noise, limits",
            "it would be trivial on an ideal scanner"),
    Subject("D", "Image quality and artefacts", "how good the image is, and what went wrong",
            "it is about the whole image: SNR, contrast, resolution, time, artefacts"),
)
KINDS = (
    Kind(1, "measure", "a quantity with its unit", "T1 of a vial; the error of the centre frequency"),
    Kind(2, "set", "the setting that meets one requirement", "the inversion time that nulls a tissue; the field of view without wrap"),
    Kind(3, "trade", "one protocol that meets several requirements at once", "contrast, SNR and pixel size inside a time limit"),
    Kind(4, "request", "a scan for a stated purpose: the method is not named and the system has hidden properties",
         "the fastest T1 map of a stated precision on an unfamiliar scanner"),
    Kind(5, "design", "the sequence itself: an RF pulse, a gradient waveform, the order and timing of its parts",
         "a shorter slice-selective pulse inside the limits on peak B1 and RF power"),
)
STATUSES = ("formed", "partial", "computed", "absent")
STATUS_WORDS = {"formed": "formed", "partial": "in part", "computed": "computed, not served", "absent": "not yet"}
# The pieces of engine work that items wait for (``Item.needs``): name, what it is. A closed list; each unlocks several items.
ENABLERS = (
    ("3D world", "a seeded volume with structure through the slice (a head, a test object) that gives a plane at any position and "
                 "tilt, and the field of its susceptibility in 3D with a shim, exist as libraries; still to do: the slice engine that "
                 "scans the volume (a slice selected in it, several slices, volume encoding)"),
    ("receive array", "several receive channels with sensitivity maps and noise correlation, the multi-channel data, coil combination "
                      "and SENSE exist as a library; still to do: the console serving them (a coil to choose, an acceleration factor, "
                      "what a scan returns), and the methods that work in k-space"),
    ("spin systems", "the density-matrix simulator for coupled spins, with its spectroscopy sequences, metabolite basis and "
                     "quantification, exists as a library; still to do: a voxel that holds metabolites in the scanner world and a "
                     "spectroscopy sequence in the console"),
    ("field model", "Maxwell's equations on the slice for a body and a coil (B1+, B1-, SAR, RF shimming) exist as a library, in two "
                    "dimensions; still to do: the scanner imaging and pricing RF with these fields instead of its declared patterns, "
                    "and the third dimension"),
    ("core in the console", "the signal core over lists of blocks (Bloch ensemble, phase graphs, periodic solve) is behind the console: a "
                            "sequence registry, image formation with an amplitude that changes from line to line, and four sequences (fast "
                            "and multi-echo spin echo, multi-echo gradient echo, diffusion-weighted spin echo) on a console whose card "
                            "lists them, offered to an agent by the tool layer of an environment whose world lists them; still to do: the other sequences "
                            "of the core's library (prepared trains, two pools, real spoilers, what survives a repetition), and the six "
                            "sequences of every console on the core"),
    ("realistic anatomy", "textured digital anatomy instead of classes of one value each"),
    ("measured data", "statistics of public phantom sessions on real scanners are compared with the engine in a table that anyone "
                      "can rebuild; still to do: a scan of the same object with the console's sequences, compared pixel by pixel"),
)
ORDINARY = "ordinary engine work"                            # the heading of the items that wait for none of them
CHECKS_HEADING = "The check behind every item that is formed, formed in part or computed"       # a heading of the page, linked from the list above it
SERVED = ("formed", "partial")                               # what an agent meets at the console: only this can be asked by a task
TEXTBOOKS = (
    "Bernstein, King, Zhou, Handbook of MRI Pulse Sequences",
    "Brown, Cheng, Haacke, Thompson, Venkatesan, Magnetic Resonance Imaging: Physical Principles and Sequence Design, 2nd ed.",
    "Nishimura, Principles of Magnetic Resonance Imaging",
    "McRobbie, Moore, Graves, Prince, MRI from Picture to Proton, 3rd ed.",
    "Liang, Lauterbur, Principles of Magnetic Resonance Imaging: A Signal Processing Perspective",
    "Dale, Brown, Semelka, MRI: Basic Principles and Applications, 5th ed.",
    "Westbrook, Talbot, MRI in Practice, 5th ed.",
)
OUTLINES = (
    "ARRT, MRI examination content specifications (2025)",
    "ABR, qualifying (core) exam physics study guide, section 13 (2023)",
    "AAPM, diagnostic radiology residents physics curriculum, module 13 (2018)",
    "AAPM Report 197, graduate programmes, section 3.2.5",
    "EFOMP, core curriculum for medical physicists in radiology, section 13 (2011)",
    "ASRT / AEIRS / SMRT, magnetic resonance curriculum (2015)",
    "IAEA, Diagnostic Radiology Physics handbook, chapters 14 and 15, with TCS-47",
    "ISMRM, weekend courses MR Physics and Engineering, Artifacts, Image Reconstruction (2022)",
    "ESMRMB, Practical MR Physics (2025)",
    "AAPM Report 100 with the ACR quality-control tests",
)


def package_row(evidence: str):
    """("design", "E3") for "design:E3": the prefix and the row id of a row of a package of ``EVIDENCE_PACKAGES``;
    None for every other check. The one pattern of such a row, built from the keys of the table."""
    m = re.match(rf"^({'|'.join(map(re.escape, EVIDENCE_PACKAGES))}):({_ROW_ID})$", evidence)
    return m.groups() if m else None


def _evidence(parts) -> tuple:
    """Named checks of a row; an argument that holds only validation rows ("K6 K22", "design:E3 design:E4",
    "core:N1 spins:S7") is split."""
    out = []
    for p in parts:
        out += p.split() if all(_VALIDATE_ROW.match(x) or package_row(x) for x in p.split()) else [p]
    return tuple(out)


def _formed(id, name, rests_on, *evidence, n=None):
    return (id, name, rests_on, "formed", "", _evidence(evidence), n, "")


def _partial(id, name, rests_on, missing, *evidence, n=None, needs=""):
    return (id, name, rests_on, "partial", missing, _evidence(evidence), n, needs)


def _computed(id, name, rests_on, missing, *evidence, n=None, needs=""):
    return (id, name, rests_on, "computed", missing, _evidence(evidence), n, needs)


def _absent(id, name, rests_on, missing, n=None, needs=""):
    return (id, name, rests_on, "absent", missing, (), n, needs)


_M = "tests/test_metamorphic.py::"
_X = "examples/textbook/"
# (subject, block, rows). A row: id, name, the items it rests on, then what its status takes. The order is the teaching
# order: inside a subject an item stands after the items of that subject it rests on.
_BLOCKS = (
    ("A", "spins and relaxation", (
        _formed("A1", "Equilibrium magnetisation and proton density", "", _M + "test_r1_proton_density_scales_the_kspace", n=9),
        _formed("A2", "Resonance: the Larmor frequency and its dependence on the field", "A1",
                "K18", _M + "test_r12_doubling_the_field_doubles_the_fat_displacement", n=8),
        _formed("A3", "Excitation: the rotating frame and the flip angle", "A2",
                "R1", _M + "test_r3_transmit_gain_and_nominal_flip_angle_enter_as_a_product", n=9),
        _formed("A4", "T1: recovery of the longitudinal magnetisation", "A3", "S10 S13", _X + "02_inversion_recovery.py", n=9),
        _formed("A5", "T2: decay of the transverse magnetisation, seen with a spin echo", "A3", _X + "03_t2_spin_echo.py", n=9),
        _formed("A6", "T2*: dephasing in an uneven field, refocused by a spin echo and not by a gradient echo", "A5",
                "K9", _X + "04_t2star_field_map.py", n=9),
    )),
    ("A", "sequences and steady states", (
        _formed("A7", "Spin echo: signal against repetition time and echo time", "A4 A5", "S10 R9", n=9),
        _formed("A8", "Image weighting: proton-density, T1, T2 and T2* contrast", "A6 A7", "S10 S1"),
        _formed("A9", "Inversion recovery: the signed signal and its null point (STIR, FLAIR)", "A4 A7", "S11 S13", _X + "02_inversion_recovery.py", n=8),
        _formed("A10", "Spoiled gradient echo: the steady state and the Ernst angle", "A3 A4 A6", "S1 S4", _X + "01_ernst_angle.py", n=9),
        _formed("A11", "Imperfect spoiling: RF phase cycling and the echoes that survive", "A10", "S5 S9"),
        _formed("A12", "Balanced steady-state free precession: signal against T2 / T1 and against frequency", "A6 A10", "S14 S18",
                "core:Q10 core:Q10b", n=6),
        _computed("A13", "Approach to the steady state: dummy cycles and the first lines of a scan", "A10",
                  "the transient is computed by a separate chain of rotations and, in the signal core, by stepping a list of blocks "
                  "from equilibrium or along a prepared train; the scanner serves every line at the steady state",
                  _X + "12_approach_to_steady_state.py", "design:E9", "core:N8", "core:N15", "core:N15b",
                  "tests/test_core_evaluators.py::test_the_periodic_state_is_a_fixed_point", needs="core in the console"),
        _partial("A14", "Echo trains: fast (turbo) spin echo, echo train length, effective echo time", "A7 B8",
                 "a console whose card lists them serves a fast spin echo and a multi-echo spin echo from the signal core: echo-train "
                 "length, echo spacing, effective echo time, T2 blurring, the stimulated echoes of the slice profile and of the transmit "
                 "field, the ratio of the refocusing slice to the excited one, the RF power of the train, and the periodic state of the "
                 "whole repetition with its real crushers: the gratings along z that reach the next excitation. The transverse "
                 "magnetisation at the end of a repetition is taken as dephased (it depends on off-resonance and T2'). One refocusing "
                 "angle per train, at most 32 echoes, fat without J coupling, the carried gratings cut at 16 orders. The family "
                 "T2Echoes has such a console (multi-echo spin echo); on a 2D slice it fits the echoes of a train with one effective "
                 "refocusing angle, not with the slice profile. No shipped environment offers the fast spin echo", "J7 J8 J9 J12 J16 J17 J18 J20 J24 J37 J38",
                 "design:E2 design:E3 design:E4", "core:N1 core:N1b core:N1c core:N2 core:N14 core:N14b core:N14c", "measured:M3b", n=8,
                 needs="core in the console"),
        _partial("A15", "Magnetisation preparation: saturation recovery, T2 preparation, prepared gradient echo", "A9 A10",
                  "a console whose card lists them serves saturation recovery with a gradient echo, and an inversion or a saturation "
                  "in front of the fast spin echo (the null of a fluid, T1 from a series), from the signal core; the preparation is an "
                  "event on the longitudinal magnetisation of the whole object. An inversion-prepared gradient echo exists as a checked "
                  "formula, and the signal core computes a prepared train of gradient echoes and T2 preparation from lists of blocks, "
                  "which no console serves", "S3", "J41 J42 J43 J44 J45 J46 J47", "core:Q12 core:N8 core:N9 core:N9b core:N10",
                  needs="core in the console"),
    )),
    ("A", "selective signal", (
        _formed("A16", "Chemical shift: fat and water resonate 3.4 ppm apart", "A2", "K3", _X + "05_chemical_shift.py", n=8),
        _formed("A17", "In-phase and opposed-phase echo times", "A6 A16", "K18", _X + "11_in_opposed_phase.py", n=8),
        _formed("A18", "Fat suppression by a frequency-selective saturation pulse", "A16 C3",
                "K28 K31 K32 S16 S17", "tests/test_fat_suppress.py::test_physics_the_saturated_band_has_the_width_of_the_hidden_pulse",
                "core:Q16 core:Q16b core:Q16c core:Q16d", n=7),
        _absent("A19", "Spatial saturation bands", "A3 B2", "there is no spatial saturation pulse", n=7, needs="core in the console"),
        _computed("A20", "Magnetisation transfer and chemical exchange", "A4",
                  "the signal core computes two exchanging pools: magnetisation transfer with a saturation pulse on the bound pool, in "
                  "a gradient echo and under continuous irradiation (the direct effect of the pulse on free water is left out), and "
                  "chemical exchange, whose free induction decay equals the closed form of the Bloch-McConnell equations and its two "
                  "limits, in no sequence of its library. The tissue table holds the bound-pool fraction, exchange rate and bound-pool T2 "
                  "of seven tissues from one paper; no console sequence reads them or saturates a bound pool",
                  "core:X6 core:X6b core:X6c core:X6d core:M2 core:M3",
                  "tests/test_served_prepared.py::test_two_pool_numbers_are_additive_sourced_and_read_by_no_existing_sequence", n=4,
                  needs="core in the console"),
    )),
    ("A", "contrast beyond relaxation", (
        _partial("A21", "Diffusion weighting: b-value, apparent diffusion coefficient, tensor", "A7 B1",
                 "a console whose card lists it serves a diffusion-weighted spin echo from the signal core: the b-value of the lobes "
                 "it plays, the echo time they cost, signal x exp(-b D) for the one free, isotropic coefficient of a tissue. No tensor "
                 "(the design layer computes the b-tensor of a gradient program), no kurtosis, no perfusion fraction, no stimulated-echo "
                 "or echo-planar readout; 16 of the 25 coefficients of the tissue table are declared. No shipped environment has such "
                 "a console; the tool layer of an environment offers the sequence when its world lists it", "J31 J32 J33 J35 J36 J15", "design:G5 design:G6 design:E10", "core:X7 core:N11 core:N11c core:N12", n=9,
                 needs="core in the console"),
        _absent("A22", "Flow: inflow, time-of-flight angiography, velocity encoding", "A10 B1 B2",
                "nothing flows: blood is a static tissue", n=8, needs="core in the console"),
        _absent("A23", "Contrast agents: relaxivity and concentration", "A4 A5",
                "no material carries an agent; a dilution series could be built from vials", n=8),
        _absent("A24", "Perfusion: passage of a bolus, arterial spin labelling", "A22 A23",
                "no blood arrives through the plane and no tissue changes during the session; the vessels of the 3D head carry a "
                "direction and a speed as numbers only, and nothing flows until a slice engine uses them", n=8, needs="3D world"),
        _partial("A25", "Susceptibility-weighted imaging: the phase of a gradient echo as contrast", "A6 C2",
                 "phase images show the field of air cavities; no tissue of the scanner has a susceptibility of its own (no veins, no "
                 "iron). The 3D head declares one for venous blood, deep grey matter and a lesion; no scan is formed of it", "K20", needs="3D world"),
        _computed("A26", "Quantitative susceptibility mapping", "A25",
                  "what exists, in the 3D field library: the forward model from a susceptibility distribution to its field, the phase "
                  "of several echoes from that field, and one reference inversion, truncated division in k-space, which on a sphere "
                  "without noise gives the known underestimate. What does not: removal of the background field, masks, unwrapping "
                  "of the phase, any regularised inversion, and a scan that gives the phase of a volume (the scanner knows the field "
                  "in one slice)", "fields3d:F1 fields3d:F43 fields3d:F44", needs="3D world"),
        _absent("A27", "BOLD contrast and functional MRI", "A6 B18",
                "no tissue changes its T2* during the session, and a scan is not repeated as a series"),
        _computed("A28", "Spectroscopy: single voxel and spectroscopic imaging", "A16",
                  "the spin-system package computes what the single-voxel sequences PRESS, STEAM and semi-LASER leave of a "
                  "metabolite, coupled protons included, with slice-selective refocusing pulses and the displacement of the selected "
                  "slice by the chemical shift, and it fits a spectrum with a basis of metabolite spectra; the scanner world has no "
                  "voxel that holds metabolites, the console has no spectroscopy sequence and returns images of water and fat only, "
                  "and spectroscopic imaging is not computed",
                  "spins:S1 spins:S3 spins:S4 spins:S5 spins:S6 spins:Q1 spins:Q4", n=8, needs="spin systems"),
        _computed("A31", "J-coupling: multiplets, the echo of a coupled spin, spectral editing", "A7 A28",
                  "the spin-system package computes the multiplets of weakly and strongly coupled protons, how the echo of a coupled "
                  "spin depends on the echo time, the edited difference of MEGA-PRESS for GABA and lactate and two-dimensional "
                  "J-resolved signals; every tissue of the scanner is a set of uncoupled lines, and no console sequence acquires a "
                  "spectrum or plays an editing pulse",
                  "spins:O5 spins:O5b spins:E6 spins:S1 spins:S7 spins:S7b spins:S7d spins:S8", needs="spin systems"),
        _computed("A32", "Metabolite quantification: fitting a basis set, water referencing", "A28 A31",
                  "the spin-system package simulates a basis of metabolite spectra for a sequence, fits a spectrum as a sum of them "
                  "with one shift, one width and one phase, gives the Cramer-Rao bounds of the fit and turns an amplitude into a "
                  "concentration with the water signal as reference; it has fitted its own simulated spectra only: the scanner "
                  "returns no spectrum, and neither macromolecules nor a baseline are in the basis or in the fit",
                  "spins:M1b spins:M3 spins:Q1 spins:Q2 spins:Q3 spins:Q4", needs="spin systems"),
    )),
    ("A", "quantitative mapping", (
        _formed("A29", "Relaxation-time mapping: T1, T2 and T2* from a series of scans, and what biases them", "A5 A6 A9 A10",
                "R12", _X + "13_slice_profile_vfa.py", _X + "03_t2_spin_echo.py", _X + "04_t2star_field_map.py", n=3),
        _formed("A30", "Fat fraction from in-phase and opposed-phase echoes (Dixon)", "A17", "K18", _X + "11_in_opposed_phase.py", n=8),
    )),
    ("B", "slice selection", (
        _formed("B1", "A gradient makes the frequency depend on position", "A2", "K5 K19"),
        _formed("B2", "Slice selection: thickness = bandwidth of the pulse / gradient", "A3 B1", "C5 C6 R13", "core:C2", n=7),
        _formed("B3", "Slice profile: what a real pulse excites across the slice", "B2", "R2 R4", n=7),
        _absent("B4", "Several slices: slices per repetition, gap, order", "B2 B3",
                "a scan is one slice; the pulses of neighbouring slices are not played. A seeded 3D world that gives a plane at any "
                "position exists as a library; the slice engine that scans it is the missing part", n=7, needs="3D world"),
        _absent("B5", "Slice position, oblique planes, planning in three planes", "B2",
                "the object of a scan is one axial section. The seeded 3D worlds (a head, a test object) give axial, sagittal, "
                "coronal and tilted planes as a library, but no scan is formed of them: the slice engine is the missing part", needs="3D world"),
    )),
    ("B", "Fourier encoding", (
        _formed("B6", "Frequency encoding and receiver bandwidth", "B1", "K3", _M + "test_r6_four_times_the_bandwidth_doubles_the_noise", n=6),
        _formed("B7", "Phase encoding: one line per repetition; scan time = TR x lines x averages", "B6",
                "C2", _M + "test_r16_spin_echo_price_table_by_hand", n=6),
        _formed("B8", "k-space: the image is the Fourier transform of the data", "B6 B7", "K1 K2", n=8),
        _formed("B9", "Field of view: the sampling step of k-space; a larger object folds over", "B8",
                "K6 K22", _X + "06_wrap_around.py", n=10),
        _formed("B10", "Matrix, pixel and voxel: resolution is the extent of k-space", "B8", "K19", _X + "07_gibbs_ringing.py", n=10),
        _formed("B11", "Point spread of a finite matrix: ringing at sharp edges", "B10", "K8", n=6),
        _formed("B12", "Phase-encode and readout axes, and the direction of the readout", "B6 B7",
                "K4", _M + "test_r8_swapping_the_phase_encode_axis_gives_the_same_image",
                _M + "test_r11_reversing_the_readout_reverses_the_fat_displacement", n=6),
        _formed("B13", "Off-resonance during the readout: displacement = frequency / bandwidth", "A16 B6", "K3 K5"),
        _formed("B14", "Decay during the readout: blurring by T2*", "A6 B10", "K9 K10"),
    )),
    ("B", "coverage of k-space", (
        _formed("B15", "Rectangular field of view and phase oversampling", "B9",
                "K7", _M + "test_r7_phase_oversampling_changes_the_price_not_the_image"),
        _formed("B16", "Placing the field of view: centre and in-plane rotation", "B9", "K21 K23"),
        _formed("B17", "Partial Fourier: acquiring part of k-space", "B8 B11", "K15 K29 K30 K37", n=7),
        _formed("B18", "Echo-planar readout: echo spacing, bandwidth along phase, distortion", "B8 B13 B14",
                "K12", _X + "10_epi_distortion.py", n=9),
        _partial("B19", "Segmented acquisition and the order of the lines", "B8",
                 "the fast spin echo of a console that lists it fills k-space in shots of one line per echo, in centric or linear "
                 "order, and the image is formed with the amplitude of each line; every other sequence acquires its lines in one "
                 "order, one per repetition (echo-planar: one shot)", "J12 J16 J17 J39", n=7, needs="core in the console"),
        _partial("B20", "Three-dimensional encoding: a second phase-encode axis, the slab", "B2 B7",
                 "a non-selective 3D scan is priced and its central partition is imaged; no slab profile, no fold along "
                 "the slab", "C7 C8", n=7, needs="3D world"),
        _absent("B21", "Radial and spiral trajectories", "B8", "sampling is Cartesian", n=7),
        _computed("B22", "Parallel imaging: fewer lines, unfolded with the coil sensitivities", "B9 C11",
                  "the array library leaves out phase-encode lines (factors 2 to 4) and unfolds the image with the sensitivity maps "
                  "(SENSE), with the g-factor in closed form and measured with repeated noise, for maps that are given (with maps "
                  "estimated from a prescan the error is reported without a tolerance); the methods that work in k-space (GRAPPA) "
                  "and the acceleration of several slices at once are not built, and the console has one receive channel and no "
                  "acceleration factor",
                  "array:P1 array:P2 array:P3 array:P4 array:P6 array:P7 array:P11 array:P12 array:P16 array:P17 array:P21 array:P22 array:P33 array:P34",
                  "em:C6 em:C6b", n=8, needs="receive array"),
    )),
    ("B", "reconstruction", (
        _formed("B23", "Reconstruction by the inverse Fourier transform: magnitude and phase images", "B8", "K1 K20"),
        _absent("B24", "Options of the reconstruction: zero filling, filters, intensity and distortion correction", "B11 B23",
                "the reconstruction is fixed, with nothing to choose: the unfiltered transform of the acquired matrix, and homodyne "
                "detection where part of k-space was left out"),
        _absent("B25", "Raw data: looking at k-space, spikes", "B8", "only images leave the scanner"),
        _absent("B26", "Reconstruction from incomplete data: compressed sensing and learned methods", "B8 B23",
                "the reconstruction is fixed, and objects made of classes of one value have no texture to judge such a method on",
                n=2, needs="realistic anatomy"),
    )),
    ("C", "static field", (
        _partial("C1", "Field strength: what changes between 1.5 T and 3 T", "A2 A4 A16",
                 "two field strengths, one per session; nothing below 1.5 T or above 3 T",
                 "tests/test_scanner.py::test_field_strength_changes_tissue_and_limits", n=3),
        _formed("C2", "Field inhomogeneity and the field map from two echo times", "A6 B23", "K20", _X + "04_t2star_field_map.py", n=8),
        _formed("C3", "Centre frequency and its adjustment", "A16 C2", "K5 K33", n=8),
        _partial("C4", "Shimming", "C2",
                 "the two in-plane linear terms can be set for a scan; there is no term through the slice, none above first "
                 "order and no automatic shim. The 3D field library fits a shim of first or second order (nine solid harmonics) "
                 "over a volume, with a declared limit on each channel, and returns the fixed first-order shim of the engine on "
                 "the engine's own field; a second-order setting at the console waits for the slice engine on the 3D world",
                 "K34", "fields3d:F33 fields3d:F34 fields3d:F35 fields3d:F36 fields3d:F37 fields3d:F38", n=8, needs="3D world"),
        _partial("C5", "Magnetic susceptibility: the field around air and tissue boundaries", "C2",
                 "the scanner adds the field of a few air spheres in closed form; it does not compute the field from the "
                 "susceptibility of the object. The 3D field library does, with the dipole kernel, for any distribution of "
                 "susceptibility (checked on a sphere, cylinders, a shell and a spheroid, and run on the seeded 3D head); no scan "
                 "uses that field yet",
                 "R11", "tests/test_artefact_clinic.py::test_signal_loss_of_the_qa_gradient_follows_the_through_slice_formula",
                 "fields3d:F1 fields3d:F2 fields3d:F6 fields3d:F7 fields3d:F12 fields3d:F15",
                 "tests/test_fields3d.py::test_the_seeded_volume_world_end_to_end", needs="3D world"),
    )),
    ("C", "RF chain", (
        _formed("C6", "Transmit field: the flip angle actually reached, its pattern, flip-angle mapping", "A3 A10",
                "S2 K35 K36", "mrgym qa acr --fault b1_nonuniformity",
                "tests/test_engine_0_3.py::test_tx_scale_brings_a_nominal_90_degree_pulse_to_90_degrees", n=7),
        _formed("C7", "RF pulses: duration, bandwidth, peak amplitude and energy", "A3 B2", "C4 C5"),
        _partial("C8", "Adiabatic pulses: an inversion that tolerates the transmit field", "A9 C6",
                 "the console's inversion is one efficiency number; the hyperbolic-secant pulse and its threshold are computed in "
                 "the design layer and played by no sequence", "S11", "design:P6 design:P7", needs="core in the console"),
        _partial("C9", "RF pulse design: shaping a pulse to a profile", "B3 C7",
                 "two tasks ask for a slice-selective excitation and the scan rfscan plays a stored design through a modelled "
                 "transmit chain (scale, amplifier compression, gradient delay, off-resonance); refocusing, inversion and "
                 "variable-rate designs are generated and simulated only and nothing relaxes during a pulse. The first task "
                 "compares the duration with a reference search; the second with a bound that is proven for hard sub-pulses "
                 "under one constant gradient, tight where the gradient limit decides and loose where the RF power limit does",
                 "design:P3 design:P4 design:P9 design:F1 design:F4 design:B1 design:B3 design:B4",
                 "tests/test_rf_pulse.py::test_the_rotation_of_the_verifier_is_the_bloch_equation",
                 "tests/test_rf_pulse.py::test_the_stripe_of_rfscan_is_the_sine_of_the_flip_angle_and_the_log_replays",
                 "tests/test_rf_pulse_v1.py::test_the_rotation_of_the_verifier_is_a_train_of_hard_pulses",
                 "tests/test_design_bound.py::test_the_bound_is_reached_by_the_minimum_phase_pulse_and_not_beaten", n=1),
        _formed("C10", "Receive coil: sensitivity profile and shading", "B23", "mrgym qa acr --fault receive_coil", n=9),
        _computed("C11", "Receive arrays: coil elements, their combination, the choice of coil", "C10",
                  "the array library gives a world several receive elements, each with a sensitivity map (loops, in the quasi-static "
                  "limit) and a noise covariance between them, forms the multi-channel data and combines the channels (root sum of "
                  "squares, the optimal combination, an adaptive one) with the SNR of each; the field model computes receive maps "
                  "and their noise matrix from Maxwell's equations, and the two are not joined. The console has one receive "
                  "channel: there is no coil to choose and no coil image is returned",
                  "array:F1 array:F4 array:F5 array:N1 array:N3 array:N4 array:N5 array:C6 array:C7 array:C8 array:C9 array:C10 array:C11",
                  "em:C4 em:C5", n=9, needs="receive array"),
        _formed("C12", "Receiver noise: its level and its dependence on bandwidth", "B6",
                "K15", _M + "test_r6_four_times_the_bandwidth_doubles_the_noise"),
        _absent("C13", "Receiver gain and dynamic range", "C12", "samples are floating-point numbers: no converter, no clipping"),
        _formed("C14", "RF interference: a narrow-band line across the image", "B6 C12", "K14", "mrgym qa acr --fault rf_interference", n=7),
        _computed("C34", "RF fields in the body: B1+ and B1- from Maxwell's equations, the wavelength in tissue, dielectric effects", "C6 C10",
                  "the field model solves Maxwell's equations on the slice for the tissues of a world and the currents of a coil: "
                  "B1+ and B1-, the wavelength in tissue, the central brightening at 7 T and its near absence at 1.5 T. It has two "
                  "dimensions, with fields that do not change along z (no end rings, no end of the head), and the scanner still "
                  "images with declared transmit and receive patterns",
                  "em:A4 em:A8 em:A8b em:A11 em:A11b em:A11c em:S3 em:S3b em:S10 em:S10c em:S11 em:C1", needs="field model"),
        _computed("C36", "Electrical properties from the transmit field: mapping conductivity and permittivity", "C34",
                  "the field model recovers conductivity and permittivity from a complex B1+ map by the Helmholtz relation, inside "
                  "a uniform region and without noise, and shows where that fails: at a boundary, and with the transceive-phase "
                  "assumption for a coil of few rungs or a linear coil. The transmit pattern of the scanner is declared and does "
                  "not come from the conductivity and permittivity of its tissues, so a scan holds nothing to recover; neither "
                  "noise nor the third dimension is treated",
                  "em:E1 em:E1b em:E2 em:E3 em:E3a em:E3a2 em:E3b em:E4", needs="field model"),
    )),
    ("C", "gradient chain", (
        _formed("C15", "Gradient amplitude and slew rate: the shortest TE and TR, the thinnest slice", "B2 B6 B7",
                "C6", "tests/test_scanner.py::test_refusals_are_typed", n=9),
        _formed("C16", "Gradient calibration: the scale of the image", "B10",
                "tests/test_acr_phantom.py::test_a_gradient_calibration_error_scales_the_image_along_its_axis_and_is_absent_by_default",
                "mrgym qa acr --fault gradient_x", n=4),
        _partial("C17", "Gradient non-linearity: distortion away from the isocentre", "C16",
                 "positions are scaled by one declared term, 1 - g (r / 250 mm)^2, the same along both axes, and the brightness "
                 "does not change with it; there is no expansion of a real coil, nothing along z and no correction of the distortion",
                 "tests/test_gradient_cal.py::test_under_non_linear_gradients_the_edge_of_the_disc_is_not_at_the_scale_of_the_isocentre"),
        _formed("C18", "Gradient timing: a delay between gradient and sampling", "B18", "K13", n=6),
        _absent("C19", "Eddy currents and concomitant fields", "C15", "gradients are played as ideal trapezoids", n=6),
        _partial("C20", "Spoiler and crusher gradients: the moment and what survives it", "A11 B1",
                 "the spoiler moment of a gradient echo is a parameter and acts on the signal; the crushers of a spin echo are "
                 "ideal: what they dephase is gone, so there is no stimulated echo. The echo trains and the diffusion-weighted spin "
                 "echo of a console that lists them play crusher lobes with their duration, area and diffusion weighting, and serve "
                 "the gratings that survive them, inside a repetition and from one repetition to the next (a sixth of the signal of a "
                 "fluid at a TR of 0.5 s); what is transverse at the end of a repetition is taken as dephased. A spoiler of a "
                 "gradient echo played as a lobe with its own duration is computed by the signal core and served by no sequence",
                 "S6 S8", "J8 J9 J20 J23 J24", "core:Q7 core:N1 core:N12 core:N13 core:N14 core:N14b core:N5 core:N6 core:N6c",
                 needs="core in the console"),
        _computed("C21", "Gradient programs: lobes, moments and their nulling, the shortest waveform inside the limits", "C15 C20",
                  "the design layer computes moments, k-space and limit checks of a gradient program and the shortest lobe for "
                  "given moments; the console builds its own trapezoids and no waveform can be entered",
                  "design:G1 design:G3 design:G4 design:G7 design:G9", needs="core in the console"),
    )),
    ("C", "operating limits", (
        _partial("C22", "RF power: what raises B1+rms, and its limit", "A3 C7",
                 "the console prices RF as B1+rms against a limit, from the nominal flip angle times the transmit factor of the "
                 "scan (`tx_scale`); it has no figure in W/kg. The field "
                 "model computes the power a body absorbs for a given B1+ and how it grows with the square of the frequency; the "
                 "console does not use it", "C1 C5 C9", "em:A10 em:S12", n=9, needs="field model"),
        _computed("C23", "SAR in the body: whole-body and local heating", "C22",
                  "the field model computes the power a body absorbs on the slice from the electric field of the coil: SAR at a "
                  "point, averaged over a square of 21.5 mm (its stand-in for the 10 g average: a definition of the package, not "
                  "that of a standard) and averaged over the slice. It has two dimensions: no end rings, no neck or shoulders, no "
                  "whole-body figure; no temperature is computed and no number of it is a safety figure. The console still prices "
                  "RF as B1+rms only",
                  "em:A10 em:A10b em:A10c em:S3e em:S7 em:S12 em:S13 em:S13b em:S13c em:H1", n=9, needs="field model"),
        _computed("C35", "RF shimming and parallel transmission: channel weights for a uniform B1+ inside limits on power and SAR", "C23 C34",
                  "the field model computes static RF shimming: the weights of the channels of a transmit array, phase only or "
                  "magnitude and phase, that make B1+ uniform over a region under limits on power and on local SAR, and the "
                  "trade-off between uniformity and the SAR limit. Pulses that differ between the channels (spatially selective "
                  "parallel transmission) are not built, the fields are two-dimensional, and the scanner has one transmit channel "
                  "with a declared pattern",
                  "em:C3 em:H1 em:H2 em:H3 em:H3c em:H3d em:H3e em:H4 em:H5b em:H5c", needs="field model"),
        _computed("C24", "Nerve stimulation: the limit on dB/dt", "C15",
                  "the design layer computes a stimulation figure of a gradient program from a declared model; the console plays "
                  "gradients at one fixed slew rate and has no such limit", "design:G8", n=8, needs="core in the console"),
        _absent("C25", "Acoustic noise of the gradients", "C15", "there is no model of sound", n=8),
        _absent("C26", "Implants and devices: scanning inside the conditions of a label", "C22 C24",
                "no object holds a device and no task states a label. The field model stops short of the heating at a device: its "
                "electric field has one component, along z, on a grid of millimetres", needs="field model"),
        _absent("C27", "The static field around the magnet: fringe field, forces on objects, zones, screening", "A2",
                "the field outside the bore and the force on an object in it are not modelled: this is the field of the magnet "
                "itself, which neither the RF field model nor the susceptibility field in 3D holds", n=8),
    )),
    ("C", "quality control", (
        _formed("C28", "Phantom test: geometric accuracy", "B10 C16",
                "tests/test_acr_phantom.py::test_the_phantom_measures_190_and_148_mm_on_a_perfect_system", "mrgym qa acr --fault gradient_x", n=4),
        _formed("C29", "Phantom tests of the image: uniformity, ghosting, resolution, low contrast", "C6 C10 D1",
                "tests/test_qa_acr.py::test_uniformity_of_a_linear_shading_is_the_analytic_value",
                "tests/test_qa_acr.py::test_ghosting_ratio_of_a_known_background",
                "tests/test_qa_acr.py::test_a_healthy_system_passes_every_simulated_test_at_1_5_t",
                "tests/test_acr_phantom.py::test_layout_of_the_resolution_insert_and_of_the_low_contrast_section", n=4),
        _absent("C30", "Phantom tests: slice thickness and slice position", "B3",
                "the test objects of the scanner are flat. A 3D test object with ramps and crossed wedges through the slice exists as "
                "a library (drawn angles, not the dimensions of the ACR phantom); no scan is formed of it: the slice engine is the "
                "missing part", n=4, needs="3D world"),
        _formed("C31", "From a failed test to its cause", "C28 C29",
                "tests/test_qa_acr.py::test_one_fault_fails_the_test_the_guidance_names_and_no_other", n=7),
        _absent("C32", "Stability over time: field drift, repeated scans", "C3 C12",
                "the system does not change during a session; a repeated scan differs by its noise only"),
        _partial("C33", "Agreement with measurement: a phantom on a real scanner against the same phantom here", "C28 C29",
                 "statistics of images, region signals and fitted values of public phantom sessions are compared with the engine's model of the same protocol "
                 "(for an echo train with the design layer and with the multi-echo spin echo of the console), and the differences found are listed; "
                 "no image of the console's sequences has been compared pixel by pixel with a scan of the same object",
                 *(f"measured:{rung}" for rung in ("M0", "M1", "M2", "M2b", "M3b", "M4/M8", "M5", "M6", "M7", "M9")), needs="measured data"),
    )),
    ("D", "quality", (
        _formed("D1", "Signal-to-noise ratio: voxel volume x square root of the sampling time", "B7 B10 C12", "K15 K17", n=10),
        _formed("D2", "Measuring noise and SNR: the background of a magnitude image, the noise floor", "B23 D1",
                "K16", _X + "08_noise.py", "tests/test_snr_spec.py::test_the_background_of_a_magnitude_image_gives_the_noise"),
        _formed("D3", "Contrast and contrast-to-noise between two tissues", "A8 D1",
                "K15", "tests/test_protocol_to_spec.py::test_the_signal_model_of_the_oracle_matches_noise_free_scans", n=5),
        _formed("D4", "Spatial resolution: what limits the smallest detail", "B10 B11 B14", "K8 K10"),
        _formed("D5", "SNR, resolution and scan time: what each costs the others", "B7 D1 D4",
                "K17", "tests/test_snr_spec.py::test_no_neighbouring_protocol_beats_the_oracle_on_the_engine", n=7),
        _formed("D6", "Partial volume: a voxel that holds two tissues", "B10",
                "K2", "tests/test_artefact_clinic.py::test_truth_image_has_the_scale_of_a_scan_and_block_average_keeps_a_flat_image"),
    )),
    ("D", "artefacts", (
        _formed("D7", "Wrap-around (aliasing)", "B9", "K6 K7", n=7),
        _formed("D8", "Truncation ringing (Gibbs)", "B11", "K8", _X + "07_gibbs_ringing.py", n=6),
        _formed("D9", "Displacement of fat by its chemical shift", "A16 B13", "K3", _X + "05_chemical_shift.py"),
        _formed("D10", "Cancellation at water-fat boundaries at opposed-phase echo times", "A17", "K18"),
        _formed("D11", "Distortion by off-resonance: displacement, stretching, pile-up", "B13 B18 C5", "K5 K12", _X + "10_epi_distortion.py", n=8),
        _formed("D12", "Signal loss by a field gradient through the slice", "A6 B3 C5", "R11"),
        _formed("D13", "Shading by the transmit and receive fields", "C6 C10",
                "mrgym qa acr --fault b1_nonuniformity", "mrgym qa acr --fault receive_coil"),
        _formed("D14", "Dark bands of balanced steady-state images", "A12 C2", "K11", _X + "09_bssfp_bands.py"),
        _formed("D15", "Interference line (zipper)", "C14", "K14"),
        _partial("D16", "Ghosts of periodic motion along the phase-encode axis", "B7 B12",
                 "one tissue class pulses or shifts periodically; no irregular motion, and nothing moves through the slice",
                 "K25 K26", n=8, needs="3D world"),
        _formed("D17", "Half-field-of-view ghost of echo-planar images", "B18 C18", "K13"),
        _absent("D18", "Gating, triggering and breath-holding against motion", "D16", "the console has no trigger: a scan is not synchronised with the motion"),
        _absent("D19", "Cross-talk between neighbouring slices", "B4",
                "a scan is one slice. The 3D world exists as a library; neighbouring slices excited in one repetition take the slice "
                "engine", n=3, needs="3D world"),
        _absent("D20", "Metal: distortion of the slice itself, and the sequences against it", "D11 D12",
                "the slice is flat: how it bends near metal takes the third dimension. The 3D field library computes the field of a "
                "susceptibility distribution to first order (enough for titanium, not for steel); no world holds metal and no slice "
                "is selected in such a field", needs="3D world"),
        _formed("D21", "Telling artefacts from the object, and choosing a remedy by its price", "D7 D8 D9 D10 D12 D16",
                "tests/test_artefact_clinic.py::test_twins_switch_one_mechanism_each_on_a_hand_made_object",
                "tests/test_artefact_clinic.py::test_each_wrong_deliverable_fails_exactly_its_own_check"),
    )),
)
ITEMS = tuple(Item(i, name, block, tuple(rests_on.split()), status, reason, ev, n, needs)
              for _, block, rows in _BLOCKS for i, name, rests_on, status, reason, ev, n, needs in rows)
BLOCKS = {s.letter: tuple(block for letter, block, _ in _BLOCKS if letter == s.letter) for s in SUBJECTS}


def _f(kind, isolates, uses="", builds_on=""):
    return Family(kind, isolates, tuple(uses.split()), tuple(builds_on.split()))


# name: kind, the item it isolates, the items it uses besides, the families it builds on. One line per family.
FAMILIES = {
    "T1Transmit": _f(1, "A29", "A10 A11 B3 B20 C6", "T1Vial TransmitAdjust"),
    "T2Decay": _f(1, "A5", "A4 A7"),
    "FatFraction": _f(1, "A30", "A6 A16 A17"),
    "NullTissue": _f(2, "A9", "A4 C6 C8"),
    "FatSuppress": _f(2, "A18", "A9 A16 C2 C3 C6 C8", "NullTissue FrequencyAdjust"),
    "ProtocolToSpec": _f(3, "A8", "A5 A9 B7 D1 D3", "NullTissue SNRSpec"),
    "NoWrap": _f(2, "B9", "B7 B12 B15 B16"),
    "PhantomQA": _f(1, "C31", "B13 C2 C6 C10 C16 C28 C29 D16", "GradientCal"),
    "FrequencyAdjust": _f(1, "C3", "A16 C2"),
    "TransmitAdjust": _f(1, "C6", "A10 B3 B20"),
    "SNRSpec": _f(2, "D5", "A4 A5 B6 B7 C12 D1 D2", "NoiseSNR"),
    "BandFree": _f(2, "D14", "A12 A16 B3 B6 C2 C3", "FrequencyAdjust"),
    "NoiseSNR": _f(1, "C12", "C10 D1 D2"),
    "GradientCal": _f(1, "C16", "B12 B13 C2 C17"),
    "ArtefactClinic": _f(3, "D21", "B15 D1 D6 D7 D8 D9 D10 D12 D16", "NoWrap ShiftFree GhostAside"),
    "ShiftFree": _f(3, "B13", "A16 B6 B12 C2 D1"),
    "GhostAside": _f(2, "D16", "B7 B12"),
    "Resolve": _f(2, "B10", "B7 B11 B15 B17 D1", "NoWrap"),
    "FoldAway": _f(2, "B15", "B9 B11 B12 B16", "NoWrap"),
    "T1Vial": _f(1, "A4", "A9"),
    "ErnstFlip": _f(2, "A10", "A4 B3 C6"),
    "EPISetup": _f(3, "D11", "B18 C2 C18 D1 D17", "FrequencyAdjust"),
    "T2Echoes": _f(1, "A29", "A5 A14 C6 C20", "T2Decay"),
    "RFPulse": _f(5, "C9", "B3 C6 C7", "TransmitAdjust"),
}
# Families that are being built: the cell each is meant for. A line moves to FAMILIES when its family is registered.
PLANNED = {
    "Conspicuity": _f(3, "A8", "A7 A9 A10 B7 D1 D3", "T1Vial T2Decay"),
}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # the checkout, when there is one


# ------------------------------------------------------------------ look-ups
def items() -> dict:
    return {i.id: i for i in ITEMS}


def subject(letter: str) -> Subject:
    return next(s for s in SUBJECTS if s.letter == letter)


def kind(number: int) -> Kind:
    return next(k for k in KINDS if k.number == number)


def registered() -> dict:
    """{family name: [environment ids]} of the scored ids of this package (a sandbox has no row; a family that another
    package registers through the entry-point group is not asked for one and is listed apart by ``groups()``)."""
    import mrgym
    out = {}
    for env_id in mrgym.list_envs():
        s = mrgym.spec(env_id)
        if s.family.startswith("mrgym.envs.") and s.kind != "sandbox":
            out.setdefault(s.name, []).append(env_id)
    return {name: sorted(ids, key=lambda i: mrgym.spec(i).version) for name, ids in out.items()}       # the newest id last


def family_of(env_id: str):
    """(name, row) of the family of an environment id, or None when it has no row (a sandbox, a private family)."""
    from .registry import ENV_ID_RE
    m = ENV_ID_RE.match(env_id)
    return (m["name"], FAMILIES[m["name"]]) if m and m["name"] in FAMILIES else None


def cell(row: Family) -> tuple:
    """(subject letter, block, kind number) of a family: subject and block are those of the item it isolates."""
    i = items()[row.isolates]
    return i.id[0], i.block, row.kind


def tasks_of(item_id: str, planned: bool = False) -> tuple:
    """(names of the families that isolate the item, names that use it besides)."""
    rows = {**FAMILIES, **PLANNED} if planned else FAMILIES
    return [n for n, f in rows.items() if f.isolates == item_id], [n for n, f in rows.items() if item_id in f.uses]


# ------------------------------------------------------------------ validation
def evidence_exists(evidence: str, root: str | None = None):
    """Whether a named check exists: True, False, or None when it cannot be looked up here (a test or an example,
    in an installed package without the checkout)."""
    if _VALIDATE_ROW.match(evidence):
        from . import coverage
        return evidence in coverage.validate_ids()
    row = package_row(evidence)
    if row:
        ids = package_ids(row[0], root)
        return None if ids is None else row[1] in ids
    if evidence.startswith("mrgym qa acr"):
        from . import qa
        return "--fault " not in evidence or evidence.split("--fault ")[1] in qa.FAULTS
    root = root or ROOT
    if _MEASURED_RUNG.match(evidence):
        if not os.path.isdir(os.path.join(root, "tests")):
            return None
        try:
            with open(os.path.join(root, "docs", "physics", "measured", "results.json"), encoding="utf-8") as f:
                rungs = {r["rung"].replace(" ", ""): r["rows"] for r in json.load(f)["rungs"]}
        except (OSError, ValueError, KeyError):
            return False
        return bool(rungs.get(_MEASURED_RUNG.match(evidence).group(1)))
    path, _, name = evidence.partition("::")
    if not os.path.isdir(os.path.join(root, "tests")):
        return None
    try:
        with open(os.path.join(root, path), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    if path.startswith("tests/"):
        return bool(name) and re.search(rf"^def {re.escape(name)}\(", text, re.M) is not None
    return path.startswith("examples/textbook/") and ("check(" in text or "assert " in text)


@functools.cache
def _ids_in(module: str, pattern: str) -> frozenset:
    import importlib
    import inspect
    try:
        return frozenset(re.findall(pattern, inspect.getsource(importlib.import_module(module))))
    except ImportError:
        return frozenset()


@functools.cache
def _ids_on(page: str) -> frozenset:
    try:
        with open(page, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return frozenset()
    begin, end = _TABLE_MARKS
    return frozenset(re.findall(_ON_THE_PAGE, text.split(begin)[1].split(end)[0])) if begin in text and end in text else frozenset()


def package_ids(prefix: str, root: str | None = None):
    """Ids of the rows of a package of ``EVIDENCE_PACKAGES``: those written out in the source of its module, or, for a
    package read ``_ON_THE_PAGE``, those of the table on its page (running the rows takes longer; the test of each
    package, as ``tests/test_core_validate.py``, asserts that every row passes and that its page under docs/physics/
    shows them). Empty when the module cannot be imported or the page has no table; None when the page cannot be looked
    up here (an installed package without the checkout)."""
    module, pattern, page = EVIDENCE_PACKAGES[prefix][:3]
    if pattern is not _ON_THE_PAGE:
        return _ids_in(module, pattern)
    docs = os.path.join(root or ROOT, "docs")
    return _ids_on(os.path.join(docs, page)) if os.path.isdir(docs) else None


def package_problems(root: str | None = None) -> list:
    """Problems of ``EVIDENCE_PACKAGES`` (empty: none): a prefix that is not a plain word, a module that cannot be
    imported, a pattern that finds no row id (in the source, or in the table of the page), a page that is not under
    docs/ (looked up in a checkout only)."""
    import importlib
    bad, docs = [], os.path.join(root or ROOT, "docs")
    for prefix, (module, pattern, page, command) in EVIDENCE_PACKAGES.items():
        if not re.match(r"^[a-z][a-z0-9]*$", prefix):
            bad.append(f"EVIDENCE_PACKAGES: the prefix {prefix!r} is not a word of small letters and digits")
        try:
            importlib.import_module(module)
        except ImportError:
            bad.append(f"EVIDENCE_PACKAGES: {prefix!r} names the module {module}, which cannot be imported: list a package when it is merged")
            continue
        if package_ids(prefix, root) == frozenset():
            where = f"the table of docs/{page}" if pattern is _ON_THE_PAGE else f"the source of {module}"
            bad.append(f"EVIDENCE_PACKAGES: the pattern of {prefix!r} finds no row id in {where}")
        if os.path.isdir(docs) and not os.path.isfile(os.path.join(docs, page)):
            bad.append(f"EVIDENCE_PACKAGES: the page docs/{page} of {prefix!r} does not exist")
        if not command:
            bad.append(f"EVIDENCE_PACKAGES: {prefix!r} names no command")
    return bad


def _cycle(edges: dict):
    """A node on a cycle of {node: prerequisites}, or None."""
    state = {}

    def visit(n):
        if state.get(n) == 1:
            return n
        if state.get(n) == 2:
            return None
        state[n] = 1
        for m in edges.get(n, ()):
            hit = visit(m)
            if hit:
                return hit
        state[n] = 2
        return None
    for n in edges:
        hit = visit(n)
        if hit:
            return hit
    return None


def validate(root: str | None = None) -> list:
    """Problems of the map (empty: none). Checked: the table of validated packages (``package_problems``); ids, blocks,
    statuses and enabling pieces of the items; that prerequisites exist,
    are acyclic and, inside a subject, stand earlier in the list; that every named check exists; that every scored
    registered family has a row and every row a registered family; that a family isolates and uses served items only;
    that prerequisite families exist and are acyclic; that a family of kind 3 or more uses items of two blocks."""
    bad = package_problems(root)
    letters = [s.letter for s in SUBJECTS]
    by_id, order = items(), {i.id: k for k, i in enumerate(ITEMS)}
    if len(by_id) != len(ITEMS):
        bad.append("item ids are not unique: " + ", ".join(sorted(i for i, c in collections.Counter(x.id for x in ITEMS).items() if c > 1)))
    for i in ITEMS:
        if not re.match(rf"^[{''.join(letters)}][1-9]\d*$", i.id) or i.block not in BLOCKS.get(i.id[0], ()):
            bad.append(f"{i.id}: an id is a subject letter and a number, in a block of that subject")
        if i.status not in STATUSES:
            bad.append(f"{i.id}: status {i.status!r} is not one of {STATUSES}")
        if (i.status == "formed") == bool(i.reason):
            bad.append(f"{i.id}: a reason is given exactly when an item is not fully formed")
        if i.needs and (i.status == "formed" or i.needs not in [name for name, _ in ENABLERS]):
            bad.append(f"{i.id}: needs is one of {[name for name, _ in ENABLERS]}, or empty, and empty for a formed item")
        if (i.status in ("formed", "partial", "computed")) != bool(i.evidence):
            bad.append(f"{i.id}: evidence is named exactly when an item is formed, partial or computed")
        for n in i.rests_on:
            if n not in by_id:
                bad.append(f"{i.id} rests on {n}, which is not an item")
            elif n[0] == i.id[0] and order[n] > order[i.id]:
                bad.append(f"{i.id} stands before {n}, on which it rests: the list of a subject is in teaching order")
        bad += [f"{i.id}: the check {e!r} does not exist" for e in i.evidence if evidence_exists(e, root) is False]
    hit = _cycle({i.id: [n for n in i.rests_on if n in by_id] for i in ITEMS})
    if hit:
        bad.append(f"the prerequisites of the items form a cycle through {hit}")
    both = set(FAMILIES) & set(PLANNED)
    if both:
        bad.append("in FAMILIES and in PLANNED: " + ", ".join(sorted(both)))
    rows = {**PLANNED, **FAMILIES}
    for name, f in rows.items():
        if f.kind not in [k.number for k in KINDS]:
            bad.append(f"{name}: kind {f.kind!r} is not one of {[k.number for k in KINDS]}")
        for i in (f.isolates, *f.uses):
            if i not in by_id:
                bad.append(f"{name} names {i}, which is not an item")
            elif by_id[i].status not in SERVED:
                bad.append(f"{name} names {i}, which the scanner does not serve ({by_id[i].status})")
        if f.isolates in f.uses:
            bad.append(f"{name}: the item it isolates ({f.isolates}) is not listed again among the items it uses")
        bad += [f"{name} builds on {n}, which has no row" for n in f.builds_on if n not in rows]
        blocks = {(i[0], by_id[i].block) for i in f.uses if i in by_id}
        if f.kind >= 3 and len(blocks) < 2:
            bad.append(f"{name} is of kind {f.kind} and uses items of {len(blocks)} block(s); a task that brings knowledge together uses at least two")
    hit = _cycle({n: [m for m in f.builds_on if m in rows] for n, f in rows.items()})
    if hit:
        bad.append(f"the prerequisite families form a cycle through {hit}")
    reg = registered()
    bad += [f"{', '.join(ids)}: the family {name} is registered and has no row in FAMILIES"
            + (": move its line from PLANNED" if name in PLANNED else "") for name, ids in reg.items() if name not in FAMILIES]
    bad += [f"{name} has a row in FAMILIES and no registered environment: its line belongs in PLANNED" for name in FAMILIES if name not in reg]
    return bad


# ------------------------------------------------------------------ counts and the table
def coverage(planned: bool = False) -> dict:
    """Counts per subject and per block: items listed, by status, isolated by a task, used by a task (isolated counts
    as used). ``planned``: count the families in construction as if they were built."""
    rows = {**FAMILIES, **PLANNED} if planned else FAMILIES
    isolated = {f.isolates for f in rows.values()}
    used = isolated | {i for f in rows.values() for i in f.uses}

    def count(group):
        return dict(listed=len(group), formed=sum(i.status == "formed" for i in group), partial=sum(i.status == "partial" for i in group),
                    computed=sum(i.status == "computed" for i in group), absent=sum(i.status == "absent" for i in group),
                    isolated=sum(i.id in isolated for i in group), used=sum(i.id in used for i in group))
    out = dict(subjects={}, total=count(ITEMS), families=len(rows))
    for s in SUBJECTS:
        mine = [i for i in ITEMS if i.id[0] == s.letter]
        out["subjects"][s.letter] = dict(count(mine), blocks={b: count([i for i in mine if i.block == b]) for b in BLOCKS[s.letter]})
    return out


def dependents() -> dict:
    """{item id: number of items that rest on it, directly or through other items}."""
    direct = {i.id: set() for i in ITEMS}
    for i in ITEMS:
        for n in i.rests_on:
            direct[n].add(i.id)
    out = {}
    for start in direct:
        seen, stack = set(), [start]
        while stack:
            for nxt in direct[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out[start] = len(seen)
    return out


def not_isolated(planned: bool = False) -> list:
    """Formed items that no task isolates, the ones most other items rest on first."""
    rows = {**FAMILIES, **PLANNED} if planned else FAMILIES
    taken, dep = {f.isolates for f in rows.values()}, dependents()
    return sorted((i for i in ITEMS if i.status == "formed" and i.id not in taken), key=lambda i: -dep[i.id])


def missing() -> list:
    """Items the scanner does not serve yet (computed or absent), those named by most outlines first (items without a
    count last)."""
    return sorted((i for i in ITEMS if i.status in ("computed", "absent")), key=lambda i: -(i.outlines or 0))


def waiting() -> dict:
    """{enabling piece: [ids of the items that are not formed and wait for it]}, in the order of ``ENABLERS``, then ""
    for the items that need only ordinary work on the present engine. Not formed: "computed" or "absent"."""
    out = {name: [] for name, _ in ENABLERS}
    out[""] = []
    for i in ITEMS:
        if i.status in ("computed", "absent"):
            out[i.needs].append(i.id)
    return out


def table(root: str | None = None) -> dict:
    """The map as plain data: subjects, kinds, items with their tasks, families with their registered ids, the
    families in construction, the counts and the problems of ``validate()``. ``unchecked``: the named tests and
    examples that could not be looked up (an installed package without the checkout)."""
    import mrgym
    reg = registered()

    def fam(name, f, built):
        letter, block, _ = cell(f)
        ids = reg.get(name, []) if built else []
        return dict(name=name, built=built, ids=ids, subject=letter, block=block, kind=f.kind, kind_name=kind(f.kind).name, isolates=f.isolates,
                    uses=list(f.uses), builds_on=list(f.builds_on), state=[mrgym.spec(i).kind for i in ids])
    rows = []
    for i in ITEMS:
        iso, use = tasks_of(i.id)
        iso_p, use_p = tasks_of(i.id, planned=True)
        rows.append(dict(i._asdict(), rests_on=list(i.rests_on), evidence=list(i.evidence), subject=i.id[0], isolated_by=iso, used_by=use,
                         planned_isolated_by=[n for n in iso_p if n not in iso], planned_used_by=[n for n in use_p if n not in use]))
    problems = validate(root)
    unchecked = sorted({e for i in ITEMS for e in i.evidence if evidence_exists(e, root) is None})
    return dict(subjects=[s._asdict() for s in SUBJECTS], kinds=[k._asdict() for k in KINDS], blocks={k: list(v) for k, v in BLOCKS.items()},
                items=rows, families=[fam(n, f, True) for n, f in FAMILIES.items()], planned=[fam(n, f, False) for n, f in PLANNED.items()],
                coverage=coverage(), coverage_with_planned=coverage(planned=True), waiting=waiting(), problems=problems, unchecked=unchecked, ok=not problems)


def _grid(t: dict) -> dict:
    """{(subject letter, kind number): (built names, names in construction)}, in the order of the tables."""
    out = {(s["letter"], k["number"]): ([], []) for s in t["subjects"] for k in t["kinds"]}
    for f in t["families"] + t["planned"]:
        out[(f["subject"], f["kind"])][0 if f["built"] else 1].append(f["name"])
    return out


def _counts_line(c: dict) -> str:
    return f"{c['formed']}" + (f" (+{c['partial']} in part)" if c["partial"] else "")


def text_table(t: dict, with_items: bool = False) -> str:
    """What ``mrgym map`` prints: the grid, the counts and, with ``with_items``, every item."""
    grid, c = _grid(t), t["coverage"]
    kinds = [f"{k['number']} {k['name']}" for k in t["kinds"]]
    width = max([18] + [len(n) + 2 for f in t["families"] + t["planned"] for n in [f["name"]]])
    out = [f"The map: {c['total']['listed']} pieces of MRI physics in {len(t['subjects'])} subjects; {len(t['families'])} task families, "
           f"{len(t['planned'])} in construction (in brackets)", "", " " * 39 + "".join(f"{k:<{width}}" for k in kinds)]
    for s in t["subjects"]:
        cells = [grid[(s["letter"], k["number"])] for k in t["kinds"]]
        cells = [built + [f"({n})" for n in planned] for built, planned in cells]
        for line in range(max(1, max(len(x) for x in cells))):
            head = f"{s['letter']}  {s['title']}" if line == 0 else ""
            out.append(f"{head:<39}" + "".join(f"{(x[line] if line < len(x) else ('-' if line == 0 else '')):<{width}}" for x in cells).rstrip())
    out += ["", f"{'':39}{'listed':>8}{'formed':>8}{'in part':>9}{'computed':>10}{'not yet':>9}{'isolated by a task':>20}{'used by a task':>16}"]
    for head, n in [(s["letter"] + "  " + s["title"], c["subjects"][s["letter"]]) for s in t["subjects"]] + [("all", c["total"])]:
        out.append(f"{head:<39}{n['listed']:>8}{n['formed']:>8}{n['partial']:>9}{n['computed']:>10}{n['absent']:>9}{n['isolated']:>20}{n['used']:>16}")
    out += ["computed: the package computes it and checks it, but no console sequence or task reaches it", "",
            f"not formed yet ({c['total']['computed'] + c['total']['absent']}), by the piece of engine work each waits for:"]
    out += [f"  {len(ids):>3}  {name or ORDINARY}" for name, ids in t["waiting"].items()]
    if with_items:
        for s in t["subjects"]:
            out += ["", f"{s['letter']}  {s['title']}: {s['line']}"]
            block = None
            for r in t["items"]:
                if r["subject"] != s["letter"]:
                    continue
                if r["block"] != block:
                    block = r["block"]
                    out.append(f"   {block}")
                tasks = ", ".join(r["isolated_by"] + [f"({n})" for n in r["planned_isolated_by"]]) or "-"
                out.append(f"     {r['id']:<4} {STATUS_WORDS[r['status']]:<21} {tasks:<28} {r['name']}")
        out += ["", "third column: the task that isolates the item (in brackets: in construction)"]
    out.append("")
    if t["problems"]:
        out += [f"problems: {len(t['problems'])}"] + [f"  - {p}" for p in t["problems"]]
    else:
        out.append("the map is consistent with the registry; " + (f"{len(t['unchecked'])} named tests and examples were not looked up (no checkout)"
                                                                  if t["unchecked"] else "every named check exists"))
    return "\n".join(out)


# ------------------------------------------------------------------ generated text
def _slug(env_id: str) -> str:
    return env_id.replace("/", "-").lower()


def _link(f: dict, prefix: str) -> str:
    return f"[{f['name']}]({prefix}{_slug(f['ids'][-1])}/)" if f["ids"] else f"*{f['name']}*"


def _names(names, fams, prefix):
    return " · ".join(_link(fams[n], prefix) for n in names) or "-"


def _evidence_text(evidence) -> str:
    """Named checks as one cell: the rows of ``mrgym validate`` together, those of each package of
    ``EVIDENCE_PACKAGES`` in the order of the table, the rungs of the comparison with measured data, then the tests,
    examples and commands."""
    rows = [e for e in evidence if _VALIDATE_ROW.match(e)]
    cells = [f"`mrgym validate` {', '.join(rows)}"] if rows else []
    for prefix, (_, _, _, command) in EVIDENCE_PACKAGES.items():
        mine = [row[1] for row in map(package_row, evidence) if row and row[0] == prefix]
        cells += [f"`{command}` {', '.join(mine)}"] if mine else []
    measured = [e.split(":")[1] for e in evidence if _MEASURED_RUNG.match(e)]
    cells += [f"`python tools/measured/run_all.py`, the rows of rungs {', '.join(measured)} ([measured](physics/measured.md))"] if measured else []
    return "; ".join(cells + [f"`{e}`" for e in evidence if not _VALIDATE_ROW.match(e) and not package_row(e) and not _MEASURED_RUNG.match(e)])


def _row_sources() -> str:
    """The one place where the page names the tables of rows: ``mrgym validate``, then every package of
    ``EVIDENCE_PACKAGES`` with the command a reader runs and its page."""
    parts = ["`mrgym validate` ([validation](physics/validation.md))"]
    parts += [f"`{command}` ([{prefix}]({page}))" for prefix, (_, _, page, command) in EVIDENCE_PACKAGES.items()]
    return "Rows " + ", ".join(f"of {p}" for p in parts[:-1]) + f" and of {parts[-1]}"


def _anchor(heading: str) -> str:
    """The anchor a Markdown renderer gives a heading (tools/check_links.py reads them the same way)."""
    return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def doc_page(t: dict | None = None, cards: str = "../envs/") -> str:
    """``docs/tasks.md``: the grid and the counts on one screen, then the items of each subject, then what is formed in
    part and what is not formed yet, grouped by the piece of engine work each waits for, then the check behind every
    item that names one."""
    t = t or table()
    fams = {f["name"]: f for f in t["families"] + t["planned"]}
    grid, c, total = _grid(t), t["coverage"], t["coverage"]["total"]
    out = ["<!-- Generated by tools/generate.py from src/mrgym/physics_map.py. Do not edit: change the table there and regenerate. -->",
           "# The map", "",
           "Which MRI physics this scanner covers, and which task asks for which piece of it.", "",
           f"The list holds {total['listed']} pieces of MRI physics and applied MR physics, as textbooks and curricula name them. The scanner "
           f"forms {total['formed']} of them and {total['partial']} more in part. The other {total['computed'] + total['absent']} are not formed "
           f"yet: {total['computed']} of them the package already computes and checks without serving them, and "
           f"[the last list](#not-formed-yet) says which piece of engine work each one waits for. "
           "Statistics of its images and signals have been compared with public phantom measurements on real scanners "
           "([measured](physics/measured.md)); no image of its sequences has been compared pixel by pixel with a scan of the same object.", "",
           "## On one screen", "",
           "Rows: what you need to know. Columns: what you do with it. A cell holds the tasks; *italic* means in construction, not playable yet.", "",
           "| | " + " | ".join(f"{k['number']} {k['name'].capitalize()}" for k in t["kinds"]) + " |", "|---|" + "---|" * len(t["kinds"])]
    for s in t["subjects"]:
        cells = [grid[(s["letter"], k["number"])] for k in t["kinds"]]
        out.append(f"| **{s['letter']}** {s['title']} | " + " | ".join(_names(b + p, fams, cards) for b, p in cells) + " |")
    out += ["", "| Subject | Listed | Formed | Isolated by a task | Used by a task |", "|---|---:|---:|---:|---:|"]
    for s in t["subjects"]:
        n = c["subjects"][s["letter"]]
        out.append(f"| **{s['letter']}** {s['title']}: {s['line']} | {n['listed']} | {_counts_line(n)} | {n['isolated']} | {n['used']} |")
    out += [f"| All | {total['listed']} | {_counts_line(total)} | {total['isolated']} | {total['used']} |", "",
            "Listed: items on the map. Formed: an agent meets it at the console, and a check that exists shows it. Isolated by a task: "
            "one task is about this item. Used by a task: a task takes it, as its own item or beside it. Tasks in construction are not "
            "counted, and neither is what the package computes without serving it.", "",
            "## The two axes", "", "| | Subject | A piece of physics is filed here when |", "|---|---|---|"]
    out += [f"| **{s['letter']}** | {s['title']}: {s['line']} | {s['test']} |" for s in t["subjects"]]
    out += ["", "| | Kind of task | What is delivered | For example |", "|---|---|---|---|"]
    out += [f"| **{k['number']}** | {k['name'].capitalize()} | {k['delivers']} | {k['example']} |" for k in t["kinds"]]
    out += ["", "Kinds 1 to 3 are the knowledge, one piece at a time and then several together. Kind 4 asks for the best scan the "
            "console can give. Kind 5 lets the agent build what the console does not offer. A task of kind 3 or more names the "
            "simple items it brings together; whether training on the simple ones carries upward is a measurement, not a claim of this page.", ""]
    for s in t["subjects"]:
        n = c["subjects"][s["letter"]]
        out += [f"## {s['letter']} · {s['title']}", "",
                f"{_cap(s['line'])}. {n['listed']} items, {_counts_line(n)} formed, {n['isolated']} isolated by a task, {n['used']} used by a task.", "",
                "| | Item | Rests on | Scanner | Task that isolates it | Tasks that use it |", "|---|---|---|---|---|---|"]
        block = None
        for r in t["items"]:
            if r["subject"] != s["letter"]:
                continue
            if r["block"] != block:
                block = r["block"]
                out.append(f"| | **{block}** | | | | |")
            out.append(f"| {r['id']} | {r['name']} | {' '.join(r['rests_on']) or '-'} | {STATUS_WORDS[r['status']]} | "
                       f"{_names(r['isolated_by'] + r['planned_isolated_by'], fams, cards)} | {_names(r['used_by'] + r['planned_used_by'], fams, cards)} |")
        out.append("")
    by_id = {r["id"]: r for r in t["items"]}
    out += ["## Formed in part", "", "| | Item | What is missing | Waits for |", "|---|---|---|---|"]
    out += [f"| {r['id']} | {r['name']} | {_cap(r['reason'])} | {r['needs'] or ORDINARY} |" for r in t["items"] if r["status"] == "partial"]
    waits = t["waiting"]
    pieces = [name for name, _ in ENABLERS if waits[name]]
    n_all, n_piece = sum(len(v) for v in waits.values()), sum(len(waits[name]) for name in pieces)
    out += ["", "## Not formed yet", "",
            f"{n_all} items. {n_piece} of them wait for one of {len(pieces)} pieces of engine work; the other {len(waits[''])} take only "
            "ordinary work on the present engine. \"Computed\" in the third column: the package already computes the item and checks it "
            "against a closed form or a second method, and no console sequence, parameter or task reaches it yet; "
            f"[the checks](#{_anchor(CHECKS_HEADING)}) name the rows.", "",
            "| Waits for | What it is | Items |", "|---|---|---:|"]
    out += [f"| **{name}** | {what} | {len(waits[name])} |" for name, what in ENABLERS]
    out += [f"| {ORDINARY} | nothing new: a term, a parameter or an option of the present engine | {len(waits[''])} |", ""]
    for name, what in [*ENABLERS, ("", "")]:
        if not waits[name]:
            continue
        out += [f"### {_cap(name or ORDINARY)}", ""] + ([f"{_cap(what)}.", ""] if what else [])
        out += ["| | Item | In the package | What is missing |", "|---|---|---|---|"]
        out += [f"| {i} | {by_id[i]['name']} | {'computed' if by_id[i]['status'] == 'computed' else '-'} | {_cap(by_id[i]['reason'])} |" for i in waits[name]]
        out.append("")
    out += ["## The tasks and what they build on", "",
            "Declared from the task cards, not measured: the tasks whose skill a task builds on.", "",
            "| Task | Subject · block | Kind | Isolates | Uses | Builds on |", "|---|---|---|---|---|---|"]
    for f in t["families"] + t["planned"]:
        out.append(f"| {_link(f, cards)} | {f['subject']} · {f['block']} | {f['kind']} {f['kind_name']} | {f['isolates']} | {' '.join(f['uses']) or '-'} | "
                   f"{_names(f['builds_on'], fams, cards)} |")
    out += ["", f"## {CHECKS_HEADING}", "",
            f"{_row_sources()}, tests, textbook examples that assert their result, commands.", "",
            "| | Check |", "|---|---|"]
    out += [f"| {r['id']} | {_evidence_text(r['evidence'])} |" for r in t["items"] if r["evidence"]]
    out += ["", "## Where the list comes from", "",
            f"The items were drawn up from the tables of contents of {len(TEXTBOOKS)} textbooks and from {len(OUTLINES)} teaching and "
            "examination outlines, read at the level of their headings. They are a map of topics; no examination item is reproduced.", ""]
    out += [f"- {b}" for b in TEXTBOOKS + OUTLINES] + [""]
    return "\n".join(out)


def card_line(env_id: str, page: str = "../../docs/tasks.md") -> str | None:
    """The line under the title of an environment card: subject, block, kind and the item the task isolates. None for
    an id without a row (the sandbox, a private family)."""
    found = family_of(env_id)
    if not found:
        return None
    letter, block, number = cell(found[1])
    i = items()[found[1].isolates]
    return (f"**{letter} · {subject(letter).title}** · {block} · kind {number}, {kind(number).name} · isolates {i.id}: {i.name}. "
            f"[The map]({page})")


def groups() -> list:
    """The served ids in the order of the map: [(letter, heading, [(environment id, kind number or None)])]. One group
    per subject (by kind, then in the order of ``FAMILIES``), then the sandbox ("-") and the ids whose family has no
    row ("?": for example a private family)."""
    import mrgym
    placed, sandbox, rest = {s.letter: [] for s in SUBJECTS}, [], []
    for env_id in mrgym.list_envs():
        s, found = mrgym.spec(env_id), family_of(env_id)
        if found and s.family and s.kind != "sandbox":
            letter, _, number = cell(found[1])
            placed[letter].append((number, list(FAMILIES).index(found[0]), env_id))
        else:
            (sandbox if s.kind == "sandbox" else rest).append((env_id, None))
    out = [(s.letter, s.title, [(env_id, number) for number, _, env_id in sorted(placed[s.letter])]) for s in SUBJECTS]
    return out + [("-", "Sandbox", sandbox)] * bool(sandbox) + [("?", "Not on the map (no row in mrgym.physics_map)", rest)] * bool(rest)


def readme_table(what=None, status=None, cards: str = "envs/", not_ready: str = "not ready") -> str:
    """The task table of the front page: one table with a bold row per subject (its letter, title and plain line) and
    under it the tasks of the subject, by kind and, inside a kind, in the order of ``status``. Columns: Task, What you
    do, Kind, Status. The sandbox stands first, outside the subjects; an id whose family has no row comes last.

    ``what``       {environment id: a plain phrase}; an id that is not in it shows the title of its card
    ``status``     {kind of the registry: the word shown}, in the order in which they are listed; default: the kind itself
    ``not_ready``  the word shown, whatever the kind, for an id that is registered as known not to work as stated on
                   the current engine (``EnvSpec.not_ready``; its card says why); such ids stand last inside a kind"""
    import mrgym
    what, status = what or {}, status or {}
    rank = {k: n for n, k in enumerate(status)}

    def row(env_id, number):
        s = mrgym.spec(env_id)
        return (f"| [`{env_id}`]({cards}{_slug(env_id)}/) | {what.get(env_id) or mrgym.card(env_id)['title']} | "
                f"{kind(number).name if number else '-'} | {not_ready if s.not_ready else status.get(s.kind, s.kind)} |")
    found = groups()
    out = ["| Task | What you do | Kind | Status |", "|---|---|---|---|"]
    out += [row(env_id, None) for letter, _, rows in found if letter == "-" for env_id, _ in rows]
    for letter, title, rows in found:
        if letter in "-?":
            continue
        out.append(f"| **{letter} · {title}**: {subject(letter).line} | | | |")
        ordered = sorted(rows, key=lambda r: (r[1], len(rank) + 1 if mrgym.spec(r[0]).not_ready       # stable: then the order of FAMILIES
                                              else rank.get(mrgym.spec(r[0]).kind, len(rank))))
        out += [row(env_id, number) for env_id, number in ordered]
    out += [row(env_id, None) for letter, _, rows in found if letter == "?" for env_id, _ in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ command
def add_parser(sub):
    p = sub.add_parser("map", help="the map: subjects x kinds of task, which physics is formed, which task asks for it")
    p.add_argument("--items", action="store_true", help="also print every item with its status and the task that isolates it")
    p.add_argument("--assert", dest="enforce", action="store_true", help="exit 1 unless the map has no problem")
    p.add_argument("--json", action="store_true")
    p.add_argument("--write", metavar="PATH", help="write the page of docs/tasks.md (refused while the map has a problem)")
    p.set_defaults(fn=cmd_map)
    return p


def cmd_map(a) -> int:
    t = table()
    if a.write and t["ok"]:
        with open(a.write, "w", encoding="utf-8") as f:
            f.write(doc_page(t))
    print(json.dumps(t, indent=1) if a.json else text_table(t, a.items))
    return 1 if (a.enforce or a.write) and not t["ok"] else 0
