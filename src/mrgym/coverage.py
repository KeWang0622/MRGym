"""The competency map: what the field examines, what the scanner forms, and which environment asks for it.

    mrgym coverage             the table
    mrgym coverage --assert    exit 1 unless the stated thresholds hold
    mrgym coverage --json

The rows are the Image Production items of the ARRT MRI content specification that a scanner simulator can carry
(``SOURCE``): the 18 imaging parameters (2.A), the 11 imaging options (2.C), the 14 artefacts (1.C) and the quality
control items (1.D; "equipment inspection" is an inspection of hardware and is listed as not applicable). The outline
is used as a map of topics. No examination item is reproduced.

Two columns are data of this module, written by reading the console and the engine:
    engine    "now"      the console has the parameter or the engine forms the effect
              "partial"  part of the item (the note says which part)
              "absent"   not modelled
              "n/a"      not something a simulator can carry
    how       the console parameter, the mechanism or the command that shows it
One column is computed: the registered scored environments whose family declares the row in ``task.COMPETENCIES``
(an optional attribute of a family, ``mrgym/envs/family.py``). An environment that declares nothing is listed as
such; it exercises no row here until its family says which.

Thresholds (``THRESHOLDS``; printed always, enforced by ``--assert``): the number of rows per group that some scored
environment exercises, that every scored environment declares at least one row, that every declared id is a row of
the map, and that every row marked "now" is exercised by an environment or names a check that exists (a row of
``mrgym validate`` or a command).

This map follows one outline line by line. The map of the physics itself, by subject and with every task placed on it,
is ``mrgym.physics_map`` (``mrgym map``, ``docs/tasks.md``).
"""
from __future__ import annotations

import importlib
import json

SOURCE = ("ARRT Examination Content Specifications, Magnetic Resonance Imaging (board approved January 2024, implemented "
          "February 1, 2025), Image Production: 1.C Artifacts, 1.D Quality Control, 2.A Imaging Parameters, 2.C Imaging Options")
GROUPS = ("parameter", "option", "artefact", "qc")
GROUP_TITLES = dict(parameter="imaging parameters (2.A)", option="imaging options (2.C)", artefact="artefacts (1.C)", qc="quality control (1.D)")
# id, outline number, item as the outline names it, engine, how (parameter / mechanism), check (a row of `mrgym validate` or a command)
_ROWS = (
    ("parameter.tr", "2.A.1", "repetition time (TR)", "now", "tr_ms", "S1"),
    ("parameter.te", "2.A.2", "echo time (TE)", "now", "te_ms", "S10"),
    ("parameter.ti", "2.A.3", "inversion time (TI)", "now", "ti_ms (ir)", "S13"),
    ("parameter.averages", "2.A.4", "number of signal averages (NSA, NEX)", "now", "averages", "K15"),
    ("parameter.flip_angle", "2.A.5", "flip angle", "now", "flip_deg", "S4"),
    ("parameter.fov", "2.A.6", "field of view (FOV)", "now", "fov_mm, phase_fov", "K19"),
    ("parameter.matrix", "2.A.7", "matrix", "now", "matrix", "K15"),
    ("parameter.pixel", "2.A.8", "pixel", "now", "fov_mm / matrix (plan: voxel_mm)", "K17"),
    ("parameter.voxel", "2.A.9", "voxel", "now", "pixel x slice_mm (plan: voxel_mm)", "K17"),
    ("parameter.number_of_slices", "2.A.10", "number of slices", "absent", "one 2D slice per scan", None),
    ("parameter.slice_thickness_gap", "2.A.11", "slice thickness and gap", "partial", "slice_mm (the slice gradient is calibrated per pulse); no gap (one slice)", "R13"),
    ("parameter.phase_frequency_direction", "2.A.12", "phase and frequency", "now", "pe_dir, ro_polarity", "K3"),
    ("parameter.etl", "2.A.13", "echo train length (ETL)", "now", "etl of fse, n_echoes of mese (a console whose card lists them)", "J37"),
    ("parameter.effective_te", "2.A.14", "effective TE", "now", "te_ms and echo_order of fse: the echo that fills the centre of k-space", "J18"),
    ("parameter.receive_bandwidth", "2.A.15", "bandwidth (transmit, receive)", "now", "bandwidth_hz_per_px; rf_pulse (time-bandwidth product of the excitation)", "K3"),
    ("parameter.concatenations", "2.A.16", "concatenations (number of acquisitions per TR)", "absent", "one slice", None),
    ("parameter.b_value", "2.A.17", "b-value", "now", "b_value of dwse (a console whose card lists it); the echo time it costs", "J31"),
    ("parameter.venc", "2.A.18", "velocity encoding (VENC)", "absent", "no flow", None),
    ("option.2d_3d", "2.C.1", "2D and 3D", "now", "excitation: selective / nonselective", "C1"),
    ("option.slice_order", "2.C.2", "slice order (sequential, interleaving)", "absent", "one slice", None),
    ("option.saturation_band", "2.C.3", "spatial saturation pulse / band", "absent", "", None),
    ("option.gradient_moment_nulling", "2.C.4", "gradient moment nulling", "absent", "no flow", None),
    ("option.suppression", "2.C.5", "suppression techniques (e.g., fat, water, Dixon method)", "now", "fat_sat, fatsat_offset_hz; ir (STIR, FLAIR); multi-echo spgr", "S13"),
    ("option.gating", "2.C.6", "physiologic gating and triggering", "absent", "", None),
    ("option.in_out_of_phase", "2.C.7", "in-phase and out-of-phase", "now", "te_ms of spgr with the six-peak fat spectrum", "K18"),
    ("option.rectangular_fov", "2.C.8", "rectangular FOV", "now", "phase_fov", "K19"),
    ("option.anti_aliasing", "2.C.9", "anti-aliasing", "now", "phase_oversampling", "K7"),
    ("option.parallel_imaging", "2.C.10", "parallel imaging", "absent", "single receive channel", None),
    ("option.filtering", "2.C.11", "filtering", "absent", "the reconstruction is an unfiltered inverse FFT", None),
    ("artefact.aliasing", "1.C.1", "aliasing (wrap around)", "now", "k-space sampling of the phase-encode axis", "K6"),
    ("artefact.gibbs", "1.C.2", "Gibbs, truncation", "now", "finite k-space", "K8"),
    ("artefact.chemical_shift", "1.C.3", "chemical shift", "now", "fat displaced along the readout (EPI: along phase)", "K3"),
    ("artefact.chemical_misregistration", "1.C.4", "chemical misregistration", "now", "water-fat cancellation at opposed-phase echo times", "K18"),
    ("artefact.susceptibility", "1.C.5", "magnetic susceptibility", "now", "dipole field of air cavities: distortion, pile-up, through-slice dropout", "K12"),
    ("artefact.rf_zipper", "1.C.6", "radiofrequency (e.g., zipper)", "now", "interference line of the hidden system (rfi)", "K14"),
    ("artefact.motion_ghost", "1.C.7", "motion and flow (e.g., patient motion, ghosting)", "partial", "periodic motion of a tissue class (ghost, blur); no flow", "mrgym qa acr --fault signal_instability"),
    ("artefact.partial_volume", "1.C.8", "partial volume averaging", "now", "soft partial-volume weights on a master grid finer than any pixel", None),
    ("artefact.cross_talk", "1.C.9", "cross-talk", "absent", "one slice", None),
    ("artefact.cross_excitation", "1.C.10", "cross-excitation", "absent", "one slice", None),
    ("artefact.moire", "1.C.11", "moire pattern", "absent", "not checked", None),
    ("artefact.parallel_imaging", "1.C.12", "parallel imaging artifacts", "absent", "single receive channel", None),
    ("artefact.eddy_currents", "1.C.13", "eddy currents", "partial", "a gradient delay: the Nyquist ghost of EPI", "K13"),
    ("artefact.b1_shading", "1.C.14", "dielectric effect", "now", "field-dependent transmit pattern (B1+), larger at 3 T", "mrgym qa acr --fault b1_nonuniformity"),
    ("artefact.b0_distortion", "1.C.5", "magnetic susceptibility: geometric distortion by a field gradient (with 1.C.5)", "now", "displacement along the readout by off-resonance / bandwidth", "K5"),
    ("qc.slice_thickness", "1.D.1", "slice thickness", "absent", "the excitation has a slice profile, but no object has structure through the slice to measure it on", None),
    ("qc.spatial_resolution", "1.D.2", "spatial resolution", "now", "resolution insert of the acr world", "mrgym qa acr"),
    ("qc.contrast_resolution", "1.D.3", "contrast resolution", "now", "low-contrast sections of the acr world", "mrgym qa acr"),
    ("qc.snr", "1.D.4", "signal to noise", "now", "receiver noise of the hidden system", "K15"),
    ("qc.centre_frequency", "1.D.5", "center frequency", "now", "centre-frequency offset of the hidden system; cf_shift_hz adjusts it", "K33"),
    ("qc.transmit_gain", "1.D.6", "transmit gain", "now", "transmit-gain error of the hidden system; afi; tx_scale adjusts it", "K35"),
    ("qc.geometric_accuracy", "1.D.7", "geometric accuracy", "now", "acr world; gradient calibration error of the hidden system", "mrgym qa acr"),
    ("qc.equipment_inspection", "1.D.8", "equipment inspection (e.g., coils, cables, door seals)", "n/a", "an inspection of hardware", None),
    # ---- beyond the outline's own lines: what the QC items are for (ARRT task inventory, tasks 45 and 46: perform and interpret QC tests)
    ("qc.signal_uniformity", "1.D (ACR 5)", "image intensity uniformity (ACR phantom test 5; with 1.D)", "now", "transmit and receive profiles; acr world", "mrgym qa acr"),
    ("qc.ghosting", "1.D (ACR 6)", "percent-signal ghosting (ACR phantom test 6; with 1.D)", "now", "pulse-to-pulse instability of the hidden system; acr world", "mrgym qa acr"),
    ("qc.fault_diagnosis", "1.D (ACR x.5)", "causes of failure of a QC test and corrective action (ACR guidance, sections x.5)", "now", "hidden system faults", "mrgym qa acr --matrix"),
)
COMPETENCIES = [dict(id=i, outline=o, item=item, group=i.split(".")[0], engine=e, how=how, check=check) for i, o, item, e, how, check in _ROWS]
# Rows that are a second view of an outline line (they do not count in the outline's own totals).
EXTRA = ("artefact.b0_distortion", "qc.signal_uniformity", "qc.ghosting", "qc.fault_diagnosis")
OUTLINE_TOTALS = dict(parameter=18, option=11, artefact=14, qc=7)       # qc: 8 lines of the outline, one of them not applicable
# The targets of this map: rows per group that a scored environment exercises.
THRESHOLDS = dict(parameter=15, artefact=10, qc=6)


def ids() -> list:
    return [c["id"] for c in COMPETENCIES]


def declared() -> dict:
    """{environment id: tuple of competency ids, or None when the family declares none} for every scored id."""
    import mrgym
    out = {}
    for env_id in mrgym.list_envs():
        spec = mrgym.spec(env_id)
        if spec.kind == "sandbox" or not spec.family:
            continue
        try:
            task = importlib.import_module(spec.family + ".task")
        except ModuleNotFoundError:
            out[env_id] = None
            continue
        c = getattr(task, "COMPETENCIES", None)
        out[env_id] = None if c is None else tuple(c)
    return out


def _check_exists(check: str | None) -> bool:
    """A row of ``mrgym validate`` (by id), or a command of this package (``mrgym qa ...``)."""
    if not check:
        return False
    if check.startswith("mrgym qa acr"):
        from . import qa
        fault = check.split("--fault ")[1] if "--fault " in check else None
        return fault is None or fault in qa.FAULTS
    return check in validate_ids()


def validate_ids() -> set:
    """Ids of the rows of ``mrgym validate``, read from the source of the module (running them takes a minute)."""
    import inspect
    import re

    from . import validate
    from .scanner import served_checks  # rows J7 on: the sequences served from the block-list core
    shadow = {f"J{i + 1}" for i in range(len(validate.SHADOW))}          # rows J1 to J6 are made in a loop over ``validate.SHADOW``
    return set(re.findall(r'_row\("([A-Z]+[0-9]+)"', inspect.getsource(validate) + inspect.getsource(served_checks))) | shadow


def table() -> dict:
    """The map with its computed column and the counts."""
    dec = declared()
    known = set(ids())
    by_row = {i: sorted(e for e, c in dec.items() if c and i in c) for i in known}
    rows = [dict(c, environments=by_row[c["id"]], check_exists=_check_exists(c["check"])) for c in COMPETENCIES]
    counts = {}
    for g in GROUPS:
        own = [r for r in rows if r["group"] == g and r["id"] not in EXTRA and r["engine"] != "n/a"]
        counts[g] = dict(outline=OUTLINE_TOTALS[g], rows=len(own), now=sum(r["engine"] == "now" for r in own),
                         partial=sum(r["engine"] == "partial" for r in own), absent=sum(r["engine"] == "absent" for r in own),
                         exercised=sum(bool(r["environments"]) for r in own), threshold=THRESHOLDS.get(g))
    problems = []
    for g, need in THRESHOLDS.items():
        if counts[g]["exercised"] < need:
            problems.append(f"{GROUP_TITLES[g]}: {counts[g]['exercised']} of {counts[g]['outline']} exercised by a scored environment; the threshold is {need}")
    for g in GROUPS:
        if counts[g]["rows"] != counts[g]["outline"]:
            problems.append(f"{GROUP_TITLES[g]}: the map holds {counts[g]['rows']} rows, the outline {counts[g]['outline']}")
    for env_id, c in dec.items():
        if not c:
            problems.append(f"{env_id} declares no competency (task.COMPETENCIES)")
        else:
            problems += [f"{env_id} declares {i!r}, which is not a row of the map" for i in c if i not in known]
    for r in rows:
        if r["engine"] == "now" and not r["environments"] and not r["check_exists"]:
            problems.append(f"{r['id']} is marked 'now' but no environment exercises it and its check ({r['check']}) does not exist")
    return dict(source=SOURCE, rows=rows, counts=counts, environments={e: (list(c) if c else None) for e, c in dec.items()},
                thresholds=dict(THRESHOLDS), problems=problems, ok=not problems)


def text_table(t: dict) -> str:
    out = [f"Competency map · source: {t['source']}", ""]
    w = max(len(r["id"]) for r in t["rows"])
    for g in GROUPS:
        c = t["counts"][g]
        need = f" (threshold for an A: {c['threshold']})" if c["threshold"] else ""
        out.append(f"== {GROUP_TITLES[g]}: {c['outline']} items · engine forms {c['now']} fully, {c['partial']} partly, {c['absent']} not · "
                   f"exercised by a scored environment: {c['exercised']}{need}")
        for r in t["rows"]:
            if r["group"] != g:
                continue
            envs = ", ".join(e.split("/")[1] for e in r["environments"]) or "-"
            extra = " *" if r["id"] in EXTRA else ""
            out.append(f"  {r['outline']:<14}{r['id']:<{w}}  {r['engine']:<8} {envs:<34} {r['item']}{extra}")
        out.append("")
    out.append("* a second view of a line of the outline or of the ACR guidance; not counted in the totals")
    out.append("")
    out.append("scored environments and what their families declare:")
    for e, c in t["environments"].items():
        out.append(f"  {e:<26} " + (f"{len(c)} rows" if c else "declares none"))
    out.append("")
    if t["problems"]:
        out.append(f"thresholds and consistency: {len(t['problems'])} not met (enforced by --assert)")
        out += [f"  - {p}" for p in t["problems"]]
    else:
        out.append("thresholds and consistency: all met")
    return "\n".join(out)


def doc_page(t: dict) -> str:
    """``docs/community/competency_map.md``: the table of the command inside a page."""
    return "\n".join(["# Competency map", "",
                      "Generated by `mrgym coverage --write docs/community/competency_map.md`. Do not edit: regenerate (a test compares this page with "
                      "the command; a family that is added or that declares its `COMPETENCIES` changes it).", "",
                      "What the field examines (the Image Production outline of the ARRT MRI content specification; topics only, no examination item), "
                      "what the console and the engine form, and which scored environment asks for it. Columns: outline number, id, engine "
                      "(now / partial / absent / n/a), environments, item.", "", "```", text_table(t), "```", ""])


def add_parser(sub):
    p = sub.add_parser("coverage", help="the competency map: examined items x what the engine forms x which environment asks for it")
    p.add_argument("--assert", dest="enforce", action="store_true", help="exit 1 unless the thresholds and the consistency checks hold")
    p.add_argument("--json", action="store_true")
    p.add_argument("--write", metavar="PATH", help="write the page of docs/community/competency_map.md")
    p.set_defaults(fn=cmd_coverage)
    return p


def cmd_coverage(a) -> int:
    t = table()
    if a.write:
        with open(a.write, "w", encoding="utf-8") as f:
            f.write(doc_page(t))
    print(json.dumps(t, indent=1) if a.json else text_table(t))
    return 1 if a.enforce and not t["ok"] else 0
