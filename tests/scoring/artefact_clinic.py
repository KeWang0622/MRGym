"""Scoring table of ``Scanner/ArtefactClinic-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): every finding is given exactly its causes; on a noise-free scan of the delivered protocol
every finding agrees with the same scan without the artefact mechanisms; scan time, pixel size, SNR, the TE window and
the locked parameters are as the card requires.

What is derived here, without the task module: the scan time of a deliverable from the console's dry run, the pixel
size as field of view over matrix, the window and the locked values from the card, and the diagnosis from the hidden
causes of the instance. The SNR floor is the number on the card, or at level 3 the share on the card times the SNR of
the stored remedy (rounded to 0.1 by the generator). The passing deliverable is the remedy the generator stored with the instance (hidden with the
world; it is the instance's own proof that a clean protocol exists) together with the hidden causes. The twin
comparison itself is covered by ``tests/test_artefact_clinic.py`` on a hand-made object.
"""
from mrgym.scanner import console
from scoring import Row

ENV_ID = "Scanner/ArtefactClinic-v0"
INSTANCES = {"L2": dict(level=2, seed=1), "L3": dict(level=3, seed=0)}
FAT_PPM_HZ_PER_T = 3.4e-6 * 42.577478518e6


def good(case):
    h = case.cfg["hidden"]
    return dict(causes={k: list(v) for k, v in h["causes"].items()}, protocol=dict(h["remedy"]))


def protocol(**change):
    """The good deliverable with protocol parameters changed; a callable value gets the good protocol."""
    def make(case):
        a = good(case)
        a["protocol"].update({k: v(a["protocol"]) if callable(v) else v for k, v in change.items()})
        return a
    return make


def other_cause(case):
    """The first finding with another diagnosis: none where it has a cause, the first cause of the card where it has none."""
    a = good(case)
    first = case.cfg["task"]["findings"][0]["id"]
    a["causes"][first] = [] if a["causes"][first] else [case.cfg["task"]["causes_vocabulary"][0]]
    return a


def two_fat_periods(case):
    return 2e3 / (FAT_PPM_HZ_PER_T * case.cfg["task"]["exam"]["field_t"])


def too_slow(case, score):
    req = case.cfg["task"]["requirements"]
    scan_s = console.plan(score["answer"]["protocol"], case.exam)["scan_s"]
    assert scan_s > req["max_scan_s"] and abs(score["scan_s"] - scan_s) < 1e-9


def outside_window(case, score):
    lo, hi = case.cfg["task"]["requirements"]["te_ms"]
    assert not lo <= score["answer"]["protocol"]["te_ms"] <= hi


def within_every_requirement(case, score):
    req, p = case.cfg["task"]["requirements"], score["answer"]["protocol"]
    assert console.plan(p, case.exam)["scan_s"] <= req["max_scan_s"] and p["fov_mm"] / p["matrix"] <= req["max_pixel_mm"]
    assert req["te_ms"][0] <= p["te_ms"] <= req["te_ms"][1] and all(p[k] == v for k, v in req["locked"].items())
    floor = req["min_snr_reference"] if "min_snr_reference" in req else req["snr_share"] * case.cfg["hidden"]["snr_remedy"]   # level 3: a share
    assert score["snr_reference"] >= floor - 0.06 and score["diagnosis_correct"] == score["findings"]
    assert score["error"] <= 1.0 and all(score["checks"].values())


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None


SHAPE = 'the deliverable is {"causes": {'
CONSOLE = "the console rejects the delivered protocol"
ROWS = [
    Row("L2", "the hidden causes and the stored remedy", good, passed=True, expect=within_every_requirement),
    Row("L2", "one finding with another diagnosis", other_cause, passed=False, failing=("diagnosis_F1",), reason="wrong diagnosis of F1"),
    Row("L2", "eight more averages: over the scan-time limit", protocol(averages=lambda p: p["averages"] + 8), passed=False,
        failing=("scan_time",), reason="scan time", expect=too_slow),
    Row("L2", "a locked parameter changed by one degree", protocol(flip_deg=lambda p: p["flip_deg"] - 1.0), passed=False,
        failing=("locked_parameters",)),
    Row("L2", "the echo two fat periods later: outside the window", lambda c: protocol(te_ms=lambda p: round(p["te_ms"] + two_fat_periods(c), 2))(c),
        passed=False, failing=("te_window",), expect=outside_window),
    Row("L3", "the hidden causes and the stored remedy", good, passed=True, expect=within_every_requirement),
    Row("L3", "one finding with another diagnosis", other_cause, passed=False, failing=("diagnosis_F1",), reason="wrong diagnosis of F1"),
    Row("L3", "a locked parameter changed by one degree", protocol(flip_deg=lambda p: p["flip_deg"] - 1.0), passed=False,
        failing=("locked_parameters",)),
    # ---- deliverables the console or the schema cannot evaluate: refused, the episode continues
    Row("L2", "the echo time in seconds", protocol(te_ms=lambda p: p["te_ms"] / 1000.0), refused=True, reason=CONSOLE, expect=not_evaluated),
    Row("L2", "a finding is missing", lambda c: dict(good(c), causes={"F1": []}), refused=True, reason=SHAPE),
    Row("L2", "a cause that is not in the vocabulary", lambda c: dict(good(c), causes=dict(good(c)["causes"], F1=["zipper"])), refused=True,
        reason="causes of F1"),
    Row("L2", "three causes for one finding", lambda c: dict(good(c), causes=dict(good(c)["causes"], F1=list(c.cfg["task"]["causes_vocabulary"][:3]))),
        refused=True, reason="causes of F1"),
    Row("L2", "the protocol without the diagnosis", lambda c: dict(protocol=good(c)["protocol"]), refused=True, reason=SHAPE),
    Row("L2", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L3", "a cause this level does not serve", lambda c: dict(good(c), causes=dict(good(c)["causes"], F1=["truncation_ringing"])), refused=True,
        reason="causes of F1"),
]
