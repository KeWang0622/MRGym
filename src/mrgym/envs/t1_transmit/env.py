"""``Scanner/T1Transmit-v0``: measure T1 in marked regions with an uncalibrated transmit field.

A thin binding: the world, pricing, noise, log and replay are ``mrgym.scanner``; the instance generator and the
scoring rule are ``task``; the scanner tools are ``ScannerEnv``. This file adds the deliverable and the card.
"""
from __future__ import annotations

from ... import admission
from ... import render as R
from ...tools import ToolSpec
from ..scanner_env import ScannerEnv
from . import controls, task

ENV_ID = "Scanner/T1Transmit-v0"


class T1TransmitEnv(ScannerEnv):
    def __init__(self, spec, tolerance: float | None = None, calibration_draws: int = 32, **kw):
        super().__init__(spec, **kw)
        self._fixed_tolerance, self._n_cal = tolerance, calibration_draws

    def make_instance(self, seed_hex, task_id):
        self._tolerance = self._fixed_tolerance
        return task.make_instance(seed_hex, self.level, task_id=task_id)

    def task_text(self):
        return task.statement(self._cfg)

    def terminal_tool(self):
        ids = [r["id"] for r in self._regions] if self._cfg else []
        return ToolSpec("submit", "Deliver the result and end the episode. T1 of every marked region in milliseconds"
                        + (f" (regions: {', '.join(ids)})." if ids else "."),
                        {"type": "object", "additionalProperties": False, "required": ["units", "t1_ms"],
                         "properties": {"units": {"type": "string", "enum": ["ms"]},
                                        "t1_ms": {"type": "object", "additionalProperties": False if ids else {"type": "number"},
                                                  "required": ids, "properties": {i: {"type": "number"} for i in ids}}}},
                        terminal=True)

    def tolerance(self) -> float:
        """Pass tolerance of this instance: from the reference policy's error over noise draws the episode never uses
        (draws 1..n; the episode runs on draw 0). Read from the stored admission run when the instance is in it,
        otherwise computed on first use."""
        if self._tolerance is None:
            stored = admission.stored_instance(ENV_ID, self._cfg["task_id"])
            if stored and stored.get("tolerance"):
                self._tolerance = stored["tolerance"]
            else:
                errs = []
                for d in range(1, self._n_cal + 1):
                    answer, sess, _ = controls.run_policy(self._cfg, controls.reference, d, self._session.engine)
                    e = task.score(self._cfg, answer, 1.0, sess.n_scans, world=self._session.world)["error"]
                    errs.append(10.0 if e is None else e)
                self._tolerance = controls.tolerance_from(errs)
        return self._tolerance

    def finish(self, a, rep):
        s = task.score(self._cfg, a, self.tolerance(), rep["scans"], rep["ok"] and not rep["over_budget"], world=self._session.world)
        text = "answer received; the episode is over"
        if not s["gate_deliverable"]:
            text += f"\nthe deliverable is not valid: {s['diagnostic']}"
        if self.reveal_score:
            text += (f"\nreward {s['reward']:.3f} · rms relative T1 error {R.sig(s['error'], 3)} · pass tolerance {R.sig(s['tolerance'], 3)} · "
                     f"error of the no-scan answer {R.sig(s['e_prior'], 3)} · gates: log {s['gate_log']}, measured {s['gate_measured']}, "
                     f"deliverable {s['gate_deliverable']}")
        return s["reward"], s, text


CONTROL_ROWS = [
    ("reference", "pass", "maps B1 (dual-TR, signed), sets flip angles from it, fills the budget, joint fit"),
    ("prior", "fail", "zero scans, the prior's geometric mean (defines reward 0)"),
    ("blind_vfa", "fail", "textbook two-angle VFA with nominal flip angles (confounder-blind)"),
    ("fixed_recipe", "fail", "one constant protocol tuned on other worlds, applied blindly"),
    ("brute_force", "fail", "uniform sweep at console defaults"),
    ("ir_only", "fail", "inversion recovery only, as much as fits"),
    ("wrong_units", "fail", "the reference's numbers in seconds"),
    ("seed_attacker", "fail", "rebuilds worlds from guessable seeds with the public generator"),
    ("open_loop", "reported", "the fixed recipe with averages scaled to the budget"),
    ("short_tr_ideal", "reported", "B1-corrected pipeline at the shortest TR and default spoilers"),
    ("ir_unpriced", "reported", "inversion recovery only, with 40 times the budget"),
]


def card() -> dict:
    return dict(
        id=ENV_ID, title="T1 in marked regions with an uncalibrated transmit field", kind="practice",
        given="A QA phantom with 4 to 7 vials, the scanner, a scan-time budget of 45 to 100 s",
        deliver="T1 of every marked vial, in ms",
        scored_by="RMS relative T1 error against the hidden phantom; full marks inside the reference policy's own noise band",
        summary="A hidden QA phantom (gel body and 4 to 7 vials) on the 3 T scanner. The agent must deliver T1 per marked "
                "region within a scan-time budget. The answer is wrong unless the transmit-field error is mapped and "
                "corrected and the gradient-echo protocol keeps spoiling errors small.",
        why_practice="Every instance needs the same procedure, so one constant protocol passes. The admission gate rejects "
                     "the environment for that reason; it stays as a practice task and as the worked example of the gate.",
        world=dict(phantom="QA phantom (disc of gel with vials), 3 T, body-coil receive",
                   sequences_useful=["afi", "spgr", "ir"],
                   agent_side_formulas="ideal-spoiling Ernst / AFI closed forms (datasheet section 5); the served signal is "
                                       "the EPG steady state and deviates from them at short TR and weak spoilers"),
        hidden=["geometry of body and vials", "PD, T1, T2, T2', diffusion coefficient per material",
                "B1+ pattern and transmit-gain error (8 to 25 %)", "B0 pattern and centre-frequency offset",
                "receive profile", "inversion efficiency", "noise level", "the 128-bit world seed"],
        told=["region centres and radii", "T1 prior range 300 to 2500 ms", "the instrument datasheet",
              "budget and request limits", "realised protocol, timing, limits and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=list(d["budget_s"])) for lv, d in task.LEVELS.items()},
        limits=dict(max_scanner_requests=48, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"units": "ms", "t1_ms": {"<region id>": "<number>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "deliverable valid: units \"ms\", exactly the region ids, values in 30 to 20000 ms"],
                    error="E = sqrt(mean over regions of (T1_delivered / T1_true - 1)^2)",
                    score="clip(ln(E_prior / E) / ln(E_prior / tol), 0, 1); E_prior = error of the no-scan answer; "
                          "tol = max(2 x median, 1.25 x 95th percentile) of the reference policy's error over noise draws on this instance",
                    passing="E <= tol (reward 1)"),
        controls=[dict(name=n, must=m, what=w) for n, m, w in CONTROL_ROWS],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise",
    )
