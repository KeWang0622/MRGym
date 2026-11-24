"""``Scanner/NoWrap-v0``: prescribe the fastest geometry that shows the whole object without wrap-around."""
from __future__ import annotations

from ...tools import ToolSpec
from ..scanner_env import ScannerEnv
from . import controls, task

ENV_ID = "Scanner/NoWrap-v0"


class NoWrapEnv(ScannerEnv):
    def make_instance(self, seed_hex, task_id):
        return task.make_instance(seed_hex, self.level, task_id=task_id)

    def task_text(self):
        return task.statement(self._cfg)

    def terminal_tool(self):
        return ToolSpec("submit", "Deliver the geometry of the target spin-echo scan and end the episode. The field of view is "
                        "in millimetres.",
                        {"type": "object", "additionalProperties": False, "required": ["pe_dir", "fov_mm", "phase_fov", "matrix"],
                         "properties": {"pe_dir": {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis"},
                                        "fov_mm": {"type": "number", "description": "readout field of view, mm"},
                                        "phase_fov": {"type": "number", "description": "phase field of view as a fraction of fov_mm"},
                                        "matrix": {"type": "integer", "description": "pixels along the readout (even)"},
                                        "phase_oversampling": {"type": "number", "description": "fraction; default 0"}}},
                        terminal=True)

    def finish(self, a, rep):
        s = task.score(self._cfg, a, rep["scans"], rep["ok"] and not rep["over_budget"], world=self._session.world,
                       engine=self._session.engine)
        text = "geometry received; the episode is over"
        if not s["gate_deliverable"]:
            text += f"\nthe deliverable is not valid: {s['diagnostic']}"
        if self.reveal_score:
            if s["scan_s"] is not None:
                text += (f"\nreward {s['reward']:.3f} · delivered scan {s['scan_s']:.1f} s ({s['lines']} lines) · fastest valid "
                         f"{s['scan_s_fastest']:.1f} s · safe geometry {s['scan_s_safe']:.1f} s · gates: resolution {s['gate_resolution']}, "
                         f"whole object {s['gate_coverage']}, no wrap {s['gate_no_wrap']} (wrap energy {s['wrap_fraction']:.1e}), "
                         f"measured {s['gate_measured']}, log {s['gate_log']}")
            else:
                text += f"\nreward {s['reward']:.3f}"
        return s["reward"], s, text


def card() -> dict:
    rows = [("reference", "pass")] + [(n, "fail") for n in controls.ADMISSION_CONTROLS] + [(n, "reported") for n in controls.REPORTED_CONTROLS]
    return dict(
        id=ENV_ID, title="The fastest geometry without wrap-around", kind="admitted",
        given="An object of hidden size, orientation and position; the scanner; 30 s of scan time for localisers",
        deliver="Geometry of a spin-echo scan: phase-encode axis, field of view, phase field of view, matrix",
        scored_by="Re-scan on the hidden object: whole object inside, no wrap-around, pixel as required; then scan time against the fastest valid geometry",
        summary="The first decision of an examination. A hidden test object lies near the isocentre of the 1.5 T scanner. "
                "The agent looks (a localiser costs scan time), then prescribes the geometry of a spin-echo scan whose "
                "time is proportional to the number of phase-encode lines. Which axis to phase-encode and how large the "
                "phase field of view must be depend on the object, so the right answer differs between episodes.",
        world=dict(phantom="QA phantom (an elliptical disc of gel with vials), 1.5 T",
                   sequences_useful=["spgr", "se"],
                   agent_side_formulas="scan time = overhead + dummy cycle + TR x phase-encode lines; the readout is "
                                       "oversampled and does not wrap; the phase-encode direction wraps unless the "
                                       "encoded field of view covers the object"),
        hidden=["size, orientation and position of the object", "its materials", "fields and system state",
                "noise level", "the 128-bit world seed"],
        told=["the target sequence and the required pixel size", "the instrument datasheet", "budget and request limits",
              "realised protocol, timing, limits and cost of every plan and scan", "the extent of the bright pixels of any stored image"],
        levels={lv: dict(summary=d["summary"], budget_s=list(d["budget_s"])) for lv, d in task.LEVELS.items()},
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"pe_dir": "y | x", "fov_mm": "<mm>", "phase_fov": "<fraction>", "matrix": "<even integer>",
                                                  "phase_oversampling": "<fraction, optional>"}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts the geometry", f"pixel <= the required size",
                           "the whole object lies inside the image (hidden extent, 0.5 mm slack)",
                           f"wrap-around energy of a noise-free re-scan <= {task.WRAP_LIMIT:g} of the image"],
                    error="T / T_opt - 1: scan time of the delivered geometry over the fastest valid one found by the verifier",
                    score=f"clip(ln(T_safe / T) / ln(T_safe / ({task.TOL} x T_opt)), 0, 1); T_safe = scan time of the geometry that is safe for any object",
                    passing=f"all gates and T <= {task.TOL} x T_opt (reward 1)"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's re-scan is noise-free",
    )
