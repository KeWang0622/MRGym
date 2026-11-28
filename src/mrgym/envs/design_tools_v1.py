"""The design tools as ``Scanner/RFPulse-v1`` offers them: the two tools of ``design_tools.py`` in fewer words, a
simulator that plays a design the way the family's scanner does, a short ``simulate`` answer, and the family's own
manual under the topic ``design``.

``design_tools.py`` is not changed by this module: it is the environment class of ``Scanner/RFPulse-v0``, whose
fingerprint covers its source. A family registers ``mrgym.envs.design_tools_v1:DesignFamilyEnvV1`` and declares

    task.DESIGN_TOOLS = dict(kinds=(...), chain=dict(model=..., guard=...))     the generators; how a sample is played
    task.DESIGN_MANUAL                                                          the text of ``scanner_info`` topic design

Sizes (``tests/test_rf_pulse_v1.py``): tools plus task text stay inside the largest ceiling of the other families,
and a ``simulate`` answer stays below 220 tokens.
"""
from __future__ import annotations

import dataclasses

from ..design import hardware, tool
from ..env import Observation
from ..tools import ToolSpec
from . import design_tools, family

EXAMPLE = {"kind": "minphase", "flip_deg": 30, "duration_ms": 0.6, "tbw": 4.08, "thickness_mm": 5, "transition": 0.3, "tol_in_deg": 1, "max_out_deg": 1}
WORDS = dict(
    kind="sinc: windowed sinc; minphase: the shortest pulse for a flip-angle mask",
    flip_deg="flip angle in the slice, deg", duration_ms="ms, a multiple of 0.01", tbw="gradient = tbw / (0.042577 x thickness_mm x duration_ms) mT/m",
    thickness_mm="slice thickness, mm", window="default hann", amplitude="factor on every sample; default 1",
    transition="W: slice |z| <= (1 - W) thickness / 2, outside from (1 + W) thickness / 2", tol_in_deg="flip angle in the slice within flip_deg +- this",
    max_out_deg="flip angle outside at most this", freq_hz="frequency offset of the pulse, Hz; default 0",
    rephase_fraction="rephasing lobe / (gradient x duration), negative: of the gradient's sign; default: the best on the ideal model")
DESIGN = ("Make an RF pulse from a generator and store it under a number. Returns duration, peak B1, energy, B1rms over the task's TR, "
          "gradient and the limit check. Parameters only: no samples. Free.")
SIMULATE = ("Flip-angle profile of a stored design on the model (hard sub-pulses, no relaxation), against the task's specification. "
            "Ideal unless you give what you measured. It does not see this scanner. Free.")


def _properties(kinds) -> dict:
    used = [k for k in tool.KINDS if k in kinds]
    props = {"kind": {"type": "string", "enum": used, "description": WORDS["kind"]}}
    for name in dict.fromkeys(n for k in used for n in tool.REQUIRED[k] + tool.OPTIONAL[k]):
        who = [k for k in used if name in tool.REQUIRED[k] + tool.OPTIONAL[k]]
        only = "" if len(who) == len(used) else f" ({', '.join(who)})"
        if name in tool.PARAMETERS:
            lo, hi = tool.range_of(who[0], name)
            props[name] = {"type": "number", "minimum": lo, "maximum": hi, "description": WORDS[name] + only}
        else:                                                            # the window of a sinc: the only choice among these generators
            props[name] = {"type": "string", "enum": list(tool.WINDOWS), "description": WORDS[name] + only}
    return props


def tool_specs(kinds) -> list[ToolSpec]:
    obj = lambda props, required: {"type": "object", "additionalProperties": False, "required": list(required), "properties": props}   # noqa: E731
    num = lambda lo, hi, what: {"type": "number", "minimum": lo, "maximum": hi, "description": what}   # noqa: E731
    return [
        ToolSpec("design", DESIGN, obj(_properties(kinds), ["kind"]), counted=True, examples=(EXAMPLE,)),
        ToolSpec("simulate", SIMULATE,
                 obj({"design": {"type": "integer", "minimum": 1, "maximum": tool.MAX_DESIGNS, "description": "number of a stored design"},
                      "b1_scale": num(0.3, 3.0, "actual / nominal B1; default 1"), "grad_delay_us": num(-200.0, 200.0, "slice gradient later than the RF by this; default 0"),
                      "off_resonance_hz": num(-2000.0, 2000.0, "default 0")}, ["design"]), examples=({"design": 1, "b1_scale": 0.9},))]


class DesignToolsV1(design_tools.DesignTools):
    def _with_design(self, tools, kinds):
        own = {t.name: t for t in tool_specs(kinds)}                    # the base class places its two tools (for the generators it has words for); they are replaced
        base = super()._with_design(tools, tuple(k for k in kinds if k in design_tools.KIND_WORDS))
        return [dataclasses.replace(own[t.name]) if t.name in own else t for t in base]

    def _tool_scanner_info(self, a):
        if a.get("topic") == design_tools.TOPIC and self._design_kinds():
            text = self.task.DESIGN_MANUAL
            return Observation(text=text, data={"markdown": text})
        return super()._tool_scanner_info(a)

    def _tool_simulate(self, a):
        d = self._stored_design(a["design"])
        pulse, _ = tool.build(d["call"], self._session.design_limits())
        chain = hardware.Chain(tx_scale=float(a.get("b1_scale", 1.0)), grad_delay_us=float(a.get("grad_delay_us", 0.0)),
                               df_hz=float(a.get("off_resonance_hz", 0.0)), **self.task.DESIGN_TOOLS["chain"])
        text, data = tool.simulate(pulse, chain, self._design_card().get("specification"), index=d["index"], brief=True)
        return Observation(text=text, data=dict(design=d["index"], **data))


class DesignFamilyEnvV1(DesignToolsV1, family.FamilyEnv):
    """``Scanner/<Family>-vN`` of a design family that plays designs as hard sub-pulses: ``FamilyEnv`` with these tools."""
