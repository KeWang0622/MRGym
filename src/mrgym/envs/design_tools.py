"""Two tools for design tasks, offered only by a family that declares them (``task.DESIGN_TOOLS``) and is registered
with the class ``DesignFamilyEnv`` of this module.

    design     free         make a pulse from a generator call and store it; returns its number and its record
    simulate   free         the flip-angle profile of a stored design on the model the agent is told about

A design is parameters: a generator and a handful of numbers, or at most 64 control points. No sample arrays and no
code (docs/DESIGN.md, rule 2). A stored design never changes; ``scan`` plays it on the scanner (``sequence: rfscan``,
``rf_design: <number>``), ``submit`` delivers its number.

A family declares ``task.DESIGN_TOOLS = dict(kinds=(...))`` (which generators it offers) and puts into the task card of
every instance ``design = dict(limits=dict(b1_peak_ut, b1rms_ut, tr_ms), specification=<spec or None>)``; its world
spec lists ``rfscan`` among the console's sequences and gives ``rf_hardware``; its registration names
``mrgym.envs.design_tools:DesignFamilyEnv`` as the entry point. Every other family keeps ``mrgym.envs.family:FamilyEnv``,
whose module is not changed by one statement: its tool list, schemas, answers and scoring source are what they were
(``python -m mrgym.compat.fingerprint --check``).

The simulator is the public model: the rotation of a piecewise-constant pulse under its gradient, without
relaxation. It is ideal unless the caller says otherwise; the four things a scanner does to a design
(``mrgym.design.hardware``) are arguments, so that a measured value can be put in. It never sees the hidden scanner.
"""
from __future__ import annotations

import dataclasses
import importlib.resources

from .. import interface as I
from ..design import hardware, tool
from ..env import Observation
from ..scanner import console, rfscan, session
from ..tools import ToolSpec
from . import family, scanner_env

TOPIC = "design"
EXAMPLES = dict(
    sinc={"kind": "sinc", "flip_deg": 30, "duration_ms": 2, "tbw": 4, "thickness_mm": 5},
    slr={"kind": "slr", "flip_deg": 30, "duration_ms": 2, "tbw": 4, "thickness_mm": 5, "tol_in_deg": 1, "max_out_deg": 1},
    sech={"kind": "sech", "b1_peak_ut": 12, "duration_ms": 8, "beta_rad_per_ms": 1.2, "mu": 5},
    points={"kind": "points", "amplitude_ut": [0, 2, 6, 2, 0], "duration_ms": 2, "bandwidth_hz": 2000, "thickness_mm": 5},
    verse={"kind": "verse", "of": 1, "b1_peak_ut": 10})
PLAN_WORDS = dict(rf_design="number of the stored design to play as the excitation",
                  rf_axis="slice gradient on: readout = image of the pulse's profile, slice = a 2D image")
KIND_WORDS = dict(
    sinc="windowed sinc", slr="Shinnar-Le Roux, equiripple in the flip angle", sech="hyperbolic-secant inversion",
    points="1 to 64 control points, interpolated", verse="variable-rate version of a stored design")


def manual() -> str:
    """The section of the instrument manual on designed pulses (the same text as section 8 of docs/scanner.md)."""
    md = importlib.resources.files("mrgym.scanner").joinpath("datasheet.md").read_text(encoding="utf-8")
    return md[md.index("## 8."):].split("\n", 1)[1].strip().replace("**", "")


def _properties(kinds) -> dict:
    used = [k for k in tool.KINDS if k in kinds]
    props = {"kind": {"type": "string", "enum": used, "description": "; ".join(f"{k}: {KIND_WORDS[k]}" for k in used)}}
    names = [n for k in used for n in tool.REQUIRED[k] + tool.OPTIONAL[k]]
    for name in dict.fromkeys(names):
        who = [k for k in used if name in tool.REQUIRED[k] + tool.OPTIONAL[k]]
        only = "" if len(who) == len(used) else f" ({', '.join(who)})"
        need = [k for k in who if name in tool.REQUIRED[k]]
        opt = "" if len(need) == len(who) else "; optional" if not need else f"; optional for {', '.join(k for k in who if k not in need)}"
        if name in tool.PARAMETERS:
            lo, hi, unit, what = tool.PARAMETERS[name]
            props[name] = {"type": "number", "minimum": lo, "maximum": hi, "description": f"{what}{only}{opt}"}
        elif name == "window":
            props[name] = {"type": "string", "enum": list(tool.WINDOWS), "description": "default hann (sinc)"}
        elif name == "interp":
            props[name] = {"type": "string", "enum": list(tool.INTERPS), "description": "between control points; default cubic (points)"}
        elif name == "amplitude_ut":
            props[name] = {"type": "array", "maxItems": 64, "items": {"type": "number", "minimum": -tool.POINT_RANGE_UT, "maximum": tool.POINT_RANGE_UT},
                           "description": "B1 at control points equally spaced from the start to the end of the pulse, uT (points)"}
        elif name == "phase_deg":
            props[name] = {"type": "array", "maxItems": 64, "items": {"type": "number", "minimum": -360, "maximum": 360},
                           "description": "phase of each control point; default 0 (points)"}
        elif name == "of":
            props[name] = {"type": "integer", "minimum": 1, "maximum": tool.MAX_DESIGNS, "description": "number of the stored design to rescale (verse)"}
    return props


def tool_specs(kinds, has_spec: bool) -> list[ToolSpec]:
    """The two tools for a family that offers the generators ``kinds``."""
    used = [k for k in tool.KINDS if k in kinds]
    obj = lambda props, required: {"type": "object", "additionalProperties": False, "required": list(required), "properties": props}   # noqa: E731
    verdict = " and the verdict against the task's specification" if has_spec else ""
    return [
        ToolSpec("design", "Make an RF pulse from a generator and its parameters and store it under a number; a stored design never changes. "
                 "Returns duration, peak B1, energy, B1rms over the task's TR, gradient, and the limit check. Parameters only: no samples. "
                 "Scan it with sequence rfscan, rf_design = its number. Free.",
                 obj(_properties(used), ["kind"]), counted=True, examples=(EXAMPLES[used[0] if "slr" not in used else "slr"],)),
        ToolSpec("simulate", "Flip-angle profile of a stored design on the public model (piecewise-constant pulse, no relaxation): "
                 f"figures of merit{verdict} and one row of flip angles. Ideal unless you give what you measured: the transmit "
                 "scale, the amplifier's saturation level, the gradient delay, an off-resonance. It does not see this scanner. Free.",
                 obj({"design": {"type": "integer", "minimum": 1, "maximum": tool.MAX_DESIGNS, "description": "number of a stored design"},
                      "b1_scale": {"type": "number", "minimum": 0.3, "maximum": 3.0, "description": "actual / nominal B1; default 1"},
                      "amp_sat_ut": {"type": "number", "minimum": 1.0, "maximum": 200.0,
                                     "description": "saturation level A of the amplifier law a / (1 + (a / A)^6)^(1/6), uT; default: linear"},
                      "grad_delay_us": {"type": "number", "minimum": -200.0, "maximum": 200.0, "description": "slice gradient later than the RF by this; default 0"},
                      "off_resonance_hz": {"type": "number", "minimum": -2000.0, "maximum": 2000.0, "description": "default 0"}},
                     ["design"]), examples=({"design": 1, "b1_scale": 0.9},)),
    ]


class DesignTools:
    """Mixin of an environment class (before ``ScannerEnv`` in the bases): adds the two tools when the family declares
    them, the topic ``design`` to ``scanner_info`` and the stored designs to ``status``."""

    def _design_kinds(self):
        return tuple((getattr(getattr(self, "task", None), "DESIGN_TOOLS", None) or {}).get("kinds", ()))

    def _design_card(self) -> dict:
        return ((self._cfg or {}).get("task") or {}).get("design") or {}

    def _tool_specs(self):
        """The tool list of a design family: the scanner tools as every family has them (built for the six sequences
        every console offers), with ``rfscan`` and its two parameters added to ``plan`` and ``scan``, the topic
        ``design`` added to ``scanner_info``, and ``design`` and ``simulate`` before the terminal tool."""
        kinds = self._design_kinds()
        if not kinds:
            return super()._tool_specs()
        if self._session is None:                                        # before the first reset: no exam, no specification
            return self._with_design(scanner_env.scanner_tools(self.obs_mode, calc=self._calc) + [self.terminal_tool()], kinds)
        if self._specs is None:
            exam = {k: v for k, v in self._session.exam.items() if k not in ("sequences", "designs")}
            self._specs = self._with_design(scanner_env.scanner_tools(self.obs_mode, exam, [r["id"] for r in self._regions], self._calc)
                                            + [self.terminal_tool()], kinds)
        return self._specs

    def _with_design(self, tools, kinds):
        out = []
        for t in tools:
            props = dict(t.parameters.get("properties", {}))
            if t.name == "scanner_info":                                 # one more topic: the manual of designed pulses
                props["topic"] = dict(props["topic"], enum=list(props["topic"]["enum"]) + [TOPIC])
                t = dataclasses.replace(t, parameters=dict(t.parameters, properties=props))
            elif t.protocol:                                             # plan and scan: the scan that plays a stored design
                props["sequence"] = dict(props["sequence"], enum=list(props["sequence"]["enum"]) + [rfscan.NAME])   # described by rf_design
                for name in ("rf_design", "rf_axis"):                    # as every protocol parameter: one short line, no minimum
                    lo, hi, kind, default = rfscan.PARAMETERS[name]
                    prop = {"type": "integer"} if kind == "i" else {"type": "string", "enum": list(lo)}
                    if t.name == "plan":
                        prop["description"] = PLAN_WORDS[name] + f"; only {rfscan.NAME}; " + ("required" if default is None else f"default {default}")
                    props[name] = prop
                t = dataclasses.replace(t, parameters=dict(t.parameters, properties=props))
            if t.terminal:
                out += tool_specs(kinds, bool(self._design_card().get("specification")))
            out.append(t)
        return out

    def _tool_scanner_info(self, a):
        if a.get("topic") == TOPIC and self._design_kinds():
            text = manual()
            return Observation(text=text, data={"markdown": text})
        if a.get("parameter") in ("rf_design", "rf_axis") and self._design_kinds():
            name = a["parameter"]
            lo, hi, kind, default = rfscan.PARAMETERS[name]
            rng = f"{lo} to {hi}" if kind == "i" else " / ".join(lo)
            text = f"{name}: {console.PARAMETER_UNITS[name]}\n  {rfscan.NAME}: range {rng}, " + ("required" if default is None else f"default {default}")
            return Observation(text=text, data={"parameter": {name: dict(range=[lo, hi] if kind == "i" else list(lo), default=default)}})
        obs = super()._tool_scanner_info(a)
        if self._design_kinds() and not obs.is_error and (a.get("topic") or "overview") == "overview" and a.get("parameter") is None:
            obs.text += f"\n  {TOPIC}     designed pulses: the design and simulate tools, the scan rfscan, what this scanner does to a design"
        return obs

    def _tool_status(self, a):
        obs = super()._tool_status(a)
        table = self._session.exam.get("designs") if self._design_kinds() else None
        if table is None:
            return obs
        lines = [f"stored designs: {len(table)}"] + [
            f"  design {k} · {tool.call_text(d['call'])} · {d['facts']['duration_ms']:g} ms · "
            + ("limits met" if d["facts"]["limits"]["ok"] else "VIOLATES " + d["facts"]["limits"]["violations"][0]["code"]) for k, d in sorted(table.items())]
        head, _, tail = obs.text.rpartition("\n")                         # before the line that says how to deliver
        obs.text = "\n".join([head] + lines + [tail])
        obs.data = dict(obs.data, designs=[dict(index=k, call=d["call"], facts=d["facts"]) for k, d in sorted(table.items())])
        return obs

    def _tool_design(self, a):
        kinds = self._design_kinds()
        if a.get("kind") not in kinds:
            return self._error([I.problem("invalid_value", "kind", f"this task offers the generators {', '.join(kinds)}", "vocabulary",
                                          allowed={"values": list(kinds)})])
        d = self._scanner.design({k: v for k, v in a.items() if v is not None})
        return Observation(text=tool.summary(d["design"], d["call"], d["facts"]), data=d)

    def _stored_design(self, index):
        table = self._session.exam.get("designs") or {}
        if index not in table:
            raise session.InvalidRequest(f"no stored design {index}" + (f"; stored: {', '.join(str(k) for k in sorted(table))}" if table
                                                                         else "; make one with the design tool first"),
                                         parameter="design", not_found=True, allowed=sorted(table))
        return table[index]

    def _tool_simulate(self, a):
        d = self._stored_design(a["design"])
        pulse, _ = tool.build(d["call"], self._session.design_limits())
        chain = hardware.Chain(tx_scale=float(a.get("b1_scale", 1.0)), amp_sat_ut=a.get("amp_sat_ut"), grad_delay_us=float(a.get("grad_delay_us", 0.0)),
                               df_hz=float(a.get("off_resonance_hz", 0.0)))
        text, data = tool.simulate(pulse, chain, self._design_card().get("specification"), index=d["index"])
        return Observation(text=text, data=dict(design=d["index"], **data))


class DesignFamilyEnv(DesignTools, family.FamilyEnv):
    """``Scanner/<Family>-vN`` of a design family: ``FamilyEnv`` with the design tools."""

