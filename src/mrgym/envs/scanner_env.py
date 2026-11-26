"""What every environment on the scanner shares: the session, the scanner tools and their text.

A subclass says what is asked (``make_instance``, ``task_text``), what is delivered (``terminal_tool``,
``check_deliverable``) and how it is scored (``finish``). It never touches physics: the world, pricing, noise, log
and replay are ``mrgym.scanner``.

    scanner_info   free         one topic of the datasheet per call: the card in short, one sequence, one mechanism, one parameter
    status         free         every limit and what is left of it, the stored scans with their header lines, how to deliver
    plan           free         dry run on the console: cost and fit, and what the request does not say (logged)
    scan           scan time    run a protocol; index, cost, echo time, region read-outs; images stored under a scan index
    read_regions   free         read-outs of marked regions or other circles on a stored scan
    view           free         one stored image as a digit map (text) or PNG (vision), with its geometry
    profile        free         the numbers along one line through a stored image
    calc           free         a calculator in Python syntax; its names stay for the episode

"Free" means no scan time. Every call counts once against the tool-call limit unless it is rejected, whatever the
tool: a call refused for what it asks (error class ``task``) is a tool call, a call refused for how it is written
(``shape``, ``vocabulary``) is a rejected call (``mrgym.interface.counts_as_tool_call``).

What the text of an answer holds. An episode is a token stream that a trainer has to hold in one piece, and most of
it is the environment's own text, so an answer says what a decision needs and nothing twice: ``plan`` and ``scan`` do
not repeat the request or the defaults (``verbose``, ``status``), the datasheet comes one topic at a time, tables
carry no padding, and the status line under an answer is the call count plus whatever changed. ``Observation.data``
is not shortened: it holds everything at full precision, for code, verifiers and scripted policies.

The protocol parameters are top-level arguments of ``plan`` and ``scan`` (``sequence``, ``tr_ms``, ...). Regions are
named in two parameters with one type each: ``regions`` (ids of the task's marked regions) and ``circles`` (free
circles). ``te_ms`` may be ``"min"``; a number below the console's minimum is raised to it and reported in
``adjusted`` — the only value the interface ever changes. ``view`` shows an image and its geometry; a measurement of
what is in the image (an edge, a size) is the agent's own work, with ``profile`` and ``read_regions``.
"""
from __future__ import annotations

import copy
import importlib.resources
import json
import math
import os

from .. import calc as calc_mod
from .. import interface as I
from .. import keys
from .. import render as R
from ..env import Env, Observation
from ..scanner import console, session
from ..tools import ToolSpec

_CIRCLE = {"type": "object", "additionalProperties": False, "required": ["y_mm", "x_mm", "radius_mm"],
           "properties": {"id": {"type": "string"}, "y_mm": {"type": "number"}, "x_mm": {"type": "number"}, "radius_mm": {"type": "number"}}}
_CIRCLES = {"type": "array", "maxItems": session.MAX_ROIS, "items": _CIRCLE,
            "description": f"other circles to read out (mm from the isocentre); at most {session.MAX_ROIS} read-outs per call"}
_SCAN_INDEX = {"type": "integer", "description": "a stored scan; default: the latest"}
_IMAGE_INDEX = {"type": "integer", "description": "image of the scan (afi: 2, spgr: one per echo); default 0"}
_KIND = {"type": "string", "enum": ["magnitude", "phase"], "description": "default magnitude"}
_STORED = {"scan": "scan_index", "image": "image_index"}                  # spellings accepted for the two indices
SEQUENCE_TOPICS = tuple(console.SEQUENCES)
PHYSICS_TOPICS = ("encoding", "slice_profile", "fat", "motion", "hidden")
TOPICS = ("overview", "task", "presets", "sequences") + SEQUENCE_TOPICS + ("geometry", "cost", "outputs", "physics") + PHYSICS_TOPICS
MAX_TABLE_ROWS = 48                                                       # read-out rows printed; the rest is in ``data``
VIEW_MAX = 48                                                             # the finest digit map: 48 x 48 digits are about 2,400 tokens
VIEW_SIZE = 16                                                            # digit-map blocks along the longer side when none is asked for
LIMIT_SHOWN = 70.0                                                        # plan prints the gradient use from this percentage of its limit on
NOT_PROTOCOL = ("regions", "circles", "verbose")                          # arguments of plan / scan that are not console parameters
# One valid call per sequence, printed by scanner_info(topic=<sequence>).
SEQUENCE_EXAMPLES = dict(
    spgr={"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "matrix": 96},
    afi={"sequence": "afi", "tr_ms": 20, "flip_deg": 60, "matrix": 64},
    se={"sequence": "se", "tr_ms": 500, "te_ms": "min", "matrix": 128},
    ir={"sequence": "ir", "tr_ms": 2000, "ti_ms": 500, "matrix": 96},
    bssfp={"sequence": "bssfp", "tr_ms": 8, "flip_deg": 40, "matrix": 128},
    epi={"sequence": "epi", "tr_ms": 2000, "matrix": 64})
PLAN_EXAMPLE = {"sequence": "se", "tr_ms": 500, "te_ms": "min", "fov_mm": 240, "matrix": 120, "phase_fov": 0.75}
SCAN_EXAMPLE = {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "fov_mm": 288, "matrix": 96}
# What a parameter is, in the fewest words (the schema of ``plan``). Range, sequences and default are generated from
# the console's tables; the full meaning is behind scanner_info (topic = geometry, or the sequence).
WHAT = dict(
    tr_ms="repetition time", te_ms="echo time", ti_ms="inversion time", flip_deg="nominal flip angle", fov_mm="readout field of view",
    phase_fov="phase FOV / fov_mm", matrix="readout pixels, even", slice_mm="", bandwidth_hz_per_px="",
    pe_dir="phase-encode axis: y = rows, x = columns", ro_polarity="readout direction", partial_fourier="fraction of the phase lines acquired",
    phase_oversampling="extra phase FOV, as a fraction", averages="", centre_y_mm="FOV centre from the isocentre",
    centre_x_mm="FOV centre from the isocentre", rotation_deg="in-plane rotation of the FOV, rows from +y towards +x", rf_pulse="excitation pulse",
    excitation="nonselective = 3D", fat_sat="fat saturation", fatsat_offset_hz="frequency offset of the fat saturation",
    spoiler_mt_ms_per_m="spoiler moment per TR", rf_phase_increment_deg="RF spoiling increment", n_echoes="", echo_spacing_ms="",
    tr_ratio="TR2 / TR1", phase_cycle_deg="RF phase advance per TR", pe_polarity="sign of the phase blips")
SEQUENCE_WORDS = dict(spgr="spoiled gradient echo", afi="flip-angle map (dual TR)", se="spin echo", ir="inversion recovery", bssfp="balanced SSFP",
                      epi="gradient-echo EPI")


def datasheet() -> str:
    """The instrument datasheet shipped with the package (the same text as docs/scanner.md)."""
    return importlib.resources.files("mrgym.scanner").joinpath("datasheet.md").read_text(encoding="utf-8")


# Paragraphs of section 5 of the datasheet by the words they start with -> the topic that serves them.
_LEADS = {"Encoding": "encoding", "Slice selection": "slice_profile", "Gradient echoes": "gradient_echo", "Spin echoes": "spin_echo",
          "Fat saturation": "fat", "Motion": "motion"}
_ECHO = dict(spgr="gradient_echo", afi="gradient_echo", epi="gradient_echo", se="spin_echo", ir="spin_echo")


def datasheet_parts() -> dict:
    """The datasheet cut into the pieces ``scanner_info`` serves: ``cost`` (section 4), ``outputs`` (6), ``hidden`` (7),
    ``notation`` and one piece per paragraph of section 5 (``encoding``, ``slice_profile``, ``gradient_echo``,
    ``spin_echo``, ``fat``, ``motion`` and one per sequence). Nothing of sections 4 to 7 is in no piece."""
    md = datasheet()
    cut = lambda a, b: md[md.index(a):md.index(b) if b else len(md)].split("\n", 1)[1].strip()       # noqa: E731
    parts = {k: v.replace("**", "") for k, v in dict(cost=cut("## 4.", "## 5."), outputs=cut("## 6.", "## 7."), hidden=cut("## 7.", None)).items()}
    key = None
    for para in cut("## 5.", "## 6.").split("\n\n"):
        if para.startswith("**"):
            lead = para[2:para.index("**", 2)]
            key = next((s for s in console.SEQUENCES if lead.startswith(f"`{s}`")), None) or next((v for k, v in _LEADS.items() if lead.startswith(k)), None)
            if key is None:
                raise ValueError(f"datasheet: no topic for the paragraph {lead!r}")
            parts[key] = para.replace("**", "")
        elif key is None:
            parts["notation"] = para
        else:
            parts[key] += "\n\n" + para
    return parts


def _object(properties: dict, required=()) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(required), "properties": properties}


def _range_text(name: str, used: list) -> str:
    """The fixed range of a parameter over the sequences that have it, in a few characters; '' for a time whose
    minimum depends on the rest of the protocol, a switch or an enum."""
    specs = console.SEQUENCES
    if specs[used[0]][name][2] not in "fi" or name in console.PROTOCOL_DEPENDENT:
        return ""
    by = {}
    for q in used:
        by.setdefault((specs[q][name][0], specs[q][name][1]), []).append(q)
    (main, _), *rest = sorted(by.items(), key=lambda kv: -len(kv[1]))
    show = lambda r: f"+-{r[1]:g}" if r[0] == -r[1] else f"{r[0]:g}-{r[1]:g}"      # noqa: E731
    return show(main) + ("".join(f" ({', '.join(qs)}: {show(r)})" for r, qs in rest))


def protocol_properties(exam: dict | None = None, bare: bool = False) -> dict:
    """The protocol parameters as schema properties, generated from the console's tables (``console.SEQUENCES``) so
    that they cannot drift from what ``plan`` accepts. One type per parameter; the only unions are the two times
    that accept ``"min"``. A description says what the parameter is, its fixed range, the sequences that have it and
    its default as a value a call can carry, in one line; ranges are stated there and checked by the console, for
    ``plan``, ``scan`` and a delivered protocol alike. ``bare``: name, type and the values of an enum only (the
    ``scan`` tool and a delivered protocol take the parameters ``plan`` documents)."""
    full = console.protocol_schema(exam, brief=True)["properties"]
    seqs = sorted(console.SEQUENCES)
    out = {}
    for name, schema in full.items():
        prop = {k: v for k, v in schema.items() if k != "description"}
        if bare:
            out[name] = prop
            continue
        if name == "sequence":
            out[name] = dict(prop, description=", ".join(f"{k} = {SEQUENCE_WORDS[k]}" for k in console.SEQUENCES))
            continue
        used = [q for q in seqs if name in console.SEQUENCES[q]]
        fixed = [q for q in seqs if name in console.FIXED.get(q, {})]
        others = [q for q in seqs if q not in used and q not in fixed]
        parts = [", ".join(x for x in (WHAT.get(name, ""), _range_text(name, used)) if x)]
        if others:
            parts.append(("not " + ", ".join(others)) if len(others) < len(used) else ("only " + ", ".join(used)))
        by_default = {}
        for q in used:
            by_default.setdefault(console.default_text(q, name, exam), []).append(q)
        required = by_default.pop(None, [])
        if required and by_default:
            parts.append(f"required for {', '.join(required)}")
        if by_default:
            (main, most), *rest = sorted(by_default.items(), key=lambda kv: -len(kv[1]))
            text = f"default {main}" + ("".join(f" ({', '.join(qs)}: {d})" for d, qs in rest) if not required or rest else f" for {', '.join(most)}")
            if console.accepts_min(name):
                text += " = the shortest" + (" (a smaller number is raised to it)" if name == "te_ms" else "")
            parts.append(text)
        out[name] = dict(prop, description="; ".join(x for x in parts if x) or "required")
    return out


def protocol_object(exam: dict | None = None) -> dict:
    """The schema of a protocol delivered as one argument: the parameters of ``plan`` by name and type (``plan``
    documents them; their ranges are checked as for ``plan``)."""
    return _object(protocol_properties(exam, bare=True), ["sequence", "tr_ms"])


def scanner_tools(obs_mode: str = "text", exam: dict | None = None, region_ids=(), calc: bool = True) -> list[ToolSpec]:
    """The scanner tools, the same in every environment. ``exam``: the session's console card (the default field of
    view is then printed as a number). ``region_ids``: ids of the task's marked regions; ``regions`` is offered only
    when there are any.

    A description says what the tool is for and what it costs. How to use the instrument (parameter tables, signal
    models, the cost rule) is behind ``scanner_info``."""
    view_kind = "a PNG with the task regions outlined" if obs_mode == "vision" else "a digit map (block means, 0-9)"
    required = ["sequence", "tr_ms"]
    readouts = {}
    if region_ids:
        readouts["regions"] = {"type": "array", "maxItems": session.MAX_ROIS, "items": {"type": "string"},
                               "description": f"marked regions to read out ({', '.join(region_ids)}); default [\"all\"]; [] none"}
    readouts["circles"] = _CIRCLES
    circle = {"y_mm": 0, "x_mm": 0, "radius_mm": 10}
    verbose = {"verbose": {"type": "boolean", "description": "also print every realised parameter and the timing; default false"}}
    tools = [
        ToolSpec("scanner_info", "Datasheet of the instrument, one topic per call; overview lists the topics. Free.",
                 _object({"topic": {"type": "string", "enum": list(TOPICS), "description": "default overview"},
                          "parameter": {"type": "string", "description": "a protocol parameter: range, default, meaning"}}),
                 examples=({"topic": "se"},)),
        ToolSpec("status", "Every limit with what is used and left, the stored scans, how to deliver. Free.",
                 _object({}), examples=({},)),
        ToolSpec("plan", "Dry run of a protocol: its scan seconds as a scan now and what is left after it, the minimum TE / TR, the console's "
                 "header line; or a refusal with the nearest valid value. No signal. Free.",
                 _object(dict(protocol_properties(exam), **verbose), required), counted=True, protocol=True, aliases=dict(console.ALIASES),
                 examples=(PLAN_EXAMPLE,)),
        ToolSpec("scan", "Run a protocol (the parameters of plan) and store its images under a scan_index. Costs its scan time; a refused "
                 "scan costs none. Returns per region and image: pixels, magnitude and phase of the complex mean (no noise floor; a phase "
                 "alone gives no sign: scanner_info outputs), mean and SD of the magnitude.",
                 _object(dict(protocol_properties(exam, bare=True), **readouts), required), cost="scan_time", counted=True, protocol=True,
                 aliases=dict(console.ALIASES), examples=(SCAN_EXAMPLE,), see="plan"),
        ToolSpec("read_regions", "Read-outs of a stored scan, as in scan: marked regions and / or other circles (background, for the noise). Free.",
                 _object(dict(scan_index=_SCAN_INDEX, **readouts)), aliases=dict(_STORED),
                 examples=({"scan_index": 0, "circles": [circle]},)),
        ToolSpec("view", f"A stored image as {view_kind}, with its geometry. A picture: profile and read_regions give numbers. Free.",
                 _object({"scan_index": _SCAN_INDEX, "image_index": _IMAGE_INDEX, "kind": _KIND,
                          "size": {"type": "integer", "minimum": 8, "maximum": VIEW_MAX, "description": f"blocks along the longer side; default {VIEW_SIZE}"}}),
                 aliases=dict(_STORED), examples=({"scan_index": 0},)),
        ToolSpec("profile", "Values along a line through a stored image, as position (mm) and value: for an edge, a shift, a width. "
                 "axis y: down the image at x = position_mm; axis x: across it at y = position_mm. Pixels across width_mm, and along the "
                 "line when there are more than samples, are combined by reduce (max over the full width: a projection). Free.",
                 _object({"scan_index": _SCAN_INDEX,
                          "axis": {"type": "string", "enum": ["y", "x"], "description": "direction of the line"},
                          "image_index": _IMAGE_INDEX, "kind": _KIND,
                          "position_mm": {"type": "number", "description": "the line's place on the other axis; default centre"},
                          "start_mm": {"type": "number", "description": "default: one edge of the image"},
                          "stop_mm": {"type": "number", "description": "default: the other edge"},
                          "width_mm": {"type": "number", "description": "default one pixel"},
                          "samples": {"type": "integer", "description": f"default one per pixel up to {session.MAX_PROFILE_SAMPLES}, the most per call"},
                          "reduce": {"type": "string", "enum": ["mean", "max"], "description": "default mean"}},
                         ["axis"]),
                 aliases=dict(_STORED), examples=({"scan_index": 0, "axis": "y", "position_mm": 0, "width_mm": 10},)),
    ]
    if calc:
        tools.append(ToolSpec(
            "calc", "Calculator in Python syntax; it knows nothing about the scanner. name = value (kept between calls), for / if / "
            "def / try, % formats. a + b joins lists, [0] * n repeats, x * 2 scales a list. math / numpy names (exp arctan2 mean "
            "polyfit interp ...); brentq(f, a, b), minimize(f, x0), curve_fit(f, x, y, p0). Stops after "
            f"{calc_mod.MAX_STEPS // 1_000_000} million steps. Shows print(...) and bare expressions. Free.",
            _object({"expressions": {"type": "string", "description": "the statements, as one text"}}, ["expressions"]),
            lines=("expressions",), examples=({"expressions": "t = 3 + 0.5 * 97; t / 60"},)))
    return tools


class ScannerEnv(Env):
    """Base class of the ``Scanner/...`` environments."""

    def __init__(self, spec, log_dir: str | None = None, calc: bool = True, **kw):
        super().__init__(spec, **kw)
        self._log_dir, self._calc = log_dir, bool(calc)
        self._cfg = self._session = self._scanner = self._specs = None
        self._regions = []
        self._shown = None                                               # what the last status line printed (scan seconds used, rejected calls)
        self._names = {}                                                 # what calc has assigned in this episode

    # ------------------------------------------------------------------ hooks of a subclass
    def make_instance(self, seed_hex: str, task_id: str) -> dict:      # -> episode config for scanner.session
        raise NotImplementedError

    def task_text(self) -> str:                                         # -> the statement the agent reads at reset
        raise NotImplementedError

    def terminal_tool(self) -> ToolSpec:
        raise NotImplementedError

    def check_deliverable(self, arguments: dict):
        """None when the deliverable can be evaluated; otherwise why not. The episode then continues (the refusal is
        an ``invalid_deliverable`` error). Return a sentence, or ``{"message", "violations", "protocol"}`` with the
        console's violations of the protocol the deliverable stands for: the refusal then names the parameter, the
        nearest valid value and a fix."""
        return None

    def finish(self, arguments: dict, replay: dict) -> tuple[float, dict, str]:   # -> (reward, score dict, text for the agent)
        raise NotImplementedError

    # ------------------------------------------------------------------ Env hooks
    def _reset(self, seed_hex):
        task_id = keys.task_id_for(self.spec, seed_hex, self.split)  # trainer-side label; never shown to the agent
        self._cfg = dict(self.make_instance(seed_hex, task_id))
        # one limit on calls, the one the agent is told: the scanner's own request counter can never bind first
        self._cfg["max_requests"] = max(int(self._cfg.get("max_requests", 64)), self.max_steps)
        if self._session:
            self._session.close()
        log_dir = None
        if self._log_dir:                                                # one directory per episode: a second play of the same
            base, n = os.path.join(self._log_dir, self._episode["id"]), 1   # instance must not inherit the first one's scans
            while log_dir is None:
                try:
                    os.makedirs(base if n == 1 else f"{base}.{n}")       # atomic, so parallel workers get distinct directories
                    log_dir = base if n == 1 else f"{base}.{n}"
                except FileExistsError:
                    n += 1
        self._session = session.ScannerSession(self._cfg, log_dir=log_dir, handle=self._episode["id"])
        self._scanner = session.LocalScanner(self._session)
        self._regions = self._cfg.get("task", {}).get("regions", [])
        self._specs = self._shown = None
        self._names = {}
        c = self._cfg
        return Observation(text=self.task_text().rstrip(), data=dict(task=c.get("task", {}), budget_s=c["budget_s"], level=self.level, env_id=self.spec.id))

    def _info(self):
        """``task_id``: the trainer-side label of the instance. Every instance of a level is served: the gate gives one
        verdict per (environment, level), ``info["gate"]`` where a family reports it, never one per instance."""
        return dict(task_id=self._cfg["task_id"])

    def _tool_specs(self):
        if self._session is None:                                        # before the first reset: the tools without an exam
            return scanner_tools(self.obs_mode, calc=self._calc) + [self.terminal_tool()]
        if self._specs is None:
            self._specs = scanner_tools(self.obs_mode, self._session.exam, [r["id"] for r in self._regions], self._calc) + [self.terminal_tool()]
        return self._specs

    def limits_text(self) -> str:
        """The Limits block that ends the task text: every number of ``limits()``, once, in the fewest words."""
        terminal = self.terminal_tool().name
        parts = self._limit_sentences() + self._call_limit_sentences(terminal)
        if "max_reply_tokens" in self._harness:
            parts.append(f"Reply length: {self._harness['max_reply_tokens']} tokens.")
        parts.append(f"The episode ends at `{terminal}` or at a limit; without `{terminal}` the score is 0.")
        return "Limits. " + " ".join(parts)

    def _limits(self):
        return dict(scan_time_budget_s=float(self._cfg["budget_s"]))

    def _limit_sentences(self):
        return [f"Scan time: {self._cfg['budget_s']:g} s (only `scan` spends it)."]

    def _spent(self):
        return self._session.used if self._session else 0.0

    def _status(self):
        """The status dict of every observation. ``scanner_requests`` / ``max_scanner_requests`` are the scanner's own
        counter of logged requests, kept for recorders and viewers; the limit is never below the tool-call limit, so
        it ends no episode and no text names it."""
        b = self._session.budget()
        return dict(scan_time_used_s=b["used_s"], scan_time_budget_s=b["budget_s"], scan_time_left_s=b["remaining_s"], scans=b["scans"],
                    scanner_requests=b["requests"], max_scanner_requests=b["max_requests"])

    def _status_parts(self, s):
        """The parts of the one short line under every result: the tool calls used of the limit, always; the scan time
        left and the rejected calls only when they changed since the line before (and at reset). The numbers are
        those of ``limits()``; the ``status`` tool prints all of them at any time."""
        shown = self._shown or {}
        parts = []
        if shown.get("used") != s["scan_time_used_s"]:
            parts.append(f"scan time left {s['scan_time_left_s']:.2f} of {s['scan_time_budget_s']:g} s")
        parts.append(f"calls {s['tool_calls']}/{s['max_tool_calls']}")
        if shown.get("rejected") != s["rejected_calls"]:
            parts.append(f"rejected {s['rejected_calls']}/{s['max_rejected_calls']}" + (" (nothing charged)" if shown and s["rejected_calls"] else ""))
        if s["tool_calls_left"] <= 3 and not self._episode["done"]:
            parts.append(f"{self.terminal_tool().name} now or the episode scores 0")
        self._shown = dict(used=s["scan_time_used_s"], rejected=s["rejected_calls"])
        return parts

    def _charged_line(self, counted):
        return None                                                      # the status line says what a refusal cost

    def _check(self, name, a):
        """What the console says about the protocol of a ``plan`` / ``scan`` before anything runs: missing parameters,
        values outside the range of the sequence, an odd matrix. These are refusals of the call, not of the scanner."""
        spec = next(t for t in self._tool_specs() if t.name == name)
        if not spec.protocol:
            return []
        protocol = self._protocol(a)
        seq, te = protocol.get("sequence"), protocol.get("te_ms")
        if seq in console.SEQUENCES and "te_ms" in console.SEQUENCES[seq] and I.T._is_number(te) and 0 <= te < console.SEQUENCES[seq]["te_ms"][0]:
            protocol = dict(protocol, te_ms=console.MIN)                 # a short echo time is raised, never refused (``_adjust``)
        return I.console_problems(console.validate(protocol, self._session.exam)[2], protocol, self._session.exam)

    def _call(self, name, a):
        try:
            if name == self.terminal_tool().name:
                return self._terminal(a)
            return getattr(self, "_tool_" + name)(a)
        except session.ScannerError as e:
            return self._refusal(e, a)

    def _error(self, problems):
        data = I.error_data(problems)
        return Observation(is_error=True, data=data, text=I.error_text(data))

    def _refusal(self, e: session.ScannerError, a: dict):
        """A refusal of the scanner session in the error vocabulary."""
        x = e.extra
        if x.get("violations"):
            return self._error(I.console_problems(x["violations"], self._protocol(a), self._session.exam))
        if isinstance(e, session.BudgetExhausted):
            return self._error([I.problem("budget", None, f"this scan needs {x.get('cost_s', 0):.2f} s and {x.get('remaining_s', 0):.2f} s are left; "
                                          "any accepted protocol that fits can be scanned", code="SCAN_TIME", requested=x.get("cost_s"),
                                          allowed={"max": x.get("remaining_s")})])
        if isinstance(e, session.RequestLimit):
            return self._error([I.problem("budget", None, e.message, code="REQUEST_LIMIT")])
        name = x.get("parameter")
        if x.get("not_found"):
            return self._error([I.problem("not_found", name, e.message, allowed={"values": x.get("allowed")},
                                          nearest_valid=x["allowed"][-1] if x.get("allowed") else None,
                                          fix={name: x["allowed"][-1]} if x.get("allowed") and name else None)])
        near = x.get("nearest_valid")
        return self._error([I.problem("invalid_value", name, e.message, "task", requested=x.get("requested"),
                                      allowed={"min": x["low"], "max": x["high"]} if "low" in x else ({"values": x["allowed"]} if x.get("allowed") else None),
                                      nearest_valid=near, fix=I._fix_at(name, near) if name and near is not None else None)])

    # ------------------------------------------------------------------ trainer side
    def use_session(self, sess) -> None:
        """Replace the scanner session of the running episode: the service binds its durable session (log and head on
        disk), the verifier a recorded one. The deliverable is then scored by the same code as in process."""
        if self._session and self._session is not sess:
            self._session.close()
        self._session, self._scanner = sess, session.LocalScanner(sess)

    def episode_config(self) -> dict:
        """The hidden episode config (holds the seed). For whoever runs the episode: serving the same episode over
        HTTP (``mrgym serve``), or verifying a log. Never show it to the agent. A copy: changing it changes nothing."""
        return copy.deepcopy(dict(self._cfg))

    def scanner_log(self) -> list:
        """The scanner's hash-chained log of this episode (a copy)."""
        return copy.deepcopy(self._session.events)

    def stored_scans(self) -> dict:
        """Trainer side: the scans of this episode, {scan index: {"magnitude", "phase" (float32 [n_images, ny, nx]),
        "realised", "fov_yx_mm"}}. What a viewer or a recorder needs."""
        return dict(self._scanner.scans)

    def hidden_world(self) -> dict:
        """Trainer side: the hidden world of this episode (arrays on the master grid, tissue values, fields, system
        state). For figures, recorders and verifiers; never for the agent."""
        return self._session.world

    # ------------------------------------------------------------------ arguments
    @staticmethod
    def _protocol(a: dict) -> dict:
        return {k: v for k, v in a.items() if k not in NOT_PROTOCOL}

    def _adjust(self, protocol: dict):
        """The one adjustment the interface makes: a numeric ``te_ms`` below the console's minimum for this protocol is
        raised to that minimum. Returns (protocol to run, adjusted list). Raising the echo time never raises the cost
        of a scan: the repetition time is not touched (if the new echo time makes it too short the call is refused)."""
        te = protocol.get("te_ms")
        if isinstance(te, bool) or not isinstance(te, (int, float)):
            return protocol, []
        probe = console.plan(dict(protocol, te_ms=console.MIN), self._session.exam)
        te_min = (probe.get("timing") or {}).get("te_min_ms")
        if te_min is None or te >= te_min - 1e-9:
            return protocol, []
        return dict(protocol, te_ms=te_min), [dict(parameter="te_ms", requested=te, realised=te_min,
                                                   reason="console minimum for this matrix, field of view and bandwidth")]

    @staticmethod
    def _adjust_notes(adjusted):
        return [f"note: {x['parameter']} {x['requested']:g} -> {x['realised']:g} {x['parameter'].rsplit('_', 1)[-1]} ({x['reason']})" for x in adjusted]

    def _rois(self, a: dict) -> list:
        """The circles to read out: the selected marked regions of the task (default: all) and the free circles."""
        marked = {r["id"]: r for r in self._regions}
        ids, circles = a.get("regions"), a.get("circles") or []
        if ids is None:                                                  # nothing was said about the marked regions: all of them, except
            named = {c.get("id") for c in circles}                       # where a circle of the call carries the id (that circle is read),
            ids = [i for i in marked if i not in named]                  # and none when the circles alone fill the call
            if len(ids) + len(circles) > session.MAX_ROIS >= len(circles):
                ids = []
        elif I.ALL_REGIONS in ids:
            ids = list(marked)
        unknown = [i for i in ids if i not in marked]
        if unknown:
            guess = I.did_you_mean(unknown[0], list(marked))
            fixed = [guess if i == unknown[0] and guess else i for i in ids if i in marked or (i == unknown[0] and guess)]
            raise _Refuse(I.problem("not_found", "regions", (f"no marked region {I._show(unknown[0], 30)}; " if marked else "this task marks no regions; ")
                                    + (f"marked regions: {', '.join(marked)}" if marked else "give circles instead"),
                                    allowed={"values": list(marked) + [I.ALL_REGIONS]} if marked else None, did_you_mean=guess,
                                    fix={"regions": fixed} if marked else {"regions": None}))
        rois = [dict(marked[i]) for i in dict.fromkeys(ids)] + [dict(c) for c in circles]
        taken = [r.get("id", f"roi{k}") for k, r in enumerate(rois)]     # a circle without an id is named by the scanner: roi<k>
        twice = next((i for i in taken if taken.count(i) > 1), None)
        if twice is not None:
            raise _Refuse(I.problem("invalid_value", "circles", f"two read-outs carry the id {I._show(twice, 30)}; give each circle its own id", "task"))
        if len(rois) > session.MAX_ROIS:
            raise _Refuse(I.problem("invalid_value", "circles", f"{len(rois)} read-outs in one call; at most {session.MAX_ROIS}", "task",
                                    allowed={"max_items": session.MAX_ROIS}))
        return rois

    # ------------------------------------------------------------------ tools
    def _tool_scanner_info(self, a):
        """One topic of the datasheet per call. ``data`` of the overview is the whole instrument card (what a scripted
        policy reads); the text of every topic is the short form of it."""
        card, topic, exam = self._session.info(), a.get("topic"), self._session.exam
        if a.get("parameter") is not None:
            got = self._parameter_info(a["parameter"])
            if topic is None or got.is_error:
                return got
            rest = self._tool_scanner_info({"topic": topic})
            return Observation(text=got.text + "\n" + rest.text, data=dict(rest.data, **got.data))
        topic = topic or "overview"
        if topic == "task":
            return Observation(text="\n".join(f"{k}: {json.dumps(v, separators=(', ', ': '))}" for k, v in card["task"].items()), data=card["task"])
        if topic == "sequences":
            lines = [f"{s}: {d['description']}" for s, d in card["sequences"].items()]
            lines.append("topic = one of them: its signal model, its parameters and an example call")
            return Observation(text="\n".join(lines), data=card["sequences"])
        if topic in SEQUENCE_TOPICS:
            return Observation(text=self._sequence_text(topic, card), data={topic: card["sequences"][topic]})
        if topic == "geometry":
            names = [k for k in console.parameter_names() if self._shared(k) and k not in console.PROTOCOL_DEPENDENT]
            rows = []
            for k in names:
                usual = next(s for s in SEQUENCE_TOPICS if self._common(s, k))
                differs = [f"{s}: {self._range(s, k)}, default {console.default_text(s, k, exam)}" for s in SEQUENCE_TOPICS if not self._common(s, k)]
                rows.append([k, self._range(usual, k), console.default_text(usual, k, exam),
                             console.PARAMETER_UNITS[k] + (f" ({'; '.join(differs)})" if differs else "")])
            text = "parameters every sequence takes: parameter, range, default: meaning\n" + "\n".join(f"{k}, {r}, {d}: {m}" for k, r, d, m in rows)
            return Observation(text=text, data={"parameters": {k: {s: card["sequences"][s]["parameters"][k] for s in SEQUENCE_TOPICS} for k in names}})
        if topic == "presets":
            from ..scanner import presets
            rows = presets.presets(exam["b0_t"], exam)
            lines = [f"starting protocols at {exam['b0_t']:g} T from the nominal tissue table (this session's tissue and system differ: a "
                     "starting point, not an answer); geometry at the defaults"]
            lines += [f"{k}: {json.dumps(_plain(v['protocol']), separators=(', ', ': '))}   [{v['rule']}]" for k, v in rows.items()]
            return Observation(text="\n".join(lines), data={"presets": rows})
        parts = datasheet_parts()
        if topic == "physics":
            text = "\n".join([
                "signal mechanisms, one topic each (scanner_info(topic=...)):",
                f"  {' | '.join(SEQUENCE_TOPICS)}   the signal model of that sequence, with its parameters",
                "  encoding        wrap-around, ringing, displacement of off-resonant spins and of fat along the readout",
                "  slice_profile   what a selective pulse does to the flip angle through the slice (it biases flip-angle and T1 methods)",
                "  fat             fat saturation and its frequency offset",
                "  motion          periodic motion and its ghosts",
                "  hidden          what is hidden and what is not simulated",
                parts["notation"]])
            return Observation(text=text, data={"topics": list(SEQUENCE_TOPICS + PHYSICS_TOPICS), "notation": parts["notation"]})
        if topic in ("cost", "outputs") + PHYSICS_TOPICS:
            return Observation(text=parts[topic], data={"markdown": parts[topic]})
        cost, hw = card["cost"], card["hardware"]
        text = "\n".join([
            f"{card['scanner']}, {card['field_strength_t']:g} T · {card['slice']} · default fov_mm {card['default_fov_mm']:g} · smallest pixel "
            f"{card['min_pixel_mm']:g} mm · {card['engine']}",
            "images [n_images, ny, nx]: row = y, column = x; the readout runs along x when pe_dir = y (default); pixel positions, a shifted "
            "or turned field of view: topic=outputs",
            f"scan time: {cost['rule']}; overhead {cost['overhead_s_per_scan']:g} s; dummy cycles: {cost['dummy_time_ms_steady_state_sequences'] / 1000:g} s "
            f"(spgr, afi, bssfp), {cost['dummy_cycles_se_ir']} cycle (se, ir), {cost['dummy_shots_epi']} shots (epi)",
            f"hardware: gradient {hw['gradient_max_mt_per_m']:g} mT/m, slew {hw['slew_max_t_per_m_s']:g} T/m/s, B1rms limit {hw['b1rms_limit_ut']:g} uT, "
            f"peak B1 {hw['b1_peak_limit_ut']:g} uT, time raster {hw['time_raster_ms']:g} ms",
            "topics (scanner_info(topic=...)):",
            f"  {' | '.join(SEQUENCE_TOPICS)}   signal model, parameters and an example call of that sequence",
            "  geometry   the parameters every sequence takes: range, default, meaning",
            "  cost       scan time, minimum TE and TR, RF power and gradient limits, noise",
            "  outputs    images, read-outs, view, profile; positions on a turned field of view",
            "  physics    index of the signal mechanisms (encoding, slice_profile, fat, motion, hidden)",
            "  task       the task card        presets   named starting protocols",
            "scanner_info(parameter=<name>): range, default and meaning of one protocol parameter"])
        return Observation(text=text, data=card)

    @staticmethod
    def _shared(name: str) -> bool:
        """A parameter that every sequence takes."""
        return all(name in spec for spec in console.SEQUENCES.values())

    @staticmethod
    def _common(seq: str, name: str) -> bool:
        """A shared parameter with a default, whose range and default in ``seq`` are those of most sequences."""
        if not ScannerEnv._shared(name) or name in console.PROTOCOL_DEPENDENT or console.SEQUENCES[seq][name][3] is None:
            return False
        specs = [spec[name] for spec in console.SEQUENCES.values()]
        return console.SEQUENCES[seq][name] == max(specs, key=specs.count)

    @staticmethod
    def _range(seq: str, name: str) -> str:
        lo, hi, kind, _ = console.SEQUENCES[seq][name]
        return "true / false" if kind == "b" else " / ".join(map(str, lo)) if kind == "c" else f"{lo:g} to {hi:g}"

    def _sequence_text(self, seq: str, card: dict) -> str:
        """Signal model, own parameters, the shared parameters with their defaults in one line, one example call."""
        parts, exam = datasheet_parts(), self._session.exam
        default = lambda k: console.default_text(seq, k, exam)             # noqa: E731
        own = [k for k in console.parameter_names() if k in console.SEQUENCES[seq] and not self._common(seq, k)]
        rest = [k for k in console.parameter_names() if k in console.SEQUENCES[seq] and self._common(seq, k)]
        fixed = "".join(f"{k} is fixed at {v:g}. " for k, v in console.FIXED.get(seq, {}).items())
        lines = [parts[seq]]
        if seq in _ECHO:
            lines.append(parts[_ECHO[seq]])
        lines += [(fixed + f"Parameters of {seq} (range; default):").strip(),
                  "\n".join(f"  {k}: {self._range(seq, k)}; " + ("required" if default(k) is None else f"default {default(k)}") for k in own),
                  "taken by every sequence (default): " + ", ".join(f"{k} {default(k)}" for k in rest) + "; range and meaning: topic=geometry",
                  f"example: plan {json.dumps(SEQUENCE_EXAMPLES[seq], separators=(', ', ': '))}"]
        return "\n".join(lines)

    def _parameter_info(self, name):
        """Range, default and meaning of one protocol parameter, per sequence where they differ."""
        names = console.parameter_names()
        key = console.ALIASES.get(name, name) if isinstance(name, str) else name
        if key not in names:
            guess = I.did_you_mean(key, names + list(console.ALIASES))
            guess = console.ALIASES.get(guess, guess)
            return self._error([I.problem("invalid_value", "parameter", f"no protocol parameter {I._show(name, 40)}" + (f"; did you mean {guess}?" if guess else "")
                                          + ("" if guess else f"; parameters: {', '.join(names)}"), "vocabulary", did_you_mean=guess,
                                          allowed={"values": names}, fix={"parameter": guess} if guess else None)])
        used = [s for s in console.SEQUENCES if key in console.SEQUENCES[s]]
        by = {}
        for s in used:
            by.setdefault((self._range(s, key), console.default_text(s, key, self._session.exam) or "required"), []).append(s)
        unit = console.PARAMETER_UNIT.get(key, "")
        lines = [f"{key}: {console.PARAMETER_UNITS[key]}" + (f" (unit: {unit})" if unit and unit not in console.PARAMETER_UNITS[key] else "")]
        lines += [f"  {', '.join(ss)}: range {r}, default {d}" for (r, d), ss in by.items()]
        fixed = [f"{s}: fixed at {v[key]:g}" for s, v in console.FIXED.items() if key in v]
        missing = [s for s in console.SEQUENCES if s not in used and not any(f.startswith(s + ":") for f in fixed)]
        if fixed or missing:
            lines.append("  " + "; ".join(fixed + ([f"not a parameter of {', '.join(missing)}"] if missing else [])))
        return Observation(text="\n".join(lines), data={"parameter": {key: {s: self._session.info()["sequences"][s]["parameters"][key] for s in used}}})

    def _tool_status(self, a):
        s = dict(self._full_status())
        s.update(tool_calls=s["tool_calls"] + 1, tool_calls_left=max(0, s["tool_calls_left"] - 1))      # this call included
        ep, lim = self._episode, self.limits()
        scans = [dict(scan_index=k, cost_s=v["cost_s"], images=int(v["magnitude"].shape[0]), shape=list(v["magnitude"].shape[1:]),
                      header=v["header"]) for k, v in sorted(self._scanner.scans.items())]
        terminal = self.terminal_tool()
        props, need = terminal.parameters.get("properties", {}), terminal.parameters.get("required", [])
        lines = [f"scan time: {s['scan_time_used_s']:.2f} of {s['scan_time_budget_s']:g} s used, {s['scan_time_left_s']:.2f} s left",
                 f"tool calls: {s['tool_calls']} of {s['max_tool_calls']} used, {s['tool_calls_left']} left (this call included)",
                 f"rejected calls: {s['rejected_calls']} of {s['max_rejected_calls']}; refused {terminal.name} calls: "
                 f"{ep['invalid_deliverables']} of {self.max_invalid_deliverables}",
                 f"stored scans: {len(scans)}"]
        lines += [f"  scan_index {x['scan_index']} · {x['cost_s']:.2f} s · {x['images']} image{'s' if x['images'] > 1 else ''} · {x['header']}" for x in scans]
        lines.append(f"{terminal.name} (ends the episode): " + (", ".join(
            f"{k}{'' if k in need else ' (optional)'}: {_type_text(v)}" for k, v in props.items()) or "no parameters"))
        self._shown = None                                               # the status line under this answer is the full one
        return Observation(text="\n".join(lines), data=dict(limits=lim, used=dict(
            scan_time_s=s["scan_time_used_s"], scan_time_left_s=s["scan_time_left_s"], tool_calls=s["tool_calls"], tool_calls_left=s["tool_calls_left"],
            rejected_calls=s["rejected_calls"], invalid_deliverables=ep["invalid_deliverables"]), scans=scans,
            terminal=dict(name=terminal.name, parameters=terminal.parameters)))

    def _tool_plan(self, a):
        """The console's verdict in two lines: what the protocol costs as a scan now and what that leaves of the scan
        time; then what the request does not say: the image the protocol gives (pixels as (y, x), field of view, the
        phase-encode axis and its lines), the echo time, the minimum TE / TR the console enforces, RF power, fat shift
        and relative SNR. What the console changed against the request is said, the phase side included (a whole, even
        number of pixels). A third line names the marked regions the image would miss, when there are any. What the
        request said and what the defaults fill in is not repeated, unless ``verbose`` (the header line, every
        realised parameter, the timing). ``data`` holds everything, always."""
        protocol, adjusted = self._adjust(self._protocol(a))
        p = self._scanner.plan(protocol)
        if not p["accept"]:
            return self._error(I.console_problems(p["violations"], protocol, self._session.exam))
        parts, t, real = p["scan_s_parts"], p["timing"], p["realised"]
        # the price is that of scanning this protocol now; a plan is also how a deliverable is looked at, and a
        # deliverable is not scanned: so the line says "as a scan now", what is left after it, and never "the budget"
        if p["fits_budget"]:
            first = f"plan ok · as a scan now: {p['scan_s']:.2f} s, leaves {max(0.0, p['remaining_s'] - p['scan_s']):.2f} s"
        else:
            first = f"plan ok · as a scan now: needs {p['scan_s']:.2f} s, {p['remaining_s']:.2f} s left"
        # a console that reuses the adjustments of the scan before prices the next scan lower than the same protocol as
        # a scan of its own, which is what a verifier plans for a deliverable: both prices, each with its meaning
        own = console.plan(protocol, self._session.exam)
        alone = own["scan_s"] if own["accept"] and abs(own["scan_s"] - p["scan_s"]) > 1e-9 else None
        if alone is not None:
            first += f" (reuses the adjustments of the scan before) · as a scan of its own: {alone:.2f} s"
        low = console.SEQUENCES[real["sequence"]]                        # the minimum the console enforces: the timing's or the range's
        enforced = lambda k: max(t[k], low[k[:-len("_min_ms")] + "_ms"][0]) if k[:-len("_min_ms")] + "_ms" in low else t[k]   # noqa: E731
        mins = ", ".join(f"{k[:-len('_min_ms')]}_ms {enforced(k):g}" for k in t if k.endswith("_min_ms"))
        changed = [f"{k} {v:g} -> {real[k]:g}" for k, v in protocol.items()
                   if I.T._is_number(v) and I.T._is_number(real.get(k)) and abs(real[k] - v) > 1e-9]
        asked = protocol.get("phase_fov")                                # the phase side is a whole, even number of pixels
        if I.T._is_number(asked) and abs(real["matrix_pe"] - real["matrix"] * asked) > 1e-6:
            changed.append(f"phase_fov {asked:g} -> {real['matrix_pe'] / real['matrix']:g} ({real['matrix_pe']} px, an even number)")
        lines = [first] + self._adjust_notes(adjusted)
        if changed:
            lines.append("realised (on the console's raster): " + ", ".join(changed))
        d = p["derived"]
        (fy, fx), pe = session.fov_yx(real), real["pe_dir"]
        ny, nx = (real["matrix_pe"], real["matrix"]) if pe == "y" else (real["matrix"], real["matrix_pe"])
        lines += [f"{ny}x{nx} px (y, x), FOV {fy:g} x {fx:g} mm, phase-encode {pe}: {real['lines']} lines · {self._echo(real)}"
                  f" · minimums: {mins} · SAR {p['limits']['rf_power_percent']:.0f} % · WFS {d['water_fat_shift_px']:.1f} px ({d['water_fat_shift_axis']})"
                  f" · rel. SNR {d['relative_snr']:.2f}"
                  + (f" · gradient at {p['limits']['gradient_percent']:g} % of its limit" if p["limits"]["gradient_percent"] >= LIMIT_SHOWN else "")]
        none, cut = self._regions_missed(real, (ny, nx))
        if none or cut:
            lines.append("; ".join(x for x in ("no pixel of this image in " + ", ".join(none) if none else "",
                                               ("partly outside it: " if none else "partly outside this image: ") + ", ".join(cut) if cut else "") if x))
        if a.get("verbose"):
            lines += [p["header"], f"cost: overhead {parts['overhead_s']:g} + dummy cycles {parts['dummy_s']:.2f} + acquisition {parts['acquisition_s']:.2f} s",
                      "realised: " + R.kv(real), "timing: " + R.kv(t)]
        data = dict(p, adjusted=adjusted) if adjusted else p
        if alone is not None:
            data = dict(data, scan_s_own_scan=alone)
        if none or cut:
            data = dict(data, regions_without_pixels=none, regions_partly_outside=cut)
        return Observation(text="\n".join(lines), data=data)

    def _regions_missed(self, real: dict, shape: tuple) -> tuple:
        """The marked regions a scan of this realised protocol would not hold: (ids of the regions in which no pixel
        of the image lies: their read-out would be empty; ids of the regions the edge of the image cuts). The first
        by the rule of the read-out itself (``session.roi_readouts``): a pixel belongs to a region when its centre lies
        in the circle."""
        if not self._regions:
            return [], []
        fov, ctr, rot = session.fov_yx(real), session.centre_yx(real), session.rotation_of(real)
        ygrid, xgrid = session.pixel_grid(shape, fov, ctr, rot)
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        edge = [(-(n // 2) - 0.5) * f / n for n, f in zip(shape, fov)], [(n - n // 2 - 0.5) * f / n for n, f in zip(shape, fov)]
        none, cut = [], []
        for r in self._regions:
            if not ((ygrid - r["y_mm"]) ** 2 + (xgrid - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2).any():
                none.append(r["id"])
                continue
            dy, dx = r["y_mm"] - ctr[0], r["x_mm"] - ctr[1]              # the centre of the circle along the image's rows and columns
            at = (dy * c + dx * s, -dy * s + dx * c)
            if any(at[k] - r["radius_mm"] < edge[0][k] - 1e-9 or at[k] + r["radius_mm"] > edge[1][k] + 1e-9 for k in (0, 1)):
                cut.append(r["id"])
        return none, cut

    @staticmethod
    def _echo(real: dict) -> str:
        """The echo time the console realised (every echo of a multi-echo scan)."""
        times = real.get("echo_times_ms") or [real["te_ms"]]
        return "te_ms " + ", ".join(f"{x:g}" for x in times)

    @staticmethod
    def _readout_table(readouts, scan_index=None):
        """Read-outs as rows. The image column is printed only when a scan holds more than one image."""
        many = any(len(r["mean_magnitude"]) > 1 for r in readouts)
        rows = []
        for r in readouts:
            for i in range(len(r["mean_magnitude"])):
                rows.append([r["id"], r["n_pixels"]] + ([i] if many else []) + [
                    R.sig(r["complex_mean_magnitude"][i]), R.sig(r["complex_mean_phase_rad"][i], 4), R.sig(r["mean_magnitude"][i]), R.sig(r["sd_magnitude"][i], 3)])
        text = R.rows(["region", "n_px"] + (["image"] if many else []) + ["|complex_mean|", "phase_rad", "mean|S|", "sd|S|"], rows[:MAX_TABLE_ROWS])
        if len(rows) > MAX_TABLE_ROWS:
            shown = list(dict.fromkeys(r[0] for r in rows[:MAX_TABLE_ROWS]))
            text += (f"\n... {len(rows) - MAX_TABLE_ROWS} more rows (shown: {shown[0]} to {shown[-1]}): "
                     f"read_regions(scan_index={scan_index}, regions=[...]) prints the regions you name")
        return text

    def _tool_scan(self, a):
        try:
            rois = self._rois(a)
        except _Refuse as r:
            return self._error([r.problem])
        protocol, adjusted = self._adjust(self._protocol(a))
        resp = self._scanner.scan(protocol, rois)
        resp.pop("arrays")
        k, shape = resp["scan_index"], resp["shape"]
        # the header line of a stored scan is behind ``status``; here: what the call did not say (index, image, cost, echo time)
        text = "\n".join([
            f"scan_index {k} · {shape[0]} image{'s' if shape[0] > 1 else ''} {shape[1]}x{shape[2]} (y, x) stored · cost {resp['cost_s']:.2f} s · "
            f"{self._echo(resp['realised'])}"]
            + self._adjust_notes(adjusted) + [
            self._readout_table(resp["roi_readouts"], k) if resp["roi_readouts"] else "no regions requested"])
        return Observation(text=text, data=dict(resp, adjusted=adjusted) if adjusted else resp)

    def _tool_read_regions(self, a):
        try:
            rois = self._rois(a)
        except _Refuse as r:
            return self._error([r.problem])
        if not rois:
            return self._error([I.problem("invalid_value", "circles", "nothing to read out: this task marks no regions, so give at least one circle",
                                          "task", fix={"circles": [{"y_mm": 0, "x_mm": 0, "radius_mm": 10}]})])
        k = self._scan_index(a)
        out = self._scanner.read_regions(k, rois)
        return Observation(text=f"read-out of scan_index {k}\n" + self._readout_table(out, k), data={"scan_index": k, "roi_readouts": out})

    def _scan_index(self, a) -> int:
        """The stored scan a call names; without one, the latest (0 before any scan: the scanner then says that none is stored)."""
        k = a.get("scan_index")
        return k if k is not None else max(self._scanner.scans, default=0)

    def _tool_view(self, a):
        v = self._scanner.image(self._scan_index(a), a.get("image_index", 0), a.get("kind", "magnitude"))
        img, fov, ctr = v.pop("array"), tuple(v["fov_yx_mm"]), tuple(v["centre_yx_mm"])
        rot = float(v.get("rotation_deg", 0.0) or 0.0)
        frame = session.frame_centre(ctr, rot)                           # where the centre lies along the image's own rows and columns
        px = (fov[0] / img.shape[0], fov[1] / img.shape[1])
        data = dict(scan_index=v["scan"], image_index=v["image"], kind=v["kind"], shape=v["shape"], fov_yx_mm=v["fov_yx_mm"],
                    centre_yx_mm=v["centre_yx_mm"], pixel_yx_mm=[round(px[0], 4), round(px[1], 4)], sequence=v["sequence"])
        if rot:
            data.update(rotation_deg=rot, frame_centre_yx_mm=[round(frame[0], 3), round(frame[1], 3)])
        cap = (f"scan_index {v['scan']} ({v['sequence']}), image_index {v['image']}, {v['kind']}, {img.shape[0]}x{img.shape[1]} (y, x), "
               f"FOV {fov[0]:g} x {fov[1]:g} mm, pixel {px[0]:.3g} x {px[1]:.3g} mm"
               + (f", centred at y {ctr[0]:g}, x {ctr[1]:g} mm" if any(ctr) else "") + (f", turned by {rot:+g} degrees" if rot else ""))
        turned = self._turned_note(rot)
        if self.obs_mode == "vision":
            png, rng = R.image_png(img, v["kind"], circles=self._regions, fov_yx_mm=fov, centre_yx_mm=ctr, rotation_deg=rot)
            w, h = R.png_size(png)
            return Observation(text=f"{cap}; grey scale spans {rng}; {w}x{h} px PNG, top = {frame[0] - fov[0] / 2:g} mm (y), left = {frame[1] - fov[1] / 2:g} mm (x)"
                               + ("; white rings = task regions" if self._regions else "") + (f"\n{turned}" if turned else ""),
                               images=[dict(png=png, caption=cap)], data=dict(data, png_bytes=len(png), display_range=rng))
        size = a.get("size", VIEW_SIZE)
        grid, legend = R.digit_map(img, v["kind"], size, fov, frame)
        # where the digits lie, so that a size is read off and not counted: the y of every row beside it, and under the
        # map the x of the columns (the map itself is the one ``data`` holds)
        ys, xs = R.digit_map_axes(img.shape, size, fov, frame)
        rows = grid.splitlines()
        mm = lambda step: (lambda q: f"{q:.0f}" if step >= 4 else f"{q:.1f}")   # noqa: E731  whole millimetres for blocks of 4 mm or more
        my, mx = mm(fov[0] / len(ys)), mm(fov[1] / len(xs))
        ruled = "\n".join([f"{row} {my(y)}" for row, y in zip(rows, ys)] + [R.ruler(xs, mx)])
        legend += "; block centres: y_mm after each row, x_mm below, where a number starts"
        return Observation(text="\n".join([cap, legend] + ([turned] if turned else []) + [ruled]), data=dict(data, digit_map=rows))

    @staticmethod
    def _turned_note(rot: float) -> str:
        """What positions mean on a turned image, in one sentence (empty without a rotation)."""
        if not rot:
            return ""
        return (f"turned by r = {rot:+g} degrees: y and x here run along the image's own rows and columns; (y, x) on the image is the patient "
                "position (y cos r - x sin r, y sin r + x cos r); regions and circles are always in patient coordinates")

    def _profile(self, a):
        """The scanner's profile. A range that reaches beyond the image is read where the image is: ``start_mm`` /
        ``stop_mm`` outside it are moved to its edge and the move is reported (``adjusted``), as long as part of the
        range lies inside. A line that misses the image (``position_mm``) is refused."""
        args = {k: a.get(k) for k in ("position_mm", "start_mm", "stop_mm", "width_mm", "samples")}
        adjusted = []
        if isinstance(args["samples"], int) and not isinstance(args["samples"], bool) and args["samples"] > session.MAX_PROFILE_SAMPLES:
            adjusted.append(dict(parameter="samples", requested=args["samples"], realised=session.MAX_PROFILE_SAMPLES, reason="the most a call returns"))
            args["samples"] = session.MAX_PROFILE_SAMPLES
        if all(I.T._is_number(args[k]) for k in ("start_mm", "stop_mm")) and args["stop_mm"] < args["start_mm"]:
            args["start_mm"], args["stop_mm"] = args["stop_mm"], args["start_mm"]      # the two ends of one range, in either order
        for _ in range(3):
            try:
                return self._scanner.profile(self._scan_index(a), a["axis"], a.get("image_index", 0), a.get("kind", "magnitude"),
                                             args["position_mm"], args["start_mm"], args["stop_mm"], args["width_mm"], args["samples"],
                                             a.get("reduce", "mean")), adjusted
            except session.ScannerError as e:
                x = e.extra
                name, edge = x.get("parameter"), x.get("nearest_valid")
                asked = args.get(name) if name in ("start_mm", "stop_mm") else None
                other = args.get("stop_mm" if name == "start_mm" else "start_mm")
                # a start before the image or a stop behind it, with the other end inside: the part of the range that exists
                inside = I.T._is_number(asked) and "low" in x and I.T._is_number(edge) and not adjusted_has(adjusted, name) and (
                    (asked < x["low"] and (not I.T._is_number(other) or other > x["low"])) if name == "start_mm" else
                    (asked > x["high"] and (not I.T._is_number(other) or other < x["high"])))
                if not inside:
                    raise
                adjusted.append(dict(parameter=name, requested=args[name], realised=edge, reason="the edge of the image"))
                args[name] = None                                        # the scanner's own default: the edge of the image
        raise session.InvalidRequest("the range lies outside the image", parameter="start_mm")

    def _width_cut(self, a: dict, p: dict) -> bool:
        """Whether the band ``position_mm`` +- ``width_mm`` / 2 of a profile reached beyond the image, so that the
        scanner read the part inside it: fewer pixels than asked for, about another centre."""
        width = a.get("width_mm")
        if not I.T._is_number(width) or width <= p["width_mm"] + 1e-6:
            return False
        s = self._scanner.stored(p["scan"])
        ypos, xpos = session.pixel_positions(s["magnitude"].shape, s["fov_yx_mm"], session.frame_centre(s["centre_yx_mm"], s.get("rotation_deg", 0.0) or 0.0))
        across = xpos if p["axis"] == "y" else ypos
        half = abs(across[1] - across[0]) / 2 if len(across) > 1 else 0.0
        at = a["position_mm"] if I.T._is_number(a.get("position_mm")) else float(across[len(across) // 2])
        return bool(at - width / 2 < across[0] - half - 1e-9 or at + width / 2 > across[-1] + half + 1e-9)

    def _tool_profile(self, a):
        p, adjusted = self._profile(a)
        axis, other = p["axis"], "x" if p["axis"] == "y" else "y"
        if self._width_cut(a, p):                                        # fewer pixels than asked for, because the image ends
            adjusted.append(dict(parameter="width_mm", requested=a["width_mm"], realised=p["width_mm"],
                                 reason=f"cut at the edge of the image; the line is centred at {other} = {p['position_mm']:g} mm"))
        rot, patient = p.pop("rotation_deg", 0.0), p.pop("patient_yx_mm", None)
        data = dict(scan_index=p.pop("scan"), image_index=p.pop("image"), **p)
        if rot:
            data.update(rotation_deg=rot, patient_yx_mm=patient)
        if adjusted:
            data["adjusted"] = adjusted
        unit = "rad" if p["kind"] == "phase" else "image units"
        head = (f"profile of scan_index {data['scan_index']} image_index {data['image_index']} ({p['sequence']}, {p['kind']}) along {axis} "
                f"at {other} = {p['position_mm']:g} mm, {p['width_px']} px = {p['width_mm']:g} mm wide, {p['reduce']} per sample")
        columns = f"{axis}_mm value" + (" patient_y_mm patient_x_mm" if rot else "")
        span = (f"{len(p['values'])} samples of {p['pixels_per_sample']} px = {p['sample_mm']:g} mm, {axis} from {p['start_mm']:g} to {p['stop_mm']:g} mm; "
                f"rows: {columns} (centre of the sample; {unit})")
        lines = [head] + self._adjust_notes(adjusted) + [span]
        digits = 1 if p["sample_mm"] >= 0.5 else 2                       # a tenth of a millimetre places a sample of half a millimetre or more
        mm = lambda y: f"{y:.{digits}f}".rstrip("0").rstrip(".")          # noqa: E731
        top = max((abs(v) for v in p["values"]), default=0.0)
        places = max(0, 3 - math.floor(math.log10(top))) if top > 0 else 0   # four significant digits at the largest value, the same
        val = lambda v: f"{v:.{places}f}"                                 # noqa: E731  resolution on every row (data: full precision)
        if rot:                                                          # a turned image: every sample also in patient coordinates
            table = "\n".join(f"{mm(y)} {val(v)} {mm(q[0])} {mm(q[1])}" for y, v, q in zip(p["positions_mm"], p["values"], patient))
            return Observation(text="\n".join(lines + [self._turned_note(rot), table]), data=data)
        table = "\n".join(f"{mm(y)} {val(v)}" for y, v in zip(p["positions_mm"], p["values"]))
        return Observation(text="\n".join(lines + [table]), data=data)

    def _tool_calc(self, a):
        text = a["expressions"]
        try:
            rows = calc_mod.calc(text, self._names)
        except SyntaxError as e:
            lines = text.split("\n") if isinstance(text, str) else []
            at = lines[e.lineno - 1].strip() if e.lineno and 0 < e.lineno <= len(lines) else ""
            return self._error([I.problem("malformed_call", "expressions", f"line {e.lineno} is not Python ({(e.msg or 'syntax')[:60]})"
                                          + (f": {I._show(at, 60)}" if at else "") + "; write for example a = 3 + 0.5 * 96; print(a / 60)")])
        except calc_mod._CALC_ERRORS as e:
            kind = getattr(e, "kind", "value")
            where = f" (line {e.line})" if getattr(e, "line", None) else ""
            tail = "; assign it first: name = value" if kind == "unknown" else ""
            return self._error([I.problem("invalid_value", "expressions", f"{str(e)[:200]}{where}{tail}", "shape" if kind == "unsupported" else "task")])
        if not rows:
            return Observation(text="calc: nothing to evaluate", data={"values": []})
        return Observation(text=calc_mod.render(rows), data={"values": [dict(name=name, value=v) for name, v in rows]})

    def _terminal(self, a):
        why = self.check_deliverable(a)
        if why:
            return self._error(self._deliverable_problems(why))
        # the verifier's check on the scanner's own log (same engine object: an independent verifier rebuilds it from the seed)
        head = self._session.seal("submit")                              # nothing is accepted after the deliverable
        rep = session.replay(self._cfg, self._session.events, engine=self._session.engine, head=self._session.receipt(with_key=True))
        reward, score, text = self.finish(a, rep)
        self._finish(reward, dict(score=score, replay=rep, submitted=True, budget=self._session.budget(), log_head=head))
        data = {"received": True}
        if self.reveal_score:
            data["score"] = score
        return Observation(text=text, data=data)

    def _deliverable_problems(self, why) -> list:
        """An ``invalid_deliverable`` refusal: nothing was delivered and the episode continues."""
        tail = "; nothing was delivered"
        if isinstance(why, dict) and why.get("violations"):
            out = []
            for p in I.console_problems(why["violations"], why.get("protocol"), self._session.exam):
                extra = {k: p[k] for k in ("code", "requested", "allowed", "nearest_valid", "fix", "did_you_mean") if k in p}
                out.append(I.problem("invalid_deliverable", p["parameter"], "the console cannot evaluate this deliverable: " + p["message"] + tail, **extra))
            return out
        message = why.get("message") if isinstance(why, dict) else str(why)
        extra = {k: why[k] for k in ("parameter", "requested", "allowed", "nearest_valid", "fix") if isinstance(why, dict) and k in why}
        return [I.problem("invalid_deliverable", extra.pop("parameter", None), f"{message}{tail}", **extra)]

    def close(self):
        if self._session:
            self._session.close()
        super().close()


def adjusted_has(adjusted: list, name: str) -> bool:
    return any(x["parameter"] == name for x in adjusted)


def _plain(value):
    """A protocol with whole numbers written without their ``.0`` (the same call, fewer characters)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return int(value) if isinstance(value, float) and value == int(value) else value


class _Refuse(Exception):
    def __init__(self, problem):
        super().__init__(problem["message"])
        self.problem = problem


def _type_text(schema: dict) -> str:
    """One parameter of a schema as a few words: its type, its values or range."""
    if "anyOf" in schema:
        return " or ".join(_type_text(s) for s in schema["anyOf"])
    if "enum" in schema:
        return " | ".join(json.dumps(e) for e in schema["enum"])
    t = schema.get("type", "value")
    if t == "object":
        inner = ", ".join(f"{k}: {_type_text(v)}" for k, v in schema.get("properties", {}).items())
        return "{" + (inner or "...") + "}"
    if t == "array":
        return f"list of {_type_text(schema.get('items', {}))}"
    if "minimum" in schema or "maximum" in schema:
        return f"{t} {schema.get('minimum', '')}..{schema.get('maximum', '')}"
    return t
