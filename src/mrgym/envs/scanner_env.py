"""What every environment on the scanner shares: the session, five scanner tools and their text.

A subclass says what is asked (``make_instance``, ``task_text``), what is delivered (``terminal_tool``) and how it is
scored (``finish``). It never touches physics: the world, pricing, noise, log and replay are ``mrgym.scanner``.

    scanner_info   free, not counted     instrument card, sequence parameters, signal mechanisms, the task card
    plan           free, counted         dry run on the console: realised protocol, timing, scan seconds, limits
    scan           scan time, counted    run a protocol; header line, region read-outs; images stored
    read_regions   free, not counted     other circles on a stored scan
    view           free, not counted     one stored image as a digit map (text) or PNG (vision), with its extent in mm
"""
from __future__ import annotations

import importlib.resources
import json
import os

from .. import admission
from .. import render as R
from ..env import Env, Observation
from ..scanner import console, session
from ..tools import ToolSpec

_CIRCLE = {"type": "object", "additionalProperties": False, "required": ["y_mm", "x_mm", "radius_mm"],
           "properties": {"id": {"type": "string", "description": "label echoed in the read-out"},
                          "y_mm": {"type": "number", "minimum": -session.MAX_POSITION_MM, "maximum": session.MAX_POSITION_MM},
                          "x_mm": {"type": "number", "minimum": -session.MAX_POSITION_MM, "maximum": session.MAX_POSITION_MM},
                          "radius_mm": {"type": "number", "minimum": 1, "maximum": session.MAX_POSITION_MM}}}
_REGIONS = {"anyOf": [{"type": "string", "enum": ["task"], "description": "the task's marked regions"},
                      {"type": "array", "maxItems": session.MAX_ROIS, "items": _CIRCLE}],
            "description": f"circles to read out, at most {session.MAX_ROIS}; \"task\" (default) = the task's marked regions, if any"}
TOPICS = ("overview", "sequences", "physics", "task")


def datasheet() -> str:
    """The instrument datasheet shipped with the package (the same text as docs/scanner.md)."""
    return importlib.resources.files("mrgym.scanner").joinpath("datasheet.md").read_text(encoding="utf-8")


def scanner_tools(obs_mode: str = "text") -> list[ToolSpec]:
    """The five scanner tools, the same in every environment."""
    view_kind = ("a PNG image with the task regions outlined" if obs_mode == "vision"
                 else "a digit map (block means on a 0-9 scale)")
    proto = console.protocol_schema()
    return [
        ToolSpec("scanner_info", "Instrument datasheet. No argument: overview (field strength, geometry, calls, cost rule, "
                 "hardware limits). topic=sequences: every protocol parameter with range, unit and default. topic=physics: "
                 "the signal mechanism of each sequence, what is hidden and what is not simulated. topic=task: the task "
                 "card. Free; not counted.",
                 {"type": "object", "additionalProperties": False, "properties": {
                     "topic": {"type": "string", "enum": list(TOPICS)}}}),
        ToolSpec("plan", "Dry run of a protocol on the console: realised parameters (defaults filled, times snapped to "
                 "the raster), minimum TE / TR, scan seconds, RF power and gradient use as percent of the limit, the header "
                 "line, accept or reject with typed reasons. Never returns signal. Free; counts as a scanner request.",
                 {"type": "object", "additionalProperties": False, "required": ["protocol"],
                  "properties": {"protocol": proto}}, counted=True),
        ToolSpec("scan", "Run a protocol. Charges its scan time to the budget; a refused scan costs nothing. Returns the "
                 "header line, the cost, and per region and image: pixel count, magnitude and phase of the complex mean "
                 "(no noise floor), mean and SD of the magnitude (has a noise floor). The images are stored under the "
                 "returned scan index for read_regions and view.",
                 {"type": "object", "additionalProperties": False, "required": ["protocol"],
                  "properties": {"protocol": proto, "regions": _REGIONS}}, cost="scan_time", counted=True),
        ToolSpec("read_regions", "Read out other circles on the stored images of an earlier scan (for example a "
                 "background region to measure noise). Same columns as scan. Free; not counted.",
                 {"type": "object", "additionalProperties": False, "required": ["scan", "regions"],
                  "properties": {"scan": {"type": "integer", "minimum": 0, "description": "scan index"},
                                 "regions": {"type": "array", "maxItems": session.MAX_ROIS, "items": _CIRCLE}}}),
        ToolSpec("view", f"Look at one stored image of an earlier scan: {view_kind}, plus the extent in mm of the pixels "
                 "brighter than 20 % of the image's 99.5th percentile (isolated pixels ignored) and whether they touch the "
                 "image border. Free; not counted.",
                 {"type": "object", "additionalProperties": False, "required": ["scan"],
                  "properties": {"scan": {"type": "integer", "minimum": 0, "description": "scan index"},
                                 "image": {"type": "integer", "minimum": 0, "description": "image index within the scan (default 0)"},
                                 "kind": {"type": "string", "enum": ["magnitude", "phase"]},
                                 "size": {"type": "integer", "minimum": 8, "maximum": 64,
                                          "description": "digit-map blocks along the longer side (default 32; text mode only)"}}}),
    ]


class ScannerEnv(Env):
    """Base class of the ``Scanner/...`` environments."""

    def __init__(self, spec, log_dir: str | None = None, **kw):
        super().__init__(spec, **kw)
        self._log_dir = log_dir
        self._cfg = self._session = self._scanner = None
        self._shallow_schemas = {}

    # ------------------------------------------------------------------ hooks of a subclass
    def make_instance(self, seed_hex: str, task_id: str) -> dict:      # -> episode config for scanner.session
        raise NotImplementedError

    def task_text(self) -> str:                                         # -> the statement the agent reads at reset
        raise NotImplementedError

    def terminal_tool(self) -> ToolSpec:
        raise NotImplementedError

    def finish(self, arguments: dict, replay: dict) -> tuple[float, dict, str]:   # -> (reward, score dict, text for the agent)
        raise NotImplementedError

    # ------------------------------------------------------------------ Env hooks
    def _reset(self, seed_hex):
        task_id = admission.task_id_for(self.spec, seed_hex)             # one-way: the id does not give the seed
        self._cfg = self.make_instance(seed_hex, task_id)
        if self._session:
            self._session.close()
        log_dir = os.path.join(self._log_dir, self._episode["id"]) if self._log_dir else None
        self._session = session.ScannerSession(self._cfg, log_dir=log_dir)
        self._scanner = session.LocalScanner(self._session)
        self._regions = self._cfg.get("task", {}).get("regions", [])
        specs = self._tool_specs()
        # the console validates the content of a protocol (typed violations, logged); the facade only its shape
        self._shallow_schemas = {t.name: dict(t.parameters, properties=dict(t.parameters["properties"], protocol={"type": "object"}))
                                 for t in specs if "protocol" in t.parameters["properties"]}
        c = self._cfg
        tools = "; ".join(f"{t.name} ({'ends the episode' if t.terminal else t.cost.replace('_', ' ')})" for t in specs)
        text = self.task_text().rstrip() + (f"\n\nBudget: {c['budget_s']:.0f} s of scan time, counted by the scanner. At most "
                                            f"{c['max_requests']} scanner requests (plans + scans) and {self.max_steps} tool calls.\n"
                                            f"Tools: {tools}.")
        return Observation(text=text, data=dict(task=c.get("task", {}), budget_s=c["budget_s"], max_requests=c["max_requests"],
                                                level=self.level, env_id=self.spec.id))

    def _info(self):
        """``admitted``: True / False when this instance is in the stored control-policy run, None otherwise."""
        stored = admission.stored_instance(self.spec.id, self._cfg["task_id"]) if self.spec.family else None
        return dict(task_id=self._cfg["task_id"], admitted=stored["admitted"] if stored else None)

    def _tool_specs(self):
        return scanner_tools(self.obs_mode) + [self.terminal_tool()]

    def _status(self):
        b = self._session.budget()
        return dict(scan_time_used_s=b["used_s"], scan_time_budget_s=b["budget_s"], scan_time_left_s=b["remaining_s"],
                    scans=b["scans"], scanner_requests=b["requests"], max_scanner_requests=b["max_requests"])

    def _status_line(self, s):
        return (f"[scan time {s['scan_time_used_s']:.2f}/{s['scan_time_budget_s']:.0f} s used · {s['scan_time_left_s']:.2f} s left · "
                f"scans {s['scans']} · scanner requests {s['scanner_requests']}/{s['max_scanner_requests']} · "
                f"tool calls {s['tool_calls']}/{s['max_tool_calls']}]")

    def _call(self, name, a):
        try:
            if name == self.terminal_tool().name:
                return self._terminal(a)
            return getattr(self, "_tool_" + name)(a)
        except session.ScannerError as e:
            extra = "".join(f"\n- {v['code']} {v.get('parameter') or ''}: {v['message']}" for v in e.extra.get("violations", []))
            return Observation(is_error=True, data=e.to_json(),
                               text=f"refused: {e.type}: {e.message if not extra else 'the console rejects this protocol'}{extra}\nnothing was charged")

    # ------------------------------------------------------------------ trainer side
    def episode_config(self) -> dict:
        """The hidden episode config (holds the seed). For whoever runs the episode: serving the same episode over
        HTTP (``mrgym serve``), or verifying a log. Never show it to the agent."""
        return dict(self._cfg)

    def scanner_log(self) -> list:
        """The scanner's hash-chained log of this episode."""
        return list(self._session.events)

    def stored_scans(self) -> dict:
        """Trainer side: the scans of this episode, {scan index: {"magnitude", "phase" (float32 [n_images, ny, nx]),
        "realised", "fov_yx_mm"}}. What a viewer or a recorder needs."""
        return dict(self._scanner.scans)

    def hidden_world(self) -> dict:
        """Trainer side: the hidden world of this episode (arrays on the master grid, tissue values, fields, system
        state). For figures, recorders and verifiers; never for the agent."""
        return self._session.world

    # ------------------------------------------------------------------ tools
    def _tool_scanner_info(self, a):
        card, topic = self._session.info(), a.get("topic", "overview")
        if topic == "task":
            return Observation(text=json.dumps(card["task"], indent=1), data=card["task"])
        if topic == "sequences":
            lines = []
            for s, d in card["sequences"].items():
                rows = []
                for k, p in d["parameters"].items():
                    rng = "true / false" if p["type"] == "boolean" else " / ".join(map(str, p["choices"])) if p["type"] == "choice" \
                        else f"{p['min']} .. {p['max']}"
                    rows.append([k, rng, p["unit"], "required" if p.get("required") else p.get("default")])
                lines += [f"{s}: {d['description']}", R.table(["parameter", "range", "unit", "default"], rows), ""]
            return Observation(text="\n".join(lines).rstrip(), data=card["sequences"])
        if topic == "physics":
            md = datasheet()
            text = md[md.index("## 5."):md.index("## 6.")] + md[md.index("## 7."):]
            return Observation(text=text.strip(), data={"markdown": text})
        keep = {k: card[k] for k in ("scanner", "api_version", "engine", "field_strength_t", "slice", "geometry", "default_fov_mm",
                                     "min_pixel_mm", "budget_s", "max_requests", "cost", "hardware", "outputs")}
        text = "\n".join(f"{k}: {json.dumps(v)}" for k, v in keep.items()) \
            + f"\nsequences: {sorted(card['sequences'])} (see topic=sequences, topic=physics)"
        return Observation(text=text, data=card)

    def _tool_plan(self, a):
        p = self._scanner.plan(a["protocol"])
        if not p["accept"]:
            return Observation(is_error=True, data=p, text="plan: rejected\n" + "\n".join(
                f"- {v['code']} {v.get('parameter') or ''}: {v['message']}" for v in p["violations"]))
        parts = p["scan_s_parts"]
        text = "\n".join([
            f"plan: accepted · would cost {p['scan_s']:.2f} s (overhead {parts['overhead_s']:g} + dummy cycles {parts['dummy_s']:.2f} "
            f"+ acquisition {parts['acquisition_s']:.2f}) · fits the remaining budget: {'yes' if p['fits_budget'] else 'NO'}",
            p["header"],
            "realised: " + R.kv(p["realised"]),
            "timing: " + R.kv(p["timing"]),
            "limits (% of limit): " + R.kv(p["limits"])])
        return Observation(text=text, data=p)

    def _rois(self, regions):
        if regions in (None, "task"):
            return [dict(r) for r in self._regions]
        return regions

    @staticmethod
    def _readout_table(readouts):
        rows = []
        for r in readouts:
            for i in range(len(r["mean_magnitude"])):
                rows.append([r["id"], r["n_pixels"], i, R.sig(r["complex_mean_magnitude"][i]), R.sig(r["complex_mean_phase_rad"][i], 4),
                             R.sig(r["mean_magnitude"][i]), R.sig(r["sd_magnitude"][i], 3)])
        return R.table(["region", "n_px", "image", "|complex mean|", "phase_rad", "mean |S|", "sd |S|"], rows)

    def _tool_scan(self, a):
        resp = self._scanner.scan(a["protocol"], self._rois(a.get("regions")))
        resp.pop("arrays")
        k, shape = resp["scan_index"], resp["shape"]
        text = "\n".join([
            f"scan {k}: {shape[0]} image{'s' if shape[0] > 1 else ''} {shape[1]}x{shape[2]} (y, x) stored · cost {resp['cost_s']:.2f} s",
            resp["header"],
            self._readout_table(resp["roi_readouts"]) if resp["roi_readouts"] else "no regions requested"])
        return Observation(text=text, data=resp)

    def _tool_read_regions(self, a):
        out = self._scanner.read_regions(a["scan"], a["regions"])
        return Observation(text=f"read-out of scan {a['scan']}\n" + self._readout_table(out),
                           data={"scan": a["scan"], "roi_readouts": out})

    def _tool_view(self, a):
        v = self._scanner.view(a["scan"], a.get("image", 0), a.get("kind", "magnitude"))
        img, fov, box = v.pop("array"), tuple(v["fov_yx_mm"]), v["bounding_box"]
        cap = f"scan {v['scan']} ({v['sequence']}), image {v['image']}, {v['kind']}, {img.shape[0]}x{img.shape[1]} (y, x), FOV {fov[0]:g} x {fov[1]:g} mm"
        ruler = ("extent of pixels above 20 % of the 99.5th percentile: y {:.1f} .. {:.1f} mm, x {:.1f} .. {:.1f} mm{}".format(
            *box["y_mm"], *box["x_mm"], "; they touch the image border" if box["touches_edge"] else "") if box else "the image is empty")
        if self.obs_mode == "vision":
            png, rng = R.image_png(img, v["kind"], circles=self._regions, fov_yx_mm=fov)
            w, h = R.png_size(png)
            return Observation(text=f"{cap}; grey scale spans {rng}; {w}x{h} px PNG, top = {-fov[0] / 2:g} mm (y), left = {-fov[1] / 2:g} mm (x)"
                               + ("; white rings = task regions" if self._regions else "") + f"\n{ruler}",
                               images=[dict(png=png, caption=cap)], data=dict(v, png_bytes=len(png), display_range=rng))
        grid, legend = R.digit_map(img, v["kind"], a.get("size", 32), fov)
        return Observation(text=f"{cap}\n{ruler}\n{legend}\n{grid}", data=dict(v, digit_map=grid.splitlines()))

    def _terminal(self, a):
        # the verifier's check on the scanner's own log (same engine object: an independent verifier rebuilds it from the seed)
        rep = session.replay(self._cfg, self._session.events, engine=self._session.engine)
        reward, score, text = self.finish(a, rep)
        self._finish(reward, dict(score=score, replay=rep, submitted=True, budget=self._session.budget()))
        data = {"received": True}
        if self.reveal_score:
            data["score"] = score
        return Observation(text=text, data=data)

    def close(self):
        if self._session:
            self._session.close()
        super().close()
