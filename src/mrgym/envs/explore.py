"""``Scanner/Explore-v0``: the scanner with nothing asked. No reward; for demos, for people, for learning the console.

    env = mrgym.make("Scanner/Explore-v0", phantom="brain", field_t=3.0, lesion="ms", seed=1)
"""
from __future__ import annotations

from ..registry import Error
from ..scanner import world as W
from ..tools import ToolSpec
from .scanner_env import ScannerEnv

ENV_ID = "Scanner/Explore-v0"
PHANTOMS = ("brain", "abdomen", "qa")
LESIONS = {"brain": (None, "ms", "tumour"), "abdomen": (None, "liver"), "qa": (None,)}
BUDGET_S = 3600.0


def world_spec(phantom: str = "brain", field_t: float = 3.0, lesion: str | None = None) -> dict:
    if phantom not in PHANTOMS:
        raise Error(f"phantom must be one of {PHANTOMS}")
    if lesion not in LESIONS[phantom]:
        raise Error(f"lesion for {phantom} must be one of {LESIONS[phantom]}")
    if float(field_t) not in W.B1RMS_LIMIT_UT:
        raise Error("field_t must be 1.5 or 3.0")
    spec = dict(kind=phantom, b0_t=float(field_t), lesion=lesion)
    if phantom == "qa":
        spec["qa"] = dict(n_oil=2)
    return spec


class ExploreEnv(ScannerEnv):
    def __init__(self, spec, phantom: str = "brain", field_t: float = 3.0, lesion: str | None = None, budget_s: float = BUDGET_S, **kw):
        super().__init__(spec, **kw)
        self._world_spec, self._budget = world_spec(phantom, field_t, lesion), float(budget_s)
        self.phantom = phantom

    def make_instance(self, seed_hex, task_id):
        world = W.make_world(seed_hex, self._world_spec, layout_only=True) if self.phantom == "qa" else None
        task = dict(family="explore", phantom=self.phantom, regions=W.regions_public(world) if world else [])
        return dict(task_id=task_id, family="explore", level=self.level, seed=seed_hex, world_spec=self._world_spec,
                    budget_s=self._budget, max_requests=512, task=task)

    def task_text(self):
        what = {"brain": "a head", "abdomen": "an abdomen", "qa": "a QA phantom with vials (two of them hold oil)"}[self.phantom]
        return (f"# Free exploration\n\nYou operate a virtual MR scanner at {self._world_spec['b0_t']:g} T through tool calls. "
                f"{what.capitalize()} lies in the field of view. Nothing is asked and nothing is scored: plan and run any protocol, "
                "look at the images, read out regions. `finish` ends the session.")

    def terminal_tool(self):
        return ToolSpec("finish", "End the session. This environment pays no reward.",
                        {"type": "object", "additionalProperties": False, "properties": {"notes": {"type": "string"}}}, terminal=True)

    def finish(self, a, rep):
        return 0.0, dict(reward=0.0, passed=None, note="sandbox: no reward", notes=a.get("notes")), "session closed; this environment pays no reward"


def card() -> dict:
    return dict(
        id=ENV_ID, title="The scanner, nothing asked", kind="sandbox",
        given="A head, an abdomen or a QA phantom at 1.5 or 3 T; one hour of scan time",
        deliver="Nothing", scored_by="No reward",
        summary="The virtual scanner with a subject on the table and no task: for demos, for people at the command line "
                "(`mrgym play`), and for an agent to learn what the console does before it is asked for anything.",
        world=dict(phantom="keyword phantom = brain (default) | abdomen | qa; field_t = 3.0 (default) | 1.5; lesion = ms | tumour (brain), liver (abdomen)",
                   sequences_useful=["spgr", "afi", "se", "ir", "bssfp", "epi"], agent_side_formulas="datasheet section 5"),
        hidden=["the subject (anatomy and tissue values)", "fields and system state", "noise level", "the 128-bit world seed"],
        told=["the instrument datasheet", "budget and request limits", "realised protocol, timing, limits and cost of every plan and scan"],
        levels={1: dict(summary="any subject", budget_s=[BUDGET_S, BUDGET_S])},
        limits=dict(max_scanner_requests=512, max_tool_calls=1024),
        deliverable={"tool": "finish", "format": {"notes": "<optional text>"}},
        reward=dict(range=[0.0, 0.0], when="never", gates=[], error="-", score="-", passing="-"),
        controls=[], determinism="same (seed, keywords, tool calls) -> same observations and log hash chain; a repeated scan draws new noise",
    )
