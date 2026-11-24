"""The episode viewer: one self-contained HTML file that replays recorded episodes.

    from mrgym import record, viewer
    viewer.build([record.load("runs/episode/episode_bundle.json")], "viewer.html")

    mrgym view runs/a/episode_bundle.json runs/b/episode_bundle.json --out viewer.html

The file embeds the bundles as JSON and draws the images in the browser from the float32 data, so the grey-scale
window can be changed and magnitude and phase can be switched. It makes no network request and needs no server: open
it from disk. Another bundle can be loaded with the "Open bundle" button (the file is read locally).

What a reader sees: the timeline of steps; for each step the request, the console's reply text and any agent messages
recorded for it; the images of the scan the step refers to, with the task regions; the scan-time budget. The reward,
the gates, the delivered field of view over the hidden object and the hidden maps appear at the last step only.
"""
from __future__ import annotations

import importlib.resources
import json
import os

from .registry import Error

TEMPLATE = "viewer_template.html"
SLOT = "__EPISODES__"


def template() -> str:
    """The HTML template shipped with the package."""
    return importlib.resources.files("mrgym").joinpath(TEMPLATE).read_text(encoding="utf-8")


def render(bundles: list[dict]) -> str:
    """The viewer as one HTML string with ``bundles`` embedded."""
    bundles = list(bundles)
    for b in bundles:
        if not isinstance(b, dict) or not str(b.get("format", "")).startswith("mrgym-episode-bundle/"):
            raise Error("viewer.build takes episode bundles (see mrgym.record)")
    # "<" never appears raw in the embedded JSON, so no bundle text can close the script element
    payload = json.dumps(bundles, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = template()
    if html.count(SLOT) != 1:
        raise Error("the viewer template is damaged: it must hold exactly one episode slot")
    return html.replace(SLOT, payload)


def build(bundles: list[dict], out_path: str) -> str:
    """Write the viewer for ``bundles`` to ``out_path`` (one HTML file, no external assets). Returns the path."""
    html = render(bundles)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def cmd_view(a) -> int:
    from . import record
    bundles = [record.load(p) for p in a.bundle]
    path = build(bundles, a.out)
    print(path)
    print(f"{len(bundles)} episode(s), {os.path.getsize(path) / 1024:.0f} kB, self-contained: open it in a browser")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("view", help="one-file HTML viewer of recorded episodes")
    p.add_argument("bundle", nargs="+", help="episode_bundle.json files, or the folders that hold them")
    p.add_argument("--out", default="viewer.html")
    p.set_defaults(fn=cmd_view)
