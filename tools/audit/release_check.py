"""Refuse a release whose tag, package version and changelog heading do not agree.

    python tools/audit/release_check.py v0.2.0

The tag is ``v<version>``; ``[project] version`` in pyproject.toml and ``mrgym.__version__`` are that version; and
CHANGELOG.md has a heading ``## <version>`` (the "Unreleased" notes were moved under it).
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read(*parts) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def versions() -> dict:
    project = re.search(r'^version\s*=\s*"([^"]+)"', read("pyproject.toml"), re.M)
    package = re.search(r'^__version__\s*=\s*"([^"]+)"', read("src", "mrgym", "__init__.py"), re.M)
    headings = re.findall(r"^## \[?([0-9][^\]\s]*)\]?", read("CHANGELOG.md"), re.M)
    return dict(pyproject=project.group(1) if project else None, package=package.group(1) if package else None, changelog=headings)


def problems(tag: str) -> list[str]:
    v = versions()
    want = tag[1:] if tag.startswith("v") else None
    out = []
    if want is None:
        out.append(f"the tag {tag!r} does not look like v<version>")
    if v["pyproject"] != v["package"]:
        out.append(f"pyproject.toml says {v['pyproject']}, mrgym.__version__ says {v['package']}")
    if want is not None and v["pyproject"] != want:
        out.append(f"the tag is {tag}, the package version is {v['pyproject']}")
    if want is not None and want not in v["changelog"]:
        out.append(f"CHANGELOG.md has no heading '## {want}' (headings: {v['changelog'][:3]})")
    if want is not None and re.search(r"(dev|a|b|rc)\d*$", want):
        out.append(f"{want} is a pre-release: publish it as a GitHub pre-release only")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    bad = problems(argv[0])
    for p in bad:
        print("release check:", p)
    print("release check: " + ("ok" if not bad else "refused"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
