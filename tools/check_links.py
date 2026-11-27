"""Check every relative link of the Markdown pages: the file exists and, for ``page.md#anchor``, so does the heading.

    python tools/check_links.py                    # README.md, the other root pages, docs/, envs/, examples/, integrations/
    python tools/check_links.py README.md docs/*.md

Exits 1 and prints ``page:line: target`` for every broken link. Links to other sites are not opened.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ("*.md", "docs/**/*.md", "envs/**/*.md", "examples/**/*.md", "integrations/*.md", "tools/**/*.md")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)|(?:src|href)=\"([^\"]+)\"")


def anchors(path):
    """The anchors a Markdown renderer gives the headings of ``path`` (lower case, punctuation dropped, spaces to hyphens)."""
    with open(path, encoding="utf-8") as f:
        heads = [ln.lstrip("#").strip() for ln in re.sub(r"```.*?```", "", f.read(), flags=re.S).splitlines() if ln.startswith("#")]
    return {re.sub(r"[^a-z0-9 _-]", "", h.lower()).replace(" ", "-") for h in heads}


def broken(page):
    out, here = [], os.path.dirname(page)
    with open(page, encoding="utf-8") as f:
        text = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), f.read(), flags=re.S)     # code is not prose
    for n, line in enumerate(text.splitlines(), 1):
        for target in (a or b for a, b in LINK.findall(line)):
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = target.partition("#")
            full = os.path.normpath(os.path.join(here, path)) if path else page
            if not os.path.exists(full) or (anchor and full.endswith(".md") and anchor not in anchors(full)):
                out.append(f"{os.path.relpath(page, ROOT)}:{n}: {target}")
    return out


if __name__ == "__main__":
    pages = sys.argv[1:] or sorted({p for pat in PAGES for p in glob.glob(os.path.join(ROOT, pat), recursive=True)})
    bad = [b for p in pages for b in broken(os.path.abspath(p))]
    print("\n".join(bad) if bad else f"{len(pages)} pages: every relative link resolves")
    sys.exit(1 if bad else 0)
