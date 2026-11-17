"""Command cards of the R set, rendered from tldr-pages (CC-BY 4.0).

A card is the text a lookup record would put in front of the request:

    wc: Count lines, words, and bytes.
    - Count all lines in a file: wc -l {{path/to/file}}
    - ...

`{{[-l|--lines]}}` option placeholders are shown in their short form, as the
tldr client specification allows; value placeholders `{{path/to/file}}` stay,
the model has to fill them. "More information" / "See also" lines are dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

OPT = re.compile(r"\{\{\[([^|\]]+)\|[^\]]*\]\}\}")


def render_command(cmd: str) -> str:
    return OPT.sub(lambda m: m.group(1), cmd)


def parse_page(text: str) -> tuple[str, str, list[tuple[str, str]]]:
    """(name, description, [(what, command)]) of one tldr page."""
    name, desc, examples = "", [], []
    what = None
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("# "):
            name = line[2:].strip()
        elif line.startswith("> "):
            t = line[2:].strip()
            if not t.startswith(("More information", "See also")):
                desc.append(t)
        elif line.startswith("- "):
            what = line[2:].rstrip(":").strip()
        elif line.startswith("`") and line.endswith("`") and what is not None:
            examples.append((what, render_command(line[1:-1])))
            what = None
    return name, " ".join(desc), examples


def load_page(tldr_root: Path, page: str) -> tuple[str, str, list[tuple[str, str]]]:
    """`page` is `platform/name`, e.g. `common/wc`, `linux/ss`, `osx/pbcopy`."""
    return parse_page((tldr_root / "pages" / f"{page}.md").read_text())


def card(tldr_root: Path, page: str, keep: list[int] | None = None, label: str | None = None) -> str:
    """The card text of one page; `keep` picks examples by index (default: all)."""
    name, desc, ex = load_page(tldr_root, page)
    if keep is not None:
        ex = [ex[i] for i in keep if i < len(ex)]
    head = f"{name}: {desc}" if label is None else f"{label} — {name}: {desc}"
    return "\n".join([head] + [f"- {w}: {c}" for w, c in ex])


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1])
    for p in sys.argv[2:]:
        try:
            print(card(root, p))
        except FileNotFoundError:
            print(f"!! missing {p}")
        print()
