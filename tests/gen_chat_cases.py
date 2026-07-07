#!/usr/bin/env python3
"""Chat-template parity fixture: render the model's Jinja template with
the reference jinja2 (transformers env semantics: trim_blocks,
lstrip_blocks, loopcontrols) for several message sets.

Usage: gen_chat_cases.py <model_snapshot_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import jinja2

CASES = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "2+2?"}],
    [{"role": "user", "content": "Capital of France?"},
     {"role": "assistant", "content": "Paris."},
     {"role": "user", "content": "And Germany?"}],
]


def main():
    snap = Path(sys.argv[1])
    template = (snap / "chat_template.jinja").read_text()
    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    tpl = env.from_string(template)
    cases = []
    for msgs in CASES:
        rendered = tpl.render(messages=msgs, add_generation_prompt=True)
        cases.append({
            "messages": [[m["role"], m["content"]] for m in msgs],
            "rendered": rendered,
        })
    json.dump({"template": template, "cases": cases},
              open(sys.argv[2], "w"), ensure_ascii=False)
    print(f"wrote {len(cases)} chat cases → {sys.argv[2]}")


if __name__ == "__main__":
    main()
