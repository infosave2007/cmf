"""Chat parity of an Apex CMF against the PyTorch reference (`apex_ref.py chat`):
1. the prompt cortiq renders from the file's template (CMF_PROMPT_DUMP=1)
   against the checkpoint's template rendered by transformers;
2. greedy tokens from the reference's prompt ids (CMF_PROMPT_IDS) against
   the reference's greedy continuation.

usage: python3 -I chat_cmp.py REF_CHAT.json MODEL.cmf [CORTIQ]
"""

import json
import os
import re
import subprocess
import sys
import tempfile


def run(cortiq, model, prompt, env, n):
    out = subprocess.run(
        [cortiq, "run", model, "--prompt", prompt, "--greedy", "--max-tokens", str(n)],
        env={**os.environ, "XDG_RUNTIME_DIR": "/tmp", **env},
        capture_output=True,
        text=True,
        errors="replace",
    )
    return out.stdout, out.stderr


def main():
    ref_path, model = sys.argv[1], sys.argv[2]
    cortiq = sys.argv[3] if len(sys.argv) > 3 else "/root/apex/bin/cortiq"
    for rec in json.load(open(ref_path)):
        _, err = run(cortiq, model, rec["prompt"], {"CMF_PROMPT_DUMP": "1"}, 1)
        m = re.search(r"--- rendered prompt \((\d+) tokens\) ---\n(.*?)\n--- end ---", err, re.S)
        rendered = m[2] if m else None
        same = rendered == rec["rendered"]
        print(f"## {rec['prompt']}")
        print(f"template: {'identical' if same else 'DIFFERENT'} "
              f"({m[1] if m else '?'} vs {len(rec['prompt_ids'])} tokens)")
        if not same:
            print(f"  cortiq: {rendered!r}\n  ref:    {rec['rendered']!r}")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rec["prompt_ids"], f)
        out, _ = run(cortiq, model, rec["prompt"], {"CMF_PROMPT_IDS": f.name}, len(rec["generated_ids"]))
        os.unlink(f.name)
        print(f"  ref greedy:    {rec['generated_text']!r}")
        print(f"  cortiq greedy: {out.strip()!r}")


if __name__ == "__main__":
    main()
