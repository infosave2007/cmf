"""Chat parity of an LFM3 CMF against the Transformers reference (`lfm3_ref.py chat`):
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
    cortiq = sys.argv[3] if len(sys.argv) > 3 else "/root/lfm3/bin/cortiq"
    for rec in json.load(open(ref_path)):
        _, err = run(cortiq, model, rec["prompt"], {"CMF_PROMPT_DUMP": "1"}, 1)
        m = re.search(r"--- rendered prompt \((\d+) tokens\) ---\n(.*?)\n--- end ---", err, re.S)
        rendered = m[2] if m else None
        # The dump decodes without special tokens; the ids it prints (the
        # first and last 12) are what is compared against the reference.
        ids = lambda name: [int(t) for t in re.findall(r"(\d+):", (re.search(rf"^{name} (.*)$", err, re.M) or [""])[0])]
        want = rec["prompt_ids"]
        n = int(m[1]) if m else -1
        same = n == len(want) and ids("head") == want[:12] and ids("tail") == want[-12:]
        print(f"## {rec['prompt']}")
        print(f"template ids: {'identical' if same else 'DIFFERENT'} ({n} vs {len(want)} tokens)")
        if not same:
            print(f"  cortiq: {rendered!r} head {ids('head')} tail {ids('tail')}\n  ref:    {rec['rendered']!r} {want}")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rec["prompt_ids"], f)
        out, _ = run(cortiq, model, rec["prompt"], {"CMF_PROMPT_IDS": f.name}, len(rec["generated_ids"]))
        os.unlink(f.name)
        # `cortiq run` prints a banner, the prompt, the answer and a stats line.
        body = out.split(f"Prompt: {rec['prompt']}\n\n", 1)[-1].rsplit("\n[", 1)[0].strip()
        ref = rec["generated_text"].replace("<|im_end|>", "").strip()
        print(f"  greedy: {'identical' if body == ref else 'DIFFERENT'}")
        if body != ref:
            print(f"  ref:    {ref!r}\n  cortiq: {body!r}")


if __name__ == "__main__":
    main()
