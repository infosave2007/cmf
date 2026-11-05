#!/usr/bin/env python3
"""Where a planted fact's rows land in the engine's own exact-row score. Answers come through
`cortiq run`; the score of every prompt row comes from CMF_DORY_EXACT_LOG.
exact_probe.py CORTIQ MODEL HF_DIR WIKI CHARS TRIALS [ENV=VAL ...]"""
import os, random, subprocess, sys
import numpy as np
from transformers import PreTrainedTokenizerFast

cortiq, model, hf, wiki, chars, trials = sys.argv[1:7]
chars, trials = int(chars), int(trials)
log = "/tmp/exact_score.bin"
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", CMF_DORY_EXACT_LOG=log, RUST_LOG="cortiq_engine=info", **dict(a.split("=", 1) for a in sys.argv[7:]))
tok = PreTrainedTokenizerFast(tokenizer_file=hf + "/tokenizer.json", bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<|im_end|>")
tok.chat_template = open(hf + "/chat_template.jinja").read()
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(41)
KINDS = [
    lambda k: (f"Remember this pass key: {k}. It is important.", "What was the pass key given above? Reply with the digits only.", k),
    lambda k: (f"Note: the access code for the archive is {k}.", "What is the access code for the archive mentioned above? Reply with the digits only.", k),
    lambda k: (f"The password of the staging server is qX{k}vT.", "What is the password of the staging server mentioned above? Reply with the password only.", f"qX{k}vT"),
    lambda k: (f"The meeting with Ilmari Voskresky was moved to room {k}.", "Which room was the meeting with Ilmari Voskresky moved to? Reply with the number only.", k),
]
for t in range(trials):
    i = t % len(KINDS)
    key = "".join(random.choice("0123456789") for _ in range(6))
    note, q, ans = KINDS[i](key)
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    cut = random.randrange(0, chars * 8 // 10)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\n{q}"
    prompt = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    c0 = prompt.index(ans)
    frows = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a < c0 + len(ans) and b > c0]
    if os.path.exists(log):
        os.remove(log)
    r = subprocess.run([cortiq, "run", model, "--prompt", msg, "--greedy", "--no-think", "--max-tokens", "24"], capture_output=True, text=True, env=env, timeout=1500)
    tail = r.stdout[-140:]
    hit = ans in tail
    chosen = [l for l in r.stderr.splitlines() if "exact rows" in l]
    if os.path.exists(log):
        s = np.fromfile(log, dtype=np.float32)
        order = np.argsort(np.argsort(-s))
        ranks = order[frows] if max(frows) < len(s) else []
        print(f"trial {t} kind {i} prompt rows {len(enc['input_ids'])} scored {len(s)} fact rows {frows[0]}..{frows[-1]} ranks {list(ranks)} {'OK' if hit else 'miss'} {tail[-90:]!r}", flush=True)
    else:
        print(f"trial {t}: no score file; {'OK' if hit else 'miss'} {tail[-90:]!r} {r.stderr[-200:]!r}", flush=True)
    print("   ", chosen[-1][-70:] if chosen else "no choice logged", flush=True)
