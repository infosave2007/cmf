#!/usr/bin/env python3
"""Teacher-forced perplexity of a stock mlx-lm model under the shared window protocol:
WINDOWS windows of LEN tokens placed at a fixed STRIDE from the start of the raw token stream
(no BOS), the LEN-1 next-token predictions of every window summed before the exp — the same
tokens `cortiq ppl --windows W --window-len L --window-stride S` and
`llama-perplexity -c L --ppl-stride S --chunks W` score.

mlx_ppl.py MODEL_DIR TEXT_FILE [WINDOWS=64] [LEN=512] [STRIDE=511] [IDS_OUT] [SCORE_FROM=0]

SCORE_FROM scores each window's predictions from that position on, the earlier ones being
context only (256 with LEN 512 and STRIDE 512 = `llama-perplexity -c 512`, which scores the
second half of every chunk; `cortiq ppl --window-score-from 256`).

The model directory's own code runs (trust_remote_code); IDS_OUT gets the token ids, space
separated, for the tokenizer parity check against the other engines."""
import math
import sys
import time

import mlx.core as mx
from mlx_lm import load


def main():
    model_dir, text_file = sys.argv[1], sys.argv[2]
    windows = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    wlen = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else wlen - 1
    ids_out = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    score_from = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    model, tok = load(model_dir, trust_remote_code=True)
    text = open(text_file, encoding="utf-8").read()
    ids = tok.encode(text, add_special_tokens=False)
    if ids_out:
        open(ids_out, "w").write(" ".join(map(str, ids)))
    need = (windows - 1) * stride + wlen
    if len(ids) < need:
        sys.exit(f"corpus has {len(ids)} tokens, {windows} windows need {need}")
    nll, cnt = 0.0, 0
    t0 = time.time()
    for w in range(windows):
        seg = ids[w * stride : w * stride + wlen]
        x = mx.array(seg)[None]
        logits = model(x)[0].astype(mx.float32)
        lg = logits[score_from:-1]
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array(seg[score_from + 1 :])
        nll -= mx.take_along_axis(lp, tgt[:, None], axis=-1).sum().item()
        cnt += wlen - 1 - score_from
        mx.clear_cache()
    tail = f", from position {score_from}" if score_from else ""
    print(f"windows: {windows} x {wlen} tokens at stride {stride} ({cnt} scored{tail})")
    print(f"PPL = {math.exp(nll / cnt):.3f} (mlx {model_dir}, {time.time() - t0:.0f} s, peak {mx.get_peak_memory() / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
