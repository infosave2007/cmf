"""Chat calibration text for `cortiq quantize-moe`: UltraChat-200k
conversations rendered through the checkpoint's own chat template, one
after another, until OUT holds about MAX_CHARS characters.

usage: python3 -I render_chat.py MODEL_DIR ULTRACHAT.parquet OUT.txt [MAX_CHARS]
"""

import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    model_dir, parquet, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_chars = int(sys.argv[4]) if len(sys.argv) > 4 else 2_000_000
    tok = AutoTokenizer.from_pretrained(model_dir)
    rows = pq.read_table(parquet, columns=["messages"]).column("messages").to_pylist()
    total, n = 0, 0
    with open(out_path, "w") as f:
        for messages in rows:
            msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
            if not msgs or msgs[0]["role"] != "user":
                continue
            text = tok.apply_chat_template(msgs, tokenize=False)
            f.write(text)
            total += len(text)
            n += 1
            if total >= max_chars:
                break
    print(f"{n} conversations, {total} chars -> {out_path}")


if __name__ == "__main__":
    main()
