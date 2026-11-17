"""Run an open model on the R set, card in context, two worked examples.

    python -I run_model.py --model-dir <dir> --style smollm|boris --tasks r.jsonl \
        --tldr <dir>/tldr-main --out <run dir>/<name>.outputs.jsonl [--limit N] [--device mps|cpu]

Every model sees the same task text and the same two worked examples of the
task's category (none of them from R: the `sleep` and `cal` cards and made-up
tables). Only the wrapping differs:

  smollm  the model's chat template, a one-line system message, the answer
          alone as the assistant turn.
  boris   Boris-2-Think's question/answer form: the assistant turn is
          `<think>…</think>` and a `Final answer:` line, as on its card.
          Generation runs through boris_cached.py (the model's own code has
          no KV cache), checked against the model's forward at start.

Greedy decoding. The answer is extracted here (first line of the reply, or
the `Final answer:` line) and graded by grade.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cards import card  # noqa: E402

SYSTEM = "You are a command-line assistant. Read the reference material and answer exactly as asked, with no explanation."

WIN_TIMEOUT = "[Windows] — timeout: Pause the command processor for a number of seconds.\n- Wait for a number of seconds: timeout /t {{seconds}}"
DEMO_RULES = 'Rules for the next action:\n- If the output contains "disk full", answer CLEAN.\n- Otherwise answer OK.'


def demos(tldr: Path) -> dict[str, list[tuple[str, str, str]]]:
    """category -> [(prompt, thinking, answer)]"""
    sleep, cal = card(tldr, "common/sleep"), card(tldr, "common/cal")
    one = "Answer with one shell command."
    orn = "Answer with one shell command, or with NONE if no card can do the task."
    cmd = (f"Reference card:\n{sleep}\n\nTask: Wait for 30 seconds.\n\n{one}",
           "The matching example is `sleep {{seconds}}`. The task asks for 30 seconds.", "sleep 30")
    cmd2 = (f"Reference card:\n{cal}\n\nTask: Show the calendar of the year 2031.\n\n{one}",
            "The matching example is `cal {{year}}`. The year is 2031.", "cal 2031")
    osd = (f"Reference card:\n[Linux] — {sleep}\n\n{WIN_TIMEOUT}\n\nTask: On Windows, wait for 15 seconds.\n\n{one}",
           "The task is for Windows, so the Windows section applies: `timeout /t {{seconds}}` with 15.", "timeout /t 15")
    pos = (f"Reference cards:\n{sleep}\n\n{cal}\n\nTask: Pause for 5 seconds.\n\n{orn}",
           "A pause is a delay, which is the sleep card: `sleep {{seconds}}` with 5.", "sleep 5")
    none = (f"Reference cards:\n{sleep}\n\n{cal}\n\nTask: Print the kernel version.\n\n{orn}",
            "sleep delays and cal shows calendars. Neither prints the kernel version.", "NONE")
    lab = (f"{DEMO_RULES}\n\nPrevious command: cp big.iso /mnt/usb\nIts output:\ncp: error writing '/mnt/usb/big.iso': disk full\n\n"
           "Task: Decide the next action.\n\nAnswer with one word.", 'The output contains "disk full", so the first rule applies.', "CLEAN")
    lab2 = (f"{DEMO_RULES}\n\nPrevious command: cp a.txt /mnt/usb\nIts output:\n(no output)\n\n"
            "Task: Decide the next action.\n\nAnswer with one word.", 'The output does not contain "disk full", so the last rule applies.', "OK")
    fix = (f"Reference card:\n{sleep}\n\nThe command `sleep` failed with:\nsleep: missing operand\n\n"
           f"Task: The pause must last 10 seconds. Give the corrected command.\n\n{one}",
           "The card's form is `sleep {{seconds}}`; the operand was missing. With 10 seconds it is complete.", "sleep 10")
    two = (f"Reference cards:\n{sleep}\n\n{cal}\n\nTask: Wait 3 seconds, then show the calendar of 2030.\n\n{one}",
           "The sleep card has `sleep 20 && {{command}}`. The delay is 3 and the command is `cal 2030`.", "sleep 3 && cal 2030")
    num = ("Output of `wc -l a.txt b.txt`:\n  10 a.txt\n  25 b.txt\n  35 total\n\nQuestion: How many lines does b.txt have?\nAnswer with a number.",
           "The row for b.txt shows 25.", "25")
    num2 = ("Output of `du -sk in out`:\n120\tin\n300\tout\n\nQuestion: How many KB do in and out take together?\nAnswer with a number.",
            "in is 120 and out is 300. 120 + 300 = 420.", "420")
    name = ("Output of `wc -c a.txt b.txt`:\n  900 a.txt\n  140 b.txt\n 1040 total\n\nQuestion: Which file is smaller?\nAnswer with the file name.",
            "a.txt has 900 bytes and b.txt has 140. 140 is less.", "b.txt")
    return {"slot": [cmd, cmd2], "direct": [cmd, cmd2], "os": [cmd, osd], "choose": [pos, none], "none": [pos, none],
            "react": [lab, lab2], "fix": [cmd, fix], "two": [cmd, two], "arith": [num, num2], "arith.name": [name, num]}


def extract(raw: str, style: str) -> tuple[str, dict]:
    flags = {}
    text = raw.replace("<|im_end|>", "").strip()
    if style == "boris":
        flags["think"] = "<think>" in text
        flags["final"] = "Final answer:" in text
        if flags["final"]:
            text = text.rsplit("Final answer:", 1)[1]
        elif "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    line = next((l.strip() for l in text.strip().splitlines() if l.strip()), "")
    line = re.sub(r"^\$\s+", "", line.strip("`").strip())
    return line, flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--style", required=True, choices=("smollm", "boris"))
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tldr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if args.limit:
        step = max(1, len(tasks) // args.limit)
        tasks = tasks[::step][: args.limit]
    dm = demos(Path(args.tldr))
    trust = args.style == "boris"  # its modeling file was read before use: plain PyTorch
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=trust, dtype=torch.float32)
    model.to(args.device).eval()
    cache = None
    if args.style == "boris":
        from boris_cached import BorisCache, self_check
        cache = BorisCache(model)

    def messages(t):
        key = "arith.name" if t["kind"] == "name" else t["cat"]
        msgs = [{"role": "system", "content": SYSTEM}] if args.style == "smollm" else []
        for p, think, ans in dm[key]:
            msgs.append({"role": "user", "content": p})
            msgs.append({"role": "assistant",
                         "content": ans if args.style == "smollm" else f"<think>\n{think}\n</think>\n\nFinal answer: {ans}"})
        msgs.append({"role": "user", "content": t["prompt"]})
        return msgs

    texts = [tok.apply_chat_template(messages(t), add_generation_prompt=True, tokenize=False) for t in tasks]
    enc = [tok(x, add_special_tokens=False).input_ids for x in texts]
    order = sorted(range(len(tasks)), key=lambda i: len(enc[i]))
    max_new = args.max_new or (220 if args.style == "boris" else 64)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if args.style == "smollm" else set())
    raws: dict[int, str] = {}
    if cache is not None:
        worst, flips = self_check(model, enc[order[0]], 8)
        print(f"boris_cached self-check: max |dlogit| {worst:.2e}, argmax flips {flips}", file=sys.stderr, flush=True)
        if flips:
            raise SystemExit("the cached forward disagrees with the model's own forward")
    t0 = time.time()
    new_tokens = 0
    with torch.inference_mode():
        for s in range(0, len(order), args.batch):
            chunk = order[s:s + args.batch]
            b, t_in = len(chunk), max(len(enc[i]) for i in chunk)
            x = torch.full((b, t_in + max_new), pad, dtype=torch.long)
            mask = torch.zeros((b, t_in + max_new), dtype=torch.long)
            for r, i in enumerate(chunk):
                x[r, t_in - len(enc[i]):t_in] = torch.tensor(enc[i])
                mask[r, t_in - len(enc[i]):t_in] = 1
            x, mask = x.to(args.device), mask.to(args.device)
            if args.style == "smollm":
                y = model.generate(input_ids=x[:, :t_in], attention_mask=mask[:, :t_in], max_new_tokens=max_new,
                                   do_sample=False, pad_token_id=pad)
                for r, i in enumerate(chunk):
                    out = y[r, t_in:].tolist()
                    cut = next((k for k, v in enumerate(out) if v in eos), len(out))
                    raws[i] = tok.decode(out[:cut])
                    new_tokens += cut
            else:
                # Boris-2's own code has no KV cache (every step re-forwards
                # the prefix); boris_cached runs the same arithmetic
                # incrementally, one task at a time.
                for i in chunk:
                    cache.reset()
                    lg = cache.forward(enc[i])
                    out: list[int] = []
                    for _ in range(max_new):
                        nxt = int(lg.argmax())
                        out.append(nxt)
                        new_tokens += 1
                        if nxt in eos or "<|im_end|>" in tok.decode(out[-12:]):
                            break
                        lg = cache.forward([nxt])
                    raws[i] = tok.decode(out).split("<|im_end|>")[0]
            print(f"{min(s + args.batch, len(order))}/{len(order)} tasks, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    with open(args.out, "w") as f:
        for i, t in enumerate(tasks):
            ans, flags = extract(raws[i], args.style)
            f.write(json.dumps({"id": t["id"], "answer": ans, "raw": raws[i], "prompt_tokens": len(enc[i]), **flags},
                               ensure_ascii=False) + "\n")
    print(f"{len(tasks)} tasks, {new_tokens} new tokens, {time.time() - t0:.0f} s, device {args.device}")


if __name__ == "__main__":
    main()
