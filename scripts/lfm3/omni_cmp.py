"""LFM3-Omni parity against lfm3_omni_ref.py: the greedy image descriptions
and transcripts through `cortiq run` (prompt length, text), the projected
speech rows (CMF_LFM3_AUDIO_DUMP is not on the VL path, so the transcript
stands for them). TTS: tts_cmp.py on the same file.

usage: python3 -I omni_cmp.py REF_DIR MODEL.cmf CORTIQ --image a.jpg [...] --wav a.wav [...]
"""

import json
import os
import subprocess
import sys


def run(cortiq, model, extra):
    out = subprocess.run([cortiq, "run", model, "--greedy", *extra], capture_output=True, text=True,
                         env={**os.environ, "XDG_RUNTIME_DIR": "/tmp", "CMF_PROMPT_DUMP": "1"})
    if out.returncode:
        return None, out.stderr[-800:]
    plen = next((l for l in out.stderr.splitlines() if "rendered prompt" in l), "?")
    text = " ".join(l for l in out.stdout.splitlines()
                    if l.strip() and not l.startswith(("Loading", "sampling", "Ready", "Prompt", "[")))
    return text.strip(), plen


def main():
    args = sys.argv[1:]
    ref_dir, model, cortiq = args[0], args[1], args[2]
    images, wavs, cur = [], [], None
    for a in args[3:]:
        if a in ("--image", "--wav"):
            cur = a
        elif cur == "--image":
            images.append(a)
        else:
            wavs.append(a)
    ref = json.load(open(os.path.join(ref_dir, "ref.json")))
    ref_img = [r for r in ref if "image" in r]
    ref_wav = [r for r in ref if "wav" in r]
    for img, want in zip(images, ref_img):
        text, plen = run(cortiq, model, ["--image", img, "--prompt", "Describe this image in one sentence.",
                                         "--max-tokens", "48"])
        print(f"## {os.path.basename(img)}  {plen} (ref {want['prompt_len']} ids)")
        print(f"  ref   : {want['text']!r}\n  cortiq: {text!r}  {'SAME' if text == want['text'].strip() else ''}")
    for wav, want in zip(wavs, ref_wav):
        text, plen = run(cortiq, model, ["--system", "Perform ASR in English.", "--audio", wav, "--prompt", "",
                                         "--max-tokens", "96"])
        print(f"## {os.path.basename(wav)}  {plen} (ref {want['prompt_len']} ids)")
        print(f"  ref   : {want['text']!r}\n  cortiq: {text!r}  {'SAME' if text == want['text'].strip() else ''}")


if __name__ == "__main__":
    main()
