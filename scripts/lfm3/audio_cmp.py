"""LFM3-Audio parity: every clip of the Transformers reference
(lfm3_audio_ref.py) through `cortiq run --audio` with the encoder dumps on
(CMF_LFM3_AUDIO_DUMP): log-mel features, encoder rows and projected rows
against the reference (max |d|, worst row cosine), the prompt length, and
the greedy transcript.

usage: python3 -I audio_cmp.py REF_DIR MODEL.cmf WAV [WAV ...] [--cortiq PATH]
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def cos_rows(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    return num / den


def main():
    args = sys.argv[1:]
    cortiq = "/root/lfm3/bin/cortiq"
    if "--cortiq" in args:
        i = args.index("--cortiq")
        cortiq = args[i + 1]
        del args[i : i + 2]
    ref_dir, model, wavs = args[0], args[1], args[2:]
    ref = json.load(open(os.path.join(ref_dir, "ref.json")))
    for i, wav in enumerate(wavs):
        want = ref[i]
        with tempfile.TemporaryDirectory() as dump:
            env = {**os.environ, "CMF_LFM3_AUDIO_DUMP": dump, "XDG_RUNTIME_DIR": "/tmp", "CMF_PROMPT_DUMP": "1"}
            out = subprocess.run(
                [cortiq, "run", model, "--system", "Perform ASR in English.", "--audio", wav, "--prompt", "",
                 "--greedy", "--max-tokens", "96"],
                capture_output=True, text=True, env=env)
            if out.returncode:
                print("FAILED:", out.stderr[-1500:])
                continue
            mel = np.fromfile(os.path.join(dump, "mel_0.f32"), dtype=np.float32)
            enc = np.fromfile(os.path.join(dump, "enc_0.f32"), dtype=np.float32)
            rows = np.fromfile(os.path.join(dump, "rows_0.f32"), dtype=np.float32)
        rm = np.load(os.path.join(ref_dir, f"clip{i}_mel.npy"))
        re_ = np.load(os.path.join(ref_dir, f"clip{i}_enc.npy"))
        rr = np.load(os.path.join(ref_dir, f"clip{i}_rows.npy"))
        mel = mel.reshape(rm.shape[0], -1)
        enc = enc.reshape(-1, re_.shape[-1])
        rows = rows.reshape(-1, rr.shape[-1])
        print(f"## {os.path.basename(wav)}")
        print(f"  mel {mel.shape} vs {rm.shape}: max |d| {np.abs(mel - rm).max():.2e}" if mel.shape == rm.shape
              else f"  mel shape {mel.shape} vs {rm.shape}")
        for name, a, b in (("encoder", enc, re_), ("rows", rows, rr)):
            if a.shape != b.shape:
                print(f"  {name} shape {a.shape} vs {b.shape}")
                continue
            c = cos_rows(a, b)
            print(f"  {name} {a.shape}: max |d| {np.abs(a - b).max():.2e}, worst row cos {c.min():.6f}")
        text = out.stdout.strip().splitlines()
        prompt_lines = [l for l in out.stderr.splitlines() if "rendered prompt" in l]
        print(f"  prompt: {prompt_lines[0] if prompt_lines else '?'} (ref {len(want['prompt_ids'])} ids)")
        got = "\n".join(l for l in text if not l.startswith(("Loading", "sampling", "Ready", "Prompt", "[")))
        print(f"  ref   : {want['text']!r}")
        print(f"  cortiq: {got.strip()!r}")


if __name__ == "__main__":
    main()
