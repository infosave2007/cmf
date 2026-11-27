#!/usr/bin/env python3
"""Build the benchmark recordings: test-set clips laid end to end (0.4 s of
silence between them) up to each target length, 16 kHz mono PCM16.

    make_bench_clips.py MANIFEST.tsv OUT_DIR [--lengths 10,30,60,300,1800]

Writes OUT_DIR/bench_<seconds>s.wav and OUT_DIR/bench.tsv
(`<file>\t<seconds>\t<reference>`, the reference being the clips' own
transcripts joined by spaces). A length under 30 s stays one segment; longer
ones go through the voice-activity cut.
"""
import argparse, os
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--lengths", default="10,30,60,300,1800")
ap.add_argument("--gap", type=float, default=0.4)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = os.path.dirname(a.manifest)
clips = [l.rstrip("\n").split("\t") for l in open(a.manifest, encoding="utf-8")]
gap = np.zeros(int(a.gap * 16000), dtype=np.float32)
with open(os.path.join(a.out, "bench.tsv"), "w", encoding="utf-8") as man:
    for secs in [float(x) for x in a.lengths.split(",")]:
        parts, texts, total, i = [], [], 0, 0
        target = int(secs * 16000)
        while total < target:
            name, _, ref = clips[i % len(clips)]
            wav, sr = sf.read(os.path.join(base, name), dtype="float32")
            assert sr == 16000
            if total + len(wav) > target and parts:
                break
            parts += [wav, gap]
            texts.append(ref)
            total += len(wav) + len(gap)
            i += 1
        audio = np.concatenate(parts)
        if len(audio) < target:
            audio = np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])
        audio = audio[:target]
        name = f"bench_{int(secs)}s.wav"
        sf.write(os.path.join(a.out, name), audio, 16000, subtype="PCM_16")
        man.write(f"{name}\t{len(audio) / 16000:.3f}\t{' '.join(texts)}\n")
        print(name, f"{len(audio) / 16000:.1f}s", len(texts), "clips")
