#!/usr/bin/env python3
"""Extract clips of a parquet speech test set (HF `audio` + `transcription`
columns) as 16 kHz mono WAV files plus a manifest.

    extract_clips.py test.parquet OUT_DIR [--limit N] [--prefix far]

The manifest (OUT_DIR/manifest.tsv) holds `<file>\t<seconds>\t<reference>`.
"""
import argparse, io, os, sys
import pyarrow.parquet as pq
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("parquet", nargs="+")
ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--prefix", default="clip")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
n = 0
with open(os.path.join(a.out, "manifest.tsv"), "a", encoding="utf-8") as man:
    for path in a.parquet:
        pf = pq.ParquetFile(path)
        text_col = next(c for c in ("transcription", "sentence", "text") if c in pf.schema.names)
        for batch in pf.iter_batches(batch_size=64, columns=["audio", text_col]):
            for audio, text in zip(batch.column("audio").to_pylist(), batch.column(text_col).to_pylist()):
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
                if sr != 16000:
                    raise SystemExit(f"{path}: sample rate {sr}, expected 16000")
                name = f"{a.prefix}_{n:05d}.wav"
                sf.write(os.path.join(a.out, name), wav, 16000, subtype="PCM_16")
                man.write(f"{name}\t{len(wav) / 16000:.3f}\t{(text or '').strip()}\n")
                n += 1
                if a.limit and n >= a.limit:
                    print(f"{n} clips -> {a.out}", file=sys.stderr)
                    raise SystemExit(0)
print(f"{n} clips -> {a.out}", file=sys.stderr)
