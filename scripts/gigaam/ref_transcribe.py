#!/usr/bin/env python3
"""Reference GigaAM-v4 run through the authors' `modeling_gigaam.py`
(Transformers, trust_remote_code): transcripts of a clip list as JSONL, and
optionally the intermediate tensors of one clip for the layer parity test.

    ref_transcribe.py MODEL_DIR MANIFEST_OR_WAVS... --out ref.jsonl
        [--device cpu|cuda] [--fp16] [--longform] [--dump DIR]

A manifest is a `.tsv` of `<file>\t<seconds>\t<reference>` beside its clips.
`--dump DIR` writes float32 `.npy` files for the FIRST clip: `features`
(mel, [64, frames]), `frontend` ([T, 768]), `stack{0,1,2}`, `encoded`
([T, 768]), and per head the token ids and frames.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("inputs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--longform", action="store_true")
ap.add_argument("--words", action="store_true")
ap.add_argument("--dump")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

clips = []
for p in a.inputs:
    if p.endswith(".tsv"):
        base = os.path.dirname(p)
        for line in open(p, encoding="utf-8"):
            name, secs, ref = line.rstrip("\n").split("\t")
            clips.append((os.path.join(base, name), float(secs), ref))
    else:
        clips.append((p, 0.0, ""))
if a.limit:
    clips = clips[: a.limit]

t0 = time.time()
model = AutoModel.from_pretrained(
    a.model, trust_remote_code=True, device=a.device, fp16_encoder=a.fp16
).eval()
print(f"load {time.time() - t0:.1f}s on {a.device}", file=sys.stderr)
core = model.model

if a.dump:
    os.makedirs(a.dump, exist_ok=True)
    wav, length = core.prepare_wav(clips[0][0])
    saved = {}
    enc = core.encoder
    hooks = [
        enc.encoder_embed.register_forward_hook(lambda m, i, o: saved.__setitem__("frontend", o[0][0])),
    ]
    for k, stack in enumerate(enc.encoder.encoders):
        hooks.append(stack.register_forward_hook(lambda m, i, o, k=k: saved.__setitem__(f"stack{k}", o[0])))
        hooks.append(stack.layers[0].register_forward_hook(lambda m, i, o, k=k: saved.__setitem__(f"stack{k}_layer0", o[0])))
    with torch.inference_mode():
        feats, flen = core.preprocessor(wav, length)
        encoded, elen = core.forward(wav, length)
        dec = core._decode(encoded, elen, length, True)
        raw = {name: d.decode(core.heads[name], encoded.float(), elen) for name, d in core.decodings.items()}
    for h in hooks:
        h.remove()
    np.save(os.path.join(a.dump, "samples.npy"), wav[0].float().cpu().numpy())
    np.save(os.path.join(a.dump, "features.npy"), feats[0].float().cpu().numpy())
    np.save(os.path.join(a.dump, "encoded.npy"), encoded[0].float().cpu().numpy().T.copy())
    for k, v in saved.items():
        np.save(os.path.join(a.dump, f"{k}.npy"), v.float().cpu().numpy())
    meta = {"clip": clips[0][0], "frames": int(flen[0]), "encoded_len": int(elen[0]), "heads": {}}
    for name, items in raw.items():
        text, ids, frames = items[0]
        meta["heads"][name] = {"text": text, "ids": ids, "frames": frames}
        if hasattr(core.heads[name], "decoder_layers"):
            with torch.inference_mode():
                lp = core.heads[name](encoded.float())
            np.save(os.path.join(a.dump, f"ctc_{name}.npy"), lp[0].float().cpu().numpy())
    json.dump(meta, open(os.path.join(a.dump, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"dump -> {a.dump}", file=sys.stderr)

total_audio = total_time = 0.0
with open(a.out, "w", encoding="utf-8") as out:
    for i, (path, secs, ref) in enumerate(clips):
        t1 = time.time()
        if a.longform:
            r = model.transcribe_longform(path, word_timestamps=a.words)
            row = {name: v.text for name, v in r.items()}
            row["segments"] = [[s.start, s.end] for s in r["e2e"].segments]
        else:
            try:
                r = model.transcribe(path, word_timestamps=a.words)
            except ValueError:  # over 30 s: the package's own long-form path
                r = model.transcribe_longform(path)
            row = {name: v.text for name, v in r.items()}
            if a.words and hasattr(r["e2e"], "words") and not hasattr(r["e2e"], "segments"):
                row["words"] = {name: [[w.text, round(w.start, 3), round(w.end, 3)] for w in v.words] for name, v in r.items()}
        dt = time.time() - t1
        total_audio += secs
        total_time += dt
        row.update(file=os.path.basename(path), seconds=secs, ref=ref, time=round(dt, 4))
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(clips)} clips, {total_audio / max(total_time, 1e-9):.1f}x real time", file=sys.stderr)
print(f"{len(clips)} clips, audio {total_audio:.1f}s in {total_time:.1f}s", file=sys.stderr)
