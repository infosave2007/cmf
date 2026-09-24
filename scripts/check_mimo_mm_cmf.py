#!/usr/bin/env python3
"""MiMo-V2 tower gate (milestone M1: G1.1 names/counts, G1.2 dtypes + exactness).

Checks a companion `<stem>.mm.cmf` (`cortiq convert --mimo-towers mm-only`)
or the tower part of a single-file multimodal CMF against the SOURCE
checkpoint, independently of the Rust converter:

  G1.1  the tower tensors are exactly the source's visual.* (364),
        audio_encoder.* + speech_embeddings.* (95) and
        audio_tokenizer/model.safetensors encoder.* minus the codebook training
        state (389), the latter under `audio_tokenizer.`; nothing from
        decoder.*, cluster_size, embed_avg or inited; plus the two config
        blobs, byte-equal to config.json and audio_tokenizer/config.json.
        A companion carries nothing else.
  G1.2  2-D tower matrices have the expected codec (--matrices q4tp|q8_2f|exact,
        or 'any'); speech_embeddings and every rank!=2 tensor are the source
        bytes verbatim (BF16 in the release: F16 cannot hold every BF16 value,
        see --f16-census); RVQ codebooks F32 and byte-equal to the source.
        With --matrices exact every matrix is checked byte-equal as well.
  --f16-census  counts the source values of the kept BF16 tensors that do
        NOT survive BF16 -> F16 -> F32 (why F16 is not the exact codec).

Reads only the directory, the header and the payloads of float tensors
(numpy). Header: arch_name mimo_v2_mm, base_arch mimo_v2, hidden_size, the
nine pinned special ids, config sha256.

  python3 scripts/check_mimo_mm_cmf.py --model out/x.mm.cmf --source /root/mimo/src \
      --matrices q4tp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_dsv41_cmf import CheckError, parse_artifact  # noqa: E402

TOWER_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.")
AT_PREFIX = "audio_tokenizer."
AT_DROPPED = ("._codebook.cluster_size", "._codebook.embed_avg", "._codebook.inited")
BLOBS = ("mm.config_json", "audio_tokenizer.config_json")
PINNED = {
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<|audio_pad|>": 151669,
    "<|mimo_video_start|>": 151670,
    "<|mimo_video_end|>": 151671,
    "<|mimo_audio_start|>": 151673,
    "<|mimo_audio_end|>": 151674,
}
RELEASE_COUNTS = {"visual": 364, "audio_encoder+speech_embeddings": 95, "audio_tokenizer": 389}


def st_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_bytes(path: Path, meta: dict, base: int) -> bytes:
    a, b = meta["data_offsets"]
    with path.open("rb") as f:
        f.seek(base + a)
        return f.read(b - a)


def group(name: str) -> str:
    if name.startswith("visual."):
        return "visual"
    if name.startswith(("audio_encoder.", "speech_embeddings.")):
        return "audio_encoder+speech_embeddings"
    if name.startswith(AT_PREFIX + "encoder."):
        return "audio_tokenizer"
    return "other"


def is_codebook(name: str) -> bool:
    return name.startswith(AT_PREFIX + "encoder.quantizer.") and name.endswith("._codebook.embed")


def source_inventory(src: Path) -> dict[str, tuple[Path, dict, int]]:
    """CMF name -> (source file, safetensors meta, data base)."""
    out: dict[str, tuple[Path, dict, int]] = {}
    index = src / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        files = sorted({f for k, f in wm.items() if k.startswith(TOWER_PREFIXES)})
    else:
        files = ["model.safetensors"]
    for f in files:
        h, base = st_header(src / f)
        for k, m in h.items():
            if k != "__metadata__" and k.startswith(TOWER_PREFIXES):
                out[k] = (src / f, m, base)
    at = src / "audio_tokenizer" / "model.safetensors"
    h, base = st_header(at)
    dropped = Counter()
    for k, m in h.items():
        if k == "__metadata__":
            continue
        if k.startswith("decoder."):
            dropped["decoder"] += 1
        elif k.endswith(AT_DROPPED):
            dropped["codebook_state"] += 1
        elif k.startswith("encoder."):
            out[AT_PREFIX + k] = (at, m, base)
        else:
            raise CheckError(f"audio tokenizer tensor {k!r} is neither encoder.* nor decoder.*")
    print(f"source: {len(out)} tower tensors kept, dropped {dict(dropped)}")
    return out


def cmf_payload(model: Path, data_off: int, e) -> bytes:
    with model.open("rb") as f:
        f.seek(data_off + e.off)
        return f.read(e.nbytes)


def as_f32(raw: bytes, dtype: str) -> np.ndarray:
    if dtype in ("F32", "f32"):
        return np.frombuffer(raw, dtype="<f4")
    if dtype in ("F16", "f16"):
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype in ("BF16", "bf16"):
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32)
    raise CheckError(f"no float view for {dtype}")


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path, help="HF checkpoint dir (with audio_tokenizer/)")
    ap.add_argument("--matrices", default="q4tp",
                    help="expected dtype of 2-D tower matrices: q4tp, q8_2f, exact (= source bytes) or any")
    ap.add_argument("--f16-census", action="store_true",
                    help="count BF16 source values that F16 cannot hold exactly")
    ap.add_argument("--release", action="store_true", help="also require the release counts 364/95/389")
    args = ap.parse_args(argv)

    errors: list[str] = []
    art = parse_artifact(args.model)
    data_off = art.sections["data"][0]
    hdr = art.header
    entries = {e.name: e for e in art.entries}
    companion = hdr["arch"]["arch_name"] == "mimo_v2_mm"
    prov = (hdr.get("provenance") or {}).get("mimo_mm") or {}

    # Header.
    if hdr["arch"]["arch_name"] not in ("mimo_v2_mm", "mimo_v2"):
        errors.append(f"arch_name {hdr['arch']['arch_name']!r}")
    if prov.get("base_arch") != "mimo_v2":
        errors.append(f"provenance.mimo_mm.base_arch {prov.get('base_arch')!r}")
    cfg_bytes = (args.source / "config.json").read_bytes()
    at_cfg_bytes = (args.source / "audio_tokenizer" / "config.json").read_bytes()
    cfg = json.loads(cfg_bytes)
    if hdr["arch"]["hidden_size"] != cfg["hidden_size"]:
        errors.append(f"hidden_size {hdr['arch']['hidden_size']} != source {cfg['hidden_size']}")
    if prov.get("special_tokens") != PINNED:
        errors.append(f"special_tokens {prov.get('special_tokens')}")
    if prov.get("config_sha256") != hashlib.sha256(cfg_bytes).hexdigest():
        errors.append("config_sha256 does not match source config.json")
    if prov.get("audio_tokenizer_config_sha256") != hashlib.sha256(at_cfg_bytes).hexdigest():
        errors.append("audio_tokenizer_config_sha256 does not match")

    # G1.1 names.
    src = source_inventory(args.source)
    tower_names = {n for n in entries if group(n) != "other"}
    missing = sorted(set(src) - tower_names)
    extra = sorted(tower_names - set(src))
    if missing:
        errors.append(f"G1.1: {len(missing)} source tower tensors missing, e.g. {missing[:3]}")
    if extra:
        errors.append(f"G1.1: {len(extra)} tower tensors not in the source, e.g. {extra[:3]}")
    bad_drop = [n for n in entries if n.startswith(AT_PREFIX + "decoder.") or n.endswith(AT_DROPPED)]
    if bad_drop:
        errors.append(f"G1.1: dropped tensors present: {bad_drop[:3]}")
    for b, want in zip(BLOBS, (cfg_bytes, at_cfg_bytes)):
        e = entries.get(b)
        if e is None or e.dtype != "u8" or cmf_payload(args.model, data_off, e) != want:
            errors.append(f"G1.1: blob {b} missing or not byte-equal to the source file")
    if companion:
        others = sorted(n for n in entries if group(n) == "other" and n not in BLOBS)
        if others:
            errors.append(f"G1.1: companion carries non-tower tensors {others[:3]}")
    counts = Counter(group(n) for n in tower_names)
    print(f"G1.1 counts: {dict(counts)} + {sum(b in entries for b in BLOBS)} blobs "
          f"(total CMF tensors {len(entries)})")
    if args.release:
        for g, n in RELEASE_COUNTS.items():
            if counts.get(g) != n:
                errors.append(f"G1.1: {g} has {counts.get(g)} tensors, release has {n}")

    # G1.2 dtypes + exactness.
    hist: dict[str, Counter] = {}
    exact_checked = Counter()
    nbytes = Counter()
    census = Counter()
    for n in sorted(tower_names & set(src)):
        e = entries[n]
        path, meta, base = src[n]
        shape = tuple(meta["shape"])
        if tuple(e.shape) != shape:
            errors.append(f"G1.2: {n} shape {e.shape} != source {shape}")
            continue
        g = group(n)
        nbytes[g] += e.nbytes
        src_dt = meta["dtype"].lower()
        if is_codebook(n):
            cat, want = "codebooks", "f32"
        elif n.startswith("speech_embeddings."):
            cat, want = "tables", src_dt
        elif len(shape) != 2:
            cat, want = "other", src_dt
        else:
            cat = "matrices"
            want = {"exact": src_dt, "any": None}.get(args.matrices, args.matrices)
        hist.setdefault(f"{g}/{cat}", Counter())[e.dtype] += 1
        if want is not None and e.dtype != want:
            errors.append(f"G1.2: {n} is {e.dtype}, expected {want}")
            continue
        raw = None
        if e.dtype in ("f16", "bf16", "f32"):
            raw = st_bytes(path, meta, base)
            cm = cmf_payload(args.model, data_off, e)
            if e.dtype == src_dt:
                ok = cm == raw
            else:  # e.g. an F32 codebook from a BF16 source
                ok = bit_equal(as_f32(cm, e.dtype), as_f32(raw, meta["dtype"]))
            if ok:
                exact_checked[e.dtype] += 1
            else:
                errors.append(f"G1.2: {n} {e.dtype} is not the exact {meta['dtype']} source")
        if args.f16_census and meta["dtype"] == "BF16":
            raw = raw if raw is not None else st_bytes(path, meta, base)
            v = as_f32(raw, "BF16")
            back = v.astype(np.float16).astype(np.float32)
            census["values"] += v.size
            census["inexact"] += int((back.view(np.uint32) != v.view(np.uint32)).sum())
    for k in sorted(hist):
        print(f"G1.2 {k}: {dict(hist[k])}")
    print(f"G1.2 byte-exact vs source: {dict(exact_checked)}")
    if args.f16_census:
        print(f"F16 census over kept BF16 tensors: {census['inexact']} of {census['values']} values "
              f"change under BF16->F16->F32")
    print("payload bytes per group: " + ", ".join(f"{g} {v / 1e6:.1f} MB" for g, v in sorted(nbytes.items())))
    print(f"file size: {art.size} bytes ({art.size / 1e9:.3f} GB)")
    if errors:
        for e in errors[:40]:
            print("FAIL", e)
        print(f"{len(errors)} error(s)")
        return 1
    print("PASS G1.1 G1.2")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
