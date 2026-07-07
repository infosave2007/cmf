#!/usr/bin/env python3
"""Generate a deterministic tiny Qwen-like checkpoint for CMF tests.

Creates in --out:
  config.json, weights.npz, tokenizer.json, masks/general.json, masks/coding.json

The tiny arch (2 layers, hidden 32) is real enough to exercise every
format path: GQA shapes, norms as f16, masks with quality contract.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HID, INTER, LAYERS, HEADS, KV, HEAD_DIM, VOCAB = 32, 64, 2, 4, 2, 8, 128

PRESETS = {
    # (hid, inter, layers, heads, kv, head_dim, vocab)
    "tiny": (32, 64, 2, 4, 2, 8, 128),
    # ~50M params — big enough for meaningful CPU benchmarks
    "medium": (512, 1408, 8, 8, 2, 64, 8192),
}


def main(out_dir: str, seed: int = 42, preset: str = "tiny"):
    global HID, INTER, LAYERS, HEADS, KV, HEAD_DIM, VOCAB
    HID, INTER, LAYERS, HEADS, KV, HEAD_DIM, VOCAB = PRESETS[preset]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    config = {
        "model_type": "tiny-qwen",
        "hidden_size": HID,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "layer_types": ["full_attention"] * LAYERS,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights = {
        "model.embed_tokens.weight": w(VOCAB, HID, scale=0.1),
        "model.norm.weight": np.ones(HID, np.float32) + w(HID)[0] * 0,
        "lm_head.weight": w(VOCAB, HID, scale=0.1),
    }
    def layer_block(prefix: str):
        weights[prefix + "input_layernorm.weight"] = (1.0 + rng.standard_normal(HID) * 0.02).astype(np.float32)
        weights[prefix + "post_attention_layernorm.weight"] = (1.0 + rng.standard_normal(HID) * 0.02).astype(np.float32)
        weights[prefix + "self_attn.q_proj.weight"] = w(HEADS * HEAD_DIM, HID)
        weights[prefix + "self_attn.k_proj.weight"] = w(KV * HEAD_DIM, HID)
        weights[prefix + "self_attn.v_proj.weight"] = w(KV * HEAD_DIM, HID)
        weights[prefix + "self_attn.o_proj.weight"] = w(HID, HEADS * HEAD_DIM)
        weights[prefix + "mlp.gate_proj.weight"] = w(INTER, HID)
        weights[prefix + "mlp.up_proj.weight"] = w(INTER, HID)
        weights[prefix + "mlp.down_proj.weight"] = w(HID, INTER)

    for li in range(LAYERS):
        layer_block(f"model.layers.{li}.")

    # MTP head (DeepSeek/Qwen style, spec §2.1): shared embed + lm_head.
    weights["model.mtp.enorm.weight"] = np.ones(HID, np.float32)
    weights["model.mtp.hnorm.weight"] = np.ones(HID, np.float32)
    weights["model.mtp.eh_proj.weight"] = w(HID, 2 * HID)
    layer_block("model.mtp.layers.0.")
    weights["model.mtp.norm.weight"] = np.ones(HID, np.float32)

    np.savez(out / "weights.npz", **weights)

    # Minimal tokenizer: full printable ASCII (32..126) + Ġ/Ċ marks.
    vocab = {chr(32 + i): i for i in range(95)}  # ' '..'~' → 0..94
    vocab["Ġ"] = 95  # space-prefix mark (GPT-style)
    vocab["Ċ"] = 96  # newline mark
    for i in range(97, VOCAB - 2):
        vocab[f"<unused{i}>"] = i
    vocab["<|endoftext|>"] = VOCAB - 2
    vocab["<pad>"] = VOCAB - 1
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {"id": VOCAB - 2, "content": "<|endoftext|>", "special": True},
            {"id": VOCAB - 1, "content": "<pad>", "special": True},
        ],
    }
    (out / "tokenizer.json").write_text(json.dumps(tokenizer))

    masks_dir = out / "masks"
    masks_dir.mkdir(exist_ok=True)
    # general: everything active, measured quality
    (masks_dir / "0_general.json").write_text(json.dumps({
        "ffn_masks": [[1.0] * INTER for _ in range(LAYERS)],
        "head_masks": [[1.0] * HEADS for _ in range(LAYERS)],
        "layer_gates": [True] * LAYERS,
        "metadata": {
            "task_name": "general", "sparsity": 0.0,
            "description": "full network",
            "quality": {"metric": "heldout_ppl_ratio", "value": 1.0,
                        "baseline_dense": 1.0, "n_samples": 8},
        },
    }))
    # coding: half the neurons, 3 of 4 heads, quality NOT measured
    (masks_dir / "1_coding.json").write_text(json.dumps({
        "ffn_masks": [[1.0 if i % 2 == 0 else 0.0 for i in range(INTER)]
                      for _ in range(LAYERS)],
        "head_masks": [[1.0, 1.0, 1.0, 0.0] for _ in range(LAYERS)],
        "layer_gates": [True] * LAYERS,
        "metadata": {"task_name": "coding", "sparsity": 0.5,
                     "description": "half core, unmeasured"},
    }))

    print(f"tiny model written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", default="tiny", choices=list(PRESETS))
    a = p.parse_args()
    main(a.out, a.seed, a.preset)
