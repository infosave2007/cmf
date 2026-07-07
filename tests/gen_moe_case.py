#!/usr/bin/env python3
"""Tiny random Qwen2-MoE / Qwen3-MoE checkpoint + golden reference.

The reference (prompt logits + greedy continuation) comes from
transformers itself — the same harness contract as gen_reference.py:
{quant, prompt_ids, greedy_ids, first_logits} for golden_parity.rs.

qwen2: shared expert + sigmoid gate, qkv-bias, norm_topk_prob=False,
       decoder_sparse_step=2 → alternating dense/MoE layers.
qwen3: no shared expert, qk-norm, norm_topk_prob=True, all layers MoE.
qwen3_next: QWEN3.5-MOE ARCHITECTURE (AgentWorld) — GatedDeltaNet
       linear-attention + full-attention alternation + MoE with shared
       expert + gemma norms; the newest combination, the main gate.
"""
import argparse
import json
from pathlib import Path

import torch

VOCAB = 128
PROMPT = [3, 17, 42, 9, 88, 21]
GREEDY_STEPS = 8
# Greedy parity is only honest when argmax is not hanging by a thread:
# require a top1-top2 margin at every step, otherwise regenerate with a seed.
MIN_MARGIN = 1e-2


def build(family: str, seed: int):
    torch.manual_seed(seed)
    common = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96,
        moe_intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, mlp_only_layers=[],
        max_position_embeddings=256, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False,
    )
    if family == "qwen2":
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
        cfg = Qwen2MoeConfig(
            shared_expert_intermediate_size=80, decoder_sparse_step=2,
            norm_topk_prob=False, **common)
        model = Qwen2MoeForCausalLM(cfg)
    elif family == "qwen3":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        cfg = Qwen3MoeConfig(
            head_dim=16, decoder_sparse_step=1, norm_topk_prob=True,
            **common)
        model = Qwen3MoeForCausalLM(cfg)
    elif family == "qwen3_next":
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        cfg = Qwen3NextConfig(
            head_dim=16, partial_rotary_factor=0.25,
            layer_types=["linear_attention", "full_attention",
                         "linear_attention", "full_attention"],
            linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            shared_expert_intermediate_size=80,
            decoder_sparse_step=1, norm_topk_prob=True,
            **common)
        model = Qwen3NextForCausalLM(cfg)
    else:
        raise SystemExit(f"unknown family {family}")
    # Norms 1.0 and bias 0.0 after init are trivial — perturb them so
    # parity actually exercises these paths.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.3)
            elif n.endswith(".bias"):
                p.add_(torch.randn_like(p) * 0.1)
    return model.float().eval()


def reference(model):
    ids = list(PROMPT)
    with torch.no_grad():
        first = model(torch.tensor([ids])).logits[0, -1]
        greedy, ok = [], True
        for _ in range(GREEDY_STEPS):
            lg = model(torch.tensor([ids])).logits[0, -1]
            top2 = torch.topk(lg, 2).values
            if float(top2[0] - top2[1]) < MIN_MARGIN:
                ok = False
                break
            nxt = int(lg.argmax())
            greedy.append(nxt)
            ids.append(nxt)
    return first, greedy, ok


def tiny_tokenizer(out: Path):
    vocab = {chr(32 + i): i for i in range(95)}
    vocab["Ġ"], vocab["Ċ"] = 95, 96
    for i in range(97, VOCAB - 2):
        vocab[f"<unused{i}>"] = i
    vocab["<|endoftext|>"] = VOCAB - 2
    vocab["<pad>"] = VOCAB - 1
    (out / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {"id": VOCAB - 2, "content": "<|endoftext|>", "special": True},
            {"id": VOCAB - 1, "content": "<pad>", "special": True},
        ],
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["qwen2", "qwen3", "qwen3_next"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--quant", default="F32")
    a = ap.parse_args()

    for seed in range(20):
        model = build(a.family, seed)
        first, greedy, ok = reference(model)
        if ok:
            break
    else:
        raise SystemExit("no seed found with a confident greedy margin")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tiny_tokenizer(out)
    json.dump({
        "quant": a.quant,
        "prompt_ids": PROMPT,
        "greedy_ids": greedy,
        "first_logits": [float(v) for v in first],
    }, open(a.ref, "w"))
    print(f"  {a.family}-moe seed={seed}: greedy={greedy}")


if __name__ == "__main__":
    main()
