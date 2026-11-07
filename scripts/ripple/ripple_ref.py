#!/usr/bin/env python3
"""Reference numbers for Ripple-65B (ricursiveAI/Ripple-65B-Preview) from its own code.

  ripple_ref.py tiny SRC OUT            a small random model with SRC's tokenizer and code, every
                                        feature of the architecture on: for exactness checks
  ripple_ref.py ppl MODEL TEXT T OUT    teacher-forced pass over the first T tokens of TEXT:
                                        OUT.npz with ids, nll, argmax per position and the mean
                                        hidden state after every layer (what ppl_cmp.py reads)

`ppl` on the published model streams the layers that do not fit the card from the safetensors
(RIPPLE_GPU_GIB, default 36). A model that fits is loaded whole; RIPPLE_F32=1 runs it in float32.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def tiny(src, out):
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    cfg.hidden_size = 64
    cfg.intermediate_size = 192
    cfg.num_hidden_layers = 10
    # Ten heads a KV group and a window past the kernels' split point, as the published model has.
    cfg.num_attention_heads = 40
    cfg.num_key_value_heads = 4
    cfg.head_dim = 16
    cfg.sliding_window = 320
    cfg.embed_scale = 8.0
    cfg.max_position_embeddings = 4096
    cfg.layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * 2
    cfg.hybrid_layer_pattern = [1 if t == "sliding_attention" else 0 for t in cfg.layer_types]
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("value_res_lambda"):
                p.normal_(0.0, 2.0)
            elif p.ndim == 1:  # norm weights
                p.uniform_(0.5, 1.5)
            elif "embed_tokens" in name:
                p.normal_(0.0, 0.125)
            elif "mlp.gate_proj" in name or "mlp.up_proj" in name:
                p.normal_(0.0, 1.0)  # pre-activations past the clamp at 20 on some rows
            elif "lm_head" in name:
                p.normal_(0.0, 0.3)
            else:
                p.normal_(0.0, 0.25)
            # f16-exact values: a CMF written with --quant f16 holds the same numbers.
            p.copy_(p.half().float())
    model.save_pretrained(out, safe_serialization=True)
    for f in os.listdir(src):
        if f.endswith((".py", ".jinja")) or f.startswith(("tokenizer", "special_tokens", "added_tokens", "vocab", "merges")):
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
    lam = [float(l.self_attn.value_res_lambda[0]) for l in model.model.layers]
    print("tiny model in", out, "lambda", " ".join(f"{x:.2f}" for x in lam))


def ppl(model_dir, text_path, t, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(open(text_path).read(), add_special_tokens=False)["input_ids"][: t + 1]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    big = cfg["hidden_size"] * cfg["num_hidden_layers"] > 100_000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if big:
        gib = os.environ.get("RIPPLE_GPU_GIB", "36")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: f"{gib}GiB", "cpu": "1GiB"},
            offload_folder=os.path.join(os.path.dirname(out) or ".", "offload"),
        )
        places = {}
        for v in model.hf_device_map.values():
            places[str(v)] = places.get(str(v), 0) + 1
        print("device map:", places, flush=True)
    else:
        dt = torch.float32 if os.environ.get("RIPPLE_F32", "1") == "1" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dt).to(dev)
    model.eval()
    x = torch.tensor([ids[:-1]], device=dev)
    with torch.no_grad():
        o = model(input_ids=x, use_cache=False, output_hidden_states=True)
    logits = o.logits[0].float()
    tgt = torch.tensor(ids[1:], device=logits.device)
    nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="none").cpu().numpy()
    top = logits.argmax(-1).cpu().numpy()
    hid = np.stack([h[0].float().mean(0).cpu().numpy() for h in o.hidden_states])
    last = logits[-1].cpu().numpy()
    np.savez(out, ids=np.array(ids), nll=nll, argmax=top, hidden_mean=hid, last_logits=last)
    print(f"{len(nll)} positions | ppl {float(np.exp(nll.mean())):.4f} | nll {float(nll.mean()):.6f} | into {out}")


if __name__ == "__main__":
    if sys.argv[1] == "tiny":
        tiny(sys.argv[2], sys.argv[3])
    else:
        ppl(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
