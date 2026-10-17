#!/usr/bin/env python3
"""Integration check of the torch model (qmodel.py) against cortiq itself, on tiny random
checkpoints of the supported architectures — no download, a few seconds of CPU.

For each kind (llama with q/k norms + biases off, qwen3_5 hybrid with GDN + gated attention)
a random HF checkpoint is written (Qwen3.5 hub layout: `model.language_model.*`, wrapper
config with `text_config`), converted with `cortiq convert` to f16 / q4tp / q2tp, and the
NLL of the same windows is computed by `cortiq ppl` (CPU exact path, CMF_PPL_PROGRESS
prints each window's exact_nll) and by qmodel.Model. The tokenizer is borrowed from any HF
model directory (only the vocabulary size matters).

  cortiq_parity.py --cortiq target/release/cortiq --tokenizer-dir /path/to/hf/model --work /tmp/parity
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import calib  # noqa: E402
import cmfio  # noqa: E402
import qmodel  # noqa: E402


def vocab_size(tok_dir):
    tj = json.load(open(os.path.join(tok_dir, "tokenizer.json")))
    ids = list(tj["model"]["vocab"].values()) + [t["id"] for t in tj.get("added_tokens", [])]
    return max(ids) + 1


def write_checkpoint(kind, out, tok_dir, seed=0):
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    V = vocab_size(tok_dir)
    V = (V + 31) // 32 * 32
    H, I, nh, nkv, hd = 128, 256, 4, 2, 32
    if kind == "llama":
        types = ["full_attention"] * 2
        tc = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": H,
              "intermediate_size": I, "num_hidden_layers": 2, "num_attention_heads": nh, "num_key_value_heads": nkv,
              "head_dim": hd, "vocab_size": V, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
              "max_position_embeddings": 4096, "tie_word_embeddings": False, "hidden_act": "silu"}
        cfg, pre = tc, "model."
    else:
        types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        tc = {"model_type": "qwen3_5_text", "hidden_size": H, "intermediate_size": I, "num_hidden_layers": 4,
              "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd, "vocab_size": V,
              "max_position_embeddings": 262144, "partial_rotary_factor": 0.25,
              "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
              "linear_conv_kernel_dim": 4, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
              "linear_key_head_dim": 32, "linear_value_head_dim": 32, "layer_types": types,
              "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "hidden_act": "silu"}
        cfg = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
               "tie_word_embeddings": False, "text_config": tc}
        pre = "model.language_model."
    t = {}

    def mat(name, rows, cols, gain=1.0):
        t[name] = (rng.normal(0, gain / np.sqrt(cols), (rows, cols))).astype(np.float32)

    def vec(name, n, center, sd=0.1):
        t[name] = (center + rng.normal(0, sd, n)).astype(np.float32)

    off = 0.0 if kind == "llama" else 1.0     # qwen3_5 norms are zero-centred: stored w, applied 1 + w
    for l, ty in enumerate(types):
        p = f"{pre}layers.{l}."
        vec(p + "input_layernorm.weight", H, 1.0 - off)
        vec(p + "post_attention_layernorm.weight", H, 1.0 - off)
        if ty == "full_attention":
            mat(p + "self_attn.q_proj.weight", nh * hd * (1 if kind == "llama" else 2), H, 2.0)
            mat(p + "self_attn.k_proj.weight", nkv * hd, H, 2.0)
            mat(p + "self_attn.v_proj.weight", nkv * hd, H)
            mat(p + "self_attn.o_proj.weight", H, nh * hd)
            if kind != "llama":
                vec(p + "self_attn.q_norm.weight", hd, 0.0)
                vec(p + "self_attn.k_norm.weight", hd, 0.0)
        else:
            c = 2 * 2 * 32 + 4 * 32
            mat(p + "linear_attn.in_proj_qkv.weight", c, H, 2.0)
            mat(p + "linear_attn.in_proj_z.weight", 4 * 32, H)
            mat(p + "linear_attn.in_proj_b.weight", 4, H)
            mat(p + "linear_attn.in_proj_a.weight", 4, H)
            mat(p + "linear_attn.out_proj.weight", H, 4 * 32)
            t[p + "linear_attn.conv1d.weight"] = rng.normal(0, 0.5, (c, 1, 4)).astype(np.float32)
            t[p + "linear_attn.A_log"] = np.log(rng.uniform(1, 16, 4)).astype(np.float32)
            t[p + "linear_attn.dt_bias"] = rng.normal(0, 0.5, 4).astype(np.float32)
            vec(p + "linear_attn.norm.weight", 32, 1.0)
        mat(p + "mlp.gate_proj.weight", I, H)
        mat(p + "mlp.up_proj.weight", I, H)
        mat(p + "mlp.down_proj.weight", H, I)
    vec(pre + "norm.weight", H, 1.0 - off)
    t[pre + "embed_tokens.weight"] = rng.normal(0, 1.0, (V, H)).astype(np.float32)
    mat("lm_head.weight", V, H, 6.0)
    os.makedirs(out, exist_ok=True)
    # bf16 storage like real checkpoints (the converter's f16 is then exact)
    t = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in t.items()}
    from safetensors.torch import save_file as save_t
    save_t(t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    for f in os.listdir(tok_dir):
        if f.startswith("tokenizer") or f in ("special_tokens_map.json", "vocab.json", "merges.txt"):
            shutil.copy(os.path.join(tok_dir, f), out)
    return out


def cortiq_window_nll(cortiq, model, text, n, wl, ids_out):
    env = dict(os.environ, CMF_GPU="0", CMF_PPL_PROGRESS="1", CMF_PPL_IDS_OUT=ids_out)
    r = subprocess.run([cortiq, "ppl", model, "--file", text, "--windows", str(n), "--window-len", str(wl)],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    out = (r.stdout + r.stderr).replace("\r", "\n")
    cum = [float(x) for x in re.findall(r"exact_nll=([0-9.]+)", out)]   # running sum over windows
    return list(np.diff([0.0] + cum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--text", default=None, help="text to score (default: this repo's README.md)")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--wl", type=int, default=160)
    ap.add_argument("--kinds", nargs="+", default=["llama", "qwen3_5"])
    ap.add_argument("--quants", nargs="+", default=["f16", "q4tp", "q2tp"])
    ap.add_argument("--refit", action="store_true", help="also run metric_refit.py end to end on the tiny files")
    args = ap.parse_args()
    text = args.text or os.path.join(HERE, "..", "..", "..", "README.md")
    qmodel.setup_torch(0, tf32=False)
    worst = 0.0
    for kind in args.kinds:
        hf = write_checkpoint(kind, os.path.join(args.work, kind, "hf"), args.tokenizer_dir)
        for q in args.quants:
            cmf = os.path.join(args.work, kind, f"{q}.cmf")
            r = subprocess.run([args.cortiq, "convert", "--model", hf, "--quant", q, "--output", cmf],
                               capture_output=True, text=True, env=dict(os.environ, CMF_GPU="0"))
            if r.returncode:
                raise RuntimeError(r.stderr[-3000:])
            ids_p = os.path.join(args.work, kind, "ids.txt")
            ref = cortiq_window_nll(args.cortiq, cmf, text, args.windows, args.wl, ids_p)
            ids = cmfio.load_ids(ids_p)
            w = calib.gather(ids, calib.even_starts(len(ids), args.windows, args.wl), args.wl)
            m = cmfio.CmfFile(cmf)
            dts = sorted({t.dtype for t in m.tensors.values()})
            mine = qmodel.Model(m, "cpu").evaluate(w, mb=1)["nll"]
            rel = np.abs(np.array(mine) - np.array(ref)) / np.array(ref)
            worst = max(worst, float(rel.max()))
            print(f"{kind:8s} {q:5s} dtypes {dts}\n   cortiq {np.round(ref, 4).tolist()}\n   torch  "
                  f"{np.round(mine, 4).tolist()}\n   max rel diff {rel.max():.2e} "
                  f"(per-token NLL {np.mean(ref) / (args.wl - 1):.3f})", flush=True)
    print(f"worst relative NLL difference {worst:.2e}")
    ok = worst < 1e-4
    if args.refit:
        ok &= refit_chain(args, text)
    return 0 if ok else 1


def refit_chain(args, text):
    """metric_refit.py end to end on the real tiny files: q4tp / q2tp student, f16 teacher, write-back through
    cortiq patch-tensor --raw + verify; then the written file must (a) differ from its source only in the f16
    fields, (b) score better than the source on the training windows, (c) score identically in torch and cortiq."""
    import metric_refit
    good = True
    for kind in args.kinds:
        d = os.path.join(args.work, kind)
        ids_p = os.path.join(d, "ids.txt")
        for q in [x for x in args.quants if x != "f16"]:
            out = os.path.join(d, f"refit_{q}")
            dst = os.path.join(d, f"{q}-genesis.cmf")
            man = metric_refit.main(["--model", os.path.join(d, f"{q}.cmf"), "--teacher", os.path.join(d, "f16.cmf"),
                                     "--corpus", f"{ids_p},train=12,held=0:3", "--out", out, "--wl", "64",
                                     "--bs", "4", "--mb", "2", "--eval-mb", "4", "--epochs", "4", "--eval-every", "3",
                                     "--step", "--lr", "3e-3", "--device", "cpu", "--write", dst, "--cortiq", args.cortiq,
                                     "--topk", "16"])
            src, new = cmfio.CmfFile(os.path.join(d, f"{q}.cmf")), cmfio.CmfFile(dst)
            stray = 0
            for n, t in src.tensors.items():
                a, b = src.payload(n), new.payload(n)
                if np.array_equal(a, b):
                    continue
                if t.dtype in qmodel.TILED:
                    import cmfcodec as cc
                    nb = cc.Tiled(t.dtype, a, *t.shape).nb
                    diff = np.nonzero(a != b)[0]
                    stray += int(((diff < nb) | (diff >= nb + 4 * t.shape[0])).sum())
                elif not (len(t.shape) == 1 and t.dtype in qmodel.FLOATS):
                    stray += 1
            ids = cmfio.load_ids(ids_p)
            w = calib.gather(ids, calib.even_starts(len(ids), 2, 64), 64)
            ref = cortiq_window_nll(args.cortiq, dst, text, 2, 64, os.path.join(d, "ids2.txt"))
            mine = qmodel.Model(new, "cpu").evaluate(w, mb=1)["nll"]
            rel = float(np.max(np.abs(np.array(mine) - np.array(ref)) / np.array(ref)))
            gain = man["hist"][0]["held_ce"] - man["best_held_ce"]
            this = stray == 0 and gain > 0 and rel < 1e-4
            good &= this
            print(f"{kind:8s} {q:5s} genesis: held-out CE {man['hist'][0]['held_ce']:.4f} -> {man['best_held_ce']:.4f} "
                  f"(snapped {man['held_ce_snapped']:.4f}), {man['n_payloads']} payloads, bytes outside f16 fields "
                  f"{stray}, written file torch vs cortiq max rel {rel:.1e} -> {'ok' if this else 'FAIL'}", flush=True)
    return good


if __name__ == "__main__":
    sys.exit(main())
