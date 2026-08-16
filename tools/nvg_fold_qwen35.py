#!/usr/bin/env python3
"""
NVG holographic fold of a Qwen3.5/3.8 hybrid (GDN + full attention) checkpoint —
FFN width, attention q-heads and GDN value heads — in closed form, from
activation Grams collected on a calibration mix. See docs/NVG_FOLD_ATTN_GDN.ru.md.

    W' = W[:, S] + W[:, P] · Πᵀ,   Π = (Σ_SS + εI)⁻¹ Σ_SP

for the linear read-out of each block (down_proj / o_proj / out_proj); the
non-linear part of every kept unit is untouched. Streams the bf16 shards into
RAM (no full-checkpoint disk footprint), runs the calibration forward on CPU
(or a device), folds, writes a same-schema safetensors checkpoint + config.

usage: nvg_fold_qwen35.py --model Qwen/Qwen3.8-27B --out /dev/shm/qwen38-fold \
         --calib calib.txt --calib-tokens 30000 --ffn 12288 --attn-heads 16 --gdn-v-heads 32
"""
import argparse, json, os, sys, time, math, gc, re
# This is a CPU job: keep transformers on its own torch GDN implementation.
# With flash-linear-attention importable it picks fla's kernels, whose CPU
# fallback measured 213 s a sequence against 60 for the torch path.
sys.modules["fla"] = None
import numpy as np
import torch

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_calib(path, tok, n_tokens, seq):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ids = ids[: n_tokens]
    n = len(ids) // seq
    arr = torch.tensor(ids[: n * seq], dtype=torch.long).view(n, seq)
    log(f"calibration: {len(ids)} tokens → {n} sequences of {seq}")
    return arr

class Gram:
    """f32 Gram accumulator Σ = Σ_t a_t a_tᵀ over the last dim; count of rows."""
    def __init__(self, dim, dtype=torch.float32):
        self.g = torch.zeros(dim, dim, dtype=dtype)
        self.n = 0
    def add(self, a):
        a = a.reshape(-1, a.shape[-1]).to(torch.float32)
        self.g.addmm_(a.t(), a)
        self.n += a.shape[0]

def solve_pi(S_full, keep, drop, eps_rel):
    """Π = (Σ_SS + εI)⁻¹ Σ_SP in float64. Returns Π (|S|×|P|) and the
    relative regression residual tr(Σ_PP − Σ_PS Π)/tr(Σ_PP)."""
    Sd = S_full.double()
    Sss = Sd[keep][:, keep]
    Ssp = Sd[keep][:, drop]
    Spp_tr = Sd[drop, drop].sum().item()
    eps = eps_rel * Sss.diagonal().mean().item()
    A = Sss + eps * torch.eye(Sss.shape[0], dtype=torch.float64)
    L = torch.linalg.cholesky(A)
    Pi = torch.cholesky_solve(Ssp, L)  # (|S|×|P|)
    resid = (Spp_tr - (Ssp * Pi).sum().item()) / max(Spp_tr, 1e-30)
    return Pi, resid

def keep_by_groups(imp, groups, per_group):
    """imp: [n_units]; groups: n_units//groups units per group, contiguous;
    keep the top `per_group` of each group. Returns sorted keep, drop."""
    n = imp.shape[0]
    gsz = n // groups
    keep = []
    for g in range(groups):
        seg = imp[g * gsz:(g + 1) * gsz]
        top = torch.topk(seg, per_group).indices + g * gsz
        keep.extend(top.tolist())
    keep = sorted(keep)
    drop = sorted(set(range(n)) - set(keep))
    return keep, drop

def expand(units, width):
    """unit indices → flat channel indices (each unit is `width` channels)."""
    return [u * width + d for u in units for d in range(width)]

def fold_layers(args, model, sd, put, grams, report, layer_ids, layer_types, F, nh, nkv, hd, nv, nk, dk, dv, qpg, vpg, keep_qpg, keep_vpg):
    for i in layer_ids:
        layer = model.model.layers[i]
        pre = f"model.layers.{i}."
        # copy the untouched pieces first
        for k in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            put(pre + k, sd[pre + k])
        # FFN
        Wg, Wu, Wd = sd[pre + "mlp.gate_proj.weight"], sd[pre + "mlp.up_proj.weight"], sd[pre + "mlp.down_proj.weight"]
        if not args.no_ffn and args.ffn < F:
            G = grams[f"ffn.{i}"].g
            diag = G.diagonal().double() / max(grams[f"ffn.{i}"].n, 1)
            wnorm = Wd.float().pow(2).sum(0).double()  # ‖W_down[:,j]‖²
            imp = diag * wnorm  # Born: energy the neuron delivers to the stream
            keep = torch.topk(imp, args.ffn).indices.sort().values.tolist()
            drop = sorted(set(range(F)) - set(keep))
            Pi, resid = solve_pi(G, keep, drop, args.eps)
            Wd_new = Wd.float()[:, keep].double() + Wd.float()[:, drop].double() @ Pi.t()
            # output-weighted residual: ‖W_P (a_P − Πᵀa_S)‖² / ‖W a‖² (trace form).
            # f32 GEMMs: the f64 einsum over the 17408² Gram was 3 TFLOP a
            # layer and made the fold slower than the calibration.
            Gf = G / max(grams[f"ffn.{i}"].n, 1)
            Wp = Wd.float()[:, drop]
            Rpp = (Gf[drop][:, drop].double() - Gf[drop][:, keep].double() @ Pi).float()
            num = ((Wp @ Rpp) * Wp).sum().item()
            Wf = Wd.float()
            den = ((Wf @ Gf) * Wf).sum().item()
            put(pre + "mlp.gate_proj.weight", Wg[keep])
            put(pre + "mlp.up_proj.weight", Wu[keep])
            put(pre + "mlp.down_proj.weight", Wd_new)
            report.append((i, "ffn", len(keep), resid, num / max(den, 1e-30)))
            log(f"L{i:02d} ffn  keep {len(keep)}/{F}  resid {resid:.4f}  out-resid {num/max(den,1e-30):.4f}")
        else:
            put(pre + "mlp.gate_proj.weight", Wg); put(pre + "mlp.up_proj.weight", Wu); put(pre + "mlp.down_proj.weight", Wd)
        # attention
        if layer_types[i] == "full_attention":
            Wq, Wk, Wv, Wo = (sd[pre + f"self_attn.{n}.weight"] for n in ["q_proj", "k_proj", "v_proj", "o_proj"])
            put(pre + "self_attn.k_proj.weight", Wk); put(pre + "self_attn.v_proj.weight", Wv)
            put(pre + "self_attn.q_norm.weight", sd[pre + "self_attn.q_norm.weight"])
            put(pre + "self_attn.k_norm.weight", sd[pre + "self_attn.k_norm.weight"])
            if not args.no_attn and keep_qpg < qpg:
                G = grams[f"attn.{i}"].g
                diag = G.diagonal().double() / max(grams[f"attn.{i}"].n, 1)
                wn = Wo.float().pow(2).sum(0).double()
                imp_ch = diag * wn
                imp_head = imp_ch.view(nh, hd).sum(1)
                kh, dh = keep_by_groups(imp_head, nkv, keep_qpg)
                keep, drop = expand(kh, hd), expand(dh, hd)
                Pi, resid = solve_pi(G, keep, drop, args.eps)
                Wo_new = Wo.float()[:, keep].double() + Wo.float()[:, drop].double() @ Pi.t()
                # q_proj rows: per head [q(hd) | gate(hd)] contiguous
                qrows = [h * 2 * hd + d for h in kh for d in range(2 * hd)]
                put(pre + "self_attn.q_proj.weight", Wq[qrows])
                put(pre + "self_attn.o_proj.weight", Wo_new)
                report.append((i, "attn", len(kh), resid, float("nan")))
                log(f"L{i:02d} attn keep heads {kh}  resid {resid:.4f}")
            else:
                put(pre + "self_attn.q_proj.weight", Wq); put(pre + "self_attn.o_proj.weight", Wo)
        else:
            la = "linear_attn."
            names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
            W = {n: sd[pre + la + n + ".weight"] for n in names}
            conv = sd[pre + la + "conv1d.weight"]
            A_log, dt_bias, normw = sd[pre + la + "A_log"], sd[pre + la + "dt_bias"], sd[pre + la + "norm.weight"]
            put(pre + la + "norm.weight", normw)
            if not args.no_gdn and keep_vpg < vpg:
                G = grams[f"gdn.{i}"].g
                diag = G.diagonal().double() / max(grams[f"gdn.{i}"].n, 1)
                wn = W["out_proj"].float().pow(2).sum(0).double()
                imp_head = (diag * wn).view(nv, dv).sum(1)
                kh, dh = keep_by_groups(imp_head, nk, keep_vpg)
                keep, drop = expand(kh, dv), expand(dh, dv)
                Pi, resid = solve_pi(G, keep, drop, args.eps)
                Wout_new = W["out_proj"].float()[:, keep].double() + W["out_proj"].float()[:, drop].double() @ Pi.t()
                key_dim = nk * dk
                qk_rows = list(range(2 * key_dim))
                v_rows = [2 * key_dim + c for c in keep]
                put(pre + la + "in_proj_qkv.weight", W["in_proj_qkv"][qk_rows + v_rows])
                put(pre + la + "in_proj_z.weight", W["in_proj_z"][keep])
                put(pre + la + "in_proj_b.weight", W["in_proj_b"][kh])
                put(pre + la + "in_proj_a.weight", W["in_proj_a"][kh])
                put(pre + la + "conv1d.weight", conv[qk_rows + v_rows])
                put(pre + la + "A_log", A_log[kh]); put(pre + la + "dt_bias", dt_bias[kh])
                put(pre + la + "out_proj.weight", Wout_new)
                report.append((i, "gdn", len(kh), resid, float("nan")))
                log(f"L{i:02d} gdn  keep v-heads {len(kh)}/{nv}  resid {resid:.4f}")
            else:
                for n in names: put(pre + la + n + ".weight", W[n])
                put(pre + la + "conv1d.weight", conv); put(pre + la + "A_log", A_log); put(pre + la + "dt_bias", dt_bias)
        grams.pop(f"ffn.{i}", None); grams.pop(f"attn.{i}", None); grams.pop(f"gdn.{i}", None)
        gc.collect()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib-tokens", type=int, default=30000)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=12288)
    ap.add_argument("--attn-heads", type=int, default=16)
    ap.add_argument("--gdn-v-heads", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-3, help="ridge, relative to mean diag")
    ap.add_argument("--shard-dir", default="/dev/shm/hfshards")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=60)
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--no-gdn", action="store_true")
    ap.add_argument("--no-ffn", action="store_true")
    ap.add_argument("--dry-layers", type=int, default=0, help="debug: only run N layers of calibration")
    ap.add_argument("--groups", type=int, default=3, help="calibration passes: the layers' Grams are collected in this many groups (RAM: model + one group's Grams; a 27B FFN Gram is 1.2 GB f32 per layer)")
    ap.add_argument("--shard-dirs", default="", help="comma-separated directories; group g's shards are written to dirs[g % n] and symlinked into --out (a box whose filesystems are each too small for the whole output)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    cfg_all = AutoConfig.from_pretrained(args.model)
    tc = cfg_all.text_config if hasattr(cfg_all, "text_config") else cfg_all
    H = tc.hidden_size; L = tc.num_hidden_layers; F = tc.intermediate_size
    nh, nkv, hd = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    nv, nk, dk, dv = tc.linear_num_value_heads, tc.linear_num_key_heads, tc.linear_key_head_dim, tc.linear_value_head_dim
    layer_types = list(tc.layer_types)
    log(f"{args.model}: H={H} L={L} F={F} attn {nh}/{nkv}x{hd} gdn nv={nv} nk={nk} {dk}/{dv}")
    assert nh % nkv == 0 and nv % nk == 0
    qpg, vpg = nh // nkv, nv // nk
    assert args.attn_heads % nkv == 0 and args.gdn_v_heads % nk == 0
    keep_qpg, keep_vpg = args.attn_heads // nkv, args.gdn_v_heads // nk
    assert keep_qpg <= qpg and keep_vpg <= vpg and args.ffn <= F

    # ── build the text model in RAM (bf16), stream the shards ──
    tok = AutoTokenizer.from_pretrained(args.model)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
    torch.set_default_dtype(torch.float32)
    model = model.to_empty(device="cpu")
    # rotary buffers are computed in __init__ on meta → recompute
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn"):
            inv_freq, scaling = m.rope_init_fn(m.config, "cpu")
            m.inv_freq = inv_freq; m.original_inv_freq = inv_freq
    index = json.load(open(hf_hub_download(args.model, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    os.makedirs(args.shard_dir, exist_ok=True)
    sd_model = model.state_dict()
    loaded = set()
    mtp_tensors = {}
    for sh in shards:
        t0 = time.time()
        p = hf_hub_download(args.model, sh, local_dir=args.shard_dir)
        sd = load_file(p)
        for k, v in sd.items():
            if k.startswith("mtp."):
                mtp_tensors[k] = v.clone(); continue
            if k.startswith("model.visual"):
                continue
            k2 = k.replace("model.language_model.", "model.")
            if k2 in sd_model:
                sd_model[k2].copy_(v.to(sd_model[k2].dtype)); loaded.add(k2)
            else:
                log("  unmatched key:", k)
        del sd
        os.remove(p)
        log(f"shard {sh}: {time.time()-t0:.0f}s, loaded {len(loaded)}/{len(sd_model)}")
    missing = [k for k in sd_model if k not in loaded]
    log("missing after load:", missing[:10], len(missing))
    model.eval()
    gc.collect()

    calib = load_calib(args.calib, tok, args.calib_tokens, args.seq)
    if args.dry_layers:
        model.model.layers = model.model.layers[: args.dry_layers]
    nL = len(model.model.layers)
    groups = [list(range(g * nL // args.groups, (g + 1) * nL // args.groups)) for g in range(args.groups)]
    report = []
    sd = model.state_dict()  # views into the live params
    new_sd = {}
    def put(k, v):
        new_sd[k] = v.detach().to(torch.bfloat16).contiguous().clone()
    grams = {}
    def mk_hook(name):
        def h(mod, inp):
            grams[name].add(inp[0].detach())
        return h
    # shard writer: folded tensors leave RAM as soon as a group is done
    # (the frozen model stays for the next group's calibration pass)
    os.makedirs(args.out, exist_ok=True)
    weight_map = {}
    shard_i = [0]
    total_params = [0]
    shard_dirs = [d for d in args.shard_dirs.split(",") if d]
    for d in shard_dirs: os.makedirs(d, exist_ok=True)
    flush_n = [0]
    def flush_shards(sdict):
        cur = {}; cur_b = 0
        tgt = shard_dirs[flush_n[0] % len(shard_dirs)] if shard_dirs else args.out
        flush_n[0] += 1
        def emit():
            nonlocal cur, cur_b
            if not cur: return
            name = f"model-{shard_i[0]:05d}.safetensors"
            path = os.path.join(tgt, name)
            save_file(cur, path, metadata={"format": "pt"})
            if tgt != args.out:
                link = os.path.join(args.out, name)
                if os.path.lexists(link): os.remove(link)
                os.symlink(path, link)
            for kk in cur: weight_map[kk] = name
            shard_i[0] += 1; cur = {}; cur_b = 0
        for k, v in sdict.items():
            k2 = k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k
            b = v.numel() * v.element_size()
            total_params[0] += v.numel()
            if cur_b + b > 4e9 and cur: emit()
            cur[k2] = v; cur_b += b
        emit()
        sdict.clear()
    for gi, layer_ids in enumerate(groups):
        # ── hooks: Grams of the read-out inputs, this group's layers only ──
        handles = []
        for i in layer_ids:
            layer = model.model.layers[i]
            if not args.no_ffn:
                grams[f"ffn.{i}"] = Gram(F)
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(mk_hook(f"ffn.{i}")))
            if layer_types[i] == "full_attention" and not args.no_attn:
                grams[f"attn.{i}"] = Gram(nh * hd)
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk_hook(f"attn.{i}")))
            if layer_types[i] != "full_attention" and not args.no_gdn:
                grams[f"gdn.{i}"] = Gram(nv * dv)
                handles.append(layer.linear_attn.out_proj.register_forward_pre_hook(mk_hook(f"gdn.{i}")))
        t0 = time.time()
        with torch.no_grad():
            for si in range(calib.shape[0]):
                ids = calib[si:si + 1]
                model.model(input_ids=ids)  # no lm_head: the Grams are what we want
                if si == 0 or si % 4 == 3:
                    el = time.time() - t0
                    log(f"group {gi+1}/{len(groups)} layers {layer_ids[0]}..{layer_ids[-1]}: calib seq {si+1}/{calib.shape[0]}  {el:.0f}s  ({el/(si+1):.1f}s/seq)")
        for hnd in handles: hnd.remove()
        log(f"group {gi+1} calibration done in {time.time()-t0:.0f}s")
        fold_layers(args, model, sd, put, grams, report, layer_ids, layer_types, F, nh, nkv, hd, nv, nk, dk, dv, qpg, vpg, keep_qpg, keep_vpg)
        grams.clear(); gc.collect()
        flush_shards(new_sd); gc.collect()
        log(f"group {gi+1} written ({shard_i[0]} shards so far)")

    put("model.embed_tokens.weight", sd["model.embed_tokens.weight"])
    put("model.norm.weight", sd["model.norm.weight"])
    put("lm_head.weight", sd["lm_head.weight"])
    flush_shards(new_sd)

    # ── index + config ──
    total = total_params[0]
    total_bytes = total * 2
    json.dump({"metadata": {"total_size": int(total_bytes)}, "weight_map": weight_map},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)
    # config: source config with the folded dims, MTP off (its tensors are not written)
    cfg = json.load(open(hf_hub_download(args.model, "config.json")))
    tcj = cfg["text_config"] if "text_config" in cfg else cfg
    tcj["intermediate_size"] = args.ffn if not args.no_ffn else F
    tcj["num_attention_heads"] = args.attn_heads if not args.no_attn else nh
    tcj["linear_num_value_heads"] = args.gdn_v_heads if not args.no_gdn else nv
    tcj["mtp_num_hidden_layers"] = 0
    tcj["nvg_fold"] = {"source": args.model, "calib_tokens": int(calib.numel()), "eps": args.eps,
                       "report": [[i, kind, n, r, o] for (i, kind, n, r, o) in report]}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "special_tokens_map.json", "vocab.json", "merges.txt"]:
        try:
            p = hf_hub_download(args.model, f)
            import shutil; shutil.copy(p, os.path.join(args.out, f))
        except Exception:
            pass
    log(f"written {args.out}: {total/1e9:.2f}B params, {shard_i[0]} shards")
    log("fold report (layer, kind, kept, resid, out-resid):")
    for r in report: log("  ", r)

if __name__ == "__main__":
    main()
