#!/usr/bin/env python3
"""
Heal data + teacher targets for a KD heal of a folded/quantized Qwen3.5/3.8:
  1. builds a general token stream (wikitext-103 train + ultrachat + local code/docs),
  2. runs the bf16 teacher layer-streamed on one GPU (the fcd_tail_heal machinery),
  3. writes tokens.pt ([N, seq] int64) and targets.pt (top-k log-probs + ids, f16/int32).
usage: heal_data_teacher.py --teacher Qwen/Qwen3.8-27B --out /dev/shm/heal --tokens 1500000 --seq 1024
"""
import argparse, os, sys, time, json, math, glob, random
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcd_tail_heal import HFShards, make_layer, run_layers, log

def build_text(tok, n_tokens, extra_dirs):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    parts = []
    # wikitext-103 train (raw): ~60% of the budget
    p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    wt = "".join(df["text"].tolist()[:400000])
    parts.append(("wikitext103", wt))
    # ultrachat: chat turns rendered plainly (~25%)
    try:
        p = hf_hub_download("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        chats = []
        for msgs in df["messages"].tolist()[:6000]:
            s = ""
            for m in msgs:
                role = m["role"]; s += f"<|im_start|>{role}\n{m['content']}<|im_end|>\n"
            chats.append(s)
        parts.append(("ultrachat", "\n".join(chats)))
    except Exception as e:
        log("ultrachat unavailable:", e)
    # local code/docs (~15%)
    code = ""
    for d in extra_dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.*"), recursive=True))[:400]:
            if f.split(".")[-1] in ("rs", "py", "md", "js", "html", "toml", "wgsl", "ts"):
                try:
                    code += open(f, encoding="utf-8", errors="ignore").read()[:20000] + "\n\n"
                except Exception:
                    pass
    parts.append(("code", code))
    # tokenize each part, take proportional shares, interleave in blocks
    ids_all = []
    shares = {"wikitext103": 0.55, "ultrachat": 0.30, "code": 0.15}
    for name, text in parts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        want = int(n_tokens * shares.get(name, 0.1))
        ids = ids[:want]
        log(f"{name}: {len(ids)} tokens")
        ids_all.append(ids)
    return ids_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1500000)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32, help="sequences per teacher pass (VRAM: hiddens + logits)")
    ap.add_argument("--extra-dirs", default="/root/cmf/crates,/root/cmf/docs,/root/cmf/tools")
    ap.add_argument("--tmp", default="/dev/shm/healtmp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    cfg_all = AutoConfig.from_pretrained(args.teacher)
    tc = cfg_all.text_config if hasattr(cfg_all, "text_config") else cfg_all
    tc._attn_implementation = "sdpa"
    L, H = tc.num_hidden_layers, tc.hidden_size
    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok_path = os.path.join(args.out, "tokens.pt")
    if os.path.exists(tok_path):
        tokens = torch.load(tok_path)
        log(f"tokens reused: {tokens.shape}")
    else:
        parts = build_text(tok, args.tokens, args.extra_dirs.split(","))
        # sequences: chunk each part, shuffle sequences together
        seqs = []
        for ids in parts:
            n = len(ids) // args.seq
            for i in range(n):
                seqs.append(ids[i * args.seq:(i + 1) * args.seq])
        random.seed(0); random.shuffle(seqs)
        tokens = torch.tensor(seqs, dtype=torch.long)
        torch.save(tokens, tok_path)
        log(f"tokens: {tokens.shape} ({tokens.numel()} tokens)")
    N, T = tokens.shape
    rotary = Qwen3_5TextRotaryEmbedding(config=tc).to(dev)
    hf = HFShards(args.teacher, os.path.join(args.tmp, "hf"))
    P = "model.language_model."
    tk = args.topk
    tgt_v = torch.empty(N, T, tk, dtype=torch.float16)
    tgt_i = torch.empty(N, T, tk, dtype=torch.int32)
    # the teacher passes: each pass streams the whole checkpoint once over `batch` sequences.
    # To stream the checkpoint ONCE for all sequences, keep every batch's hidden resident:
    # N×T×H bf16 = 1.5M×5120×2 = 15 GB — fits the 5090 next to a layer; do it in two halves if not.
    per_pass = args.batch
    passes = list(range(0, N, per_pass))
    t_emb = hf.tensor(P + "embed_tokens.weight")
    log(f"{N} sequences → {len(passes)} passes of ≤{per_pass}")
    # ONE stream of layers, all sequences resident (in chunks of `per_pass` for the attention memory)
    h_list = [F.embedding(tokens[s:s + per_pass].to(dev), t_emb.to(dev)).to(torch.bfloat16) for s in passes]
    del t_emb
    pending = {}
    def teacher_layers():
        for sh_name in hf.shards:
            sd = hf.load_shard(sh_name)
            pending.update({k: v for k, v in sd.items() if k.startswith(P + "layers.")})
            del sd
            while True:
                i = teacher_layers.next_i
                keys = [k for k in pending if k.startswith(f"{P}layers.{i}.")]
                if not keys:
                    break
                from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
                with torch.device("meta"):
                    n_own = len(Qwen3_5DecoderLayer(tc, i).state_dict())
                if len(keys) < n_own:
                    break
                sd_l = {k.replace(P, "model."): pending.pop(k) for k in keys}
                yield i, make_layer(tc, i, sd_l, dev, torch.bfloat16, f"model.layers.{i}.")
                teacher_layers.next_i += 1
    teacher_layers.next_i = 0
    t0 = time.time()
    h_list = run_layers(tc, rotary, teacher_layers(), h_list, dev, 0, L - 1, tag="teacher")
    log(f"teacher layers: {time.time()-t0:.0f}s")
    t_norm = hf.tensor(P + "norm.weight").to(dev).float()
    t_head = hf.tensor("lm_head.weight").to(dev)
    nll, cnt = 0.0, 0
    for pi, s in enumerate(passes):
        h = h_list[pi]
        Bt = h.shape[0]
        hh = h.reshape(-1, H)
        ids_flat = tokens[s:s + Bt].reshape(-1).to(dev)
        for c in range(0, hh.shape[0], 2048):
            x = hh[c:c + 2048].float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps) * (1.0 + t_norm)
            lg = (x.to(torch.bfloat16) @ t_head.t()).float()
            lp = torch.log_softmax(lg, -1)
            v, ix = lp.topk(tk, dim=-1)
            r0 = c
            rows = torch.arange(r0, r0 + lg.shape[0], device=dev)
            b_idx = rows // T; t_idx = rows % T
            tgt_v[s + b_idx.cpu(), t_idx.cpu()] = v.half().cpu()
            tgt_i[s + b_idx.cpu(), t_idx.cpu()] = ix.int().cpu()
            valid = t_idx < (T - 1)
            nxt = ids_flat[(rows + 1).clamp(max=ids_flat.numel() - 1)]
            nll += -(lp[torch.arange(lg.shape[0], device=dev), nxt][valid]).sum().item(); cnt += valid.sum().item()
        h_list[pi] = None
        torch.cuda.empty_cache()
    log(f"teacher ppl on the heal set: {math.exp(nll / max(cnt, 1)):.3f}")
    torch.save({"v": tgt_v, "i": tgt_i, "topk": tk}, os.path.join(args.out, "targets.pt"))
    log(f"saved {args.out}/targets.pt ({tgt_v.numel()*2/1e9:.2f} GB + ids)")

if __name__ == "__main__":
    main()
