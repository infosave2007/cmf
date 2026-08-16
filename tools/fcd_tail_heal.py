#!/usr/bin/env python3
"""
FCD tail heal for a QUANTIZED CMF model: keep the file's own numerics for
layers 0..K-1 (dequantized through `cortiq dequant`), train the last
layers (default: the last 2) + final norm against the bf16 original as
teacher (0.3·CE + 0.7·KL on the teacher's top-k), and export the healed
tail at f16 into a copy of the .cmf (`cortiq patch-tensor`).

Layer-streamed on ONE GPU: every layer's weights visit the device once,
the activations of every calibration/eval token stay resident (~400 MB),
so a 27B needs neither the model in VRAM nor the model in RAM.

usage: fcd_tail_heal.py --cmf model.cmf --teacher Qwen/Qwen3.8-27B \
         --calib calib.txt --eval wikitext2_test.txt --out /dev/shm/fcd-q4tp \
         --layers 62,63 --steps 200 --lr 2e-5
"""
import argparse, json, os, sys, time, gc, subprocess, math, shutil
import numpy as np
import torch
import torch.nn.functional as F

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-2000:]}")
    return r.stdout

def read_bf16(path, shape):
    a = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)

def cmf_tensor_list(cortiq, cmf, prefix):
    out = sh(f"{cortiq} info {cmf} --tensors '{prefix}' 2>/dev/null")
    res = []
    for ln in out.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        name, dtype, shape, nbytes = ln.split("\t")
        res.append((name, dtype, json.loads(shape)))
    return res

def cmf_dequant_prefix(cortiq, cmf, prefix, tmpdir):
    """Dequantize every tensor under `prefix` (bf16 files) → dict name→tensor."""
    os.makedirs(tmpdir, exist_ok=True)
    sh(f"{cortiq} dequant {cmf} --name '{prefix}' --out {tmpdir} --dtype bf16 --all 2>/dev/null")
    res = {}
    for name, dtype, shape in cmf_tensor_list(cortiq, cmf, prefix):
        p = os.path.join(tmpdir, name + ".bin")
        res[name] = read_bf16(p, shape).clone()
        os.remove(p)
    return res

class HFShards:
    """Streams the bf16 checkpoint shard by shard; yields complete layers in order."""
    def __init__(self, repo, tmpdir):
        from huggingface_hub import hf_hub_download
        self.repo, self.tmpdir, self.dl = repo, tmpdir, hf_hub_download
        os.makedirs(tmpdir, exist_ok=True)
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self.shards = sorted(set(self.wm.values()))
    def load_shard(self, sh_name):
        from safetensors.torch import load_file
        p = self.dl(self.repo, sh_name, local_dir=self.tmpdir)
        sd = load_file(p)
        os.remove(p)
        return sd
    def tensor(self, key):
        sd = self.load_shard(self.wm[key])
        return sd[key]

def make_layer(cfg, i, sd_layer, device, dtype, prefix):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(cfg, i)
    layer = layer.to_empty(device=device)
    own = layer.state_dict()
    for k in own:
        src = sd_layer[prefix + k]
        own[k].copy_(src.to(device=device, dtype=own[k].dtype))
    return layer.to(dtype).eval()

def run_layers(cfg, rotary, layers_iter, h_list, device, first_layer, last_layer, tag=""):
    """h_list: list of [B,T,H] tensors (bf16, on device); layers_iter yields (i, layer)."""
    seen = 0
    for i, layer in layers_iter:
        if i < first_layer or i > last_layer:
            continue
        seen += 1
        for bi, h in enumerate(h_list):
            B, T, _ = h.shape
            pos = torch.arange(T, device=device).view(1, 1, -1).expand(4, B, -1)
            text_pos, pos_ids = pos[0], pos[1:]
            pe = rotary(h, pos_ids)
            with torch.no_grad():
                out = layer(h, position_embeddings=pe, attention_mask=None, position_ids=text_pos)
            h_list[bi] = out if torch.is_tensor(out) else out[0]
        if i in (0, 1, 2, 3, 15, 31, 47, 61, 62, 63):
            log(f"  {tag} after layer {i}: mean|h| {h_list[0].float().abs().mean().item():.4f}  rms {h_list[0].float().pow(2).mean().sqrt().item():.3f}")
        del layer
        torch.cuda.empty_cache()
    log(f"  {tag}: {seen} layers run")
    return h_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmf", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--cortiq", default="cortiq")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--eval", required=True, help="wikitext-2 test text; first 12 windows of 512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="62,63")
    ap.add_argument("--calib-tokens", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--eval-windows", type=int, default=12)
    ap.add_argument("--eval-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--kl", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--val-seqs", type=int, default=4)
    ap.add_argument("--tmp", default="/dev/shm/fcdtmp")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--reuse-hstud", action="store_true", help="load h_stud.pt from --out instead of running the student trunk")
    ap.add_argument("--reuse-teacher", action="store_true", help="load teacher.pt (targets + bf16 tail) from --out")
    ap.add_argument("--bf16-train", action="store_true", help="bf16 trainable params + bitsandbytes AdamW8bit (a 4-layer tail in fp32/Adam32 does not fit 32 GB)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm
    cfg_all = AutoConfig.from_pretrained(args.teacher)
    tc = cfg_all.text_config if hasattr(cfg_all, "text_config") else cfg_all
    tc._attn_implementation = "sdpa"
    L, H, V = tc.num_hidden_layers, tc.hidden_size, tc.vocab_size
    train_layers = [int(x) for x in args.layers.split(",")]
    K = min(train_layers)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    rotary = Qwen3_5TextRotaryEmbedding(config=tc).to(dev)

    # ── tokens ──
    ids = tok(open(args.calib, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"][: args.calib_tokens]
    n = len(ids) // args.seq
    calib = torch.tensor(ids[: n * args.seq]).view(n, args.seq)
    eids = tok(open(args.eval, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"]
    ne = args.eval_windows
    ev = torch.tensor(eids[: ne * args.eval_len]).view(ne, args.eval_len)
    log(f"calib {calib.shape}  eval {ev.shape}")
    batches = [calib.to(dev), ev.to(dev)]  # [B,T] each

    # ── student frozen trunk: layers 0..K-1 from the CMF's own numerics ──
    stud_dir = os.path.join(args.tmp, "stud")
    emb = cmf_dequant_prefix(args.cortiq, args.cmf, "model.embed_tokens.weight", stud_dir)["model.embed_tokens.weight"]
    def embed(ids_b, table):
        return F.embedding(ids_b, table.to(dev)).to(torch.bfloat16)
    hs_path = os.path.join(args.out, "h_stud.pt")
    if args.reuse_hstud and os.path.exists(hs_path):
        h_stud = [h.to(dev) for h in torch.load(hs_path)]
        log(f"student trunk: reused {hs_path}")
        del emb
    else:
        h_stud = [embed(b, emb) for b in batches]
        del emb
        t0 = time.time()
        def student_layers():
            for i in range(K):
                sd_l = cmf_dequant_prefix(args.cortiq, args.cmf, f"model.layers.{i}.", stud_dir)
                yield i, make_layer(tc, i, sd_l, dev, torch.bfloat16, f"model.layers.{i}.")
                if i % 8 == 7:
                    log(f"student trunk: layer {i} done ({time.time()-t0:.0f}s)")
        h_stud = run_layers(tc, rotary, student_layers(), h_stud, dev, 0, K - 1, tag="student")
        torch.save([h.cpu() for h in h_stud], hs_path)
        log(f"student trunk done: {time.time()-t0:.0f}s")
    # the student's own tail + norm + lm_head (quantized numerics), for the baseline and the FCD init
    stud_tail = {}
    for i in train_layers:
        stud_tail.update(cmf_dequant_prefix(args.cortiq, args.cmf, f"model.layers.{i}.", stud_dir))
    stud_tail.update(cmf_dequant_prefix(args.cortiq, args.cmf, "model.norm.weight", stud_dir))
    lm_head = cmf_dequant_prefix(args.cortiq, args.cmf, "lm_head.weight", stud_dir)["lm_head.weight"].to(dev)

    # ── teacher: bf16 checkpoint, layer-streamed; top-k targets ──
    hf = HFShards(args.teacher, os.path.join(args.tmp, "hf"))
    P = "model.language_model."
    tk = args.topk
    teach_path = os.path.join(args.out, "teacher.pt")
    if args.reuse_teacher and os.path.exists(teach_path):
        tp_ = torch.load(teach_path)
        targets = [(v.to(dev), i.to(dev)) for v, i in tp_["targets"]]
        tail_bf16 = tp_["tail_bf16"]
        log(f"teacher: reused {teach_path}")
    else:
      t_emb = hf.tensor(P + "embed_tokens.weight")
      h_t = [embed(b, t_emb) for b in batches]
      del t_emb
      pending = {}
      def teacher_layers():
          need = None
          for sh_name in hf.shards:
              sd = hf.load_shard(sh_name)
              pending.update({k: v for k, v in sd.items() if k.startswith(P + "layers.")})
              del sd
              # emit complete layers in order
              while True:
                  i = teacher_layers.next_i
                  keys = [k for k in pending if k.startswith(f"{P}layers.{i}.")]
                  if not keys:
                      break
                  # complete? compare with the module's state_dict size
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
      tail_bf16 = {}
      def teacher_layers_capture():
          for i, layer in teacher_layers():
              if i in train_layers:
                  tail_bf16.update({f"model.layers.{i}." + k: v.detach().cpu().clone() for k, v in layer.state_dict().items()})
              yield i, layer
      h_t = run_layers(tc, rotary, teacher_layers_capture(), h_t, dev, 0, L - 1, tag="teacher")
      log(f"teacher layers done: {time.time()-t0:.0f}s")
      t_norm = hf.tensor(P + "norm.weight").to(dev)
      t_head = hf.tensor("lm_head.weight").to(dev)
      tail_bf16["model.norm.weight"] = t_norm.cpu().clone()
      def head_logits(h, normw, head, chunk=1024):
          # returns generator of (start, logits[chunk, V]) in f32
          Bt, T, _ = h.shape
          hh = h.reshape(-1, H)
          for s in range(0, hh.shape[0], chunk):
              x = hh[s:s + chunk].float()
              # Qwen3.5's RMSNorm is zero-centred: output · (1 + w)
              x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps) * (1.0 + normw.float())
              yield s, x.to(torch.bfloat16) @ head.t()
      targets = []
      for bi, h in enumerate(h_t):
          Bt, T, _ = h.shape
          ids_flat = batches[bi].reshape(-1)
          top_v = torch.empty(Bt * T, tk, dtype=torch.float32, device=dev)
          top_i = torch.empty(Bt * T, tk, dtype=torch.long, device=dev)
          nll = 0.0; cnt = 0
          for s, lg in head_logits(h, t_norm, t_head):
              lg = lg.float()
              lp = torch.log_softmax(lg, -1)
              v, ix = lp.topk(tk, dim=-1)
              top_v[s:s + lg.shape[0]] = v; top_i[s:s + lg.shape[0]] = ix
              # teacher ppl on this batch (next-token): position p predicts token p+1 within each row
              n_rows = lg.shape[0]
              rows = torch.arange(s, s + n_rows, device=dev)
              valid = (rows % T) < (T - 1)
              tgt = ids_flat[(rows + 1).clamp(max=ids_flat.numel() - 1)]
              nll += -(lp[torch.arange(n_rows, device=dev), tgt][valid]).sum().item(); cnt += valid.sum().item()
          targets.append((top_v.view(Bt, T, tk), top_i.view(Bt, T, tk)))
          log(f"teacher batch {bi}: ppl {math.exp(nll/max(cnt,1)):.3f} over {cnt} tokens")
      del h_t, t_head
      torch.save({"targets": [(v.cpu(), i.cpu()) for v, i in targets], "tail_bf16": tail_bf16}, teach_path)
      log(f"teacher: saved {teach_path}")
    torch.cuda.empty_cache()

    # ── the trainable tail ──
    def build_tail(init_sd, dtype=torch.float32):
        mods = {}
        for i in train_layers:
            mods[i] = make_layer(tc, i, init_sd, dev, dtype, f"model.layers.{i}.")
        norm = Qwen3_5RMSNorm(H, eps=tc.rms_norm_eps).to(dev)
        norm.weight.data.copy_(init_sd["model.norm.weight"].to(dev, dtype))
        return mods, norm
    def tail_forward(mods, norm, h, T):
        B = h.shape[0]
        pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, B, -1)
        text_pos, pos_ids = pos[0], pos[1:]
        x = h.to(next(iter(mods.values())).parameters().__next__().dtype)
        pe = rotary(x, pos_ids)
        for i in train_layers:
            out = mods[i](x, position_embeddings=pe, attention_mask=None, position_ids=text_pos)
            x = out if torch.is_tensor(out) else out[0]
        return norm(x)
    def loss_on(mods, norm, bi, rows, use_ce=True):
        """rows: indices into batch bi. Returns (loss, ce, kl, ntok)."""
        h = h_stud[bi][rows]
        T = h.shape[1]
        x = tail_forward(mods, norm, h, T)  # [b,T,H]
        lg = (x.to(torch.bfloat16).reshape(-1, H) @ lm_head.t()).float()  # [bT, V]
        lp = torch.log_softmax(lg, -1)
        ids_b = batches[bi][rows]
        tv, ti = targets[bi][0][rows].reshape(-1, tk), targets[bi][1][rows].reshape(-1, tk)
        # KL(teacher_topk ‖ student) with the teacher renormalised on its top-k
        tp = torch.softmax(tv, -1)
        slp = lp.gather(1, ti)
        kl = (tp * (torch.log(tp + 1e-12) - slp)).sum(-1)
        # CE on next token within each row
        b = ids_b.shape[0]
        lp3 = lp.view(b, T, -1)
        ce = -lp3[:, :-1].gather(2, ids_b[:, 1:].unsqueeze(-1)).squeeze(-1)
        return ce.mean(), kl.mean()
    def loss_rows(mods, norm, bi, rows):
        """mean (ce, kl) over rows, one row at a time (memory: the 248k logits)."""
        with torch.no_grad():
            ces, kls = [], []
            for r in rows:
                ce, kl = loss_on(mods, norm, bi, torch.tensor([r], device=dev))
                ces.append(ce.item()); kls.append(kl.item())
            return torch.tensor(sum(ces) / len(ces)), torch.tensor(sum(kls) / len(kls))
    def eval_ppl(mods, norm, bi):
        with torch.no_grad():
            tot, cnt = 0.0, 0
            B = h_stud[bi].shape[0]
            for r in range(0, B):
                rows = torch.arange(r, r + 1, device=dev)
                ce, _ = loss_on(mods, norm, bi, rows)
                n_t = rows.numel() * (h_stud[bi].shape[1] - 1)
                tot += ce.item() * n_t; cnt += n_t
            return math.exp(tot / cnt)
    n_calib = calib.shape[0]
    val_rows = list(range(n_calib - args.val_seqs, n_calib))
    tr_rows = list(range(0, n_calib - args.val_seqs))
    results = {}
    def report(tag, mods, norm):
        pe = eval_ppl(mods, norm, 1)
        # val on the calib tail rows
        ce, kl = loss_rows(mods, norm, 0, val_rows)
        results[tag] = {"eval_ppl": pe, "val_ce": ce.item(), "val_kl": kl.item()}
        log(f"[{tag}] eval ppl {pe:.4f}  val ce {ce.item():.4f}  val kl {kl.item():.4f}")
    # (a) all quantized (student tail as in the file)
    mods, norm = build_tail(stud_tail, torch.bfloat16); report("quantized", mods, norm); del mods, norm
    # (b) tail restored to bf16 originals, no training
    mods, norm = build_tail(tail_bf16, torch.bfloat16); report("tail_bf16_notrain", mods, norm); del mods, norm
    torch.cuda.empty_cache()

    def train(init_sd, tag):
        mods, norm = build_tail(init_sd, torch.bfloat16 if args.bf16_train else torch.float32)
        params = [p for m in mods.values() for p in m.parameters()] + list(norm.parameters())
        for p in params: p.requires_grad_(True)
        if args.bf16_train:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        else:
            opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        best = (float("inf"), None)
        g = torch.Generator(device="cpu").manual_seed(0)
        report(f"{tag}:step0", mods, norm)
        best = (results[f"{tag}:step0"]["val_kl"] * args.kl + results[f"{tag}:step0"]["val_ce"] * (1 - args.kl),
                {k: v.detach().clone() for k, v in [(f"m{i}.{n}", p) for i, m in mods.items() for n, p in m.state_dict().items()] + [("norm", norm.weight.detach().clone())]})
        for step in range(1, args.steps + 1):
            rows = [tr_rows[j] for j in torch.randperm(len(tr_rows), generator=g)[: args.bs].tolist()]
            opt.zero_grad(set_to_none=True)
            ce_acc, kl_acc = 0.0, 0.0
            for r in rows:  # one sequence at a time: the 248k-wide logits are the memory
                ce, kl = loss_on(mods, norm, 0, torch.tensor([r], device=dev))
                loss = ((1 - args.kl) * ce + args.kl * kl) / len(rows)
                loss.backward()
                ce_acc += ce.item() / len(rows); kl_acc += kl.item() / len(rows)
            ce = torch.tensor(ce_acc); kl = torch.tensor(kl_acc)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 25 == 0 or step == args.steps:
                vce, vkl = loss_rows(mods, norm, 0, val_rows)
                score = (1 - args.kl) * vce.item() + args.kl * vkl.item()
                log(f"[{tag}] step {step}: train ce {ce.item():.4f} kl {kl.item():.4f} | val ce {vce.item():.4f} kl {vkl.item():.4f} score {score:.4f}")
                if score < best[0]:
                    best = (score, {k: v.detach().clone() for k, v in [(f"m{i}.{n}", p) for i, m in mods.items() for n, p in m.state_dict().items()] + [("norm", norm.weight.detach().clone())]})
        # restore best
        for i, m in mods.items():
            m.load_state_dict({n: best[1][f"m{i}.{n}"] for n in m.state_dict()})
        norm.weight.data.copy_(best[1]["norm"])
        report(f"{tag}:best", mods, norm)
        return mods, norm
    mods_q, norm_q = train(stud_tail, "fcd_from_quant")
    sd_q = {f"model.layers.{i}.{n}": p.detach().float().cpu() for i, m in mods_q.items() for n, p in m.state_dict().items()}
    sd_q["model.norm.weight"] = norm_q.weight.detach().float().cpu()
    del mods_q, norm_q; torch.cuda.empty_cache()
    mods_b, norm_b = train(tail_bf16, "fcd_from_bf16")
    sd_b = {f"model.layers.{i}.{n}": p.detach().float().cpu() for i, m in mods_b.items() for n, p in m.state_dict().items()}
    sd_b["model.norm.weight"] = norm_b.weight.detach().float().cpu()
    del mods_b, norm_b; torch.cuda.empty_cache()
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    log("results:", json.dumps(results, indent=1))
    # ── export the better trained tail at f16 into a copy of the cmf ──
    pick = "fcd_from_quant" if results["fcd_from_quant:best"]["eval_ppl"] <= results["fcd_from_bf16:best"]["eval_ppl"] else "fcd_from_bf16"
    sd_pick = sd_q if pick == "fcd_from_quant" else sd_b
    raw_dir = os.path.join(args.out, "tail_f32"); os.makedirs(raw_dir, exist_ok=True)
    sets = []
    for k, v in sd_pick.items():
        p = os.path.join(raw_dir, k + ".f32")
        v.contiguous().numpy().astype(np.float32).tofile(p)
        sets.append(f"--set '{k}={p}'")
    log(f"tail ({pick}) written as f32 under {raw_dir}")
    if not args.skip_export:
        out_cmf = os.path.join(args.out, os.path.basename(args.cmf).replace(".cmf", f".fcd-{pick}.cmf"))
        cmd = f"{args.cortiq} patch-tensor {args.cmf} {' '.join(sets)} --dtype f16 --output {out_cmf}"
        log("export:", pick, "→", out_cmf)
        print(sh(cmd))
        log("done; run: cortiq ppl", out_cmf)

if __name__ == "__main__":
    main()
