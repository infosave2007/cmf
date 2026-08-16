#!/usr/bin/env python3
"""
KD-LoRA heal of a folded Qwen3.5/3.8 checkpoint on ONE consumer GPU:
  base weights in bnb nf4 (frozen), LoRA on every projection, loss =
  0.3·CE + 0.7·KL(teacher top-k ‖ student) against precomputed teacher
  targets (heal_data_teacher.py), then the adapters are MERGED into the
  bf16 fold weights on the CPU (no double quantization) and written as a
  new checkpoint for `cortiq convert`.

usage: nvg_heal_kd.py --model /dev/shm/qwen38-fold19b --data /dev/shm/heal \
         --out /dev/shm/qwen38-fold19b-healed --epochs 2 --lr 2e-4 --r 64
"""
import argparse, os, sys, time, json, math, gc, glob
import torch, torch.nn.functional as F

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_fold(model_dir, tc, dtype=torch.bfloat16):
    """The fold checkpoint (model.language_model.* keys) into a text-only model on CPU."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from safetensors.torch import load_file
    # bf16 from the first byte: an f32 skeleton of a 19.8B model is 79 GB
    # and the bf16 conversion on top of it is what the box's 124 GB cgroup
    # killed, silently, on the first run.
    torch.set_default_dtype(dtype)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
    torch.set_default_dtype(torch.float32)
    model = model.to_empty(device="cpu")
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn"):
            inv_freq, _ = m.rope_init_fn(m.config, "cpu")
            m.inv_freq = inv_freq; m.original_inv_freq = inv_freq
    sd = model.state_dict()
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    seen = set()
    for f in files:
        part = load_file(os.path.join(model_dir, f))
        for k, v in part.items():
            k2 = k.replace("model.language_model.", "model.")
            if k2 in sd:
                sd[k2].copy_(v.to(sd[k2].dtype)); seen.add(k2)
        del part
    missing = [k for k in sd if k not in seen]
    log(f"fold loaded: {len(seen)} tensors, missing {len(missing)}: {missing[:5]}")
    model.to(dtype)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with tokens.pt + targets.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default="/workspace/models/wikitext2_test.txt")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--kl", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-seqs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    from transformers import AutoConfig, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    cfg_all = AutoConfig.from_pretrained(args.model)
    tc = cfg_all.text_config if hasattr(cfg_all, "text_config") else cfg_all
    tc._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(args.model)
    tokens = torch.load(os.path.join(args.data, "tokens.pt"))
    tg = torch.load(os.path.join(args.data, "targets.pt"))
    tv, ti, tk = tg["v"], tg["i"], tg["topk"]
    N, T = tokens.shape
    val_rows = list(range(N - args.val_seqs, N)); tr_rows = list(range(0, N - args.val_seqs))
    log(f"data: {N}×{T} tokens, top-{tk} targets; train {len(tr_rows)} val {len(val_rows)}")
    # eval windows (wikitext-2 test, 12×512)
    eids = tok(open(args.eval, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)["input_ids"]
    ev = torch.tensor(eids[: 12 * 512]).view(12, 512)

    # ── model: bf16 on CPU → nf4 linears → GPU → LoRA ──
    t0 = time.time()
    model = load_fold(args.model, tc)
    log(f"loaded in {time.time()-t0:.0f}s")
    # nf4 by hand: transformers' `replace_with_bnb_linear` builds the new
    # modules on the meta device (it expects a loader to fill them); the
    # weights are already here, so wrap them into Params4bit directly and
    # let `.to(cuda)` quantize.
    import bitsandbytes as bnb
    n_q = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and child_name != "lm_head":
                new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                        compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4")
                new.weight = bnb.nn.Params4bit(child.weight.data.contiguous(), requires_grad=False,
                                               compress_statistics=True, quant_type="nf4")
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)
                setattr(mod, child_name, new)
                n_q += 1
    log(f"{n_q} linears wrapped for nf4")
    model.config.use_cache = False
    t0 = time.time()
    model.to(dev)  # quantizes the Linear4bit weights on the way
    torch.cuda.synchronize()
    log(f"nf4 model on GPU in {time.time()-t0:.0f}s, VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                      "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params: p.data = p.data.float()  # fp32 adapters
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    steps_per_epoch = len(tr_rows) // (args.bs * args.accum)
    total_steps = int(args.epochs * steps_per_epoch)
    if args.max_steps: total_steps = min(total_steps, args.max_steps)
    warm = max(10, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    log(f"steps: {total_steps} ({steps_per_epoch}/epoch), tokens/step {args.bs*args.accum*T}")

    def kd_loss(logits, ids, rows):
        # logits [b,T,V] f32; ids [b,T]; teacher rows
        lp = torch.log_softmax(logits.float(), -1)
        v = tv[rows].to(dev).float(); i = ti[rows].to(dev).long()
        tp = torch.softmax(v, -1)
        slp = lp.gather(-1, i)
        kl = (tp * (torch.log(tp + 1e-12) - slp)).sum(-1).mean()
        ce = -lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
        return ce, kl
    def eval_ppl(ids_b):
        model.eval(); tot, cnt = 0.0, 0
        with torch.no_grad():
            for r in range(0, ids_b.shape[0], 2):
                x = ids_b[r:r + 2].to(dev)
                lg = model(input_ids=x).logits.float()
                lp = torch.log_softmax(lg, -1)
                ce = -lp[:, :-1].gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
                tot += ce.sum().item(); cnt += ce.numel()
        model.train()
        return math.exp(tot / cnt)
    def val_loss():
        model.eval(); ces, kls = [], []
        with torch.no_grad():
            for r in val_rows:
                x = tokens[r:r + 1].to(dev)
                lg = model(input_ids=x).logits
                ce, kl = kd_loss(lg, x, [r]); ces.append(ce.item()); kls.append(kl.item())
        model.train()
        return sum(ces) / len(ces), sum(kls) / len(kls)
    p0 = eval_ppl(ev); vce, vkl = val_loss()
    log(f"[step 0] wikitext ppl {p0:.3f}  val ce {vce:.4f} kl {vkl:.4f}")
    best = (vkl * args.kl + vce * (1 - args.kl), None)
    hist = [{"step": 0, "ppl": p0, "val_ce": vce, "val_kl": vkl}]
    g = torch.Generator().manual_seed(0)
    model.train()
    step = 0; t0 = time.time()
    order = []
    while step < total_steps:
        if not order:
            order = torch.randperm(len(tr_rows), generator=g).tolist()
        opt.zero_grad(set_to_none=True)
        acc_ce = acc_kl = 0.0
        for _ in range(args.accum):
            rows = [tr_rows[order.pop()] for _ in range(args.bs) if order]
            x = tokens[rows].to(dev)
            lg = model(input_ids=x).logits
            ce, kl = kd_loss(lg, x, rows)
            loss = ((1 - args.kl) * ce + args.kl * kl) / args.accum
            loss.backward()
            acc_ce += ce.item() / args.accum; acc_kl += kl.item() / args.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        if step % 10 == 0:
            el = time.time() - t0
            log(f"step {step}/{total_steps}  ce {acc_ce:.4f} kl {acc_kl:.4f}  lr {sched.get_last_lr()[0]:.2e}  {el/step:.1f}s/step  eta {el/step*(total_steps-step)/60:.0f}m")
        if step % args.eval_every == 0 or step == total_steps:
            p = eval_ppl(ev); vce, vkl = val_loss()
            score = vkl * args.kl + vce * (1 - args.kl)
            hist.append({"step": step, "ppl": p, "val_ce": vce, "val_kl": vkl})
            log(f"[step {step}] wikitext ppl {p:.3f}  val ce {vce:.4f} kl {vkl:.4f}  score {score:.4f}")
            if score < best[0]:
                best = (score, step)
                model.save_pretrained(os.path.join(args.out, "adapter_best"))
            model.save_pretrained(os.path.join(args.out, "adapter_last"))
            json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
    log(f"training done; best step {best[1]}")
    # ── merge the best adapter into the bf16 fold weights on the CPU ──
    from safetensors.torch import load_file, save_file
    ad_dir = os.path.join(args.out, "adapter_best" if best[1] else "adapter_last")
    ad = load_file(os.path.join(ad_dir, "adapter_model.safetensors"))
    scale = args.alpha / args.r
    # adapter keys look like base_model.model.model.layers.N.mlp.gate_proj.lora_A.weight
    def key_of(name):  # → the fold checkpoint's key
        base = name.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        return base.replace("model.", "model.language_model.", 1) + ".weight"
    deltas = {}
    for k in ad:
        if k.endswith("lora_A.weight"):
            kb = k.replace("lora_A", "lora_B")
            A, B = ad[k].float(), ad[kb].float()
            deltas[key_of(k)] = (B @ A) * scale
    log(f"merging {len(deltas)} adapters")
    idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    out_dir = os.path.join(args.out, "merged"); os.makedirs(out_dir, exist_ok=True)
    wm = {}
    n_merged = 0
    for f in files:
        part = load_file(os.path.join(args.model, f))
        new = {}
        for k, v in part.items():
            if k in deltas:
                new[k] = (v.float() + deltas[k]).to(torch.bfloat16).contiguous(); n_merged += 1
            else:
                new[k] = v
        save_file(new, os.path.join(out_dir, f), metadata={"format": "pt"})
        for k in new: wm[k] = f
        del part, new; gc.collect()
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    import shutil
    for fn in os.listdir(args.model):
        if fn.endswith((".json", ".jinja", ".txt")) and fn != "model.safetensors.index.json":
            shutil.copy(os.path.join(args.model, fn), os.path.join(out_dir, fn))
    log(f"merged checkpoint: {out_dir} ({n_merged} tensors merged)")

if __name__ == "__main__":
    main()
