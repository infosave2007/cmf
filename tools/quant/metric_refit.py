#!/usr/bin/env python3
"""Metric genesis: teacher-anchored end-to-end refit of the f16 values a quantized CMF
already stores — codes frozen, a real CMF file out.

Trainable (log2 units, init 0; see qmodel.py): per output row of every q4tp / q2tp linear
the row `lo` (exact row gain) and, with --step, the row `step` (a tilt of the group-scale
ladder); per row of every q8_2f linear its row scale; per channel every RMSNorm gain (layer
norms, q/k norms, GDN output norm, final norm). Every code byte, every 5-bit rung, the
embedding and the head stay byte-identical.

Loss: KL(teacher || student) over the teacher's top-K tokens plus one tail bucket, summed
over positions, mean over a batch of --bs windows of --wl tokens. The teacher is a reference
CMF of the same model (f16 / bf16 / q8_2f), run once in exact f32 and cached (calib.py).
Adam (0.9, 0.99), lr --lr (rows, norms) and --lr/--step-lr-div (step), 8 warm-up steps,
cosine to 0.1x; held-out CE / KL every --eval-every steps; the checkpoint with the lowest
held-out CE is snapped to f16 and written: OUT/raw/<tensor>.bin (payloads for
`cortiq patch-tensor --raw`), OUT/manifest.json, OUT/trained.npz (continuous values), and
with --write FILE.cmf the patched, verified container.

Memory: --residency resident keeps the compact codes of every layer on the GPU;
--residency stream keeps them in pinned host RAM and moves one decoder layer at a time, with
boundary activations and the teacher cache in host RAM — the same numbers bit for bit, so a
27B model trains on one 32 GB GPU. --mb sets the micro-batch (same objective, gradients
accumulated). Seeded and deterministic (deterministic torch kernels, seeded permutations).

  # MiniCPM5 G-C2 (the q4tp genesis study's arm):
  metric_refit.py --model C2.cmf --teacher f16.cmf --out arms/G-C2 --step --epochs 2 \\
      --corpus train.ids,train=1024,held=65536:8+200000:16 --write G-C2.cmf --cortiq cortiq
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import calib  # noqa: E402
import cmfio  # noqa: E402
import qmodel  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="quantized source CMF (codes are kept)")
    ap.add_argument("--teacher", required=True, help="reference CMF of the same model (f16/bf16/q8_2f)")
    ap.add_argument("--corpus", action="append", required=True, help="PATH[,train=N][,held=OFF:CNT+...][,name=X]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", help="also write the refit model here (via cortiq patch-tensor --raw + verify)")
    ap.add_argument("--cortiq", default="cortiq", help="cortiq binary with patch-tensor --raw")
    ap.add_argument("--nwin", type=int, default=1024, help="train windows over all corpora without train=")
    ap.add_argument("--wl", type=int, default=512)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--step", action="store_true", help="also train the per-row step of q4tp/q2tp")
    ap.add_argument("--no-norm", action="store_true", help="freeze the RMSNorm gains")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-lr-div", type=float, default=16.0)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--mb", type=int, default=0, help="windows per training micro-batch (default: --bs)")
    ap.add_argument("--eval-mb", type=int, default=8, help="windows per micro-batch of the forward-only passes "
                    "(teacher cache, held-out evals)")
    ap.add_argument("--eval-every", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0, help="train N steps (smoke tests; the schedule spans N)")
    ap.add_argument("--stop-after", type=int, default=0, help="stop at step N of the full schedule without "
                    "writing (checks; resume continues it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residency", choices=("resident", "stream"), default="resident")
    ap.add_argument("--act-device", default=None, help="boundary activations: cpu or cuda (default: cpu when streaming)")
    ap.add_argument("--teacher-cache", default=None, help="teacher top-K cache dir (default OUT/teacher)")
    ap.add_argument("--teacher-block", type=int, default=256)
    ap.add_argument("--vchunk", type=int, default=16384, help="vocabulary rows per head chunk")
    ap.add_argument("--head-windows", type=int, default=1, help="windows per vocabulary scan of the head "
                    "(more = fewer, larger kernels; 1 reproduces the studies' per-window head bit for bit)")
    ap.add_argument("--no-tf32", action="store_true", help="exact f32 matmuls in training (default TF32, as the studies)")
    ap.add_argument("--gpu-cap-gb", type=float, default=0, help="cap this process's CUDA allocator (shared GPUs)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from OUT/state.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    a.mb = a.mb or a.bs
    if a.bs % a.mb:
        ap.error("--bs must be a multiple of --mb")
    return a


class Log:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, *a):
        print(f"[{time.time() - self.t0:7.0f}s]", *a, flush=True)


def held_eval(model, held_w, teacher, owner, n_corp, mb, wl):
    r = model.evaluate(held_w, mb=mb, teacher=(teacher["held_lp"], teacher["held_ix"]))
    cnt = len(held_w) * (wl - 1)
    per = {}
    for i in range(n_corp):
        m = owner == i
        if m.any():
            per[i] = float(r["nll"][m].sum() / (m.sum() * (wl - 1)))
    return float(r["kl"].sum() / cnt), float(r["nll"].sum() / cnt), per


def snapshot(model):
    s = {"a": {}, "b": {}, "n": {}}
    for lin in model.linears():
        if lin.a is not None:
            s["a"][lin.name] = lin.a.detach().cpu().numpy().copy()
        if lin.b is not None:
            s["b"][lin.name] = lin.b_clamped().detach().cpu().numpy().copy()
    for g in model.gains():
        if g.n is not None:
            s["n"][g.name] = g.n.detach().cpu().numpy().copy()
    return s


def values_digest(model):
    """sha1 of every trainable value in a fixed order (bit-for-bit comparison of runs)"""
    import hashlib
    h = hashlib.sha1()
    for p in model.params():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_values(model, snap):
    """set the trainable tensors to a snapshot's values"""
    with torch.no_grad():
        for lin in model.linears():
            if lin.a is not None:
                lin.a.copy_(torch.from_numpy(np.asarray(snap["a"][lin.name], np.float32)))
            if lin.b is not None:
                lin.b.copy_(torch.from_numpy(np.asarray(snap["b"][lin.name], np.float32)))
        for g in model.gains():
            if g.n is not None:
                g.n.copy_(torch.from_numpy(np.asarray(snap["n"][g.name], np.float32)))


def write_payloads(model, snap, raw_dir):
    """f16-snapped payloads of every trained tensor -> raw_dir; returns (paths, the snapped
    values as a snapshot, stats)"""
    os.makedirs(raw_dir, exist_ok=True)
    paths, eff = {}, {"a": {}, "b": {}, "n": {}}
    st = {"rows": 0, "rows_lo_changed": 0, "rows_step_changed": 0, "mean_abs_log2_gain": {}}
    for lin in model.linears():
        if lin.a is None:
            continue
        a = snap["a"][lin.name]
        b = snap["b"].get(lin.name)
        payload, a_eff, b_eff = lin.snapped(a, b)
        p = os.path.join(raw_dir, f"{lin.name}.bin")
        with open(p, "wb") as f:
            f.write(payload)
        paths[lin.name] = p
        eff["a"][lin.name] = a_eff.astype(np.float32)
        if b_eff is not None:
            eff["b"][lin.name] = b_eff.astype(np.float32)
        if lin.dtype in qmodel.TILED:
            st["rows"] += lin.rows
            st["rows_lo_changed"] += int((a_eff != 0).sum())
            if b_eff is not None:
                st["rows_step_changed"] += int((b_eff != 0).sum())
        kind = lin.name.split(".")[-2]
        st["mean_abs_log2_gain"].setdefault(kind, []).append(float(np.abs(a).mean()))
    for g in model.gains():
        if g.n is None:
            continue
        payload, n_eff = g.snapped(snap["n"][g.name])
        p = os.path.join(raw_dir, f"{g.name}.bin")
        with open(p, "wb") as f:
            f.write(payload)
        paths[g.name] = p
        eff["n"][g.name] = n_eff.astype(np.float32)
        st["mean_abs_log2_gain"].setdefault("norm", []).append(float(np.abs(snap["n"][g.name]).mean()))
    st["mean_abs_log2_gain"] = {k: float(np.mean(v)) for k, v in st["mean_abs_log2_gain"].items()}
    return paths, eff, st


def lr_factor(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main(argv=None):
    args = get_args(argv)
    log = Log()
    qmodel.setup_torch(args.seed, tf32=not args.no_tf32, deterministic=True, threads=args.threads or None)
    dev = qmodel.resolve_device(args.device, args.gpu_cap_gb)
    os.makedirs(args.out, exist_ok=True)
    corpora = calib.place([calib.parse_corpus(s) for s in args.corpus], args.wl, args.nwin)
    train_w, held_w, owner = calib.build_sets(corpora, args.wl)
    log(f"corpora: " + ", ".join(f"{c.name} {len(c.train_starts)}+{len(c.held_starts)}" for c in corpora)
        + f" windows of {args.wl}")

    def make(cmf, residency=args.residency, **kw):
        return qmodel.Model(cmf, dev, residency=residency, act_device=args.act_device, vchunk=args.vchunk,
                            head_windows=args.head_windows, log=log, **kw)

    # the teacher (a float reference, 2-4x the student's bytes) streams straight from the
    # page cache: one pass per block, nothing pinned
    teacher = calib.teacher_cache(args.teacher_cache or os.path.join(args.out, "teacher"), args.teacher, corpora,
                                  args.wl, args.topk, lambda c: make(c, "stream", pin=False),
                                  block=args.teacher_block, mb=args.eval_mb, log=log)
    src = cmfio.CmfFile(args.model)
    model = make(src, train=True, step=args.step, train_norms=not args.no_norm)
    lins = list(model.linears())
    A = [l.a for l in lins if l.a is not None]
    B = [l.b for l in lins if l.b is not None]
    N = [g.n for g in model.gains() if g.n is not None]
    kinds = {}
    for l in lins:
        kinds[l.dtype] = kinds.get(l.dtype, 0) + 1
    log(f"student {args.model}: {len(model.layers)} layers, linears {kinds}, trainable rows {sum(a.numel() for a in A)}, "
        f"step {sum(b.numel() for b in B)}, norm channels {sum(n.numel() for n in N)}; "
        f"residency {args.residency}, body {model.body_bytes() / 2**30:.2f} GiB compact; "
        f"GPU {torch.cuda.memory_allocated(dev) / 2**30 if dev.type == 'cuda' else 0:.2f} GiB")
    groups = [{"params": A, "lr": args.lr}]
    if B:
        groups.append({"params": B, "lr": args.lr / args.step_lr_div})
    if N:
        groups.append({"params": N, "lr": args.lr})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    base_lr = [g["lr"] for g in opt.param_groups]
    nwin = len(train_w)
    spe = nwin // args.bs
    total = args.max_steps or int(round(args.epochs * spe))
    warm = min(args.warmup, total)
    gen = torch.Generator().manual_seed(args.seed)
    state_p = os.path.join(args.out, "state.pt")
    step, run_kl, hist, best, perm = 0, 0.0, [], None, None
    t_train = 0.0
    if args.resume and os.path.exists(state_p):
        s = torch.load(state_p, weights_only=False)
        load_values(model, s["values"])
        opt.load_state_dict(s["opt"])
        gen.set_state(s["gen"])
        step, hist, best, perm, t_train = s["step"], s["hist"], s["best"], s["perm"], s.get("t_train", 0.0)
        log(f"resumed at step {step}")
    if not hist:
        hkl, hce, per = held_eval(model, held_w, teacher, owner, len(corpora), args.eval_mb, args.wl)
        hist.append({"step": 0, "held_kl": hkl, "held_ce": hce, "held_ce_corpus": per})
        best = (hce, 0, snapshot(model))
        log(f"step 0: held-out KL {hkl:.5f} CE {hce:.5f} (teacher {teacher['meta']['held_ce']:.5f}); "
            f"{total} steps of {args.bs}x{args.wl}, micro-batch {args.mb}")
    count = args.bs * (args.wl - 1)
    while step < total:
        if perm is None or step % spe == 0:
            # one permutation per epoch, drawn in the same order as the studies' runs
            perm = torch.randperm(nwin, generator=gen)
        t0 = time.time()
        idx = perm[(step % spe) * args.bs:(step % spe + 1) * args.bs].numpy()
        f = lr_factor(step, total, warm)
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
        opt.zero_grad(set_to_none=True)
        lp = torch.from_numpy(np.ascontiguousarray(teacher["train_lp"][idx]))
        ix = torch.from_numpy(np.ascontiguousarray(teacher["train_ix"][idx]))
        kl = model.train_step(train_w[idx], lp, ix, args.mb, count)
        opt.step()
        step += 1
        run_kl += kl / count
        t_train += time.time() - t0
        if step % args.eval_every == 0 or step == total:
            hkl, hce, per = held_eval(model, held_w, teacher, owner, len(corpora), args.eval_mb, args.wl)
            n_since = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
            hist.append({"step": step, "train_kl": run_kl / n_since, "held_kl": hkl, "held_ce": hce,
                         "held_ce_corpus": per, "values_sha1": values_digest(model)})
            peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else 0
            log(f"step {step}/{total} train KL {run_kl / n_since:.5f}  held-out KL {hkl:.5f} CE {hce:.5f}  "
                f"({t_train / step:.1f} s/step, GPU peak {peak:.2f} GiB)")
            run_kl = 0.0
            if hce < best[0]:
                best = (hce, step, snapshot(model))
            torch.save({"values": snapshot(model), "opt": opt.state_dict(), "gen": gen.get_state(), "step": step,
                        "hist": hist, "best": best, "perm": perm, "t_train": t_train}, state_p + ".tmp")
            os.replace(state_p + ".tmp", state_p)
            if args.stop_after and step >= args.stop_after:
                break
    if args.stop_after and step < total:
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump({"args": vars(args), "hist": hist, "stopped_at": step, "train_seconds": t_train}, f, indent=1)
        log(f"stopped at step {step} of {total} (--stop-after); state in {state_p}")
        return None
    log(f"best held-out CE {best[0]:.5f} at step {best[1]} (step 0: {hist[0]['held_ce']:.5f}, "
        f"teacher {teacher['meta']['held_ce']:.5f})")
    snap = best[2]
    np.savez_compressed(os.path.join(args.out, "trained.npz"), **{f"a|{k}": v for k, v in snap["a"].items()},
                        **{f"b|{k}": v for k, v in snap["b"].items()}, **{f"n|{k}": v for k, v in snap["n"].items()})
    paths, eff, st = write_payloads(model, snap, os.path.join(args.out, "raw"))
    load_values(model, eff)
    skl, sce, _ = held_eval(model, held_w, teacher, owner, len(corpora), args.eval_mb, args.wl)
    log(f"f16-snapped checkpoint: held-out KL {skl:.5f} CE {sce:.5f} (unsnapped {best[0]:.5f}); "
        f"lo changed in {st['rows_lo_changed']}/{st['rows']} rows, step in {st['rows_step_changed']}")
    man = {"args": vars(args), "corpora": calib.describe(corpora), "hist": hist, "best_step": best[1],
           "best_held_ce": best[0], "held_ce_snapped": sce, "held_kl_snapped": skl,
           "teacher": {k: v for k, v in teacher["meta"].items() if k != "key"}, "n_payloads": len(paths),
           "train_seconds": t_train, "seconds": time.time() - log.t0, **st}
    if args.write:
        cmfio.patch_raw(args.cortiq, args.model, args.write, paths)
        man["written"] = os.path.abspath(args.write)
        log(f"wrote and verified {args.write}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    log("done:", len(paths), "payloads;", json.dumps(st["mean_abs_log2_gain"]))
    return man


if __name__ == "__main__":
    main()
