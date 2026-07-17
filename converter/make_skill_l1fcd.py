#!/usr/bin/env python3
"""Полный gentle-рецепт владельца (vmfcore/heal_l1fcd.py) на Qwen3.5:
Phase A — ОБУЧАЕМАЯ L1-маска на входе down_proj (логиты+STE, прогрессивный
L1, ловим ДНО денойзинга по held-out); Phase B — FCD последних N слоёв
(FFN-часть; attention заморожен — отличие от оригинала, честно отмечено).
Экспорт: запечённые маской down_proj (Factory-Hard: зануление столбцов,
P2 claim 1) + FCD-тензоры → каталог make_skill-формата для --skills.

Usage: make_skill_l1fcd.py --model SNAP --id ru --files … --out dir
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import heal_vmf_phase as H

DEV = H.DEV


def chunks_from(files, tok, chunk=256, need=112):
    out = []
    for f in files:
        ids = tok.encode(Path(f).read_text(errors="ignore")).ids
        for i in range(0, len(ids) - chunk, chunk):
            out.append(ids[i:i + chunk])
        if len(out) >= need:
            break
    if len(out) < 24:
        raise SystemExit(f"домен мал: {len(out)} чанков")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fcd-layers", type=int, default=4)
    ap.add_argument("--steps-a", type=int, default=240)
    ap.add_argument("--steps-b", type=int, default=120)
    ap.add_argument("--phi-layer", type=int, default=12)
    a = ap.parse_args()

    runner = H.Runner(Path(a.model))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(runner.snap / "tokenizer.json"))
    chunks = chunks_from(a.files, tok)
    held = chunks[:12]
    calib = chunks[12:]
    nL = len(runner.layer_types)
    inter = runner.cfg.get("text_config", runner.cfg)["intermediate_size"]
    fcd = list(range(nL - a.fcd_layers, nL))
    print(f"device={DEV} | '{a.id}' | {len(calib)} calib + {len(held)} held | "
          f"inter={inter} | FCD слои {fcd}", flush=True)

    # Веса в память один раз (иначе каждый шаг стримит 1.7GB с диска).
    W = {li: runner.layer_weights(li, runner.layer_types[li] == "linear_attention")
         for li in range(nL)}
    embed = runner.t("model.embed_tokens.weight")
    fnorm = runner.t("model.norm.weight")
    lm_name = ("lm_head.weight" if "lm_head.weight" in runner.src.entries
               else "model.embed_tokens.weight")
    lm = embed if lm_name != "lm_head.weight" else runner.t(lm_name)

    # Тренируемое: маска-логиты (все слои) + FFN последних N (Phase B).
    logit = {li: torch.nn.Parameter(torch.full((inter,), 2.0, device=DEV))
             for li in range(nL)}
    ffn = {}
    for li in fcd:
        for t in ("gate", "up", "down"):
            ffn[f"{li}_{t}"] = torch.nn.Parameter(W[li][t].clone())

    TAU = 0.5
    mode = ["off"]

    def forward(ids_list, use_ffn):
        ids_t = torch.tensor(ids_list, device=DEV)
        h = embed[ids_t].float()
        for li in range(nL):
            Wl = W[li]
            hn = H.gemma_rms(h, Wl["input_ln"])
            with torch.no_grad():
                at = runner.attn_out(li, hn, Wl)
            h = h + at
            pn = H.gemma_rms(h, Wl["post_ln"])
            gate_w = ffn[f"{li}_gate"] if (use_ffn and li in fcd) else Wl["gate"]
            up_w = ffn[f"{li}_up"] if (use_ffn and li in fcd) else Wl["up"]
            down_w = ffn[f"{li}_down"] if (use_ffn and li in fcd) else Wl["down"]
            act = torch.nn.functional.silu(pn @ gate_w.T) * (pn @ up_w.T)
            if mode[0] != "off":
                soft = torch.sigmoid(logit[li])
                g = soft if mode[0] == "soft" \
                    else (soft > TAU).float() + (soft - soft.detach())
                act = act * g
            h = h + act @ down_w.T
        return H.gemma_rms(h, fnorm), ids_t

    def lm_loss(ids_list, use_ffn):
        hh, ids_t = forward(ids_list, use_ffn)
        lg = (hh[:-1] @ lm.T).float()
        return torch.nn.functional.cross_entropy(lg, ids_t[1:])

    def ppl(use_ffn):
        with torch.no_grad():
            tot = n = 0
            for c in held:
                hh, ids_t = forward(c, use_ffn)
                lg = (hh[:-1] @ lm.T).float()
                tot += float(torch.nn.functional.cross_entropy(
                    lg, ids_t[1:], reduction="sum"))
                n += len(c) - 1
        return math.exp(tot / n)

    def sparsity():
        alive = sum(int((torch.sigmoid(logit[li]) > TAU).sum()) for li in range(nL))
        return 1 - alive / (nL * inter)

    mode[0] = "off"
    p_base = ppl(False)
    print(f"  baseline (full): {p_base:.3f}", flush=True)

    # Phase A — маска (LM-loss + прогрессивный L1, дно по held-out)
    optA = torch.optim.AdamW(list(logit.values()), lr=0.1)
    l1, best = 0.01, (p_base, None, 0.0)
    t0 = time.time()
    for step in range(a.steps_a):
        mode[0] = "soft"
        l1reg = torch.stack([torch.sigmoid(logit[li]).mean()
                             for li in range(nL)]).mean()
        loss = lm_loss(calib[step % len(calib)], False) + l1 * l1reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(logit.values()), 1.0)
        optA.step()
        optA.zero_grad()
        if (step + 1) % 30 == 0:
            l1 += 0.005
            mode[0] = "hard"
            hp, sp = ppl(False), sparsity()
            if hp < best[0]:
                best = (hp, {li: logit[li].detach().clone() for li in range(nL)}, sp)
            print(f"    [A] шаг {step+1}: L1={l1:.3f} прунинг={sp:.0%} "
                  f"hard-PPL={hp:.3f} (дно {best[0]:.3f}@{best[2]:.0%})", flush=True)
    if best[1]:
        for li in range(nL):
            logit[li].data = best[1][li]
    mode[0] = "hard"
    p_mask = ppl(False)
    print(f"  [A] {time.time()-t0:.0f}s: прунинг {sparsity():.0%}, "
          f"masked-PPL {p_mask:.3f}", flush=True)

    # Phase B — FCD (FFN последних N, cosine)
    optB = torch.optim.AdamW(list(ffn.values()), lr=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optB, T_max=a.steps_b,
                                                     eta_min=1e-6)
    bestB = (p_mask, {k: v.detach().clone() for k, v in ffn.items()})
    for step in range(a.steps_b):
        mode[0] = "hard"
        loss = lm_loss(calib[step % len(calib)], True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ffn.values()), 1.0)
        optB.step()
        optB.zero_grad()
        sch.step()
        if (step + 1) % 30 == 0:
            cur = ppl(True)
            if cur < bestB[0]:
                bestB = (cur, {k: v.detach().clone() for k, v in ffn.items()})
            print(f"    [B] шаг {step+1}: held-PPL {cur:.3f} "
                  f"(best {bestB[0]:.3f})", flush=True)
    p_fcd = bestB[0]
    verdict = "СПЕЦИАЛИСТ ≤ baseline ✓" if p_fcd <= p_base else "не обогнал"
    print(f"=== ИТОГ: baseline {p_base:.3f} | маска {p_mask:.3f} | "
          f"маска+FCD {p_fcd:.3f} → {verdict}", flush=True)

    # Экспорт: запечённые down_proj (все слои с прунингом) + FCD-тензоры.
    out = Path(a.out)
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    n_exp = 0
    # keep[li] — per-layer boolean vector of live neurons (True = live).
    # This is the keep-set for physical defragmentation (P2 claims 9/10):
    # `convert --defrag` drops gate/up rows and down columns by it. gate/up
    # rows of dead neurons are NOT zeroed in the bake (only down columns
    # are), so an explicit mask is required — zero-column autodetection
    # would miss them. See CMF_V2_SPEC §12.
    keep = np.zeros((nL, inter), dtype=bool)
    for li in range(nL):
        m = (torch.sigmoid(logit[li].data) > TAU)
        keep[li] = m.detach().cpu().numpy().astype(bool)
        down = (bestB[1][f"{li}_down"] if li in fcd else W[li]["down"]).clone()
        if (~m).any():
            down[:, ~m] = 0.0  # Factory-Hard: зануление = запекание маски
        if li in fcd or (~m).any():
            np.save(out / "tensors" / f"model.layers.{li}.mlp.down_proj.weight.npy",
                    down.detach().cpu().numpy().astype(np.float32))
            n_exp += 1
        if li in fcd:
            for t, nm in (("gate", "gate_proj"), ("up", "up_proj")):
                np.save(out / "tensors" / f"model.layers.{li}.mlp.{nm}.weight.npy",
                        bestB[1][f"{li}_{t}"].detach().cpu().numpy().astype(np.float32))
                n_exp += 2
    np.save(out / "ffn_keep.npy", keep)
    kept_per_layer = [int(keep[li].sum()) for li in range(nL)]
    print(f"  keep-mask: ffn_keep.npy [{nL}x{inter}], live neurons "
          f"{sum(kept_per_layer)}/{nL * inter} "
          f"({1 - sum(kept_per_layer) / (nL * inter):.0%} pruned)", flush=True)

    # Selection descriptor: φ доменного корпуса. ВАЖНО (этап 67): дескриптор
    # должен РАЗЛИЧАТЬ домены для роутера — фит на МНОГИХ чанках (≥32, не 8)
    # + rank 4 + phi_layer у слоёв скилла (по умолчанию первый FCD-слой).
    # Иначе recon-E не разделяет (измерено: 8 чанков rank2 → E(rus)≈E(eng);
    # 40 чанков rank4 @L20 → зазор +0.245 → dynamic бьёт backbone).
    RANK = 4
    phi_l = a.phi_layer if a.phi_layer < nL else fcd[0]
    vecs = []
    for c in calib[: min(len(calib), 40)]:
        hh = embed[torch.tensor(c, device=DEV)].float()
        for li in range(phi_l + 1):
            Wl = W[li]
            with torch.no_grad():
                hh = hh + runner.attn_out(li, H.gemma_rms(hh, Wl["input_ln"]), Wl)
                hh = hh + H.mlp(H.gemma_rms(hh, Wl["post_ln"]), Wl)
        vecs.append(hh.mean(0).detach().cpu().numpy())
    vecs = np.stack(vecs)
    mean = vecs.mean(0)
    _, _, vt = np.linalg.svd(vecs - mean, full_matrices=False)
    b64 = lambda x: base64.b64encode(np.asarray(x, np.float16).tobytes()).decode()
    json.dump({
        "id": a.id, "layers": fcd,
        "selection": {"metric": "mse", "phi_layer": int(phi_l),
                       "mean": b64(mean), "basis": b64(vt[:RANK].reshape(-1)),
                       "rank": RANK},
        "quality": {"metric": "ppl", "backbone": round(p_base, 3),
                     "masked": round(p_mask, 3), "overlaid": round(p_fcd, 3),
                     "held_out_chunks": len(held), "recipe": "L1+FCD"},
        # keep-set for physical defrag (P2 claims 9/10): matrix in
        # ffn_keep.npy, counts here for honest provenance.
        "defrag": {"keep_file": "ffn_keep.npy", "intermediate_size": inter,
                    "kept_per_layer": kept_per_layer,
                    "pruned_ratio": round(1 - sum(kept_per_layer) / (nL * inter), 4)},
    }, open(out / "skill.json", "w"))
    print(f"  ✓ {out}: {n_exp} тензоров + skill.json", flush=True)


if __name__ == "__main__":
    main()
