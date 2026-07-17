#!/usr/bin/env python3
"""FCD-heal of the vmf_phase linear core against the GatedDeltaNet teacher.

Layer-local distillation (fits a 25 GB machine — one layer at a time):
  1. CAPTURE: run the ORIGINAL model (torch, layer-streamed from
     safetensors) on calibration tokens; save each linear layer's input
     and its GDN output (the teacher signal). Also report the original
     model's calibration PPL — a self-check that this forward is right.
  2. HEAL: per layer, train the vmf_phase student (thq/thk/v/out/A_log,
     init = the converter's fold) to match the teacher output (MSE).
  3. EVALUATE: PPL of the fully-swapped (healed) model vs original.
  4. EXPORT: healed weights per layer → npz; the converter picks them up
     via --heal-dir instead of the fold init.

Teacher GDN forward reuses the validated oracle (vmfcore/gdn_layer.py).
Full-attention layers follow the real config: gemma-style (1+w) norms
(incl. qk-norm), output gate, partial rotary 0.25, rope theta from
config — the tiny oracle used full rotary, the real model does not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_dtgma_to_cmf as conv  # noqa: E402
from gdn_layer import gdn_layer_torch  # noqa: E402  (validated teacher)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

CALIB_TEXT = (
    "The theory of general relativity describes gravitation as a geometric "
    "property of spacetime. Energy and momentum curve spacetime, and this "
    "curvature tells matter how to move. "
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
    "        a, b = b, a + b\n    return a\n\n"
    "Вакуумный конденсат описывается комплексным параметром порядка: "
    "амплитуда задаёт плотность энергии, а голдстоуновская фаза — "
    "когерентность и порядок. Плавление конденсата в плотной среде "
    "останавливает коллапс и снимает сингулярность. "
    "In machine learning, attention mechanisms allow models to weigh the "
    "relevance of different parts of the input sequence when producing "
    "each element of the output. SELECT name, COUNT(*) FROM users GROUP "
    "BY name ORDER BY 2 DESC LIMIT 10; "
    "История науки показывает, что красивые теории побеждают тогда, "
    "когда их предсказания фальсифицируемы и подтверждаются измерением. "
)


def gather_corpus(max_chars: int = 240_000) -> str:
    """Diverse local calibration corpus (docs + Rust + Python, ru+en).

    v1 tiled the 214-token CALIB_TEXT ×4.8 to fill the window — 158
    unique tokens. The 31M-param students interpolated that handful of
    (x, y) pairs to nMSE ~1e-4 and collapsed under the drift of the
    full swap (calib PPL 44821). Distinct tokens are the cure, not a
    nicety.
    """
    cmf = Path(__file__).resolve().parent.parent
    repo = cmf.parent
    files: list[Path] = []
    files += sorted(cmf.glob("docs/*.md"))
    files += sorted((cmf / "crates").rglob("*.rs"))[:8]
    files += sorted(cmf.glob("converter/*.py"))[:3]
    files += sorted((repo / "vmfcore").glob("*.py"))[:5]
    texts = []
    for f in files:
        try:
            texts.append(f.read_text(errors="ignore")[:8000])
        except OSError:
            continue
    # Round-robin 1200-char chunks: every training window then mixes
    # genres (docs, Rust, Python, ru+en) instead of soaking in one file.
    chunks = [[t[i:i + 1200] for i in range(0, len(t), 1200)] for t in texts]
    parts, total = [CALIB_TEXT], len(CALIB_TEXT)
    for r in range(max(len(c) for c in chunks)):
        for c in chunks:
            if r < len(c):
                parts.append(c[r])
                total += len(c[r])
        if total > max_chars:
            break
    return "\n\n".join(parts)


# ───────────────────────── model pieces (torch, real config) ─────────────────────────

def gemma_rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = x.float().pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * (1.0 + w)


def rope_partial(x: torch.Tensor, pos0: int, rdim: int, theta: float) -> torch.Tensor:
    """x [T, H, hd]; rotate first rdim dims, pairs (j, j+rdim/2) — the
    same convention as the Rust engine and vmfcore rope_inplace."""
    T, H, hd = x.shape
    half = rdim // 2
    # MPS has no f64 — build the tables in f64 on CPU, move as f32.
    j = torch.arange(half, dtype=torch.float64)
    freq = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float64), j / half)
    ang = torch.arange(pos0, pos0 + T, dtype=torch.float64)[:, None] * freq
    cos = ang.cos().float().to(x.device)[:, None, :]  # [T,1,half]
    sin = ang.sin().float().to(x.device)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:rdim]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rdim] = x2 * cos + x1 * sin
    return out


def gated_attention(x: torch.Tensor, W: dict, nh: int, nkv: int, hd: int,
                    rdim: int, theta: float, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5 full attention: per-head [q; gate] split, gemma qk-norm,
    partial RoPE, GQA softmax, sigmoid output gate, o_proj."""
    T = x.shape[0]
    rep = nh // nkv
    qg = (x @ W["q_proj"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
    k = (x @ W["k_proj"].T).reshape(T, nkv, hd)
    v = (x @ W["v_proj"].T).reshape(T, nkv, hd)
    q = gemma_rms(q, W["q_norm"], eps)
    k = gemma_rms(k, W["k_norm"], eps)
    q = rope_partial(q, 0, rdim, theta)
    k = rope_partial(k, 0, rdim, theta)

    kf = k.repeat_interleave(rep, dim=1)  # [T, nh, hd]
    vf = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q, kf) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = scores.softmax(-1)
    out = torch.einsum("hts,shd->thd", probs, vf).reshape(T, nh * hd)
    out = out * torch.sigmoid(gate)
    return out @ W["o_proj"].T


def mlp(x: torch.Tensor, W: dict) -> torch.Tensor:
    g = x @ W["gate"].T
    return (torch.nn.functional.silu(g) * (x @ W["up"].T)) @ W["down"].T


# ───────────────────────── vmf_phase student (chunked recurrence) ─────────────────────────

class VmfPhaseStudent(torch.nn.Module):
    """The canonical linear core, trainable. Chunked recurrence keeps the
    sequential loop at seq/chunk iterations (fast enough on MPS)."""

    def __init__(self, thq, thk, v_proj, out_proj, a_log, nh, nph, dv):
        super().__init__()
        self.nh, self.nph, self.dv = nh, nph, dv
        self.thq = torch.nn.Parameter(thq)
        self.thk = torch.nn.Parameter(thk)
        self.v_proj = torch.nn.Parameter(v_proj)
        self.out_proj = torch.nn.Parameter(out_proj)
        self.a_log = torch.nn.Parameter(a_log)  # [nh, p2]

    def forward(self, x: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        T = x.shape[0]
        nh, nph, dv = self.nh, self.nph, self.dv
        p2 = 2 * nph
        thq = (x @ self.thq.T).reshape(T, nh, nph)
        thk = (x @ self.thk.T).reshape(T, nh, nph)
        v = (x @ self.v_proj.T).reshape(T, nh, dv)
        fq = torch.cat([thq.cos(), thq.sin()], dim=-1)  # [T, nh, p2]
        fk = torch.cat([thk.cos(), thk.sin()], dim=-1)
        # decay ∈ [exp(-0.7), 1) ≈ [0.5, 1): slow Goldstone modes; keeps
        # decay^{-chunk} finite in f32 for the chunked scan.
        decay = torch.exp(-torch.exp(self.a_log.clamp(-10.0, -0.3567)))

        out = torch.empty(T, nh, dv, device=x.device, dtype=x.dtype)
        S = torch.zeros(nh, p2, dv, device=x.device, dtype=x.dtype)
        for c0 in range(0, T, chunk):
            c1 = min(c0 + chunk, T)
            L = c1 - c0
            t = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
            # decay^t  [L, nh, p2]; within-chunk relative decays
            dpow = decay.unsqueeze(0) ** t.view(L, 1, 1)
            # contribution of the carried state: fq_t · (decay^t ⊙ S)
            out[c0:c1] = torch.einsum("lhp,lhp,hpd->lhd", fq[c0:c1], dpow, S)
            # intra-chunk: o_t += Σ_{τ≤t} decay^{t-τ} (fq_t·fk_τ) v_τ
            # decay^{t-τ} = dpow_t / dpow_τ (safe: L ≤ chunk keeps ratios finite)
            inv = dpow.reciprocal()
            a = fk[c0:c1] * inv  # [L,nh,p2]
            qd = fq[c0:c1] * dpow
            att = torch.einsum("lhp,mhp->hlm", qd, a)  # [nh, L, L]
            att = att.tril()
            out[c0:c1] += torch.einsum("hlm,mhd->lhd", att, v[c0:c1])
            # carry state: S' = decay^L ⊙ S + Σ_τ decay^{L-τ} fk_τ⊗v_τ
            S = dpow[-1].unsqueeze(-1) * S + torch.einsum(
                "lhp,lhd->hpd", a * dpow[-1].unsqueeze(0), v[c0:c1]
            )
        return out.reshape(T, nh * dv) @ self.out_proj.T


def fold_init(qkv_w, out_w, a_log_gdn, nk, nv, dk, dv, nph):
    """Same init as the converter's VmfPhaseFoldSource (v/out/A_log carried,
    thq/thk subsampled q/k rows)."""
    kd = nk * dk
    rep = nv // nk
    v_proj = qkv_w[2 * kd:2 * kd + nv * dv].clone()
    H = qkv_w.shape[1]
    thq = torch.empty(nv * nph, H)
    thk = torch.empty(nv * nph, H)
    for h in range(nv):
        ko = h // rep
        for i in range(nph):
            src = ko * dk + (i * dk) // nph
            thq[h * nph + i] = qkv_w[src]
            thk[h * nph + i] = qkv_w[kd + src]
    a_log = a_log_gdn.view(nv, 1).repeat(1, 2 * nph).clamp(-10.0, -0.3567).clone()
    return thq, thk, v_proj, out_w.clone(), a_log


# ───────────────────────── capture + heal + evaluate ─────────────────────────

class Runner:
    def __init__(self, snap: Path):
        self.snap = snap
        self.cfg = json.load(open(snap / "config.json"))
        tc = self.cfg.get("text_config", self.cfg)
        self.H = tc["hidden_size"]
        self.nh = tc["num_attention_heads"]
        self.nkv = tc["num_key_value_heads"]
        self.hd = tc["head_dim"]
        self.nk = tc["linear_num_key_heads"]
        self.nv = tc["linear_num_value_heads"]
        self.dk = tc["linear_key_head_dim"]
        self.dv = tc["linear_value_head_dim"]
        self.kk = tc["linear_conv_kernel_dim"]
        self.layer_types = tc["layer_types"]
        rp = tc.get("rope_parameters", {})
        self.theta = rp.get("rope_theta", tc.get("rope_theta", 1e7))
        self.rdim = int(self.hd * rp.get("partial_rotary_factor",
                                         tc.get("partial_rotary_factor", 1.0)))
        self.src = conv.SafetensorsSource(sorted(snap.glob("*.safetensors")))

    def t(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.src.load(name)).to(DEV)

    def layer_weights(self, li: int, linear: bool) -> dict:
        p = f"model.layers.{li}."
        w = {
            "input_ln": self.t(p + "input_layernorm.weight"),
            "post_ln": self.t(p + "post_attention_layernorm.weight"),
            "gate": self.t(p + "mlp.gate_proj.weight"),
            "up": self.t(p + "mlp.up_proj.weight"),
            "down": self.t(p + "mlp.down_proj.weight"),
        }
        if linear:
            for k in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                      "A_log", "dt_bias", "norm", "out_proj"):
                w[k] = self.t(f"{p}linear_attn.{k}.weight"
                              if k not in ("A_log", "dt_bias") else f"{p}linear_attn.{k}")
            w["conv1d"] = self.t(p + "linear_attn.conv1d.weight")
        else:
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                w[k] = self.t(f"{p}self_attn.{k}.weight")
        return w

    def attn_out(self, li: int, hn: torch.Tensor, W: dict,
                 healed: dict | None = None) -> torch.Tensor:
        if self.layer_types[li] == "linear_attention":
            if healed is not None and li in healed:
                # Students park on CPU (40+ of them = ~15 GB); visit MPS
                # one at a time.
                m = healed[li]
                m.to(DEV)
                out = m(hn)
                m.to("cpu")
                if DEV == "mps":
                    torch.mps.empty_cache()
                return out
            # The oracle's recurrent teacher allocates on CPU (and a
            # per-step python loop is dispatch-bound on MPS anyway) —
            # run the teacher on CPU, bring the result back.
            Wg = {k: W[k].cpu() for k in
                  ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                   "A_log", "dt_bias", "conv1d", "norm", "out_proj")}
            return gdn_layer_torch(hn.float().cpu(), Wg,
                                   self.nk, self.nv, self.dk, self.dv).to(hn.device)
        return gated_attention(hn, W, self.nh, self.nkv, self.hd, self.rdim, self.theta)

    def forward_capture(self, ids: list[int], capture_dir: Path | None,
                        healed: dict | None = None) -> float:
        """Full forward; optionally captures linear-layer (input, teacher
        output) pairs; returns calibration PPL (the honest self-check)."""
        embed = self.t("model.embed_tokens.weight")
        h = embed[torch.tensor(ids, device=DEV)].float()
        del embed
        for li, lt in enumerate(self.layer_types):
            W = self.layer_weights(li, lt == "linear_attention")
            hn = gemma_rms(h, W["input_ln"])
            a = self.attn_out(li, hn, W, healed)
            if capture_dir is not None and lt == "linear_attention":
                np.savez(capture_dir / f"calib_L{li}.npz",
                         x=hn.cpu().numpy().astype(np.float32),
                         y=a.detach().cpu().numpy().astype(np.float32))
            h = h + a
            h = h + mlp(gemma_rms(h, W["post_ln"]), W)
            del W
            if DEV == "mps":
                torch.mps.empty_cache()
        h = gemma_rms(h, self.t("model.norm.weight"))
        # Tied embeddings (Qwen3.5-0.8B): no lm_head tensor in the file.
        lm_name = ("lm_head.weight"
                   if "lm_head.weight" in self.src.entries
                   else "model.embed_tokens.weight")
        lm = self.t(lm_name)
        tgt = torch.tensor(ids[1:], device=DEV)
        # Chunked CE: the full [T, vocab] logits are ~6 GB at T=6144 —
        # that plus resident healed students OOMs MPS.
        loss_sum, cnt = 0.0, 0
        for c0 in range(0, len(ids) - 1, 512):
            c1 = min(c0 + 512, len(ids) - 1)
            lg = (h[c0:c1] @ lm.T).float()
            loss_sum += float(torch.nn.functional.cross_entropy(
                lg, tgt[c0:c1], reduction="sum"))
            cnt += c1 - c0
            del lg
            if DEV == "mps":
                torch.mps.empty_cache()
        del lm
        return float(np.exp(loss_sum / max(cnt, 1)))


def tokenize(snap: Path, n_tokens: int) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    ids = tok.encode(gather_corpus()).ids
    if len(ids) < n_tokens:  # never tile silently — repetition is poison
        raise SystemExit(f"calib corpus too small: {len(ids)} < {n_tokens}")
    return ids[:n_tokens]


def heal_layer(runner: Runner, li: int, capture_dir: Path, out_dir: Path,
               nph: int, steps: int, lr: float, val_tokens: int = 384,
               noise: float = 0.05, window: int = 0,
               burnin: int = 256) -> tuple[float, float, float, float]:
    """Distill one layer. Selection is by HELD-OUT nMSE: the training
    window is the sequence prefix, validation is the (state-warmed)
    suffix, and Gaussian input noise simulates the residual-stream
    drift the student will actually see inside the swapped model.

    window > 0: each step trains on a random contiguous window instead
    of the full prefix — step cost stays constant while the captured
    corpus grows. The first `burnin` positions warm the recurrent state
    from zero and are excluded from the loss (the teacher's state at
    the window start is unknown)."""
    data = np.load(capture_dir / f"calib_L{li}.npz")
    x = torch.from_numpy(data["x"]).to(DEV)
    y = torch.from_numpy(data["y"]).to(DEV)
    T = x.shape[0]
    tv = min(val_tokens, T // 4)
    tt = T - tv

    p = f"model.layers.{li}.linear_attn."
    qkv = torch.from_numpy(runner.src.load(p + "in_proj_qkv.weight"))
    outw = torch.from_numpy(runner.src.load(p + "out_proj.weight"))
    alog = torch.from_numpy(runner.src.load(p + "A_log"))
    thq, thk, vp, op, al = fold_init(qkv, outw, alog, runner.nk, runner.nv,
                                     runner.dk, runner.dv, nph)
    student = VmfPhaseStudent(thq, thk, vp, op, al,
                              runner.nv, nph, runner.dv).to(DEV)

    yt_var = float(y[:tt].float().pow(2).mean())
    yv_var = float(y[tt:].float().pow(2).mean()) if tv else yt_var
    x_rms = float(x.float().pow(2).mean().sqrt())

    def split_nmse() -> tuple[float, float]:
        with torch.no_grad():
            o = student(x)
            tr = float((o[:tt] - y[:tt]).float().pow(2).mean()) / yt_var
            vl = float((o[tt:] - y[tt:]).float().pow(2).mean()) / yv_var if tv else tr
        return tr, vl

    # Calibrated init: rescale out_proj so the student's output RMS
    # matches the teacher's — the optimizer starts near the right
    # magnitude instead of spending its budget on rescaling.
    with torch.no_grad():
        s_var = float(student(x[:tt]).float().pow(2).mean())
        student.out_proj.mul_(math.sqrt(yt_var / max(s_var, 1e-12)))
    tr0, vl0 = split_nmse()

    optim = torch.optim.Adam(student.parameters(), lr=lr)
    best = (vl0, tr0,
            {k: v.detach().clone() for k, v in student.state_dict().items()})
    for step in range(steps):
        optim.zero_grad()
        if window and tt > window:
            s = int(torch.randint(0, tt - window, (1,)))
            xw, yw, b = x[s:s + window], y[s:s + window], (burnin if s else 0)
        else:
            xw, yw, b = x[:tt], y[:tt], 0
        xn = xw + noise * x_rms * torch.randn_like(xw) if noise else xw
        loss = (student(xn)[b:] - yw[b:]).float().pow(2).mean() \
            / float(yw[b:].float().pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optim.step()
        if (step + 1) % 25 == 0 or step == steps - 1:
            tr, vl = split_nmse()  # clean forward; selection is held-out
            if vl < best[0]:
                best = (vl, tr, {k: v.detach().clone()
                                 for k, v in student.state_dict().items()})
    # Keep the best-by-val snapshot — never export something that
    # generalizes worse than the calibrated init.
    student.load_state_dict(best[2])
    vl1, tr1 = best[0], best[1]

    np.savez(out_dir / f"heal_L{li}.npz",
             thq=student.thq.detach().cpu().numpy().astype(np.float32),
             thk=student.thk.detach().cpu().numpy().astype(np.float32),
             v_proj=student.v_proj.detach().cpu().numpy().astype(np.float32),
             out_proj=student.out_proj.detach().cpu().numpy().astype(np.float32),
             a_log=student.a_log.detach().clamp(-10.0, -0.3567)
                 .cpu().numpy().astype(np.float32).reshape(-1))
    return tr0, tr1, vl0, vl1


def load_healed(runner: Runner, out_dir: Path, nph: int,
                only: set[int] | None = None) -> dict:
    healed = {}
    for li, lt in enumerate(runner.layer_types):
        if only is not None and li not in only:
            continue
        f = out_dir / f"heal_L{li}.npz"
        if lt == "linear_attention" and f.exists():
            z = np.load(f)
            m = VmfPhaseStudent(
                torch.from_numpy(z["thq"]), torch.from_numpy(z["thk"]),
                torch.from_numpy(z["v_proj"]), torch.from_numpy(z["out_proj"]),
                torch.from_numpy(z["a_log"]).reshape(runner.nv, 2 * nph),
                runner.nv, nph, runner.dv)  # CPU-resident; MPS per layer
            m.requires_grad_(False)
            healed[li] = m
    return healed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF snapshot dir")
    ap.add_argument("--out", required=True, help="heal output dir")
    ap.add_argument("--nphase", type=int, default=64)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--val-tokens", type=int, default=384,
                    help="held-out suffix for snapshot selection")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="input-noise augmentation, ×rms(x)")
    ap.add_argument("--window", type=int, default=0,
                    help="random training window per step (0 = full prefix)")
    ap.add_argument("--progressive", type=int, default=8,
                    help="re-capture through the healed prefix every N "
                         "linear layers (0 = off); kills compounding drift")
    ap.add_argument("--layers", default="all", help="'all' | 'N' | 'A-B'")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--eval", action="store_true", help="PPL of the healed swap")
    a = ap.parse_args()

    snap = Path(a.model)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = out_dir / "calib"
    capture_dir.mkdir(exist_ok=True)

    runner = Runner(snap)
    ids = tokenize(snap, a.calib_tokens)
    print(f"device={DEV} | calib tokens={len(ids)} | rdim={runner.rdim} "
          f"theta={runner.theta}", flush=True)

    if not a.skip_capture:
        t0 = time.time()
        ppl = runner.forward_capture(ids, capture_dir)
        print(f"CAPTURE done in {time.time()-t0:.0f}s | ORIGINAL model calib "
              f"PPL = {ppl:.2f}  (sanity: must be sane, otherwise the "
              f"capture forward is wrong)", flush=True)

    lin = [i for i, t in enumerate(runner.layer_types) if t == "linear_attention"]
    if a.layers != "all":
        if "-" in a.layers:
            lo, hi = map(int, a.layers.split("-"))
            lin = [i for i in lin if lo <= i <= hi]
        else:
            lin = [int(a.layers)]

    for n, li in enumerate(lin):
        if a.progressive and n and n % a.progressive == 0:
            # Re-capture through the already-healed prefix: downstream
            # students then train on the inputs they will actually see
            # (teacher target = the GDN operator on those same inputs).
            t0 = time.time()
            done = load_healed(runner, out_dir, a.nphase)  # all on disk
            runner.forward_capture(ids, capture_dir, done)
            print(f"RECAPTURE through {len(done)} healed layers "
                  f"in {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        tr0, tr1, vl0, vl1 = heal_layer(runner, li, capture_dir, out_dir,
                                        a.nphase, a.steps, a.lr,
                                        a.val_tokens, a.noise, a.window)
        print(f"HEAL L{li:02d}: train {tr0:.4f} → {tr1:.4f} | "
              f"val {vl0:.4f} → {vl1:.4f} (×{vl0/max(vl1,1e-9):.1f}) "
              f"in {time.time()-t0:.0f}s", flush=True)

    if a.eval:
        healed = load_healed(runner, out_dir, a.nphase)
        ppl = runner.forward_capture(ids, None, healed)
        print(f"HEALED swap calib PPL = {ppl:.2f} ({len(healed)} layers swapped)",
              flush=True)


if __name__ == "__main__":
    main()
