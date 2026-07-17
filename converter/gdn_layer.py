"""Полный forward GatedDeltaNet-слоя (Qwen3.5/Qwen3-Next) на РЕАЛЬНЫХ именах тензоров —
numpy-оракул + torch-референс, end-to-end паритет. Это ядро для конвертера и Rust-рантайма (путь A).
Веса: in_proj_qkv[2kd+vd,H], in_proj_z[vd,H], in_proj_a[nv,H], in_proj_b[nv,H],
A_log[nv], dt_bias[nv], conv1d[2kd+vd,1,K], norm[dv], out_proj[H,vd]."""
import numpy as np, torch, torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)


def _silu(x): return x / (1 + np.exp(-x))
def _sig(x): return 1 / (1 + np.exp(-x))
def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)  # стабильный


def _causal_conv_silu(x, w):  # x [T,C], w [C,1,K] → [T,C], causal + SiLU
    T, C = x.shape; K = w.shape[-1]
    xp = np.pad(x.T, ((0, 0), (K - 1, 0)))  # [C, T+K-1]
    out = np.zeros((C, T))
    for j in range(K):
        out += xp[:, j:j + T] * w[:, 0, j][:, None]
    return _silu(out.T)


def _recurrent_gdr(q, k, v, g, beta):  # q,k [T,H,Dk]; v [T,H,Dv]; g,beta [T,H]
    def l2n(x): return x / np.sqrt((x ** 2).sum(-1, keepdims=True) + 1e-6)
    q, k = l2n(q), l2n(k)
    T, H, Dk = k.shape; Dv = v.shape[-1]
    q = q.astype(np.float64) / np.sqrt(Dk); k = k.astype(np.float64); v = v.astype(np.float64)
    g = g.astype(np.float64); beta = beta.astype(np.float64)
    out = np.zeros((T, H, Dv)); S = np.zeros((H, Dk, Dv))
    for i in range(T):
        S = S * np.exp(g[i])[:, None, None]
        kv = (S * k[i][:, :, None]).sum(1)                 # [H,Dv]
        delta = (v[i] - kv) * beta[i][:, None]             # [H,Dv]
        S = S + k[i][:, :, None] * delta[:, None, :]       # [H,Dk,Dv]
        out[i] = (S * q[i][:, :, None]).sum(1)
    return out


def gdn_layer_np(x, W, nk, nv, dk, dv, eps=1e-6):
    T, H = x.shape; kd, vd = nk * dk, nv * dv; rep = nv // nk
    qkv = x @ W["in_proj_qkv"].T
    z = x @ W["in_proj_z"].T
    a = x @ W["in_proj_a"].T
    b = x @ W["in_proj_b"].T
    qkv = _causal_conv_silu(qkv, W["conv1d"])
    q, k, v = qkv[:, :kd], qkv[:, kd:2 * kd], qkv[:, 2 * kd:]
    q = q.reshape(T, nk, dk); k = k.reshape(T, nk, dk); v = v.reshape(T, nv, dv)
    beta = _sig(b); g = -np.exp(W["A_log"]) * _softplus(a + W["dt_bias"])
    q = np.repeat(q, rep, axis=1); k = np.repeat(k, rep, axis=1)
    o = _recurrent_gdr(q, k, v, g, beta)
    z = z.reshape(T, nv, dv)
    var = (o ** 2).mean(-1, keepdims=True)
    o = W["norm"] * (o / np.sqrt(var + eps)) * _silu(z)
    return (o.reshape(T, vd) @ W["out_proj"].T).astype(np.float32)


def gdn_layer_torch(x, Wt, nk, nv, dk, dv, eps=1e-6):
    T, H = x.shape; kd, vd = nk * dk, nv * dv; rep = nv // nk
    qkv = x @ Wt["in_proj_qkv"].T; z = x @ Wt["in_proj_z"].T
    a = x @ Wt["in_proj_a"].T; b = x @ Wt["in_proj_b"].T
    qkv = F.silu(F.conv1d(qkv.T[None], Wt["conv1d"], groups=kd * 2 + vd, padding=dk * 0 + Wt["conv1d"].shape[-1] - 1)[0, :, :T].T)
    q, k, v = torch.split(qkv, [kd, kd, vd], -1)
    q = q.reshape(T, nk, dk); k = k.reshape(T, nk, dk); v = v.reshape(T, nv, dv)
    beta = b.sigmoid(); g = -Wt["A_log"].exp() * F.softplus(a + Wt["dt_bias"])
    q = q.repeat_interleave(rep, 1); k = k.repeat_interleave(rep, 1)

    def l2n(t): return t / (t.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    q, k = l2n(q), l2n(k); q = q / dk ** 0.5
    o = torch.zeros(T, nv, dv); S = torch.zeros(nv, dk, dv)
    for i in range(T):
        S = S * g[i].exp()[:, None, None]
        kv = (S * k[i][:, :, None]).sum(1)
        delta = (v[i] - kv) * beta[i][:, None]
        S = S + k[i][:, :, None] * delta[:, None, :]
        o[i] = (S * q[i][:, :, None]).sum(1)
    z = z.reshape(T, nv, dv); var = o.pow(2).mean(-1, keepdim=True)
    o = Wt["norm"] * (o * torch.rsqrt(var + eps)) * F.silu(z)
    return (o.reshape(T, vd) @ Wt["out_proj"].T)


def main():
    H, nk, nv, dk, dv, K, T = 128, 4, 12, 32, 32, 4, 20
    kd, vd = nk * dk, nv * dv
    rng = np.random.RandomState(0)
    W = {"in_proj_qkv": rng.randn(2 * kd + vd, H) * 0.05, "in_proj_z": rng.randn(vd, H) * 0.05,
         "in_proj_a": rng.randn(nv, H) * 0.05, "in_proj_b": rng.randn(nv, H) * 0.05,
         "A_log": rng.rand(nv) * 2.0, "dt_bias": rng.randn(nv) * 0.3,
         "conv1d": rng.randn(2 * kd + vd, 1, K) * 0.3, "norm": np.ones(dv, np.float32),
         "out_proj": rng.randn(H, vd) * 0.05}
    W = {k: v.astype(np.float32) for k, v in W.items()}
    x = (rng.randn(T, H) * 0.5).astype(np.float32)
    o_np = gdn_layer_np(x, W, nk, nv, dk, dv)
    Wt = {k: torch.tensor(v) for k, v in W.items()}
    o_t = gdn_layer_torch(torch.tensor(x), Wt, nk, nv, dk, dv).detach().numpy()
    rel = np.abs(o_np - o_t).max() / (np.abs(o_t).max() + 1e-9)
    print(f"GatedDeltaNet ПОЛНЫЙ слой: numpy vs torch  max_rel={rel:.2e}  out_std={o_t.std():.3f}  "
          f"{'OK' if rel < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
