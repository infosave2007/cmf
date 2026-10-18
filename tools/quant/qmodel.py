"""Torch model over a REAL CMF file for the offline quantization tools (metric refit,
escort sensitivity, exact perplexity).

Every quantized weight is decoded on the device from the file's own payload with the
decoder's exact f32 arithmetic (cmfcodec.py), so with all trainable deltas at zero the
model computes what cortiq computes up to f32 summation order. Trainable values are the
f16 fields the file already stores (log2 units, init 0):

  a[r]   per output row of a q4tp / q2tp linear: row `lo` += a (an exact row gain, the rung
         scales are 2^(lo + k*step)); of a q8_2f linear: row scale *= 2^a
  b[r]   (step training) per row `step` += b of a q4tp / q2tp linear, clamped b >= -step/2
  n[c]   per channel of an RMSNorm gain: effective gain (g + o) *= 2^n, o = 1 for the
         zero-centred x*(1+w) norms (Gemma, Qwen3.5), 0 otherwise

Architectures (from the CMF header `arch`, tensor names = cortiq's canonical HF names):
  - llama family: llama / mistral / MiniCPM5 / qwen2 (q/k/v biases) / qwen3 (q/k norms)
  - qwen3_5 hybrid (Qwen3.5 / Qwen3.8 dense): Gated DeltaNet layers (conv + delta rule +
    gated RMSNorm) and gated full attention (q_proj carries [query, gate] per head, q/k
    norms, partial RoPE), all norms zero-centred except the GDN output norm.

Residency: `resident` keeps every layer's compact weights (packed codes + f32 group scales)
on the device; `stream` keeps them pinned in host RAM and moves one decoder layer at a time
(prefetching the next on a side stream). Both run the same layer-major schedule with the
same kernels, so they produce bit-identical numbers; boundary activations live on
`act_device` (host RAM by default when streaming).
"""
import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import cmfcodec as cc

TILED = ("Q4TiledP", "Q2TiledP")
FLOATS = ("F32", "F16", "Bf16")


def setup_torch(seed=0, tf32=True, deterministic=True, threads=None):
    """Deterministic kernels (the streaming/resident equivalence and reruns rely on it)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    if threads:
        torch.set_num_threads(threads)


def resolve_device(name="cuda", cap_gb=0.0):
    """torch.device with an explicit index; with cap_gb > 0 the process's CUDA allocator is
    capped (a GPU shared with other jobs)."""
    dev = torch.device(name)
    if dev.type == "cuda":
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        if cap_gb:
            tot = torch.cuda.get_device_properties(dev).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 2 ** 30 / tot), dev.index)
    return dev


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)


# ----------------------------------------------------------------------------- architecture
class Arch:
    """Geometry and tensor naming from a CmfFile header + directory."""

    def __init__(self, cmf):
        a = cmf.arch
        self.name = a.get("arch_name", "")
        unsupported = [k for k in ("yarn", "mla", "moe", "qwen4_exp", "g3n", "prism_hadamard", "anchor_core",
                                   "final_logit_softcapping", "attn_logit_softcapping", "head_clusters",
                                   "attention_heads_per_layer", "kv_heads_per_layer", "v_head_dim",
                                   "sliding_window", "global_head_dim", "query_pre_attn_scalar", "mtp_only")
                       if a.get(k) not in (None, False, 0)]
        if a.get("num_loops", 1) not in (1, None):
            unsupported.append("num_loops")
        if a.get("hidden_act", "silu") != "silu":
            unsupported.append(f"hidden_act={a.get('hidden_act')}")
        if a.get("qk_norm_after_rope"):
            unsupported.append("qk_norm_after_rope")
        if unsupported:
            raise SystemExit(f"{self.name}: not supported by the refit tools yet: {unsupported}")
        self.hidden = a["hidden_size"]
        self.n_layers = a["num_layers"]
        self.heads = a["num_attention_heads"]
        self.kv = a["num_kv_heads"]
        self.hd = a["head_dim"]
        self.vocab = a["vocab_size"]
        self.eps = float(a.get("rms_norm_eps", 1e-6))
        self.theta = float(a.get("rope_theta", 10000.0))
        self.rot = int(self.hd * float(a.get("partial_rotary_factor", 1.0)))
        self.freq_factors = a.get("rope_freq_factors")
        self.norm_offset = 1.0 if str(a.get("norm_style", "qwen")).lower() == "gemma" else 0.0
        self.embed_mult = float(a.get("embed_multiplier", 1.0) or 1.0)
        self.logit_mult = float(a.get("logit_multiplier") or 1.0)
        types = a.get("layer_types") or ["FullAttention"] * self.n_layers
        self.kinds = []
        for t in types:
            if t == "FullAttention":
                self.kinds.append("attn")
            elif t == "LinearAttention":
                self.kinds.append("gdn")
            else:
                raise SystemExit(f"{self.name}: layer type {t} not supported by the refit tools yet")
        if "gdn" in self.kinds:
            core = a.get("linear_core") or {}
            if core and core.get("kind") not in (None, "gated_delta_net"):
                raise SystemExit(f"linear core {core.get('kind')} not supported (gated_delta_net only)")
            self.nk = a["linear_num_key_heads"]
            self.nv = a["linear_num_value_heads"]
            self.dk = a["linear_key_head_dim"]
            self.dv = a["linear_value_head_dim"]
            self.conv_k = a.get("linear_conv_kernel_dim") or 4
        self.head_name = "lm_head.weight" if "lm_head.weight" in cmf else "model.embed_tokens.weight"
        self.embed_name = "model.embed_tokens.weight"
        self.final_norm = "model.norm.weight"
        p0 = self._first("attn")
        self.attn_gate = False
        if p0 is not None:
            q = cmf.tensor(f"model.layers.{p0}.self_attn.q_proj.weight")
            self.attn_gate = q.shape[0] == 2 * self.heads * self.hd
            if not self.attn_gate and q.shape[0] != self.heads * self.hd:
                raise SystemExit(f"q_proj rows {q.shape[0]} fit neither heads*hd nor 2*heads*hd")

    def _first(self, kind):
        for l, k in enumerate(self.kinds):
            if k == kind:
                return l
        return None

    def layer_spec(self, l, cmf):
        """(linears, norms, misc) of layer l: lists of (role, tensor name[, norm offset])."""
        p = f"model.layers.{l}."
        lin, norms, misc = [], [], []
        norms.append(("in", p + "input_layernorm.weight", self.norm_offset))
        norms.append(("post", p + "post_attention_layernorm.weight", self.norm_offset))
        if self.kinds[l] == "attn":
            for r in ("q", "k", "v", "o"):
                lin.append((r, p + f"self_attn.{r}_proj.weight"))
                if p + f"self_attn.{r}_proj.bias" in cmf:
                    misc.append((r + "_bias", p + f"self_attn.{r}_proj.bias"))
            for r in ("q", "k"):
                if p + f"self_attn.{r}_norm.weight" in cmf:
                    norms.append((r + "_norm", p + f"self_attn.{r}_norm.weight", self.norm_offset))
        else:
            for r in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                lin.append((r, p + f"linear_attn.{r}.weight"))
            misc += [("conv1d", p + "linear_attn.conv1d.weight"), ("A_log", p + "linear_attn.A_log"),
                     ("dt_bias", p + "linear_attn.dt_bias")]
            # the GDN output norm is a plain x*w gain whatever the model's norm style
            norms.append(("gdn_norm", p + "linear_attn.norm.weight", 0.0))
        for r in ("gate", "up", "down"):
            lin.append((r, p + f"mlp.{r}_proj.weight"))
        for _, n, *_ in lin + misc + norms:
            if n not in cmf:
                raise SystemExit(f"{cmf.path}: missing tensor {n} for a {self.kinds[l]} layer")
        return lin, norms, misc


# ----------------------------------------------------------------------------- weights
def _host(arr, pin):
    """host tensor for a weight section: a page-locked copy when it will be streamed to the
    device every step (pin), otherwise zero-copy over the array — for file sections that is
    the read-only mmap, so an unpinned 54 GB teacher costs page cache, not RAM"""
    arr = np.ascontiguousarray(arr)
    if pin and torch.cuda.is_available():
        return torch.from_numpy(np.array(arr)).pin_memory()
    if not arr.flags.writeable:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)    # never written through
            return torch.from_numpy(arr)
    return torch.from_numpy(arr)


class QLinear:
    """One linear from the file. Host copies are compact (packed codes, f32 group scales,
    rung powers); `load` moves them to the device, `release` drops them."""

    def __init__(self, name, tensor, payload, train=False, step=False, pin=True):
        self.name, self.dtype = name, tensor.dtype
        if len(tensor.shape) != 2:
            raise SystemExit(f"{name}: linear expected 2-D, got {tensor.shape}")
        self.rows, self.cols = tensor.shape
        self.payload = payload            # zero-copy view, kept for the write-back
        self.a = self.b = None
        self.dev = {}
        if self.dtype in TILED:
            t = self.tiled = cc.Tiled(self.dtype, payload, self.rows, self.cols)
            self.host = {"codes": _host(t.codes, pin), "s0": _host(t.scales(), pin),
                         "k": _host(t.rung_power(), pin)}
            self.st = cc.f16_val(t.st).astype(np.float32)
            self.trainable = bool(train)
            self.step = bool(train and step)
        elif self.dtype == "Q8_2f":
            q = self.q8 = cc.Q8_2f(payload, self.rows, self.cols)
            self.host = {"q": _host(q.q, pin), "rs": _host(q.rs, pin), "cs": _host(q.cs, pin)}
            self.trainable, self.step = bool(train), False
        elif self.dtype in FLOATS:
            raw = np.asarray(payload).view("<u2" if self.dtype != "F32" else "<f4").reshape(self.rows, self.cols)
            self.host = {"w": _host(raw.view(np.int16) if self.dtype != "F32" else raw, pin)}
            self.trainable, self.step = False, False
        else:
            raise SystemExit(f"{name}: dtype {self.dtype} not supported by the refit tools")

    # -- parameters (live on the device for the whole run)
    def init_params(self, dev):
        if self.trainable:
            self.a = torch.zeros(self.rows, device=dev, requires_grad=True)
            if self.step:
                self.b = torch.zeros(self.rows, device=dev, requires_grad=True)
                self.st_t = torch.from_numpy(self.st).to(dev)

    def params(self):
        return [p for p in (self.a, self.b) if p is not None]

    def b_clamped(self):
        return torch.maximum(self.b, -0.5 * self.st_t)

    # -- residency
    def load(self, dev, non_blocking=False):
        if not self.dev:
            self.dev = {k: v.to(dev, non_blocking=non_blocking) for k, v in self.host.items()}

    def release(self):
        self.dev = {}

    def record_stream(self, stream):
        for v in self.dev.values():
            v.record_stream(stream)

    def device_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.host.values())

    # -- decode
    def scale(self):
        """group / row scale with the trainable deltas (autograd graph to a, b) or None"""
        d = self.dev
        if self.dtype in TILED:
            if self.a is None:
                return d["s0"]
            e = self.a[:, None]
            if self.b is not None:
                e = e + d["k"].float() * self.b_clamped()[:, None]
            return d["s0"] * torch.exp2(e)
        if self.dtype == "Q8_2f":
            return d["rs"] if self.a is None else d["rs"] * torch.exp2(self.a)
        return None

    def levels(self, r0=0, r1=None):
        """code multipliers [rows, cols] (q8_2f: the int8 codes as f32), unpacked straight into
        one f32 buffer (no per-field temporaries)"""
        d = self.dev
        r1 = self.rows if r1 is None else r1
        n = r1 - r0
        if self.dtype in TILED:
            c = d["codes"][r0:r1]
            out = torch.empty(n, self.cols, device=c.device)
            if self.dtype == "Q4TiledP":
                o = out.view(n, self.cols // 2, 2)
                o[..., 0].copy_(c & 15)
                o[..., 1].copy_(c >> 4)
                return out.sub_(8.0)
            o = out.view(n, self.cols // 4, 4)
            for k in range(4):
                o[..., k].copy_((c >> (2 * k)) & 3)
            return out.sub_(1.5)
        if self.dtype == "Q8_2f":
            return d["q"][r0:r1].float()
        raise AssertionError(self.dtype)

    def weight(self, S=None, r0=0, r1=None):
        """f32 weight rows [r0, r1) on the device, the decoder's arithmetic (in place on the
        unpacked levels unless an autograd graph through S is being recorded)"""
        d = self.dev
        r1 = self.rows if r1 is None else r1
        n = r1 - r0
        graph = S is not None and S.requires_grad and torch.is_grad_enabled()
        if self.dtype in TILED:
            s = (d["s0"] if S is None else S)[r0:r1]
            lv = self.levels(r0, r1).view(n, -1, cc.G)
            return (lv * s[:, :, None] if graph else lv.mul_(s[:, :, None])).view(n, self.cols)
        if self.dtype == "Q8_2f":
            s = (d["rs"] if S is None else S)[r0:r1]
            if graph:
                return (d["q"][r0:r1].float() * s[:, None]) * d["cs"][None, :]
            return d["q"][r0:r1].float().mul_(s[:, None]).mul_(d["cs"][None, :])
        w = d["w"][r0:r1]
        if self.dtype == "F16":
            return w.view(torch.float16).float()
        if self.dtype == "Bf16":
            return (w.to(torch.int32) << 16).view(torch.float32)
        return w

    def scale_grad(self, dW, r0=0, r1=None):
        """d loss / d scale of rows [r0, r1) from d loss / d W of those rows (codes frozen)"""
        r1 = self.rows if r1 is None else r1
        n = r1 - r0
        if self.dtype in TILED:
            return dW.view(n, -1, cc.G).mul_(self.levels(r0, r1).view(n, -1, cc.G)).sum(-1)
        return dW.mul_(self.dev["q"][r0:r1].float().mul_(self.dev["cs"][None, :])).sum(1)

    def row_blocks(self):
        """row ranges decoded at a time: at most ~ROW_BLOCK_ELEMS f32 weights per block, so
        the transient decode of a 17408 x 5120 projection never exceeds ~128 MB"""
        step = max(64, ROW_BLOCK_ELEMS // max(1, self.cols))
        return [(r0, min(r0 + step, self.rows)) for r0 in range(0, self.rows, step)]

    def __call__(self, x2):
        S = self.scale()
        return _QMatmul.apply(x2, S, self)

    # -- write-back
    def snapped(self, a, b):
        """the f16 fields the file will hold for trained (a, b), and the deltas they realise:
        returns (payload bytes, a_eff, b_eff) — a_eff/b_eff in the same log2 units."""
        a = np.asarray(a, np.float64)
        if self.dtype in TILED:
            t = self.tiled
            lo = cc.f16_val(t.lo).astype(np.float64)
            lo2 = cc.f16_bits(lo + a)
            st = cc.f16_val(t.st).astype(np.float64)
            st2 = t.st if b is None else cc.f16_bits(st + np.maximum(np.asarray(b, np.float64), -0.5 * st))
            a_eff = cc.f16_val(lo2).astype(np.float64) - lo
            b_eff = None if b is None else cc.f16_val(st2).astype(np.float64) - st
            return t.with_row_params(lo2, st2), a_eff, b_eff
        q = self.q8
        rs = q.rs.astype(np.float64)
        rs2 = cc.f16_bits(rs * np.exp2(a))
        v2 = cc.f16_val(rs2).astype(np.float64)
        ok = (rs > 0) & (v2 > 0)
        a_eff = np.where(ok, np.log2(np.where(ok, v2, 1.0) / np.where(ok, rs, 1.0)), 0.0)
        return q.with_row_scales(rs2), a_eff, None


# f32 weights decoded at a time by the linear (rows are split into blocks above this)
ROW_BLOCK_ELEMS = 1 << 25


class _QMatmul(torch.autograd.Function):
    """y = x @ W(S)^T with W re-decoded in backward instead of saved: only x and the small
    scale tensor are kept, and W is decoded in row blocks (`QLinear.row_blocks`), so a
    layer's forward or backward never holds more than one block of f32 weight."""

    @staticmethod
    def forward(ctx, x, S, lin):
        ctx.lin = lin
        ctx.save_for_backward(x, S if S is not None else torch.empty(0))
        ctx.has_s = S is not None
        blocks = lin.row_blocks()
        if len(blocks) == 1:
            return x @ lin.weight(S).t()
        y = x.new_empty(x.shape[0], lin.rows)
        for r0, r1 in blocks:
            y[:, r0:r1] = x @ lin.weight(S, r0, r1).t()
        return y

    @staticmethod
    def backward(ctx, gy):
        x, S = ctx.saved_tensors
        lin = ctx.lin
        S = S if ctx.has_s else None
        want_x = ctx.needs_input_grad[0]
        want_s = ctx.has_s and ctx.needs_input_grad[1]
        blocks = lin.row_blocks()
        if len(blocks) == 1:
            W = lin.weight(S)
            gx = gy @ W if want_x else None
            del W
            gS = lin.scale_grad(gy.t() @ x) if want_s else None
            return gx, gS, None
        gx = torch.zeros_like(x) if want_x else None
        gS = torch.empty_like(S) if want_s else None
        for r0, r1 in blocks:
            g = gy[:, r0:r1]
            if want_x:
                gx += g @ lin.weight(S, r0, r1)
            if want_s:
                gS[r0:r1] = lin.scale_grad(g.t() @ x, r0, r1)
        return gx, gS, None


class Gain:
    """An RMSNorm gain from the file (any float dtype); effective gain (g + offset) * 2^n."""

    def __init__(self, name, tensor, payload, offset, train=False):
        self.name, self.dtype, self.offset = name, tensor.dtype, float(offset)
        if self.dtype not in FLOATS:
            raise SystemExit(f"{name}: norm gain dtype {self.dtype} not supported")
        self.shape = tensor.shape
        self.g0 = cc.decode_float(self.dtype, payload, tensor.shape).astype(np.float32).ravel()
        self.trainable = bool(train)
        self.n = None
        self.base = None

    def init_params(self, dev):
        self.base = torch.from_numpy(self.g0 + np.float32(self.offset)).to(dev)
        if self.trainable:
            self.n = torch.zeros(self.g0.size, device=dev, requires_grad=True)

    def params(self):
        return [] if self.n is None else [self.n]

    def eff(self):
        return self.base if self.n is None else self.base * torch.exp2(self.n)

    def snapped(self, n):
        """(payload bytes, n_eff) for a trained n"""
        g0 = self.g0.astype(np.float64) + self.offset
        stored = cc.round_float(self.dtype, g0 * np.exp2(np.asarray(n, np.float64)) - self.offset).astype(np.float64)
        g1 = stored + self.offset
        ok = (g0 != 0) & (g1 != 0) & (np.sign(g0) == np.sign(g1))
        n_eff = np.where(ok, np.log2(np.abs(np.where(ok, g1, 1.0)) / np.abs(np.where(ok, g0, 1.0))), 0.0)
        return cc.encode_float(self.dtype, stored), n_eff


def _float_tensor(cmf, name, dev):
    t = cmf.tensor(name)
    if t.dtype in FLOATS:
        v = cc.decode_float(t.dtype, cmf.payload(name), t.shape)
    else:
        v = cc.dequant(t.dtype, cmf.payload(name), t.shape)
    return torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(dev)


# ----------------------------------------------------------------------------- forward pieces
def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


_ROPE = {}


def rope_tables(T, rot, theta, factors, dev):
    key = (T, rot, theta, None if factors is None else tuple(factors), str(dev))
    if key not in _ROPE:
        inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64, device=dev) / rot))
        if factors is not None:
            inv = inv / torch.tensor(factors, dtype=torch.float64, device=dev)[: inv.numel()]
        t = torch.arange(T, dtype=torch.float64, device=dev)
        fr = torch.outer(t, inv)
        emb = torch.cat([fr, fr], -1)
        _ROPE[key] = (emb.cos().float(), emb.sin().float())
    return _ROPE[key]


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def apply_rope(x, cos, sin, rot):
    if rot == x.shape[-1]:
        return x * cos + rot_half(x) * sin
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat([xr * cos + rot_half(xr) * sin, xp], -1)


_MASK = {}


def causal_attention(q, k, v):
    """q [B, h, T, d], k/v [B, kv, T, d]; explicit softmax attention (deterministic)."""
    B, h, T, d = q.shape
    rep = h // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    key = (T, str(q.device))
    if key not in _MASK:
        _MASK[key] = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(d))
    s = s.masked_fill(_MASK[key], float("-inf"))
    return torch.softmax(s, -1) @ v


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def chunk_gated_delta_rule(q, k, v, g, beta, chunk=64):
    """Gated delta rule over a whole window (fresh state), chunked form of
    S_t = g_t S_{t-1} + k_t (beta_t (v_t - S_{t-1}^T k_t))^T, o_t = S_t^T q_t / sqrt(dk), with
    l2-normalised q/k. q/k [B, T, h, dk], v [B, T, h, dv], g (log decay) / beta [B, T, h].
    The in-chunk recursion is a unit-lower-triangular solve (exact rewrite of the reference
    forward-substitution loop)."""
    q, k = l2norm(q.float()), l2norm(k.float())
    q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v.float())]
    beta, g = beta.float().transpose(1, 2).contiguous(), g.float().transpose(1, 2).contiguous()
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
        beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    n = (T + pad) // chunk
    q = q * (dk ** -0.5)
    kb, vb = k * beta[..., None], v * beta[..., None]
    q, k, v, kb, vb = [x.reshape(B, H, n, chunk, x.shape[-1]) for x in (q, k, v, kb, vb)]
    g = g.reshape(B, H, n, chunk).cumsum(-1)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(tri, (g[..., :, None] - g[..., None, :]), torch.zeros((), device=q.device)).exp() * tri
    strict = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
    A = torch.where(strict, (kb @ k.transpose(-1, -2)) * decay, torch.zeros((), device=q.device))
    eye = torch.eye(chunk, device=q.device).expand_as(A)
    T_ = torch.linalg.solve_triangular(eye + A, eye, upper=False, unitriangular=True)
    u = T_ @ vb
    w = T_ @ (kb * g.exp()[..., None])
    S = torch.zeros(B, H, dk, dv, device=q.device)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    outs = []
    for i in range(n):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        att = (qi @ ki.transpose(-1, -2) * decay[:, :, i]).masked_fill(upper, 0.0)
        v_new = u[:, :, i] - w[:, :, i] @ S
        outs.append((qi * gi[..., None].exp()) @ S + att @ v_new)
        S = S * gi[..., -1, None, None].exp() + (ki * (gi[..., -1, None] - gi).exp()[..., None]).transpose(-1, -2) @ v_new
    o = torch.stack(outs, 2).reshape(B, H, n * chunk, dv)[:, :, :T]
    return o.transpose(1, 2)


class Layer:
    """One decoder layer: its linears, gains and small frozen tensors."""

    def __init__(self, arch, l, cmf, train=False, step=False, train_norms=False, pin=True):
        self.arch, self.l, self.kind = arch, l, arch.kinds[l]
        lin, norms, misc = arch.layer_spec(l, cmf)
        self.lin = {}
        for role, n in lin:
            t = cmf.tensor(n)
            self.lin[role] = QLinear(n, t, cmf.payload(n), train=train, step=step, pin=pin)
        self.gain = {role: Gain(n, cmf.tensor(n), cmf.payload(n), off, train=train_norms) for role, n, off in norms}
        self.misc_names = dict(misc)
        self.misc = {}
        self.cmf = cmf

    def init_params(self, dev):
        for x in list(self.lin.values()) + list(self.gain.values()):
            x.init_params(dev)
        for role, n in self.misc_names.items():
            self.misc[role] = _float_tensor(self.cmf, n, dev)
        if "conv1d" in self.misc:
            self.misc["conv1d"] = self.misc["conv1d"].reshape(self.misc["conv1d"].shape[0], -1)

    def params(self):
        out = []
        for x in list(self.lin.values()) + list(self.gain.values()):
            out += x.params()
        return out

    def load(self, dev, non_blocking=False):
        for x in self.lin.values():
            x.load(dev, non_blocking)

    def release(self):
        for x in self.lin.values():
            x.release()

    def record_stream(self, stream):
        for x in self.lin.values():
            x.record_stream(stream)

    def device_bytes(self):
        return sum(x.device_bytes() for x in self.lin.values())

    # -- forward
    def _lin(self, role, x2):
        y = self.lin[role](x2)
        b = self.misc.get(role + "_bias")
        return y if b is None else y + b

    def forward(self, x):
        A = self.arch
        B, T, H = x.shape
        a2 = rms(x, self.gain["in"].eff(), A.eps).reshape(B * T, H)
        mix = self._attn(a2, B, T) if self.kind == "attn" else self._gdn(a2, B, T)
        x = x + mix.view(B, T, H)
        m = rms(x, self.gain["post"].eff(), A.eps).reshape(B * T, H)
        h = F.silu(self._lin("gate", m)) * self._lin("up", m)
        return x + self._lin("down", h).view(B, T, H)

    def _attn(self, a2, B, T):
        A = self.arch
        nh, nkv, hd = A.heads, A.kv, A.hd
        q = self._lin("q", a2)
        gate = None
        if A.attn_gate:
            qg = q.view(B, T, nh, 2 * hd)
            q, gate = qg[..., :hd], qg[..., hd:].reshape(B, T, nh * hd)
        q = q.reshape(B, T, nh, hd)
        k = self._lin("k", a2).view(B, T, nkv, hd)
        v = self._lin("v", a2).view(B, T, nkv, hd)
        if "q_norm" in self.gain:
            q = rms(q, self.gain["q_norm"].eff(), A.eps)
            k = rms(k, self.gain["k_norm"].eff(), A.eps)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = rope_tables(T, A.rot, A.theta, A.freq_factors, a2.device)
        q, k = apply_rope(q, cos, sin, A.rot), apply_rope(k, cos, sin, A.rot)
        o = causal_attention(q, k, v).transpose(1, 2).reshape(B * T, nh * hd)
        if gate is not None:
            o = o * torch.sigmoid(gate.reshape(B * T, nh * hd))
        return self._lin("o", o)

    def _gdn(self, a2, B, T):
        A = self.arch
        nk, nv, dk, dv, kk = A.nk, A.nv, A.dk, A.dv, A.conv_k
        mixed = self._lin("in_proj_qkv", a2).view(B, T, -1)
        w = self.misc["conv1d"]                     # [C, kk], tap kk-1 multiplies the current position
        pre = mixed * w[:, kk - 1]
        for j in range(kk - 1):
            s = kk - 1 - j
            pre = pre + F.pad(mixed[:, :T - s], (0, 0, s, 0)) * w[:, j]
        act = F.silu(pre)
        q = act[..., :nk * dk].reshape(B, T, nk, dk)
        k = act[..., nk * dk:2 * nk * dk].reshape(B, T, nk, dk)
        v = act[..., 2 * nk * dk:].reshape(B, T, nv, dv)
        z = self._lin("in_proj_z", a2).view(B, T, nv, dv)
        beta = torch.sigmoid(self._lin("in_proj_b", a2).view(B, T, nv))
        g = -torch.exp(self.misc["A_log"]) * F.softplus(self._lin("in_proj_a", a2).view(B, T, nv) + self.misc["dt_bias"])
        if nv > nk:
            q = q.repeat_interleave(nv // nk, 2)
            k = k.repeat_interleave(nv // nk, 2)
        o = chunk_gated_delta_rule(q, k, v, g, beta)
        o = rms(o, self.gain["gdn_norm"].eff(), A.eps) * F.silu(z)
        return self._lin("out_proj", o.reshape(B * T, nv * dv))


# ----------------------------------------------------------------------------- embedding / head
class Embed:
    """Embedding rows decoded on the host from the file payload (only the rows used)."""

    def __init__(self, cmf, name, mult):
        self.t = cmf.tensor(name)
        self.payload = cmf.payload(name)
        self.mult = mult
        rows, cols = self.t.shape
        if self.t.dtype in TILED:
            self.tiled = cc.Tiled(self.t.dtype, self.payload, rows, cols)
            self.scales = self.tiled.scales()
        elif self.t.dtype == "Q8_2f":
            self.q8 = cc.Q8_2f(self.payload, rows, cols)

    def rows(self, ids):
        """ids: int array -> f32 [len(ids), hidden] (numpy)"""
        dt = self.t.dtype
        rows, cols = self.t.shape
        ids = np.asarray(ids, dtype=np.int64)
        u, inv = np.unique(ids, return_inverse=True)
        if dt in TILED:
            lv = cc.unpack_levels(dt, self.tiled.codes[u], cols)
            w = (lv.reshape(len(u), -1, cc.G) * self.scales[u][:, :, None]).reshape(len(u), cols)
        elif dt == "Q8_2f":
            q = self.q8
            w = q.q[u].astype(np.float32) * q.rs[u][:, None] * q.cs[None, :]
        else:
            # decode only the rows used (a 248k x 5120 table is 5 GB in f32)
            width = 4 if dt == "F32" else 2
            sel = np.asarray(self.payload).reshape(rows, cols * width)[u]
            w = cc.decode_float(dt, sel.reshape(-1), (len(u), cols))
        w = w[inv]
        if self.mult != 1.0:
            w = w * np.float32(self.mult)
        return w.astype(np.float32)


class Head:
    """The output projection, compact on the device, decoded per vocab chunk."""

    def __init__(self, cmf, name, logit_mult, pin=True):
        t = cmf.tensor(name)
        self.lin = QLinear(name, t, cmf.payload(name), train=False, pin=pin)
        self.vocab = t.shape[0]
        self.mult = logit_mult

    def load(self, dev):
        self.lin.load(dev)

    def rows(self, r0, r1):
        return self.lin.weight(None, r0, r1)

    def logits(self, xn, r0, r1):
        z = xn @ self.rows(r0, r1).t()
        return z * self.mult if self.mult != 1.0 else z

    def scan(self, xn, vchunk, ix=None, tgt=None, topk=0):
        """one pass over the vocabulary: log-sum-exp per position, the logits at the
        teacher indices `ix` [P, K] and at the targets `tgt` [P], and (topk > 0) the top-k
        logits and their indices."""
        P = xn.shape[0]
        lse_parts = []
        z_ix = None if ix is None else torch.zeros(ix.shape, device=xn.device)
        z_t = None if tgt is None else torch.zeros(P, device=xn.device)
        tv = ti = None
        for c0 in range(0, self.vocab, vchunk):
            c1 = min(c0 + vchunk, self.vocab)
            z = self.logits(xn, c0, c1)
            lse_parts.append(torch.logsumexp(z, 1))
            # masked gathers instead of nonzero(): no host synchronisation per chunk
            if ix is not None:
                inside = (ix >= c0) & (ix < c1)
                z_ix = torch.where(inside, z.gather(1, (ix - c0).clamp(0, c1 - c0 - 1)), z_ix)
            if tgt is not None:
                inside = (tgt >= c0) & (tgt < c1)
                z_t = torch.where(inside, z.gather(1, (tgt - c0).clamp(0, c1 - c0 - 1)[:, None])[:, 0], z_t)
            if topk:
                v, i = z.topk(min(topk, c1 - c0), 1)
                i = i + c0
                if tv is None:
                    tv, ti = v, i
                else:
                    v, j = torch.cat([tv, v], 1).topk(topk, 1)
                    tv, ti = v, torch.cat([ti, i], 1).gather(1, j)
        lse = torch.logsumexp(torch.stack(lse_parts, 1), 1)
        return lse, z_ix, z_t, tv, ti

    def grad_input(self, xn, lse, alpha, ix, sparse, vchunk):
        """sum_j dL/dz_j W_j for dL/dz_j = q_j alpha + [j in ix] sparse (chunked)"""
        gx = torch.zeros_like(xn)
        for c0 in range(0, self.vocab, vchunk):
            c1 = min(c0 + vchunk, self.vocab)
            W = self.rows(c0, c1)
            z = xn @ W.t()
            if self.mult != 1.0:
                z = z * self.mult
            G = torch.exp(z - lse[:, None]) * alpha[:, None]
            # the top-K indices of a row are distinct: scatter-add touches each in-chunk entry once,
            # the out-of-chunk ones (clamped) receive +0
            inside = (ix >= c0) & (ix < c1)
            G.scatter_add_(1, (ix - c0).clamp(0, c1 - c0 - 1), torch.where(inside, sparse, torch.zeros_like(sparse)))
            gx += G @ W
        return gx * self.mult if self.mult != 1.0 else gx


class _HeadKL(torch.autograd.Function):
    """sum over positions of KL(teacher top-K + tail bucket || student), with the
    vocabulary scanned in chunks in both directions (no [P, vocab] tensor is kept)."""

    @staticmethod
    def forward(ctx, xn, head, lp_t, ix_t, vchunk):
        lse, z_ix, _, _, _ = head.scan(xn, vchunk, ix=ix_t)
        lq = z_ix - lse[:, None]
        pt = lp_t.exp()
        psum = pt.sum(1)
        tail_t = (1 - psum).clamp_min(1e-7)
        S = lq.exp().sum(1)
        tail_q = (1 - S).clamp_min(1e-7)
        kl = (pt * (lp_t - lq)).sum() + (tail_t * (tail_t.log() - tail_q.log())).sum()
        ctx.head, ctx.vchunk = head, vchunk
        ctx.save_for_backward(xn, lse, lq, pt, psum, tail_t, tail_q, S, ix_t)
        return kl

    @staticmethod
    def backward(ctx, gkl):
        xn, lse, lq, pt, psum, tail_t, tail_q, S, ix_t = ctx.saved_tensors
        r = torch.where(1 - S > 1e-7, tail_t / tail_q, torch.zeros_like(tail_q))
        alpha = psum - r * S
        sparse = r[:, None] * lq.exp() - pt
        gx = ctx.head.grad_input(xn, lse, alpha, ix_t, sparse, ctx.vchunk) * gkl
        return gx, None, None, None, None


def kl_terms(lp_t, lq):
    """per-position KL(teacher top-K + tail || student) from teacher log-probs and the
    student's log-probs at the same indices (forward only)"""
    pt = lp_t.exp()
    tail_t = (1 - pt.sum(1)).clamp_min(1e-7)
    tail_q = (1 - lq.exp().sum(1)).clamp_min(1e-7)
    return (pt * (lp_t - lq)).sum(1) + tail_t * (tail_t.log() - tail_q.log())


# ----------------------------------------------------------------------------- the model
class Model:
    """A CMF file as a torch model.

    train: the q4tp/q2tp/q8_2f linears get row gains a (and with step=True the per-row step
    tilt b); train_norms: every RMSNorm gain (layer norms, q/k norms, GDN output norm, final
    norm) gets per-channel n. Embedding and head are always frozen."""

    def __init__(self, cmf, dev="cuda", residency="resident", act_device=None, train=False, step=False,
                 train_norms=False, pin=True, vchunk=16384, head_windows=1, log=print):
        self.cmf, self.dev = cmf, torch.device(dev)
        self.arch = A = Arch(cmf)
        if residency not in ("resident", "stream"):
            raise ValueError(residency)
        self.stream_mode = residency == "stream"
        self.act_device = torch.device(act_device or ("cpu" if self.stream_mode else dev))
        self.vchunk = vchunk
        # windows per vocabulary scan of the head (1 = per window; more = fewer, larger
        # kernels: faster on a launch-bound host, same values up to GEMM blocking)
        self.head_windows = head_windows
        self.log = log
        # page-locked host copies only pay off for what is streamed every step
        pin = pin and self.dev.type == "cuda" and self.stream_mode
        self.pin = pin
        self.layers = [Layer(A, l, cmf, train, step, train_norms, pin) for l in range(A.n_layers)]
        self.final = Gain(A.final_norm, cmf.tensor(A.final_norm), cmf.payload(A.final_norm), A.norm_offset,
                          train=train_norms)
        self.embed = Embed(cmf, A.embed_name, A.embed_mult)
        self.head = Head(cmf, A.head_name, A.logit_mult, pin=False)
        for x in self.layers:
            x.init_params(self.dev)
        self.final.init_params(self.dev)
        self.head.load(self.dev)
        if not self.stream_mode:
            for x in self.layers:
                x.load(self.dev)
        self.side = torch.cuda.Stream(self.dev) if (self.stream_mode and self.dev.type == "cuda") else None
        self._loaded = None

    # -- bookkeeping
    def params(self):
        out = []
        for x in self.layers:
            out += x.params()
        return out + self.final.params()

    def linears(self):
        for x in self.layers:
            yield from x.lin.values()

    def gains(self):
        for x in self.layers:
            yield from x.gain.values()
        yield self.final

    def body_bytes(self):
        return sum(x.device_bytes() for x in self.layers)

    # -- streaming
    def _prefetch(self, l):
        if not self.stream_mode or l >= len(self.layers) or l < 0:
            return
        layer = self.layers[l]
        if self.side is not None:
            with torch.cuda.stream(self.side):
                layer.load(self.dev, non_blocking=True)
        else:
            layer.load(self.dev)

    def _enter(self, l, nxt):
        """make layer l usable on the current stream; start loading layer `nxt`"""
        if self.stream_mode:
            layer = self.layers[l]
            if not layer.lin[next(iter(layer.lin))].dev:
                self._prefetch(l)
            if self.side is not None:
                torch.cuda.current_stream(self.dev).wait_stream(self.side)
                layer.record_stream(torch.cuda.current_stream(self.dev))
            self._prefetch(nxt)

    def _leave(self, l):
        if self.stream_mode:
            self.layers[l].release()

    def swap(self, l, role, qlin):
        """replace one linear (escort sensitivity); returns the previous one"""
        layer = self.layers[l]
        old = layer.lin[role]
        if old.dev:
            qlin.load(self.dev)
        qlin.init_params(self.dev)
        layer.lin[role] = qlin
        return old

    # -- activation transfers: page-locked host buffers + asynchronous copies, so parking a
    # boundary in host RAM never stalls the device queue (values are plain copies)
    def to_act(self, t):
        if self.act_device.type == "cpu" and t.device.type == "cuda":
            out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            out.copy_(t, non_blocking=True)
            return out
        return t.to(self.act_device)

    def to_dev(self, t):
        """host -> device; a page-locked boundary must arrive whole (contiguous): the CPU never
        reads it, so the copy stays ordered after the asynchronous device -> host copy"""
        if t.device.type == "cpu" and t.is_pinned() and not t.is_contiguous():
            raise AssertionError("strided slice of an in-flight page-locked activation; slice on the device")
        return t.to(self.dev, non_blocking=True)

    # -- inputs
    def embed_ids(self, ids):
        """ids [B, T] numpy -> [B, T, H] f32 on act_device"""
        B, T = ids.shape
        x = torch.from_numpy(self.embed.rows(ids.reshape(-1))).view(B, T, -1)
        return x.to(self.act_device)

    # -- forward schedules
    def forward_layers(self, xs, l0=0, l1=None, keep=False, own=False):
        """layer-major forward of micro-batches xs (list of [b, T, H] on act_device) through
        layers [l0, l1); returns the outputs (and with keep=True every boundary). Without keep
        each boundary is replaced in place as soon as its successor exists, so a pass holds one
        set of boundaries plus one micro-batch, not two; own=True also hands over the list xs
        itself (its inputs are dropped as they are consumed; xs ends up holding the outputs)."""
        l1 = len(self.layers) if l1 is None else l1
        bounds = [list(xs)] if keep else None
        cur = xs if (own and not keep) else list(xs)
        with torch.no_grad():
            for l in range(l0, l1):
                self._enter(l, l + 1 if l + 1 < l1 else -1)
                layer = self.layers[l]
                if keep:
                    cur = [self.to_act(layer.forward(self.to_dev(x))) for x in cur]
                    bounds.append(cur)
                else:
                    for j in range(len(cur)):
                        cur[j] = self.to_act(layer.forward(self.to_dev(cur[j])))
                self._leave(l)
        return bounds if keep else cur

    def head_eval(self, x, tgt, lp_t=None, ix_t=None, topk=0):
        """x [b, T, H] final hidden -> per window: CE sum, KL sum (or None) and, with topk,
        the (log-prob, index) top-k of every position (teacher caching). `head_windows`
        windows share one vocabulary scan."""
        b, T, H = x.shape
        out_ce, out_kl, tops = [], [], []
        hw = max(1, self.head_windows)
        with torch.no_grad():
            if hw == 1:
                # the per-window path, kept op for op (reproducible against earlier runs)
                for i in range(b):
                    xn = rms(self.to_dev(x[i, :T - 1]), self.final.eff(), self.arch.eps)
                    t = torch.from_numpy(np.asarray(tgt[i, 1:], dtype=np.int64)).to(self.dev)
                    ix = None if ix_t is None else ix_t[i].to(self.dev).long()
                    lse, z_ix, z_t, tv, ti = self.head.scan(xn, self.vchunk, ix=ix, tgt=t, topk=topk)
                    out_ce.append(float((lse - z_t).double().sum()))
                    if ix is not None:
                        out_kl.append(float(kl_terms(lp_t[i].to(self.dev), z_ix - lse[:, None]).double().sum()))
                    if topk:
                        tops.append(((tv - lse[:, None]).cpu(), ti.to(torch.int32).cpu()))
                return out_ce, (out_kl if ix_t is not None else None), tops
            for i0 in range(0, b, hw):
                i1 = min(b, i0 + hw)
                n = i1 - i0
                # move the contiguous block first and slice on the device: a strided slice of a
                # page-locked boundary would be gathered by the CPU while its async copy is in flight
                xd = self.to_dev(x[i0:i1])
                xn = rms(xd[:, :T - 1].reshape(n * (T - 1), H), self.final.eff(), self.arch.eps)
                t = torch.from_numpy(np.asarray(tgt[i0:i1, 1:], dtype=np.int64).reshape(-1)).to(self.dev)
                ix = None if ix_t is None else ix_t[i0:i1].reshape(n * (T - 1), -1).to(self.dev).long()
                lse, z_ix, z_t, tv, ti = self.head.scan(xn, self.vchunk, ix=ix, tgt=t, topk=topk)
                out_ce += [float(v) for v in (lse - z_t).double().view(n, T - 1).sum(1)]
                if ix is not None:
                    k = kl_terms(lp_t[i0:i1].reshape(n * (T - 1), -1).to(self.dev), z_ix - lse[:, None])
                    out_kl += [float(v) for v in k.double().view(n, T - 1).sum(1)]
                if topk:
                    lp_top = (tv - lse[:, None]).view(n, T - 1, -1).cpu()
                    ix_top = ti.to(torch.int32).view(n, T - 1, -1).cpu()
                    tops += [(lp_top[k_], ix_top[k_]) for k_ in range(n)]
        return out_ce, (out_kl if ix_t is not None else None), tops

    def evaluate(self, windows, mb=8, teacher=None, topk=0, block=None):
        """windows [N, T] ids -> dict(nll per window [N], kl per window or None, top-k)"""
        N = windows.shape[0]
        block = block or N
        nll, kl, lps, ixs = [], [], [], []
        for b0 in range(0, N, block):
            wb = windows[b0:b0 + block]
            outs = self.forward_layers([self.embed_ids(wb[i:i + mb]) for i in range(0, len(wb), mb)], own=True)
            for j, x in enumerate(outs):
                s0 = b0 + j * mb
                tg = wb[j * mb:j * mb + x.shape[0]]
                lp_t = ix_t = None
                if teacher is not None:
                    lp_t = torch.from_numpy(np.asarray(teacher[0][s0:s0 + x.shape[0]]))
                    ix_t = torch.from_numpy(np.asarray(teacher[1][s0:s0 + x.shape[0]]))
                ce, k, tops = self.head_eval(x, tg, lp_t, ix_t, topk)
                nll += ce
                if k is not None:
                    kl += k
                for lp, ix in tops:
                    lps.append(lp.numpy())
                    ixs.append(ix.numpy())
            del outs
        res = {"nll": np.array(nll), "kl": np.array(kl) if teacher is not None else None}
        if topk:
            res["lp"], res["ix"] = np.stack(lps), np.stack(ixs)
        return res

    def train_step(self, windows, lp_t, ix_t, mb, count):
        """one optimizer step's gradients: layer-major forward of the micro-batches, the
        chunked KL head on each, layer-major backward (each layer recomputed from its saved
        input). Gradients are ACCUMULATED into the parameters' .grad; returns the KL sum.
        count = the normaliser of the step's loss (windows * (T - 1))."""
        xs = [self.embed_ids(windows[i:i + mb]) for i in range(0, len(windows), mb)]
        bounds = self.forward_layers(xs, keep=True)
        T = windows.shape[1]
        grads, kl_tot = [], 0.0
        fin_params = self.final.params()
        for j, x in enumerate(bounds[-1]):
            s0 = j * mb
            xl = self.to_dev(x).requires_grad_(True)
            b, H = xl.shape[0], xl.shape[2]
            hw = max(1, self.head_windows)
            kl = 0.0
            for i0 in range(0, b, hw):
                i1 = min(b, i0 + hw)
                n = i1 - i0
                xn = rms(xl[i0:i1, :T - 1].reshape(n * (T - 1), H), self.final.eff(), self.arch.eps)
                kl = kl + _HeadKL.apply(xn, self.head, lp_t[s0 + i0:s0 + i1].reshape(n * (T - 1), -1).to(self.dev),
                                        ix_t[s0 + i0:s0 + i1].reshape(n * (T - 1), -1).to(self.dev).long(),
                                        self.vchunk)
            loss = kl / count
            loss.backward()
            kl_tot += float(kl.detach())
            grads.append(self.to_act(xl.grad))
            del xl
        for l in range(len(self.layers) - 1, -1, -1):
            self._enter(l, l - 1)
            layer = self.layers[l]
            for j in range(len(xs)):
                x = self.to_dev(bounds[l][j]).requires_grad_(l > 0)
                y = layer.forward(x)
                y.backward(self.to_dev(grads[j]))
                grads[j] = self.to_act(x.grad) if l > 0 else None
                del x, y
            self._leave(l)
            bounds[l + 1] = None
        return kl_tot
