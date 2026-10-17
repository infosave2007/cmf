"""Unit tests of the offline quantization tools (CPU; the CUDA variants run when a GPU is
present). Run: python -m unittest discover -s tools/quant/tests -v"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import calib  # noqa: E402
import cmfcodec as cc  # noqa: E402
import cmfio  # noqa: E402
import qmodel  # noqa: E402

torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------------- fixtures
def rand_tiled(rng, dtype, rows, cols):
    gpr = cols // cc.G
    if dtype == "Q4TiledP":
        q = rng.integers(-8, 8, size=(rows, cols))
    else:
        q = rng.integers(0, 4, size=(rows, cols))
    lo = cc.f16_bits(rng.uniform(-9.0, -6.0, size=rows))
    st = cc.f16_bits(rng.uniform(0.02, 0.08, size=rows))
    code = rng.integers(0, 32, size=(rows, gpr)).astype(np.uint8)
    return cc.encode_tiled(dtype, q, lo, st, code)


def rand_q8(rng, rows, cols):
    q = rng.integers(-127, 128, size=(rows, cols)).astype(np.int8)
    rs = cc.f16_bits(rng.uniform(0.5, 1.5, size=rows) / 127)
    cs = cc.f16_bits(rng.uniform(0.02, 0.06, size=cols))
    return q.tobytes() + rs.astype("<u2").tobytes() + cs.astype("<u2").tobytes()


class FakeCmf:
    """In-memory stand-in for cmfio.CmfFile (same accessors)."""

    def __init__(self, arch):
        self.arch, self.tensors, self._p, self.path = arch, {}, {}, "<fake>"

    def add(self, name, dtype, shape, payload):
        payload = np.frombuffer(bytes(payload), dtype=np.uint8)
        if cc.payload_bytes(dtype, shape) not in (None, payload.size):
            raise AssertionError((name, dtype, shape, payload.size))
        self.tensors[name] = cmfio.Tensor(name, dtype, shape, 0, payload.size, 0, 0)
        self._p[name] = payload

    def __contains__(self, n):
        return n in self.tensors

    def tensor(self, n):
        return self.tensors[n]

    def payload(self, n):
        return self._p[n]


def tiny_cmf(kind="llama", seed=0, lin_dtype="Q4TiledP", head_dtype="Q8_2f"):
    rng = np.random.default_rng(seed)
    H, I, V, nh, nkv, hd = 64, 128, 300, 4, 2, 16
    kinds = ["FullAttention", "FullAttention"] if kind == "llama" else \
        ["LinearAttention", "LinearAttention", "LinearAttention", "FullAttention"]
    arch = {"arch_name": "llama" if kind == "llama" else "qwen3_5_text", "hidden_size": H, "intermediate_size": I,
            "num_layers": len(kinds), "num_attention_heads": nh, "num_kv_heads": nkv, "head_dim": hd,
            "vocab_size": V, "layer_types": kinds, "rms_norm_eps": 1e-6,
            "norm_style": "qwen" if kind == "llama" else "gemma", "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0 if kind == "llama" else 0.25, "tie_word_embeddings": False}
    if kind != "llama":
        arch.update(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                    linear_value_head_dim=16, linear_conv_kernel_dim=4, linear_core={"kind": "gated_delta_net"})
    m = FakeCmf(arch)

    def lin(name, rows, cols, dtype=lin_dtype):
        if dtype in qmodel.TILED:
            m.add(name, dtype, (rows, cols), rand_tiled(rng, dtype, rows, cols))
        elif dtype == "Q8_2f":
            m.add(name, dtype, (rows, cols), rand_q8(rng, rows, cols))
        else:
            m.add(name, "F16", (rows, cols), cc.encode_float("F16", rng.normal(0, 0.05, (rows, cols))))

    def norm(name, n, center):
        m.add(name, "F16", (n,), cc.encode_float("F16", center + rng.normal(0, 0.05, n)))

    off = 0.0 if kind == "llama" else 1.0
    for l, t in enumerate(kinds):
        p = f"model.layers.{l}."
        norm(p + "input_layernorm.weight", H, 1.0 - off)
        norm(p + "post_attention_layernorm.weight", H, 1.0 - off)
        if t == "FullAttention":
            lin(p + "self_attn.q_proj.weight", nh * hd * (1 if kind == "llama" else 2), H)
            lin(p + "self_attn.k_proj.weight", nkv * hd, H)
            lin(p + "self_attn.v_proj.weight", nkv * hd, H)
            lin(p + "self_attn.o_proj.weight", H, nh * hd)
            if kind != "llama":
                norm(p + "self_attn.q_norm.weight", hd, 0.0)
                norm(p + "self_attn.k_norm.weight", hd, 0.0)
        else:
            lin(p + "linear_attn.in_proj_qkv.weight", 2 * 2 * 16 + 4 * 16, H)
            lin(p + "linear_attn.in_proj_z.weight", 4 * 16, H)
            lin(p + "linear_attn.in_proj_b.weight", 4, H, "F16")
            lin(p + "linear_attn.in_proj_a.weight", 4, H, "F16")
            lin(p + "linear_attn.out_proj.weight", H, 4 * 16)
            m.add(p + "linear_attn.conv1d.weight", "F16", (128, 1, 4), cc.encode_float("F16", rng.normal(0, 0.3, (128, 1, 4))))
            m.add(p + "linear_attn.A_log", "F32", (4,), cc.encode_float("F32", np.log(rng.uniform(1, 16, 4))))
            m.add(p + "linear_attn.dt_bias", "F32", (4,), cc.encode_float("F32", rng.normal(0, 0.5, 4)))
            norm(p + "linear_attn.norm.weight", 16, 1.0)
        lin(p + "mlp.gate_proj.weight", I, H)
        lin(p + "mlp.up_proj.weight", I, H)
        lin(p + "mlp.down_proj.weight", H, I)
    norm("model.norm.weight", H, 1.0 - off)
    lin("model.embed_tokens.weight", V, H, head_dtype)
    lin("lm_head.weight", V, H, head_dtype)
    return m


def perturbed(m, seed=99):
    """a copy of a fake model whose 1-D gains are scaled by 1 + N(0, 0.1): a teacher the
    student can move towards"""
    rng = np.random.default_rng(seed)
    t = FakeCmf(m.arch)
    for n, ten in m.tensors.items():
        p = m.payload(n)
        if len(ten.shape) == 1 and ten.dtype == "F16" and n.endswith("norm.weight"):
            v = cc.decode_float("F16", p, ten.shape)
            p = cc.encode_float("F16", v * (1 + rng.normal(0, 0.1, v.shape)) + rng.normal(0, 0.05, v.shape))
        t.add(n, ten.dtype, ten.shape, bytes(p))
    return t


def teacher_of(model, windows, k):
    r = model.evaluate(windows, mb=2, topk=k)
    return torch.from_numpy(r["lp"]), torch.from_numpy(r["ix"])


def trained_values(model):
    return [p.detach().cpu().clone() for p in model.params()]


# ----------------------------------------------------------------------------- codecs
class CodecTest(unittest.TestCase):
    def test_tiled_round_trip_and_field_splice(self):
        rng = np.random.default_rng(1)
        for dt in ("Q4TiledP", "Q2TiledP"):
            raw = rand_tiled(rng, dt, 6, 96)
            t = cc.Tiled(dt, np.frombuffer(raw, np.uint8), 6, 96)
            q = t.levels() + (0 if dt == "Q4TiledP" else 1.5)
            self.assertEqual(cc.encode_tiled(dt, q.astype(np.int64), t.lo, t.st, t.code), raw)
            # splicing new row params changes exactly the 4*rows param bytes
            lo2 = t.lo ^ np.uint16(1)
            new = np.frombuffer(t.with_row_params(lo2, t.st), np.uint8)
            diff = np.nonzero(new != np.frombuffer(raw, np.uint8))[0]
            self.assertTrue(len(diff) > 0 and diff.min() >= t.nb and diff.max() < t.nb + 6 * 4)
            self.assertEqual(t.with_row_params(t.lo, t.st), raw)

    def test_ladder_is_sequential_f32(self):
        lo, st = cc.f16_bits(np.array([0.0, -3.0])), cc.f16_bits(np.array([1.0, 0.5]))
        lad = cc.ladder(lo, st)
        np.testing.assert_array_equal(lad[0], 2.0 ** np.arange(32, dtype=np.float32))
        ref = np.float32(2 ** -3)
        for c in range(32):
            self.assertEqual(lad[1, c], ref)
            ref = np.float32(ref * np.float32(np.exp2(np.float32(0.5))))
        self.assertTrue(np.all(cc.ladder2(lo, st)[:, 0] == 0))
        np.testing.assert_array_equal(cc.ladder2(lo, st)[:, 1:], lad[:, :31])

    def test_q8_row_scale_splice(self):
        rng = np.random.default_rng(2)
        raw = rand_q8(rng, 5, 64)
        q = cc.Q8_2f(np.frombuffer(raw, np.uint8), 5, 64)
        new = np.frombuffer(q.with_row_scales(q.rs_bits ^ np.uint16(1)), np.uint8)
        diff = np.nonzero(new != np.frombuffer(raw, np.uint8))[0]
        self.assertTrue(diff.min() >= 5 * 64 and diff.max() < 5 * 64 + 10)

    def test_float_round_trip(self):
        v = np.array([1.0, -2.5, 3e-3, 65000.0], np.float32)
        for dt in ("F32", "F16", "Bf16"):
            back = cc.decode_float(dt, np.frombuffer(cc.encode_float(dt, v), np.uint8), v.shape)
            np.testing.assert_allclose(back, v, rtol=1e-2)
            np.testing.assert_array_equal(cc.round_float(dt, back), back)


# ----------------------------------------------------------------------------- gradients
class GradTest(unittest.TestCase):
    def test_qmatmul_matches_autograd(self):
        rng = np.random.default_rng(3)
        for dt in ("Q4TiledP", "Q2TiledP", "Q8_2f"):
            rows, cols = 8, 64
            raw = rand_tiled(rng, dt, rows, cols) if dt in qmodel.TILED else rand_q8(rng, rows, cols)
            t = cmfio.Tensor("w", dt, (rows, cols), 0, len(raw), 0, 0)
            lin = qmodel.QLinear("w", t, np.frombuffer(raw, np.uint8), train=True, step=True, pin=False)
            lin.init_params("cpu")
            lin.load("cpu")
            with torch.no_grad():
                lin.a.copy_(torch.randn(rows) * 0.05)
                if lin.b is not None:
                    lin.b.copy_(torch.randn(rows) * 0.002)
            x = torch.randn(5, cols, requires_grad=True)
            y = lin(x)
            gy = torch.randn_like(y)
            y.backward(gy)
            g_custom = [x.grad.clone(), lin.a.grad.clone()] + ([lin.b.grad.clone()] if lin.b is not None else [])
            x.grad = None
            lin.a.grad = None
            if lin.b is not None:
                lin.b.grad = None
            W = lin.weight(lin.scale())          # same decode, plain autograd graph
            y2 = x @ W.t()
            torch.testing.assert_close(y, y2, rtol=1e-6, atol=1e-6)
            y2.backward(gy)
            g_ref = [x.grad, lin.a.grad] + ([lin.b.grad] if lin.b is not None else [])
            for a, b in zip(g_custom, g_ref):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    def test_device_decode_is_the_decoder_bit_for_bit(self):
        rng = np.random.default_rng(8)
        for dt in ("Q4TiledP", "Q2TiledP", "Q8_2f"):
            raw = rand_tiled(rng, dt, 40, 96) if dt in qmodel.TILED else rand_q8(rng, 40, 96)
            t = cmfio.Tensor("w", dt, (40, 96), 0, len(raw), 0, 0)
            lin = qmodel.QLinear("w", t, np.frombuffer(raw, np.uint8), pin=False)
            lin.load("cpu")
            ref = cc.dequant(dt, np.frombuffer(raw, np.uint8), (40, 96))
            np.testing.assert_array_equal(lin.weight().numpy(), ref)
            np.testing.assert_array_equal(lin.weight(None, 7, 23).numpy(), ref[7:23])

    def test_row_blocked_matmul_equals_one_block(self):
        rng = np.random.default_rng(6)
        rows, cols = 200, 64
        for dt in ("Q4TiledP", "Q2TiledP", "Q8_2f", "F16"):
            if dt in qmodel.TILED:
                raw = rand_tiled(rng, dt, rows, cols)
            elif dt == "Q8_2f":
                raw = rand_q8(rng, rows, cols)
            else:
                raw = cc.encode_float("F16", rng.normal(0, 0.05, (rows, cols)))
            t = cmfio.Tensor("w", dt, (rows, cols), 0, len(raw), 0, 0)
            lin = qmodel.QLinear("w", t, np.frombuffer(raw, np.uint8), train=dt != "F16", step=True, pin=False)
            lin.init_params("cpu")
            lin.load("cpu")
            if lin.a is not None:
                with torch.no_grad():
                    lin.a.copy_(torch.randn(rows) * 0.05)
            x0 = torch.randn(9, cols)
            gy = torch.randn(9, rows)
            outs = []
            for elems in (1 << 25, 64 * 70):           # one block / blocks of 70 rows (last one 60)
                old, qmodel.ROW_BLOCK_ELEMS = qmodel.ROW_BLOCK_ELEMS, elems
                try:
                    self.assertEqual(len(lin.row_blocks()), 1 if elems == 1 << 25 else 3)
                    x = x0.clone().requires_grad_(True)
                    y = lin(x)
                    y.backward(gy)
                    outs.append((y.detach(), x.grad, None if lin.a is None else lin.a.grad.clone()))
                    if lin.a is not None:
                        lin.a.grad = None
                        if lin.b is not None:
                            lin.b.grad = None
                finally:
                    qmodel.ROW_BLOCK_ELEMS = old
            for a, b in zip(outs[0], outs[1]):
                if a is not None:
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_head_kl_matches_dense_reference(self):
        torch.manual_seed(0)
        m = tiny_cmf("llama", seed=4, head_dtype="Q4TiledP")
        head = qmodel.Head(m, "lm_head.weight", 1.0, pin=False)
        head.load("cpu")
        P, K = 7, 5
        xn = torch.randn(P, 64, requires_grad=True)
        W = head.rows(0, head.vocab)
        with torch.no_grad():
            tl = torch.log_softmax(torch.randn(P, head.vocab) * 3, -1)
            lp_t, ix_t = tl.topk(K, -1)
        kl = qmodel._HeadKL.apply(xn, head, lp_t, ix_t, 64)        # 5 vocabulary chunks
        kl.backward()
        g = xn.grad.clone()
        xn.grad = None
        lq = torch.log_softmax(xn @ W.t(), -1).gather(1, ix_t)
        pt = lp_t.exp()
        tt = (1 - pt.sum(1)).clamp_min(1e-7)
        tq = (1 - lq.exp().sum(1)).clamp_min(1e-7)
        ref = (pt * (lp_t - lq)).sum() + (tt * (tt.log() - tq.log())).sum()
        ref.backward()
        torch.testing.assert_close(kl, ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g, xn.grad, rtol=1e-4, atol=1e-6)
        # eval scan: lse, target logits, top-k equal the dense ones
        tgt = torch.randint(0, head.vocab, (P,))
        lse, z_ix, z_t, tv, ti = head.scan(xn.detach(), 64, ix=ix_t, tgt=tgt, topk=K)
        z = xn.detach() @ W.t()
        torch.testing.assert_close(lse, torch.logsumexp(z, 1))
        torch.testing.assert_close(z_t, z.gather(1, tgt[:, None])[:, 0])
        torch.testing.assert_close(tv, z.topk(K, 1).values)

    def test_gated_delta_rule_matches_recurrence(self):
        torch.manual_seed(5)
        B, T, h, dk, dv = 2, 100, 3, 8, 6
        q, k = torch.randn(B, T, h, dk, dtype=torch.float64), torch.randn(B, T, h, dk, dtype=torch.float64)
        v = torch.randn(B, T, h, dv, dtype=torch.float64)
        g = -torch.rand(B, T, h, dtype=torch.float64) * 0.5
        beta = torch.rand(B, T, h, dtype=torch.float64)
        out = qmodel.chunk_gated_delta_rule(q.float(), k.float(), v.float(), g.float(), beta.float(), chunk=16)
        qn, kn = qmodel.l2norm(q), qmodel.l2norm(k)
        S = torch.zeros(B, h, dk, dv, dtype=torch.float64)
        ref = torch.zeros(B, T, h, dv, dtype=torch.float64)
        for t in range(T):
            S = S * g[:, t].exp()[..., None, None]
            kv = (S * kn[:, t, :, :, None]).sum(-2)
            S = S + kn[:, t, :, :, None] * ((v[:, t] - kv) * beta[:, t, :, None])[:, :, None, :]
            ref[:, t] = (S * (qn[:, t] / dk ** 0.5)[..., None]).sum(-2)
        torch.testing.assert_close(out.double(), ref, rtol=2e-4, atol=2e-5)


# ----------------------------------------------------------------------------- model / training
class ModelTest(unittest.TestCase):
    def run_steps(self, m, residency, dev, steps=4, act=None):
        qmodel.setup_torch(0, tf32=False)
        teacher_model = qmodel.Model(perturbed(m), dev, residency="resident")
        rng = np.random.default_rng(7)
        w = rng.integers(0, 300, size=(4, 40))
        lp, ix = teacher_of(teacher_model, w, 8)
        student = qmodel.Model(m, dev, residency=residency, act_device=act, train=True, step=True, train_norms=True)
        opt = torch.optim.Adam(student.params(), lr=1e-3)
        kls = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            kls.append(student.train_step(w, lp, ix, 2, w.shape[0] * 39))
            opt.step()
        ev = student.evaluate(w, mb=2, teacher=(lp.numpy(), ix.numpy()))
        return kls, trained_values(student), ev

    def check_modes(self, kind, dev):
        m = tiny_cmf(kind, seed=11)
        k1, v1, e1 = self.run_steps(m, "resident", dev)
        k2, v2, e2 = self.run_steps(m, "stream", dev, act="cpu")
        self.assertEqual(k1, k2)
        for a, b in zip(v1, v2):
            self.assertTrue(torch.equal(a, b))
        np.testing.assert_array_equal(e1["nll"], e2["nll"])
        self.assertTrue(np.all(np.isfinite(e1["nll"])))
        # trained values move and the KL to the teacher falls on the training windows
        self.assertTrue(any(float(x.abs().max()) > 0 for x in v1))
        self.assertLess(k1[-1], k1[0])
        return k1

    def test_llama_stream_equals_resident_cpu(self):
        self.check_modes("llama", "cpu")

    def test_qwen35_stream_equals_resident_cpu(self):
        self.check_modes("qwen35", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA")
    def test_stream_equals_resident_cuda(self):
        for kind in ("llama", "qwen35"):
            self.check_modes(kind, "cuda")

    def test_teacher_of_itself_has_zero_kl(self):
        m = tiny_cmf("qwen35", seed=12)
        model = qmodel.Model(m, "cpu")
        w = np.random.default_rng(1).integers(0, 300, size=(2, 33))
        lp, ix = teacher_of(model, w, 300)          # full vocabulary: no tail
        ev = model.evaluate(w, mb=1, teacher=(lp.numpy(), ix.numpy()))
        self.assertLess(float(np.abs(ev["kl"]).max()), 1e-3)
        # the CE of the scan equals a dense log-softmax CE
        self.assertTrue(np.all(ev["nll"] > 0))

    def test_embedding_rows_match_full_decode(self):
        for dt in ("F16", "Q8_2f", "Q4TiledP"):
            m = tiny_cmf("llama", seed=14, head_dtype=dt)
            e = qmodel.Embed(m, "model.embed_tokens.weight", 1.0)
            ids = np.array([5, 0, 299, 5, 17])
            full = cc.dequant(dt, m.payload("model.embed_tokens.weight"), (300, 64))
            np.testing.assert_array_equal(e.rows(ids), full[ids])

    def test_snapped_write_back_touches_only_fields(self):
        m = tiny_cmf("llama", seed=13)
        model = qmodel.Model(m, "cpu", train=True, step=True, train_norms=True)
        lin = next(l for l in model.linears() if l.dtype == "Q4TiledP")
        payload, a_eff, b_eff = lin.snapped(np.zeros(lin.rows), np.zeros(lin.rows))
        self.assertEqual(payload, bytes(lin.payload))
        self.assertTrue(np.all(a_eff == 0) and np.all(b_eff == 0))
        payload, a_eff, _ = lin.snapped(np.full(lin.rows, 0.01), None)
        diff = np.nonzero(np.frombuffer(payload, np.uint8) != lin.payload)[0]
        self.assertTrue(diff.min() >= lin.tiled.nb and diff.max() < lin.tiled.nb + 4 * lin.rows)
        self.assertTrue(np.all(np.abs(a_eff - 0.01) < 0.01))
        # zero-centred gain: stored w' = (1 + w) 2^n - 1
        g = qmodel.Gain("g", cmfio.Tensor("g", "F16", (4,), 0, 8, 0, 0),
                        np.frombuffer(cc.encode_float("F16", np.array([0.1, -0.2, 0.0, 0.5])), np.uint8), 1.0)
        payload, n_eff = g.snapped(np.full(4, 0.5))
        stored = cc.decode_float("F16", np.frombuffer(payload, np.uint8), (4,))
        np.testing.assert_allclose(stored + 1, (g.g0 + 1) * 2 ** 0.5, rtol=2e-3)
        np.testing.assert_allclose(n_eff, 0.5, atol=3e-3)


class CalibTest(unittest.TestCase):
    def test_study_window_placement(self):
        """the q2tp/q4tp studies' e2e_metric.py placement, verbatim"""
        WL, n, nwin = 512, 2400483, 1024
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.npy")
            np.save(p, np.arange(n, dtype=np.int32))
            c = calib.place([calib.parse_corpus(f"{p},train={nwin},held=65536:8+200000:16")], WL, 0)[0]
        busy = []
        st_ = (n - 65536 - WL - 1) // 8
        busy += [(65536 + k * st_, 65536 + k * st_ + WL) for k in range(8)]
        st_ = (n - 200000 - WL - 1) // 16
        busy += [(200000 + k * st_, 200000 + k * st_ + WL) for k in range(16)]
        stp = (n - WL - 1 - 4096) // nwin
        starts = []
        for j in range(nwin):
            q = j * stp
            while any(q < b1 and q + WL > b0 for b0, b1 in busy):
                q += 128
            starts.append(q)
        self.assertEqual(c.train_starts, starts)
        self.assertEqual(c.held_starts, [b0 for b0, _ in busy])

    def test_mixed_corpora_split(self):
        with tempfile.TemporaryDirectory() as d:
            ps = []
            for i, n in enumerate((50000, 80000)):
                ps.append(os.path.join(d, f"c{i}.npy"))
                np.save(ps[-1], np.arange(n, dtype=np.int32))
            cs = calib.place([calib.parse_corpus(ps[0]), calib.parse_corpus(ps[1] + ",held=1000:4")], 128, 21)
            tr, he, owner = calib.build_sets(cs, 128)
        self.assertEqual([len(c.train_starts) for c in cs], [11, 10])
        self.assertEqual(len(he), 12 + 4)
        self.assertEqual(list(np.bincount(owner)), [12, 4])
        for c in cs:
            for s in c.train_starts:
                self.assertFalse(any(s < h + 128 and s + 128 > h for h in c.held_starts))


if __name__ == "__main__":
    unittest.main()
