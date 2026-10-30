//! Nemotron-H language model (the decoder of Nemotron 3.5 Transcribe):
//! a stack of blocks, each ONE mixer behind its own RMSNorm,
//! `h = h + mixer(rms(h))`, the mixer being
//!
//! * Mamba-2: `in_proj → [z | xBC | dt]`, depthwise causal conv over xBC +
//!   SiLU, the selective scan `S = exp(dt·A)·S + dt·x⊗B`, `y = S·C + D·x`,
//!   the gated group RMSNorm `rms_g(y·silu(z))` and `out_proj`;
//! * attention: GQA softmax attention with no positional encoding;
//! * MoE: sigmoid router with a selection-only bias, top-k renormalized and
//!   scaled, non-gated `down(relu(up·x)²)` experts plus a shared expert.
//!
//! Every projection is a `QTensor`, so q8_2f / q4tp files run on the CMF
//! CPU kernels and on the Metal / Vulkan backends without a second weight
//! copy. Activations are f32.

use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{
    CmfModel, TensorDtype,
    quant::{bf16_to_f32, f16_to_f32},
};
use std::sync::Arc;

pub(crate) fn tensor_f32(m: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let e = m
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let b = m.entry_bytes(e);
    let n = e.n_elems();
    match e.dtype {
        TensorDtype::F32 if b.len() == n * 4 => Ok(b
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()),
        TensorDtype::F16 if b.len() == n * 2 => Ok(b
            .chunks_exact(2)
            .map(|x| f16_to_f32(u16::from_le_bytes([x[0], x[1]])))
            .collect()),
        TensorDtype::Bf16 if b.len() == n * 2 => Ok(b
            .chunks_exact(2)
            .map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]])))
            .collect()),
        _ => Err(format!(
            "{name}: expected a floating point tensor, got {:?}",
            e.dtype
        )),
    }
}

/// Every value of a 2-D tensor of any codec, dequantized (offline tools).
pub fn tensor_values(m: &Arc<CmfModel>, name: &str) -> Result<Vec<f32>, String> {
    let q = QTensor::from_model(m, name)?;
    let (rows, cols) = (q.rows(), q.cols());
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        q.row_f32(r, &mut out[r * cols..(r + 1) * cols]);
    }
    Ok(out)
}

/// Geometry of the stack, from the source `llm_config`.
#[derive(Clone, Debug)]
pub struct Geom {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub m_heads: usize,
    pub m_head_dim: usize,
    pub m_groups: usize,
    pub m_state: usize,
    pub conv_k: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub n_exp: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub route_scale: f32,
    pub norm_topk: bool,
}

impl Geom {
    pub fn from_config(c: &serde_json::Value) -> Result<Self, String> {
        let u = |k: &str| {
            c.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("llm config: missing {k}"))
        };
        Ok(Self {
            hidden: u("hidden_size")?,
            vocab: u("vocab_size")?,
            eps: c
                .get("layer_norm_epsilon")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5) as f32,
            m_heads: u("mamba_num_heads")?,
            m_head_dim: u("mamba_head_dim")?,
            m_groups: u("n_groups")?,
            m_state: u("ssm_state_size")?,
            conv_k: u("conv_kernel")?,
            nh: u("num_attention_heads")?,
            nkv: u("num_key_value_heads")?,
            hd: u("head_dim")?,
            n_exp: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            moe_inter: u("moe_intermediate_size")?,
            shared_inter: u("moe_shared_expert_intermediate_size")?,
            route_scale: c
                .get("routed_scaling_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32,
            norm_topk: c
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        })
    }
    pub fn m_inner(&self) -> usize {
        self.m_heads * self.m_head_dim
    }
    pub fn conv_dim(&self) -> usize {
        self.m_inner() + 2 * self.m_groups * self.m_state
    }
    pub fn in_proj_rows(&self) -> usize {
        self.m_inner() + self.conv_dim() + self.m_heads
    }
}

pub struct MambaW {
    pub in_proj: QTensor,
    pub out_proj: QTensor,
    /// `[conv_dim · kernel]`, tap `kernel − 1` multiplies the current token
    pub conv_w: Vec<f32>,
    pub conv_b: Vec<f32>,
    /// `−exp(A_log)`
    pub a: Vec<f32>,
    pub d: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
}

pub struct AttnW {
    pub q: QTensor,
    pub k: QTensor,
    pub v: QTensor,
    pub o: QTensor,
}

pub struct MoeW {
    /// `[n_exp, hidden]` f32
    pub router: QTensor,
    pub bias: Vec<f32>,
    pub up: Vec<QTensor>,
    pub down: Vec<QTensor>,
    pub sh_up: QTensor,
    pub sh_down: QTensor,
}

/// The multi-token-prediction module: `mtp.layers.0` (the fusion of the
/// next token's embedding with a hidden state, then an attention block) and
/// `mtp.layers.1` (a MoE block and the output norm); one set of weights for
/// every draft depth, the language model's head on top.
pub struct Mtp {
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    /// `[hidden, 2·hidden]` over `[enorm(embed) | hnorm(hidden)]`
    pub eh: QTensor,
    pub n_attn: Vec<f32>,
    pub attn: AttnW,
    pub n_moe: Vec<f32>,
    pub moe: MoeW,
    pub fnorm: Vec<f32>,
}

/// MoE inputs captured for GPTQ calibration, per block index (empty for
/// non-MoE blocks): the normed rows `[n · hidden]` and each row's `top_k`
/// routed experts.
#[derive(Default)]
pub struct MoeCapture {
    pub x: Vec<Vec<f32>>,
    pub picks: Vec<Vec<u16>>,
}

pub enum Mixer {
    Mamba(MambaW),
    Attn(AttnW),
    Moe(MoeW),
}

pub struct Block {
    pub norm: Vec<f32>,
    pub mixer: Mixer,
    /// index into the per-kind state vectors (Mamba ring/scan, KV)
    pub slot: usize,
}

pub struct NemotronH {
    pub g: Geom,
    pub embed: QTensor,
    pub lm_head: QTensor,
    pub final_norm: Vec<f32>,
    pub blocks: Vec<Block>,
    pub n_mamba: usize,
    pub n_attn: usize,
    model: Option<Arc<CmfModel>>,
    /// Per MoE block, its experts most used first (a partial device bank is
    /// filled in this order); empty when the file does not say.
    pub hot: Vec<Vec<usize>>,
    /// The draft module, when the file carries it.
    pub mtp: Option<Mtp>,
    /// GPTQ calibration: per block, the MoE input rows and routed experts.
    capture: std::sync::Mutex<Option<MoeCapture>>,
    /// The device decode frame (wgpu), built on first use; `Err` once refused.
    #[cfg(feature = "gpu")]
    frame: std::sync::Mutex<Option<Result<crate::gpu_wgpu::nemotron::NemoFrame, ()>>>,
}

/// Decoding state: Mamba conv rings + scan states, attention KV.
pub struct State {
    /// `[(k−1) · conv_dim]` per Mamba block, oldest tap first
    pub conv: Vec<Vec<f32>>,
    /// `[heads · head_dim · state]` per Mamba block
    pub ssm: Vec<Vec<f32>>,
    /// `[pos · nkv · hd]` per attention block
    pub kc: Vec<Vec<f32>>,
    pub vc: Vec<Vec<f32>>,
    pub pos: usize,
}

impl NemotronH {
    pub fn load(m: &Arc<CmfModel>, cfg: &serde_json::Value) -> Result<Self, String> {
        let g = Geom::from_config(cfg)?;
        let types: Vec<String> = cfg
            .get("layers_block_type")
            .and_then(|v| v.as_array())
            .ok_or("llm config: missing layers_block_type")?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let q = |n: &str| QTensor::from_model(m, n);
        let mut blocks = Vec::with_capacity(types.len());
        let (mut n_mamba, mut n_attn) = (0, 0);
        for (i, t) in types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let norm = tensor_f32(m, &format!("{p}.norm.weight"))?;
            let (mixer, slot) = match t.as_str() {
                "mamba" => {
                    let a_log = tensor_f32(m, &format!("{p}.mixer.A_log"))?;
                    let w = MambaW {
                        in_proj: q(&format!("{p}.mixer.in_proj.weight"))?,
                        out_proj: q(&format!("{p}.mixer.out_proj.weight"))?,
                        conv_w: tensor_f32(m, &format!("{p}.mixer.conv1d.weight"))?,
                        conv_b: tensor_f32(m, &format!("{p}.mixer.conv1d.bias"))?,
                        a: a_log.iter().map(|v| -v.exp()).collect(),
                        d: tensor_f32(m, &format!("{p}.mixer.D"))?,
                        dt_bias: tensor_f32(m, &format!("{p}.mixer.dt_bias"))?,
                        norm: tensor_f32(m, &format!("{p}.mixer.norm.weight"))?,
                    };
                    if w.in_proj.rows() != g.in_proj_rows()
                        || w.conv_w.len() != g.conv_dim() * g.conv_k
                        || w.a.len() != g.m_heads
                    {
                        return Err(format!("{p}: Mamba-2 shapes do not match the config"));
                    }
                    n_mamba += 1;
                    (Mixer::Mamba(w), n_mamba - 1)
                }
                "attention" => {
                    n_attn += 1;
                    (
                        Mixer::Attn(AttnW {
                            q: q(&format!("{p}.mixer.q_proj.weight"))?,
                            k: q(&format!("{p}.mixer.k_proj.weight"))?,
                            v: q(&format!("{p}.mixer.v_proj.weight"))?,
                            o: q(&format!("{p}.mixer.o_proj.weight"))?,
                        }),
                        n_attn - 1,
                    )
                }
                "moe" => {
                    let router = tensor_f32(m, &format!("{p}.mlp.gate.weight"))?;
                    let mut up = Vec::with_capacity(g.n_exp);
                    let mut down = Vec::with_capacity(g.n_exp);
                    for e in 0..g.n_exp {
                        up.push(q(&format!("{p}.mlp.experts.{e}.up_proj.weight"))?);
                        down.push(q(&format!("{p}.mlp.experts.{e}.down_proj.weight"))?);
                    }
                    (
                        Mixer::Moe(MoeW {
                            router: QTensor::from_f32(router, g.n_exp, g.hidden),
                            bias: tensor_f32(m, &format!("{p}.mlp.gate.e_score_correction_bias"))?,
                            up,
                            down,
                            sh_up: q(&format!("{p}.mlp.shared_experts.up_proj.weight"))?,
                            sh_down: q(&format!("{p}.mlp.shared_experts.down_proj.weight"))?,
                        }),
                        0,
                    )
                }
                other => return Err(format!("unsupported Nemotron-H block type '{other}'")),
            };
            blocks.push(Block { norm, mixer, slot });
        }
        let mtp = Self::load_mtp(m, &g)?;
        Ok(Self {
            embed: q("model.embed_tokens.weight")?,
            lm_head: q("lm_head.weight")?,
            final_norm: tensor_f32(m, "model.norm.weight")?,
            g,
            blocks,
            n_mamba,
            n_attn,
            mtp,
            model: Some(m.clone()),
            hot: Vec::new(),
            capture: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            frame: std::sync::Mutex::new(None),
        })
    }

    fn load_mtp(m: &Arc<CmfModel>, g: &Geom) -> Result<Option<Mtp>, String> {
        let p0 = "mtp.layers.0";
        let p1 = "mtp.layers.1";
        if m.tensor(&format!("{p0}.eh_proj.weight")).is_none() {
            return Ok(None);
        }
        let q = |n: &str| QTensor::from_model(m, n);
        let mut up = Vec::with_capacity(g.n_exp);
        let mut down = Vec::with_capacity(g.n_exp);
        for e in 0..g.n_exp {
            up.push(q(&format!("{p1}.mlp.experts.{e}.up_proj.weight"))?);
            down.push(q(&format!("{p1}.mlp.experts.{e}.down_proj.weight"))?);
        }
        Ok(Some(Mtp {
            enorm: tensor_f32(m, &format!("{p0}.enorm.weight"))?,
            hnorm: tensor_f32(m, &format!("{p0}.hnorm.weight"))?,
            eh: q(&format!("{p0}.eh_proj.weight"))?,
            n_attn: tensor_f32(m, &format!("{p0}.norm.weight"))?,
            attn: AttnW {
                q: q(&format!("{p0}.mixer.q_proj.weight"))?,
                k: q(&format!("{p0}.mixer.k_proj.weight"))?,
                v: q(&format!("{p0}.mixer.v_proj.weight"))?,
                o: q(&format!("{p0}.mixer.o_proj.weight"))?,
            },
            n_moe: tensor_f32(m, &format!("{p1}.norm.weight"))?,
            moe: MoeW {
                router: QTensor::from_f32(tensor_f32(m, &format!("{p1}.mlp.gate.weight"))?, g.n_exp, g.hidden),
                bias: tensor_f32(m, &format!("{p1}.mlp.gate.e_score_correction_bias"))?,
                up,
                down,
                sh_up: q(&format!("{p1}.mlp.shared_experts.up_proj.weight"))?,
                sh_down: q(&format!("{p1}.mlp.shared_experts.down_proj.weight"))?,
            },
            fnorm: tensor_f32(m, &format!("{p1}.final_layernorm.weight"))?,
        }))
    }

    /// The residual stream after the last block for every row of `rows`
    /// (before the final norm), advancing `st`.
    pub fn forward_h(&self, st: &mut State, rows: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        let hd = self.g.hidden;
        let n = rows.len() / hd;
        let mut h = rows.to_vec();
        let mut xn = vec![0.0f32; n * hd];
        for (bi, b) in self.blocks.iter().enumerate() {
            for (src, dst) in h.chunks_exact(hd).zip(xn.chunks_exact_mut(hd)) {
                rms_norm(src, &b.norm, self.g.eps, dst);
            }
            let out = match &b.mixer {
                Mixer::Mamba(w) => {
                    self.mamba(w, &mut st.conv[b.slot], &mut st.ssm[b.slot], &xn, n, pool)
                }
                Mixer::Attn(w) => self.attention(
                    w,
                    &mut st.kc[b.slot],
                    &mut st.vc[b.slot],
                    st.pos,
                    &xn,
                    n,
                    pool,
                ),
                Mixer::Moe(w) => self.moe(bi, w, &xn, n, pool),
            };
            for (a, o) in h.iter_mut().zip(&out) {
                *a += o;
            }
        }
        st.pos += n;
        h
    }

    /// One draft depth over `n` positions from `pos`: `hidden` (the state
    /// the depth reads, `[n · hidden]`) fused with `next` (the input rows of
    /// the following positions: token embeddings, or the audio embeddings
    /// inside the prompt); the depth's KV rows extend `kv`. Returns the
    /// depth's output states `[n · hidden]` (after its output norm).
    pub fn mtp_forward(
        &self,
        kv: &mut (Vec<f32>, Vec<f32>),
        pos: usize,
        hidden: &[f32],
        next: &[f32],
        h_first: bool,
        pool: Option<&Pool>,
    ) -> Option<Vec<f32>> {
        let w = self.mtp.as_ref()?;
        let hd = self.g.hidden;
        let n = next.len() / hd;
        let mut cat = vec![0.0f32; n * 2 * hd];
        let mut tmp = vec![0.0f32; hd];
        for t in 0..n {
            let (a, b) = cat[t * 2 * hd..(t + 1) * 2 * hd].split_at_mut(hd);
            let (de, dh) = if h_first { (b, a) } else { (a, b) };
            rms_norm(&next[t * hd..(t + 1) * hd], &w.enorm, self.g.eps, de);
            tmp.copy_from_slice(&hidden[t * hd..(t + 1) * hd]);
            rms_norm(&tmp, &w.hnorm, self.g.eps, dh);
        }
        let mut x = vec![0.0f32; n * hd];
        project(&w.eh, &cat, n, &mut x, pool);
        let mut xn = vec![0.0f32; n * hd];
        for (src, dst) in x.chunks_exact(hd).zip(xn.chunks_exact_mut(hd)) {
            rms_norm(src, &w.n_attn, self.g.eps, dst);
        }
        let a = self.attention(&w.attn, &mut kv.0, &mut kv.1, pos, &xn, n, pool);
        for (v, o) in x.iter_mut().zip(&a) {
            *v += o;
        }
        for (src, dst) in x.chunks_exact(hd).zip(xn.chunks_exact_mut(hd)) {
            rms_norm(src, &w.n_moe, self.g.eps, dst);
        }
        let m = self.moe(usize::MAX, &w.moe, &xn, n, pool);
        for (v, o) in x.iter_mut().zip(&m) {
            *v += o;
        }
        let mut out = vec![0.0f32; n * hd];
        for (src, dst) in x.chunks_exact(hd).zip(out.chunks_exact_mut(hd)) {
            rms_norm(src, &w.fnorm, self.g.eps, dst);
        }
        Some(out)
    }

    /// Argmax of the head over `n` normed rows.
    pub fn head_argmax(&self, xn: &[f32], pool: Option<&Pool>) -> Vec<u32> {
        let hd = self.g.hidden;
        let n = xn.len() / hd;
        let v = self.lm_head.rows();
        let mut out = Vec::with_capacity(n);
        let mut logits = vec![0.0f32; v];
        for t in 0..n {
            self.lm_head.matvec(&xn[t * hd..(t + 1) * hd], &mut logits, pool);
            let mut best = 0usize;
            for (i, &l) in logits.iter().enumerate() {
                if l > logits[best] {
                    best = i;
                }
            }
            out.push(best as u32);
        }
        out
    }

    pub fn new_state(&self) -> State {
        let g = &self.g;
        State {
            conv: vec![vec![0.0; (g.conv_k - 1) * g.conv_dim()]; self.n_mamba],
            ssm: vec![vec![0.0; g.m_heads * g.m_head_dim * g.m_state]; self.n_mamba],
            kc: vec![Vec::new(); self.n_attn],
            vc: vec![Vec::new(); self.n_attn],
            pos: 0,
        }
    }

    /// Start recording every MoE block's input rows and routes.
    pub fn start_capture(&self) {
        let n = self.blocks.len();
        *self.capture.lock().unwrap() = Some(MoeCapture {
            x: vec![Vec::new(); n],
            picks: vec![Vec::new(); n],
        });
    }

    pub fn take_capture(&self) -> Option<MoeCapture> {
        self.capture.lock().unwrap().take()
    }

    pub fn embed_row(&self, id: u32, dst: &mut [f32]) {
        self.embed.row_f32(id as usize, dst);
    }

    /// Run `rows` (`[n · hidden]` input embeddings) through the stack,
    /// advancing `st`; returns the last position's logits.
    pub fn forward(&self, st: &mut State, rows: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        let hd = self.g.hidden;
        let n = rows.len() / hd;
        assert!(n > 0 && rows.len() == n * hd);
        let mut h = rows.to_vec();
        let mut xn = vec![0.0f32; n * hd];
        for (bi, b) in self.blocks.iter().enumerate() {
            for (src, dst) in h.chunks_exact(hd).zip(xn.chunks_exact_mut(hd)) {
                rms_norm(src, &b.norm, self.g.eps, dst);
            }
            let out = match &b.mixer {
                Mixer::Mamba(w) => {
                    self.mamba(w, &mut st.conv[b.slot], &mut st.ssm[b.slot], &xn, n, pool)
                }
                Mixer::Attn(w) => self.attention(
                    w,
                    &mut st.kc[b.slot],
                    &mut st.vc[b.slot],
                    st.pos,
                    &xn,
                    n,
                    pool,
                ),
                Mixer::Moe(w) => self.moe(bi, w, &xn, n, pool),
            };
            for (a, o) in h.iter_mut().zip(&out) {
                *a += o;
            }
        }
        st.pos += n;
        let last = &h[(n - 1) * hd..];
        let mut normed = vec![0.0; hd];
        rms_norm(last, &self.final_norm, self.g.eps, &mut normed);
        let mut logits = vec![0.0; self.lm_head.rows()];
        self.lm_head.matvec(&normed, &mut logits, pool);
        logits
    }

    fn mamba(
        &self,
        w: &MambaW,
        ring: &mut [f32],
        ssm: &mut [f32],
        xn: &[f32],
        n: usize,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let g = &self.g;
        let (inner, cdim, nh, hdim, ng, ns, kk) = (
            g.m_inner(),
            g.conv_dim(),
            g.m_heads,
            g.m_head_dim,
            g.m_groups,
            g.m_state,
            g.conv_k,
        );
        let pr = g.in_proj_rows();
        let mut zxbcdt = vec![0.0f32; n * pr];
        project(&w.in_proj, xn, n, &mut zxbcdt, pool);
        // depthwise causal conv + SiLU over xBC, channel-parallel
        let mut xbc = vec![0.0f32; n * cdim];
        {
            let xbc_addr = SendPtr(xbc.as_mut_ptr());
            let ring_addr = SendPtr(ring.as_mut_ptr());
            let zx = &zxbcdt;
            let run = |c0: usize, c1: usize| {
                let xbc = xbc_addr;
                let ring = ring_addr;
                let mut hist = vec![0.0f32; kk - 1 + n];
                for c in c0..c1 {
                    for j in 0..kk - 1 {
                        hist[j] = unsafe { *ring.0.add(j * cdim + c) };
                    }
                    for t in 0..n {
                        hist[kk - 1 + t] = zx[t * pr + inner + c];
                    }
                    let taps = &w.conv_w[c * kk..(c + 1) * kk];
                    for t in 0..n {
                        let mut acc = w.conv_b[c];
                        for j in 0..kk {
                            acc += taps[j] * hist[t + j];
                        }
                        unsafe { *xbc.0.add(t * cdim + c) = silu(acc) };
                    }
                    for j in 0..kk - 1 {
                        unsafe { *ring.0.add(j * cdim + c) = hist[n + j] };
                    }
                }
            };
            par_rows(pool, cdim, &run);
        }
        // selective scan, head-parallel
        let mut y = vec![0.0f32; n * inner];
        {
            let y_addr = SendPtr(y.as_mut_ptr());
            let s_addr = SendPtr(ssm.as_mut_ptr());
            let zx = &zxbcdt;
            let xbc = &xbc;
            let hpg = nh / ng;
            let run = |h0: usize, h1: usize| {
                let y = y_addr;
                let s_all = s_addr;
                for hh in h0..h1 {
                    let grp = hh / hpg;
                    let s = unsafe {
                        std::slice::from_raw_parts_mut(s_all.0.add(hh * hdim * ns), hdim * ns)
                    };
                    for t in 0..n {
                        let raw = zx[t * pr + inner + cdim + hh] + w.dt_bias[hh];
                        let dt = softplus(raw);
                        let da = (dt * w.a[hh]).exp();
                        let row = &xbc[t * cdim..(t + 1) * cdim];
                        let x = &row[hh * hdim..(hh + 1) * hdim];
                        let bv = &row[inner + grp * ns..inner + (grp + 1) * ns];
                        let cv = &row[inner + ng * ns + grp * ns..inner + ng * ns + (grp + 1) * ns];
                        for p in 0..hdim {
                            let sp = &mut s[p * ns..(p + 1) * ns];
                            let dx = dt * x[p];
                            let mut acc = 0.0f32;
                            for k in 0..ns {
                                let v = da * sp[k] + dx * bv[k];
                                sp[k] = v;
                                acc += v * cv[k];
                            }
                            unsafe { *y.0.add(t * inner + hh * hdim + p) = acc + w.d[hh] * x[p] };
                        }
                    }
                }
            };
            par_rows(pool, nh, &run);
        }
        // gate + group RMSNorm
        let gs = inner / ng;
        for t in 0..n {
            let z = &zxbcdt[t * pr..t * pr + inner];
            let yr = &mut y[t * inner..(t + 1) * inner];
            for (v, &zz) in yr.iter_mut().zip(z) {
                *v *= silu(zz);
            }
            for (gi, grp) in yr.chunks_exact_mut(gs).enumerate() {
                let ms = grp.iter().map(|v| v * v).sum::<f32>() / gs as f32;
                let inv = 1.0 / (ms + g.eps).sqrt();
                for (j, v) in grp.iter_mut().enumerate() {
                    *v = *v * inv * w.norm[gi * gs + j];
                }
            }
        }
        let mut out = vec![0.0f32; n * g.hidden];
        project(&w.out_proj, &y, n, &mut out, pool);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        w: &AttnW,
        kc: &mut Vec<f32>,
        vc: &mut Vec<f32>,
        pos: usize,
        xn: &[f32],
        n: usize,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let g = &self.g;
        let (nh, nkv, hd) = (g.nh, g.nkv, g.hd);
        let mut q = vec![0.0f32; n * nh * hd];
        let mut k = vec![0.0f32; n * nkv * hd];
        let mut v = vec![0.0f32; n * nkv * hd];
        project(&w.q, xn, n, &mut q, pool);
        project(&w.k, xn, n, &mut k, pool);
        project(&w.v, xn, n, &mut v, pool);
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let total = pos + n;
        debug_assert_eq!(kc.len(), total * nkv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut o = vec![0.0f32; n * nh * hd];
        {
            let o_addr = SendPtr(o.as_mut_ptr());
            let (kc, vc, q) = (&*kc, &*vc, &q);
            let rep = nh / nkv;
            let run = |r0: usize, r1: usize| {
                let o = o_addr;
                let mut sc = vec![0.0f32; total];
                for r in r0..r1 {
                    let (t, hh) = (r / nh, r % nh);
                    let kvh = hh / rep;
                    let qv = &q[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
                    let upto = pos + t + 1;
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..upto {
                        let kv = &kc[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                        let s = dot(qv, kv) * scale;
                        sc[j] = s;
                        mx = mx.max(s);
                    }
                    let mut den = 0.0f32;
                    for s in &mut sc[..upto] {
                        *s = (*s - mx).exp();
                        den += *s;
                    }
                    let out =
                        unsafe { std::slice::from_raw_parts_mut(o.0.add((t * nh + hh) * hd), hd) };
                    out.fill(0.0);
                    for j in 0..upto {
                        let p = sc[j] / den;
                        let vv = &vc[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                        for (a, &b) in out.iter_mut().zip(vv) {
                            *a += p * b;
                        }
                    }
                }
            };
            par_rows(pool, n * nh, &run);
        }
        let mut out = vec![0.0f32; n * g.hidden];
        project(&w.o, &o, n, &mut out, pool);
        out
    }

    /// Router: sigmoid scores, choice by score + bias, weights from the
    /// unbiased scores, renormalized (`sum + 1e-20`) and scaled.
    pub fn route(&self, w: &MoeW, logits: &[f32], sel: &mut Vec<(usize, f32)>) {
        let g = &self.g;
        sel.clear();
        let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
        let mut order: Vec<usize> = (0..g.n_exp).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + w.bias[b])
                .partial_cmp(&(scores[a] + w.bias[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut sum = 0.0f32;
        for &e in &order[..g.top_k] {
            sel.push((e, scores[e]));
            sum += scores[e];
        }
        let den = if g.norm_topk { sum + 1e-20 } else { 1.0 };
        for s in sel.iter_mut() {
            s.1 = s.1 / den * g.route_scale;
        }
    }

    fn moe(&self, bi: usize, w: &MoeW, xn: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let g = &self.g;
        let hd = g.hidden;
        let mut logits = vec![0.0f32; n * g.n_exp];
        w.router.matmat(xn, n, &mut logits, pool);
        // tokens per expert
        let mut lists: Vec<Vec<(usize, f32)>> = vec![Vec::new(); g.n_exp];
        let mut sel = Vec::with_capacity(g.top_k);
        let mut cap = self.capture.lock().unwrap();
        let mut cap = cap.as_mut().filter(|_| bi != usize::MAX);
        if let Some(c) = cap.as_mut() {
            c.x[bi].extend_from_slice(xn);
        }
        for t in 0..n {
            self.route(w, &logits[t * g.n_exp..(t + 1) * g.n_exp], &mut sel);
            for &(e, wt) in &sel {
                lists[e].push((t, wt));
                if let Some(c) = cap.as_mut() {
                    c.picks[bi].push(e as u16);
                }
            }
        }
        drop(cap);
        let mut out = vec![0.0f32; n * hd];
        let mut xs = Vec::new();
        let mut mid = Vec::new();
        let mut ys = Vec::new();
        for (e, list) in lists.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let m = list.len();
            xs.clear();
            for &(t, _) in list {
                xs.extend_from_slice(&xn[t * hd..(t + 1) * hd]);
            }
            mid.resize(m * g.moe_inter, 0.0);
            project(&w.up[e], &xs, m, &mut mid, pool);
            for v in mid.iter_mut() {
                *v = relu2(*v);
            }
            ys.resize(m * hd, 0.0);
            project(&w.down[e], &mid, m, &mut ys, pool);
            for (i, &(t, wt)) in list.iter().enumerate() {
                for (a, &b) in out[t * hd..(t + 1) * hd]
                    .iter_mut()
                    .zip(&ys[i * hd..(i + 1) * hd])
                {
                    *a += wt * b;
                }
            }
        }
        let mut smid = vec![0.0f32; n * g.shared_inter];
        project(&w.sh_up, xn, n, &mut smid, pool);
        for v in smid.iter_mut() {
            *v = relu2(*v);
        }
        let mut sy = vec![0.0f32; n * hd];
        project(&w.sh_down, &smid, n, &mut sy, pool);
        for (a, b) in out.iter_mut().zip(&sy) {
            *a += b;
        }
        out
    }
}

#[cfg(feature = "gpu")]
impl NemotronH {
    /// KV capacity of the device frame.
    const FRAME_CAP: usize = 65536;
    /// Greedy tokens per frame submit.
    const FRAME_STEPS: usize = 16;

    fn frame_desc(&self) -> Option<crate::gpu_wgpu::nemotron::NemoDesc<'_>> {
        use crate::gpu_wgpu::nemotron::{ExpertCodec, NemoBlockDesc, NemoDesc};
        let m = self.model.as_ref()?;
        let raw = |q: &QTensor, want: TensorDtype| -> Option<&[u8]> {
            let idx = q.model_idx()?;
            let e = &m.tensors[idx];
            (e.dtype == want).then(|| m.entry_bytes(e))
        };
        let q8 = |q: &QTensor| raw(q, TensorDtype::Q8_2f);
        let mut blocks = Vec::with_capacity(self.blocks.len());
        let mut moe = 0usize;
        for b in &self.blocks {
            blocks.push(match &b.mixer {
                Mixer::Mamba(w) => NemoBlockDesc::Mamba {
                    norm: &b.norm,
                    in_proj: q8(&w.in_proj)?,
                    out_proj: q8(&w.out_proj)?,
                    conv_w: &w.conv_w,
                    conv_b: &w.conv_b,
                    par: w.a.iter().chain(&w.d).chain(&w.dt_bias).copied().collect(),
                    gnorm: &w.norm,
                },
                Mixer::Attn(w) => NemoBlockDesc::Attn {
                    norm: &b.norm,
                    q: q8(&w.q)?,
                    k: q8(&w.k)?,
                    v: q8(&w.v)?,
                    o: q8(&w.o)?,
                },
                Mixer::Moe(w) => {
                    fn bank<'m>(
                        m: &'m CmfModel,
                        v: &[QTensor],
                        want: TensorDtype,
                    ) -> Option<Vec<&'m [u8]>> {
                        v.iter()
                            .map(|q| {
                                let e = &m.tensors[q.model_idx()?];
                                (e.dtype == want).then(|| m.entry_bytes(e))
                            })
                            .collect()
                    }
                    let (q4t, q8t) = (TensorDtype::Q4TiledP, TensorDtype::Q8_2f);
                    let (codec, up, down) = match (bank(m, &w.up, q4t), bank(m, &w.down, q4t)) {
                        (Some(u), Some(d)) => (ExpertCodec::Q4tp, u, d),
                        _ => (ExpertCodec::Q8, bank(m, &w.up, q8t)?, bank(m, &w.down, q8t)?),
                    };
                    moe += 1;
                    NemoBlockDesc::Moe {
                        norm: &b.norm,
                        router: w.router.as_f32()?,
                        bias: &w.bias,
                        codec,
                        up,
                        down,
                        hot: self.hot.get(moe - 1).cloned(),
                        sh_up: q8(&w.sh_up)?,
                        sh_down: q8(&w.sh_down)?,
                    }
                }
            });
        }
        let g = &self.g;
        Some(NemoDesc {
            hidden: g.hidden,
            vocab: g.vocab,
            eps: g.eps,
            m_heads: g.m_heads,
            m_head_dim: g.m_head_dim,
            m_groups: g.m_groups,
            m_state: g.m_state,
            nh: g.nh,
            nkv: g.nkv,
            hd: g.hd,
            n_exp: g.n_exp,
            top_k: g.top_k,
            moe_inter: g.moe_inter,
            shared_inter: g.shared_inter,
            route_scale: g.route_scale,
            norm_topk: g.norm_topk,
            blocks,
            embed: q8(&self.embed)?,
            lm_head: q8(&self.lm_head)?,
            final_norm: &self.final_norm,
            cap: Self::FRAME_CAP,
            model: Some(m.clone()),
            // the audio towers share the card
            reserve: 2_000_000_000,
            mtp: self.mtp_desc(),
        })
    }

    /// The draft layer for the frame (speculation; `CMF_NEMO_MTP=0` turns
    /// it off): used when every expert fits the card.
    fn mtp_desc(&self) -> Option<crate::gpu_wgpu::nemotron::NemoMtpDesc<'_>> {
        if std::env::var("CMF_NEMO_MTP").as_deref() == Ok("0") {
            return None;
        }
        let w = self.mtp.as_ref()?;
        let m = self.model.as_ref()?;
        fn raw<'m>(m: &'m CmfModel, q: &QTensor, want: TensorDtype) -> Option<&'m [u8]> {
            let e = &m.tensors[q.model_idx()?];
            (e.dtype == want).then(|| m.entry_bytes(e))
        }
        let q8 = TensorDtype::Q8_2f;
        let codec = match self.blocks.iter().find_map(|b| match &b.mixer {
            Mixer::Moe(w) => Some(&w.up[0]),
            _ => None,
        }) {
            Some(q) if raw(m, q, TensorDtype::Q4TiledP).is_some() => TensorDtype::Q4TiledP,
            _ => q8,
        };
        let bank = |v: &[QTensor]| v.iter().map(|q| raw(m, q, codec)).collect::<Option<Vec<_>>>();
        Some(crate::gpu_wgpu::nemotron::NemoMtpDesc {
            enorm: &w.enorm,
            hnorm: &w.hnorm,
            eh: raw(m, &w.eh, q8)?,
            h_first: std::env::var("CMF_NEMO_MTP_HFIRST").as_deref() == Ok("1"),
            n_attn: &w.n_attn,
            q: raw(m, &w.attn.q, q8)?,
            k: raw(m, &w.attn.k, q8)?,
            v: raw(m, &w.attn.v, q8)?,
            o: raw(m, &w.attn.o, q8)?,
            n_moe: &w.n_moe,
            router: w.moe.router.as_f32()?,
            bias: &w.moe.bias,
            up: bank(&w.moe.up)?,
            down: bank(&w.moe.down)?,
            sh_up: raw(m, &w.moe.sh_up, q8)?,
            sh_down: raw(m, &w.moe.sh_down, q8)?,
            fnorm: &w.fnorm,
        })
    }

    /// Hand the decoding state to the device frame (built on first use).
    /// False: no frame here (CPU decode continues); the reason is logged.
    pub fn frame_seed(&self, st: &State) -> bool {
        if std::env::var("CMF_NEMO_FRAME").as_deref() == Ok("0") || !crate::gpu::wgpu_active() {
            return false;
        }
        let mut slot = self.frame.lock().unwrap();
        if slot.is_none() {
            // Unified memory: the frame COPIES the decoder into device
            // buffers next to the mapped file; a host without room for both
            // swaps (measured: 24 GB M4, 2.4 against 9-15 tok/s per-op).
            if !crate::gpu::discrete() && std::env::var("CMF_NEMO_EXPERT_GB").is_err() {
                let need = self
                    .model
                    .as_ref()
                    .and_then(|m| std::fs::metadata(&m.path).ok())
                    .map_or(0, |md| md.len());
                let total = crate::fcd::physical_total_bytes().unwrap_or(u64::MAX);
                if need.saturating_mul(2) > total {
                    tracing::info!(
                        "nemotron frame: not built on unified memory ({:.1} GB model, {:.1} GB RAM)",
                        need as f64 / 1e9,
                        total as f64 / 1e9
                    );
                    *slot = Some(Err(()));
                    return false;
                }
            }
            let t0 = std::time::Instant::now();
            let built = self
                .frame_desc()
                .and_then(|d| crate::gpu_wgpu::nemotron::NemoFrame::build(&d));
            match &built {
                Some(f) => tracing::info!(
                    "nemotron frame: {:.2} GB resident in {:.1}s",
                    f.bytes as f64 / 1e9,
                    t0.elapsed().as_secs_f64()
                ),
                None => tracing::info!(
                    "nemotron frame: not built (codecs or device outside the frame contract)"
                ),
            }
            *slot = Some(built.ok_or(()));
            if let Some(Ok(f)) = slot.as_mut()
                && f.has_mtp()
                && !f.prepare_verify()
            {
                tracing::info!("nemotron frame: no two-position verify pass, drafts off");
            }
        }
        match slot.as_mut() {
            Some(Ok(f)) => f.seed(&st.conv, &st.ssm, &st.kc, &st.vc, st.pos),
            _ => false,
        }
    }

    /// Prefill `rows` through the device frame from an empty state. False:
    /// the frame is absent or cannot prefill (the caller prefills on the
    /// host and seeds).
    pub fn frame_prefill(&self, rows: &[f32], next: Option<&[f32]>, pool: Option<&Pool>) -> bool {
        let hd = self.g.hidden;
        if !self.frame_seed(&self.new_state()) {
            return false;
        }
        if rows.is_empty() {
            return true;
        }
        let mut slot = self.frame.lock().unwrap();
        let Some(Ok(f)) = slot.as_mut() else {
            return false;
        };
        debug_assert_eq!(rows.len() % hd, 0);
        // batched passes when the device takes them, position by position
        // otherwise; a partial expert bank's misses are completed here
        let batched = std::env::var("CMF_NEMO_BATCH").as_deref() != Ok("0");
        let mut host = |moe: usize, e: usize, x: &[f32], n: usize| self.complete_rows(moe, e, x, n, pool);
        let draft = f.has_mtp();
        (batched && f.feed_batched(rows, Some(&mut host), draft, next)) || f.feed(rows)
    }

    /// Greedy tokens on the device after `row` (the next input embedding)
    /// until `keep(ids)` returns false or `max` ids are out. Full expert
    /// banks keep two batches in flight; misses of a partial bank are
    /// completed here on the CPU.
    pub fn frame_decode(
        &self,
        row: &[f32],
        max: usize,
        pool: Option<&Pool>,
        mut keep: impl FnMut(&[u32]) -> bool,
    ) -> Result<(), String> {
        let mut slot = self.frame.lock().unwrap();
        let Some(Ok(f)) = slot.as_mut() else {
            return Err("no device decode frame".into());
        };
        let fail = || "device decode frame failed".to_string();
        let mut left = max;
        if left == 0 {
            return Ok(());
        }
        if f.full() && f.has_mtp() && f.verify_ready() {
            return self.decode_mtp(f, row, max, &mut keep);
        }
        if f.full() {
            let k = left.min(Self::FRAME_STEPS);
            if !f.greedy_start(Some(row), k) {
                return Err(fail());
            }
            left -= k;
            loop {
                if left > 0 {
                    let k = left.min(Self::FRAME_STEPS);
                    if f.greedy_start(None, k) {
                        left -= k;
                    }
                }
                let ids = f.greedy_finish().ok_or_else(fail)?;
                if !keep(&ids) {
                    f.greedy_drain();
                    return Ok(());
                }
                if left == 0 && f.in_flight() == 0 {
                    return Ok(());
                }
            }
        }
        // a partial bank: one MoE layer per submit, misses completed here
        let mut first = Some(row);
        let mut host_s = 0.0f64;
        let mut complete = |moe: usize, xn: &[f32], picks: &[(usize, f32)], missing: &[usize]| {
            let t = std::time::Instant::now();
            let o = self.complete(moe, xn, picks, missing, pool);
            host_s += t.elapsed().as_secs_f64();
            o
        };
        while left > 0 {
            let ids = f.greedy_seg(first.take(), 1, &mut complete).ok_or_else(fail)?;
            left -= 1;
            if !keep(&ids) {
                break;
            }
        }
        if std::env::var("CMF_NEMO_PROF").as_deref() == Ok("1") {
            eprintln!("nemotron frame: {:.1} ms completing missing experts on the host", host_s * 1e3);
        }
        if std::env::var("CMF_NEMO_PROF").as_deref() == Ok("1") {
            if let Some((miss, adm)) = f.arena_stats() {
                eprintln!("nemotron frame: {miss} expert misses completed on the host, {adm} admitted");
            }
        }
        Ok(())
    }

    /// Speculative greedy decode with the draft layer: each round verifies
    /// the next token and one draft in a batched pass, keeps the draft when
    /// the model agrees (two tokens for the pass) or rolls the state back
    /// to the token, then extends the draft layer over the kept positions
    /// and drafts again. Greedy-exact: every emitted id is the main model's
    /// argmax.
    fn decode_mtp(
        &self,
        f: &mut crate::gpu_wgpu::nemotron::NemoFrame,
        row: &[f32],
        max: usize,
        keep: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<(), String> {
        let hd = self.g.hidden;
        let fail = || "device decode frame failed".to_string();
        let pre = std::env::var("CMF_NEMO_MTP_PRE").as_deref() == Ok("1");
        let prof = std::env::var("CMF_NEMO_PROF").as_deref() == Ok("1");
        let emb = |t: u32| {
            let mut r = vec![0.0f32; hd];
            self.embed_row(t, &mut r);
            r
        };
        // the hidden row the draft layer reads: the residual before or after the final norm
        let pick = |hn: &[f32], hr: &[f32], i: usize| -> Vec<f32> {
            let src = if pre { hr } else { hn };
            src[i * hd..(i + 1) * hd].to_vec()
        };
        let p0 = f.pos;
        let (y0, h0) = f.step_hidden(row).ok_or_else(fail)?;
        let mut out = 1usize;
        if !keep(&[y0]) || out >= max {
            return Ok(());
        }
        let mut tok = y0;
        let mut tok_e = emb(tok);
        let (mut t_ver, mut t_mtp) = (0.0f64, 0.0f64);
        let t = std::time::Instant::now();
        let mut draft = f.mtp_steps(&[(&h0, &tok_e, p0)], true).ok_or_else(fail)?;
        t_mtp += t.elapsed().as_secs_f64();
        let (mut rounds, mut kept) = (0usize, 0usize);
        while out < max {
            let p = f.pos;
            let d_e = emb(draft);
            let mut rows = tok_e.clone();
            rows.extend_from_slice(&d_e);
            let t = std::time::Instant::now();
            let (y, hn) = f.verify2(&rows).ok_or_else(fail)?;
            t_ver += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let hr = &hn;
            rounds += 1;
            if y[0] == draft {
                f.accept2(2);
                kept += 1;
                let take = (max - out).min(2);
                out += take;
                if !keep(&y[..take]) || out >= max {
                    break;
                }
                let (h0, h1) = (pick(&hn, &hr, 0), pick(&hn, &hr, 1));
                tok = y[1];
                tok_e = emb(tok);
                draft = f
                    .mtp_steps(&[(&h0, &d_e, p), (&h1, &tok_e, p + 1)], true)
                    .ok_or_else(fail)?;
                t_mtp += t.elapsed().as_secs_f64();
            } else {
                f.accept2(1);
                out += 1;
                if !keep(&y[..1]) || out >= max {
                    break;
                }
                let h0 = pick(&hn, &hr, 0);
                tok = y[0];
                tok_e = emb(tok);
                draft = f.mtp_steps(&[(&h0, &tok_e, p)], true).ok_or_else(fail)?;
                t_mtp += t.elapsed().as_secs_f64();
            }
        }
        if prof {
            eprintln!(
                "nemotron draft: {kept}/{rounds} drafts kept ({:.1}%), {:.2} tokens per pass, verify {:.2} ms, draft {:.2} ms per round",
                100.0 * kept as f64 / rounds.max(1) as f64,
                out as f64 / (rounds + 1) as f64,
                t_ver * 1e3 / rounds.max(1) as f64,
                t_mtp * 1e3 / (rounds + 1) as f64
            );
        }
        Ok(())
    }

    /// Expert `e` of MoE block `moe` over `n` normed rows:
    /// `down(relu²(up(x)))`, unweighted, `[n · hidden]`.
    fn complete_rows(&self, moe: usize, e: usize, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let g = &self.g;
        let Some(w) = self
            .blocks
            .iter()
            .filter_map(|b| match &b.mixer {
                Mixer::Moe(w) => Some(w),
                _ => None,
            })
            .nth(moe)
        else {
            return Vec::new();
        };
        let mut mid = vec![0.0f32; n * g.moe_inter];
        project(&w.up[e], x, n, &mut mid, pool);
        for v in mid.iter_mut() {
            *v = relu2(*v);
        }
        let mut y = vec![0.0f32; n * g.hidden];
        project(&w.down[e], &mid, n, &mut y, pool);
        y
    }

    /// The outputs of routed experts the device bank does not hold, for
    /// one row: `down(relu²(up(xn)))`, unweighted.
    fn complete(
        &self,
        moe: usize,
        xn: &[f32],
        picks: &[(usize, f32)],
        missing: &[usize],
        pool: Option<&Pool>,
    ) -> Vec<Vec<f32>> {
        let g = &self.g;
        let Some(w) = self
            .blocks
            .iter()
            .filter_map(|b| match &b.mixer {
                Mixer::Moe(w) => Some(w),
                _ => None,
            })
            .nth(moe)
        else {
            return Vec::new();
        };
        missing
            .iter()
            .map(|&i| {
                let e = picks[i].0;
                let mut mid = vec![0.0f32; g.moe_inter];
                project(&w.up[e], xn, 1, &mut mid, pool);
                for v in mid.iter_mut() {
                    *v = relu2(*v);
                }
                let mut y = vec![0.0f32; g.hidden];
                project(&w.down[e], &mid, 1, &mut y, pool);
                y
            })
            .collect()
    }
}

#[cfg(not(feature = "gpu"))]
impl NemotronH {
    pub fn frame_seed(&self, _st: &State) -> bool {
        false
    }
    pub fn frame_prefill(&self, _rows: &[f32], _next: Option<&[f32]>, _pool: Option<&Pool>) -> bool {
        false
    }
    pub fn frame_decode(
        &self,
        _row: &[f32],
        _max: usize,
        _pool: Option<&Pool>,
        _keep: impl FnMut(&[u32]) -> bool,
    ) -> Result<(), String> {
        Err("no device decode frame".into())
    }
}

/// `out[n · rows] = x[n · cols] · Wᵀ` — matvec for one row, matmat otherwise.
pub(crate) fn project(w: &QTensor, x: &[f32], n: usize, out: &mut [f32], pool: Option<&Pool>) {
    if n == 1 {
        w.matvec(x, out, pool);
    } else {
        w.matmat(x, n, out, pool);
    }
}

pub(crate) fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = g * (v * inv);
    }
}

#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    // torch.nn.functional.softplus: threshold 20
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[inline]
fn relu2(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x * y;
    }
    s
}

#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Split `0..rows` over the pool (serially without one).
pub(crate) fn par_rows(pool: Option<&Pool>, rows: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) if rows > 1 => p.run_rows(rows, f),
        _ => f(0, rows),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> NemotronH {
        let g = Geom {
            hidden: 8,
            vocab: 4,
            eps: 1e-5,
            m_heads: 4,
            m_head_dim: 2,
            m_groups: 2,
            m_state: 3,
            conv_k: 4,
            nh: 4,
            nkv: 2,
            hd: 2,
            n_exp: 4,
            top_k: 2,
            moe_inter: 6,
            shared_inter: 5,
            route_scale: 2.5,
            norm_topk: true,
        };
        let seed = std::cell::Cell::new(1u64);
        let rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed.set(
                        seed.get()
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407),
                    );
                    ((seed.get() >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.6
                })
                .collect()
        };
        let qt = |r: usize, c: usize| QTensor::from_f32(rnd(r * c), r, c);
        let mamba = Mixer::Mamba(MambaW {
            in_proj: qt(g.in_proj_rows(), g.hidden),
            out_proj: qt(g.hidden, g.m_inner()),
            conv_w: rnd(g.conv_dim() * g.conv_k),
            conv_b: rnd(g.conv_dim()),
            a: rnd(g.m_heads).iter().map(|v| -(v.abs() + 0.1)).collect(),
            d: rnd(g.m_heads),
            dt_bias: rnd(g.m_heads),
            norm: rnd(g.m_inner()).iter().map(|v| 1.0 + v).collect(),
        });
        let attn = Mixer::Attn(AttnW {
            q: qt(g.nh * g.hd, g.hidden),
            k: qt(g.nkv * g.hd, g.hidden),
            v: qt(g.nkv * g.hd, g.hidden),
            o: qt(g.hidden, g.nh * g.hd),
        });
        let moe = Mixer::Moe(MoeW {
            router: qt(g.n_exp, g.hidden),
            bias: rnd(g.n_exp),
            up: (0..g.n_exp).map(|_| qt(g.moe_inter, g.hidden)).collect(),
            down: (0..g.n_exp).map(|_| qt(g.hidden, g.moe_inter)).collect(),
            sh_up: qt(g.shared_inter, g.hidden),
            sh_down: qt(g.hidden, g.shared_inter),
        });
        let norm = || vec![1.0f32; 8];
        NemotronH {
            embed: qt(g.vocab, g.hidden),
            lm_head: qt(g.vocab, g.hidden),
            final_norm: norm(),
            blocks: vec![
                Block {
                    norm: norm(),
                    mixer: mamba,
                    slot: 0,
                },
                Block {
                    norm: norm(),
                    mixer: moe,
                    slot: 0,
                },
                Block {
                    norm: norm(),
                    mixer: attn,
                    slot: 0,
                },
            ],
            g,
            n_mamba: 1,
            n_attn: 1,
            model: None,
            hot: Vec::new(),
            mtp: None,
            capture: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu")]
            frame: std::sync::Mutex::new(None),
        }
    }

    /// A batched prefill and a token-by-token decode are the same function.
    #[test]
    fn batched_prefill_equals_stepwise_decode() {
        let m = tiny();
        let rows: Vec<f32> = (0..5 * 8)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let mut a = m.new_state();
        let la = m.forward(&mut a, &rows, None);
        let mut b = m.new_state();
        let mut lb = Vec::new();
        for t in 0..5 {
            lb = m.forward(&mut b, &rows[t * 8..(t + 1) * 8], None);
        }
        for (x, y) in la.iter().zip(&lb) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
        for (x, y) in a.ssm[0].iter().zip(&b.ssm[0]) {
            assert!((x - y).abs() < 1e-5);
        }
        assert_eq!(a.pos, 5);
        assert_eq!(a.kc[0].len(), 5 * 2 * 2);
    }

    #[test]
    fn router_matches_reference_contract() {
        let m = tiny();
        let Mixer::Moe(w) = &m.blocks[1].mixer else {
            unreachable!()
        };
        let logits = [0.3f32, -1.0, 2.0, 0.1];
        let mut sel = Vec::new();
        m.route(w, &logits, &mut sel);
        let s: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
        let mut idx: Vec<usize> = (0..4).collect();
        idx.sort_by(|&a, &b| (s[b] + w.bias[b]).partial_cmp(&(s[a] + w.bias[a])).unwrap());
        let sum = s[idx[0]] + s[idx[1]] + 1e-20;
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].0, idx[0]);
        assert!((sel[0].1 - s[idx[0]] / sum * 2.5).abs() < 1e-6);
        assert!((sel[0].1 + sel[1].1 - 2.5).abs() < 1e-5);
    }
}
