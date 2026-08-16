//! Embryo genome: configuration, the flat parameter arena, and the fixed
//! training graph (forward + hand-rolled backward) on our Metal kernels.
//! See docs/NATIVE_MODEL_TECH.ru.md §3.
//!
//! One layer:
//!   x ─► RMSNorm ─► MIXER (hybrid_k, or the softmax anchor every 8th) ─► +res
//!     ─► RMSNorm ─► FFN (SwiGLU; the shared expert — routed experts are the
//!        growth slots, next) ─► +res
//! Head: tied embedding, full softmax (the hierarchical 128×256 head is next).
//!
//! No autograd: the graph is fixed, every block's backward is written out
//! (llm.c style, same discipline as the runtime's `fcd_ops`).

/// Embryo-0 as born (§3.2). All matrix dims are multiples of 64 so the
/// GEMM tile contract holds without edge paths.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbryoCfg {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    /// every `anchor_every`-th layer is a softmax anchor (o1-ready)
    pub anchor_every: usize,
    // hybrid_k mixer
    pub heads: usize,
    pub nphase: usize,
    pub dv: usize,
    /// decay horizons: log grid [h_min, h_max] over the phase pairs
    pub horizon_min: f64,
    pub horizon_max: f64,
    /// κ = σ(W_κ x + kappa_bias)
    pub kappa_bias: f32,
    // anchor (GQA softmax)
    pub anchor_q_heads: usize,
    pub anchor_kv_heads: usize,
    pub anchor_hd: usize,
    pub rope_base: f32,
    // experts
    pub experts: usize,
    pub inter: usize,
    // hierarchical head
    pub head_clusters: usize,
    pub mtp_heads: usize,
    pub seq: usize,
    pub norm_eps: f32,
}

impl EmbryoCfg {
    pub fn embryo0() -> Self {
        EmbryoCfg {
            vocab: 32768,
            hidden: 384,
            layers: 8,
            anchor_every: 8,
            heads: 8,
            nphase: 32,
            dv: 128,
            horizon_min: 8.0,
            horizon_max: 2048.0,
            kappa_bias: 2.0,
            anchor_q_heads: 8,
            anchor_kv_heads: 2,
            anchor_hd: 128,
            rope_base: 10000.0,
            experts: 4,
            inter: 768,
            head_clusters: 128,
            mtp_heads: 2,
            seq: 1024,
            norm_eps: 1e-6,
        }
    }
    /// A tiny genome for smoke tests and gradchecks (same shape family,
    /// every dim still a multiple of the GEMM tile).
    pub fn tiny() -> Self {
        EmbryoCfg {
            vocab: 512,
            hidden: 64,
            layers: 2,
            anchor_every: 2,
            heads: 2,
            nphase: 32,
            dv: 64,
            horizon_min: 4.0,
            horizon_max: 128.0,
            kappa_bias: 2.0,
            anchor_q_heads: 2,
            anchor_kv_heads: 1,
            anchor_hd: 64,
            rope_base: 10000.0,
            experts: 0,
            inter: 128,
            head_clusters: 0,
            mtp_heads: 0,
            seq: 64,
            norm_eps: 1e-6,
        }
    }
    pub fn is_anchor(&self, layer: usize) -> bool {
        (layer + 1) % self.anchor_every == 0
    }
    /// κ pre-activation columns the projection GEMM produces (padded to
    /// the GEMM tile; only `heads` are real).
    pub fn kappa_ld(&self) -> usize {
        self.heads.div_ceil(64) * 64
    }
    /// Parameter count (total, active per token) INCLUDING the routed
    /// experts of §3 (which the trainer does not instantiate yet).
    pub fn params(&self) -> (usize, usize) {
        let h = self.hidden;
        let embed = self.vocab * h; // tied lm_head
        let mixer = 2 * (self.heads * self.nphase * h) // thq, thk
            + self.heads * self.dv * h                  // v_proj
            + h * self.heads * self.dv                  // out_proj
            + self.heads * h;                           // κ gate
        let anchor = self.anchor_q_heads * self.anchor_hd * h
            + 2 * self.anchor_kv_heads * self.anchor_hd * h
            + h * self.anchor_q_heads * self.anchor_hd;
        let ffn_one = 3 * h * self.inter;
        let ffn_total = ffn_one * (self.experts + 1);
        let ffn_active = ffn_one * (1 + self.experts.min(1)); // top-1 + shared
        let norms = 2 * h;
        let mut total = embed + h;
        let mut active = embed + h;
        for l in 0..self.layers {
            let mix = if self.is_anchor(l) { anchor } else { mixer };
            total += mix + ffn_total + norms;
            active += mix + ffn_active + norms;
        }
        (total, active)
    }
}

// ---------------------------------------------------------------------
// Parameter arena layout
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FfnOffs {
    pub wg: usize, // [I, H]
    pub wu: usize, // [I, H]
    pub wd: usize, // [H, I]
}

#[derive(Clone, Debug)]
pub enum LayerOffs {
    Mixer {
        ln1: usize,
        wq: usize,   // [nh·nph, H]
        wk: usize,   // [nh·nph, H]
        wv: usize,   // [nh·dv, H]
        wkap: usize, // [kappa_ld, H] (rows ≥ nh are zero, never trained)
        wo: usize,   // [H, nh·dv]
        ln2: usize,
        ffn: FfnOffs,
    },
    Anchor {
        ln1: usize,
        wq: usize, // [qh·hd, H]
        wk: usize, // [kvh·hd, H]
        wv: usize, // [kvh·hd, H]
        wo: usize, // [H, qh·hd]
        ln2: usize,
        ffn: FfnOffs,
    },
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub total: usize,
    pub embed: usize, // [V, H]
    pub final_norm: usize,
    pub layers: Vec<LayerOffs>,
    /// (name, offset, len) of every tensor — checkpoints and the CMF export
    pub names: Vec<(String, usize, usize)>,
}

impl Layout {
    pub fn new(cfg: &EmbryoCfg) -> Layout {
        let h = cfg.hidden;
        let mut off = 0usize;
        let mut names = Vec::new();
        let mut take = |name: String, n: usize| -> usize {
            let o = off;
            names.push((name, o, n));
            off += n;
            o
        };
        let embed = take("embed".into(), cfg.vocab * h);
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let ffn = |take: &mut dyn FnMut(String, usize) -> usize| FfnOffs {
                wg: take(format!("layers.{l}.ffn.gate"), cfg.inter * h),
                wu: take(format!("layers.{l}.ffn.up"), cfg.inter * h),
                wd: take(format!("layers.{l}.ffn.down"), h * cfg.inter),
            };
            if cfg.is_anchor(l) {
                let ln1 = take(format!("layers.{l}.ln1"), h);
                let wq = take(format!("layers.{l}.attn.q"), cfg.anchor_q_heads * cfg.anchor_hd * h);
                let wk = take(format!("layers.{l}.attn.k"), cfg.anchor_kv_heads * cfg.anchor_hd * h);
                let wv = take(format!("layers.{l}.attn.v"), cfg.anchor_kv_heads * cfg.anchor_hd * h);
                let wo = take(format!("layers.{l}.attn.o"), h * cfg.anchor_q_heads * cfg.anchor_hd);
                let ln2 = take(format!("layers.{l}.ln2"), h);
                let f = ffn(&mut take);
                layers.push(LayerOffs::Anchor { ln1, wq, wk, wv, wo, ln2, ffn: f });
            } else {
                let ln1 = take(format!("layers.{l}.ln1"), h);
                let wq = take(format!("layers.{l}.hk.thq"), cfg.heads * cfg.nphase * h);
                let wk = take(format!("layers.{l}.hk.thk"), cfg.heads * cfg.nphase * h);
                let wv = take(format!("layers.{l}.hk.v"), cfg.heads * cfg.dv * h);
                let wkap = take(format!("layers.{l}.hk.kappa"), cfg.kappa_ld() * h);
                let wo = take(format!("layers.{l}.hk.o"), h * cfg.heads * cfg.dv);
                let ln2 = take(format!("layers.{l}.ln2"), h);
                let f = ffn(&mut take);
                layers.push(LayerOffs::Mixer { ln1, wq, wk, wv, wkap, wo, ln2, ffn: f });
            }
        }
        let final_norm = take("final_norm".into(), h);
        Layout { total: off, embed, final_norm, layers, names }
    }
}

/// Standard normal samples (Box–Muller over the splitmix stream).
pub fn gauss_vec(seed: u64, n: usize) -> Vec<f32> {
    let u = crate::ops::lcg_vec(seed, 2 * n + 2);
    (0..n)
        .map(|i| {
            let a = (u[2 * i] as f64 * 0.5 + 0.5).clamp(1e-12, 1.0);
            let b = u[2 * i + 1] as f64 * 0.5 + 0.5;
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        })
        .collect()
}

/// Initialise a parameter arena on the host: N(0, 0.02) matrices, output
/// projections scaled by 1/√(2·layers), norms 1, κ pad rows 0.
pub fn init_params(cfg: &EmbryoCfg, lay: &Layout, seed: u64) -> Vec<f32> {
    let mut p = vec![0.0f32; lay.total];
    let std = 0.02f32;
    let out_scale = 1.0 / (2.0 * cfg.layers as f32).sqrt();
    let mut seed_i = seed;
    let mut fill = |p: &mut [f32], off: usize, n: usize, s: f32| {
        seed_i += 1;
        let g = gauss_vec(seed_i, n);
        for i in 0..n {
            p[off + i] = g[i] * s;
        }
    };
    let h = cfg.hidden;
    fill(&mut p, lay.embed, cfg.vocab * h, std);
    for (l, lo) in lay.layers.iter().enumerate() {
        let _ = l;
        match lo {
            LayerOffs::Mixer { ln1, wq, wk, wv, wkap, wo, ln2, ffn } => {
                p[*ln1..*ln1 + h].fill(1.0);
                p[*ln2..*ln2 + h].fill(1.0);
                // phase projections: θ = W·x̂ with x̂ RMS-normed → θ std ≈ s·√H;
                // s = 0.05 gives θ ≈ N(0, 1) — a full turn of phase spread.
                let s_theta = 1.0 / (h as f32).sqrt();
                fill(&mut p, *wq, cfg.heads * cfg.nphase * h, s_theta);
                fill(&mut p, *wk, cfg.heads * cfg.nphase * h, s_theta);
                fill(&mut p, *wv, cfg.heads * cfg.dv * h, std);
                fill(&mut p, *wkap, cfg.heads * h, std); // pad rows stay 0
                fill(&mut p, *wo, h * cfg.heads * cfg.dv, std * out_scale);
                fill(&mut p, ffn.wg, cfg.inter * h, std);
                fill(&mut p, ffn.wu, cfg.inter * h, std);
                fill(&mut p, ffn.wd, h * cfg.inter, std * out_scale);
            }
            LayerOffs::Anchor { ln1, wq, wk, wv, wo, ln2, ffn } => {
                p[*ln1..*ln1 + h].fill(1.0);
                p[*ln2..*ln2 + h].fill(1.0);
                fill(&mut p, *wq, cfg.anchor_q_heads * cfg.anchor_hd * h, std);
                fill(&mut p, *wk, cfg.anchor_kv_heads * cfg.anchor_hd * h, std);
                fill(&mut p, *wv, cfg.anchor_kv_heads * cfg.anchor_hd * h, std);
                fill(&mut p, *wo, h * cfg.anchor_q_heads * cfg.anchor_hd, std * out_scale);
                fill(&mut p, ffn.wg, cfg.inter * h, std);
                fill(&mut p, ffn.wu, cfg.inter * h, std);
                fill(&mut p, ffn.wd, h * cfg.inter, std * out_scale);
            }
        }
    }
    p[lay.final_norm..lay.final_norm + h].fill(1.0);
    p
}

#[cfg(target_os = "macos")]
pub use gpu::*;

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::metal::{Cmd, Ctx, GBuf, HkDims, HkGrads, HkScratch, HkWork, Op, ctx, hk_pow_table};
    use crate::ops::hk_decay_grid;

    /// Per-layer activation buffers kept for the backward (M = B·T rows).
    pub enum LayerActs {
        Mixer {
            x_in: GBuf,   // [M,H] residual stream entering the layer
            x1: GBuf,     // [M,H] normed
            inv1: GBuf,   // [M]
            thq: GBuf,    // [M, nh·nph]
            thk: GBuf,
            v: GBuf,      // [M, nh·dv]
            kpre: GBuf,   // [M, kappa_ld]
            kappa: GBuf,  // [M, nh]
            phq: GBuf,    // [M, nh·2nph]
            phk: GBuf,
            kv: GBuf,     // [M, nh·dv]
            states: GBuf, // [B·nh·(T/64+1)·2nph·dv]
            o: GBuf,      // [M, nh·dv]
            x_mid: GBuf,  // [M,H]
            x2: GBuf,     // [M,H]
            inv2: GBuf,
            gte: GBuf,    // [M,I]
            up: GBuf,
            hh: GBuf,
        },
        Anchor {
            x_in: GBuf,
            x1: GBuf,
            inv1: GBuf,
            q: GBuf, // [M, qh·hd] (after RoPE)
            k: GBuf, // [M, kvh·hd]
            v: GBuf, // [M, kvh·hd]
            p: GBuf, // [B, qh, T, T] softmax probabilities
            o: GBuf, // [M, qh·hd]
            x_mid: GBuf,
            x2: GBuf,
            inv2: GBuf,
            gte: GBuf,
            up: GBuf,
            hh: GBuf,
        },
    }

    /// Scratch for the backward, shared by all layers.
    pub struct Scratch {
        pub dx: GBuf,     // [M,H] the gradient flowing down the residual stream
        pub dx1: GBuf,    // [M,H]
        pub dx2: GBuf,    // [M,H]
        pub dbig: GBuf,   // [M, max(nh·dv, qh·hd)]  do / dq
        pub dk: GBuf,     // [M, max(nh·nph, kvh·hd)]
        pub dk2: GBuf,    // [M, nh·nph]  dthk
        pub dv: GBuf,     // [M, max(nh·dv, kvh·hd)]
        pub dkap: GBuf,   // [M, nh]
        pub dkpre: GBuf,  // [M, kappa_ld]
        pub dstates: GBuf,
        pub dkv: GBuf,    // [M, nh·dv]
        pub dq: GBuf,     // [M, qh·hd]  anchor dQ
        pub dphq: GBuf,   // [M, nh·2nph]
        pub dphk: GBuf,
        pub dp: GBuf,     // [T,T] one score block
        pub dffn: GBuf,   // [M,I] dh
        pub dgte: GBuf,   // [M,I]
        pub dup: GBuf,    // [M,I]
        pub logits: GBuf, // [R, V] head row chunk
        // hybrid_k GEMM-formulation scratch (chunk-major tables + A)
        pub hk_qt: GBuf,
        pub hk_kt: GBuf,
        pub hk_qp: GBuf,
        pub hk_kh: GBuf,
        pub hk_dqt: GBuf,
        pub hk_dkt: GBuf,
        pub hk_dqi: GBuf,
        pub hk_dki: GBuf,
        pub hk_a: GBuf,
        pub loss: GBuf,   // [M]
        pub partial: GBuf,
    }

    pub struct EmbryoGpu {
        pub cfg: EmbryoCfg,
        pub lay: Layout,
        pub b: usize,
        pub t: usize,
        pub p: GBuf,
        pub g: GBuf,
        pub m: GBuf,
        pub v: GBuf,
        pub pow: GBuf,
        pub acts: Vec<LayerActs>,
        pub x_out: GBuf, // [M,H] last layer output (pre final norm)
        pub xf: GBuf,    // [M,H] final-normed
        pub invf: GBuf,
        pub dxf: GBuf,   // [M,H]
        pub tok: GBuf,   // [M] u32 inputs
        pub tgt: GBuf,   // [M] u32 targets
        pub scratch: Scratch,
        pub step: u32,
        /// rows per head chunk (logits [head_rows, V] materialised at a time)
        pub head_rows: usize,
    }

    impl EmbryoGpu {
        pub fn new(cfg: EmbryoCfg, b: usize, t: usize, params: &[f32]) -> Option<EmbryoGpu> {
            let c = ctx()?;
            let lay = Layout::new(&cfg);
            assert_eq!(params.len(), lay.total);
            assert!(t % 64 == 0 && (b * t) % 64 == 0, "B·T and T must be multiples of 64");
            let m = b * t;
            let h = cfg.hidden;
            let z = |n: usize| GBuf::zeros(c, n);
            let nhd = cfg.heads * cfg.dv;
            let nhp = cfg.heads * cfg.nphase;
            let qhd = cfg.anchor_q_heads * cfg.anchor_hd;
            let kvhd = cfg.anchor_kv_heads * cfg.anchor_hd;
            let nst = b * cfg.heads * (t / 64 + 1) * 2 * cfg.nphase * cfg.dv;
            let mut acts = Vec::new();
            for l in 0..cfg.layers {
                if cfg.is_anchor(l) {
                    acts.push(LayerActs::Anchor {
                        x_in: z(m * h),
                        x1: z(m * h),
                        inv1: z(m),
                        q: z(m * qhd),
                        k: z(m * kvhd),
                        v: z(m * kvhd),
                        p: z(b * cfg.anchor_q_heads * t * t),
                        o: z(m * qhd),
                        x_mid: z(m * h),
                        x2: z(m * h),
                        inv2: z(m),
                        gte: z(m * cfg.inter),
                        up: z(m * cfg.inter),
                        hh: z(m * cfg.inter),
                    });
                } else {
                    acts.push(LayerActs::Mixer {
                        x_in: z(m * h),
                        x1: z(m * h),
                        inv1: z(m),
                        thq: z(m * nhp),
                        thk: z(m * nhp),
                        v: z(m * nhd),
                        kpre: z(m * cfg.kappa_ld()),
                        kappa: z(m * cfg.heads),
                        phq: z(m * 2 * nhp),
                        phk: z(m * 2 * nhp),
                        kv: z(m * nhd),
                        states: z(nst),
                        o: z(m * nhd),
                        x_mid: z(m * h),
                        x2: z(m * h),
                        inv2: z(m),
                        gte: z(m * cfg.inter),
                        up: z(m * cfg.inter),
                        hh: z(m * cfg.inter),
                    });
                }
            }
            let head_rows = 1024.min(m);
            assert!(m % head_rows == 0);
            let scratch = Scratch {
                dx: z(m * h),
                dx1: z(m * h),
                dx2: z(m * h),
                dbig: z(m * nhd.max(qhd)),
                dk: z(m * nhp.max(kvhd)),
                dk2: z(m * nhp),
                dv: z(m * nhd.max(kvhd)),
                dkap: z(m * cfg.heads),
                dkpre: z(m * cfg.kappa_ld()),
                dstates: z(nst),
                dkv: z(m * nhd),
                dq: z(m * qhd),
                dphq: z(m * 2 * nhp),
                dphk: z(m * 2 * nhp),
                dp: z(t * t),
                dffn: z(m * cfg.inter),
                dgte: z(m * cfg.inter),
                dup: z(m * cfg.inter),
                logits: z(head_rows * cfg.vocab),
                hk_qt: z(m * 2 * nhp),
                hk_kt: z(m * 2 * nhp),
                hk_qp: z(m * 2 * nhp),
                hk_kh: z(m * 2 * nhp),
                hk_dqt: z(m * 2 * nhp),
                hk_dkt: z(m * 2 * nhp),
                hk_dqi: z(m * 2 * nhp),
                hk_dki: z(m * 2 * nhp),
                hk_a: z(b * cfg.heads * (t / 64) * 4096),
                loss: z(m),
                partial: z(4096),
            };
            let decay = hk_decay_grid(cfg.heads, cfg.nphase, cfg.horizon_min, cfg.horizon_max);
            let pow = GBuf::from_slice(c, &hk_pow_table(&decay, cfg.heads, cfg.nphase));
            Some(EmbryoGpu {
                p: GBuf::from_slice(c, params),
                g: z(lay.total),
                m: z(lay.total),
                v: z(lay.total),
                pow,
                acts,
                x_out: z(m * h),
                xf: z(m * h),
                invf: z(m),
                dxf: z(m * h),
                tok: GBuf::from_u32(c, &vec![0u32; m]),
                tgt: GBuf::from_u32(c, &vec![0u32; m]),
                scratch,
                step: 0,
                head_rows,
                cfg,
                lay,
                b,
                t,
            })
        }

        pub fn hk_scratch(&self) -> HkScratch<'_> {
            let s = &self.scratch;
            HkScratch { qt: &s.hk_qt, kt: &s.hk_kt, qp: &s.hk_qp, kh: &s.hk_kh, dqt: &s.hk_dqt, dkt: &s.hk_dkt, dqi: &s.hk_dqi, dki: &s.hk_dki, a: &s.hk_a }
        }

        pub fn ctx(&self) -> &'static Ctx {
            ctx().expect("Metal context")
        }

        fn ffn_fwd(&self, cmd: &Cmd, m: usize, ffn: &FfnOffs, x2: &GBuf, gte: &GBuf, up: &GBuf, hh: &GBuf, x_mid: &GBuf, x_out: &GBuf) {
            let (h, i) = (self.cfg.hidden, self.cfg.inter);
            // gate/up: [M,H]·[I,H]ᵀ
            cmd.gemm(Op::N, Op::T, m, i, h, 1.0, x2, 0, h, &self.p, ffn.wg, h, 0.0, gte, 0, i);
            cmd.gemm(Op::N, Op::T, m, i, h, 1.0, x2, 0, h, &self.p, ffn.wu, h, 0.0, up, 0, i);
            cmd.swiglu_fwd(gte, up, hh, m * i);
            // x_out = x_mid + hh·Wdᵀ  ([M,I]·[H,I]ᵀ)
            cmd.copy(x_mid, 0, x_out, 0, m * h);
            cmd.gemm(Op::N, Op::T, m, h, i, 1.0, hh, 0, i, &self.p, ffn.wd, i, 1.0, x_out, 0, h);
        }

        /// FFN backward: dx_out (in s.dx) → accumulates dx_mid into s.dx via
        /// the ln2 backward; weight grads into g.
        #[allow(clippy::too_many_arguments)]
        fn ffn_bwd(&self, cmd: &Cmd, m: usize, ffn: &FfnOffs, ln2: usize, x_mid: &GBuf, x2: &GBuf, inv2: &GBuf, gte: &GBuf, up: &GBuf, hh: &GBuf) {
            let s = &self.scratch;
            let (h, i) = (self.cfg.hidden, self.cfg.inter);
            // dhh = dx·Wd  ([M,H]·[H,I])
            cmd.gemm(Op::N, Op::N, m, i, h, 1.0, &s.dx, 0, h, &self.p, ffn.wd, i, 0.0, &s.dffn, 0, i);
            // dWd += dxᵀ·hh  ([H,M]·[M,I])
            cmd.gemm(Op::T, Op::N, h, i, m, 1.0, &s.dx, 0, h, hh, 0, i, 1.0, &self.g, ffn.wd, i);
            cmd.swiglu_bwd(gte, up, &s.dffn, &s.dgte, &s.dup, m * i);
            // dx2 = dgte·Wg + dup·Wu
            cmd.gemm(Op::N, Op::N, m, h, i, 1.0, &s.dgte, 0, i, &self.p, ffn.wg, h, 0.0, &s.dx2, 0, h);
            cmd.gemm(Op::N, Op::N, m, h, i, 1.0, &s.dup, 0, i, &self.p, ffn.wu, h, 1.0, &s.dx2, 0, h);
            // dWg += dgteᵀ·x2 ; dWu += dupᵀ·x2
            cmd.gemm(Op::T, Op::N, i, h, m, 1.0, &s.dgte, 0, i, x2, 0, h, 1.0, &self.g, ffn.wg, h);
            cmd.gemm(Op::T, Op::N, i, h, m, 1.0, &s.dup, 0, i, x2, 0, h, 1.0, &self.g, ffn.wu, h);
            // dx_mid = dx_out + rmsnorm_bwd(x_mid; dx2)  (accumulate into s.dx)
            cmd.rmsnorm_bwd_at(x_mid, &self.p, ln2, &s.dx2, inv2, &s.dx, 1.0, &self.g, ln2, m, h);
        }

        /// Forward through layer `l` from acts[l].x_in into `x_out`.
        pub(crate) fn layer_fwd(&self, cmd: &Cmd, l: usize, x_out: &GBuf) {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            match (&self.lay.layers[l], &self.acts[l]) {
                (
                    LayerOffs::Mixer { ln1, wq, wk, wv, wkap, wo, ln2, ffn },
                    LayerActs::Mixer { x_in, x1, inv1, thq, thk, v, kpre, kappa, phq, phk, kv, states, o, x_mid, x2, inv2, gte, up, hh },
                ) => {
                    let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
                    cmd.rmsnorm_fwd_at(x_in, &self.p, *ln1, x1, inv1, m, h, cfg.norm_eps);
                    cmd.gemm(Op::N, Op::T, m, nh * nph, h, 1.0, x1, 0, h, &self.p, *wq, h, 0.0, thq, 0, nh * nph);
                    cmd.gemm(Op::N, Op::T, m, nh * nph, h, 1.0, x1, 0, h, &self.p, *wk, h, 0.0, thk, 0, nh * nph);
                    cmd.gemm(Op::N, Op::T, m, nh * dv, h, 1.0, x1, 0, h, &self.p, *wv, h, 0.0, v, 0, nh * dv);
                    let kld = cfg.kappa_ld();
                    cmd.gemm(Op::N, Op::T, m, kld, h, 1.0, x1, 0, h, &self.p, *wkap, h, 0.0, kpre, 0, kld);
                    cmd.kappa_fwd(kpre, kappa, m, nh, kld, cfg.kappa_bias);
                    let d = HkDims { b, t, nh, nph, dv };
                    let w = HkWork { thq, thk, v, kappa, pow: &self.pow, phq, phk, kv, states, out: o };
                    if hk_simt() { cmd.hk_forward(&d, &w) } else { cmd.hk_forward_gemm(&d, &w, &self.hk_scratch()) }
                    // x_mid = x_in + o·Woᵀ   ([M, nh·dv]·[H, nh·dv]ᵀ)
                    cmd.copy(x_in, 0, x_mid, 0, m * h);
                    cmd.gemm(Op::N, Op::T, m, h, nh * dv, 1.0, o, 0, nh * dv, &self.p, *wo, nh * dv, 1.0, x_mid, 0, h);
                    cmd.rmsnorm_fwd_at(x_mid, &self.p, *ln2, x2, inv2, m, h, cfg.norm_eps);
                    self.ffn_fwd(cmd, m, ffn, x2, gte, up, hh, x_mid, x_out);
                }
                (
                    LayerOffs::Anchor { ln1, wq, wk, wv, wo, ln2, ffn },
                    LayerActs::Anchor { x_in, x1, inv1, q, k, v, p, o, x_mid, x2, inv2, gte, up, hh },
                ) => {
                    let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
                    let (qd, kd) = (qh * hd, kvh * hd);
                    cmd.rmsnorm_fwd_at(x_in, &self.p, *ln1, x1, inv1, m, h, cfg.norm_eps);
                    cmd.gemm(Op::N, Op::T, m, qd, h, 1.0, x1, 0, h, &self.p, *wq, h, 0.0, q, 0, qd);
                    cmd.gemm(Op::N, Op::T, m, kd, h, 1.0, x1, 0, h, &self.p, *wk, h, 0.0, k, 0, kd);
                    cmd.gemm(Op::N, Op::T, m, kd, h, 1.0, x1, 0, h, &self.p, *wv, h, 0.0, v, 0, kd);
                    cmd.rope(q, 0, m, t, qh, hd, cfg.rope_base, false);
                    cmd.rope(k, 0, m, t, kvh, hd, cfg.rope_base, false);
                    let scale = 1.0 / (hd as f32).sqrt();
                    let group = qh / kvh;
                    for bi in 0..b {
                        for i in 0..qh {
                            let g = i / group;
                            let p_off = (bi * qh + i) * t * t;
                            // S = Q_i·K_gᵀ·scale
                            cmd.gemm(Op::N, Op::T, t, t, hd, scale, q, bi * t * qd + i * hd, qd, k, bi * t * kd + g * hd, kd, 0.0, p, p_off, t);
                            cmd.causal_softmax(p, p_off, t);
                            // O_i = P·V_g
                            cmd.gemm(Op::N, Op::N, t, hd, t, 1.0, p, p_off, t, v, bi * t * kd + g * hd, kd, 0.0, o, bi * t * qd + i * hd, qd);
                        }
                    }
                    cmd.copy(x_in, 0, x_mid, 0, m * h);
                    cmd.gemm(Op::N, Op::T, m, h, qd, 1.0, o, 0, qd, &self.p, *wo, qd, 1.0, x_mid, 0, h);
                    cmd.rmsnorm_fwd_at(x_mid, &self.p, *ln2, x2, inv2, m, h, cfg.norm_eps);
                    self.ffn_fwd(cmd, m, ffn, x2, gte, up, hh, x_mid, x_out);
                }
                _ => unreachable!("layout/acts kind mismatch"),
            }
        }

        /// Backward through layer `l`: s.dx holds dL/dx_out on entry and
        /// dL/dx_in on exit.
        pub(crate) fn layer_bwd(&self, cmd: &Cmd, l: usize) {
            let cfg = &self.cfg;
            let (b, t, h) = (self.b, self.t, cfg.hidden);
            let m = b * t;
            let s = &self.scratch;
            match (&self.lay.layers[l], &self.acts[l]) {
                (
                    LayerOffs::Mixer { ln1, wq, wk, wv, wkap, wo, ln2, ffn },
                    LayerActs::Mixer { x_in, x1, inv1, thq, thk, v, kpre: _, kappa, phq, phk, kv, states, o, x_mid, x2, inv2, gte, up, hh },
                ) => {
                    let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
                    self.ffn_bwd(cmd, m, ffn, *ln2, x_mid, x2, inv2, gte, up, hh);
                    // do = dx_mid·Wo  ([M,H]·[H, nh·dv]);  dWo += dx_midᵀ·o
                    cmd.gemm(Op::N, Op::N, m, nh * dv, h, 1.0, &s.dx, 0, h, &self.p, *wo, nh * dv, 0.0, &s.dbig, 0, nh * dv);
                    cmd.gemm(Op::T, Op::N, h, nh * dv, m, 1.0, &s.dx, 0, h, o, 0, nh * dv, 1.0, &self.g, *wo, nh * dv);
                    let d = HkDims { b, t, nh, nph, dv };
                    let w = HkWork { thq, thk, v, kappa, pow: &self.pow, phq, phk, kv, states, out: o };
                    let gr = HkGrads {
                        dout: &s.dbig,
                        dstates: &s.dstates,
                        dkv: &s.dkv,
                        dphq: &s.dphq,
                        dphk: &s.dphk,
                        dthq: &s.dk,
                        dthk: &s.dk2,
                        dv: &s.dv,
                        dkappa: &s.dkap,
                    };
                    if hk_simt() { cmd.hk_backward(&d, &w, &gr, 0.0) } else { cmd.hk_backward_gemm(&d, &w, &gr, &self.hk_scratch(), 0.0) }
                    let kld = cfg.kappa_ld();
                    cmd.kappa_bwd(kappa, &s.dkap, &s.dkpre, m, nh, kld);
                    // dx1 = dthq·Wq + dthk·Wk + dv·Wv + dkpre·Wκ
                    cmd.gemm(Op::N, Op::N, m, h, nh * nph, 1.0, &s.dk, 0, nh * nph, &self.p, *wq, h, 0.0, &s.dx1, 0, h);
                    cmd.gemm(Op::N, Op::N, m, h, nh * nph, 1.0, &s.dk2, 0, nh * nph, &self.p, *wk, h, 1.0, &s.dx1, 0, h);
                    cmd.gemm(Op::N, Op::N, m, h, nh * dv, 1.0, &s.dv, 0, nh * dv, &self.p, *wv, h, 1.0, &s.dx1, 0, h);
                    cmd.gemm(Op::N, Op::N, m, h, kld, 1.0, &s.dkpre, 0, kld, &self.p, *wkap, h, 1.0, &s.dx1, 0, h);
                    // weight grads
                    cmd.gemm(Op::T, Op::N, nh * nph, h, m, 1.0, &s.dk, 0, nh * nph, x1, 0, h, 1.0, &self.g, *wq, h);
                    cmd.gemm(Op::T, Op::N, nh * nph, h, m, 1.0, &s.dk2, 0, nh * nph, x1, 0, h, 1.0, &self.g, *wk, h);
                    cmd.gemm(Op::T, Op::N, nh * dv, h, m, 1.0, &s.dv, 0, nh * dv, x1, 0, h, 1.0, &self.g, *wv, h);
                    cmd.gemm(Op::T, Op::N, kld, h, m, 1.0, &s.dkpre, 0, kld, x1, 0, h, 1.0, &self.g, *wkap, h);
                    // dx_in = dx_mid + rmsnorm_bwd(x_in; dx1)
                    cmd.rmsnorm_bwd_at(x_in, &self.p, *ln1, &s.dx1, inv1, &s.dx, 1.0, &self.g, *ln1, m, h);
                }
                (
                    LayerOffs::Anchor { ln1, wq, wk, wv, wo, ln2, ffn },
                    LayerActs::Anchor { x_in, x1, inv1, q, k, v, p, o, x_mid, x2, inv2, gte, up, hh },
                ) => {
                    let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
                    let (qd, kd) = (qh * hd, kvh * hd);
                    self.ffn_bwd(cmd, m, ffn, *ln2, x_mid, x2, inv2, gte, up, hh);
                    // do = dx_mid·Wo ; dWo += dx_midᵀ·o
                    cmd.gemm(Op::N, Op::N, m, qd, h, 1.0, &s.dx, 0, h, &self.p, *wo, qd, 0.0, &s.dbig, 0, qd);
                    cmd.gemm(Op::T, Op::N, h, qd, m, 1.0, &s.dx, 0, h, o, 0, qd, 1.0, &self.g, *wo, qd);
                    // attention backward per (b, head); dQ in s.dq, dK/dV in s.dk/s.dv
                    let dq = &s.dq;
                    assert!(dq.len >= m * qd && s.dk.len >= m * kd && s.dv.len >= m * kd);
                    let scale = 1.0 / (hd as f32).sqrt();
                    let group = qh / kvh;
                    // dK, dV accumulate over the heads of a group: zero first
                    cmd.axpby(0.0, &s.dk, 0.0, &s.dk, m * kd);
                    cmd.axpby(0.0, &s.dv, 0.0, &s.dv, m * kd);
                    for bi in 0..b {
                        for i in 0..qh {
                            let g = i / group;
                            let p_off = (bi * qh + i) * t * t;
                            let do_off = bi * t * qd + i * hd;
                            let kv_off = bi * t * kd + g * hd;
                            // dP = dO_i·V_gᵀ
                            cmd.gemm(Op::N, Op::T, t, t, hd, 1.0, &s.dbig, do_off, qd, v, kv_off, kd, 0.0, &s.dp, 0, t);
                            // dV_g += P_iᵀ·dO_i
                            cmd.gemm(Op::T, Op::N, t, hd, t, 1.0, p, p_off, t, &s.dbig, do_off, qd, 1.0, &s.dv, kv_off, kd);
                            // dS = P⊙(dP − rowsum)
                            cmd.softmax_bwd(p, p_off, &s.dp, 0, t);
                            // dQ_i = dS·K_g·scale
                            cmd.gemm(Op::N, Op::N, t, hd, t, scale, &s.dp, 0, t, k, kv_off, kd, 0.0, dq, do_off, qd);
                            // dK_g += dSᵀ·Q_i·scale
                            cmd.gemm(Op::T, Op::N, t, hd, t, scale, &s.dp, 0, t, q, do_off, qd, 1.0, &s.dk, kv_off, kd);
                        }
                    }
                    cmd.rope(dq, 0, m, t, qh, hd, cfg.rope_base, true);
                    cmd.rope(&s.dk, 0, m, t, kvh, hd, cfg.rope_base, true);
                    // dx1 = dq·Wq + dk·Wk + dv·Wv
                    cmd.gemm(Op::N, Op::N, m, h, qd, 1.0, dq, 0, qd, &self.p, *wq, h, 0.0, &s.dx1, 0, h);
                    cmd.gemm(Op::N, Op::N, m, h, kd, 1.0, &s.dk, 0, kd, &self.p, *wk, h, 1.0, &s.dx1, 0, h);
                    cmd.gemm(Op::N, Op::N, m, h, kd, 1.0, &s.dv, 0, kd, &self.p, *wv, h, 1.0, &s.dx1, 0, h);
                    cmd.gemm(Op::T, Op::N, qd, h, m, 1.0, dq, 0, qd, x1, 0, h, 1.0, &self.g, *wq, h);
                    cmd.gemm(Op::T, Op::N, kd, h, m, 1.0, &s.dk, 0, kd, x1, 0, h, 1.0, &self.g, *wk, h);
                    cmd.gemm(Op::T, Op::N, kd, h, m, 1.0, &s.dv, 0, kd, x1, 0, h, 1.0, &self.g, *wv, h);
                    cmd.rmsnorm_bwd_at(x_in, &self.p, *ln1, &s.dx1, inv1, &s.dx, 1.0, &self.g, *ln1, m, h);
                }
                _ => unreachable!(),
            }
        }

        pub(crate) fn x_in(&self, l: usize) -> &GBuf {
            match &self.acts[l] {
                LayerActs::Mixer { x_in, .. } | LayerActs::Anchor { x_in, .. } => x_in,
            }
        }

        /// Encode one full training step's forward + backward (grads into
        /// `g`, zeroed first) + the grad-norm partials. Tokens/targets must
        /// already be in `tok`/`tgt`. Loss per position lands in scratch.loss.
        pub fn encode_fwd_bwd(&self, cmd: &Cmd) -> usize {
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, self.b * self.t);
            let s = &self.scratch;
            cmd.axpby(0.0, &self.g, 0.0, &self.g, self.lay.total);
            // embed
            cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
            for l in 0..cfg.layers {
                let out: &GBuf = if l + 1 < cfg.layers { self.x_in(l + 1) } else { &self.x_out };
                self.layer_fwd(cmd, l, out);
            }
            cmd.rmsnorm_fwd_at(&self.x_out, &self.p, self.lay.final_norm, &self.xf, &self.invf, m, h, cfg.norm_eps);
            // head (tied, full softmax) in row chunks: loss + dxf + dE
            let r = self.head_rows;
            let scale = 1.0 / m as f32;
            for c0 in (0..m).step_by(r) {
                cmd.gemm(Op::N, Op::T, r, cfg.vocab, h, 1.0, &self.xf, c0 * h, h, &self.p, self.lay.embed, h, 0.0, &s.logits, 0, cfg.vocab);
                cmd.softmax_ce_at(&s.logits, 0, &self.tgt, c0, &s.loss, c0, r, cfg.vocab, scale);
                // dxf_chunk = dlogits·E ; dE += dlogitsᵀ·xf_chunk
                cmd.gemm(Op::N, Op::N, r, h, cfg.vocab, 1.0, &s.logits, 0, cfg.vocab, &self.p, self.lay.embed, h, 0.0, &self.dxf, c0 * h, h);
                cmd.gemm(Op::T, Op::N, cfg.vocab, h, r, 1.0, &s.logits, 0, cfg.vocab, &self.xf, c0 * h, h, 1.0, &self.g, self.lay.embed, h);
            }
            // final norm backward → s.dx
            cmd.rmsnorm_bwd_at(&self.x_out, &self.p, self.lay.final_norm, &self.dxf, &self.invf, &s.dx, 0.0, &self.g, self.lay.final_norm, m, h);
            for l in (0..cfg.layers).rev() {
                self.layer_bwd(cmd, l);
            }
            // embedding backward (tied: adds to the same rows the head wrote)
            cmd.embed_scatter_add(&self.g, self.lay.embed, &self.tok, &s.dx, m, h);
            cmd.sumsq(&self.g, self.lay.total, &s.partial)
        }

        /// One optimiser step: AdamW over the arena with global-norm clip.
        /// `gnorm` is the pre-clip gradient norm (already computed).
        pub fn encode_adamw(&self, cmd: &Cmd, lr: f32, wd: f32, clip: f32, gnorm: f32, step: u32) {
            let gscale = if gnorm > clip { clip / gnorm } else { 1.0 };
            cmd.adamw(&self.p, &self.g, &self.m, &self.v, self.lay.total, lr, 0.9, 0.95, 1e-8, wd, step, gscale);
        }

        /// Full step: upload batch, fwd+bwd, clip, AdamW. Returns
        /// (mean loss, grad norm, gpu ms).
        pub fn train_step(&mut self, tokens: &[u32], targets: &[u32], lr: f32, wd: f32, clip: f32) -> (f32, f32, f64) {
            let m = self.b * self.t;
            assert!(tokens.len() == m && targets.len() == m);
            let c = self.ctx();
            unsafe {
                std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tok.buf.contents() as *mut u32, m);
                std::ptr::copy_nonoverlapping(targets.as_ptr(), self.tgt.buf.contents() as *mut u32, m);
            }
            let cmd = Cmd::new(c);
            let groups = self.encode_fwd_bwd(&cmd);
            let ms1 = cmd.commit();
            let gnorm = self.scratch.partial.as_slice()[..groups].iter().map(|x| *x as f64).sum::<f64>().sqrt() as f32;
            let loss = (self.scratch.loss.as_slice()[..m].iter().map(|x| *x as f64).sum::<f64>() / m as f64) as f32;
            self.step += 1;
            let cmd = Cmd::new(c);
            self.encode_adamw(&cmd, lr, wd, clip, gnorm, self.step);
            let ms2 = cmd.commit();
            (loss, gnorm, ms1 + ms2)
        }

        /// Forward only (no grads): mean loss on a batch already in tok/tgt.
        pub fn eval_loss(&self, tokens: &[u32], targets: &[u32]) -> f32 {
            let m = self.b * self.t;
            unsafe {
                std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tok.buf.contents() as *mut u32, m);
                std::ptr::copy_nonoverlapping(targets.as_ptr(), self.tgt.buf.contents() as *mut u32, m);
            }
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, m);
            let s = &self.scratch;
            let cmd = Cmd::new(self.ctx());
            cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
            for l in 0..cfg.layers {
                let out: &GBuf = if l + 1 < cfg.layers { self.x_in(l + 1) } else { &self.x_out };
                self.layer_fwd(&cmd, l, out);
            }
            cmd.rmsnorm_fwd_at(&self.x_out, &self.p, self.lay.final_norm, &self.xf, &self.invf, m, h, cfg.norm_eps);
            let r = self.head_rows;
            for c0 in (0..m).step_by(r) {
                cmd.gemm(Op::N, Op::T, r, cfg.vocab, h, 1.0, &self.xf, c0 * h, h, &self.p, self.lay.embed, h, 0.0, &s.logits, 0, cfg.vocab);
                cmd.softmax_ce_at(&s.logits, 0, &self.tgt, c0, &s.loss, c0, r, cfg.vocab, 1.0 / m as f32);
            }
            cmd.commit();
            (self.scratch.loss.as_slice()[..m].iter().map(|x| *x as f64).sum::<f64>() / m as f64) as f32
        }

        pub fn params_host(&self) -> Vec<f32> {
            self.p.to_vec()
        }
        pub fn grads_host(&self) -> Vec<f32> {
            self.g.to_vec()
        }
        pub fn set_params(&self, p: &[f32]) {
            self.p.write_from(p);
        }
    }
}

#[cfg(target_os = "macos")]
impl EmbryoGpu {
    /// Per-phase GPU time of one step (each phase its own command buffer):
    /// the profile the kernel work is prioritised from.
    pub fn profile_step(&self) -> Vec<(String, f64)> {
        use crate::metal::{Cmd, Op};
        let cfg = &self.cfg;
        let (h, m) = (cfg.hidden, self.b * self.t);
        let s = &self.scratch;
        let c = self.ctx();
        let mut out = Vec::new();
        let time = |name: String, f: &dyn Fn(&Cmd)| -> f64 {
            let cmd = Cmd::new(c);
            f(&cmd);
            cmd.commit()
        };
        out.push(("zero grads".into(), time("z".into(), &|cmd| cmd.axpby(0.0, &self.g, 0.0, &self.g, self.lay.total))));
        out.push(("embed".into(), time("e".into(), &|cmd| cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h))));
        for l in 0..cfg.layers {
            let ms = time(format!("fwd {l}"), &|cmd| {
                let o: &crate::metal::GBuf = if l + 1 < cfg.layers { self.x_in(l + 1) } else { &self.x_out };
                self.layer_fwd(cmd, l, o);
            });
            out.push((format!("layer {l} fwd{}", if cfg.is_anchor(l) { " (anchor)" } else { "" }), ms));
        }
        out.push(("final norm".into(), time("f".into(), &|cmd| cmd.rmsnorm_fwd_at(&self.x_out, &self.p, self.lay.final_norm, &self.xf, &self.invf, m, h, cfg.norm_eps))));
        let r = self.head_rows;
        out.push(("head fwd+bwd".into(), time("h".into(), &|cmd| {
            for c0 in (0..m).step_by(r) {
                cmd.gemm(Op::N, Op::T, r, cfg.vocab, h, 1.0, &self.xf, c0 * h, h, &self.p, self.lay.embed, h, 0.0, &s.logits, 0, cfg.vocab);
                cmd.softmax_ce_at(&s.logits, 0, &self.tgt, c0, &s.loss, c0, r, cfg.vocab, 1.0 / m as f32);
                cmd.gemm(Op::N, Op::N, r, h, cfg.vocab, 1.0, &s.logits, 0, cfg.vocab, &self.p, self.lay.embed, h, 0.0, &self.dxf, c0 * h, h);
                cmd.gemm(Op::T, Op::N, cfg.vocab, h, r, 1.0, &s.logits, 0, cfg.vocab, &self.xf, c0 * h, h, 1.0, &self.g, self.lay.embed, h);
            }
        })));
        out.push(("final norm bwd".into(), time("fb".into(), &|cmd| cmd.rmsnorm_bwd_at(&self.x_out, &self.p, self.lay.final_norm, &self.dxf, &self.invf, &s.dx, 0.0, &self.g, self.lay.final_norm, m, h))));
        for l in (0..cfg.layers).rev() {
            let ms = time(format!("bwd {l}"), &|cmd| self.layer_bwd(cmd, l));
            out.push((format!("layer {l} bwd{}", if cfg.is_anchor(l) { " (anchor)" } else { "" }), ms));
        }
        out.push(("embed bwd".into(), time("eb".into(), &|cmd| cmd.embed_scatter_add(&self.g, self.lay.embed, &self.tok, &s.dx, m, h))));
        out.push(("grad norm".into(), time("gn".into(), &|cmd| { cmd.sumsq(&self.g, self.lay.total, &s.partial); })));
        out.push(("adamw".into(), time("a".into(), &|cmd| self.encode_adamw(cmd, 1e-4, 0.1, 1.0, 1.0, 1))));
        out
    }

    /// Per-kernel profile of one hybrid_k mixer layer's forward + backward
    /// (kernel by kernel, each its own command buffer).
    pub fn profile_hk_layer(&self, l: usize) -> Vec<(String, f64)> {
        use crate::metal::{Cmd, HkDims, HkGrads, HkWork};
        let cfg = &self.cfg;
        let (b, t) = (self.b, self.t);
        let s = &self.scratch;
        let c = self.ctx();
        let LayerActs::Mixer { thq, thk, v, kappa, phq, phk, kv, states, o, .. } = &self.acts[l] else { return vec![] };
        let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
        let d = HkDims { b, t, nh, nph, dv };
        let w = HkWork { thq, thk, v, kappa, pow: &self.pow, phq, phk, kv, states, out: o };
        let gr = HkGrads { dout: &s.dbig, dstates: &s.dstates, dkv: &s.dkv, dphq: &s.dphq, dphk: &s.dphk, dthq: &s.dk, dthk: &s.dk2, dv: &s.dv, dkappa: &s.dkap };
        let mut out = Vec::new();
        let cmd = Cmd::new(c);
        cmd.hk_forward(&d, &w);
        out.push(("hk forward SIMT (φ, kv, states, chunks)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_backward(&d, &w, &gr, 0.0);
        out.push(("hk backward SIMT (dstates, chunks, split, dθ)".into(), cmd.commit()));
        let sc = self.hk_scratch();
        let cmd = Cmd::new(c);
        cmd.hk_forward_gemm(&d, &w, &sc);
        out.push(("hk forward GEMM".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_backward_gemm(&d, &w, &gr, &sc, 0.0);
        out.push(("hk backward GEMM".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_states_only(&d, &w);
        out.push(("  states scan SIMT (fwd)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_dstates_only(&d, &w, &gr);
        out.push(("  dstates scan SIMT (bwd)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_states_par(&d, &w);
        out.push(("  states scan cell-parallel (fwd)".into(), cmd.commit()));
        let cmd = Cmd::new(c);
        cmd.hk_dstates_par(&d, &w, &gr);
        out.push(("  dstates scan cell-parallel (bwd)".into(), cmd.commit()));
        out
    }
}

/// `EMBRYO_HK_SIMT=1` selects the reference SIMT chunk kernels instead of
/// the batched-GEMM formulation (A/B and debugging).
#[cfg(target_os = "macos")]
fn hk_simt() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("EMBRYO_HK_SIMT").is_ok_and(|v| v != "0"))
}
