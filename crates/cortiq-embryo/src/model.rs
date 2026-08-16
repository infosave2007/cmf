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
            vocab: 4096,
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
            head_clusters: 64,
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
    /// [head_clusters, H] cluster embeddings of the hierarchical head
    /// (usize::MAX when the head is flat)
    pub head_clusters: usize,
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
        let head_clusters =
            if cfg.head_clusters > 0 { take("head.clusters".into(), cfg.head_clusters * h) } else { usize::MAX };
        Layout { total: off, embed, final_norm, head_clusters, layers, names }
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
    if cfg.head_clusters > 0 {
        fill(&mut p, lay.head_clusters, cfg.head_clusters * h, std);
    }
    p
}

#[cfg(target_os = "macos")]
pub use gpu::*;

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::metal::{Cmd, Ctx, GBuf, GemmBatch, HkDims, HkGrads, HkScratch, HkWork, Op, ctx, hk_pow_table};
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
        pub dp: GBuf,     // [qh, T, T] one sequence's score blocks
        pub dkh: GBuf,    // [B, qh, T, hd] per-head dK partials
        pub dvh: GBuf,    // [B, qh, T, hd]
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
        // hierarchical head (cfg.head_clusters > 0)
        pub tgt_cluster: GBuf, // [M] u32 target cluster ids
        pub head_idx: GBuf,    // [Mpad] i32 grouped row → token index (−1 pad)
        /// (cluster, row offset, padded rows) of the grouped rows
        pub head_groups: std::cell::RefCell<Vec<(usize, usize, usize)>>,
        pub hg: GBuf,    // [Mpad, H] gathered rows
        pub dhg: GBuf,   // [Mpad, H]
        pub lw: GBuf,    // [Mpad, S] within-cluster logits
        pub lc: GBuf,    // [M, C] cluster logits
        pub loss2: GBuf, // [M]
        pub mpad: usize,
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
            let ncl = cfg.head_clusters;
            let mpad = m + ncl * 64;
            let cs = if ncl > 0 { cfg.vocab / ncl } else { 0 };
            if ncl > 0 {
                assert!(
                    cfg.vocab % ncl == 0 && cs % 64 == 0 && ncl % 64 == 0,
                    "hierarchical head: vocab = C·S with C, S multiples of 64"
                );
            }
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
                dp: z(cfg.anchor_q_heads * t * t),
                dkh: z(m * qhd),
                dvh: z(m * qhd),
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
                tgt_cluster: GBuf::from_u32(c, &vec![0u32; m]),
                head_idx: GBuf::from_u32(c, &vec![u32::MAX; mpad.max(1)]),
                head_groups: std::cell::RefCell::new(Vec::new()),
                hg: z(mpad * h),
                dhg: z(mpad * h),
                lw: z(mpad * cs.max(1)),
                lc: z(m * ncl.max(1)),
                loss2: z(m),
                mpad,
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
                    // batched over z = (b, kv-group g, head-in-group j), head i = g·group + j
                    let sq = [t * qd, group * hd, hd]; // q / o column blocks
                    let sk = [t * kd, hd, 0]; // k / v (shared across the group)
                    let sp = [qh * t * t, group * t * t, t * t]; // P blocks
                    let bt = GemmBatch { nb: b, nh: kvh, nc: group, sa: sq, sb: sk, sc: sp };
                    // S = Q_i·K_gᵀ·scale (all heads, all sequences)
                    cmd.gemm_ex(Op::N, Op::T, t, t, hd, scale, q, 0, qd, k, 0, kd, 0.0, p, 0, t, &bt, false);
                    cmd.causal_softmax_blocks(p, 0, t, b * qh);
                    // O_i = P·V_g
                    let bt = GemmBatch { nb: b, nh: kvh, nc: group, sa: sp, sb: sk, sc: sq };
                    cmd.gemm_ex(Op::N, Op::N, t, hd, t, 1.0, p, 0, t, v, 0, kd, 0.0, o, 0, qd, &bt, false);
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
                    // per sequence b (dP scratch holds one sequence's qh blocks), batched
                    // over (g, j); per-head dK/dV partials land head-major in s.dkh/s.dvh
                    // and are group-summed afterwards (no accumulation races).
                    let sq = [0, group * hd, hd];
                    let sk = [0, hd, 0];
                    let sp = [0, group * t * t, t * t];
                    let sh = [0, group * t * hd, t * hd]; // head-major [qh][T][hd] within one b
                    for bi in 0..b {
                        let q_off = bi * t * qd;
                        let kv_off = bi * t * kd;
                        let p_off = bi * qh * t * t;
                        let h_off = bi * qh * t * hd;
                        // dP = dO_i·V_gᵀ
                        let bt = GemmBatch { nb: 1, nh: kvh, nc: group, sa: sq, sb: sk, sc: sp };
                        cmd.gemm_ex(Op::N, Op::T, t, t, hd, 1.0, &s.dbig, q_off, qd, v, kv_off, kd, 0.0, &s.dp, 0, t, &bt, false);
                        // dV_i(partial) = P_iᵀ·dO_i
                        let bt = GemmBatch { nb: 1, nh: kvh, nc: group, sa: sp, sb: sq, sc: sh };
                        cmd.gemm_ex(Op::T, Op::N, t, hd, t, 1.0, p, p_off, t, &s.dbig, q_off, qd, 0.0, &s.dvh, h_off, hd, &bt, false);
                        // dS = P⊙(dP − rowsum)
                        cmd.softmax_bwd_blocks(p, p_off, &s.dp, 0, t, qh);
                        // dQ_i = dS·K_g·scale
                        let bt = GemmBatch { nb: 1, nh: kvh, nc: group, sa: sp, sb: sk, sc: sq };
                        cmd.gemm_ex(Op::N, Op::N, t, hd, t, scale, &s.dp, 0, t, k, kv_off, kd, 0.0, dq, q_off, qd, &bt, false);
                        // dK_i(partial) = dSᵀ·Q_i·scale
                        let bt = GemmBatch { nb: 1, nh: kvh, nc: group, sa: sp, sb: sq, sc: sh };
                        cmd.gemm_ex(Op::T, Op::N, t, hd, t, scale, &s.dp, 0, t, q, q_off, qd, 0.0, &s.dkh, h_off, hd, &bt, false);
                    }
                    cmd.group_sum_heads(&s.dkh, &s.dk, b, t, qh, kvh, hd);
                    cmd.group_sum_heads(&s.dvh, &s.dv, b, t, qh, kvh, hd);
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


        /// Host-side prep of the hierarchical head for the batch already in
        /// `tgt`: target cluster ids and the rows grouped by cluster (each
        /// group padded to a multiple of 64 with −1). No-op for the flat head.
        pub fn prepare_head(&self, targets: &[u32]) {
            let ncl = self.cfg.head_clusters;
            if ncl == 0 {
                return;
            }
            let m = self.b * self.t;
            let cs = self.cfg.vocab / ncl;
            let mut tc = vec![0u32; m];
            let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); ncl];
            for (i, &t) in targets.iter().enumerate() {
                let c = (t as usize / cs).min(ncl - 1);
                tc[i] = c as u32;
                buckets[c].push(i as i32);
            }
            let mut idx: Vec<i32> = Vec::with_capacity(self.mpad);
            let mut groups = Vec::new();
            for (c, bk) in buckets.iter().enumerate() {
                if bk.is_empty() {
                    continue;
                }
                let off = idx.len();
                idx.extend_from_slice(bk);
                let pad = bk.len().div_ceil(64) * 64;
                idx.resize(off + pad, -1);
                groups.push((c, off, pad));
            }
            assert!(idx.len() <= self.mpad);
            unsafe {
                std::ptr::copy_nonoverlapping(tc.as_ptr(), self.tgt_cluster.buf.contents() as *mut u32, m);
                std::ptr::copy_nonoverlapping(idx.as_ptr(), self.head_idx.buf.contents() as *mut i32, idx.len());
            }
            *self.head_groups.borrow_mut() = groups;
        }

        /// Head: loss (+ dxf and dE/dC when `train`). Flat full-softmax or
        /// hierarchical (cluster CE + within-target-cluster CE), tied to E.
        pub(crate) fn encode_head(&self, cmd: &Cmd, train: bool) {
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, self.b * self.t);
            let s = &self.scratch;
            let scale = 1.0 / m as f32;
            let ncl = cfg.head_clusters;
            if ncl == 0 {
                let r = self.head_rows;
                for c0 in (0..m).step_by(r) {
                    cmd.gemm(Op::N, Op::T, r, cfg.vocab, h, 1.0, &self.xf, c0 * h, h, &self.p, self.lay.embed, h, 0.0, &s.logits, 0, cfg.vocab);
                    cmd.softmax_ce_at(&s.logits, 0, &self.tgt, c0, &s.loss, c0, r, cfg.vocab, scale);
                    if train {
                        cmd.gemm(Op::N, Op::N, r, h, cfg.vocab, 1.0, &s.logits, 0, cfg.vocab, &self.p, self.lay.embed, h, 0.0, &self.dxf, c0 * h, h);
                        cmd.gemm(Op::T, Op::N, cfg.vocab, h, r, 1.0, &s.logits, 0, cfg.vocab, &self.xf, c0 * h, h, 1.0, &self.g, self.lay.embed, h);
                    }
                }
                return;
            }
            let cs = cfg.vocab / ncl;
            let hc = self.lay.head_clusters;
            // level 1: clusters
            cmd.gemm(Op::N, Op::T, m, ncl, h, 1.0, &self.xf, 0, h, &self.p, hc, h, 0.0, &self.lc, 0, ncl);
            cmd.softmax_ce_at(&self.lc, 0, &self.tgt_cluster, 0, &s.loss, 0, m, ncl, scale);
            if train {
                cmd.gemm(Op::N, Op::N, m, h, ncl, 1.0, &self.lc, 0, ncl, &self.p, hc, h, 0.0, &self.dxf, 0, h);
                cmd.gemm(Op::T, Op::N, ncl, h, m, 1.0, &self.lc, 0, ncl, &self.xf, 0, h, 1.0, &self.g, hc, h);
            }
            // level 2: within the target cluster, rows grouped by cluster
            let groups = self.head_groups.borrow();
            let rows_total = groups.last().map(|(_, off, pad)| off + pad).unwrap_or(0);
            assert!(
                rows_total >= m,
                "hierarchical head: prepare_head(targets) must precede the encode (grouped {rows_total} rows for {m} tokens)"
            );
            cmd.gather_rows(&self.xf, &self.head_idx, &self.hg, rows_total, h);
            for &(c, off, pad) in groups.iter() {
                cmd.gemm(Op::N, Op::T, pad, cs, h, 1.0, &self.hg, off * h, h, &self.p, self.lay.embed + c * cs * h, h, 0.0, &self.lw, off * cs, cs);
            }
            cmd.softmax_ce_idx(&self.lw, &self.head_idx, &self.tgt, &self.loss2, rows_total, cs, scale);
            if train {
                for &(c, off, pad) in groups.iter() {
                    cmd.gemm(Op::N, Op::N, pad, h, cs, 1.0, &self.lw, off * cs, cs, &self.p, self.lay.embed + c * cs * h, h, 0.0, &self.dhg, off * h, h);
                    cmd.gemm(Op::T, Op::N, cs, h, pad, 1.0, &self.lw, off * cs, cs, &self.hg, off * h, h, 1.0, &self.g, self.lay.embed + c * cs * h, h);
                }
                cmd.scatter_add_rows(&self.dxf, &self.head_idx, &self.dhg, rows_total, h);
            }
        }

        /// Mean loss of the batch just run (both levels of the head).
        pub fn read_loss(&self) -> f32 {
            let m = self.b * self.t;
            let l1: f64 = self.scratch.loss.as_slice()[..m].iter().map(|x| *x as f64).sum();
            let l2: f64 = if self.cfg.head_clusters > 0 { self.loss2.as_slice()[..m].iter().map(|x| *x as f64).sum() } else { 0.0 };
            ((l1 + l2) / m as f64) as f32
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
            self.encode_head(cmd, true);
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
            self.prepare_head(targets);
            let cmd = Cmd::new(c);
            let groups = self.encode_fwd_bwd(&cmd);
            let ms1 = cmd.commit();
            let gnorm = self.scratch.partial.as_slice()[..groups].iter().map(|x| *x as f64).sum::<f64>().sqrt() as f32;
            let loss = self.read_loss();
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
            self.prepare_head(targets);
            let cfg = &self.cfg;
            let (h, m) = (cfg.hidden, m);
            let cmd = Cmd::new(self.ctx());
            cmd.embed_gather_at(&self.p, self.lay.embed, &self.tok, self.x_in(0), m, h);
            for l in 0..cfg.layers {
                let out: &GBuf = if l + 1 < cfg.layers { self.x_in(l + 1) } else { &self.x_out };
                self.layer_fwd(&cmd, l, out);
            }
            cmd.rmsnorm_fwd_at(&self.x_out, &self.p, self.lay.final_norm, &self.xf, &self.invf, m, h, cfg.norm_eps);
            self.encode_head(&cmd, false);
            cmd.commit();
            self.read_loss()
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
        out.push(("head fwd+bwd".into(), time("h".into(), &|cmd| self.encode_head(cmd, true))));
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
