//! Native MOSS-Image denoiser.
//!
//! MOSS-Image is deliberately not routed through Z-Image: its 42 blocks use
//! joint [image,text] attention, four-axis RoPE, six-way AdaRMS modulation
//! and SiTU.  This module keeps source tensor suffixes under `moss.` and
//! uses CMF's mapped quantized projections.  The hot dense products therefore
//! use the existing q4tp/q8 device path on Metal/Vulkan, while the portable
//! f32 implementation remains an exact-shape fallback.

use crate::dit::{Proj, cmf_f32};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// `CMF_MOSS_PROF=1`: wall-time totals per MOSS DiT op class,
/// accumulated across every block of every forward of every step and
/// printed when generation ends (see `moss_imagegen`).  Same contract
/// as `cpuprof`/`dit::prof`: off by default, one cached bool per
/// instrumented site, zero nanoseconds recorded when disabled.
pub mod prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const VEMB: usize = 0;
    pub const TIME: usize = 1;
    pub const ROPE: usize = 2;
    pub const NORMS: usize = 3;
    pub const QKNORM: usize = 4;
    pub const ATTN: usize = 5;
    pub const ATTN_GPU: usize = 6;
    pub const MATMAT: usize = 7;
    pub const ELEM: usize = 8;
    pub const HEAD: usize = 9;
    const N: usize = 10;
    const NAMES: [&str; N] = [
        "latent+text layout",
        "time+modulation",
        "rope build",
        "rmsnorms",
        "qknorm+rope",
        "attention(total)",
        "  of which gpu",
        "linears(matmat)",
        "elementwise",
        "head+tail",
    ];

    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

    static ON: OnceLock<bool> = OnceLock::new();
    #[inline]
    pub fn on() -> bool {
        *ON.get_or_init(|| std::env::var("CMF_MOSS_PROF").is_ok_and(|v| v != "0"))
    }

    /// Scope timer; records on drop. `None` when profiling is off.
    pub struct Timer(Option<(usize, std::time::Instant)>);

    impl Drop for Timer {
        #[inline]
        fn drop(&mut self) {
            if let Some((s, t0)) = self.0.take() {
                NS[s].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                CALLS[s as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn time(s: usize) -> Timer {
        if on() {
            Timer(Some((s, std::time::Instant::now())))
        } else {
            Timer(None)
        }
    }

    /// Record an externally measured span (e.g. a GPU arm measured by
    /// its own clock) without nesting a second timer.
    pub fn add(s: usize, ns: u64) {
        NS[s].fetch_add(ns, Ordering::Relaxed);
        CALLS[s].fetch_add(1, Ordering::Relaxed);
    }

    pub fn hit(s: usize) -> u64 {
        CALLS[s].load(Ordering::Relaxed)
    }

    pub fn reset() {
        for i in 0..N {
            NS[i].store(0, Ordering::Relaxed);
            CALLS[i].store(0, Ordering::Relaxed);
        }
    }

    /// One block of `name: ms (share%)` lines; `base` is the wall-clock
    /// budget the shares are taken against (the summed step time).
    pub fn report(base_secs: f64) -> String {
        let mut out = String::new();
        let accounted: f64 = (0..N)
            .filter(|&s| s != ATTN_GPU)
            .map(|s| NS[s].load(Ordering::Relaxed) as f64)
            .sum();
        for s in 0..N {
            let ns = NS[s].load(Ordering::Relaxed);
            if ns == 0 {
                continue;
            }
            let calls = CALLS[s].load(Ordering::Relaxed);
            let ms = ns as f64 / 1e6;
            if s == ATTN_GPU {
                out.push_str(&format!("    {name:<22} {ms:>10.1} ms ({calls} hits)\n", name = NAMES[s]));
            } else {
                let share = if base_secs > 0.0 {
                    ns as f64 / 1e9 / base_secs * 100.0
                } else {
                    0.0
                };
                out.push_str(&format!(
                    "    {name:<22} {ms:>10.1} ms ({share:>5.1}% of steps, {calls} calls)\n",
                    name = NAMES[s]
                ));
            }
        }
        let acc = accounted / 1e9;
        out.push_str(&format!(
            "    {:<22} {acc:>10.3} s of {base_secs:.3} s step budget\n",
            "accounted"
        ));
        let (ok, secs) = crate::gpu::op_stats();
        out.push_str(&format!(
            "    {:<22} {ok} submissions, {secs:.3} s wall ({} gpu-branch tries)\n",
            "device ops",
            crate::gpu::op_tries()
        ));
        out
    }
}

/// Native MOSS transformer geometry, read from the source config stored in
/// `moss.config_json`.
#[derive(Clone, Debug)]
pub struct MossConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub latent_channels: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub text_dim: usize,
    pub rope_base: f64,
}

impl MossConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |name: &str| -> Result<usize, String> {
            v[name]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("MOSS config missing integer {name}"))
        };
        let dim = u("dim")?;
        let num_heads = u("num_heads")?;
        let freq_dim = u("freq_dim")?;
        if dim == 0 || num_heads == 0 || dim % num_heads != 0 || (dim / num_heads) % 8 != 0 {
            return Err(format!(
                "invalid MOSS dim/head geometry: dim={dim}, num_heads={num_heads}"
            ));
        }
        if freq_dim == 0 || freq_dim % 2 != 0 {
            return Err(format!(
                "MOSS freq_dim must be positive and even, got {freq_dim}"
            ));
        }
        Ok(Self {
            dim,
            ffn_dim: u("ffn_dim")?,
            freq_dim,
            latent_channels: u("latent_channels")?,
            num_heads,
            num_layers: u("num_layers")?,
            text_dim: u("text_dim")?,
            rope_base: v["rope_base"].as_f64().unwrap_or(10_000.0),
        })
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }
}

struct TextBlock {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qkv: Proj,
    qkv_b: Vec<f32>,
    proj: Proj,
    proj_b: Vec<f32>,
    ffn_gate: Proj,
    ffn_gate_b: Vec<f32>,
    ffn_up: Proj,
    ffn_up_b: Vec<f32>,
    ffn_down: Proj,
    ffn_down_b: Vec<f32>,
}

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    ffn_up_gate: Proj,
    ffn_down: Proj,
}

/// The MOSS DiT loaded directly from one CMF.  Quantized projection tensors
/// remain mmap-backed, rather than being expanded to f32 at load time.
pub struct MossDit {
    pub cfg: MossConfig,
    video_embedder: Proj,
    video_embedder_b: Vec<f32>,
    text_proj: Proj,
    text_proj_b: Vec<f32>,
    text_blocks: Vec<TextBlock>,
    time_0_w: Vec<f32>,
    time_0_b: Vec<f32>,
    time_2_w: Vec<f32>,
    time_2_b: Vec<f32>,
    shared_mod_w: Vec<f32>,
    shared_mod_b: Vec<f32>,
    head_mod_w: Vec<f32>,
    head_mod_b: Vec<f32>,
    layers: Vec<Block>,
    final_norm_video: Vec<f32>,
    final_linear_video: Proj,
    pool: Option<Arc<Pool>>,
    /// Reusable per-run scratch. Every block/step resizes into it instead
    /// of allocating MB-scale Vecs (the layer loop used to churn ~GB per
    /// forward through malloc/mmap). Borrowed with `RefCell` — the forward
    /// is strictly sequential on one thread.
    work: std::cell::RefCell<Work>,
}

/// Attention scratch: transposed head planes, one score matrix, and the
/// output row buffer — all n-dependent, all reused across every call.
#[derive(Default)]
struct AttnW {
    qh: Vec<f32>,
    kh: Vec<f32>,
    vh: Vec<f32>,
    scores: Vec<f32>,
    vals_t: Vec<f32>,
    head_out: Vec<f32>,
    out: Vec<f32>,
}

/// Per-block scratch (norms, projections, FFN), sized by the largest
/// sequence seen — the image+text forward at 512².
#[derive(Default)]
struct Blk {
    norm: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    aw: AttnW,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn: Vec<f32>,
    out: Vec<f32>,
}

/// Forward-level scratch (latent layout, embeddings, head image).
#[derive(Default)]
struct Fwd {
    image_in: Vec<f32>,
    video: Vec<f32>,
    x: Vec<f32>,
    image: Vec<f32>,
}

#[derive(Default)]
struct Work {
    f: Fwd,
    b: Blk,
}

/// Length-exact view of a scratch buffer (first call sizes it, later
/// calls with the same length are a no-op borrow).
fn fit(buf: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buf.len() != len {
        buf.resize(len, 0f32);
    }
    buf
}

impl MossDit {
    /// Load the MOSS DiT from `moss.*` tensors in a packaged model.
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("moss.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("moss.config_json: {e}"))?;
        let cfg = MossConfig::from_json(&cfg)?;
        let f = |suffix: &str| cmf_f32(model, &format!("moss.{suffix}"));
        let p = |suffix: &str| Proj::from_model(model, &format!("moss.{suffix}"));
        let text_blocks = (0..2)
            .map(|i| {
                let x = format!("text_embedder.blocks.{i}");
                Ok(TextBlock {
                    norm1: f(&format!("{x}.norm1.weight"))?,
                    norm2: f(&format!("{x}.norm2.weight"))?,
                    q_norm: f(&format!("{x}.q_norm.weight"))?,
                    k_norm: f(&format!("{x}.k_norm.weight"))?,
                    qkv: p(&format!("{x}.qkv.weight"))?,
                    qkv_b: f(&format!("{x}.qkv.bias"))?,
                    proj: p(&format!("{x}.proj.weight"))?,
                    proj_b: f(&format!("{x}.proj.bias"))?,
                    ffn_gate: p(&format!("{x}.ffn_gate.weight"))?,
                    ffn_gate_b: f(&format!("{x}.ffn_gate.bias"))?,
                    ffn_up: p(&format!("{x}.ffn_up.weight"))?,
                    ffn_up_b: f(&format!("{x}.ffn_up.bias"))?,
                    ffn_down: p(&format!("{x}.ffn_down.weight"))?,
                    ffn_down_b: f(&format!("{x}.ffn_down.bias"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let layers = (0..cfg.num_layers)
            .map(|i| {
                let x = format!("layers.{i}");
                Ok(Block {
                    norm1: f(&format!("{x}.norm1.weight"))?,
                    norm2: f(&format!("{x}.norm2.weight"))?,
                    q_norm: f(&format!("{x}.q_norm.weight"))?,
                    k_norm: f(&format!("{x}.k_norm.weight"))?,
                    q: p(&format!("{x}.proj.q.weight"))?,
                    k: p(&format!("{x}.proj.k.weight"))?,
                    v: p(&format!("{x}.proj.v.weight"))?,
                    o: p(&format!("{x}.proj.o.weight"))?,
                    ffn_up_gate: p(&format!("{x}.proj.ffn_up_gate.weight"))?,
                    ffn_down: p(&format!("{x}.proj.ffn_down.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let dit = Self {
            video_embedder: p("video_embedder.weight")?,
            video_embedder_b: f("video_embedder.bias")?,
            text_proj: p("text_embedder.proj.weight")?,
            text_proj_b: f("text_embedder.proj.bias")?,
            time_0_w: f("time_embedding.0.weight")?,
            time_0_b: f("time_embedding.0.bias")?,
            time_2_w: f("time_embedding.2.weight")?,
            time_2_b: f("time_embedding.2.bias")?,
            shared_mod_w: f("shared_block_modulation.weight")?,
            shared_mod_b: f("shared_block_modulation.bias")?,
            head_mod_w: f("video_head_modulation.weight")?,
            head_mod_b: f("video_head_modulation.bias")?,
            final_norm_video: f("final_norm_video.weight")?,
            final_linear_video: p("final_linear_video.weight")?,
            text_blocks,
            layers,
            pool: Pool::from_env(),
            work: std::cell::RefCell::new(Work::default()),
            cfg,
        };
        dit.validate()?;
        Ok(dit)
    }

    fn validate(&self) -> Result<(), String> {
        let d = self.cfg.dim;
        if self.layers.len() != self.cfg.num_layers || self.text_blocks.len() != 2 {
            return Err("MOSS block count mismatch".into());
        }
        if self.video_embedder.cols() != self.cfg.latent_channels
            || self.video_embedder.rows() != d
            || self.text_proj.cols() != self.cfg.text_dim
            || self.text_proj.rows() != d
            || self.final_linear_video.cols() != d
            || self.final_linear_video.rows() != self.cfg.latent_channels
            || self.final_norm_video.len() != d
        {
            return Err("MOSS top-level projection geometry mismatch".into());
        }
        if self.shared_mod_w.len() != 6 * d * d || self.shared_mod_b.len() != 6 * d {
            return Err("MOSS shared modulation geometry mismatch".into());
        }
        if self.head_mod_w.len() != 2 * d * d || self.head_mod_b.len() != 2 * d {
            return Err("MOSS head modulation geometry mismatch".into());
        }
        Ok(())
    }

    fn pool(&self) -> Option<&Pool> {
        self.pool.as_deref()
    }

    /// Prepare the encoder's final Qwen3.5 hidden states once per prompt.
    /// MOSS's text projection/refiner does not depend on the latent or the
    /// timestep, so retaining this result avoids repeating two blocks on
    /// every CFG/denoising forward.
    pub fn prepare_text(&self, text: &[f32], tokens: usize) -> Result<Vec<f32>, String> {
        if tokens == 0 || text.len() != tokens * self.cfg.text_dim {
            return Err(format!(
                "MOSS text conditioning has {} values for {tokens} tokens of width {}",
                text.len(),
                self.cfg.text_dim
            ));
        }
        let mut work = self.work.borrow_mut();
        let Work { b: blk, .. } = &mut *work;
        let mut x: Vec<f32> = Vec::new();
        linear_into(
            &self.text_proj,
            text,
            tokens,
            Some(&self.text_proj_b),
            self.pool(),
            &mut x,
        );
        let rope = Rope::new(&self.cfg, 0, 0, tokens);
        for b in &self.text_blocks {
            text_block(b, &mut x, tokens, &rope, &self.cfg, self.pool(), blk);
        }
        Ok(x)
    }

    /// One MOSS velocity forward.  `latents` is channels-first
    /// `[latent_channels, h, w]`; `text` is the result of
    /// [`Self::prepare_text`].  Passing a skip set reproduces MOSS's
    /// negative-branch layer skip without changing sequence geometry.
    pub fn forward(
        &self,
        latents: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        text: &[f32],
        text_tokens: usize,
        skip_layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        if h == 0 || w == 0 || latents.len() != self.cfg.latent_channels * h * w {
            return Err("MOSS latent shape is not [C,H,W]".into());
        }
        if text_tokens == 0 || text.len() != text_tokens * self.cfg.dim {
            return Err("MOSS prepared text shape mismatch".into());
        }
        if skip_layers.iter().any(|&i| i >= self.layers.len()) {
            return Err("MOSS skip layer out of range".into());
        }
        let image_tokens = h * w;
        let pool = self.pool();
        let mut work = self.work.borrow_mut();
        let Work { f, b } = &mut *work;
        let _t_vemb = prof::time(prof::VEMB);
        fit(&mut f.image_in, image_tokens * self.cfg.latent_channels);
        for token in 0..image_tokens {
            for c in 0..self.cfg.latent_channels {
                f.image_in[token * self.cfg.latent_channels + c] =
                    latents[c * image_tokens + token];
            }
        }
        drop(_t_vemb);
        linear_into(
            &self.video_embedder,
            &f.image_in,
            image_tokens,
            Some(&self.video_embedder_b),
            pool,
            &mut f.video,
        );
        let n = image_tokens + text_tokens;
        f.x.clear();
        f.x.extend_from_slice(&f.video);
        f.x.extend_from_slice(text);
        debug_assert_eq!(f.x.len(), n * self.cfg.dim);
        let _t_rope = prof::time(prof::ROPE);
        let rope = Rope::new(&self.cfg, h, w, text_tokens);
        drop(_t_rope);
        let _t_time = prof::time(prof::TIME);
        let time = self.time_embedding(timestep);
        let modulation = linear_small(&time, &self.shared_mod_w, &self.shared_mod_b);
        let head = linear_small(&time, &self.head_mod_w, &self.head_mod_b);
        drop(_t_time);
        for (i, block) in self.layers.iter().enumerate() {
            if !skip_layers.contains(&i) {
                main_block(block, &mut f.x, n, &rope, &modulation, &self.cfg, pool, b);
            }
        }
        let d = self.cfg.dim;
        fit(&mut f.image, image_tokens * d);
        f.image.copy_from_slice(&f.x[..image_tokens * d]);
        rms_rows_inplace(
            &mut f.image,
            image_tokens,
            d,
            &self.final_norm_video,
            None,
            None,
            pool,
        );
        let (shift, scale) = (&head[..d], &head[d..]);
        {
            let _t_head = prof::time(prof::HEAD);
            for row in f.image.chunks_exact_mut(d) {
                for ((v, &s), &a) in row.iter_mut().zip(scale).zip(shift) {
                    *v = *v * (1.0 + s) + a;
                }
            }
        }
        linear_into(
            &self.final_linear_video,
            &f.image,
            image_tokens,
            None,
            pool,
            &mut b.out,
        );
        let mut velocity = vec![0f32; self.cfg.latent_channels * image_tokens];
        {
            let _t_head = prof::time(prof::HEAD);
            let out = &b.out;
            for token in 0..image_tokens {
                for c in 0..self.cfg.latent_channels {
                    velocity[c * image_tokens + token] = out[token * self.cfg.latent_channels + c];
                }
            }
        }
        Ok(velocity)
    }

    fn time_embedding(&self, t: f32) -> Vec<f32> {
        // The source deliberately computes phase in f64, then sends the
        // concatenated [cos, sin] embedding through f32 Linear modules.
        let half = self.cfg.freq_dim / 2;
        let mut input = vec![0f32; self.cfg.freq_dim];
        for i in 0..half {
            let freq = self.cfg.rope_base.powf(-(i as f64) / half as f64);
            let p = t as f64 * freq;
            input[i] = p.cos() as f32;
            input[half + i] = p.sin() as f32;
        }
        let mut h = linear_small(&input, &self.time_0_w, &self.time_0_b);
        for v in &mut h {
            *v = silu(*v);
        }
        linear_small(&h, &self.time_2_w, &self.time_2_b)
    }
}

/// Four-axis MOSS RoPE, represented as fully expanded sin/cos factors for
/// source `_rotate`: the first 64 dimensions rotate the first half, and the
/// same factors repeat for the second half.
struct Rope {
    sin: Vec<f32>,
    cos: Vec<f32>,
}

impl Rope {
    fn new(cfg: &MossConfig, h: usize, w: usize, text_len: usize) -> Self {
        let hd = cfg.head_dim();
        let bands = hd / 8;
        let n_img = h * w;
        let n = n_img + text_len;
        let mut sin = vec![0f32; n * hd];
        let mut cos = vec![0f32; n * hd];
        // The band frequency depends only on the band — hoist the powf out
        // of the per-token loop (same f64 expression, so bit-identical).
        let freqs: Vec<f64> = (0..bands)
            .map(|band| 1.0 / cfg.rope_base.powf(band as f64 / bands as f64))
            .collect();
        for token in 0..n {
            let coords = if token < n_img {
                let yy = token / w;
                let xx = token % w;
                [
                    0.0,
                    text_len as f64,
                    yy as f64 - (h.saturating_sub(1) as f64) / 2.0,
                    xx as f64 - (w.saturating_sub(1) as f64) / 2.0,
                ]
            } else {
                [0.0, (token - n_img) as f64, 0.0, 0.0]
            };
            for axis in 0..4 {
                for band in 0..bands {
                    let phase = coords[axis] * freqs[band];
                    let j = axis * bands + band;
                    let s = phase.sin() as f32;
                    let c = phase.cos() as f32;
                    sin[token * hd + j] = s;
                    cos[token * hd + j] = c;
                    sin[token * hd + hd / 2 + j] = s;
                    cos[token * hd + hd / 2 + j] = c;
                }
            }
        }
        Self { sin, cos }
    }
}

// ── portable f32 kernels ────────────────────────────────────────────────

struct SendRows(*mut f32);
unsafe impl Send for SendRows {}
unsafe impl Sync for SendRows {}
impl SendRows {
    /// SAFETY: callers divide the buffer into non-overlapping row ranges.
    ///
    /// The `&self` receiver is deliberate: the wrapper is shared across
    /// worker closures (`Sync`) while each call carves out a disjoint slice
    /// of the same backing buffer, so no two live `&mut` ever alias —
    /// exactly the case `mut_from_ref` cannot see through.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }

    /// SAFETY: read-only view of a source range the caller only reads;
    /// used where the wrapper holds a const buffer cast for convenience.
    unsafe fn rrow(&self, off: usize, len: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.0.add(off), len) }
    }
}

fn rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn situ(hidden: f32, gate: f32) -> f32 {
    (4.0 * (gate / 4.0).tanh() * gate.sigmoid()) * (25.0 * (hidden / 25.0).tanh())
}

trait Sigmoid {
    fn sigmoid(self) -> Self;
}
impl Sigmoid for f32 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}

fn linear_small(x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    assert_eq!(w.len() % x.len(), 0, "small MOSS linear geometry");
    let rows = w.len() / x.len();
    assert_eq!(bias.len(), rows, "small MOSS linear bias geometry");
    w.chunks_exact(x.len())
        .zip(bias)
        .map(|(r, &b)| b + r.iter().zip(x).map(|(&a, &v)| a * v).sum::<f32>())
        .collect()
}

/// `y[n, rows] = x[n, cols] · Wᵀ (+ bias)`, writing into the reusable
/// `buf` instead of allocating per call. Returns the filled region.
fn linear_into<'a>(
    p: &Proj,
    x: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    pool: Option<&Pool>,
    buf: &'a mut Vec<f32>,
) -> &'a mut [f32] {
    let _t = prof::time(prof::MATMAT);
    assert_eq!(x.len(), n * p.cols(), "MOSS projection input geometry");
    let rows = p.rows();
    let out = fit(buf, n * rows);
    p.matmat(x, n, out, pool);
    if let Some(b) = bias {
        assert_eq!(b.len(), rows, "MOSS projection bias geometry");
        for row in out.chunks_exact_mut(rows) {
            for (v, &b) in row.iter_mut().zip(b) {
                *v += b;
            }
        }
    }
    out
}

/// Plain Diffusers RMSNorm, optionally affine-modulated after normalization.
fn rms_rows_inplace(
    x: &mut [f32],
    n: usize,
    d: usize,
    weight: &[f32],
    scale: Option<&[f32]>,
    shift: Option<&[f32]>,
    pool: Option<&Pool>,
) {
    let _t = prof::time(prof::NORMS);
    assert_eq!(x.len(), n * d);
    assert_eq!(weight.len(), d);
    if let Some(s) = scale {
        assert_eq!(s.len(), d);
    }
    if let Some(s) = shift {
        assert_eq!(s.len(), d);
    }
    let ptr = SendRows(x.as_mut_ptr());
    rows(pool, n, &|lo, hi| {
        for r in lo..hi {
            // SAFETY: each worker owns distinct row indices.
            let row = unsafe { ptr.row(r * d, d) };
            let mean = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (mean + 1e-6).sqrt();
            for j in 0..d {
                let mut v = row[j] * inv * weight[j];
                if let Some(s) = scale {
                    v *= 1.0 + s[j];
                }
                if let Some(a) = shift {
                    v += a[j];
                }
                row[j] = v;
            }
        }
    });
}

fn qk_norm_rope(
    q: &mut [f32],
    n: usize,
    heads: usize,
    hd: usize,
    weight: &[f32],
    rope: &Rope,
    pool: Option<&Pool>,
) {
    let _t = prof::time(prof::QKNORM);
    assert_eq!(q.len(), n * heads * hd);
    assert_eq!(weight.len(), hd);
    let ptr = SendRows(q.as_mut_ptr());
    rows(pool, n, &|lo, hi| {
        for t in lo..hi {
            for h in 0..heads {
                // SAFETY: distinct token ranges are disjoint.
                let row = unsafe { ptr.row((t * heads + h) * hd, hd) };
                let mean = row.iter().map(|&v| v * v).sum::<f32>() / hd as f32;
                let inv = 1.0 / (mean + 1e-6).sqrt();
                for j in 0..hd {
                    row[j] *= inv * weight[j];
                }
                for j in 0..hd / 2 {
                    let a = row[j];
                    let b = row[j + hd / 2];
                    let c = rope.cos[t * hd + j];
                    let s = rope.sin[t * hd + j];
                    row[j] = a * c - b * s;
                    row[j + hd / 2] = b * c + a * s;
                }
            }
        }
    });
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Full bidirectional attention.  The GPU path consumes head-major q/k/v
/// planes and writes token-major rows; portable fallback holds one score
/// matrix per head (never one per layer) to bound memory.  All scratch
/// comes from `w` (reused across every call of the run), and the
/// token→head transposes run over the pooled workers — disjoint token
/// ranges keep the result bit-identical to the serial order.
fn attention<'a>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    hd: usize,
    pool: Option<&Pool>,
    w: &'a mut AttnW,
) -> &'a mut [f32] {
    let _t = prof::time(prof::ATTN);
    let hidden = heads * hd;
    fit(&mut w.qh, heads * n * hd);
    fit(&mut w.kh, heads * n * hd);
    fit(&mut w.vh, heads * n * hd);
    fit(&mut w.out, n * hidden);
    {
        // Parallel transpose: token t owns rows [t*heads, (t+1)*heads) of
        // every plane, so workers take disjoint token spans.
        let qp = SendRows(q.as_ptr() as *mut f32);
        let kp = SendRows(k.as_ptr() as *mut f32);
        let vp = SendRows(v.as_ptr() as *mut f32);
        let qhp = SendRows(w.qh.as_mut_ptr());
        let khp = SendRows(w.kh.as_mut_ptr());
        let vhp = SendRows(w.vh.as_mut_ptr());
        rows(pool, n, &|lo, hi| {
            for t in lo..hi {
                for h in 0..heads {
                    // SAFETY: each (t, h) cell is written by exactly one
                    // worker (token ranges are disjoint) and read from
                    // distinct q/k/v source cells.
                    let src = t * heads + h;
                    let dst = (h * n + t) * hd;
                    unsafe {
                        qhp
                            .row(dst, hd)
                            .copy_from_slice(qp.rrow(src * hd, hd));
                        khp
                            .row(dst, hd)
                            .copy_from_slice(kp.rrow(src * hd, hd));
                        vhp
                            .row(dst, hd)
                            .copy_from_slice(vp.rrow(src * hd, hd));
                    }
                }
            }
        });
    }
    if n >= 128 && crate::gpu::enabled_here() && !crate::gpu::mm_killed() {
        let g = if prof::on() { Some(std::time::Instant::now()) } else { None };
        if crate::gpu::dit_attention(
            &w.qh,
            &w.kh,
            &w.vh,
            heads,
            heads,
            n,
            hd,
            1.0 / (hd as f32).sqrt(),
            &mut w.out,
        ) {
            if let Some(g) = g {
                prof::add(prof::ATTN_GPU, g.elapsed().as_nanos() as u64);
            }
            return &mut w.out;
        }
    }
    fit(&mut w.scores, n * n);
    fit(&mut w.vals_t, hd * n);
    fit(&mut w.head_out, n * hd);
    let AttnW {
        qh,
        kh,
        vh,
        scores,
        vals_t,
        head_out,
        out,
    } = w;
    for h in 0..heads {
        let qs = &qh[h * n * hd..(h + 1) * n * hd];
        let ks = &kh[h * n * hd..(h + 1) * n * hd];
        let vs = &vh[h * n * hd..(h + 1) * n * hd];
        // GEMM wants the right operand laid out by output row.  Each score
        // row is q[t] dot k[j], so transpose logically into k's rows works.
        crate::fcd_ops::gemm_nt(qs, ks, scores, n, hd, n, pool);
        let sp = SendRows(scores.as_mut_ptr());
        rows(pool, n, &|lo, hi| {
            for r in lo..hi {
                // SAFETY: row ranges do not overlap.
                softmax(unsafe { sp.row(r * n, n) });
            }
        });
        for d in 0..hd {
            for t in 0..n {
                vals_t[d * n + t] = vs[t * hd + d];
            }
        }
        crate::fcd_ops::gemm_nt(scores, vals_t, head_out, n, n, hd, pool);
        for t in 0..n {
            out[t * hidden + h * hd..t * hidden + (h + 1) * hd]
                .copy_from_slice(&head_out[t * hd..(t + 1) * hd]);
        }
    }
    out
}

fn text_block(
    b: &TextBlock,
    x: &mut [f32],
    n: usize,
    rope: &Rope,
    cfg: &MossConfig,
    pool: Option<&Pool>,
    w: &mut Blk,
) {
    let d = cfg.dim;
    let Blk {
        norm,
        qkv,
        q,
        k,
        v,
        aw,
        gate,
        up,
        out,
        ..
    } = w;
    fit(norm, n * d).copy_from_slice(&x[..n * d]);
    rms_rows_inplace(norm, n, d, &b.norm1, None, None, pool);
    linear_into(&b.qkv, norm, n, Some(&b.qkv_b), pool, qkv);
    fit(q, n * d).copy_from_slice(&qkv[..n * d]);
    fit(k, n * d).copy_from_slice(&qkv[n * d..2 * n * d]);
    fit(v, n * d).copy_from_slice(&qkv[2 * n * d..3 * n * d]);
    qk_norm_rope(
        &mut q[..],
        n,
        cfg.num_heads,
        cfg.head_dim(),
        &b.q_norm,
        rope,
        pool,
    );
    qk_norm_rope(
        &mut k[..],
        n,
        cfg.num_heads,
        cfg.head_dim(),
        &b.k_norm,
        rope,
        pool,
    );
    let attn_out = attention(&q[..], &k[..], &v[..], n, cfg.num_heads, cfg.head_dim(), pool, aw);
    linear_into(&b.proj, attn_out, n, Some(&b.proj_b), pool, out);
    for (a, &val) in x.iter_mut().zip(out[..n * d].iter()) {
        *a += val;
    }
    fit(norm, n * d).copy_from_slice(&x[..n * d]);
    rms_rows_inplace(norm, n, d, &b.norm2, None, None, pool);
    linear_into(&b.ffn_gate, norm, n, Some(&b.ffn_gate_b), pool, gate);
    linear_into(&b.ffn_up, norm, n, Some(&b.ffn_up_b), pool, up);
    let inter = cfg.ffn_dim;
    for (a, &g) in up[..n * inter].iter_mut().zip(gate[..n * inter].iter()) {
        *a = situ(*a, g);
    }
    linear_into(&b.ffn_down, &up[..n * inter], n, Some(&b.ffn_down_b), pool, out);
    for (a, &val) in x.iter_mut().zip(out[..n * d].iter()) {
        *a += val;
    }
}

fn main_block(
    b: &Block,
    x: &mut [f32],
    n: usize,
    rope: &Rope,
    modulate: &[f32],
    cfg: &MossConfig,
    pool: Option<&Pool>,
    w: &mut Blk,
) {
    let d = cfg.dim;
    assert_eq!(modulate.len(), 6 * d);
    let (shift_a, rest) = modulate.split_at(d);
    let (scale_a, rest) = rest.split_at(d);
    let (gate_a, rest) = rest.split_at(d);
    let (shift_f, rest) = rest.split_at(d);
    let (scale_f, gate_f) = rest.split_at(d);
    let Blk {
        norm,
        q,
        k,
        v,
        aw,
        up,
        ffn,
        out,
        ..
    } = w;
    fit(norm, n * d).copy_from_slice(&x[..n * d]);
    rms_rows_inplace(
        norm,
        n,
        d,
        &b.norm1,
        Some(scale_a),
        Some(shift_a),
        pool,
    );
    linear_into(&b.q, norm, n, None, pool, q);
    linear_into(&b.k, norm, n, None, pool, k);
    linear_into(&b.v, norm, n, None, pool, v);
    qk_norm_rope(
        &mut q[..],
        n,
        cfg.num_heads,
        cfg.head_dim(),
        &b.q_norm,
        rope,
        pool,
    );
    qk_norm_rope(
        &mut k[..],
        n,
        cfg.num_heads,
        cfg.head_dim(),
        &b.k_norm,
        rope,
        pool,
    );
    let attn_out = attention(&q[..], &k[..], &v[..], n, cfg.num_heads, cfg.head_dim(), pool, aw);
    linear_into(&b.o, attn_out, n, None, pool, out);
    // Residual gate: rows are disjoint across workers, so each output
    // cell keeps its serial value — bit-identical, just parallel.
    {
        let _t = prof::time(prof::ELEM);
        let xp = SendRows(x.as_mut_ptr());
        let op = SendRows(out.as_mut_ptr());
        rows(pool, n, &|lo, hi| {
            for row in lo..hi {
                // SAFETY: workers own disjoint row ranges of x and out.
                let xs = unsafe { xp.row(row * d, d) };
                let os = unsafe { op.row(row * d, d) };
                for j in 0..d {
                    xs[j] += os[j] * gate_a[j];
                }
            }
        });
    }
    fit(norm, n * d).copy_from_slice(&x[..n * d]);
    rms_rows_inplace(
        norm,
        n,
        d,
        &b.norm2,
        Some(scale_f),
        Some(shift_f),
        pool,
    );
    let inter = cfg.ffn_dim;
    linear_into(&b.ffn_up_gate, norm, n, None, pool, up);
    fit(ffn, n * inter);
    {
        let _t = prof::time(prof::ELEM);
        let fp = SendRows(ffn.as_mut_ptr());
        let upp = SendRows(up.as_mut_ptr());
        rows(pool, n, &|lo, hi| {
            for row in lo..hi {
                // SAFETY: disjoint output rows; the up_gate source is
                // read-only (rrow) and indexed within this row's span.
                let f = unsafe { fp.row(row * inter, inter) };
                let u = unsafe { upp.rrow(row * 2 * inter, 2 * inter) };
                for i in 0..inter {
                    f[i] = situ(u[i], u[inter + i]);
                }
            }
        });
    }
    linear_into(&b.ffn_down, &ffn[..n * inter], n, None, pool, out);
    {
        let _t = prof::time(prof::ELEM);
        let xp = SendRows(x.as_mut_ptr());
        let op = SendRows(out.as_mut_ptr());
        rows(pool, n, &|lo, hi| {
            for row in lo..hi {
                // SAFETY: workers own disjoint row ranges of x and out.
                let xs = unsafe { xp.row(row * d, d) };
                let os = unsafe { op.row(row * d, d) };
                for j in 0..d {
                    xs[j] += os[j] * gate_f[j];
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_layout_repeats_quarter_axis_frequencies() {
        let cfg = MossConfig {
            dim: 64,
            ffn_dim: 64,
            freq_dim: 8,
            latent_channels: 4,
            num_heads: 1,
            num_layers: 1,
            text_dim: 8,
            rope_base: 10_000.0,
        };
        let r = Rope::new(&cfg, 2, 2, 3);
        assert_eq!(r.sin.len(), 7 * 64);
        for t in 0..7 {
            assert_eq!(
                &r.sin[t * 64..t * 64 + 32],
                &r.sin[t * 64 + 32..(t + 1) * 64]
            );
            assert_eq!(
                &r.cos[t * 64..t * 64 + 32],
                &r.cos[t * 64 + 32..(t + 1) * 64]
            );
        }
    }

    #[test]
    fn situ_matches_definition() {
        let h = 2.5f32;
        let g = -1.25f32;
        let got = situ(h, g);
        let want =
            (4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp()))) * (25.0 * (h / 25.0).tanh());
        assert!((got - want).abs() < 1e-7);
    }
}
