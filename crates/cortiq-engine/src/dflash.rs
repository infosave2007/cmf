//! DFlash 2: a drafter that proposes a block of tokens in one pass.
//!
//! The drafter is a few Qwen3-shaped layers beside its target. It never reads
//! the target's tokens before the anchor: what it knows of the text comes from
//! several of the target's layer outputs, joined a row at a time and projected
//! into every draft layer's keys and values.
//!
//! ```text
//! context   t = norm(FC · [target layer outputs of a row])
//!           K_l = rot(norm(Wk_l · t)), V_l = Wv_l · t          kept per position
//!           (a layer may be without rotation, and each has a window of its own)
//! block     rows = [anchor, MASK, …, MASK] at positions a, a + 1, …
//!           x = norm(h); x, d = conv_in(x)                     grouped causal conv over the rows
//!           q, k, v = W · x; attention over K_l, V_l before a and the block's own rows (not causal)
//!           h += conv_out(O · y, d); the MLP is wrapped the same way
//! tokens    logits = head · norm(h) (the target's head), its top candidates re-scored
//!           in order by the selector: unary + (P[prev] ⊙ S·h) · C[candidate]
//! ```
//!
//! Row j ≥ 1 of the block drafts the token at position a + j. The sidecar
//! (`<stem>.dflash.cmf`, namespace `dflash.`) is written by `cortiq convert
//! --dflash-sidecar`; the embedding and the head are the target's.

use crate::pool::Pool;
use crate::qtensor::{QTensor, sgemm_public};
use cortiq_core::quant::f16_to_f32;
use cortiq_core::format::SelectionDescriptor;
use cortiq_core::{CmfModel, TensorDtype};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SIDECAR_ARCH: &str = "dflash2_sidecar";
/// Where a sidecar's header keeps its router (`RouteFit`), under `provenance`.
pub const ROUTE_KEY: &str = "dflash_route";
pub const NAMESPACE: &str = "dflash.";
/// Context rows projected per pass: bounds the joined-taps buffer.
const CONTEXT_CHUNK: usize = 64;
/// Context rows that may wait unsettled (`queue_context`): 100 KB a row on
/// the 27B's five tapped layers.
const PENDING_ROWS: usize = 512;

/// `<stem>.dflash.cmf` beside the target `<stem>.cmf`.
pub fn sidecar_path(main: &Path) -> PathBuf {
    let name = main.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if name.ends_with(".dflash.cmf") {
        return main.to_path_buf();
    }
    let stem = name.strip_suffix(".cmf").unwrap_or(&name);
    main.with_file_name(format!("{stem}.dflash.cmf"))
}

/// What a DFlash 2 checkpoint's config fixes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// Rows of a block: the anchor and `block − 1` drafts.
    pub block: usize,
    pub mask_token: u32,
    pub conv_group: usize,
    pub conv_kernel: usize,
    pub selector_rank: usize,
    pub selector_top: usize,
    /// The target's layers whose outputs a context row joins, zero-based:
    /// the output of layer `id`, before any norm.
    pub taps: Vec<usize>,
    pub target_layers: usize,
    /// Positions a block row sees back; 0 is all of them.
    pub window: usize,
    /// The same per layer where the layers differ (a full-attention layer
    /// among sliding ones: 0). Empty: `window` for every layer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub windows: Vec<usize>,
    /// Layers whose queries and keys carry no rotary position.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub no_rope: Vec<usize>,
    pub rope_theta: f64,
    pub eps: f64,
}

impl Geometry {
    /// The geometry a sidecar declares.
    pub fn of(model: &CmfModel) -> Result<Self, String> {
        if model.arch().arch_name != SIDECAR_ARCH {
            return Err(format!("{} is not a DFlash sidecar", model.arch().arch_name));
        }
        let v = model.header.provenance.as_ref().and_then(|p| p.get("dflash")).ok_or("the sidecar's header has no dflash record")?;
        let g: Self = serde_json::from_value(v.clone()).map_err(|e| format!("the sidecar's dflash record: {e}"))?;
        if g.layers == 0
            || g.block < 2
            || g.taps.is_empty()
            || g.kv_heads == 0
            || g.heads % g.kv_heads != 0
            || g.head_dim % 2 != 0
            || g.conv_group == 0
            || g.hidden % g.conv_group != 0
            || !(g.windows.is_empty() || g.windows.len() == g.layers)
            || g.no_rope.iter().any(|&l| l >= g.layers)
        {
            return Err(format!("unsupported DFlash geometry {g:?}"));
        }
        Ok(g)
    }

    /// Width of a context row as the target hands it over.
    pub fn tap_width(&self) -> usize {
        self.taps.len() * self.hidden
    }

    /// Positions a block row of `layer` sees back; 0 is all of them.
    pub fn window_of(&self, layer: usize) -> usize {
        self.windows.get(layer).copied().unwrap_or(self.window)
    }

    /// Whether `layer` rotates its queries and keys by position.
    pub fn rotary(&self, layer: usize) -> bool {
        !self.no_rope.contains(&layer)
    }
}

struct Conv {
    /// `[stage][offset][hidden]`: stage 0 goes before the sub-layer, 1 after.
    base: Vec<f32>,
    proj: Small,
}

/// A small matrix (a kernel projection, the selector's) unpacked to f32: the
/// host has it applied to a block's rows before a submission to the device
/// would come back.
struct Small {
    w: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Small {
    fn of(t: QTensor) -> Self {
        let (rows, cols) = (t.rows(), t.cols());
        let mut w = vec![0f32; rows * cols];
        for (r, row) in w.chunks_exact_mut(cols).enumerate() {
            t.row_f32(r, row);
        }
        Self { w, rows, cols }
    }

    /// `[b][rows]` from `x`, `[b][cols]`. The rows go across the pool's
    /// threads: `sgemm_public` runs one thread on x86, and a block's ten
    /// kernel projections took 260 ms of a 340 ms block that way.
    fn apply(&self, x: &[f32], b: usize, pool: Option<&Pool>) -> Vec<f32> {
        let (rows, cols) = (self.rows, self.cols);
        let mut out = vec![0f32; b * rows];
        let Some(pool) = pool.filter(|_| b * rows * cols >= 1 << 20 && rows >= 64) else {
            sgemm_public(b, rows, cols, 1.0, x, cols, &self.w, cols, true, &mut out, rows);
            return out;
        };
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}
        impl SendPtr {
            fn at(&self, i: usize) -> *mut f32 {
                // SAFETY: callers index inside the buffer the pointer came from.
                unsafe { self.0.add(i) }
            }
        }
        let dst = SendPtr(out.as_mut_ptr());
        pool.run_rows(rows, &|r0, r1| {
            let n = r1 - r0;
            let mut part = vec![0f32; b * n];
            sgemm_public(b, n, cols, 1.0, x, cols, &self.w[r0 * cols..r1 * cols], cols, true, &mut part, n);
            for bi in 0..b {
                // SAFETY: the chunks' row ranges are disjoint, so no two
                // workers write the same element; `out` outlives the run.
                unsafe {
                    std::ptr::copy_nonoverlapping(part[bi * n..].as_ptr(), dst.at(bi * rows + r0), n);
                }
            }
        });
        out
    }
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q: QTensor,
    k: QTensor,
    v: QTensor,
    o: QTensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: QTensor,
    up: QTensor,
    down: QTensor,
    attn_conv: Conv,
    mlp_conv: Conv,
}

/// A layer's keys and values of the context positions, per KV head, a row of
/// `head_dim` per position.
#[derive(Default)]
struct Ctx {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Rows that fell out of the layer's window and were let go: the layer
    /// holds positions `first + dropped .. first + len`.
    dropped: usize,
}

pub struct Drafter {
    pub g: Geometry,
    /// The sidecar's router, when it carries one.
    pub route: Option<Route>,
    model: Arc<CmfModel>,
    fc: QTensor,
    hidden_norm: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    sel: Small,
    /// Directory indices of the predecessor and successor codebooks.
    books: [usize; 2],
    inv_freq: Vec<f32>,
    ctx: Vec<Ctx>,
    /// The context covers positions `first .. first + len`; a layer holds
    /// the part of them its window still reaches (`Ctx::dropped`).
    first: usize,
    len: usize,
    /// Rows that wait to become context (`queue_context`): their layer
    /// outputs, from position `first + len` on.
    pending: Vec<f32>,
    /// Matrices go to the device when it takes them.
    device: bool,
    /// Matrices that read the same rows go to the device in one submission
    /// (`CMF_DFLASH_FUSED=0`: a submission each).
    fused: bool,
}

fn vector(model: &CmfModel, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let name = format!("{NAMESPACE}{name}");
    let e = model.tensor(&name).ok_or_else(|| format!("the sidecar has no {name}"))?;
    if e.dtype != TensorDtype::F32 || e.shape.iter().product::<usize>() != len {
        return Err(format!("{name}: {:?} {:?}, expected {len} f32", e.dtype, e.shape));
    }
    Ok(model.entry_bytes(e).chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Where a block's time goes (`CMF_DFLASH_TIME=1`): microseconds in the kernel
/// projections, the attention projections, the attention itself, its output
/// projection, the MLP, and the blocks counted.
static STAGE_US: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn stage_timed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DFLASH_TIME").is_ok())
}

/// Run `f`, its time added to stage `i` when the stages are timed.
fn stage<T>(i: usize, f: impl FnOnce() -> T) -> T {
    if !stage_timed() {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    STAGE_US[i].fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    r
}

/// A line of the stages' average milliseconds a block, when they are timed.
pub fn stage_report() -> Option<String> {
    let v: Vec<f64> = STAGE_US.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64).collect();
    (stage_timed() && v[5] > 0.0).then(|| {
        let ms = |i: usize| v[i] / 1e3 / v[5];
        format!("a block: kernel projections {:.1} ms, q k v {:.1}, attention {:.1}, o {:.1}, MLP {:.1}", ms(0), ms(1), ms(2), ms(3), ms(4))
    })
}

/// Which drafter a round is for, as the sidecar keeps it: a subspace of the
/// target's rows at the tip for the rounds this block won and one for the
/// rounds the target's own MTP chain won, each fitted on its own rows alone
/// (`router::fit_descriptor`, the engine's recon-argmin routing). `cortiq
/// dflash-route-fit` writes it.
#[derive(Clone, Serialize, Deserialize)]
pub struct RouteFit {
    pub block: SelectionDescriptor,
    pub chain: SelectionDescriptor,
    /// Rounds each was fitted on.
    pub block_rows: usize,
    pub chain_rows: usize,
}

struct Subspace {
    mean: Vec<f32>,
    basis: Vec<f32>,
    rank: usize,
}

impl Subspace {
    fn of(d: &SelectionDescriptor, hidden: usize) -> Result<Self, String> {
        let (mean, basis) = (crate::router::decode_f16(&d.mean), crate::router::decode_f16(&d.basis));
        match (mean, basis) {
            (Some(mean), Some(basis)) if mean.len() == hidden && basis.len() == d.rank * hidden => Ok(Self { mean, basis, rank: d.rank }),
            _ => Err(format!("a router subspace is not {hidden} wide at rank {}", d.rank)),
        }
    }
}

/// A sidecar's router, ready to read rows.
pub struct Route {
    block: Subspace,
    chain: Subspace,
}

impl Route {
    pub fn of(fit: &RouteFit, hidden: usize) -> Result<Self, String> {
        Ok(Self { block: Subspace::of(&fit.block, hidden)?, chain: Subspace::of(&fit.chain, hidden)? })
    }

    /// How much better the block's subspace holds the target's row `row`
    /// than the chain's does: above zero the round is the block's.
    pub fn margin(&self, row: &[f32]) -> f32 {
        let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let unit: Vec<f32> = row.iter().map(|v| v / n).collect();
        let err = |s: &Subspace| crate::router::recon_error(&unit, &s.mean, &s.basis, s.rank);
        err(&self.chain) - err(&self.block)
    }
}

fn matrix(model: &Arc<CmfModel>, name: &str, rows: usize, cols: usize) -> Result<QTensor, String> {
    let name = format!("{NAMESPACE}{name}");
    let t = QTensor::from_model(model, &name)?;
    if t.rows() != rows || t.cols() != cols {
        return Err(format!("{name}: [{}, {}], expected [{rows}, {cols}]", t.rows(), t.cols()));
    }
    Ok(t)
}

/// `x · rsqrt(mean(x²) + eps) ⊙ w`, a row of `w.len()` at a time.
fn rms(x: &mut [f32], w: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(w.len()) {
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / w.len() as f64;
        let s = 1.0 / (ms as f32 + eps).sqrt();
        for (v, g) in row.iter_mut().zip(w) {
            *v = *v * s * g;
        }
    }
}

impl Drafter {
    /// The drafter of `main`'s sidecar, when it has one.
    pub fn beside(main: &Path) -> Option<Result<Self, String>> {
        let p = sidecar_path(main);
        p.exists().then(|| CmfModel::open(&p).map_err(|e| format!("{}: {e}", p.display())).and_then(|m| Self::new(Arc::new(m))))
    }

    /// The drafter of `main`'s sidecar when there is one: it must have been
    /// converted for this very trunk.
    pub fn for_target(main: &CmfModel) -> Option<Result<Self, String>> {
        let r = Self::beside(&main.path)?;
        Some(r.and_then(|d| {
            let a = main.arch();
            if a.num_layers != d.g.target_layers || a.hidden_size != d.g.hidden || a.vocab_size != d.g.vocab {
                return Err(format!("it drafts for {} layers of {} over {} tokens", d.g.target_layers, d.g.hidden, d.g.vocab));
            }
            let want = d.model.header.provenance.as_ref().and_then(|p| p["base_binding"]["trunk_hash64"].as_str().map(str::to_string));
            let have = format!("{:016x}", main.trunk_hash());
            if want.as_deref() != Some(have.as_str()) {
                return Err(format!("it was converted for the trunk {}, this one is {have}", want.as_deref().unwrap_or("?")));
            }
            Ok(d)
        }))
    }

    pub fn new(model: Arc<CmfModel>) -> Result<Self, String> {
        let g = Geometry::of(&model)?;
        let (hs, ffn, qd, kd) = (g.hidden, g.intermediate, g.heads * g.head_dim, g.kv_heads * g.head_dim);
        let conv = |p: &str| -> Result<Conv, String> {
            Ok(Conv {
                base: vector(&model, &format!("{p}.base_kernel"), 2 * g.conv_kernel * hs)?,
                proj: Small::of(matrix(&model, &format!("{p}.kernel_projection.weight"), 2 * g.conv_kernel * (hs / g.conv_group), hs)?),
            })
        };
        let layers = (0..g.layers)
            .map(|i| {
                let p = format!("layers.{i}");
                Ok(Layer {
                    in_norm: vector(&model, &format!("{p}.input_layernorm.weight"), hs)?,
                    post_norm: vector(&model, &format!("{p}.post_attention_layernorm.weight"), hs)?,
                    q: matrix(&model, &format!("{p}.self_attn.q_proj.weight"), qd, hs)?,
                    k: matrix(&model, &format!("{p}.self_attn.k_proj.weight"), kd, hs)?,
                    v: matrix(&model, &format!("{p}.self_attn.v_proj.weight"), kd, hs)?,
                    o: matrix(&model, &format!("{p}.self_attn.o_proj.weight"), hs, qd)?,
                    q_norm: vector(&model, &format!("{p}.self_attn.q_norm.weight"), g.head_dim)?,
                    k_norm: vector(&model, &format!("{p}.self_attn.k_norm.weight"), g.head_dim)?,
                    gate: matrix(&model, &format!("{p}.mlp.gate_proj.weight"), ffn, hs)?,
                    up: matrix(&model, &format!("{p}.mlp.up_proj.weight"), ffn, hs)?,
                    down: matrix(&model, &format!("{p}.mlp.down_proj.weight"), hs, ffn)?,
                    attn_conv: conv(&format!("{p}.attention_conv"))?,
                    mlp_conv: conv(&format!("{p}.mlp_conv"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let book = |name: &str| -> Result<usize, String> {
            let name = format!("{NAMESPACE}candidate_selector.{name}");
            let i = model.tensor_index(&name).ok_or_else(|| format!("the sidecar has no {name}"))?;
            let e = &model.tensors[i];
            if e.dtype != TensorDtype::F16 || e.shape != [g.vocab, g.selector_rank] {
                return Err(format!("{name}: {:?} {:?}, expected f16 [{}, {}]", e.dtype, e.shape, g.vocab, g.selector_rank));
            }
            Ok(i)
        };
        let half = g.head_dim / 2;
        // a router that does not read is a sidecar without one
        let route = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get(ROUTE_KEY))
            .and_then(|v| serde_json::from_value::<RouteFit>(v.clone()).ok())
            .and_then(|fit| Route::of(&fit, hs).map_err(|e| tracing::warn!("DFlash sidecar: {e}")).ok());
        Ok(Self {
            route,
            fc: matrix(&model, "fc.weight", hs, g.tap_width())?,
            hidden_norm: vector(&model, "hidden_norm.weight", hs)?,
            norm: vector(&model, "norm.weight", hs)?,
            sel: Small::of(matrix(&model, "candidate_selector.hidden_projection.weight", g.selector_rank, hs)?),
            books: [book("predecessor_codebook")?, book("successor_codebook")?],
            inv_freq: (0..half).map(|i| (g.rope_theta.powf(-(2.0 * i as f64) / g.head_dim as f64)) as f32).collect(),
            ctx: (0..g.layers).map(|_| Ctx { k: vec![Vec::new(); g.kv_heads], v: vec![Vec::new(); g.kv_heads], dropped: 0 }).collect(),
            first: 0,
            len: 0,
            pending: Vec::new(),
            device: crate::gpu::enabled(),
            fused: std::env::var("CMF_DFLASH_FUSED").as_deref() != Ok("0"),
            layers,
            model,
            g,
        })
    }

    /// Keep every matrix on the host (a parity check, a device that is busy).
    pub fn host_only(&mut self) {
        self.device = false;
    }

    /// The position the next context row belongs to.
    pub fn next_position(&self) -> usize {
        self.first + self.len + self.pending.len() / self.g.tap_width()
    }

    /// The positions the context covers: its first, and one past its last
    /// (rows that wait to be settled included).
    pub fn span(&self) -> (usize, usize) {
        (self.first, self.next_position())
    }

    pub fn reset(&mut self) {
        self.truncate(0);
        self.first = 0;
    }

    /// Forget the context from position `pos` on (a rewind of the target).
    pub fn truncate(&mut self, pos: usize) {
        let held = self.first + self.len;
        self.pending.truncate(pos.saturating_sub(held) * self.g.tap_width());
        let keep = pos.saturating_sub(self.first).min(self.len);
        let hd = self.g.head_dim;
        for c in &mut self.ctx {
            let rows = keep.saturating_sub(c.dropped);
            for h in c.k.iter_mut().chain(c.v.iter_mut()) {
                h.truncate(rows * hd);
            }
            c.dropped = c.dropped.min(keep);
        }
        self.len = keep;
    }

    /// A layer's keys and values of the positions it holds, for one KV head:
    /// rows of `head_dim`, the last of them the last settled position's.
    pub fn context(&self, layer: usize, kv_head: usize) -> (&[f32], &[f32]) {
        (&self.ctx[layer].k[kv_head], &self.ctx[layer].v[kv_head])
    }

    fn mm(&self, w: &QTensor, x: &[f32], b: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut out = vec![0f32; b * w.rows()];
        if !(self.device && w.device_matmat(x, b, &mut out)) {
            w.matmat(x, b, &mut out, pool);
        }
        out
    }


    /// The sidecar's directory index of a four-bit matrix the device can take.
    fn tiled(&self, w: &QTensor) -> Option<usize> {
        w.mapped_q4tp().filter(|(m, _)| Arc::ptr_eq(m, &self.model)).map(|(_, i)| i)
    }

    /// Several matrices over the same rows: one submission where the device
    /// takes them together (four-bit matrices on Metal), a call each otherwise.
    fn many(&self, ws: &[&QTensor], x: &[f32], b: usize, pool: Option<&Pool>) -> Vec<Vec<f32>> {
        let mut outs: Vec<Vec<f32>> = ws.iter().map(|w| vec![0f32; b * w.rows()]).collect();
        if self.device && self.fused {
            let jobs: Option<Vec<(usize, usize, usize, usize)>> = ws.iter().map(|w| self.tiled(w).map(|i| (i, w.rows(), w.cols(), w.rows()))).collect();
            if let Some(jobs) = jobs {
                let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
                // a block's rows on the narrow kernel, a prompt's on the wide one
                if b <= 8 && crate::gpu::q4tp_rows8_many(&self.model, &jobs, x, b, &mut refs) {
                    return outs;
                }
                let wide: Vec<(usize, usize, usize)> = jobs.iter().map(|j| (j.0, j.1, j.2)).collect();
                if crate::gpu::q4tp_matmat_many(&self.model, &wide, x, b, &mut refs) {
                    return outs;
                }
            }
        }
        for (w, o) in ws.iter().zip(outs.iter_mut()) {
            if !(self.device && w.device_matmat(x, b, o)) {
                w.matmat(x, b, o, pool);
            }
        }
        outs
    }

    /// A layer's MLP, `down · (silu(gate · x) ⊙ up · x)`: one submission where
    /// the device takes the three matrices together.
    fn mlp(&self, l: &Layer, x: &[f32], b: usize, pool: Option<&Pool>) -> Vec<f32> {
        if self.device && self.fused {
            if let (Some(g), Some(u), Some(d)) = (self.tiled(&l.gate), self.tiled(&l.up), self.tiled(&l.down)) {
                let (hs, ffn) = (self.g.hidden, self.g.intermediate);
                let mut out = vec![0f32; b * hs];
                if (b <= 8 && crate::gpu::q4tp_rows8_ffn(&self.model, g, u, d, x, b, hs, ffn, &mut out))
                    || crate::gpu::q4tp_ffn(&self.model, g, u, d, x, b, hs, ffn, &mut out)
                {
                    return out;
                }
            }
        }
        let mut gu = self.many(&[&l.gate, &l.up], x, b, pool);
        let u = gu.pop().unwrap();
        let mut a = gu.pop().unwrap();
        for (a, u) in a.iter_mut().zip(&u) {
            *a = *a / (1.0 + (-*a).exp()) * u;
        }
        self.mm(&l.down, &a, b, pool)
    }

    /// Rotate `x` (rows of `head_dim`, `heads` per position) to the position
    /// of each row: `pos(r)` for the r-th group of `heads` rows.
    fn rotate(&self, x: &mut [f32], heads: usize, pos: impl Fn(usize) -> usize) {
        let (hd, half) = (self.g.head_dim, self.g.head_dim / 2);
        let mut cs = vec![(0f32, 0f32); half];
        for (r, row) in x.chunks_exact_mut(heads * hd).enumerate() {
            let p = pos(r) as f32;
            for (c, f) in cs.iter_mut().zip(&self.inv_freq) {
                let a = p * f;
                *c = (a.cos(), a.sin());
            }
            for h in row.chunks_exact_mut(hd) {
                let (lo, hi) = h.split_at_mut(half);
                for i in 0..half {
                    let (c, s) = cs[i];
                    let (a, b) = (lo[i], hi[i]);
                    lo[i] = a * c - b * s;
                    hi[i] = b * c + a * s;
                }
            }
        }
    }

    /// Add the target's rows of positions `pos ..` to the context: `taps` is
    /// `n` rows of the tapped layers' outputs, joined in the order of
    /// `Geometry::taps`. Rows at positions already held replace them and
    /// what follows; a gap starts the context over.
    pub fn push_context(&mut self, pos: usize, taps: &[f32], pool: Option<&Pool>) {
        self.queue_context(pos, taps, pool);
        self.settle(pool);
    }

    /// `push_context` put off: the rows wait until a block is about to read
    /// them (`settle`), so a target that decodes a stretch without this
    /// drafter pays nothing for them. At most `PENDING_ROWS` wait; past that
    /// the oldest go through the drafter's matrices a batch at a time.
    pub fn queue_context(&mut self, pos: usize, taps: &[f32], pool: Option<&Pool>) {
        let tw = self.g.tap_width();
        assert!(taps.len() % tw == 0, "dflash: a context row is {tw} wide");
        if pos > self.next_position() || pos < self.first {
            self.truncate(0);
            self.first = pos;
        } else {
            self.truncate(pos);
        }
        self.pending.extend_from_slice(taps);
        while self.pending.len() > PENDING_ROWS * tw {
            let rows: Vec<f32> = self.pending.drain(..CONTEXT_CHUNK * tw).collect();
            self.settle_rows(self.first + self.len, &rows, pool);
        }
    }

    /// Make every waiting row context.
    pub fn settle(&mut self, pool: Option<&Pool>) {
        if self.pending.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.pending);
        self.settle_rows(self.first + self.len, &rows, pool);
    }

    fn settle_rows(&mut self, pos: usize, taps: &[f32], pool: Option<&Pool>) {
        let (hs, tw, hd, kv) = (self.g.hidden, self.g.tap_width(), self.g.head_dim, self.g.kv_heads);
        let eps = self.g.eps as f32;
        for (ci, chunk) in taps.chunks(CONTEXT_CHUNK * tw).enumerate() {
            let (n, at) = (chunk.len() / tw, pos + ci * CONTEXT_CHUNK);
            let mut t = self.mm(&self.fc, chunk, n, pool);
            rms(&mut t, &self.hidden_norm, eps);
            debug_assert_eq!(t.len(), n * hs);
            let ws: Vec<&QTensor> = self.layers.iter().flat_map(|l| [&l.k, &l.v]).collect();
            let mut kvs = self.many(&ws, &t, n, pool).into_iter();
            for li in 0..self.layers.len() {
                let l = &self.layers[li];
                let (mut k, v) = (kvs.next().unwrap(), kvs.next().unwrap());
                rms(&mut k, &l.k_norm, eps);
                if self.g.rotary(li) {
                    self.rotate(&mut k, kv, |r| at + r);
                }
                let c = &mut self.ctx[li];
                for r in 0..n {
                    for h in 0..kv {
                        let o = (r * kv + h) * hd;
                        c.k[h].extend_from_slice(&k[o..o + hd]);
                        c.v[h].extend_from_slice(&v[o..o + hd]);
                    }
                }
            }
            self.len += n;
        }
        // no row of a layer sees further back than its window: drop what fell out of it
        for (li, c) in self.ctx.iter_mut().enumerate() {
            let (w, held) = (self.g.window_of(li), self.len - c.dropped);
            if w > 0 && held > 2 * w {
                let cut = held - w;
                for h in c.k.iter_mut().chain(c.v.iter_mut()) {
                    h.drain(..cut * hd);
                }
                c.dropped += cut;
            }
        }
    }

    /// The first stage of a row convolution: the rows convolved, and the
    /// second stage's kernels (computed from the same rows).
    fn conv_in(&self, c: &Conv, x: &[f32], rows: usize, pool: Option<&Pool>) -> (Vec<f32>, Vec<f32>) {
        let d = stage(0, || c.proj.apply(x, rows, pool));
        (self.conv(c, 0, x, &d, rows), d)
    }

    /// `y[r] = Σ_o (base[stage][o] + d[r][stage][o][group]) ⊙ x[r − o]`.
    fn conv(&self, c: &Conv, stage: usize, x: &[f32], d: &[f32], rows: usize) -> Vec<f32> {
        let (hs, gs, kn) = (self.g.hidden, self.g.conv_group, self.g.conv_kernel);
        let groups = hs / gs;
        let mut y = vec![0f32; rows * hs];
        for r in 0..rows {
            let out = &mut y[r * hs..(r + 1) * hs];
            for o in 0..kn.min(r + 1) {
                let src = &x[(r - o) * hs..(r - o + 1) * hs];
                let base = &c.base[(stage * kn + o) * hs..(stage * kn + o + 1) * hs];
                let dyn_ = &d[r * 2 * kn * groups + (stage * kn + o) * groups..][..groups];
                for g in 0..groups {
                    let k = dyn_[g];
                    for i in g * gs..(g + 1) * gs {
                        out[i] += (base[i] + k) * src[i];
                    }
                }
            }
        }
        y
    }

    /// One block: the rows `[anchor, MASK…]` at positions `anchor_pos ..`,
    /// from their embeddings `h` (`block` rows of `hidden`). Returns every
    /// row after the last norm; row j ≥ 1 drafts the token at `anchor_pos + j`.
    pub fn block(&self, anchor_pos: usize, mut h: Vec<f32>, pool: Option<&Pool>) -> Vec<f32> {
        let g = &self.g;
        let (hs, hd, nh, kv, rows) = (g.hidden, g.head_dim, g.heads, g.kv_heads, g.block);
        assert_eq!(h.len(), rows * hs, "dflash: a block is {rows} rows");
        debug_assert!(self.pending.is_empty(), "dflash: rows wait to become context (settle)");
        let (rep, eps, scale) = (nh / kv, g.eps as f32, (hd as f32).powf(-0.5));
        let m = rep * rows;
        let (mut qg, mut sc, mut og) = (vec![0f32; m * hd], Vec::new(), vec![0f32; m * hd]);
        let (mut kb, mut vb) = (vec![0f32; rows * hd], vec![0f32; rows * hd]);
        for (li, l) in self.layers.iter().enumerate() {
            // the layer's context rows before the anchor, as far back as its first row sees
            let (c, window) = (&self.ctx[li], g.window_of(li));
            let base = self.first + c.dropped;
            let hi = anchor_pos.min(self.first + self.len).max(base);
            let lo = if window > 0 { (anchor_pos + 1).saturating_sub(window).clamp(base, hi) } else { base };
            let n = hi - lo;
            let (clo, chi) = ((lo - base) * hd, (hi - base) * hd);
            sc.clear();
            sc.resize(m * (n + rows), 0f32);
            let mut x = h.clone();
            rms(&mut x, &l.in_norm, eps);
            let (x, ad) = self.conv_in(&l.attn_conv, &x, rows, pool);
            let mut qkv = stage(1, || self.many(&[&l.q, &l.k, &l.v], &x, rows, pool));
            let t_attn = stage_timed().then(std::time::Instant::now);
            let v = qkv.pop().unwrap();
            let mut k = qkv.pop().unwrap();
            let mut q = qkv.pop().unwrap();
            rms(&mut q, &l.q_norm, eps);
            rms(&mut k, &l.k_norm, eps);
            if g.rotary(li) {
                self.rotate(&mut q, nh, |r| anchor_pos + r);
                self.rotate(&mut k, kv, |r| anchor_pos + r);
            }
            let mut y = vec![0f32; rows * nh * hd];
            for gi in 0..kv {
                for qi in 0..rep {
                    for r in 0..rows {
                        let src = (r * nh + gi * rep + qi) * hd;
                        qg[(qi * rows + r) * hd..][..hd].copy_from_slice(&q[src..src + hd]);
                    }
                }
                for r in 0..rows {
                    kb[r * hd..][..hd].copy_from_slice(&k[(r * kv + gi) * hd..][..hd]);
                    vb[r * hd..][..hd].copy_from_slice(&v[(r * kv + gi) * hd..][..hd]);
                }
                // scores of every (head, row) over the context and the block's rows
                let w = n + rows;
                if n > 0 {
                    sgemm_public(m, n, hd, scale, &qg, hd, &c.k[gi][clo..chi], hd, true, &mut sc, w);
                }
                sgemm_public(m, rows, hd, scale, &qg, hd, &kb, hd, true, &mut sc[n..], w);
                for i in 0..m {
                    let row = &mut sc[i * w..(i + 1) * w];
                    // a row further on sees less far back
                    if window > 0 {
                        let from = (anchor_pos + i % rows + 1).saturating_sub(window);
                        row[..from.saturating_sub(lo).min(n)].fill(f32::NEG_INFINITY);
                    }
                    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for s in row.iter_mut() {
                        *s = (*s - top).exp();
                        sum += *s;
                    }
                    row.iter_mut().for_each(|s| *s /= sum);
                }
                sgemm_public(m, hd, rows, 1.0, &sc[n..], w, &vb, hd, false, &mut og, hd);
                if n > 0 {
                    let mut oc = vec![0f32; m * hd];
                    sgemm_public(m, hd, n, 1.0, &sc, w, &c.v[gi][clo..chi], hd, false, &mut oc, hd);
                    og.iter_mut().zip(&oc).for_each(|(a, b)| *a += b);
                }
                for qi in 0..rep {
                    for r in 0..rows {
                        let dst = (r * nh + gi * rep + qi) * hd;
                        y[dst..dst + hd].copy_from_slice(&og[(qi * rows + r) * hd..][..hd]);
                    }
                }
            }
            if let Some(t) = t_attn {
                STAGE_US[2].fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            let o = stage(3, || self.many(&[&l.o], &y, rows, pool).pop().unwrap());
            let o = self.conv(&l.attn_conv, 1, &o, &ad, rows);
            h.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            let mut x = h.clone();
            rms(&mut x, &l.post_norm, eps);
            let (x, md) = self.conv_in(&l.mlp_conv, &x, rows, pool);
            let d = stage(4, || self.mlp(l, &x, rows, pool));
            let d = self.conv(&l.mlp_conv, 1, &d, &md, rows);
            h.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        }
        rms(&mut h, &self.norm, eps);
        if stage_timed() {
            STAGE_US[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        h
    }

    fn book_row(&self, book: usize, token: u32, dst: &mut [f32]) {
        let e = &self.model.tensors[self.books[book]];
        let rank = self.g.selector_rank;
        let bytes = &self.model.entry_bytes(e)[token as usize * rank * 2..][..rank * 2];
        for (d, b) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
            *d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        }
    }

    /// The drafted tokens of the rows `hidden` (rows of the block after the
    /// anchor's, after the last norm) from their logits under the target's
    /// head: each row's top candidates re-scored against the token chosen
    /// before it, starting from the anchor. The logits may cover only the
    /// head's first rows (a shortlist of the vocabulary). Returns the chosen
    /// tokens and, for reference, every row's own top token.
    pub fn select(&self, hidden: &[f32], logits: &[f32], anchor: u32) -> (Vec<u32>, Vec<u32>) {
        let (hs, rank, top, vocab) = (self.g.hidden, self.g.selector_rank, self.g.selector_top, self.g.vocab);
        let n = hidden.len() / hs;
        let width = logits.len() / n.max(1);
        assert!(logits.len() == n * width && (top..=vocab).contains(&width), "dflash: logits of {n} rows over the head's rows");
        let hp = self.sel.apply(hidden, n, None);
        let (mut prev, mut chosen, mut plain) = (anchor, Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut pc, mut sc) = (vec![0f32; rank], vec![0f32; rank]);
        for j in 0..n {
            let cand = top_k(&logits[j * width..(j + 1) * width], top);
            plain.push(cand[0].1);
            self.book_row(0, prev, &mut pc);
            pc.iter_mut().zip(&hp[j * rank..(j + 1) * rank]).for_each(|(p, h)| *p *= h);
            let mut best = (f32::NEG_INFINITY, cand[0].1);
            for &(unary, tok) in &cand {
                self.book_row(1, tok, &mut sc);
                let s = unary + pc.iter().zip(&sc).map(|(a, b)| a * b).sum::<f32>();
                if s > best.0 {
                    best = (s, tok);
                }
            }
            prev = best.1;
            chosen.push(prev);
        }
        (chosen, plain)
    }
}

/// The `k` largest of `x` with their indices, largest first.
pub fn top_k(x: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        if best.len() == k && v <= best[k - 1].0 {
            continue;
        }
        let at = best.partition_point(|b| b.0 >= v);
        best.insert(at, (v, i as u32));
        best.truncate(k);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_and_keeps_first_of_equals() {
        let x = [0.5, 2.0, -1.0, 2.0, 3.5, 0.0];
        assert_eq!(top_k(&x, 3), vec![(3.5, 4), (2.0, 1), (2.0, 3)]);
        assert_eq!(top_k(&x, 1), vec![(3.5, 4)]);
    }

    #[test]
    fn sidecar_name() {
        assert_eq!(sidecar_path(Path::new("/m/a.cmf")), PathBuf::from("/m/a.dflash.cmf"));
        assert_eq!(sidecar_path(Path::new("/m/a.dflash.cmf")), PathBuf::from("/m/a.dflash.cmf"));
    }
}
