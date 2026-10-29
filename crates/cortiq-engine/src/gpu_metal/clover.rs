//! Block-diffusion (Clover) passes and commits on native Metal.
//!
//! A preview runs the whole 32-row block through every layer over the
//! committed state without advancing it; a commit runs rows causally and
//! advances it (the prompt in chunks, each resolved block) — `pipeline::
//! clover`. The general Metal graphs cannot take Clover: the rows graph has
//! no MoE arm and binds the file mapping, and a 19 GB file is past the M4
//! 24 GB working set (17.8 GB) — every command buffer would re-wire it. This
//! executor keeps the device's books small instead:
//!
//! * every tensor but the routed experts (the q8_2f attention / Gated
//!   DeltaNet projections, the shared experts, the head: ~2 GB) is bound
//!   through a no-copy view of its own page-rounded span of the mapping;
//! * a routed expert gets its views when a layer's routing first asks for
//!   it, kept in an LRU bounded in bytes (`ExpertViews`,
//!   `CMF_CLOVER_METAL_CACHE_MB`, 6 GB by default). Nothing is copied: the
//!   page cache is the only copy of the weights (owned copies doubled the
//!   footprint on a 24 GB machine and the file paged in from disk on every
//!   pass: 1.3 s against 0.37 s);
//! * a pass is one command buffer per layer: the mixer, the post norm and
//!   the router logits, then ONE wait — the host takes the block's expert
//!   coreset and every row's top-k with the CPU path's own functions — and
//!   the expert panels plus the next layer's mixer in the next buffer.
//!
//! The committed GDN states and K/V rows are uploaded when the pipeline's
//! epoch (request, committed length) changes; a commit advances them on the
//! device and `commit_results` hands them back to the host owners. Kernels:
//! `clover_msl.metal` (preview conv / mixer / attend), the rows graph's
//! causal GDN kernels and the main library's norms, RoPE, attend, panel and
//! combine kernels; q8_2f GEMMs through `moe_q8`'s blob GEMM (half X tiles
//! by default), expert panels through `q4tp_mul_mm_n8` (`n8_max`).

use super::{ctx, moe_q8, wait_fast_checked, Ctx};
use cortiq_core::CmfModel;
use metal::{Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

const CMSL: &str = include_str!("clover_msl.metal");

struct Pipes {
    conv: ComputePipelineState,
    mix: ComputePipelineState,
    attend: ComputePipelineState,
    /// `moe_q8`'s blob GEMM with f32 operand tiles and with half X tiles
    /// (W exact in half, the row scale on the f32 result).
    q8: [ComputePipelineState; 2],
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Result<&'static Pipes, String> {
    PIPES
        .get_or_init(|| {
            let opts = metal::CompileOptions::new();
            opts.set_language_version(metal::MTLLanguageVersion::V3_0);
            let lib = c
                ._device
                .new_library_with_source(CMSL, &opts)
                .map_err(|e| format!("clover MSL compile: {e}"))?;
            let pso = |name: &str| -> Result<ComputePipelineState, String> {
                let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
                c._device
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(|e| format!("pipeline {name}: {e}"))
            };
            let q8 = |mode: u32| -> Result<ComputePipelineState, String> {
                let src = format!("#define Q8MM_MODE {mode}\n{}", moe_q8::QMSL);
                let lib = c
                    ._device
                    .new_library_with_source(&src, &opts)
                    .map_err(|e| format!("q8 blob GEMM (mode {mode}) compile: {e}"))?;
                let f = lib.get_function("q8f_mul_mm_blob", None).map_err(|e| format!("q8f_mul_mm_blob: {e}"))?;
                c._device
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(|e| format!("pipeline q8f_mul_mm_blob: {e}"))
            };
            Ok(Pipes {
                conv: pso("bdp_conv")?,
                mix: pso("bdp_mix")?,
                attend: pso("bdp_attend_blk")?,
                q8: [q8(2)?, q8(1)?],
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Whether the device and every kernel library the executor needs are up
/// (Err says which is not).
pub fn ready() -> Result<(), String> {
    pipes(ctx().ok_or("clover metal: no device")?).map(|_| ())
}

/// Geometry of the Gated DeltaNet layers.
#[derive(Clone, Copy)]
pub struct GdnGeom {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
    pub eps: f32,
    pub gate_sigmoid: bool,
}

impl GdnGeom {
    fn c_dim(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
    fn ring_len(&self) -> usize {
        (self.kk - 1) * self.c_dim()
    }
    fn state_len(&self) -> usize {
        self.ring_len() + self.nv * self.dk * self.dv
    }
}

/// A GDN layer: q8_2f projections by directory index, the host's f32
/// tensors, the Clover preview weights and the committed `[ring | S]`.
pub struct GdnLayer<'a> {
    pub qkv: usize,
    pub z: usize,
    pub out: usize,
    /// in_proj_a and in_proj_b `[nv × hidden]` (b: the commit's β).
    pub a: &'a [f32],
    pub bproj: &'a [f32],
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub conv_right: &'a [f32],
    pub logit_scale: &'a [f32],
    pub history_bias: &'a [f32],
    pub history_c: &'a [f32],
    pub state: &'a [f32],
}

/// A gated full-attention layer and its committed K/V rows (per KV head,
/// `[stored × hd]`).
pub struct AttnLayer<'a> {
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub output_gate: bool,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub rd: usize,
    pub scale: f32,
    pub rope_scale: f32,
    pub inv_freq: &'a [f32],
    pub k: Vec<&'a [f32]>,
    pub v: Vec<&'a [f32]>,
}

pub enum Mixer<'a> {
    Gdn(GdnLayer<'a>),
    Attn(AttnLayer<'a>),
}

/// The routed-experts FFN: f32 router rows, the shared expert's sigmoid
/// gate row, and the q4tp (gate, up, down) directory indices.
pub struct MoeLayer<'a> {
    pub router: &'a [f32],
    pub sgate: &'a [f32],
    pub experts: Vec<(usize, usize, usize)>,
    pub shared: (usize, usize, usize),
}

pub struct Layer<'a> {
    pub input_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub mixer: Mixer<'a>,
    pub moe: MoeLayer<'a>,
}

/// One preview's description.
pub struct Pass<'a> {
    pub model: &'a Arc<CmfModel>,
    pub layers: Vec<Layer<'a>>,
    pub final_norm: &'a [f32],
    /// The q8_2f head and its row count.
    pub lm_head: usize,
    pub lm_rows: usize,
    pub hidden: usize,
    /// Expert (and shared expert) intermediate width.
    pub inter: usize,
    pub eps: f32,
    pub gemma: bool,
    pub gdn: GdnGeom,
    /// (request, committed length): the state the block previews over.
    pub epoch: (u64, usize),
}

/// The host's routing of one layer: (layer, router logits `[b][ne]`, b) →
/// each row's experts and mixing weights.
pub type RouteFn<'r> = dyn FnMut(usize, &[f32], usize) -> Option<Vec<Vec<(usize, f32)>>> + 'r;

/// Where a preview's time went (host side), and the expert cache's traffic.
#[derive(Clone, Copy, Debug, Default)]
pub struct PreviewStats {
    /// Waiting for the device (every layer's buffer, the head).
    pub gpu_s: f64,
    /// Of that, the command buffers' own execution (GPU start → end).
    pub exec_s: f64,
    /// Coreset and routing on the host.
    pub route_s: f64,
    /// Expert slots filled from the mapping.
    pub fill_s: f64,
    pub hits: u64,
    pub misses: u64,
    /// `CMF_CLOVER_METAL_PROF=1`: device execution per stage — GDN mixers,
    /// attention mixers, post norm + router, the expert stage's gather,
    /// gate|up, SiLU·up, down and combine, the head (each stage its own
    /// command buffer, so the wall time inflates).
    pub stage_s: [f64; 9],
}

struct GdnConst {
    a: Buffer,
    bproj: Buffer,
    conv1d: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    norm: Buffer,
    right: Buffer,
    lsc: Buffer,
    hb: Buffer,
    hc: Buffer,
    /// The committed `[ring | S]`, uploaded per epoch.
    state: Buffer,
}

struct AttnConst {
    nkv: usize,
    hd: usize,
    qn: Buffer,
    kn: Buffer,
    invf: Buffer,
    /// `[nkv][cap][hd]` committed rows + the block's rows past them.
    k: Buffer,
    v: Buffer,
    cap: usize,
    uploaded: usize,
}

struct LayerConst {
    in_norm: Buffer,
    post_norm: Buffer,
    /// Router rows plus the shared gate as row `ne`: `[ne + 1][hidden]`.
    router: Buffer,
    gdn: Option<GdnConst>,
    attn: Option<AttnConst>,
}

/// A weight bound for a kernel: a no-copy view and the tensor's offset in it.
type View = (Buffer, u64);

/// A no-copy view of tensor `idx`'s page-rounded span of the mapping.
fn tensor_view(c: &Ctx, model: &CmfModel, idx: usize) -> Result<(View, usize), String> {
    let bytes = model.primary_bytes();
    let page = super::page_size();
    let e = &model.tensors[idx];
    let abs = model
        .entry_abs_offset(e)
        .ok_or_else(|| format!("clover metal: tensor {idx} is outside the primary mapping"))?;
    let len = model.entry_bytes(e).len();
    let (base, end) = (abs / page * page, (abs + len).div_ceil(page) * page);
    if !(bytes.as_ptr() as usize).is_multiple_of(page) || end > bytes.len().div_ceil(page) * page {
        return Err(format!("clover metal: tensor {idx} cannot be viewed"));
    }
    // SAFETY: the span lies inside the mapping, which the model Arc held by
    // the caller's `Pass` keeps alive while the executor holds the view
    // (`Exec` is rebuilt for any other model).
    let buf = unsafe {
        c._device.new_buffer_with_bytes_no_copy(
            bytes.as_ptr().add(base) as *const c_void,
            (end - base) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    };
    Ok(((buf, (abs - base) as u64), end - base))
}

struct ExpertView {
    w: [View; 3],
    bytes: usize,
    used: u64,
}

/// The routed experts' views, least recently used out past `cap` bytes.
struct ExpertViews {
    map: HashMap<(usize, usize), ExpertView>,
    bytes: usize,
    cap: usize,
    tick: u64,
}

impl ExpertViews {
    /// (layer, expert)'s gate / up / down views. A victim is never an
    /// expert the current layer already asked for.
    fn get(
        &mut self,
        c: &Ctx,
        model: &CmfModel,
        layer: usize,
        e: usize,
        trio: (usize, usize, usize),
        st: &mut PreviewStats,
    ) -> Result<[View; 3], String> {
        if let Some(v) = self.map.get_mut(&(layer, e)) {
            v.used = self.tick;
            st.hits += 1;
            return Ok(v.w.clone());
        }
        let t0 = std::time::Instant::now();
        let (g, gb) = tensor_view(c, model, trio.0)?;
        let (u, ub) = tensor_view(c, model, trio.1)?;
        let (d, db) = tensor_view(c, model, trio.2)?;
        let bytes = gb + ub + db;
        while self.bytes + bytes > self.cap {
            let Some(victim) = self
                .map
                .iter()
                .filter(|(_, v)| v.used < self.tick)
                .min_by_key(|(_, v)| v.used)
                .map(|(k, _)| *k)
            else {
                break; // every view belongs to this layer: run over the cap
            };
            let old = self.map.remove(&victim).unwrap();
            self.bytes -= old.bytes;
        }
        let w = [g, u, d];
        self.map.insert((layer, e), ExpertView { w: w.clone(), bytes, used: self.tick });
        self.bytes += bytes;
        st.fill_s += t0.elapsed().as_secs_f64();
        st.misses += 1;
        Ok(w)
    }
}

struct Exec {
    model_uid: u64,
    /// Views of every tensor but the routed experts.
    views: HashMap<usize, View>,
    consts: Vec<LayerConst>,
    final_norm: Buffer,
    /// x pre-scales of 1.0 for the narrow-panel kernels.
    ones: Buffer,
    experts: ExpertViews,
    epoch: Option<(u64, usize)>,
    /// The last commit's (rows, prefix length, GDN state length).
    last_commit: Option<(usize, usize, usize)>,
    scratch: HashMap<&'static str, Buffer>,
}

static EXEC: Mutex<Option<Exec>> = Mutex::new(None);

fn shared_buf(c: &Ctx, bytes: usize) -> Buffer {
    c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

fn f32_buf(c: &Ctx, data: &[f32]) -> Buffer {
    let b = shared_buf(c, data.len() * 4);
    write_f32(&b, 0, data);
    b
}

fn write_f32(b: &Buffer, at: usize, data: &[f32]) {
    // SAFETY: callers size `b` for `at + data.len()` floats.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (b.contents() as *mut f32).add(at), data.len()) }
}

fn set_u32(enc: &ComputeCommandEncoderRef, slot: u64, v: u32) {
    enc.set_bytes(slot, 4, &v as *const u32 as *const c_void);
}

fn set_f32(enc: &ComputeCommandEncoderRef, slot: u64, v: f32) {
    enc.set_bytes(slot, 4, &v as *const f32 as *const c_void);
}

fn bind(enc: &ComputeCommandEncoderRef, bufs: &[(&Buffer, u64)]) {
    for (i, (b, off)) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), *off);
    }
}

impl Exec {
    fn new(c: &Ctx, d: &Pass) -> Result<Exec, String> {
        let model = d.model.as_ref();
        let mut idxs = vec![d.lm_head];
        for l in &d.layers {
            match &l.mixer {
                Mixer::Gdn(g) => idxs.extend([g.qkv, g.z, g.out]),
                Mixer::Attn(a) => idxs.extend([a.wq, a.wk, a.wv, a.wo]),
            }
            idxs.extend([l.moe.shared.0, l.moe.shared.1, l.moe.shared.2]);
        }
        let t0 = std::time::Instant::now();
        let mut views = HashMap::new();
        let mut view_bytes = 0usize;
        for &i in &idxs {
            if let std::collections::hash_map::Entry::Vacant(v) = views.entry(i) {
                let (w, n) = tensor_view(c, model, i)?;
                v.insert(w);
                view_bytes += n;
            }
        }
        let ne = d.layers[0].moe.experts.len();
        let hs = d.hidden;
        let mut consts = Vec::with_capacity(d.layers.len());
        for l in &d.layers {
            let mut router = Vec::with_capacity((ne + 1) * hs);
            router.extend_from_slice(l.moe.router);
            router.extend_from_slice(l.moe.sgate);
            let gdn = match &l.mixer {
                Mixer::Gdn(g) => Some(GdnConst {
                    a: f32_buf(c, g.a),
                    bproj: f32_buf(c, g.bproj),
                    conv1d: f32_buf(c, g.conv1d),
                    a_log: f32_buf(c, g.a_log),
                    dt_bias: f32_buf(c, g.dt_bias),
                    norm: f32_buf(c, g.norm),
                    right: f32_buf(c, g.conv_right),
                    lsc: f32_buf(c, g.logit_scale),
                    hb: f32_buf(c, g.history_bias),
                    hc: f32_buf(c, g.history_c),
                    state: shared_buf(c, d.gdn.state_len() * 4),
                }),
                Mixer::Attn(_) => None,
            };
            let attn = match &l.mixer {
                Mixer::Attn(a) => Some(AttnConst {
                    nkv: a.nkv,
                    hd: a.hd,
                    qn: f32_buf(c, a.q_norm.unwrap_or(&[0.0])),
                    kn: f32_buf(c, a.k_norm.unwrap_or(&[0.0])),
                    invf: f32_buf(c, a.inv_freq),
                    k: shared_buf(c, 16),
                    v: shared_buf(c, 16),
                    cap: 0,
                    uploaded: 0,
                }),
                Mixer::Gdn(_) => None,
            };
            consts.push(LayerConst {
                in_norm: f32_buf(c, l.input_norm),
                post_norm: f32_buf(c, l.post_norm),
                router: f32_buf(c, &router),
                gdn,
                attn,
            });
        }
        let cap_mb = std::env::var("CMF_CLOVER_METAL_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6144);
        tracing::info!(
            "clover metal: {} MB of attention / DeltaNet / shared / head weights in {} views ({:.2} s), expert views up to {} MB",
            view_bytes >> 20,
            views.len(),
            t0.elapsed().as_secs_f64(),
            cap_mb
        );
        Ok(Exec {
            model_uid: model.uid(),
            views,
            consts,
            final_norm: f32_buf(c, d.final_norm),
            ones: f32_buf(c, &[1.0; 8]),
            experts: ExpertViews { map: HashMap::new(), bytes: 0, cap: cap_mb << 20, tick: 1 },
            epoch: None,
            last_commit: None,
            scratch: HashMap::new(),
        })
    }

    /// A scratch buffer of at least `bytes`, kept across passes.
    fn scratch(&mut self, c: &Ctx, name: &'static str, bytes: usize) -> Buffer {
        let e = self.scratch.entry(name).or_insert_with(|| shared_buf(c, bytes));
        if (e.length() as usize) < bytes {
            *e = shared_buf(c, bytes);
        }
        e.clone()
    }

    fn view(&self, idx: usize) -> Result<View, String> {
        self.views.get(&idx).cloned().ok_or_else(|| format!("clover metal: tensor {idx} has no view"))
    }

    /// The committed GDN states and K/V rows on the device for this epoch,
    /// with room for `b` block rows past the prefix.
    fn sync(&mut self, c: &Ctx, d: &Pass, b: usize) -> Result<(), String> {
        let new_request = self.epoch.is_none_or(|(r, _)| r != d.epoch.0);
        let changed = self.epoch != Some(d.epoch);
        for (l, k) in d.layers.iter().zip(self.consts.iter_mut()) {
            match (&l.mixer, k.gdn.as_ref(), k.attn.as_mut()) {
                (Mixer::Gdn(g), Some(gc), _) if changed => {
                    if g.state.len() == d.gdn.state_len() {
                        write_f32(&gc.state, 0, g.state);
                    } else {
                        // a fresh layer reads a zero state
                        unsafe { std::ptr::write_bytes(gc.state.contents() as *mut u8, 0, d.gdn.state_len() * 4) };
                    }
                }
                (Mixer::Attn(a), _, Some(ac)) => {
                    let stored = a.k.first().map_or(0, |k| k.len() / a.hd);
                    if a.k.len() != a.nkv || a.v.len() != a.nkv || a.k.iter().chain(&a.v).any(|x| x.len() != stored * a.hd) {
                        return Err("clover metal: K/V heads of unequal length".into());
                    }
                    if new_request || stored < ac.uploaded {
                        ac.uploaded = 0;
                    }
                    if stored + b > ac.cap {
                        let cap = (stored + b).next_power_of_two().max(1024);
                        ac.k = shared_buf(c, a.nkv * cap * a.hd * 4);
                        ac.v = shared_buf(c, a.nkv * cap * a.hd * 4);
                        ac.cap = cap;
                        ac.uploaded = 0;
                    }
                    for h in 0..a.nkv {
                        let (lo, hi) = (ac.uploaded * a.hd, stored * a.hd);
                        write_f32(&ac.k, h * ac.cap * a.hd + lo, &a.k[h][lo..hi]);
                        write_f32(&ac.v, h * ac.cap * a.hd + lo, &a.v[h][lo..hi]);
                    }
                    ac.uploaded = stored;
                }
                _ => {}
            }
        }
        self.epoch = Some(d.epoch);
        Ok(())
    }
}

/// `CMF_CLOVER_SKIP=<letters>`: leave parts of a pass out (cost attribution
/// by elimination; the output is garbage while set): q GDN qkv/z GEMMs,
/// m GDN preview mixer, o GDN out GEMM.
fn skip(what: char) -> bool {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| std::env::var("CMF_CLOVER_SKIP").unwrap_or_default()).contains(what)
}

/// Which q8 blob GEMM the projections and the head take (`CMF_CLOVER_Q8=
/// half` | `f32`). Half X tiles by default: on the M4 the head 11 against
/// 14 ms, and the first pass agrees with the PyTorch release as well
/// (top-1 31/32 both, mean top-1 logit gap 0.21 against 0.31).
fn q8_mode() -> usize {
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("CMF_CLOVER_Q8").as_deref() {
        Ok("f32") => 0,
        _ => 1,
    })
}

/// q8_2f GEMM of `b` rows over a blob view (`q8f_mul_mm_blob`, the weight's
/// input field applied to X as it is staged).
#[allow(clippy::too_many_arguments)]
fn q8mm(p: &Pipes, enc: &ComputeCommandEncoderRef, w: View, x: &Buffer, y: &Buffer, b: usize, rows: usize, cols: usize) {
    enc.set_compute_pipeline_state(&p.q8[q8_mode()]);
    enc.set_buffer(0, Some(&w.0), w.1);
    enc.set_buffer(1, Some(x), 0);
    enc.set_buffer(2, Some(y), 0);
    set_u32(enc, 3, cols as u32);
    set_u32(enc, 4, rows as u32);
    set_u32(enc, 5, b as u32);
    set_u32(enc, 6, 1);
    enc.dispatch_thread_groups(
        MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
        MTLSize::new(128, 1, 1),
    );
}

/// q4tp GEMM (`q4tp_mul_mm`) of `nb` packed rows from row `row0`, weight
/// at an explicit buffer offset.
#[allow(clippy::too_many_arguments)]
fn q4mm(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    w: &Buffer,
    w_off: u64,
    x: &Buffer,
    y: &Buffer,
    row0: usize,
    nb: usize,
    rows: usize,
    cols: usize,
) {
    enc.set_compute_pipeline_state(&c.q4tpmm);
    enc.set_buffer(0, Some(w), w_off);
    enc.set_buffer(1, Some(x), (row0 * cols * 4) as u64);
    enc.set_buffer(2, Some(y), (row0 * rows * 4) as u64);
    set_u32(enc, 3, cols as u32);
    set_u32(enc, 4, rows as u32);
    set_u32(enc, 5, nb as u32);
    set_f32(enc, 6, 1.0);
    enc.dispatch_thread_groups(
        MTLSize::new((nb as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
        MTLSize::new(128, 1, 1),
    );
}

/// The widest panel the eight-row kernel takes (`CMF_CLOVER_N8_MAX`, 32:
/// every panel — on the M4 eight-row passes beat the 32-row tile GEMM even
/// for the shared expert's full block, and 32- and 16-row variants that
/// unpack each weight tile once measured slower still: gate|up 98–108 and
/// 74–81 against 67–75 ms a pass).
fn n8_max() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_CLOVER_N8_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}

/// One expert panel's GEMM: `nr` packed rows from `row0` — eight rows a
/// `q4tp_mul_mm_n8` dispatch, or the 32-row tile GEMM for a wide panel.
#[allow(clippy::too_many_arguments)]
fn q4rows(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    w: &View,
    ones: &Buffer,
    x: &Buffer,
    y: &Buffer,
    row0: usize,
    nr: usize,
    rows: usize,
    cols: usize,
) {
    if nr > n8_max() {
        q4mm(c, enc, &w.0, w.1, x, y, row0, nr, rows, cols);
        return;
    }
    let mut r = 0;
    while r < nr {
        let nb = (nr - r).min(8);
        enc.set_compute_pipeline_state(&c.q4tpmmn8);
        enc.set_buffer(0, Some(&w.0), w.1);
        enc.set_buffer(1, Some(x), ((row0 + r) * cols * 4) as u64);
        enc.set_buffer(2, Some(y), ((row0 + r) * rows * 4) as u64);
        set_u32(enc, 3, cols as u32);
        set_u32(enc, 4, rows as u32);
        set_u32(enc, 5, nb as u32);
        enc.set_buffer(6, Some(ones), 0);
        enc.dispatch_thread_groups(MTLSize::new((rows as u64).div_ceil(64), 1, 1), MTLSize::new(128, 1, 1));
        r += nb;
    }
}

/// `rmsnorm_rows` (`d` None) or `add_rmsnorm_rows` (h += d first) over b rows.
#[allow(clippy::too_many_arguments)]
fn norm_rows(
    c: &Ctx,
    enc: &ComputeCommandEncoderRef,
    h: &Buffer,
    d: Option<&Buffer>,
    w: &Buffer,
    o: &Buffer,
    b: usize,
    n: usize,
    gemma: bool,
    eps: f32,
) {
    match d {
        None => {
            enc.set_compute_pipeline_state(&c.rmsrows);
            bind(enc, &[(h, 0), (w, 0), (o, 0)]);
            set_u32(enc, 3, n as u32);
            set_u32(enc, 4, gemma as u32);
            set_f32(enc, 5, eps);
        }
        Some(d) => {
            enc.set_compute_pipeline_state(&c.addnorm);
            bind(enc, &[(h, 0), (d, 0), (w, 0), (o, 0)]);
            set_u32(enc, 4, n as u32);
            set_u32(enc, 5, gemma as u32);
            set_f32(enc, 6, eps);
            set_u32(enc, 7, 1);
        }
    }
    enc.dispatch_thread_groups(MTLSize::new(b as u64, 1, 1), MTLSize::new(256, 1, 1));
}

/// `f32_mv_rows8`: y[b][rows] = W[rows][cols] · x[b][cols].
#[allow(clippy::too_many_arguments)]
fn f32mv(c: &Ctx, enc: &ComputeCommandEncoderRef, w: &Buffer, x: &Buffer, y: &Buffer, b: usize, rows: usize, cols: usize) {
    enc.set_compute_pipeline_state(&c.f32mvr8);
    bind(enc, &[(w, 0), (x, 0), (y, 0)]);
    set_u32(enc, 3, cols as u32);
    set_u32(enc, 4, rows as u32);
    set_u32(enc, 5, b as u32);
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(8), (b as u64).div_ceil(8), 1),
        MTLSize::new(256, 1, 1),
    );
}

fn commit_wait(cmd: &CommandBuffer, st: &mut PreviewStats) -> Result<(), String> {
    let t = std::time::Instant::now();
    cmd.commit();
    let r = wait_fast_checked(cmd);
    st.gpu_s += t.elapsed().as_secs_f64();
    st.exec_s += super::cmd_gpu_ms(cmd) / 1e3;
    r
}

/// Checks the shapes the kernels are written for (the preview mixer holds
/// the block's keys in threadgroup memory: kb·dk ≤ 4096).
fn fits(d: &Pass, b: usize, causal: bool) -> Result<(), String> {
    let g = d.gdn;
    let ok = (1..=256).contains(&b)
        && d.hidden.is_multiple_of(32)
        && !d.layers.is_empty()
        && g.dk <= 256
        && (causal || b * g.dk <= 4096)
        && g.dv <= 256
        && 256 % g.dv == 0
        && g.nk > 0
        && g.nv.is_multiple_of(g.nk)
        && g.kk >= 2
        && g.kk <= 8
        && (g.dk == 64 || g.dk == 128)
        && d.layers.iter().all(|l| {
            l.moe.experts.len() == d.layers[0].moe.experts.len()
                && l.moe.router.len() == l.moe.experts.len() * d.hidden
                && l.moe.sgate.len() == d.hidden
                && match &l.mixer {
                    Mixer::Gdn(x) => x.a.len() == g.nv * d.hidden && x.bproj.len() == g.nv * d.hidden,
                    Mixer::Attn(a) => {
                        a.hd <= 256
                            && a.hd % 32 == 0
                            && a.rd <= a.hd
                            && (a.rd / 2) % 32 == 0
                            && a.nkv > 0
                            && a.nh % a.nkv == 0
                            && a.nh / a.nkv <= 8
                    }
                }
        });
    if ok {
        Ok(())
    } else {
        Err("clover metal: a layer's shape is outside the preview kernels".into())
    }
}

/// One denoising pass: the block's `b` embedded rows `h0` at absolute
/// position `pos` over the committed state of `d.epoch`; `logits` comes
/// back `[b][lm_rows]`.
pub fn preview(
    d: &Pass,
    h0: &[f32],
    b: usize,
    pos: usize,
    route: &mut RouteFn<'_>,
    logits: &mut Vec<f32>,
    st: &mut PreviewStats,
) -> Result<(), String> {
    run(d, h0, b, pos, false, route, Some(logits), st)
}

/// Commit `b` rows at `pos` causally (the prompt in chunks, a resolved
/// block): the device's GDN states and K/V rows advance; `commit_results`
/// then hands them to the host owners. A failure leaves the device state
/// unknown — the next call re-uploads it.
pub fn commit(d: &Pass, h0: &[f32], b: usize, pos: usize, route: &mut RouteFn<'_>, st: &mut PreviewStats) -> Result<(), String> {
    run(d, h0, b, pos, true, route, None, st)
}

/// `commit` that also runs the last FFN and the head: the causal logits of
/// the rows, `[b][lm_rows]` (perplexity on the commit path).
pub fn commit_scored(
    d: &Pass,
    h0: &[f32],
    b: usize,
    pos: usize,
    route: &mut RouteFn<'_>,
    logits: &mut Vec<f32>,
    st: &mut PreviewStats,
) -> Result<(), String> {
    run(d, h0, b, pos, true, route, Some(logits), st)
}

/// What `commit_results` hands the host for one layer.
pub enum CommitPart<'s> {
    /// A GDN layer's new `[ring | S]`.
    State(&'s [f32]),
    /// An attention layer's appended K and V rows, `[b][nkv·hd]` each.
    Kv(&'s [f32], &'s [f32]),
}

/// After `commit`: every layer's new state or K/V rows, to the host owners.
pub fn commit_results(out: &mut dyn FnMut(usize, CommitPart<'_>)) -> Result<(), String> {
    let guard = EXEC.lock().unwrap();
    let ex = guard.as_ref().ok_or("clover metal: no executor")?;
    let (b, stored, state_len) = ex.last_commit.ok_or("clover metal: no commit to read")?;
    for (li, k) in ex.consts.iter().enumerate() {
        if let Some(gc) = &k.gdn {
            // SAFETY: the buffer holds state_len floats; the commit completed.
            out(li, CommitPart::State(unsafe { std::slice::from_raw_parts(gc.state.contents() as *const f32, state_len) }));
        }
        if let Some(ac) = &k.attn {
            let (nkv, hd) = (ac.nkv, ac.hd);
            let (mut kr, mut vr) = (vec![0f32; b * nkv * hd], vec![0f32; b * nkv * hd]);
            for (src, dst) in [(&ac.k, &mut kr), (&ac.v, &mut vr)] {
                let p = src.contents() as *const f32;
                for r in 0..b {
                    for h in 0..nkv {
                        // SAFETY: rows stored..stored+b of head h lie inside the cap.
                        let row = unsafe { std::slice::from_raw_parts(p.add((h * ac.cap + stored + r) * hd), hd) };
                        dst[(r * nkv + h) * hd..(r * nkv + h + 1) * hd].copy_from_slice(row);
                    }
                }
            }
            out(li, CommitPart::Kv(&kr, &vr));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run(
    d: &Pass,
    h0: &[f32],
    b: usize,
    pos: usize,
    causal: bool,
    route: &mut RouteFn<'_>,
    logits: Option<&mut Vec<f32>>,
    st: &mut PreviewStats,
) -> Result<(), String> {
    let c = ctx().ok_or("clover metal: no device")?;
    let p = pipes(c)?;
    fits(d, b, causal)?;
    let hs = d.hidden;
    if h0.len() != b * hs {
        return Err("clover metal: hidden rows of the wrong size".into());
    }
    let mut guard = EXEC.lock().unwrap();
    if guard.as_ref().is_none_or(|e| e.model_uid != d.model.uid() || e.consts.len() != d.layers.len()) {
        *guard = None;
        *guard = Some(Exec::new(c, d)?);
    }
    let ex = guard.as_mut().unwrap();
    ex.sync(c, d, b)?;
    let r = run_on(c, p, ex, d, h0, b, pos, causal, route, logits, st);
    if r.is_err() && causal {
        // the states may be half-advanced: upload everything again
        ex.epoch = None;
        for k in &mut ex.consts {
            if let Some(ac) = k.attn.as_mut() {
                ac.uploaded = 0;
            }
        }
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn run_on(
    c: &Ctx,
    p: &Pipes,
    ex: &mut Exec,
    d: &Pass,
    h0: &[f32],
    b: usize,
    pos: usize,
    causal: bool,
    route: &mut RouteFn<'_>,
    logits: Option<&mut Vec<f32>>,
    st: &mut PreviewStats,
) -> Result<(), String> {
    let hs = d.hidden;
    let g = d.gdn;
    let (cdim, vd, kd) = (g.c_dim(), g.nv * g.dv, g.nk * g.dk);
    let ne = d.layers[0].moe.experts.len();
    let inter = d.inter;
    let gemma = d.gemma;
    // row-scaled scratch
    let h = ex.scratch(c, "h", b * hs * 4);
    let n = ex.scratch(c, "n", b * hs * 4);
    let dl = ex.scratch(c, "d", b * hs * 4);
    let lg = ex.scratch(c, "lg", b * (ne + 1) * 4);
    write_f32(&h, 0, h0);
    let max_rows = |pick: fn(&Mixer) -> usize| d.layers.iter().map(|l| pick(&l.mixer)).max().unwrap_or(0);
    let wq_rows = max_rows(|m| match m {
        Mixer::Attn(a) => a.nh * a.hd * if a.output_gate { 2 } else { 1 },
        _ => 0,
    });
    let nhd = max_rows(|m| match m {
        Mixer::Attn(a) => a.nh * a.hd,
        _ => 0,
    });
    let kvd = max_rows(|m| match m {
        Mixer::Attn(a) => a.nkv * a.hd,
        _ => 0,
    });
    let nblk_max = d
        .layers
        .iter()
        .filter_map(|l| match &l.mixer {
            Mixer::Attn(a) => Some((a.k.first().map_or(0, |k| k.len() / a.hd) + b).div_ceil(128) * a.nh),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let qkv_b = ex.scratch(c, "qkv", b * cdim * 4);
    let cq_b = ex.scratch(c, "cq", b * cdim * 4);
    let z_b = ex.scratch(c, "z", b * vd * 4);
    let a_b = ex.scratch(c, "a", b * g.nv * 4);
    let o_b = ex.scratch(c, "o", b * vd * 4);
    let q_b = ex.scratch(c, "q", b * wq_rows * 4);
    let k_b = ex.scratch(c, "k", b * kvd * 4);
    let v_b = ex.scratch(c, "v", b * kvd * 4);
    let qr_b = ex.scratch(c, "qr", b * nhd * 4);
    let gt_b = ex.scratch(c, "gt", b * nhd * 4);
    let ao_b = ex.scratch(c, "ao", b * nhd * 4);
    let part_b = ex.scratch(c, "part", b * nblk_max * (256 + 2) * 4);
    let bb_b = ex.scratch(c, "bb", b * g.nv * 4);
    let gg_b = ex.scratch(c, "gg", b * g.nv * 4);
    let bt_b = ex.scratch(c, "bt", b * g.nv * 4);
    let iq_b = ex.scratch(c, "iq", b * g.nk * 4);
    let ik_b = ex.scratch(c, "ik", b * g.nk * 4);
    let gcfg = super::GdnGpuCfg {
        nv: g.nv,
        nk: g.nk,
        dk: g.dk,
        dv: g.dv,
        kk: g.kk,
        hidden: hs,
        inter: d.inter,
        c_dim: cdim,
        eps: g.eps,
        gemma: d.gemma,
    };

    let prof = std::env::var("CMF_CLOVER_METAL_PROF").is_ok_and(|v| v == "1");
    // profiling: close the stage's command buffer and book its device time
    let cut = |cmd: &mut CommandBuffer, st: &mut PreviewStats, slot: usize| -> Result<(), String> {
        if prof {
            commit_wait(cmd, st)?;
            st.stage_s[slot] += super::cmd_gpu_ms(cmd) / 1e3;
            *cmd = c.queue.new_command_buffer().to_owned();
        }
        Ok(())
    };
    let mut cmd = c.queue.new_command_buffer().to_owned();
    for (li, l) in d.layers.iter().enumerate() {
        let k = &ex.consts[li];
        // ── mixer, post norm, router logits ──
        let mut enc = cmd.new_compute_command_encoder();
        norm_rows(c, enc, &h, (li > 0).then_some(&dl), &k.in_norm, &n, b, hs, gemma, d.eps);
        match (&l.mixer, k.gdn.as_ref(), k.attn.as_ref()) {
            (Mixer::Gdn(gl), Some(gc), _) if causal => {
                // the causal step of the trunk (the rows graph's kernels):
                // conv over [ring | rows], gates, q/k norms, the delta rule
                // writing S, the ring shifted, out_proj
                q8mm(p, enc, ex.view(gl.qkv)?, &n, &qkv_b, b, cdim, hs);
                q8mm(p, enc, ex.view(gl.z)?, &n, &z_b, b, vd, hs);
                f32mv(c, enc, &gc.a, &n, &a_b, b, g.nv, hs);
                f32mv(c, enc, &gc.bproj, &n, &bb_b, b, g.nv, hs);
                enc.set_compute_pipeline_state(&c.gdnconvb);
                bind(enc, &[(&qkv_b, 0), (&gc.state, 0), (&gc.conv1d, 0), (&cq_b, 0)]);
                set_u32(enc, 4, cdim as u32);
                set_u32(enc, 5, g.kk as u32);
                set_u32(enc, 6, b as u32);
                enc.dispatch_threads(MTLSize::new(cdim as u64, b as u64, 1), MTLSize::new(256, 1, 1));
                enc.set_compute_pipeline_state(&c.gdngatesb);
                bind(enc, &[(&a_b, 0), (&bb_b, 0), (&gc.a_log, 0), (&gc.dt_bias, 0), (&gg_b, 0), (&bt_b, 0)]);
                set_u32(enc, 6, g.nv as u32);
                set_u32(enc, 7, (b * g.nv) as u32);
                enc.dispatch_threads(MTLSize::new((b * g.nv) as u64, 1, 1), MTLSize::new(64, 1, 1));
                enc.set_compute_pipeline_state(&c.gdnqknb);
                bind(enc, &[(&cq_b, 0), (&iq_b, 0), (&ik_b, 0)]);
                for (i, v) in [g.nk, g.dk, cdim, b].iter().enumerate() {
                    set_u32(enc, 3 + i as u64, *v as u32);
                }
                enc.dispatch_threads(MTLSize::new(((b * g.nk) as u64).div_ceil(8) * 256, 1, 1), MTLSize::new(256, 1, 1));
                super::encode_gdn_state_b(
                    c,
                    enc,
                    &gc.state,
                    (g.ring_len() * 4) as u64,
                    &cq_b,
                    &z_b,
                    &gg_b,
                    &bt_b,
                    &iq_b,
                    &ik_b,
                    &gc.norm,
                    &o_b,
                    &gcfg,
                    b,
                    3,
                    !super::state4_on(),
                );
                enc.set_compute_pipeline_state(&c.gdnringcb);
                bind(enc, &[(&gc.state, 0), (&qkv_b, 0)]);
                set_u32(enc, 2, cdim as u32);
                set_u32(enc, 3, g.kk as u32);
                set_u32(enc, 4, b as u32);
                enc.dispatch_threads(MTLSize::new(cdim as u64, 1, 1), MTLSize::new(256, 1, 1));
                q8mm(p, enc, ex.view(gl.out)?, &o_b, &dl, b, hs, vd);
            }
            (Mixer::Gdn(gl), Some(gc), _) => {
                if !skip('q') {
                    q8mm(p, enc, ex.view(gl.qkv)?, &n, &qkv_b, b, cdim, hs);
                    q8mm(p, enc, ex.view(gl.z)?, &n, &z_b, b, vd, hs);
                }
                f32mv(c, enc, &gc.a, &n, &a_b, b, g.nv, hs);
                enc.set_compute_pipeline_state(&p.conv);
                bind(enc, &[(&qkv_b, 0), (&gc.conv1d, 0), (&gc.right, 0), (&gc.state, 0), (&cq_b, 0)]);
                set_u32(enc, 5, cdim as u32);
                set_u32(enc, 6, g.kk as u32);
                set_u32(enc, 7, b as u32);
                enc.dispatch_threads(MTLSize::new(cdim as u64, b as u64, 1), MTLSize::new(256, 1, 1));
                if !skip('m') {
                enc.set_compute_pipeline_state(&p.mix);
                bind(
                    enc,
                    &[
                        (&cq_b, 0),
                        (&z_b, 0),
                        (&a_b, 0),
                        (&gc.a_log, 0),
                        (&gc.dt_bias, 0),
                        (&gc.norm, 0),
                        (&gc.state, (g.ring_len() * 4) as u64),
                        (&o_b, 0),
                        (&gc.lsc, 0),
                        (&gc.hb, 0),
                        (&gc.hc, 0),
                    ],
                );
                for (i, v) in [g.nv, g.dk, g.dv, kd, g.nv / g.nk, cdim, b, g.gate_sigmoid as usize].iter().enumerate() {
                    set_u32(enc, 11 + i as u64, *v as u32);
                }
                set_f32(enc, 19, g.eps);
                set_f32(enc, 20, (b as f64 / 32.0).ln() as f32);
                enc.dispatch_thread_groups(
                    MTLSize::new(g.nv as u64, (b as u64).div_ceil(4), 1),
                    MTLSize::new(256, 1, 1),
                );
                }
                if !skip('o') {
                    q8mm(p, enc, ex.view(gl.out)?, &o_b, &dl, b, hs, vd);
                }
            }
            (Mixer::Attn(al), _, Some(ac)) => {
                let (nh, nkv, hd) = (al.nh, al.nkv, al.hd);
                let stored = ac.uploaded;
                q8mm(p, enc, ex.view(al.wq)?, &n, &q_b, b, nh * hd * if al.output_gate { 2 } else { 1 }, hs);
                q8mm(p, enc, ex.view(al.wk)?, &n, &k_b, b, nkv * hd, hs);
                q8mm(p, enc, ex.view(al.wv)?, &n, &v_b, b, nkv * hd, hs);
                let flags = (al.output_gate as u32)
                    | ((al.q_norm.is_some() as u32) << 1)
                    | ((al.k_norm.is_some() as u32) << 2)
                    | ((gemma as u32) << 3);
                enc.set_compute_pipeline_state(&c.rqknb);
                bind(enc, &[(&q_b, 0), (&k_b, 0), (&qr_b, 0), (&gt_b, 0), (&ac.qn, 0), (&ac.kn, 0), (&ac.invf, 0)]);
                for (i, v) in [nh, nkv, hd, al.rd, pos, flags as usize].iter().enumerate() {
                    set_u32(enc, 7 + i as u64, *v as u32);
                }
                set_f32(enc, 13, d.eps);
                set_u32(enc, 14, b as u32);
                set_f32(enc, 15, al.rope_scale);
                enc.dispatch_threads(MTLSize::new(((nh + nkv) * 32) as u64, b as u64, 1), MTLSize::new(256, 1, 1));
                // the block's rows past the committed prefix
                enc.set_compute_pipeline_state(&c.kvappb);
                bind(enc, &[(&k_b, 0), (&v_b, 0), (&ac.k, 0), (&ac.v, 0)]);
                for (i, v) in [nkv, hd, ac.cap, stored, b].iter().enumerate() {
                    set_u32(enc, 4 + i as u64, *v as u32);
                }
                enc.dispatch_threads(MTLSize::new((nkv * hd) as u64, b as u64, 1), MTLSize::new(256, 1, 1));
                let n_all = stored + b;
                let nblk = n_all.div_ceil(128);
                let hpk = nh / nkv;
                if causal {
                    // row e over the prefix and rows ..=e
                    if !super::encode_gqa_attend_blk(
                        c, enc, &qr_b, &ac.k, &ac.v, 0, &ao_b, nh, nkv, hd, ac.cap, stored, b, al.scale, false,
                    ) {
                        return Err("clover metal: the causal attend refused the layer".into());
                    }
                } else {
                enc.set_compute_pipeline_state(&p.attend);
                bind(enc, &[(&qr_b, 0), (&ac.k, 0), (&ac.v, 0), (&part_b, 0)]);
                for (i, v) in [nh, hpk, hd, ac.cap, n_all, b, nblk].iter().enumerate() {
                    set_u32(enc, 4 + i as u64, *v as u32);
                }
                set_f32(enc, 11, al.scale);
                enc.dispatch_thread_groups(
                    MTLSize::new(nkv as u64, nblk as u64, b as u64),
                    MTLSize::new((hpk * 32) as u64, 1, 1),
                );
                enc.set_compute_pipeline_state(&c.gqacomb);
                bind(enc, &[(&part_b, 0), (&ao_b, 0)]);
                for (i, v) in [nh, hd, b, nblk].iter().enumerate() {
                    set_u32(enc, 2 + i as u64, *v as u32);
                }
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, b as u64, 1), MTLSize::new(hd as u64, 1, 1));
                }
                if al.output_gate {
                    enc.set_compute_pipeline_state(&c.sgate);
                    bind(enc, &[(&ao_b, 0), (&gt_b, 0)]);
                    set_u32(enc, 2, (b * nh * hd) as u32);
                    enc.dispatch_threads(MTLSize::new((b * nh * hd) as u64, 1, 1), MTLSize::new(256, 1, 1));
                }
                q8mm(p, enc, ex.view(al.wo)?, &ao_b, &dl, b, hs, nh * hd);
            }
            _ => return Err(format!("clover metal: layer {li} has no device constants for its mixer")),
        }
        if causal && logits.is_none() && li + 1 == d.layers.len() {
            // a commit needs the states and K/V rows, not the last FFN
            enc.end_encoding();
            break;
        }
        if prof {
            enc.end_encoding();
            commit_wait(&cmd, st)?;
            st.stage_s[if matches!(l.mixer, Mixer::Gdn(_)) { 0 } else { 1 }] += super::cmd_gpu_ms(&cmd) / 1e3;
            cmd = c.queue.new_command_buffer().to_owned();
            enc = cmd.new_compute_command_encoder();
        }
        if prof {
            st.stage_s[2] += super::cmd_gpu_ms(&cmd) / 1e3;
        }
        norm_rows(c, enc, &h, Some(&dl), &k.post_norm, &n, b, hs, gemma, d.eps);
        f32mv(c, enc, &k.router, &n, &lg, b, ne + 1, hs);
        enc.end_encoding();
        commit_wait(&cmd, st)?;
        cmd = c.queue.new_command_buffer().to_owned();

        // ── host: coreset, routes, expert slots ──
        let t_route = std::time::Instant::now();
        // SAFETY: the buffer holds b·(ne+1) floats the device just wrote.
        let lgx = unsafe { std::slice::from_raw_parts(lg.contents() as *const f32, b * (ne + 1)) };
        let mut logits_ne = Vec::with_capacity(b * ne);
        for r in 0..b {
            logits_ne.extend_from_slice(&lgx[r * (ne + 1)..r * (ne + 1) + ne]);
        }
        let routes = route(li, &logits_ne, b).ok_or("clover metal: routing failed")?;
        if routes.len() != b {
            return Err("clover metal: routing returned the wrong row count".into());
        }
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        for (r, row) in routes.iter().enumerate() {
            for &(e, w) in row {
                if e >= ne {
                    return Err("clover metal: routed expert out of range".into());
                }
                assign[e].push((r, w));
            }
        }
        st.route_s += t_route.elapsed().as_secs_f64();
        ex.experts.tick += 1;
        // panels: the routed experts in ascending order, then the shared one
        let mut panels: Vec<([View; 3], usize, usize)> = Vec::new(); // (gate/up/down, row0, rows)
        let mut gidx: Vec<u32> = Vec::new();
        let mut slots: Vec<Vec<(u32, f32)>> = vec![Vec::new(); b];
        for (e, list) in assign.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let w = ex.experts.get(c, d.model, li, e, l.moe.experts[e], st)?;
            let row0 = gidx.len();
            for &(r, w) in list {
                slots[r].push((gidx.len() as u32, w));
                gidx.push(r as u32);
            }
            panels.push((w, row0, list.len()));
        }
        let row0 = gidx.len();
        for (r, slot) in slots.iter_mut().enumerate() {
            let sg = lgx[r * (ne + 1) + ne];
            slot.push((gidx.len() as u32, 1.0 / (1.0 + (-sg).exp())));
            gidx.push(r as u32);
        }
        let sh = l.moe.shared;
        panels.push(([ex.view(sh.0)?, ex.view(sh.1)?, ex.view(sh.2)?], row0, b));
        let n_as = gidx.len();
        let kmax = slots.iter().map(|s| s.len()).max().unwrap_or(1);
        let gidx_b = ex.scratch(c, "gidx", n_as * 4);
        let cslot_b = ex.scratch(c, "cslot", b * kmax * 4);
        let cw_b = ex.scratch(c, "cw", b * kmax * 4);
        // SAFETY: sized above; the device is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(gidx.as_ptr(), gidx_b.contents() as *mut u32, n_as);
            let (ps, pw) = (cslot_b.contents() as *mut u32, cw_b.contents() as *mut f32);
            for (r, row) in slots.iter().enumerate() {
                for j in 0..kmax {
                    // short rows repeat their first slot at weight 0
                    let (s, w) = row.get(j).copied().unwrap_or((row[0].0, 0.0));
                    *ps.add(r * kmax + j) = s;
                    *pw.add(r * kmax + j) = w;
                }
            }
        }
        let xg_b = ex.scratch(c, "xg", n_as * hs * 4);
        let gb = ex.scratch(c, "gb", n_as * inter * 4);
        let ub = ex.scratch(c, "ub", n_as * inter * 4);
        let act_b = ex.scratch(c, "act", n_as * inter * 4);
        let up_b = ex.scratch(c, "up", n_as * 4);
        let eo_b = ex.scratch(c, "eo", n_as * hs * 4);

        // ── experts ──
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.gatherr);
        bind(enc, &[(&n, 0), (&xg_b, 0), (&gidx_b, 0)]);
        set_u32(enc, 3, hs as u32);
        enc.dispatch_threads(MTLSize::new(hs as u64, n_as as u64, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cut(&mut cmd, st, 3)?;
        let enc = cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (w, r0, nr) in &panels {
            q4rows(c, enc, &w[0], &ex.ones, &xg_b, &gb, *r0, *nr, inter, hs);
            q4rows(c, enc, &w[1], &ex.ones, &xg_b, &ub, *r0, *nr, inter, hs);
        }
        enc.end_encoding();
        cut(&mut cmd, st, 4)?;
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.moeact);
        bind(enc, &[(&gb, 0), (&ub, 0), (&act_b, 0), (&up_b, 0)]);
        set_u32(enc, 4, inter as u32);
        enc.dispatch_thread_groups(MTLSize::new(n_as as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cut(&mut cmd, st, 5)?;
        let enc = cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (w, r0, nr) in &panels {
            q4rows(c, enc, &w[2], &ex.ones, &act_b, &eo_b, *r0, *nr, hs, inter);
        }
        enc.end_encoding();
        cut(&mut cmd, st, 6)?;
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.moecomb);
        bind(enc, &[(&eo_b, 0), (&dl, 0), (&cslot_b, 0), (&cw_b, 0)]);
        set_u32(enc, 4, hs as u32);
        set_u32(enc, 5, kmax as u32);
        enc.set_buffer(6, Some(&up_b), 0);
        enc.dispatch_threads(MTLSize::new(hs as u64, b as u64, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cut(&mut cmd, st, 7)?;

    }
    // a commit advanced the device state: book the rows it appended
    let book = |ex: &mut Exec| {
        let stored = ex.consts.iter().find_map(|k| k.attn.as_ref().map(|a| a.uploaded)).unwrap_or(0);
        for k in &mut ex.consts {
            if let Some(ac) = k.attn.as_mut() {
                ac.uploaded += b;
            }
        }
        ex.last_commit = Some((b, stored, g.state_len()));
        ex.epoch = Some((d.epoch.0, pos + b));
    };
    let Some(logits) = logits else {
        // the last layer's FFN delta is not needed: the hidden rows end here
        commit_wait(&cmd, st)?;
        book(ex);
        return Ok(());
    };
    // ── final norm and the head ──
    let lm_rows = d.lm_rows;
    let lo_b = ex.scratch(c, "logits", b * lm_rows * 4);
    let enc = cmd.new_compute_command_encoder();
    norm_rows(c, enc, &h, Some(&dl), &ex.final_norm, &n, b, hs, gemma, d.eps);
    q8mm(p, enc, ex.view(d.lm_head)?, &n, &lo_b, b, lm_rows, hs);
    enc.end_encoding();
    commit_wait(&cmd, st)?;
    if prof {
        st.stage_s[8] += super::cmd_gpu_ms(&cmd) / 1e3;
    }
    if causal {
        book(ex);
    }
    logits.resize(b * lm_rows, 0.0);
    // SAFETY: b·lm_rows floats the device just wrote.
    unsafe { std::ptr::copy_nonoverlapping(lo_b.contents() as *const f32, logits.as_mut_ptr(), b * lm_rows) };
    Ok(())
}

/// Drop the executor (weight views, expert views, device state).
pub fn release() {
    *EXEC.lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preview kernels and both q8 blob GEMM variants compile on this
    /// device (skipped without Metal).
    #[test]
    fn clover_kernels_compile() {
        let Some(c) = ctx() else {
            return;
        };
        if let Err(e) = pipes(c) {
            panic!("{e}");
        }
    }
}
