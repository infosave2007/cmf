//! The resident Embryo graph on CUDA (`gpu::forward_embryo_graph` and its
//! chunk, state and reset entries): the wide token path of
//! `gpu_wgpu/embryo_wide.wgsl` in `embryo.cu`, over the same packed model
//! (matrices in the file's q8_2f or f16, the small tensors in f32, the meta
//! records) and the same scratch image. Recurrent state and the anchors'
//! rings stay on the card for the whole sequence; a position is one hidden
//! row up, the layers' launches, and for the last position of a call the
//! head and the vocabulary's logits down. A file the wide path does not
//! take (`gpu::EmbryoGraphModel::wide` false) declines: the host runs it.

use super::*;
use std::collections::HashMap;
use std::sync::Mutex;

const EW_SRC: &str = include_str!("embryo.cu");

#[derive(Clone, Copy)]
struct EwFuncs {
    norm: cs::CUfunction,
    post_norm: cs::CUfunction,
    final_norm: cs::CUfunction,
    gdn_project: cs::CUfunction,
    gdn_step: cs::CUfunction,
    gdn_out: cs::CUfunction,
    anchor_project: cs::CUfunction,
    attend_score: cs::CUfunction,
    attend_weights: cs::CUfunction,
    attend_part: cs::CUfunction,
    attend_sum: cs::CUfunction,
    anchor_out: cs::CUfunction,
    route_rows: cs::CUfunction,
    route_final: cs::CUfunction,
    ffn_up: cs::CUfunction,
    ffn_down: cs::CUfunction,
    head: cs::CUfunction,
}

/// `Params` of the kernels, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    position: u32,
    max_seq: u32,
    vocab: u32,
    flags: u32,
    eps: f32,
    phase_mass: f32,
    layer: u32,
    chunk: u32,
}

struct DevModel {
    weights: Buf,
    meta: Buf,
    head: Buf,
    clusters: Buf,
    final_norm: Buf,
    mats: Buf,
}

struct DevSeq {
    state: Buf,
    kv: Buf,
    hidden: Buf,
    scratch: Buf,
    logits: Buf,
    initialized: bool,
    next_position: usize,
}

#[derive(Default)]
struct EwState {
    /// f16, q8_2f: the module's functions, or why it did not build.
    f: [Option<Result<EwFuncs, String>>; 2],
    models: HashMap<u64, DevModel>,
    seqs: HashMap<(u64, u64), DevSeq>,
}

// SAFETY: the functions and allocations are used under the context's state lock.
unsafe impl Send for EwState {}

static STATE: Mutex<Option<EwState>> = Mutex::new(None);

/// `S_ROPE` of the kernels: the angles of the current position, in f32 words.
const S_ROPE: usize = 53_248;

/// Scratch words of one sequence: the wgpu image's size (its chunk window
/// is unused here, the planes the token path reads lie below it).
fn scratch_words() -> usize {
    let c = crate::gpu::EMBRYO_CHUNK_MAX;
    40_960 + c * 1024 * 10 + c * 2048 * 2 + c * 16_512 + c * 1032 + c * 8 + c + 8 * c + 8
}

fn funcs(c: &Ctx, s: &mut EwState, q8: bool) -> Res<EwFuncs> {
    let slot = &mut s.f[q8 as usize];
    if slot.is_none() {
        let made = (|| -> Res<EwFuncs> {
            let m = compile(EW_SRC, vec![format!("-DQ8={}", q8 as u32)], c.arch)?;
            Ok(EwFuncs {
                norm: func(m, "ew_norm")?,
                post_norm: func(m, "ew_post_norm")?,
                final_norm: func(m, "ew_final_norm")?,
                gdn_project: func(m, "ew_gdn_project")?,
                gdn_step: func(m, "ew_gdn_step")?,
                gdn_out: func(m, "ew_gdn_out")?,
                anchor_project: func(m, "ew_anchor_project")?,
                attend_score: func(m, "ew_attend_score")?,
                attend_weights: func(m, "ew_attend_weights")?,
                attend_part: func(m, "ew_attend_part")?,
                attend_sum: func(m, "ew_attend_sum")?,
                anchor_out: func(m, "ew_anchor_out")?,
                route_rows: func(m, "ew_route_rows")?,
                route_final: func(m, "ew_route_final")?,
                ffn_up: func(m, "ew_ffn_up")?,
                ffn_down: func(m, "ew_ffn_down")?,
                head: func(m, "ew_head")?,
            })
        })();
        *slot = Some(made);
    }
    slot.as_ref().expect("set above").clone()
}

/// Is the wide token path there on this card?
pub fn embryo_wide_ready(q8: bool) -> bool {
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(EwState::default);
    match funcs(c, s, q8) {
        Ok(_) => true,
        Err(e) => {
            decline("embryo wide", &e);
            false
        }
    }
}

/// The geometry the kernels hold (the wgpu entry's checks for the wide path).
fn fits(m: &crate::gpu::EmbryoGraphModel, hidden_len: usize, n: usize) -> bool {
    let gdn_ok = m.gdn_layers == 0
        || (m.gdn_heads > 0
            && m.gdn_k_heads > 0
            && m.gdn_heads % m.gdn_k_heads == 0
            && m.gdn_dk > 0
            && m.gdn_dk <= 128
            && m.gdn_dv > 0
            && m.gdn_dv <= 256
            && m.gdn_dv % 4 == 0
            && m.gdn_kk > 0
            && m.gdn_heads <= 512
            && m.gdn_heads * m.gdn_dv <= 1024
            && m.gdn_c_dim() <= 2048
            && m.gdn_c_dim() % 4 == 0);
    m.wide
        && m.bounded
        && n > 0
        && n <= crate::gpu::EMBRYO_CHUNK_MAX
        && hidden_len == n * m.hidden
        && m.hidden > 0
        && m.hidden <= 1024
        && m.intermediate > 0
        && m.intermediate <= 1024
        && m.layers > 0
        && m.phase_layers == 0
        && gdn_ok
        && m.anchor_q_heads > 0
        && m.anchor_kv_heads > 0
        && m.anchor_q_heads % m.anchor_kv_heads == 0
        && m.anchor_q_heads * m.anchor_head_dim <= 1024
        && m.anchor_kv_heads * m.anchor_head_dim <= 1024
        && m.vocab > 0
        && m.cluster_count > 0
        && m.cluster_count <= 256
        && m.cluster_size <= 256
        && m.vocab >= 2 * m.cluster_count
}

pub fn forward_embryo_graph(
    model: &Arc<crate::gpu::EmbryoGraphModel>,
    kv_id: u64,
    hidden: &[f32],
    position: usize,
    logits: &mut Vec<f32>,
) -> bool {
    forward_rows(model, kv_id, hidden, 1, position, logits)
}

pub fn forward_embryo_graph_chunk(
    model: &Arc<crate::gpu::EmbryoGraphModel>,
    kv_id: u64,
    rows: &[f32],
    position: usize,
    n: usize,
    logits: &mut Vec<f32>,
) -> bool {
    forward_rows(model, kv_id, rows, n, position, logits)
}

fn forward_rows(
    model: &Arc<crate::gpu::EmbryoGraphModel>,
    kv_id: u64,
    rows: &[f32],
    n: usize,
    position: usize,
    logits: &mut Vec<f32>,
) -> bool {
    let dbg = std::env::var("CMF_EMBRYO_DBG").as_deref() == Ok("1");
    if !fits(model, rows.len(), n) {
        if dbg {
            eprintln!("embryo-dbg: cuda: geometry or n = {n} outside the wide kernels");
        }
        return false;
    }
    if !matches!(std::env::var("CMF_EMBRYO_RESIDENT").as_deref(), Ok("parallel") | Ok("1")) {
        return false;
    }
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(EwState::default);
    let f = match funcs(c, s, model.wide_q8) {
        Ok(f) => f,
        Err(e) => {
            decline("embryo wide", &e);
            return false;
        }
    };
    let key = (model.id, kv_id);
    // the device image continues only the sequence it holds
    if position != 0 && s.seqs.get(&key).is_none_or(|q| !q.initialized || q.next_position != position) {
        if dbg {
            eprintln!("embryo-dbg: cuda: detached position {position}");
        }
        if let Some(q) = s.seqs.get_mut(&key) {
            q.initialized = false;
            q.next_position = 0;
        }
        return false;
    }
    let t0 = std::time::Instant::now();
    match unsafe { run(c, s, f, model, kv_id, rows, n, position, logits) } {
        Ok(()) => {
            let q = s.seqs.get_mut(&key).expect("made by run");
            q.initialized = true;
            q.next_position = position + n;
            if std::env::var("CMF_EMBRYO_PROFILE").as_deref() == Ok("1") {
                eprintln!("embryo-profile: cuda wide n={n} total={:.3}ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "embryo resident graph: cuda wide active card={} layers={} hidden={} vocab={} {} matrices {} MB",
                    c.name,
                    model.layers,
                    model.hidden,
                    model.vocab,
                    if model.wide_q8 { "q8_2f" } else { "f16" },
                    (model.mats.len() + model.lm_head_packed.len()) / 1_000_000,
                );
            }
            true
        }
        Err(e) => {
            let _ = c.sync();
            decline("embryo", &e);
            s.seqs.remove(&key);
            false
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn run(
    c: &Ctx,
    s: &mut EwState,
    f: EwFuncs,
    model: &crate::gpu::EmbryoGraphModel,
    kv_id: u64,
    rows: &[f32],
    n: usize,
    position: usize,
    logits: &mut Vec<f32>,
) -> Res<()> {
    if !s.models.contains_key(&model.id) {
        let dm = DevModel {
            weights: Buf::from_slice(&model.weights)?,
            meta: Buf::from_slice(&model.meta)?,
            head: Buf::from_slice(&model.lm_head_packed)?,
            clusters: Buf::from_slice(&model.clusters)?,
            final_norm: Buf::from_slice(&model.final_norm)?,
            mats: Buf::from_slice(&model.mats)?,
        };
        s.models.insert(model.id, dm);
    }
    let dm = &s.models[&model.id];
    let key = (model.id, kv_id);
    if !s.seqs.contains_key(&key) {
        let state_bytes = (model.state_layers * model.state_stride).max(1) * 4;
        let kv_bytes = (model.kv_layers * model.kv_stride).max(1) * 4;
        let logits_words = model.vocab + 2 * model.cluster_count;
        let q = DevSeq {
            state: Buf::direct(state_bytes)?,
            kv: Buf::direct(kv_bytes)?,
            hidden: Buf::direct(model.hidden * 4 + 64)?,
            scratch: Buf::direct(scratch_words() * 4)?,
            logits: Buf::direct(logits_words * 4 + 64)?,
            initialized: false,
            next_position: 0,
        };
        // a never-written plane reads as zeros, as the wgpu buffers do
        unsafe { cu::memset_d8_async(q.scratch.ptr, 0, q.scratch.bytes, c.stream).map_err(derr)? };
        s.seqs.insert(key, q);
    }
    let q = s.seqs.get_mut(&key).expect("inserted above");
    if !q.initialized || position == 0 {
        unsafe {
            cu::memset_d8_async(q.state.ptr, 0, q.state.bytes, c.stream).map_err(derr)?;
            cu::memset_d8_async(q.kv.ptr, 0, q.kv.bytes, c.stream).map_err(derr)?;
        }
    }
    static GATE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let gate = *GATE.get_or_init(|| {
        std::env::var("CMF_EMBRYO_HEAD_GATE").ok().and_then(|v| v.parse().ok()).unwrap_or(-9.21)
    });
    let hidden = model.hidden;
    let g32 = |threads: usize| (threads.div_ceil(32).max(1) as u32, 1u32);
    let parts = |cols: usize| cols.div_ceil(128);
    let eps = f32::from_bits(model.meta.get(19).copied().unwrap_or(0));
    let flags = model.norm_gemma as u32;
    let hd = model.anchor_head_dim;
    let mut rope = vec![0f32; 2 * model.rope_freq.len()];
    for t in 0..n {
        let pos = position + t;
        unsafe { cu::memcpy_htod_async(q.hidden.ptr, &rows[t * hidden..(t + 1) * hidden], c.stream).map_err(derr)? };
        if !model.rope_freq.is_empty() {
            let half = model.rope_freq.len();
            for (i, &fr) in model.rope_freq.iter().enumerate() {
                let (sn, cs) = (pos as f64 * fr as f64).sin_cos();
                rope[i] = cs as f32;
                rope[half + i] = sn as f32;
            }
            unsafe { cu::memcpy_htod_async(q.scratch.ptr + (S_ROPE * 4) as u64, &rope, c.stream).map_err(derr)? };
        }
        for li in 0..model.layers {
            let r = crate::gpu::EMBRYO_META_HEADER + li * 64;
            let kind = model.meta[r];
            let p = Params {
                position: pos as u32,
                max_seq: model.max_seq as u32,
                vocab: model.vocab as u32,
                flags,
                eps,
                phase_mass: gate,
                layer: li as u32,
                chunk: 1,
            };
            let go = |func: cs::CUfunction, grid: (u32, u32), block: u32| -> Res<()> {
                unsafe {
                    c.launch(
                        func,
                        grid,
                        block,
                        0,
                        kargs!(
                            dm.weights.ptr,
                            dm.meta.ptr,
                            dm.head.ptr,
                            q.state.ptr,
                            q.logits.ptr,
                            p,
                            q.scratch.ptr,
                            dm.mats.ptr,
                            dm.clusters.ptr,
                            dm.final_norm.ptr,
                            q.hidden.ptr,
                            q.kv.ptr
                        ),
                    )
                }
            };
            go(f.norm, (1, 1), 256)?;
            if kind == 4 {
                let (nv, dv, cdim) = (model.gdn_heads, model.gdn_dv, model.gdn_c_dim());
                go(f.gdn_project, g32(cdim + nv * dv + nv), 32)?;
                go(f.gdn_step, (nv as u32, 1), 128)?;
                go(f.gdn_out, g32(hidden * parts(nv * dv)), 32)?;
            } else {
                let (qh, qrows, krows) = (model.anchor_q_heads, model.anchor_q_heads * hd, model.anchor_kv_heads * hd);
                let keys = model.anchor_sink + (pos + 1).min(model.anchor_window);
                go(f.anchor_project, g32(qrows + 2 * krows), 32)?;
                go(f.attend_score, g32(qh * keys.div_ceil(4)), 32)?;
                go(f.attend_weights, (qh as u32, 1), 1)?;
                go(f.attend_part, g32(qh * keys.div_ceil(32) * (hd / 4).div_ceil(8)), 32)?;
                go(f.attend_sum, g32(qrows / 4), 32)?;
                go(f.anchor_out, g32(hidden * parts(qrows)), 32)?;
            }
            go(f.post_norm, (1, 1), 256)?;
            let moe = model.meta[r + 15] == 1;
            if moe {
                let (ne, rank) = (model.meta[r + 16] as usize, model.meta[r + 20] as usize);
                go(f.route_rows, g32(ne * (rank + 1) * parts(hidden)), 32)?;
                go(f.route_final, (1, 1), 256)?;
            }
            go(f.ffn_up, g32(model.intermediate * if moe { 2 } else { 1 }), 32)?;
            go(f.ffn_down, g32(hidden * parts(model.intermediate)), 32)?;
            if t + 1 == n && li + 1 == model.layers {
                go(f.final_norm, (1, 1), 256)?;
                go(f.head, (model.cluster_count as u32, 1), 256)?;
            }
        }
    }
    logits.resize(model.vocab, 0.0);
    unsafe { cu::memcpy_dtoh_async(&mut logits[..], q.logits.ptr, c.stream).map_err(derr)? };
    c.sync()
}

/// `(state bytes, ring bytes)` the card holds for `kv_id`.
pub fn embryo_device_state_bytes(kv_id: u64) -> Option<(u64, u64)> {
    let g = STATE.lock().ok()?;
    let s = g.as_ref()?;
    s.seqs.iter().find(|((_, id), _)| *id == kv_id).map(|(_, q)| (q.state.bytes as u64, q.kv.bytes as u64))
}

/// Next position of the card's sequence `kv_id` (`None` = the host owns it).
pub fn embryo_device_next_position(kv_id: u64) -> Option<usize> {
    let g = STATE.lock().ok()?;
    let s = g.as_ref()?;
    s.seqs
        .iter()
        .find(|((_, id), _)| *id == kv_id)
        .and_then(|(_, q)| q.initialized.then_some(q.next_position))
}

/// Drop the card's image of a sequence; the model stays up.
pub fn embryo_graph_reset(kv_id: u64) {
    let Some(c) = ctx() else { return };
    if c.current().is_err() {
        return;
    }
    let _st = c.st.lock().unwrap();
    if let Ok(mut g) = STATE.lock() {
        if let Some(s) = g.as_mut() {
            let _ = c.sync();
            s.seqs.retain(|(_, id), _| *id != kv_id);
        }
    }
}
