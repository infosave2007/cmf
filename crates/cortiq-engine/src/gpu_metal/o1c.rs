//! A sealed layer's attention for a chunk of rows on Metal: the twin of `gpu_wgpu/o1_batch.rs`
//! and the chunk stages of `gpu_wgpu/o1_stages.rs`, with the near window up to 2052 keys.
//! A token is a chunk of one row, so `encode_attn_device` and `encode_attn_b` share them.

use super::*;

const MSL: &str = include_str!("o1c_msl.metal");

/// Rows per sub-chunk, as `o1_batch::OB_K`.
pub(crate) const OB_K: usize = 64;
/// Sink and window together: the reach of the file's `nsrect`-free geometry.
pub(crate) const MAX_NEAR: usize = 2052;

pub(crate) struct Pipes {
    far_e: ComputePipelineState,
    score: ComputePipelineState,
    land: ComputePipelineState,
    max: ComputePipelineState,
    far: ComputePipelineState,
    soft: ComputePipelineState,
    sum: ComputePipelineState,
    add: ComputePipelineState,
    far_commit: ComputePipelineState,
    ring_commit: ComputePipelineState,
    meta: ComputePipelineState,
    copy: ComputePipelineState,
}

unsafe impl Send for Pipes {}
unsafe impl Sync for Pipes {}

static PIPES: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();

pub(crate) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    if std::env::var("CMF_O1_STAGES").as_deref() == Ok("0") {
        return None;
    }
    PIPES
        .get_or_init(|| {
            let opts = metal::CompileOptions::new();
            opts.set_language_version(metal::MTLLanguageVersion::V3_0);
            let lib = match c._device.new_library_with_source(MSL, &opts) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("o1c MSL compile: {e}");
                    return None;
                }
            };
            let pso = |name: &str| -> Option<ComputePipelineState> {
                let f = lib.get_function(name, None).ok()?;
                c._device.new_compute_pipeline_state_with_function(&f).ok()
            };
            Some(Pipes {
                far_e: pso("o1c_far_e")?,
                score: pso("o1c_score")?,
                land: pso("o1c_land")?,
                max: pso("o1c_max")?,
                far: pso("o1c_far")?,
                soft: pso("o1c_soft")?,
                sum: pso("o1c_sum")?,
                add: pso("o1c_add")?,
                far_commit: pso("o1c_far_commit")?,
                ring_commit: pso("o1c_ring_commit")?,
                meta: pso("o1c_meta")?,
                copy: pso("o1c_copy")?,
            })
        })
        .as_ref()
}

/// A chunk's planes. Layers run one after another, so all of them share one set.
pub(crate) struct Scratch {
    key: (usize, usize, usize, usize, usize),
    s: Buffer,
    sc: Buffer,
    fs: Buffer,
    nm: Buffer,
    hd: Buffer,
    gn: Buffer,
    cf: Buffer,
    inv: Buffer,
    pz: Buffer,
}

unsafe impl Send for Scratch {}
unsafe impl Sync for Scratch {}

static SCRATCH: std::sync::Mutex<Option<Arc<Scratch>>> = std::sync::Mutex::new(None);

fn shared(c: &Ctx, bytes: usize) -> Buffer {
    c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

fn scratch(c: &Ctx, od: &O1MetalDev) -> Arc<Scratch> {
    let (g, hpg, m, d, ncap) = (od.g, od.h, od.m, od.d, od.w + od.ns);
    let key = (g, hpg, m, d, ncap);
    let mut slot = SCRATCH.lock().unwrap();
    if let Some(s) = slot.as_ref().filter(|s| s.key == key) {
        return s.clone();
    }
    let k = OB_K;
    let nh = g * hpg;
    let ghm = nh * m;
    let s = Arc::new(Scratch {
        key,
        s: shared(c, (2 * ghm * k + 2 * ghm) * 4),
        sc: shared(c, k * nh * ncap * 4),
        fs: shared(c, k * nh * m * 4),
        nm: shared(c, k * nh * 4),
        hd: shared(c, k * nh * 16),
        gn: shared(c, k * nh * m * 4),
        cf: shared(c, k * nh * k * 4),
        inv: shared(c, k * nh * 4),
        pz: shared(c, k * nh * 8 * d * 4),
    });
    *slot = Some(s.clone());
    s
}

/// This state takes the stages: a layer of even vec4 heads with up to 32 landmarks, no
/// rectifier on the far mass and the same width for keys and values.
pub(crate) fn admits(c: &Ctx, od: &O1MetalDev) -> bool {
    od.d % 4 == 0 && od.d == od.dv && od.m <= 32 && od.nsrect >> 8 == 0 && od.w + od.ns <= MAX_NEAR && pipes(c).is_some()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OC {
    hpg: u32,
    m: u32,
    w: u32,
    ns: u32,
    d: u32,
    nh: u32,
    ng: u32,
    k: u32,
    r0: u32,
    ghm: u32,
    scale: f32,
    pad: u32,
}

fn uniform(od: &O1MetalDev, kk: usize, r0: usize) -> OC {
    OC {
        hpg: od.h as u32,
        m: od.m as u32,
        w: od.w as u32,
        ns: od.ns as u32,
        d: od.d as u32,
        nh: (od.g * od.h) as u32,
        ng: od.g as u32,
        k: kk as u32,
        r0: r0 as u32,
        ghm: (od.g * od.h * od.m) as u32,
        scale: od.scale,
        pad: 0,
    }
}

fn run(
    enc: &metal::ComputeCommandEncoderRef,
    conc: bool,
    pso: &ComputePipelineState,
    bufs: &[&Buffer],
    u: &OC,
    grid: (u64, u64, u64),
    tg: u64,
    per_group: bool,
) {
    enc.set_compute_pipeline_state(pso);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_bytes(bufs.len() as u64, std::mem::size_of::<OC>() as u64, u as *const OC as *const std::ffi::c_void);
    if per_group {
        enc.dispatch_thread_groups(MTLSize::new(grid.0, grid.1, grid.2), MTLSize::new(tg, 1, 1));
    } else {
        enc.dispatch_threads(MTLSize::new(grid.0, grid.1, grid.2), MTLSize::new(tg, 1, 1));
    }
    if conc {
        enc_barrier(enc);
    }
}

fn attend(
    enc: &metal::ComputeCommandEncoderRef,
    conc: bool,
    p: &Pipes,
    od: &O1MetalDev,
    s: &Scratch,
    q: &Buffer,
    kb: &Buffer,
    vb: &Buffer,
    out: &Buffer,
    u: &OC,
    kk: usize,
) {
    let (g, nh, ncap, m) = (od.g as u64, (od.g * od.h) as u64, (od.w + od.ns) as u64, od.m as u64);
    let e4 = (od.d / 4) as u64;
    let k = kk as u64;
    run(enc, conc, &p.score, &[&od.meta, q, &od.ring_k, &od.sink_k, kb, &s.sc], u, (ncap, g, k), 32, false);
    run(enc, conc, &p.land, &[q, &od.k_tilde, &s.fs], u, (m, g, k), 32, false);
    run(enc, conc, &p.max, &[&od.meta, &s.sc, &s.nm], u, (nh, k, 1), 32, true);
    run(enc, conc, &p.far, &[&od.meta, &s.fs, &s.nm, &od.mu, &s.s, &s.hd, &s.gn, &s.cf], u, (nh, k, 1), 8, false);
    run(enc, conc, &p.soft, &[&od.meta, &s.sc, &s.hd, &s.inv], u, (nh, k, 1), 32, true);
    run(enc, conc, &p.sum, &[&od.meta, &s.sc, &od.ring_v, &od.sink_v, vb, &s.pz], u, (e4 * 8 * g, k, 1), 32, false);
    run(
        enc,
        conc,
        &p.add,
        &[&od.meta, &s.pz, &s.hd, &s.gn, &od.that, &s.inv, &s.cf, &od.ring_v, out],
        u,
        (nh * e4, k, 1),
        32,
        false,
    );
}

fn far_e(enc: &metal::ComputeCommandEncoderRef, conc: bool, p: &Pipes, od: &O1MetalDev, s: &Scratch, u: &OC) {
    let ghm = (od.g * od.h * od.m) as u64;
    run(enc, conc, &p.far_e, &[&od.meta, &od.ring_k, &od.qt, &od.mz, &s.s], u, (ghm, 1, 1), 32, false);
}

fn state_commit(
    enc: &metal::ComputeCommandEncoderRef,
    conc: bool,
    p: &Pipes,
    od: &O1MetalDev,
    s: &Scratch,
    kb: &Buffer,
    vb: &Buffer,
    u: &OC,
    kk: usize,
) {
    let ghm = (od.g * od.h * od.m) as u64;
    let e4 = (od.d / 4) as u64;
    run(enc, conc, &p.far_commit, &[&od.meta, &od.ring_v, &od.mz, &od.that, &s.s], u, (e4, ghm, 1), 32, false);
    run(
        enc,
        conc,
        &p.ring_commit,
        &[&od.meta, &od.ring_k, &od.ring_v, kb, vb],
        u,
        (e4, od.g as u64, kk as u64),
        32,
        false,
    );
    run(enc, conc, &p.meta, &[&od.meta], u, (od.g as u64, 1, 1), 16, false);
}

/// The attention of `rows` rows of a sealed layer into `out`, and with `commit` the state after
/// them. `q`: [rows][nh][d], `kb`, `vb`: [rows][g][d], `out`: [rows][nh][d].
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    conc: bool,
    od: &O1MetalDev,
    q: &Buffer,
    kb: &Buffer,
    vb: &Buffer,
    out: &Buffer,
    rows: usize,
    commit: bool,
) -> bool {
    let Some(p) = pipes(c) else { return false };
    let s = scratch(c, od);
    let mut r0 = 0;
    while r0 < rows {
        let kk = OB_K.min(rows - r0);
        let u = uniform(od, kk, r0);
        far_e(enc, conc, p, od, &s, &u);
        attend(enc, conc, p, od, &s, q, kb, vb, out, &u, kk);
        if commit {
            state_commit(enc, conc, p, od, &s, kb, vb, &u, kk);
        }
        r0 += kk;
    }
    true
}

/// `bytes` of `src` into `dst`, for rows that must outlive the graph's scratch.
pub(crate) fn copy(c: &Ctx, enc: &metal::ComputeCommandEncoderRef, conc: bool, src: &Buffer, dst: &Buffer, bytes: usize) {
    let Some(p) = pipes(c) else { return };
    enc.set_compute_pipeline_state(&p.copy);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(dst), 0);
    enc.dispatch_threads(MTLSize::new((bytes / 16) as u64, 1, 1), MTLSize::new(256, 1, 1));
    if conc {
        enc_barrier(enc);
    }
}

/// The accepted rows of a verify: every sealed layer of `kv_id` moves over them. The rows' keys
/// and values wait in the layer's `spec_k` and `spec_v`.
pub(crate) fn spec_commit(kv_id: u64, rows: usize) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let devs: Vec<O1MetalDev> = {
        let reg = c.o1m.lock().unwrap();
        reg.iter().filter(|((id, _), _)| *id == kv_id).map(|(_, d)| d.clone()).collect()
    };
    if devs.is_empty() {
        return true;
    }
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for od in &devs {
        let s = scratch(c, od);
        let mut r0 = 0;
        while r0 < rows {
            let kk = OB_K.min(rows - r0);
            let u = uniform(od, kk, r0);
            far_e(enc, false, p, od, &s, &u);
            state_commit(enc, false, p, od, &s, &od.spec_k, &od.spec_v, &u, kk);
            r0 += kk;
        }
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    cmd.status() != metal::MTLCommandBufferStatus::Error
}

/// What the prompt path asks a rows graph to keep: for the layers still collecting, their
/// queries; for sealed ones, the queries, keys and values on their way to the archive.
pub(crate) struct Capture {
    rows: Vec<(usize, Buffer, Buffer, Buffer, usize, usize)>,
    pub(crate) q_only: Vec<usize>,
    pub(crate) sealed: bool,
}

static CAPTURE: std::sync::Mutex<Option<Capture>> = std::sync::Mutex::new(None);
unsafe impl Send for Capture {}

/// Keep the queries of these layers, for a first seal.
pub(crate) fn qcap_begin(layers: &[usize]) {
    *CAPTURE.lock().unwrap() = Some(Capture { rows: Vec::new(), q_only: layers.to_vec(), sealed: false });
}

/// Keep queries, keys and values of sealed layers, for the far field's rebuild.
pub(crate) fn acap_begin() {
    *CAPTURE.lock().unwrap() = Some(Capture { rows: Vec::new(), q_only: Vec::new(), sealed: true });
}

pub(crate) fn cap_end() {
    *CAPTURE.lock().unwrap() = None;
}

/// Whether a first seal asks for this layer's queries.
pub(crate) fn wants_q(li: usize) -> bool {
    CAPTURE.lock().unwrap().as_ref().is_some_and(|c| !c.sealed && c.q_only.contains(&li))
}

/// A rows graph's copies for this layer, if it is asked for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    li: usize,
    sealed: bool,
    q: &Buffer,
    kb: &Buffer,
    vb: &Buffer,
    qn: usize,
    kn: usize,
) {
    let mut slot = CAPTURE.lock().unwrap();
    let Some(cap) = slot.as_mut() else { return };
    let want_all = cap.sealed && sealed;
    let want_q = !cap.sealed && cap.q_only.contains(&li);
    if !want_all && !want_q {
        return;
    }
    let qb = shared(c, qn * 4);
    copy(c, enc, false, q, &qb, qn * 4);
    let (kc, vc) = if want_all {
        let (a, b) = (shared(c, kn * 4), shared(c, kn * 4));
        copy(c, enc, false, kb, &a, kn * 4);
        copy(c, enc, false, vb, &b, kn * 4);
        (a, b)
    } else {
        (shared(c, 16), shared(c, 16))
    };
    cap.rows.push((li, qb, kc, vc, qn, if want_all { kn } else { 0 }));
}

/// What was kept since the last call, once the graph has finished: (layer, q, k, v).
pub(crate) fn cap_take() -> Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut slot = CAPTURE.lock().unwrap();
    let Some(cap) = slot.as_mut() else { return Vec::new() };
    let read = |b: &Buffer, n: usize| -> Vec<f32> {
        let mut v = vec![0f32; n];
        // SAFETY: a shared buffer of at least `n` floats, and the graph that wrote it is done.
        unsafe { std::ptr::copy_nonoverlapping(b.contents() as *const f32, v.as_mut_ptr(), n) };
        v
    };
    let mut out: Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    for (li, q, k, v, qn, kn) in cap.rows.drain(..) {
        let (q, k, v) = (read(&q, qn), read(&k, kn), read(&v, kn));
        match out.iter_mut().find(|e| e.0 == li) {
            // The halves of one chunk, in the rows' order: the second half is behind the first
            // at every layer, though the first may already be a layer further on.
            Some(last) => {
                last.1.extend(q);
                last.2.extend(k);
                last.3.extend(v);
            }
            _ => out.push((li, q, k, v)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunk kernels against `nystrom`'s serial steps: the attention's output, then the
    /// state after the chunk through the next chunk's output.
    #[test]
    fn o1c_matches_serial_steps() {
        let Some(c) = ctx() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        if pipes(c).is_none() {
            eprintln!("o1c kernels unavailable — skipping");
            return;
        }
        // (d, m, w, sink, hpg, prompt t, second group's window, chunks)
        for (d, m, w, sink, hpg, t, w2, chunks) in [
            // Both rings full at the start, 70 rows = two sub-chunks.
            (8usize, 4usize, 64usize, 2usize, 2usize, 100usize, 64usize, [70usize, 9]),
            // Wide group fills during the chunk: its first eviction meets an empty far field.
            (8, 4, 64, 2, 2, 100, 128, [70, 40]),
            // Production head geometry (Qwen3.8: d 256, 6 heads a group, 32 landmarks).
            (256, 32, 128, 4, 6, 300, 512, [64, 64]),
            // A verify and a token: chunks of five and of one.
            (256, 32, 128, 4, 6, 300, 128, [5, 1]),
        ] {
            let jit = |a: usize, b: usize| ((a * 37 + b * 13 + 3) % 83) as f32 / 83.0 - 0.5;
            let mut states = Vec::new();
            for (gi, wg) in [(0usize, w), (1, w2)] {
                let ks: Vec<f32> = (0..t * d).map(|i| jit(i, 1 + 20 * gi)).collect();
                let vs: Vec<f32> = (0..t * d).map(|i| jit(i, 2 + 20 * gi)).collect();
                let qs: Vec<Vec<f32>> = (0..hpg).map(|h| (0..t * d).map(|i| jit(i, 3 + h + 20 * gi)).collect()).collect();
                let refs: Vec<&[f32]> = qs.iter().map(|v| v.as_slice()).collect();
                let mut st = crate::nystrom::NystromState::new_group(m, w, sink, hpg).with_window(wg);
                st.prefill_group(&refs, &ks, &vs, t, d, d);
                states.push(st);
            }
            let (ng, nh) = (2usize, 2 * hpg);
            let key = (u64::MAX - 7, usize::MAX - 7);
            {
                let views: Vec<_> = states.iter().map(|s| s.device_view()).collect();
                assert!(!views[0].exact_only, "t must exceed w+8 for this test");
                assert!(o1_ensure_metal(c, key.0, key.1, &views, 1, MAX_NEAR).is_some(), "o1 upload");
            }
            let od = c.o1m.lock().unwrap().get(&key).unwrap().clone();
            let stor = |data: &[f32]| {
                let b = shared(c, data.len() * 4);
                // SAFETY: a fresh shared buffer of this size.
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents() as *mut f32, data.len()) };
                b
            };
            let mut base = 0usize;
            for (ci, &rows) in chunks.iter().enumerate() {
                let q: Vec<f32> = (0..rows * nh * d).map(|i| jit(i + base * nh * d, 5)).collect();
                let k: Vec<f32> = (0..rows * ng * d).map(|i| jit(i + base * ng * d, 6)).collect();
                let v: Vec<f32> = (0..rows * ng * d).map(|i| jit(i + base * ng * d, 7)).collect();
                base += rows;
                let mut want = vec![0f32; rows * nh * d];
                for r in 0..rows {
                    for (g, st) in states.iter_mut().enumerate() {
                        let qo = (r * nh + g * hpg) * d;
                        let ko = (r * ng + g) * d;
                        st.step_group(&q[qo..qo + hpg * d], &k[ko..ko + d], &v[ko..ko + d], &mut want[qo..qo + hpg * d]);
                    }
                }
                let (qb, kb, vb) = (stor(&q), stor(&k), stor(&v));
                let ob = shared(c, rows * nh * d * 4);
                let cmd = c.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                assert!(encode(c, enc, false, &od, &qb, &kb, &vb, &ob, rows, true));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let mut got = vec![0f32; rows * nh * d];
                // SAFETY: a shared buffer of this size, and the command buffer is done.
                unsafe { std::ptr::copy_nonoverlapping(ob.contents() as *const f32, got.as_mut_ptr(), got.len()) };
                assert!(got.iter().all(|x| x.is_finite()), "non-finite output");
                let md = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                eprintln!("o1c d={d} m={m} w={w}/{w2} hpg={hpg} chunk {ci} ({rows} rows): max|Δ| {md:.2e}");
                assert!(md < 1e-3, "o1c ≠ serial steps (d={d} w={w}/{w2} chunk {ci}): max|Δ| = {md}");
            }
            c.o1m.lock().unwrap().remove(&key);
        }
    }
}
