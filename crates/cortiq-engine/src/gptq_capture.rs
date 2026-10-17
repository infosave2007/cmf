//! Calibration-time Hessian capture for GPTQ / error-feedback transfer.
//!
//! While `begin()` is active, every `QTensor::matmat` (the batched prefill
//! path) folds its input activations into a per-tensor second-moment
//! `H = Σ_t x_t·x_tᵀ` and per-channel `Σ x²` (for the activation RMS field
//! of the two-field outlier score). One global hook covers every linear in
//! the model — attention projections, FFN, experts — with no per-layer
//! wiring. Zero cost when off (one relaxed atomic load). Single-threaded
//! accumulation from the caller's thread, so it never races the parallel
//! matmat that follows.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static ON: AtomicBool = AtomicBool::new(false);
/// Accumulate the full `H = X·Xᵀ` (needed only for the GPTQ fold). Off ⇒
/// diagonal-only (`Σx²`), which is all the ternary + per-row correction
/// path needs — and the only thing that fits for a 12B (full H would be
/// ~100 GB across the model).
static FULL_H: AtomicBool = AtomicBool::new(true);
static REG: Mutex<Option<State>> = Mutex::new(None);
/// Keep the dense Hessians on the device (`set_device`): each distinct input
/// gets a `gpu::gram_*` accumulator, its rows staged here and folded
/// `STAGE_ROWS` at a time.
static DEVICE: AtomicBool = AtomicBool::new(false);
/// Rows a device accumulator collects before one `H += XᵀX` dispatch.
const STAGE_ROWS: usize = 1024;

struct Entry {
    acc: HessianAcc,
    dev: Option<u64>,
    stage: Vec<f32>,
}

/// One capture pass. Projections that read the same panel (q/k/v, gate/up,
/// a GDN block's qkv/z) arrive back to back with byte-identical input: the
/// first one owns the Hessian, the others alias it and cost a memcmp.
#[derive(Default)]
struct State {
    map: HashMap<String, Entry>,
    alias: HashMap<String, String>,
    /// The owner of the last folded input and a copy of that input.
    last: Option<(String, Vec<f32>)>,
    /// Aliases whose input later differed from their owner's, and device
    /// accumulators that failed: their Hessians are not valid.
    broken: Vec<String>,
}

/// What a capture pass produced: Hessians by owner, `(alias, owner)` pairs
/// for the projections that shared an owner's input, and the names whose
/// capture is invalid (see [`State::broken`]).
pub struct Captured {
    pub hess: HashMap<String, HessianAcc>,
    pub alias: Vec<(String, String)>,
    pub broken: Vec<String>,
    /// [`end_shared_lazy`] only: owners whose dense Hessian is still on the
    /// device (their `h` is empty) — `gpu::gram_take(id)` reads it back.
    pub device: HashMap<String, u64>,
}

/// Accumulate dense Hessians on the device from the next `begin` on.
pub fn set_device(on: bool) {
    DEVICE.store(on, Ordering::SeqCst);
}
/// Which tensors to capture (`None` = all). A dense Hessian costs
/// `8·cols²` bytes — 300 MB for one 6144-wide `down_proj` — so a caller that
/// quantizes only some linears, or a model a few layers at a time, keeps
/// everything else out of RAM and skips its O(b·cols²) accumulation.
type CaptureFilter = Box<dyn Fn(&str) -> bool + Send>;
static FILTER: Mutex<Option<CaptureFilter>> = Mutex::new(None);

/// Accumulated input statistics of one linear layer over the calibration set.
pub struct HessianAcc {
    pub cols: usize,
    /// `H = Σ_t x_t·x_tᵀ`, dense `[cols·cols]`, f64.
    pub h: Vec<f64>,
    /// `Σ_t x_t²` per input channel `[cols]` — RMS = sqrt(sumsq / count).
    pub sumsq: Vec<f64>,
    /// Token-position samples folded in.
    pub count: usize,
}

impl HessianAcc {
    /// Per-input-channel activation RMS.
    pub fn rms(&self) -> Vec<f32> {
        let n = self.count.max(1) as f64;
        self.sumsq.iter().map(|&s| (s / n).sqrt() as f32).collect()
    }
}

/// Start capturing (clears any prior registry). `full_h` = accumulate the
/// dense Hessian (fold path); false = diagonal-only (ternary + correction).
pub fn begin(full_h: bool) {
    *FILTER.lock().unwrap() = None;
    crate::gpu::gram_clear();
    *REG.lock().unwrap() = Some(State::default());
    FULL_H.store(full_h, Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
}

/// [`begin`], capturing only the tensors whose name passes `keep`. The
/// others cost one closure call per matmat and no memory.
pub fn begin_filtered(full_h: bool, keep: impl Fn(&str) -> bool + Send + 'static) {
    begin(full_h);
    *FILTER.lock().unwrap() = Some(Box::new(keep));
}

/// Stop capturing and take the accumulated Hessians by name. An alias gets
/// its own copy of its owner's Hessian; invalid captures are left out.
pub fn end() -> HashMap<String, HessianAcc> {
    let Captured { mut hess, alias, broken, .. } = end_shared();
    for (a, owner) in alias {
        if let Some(src) = hess.get(&owner) {
            let copy = HessianAcc {
                cols: src.cols,
                h: src.h.clone(),
                sumsq: src.sumsq.clone(),
                count: src.count,
            };
            hess.insert(a, copy);
        }
    }
    for n in &broken {
        hess.remove(n);
    }
    hess
}

/// Stop capturing: owners' Hessians (dense ones full and symmetric),
/// aliases and invalid names, without copying any shared Hessian.
pub fn end_shared() -> Captured {
    end_impl(false)
}

/// [`end_shared`], leaving device-accumulated Hessians on the device: a
/// 27B's 64 layers hold ~50 GB of packed f32 triangles there, ~200 GB as
/// host f64 — the caller reads each back when its fold needs it.
pub fn end_shared_lazy() -> Captured {
    end_impl(true)
}

fn end_impl(lazy: bool) -> Captured {
    ON.store(false, Ordering::SeqCst);
    *FILTER.lock().unwrap() = None;
    let ns = ACC_NS.swap(0, Ordering::Relaxed);
    let st = REG.lock().unwrap().take().unwrap_or_default();
    let t0 = std::time::Instant::now();
    let mut broken = st.broken;
    let mut hess = HashMap::with_capacity(st.map.len());
    let mut on_device = 0usize;
    let mut device = HashMap::new();
    for (name, mut e) in st.map {
        if let Some(id) = e.dev {
            on_device += 1;
            let n = e.acc.cols;
            let rows = e.stage.len() / n.max(1);
            let flushed = rows == 0 || crate::gpu::gram_add(id, &e.stage, rows);
            if lazy {
                if flushed {
                    device.insert(name.clone(), id);
                    hess.insert(name, e.acc);
                } else {
                    broken.push(name);
                }
                continue;
            }
            match crate::gpu::gram_take(id) {
                Some(h) if flushed => e.acc.h = h,
                _ => {
                    broken.push(name);
                    continue;
                }
            }
        } else {
            let n = e.acc.cols;
            let acc = &mut e.acc;
            if acc.h.len() == n * n {
                for i in 0..n {
                    for j in (i + 1)..n {
                        acc.h[j * n + i] = acc.h[i * n + j];
                    }
                }
            }
        }
        hess.insert(name, e.acc);
    }
    if ns > 0 {
        eprintln!(
            "gptq_capture: {:.1} s folding inputs ({} Hessians, {on_device} on the device, {} aliases; read back in {:.1} s)",
            ns as f64 * 1e-9,
            hess.len(),
            st.alias.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    let alias = st
        .alias
        .into_iter()
        .filter(|(a, o)| !broken.contains(a) && hess.contains_key(o))
        .collect();
    if !broken.is_empty() {
        eprintln!("gptq_capture: {} invalid captures: {broken:?}", broken.len());
    }
    Captured { hess, alias, broken, device }
}

#[inline]
pub fn capturing() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Fold a batch of `b` input vectors (`xs = [b·cols]`) into `name`'s Hessian.
pub fn accumulate(name: &str, xs: &[f32], b: usize, cols: usize) {
    let t0 = std::time::Instant::now();
    accumulate_inner(name, xs, b, cols);
    ACC_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Wall time spent inside `accumulate` since the last `end()`.
static ACC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn accumulate_inner(name: &str, xs: &[f32], b: usize, cols: usize) {
    if let Some(keep) = FILTER.lock().unwrap().as_ref()
        && !keep(name)
    {
        return;
    }
    let mut guard = REG.lock().unwrap();
    let Some(st) = guard.as_mut() else {
        return;
    };
    let full = FULL_H.load(Ordering::Relaxed);
    if full {
        if let Some(owner) = st.alias.get(name) {
            let same = matches!(&st.last, Some((o, x)) if o == owner && x.as_slice() == xs);
            if !same && !st.broken.iter().any(|n| n == name) {
                st.broken.push(name.to_string());
            }
            return;
        }
        if !st.map.contains_key(name)
            && let Some((owner, x)) = &st.last
            && owner != name
            && x.as_slice() == xs
            && st.map.get(owner).is_some_and(|e| e.acc.cols == cols)
        {
            st.alias.insert(name.to_string(), owner.clone());
            return;
        }
    }
    let device = full && DEVICE.load(Ordering::Relaxed);
    let e = st.map.entry(name.to_string()).or_insert_with(|| {
        let dev = if device { crate::gpu::gram_alloc(cols) } else { None };
        if device && dev.is_none() {
            eprintln!("gptq_capture: no device Hessian for {name} ({cols}²) — host fold");
        }
        Entry {
            acc: HessianAcc {
                cols,
                h: if full && dev.is_none() {
                    vec![0.0; cols * cols]
                } else {
                    Vec::new()
                },
                sumsq: vec![0.0; cols],
                count: 0,
            },
            dev,
            stage: Vec::new(),
        }
    });
    if e.acc.cols != cols {
        return; // shape mismatch — skip defensively
    }
    for bi in 0..b {
        let x = &xs[bi * cols..(bi + 1) * cols];
        for (i, &xi) in x.iter().enumerate() {
            e.acc.sumsq[i] += xi as f64 * xi as f64;
        }
        e.acc.count += 1;
    }
    let mut failed = false;
    if full {
        match e.dev {
            Some(id) => {
                e.stage.extend_from_slice(xs);
                if e.stage.len() >= STAGE_ROWS * cols {
                    failed = !crate::gpu::gram_add(id, &e.stage, e.stage.len() / cols);
                    e.stage.clear();
                }
            }
            None => gram_upper_add(&mut e.acc.h, xs, b, cols),
        }
    }
    if failed && !st.broken.iter().any(|n| n == name) {
        st.broken.push(name.to_string());
    }
    if full {
        match &mut st.last {
            Some((o, x)) => {
                o.clear();
                o.push_str(name);
                x.clear();
                x.extend_from_slice(xs);
            }
            None => st.last = Some((name.to_string(), xs.to_vec())),
        }
    }
}

/// `H[i][j] += Σ_t x_t[i]·x_t[j]` for `j ≥ i` (upper triangle only; `end()`
/// mirrors it). The batch is transposed so each entry is one contiguous
/// length-`b` dot product, and rows are dealt round-robin over threads so
/// the triangle's uneven row lengths balance out. The former rank-1 update
/// per token streamed the whole `cols²` matrix once per token and made
/// calibrating even a 2B model take hours; this is compute-bound instead.
fn gram_upper_add(h: &mut [f64], xs: &[f32], b: usize, cols: usize) {
    if b == 0 || cols == 0 {
        return;
    }
    let mut xt = vec![0f32; cols * b];
    for t in 0..b {
        let row = &xs[t * cols..(t + 1) * cols];
        for (i, &v) in row.iter().enumerate() {
            xt[i * b + t] = v;
        }
    }
    let nth = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(cols.div_ceil(16))
        .max(1);
    struct SendPtr(*mut f64);
    unsafe impl Sync for SendPtr {}
    unsafe impl Send for SendPtr {}
    let hp = SendPtr(h.as_mut_ptr());
    let hp = &hp;
    let xt = &xt;
    std::thread::scope(|s| {
        for k in 0..nth {
            s.spawn(move || {
                // Rows in pairs; pair p goes to thread p % nth. Each (i, j)
                // belongs to exactly one thread (the owner of row i).
                let mut i = 2 * k;
                while i < cols {
                    // SAFETY: disjoint rows of H per thread (see above).
                    unsafe { gram_pair_dispatch(xt, b, cols, i, hp.0) };
                    i += 2 * nth;
                }
            });
        }
    });
}

unsafe fn gram_pair_dispatch(xt: &[f32], b: usize, cols: usize, i: usize, h: *mut f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { gram_pair_avx2(xt, b, cols, i, h) };
        }
    }
    unsafe { gram_pair(xt, b, cols, i, h) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gram_pair_avx2(xt: &[f32], b: usize, cols: usize, i: usize, h: *mut f64) {
    unsafe { gram_pair(xt, b, cols, i, h) }
}

/// Upper-triangle entries of rows `i` and `i+1`: a 2×4 register block of
/// 8-lane accumulators over the batch axis (10 vector registers — fits the
/// 16 of AVX2), so each loaded activation lane feeds 4 or 2 products
/// instead of one.
#[inline(always)]
unsafe fn gram_pair(xt: &[f32], b: usize, cols: usize, i: usize, h: *mut f64) {
    let two = i + 1 < cols;
    let a0 = &xt[i * b..(i + 1) * b];
    let a1 = if two { &xt[(i + 1) * b..(i + 2) * b] } else { a0 };
    let n8 = b / 8;
    let mut j = i;
    while j + 4 <= cols {
        let ys = [
            &xt[j * b..(j + 1) * b],
            &xt[(j + 1) * b..(j + 2) * b],
            &xt[(j + 2) * b..(j + 3) * b],
            &xt[(j + 3) * b..(j + 4) * b],
        ];
        let mut acc = [[[0f32; 8]; 4]; 2];
        for c in 0..n8 {
            let x0: &[f32; 8] = a0[c * 8..c * 8 + 8].try_into().unwrap();
            let x1: &[f32; 8] = a1[c * 8..c * 8 + 8].try_into().unwrap();
            for q in 0..4 {
                let y: &[f32; 8] = ys[q][c * 8..c * 8 + 8].try_into().unwrap();
                for l in 0..8 {
                    acc[0][q][l] += x0[l] * y[l];
                    acc[1][q][l] += x1[l] * y[l];
                }
            }
        }
        for q in 0..4 {
            let mut s0 = acc[0][q].iter().sum::<f32>();
            let mut s1 = acc[1][q].iter().sum::<f32>();
            for t in n8 * 8..b {
                s0 += a0[t] * ys[q][t];
                s1 += a1[t] * ys[q][t];
            }
            let jj = j + q;
            unsafe { *h.add(i * cols + jj) += s0 as f64 };
            if two && jj > i {
                unsafe { *h.add((i + 1) * cols + jj) += s1 as f64 };
            }
        }
        j += 4;
    }
    while j < cols {
        let y = &xt[j * b..(j + 1) * b];
        unsafe { *h.add(i * cols + j) += dot_f32(a0, y) as f64 };
        if two && j > i {
            unsafe { *h.add((i + 1) * cols + j) += dot_f32(a1, y) as f64 };
        }
        j += 1;
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for l in 0..8 {
            acc[l] += a[c * 8 + l] * b[c * 8 + l];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for k in chunks * 8..a.len() {
        s += a[k] * b[k];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capture registry is process-global: tests that begin/end a pass
    /// take this so the harness's parallel threads do not interleave them.
    static CAPTURE_TEST: Mutex<()> = Mutex::new(());

    /// A filtered capture keeps only the named tensors, accumulates them
    /// exactly as an unfiltered one would, and `end()` clears the filter.
    #[test]
    fn filtered_capture_keeps_only_the_named_tensors() {
        let _g = CAPTURE_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let (b, cols) = (5usize, 6usize);
        let xs: Vec<f32> = (0..b * cols).map(|k| (k as f32 * 0.37).sin()).collect();
        begin_filtered(true, |n| n.ends_with(".keep"));
        accumulate("l0.keep", &xs, b, cols);
        accumulate("l0.drop", &xs, b, cols);
        let got = end();
        assert_eq!(got.len(), 1, "only the filtered name is captured");
        let acc = &got["l0.keep"];
        assert_eq!(acc.count, b);
        let mut want = vec![0f64; cols * cols];
        gram_upper_add(&mut want, &xs, b, cols);
        for i in 0..cols {
            for j in i..cols {
                assert_eq!(acc.h[i * cols + j], want[i * cols + j]);
                assert_eq!(acc.h[j * cols + i], want[i * cols + j], "mirrored");
            }
        }
        begin(true);
        accumulate("l0.drop", &xs, b, cols);
        assert_eq!(end().len(), 1, "a later unfiltered capture sees everything");
    }

    /// Back-to-back projections of one panel share one Hessian; an alias
    /// whose input later differs from its owner's is reported invalid.
    #[test]
    fn identical_inputs_alias_one_hessian() {
        let _g = CAPTURE_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let (b, cols) = (4usize, 8usize);
        let xs: Vec<f32> = (0..b * cols).map(|k| (k as f32 * 0.21).cos()).collect();
        let ys: Vec<f32> = xs.iter().map(|v| v * 1.5 + 0.1).collect();
        begin(true);
        accumulate("l.q", &xs, b, cols);
        accumulate("l.k", &xs, b, cols);
        accumulate("l.o", &ys, b, cols);
        accumulate("l.q", &ys, b, cols);
        accumulate("l.k", &ys, b, cols);
        let c = end_shared();
        assert_eq!(c.alias, vec![("l.k".to_string(), "l.q".to_string())]);
        assert_eq!(c.hess.len(), 2, "q and o own Hessians, k aliases q");
        assert_eq!(c.hess["l.q"].count, 2 * b);
        assert!(c.broken.is_empty());
        begin(true);
        accumulate("l.q", &xs, b, cols);
        accumulate("l.k", &xs, b, cols);
        accumulate("l.q", &ys, b, cols);
        accumulate("l.k", &xs, b, cols);
        let c = end_shared();
        assert_eq!(c.broken, vec!["l.k".to_string()]);
        assert!(c.alias.is_empty(), "a broken alias is not handed out");
        begin(true);
        accumulate("l.q", &xs, b, cols);
        accumulate("l.k", &xs, b, cols);
        let m = end();
        assert_eq!(m["l.k"].h, m["l.q"].h, "end() copies the owner's Hessian");
    }

    /// The device accumulator (two staged batches, a width that is not a
    /// tile multiple) against an f64 reference. Skips without a device.
    #[test]
    fn device_gram_matches_host() {
        let (cols, b1, b2) = (200usize, 37usize, 300usize);
        let Some(id) = crate::gpu::gram_alloc(cols) else {
            eprintln!("no device — skipped");
            return;
        };
        let x: Vec<f32> = (0..(b1 + b2) * cols)
            .map(|k| ((k * 7919 % 1013) as f32 - 506.0) / 97.0)
            .collect();
        assert!(crate::gpu::gram_add(id, &x[..b1 * cols], b1));
        assert!(crate::gpu::gram_add(id, &x[b1 * cols..], b2));
        let h = crate::gpu::gram_take(id).expect("readback");
        let mut worst = 0f64;
        for i in 0..cols {
            for j in 0..cols {
                let r: f64 = (0..b1 + b2).map(|t| x[t * cols + i] as f64 * x[t * cols + j] as f64).sum();
                let e = (h[i * cols + j] - r).abs() / (1.0 + r.abs());
                worst = worst.max(e);
            }
        }
        assert!(worst < 1e-5, "relative error {worst:e}");
    }

    #[test]
    fn gram_matches_rank1_reference() {
        let (b, cols) = (13usize, 37usize);
        let xs: Vec<f32> = (0..b * cols)
            .map(|k| ((k * 7919 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let mut h = vec![0f64; cols * cols];
        gram_upper_add(&mut h, &xs, b, cols);
        for i in 0..cols {
            for j in i..cols {
                let r: f64 = (0..b)
                    .map(|t| xs[t * cols + i] as f64 * xs[t * cols + j] as f64)
                    .sum();
                assert!((h[i * cols + j] - r).abs() < 1e-3 * (1.0 + r.abs()), "{i},{j}");
            }
            for j in 0..i {
                assert_eq!(h[i * cols + j], 0.0);
            }
        }
    }
}
