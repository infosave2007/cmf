//! Whole q4tp experts on the host's cores at the speed of the memory bus: the
//! kernel behind [`QTensor::moe_expert_rows_fast`], for the experts a card's
//! arena does not hold.
//!
//! The row kernels of `moe_gate_up_rows` / `moe_down_rows` spend most of a
//! missed expert outside its weights: a scalar ladder and a 5-bit decode per
//! row, a horizontal sum per tile, the activation split on the calling
//! thread, a vector per output row. Here one dispatch runs the four stages
//! (split the tokens, gate and up, split the gate rows, down) on every core,
//! sixteen rows' ladders are one 512-bit chain, a row's scales come from two
//! table lookups, and the weights are read once: 64 nibbles a step straight
//! into `vpdpbusd` against up to four tokens, the partial sums staying in
//! sixteen float lanes until the row ends.
//!
//! The scales are bit for bit those of `Q4tpView::scales_into` (the same
//! `exp2_128`, the same multiply chain). A row's sum is taken in sixteen
//! lanes instead of tile by tile, so an output differs from the reference
//! kernels by float rounding only.

use super::*;
use core::arch::x86_64::*;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;

/// Rows of a ladder block: the ladders of sixteen rows are one chain.
const BLK: usize = 16;

/// An activation row split for the int8 kernel, and `-8 Σ` of every four of
/// its values: `vpdpbusd` starts from it, so the unsigned nibbles `w + 8`
/// sum to the signed weights' dot.
struct Act {
    s: SplitAct,
    neg: Vec<i32>,
}

fn act_of(x: &[f32]) -> Act {
    let s = split_act(x);
    let neg = s
        .xq
        .chunks_exact(4)
        .map(|c| -8 * (c[0] as i32 + c[1] as i32 + c[2] as i32 + c[3] as i32))
        .collect();
    Act { s, neg }
}

/// A worker's tables: the ladders of a block of rows (`[row][rung]`) and
/// the scales of the row under way, for two matrices.
struct Scratch {
    gtab: [f32; BLK * 32],
    utab: [f32; BLK * 32],
    gsc: Vec<f32>,
    usc: Vec<f32>,
}

impl Scratch {
    /// `gpr`: the most tiles a row has. The decode stores sixteen scales at
    /// a time and the dot loads four, so the buffers carry a margin.
    fn new(gpr: usize) -> Self {
        let n = gpr.next_multiple_of(16) + 16;
        Self {
            gtab: [0.0; BLK * 32],
            utab: [0.0; BLK * 32],
            gsc: vec![0.0; n],
            usc: vec![0.0; n],
        }
    }
}

/// `exp2_128` on the four quarters of a 512-bit vector: the same bits a
/// lane of `ladder_4` gets.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn exp2_512(x: __m512) -> __m512 {
    unsafe {
        let r = _mm512_castps128_ps512(exp2_128(_mm512_castps512_ps128(x)));
        let r = _mm512_insertf32x4::<1>(r, exp2_128(_mm512_extractf32x4_ps::<1>(x)));
        let r = _mm512_insertf32x4::<2>(r, exp2_128(_mm512_extractf32x4_ps::<2>(x)));
        _mm512_insertf32x4::<3>(r, exp2_128(_mm512_extractf32x4_ps::<3>(x)))
    }
}

/// Transpose a 16×16 block of floats: `out[j]` lane `i` = `r[i]` lane `j`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
fn transpose16(r: &[__m512; 16]) -> [__m512; 16] {
    let z = _mm512_setzero_ps();
    let (mut t, mut u, mut o) = ([z; 16], [z; 16], [z; 16]);
    for i in 0..8 {
        t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for i in 0..4 {
        let (a, b) = (_mm512_castps_pd(t[4 * i]), _mm512_castps_pd(t[4 * i + 1]));
        let (c, d) = (_mm512_castps_pd(t[4 * i + 2]), _mm512_castps_pd(t[4 * i + 3]));
        u[4 * i] = _mm512_castpd_ps(_mm512_unpacklo_pd(a, c));
        u[4 * i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(a, c));
        u[4 * i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(b, d));
        u[4 * i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(b, d));
    }
    for m in 0..4 {
        let a = _mm512_shuffle_f32x4::<0x88>(u[m], u[4 + m]);
        let b = _mm512_shuffle_f32x4::<0xdd>(u[m], u[4 + m]);
        let c = _mm512_shuffle_f32x4::<0x88>(u[8 + m], u[12 + m]);
        let d = _mm512_shuffle_f32x4::<0xdd>(u[8 + m], u[12 + m]);
        o[m] = _mm512_shuffle_f32x4::<0x88>(a, c);
        o[8 + m] = _mm512_shuffle_f32x4::<0xdd>(a, c);
        o[4 + m] = _mm512_shuffle_f32x4::<0x88>(b, d);
        o[12 + m] = _mm512_shuffle_f32x4::<0xdd>(b, d);
    }
    o
}

/// The 32-rung ladders of sixteen consecutive rows, `tab[row][rung]`, from
/// their `{f16 lo, f16 step}` params at `params`.
///
/// # Safety
/// `params` holds sixteen rows' params (64 bytes); AVX-512 F/BW/VL.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn ladders16(params: *const u8, tab: &mut [f32; BLK * 32]) {
    unsafe {
        let v = _mm512_loadu_si512(params.cast());
        let lo = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(v));
        let st = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32::<16>(v)));
        let ratio = exp2_512(st);
        let mut t = exp2_512(lo);
        let z = _mm512_setzero_ps();
        let (mut c0, mut c1) = ([z; 16], [z; 16]);
        for c in 0..32 {
            if c > 0 {
                t = _mm512_mul_ps(t, ratio);
            }
            if c < 16 {
                c0[c] = t;
            } else {
                c1[c - 16] = t;
            }
        }
        let (r0, r1) = (transpose16(&c0), transpose16(&c1));
        for i in 0..BLK {
            _mm512_storeu_ps(tab.as_mut_ptr().add(i * 32), r0[i]);
            _mm512_storeu_ps(tab.as_mut_ptr().add(i * 32 + 16), r1[i]);
        }
    }
}

/// A row's tile scales: its 5-bit codes (`stride` bytes at `codes`) looked
/// up in its ladder `tab`, sixteen at a time.
///
/// # Safety
/// `codes` holds `stride` bytes, `tab` 32 floats, `sc` room for
/// `gpr.next_multiple_of(16)` floats; AVX-512 F/BW/VL.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn scales_row(codes: *const u8, stride: usize, gpr: usize, tab: *const f32, sc: *mut f32) {
    unsafe {
        let (t_lo, t_hi) = (_mm512_loadu_ps(tab), _mm512_loadu_ps(tab.add(16)));
        let sh = _mm512_setr_epi64(0, 5, 10, 15, 20, 25, 30, 35);
        let m31 = _mm512_set1_epi32(31);
        let (mut g, mut off) = (0usize, 0usize);
        while g < gpr {
            // Sixteen codes are ten bytes; the masked load stops at the
            // row's last byte.
            let left = stride.saturating_sub(off);
            let k: __mmask16 = if left >= 10 { 0x03FF } else { (1u16 << left) - 1 };
            let x = _mm_maskz_loadu_epi8(k, codes.add(off).cast());
            let w0 = _mm_cvtsi128_si64(x);
            let w1 = _mm_cvtsi128_si64(_mm_srli_si128::<5>(x));
            let a = _mm512_cvtepi64_epi32(_mm512_srlv_epi64(_mm512_set1_epi64(w0), sh));
            let b = _mm512_cvtepi64_epi32(_mm512_srlv_epi64(_mm512_set1_epi64(w1), sh));
            let idx = _mm512_and_si512(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b), m31);
            _mm512_storeu_ps(sc.add(g), _mm512_permutex2var_ps(t_lo, idx, t_hi));
            g += 16;
            off += 10;
        }
    }
}

/// One row of two matrices (gate and up) against `N` activation rows:
/// `Σ (nibble - 8) · scale[tile] · xq`, the weights read once.
///
/// # Safety
/// `gn`/`un` hold `steps * 32` bytes, `gsc`/`usc` `steps * 2 + 2` floats,
/// every `xq` `steps * 64` values and every `neg` `steps * 16`; AVX-512
/// F/BW/VL/VNNI.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn dot2<const N: usize>(
    gn: *const u8,
    un: *const u8,
    gsc: *const f32,
    usc: *const f32,
    steps: usize,
    xq: &[*const i8; N],
    neg: &[*const i32; N],
) -> ([f32; N], [f32; N]) {
    unsafe {
        let m = _mm512_set1_epi16(0x0F0F);
        let pick = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let (mut ag, mut au) = ([_mm512_setzero_ps(); N], [_mm512_setzero_ps(); N]);
        for s in 0..steps {
            // 32 bytes = 64 nibbles = two tiles; a byte's low nibble is the
            // earlier weight.
            let vg = _mm512_cvtepu8_epi16(_mm256_loadu_si256(gn.add(s * 32).cast()));
            let vu = _mm512_cvtepu8_epi16(_mm256_loadu_si256(un.add(s * 32).cast()));
            let wg = _mm512_and_si512(_mm512_or_si512(vg, _mm512_slli_epi16::<4>(vg)), m);
            let wu = _mm512_and_si512(_mm512_or_si512(vu, _mm512_slli_epi16::<4>(vu)), m);
            let sg = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(_mm_loadu_ps(gsc.add(2 * s))));
            let su = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(_mm_loadu_ps(usc.add(2 * s))));
            for k in 0..N {
                let x = _mm512_loadu_si512(xq[k].add(s * 64).cast());
                let n = _mm512_loadu_si512(neg[k].add(s * 16).cast());
                ag[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(n, wg, x)), sg, ag[k]);
                au[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(n, wu, x)), su, au[k]);
            }
        }
        let (mut g, mut u) = ([0f32; N], [0f32; N]);
        for k in 0..N {
            g[k] = _mm512_reduce_add_ps(ag[k]);
            u[k] = _mm512_reduce_add_ps(au[k]);
        }
        (g, u)
    }
}

/// [`dot2`] for one matrix (down).
///
/// # Safety
/// As [`dot2`].
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn dot1<const N: usize>(
    nb: *const u8,
    sc: *const f32,
    steps: usize,
    xq: &[*const i8; N],
    neg: &[*const i32; N],
) -> [f32; N] {
    unsafe {
        let m = _mm512_set1_epi16(0x0F0F);
        let pick = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let mut acc = [_mm512_setzero_ps(); N];
        for s in 0..steps {
            let v = _mm512_cvtepu8_epi16(_mm256_loadu_si256(nb.add(s * 32).cast()));
            let w = _mm512_and_si512(_mm512_or_si512(v, _mm512_slli_epi16::<4>(v)), m);
            let sv = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(_mm_loadu_ps(sc.add(2 * s))));
            for k in 0..N {
                let x = _mm512_loadu_si512(xq[k].add(s * 64).cast());
                let n = _mm512_loadu_si512(neg[k].add(s * 16).cast());
                acc[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(n, w, x)), sv, acc[k]);
            }
        }
        let mut d = [0f32; N];
        for k in 0..N {
            d[k] = _mm512_reduce_add_ps(acc[k]);
        }
        d
    }
}

/// [`dot2`] for a chunk of `N ≤ 4` activation rows.
///
/// # Safety
/// As [`dot2`]; `ch.len() == N`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn dot2_chunk<const N: usize>(
    gn: *const u8,
    un: *const u8,
    gsc: *const f32,
    usc: *const f32,
    steps: usize,
    ch: &[&Act],
) -> ([f32; 4], [f32; 4]) {
    unsafe {
        let xq: [*const i8; N] = std::array::from_fn(|k| ch[k].s.xq.as_ptr());
        let neg: [*const i32; N] = std::array::from_fn(|k| ch[k].neg.as_ptr());
        let (g, u) = dot2::<N>(gn, un, gsc, usc, steps, &xq, &neg);
        let (mut go, mut uo) = ([0f32; 4], [0f32; 4]);
        go[..N].copy_from_slice(&g);
        uo[..N].copy_from_slice(&u);
        (go, uo)
    }
}

/// [`dot1`] for a chunk of `N ≤ 4` activation rows.
///
/// # Safety
/// As [`dot1`]; `ch.len() == N`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn dot1_chunk<const N: usize>(nb: *const u8, sc: *const f32, steps: usize, ch: &[&Act]) -> [f32; 4] {
    unsafe {
        let xq: [*const i8; N] = std::array::from_fn(|k| ch[k].s.xq.as_ptr());
        let neg: [*const i32; N] = std::array::from_fn(|k| ch[k].neg.as_ptr());
        let d = dot1::<N>(nb, sc, steps, &xq, &neg);
        let mut o = [0f32; 4];
        o[..N].copy_from_slice(&d);
        o
    }
}

/// Rows `r0 .. r0 + BLK` of an expert's gate and up against its tokens:
/// `avals[(base + k) * inter + r] = silu(gate_r · x_k) * (up_r · x_k)`.
///
/// # Safety
/// The views are `inter × hid` q4tp planes with `r0 + BLK <= inter`, every
/// act is `hid` wide, `avals` holds `(base + acts.len()) * inter` floats and
/// no other thread writes these rows of these pairs; AVX-512 F/BW/VL/VNNI.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gate_up_block(
    g: &Q4tpView,
    u: &Q4tpView,
    r0: usize,
    hid: usize,
    inter: usize,
    acts: &[&Act],
    avals: SendMut,
    base: usize,
    scr: &mut Scratch,
) {
    unsafe {
        let gpr = hid / GROUP_SIZE;
        let steps = hid / 64;
        let row = gpr * Q4TP_NIB;
        ladders16(g.params.as_ptr().add(r0 * 4), &mut scr.gtab);
        ladders16(u.params.as_ptr().add(r0 * 4), &mut scr.utab);
        for i in 0..BLK {
            let r = r0 + i;
            scales_row(g.codes.as_ptr().add(r * g.stride), g.stride, gpr, scr.gtab.as_ptr().add(i * 32), scr.gsc.as_mut_ptr());
            scales_row(u.codes.as_ptr().add(r * u.stride), u.stride, gpr, scr.utab.as_ptr().add(i * 32), scr.usc.as_mut_ptr());
            let gn = g.nib.as_ptr().add(r * row);
            let un = u.nib.as_ptr().add(r * row);
            let (gsc, usc) = (scr.gsc.as_ptr(), scr.usc.as_ptr());
            let mut k0 = 0;
            while k0 < acts.len() {
                let n = (acts.len() - k0).min(4);
                let ch = &acts[k0..k0 + n];
                let (gv, uv) = match n {
                    1 => dot2_chunk::<1>(gn, un, gsc, usc, steps, ch),
                    2 => dot2_chunk::<2>(gn, un, gsc, usc, steps, ch),
                    3 => dot2_chunk::<3>(gn, un, gsc, usc, steps, ch),
                    _ => dot2_chunk::<4>(gn, un, gsc, usc, steps, ch),
                };
                for (k, a) in ch.iter().enumerate() {
                    let (mut gvv, mut uvv) = (gv[k] * a.s.sx, uv[k] * a.s.sx);
                    for &(j, xv) in &a.s.outliers {
                        let og = q4tp_outlier(g.nib, r, gpr, j, &scr.gsc[..gpr]);
                        let ou = q4tp_outlier(u.nib, r, gpr, j, &scr.usc[..gpr]);
                        gvv += og.0 * og.1 * xv;
                        uvv += ou.0 * ou.1 * xv;
                    }
                    *avals.at((base + k0 + k) * inter + r) = gvv / (1.0 + (-gvv).exp()) * uvv;
                }
                k0 += n;
            }
        }
    }
}

/// Rows `r0 .. r0 + BLK` of an expert's down against the gate rows of its
/// pairs: `out[(base + k) * hid + r] = down_r · a_k`.
///
/// # Safety
/// The view is a `hid × inter` q4tp plane with `r0 + BLK <= hid`, every act
/// is `inter` wide, `out` holds `(base + acts.len()) * hid` floats and no
/// other thread writes these rows of these pairs; AVX-512 F/BW/VL/VNNI.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn down_block(
    d: &Q4tpView,
    r0: usize,
    hid: usize,
    inter: usize,
    acts: &[&Act],
    out: SendMut,
    base: usize,
    scr: &mut Scratch,
) {
    unsafe {
        let gpr = inter / GROUP_SIZE;
        let steps = inter / 64;
        let row = gpr * Q4TP_NIB;
        ladders16(d.params.as_ptr().add(r0 * 4), &mut scr.gtab);
        for i in 0..BLK {
            let r = r0 + i;
            scales_row(d.codes.as_ptr().add(r * d.stride), d.stride, gpr, scr.gtab.as_ptr().add(i * 32), scr.gsc.as_mut_ptr());
            let nb = d.nib.as_ptr().add(r * row);
            let sc = scr.gsc.as_ptr();
            let mut k0 = 0;
            while k0 < acts.len() {
                let n = (acts.len() - k0).min(4);
                let ch = &acts[k0..k0 + n];
                let dv = match n {
                    1 => dot1_chunk::<1>(nb, sc, steps, ch),
                    2 => dot1_chunk::<2>(nb, sc, steps, ch),
                    3 => dot1_chunk::<3>(nb, sc, steps, ch),
                    _ => dot1_chunk::<4>(nb, sc, steps, ch),
                };
                for (k, a) in ch.iter().enumerate() {
                    let mut v = dv[k] * a.s.sx;
                    for &(j, xv) in &a.s.outliers {
                        let (w, s) = q4tp_outlier(d.nib, r, gpr, j, &scr.gsc[..gpr]);
                        v += w * s * xv;
                    }
                    *out.at((base + k0 + k) * hid + r) = v;
                }
                k0 += n;
            }
        }
    }
}

/// Wait until `done` reaches `n`: the stages of one dispatch meet here. A
/// stage is microseconds, so the wait spins before it yields.
fn wait(done: &AtomicUsize, n: usize) {
    let mut spins = 0u32;
    while done.load(Ordering::Acquire) < n {
        if spins < 1 << 14 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

/// The kernel behind [`QTensor::moe_expert_rows_fast`].
pub(super) fn run(
    experts: &[(&QTensor, &QTensor, &QTensor)],
    groups: &[Vec<usize>],
    xs: &[f32],
    out: &mut [f32],
    pool: Option<&Pool>,
) -> bool {
    run_stages(experts, groups, xs, out, pool, None, None)
}

/// [`run`] with the stages open to the tests: `gs_in` replaces the gate/up
/// stage with these gate rows (`[pair][inter]`), `gs_out` receives the gate
/// rows the down stage read.
pub(super) fn run_stages(
    experts: &[(&QTensor, &QTensor, &QTensor)],
    groups: &[Vec<usize>],
    xs: &[f32],
    out: &mut [f32],
    pool: Option<&Pool>,
    gs_in: Option<&[f32]>,
    gs_out: Option<&mut Vec<f32>>,
) -> bool {
    if !(avx512vnni_enabled() && a8w8_enabled()) || experts.is_empty() || experts.len() != groups.len() {
        return false;
    }
    let (inter, hid) = (experts[0].0.rows(), experts[0].0.cols());
    if hid == 0 || inter == 0 || hid % 64 != 0 || inter % 64 != 0 || xs.len() % hid != 0 {
        return false;
    }
    let b = xs.len() / hid;
    let n_pairs: usize = groups.iter().map(|g| g.len()).sum();
    if n_pairs == 0 || out.len() != n_pairs * hid || groups.iter().flatten().any(|&t| t >= b) {
        return false;
    }
    if gs_in.is_some_and(|g| g.len() != n_pairs * inter) {
        return false;
    }
    // A q4tp plane of the kernel's shape, read as the file holds it.
    fn view(t: &QTensor, rows: usize, cols: usize) -> Option<Q4tpView<'_>> {
        let q4tp = matches!(
            t,
            QTensor::Mapped {
                dtype: TensorDtype::Q4TiledP,
                ..
            }
        );
        if !q4tp
            || t.per_tensor_only()
            || t.has_prism_contract()
            || t.exact_cpu_default()
            || t.rows() != rows
            || t.cols() != cols
        {
            return None;
        }
        let bytes = t.quant_bytes();
        let (_, codes_off, stride) = q4tp_sections(rows, cols);
        (bytes.len() >= codes_off + rows * stride).then(|| Q4tpView::new(bytes, rows, cols))
    }
    let mut views = Vec::with_capacity(experts.len() * 3);
    for (g, u, d) in experts {
        let (Some(g), Some(u), Some(d)) = (view(g, inter, hid), view(u, inter, hid), view(d, hid, inter)) else {
            return false;
        };
        views.extend([g, u, d]);
    }
    let mut offs = Vec::with_capacity(experts.len() + 1);
    let mut at = 0usize;
    for g in groups {
        offs.push(at);
        at += g.len();
    }
    offs.push(at);
    let mut used = vec![false; b];
    for &t in groups.iter().flatten() {
        used[t] = true;
    }
    let toks: Vec<usize> = (0..b).filter(|&t| used[t]).collect();
    let tok_acts: Vec<OnceLock<Act>> = (0..b).map(|_| OnceLock::new()).collect();
    let pair_acts: Vec<OnceLock<Act>> = (0..n_pairs).map(|_| OnceLock::new()).collect();
    let mut avals: Vec<f32> = match gs_in {
        Some(g) => g.to_vec(),
        None => vec![0f32; n_pairs * inter],
    };
    let (bg, bd) = (inter / BLK, hid / BLK);
    let n_gu = if gs_in.is_some() { 0 } else { experts.len() * bg };
    let n_dn = experts.len() * bd;
    let (av, ov) = (SendMut(avals.as_mut_ptr()), SendMut(out.as_mut_ptr()));
    let gpr_max = hid.max(inter) / GROUP_SIZE;
    let cur: [AtomicUsize; 4] = std::array::from_fn(|_| AtomicUsize::new(0));
    let done: [AtomicUsize; 4] = std::array::from_fn(|_| AtomicUsize::new(0));
    // One dispatch, four stages; every participant takes work from a stage's
    // cursor until it is empty, then waits for the stage to be done. Nothing
    // in it panics: a participant that left a stage early would hang the rest.
    let work = |_w: usize, _n: usize| {
        let mut scr = Scratch::new(gpr_max);
        let mut ga: Vec<&Act> = Vec::new();
        // 1. the tokens' int8 split
        loop {
            let i = cur[0].fetch_add(1, Ordering::Relaxed);
            if i >= toks.len() {
                break;
            }
            let t = toks[i];
            let _ = tok_acts[t].set(act_of(&xs[t * hid..(t + 1) * hid]));
            done[0].fetch_add(1, Ordering::Release);
        }
        wait(&done[0], toks.len());
        // 2. gate and up, a block of rows of an expert at a time
        let mut held = usize::MAX;
        loop {
            let u0 = cur[1].fetch_add(2, Ordering::Relaxed);
            if u0 >= n_gu {
                break;
            }
            let u1 = (u0 + 2).min(n_gu);
            for unit in u0..u1 {
                let (e, blk) = (unit / bg, unit % bg);
                if groups[e].is_empty() {
                    continue;
                }
                if held != e {
                    ga.clear();
                    ga.extend(groups[e].iter().filter_map(|&t| tok_acts[t].get()));
                    held = e;
                }
                if ga.len() == groups[e].len() {
                    // SAFETY: the views were checked against the shape; a
                    // unit is one worker's, so its rows are written once.
                    unsafe { gate_up_block(&views[e * 3], &views[e * 3 + 1], blk * BLK, hid, inter, &ga, av, offs[e], &mut scr) };
                }
            }
            done[1].fetch_add(u1 - u0, Ordering::Release);
        }
        wait(&done[1], n_gu);
        // 3. the gate rows' int8 split
        loop {
            let p = cur[2].fetch_add(1, Ordering::Relaxed);
            if p >= n_pairs {
                break;
            }
            // SAFETY: stage 2 is done, nothing writes `avals` any more.
            let row = unsafe { std::slice::from_raw_parts(av.at(p * inter), inter) };
            let _ = pair_acts[p].set(act_of(row));
            done[2].fetch_add(1, Ordering::Release);
        }
        wait(&done[2], n_pairs);
        // 4. down
        let mut held = usize::MAX;
        loop {
            let u0 = cur[3].fetch_add(8, Ordering::Relaxed);
            if u0 >= n_dn {
                break;
            }
            let u1 = (u0 + 8).min(n_dn);
            for unit in u0..u1 {
                let (e, blk) = (unit / bd, unit % bd);
                if groups[e].is_empty() {
                    continue;
                }
                if held != e {
                    ga.clear();
                    ga.extend(pair_acts[offs[e]..offs[e + 1]].iter().filter_map(|a| a.get()));
                    held = e;
                }
                if ga.len() == groups[e].len() {
                    // SAFETY: as stage 2, on the output rows.
                    unsafe { down_block(&views[e * 3 + 2], blk * BLK, hid, inter, &ga, ov, offs[e], &mut scr) };
                }
            }
            done[3].fetch_add(u1 - u0, Ordering::Release);
        }
        wait(&done[3], n_dn);
    };
    match pool {
        Some(p) => p.run(&work),
        None => work(0, 1),
    }
    if let Some(g) = gs_out {
        *g = avals;
    }
    true
}

/// The scales of rows `r0 .. r0 + BLK` as the kernel decodes them
/// (`[row][gpr]`); `None` without AVX-512 VNNI.
#[cfg(test)]
pub(super) fn block_scales(v: &Q4tpView, r0: usize, gpr: usize) -> Option<Vec<f32>> {
    if !avx512vnni_enabled() {
        return None;
    }
    let mut scr = Scratch::new(gpr);
    let mut out = Vec::with_capacity(BLK * gpr);
    unsafe {
        ladders16(v.params.as_ptr().add(r0 * 4), &mut scr.gtab);
        for i in 0..BLK {
            scales_row(
                v.codes.as_ptr().add((r0 + i) * v.stride),
                v.stride,
                gpr,
                scr.gtab.as_ptr().add(i * 32),
                scr.gsc.as_mut_ptr(),
            );
            out.extend_from_slice(&scr.gsc[..gpr]);
        }
    }
    Some(out)
}
