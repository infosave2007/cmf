//! Whole q4tp experts on the host's cores: the kernel behind [`QTensor::moe_expert_rows_fast`], for the experts a card's
//! arena does not hold.
//!
//! The row kernels of `moe_gate_up_rows` / `moe_down_rows` spend most of a
//! missed expert outside its weights: a scalar ladder and a 5-bit decode per
//! row, a horizontal sum per tile, the activation split on the calling
//! thread, a vector per output row. Here sixteen rows' ladders are one
//! 512-bit chain, a row's scales come from two table lookups, and the
//! weights are read once and stay packed: 64 bytes (128 nibbles) a step, the
//! low and the high nibbles each straight into `vpdpbusd` against the even
//! and the odd columns of up to four tokens, the partial sums staying in
//! sixteen float lanes until the row ends.
//!
//! One dispatch, no barrier in it. The caller splits the few token rows
//! before it wakes the cores. The cores take blocks of gate and up rows from
//! one cursor; the core that finishes an expert's last block splits that
//! expert's gate rows and publishes the expert, and a core that finds the
//! cursor empty goes on to the down rows of the published experts, in the
//! order they were published. Nothing waits for a stage to end: a call of a
//! dozen experts is a few hundred microseconds, and four barriers were a
//! quarter of it.
//!
//! The scales are bit for bit those of `Q4tpView::scales_into` (the same
//! `exp2_128`, the same multiply chain). A row's sum is taken in sixteen
//! lanes instead of tile by tile, the int8 split sums the squares in lanes,
//! and the SiLU's exponent is a sixteen-lane polynomial (2e-7 of the
//! library's), so an output differs from the reference kernels by float
//! rounding only.

use super::*;
use core::arch::x86_64::*;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize};
use std::time::Instant;

/// The kernel's own clock (`CMF_QWEN_PROF`): where a dispatch's time goes,
/// summed over the participants since the last [`prof_take`].
pub(super) static PROF_ON: AtomicBool = AtomicBool::new(false);

/// What [`prof_take`] returns, in this order: dispatches, participants,
/// experts, (expert, token) pairs, token rows, outliers of the token rows
/// and of the gate rows, then nanoseconds: the caller's setup and the
/// tokens' split in it, the dispatch's wall, and per participant the delay
/// until it entered, its gate and up blocks, the gate rows it split, its
/// down blocks and its waits for an expert to be published.
pub(super) const PROF_SLOTS: usize = 15;
#[allow(clippy::declare_interior_mutable_const)]
const PROF_ZERO: AtomicU64 = AtomicU64::new(0);
static PROF: [AtomicU64; PROF_SLOTS] = [PROF_ZERO; PROF_SLOTS];

/// The clock's sums since the last call, and zero them.
pub(super) fn prof_take() -> [u64; PROF_SLOTS] {
    std::array::from_fn(|i| PROF[i].swap(0, Ordering::Relaxed))
}

/// The kernel's knobs: bytes of weights a row prefetches ahead of itself
/// (`CMF_QWEN_CPU_PF`, 0 = none), and the blocks a core takes at a time of
/// gate/up and of down (`CMF_QWEN_CPU_CLAIM=<gate/up>:<down>`).
static TUNE: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(8), AtomicUsize::new(32)];

fn tune() -> (usize, usize, usize) {
    static ENV: OnceLock<()> = OnceLock::new();
    ENV.get_or_init(|| {
        if let Some(v) = std::env::var("CMF_QWEN_CPU_PF").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            TUNE[0].store(v, Ordering::Relaxed);
        }
        if let Ok(v) = std::env::var("CMF_QWEN_CPU_CLAIM") {
            for (i, x) in v.split(':').take(2).enumerate() {
                if let Ok(n) = x.trim().parse::<usize>() {
                    TUNE[1 + i].store(n.max(1), Ordering::Relaxed);
                }
            }
        }
    });
    (TUNE[0].load(Ordering::Relaxed), TUNE[1].load(Ordering::Relaxed).max(1), TUNE[2].load(Ordering::Relaxed).max(1))
}

/// Set a knob of [`tune`] (0: the prefetch distance, 1 and 2: the claims)
/// for the calls to come, and return what it was: a measurement compares
/// them in one process.
pub(super) fn set_tune(i: usize, v: usize) -> usize {
    let _ = tune();
    TUNE.get(i).map_or(0, |t| t.swap(v, Ordering::Relaxed))
}

/// Rows of a ladder block: the ladders of sixteen rows are one chain.
const BLK: usize = 16;

/// A pointer the cores of one dispatch share: what they write through it
/// never overlaps.
struct Raw<T>(*mut T);
impl<T> Clone for Raw<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Raw<T> {}
unsafe impl<T> Send for Raw<T> {}
unsafe impl<T> Sync for Raw<T> {}

impl<T> Raw<T> {
    #[inline]
    fn at(self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

/// Activation rows split for the int8 kernel, flat and kept between calls:
/// row `s` is `xq[s * len ..]`, its scale `sx[s]`, its `on[s]` outliers
/// (index, value) at `out[s * cap ..]`. A row's values lie as the kernel's
/// step reads them: of every 128 columns the 64 even ones, then the 64 odd
/// ones — a packed byte holds an even column's nibble low and the next odd
/// column's high, so 64 bytes of weights meet both halves without being
/// unpacked. `neg[s * len / 8 ..]` holds `-8 Σ` of every eight columns:
/// `vpdpbusd` starts from it, so the unsigned nibbles `w + 8` sum to the
/// signed weights' dot.
#[derive(Default)]
struct Acts {
    xq: Vec<i8>,
    neg: Vec<i32>,
    sx: Vec<f32>,
    on: Vec<u32>,
    out: Vec<(u32, f32)>,
}

/// [`Acts`] as the cores see it.
#[derive(Clone, Copy)]
struct ActsRaw {
    len: usize,
    cap: usize,
    xq: Raw<i8>,
    neg: Raw<i32>,
    sx: Raw<f32>,
    on: Raw<u32>,
    out: Raw<(u32, f32)>,
}

impl Acts {
    /// Room for `n` rows of `len` values. An outlier is beyond eight times
    /// the row's rms, so a row has fewer than `len / 64` of them.
    fn fit(&mut self, n: usize, len: usize) -> ActsRaw {
        let cap = len / 64 + 1;
        if self.xq.len() < n * len {
            self.xq.resize(n * len, 0);
        }
        if self.neg.len() < n * (len / 8) {
            self.neg.resize(n * (len / 8), 0);
        }
        if self.sx.len() < n {
            self.sx.resize(n, 0.0);
        }
        if self.on.len() < n {
            self.on.resize(n, 0);
        }
        if self.out.len() < n * cap {
            self.out.resize(n * cap, (0, 0.0));
        }
        ActsRaw {
            len,
            cap,
            xq: Raw(self.xq.as_mut_ptr()),
            neg: Raw(self.neg.as_mut_ptr()),
            sx: Raw(self.sx.as_mut_ptr()),
            on: Raw(self.on.as_mut_ptr()),
            out: Raw(self.out.as_mut_ptr()),
        }
    }
}

/// `split_act` for one row of `a`, sixteen values a step: the same rms
/// threshold, outliers, scale and rounding (the squares are summed in
/// sixteen lanes of doubles, so the threshold can differ from the scalar
/// split's in its last bit). A row of denormals whose squares vanish makes
/// every value an outlier; only the first `cap` are kept, the rest read as
/// zero.
///
/// # Safety
/// `x` holds `a.len` floats, a multiple of 128; `a` was fitted for rows of
/// that length and more than `slot` of them, and no other thread touches
/// row `slot`; AVX-512 F/BW/VL/VNNI.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn split_row(x: *const f32, a: ActsRaw, slot: usize) {
    unsafe {
        let len = a.len;
        let (mut s0, mut s1) = (_mm512_setzero_pd(), _mm512_setzero_pd());
        let mut j = 0;
        while j < len {
            let v = _mm512_loadu_ps(x.add(j));
            let q = _mm512_mul_ps(v, v);
            s0 = _mm512_add_pd(s0, _mm512_cvtps_pd(_mm512_castps512_ps256(q)));
            s1 = _mm512_add_pd(s1, _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(q)))));
            j += 16;
        }
        let sum = _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
        let rms = (sum / len.max(1) as f64).sqrt() as f32;
        let thr = 8.0 * rms;
        let vthr = _mm512_set1_ps(thr);
        // the bulk's absmax and the outliers; a NaN is neither
        let mut acc = _mm512_setzero_ps();
        let outp = a.out.at(slot * a.cap);
        let mut n_out = 0usize;
        j = 0;
        while j < len {
            let v = _mm512_loadu_ps(x.add(j));
            let av = _mm512_abs_ps(v);
            let m = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(av, vthr);
            acc = _mm512_max_ps(_mm512_maskz_mov_ps(!m, av), acc);
            let mut mm = m as u32;
            while mm != 0 {
                let i = mm.trailing_zeros() as usize;
                mm &= mm - 1;
                if n_out < a.cap {
                    *outp.add(n_out) = ((j + i) as u32, *x.add(j + i));
                    n_out += 1;
                }
            }
            j += 16;
        }
        let amax = _mm512_reduce_max_ps(acc);
        let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        *a.sx.at(slot) = sx;
        *a.on.at(slot) = n_out as u32;
        // `q8_round(v / sx)`: clamp (a NaN stays one and rounds to 0), then
        // halves away from zero; an outlier's place is 0
        let vinv = _mm512_set1_ps(1.0 / sx);
        let (lo, hi) = (_mm512_set1_ps(-127.0), _mm512_set1_ps(127.0));
        let (half, nhalf) = (_mm512_set1_ps(0.5), _mm512_set1_ps(-0.5));
        let one = _mm512_set1_epi32(1);
        let ones8 = _mm512_set1_epi8(1);
        let zero = _mm512_setzero_si512();
        let xqp = a.xq.at(slot * len);
        let negp = a.neg.at(slot * (len / 8));
        let low = _mm512_set1_epi16(0x00FF);
        let halves = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
        let mut first = zero;
        j = 0;
        while j < len {
            let mut w = zero;
            for q in 0..4 {
                let v = _mm512_loadu_ps(x.add(j + q * 16));
                let m = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_abs_ps(v), vthr);
                let t = _mm512_min_ps(hi, _mm512_max_ps(lo, _mm512_mul_ps(v, vinv)));
                let i = _mm512_cvttps_epi32(t);
                let f = _mm512_sub_ps(t, _mm512_cvtepi32_ps(i));
                let up = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(f, half);
                let dn = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(f, nhalf);
                let i = _mm512_mask_add_epi32(i, up, i, one);
                let i = _mm512_mask_sub_epi32(i, dn, i, one);
                let b = _mm512_cvtepi32_epi8(_mm512_maskz_mov_epi32(!m, i));
                w = match q {
                    0 => _mm512_castsi128_si512(b),
                    1 => _mm512_inserti32x4::<1>(w, b),
                    2 => _mm512_inserti32x4::<2>(w, b),
                    _ => _mm512_inserti32x4::<3>(w, b),
                };
            }
            if j % 128 == 0 {
                first = w;
            } else {
                // 128 columns: the even ones, then the odd ones (a pack is
                // by 128-bit lanes, the permute puts its halves in order)
                let ev = _mm512_packus_epi16(_mm512_and_si512(first, low), _mm512_and_si512(w, low));
                let od = _mm512_packus_epi16(_mm512_srli_epi16::<8>(first), _mm512_srli_epi16::<8>(w));
                let ev = _mm512_permutexvar_epi64(halves, ev);
                let od = _mm512_permutexvar_epi64(halves, od);
                _mm512_storeu_si512(xqp.add(j - 64).cast(), ev);
                _mm512_storeu_si512(xqp.add(j).cast(), od);
                // a lane of `ev` and the same lane of `od` are eight columns in a row
                let d = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(zero, ones8, ev), ones8, od);
                _mm512_storeu_si512(negp.add((j - 64) / 8).cast(), _mm512_sub_epi32(zero, _mm512_slli_epi32::<3>(d)));
            }
            j += 64;
        }
    }
}

/// A worker's tables, kept between calls: the ladders of a block of rows
/// (`[row][rung]`) and the scales of the row under way, for two matrices,
/// and a block's gate and up sums (`[pair][row]`) until its SiLU.
struct Scratch {
    gtab: [f32; BLK * 32],
    utab: [f32; BLK * 32],
    gsc: Vec<f32>,
    usc: Vec<f32>,
    gb: Vec<f32>,
    ub: Vec<f32>,
}

impl Scratch {
    const fn empty() -> Self {
        Self {
            gtab: [0.0; BLK * 32],
            utab: [0.0; BLK * 32],
            gsc: Vec::new(),
            usc: Vec::new(),
            gb: Vec::new(),
            ub: Vec::new(),
        }
    }

    /// `gpr`: the most tiles a row has, `pairs`: the most tokens an expert
    /// serves. The decode stores sixteen scales at a time and the dot loads
    /// four, so the scale buffers carry a margin.
    fn fit(&mut self, gpr: usize, pairs: usize) {
        let n = gpr.next_multiple_of(16) + 16;
        if self.gsc.len() < n {
            self.gsc.resize(n, 0.0);
            self.usc.resize(n, 0.0);
        }
        let m = pairs.max(1) * BLK;
        if self.gb.len() < m {
            self.gb.resize(m, 0.0);
            self.ub.resize(m, 0.0);
        }
    }

    #[cfg(test)]
    fn new(gpr: usize) -> Self {
        let mut s = Self::empty();
        s.fit(gpr, 1);
        s
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::empty()) };
    static CALL: RefCell<Call> = RefCell::new(Call::default());
}

/// What a call builds before its dispatch, kept by the calling thread
/// between calls: a layer's misses come every few hundred microseconds, and
/// fresh buffers a call were page faults and zeroing.
#[derive(Default)]
struct Call {
    /// The split rows of the tokens in use and of the pairs' gate rows.
    tok: Acts,
    pair: Acts,
    /// The gate rows, `[pair][inter]`.
    avals: Vec<f32>,
    /// Where an expert's pairs start, the experts that have any, the tokens
    /// in use, a token's row in `tok`, that row of every pair, and the
    /// pairs' own rows in `pair`.
    offs: Vec<usize>,
    act: Vec<u32>,
    toks: Vec<u32>,
    slot_of: Vec<u32>,
    pair_tok: Vec<u32>,
    pair_ix: Vec<u32>,
    /// Gate/up blocks an expert still has out, and the experts whose gate
    /// rows are split, in the order they were published.
    left: Vec<AtomicU32>,
    queue: Vec<AtomicUsize>,
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

/// Where code `i` of sixteen 5-bit codes starts in their ten bytes: the two
/// bytes that hold it, for a byte shuffle into 32-bit lanes, and its shift
/// there.
const fn code_bytes() -> [u8; 64] {
    let mut t = [0x80u8; 64];
    let mut i = 0;
    while i < 16 {
        t[4 * i] = (5 * i / 8) as u8;
        t[4 * i + 1] = (5 * i / 8) as u8 + 1;
        i += 1;
    }
    t
}
const fn code_shifts() -> [u32; 16] {
    let mut t = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = (5 * i % 8) as u32;
        i += 1;
    }
    t
}
static CODE_BYTES: [u8; 64] = code_bytes();
static CODE_SHIFTS: [u32; 16] = code_shifts();

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
        let pick = _mm512_loadu_si512(CODE_BYTES.as_ptr().cast());
        let sh = _mm512_loadu_si512(CODE_SHIFTS.as_ptr().cast());
        let m31 = _mm512_set1_epi32(31);
        let (mut g, mut off) = (0usize, 0usize);
        while g < gpr {
            // Sixteen codes are ten bytes; the masked load stops at the
            // row's last byte. Every quarter of the vector holds the ten
            // bytes, the shuffle puts a code's two bytes into its lane.
            let left = stride.saturating_sub(off);
            let k: __mmask16 = if left >= 10 { 0x03FF } else { (1u16 << left) - 1 };
            let x = _mm512_broadcast_i32x4(_mm_maskz_loadu_epi8(k, codes.add(off).cast()));
            let idx = _mm512_and_si512(_mm512_srlv_epi32(_mm512_shuffle_epi8(x, pick), sh), m31);
            _mm512_storeu_ps(sc.add(g), _mm512_permutex2var_ps(t_lo, idx, t_hi));
            g += 16;
            off += 10;
        }
    }
}

/// `e^x` in sixteen lanes: the argument reduced by a whole power of two, a
/// degree-5 polynomial on the rest (the Cephes single-precision one, 2e-7
/// of the library's value). The argument is clamped to the floats' range; a
/// NaN stays one.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::excessive_precision)]
fn exp_512(x: __m512) -> __m512 {
    let x = _mm512_min_ps(_mm512_set1_ps(88.376_26), _mm512_max_ps(_mm512_set1_ps(-87.336_54), x));
    let n = _mm512_roundscale_ps::<0x08>(_mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)));
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693_359_375), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm512_set1_ps(1.987_569_15e-4);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.398_199_950_7e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.333_451_907_3e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.166_579_589_4e-2));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.666_666_545_9e-1));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.000_000_120_1e-1));
    let y = _mm512_add_ps(_mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r), _mm512_set1_ps(1.0));
    _mm512_scalef_ps(y, n)
}

/// Ask for the cache lines of `bytes` bytes at `p`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(unused_unsafe)]
fn prefetch(p: *const u8, bytes: usize) {
    let mut o = 0;
    while o < bytes {
        // a prefetch of an address outside the plane is a no-op, not a fault
        unsafe { _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(o).cast()) };
        o += 64;
    }
}

/// One row of two matrices (gate and up) against `N` activation rows:
/// `Σ (nibble - 8) · scale[tile] · xq`, the weights read once and never
/// unpacked: a step is 64 bytes = 128 weights = four tiles, the low nibbles
/// against the even columns and the high ones against the odd columns, so a
/// 32-bit lane sums eight columns in a row and four lanes are a tile.
///
/// # Safety
/// `gn`/`un` hold `steps * 64` bytes, `gsc`/`usc` `steps * 4` floats, every
/// `xq` `steps * 128` values laid as [`Acts`] says and every `neg`
/// `steps * 16`; AVX-512 F/BW/VL/VNNI.
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
        let m = _mm512_set1_epi8(0x0F);
        let pick = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        let (mut ag, mut au) = ([_mm512_setzero_ps(); N], [_mm512_setzero_ps(); N]);
        for s in 0..steps {
            // a byte's low nibble is the earlier (even) weight
            let vg = _mm512_loadu_si512(gn.add(s * 64).cast());
            let vu = _mm512_loadu_si512(un.add(s * 64).cast());
            let (gl, gh) = (_mm512_and_si512(vg, m), _mm512_and_si512(_mm512_srli_epi16::<4>(vg), m));
            let (ul, uh) = (_mm512_and_si512(vu, m), _mm512_and_si512(_mm512_srli_epi16::<4>(vu), m));
            let sg = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(_mm_loadu_ps(gsc.add(4 * s))));
            let su = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(_mm_loadu_ps(usc.add(4 * s))));
            for k in 0..N {
                let xe = _mm512_loadu_si512(xq[k].add(s * 128).cast());
                let xo = _mm512_loadu_si512(xq[k].add(s * 128 + 64).cast());
                let n = _mm512_loadu_si512(neg[k].add(s * 16).cast());
                let dg = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(n, gl, xe), gh, xo);
                let du = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(n, ul, xe), uh, xo);
                ag[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dg), sg, ag[k]);
                au[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(du), su, au[k]);
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
        let m = _mm512_set1_epi8(0x0F);
        let pick = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        let mut acc = [_mm512_setzero_ps(); N];
        for s in 0..steps {
            let v = _mm512_loadu_si512(nb.add(s * 64).cast());
            let (wl, wh) = (_mm512_and_si512(v, m), _mm512_and_si512(_mm512_srli_epi16::<4>(v), m));
            let sv = _mm512_permutexvar_ps(pick, _mm512_castps128_ps512(_mm_loadu_ps(sc.add(4 * s))));
            for k in 0..N {
                let xe = _mm512_loadu_si512(xq[k].add(s * 128).cast());
                let xo = _mm512_loadu_si512(xq[k].add(s * 128 + 64).cast());
                let n = _mm512_loadu_si512(neg[k].add(s * 16).cast());
                let d = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(n, wl, xe), wh, xo);
                acc[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), sv, acc[k]);
            }
        }
        let mut d = [0f32; N];
        for k in 0..N {
            d[k] = _mm512_reduce_add_ps(acc[k]);
        }
        d
    }
}

/// [`dot2`] for a chunk of `N ≤ 4` activation rows, the rows `ch` of `a`.
///
/// # Safety
/// As [`dot2`]; `ch.len() == N`, every index a row of `a`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn dot2_chunk<const N: usize>(
    gn: *const u8,
    un: *const u8,
    gsc: *const f32,
    usc: *const f32,
    steps: usize,
    a: ActsRaw,
    ch: &[u32],
) -> ([f32; 4], [f32; 4]) {
    unsafe {
        let xq: [*const i8; N] = std::array::from_fn(|k| a.xq.at(ch[k] as usize * a.len) as *const i8);
        let neg: [*const i32; N] = std::array::from_fn(|k| a.neg.at(ch[k] as usize * (a.len / 8)) as *const i32);
        let (g, u) = dot2::<N>(gn, un, gsc, usc, steps, &xq, &neg);
        let (mut go, mut uo) = ([0f32; 4], [0f32; 4]);
        go[..N].copy_from_slice(&g);
        uo[..N].copy_from_slice(&u);
        (go, uo)
    }
}

/// [`dot1`] for a chunk of `N ≤ 4` activation rows, the rows `ch` of `a`.
///
/// # Safety
/// As [`dot1`]; `ch.len() == N`, every index a row of `a`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn dot1_chunk<const N: usize>(nb: *const u8, sc: *const f32, steps: usize, a: ActsRaw, ch: &[u32]) -> [f32; 4] {
    unsafe {
        let xq: [*const i8; N] = std::array::from_fn(|k| a.xq.at(ch[k] as usize * a.len) as *const i8);
        let neg: [*const i32; N] = std::array::from_fn(|k| a.neg.at(ch[k] as usize * (a.len / 8)) as *const i32);
        let d = dot1::<N>(nb, sc, steps, &xq, &neg);
        let mut o = [0f32; 4];
        o[..N].copy_from_slice(&d);
        o
    }
}

/// Rows `r0 .. r0 + BLK` of an expert's gate and up against its tokens (the
/// rows `ix` of `ta`): `avals[(base + k) * inter + r] = silu(gate_r · x_k) *
/// (up_r · x_k)`. `pf`: bytes of weights a row asks for ahead of itself.
///
/// # Safety
/// The views are `inter × hid` q4tp planes with `r0 + BLK <= inter`, `ta`
/// holds rows of `hid` values, `avals` holds `(base + ix.len()) * inter`
/// floats, `scr` is fitted for the planes and `ix.len()` pairs, and no other
/// thread writes these rows of these pairs; AVX-512 F/BW/VL/VNNI.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gate_up_block(
    g: &Q4tpView,
    u: &Q4tpView,
    r0: usize,
    hid: usize,
    inter: usize,
    ta: ActsRaw,
    ix: &[u32],
    avals: Raw<f32>,
    base: usize,
    scr: &mut Scratch,
    pf: usize,
) {
    unsafe {
        let gpr = hid / GROUP_SIZE;
        let steps = hid / 128;
        let row = gpr * Q4TP_NIB;
        ladders16(g.params.as_ptr().add(r0 * 4), &mut scr.gtab);
        ladders16(u.params.as_ptr().add(r0 * 4), &mut scr.utab);
        let (gb, ub) = (scr.gb.as_mut_ptr(), scr.ub.as_mut_ptr());
        for i in 0..BLK {
            let r = r0 + i;
            scales_row(g.codes.as_ptr().add(r * g.stride), g.stride, gpr, scr.gtab.as_ptr().add(i * 32), scr.gsc.as_mut_ptr());
            scales_row(u.codes.as_ptr().add(r * u.stride), u.stride, gpr, scr.utab.as_ptr().add(i * 32), scr.usc.as_mut_ptr());
            let gn = g.nib.as_ptr().add(r * row);
            let un = u.nib.as_ptr().add(r * row);
            if pf != 0 {
                prefetch(gn.wrapping_add(pf), row);
                prefetch(un.wrapping_add(pf), row);
            }
            let (gsc, usc) = (scr.gsc.as_ptr(), scr.usc.as_ptr());
            let mut k0 = 0;
            while k0 < ix.len() {
                let n = (ix.len() - k0).min(4);
                let ch = &ix[k0..k0 + n];
                let (gv, uv) = match n {
                    1 => dot2_chunk::<1>(gn, un, gsc, usc, steps, ta, ch),
                    2 => dot2_chunk::<2>(gn, un, gsc, usc, steps, ta, ch),
                    3 => dot2_chunk::<3>(gn, un, gsc, usc, steps, ta, ch),
                    _ => dot2_chunk::<4>(gn, un, gsc, usc, steps, ta, ch),
                };
                for (k, &s) in ch.iter().enumerate() {
                    let s = s as usize;
                    let sx = *ta.sx.at(s);
                    let (mut gvv, mut uvv) = (gv[k] * sx, uv[k] * sx);
                    let outl = std::slice::from_raw_parts(ta.out.at(s * ta.cap) as *const (u32, f32), *ta.on.at(s) as usize);
                    for &(j, xv) in outl {
                        let og = q4tp_outlier(g.nib, r, gpr, j as usize, &scr.gsc[..gpr]);
                        let ou = q4tp_outlier(u.nib, r, gpr, j as usize, &scr.usc[..gpr]);
                        gvv += og.0 * og.1 * xv;
                        uvv += ou.0 * ou.1 * xv;
                    }
                    *gb.add((k0 + k) * BLK + i) = gvv;
                    *ub.add((k0 + k) * BLK + i) = uvv;
                }
                k0 += n;
            }
        }
        // the block's SiLU, a pair's sixteen rows at a time
        let one = _mm512_set1_ps(1.0);
        for k in 0..ix.len() {
            let gq = _mm512_loadu_ps(gb.add(k * BLK));
            let uq = _mm512_loadu_ps(ub.add(k * BLK));
            let den = _mm512_add_ps(one, exp_512(_mm512_sub_ps(_mm512_setzero_ps(), gq)));
            _mm512_storeu_ps(avals.at((base + k) * inter + r0), _mm512_mul_ps(_mm512_div_ps(gq, den), uq));
        }
    }
}

/// Rows `r0 .. r0 + BLK` of an expert's down against the gate rows of its
/// pairs (the rows `ix` of `pa`): `out[(base + k) * hid + r] = down_r · a_k`.
///
/// # Safety
/// The view is a `hid × inter` q4tp plane with `r0 + BLK <= hid`, `pa` holds
/// rows of `inter` values, `out` holds `(base + ix.len()) * hid` floats,
/// `scr` is fitted for the plane, and no other thread writes these rows of
/// these pairs; AVX-512 F/BW/VL/VNNI.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn down_block(
    d: &Q4tpView,
    r0: usize,
    hid: usize,
    inter: usize,
    pa: ActsRaw,
    ix: &[u32],
    out: Raw<f32>,
    base: usize,
    scr: &mut Scratch,
    pf: usize,
) {
    unsafe {
        let gpr = inter / GROUP_SIZE;
        let steps = inter / 128;
        let row = gpr * Q4TP_NIB;
        ladders16(d.params.as_ptr().add(r0 * 4), &mut scr.gtab);
        for i in 0..BLK {
            let r = r0 + i;
            scales_row(d.codes.as_ptr().add(r * d.stride), d.stride, gpr, scr.gtab.as_ptr().add(i * 32), scr.gsc.as_mut_ptr());
            let nb = d.nib.as_ptr().add(r * row);
            if pf != 0 {
                prefetch(nb.wrapping_add(pf), row);
            }
            let sc = scr.gsc.as_ptr();
            let mut k0 = 0;
            while k0 < ix.len() {
                let n = (ix.len() - k0).min(4);
                let ch = &ix[k0..k0 + n];
                let dv = match n {
                    1 => dot1_chunk::<1>(nb, sc, steps, pa, ch),
                    2 => dot1_chunk::<2>(nb, sc, steps, pa, ch),
                    3 => dot1_chunk::<3>(nb, sc, steps, pa, ch),
                    _ => dot1_chunk::<4>(nb, sc, steps, pa, ch),
                };
                for (k, &s) in ch.iter().enumerate() {
                    let s = s as usize;
                    let mut v = dv[k] * *pa.sx.at(s);
                    let outl = std::slice::from_raw_parts(pa.out.at(s * pa.cap) as *const (u32, f32), *pa.on.at(s) as usize);
                    for &(j, xv) in outl {
                        let (w, sc) = q4tp_outlier(d.nib, r, gpr, j as usize, &scr.gsc[..gpr]);
                        v += w * sc * xv;
                    }
                    *out.at((base + k0 + k) * hid + r) = v;
                }
                k0 += n;
            }
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
    if hid == 0 || inter == 0 || hid % 128 != 0 || inter % 128 != 0 || xs.len() % hid != 0 {
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
    let prof = PROF_ON.load(Ordering::Relaxed);
    let t_call = Instant::now();
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
    let (pf, claim_gu, claim_dn) = tune();
    let (bg, bd) = (inter / BLK, hid / BLK);
    let gpr_max = hid.max(inter) / GROUP_SIZE;
    let max_group = groups.iter().map(|g| g.len()).max().unwrap_or(1);
    let ov = Raw(out.as_mut_ptr());
    CALL.with(|c| {
        let st = &mut *c.borrow_mut();
        // the pairs of every expert, the experts that have any, the tokens in use
        st.offs.clear();
        st.act.clear();
        st.toks.clear();
        st.pair_tok.clear();
        st.slot_of.clear();
        st.slot_of.resize(b, u32::MAX);
        let mut at = 0usize;
        for (e, g) in groups.iter().enumerate() {
            st.offs.push(at);
            at += g.len();
            if !g.is_empty() {
                st.act.push(e as u32);
            }
            for &t in g {
                if st.slot_of[t] == u32::MAX {
                    st.slot_of[t] = st.toks.len() as u32;
                    st.toks.push(t as u32);
                }
                st.pair_tok.push(st.slot_of[t]);
            }
        }
        st.offs.push(at);
        for p in st.pair_ix.len()..n_pairs {
            st.pair_ix.push(p as u32);
        }
        if st.avals.len() < n_pairs * inter {
            st.avals.resize(n_pairs * inter, 0.0);
        }
        let n_act = st.act.len();
        if st.left.len() < n_act {
            st.left.resize_with(n_act, || AtomicU32::new(0));
            st.queue.resize_with(n_act, || AtomicUsize::new(usize::MAX));
        }
        let ta = st.tok.fit(st.toks.len(), hid);
        let pa = st.pair.fit(n_pairs, inter);
        let av = Raw(st.avals.as_mut_ptr());
        // the tokens' int8 split: a few rows, cheaper here than a stage of the dispatch
        let t_split = Instant::now();
        for (s, &t) in st.toks.iter().enumerate() {
            // SAFETY: `t < b`, `xs` holds `b` rows of `hid`, `ta` was fitted for the tokens in use.
            unsafe { split_row(xs.as_ptr().add(t as usize * hid), ta, s) };
        }
        let split_ns = t_split.elapsed().as_nanos() as u64;
        let q_len = AtomicUsize::new(0);
        let n_gu = match gs_in {
            Some(g) => {
                // the gate rows are given: every expert is published before the dispatch
                st.avals[..n_pairs * inter].copy_from_slice(g);
                for p in 0..n_pairs {
                    // SAFETY: `avals` holds `n_pairs` rows of `inter`, `pa` was fitted for them.
                    unsafe { split_row(av.at(p * inter), pa, p) };
                }
                for (i, q) in st.queue[..n_act].iter().enumerate() {
                    q.store(i, Ordering::Relaxed);
                }
                q_len.store(n_act, Ordering::Relaxed);
                0
            }
            None => {
                for i in 0..n_act {
                    st.left[i].store(bg as u32, Ordering::Relaxed);
                    st.queue[i].store(usize::MAX, Ordering::Relaxed);
                }
                n_act * bg
            }
        };
        // A pool on two NUMA nodes (`Pool::bind_two_nodes`): the lower half of every matrix's rows is the home
        // side's work, the upper half the far side's (whoever bound the pool put those rows' bytes on the far
        // node). A core works its own side and then helps the other, so the arithmetic and the order of every
        // row's sum are the ones of a single side.
        let sides = if pool.is_some_and(|p| p.two_nodes()) && bg % 2 == 0 && bd % 2 == 0 { 2 } else { 1 };
        let (sg, sd) = (bg / sides, bd / sides);
        let (n_gu, n_dn) = (n_gu / sides, n_act * sd);
        // a core takes at most half its even share at a time: a call of few experts leaves every core a block
        let cores = pool.map_or(1, |p| p.n_workers() + 1).div_ceil(sides);
        let claim_gu = claim_gu.min((n_gu / (2 * cores)).max(1));
        let claim_dn = claim_dn.min((n_dn / (2 * cores)).max(1));
        let (offs, act, pair_tok, pair_ix) = (&st.offs[..], &st.act[..], &st.pair_tok[..], &st.pair_ix[..]);
        let (left, queue) = (&st.left[..], &st.queue[..]);
        let cur_gu = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let cur_dn = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let views = &views[..];
        let t_disp = Instant::now();
        if prof {
            PROF[7].fetch_add((t_disp - t_call).as_nanos() as u64, Ordering::Relaxed);
            PROF[8].fetch_add(split_ns, Ordering::Relaxed);
        }
        // One dispatch. Nothing in it panics, and nothing in it waits for
        // more than the blocks other cores already hold: a core is in the
        // down rows only once every gate/up block is taken, and the cores
        // that took them publish their experts without waiting for anyone.
        let work = |w: usize, _n: usize| {
            let own = usize::from(sides == 2 && pool.is_some_and(|p| p.far_worker(w)));
            let t_in = if prof { t_disp.elapsed().as_nanos() as u64 } else { 0 };
            let (mut ns_split, mut ns_wait) = (0u64, 0u64);
            SCRATCH.with(|scr| {
                let scr = &mut *scr.borrow_mut();
                scr.fit(gpr_max, max_group);
                // gate and up, a block of rows of an expert at a time: this core's side, then the other
                for side in (0..sides).map(|k| own ^ k) {
                    loop {
                        let u0 = cur_gu[side].fetch_add(claim_gu, Ordering::Relaxed);
                        if u0 >= n_gu {
                            break;
                        }
                        for unit in u0..(u0 + claim_gu).min(n_gu) {
                            let (ai, blk) = (unit / sg, side * sg + unit % sg);
                            let e = act[ai] as usize;
                            let (p0, p1) = (offs[e], offs[e + 1]);
                            // SAFETY: the views were checked against the shape; a unit is one
                            // core's, so its rows are written once.
                            unsafe { gate_up_block(&views[e * 3], &views[e * 3 + 1], blk * BLK, hid, inter, ta, &pair_tok[p0..p1], av, p0, scr, pf) };
                            if left[ai].fetch_sub(1, Ordering::AcqRel) == 1 {
                                // the expert's last block: its gate rows are whole, this
                                // core splits them and publishes the expert
                                let t0 = prof.then(Instant::now);
                                for p in p0..p1 {
                                    // SAFETY: every block of the expert is written (the countdown
                                    // orders them before this read), and only this core reads them.
                                    unsafe { split_row(av.at(p * inter), pa, p) };
                                }
                                queue[q_len.fetch_add(1, Ordering::AcqRel)].store(ai, Ordering::Release);
                                if let Some(t0) = t0 {
                                    ns_split += t0.elapsed().as_nanos() as u64;
                                }
                            }
                        }
                    }
                }
                let t_gu = if prof { t_disp.elapsed().as_nanos() as u64 } else { 0 };
                // down, the experts in the order they were published
                let mut held = (usize::MAX, 0usize);
                for side in (0..sides).map(|k| own ^ k) {
                    loop {
                        let u0 = cur_dn[side].fetch_add(claim_dn, Ordering::Relaxed);
                        if u0 >= n_dn {
                            break;
                        }
                        for unit in u0..(u0 + claim_dn).min(n_dn) {
                            let (pos, blk) = (unit / sd, side * sd + unit % sd);
                            if held.0 != pos {
                                let mut ai = queue[pos].load(Ordering::Acquire);
                                if ai == usize::MAX {
                                    let t0 = prof.then(Instant::now);
                                    let mut spins = 0u32;
                                    while ai == usize::MAX {
                                        if spins < 1 << 16 {
                                            spins += 1;
                                            std::hint::spin_loop();
                                        } else {
                                            std::thread::yield_now();
                                        }
                                        ai = queue[pos].load(Ordering::Acquire);
                                    }
                                    if let Some(t0) = t0 {
                                        ns_wait += t0.elapsed().as_nanos() as u64;
                                    }
                                }
                                held = (pos, ai);
                            }
                            let e = act[held.1] as usize;
                            let (p0, p1) = (offs[e], offs[e + 1]);
                            // SAFETY: as the gate/up blocks, on the output rows; the expert's
                            // gate rows were split before it was published.
                            unsafe { down_block(&views[e * 3 + 2], blk * BLK, hid, inter, pa, &pair_ix[p0..p1], ov, p0, scr, pf) };
                        }
                    }
                }
                if prof {
                    let t_end = t_disp.elapsed().as_nanos() as u64;
                    for (slot, v) in [
                        (1, 1),
                        (10, t_in),
                        (11, (t_gu - t_in).saturating_sub(ns_split)),
                        (12, ns_split),
                        (13, (t_end - t_gu).saturating_sub(ns_wait)),
                        (14, ns_wait),
                    ] {
                        PROF[slot].fetch_add(v, Ordering::Relaxed);
                    }
                }
            });
        };
        match pool {
            Some(p) => p.run(&work),
            None => work(0, 1),
        }
        if prof {
            let outl = |on: &[u32]| on.iter().map(|&n| n as u64).sum::<u64>();
            for (slot, v) in [
                (0, 1),
                (2, n_act as u64),
                (3, n_pairs as u64),
                (4, st.toks.len() as u64),
                (5, outl(&st.tok.on[..st.toks.len()])),
                (6, outl(&st.pair.on[..n_pairs])),
                (9, t_disp.elapsed().as_nanos() as u64),
            ] {
                PROF[slot].fetch_add(v, Ordering::Relaxed);
            }
        }
        if let Some(g) = gs_out {
            g.clear();
            g.extend_from_slice(&st.avals[..n_pairs * inter]);
        }
    });
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
