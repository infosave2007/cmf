//! NVIDIA Nemotron Labs Dory: a looped normalized GPT.
//!
//! The layer stack is three blocks, input, recurrent and output, and the
//! recurrent block runs `loops` times per token with the input block's output
//! mixed back in before each pass. A physical layer is one sub-layer, attention
//! or a SwiGLU MLP, and everything lives on the unit sphere (nGPT):
//!
//! ```text
//! attention  q,k,v,g = W·x; rotate q,k; q = s_q²·q̂, k = k̂ (per head)
//!            a = softmax(q·k/√hd) over the window, y = Σ a·v (16 q heads over 4 kv heads)
//!            y = (y ⊙ σ(g ⊙ s_g²))^ , out = O·y
//! mlp        g,u = W·x ⊙ s·√hidden; y = (silu(g) ⊙ u)^ ; out = D·y
//! residual   x = (x̂ + |α|·(ôut − x̂))^
//! loop       state = input; h = (state + input)^; h = R(h); state = ĥ
//! logits     H·h ⊙ scale / σ₀
//! ```
//!
//! `per_loop` keeps a KV cache per loop; `last_loop` keeps one per recurrent
//! attention layer, each loop overwriting this chunk's rows. Earlier rows then
//! hold their last loop's keys, so the loop count can change from token to token
//! and the cache has no holes.

use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfModel, DoryConfig, ModelArch};
use std::sync::Arc;

/// `justnorm`'s epsilon in the reference.
pub const EPS: f32 = 1e-6;
pub const ARCH: &str = "dory";
/// The most loops a request may ask for.
pub const MAX_LOOPS: usize = 16;
/// Rows a chat turn may go back on the device: a sliding ring holds the window and
/// this much, so a turn whose template rewrites the last answer keeps the rest.
pub const REWIND: usize = 448;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvMode {
    PerLoop,
    LastLoop,
}

impl KvMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "per_loop" => Some(Self::PerLoop),
            "last_loop" => Some(Self::LastLoop),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DoryCfg {
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub nh: usize,
    pub nk: usize,
    pub hd: usize,
    pub window: usize,
    pub n_in: usize,
    pub n_rec: usize,
    pub n_out: usize,
    pub loops: usize,
    pub kv: KvMode,
    pub attn_scale: f32,
    /// Full layers read at most this many keys and keep no more: a constant cost per
    /// token and bounded memory at any context length (`CMF_DORY_FULL_WINDOW`).
    pub full_window: Option<usize>,
}

pub struct AttnWeights {
    pub q: QTensor,
    pub k: QTensor,
    pub v: QTensor,
    pub g: QTensor,
    pub o: QTensor,
    /// `(sqk·√hd/σ₀)²`
    pub sq: Vec<f32>,
    /// `(s_gate·√g₀/σ₀)²`
    pub sg: Vec<f32>,
    pub alpha: Vec<f32>,
    pub sliding: bool,
    /// Position among the attention layers: the KV slot table's row.
    pub ordinal: usize,
}

pub struct MlpWeights {
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    /// `suv·√hidden`
    pub sg: Vec<f32>,
    pub su: Vec<f32>,
    pub alpha: Vec<f32>,
}

pub enum Layer {
    Attn(Box<AttnWeights>),
    Mlp(Box<MlpWeights>),
}

pub struct Dory {
    /// The file, for a device that binds its mapping as it is.
    pub file: Option<Arc<CmfModel>>,
    pub cfg: DoryCfg,
    pub embed: QTensor,
    pub head: QTensor,
    /// `logit_scale/σ₀`
    pub logit_scale: Vec<f32>,
    pub layers: Vec<Layer>,
    pub inv_full: Vec<f32>,
    pub inv_sliding: Vec<f32>,
    pub attn_layers: usize,
    pub o1: Option<DoryO1>,
    /// How a token leaves the loop early, on the last-loop cache.
    pub gate: Option<Gate>,
    /// Passes over a prompt chunk whose logits nobody reads (`CMF_DORY_PROMPT_LOOPS`):
    /// its cache rows are all that stays of it. 0: as many as any other chunk.
    pub prompt_loops: usize,
    /// Rows of a sealed prompt the far field would lose (`CMF_DORY_EXACT=FILE:rows`).
    pub exact: Option<ExactRows>,
}

/// Which rows of a prompt stay exact past the seal: passwords, codes and the like, found
/// by what the layers do with them and not by what they look like. Per KV head of each
/// full layer, the part of a key outside the span of the far field's landmarks and the
/// value's norm, as ranks within the prompt; a linear score over them, fitted on planted
/// facts (`scripts/dory/router/illum_budget.py`).
#[derive(Clone)]
pub struct ExactRows {
    pub weights: Arc<Vec<f32>>,
    pub budget: usize,
}

/// Keys a sealed token reads exactly, on any backend.
pub const MAX_NEAR: usize = 8192;

/// A sealed prompt's last rows read the chosen ones: a question carries what it found in
/// the text on to the answer, so the choice comes before it.
pub const EXACT_TAIL: usize = 512;

/// Neighbours kept on each side of a chosen row: a fact is several tokens.
const EXACT_NEIGHBOURS: usize = 2;

fn rank_share(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_unstable_by(|&a, &b| v[a as usize].total_cmp(&v[b as usize]));
    let mut out = vec![0f32; n];
    let d = (n.max(2) - 1) as f32;
    for (i, &r) in idx.iter().enumerate() {
        out[r as usize] = i as f32 / d;
    }
    out
}

/// Per row, how much of a head's key lies outside the span of `m` segment means.
fn key_residuals(ks: &[f32], rows: usize, nk: usize, kh: usize, hd: usize, m: usize) -> Vec<f32> {
    const RIDGE: f64 = 1e-4;
    let at = |r: usize| &ks[(r * nk + kh) * hd..(r * nk + kh + 1) * hd];
    let mut l = vec![0f32; m * hd];
    for a in 0..m {
        let (lo, hi) = (a * rows / m, ((a + 1) * rows / m).max(a * rows / m + 1).min(rows));
        let row = &mut l[a * hd..(a + 1) * hd];
        for r in lo..hi {
            crate::attention::axpy_f32(row, at(r), 1.0);
        }
        let f = 1.0 / (hi - lo) as f32;
        row.iter_mut().for_each(|x| *x *= f);
    }
    // The inverse of the landmarks' Gram matrix, by Cholesky.
    let mut g = vec![0f64; m * m];
    for a in 0..m {
        for b in 0..=a {
            let v = crate::attention::dot_f32(&l[a * hd..(a + 1) * hd], &l[b * hd..(b + 1) * hd]) as f64;
            g[a * m + b] = v + if a == b { RIDGE } else { 0.0 };
        }
    }
    for a in 0..m {
        for b in 0..=a {
            let mut v = g[a * m + b];
            for c in 0..b {
                v -= g[a * m + c] * g[b * m + c];
            }
            g[a * m + b] = if a == b { v.max(1e-12).sqrt() } else { v / g[b * m + b] };
        }
    }
    let mut inv = vec![0f32; m * m];
    let mut col = vec![0f64; m];
    for e in 0..m {
        for a in 0..m {
            let mut v = if a == e { 1.0 } else { 0.0 };
            for c in 0..a {
                v -= g[a * m + c] * col[c];
            }
            col[a] = v / g[a * m + a];
        }
        for a in (0..m).rev() {
            let mut v = col[a];
            for c in a + 1..m {
                v -= g[c * m + a] * col[c];
            }
            col[a] = v / g[a * m + a];
        }
        for a in 0..m {
            inv[a * m + e] = col[a] as f32;
        }
    }
    let mut y = vec![0f32; m];
    let mut c = vec![0f32; m];
    (0..rows)
        .map(|r| {
            let k = at(r);
            for a in 0..m {
                y[a] = crate::attention::dot_f32(&l[a * hd..(a + 1) * hd], k);
            }
            for a in 0..m {
                c[a] = crate::attention::dot_f32(&inv[a * m..(a + 1) * m], &y);
            }
            crate::attention::dot_f32(k, k) - crate::attention::dot_f32(&c, &y) - RIDGE as f32 * crate::attention::dot_f32(&c, &c)
        })
        .collect()
}

/// The rows to keep exact, ascending. `layers` are the full layers' keys and values,
/// `[row][nk][hd]`; `lo..hi` the rows that can go, which leaves out the sinks and the
/// window. A row goes with its neighbours until the budget is spent.
pub fn pick_exact_rows(x: &ExactRows, layers: &[(&[f32], &[f32])], rows: usize, nk: usize, hd: usize, m: usize, lo: usize, hi: usize) -> Vec<usize> {
    let hi = hi.min(rows);
    if hi <= lo || x.weights.len() != layers.len() * 2 * nk || rows < 2 {
        return Vec::new();
    }
    let m = m.min(rows).max(1);
    let tasks: Vec<(usize, usize)> = (0..layers.len()).flat_map(|l| (0..nk).map(move |h| (l, h))).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16).min(tasks.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let total = std::sync::Mutex::new(vec![0f32; rows]);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(l, kh)) = tasks.get(t) else { break };
                let (ks, vs) = layers[l];
                let res = rank_share(&key_residuals(ks, rows, nk, kh, hd, m));
                let vn: Vec<f32> = (0..rows)
                    .map(|r| {
                        let v = &vs[(r * nk + kh) * hd..(r * nk + kh + 1) * hd];
                        crate::attention::dot_f32(v, v)
                    })
                    .collect();
                let vn = rank_share(&vn);
                let (wr, wv) = (x.weights[(l * 2) * nk + kh], x.weights[(l * 2 + 1) * nk + kh]);
                let mut tot = total.lock().unwrap();
                for r in 0..rows {
                    tot[r] += wr * res[r] + wv * vn[r];
                }
            });
        }
    });
    let score = total.into_inner().unwrap();
    if let Ok(f) = std::env::var("CMF_DORY_EXACT_LOG") {
        let b: Vec<u8> = score.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(f, b);
    }
    let mut order: Vec<u32> = (lo as u32..hi as u32).collect();
    order.sort_unstable_by(|&a, &b| score[b as usize].total_cmp(&score[a as usize]));
    let mut keep = vec![false; rows];
    let mut count = 0usize;
    'pick: for r in order {
        let r = r as usize;
        for j in r.saturating_sub(EXACT_NEIGHBOURS).max(lo)..(r + EXACT_NEIGHBOURS + 1).min(hi) {
            if count >= x.budget {
                break 'pick;
            }
            if !keep[j] {
                keep[j] = true;
                count += 1;
            }
        }
    }
    (0..rows).filter(|&r| keep[r]).collect()
}

/// The order rows go into a seal: the sinks, the rows kept exact, then the rest as they
/// came, so the window stays the last rows and the kept ones leave the far field.
pub fn seal_order(rows: usize, sink: usize, keep: &[usize]) -> Vec<u32> {
    let mut chosen = vec![false; rows];
    keep.iter().for_each(|&r| chosen[r] = true);
    (0..sink.min(rows))
        .chain(keep.iter().copied())
        .chain((sink.min(rows)..rows).filter(|&r| !chosen[r]))
        .map(|r| r as u32)
        .collect()
}

/// O(1) far field for the full-attention layers (`nystrom`): `CMF_DORY_O1=m,w,sink,seal`.
/// A slot runs exact up to position `seal`, then an exact window of `w` keys plus a
/// skeleton of `m` landmarks. Sliding layers are bounded already.
#[derive(Clone, Copy, Debug)]
pub struct DoryO1 {
    pub m: usize,
    pub w: usize,
    pub sink: usize,
    pub seal: usize,
}

impl DoryO1 {
    pub fn from_env() -> Option<Self> {
        let v = std::env::var("CMF_DORY_O1").ok()?;
        let n: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        // Below the window plus a margin the state would be exact only.
        (n.len() == 4).then(|| Self { m: n[0], w: n[1], sink: n[2], seal: n[3].max(n[1] + n[2] + 16) })
    }
}

/// One KV slot: rows `first..first + len` of keys and values, `[row][nk][hd]`.
#[derive(Default)]
pub struct KvSlot {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub first: usize,
    /// Sealed, and its rows still kept for the choice of exact ones.
    pub arch: bool,
    /// Every row's queries `[row][nh][hd]`, until the O(1) seal.
    pub qt: Vec<f32>,
    /// One state per KV head after the seal.
    pub far: Vec<crate::nystrom::NystromState>,
}

impl KvSlot {
    fn rows(&self, row: usize) -> usize {
        if row == 0 { 0 } else { self.k.len() / row }
    }
}

/// Host KV state: `slots[attention ordinal][loop]`.
#[derive(Default)]
pub struct DoryState {
    pub slots: Vec<Vec<KvSlot>>,
    pub pos: usize,
    scratch: Scratch,
    /// The logits of a token whose first read-out ended it.
    exit_logits: Option<Vec<f32>>,
    exits: ExitRate,
    pub gate_tokens: u64,
    pub gate_one: u64,
}

impl DoryState {
    pub fn clear(&mut self) {
        for row in &mut self.slots {
            for s in row {
                s.k.clear();
                s.v.clear();
                s.first = 0;
                s.qt.clear();
                s.far.clear();
            }
        }
        self.pos = 0;
    }

    /// Forget the rows from position `n` on.
    pub fn rollback(&mut self, n: usize, kv_row: usize) {
        for row in &mut self.slots {
            for s in row {
                let end = s.first + s.rows(kv_row);
                if end > n {
                    let keep = n.saturating_sub(s.first) * kv_row;
                    s.k.truncate(keep);
                    s.v.truncate(keep);
                }
            }
        }
        self.pos = self.pos.min(n);
    }
}

fn load_vec(model: &CmfModel, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let v = crate::loader::load_f32(model, name, &crate::loader::Overlay::None)?;
    if v.len() != n {
        return Err(format!("{name}: {} values, wanted {n}", v.len()));
    }
    Ok(v)
}

impl Dory {
    pub fn from_model(model: &Arc<CmfModel>) -> Result<Self, String> {
        let arch: &ModelArch = model.arch();
        let rec: &DoryConfig = arch.dory.as_ref().ok_or("dory: no arch record")?;
        let kv = KvMode::parse(&rec.recurrent_kv_cache_mode)
            .ok_or_else(|| format!("dory: KV mode {}", rec.recurrent_kv_cache_mode))?;
        let (hidden, ffn, nh, nk, hd) =
            (arch.hidden_size, arch.intermediate_size, arch.num_attention_heads, arch.num_kv_heads, arch.head_dim);
        let std = rec.initializer_range;
        let alpha_f = (rec.ngpt_alpha_init / std) as f32;
        let sqk_f = ((hd as f64).sqrt() / std) as f32;
        let gate_f = (rec.gate_logit_scale_init.sqrt() / std) as f32;
        let suv_f = (hidden as f64).sqrt() as f32;
        let q_size = nh * hd;
        let t = |name: String| QTensor::from_model(model, &name);
        let mut layers = Vec::with_capacity(rec.layer_pattern.len());
        let mut ordinal = 0usize;
        for (li, kind) in rec.layer_pattern.chars().enumerate() {
            let p = format!("model.layers.{li}");
            let m = format!("{p}.mixer");
            if kind == '-' {
                let scale = |name: &str| -> Result<Vec<f32>, String> {
                    let mut v = load_vec(model, &format!("{m}.{name}"), ffn)?;
                    v.iter_mut().for_each(|x| *x *= suv_f);
                    Ok(v)
                };
                let mut alpha = load_vec(model, &format!("{p}.mlp_alpha"), hidden)?;
                alpha.iter_mut().for_each(|x| *x = (*x * alpha_f).abs());
                layers.push(Layer::Mlp(Box::new(MlpWeights {
                    gate: t(format!("{m}.gate_proj.weight"))?,
                    up: t(format!("{m}.up_proj.weight"))?,
                    down: t(format!("{m}.down_proj.weight"))?,
                    sg: scale("suv_gate")?,
                    su: scale("suv_up")?,
                    alpha,
                })));
            } else {
                let sq_vec = |name: &str, f: f32| -> Result<Vec<f32>, String> {
                    let mut v = load_vec(model, &format!("{m}.{name}"), q_size)?;
                    v.iter_mut().for_each(|x| {
                        let s = *x * f;
                        *x = s * s;
                    });
                    Ok(v)
                };
                let mut alpha = load_vec(model, &format!("{p}.attn_alpha"), hidden)?;
                alpha.iter_mut().for_each(|x| *x = (*x * alpha_f).abs());
                layers.push(Layer::Attn(Box::new(AttnWeights {
                    q: t(format!("{m}.q_proj.weight"))?,
                    k: t(format!("{m}.k_proj.weight"))?,
                    v: t(format!("{m}.v_proj.weight"))?,
                    g: t(format!("{m}.g_proj.weight"))?,
                    o: t(format!("{m}.o_proj.weight"))?,
                    sq: sq_vec("sqk", sqk_f)?,
                    sg: sq_vec("s_gate", gate_f)?,
                    alpha,
                    sliding: kind == 'S',
                    ordinal,
                })));
                ordinal += 1;
            }
        }
        let mut logit_scale = load_vec(model, "lm_head.logit_scale", arch.vocab_size)?;
        let inv = 1.0 / std as f32;
        logit_scale.iter_mut().for_each(|x| *x *= inv);
        let mut cfg = DoryCfg {
            hidden,
            ffn,
            vocab: arch.vocab_size,
            nh,
            nk,
            hd,
            window: rec.sliding_window,
            n_in: rec.n_input_layers,
            n_rec: rec.n_recurrent_layers,
            n_out: rec.n_output_layers,
            loops: rec.n_recurrent_loops,
            kv,
            attn_scale: 1.0 / (hd as f32).sqrt(),
            full_window: std::env::var("CMF_DORY_FULL_WINDOW").ok().and_then(|v| v.parse().ok()),
        };
        if let Some(n) = std::env::var("CMF_DORY_LOOPS").ok().and_then(|v| v.parse::<usize>().ok()) {
            if n > MAX_LOOPS {
                return Err(format!("CMF_DORY_LOOPS={n}: want at most {MAX_LOOPS}"));
            }
            cfg.loops = n;
        }
        if let Ok(v) = std::env::var("CMF_DORY_KV") {
            cfg.kv = KvMode::parse(&v).ok_or_else(|| format!("CMF_DORY_KV={v}: want per_loop or last_loop"))?;
        }
        let auto = std::env::var("CMF_DORY_GATE").as_deref() == Ok("auto");
        let mut gate = gate_from_env();
        let mut o1 = DoryO1::from_env();
        let mut prompt_loops: usize = std::env::var("CMF_DORY_PROMPT_LOOPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut exact = exact_from_env();
        // A turbo file runs so unless a flag says otherwise; a fixed loop count is one.
        if let (Some(t), None) = (&rec.turbo, std::env::var_os("CMF_DORY_LOOPS")) {
            if std::env::var_os("CMF_DORY_KV").is_none() {
                cfg.kv = KvMode::LastLoop;
            }
            if gate.is_none() && cfg.kv == KvMode::LastLoop {
                gate = Some(Gate::Recon(Arc::new(Topologies::from_model(model)?), t.exit_bias));
            }
            if o1.is_none() && std::env::var("CMF_DORY_O1").as_deref() != Ok("off") {
                let [m, w, sink, seal] = t.o1;
                o1 = Some(DoryO1 { m, w, sink, seal: seal.max(w + sink + 16) });
            }
            if std::env::var_os("CMF_DORY_PROMPT_LOOPS").is_none() {
                prompt_loops = 1;
            }
            if t.exact_rows > 0 && std::env::var_os("CMF_DORY_EXACT").is_none() && o1.is_some() {
                let w = crate::loader::load_f32(model, "dory.exact.weights", &crate::loader::Overlay::None)?;
                exact = Some(ExactRows { weights: Arc::new(w), budget: t.exact_rows });
            }
        }
        if let (Some(x), Some(o)) = (&exact, &o1) {
            if o.w + o.sink + x.budget > MAX_NEAR {
                return Err(format!("dory: window {} + sinks {} + exact rows {} go over the near field's {MAX_NEAR}", o.w, o.sink, x.budget));
            }
        }
        // `--loops auto` without a router in the file: the top probability decides.
        if auto && gate.is_none() {
            gate = Some(Gate::Conf(0.7));
        }
        Ok(Self {
            file: Some(model.clone()),
            cfg,
            embed: t("model.embed_tokens.weight".into())?,
            head: t("lm_head.weight".into())?,
            logit_scale,
            layers,
            inv_full: crate::attention::rope_inv_freq(hd, rec.rope_theta_full as f32),
            inv_sliding: crate::attention::rope_inv_freq(hd, rec.rope_theta_sliding as f32),
            attn_layers: ordinal,
            o1,
            gate,
            prompt_loops,
            exact,
        })
    }

    /// The passes over one chunk of a forward.
    pub fn chunk_loops(&self, loops: usize, want: Want) -> usize {
        if want == Want::None && self.prompt_loops > 0 && self.cfg.kv == KvMode::LastLoop {
            loops.min(self.prompt_loops)
        } else {
            loops
        }
    }

    /// Which loop's cache a layer uses.
    #[inline]
    fn slot_loop(&self, li: usize, loop_i: usize) -> usize {
        let c = &self.cfg;
        if c.kv == KvMode::PerLoop && li >= c.n_in && li < c.n_in + c.n_rec { loop_i } else { 0 }
    }
}

/// `x / sqrt(Σx² + ε)`
#[inline]
pub fn justnorm(x: &mut [f32]) {
    let r = 1.0 / (crate::attention::dot_f32(x, x) + EPS).sqrt();
    for v in x.iter_mut() {
        *v *= r;
    }
}

/// The nGPT residual, written into `x`.
fn slerp(x: &mut [f32], out: &mut [f32], alpha: &[f32]) {
    justnorm(x);
    justnorm(out);
    for ((a, b), al) in x.iter_mut().zip(out.iter()).zip(alpha) {
        *a += *al * (*b - *a);
    }
    justnorm(x);
}

/// `exp` on eight lanes, written so a compiler vectorizes it (Cephes `expf`, within
/// 2e-7 on the range an activation or a softmax reaches).
#[inline(always)]
fn exp8(x: &[f32; 8]) -> [f32; 8] {
    let mut r = [0f32; 8];
    for i in 0..8 {
        let v = x[i].clamp(-87.0, 88.0);
        let n = (v * std::f32::consts::LOG2_E + 0.5).floor();
        let g = v - n * 0.693_359_4 + n * 2.121_944_4e-4;
        let p = ((((((1.987_569_1e-4 * g + 1.398_2e-3) * g + 8.333_452e-3) * g + 4.166_579_6e-2) * g + 1.666_666_5e-1) * g + 0.5) * g * g) + g + 1.0;
        r[i] = p * f32::from_bits(((n as i32 + 127) << 23) as u32);
    }
    r
}

macro_rules! lanes8 {
    ($body:ident, $avx:ident, ($($a:ident : $t:ty),*)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx($($a: $t),*) {
            $body($($a),*)
        }
    };
}

/// `g[i] = silu(g[i]·a[i])·u[i]·b[i]`
#[inline(always)]
fn silu_body(g: &mut [f32], u: &[f32], a: &[f32], b: &[f32]) {
    let n = g.len() & !7;
    for i in (0..n).step_by(8) {
        let mut x = [0f32; 8];
        let mut m = [0f32; 8];
        for j in 0..8 {
            x[j] = g[i + j] * a[i + j];
            m[j] = -x[j];
        }
        let e = exp8(&m);
        for j in 0..8 {
            g[i + j] = x[j] / (1.0 + e[j]) * (u[i + j] * b[i + j]);
        }
    }
    for i in n..g.len() {
        let x = g[i] * a[i];
        g[i] = x / (1.0 + (-x).exp()) * (u[i] * b[i]);
    }
}
lanes8!(silu_body, silu_avx2, (g: &mut [f32], u: &[f32], a: &[f32], b: &[f32]));

fn silu_rows(g: &mut [f32], u: &[f32], a: &[f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::qtensor::avx2_enabled() {
        return unsafe { silu_avx2(g, u, a, b) };
    }
    silu_body(g, u, a, b)
}

/// `y[i] = y[i]·σ(g[i]·s[i])`
#[inline(always)]
fn gate_body(y: &mut [f32], g: &[f32], s: &[f32]) {
    let n = y.len() & !7;
    for i in (0..n).step_by(8) {
        let mut m = [0f32; 8];
        for j in 0..8 {
            m[j] = -(g[i + j] * s[i + j]);
        }
        let e = exp8(&m);
        for j in 0..8 {
            y[i + j] /= 1.0 + e[j];
        }
    }
    for i in n..y.len() {
        y[i] /= 1.0 + (-(g[i] * s[i])).exp();
    }
}
lanes8!(gate_body, gate_avx2, (y: &mut [f32], g: &[f32], s: &[f32]));

fn gate_rows(y: &mut [f32], g: &[f32], s: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::qtensor::avx2_enabled() {
        return unsafe { gate_avx2(y, g, s) };
    }
    gate_body(y, g, s)
}

/// Softmax of a row of scores, in place.
#[inline(always)]
fn soft_body(row: &mut [f32]) {
    let n = row.len();
    let mut top = f32::NEG_INFINITY;
    for v in row.iter() {
        top = top.max(*v);
    }
    let mut den = 0f32;
    let blocks = n & !7;
    for i in (0..blocks).step_by(8) {
        let mut m = [0f32; 8];
        for j in 0..8 {
            m[j] = row[i + j] - top;
        }
        let e = exp8(&m);
        let mut acc = 0f32;
        for j in 0..8 {
            row[i + j] = e[j];
            acc += e[j];
        }
        den += acc;
    }
    for v in row[blocks..].iter_mut() {
        *v = (*v - top).exp();
        den += *v;
    }
    let inv = 1.0 / den;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
lanes8!(soft_body, soft_avx2, (row: &mut [f32]));

fn soft_row(row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::qtensor::avx2_enabled() {
        return unsafe { soft_avx2(row) };
    }
    soft_body(row)
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline]
    fn at(&self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

fn for_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize) + Sync)) {
    match pool {
        Some(p) if n > 1 => p.run_rows(n, &|a, b| (a..b).for_each(f)),
        _ => (0..n).for_each(f),
    }
}

thread_local! {
    static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The chunk's scratch, kept between forwards: a token touches 112 layers.
#[derive(Default)]
pub struct Scratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    out: Vec<f32>,
    /// `[kind][row][cos half | sin half]`: sliding, full.
    rope: [Vec<f32>; 2],
}

fn fit(v: &mut Vec<f32>, n: usize) {
    v.resize(n, 0.0);
}

/// One attention sub-layer over the rows `h[t][hidden]`, positions from `pos0`.
#[allow(clippy::too_many_arguments)]
fn attn_layer(
    m: &Dory,
    w: &AttnWeights,
    slot: &mut KvSlot,
    h: &[f32],
    t: usize,
    pos0: usize,
    overwrite: bool,
    pool: Option<&Pool>,
    sc: &mut Scratch,
) {
    let c = &m.cfg;
    let (nh, nk, hd, hidden) = (c.nh, c.nk, c.hd, c.hidden);
    let q_size = nh * hd;
    let kv_row = nk * hd;
    fit(&mut sc.q, t * q_size);
    fit(&mut sc.k, t * kv_row);
    fit(&mut sc.v, t * kv_row);
    fit(&mut sc.g, t * q_size);
    fit(&mut sc.out, t * hidden);
    if t == 1 && !crate::gptq_capture::capturing() {
        QTensor::matvec_many([&w.q, &w.k, &w.v, &w.g], h, [&mut sc.q[..], &mut sc.k[..], &mut sc.v[..], &mut sc.g[..]], pool);
    } else {
        w.q.matmat(h, t, &mut sc.q, pool);
        w.k.matmat(h, t, &mut sc.k, pool);
        w.v.matmat(h, t, &mut sc.v, pool);
        w.g.matmat(h, t, &mut sc.g, pool);
    }
    let half = hd / 2;
    let rope = &sc.rope[usize::from(!w.sliding)];
    for r in 0..t {
        let table = &rope[r * hd..(r + 1) * hd];
        let (cs, sn) = table.split_at(half);
        let rotate = |row: &mut [f32]| {
            let (a, b) = row.split_at_mut(half);
            for j in 0..half {
                let (x, y) = (a[j], b[j]);
                a[j] = x * cs[j] - y * sn[j];
                b[j] = y * cs[j] + x * sn[j];
            }
            justnorm(row);
        };
        for (hi, row) in sc.q[r * q_size..(r + 1) * q_size].chunks_exact_mut(hd).enumerate() {
            rotate(row);
            for (x, s) in row.iter_mut().zip(&w.sq[hi * hd..(hi + 1) * hd]) {
                *x *= *s;
            }
        }
        for row in sc.k[r * kv_row..(r + 1) * kv_row].chunks_exact_mut(hd) {
            rotate(row);
        }
    }
    let rep = nh / nk;
    // A sealed slot: the exact window and the far skeleton, row by row.
    if !slot.far.is_empty() {
        if slot.arch {
            if overwrite {
                let keep = slot.k.len() - t * kv_row;
                slot.k.truncate(keep);
                slot.v.truncate(keep);
                let keep = slot.qt.len() - t * q_size;
                slot.qt.truncate(keep);
            }
            slot.k.extend_from_slice(&sc.k);
            slot.v.extend_from_slice(&sc.v);
            slot.qt.extend_from_slice(&sc.q);
        }
        let mut y = vec![0f32; rep * hd];
        for r in 0..t {
            for kh in 0..nk {
                let at_q = r * q_size + kh * rep * hd;
                let at_k = (r * nk + kh) * hd;
                let (q, k, v) = (&sc.q[at_q..at_q + rep * hd], &sc.k[at_k..at_k + hd], &sc.v[at_k..at_k + hd]);
                if overwrite {
                    slot.far[kh].restep_group(q, k, v, &mut y);
                } else {
                    slot.far[kh].step_group(q, k, v, &mut y);
                }
                gate_rows(&mut y, &sc.g[at_q..at_q + rep * hd], &w.sg[kh * rep * hd..(kh + 1) * rep * hd]);
                sc.g[at_q..at_q + rep * hd].copy_from_slice(&y);
            }
        }
    } else {
        if !w.sliding && m.o1.is_some() {
            if overwrite {
                let keep = slot.qt.len() - t * q_size;
                slot.qt.truncate(keep);
            }
            slot.qt.extend_from_slice(&sc.q);
        }
        // The chunk's rows join the cache; each one attends only up to itself.
        if overwrite {
            let keep = slot.k.len() - t * kv_row;
            slot.k.truncate(keep);
            slot.v.truncate(keep);
        }
        slot.k.extend_from_slice(&sc.k);
        slot.v.extend_from_slice(&sc.v);
        let (first, keys, vals) = (slot.first, &slot.k[..], &slot.v[..]);
        let scale = c.attn_scale;
        let res = SendPtr(sc.g.as_mut_ptr());
        let window = if w.sliding { c.window } else { c.full_window.unwrap_or(usize::MAX) };
        let q_ref = &sc.q;
        let sg = &w.sg;
        for_rows(pool, t * nk, &|task| {
            let (r, kh) = (task / nk, task % nk);
            let p = pos0 + r;
            let lo = p.saturating_add(1).saturating_sub(window).max(first);
            let n = p + 1 - lo;
            let at_q = r * q_size + kh * rep * hd;
            SCORES.with(|buf| {
                let mut buf = buf.borrow_mut();
                buf.resize(rep * (n + hd), 0.0);
                let (scores, ys) = buf.split_at_mut(rep * n);
                // A key once, against each query head of its group.
                for e in 0..n {
                    let row = (lo + e - first) * kv_row + kh * hd;
                    let key = &keys[row..row + hd];
                    for h in 0..rep {
                        let qr = &q_ref[at_q + h * hd..at_q + (h + 1) * hd];
                        scores[h * n + e] = crate::attention::dot_f32(qr, key) * scale;
                    }
                }
                ys.iter_mut().for_each(|v| *v = 0.0);
                for h in 0..rep {
                    soft_row(&mut scores[h * n..(h + 1) * n]);
                }
                for e in 0..n {
                    let row = (lo + e - first) * kv_row + kh * hd;
                    let val = &vals[row..row + hd];
                    for h in 0..rep {
                        crate::attention::axpy_f32(&mut ys[h * hd..(h + 1) * hd], val, scores[h * n + e]);
                    }
                }
                // `g` becomes the gated attention output.
                let g = unsafe { std::slice::from_raw_parts_mut(res.at(at_q), rep * hd) };
                gate_rows(ys, g, &sg[kh * rep * hd..(kh + 1) * rep * hd]);
                g.copy_from_slice(ys);
            });
        });
    }
    for row in sc.g.chunks_exact_mut(q_size) {
        justnorm(row);
    }
    if t == 1 && !crate::gptq_capture::capturing() {
        w.o.matvec(&sc.g, &mut sc.out, pool);
    } else {
        w.o.matmat(&sc.g, t, &mut sc.out, pool);
    }
}

fn mlp_layer(w: &MlpWeights, h: &[f32], t: usize, ffn: usize, hidden: usize, pool: Option<&Pool>, sc: &mut Scratch) {
    fit(&mut sc.g, t * ffn);
    fit(&mut sc.u, t * ffn);
    fit(&mut sc.out, t * hidden);
    let single = t == 1 && !crate::gptq_capture::capturing();
    if single {
        QTensor::matvec_many([&w.gate, &w.up], h, [&mut sc.g[..], &mut sc.u[..]], pool);
    } else {
        w.gate.matmat(h, t, &mut sc.g, pool);
        w.up.matmat(h, t, &mut sc.u, pool);
    }
    for (gr, ur) in sc.g.chunks_exact_mut(ffn).zip(sc.u.chunks_exact(ffn)) {
        silu_rows(gr, ur, &w.sg, &w.su);
        justnorm(gr);
    }
    if single {
        w.down.matvec(&sc.g, &mut sc.out, pool);
    } else {
        w.down.matmat(&sc.g, t, &mut sc.out, pool);
    }
}

/// The rows' final hidden states, through `loops` passes of the recurrent block.
#[allow(clippy::too_many_arguments)]
fn stack(
    m: &Dory,
    st: &mut DoryState,
    h: &mut Vec<f32>,
    t: usize,
    pos0: usize,
    loops: usize,
    pool: Option<&Pool>,
    capture: Option<&mut Capture>,
) {
    let c = &m.cfg;
    let hidden = c.hidden;
    let mut sc = std::mem::take(&mut st.scratch);
    let half = c.hd / 2;
    for (ti, inv) in [&m.inv_sliding, &m.inv_full].iter().enumerate() {
        fit(&mut sc.rope[ti], t * c.hd);
        for (r, row) in sc.rope[ti].chunks_exact_mut(c.hd).enumerate() {
            let at = (pos0 + r) as f64;
            for j in 0..half {
                let (sn, cs) = (at * inv[j] as f64).sin_cos();
                row[j] = cs as f32;
                row[half + j] = sn as f32;
            }
        }
    }
    let run = |li: usize, loop_i: usize, h: &mut [f32], st: &mut DoryState, sc: &mut Scratch| {
        match &m.layers[li] {
            Layer::Attn(w) => {
                let sl = m.slot_loop(li, loop_i);
                let row = &mut st.slots[w.ordinal];
                if row.len() <= sl {
                    row.resize_with(sl + 1, KvSlot::default);
                }
                let recurrent = li >= c.n_in && li < c.n_in + c.n_rec;
                // A second read-out replaces the first one's rows too.
                let overwrite = c.kv == KvMode::LastLoop && (recurrent || li >= c.n_in + c.n_rec) && loop_i > 0;
                attn_layer(m, w, &mut row[sl], h, t, pos0, overwrite, pool, sc);
                for (x, o) in h.chunks_exact_mut(hidden).zip(sc.out.chunks_exact_mut(hidden)) {
                    slerp(x, o, &w.alpha);
                }
            }
            Layer::Mlp(w) => {
                mlp_layer(w, h, t, c.ffn, hidden, pool, sc);
                for (x, o) in h.chunks_exact_mut(hidden).zip(sc.out.chunks_exact_mut(hidden)) {
                    slerp(x, o, &w.alpha);
                }
            }
        }
    };
    for li in 0..c.n_in {
        run(li, 0, h, st, &mut sc);
    }
    let input = h.clone();
    let mut capture = capture;
    if let Some(cp) = capture.as_deref_mut() {
        cp.input.extend_from_slice(&input);
    }
    let mut state = input.clone();
    // Under a gate one row runs its first pass here, and the rest below.
    let early = t == 1 && m.gate.is_some();
    for i in 0..if early { 1 } else { loops } {
        for ((x, s), a) in h.iter_mut().zip(&state).zip(&input) {
            *x = *s + *a;
        }
        for row in h.chunks_exact_mut(hidden) {
            justnorm(row);
        }
        for li in c.n_in..c.n_in + c.n_rec {
            run(li, i, h, st, &mut sc);
        }
        state.copy_from_slice(h);
        for row in state.chunks_exact_mut(hidden) {
            justnorm(row);
        }
        if let Some(cp) = capture.as_deref_mut() {
            cp.loops.extend_from_slice(&state);
        }
    }
    // An early exit: one pass, a decision, and the second only if it asks for one.
    if let (true, Some(g)) = (t == 1 && early, m.gate.clone()) {
        let mut stay = false;
        let mut readout = false;
        if let Gate::Recon(top, bias) = &g {
            stay = top.score(&state) < *bias;
        }
        h.copy_from_slice(&state);
        if let Gate::Conf(th) = &g {
            if st.exits.probe() {
                readout = true;
                for li in c.n_in + c.n_rec..c.n_in + c.n_rec + c.n_out {
                    run(li, 0, h, st, &mut sc);
                }
                let mut lg = Vec::with_capacity(c.vocab);
                head(m, h, 1, Want::Last, pool, &mut lg);
                stay = top_prob(&lg) < *th;
                st.exits.note(!stay);
                if !stay {
                    st.exit_logits = Some(lg);
                }
            } else {
                stay = true;
            }
        } else if !stay {
            for li in c.n_in + c.n_rec..c.n_in + c.n_rec + c.n_out {
                run(li, 0, h, st, &mut sc);
            }
        }
        if stay {
            for ((x, s), a) in h.iter_mut().zip(&state).zip(&input) {
                *x = *s + *a;
            }
            justnorm(h);
            for li in c.n_in..c.n_in + c.n_rec {
                run(li, 1, h, st, &mut sc);
            }
            justnorm(h);
            for li in c.n_in + c.n_rec..c.n_in + c.n_rec + c.n_out {
                run(li, usize::from(readout), h, st, &mut sc);
            }
        }
        st.gate_tokens += 1;
        st.gate_one += u64::from(!stay);
    } else {
        h.copy_from_slice(&state);
        for li in c.n_in + c.n_rec..c.n_in + c.n_rec + c.n_out {
            run(li, 0, h, st, &mut sc);
        }
    }
    st.scratch = sc;
    if let Some(o1) = m.o1 {
        if pos0 + t >= o1.seal {
            seal_far(m, st, o1, false);
        }
    }
    // Sliding layers need only the window behind the next position. Trimming
    // waits for the chunk's end: a later last-loop pass reads what is behind it.
    let kv_row = c.nk * c.hd;
    for w in m.layers.iter().filter_map(|l| match l {
        Layer::Attn(w) => Some(w),
        _ => None,
    }) {
        let keep = if w.sliding { c.window } else { c.full_window.unwrap_or(usize::MAX / 4) };
        for s in st.slots[w.ordinal].iter_mut() {
            let n = s.rows(kv_row);
            if n > 2 * keep {
                let drop = n - keep;
                s.k.drain(..drop * kv_row);
                s.v.drain(..drop * kv_row);
                s.first += drop;
            }
        }
    }
}

/// Freeze the full-attention slots' landmarks over what they hold, and go on from
/// them row by row (`nystrom::NystromState::prefill_group`). Under a choice of exact rows
/// a prompt's end does this again over all its rows, with the chosen ones out of the far
/// field: a copy of a fact there drowns the exact one.
fn seal_far(m: &Dory, st: &mut DoryState, o1: DoryO1, last: bool) {
    let c = &m.cfg;
    let (nk, nh, hd) = (c.nk, c.nh, c.hd);
    let rep = nh / nk;
    let kv_row = nk * hd;
    let full: Vec<usize> = m
        .layers
        .iter()
        .filter_map(|l| match l {
            Layer::Attn(w) if !w.sliding => Some(w.ordinal),
            _ => None,
        })
        .collect();
    let keep = match (&m.exact, last) {
        (Some(x), true) => {
            let layers: Vec<(&[f32], &[f32])> = full.iter().map(|&o| (&st.slots[o][0].k[..], &st.slots[o][0].v[..])).collect();
            let rows = layers[0].0.len() / kv_row;
            pick_exact_rows(x, &layers, rows, nk, hd, o1.m, o1.sink, rows.saturating_sub(o1.w))
        }
        _ => Vec::new(),
    };
    for o in full {
        for s in st.slots[o].iter_mut() {
            if (!s.far.is_empty() && !last) || s.k.is_empty() {
                continue;
            }
            let t = s.k.len() / kv_row;
            let order = seal_order(t, o1.sink, &keep);
            let mut far = Vec::with_capacity(nk);
            for kh in 0..nk {
                let mut ks = vec![0f32; t * hd];
                let mut vs = vec![0f32; t * hd];
                let mut qs: Vec<Vec<f32>> = (0..rep).map(|_| vec![0f32; t * hd]).collect();
                for (r, &from) in order.iter().enumerate() {
                    let from = from as usize;
                    let at = (from * nk + kh) * hd;
                    ks[r * hd..(r + 1) * hd].copy_from_slice(&s.k[at..at + hd]);
                    vs[r * hd..(r + 1) * hd].copy_from_slice(&s.v[at..at + hd]);
                    for (h, q) in qs.iter_mut().enumerate() {
                        let at = (from * nh + kh * rep + h) * hd;
                        q[r * hd..(r + 1) * hd].copy_from_slice(&s.qt[at..at + hd]);
                    }
                }
                let mut n = crate::nystrom::NystromState::new_group(o1.m, o1.w, o1.sink + keep.len(), rep);
                let refs: Vec<&[f32]> = qs.iter().map(|q| &q[..]).collect();
                n.prefill_group(&refs, &ks, &vs, t, hd, hd);
                far.push(n);
            }
            s.far = far;
            if m.exact.is_some() && !last {
                s.arch = true;
            } else {
                s.arch = false;
                s.qt = Vec::new();
                s.k = Vec::new();
                s.v = Vec::new();
            }
        }
    }
}

/// A prompt's end on the host: the rows it should not lose leave the far field.
fn finish_exact(m: &Dory, st: &mut DoryState) {
    let Some(o1) = m.o1 else { return };
    if m.exact.is_some() && st.slots.iter().flatten().any(|s| s.arch) {
        seal_far(m, st, o1, true);
    }
}

/// The host's KV bytes: `(exact rows, O(1) states)`.
pub fn host_kv_bytes(st: &DoryState) -> (usize, usize) {
    let (mut exact, mut far) = (0, 0);
    for s in st.slots.iter().flatten() {
        exact += (s.k.len() + s.v.len()) * 4;
        far += s.far.iter().map(|n| n.memory_bytes()).sum::<usize>();
    }
    (exact, far)
}

/// Per-row signals of a forward, for the loop router.
#[derive(Default)]
pub struct Capture {
    pub input: Vec<f32>,
    pub loops: Vec<f32>,
}

/// What a forward leaves in `logits`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// Nothing: the KV state moved, no logits are read.
    None,
    Last,
    All,
}

/// How a token leaves the loop early: by its top probability after one pass
/// (`conf:0.7`), which reads the output block and the head before deciding, or by
/// patent-1 reconstruction over the state after one pass (`recon:FILE:bias`).
#[derive(Clone)]
pub enum Gate {
    Conf(f32),
    Recon(std::sync::Arc<Topologies>, f32),
}

pub struct Topologies {
    dim: usize,
    rank: usize,
    me: Vec<f32>,
    be: Vec<f32>,
    mh: Vec<f32>,
    bh: Vec<f32>,
}

impl Topologies {
    /// The router a turbo file carries.
    fn from_model(model: &CmfModel) -> Result<Self, String> {
        let dim = model.arch().hidden_size;
        let v = |n: &str| crate::loader::load_f32(model, n, &crate::loader::Overlay::None);
        let (me, be, mh, bh) = (v("dory.router.easy.mean")?, v("dory.router.easy.basis")?, v("dory.router.hard.mean")?, v("dory.router.hard.basis")?);
        if me.len() != dim || mh.len() != dim || be.is_empty() || be.len() % dim != 0 || bh.len() != be.len() {
            return Err("dory.router: shapes".into());
        }
        Ok(Self { dim, rank: be.len() / dim, me, be, mh, bh })
    }

    fn read(path: &str) -> Option<Self> {
        let b = std::fs::read(path).ok()?;
        let w: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let (rank, dim) = (w[0] as usize, w[1] as usize);
        let mut at = 2;
        let mut take = |n: usize| {
            let v = w[at..at + n].to_vec();
            at += n;
            v
        };
        let me = take(dim);
        let be = take(rank * dim);
        let mh = take(dim);
        let bh = take(rank * dim);
        Some(Self { dim, rank, me, be, mh, bh })
    }

    /// Reconstruction error of `x` by one topology.
    fn err(&self, x: &[f32], mu: &[f32], b: &[f32]) -> f32 {
        let y: Vec<f32> = x.iter().zip(mu).map(|(a, m)| a - m).collect();
        let mut e: f32 = y.iter().map(|v| v * v).sum();
        for r in b.chunks_exact(self.dim).take(self.rank) {
            let d: f32 = r.iter().zip(&y).map(|(a, c)| a * c).sum();
            e -= d * d;
        }
        e
    }

    /// Positive: the easy class reconstructs it better.
    pub fn score(&self, x: &[f32]) -> f32 {
        self.err(x, &self.mh, &self.bh) - self.err(x, &self.me, &self.be)
    }
}

fn exact_from_env() -> Option<ExactRows> {
    let v = std::env::var("CMF_DORY_EXACT").ok()?;
    let (file, rows) = v.rsplit_once(':')?;
    let b = std::fs::read(file).ok()?;
    let w: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Some(ExactRows { weights: Arc::new(w.get(1..)?.to_vec()), budget: rows.parse().ok()? })
}

fn gate_from_env() -> Option<Gate> {
    let v = std::env::var("CMF_DORY_GATE").ok()?;
    let mut it = v.split(':');
    match it.next()? {
        "conf" => Some(Gate::Conf(it.next()?.parse().ok()?)),
        "recon" => {
            let t = Topologies::read(it.next()?)?;
            Some(Gate::Recon(Arc::new(t), it.next()?.parse().ok()?))
        }
        _ => None,
    }
}

/// On text where the first read-out rarely ends a token it only costs time: below this
/// rate of early exits a token goes straight to the second pass, and one in eight still
/// reads out, to notice when the text turns easy.
#[derive(Clone, Copy)]
pub struct ExitRate {
    rate: f32,
    tokens: u32,
}

impl Default for ExitRate {
    fn default() -> Self {
        Self { rate: 1.0, tokens: 0 }
    }
}

impl ExitRate {
    /// Is this token worth a read-out after its first pass?
    pub fn probe(&mut self) -> bool {
        self.tokens = self.tokens.wrapping_add(1);
        self.rate >= 0.35 || self.tokens % 8 == 0
    }

    pub fn note(&mut self, exited: bool) {
        self.rate += 0.1 * (f32::from(u8::from(exited)) - self.rate);
    }
}

/// The top probability of a row of logits, without an exp per entry of the vocabulary.
pub fn top_prob(lg: &[f32]) -> f32 {
    let top = lg.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut den = 0f64;
    for &x in lg {
        let t = x - top;
        if t > -20.0 {
            den += (t as f64).exp();
        }
    }
    (1.0 / den) as f32
}

/// Rows of a forward on the host.
pub const HOST_CHUNK: usize = 64;

/// Host forward of a chunk.
#[allow(clippy::too_many_arguments)]
pub fn forward_host(
    m: &Dory,
    st: &mut DoryState,
    ids: &[u32],
    pos0: usize,
    loops: usize,
    want: Want,
    pool: Option<&Pool>,
    capture: Option<&mut Capture>,
    logits: &mut Vec<f32>,
) {
    let c = &m.cfg;
    if st.slots.is_empty() {
        st.slots.resize_with(m.attn_layers, Vec::new);
    }
    let mut capture = capture;
    let mut done = 0usize;
    logits.clear();
    let tail_from = (pos0 + ids.len()).saturating_sub(EXACT_TAIL);
    // q8's batch product has no fast path on x86: a row at a time goes 8 times faster
    // there, through the fused matvecs.
    let scalar_q8 = cfg!(target_arch = "x86_64")
        && m.layers.iter().any(|l| matches!(l, Layer::Attn(w) if matches!(w.q, QTensor::Mapped { dtype: cortiq_core::TensorDtype::Q8_2f, .. })));
    for chunk in ids.chunks(if scalar_q8 { 1 } else { HOST_CHUNK }) {
        let t = chunk.len();
        if pos0 + done >= tail_from {
            finish_exact(m, st);
        }
        let mut h = vec![0f32; t * c.hidden];
        for (i, row) in h.chunks_exact_mut(c.hidden).enumerate() {
            if (chunk[i] as usize) < m.embed.rows() {
                m.embed.row_f32(chunk[i] as usize, row);
            }
        }
        let w = if done + t == ids.len() || want == Want::All { want } else { Want::None };
        stack(m, st, &mut h, t, pos0 + done, m.chunk_loops(loops, w), pool, capture.as_deref_mut());
        done += t;
        st.pos = pos0 + done;
        if let Some(lg) = st.exit_logits.take() {
            if want != Want::None {
                if want == Want::Last {
                    logits.clear();
                }
                logits.extend_from_slice(&lg);
            }
            continue;
        }
        match want {
            Want::All => head(m, &h, t, Want::All, pool, logits),
            Want::Last if done == ids.len() => {
                logits.clear();
                head(m, &h, t, Want::Last, pool, logits)
            }
            _ => {}
        }
    }
    finish_exact(m, st);
}

/// The head over the wanted rows, with the per-token logit scale: `logits` grows.
pub fn head(m: &Dory, h: &[f32], t: usize, want: Want, pool: Option<&Pool>, logits: &mut Vec<f32>) {
    let c = &m.cfg;
    let (rows, from) = match want {
        Want::None => return,
        Want::Last => (1, t - 1),
        Want::All => (t, 0),
    };
    let at = logits.len();
    logits.resize(at + rows * c.vocab, 0.0);
    let x = &h[from * c.hidden..t * c.hidden];
    if rows == 1 && !crate::gptq_capture::capturing() {
        m.head.matvec(x, &mut logits[at..], pool);
    } else {
        m.head.matmat(x, rows, &mut logits[at..], pool);
    }
    for row in logits[at..].chunks_exact_mut(c.vocab) {
        for (x, s) in row.iter_mut().zip(&m.logit_scale) {
            *x *= *s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(window: usize, kv: KvMode, loops: usize) -> Dory {
        let (hidden, ffn, nh, nk, hd, vocab) = (16usize, 16usize, 4usize, 2usize, 8usize, 11usize);
        let mut seed = 1u32;
        let mut rnd = move |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.7
                })
                .collect()
        };
        let mat = |v: Vec<f32>, r: usize, c: usize| QTensor::from_f32(v, r, c);
        let q_size = nh * hd;
        let kv_row = nk * hd;
        let pattern = "S-*-S-*-S-*-";
        let mut layers = Vec::new();
        let mut ordinal = 0;
        for ch in pattern.chars() {
            if ch == '-' {
                layers.push(Layer::Mlp(Box::new(MlpWeights {
                    gate: mat(rnd(ffn * hidden), ffn, hidden),
                    up: mat(rnd(ffn * hidden), ffn, hidden),
                    down: mat(rnd(hidden * ffn), hidden, ffn),
                    sg: vec![1.5; ffn],
                    su: vec![1.2; ffn],
                    alpha: vec![0.3; hidden],
                })));
            } else {
                layers.push(Layer::Attn(Box::new(AttnWeights {
                    q: mat(rnd(q_size * hidden), q_size, hidden),
                    k: mat(rnd(kv_row * hidden), kv_row, hidden),
                    v: mat(rnd(kv_row * hidden), kv_row, hidden),
                    g: mat(rnd(q_size * hidden), q_size, hidden),
                    o: mat(rnd(hidden * q_size), hidden, q_size),
                    sq: vec![4.0; q_size],
                    sg: vec![2.0; q_size],
                    alpha: vec![0.2; hidden],
                    sliding: ch == 'S',
                    ordinal,
                })));
                ordinal += 1;
            }
        }
        Dory {
            file: None,
            cfg: DoryCfg {
                hidden,
                ffn,
                vocab,
                nh,
                nk,
                hd,
                window,
                n_in: 4,
                n_rec: 4,
                n_out: 4,
                loops,
                kv,
                attn_scale: 1.0 / (hd as f32).sqrt(),
                full_window: None,
            },
            embed: mat(rnd(vocab * hidden), vocab, hidden),
            head: mat(rnd(vocab * hidden), vocab, hidden),
            logit_scale: vec![3.0; vocab],
            layers,
            inv_full: crate::attention::rope_inv_freq(hd, 1.0e7),
            inv_sliding: crate::attention::rope_inv_freq(hd, 1.0e4),
            attn_layers: ordinal,
            o1: None,
            gate: None,
            prompt_loops: 0,
            exact: None,
        }
    }

    fn run(m: &Dory, ids: &[u32], chunk: usize) -> Vec<f32> {
        let mut st = DoryState::default();
        let mut logits = Vec::new();
        let mut at = 0;
        for c in ids.chunks(chunk) {
            logits.clear();
            forward_host(m, &mut st, c, at, m.cfg.loops, Want::Last, None, None, &mut logits);
            at += c.len();
        }
        logits
    }

    #[test]
    fn exp8_tracks_the_library_exp() {
        for v in [-80.0f32, -20.0, -3.3, -1.0, -0.01, 0.0, 0.5, 2.7, 13.0, 60.0] {
            let e = exp8(&[v; 8])[3];
            assert!((e / v.exp() - 1.0).abs() < 1e-6, "{v}: {e} {}", v.exp());
        }
    }

    #[test]
    fn activation_matches_the_scalar_form() {
        let n = 21;
        let mut g: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.37).collect();
        let g0 = g.clone();
        let u: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 * 0.01).collect();
        let (a, b) = (vec![1.3f32; n], vec![0.7f32; n]);
        silu_rows(&mut g, &u, &a, &b);
        for i in 0..n {
            let x = g0[i] * a[i];
            let want = x / (1.0 + (-x).exp()) * (u[i] * b[i]);
            assert!((g[i] - want).abs() <= 1e-5 * want.abs().max(1.0), "{i}: {} {want}", g[i]);
        }
    }

    #[test]
    fn a_chunk_and_its_rows_give_the_same_logits() {
        let ids: Vec<u32> = (0..23).map(|i| (i * 7 % 11) as u32).collect();
        // Past the window on the sliding layers, per-loop caches.
        let m = tiny(5, KvMode::PerLoop, 2);
        let (a, b, c) = (run(&m, &ids, 1), run(&m, &ids, 4), run(&m, &ids, 64));
        for ((x, y), z) in a.iter().zip(&b).zip(&c) {
            assert!((x - y).abs() < 2e-4 && (x - z).abs() < 2e-4, "{x} {y} {z}");
        }
    }

    #[test]
    fn the_last_loop_cache_rewrites_a_position_in_place() {
        let m = tiny(5, KvMode::LastLoop, 2);
        let mut st = DoryState::default();
        let mut logits = Vec::new();
        forward_host(&m, &mut st, &[1, 2, 3, 4, 5, 6, 7, 8], 0, 2, Want::Last, None, None, &mut logits);
        let kv_row = m.cfg.nk * m.cfg.hd;
        for row in &st.slots {
            assert_eq!(row.len(), 1, "one slot per layer");
            assert_eq!(row[0].k.len(), 8 * kv_row);
        }
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_far_field_seals_and_goes_on_under_the_last_loop_cache() {
        let mut m = tiny(5, KvMode::LastLoop, 2);
        m.o1 = Some(DoryO1 { m: 4, w: 6, sink: 0, seal: 24 });
        let mut st = DoryState::default();
        let mut logits = Vec::new();
        let ids: Vec<u32> = (0..40).map(|i| (i * 5 % 11) as u32).collect();
        forward_host(&m, &mut st, &ids[..30], 0, 2, Want::Last, None, None, &mut logits);
        for (at, id) in ids.iter().enumerate().skip(30) {
            logits.clear();
            forward_host(&m, &mut st, std::slice::from_ref(id), at, 2, Want::Last, None, None, &mut logits);
            assert!(logits.iter().all(|v| v.is_finite()));
        }
        let (exact, far) = host_kv_bytes(&st);
        assert!(far > 0 && exact > 0, "the full layers went over to the far field");
    }

    #[test]
    fn exact_rows_take_what_the_landmarks_cannot_span() {
        // Keys in a thin subspace, one row off it: the row and its neighbours come first.
        let (rows, nk, hd) = (400usize, 1usize, 16usize);
        let mut ks = vec![0f32; rows * hd];
        for r in 0..rows {
            ks[r * hd] = 1.0;
            ks[r * hd + 1] = (r as f32 * 0.01).sin() * 0.1;
        }
        ks[200 * hd + 7] = 1.0;
        let vs = vec![1f32; rows * hd];
        let x = ExactRows { weights: Arc::new(vec![1.0, 0.0]), budget: 5 };
        let keep = pick_exact_rows(&x, &[(&ks, &vs)], rows, nk, hd, 8, 4, 300);
        assert_eq!(keep, vec![198, 199, 200, 201, 202]);
    }

    #[test]
    fn a_prompt_keeps_its_chosen_rows_past_the_seal() {
        let mut m = tiny(5, KvMode::LastLoop, 2);
        m.o1 = Some(DoryO1 { m: 4, w: 6, sink: 1, seal: 24 });
        let layers = m.layers.iter().filter(|l| matches!(l, Layer::Attn(w) if !w.sliding)).count();
        m.exact = Some(ExactRows { weights: Arc::new(vec![1.0; layers * 2 * m.cfg.nk]), budget: 3 });
        let mut st = DoryState::default();
        let mut logits = Vec::new();
        let ids: Vec<u32> = (0..40).map(|i| (i * 5 % 11) as u32).collect();
        forward_host(&m, &mut st, &ids, 0, 2, Want::Last, None, None, &mut logits);
        let mut seen = 0;
        for s in st.slots.iter().flatten().filter(|s| !s.far.is_empty()) {
            assert!(!s.arch && s.k.is_empty(), "the rows are gone from the exact store");
            assert_eq!(s.far[0].device_view().sink_len, 4);
            seen += 1;
        }
        assert!(seen > 0);
        logits.clear();
        forward_host(&m, &mut st, &[3], 40, 2, Want::Last, None, None, &mut logits);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rollback_keeps_the_prefix_a_new_turn_extends() {
        let m = tiny(50, KvMode::PerLoop, 2);
        let ids: Vec<u32> = (0..12).map(|i| (i * 3 % 11) as u32).collect();
        let want = run(&m, &ids, 1);
        let mut st = DoryState::default();
        let mut logits = Vec::new();
        forward_host(&m, &mut st, &ids[..9], 0, 2, Want::Last, None, None, &mut logits);
        st.rollback(5, m.cfg.nk * m.cfg.hd);
        forward_host(&m, &mut st, &ids[5..], 5, 2, Want::Last, None, None, &mut logits);
        for (x, y) in logits.iter().zip(&want) {
            assert!((x - y).abs() < 2e-4, "{x} {y}");
        }
    }

    #[test]
    fn top_prob_of_a_sure_row() {
        let mut lg = vec![-30.0f32; 100];
        lg[7] = 30.0;
        assert!(top_prob(&lg) > 0.999);
        assert!((top_prob(&[1.0, 1.0]) - 0.5).abs() < 1e-5);
    }
}
