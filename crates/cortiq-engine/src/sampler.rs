//! Token sampling — temperature, top-p, top-k, min-p, repetition penalty.
//!
//! Randomness comes from an explicit SplitMix64 PRNG carried by the
//! caller: reproducible with a seed, unbiased across the whole CDF
//! (the v1 `subsec_nanos` source could never pick past ~23% of it).

use serde::{Deserialize, Serialize};

/// SplitMix64 — tiny, fast, statistically solid for sampling.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

/// Reusable per-pipeline sampling workspace. The epoch table lets the
/// repetition penalty visit each token id once without allocating a HashSet
/// or clearing a vocab-sized boolean vector on every decode step.
#[derive(Debug, Default)]
pub struct SamplerScratch {
    seen_epoch: Vec<u32>,
    epoch: u32,
    /// Distinct-token set for the presence penalty; reused per token.
    presence_seen: std::collections::HashSet<u32>,
    /// The working copy of the logits. At a 129k vocab that is half a
    /// megabyte allocated, filled and dropped per token; the struct that
    /// exists to hold scratch may as well hold this one too.
    probs: Vec<f32>,
    /// The SECOND whole-vocab copy — top-k's partition buffer. Qwen3.6's
    /// vocab is 248320, so this was another megabyte allocated, filled
    /// and dropped per token, on the same hot path and for the same
    /// reason. Same fix.
    topk: Vec<f32>,
    /// The sparse chain's candidate list and its per-grain partials.
    cand: Vec<(u32, f32)>,
    cand_parts: Vec<Vec<(u32, f32)>>,
    sum_parts: Vec<f32>,
    sparse: Sparse,
}

impl SamplerScratch {
    fn begin_seen(&mut self, vocab_size: usize) -> u32 {
        if self.seen_epoch.len() < vocab_size {
            self.seen_epoch.resize(vocab_size, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen_epoch.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Seed from OS entropy (address-space + time mix) when none given.
    pub fn from_entropy() -> Self {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let addr = Box::into_raw(Box::new(0u8)) as u64;
        // SAFETY: pointer came from Box::into_raw just above.
        unsafe { drop(Box::from_raw(addr as *mut u8)) };
        Self::new(t.as_nanos() as u64 ^ addr.rotate_left(17) ^ 0x9E3779B97F4A7C15)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sampling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    /// Flat additive penalty on every token that has appeared at least
    /// once (OpenAI-style presence penalty). Qwen3.8's instruct sampling
    /// asks for 1.5 here — the multiplicative repetition_penalty is a
    /// different curve and cannot stand in for it.
    #[serde(default)]
    pub presence_penalty: f32,
    /// Fixed seed for reproducible generation (None = entropy).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Token IDs to suppress (force logit to -inf).
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            min_p: 0.05,
            seed: None,
            suppress_tokens: Vec::new(),
        }
    }
}

/// Sample next token from logits. Chain order is fixed:
/// rep-penalty → temperature → softmax → min-p → top-k → top-p → sample.
pub fn sample(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    rng: &mut SplitMix64,
) -> u32 {
    let mut scratch = SamplerScratch::default();
    sample_with_scratch(logits, config, past_tokens, rng, &mut scratch)
}

/// Sampling entry point for hot decode loops with reusable scratch storage.
pub fn sample_with_scratch(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    rng: &mut SplitMix64,
    scratch: &mut SamplerScratch,
) -> u32 {
    sample_with_scratch_pool(logits, config, past_tokens, rng, scratch, None)
}

/// The same chain with the whole-vocab passes spread over the CPU pool.
///
/// WHY: at Qwen3.8's 248 320-entry vocab the serial sampler is ~14 passes
/// over a megabyte plus 248k `exp` and a `select_nth` over a second copy
/// — measured on the RTX 5090 pod as the gap between `bench --core` and
/// the production loop (50.9 against 46.8 tok/s with penalty+confidence
/// alone; the temperature path pays the softmax and the partition on top).
/// The GPU graph owns the token, so during decode the pool sits idle —
/// this is free work.
///
/// WHAT IS PRESERVED: every value. The parallel passes are elementwise
/// (each output depends on its own input only), the sums that feed
/// divisions stay sequential in index order, and top-k's threshold is
/// the k-th largest VALUE — the same number `select_nth` returned. The
/// sampled token is bit-identical to the serial chain for the same seed.
pub fn sample_with_scratch_pool(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    rng: &mut SplitMix64,
    scratch: &mut SamplerScratch,
    pool: Option<&crate::pool::Pool>,
) -> u32 {
    if config.temperature < 1e-6
        && config.repetition_penalty == 1.0
        && config.presence_penalty == 0.0
        && config.suppress_tokens.is_empty()
    {
        return argmax(logits);
    }
    // Borrowed from the scratch and handed back at the single exit: at a
    // 129k vocab this copy is half a megabyte allocated, filled and dropped
    // per token, and the struct that exists to hold scratch may as well
    // hold it. Every early return goes through `done` so the buffer never
    // leaks back to the allocator.
    if config.temperature < 1e-6 {
        // greedy over the penalized logits, no working copy
        return argmax_penalized(logits, config, past_tokens, scratch, pool);
    }
    if sparse_ok(config) {
        // The sparse chain: same distribution, a tenth of the passes.
        let mut sp = std::mem::take(&mut scratch.sparse);
        let ok = sparse_distribution_into(logits, config, past_tokens, scratch, pool, &mut sp);
        let t = if ok {
            draw_sparse(&sp, rng)
        } else {
            argmax(logits)
        };
        scratch.sparse = sp;
        return t;
    }
    let mut probs = std::mem::take(&mut scratch.probs);
    let normalized = chain(logits, config, past_tokens, scratch, pool, &mut probs);

    let mut done = |probs: Vec<f32>, tok: u32| -> u32 {
        scratch.probs = probs;
        tok
    };

    if !normalized {
        // Everything filtered out — fall back to greedy over original logits.
        let t = argmax(logits);
        return done(probs, t);
    }
    let t = categorical_sample(&probs, rng.next_f32());
    done(probs, t)
}

/// Greedy over the PENALIZED logits without the working copy: one pass
/// that applies the repetition / presence penalty and the suppress list
/// on the fly (a membership table over the vocab, built from the past
/// tokens) and keeps the argmax with the same tie rule as `argmax`
/// (highest index among equal maxima). Bit-identical to
/// `chain` + `argmax` for temperature 0 — the values compared are the
/// same expressions — and it is what a greedy decode with penalties pays
/// per token, and what a speculative round pays per draft and per
/// verified row (nine such passes a round at k=4).
pub fn argmax_penalized(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    scratch: &mut SamplerScratch,
    pool: Option<&crate::pool::Pool>,
) -> u32 {
    let n = logits.len();
    if config.repetition_penalty == 1.0
        && config.presence_penalty == 0.0
        && config.suppress_tokens.is_empty()
    {
        return argmax(logits);
    }
    // Membership: seen_epoch[i] == epoch for past tokens (each once).
    let epoch = scratch.begin_seen(n);
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < n {
            scratch.seen_epoch[idx] = epoch;
        }
    }
    let rep = config.repetition_penalty;
    let pres = config.presence_penalty;
    // Suppressed ids get -inf; a second, rarer set — keep it exact.
    let suppress = &config.suppress_tokens;
    let seen = &scratch.seen_epoch;
    let value = |i: usize| -> f32 {
        let mut v = logits[i];
        if suppress.iter().any(|&t| t as usize == i) {
            return f32::NEG_INFINITY;
        }
        if seen[i] == epoch {
            if rep != 1.0 {
                if v > 0.0 {
                    v /= rep;
                } else {
                    v *= rep;
                }
            }
            if pres != 0.0 {
                v -= pres;
            }
        }
        v
    };
    // The suppress list is scanned per element above; keep that path
    // serial and rare. The common (no suppress) case runs the pool.
    let best_in = |s: usize, e: usize| -> (usize, f32) {
        let mut bi = s;
        let mut bv = f32::NEG_INFINITY;
        for i in s..e {
            let v = value(i);
            if v >= bv {
                bv = v;
                bi = i;
            }
        }
        (bi, bv)
    };
    match pool {
        Some(p) if n >= PAR_MIN && suppress.is_empty() => {
            let m = std::sync::Mutex::new(Vec::<(usize, f32)>::new());
            p.run_rows(n, &|s, e| {
                let r = best_in(s, e);
                m.lock().unwrap().push(r);
            });
            let mut parts = m.into_inner().unwrap();
            // Same rule across chunks: the max value, and among equal
            // maxima the HIGHEST index — chunk order does not matter once
            // sorted by index.
            parts.sort_by_key(|(i, _)| *i);
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, v) in parts {
                if v >= bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        }
        _ => best_in(0, n).0 as u32,
    }
}

/// The chain up to the draw, into `probs`: penalties, temperature,
/// softmax, min-p, top-k, top-p, renormalize. Returns false when the
/// filters left nothing (the caller's greedy fallback); for a greedy
/// config it stops after the penalties (`probs` then holds penalized
/// logits, and argmax over them is the token).
fn chain(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    scratch: &mut SamplerScratch,
    pool: Option<&crate::pool::Pool>,
    probs: &mut Vec<f32>,
) -> bool {
    probs.clear();
    probs.extend_from_slice(logits);
    apply_penalties(probs, config, past_tokens, scratch);

    if config.temperature < 1e-6 {
        return true;
    }
    if config.temperature != 1.0 {
        let t = config.temperature;
        par_map(pool, probs, &move |p| p / t);
    }

    softmax_inplace_pool(pool, probs);

    if config.min_p > 0.0 {
        let max_prob = par_max(pool, probs, 0.0);
        let threshold = max_prob * config.min_p;
        par_map(pool, probs, &move |p| if p < threshold { 0.0 } else { p });
    }

    if config.top_k > 0 && (config.top_k as usize) < probs.len() {
        apply_top_k_pool(pool, probs, config.top_k as usize);
    }

    if config.top_p < 1.0 && config.top_p > 0.0 {
        apply_top_p(probs, config.top_p);
    }

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        par_map(pool, probs, &move |p| p / sum);
        true
    } else {
        false
    }
}

/// The chain's first stage — suppress, repetition and presence penalties
/// — in place on a working copy of the logits. Every penalty only LOWERS
/// a logit, which is what lets the sparse chain below bound its
/// candidates.
fn apply_penalties(
    probs: &mut [f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    scratch: &mut SamplerScratch,
) {
    for &tok in &config.suppress_tokens {
        if (tok as usize) < probs.len() {
            probs[tok as usize] = f32::NEG_INFINITY;
        }
    }
    if config.repetition_penalty != 1.0 {
        apply_repetition_penalty(probs, past_tokens, config.repetition_penalty, scratch);
    }
    if config.presence_penalty != 0.0 {
        // Once per DISTINCT seen token — presence, not frequency. The
        // scratch set the repetition penalty uses would serve, but it is
        // only built on its own branch; a local pass stays correct when
        // rep-penalty is 1.0 (Qwen3.8's recommended pairing).
        let mut seen = std::mem::take(&mut scratch.presence_seen);
        seen.clear();
        seen.extend(past_tokens.iter().copied());
        for &tok in &seen {
            if (tok as usize) < probs.len() {
                probs[tok as usize] -= config.presence_penalty;
            }
        }
        scratch.presence_seen = seen;
    }
}

fn config_penalized(config: &SamplerConfig) -> bool {
    config.repetition_penalty != 1.0
        || config.presence_penalty != 0.0
        || !config.suppress_tokens.is_empty()
}

/// Largest top-k the sparse chain serves. Past this the dense chain is
/// the better tool anyway.
pub const SPARSE_TOPK_MAX: usize = 256;

/// Whether `config` can go through the sparse chain: a real temperature
/// and a top-k in 1..=256. Qwen's recommended instruct settings
/// (0.7 / top-p 0.8 / top-k 20 / presence 1.5) do.
pub fn sparse_ok(config: &SamplerConfig) -> bool {
    config.temperature >= 1e-6 && config.top_k > 0 && (config.top_k as usize) <= SPARSE_TOPK_MAX
}

/// A distribution over at most `SPARSE_TOPK_MAX` tokens: `(id, prob)`
/// sorted by id, probs summing to 1.
pub type Sparse = Vec<(u32, f32)>;

/// The sampler chain's distribution as a SPARSE list — the same
/// distribution `chain` builds over the whole vocab, for configs with a
/// top-k, at a fraction of the cost. The dense chain copies the vocab,
/// exponentiates it, selects, filters and normalises it — six or seven
/// passes over 248k floats — and every one of them past the selection
/// touches only the k survivors. Here: penalties on a copy ONLY when
/// there are penalties, one pooled pass that selects the top-k penalized
/// logits, one pooled pass for the vocab-wide softmax denominator (top-p
/// is defined against the FULL normalisation, so the denominator must
/// see every token), and the rest over k entries.
///
/// Why it is the same distribution: softmax → min-p → top-k → top-p →
/// renormalise, in the dense order. Softmax is monotone in the logit, so
/// the top-k SET is the top-k of the penalized logits; min-p drops
/// tokens below `max_prob·min_p`, i.e. below `exp((l − l_max)/T) <
/// min_p` — every token outside the top-k that fails it is dropped
/// either way, and the ones inside are tested exactly as the dense chain
/// tests them; top-p cuts on the cumulative FULL-normalised probs of the
/// survivors sorted descending, computed here from the same terms. What
/// differs is floating-point: the denominator's summation order and the
/// exp of `(l − l_max)/T` against `l/T − max(l/T)`. Ties at the k-th
/// place resolve by lower id here where the dense chain keeps them all.
///
/// Returns false when the chain filtered everything (the dense chain's
/// `!normalized`) — the caller falls back to greedy the same way.
pub fn sparse_distribution_into(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    scratch: &mut SamplerScratch,
    pool: Option<&crate::pool::Pool>,
    out: &mut Sparse,
) -> bool {
    debug_assert!(sparse_ok(config));
    out.clear();
    let k = (config.top_k as usize).min(logits.len());
    if k == 0 {
        return false;
    }
    let penalized = config_penalized(config);
    let mut probs = std::mem::take(&mut scratch.probs);
    if penalized {
        probs.clear();
        probs.extend_from_slice(logits);
        apply_penalties(&mut probs, config, past_tokens, scratch);
    }
    let src: &[f32] = if penalized { &probs } else { logits };
    let t = if config.temperature > 0.0 {
        config.temperature
    } else {
        1.0
    };
    let mut cand = std::mem::take(&mut scratch.cand);
    par_topk(pool, src, k, &mut cand, &mut scratch.cand_parts);
    // `cand` is by value descending, ties by id — the top-k AND every tie
    // at the k-th place, as the dense chain keeps them.
    let ok = if let Some(&(_, lmax)) = cand.first().filter(|c| c.1.is_finite()) {
        let sum_all = par_sum_exp(pool, src, lmax, t, &mut scratch.sum_parts);
        // e_i = exp((l_i − l_max)/T); min-p against e_i < min_p (max_prob
        // is e = 1 over the same denominator); probs e_i / sum_all.
        let min_p = config.min_p;
        let mut cum = 0.0f32;
        let mut cut = false;
        for &(id, l) in cand.iter() {
            if cut {
                break;
            }
            let e = ((l - lmax) / t).exp();
            if min_p > 0.0 && e < min_p {
                continue;
            }
            let pr = e / sum_all;
            if pr <= 0.0 {
                continue;
            }
            out.push((id, pr));
            cum += pr;
            if config.top_p < 1.0 && config.top_p > 0.0 && cum >= config.top_p {
                cut = true;
            }
        }
        // renormalise over the survivors and order by id
        let sum: f32 = out.iter().map(|c| c.1).sum();
        if sum > 0.0 {
            for c in out.iter_mut() {
                c.1 /= sum;
            }
            out.sort_unstable_by_key(|c| c.0);
            true
        } else {
            out.clear();
            false
        }
    } else {
        false
    };
    scratch.cand = cand;
    scratch.probs = probs;
    ok
}

/// The k largest of `src` as `(id, value)` sorted by value descending,
/// ties by id ascending — INCLUDING every value tied with the k-th, which
/// is what the dense chain's `p < threshold → 0` keeps. Two pooled passes:
/// a per-grain k-slot selection merged in grain order for the k-th value,
/// then a gather of everything at or above it. Nothing here depends on
/// scheduling, so a seed reproduces.
fn par_topk(
    pool: Option<&crate::pool::Pool>,
    src: &[f32],
    k: usize,
    out: &mut Vec<(u32, f32)>,
    parts: &mut Vec<Vec<(u32, f32)>>,
) {
    let better = |a: (u32, f32), b: (u32, f32)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
    // sorted insertion into a fixed k-slot list: the compare against the
    // current k-th is what nearly every element pays, and nothing else.
    let scan = |s: usize, e: usize, best: &mut Vec<(u32, f32)>| {
        best.clear();
        for i in s..e {
            let c = (i as u32, src[i]);
            if best.len() < k {
                let pos = best.iter().position(|&b| better(c, b)).unwrap_or(best.len());
                best.insert(pos, c);
            } else if better(c, best[k - 1]) {
                let pos = best.iter().position(|&b| better(c, b)).unwrap_or(k - 1);
                best.pop();
                best.insert(pos, c);
            }
        }
    };
    let by_value_desc = |a: &(u32, f32), b: &(u32, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    // A degenerate row (thousands tied at the k-th place) is capped: the
    // dense chain would keep them all; nobody samples such a row on
    // purpose.
    let cap = k * 4 + 64;
    // gather everything ≥ kth into `slot`, at most `cap` entries
    let gather = |s: usize, e: usize, kth: f32, slot: &mut Vec<(u32, f32)>| {
        slot.clear();
        for i in s..e {
            let v = src[i];
            if v >= kth {
                slot.push((i as u32, v));
                if slot.len() >= cap {
                    break;
                }
            }
        }
    };
    out.clear();
    match pool {
        Some(p) if src.len() >= PAR_MIN => {
            let n = src.len();
            let grain = crate::pool::grain_for(n, p.n_workers() + 1);
            let ng = n.div_ceil(grain);
            parts.resize_with(ng, Vec::new);
            let pp = crate::pool::SendMutT::new(parts.as_mut_ptr());
            p.run_rows(n, &|s, e| {
                // SAFETY: grain g is written by exactly one range (start =
                // g·grain) and `parts` outlives the joined dispatch.
                let slot = unsafe { &mut *pp.at(s / grain) };
                scan(s, e, slot);
            });
            for g in 0..ng {
                out.extend_from_slice(&parts[g]);
            }
            out.sort_unstable_by(by_value_desc);
            out.truncate(k);
            let Some(&(_, kth)) = out.last() else {
                return;
            };
            if !kth.is_finite() {
                return; // -inf ties are the filtered-out set; keep the k
            }
            p.run_rows(n, &|s, e| {
                let slot = unsafe { &mut *pp.at(s / grain) };
                gather(s, e, kth, slot);
            });
            out.clear();
            for g in 0..ng {
                out.extend_from_slice(&parts[g]);
                if out.len() >= cap {
                    break;
                }
            }
            out.sort_unstable_by(by_value_desc);
            out.truncate(cap);
        }
        _ => {
            scan(0, src.len(), out);
            let Some(&(_, kth)) = out.last() else {
                return;
            };
            if !kth.is_finite() {
                return;
            }
            let mut all = std::mem::take(out);
            gather(0, src.len(), kth, &mut all);
            all.sort_unstable_by(by_value_desc);
            all.truncate(cap);
            *out = all;
        }
    }
}

/// Σ exp((l − lmax)/t) over the vocab, per-grain partials summed in
/// grain order (deterministic across runs, so a seed reproduces).
fn par_sum_exp(
    pool: Option<&crate::pool::Pool>,
    src: &[f32],
    lmax: f32,
    t: f32,
    parts: &mut Vec<f32>,
) -> f32 {
    let term = |s: usize, e: usize| -> f32 {
        let mut acc = 0.0f32;
        for &l in &src[s..e] {
            acc += ((l - lmax) / t).exp();
        }
        acc
    };
    match pool {
        Some(p) if src.len() >= PAR_MIN => {
            let n = src.len();
            let grain = crate::pool::grain_for(n, p.n_workers() + 1);
            let ng = n.div_ceil(grain);
            parts.clear();
            parts.resize(ng, 0.0);
            let pp = crate::pool::SendMut::new(parts.as_mut_ptr());
            p.run_rows(n, &|s, e| {
                // SAFETY: one writer per grain slot; joined before read.
                unsafe { *pp.at(s / grain) = term(s, e) };
            });
            parts.iter().sum()
        }
        _ => term(0, src.len()),
    }
}

/// Draw from a sparse distribution: inverse CDF in id order — the same
/// walk the dense `categorical_sample` makes over the vocab, so a seed
/// lands on the same token when the survivor set and probs agree.
pub fn draw_sparse(p: &[(u32, f32)], rng: &mut SplitMix64) -> u32 {
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for &(id, pr) in p {
        cum += pr;
        if r < cum {
            return id;
        }
    }
    p.iter().rev().find(|c| c.1 > 0.0).map(|c| c.0).unwrap_or(0)
}

fn sparse_get(p: &[(u32, f32)], id: u32) -> f32 {
    p.binary_search_by_key(&id, |c| c.0)
        .map(|i| p[i].1)
        .unwrap_or(0.0)
}

/// `spec_accept_or_correct` over sparse distributions: accept the draft
/// `d` with min(1, p[d]/q[d]); on rejection draw the correction from the
/// residual max(0, p − q) over p's support (q's support outside p
/// contributes nothing to the residual). Empty residual → a draw from p.
pub fn spec_accept_or_correct_sparse(
    p: &[(u32, f32)],
    q: &[(u32, f32)],
    d: u32,
    rng: &mut SplitMix64,
    res: &mut Sparse,
) -> Option<u32> {
    let (pd, qd) = (sparse_get(p, d), sparse_get(q, d));
    let r = rng.next_f32();
    if qd > 0.0 && r * qd < pd {
        return None;
    }
    res.clear();
    let mut total = 0.0f32;
    for &(id, pi) in p {
        let ri = pi - sparse_get(q, id);
        if ri > 0.0 {
            res.push((id, ri));
            total += ri;
        }
    }
    if total <= 0.0 {
        return Some(draw_sparse(p, rng));
    }
    for c in res.iter_mut() {
        c.1 /= total;
    }
    Some(draw_sparse(res, rng))
}

/// The distribution the sampler would draw from — the whole chain minus
/// the draw — as a normalized vector over the vocab, in `out`. Greedy
/// configs (and the filtered-out fallback) come back as a one-hot, so a
/// caller can treat every configuration uniformly. This is what
/// speculative SAMPLING needs from both the draft head and the verify:
/// accept-with-min(1, p/q), correct from max(0, p − q).
pub fn distribution_into(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[u32],
    scratch: &mut SamplerScratch,
    pool: Option<&crate::pool::Pool>,
    out: &mut Vec<f32>,
) {
    let one_hot = |out: &mut Vec<f32>, t: usize, n: usize| {
        out.clear();
        out.resize(n, 0.0);
        if t < n {
            out[t] = 1.0;
        }
    };
    if config.temperature < 1e-6
        && config.repetition_penalty == 1.0
        && config.presence_penalty == 0.0
        && config.suppress_tokens.is_empty()
    {
        return one_hot(out, argmax(logits) as usize, logits.len());
    }
    let mut probs = std::mem::take(&mut scratch.probs);
    let normalized = chain(logits, config, past_tokens, scratch, pool, &mut probs);
    if config.temperature < 1e-6 {
        let t = argmax(&probs) as usize;
        scratch.probs = probs;
        return one_hot(out, t, logits.len());
    }
    if !normalized {
        scratch.probs = probs;
        return one_hot(out, argmax(logits) as usize, logits.len());
    }
    out.clear();
    out.extend_from_slice(&probs);
    scratch.probs = probs;
}

/// Draw from a normalized distribution with the caller's RNG.
pub fn draw(probs: &[f32], rng: &mut SplitMix64) -> u32 {
    categorical_sample(probs, rng.next_f32())
}

/// One step of speculative sampling (Leviathan et al. / Chen et al.):
/// the draft `d` was drawn from `q`; the target distribution at the same
/// position is `p`. Returns `None` when `d` is accepted (with probability
/// min(1, p[d]/q[d])) and `Some(c)` when it is rejected, `c` drawn from
/// the residual max(0, p − q) renormalized — which is exactly what makes
/// the emitted token stream distributed as `p`, draft or no draft. When
/// the residual is empty (p ⊆ q, so p == q on the support) the correction
/// falls back to a draw from `p` itself. `scratch` holds the residual;
/// the pool spreads the vocab-wide pass.
pub fn spec_accept_or_correct(
    p: &[f32],
    q: &[f32],
    d: u32,
    rng: &mut SplitMix64,
    scratch: &mut Vec<f32>,
    pool: Option<&crate::pool::Pool>,
) -> Option<u32> {
    let di = d as usize;
    let (pd, qd) = (
        p.get(di).copied().unwrap_or(0.0),
        q.get(di).copied().unwrap_or(0.0),
    );
    let r = rng.next_f32();
    // accept iff r < min(1, pd/qd)  ⇔  r·qd < pd (qd > 0 since d was drawn from q)
    if qd > 0.0 && r * qd < pd {
        return None;
    }
    let n = p.len().min(q.len());
    scratch.clear();
    scratch.extend_from_slice(&p[..n]);
    // residual = max(0, p − q), elementwise over the pool
    {
        let qp = q.as_ptr() as usize;
        let sm = crate::pool::SendMut::new(scratch.as_mut_ptr());
        let body = move |s: usize, e: usize| {
            // SAFETY: disjoint ranges; q outlives the (joined) dispatch.
            let qs = unsafe { std::slice::from_raw_parts(qp as *const f32, n) };
            for i in s..e {
                unsafe {
                    let x = sm.at(i);
                    *x = (*x - qs[i]).max(0.0);
                }
            }
        };
        match pool {
            Some(pl) if n >= PAR_MIN => pl.run_rows(n, &body),
            _ => body(0, n),
        }
    }
    let sum: f32 = scratch.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        par_map(pool, scratch, &move |v| v * inv);
        Some(categorical_sample(scratch, rng.next_f32()))
    } else {
        Some(categorical_sample(&p[..n], rng.next_f32()))
    }
}

/// Greedy: index of the maximum value.
///
/// Four running maxima instead of one: the scalar `max_by` carried a loop
/// dependency through the comparison, which at a 129k vocab is a tenth of a
/// millisecond of pure serial work per token.
///
/// Ties resolve to the HIGHEST index — not an arbitrary choice, it is what
/// `Iterator::max_by` does (it keeps the last of several equal maxima) and
/// therefore what this has always returned. `explain`'s preview compares
/// its own argmax against what greedy emits, and that test is what catches
/// the flip.
pub fn argmax(values: &[f32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let n = values.len();
    let mut best = [(0usize, f32::NEG_INFINITY); 4];
    for (l, b) in best.iter_mut().enumerate() {
        b.0 = l.min(n - 1);
    }
    let mut i = 0;
    while i + 4 <= n {
        for l in 0..4 {
            let v = values[i + l];
            if v >= best[l].1 {
                best[l] = (i + l, v);
            }
        }
        i += 4;
    }
    let mut bi = best[0].0;
    let mut bv = best[0].1;
    for b in &best[1..] {
        if b.1 > bv || (b.1 == bv && b.0 > bi) {
            bi = b.0;
            bv = b.1;
        }
    }
    while i < n {
        if values[i] >= bv {
            bv = values[i];
            bi = i;
        }
        i += 1;
    }
    bi as u32
}

fn softmax_inplace(logits: &mut [f32]) {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Below this length the pool's dispatch costs more than the pass.
const PAR_MIN: usize = 1 << 14;

/// Elementwise `buf[i] = f(buf[i])` over the pool (serial without one, or
/// for short buffers). Each output depends on its own input alone, so the
/// chunking cannot change a single bit.
fn par_map(pool: Option<&crate::pool::Pool>, buf: &mut [f32], f: &(dyn Fn(f32) -> f32 + Sync)) {
    match pool {
        Some(p) if buf.len() >= PAR_MIN => {
            let out = crate::pool::SendMut::new(buf.as_mut_ptr());
            p.run_rows(buf.len(), &move |s, e| {
                for i in s..e {
                    // SAFETY: ranges from run_rows are disjoint and the
                    // buffer outlives the (joined) dispatch.
                    unsafe {
                        let q = out.at(i);
                        *q = f(*q);
                    }
                }
            });
        }
        _ => {
            for v in buf.iter_mut() {
                *v = f(*v);
            }
        }
    }
}

/// `fold(init, f32::max)` over the pool. Max is order-free on non-NaN
/// input, so per-chunk maxima combined give the serial fold's answer.
fn par_max(pool: Option<&crate::pool::Pool>, buf: &[f32], init: f32) -> f32 {
    match pool {
        Some(p) if buf.len() >= PAR_MIN => {
            let m = std::sync::Mutex::new(init);
            p.run_rows(buf.len(), &|s, e| {
                let local = buf[s..e].iter().cloned().fold(init, f32::max);
                let mut g = m.lock().unwrap();
                *g = g.max(local);
            });
            m.into_inner().unwrap()
        }
        _ => buf.iter().cloned().fold(init, f32::max),
    }
}

/// `softmax_inplace` with the exp and the normalisation spread over the
/// pool. The max is order-free, the exp is elementwise, and the SUM stays
/// a sequential index-order fold — exactly the serial loop's accumulation
/// — so the probabilities are bit-identical.
fn softmax_inplace_pool(pool: Option<&crate::pool::Pool>, logits: &mut [f32]) {
    if pool.is_none() || logits.len() < PAR_MIN {
        return softmax_inplace(logits);
    }
    let max_val = par_max(pool, logits, f32::NEG_INFINITY);
    par_map(pool, logits, &move |v| (v - max_val).exp());
    let sum: f32 = logits.iter().sum();
    if sum > 0.0 {
        par_map(pool, logits, &move |v| v / sum);
    }
}

/// The k-th largest value of `probs` (k ≥ 1, k ≤ len), by the same
/// descending `partial_cmp` order `select_nth_unstable_by` used — one
/// streaming pass with a k-slot min-heap instead of a whole-vocab copy
/// and partition. Values, not indices, so ties give the same threshold.
fn kth_largest(probs: &[f32], k: usize) -> f32 {
    use std::cmp::Ordering;
    // Min-heap on the k largest seen so far: `heap[0]` is the smallest of
    // them, i.e. the running k-th largest.
    let mut heap: Vec<f32> = Vec::with_capacity(k);
    let desc = |a: f32, b: f32| b.partial_cmp(&a).unwrap_or(Ordering::Equal);
    let sift_down = |h: &mut [f32], mut i: usize| {
        let n = h.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            // child "smaller" in the descending order = later in it
            if l < n && desc(h[l], h[m]) == Ordering::Greater {
                m = l;
            }
            if r < n && desc(h[r], h[m]) == Ordering::Greater {
                m = r;
            }
            if m == i {
                break;
            }
            h.swap(i, m);
            i = m;
        }
    };
    let sift_up = |h: &mut [f32], mut i: usize| {
        while i > 0 {
            let parent = (i - 1) / 2;
            if desc(h[i], h[parent]) == Ordering::Greater {
                h.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    };
    for &v in probs {
        if heap.len() < k {
            heap.push(v);
            let n = heap.len();
            sift_up(&mut heap, n - 1);
        } else if desc(v, heap[0]) == Ordering::Less {
            // v is larger than the current k-th largest: replace it.
            heap[0] = v;
            sift_down(&mut heap, 0);
        }
    }
    heap[0]
}

/// `apply_top_k` without the second vocab-sized copy: the threshold is
/// the k-th largest value from one streaming pass, the zeroing is an
/// elementwise pass over the pool. Same kept set, same values.
fn apply_top_k_pool(pool: Option<&crate::pool::Pool>, probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let threshold = kth_largest(probs, k);
    par_map(pool, probs, &move |p| if p < threshold { 0.0 } else { p });
}

/// Top-1 probability of `id` under a softmax at temperature `temp` — the
/// per-token confidence — with the exp pass over the pool and the sum
/// sequential in index order (bit-identical to the serial fold). Uses
/// the scratch's partition buffer, idle now that top-k streams.
pub fn top1_prob_pool(
    pool: Option<&crate::pool::Pool>,
    scratch: &mut SamplerScratch,
    logits: &[f32],
    id: u32,
    temp: f32,
) -> f32 {
    let t = if temp > 1e-3 { temp } else { 1.0 };
    let max = par_max(pool, logits, f32::NEG_INFINITY);
    let mut e = std::mem::take(&mut scratch.topk);
    e.clear();
    e.extend_from_slice(logits);
    par_map(pool, &mut e, &move |v| ((v - max) / t).exp());
    let sum: f32 = e.iter().sum();
    let out = if sum > 0.0 {
        (((logits[id as usize] - max) / t).exp()) / sum
    } else {
        0.0
    };
    scratch.topk = e;
    out
}

fn apply_repetition_penalty(
    logits: &mut [f32],
    past_tokens: &[u32],
    penalty: f32,
    scratch: &mut SamplerScratch,
) {
    let epoch = scratch.begin_seen(logits.len());
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < logits.len() && scratch.seen_epoch[idx] != epoch {
            scratch.seen_epoch[idx] = epoch;
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Keep the k highest-probability tokens (plus exact ties at the
/// threshold), zero the rest. Selection, not a full vocab sort — the
/// old double `sort_by` over ~150k probs was ~1ms of pure per-token
/// overhead (roadmap §3 P0).
fn apply_top_k(probs: &mut [f32], k: usize, sel: &mut Vec<f32>) {
    if k == 0 || k >= probs.len() {
        return;
    }
    // `select_nth_unstable` permutes, so the partition needs its own
    // buffer — but not a FRESH one each token.
    sel.clear();
    sel.extend_from_slice(probs);
    // k-th largest = (k-1)-th index in a descending partition.
    let (_, kth, _) = sel.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = *kth;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Nucleus: keep the smallest prefix of tokens whose cumulative
/// probability reaches top_p. Only surviving (non-zero) candidates are
/// sorted — after top-k that is ≤ k elements, not the whole vocab; the
/// kept set is marked in-place instead of a per-token HashSet.
fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, p)| p > 0.0)
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero the dropped tail directly — indices, not membership tests.
    for &(i, _) in &indexed[cutoff_idx..] {
        probs[i] = 0.0;
    }
}

/// Inverse-CDF sampling with an externally supplied uniform r ∈ [0, 1).
fn categorical_sample(probs: &[f32], r: f32) -> u32 {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    probs.iter().rposition(|&p| p > 0.0).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    /// The four-lane argmax against the obvious scalar one, including the
    /// tie rule: `max_by` keeps the LAST of several equal maxima, and a
    /// lane split that quietly picked the first would move greedy output on
    /// any model with two equally-likely tokens.
    #[test]
    fn argmax_lanes_match_the_scalar_one_ties_and_all() {
        // The reference IS the old implementation, `max_by` and all — the
        // point is that nothing observable changed, tie rule included.
        let scalar = |v: &[f32]| -> u32 {
            v.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        for n in 0..40usize {
            for seed in 0..8u64 {
                let mut r = super::SplitMix64::new(seed * 7 + n as u64);
                // Quantized to few distinct values on purpose: ties are the
                // case the lanes can get wrong and random floats never hit.
                let v: Vec<f32> = (0..n).map(|_| ((r.next_u64() % 5) as f32) - 2.0).collect();
                assert_eq!(super::argmax(&v), scalar(&v), "n={n} seed={seed} {v:?}");
            }
        }
        let flat = vec![f32::NEG_INFINITY; 13];
        assert_eq!(super::argmax(&flat), scalar(&flat));
    }

    use super::*;

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = SplitMix64::new(1);
        assert_eq!(sample(&logits, &config, &[], &mut rng), 1);
    }

    /// `argmax_penalized` must equal the copy-and-penalize chain's argmax
    /// — same values, same tie rule — with and without the pool.
    #[test]
    fn argmax_penalized_matches_chain_argmax() {
        let pool = crate::pool::Pool::new(3);
        let n = 40_000usize;
        for seed in 0..8u64 {
            let mut r = SplitMix64::new(seed + 3);
            // coarse values so ties happen
            let logits: Vec<f32> = (0..n)
                .map(|_| ((r.next_u64() % 41) as f32 - 20.0) / 4.0)
                .collect();
            let past: Vec<u32> = (0..2000)
                .map(|_| (r.next_u64() % n as u64) as u32)
                .collect();
            for cfg in [
                SamplerConfig {
                    temperature: 0.0,
                    repetition_penalty: 1.1,
                    ..Default::default()
                },
                SamplerConfig {
                    temperature: 0.0,
                    repetition_penalty: 1.0,
                    presence_penalty: 1.5,
                    ..Default::default()
                },
                SamplerConfig {
                    temperature: 0.0,
                    repetition_penalty: 1.3,
                    presence_penalty: 0.7,
                    suppress_tokens: vec![5, 77, 3000],
                    ..Default::default()
                },
            ] {
                let mut s1 = SamplerScratch::default();
                let mut probs = Vec::new();
                chain(&logits, &cfg, &past, &mut s1, None, &mut probs);
                let want = argmax(&probs);
                let mut s2 = SamplerScratch::default();
                let got_serial = argmax_penalized(&logits, &cfg, &past, &mut s2, None);
                let got_pool = argmax_penalized(&logits, &cfg, &past, &mut s2, Some(&pool));
                assert_eq!(want, got_serial, "serial, seed {seed} cfg {cfg:?}");
                assert_eq!(want, got_pool, "pool, seed {seed} cfg {cfg:?}");
            }
        }
    }

    /// The pool chain must sample the SAME token as the serial one on the
    /// same seed — that is the whole contract of the parallel passes.
    #[test]
    fn pool_chain_matches_serial_bit_for_bit() {
        let pool = crate::pool::Pool::new(3);
        let n = 40_000usize; // above PAR_MIN so the pool arm is exercised
        for seed in 0..6u64 {
            let mut r = SplitMix64::new(seed + 11);
            let logits: Vec<f32> = (0..n)
                .map(|i| ((r.next_u64() % 2001) as f32 - 1000.0) / 90.0 + (i % 7) as f32 * 0.01)
                .collect();
            let past: Vec<u32> = (0..500).map(|_| (r.next_u64() % n as u64) as u32).collect();
            for cfg in [
                SamplerConfig::default(),
                SamplerConfig {
                    temperature: 0.7,
                    top_p: 0.8,
                    top_k: 20,
                    min_p: 0.0,
                    presence_penalty: 1.5,
                    repetition_penalty: 1.0,
                    ..Default::default()
                },
                SamplerConfig {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: 20,
                    min_p: 0.0,
                    ..Default::default()
                },
                SamplerConfig {
                    temperature: 1.3,
                    top_p: 1.0,
                    top_k: 0,
                    min_p: 0.02,
                    ..Default::default()
                },
            ] {
                let mut s1 = SamplerScratch::default();
                let mut s2 = SamplerScratch::default();
                for step in 0..5u64 {
                    let mut r1 = SplitMix64::new(seed * 100 + step);
                    let mut r2 = r1.clone();
                    let a = sample_with_scratch(&logits, &cfg, &past, &mut r1, &mut s1);
                    let b = sample_with_scratch_pool(
                        &logits,
                        &cfg,
                        &past,
                        &mut r2,
                        &mut s2,
                        Some(&pool),
                    );
                    assert_eq!(a, b, "seed {seed} step {step} cfg {cfg:?}");
                    // and the working copies agree value for value
                    assert_eq!(s1.probs, s2.probs, "probs differ seed {seed} step {step}");
                }
            }
        }
    }

    /// Speculative sampling must reproduce the TARGET distribution however
    /// good or bad the draft is: draw d ~ q, accept with min(1, p/q), else
    /// correct from max(0, p − q). Empirical law over many trials against
    /// p itself, for a sharp draft, a flat draft and a wrong draft.
    #[test]
    fn spec_accept_or_correct_reproduces_the_target() {
        let n = 40usize;
        let mk = |seed: u64, sharp: f32| -> Vec<f32> {
            let mut r = SplitMix64::new(seed);
            let mut v: Vec<f32> = (0..n)
                .map(|_| ((r.next_u64() % 1000) as f32 / 1000.0).powf(sharp))
                .collect();
            // a few exact zeros, like a top-k'd distribution
            for i in 0..n {
                if (i * 7 + seed as usize) % 5 == 0 {
                    v[i] = 0.0;
                }
            }
            let s: f32 = v.iter().sum();
            v.iter().map(|x| x / s).collect()
        };
        let p = mk(3, 3.0);
        for (qi, q) in [mk(3, 3.0), mk(11, 1.0), mk(29, 6.0)]
            .into_iter()
            .enumerate()
        {
            let mut rng = SplitMix64::new(77 + qi as u64);
            let mut counts = vec![0u64; n];
            let mut scratch = Vec::new();
            let trials = 400_000u64;
            for _ in 0..trials {
                let d = categorical_sample(&q, rng.next_f32());
                let t = match spec_accept_or_correct(&p, &q, d, &mut rng, &mut scratch, None) {
                    None => d,
                    Some(c) => c,
                };
                counts[t as usize] += 1;
            }
            let l1: f64 = (0..n)
                .map(|i| (counts[i] as f64 / trials as f64 - p[i] as f64).abs())
                .sum();
            eprintln!("spec q#{qi}: L1(empirical, p) = {l1:.4}");
            assert!(
                l1 < 0.01,
                "q#{qi}: empirical distribution drifted from p, L1 {l1}"
            );
            // and nothing outside p's support was ever emitted
            for i in 0..n {
                if p[i] == 0.0 {
                    assert_eq!(counts[i], 0, "q#{qi}: token {i} outside p emitted");
                }
            }
        }
    }

    /// The sparse chain is the dense chain: same survivor set, same
    /// probabilities (to fp), serial and pooled, with and without
    /// penalties, min-p, top-p — and a seed draws the same token.
    #[test]
    fn sparse_chain_matches_the_dense_chain() {
        let pool = crate::pool::Pool::new(3);
        let n = 40_000usize; // above PAR_MIN: the pooled arms run
        for seed in 0..5u64 {
            let mut r = SplitMix64::new(100 + seed);
            let logits: Vec<f32> = (0..n)
                .map(|_| ((r.next_u64() % 3000) as f32 - 1500.0) / 120.0)
                .collect();
            let past: Vec<u32> = (0..400).map(|_| (r.next_u64() % n as u64) as u32).collect();
            for cfg in [
                SamplerConfig {
                    temperature: 0.7,
                    top_p: 0.8,
                    top_k: 20,
                    min_p: 0.0,
                    presence_penalty: 1.5,
                    repetition_penalty: 1.0,
                    ..Default::default()
                },
                SamplerConfig {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: 40,
                    min_p: 0.05,
                    presence_penalty: 0.0,
                    repetition_penalty: 1.1,
                    ..Default::default()
                },
                SamplerConfig {
                    temperature: 0.6,
                    top_p: 1.0,
                    top_k: 3,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                    repetition_penalty: 1.0,
                    suppress_tokens: vec![5, 6, 7],
                    ..Default::default()
                },
            ] {
                assert!(sparse_ok(&cfg));
                let mut sd = SamplerScratch::default();
                let mut dense = Vec::new();
                distribution_into(&logits, &cfg, &past, &mut sd, None, &mut dense);
                for pl in [None, Some(&pool)] {
                    let mut ss = SamplerScratch::default();
                    let mut sp = Vec::new();
                    let ok = sparse_distribution_into(&logits, &cfg, &past, &mut ss, pl, &mut sp);
                    assert!(ok, "seed {seed} cfg {cfg:?}");
                    let dense_nz: Vec<(u32, f32)> = dense
                        .iter()
                        .enumerate()
                        .filter(|&(_, &v)| v > 0.0)
                        .map(|(i, &v)| (i as u32, v))
                        .collect();
                    assert_eq!(
                        dense_nz.len(),
                        sp.len(),
                        "seed {seed} pool {} cfg {cfg:?}: support {:?} vs {:?}",
                        pl.is_some(),
                        dense_nz,
                        sp
                    );
                    for (a, b) in dense_nz.iter().zip(sp.iter()) {
                        assert_eq!(a.0, b.0, "seed {seed} cfg {cfg:?}: ids differ");
                        assert!(
                            (a.1 - b.1).abs() <= 2e-5 * a.1.max(1e-3),
                            "seed {seed} cfg {cfg:?}: prob {} vs {}",
                            a.1,
                            b.1
                        );
                    }
                    // the seed lands on the same token (both walk the ids
                    // in order); allow a boundary rounding miss or two
                    let mut agree = 0usize;
                    let trials = 400usize;
                    for k in 0..trials as u64 {
                        let mut r1 = SplitMix64::new(500 + k);
                        let mut r2 = SplitMix64::new(500 + k);
                        let a = categorical_sample(&dense, r1.next_f32());
                        let b = draw_sparse(&sp, &mut r2);
                        agree += (a == b) as usize;
                    }
                    assert!(agree >= trials - 2, "seed {seed} cfg {cfg:?}: agree {agree}/{trials}");
                    // and the public entry uses it
                    let mut r1 = SplitMix64::new(9);
                    let mut r2 = SplitMix64::new(9);
                    let a = categorical_sample(&dense, r1.next_f32());
                    let mut s3 = SamplerScratch::default();
                    let b = sample_with_scratch_pool(&logits, &cfg, &past, &mut r2, &mut s3, pl);
                    assert_eq!(a, b, "seed {seed} cfg {cfg:?}: entry draw");
                }
            }
        }
    }

    /// The sparse accept/correct emits the target distribution, like its
    /// dense twin — the same 400k-trial law test over sparse p and q.
    #[test]
    fn spec_accept_or_correct_sparse_reproduces_the_target() {
        let n = 40usize;
        let mk = |seed: u64, sharp: f32| -> Vec<(u32, f32)> {
            let mut r = SplitMix64::new(seed);
            let mut v: Vec<f32> = (0..n)
                .map(|_| ((r.next_u64() % 1000) as f32 / 1000.0).powf(sharp))
                .collect();
            for i in 0..n {
                if (i * 7 + seed as usize) % 5 == 0 {
                    v[i] = 0.0;
                }
            }
            let s: f32 = v.iter().sum();
            v.iter()
                .enumerate()
                .filter(|&(_, &x)| x > 0.0)
                .map(|(i, &x)| (i as u32, x / s))
                .collect()
        };
        let p = mk(3, 3.0);
        for (qi, q) in [mk(3, 3.0), mk(11, 1.0), mk(29, 6.0)]
            .into_iter()
            .enumerate()
        {
            let mut rng = SplitMix64::new(77 + qi as u64);
            let mut counts = vec![0u64; n];
            let mut res = Vec::new();
            let trials = 400_000u64;
            for _ in 0..trials {
                let d = draw_sparse(&q, &mut rng);
                let t = match spec_accept_or_correct_sparse(&p, &q, d, &mut rng, &mut res) {
                    None => d,
                    Some(c) => c,
                };
                counts[t as usize] += 1;
            }
            let l1: f64 = (0..n)
                .map(|i| (counts[i] as f64 / trials as f64 - sparse_get(&p, i as u32) as f64).abs())
                .sum();
            eprintln!("sparse spec q#{qi}: L1(empirical, p) = {l1:.4}");
            assert!(l1 < 0.01, "q#{qi}: drifted, L1 {l1}");
            for i in 0..n {
                if sparse_get(&p, i as u32) == 0.0 {
                    assert_eq!(counts[i], 0, "q#{qi}: token {i} outside p emitted");
                }
            }
        }
    }

    /// `distribution_into` is the sampler's own chain: drawing from it
    /// with the same uniform lands on the same token as `sample_*`.
    #[test]
    fn distribution_matches_the_sampler_draw() {
        let n = 20_000usize;
        let mut r = SplitMix64::new(9);
        let logits: Vec<f32> = (0..n)
            .map(|_| ((r.next_u64() % 3000) as f32 - 1500.0) / 120.0)
            .collect();
        let past: Vec<u32> = (0..300).map(|_| (r.next_u64() % n as u64) as u32).collect();
        for cfg in [
            SamplerConfig::default(),
            SamplerConfig {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 1.5,
                repetition_penalty: 1.0,
                ..Default::default()
            },
            SamplerConfig {
                temperature: 0.0,
                repetition_penalty: 1.1,
                ..Default::default()
            },
            SamplerConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                ..Default::default()
            },
        ] {
            let mut s1 = SamplerScratch::default();
            let mut s2 = SamplerScratch::default();
            for step in 0..4u64 {
                let mut r1 = SplitMix64::new(step + 1);
                let mut r2 = r1.clone();
                let a = sample_with_scratch(&logits, &cfg, &past, &mut r1, &mut s1);
                let mut dist = Vec::new();
                distribution_into(&logits, &cfg, &past, &mut s2, None, &mut dist);
                assert!(
                    (dist.iter().sum::<f32>() - 1.0).abs() < 1e-3,
                    "not normalized: {}",
                    dist.iter().sum::<f32>()
                );
                let b = if cfg.temperature < 1e-6 {
                    argmax(&dist)
                } else {
                    draw(&dist, &mut r2)
                };
                assert_eq!(a, b, "cfg {cfg:?} step {step}");
            }
        }
    }

    /// The streaming k-th largest against `select_nth_unstable_by` — the
    /// value the old partition returned, ties and zeros included.
    #[test]
    fn kth_largest_equals_select_nth() {
        for seed in 0..20u64 {
            let mut r = SplitMix64::new(seed);
            let n = 50 + (r.next_u64() % 3000) as usize;
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    if r.next_u64() % 3 == 0 {
                        0.0
                    } else {
                        (r.next_u64() % 97) as f32 / 97.0
                    }
                })
                .collect();
            for k in [1usize, 2, 5, 20, 40, n / 2, n - 1] {
                let mut sel = v.clone();
                let (_, kth, _) = sel.select_nth_unstable_by(k - 1, |a, b| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                assert_eq!(kth_largest(&v, k), *kth, "seed {seed} k {k}");
            }
        }
    }

    /// The pooled confidence equals the serial formula, bit for bit.
    #[test]
    fn top1_prob_pool_matches_serial() {
        let pool = crate::pool::Pool::new(2);
        let n = 30_000usize;
        let mut r = SplitMix64::new(5);
        let logits: Vec<f32> = (0..n)
            .map(|_| ((r.next_u64() % 1000) as f32) / 37.0)
            .collect();
        let serial = |id: u32, temp: f32| -> f32 {
            let t = if temp > 1e-3 { temp } else { 1.0 };
            let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let sum: f32 = logits.iter().map(|&v| ((v - max) / t).exp()).sum();
            (((logits[id as usize] - max) / t).exp()) / sum
        };
        let mut sc = SamplerScratch::default();
        for (id, t) in [(3u32, 1.0f32), (777, 0.7), (29_999, 2.0), (12, 0.0)] {
            let a = serial(id, t);
            let b = top1_prob_pool(Some(&pool), &mut sc, &logits, id, t);
            assert_eq!(a.to_bits(), b.to_bits(), "id {id} t {t}: {a} vs {b}");
        }
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(logits[2] > logits[1] && logits[1] > logits[0]);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut scratch = SamplerScratch::default();
        apply_repetition_penalty(&mut logits, &[1, 3], 2.0, &mut scratch);
        assert_eq!(logits, vec![1.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn repetition_penalty_applies_once_per_unique_token() {
        let mut logits = vec![1.0, 4.0, -6.0];
        let mut scratch = SamplerScratch::default();
        apply_repetition_penalty(&mut logits, &[1, 1, 2, 1, 2], 2.0, &mut scratch);
        assert_eq!(logits, vec![1.0, 2.0, -12.0]);
    }

    #[test]
    fn top_k_keeps_exactly_k() {
        let mut probs = vec![0.1, 0.4, 0.05, 0.3, 0.15];
        apply_top_k(&mut probs, 2, &mut Vec::new());
        let kept = probs.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(kept, 2, "top-k must keep exactly k (was k+1 in v1)");
        assert!(probs[1] > 0.0 && probs[3] > 0.0);
    }

    #[test]
    fn rng_reaches_full_cdf() {
        // v1 bug: r < 0.233 always, so the CDF tail was unreachable.
        // With uniform probs the LAST index must be sampled sometimes.
        let probs = vec![0.25f32; 4];
        let mut rng = SplitMix64::new(42);
        let mut hits = [0usize; 4];
        for _ in 0..4000 {
            let i = categorical_sample(&probs, rng.next_f32()) as usize;
            hits[i] += 1;
        }
        for (i, &h) in hits.iter().enumerate() {
            assert!(h > 700, "index {i} sampled only {h}/4000 — biased RNG");
        }
    }

    #[test]
    fn same_seed_same_sequence() {
        let logits: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin()).collect();
        let config = SamplerConfig {
            temperature: 1.0,
            seed: Some(7),
            ..Default::default()
        };
        let run = |seed: u64| -> Vec<u32> {
            let mut rng = SplitMix64::new(seed);
            (0..16)
                .map(|_| sample(&logits, &config, &[], &mut rng))
                .collect()
        };
        assert_eq!(run(7), run(7), "same seed must reproduce");
        assert_ne!(run(7), run(8), "different seed must differ");
    }
}
