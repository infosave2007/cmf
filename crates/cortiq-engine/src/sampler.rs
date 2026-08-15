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
    let mut probs = std::mem::take(&mut scratch.probs);
    let normalized = chain(logits, config, past_tokens, scratch, pool, &mut probs);

    let mut done = |probs: Vec<f32>, tok: u32| -> u32 {
        scratch.probs = probs;
        tok
    };

    if config.temperature < 1e-6 {
        let t = argmax(&probs); // greedy over the penalized logits
        return done(probs, t);
    }
    if !normalized {
        // Everything filtered out — fall back to greedy over original logits.
        let t = argmax(logits);
        return done(probs, t);
    }
    let t = categorical_sample(&probs, rng.next_f32());
    done(probs, t)
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
