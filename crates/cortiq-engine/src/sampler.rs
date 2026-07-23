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
    if config.temperature < 1e-6
        && config.repetition_penalty == 1.0
        && config.suppress_tokens.is_empty()
    {
        return argmax(logits);
    }
    let mut probs = logits.to_vec();

    for &tok in &config.suppress_tokens {
        if (tok as usize) < probs.len() {
            probs[tok as usize] = f32::NEG_INFINITY;
        }
    }

    if config.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut probs, past_tokens, config.repetition_penalty, scratch);
    }

    if config.temperature < 1e-6 {
        return argmax(&probs); // greedy
    }
    if config.temperature != 1.0 {
        for p in probs.iter_mut() {
            *p /= config.temperature;
        }
    }

    softmax_inplace(&mut probs);

    if config.min_p > 0.0 {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = max_prob * config.min_p;
        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    if config.top_k > 0 && (config.top_k as usize) < probs.len() {
        apply_top_k(&mut probs, config.top_k as usize);
    }

    if config.top_p < 1.0 && config.top_p > 0.0 {
        apply_top_p(&mut probs, config.top_p);
    }

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        // Everything filtered out — fall back to greedy over original logits.
        return argmax(logits);
    }

    categorical_sample(&probs, rng.next_f32())
}

/// Greedy: index of the maximum value.
pub fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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
fn apply_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let mut sel: Vec<f32> = probs.to_vec();
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
        apply_top_k(&mut probs, 2);
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
