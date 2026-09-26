//! Signal composition `x = [φ_P ; 0.5·φ_H]` (spec §0, §3.3).
//!
//! * φ_P: the native encoder ([`crate::bert::Encoder`]), `dim_p` values (384
//!   for the release encoder), unit length;
//! * φ_H: `hashfeat::dense(text, 4096)` (the `cortiq-hashfeat-v1` contract);
//! * `x = [φ_P ; 0.5·φ_H]`, `dim_p + 4096` values, not renormalised. The
//!   factor 0.5 is exact in f32, so `x` from the stored rows
//!   ([`crate::rows::Row::signal`]) is bit-identical to `x` from the text.
//!
//! [`SignalEncoder::features_batch`] spreads texts over `threads` scoped
//! threads; every text is encoded on its own (no padding, no shared state), so
//! the result does not depend on the number of threads.

use crate::bert::{Encoder, GoldenReport};
use crate::container::DecisionModel;
use crate::hashfeat;
use crate::rows::{PHI_H_WEIGHT, Row, Source, Split};
use anyhow::{Result, ensure};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// φ_H dimension (`cortiq-hashfeat-v1`).
pub const PHI_H_DIM: usize = hashfeat::DIM;

/// φ_H of a text: `hashfeat::dense(text, 4096)`.
pub fn phi_h(text: &str) -> Vec<f32> {
    hashfeat::dense(text, PHI_H_DIM)
}

/// `out = [φ_P ; 0.5·φ_H]` (`out.len() == phi_p.len() + phi_h.len()`).
pub fn compose_into(phi_p: &[f32], phi_h: &[f32], out: &mut [f32]) {
    let dp = phi_p.len();
    assert_eq!(
        out.len(),
        dp + phi_h.len(),
        "signal buffer holds {} values, [φ_P ; φ_H] has {}",
        out.len(),
        dp + phi_h.len()
    );
    out[..dp].copy_from_slice(phi_p);
    for (o, &h) in out[dp..].iter_mut().zip(phi_h) {
        *o = PHI_H_WEIGHT * h;
    }
}

/// `[φ_P ; 0.5·φ_H]`.
pub fn compose(phi_p: &[f32], phi_h: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; phi_p.len() + phi_h.len()];
    compose_into(phi_p, phi_h, &mut out);
    out
}

/// φ_P and the dense φ_H of one text.
#[derive(Clone, Debug, PartialEq)]
pub struct Features {
    pub phi_p: Vec<f32>,
    /// Dense, [`PHI_H_DIM`] values.
    pub phi_h: Vec<f32>,
}

impl Features {
    /// `dim_p + dim_h`.
    pub fn dim(&self) -> usize {
        self.phi_p.len() + self.phi_h.len()
    }

    /// `[φ_P ; 0.5·φ_H]` into `out`.
    pub fn signal_into(&self, out: &mut [f32]) {
        compose_into(&self.phi_p, &self.phi_h, out);
    }

    /// `[φ_P ; 0.5·φ_H]`.
    pub fn signal(&self) -> Vec<f32> {
        compose(&self.phi_p, &self.phi_h)
    }

    /// A `cortiq-decision-rows-v1` row of these features (φ_H stored sparse).
    pub fn to_row(&self, task: u32, split: Split, flags: u8, source: Source, weight: f32) -> Row {
        Row::from_dense(
            task,
            split,
            flags,
            source,
            weight,
            self.phi_p.clone(),
            &self.phi_h,
        )
    }

    /// The features a stored row holds (its dense φ_H).
    pub fn from_row(row: &Row, dim_h: usize) -> Self {
        Self {
            phi_p: row.phi_p.clone(),
            phi_h: row.phi_h_dense(dim_h),
        }
    }

    /// Bit-for-bit equality of φ_P and of the φ_H a row stores (the sparse
    /// form drops zeros, so this compares the dense vectors).
    pub fn bit_eq(&self, other: &Self) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        eq(&self.phi_p, &other.phi_p) && eq(&self.phi_h, &other.phi_h)
    }
}

/// Wall time of the stages of one text (spec §6.6).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// WordPiece.
    pub tokenize: Duration,
    /// BERT forward, pooling and both L2 steps.
    pub encode: Duration,
    /// φ_H.
    pub hash: Duration,
    /// Tokens of the text, `[CLS]` and `[SEP]` included.
    pub tokens: usize,
}

/// The text → signal path of a decision file: encoder + hashing contract.
#[derive(Clone, Debug)]
pub struct SignalEncoder {
    encoder: Encoder,
}

impl SignalEncoder {
    pub fn new(encoder: Encoder) -> Self {
        Self { encoder }
    }

    /// The encoder of a decision file after the golden check (spec §2.8): the
    /// φ_P of the recorded golden texts must match `decision.encoder.golden`
    /// ([`crate::bert::GOLDEN_MAX_ABS`]).
    pub fn from_model(model: &DecisionModel) -> Result<(Self, GoldenReport)> {
        let encoder = Encoder::from_model(model)?;
        ensure!(
            encoder.dim() == model.encoder_dim(),
            "encoder dim {} differs from the representation's {}",
            encoder.dim(),
            model.encoder_dim()
        );
        ensure!(
            model.hashing_dim() == PHI_H_DIM,
            "hashing dim {} is not {PHI_H_DIM}",
            model.hashing_dim()
        );
        let report = encoder.verify_golden(model)?;
        Ok((Self { encoder }, report))
    }

    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// φ_P dimension.
    pub fn dim_p(&self) -> usize {
        self.encoder.dim()
    }

    /// φ_H dimension.
    pub fn dim_h(&self) -> usize {
        PHI_H_DIM
    }

    /// Signal dimension `dim_p + dim_h`.
    pub fn dim(&self) -> usize {
        self.dim_p() + PHI_H_DIM
    }

    /// φ_P and φ_H of a text.
    pub fn features(&self, text: &str) -> Features {
        Features {
            phi_p: self.encoder.encode(text),
            phi_h: phi_h(text),
        }
    }

    /// [`SignalEncoder::features`] with the time of each stage.
    pub fn features_timed(&self, text: &str) -> (Features, Timings) {
        let t0 = Instant::now();
        let ids = self.encoder.tokenize(text);
        let t1 = Instant::now();
        let phi_p = self.encoder.embed_ids(&ids);
        let t2 = Instant::now();
        let phi_h = phi_h(text);
        let t3 = Instant::now();
        (
            Features { phi_p, phi_h },
            Timings {
                tokenize: t1 - t0,
                encode: t2 - t1,
                hash: t3 - t2,
                tokens: ids.len(),
            },
        )
    }

    /// `x = [φ_P ; 0.5·φ_H]` of a text.
    pub fn signal(&self, text: &str) -> Vec<f32> {
        self.features(text).signal()
    }

    /// Features of every text, in input order, on up to `threads` scoped
    /// threads (0 or 1: the calling thread). The result does not depend on
    /// `threads`.
    pub fn features_batch<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        threads: usize,
    ) -> Vec<Features> {
        let threads = threads.clamp(1, texts.len().max(1));
        if threads == 1 {
            return texts.iter().map(|t| self.features(t.as_ref())).collect();
        }
        let next = AtomicUsize::new(0);
        let mut parts: Vec<Vec<(usize, Features)>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= texts.len() {
                                break;
                            }
                            out.push((i, self.features(texts[i].as_ref())));
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("encoder thread panicked"))
                .collect()
        });
        let mut slots: Vec<Option<Features>> = vec![None; texts.len()];
        for part in parts.iter_mut() {
            for (i, f) in part.drain(..) {
                slots[i] = Some(f);
            }
        }
        slots
            .into_iter()
            .map(|f| f.expect("every text encoded"))
            .collect()
    }

    /// Signals of every text, row-major `[texts.len(), dim]`.
    pub fn signals_batch<S: AsRef<str> + Sync>(&self, texts: &[S], threads: usize) -> Vec<f32> {
        let dim = self.dim();
        let mut out = vec![0.0f32; texts.len() * dim];
        for (f, row) in self
            .features_batch(texts, threads)
            .iter()
            .zip(out.chunks_exact_mut(dim))
        {
            f.signal_into(row);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_is_exact_half() {
        let p = [0.6f32, 0.8];
        let h = [0.0f32, 1.0, -0.25, 3.0e-8];
        let x = compose(&p, &h);
        assert_eq!(x, vec![0.6, 0.8, 0.0, 0.5, -0.125, 1.5e-8]);
        let f = Features {
            phi_p: p.to_vec(),
            phi_h: h.to_vec(),
        };
        let row = f.to_row(
            3,
            Split::Train,
            crate::rows::FLAG_ODD_HALF,
            Source::Data,
            1.0,
        );
        // The stored row's signal equals the signal of the features, bit for bit.
        let xs = row.signal(h.len());
        assert!(x.iter().zip(&xs).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(Features::from_row(&row, h.len()).bit_eq(&f));
    }

    #[test]
    fn phi_h_is_the_contract() {
        let t = "How do I top up my card?";
        assert_eq!(phi_h(t), hashfeat::dense(t, hashfeat::DIM));
        assert_eq!(phi_h(t).len(), 4096);
    }
}
