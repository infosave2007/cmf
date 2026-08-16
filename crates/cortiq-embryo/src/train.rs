//! Data + the step loop for the birth: token shards, batches, schedule,
//! checkpoints. Everything host-side and portable; the GPU work is in
//! `model::EmbryoGpu`.

use std::io::{Read, Write};
use std::path::Path;

/// A flat token stream (u16 little-endian on disk — any tokenizer with a
/// vocabulary ≤ 65536; the Embryo vocab is 32768).
pub struct Shard {
    pub tokens: Vec<u16>,
}

impl Shard {
    pub fn load(path: &Path) -> anyhow::Result<Shard> {
        let mut f = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        anyhow::ensure!(bytes.len() % 2 == 0, "shard {}: odd byte length", path.display());
        let tokens = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Shard { tokens })
    }
    /// Bytes as tokens (vocab 256) — the zero-dependency smoke corpus.
    pub fn from_bytes(text: &[u8]) -> Shard {
        Shard { tokens: text.iter().map(|b| *b as u16).collect() }
    }
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut f = std::fs::File::create(path)?;
        let mut bytes = Vec::with_capacity(self.tokens.len() * 2);
        for t in &self.tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        f.write_all(&bytes)?;
        Ok(())
    }
}

/// Deterministic batch sampler: B random windows of T+1 tokens.
pub struct Sampler {
    pub b: usize,
    pub t: usize,
    state: u64,
}

impl Sampler {
    pub fn new(b: usize, t: usize, seed: u64) -> Sampler {
        Sampler { b, t, state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Fills tokens/targets ([B·T] each) from the shard.
    pub fn batch(&mut self, shard: &Shard, tokens: &mut Vec<u32>, targets: &mut Vec<u32>) {
        let n = shard.tokens.len();
        assert!(n > self.t + 1, "shard shorter than one window");
        tokens.clear();
        targets.clear();
        for _ in 0..self.b {
            let start = (self.next_u64() % (n - self.t - 1) as u64) as usize;
            let w = &shard.tokens[start..start + self.t + 1];
            tokens.extend(w[..self.t].iter().map(|x| *x as u32));
            targets.extend(w[1..].iter().map(|x| *x as u32));
        }
    }
    /// Fixed evenly spaced windows for a deterministic validation set.
    pub fn fixed_batch(shard: &Shard, b: usize, t: usize, index: usize, tokens: &mut Vec<u32>, targets: &mut Vec<u32>) {
        let n = shard.tokens.len();
        tokens.clear();
        targets.clear();
        for i in 0..b {
            let k = index * b + i;
            let start = (k * 7919) % (n - t - 1);
            let w = &shard.tokens[start..start + t + 1];
            tokens.extend(w[..t].iter().map(|x| *x as u32));
            targets.extend(w[1..].iter().map(|x| *x as u32));
        }
    }
}

/// Warmup + cosine learning-rate schedule.
pub fn lr_at(step: usize, total: usize, warmup: usize, peak: f32, floor: f32) -> f32 {
    if step < warmup {
        return peak * (step + 1) as f32 / warmup as f32;
    }
    let p = ((step - warmup) as f32 / (total.saturating_sub(warmup)).max(1) as f32).min(1.0);
    floor + 0.5 * (peak - floor) * (1.0 + (std::f32::consts::PI * p).cos())
}

/// Checkpoint: config JSON + raw f32 params (+ optional m/v) — the plain
/// trainer format; `.cmf` export is the runtime's container.
pub fn save_checkpoint(path: &Path, cfg: &crate::model::EmbryoCfg, step: u32, params: &[f32], m: Option<&[f32]>, v: Option<&[f32]>) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let hdr = serde_json::json!({ "cfg": cfg, "step": step, "n": params.len(), "opt": m.is_some() });
    let hs = serde_json::to_vec(&hdr)?;
    f.write_all(&(hs.len() as u64).to_le_bytes())?;
    f.write_all(&hs)?;
    let w = |f: &mut std::fs::File, x: &[f32]| -> anyhow::Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        f.write_all(bytes)?;
        Ok(())
    };
    w(&mut f, params)?;
    if let (Some(m), Some(v)) = (m, v) {
        w(&mut f, m)?;
        w(&mut f, v)?;
    }
    Ok(())
}

pub struct Checkpoint {
    pub cfg: crate::model::EmbryoCfg,
    pub step: u32,
    pub params: Vec<f32>,
    pub m: Option<Vec<f32>>,
    pub v: Option<Vec<f32>>,
}

pub fn load_checkpoint(path: &Path) -> anyhow::Result<Checkpoint> {
    let mut f = std::fs::File::open(path)?;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8)?;
    let hl = u64::from_le_bytes(b8) as usize;
    let mut hs = vec![0u8; hl];
    f.read_exact(&mut hs)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hs)?;
    let cfg: crate::model::EmbryoCfg = serde_json::from_value(hdr["cfg"].clone())?;
    let step = hdr["step"].as_u64().unwrap_or(0) as u32;
    let n = hdr["n"].as_u64().unwrap_or(0) as usize;
    let opt = hdr["opt"].as_bool().unwrap_or(false);
    let rd = |f: &mut std::fs::File, n: usize| -> anyhow::Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        f.read_exact(&mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let params = rd(&mut f, n)?;
    let (m, v) = if opt { (Some(rd(&mut f, n)?), Some(rd(&mut f, n)?)) } else { (None, None) };
    Ok(Checkpoint { cfg, step, params, m, v })
}
