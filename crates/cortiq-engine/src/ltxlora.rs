//! Runtime LoRA for the LTX-2.5 DiT, and the reference-slot embedding the
//! multi-subject adapters ship alongside it.
//!
//! The container's weights are q4tp — four bits on a per-row scale ladder —
//! so a low-rank update cannot be folded into them without dequantizing the
//! whole DiT and requantizing it, which would cost the file's size in RAM and
//! throw away the codec's error budget on an update the size of a rounding
//! step. The branch is evaluated instead:
//!
//! ```text
//! y = x·Wᵀ + s · (x·Aᵀ)·Bᵀ
//! ```
//!
//! `A` is `[rank, in]` and `B` is `[out, rank]`, which is the layout
//! `lora_A.weight` / `lora_B.weight` already have. At rank 128 against a
//! 4096×4096 projection that is 2·4096·128 multiply-adds against the base's
//! 16.8 M — about 6% — so the branch costs a few per cent of a step and no
//! memory beyond the adapter itself.
//!
//! `SlotEmbed` is the second half of the multi-subject adapters: a Fourier
//! feature of the slot index through a two-layer MLP, added to the *latent*
//! channels of a reference image before it is prepended to the sequence. See
//! [`crate::ltxpipe::Conditioning::with_references`] for where the tokens go.

use crate::ltxdit::{Shared, rows};
use crate::pool::Pool;
use std::collections::HashMap;
use std::path::Path;

/// One projection's low-rank branch, already scaled.
pub struct LoraBranch {
    a: Vec<f32>, // [rank, in]
    b: Vec<f32>, // [out, rank]
    rank: usize,
    inn: usize,
    out: usize,
    scale: f32,
    /// Stable for this branch's lifetime — keys its device-resident copy so
    /// the adapter uploads once per render rather than once per step.
    id: usize,
    /// The router's state: how loudly this branch spoke on the first step
    /// it ran, and whether it still runs. See [`route_threshold`].
    resonance: std::sync::atomic::AtomicU32,
    live: std::sync::atomic::AtomicBool,
}

/// `CMF_LORA_ROUTE=<r>` turns on branch routing: a branch whose first-step
/// contribution `‖s·ΔY‖ / ‖Y‖` is below `r` is switched off for the rest of
/// the render, and the projection it sits on goes back to the fused device
/// path it would have taken without an adapter.
///
/// The point is not the branch's own arithmetic — that is a few per cent —
/// but what carrying one *costs the base*: a branch has to read the panel
/// its projection produces, so the fused kernels that keep qkv, the
/// attention output and the FFN's middle on the card have to stand down and
/// ship those panels home. Turning off the branches that do not matter buys
/// that fusion back for most of the blocks.
///
/// Adapters are not uniform across depth — measured on
/// `fal/MiniMax-H3-Realism-People-LoRA`, the numbers are in `docs/LORA.md`.
pub fn route_threshold() -> Option<f32> {
    use std::sync::atomic::Ordering::Relaxed;
    // `u32::MAX` is "not read yet"; a stored 0 means routing is off. Held
    // as an atomic rather than a `OnceLock` so a caller — or a test — can
    // set it after the first read.
    let mut bits = ROUTE.load(Relaxed);
    if bits == u32::MAX {
        bits = std::env::var("CMF_LORA_ROUTE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v > 0.0)
            .map_or(0, f32::to_bits);
        ROUTE.store(bits, Relaxed);
    }
    (bits != 0).then(|| f32::from_bits(bits))
}

static ROUTE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Set the routing threshold programmatically, overriding the environment.
pub fn set_route_threshold(t: Option<f32>) {
    ROUTE.store(
        t.filter(|v| *v > 0.0).map_or(0, f32::to_bits),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Whether anything wants a branch MEASURED this render — the probe's report
/// or the router's threshold. A fused base+branch kernel has to stand down
/// when this is true, or the measurement never happens.
pub fn wants_measurement() -> bool {
    probe_on() || route_threshold().is_some()
}

/// Whether the per-branch report is wanted (`CMF_LORA_PROBE=1`).
pub fn probe_report_on() -> bool {
    probe_on()
}

fn probe_on() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("CMF_LORA_PROBE").is_ok())
}

static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

impl LoraBranch {
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The pieces the fused device path needs.
    #[cfg(target_os = "macos")]
    pub(crate) fn side(&self) -> crate::gpu_metal::LoraSide<'_> {
        crate::gpu_metal::LoraSide {
            a: &self.a,
            b: &self.b,
            rank: self.rank,
            scale: self.scale,
            id: self.id,
        }
    }

    /// Whether the router still lets this branch run. The fused device
    /// paths ask this, not `is_some`: a branch switched off must give the
    /// fusion back, or routing saves the arithmetic and keeps the cost.
    pub fn live(&self) -> bool {
        self.live.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The measured first-step contribution, once it has run.
    pub fn resonance(&self) -> f32 {
        f32::from_bits(self.resonance.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// `dst += scale · (x·Aᵀ)·Bᵀ`, over `n` rows of `x`.
    ///
    /// Both halves are N·Kᵀ products, which is the shape `gemm_nt` is built
    /// for — Accelerate's AMX path on Apple silicon, register-blocked SIMD
    /// elsewhere, and the wgpu arm when the shape earns it. Written as two
    /// scalar loops instead, rank 128 over 480 branches cost 40 s a step on
    /// an M4 against the base GEMMs' 9; the arithmetic is small only if it
    /// runs on the same machinery as everything else.
    pub fn add(&self, x: &[f32], n: usize, dst: &mut [f32], pool: Option<&Pool>) {
        debug_assert_eq!(x.len(), n * self.inn);
        debug_assert_eq!(dst.len(), n * self.out);
        if n == 0 || !self.live() {
            return;
        }
        // Held on the host on purpose. These are small f32 GEMMs standing
        // beside a q4tp GEMM that already owns the device, and the generic
        // `GemmNt` probe will happily send them there — where they queue
        // behind the base projection and pay a submit and a readback each.
        // Measured on an M4, 384 tokens: 39.7 s a step through the probe
        // against 12.4 pinned to the host, for GEMMs whose own arithmetic is
        // 1.1 s of that. Accelerate runs these shapes at 250-1300 GFLOP/s.
        let mut h = vec![0f32; n * self.rank];
        let mut d = vec![0f32; n * self.out];
        crate::gpu::cpu_scope(|| {
            crate::fcd_ops::gemm_nt(x, &self.a, &mut h, n, self.inn, self.rank, pool);
            crate::fcd_ops::gemm_nt(&h, &self.b, &mut d, n, self.rank, self.out, pool);
        });
        let scale = self.scale;
        // The router's measurement, and the probe's: the branch against
        // the base it is correcting, both as sums of squares over the whole
        // panel. Taken BEFORE the add, so `dst` is still the base alone.
        // One pass over two panels once per render — the branch's own GEMMs
        // are two orders of magnitude more work.
        let measure = (probe_on() || route_threshold().is_some())
            && self.resonance.load(std::sync::atomic::Ordering::Relaxed) == 0;
        if measure {
            let (mut dd, mut bb) = (0f64, 0f64);
            for (&dv, &bv) in d.iter().zip(dst.iter()) {
                dd += (scale * dv) as f64 * (scale * dv) as f64;
                bb += bv as f64 * bv as f64;
            }
            let r = if bb > 0.0 {
                (dd / bb).sqrt() as f32
            } else {
                f32::INFINITY
            };
            // A zero ratio would read as "not measured yet" on the next
            // step; the smallest positive float says "measured, silent".
            self.resonance.store(
                r.max(f32::MIN_POSITIVE).to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            if let Some(t) = route_threshold() {
                if r < t {
                    self.live.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        let sink = Shared(dst.as_mut_ptr());
        rows(pool, n, &|s, e| {
            let row = unsafe { sink.at(s * self.out, (e - s) * self.out) };
            for (o, v) in row.iter_mut().enumerate() {
                *v += scale * d[s * self.out + o];
            }
        });
    }
}

/// The learned per-reference slot embedding: `slot_id` through Fourier
/// features and a two-layer SiLU MLP, landing on the latent's channel count.
///
/// Reproduces the adapter's own definition exactly — the index is divided by
/// 16 before the phases are taken, and the raw scaled value is the first
/// feature, so the input width is `1 + 2·num_frequencies`.
pub struct SlotEmbed {
    freqs: Vec<f32>,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    hidden: usize,
    dim: usize,
}

impl SlotEmbed {
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The embedding for slot `slot_id` (1-based, as the adapter numbers them).
    pub fn embed(&self, slot_id: usize) -> Vec<f32> {
        let scaled = slot_id as f32 / 16.0;
        let mut feat = Vec::with_capacity(1 + 2 * self.freqs.len());
        feat.push(scaled);
        for f in &self.freqs {
            feat.push((scaled * f).sin());
        }
        for f in &self.freqs {
            feat.push((scaled * f).cos());
        }
        let win = feat.len();
        let mut hid = vec![0f32; self.hidden];
        for (o, hv) in hid.iter_mut().enumerate() {
            let row = &self.w0[o * win..(o + 1) * win];
            let mut acc = self.b0[o];
            for (fv, wv) in feat.iter().zip(row) {
                acc += fv * wv;
            }
            // SiLU
            *hv = acc / (1.0 + (-acc).exp());
        }
        let mut outv = vec![0f32; self.dim];
        for (o, ov) in outv.iter_mut().enumerate() {
            let row = &self.w2[o * self.hidden..(o + 1) * self.hidden];
            let mut acc = self.b2[o];
            for (hv, wv) in hid.iter().zip(row) {
                acc += hv * wv;
            }
            *ov = acc;
        }
        outv
    }
}

/// Every branch an adapter file carries, keyed by the projection it belongs
/// to — the container's name with the `dit.` prefix dropped, which is what
/// the adapters use after their own `diffusion_model.`.
pub struct LoraBank {
    pairs: HashMap<String, (Vec<f32>, Vec<f32>, usize, usize, usize)>,
    pub slot: Option<SlotEmbed>,
    pub meta: HashMap<String, String>,
    scale: f32,
}

/// Read a `.safetensors` into `{name: (shape, f32 data)}`. Shared with the
/// latent upscaler next door, which loads its weights the same way.
pub(crate) fn read_safetensors(
    path: &Path,
) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, String> {
    st_read(path).map(|(t, _)| t)
}

fn st_read(path: &Path) -> Result<(HashMap<String, (Vec<usize>, Vec<f32>)>, HashMap<String, String>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err("lora: truncated safetensors header".into());
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(
        bytes.get(8..8 + hlen).ok_or("lora: header past end of file")?,
    )
    .map_err(|e| format!("lora header: {e}"))?;
    let base = 8 + hlen;
    let obj = header.as_object().ok_or("lora: header not an object")?;
    let mut meta = HashMap::new();
    let mut out = HashMap::new();
    for (name, m) in obj {
        if name == "__metadata__" {
            if let Some(o) = m.as_object() {
                for (k, v) in o {
                    if let Some(s) = v.as_str() {
                        meta.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }
        let dtype = m["dtype"].as_str().ok_or("lora: dtype")?;
        let shape: Vec<usize> = m["shape"]
            .as_array()
            .ok_or("lora: shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = m["data_offsets"].as_array().ok_or("lora: offsets")?;
        let s = offs[0].as_u64().unwrap_or(0) as usize + base;
        let e = offs[1].as_u64().unwrap_or(0) as usize + base;
        let raw = bytes.get(s..e).ok_or("lora: tensor span past end of file")?;
        let mut data = Vec::with_capacity(shape.iter().product::<usize>().max(1));
        match dtype {
            "F32" => {
                for c in raw.chunks_exact(4) {
                    data.push(f32::from_le_bytes(c.try_into().unwrap()));
                }
            }
            "F16" => {
                for c in raw.chunks_exact(2) {
                    data.push(cortiq_core::quant::f16_to_f32(u16::from_le_bytes(
                        c.try_into().unwrap(),
                    )));
                }
            }
            "BF16" => {
                for c in raw.chunks_exact(2) {
                    let b = u16::from_le_bytes(c.try_into().unwrap());
                    data.push(f32::from_bits((b as u32) << 16));
                }
            }
            other => return Err(format!("lora: unsupported dtype {other} on {name}")),
        }
        out.insert(name.clone(), (shape, data));
    }
    Ok((out, meta))
}

impl LoraBank {
    /// Read an adapter. `scale` multiplies every branch — the strength dial.
    ///
    /// `alpha` is honoured when the file records it: the trained convention is
    /// `scale = strength · alpha / rank`, and an adapter that ships neither
    /// `alpha` nor `lora_alpha` is taken at `strength` as-is, which is what
    /// the diffusers loaders do for a file whose A/B are already scaled.
    pub fn load(path: &Path, strength: f32) -> Result<LoraBank, String> {
        let (tensors, meta) = st_read(path)?;
        let mut a_side: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        let mut b_side: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        let mut slot_parts: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, val) in tensors {
            // Three conventions in the wild for the same thing: the
            // ComfyUI single-file adapters write `diffusion_model.`, the
            // PEFT ones `base_model.model.` (keeping the module path,
            // `dit.` included, behind it), and a few write the module
            // path bare. Strip whichever is there and key on the rest.
            let short = name
                .strip_prefix("diffusion_model.")
                .or_else(|| name.strip_prefix("base_model.model."))
                .or_else(|| name.strip_prefix("transformer."))
                .unwrap_or(&name);
            // PEFT keeps the module path after its own prefix, `dit.` and
            // all; the container's names carry the same `dit.` and the
            // lookup strips it. Normalize to the same side of that prefix
            // here, or a PEFT adapter binds nothing at all.
            let short = short.strip_prefix("dit.").unwrap_or(short).to_string();
            if let Some(rest) = short.strip_prefix("reference_slot_embedding.") {
                slot_parts.insert(rest.to_string(), val);
            } else if let Some(base) = short.strip_suffix(".lora_A.weight") {
                a_side.insert(base.to_string(), val);
            } else if let Some(base) = short.strip_suffix(".lora_B.weight") {
                b_side.insert(base.to_string(), val);
            } else if let Some(base) = short.strip_suffix(".lora_down.weight") {
                a_side.insert(base.to_string(), val);
            } else if let Some(base) = short.strip_suffix(".lora_up.weight") {
                b_side.insert(base.to_string(), val);
            }
        }
        let alpha = meta
            .get("alpha")
            .or_else(|| meta.get("lora_alpha"))
            .and_then(|v| v.parse::<f32>().ok());

        let mut pairs = HashMap::new();
        for (base, (ashape, adata)) in a_side {
            let Some((bshape, bdata)) = b_side.remove(&base) else {
                return Err(format!("lora: {base} has an A side and no B side"));
            };
            if ashape.len() != 2 || bshape.len() != 2 {
                return Err(format!("lora: {base} is not a matrix pair"));
            }
            let (rank, inn) = (ashape[0], ashape[1]);
            let (out, rank_b) = (bshape[0], bshape[1]);
            if rank != rank_b {
                return Err(format!(
                    "lora: {base} rank mismatch — A is {rank}, B is {rank_b}"
                ));
            }
            let scale = match alpha {
                Some(al) if rank > 0 => strength * al / rank as f32,
                _ => strength,
            };
            pairs.insert(base, (adata, bdata, rank, inn, out));
            let _ = scale; // per-branch scale is uniform; kept on the bank
        }
        if !b_side.is_empty() {
            let orphan = b_side.keys().next().cloned().unwrap_or_default();
            return Err(format!("lora: {orphan} has a B side and no A side"));
        }

        // The slot embedding is present only on the multi-reference adapters.
        // Half of it is not a thing we can guess at: refuse a partial one
        // rather than silently render without the reference conditioning.
        let slot = if slot_parts.is_empty() {
            None
        } else {
            let need = |k: &str| -> Result<&(Vec<usize>, Vec<f32>), String> {
                slot_parts
                    .get(k)
                    .ok_or_else(|| format!("lora: reference_slot_embedding.{k} is missing"))
            };
            let freqs = need("frequencies")?.1.clone();
            let (s0, w0) = need("net.0.weight").map(|t| (t.0.clone(), t.1.clone()))?;
            let b0 = need("net.0.bias")?.1.clone();
            let (s2, w2) = need("net.2.weight").map(|t| (t.0.clone(), t.1.clone()))?;
            let b2 = need("net.2.bias")?.1.clone();
            if s0.len() != 2 || s2.len() != 2 {
                return Err("lora: slot embedding layers are not matrices".into());
            }
            if s0[1] != 1 + 2 * freqs.len() {
                return Err(format!(
                    "lora: slot embedding takes {} features, {} frequencies imply {}",
                    s0[1],
                    freqs.len(),
                    1 + 2 * freqs.len()
                ));
            }
            Some(SlotEmbed {
                freqs,
                w0,
                b0,
                w2,
                b2,
                hidden: s0[0],
                dim: s2[0],
            })
        };

        let scale = match alpha {
            Some(al) => {
                let r = pairs.values().next().map(|p| p.2).unwrap_or(1).max(1);
                strength * al / r as f32
            }
            None => strength,
        };
        Ok(LoraBank { pairs, slot, meta, scale })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The rank the file was trained at, for the log line.
    pub fn rank(&self) -> usize {
        self.pairs.values().next().map(|p| p.2).unwrap_or(0)
    }

    /// Every projection this adapter names. The caller binds what its
    /// container has and reports the rest: an adapter that trained a
    /// module we fold away (adaLN, on a curve-form pack) must say so
    /// rather than render as though it had been applied.
    pub fn keys(&self) -> Vec<&str> {
        self.pairs.keys().map(|s| s.as_str()).collect()
    }

    /// The branch for a projection of a KNOWN shape. A branch whose matrices
    /// do not match the panel it will be handed belongs to another model —
    /// `add` writes `n·out` floats through a raw pointer, so a mismatch is a
    /// buffer overrun, not a bad picture. Refuse it by name instead.
    pub fn branch_for(
        &self,
        name: &str,
        out: usize,
        inn: usize,
    ) -> Result<Option<LoraBranch>, String> {
        let Some(br) = self.branch(name) else {
            return Ok(None);
        };
        if br.out != out || br.inn != inn {
            return Err(format!(
                "lora: {name} is [{}, {}] in the adapter and [{out}, {inn}] in this container \
                 — that adapter was trained for a different model",
                br.out, br.inn
            ));
        }
        Ok(Some(br))
    }

    /// The branch for a container tensor name, if this adapter carries one.
    /// `name` is the projection without `.weight` — `dit.transformer_blocks.0.attn1.to_q`.
    pub fn branch(&self, name: &str) -> Option<LoraBranch> {
        let key = name.strip_prefix("dit.").unwrap_or(name);
        let (a, b, rank, inn, out) = self.pairs.get(key)?;
        Some(LoraBranch {
            a: a.clone(),
            b: b.clone(),
            rank: *rank,
            inn: *inn,
            out: *out,
            scale: self.scale,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            resonance: Default::default(),
            live: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// How the adapter wants reference tokens placed. Both are refused rather
    /// than approximated when they are not what this implementation does.
    pub fn check_reference_convention(&self) -> Result<(), String> {
        if let Some(order) = self.meta.get("reference_token_order") {
            if order != "prepend" {
                return Err(format!(
                    "lora: reference_token_order={order}, this build only prepends"
                ));
            }
        }
        if let Some(off) = self.meta.get("reference_slot_time_offsets") {
            if off != "pic1_based_negative_time" {
                return Err(format!(
                    "lora: reference_slot_time_offsets={off}, this build only places \
                     references at negative latent frames"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The branch must equal a plain dense evaluation of `s·(xAᵀ)Bᵀ`.
    #[test]
    fn branch_matches_dense() {
        let (n, inn, rank, out) = (3usize, 5usize, 2usize, 4usize);
        let a: Vec<f32> = (0..rank * inn).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..out * rank).map(|i| (i as f32 * 0.11).cos()).collect();
        let x: Vec<f32> = (0..n * inn).map(|i| (i as f32 * 0.7).sin()).collect();
        let br = LoraBranch {
            a: a.clone(),
            b: b.clone(),
            rank,
            inn,
            out,
            scale: 0.5,
            id: 0,
            resonance: Default::default(),
            live: std::sync::atomic::AtomicBool::new(true),
        };
        let mut got = vec![1.5f32; n * out];
        br.add(&x, n, &mut got, None);
        for t in 0..n {
            for o in 0..out {
                let mut acc = 0f32;
                for r in 0..rank {
                    let h: f32 = (0..inn).map(|i| x[t * inn + i] * a[r * inn + i]).sum();
                    acc += h * b[o * rank + r];
                }
                let want = 1.5 + 0.5 * acc;
                assert!(
                    (got[t * out + o] - want).abs() < 1e-4,
                    "row {t} col {o}: {} vs {want}",
                    got[t * out + o]
                );
            }
        }
    }

    /// Write a minimal safetensors carrying one A/B pair per name.
    fn write_pairs(path: &std::path::Path, pairs: &[(&str, usize, usize)]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (base, inn, rank) in pairs {
            for (suffix, shape) in [
                ("lora_A.weight", vec![*rank, *inn]),
                ("lora_B.weight", vec![*inn, *rank]),
            ] {
                let count: usize = shape.iter().product();
                let start = blob.len();
                for i in 0..count {
                    blob.extend_from_slice(&(i as f32 * 0.25).to_le_bytes());
                }
                header.insert(
                    format!("{base}.{suffix}"),
                    serde_json::json!({
                        "dtype": "F32",
                        "shape": shape,
                        "data_offsets": [start, blob.len()],
                    }),
                );
            }
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hdr).unwrap();
        f.write_all(&blob).unwrap();
    }

    /// Adapter names for the video DiT next door must land on ITS
    /// projections too: the H3 adapters write `diffusion_model.blocks.N.…`
    /// and the PEFT ones `base_model.model.dit.blocks.N.…`, and both have
    /// to key on what `mmh3` asks for — `dit.blocks.N.attn.qkv_proj`.
    #[test]
    fn mmh3_names_bind_to_container_projections() {
        let dir = std::env::temp_dir().join(format!("mmh3lora{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("h3.safetensors");
        write_pairs(
            &path,
            &[
                ("diffusion_model.blocks.0.attn.qkv_proj", 3, 2),
                ("base_model.model.dit.blocks.1.mlp.fc1", 3, 2),
            ],
        );
        let bank = LoraBank::load(&path, 1.0).unwrap();
        assert!(bank.branch("dit.blocks.0.attn.qkv_proj").is_some());
        assert!(bank.branch("dit.blocks.1.mlp.fc1").is_some());
        assert!(bank.branch("dit.blocks.2.mlp.fc1").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An adapter for another model must be refused by name. The branch
    /// writes `n·out` floats through a raw pointer, so a shape it was not
    /// trained for is a buffer overrun, not a bad picture.
    #[test]
    fn a_branch_of_the_wrong_shape_is_refused() {
        let dir = std::env::temp_dir().join(format!("lorashape{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wrong.safetensors");
        // A [2, 3] and B [3, 2]: a branch from 3 inputs to 3 outputs.
        write_pairs(&path, &[("diffusion_model.blocks.0.attn.qkv_proj", 3, 2)]);
        let bank = LoraBank::load(&path, 1.0).unwrap();
        let key = "dit.blocks.0.attn.qkv_proj";
        assert!(
            matches!(bank.branch_for(key, 3, 3), Ok(Some(_))),
            "its own shape binds"
        );
        let err = match bank.branch_for(key, 4096, 3) {
            Err(e) => e,
            Ok(_) => panic!("a [4096, 3] projection must not take a [3, 3] branch"),
        };
        assert!(err.contains("different model"), "{err}");
        assert!(matches!(bank.branch_for(key, 3, 4096), Err(_)));
        assert!(matches!(
            bank.branch_for("dit.blocks.9.attn.qkv_proj", 3, 3),
            Ok(None)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The router silences a branch that contributes nothing and leaves a
    /// loud one alone — measured on the first call, applied from the second.
    #[test]
    fn router_silences_a_quiet_branch() {
        set_route_threshold(Some(0.01));
        let quiet = LoraBranch {
            a: vec![1.0; 2],
            b: vec![1e-6; 2],
            rank: 1,
            inn: 2,
            out: 2,
            scale: 1.0,
            id: 0,
            resonance: Default::default(),
            live: std::sync::atomic::AtomicBool::new(true),
        };
        let mut out = vec![1.0f32; 2];
        quiet.add(&[1.0, 1.0], 1, &mut out, None);
        assert!(!quiet.live(), "a 1e-6 branch on a unit base must be routed off");
        let loud = LoraBranch {
            a: vec![1.0; 2],
            b: vec![1.0; 2],
            rank: 1,
            inn: 2,
            out: 2,
            scale: 1.0,
            id: 0,
            resonance: Default::default(),
            live: std::sync::atomic::AtomicBool::new(true),
        };
        let mut out = vec![1.0f32; 2];
        loud.add(&[1.0, 1.0], 1, &mut out, None);
        assert!(loud.live(), "a branch the size of the base must stay");
        assert!(loud.resonance() > 1.0);
        set_route_threshold(None);
    }

    /// A zero-rank-B adapter must be the identity on the output.
    #[test]
    fn zero_b_changes_nothing() {
        let br = LoraBranch {
            a: vec![1.0; 4],
            b: vec![0.0; 6],
            rank: 2,
            inn: 2,
            out: 3,
            scale: 1.0,
            id: 0,
            resonance: Default::default(),
            live: std::sync::atomic::AtomicBool::new(true),
        };
        let mut out = vec![7.0f32; 3];
        br.add(&[1.0, 2.0], 1, &mut out, None);
        assert_eq!(out, vec![7.0, 7.0, 7.0]);
    }

    /// A file's names must land on the container's projections: the adapters
    /// write `diffusion_model.transformer_blocks.N.attn1.to_q.lora_A.weight`
    /// and the container calls that tensor
    /// `dit.transformer_blocks.N.attn1.to_q.weight`, so the bank has to strip
    /// one prefix and the lookup the other.
    #[test]
    fn names_bind_to_container_projections() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("cmf_lora_name_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        // rank 2, in 3, out 4 — one pair, plus a slot embedding
        let names = [
            ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", vec![2usize, 3]),
            ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", vec![4, 2]),
        ];
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (n, shape) in &names {
            let count: usize = shape.iter().product();
            let start = blob.len();
            for i in 0..count {
                blob.extend_from_slice(&(i as f32 * 0.25).to_le_bytes());
            }
            header.insert(
                (*n).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, blob.len()],
                }),
            );
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hdr).unwrap();
        f.write_all(&blob).unwrap();
        drop(f);

        let bank = LoraBank::load(&path, 1.0).expect("load");
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.rank(), 2);
        assert!(bank.slot.is_none());
        let br = bank
            .branch("dit.transformer_blocks.0.attn1.to_q")
            .expect("the container's name must find the adapter's branch");
        assert_eq!(br.rank(), 2);
        assert!(bank.branch("dit.transformer_blocks.0.attn1.to_k").is_none());
        // and the branch computes: x = [1,0,0] picks A's first column
        let mut out = vec![0f32; 4];
        br.add(&[1.0, 0.0, 0.0], 1, &mut out, None);
        // A = [[0,.25,.5],[.75,1,1.25]] so h = [0, .75]
        // B = [[0,.25],[.5,.75],[1,1.25],[1.5,1.75]] so out = .75 * B[:,1]
        let want = [0.25, 0.75, 1.25, 1.75].map(|v: f32| v * 0.75);
        for (g, w) in out.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A file with only one side of a pair is a broken adapter, and must say
    /// so rather than render as if the branch were zero.
    #[test]
    fn orphan_side_is_refused() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("cmf_lora_orphan_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("orphan.safetensors");
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for i in 0..6 {
            blob.extend_from_slice(&(i as f32).to_le_bytes());
        }
        header.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
            serde_json::json!({"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}),
        );
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hdr).unwrap();
        f.write_all(&blob).unwrap();
        drop(f);
        let err = match LoraBank::load(&path, 1.0) {
            Err(e) => e,
            Ok(_) => panic!("an adapter with a lone A side must be refused"),
        };
        assert!(err.contains("no B side"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The slot embedding's feature vector is `[v, sin(v·f), cos(v·f)]` with
    /// `v = slot/16` — check it against a hand-evaluated one-frequency MLP.
    #[test]
    fn slot_embedding_matches_definition() {
        let s = SlotEmbed {
            freqs: vec![2.0],
            // hidden = 1, input width 3
            w0: vec![1.0, 0.5, -0.25],
            b0: vec![0.1],
            w2: vec![2.0],
            b2: vec![-0.3],
            hidden: 1,
            dim: 1,
        };
        let v = 3.0f32 / 16.0;
        let feat = [v, (v * 2.0).sin(), (v * 2.0).cos()];
        let pre = 0.1 + feat[0] * 1.0 + feat[1] * 0.5 + feat[2] * -0.25;
        let hid = pre / (1.0 + (-pre).exp());
        let want = -0.3 + hid * 2.0;
        let got = s.embed(3);
        assert_eq!(got.len(), 1);
        assert!((got[0] - want).abs() < 1e-6, "{} vs {want}", got[0]);
    }
}
