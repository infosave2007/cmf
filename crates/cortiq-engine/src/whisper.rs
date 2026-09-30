//! Whisper ASR inference directly from CMF tensors.
//!
//! This intentionally implements the original 30-second Whisper path (128-bin
//! log-mel input, two stride-aware encoder convolutions, encoder/decoder
//! transformer, greedy text decoding). Linear layers use `QTensor`, so the
//! regular CMF CPU kernels and the existing Metal/Vulkan matrix backends are
//! shared instead of dequantizing a checkpoint into a second model copy.

use crate::{pool::Pool, qtensor::QTensor, tokenizer::Tokenizer};
use cortiq_core::{
    CmfModel, TensorDtype,
    quant::{bf16_to_f32, f16_to_f32},
};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{
    path::Path,
    sync::{Arc, OnceLock},
    time::Instant,
};

const SR: usize = 16_000;
const CHUNK: usize = 30 * SR;
const MEL: usize = 128;
const HOP: usize = 160;
const FFT: usize = 400;

struct Linear {
    w: QTensor,
    b: Vec<f32>,
}
impl Linear {
    fn load(m: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        let w = QTensor::from_model(m, &format!("{p}.weight"))?;
        // Whisper omits the key-projection bias in both self and cross
        // attention; other affine projections carry it. Missing bias is
        // therefore a defined all-zero vector, not a malformed checkpoint.
        let bias_name = format!("{p}.bias");
        let b = if m.tensor(&bias_name).is_some() {
            vec_tensor(m, &bias_name)?
        } else {
            vec![0.0; w.rows()]
        };
        Ok(Self { w, b })
    }
    fn apply(&self, x: &[f32], rows: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut y = vec![0.0; rows * self.w.rows()];
        self.w.matmat(x, rows, &mut y, pool);
        self.add_bias(&mut y);
        y
    }
    fn add_bias(&self, y: &mut [f32]) {
        if !self.b.is_empty() {
            for row in y.chunks_exact_mut(self.w.rows()) {
                for (v, &b) in row.iter_mut().zip(&self.b) {
                    *v += b;
                }
            }
        }
    }
}

/// Whisper's token-by-token decoder shares one normalized activation across
/// self-attention Q/K/V. Use the existing one-submit GPU batch path when the
/// three CMF tensors are device-batchable; otherwise share one CPU pool job.
fn apply_qkv(
    q: &Linear,
    k: &Linear,
    v: &Linear,
    x: &[f32],
    pool: Option<&Pool>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut qy = vec![0.0; q.w.rows()];
    let mut ky = vec![0.0; k.w.rows()];
    let mut vy = vec![0.0; v.w.rows()];
    let jobs = (
        crate::qtensor::gpu_batch_job(&q.w, x),
        crate::qtensor::gpu_batch_job(&k.w, x),
        crate::qtensor::gpu_batch_job(&v.w, x),
    );
    if let (Some((qm, qj)), Some((km, kj)), Some((vm, vj))) = jobs {
        if Arc::ptr_eq(&qm, &km) && Arc::ptr_eq(&qm, &vm) {
            let mut outputs = [&mut qy[..], &mut ky[..], &mut vy[..]];
            if crate::gpu::matvec_batch(&qm, &[qj, kj, vj], &mut outputs) {
                q.add_bias(&mut qy);
                k.add_bias(&mut ky);
                v.add_bias(&mut vy);
                return (qy, ky, vy);
            }
        }
    }
    QTensor::matvec_many([&q.w, &k.w, &v.w], x, [&mut qy, &mut ky, &mut vy], pool);
    q.add_bias(&mut qy);
    k.add_bias(&mut ky);
    v.add_bias(&mut vy);
    (qy, ky, vy)
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}
impl Norm {
    fn load(m: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self {
            w: vec_tensor(m, &format!("{p}.weight"))?,
            b: vec_tensor(m, &format!("{p}.bias"))?,
        })
    }
    fn apply(&self, x: &mut [f32]) {
        let n = x.len();
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = x
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let inv = (var + 1e-5).sqrt().recip() as f32;
        for i in 0..n {
            x[i] = (x[i] - mean as f32) * inv * self.w[i] + self.b[i];
        }
    }
    fn apply_rows(&self, x: &mut [f32]) {
        for r in x.chunks_exact_mut(self.w.len()) {
            self.apply(r);
        }
    }
}

struct Conv1d {
    w: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel: usize,
    stride: usize,
}
impl Conv1d {
    fn load(m: &Arc<CmfModel>, name: &str, stride: usize) -> Result<Self, String> {
        let wn = format!("{name}.weight");
        let e = m
            .tensor(&wn)
            .ok_or_else(|| format!("missing Whisper tensor {wn}"))?;
        if e.shape.len() != 3 {
            return Err(format!("{wn}: expected [out,in,kernel]"));
        }
        let (cout, cin, kernel) = (e.shape[0], e.shape[1], e.shape[2]);
        let src = tensor_f32(m, &wn)?;
        // Conv1d weights are [out, input-channel, tap]. Matmat consumes a
        // patch with tap-major/channel-minor layout; materialize only these
        // small projection kernels (about 20 MiB for large-v3-turbo).
        let mut transposed = vec![0.0; cout * cin * kernel];
        for o in 0..cout {
            for c in 0..cin {
                for k in 0..kernel {
                    transposed[o * cin * kernel + k * cin + c] =
                        src[o * cin * kernel + c * kernel + k];
                }
            }
        }
        Ok(Self {
            w: transposed,
            bias: vec_tensor(m, &format!("{name}.bias"))?,
            cin,
            cout,
            kernel,
            stride,
        })
    }
    fn apply(&self, x: &[f32], frames: usize, pool: Option<&Pool>) -> Vec<f32> {
        let out_frames = frames.div_ceil(self.stride);
        let pad = self.kernel / 2;
        let mut patches = vec![0.0; out_frames * self.cin * self.kernel];
        for t in 0..out_frames {
            for k in 0..self.kernel {
                let it = (t * self.stride + k).checked_sub(pad);
                if let Some(it) = it.filter(|&i| i < frames) {
                    for c in 0..self.cin {
                        patches[t * self.cin * self.kernel + k * self.cin + c] = x[c * frames + it];
                    }
                }
            }
        }
        let mut y = vec![0.0; out_frames * self.cout];
        // `gemm_nt` is the shared blocked CPU/GPU F32 GEMM. Calling the
        // generic F32 QTensor loop here made the 1280×1280 second conv
        // reread the same filter once per frame and left the accelerator out.
        crate::fcd_ops::gemm_nt(
            &patches,
            &self.w,
            &mut y,
            out_frames,
            self.cin * self.kernel,
            self.cout,
            pool,
        );
        for row in y.chunks_exact_mut(self.cout) {
            for (v, &b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        }
        // Return channels-first for the next conv / stable encoder layout.
        let mut cf = vec![0.0; self.cout * out_frames];
        for t in 0..out_frames {
            for c in 0..self.cout {
                cf[c * out_frames + t] = y[t * self.cout + c];
            }
        }
        cf
    }
}

struct EncLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n1: Norm,
    n2: Norm,
    fc1: Linear,
    fc2: Linear,
}
struct Enc {
    layers: Vec<EncLayer>,
    pos: QTensor,
    conv1: Conv1d,
    conv2: Conv1d,
    norm: Norm,
}
struct DecLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n1: Norm,
    cq: Linear,
    ck: Linear,
    cv: Linear,
    co: Linear,
    n2: Norm,
    n3: Norm,
    fc1: Linear,
    fc2: Linear,
    ek: Vec<f32>,
    ev: Vec<f32>,
    sk: Vec<f32>,
    sv: Vec<f32>,
}
struct Dec {
    layers: Vec<DecLayer>,
    embed: QTensor,
    pos: QTensor,
    norm: Norm,
}
struct Whisper {
    enc: Enc,
    dec: Dec,
    hidden: usize,
    heads: usize,
    enc_heads: usize,
    enc_len: usize,
    max_target: usize,
    vocab: usize,
}

impl Whisper {
    fn load(m: &Arc<CmfModel>, cfg: &serde_json::Value) -> Result<Self, String> {
        let usize_cfg = |k: &str, d: usize| {
            cfg.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let hidden = usize_cfg("d_model", 0);
        let enc_n = usize_cfg("encoder_layers", 0);
        let dec_n = usize_cfg("decoder_layers", 0);
        let enc_heads = usize_cfg("encoder_attention_heads", 0);
        let heads = usize_cfg("decoder_attention_heads", 0);
        let enc_len = usize_cfg("max_source_positions", 1500);
        let max_target = usize_cfg("max_target_positions", 448);
        if hidden == 0
            || enc_n == 0
            || dec_n == 0
            || heads == 0
            || enc_heads == 0
            || hidden % heads != 0
            || hidden % enc_heads != 0
        {
            return Err("invalid Whisper CMF geometry".into());
        }
        let mut el = Vec::with_capacity(enc_n);
        for i in 0..enc_n {
            let p = format!("model.encoder.layers.{i}");
            el.push(EncLayer {
                q: Linear::load(m, &format!("{p}.self_attn.q_proj"))?,
                k: Linear::load(m, &format!("{p}.self_attn.k_proj"))?,
                v: Linear::load(m, &format!("{p}.self_attn.v_proj"))?,
                o: Linear::load(m, &format!("{p}.self_attn.out_proj"))?,
                n1: Norm::load(m, &format!("{p}.self_attn_layer_norm"))?,
                n2: Norm::load(m, &format!("{p}.final_layer_norm"))?,
                fc1: Linear::load(m, &format!("{p}.fc1"))?,
                fc2: Linear::load(m, &format!("{p}.fc2"))?,
            });
        }
        let mut dl = Vec::with_capacity(dec_n);
        for i in 0..dec_n {
            let p = format!("model.decoder.layers.{i}");
            dl.push(DecLayer {
                q: Linear::load(m, &format!("{p}.self_attn.q_proj"))?,
                k: Linear::load(m, &format!("{p}.self_attn.k_proj"))?,
                v: Linear::load(m, &format!("{p}.self_attn.v_proj"))?,
                o: Linear::load(m, &format!("{p}.self_attn.out_proj"))?,
                n1: Norm::load(m, &format!("{p}.self_attn_layer_norm"))?,
                cq: Linear::load(m, &format!("{p}.encoder_attn.q_proj"))?,
                ck: Linear::load(m, &format!("{p}.encoder_attn.k_proj"))?,
                cv: Linear::load(m, &format!("{p}.encoder_attn.v_proj"))?,
                co: Linear::load(m, &format!("{p}.encoder_attn.out_proj"))?,
                n2: Norm::load(m, &format!("{p}.encoder_attn_layer_norm"))?,
                n3: Norm::load(m, &format!("{p}.final_layer_norm"))?,
                fc1: Linear::load(m, &format!("{p}.fc1"))?,
                fc2: Linear::load(m, &format!("{p}.fc2"))?,
                ek: Vec::new(),
                ev: Vec::new(),
                sk: vec![0.0; heads * max_target * (hidden / heads)],
                sv: vec![0.0; heads * max_target * (hidden / heads)],
            });
        }
        Ok(Self {
            enc: Enc {
                layers: el,
                pos: QTensor::from_model(m, "model.encoder.embed_positions.weight")?,
                conv1: Conv1d::load(m, "model.encoder.conv1", 1)?,
                conv2: Conv1d::load(m, "model.encoder.conv2", 2)?,
                norm: Norm::load(m, "model.encoder.layer_norm")?,
            },
            dec: Dec {
                layers: dl,
                embed: QTensor::from_model(m, "model.decoder.embed_tokens.weight")?,
                pos: QTensor::from_model(m, "model.decoder.embed_positions.weight")?,
                norm: Norm::load(m, "model.decoder.layer_norm")?,
            },
            hidden,
            heads,
            enc_heads,
            enc_len,
            max_target,
            vocab: usize_cfg("vocab_size", 0),
        })
    }

    fn encode(
        &self,
        feat: &[f32],
        mel_frames: usize,
        encoder_limit: Option<usize>,
        pool: Option<&Pool>,
    ) -> Result<(Vec<f32>, usize), String> {
        if mel_frames == 0 || mel_frames > CHUNK / HOP || feat.len() != MEL * mel_frames {
            return Err(format!(
                "Whisper feature shape {}, expected {} mel frames",
                feat.len(),
                MEL * mel_frames
            ));
        }
        let mut f1 = self.enc.conv1.apply(feat, mel_frames, pool);
        let len1 = mel_frames;
        for z in &mut f1 {
            *z = gelu_exact(*z);
        }
        let mut x = self.enc.conv2.apply(&f1, len1, pool);
        for z in &mut x {
            *z = gelu_exact(*z);
        }
        let conv_len = x.len() / self.hidden;
        if self.hidden * conv_len != x.len() {
            return Err(format!(
                "Whisper encoder conv output {}×{}, expected {}×{}",
                conv_len,
                self.hidden,
                mel_frames.div_ceil(2),
                self.hidden
            ));
        }
        let len = encoder_limit.unwrap_or(conv_len).min(conv_len);
        if len == 0 {
            return Err("Whisper encoder sequence is empty after trimming".into());
        }
        if len > self.enc_len {
            return Err(format!(
                "Whisper encoder sequence {len} exceeds positional table {}",
                self.enc_len
            ));
        }
        if len < conv_len {
            let mut cropped = vec![0.0; self.hidden * len];
            for c in 0..self.hidden {
                cropped[c * len..(c + 1) * len]
                    .copy_from_slice(&x[c * conv_len..c * conv_len + len]);
            }
            x = cropped;
        }
        for t in 0..len {
            let mut p = vec![0.0; self.hidden];
            self.enc.pos.row_f32(t, &mut p);
            for c in 0..self.hidden {
                x[c * len + t] += p[c];
            }
        }
        let mut rows = channels_to_rows(&x, len, self.hidden);
        for l in &self.enc.layers {
            let mut n = rows.clone();
            l.n1.apply_rows(&mut n);
            let mut q = l.q.apply(&n, len, pool);
            scale_queries(&mut q, self.hidden / self.enc_heads);
            let k = l.k.apply(&n, len, pool);
            let v = l.v.apply(&n, len, pool);
            let a = attention(&q, &k, &v, len, self.enc_heads, self.hidden, pool);
            let o = l.o.apply(&a, len, pool);
            add_inplace(&mut rows, &o);
            let mut n = rows.clone();
            l.n2.apply_rows(&mut n);
            let mut ff = l.fc1.apply(&n, len, pool);
            for z in &mut ff {
                *z = gelu_exact(*z);
            }
            let ff = l.fc2.apply(&ff, len, pool);
            add_inplace(&mut rows, &ff);
        }
        self.enc.norm.apply_rows(&mut rows);
        Ok((rows, len))
    }

    fn prep_cross(&mut self, enc: &[f32], n: usize, pool: Option<&Pool>) {
        debug_assert!(n <= self.enc_len);
        for l in &mut self.dec.layers {
            let k = l.ck.apply(enc, n, pool);
            let v = l.cv.apply(enc, n, pool);
            l.ek = to_heads(&k, n, self.heads, self.hidden / self.heads);
            l.ev = to_heads(&v, n, self.heads, self.hidden / self.heads);
        }
    }
    fn decode_step(
        &mut self,
        token: u32,
        pos: usize,
        encoder_len: usize,
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>, String> {
        if pos >= self.max_target {
            return Err(format!(
                "decoder position {pos} exceeds Whisper max_target_positions {}",
                self.max_target
            ));
        }
        let mut x = vec![0.0; self.hidden];
        self.dec.embed.row_f32(token as usize, &mut x);
        let mut pe = vec![0.0; self.hidden];
        self.dec.pos.row_f32(pos, &mut pe);
        for i in 0..self.hidden {
            x[i] += pe[i];
        }
        let hd = self.hidden / self.heads;
        for l in &mut self.dec.layers {
            let mut n = x.clone();
            l.n1.apply(&mut n);
            let (mut q, k, v) = apply_qkv(&l.q, &l.k, &l.v, &n, pool);
            scale_queries(&mut q, hd);
            let qh = to_heads(&q, 1, self.heads, hd);
            let kh = to_heads(&k, 1, self.heads, hd);
            let vh = to_heads(&v, 1, self.heads, hd);
            for h in 0..self.heads {
                let o = h * self.max_target * hd + pos * hd;
                l.sk[o..o + hd].copy_from_slice(&kh[h * hd..(h + 1) * hd]);
                l.sv[o..o + hd].copy_from_slice(&vh[h * hd..(h + 1) * hd]);
            }
            let mut a = vec![0.0; self.hidden];
            for h in 0..self.heads {
                let mut scores = vec![0.0; pos + 1];
                for t in 0..=pos {
                    let base = h * self.max_target * hd + t * hd;
                    let mut s = 0.0;
                    for d in 0..hd {
                        s += qh[h * hd + d] * l.sk[base + d];
                    }
                    scores[t] = s;
                }
                softmax(&mut scores);
                for d in 0..hd {
                    let mut z = 0.0;
                    for t in 0..=pos {
                        z += scores[t] * l.sv[h * self.max_target * hd + t * hd + d];
                    }
                    a[h * hd + d] = z;
                }
            }
            let o = l.o.apply(&a, 1, pool);
            add_inplace(&mut x, &o);
            let mut n = x.clone();
            l.n2.apply(&mut n);
            let q = l.cq.apply(&n, 1, pool);
            let mut qh = to_heads(&q, 1, self.heads, hd);
            scale_queries(&mut qh, hd);
            let mut ctx = vec![0.0; self.hidden];
            for h in 0..self.heads {
                let mut scores = vec![0.0; encoder_len];
                for t in 0..encoder_len {
                    let base = h * encoder_len * hd + t * hd;
                    let mut s = 0.0;
                    for d in 0..hd {
                        s += qh[h * hd + d] * l.ek[base + d];
                    }
                    scores[t] = s;
                }
                softmax(&mut scores);
                for d in 0..hd {
                    let mut z = 0.0;
                    for t in 0..encoder_len {
                        z += scores[t] * l.ev[h * encoder_len * hd + t * hd + d];
                    }
                    ctx[h * hd + d] = z;
                }
            }
            qh.clear();
            let o = l.co.apply(&ctx, 1, pool);
            add_inplace(&mut x, &o);
            let mut n = x.clone();
            l.n3.apply(&mut n);
            let mut ff = l.fc1.apply(&n, 1, pool);
            for z in &mut ff {
                *z = gelu_exact(*z);
            }
            let ff = l.fc2.apply(&ff, 1, pool);
            add_inplace(&mut x, &ff);
        }
        self.dec.norm.apply(&mut x);
        let mut logits = vec![0.0; self.vocab];
        self.dec.embed.matvec(&x, &mut logits, pool);
        Ok(logits)
    }
}

/// Transcribe a WAV file. Long inputs are decoded in non-overlapping 30s
/// windows, with a single persistent CPU pool shared by all projections.
pub fn transcribe(
    model: &Arc<CmfModel>,
    wav: &Path,
    language: &str,
    task: &str,
    max_new_tokens: usize,
) -> Result<String, String> {
    if !model.header.arch.arch_name.eq_ignore_ascii_case("whisper") {
        return Err(format!(
            "{} is not a Whisper CMF checkpoint",
            model.path.display()
        ));
    }
    if !matches!(task, "transcribe" | "translate") {
        return Err("--task must be transcribe or translate".into());
    }
    if max_new_tokens == 0 || max_new_tokens > 448 {
        return Err("--max-new-tokens must be in 1..=448".into());
    }
    let cfg = model
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.get("whisper_config"))
        .ok_or("Whisper CMF is missing the source config in provenance")?;
    let mut net = Whisper::load(model, cfg)?;
    let vocab = model
        .vocab
        .as_deref()
        .ok_or("Whisper CMF has no embedded tokenizer.json")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("Whisper tokenizer: {e}"))?;
    let special = |s: &str| {
        tok.token_to_id(s)
            .ok_or_else(|| format!("tokenizer is missing Whisper token {s}"))
    };
    let prefix = [
        special("<|startoftranscript|>")?,
        special(&format!("<|{language}|>"))?,
        special(if task == "translate" {
            "<|translate|>"
        } else {
            "<|transcribe|>"
        })?,
        special("<|notimestamps|>")?,
    ];
    if prefix.iter().any(|&id| id as usize >= net.vocab) {
        return Err("Whisper tokenizer prompt IDs exceed CMF vocabulary size".into());
    }
    let eos = special("<|endoftext|>")?;
    let timestamp = special("<|0.00|>")?;
    let sot = prefix[0];
    let mut forbidden = prefix.to_vec();
    // Whisper's control-token IDs are contiguous between start-of-transcript
    // and the first timestamp. Only ordinary text and EOT are legal output.
    forbidden.extend(sot..timestamp);
    let mut begin_forbidden = Vec::new();
    if let Some(generation) = model
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.get("whisper_generation_config"))
    {
        for key in ["suppress_tokens", "begin_suppress_tokens"] {
            if let Some(ids) = generation.get(key).and_then(|v| v.as_array()) {
                for id in ids.iter().filter_map(|v| v.as_u64()).map(|v| v as u32) {
                    if key == "suppress_tokens" {
                        forbidden.push(id);
                    } else {
                        begin_forbidden.push(id);
                    }
                }
            }
        }
    }
    for name in ["<|nospeech|>", "<|startofprev|>", "<|endofprev|>"] {
        if let Some(id) = tok.token_to_id(name) {
            forbidden.push(id);
        }
    }
    let (mono, rate) = read_wav(wav)?;
    let mono = resample_sinc(&mono, rate, SR);
    if mono.is_empty() {
        return Err("audio file contains no samples".into());
    }
    let nworkers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
        .saturating_sub(1);
    let pool = Pool::new(nworkers);
    let profile = std::env::var_os("CMF_WHISPER_PROFILE").is_some();
    let trim_encoder = std::env::var("CMF_WHISPER_TRIM_ENCODER")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    let mut texts = Vec::new();
    for (segment_index, segment) in mono.chunks(CHUNK).enumerate() {
        // Exact/near digital silence needs neither a 30 s encoder pass nor a
        // decoder pass; skipping it also avoids Whisper's known blank-audio
        // hallucinations on an otherwise empty recording.
        if is_near_silence(segment) {
            continue;
        }
        let mut padded = vec![0.0; CHUNK];
        padded[..segment.len()].copy_from_slice(segment);
        let mel_frames = if trim_encoder {
            (segment.len().div_ceil(HOP) + 4).min(CHUNK / HOP)
        } else {
            CHUNK / HOP
        };
        // Keep a full 30-second window by default to preserve the reference
        // model's attention context. The opt-in trim keeps only the useful
        // audio prefix plus convolution/STFT context; the attention context is
        // consequently shorter and should be validated for the target audio.
        let encoder_limit =
            (trim_encoder && mel_frames < CHUNK / HOP).then_some(mel_frames.saturating_sub(1) / 2);
        let t0 = Instant::now();
        let feat = log_mel(&padded, mel_frames);
        let mel_time = t0.elapsed();
        let t0 = Instant::now();
        let (enc, n) = net.encode(&feat, mel_frames, encoder_limit, Some(&pool))?;
        let encode_time = t0.elapsed();
        let t0 = Instant::now();
        net.prep_cross(&enc, n, Some(&pool));
        let cross_time = t0.elapsed();
        let t0 = Instant::now();
        let mut logits = Vec::new();
        for (i, &id) in prefix.iter().enumerate() {
            logits = net.decode_step(id, i, n, Some(&pool))?;
        }
        let mut output = Vec::new();
        let token_limit = max_new_tokens.min(net.max_target.saturating_sub(prefix.len()));
        for generated in 0..token_limit {
            for &id in &forbidden {
                if let Some(v) = logits.get_mut(id as usize) {
                    *v = f32::NEG_INFINITY;
                }
            }
            if generated == 0 {
                for &id in &begin_forbidden {
                    if let Some(v) = logits.get_mut(id as usize) {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            let mut next = argmax(&logits);
            // With no-timestamps decoding, timestamp tokens are never valid
            // transcript text. Also terminate at end-of-text as HF does.
            if next >= timestamp && (timestamp as usize) < logits.len() {
                logits[timestamp as usize..].fill(f32::NEG_INFINITY);
                next = argmax(&logits);
            }
            if next == eos {
                break;
            }
            output.push(next);
            if generated + 1 < max_new_tokens {
                logits = net.decode_step(next, prefix.len() + generated, n, Some(&pool))?;
            }
        }
        let text = tok.decode(&output).trim().to_owned();
        if !text.is_empty() {
            texts.push(text);
        }
        if profile {
            eprintln!(
                "whisper-profile segment={} duration_s={:.3} mel_frames={} encoder_frames={} trim={} tokens={} logmel_ms={} encoder_ms={} cross_ms={} decode_ms={}",
                segment_index + 1,
                segment.len() as f64 / SR as f64,
                mel_frames,
                n,
                trim_encoder,
                output.len(),
                mel_time.as_millis(),
                encode_time.as_millis(),
                cross_time.as_millis(),
                t0.elapsed().as_millis(),
            );
        }
    }
    Ok(texts.join(" "))
}

fn vec_tensor(m: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    tensor_f32(m, name)
}
fn tensor_f32(m: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let e = m
        .tensor(name)
        .ok_or_else(|| format!("missing Whisper tensor {name}"))?;
    let b = m.entry_bytes(e);
    let n = e.n_elems();
    match e.dtype {
        TensorDtype::F32 if b.len() == n * 4 => Ok(b
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()),
        TensorDtype::F16 if b.len() == n * 2 => Ok(b
            .chunks_exact(2)
            .map(|x| f16_to_f32(u16::from_le_bytes([x[0], x[1]])))
            .collect()),
        TensorDtype::Bf16 if b.len() == n * 2 => Ok(b
            .chunks_exact(2)
            .map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]])))
            .collect()),
        _ => Err(format!(
            "{name}: expected floating point tensor, got {:?}",
            e.dtype
        )),
    }
}
fn channels_to_rows(cf: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut o = vec![0.0; cf.len()];
    for i in 0..t {
        for j in 0..c {
            o[i * c + j] = cf[j * t + i];
        }
    }
    o
}
fn to_heads(x: &[f32], n: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut o = vec![0.0; x.len()];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..hd {
                o[h * n * hd + t * hd + d] = x[t * heads * hd + h * hd + d];
            }
        }
    }
    o
}
fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}
fn scale_queries(q: &mut [f32], head_dim: usize) {
    let s = 1.0 / (head_dim as f32).sqrt();
    for v in q {
        *v *= s;
    }
}
/// erf-based GELU (`activation_function: gelu` in HF Whisper). The compact
/// Abramowitz-Stegun erf approximation is within ~1.5e-7 and avoids depending
/// on a platform libm symbol in the portable CPU/GPU runtime.
fn gelu_exact(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let erf_abs = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-a * a).exp();
    let erf = if z < 0.0 { -erf_abs } else { erf_abs };
    0.5 * x * (1.0 + erf)
}
fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s.max(f32::MIN_POSITIVE);
    for v in x {
        *v *= inv;
    }
}
fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0 as u32)
        .unwrap_or(0)
}

/// Flash/online GPU attention where available; a bounded-memory, exact
/// softmax fallback handles CPU-only hosts and avoids materializing S×S×H.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    hid: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let hd = hid / heads;
    let qh = to_heads(q, n, heads, hd);
    let kh = to_heads(k, n, heads, hd);
    let vh = to_heads(v, n, heads, hd);
    let mut oh = vec![0.0; heads * n * hd];
    if crate::gpu::dit_attention(&qh, &kh, &vh, heads, heads, n, hd, 1.0, &mut oh) {
        // Device attention writes the projection-ready [time, hidden] panel;
        // only the CPU fallback below retains the head-major scratch layout.
        return oh;
    }
    for h in 0..heads {
        let hb = h * n * hd;
        let mut scores = vec![0.0; n * n];
        crate::fcd_ops::gemm_nt(
            &qh[hb..hb + n * hd],
            &kh[hb..hb + n * hd],
            &mut scores,
            n,
            hd,
            n,
            pool,
        );
        for row in scores.chunks_exact_mut(n) {
            softmax(row);
        }
        let mut vt = vec![0.0; hd * n];
        for t in 0..n {
            for d in 0..hd {
                vt[d * n + t] = vh[hb + t * hd + d];
            }
        }
        crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh[hb..hb + n * hd], n, n, hd, pool);
    }
    heads_to_rows(&oh, n, heads, hd)
}

/// Convert the attention kernel's `[head, time, head_dim]` layout back to the
/// transformer projection layout `[time, hidden]`.
fn heads_to_rows(x: &[f32], n: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..hd {
                out[t * heads * hd + h * hd + d] = x[h * n * hd + t * hd + d];
            }
        }
    }
    out
}

fn read_wav(path: &Path) -> Result<(Vec<f32>, usize), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 44 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("input must be RIFF/WAVE".into());
    }
    let (mut format, mut ch, mut rate, mut bits) = (0u16, 0usize, 0usize, 0usize);
    let mut data = None;
    let mut i = 12;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let st = i + 8;
        let end = st.saturating_add(len).min(b.len());
        if &b[i..i + 4] == b"fmt " && end >= st + 16 {
            format = u16::from_le_bytes([b[st], b[st + 1]]);
            ch = u16::from_le_bytes([b[st + 2], b[st + 3]]) as usize;
            rate = u32::from_le_bytes(b[st + 4..st + 8].try_into().unwrap()) as usize;
            bits = u16::from_le_bytes([b[st + 14], b[st + 15]]) as usize;
            if format == 0xfffe && end >= st + 26 {
                format = u16::from_le_bytes([b[st + 24], b[st + 25]]);
            }
        } else if &b[i..i + 4] == b"data" {
            data = Some((st, end));
            break;
        }
        i = st.saturating_add(len).saturating_add(len & 1);
    }
    let (st, end) = data.ok_or("WAV has no data chunk")?;
    if ch == 0 || rate == 0 {
        return Err("invalid WAV channels or sample rate".into());
    }
    if !matches!((format, bits), (1, 8 | 16 | 24 | 32) | (3, 32 | 64)) {
        return Err(format!(
            "unsupported WAV encoding format={format} bits={bits}; use PCM or IEEE float WAV"
        ));
    }
    let bytes = bits / 8;
    let frame = bytes * ch;
    if frame == 0 {
        return Err("invalid WAV frame width".into());
    }
    let mut out = Vec::with_capacity((end - st) / frame);
    for fr in b[st..end].chunks_exact(frame) {
        let mut sum = 0.0f64;
        for x in fr.chunks_exact(bytes) {
            let v = match (format, bits) {
                (1, 8) => (x[0] as f32 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([x[0], x[1]]) as f32 / 32768.0,
                (1, 24) => {
                    let q = (x[0] as i32) | ((x[1] as i32) << 8) | ((x[2] as i32) << 16);
                    ((q << 8) >> 8) as f32 / 8_388_608.0
                }
                (1, 32) => i32::from_le_bytes(x.try_into().unwrap()) as f32 / 2_147_483_648.0,
                (3, 32) => f32::from_le_bytes(x.try_into().unwrap()),
                (3, 64) => f64::from_le_bytes(x.try_into().unwrap()) as f32,
                _ => 0.0,
            };
            sum += if v.is_finite() {
                v.clamp(-1.0, 1.0) as f64
            } else {
                0.0
            };
        }
        out.push((sum / ch as f64) as f32);
    }
    Ok((out, rate))
}
fn resample_sinc(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let n = (x.len() as u128 * to as u128 / from as u128) as usize;
    let cutoff = (to as f64 / from as f64).min(1.0);
    let half = 24isize;
    let mut y = vec![0.0; n];
    for (i, out) in y.iter_mut().enumerate() {
        let pos = i as f64 * from as f64 / to as f64;
        let center = pos.floor() as isize;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for j in center - half + 1..=center + half {
            if j < 0 || j >= x.len() as isize {
                continue;
            }
            let d = (pos - j as f64) * cutoff;
            let sinc = if d.abs() < 1e-10 {
                1.0
            } else {
                (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d)
            };
            let w = 0.5 + 0.5 * (std::f64::consts::PI * (pos - j as f64) / half as f64).cos();
            let a = sinc * cutoff * w;
            sum += x[j as usize] as f64 * a;
            norm += a;
        }
        *out = if norm.abs() > 1e-12 {
            (sum / norm) as f32
        } else {
            0.0
        };
    }
    y
}

fn hz_to_mel(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}
fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m < min_log_mel {
        m * f_sp
    } else {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    }
}

struct MelConfig {
    window: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
    filters: Vec<f64>,
    filter_ranges: Vec<(usize, usize)>,
}

fn mel_config() -> &'static MelConfig {
    static CONFIG: OnceLock<MelConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let window = (0..FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos())
            .collect();
        let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT);
        let edges: Vec<f64> = (0..MEL + 2)
            .map(|i| {
                mel_to_hz(
                    hz_to_mel(0.0)
                        + (hz_to_mel(8000.0) - hz_to_mel(0.0)) * i as f64 / (MEL + 1) as f64,
                )
            })
            .collect();
        let mut filters = vec![0.0f64; MEL * 201];
        for m in 0..MEL {
            let norm = 2.0 / (edges[m + 2] - edges[m]);
            for k in 0..201 {
                let hz = k as f64 * SR as f64 / FFT as f64;
                let lo = (hz - edges[m]) / (edges[m + 1] - edges[m]);
                let hi = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
                filters[m * 201 + k] = lo.min(hi).max(0.0) * norm;
            }
        }
        let filter_ranges = (0..MEL)
            .map(|m| {
                let row = &filters[m * 201..(m + 1) * 201];
                let start = row.iter().position(|&weight| weight > 0.0).unwrap_or(0);
                let end = row
                    .iter()
                    .rposition(|&weight| weight > 0.0)
                    .map_or(start, |k| k + 1);
                (start, end)
            })
            .collect();
        MelConfig {
            window,
            fft,
            filters,
            filter_ranges,
        }
    })
}

fn is_near_silence(samples: &[f32]) -> bool {
    samples.is_empty()
        || samples
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            <= samples.len() as f64 * 1e-10
}

fn log_mel(samples: &[f32], frames: usize) -> Vec<f32> {
    let cfg = mel_config();
    debug_assert!(frames > 0 && frames <= CHUNK / HOP);
    let mut mel = vec![0.0f32; MEL * frames];
    let mut spec = vec![0.0f64; 201];
    let mut fft_buf = vec![Complex::new(0.0, 0.0); FFT];
    for t in 0..frames {
        let center = t * HOP;
        fft_frame_power(
            samples,
            center,
            &cfg.window,
            cfg.fft.as_ref(),
            &mut fft_buf,
            &mut spec,
        );
        for m in 0..MEL {
            let mut e = 0.0;
            let (start, end) = cfg.filter_ranges[m];
            for k in start..end {
                e += spec[k] * cfg.filters[m * 201 + k];
            }
            mel[m * frames + t] = e.max(1e-10).log10() as f32;
        }
    }
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut mel {
        *v = ((*v).max(max - 8.0) + 4.0) / 4.0;
    }
    mel
}

/// Calculate the one-sided power spectrum for one Whisper STFT frame. The
/// original 400-point transform is retained exactly (unlike zero-padding to
/// a convenient power-of-two size), but RustFFT evaluates it in O(N log N)
/// rather than the old O(N^2) scalar DFT.
fn fft_frame_power(
    samples: &[f32],
    center: usize,
    window: &[f64],
    fft: &dyn Fft<f64>,
    buffer: &mut [Complex<f64>],
    out: &mut [f64],
) {
    debug_assert_eq!(window.len(), FFT);
    debug_assert_eq!(buffer.len(), FFT);
    debug_assert_eq!(out.len(), FFT / 2 + 1);
    for i in 0..FFT {
        let ix = center as isize + i as isize - FFT as isize / 2;
        let ix = reflect(ix, samples.len());
        buffer[i] = Complex::new(samples[ix] as f64 * window[i], 0.0);
    }
    fft.process(buffer);
    let bins = out.len();
    for (power, bin) in out.iter_mut().zip(&buffer[..bins]) {
        *power = bin.norm_sqr();
    }
}
fn reflect(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut x = i;
    let hi = n as isize;
    while x < 0 || x >= hi {
        if x < 0 {
            x = -x;
        }
        if x >= hi {
            x = 2 * hi - 2 - x;
        }
    }
    x as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_gelu_matches_reference_values() {
        assert_eq!(gelu_exact(0.0), 0.0);
        assert!((gelu_exact(1.0) - 0.841_344_7).abs() < 2e-6);
        assert!((gelu_exact(-1.0) + 0.158_655_3).abs() < 2e-6);
        assert!((gelu_exact(8.0) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn sinc_resampler_preserves_rate_and_dc() {
        let input = vec![0.25; 3200];
        assert_eq!(resample_sinc(&input, SR, SR), input);
        let output = resample_sinc(&input, 48_000, SR);
        assert_eq!(output.len(), 1066);
        assert!(
            output[40..output.len() - 40]
                .iter()
                .all(|&x| (x - 0.25).abs() < 1e-4)
        );
    }

    #[test]
    fn reflected_stft_edges_match_torch_reflect_padding() {
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-2, 5), 2);
        assert_eq!(reflect(5, 5), 3);
    }

    #[test]
    fn rustfft_preserves_the_original_400_point_stft_spectrum() {
        let cfg = mel_config();
        let samples: Vec<f32> = (0..1024)
            .map(|i| {
                let phase = i as f32;
                (phase * 0.071).sin() * 0.4 + (phase * 0.013).cos() * 0.2
            })
            .collect();
        let mut buffer = vec![Complex::new(0.0, 0.0); FFT];
        let mut got = vec![0.0; FFT / 2 + 1];
        for center in [0, 160, 512, 1023] {
            fft_frame_power(
                &samples,
                center,
                &cfg.window,
                cfg.fft.as_ref(),
                &mut buffer,
                &mut got,
            );
            for k in 0..=FFT / 2 {
                let mut re = 0.0;
                let mut im = 0.0;
                for i in 0..FFT {
                    let ix = reflect(
                        center as isize + i as isize - FFT as isize / 2,
                        samples.len(),
                    );
                    let v = samples[ix] as f64 * cfg.window[i];
                    let a = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / FFT as f64;
                    re += v * a.cos();
                    im -= v * a.sin();
                }
                let want = re * re + im * im;
                assert!(
                    (got[k] - want).abs() <= want.max(1.0) * 1e-11,
                    "center={center} bin={k}: got={} want={want}",
                    got[k]
                );
            }
        }
    }

    #[test]
    fn sparse_mel_filters_match_the_original_dense_reduction() {
        let cfg = mel_config();
        let spectrum: Vec<f64> = (0..201)
            .map(|k| (k as f64 * 0.17).sin().abs() + k as f64 * 1e-3)
            .collect();
        for m in 0..MEL {
            let dense: f64 = (0..201)
                .map(|k| spectrum[k] * cfg.filters[m * 201 + k])
                .sum();
            let (start, end) = cfg.filter_ranges[m];
            let sparse =
                (start..end).fold(0.0, |sum, k| sum + spectrum[k] * cfg.filters[m * 201 + k]);
            assert_eq!(sparse, dense, "mel filter {m}");
        }
    }

    #[test]
    fn trimmed_log_mel_matches_full_window_prefix() {
        let audio_len = 16_137;
        let mut samples = vec![0.0f32; CHUNK];
        for (i, x) in samples[..audio_len].iter_mut().enumerate() {
            *x = (i as f32 * 0.037).sin() * 0.2;
        }
        let frames = audio_len.div_ceil(HOP) + 4;
        let full = log_mel(&samples, CHUNK / HOP);
        let trimmed = log_mel(&samples, frames);
        for m in 0..MEL {
            assert_eq!(
                &trimmed[m * frames..(m + 1) * frames],
                &full[m * (CHUNK / HOP)..m * (CHUNK / HOP) + frames],
                "mel band {m}"
            );
        }
    }

    #[test]
    fn trimmed_convolutions_preserve_the_encoder_prefix() {
        let make_conv = |cin: usize, cout: usize, stride: usize| Conv1d {
            w: (0..cout * cin * 3)
                .map(|i| ((i * 13 % 31) as f32 - 15.0) / 100.0)
                .collect(),
            bias: (0..cout).map(|i| i as f32 / 50.0).collect(),
            cin,
            cout,
            kernel: 3,
            stride,
        };
        let frames = 14;
        let full_frames = CHUNK / HOP;
        let mel: Vec<f32> = (0..2 * full_frames)
            .map(|i| ((i * 7 % 97) as f32 - 48.0) / 97.0)
            .collect();
        let mut short_mel = vec![0.0; 2 * frames];
        for c in 0..2 {
            short_mel[c * frames..(c + 1) * frames]
                .copy_from_slice(&mel[c * full_frames..c * full_frames + frames]);
        }
        let conv1 = make_conv(2, 3, 1);
        let full1 = conv1.apply(&mel, full_frames, None);
        let short1 = conv1.apply(&short_mel, frames, None);
        let conv2 = make_conv(3, 4, 2);
        let full2 = conv2.apply(&full1, full_frames, None);
        let short2 = conv2.apply(&short1, frames, None);
        let keep = (frames - 1) / 2;
        let full_len = full_frames.div_ceil(2);
        for c in 0..conv2.cout {
            assert_eq!(
                &short2[c * short2.len() / conv2.cout..c * short2.len() / conv2.cout + keep],
                &full2[c * full_len..c * full_len + keep],
                "encoder channel {c}"
            );
        }
    }

    #[test]
    fn attention_heads_return_to_projection_row_layout() {
        // 2 heads × 2 time steps × 2 channels.
        let head_major = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        assert_eq!(
            heads_to_rows(&head_major, 2, 2, 2),
            [0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0]
        );
    }

    #[test]
    fn silence_fast_path_is_conservative() {
        assert!(is_near_silence(&[0.0; 8]));
        assert!(is_near_silence(&[1e-6, -1e-6]));
        assert!(!is_near_silence(&[1e-4]));
        assert!(!is_near_silence(&[0.01, 0.0]));
    }
}
