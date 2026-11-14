//! Qwen3-TTS 12 Hz (CustomVoice): text to 24 kHz speech.
//!
//! The talker is a Qwen3 decoder over the codec vocabulary — the file's
//! language model, driven row by row through `Pipeline::speech_*`. Its
//! prompt is built from the text embedding and its projection
//! (`qtts.text_*`), the codec's control codes and the speaker's code; each
//! step samples the first codebook of a 12.5 Hz frame from the codec head.
//! The code predictor, a five-layer Qwen3 over the frame's sixteen slots
//! (`qtts.code_predictor.*`), samples the other fifteen; the frame's sixteen
//! code embeddings, summed with the text pad, are the next step's input. The
//! speech tokenizer's decoder (`qtts.codec.decoder.*`) turns the frames into
//! audio: RVQ, a causal convolution, an eight-layer sliding-window
//! transformer, two ConvNeXt upsamplings and a SnakeBeta vocoder, 1920
//! samples a frame. Everything past the talker runs in f32 on the host.
//!
//! The prompt and the sampling follow `qwen_tts` 0.1.1
//! (`Qwen3TTSForConditionalGeneration.generate`, non-streaming text, the
//! release's `generation_config.json`).

use crate::inference::{gelu_erf, silu};
use crate::pipeline::Pipeline;
use crate::pool::Pool;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::Arc;

/// Samples a frame decodes to at 24 kHz.
pub const FRAME_SAMPLES: usize = 1920;

/// Stage clocks of `CMF_QTTS_PROF=1` (microseconds): the code predictor's
/// prefill, steps and heads; the decoder's RVQ + causal conv, transformer,
/// upsampling, vocoder input, its four blocks and the output.
static PROF: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
const PROF_NAMES: [&str; 12] =
    ["cp prefill", "cp steps", "cp heads", "rvq+conv", "transformer", "upsample", "conv in", "block 1", "block 2", "block 3", "block 4", "out"];

fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_QTTS_PROF").is_some())
}

fn prof(slot: usize, t: std::time::Instant) {
    if prof_on() {
        PROF[slot].fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The stage clocks, one line, `CMF_QTTS_PROF=1` only.
pub fn prof_report() -> Option<String> {
    prof_on().then(|| {
        PROF_NAMES
            .iter()
            .zip(&PROF)
            .map(|(n, a)| format!("{n} {:.0} ms", a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3))
            .collect::<Vec<_>>()
            .join(" · ")
    })
}

/// `f(lo, hi)` over `rows`, split across the pool when there is one.
pub(crate) fn par_rows(pool: Option<&Pool>, rows: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) if rows > 1 => p.run_rows(rows, f),
        _ => f(0, rows),
    }
}
/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;

/// Whether the file is a Qwen3-TTS conversion.
pub fn is_qwen3_tts(model: &CmfModel) -> bool {
    model.header.provenance.as_ref().is_some_and(|p| p.get("qwen3_tts").is_some())
}

fn ju(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("qwen3_tts: config lacks {k}"))
}

fn jf(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

/// The sampling of one level (the talker's first codebook, or the code
/// predictor's fifteen).
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

/// What `generate` needs from the file's `provenance.qwen3_tts`.
#[derive(Clone, Debug)]
pub struct Qwen3TtsConfig {
    pub tts_bos: u32,
    pub tts_eos: u32,
    pub tts_pad: u32,
    pub codec_bos: u32,
    pub codec_eos: u32,
    pub codec_pad: u32,
    pub think: u32,
    pub nothink: u32,
    pub think_bos: u32,
    pub think_eos: u32,
    /// `(name, codec id)`, lowercase.
    pub languages: Vec<(String, u32)>,
    /// `(name, codec id, dialect)`, lowercase.
    pub speakers: Vec<(String, u32, Option<String>)>,
    pub hidden: usize,
    pub text_hidden: usize,
    pub groups: usize,
    pub vocab: usize,
    pub talker: Sampling,
    pub sub: Sampling,
    pub repetition_penalty: f32,
    pub max_frames: usize,
    cp: Value,
    codec: Value,
}

impl Qwen3TtsConfig {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let p = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("qwen3_tts"))
            .ok_or("qwen3_tts: the file has no provenance.qwen3_tts")?;
        let cfg = p.get("config").ok_or("qwen3_tts: provenance has no config")?;
        let tc = cfg.get("talker_config").ok_or("qwen3_tts: config has no talker_config")?;
        let gcfg = p.get("generation").cloned().unwrap_or(Value::Null);
        let id = |v: &Value, k: &str| ju(v, k).map(|x| x as u32);
        let mut languages: Vec<(String, u32)> = tc
            .get("codec_language_id")
            .and_then(|m| m.as_object())
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.to_lowercase(), v.as_u64()? as u32))).collect())
            .unwrap_or_default();
        languages.sort();
        let dialects = tc.get("spk_is_dialect").and_then(|m| m.as_object());
        let mut speakers: Vec<(String, u32, Option<String>)> = tc
            .get("spk_id")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let d = dialects.and_then(|d| d.get(k)).and_then(|d| d.as_str()).map(|s| s.to_lowercase());
                        Some((k.to_lowercase(), v.as_u64()? as u32, d))
                    })
                    .collect()
            })
            .unwrap_or_default();
        speakers.sort();
        let samp = |pre: &str| Sampling {
            do_sample: gcfg.get(&format!("{pre}dosample")).or_else(|| gcfg.get(&format!("{pre}do_sample"))).and_then(|v| v.as_bool()).unwrap_or(true),
            temperature: jf(&gcfg, &format!("{pre}temperature"), 0.9) as f32,
            top_k: jf(&gcfg, &format!("{pre}top_k"), 50.0) as usize,
            top_p: jf(&gcfg, &format!("{pre}top_p"), 1.0) as f32,
        };
        Ok(Self {
            tts_bos: id(cfg, "tts_bos_token_id")?,
            tts_eos: id(cfg, "tts_eos_token_id")?,
            tts_pad: id(cfg, "tts_pad_token_id")?,
            codec_bos: id(tc, "codec_bos_id")?,
            codec_eos: id(tc, "codec_eos_token_id")?,
            codec_pad: id(tc, "codec_pad_id")?,
            think: id(tc, "codec_think_id")?,
            nothink: id(tc, "codec_nothink_id")?,
            think_bos: id(tc, "codec_think_bos_id")?,
            think_eos: id(tc, "codec_think_eos_id")?,
            languages,
            speakers,
            hidden: ju(tc, "hidden_size")?,
            text_hidden: ju(tc, "text_hidden_size")?,
            groups: ju(tc, "num_code_groups")?,
            vocab: ju(tc, "vocab_size")?,
            talker: Sampling { do_sample: gcfg.get("do_sample").and_then(|v| v.as_bool()).unwrap_or(true), ..samp("") },
            sub: samp("subtalker_"),
            repetition_penalty: jf(&gcfg, "repetition_penalty", 1.05) as f32,
            max_frames: jf(&gcfg, "max_new_tokens", 8192.0) as usize,
            cp: tc.get("code_predictor_config").cloned().ok_or("qwen3_tts: no code_predictor_config")?,
            codec: p.get("codec").and_then(|c| c.get("decoder_config")).cloned().ok_or("qwen3_tts: no speech tokenizer decoder_config")?,
        })
    }

    pub fn speaker_names(&self) -> Vec<&str> {
        self.speakers.iter().map(|s| s.0.as_str()).collect()
    }

    pub fn language_names(&self) -> Vec<&str> {
        self.languages.iter().map(|s| s.0.as_str()).collect()
    }
}

// ───────────────────────────── weights ─────────────────────────────

/// A matrix in f32 for the host GEMM: `y[n, rows] = x[n, cols] · Wᵀ (+ b)`.
pub(crate) struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    rows: usize,
    cols: usize,
}

impl Lin {
    pub(crate) fn load(model: &CmfModel, name: &str, rows: usize, cols: usize, bias: bool) -> Result<Self, String> {
        let w = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        if w.len() != rows * cols {
            return Err(format!("{name}.weight: {} values, expected [{rows}, {cols}]", w.len()));
        }
        let b = if bias { Some(vec_n(model, &format!("{name}.bias"), rows)?) } else { None };
        Ok(Self { w, b, rows, cols })
    }

    pub(crate) fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut y = vec![0f32; n * self.rows];
        crate::fcd_ops::gemm_nt(x, &self.w, &mut y, n, self.cols, self.rows, pool);
        if let Some(b) = &self.b {
            for r in y.chunks_mut(self.rows) {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        }
        y
    }
}

pub(crate) fn vec_n(model: &CmfModel, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != n {
        return Err(format!("{name}: {} values, expected {n}", v.len()));
    }
    Ok(v)
}

pub(crate) fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps as f64).sqrt();
    x.iter().zip(w).map(|(&v, &g)| (v as f64 * r) as f32 * g).collect()
}

/// RoPE (`rotate_half`) on one head of `hd` at `pos`.
fn rope(x: &mut [f32], pos: usize, theta: f64) {
    let half = x.len() / 2;
    let hd = x.len() as f64;
    for j in 0..half {
        let inv = 1.0 / theta.powf(2.0 * j as f64 / hd);
        let (s, c) = (pos as f64 * inv as f32 as f64).sin_cos();
        let (a, b) = (x[j] as f64, x[j + half] as f64);
        x[j] = (a * c - b * s) as f32;
        x[j + half] = (b * c + a * s) as f32;
    }
}

// ───────────────────────────── text side ─────────────────────────────

/// The talker's text input: the text embedding row of a token through the
/// text projection (`fc2(silu(fc1(e) + b1)) + b2`).
struct TextIn {
    model: Arc<CmfModel>,
    emb: usize,
    width: usize,
    fc1: Lin,
    fc2: Lin,
}

impl TextIn {
    fn load(model: &Arc<CmfModel>, c: &Qwen3TtsConfig) -> Result<Self, String> {
        let name = "qtts.text_embedding.weight";
        let emb = model.tensor_index(name).ok_or("qwen3_tts: no qtts.text_embedding.weight")?;
        let e = &model.tensors[emb];
        if e.shape.len() != 2 || e.shape[1] != c.text_hidden {
            return Err(format!("{name}: shape {:?}", e.shape));
        }
        Ok(Self {
            model: model.clone(),
            emb,
            width: c.text_hidden,
            fc1: Lin::load(model, "qtts.text_projection.linear_fc1", c.text_hidden, c.text_hidden, true)?,
            fc2: Lin::load(model, "qtts.text_projection.linear_fc2", c.hidden, c.text_hidden, true)?,
        })
    }

    fn embed_row(&self, id: u32) -> Result<Vec<f32>, String> {
        let e = &self.model.tensors[self.emb];
        let rows = e.shape[0];
        if id as usize >= rows {
            return Err(format!("qwen3_tts: text token {id} past the embedding's {rows} rows"));
        }
        let bytes = self.model.entry_bytes(e);
        let w = self.width;
        match e.dtype {
            cortiq_core::TensorDtype::F16 => {
                let r = &bytes[id as usize * w * 2..(id as usize + 1) * w * 2];
                Ok(r.chunks_exact(2).map(|b| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect())
            }
            cortiq_core::TensorDtype::F32 => {
                let r = &bytes[id as usize * w * 4..(id as usize + 1) * w * 4];
                Ok(r.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
            }
            d => Err(format!("qwen3_tts: text embedding stored as {d:?}, expected F16 or F32")),
        }
    }

    /// The projected rows of `ids`, `[n][hidden]`.
    fn rows(&self, ids: &[u32], pool: Option<&Pool>) -> Result<Vec<f32>, String> {
        let mut e = Vec::with_capacity(ids.len() * self.width);
        for &id in ids {
            e.extend(self.embed_row(id)?);
        }
        let mut h = self.fc1.apply(&e, ids.len(), pool);
        for v in &mut h {
            *v = silu(*v);
        }
        Ok(self.fc2.apply(&h, ids.len(), pool))
    }
}

// ───────────────────────────── code predictor ─────────────────────────────

/// The code predictor: a Qwen3 decoder over a frame's slots
/// `[talker hidden, code 0, code 1 …]`, each step one more code. Its layers
/// run as a pipeline of their own — the file opened again as a five-layer
/// Qwen3 (`CmfModel::open_view`) — so they reach the same device graphs as
/// the talker; its per-codebook embeddings and heads stay on the host.
pub struct CodePredictor {
    pipe: Pipeline,
    /// A frame ran: the next ones rewind the sequence instead of resetting it.
    warm: bool,
    /// `[group - 1]`: the embedding of codebook g (1..groups).
    emb: Vec<Vec<f32>>,
    /// `[group - 1]`: the head that predicts codebook g.
    heads: Vec<Lin>,
    hidden: usize,
    vocab: usize,
    pool: Option<Arc<Pool>>,
}

impl CodePredictor {
    pub fn from_model(model: &CmfModel, c: &Qwen3TtsConfig, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let cp = &c.cp;
        let (hidden, nl, nh, nkv, hd, ffn, vocab) = (
            ju(cp, "hidden_size")?,
            ju(cp, "num_hidden_layers")?,
            ju(cp, "num_attention_heads")?,
            ju(cp, "num_key_value_heads")?,
            ju(cp, "head_dim")?,
            ju(cp, "intermediate_size")?,
            ju(cp, "vocab_size")?,
        );
        if hidden != c.hidden {
            return Err("qwen3_tts: a code predictor of another width (small_to_mtp_projection) is not supported".into());
        }
        let mut arch = model.header.arch.clone();
        if (arch.num_attention_heads, arch.num_kv_heads, arch.head_dim) != (nh, nkv, hd) {
            return Err("qwen3_tts: the code predictor's heads differ from the talker's".into());
        }
        arch.num_layers = nl;
        arch.layer_types.truncate(nl);
        arch.intermediate_size = ffn;
        arch.vocab_size = vocab;
        let p = "qtts.code_predictor.";
        let rename = |n: &str| -> Option<String> {
            let r = n.strip_prefix(p)?;
            if let Some(l) = r.strip_prefix("model.layers.") {
                return Some(format!("model.layers.{l}"));
            }
            match r {
                "model.norm.weight" => Some("model.norm.weight".into()),
                // the pipeline wants an embedding and a head; rows come in
                // as embeddings and each step reads its own head below
                "model.codec_embedding.0.weight" => Some("model.embed_tokens.weight".into()),
                "lm_head.0.weight" => Some("lm_head.weight".into()),
                _ => None,
            }
        };
        let view = CmfModel::open_view(&model.path, &rename, arch).map_err(|e| format!("qwen3_tts: code predictor view: {e}"))?;
        let pipe = Pipeline::from_model(&Arc::new(view), crate::SamplerConfig::default()).map_err(|e| format!("qwen3_tts: code predictor: {e}"))?;
        let g = c.groups - 1;
        let mut emb = Vec::with_capacity(g);
        let mut heads = Vec::with_capacity(g);
        for i in 0..g {
            let e = crate::dit::cmf_f32(model, &format!("{p}model.codec_embedding.{i}.weight"))?;
            if e.len() != vocab * hidden {
                return Err(format!("{p}model.codec_embedding.{i}: {} values", e.len()));
            }
            emb.push(e);
            heads.push(Lin::load(model, &format!("{p}lm_head.{i}"), vocab, hidden, false)?);
        }
        Ok(Self { pipe, warm: false, emb, heads, hidden, vocab, pool })
    }

    /// Codebook `g`'s embedding row of `code` (g in 1..groups).
    pub fn code_row(&self, g: usize, code: u32) -> &[f32] {
        let e = &self.emb[g - 1];
        &e[code as usize * self.hidden..(code as usize + 1) * self.hidden]
    }

    /// The other codebooks of a frame: `talker_h` (the talker's
    /// final-normed state) and `code0_row` (the talker's embedding of the
    /// first code) are the prefix; each step reads its own head.
    pub fn frame(&mut self, talker_h: &[f32], code0_row: &[f32], s: &Sampling, rng: &mut SplitMix64) -> Result<Vec<u32>, String> {
        let g = self.heads.len();
        let t = std::time::Instant::now();
        // the first frame sets the sequence up; later ones rewind it (a full
        // reset frees and rebuilds the device mirrors, 20-40 ms a frame)
        let rewind = std::env::var("CMF_QTTS_REWIND").as_deref() != Ok("0");
        let mut h = if self.warm && rewind && self.pipe.speech_rewind() {
            self.pipe.speech_step(talker_h, 0);
            self.pipe.speech_step(code0_row, 1)
        } else {
            self.warm = true;
            let rows = [talker_h, code0_row].concat();
            self.pipe.speech_prefill(&[0, 0], Some(&rows))?
        };
        prof(0, t);
        let mut codes = Vec::with_capacity(g);
        for gi in 0..g {
            let t = std::time::Instant::now();
            let logits = self.heads[gi].apply(&h, 1, self.pool.as_deref());
            let code = sample(&logits, s, None, 1.0, &[], rng);
            prof(2, t);
            debug_assert!((code as usize) < self.vocab);
            codes.push(code);
            if gi + 1 == g {
                break;
            }
            let row = self.code_row(gi + 1, code).to_vec();
            let t = std::time::Instant::now();
            h = self.pipe.speech_step(&row, 2 + gi);
            prof(1, t);
        }
        Ok(codes)
    }
}

// ───────────────────────────── sampling ─────────────────────────────

/// HF's processors in their order: repetition penalty over `seen`,
/// suppressed ids, temperature, top-k, top-p; then a draw (or the argmax).
fn sample(logits: &[f32], s: &Sampling, suppress: Option<&dyn Fn(usize) -> bool>, rep: f32, seen: &[u32], rng: &mut SplitMix64) -> u32 {
    let mut l: Vec<f32> = logits.to_vec();
    if rep != 1.0 {
        let mut done = std::collections::HashSet::new();
        for &t in seen {
            if done.insert(t) {
                let v = &mut l[t as usize];
                *v = if *v < 0.0 { *v * rep } else { *v / rep };
            }
        }
    }
    if let Some(f) = suppress {
        for (i, v) in l.iter_mut().enumerate() {
            if f(i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    if !s.do_sample || s.temperature <= 0.0 {
        return l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    }
    for v in &mut l {
        *v /= s.temperature;
    }
    let mut idx: Vec<usize> = (0..l.len()).filter(|&i| l[i].is_finite()).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    if s.top_k > 0 && s.top_k < idx.len() {
        idx.truncate(s.top_k);
    }
    let mx = l[idx[0]];
    let mut w: Vec<f64> = idx.iter().map(|&i| ((l[i] - mx) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    if s.top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = w.len();
        for (j, &wi) in w.iter().enumerate() {
            cum += wi / total;
            if cum >= s.top_p as f64 {
                keep = j + 1;
                break;
            }
        }
        w.truncate(keep);
        idx.truncate(keep);
    }
    let total: f64 = w.iter().sum();
    let mut r = rng.next_f32() as f64 * total;
    for (&i, &wi) in idx.iter().zip(&w) {
        r -= wi;
        if r <= 0.0 {
            return i as u32;
        }
    }
    *idx.last().unwrap() as u32
}

// ───────────────────────────── generation ─────────────────────────────

/// What to say and how.
pub struct Request<'a> {
    pub text: &'a str,
    /// A language name (`russian`, `english` …) or `auto`.
    pub language: &'a str,
    /// A built-in speaker name (`serena` …), or empty for none.
    pub speaker: &'a str,
    pub seed: u64,
    /// At most this many frames (12.5 a second).
    pub max_frames: Option<usize>,
    /// `Some`: the talker and the code predictor at these temperatures
    /// (0 = greedy) instead of the release's.
    pub temperature: Option<f32>,
}

/// The frames `[n][groups]` of an utterance.
pub struct Speech {
    pub frames: Vec<Vec<u32>>,
}

/// Map a user language (`ru`, `Russian`, `русский`…) to the file's names.
pub fn language_name(cfg: &Qwen3TtsConfig, lang: &str) -> Option<String> {
    let l = lang.trim().to_lowercase();
    if l == "auto" || l.is_empty() {
        return Some("auto".into());
    }
    let alias = match l.as_str() {
        "zh" | "cn" => "chinese",
        "en" => "english",
        "de" => "german",
        "it" => "italian",
        "pt" => "portuguese",
        "es" => "spanish",
        "ja" | "jp" => "japanese",
        "ko" | "kr" => "korean",
        "fr" => "french",
        "ru" | "русский" => "russian",
        other => other,
    };
    cfg.languages.iter().any(|(n, _)| n == alias).then(|| alias.to_string())
}

/// Generate the frames of `req.text` with the talker `pipeline` and the
/// code predictor `cp`.
pub fn generate(
    pipeline: &mut Pipeline,
    cfg: &Qwen3TtsConfig,
    model: &Arc<CmfModel>,
    cp: &mut CodePredictor,
    req: &Request<'_>,
    pool: Option<&Pool>,
) -> Result<Speech, String> {
    let text_in = TextIn::load(model, cfg)?;
    let hs = cfg.hidden;
    if pipeline.hidden_size != hs {
        return Err(format!("qwen3_tts: talker width {} against the config's {hs}", pipeline.hidden_size));
    }
    // the assistant turn of the text (qwen_tts `_build_assistant_text`)
    let ids = pipeline.tokenizer.encode(&format!("<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n", req.text));
    if ids.len() < 9 {
        return Err("qwen3_tts: empty text".into());
    }
    let (role, text) = (&ids[..3], &ids[3..ids.len() - 5]);
    let lang = language_name(cfg, req.language).ok_or_else(|| {
        format!("qwen3_tts: language {:?} is not one of auto, {}", req.language, cfg.language_names().join(", "))
    })?;
    let spk = req.speaker.trim().to_lowercase();
    let speaker = if spk.is_empty() {
        None
    } else {
        Some(cfg.speakers.iter().find(|s| s.0 == spk).ok_or_else(|| {
            format!("qwen3_tts: speaker {:?} is not one of {}", req.speaker, cfg.speaker_names().join(", "))
        })?)
    };
    // a dialect speaker speaks its dialect when the language is Chinese or auto
    let mut lang_id = cfg.languages.iter().find(|l| l.0 == lang).map(|l| l.1);
    if let Some((_, _, Some(d))) = speaker {
        if lang == "chinese" || lang == "auto" {
            lang_id = cfg.languages.iter().find(|l| &l.0 == d).map(|l| l.1);
        }
    }
    let mut codec: Vec<u32> = match lang_id {
        None => vec![cfg.nothink, cfg.think_bos, cfg.think_eos],
        Some(l) => vec![cfg.think, cfg.think_bos, l, cfg.think_eos],
    };
    if let Some(s) = speaker {
        codec.push(s.1);
    }
    codec.extend([cfg.codec_pad, cfg.codec_bos]);
    let specials = text_in.rows(&[cfg.tts_bos, cfg.tts_eos, cfg.tts_pad], pool)?;
    let (tts_bos, tts_eos, tts_pad) = (&specials[..hs], &specials[hs..2 * hs], &specials[2 * hs..]);
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let mut rows = text_in.rows(role, pool)?;
    // tts_pad × (len - 2), tts_bos over the codec's control codes but its last
    for (j, &c) in codec[..codec.len() - 1].iter().enumerate() {
        let t = if j + 2 < codec.len() { tts_pad } else { tts_bos };
        rows.extend(add(t, &pipeline.embed_id(c)));
    }
    // non-streaming: the whole text over codec pads, the text's end, then the codec's bos
    let pad_row = pipeline.embed_id(cfg.codec_pad);
    let text_rows = text_in.rows(text, pool)?;
    for r in text_rows.chunks(hs) {
        rows.extend(add(r, &pad_row));
    }
    rows.extend(add(tts_eos, &pad_row));
    rows.extend(add(tts_pad, &pipeline.embed_id(cfg.codec_bos)));
    let n = rows.len() / hs;
    let dummy = vec![0u32; n];
    let mut h = pipeline.speech_prefill(&dummy, Some(&rows))?;
    let mut pos = n;

    let mut rng = SplitMix64::new(req.seed);
    let (mut talk, mut sub) = (cfg.talker, cfg.sub);
    if let Some(t) = req.temperature {
        for s in [&mut talk, &mut sub] {
            s.temperature = t;
            s.do_sample = t > 0.0;
        }
    }
    // the release's 8192 frames is eleven minutes: an utterance that never
    // samples its end stops after 6 s plus 0.64 s a text token instead
    let max = req.max_frames.unwrap_or_else(|| cfg.max_frames.min(75 + 8 * text.len()));
    let (vocab, eos) = (cfg.vocab, cfg.codec_eos as usize);
    let mut seen: Vec<u32> = Vec::new();
    let mut frames = Vec::new();
    let prof = std::env::var_os("CMF_QTTS_PROF").is_some();
    let (mut t_cp, mut t_lm) = (0f64, 0f64);
    for step in 0..max {
        let logits = pipeline.speech_logits(&h);
        // the codec's control range is never sampled, its end only after two frames
        let sup = |i: usize| (i >= vocab - 1024 && i != eos) || (i == eos && step < 2);
        let code0 = sample(&logits[..vocab.min(logits.len())], &talk, Some(&sup), cfg.repetition_penalty, &seen, &mut rng);
        if code0 as usize == eos {
            break;
        }
        seen.push(code0);
        let e0 = pipeline.embed_id(code0);
        let t0 = std::time::Instant::now();
        let rest = cp.frame(&h, &e0, &sub, &mut rng)?;
        t_cp += t0.elapsed().as_secs_f64();
        let mut row = e0.clone();
        for (g, &c) in rest.iter().enumerate() {
            for (a, b) in row.iter_mut().zip(cp.code_row(g + 1, c)) {
                *a += b;
            }
        }
        for (a, b) in row.iter_mut().zip(tts_pad) {
            *a += b;
        }
        let mut f = Vec::with_capacity(cfg.groups);
        f.push(code0);
        f.extend(rest);
        frames.push(f);
        let t1 = std::time::Instant::now();
        h = pipeline.speech_step(&row, pos);
        t_lm += t1.elapsed().as_secs_f64();
        pos += 1;
    }
    if prof {
        eprintln!("qwen3-tts: {} frames | code predictor {:.0} ms, talker {:.0} ms", frames.len(), t_cp * 1e3, t_lm * 1e3);
    }
    Ok(Speech { frames })
}

// ───────────────────────────── codec ─────────────────────────────

/// `[Cout][K][Cin]` taps (from the source `[Cout][Cin][K]`) of a causal
/// convolution: left padding `(K-1)·dil`, stride 1.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    dil: usize,
}

impl Conv {
    fn load(model: &CmfModel, name: &str, cin: usize, cout: usize, k: usize, dil: usize, bias: bool) -> Result<Self, String> {
        let src = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        if src.len() != cout * cin * k {
            return Err(format!("{name}.weight: {} values, expected [{cout}, {cin}, {k}]", src.len()));
        }
        let mut w = vec![0f32; src.len()];
        for o in 0..cout {
            for i in 0..cin {
                for t in 0..k {
                    w[(o * k + t) * cin + i] = src[(o * cin + i) * k + t];
                }
            }
        }
        let b = if bias { vec_n(model, &format!("{name}.bias"), cout)? } else { vec![0f32; cout] };
        Ok(Self { w, b, cin, cout, k, dil })
    }

    /// `x` `[T][cin]` → `[T][cout]`.
    fn apply(&self, x: &[f32], t: usize, pool: Option<&Pool>) -> Vec<f32> {
        let (cin, k, dil) = (self.cin, self.k, self.dil);
        let pad = (k - 1) * dil;
        let mut y = vec![0f32; t * self.cout];
        // im2col a block of rows at a time
        let block = (1usize << 22) / (k * cin).max(1);
        let block = block.clamp(1, t.max(1));
        let mut col = vec![0f32; block * k * cin];
        let mut t0 = 0;
        while t0 < t {
            let nb = block.min(t - t0);
            let cp = SendMut(col.as_mut_ptr());
            par_rows(pool, nb, &|lo, hi| {
                for r in lo..hi {
                    let tt = t0 + r;
                    for j in 0..k {
                        let src = tt as isize + (j * dil) as isize - pad as isize;
                        // SAFETY: row r of `col` belongs to this worker alone
                        let dst = unsafe { std::slice::from_raw_parts_mut(cp.get().add((r * k + j) * cin), cin) };
                        if src < 0 {
                            dst.fill(0.0);
                        } else {
                            dst.copy_from_slice(&x[src as usize * cin..(src as usize + 1) * cin]);
                        }
                    }
                }
            });
            crate::fcd_ops::gemm_nt(&col[..nb * k * cin], &self.w, &mut y[t0 * self.cout..(t0 + nb) * self.cout], nb, k * cin, self.cout, pool);
            t0 += nb;
        }
        let (cout, b) = (self.cout, &self.b);
        let yp = SendMut(y.as_mut_ptr());
        par_rows(pool, t, &|lo, hi| {
            // SAFETY: rows lo..hi of `y` belong to this worker alone
            let rows = unsafe { std::slice::from_raw_parts_mut(yp.get().add(lo * cout), (hi - lo) * cout) };
            for r in rows.chunks_mut(cout) {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        });
        y
    }
}

/// A transposed convolution of kernel `2s`, stride `s`, the right `k - s`
/// samples cut (`Qwen3TTSTokenizerV2CausalTransConvNet`): `[T][cin]` →
/// `[T·s][cout]`.
struct ConvT {
    /// `[cout·k][cin]` (rows `(o, j)`), from the source `[cin][cout][k]`.
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    s: usize,
}

impl ConvT {
    fn load(model: &CmfModel, name: &str, cin: usize, cout: usize, k: usize, s: usize) -> Result<Self, String> {
        let src = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        if src.len() != cin * cout * k {
            return Err(format!("{name}.weight: {} values, expected [{cin}, {cout}, {k}]", src.len()));
        }
        let mut w = vec![0f32; src.len()];
        for i in 0..cin {
            for o in 0..cout {
                for j in 0..k {
                    w[(o * k + j) * cin + i] = src[(i * cout + o) * k + j];
                }
            }
        }
        Ok(Self { w, b: vec_n(model, &format!("{name}.bias"), cout)?, cin, cout, k, s })
    }

    fn apply(&self, x: &[f32], t: usize, pool: Option<&Pool>) -> Vec<f32> {
        let (cout, k, s) = (self.cout, self.k, self.s);
        let full = (t - 1) * s + k;
        let keep = full - (k - s);
        let mut y = vec![0f32; keep * cout];
        let block = ((1usize << 22) / (cout * k)).clamp(1, t.max(1));
        let mut part = vec![0f32; block * cout * k];
        let mut t0 = 0;
        while t0 < t {
            let nb = block.min(t - t0);
            crate::fcd_ops::gemm_nt(&x[t0 * self.cin..(t0 + nb) * self.cin], &self.w, &mut part[..nb * cout * k], nb, self.cin, cout * k, pool);
            let (yp, part) = (SendMut(y.as_mut_ptr()), &part);
            // output channels split across workers: the overlapping taps of
            // neighbouring input rows land in one worker's columns
            par_rows(pool, cout, &|o0, o1| {
                for r in 0..nb {
                    let tt = t0 + r;
                    for o in o0..o1 {
                        for j in 0..k {
                            let at = tt * s + j;
                            if at < keep {
                                // SAFETY: column o of `y` belongs to this worker alone
                                unsafe { *yp.get().add(at * cout + o) += part[(r * cout + o) * k + j] };
                            }
                        }
                    }
                }
            });
            t0 += nb;
        }
        let (b, yp) = (&self.b, SendMut(y.as_mut_ptr()));
        par_rows(pool, keep, &|lo, hi| {
            // SAFETY: rows lo..hi of `y` belong to this worker alone
            let rows = unsafe { std::slice::from_raw_parts_mut(yp.get().add(lo * cout), (hi - lo) * cout) };
            for r in rows.chunks_mut(cout) {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        });
        y
    }
}

/// SnakeBeta with its parameters stored as logs: `x + sin²(x·e^α) / (e^β + 1e-9)`.
struct Snake {
    a: Vec<f32>,
    inv_b: Vec<f32>,
}

impl Snake {
    fn load(model: &CmfModel, name: &str, c: usize) -> Result<Self, String> {
        let a = vec_n(model, &format!("{name}.alpha"), c)?.iter().map(|v| v.exp()).collect();
        let inv_b = vec_n(model, &format!("{name}.beta"), c)?.iter().map(|v| 1.0 / (v.exp() + 1e-9)).collect();
        Ok(Self { a, inv_b })
    }

    fn apply(&self, x: &mut [f32], pool: Option<&Pool>) {
        let c = self.a.len();
        let rows = x.len() / c;
        let xp = SendMut(x.as_mut_ptr());
        par_rows(pool, rows, &|lo, hi| {
            // SAFETY: rows lo..hi of `x` belong to this worker alone
            let part = unsafe { std::slice::from_raw_parts_mut(xp.get().add(lo * c), (hi - lo) * c) };
            for r in part.chunks_mut(c) {
                for ((v, &a), &ib) in r.iter_mut().zip(&self.a).zip(&self.inv_b) {
                    let s = (*v * a).sin();
                    *v += ib * s * s;
                }
            }
        });
    }
}

struct ResUnit {
    act1: Snake,
    conv1: Conv,
    act2: Snake,
    conv2: Conv,
}

struct DecBlock {
    act: Snake,
    up: ConvT,
    units: Vec<ResUnit>,
    cout: usize,
}

struct XLayer {
    in_norm: Vec<f32>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    attn_scale: Vec<f32>,
    post_norm: Vec<f32>,
    gate: Lin,
    up: Lin,
    down: Lin,
    mlp_scale: Vec<f32>,
}

struct ConvNext {
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    pw1: Lin,
    pw2: Lin,
    gamma: Vec<f32>,
}

/// The speech tokenizer's decoder (`Qwen3TTSTokenizerV2Decoder`).
pub struct Codec {
    /// `[quantizer][code][256]`, the first the semantic one.
    books: Vec<Vec<f32>>,
    book_dim: usize,
    proj_first: Lin,
    proj_rest: Lin,
    pre_conv: Conv,
    in_proj: Lin,
    layers: Vec<XLayer>,
    norm: Vec<f32>,
    out_proj: Lin,
    ups: Vec<(ConvT, ConvNext)>,
    conv_in: Conv,
    blocks: Vec<DecBlock>,
    act_out: Snake,
    conv_out: Conv,
    hidden: usize,
    heads: usize,
    hd: usize,
    eps: f32,
    theta: f64,
    window: usize,
    pool: Option<Arc<Pool>>,
}

impl Codec {
    pub fn from_model(model: &CmfModel, cfg: &Qwen3TtsConfig, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let c = &cfg.codec;
        let (latent, cbd, cbs, dec_dim, hidden, ffn, nl, heads, hd, nq) = (
            ju(c, "latent_dim")?,
            ju(c, "codebook_dim")?,
            ju(c, "codebook_size")?,
            ju(c, "decoder_dim")?,
            ju(c, "hidden_size")?,
            ju(c, "intermediate_size")?,
            ju(c, "num_hidden_layers")?,
            ju(c, "num_attention_heads")?,
            ju(c, "head_dim")?,
            ju(c, "num_quantizers")?,
        );
        let ups: Vec<usize> = c.get("upsample_rates").and_then(|v| v.as_array()).ok_or("qwen3_tts: no upsample_rates")?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let ratios: Vec<usize> = c.get("upsampling_ratios").and_then(|v| v.as_array()).ok_or("qwen3_tts: no upsampling_ratios")?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        if ups.iter().chain(&ratios).product::<usize>() != FRAME_SAMPLES {
            return Err("qwen3_tts: the decoder does not upsample 1920×".into());
        }
        let p = "qtts.codec.decoder";
        let bd = cbd / 2;
        let book = |pre: &str, i: usize| -> Result<Vec<f32>, String> {
            let sum = crate::dit::cmf_f32(model, &format!("{p}.quantizer.{pre}.vq.layers.{i}._codebook.embedding_sum"))?;
            let usage = vec_n(model, &format!("{p}.quantizer.{pre}.vq.layers.{i}._codebook.cluster_usage"), cbs)?;
            if sum.len() != cbs * bd {
                return Err(format!("qwen3_tts: codebook {pre}.{i} has {} values", sum.len()));
            }
            Ok(sum.chunks(bd).zip(&usage).flat_map(|(r, &u)| r.iter().map(move |v| v / u.max(1e-5))).collect())
        };
        let mut books = vec![book("rvq_first", 0)?];
        for i in 0..nq - 1 {
            books.push(book("rvq_rest", i)?);
        }
        let proj = |pre: &str| -> Result<Lin, String> {
            let w = crate::dit::cmf_f32(model, &format!("{p}.quantizer.{pre}.output_proj.weight"))?;
            if w.len() != cbd * bd {
                return Err(format!("qwen3_tts: {pre}.output_proj has {} values", w.len()));
            }
            Ok(Lin { w, b: None, rows: cbd, cols: bd })
        };
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let b = format!("{p}.pre_transformer.layers.{l}");
            layers.push(XLayer {
                in_norm: vec_n(model, &format!("{b}.input_layernorm.weight"), hidden)?,
                q: Lin::load(model, &format!("{b}.self_attn.q_proj"), heads * hd, hidden, false)?,
                k: Lin::load(model, &format!("{b}.self_attn.k_proj"), heads * hd, hidden, false)?,
                v: Lin::load(model, &format!("{b}.self_attn.v_proj"), heads * hd, hidden, false)?,
                o: Lin::load(model, &format!("{b}.self_attn.o_proj"), hidden, heads * hd, false)?,
                attn_scale: vec_n(model, &format!("{b}.self_attn_layer_scale.scale"), hidden)?,
                post_norm: vec_n(model, &format!("{b}.post_attention_layernorm.weight"), hidden)?,
                gate: Lin::load(model, &format!("{b}.mlp.gate_proj"), ffn, hidden, false)?,
                up: Lin::load(model, &format!("{b}.mlp.up_proj"), ffn, hidden, false)?,
                down: Lin::load(model, &format!("{b}.mlp.down_proj"), hidden, ffn, false)?,
                mlp_scale: vec_n(model, &format!("{b}.mlp_layer_scale.scale"), hidden)?,
            });
        }
        let mut upsample = Vec::new();
        for (i, &f) in ratios.iter().enumerate() {
            let b = format!("{p}.upsample.{i}");
            let dw_raw = crate::dit::cmf_f32(model, &format!("{b}.1.dwconv.conv.weight"))?;
            if dw_raw.len() != latent * 7 {
                return Err(format!("qwen3_tts: {b} dwconv has {} values", dw_raw.len()));
            }
            upsample.push((
                ConvT::load(model, &format!("{b}.0.conv"), latent, latent, f, f)?,
                ConvNext {
                    dw: dw_raw,
                    dw_b: vec_n(model, &format!("{b}.1.dwconv.conv.bias"), latent)?,
                    ln_w: vec_n(model, &format!("{b}.1.norm.weight"), latent)?,
                    ln_b: vec_n(model, &format!("{b}.1.norm.bias"), latent)?,
                    pw1: Lin::load(model, &format!("{b}.1.pwconv1"), 4 * latent, latent, true)?,
                    pw2: Lin::load(model, &format!("{b}.1.pwconv2"), latent, 4 * latent, true)?,
                    gamma: vec_n(model, &format!("{b}.1.gamma"), latent)?,
                },
            ));
        }
        let mut blocks = Vec::new();
        for (i, &r) in ups.iter().enumerate() {
            let (cin, cout) = (dec_dim >> i, dec_dim >> (i + 1));
            let b = format!("{p}.decoder.{}.block", i + 1);
            let mut units = Vec::new();
            for (u, dil) in [1usize, 3, 9].into_iter().enumerate() {
                let ub = format!("{b}.{}", u + 2);
                units.push(ResUnit {
                    act1: Snake::load(model, &format!("{ub}.act1"), cout)?,
                    conv1: Conv::load(model, &format!("{ub}.conv1.conv"), cout, cout, 7, dil, true)?,
                    act2: Snake::load(model, &format!("{ub}.act2"), cout)?,
                    conv2: Conv::load(model, &format!("{ub}.conv2.conv"), cout, cout, 1, 1, true)?,
                });
            }
            blocks.push(DecBlock {
                act: Snake::load(model, &format!("{b}.0"), cin)?,
                up: ConvT::load(model, &format!("{b}.1.conv"), cin, cout, 2 * r, r)?,
                units,
                cout,
            });
        }
        let out_dim = dec_dim >> ups.len();
        let last = ups.len() + 1;
        Ok(Self {
            books,
            book_dim: bd,
            proj_first: proj("rvq_first")?,
            proj_rest: proj("rvq_rest")?,
            pre_conv: Conv::load(model, &format!("{p}.pre_conv.conv"), cbd, latent, 3, 1, true)?,
            in_proj: Lin::load(model, &format!("{p}.pre_transformer.input_proj"), hidden, latent, true)?,
            layers,
            norm: vec_n(model, &format!("{p}.pre_transformer.norm.weight"), hidden)?,
            out_proj: Lin::load(model, &format!("{p}.pre_transformer.output_proj"), latent, hidden, true)?,
            ups: upsample,
            conv_in: Conv::load(model, &format!("{p}.decoder.0.conv"), latent, dec_dim, 7, 1, true)?,
            blocks,
            act_out: Snake::load(model, &format!("{p}.decoder.{last}"), out_dim)?,
            conv_out: Conv::load(model, &format!("{p}.decoder.{}.conv", last + 1), out_dim, 1, 7, 1, true)?,
            hidden,
            heads,
            hd,
            eps: jf(c, "rms_norm_eps", 1e-5) as f32,
            theta: jf(c, "rope_theta", 1e4),
            window: c.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(72) as usize,
            pool,
        })
    }

    /// One forward over `frames` (positions from 0): 1920 samples a frame.
    fn forward(&self, frames: &[Vec<u32>]) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let t = frames.len();
        let bd = self.book_dim;
        let t_stage = std::time::Instant::now();
        // RVQ: the semantic codebook and the sum of the acoustic ones, each projected
        let mut first = vec![0f32; t * bd];
        let mut rest = vec![0f32; t * bd];
        for (f, fr) in frames.iter().enumerate() {
            for (q, &code) in fr.iter().enumerate().take(self.books.len()) {
                let row = &self.books[q][code as usize * bd..(code as usize + 1) * bd];
                let dst = if q == 0 { &mut first[f * bd..(f + 1) * bd] } else { &mut rest[f * bd..(f + 1) * bd] };
                for (d, &v) in dst.iter_mut().zip(row) {
                    *d += v;
                }
            }
        }
        let mut x = self.proj_first.apply(&first, t, pool);
        for (a, b) in x.iter_mut().zip(self.proj_rest.apply(&rest, t, pool)) {
            *a += b;
        }
        let x = self.pre_conv.apply(&x, t, pool);
        prof(3, t_stage);
        let t_stage = std::time::Instant::now();
        // the transformer, causal within the window
        let mut h = self.in_proj.apply(&x, t, pool);
        let (hs, nh, hd) = (self.hidden, self.heads, self.hd);
        let scale = 1.0 / (hd as f32).sqrt();
        for l in &self.layers {
            let xn: Vec<f32> = h.chunks(hs).flat_map(|r| rms(r, &l.in_norm, self.eps)).collect();
            let mut q = l.q.apply(&xn, t, pool);
            let mut k = l.k.apply(&xn, t, pool);
            let v = l.v.apply(&xn, t, pool);
            for r in 0..t {
                for hh in 0..nh {
                    rope(&mut q[(r * nh + hh) * hd..(r * nh + hh + 1) * hd], r, self.theta);
                    rope(&mut k[(r * nh + hh) * hd..(r * nh + hh + 1) * hd], r, self.theta);
                }
            }
            let mut att = vec![0f32; t * nh * hd];
            let w = self.window;
            let f = |lo: usize, hi: usize, att: &mut [f32]| {
                for r in lo..hi {
                    let j0 = (r + 1).saturating_sub(w);
                    for hh in 0..nh {
                        let qv = &q[(r * nh + hh) * hd..(r * nh + hh + 1) * hd];
                        let mut sc: Vec<f32> = (j0..=r)
                            .map(|j| qv.iter().zip(&k[(j * nh + hh) * hd..(j * nh + hh + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() * scale)
                            .collect();
                        let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut z = 0f32;
                        for s in &mut sc {
                            *s = (*s - mx).exp();
                            z += *s;
                        }
                        let out = &mut att[((r - lo) * nh + hh) * hd..((r - lo) * nh + hh + 1) * hd];
                        for (jj, &p) in sc.iter().enumerate() {
                            let vv = &v[((j0 + jj) * nh + hh) * hd..((j0 + jj) * nh + hh + 1) * hd];
                            for (o, &b) in out.iter_mut().zip(vv) {
                                *o += p / z * b;
                            }
                        }
                    }
                }
            };
            match pool {
                Some(p) if t > 8 => {
                    let ap = SendMut(att.as_mut_ptr());
                    p.run_rows(t, &|lo, hi| {
                        // SAFETY: rows lo..hi of `att` belong to this worker alone
                        let sl = unsafe { std::slice::from_raw_parts_mut(ap.get().add(lo * nh * hd), (hi - lo) * nh * hd) };
                        f(lo, hi, sl);
                    });
                }
                _ => f(0, t, &mut att),
            }
            let o = l.o.apply(&att, t, pool);
            for (r, orow) in h.chunks_mut(hs).zip(o.chunks(hs)) {
                for ((a, &b), &s) in r.iter_mut().zip(orow).zip(&l.attn_scale) {
                    *a += s * b;
                }
            }
            let xn: Vec<f32> = h.chunks(hs).flat_map(|r| rms(r, &l.post_norm, self.eps)).collect();
            let g = l.gate.apply(&xn, t, pool);
            let u = l.up.apply(&xn, t, pool);
            let m: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
            let d = l.down.apply(&m, t, pool);
            for (r, drow) in h.chunks_mut(hs).zip(d.chunks(hs)) {
                for ((a, &b), &s) in r.iter_mut().zip(drow).zip(&l.mlp_scale) {
                    *a += s * b;
                }
            }
        }
        let hn: Vec<f32> = h.chunks(hs).flat_map(|r| rms(r, &self.norm, self.eps)).collect();
        let mut x = self.out_proj.apply(&hn, t, pool);
        prof(4, t_stage);
        let t_stage = std::time::Instant::now();
        let mut len = t;
        // two upsamplings, each a ConvNeXt block after it
        for (up, cn) in &self.ups {
            x = up.apply(&x, len, pool);
            len *= up.s;
            x = cn.apply(&x, len, pool);
        }
        prof(5, t_stage);
        let t_stage = std::time::Instant::now();
        let mut x = self.conv_in.apply(&x, len, pool);
        prof(6, t_stage);
        for (bi, b) in self.blocks.iter().enumerate() {
            let t_stage = std::time::Instant::now();
            b.act.apply(&mut x, pool);
            x = b.up.apply(&x, len, pool);
            len *= b.up.s;
            for u in &b.units {
                let mut y = x.clone();
                u.act1.apply(&mut y, pool);
                let mut y = u.conv1.apply(&y, len, pool);
                u.act2.apply(&mut y, pool);
                let y = u.conv2.apply(&y, len, pool);
                for (a, b2) in x.iter_mut().zip(&y) {
                    *a += b2;
                }
            }
            debug_assert_eq!(x.len(), len * b.cout);
            prof(7 + bi.min(3), t_stage);
        }
        let t_stage = std::time::Instant::now();
        self.act_out.apply(&mut x, pool);
        let mut wav = self.conv_out.apply(&x, len, pool);
        for v in &mut wav {
            *v = v.clamp(-1.0, 1.0);
        }
        prof(11, t_stage);
        wav
    }

    /// The waveform of `frames`, decoded 300 frames at a time over 25 frames
    /// of left context (`chunked_decode`).
    pub fn decode(&self, frames: &[Vec<u32>]) -> Vec<f32> {
        const CHUNK: usize = 300;
        const LEFT: usize = 25;
        let mut out = Vec::with_capacity(frames.len() * FRAME_SAMPLES);
        let mut start = 0;
        while start < frames.len() {
            let end = (start + CHUNK).min(frames.len());
            let ctx = if start > LEFT { LEFT } else { start };
            let w = self.forward(&frames[start - ctx..end]);
            out.extend_from_slice(&w[ctx * FRAME_SAMPLES..]);
            start = end;
        }
        out
    }
}

impl ConvNext {
    fn apply(&self, x: &[f32], t: usize, pool: Option<&Pool>) -> Vec<f32> {
        let c = self.ln_w.len();
        // causal depthwise k7
        let mut d = vec![0f32; t * c];
        let dp = SendMut(d.as_mut_ptr());
        par_rows(pool, t, &|lo, hi| {
            for r in lo..hi {
                // SAFETY: row r of `d` belongs to this worker alone
                let row = unsafe { std::slice::from_raw_parts_mut(dp.get().add(r * c), c) };
                for (ch, out) in row.iter_mut().enumerate() {
                    let mut acc = self.dw_b[ch];
                    for j in 0..7 {
                        let src = r as isize + j as isize - 6;
                        if src >= 0 {
                            acc += self.dw[ch * 7 + j] * x[src as usize * c + ch];
                        }
                    }
                    *out = acc;
                }
            }
        });
        for row in d.chunks_mut(c) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / c as f64;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for ((v, &w), &b) in row.iter_mut().zip(&self.ln_w).zip(&self.ln_b) {
                *v = ((*v as f64 - mean) * inv) as f32 * w + b;
            }
        }
        let mut m = self.pw1.apply(&d, t, pool);
        for v in &mut m {
            *v = gelu_erf(*v);
        }
        let y = self.pw2.apply(&m, t, pool);
        let mut out = x.to_vec();
        for (orow, yrow) in out.chunks_mut(c).zip(y.chunks(c)) {
            for ((o, &v), &g) in orow.iter_mut().zip(yrow).zip(&self.gamma) {
                *o += g * v;
            }
        }
        out
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SendMut(pub(crate) *mut f32);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}
impl SendMut {
    pub(crate) fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sampler is greedy without sampling and honours suppression and
    /// the repetition penalty in HF's order.
    #[test]
    fn greedy_sample_suppresses_and_penalizes() {
        let mut rng = SplitMix64::new(1);
        let g = Sampling { do_sample: false, temperature: 0.9, top_k: 50, top_p: 1.0 };
        let l = [0.0f32, 3.0, 2.9, 1.0];
        assert_eq!(sample(&l, &g, None, 1.0, &[], &mut rng), 1);
        assert_eq!(sample(&l, &g, Some(&|i| i == 1), 1.0, &[], &mut rng), 2);
        // 3.0 / 1.05 < 2.9: the repeated code yields
        assert_eq!(sample(&l, &g, None, 1.05, &[1], &mut rng), 2);
    }

    /// RoPE keeps the norm and rotates the pair (j, j + half).
    #[test]
    fn rope_rotates_halves() {
        let mut x = vec![1.0f32, 0.0, 0.0, 0.0];
        rope(&mut x, 1, 10000.0);
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[2] - 1f32.sin()).abs() < 1e-6);
    }
}
