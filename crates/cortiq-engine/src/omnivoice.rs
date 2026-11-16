//! OmniVoice (k2-fsa/OmniVoice): zero-shot text-to-speech in 600+
//! languages as a diffusion language model.
//!
//! A Qwen3 stack runs BOTH ways over one row: style tokens (language,
//! voice-design instruct), the text, an optional reference clip's audio
//! codes and the target's codes, every target code starting as the mask
//! id. An audio position's row is the sum of its 8 codebooks' embeddings;
//! the audio heads predict all 8 codebooks at every position. Over
//! `num_step` passes (32) the most confident masked codes are fixed on a
//! shifted time schedule, earlier codebooks first (a penalty per codebook
//! level), the conditional logits guided against an unconditional pass that
//! sees the target codes alone. The Higgs audio tokenizer v2 decodes the 8
//! codes of a frame (25 a second) to 960 samples at 24 kHz: residual
//! codebooks, a projection and a DAC decoder.
//!
//! The conditional and unconditional rows go through one masked forward
//! (`Pipeline::forward_masked_head`, d1's set forward) as two groups that
//! never see each other; RoPE is relative, so the unconditional group's
//! later row positions change nothing. The reference is `omnivoice` 0.2.1
//! (`models/omnivoice.py`, `utils/{duration,audio,text}.py`).

use crate::audiovae::{Conv1d, ConvT1d, Snake};
use crate::pipeline::masked::SetMask;
use crate::pipeline::Pipeline;
use crate::pool::Pool;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::Arc;

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;

/// Whether the file is an OmniVoice conversion.
pub fn is_omnivoice(model: &CmfModel) -> bool {
    model.header.provenance.as_ref().is_some_and(|p| p.get("omnivoice").is_some())
}

/// The checkpoint's and the audio tokenizer's configuration.
#[derive(Clone, Debug)]
pub struct OmniVoiceConfig {
    /// Audio codebooks (8).
    pub books: usize,
    /// Ids per codebook, the mask id included (1025).
    pub vocab: usize,
    pub mask_id: u32,
    pub sample_rate: usize,
    /// Samples per frame (960).
    pub hop: usize,
    /// The acoustic decoder's upsampling strides.
    pub upsampling: Vec<usize>,
}

impl OmniVoiceConfig {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let p = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("omnivoice"))
            .ok_or("omnivoice: the file has no provenance.omnivoice")?;
        let c = p.get("config").ok_or("omnivoice: provenance has no config")?;
        let k = p.get("codec").ok_or("omnivoice: provenance has no codec config")?;
        let u = |v: &Value, key: &str, d: usize| v.get(key).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let ac = k.get("acoustic_model_config").cloned().unwrap_or(Value::Null);
        let upsampling: Vec<usize> = ac
            .get("upsampling_ratios")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
            .unwrap_or_else(|| vec![8, 5, 4, 2, 3]);
        let cfg = Self {
            books: u(c, "num_audio_codebook", 8),
            vocab: u(c, "audio_vocab_size", 1025),
            mask_id: u(c, "audio_mask_id", 1024) as u32,
            sample_rate: u(k, "sample_rate", 24_000),
            hop: upsampling.iter().product(),
            upsampling,
        };
        if cfg.books == 0 || cfg.mask_id as usize >= cfg.vocab || cfg.hop == 0 {
            return Err(format!("omnivoice: implausible config {cfg:?}"));
        }
        Ok(cfg)
    }

    /// Frames a second (25).
    pub fn frame_rate(&self) -> f64 {
        self.sample_rate as f64 / self.hop as f64
    }
}

/// The release's generation defaults (`OmniVoiceGenerationConfig`).
#[derive(Clone, Debug)]
pub struct GenConfig {
    pub steps: usize,
    pub guidance: f32,
    pub t_shift: f32,
    pub layer_penalty: f32,
    /// Gumbel temperature on which positions unmask (0: by confidence alone).
    pub position_temperature: f32,
    /// Gumbel temperature on the codes (0: argmax).
    pub class_temperature: f32,
    pub denoise: bool,
    pub postprocess: bool,
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            steps: 32,
            guidance: 2.0,
            t_shift: 0.1,
            layer_penalty: 5.0,
            position_temperature: 5.0,
            class_temperature: 0.0,
            denoise: true,
            postprocess: true,
            seed: 0,
        }
    }
}

/// A reference voice: its codes `[books][frames]`, transcript and loudness.
#[derive(Clone, Debug)]
pub struct VoicePrompt {
    pub codes: Vec<Vec<u32>>,
    pub text: String,
    pub rms: f32,
}

pub struct Request<'a> {
    pub text: &'a str,
    /// A language code or name; None for language-agnostic.
    pub language: Option<&'a str>,
    /// Voice design: "female, low pitch, british accent", "男，四川话", …
    pub instruct: Option<&'a str>,
    pub voice: Option<&'a VoicePrompt>,
    /// Seconds of speech (overrides `speed`).
    pub duration: Option<f32>,
    pub speed: Option<f32>,
    pub generation: GenConfig,
}

// ───────────────────────────── text side ─────────────────────────────

/// `_resolve_language`: a known code as is, a name to its code, else None.
pub fn resolve_language(lang: Option<&str>) -> Option<String> {
    let l = lang?.trim();
    if l.is_empty() || l.eq_ignore_ascii_case("none") {
        return None;
    }
    let langs = crate::omnivoice_langs::LANGS;
    if langs.iter().any(|(_, id)| *id == l) {
        return Some(l.to_string());
    }
    let key = l.to_lowercase();
    if let Some((_, id)) = langs.iter().find(|(n, _)| *n == key) {
        return Some(id.to_string());
    }
    tracing::warn!("omnivoice: language {l:?} is not recognized; speaking without a language id");
    None
}

const CATEGORIES_EN_ZH: [&[(&str, &str)]; 4] = [
    &[("male", "男"), ("female", "女")],
    &[("child", "儿童"), ("teenager", "少年"), ("young adult", "青年"), ("middle-aged", "中年"), ("elderly", "老年")],
    &[
        ("very low pitch", "极低音调"),
        ("low pitch", "低音调"),
        ("moderate pitch", "中音调"),
        ("high pitch", "高音调"),
        ("very high pitch", "极高音调"),
    ],
    &[("whisper", "耳语")],
];
const ACCENTS: [&str; 10] = [
    "american accent",
    "british accent",
    "australian accent",
    "chinese accent",
    "canadian accent",
    "indian accent",
    "korean accent",
    "portuguese accent",
    "russian accent",
    "japanese accent",
];
const DIALECTS: [&str; 12] =
    ["河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"];

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// `_resolve_instruct`: validate a voice-design string and write it in one
/// language (Chinese when `use_zh`, a dialect forces Chinese, an accent
/// English), at most one item per category.
pub fn resolve_instruct(instruct: Option<&str>, use_zh: bool) -> Result<Option<String>, String> {
    let Some(s) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let valid = |n: &str| {
        CATEGORIES_EN_ZH.iter().any(|c| c.iter().any(|(e, z)| *e == n || *z == n)) || ACCENTS.contains(&n) || DIALECTS.contains(&n)
    };
    let mut items = Vec::new();
    let mut bad = Vec::new();
    for raw in s.split([',', '，']).map(str::trim).filter(|x| !x.is_empty()) {
        let n = raw.to_lowercase();
        if valid(&n) {
            items.push(n);
        } else {
            bad.push(raw.to_string());
        }
    }
    if !bad.is_empty() {
        let mut en: Vec<&str> = CATEGORIES_EN_ZH.iter().flat_map(|c| c.iter().map(|(e, _)| *e)).chain(ACCENTS).collect();
        en.sort_unstable();
        return Err(format!(
            "omnivoice: unsupported instruct items {bad:?}; English items: {}; Chinese items use a full-width comma (男，河南话)",
            en.join(", ")
        ));
    }
    let dialect = items.iter().any(|n| n.ends_with('话'));
    let accent = items.iter().any(|n| n.contains(" accent"));
    if dialect && accent {
        return Err("omnivoice: a Chinese dialect and an English accent cannot be mixed in one instruct".into());
    }
    let zh = if dialect { true } else if accent { false } else { use_zh };
    let items: Vec<String> = items
        .into_iter()
        .map(|n| {
            for c in CATEGORIES_EN_ZH {
                for (e, z) in c {
                    if zh && n == *e {
                        return z.to_string();
                    }
                    if !zh && n == *z {
                        return e.to_string();
                    }
                }
            }
            n
        })
        .collect();
    let mut groups: Vec<Vec<&str>> = CATEGORIES_EN_ZH.iter().map(|c| c.iter().flat_map(|(e, z)| [*e, *z]).collect()).collect();
    groups.push(ACCENTS.to_vec());
    groups.push(DIALECTS.to_vec());
    for g in &groups {
        let hits: Vec<&String> = items.iter().filter(|n| g.contains(&n.as_str())).collect();
        if hits.len() > 1 {
            return Err(format!("omnivoice: conflicting instruct items {hits:?}: one per category (gender, age, pitch, style, accent, dialect)"));
        }
    }
    let has_zh = items.iter().any(|n| n.chars().any(is_cjk));
    Ok(Some(items.join(if has_zh { "，" } else { ", " })))
}

/// `_combine_text`: reference transcript and text, newlines dropped,
/// Chinese parentheses made ASCII, blank runs collapsed, no blanks beside
/// Chinese characters.
pub fn combine_text(text: &str, ref_text: Option<&str>) -> String {
    let full = match ref_text.filter(|r| !r.is_empty()) {
        Some(r) => format!("{} {}", r.trim(), text.trim()),
        None => text.trim().to_string(),
    };
    let full: String = full.chars().filter(|&c| c != '\r' && c != '\n').map(|c| match c {
        '\u{ff08}' => '(',
        '\u{ff09}' => ')',
        c => c,
    }).collect();
    // [ \t]+ → " "
    let mut collapsed = String::with_capacity(full.len());
    let mut blank = false;
    for c in full.chars() {
        if c == ' ' || c == '\t' {
            if !blank {
                collapsed.push(' ');
            }
            blank = true;
        } else {
            collapsed.push(c);
            blank = false;
        }
    }
    // (?<=CJK)\s+ | \s+(?=CJK) removed
    let chars: Vec<char> = collapsed.chars().collect();
    let mut out = String::with_capacity(collapsed.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let before = i > 0 && is_cjk(chars[i - 1]);
            let after = j < chars.len() && is_cjk(chars[j]);
            if !before && !after {
                out.extend(&chars[i..j]);
            }
            i = j;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

const NONVERBAL: [&str; 13] = [
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
];

/// `_tokenize_with_nonverbal_tags`: every `[laughter]`-style tag is its own
/// tokenizer call, so its ids never depend on the text around it.
fn tokenize_with_tags(pipeline: &Pipeline, text: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut last = 0;
    let mut i = 0;
    let b = text.as_bytes();
    while i < b.len() {
        if b[i] == b'[' {
            if let Some(tag) = NONVERBAL.iter().find(|t| text[i + 1..].starts_with(*t) && text[i + 1 + t.len()..].starts_with(']')) {
                let end = i + tag.len() + 2;
                if i > last {
                    ids.extend(pipeline.tokenizer.encode(&text[last..i]));
                }
                ids.extend(pipeline.tokenizer.encode(&text[i..end]));
                last = end;
                i = end;
                continue;
            }
        }
        i += 1;
    }
    if last < text.len() {
        ids.extend(pipeline.tokenizer.encode(&text[last..]));
    }
    ids
}

/// `add_punctuation`: a final full stop when the text ends without one.
pub fn add_punctuation(text: &str) -> String {
    let t = text.trim();
    let Some(last) = t.chars().last() else {
        return String::new();
    };
    const END: &str = ";:,.!?…)]}\"'“”‘’；：，。！？、）】";
    if END.contains(last) {
        t.to_string()
    } else if t.chars().any(is_cjk) {
        format!("{t}。")
    } else {
        format!("{t}.")
    }
}

// ───────────────────────────── duration ─────────────────────────────

/// `RuleDurationEstimator`'s per-character weight.
fn char_weight(c: char) -> f64 {
    use unicode_categories::UnicodeCategories;
    let code = c as u32;
    if c.is_ascii_alphabetic() {
        return 1.0;
    }
    if code == 32 {
        return 0.2;
    }
    if code == 0x0640 || c.is_mark() {
        return 0.0;
    }
    if c.is_punctuation() || c.is_symbol() {
        return 0.5;
    }
    if c.is_separator() {
        return 0.2;
    }
    if c.is_number() {
        return 3.5;
    }
    // (upper bound, weight): the first bound at or past the code point
    const RANGES: &[(u32, f64)] = &[
        (0x02AF, 1.0),
        (0x03FF, 1.0),
        (0x052F, 1.0),
        (0x058F, 1.0),
        (0x05FF, 1.5),
        (0x077F, 1.5),
        (0x089F, 1.5),
        (0x08FF, 1.5),
        (0x097F, 1.8),
        (0x09FF, 1.8),
        (0x0A7F, 1.8),
        (0x0AFF, 1.8),
        (0x0B7F, 1.8),
        (0x0BFF, 1.8),
        (0x0C7F, 1.8),
        (0x0CFF, 1.8),
        (0x0D7F, 1.8),
        (0x0DFF, 1.8),
        (0x0EFF, 1.5),
        (0x0FFF, 1.8),
        (0x109F, 1.8),
        (0x10FF, 1.0),
        (0x11FF, 2.5),
        (0x137F, 3.0),
        (0x139F, 3.0),
        (0x13FF, 1.0),
        (0x167F, 1.0),
        (0x169F, 1.0),
        (0x16FF, 1.0),
        (0x171F, 1.0),
        (0x173F, 1.0),
        (0x175F, 1.0),
        (0x177F, 1.0),
        (0x17FF, 1.8),
        (0x18AF, 1.0),
        (0x18FF, 1.0),
        (0x194F, 1.8),
        (0x19DF, 1.8),
        (0x19FF, 1.8),
        (0x1A1F, 1.8),
        (0x1AAF, 1.8),
        (0x1B7F, 1.8),
        (0x1BBF, 1.8),
        (0x1BFF, 1.8),
        (0x1C4F, 1.8),
        (0x1C7F, 1.8),
        (0x1C8F, 1.0),
        (0x1CBF, 1.0),
        (0x1CCF, 1.8),
        (0x1CFF, 1.8),
        (0x1D7F, 1.0),
        (0x1DBF, 1.0),
        (0x1DFF, 1.0),
        (0x1EFF, 1.0),
        (0x309F, 2.2),
        (0x30FF, 2.2),
        (0x312F, 3.0),
        (0x318F, 2.5),
        (0x9FFF, 3.0),
        (0xA4CF, 3.0),
        (0xA4FF, 1.0),
        (0xA63F, 1.0),
        (0xA69F, 1.0),
        (0xA6FF, 1.0),
        (0xA7FF, 1.0),
        (0xA82F, 1.8),
        (0xA87F, 1.0),
        (0xA8DF, 1.8),
        (0xA8FF, 1.8),
        (0xA92F, 1.8),
        (0xA95F, 1.8),
        (0xA97F, 2.5),
        (0xA9DF, 1.8),
        (0xA9FF, 1.8),
        (0xAA5F, 1.8),
        (0xAA7F, 1.8),
        (0xAADF, 1.8),
        (0xAAFF, 1.8),
        (0xAB2F, 3.0),
        (0xAB6F, 1.0),
        (0xABBF, 1.0),
        (0xABFF, 1.8),
        (0xD7AF, 2.5),
        (0xFAFF, 3.0),
        (0xFDFF, 1.5),
        (0xFE6F, 1.0),
        (0xFEFF, 1.5),
        (0xFFEF, 1.0),
    ];
    let idx = RANGES.partition_point(|&(b, _)| b < code);
    if let Some(&(_, w)) = RANGES.get(idx) {
        return w;
    }
    if code > 0x20000 {
        return 3.0;
    }
    1.0
}

fn text_weight(t: &str) -> f64 {
    t.chars().map(char_weight).sum()
}

/// `RuleDurationEstimator.estimate_duration` (in frames when `ref_frames`
/// is): the reference's speaking rate, with short estimates boosted on a
/// power curve below 50.
fn estimate_duration(target: &str, ref_text: &str, ref_frames: f64) -> f64 {
    if ref_frames <= 0.0 || ref_text.is_empty() {
        return 0.0;
    }
    let rw = text_weight(ref_text);
    if rw == 0.0 {
        return 0.0;
    }
    let est = text_weight(target) / (rw / ref_frames);
    const LOW: f64 = 50.0;
    if est < LOW {
        LOW * (est / LOW).powf(1.0 / 3.0)
    } else {
        est
    }
}

/// `_estimate_target_tokens`: frames for `text` at the reference's rate
/// ("Nice to meet you." in 25 frames without one).
pub fn estimate_frames(text: &str, ref_text: Option<&str>, ref_frames: Option<usize>, speed: f32) -> usize {
    let (rt, rf) = match (ref_text.filter(|r| !r.is_empty()), ref_frames) {
        (Some(t), Some(f)) => (t, f as f64),
        _ => ("Nice to meet you.", 25.0),
    };
    let mut est = estimate_duration(text, rt, rf);
    if speed > 0.0 && speed != 1.0 {
        est /= speed as f64;
    }
    (est as i64).max(1) as usize
}

// ───────────────────────────── long text ─────────────────────────────

const ABBREVIATIONS: [&str; 52] = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.", "Capt.", "Gen.", "Sen.",
    "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.",
    "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.",
    "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.",
];

/// `chunk_text_punctuation`: sentences at punctuation (not after an
/// abbreviation), packed into chunks of at most `chunk_len` characters, a
/// chunk shorter than `min_len` merged into its neighbour.
pub fn chunk_text_punctuation(text: &str, chunk_len: usize, min_len: usize) -> Vec<String> {
    const SPLIT: &str = ".,;:!?。，；：！？";
    const CLOSING: &str = "\"'“”‘’）]》>」】";
    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    for ch in text.chars() {
        if cur.is_empty() && !sentences.is_empty() && (SPLIT.contains(ch) || CLOSING.contains(ch)) {
            sentences.last_mut().unwrap().push(ch);
            continue;
        }
        cur.push(ch);
        if SPLIT.contains(ch) {
            let mut abbr = false;
            if ch == '.' {
                let t: String = cur.iter().collect();
                if let Some(w) = t.split_whitespace().last() {
                    abbr = ABBREVIATIONS.contains(&w) || w == "def.";
                }
            }
            if !abbr {
                sentences.push(std::mem::take(&mut cur));
            }
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    let mut merged: Vec<Vec<char>> = Vec::new();
    let mut chunk: Vec<char> = Vec::new();
    for sen in sentences {
        if chunk.len() + sen.len() <= chunk_len {
            chunk.extend(sen);
        } else {
            if !chunk.is_empty() {
                merged.push(chunk);
            }
            chunk = sen;
        }
    }
    if !chunk.is_empty() {
        merged.push(chunk);
    }
    let first_short = merged.first().is_some_and(|c| c.len() < min_len);
    let mut fin: Vec<Vec<char>> = Vec::new();
    for (i, c) in merged.into_iter().enumerate() {
        if i == 1 && first_short {
            fin.last_mut().unwrap().extend(c);
        } else if c.len() >= min_len || fin.is_empty() {
            fin.push(c);
        } else {
            fin.last_mut().unwrap().extend(c);
        }
    }
    fin.into_iter().map(|c| c.into_iter().collect::<String>().trim().to_string()).filter(|s| !s.is_empty()).collect()
}

/// `cross_fade_chunks`: chunks joined by 0.1 s of silence with 0.1 s fades
/// on both sides of every join.
pub fn cross_fade(chunks: &[Vec<f32>], rate: usize) -> Vec<f32> {
    let Some(first) = chunks.first() else { return Vec::new() };
    let fade_n = (0.3 * rate as f64) as usize / 3;
    let mut out = first.clone();
    for c in &chunks[1..] {
        let k = fade_n.min(out.len());
        let n = out.len();
        for i in 0..k {
            let w = if k == 1 { 1.0 } else { 1.0 - i as f32 / (k - 1) as f32 };
            out[n - k + i] *= w;
        }
        out.extend(std::iter::repeat_n(0f32, fade_n));
        let mut nx = c.clone();
        let k = fade_n.min(nx.len());
        for (i, v) in nx.iter_mut().take(k).enumerate() {
            *v *= if k == 1 { 0.0 } else { i as f32 / (k - 1) as f32 };
        }
        out.extend(nx);
    }
    out
}

/// Seconds of estimated speech past which a request goes in chunks
/// (`audio_chunk_threshold`), and the chunks' length (`audio_chunk_duration`).
pub const CHUNK_THRESHOLD_S: f64 = 30.0;
pub const CHUNK_S: f64 = 15.0;

/// `_generate_chunked` for one request: the text in chunks of ~15 s, each
/// with the reference voice, or (without one) the first chunk's own speech
/// as the reference of the rest. The codes of every chunk, in order; one
/// chunk when the estimate is under the threshold.
pub fn generate_chunks(
    pipeline: &Pipeline,
    model: &Arc<CmfModel>,
    cfg: &OmniVoiceConfig,
    req: &Request,
    pool: Option<&Pool>,
) -> Result<Vec<Vec<Vec<u32>>>, String> {
    let speed = req.speed.unwrap_or(1.0);
    let est = estimate_frames(req.text, req.voice.map(|v| v.text.as_str()), req.voice.map(|v| v.codes[0].len()), speed);
    if req.duration.is_some() || (est as f64) <= CHUNK_THRESHOLD_S * cfg.frame_rate() {
        return Ok(vec![generate(pipeline, model, cfg, req, pool)?]);
    }
    let per_char = est as f64 / req.text.chars().count().max(1) as f64;
    let chunk_len = (CHUNK_S * cfg.frame_rate() / per_char) as usize;
    let chunks = chunk_text_punctuation(req.text, chunk_len.max(1), 3);
    let mut out = Vec::with_capacity(chunks.len());
    let mut first: Option<VoicePrompt> = None;
    for (i, t) in chunks.iter().enumerate() {
        let voice = match (req.voice, &first) {
            (Some(v), _) => Some(v),
            (None, Some(f)) => Some(f),
            (None, None) => None,
        };
        let r = Request { text: t, language: req.language, instruct: req.instruct, voice, duration: None, speed: req.speed, generation: req.generation.clone() };
        let codes = generate(pipeline, model, cfg, &r, pool)?;
        if i == 0 && req.voice.is_none() {
            // the first chunk speaks for the rest: its codes and text as the reference
            // (no loudness: the release keeps the first chunk's RMS unset)
            first = Some(VoicePrompt { codes: codes.clone(), text: t.clone(), rms: f32::NAN });
        }
        out.push(codes);
    }
    Ok(out)
}

// ───────────────────────────── generation ─────────────────────────────

/// `_get_time_steps` in float32, as the reference computes it.
fn time_steps(steps: usize, shift: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            shift * t / (1.0 + (shift - 1.0) * t)
        })
        .collect()
}

/// Codes unmasked at each step: the shifted schedule's share of `total`.
fn schedule(total: usize, steps: usize, shift: f32) -> Vec<usize> {
    let ts = time_steps(steps, shift);
    let mut rem = total;
    (0..steps)
        .map(|s| {
            let n = if s + 1 == steps {
                rem
            } else {
                ((total as f64 * (ts[s + 1] as f64 - ts[s] as f64)).ceil() as usize).min(rem)
            };
            rem -= n;
            n
        })
        .collect()
}

fn log_softmax(x: &[f32], out: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = x.iter().map(|&v| ((v - m) as f64).exp()).sum();
    let l = m + s.ln() as f32;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v - l;
    }
}

fn gumbel(rng: &mut SplitMix64) -> f32 {
    let u = rng.next_f32();
    -(-(u + 1e-10).ln() + 1e-10).ln()
}

/// The text embedding (`ov.text_embedding`): rows of the Qwen3 vocabulary.
struct TextEmbedding {
    model: Arc<CmfModel>,
    idx: usize,
    width: usize,
}

impl TextEmbedding {
    fn load(model: &Arc<CmfModel>, width: usize) -> Result<Self, String> {
        let name = "ov.text_embedding.weight";
        let idx = model.tensor_index(name).ok_or("omnivoice: no ov.text_embedding.weight")?;
        let e = &model.tensors[idx];
        if e.shape.len() != 2 || e.shape[1] != width {
            return Err(format!("{name}: shape {:?}, expected [_, {width}]", e.shape));
        }
        Ok(Self { model: model.clone(), idx, width })
    }

    fn row(&self, id: u32) -> Result<Vec<f32>, String> {
        let e = &self.model.tensors[self.idx];
        if id as usize >= e.shape[0] {
            return Err(format!("omnivoice: text token {id} past the embedding's {} rows", e.shape[0]));
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
            d => Err(format!("omnivoice: text embedding stored as {d:?}, expected F16 or F32")),
        }
    }
}

/// Per-step timings (`CMF_OV_PROF=1`).
fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_OV_PROF").is_some())
}

/// The speech prompt of a request: its row layout and target length.
pub struct Prompt {
    /// Row ids per codebook `[books][c_len]`; text rows repeat one id.
    pub ids: Vec<Vec<u32>>,
    /// Rows from here on are audio (the reference's codes, then the target).
    pub audio_start: usize,
    /// Target frames.
    pub frames: usize,
}

/// `_prepare_inference_inputs` (plus `_preprocess_all`'s language, instruct
/// and length handling) for one request.
pub fn build_prompt(pipeline: &Pipeline, cfg: &OmniVoiceConfig, req: &Request) -> Result<Prompt, String> {
    if req.text.trim().is_empty() {
        return Err("omnivoice: empty text".into());
    }
    let lang = resolve_language(req.language);
    let instruct = resolve_instruct(req.instruct, req.text.chars().any(is_cjk))?;
    let voice = req.voice;
    if let Some(v) = voice {
        if v.codes.len() != cfg.books || v.codes.iter().any(|c| c.len() != v.codes[0].len()) {
            return Err(format!("omnivoice: a reference needs {} codebooks of one length", cfg.books));
        }
    }
    let ref_frames = voice.map(|v| v.codes[0].len());
    let est = estimate_frames(req.text, voice.map(|v| v.text.as_str()), ref_frames, if req.duration.is_some() { 1.0 } else { req.speed.unwrap_or(1.0) });
    let frames = match req.duration {
        Some(d) => ((d as f64 * cfg.frame_rate()) as i64).max(1) as usize,
        None => est,
    };
    let mut style = String::new();
    if req.generation.denoise && voice.is_some() {
        style.push_str("<|denoise|>");
    }
    style.push_str(&format!(
        "<|lang_start|>{}<|lang_end|><|instruct_start|>{}<|instruct_end|>",
        lang.as_deref().unwrap_or("None"),
        instruct.as_deref().unwrap_or("None")
    ));
    let mut text_ids = pipeline.tokenizer.encode(&style);
    let full = combine_text(req.text, voice.map(|v| v.text.as_str()));
    text_ids.extend(tokenize_with_tags(pipeline, &format!("<|text_start|>{full}<|text_end|>")));
    let audio_start = text_ids.len();
    let mut ids: Vec<Vec<u32>> = (0..cfg.books).map(|_| text_ids.clone()).collect();
    for (c, row) in ids.iter_mut().enumerate() {
        if let Some(v) = voice {
            row.extend(&v.codes[c]);
        }
        row.extend(std::iter::repeat_n(cfg.mask_id, frames));
    }
    Ok(Prompt { ids, audio_start, frames })
}

/// `_generate_iterative` for one request: the target's codes `[books][frames]`.
pub fn generate(
    pipeline: &Pipeline,
    model: &Arc<CmfModel>,
    cfg: &OmniVoiceConfig,
    req: &Request,
    pool: Option<&Pool>,
) -> Result<Vec<Vec<u32>>, String> {
    let prompt = build_prompt(pipeline, cfg, req)?;
    if let Ok(f) = std::env::var("CMF_OV_PROMPT_DUMP") {
        let _ = std::fs::write(&f, serde_json::to_vec(&serde_json::json!({"ids": prompt.ids, "audio_start": prompt.audio_start, "frames": prompt.frames})).unwrap_or_default());
    }
    generate_prompt(pipeline, model, cfg, &prompt, &req.generation, pool)
}

pub fn generate_prompt(
    pipeline: &Pipeline,
    model: &Arc<CmfModel>,
    cfg: &OmniVoiceConfig,
    prompt: &Prompt,
    gc: &GenConfig,
    pool: Option<&Pool>,
) -> Result<Vec<Vec<u32>>, String> {
    let h = pipeline.hidden_size;
    let (books, v) = (cfg.books, cfg.vocab);
    if pipeline.vocab_size != books * v {
        return Err(format!("omnivoice: head width {} against {books} × {v}", pipeline.vocab_size));
    }
    let text = TextEmbedding::load(model, h)?;
    // The audio heads as f32 (f16 in the file): the target rows' logits are
    // one device GEMM (`gpu::gemm_nt_f32`, the weight cached on the card)
    // over the forward's hidden rows, or the host GEMM.
    let heads = crate::dit::cmf_f32(model, "lm_head.weight")?;
    if heads.len() != books * v * h {
        return Err(format!("omnivoice: lm_head holds {} values, expected {} × {h}", heads.len(), books * v));
    }
    let c_len = prompt.ids[0].len();
    let t_len = prompt.frames;
    let t0 = c_len - t_len;
    let audio_row = |codes: &[u32]| -> Vec<f32> {
        let mut r = vec![0f32; h];
        for (c, &id) in codes.iter().enumerate() {
            for (a, b) in r.iter_mut().zip(pipeline.embed_id((c * v) as u32 + id)) {
                *a += b;
            }
        }
        r
    };
    let col = |p: usize| -> Vec<u32> { prompt.ids.iter().map(|r| r[p]).collect() };
    let n = c_len + t_len;
    let mut rows = vec![0f32; n * h];
    for p in 0..c_len {
        let r = if p < prompt.audio_start { text.row(prompt.ids[0][p])? } else { audio_row(&col(p)) };
        rows[p * h..(p + 1) * h].copy_from_slice(&r);
    }
    // the unconditional group: the target rows alone
    for t in 0..t_len {
        let (a, b) = rows.split_at_mut(c_len * h);
        b[t * h..(t + 1) * h].copy_from_slice(&a[(t0 + t) * h..(t0 + t + 1) * h]);
    }
    let group: Vec<i32> = (0..n).map(|p| i32::from(p >= c_len)).collect();
    let head_rows: Vec<usize> = (t0..c_len).chain(c_len..n).collect();
    let mut tokens = vec![vec![cfg.mask_id; t_len]; books];
    let sched = schedule(t_len * books, gc.steps, gc.t_shift);
    let mut rng = SplitMix64::new(gc.seed);
    let vv = books * v;
    let mut c_lp = vec![0f32; v];
    let mut u_lp = vec![0f32; v];
    let mut comb = vec![0f32; v];
    let mut lp = vec![0f32; v];
    for (step, &k) in sched.iter().enumerate() {
        if k == 0 {
            continue;
        }
        let t_fwd = std::time::Instant::now();
        let (hidden, _) = pipeline.forward_masked_head(&rows, &SetMask { state: 0, group: &group }, &[])?;
        let ms_fwd = t_fwd.elapsed().as_secs_f64() * 1e3;
        let t_head = std::time::Instant::now();
        let x: Vec<f32> = head_rows.iter().flat_map(|&r| hidden[r * h..(r + 1) * h].iter().copied()).collect();
        let nr = head_rows.len();
        let mut logits = vec![0f32; nr * books * v];
        if !crate::gpu::gemm_nt_f32(&x, &heads, &mut logits, nr, h, books * v) {
            crate::fcd_ops::gemm_nt(&x, &heads, &mut logits, nr, h, books * v, pool);
        }
        let ms_head = t_head.elapsed().as_secs_f64() * 1e3;
        if logits.len() < 2 * t_len * vv {
            return Err("omnivoice: the head returned too few rows".into());
        }
        if step == 0 {
            if let Ok(f) = std::env::var("CMF_OV_STEP0_DUMP") {
                let bytes: Vec<u8> = logits[..2 * t_len * vv].iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(&f, bytes);
            }
        }
        let mut pred = vec![0u32; books * t_len];
        let mut score = vec![f32::NEG_INFINITY; books * t_len];
        for c in 0..books {
            for t in 0..t_len {
                if tokens[c][t] != cfg.mask_id {
                    continue;
                }
                let cl = &logits[t * vv + c * v..t * vv + (c + 1) * v];
                let ul = &logits[(t_len + t) * vv + c * v..(t_len + t) * vv + (c + 1) * v];
                if gc.guidance != 0.0 {
                    log_softmax(cl, &mut c_lp);
                    log_softmax(ul, &mut u_lp);
                    for i in 0..v {
                        comb[i] = c_lp[i] + gc.guidance * (c_lp[i] - u_lp[i]);
                    }
                    log_softmax(&comb, &mut lp);
                } else {
                    log_softmax(cl, &mut lp);
                }
                lp[cfg.mask_id as usize] = f32::NEG_INFINITY;
                let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
                for (i, &x) in lp.iter().enumerate() {
                    if x > best {
                        best = x;
                        bi = i;
                    }
                }
                let id = if gc.class_temperature > 0.0 {
                    // `_filter_top_k` (10 %) then a Gumbel draw
                    let keep = (0.1 * v as f64).ceil() as usize;
                    let mut order: Vec<usize> = (0..v).collect();
                    order.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
                    let mut top = (f32::NEG_INFINITY, bi);
                    for &i in &order[..keep] {
                        let s = lp[i] / gc.class_temperature + gumbel(&mut rng);
                        if s > top.0 {
                            top = (s, i);
                        }
                    }
                    top.1
                } else {
                    bi
                };
                pred[c * t_len + t] = id as u32;
                let mut s = best - c as f32 * gc.layer_penalty;
                if gc.position_temperature > 0.0 {
                    s = s / gc.position_temperature + gumbel(&mut rng);
                }
                score[c * t_len + t] = s;
            }
        }
        let mut order: Vec<usize> = (0..books * t_len).filter(|&i| score[i] > f32::NEG_INFINITY).collect();
        order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        let mut touched = vec![false; t_len];
        for &i in order.iter().take(k) {
            let (c, t) = (i / t_len, i % t_len);
            tokens[c][t] = pred[i];
            touched[t] = true;
        }
        for t in (0..t_len).filter(|&t| touched[t]) {
            let codes: Vec<u32> = (0..books).map(|c| tokens[c][t]).collect();
            let r = audio_row(&codes);
            rows[(t0 + t) * h..(t0 + t + 1) * h].copy_from_slice(&r);
            rows[(c_len + t) * h..(c_len + t + 1) * h].copy_from_slice(&r);
        }
        if prof_on() {
            eprintln!("omnivoice: step {step}: {n} rows forward {ms_fwd:.1} ms, head {ms_head:.1} ms, {k} codes fixed");
        }
    }
    Ok(tokens)
}

// ───────────────────────────── audio tokenizer ─────────────────────────────

struct Unit {
    s1: Snake,
    c1: Conv1d,
    s2: Snake,
    c2: Conv1d,
}

impl Unit {
    fn load(model: &Arc<CmfModel>, p: &str, dilation: usize) -> Result<Self, String> {
        Ok(Self {
            s1: Snake::load(model, &format!("{p}.snake1"))?,
            c1: Conv1d::load(model, &format!("{p}.conv1"), 3 * dilation, dilation)?,
            s2: Snake::load(model, &format!("{p}.snake2"))?,
            c2: Conv1d::load(model, &format!("{p}.conv2"), 0, 1)?,
        })
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut h = x.to_vec();
        self.s1.apply_pool(&mut h, n, pool);
        let mut h = self.c1.apply(&h, n, pool);
        self.s2.apply_pool(&mut h, n, pool);
        let mut r = self.c2.apply(&h, n, pool);
        for (a, &b) in r.iter_mut().zip(x) {
            *a += b;
        }
        r
    }
}

struct Block {
    snake: Snake,
    up: ConvT1d,
    units: Vec<Unit>,
}

/// The Higgs audio tokenizer v2's decoding half: residual codebooks with
/// their output projections, `fc2`, and the DAC decoder (no final tanh,
/// transposed convolutions padded to exactly `stride` samples a step).
pub struct Codec {
    /// `[books][1024 × dim]` codebook rows.
    embed: Vec<Vec<f32>>,
    dim: usize,
    /// `[books]` project_out `[hidden × dim]` and its bias.
    proj: Vec<(Vec<f32>, Vec<f32>)>,
    hidden: usize,
    fc2: (Vec<f32>, Vec<f32>),
    acoustic: usize,
    conv1: Conv1d,
    blocks: Vec<Block>,
    snake: Snake,
    conv2: Conv1d,
    pool: Option<Arc<Pool>>,
}

impl Codec {
    pub fn from_model(model: &Arc<CmfModel>, cfg: &OmniVoiceConfig, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let f = |n: &str| crate::dit::cmf_f32(model, n);
        let shape = |n: &str| model.tensor(n).map(|e| e.shape.clone()).ok_or_else(|| format!("omnivoice: missing {n}"));
        let q = "ov.codec.quantizer.quantizers";
        let es = shape(&format!("{q}.0.codebook.embed"))?;
        let dim = es[1];
        let ps = shape(&format!("{q}.0.project_out.weight"))?;
        let hidden = ps[0];
        let mut embed = Vec::with_capacity(cfg.books);
        let mut proj = Vec::with_capacity(cfg.books);
        for c in 0..cfg.books {
            embed.push(f(&format!("{q}.{c}.codebook.embed"))?);
            proj.push((f(&format!("{q}.{c}.project_out.weight"))?, f(&format!("{q}.{c}.project_out.bias"))?));
        }
        let fs = shape("ov.codec.fc2.weight")?;
        let acoustic = fs[0];
        let d = "ov.codec.acoustic_decoder";
        let blocks = cfg
            .upsampling
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                Ok(Block {
                    snake: Snake::load(model, &format!("{d}.block.{i}.snake1"))?,
                    up: ConvT1d::load_padded(model, &format!("{d}.block.{i}.conv_t1"), s, s.div_ceil(2), s % 2)?,
                    units: [1usize, 3, 9]
                        .iter()
                        .enumerate()
                        .map(|(j, &dl)| Unit::load(model, &format!("{d}.block.{i}.res_unit{}", j + 1), dl))
                        .collect::<Result<_, String>>()?,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            embed,
            dim,
            proj,
            hidden,
            fc2: (f("ov.codec.fc2.weight")?, f("ov.codec.fc2.bias")?),
            acoustic,
            conv1: Conv1d::load(model, &format!("{d}.conv1"), 3, 1)?,
            blocks,
            snake: Snake::load(model, &format!("{d}.snake1"))?,
            conv2: Conv1d::load(model, &format!("{d}.conv2"), 3, 1)?,
            pool,
        })
    }

    /// Codes `[books][frames]` → mono samples at 24 kHz (`frames × hop`).
    pub fn decode(&self, codes: &[Vec<u32>]) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let frames = codes.first().map_or(0, |c| c.len());
        if frames == 0 {
            return Vec::new();
        }
        let (hd, dim) = (self.hidden, self.dim);
        // Σ_c project_out_c(codebook_c[code])
        let mut q = vec![0f32; frames * hd];
        for (c, cb) in codes.iter().enumerate() {
            let mut e = vec![0f32; frames * dim];
            for (t, &id) in cb.iter().enumerate() {
                let id = (id as usize).min(self.embed[c].len() / dim - 1);
                e[t * dim..(t + 1) * dim].copy_from_slice(&self.embed[c][id * dim..(id + 1) * dim]);
            }
            let mut y = vec![0f32; frames * hd];
            crate::fcd_ops::gemm_nt(&e, &self.proj[c].0, &mut y, frames, dim, hd, pool);
            for (row, qr) in y.chunks_exact(hd).zip(q.chunks_exact_mut(hd)) {
                for ((a, &b), &bias) in qr.iter_mut().zip(row).zip(&self.proj[c].1) {
                    *a += b + bias;
                }
            }
        }
        let ac = self.acoustic;
        let mut z = vec![0f32; frames * ac];
        crate::fcd_ops::gemm_nt(&q, &self.fc2.0, &mut z, frames, hd, ac, pool);
        // [frames, ac] + bias → channel-major [ac, frames]
        let mut x = vec![0f32; ac * frames];
        for t in 0..frames {
            for ch in 0..ac {
                x[ch * frames + t] = z[t * ac + ch] + self.fc2.1[ch];
            }
        }
        let prof = prof_on();
        let t = std::time::Instant::now();
        let mut h = self.conv1.apply(&x, frames, pool);
        let mut n = frames;
        let mut last = t.elapsed().as_secs_f64();
        if prof {
            eprintln!("[ov codec] codebooks + conv1: {:.3} s", last);
        }
        for (bi, b) in self.blocks.iter().enumerate() {
            b.snake.apply_pool(&mut h, n, pool);
            h = if std::env::var("CMF_OV_CONVT_REF").is_ok() { b.up.apply(&h, n, pool) } else { b.up.apply_gemm(&h, n, pool) };
            n *= b.up.stride;
            let up = t.elapsed().as_secs_f64();
            for u in &b.units {
                h = u.apply(&h, n, pool);
            }
            let now = t.elapsed().as_secs_f64();
            if prof {
                eprintln!("[ov codec] block {bi}: {} ch × {n}: up {:.3} s, units {:.3} s", h.len() / n, up - last, now - up);
            }
            last = now;
        }
        self.snake.apply_pool(&mut h, n, pool);
        let y = self.conv2.apply(&h, n, pool);
        if prof {
            eprintln!("[ov codec] conv2: {:.3} s, total {:.3} s", t.elapsed().as_secs_f64() - last, t.elapsed().as_secs_f64());
        }
        y
    }
}

// ───────────────────────────── voice cloning ─────────────────────────────

/// A 1-D convolution with stride, padding, dilation and groups as
/// im2col + the host GEMM, the time axis in tiles: `[in, n]` → `[out, n_out]`.
struct SConv {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    out: usize,
    inp: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
    groups: usize,
}

impl SConv {
    fn new(w: Vec<f32>, b: Option<Vec<f32>>, shape: &[usize], stride: usize, pad: usize, dil: usize, groups: usize) -> Result<Self, String> {
        if shape.len() != 3 || w.len() != shape.iter().product::<usize>() {
            return Err(format!("omnivoice: conv weight of shape {shape:?}"));
        }
        Ok(Self { w, b, out: shape[0], inp: shape[1] * groups, k: shape[2], stride, pad, dil, groups })
    }

    fn load(model: &CmfModel, name: &str, stride: usize, pad: usize, dil: usize, groups: usize) -> Result<Self, String> {
        let shape = model.tensor(&format!("{name}.weight")).map(|e| e.shape.clone()).ok_or_else(|| format!("omnivoice: missing {name}.weight"))?;
        let w = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        let b = crate::dit::cmf_f32(model, &format!("{name}.bias")).ok();
        Self::new(w, b, &shape, stride, pad, dil, groups)
    }

    fn out_len(&self, n: usize) -> usize {
        (n + 2 * self.pad).saturating_sub(self.dil * (self.k - 1) + 1) / self.stride + 1
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> (Vec<f32>, usize) {
        let n_out = self.out_len(n);
        let (gi, go) = (self.inp / self.groups, self.out / self.groups);
        let kk = gi * self.k;
        let mut y = vec![0f32; self.out * n_out];
        const TILE: usize = 1 << 24; // floats of one column buffer
        let span = (TILE / kk.max(1)).clamp(1, n_out.max(1));
        for g in 0..self.groups {
            let wg = &self.w[g * go * kk..(g + 1) * go * kk];
            let mut t0 = 0;
            while t0 < n_out {
                let m = span.min(n_out - t0);
                let mut col = vec![0f32; m * kk];
                for (r, row) in col.chunks_exact_mut(kk).enumerate() {
                    let t = t0 + r;
                    for c in 0..gi {
                        let src = &x[(g * gi + c) * n..(g * gi + c + 1) * n];
                        for j in 0..self.k {
                            let p = (t * self.stride + j * self.dil) as isize - self.pad as isize;
                            if p >= 0 && (p as usize) < n {
                                row[c * self.k + j] = src[p as usize];
                            }
                        }
                    }
                }
                let mut yt = vec![0f32; m * go];
                crate::fcd_ops::gemm_nt(&col, wg, &mut yt, m, kk, go, pool);
                for o in 0..go {
                    let bias = self.b.as_ref().map_or(0.0, |b| b[g * go + o]);
                    let dst = &mut y[(g * go + o) * n_out + t0..(g * go + o) * n_out + t0 + m];
                    for (r, d) in dst.iter_mut().enumerate() {
                        *d = yt[r * go + o] + bias;
                    }
                }
                t0 += m;
            }
        }
        (y, n_out)
    }
}

fn elu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = v.exp_m1();
        }
    }
}

fn layer_norm_rows(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let d = w.len();
    for row in x.chunks_exact_mut(d) {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let r = 1.0 / (var + eps as f64).sqrt();
        for ((v, &g), &bb) in row.iter_mut().zip(w).zip(b) {
            *v = ((*v as f64 - mean) * r) as f32 * g + bb;
        }
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

struct HubertLayer {
    q: crate::qwen3_tts::Lin,
    k: crate::qwen3_tts::Lin,
    v: crate::qwen3_tts::Lin,
    o: crate::qwen3_tts::Lin,
    ln: (Vec<f32>, Vec<f32>),
    ff1: crate::qwen3_tts::Lin,
    ff2: crate::qwen3_tts::Lin,
    ln2: (Vec<f32>, Vec<f32>),
}

/// The semantic model: HuBERT base (a 7-layer convolutional front end with a
/// group norm on the first, a projection, a grouped positional convolution
/// and 12 post-norm transformer layers), every hidden state averaged.
struct Hubert {
    convs: Vec<SConv>,
    gn: (Vec<f32>, Vec<f32>),
    proj_ln: (Vec<f32>, Vec<f32>),
    proj: crate::qwen3_tts::Lin,
    pos: SConv,
    enc_ln: (Vec<f32>, Vec<f32>),
    layers: Vec<HubertLayer>,
    heads: usize,
}

impl Hubert {
    fn load(model: &CmfModel, sm: &Value) -> Result<Self, String> {
        let p = "ov.codec.semantic_model";
        let f = |n: &str| crate::dit::cmf_f32(model, n);
        let ua = |k: &str| -> Vec<usize> {
            sm.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).unwrap_or_default()
        };
        let (kernels, strides) = (ua("conv_kernel"), ua("conv_stride"));
        let hs = sm.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as usize;
        let heads = sm.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
        let nl = sm.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
        let ffn = sm.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(3072) as usize;
        let pk = sm.get("num_conv_pos_embeddings").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
        let pg = sm.get("num_conv_pos_embedding_groups").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
        if kernels.is_empty() || kernels.len() != strides.len() {
            return Err("omnivoice: the semantic model's config lacks conv_kernel / conv_stride".into());
        }
        let convs = kernels
            .iter()
            .zip(&strides)
            .enumerate()
            .map(|(i, (&k, &s))| {
                let c = SConv::load(model, &format!("{p}.feature_extractor.conv_layers.{i}.conv"), s, 0, 1, 1)?;
                if c.k != k {
                    return Err(format!("omnivoice: HuBERT conv {i} has kernel {}, config {k}", c.k));
                }
                Ok(c)
            })
            .collect::<Result<Vec<_>, String>>()?;
        // the positional convolution's weight norm (dim 2): w = g · v / ‖v‖ per tap
        let g = f(&format!("{p}.encoder.pos_conv_embed.conv.parametrizations.weight.original0"))?;
        let vname = format!("{p}.encoder.pos_conv_embed.conv.parametrizations.weight.original1");
        let vshape = model.tensor(&vname).map(|e| e.shape.clone()).ok_or("omnivoice: missing the positional conv weight")?;
        let mut v = f(&vname)?;
        let (o_, i_, k_) = (vshape[0], vshape[1], vshape[2]);
        if g.len() != k_ || k_ != pk {
            return Err("omnivoice: positional conv weight norm of another shape".into());
        }
        for kk in 0..k_ {
            let mut ss = 0f64;
            for a in 0..o_ * i_ {
                ss += (v[a * k_ + kk] as f64).powi(2);
            }
            let s = g[kk] as f64 / ss.sqrt().max(1e-12);
            for a in 0..o_ * i_ {
                v[a * k_ + kk] = (v[a * k_ + kk] as f64 * s) as f32;
            }
        }
        let pos = SConv::new(v, f(&format!("{p}.encoder.pos_conv_embed.conv.bias")).ok(), &vshape, 1, pk / 2, 1, pg)?;
        let ln = |n: &str| -> Result<(Vec<f32>, Vec<f32>), String> { Ok((f(&format!("{n}.weight"))?, f(&format!("{n}.bias"))?)) };
        let lin = |n: &str, r: usize, c: usize| crate::qwen3_tts::Lin::load(model, n, r, c, true);
        let cd = convs.last().map_or(512, |c| c.out);
        let layers = (0..nl)
            .map(|l| {
                let q = format!("{p}.encoder.layers.{l}");
                Ok(HubertLayer {
                    q: lin(&format!("{q}.attention.q_proj"), hs, hs)?,
                    k: lin(&format!("{q}.attention.k_proj"), hs, hs)?,
                    v: lin(&format!("{q}.attention.v_proj"), hs, hs)?,
                    o: lin(&format!("{q}.attention.out_proj"), hs, hs)?,
                    ln: ln(&format!("{q}.layer_norm"))?,
                    ff1: lin(&format!("{q}.feed_forward.intermediate_dense"), ffn, hs)?,
                    ff2: lin(&format!("{q}.feed_forward.output_dense"), hs, ffn)?,
                    ln2: ln(&format!("{q}.final_layer_norm"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            convs,
            gn: ln(&format!("{p}.feature_extractor.conv_layers.0.layer_norm"))?,
            proj_ln: ln(&format!("{p}.feature_projection.layer_norm"))?,
            proj: lin(&format!("{p}.feature_projection.projection"), hs, cd)?,
            pos,
            enc_ln: ln(&format!("{p}.encoder.layer_norm"))?,
            layers,
            heads,
        })
    }

    /// 16 kHz samples → the mean of every hidden state `[frames, hidden]`.
    fn features(&self, wav16: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        let gelu = |x: &mut [f32]| x.iter_mut().for_each(|v| *v = crate::inference::gelu_erf(*v));
        let mut h = wav16.to_vec();
        let mut n = h.len();
        for (i, c) in self.convs.iter().enumerate() {
            let (y, m) = c.apply(&h, n, pool);
            h = y;
            n = m;
            if i == 0 {
                // GroupNorm with one channel a group: each channel over time
                for (ch, row) in h.chunks_exact_mut(n).enumerate() {
                    let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                    let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
                    let r = 1.0 / (var + 1e-5).sqrt();
                    for v in row.iter_mut() {
                        *v = ((*v as f64 - mean) * r) as f32 * self.gn.0[ch] + self.gn.1[ch];
                    }
                }
            }
            gelu(&mut h);
        }
        let cd = self.convs.last().map_or(0, |c| c.out);
        let mut x = transpose(&h, cd, n); // [n, cd]
        layer_norm_rows(&mut x, &self.proj_ln.0, &self.proj_ln.1, 1e-5);
        let mut hid = self.proj.apply(&x, n, pool);
        let hs = hid.len() / n.max(1);
        // positional convolution over [hs, n], its last frame dropped, GELU
        let (mut pe, m) = self.pos.apply(&transpose(&hid, n, hs), n, pool);
        let pe = {
            gelu(&mut pe);
            let mut keep = vec![0f32; hs * n];
            for c in 0..hs {
                keep[c * n..(c + 1) * n].copy_from_slice(&pe[c * m..c * m + n]);
            }
            transpose(&keep, hs, n)
        };
        for (a, b) in hid.iter_mut().zip(&pe) {
            *a += b;
        }
        layer_norm_rows(&mut hid, &self.enc_ln.0, &self.enc_ln.1, 1e-5);
        let mut sum: Vec<f64> = hid.iter().map(|&v| v as f64).collect();
        let hd = hs / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for l in &self.layers {
            let q = l.q.apply(&hid, n, pool);
            let k = l.k.apply(&hid, n, pool);
            let v = l.v.apply(&hid, n, pool);
            let mut att = vec![0f32; n * hs];
            for hh in 0..self.heads {
                let pick = |x: &[f32]| -> Vec<f32> { (0..n).flat_map(|t| x[t * hs + hh * hd..t * hs + (hh + 1) * hd].to_vec()).collect() };
                let (qh, kh, vh) = (pick(&q), pick(&k), pick(&v));
                let mut sc = vec![0f32; n * n];
                crate::fcd_ops::gemm_nt(&qh, &kh, &mut sc, n, hd, n, pool);
                for row in sc.chunks_exact_mut(n) {
                    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b * scale));
                    let mut s = 0f64;
                    for x in row.iter_mut() {
                        *x = (*x * scale - m).exp();
                        s += *x as f64;
                    }
                    let inv = (1.0 / s) as f32;
                    row.iter_mut().for_each(|x| *x *= inv);
                }
                let mut oh = vec![0f32; n * hd];
                crate::fcd_ops::gemm_nt(&sc, &transpose(&vh, n, hd), &mut oh, n, n, hd, pool);
                for t in 0..n {
                    att[t * hs + hh * hd..t * hs + (hh + 1) * hd].copy_from_slice(&oh[t * hd..(t + 1) * hd]);
                }
            }
            let o = l.o.apply(&att, n, pool);
            for (a, b) in hid.iter_mut().zip(&o) {
                *a += b;
            }
            layer_norm_rows(&mut hid, &l.ln.0, &l.ln.1, 1e-5);
            let mut f1 = l.ff1.apply(&hid, n, pool);
            gelu(&mut f1);
            let f2 = l.ff2.apply(&f1, n, pool);
            for (a, b) in hid.iter_mut().zip(&f2) {
                *a += b;
            }
            layer_norm_rows(&mut hid, &l.ln2.0, &l.ln2.1, 1e-5);
            for (s, &v) in sum.iter_mut().zip(&hid) {
                *s += v as f64;
            }
        }
        let k = (self.layers.len() + 1) as f64;
        sum.iter().map(|&s| (s / k) as f32).collect()
    }
}

struct EncBlock {
    units: Vec<Unit>,
    snake: Snake,
    down: SConv,
}

/// The Higgs audio tokenizer v2's encoding half, for a reference voice:
/// HuBERT features (every second frame) through the semantic encoder, the
/// DAC encoder over the 24 kHz samples, `fc`, and the residual codebooks.
pub struct Encoder {
    hubert: Hubert,
    sem_conv: SConv,
    /// `[block][unit]` (conv k3, conv k1, no biases) and the block's conv.
    sem_blocks: Vec<(Vec<(SConv, SConv)>, SConv)>,
    ac_conv1: SConv,
    ac_blocks: Vec<EncBlock>,
    ac_snake: Snake,
    ac_conv2: SConv,
    fc: crate::qwen3_tts::Lin,
    /// `[books]` project_in `[dim × hidden]` + bias, codebook rows, their norms.
    rvq: Vec<(crate::qwen3_tts::Lin, Vec<f32>, Vec<f32>, crate::qwen3_tts::Lin)>,
    dim: usize,
    hidden: usize,
    cfg: OmniVoiceConfig,
    sem_rate: u32,
    sem_down: usize,
    pool: Option<Arc<Pool>>,
}

impl Encoder {
    pub fn from_model(model: &Arc<CmfModel>, cfg: &OmniVoiceConfig, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let k = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("omnivoice"))
            .and_then(|p| p.get("codec"))
            .cloned()
            .ok_or("omnivoice: provenance has no codec config")?;
        let sm = k.get("semantic_model_config").cloned().unwrap_or(Value::Null);
        let hubert = Hubert::load(model, &sm)?;
        let u = |v: &Value, key: &str, d: usize| v.get(key).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let ks = u(&k, "kernel_size", 3);
        let uks = u(&k, "unit_kernel_size", 3);
        let ua = |key: &str, d: Vec<usize>| -> Vec<usize> {
            k.get(key).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).unwrap_or(d)
        };
        let (strides, dil) = (ua("strides", vec![1, 1]), ua("block_dilations", vec![1, 1]));
        let s = "ov.codec.encoder_semantic";
        let sem_conv = SConv::load(model, &format!("{s}.conv"), 1, ks / 2, 1, 1)?;
        let sem_blocks = strides
            .iter()
            .enumerate()
            .map(|(i, &st)| {
                let units = dil
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| {
                        Ok((
                            SConv::load(model, &format!("{s}.conv_blocks.{i}.res_units.{j}.conv1"), 1, ((uks - 1) / 2) * d, d, 1)?,
                            SConv::load(model, &format!("{s}.conv_blocks.{i}.res_units.{j}.conv2"), 1, 0, 1, 1)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let kern = if st == 1 { 3 } else { 2 * st };
                Ok((units, SConv::load(model, &format!("{s}.conv_blocks.{i}.conv"), st, (kern - 1) / 2, 1, 1)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let ac = k.get("acoustic_model_config").cloned().unwrap_or(Value::Null);
        let down: Vec<usize> = ac
            .get("downsampling_ratios")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
            .unwrap_or_else(|| vec![8, 5, 4, 2, 3]);
        let a = "ov.codec.acoustic_encoder";
        let ac_blocks = down
            .iter()
            .enumerate()
            .map(|(i, &st)| {
                Ok(EncBlock {
                    units: [1usize, 3, 9]
                        .iter()
                        .enumerate()
                        .map(|(j, &d)| Unit::load(model, &format!("{a}.block.{i}.res_unit{}", j + 1), d))
                        .collect::<Result<_, String>>()?,
                    snake: Snake::load(model, &format!("{a}.block.{i}.snake1"))?,
                    down: SConv::load(model, &format!("{a}.block.{i}.conv1"), st, st.div_ceil(2), 1, 1)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let q = "ov.codec.quantizer.quantizers";
        let es = model.tensor(&format!("{q}.0.codebook.embed")).map(|e| e.shape.clone()).ok_or("omnivoice: no codebooks")?;
        let (rows, dim) = (es[0], es[1]);
        let hidden = model.tensor("ov.codec.fc.weight").map(|e| e.shape[0]).ok_or("omnivoice: no ov.codec.fc")?;
        let rvq = (0..cfg.books)
            .map(|c| {
                let e = crate::dit::cmf_f32(model, &format!("{q}.{c}.codebook.embed"))?;
                let norms: Vec<f32> = e.chunks_exact(dim).map(|r| r.iter().map(|v| v * v).sum()).collect();
                if norms.len() != rows {
                    return Err("omnivoice: codebook of another size".to_string());
                }
                Ok((
                    crate::qwen3_tts::Lin::load(model, &format!("{q}.{c}.project_in"), dim, hidden, true)?,
                    e,
                    norms,
                    crate::qwen3_tts::Lin::load(model, &format!("{q}.{c}.project_out"), hidden, dim, true)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let sem_rate = u(&k, "semantic_sample_rate", 16_000) as u32;
        let hop: usize = down.iter().product();
        let sem_down = ((hop as f64 / (cfg.sample_rate as f64 / sem_rate as f64)) / u(&k, "downsample_factor", 320) as f64) as usize;
        Ok(Self {
            hubert,
            sem_conv,
            sem_blocks,
            ac_conv1: SConv::load(model, &format!("{a}.conv1"), 1, 3, 1, 1)?,
            ac_blocks,
            ac_snake: Snake::load(model, &format!("{a}.snake1"))?,
            ac_conv2: SConv::load(model, &format!("{a}.conv2"), 1, 1, 1, 1)?,
            fc: crate::qwen3_tts::Lin::load(model, "ov.codec.fc", hidden, hidden, true)?,
            rvq,
            dim,
            hidden,
            cfg: cfg.clone(),
            sem_rate,
            sem_down: sem_down.max(1),
            pool,
        })
    }

    /// The acoustic encoder's output length for `n` samples.
    fn acoustic_len(&self, n: usize) -> usize {
        let mut l = self.ac_conv1.out_len(n);
        for b in &self.ac_blocks {
            l = b.down.out_len(l);
        }
        self.ac_conv2.out_len(l)
    }

    /// `HiggsAudioV2TokenizerModel.encode`: 24 kHz samples → `[books][frames]`.
    pub fn encode(&self, wav: &[f32]) -> Result<Vec<Vec<u32>>, String> {
        let pool = self.pool.as_deref();
        let wav16 = crate::mimo_audio::resample_sinc(wav, self.cfg.sample_rate as u32, self.sem_rate);
        let mut padded = vec![0f32; 160];
        padded.extend(&wav16);
        padded.extend([0f32; 160]);
        let feats = self.hubert.features(&padded, pool);
        let sh = self.sem_conv.inp;
        let frames_all = feats.len() / sh;
        let keep: Vec<usize> = (0..frames_all).step_by(self.sem_down).collect();
        let ns = keep.len();
        let mut x = vec![0f32; sh * ns];
        for (j, &t) in keep.iter().enumerate() {
            for c in 0..sh {
                x[c * ns + j] = feats[t * sh + c];
            }
        }
        let (mut e, _) = self.sem_conv.apply(&x, ns, pool);
        for (units, conv) in &self.sem_blocks {
            for (c1, c2) in units {
                let mut h = e.clone();
                elu(&mut h);
                let (mut h, _) = c1.apply(&h, ns, pool);
                elu(&mut h);
                let (h, _) = c2.apply(&h, ns, pool);
                for (a, b) in e.iter_mut().zip(&h) {
                    *a += b;
                }
            }
            e = conv.apply(&e, ns, pool).0;
        }
        // the acoustic side, padded by half a hop when the lengths disagree
        let input: Vec<f32> = if self.acoustic_len(wav.len()) != ns {
            let p = self.cfg.hop / 2;
            let mut v = vec![0f32; p];
            v.extend(wav);
            v.extend(std::iter::repeat_n(0f32, p));
            v
        } else {
            wav.to_vec()
        };
        let (mut h, mut n) = self.ac_conv1.apply(&input, input.len(), pool);
        for b in &self.ac_blocks {
            for u in &b.units {
                h = u.apply(&h, n, pool);
            }
            b.snake.apply(&mut h, n);
            let (y, m) = b.down.apply(&h, n, pool);
            h = y;
            n = m;
        }
        self.ac_snake.apply(&mut h, n);
        let (ea, na) = self.ac_conv2.apply(&h, n, pool);
        if na != ns {
            return Err(format!("omnivoice: acoustic frames {na} against semantic {ns}"));
        }
        let ach = ea.len() / na;
        // [frames, acoustic + semantic] → fc
        let hid = self.hidden;
        if ach + sh != hid {
            return Err(format!("omnivoice: {ach} + {sh} channels against fc's {hid}"));
        }
        let mut cat = vec![0f32; ns * hid];
        for t in 0..ns {
            for c in 0..ach {
                cat[t * hid + c] = ea[c * ns + t];
            }
            for c in 0..sh {
                cat[t * hid + ach + c] = e[c * ns + t];
            }
        }
        let mut resid = self.fc.apply(&cat, ns, pool);
        let mut codes = Vec::with_capacity(self.rvq.len());
        for (pin, emb, norms, pout) in &self.rvq {
            let xi = pin.apply(&resid, ns, pool); // [ns, dim]
            let rows = norms.len();
            let mut dots = vec![0f32; ns * rows];
            crate::fcd_ops::gemm_nt(&xi, emb, &mut dots, ns, self.dim, rows, pool);
            let mut ids = Vec::with_capacity(ns);
            let mut q = vec![0f32; ns * self.dim];
            for t in 0..ns {
                // nearest row: max of −(‖x‖² − 2x·e + ‖e‖²)
                let xx: f32 = xi[t * self.dim..(t + 1) * self.dim].iter().map(|v| v * v).sum();
                let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
                for r in 0..rows {
                    let d = -(xx - 2.0 * dots[t * rows + r] + norms[r]);
                    if d > best {
                        best = d;
                        bi = r;
                    }
                }
                ids.push(bi as u32);
                q[t * self.dim..(t + 1) * self.dim].copy_from_slice(&emb[bi * self.dim..(bi + 1) * self.dim]);
            }
            let qo = pout.apply(&q, ns, pool);
            for (a, b) in resid.iter_mut().zip(&qo) {
                *a -= b;
            }
            codes.push(ids);
        }
        Ok(codes)
    }

    /// `create_voice_clone_prompt` with a transcript: loudness lifted to an
    /// RMS of 0.1 when quieter, silences trimmed, cut to whole frames, encoded.
    pub fn voice_prompt(&self, wav24: &[f32], ref_text: &str) -> Result<VoicePrompt, String> {
        if ref_text.trim().is_empty() {
            return Err("omnivoice: the reference transcript is empty".into());
        }
        let rms = (wav24.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / wav24.len().max(1) as f64).sqrt() as f32;
        let mut w = wav24.to_vec();
        if rms > 0.0 && rms < 0.1 {
            w.iter_mut().for_each(|v| *v *= 0.1 / rms);
        }
        let mut w = remove_silence(&w, self.cfg.sample_rate, 200, 100, 200);
        if w.is_empty() {
            return Err("omnivoice: the reference is empty after silence removal".into());
        }
        let secs = w.len() as f64 / self.cfg.sample_rate as f64;
        if secs > 20.0 {
            tracing::warn!("omnivoice: the reference is {secs:.1} s long; 3-10 s clones best");
        }
        w.truncate(w.len() - w.len() % self.cfg.hop);
        if w.is_empty() {
            return Err("omnivoice: the reference is shorter than one frame".into());
        }
        Ok(VoicePrompt { codes: self.encode(&w)?, text: add_punctuation(ref_text), rms })
    }
}

// ───────────────────────────── post-processing ─────────────────────────────

/// pydub's view of a mono 16-bit clip at `rate`: milliseconds, slices, RMS.
struct Seg<'a> {
    s: &'a [i16],
    rate: usize,
}

/// Python's `round` (half to even).
fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    let fi = f as i64;
    if d > 0.5 || (d == 0.5 && fi % 2 != 0) { fi + 1 } else { fi }
}

impl Seg<'_> {
    fn len_ms(&self) -> i64 {
        round_half_even(1000.0 * self.s.len() as f64 / self.rate as f64)
    }

    fn frame(&self, ms: i64) -> usize {
        (ms as f64 * (self.rate as f64 / 1000.0)) as usize
    }

    /// `seg[a:b]` in milliseconds, padded with silence to the frames asked
    /// for when the rounded length runs past the samples (as pydub does).
    fn slice(&self, a: i64, b: i64) -> Vec<i16> {
        let len = self.len_ms();
        let (fa, fb) = (self.frame(a.min(len)), self.frame(b.min(len)));
        let n = self.s.len();
        let mut out: Vec<i16> = if fa < fb && fa < n { self.s[fa..fb.min(n)].to_vec() } else { Vec::new() };
        let want = fb.saturating_sub(fa);
        if !out.is_empty() && out.len() < want {
            out.resize(want, 0);
        }
        out
    }

    fn rms(x: &[i16]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (ss / x.len() as f64).sqrt().floor()
    }

    fn dbfs(x: &[i16]) -> f64 {
        let r = Self::rms(x);
        if r == 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * (r / 32768.0).log10()
        }
    }

    /// `detect_silence` with `seek_step`.
    fn silence(&self, min_len: i64, thresh_db: f64, step: i64) -> Vec<(i64, i64)> {
        let len = self.len_ms();
        if len < min_len {
            return Vec::new();
        }
        let thresh = 10f64.powf(thresh_db / 20.0) * 32768.0;
        let last = len - min_len;
        let mut starts: Vec<i64> = (0..=last).step_by(step as usize).collect();
        if last % step != 0 {
            starts.push(last);
        }
        let silent: Vec<i64> = starts.into_iter().filter(|&i| Self::rms(&self.slice(i, i + min_len)) <= thresh).collect();
        let Some((&first, rest)) = silent.split_first() else {
            return Vec::new();
        };
        let mut ranges = Vec::new();
        let (mut prev, mut cur) = (first, first);
        for &i in rest {
            let continuous = i == prev + step;
            let gap = i > prev + min_len;
            if !continuous && gap {
                ranges.push((cur, prev + min_len));
                cur = i;
            }
            prev = i;
        }
        ranges.push((cur, prev + min_len));
        ranges
    }

    fn nonsilent(&self, min_len: i64, thresh_db: f64, step: i64) -> Vec<(i64, i64)> {
        let sil = self.silence(min_len, thresh_db, step);
        let len = self.len_ms();
        if sil.is_empty() {
            return vec![(0, len)];
        }
        if sil[0] == (0, len) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut prev = 0;
        for &(s, e) in &sil {
            out.push((prev, s));
            prev = e;
        }
        if sil.last().is_some_and(|l| l.1 != len) {
            out.push((prev, len));
        }
        if out.first() == Some(&(0, 0)) {
            out.remove(0);
        }
        out
    }

    /// `detect_leading_silence` (10 ms chunks).
    fn leading_silence(&self, thresh_db: f64) -> i64 {
        let len = self.len_ms();
        let mut ms = 0;
        while Self::dbfs(&self.slice(ms, ms + 10)) < thresh_db && ms < len {
            ms += 10;
        }
        ms.min(len)
    }

    /// `seg[a:]`.
    fn tail(&self, a: i64) -> Vec<i16> {
        self.slice(a, self.len_ms())
    }
}

/// `remove_silence`: middle silences over `mid` ms shortened to `mid`, edges
/// trimmed to `lead` / `trail` ms (pydub at -50 dBFS, 10 ms steps).
pub fn remove_silence(audio: &[f32], rate: usize, mid: i64, lead: i64, trail: i64) -> Vec<f32> {
    let pcm: Vec<i16> = audio.iter().map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0) as i16).collect();
    let wave: Vec<i16> = if mid > 0 {
        let seg = Seg { s: &pcm, rate };
        let len = seg.len_ms();
        let mut ranges: Vec<(i64, i64)> = seg.nonsilent(mid, -50.0, 10).into_iter().map(|(s, e)| (s - mid, e + mid)).collect();
        for i in 0..ranges.len().saturating_sub(1) {
            let (last_end, next_start) = (ranges[i].1, ranges[i + 1].0);
            if next_start < last_end {
                let m = (last_end + next_start).div_euclid(2);
                ranges[i].1 = m;
                ranges[i + 1].0 = m;
            }
        }
        ranges.iter().flat_map(|&(s, e)| seg.slice(s.max(0), e.min(len))).collect()
    } else {
        pcm
    };
    let seg = Seg { s: &wave, rate };
    let start = (seg.leading_silence(-50.0) - lead).max(0);
    let mut w = seg.tail(start);
    w.reverse();
    let seg = Seg { s: &w, rate };
    let start = (seg.leading_silence(-50.0) - trail).max(0);
    let mut w = seg.tail(start);
    w.reverse();
    w.iter().map(|&v| v as f32 / 32768.0).collect()
}

/// `fade_and_pad_audio`: linear fades of `fade` s, `pad` s of silence a side.
pub fn fade_and_pad(audio: &[f32], rate: usize, pad: f32, fade: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let mut x = audio.to_vec();
    let k = ((fade as f64 * rate as f64) as usize).min(x.len() / 2);
    if k > 0 {
        let n = x.len();
        for i in 0..k {
            let w = if k == 1 { 0.0 } else { i as f32 / (k - 1) as f32 };
            x[i] *= w;
            x[n - k + i] *= 1.0 - w;
        }
    }
    let p = (pad as f64 * rate as f64) as usize;
    let mut out = vec![0f32; p];
    out.extend(x);
    out.extend(std::iter::repeat_n(0f32, p));
    out
}

/// `_post_process_audio`: silences, loudness (the reference's RMS when it
/// was quiet, else a 0.5 peak without a reference), fades and padding.
pub fn post_process(audio: &[f32], rate: usize, ref_rms: Option<f32>, postprocess: bool) -> Vec<f32> {
    let mut x = if postprocess { remove_silence(audio, rate, 500, 100, 100) } else { audio.to_vec() };
    match ref_rms {
        Some(r) if r < 0.1 => x.iter_mut().for_each(|v| *v *= r / 0.1),
        Some(_) => {}
        None => {
            let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            if peak > 1e-6 {
                x.iter_mut().for_each(|v| *v = *v / peak * 0.5);
            }
        }
    }
    fade_and_pad(&x, rate, 0.1, 0.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_covers_every_code() {
        let s = schedule(108 * 8, 32, 0.1);
        assert_eq!(s.len(), 32);
        assert_eq!(s.iter().sum::<usize>(), 108 * 8);
        // t_shift 0.1 spends the early steps on few codes, the late ones on many
        assert!(s[0] < s[30]);
        assert_eq!(s[0], (864.0 * (0.1f64 * (1.0 / 32.0) / (1.0 - 0.9 * (1.0 / 32.0)))).ceil() as usize);
    }

    #[test]
    fn duration_follows_the_reference_rate() {
        // "Nice to meet you." is 25 frames; a short text is boosted toward 50
        let f = estimate_frames("Hello, this is a test of the OmniVoice model running natively in cortiq.", None, None, 1.0);
        assert!(f > 50, "{f}");
        assert_eq!(estimate_frames("Hi.", None, None, 1.0), (50.0 * ((text_weight("Hi.") / (text_weight("Nice to meet you.") / 25.0)) / 50.0f64).powf(1.0 / 3.0)) as usize);
    }

    #[test]
    fn instruct_is_normalized() {
        assert_eq!(resolve_instruct(Some("Female, Low Pitch"), false).unwrap().as_deref(), Some("female, low pitch"));
        assert_eq!(resolve_instruct(Some("female, low pitch"), true).unwrap().as_deref(), Some("女，低音调"));
        assert_eq!(resolve_instruct(Some("男, 四川话"), false).unwrap().as_deref(), Some("男，四川话"));
        assert!(resolve_instruct(Some("male, female"), false).is_err());
        assert!(resolve_instruct(Some("robot"), false).is_err());
    }

    #[test]
    fn text_is_combined_like_the_reference() {
        assert_eq!(combine_text("  你好 世界 ", Some("Hi  there\n")), "Hi there你好世界");
        assert_eq!(combine_text("a\t\tb （c）", None), "a b (c)");
    }

    #[test]
    fn languages_resolve() {
        assert_eq!(resolve_language(Some("English")).as_deref(), Some("en"));
        assert_eq!(resolve_language(Some("ru")).as_deref(), Some("ru"));
        assert_eq!(resolve_language(Some("none")), None);
    }

    #[test]
    fn chunks_match_the_reference() {
        // `chunk_text_punctuation` of omnivoice 0.2.1 on the same inputs
        let c = |t: &str, l: usize| chunk_text_punctuation(t, l, 3);
        assert_eq!(
            c("Mr. Smith went to Washington. He met Dr. Brown, who said: hello! Then they left... The end.", 30),
            ["Mr. Smith went to Washington.", "He met Dr. Brown, who said:", "hello! Then they left...", "The end."]
        );
        assert_eq!(c("你好，世界。这是一个测试！还有更多的内容，对吧？", 8), ["你好，世界。", "这是一个测试！", "还有更多的内容，", "对吧？"]);
        assert_eq!(
            c("Short. A much longer sentence follows here, with commas, and more words. Final one.", 20),
            ["Short.", "A much longer sentence follows here,", "with commas,", "and more words.", "Final one."]
        );
    }

    #[test]
    fn fade_and_pad_shape() {
        let x = vec![1f32; 4800];
        let y = fade_and_pad(&x, 24_000, 0.1, 0.1);
        assert_eq!(y.len(), 4800 + 2 * 2400);
        assert_eq!(y[2400], 0.0);
        assert_eq!(y[2400 + 2399], 1.0);
    }
}
