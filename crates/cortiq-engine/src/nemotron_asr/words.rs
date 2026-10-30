//! Word timestamps of Nemotron 3.5 Transcribe, as the Transformers reference
//! computes them: the CTC timestamp head on the ASR encoder states
//! (`get_word_timestamp_logits`) and the forced alignment of the
//! transcript's words on its frames
//! (`Nemotron3_5TranscribeProcessor.align_word_timestamps`, itself a port of
//! NeMo's `MultiSpeakerSOTWordTimestampAligner`).
//!
//! Head: a transposed conv (kernel = stride = 2) doubles the encoder frames,
//! a 2-layer transformer of the ASR encoder's kind runs on them in the
//! encoder's chunks, `x + 0.1·LN(h)` is classified over the alignment
//! tokenizer's 23 862 SentencePiece pieces and the CTC blank.
//!
//! Alignment: the transcript's words (per `<spk:N>` tag when it has tags)
//! are tokenized with the alignment tokenizer, force-aligned (CTC Viterbi)
//! on the frames where the tag's diarization speaker is active, and snapped
//! to that speaker's diarization segments.

use super::{Encoder, Lin, sigmoid};
use crate::nemotron_h::tensor_f32;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// One aligned word.
#[derive(Clone, Debug)]
pub struct WordTime {
    pub word: String,
    pub start: f64,
    pub end: f64,
    /// the `N` of the word's `<spk:N>` tag; 0 in a transcript without tags
    pub speaker: Option<usize>,
}

/// The CTC timestamp head and its tokenizer.
pub struct WordTimes {
    /// transposed conv as a GEMM: rows `(j, out)` `[factor·d, d]`
    up: Lin,
    up_b: Vec<f32>,
    tower: Encoder,
    head: Lin,
    scale: f32,
    factor: usize,
    bpe: AlignBpe,
}

impl WordTimes {
    /// None when the file carries no timestamp head.
    pub fn load(m: &CmfModel, prov: &serde_json::Value) -> Result<Option<Self>, String> {
        let p = "ctc_timestamp";
        if m.tensor(&format!("{p}.head.weight")).is_none() {
            return Ok(None);
        }
        let cfg = &prov["word_timestamp_config"];
        let heads = cfg
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let rot = cfg
            .get("rope_parameters")
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let factor = cfg
            .get("upsampling_factor")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let scale = cfg
            .get("residual_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1) as f32;
        let tok = prov
            .get("alignment_tokenizer")
            .and_then(|v| v.as_str())
            .ok_or("timestamp head without its alignment tokenizer")?;
        let up = Lin::load(m, &format!("{p}.upsample"), false)?;
        let up_b = tensor_f32(m, &format!("{p}.upsample.bias"))?;
        let tower = Encoder::load_tower(m, &format!("{p}.transformer"), heads, rot)?;
        let head = Lin::load(m, &format!("{p}.head"), true)?;
        if up.out != factor * up.inp || up_b.len() != up.inp || head.inp != up.inp {
            return Err("timestamp head: shapes do not match the config".into());
        }
        Ok(Some(Self {
            up,
            up_b,
            tower,
            head,
            scale,
            factor,
            bpe: AlignBpe::from_json(tok)?,
        }))
    }

    /// Head logits `[n·factor · classes]` of `n` encoder states (before
    /// `asr_norm`), run per encoder chunk as the reference.
    pub fn logits(
        &self,
        asr: &[f32],
        n: usize,
        valid: &[bool],
        chunks: &[((usize, usize), (usize, usize))],
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let d = self.up.inp;
        let f = self.factor;
        let mut states = vec![0.0f32; n * f * d];
        for &((cs, ce), (a, b)) in chunks {
            let m = ce - cs;
            // y[t·f + j] = W_j x_t + b: the GEMM's row t is [j = 0 | j = 1 | …]
            let mut x = self.up.apply(&asr[cs * d..ce * d], m, pool);
            for r in x.chunks_exact_mut(d) {
                for (v, bb) in r.iter_mut().zip(&self.up_b) {
                    *v += bb;
                }
            }
            let mask: Vec<bool> = (0..m * f).map(|i| valid[cs + i / f]).collect();
            let h = self.tower.run(&x, m * f, &mask, pool);
            for i in a * f..b * f {
                let dst = &mut states[(cs * f + i) * d..(cs * f + i + 1) * d];
                for ((o, xv), hv) in dst.iter_mut().zip(&x[i * d..(i + 1) * d]).zip(&h[i * d..(i + 1) * d]) {
                    *o = xv + self.scale * hv;
                }
            }
        }
        self.head.apply(&states, n * f, pool)
    }

    /// Times of the words of `text` (the generated transcript, its
    /// `<spk:N>` tags kept). `logits`: the head's output for the
    /// recording's `tokens` encoder frames; `valid`: its spectrogram frames;
    /// `speakers`: the diarization logits `[frames · n_spk]`, used when the
    /// transcript has speaker tags.
    pub fn align(
        &self,
        text: &str,
        logits: &[f32],
        tokens: usize,
        valid: usize,
        speakers: Option<(&[f32], usize)>,
    ) -> Result<Vec<WordTime>, String> {
        let classes = self.head.out;
        let blank = classes - 1;
        let n_frames = tokens * self.factor;
        if logits.len() < n_frames * classes {
            return Err("timestamp head: fewer frames than the recording".into());
        }
        let audio_duration = valid as f64 * HOP_S;
        let frame_duration = audio_duration / n_frames as f64;
        let mut clean = text.to_string();
        for pat in ["<|im_end|>", "<|im_start|>", "<think>", "</think>", "<|audio|>"] {
            clean = clean.replace(pat, " ");
        }
        let streams = self.bpe.streams(&clean, blank)?;
        if streams.is_empty() {
            return Err("transcript is empty".into());
        }
        // log-softmax per frame
        let mut lp = logits[..n_frames * classes].to_vec();
        for r in lp.chunks_exact_mut(classes) {
            let mx = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s = r.iter().map(|v| ((v - mx) as f64).exp()).sum::<f64>().ln() as f32 + mx;
            for v in r.iter_mut() {
                *v -= s;
            }
        }
        let tagged = streams.iter().any(|(t, _)| *t != Some(0)) || streams.len() > 1;
        let mut probs: Option<(Vec<f32>, usize)> = None;
        let mut segs: Vec<(usize, f64, f64)> = Vec::new();
        let mut tag_spk: HashMap<Option<usize>, Option<usize>> =
            streams.iter().map(|(t, _)| (*t, None)).collect();
        if let (true, Some((sl, ns))) = (tagged, speakers) {
            // mean of the probabilities over the spectrogram frames of each
            // encoder frame, resampled onto the head's frames
            let sub = 8usize;
            let mut enc = vec![0.0f32; tokens * ns];
            for t in 0..tokens {
                for s in 0..ns {
                    let mut acc = 0.0;
                    for j in 0..sub {
                        let fi = t * sub + j;
                        acc += sl.get(fi * ns + s).map_or(0.0, |&l| sigmoid(l));
                    }
                    enc[t * ns + s] = acc / sub as f32;
                }
            }
            let p = resample(&enc, tokens, ns, n_frames);
            let map = match_tags(&lp, classes, &streams, &p, ns, 0.5, blank);
            tag_spk = map;
            // diarization segments on the spectrogram frames
            for s in 0..ns {
                let mut start: Option<usize> = None;
                for fi in 0..=valid {
                    let on = fi < valid && sigmoid(sl[fi * ns + s]) > 0.5;
                    match (on, start) {
                        (true, None) => start = Some(fi),
                        (false, Some(st)) => {
                            let (a, b) = (st as f64 * HOP_S, (fi as f64 * HOP_S).min(audio_duration));
                            if b > a {
                                segs.push((s, a, b));
                            }
                            start = None;
                        }
                        _ => {}
                    }
                }
            }
            probs = Some((p, ns));
        }
        let mut words = Vec::new();
        for (tag, stream) in &streams {
            let spk = tag_spk.get(tag).copied().flatten();
            let column: Vec<f32> = match (&probs, spk) {
                (None, _) => vec![1.0; n_frames],
                (Some(_), None) => vec![0.0; n_frames],
                (Some((p, ns)), Some(s)) => (0..n_frames).map(|t| p[t * ns + s]).collect(),
            };
            let Some(mut ts) = align_words(&lp, classes, stream, &column, blank, frame_duration, 1.0)
            else {
                continue;
            };
            let speaker_segs: Vec<(f64, f64)> = segs
                .iter()
                .filter(|(s, _, _)| Some(*s) == spk)
                .map(|&(_, a, b)| (a, b))
                .collect();
            refine(&mut ts, &speaker_segs, frame_duration, 12 * self.factor);
            words.extend(ts.into_iter().map(|(w, a, b)| WordTime {
                word: w,
                start: a,
                end: b,
                speaker: *tag,
            }));
        }
        words.sort_by(|x, y| x.start.total_cmp(&y.start).then(x.end.total_cmp(&y.end)));
        Ok(words)
    }
}

/// Spectrogram hop in seconds.
const HOP_S: f64 = 160.0 / 16000.0;

/// Linear interpolation of `[frames · ns]` onto `n` frames, ends aligned.
fn resample(p: &[f32], frames: usize, ns: usize, n: usize) -> Vec<f32> {
    if frames == n {
        return p.to_vec();
    }
    let mut out = vec![0.0f32; n * ns];
    for t in 0..n {
        let pos = if n > 1 {
            t as f64 * (frames - 1) as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let fr = (pos - lo as f64) as f32;
        for s in 0..ns {
            out[t * ns + s] = p[lo * ns + s] * (1.0 - fr) + p[hi * ns + s] * fr;
        }
    }
    out
}

/// `(word, token ids)` units per tag, in order of first appearance.
type Streams = Vec<(Option<usize>, Vec<(String, Vec<u32>)>)>;

/// The diarization speaker of each tag: token counts of the tag's words
/// against the greedy tokens decoded while each speaker is active,
/// IDF-weighted cosine similarity minus a speech-amount penalty, matched one
/// to one by the Hungarian method.
fn match_tags(
    lp: &[f32],
    classes: usize,
    streams: &Streams,
    probs: &[f32],
    ns: usize,
    thr: f32,
    blank: usize,
) -> HashMap<Option<usize>, Option<usize>> {
    let n_frames = lp.len() / classes;
    let mut cls: Vec<usize> = streams
        .iter()
        .flat_map(|(_, w)| w.iter().flat_map(|(_, ids)| ids.iter().map(|&i| i as usize)))
        .collect();
    cls.sort_unstable();
    cls.dedup();
    let mut allowed = vec![blank];
    allowed.extend(cls);
    let greedy: Vec<usize> = (0..n_frames)
        .map(|t| {
            let r = &lp[t * classes..(t + 1) * classes];
            let mut best = allowed[0];
            for &c in &allowed[1..] {
                if r[c] > r[best] {
                    best = c;
                }
            }
            best
        })
        .collect();
    let nt = streams.len();
    let mut tag_counts: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nt];
    for (i, (_, w)) in streams.iter().enumerate() {
        for (_, ids) in w {
            for &id in ids {
                *tag_counts[i].entry(id as usize).or_default() += 1.0;
            }
        }
    }
    let mut spk_counts: Vec<HashMap<usize, f64>> = vec![HashMap::new(); ns];
    for t in 0..n_frames {
        let g = greedy[t];
        let onset = g != blank && (t == 0 || greedy[t - 1] != g);
        if !onset {
            continue;
        }
        for (s, c) in spk_counts.iter_mut().enumerate() {
            if probs[t * ns + s] >= thr {
                *c.entry(g).or_default() += 1.0;
            }
        }
    }
    let mut df: HashMap<usize, f64> = HashMap::new();
    for c in &spk_counts {
        for (&k, &v) in c {
            if v > 0.0 {
                *df.entry(k).or_default() += 1.0;
            }
        }
    }
    let idf = |k: usize| (1.0 + ns as f64 / (1.0 + df.get(&k).copied().unwrap_or(0.0))).ln();
    let vec_of = |c: &HashMap<usize, f64>| -> HashMap<usize, f64> {
        c.iter().map(|(&k, &v)| (k, (v * idf(k)).sqrt())).collect()
    };
    let tv: Vec<HashMap<usize, f64>> = tag_counts.iter().map(vec_of).collect();
    let sv: Vec<HashMap<usize, f64>> = spk_counts.iter().map(vec_of).collect();
    let norm = |v: &HashMap<usize, f64>| v.values().map(|x| x * x).sum::<f64>().sqrt();
    let mut sim = vec![vec![0.0f64; ns]; nt];
    for i in 0..nt {
        for s in 0..ns {
            let dotp: f64 = tv[i]
                .iter()
                .map(|(k, a)| a * sv[s].get(k).copied().unwrap_or(0.0))
                .sum();
            let den = (norm(&tv[i]) * norm(&sv[s])).max(1e-8);
            let ta: f64 = tag_counts[i].values().sum();
            let sa: f64 = spk_counts[s].values().sum();
            sim[i][s] = dotp / den - 0.1 * ((ta + 1.0) / (sa + 1.0)).ln().abs();
        }
    }
    let assign = hungarian_max(&sim);
    streams
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (*t, assign[i]))
        .collect()
}

/// Maximum-weight assignment of rows to columns (each used once); rows
/// beyond the columns stay unassigned.
fn hungarian_max(w: &[Vec<f64>]) -> Vec<Option<usize>> {
    let n = w.len();
    let m = w.first().map_or(0, |r| r.len());
    let k = n.max(m);
    if k == 0 {
        return vec![None; n];
    }
    let big = w
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    // square cost matrix, minimized (padding cost = big: never preferred)
    let cost = |i: usize, j: usize| -> f64 {
        if i < n && j < m {
            big - w[i][j]
        } else {
            big + 1.0
        }
    };
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0; k + 1], vec![0.0; k + 1]);
    let (mut p, mut way) = (vec![0usize; k + 1], vec![0usize; k + 1]);
    for i in 1..=k {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; k + 1];
        let mut used = vec![false; k + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=k {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=k {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; n];
    for j in 1..=k {
        let i = p[j];
        if i >= 1 && i <= n && j <= m {
            out[i - 1] = Some(j - 1);
        }
    }
    out
}

/// Start and end of each word, force-aligned on the frames where
/// `probs >= 0.01`, `ln(probs)` added to the token log-probabilities. None
/// when the words do not fit on those frames.
fn align_words(
    lp: &[f32],
    classes: usize,
    words: &[(String, Vec<u32>)],
    probs: &[f32],
    blank: usize,
    frame_duration: f64,
    max_word: f64,
) -> Option<Vec<(String, f64, f64)>> {
    let target: Vec<usize> = words
        .iter()
        .flat_map(|(_, ids)| ids.iter().map(|&i| i as usize))
        .collect();
    let frames: Vec<usize> = (0..probs.len()).filter(|&t| probs[t] >= 0.01).collect();
    let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
    if target.is_empty() || frames.len() < target.len() + repeats {
        return None;
    }
    let t_n = frames.len();
    let em = |t: usize, c: usize| -> f32 {
        let f = frames[t];
        let v = lp[f * classes + c];
        if c == blank {
            v
        } else {
            v + probs[f].max(1e-6).ln()
        }
    };
    let labels = ctc_viterbi(t_n, target.len(), &|t, c| em(t, c), &target, blank)?;
    // one span per target token: runs of equal non-blank labels
    let mut spans = Vec::with_capacity(target.len());
    let mut t = 0;
    while t < t_n {
        let l = labels[t];
        let mut e = t + 1;
        while e < t_n && labels[e] == l {
            e += 1;
        }
        if l != blank {
            spans.push((t, e));
        }
        t = e;
    }
    if spans.len() != target.len() {
        return None;
    }
    let mut out = Vec::with_capacity(words.len());
    let mut k = 0;
    for (w, ids) in words {
        let first = spans[k];
        let last = spans[k + ids.len() - 1];
        k += ids.len();
        let start = frames[first.0] as f64 * frame_duration;
        let end = ((frames[last.1 - 1] + 1) as f64 * frame_duration).min(start + max_word);
        out.push((w.clone(), start, end));
    }
    Some(out)
}

/// CTC forced alignment (Viterbi over the blank-interleaved target): the
/// label of every frame on the best path.
fn ctc_viterbi(
    t_n: usize,
    l_n: usize,
    em: &dyn Fn(usize, usize) -> f32,
    target: &[usize],
    blank: usize,
) -> Option<Vec<usize>> {
    let s_n = 2 * l_n + 1;
    let lab = |s: usize| if s % 2 == 0 { blank } else { target[s / 2] };
    let ninf = f32::NEG_INFINITY;
    let mut alpha = vec![ninf; s_n];
    let mut back = vec![0u8; t_n * s_n];
    alpha[0] = em(0, blank);
    if s_n > 1 {
        alpha[1] = em(0, lab(1));
    }
    let mut next = vec![ninf; s_n];
    for t in 1..t_n {
        // states that can still reach the end
        let lo = s_n.saturating_sub(2 * (t_n - t));
        for s in 0..s_n {
            next[s] = ninf;
            if s + 2 < lo {
                continue;
            }
            let mut best = alpha[s];
            let mut from = 0u8;
            if s >= 1 && alpha[s - 1] > best {
                best = alpha[s - 1];
                from = 1;
            }
            if s >= 2 && s % 2 == 1 && lab(s) != lab(s - 2) && alpha[s - 2] > best {
                best = alpha[s - 2];
                from = 2;
            }
            if best > ninf {
                next[s] = best + em(t, lab(s));
            }
            back[t * s_n + s] = from;
        }
        std::mem::swap(&mut alpha, &mut next);
    }
    let mut s = if s_n >= 2 && alpha[s_n - 2] > alpha[s_n - 1] {
        s_n - 2
    } else {
        s_n - 1
    };
    if alpha[s] == ninf {
        return None;
    }
    let mut labels = vec![blank; t_n];
    for t in (0..t_n).rev() {
        labels[t] = lab(s);
        if t > 0 {
            s -= back[t * s_n + s] as usize;
        }
    }
    Some(labels)
}

/// Snap one speaker's words (in time order) to its diarization segments:
/// the first word of a segment starts with it, a word ends where the next
/// word of its segment starts, else with its segment; shifts and durations
/// capped at `max_frames` frames of at most 80 ms.
fn refine(words: &mut [(String, f64, f64)], segs: &[(f64, f64)], frame_duration: f64, max_frames: usize) {
    let limit = max_frames as f64 * frame_duration.min(0.08);
    let mut segs = segs.to_vec();
    segs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let seg_of: Vec<Option<usize>> = words
        .iter()
        .map(|w| {
            let idx = segs.partition_point(|s| s.0 <= w.1);
            (idx > 0 && w.1 < segs[idx - 1].1).then(|| idx - 1)
        })
        .collect();
    for i in 0..words.len() {
        if let Some(sg) = seg_of[i] {
            let first = i == 0 || seg_of[i - 1] != Some(sg);
            let shift = words[i].1 - segs[sg].0;
            if first && shift > 0.0 && shift <= limit {
                words[i].1 = segs[sg].0;
            }
        }
    }
    for i in 0..words.len() {
        let end = match seg_of[i] {
            Some(sg) if i + 1 >= words.len() || seg_of[i + 1] != Some(sg) => segs[sg].1,
            _ => {
                if i + 1 < words.len() {
                    words[i + 1].1
                } else {
                    words[i].2
                }
            }
        };
        words[i].2 = words[i].1.max(end.min(words[i].1 + limit));
    }
}

/// The alignment tokenizer: SentencePiece BPE (Metaspace, the 49 tag
/// pieces split out first), `encode(" " + word)` as the reference.
struct AlignBpe {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    added: HashMap<String, u32>,
    unk: u32,
    space: Option<u32>,
}

/// Unicode scripts that make a transcript Chinese/Japanese (aligned per
/// character), by priority; Thai is refused.
fn script_language(text: &str) -> Option<&'static str> {
    let in_ranges = |c: char, r: &[(u32, u32)]| r.iter().any(|&(a, b)| (a..=b).contains(&(c as u32)));
    let has = |r: &[(u32, u32)]| text.chars().any(|c| in_ranges(c, r));
    if has(&[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]) {
        Some("ko")
    } else if has(&[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]) {
        Some("ja")
    } else if has(&[(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)]) {
        Some("zh")
    } else if has(&[(0x0E00, 0x0E7F)]) {
        Some("th")
    } else {
        None
    }
}

impl AlignBpe {
    fn from_json(s: &str) -> Result<Self, String> {
        let j: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("alignment tokenizer: {e}"))?;
        let model = &j["model"];
        let vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .ok_or("alignment tokenizer: no vocab")?
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_u64()? as u32)))
            .collect();
        let mut ranks = HashMap::new();
        for (i, m) in model["merges"].as_array().ok_or("alignment tokenizer: no merges")?.iter().enumerate() {
            let pair = match m {
                serde_json::Value::Array(a) if a.len() == 2 => (
                    a[0].as_str().unwrap_or_default().to_string(),
                    a[1].as_str().unwrap_or_default().to_string(),
                ),
                serde_json::Value::String(s) => match s.split_once(' ') {
                    Some((a, b)) => (a.to_string(), b.to_string()),
                    None => continue,
                },
                _ => continue,
            };
            ranks.entry(pair).or_insert(i);
        }
        let added = j["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32)))
                    .collect()
            })
            .unwrap_or_default();
        let unk = vocab.get("<unk>").copied().unwrap_or(0);
        let space = vocab.get("\u{2581}").copied();
        Ok(Self {
            vocab,
            ranks,
            added,
            unk,
            space,
        })
    }

    /// `encode(" " + word)`.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        if let Some(&id) = self.added.get(word) {
            return self.space.into_iter().chain([id]).collect();
        }
        let norm: String = format!(" {word}").nfkc().collect::<String>().replace(' ', "\u{2581}");
        let mut sym: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone()))
                    && best.is_none_or(|b| r < b.0)
                {
                    best = Some((r, i));
                }
            }
            let Some((_, i)) = best else {
                break;
            };
            let merged = format!("{}{}", sym[i], sym[i + 1]);
            sym[i] = merged;
            sym.remove(i + 1);
        }
        let mut ids = Vec::with_capacity(sym.len());
        for s in sym {
            let id = self.vocab.get(&s).copied().unwrap_or(self.unk);
            if !(id == self.unk && ids.last() == Some(&self.unk)) {
                ids.push(id);
            }
        }
        ids
    }

    /// A character is its own piece when the vocabulary has one, else its
    /// pieces without the bare word boundary.
    fn encode_char(&self, c: char) -> Vec<u32> {
        let s = c.to_string();
        if let Some(&id) = self.vocab.get(&s)
            && id != self.unk
        {
            return vec![id];
        }
        self.encode_word(&s)
            .into_iter()
            .filter(|&i| Some(i) != self.space)
            .collect()
    }

    /// The words of each tag of a transcript (tag None before the first
    /// `<spk:N>`, tag 0 for a transcript without tags).
    fn streams(&self, text: &str, blank: usize) -> Result<Streams, String> {
        let text = text.replace('—', " ");
        // split on <spk:N> (case-insensitive)
        let mut parts: Vec<(Option<usize>, String)> = Vec::new();
        let b = text.as_bytes();
        let mut cur: Option<usize> = None;
        let mut last = 0usize;
        let mut tagged = false;
        let mut i = 0;
        while i + 5 <= b.len() {
            if b[i] == b'<' && b[i + 1..i + 5].eq_ignore_ascii_case(b"spk:") {
                let nd = b[i + 5..].iter().take_while(|c| c.is_ascii_digit()).count();
                if nd > 0 && b.get(i + 5 + nd) == Some(&b'>') {
                    parts.push((cur, text[last..i].to_string()));
                    cur = text[i + 5..i + 5 + nd].parse().ok();
                    tagged = true;
                    last = i + 6 + nd;
                    i = last;
                    continue;
                }
            }
            i += 1;
        }
        parts.push((cur, text[last..].to_string()));
        if !tagged {
            parts[0].0 = Some(0);
        }
        let words: Vec<(Option<usize>, &str)> = parts
            .iter()
            .flat_map(|(t, p)| p.split_whitespace().map(move |w| (*t, w)))
            .collect();
        let is_tag = |w: &str| {
            (w.starts_with("<ev:") || w.starts_with("<sty:") || w.to_lowercase().starts_with("<spk:"))
                && w.ends_with('>')
        };
        let plain: String = words
            .iter()
            .filter(|(_, w)| !is_tag(w))
            .map(|(_, w)| *w)
            .collect::<Vec<_>>()
            .join(" ");
        let lang = script_language(&plain);
        if lang == Some("th") {
            return Err("word timestamps are not supported for Thai".into());
        }
        let mut out: Streams = Vec::new();
        for (tag, w) in words {
            let units: Vec<(String, Vec<u32>)> = if matches!(lang, Some("zh") | Some("ja")) && !is_tag(w) {
                w.nfc()
                    .filter(|c| c.is_alphanumeric() || is_mark(*c))
                    .map(|c| (c.to_string(), self.encode_char(c)))
                    .collect()
            } else {
                vec![(w.to_string(), self.encode_word(w))]
            };
            if units
                .iter()
                .any(|(_, ids)| ids.is_empty() || ids.iter().any(|&i| i as usize >= blank))
            {
                return Err(format!("alignment tokenizer cannot represent {w:?}"));
            }
            match out.iter_mut().find(|(t, _)| *t == tag) {
                Some((_, v)) => v.extend(units),
                None => out.push((tag, units)),
            }
        }
        Ok(out)
    }
}

/// Combining marks (Unicode category M) — the lexical characters of a
/// Chinese/Japanese word are its letters, marks and numbers.
fn is_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F | 0x3099..=0x309A)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_aligns_a_clear_path() {
        // 6 frames, target [1, 2], blank 0: frames 1-2 say 1, frames 4 says 2
        let lp = [
            [0.0f32, -9.0, -9.0],
            [-9.0, 0.0, -9.0],
            [-9.0, 0.0, -9.0],
            [0.0, -9.0, -9.0],
            [-9.0, -9.0, 0.0],
            [0.0, -9.0, -9.0],
        ];
        let em = |t: usize, c: usize| lp[t][c];
        let labels = ctc_viterbi(6, 2, &em, &[1, 2], 0).unwrap();
        assert_eq!(labels, vec![0, 1, 1, 0, 2, 0]);
        // repeated tokens need a blank between them
        let lp2 = [[-9.0f32, 0.0], [0.0, -9.0], [-9.0, 0.0]];
        let em2 = |t: usize, c: usize| lp2[t][c];
        assert_eq!(ctc_viterbi(3, 2, &em2, &[1, 1], 0).unwrap(), vec![1, 0, 1]);
    }

    #[test]
    fn hungarian_maximizes_and_leaves_extra_rows() {
        let w = vec![vec![1.0, 5.0], vec![4.0, 1.0], vec![0.5, 0.5]];
        let a = hungarian_max(&w);
        assert_eq!(a[0], Some(1));
        assert_eq!(a[1], Some(0));
        assert_eq!(a[2], None);
    }
}
