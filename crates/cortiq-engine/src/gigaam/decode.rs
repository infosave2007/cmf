//! The recognition heads of GigaAM: greedy CTC over characters, greedy
//! RNN-T over SentencePiece units, and the way from token ids and their
//! frames to text and word times.

use super::{Lin, ops, tensor_f32};
use cortiq_core::CmfModel;
use std::sync::Arc;

/// A word and where it was said, in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Token ids of one segment and the encoder frame each was emitted at.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Decoded {
    pub ids: Vec<u32>,
    pub frames: Vec<u32>,
}

/// The vocabulary of a head.
enum Vocab {
    /// one string a token (CTC)
    Chars(Vec<String>),
    /// SentencePiece units: `▁` opens a word
    Pieces(Vec<String>),
}

impl Vocab {
    fn len(&self) -> usize {
        match self {
            Self::Chars(v) | Self::Pieces(v) => v.len(),
        }
    }

    fn piece(&self, id: u32) -> &str {
        match self {
            Self::Chars(v) | Self::Pieces(v) => v.get(id as usize).map_or("", String::as_str),
        }
    }

    fn decode(&self, ids: &[u32]) -> String {
        match self {
            Self::Chars(_) => ids.iter().map(|&i| self.piece(i)).collect(),
            Self::Pieces(_) => {
                // SentencePiece: units joined, `▁` is a space, no space before the first word,
                // the unknown unit reads as " ⁇ "
                let mut s = String::new();
                for &i in ids {
                    if i == 0 {
                        s.push_str(" \u{2047} ");
                    } else {
                        s.push_str(&self.piece(i).replace('\u{2581}', " "));
                    }
                }
                s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
            }
        }
    }
}

/// The prediction network and the joint of an RNN-T head.
struct Rnnt {
    embed: Vec<f32>,
    hidden: usize,
    /// `[4·hidden, hidden]` each, gate order input, forget, cell, output
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    /// the two LSTM biases summed
    bias: Vec<f32>,
    enc: Lin,
    /// `[joint, hidden]` and its bias
    pred_w: Vec<f32>,
    pred_b: Vec<f32>,
    /// `[classes, joint]` and its bias
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    joint: usize,
    classes: usize,
    max_symbols: usize,
}

enum Kind {
    Ctc(Lin),
    Rnnt(Box<Rnnt>),
}

/// One recognition head.
pub(crate) struct Head {
    kind: Kind,
    vocab: Vocab,
    blank: u32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Head {
    pub(crate) fn load(m: &Arc<CmfModel>, cfg: &serde_json::Value, tokenizers: Option<&serde_json::Value>, name: &str) -> Result<Self, String> {
        let strings = |v: &serde_json::Value| -> Option<Vec<String>> {
            v.as_array().map(|a| a.iter().map(|s| s.as_str().unwrap_or("").to_string()).collect())
        };
        let d = cfg.pointer(&format!("/decodings/{name}")).ok_or_else(|| format!("GigaAM config: no decoding for head '{name}'"))?;
        let target = cfg.pointer(&format!("/heads/{name}/_target_")).and_then(|v| v.as_str()).unwrap_or("");
        let p = format!("model.heads.{name}");
        if target.ends_with("CTCHead") {
            let vocab = d.get("vocabulary").and_then(strings).ok_or_else(|| format!("GigaAM config: head '{name}' has no vocabulary"))?;
            let lin = Lin::load(m, &format!("{p}.decoder_layers.0"))?;
            if lin.rows != vocab.len() + 1 {
                return Err(format!("GigaAM head '{name}': {} classes for {} characters", lin.rows, vocab.len()));
            }
            let blank = vocab.len() as u32;
            return Ok(Self { kind: Kind::Ctc(lin), vocab: Vocab::Chars(vocab), blank });
        }
        if !target.ends_with("RNNTHead") {
            return Err(format!("GigaAM head '{name}': unsupported head '{target}'"));
        }
        let vocab = match tokenizers.and_then(|t| t.get(name)).and_then(strings) {
            Some(pieces) => Vocab::Pieces(pieces),
            None => Vocab::Chars(d.get("vocabulary").and_then(strings).ok_or_else(|| format!("GigaAM file: no tokenizer for head '{name}'"))?),
        };
        let shape = |t: &str| m.tensor(t).map(|e| e.shape.clone()).ok_or_else(|| format!("missing GigaAM tensor {t}"));
        let emb = shape(&format!("{p}.decoder.embed.weight"))?;
        let hidden = emb[1];
        let layers = cfg.pointer(&format!("/heads/{name}/decoder/pred_rnn_layers")).and_then(|v| v.as_u64()).unwrap_or(1);
        if layers != 1 {
            return Err(format!("GigaAM head '{name}': {layers} LSTM layers (one is supported)"));
        }
        let mut bias = tensor_f32(m, &format!("{p}.decoder.lstm.bias_ih_l0"))?;
        for (b, h) in bias.iter_mut().zip(tensor_f32(m, &format!("{p}.decoder.lstm.bias_hh_l0"))?) {
            *b += h;
        }
        let out = shape(&format!("{p}.joint.joint_net.1.weight"))?;
        let r = Rnnt {
            embed: tensor_f32(m, &format!("{p}.decoder.embed.weight"))?,
            hidden,
            w_ih: tensor_f32(m, &format!("{p}.decoder.lstm.weight_ih_l0"))?,
            w_hh: tensor_f32(m, &format!("{p}.decoder.lstm.weight_hh_l0"))?,
            bias,
            enc: Lin::load(m, &format!("{p}.joint.enc"))?,
            pred_w: tensor_f32(m, &format!("{p}.joint.pred.weight"))?,
            pred_b: tensor_f32(m, &format!("{p}.joint.pred.bias"))?,
            out_w: tensor_f32(m, &format!("{p}.joint.joint_net.1.weight"))?,
            out_b: tensor_f32(m, &format!("{p}.joint.joint_net.1.bias"))?,
            joint: out[1],
            classes: out[0],
            max_symbols: d.get("max_symbols_per_step").and_then(|v| v.as_u64()).unwrap_or(10) as usize,
        };
        if r.classes != vocab.len() + 1 || emb[0] != r.classes || r.w_ih.len() != 4 * hidden * hidden {
            return Err(format!("GigaAM head '{name}': {} classes for {} units", r.classes, vocab.len()));
        }
        let blank = vocab.len() as u32;
        Ok(Self { kind: Kind::Rnnt(Box::new(r)), vocab, blank })
    }

    /// Greedy decoding of one segment: `enc` is `[t, width]` encoder rows.
    /// Runs on the calling thread (segments and heads are spread by the caller).
    pub(crate) fn decode(&self, enc: &[f32], t: usize, width: usize, pool: Option<&crate::pool::Pool>) -> Decoded {
        debug_assert_eq!(enc.len(), t * width);
        // `pool`: None when the caller's threads are busy with other segments
        let ex = super::Exec { gpu: false, ..super::Exec::new(pool) };
        match &self.kind {
            Kind::Ctc(lin) => {
                let logits = lin.apply(enc, t, &ex);
                let mut out = Decoded::default();
                let mut prev = self.blank;
                for (i, row) in logits.chunks_exact(lin.rows).enumerate() {
                    let label = argmax(row);
                    if label != self.blank && label != prev {
                        out.ids.push(label);
                        out.frames.push(i as u32);
                    }
                    prev = label;
                }
                out
            }
            Kind::Rnnt(r) => r.greedy(&r.enc.apply(enc, t, &ex), t, self.blank, pool),
        }
    }

    pub(crate) fn text(&self, ids: &[u32]) -> String {
        self.vocab.decode(ids)
    }

    /// Words with their times: a word starts at its first token's frame and
    /// ends after its last token's frame; `shift` is seconds per frame,
    /// `offset` the segment's start.
    pub(crate) fn words(&self, d: &Decoded, shift: f64, offset: f64) -> Vec<Word> {
        let mut words = Vec::new();
        let mut text = String::new();
        let mut span: Option<(u32, u32)> = None;
        let round = |v: f64| (v * 1000.0).round() / 1000.0;
        let mut commit = |text: &mut String, span: &mut Option<(u32, u32)>| {
            if let Some((a, b)) = span.take() {
                let t = text.trim();
                if !t.is_empty() {
                    words.push(Word { text: t.to_string(), start: round(a as f64 * shift + offset), end: round((b + 1) as f64 * shift + offset) });
                }
            }
            text.clear();
        };
        for (&id, &frame) in d.ids.iter().zip(&d.frames) {
            let piece = self.vocab.piece(id);
            let piece = if let Some(rest) = piece.strip_prefix('\u{2581}') {
                commit(&mut text, &mut span);
                rest
            } else if piece == " " {
                commit(&mut text, &mut span);
                continue;
            } else {
                piece
            };
            text.push_str(piece);
            span = Some((span.map_or(frame, |s| s.0), frame));
        }
        commit(&mut text, &mut span);
        words
    }
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best as u32
}

impl Rnnt {
    /// One LSTM step: `(h, c) ← cell(x, h, c)`; `x` empty stands for zeros.
    fn step(&self, x: &[f32], h: &mut [f32], c: &mut [f32], gates: &mut [f32], pool: Option<&crate::pool::Pool>) {
        let n = self.hidden;
        gates.copy_from_slice(&self.bias);
        let mut acc = vec![0f32; 4 * n];
        if !x.is_empty() {
            ops::matmul(&ops::Weight::Dense(&self.w_ih), 4 * n, n, x, 1, &mut acc, pool);
            ops::axpy(gates, 1.0, &acc);
        }
        ops::matmul(&ops::Weight::Dense(&self.w_hh), 4 * n, n, h, 1, &mut acc, pool);
        ops::axpy(gates, 1.0, &acc);
        for i in 0..n {
            let (ig, fg, gg, og) = (sigmoid(gates[i]), sigmoid(gates[n + i]), gates[2 * n + i].tanh(), sigmoid(gates[3 * n + i]));
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * c[i].tanh();
        }
    }

    /// Greedy search over `[t, joint]` projected encoder frames: at a frame,
    /// emit units until the joint says blank (at most `max_symbols`), then
    /// move on. The joint is evaluated for a run of frames at once — the
    /// prediction does not change until a unit is emitted.
    fn greedy(&self, enc: &[f32], t: usize, blank: u32, pool: Option<&crate::pool::Pool>) -> Decoded {
        const WINDOW: usize = 8;
        let (n, j) = (self.hidden, self.joint);
        let mut out = Decoded::default();
        let (mut h, mut c) = (vec![0f32; n], vec![0f32; n]);
        // the state behind the prediction in hand; it becomes the decoder's
        // state when that prediction emits a unit
        let (mut ph, mut pc) = (vec![0f32; n], vec![0f32; n]);
        let mut gates = vec![0f32; 4 * n];
        let mut pred = vec![0f32; j];
        let mut have_pred = false;
        let mut last: Option<u32> = None;
        let mut hid = vec![0f32; WINDOW * j];
        let mut logits = vec![0f32; WINDOW * self.classes];
        let (mut pos, mut repeats) = (0usize, 0usize);
        while pos < t && self.max_symbols > 0 {
            if !have_pred {
                ph.copy_from_slice(&h);
                pc.copy_from_slice(&c);
                match last {
                    None => self.step(&[], &mut ph, &mut pc, &mut gates, pool),
                    Some(id) => self.step(&self.embed[id as usize * n..(id as usize + 1) * n], &mut ph, &mut pc, &mut gates, pool),
                }
                ops::matmul(&ops::Weight::Dense(&self.pred_w), j, n, &ph, 1, &mut pred, None);
                ops::axpy(&mut pred, 1.0, &self.pred_b);
                have_pred = true;
            }
            let count = WINDOW.min(t - pos);
            for w in 0..count {
                let e = &enc[(pos + w) * j..(pos + w + 1) * j];
                for ((d, &a), &b) in hid[w * j..(w + 1) * j].iter_mut().zip(e).zip(&pred) {
                    *d = (a + b).max(0.0);
                }
            }
            ops::matmul(&ops::Weight::Dense(&self.out_w), self.classes, j, &hid[..count * j], count, &mut logits[..count * self.classes], pool);
            let mut emitted = None;
            for w in 0..count {
                let row = &mut logits[w * self.classes..(w + 1) * self.classes];
                ops::axpy(row, 1.0, &self.out_b);
                let token = argmax(row);
                if token != blank {
                    emitted = Some((w, token));
                    break;
                }
            }
            let Some((delta, token)) = emitted else {
                pos += count;
                repeats = 0;
                continue;
            };
            pos += delta;
            repeats = if delta > 0 { 1 } else { repeats + 1 };
            out.ids.push(token);
            out.frames.push(pos as u32);
            last = Some(token);
            h.copy_from_slice(&ph);
            c.copy_from_slice(&pc);
            have_pred = false;
            if repeats >= self.max_symbols {
                pos += 1;
                repeats = 0;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces() -> Vocab {
        Vocab::Pieces(["<unk>", "\u{2581}при", "вет", ",", "\u{2581}мир", "!"].iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn sentencepiece_units_join_into_text() {
        let v = pieces();
        assert_eq!(v.decode(&[1, 2, 3, 4, 5]), "привет, мир!");
        assert_eq!(v.decode(&[2, 0, 4]), "вет \u{2047}  мир");
        assert_eq!(v.decode(&[]), "");
        let chars = Vocab::Chars(["a", " ", "b"].iter().map(|s| s.to_string()).collect());
        assert_eq!(chars.decode(&[0, 1, 2, 2]), "a bb");
    }

    #[test]
    fn words_take_their_first_and_last_frames() {
        let head = Head { kind: Kind::Ctc(dummy_lin()), vocab: pieces(), blank: 6 };
        let d = Decoded { ids: vec![1, 2, 3, 4, 5], frames: vec![10, 12, 13, 20, 24] };
        let w = head.words(&d, 0.04, 100.0);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0], Word { text: "привет,".into(), start: 100.4, end: 100.56 });
        assert_eq!(w[1], Word { text: "мир!".into(), start: 100.8, end: 101.0 });
        // characters: a space closes the word and has no time of its own
        let chars = Head { kind: Kind::Ctc(dummy_lin()), vocab: Vocab::Chars(["a", " ", "b"].iter().map(|s| s.to_string()).collect()), blank: 3 };
        let d = Decoded { ids: vec![0, 0, 1, 2], frames: vec![1, 2, 3, 7] };
        let w = chars.words(&d, 0.5, 0.0);
        assert_eq!(w, vec![Word { text: "aa".into(), start: 0.5, end: 1.5 }, Word { text: "b".into(), start: 3.5, end: 4.0 }]);
    }

    fn dummy_lin() -> Lin {
        let model = Arc::new(tiny_model());
        Lin::load_named(&model, "w", None).unwrap()
    }

    fn tiny_model() -> CmfModel {
        use cortiq_core::format::{CmfHeader, TensorSpec};
        use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": super::super::ARCH_NAME, "hidden_size": 2, "intermediate_size": 2, "num_layers": 1,
            "num_attention_heads": 1, "num_kv_heads": 1, "head_dim": 2, "vocab_size": 2,
            "layer_types": ["FullAttention"], "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 16, "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
        }))
        .unwrap();
        let header = CmfHeader {
            format: "cmf".into(),
            version: cortiq_core::CMF_VERSION,
            arch,
            quant_type: QuantType::F32,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
            routing: None,
            genome: None,
            lineage: Vec::new(),
            router: None,
            segments: Vec::new(),
        };
        let data: Vec<u8> = [1f32, 0.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        // a directory of this call's own: the tests of this module run side by side
        static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("cortiq-gigaam-decode-{}-{call}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.cmf");
        CmfModel::write(&path, &header, &[TensorSpec { name: "w".into(), dtype: TensorDtype::F32, shape: vec![2, 2], data }], None, None).unwrap();
        let m = CmfModel::open(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        m
    }

    #[test]
    fn ctc_collapses_repeats_and_blanks() {
        // identity "logits": the encoder row is the class score; blank is class 2 of 3
        let model = Arc::new(tiny_model());
        let lin = Lin::load_named(&model, "w", None).unwrap();
        let head = Head { kind: Kind::Ctc(lin), vocab: Vocab::Chars(vec!["a".into()]), blank: 1 };
        // frames: a a blank a blank blank a a
        let seq = [0, 0, 1, 0, 1, 1, 0, 0];
        let enc: Vec<f32> = seq.iter().flat_map(|&c| if c == 0 { [1.0f32, 0.0] } else { [0.0, 1.0] }).collect();
        let d = head.decode(&enc, seq.len(), 2, None);
        assert_eq!(d.ids, vec![0, 0, 0]);
        assert_eq!(d.frames, vec![0, 3, 6]);
        assert_eq!(head.text(&d.ids), "aaa");
    }
}
