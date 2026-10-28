//! Teacher data as `compress` consumes it: every usable record of a
//! finalized `.tdata`, its exact token ids and its prompt / think / answer
//! boundaries, held in memory (a few bytes per token — 5 M tokens are
//! 20 MB) so each stage can cut its own windows and loss masks.

use crate::tdata::{Dataset, TdataSelector, TdataSplit, TdataSplitSelector};
use anyhow::{Context, Result, ensure};
use cortiq_core::CmfModel;
use serde_json::Value;

/// One usable teacher record.
#[derive(Clone, Debug)]
pub struct Rec {
    pub rid: String,
    pub split: TdataSplit,
    pub category: String,
    /// Full rendered sequence: prompt (template included), think, answer.
    pub ids: Vec<u32>,
    pub prompt: usize,
    pub think: usize,
    pub answer: usize,
}

impl Rec {
    /// Generated tokens (think + answer).
    pub fn generated(&self) -> usize {
        self.think + self.answer
    }
}

/// Which target tokens a loss or score covers (`--loss-on`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossOn {
    /// The answer only (reasoning is context).
    Answer,
    /// Everything the teacher generated: reasoning and answer.
    Generated,
    /// Every token after the first (plain language modelling, the notebook).
    All,
}

impl LossOn {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "answer" => Self::Answer,
            "generated" => Self::Generated,
            "all" => Self::All,
            other => anyhow::bail!("--loss-on {other:?}: expected answer, generated or all"),
        })
    }

    /// First target index scored in a record.
    pub fn first_target(self, r: &Rec) -> usize {
        match self {
            Self::Answer => r.prompt + r.think,
            Self::Generated => r.prompt,
            Self::All => 1,
        }
    }
}

pub struct Data {
    pub path: String,
    pub manifest: Value,
    pub recs: Vec<Rec>,
    /// Records the selector saw but the producer marked unusable.
    pub rejected: usize,
}

impl Data {
    /// Read every record of `spec` (a `.tdata` path; a `:split`/`=share`
    /// suffix narrows which records count as available at all).
    pub fn load(spec: &str) -> Result<Self> {
        let (reader, mut sel) = Dataset::open_selector(spec)?;
        // A bare path (or one with only `=SHARE`) means every split here:
        // compress needs train, val and test.
        let suffix = spec.rfind(".tdata").map_or("", |i| &spec[i + 6..]);
        if !suffix.starts_with(':') {
            sel.split = TdataSplitSelector::All;
        }
        let all = TdataSelector {
            split: TdataSplitSelector::All,
            ..sel.clone()
        }
        .include_rejected();
        let mut recs = Vec::new();
        let mut rejected = 0usize;
        for sample in reader.stream(all)? {
            let s = sample?;
            if !s.record.usable {
                rejected += 1;
                continue;
            }
            if !matches_split(sel.split, s.record.split) || !sel.share.contains_rid(&s.record.rid) {
                continue;
            }
            let ix = s.index;
            recs.push(Rec {
                rid: s.record.rid.clone(),
                split: s.record.split,
                category: s.record.category.clone().unwrap_or_default(),
                ids: s.tokens,
                prompt: ix.prompt as usize,
                think: ix.think as usize,
                answer: ix.answer as usize,
            });
        }
        ensure!(!recs.is_empty(), "{spec}: no usable records");
        Ok(Self {
            path: reader.root().display().to_string(),
            manifest: reader.manifest.raw.clone(),
            recs,
            rejected,
        })
    }

    pub fn split(&self, s: TdataSplit) -> impl Iterator<Item = &Rec> {
        self.recs.iter().filter(move |r| r.split == s)
    }

    pub fn count(&self, s: TdataSplit) -> (usize, usize) {
        self.split(s).fold((0, 0), |(n, t), r| (n + 1, t + r.ids.len()))
    }

    /// The manifest's `teacher.directory_hash` (identity of the teacher).
    pub fn teacher_id(&self) -> String {
        self.manifest
            .pointer("/teacher/directory_hash")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string()
    }

    /// Refuse a model whose tokenizer is not the one the data was written
    /// with: the ids would mean other words. Vocabulary bytes and the
    /// tokenizer configuration must hash the same as the data's contract.
    pub fn check_tokenizer(&self, model: &CmfModel, what: &str) -> Result<()> {
        let tok = self
            .manifest
            .get("tokenizer")
            .context("teacher data has no tokenizer contract (manifest.tokenizer)")?;
        let vocab = model
            .vocab
            .as_deref()
            .with_context(|| format!("{what} has no embedded tokenizer"))?;
        let want_vocab = tok.get("vocab_sha256").and_then(Value::as_str).unwrap_or("");
        let have_vocab = crate::knowledge::sha256_hex(vocab);
        let cfg = serde_json::to_vec(&model.header.tokenizer_config)?;
        let want_cfg = tok.get("config_sha256").and_then(Value::as_str).unwrap_or("");
        let have_cfg = crate::knowledge::sha256_hex(&cfg);
        ensure!(
            want_vocab == have_vocab,
            "{what}: tokenizer vocabulary differs from the teacher data's ({} vs {}); \
             the token ids would not mean the same words — generate the data with a teacher of the same tokenizer",
            &have_vocab[..12.min(have_vocab.len())],
            &want_vocab[..12.min(want_vocab.len())]
        );
        if want_cfg != have_cfg {
            eprintln!(
                "  note: {what}: tokenizer configuration differs from the data's (vocabulary matches) — template/special-token settings may differ"
            );
        }
        if let Some(v) = tok.get("vocab_size").and_then(Value::as_u64) {
            for r in &self.recs {
                if let Some(&bad) = r.ids.iter().find(|&&t| t as u64 >= v.max(1)) {
                    anyhow::bail!("{}: token id {bad} outside the vocabulary ({v})", r.rid);
                }
            }
        }
        Ok(())
    }
}

fn matches_split(sel: TdataSplitSelector, s: TdataSplit) -> bool {
    match sel {
        TdataSplitSelector::All => true,
        TdataSplitSelector::Train => s == TdataSplit::Train,
        TdataSplitSelector::Val => s == TdataSplit::Val,
        TdataSplitSelector::Test => s == TdataSplit::Test,
    }
}

/// A flat token stream of whole records with per-token loss weights and
/// record starts — the shape `fcd::FcdStream` takes. `pad` tokens of
/// weight 0 close the stream so a window can start at any record.
pub struct Stream {
    pub ids: Vec<u32>,
    pub weight: Vec<f32>,
    pub starts: Vec<usize>,
    pub scored: usize,
}

impl Stream {
    pub fn build<'a>(recs: impl Iterator<Item = &'a Rec>, on: LossOn, pad_id: u32, pad: usize) -> Self {
        let (mut ids, mut weight, mut starts) = (Vec::new(), Vec::new(), Vec::new());
        let mut scored = 0usize;
        for r in recs {
            starts.push(ids.len());
            let first = on.first_target(r).max(1);
            for (i, &t) in r.ids.iter().enumerate() {
                ids.push(t);
                let w = if i >= first { 1.0 } else { 0.0 };
                if w > 0.0 {
                    scored += 1;
                }
                weight.push(w);
            }
        }
        ids.extend(std::iter::repeat_n(pad_id, pad));
        weight.extend(std::iter::repeat_n(0.0, pad));
        Self { ids, weight, starts, scored }
    }
}
