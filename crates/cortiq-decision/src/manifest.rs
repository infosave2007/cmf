//! `decision.manifest` and `decision.skill.{id}.manifest` schemas (spec §2.3, §2.4),
//! the overlay manifest of a generation (§5.10), the hashing record (§2.6) and the
//! tensor names of the decision profile (§2.2).
//!
//! Every manifest is stored as canonical JSON ([`crate::canonical`]) in a U8
//! tensor and parsed into the typed structs below with `deny_unknown_fields`:
//!
//! * `decision.manifest` ([`DecisionManifest`]): the representation (encoder,
//!   hashing contract, signal), `representation_id = sha256(canonical(representation))`
//!   and the sha256 of every skill manifest; `model_sha = sha256(bytes)`;
//! * `decision.skill.{id}.manifest` ([`SkillManifest`]): recipe, labels, tasks with
//!   the sha256 of every `mean`/`basis`, the certified gate and its evidence, the
//!   rubric, the input data, the rows blob and, after offline learning, the
//!   `learned` record (§5.14);
//! * `decision.overlay.manifest` ([`OverlayManifest`]): a generation's full skill
//!   manifests relative to the base file.
//!
//! The validators here check one manifest on its own; [`crate::container`] checks
//! them against the tensors of a file.
//!
//! Numbers of the gate are the exact f64 value of an f32 and are read back as
//! f32; `err_mean`/`err_std` are f64 and used as f32 (spec §2.4).

use crate::canonical;
use crate::certify::{self, Certification, GridRow};
use crate::fit;
use crate::hashfeat;
use crate::resonance::ErrStats;
use crate::rows;
use anyhow::{Result, bail, ensure};
use cortiq_core::TensorDtype;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

// ------------------------------------------------------------------ names and constants

/// Arch name (profile) of a full decision file.
pub const BASE_PROFILE: &str = "cortiq-decision-ph-v1";
/// Arch name (profile) of a generation overlay (spec §5.10).
pub const OVERLAY_PROFILE: &str = "cortiq-decision-overlay-v1";
/// Research profiles of v3 and earlier; refused with a rebuild hint (spec §2.8).
pub const LEGACY_PROFILES: [&str; 3] = [
    "cortiq-decision-affine-v1",
    "cortiq-decision-affine-core-v1",
    "cortiq-decision-fcd-v1",
];
/// `decision.manifest` schema.
pub const MANIFEST_SCHEMA: &str = "cortiq-decision/1";
/// Skill manifest schema.
pub const SKILL_SCHEMA: &str = "cortiq-decision-skill/1";
/// Overlay manifest schema.
pub const OVERLAY_SCHEMA: &str = "cortiq-decision-overlay/1";
/// Largest manifest tensor (spec §2.3).
pub const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
/// Default model id and display name (spec §2.3).
pub const DEFAULT_MODEL_ID: &str = "cortiq/decision";
pub const DEFAULT_NAME: &str = "Cortiq Decision";

/// Tensor of the file manifest.
pub const MANIFEST_TENSOR: &str = "decision.manifest";
/// Tensor of the overlay manifest.
pub const OVERLAY_MANIFEST_TENSOR: &str = "decision.overlay.manifest";
/// Prefix of every encoder tensor.
pub const ENCODER_PREFIX: &str = "decision.encoder.";
/// The WordPiece vocabulary (the bytes of vocab.txt).
pub const VOCAB_TENSOR: &str = "decision.encoder.vocab";
/// φ_P of the golden texts, `[ENCODER_GOLDEN_COUNT, dim]`.
pub const GOLDEN_TENSOR: &str = "decision.encoder.golden";
/// Number of encoder golden texts.
pub const ENCODER_GOLDEN_COUNT: usize = 8;
/// Longest golden text or label description handled here (bytes).
pub const MAX_TEXT_BYTES: usize = 32 * 1024;

/// Encoder record values implemented by the native encoder (spec §1, §2.3).
pub const ENCODER_KIND: &str = "bert-wordpiece-v1";
pub const ACTIVATION: &str = "gelu_erf";
pub const POOLING: &str = "mean-all-tokens";
pub const NORMALIZATION: [&str; 2] = ["l2-eps1e-12-seq-f32", "l2-router-mul-inv-seq-f32"];
pub const TOKENIZER_MODEL: &str = "wordpiece";
pub const PRE_TOKENIZER: &str = "bert";
pub const TEMPLATE: &str = "[CLS] $A [SEP]";
pub const TRUNCATION_DIRECTION: &str = "right";
/// Source name of the release encoder.
pub const RELEASE_ENCODER_SOURCE: &str =
    "cortiq-router registry_bake/encoder.onnx (bge-small-en-v1.5, MLP pruned 75% by Cortiq NVG)";
/// sha256 of the release encoder.onnx.
pub const RELEASE_ONNX_SHA256: &str =
    "a59c5dbd48ff866dcb26603fcb30b54e904470a146e5e3bc3ee75a64b069e442";
/// Encoder and signal dimensions of the release file: `x = [φ_P(384) ; 0.5·φ_H(4096)]`.
pub const RELEASE_ENCODER_DIM: usize = 384;
pub const RELEASE_SIGNAL_DIM: usize = RELEASE_ENCODER_DIM + hashfeat::DIM;

/// Signal record values (spec §2.3).
pub const SIGNAL_KIND: &str = "concat-v1";
pub const PHI_P: &str = "phi_P";
pub const PHI_H: &str = "phi_H";
pub const PHI_P_WEIGHT: f64 = 1.0;
pub const PHI_H_WEIGHT: f64 = rows::PHI_H_WEIGHT as f64;

/// Recipe values of the only implemented topology (spec §2.4, §3.4).
pub const TOPOLOGY: &str = "affine";
pub const K_RULE: &str = "min(K,n-1)";
/// Gate rule values (spec §2.4, §3.6).
pub const ALPHA_RULE: &str = "0.05/14";
/// Holdout rule recorded in the skill manifest (spec §3.2).
pub const HOLDOUT_RULE: &str = "per label, calibration rows in sha256 order, index%5==4";
/// Calibration source values.
pub const CALIBRATION_FROM_FILE: &str = "file";
pub const CALIBRATION_CARVE_OUT: &str = "carve-out";
/// Calibration source of an auto-skill: its calibration subset is recomputed
/// from the learned rows at every attempt ([`crate::certify::HALVES_RULE_AUTO`]),
/// never stored (0.8.6).
pub const CALIBRATION_LEARNED: &str = "learned";

/// Reserved id prefix of an auto-skill: a choice contract learned from oracle
/// answers (0.8.6). A user skill never gets this prefix
/// ([`crate::container::FileBuilder::add_skill`], the offline builder).
pub const AUTO_SKILL_PREFIX: &str = "auto-";
/// Hex characters of the contract sha256 in an auto-skill id.
pub const AUTO_SKILL_ID_HEX: usize = 12;

/// Eight encoder golden texts a builder may use (the file records its own list).
pub const DEFAULT_ENCODER_GOLDEN_TEXTS: [&str; ENCODER_GOLDEN_COUNT] = [
    "I still have not received my new card",
    "How do I top up my account with a cheque?",
    "what's the exchange rate for EUR -> USD?",
    "PIN blocked!!!",
    "Привет, мир! Как мне пополнить счёт?",
    "日本語のテキスト 口座の残高を教えて",
    "café crème naïve façade 😀",
    "",
];

/// `decision.skill.{id}.manifest`.
pub fn skill_manifest_tensor(id: &str) -> String {
    format!("decision.skill.{id}.manifest")
}

/// `decision.skill.{id}.task.{i}.mean`.
pub fn task_mean_tensor(id: &str, i: usize) -> String {
    format!("decision.skill.{id}.task.{i}.mean")
}

/// `decision.skill.{id}.task.{i}.basis`.
pub fn task_basis_tensor(id: &str, i: usize) -> String {
    format!("decision.skill.{id}.task.{i}.basis")
}

/// `decision.skill.{id}.rows`.
pub fn rows_tensor(id: &str) -> String {
    format!("decision.skill.{id}.rows")
}

/// `decision.skill.{id}.rows.learned` (learned rows kept apart from the build rows).
pub fn rows_learned_tensor(id: &str) -> String {
    format!("decision.skill.{id}.rows.learned")
}

/// Is `id` a skill id (`[a-z0-9][a-z0-9_-]{0,63}`)?
pub fn valid_skill_id(id: &str) -> bool {
    let b = id.as_bytes();
    !b.is_empty()
        && b.len() <= 64
        && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
        && b[1..]
            .iter()
            .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'-')
}

/// Does `id` carry the reserved auto-skill prefix?
pub fn is_auto_skill_id(id: &str) -> bool {
    id.starts_with(AUTO_SKILL_PREFIX)
}

/// The option ids of a choice contract in their canonical form: sorted
/// bytewise, duplicates dropped, as a JSON array. Hashing the canonical array
/// (as `cache::skill_scope` does) keeps the key injective: ids may hold any
/// byte, so a joined string would not be.
pub fn contract_ids(ids: &[&str]) -> Vec<String> {
    let set: BTreeSet<&str> = ids.iter().copied().collect();
    set.into_iter().map(str::to_string).collect()
}

/// sha256 of a contract: the canonical JSON array of [`contract_ids`].
pub fn contract_sha256(ids: &[&str]) -> String {
    let v = Value::Array(contract_ids(ids).into_iter().map(Value::String).collect());
    canonical::sha256_hex(&v)
}

/// The id of the auto-skill of a choice contract: `auto-` + the first 12 hex
/// characters of [`contract_sha256`]. Order-independent; a valid skill id.
pub fn auto_skill_id(ids: &[&str]) -> String {
    format!(
        "{AUTO_SKILL_PREFIX}{}",
        &contract_sha256(ids)[..AUTO_SKILL_ID_HEX]
    )
}

/// Is `label` a label (1..=256 bytes)?
pub fn valid_label(label: &str) -> bool {
    (1..=256).contains(&label.len())
}

/// 64 lowercase hex characters.
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Lowercase hex sha256 of bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Is `x` the exact f64 value of a finite f32?
pub fn is_f32_exact(x: f64) -> bool {
    x.is_finite() && (x as f32) as f64 == x
}

/// `created_unix` of a new manifest: `SOURCE_DATE_EPOCH` when set, else 0 (spec §2.3).
pub fn created_unix_from_env() -> Result<u64> {
    parse_source_date_epoch(std::env::var_os("SOURCE_DATE_EPOCH").as_deref())
}

/// `SOURCE_DATE_EPOCH` as a `created_unix`: unset or empty is 0, anything else must
/// be a non-negative decimal integer.
pub fn parse_source_date_epoch(v: Option<&std::ffi::OsStr>) -> Result<u64> {
    let Some(v) = v else { return Ok(0) };
    let s = v
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("SOURCE_DATE_EPOCH is not UTF-8"))?;
    if s.is_empty() {
        return Ok(0);
    }
    s.parse::<u64>()
        .map_err(|_| anyhow::anyhow!("SOURCE_DATE_EPOCH '{s}' is not a non-negative integer"))
}

fn check_sha(what: &str, s: &str) -> Result<()> {
    ensure!(is_sha256_hex(s), "{what} is not a lowercase hex sha256");
    Ok(())
}

fn check_text(what: &str, s: &str, max: usize) -> Result<()> {
    ensure!(
        !s.is_empty() && s.len() <= max,
        "{what} must have 1..={max} bytes"
    );
    Ok(())
}

// ------------------------------------------------------------------ decision.manifest

/// `decision.manifest` (spec §2.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionManifest {
    pub schema: String,
    pub profile: String,
    pub model_id: String,
    pub name: String,
    pub created_unix: u64,
    /// Kept as parsed: its canonical bytes define `representation_id`.
    /// Typed view: [`Representation::from_value`].
    pub representation: Value,
    pub representation_id: String,
    pub skills: Vec<SkillRef>,
    pub generation: u64,
}

/// A skill of the file and the sha256 of its manifest bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillRef {
    pub id: String,
    pub manifest_sha256: String,
}

impl DecisionManifest {
    /// Structural checks (schema, profile, identities, skill ids). The
    /// representation is checked by [`Representation::validate`] and
    /// [`check_hashing`]; `representation_id` by [`representation_id`].
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == MANIFEST_SCHEMA,
            "unsupported manifest schema '{}' (expected {MANIFEST_SCHEMA})",
            self.schema
        );
        ensure!(
            self.profile == BASE_PROFILE,
            "manifest profile '{}' (expected {BASE_PROFILE})",
            self.profile
        );
        check_text("model_id", &self.model_id, 256)?;
        check_text("name", &self.name, 256)?;
        check_sha("representation_id", &self.representation_id)?;
        let mut seen = BTreeSet::new();
        for s in &self.skills {
            ensure!(valid_skill_id(&s.id), "invalid skill id '{}'", s.id);
            ensure!(seen.insert(s.id.as_str()), "duplicate skill id '{}'", s.id);
            check_sha(
                &format!("skill '{}' manifest_sha256", s.id),
                &s.manifest_sha256,
            )?;
        }
        Ok(())
    }
}

/// `representation_id` = sha256 of the canonical representation.
pub fn representation_id(representation: &Value) -> String {
    canonical::sha256_hex(representation)
}

// ------------------------------------------------------------------ representation

/// Typed view of the representation (spec §2.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Representation {
    pub encoder: EncoderRecord,
    /// The hashing record (spec §2.6), compared with the compiled contract by
    /// [`check_hashing`].
    pub hashing: Value,
    pub signal: SignalRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderRecord {
    pub kind: String,
    pub source: EncoderSource,
    pub config: EncoderConfig,
    pub tokenizer: TokenizerRecord,
    pub pooling: String,
    pub normalization: Vec<String>,
    pub dim: u64,
    /// See [`tensors_sha256`]: every `decision.encoder.*` tensor in name order.
    pub tensors_sha256: String,
    pub golden_texts: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderSource {
    pub name: String,
    pub onnx_sha256: String,
    pub tokenizer_json_sha256: String,
    /// sha256 of vocab.txt, which is also the `decision.encoder.vocab` tensor.
    pub vocab_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderConfig {
    pub layers: u64,
    pub hidden: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub intermediate: u64,
    pub max_position: u64,
    pub vocab: u64,
    pub type_vocab: u64,
    pub token_type: u64,
    pub ln_eps: f64,
    pub activation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerRecord {
    pub model: String,
    pub unk: String,
    pub prefix: String,
    pub max_input_chars_per_word: u64,
    pub normalizer: NormalizerRecord,
    pub pre_tokenizer: String,
    pub template: String,
    pub ids: SpecialIds,
    pub truncation: Truncation,
    /// Versions of the crates the Unicode tables come from.
    pub tables: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizerRecord {
    pub clean_text: bool,
    pub handle_chinese_chars: bool,
    /// `null` (HF: strip when lowercasing).
    pub strip_accents: Option<bool>,
    pub lowercase: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialIds {
    pub pad: u64,
    pub unk: u64,
    pub cls: u64,
    pub sep: u64,
    pub mask: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Truncation {
    pub max_length: u64,
    pub direction: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalRecord {
    pub kind: String,
    pub parts: Vec<SignalPart>,
    pub dim: u64,
    pub renormalize: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalPart {
    pub name: String,
    pub dim: u64,
    pub weight: f64,
}

impl NormalizerRecord {
    /// `BertNormalizer{clean_text, handle_chinese_chars, strip_accents=null, lowercase}`.
    pub fn bert_uncased() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    }
}

impl TokenizerRecord {
    /// The HF bert-uncased WordPiece set-up the native tokenizer implements.
    pub fn bert_uncased(ids: SpecialIds, max_length: u64, tables: impl Into<String>) -> Self {
        Self {
            model: TOKENIZER_MODEL.into(),
            unk: "[UNK]".into(),
            prefix: "##".into(),
            max_input_chars_per_word: 100,
            normalizer: NormalizerRecord::bert_uncased(),
            pre_tokenizer: PRE_TOKENIZER.into(),
            template: TEMPLATE.into(),
            ids,
            truncation: Truncation {
                max_length,
                direction: TRUNCATION_DIRECTION.into(),
            },
            tables: tables.into(),
        }
    }
}

impl SignalRecord {
    /// The PH signal `[φ_P ; 0.5·φ_H]` of an encoder of `dim_p`.
    pub fn ph(dim_p: u64) -> Self {
        Self {
            kind: SIGNAL_KIND.into(),
            parts: vec![
                SignalPart {
                    name: PHI_P.into(),
                    dim: dim_p,
                    weight: PHI_P_WEIGHT,
                },
                SignalPart {
                    name: PHI_H.into(),
                    dim: hashfeat::DIM as u64,
                    weight: PHI_H_WEIGHT,
                },
            ],
            dim: dim_p + hashfeat::DIM as u64,
            renormalize: false,
        }
    }
}

impl EncoderConfig {
    /// Checks of the values the native encoder implements.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=64).contains(&self.layers),
            "encoder layers must be 1..=64"
        );
        ensure!(
            (1..=8192).contains(&self.hidden),
            "encoder hidden must be 1..=8192"
        );
        ensure!(
            self.heads >= 1 && self.head_dim >= 1 && self.heads * self.head_dim == self.hidden,
            "heads × head_dim must equal hidden"
        );
        ensure!(
            (1..=65536).contains(&self.intermediate),
            "encoder intermediate must be 1..=65536"
        );
        ensure!(
            (2..=65536).contains(&self.max_position),
            "max_position must be 2..=65536"
        );
        ensure!(
            (1..=1 << 22).contains(&self.vocab),
            "vocab must be 1..=4194304"
        );
        ensure!(
            (1..=64).contains(&self.type_vocab) && self.token_type < self.type_vocab,
            "token_type must index type_vocab"
        );
        ensure!(
            self.ln_eps.is_finite() && self.ln_eps > 0.0,
            "ln_eps must be positive"
        );
        ensure!(
            self.activation == ACTIVATION,
            "unsupported activation '{}'",
            self.activation
        );
        Ok(())
    }

    /// Names (without [`ENCODER_PREFIX`]) and shapes of the F32 weight tensors.
    pub fn weight_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden as usize;
        let f = self.intermediate as usize;
        let mut v = vec![
            (
                "embeddings.word_embeddings.weight".to_string(),
                vec![self.vocab as usize, h],
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                vec![self.max_position as usize, h],
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                vec![self.type_vocab as usize, h],
            ),
            ("embeddings.LayerNorm.weight".to_string(), vec![h]),
            ("embeddings.LayerNorm.bias".to_string(), vec![h]),
        ];
        for l in 0..self.layers {
            let p = format!("layer.{l}.");
            for m in ["query", "key", "value"] {
                v.push((format!("{p}attention.self.{m}.weight"), vec![h, h]));
                v.push((format!("{p}attention.self.{m}.bias"), vec![h]));
            }
            v.push((format!("{p}attention.output.dense.weight"), vec![h, h]));
            v.push((format!("{p}attention.output.dense.bias"), vec![h]));
            v.push((format!("{p}attention.output.LayerNorm.weight"), vec![h]));
            v.push((format!("{p}attention.output.LayerNorm.bias"), vec![h]));
            v.push((format!("{p}intermediate.dense.weight"), vec![f, h]));
            v.push((format!("{p}intermediate.dense.bias"), vec![f]));
            v.push((format!("{p}output.dense.weight"), vec![h, f]));
            v.push((format!("{p}output.dense.bias"), vec![h]));
            v.push((format!("{p}output.LayerNorm.weight"), vec![h]));
            v.push((format!("{p}output.LayerNorm.bias"), vec![h]));
        }
        v
    }

    /// Every encoder tensor with its dtype and shape (full names; the vocab's
    /// shape is `[vocab_bytes]`).
    pub fn tensor_layout(&self, vocab_bytes: usize) -> Vec<(String, TensorDtype, Vec<usize>)> {
        let mut v: Vec<(String, TensorDtype, Vec<usize>)> = self
            .weight_shapes()
            .into_iter()
            .map(|(n, s)| (format!("{ENCODER_PREFIX}{n}"), TensorDtype::F32, s))
            .collect();
        v.push((
            GOLDEN_TENSOR.to_string(),
            TensorDtype::F32,
            vec![ENCODER_GOLDEN_COUNT, self.hidden as usize],
        ));
        v.push((VOCAB_TENSOR.to_string(), TensorDtype::U8, vec![vocab_bytes]));
        v
    }
}

impl EncoderRecord {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.kind == ENCODER_KIND,
            "unsupported encoder kind '{}'",
            self.kind
        );
        check_text("encoder source name", &self.source.name, 1024)?;
        check_sha("onnx_sha256", &self.source.onnx_sha256)?;
        check_sha("tokenizer_json_sha256", &self.source.tokenizer_json_sha256)?;
        check_sha("vocab_sha256", &self.source.vocab_sha256)?;
        self.config.validate()?;
        let t = &self.tokenizer;
        ensure!(
            t.model == TOKENIZER_MODEL,
            "unsupported tokenizer model '{}'",
            t.model
        );
        check_text("tokenizer unk", &t.unk, 256)?;
        check_text("tokenizer prefix", &t.prefix, 256)?;
        ensure!(
            t.max_input_chars_per_word >= 1,
            "max_input_chars_per_word must be positive"
        );
        ensure!(
            t.normalizer == NormalizerRecord::bert_uncased(),
            "unsupported normalizer (implemented: clean_text, handle_chinese_chars, strip_accents null, lowercase)"
        );
        ensure!(
            t.pre_tokenizer == PRE_TOKENIZER,
            "unsupported pre_tokenizer '{}'",
            t.pre_tokenizer
        );
        ensure!(
            t.template == TEMPLATE,
            "unsupported template '{}'",
            t.template
        );
        let ids = &t.ids;
        for (n, id) in [
            ("pad", ids.pad),
            ("unk", ids.unk),
            ("cls", ids.cls),
            ("sep", ids.sep),
            ("mask", ids.mask),
        ] {
            ensure!(
                id < self.config.vocab,
                "special id {n} {id} outside the vocab"
            );
        }
        ensure!(
            t.truncation.direction == TRUNCATION_DIRECTION,
            "unsupported truncation direction '{}'",
            t.truncation.direction
        );
        ensure!(
            t.truncation.max_length >= 2 && t.truncation.max_length <= self.config.max_position,
            "truncation max_length must be 2..=max_position"
        );
        check_text("tokenizer tables", &t.tables, 1024)?;
        ensure!(
            self.pooling == POOLING,
            "unsupported pooling '{}'",
            self.pooling
        );
        ensure!(
            self.normalization
                .iter()
                .map(String::as_str)
                .eq(NORMALIZATION),
            "unsupported normalization {:?}",
            self.normalization
        );
        ensure!(
            self.dim == self.config.hidden,
            "encoder dim {} != hidden {}",
            self.dim,
            self.config.hidden
        );
        check_sha("tensors_sha256", &self.tensors_sha256)?;
        ensure!(
            self.golden_texts.len() == ENCODER_GOLDEN_COUNT,
            "expected {ENCODER_GOLDEN_COUNT} encoder golden texts, found {}",
            self.golden_texts.len()
        );
        for g in &self.golden_texts {
            ensure!(g.len() <= MAX_TEXT_BYTES, "golden text longer than 32 KiB");
        }
        Ok(())
    }
}

impl Representation {
    /// A representation of `encoder` with the compiled hashing contract and the PH signal.
    pub fn new(encoder: EncoderRecord) -> Self {
        let dim = encoder.dim;
        Self {
            encoder,
            hashing: hashfeat::contract_record(),
            signal: SignalRecord::ph(dim),
        }
    }

    /// Typed view of a stored representation.
    pub fn from_value(v: &Value) -> Result<Self> {
        Self::deserialize(v).map_err(|e| anyhow::anyhow!("representation: {e}"))
    }

    pub fn to_value(&self) -> Result<Value> {
        Ok(serde_json::to_value(self)?)
    }

    /// Encoder and signal checks (the hashing record: [`check_hashing`]).
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        let s = &self.signal;
        ensure!(
            s.kind == SIGNAL_KIND,
            "unsupported signal kind '{}'",
            s.kind
        );
        ensure!(!s.renormalize, "a renormalized signal is not implemented");
        ensure!(
            *s == SignalRecord::ph(self.encoder.dim),
            "signal must be [phi_P({}) weight 1 ; phi_H({}) weight 0.5], dim {}",
            self.encoder.dim,
            hashfeat::DIM,
            self.encoder.dim + hashfeat::DIM as u64
        );
        ensure!(self.hashing.is_object(), "hashing record is not an object");
        Ok(())
    }

    /// `dim_p + dim_h`.
    pub fn signal_dim(&self) -> usize {
        self.signal.dim as usize
    }

    /// The encoder (φ_P) dimension.
    pub fn encoder_dim(&self) -> usize {
        self.encoder.dim as usize
    }

    /// The hashing (φ_H) dimension.
    pub fn hashing_dim(&self) -> usize {
        hashfeat::DIM
    }
}

/// Compare a stored hashing record with the contract compiled into this build
/// (spec §2.6): every field and every golden sha256 must be equal bit for bit.
/// A different `unicode_version` is only a warning (returned).
pub fn check_hashing(stored: &Value) -> std::result::Result<Vec<String>, String> {
    let expected = hashfeat::contract_record();
    let Value::Object(s) = stored else {
        return Err("hashing record is not an object".into());
    };
    let Value::Object(e) = &expected else {
        return Err("compiled hashing record is not an object".into());
    };
    let mut warnings = Vec::new();
    let su = s.get("unicode_version").and_then(Value::as_str);
    let eu = e
        .get("unicode_version")
        .and_then(Value::as_str)
        .unwrap_or("");
    match su {
        None => return Err("hashing record has no unicode_version".into()),
        Some(v) if v != eu => warnings.push(format!(
            "hashing contract recorded with Unicode {v}, this build uses Unicode {eu} (golden hashes still equal)"
        )),
        Some(_) => {}
    }
    let strip = |m: &Map<String, Value>| {
        let mut m = m.clone();
        m.remove("unicode_version");
        Value::Object(m)
    };
    let (sv, ev) = (strip(s), strip(e));
    if canonical::to_string(&sv) == canonical::to_string(&ev) {
        return Ok(warnings);
    }
    // Name the first difference.
    let (Value::Object(sm), Value::Object(em)) = (&sv, &ev) else {
        unreachable!("objects above")
    };
    let keys: BTreeSet<&String> = sm.keys().chain(em.keys()).collect();
    for k in keys {
        let (a, b) = (sm.get(k.as_str()), em.get(k.as_str()));
        if a.map(canonical::to_string) == b.map(canonical::to_string) {
            continue;
        }
        if k == "golden" {
            if let (Some(Value::Array(ga)), Some(Value::Array(gb))) = (a, b) {
                if ga.len() != gb.len() {
                    return Err(format!(
                        "{} golden texts in the file, {} in this build",
                        ga.len(),
                        gb.len()
                    ));
                }
                for (i, (x, y)) in ga.iter().zip(gb).enumerate() {
                    if x.get("text") != y.get("text") {
                        return Err(format!("golden[{i}] text differs"));
                    }
                    if x != y {
                        return Err(format!(
                            "golden[{i}] dense sha256 differs (file {}, this build {})",
                            x.get("dense_f32le_sha256")
                                .and_then(Value::as_str)
                                .unwrap_or("?"),
                            y.get("dense_f32le_sha256")
                                .and_then(Value::as_str)
                                .unwrap_or("?")
                        ));
                    }
                }
            }
        }
        return Err(match (a, b) {
            (None, _) => format!("field '{k}' missing in the file"),
            (_, None) => format!("unknown field '{k}' in the file"),
            _ => format!("field '{k}' differs"),
        });
    }
    Err("record differs".into())
}

/// One tensor entering [`tensors_sha256`].
#[derive(Clone, Copy, Debug)]
pub struct TensorDigest<'a> {
    pub name: &'a str,
    pub dtype: TensorDtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

/// sha256 over (name, dtype, shape, bytes) of tensors in byte order of their names:
/// per tensor `u64le(len(name)) ‖ name ‖ u8(dtype id) ‖ u64le(ndim) ‖ u64le(dim)… ‖
/// u64le(len(bytes)) ‖ bytes`.
pub fn tensors_sha256<'a>(tensors: impl IntoIterator<Item = TensorDigest<'a>>) -> String {
    let mut v: Vec<TensorDigest<'a>> = tensors.into_iter().collect();
    v.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut h = Sha256::new();
    for t in v {
        h.update((t.name.len() as u64).to_le_bytes());
        h.update(t.name.as_bytes());
        h.update([t.dtype.id()]);
        h.update((t.shape.len() as u64).to_le_bytes());
        for &d in t.shape {
            h.update((d as u64).to_le_bytes());
        }
        h.update((t.data.len() as u64).to_le_bytes());
        h.update(t.data);
    }
    format!("{:x}", h.finalize())
}

// ------------------------------------------------------------------ skill manifest

/// `decision.skill.{id}.manifest` (spec §2.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillManifest {
    pub schema: String,
    pub id: String,
    pub taxonomy_version: u64,
    pub representation_id: String,
    pub recipe: Recipe,
    /// Task labels in task order (`labels[i] == tasks[i].label`).
    pub labels: Vec<String>,
    pub tasks: Vec<TaskRecord>,
    pub gate: Gate,
    /// `null` when the skill was trained without `--question`.
    pub rubric: Option<Rubric>,
    pub data: DataRecord,
    pub rows: RowsRecord,
    /// Learned rows kept in their own tensor (a generation, or a file materialised
    /// from one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_learned: Option<LearnedRowsRecord>,
    /// Offline learning through the oracle (spec §5.14).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learned: Option<LearnedRecord>,
}

/// Fit recipe (spec §2.4, §3.4); `K` is per skill (spec §3.9).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub topology: String,
    #[serde(rename = "K")]
    pub k_max: u64,
    pub k_rule: String,
    pub eig_drop: f64,
    pub min_rows_active: u64,
    pub err_std_floor: f64,
    pub fit: String,
    pub sign: String,
    /// Where `K` came from when it is not the default (e.g. the CV report and its sha256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_source: Option<String>,
}

impl Recipe {
    /// The implemented recipe with `K` directions at most.
    pub fn standard(k_max: u64) -> Self {
        Self {
            topology: TOPOLOGY.into(),
            k_max,
            k_rule: K_RULE.into(),
            eig_drop: fit::EIG_DROP,
            min_rows_active: fit::MIN_ROWS_ACTIVE as u64,
            err_std_floor: fit::ERR_STD_FLOOR,
            fit: fit::FIT_NAME.into(),
            sign: fit::SIGN_RULE.into(),
            k_source: None,
        }
    }

    pub fn validate(&self, signal_dim: usize) -> Result<()> {
        ensure!(
            (1..=signal_dim as u64).contains(&self.k_max),
            "recipe K {} outside 1..={signal_dim}",
            self.k_max
        );
        let mut std = Self::standard(self.k_max);
        std.k_source.clone_from(&self.k_source);
        ensure!(
            *self == std,
            "unsupported recipe (implemented: affine, min(K,n-1), eig_drop 1e-8, min_rows_active 2, err_std_floor 1e-4, gram-eigh-f64-v1, max-abs-positive)"
        );
        if let Some(s) = &self.k_source {
            check_text("recipe k_source", s, 1024)?;
        }
        Ok(())
    }
}

/// Task state (spec §2.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// Scored.
    Active,
    /// A cold-start label waiting for enough examples; not scored.
    Quarantined,
    /// Too few rows to be scored.
    Inactive,
}

/// Where a task's label comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskOrigin {
    /// A label of the training data.
    Data,
    /// A label first seen from the oracle or feedback (spec §5.8).
    ColdStart,
}

/// One task of a skill.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRecord {
    pub i: u64,
    pub label: String,
    pub state: TaskState,
    pub origin: TaskOrigin,
    /// Rank of the basis (`decision.skill.{id}.task.{i}.basis` is `[k, dim]`).
    pub k: u64,
    /// Rows the topology was fitted on.
    pub n_train: u64,
    pub err_mean: f64,
    pub err_std: f64,
    /// sha256 of the `mean` tensor bytes; `null` when the task has no topology yet.
    pub mean_sha256: Option<String>,
    /// sha256 of the `basis` tensor bytes; `null` exactly when `k = 0`.
    pub basis_sha256: Option<String>,
}

impl TaskRecord {
    /// The runtime f32 statistics.
    pub fn stats(&self) -> ErrStats {
        ErrStats::from_f64(self.err_mean, self.err_std)
    }

    pub fn is_active(&self) -> bool {
        self.state == TaskState::Active
    }

    pub fn has_topology(&self) -> bool {
        self.mean_sha256.is_some()
    }
}

/// The certified gate and its evidence (spec §2.4, §3.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub temperature: f64,
    pub novelty_theta: f64,
    pub tau: f64,
    pub certified: bool,
    pub rule: GateRule,
    pub evidence: Evidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateRule {
    pub thresholds: Vec<f64>,
    pub alpha: String,
    pub min_accepted: u64,
    pub target: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub even: EvenEvidence,
    pub odd: OddEvidence,
    pub calibration: CalibrationEvidence,
}

/// The even half: `T` and `θ`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvenEvidence {
    /// Rows of the even half.
    pub n: u64,
    /// Even rows that entered `T` (label with an active task).
    pub n_t: u64,
    /// Mean NLL at the chosen temperature.
    pub nll: f64,
    /// The optimiser's `log T` before rounding to f32.
    pub log_t: f64,
    /// Function evaluations and status of `fminbound` (0 converged).
    pub nfev: u64,
    pub status: u64,
}

/// The odd half: the gate grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OddEvidence {
    pub n: u64,
    /// Rows accepted by the final gate and how many are correct.
    pub accepted: u64,
    pub correct: u64,
    pub grid: Vec<GridEntry>,
    pub grid_theta_off: Vec<GridEntry>,
}

/// All calibration rows: winner equals the label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationEvidence {
    pub n: u64,
    pub correct: u64,
}

/// One threshold of the grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridEntry {
    pub t: f64,
    pub accepted: u64,
    pub correct: u64,
    pub lb: f64,
    pub novelty_rejected: u64,
}

/// The f32 gate the runtime applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateParams {
    pub temperature: f32,
    pub novelty_theta: f32,
    pub tau: f32,
    pub certified: bool,
}

impl GridEntry {
    fn of(r: &GridRow) -> Self {
        Self {
            t: r.threshold,
            accepted: r.accepted as u64,
            correct: r.correct as u64,
            lb: r.lower_bound,
            novelty_rejected: r.novelty_rejected as u64,
        }
    }
}

impl GateRule {
    /// The fixed certification rule.
    pub fn standard() -> Self {
        Self {
            thresholds: certify::THRESHOLDS.to_vec(),
            alpha: ALPHA_RULE.into(),
            min_accepted: certify::MIN_ACCEPTED as u64,
            target: certify::TARGET,
        }
    }
}

impl Gate {
    /// The manifest record of a certification.
    pub fn from_certification(c: &Certification) -> Self {
        Self {
            temperature: c.temperature as f64,
            novelty_theta: c.novelty_theta as f64,
            tau: c.tau as f64,
            certified: c.certified,
            rule: GateRule::standard(),
            evidence: Evidence {
                even: EvenEvidence {
                    n: c.even_n as u64,
                    n_t: c.even_t_n as u64,
                    nll: c.nll_even,
                    log_t: c.temperature_fit.x,
                    nfev: c.temperature_fit.nfev as u64,
                    status: c.temperature_fit.status as u64,
                },
                odd: OddEvidence {
                    n: c.odd_n as u64,
                    accepted: c.odd_accepted as u64,
                    correct: c.odd_correct as u64,
                    grid: c.grid.iter().map(GridEntry::of).collect(),
                    grid_theta_off: c.grid_theta_off.iter().map(GridEntry::of).collect(),
                },
                calibration: CalibrationEvidence {
                    n: c.calibration_n as u64,
                    correct: c.calibration_correct as u64,
                },
            },
        }
    }

    /// The gate of a skill that was never certified: `T = 1`, `θ = 1`, `τ = 0`,
    /// no evidence (the value `build`/`learn` score with before a certification,
    /// as a manifest record). It passes [`Gate::validate`]; an auto-skill whose
    /// tasks are all quarantined carries it.
    pub fn placeholder() -> Self {
        let zero_grid = || {
            certify::THRESHOLDS
                .iter()
                .map(|&t| GridEntry {
                    t,
                    accepted: 0,
                    correct: 0,
                    lb: 0.0,
                    novelty_rejected: 0,
                })
                .collect()
        };
        Self {
            temperature: 1.0,
            novelty_theta: 1.0,
            tau: 0.0,
            certified: false,
            rule: GateRule::standard(),
            evidence: Evidence {
                even: EvenEvidence {
                    n: 0,
                    n_t: 0,
                    nll: 0.0,
                    log_t: 0.0,
                    nfev: 0,
                    status: 0,
                },
                odd: OddEvidence {
                    n: 0,
                    accepted: 0,
                    correct: 0,
                    grid: zero_grid(),
                    grid_theta_off: zero_grid(),
                },
                calibration: CalibrationEvidence { n: 0, correct: 0 },
            },
        }
    }

    /// The f32 values the runtime uses.
    pub fn params(&self) -> GateParams {
        GateParams {
            temperature: self.temperature as f32,
            novelty_theta: self.novelty_theta as f32,
            tau: self.tau as f32,
            certified: self.certified,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            is_f32_exact(self.temperature) && self.temperature > 0.0,
            "gate temperature must be a positive f32 value"
        );
        ensure!(
            is_f32_exact(self.novelty_theta) && (0.0..=1.0).contains(&self.novelty_theta),
            "gate novelty_theta must be an f32 value in [0, 1]"
        );
        ensure!(
            is_f32_exact(self.tau) && (0.0..=1.0).contains(&self.tau),
            "gate tau must be an f32 value in [0, 1]"
        );
        ensure!(
            self.rule == GateRule::standard(),
            "unsupported gate rule (implemented: the 14-threshold grid, alpha 0.05/14, min_accepted 100, target 0.95)"
        );
        if self.certified {
            ensure!(
                self.rule
                    .thresholds
                    .iter()
                    .any(|&t| (t as f32) as f64 == self.tau),
                "a certified tau must be a grid threshold"
            );
        } else {
            ensure!(self.tau == 0.0, "an uncertified gate has tau 0");
        }
        let e = &self.evidence;
        ensure!(
            e.even.nll.is_finite() && e.even.log_t.is_finite(),
            "even-half evidence must be finite"
        );
        ensure!(e.even.n_t <= e.even.n, "even n_t exceeds n");
        ensure!(
            e.odd.accepted <= e.odd.n && e.odd.correct <= e.odd.accepted,
            "odd-half counts are inconsistent"
        );
        ensure!(
            e.calibration.correct <= e.calibration.n && e.even.n + e.odd.n == e.calibration.n,
            "calibration counts are inconsistent"
        );
        for (what, grid) in [
            ("grid", &e.odd.grid),
            ("grid_theta_off", &e.odd.grid_theta_off),
        ] {
            ensure!(
                grid.len() == self.rule.thresholds.len(),
                "{what} must have one row per threshold"
            );
            for (g, &t) in grid.iter().zip(&self.rule.thresholds) {
                ensure!(g.t == t, "{what} thresholds differ from the rule");
                ensure!(
                    g.correct <= g.accepted && g.accepted <= e.odd.n,
                    "{what} counts are inconsistent"
                );
                ensure!(
                    g.lb.is_finite() && (0.0..=1.0).contains(&g.lb),
                    "{what} lower bound outside [0, 1]"
                );
            }
        }
        Ok(())
    }
}

/// The question a skill answers (from `--question`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rubric {
    pub instructions: String,
    pub criteria: Map<String, Value>,
    /// Key order of the question file when it is not the sorted order (canonical
    /// JSON sorts keys; the oracle's schema enum follows this order).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub criteria_order: Vec<String>,
}

impl Rubric {
    /// A rubric from criteria in their file order.
    pub fn new(instructions: impl Into<String>, criteria: Map<String, Value>) -> Self {
        let keys: Vec<String> = criteria.keys().cloned().collect();
        let sorted = keys.windows(2).all(|w| w[0] < w[1]);
        Self {
            instructions: instructions.into(),
            criteria,
            criteria_order: if sorted { Vec::new() } else { keys },
        }
    }

    /// Criteria keys in the question file's order.
    pub fn order(&self) -> Vec<&str> {
        if self.criteria_order.is_empty() {
            let mut k: Vec<&str> = self.criteria.keys().map(String::as_str).collect();
            k.sort_unstable();
            k
        } else {
            self.criteria_order.iter().map(String::as_str).collect()
        }
    }

    /// Criteria in the question file's order, as an insertion-ordered map.
    pub fn ordered_criteria(&self) -> Map<String, Value> {
        self.order()
            .into_iter()
            .map(|k| (k.to_string(), self.criteria[k].clone()))
            .collect()
    }
}

/// Input data of a skill.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataRecord {
    pub train: TrainRecord,
    pub calibration: CalibrationRecord,
    /// `null` without `--dev`.
    pub dev: Option<DevRecord>,
    pub halves_rule: String,
    pub holdout_rule: String,
}

/// The training rows: `n` rows fitted (after a carve-out, the rows left for
/// training), `sha256` of the input as read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainRecord {
    pub n: u64,
    pub sha256: String,
    /// The files of a union (e.g. train ∪ dev, spec §3.9), in the order read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<InputPart>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputPart {
    pub name: String,
    pub n: u64,
    pub sha256: String,
}

/// Calibration rows: `source` is `file` (`--calibration`) or `carve-out` (router
/// holdout rule over the training file, whose sha256 is then recorded).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRecord {
    pub n: u64,
    pub sha256: String,
    pub source: String,
}

/// The dev file and the rows the built skill decides correctly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevRecord {
    pub n: u64,
    pub correct: u64,
    pub sha256: String,
}

/// The rows blob of the build (spec §2.5).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowsRecord {
    pub tensor: String,
    pub layout: String,
    pub n_train: u64,
    pub n_calibration: u64,
    /// Learned rows (split 2) stored in the same blob (spec §5.14).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub n_learned: u64,
    pub sha256: String,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// Learned rows in `decision.skill.{id}.rows.learned` (all split 2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedRowsRecord {
    pub tensor: String,
    pub layout: String,
    pub n: u64,
    pub sha256: String,
}

/// Offline learning through the oracle (spec §5.14).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedRecord {
    pub traffic_sha256: String,
    pub oracle_model: String,
    pub calls: u64,
    pub answers_reused: u64,
    pub promoted_labels: Vec<String>,
    pub rejected_labels: Vec<String>,
    pub gate_before: Gate,
    pub gate_after: Gate,
}

impl SkillManifest {
    /// Active tasks (scored).
    pub fn active_tasks(&self) -> impl Iterator<Item = &TaskRecord> {
        self.tasks.iter().filter(|t| t.is_active())
    }

    /// Index of a label.
    pub fn task_of(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }

    /// An auto-skill: a choice contract learned from oracle answers (0.8.6).
    /// Derived, never stored: the reserved id prefix and no build rows
    /// (`data.train.n == 0`; every row lives in `rows.learned`). [`Self::validate`]
    /// refuses the prefix on a skill with build rows, so the two agree.
    pub fn is_auto(&self) -> bool {
        is_auto_skill_id(&self.id) && self.data.train.n == 0
    }

    /// The manifest of a new auto-skill before any fit: every contract id a
    /// quarantined cold-start task (no topology, `k 0`) in the contract's
    /// sorted order, the placeholder gate, an empty data record whose sha256 is
    /// the contract's ([`contract_sha256`]) and the `learned` calibration source.
    /// `representation_id` and the rows records are filled by the writer
    /// ([`crate::container::OverlayBuilder::add_skill`]). The learner then
    /// replaces the tasks it fits and the gate it certifies.
    pub fn auto_skeleton(ids: &[&str], rubric: Option<Rubric>, k_max: u64) -> Self {
        let labels = contract_ids(ids);
        let tasks = labels
            .iter()
            .enumerate()
            .map(|(i, label)| TaskRecord {
                i: i as u64,
                label: label.clone(),
                state: TaskState::Quarantined,
                origin: TaskOrigin::ColdStart,
                k: 0,
                n_train: 0,
                err_mean: 0.0,
                err_std: 0.0,
                mean_sha256: None,
                basis_sha256: None,
            })
            .collect();
        let sha = contract_sha256(ids);
        Self {
            schema: SKILL_SCHEMA.into(),
            id: auto_skill_id(ids),
            taxonomy_version: 1,
            representation_id: String::new(),
            recipe: Recipe::standard(k_max),
            labels,
            tasks,
            gate: Gate::placeholder(),
            rubric,
            data: DataRecord {
                train: TrainRecord {
                    n: 0,
                    sha256: sha.clone(),
                    parts: Vec::new(),
                },
                calibration: CalibrationRecord {
                    n: 0,
                    sha256: sha,
                    source: CALIBRATION_LEARNED.into(),
                },
                dev: None,
                halves_rule: certify::HALVES_RULE_AUTO.into(),
                holdout_rule: HOLDOUT_RULE.into(),
            },
            rows: RowsRecord::default(),
            rows_learned: None,
            learned: None,
        }
    }

    /// Check the manifest on its own: schema, identities, recipe, the task table,
    /// gate, rubric, data and rows records. `signal_dim` bounds `K`.
    ///
    /// An auto-skill ([`Self::is_auto`]) relaxes exactly the data checks a skill
    /// without build rows cannot meet — calibration source `learned`,
    /// `halves_rule` [`certify::HALVES_RULE_AUTO`] — and adds its own: every task
    /// `cold_start`, labels in the contract's sorted order, no calibration rows,
    /// no dev file, no learned rows in the (empty) build blob.
    pub fn validate(&self, representation_id: &str, signal_dim: usize) -> Result<()> {
        ensure!(
            self.schema == SKILL_SCHEMA,
            "unsupported skill schema '{}' (expected {SKILL_SCHEMA})",
            self.schema
        );
        ensure!(valid_skill_id(&self.id), "invalid skill id '{}'", self.id);
        // The prefix is reserved: with build rows it is a user skill in disguise
        // and the derived `is_auto()` would disagree with the id. This is a
        // deliberate load-time break for a pre-0.8.6 file whose user skill was
        // built under the prefix (0.8.5 accepted any valid id): the id is the
        // discriminator the learner and the server rely on (contract keys,
        // isolation, who teaches), so such a file is rebuilt under another id
        // rather than read leniently (CHANGELOG, Compatibility).
        ensure!(
            !is_auto_skill_id(&self.id) || self.data.train.n == 0,
            "the '{AUTO_SKILL_PREFIX}' id prefix is reserved for auto-skills (no build rows); \
             a user skill built under it is rebuilt under another id"
        );
        let auto = self.is_auto();
        ensure!(self.taxonomy_version >= 1, "taxonomy_version starts at 1");
        ensure!(
            self.representation_id == representation_id,
            "skill representation_id {} differs from the file's {representation_id}",
            self.representation_id
        );
        self.recipe.validate(signal_dim)?;
        ensure!(
            self.labels.len() == self.tasks.len(),
            "{} labels for {} tasks",
            self.labels.len(),
            self.tasks.len()
        );
        ensure!(self.tasks.len() <= 1 << 16, "too many tasks");
        let mut seen = BTreeSet::new();
        let mut last_data: Option<&str> = None;
        let mut cold = false;
        for (i, (t, l)) in self.tasks.iter().zip(&self.labels).enumerate() {
            let at = |m: &str| anyhow::anyhow!("task {i} ('{}'): {m}", t.label);
            if t.i != i as u64 {
                return Err(at("index differs from its position"));
            }
            if t.label != *l {
                return Err(at("label differs from labels[i]"));
            }
            if !valid_label(l) {
                return Err(at("labels have 1..=256 bytes"));
            }
            if !seen.insert(l.as_str()) {
                return Err(at("duplicate label"));
            }
            match t.origin {
                TaskOrigin::Data => {
                    if auto {
                        return Err(at("an auto-skill's tasks are all cold_start"));
                    }
                    if cold {
                        return Err(at("data tasks precede cold-start tasks"));
                    }
                    if last_data.is_some_and(|p| p.as_bytes() >= l.as_bytes()) {
                        return Err(at("data labels are sorted bytewise"));
                    }
                    last_data = Some(l);
                }
                TaskOrigin::ColdStart => {
                    cold = true;
                    // The contract's sorted order (so the data-task ordering
                    // rules hold for an auto-skill, D4); `last_data` doubles as
                    // the previous label here since no data task precedes.
                    if auto {
                        if last_data.is_some_and(|p| p.as_bytes() >= l.as_bytes()) {
                            return Err(at("an auto-skill's labels are sorted bytewise"));
                        }
                        last_data = Some(l);
                    }
                }
            }
            if t.k > self.recipe.k_max {
                return Err(at("k exceeds the recipe K"));
            }
            if t.k > t.n_train.saturating_sub(1) {
                return Err(at("k exceeds n_train - 1"));
            }
            if let Some(s) = &t.mean_sha256 {
                check_sha("mean_sha256", s).map_err(|e| at(&e.to_string()))?;
            }
            match (&t.basis_sha256, t.k) {
                (None, 0) => {}
                (Some(s), k) if k > 0 => {
                    check_sha("basis_sha256", s).map_err(|e| at(&e.to_string()))?;
                }
                _ => return Err(at("basis_sha256 is null exactly when k = 0")),
            }
            if t.mean_sha256.is_none() && (t.k != 0 || t.state == TaskState::Active) {
                return Err(at("a task without a mean has k = 0 and is not active"));
            }
            if !(t.err_mean.is_finite()
                && t.err_mean >= 0.0
                && t.err_std.is_finite()
                && t.err_std >= 0.0)
            {
                return Err(at("err_mean/err_std must be finite and non-negative"));
            }
            if t.state == TaskState::Active {
                if t.n_train < self.recipe.min_rows_active {
                    return Err(at("an active task needs min_rows_active rows"));
                }
                t.stats().check().map_err(|e| at(&e.to_string()))?;
            }
        }
        self.gate.validate()?;
        if let Some(r) = &self.rubric {
            ensure!(
                r.instructions.len() <= 1 << 20,
                "rubric instructions too long"
            );
            for k in r.criteria.keys() {
                ensure!(
                    seen.contains(k.as_str()),
                    "rubric criterion '{k}' is not a label of the skill"
                );
            }
            if !r.criteria_order.is_empty() {
                let order: BTreeSet<&str> = r.criteria_order.iter().map(String::as_str).collect();
                ensure!(
                    order.len() == r.criteria_order.len()
                        && order.iter().copied().eq(r
                            .criteria
                            .keys()
                            .map(String::as_str)
                            .collect::<BTreeSet<_>>()),
                    "rubric criteria_order is not a permutation of the criteria keys"
                );
            }
        }
        let d = &self.data;
        check_sha("data.train.sha256", &d.train.sha256)?;
        if !d.train.parts.is_empty() {
            let mut n = 0u64;
            for p in &d.train.parts {
                check_text("data.train part name", &p.name, 256)?;
                check_sha("data.train part sha256", &p.sha256)?;
                n += p.n;
            }
            ensure!(n == d.train.n, "data.train parts do not add up to n");
        }
        check_sha("data.calibration.sha256", &d.calibration.sha256)?;
        if auto {
            ensure!(
                d.calibration.source == CALIBRATION_LEARNED,
                "an auto-skill's calibration source is {CALIBRATION_LEARNED}"
            );
            ensure!(
                d.calibration.n == 0 && d.train.parts.is_empty() && d.dev.is_none(),
                "an auto-skill has no calibration rows, input parts or dev file"
            );
            ensure!(
                d.halves_rule == certify::HALVES_RULE_AUTO,
                "an auto-skill's halves_rule is the learned-rows rule"
            );
            ensure!(
                self.rows.n_learned == 0,
                "an auto-skill keeps every row in rows.learned (its build blob is empty)"
            );
        } else {
            ensure!(
                d.calibration.source == CALIBRATION_FROM_FILE
                    || d.calibration.source == CALIBRATION_CARVE_OUT,
                "calibration source must be file or carve-out"
            );
            ensure!(
                d.halves_rule == certify::HALVES_RULE,
                "unsupported halves_rule"
            );
        }
        if let Some(dev) = &d.dev {
            check_sha("data.dev.sha256", &dev.sha256)?;
            ensure!(dev.correct <= dev.n, "dev correct exceeds n");
        }
        ensure!(d.holdout_rule == HOLDOUT_RULE, "unsupported holdout_rule");
        ensure!(
            self.rows.tensor == rows_tensor(&self.id),
            "rows tensor must be {}",
            rows_tensor(&self.id)
        );
        ensure!(self.rows.layout == rows::LAYOUT, "unsupported rows layout");
        check_sha("rows.sha256", &self.rows.sha256)?;
        ensure!(
            self.rows.n_train == d.train.n && self.rows.n_calibration == d.calibration.n,
            "rows counts differ from the data record"
        );
        if let Some(r) = &self.rows_learned {
            ensure!(
                r.tensor == rows_learned_tensor(&self.id),
                "learned rows tensor must be {}",
                rows_learned_tensor(&self.id)
            );
            ensure!(r.layout == rows::LAYOUT, "unsupported learned rows layout");
            check_sha("rows_learned.sha256", &r.sha256)?;
        }
        if let Some(l) = &self.learned {
            check_sha("learned.traffic_sha256", &l.traffic_sha256)?;
            check_text("learned.oracle_model", &l.oracle_model, 256)?;
            for lab in l.promoted_labels.iter().chain(&l.rejected_labels) {
                ensure!(
                    seen.contains(lab.as_str()),
                    "learned label '{lab}' is not a label of the skill"
                );
            }
            l.gate_before.validate()?;
            l.gate_after.validate()?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ overlay manifest

/// `decision.overlay.manifest` of a generation (spec §5.10): full manifests of
/// the skills that differ from the base file, cumulative since generation 0.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayManifest {
    pub schema: String,
    pub generation: u64,
    /// `model_sha` of the base file (sha256 of its `decision.manifest`).
    pub base_model_sha: String,
    /// The generation this one was derived from (0 = the base file).
    pub parent: u64,
    pub created_unix: u64,
    /// Every event since the base, oldest first.
    pub events: Vec<OverlayEvent>,
    /// Skill id → full skill manifest (as stored; its sha256 is that of its
    /// canonical bytes).
    pub skills: Map<String, Value>,
}

/// One learning event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayEvent {
    pub generation: u64,
    pub skill: String,
    pub label: String,
    /// e.g. `promote`, `cold_start`, `reject`.
    pub kind: String,
    pub holdout: Value,
    pub gate: Value,
}

impl OverlayManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == OVERLAY_SCHEMA,
            "unsupported overlay schema '{}' (expected {OVERLAY_SCHEMA})",
            self.schema
        );
        ensure!(self.generation >= 1, "overlay generations start at 1");
        ensure!(
            self.parent < self.generation,
            "overlay parent {} is not older than generation {}",
            self.parent,
            self.generation
        );
        check_sha("base_model_sha", &self.base_model_sha)?;
        for id in self.skills.keys() {
            ensure!(valid_skill_id(id), "invalid skill id '{id}'");
        }
        for e in &self.events {
            ensure!(
                e.generation >= 1 && e.generation <= self.generation,
                "event generation {} outside 1..={}",
                e.generation,
                self.generation
            );
            ensure!(
                valid_skill_id(&e.skill),
                "invalid event skill id '{}'",
                e.skill
            );
            check_text("event kind", &e.kind, 64)?;
            ensure!(e.label.len() <= 256, "event label too long");
        }
        Ok(())
    }
}

/// Parse a typed manifest from a JSON value with a readable error.
pub fn from_value<T: for<'de> Deserialize<'de>>(what: &str, v: &Value) -> Result<T> {
    T::deserialize(v).map_err(|e| anyhow::anyhow!("{what}: {e}"))
}

/// Parse a manifest tensor: size bound, canonical JSON, typed parse.
pub fn parse_manifest_bytes<T: for<'de> Deserialize<'de>>(
    what: &str,
    bytes: &[u8],
) -> Result<(T, Value)> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        bail!(
            "{what}: {} bytes exceed the 16 MiB manifest limit",
            bytes.len()
        );
    }
    let v = canonical::parse_canonical(bytes).map_err(|e| anyhow::anyhow!("{what}: {e}"))?;
    let t = from_value(what, &v)?;
    Ok((t, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RID: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    /// The contract key hashes the canonical array of the sorted ids: the
    /// request order does not matter and ids with any bytes stay apart (a
    /// joined string would merge `{"a\nb","c"}` and `{"a","b\nc"}`).
    #[test]
    fn auto_skill_id_is_order_free_and_injective() {
        let a = auto_skill_id(&["travel", "food", "cruise"]);
        let b = auto_skill_id(&["cruise", "travel", "food", "food"]);
        assert_eq!(a, b);
        assert!(valid_skill_id(&a) && is_auto_skill_id(&a));
        assert_eq!(a.len(), AUTO_SKILL_PREFIX.len() + AUTO_SKILL_ID_HEX);
        assert_ne!(a, auto_skill_id(&["travel", "food", "cruise", "x"]));
        assert_ne!(auto_skill_id(&["a\nb", "c"]), auto_skill_id(&["a", "b\nc"]));
        assert_eq!(contract_ids(&["b", "a", "b"]), ["a", "b"]);
    }

    fn skeleton() -> SkillManifest {
        let mut criteria = Map::new();
        criteria.insert("food".into(), Value::Null);
        criteria.insert("cruise".into(), Value::String("a cruise".into()));
        let mut m = SkillManifest::auto_skeleton(
            &["food", "cruise"],
            Some(Rubric::new("Pick.", criteria)),
            8,
        );
        m.representation_id = RID.into();
        m.rows = RowsRecord {
            tensor: rows_tensor(&m.id),
            layout: rows::LAYOUT.into(),
            n_train: 0,
            n_calibration: 0,
            n_learned: 0,
            sha256: sha256_hex(b"empty"),
        };
        m
    }

    /// `is_auto` is derived; the skeleton validates; the relaxations are
    /// confined to auto-skills and the auto invariants are enforced.
    #[test]
    fn auto_skeleton_validates_and_the_relaxations_are_scoped() {
        let m = skeleton();
        assert!(m.is_auto());
        assert_eq!(m.labels, ["cruise", "food"]);
        assert!(m.tasks.iter().all(|t| {
            t.state == TaskState::Quarantined && t.origin == TaskOrigin::ColdStart && t.k == 0
        }));
        assert_eq!(m.data.calibration.source, CALIBRATION_LEARNED);
        assert_eq!(m.data.train.sha256, contract_sha256(&["cruise", "food"]));
        m.validate(RID, 4104).unwrap();
        assert!(Gate::placeholder().validate().is_ok());

        let refused = |f: &dyn Fn(&mut SkillManifest), what: &str| {
            let mut m = skeleton();
            f(&mut m);
            let e = m.validate(RID, 4104).unwrap_err().to_string();
            assert!(e.contains(what), "{e}");
        };
        // The prefix with build rows is a user skill in disguise.
        refused(&|m| m.data.train.n = 1, "reserved");
        // A user skill does not get the relaxations.
        refused(
            &|m| {
                m.id = "user".into();
                m.rows.tensor = rows_tensor("user");
            },
            "file or carve-out",
        );
        refused(&|m| m.tasks[0].origin = TaskOrigin::Data, "all cold_start");
        refused(
            &|m| {
                m.labels.swap(0, 1);
                m.tasks.swap(0, 1);
                m.tasks[0].i = 0;
                m.tasks[1].i = 1;
            },
            "sorted bytewise",
        );
        refused(&|m| m.data.calibration.n = 1, "no calibration rows");
        refused(
            &|m| m.data.calibration.source = CALIBRATION_CARVE_OUT.into(),
            "calibration source is learned",
        );
        refused(
            &|m| m.data.halves_rule = certify::HALVES_RULE.into(),
            "learned-rows rule",
        );
        refused(&|m| m.rows.n_learned = 1, "rows.learned");
    }
}
