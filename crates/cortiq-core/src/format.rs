//! CMF v2 binary container — envelope, tensor directory, mmap access.
//!
//! See `docs/CMF_V2_SPEC.md`. Layout summary:
//!
//! ```text
//! [0x00]  magic "CMF\x01" | version u32 = 2 | flags u32 | required_features u32
//! [0x10]  header_off/len | dir_off/len | data_off/len   (u64 LE each)
//! [0x40]  masks_off/len  | vocab_off/len | index_off/len
//! [0x70]  16 reserved bytes (zero)
//! [0x80]  header JSON → tensor directory → weight blob (4096-aligned,
//!         tensors 64-aligned) → masks → vocab → sparse index
//! ```
//!
//! The tensor directory is the ONLY source of truth for the weight blob
//! layout — there is no computable layout, by design (v1 bug class #1).
//! Every validation failure is a hard error: no silent fallbacks.

use crate::hash::hash64;
use crate::mask::{decode_masks_section, encode_masks_section, MaskCatalog, TaskMask};
use crate::quant::expected_nbytes;
use crate::types::{ModelArch, QuantType, TensorDtype};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

pub const CMF_MAGIC: [u8; 4] = *b"CMF\x01";
pub const CMF_VERSION: u32 = 2;
pub const ENVELOPE_LEN: usize = 128;
/// Weight blob is page-aligned for mmap.
pub const DATA_ALIGNMENT: u64 = 4096;
/// Every tensor inside the blob is 64-byte aligned (SIMD / cache line).
pub const TENSOR_ALIGNMENT: u64 = 64;
/// One directory record is 56 bytes (see `.vmfc` v2).
pub const DIR_RECORD_LEN: usize = 56;
pub const DIR_MAX_NDIM: usize = 6;

/// `required_features` bits. A reader MUST refuse a file with any bit
/// it does not support.
pub mod features {
    pub const TENSOR_DIR: u32 = 1 << 0;
    pub const BINARY_MASKS: u32 = 1 << 1;
    pub const QUANT_2F: u32 = 1 << 2;
    pub const DELTA_MASKS: u32 = 1 << 3;
    pub const HOT_PACKS: u32 = 1 << 4;

    /// Features this reader implements today.
    pub const SUPPORTED: u32 = TENSOR_DIR | BINARY_MASKS | QUANT_2F;
}

/// JSON header — architecture and provenance (human-readable part;
/// machine-critical data lives in binary sections).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmfHeader {
    #[serde(default = "default_format")]
    pub format: String,
    pub version: u32,
    pub arch: ModelArch,
    /// Informational default; per-tensor truth is in the directory.
    pub quant_type: QuantType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
    /// Chat/eos bundle (spec §6.1): the file — not the binary — defines
    /// chat behavior. Additive: absent in older files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_config: Option<TokenizerBundle>,
    /// Section-level integrity (spec §8.1): hex hash64 of the raw bytes
    /// of the optional sections. header/dir hashes live in the envelope
    /// reserved bytes — JSON cannot protect the JSON that carries it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section_hashes: Option<SectionHashes>,
    /// Per-skill records (spec §9): replacement tensors live in the
    /// directory as `skill.{id}.{name}`; this registry carries the
    /// selection descriptor and the honest quality contract.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<SkillRecord>,
    /// Sharding (spec §10): this file is shard `no` of `count`; every
    /// shard is a standalone valid .cmf carrying a tensor subset.
    /// Naming convention: `…-{no:05}-of-{count:05}.cmf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<ShardInfo>,
    /// Measured confidence calibration (B1): a temperature fit on held-out
    /// so the displayed Born-mass confidence is a true property of the
    /// model (softmax(logits/T)), not a raw estimate. Additive; absent =
    /// use raw (T=1). Written by `set_calibration.py` after `cortiq
    /// calibrate` measures the reliability/ECE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
}

/// Confidence-calibration record (spec §6.2). `temperature` scales the
/// logits before softmax when reporting confidence; `ece_before`/`after`
/// are the measured Expected Calibration Error (honest provenance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ece_before: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ece_after: Option<f32>,
}

/// Shard coordinates (1-based, gguf-split style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub no: usize,
    pub count: usize,
}

/// Recon-argmin routing parameters (spec §9; P1 signal-consistency):
/// E = ‖(φ−mean) − B·Bᵀ(φ−mean)‖² / ‖φ−mean‖²; pick argmin over skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionDescriptor {
    /// "mse" (normalized reconstruction error) — the only metric today.
    pub metric: String,
    /// Backbone layer whose mean-pooled hidden is φ(x).
    pub phi_layer: usize,
    /// Subspace mean, f16 LE base64, len = hidden.
    pub mean: String,
    /// Orthonormal basis rows, f16 LE base64, len = rank·hidden.
    pub basis: String,
    pub rank: usize,
}

/// One skill of the swarm (spec §9; Patent 15 per-skill record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Layers this skill specializes (a proper subset).
    #[serde(default)]
    pub layers: Vec<usize>,
    /// Selection descriptor for recon-argmin routing (208c, P1):
    /// per-skill affine subspace over φ(x) = mean-pooled hidden state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<SelectionDescriptor>,
    /// Optional input-mask task name (208b), applied with the skill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_mask_task: Option<String>,
    /// Measured quality (claim 16): overlaid vs backbone, held-out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<serde_json::Value>,
}

/// Hex-encoded hash64 per optional section (u64 as JSON number would
/// lose precision past 2^53).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SectionHashes {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub masks: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<String>,
}

/// Chat template + generation stop tokens carried by the container.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenizerBundle {
    /// Jinja chat template (chat_template.jinja / tokenizer_config.json)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    /// All ids that terminate generation (generation_config + im_end)
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_token_id: Option<u32>,
}

fn default_format() -> String {
    "cmf".to_string()
}

/// One tensor directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    /// Offset relative to the OWNING shard's `data_off`, multiple of 64.
    pub off: u64,
    pub nbytes: u64,
    /// Runtime-only: which shard's mmap holds the bytes (0 for the
    /// single-file case; not part of the 56-byte record).
    pub shard: usize,
    /// `hash64` of the tensor bytes.
    pub hash: u64,
}

impl TensorEntry {
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Input for the Rust writer: one tensor with its encoded bytes.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// Sparse index entry — precomputed per-task per-layer active group IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseIndexEntry {
    pub task_id: u32,
    pub layer_idx: usize,
    /// Active quant-group indices for FFN (sorted, group = 32 neurons).
    pub active_ffn_groups: Vec<u16>,
    /// Active head indices for attention (sorted).
    pub active_heads: Vec<u8>,
}

/// Section ranges parsed from the fixed envelope.
#[derive(Debug, Clone, Copy, Default)]
struct Envelope {
    required_features: u32,
    header: (u64, u64),
    dir: (u64, u64),
    data: (u64, u64),
    masks: (u64, u64),
    vocab: (u64, u64),
    index: (u64, u64),
    /// hash64 of the header JSON bytes (reserved [0x70]); 0 = absent.
    header_hash: u64,
    /// hash64 of the tensor-directory bytes (reserved [0x78]); 0 = absent.
    dir_hash: u64,
}

enum Backing {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// A loaded CMF model: metadata owned, weights zero-copy via mmap.
pub struct CmfModel {
    pub path: PathBuf,
    pub header: CmfHeader,
    pub required_features: u32,
    pub tensors: Vec<TensorEntry>,
    by_name: HashMap<String, usize>,
    pub masks: MaskCatalog,
    pub sparse_index: Vec<SparseIndexEntry>,
    /// Embedded tokenizer.json bytes, if present.
    pub vocab: Option<Vec<u8>>,
    backing: Backing,
    data_off: u64,
    envelope: Envelope,
    /// Shards 2..N (spec §10): (backing, data_off) per extra file;
    /// `TensorEntry.shard` 0 = this file, i>0 = extra_shards[i-1].
    extra_shards: Vec<(Backing, u64)>,
}

impl std::fmt::Debug for CmfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmfModel")
            .field("path", &self.path)
            .field("arch", &self.header.arch.arch_name)
            .field("tensors", &self.tensors.len())
            .field("masks", &self.masks.masks.len())
            .finish()
    }
}

impl CmfModel {
    /// Open and strictly validate a CMF v2 file. Any inconsistency is an
    /// error — this function never substitutes defaults.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CmfError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(CmfError::FileNotFound(path.display().to_string()));
        }
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        let backing = match unsafe { memmap2::MmapOptions::new().map(&file) } {
            Ok(m) => Backing::Mmap(m),
            Err(e) => {
                tracing::warn!("mmap failed ({e}), reading file into memory");
                Backing::Owned(std::fs::read(&path)?)
            }
        };

        let env = Self::parse_envelope(backing.bytes(), file_len)?;

        let bytes = backing.bytes();
        let section = |off: u64, len: u64| -> &[u8] {
            &bytes[off as usize..(off + len) as usize]
        };

        // Header JSON
        let header: CmfHeader = serde_json::from_slice(section(env.header.0, env.header.1))
            .map_err(|e| CmfError::Parse(format!("header JSON: {e}")))?;

        // Tensor directory
        let tensors = Self::decode_directory(section(env.dir.0, env.dir.1))?;
        for t in &tensors {
            if t.off % TENSOR_ALIGNMENT != 0 {
                return Err(CmfError::Bounds(format!(
                    "tensor '{}': offset {} not 64-aligned",
                    t.name, t.off
                )));
            }
            if t.off + t.nbytes > env.data.1 {
                return Err(CmfError::Bounds(format!(
                    "tensor '{}': [{}, {}) exceeds data section ({} bytes)",
                    t.name,
                    t.off,
                    t.off + t.nbytes,
                    env.data.1
                )));
            }
            if let Some(expect) = expected_nbytes(t.dtype, &t.shape) {
                if expect as u64 != t.nbytes {
                    return Err(CmfError::Bounds(format!(
                        "tensor '{}': nbytes {} != expected {} for {:?}{:?}",
                        t.name, t.nbytes, expect, t.dtype, t.shape
                    )));
                }
            }
        }
        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        // Masks
        let masks = if env.masks.1 > 0 {
            decode_masks_section(section(env.masks.0, env.masks.1), &header.arch)
                .map_err(CmfError::Parse)?
        } else {
            MaskCatalog::empty()
        };

        // Vocab (tokenizer.json)
        let vocab = if env.vocab.1 > 0 {
            Some(section(env.vocab.0, env.vocab.1).to_vec())
        } else {
            None
        };

        // Sparse index
        let sparse_index = if env.index.1 > 0 {
            decode_sparse_index(section(env.index.0, env.index.1))?
        } else {
            vec![]
        };

        tracing::info!(
            "Opened CMF v2: {} | {} tensors | {} masks | vocab {} | {:.1} MB",
            header.arch.arch_name,
            tensors.len(),
            masks.masks.len(),
            if vocab.is_some() { "embedded" } else { "none" },
            file_len as f64 / 1e6
        );

        Ok(Self {
            path,
            header,
            required_features: env.required_features,
            tensors,
            by_name,
            masks,
            sparse_index,
            vocab,
            backing,
            data_off: env.data.0,
            envelope: env,
            extra_shards: Vec::new(),
        })
    }

    /// Open a sharded model (spec §10): pass shard 1; siblings found by
    /// the `-{no:05}-of-{count:05}.cmf` convention. Directories merge;
    /// masks/vocab/index/skills come from shard 1.
    pub fn open_sharded(path: impl AsRef<Path>) -> Result<Self, CmfError> {
        let path = path.as_ref();
        let mut first = Self::open(path)?;
        let Some(info) = first.header.shard.clone() else {
            return Ok(first); // not sharded — plain open
        };
        if info.no != 1 {
            return Err(CmfError::Parse(format!(
                "open shard 1, not {} (of {})",
                info.no, info.count
            )));
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| CmfError::Parse("bad shard path".into()))?;
        let tag1 = format!("-{:05}-of-{:05}.cmf", 1, info.count);
        if !name.ends_with(&tag1) {
            return Err(CmfError::Parse(format!(
                "shard file must end with '{tag1}' (got '{name}')"
            )));
        }
        let stem = &name[..name.len() - tag1.len()];
        for no in 2..=info.count {
            let sib = path.with_file_name(format!(
                "{stem}-{:05}-of-{:05}.cmf",
                no, info.count
            ));
            let sh = Self::open(&sib)?;
            match &sh.header.shard {
                Some(si) if si.no == no && si.count == info.count => {}
                other => {
                    return Err(CmfError::Parse(format!(
                        "{}: wrong shard coords {other:?}",
                        sib.display()
                    )));
                }
            }
            let shard_idx = first.extra_shards.len() + 1;
            first.extra_shards.push((sh.backing, sh.envelope.data.0));
            for mut t in sh.tensors {
                t.shard = shard_idx;
                first.by_name.insert(t.name.clone(), first.tensors.len());
                first.tensors.push(t);
            }
        }
        tracing::info!(
            "sharded model: {} files, {} tensors total",
            info.count,
            first.tensors.len()
        );
        Ok(first)
    }

    fn parse_envelope(bytes: &[u8], file_len: u64) -> Result<Envelope, CmfError> {
        if bytes.len() < ENVELOPE_LEN {
            return Err(CmfError::Bounds(format!(
                "file too small for CMF envelope: {} bytes",
                bytes.len()
            )));
        }
        if bytes[0..4] != CMF_MAGIC {
            return Err(CmfError::InvalidMagic);
        }
        let u32le = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let u64le = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());

        let version = u32le(4);
        if version != CMF_VERSION {
            return Err(CmfError::UnsupportedVersion(version));
        }
        let _flags = u32le(8); // reserved
        let required_features = u32le(12);
        let unknown = required_features & !features::SUPPORTED;
        if unknown != 0 {
            return Err(CmfError::UnsupportedFeature(unknown));
        }

        let env = Envelope {
            required_features,
            header: (u64le(0x10), u64le(0x18)),
            dir: (u64le(0x20), u64le(0x28)),
            data: (u64le(0x30), u64le(0x38)),
            masks: (u64le(0x40), u64le(0x48)),
            vocab: (u64le(0x50), u64le(0x58)),
            index: (u64le(0x60), u64le(0x68)),
            header_hash: u64le(0x70),
            dir_hash: u64le(0x78),
        };

        for (name, (off, len), required) in [
            ("header", env.header, true),
            ("dir", env.dir, true),
            ("data", env.data, false),
            ("masks", env.masks, false),
            ("vocab", env.vocab, false),
            ("index", env.index, false),
        ] {
            if required && len == 0 {
                return Err(CmfError::Bounds(format!("section '{name}' is required")));
            }
            if len > 0 && off.checked_add(len).map(|end| end > file_len).unwrap_or(true) {
                return Err(CmfError::Bounds(format!(
                    "section '{name}' [{off}, {}) exceeds file ({file_len} bytes)",
                    off.saturating_add(len)
                )));
            }
        }
        if env.data.1 > 0 && env.data.0 % DATA_ALIGNMENT != 0 {
            return Err(CmfError::Bounds(format!(
                "data section offset {} not {}-aligned",
                env.data.0, DATA_ALIGNMENT
            )));
        }
        Ok(env)
    }

    fn decode_directory(bytes: &[u8]) -> Result<Vec<TensorEntry>, CmfError> {
        if bytes.len() < 16 {
            return Err(CmfError::Parse("tensor directory too short".into()));
        }
        let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let pool_off = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let records_end = 16 + count * DIR_RECORD_LEN;
        if records_end > bytes.len() || pool_off > bytes.len() || pool_off < records_end {
            return Err(CmfError::Parse(format!(
                "tensor directory malformed: count={count}, pool_off={pool_off}, len={}",
                bytes.len()
            )));
        }
        let pool = &bytes[pool_off..];

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let r = &bytes[16 + i * DIR_RECORD_LEN..16 + (i + 1) * DIR_RECORD_LEN];
            let name_off = u32::from_le_bytes(r[0..4].try_into().unwrap()) as usize;
            let name_len = u16::from_le_bytes(r[4..6].try_into().unwrap()) as usize;
            let dtype_id = r[6];
            let ndim = r[7] as usize;
            if ndim > DIR_MAX_NDIM {
                return Err(CmfError::Parse(format!("tensor #{i}: ndim {ndim} > 6")));
            }
            let mut shape = Vec::with_capacity(ndim);
            for d in 0..ndim {
                shape.push(u32::from_le_bytes(r[8 + d * 4..12 + d * 4].try_into().unwrap()) as usize);
            }
            let off = u64::from_le_bytes(r[32..40].try_into().unwrap());
            let nbytes = u64::from_le_bytes(r[40..48].try_into().unwrap());
            let hash = u64::from_le_bytes(r[48..56].try_into().unwrap());

            if name_off + name_len > pool.len() {
                return Err(CmfError::Parse(format!("tensor #{i}: name out of pool")));
            }
            let name = std::str::from_utf8(&pool[name_off..name_off + name_len])
                .map_err(|_| CmfError::Parse(format!("tensor #{i}: name is not UTF-8")))?
                .to_string();
            let dtype = TensorDtype::from_id(dtype_id).ok_or(CmfError::UnknownDtype(dtype_id))?;

            out.push(TensorEntry {
                name,
                dtype,
                shape,
                off,
                nbytes,
                shard: 0,
                hash,
            });
        }
        Ok(out)
    }

    // ───────────────────────── access ─────────────────────────

    pub fn arch(&self) -> &ModelArch {
        &self.header.arch
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Directory index of a tensor by name (same resolution as
    /// [`Self::tensor`] — engines must not re-scan the directory).
    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// Tensor-source indirection (spec §9, Patent 15 fig3/302): the
    /// skill's replacement is read IN PLACE OF the backbone tensor —
    /// either/or, never combined. None skill → backbone directly.
    pub fn resolve_tensor(&self, name: &str, skill: Option<&str>) -> Option<&TensorEntry> {
        if let Some(sid) = skill {
            if let Some(t) = self.tensor(&format!("skill.{sid}.{name}")) {
                return Some(t);
            }
        }
        self.tensor(name)
    }

    /// The per-skill delta index view (claim 2): directory entries of
    /// one skill — exactly the byte ranges lazy loading pages in.
    pub fn skill_tensors(&self, skill_id: &str) -> impl Iterator<Item = &TensorEntry> {
        let prefix = format!("skill.{skill_id}.");
        self.tensors.iter().filter(move |t| t.name.starts_with(&prefix))
    }

    /// Zero-copy bytes of a tensor from the mmap'd data section.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], CmfError> {
        let entry = self
            .tensor(name)
            .ok_or_else(|| CmfError::MissingTensor(name.to_string()))?;
        Ok(self.entry_bytes(entry))
    }

    /// All bytes of the primary mapping (GPU path: no-copy Metal buffer
    /// over the same mmap — unified memory, zero copying).
    pub fn primary_bytes(&self) -> &[u8] {
        self.backing.bytes()
    }

    /// Absolute offset of the tensor within the primary mapping
    /// (None for tensors from sibling shards).
    pub fn entry_abs_offset(&self, entry: &TensorEntry) -> Option<usize> {
        (entry.shard == 0).then(|| (self.data_off + entry.off) as usize)
    }

    pub fn entry_bytes(&self, entry: &TensorEntry) -> &[u8] {
        let (bytes, data_off) = if entry.shard == 0 {
            (self.backing.bytes(), self.data_off)
        } else {
            let (b, o) = &self.extra_shards[entry.shard - 1];
            (b.bytes(), *o)
        };
        let start = (data_off + entry.off) as usize;
        &bytes[start..start + entry.nbytes as usize]
    }

    /// Tensors belonging to layer `i` (prefix `model.layers.{i}.`).
    pub fn layer_tensors(&self, layer_idx: usize) -> Vec<&TensorEntry> {
        let prefix = format!("model.layers.{layer_idx}.");
        self.tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .collect()
    }

    /// Total parameter count estimated from matrix tensors (ndim ≥ 2).
    pub fn total_param_count(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.shape.len() >= 2)
            .map(|t| t.n_elems() as u64)
            .sum()
    }

    /// Recompute all tensor hashes; returns human-readable problems
    /// (empty = file intact).
    pub fn verify(&self) -> Vec<String> {
        let mut problems = Vec::new();

        // Section-level integrity (spec §8.1). Zero/absent = legacy file.
        let bytes = self.backing.bytes();
        let env = &self.envelope;
        let sect = |(off, len): (u64, u64)| &bytes[off as usize..(off + len) as usize];
        let check = |name: &str, stored: u64, span: (u64, u64)| -> Option<String> {
            if stored != 0 && span.1 > 0 {
                let actual = hash64(sect(span));
                if actual != stored {
                    return Some(format!(
                        "section '{name}': hash mismatch (stored {stored:016x}, \
                         actual {actual:016x})"
                    ));
                }
            }
            None
        };
        problems.extend(check("header", env.header_hash, env.header));
        problems.extend(check("dir", env.dir_hash, env.dir));
        if let Some(sh) = &self.header.section_hashes {
            for (name, hex, span) in [
                ("masks", &sh.masks, env.masks),
                ("vocab", &sh.vocab, env.vocab),
                ("index", &sh.index, env.index),
            ] {
                if let Some(hex) = hex {
                    match u64::from_str_radix(hex, 16) {
                        Ok(stored) => problems.extend(check(name, stored, span)),
                        Err(_) => problems.push(format!(
                            "section '{name}': malformed hash '{hex}'"
                        )),
                    }
                }
            }
        }

        for t in &self.tensors {
            let actual = hash64(self.entry_bytes(t));
            if actual != t.hash {
                problems.push(format!(
                    "tensor '{}': hash mismatch (stored {:016x}, actual {:016x})",
                    t.name, t.hash, actual
                ));
            }
        }
        problems
    }

    /// Approximate active weight bytes under a mask, from real tensor
    /// sizes in the directory (not from a formula).
    pub fn compute_active_size(&self, mask: &TaskMask) -> u64 {
        let arch = &self.header.arch;
        let mut total = 0u64;
        for li in 0..arch.num_layers {
            if !mask.layer_alive(li) {
                continue;
            }
            let ffn_frac = mask.ffn_active_count(li) as f64 / arch.intermediate_size.max(1) as f64;
            let head_frac =
                mask.active_head_count(li) as f64 / arch.num_attention_heads.max(1) as f64;
            for t in self.layer_tensors(li) {
                let frac = if t.name.contains(".mlp.") {
                    ffn_frac
                } else if t.name.contains(".self_attn.") {
                    head_frac
                } else {
                    1.0
                };
                total += (t.nbytes as f64 * frac) as u64;
            }
        }
        total
    }

    // ───────────────────────── writer ─────────────────────────

    /// Write a CMF v2 file. Offsets, alignment, hashes and the sparse
    /// index are computed here — the caller supplies content only.
    pub fn write(
        path: impl AsRef<Path>,
        header: &CmfHeader,
        tensors: &[TensorSpec],
        masks: Option<&MaskCatalog>,
        vocab: Option<&[u8]>,
    ) -> Result<(), CmfError> {
        let path = path.as_ref();

        // Directory + data layout.
        let mut entries = Vec::with_capacity(tensors.len());
        let mut data_cursor = 0u64;
        for t in tensors {
            if t.shape.len() > DIR_MAX_NDIM {
                return Err(CmfError::Parse(format!(
                    "tensor '{}': ndim {} > 6",
                    t.name,
                    t.shape.len()
                )));
            }
            if let Some(expect) = expected_nbytes(t.dtype, &t.shape) {
                if expect != t.data.len() {
                    return Err(CmfError::Bounds(format!(
                        "tensor '{}': data {} bytes != expected {} for {:?}{:?}",
                        t.name,
                        t.data.len(),
                        expect,
                        t.dtype,
                        t.shape
                    )));
                }
            }
            data_cursor = align_to(data_cursor, TENSOR_ALIGNMENT);
            entries.push(TensorEntry {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                off: data_cursor,
                nbytes: t.data.len() as u64,
                shard: 0,
                hash: hash64(&t.data),
            });
            data_cursor += t.data.len() as u64;
        }
        let data_len = data_cursor;

        let dir_bytes = Self::encode_directory(&entries);

        let masks_bytes = match masks {
            Some(catalog) if !catalog.masks.is_empty() => {
                Some(encode_masks_section(catalog, &header.arch).map_err(CmfError::Parse)?)
            }
            _ => None,
        };
        let index_bytes = match masks {
            Some(catalog) if !catalog.masks.is_empty() => {
                let idx = build_sparse_index(catalog, &header.arch);
                Some(encode_sparse_index(&idx))
            }
            _ => None,
        };

        // Section hashes go INTO the header (so the envelope's header
        // hash transitively covers them), then the header is serialized.
        let hex = |b: Option<&[u8]>| b.map(|b| format!("{:016x}", hash64(b)));
        let mut header = header.clone();
        if masks_bytes.is_some() || vocab.is_some() || index_bytes.is_some() {
            header.section_hashes = Some(SectionHashes {
                masks: hex(masks_bytes.as_deref()),
                vocab: hex(vocab),
                index: hex(index_bytes.as_deref()),
            });
        }
        let header_json =
            serde_json::to_vec(&header).map_err(|e| CmfError::Parse(format!("header: {e}")))?;

        let mut required_features = features::TENSOR_DIR;
        if masks_bytes.is_some() {
            required_features |= features::BINARY_MASKS;
        }
        if entries
            .iter()
            .any(|t| matches!(t.dtype, TensorDtype::Q8_2f | TensorDtype::Vbit))
        {
            required_features |= features::QUANT_2F;
        }

        // Section offsets.
        let header_off = ENVELOPE_LEN as u64;
        let dir_off = header_off + header_json.len() as u64;
        let data_off = align_to(dir_off + dir_bytes.len() as u64, DATA_ALIGNMENT);
        let masks_off = data_off + data_len;
        let masks_len = masks_bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);
        let vocab_off = masks_off + masks_len;
        let vocab_len = vocab.map(|b| b.len() as u64).unwrap_or(0);
        let index_off = vocab_off + vocab_len;
        let index_len = index_bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);

        // Envelope.
        let mut env = Vec::with_capacity(ENVELOPE_LEN);
        env.extend_from_slice(&CMF_MAGIC);
        env.extend_from_slice(&CMF_VERSION.to_le_bytes());
        env.extend_from_slice(&0u32.to_le_bytes()); // flags
        env.extend_from_slice(&required_features.to_le_bytes());
        for (off, len) in [
            (header_off, header_json.len() as u64),
            (dir_off, dir_bytes.len() as u64),
            (data_off, data_len),
            (if masks_len > 0 { masks_off } else { 0 }, masks_len),
            (if vocab_len > 0 { vocab_off } else { 0 }, vocab_len),
            (if index_len > 0 { index_off } else { 0 }, index_len),
        ] {
            env.extend_from_slice(&off.to_le_bytes());
            env.extend_from_slice(&len.to_le_bytes());
        }
        // Reserved bytes carry header/dir integrity (spec §8.1).
        env.extend_from_slice(&hash64(&header_json).to_le_bytes());
        env.extend_from_slice(&hash64(&dir_bytes).to_le_bytes());
        env.resize(ENVELOPE_LEN, 0);

        // Write out.
        let mut f = BufWriter::new(File::create(path)?);
        f.write_all(&env)?;
        f.write_all(&header_json)?;
        f.write_all(&dir_bytes)?;
        let mut pos = dir_off + dir_bytes.len() as u64;
        f.write_all(&zeros((data_off - pos) as usize))?;
        pos = data_off;
        for (spec, entry) in tensors.iter().zip(&entries) {
            let target = data_off + entry.off;
            f.write_all(&zeros((target - pos) as usize))?;
            f.write_all(&spec.data)?;
            pos = target + spec.data.len() as u64;
        }
        debug_assert_eq!(pos, data_off + data_len);
        if let Some(mb) = &masks_bytes {
            f.write_all(mb)?;
        }
        if let Some(vb) = vocab {
            f.write_all(vb)?;
        }
        if let Some(ib) = &index_bytes {
            f.write_all(ib)?;
        }
        f.flush()?;

        tracing::info!(
            "Wrote CMF v2: {} ({} tensors, {} masks, {:.1} MB)",
            path.display(),
            entries.len(),
            masks.map(|m| m.masks.len()).unwrap_or(0),
            std::fs::metadata(path)?.len() as f64 / 1e6
        );
        Ok(())
    }

    fn encode_directory(entries: &[TensorEntry]) -> Vec<u8> {
        let mut pool = Vec::new();
        let mut name_offs = Vec::with_capacity(entries.len());
        for e in entries {
            name_offs.push((pool.len() as u32, e.name.len() as u16));
            pool.extend_from_slice(e.name.as_bytes());
        }
        let pool_off = 16 + entries.len() * DIR_RECORD_LEN;

        let mut out = Vec::with_capacity(pool_off + pool.len());
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        out.extend_from_slice(&(pool_off as u64).to_le_bytes());
        for (e, (noff, nlen)) in entries.iter().zip(&name_offs) {
            out.extend_from_slice(&noff.to_le_bytes());
            out.extend_from_slice(&nlen.to_le_bytes());
            out.push(e.dtype.id());
            out.push(e.shape.len() as u8);
            for d in 0..DIR_MAX_NDIM {
                out.extend_from_slice(&(e.shape.get(d).copied().unwrap_or(0) as u32).to_le_bytes());
            }
            out.extend_from_slice(&e.off.to_le_bytes());
            out.extend_from_slice(&e.nbytes.to_le_bytes());
            out.extend_from_slice(&e.hash.to_le_bytes());
        }
        out.extend_from_slice(&pool);
        out
    }
}

fn align_to(x: u64, a: u64) -> u64 {
    (x + a - 1) / a * a
}

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

// ───────────────────── sparse index (§7 of the spec) ─────────────────────

/// Build the sparse index from mask bitfields: a 32-neuron FFN group is
/// active if it contains at least one active bit.
pub fn build_sparse_index(catalog: &MaskCatalog, arch: &ModelArch) -> Vec<SparseIndexEntry> {
    let mut out = Vec::new();
    for m in &catalog.masks {
        for li in 0..arch.num_layers {
            if !m.layer_alive(li) {
                continue;
            }
            let mut groups = Vec::new();
            if let Some(bits) = m.ffn_masks.get(li) {
                let n_groups = (arch.intermediate_size + 31) / 32;
                for g in 0..n_groups {
                    // Group g covers bits [g*32, g*32+32) = bytes [g*4, g*4+4).
                    let active = bits[g * 4..(g * 4 + 4).min(bits.len())]
                        .iter()
                        .any(|&b| b != 0);
                    if active {
                        groups.push(g as u16);
                    }
                }
            }
            let mut heads = Vec::new();
            if let Some(bits) = m.head_masks.get(li) {
                for h in 0..arch.num_attention_heads {
                    if bits.get(h / 8).map(|b| b & (1 << (h % 8)) != 0).unwrap_or(false) {
                        heads.push(h as u8);
                    }
                }
            }
            out.push(SparseIndexEntry {
                task_id: m.task_id,
                layer_idx: li,
                active_ffn_groups: groups,
                active_heads: heads,
            });
        }
    }
    out
}

/// `[u32 n_entries][u32 reserved]` then per entry:
/// `[u32 task][u32 layer][u32 n_groups][u32 n_heads][u16×g][u8×h][pad→4]`.
pub fn encode_sparse_index(entries: &[SparseIndexEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.task_id.to_le_bytes());
        out.extend_from_slice(&(e.layer_idx as u32).to_le_bytes());
        out.extend_from_slice(&(e.active_ffn_groups.len() as u32).to_le_bytes());
        out.extend_from_slice(&(e.active_heads.len() as u32).to_le_bytes());
        for g in &e.active_ffn_groups {
            out.extend_from_slice(&g.to_le_bytes());
        }
        out.extend_from_slice(&e.active_heads);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    out
}

pub fn decode_sparse_index(bytes: &[u8]) -> Result<Vec<SparseIndexEntry>, CmfError> {
    let err = |msg: &str| CmfError::Parse(format!("sparse index: {msg}"));
    if bytes.len() < 8 {
        return Err(err("too short"));
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut pos = 8usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if pos + 16 > bytes.len() {
            return Err(err("entry header out of bounds"));
        }
        let task_id = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        let layer_idx = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let n_groups = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap()) as usize;
        let n_heads = u32::from_le_bytes(bytes[pos + 12..pos + 16].try_into().unwrap()) as usize;
        pos += 16;
        if pos + n_groups * 2 + n_heads > bytes.len() {
            return Err(err("entry data out of bounds"));
        }
        let mut groups = Vec::with_capacity(n_groups);
        for g in 0..n_groups {
            groups.push(u16::from_le_bytes(bytes[pos + g * 2..pos + g * 2 + 2].try_into().unwrap()));
        }
        pos += n_groups * 2;
        let heads = bytes[pos..pos + n_heads].to_vec();
        pos += n_heads;
        pos = (pos + 3) / 4 * 4;
        out.push(SparseIndexEntry {
            task_id,
            layer_idx,
            active_ffn_groups: groups,
            active_heads: heads,
        });
    }
    Ok(out)
}

/// Errors from CMF operations. Every failure mode is loud.
#[derive(Debug, thiserror::Error)]
pub enum CmfError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Invalid CMF magic bytes")]
    InvalidMagic,
    #[error("Unsupported CMF version: {0}")]
    UnsupportedVersion(u32),
    #[error("File requires unsupported features (bits {0:#x})")]
    UnsupportedFeature(u32),
    #[error("Unknown tensor dtype id: {0}")]
    UnknownDtype(u8),
    #[error("Tensor not found: {0}")]
    MissingTensor(String),
    #[error("Bounds error: {0}")]
    Bounds(String),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}
