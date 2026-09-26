//! Strict writer and loader of decision CMF files and overlay generations (spec §2, §5.10).
//!
//! **Loader** ([`DecisionModel::open`], [`DecisionModel::open_with_overlay`]). A
//! file is refused ([`Refusal`]) when (spec §2.8):
//! * the CMF envelope does not open (cortiq-core refuses a decision arch without
//!   the DECISION bit and the bit without the arch), or it carries masks, a vocab
//!   section, a sparse index, shards, header skills or routing;
//! * the DECISION bit (0x800) is missing — every language-model file;
//! * the arch is a v3 research profile (hint: rebuild with `cortiq decision`), or
//!   not the expected profile;
//! * `decision.manifest` is missing, not U8, larger than 16 MiB, not canonical
//!   JSON, of another schema, or its `representation_id` does not recompute;
//! * the header `hidden_size` is not the signal dimension (`dim_p + 4096`; 4480
//!   for the release encoder);
//! * the hashing record differs from the compiled `cortiq-hashfeat-v1` contract or
//!   a golden hash does not recompute bit for bit (another `unicode_version` alone
//!   is a warning);
//! * a skill manifest's sha256 differs from `decision.manifest`, or it fails its
//!   own checks;
//! * a tensor the manifests name is missing or has another dtype or shape, or a
//!   tensor outside the allowlist exists (any name, `decision.*` or not);
//! * with [`Verify::Full`]: any sha256 differs (task tensors, rows blobs, the
//!   encoder aggregate `tensors_sha256`, the vocab), a rows blob does not decode or
//!   disagrees with its record, or cortiq-core's hash64 integrity check fails.
//!
//! [`Verify::Light`] (`cortiq decide`) checks the envelope, every manifest, the
//! hashing golden and the directory (names, dtypes, shapes); [`Verify::Full`]
//! (`cortiq decision verify`, the start of `serve`, and every writer before it
//! publishes a file) also hashes every byte. The encoder golden φ_P needs the
//! encoder and is checked by the encoder module.
//!
//! **Overlay** (spec §5.10): a generation file `cortiq-decision-overlay-v1` holds
//! `decision.overlay.manifest` (full manifests of the skills that differ from the
//! base, cumulative since generation 0), the replaced task tensors under their own
//! names and `decision.skill.{id}.rows.learned`. Loading = base + overlay, overlay
//! tensors first. The base's rows blob and encoder are never replaced.
//!
//! **Writers** ([`FileBuilder`], [`OverlayBuilder`]): a temp file in the output's
//! directory, fsync, a strict [`Verify::Full`] open of the temp file, then a
//! no-clobber publish (`hard_link`, which fails instead of replacing; a
//! `create_new` copy where links are unsupported), then a re-open of the output.
//! An existing output is never replaced ([`OutputExists`]). With the same inputs
//! and `SOURCE_DATE_EPOCH` the bytes are identical. `add-skill` copies the encoder
//! and every prior skill's manifest and tensors byte for byte; only
//! `decision.manifest` is new.

use crate::canonical;
use crate::manifest::{
    self, BASE_PROFILE, DecisionManifest, ENCODER_PREFIX, EncoderRecord, GOLDEN_TENSOR,
    LEGACY_PROFILES, LearnedRowsRecord, MANIFEST_SCHEMA, MANIFEST_TENSOR, MAX_MANIFEST_BYTES,
    OVERLAY_MANIFEST_TENSOR, OVERLAY_PROFILE, OVERLAY_SCHEMA, OverlayEvent, OverlayManifest,
    Representation, RowsRecord, SkillManifest, SkillRef, TaskRecord, TensorDigest, VOCAB_TENSOR,
    rows_learned_tensor, rows_tensor, sha256_hex, skill_manifest_tensor, task_basis_tensor,
    task_mean_tensor,
};
use crate::resonance::{TaskView, Topology};
use crate::rows::{self, Rows, Split};
use anyhow::{Result, bail, ensure};
use cortiq_core::format::features;
use cortiq_core::{
    CmfHeader, CmfModel, QuantType, TensorDtype, TensorEntry, TensorSpec, TensorSpecRef,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------ errors

/// Why a file is not a loadable decision model (spec §2.8).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error("cannot open {path}: {reason}")]
    Container { path: String, reason: String },
    #[error(
        "{path} is not a decision file: the DECISION feature bit (0x800) is missing (arch '{arch}'); language models are served by `cortiq run`/`cortiq serve`"
    )]
    MissingBit { path: String, arch: String },
    #[error(
        "{path}: '{arch}' is a v3 research decision profile; rebuild it with `cortiq decision train` / `cortiq decision add-skill` (profile {BASE_PROFILE})"
    )]
    LegacyProfile { path: String, arch: String },
    #[error("{path}: unsupported decision profile '{arch}' (expected '{expected}')")]
    Profile {
        path: String,
        arch: String,
        expected: String,
    },
    #[error("{path}: {reason}")]
    Header { path: String, reason: String },
    #[error("{tensor}: {reason}")]
    Manifest { tensor: String, reason: String },
    #[error("unsupported manifest schema '{found}' (expected '{expected}')")]
    Schema { found: String, expected: String },
    #[error("representation_id {stored} does not match the representation ({computed})")]
    RepresentationId { stored: String, computed: String },
    #[error("header hidden_size {header} != signal dimension {signal}")]
    HiddenSize { header: usize, signal: usize },
    #[error("hashing contract mismatch: {0}")]
    Hashing(String),
    #[error("tensor '{0}' named by the manifests is missing")]
    MissingTensor(String),
    #[error("unexpected tensor '{0}' (not in the decision allowlist)")]
    ExtraTensor(String),
    #[error("tensor '{name}': dtype {found} (expected {expected})")]
    Dtype {
        name: String,
        found: String,
        expected: String,
    },
    #[error("tensor '{name}': shape {found:?} (expected {expected:?})")]
    Shape {
        name: String,
        found: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("tensor '{name}': sha256 {found} (manifest {expected})")]
    Sha {
        name: String,
        found: String,
        expected: String,
    },
    #[error("skill '{skill}': {reason}")]
    Skill { skill: String, reason: String },
    #[error("rows of skill '{skill}': {reason}")]
    Rows { skill: String, reason: String },
    #[error("overlay: {0}")]
    Overlay(String),
    #[error("integrity: {0}")]
    Integrity(String),
}

/// A writer refused to replace an existing output (spec §3.1).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("refusing to overwrite existing output {0}")]
pub struct OutputExists(pub PathBuf);

/// How much of a file an open checks (see the module notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verify {
    /// Envelope, manifests, hashing golden, directory (names, dtypes, shapes).
    Light,
    /// [`Verify::Light`] plus every sha256, the rows blobs and hash64 integrity.
    Full,
}

// ------------------------------------------------------------------ loaded model

/// A task's `(mean, basis)` as stored (borrowed from the mmap when aligned).
pub type TopologyRef<'m> = (Cow<'m, [f32]>, Cow<'m, [f32]>);

/// A skill as loaded (base or overlay manifest).
#[derive(Clone, Debug)]
pub struct LoadedSkill {
    pub manifest: SkillManifest,
    /// Canonical bytes of the manifest (the tensor bytes in a full file).
    pub bytes: Vec<u8>,
    pub sha256: String,
    /// The manifest comes from the overlay generation.
    pub from_overlay: bool,
}

impl LoadedSkill {
    pub fn id(&self) -> &str {
        &self.manifest.id
    }
}

/// What [`DecisionModel::verify_full`] checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyReport {
    pub model_sha: String,
    pub base_model_sha: String,
    pub generation: u64,
    pub skills: usize,
    pub tensors: usize,
    pub bytes_hashed: u64,
    pub warnings: Vec<String>,
}

/// A decision file (and optionally a generation overlay), checked and ready to
/// serve. Weights stay in the mmap.
pub struct DecisionModel {
    base: CmfModel,
    overlay: Option<CmfModel>,
    manifest: DecisionManifest,
    base_model_sha: String,
    representation: Representation,
    overlay_manifest: Option<OverlayManifest>,
    overlay_sha: Option<String>,
    skills: Vec<LoadedSkill>,
    by_id: HashMap<String, usize>,
    warnings: Vec<String>,
}

impl std::fmt::Debug for DecisionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionModel")
            .field("path", &self.base.path)
            .field("model_sha", &self.model_sha())
            .field("generation", &self.generation())
            .field(
                "skills",
                &self.skills.iter().map(LoadedSkill::id).collect::<Vec<_>>(),
            )
            .finish()
    }
}

fn path_str(p: &Path) -> String {
    p.display().to_string()
}

fn dtype_name(d: TensorDtype) -> String {
    format!("{d:?}")
}

fn open_container(path: &Path) -> Result<CmfModel, Refusal> {
    CmfModel::open(path).map_err(|e| Refusal::Container {
        path: path_str(path),
        reason: e.to_string(),
    })
}

/// Bit, profile and the header/sections a decision file must (not) carry.
fn check_envelope(c: &CmfModel, expected: &str) -> Result<(), Refusal> {
    let path = path_str(&c.path);
    let arch = c.header.arch.arch_name.clone();
    if c.required_features & features::DECISION == 0 {
        return Err(Refusal::MissingBit { path, arch });
    }
    if LEGACY_PROFILES.contains(&arch.as_str()) {
        return Err(Refusal::LegacyProfile { path, arch });
    }
    if arch != expected {
        return Err(Refusal::Profile {
            path,
            arch,
            expected: expected.into(),
        });
    }
    let h = &c.header;
    let bad = |reason: &str| {
        Err(Refusal::Header {
            path: path.clone(),
            reason: reason.into(),
        })
    };
    if h.version != 2 {
        return bad("header version must be 2");
    }
    if !matches!(h.quant_type, QuantType::F32) {
        return bad("quant_type must be F32");
    }
    if h.arch.num_layers != 0 {
        return bad("a decision file has num_layers 0");
    }
    if !h.skills.is_empty() || h.routing.is_some() || h.calibration.is_some() {
        return bad("a decision file carries no header skills, routing or calibration");
    }
    if h.shard.is_some() {
        return bad("a decision file is not sharded");
    }
    if h.tokenizer_config.is_some() {
        return bad("a decision file carries no tokenizer_config");
    }
    if !c.masks.masks.is_empty() || !c.sparse_index.is_empty() || c.vocab.is_some() {
        return bad("a decision file carries no masks, sparse index or vocab section");
    }
    Ok(())
}

/// Read a U8 manifest tensor: presence, dtype, 1-D shape, 16 MiB bound.
fn manifest_bytes<'m>(c: &'m CmfModel, name: &str) -> Result<&'m [u8], Refusal> {
    let bad = |reason: String| Refusal::Manifest {
        tensor: name.into(),
        reason,
    };
    let e = c.tensor(name).ok_or_else(|| bad("missing".into()))?;
    if e.dtype != TensorDtype::U8 {
        return Err(bad(format!("dtype {} (expected U8)", dtype_name(e.dtype))));
    }
    if e.shape != [e.nbytes as usize] {
        return Err(bad(format!("shape {:?} is not [{}]", e.shape, e.nbytes)));
    }
    if e.nbytes as usize > MAX_MANIFEST_BYTES {
        return Err(bad(format!(
            "{} bytes exceed the 16 MiB manifest limit",
            e.nbytes
        )));
    }
    Ok(c.entry_bytes(e))
}

/// f32 values of little-endian bytes, borrowed when aligned (spec §2.2).
pub fn f32_view(bytes: &[u8]) -> Cow<'_, [f32]> {
    if let Some(s) = try_borrow_f32(bytes) {
        return Cow::Borrowed(s);
    }
    Cow::Owned(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect(),
    )
}

#[cfg(target_endian = "little")]
fn try_borrow_f32(bytes: &[u8]) -> Option<&[f32]> {
    bytemuck::try_cast_slice::<u8, f32>(bytes).ok()
}

#[cfg(not(target_endian = "little"))]
fn try_borrow_f32(_bytes: &[u8]) -> Option<&[f32]> {
    None
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// What the manifests say about one tensor.
#[derive(Clone, Debug)]
struct Expect {
    dtype: TensorDtype,
    /// `None` = U8 blob of any length (`shape == [nbytes]`).
    shape: Option<Vec<usize>>,
    sha256: Option<String>,
}

fn check_entry(e: &TensorEntry, x: &Expect) -> Result<(), Refusal> {
    if e.dtype != x.dtype {
        return Err(Refusal::Dtype {
            name: e.name.clone(),
            found: dtype_name(e.dtype),
            expected: dtype_name(x.dtype),
        });
    }
    let want = x.shape.clone().unwrap_or_else(|| vec![e.nbytes as usize]);
    if e.shape != want {
        return Err(Refusal::Shape {
            name: e.name.clone(),
            found: e.shape.clone(),
            expected: want,
        });
    }
    Ok(())
}

fn check_sha(name: &str, bytes: &[u8], expected: &str) -> Result<(), Refusal> {
    let found = sha256_hex(bytes);
    if found != expected {
        return Err(Refusal::Sha {
            name: name.into(),
            found,
            expected: expected.into(),
        });
    }
    Ok(())
}

/// Tensors a skill manifest names (task tensors, rows, learned rows).
fn skill_tensor_expectations(m: &SkillManifest, signal_dim: usize) -> Vec<(String, Expect)> {
    let mut v = Vec::new();
    for t in &m.tasks {
        let i = t.i as usize;
        if let Some(s) = &t.mean_sha256 {
            v.push((
                task_mean_tensor(&m.id, i),
                Expect {
                    dtype: TensorDtype::F32,
                    shape: Some(vec![signal_dim]),
                    sha256: Some(s.clone()),
                },
            ));
        }
        if let Some(s) = &t.basis_sha256 {
            v.push((
                task_basis_tensor(&m.id, i),
                Expect {
                    dtype: TensorDtype::F32,
                    shape: Some(vec![t.k as usize, signal_dim]),
                    sha256: Some(s.clone()),
                },
            ));
        }
    }
    v.push((
        m.rows.tensor.clone(),
        Expect {
            dtype: TensorDtype::U8,
            shape: None,
            sha256: Some(m.rows.sha256.clone()),
        },
    ));
    if let Some(r) = &m.rows_learned {
        v.push((
            r.tensor.clone(),
            Expect {
                dtype: TensorDtype::U8,
                shape: None,
                sha256: Some(r.sha256.clone()),
            },
        ));
    }
    v
}

fn parse_skill(
    bytes: &[u8],
    id: &str,
    representation_id: &str,
    signal_dim: usize,
) -> Result<SkillManifest, Refusal> {
    let skill_err = |reason: String| Refusal::Skill {
        skill: id.into(),
        reason,
    };
    let (m, _): (SkillManifest, Value) = manifest::parse_manifest_bytes("skill manifest", bytes)
        .map_err(|e| skill_err(e.to_string()))?;
    if m.id != id {
        return Err(skill_err(format!("manifest id is '{}'", m.id)));
    }
    m.validate(representation_id, signal_dim)
        .map_err(|e| skill_err(e.to_string()))?;
    Ok(m)
}

/// Decode a rows blob and check it against its record and the skill.
fn check_rows_blob(
    skill: &SkillManifest,
    bytes: &[u8],
    dim_p: usize,
    dim_h: usize,
    learned_only: Option<&LearnedRowsRecord>,
) -> Result<Rows, Refusal> {
    let err = |reason: String| Refusal::Rows {
        skill: skill.id.clone(),
        reason,
    };
    let r = Rows::decode(bytes).map_err(|e| err(e.to_string()))?;
    if (r.dim_p, r.dim_h) != (dim_p, dim_h) {
        return Err(err(format!(
            "blob dims ({}, {}) differ from the signal ({dim_p}, {dim_h})",
            r.dim_p, r.dim_h
        )));
    }
    if let Some(row) = r.rows.iter().find(|x| x.task as usize >= skill.tasks.len()) {
        return Err(err(format!(
            "row task {} outside the {} tasks",
            row.task,
            skill.tasks.len()
        )));
    }
    let counts = [
        r.count(Split::Train) as u64,
        r.count(Split::Calibration) as u64,
        r.count(Split::Learned) as u64,
    ];
    match learned_only {
        None => {
            let want = [
                skill.rows.n_train,
                skill.rows.n_calibration,
                skill.rows.n_learned,
            ];
            if counts != want {
                return Err(err(format!(
                    "blob has {counts:?} train/calibration/learned rows, the manifest {want:?}"
                )));
            }
        }
        Some(rec) => {
            if counts != [0, 0, rec.n] {
                return Err(err(format!(
                    "learned rows blob has {counts:?} train/calibration/learned rows, the manifest [0, 0, {}]",
                    rec.n
                )));
            }
        }
    }
    Ok(r)
}

impl DecisionModel {
    /// Open a decision file (no overlay).
    pub fn open(path: impl AsRef<Path>, verify: Verify) -> Result<Self, Refusal> {
        Self::open_with_overlay(path, None::<&Path>, verify)
    }

    /// Open a decision file and apply a generation overlay.
    pub fn open_with_overlay(
        base: impl AsRef<Path>,
        overlay: Option<impl AsRef<Path>>,
        verify: Verify,
    ) -> Result<Self, Refusal> {
        Self::open_parts(
            base.as_ref(),
            overlay.as_ref().map(AsRef::as_ref),
            verify,
            verify,
        )
    }

    /// Open with separate check levels for the base file and the overlay (a
    /// generation writer re-checks its new overlay fully on a base already served).
    pub fn open_parts(
        base_path: &Path,
        overlay: Option<&Path>,
        base_verify: Verify,
        overlay_verify: Verify,
    ) -> Result<Self, Refusal> {
        let c = open_container(base_path)?;
        check_envelope(&c, BASE_PROFILE)?;

        // decision.manifest: bytes, canonical, typed, schema.
        let mbytes = manifest_bytes(&c, MANIFEST_TENSOR)?;
        let base_model_sha = sha256_hex(mbytes);
        let mvalue = canonical::parse_canonical(mbytes).map_err(|e| Refusal::Manifest {
            tensor: MANIFEST_TENSOR.into(),
            reason: e.to_string(),
        })?;
        if let Some(s) = mvalue.get("schema").and_then(Value::as_str) {
            if s != MANIFEST_SCHEMA {
                return Err(Refusal::Schema {
                    found: s.into(),
                    expected: MANIFEST_SCHEMA.into(),
                });
            }
        }
        let manifest: DecisionManifest = manifest::from_value("decision.manifest", &mvalue)
            .map_err(|e| Refusal::Manifest {
                tensor: MANIFEST_TENSOR.into(),
                reason: e.to_string(),
            })?;
        manifest.validate().map_err(|e| Refusal::Manifest {
            tensor: MANIFEST_TENSOR.into(),
            reason: e.to_string(),
        })?;

        // Representation: identity, typed checks, dimensions, hashing contract.
        let computed = manifest::representation_id(&manifest.representation);
        if computed != manifest.representation_id {
            return Err(Refusal::RepresentationId {
                stored: manifest.representation_id.clone(),
                computed,
            });
        }
        let representation = Representation::from_value(&manifest.representation)
            .and_then(|r| r.validate().map(|()| r))
            .map_err(|e| Refusal::Manifest {
                tensor: MANIFEST_TENSOR.into(),
                reason: e.to_string(),
            })?;
        let signal_dim = representation.signal_dim();
        if c.header.arch.hidden_size != signal_dim {
            return Err(Refusal::HiddenSize {
                header: c.header.arch.hidden_size,
                signal: signal_dim,
            });
        }
        let mut warnings =
            manifest::check_hashing(&representation.hashing).map_err(Refusal::Hashing)?;

        // Skill manifests.
        let rid = manifest.representation_id.clone();
        let mut skills = Vec::with_capacity(manifest.skills.len());
        for SkillRef {
            id,
            manifest_sha256,
        } in &manifest.skills
        {
            let name = skill_manifest_tensor(id);
            let bytes = manifest_bytes(&c, &name)?;
            check_sha(&name, bytes, manifest_sha256)?;
            let m = parse_skill(bytes, id, &rid, signal_dim)?;
            skills.push(LoadedSkill {
                manifest: m,
                bytes: bytes.to_vec(),
                sha256: manifest_sha256.clone(),
                from_overlay: false,
            });
        }

        // Allowlist: every tensor the manifests name, nothing else.
        let vocab_len = c.tensor(VOCAB_TENSOR).map_or(0, |e| e.nbytes as usize);
        let mut expect: BTreeMap<String, Expect> = BTreeMap::new();
        expect.insert(
            MANIFEST_TENSOR.into(),
            Expect {
                dtype: TensorDtype::U8,
                shape: None,
                sha256: None,
            },
        );
        for (name, dtype, shape) in representation.encoder.config.tensor_layout(vocab_len) {
            let shape = if dtype == TensorDtype::U8 {
                None
            } else {
                Some(shape)
            };
            expect.insert(
                name,
                Expect {
                    dtype,
                    shape,
                    sha256: None,
                },
            );
        }
        for s in &skills {
            expect.insert(
                skill_manifest_tensor(s.id()),
                Expect {
                    dtype: TensorDtype::U8,
                    shape: None,
                    sha256: None,
                },
            );
            for (name, x) in skill_tensor_expectations(&s.manifest, signal_dim) {
                expect.insert(name, x);
            }
        }
        for e in &c.tensors {
            if !expect.contains_key(&e.name) {
                return Err(Refusal::ExtraTensor(e.name.clone()));
            }
        }
        for (name, x) in &expect {
            let e = c
                .tensor(name)
                .ok_or_else(|| Refusal::MissingTensor(name.clone()))?;
            check_entry(e, x)?;
        }

        let by_id = skills
            .iter()
            .enumerate()
            .map(|(i, s)| (s.manifest.id.clone(), i))
            .collect();
        let mut model = Self {
            base: c,
            overlay: None,
            manifest,
            base_model_sha,
            representation,
            overlay_manifest: None,
            overlay_sha: None,
            skills,
            by_id,
            warnings: Vec::new(),
        };
        if base_verify == Verify::Full {
            model.check_base_bytes()?;
        }
        if let Some(ov) = overlay {
            model.apply_overlay(ov, overlay_verify)?;
        }
        for w in &warnings {
            tracing::warn!("{}: {w}", base_path.display());
        }
        model.warnings.append(&mut warnings);
        Ok(model)
    }

    /// Full checks of the base file's bytes.
    fn check_base_bytes(&self) -> Result<u64, Refusal> {
        let c = &self.base;
        let mut hashed = 0u64;
        let problems = c.verify();
        if !problems.is_empty() {
            return Err(Refusal::Integrity(problems.join("; ")));
        }
        // Encoder: aggregate sha256 and the vocab.
        let enc: Vec<&TensorEntry> = c
            .tensors
            .iter()
            .filter(|e| e.name.starts_with(ENCODER_PREFIX))
            .collect();
        let agg = manifest::tensors_sha256(enc.iter().map(|e| TensorDigest {
            name: &e.name,
            dtype: e.dtype,
            shape: &e.shape,
            data: c.entry_bytes(e),
        }));
        hashed += enc.iter().map(|e| e.nbytes).sum::<u64>();
        let rec = &self.representation.encoder;
        if agg != rec.tensors_sha256 {
            return Err(Refusal::Integrity(format!(
                "encoder tensors_sha256 {agg} differs from the representation's {}",
                rec.tensors_sha256
            )));
        }
        let vocab = c
            .tensor_bytes(VOCAB_TENSOR)
            .map_err(|_| Refusal::MissingTensor(VOCAB_TENSOR.into()))?;
        check_sha(VOCAB_TENSOR, vocab, &rec.source.vocab_sha256)?;
        let lines = std::str::from_utf8(vocab)
            .map_err(|_| Refusal::Integrity("vocab is not UTF-8".into()))?
            .lines()
            .count();
        if lines as u64 != rec.config.vocab {
            return Err(Refusal::Integrity(format!(
                "vocab has {lines} lines, the encoder config {}",
                rec.config.vocab
            )));
        }
        let golden = c
            .tensor_bytes(GOLDEN_TENSOR)
            .map_err(|_| Refusal::MissingTensor(GOLDEN_TENSOR.into()))?;
        if !f32_view(golden).iter().all(|v| v.is_finite()) {
            return Err(Refusal::Integrity("encoder golden is not finite".into()));
        }
        // Skills.
        let dims = (self.encoder_dim(), self.hashing_dim());
        for s in &self.skills {
            for (name, x) in skill_tensor_expectations(&s.manifest, self.signal_dim()) {
                let bytes = c
                    .tensor_bytes(&name)
                    .map_err(|_| Refusal::MissingTensor(name.clone()))?;
                if let Some(sha) = &x.sha256 {
                    check_sha(&name, bytes, sha)?;
                }
                hashed += bytes.len() as u64;
            }
            for t in &s.manifest.tasks {
                self.check_task_values(&s.manifest, t, c, None)?;
            }
            check_rows_blob(
                &s.manifest,
                c.tensor_bytes(&s.manifest.rows.tensor)
                    .map_err(|_| Refusal::MissingTensor(s.manifest.rows.tensor.clone()))?,
                dims.0,
                dims.1,
                None,
            )?;
            if let Some(r) = &s.manifest.rows_learned {
                check_rows_blob(
                    &s.manifest,
                    c.tensor_bytes(&r.tensor)
                        .map_err(|_| Refusal::MissingTensor(r.tensor.clone()))?,
                    dims.0,
                    dims.1,
                    Some(r),
                )?;
            }
        }
        Ok(hashed)
    }

    /// The topology of a task has the shapes of its record and finite values (its
    /// tensors resolved in `overlay` first, then in `c`).
    fn check_task_values(
        &self,
        m: &SkillManifest,
        t: &TaskRecord,
        c: &CmfModel,
        overlay: Option<&CmfModel>,
    ) -> Result<(), Refusal> {
        if !t.has_topology() {
            return Ok(());
        }
        let i = t.i as usize;
        let get = |name: String| -> Result<&[u8], Refusal> {
            if let Some(o) = overlay {
                if let Some(e) = o.tensor(&name) {
                    return Ok(o.entry_bytes(e));
                }
            }
            c.tensor_bytes(&name)
                .map_err(|_| Refusal::MissingTensor(name.clone()))
        };
        let mean = f32_view(get(task_mean_tensor(&m.id, i))?);
        let basis = if t.k > 0 {
            f32_view(get(task_basis_tensor(&m.id, i))?)
        } else {
            Cow::Borrowed(&[][..])
        };
        TaskView::new(&mean, &basis).map_err(|e| Refusal::Skill {
            skill: m.id.clone(),
            reason: format!("task {i}: {e}"),
        })?;
        Ok(())
    }

    fn apply_overlay(&mut self, path: &Path, verify: Verify) -> Result<(), Refusal> {
        let o = open_container(path)?;
        check_envelope(&o, OVERLAY_PROFILE)?;
        let signal_dim = self.signal_dim();
        if o.header.arch.hidden_size != signal_dim {
            return Err(Refusal::HiddenSize {
                header: o.header.arch.hidden_size,
                signal: signal_dim,
            });
        }
        let obytes = manifest_bytes(&o, OVERLAY_MANIFEST_TENSOR)?;
        let ovalue = canonical::parse_canonical(obytes).map_err(|e| Refusal::Manifest {
            tensor: OVERLAY_MANIFEST_TENSOR.into(),
            reason: e.to_string(),
        })?;
        if let Some(s) = ovalue.get("schema").and_then(Value::as_str) {
            if s != OVERLAY_SCHEMA {
                return Err(Refusal::Schema {
                    found: s.into(),
                    expected: OVERLAY_SCHEMA.into(),
                });
            }
        }
        let om: OverlayManifest = manifest::from_value("decision.overlay.manifest", &ovalue)
            .and_then(|m: OverlayManifest| m.validate().map(|()| m))
            .map_err(|e| Refusal::Manifest {
                tensor: OVERLAY_MANIFEST_TENSOR.into(),
                reason: e.to_string(),
            })?;
        if om.base_model_sha != self.base_model_sha {
            return Err(Refusal::Overlay(format!(
                "generation {} belongs to base {}, this base is {}",
                om.generation, om.base_model_sha, self.base_model_sha
            )));
        }
        if om.generation <= self.manifest.generation {
            return Err(Refusal::Overlay(format!(
                "generation {} is not newer than the base's {}",
                om.generation, self.manifest.generation
            )));
        }
        let rid = self.manifest.representation_id.clone();
        let mut allowed: BTreeMap<String, Expect> = BTreeMap::new();
        allowed.insert(
            OVERLAY_MANIFEST_TENSOR.into(),
            Expect {
                dtype: TensorDtype::U8,
                shape: None,
                sha256: None,
            },
        );
        let mut replaced: Vec<(usize, LoadedSkill)> = Vec::new();
        for (id, v) in &om.skills {
            let Some(&idx) = self.by_id.get(id) else {
                return Err(Refusal::Overlay(format!(
                    "skill '{id}' is not in the base file"
                )));
            };
            let bytes = canonical::to_vec(v);
            let m = parse_skill(&bytes, id, &rid, signal_dim)?;
            let base = &self.skills[idx].manifest;
            let ov_err = |reason: String| Refusal::Overlay(format!("skill '{id}': {reason}"));
            if m.rows != base.rows {
                return Err(ov_err(
                    "the build rows blob cannot change in a generation".into(),
                ));
            }
            for (name, x) in skill_tensor_expectations(&m, signal_dim) {
                if o.tensor(&name).is_some() {
                    allowed.insert(name, x);
                    continue;
                }
                // Not in the overlay: the base must hold the same bytes, i.e. the
                // base manifest records the same tensor with the same sha256.
                let same = if name == m.rows.tensor {
                    true // the record equals the base's (checked above)
                } else if m.rows_learned.as_ref().is_some_and(|r| r.tensor == name) {
                    base.rows_learned == m.rows_learned
                } else {
                    task_same_in_base(base, &name, x.sha256.as_deref())
                };
                if !same || self.base.tensor(&name).is_none() {
                    return Err(Refusal::MissingTensor(name));
                }
            }
            replaced.push((
                idx,
                LoadedSkill {
                    manifest: m,
                    sha256: sha256_hex(&bytes),
                    bytes,
                    from_overlay: true,
                },
            ));
        }
        for e in &o.tensors {
            match allowed.get(&e.name) {
                None => return Err(Refusal::ExtraTensor(e.name.clone())),
                Some(x) => check_entry(e, x)?,
            }
        }
        if verify == Verify::Full {
            let problems = o.verify();
            if !problems.is_empty() {
                return Err(Refusal::Integrity(problems.join("; ")));
            }
            for (name, x) in &allowed {
                if let Some(sha) = &x.sha256 {
                    let bytes = o
                        .tensor_bytes(name)
                        .map_err(|_| Refusal::MissingTensor(name.clone()))?;
                    check_sha(name, bytes, sha)?;
                }
            }
            let dims = (self.encoder_dim(), self.hashing_dim());
            for (_, s) in &replaced {
                for t in &s.manifest.tasks {
                    self.check_task_values(&s.manifest, t, &self.base, Some(&o))?;
                }
                if let Some(r) = &s.manifest.rows_learned {
                    let bytes = match o.tensor(&r.tensor) {
                        Some(e) => o.entry_bytes(e),
                        None => self
                            .base
                            .tensor_bytes(&r.tensor)
                            .map_err(|_| Refusal::MissingTensor(r.tensor.clone()))?,
                    };
                    check_rows_blob(&s.manifest, bytes, dims.0, dims.1, Some(r))?;
                }
            }
        }
        for (idx, s) in replaced {
            self.skills[idx] = s;
        }
        self.overlay_sha = Some(sha256_hex(obytes));
        self.overlay_manifest = Some(om);
        self.overlay = Some(o);
        Ok(())
    }

    // ------------------------------------------------------------ accessors

    /// The base file.
    pub fn base(&self) -> &CmfModel {
        &self.base
    }

    /// The generation overlay, when one is applied.
    pub fn overlay(&self) -> Option<&CmfModel> {
        self.overlay.as_ref()
    }

    pub fn base_path(&self) -> &Path {
        &self.base.path
    }

    pub fn manifest(&self) -> &DecisionManifest {
        &self.manifest
    }

    pub fn representation(&self) -> &Representation {
        &self.representation
    }

    pub fn representation_id(&self) -> &str {
        &self.manifest.representation_id
    }

    /// sha256 of the base file's `decision.manifest` bytes.
    pub fn base_model_sha(&self) -> &str {
        &self.base_model_sha
    }

    /// Identity of the served generation: the base `model_sha`, or the sha256 of
    /// the overlay manifest bytes when an overlay is applied
    /// (`model = "cortiq/decision@" + model_sha[..12]`).
    pub fn model_sha(&self) -> &str {
        self.overlay_sha.as_deref().unwrap_or(&self.base_model_sha)
    }

    /// The overlay's generation, else the file's.
    pub fn generation(&self) -> u64 {
        self.overlay_manifest
            .as_ref()
            .map_or(self.manifest.generation, |o| o.generation)
    }

    pub fn overlay_manifest(&self) -> Option<&OverlayManifest> {
        self.overlay_manifest.as_ref()
    }

    /// Skills in file order (overlay manifests where the overlay replaces them).
    pub fn skills(&self) -> &[LoadedSkill] {
        &self.skills
    }

    pub fn skill(&self, id: &str) -> Option<&LoadedSkill> {
        self.by_id.get(id).map(|&i| &self.skills[i])
    }

    /// Non-fatal findings of the open (e.g. another Unicode version).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// `dim_p + dim_h`.
    pub fn signal_dim(&self) -> usize {
        self.representation.signal_dim()
    }

    /// φ_P dimension.
    pub fn encoder_dim(&self) -> usize {
        self.representation.encoder_dim()
    }

    /// φ_H dimension.
    pub fn hashing_dim(&self) -> usize {
        self.representation.hashing_dim()
    }

    /// Directory entry of a tensor, overlay first.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.overlay
            .as_ref()
            .and_then(|o| o.tensor(name))
            .or_else(|| self.base.tensor(name))
    }

    /// Bytes of a tensor, overlay first.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        if let Some(o) = &self.overlay {
            if let Some(e) = o.tensor(name) {
                return Ok(o.entry_bytes(e));
            }
        }
        Ok(self.base.tensor_bytes(name)?)
    }

    /// f32 values of an F32 tensor, overlay first (borrowed when aligned).
    pub fn f32_tensor(&self, name: &str) -> Result<Cow<'_, [f32]>> {
        let e = self
            .tensor(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' is missing"))?;
        ensure!(e.dtype == TensorDtype::F32, "tensor '{name}' is not F32");
        Ok(f32_view(self.tensor_bytes(name)?))
    }

    /// An encoder weight by its name without the `decision.encoder.` prefix
    /// (e.g. `layer.0.attention.self.query.weight`).
    pub fn encoder_weight(&self, suffix: &str) -> Result<Cow<'_, [f32]>> {
        let name = format!("{ENCODER_PREFIX}{suffix}");
        let e = self
            .base
            .tensor(&name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' is missing"))?;
        ensure!(e.dtype == TensorDtype::F32, "tensor '{name}' is not F32");
        Ok(f32_view(self.base.entry_bytes(e)))
    }

    /// The bytes of vocab.txt.
    pub fn vocab_bytes(&self) -> Result<&[u8]> {
        Ok(self.base.tensor_bytes(VOCAB_TENSOR)?)
    }

    /// φ_P of the encoder golden texts, `[8 × dim_p]` row-major.
    pub fn encoder_golden(&self) -> Result<Cow<'_, [f32]>> {
        Ok(f32_view(self.base.tensor_bytes(GOLDEN_TENSOR)?))
    }

    fn skill_or_err(&self, id: &str) -> Result<&LoadedSkill> {
        self.skill(id)
            .ok_or_else(|| anyhow::anyhow!("no skill '{id}' in this model"))
    }

    fn task_record(&self, skill: &str, i: usize) -> Result<&TaskRecord> {
        let s = self.skill_or_err(skill)?;
        s.manifest
            .tasks
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("skill '{skill}' has no task {i}"))
    }

    /// The mean of a task (`None` when it has no topology).
    pub fn task_mean(&self, skill: &str, i: usize) -> Result<Option<Cow<'_, [f32]>>> {
        let t = self.task_record(skill, i)?;
        if !t.has_topology() {
            return Ok(None);
        }
        Ok(Some(self.f32_tensor(&task_mean_tensor(skill, i))?))
    }

    /// The basis of a task, `[k × signal_dim]` (empty when `k = 0`).
    pub fn task_basis(&self, skill: &str, i: usize) -> Result<Cow<'_, [f32]>> {
        let t = self.task_record(skill, i)?;
        if t.k == 0 {
            return Ok(Cow::Borrowed(&[]));
        }
        self.f32_tensor(&task_basis_tensor(skill, i))
    }

    /// The borrowed topology of a task, `(mean, basis)` (`None` when it has none).
    pub fn task_view(&self, skill: &str, i: usize) -> Result<Option<TopologyRef<'_>>> {
        let Some(mean) = self.task_mean(skill, i)? else {
            return Ok(None);
        };
        Ok(Some((mean, self.task_basis(skill, i)?)))
    }

    /// An owned copy of a task's topology.
    pub fn topology(&self, skill: &str, i: usize) -> Result<Option<Topology>> {
        Ok(self.task_view(skill, i)?.map(|(m, b)| Topology {
            mean: m.into_owned(),
            basis: b.into_owned(),
        }))
    }

    /// The build rows of a skill (decoded and checked).
    pub fn rows(&self, skill: &str) -> Result<Rows> {
        let s = self.skill_or_err(skill)?;
        let bytes = self.base.tensor_bytes(&rows_tensor(skill))?;
        Ok(check_rows_blob(
            &s.manifest,
            bytes,
            self.encoder_dim(),
            self.hashing_dim(),
            None,
        )?)
    }

    /// The learned rows kept in `rows.learned`, if any.
    pub fn rows_learned(&self, skill: &str) -> Result<Option<Rows>> {
        let s = self.skill_or_err(skill)?;
        let Some(r) = &s.manifest.rows_learned else {
            return Ok(None);
        };
        let bytes = self.tensor_bytes(&r.tensor)?;
        Ok(Some(check_rows_blob(
            &s.manifest,
            bytes,
            self.encoder_dim(),
            self.hashing_dim(),
            Some(r),
        )?))
    }

    /// Run every [`Verify::Full`] check now (`cortiq decision verify`).
    pub fn verify_full(&self) -> Result<VerifyReport, Refusal> {
        let mut hashed = self.check_base_bytes()?;
        let mut tensors = self.base.tensors.len();
        if let Some(o) = &self.overlay {
            let problems = o.verify();
            if !problems.is_empty() {
                return Err(Refusal::Integrity(problems.join("; ")));
            }
            tensors += o.tensors.len();
            let dims = (self.encoder_dim(), self.hashing_dim());
            for s in self.skills.iter().filter(|s| s.from_overlay) {
                for (name, x) in skill_tensor_expectations(&s.manifest, self.signal_dim()) {
                    let bytes = self
                        .tensor_bytes(&name)
                        .map_err(|_| Refusal::MissingTensor(name.clone()))?;
                    if let Some(sha) = &x.sha256 {
                        check_sha(&name, bytes, sha)?;
                    }
                    hashed += bytes.len() as u64;
                }
                for t in &s.manifest.tasks {
                    self.check_task_values(&s.manifest, t, &self.base, Some(o))?;
                }
                if let Some(r) = &s.manifest.rows_learned {
                    let bytes = self
                        .tensor_bytes(&r.tensor)
                        .map_err(|_| Refusal::MissingTensor(r.tensor.clone()))?;
                    check_rows_blob(&s.manifest, bytes, dims.0, dims.1, Some(r))?;
                }
            }
        }
        Ok(VerifyReport {
            model_sha: self.model_sha().into(),
            base_model_sha: self.base_model_sha.clone(),
            generation: self.generation(),
            skills: self.skills.len(),
            tensors,
            bytes_hashed: hashed,
            warnings: self.warnings.clone(),
        })
    }
}

/// Does the base skill manifest record the same sha256 for this task tensor?
fn task_same_in_base(base: &SkillManifest, name: &str, sha: Option<&str>) -> bool {
    base.tasks.iter().any(|t| {
        let i = t.i as usize;
        (name == task_mean_tensor(&base.id, i) && t.mean_sha256.as_deref() == sha)
            || (name == task_basis_tensor(&base.id, i) && t.basis_sha256.as_deref() == sha)
    })
}

// ------------------------------------------------------------------ writers

/// Result of a published file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteReport {
    pub path: PathBuf,
    /// sha256 of the whole file.
    pub sha256: String,
    pub bytes: u64,
    /// `model_sha` of the written file (overlay: sha256 of its manifest).
    pub model_sha: String,
    pub tensors: usize,
}

/// A tensor to write, borrowed from an open file or owned.
#[derive(Clone, Debug)]
struct OutTensor<'a> {
    name: String,
    dtype: TensorDtype,
    shape: Vec<usize>,
    data: Cow<'a, [u8]>,
}

impl<'a> OutTensor<'a> {
    fn borrowed(c: &'a CmfModel, e: &TensorEntry) -> Self {
        Self {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: Cow::Borrowed(c.entry_bytes(e)),
        }
    }

    fn u8(name: String, data: Vec<u8>) -> Self {
        Self {
            name,
            dtype: TensorDtype::U8,
            shape: vec![data.len()],
            data: Cow::Owned(data),
        }
    }

    fn f32(name: String, shape: Vec<usize>, v: &[f32]) -> Self {
        Self {
            name,
            dtype: TensorDtype::F32,
            shape,
            data: Cow::Owned(f32_bytes(v)),
        }
    }
}

fn find_in<'a>(model: &'a DecisionModel, name: &str) -> Result<OutTensor<'a>> {
    if let Some(o) = &model.overlay {
        if let Some(e) = o.tensor(name) {
            return Ok(OutTensor::borrowed(o, e));
        }
    }
    let e = model
        .base
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("tensor '{name}' is missing"))?;
    Ok(OutTensor::borrowed(&model.base, e))
}

/// The CMF header of a decision profile (spec §2.1).
pub fn decision_header(profile: &str, signal_dim: usize) -> Result<CmfHeader> {
    Ok(serde_json::from_value(json!({
        "format": "cmf",
        "version": 2,
        "quant_type": "F32",
        "arch": {
            "arch_name": profile,
            "hidden_size": signal_dim,
            "intermediate_size": 0,
            "num_layers": 0,
            "num_attention_heads": 0,
            "num_kv_heads": 0,
            "head_dim": 0,
            "vocab_size": 0,
            "layer_types": [],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 512
        },
        "provenance": {"tool": "cortiq decision", "version": env!("CARGO_PKG_VERSION")}
    }))?)
}

/// A new skill for [`FileBuilder::add_skill`]. The builder fills the derived
/// fields of the manifest: `representation_id`, every `mean_sha256` /
/// `basis_sha256`, the `rows` record (counts and sha256 from the blob) and the
/// `rows_learned` record.
#[derive(Clone, Debug)]
pub struct NewSkill {
    pub manifest: SkillManifest,
    /// One per task: the f32 topology (`basis` = `k × signal_dim`), or `None` for
    /// a task without one.
    pub topologies: Vec<Option<Topology>>,
    /// The `cortiq-decision-rows-v1` blob.
    pub rows: Vec<u8>,
    /// Learned rows kept in their own tensor.
    pub rows_learned: Option<Vec<u8>>,
}

struct SkillOut<'a> {
    manifest: SkillManifest,
    bytes: Vec<u8>,
    sha256: String,
    tensors: Vec<OutTensor<'a>>,
}

/// Build a full decision file (`init`, `train`, `add-skill`, `materialize`).
pub struct FileBuilder<'a> {
    model_id: String,
    name: String,
    representation: Value,
    representation_id: String,
    signal_dim: usize,
    dims: (usize, usize),
    generation: u64,
    created_unix: Option<u64>,
    encoder: Vec<OutTensor<'a>>,
    skills: Vec<SkillOut<'a>>,
}

impl std::fmt::Debug for FileBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileBuilder")
            .field("representation_id", &self.representation_id)
            .field(
                "skills",
                &self
                    .skills
                    .iter()
                    .map(|s| s.manifest.id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Check the topology of a task against its record and return its tensors.
fn task_tensors<'a>(
    id: &str,
    t: &mut TaskRecord,
    topo: Option<&Topology>,
    signal_dim: usize,
) -> Result<Vec<OutTensor<'a>>> {
    let i = t.i as usize;
    let Some(topo) = topo else {
        ensure!(
            t.k == 0,
            "skill '{id}' task {i}: k = {} without a topology",
            t.k
        );
        t.mean_sha256 = None;
        t.basis_sha256 = None;
        return Ok(Vec::new());
    };
    TaskView::new(&topo.mean, &topo.basis)
        .map_err(|e| anyhow::anyhow!("skill '{id}' task {i}: {e}"))?;
    ensure!(
        topo.mean.len() == signal_dim,
        "skill '{id}' task {i}: mean has {} values, the signal {signal_dim}",
        topo.mean.len()
    );
    ensure!(
        topo.basis.len() == t.k as usize * signal_dim,
        "skill '{id}' task {i}: basis rank {} differs from k = {}",
        topo.basis.len() / signal_dim,
        t.k
    );
    let mean = OutTensor::f32(task_mean_tensor(id, i), vec![signal_dim], &topo.mean);
    t.mean_sha256 = Some(sha256_hex(&mean.data));
    let mut out = vec![mean];
    if t.k > 0 {
        let basis = OutTensor::f32(
            task_basis_tensor(id, i),
            vec![t.k as usize, signal_dim],
            &topo.basis,
        );
        t.basis_sha256 = Some(sha256_hex(&basis.data));
        out.push(basis);
    } else {
        t.basis_sha256 = None;
    }
    Ok(out)
}

/// Decode a learned-rows blob and its record.
fn learned_rows_record(
    id: &str,
    bytes: &[u8],
    dims: (usize, usize),
    tasks: usize,
) -> Result<LearnedRowsRecord> {
    let r = Rows::decode(bytes).map_err(|e| anyhow::anyhow!("skill '{id}' learned rows: {e}"))?;
    ensure!(
        (r.dim_p, r.dim_h) == dims,
        "skill '{id}' learned rows: dims ({}, {}) differ from the signal {dims:?}",
        r.dim_p,
        r.dim_h
    );
    ensure!(
        r.rows.iter().all(|x| x.split == Split::Learned),
        "skill '{id}' learned rows: every row must be split learned"
    );
    ensure!(
        r.rows.iter().all(|x| (x.task as usize) < tasks),
        "skill '{id}' learned rows: task index out of range"
    );
    Ok(LearnedRowsRecord {
        tensor: rows_learned_tensor(id),
        layout: rows::LAYOUT.into(),
        n: r.rows.len() as u64,
        sha256: sha256_hex(bytes),
    })
}

impl<'a> FileBuilder<'a> {
    /// A new encoder-only file (`cortiq decision init`). `tensors` must be exactly
    /// the encoder layout of `encoder.config` (F32 weights, the `[8, dim]` golden
    /// and the U8 vocab). `encoder.tensors_sha256` is computed here; the vocab
    /// bytes must match `encoder.source.vocab_sha256`.
    pub fn new(
        model_id: impl Into<String>,
        name: impl Into<String>,
        mut encoder: EncoderRecord,
        tensors: Vec<TensorSpec>,
    ) -> Result<Self> {
        encoder.config.validate()?;
        let vocab_len = tensors
            .iter()
            .find(|t| t.name == VOCAB_TENSOR)
            .map(|t| t.data.len())
            .ok_or_else(|| anyhow::anyhow!("missing {VOCAB_TENSOR}"))?;
        let layout: BTreeMap<String, (TensorDtype, Vec<usize>)> = encoder
            .config
            .tensor_layout(vocab_len)
            .into_iter()
            .map(|(n, d, s)| (n, (d, s)))
            .collect();
        ensure!(
            tensors.len() == layout.len(),
            "{} encoder tensors given, the layout has {}",
            tensors.len(),
            layout.len()
        );
        let mut seen = BTreeSet::new();
        for t in &tensors {
            let Some((dtype, shape)) = layout.get(&t.name) else {
                bail!("'{}' is not an encoder tensor of this config", t.name);
            };
            ensure!(
                seen.insert(t.name.as_str()),
                "duplicate tensor '{}'",
                t.name
            );
            ensure!(
                t.dtype == *dtype && t.shape == *shape,
                "'{}': {:?}{:?}, the layout wants {:?}{:?}",
                t.name,
                t.dtype,
                t.shape,
                dtype,
                shape
            );
            if t.dtype == TensorDtype::F32 {
                ensure!(
                    t.data.len() == 4 * shape.iter().product::<usize>(),
                    "'{}': {} bytes for shape {:?}",
                    t.name,
                    t.data.len(),
                    shape
                );
                ensure!(
                    f32_view(&t.data).iter().all(|v| v.is_finite()),
                    "'{}' is not finite",
                    t.name
                );
            }
        }
        let vocab = &tensors
            .iter()
            .find(|t| t.name == VOCAB_TENSOR)
            .expect("vocab present")
            .data;
        ensure!(
            sha256_hex(vocab) == encoder.source.vocab_sha256,
            "the vocab bytes do not match encoder.source.vocab_sha256"
        );
        encoder.tensors_sha256 = manifest::tensors_sha256(tensors.iter().map(|t| TensorDigest {
            name: &t.name,
            dtype: t.dtype,
            shape: &t.shape,
            data: &t.data,
        }));
        let rep = Representation::new(encoder);
        rep.validate()?;
        let value = rep.to_value()?;
        let mut encoder_out: Vec<OutTensor<'a>> = tensors
            .into_iter()
            .map(|t| OutTensor {
                name: t.name,
                dtype: t.dtype,
                shape: t.shape,
                data: Cow::Owned(t.data),
            })
            .collect();
        encoder_out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            model_id: model_id.into(),
            name: name.into(),
            representation_id: manifest::representation_id(&value),
            representation: value,
            signal_dim: rep.signal_dim(),
            dims: (rep.encoder_dim(), rep.hashing_dim()),
            generation: 0,
            created_unix: None,
            encoder: encoder_out,
            skills: Vec::new(),
        })
    }

    /// The encoder of an open model, copied byte for byte, and no skills
    /// (`cortiq decision train --encoder FROM.cmf`).
    pub fn encoder_of(model: &'a DecisionModel) -> Self {
        let mut encoder: Vec<OutTensor<'a>> = model
            .base
            .tensors
            .iter()
            .filter(|e| e.name.starts_with(ENCODER_PREFIX))
            .map(|e| OutTensor::borrowed(&model.base, e))
            .collect();
        encoder.sort_by(|a, b| a.name.cmp(&b.name));
        let m = &model.manifest;
        Self {
            model_id: m.model_id.clone(),
            name: m.name.clone(),
            representation: m.representation.clone(),
            representation_id: m.representation_id.clone(),
            signal_dim: model.signal_dim(),
            dims: (model.encoder_dim(), model.hashing_dim()),
            generation: 0,
            created_unix: None,
            encoder,
            skills: Vec::new(),
        }
    }

    /// Everything of an open model: the encoder and every skill as served (with
    /// an overlay applied, this materialises the generation: its manifests and
    /// tensors, `generation` = the overlay's). Every byte is copied.
    pub fn from_model(model: &'a DecisionModel) -> Result<Self> {
        let mut b = Self::encoder_of(model);
        b.generation = model.generation();
        for s in &model.skills {
            let mut tensors = vec![];
            for (name, _) in skill_tensor_expectations(&s.manifest, model.signal_dim()) {
                tensors.push(find_in(model, &name)?);
            }
            b.skills.push(SkillOut {
                manifest: s.manifest.clone(),
                bytes: s.bytes.clone(),
                sha256: s.sha256.clone(),
                tensors,
            });
        }
        Ok(b)
    }

    pub fn set_model_id(&mut self, model_id: impl Into<String>) -> &mut Self {
        self.model_id = model_id.into();
        self
    }

    pub fn set_name(&mut self, name: impl Into<String>) -> &mut Self {
        self.name = name.into();
        self
    }

    /// `created_unix` of the new `decision.manifest` (default: `SOURCE_DATE_EPOCH`
    /// or 0 at write time).
    pub fn set_created_unix(&mut self, t: u64) -> &mut Self {
        self.created_unix = Some(t);
        self
    }

    pub fn set_generation(&mut self, g: u64) -> &mut Self {
        self.generation = g;
        self
    }

    pub fn representation_id(&self) -> &str {
        &self.representation_id
    }

    pub fn signal_dim(&self) -> usize {
        self.signal_dim
    }

    pub fn skill_ids(&self) -> Vec<&str> {
        self.skills.iter().map(|s| s.manifest.id.as_str()).collect()
    }

    /// Add a new skill (refused when the id exists); returns the sealed manifest.
    pub fn add_skill(&mut self, skill: NewSkill) -> Result<&SkillManifest> {
        let NewSkill {
            mut manifest,
            topologies,
            rows: rows_blob,
            rows_learned,
        } = skill;
        let id = manifest.id.clone();
        ensure!(manifest::valid_skill_id(&id), "invalid skill id '{id}'");
        ensure!(
            !self.skills.iter().any(|s| s.manifest.id == id),
            "skill '{id}' already exists in this file"
        );
        ensure!(
            topologies.len() == manifest.tasks.len(),
            "skill '{id}': {} topologies for {} tasks",
            topologies.len(),
            manifest.tasks.len()
        );
        manifest.representation_id = self.representation_id.clone();
        let mut tensors = Vec::new();
        for (t, topo) in manifest.tasks.iter_mut().zip(&topologies) {
            tensors.extend(task_tensors(&id, t, topo.as_ref(), self.signal_dim)?);
        }
        let r = Rows::decode(&rows_blob).map_err(|e| anyhow::anyhow!("skill '{id}' rows: {e}"))?;
        ensure!(
            (r.dim_p, r.dim_h) == self.dims,
            "skill '{id}' rows: dims ({}, {}) differ from the signal {:?}",
            r.dim_p,
            r.dim_h,
            self.dims
        );
        ensure!(
            r.rows
                .iter()
                .all(|x| (x.task as usize) < manifest.tasks.len()),
            "skill '{id}' rows: task index out of range"
        );
        manifest.rows = RowsRecord {
            tensor: rows_tensor(&id),
            layout: rows::LAYOUT.into(),
            n_train: r.count(Split::Train) as u64,
            n_calibration: r.count(Split::Calibration) as u64,
            n_learned: r.count(Split::Learned) as u64,
            sha256: sha256_hex(&rows_blob),
        };
        drop(r);
        tensors.push(OutTensor::u8(rows_tensor(&id), rows_blob));
        manifest.rows_learned = match rows_learned {
            Some(bytes) => {
                let rec = learned_rows_record(&id, &bytes, self.dims, manifest.tasks.len())?;
                tensors.push(OutTensor::u8(rows_learned_tensor(&id), bytes));
                Some(rec)
            }
            None => None,
        };
        manifest.validate(&self.representation_id, self.signal_dim)?;
        let bytes = canonical::vec_of(&manifest)?;
        let sha256 = sha256_hex(&bytes);
        self.skills.push(SkillOut {
            manifest,
            bytes,
            sha256,
            tensors,
        });
        Ok(&self.skills.last().expect("just pushed").manifest)
    }

    /// The `decision.manifest` this builder writes.
    pub fn manifest(&self) -> Result<DecisionManifest> {
        Ok(DecisionManifest {
            schema: MANIFEST_SCHEMA.into(),
            profile: BASE_PROFILE.into(),
            model_id: self.model_id.clone(),
            name: self.name.clone(),
            created_unix: match self.created_unix {
                Some(t) => t,
                None => manifest::created_unix_from_env()?,
            },
            representation: self.representation.clone(),
            representation_id: self.representation_id.clone(),
            skills: self
                .skills
                .iter()
                .map(|s| SkillRef {
                    id: s.manifest.id.clone(),
                    manifest_sha256: s.sha256.clone(),
                })
                .collect(),
            generation: self.generation,
        })
    }

    /// Write the file to `out` (never replacing an existing file) and re-open it.
    pub fn write(&self, out: impl AsRef<Path>) -> Result<WriteReport> {
        let manifest = self.manifest()?;
        manifest.validate()?;
        let mbytes = canonical::vec_of(&manifest)?;
        let header = decision_header(BASE_PROFILE, self.signal_dim)?;
        let mut specs: Vec<TensorSpecRef<'_>> = Vec::new();
        specs.push(TensorSpecRef {
            name: MANIFEST_TENSOR.into(),
            dtype: TensorDtype::U8,
            shape: vec![mbytes.len()],
            data: &mbytes,
        });
        for t in &self.encoder {
            specs.push(spec_ref(t));
        }
        for s in &self.skills {
            specs.push(TensorSpecRef {
                name: skill_manifest_tensor(&s.manifest.id),
                dtype: TensorDtype::U8,
                shape: vec![s.bytes.len()],
                data: &s.bytes,
            });
            for t in &s.tensors {
                specs.push(spec_ref(t));
            }
        }
        let n = specs.len();
        let (sha256, bytes, model_sha) =
            persist_noclobber(out.as_ref(), &header, &specs, |p, v| {
                DecisionModel::open(p, v)
                    .map(|m| m.model_sha().to_string())
                    .map_err(anyhow::Error::from)
            })?;
        Ok(WriteReport {
            path: out.as_ref().to_path_buf(),
            sha256,
            bytes,
            model_sha,
            tensors: n,
        })
    }
}

fn spec_ref<'b>(t: &'b OutTensor<'_>) -> TensorSpecRef<'b> {
    TensorSpecRef {
        name: t.name.clone(),
        dtype: t.dtype,
        shape: t.shape.clone(),
        data: &t.data,
    }
}

/// Materialise a model (base + overlay) into a self-contained file (spec §5.10).
pub fn materialize(model: &DecisionModel, out: impl AsRef<Path>) -> Result<WriteReport> {
    FileBuilder::from_model(model)?.write(out)
}

// ------------------------------------------------------------------ overlay writer

struct OverlaySkillOut<'a> {
    value: Value,
    tensors: Vec<OutTensor<'a>>,
}

/// Build a generation overlay (spec §5.10) on a loaded model (base plus, when
/// present, its current overlay). The new generation carries every change
/// relative to the base: the skills of the current overlay (manifests and
/// replaced tensors, copied) and the skills set here.
pub struct OverlayBuilder<'a> {
    model: &'a DecisionModel,
    generation: u64,
    parent: u64,
    created_unix: Option<u64>,
    events: Vec<OverlayEvent>,
    skills: BTreeMap<String, OverlaySkillOut<'a>>,
}

/// How [`OverlayBuilder::set_skill`] treats the learned rows.
#[derive(Clone, Debug)]
pub enum LearnedRows {
    /// Keep the record (and tensor) the model serves now.
    Keep,
    /// Replace them with this blob (all rows split learned).
    Replace(Vec<u8>),
}

impl<'a> OverlayBuilder<'a> {
    /// Start generation `generation` (> the model's) from `model`.
    pub fn new(model: &'a DecisionModel, generation: u64) -> Result<Self> {
        ensure!(
            generation > model.generation(),
            "generation {generation} is not newer than the model's {}",
            model.generation()
        );
        let mut skills = BTreeMap::new();
        let mut events = Vec::new();
        if let (Some(o), Some(om)) = (&model.overlay, &model.overlay_manifest) {
            events = om.events.clone();
            for (id, v) in &om.skills {
                let s = model.skill(id).expect("overlay skills are loaded");
                let mut tensors = Vec::new();
                for (name, _) in skill_tensor_expectations(&s.manifest, model.signal_dim()) {
                    if let Some(e) = o.tensor(&name) {
                        tensors.push(OutTensor::borrowed(o, e));
                    }
                }
                skills.insert(
                    id.clone(),
                    OverlaySkillOut {
                        value: v.clone(),
                        tensors,
                    },
                );
            }
        }
        Ok(Self {
            model,
            generation,
            parent: model.generation(),
            created_unix: None,
            events,
            skills,
        })
    }

    pub fn set_created_unix(&mut self, t: u64) -> &mut Self {
        self.created_unix = Some(t);
        self
    }

    /// Record an event of this generation.
    pub fn push_event(
        &mut self,
        skill: &str,
        label: &str,
        kind: &str,
        holdout: Value,
        gate: Value,
    ) -> &mut Self {
        self.events.push(OverlayEvent {
            generation: self.generation,
            skill: skill.into(),
            label: label.into(),
            kind: kind.into(),
            holdout,
            gate,
        });
        self
    }

    /// Set the manifest of a skill in this generation. Tasks in `topologies` get
    /// new tensors (`None` = no topology); every other task must keep the record
    /// (`k`, `mean_sha256`, `basis_sha256`) the model serves now. The build rows
    /// never change; `representation_id` and the rows record are filled here.
    /// Returns the sealed manifest.
    pub fn set_skill(
        &mut self,
        mut manifest: SkillManifest,
        topologies: BTreeMap<usize, Option<Topology>>,
        learned: LearnedRows,
    ) -> Result<SkillManifest> {
        let model = self.model;
        let id = manifest.id.clone();
        let current = model
            .skill(&id)
            .ok_or_else(|| anyhow::anyhow!("skill '{id}' is not in the model"))?;
        let dim = model.signal_dim();
        manifest.representation_id = model.representation_id().to_string();
        manifest.rows = current.manifest.rows.clone();
        if let Some(&i) = topologies.keys().find(|&&i| i >= manifest.tasks.len()) {
            bail!("skill '{id}': topology for task {i} outside the task table");
        }
        let mut tensors: Vec<OutTensor<'a>> = Vec::new();
        for t in manifest.tasks.iter_mut() {
            let i = t.i as usize;
            if let Some(topo) = topologies.get(&i) {
                tensors.extend(task_tensors(&id, t, topo.as_ref(), dim)?);
                continue;
            }
            let cur = current.manifest.tasks.get(i).ok_or_else(|| {
                anyhow::anyhow!("skill '{id}' task {i} is new and needs a topology entry")
            })?;
            ensure!(
                t.k == cur.k
                    && t.mean_sha256 == cur.mean_sha256
                    && t.basis_sha256 == cur.basis_sha256,
                "skill '{id}' task {i} changed its topology record without new tensors"
            );
            // Carry tensors the current overlay replaced (cumulative generation).
            if let Some(o) = &model.overlay {
                let mut carry = |name: String| {
                    if let Some(e) = o.tensor(&name) {
                        tensors.push(OutTensor::borrowed(o, e));
                    }
                };
                if t.mean_sha256.is_some() {
                    carry(task_mean_tensor(&id, i));
                }
                if t.basis_sha256.is_some() {
                    carry(task_basis_tensor(&id, i));
                }
            }
        }
        match learned {
            LearnedRows::Keep => {
                manifest.rows_learned = current.manifest.rows_learned.clone();
                if let (Some(r), Some(o)) = (&manifest.rows_learned, &model.overlay) {
                    if let Some(e) = o.tensor(&r.tensor) {
                        tensors.push(OutTensor::borrowed(o, e));
                    }
                }
            }
            LearnedRows::Replace(bytes) => {
                let rec = learned_rows_record(
                    &id,
                    &bytes,
                    (model.encoder_dim(), model.hashing_dim()),
                    manifest.tasks.len(),
                )?;
                tensors.push(OutTensor::u8(rows_learned_tensor(&id), bytes));
                manifest.rows_learned = Some(rec);
            }
        }
        manifest.validate(model.representation_id(), dim)?;
        let value = serde_json::to_value(&manifest)?;
        self.skills.insert(id, OverlaySkillOut { value, tensors });
        Ok(manifest)
    }

    /// The overlay manifest this builder writes.
    pub fn manifest(&self) -> Result<OverlayManifest> {
        Ok(OverlayManifest {
            schema: OVERLAY_SCHEMA.into(),
            generation: self.generation,
            base_model_sha: self.model.base_model_sha().into(),
            parent: self.parent,
            created_unix: match self.created_unix {
                Some(t) => t,
                None => manifest::created_unix_from_env()?,
            },
            events: self.events.clone(),
            skills: self
                .skills
                .iter()
                .map(|(k, s)| (k.clone(), s.value.clone()))
                .collect(),
        })
    }

    /// Write the generation to `out` (never replacing a file) and re-open it on
    /// the model's base file.
    pub fn write(&self, out: impl AsRef<Path>) -> Result<WriteReport> {
        let om = self.manifest()?;
        om.validate()?;
        let obytes = canonical::vec_of(&om)?;
        let header = decision_header(OVERLAY_PROFILE, self.model.signal_dim())?;
        let mut specs: Vec<TensorSpecRef<'_>> = vec![TensorSpecRef {
            name: OVERLAY_MANIFEST_TENSOR.into(),
            dtype: TensorDtype::U8,
            shape: vec![obytes.len()],
            data: &obytes,
        }];
        let mut names = BTreeSet::new();
        for s in self.skills.values() {
            for t in &s.tensors {
                if names.insert(t.name.clone()) {
                    specs.push(spec_ref(t));
                }
            }
        }
        let n = specs.len();
        let base = self.model.base_path().to_path_buf();
        let (sha256, bytes, model_sha) =
            persist_noclobber(out.as_ref(), &header, &specs, |p, v| {
                DecisionModel::open_parts(&base, Some(p), Verify::Light, v)
                    .map(|m| m.model_sha().to_string())
                    .map_err(anyhow::Error::from)
            })?;
        Ok(WriteReport {
            path: out.as_ref().to_path_buf(),
            sha256,
            bytes,
            model_sha,
            tensors: n,
        })
    }
}

// ------------------------------------------------------------------ no-clobber persistence

static TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Removes a path on drop unless disarmed.
struct TempGuard(Option<PathBuf>);

impl Drop for TempGuard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn file_sha256(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((format!("{:x}", h.finalize()), n))
}

fn sync_dir(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Temp file in the output's directory → fsync → `check(temp, Full)` → no-clobber
/// publish → `check(out, Light)`. Returns (file sha256, length, model sha).
fn persist_noclobber(
    out: &Path,
    header: &CmfHeader,
    specs: &[TensorSpecRef<'_>],
    check: impl Fn(&Path, Verify) -> Result<String>,
) -> Result<(String, u64, String)> {
    if std::fs::symlink_metadata(out).is_ok() {
        return Err(OutputExists(out.to_path_buf()).into());
    }
    let file_name = out
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output {} has no file name", out.display()))?
        .to_string_lossy()
        .into_owned();
    let dir = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    ensure!(
        dir.is_dir(),
        "output directory {} does not exist",
        dir.display()
    );
    // Reserve a unique temp name.
    let mut guard = TempGuard(None);
    let tmp = loop {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let k = TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = dir.join(format!(
            ".{file_name}.{}.{nanos}.{k}.tmp",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&p)
        {
            Ok(_) => break p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    };
    guard.0 = Some(tmp.clone());
    CmfModel::write_ref(&tmp, header, specs, None, None)?;
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&tmp)?
        .sync_all()?;
    check(&tmp, Verify::Full)
        .map_err(|e| anyhow::anyhow!("the written file fails its own checks: {e}"))?;
    let (sha, len) = file_sha256(&tmp)?;
    match std::fs::hard_link(&tmp, out) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(OutputExists(out.to_path_buf()).into());
        }
        Err(_) => {
            // No hard links here (some network or FAT file systems): copy into a
            // file that must not exist yet.
            let mut dst = match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out)
            {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(OutputExists(out.to_path_buf()).into());
                }
                Err(e) => return Err(e.into()),
            };
            let mut out_guard = TempGuard(Some(out.to_path_buf()));
            let mut src = std::fs::File::open(&tmp)?;
            std::io::copy(&mut src, &mut dst)?;
            dst.flush()?;
            dst.sync_all()?;
            out_guard.0 = None;
        }
    }
    drop(guard);
    sync_dir(&dir);
    let (sha_out, len_out) = file_sha256(out)?;
    ensure!(
        sha_out == sha && len_out == len,
        "{} changed while it was published",
        out.display()
    );
    let model_sha = check(out, Verify::Light)?;
    Ok((sha, len, model_sha))
}
