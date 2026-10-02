//! Overlay generations and rollback (spec §5.10).
//!
//! A promotion writes `generations/gNNNNNN.cmf` (profile
//! `cortiq-decision-overlay-v1`, the DECISION bit): the overlay manifest (every
//! event and the full manifest of every skill that differs from the base since
//! generation 0), the replaced task tensors and the skills' `rows.learned`. The
//! base file is never rewritten.
//!
//! * [`publish`]: write the generation (temp file, fsync, strict re-open,
//!   no-clobber publish — [`crate::container::OverlayBuilder::write`]), open it on
//!   the base with the overlay loader ([`Verify::Full`] for the overlay), then
//!   replace `CURRENT` atomically with `gNNNNNN <sha256 of the file>`;
//! * [`open_served`]: base + the overlay `CURRENT` names (the file's sha256 must
//!   match); no `CURRENT` or generation 0 serves the base (`CURRENT` of generation
//!   0 holds the base `model_sha`);
//! * [`rollback`]: serve generation N (0 = the base) by rewriting `CURRENT`; the
//!   generation files stay; a later promotion takes the next free number and
//!   records N as its parent;
//! * [`rollback_state`]: the same without a running server (`cortiq decision
//!   rollback --state DIR --to N`), which also appends the rollback to
//!   `learn.log` (the learning counters restart);
//! * [`list`]: the generations on disk with their overlay manifests;
//! * materialising a served model into a self-contained file is
//!   [`crate::container::materialize`].

use crate::buffer::{LearnLog, LogRecord};
use crate::container::{DecisionModel, OverlayBuilder, Verify, WriteReport};
use crate::manifest::{self, OVERLAY_MANIFEST_TENSOR, OverlayManifest};
use crate::statedir::{Current, StateDir, generation_name};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// sha256 of a file's bytes (lowercase hex).
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// The model `CURRENT` names: base + overlay (see the module notes).
pub fn open_served(base: &Path, state: &StateDir, verify: Verify) -> Result<DecisionModel> {
    match state.read_current()? {
        None => Ok(DecisionModel::open(base, verify)?),
        Some(c) if c.generation == 0 => {
            let m = DecisionModel::open(base, verify)?;
            ensure!(
                m.base_model_sha() == c.sha256,
                "CURRENT names generation 0 of model {}, but {} is model {}",
                c.sha256,
                base.display(),
                m.base_model_sha()
            );
            Ok(m)
        }
        Some(c) => open_generation(base, state, &c, verify),
    }
}

fn open_generation(
    base: &Path,
    state: &StateDir,
    c: &Current,
    verify: Verify,
) -> Result<DecisionModel> {
    let path = state.generation_path(c.generation);
    let sha = file_sha256(&path)?;
    ensure!(
        sha == c.sha256,
        "{} has sha256 {sha}, CURRENT expects {}",
        path.display(),
        c.sha256
    );
    let m = DecisionModel::open_parts(base, Some(&path), verify, Verify::Full)?;
    ensure!(
        m.generation() == c.generation,
        "{} holds generation {}, not {}",
        path.display(),
        m.generation(),
        c.generation
    );
    Ok(m)
}

/// The number of the next generation: one past every generation on disk and
/// the served one.
pub fn next_generation(state: &StateDir, served: u64) -> Result<u64> {
    let max = state
        .generations()?
        .last()
        .map_or(0, |(g, _)| *g)
        .max(served);
    Ok(max + 1)
}

/// A published generation.
#[derive(Debug)]
pub struct Published {
    pub generation: u64,
    pub report: WriteReport,
    /// Base + the new overlay, opened with the overlay loader.
    pub model: DecisionModel,
}

/// Write a generation of the model whose base file is `base`, open it with the
/// overlay loader, run `check` on the opened model and only then make it
/// `CURRENT` (see the module notes). When the open or the check fails, the file
/// (never served) is removed and `CURRENT` is untouched.
pub fn publish(
    state: &StateDir,
    base: &Path,
    builder: &OverlayBuilder<'_>,
    generation: u64,
    check: impl FnOnce(&DecisionModel) -> Result<()>,
) -> Result<Published> {
    let om = builder.manifest()?;
    ensure!(
        om.generation == generation,
        "the builder writes generation {}, not {generation}",
        om.generation
    );
    let path = state.generation_path(generation);
    let report = builder.write(&path)?;
    let opened = DecisionModel::open_parts(base, Some(&path), Verify::Light, Verify::Full)
        .map_err(|e| anyhow::anyhow!("re-open {}: {e}", path.display()))
        .and_then(|m| check(&m).map(|()| m));
    let model = match opened {
        Ok(m) => m,
        Err(e) => {
            if let Err(r) = std::fs::remove_file(&path) {
                tracing::error!(error = %r, path = %path.display(), "could not remove a refused generation");
            }
            return Err(e);
        }
    };
    state.write_current(&Current {
        generation,
        sha256: report.sha256.clone(),
    })?;
    Ok(Published {
        generation,
        report,
        model,
    })
}

/// Serve generation `to` (0 = the base) from now on: `CURRENT` is replaced and
/// the model returned. The generation must exist and belong to this base.
pub fn rollback(state: &StateDir, base: &Path, to: u64, verify: Verify) -> Result<DecisionModel> {
    let (model, current) = if to == 0 {
        let m = DecisionModel::open(base, verify)?;
        let sha = m.base_model_sha().to_string();
        (
            m,
            Current {
                generation: 0,
                sha256: sha,
            },
        )
    } else {
        let path = state.generation_path(to);
        ensure!(
            path.exists(),
            "generation {to} does not exist ({})",
            path.display()
        );
        let c = Current {
            generation: to,
            sha256: file_sha256(&path)?,
        };
        (open_generation(base, state, &c, verify)?, c)
    };
    state.write_current(&current)?;
    Ok(model)
}

/// The overlay manifest of a generation file (read without its base).
pub fn overlay_manifest_of(path: &Path) -> Result<OverlayManifest> {
    let c = cortiq_core::CmfModel::open(path)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let bytes = c
        .tensor_bytes(OVERLAY_MANIFEST_TENSOR)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let (om, _): (OverlayManifest, Value) =
        manifest::parse_manifest_bytes("decision.overlay.manifest", bytes)?;
    om.validate()?;
    Ok(om)
}

/// `cortiq decision rollback --state DIR --to N` without a server: takes the
/// state `LOCK`, replaces `CURRENT` and appends the rollback to `learn.log`.
/// Generation 0 needs the base `model_sha`: from `base` when given, else from
/// the newest generation's overlay manifest.
pub fn rollback_state(state: &StateDir, to: u64, base: Option<&Path>) -> Result<Current> {
    let _lock = state.lock(false)?;
    let current = if to == 0 {
        let sha = match base {
            Some(b) => DecisionModel::open(b, Verify::Light)?
                .base_model_sha()
                .to_string(),
            None => {
                let gens = state.generations()?;
                let Some((_, newest)) = gens.last() else {
                    bail!(
                        "no generation in {} and no base file to name generation 0",
                        state.root().display()
                    );
                };
                overlay_manifest_of(newest)?.base_model_sha
            }
        };
        Current {
            generation: 0,
            sha256: sha,
        }
    } else {
        let path = state.generation_path(to);
        ensure!(
            path.exists(),
            "generation {to} does not exist ({})",
            path.display()
        );
        let om = overlay_manifest_of(&path)?;
        ensure!(
            om.generation == to,
            "{} holds generation {}",
            path.display(),
            om.generation
        );
        if let Some(b) = base {
            let m = DecisionModel::open_parts(b, Some(&path), Verify::Light, Verify::Full)?;
            ensure!(m.generation() == to, "generation mismatch");
        }
        Current {
            generation: to,
            sha256: file_sha256(&path)?,
        }
    };
    state.write_current(&current)?;
    let (log, _) = LearnLog::open(&state.learn_log_path())?;
    log.append(&LogRecord::Rollback { generation: to })?;
    Ok(current)
}

/// One generation on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationInfo {
    pub generation: u64,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub parent: u64,
    pub created_unix: u64,
    pub events: Vec<manifest::OverlayEvent>,
    /// Every skill the overlay carries (replaced or born in a generation).
    pub skills: Vec<String>,
    /// The auto-skills among them: skills the generation carries entirely,
    /// which the base file does not have (0.8.6).
    pub auto_skills: Vec<String>,
    /// `CURRENT` names it.
    pub current: bool,
}

impl GenerationInfo {
    pub fn to_json(&self) -> Value {
        json!({
            "generation": self.generation,
            "name": generation_name(self.generation),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "parent": self.parent,
            "created_unix": self.created_unix,
            "skills": self.skills,
            "auto_skills": self.auto_skills,
            "events": self.events,
            "current": self.current,
        })
    }
}

/// The generations on disk, ascending.
pub fn list(state: &StateDir) -> Result<Vec<GenerationInfo>> {
    let current = state.read_current()?.map(|c| c.generation);
    let mut out = Vec::new();
    for (g, path) in state.generations()? {
        let om = overlay_manifest_of(&path)?;
        // Without the base at hand "auto" is read off the manifest (the loader
        // refuses an overlay-born skill that is not an auto-skill, so the two
        // agree for a generation that loads).
        let auto_skills = om
            .skills
            .iter()
            .filter(|(_, v)| {
                manifest::from_value::<manifest::SkillManifest>("skill manifest", v)
                    .is_ok_and(|m| m.is_auto())
            })
            .map(|(id, _)| id.clone())
            .collect();
        out.push(GenerationInfo {
            generation: g,
            sha256: file_sha256(&path)?,
            bytes: std::fs::metadata(&path)?.len(),
            parent: om.parent,
            created_unix: om.created_unix,
            events: om.events.clone(),
            skills: om.skills.keys().cloned().collect(),
            auto_skills,
            current: current == Some(g),
            path,
        });
    }
    Ok(out)
}
