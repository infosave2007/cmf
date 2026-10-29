//! Calibration-driven physical compaction of a dense MTP draft FFN.
//!
//! This is deliberately a small, auditable operation rather than a hidden
//! branch of trunk FFN compaction: an MTP draft has its own activation
//! distribution and may have a width independent of the backbone.  The
//! command therefore obtains statistics through `Pipeline::probe_mtp_ffn`,
//! writes an explicit v2 tube plan with `trunk.mode = preserve`, then hands
//! the physical gather to the common CMF writer.

use anyhow::{Context, Result, ensure};
use cortiq_core::quant::{
    Q2TP_CHUNK, Q4TP_NIB, bf16_to_f32, dequant_tensor, expected_nbytes, f16_to_f32,
    q2tp_ladder, q2tp_sections, q4tp_code, q4tp_ladder, q4tp_sections,
};
use cortiq_core::{CmfModel, TensorDtype, mtp_sidecar_path};
use cortiq_engine::{MtpFfnProbe, Pipeline, SamplerConfig, tokenizer::Tokenizer};
use serde_json::{Map, Value, json};
use std::cmp::Ordering;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use crate::tdata::{TdataReader, TdataSplitSelector};
use crate::tube::paths_collide;

/// The physical codecs used by CMF dense FFNs have 32-wide groups.  Keeping
/// the draft plan on that boundary prevents a supposedly compact q4tp/q8_2f
/// MTP matrix from falling back to a less suitable layout in the writer.
const MTP_COMPACT_ALIGNMENT: usize = 32;

/// Admission validates the newly written runtime with a representative
/// teacher-forced prefix, not by retaining a potentially enormous first
/// calibration record a second time. It is raised automatically for unusually
/// deep draft stacks so every layer still sees positions.
const MTP_ADMISSION_TARGET_TOKENS: usize = 256;

/// Candidate admission must exercise ordinary autoregressive generation as
/// well as the teacher-forced draft probe. Keep it deliberately tiny: this is
/// an integrity gate for a conversion job, not a second calibration pass.
const MTP_PLAIN_ADMISSION_PROMPT_TOKENS: usize = 16;
const MTP_PLAIN_ADMISSION_OUTPUT_TOKENS: usize = 2;

/// A private staging name is always created next to its final destination, so
/// publishing through a hard link never crosses a filesystem boundary and can
/// never replace an existing user artifact.
static STAGE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Final paths of one `mtp-compact` invocation. The preferred external-MiMo
/// form writes a runnable main/draft pair. `publish_main = false` is the
/// preserved legacy sidecar-only form: it still stages and admits a temporary
/// pair, but publishes only the explicitly requested `.mtp.cmf` draft.
#[derive(Debug)]
struct CompactDestination {
    main: PathBuf,
    draft: PathBuf,
    external_draft: bool,
    publish_main: bool,
}

fn has_cmf_suffix(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()) == Some("cmf")
}

fn is_mtp_sidecar(path: &Path) -> bool {
    mtp_sidecar_path(path) == path
}

/// Inverse of [`mtp_sidecar_path`] for the legacy sidecar-only output form.
/// The returned main path is only used for a private staged runtime-admission
/// pair; it is not created or published by that legacy form.
fn main_path_for_mtp_sidecar(sidecar: &Path) -> Result<PathBuf> {
    let name = sidecar
        .file_name()
        .and_then(|name| name.to_str())
        .context("MTP compact: legacy sidecar output has no UTF-8 filename")?;
    let stem = name.strip_suffix(".mtp.cmf").ok_or_else(|| {
        anyhow::anyhow!(
            "MTP compact: legacy sidecar output {} must end in .mtp.cmf",
            sidecar.display()
        )
    })?;
    ensure!(
        !stem.is_empty(),
        "MTP compact: legacy sidecar output {} has an empty main-model stem",
        sidecar.display()
    );
    Ok(sidecar.with_file_name(format!("{stem}.cmf")))
}

/// `Path::parent()` returns an empty path for a bare filename. Treat that as
/// the current directory everywhere an on-disk parent is required.
fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| if path.is_absolute() { path } else { Path::new(".") })
}

/// Resolve as much of a prospective path as exists, then normalize the
/// not-yet-created suffix lexically.  This deliberately supplements
/// `paths_collide`: that helper is authoritative for existing aliases and
/// hard links, while this one catches a typo such as
/// `--plan-out candidate.cmf/plan.json` before `create_dir_all` can turn the
/// intended CMF leaf into a directory halfway through a long conversion.
///
/// The suffix starts after the nearest *canonicalizable* ancestor. A dangling
/// or concurrently changed component is still rejected later by the
/// no-clobber preparation/publication path; this helper is only an early
/// user-error guard, not a replacement for that race-safe fence.
fn prospective_path_key(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("MTP compact: reading current directory for output path")?
            .join(path)
    };
    let mut ancestor = absolute.as_path();
    let canonical_ancestor = loop {
        match fs::canonicalize(ancestor) {
            Ok(canonical) => break canonical,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let next = ancestor.parent().ok_or_else(|| {
                    anyhow::anyhow!(
                        "MTP compact: no existing ancestor while normalizing {}",
                        path.display()
                    )
                })?;
                if next == ancestor {
                    anyhow::bail!(
                        "MTP compact: no existing ancestor while normalizing {}",
                        path.display()
                    );
                }
                ancestor = next;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "MTP compact: resolving existing ancestor {} for {}",
                        ancestor.display(),
                        path.display()
                    )
                });
            }
        }
    };
    let suffix = absolute.strip_prefix(ancestor).with_context(|| {
        format!(
            "MTP compact: deriving prospective path suffix {} from {}",
            absolute.display(),
            ancestor.display()
        )
    })?;
    let mut key = canonical_ancestor;
    for component in suffix.components() {
        use std::path::Component;
        match component {
            Component::CurDir => {}
            Component::Normal(name) => key.push(name),
            Component::ParentDir => {
                if !key.pop() {
                    anyhow::bail!(
                        "MTP compact: path {} escapes filesystem root while normalizing",
                        path.display()
                    );
                }
            }
            Component::RootDir | Component::Prefix(_) => anyhow::bail!(
                "MTP compact: unexpected absolute component while normalizing {}",
                path.display()
            ),
        }
    }
    Ok(key)
}

/// Reject both aliases and parent/child relationships between paths that
/// must each become ordinary files.  Without the parent/child check, a
/// `--plan-out` below an absent candidate CMF can create a directory at the
/// candidate's final filename, wasting the entire calibration/bake before
/// publication fails.
fn paths_overlap_or_nested(a: &Path, b: &Path) -> Result<bool> {
    if paths_collide(a, b) {
        return Ok(true);
    }
    let a_key = prospective_path_key(a)?;
    let b_key = prospective_path_key(b)?;
    Ok(a_key == b_key || a_key.starts_with(&b_key) || b_key.starts_with(&a_key))
}

fn compact_destination(
    main_path: &Path,
    draft_path: &Path,
    output: &Path,
) -> Result<CompactDestination> {
    ensure!(
        has_cmf_suffix(output),
        "MTP compact: --output must name a new .cmf candidate"
    );
    let external_draft = !paths_collide(main_path, draft_path);
    if is_mtp_sidecar(output) {
        ensure!(
            external_draft,
            "MTP compact: --output ending in .mtp.cmf is a legacy sidecar-only form and is valid only when MODEL has an external MTP sidecar"
        );
        return Ok(CompactDestination {
            main: main_path_for_mtp_sidecar(output)?,
            draft: output.to_path_buf(),
            external_draft: true,
            publish_main: false,
        });
    }
    if !external_draft {
        return Ok(CompactDestination {
            main: output.to_path_buf(),
            draft: output.to_path_buf(),
            external_draft: false,
            publish_main: true,
        });
    }
    let draft = mtp_sidecar_path(output);
    Ok(CompactDestination {
        main: output.to_path_buf(),
        draft,
        external_draft: true,
        publish_main: true,
    })
}

/// `mtp-compact` produces a runnable main CMF (and, for MiMo, its derived
/// sibling sidecar). A filename ending in `.mtp.cmf` has the shared sidecar
/// identity and must never be accepted as the positional main-model input:
/// doing so could otherwise spend time preparing a partial draft before the
/// loader eventually reports its missing backbone.
fn require_main_model_input(path: &Path) -> Result<()> {
    ensure!(
        mtp_sidecar_path(path) != path,
        "MTP compact: MODEL must name the main .cmf, not an .mtp.cmf sidecar; pass its companion main CMF"
    );
    Ok(())
}

/// Check a requested final leaf without creating its parent.  This is only an
/// early usability fence: `prepare_new_file` and atomic publication repeat
/// the no-clobber check later, because another process can create the name
/// after this return.
fn check_new_file_leaf(path: &Path, label: &str) -> Result<()> {
    path.file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("MTP compact: {label} {} names no file", path.display())
        })?;
    match fs::symlink_metadata(path) {
        Ok(_) => anyhow::bail!(
            "MTP compact: refusing to overwrite existing {label} {}",
            path.display()
        ),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("MTP compact: checking {label} {}", path.display())),
    }
}

/// Reject output/plan mistakes before an expensive source hash scan.  The
/// external sidecar is not known until the source has been opened, so its
/// topology remains checked in the second fence below.
fn preflight_explicit_paths(
    source_main: &Path,
    output: &Path,
    plan_path: &Path,
) -> Result<()> {
    ensure!(
        has_cmf_suffix(output),
        "MTP compact: --output must name a new .cmf candidate"
    );
    ensure!(
        !paths_overlap_or_nested(output, source_main)?
            && !paths_overlap_or_nested(plan_path, source_main)?
            && !paths_overlap_or_nested(plan_path, output)?,
        "MTP compact: candidate output and --plan-out must be distinct ordinary-file paths, not aliases or nested below source/candidate paths"
    );
    check_new_file_leaf(output, "candidate output")?;
    check_new_file_leaf(plan_path, "MTP plan")?;
    Ok(())
}

/// Create parent directories and refuse every pre-existing leaf, including a
/// dangling symlink. `CmfModel::write` ultimately uses `File::create`, so
/// this guard must run before expensive calibration and before any writer.
fn prepare_new_file(path: &Path, label: &str) -> Result<()> {
    check_new_file_leaf(path, label)?;
    let parent = parent_dir(path);
    // `sync_parent_dir(output)` only persists a leaf inside an existing
    // directory. If `create_dir_all` makes nested output parents, persist the
    // hierarchy from the nearest existing ancestor outward as well.
    let mut created_parents = Vec::new();
    let mut ancestor = parent;
    loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                created_parents.push(ancestor.to_path_buf());
                let next = parent_dir(ancestor);
                ensure!(
                    next != ancestor,
                    "MTP compact: no existing ancestor while creating parent {} for {label}",
                    parent.display()
                );
                ancestor = next;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "MTP compact: checking parent ancestor {} for {label}",
                        ancestor.display()
                    )
                });
            }
        }
    }
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "MTP compact: creating parent for {label} {}",
            path.display()
        )
    })?;
    for created in created_parents.iter().rev() {
        sync_parent_dir(created, "creating output parent directory")?;
    }
    ensure!(
        fs::metadata(parent)
            .with_context(|| format!("MTP compact: stat parent {}", parent.display()))?
            .is_dir(),
        "MTP compact: parent {} for {label} is not a directory",
        parent.display()
    );
    // Re-check after creating parents: an actor may have claimed the leaf
    // between the inexpensive early inspection and this reservation fence.
    check_new_file_leaf(path, label)
}

/// Persist the directory entry for a newly created or hard-linked file. A
/// successful `sync_data` only makes the file payload durable; it does not
/// necessarily make the name visible after an abrupt host loss. CMF
/// conversion can run for hours, so publication deliberately pays this small
/// metadata cost instead of reporting a result that can disappear after a
/// crash.
fn sync_parent_dir(path: &Path, what: &str) -> Result<()> {
    #[cfg(unix)]
    {
        let parent = parent_dir(path);
        fs::File::open(parent)
            .with_context(|| {
                format!(
                    "MTP compact: opening parent directory {} to sync {what}",
                    parent.display()
                )
            })?
            .sync_all()
            .with_context(|| {
                format!(
                    "MTP compact: syncing parent directory {} after {what}",
                    parent.display()
                )
            })?;
    }
    #[cfg(not(unix))]
    let _ = (path, what);
    Ok(())
}

/// Create an owned sibling staging directory. Keeping all transient files in
/// it avoids the reserve/reopen race: every writer can use `create_new`, and
/// failure cleanup removes only a directory created by this invocation.
fn create_stage_dir(output: &Path, kind: &str) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "MTP compact: staging output {} has no UTF-8 name",
                output.display()
            )
        })?;
    let parent = parent_dir(output);
    for _ in 0..64 {
        let nonce = STAGE_SEQUENCE
            .fetch_add(1, AtomicOrdering::Relaxed)
            .wrapping_add(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|duration| duration.as_nanos() as u64)
                    .unwrap_or(0),
            );
        let staged = parent.join(format!(
            ".{name}.mtp-compact-{kind}-{}-{nonce:016x}.stage",
            std::process::id()
        ));
        match fs::create_dir(&staged) {
            Ok(()) => {
                if let Err(error) = sync_parent_dir(&staged, "creating private stage") {
                    let _ = fs::remove_dir(&staged);
                    return Err(error);
                }
                return Ok(staged);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("MTP compact: creating private stage {}", staged.display())
                });
            }
        }
    }
    anyhow::bail!(
        "MTP compact: could not create a private stage beside {}",
        output.display()
    )
}

fn staged_child(stage_dir: &Path, output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow::anyhow!("MTP compact: {} names no file", output.display()))?;
    Ok(stage_dir.join(name))
}

/// Publish without replacing an existing result. Staging and destination are
/// siblings, so hard-link publication is atomic and preserves a failed stage
/// for inspection rather than falling back to overwrite-capable `rename`.
fn publish_new_file(staged: &Path, output: &Path, label: &str) -> Result<()> {
    match fs::hard_link(staged, output) {
        Ok(()) => {
            sync_parent_dir(output, &format!("publishing {label}"))?;
            if let Err(error) = fs::remove_file(staged) {
                eprintln!(
                    "warning: MTP compact published {label} {}, but could not remove private stage {}: {error}",
                    output.display(),
                    staged.display()
                );
            }
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => anyhow::bail!(
            "MTP compact: refusing to overwrite {label} {}; staged validated artifact remains at {}",
            output.display(),
            staged.display()
        ),
        Err(error) => Err(error).with_context(|| {
            format!(
                "MTP compact: atomically publishing {label} {}; staged validated artifact remains at {}",
                output.display(),
                staged.display()
            )
        }),
    }
}

fn stage_backbone(source: &Path, staged: &Path) -> Result<&'static str> {
    // `hard_link` of a symlink aliases the symlink itself on Unix. Resolve it
    // first so the staged candidate is a stable regular-file alias of the
    // verified CMF bytes, not a future-following link to an arbitrary target.
    let source = fs::canonicalize(source)
        .with_context(|| format!("MTP compact: resolving source backbone {}", source.display()))?;
    match fs::hard_link(&source, staged) {
        Ok(()) => {
            sync_parent_dir(staged, "staging unchanged backbone")?;
            Ok("hard-link")
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => anyhow::bail!(
            "MTP compact: private backbone stage already exists: {}",
            staged.display()
        ),
        Err(link_error) => {
            let mut input = fs::File::open(&source).with_context(|| {
                format!("MTP compact: opening source backbone {}", source.display())
            })?;
            let expected = input
                .metadata()
                .with_context(|| format!("MTP compact: stat source backbone {}", source.display()))?
                .len();
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(staged)
                .with_context(|| {
                    format!(
                        "MTP compact: creating private backbone stage {} after hard-link failure ({link_error})",
                        staged.display()
                    )
                })?;
            let copied = io::copy(&mut input, &mut output).with_context(|| {
                format!(
                    "MTP compact: copying {} to private stage {}",
                    source.display(),
                    staged.display()
                )
            })?;
            output.sync_data().with_context(|| {
                format!("MTP compact: syncing private backbone stage {}", staged.display())
            })?;
            ensure!(
                copied == expected,
                "MTP compact: incomplete backbone copy {} -> {} ({copied} != {expected} bytes)",
                source.display(),
                staged.display()
            );
            sync_parent_dir(staged, "staging copied backbone")?;
            Ok("copy")
        }
    }
}

fn write_new_file(path: &Path, data: &[u8], label: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("MTP compact: creating staged {label} {}", path.display()))?;
    file.write_all(data)
        .with_context(|| format!("MTP compact: writing staged {label} {}", path.display()))?;
    file.sync_data()
        .with_context(|| format!("MTP compact: syncing staged {label} {}", path.display()))?;
    sync_parent_dir(path, &format!("writing staged {label}"))?;
    Ok(())
}

fn verify_open_cmf(model: &CmfModel, path: &Path, label: &str) -> Result<()> {
    // A full hash scan of a 100B-class candidate can take appreciable wall
    // time. Announce both ends of the integrity fence so a deliberately safe
    // conversion does not look stalled while it is reading immutable bytes.
    let started = Instant::now();
    eprintln!("mtp compact: verifying {label} {}", path.display());
    let integrity = model.verify();
    ensure!(
        integrity.is_empty(),
        "MTP compact: {label} integrity failed: {}",
        integrity
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ")
    );
    eprintln!(
        "mtp compact: verified {label} in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn verify_cmf(path: &Path, label: &str) -> Result<Arc<CmfModel>> {
    let model = Arc::new(
        CmfModel::open_sharded(path)
            .with_context(|| format!("MTP compact: reopening {label} {}", path.display()))?,
    );
    verify_open_cmf(&model, path, label)?;
    Ok(model)
}

/// Teacher data stores token ids, so matching tensor geometry alone is not a
/// sufficient calibration contract. A v1 dataset generated by CMF carries
/// the exact tokenizer bytes and tokenizer-bundle hashes; refuse to score
/// those ids against a different vocabulary. Older generic `.tdata` files
/// are still readable, but make their missing contract explicit in the plan
/// and on stderr rather than pretending they were bound to this model.
fn validate_tdata_tokenizer_contract(
    manifest: &Value,
    vocab: Option<&[u8]>,
    tokenizer_config_bytes: &[u8],
    source_vocab_size: usize,
) -> Result<&'static str> {
    let Some(tokenizer) = manifest.get("tokenizer") else {
        eprintln!(
            "mtp compact: warning: calibration .tdata has no tokenizer contract; token-id compatibility is not cryptographically bound to the source model"
        );
        return Ok("absent-legacy");
    };
    let tokenizer = tokenizer
        .as_object()
        .context("MTP compact: .tdata manifest.tokenizer must be an object")?;
    let expected_vocab_size = tokenizer
        .get("vocab_size")
        .and_then(Value::as_u64)
        .context("MTP compact: .tdata manifest.tokenizer.vocab_size is required")?;
    let expected_vocab_size = usize::try_from(expected_vocab_size)
        .context("MTP compact: .tdata tokenizer vocab_size does not fit this platform")?;
    ensure!(
        expected_vocab_size == source_vocab_size,
        "MTP compact: .tdata tokenizer vocab_size {expected_vocab_size} differs from source tokenizer {source_vocab_size}"
    );
    let expected_vocab_hash = tokenizer
        .get("vocab_sha256")
        .and_then(Value::as_str)
        .context("MTP compact: .tdata manifest.tokenizer.vocab_sha256 is required")?;
    let source_vocab = vocab.context(
        "MTP compact: .tdata has a tokenizer contract but source main CMF has no embedded tokenizer bytes",
    )?;
    let source_vocab_hash = crate::knowledge::sha256_hex(source_vocab);
    ensure!(
        expected_vocab_hash == source_vocab_hash,
        "MTP compact: .tdata tokenizer vocab_sha256 differs from source main tokenizer"
    );
    let expected_config_hash = tokenizer
        .get("config_sha256")
        .and_then(Value::as_str)
        .context("MTP compact: .tdata manifest.tokenizer.config_sha256 is required")?;
    let source_config_hash = crate::knowledge::sha256_hex(tokenizer_config_bytes);
    ensure!(
        expected_config_hash == source_config_hash,
        "MTP compact: .tdata tokenizer config_sha256 differs from source main tokenizer metadata"
    );
    Ok("verified-v1")
}

/// Reopen both source artifacts at an identity fence and require the exact
/// directory identities used for scoring. This does not turn a mutable source
/// into a transaction; callers must keep source files immutable until finish,
/// but it prevents baking a plainly changed or swapped source by accident.
fn revalidate_source_identity(
    phase: &str,
    main_path: &Path,
    draft_path: &Path,
    expected_main_hash: u64,
    expected_draft_hash: u64,
    expected_main_trunk_hash: u64,
    expected_draft_trunk_hash: u64,
) -> Result<Arc<CmfModel>> {
    let main_label = format!("source main {phase}");
    let main = verify_cmf(main_path, &main_label)?;
    ensure!(
        main.dir_hash() == expected_main_hash,
        "MTP compact: source main changed {phase} (expected directory hash {expected_main_hash:016x}, got {:016x})",
        main.dir_hash()
    );
    ensure!(
        main.trunk_hash() == expected_main_trunk_hash,
        "MTP compact: source main tensors changed {phase} (expected trunk hash {expected_main_trunk_hash:016x}, got {:016x})",
        main.trunk_hash()
    );
    if paths_collide(main_path, draft_path) {
        ensure!(
            expected_main_hash == expected_draft_hash
                && expected_main_trunk_hash == expected_draft_trunk_hash,
            "MTP compact: embedded source main/draft identity changed during calibration"
        );
    } else {
        let draft_label = format!("source MTP draft {phase}");
        let draft = verify_cmf(draft_path, &draft_label)?;
        ensure!(
            draft.dir_hash() == expected_draft_hash,
            "MTP compact: source MTP draft changed {phase} (expected directory hash {expected_draft_hash:016x}, got {:016x})",
            draft.dir_hash()
        );
        ensure!(
            draft.trunk_hash() == expected_draft_trunk_hash,
            "MTP compact: source MTP draft tensors changed {phase} (expected trunk hash {expected_draft_trunk_hash:016x}, got {:016x})",
            draft.trunk_hash()
        );
    }
    Ok(main)
}

/// Run a very short normal greedy decode on the CPU. The candidate's draft is
/// intentionally detached: the teacher-forced probe validates its compact
/// FFN, while this gate proves the unchanged trunk still produces the source
/// continuation autoregressively.
fn plain_cpu_greedy_admission(
    model: &Arc<CmfModel>,
    probe_tokens: &[u32],
    label: &str,
) -> Result<Vec<u32>> {
    let prompt_len = probe_tokens.len().min(MTP_PLAIN_ADMISSION_PROMPT_TOKENS);
    ensure!(
        prompt_len >= 2,
        "MTP compact: {label} CPU greedy admission prefix is too short"
    );
    let mut sampler = SamplerConfig::default();
    // Do not inherit sampling / penalties from the CLI defaults: this gate is
    // specifically an exact greedy equivalence check.
    sampler.temperature = 0.0;
    sampler.top_p = 1.0;
    sampler.top_k = 0;
    sampler.min_p = 0.0;
    sampler.repetition_penalty = 1.0;
    sampler.presence_penalty = 0.0;
    sampler.suppress_tokens.clear();
    let result = cortiq_engine::gpu::cpu_scope(|| -> std::result::Result<_, String> {
        let mut pipeline =
            Pipeline::from_model(model, sampler).map_err(|error| error.to_string())?;
        // `speculative` gates both canonical MTP and MiMo MTP rounds. The
        // conversion's CPU baseline must never depend on a batch verifier.
        // Detach the draft as well: plain admission deliberately proves the
        // trunk and should not retain a second MTP KV/sidecar while a 100B
        // model's CPU pipeline is live.
        pipeline.mtp = None;
        pipeline.mimo_mtp = None;
        pipeline.speculative = false;
        pipeline.ignore_eos = true;
        pipeline.generate_from_ids(
            &probe_tokens[..prompt_len],
            MTP_PLAIN_ADMISSION_OUTPUT_TOKENS,
            None,
            None,
        )
    })
    .map_err(|error| anyhow::anyhow!("MTP compact: CPU plain greedy admission for {label}: {error}"))?;
    ensure!(
        result.token_ids.len() == MTP_PLAIN_ADMISSION_OUTPUT_TOKENS,
        "MTP compact: CPU plain greedy admission for {label} emitted {} instead of {} token(s)",
        result.token_ids.len(),
        MTP_PLAIN_ADMISSION_OUTPUT_TOKENS
    );
    Ok(result.token_ids)
}

/// Gate publication on the actual runtime loader and one teacher-forced draft
/// pass. A valid CMF directory alone is insufficient: it would not detect a
/// stale independent FFN width or a sidecar attached under the wrong name.
fn admit_candidate(
    source_main: &Arc<CmfModel>,
    destination: &CompactDestination,
    staged_main: &Path,
    staged_draft: &Path,
    expected_mimo: bool,
    expected_unchanged_main_trunk_hash: Option<u64>,
    probe_tokens: &[u32],
    kept_widths: &[usize],
) -> Result<()> {
    let main = verify_cmf(staged_main, "staged candidate main")?;
    if let Some(expected_main_trunk_hash) = expected_unchanged_main_trunk_hash {
        ensure!(
            main.trunk_hash() == expected_main_trunk_hash,
            "MTP compact: staged candidate main differs from calibrated source (expected trunk hash {expected_main_trunk_hash:016x}, got {:016x})",
            main.trunk_hash()
        );
    }
    if destination.external_draft {
        let _draft = verify_cmf(staged_draft, "staged candidate MTP sidecar")?;
    }
    let mut pipeline = Pipeline::from_model(&main, SamplerConfig::default()).map_err(|error| {
        anyhow::anyhow!("MTP compact: loading staged candidate runtime: {error}")
    })?;
    if expected_mimo {
        ensure!(
            pipeline.mimo_mtp.is_some(),
            "MTP compact: staged candidate did not load its MiMo MTP draft"
        );
    } else {
        ensure!(
            pipeline.mtp.is_some(),
            "MTP compact: staged candidate did not load its MTP draft"
        );
    }
    let probe = pipeline.probe_mtp_ffn(probe_tokens).map_err(|error| {
        anyhow::anyhow!("MTP compact: staged candidate draft runtime probe failed: {error}")
    })?;
    validate_probe(&probe).context("MTP compact: staged candidate returned invalid draft probe")?;
    ensure!(
        probe.len() == kept_widths.len(),
        "MTP compact: staged candidate has {} draft layers, plan kept {}",
        probe.len(),
        kept_widths.len()
    );
    for (layer, (row, &kept_width)) in probe.iter().zip(kept_widths).enumerate() {
        ensure!(
            row.observed_positions > 0
                && row.mass.len() == kept_width
                && row.squared_mass.len() == kept_width,
            "MTP compact: staged candidate layer {layer} did not run at its compact width {kept_width}"
        );
    }
    // The draft probe above is the last use of this full candidate pipeline.
    // Do not keep it alive while the two independent plain CPU admission
    // pipelines below are constructed.
    drop(pipeline);
    let source_ids = plain_cpu_greedy_admission(source_main, probe_tokens, "source main")?;
    let candidate_ids = plain_cpu_greedy_admission(&main, probe_tokens, "staged candidate")?;
    ensure!(
        source_ids == candidate_ids,
        "MTP compact: CPU plain greedy admission differs between source and candidate (source {:?}, candidate {:?})",
        source_ids,
        candidate_ids
    );
    Ok(())
}

/// Keep the backbone and independently stored draft provenance separate.
/// MiMo puts its draft in a sidecar, so recording only the main directory
/// hash would let a stale or swapped sidecar look like the calibrated source.
fn source_hash_fields(main_hash: u64, draft_hash: u64) -> Map<String, Value> {
    let mut fields = Map::with_capacity(2);
    fields.insert(
        "source_directory_hash".into(),
        json!(format!("{main_hash:016x}")),
    );
    fields.insert(
        "source_draft_directory_hash".into(),
        json!(format!("{draft_hash:016x}")),
    );
    fields
}

/// Directory hashes identify their physical first files. The trunk hashes
/// additionally bind every merged tensor entry, which matters for shards.
fn source_trunk_hash_fields(main_hash: u64, draft_hash: u64) -> Map<String, Value> {
    let mut fields = Map::with_capacity(2);
    fields.insert(
        "source_trunk_hash".into(),
        json!(format!("{main_hash:016x}")),
    );
    fields.insert(
        "source_draft_trunk_hash".into(),
        json!(format!("{draft_hash:016x}")),
    );
    fields
}

fn tensor_f32(model: &CmfModel, name: &str) -> Result<Vec<f32>> {
    let entry = model
        .tensor(name)
        .with_context(|| format!("MTP compact: missing tensor {name}"))?;
    let mut values = vec![0.0f32; entry.n_elems()];
    dequant_tensor(entry, model.tensor_bytes(name)?, &mut values)
        .map_err(|error| anyhow::anyhow!("MTP compact: dequantizing {name}: {error}"))?;
    Ok(values)
}

/// Calculate `down_proj` column L2² directly from a supported encoded payload.
/// Ranking needs only one scalar per draft channel; materializing a full
/// `[hidden, intermediate]` f32 panel first is prohibitively expensive for a
/// large draft and immediately discarded afterwards. The scalar arithmetic
/// deliberately matches the corresponding full dequantizer followed by the
/// historical ranker: each channel still receives rows in increasing order,
/// so the score and its deterministic tie-break do not change.
fn direct_down_column_norm_sq(
    dtype: TensorDtype,
    source: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Option<Vec<f64>>> {
    ensure!(
        cols > 0,
        "MTP compact: direct down-projection scoring needs a positive column width"
    );
    // These common dense encodings are row-addressable too.  Keep their
    // score-only pass out of the full f32 path: MTP source files commonly use
    // q8_2f for a quality-focused draft, where a temporary full expansion is
    // still needlessly several GiB on a large model.
    let expected = || {
        expected_nbytes(dtype, &[rows, cols])
            .context("MTP compact: direct down-projection score shape overflows")
    };
    let require_payload = |expected: usize| -> Result<()> {
        ensure!(
            source.len() == expected,
            "MTP compact: direct down-projection score payload has {} bytes, expected {expected}",
            source.len()
        );
        Ok(())
    };
    match dtype {
        TensorDtype::F32 => {
            require_payload(expected()?)?;
            let mut column_norm_sq = vec![0.0f64; cols];
            for (index, bytes) in source.chunks_exact(4).enumerate() {
                let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64;
                let column = index % cols;
                column_norm_sq[column] += value * value;
            }
            return Ok(Some(column_norm_sq));
        }
        TensorDtype::F16 | TensorDtype::Bf16 => {
            require_payload(expected()?)?;
            let mut column_norm_sq = vec![0.0f64; cols];
            for (index, bytes) in source.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let value = match dtype {
                    TensorDtype::F16 => f16_to_f32(bits),
                    TensorDtype::Bf16 => bf16_to_f32(bits),
                    _ => unreachable!("matched f16/bf16 dtype"),
                } as f64;
                let column = index % cols;
                column_norm_sq[column] += value * value;
            }
            return Ok(Some(column_norm_sq));
        }
        TensorDtype::Q8Row => {
            require_payload(expected()?)?;
            let q_len = rows
                .checked_mul(cols)
                .context("MTP compact: direct q8 score weight plane overflows")?;
            let (quant, scales) = source.split_at(q_len);
            let mut column_norm_sq = vec![0.0f64; cols];
            for row in 0..rows {
                let scale = f16_to_f32(u16::from_le_bytes([
                    scales[row * 2],
                    scales[row * 2 + 1],
                ]));
                for column in 0..cols {
                    let value = quant[row * cols + column] as i8 as f32 * scale;
                    let value = value as f64;
                    column_norm_sq[column] += value * value;
                }
            }
            return Ok(Some(column_norm_sq));
        }
        TensorDtype::Q8_2f => {
            require_payload(expected()?)?;
            let q_len = rows
                .checked_mul(cols)
                .context("MTP compact: direct q8_2f score weight plane overflows")?;
            let row_scale_len = rows
                .checked_mul(2)
                .context("MTP compact: direct q8_2f score row-scale plane overflows")?;
            let (quant, rest) = source.split_at(q_len);
            let (row_scales, column_scales) = rest.split_at(row_scale_len);
            // Match `dequant_q8_2f`: the scale plane is decoded once, not
            // once per output row. This remains O(cols), rather than a
            // discarded O(rows * cols) f32 panel.
            let column_scales = (0..cols)
                .map(|column| {
                    f16_to_f32(u16::from_le_bytes([
                        column_scales[column * 2],
                        column_scales[column * 2 + 1],
                    ]))
                })
                .collect::<Vec<_>>();
            let mut column_norm_sq = vec![0.0f64; cols];
            for row in 0..rows {
                let row_scale = f16_to_f32(u16::from_le_bytes([
                    row_scales[row * 2],
                    row_scales[row * 2 + 1],
                ]));
                for column in 0..cols {
                    let value = quant[row * cols + column] as i8 as f32
                        * row_scale
                        * column_scales[column];
                    let value = value as f64;
                    column_norm_sq[column] += value * value;
                }
            }
            return Ok(Some(column_norm_sq));
        }
        _ => {}
    }
    let tile_bytes = match dtype {
        TensorDtype::Q4TiledP => Q4TP_NIB,
        TensorDtype::Q2TiledP => Q2TP_CHUNK,
        _ => return Ok(None),
    };
    ensure!(
        cols != 0 && cols % 32 == 0,
        "MTP compact: {dtype:?} direct down-projection scoring needs a positive 32-aligned width, got {cols}"
    );
    require_payload(expected()?)?;
    let groups_per_row = cols / 32;
    let row_weight_bytes = groups_per_row
        .checked_mul(tile_bytes)
        .context("MTP compact: direct down-projection score row width overflows")?;
    let (params_offset, codes_offset, code_stride) = match dtype {
        TensorDtype::Q4TiledP => q4tp_sections(rows, cols),
        TensorDtype::Q2TiledP => q2tp_sections(rows, cols),
        _ => unreachable!("non-tiled dtype returned above"),
    };
    ensure!(
        params_offset
            == rows
                .checked_mul(row_weight_bytes)
                .context("MTP compact: direct down-projection score plane offset overflows")?,
        "MTP compact: direct down-projection score plane layout is inconsistent"
    );
    let params_len = rows
        .checked_mul(4)
        .context("MTP compact: direct down-projection score parameter plane overflows")?;
    let params_end = params_offset
        .checked_add(params_len)
        .context("MTP compact: direct down-projection score parameter plane end overflows")?;
    let codes_len = rows
        .checked_mul(code_stride)
        .context("MTP compact: direct down-projection score code plane overflows")?;
    ensure!(
        params_end <= source.len()
            && codes_offset
                .checked_add(codes_len)
                .is_some_and(|end| end == source.len()),
        "MTP compact: direct down-projection score plane bounds are inconsistent"
    );
    let params = &source[params_offset..params_end];
    let mut column_norm_sq = vec![0.0f64; cols];
    for row in 0..rows {
        let codes = &source[codes_offset + row * code_stride..codes_offset + (row + 1) * code_stride];
        match dtype {
            TensorDtype::Q4TiledP => {
                let ladder = q4tp_ladder(params, row);
                for group in 0..groups_per_row {
                    let scale = ladder[q4tp_code(codes, group)];
                    let tile_offset = row * row_weight_bytes + group * Q4TP_NIB;
                    for (index, &byte) in source[tile_offset..tile_offset + Q4TP_NIB]
                        .iter()
                        .enumerate()
                    {
                        let column = group * 32 + index * 2;
                        let low = ((byte & 0x0f) as f32 - 8.0) * scale;
                        let high = ((byte >> 4) as f32 - 8.0) * scale;
                        let low = low as f64;
                        let high = high as f64;
                        column_norm_sq[column] += low * low;
                        column_norm_sq[column + 1] += high * high;
                    }
                }
            }
            TensorDtype::Q2TiledP => {
                let ladder = q2tp_ladder(params, row);
                for group in 0..groups_per_row {
                    let scale = ladder[q4tp_code(codes, group)];
                    let tile_offset = row * row_weight_bytes + group * Q2TP_CHUNK;
                    for (index, &byte) in source[tile_offset..tile_offset + Q2TP_CHUNK]
                        .iter()
                        .enumerate()
                    {
                        let column = group * 32 + index * 4;
                        for shift in 0..4 {
                            let value = (((byte >> (2 * shift)) & 0x03) as f32 - 1.5) * scale;
                            let value = value as f64;
                            column_norm_sq[column + shift] += value * value;
                        }
                    }
                }
            }
            _ => unreachable!("non-tiled dtype returned above"),
        }
    }
    Ok(Some(column_norm_sq))
}

fn down_column_norm_sq(model: &CmfModel, name: &str, rows: usize, cols: usize) -> Result<Vec<f64>> {
    let entry = model
        .tensor(name)
        .with_context(|| format!("MTP compact: missing tensor {name}"))?;
    ensure!(
        entry.shape.as_slice() == [rows, cols],
        "MTP compact: {name} has shape {:?}, expected [{rows}, {cols}]",
        entry.shape
    );
    let source = model
        .tensor_bytes(name)
        .with_context(|| format!("MTP compact: reading tensor {name}"))?;
    if let Some(norms) = direct_down_column_norm_sq(entry.dtype, source, rows, cols)? {
        return Ok(norms);
    }
    let down = tensor_f32(model, name)?;
    let expected = rows
        .checked_mul(cols)
        .context("MTP compact: down-projection shape overflows usize")?;
    ensure!(
        down.len() == expected,
        "MTP compact: {name} dequantized to {} values, expected {expected}",
        down.len()
    );
    let mut norms = vec![0.0f64; cols];
    for row in down.chunks_exact(cols) {
        for (column, &value) in row.iter().enumerate() {
            let value = value as f64;
            norms[column] += value * value;
        }
    }
    Ok(norms)
}

fn tensor_shape(model: &CmfModel, name: &str) -> Result<Vec<usize>> {
    Ok(model
        .tensor(name)
        .with_context(|| format!("MTP compact: missing tensor {name}"))?
        .shape
        .clone())
}

fn aligned_keep(width: usize, keep_ratio: f64) -> Result<usize> {
    ensure!(
        keep_ratio.is_finite() && keep_ratio > 0.0 && keep_ratio <= 1.0,
        "MTP compact: --keep-ratio must be finite and in (0, 1]"
    );
    ensure!(
        width >= MTP_COMPACT_ALIGNMENT && width % MTP_COMPACT_ALIGNMENT == 0,
        "MTP compact: draft FFN width {width} is not a positive {MTP_COMPACT_ALIGNMENT}-channel codec group"
    );
    let keep = ((width as f64 * keep_ratio).floor() as usize / MTP_COMPACT_ALIGNMENT)
        * MTP_COMPACT_ALIGNMENT;
    ensure!(
        keep >= MTP_COMPACT_ALIGNMENT,
        "MTP compact: --keep-ratio {keep_ratio} leaves fewer than one {MTP_COMPACT_ALIGNMENT}-channel group from width {width}"
    );
    Ok(keep)
}

fn admission_probe_len(record_tokens: usize, draft_layers: usize) -> Result<usize> {
    let required = draft_layers
        .checked_add(2)
        .context("MTP compact: draft layer count overflows admission prefix")?;
    ensure!(
        record_tokens >= required,
        "MTP compact: a calibration record with {record_tokens} token(s) is too short to admit a {draft_layers}-layer draft"
    );
    Ok(record_tokens.min(MTP_ADMISSION_TARGET_TOKENS.max(required)))
}

/// A short calibration sample can still contribute useful statistics to
/// shallow draft layers, but it cannot admit every layer of a deep draft.
/// Return `None` for that latter case so a mixed-length `.tdata` corpus can
/// continue until it reaches a representative, sufficiently long record.
fn admission_probe_prefix(tokens: &[u32], draft_layers: usize) -> Result<Option<&[u32]>> {
    let required = draft_layers
        .checked_add(2)
        .context("MTP compact: draft layer count overflows admission prefix")?;
    if tokens.len() < required {
        return Ok(None);
    }
    let len = admission_probe_len(tokens.len(), draft_layers)?;
    Ok(Some(&tokens[..len]))
}

/// Rank independently measured draft channels.  Kept separate from CMF I/O
/// so the score and deterministic tie break have a small, direct test.
#[cfg(test)]
fn rank_channel_scores(
    squared_mass: &[f64],
    down: &[f32],
    hidden: usize,
    width: usize,
) -> Result<Vec<(usize, f64)>> {
    let expected_down = hidden
        .checked_mul(width)
        .context("MTP compact: down-projection shape overflows usize")?;
    ensure!(
        squared_mass.len() == width && down.len() == expected_down,
        "MTP compact: score inputs do not match a {hidden} × {width} down projection"
    );
    // Accumulate each column in the same row order as the old scalar loop,
    // but traverse the row-major `down_proj` once. Wide draft matrices then
    // stream through cache rather than doing one cache-line-strided scan per
    // channel; the per-channel f64 addition order and deterministic rank tie
    // break remain unchanged.
    let mut column_norm_sq = vec![0.0f64; width];
    for row in down.chunks_exact(width) {
        for (channel, &value) in row.iter().enumerate() {
            let value = value as f64;
            column_norm_sq[channel] += value * value;
        }
    }
    let mut ranked: Vec<(usize, f64)> = (0..width)
        .map(|channel| {
            let col_norm_sq = column_norm_sq[channel];
            let activation = squared_mass[channel];
            ensure!(
                activation.is_finite() && activation >= 0.0 && col_norm_sq.is_finite(),
                "MTP compact: draft channel {channel} has invalid score inputs"
            );
            let score = activation * col_norm_sq;
            ensure!(
                score.is_finite(),
                "MTP compact: draft channel {channel} score overflow"
            );
            Ok((channel, score))
        })
        .collect::<Result<_>>()?;
    ranked.sort_by(|(ia, a), (ib, b)| {
        b.partial_cmp(a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| ia.cmp(ib))
    });
    Ok(ranked)
}

fn rank_channel_norm_scores(
    squared_mass: &[f64],
    column_norm_sq: &[f64],
    width: usize,
) -> Result<Vec<(usize, f64)>> {
    ensure!(
        squared_mass.len() == width && column_norm_sq.len() == width,
        "MTP compact: score inputs do not match a {width}-channel down projection"
    );
    let mut ranked: Vec<(usize, f64)> = (0..width)
        .map(|channel| {
            let col_norm_sq = column_norm_sq[channel];
            let activation = squared_mass[channel];
            ensure!(
                activation.is_finite() && activation >= 0.0 && col_norm_sq.is_finite(),
                "MTP compact: draft channel {channel} has invalid score inputs"
            );
            let score = activation * col_norm_sq;
            ensure!(
                score.is_finite(),
                "MTP compact: draft channel {channel} score overflow"
            );
            Ok((channel, score))
        })
        .collect::<Result<_>>()?;
    ranked.sort_by(|(ia, a), (ib, b)| {
        b.partial_cmp(a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| ia.cmp(ib))
    });
    Ok(ranked)
}

fn validate_probe(probe: &[MtpFfnProbe]) -> Result<()> {
    ensure!(
        !probe.is_empty(),
        "MTP compact: draft probe returned no layers"
    );
    for (layer, row) in probe.iter().enumerate() {
        ensure!(
            row.mass.len() == row.squared_mass.len()
                && row
                    .mass
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && row
                    .squared_mass
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
            "MTP compact: draft layer {layer} returned invalid activation statistics"
        );
    }
    Ok(())
}

/// Final calibration checks that no layer was selected from an all-zero or
/// unseen activation table. A deterministic index tie-break is useful only
/// after the data gives it a real score; otherwise compaction would silently
/// prune arbitrary channels.
fn validate_aggregate_probe(probe: &[MtpFfnProbe]) -> Result<()> {
    validate_probe(probe)?;
    for (layer, row) in probe.iter().enumerate() {
        ensure!(
            row.observed_positions > 0,
            "MTP compact: draft layer {layer} saw no teacher-forced positions"
        );
        let energy = row
            .squared_mass
            .iter()
            .try_fold(0.0f64, |total, &value| {
                let next = total + value;
                ensure!(
                    next.is_finite(),
                    "MTP compact: draft layer {layer} activation energy overflow"
                );
                Ok(next)
            })?;
        ensure!(
            energy > 0.0,
            "MTP compact: draft layer {layer} has zero activation energy; refuse arbitrary channel selection"
        );
    }
    Ok(())
}

/// Add calibration records without allowing a finite individual record to
/// overflow into an invalid plan statistic.
fn accumulate_probe(total: &mut [MtpFfnProbe], probe: &[MtpFfnProbe]) -> Result<()> {
    ensure!(
        total.len() == probe.len(),
        "MTP compact: draft layer count changed from {} to {} during calibration",
        total.len(),
        probe.len()
    );
    for (layer, (sum, row)) in total.iter_mut().zip(probe.iter()).enumerate() {
        ensure!(
            sum.mass.len() == row.mass.len() && sum.squared_mass.len() == row.squared_mass.len(),
            "MTP compact: draft layer {layer} width changed during calibration"
        );
        for (channel, (a, b)) in sum.mass.iter_mut().zip(&row.mass).enumerate() {
            let next = *a + *b;
            ensure!(
                next.is_finite() && next >= 0.0,
                "MTP compact: draft layer {layer} mass overflow at channel {channel}"
            );
            *a = next;
        }
        for (channel, (a, b)) in sum
            .squared_mass
            .iter_mut()
            .zip(&row.squared_mass)
            .enumerate()
        {
            let next = *a + *b;
            ensure!(
                next.is_finite() && next >= 0.0,
                "MTP compact: draft layer {layer} squared mass overflow at channel {channel}"
            );
            *a = next;
        }
        sum.observed_positions = sum
            .observed_positions
            .checked_add(row.observed_positions)
            .ok_or_else(|| anyhow::anyhow!("MTP compact: observed-position counter overflow"))?;
    }
    Ok(())
}

fn rank_draft_channels(
    model: &CmfModel,
    layer: usize,
    squared_mass: &[f64],
    keep_ratio: f64,
) -> Result<(usize, Vec<usize>, Value)> {
    let prefix = format!("model.mtp.layers.{layer}.mlp.");
    let gate_name = format!("{prefix}gate_proj.weight");
    let up_name = format!("{prefix}up_proj.weight");
    let down_name = format!("{prefix}down_proj.weight");
    let gate_shape = tensor_shape(model, &gate_name)?;
    let up_shape = tensor_shape(model, &up_name)?;
    let down_shape = tensor_shape(model, &down_name)?;
    let down_dtype = model
        .tensor(&down_name)
        .with_context(|| format!("MTP compact: missing tensor {down_name}"))?
        .dtype;
    ensure!(
        gate_shape.len() == 2 && up_shape.len() == 2 && down_shape.len() == 2,
        "MTP compact: draft layer {layer} FFN tensors must be matrices"
    );
    let (width, hidden) = (gate_shape[0], gate_shape[1]);
    ensure!(
        up_shape == gate_shape && down_shape == vec![hidden, width],
        "MTP compact: draft layer {layer} has mismatched FFN shapes gate={gate_shape:?}, up={up_shape:?}, down={down_shape:?}"
    );
    ensure!(
        squared_mass.len() == width,
        "MTP compact: draft layer {layer} probe width {} != tensor width {width}",
        squared_mass.len()
    );
    ensure!(
        hidden == model.arch().hidden_size,
        "MTP compact: draft layer {layer} hidden width {hidden} != header hidden_size {}",
        model.arch().hidden_size
    );
    let keep = aligned_keep(width, keep_ratio)?;
    let draft_layers = model.arch().mtp.as_ref().map_or(1, |mtp| mtp.num_layers);
    let scoring_started = Instant::now();
    eprintln!(
        "mtp compact: scoring MTP draft layer {}/{} down_proj ({} × {}, {})",
        layer.saturating_add(1),
        draft_layers,
        hidden,
        width,
        down_dtype.name()
    );
    let column_norm_sq = down_column_norm_sq(model, &down_name, hidden, width)?;
    eprintln!(
        "mtp compact: scored MTP draft layer {}/{} in {:.1}s",
        layer.saturating_add(1),
        draft_layers,
        scoring_started.elapsed().as_secs_f64()
    );
    let ranked = rank_channel_norm_scores(squared_mass, &column_norm_sq, width)
        .with_context(|| format!("MTP compact: draft layer {layer}"))?;
    let total: f64 = ranked.iter().map(|(_, score)| *score).sum();
    ensure!(
        total.is_finite() && total >= 0.0,
        "MTP compact: invalid score total"
    );
    let selected_score: f64 = ranked.iter().take(keep).map(|(_, score)| *score).sum();
    let order = ranked
        .iter()
        .take(keep)
        .map(|(channel, _)| *channel)
        .collect();
    Ok((
        width,
        order,
        json!({
            "layer": layer,
            "source_width": width,
            "kept_width": keep,
            "down_proj_dtype": down_dtype.name(),
            "score_total": total,
            "score_kept": selected_score,
            "score_kept_fraction": if total > 0.0 { Some(selected_score / total) } else { None },
        }),
    ))
}

fn draft_source(main: &Arc<CmfModel>, main_path: &str) -> Result<(PathBuf, Arc<CmfModel>)> {
    if main
        .tensors
        .iter()
        .any(|tensor| tensor.name.starts_with("model.mtp."))
    {
        return Ok((PathBuf::from(main_path), main.clone()));
    }
    let sidecar = mtp_sidecar_path(Path::new(main_path));
    ensure!(
        sidecar.exists(),
        "MTP compact: {} carries no model.mtp.* tensors and its expected sidecar {} is absent",
        main_path,
        sidecar.display()
    );
    let side = Arc::new(
        CmfModel::open_sharded(&sidecar)
            .with_context(|| format!("MTP compact: opening sidecar {}", sidecar.display()))?,
    );
    ensure!(
        side.tensors
            .iter()
            .any(|tensor| tensor.name.starts_with("model.mtp.")),
        "MTP compact: sidecar {} carries no model.mtp.* payload",
        sidecar.display()
    );
    Ok((sidecar, side))
}

/// Build an explicit independent-draft plan from train-only `.tdata`, bake it
/// to private siblings, then admit it only after the final loader and draft
/// runtime pass. The source is never edited. For an external MiMo draft, the
/// preferred `--output` names a new *main* CMF and the matching `.mtp.cmf` is
/// made beside it (the source backbone is hard-linked when possible, copied
/// only when the filesystem makes that necessary). The historical explicit
/// `.mtp.cmf` output remains a sidecar-only compatibility form.
pub fn cmd_mtp_compact(
    model_path: &str,
    data_selector: &str,
    output: &str,
    keep_ratio: f64,
    plan_out: Option<&str>,
    max_records: usize,
    max_record_tokens: usize,
) -> Result<()> {
    let source_main = Path::new(model_path);
    require_main_model_input(source_main)?;
    let output_path = Path::new(output);
    let default_plan = format!("{output}.mtp-plan.json");
    let plan_path = PathBuf::from(plan_out.unwrap_or(&default_plan));
    // This check intentionally precedes the source-wide hash scan. A typo
    // such as `--plan-out candidate.cmf/plan.json` must fail in milliseconds,
    // not after reading a 100B source model from a rented pod volume.
    preflight_explicit_paths(source_main, output_path, &plan_path)?;

    let main = verify_cmf(source_main, "source main")?;
    let (draft_path, draft) = draft_source(&main, model_path)?;
    if !paths_collide(source_main, &draft_path) {
        // Sidecars can be just as large as their backbone. Give their full
        // integrity fence the same start/end progress as every other
        // multi-hour scan, rather than looking hung after main verification.
        verify_open_cmf(&draft, &draft_path, "source MTP draft sidecar")?;
    }

    let destination = compact_destination(source_main, &draft_path, output_path)?;
    ensure!(
        // In the preserved legacy `.mtp.cmf` form, `main` is a private
        // staging companion used only to admit the compact draft; it is not
        // a final user path and must not constrain an existing manual
        // backbone at that same name.
        (!destination.publish_main
            || (!paths_overlap_or_nested(&destination.main, source_main)?
                && !paths_overlap_or_nested(&destination.main, &draft_path)?))
            && !paths_overlap_or_nested(&destination.draft, source_main)?
            && !paths_overlap_or_nested(&destination.draft, &draft_path)?
            // An in-file canonical MTP has one output CMF by design; only a
            // paired MiMo candidate needs a second distinct final filename.
            && (!destination.external_draft
                || !destination.publish_main
                || !paths_overlap_or_nested(&destination.main, &destination.draft)?),
        "MTP compact: candidate output must be distinct from source main/draft and paired main/sidecar paths must not alias or nest"
    );
    // A streamed MTP-only bake coalesces a sharded embedded draft or a
    // sharded MiMo backbone into one valid candidate. A sharded external
    // sidecar remains unsupported: the MiMo loader resolves one `.mtp.cmf`.
    ensure!(
        !destination.external_draft || draft.header.shard.is_none(),
        "MTP compact: a sharded external .mtp.cmf sidecar is not supported; first materialize one canonical sidecar"
    );
    let expected_mimo = main.arch().arch_name == "mimo_v2";
    ensure!(
        !destination.external_draft || expected_mimo,
        "MTP compact: an external .mtp.cmf draft is supported only for the MiMo runtime"
    );
    if destination.external_draft {
        // The sidecar destination is derived only after MTP discovery, so it
        // cannot be checked in the initial explicit-path fence above. Still
        // reject a collision before opening/verifying a large calibration
        // corpus; `prepare_new_file` repeats this race-sensitive check later.
        check_new_file_leaf(&destination.draft, "candidate MTP sidecar")?;
    }
    let draft_widths = crate::tube::preflight_mtp_compact_source(&draft)
        .context("MTP compact: validating source draft for physical compaction")?;
    // Reject an impossible cap before creating output directories, reading an
    // entire `.tdata` ledger, or materializing a large pipeline. The
    // declaration is independent of the runtime loader, so this is a cheap
    // and deterministic user-error path.
    let draft_layer_count = draft
        .arch()
        .mtp
        .as_ref()
        .map(|config| config.num_layers)
        .context("MTP compact: loaded draft has no MTP layer declaration")?;
    ensure!(
        draft_widths.len() == draft_layer_count,
        "MTP compact: source draft declared {draft_layer_count} layer(s), but physical preflight found {}",
        draft_widths.len()
    );
    for (layer, width) in draft_widths.iter().copied().enumerate() {
        aligned_keep(width, keep_ratio).with_context(|| {
            format!("MTP compact: validating --keep-ratio for draft layer {layer}")
        })?;
    }
    let minimum_chunk_tokens = draft_layer_count
        .checked_add(2)
        .context("MTP compact: draft layer count overflows calibration cap")?
        .max(3);
    ensure!(
        max_record_tokens >= minimum_chunk_tokens,
        "MTP compact: --max-record-tokens must be at least {minimum_chunk_tokens} for a {draft_layer_count}-layer draft"
    );

    ensure!(
        !paths_overlap_or_nested(&plan_path, source_main)?
            && !paths_overlap_or_nested(&plan_path, &draft_path)?
            && (!destination.publish_main
                || !paths_overlap_or_nested(&plan_path, &destination.main)?)
            && !paths_overlap_or_nested(&plan_path, &destination.draft)?,
        "MTP compact: --plan-out must be a distinct ordinary-file path, not an alias or child of a source/candidate CMF"
    );
    let (reader, selector) = TdataReader::open_selector(data_selector)
        .with_context(|| format!("MTP compact: opening data selector {data_selector:?}"))?;
    ensure!(
        selector.split == TdataSplitSelector::Train,
        "MTP compact: data selector must be the train split; use PATH.tdata or PATH.tdata:train so validation/test remain untouched"
    );
    let tokenizer_contract = match reader.manifest.raw.get("tokenizer") {
        Some(_) => {
            let source_vocab = main.vocab.as_deref().context(
                "MTP compact: .tdata declares a tokenizer contract but source main CMF has no embedded tokenizer bytes",
            )?;
            let source_tokenizer = Tokenizer::from_bytes(source_vocab).map_err(|error| {
                anyhow::anyhow!("MTP compact: parsing source tokenizer for .tdata contract: {error}")
            })?;
            let tokenizer_config_bytes = serde_json::to_vec(&main.header.tokenizer_config)
                .map_err(|error| anyhow::anyhow!("MTP compact: serializing source tokenizer metadata: {error}"))?;
            validate_tdata_tokenizer_contract(
                &reader.manifest.raw,
                Some(source_vocab),
                &tokenizer_config_bytes,
                source_tokenizer.vocab_size(),
            )?
        }
        None => validate_tdata_tokenizer_contract(&reader.manifest.raw, None, &[], 0)?,
    };
    reader
        .verify_for_calibration(&selector)
        .with_context(|| format!("MTP compact: verifying {data_selector:?}"))?;

    // The output leaves are still reserved without clobbering, but defer any
    // parent-directory creation until source and `.tdata` contracts have
    // passed. A typo in a multi-terabyte dataset should not litter a run
    // directory with empty output parents.
    if destination.publish_main {
        prepare_new_file(&destination.main, "candidate main")?;
    }
    if destination.external_draft {
        prepare_new_file(&destination.draft, "candidate MTP sidecar")?;
    }
    prepare_new_file(&plan_path, "MTP plan")?;

    let mut pipeline = Pipeline::from_model(&main, SamplerConfig::default())
        .map_err(|error| anyhow::anyhow!("MTP compact: loading source runtime: {error}"))?;
    if expected_mimo {
        ensure!(
            pipeline.mimo_mtp.is_some(),
            "MTP compact: source did not load its MiMo MTP draft (check CMF_MIMO_MTP)"
        );
    } else {
        ensure!(
            pipeline.mtp.is_some(),
            "MTP compact: source did not load its MTP draft (check CMF_MTP)"
        );
    }
    let mut aggregate: Option<Vec<MtpFfnProbe>> = None;
    let mut admission_tokens: Option<Vec<u32>> = None;
    let mut records = 0usize;
    let mut selected_records = 0usize;
    let mut chunks = 0usize;
    let mut tokens = 0usize;
    let calibration_started = Instant::now();
    for item in reader.stream(selector)? {
        if max_records != 0 && selected_records >= max_records {
            break;
        }
        let sample = item?;
        selected_records = selected_records
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("MTP compact: selected-record counter overflow"))?;
        let mut start = 0usize;
        let mut used_record = false;
        // Every chunk starts a fresh causal sequence. Apart from bounding the
        // draft/trunk KV, this avoids retaining any record-sized activation
        // state. When a cap would leave a one-token tail, move one token into
        // that tail so no valid calibration ID is silently discarded.
        while sample.tokens.len().saturating_sub(start) >= 2 {
            let remaining = sample.tokens.len() - start;
            let mut take = remaining.min(max_record_tokens);
            if remaining > max_record_tokens && remaining - take == 1 {
                take -= 1;
            }
            debug_assert!(take >= 2 && take <= max_record_tokens);
            let end = start
                .checked_add(take)
                .expect("calibration chunk end checked against source length");
            let chunk = &sample.tokens[start..end];
            let probe = pipeline.probe_mtp_ffn(chunk).map_err(|error| {
                anyhow::anyhow!(
                    "MTP compact: probing record {:?}, chunk {}: {error}",
                    sample.record.rid,
                    chunks.saturating_add(1)
                )
            })?;
            validate_probe(&probe).with_context(|| {
                format!(
                    "MTP compact: invalid draft activation statistics for record {:?}, chunk {}",
                    sample.record.rid,
                    chunks.saturating_add(1)
                )
            })?;
            if let Some(total) = aggregate.as_mut() {
                accumulate_probe(total, &probe).with_context(|| {
                    format!(
                        "MTP compact: aggregating draft activation statistics for record {:?}, chunk {}",
                        sample.record.rid,
                        chunks.saturating_add(1)
                    )
                })?;
            } else {
                aggregate = Some(probe);
            }
            if let Some(prefix) = admission_probe_prefix(chunk, draft_layer_count)? {
                // A minimally valid deep-draft prefix can be only
                // `layers + 2` tokens long. Retain the longest bounded
                // candidate instead: the runtime admission remains small,
                // but tests substantially more positions whenever the
                // calibration set contains a normal-length record.
                if admission_tokens
                    .as_ref()
                    .map_or(true, |current| prefix.len() > current.len())
                {
                    admission_tokens = Some(prefix.to_vec());
                }
            }
            chunks = chunks
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("MTP compact: chunk counter overflow"))?;
            tokens = tokens
                .checked_add(chunk.len())
                .ok_or_else(|| anyhow::anyhow!("MTP compact: token counter overflow"))?;
            used_record = true;
            // Long exact calibration has no safe GPU toggle yet. Emit
            // progress by chunks so a single giant record cannot look hung.
            if chunks == 1 || chunks % 8 == 0 {
                let record_progress = if used_record {
                    records.saturating_add(1)
                } else {
                    records
                };
                eprintln!(
                    "mtp compact: calibrating {chunks} chunk(s), {tokens} token(s), {record_progress} source record(s), elapsed {:.1}s",
                    calibration_started.elapsed().as_secs_f64()
                );
            }
            start = end;
        }
        if used_record {
            records = records
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("MTP compact: record counter overflow"))?;
        }
    }
    let aggregate = aggregate.ok_or_else(|| {
        anyhow::anyhow!(
            "MTP compact: no usable calibration record with at least two tokens was selected"
        )
    })?;
    validate_aggregate_probe(&aggregate)?;
    let admission_tokens = admission_tokens.ok_or_else(|| {
        anyhow::anyhow!(
            "MTP compact: no calibration record is long enough to admit all {draft_layer_count} draft layer(s); need at least {} token(s)",
            draft_layer_count.saturating_add(2)
        )
    })?;
    // The exact source probe is no longer needed once statistics and the
    // bounded admission prefix are fixed. Drop its trunk/draft KV and any
    // device-residency bookkeeping before allocating score/bake panels.
    drop(pipeline);

    let mut mtp_layers = Vec::with_capacity(aggregate.len());
    let mut report_layers = Vec::with_capacity(aggregate.len());
    let mut kept_widths = Vec::with_capacity(aggregate.len());
    for (layer, probe) in aggregate.iter().enumerate() {
        let (source_width, order, mut report) =
            rank_draft_channels(&draft, layer, &probe.squared_mass, keep_ratio)?;
        let obj = report
            .as_object_mut()
            .expect("locally constructed JSON object");
        obj.insert("observed_positions".into(), json!(probe.observed_positions));
        obj.insert(
            "activation_mass".into(),
            json!(probe.mass.iter().sum::<f64>()),
        );
        obj.insert(
            "activation_squared_mass".into(),
            json!(probe.squared_mass.iter().sum::<f64>()),
        );
        let kept_width = order.len();
        mtp_layers.push(json!({"order": order, "widths": [kept_width]}));
        kept_widths.push(kept_width);
        // Keep the source width in this independent report: it makes a plan
        // edited by hand auditable before tube bake sees it.
        report_layers.push(json!({"source_width": source_width, "report": report}));
    }

    // The calibration and scorer were allowed to retain the original mmap
    // models. From here on, use only their immutable identities and reopen a
    // fresh source at the fence below. This matters for very large main+draft
    // pairs: candidate baking must not compete with stale model mappings,
    // open shard handles, or their page-cache residency.
    let source_main_dir_hash = main.dir_hash();
    let source_draft_dir_hash = draft.dir_hash();
    let source_main_trunk_hash = main.trunk_hash();
    let source_draft_trunk_hash = draft.trunk_hash();
    let draft_activation_contract = draft.arch().hidden_act.clone();

    let mut selection = json!({
        "kind": "draft-activation-weighted-down-column-energy-v1",
        "formula": "sum_teacher_forced((activation(gate, up))^2) * down_proj_column_l2_squared",
        "activation_contract": draft_activation_contract,
        "keep_ratio": keep_ratio,
        "alignment": MTP_COMPACT_ALIGNMENT,
        "calibration_records": records,
        "calibration_selected_records": selected_records,
        "calibration_chunks": chunks,
        "calibration_tokens": tokens,
        "max_record_tokens": max_record_tokens,
        "calibration_chunking": "fresh-causal-sequence-per-contiguous-chunk-v1",
        "admission_probe_tokens": admission_tokens.len(),
        "tokenizer_contract": tokenizer_contract,
        "data_selector": data_selector,
        "source_main": model_path,
        "source_draft": draft_path.display().to_string(),
        "output_layout": if destination.external_draft {
            if destination.publish_main {
                "runnable-main-and-sidecar-v1"
            } else {
                "legacy-sidecar-only-v1"
            }
        } else {
            "runnable-embedded-main-v1"
        },
        "layers": report_layers,
    });
    selection
        .as_object_mut()
        .expect("locally constructed JSON object")
        .extend(source_hash_fields(source_main_dir_hash, source_draft_dir_hash));
    selection
        .as_object_mut()
        .expect("locally constructed JSON object")
        .extend(source_trunk_hash_fields(
            source_main_trunk_hash,
            source_draft_trunk_hash,
        ));
    let plan = json!({
        "format": "cmf-mtp-ffn-plan",
        "version": 2,
        "trunk": {"mode": "preserve"},
        "mtp": {"mode": "compact", "layers": mtp_layers},
        "selection": selection,
    });

    // Long calibration is intentionally exact and can take hours. Re-check
    // both immutable inputs immediately before materializing any candidate.
    // Release the calibrator's source mappings first; the returned fresh main
    // is retained only because an external sharded MiMo backbone may need to
    // be streamed into its candidate main below.
    drop(draft);
    drop(main);
    let source_for_bake = revalidate_source_identity(
        "after calibration",
        source_main,
        &draft_path,
        source_main_dir_hash,
        source_draft_dir_hash,
        source_main_trunk_hash,
        source_draft_trunk_hash,
    )?;

    let stage_anchor = if destination.publish_main {
        &destination.main
    } else {
        &destination.draft
    };
    let main_stage_dir = create_stage_dir(stage_anchor, "model")?;
    let plan_stage_dir = match create_stage_dir(&plan_path, "plan") {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_dir_all(&main_stage_dir);
            return Err(error);
        }
    };
    let staged_main = match staged_child(&main_stage_dir, &destination.main) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_dir_all(&main_stage_dir);
            let _ = fs::remove_dir_all(&plan_stage_dir);
            return Err(error);
        }
    };
    let staged_draft = if destination.external_draft {
        match staged_child(&main_stage_dir, &destination.draft) {
            Ok(path) => path,
            Err(error) => {
                let _ = fs::remove_dir_all(&main_stage_dir);
                let _ = fs::remove_dir_all(&plan_stage_dir);
                return Err(error);
            }
        }
    } else {
        staged_main.clone()
    };
    let staged_plan = match staged_child(&plan_stage_dir, &plan_path) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_dir_all(&main_stage_dir);
            let _ = fs::remove_dir_all(&plan_stage_dir);
            return Err(error);
        }
    };
    let mut backbone_materialization: Option<String> = None;
    let backbone_staged = (|| -> Result<()> {
        if destination.external_draft {
            if let Some(shard) = &source_for_bake.header.shard {
                eprintln!(
                    "mtp compact: streaming/coalescing {} source main shard(s) into a private candidate stage",
                    shard.count
                );
                crate::knowledge::refuse_genome_rewrite(
                    &source_for_bake,
                    "mtp-compact sharded MiMo backbone",
                )?;
                crate::tube::stream_copy_as_unsharded_new(&source_for_bake, &staged_main)?;
                backbone_materialization = Some(format!(
                    "streamed coalescing of {} source shard(s)",
                    shard.count
                ));
            } else {
                eprintln!(
                    "mtp compact: staging unchanged main backbone (hard-link when the filesystem permits)"
                );
                backbone_materialization = Some(stage_backbone(source_main, &staged_main)?.into());
            }
        }
        Ok(())
    })();
    if let Err(error) = backbone_staged {
        let _ = fs::remove_dir_all(&main_stage_dir);
        let _ = fs::remove_dir_all(&plan_stage_dir);
        return Err(error);
    }
    // `cmd_tube_bake_new` reopens the draft itself. Do not keep the fresh
    // identity-fence main mmap/FDs alive while that potentially multi-hour
    // writer and the candidate admission pipelines run.
    drop(source_for_bake);

    let staged = (|| -> Result<()> {
        write_new_file(&staged_plan, &serde_json::to_vec_pretty(&plan)?, "MTP plan")?;
        eprintln!("mtp compact: streaming compact MTP draft into a private candidate stage");
        crate::tube::cmd_tube_bake_new(
            draft_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("MTP compact: non-UTF-8 source draft path"))?,
            staged_plan
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("MTP compact: non-UTF-8 staged plan path"))?,
            staged_draft
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("MTP compact: non-UTF-8 staged draft path"))?,
        )?;
        // Source bytes must still be exactly the ones used for calibration
        // when we compare the candidate's normal greedy trunk below. This
        // repeats the full identity fence after the potentially long bake.
        let source_for_admission = revalidate_source_identity(
            "after bake",
            source_main,
            &draft_path,
            source_main_dir_hash,
            source_draft_dir_hash,
            source_main_trunk_hash,
            source_draft_trunk_hash,
        )?;
        admit_candidate(
            &source_for_admission,
            &destination,
            &staged_main,
            &staged_draft,
            expected_mimo,
            destination.external_draft.then_some(source_main_trunk_hash),
            &admission_tokens,
            &kept_widths,
        )
    })();
    if let Err(error) = staged {
        // Both directories were atomically created by this invocation. Never
        // remove a final user name on a failed calibration or admission.
        let _ = fs::remove_dir_all(&main_stage_dir);
        let _ = fs::remove_dir_all(&plan_stage_dir);
        return Err(error);
    }

    // Publish the audit plan before a user-visible candidate. A paired draft
    // goes first, so the main name becomes runnable only after both components
    // are visible. There is no safe multi-file rename on every supported
    // filesystem. In particular, do *not* try to roll back a published name:
    // another process could replace it between our hard-link and remove call.
    // Leaving a rare partial result is preferable to deleting user work.
    let published = (|| -> Result<()> {
        eprintln!("mtp compact: publishing validated candidate without overwriting existing files");
        publish_new_file(&staged_plan, &plan_path, "MTP plan")?;
        if destination.external_draft {
            // Publish the sidecar first: the main name becomes runnable only
            // after both components are visible.
            publish_new_file(&staged_draft, &destination.draft, "candidate MTP sidecar")?;
        }
        if destination.publish_main {
            publish_new_file(&staged_main, &destination.main, "candidate main")?;
        }
        Ok(())
    })();
    if let Err(error) = published {
        eprintln!(
            "warning: MTP compact publication failed; any already-published plan or sidecar is intentionally left in place rather than risking deletion of a concurrently created file"
        );
        let _ = fs::remove_dir_all(&main_stage_dir);
        let _ = fs::remove_dir_all(&plan_stage_dir);
        return Err(error);
    }
    if let Err(error) = fs::remove_dir_all(&main_stage_dir) {
        eprintln!(
            "warning: MTP compact published candidate, but could not remove private stage {}: {error}",
            main_stage_dir.display()
        );
    }
    if let Err(error) = fs::remove_dir_all(&plan_stage_dir) {
        eprintln!(
            "warning: MTP compact published candidate, but could not remove private stage {}: {error}",
            plan_stage_dir.display()
        );
    }

    let published_main = if destination.publish_main {
        format!("\n  candidate main: {}", destination.main.display())
    } else {
        "\n  candidate main: not published (legacy sidecar-only output; pair the draft with a compatible main CMF yourself)".to_owned()
    };
    println!(
        "mtp compact: runtime admission passed after {} calibration record(s), {} bounded chunk(s), {} token(s), {} draft layer(s)\n  calibration chunk cap: {} token(s){}\n  candidate draft: {}\n  plan: {}{}\n  note: this proves CMF integrity, draft execution, and a short CPU plain greedy source/candidate check; measure held-out quality and Metal/Vulkan (including any speculative graph) separately",
        records,
        chunks,
        tokens,
        aggregate.len(),
        max_record_tokens,
        published_main,
        destination.draft.display(),
        plan_path.display(),
        backbone_materialization
            .map(|how| format!("\n  backbone materialization: {how}"))
            .unwrap_or_default(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_probe, admission_probe_len, admission_probe_prefix, aligned_keep,
        cmd_mtp_compact, compact_destination, main_path_for_mtp_sidecar, paths_collide,
        paths_overlap_or_nested, preflight_explicit_paths, prepare_new_file, publish_new_file,
        rank_channel_norm_scores, rank_channel_scores, require_main_model_input, source_hash_fields,
        source_trunk_hash_fields, stage_backbone,
        validate_aggregate_probe,
        direct_down_column_norm_sq, validate_probe, validate_tdata_tokenizer_contract,
    };
    use crate::convert::{Quant, quantize_2d};
    use cortiq_core::format::TokenizerBundle;
    use cortiq_core::{
        CMF_VERSION, CmfHeader, CmfModel, LayerType, ModelArch, MtpConfig, NormStyle, QuantType,
        TensorDtype, TensorSpec,
    };
    use cortiq_engine::{MtpFfnProbe, Pipeline, SamplerConfig};
    use serde_json::json;
    use std::fs;
    use std::sync::Arc;

    const MIMO_TEST_HIDDEN: usize = 4;
    const MIMO_TEST_VOCAB: usize = 16;
    const MIMO_TEST_DRAFT_WIDTH: usize = 64;

    fn test_f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| cortiq_core::quant::f32_to_f16(*value).to_le_bytes())
            .collect()
    }

    fn test_f16_tensor(name: impl Into<String>, shape: &[usize], salt: usize) -> TensorSpec {
        let len = shape.iter().product();
        let values = (0..len)
            .map(|index| (((index * 17 + salt * 13) % 29) as f32 + 1.0) / 128.0)
            .collect::<Vec<_>>();
        TensorSpec {
            name: name.into(),
            dtype: TensorDtype::F16,
            shape: shape.to_vec(),
            data: test_f16_bytes(&values),
        }
    }

    fn test_f16_vector(name: impl Into<String>, values: &[f32]) -> TensorSpec {
        TensorSpec {
            name: name.into(),
            dtype: TensorDtype::F16,
            shape: vec![values.len()],
            data: test_f16_bytes(values),
        }
    }

    fn mimo_test_header() -> CmfHeader {
        CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
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
            arch: ModelArch {
                arch_name: "mimo_v2".into(),
                hidden_size: MIMO_TEST_HIDDEN,
                intermediate_size: MIMO_TEST_HIDDEN,
                num_layers: 1,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: MIMO_TEST_HIDDEN,
                vocab_size: MIMO_TEST_VOCAB,
                layer_types: vec![LayerType::FullAttention],
                rms_norm_eps: 1e-6,
                norm_style: NormStyle::Qwen,
                rope_theta: 10_000.0,
                tie_word_embeddings: false,
                partial_rotary_factor: 1.0,
                yarn: None,
                attention_heads_per_layer: None,
                kv_heads_per_layer: None,
                v_head_dim: Some(MIMO_TEST_HIDDEN),
                hidden_act: "silu".into(),
                embed_multiplier: 1.0,
                query_pre_attn_scalar: None,
                sliding_window: None,
                sliding_window_pattern: None,
                rope_local_base_freq: None,
                local_partial_rotary_factor: None,
                global_head_dim: None,
                num_global_kv_heads: None,
                global_partial_rotary_factor: None,
                final_logit_softcapping: None,
                mla: None,
                no_rope_layers: None,
                attention_sink: false,
                activation_situ_beta: None,
                activation_situ_linear_beta: None,
                attn_logit_softcapping: None,
                attn_v_norm: false,
                qk_norm_after_rope: false,
                mtp: Some(MtpConfig {
                    num_layers: 1,
                    ffn_intermediate_sizes: Some(vec![MIMO_TEST_DRAFT_WIDTH]),
                    share_lm_head: true,
                    share_embed: true,
                }),
                moe: None,
                qwen4_exp: None,
                block_diffusion: None,
                deepseek_v41: None,
                linear_core: None,
                anchor_core: None,
                head_clusters: None,
                max_position_embeddings: 64,
                linear_conv_kernel_dim: None,
                linear_num_key_heads: None,
                linear_num_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                rope_freq_factors: None,
                logit_multiplier: None,
                g3n: None,
                kda_gate_lower_bound: None,
                num_loops: 1,
                loop_final_norm: false,
                prism_hadamard: None,
            },
            quant_type: QuantType::F16,
            provenance: None,
        }
    }

    fn mimo_test_main_tensors() -> Vec<TensorSpec> {
        let h = MIMO_TEST_HIDDEN;
        vec![
            test_f16_tensor("model.embed_tokens.weight", &[MIMO_TEST_VOCAB, h], 1),
            test_f16_vector("model.norm.weight", &[1.0; MIMO_TEST_HIDDEN]),
            test_f16_tensor("lm_head.weight", &[MIMO_TEST_VOCAB, h], 2),
            test_f16_vector(
                "model.layers.0.input_layernorm.weight",
                &[1.0; MIMO_TEST_HIDDEN],
            ),
            test_f16_vector(
                "model.layers.0.post_attention_layernorm.weight",
                &[1.0; MIMO_TEST_HIDDEN],
            ),
            test_f16_tensor("model.layers.0.self_attn.q_proj.weight", &[h, h], 3),
            test_f16_tensor("model.layers.0.self_attn.k_proj.weight", &[h, h], 4),
            test_f16_tensor("model.layers.0.self_attn.v_proj.weight", &[h, h], 5),
            test_f16_tensor("model.layers.0.self_attn.o_proj.weight", &[h, h], 6),
            test_f16_tensor("model.layers.0.mlp.gate_proj.weight", &[h, h], 7),
            test_f16_tensor("model.layers.0.mlp.up_proj.weight", &[h, h], 8),
            test_f16_tensor("model.layers.0.mlp.down_proj.weight", &[h, h], 9),
        ]
    }

    fn mimo_test_sidecar_tensors() -> Vec<TensorSpec> {
        let h = MIMO_TEST_HIDDEN;
        let p = "model.mtp.layers.0.";
        vec![
            test_f16_vector(format!("{p}enorm.weight"), &[1.0; MIMO_TEST_HIDDEN]),
            test_f16_vector(format!("{p}hnorm.weight"), &[1.0; MIMO_TEST_HIDDEN]),
            test_f16_tensor(format!("{p}eh_proj.weight"), &[h, 2 * h], 10),
            test_f16_vector(
                format!("{p}input_layernorm.weight"),
                &[1.0; MIMO_TEST_HIDDEN],
            ),
            test_f16_tensor(format!("{p}self_attn.q_proj.weight"), &[h, h], 11),
            test_f16_tensor(format!("{p}self_attn.k_proj.weight"), &[h, h], 12),
            test_f16_tensor(format!("{p}self_attn.v_proj.weight"), &[h, h], 13),
            test_f16_tensor(format!("{p}self_attn.o_proj.weight"), &[h, h], 14),
            test_f16_vector(format!("{p}self_attn.sinks"), &[0.0]),
            test_f16_vector(
                format!("{p}post_attention_layernorm.weight"),
                &[1.0; MIMO_TEST_HIDDEN],
            ),
            test_f16_tensor(
                format!("{p}mlp.gate_proj.weight"),
                &[MIMO_TEST_DRAFT_WIDTH, h],
                15,
            ),
            test_f16_tensor(
                format!("{p}mlp.up_proj.weight"),
                &[MIMO_TEST_DRAFT_WIDTH, h],
                16,
            ),
            test_f16_tensor(
                format!("{p}mlp.down_proj.weight"),
                &[h, MIMO_TEST_DRAFT_WIDTH],
                17,
            ),
            test_f16_vector(
                format!("{p}final_layernorm.weight"),
                &[1.0; MIMO_TEST_HIDDEN],
            ),
        ]
    }

    fn write_mimo_test_tdata_with_manifest(path: &std::path::Path, manifest: serde_json::Value) {
        use crate::tdata::{TdataRecord, TdataWriter};

        let tokens = [1_u32, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut writer = TdataWriter::create(path, manifest).unwrap();
        let row = json!({
            "rid": "mimo#0",
            "pid": "mimo",
            "sample": 0,
            "category": "test",
            "split": "train",
            "usable": true,
            "messages": [{"role": "user", "content": "test"}],
            "response": {"think": "", "content": "test"},
            "finish": "stop",
            "tok": {"index": 0, "offset": 0, "prompt": 1, "think": 0, "answer": 8, "total": 9},
            "quality": {"truncated": false, "loop_suspect": false, "check": null},
            "meta": {}
        });
        let record = TdataRecord::from_json_line(&row.to_string(), 1).unwrap();
        writer.append_record(record, &tokens).unwrap();
        writer.finalize().unwrap();
    }

    fn write_mimo_test_tdata(path: &std::path::Path) {
        write_mimo_test_tdata_with_manifest(path, json!({"teacher": "synthetic"}));
    }

    #[test]
    fn compact_width_stays_codec_aligned_and_fail_closed() {
        assert_eq!(aligned_keep(128, 1.0).unwrap(), 128);
        assert_eq!(aligned_keep(128, 0.76).unwrap(), 96);
        assert_eq!(aligned_keep(128, 0.25).unwrap(), 32);
        assert!(aligned_keep(128, 0.24).is_err());
        assert!(aligned_keep(127, 0.9).is_err());
        assert!(aligned_keep(128, f64::NAN).is_err());
    }

    #[test]
    fn admission_probe_prefix_is_bounded_but_covers_deep_drafts() {
        assert_eq!(admission_probe_len(10_000, 5).unwrap(), 256);
        assert_eq!(admission_probe_len(302, 300).unwrap(), 302);
        assert!(admission_probe_len(5, 5).is_err());
        assert!(admission_probe_prefix(&[1, 2], 1).unwrap().is_none());
        assert_eq!(
            admission_probe_prefix(&[1, 2, 3, 4], 1).unwrap(),
            Some(&[1, 2, 3, 4][..])
        );
    }

    #[test]
    fn ranking_uses_draft_activation_and_down_column_energy() {
        // Four columns, two output rows. Their column L2² values are
        // [1, 1, 9, 100]; combined with the activation energy below this
        // makes the order [2, 1, 3, 0].
        let down = vec![1.0, 1.0, 3.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let ranked = rank_channel_scores(&[1.0, 16.0, 4.0, 0.1], &down, 2, 4).unwrap();
        assert_eq!(
            ranked.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![2, 1, 3, 0]
        );
        // Equal scores must be deterministic: lower original index first.
        let tied = rank_channel_scores(&[1.0, 1.0], &[1.0, 1.0], 1, 2).unwrap();
        assert_eq!(
            tied.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(rank_channel_scores(&[f64::NAN], &[1.0], 1, 1).is_err());
        assert!(rank_channel_scores(&[f64::MAX], &[f32::MAX], 1, 1).is_err());
    }

    #[test]
    fn direct_down_scoring_matches_full_dequant_ranking() {
        // Score a nontrivial width with ties broken by source index.  The
        // direct path must be bit-for-bit equivalent to first expanding the
        // full q4tp/q2tp matrix and using the historical ranker; otherwise a
        // memory optimization could silently choose different channels.
        let (rows, cols) = (7usize, 64usize);
        let values = (0..rows * cols)
            .map(|index| ((index * 37 % 149) as f32 - 74.0) / 43.0)
            .collect::<Vec<_>>();
        let mass = (0..cols)
            .map(|index| if index % 11 == 0 { 1.0 } else { (index % 7 + 1) as f64 })
            .collect::<Vec<_>>();
        let mut encoded = vec![
            (
                TensorDtype::F32,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                TensorDtype::Bf16,
                values
                    .iter()
                    // The direct scorer must be checked against the same
                    // bytes the generic BF16 decoder consumes.  Truncation
                    // is sufficient here because this is a format-path
                    // equivalence test rather than a quantizer-quality test.
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ];
        encoded.extend(
            [
                Quant::F16,
                Quant::Q8Row,
                Quant::Q8_2f,
                Quant::Q4TiledP,
                Quant::Q2TiledP,
            ]
            .into_iter()
            .map(|quant| quantize_2d(quant, &values, rows, cols)),
        );
        for (dtype, source) in encoded {
            let direct_norms = direct_down_column_norm_sq(dtype, &source, rows, cols)
                .unwrap()
                .expect("tiled codec must use direct score path");
            let mut full = vec![0.0f32; rows * cols];
            match dtype {
                TensorDtype::F32 => {
                    for (slot, bytes) in full.iter_mut().zip(source.chunks_exact(4)) {
                        *slot = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    }
                }
                TensorDtype::F16 => {
                    for (slot, bytes) in full.iter_mut().zip(source.chunks_exact(2)) {
                        *slot = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
                            bytes[0], bytes[1],
                        ]));
                    }
                }
                TensorDtype::Bf16 => {
                    for (slot, bytes) in full.iter_mut().zip(source.chunks_exact(2)) {
                        *slot = cortiq_core::quant::bf16_to_f32(u16::from_le_bytes([
                            bytes[0], bytes[1],
                        ]));
                    }
                }
                TensorDtype::Q8Row => {
                    cortiq_core::quant::dequant_q8_row(&source, rows, cols, &mut full)
                }
                TensorDtype::Q8_2f => {
                    cortiq_core::quant::dequant_q8_2f(&source, rows, cols, &mut full)
                }
                TensorDtype::Q4TiledP => {
                    cortiq_core::quant::dequant_q4tp(&source, rows, cols, &mut full)
                }
                TensorDtype::Q2TiledP => {
                    cortiq_core::quant::dequant_q2tp(&source, rows, cols, &mut full)
                }
                _ => panic!("unexpected direct-score dtype {dtype:?}"),
            }
            let reference = rank_channel_scores(&mass, &full, rows, cols).unwrap();
            let direct = rank_channel_norm_scores(&mass, &direct_norms, cols).unwrap();
            assert_eq!(
                direct, reference,
                "{dtype:?} direct score path changed ranking or score bits"
            );
        }
    }

    #[test]
    fn selection_records_independent_main_and_draft_hashes() {
        let fields = source_hash_fields(0xabc, 0xdef);
        assert_eq!(
            fields
                .get("source_directory_hash")
                .and_then(serde_json::Value::as_str),
            Some("0000000000000abc")
        );
        assert_eq!(
            fields
                .get("source_draft_directory_hash")
                .and_then(serde_json::Value::as_str),
            Some("0000000000000def")
        );
        let trunks = source_trunk_hash_fields(0x123, 0x456);
        assert_eq!(
            trunks
                .get("source_trunk_hash")
                .and_then(serde_json::Value::as_str),
            Some("0000000000000123")
        );
        assert_eq!(
            trunks
                .get("source_draft_trunk_hash")
                .and_then(serde_json::Value::as_str),
            Some("0000000000000456")
        );
    }

    #[test]
    fn calibration_tokenizer_contract_is_verified_or_explicitly_legacy() {
        let vocab = b"synthetic-tokenizer-v1";
        let config = br#"{"eos_token_id":2}"#;
        let manifest = json!({
            "tokenizer": {
                "vocab_size": 17,
                "vocab_sha256": crate::knowledge::sha256_hex(vocab),
                "config_sha256": crate::knowledge::sha256_hex(config),
            }
        });
        assert_eq!(
            validate_tdata_tokenizer_contract(&manifest, Some(vocab), config, 17).unwrap(),
            "verified-v1"
        );
        let mismatch = validate_tdata_tokenizer_contract(
            &manifest,
            Some(b"different-tokenizer"),
            config,
            17,
        )
        .unwrap_err()
        .to_string();
        assert!(mismatch.contains("vocab_sha256"), "{mismatch}");
        let partial = json!({"tokenizer": {"vocab_size": 17}});
        assert!(validate_tdata_tokenizer_contract(&partial, Some(vocab), config, 17).is_err());
        assert_eq!(
            validate_tdata_tokenizer_contract(&json!({"teacher": "legacy"}), None, &[], 0)
                .unwrap(),
            "absent-legacy"
        );
    }

    #[test]
    fn tokenizer_contract_is_checked_before_any_output_stage_is_created() {
        // Exercise the command boundary, not just the hash helper: a real
        // embedded tokenizer/config must be recorded as verified in the
        // published plan, while a mismatched dataset must fail before it can
        // create an output parent directory.
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.cmf");
        let source_sidecar = temp.path().join("source.mtp.cmf");
        let matching_data = temp.path().join("matching.tdata");
        let matching_output = temp.path().join("matching.cmf");
        let vocab = br#"{"model":{"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15},"merges":[]},"added_tokens":[]}"#;
        let mut header = mimo_test_header();
        header.tokenizer_config = Some(TokenizerBundle {
            chat_template: None,
            eos_token_ids: vec![15],
            bos_token_id: None,
            pad_token_id: None,
        });
        let config = serde_json::to_vec(&header.tokenizer_config).unwrap();
        let matching_manifest = json!({
            "teacher": "synthetic",
            "tokenizer": {
                "vocab_size": MIMO_TEST_VOCAB,
                "vocab_sha256": crate::knowledge::sha256_hex(vocab),
                "config_sha256": crate::knowledge::sha256_hex(&config),
            }
        });
        CmfModel::write(
            &source,
            &header,
            &mimo_test_main_tensors(),
            None,
            Some(vocab),
        )
        .unwrap();
        CmfModel::write(
            &source_sidecar,
            &header,
            &mimo_test_sidecar_tensors(),
            None,
            None,
        )
        .unwrap();
        write_mimo_test_tdata_with_manifest(&matching_data, matching_manifest.clone());

        cmd_mtp_compact(
            source.to_str().unwrap(),
            matching_data.to_str().unwrap(),
            matching_output.to_str().unwrap(),
            0.5,
            None,
            1,
            4,
        )
        .unwrap();
        let plan: serde_json::Value = serde_json::from_slice(
            &fs::read(temp.path().join("matching.cmf.mtp-plan.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            plan["selection"]["tokenizer_contract"],
            "verified-v1",
            "matching embedded tokenizer must be auditable in the plan"
        );

        let mismatched_data = temp.path().join("mismatched.tdata");
        let mut mismatched_manifest = matching_manifest;
        mismatched_manifest["tokenizer"]["vocab_sha256"] = json!("00");
        write_mimo_test_tdata_with_manifest(&mismatched_data, mismatched_manifest);
        let nested_output = temp.path().join("must-not-exist/nested/candidate.cmf");
        let error = cmd_mtp_compact(
            source.to_str().unwrap(),
            mismatched_data.to_str().unwrap(),
            nested_output.to_str().unwrap(),
            0.5,
            None,
            1,
            4,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("vocab_sha256"), "{error}");
        assert!(
            !nested_output.parent().unwrap().exists(),
            "bad tokenizer data must fail before staging/output parents"
        );
    }

    #[test]
    fn probe_aggregation_fails_closed_on_invalid_or_overflowed_statistics() {
        let invalid = vec![MtpFfnProbe {
            mass: vec![1.0],
            squared_mass: vec![f64::NAN],
            observed_positions: 1,
        }];
        assert!(validate_probe(&invalid).is_err());

        let mut total = vec![MtpFfnProbe {
            mass: vec![f64::MAX],
            squared_mass: vec![1.0],
            observed_positions: 1,
        }];
        let next = vec![MtpFfnProbe {
            mass: vec![f64::MAX],
            squared_mass: vec![1.0],
            observed_positions: 1,
        }];
        assert!(accumulate_probe(&mut total, &next).is_err());

        let zero = vec![MtpFfnProbe {
            mass: vec![0.0],
            squared_mass: vec![0.0],
            observed_positions: 1,
        }];
        assert!(validate_aggregate_probe(&zero).is_err());
    }

    #[test]
    fn external_sidecar_output_means_a_ready_candidate_pair() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("base.cmf");
        let source_draft = temp.path().join("base.mtp.cmf");
        fs::write(&source, b"main").unwrap();
        fs::write(&source_draft, b"draft").unwrap();

        let candidate = temp.path().join("candidate/model.cmf");
        let dest = compact_destination(&source, &source_draft, &candidate).unwrap();
        assert!(dest.external_draft);
        assert!(dest.publish_main);
        assert_eq!(dest.main, candidate);
        assert_eq!(dest.draft, temp.path().join("candidate/model.mtp.cmf"));
        let legacy = temp.path().join("candidate/model.mtp.cmf");
        let legacy_dest = compact_destination(&source, &source_draft, &legacy).unwrap();
        assert!(legacy_dest.external_draft);
        assert!(!legacy_dest.publish_main);
        assert_eq!(legacy_dest.main, temp.path().join("candidate/model.cmf"));
        assert_eq!(legacy_dest.draft, legacy);
        assert_eq!(
            main_path_for_mtp_sidecar(&temp.path().join("candidate/model.mtp.cmf")).unwrap(),
            temp.path().join("candidate/model.cmf")
        );
    }

    #[test]
    fn source_model_must_be_main_not_mtp_sidecar() {
        assert!(require_main_model_input(std::path::Path::new("base.cmf")).is_ok());
        let error = require_main_model_input(std::path::Path::new("base.mtp.cmf"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("main .cmf"), "{error}");
    }

    #[test]
    fn sidecar_output_requires_an_external_draft() {
        // Canonical MTP keeps main and draft in one file, so the legacy
        // sidecar-only spelling would otherwise publish a partial artifact.
        let source = std::path::Path::new("embedded-source.cmf");
        let error = compact_destination(
            source,
            source,
            std::path::Path::new("candidate.mtp.cmf"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("legacy sidecar-only"), "{error}");
    }

    #[test]
    fn malformed_plan_topology_is_rejected_before_source_verification() {
        // If this reaches `verify_cmf`, the deliberately invalid source would
        // produce a CMF parse/hash error instead. The topology error proves a
        // costly 100B source scan and `.tdata` open are both skipped.
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("not-even-a-cmf.cmf");
        fs::write(&source, b"not a CMF").unwrap();
        let candidate = temp.path().join("candidate.cmf");
        let nested_plan = candidate.join("plan.json");
        let error = cmd_mtp_compact(
            source.to_str().unwrap(),
            temp.path().join("unread.tdata").to_str().unwrap(),
            candidate.to_str().unwrap(),
            0.5,
            Some(nested_plan.to_str().unwrap()),
            1,
            4,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("candidate output and --plan-out"), "{error}");
        assert!(!candidate.exists(), "topology preflight must not create output");
    }

    #[test]
    fn sidecar_shaped_output_preflight_is_nonclobber_and_defers_topology() {
        // Whether the historical sidecar-only spelling is valid depends on
        // whether MODEL actually has an external MiMo draft. That topology is
        // discovered only after opening the source, but the cheap preflight
        // must still reject collisions and never create leaves.
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.cmf");
        let output = temp.path().join("candidate.mtp.cmf");
        let plan = temp.path().join("candidate.mtp.cmf.mtp-plan.json");
        preflight_explicit_paths(&source, &output, &plan).unwrap();
        assert!(!output.exists(), "preflight must not create a sidecar output");
        assert!(!plan.exists(), "preflight must not create a plan output");
    }

    #[test]
    fn nested_plan_path_is_rejected_before_data_or_output_creation() {
        // A plan below `candidate.cmf/` would otherwise make a directory at
        // the candidate's final filename during `prepare_new_file(plan)`,
        // only to fail after the expensive bake at publication time.
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.cmf");
        let source_sidecar = temp.path().join("source.mtp.cmf");
        let candidate = temp.path().join("candidate.cmf");
        let nested_plan = candidate.join("plan.json");
        CmfModel::write(&source, &mimo_test_header(), &mimo_test_main_tensors(), None, None)
            .unwrap();
        CmfModel::write(
            &source_sidecar,
            &mimo_test_header(),
            &mimo_test_sidecar_tensors(),
            None,
            None,
        )
        .unwrap();

        let error = cmd_mtp_compact(
            source.to_str().unwrap(),
            temp.path().join("unread.tdata").to_str().unwrap(),
            candidate.to_str().unwrap(),
            0.5,
            Some(nested_plan.to_str().unwrap()),
            1,
            4,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--plan-out"), "{error}");
        assert!(
            !candidate.exists(),
            "nested plan must not create a directory at the candidate CMF leaf"
        );

        // This also covers a non-existent lexical path which resolves to the
        // same final filename only after its `..` component is considered.
        assert!(paths_overlap_or_nested(
            &temp.path().join("not-created/../candidate.cmf"),
            &candidate,
        )
        .unwrap());
    }

    #[test]
    fn invalid_keep_ratio_is_rejected_before_opening_calibration_data() {
        // A source draft has enough geometry to reject a useless compact
        // width without opening the deliberately absent `.tdata` selector.
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.cmf");
        let source_sidecar = temp.path().join("source.mtp.cmf");
        let candidate = temp.path().join("must-not-exist/candidate.cmf");
        CmfModel::write(&source, &mimo_test_header(), &mimo_test_main_tensors(), None, None)
            .unwrap();
        CmfModel::write(
            &source_sidecar,
            &mimo_test_header(),
            &mimo_test_sidecar_tensors(),
            None,
            None,
        )
        .unwrap();
        let error = cmd_mtp_compact(
            source.to_str().unwrap(),
            temp.path().join("unread.tdata").to_str().unwrap(),
            candidate.to_str().unwrap(),
            0.01,
            None,
            1,
            4,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--keep-ratio"), "{error}");
        assert!(
            !candidate.parent().unwrap().exists(),
            "invalid keep ratio must fail before output parent creation"
        );
    }

    #[test]
    fn unsupported_mtp_codec_is_rejected_before_opening_calibration_data() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.cmf");
        let source_sidecar = temp.path().join("source.mtp.cmf");
        let candidate = temp.path().join("must-not-exist/candidate.cmf");
        CmfModel::write(&source, &mimo_test_header(), &mimo_test_main_tensors(), None, None)
            .unwrap();
        let mut sidecar = mimo_test_sidecar_tensors();
        let gate = sidecar
            .iter_mut()
            .find(|tensor| tensor.name == "model.mtp.layers.0.mlp.gate_proj.weight")
            .expect("test sidecar contains MTP gate");
        gate.dtype = TensorDtype::U8;
        gate.data = vec![0; gate.shape.iter().product()];
        CmfModel::write(&source_sidecar, &mimo_test_header(), &sidecar, None, None).unwrap();

        let error = cmd_mtp_compact(
            source.to_str().unwrap(),
            temp.path().join("unread.tdata").to_str().unwrap(),
            candidate.to_str().unwrap(),
            0.5,
            None,
            1,
            4,
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("physical compact writer cannot materialize"), "{error}");
        assert!(
            !candidate.parent().unwrap().exists(),
            "unsupported codec must fail before output parent creation"
        );
    }

    #[test]
    fn mimo_external_sidecar_compacts_to_a_runnable_candidate_pair() {
        // Exercise the actual production route rather than merely checking
        // that a derived filename looks right: the main has no MTP tensors,
        // so the staged main must discover the compacted sibling sidecar
        // during runtime admission. This is the layout used by MiMo-V2.
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.cmf");
        let source_sidecar = temp.path().join("source.mtp.cmf");
        let data = temp.path().join("calibration.tdata");
        let candidate = temp.path().join("candidate.cmf");
        let candidate_sidecar = temp.path().join("candidate.mtp.cmf");
        let plan = temp.path().join("candidate.cmf.mtp-plan.json");

        let header = mimo_test_header();
        CmfModel::write(&source, &header, &mimo_test_main_tensors(), None, None).unwrap();
        CmfModel::write(
            &source_sidecar,
            &header,
            &mimo_test_sidecar_tensors(),
            None,
            None,
        )
        .unwrap();
        write_mimo_test_tdata(&data);

        let too_small = temp.path().join("too-small.cmf");
        let error = cmd_mtp_compact(
            source.to_str().unwrap(),
            data.to_str().unwrap(),
            too_small.to_str().unwrap(),
            0.5,
            None,
            1,
            2,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--max-record-tokens"), "{error}");
        assert!(!too_small.exists());

        cmd_mtp_compact(
            source.to_str().unwrap(),
            data.to_str().unwrap(),
            candidate.to_str().unwrap(),
            0.5,
            None,
            1,
            4,
        )
        .unwrap();

        assert!(candidate.exists());
        assert!(candidate_sidecar.exists());
        assert!(plan.exists());
        let compact_draft = CmfModel::open(&candidate_sidecar).unwrap();
        assert!(
            compact_draft.verify().is_empty(),
            "{:?}",
            compact_draft.verify()
        );
        assert_eq!(
            compact_draft
                .arch()
                .mtp
                .as_ref()
                .and_then(|config| config.ffn_intermediate_sizes.as_deref()),
            Some(&[32][..])
        );
        assert_eq!(
            compact_draft
                .tensor("model.mtp.layers.0.mlp.gate_proj.weight")
                .unwrap()
                .shape,
            vec![32, MIMO_TEST_HIDDEN]
        );
        assert_eq!(
            compact_draft
                .tensor("model.mtp.layers.0.mlp.down_proj.weight")
                .unwrap()
                .shape,
            vec![MIMO_TEST_HIDDEN, 32]
        );

        let candidate_main = Arc::new(CmfModel::open(&candidate).unwrap());
        assert!(
            candidate_main.verify().is_empty(),
            "{:?}",
            candidate_main.verify()
        );
        let mut runtime = Pipeline::from_model(&candidate_main, SamplerConfig::default()).unwrap();
        assert!(
            runtime.mimo_mtp.is_some(),
            "candidate must find its sidecar"
        );
        let probe = runtime.probe_mtp_ffn(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(probe.len(), 1);
        assert_eq!(probe[0].mass.len(), 32);
        assert_eq!(probe[0].squared_mass.len(), 32);
        assert!(probe[0].observed_positions > 0);

        let plan: serde_json::Value = serde_json::from_slice(&fs::read(&plan).unwrap()).unwrap();
        assert_eq!(plan["selection"]["calibration_records"], 1);
        assert_eq!(plan["selection"]["calibration_selected_records"], 1);
        assert_eq!(plan["selection"]["calibration_chunks"], 3);
        // Nine tokens under cap four become 4+3+2, not 4+4+1 with the
        // singleton silently discarded. Admission must retain the longest
        // eligible bounded chunk (four), not a later minimal tail.
        assert_eq!(plan["selection"]["calibration_tokens"], 9);
        assert_eq!(plan["selection"]["max_record_tokens"], 4);
        assert_eq!(
            plan["selection"]["calibration_chunking"],
            "fresh-causal-sequence-per-contiguous-chunk-v1"
        );
        assert_eq!(plan["selection"]["admission_probe_tokens"], 4);

        // 0.8.37 accepted a sidecar output directly. Keep that compatibility
        // contract, but validate it via a private staged pair and publish only
        // the requested draft, never a surprise backbone at legacy.cmf.
        let legacy_draft = temp.path().join("legacy.mtp.cmf");
        let legacy_main = temp.path().join("legacy.cmf");
        let legacy_plan = temp.path().join("legacy.mtp.cmf.mtp-plan.json");
        cmd_mtp_compact(
            source.to_str().unwrap(),
            data.to_str().unwrap(),
            legacy_draft.to_str().unwrap(),
            0.5,
            None,
            1,
            4,
        )
        .unwrap();
        assert!(legacy_draft.exists());
        assert!(legacy_plan.exists());
        assert!(
            !legacy_main.exists(),
            "legacy sidecar-only output must not create a new backbone"
        );
        let legacy = CmfModel::open(&legacy_draft).unwrap();
        assert!(legacy.verify().is_empty(), "{:?}", legacy.verify());
        let legacy_plan: serde_json::Value =
            serde_json::from_slice(&fs::read(&legacy_plan).unwrap()).unwrap();
        assert_eq!(
            legacy_plan["selection"]["output_layout"],
            "legacy-sidecar-only-v1"
        );
        // The immutable source pair remains usable; `mtp-compact` produced a
        // separate main+sidecar candidate instead of editing either input.
        assert!(CmfModel::open(&source).unwrap().verify().is_empty());
        assert!(CmfModel::open(&source_sidecar).unwrap().verify().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn dangling_output_symlink_is_refused() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let dangling = temp.path().join("candidate.cmf");
        symlink("missing-target.cmf", &dangling).unwrap();
        assert!(prepare_new_file(&dangling, "candidate main").is_err());
        assert!(std::fs::symlink_metadata(&dangling).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn hard_linked_backbone_resolves_a_source_symlink_first() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let real = temp.path().join("real.cmf");
        let source_link = temp.path().join("source.cmf");
        let staged = temp.path().join("staged.cmf");
        fs::write(&real, b"immutable source").unwrap();
        symlink(&real, &source_link).unwrap();
        assert_eq!(stage_backbone(&source_link, &staged).unwrap(), "hard-link");
        assert!(
            !fs::symlink_metadata(&staged)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&staged).unwrap(), b"immutable source");
    }

    #[test]
    fn candidate_and_plan_paths_never_overwrite_existing_files() {
        let temp = tempfile::tempdir().unwrap();
        let existing = temp.path().join("existing.cmf");
        fs::write(&existing, b"keep").unwrap();
        assert!(prepare_new_file(&existing, "candidate main").is_err());
        assert_eq!(fs::read(&existing).unwrap(), b"keep");

        let fresh = temp.path().join("nested/fresh.cmf");
        prepare_new_file(&fresh, "candidate main").unwrap();
        assert!(!fresh.exists());
    }

    #[test]
    fn validated_stage_publishes_without_clobbering() {
        let temp = tempfile::tempdir().unwrap();
        let staged = temp.path().join("stage.cmf");
        let output = temp.path().join("candidate.cmf");
        fs::write(&staged, b"validated").unwrap();
        publish_new_file(&staged, &output, "candidate main").unwrap();
        assert_eq!(fs::read(&output).unwrap(), b"validated");
        assert!(!staged.exists());

        let staged_again = temp.path().join("stage-again.cmf");
        fs::write(&staged_again, b"new").unwrap();
        assert!(publish_new_file(&staged_again, &output, "candidate main").is_err());
        assert_eq!(fs::read(&output).unwrap(), b"validated");
        assert!(staged_again.exists());
    }

    #[test]
    fn source_and_alias_output_are_detected_as_the_same_path() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("draft.cmf");
        fs::write(&source, b"cmf").unwrap();
        let alias = temp.path().join(".").join("draft.cmf");
        assert!(paths_collide(&source, &alias));
        assert!(!paths_collide(&source, &temp.path().join("new-output.cmf")));
    }
}
