//! True tail append (spec §9.3; Patent 15 cl.3): a skill enters an existing
//! file WITHOUT rewriting a byte of it.
//!
//! Layout after an append:
//!
//! ```text
//! [0, 128)            envelope — rewritten IN PLACE, last
//! [128, old_len)      untouched: old header, old directory, trunk blob,
//!                     masks, vocab, index (all still addressed by offset)
//! [old_len, …)        new tensor payloads (64-aligned relative to data_off,
//!                     large ones page-aligned), then the new header JSON and
//!                     the new directory (old entries byte-identical, new
//!                     entries appended)
//! ```
//!
//! `data_len` grows to cover the new payloads, so the old masks/vocab/index
//! sections now lie INSIDE `[data_off, data_off + data_len)`. That is legal:
//! every reader addresses sections only through the envelope, bounds are
//! checked against the file and the data length only, and no reader checks
//! that sections are disjoint (`parse_envelope`; `python/cmf_reader.py` does
//! the same).
//!
//! Crash ordering: payloads → header → directory → fsync → envelope →
//! fsync. Until the 128-byte envelope is written the file IS the old file
//! (the tail is unreferenced bytes), so a crash at any point leaves either
//! the old or the new consistent file. [`PendingAppend`] exposes the two
//! phases so the "killed before the envelope" case is testable.

use super::*;
use crate::knowledge::{
    KNOWLEDGE_BITS, seal_genome, skill_kind, validate_expert_append_values, validate_knowledge,
};
use std::fs::OpenOptions;

/// What a tail append did.
#[derive(Debug, Clone)]
pub struct AppendReport {
    pub path: PathBuf,
    /// File length before the append: bytes `[128, old_len)` are unchanged.
    pub old_len: u64,
    pub new_len: u64,
    pub data_len_before: u64,
    pub data_len_after: u64,
    pub tensors_added: usize,
    /// The data segment the append added (None for header-only updates).
    pub segment: Option<Segment>,
    pub required_features: u32,
    pub header_hash: u64,
    pub dir_hash: u64,
    /// `genome.trunk_hash` of the result (unchanged by construction).
    pub trunk_hash: Option<u64>,
}

/// An append whose tail is written and fsynced but whose envelope is not:
/// the file on disk is still the OLD file. [`PendingAppend::commit`]
/// publishes it; dropping it abandons the tail (unreferenced bytes).
pub struct PendingAppend {
    file: File,
    envelope: [u8; ENVELOPE_LEN],
    old_envelope: [u8; ENVELOPE_LEN],
    report: AppendReport,
}

impl PendingAppend {
    pub fn report(&self) -> &AppendReport {
        &self.report
    }

    /// Rewrite the envelope in place and fsync, then re-open the result
    /// with full validation. Should the re-open fail (a bug — the content
    /// was validated before the tail was written), the old envelope is
    /// restored and the error returned: the file is the old file again.
    pub fn commit(mut self) -> Result<AppendReport, CmfError> {
        write_envelope(&mut self.file, &self.envelope)?;
        match CmfModel::open(&self.report.path) {
            Ok(_) => Ok(self.report),
            Err(e) => {
                write_envelope(&mut self.file, &self.old_envelope)?;
                Err(CmfError::Parse(format!(
                    "append rolled back — the committed file did not re-open: {e}"
                )))
            }
        }
    }
}

fn write_envelope(file: &mut File, env: &[u8; ENVELOPE_LEN]) -> Result<(), CmfError> {
    file.seek(SeekFrom::Start(0))?;
    file.write_all(env)?;
    file.sync_all()?;
    Ok(())
}

/// One writer at a time (advisory `flock`, released when the handle drops).
///
/// A held lock is retried for up to 2 s before it counts as another writer:
/// a flock lives on the open file description, so a child a sibling thread
/// forks between our previous append's close and this open keeps a copy of
/// the description (and the lock) until its exec closes it. A parallel test
/// harness spawning the CLI hit exactly that on macOS ("another writer holds
/// the append lock" right after our own append). A real concurrent writer
/// still gets the error.
fn lock_exclusive(file: &File, path: &Path) -> Result<(), CmfError> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        const TRIES: u32 = 50;
        for attempt in 0..TRIES {
            // SAFETY: plain fd + flags; advisory lock on an open handle.
            let r = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if r == 0 {
                break;
            }
            let err = std::io::Error::last_os_error();
            let busy = err.raw_os_error() == Some(libc::EWOULDBLOCK);
            if !busy || attempt + 1 == TRIES {
                return Err(CmfError::Parse(format!(
                    "{}: another writer holds the append lock{}",
                    path.display(),
                    if busy {
                        String::new()
                    } else {
                        format!(" ({err})")
                    }
                )));
            }
            std::thread::sleep(std::time::Duration::from_millis(40));
        }
    }
    #[cfg(not(unix))]
    let _ = (file, path);
    Ok(())
}

/// Open for append: take the lock, then validate the current file fully.
fn open_for_append(path: &Path) -> Result<(File, CmfModel, u64), CmfError> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    lock_exclusive(&file, path)?;
    let model = CmfModel::open(path)?;
    if model.header.shard.is_some() {
        return Err(CmfError::Parse(format!(
            "{}: tail append into a sharded file is not supported",
            path.display()
        )));
    }
    if model.required_features & features::SKILL_FILE != 0 {
        return Err(CmfError::Parse(format!(
            "{}: a standalone SKILL_FILE takes no appends — append into a runnable model",
            path.display()
        )));
    }
    let file_len = file.metadata()?.len();
    Ok((file, model, file_len))
}

fn envelope_bytes(model: &CmfModel) -> [u8; ENVELOPE_LEN] {
    let mut e = [0u8; ENVELOPE_LEN];
    e.copy_from_slice(&model.primary_bytes()[..ENVELOPE_LEN]);
    e
}

/// Recompute the bits for new content: every bit the old file carried
/// stays (a conservative writer may have set more than content implies),
/// the three knowledge bits follow content exactly.
fn appended_bits(old: u32, header: &CmfHeader, entries: &[TensorEntry], masks: bool) -> u32 {
    let derived = derive_required_features(header, entries, masks);
    ((old | derived) & !KNOWLEDGE_BITS) | (derived & KNOWLEDGE_BITS)
}

fn next_lineage(
    header: &CmfHeader,
    ev: Option<LineageEvent>,
    default: &str,
    detail: serde_json::Value,
) -> LineageEvent {
    let seq = header.lineage.last().map(|e| e.seq + 1).unwrap_or(0);
    match ev {
        Some(mut e) => {
            e.seq = seq;
            if e.event.is_empty() {
                e.event = default.into();
            }
            if e.ts.is_empty() {
                e.ts = crate::knowledge::utc_now_rfc3339();
            }
            if e.detail.is_null() {
                e.detail = detail;
            }
            e
        }
        None => LineageEvent::now(seq, default, detail),
    }
}

impl CmfModel {
    /// Append one skill record and its tensors to the END of `path`
    /// (spec §9.3): the prefix `[128, old_len)` stays byte-identical, old
    /// directory entries keep their offsets and hashes, the new header and
    /// directory are published by rewriting the envelope last.
    ///
    /// Refuses a skill id that already exists (re-bake = new id or explicit
    /// retire), tensors outside `skill.{id}.`, records bound to a base
    /// directory (`base_dir_hash` — those are standalone skill files), and
    /// anything `open()` would refuse afterwards (validated before a byte is
    /// written). `router` / `routing`: `Some` replaces the header's value,
    /// `None` keeps it (a stale router falls back to the backbone by its
    /// skills hash). A `skill_committed` lineage event is pushed (the
    /// caller's event if given; its `seq` is assigned here). For an
    /// `ffn_replace` record without `state_effect`, the writer computes it
    /// ([`ffn_replace_state_effect`]).
    pub fn append_skill(
        path: impl AsRef<Path>,
        record: SkillRecord,
        tensors: &[TensorSpec],
        router: Option<RouterPolicy>,
        routing: Option<RoutingCalibration>,
        lineage_event: Option<LineageEvent>,
    ) -> Result<AppendReport, CmfError> {
        Self::prepare_append_skill(path, record, tensors, router, routing, lineage_event)?.commit()
    }

    /// Phase 1 of [`Self::append_skill`]: payloads, header and directory
    /// written at the tail and fsynced; the envelope is NOT touched, so the
    /// file on disk still opens as the old one until
    /// [`PendingAppend::commit`].
    pub fn prepare_append_skill(
        path: impl AsRef<Path>,
        mut record: SkillRecord,
        tensors: &[TensorSpec],
        router: Option<RouterPolicy>,
        routing: Option<RoutingCalibration>,
        lineage_event: Option<LineageEvent>,
    ) -> Result<PendingAppend, CmfError> {
        let path = path.as_ref();
        let (mut file, model, file_len) = open_for_append(path)?;
        let env = model.envelope;
        let id = record.id.clone();

        if id.is_empty() {
            return Err(CmfError::Parse("append: the skill record has no id".into()));
        }
        if record.base_dir_hash.is_some() {
            return Err(CmfError::Parse(format!(
                "append: skill '{id}' carries base_dir_hash — that makes a standalone skill \
                 file, not a record inside a model"
            )));
        }
        if model.header.skills.iter().any(|s| s.id == id) {
            return Err(CmfError::Parse(format!(
                "append: skill '{id}' already exists — re-bake under a new id or retire it"
            )));
        }
        let prefix = format!("skill.{id}.");
        if model.tensors.iter().any(|t| t.name.starts_with(&prefix)) {
            return Err(CmfError::Parse(format!(
                "append: the directory already holds '{prefix}*' tensors"
            )));
        }
        if !tensors.is_empty() && (env.data.0 % DATA_ALIGNMENT != 0 || env.data.0 > file_len) {
            return Err(CmfError::Parse(format!(
                "append: data section offset {} cannot anchor tail payloads",
                env.data.0
            )));
        }

        // Layout of the new payloads, relative to data_off.
        let data_off = env.data.0;
        let mut cursor = file_len.saturating_sub(data_off).max(env.data.1);
        let mut entries = model.tensors.clone();
        let mut new_entries = Vec::with_capacity(tensors.len());
        for t in tensors {
            if !t.name.starts_with(&prefix) || t.name.len() == prefix.len() {
                return Err(CmfError::Parse(format!(
                    "append: tensor '{}' is outside the skill's namespace '{prefix}'",
                    t.name
                )));
            }
            if t.shape.len() > DIR_MAX_NDIM {
                return Err(CmfError::Parse(format!(
                    "tensor '{}': ndim {} > {DIR_MAX_NDIM}",
                    t.name,
                    t.shape.len()
                )));
            }
            if let Some(expect) = expected_nbytes(t.dtype, &t.shape) {
                if expect != t.data.len() {
                    return Err(CmfError::Bounds(format!(
                        "tensor '{}': data {} bytes != expected {expect} for {:?}{:?}",
                        t.name,
                        t.data.len(),
                        t.dtype,
                        t.shape
                    )));
                }
            }
            if new_entries.iter().any(|e: &TensorEntry| e.name == t.name) {
                return Err(CmfError::Parse(format!(
                    "append: duplicate tensor '{}'",
                    t.name
                )));
            }
            let align = if t.data.len() as u64 >= LARGE_TENSOR_MIN {
                LARGE_TENSOR_ALIGN
            } else {
                TENSOR_ALIGNMENT
            };
            let off = align_to(cursor, align);
            new_entries.push(TensorEntry {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                off,
                nbytes: t.data.len() as u64,
                shard: 0,
                hash: hash64(&t.data),
            });
            cursor = off + t.data.len() as u64;
        }
        let data_len_before = env.data.1;
        let data_len_after = if tensors.is_empty() {
            env.data.1
        } else {
            cursor
        };
        entries.extend(new_entries.iter().cloned());

        // New header.
        let mut header = model.header.clone();
        if record.state_effect.is_none() {
            match record.kind.as_deref() {
                Some(skill_kind::FFN_REPLACE) => {
                    record.state_effect =
                        Some(ffn_replace_state_effect(&header.arch, &record.layers));
                }
                Some(skill_kind::EXPERT_APPEND) => {
                    record.state_effect = Some(expert_append_state_effect(&record.layers));
                }
                _ => {}
            }
        }
        header.skills.push(record);
        if let Some(r) = router {
            header.router = Some(r);
        }
        if let Some(c) = routing {
            header.routing = Some(c);
        }
        if header.segments.is_empty() {
            let (kind, sid) = match &header.genome {
                Some(g) => ("genome", g.id.clone()),
                None => ("base", header.arch.arch_name.clone()),
            };
            header.segments.push(Segment {
                kind: kind.into(),
                id: sid,
                data_start: 0,
                data_end: data_len_before,
            });
        }
        let segment = new_entries.first().map(|first| Segment {
            kind: "skill".into(),
            id: id.clone(),
            data_start: first.off,
            data_end: data_len_after,
        });
        if let Some(s) = &segment {
            header.segments.push(s.clone());
        }
        let detail = serde_json::json!({
            "skill": id,
            "tensors": new_entries.len(),
            "data": segment.as_ref().map(|s| [s.data_start, s.data_end]),
        });
        let ev = next_lineage(&header, lineage_event, "skill_committed", detail);
        header.lineage.push(ev);

        let vocab = model.vocab.as_deref();
        // The mask / sparse-index sections stay where they are: same bytes.
        // `exec_hashes()`, not the `exec` field: `open()` fills the field
        // only for GENOME files, and a header update may BIRTH the genome
        // on a file that had none — sealed without its mask section, the
        // committed file would re-open with "trunk_hash mismatch" (NF-1).
        let exec = model.exec_hashes();
        seal_genome(&mut header, &entries, vocab, exec)?;
        let bits = appended_bits(env.required_features, &header, &entries, env.masks.1 > 0);
        header
            .arch
            .validate_operator_metadata()
            .map_err(CmfError::Parse)?;
        validate_knowledge(
            &header,
            &entries,
            vocab,
            bits,
            Some(data_len_after),
            None,
            exec,
        )?;
        crate::knowledge::check_lookup_policies(&header)?;
        // Descriptor values of the new record from the payloads given;
        // earlier records passed this when they were written.
        let n_old = model.tensors.len();
        validate_expert_append_values(&header, &entries, |e| {
            let i = entries.iter().position(|x| std::ptr::eq(x, e))?;
            match i.checked_sub(n_old) {
                Some(j) => crate::knowledge::f32_le_head(&tensors[j].data),
                None => crate::knowledge::f32_le_head(model.entry_bytes(e)),
            }
        })?;
        // Lookup tables of the new record from the payloads given, of the
        // earlier records from the file.
        crate::knowledge::validate_lookup_values(&header, &entries, true, |e| {
            let i = entries.iter().position(|x| std::ptr::eq(x, e))?;
            Some(std::borrow::Cow::Borrowed(match i.checked_sub(n_old) {
                Some(j) => tensors[j].data.as_slice(),
                None => model.entry_bytes(e),
            }))
        })?;

        let header_json =
            serde_json::to_vec(&header).map_err(|e| CmfError::Parse(format!("header: {e}")))?;
        let dir_bytes = Self::encode_directory(&entries);
        let old_envelope = envelope_bytes(&model);
        let trunk = header
            .genome
            .as_ref()
            .and_then(|g| crate::knowledge::parse_hex64(&g.trunk_hash));
        drop(model);

        // Tail: payloads, header, directory — then fsync.
        file.seek(SeekFrom::Start(file_len))?;
        let (header_off, dir_off, end) = {
            let mut w = BufWriter::new(&mut file);
            let mut pos = file_len;
            for (spec, entry) in tensors.iter().zip(&new_entries) {
                let target = data_off + entry.off;
                w.write_all(&zeros((target - pos) as usize))?;
                w.write_all(&spec.data)?;
                pos = target + spec.data.len() as u64;
            }
            let header_off = pos;
            w.write_all(&header_json)?;
            let dir_off = header_off + header_json.len() as u64;
            w.write_all(&dir_bytes)?;
            w.flush()?;
            (header_off, dir_off, dir_off + dir_bytes.len() as u64)
        };
        file.sync_all()?;

        let mut envelope = old_envelope;
        let put = |e: &mut [u8; ENVELOPE_LEN], at: usize, v: u64| {
            e[at..at + 8].copy_from_slice(&v.to_le_bytes())
        };
        envelope[12..16].copy_from_slice(&bits.to_le_bytes());
        put(&mut envelope, 0x10, header_off);
        put(&mut envelope, 0x18, header_json.len() as u64);
        put(&mut envelope, 0x20, dir_off);
        put(&mut envelope, 0x28, dir_bytes.len() as u64);
        put(&mut envelope, 0x38, data_len_after);
        let (header_hash, dir_hash) = (hash64(&header_json), hash64(&dir_bytes));
        put(&mut envelope, 0x70, header_hash);
        put(&mut envelope, 0x78, dir_hash);

        Ok(PendingAppend {
            file,
            envelope,
            old_envelope,
            report: AppendReport {
                path: path.to_path_buf(),
                old_len: file_len,
                new_len: end,
                data_len_before,
                data_len_after,
                tensors_added: new_entries.len(),
                segment,
                required_features: bits,
                header_hash,
                dir_hash,
                trunk_hash: trunk,
            },
        })
    }

    /// Header-only update through the same tail mechanism (router
    /// recalibration, status/gate updates, lineage events): the new header
    /// JSON goes to the end of the file, the directory and every payload
    /// stay where they are, the envelope is rewritten last. `section_hashes`
    /// and `segments` are layout facts and are kept from the current file
    /// whatever `f` does; everything else must still pass `open()`'s rules
    /// (a trunk change is refused by the genome hash).
    pub fn update_header_append(
        path: impl AsRef<Path>,
        f: impl FnOnce(&mut CmfHeader),
    ) -> Result<AppendReport, CmfError> {
        Self::prepare_update_header_append(path, f)?.commit()
    }

    /// Phase 1 of [`Self::update_header_append`] (see
    /// [`Self::prepare_append_skill`]).
    pub fn prepare_update_header_append(
        path: impl AsRef<Path>,
        f: impl FnOnce(&mut CmfHeader),
    ) -> Result<PendingAppend, CmfError> {
        let path = path.as_ref();
        let (mut file, model, file_len) = open_for_append(path)?;
        let env = model.envelope;
        let mut header = model.header.clone();
        f(&mut header);
        header.section_hashes = model.header.section_hashes.clone();
        header.segments = model.header.segments.clone();
        if header.shard.is_some() {
            return Err(CmfError::Parse(
                "update_header_append: cannot make a file sharded".into(),
            ));
        }
        // The genome's identity is frozen: a header update may move its
        // status or reference battery, never drop it or re-point it.
        if let Some(old) = &model.header.genome {
            let same = header.genome.as_ref().is_some_and(|g| {
                g.id == old.id
                    && g.generation == old.generation
                    && g.trunk_hash == old.trunk_hash
                    && g.master_trunk_hash == old.master_trunk_hash
                    && g.encoding == old.encoding
                    && g.parent == old.parent
            });
            if !same {
                return Err(CmfError::Parse(format!(
                    "update_header_append: genome '{}' identity (id, generation, hashes, \
                     encoding, parent) cannot change or be removed by a header update — a \
                     new genome is a new file",
                    old.id
                )));
            }
        }
        let vocab = model.vocab.as_deref();
        // The mask / sparse-index sections stay where they are: same bytes.
        // `exec_hashes()`, not the `exec` field: `open()` fills the field
        // only for GENOME files, and a header update may BIRTH the genome
        // on a file that had none — sealed without its mask section, the
        // committed file would re-open with "trunk_hash mismatch" (NF-1).
        let exec = model.exec_hashes();
        seal_genome(&mut header, &model.tensors, vocab, exec)?;
        let bits = appended_bits(
            env.required_features,
            &header,
            &model.tensors,
            env.masks.1 > 0,
        );
        header
            .arch
            .validate_operator_metadata()
            .map_err(CmfError::Parse)?;
        validate_knowledge(
            &header,
            &model.tensors,
            vocab,
            bits,
            Some(env.data.1),
            None,
            exec,
        )?;
        crate::knowledge::check_lookup_policies(&header)?;
        let header_json =
            serde_json::to_vec(&header).map_err(|e| CmfError::Parse(format!("header: {e}")))?;
        let old_envelope = envelope_bytes(&model);
        let trunk = header
            .genome
            .as_ref()
            .and_then(|g| crate::knowledge::parse_hex64(&g.trunk_hash));
        drop(model);

        file.seek(SeekFrom::Start(file_len))?;
        file.write_all(&header_json)?;
        file.sync_all()?;

        let mut envelope = old_envelope;
        envelope[12..16].copy_from_slice(&bits.to_le_bytes());
        envelope[0x10..0x18].copy_from_slice(&file_len.to_le_bytes());
        envelope[0x18..0x20].copy_from_slice(&(header_json.len() as u64).to_le_bytes());
        let header_hash = hash64(&header_json);
        envelope[0x70..0x78].copy_from_slice(&header_hash.to_le_bytes());

        Ok(PendingAppend {
            file,
            envelope,
            old_envelope,
            report: AppendReport {
                path: path.to_path_buf(),
                old_len: file_len,
                new_len: file_len + header_json.len() as u64,
                data_len_before: env.data.1,
                data_len_after: env.data.1,
                tensors_added: 0,
                segment: None,
                required_features: bits,
                header_hash,
                dir_hash: env.dir_hash,
                trunk_hash: trunk,
            },
        })
    }
}

/// What [`CmfModel::migrate_legacy_embryo_bits`] found (and, unless it was
/// a dry run, did).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyBitsMigration {
    /// Not a CMF file.
    NotCmf,
    /// No pre-0.8.1 Embryo bits: nothing to move (already migrated, a real
    /// Prism file, or a file that never carried them).
    NothingToDo { required_features: u32 },
    /// Bits 7/8 moved to 12/13 (`from` → `to`). A dry run reports the move
    /// without writing it.
    Migrated { from: u32, to: u32 },
}

impl CmfModel {
    /// Move the `BOUNDED_STATE`/`GENOME` bits that a pre-0.8.1 embryo-o1
    /// build wrote on 7/8 (today's `PRISM_HADAMARD`/`PRISM_AFFINE`) to
    /// 12/13, IN PLACE: only envelope bytes 12..16 change. Neither the
    /// header/directory hashes nor `trunk_hash` cover them, so every
    /// appended record, segment and lineage event survives — which a
    /// re-export would drop. A detached `.sig` signs the whole file and
    /// must be renewed afterwards.
    ///
    /// Decided on content: bit 7 must match `arch.anchor_core`, bit 8 must
    /// match `genome`, and a header with `arch.prism_hadamard` is a real
    /// Prism file that is left alone. Takes the append lock. The result is
    /// NOT re-validated here: the caller opens the file with
    /// [`CmfModel::open`], which may still refuse a file written before
    /// later format rules.
    pub fn migrate_legacy_embryo_bits(
        path: impl AsRef<Path>,
        dry_run: bool,
    ) -> Result<LegacyBitsMigration, CmfError> {
        use super::legacy_embryo_bits as legacy;
        use std::io::{Seek, SeekFrom, Write};
        let path = path.as_ref();
        let file = if dry_run {
            File::open(path)?
        } else {
            let f = OpenOptions::new().read(true).write(true).open(path)?;
            lock_exclusive(&f, path)?;
            f
        };
        let Some((required, header)) = peek_header(path)? else {
            return Ok(LegacyBitsMigration::NotCmf);
        };
        let set = |v: Option<&serde_json::Value>| v.is_some_and(|x| !x.is_null());
        let arch = header.get("arch");
        let has_prism = set(arch.and_then(|a| a.get("prism_hadamard")));
        let has_anchor = set(arch.and_then(|a| a.get("anchor_core")));
        let has_genome = set(header.get("genome"));
        if required & legacy::BOTH == 0 || has_prism {
            return Ok(LegacyBitsMigration::NothingToDo {
                required_features: required,
            });
        }
        if !has_anchor && !has_genome {
            return Err(CmfError::Parse(format!(
                "{}: bits {:#x} are set but the header has no prism_hadamard, anchor_core or \
                 genome — not a pre-0.8.1 Embryo file, refusing to guess",
                path.display(),
                required & legacy::BOTH
            )));
        }
        let bit7 = required & legacy::BOUNDED_STATE != 0;
        let bit8 = required & legacy::GENOME != 0;
        if bit7 != has_anchor || bit8 != has_genome {
            return Err(CmfError::Parse(format!(
                "{}: legacy bits disagree with the header (bit 7={bit7} vs anchor_core={has_anchor}, \
                 bit 8={bit8} vs genome={has_genome}) — refusing to guess",
                path.display()
            )));
        }
        if required & (features::BOUNDED_STATE | features::GENOME) != 0 {
            return Err(CmfError::Parse(format!(
                "{}: carries both the old (7/8) and the new (12/13) bits ({required:#x}) — \
                 refusing to guess",
                path.display()
            )));
        }
        let mut to = required & !legacy::BOTH;
        if has_anchor {
            to |= features::BOUNDED_STATE;
        }
        if has_genome {
            to |= features::GENOME;
        }
        if !dry_run {
            let mut f = file;
            f.seek(SeekFrom::Start(12))?;
            f.write_all(&to.to_le_bytes())?;
            f.sync_all()?;
        }
        Ok(LegacyBitsMigration::Migrated { from: required, to })
    }
}
