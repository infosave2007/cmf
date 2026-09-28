//! `cortiq lookup-build`, `cortiq lookup-policy` and `cortiq route-fit` —
//! the builder of a `lookup` record (spec §9.5.2: an explicit key → card
//! table appended to a COPY of the sealed genome), the header-only switch
//! of its routing policy, and the runtime-side fit of the request router
//! (spec §9.4) for one skill. Neither involves the trainer: the
//! table comes from a JSONL corpus, the descriptors from φ the RUNTIME
//! computes (`Pipeline::probe_phi_span` over the canonical cmf-im-v1
//! span — exactly what `route_request` scores at inference), and every
//! write is a tail append (`CmfModel::append_skill`,
//! `update_header_append`): the trunk bytes never move.

use crate::knowledge::{phi_backend_label, sha256_hex};
use anyhow::Context;
use cortiq_core::knowledge::{
    KEY_NORM, LineageEvent, LookupInfo, SkillBound, hex64, lookup_state_effect, lookup_tensors,
    normalize_key, normalized_key_hash, skill_kind,
};
use cortiq_core::{CmfHeader, CmfModel, PhiSpec, RouterPolicy, SelectionDescriptor, SkillRecord};
use cortiq_engine::lookup::{
    LookupPolicy, LookupTable, MAX_NGRAM, MatchVia, StemIndex, extract_key_with, stem_key,
};
use cortiq_engine::router::{self, RouteOptions};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

// ───────────────────────── lookup-build ─────────────────────────

pub struct LookupBuildArgs<'a> {
    /// The sealed genome (F0); never modified.
    pub base: &'a str,
    /// entries.jsonl (see [`parse_entries`]).
    pub entries: &'a str,
    /// Record id (`skill.{id}.lookup.*`).
    pub id: &'a str,
    /// Output file: a copy of the base plus the record; must differ from
    /// the base.
    pub out: &'a str,
    /// `ru,en`: slot languages in order. Default: every language the
    /// entries carry, sorted by name.
    pub langs: Option<&'a str>,
    /// Record name (default: the id).
    pub name: Option<&'a str>,
    /// `first | last | error` ([`OnDuplicate`]): which entry keeps a key
    /// two entries share.
    pub on_duplicate: &'a str,
    /// A stop list: one key per line (`#` comments), normalised like the
    /// entries' keys and left out of the table (generic words such as
    /// `plant` / `растения` that would catch every question).
    pub drop_keys: Option<&'a str>,
    /// A prompt set ([`read_prompt_set`]) the built table is probed with:
    /// a key hit by more than `suspicious_share` of the prompts is not a
    /// name and is reported (`suspicious_keys`).
    pub probe_prompts: Option<&'a str>,
    pub suspicious_share: f64,
    /// `router_and_key | key_first` ([`LookupPolicy`]): written to
    /// `lookup.policy`; `None` leaves the field out (= `router_and_key`).
    pub policy: Option<&'a str>,
    /// A large GENERAL prompt set the BUILT table is probed with through
    /// the runtime's strong-key extraction ([`general_probe`]): every
    /// strong key it hits is a general phrase `key_first` would answer
    /// from the table (review KF-1).
    pub general_prompts: Option<&'a str>,
    /// Where to write the stored keys behind those hits, one per line —
    /// a stop list for `--drop-keys`.
    pub general_stop_out: Option<&'a str>,
}

/// What to do with a key two ENTRIES share after normalisation (a key
/// repeated inside one entry is always just dropped): the corpus of a
/// merged reference lists one plant twice, and the first copy often holds
/// a stub (`нет данных в источнике`) where the second holds the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnDuplicate {
    /// The first entry in file order keeps the key.
    First,
    /// The last entry keeps it.
    Last,
    /// Refuse the build (the default): two entries naming one key is a
    /// corpus conflict the operator resolves — by merging the entries,
    /// or explicitly with `first` / `last`.
    Error,
}

impl OnDuplicate {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            "error" => Ok(Self::Error),
            other => anyhow::bail!("--on-duplicate {other}: expected first | last | error"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Last => "last",
            Self::Error => "error",
        }
    }
}

/// Knobs of [`build_table`].
pub struct BuildOptions<'a> {
    pub langs: Option<&'a [String]>,
    pub on_duplicate: OnDuplicate,
    /// Hashes of the normalised stop-list keys.
    pub drop: &'a HashSet<u64>,
}

/// One parsed entry: its keys and, per language (row order), the slot
/// object `{"card", "fields"}`.
pub struct Entry {
    pub keys: Vec<String>,
    pub slots: Vec<(String, serde_json::Value)>,
}

/// `entries.jsonl`: one object per line, `{"keys": ["Пихта бальзамическая",
/// "Abies balsamea", …], "ru": {"card": "…", "fields": {"family": "…", …}},
/// "en": {…}}`. Every object-valued key other than `keys` is a language;
/// `card` is a required string, `fields` an optional object of strings.
/// Other keys (strings, numbers) are ignored. Blank lines are skipped.
pub fn parse_entries(text: &str) -> anyhow::Result<Vec<Entry>> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("entries line {}", ln + 1);
        let v: serde_json::Value = serde_json::from_str(line).with_context(at)?;
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{}: not a JSON object", at()))?;
        let keys: Vec<String> = obj
            .get("keys")
            .and_then(|k| k.as_array())
            .ok_or_else(|| anyhow::anyhow!("{}: no \"keys\" array", at()))?
            .iter()
            .map(|k| {
                k.as_str()
                    .map(|s| s.trim().to_string())
                    .ok_or_else(|| anyhow::anyhow!("{}: a key is not a string", at()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let keys: Vec<String> = keys.into_iter().filter(|k| !k.is_empty()).collect();
        anyhow::ensure!(!keys.is_empty(), "{}: \"keys\" is empty", at());
        let mut slots = Vec::new();
        for (k, val) in obj {
            if k == "keys" {
                continue;
            }
            let Some(lang) = val.as_object() else {
                continue;
            };
            let card = lang
                .get("card")
                .and_then(|c| c.as_str())
                .ok_or_else(|| anyhow::anyhow!("{}: \"{k}\" has no string \"card\"", at()))?;
            let mut fields = serde_json::Map::new();
            if let Some(f) = lang.get("fields") {
                let f = f
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("{}: \"{k}.fields\" is not an object", at()))?;
                for (name, text) in f {
                    let text = text.as_str().ok_or_else(|| {
                        anyhow::anyhow!("{}: \"{k}.fields.{name}\" is not a string", at())
                    })?;
                    anyhow::ensure!(!name.is_empty(), "{}: an empty field name", at());
                    fields.insert(name.clone(), serde_json::Value::String(text.to_string()));
                }
            }
            slots.push((
                k.clone(),
                serde_json::json!({"card": card, "fields": serde_json::Value::Object(fields)}),
            ));
        }
        anyhow::ensure!(
            !slots.is_empty(),
            "{}: no language object (e.g. \"ru\": {{\"card\": …}})",
            at()
        );
        out.push(Entry { keys, slots });
    }
    anyhow::ensure!(!out.is_empty(), "entries: no rows");
    Ok(out)
}

/// The built table: what [`lookup_tensors`] takes plus the builder's report.
pub struct Table {
    pub info: LookupInfo,
    pub keys: Vec<(u64, u32)>,
    /// `(key as written, normalised)` parallel to `keys`.
    pub key_texts: Vec<(String, String)>,
    pub slots: Vec<String>,
    /// `{key, norm, entry, first_entry, first_key, same_entry,
    /// kept_entry}` per dropped duplicate (`kept_entry` is the winner:
    /// `first_entry`, or `entry` under `--on-duplicate last`).
    pub dropped_duplicates: Vec<serde_json::Value>,
    /// Duplicates inside one entry (harmless: a spelling repeated).
    pub duplicates_same_entry: usize,
    /// Duplicates across entries (a corpus conflict: two cards, one key).
    pub duplicates_cross_entry: usize,
    /// `{key, entry}` per key that normalises to nothing.
    pub dropped_empty: Vec<serde_json::Value>,
    /// `{key, norm, entry, words}` per key longer than [`MAX_NGRAM`]
    /// words: the runtime never tries such an n-gram, so the key is not
    /// stored (a composite `A; B (syn. C)` key must be split in the
    /// corpus).
    pub unreachable_keys: Vec<serde_json::Value>,
    /// `{key, norm, entry}` per key removed by the stop list.
    pub dropped_by_list: Vec<serde_json::Value>,
    /// Entries left without a single key (unreachable cards).
    pub unreachable_entries: Vec<usize>,
    /// `{entry, lang}` per slot of a listed language the entry does not
    /// carry (stored as an empty card; the runtime answers from another
    /// language of the entry, or misses when none has text).
    pub missing_slots: Vec<serde_json::Value>,
    /// Language objects of a language outside `--langs` (not stored).
    pub ignored_slots: usize,
}

/// Normalise, hash and de-duplicate the keys (a key repeated inside an
/// entry is dropped; a key two entries share goes by
/// [`OnDuplicate`]), drop the keys the runtime could never match (more
/// than [`MAX_NGRAM`] words) and the stop-listed ones, lay the slots out
/// as `entry · L + lang`, collect the field names.
pub fn build_table(entries: &[Entry], opts: BuildOptions<'_>) -> anyhow::Result<Table> {
    let langs: Vec<String> = match opts.langs {
        Some(l) => l.to_vec(),
        None => {
            // Every language the entries carry, sorted by name (the JSON
            // objects do not keep the file's key order).
            let mut v: Vec<String> = Vec::new();
            for e in entries {
                for (l, _) in &e.slots {
                    if !v.contains(l) {
                        v.push(l.clone());
                    }
                }
            }
            v.sort();
            v
        }
    };
    anyhow::ensure!(!langs.is_empty(), "no language (--langs or language objects)");
    for l in &langs {
        anyhow::ensure!(!l.is_empty(), "--langs: an empty language name");
        anyhow::ensure!(
            langs.iter().filter(|x| *x == l).count() == 1,
            "--langs: duplicate '{l}'"
        );
    }
    let mut fields: Vec<String> = Vec::new();
    // hash → index into `keys` / `key_texts` / `owner`
    let mut by_hash: HashMap<u64, usize> = HashMap::new();
    let mut keys: Vec<(u64, u32)> = Vec::new();
    let mut key_texts: Vec<(String, String)> = Vec::new();
    let mut dropped_duplicates = Vec::new();
    let (mut dup_same, mut dup_cross) = (0usize, 0usize);
    let mut dropped_empty = Vec::new();
    let mut unreachable_keys = Vec::new();
    let mut dropped_by_list = Vec::new();
    let mut missing_slots = Vec::new();
    let mut ignored_slots = 0usize;
    let mut slots = Vec::with_capacity(entries.len() * langs.len());
    for (ei, e) in entries.iter().enumerate() {
        for key in &e.keys {
            let norm = normalize_key(key);
            if norm.is_empty() {
                dropped_empty.push(serde_json::json!({"key": key, "entry": ei}));
                continue;
            }
            let words = norm.split(' ').count();
            if words > MAX_NGRAM {
                unreachable_keys.push(serde_json::json!({
                    "key": key, "norm": norm, "entry": ei, "words": words,
                }));
                continue;
            }
            let h = normalized_key_hash(&norm);
            if opts.drop.contains(&h) {
                dropped_by_list.push(serde_json::json!({"key": key, "norm": norm, "entry": ei}));
                continue;
            }
            if let Some(&i) = by_hash.get(&h) {
                let (first_entry, first_key) = (keys[i].1 as usize, key_texts[i].0.clone());
                let same = first_entry == ei;
                let kept = if !same && opts.on_duplicate == OnDuplicate::Last {
                    keys[i].1 = ei as u32;
                    key_texts[i] = (key.clone(), norm.clone());
                    ei
                } else {
                    first_entry
                };
                if same {
                    dup_same += 1;
                } else {
                    dup_cross += 1;
                }
                dropped_duplicates.push(serde_json::json!({
                    "key": key, "norm": norm, "entry": ei,
                    "first_entry": first_entry, "first_key": first_key,
                    "same_entry": same, "kept_entry": kept,
                }));
                continue;
            }
            by_hash.insert(h, keys.len());
            keys.push((h, ei as u32));
            key_texts.push((key.clone(), norm));
        }
        for l in &langs {
            match e.slots.iter().find(|(sl, _)| sl == l) {
                Some((_, v)) => {
                    if let Some(f) = v.get("fields").and_then(|f| f.as_object()) {
                        for name in f.keys() {
                            if !fields.contains(name) {
                                fields.push(name.clone());
                            }
                        }
                    }
                    slots.push(v.to_string());
                }
                None => {
                    missing_slots.push(serde_json::json!({"entry": ei, "lang": l}));
                    slots.push(serde_json::json!({"card": "", "fields": {}}).to_string());
                }
            }
        }
        ignored_slots += e.slots.iter().filter(|(sl, _)| !langs.contains(sl)).count();
    }
    if dup_cross > 0 && opts.on_duplicate == OnDuplicate::Error {
        let head: Vec<String> = dropped_duplicates
            .iter()
            .filter(|d| d["same_entry"] == false)
            .take(10)
            .map(|d| {
                format!(
                    "{:?} (entries {} and {})",
                    d["norm"].as_str().unwrap_or(""),
                    d["first_entry"],
                    d["entry"]
                )
            })
            .collect();
        anyhow::bail!(
            "{dup_cross} cross-entry duplicate key(s): two entries name one key — merge them in \
             the corpus, or choose --on-duplicate first|last (the file's order decides which \
             card answers): {}{}",
            head.join(", "),
            if dup_cross > head.len() { ", …" } else { "" }
        );
    }
    anyhow::ensure!(
        !keys.is_empty(),
        "no key survives normalisation ({} empty, {} duplicates, {} unreachable, {} stop-listed)",
        dropped_empty.len(),
        dropped_duplicates.len(),
        unreachable_keys.len(),
        dropped_by_list.len()
    );
    let reachable: HashSet<u32> = keys.iter().map(|k| k.1).collect();
    let unreachable_entries: Vec<usize> = (0..entries.len())
        .filter(|e| !reachable.contains(&(*e as u32)))
        .collect();
    let info = LookupInfo {
        entries: entries.len(),
        keys: keys.len(),
        key_norm: KEY_NORM.into(),
        langs,
        fields,
        policy: None,
    };
    Ok(Table {
        info,
        keys,
        key_texts,
        slots,
        dropped_duplicates,
        duplicates_same_entry: dup_same,
        duplicates_cross_entry: dup_cross,
        dropped_empty,
        unreachable_keys,
        dropped_by_list,
        unreachable_entries,
        missing_slots,
        ignored_slots,
    })
}

/// The stop list (`--drop-keys`): one key per line, `#` comments and
/// blank lines skipped; returns the hashes of the normalised keys and how
/// many lines were read.
pub fn read_drop_keys(path: &str) -> anyhow::Result<(HashSet<u64>, usize)> {
    let text = std::fs::read_to_string(path).with_context(|| path.to_string())?;
    let mut set = HashSet::new();
    let mut n = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let norm = normalize_key(line);
        if !norm.is_empty() {
            set.insert(normalized_key_hash(&norm));
            n += 1;
        }
    }
    Ok((set, n))
}

/// Probe the built table with a prompt set through the runtime's own key
/// extraction: how many prompts hit, and which keys hit more than
/// `share` of them — a key that fires on a large share of in-scope
/// questions (`plant`, `растения`, `species`) is a generic word, not a
/// name, and would answer every question that names no plant with one
/// card. The stem index here is built from the key texts themselves
/// (the runtime recovers them from the cards, so a key no card spells
/// stem-matches here but not at run time). Report: `{file, prompts,
/// hits, stem_hits, hit_rate, suspicious_share, suspicious_keys: [{key,
/// norm, entry, hits, share}]}` (`suspicious_keys` count exact hits).
pub fn probe_keys(table: &Table, path: &str, share: f64) -> anyhow::Result<serde_json::Value> {
    let (prompts, _) = read_prompt_set(path)?;
    let by_hash: HashMap<u64, usize> = table
        .keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.0, i))
        .collect();
    let stems = StemIndex::from_keys(
        table
            .key_texts
            .iter()
            .zip(&table.keys)
            .map(|((_, norm), (_, e))| (norm.as_str(), *e)),
    );
    let mut counts = vec![0usize; table.keys.len()];
    let (mut hits, mut stem_hits) = (0usize, 0usize);
    for p in &prompts {
        if let Some(h) = extract_key_with(
            p,
            &|h| by_hash.get(&h).map(|&i| table.keys[i].1),
            Some(&|h| stems.find(h)),
        ) {
            hits += 1;
            stem_hits += (h.via == MatchVia::Stem) as usize;
            if let Some(&i) = by_hash.get(&h.hash) {
                counts[i] += 1;
            }
        }
    }
    let n = prompts.len().max(1) as f64;
    let mut suspicious: Vec<(usize, usize)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c as f64 / n > share)
        .map(|(i, &c)| (i, c))
        .collect();
    suspicious.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(serde_json::json!({
        "file": path,
        "prompts": prompts.len(),
        "hits": hits,
        "stem_hits": stem_hits,
        "hit_rate": hits as f64 / n,
        "suspicious_share": share,
        "suspicious_keys": suspicious.iter().map(|&(i, c)| serde_json::json!({
            "key": table.key_texts[i].0, "norm": table.key_texts[i].1,
            "entry": table.keys[i].1, "hits": c, "share": c as f64 / n,
        })).collect::<Vec<_>>(),
    }))
}

/// The general probe (review KF-1): every prompt of a large GENERAL set
/// through the runtime's STRONG-key extraction on the built table
/// (`LookupTable::find_strong_key` — what `key_first` acts on). Each hit
/// is a general phrase the table would answer under `key_first` (`running
/// pop`, `scrambled eggs`, `live forever` — real common names that are
/// also English phrases). Returns the report `{prompts, strong_hits,
/// strong_hit_rate, distinct_keys, keys: [{key, via, source, entry, words,
/// hits, example, stored_keys}]}` (keys by hits, descending) and the
/// stored key texts behind the hits (every normalised key of the entry
/// whose stem is the one hit, for a stem hit), for a stop list.
pub fn general_probe(
    rt: &LookupTable,
    built: &Table,
    prompts: &[String],
) -> (serde_json::Value, Vec<String>) {
    struct Agg {
        via: &'static str,
        source: &'static str,
        entry: u32,
        words: usize,
        hits: usize,
        example: String,
        stored: Vec<String>,
    }
    let mut agg: HashMap<(u32, String), Agg> = HashMap::new();
    let mut strong_hits = 0usize;
    for p in prompts {
        let Some(h) = rt.find_strong_key(p) else {
            continue;
        };
        strong_hits += 1;
        let label = match h.via {
            MatchVia::Exact => h.key.clone(),
            MatchVia::Stem => h.stem.clone().unwrap_or_else(|| h.key.clone()),
        };
        let e = agg.entry((h.entry, label.clone())).or_insert_with(|| {
            let stored: Vec<String> = match h.via {
                MatchVia::Exact => vec![h.key.clone()],
                MatchVia::Stem => built
                    .key_texts
                    .iter()
                    .zip(&built.keys)
                    .filter(|((_, norm), (_, e))| *e == h.entry && stem_key(norm) == label)
                    .map(|((_, norm), _)| norm.clone())
                    .collect(),
            };
            Agg {
                via: h.via.label(),
                source: h.source.label(),
                entry: h.entry,
                words: h.words,
                hits: 0,
                example: p.chars().take(160).collect(),
                stored,
            }
        });
        e.hits += 1;
    }
    let mut rows: Vec<(String, Agg)> = agg.into_iter().map(|((_, k), a)| (k, a)).collect();
    rows.sort_by(|a, b| b.1.hits.cmp(&a.1.hits).then(a.0.cmp(&b.0)).then(a.1.entry.cmp(&b.1.entry)));
    let mut stop: Vec<String> = Vec::new();
    for (_, a) in &rows {
        for k in &a.stored {
            if !stop.contains(k) {
                stop.push(k.clone());
            }
        }
    }
    let n = prompts.len().max(1) as f64;
    let report = serde_json::json!({
        "prompts": prompts.len(),
        "strong_hits": strong_hits,
        "strong_hit_rate": strong_hits as f64 / n,
        "distinct_keys": rows.len(),
        "keys": rows.iter().map(|(k, a)| serde_json::json!({
            "key": k, "via": a.via, "source": a.source, "entry": a.entry, "words": a.words,
            "hits": a.hits, "example": a.example, "stored_keys": a.stored,
        })).collect::<Vec<_>>(),
    });
    (report, stop)
}

/// Are `a` and `b` one file? By path, and — when both exist — by
/// `(device, inode)`: a hard link or a symlink to the base passes a path
/// comparison and would be truncated by the copy.
fn same_file(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(x), Ok(y)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return x.dev() == y.dev() && x.ino() == y.ino();
        }
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// A list for the record's `origin` (the header stays small): the first
/// `cap` items and whether more were cut.
fn capped(list: &[serde_json::Value], cap: usize) -> (Vec<&serde_json::Value>, bool) {
    (list.iter().take(cap).collect(), list.len() > cap)
}

pub fn cmd_lookup_build(a: LookupBuildArgs<'_>) -> anyhow::Result<()> {
    anyhow::ensure!(
        !same_file(a.base, a.out),
        "lookup-build: --out {} is the base itself (same path, or a hard link / symlink to it) \
         — the record goes to a COPY of the sealed genome (the base is never modified)",
        a.out
    );
    let on_duplicate = OnDuplicate::parse(a.on_duplicate)?;
    let policy = a
        .policy
        .map(|p| LookupPolicy::parse(p.trim()).map_err(|e| anyhow::anyhow!("--policy: {e}")))
        .transpose()?;
    anyhow::ensure!(
        a.suspicious_share.is_finite() && (0.0..=1.0).contains(&a.suspicious_share),
        "--suspicious-share {} must be in [0, 1]",
        a.suspicious_share
    );
    let bytes = std::fs::read(a.entries).with_context(|| a.entries.to_string())?;
    let sha = sha256_hex(&bytes);
    let text = std::str::from_utf8(&bytes).context("entries are not UTF-8")?;
    let entries = parse_entries(text).with_context(|| a.entries.to_string())?;
    let langs: Option<Vec<String>> = a.langs.map(|l| {
        l.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let (drop_set, drop_n) = match a.drop_keys {
        Some(p) => read_drop_keys(p)?,
        None => (HashSet::new(), 0),
    };
    let mut table = build_table(
        &entries,
        BuildOptions {
            langs: langs.as_deref(),
            on_duplicate,
            drop: &drop_set,
        },
    )
    .map_err(|e| anyhow::anyhow!("lookup-build: {e}"))?;
    table.info.policy = policy.map(|p| p.label().to_string());
    let slot_refs: Vec<&str> = table.slots.iter().map(String::as_str).collect();
    let tensors = lookup_tensors(a.id, &table.info, &table.keys, &slot_refs)
        .map_err(|e| anyhow::anyhow!("lookup-build: {e}"))?;
    let text_bytes = tensors
        .iter()
        .find(|t| t.name.ends_with(".lookup.text"))
        .map_or(0, |t| t.data.len());
    anyhow::ensure!(
        a.general_stop_out.is_none() || a.general_prompts.is_some(),
        "--general-stop-out needs --general-prompts (the stop list is what the general probe hit)"
    );
    // Read before anything is written: a bad set fails the build early.
    let general = match a.general_prompts {
        Some(p) => Some(read_prompt_set(p)?),
        None => None,
    };
    let probe = match a.probe_prompts {
        Some(p) => Some(probe_keys(&table, p, a.suspicious_share)?),
        None => None,
    };

    let base = CmfModel::open(a.base).with_context(|| a.base.to_string())?;
    let genome = base.header.genome.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "lookup-build: {} is not a sealed genome (no header.genome) — a v2 record binds to \
             a genome",
            a.base
        )
    })?;
    anyhow::ensure!(
        !base.header.skills.iter().any(|s| s.id == a.id),
        "lookup-build: skill '{}' already exists in {}",
        a.id,
        a.base
    );
    let trunk = base.trunk_hash();
    drop(base);

    let cap = 100;
    let (dup_list, dup_cut) = capped(&table.dropped_duplicates, cap);
    let (unr_list, unr_cut) = capped(&table.unreachable_keys, cap);
    let (miss_list, miss_cut) = capped(&table.missing_slots, cap);
    let (stop_list, stop_cut) = capped(&table.dropped_by_list, cap);
    let suspicious = probe
        .as_ref()
        .and_then(|p| p["suspicious_keys"].as_array())
        .map(|v| v.len())
        .unwrap_or(0);
    let record = SkillRecord {
        id: a.id.to_string(),
        name: Some(a.name.unwrap_or(a.id).to_string()),
        kind: Some(skill_kind::LOOKUP.into()),
        lookup: Some(table.info.clone()),
        bound: Some(SkillBound {
            genome_id: genome.id.clone(),
            generation: genome.generation,
            master_trunk_hash: genome.master_trunk_hash.clone(),
        }),
        state_effect: Some(lookup_state_effect()),
        status: Some("quarantine".into()),
        origin: Some(serde_json::json!({
            "trigger": "user_corpus",
            "dataset_sha256": [sha],
            "entries_file": Path::new(a.entries).file_name().and_then(|f| f.to_str()),
            "entries": table.info.entries,
            "keys": table.info.keys,
            "langs": table.info.langs,
            "fields": table.info.fields,
            "on_duplicate": on_duplicate.label(),
            "duplicates_dropped": table.dropped_duplicates.len(),
            "duplicates_same_entry": table.duplicates_same_entry,
            "duplicates_cross_entry": table.duplicates_cross_entry,
            "empty_keys_dropped": table.dropped_empty.len(),
            "duplicate_keys": dup_list,
            "duplicate_keys_truncated": dup_cut,
            "unreachable_keys_dropped": table.unreachable_keys.len(),
            "unreachable_keys": unr_list,
            "unreachable_keys_truncated": unr_cut,
            "drop_list_keys": drop_n,
            "dropped_by_list": stop_list,
            "dropped_by_list_truncated": stop_cut,
            "unreachable_entries": table.unreachable_entries,
            "missing_slots": table.missing_slots.len(),
            "missing_slot_list": miss_list,
            "missing_slot_list_truncated": miss_cut,
            "probe": probe,
            "builder": "cortiq lookup-build",
        })),
        ..Default::default()
    };
    let event = LineageEvent::now(
        0,
        "skill_committed",
        serde_json::json!({
            "id": a.id,
            "kind": skill_kind::LOOKUP,
            "entries": table.info.entries,
            "keys": table.info.keys,
            "dataset_sha256": sha,
            "builder": "cortiq lookup-build",
        }),
    );
    // The copy is made under a private temporary name next to --out and
    // renamed over it only once the append and the checks passed: --out
    // is never a truncated or half-written file, and nothing that is not
    // the finished copy is ever removed.
    let out_path = Path::new(a.out);
    let tmp = out_path.with_file_name(format!(
        ".{}.lookup-build-{}.tmp",
        out_path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow::anyhow!("--out {} names no file", a.out))?,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);
    let staged = (|| -> anyhow::Result<cortiq_core::AppendReport> {
        std::fs::copy(a.base, &tmp)
            .with_context(|| format!("copy {} → {}", a.base, tmp.display()))?;
        // The copy inherits the base's mode: a sealed, read-only genome
        // gives a read-only copy the append could not open for writing.
        let mut perm = std::fs::metadata(&tmp)?.permissions();
        if perm.readonly() {
            #[allow(clippy::permissions_set_readonly_false)]
            perm.set_readonly(false);
            std::fs::set_permissions(&tmp, perm)?;
        }
        let report = CmfModel::append_skill(&tmp, record, &tensors, None, None, Some(event))
            .map_err(|e| anyhow::anyhow!("{}: {e}", a.out))?;
        let after = CmfModel::open(&tmp)?;
        anyhow::ensure!(
            after.trunk_hash() == trunk,
            "trunk hash changed ({} → {}) — this must not happen",
            hex64(trunk),
            hex64(after.trunk_hash())
        );
        drop(after);
        std::fs::rename(&tmp, a.out)
            .with_context(|| format!("rename {} → {}", tmp.display(), a.out))?;
        Ok(report)
    })();
    let report = match staged {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            anyhow::bail!("lookup-build: {e}");
        }
    };
    let after = Arc::new(CmfModel::open(a.out).with_context(|| a.out.to_string())?);
    // The general probe runs the RUNTIME's table (the stem index and the
    // attested binomials as the runtime builds them at open).
    let general_report = match (&general, a.general_prompts) {
        (Some((prompts, sha)), Some(file)) => {
            let rt = LookupTable::open(&after, a.id).map_err(anyhow::Error::msg)?;
            let (mut rep, stop) = general_probe(&rt, &table, prompts);
            rep["file"] = serde_json::json!(file);
            rep["sha256"] = serde_json::json!(sha);
            if let Some(out) = a.general_stop_out {
                let mut text = format!(
                    "# lookup-build --general-prompts {file}: the stored keys behind every strong \
                     key hit on {} general prompts ({} prompts hit) — general phrases key_first \
                     would answer from the table. Review, then pass as --drop-keys.\n",
                    prompts.len(),
                    rep["strong_hits"]
                );
                for k in &stop {
                    text.push_str(k);
                    text.push('\n');
                }
                std::fs::write(out, text).with_context(|| out.to_string())?;
                rep["stop_out"] = serde_json::json!({"file": out, "keys": stop.len()});
            }
            Some(rep)
        }
        _ => None,
    };
    for (what, list) in [
        ("duplicate keys dropped", &table.dropped_duplicates),
        ("empty keys dropped", &table.dropped_empty),
        (
            "keys longer than the runtime's n-gram window dropped (unreachable)",
            &table.unreachable_keys,
        ),
        ("keys removed by --drop-keys", &table.dropped_by_list),
    ] {
        if !list.is_empty() {
            eprintln!("lookup-build: {} {what}:", list.len());
            for d in list.iter().take(20) {
                eprintln!("  {d}");
            }
            if list.len() > 20 {
                eprintln!("  … and {} more (all in the JSON report)", list.len() - 20);
            }
        }
    }
    if table.duplicates_cross_entry > 0 {
        eprintln!(
            "warning: {} keys are shared by two entries — --on-duplicate {} decided which card \
             answers them (the corpus should carry one entry per name)",
            table.duplicates_cross_entry,
            on_duplicate.label()
        );
    }
    if !table.unreachable_entries.is_empty() {
        eprintln!(
            "warning: {} entries have no key left (unreachable cards): {:?}",
            table.unreachable_entries.len(),
            table.unreachable_entries
        );
    }
    if !table.missing_slots.is_empty() {
        eprintln!(
            "note: {} slots of a listed language are empty (the entry has no card in it; the \
             runtime answers from another language of the entry): first {:?}",
            table.missing_slots.len(),
            table.missing_slots.iter().take(5).collect::<Vec<_>>()
        );
    }
    if let Some(g) = &general_report {
        let hits = g["strong_hits"].as_u64().unwrap_or(0);
        eprintln!(
            "lookup-build: general probe {}: {hits} / {} prompts hold a STRONG key ({} distinct \
             keys)",
            g["file"], g["prompts"], g["distinct_keys"]
        );
        if hits > 0 {
            eprintln!(
                "warning: under key_first each of these prompts would be answered from the table \
                 instead of the backbone — general phrases that are also common names; put them \
                 in a --drop-keys file (--general-stop-out writes one) and rebuild:"
            );
            for k in g["keys"].as_array().into_iter().flatten().take(20) {
                eprintln!(
                    "  {} ×{} → entry {} ({}; e.g. {})",
                    k["key"], k["hits"], k["entry"], k["via"], k["example"]
                );
            }
        }
    } else if policy == Some(LookupPolicy::KeyFirst) {
        eprintln!(
            "warning: lookup-build --policy key_first without --general-prompts: no measurement of \
             which general phrases the strong keys catch (review KF-1) — probe a large general set \
             before activating the record"
        );
    }
    if let Some(p) = &probe {
        eprintln!(
            "lookup-build: probe {}: {} / {} prompts hit a key",
            p["file"], p["hits"], p["prompts"]
        );
        if suspicious > 0 {
            eprintln!(
                "warning: {suspicious} keys hit more than {:.1} % of the probe prompts — generic \
                 words, not names; every question without a plant name would get their card. \
                 Put them in a --drop-keys file:",
                a.suspicious_share * 100.0
            );
            for k in p["suspicious_keys"].as_array().into_iter().flatten().take(20) {
                eprintln!("  {k}");
            }
        }
    }
    let j = serde_json::json!({
        "file": a.out,
        "base": a.base,
        "skill": a.id,
        "kind": skill_kind::LOOKUP,
        "status": "quarantine",
        "key_norm": KEY_NORM,
        "entries": table.info.entries,
        "keys": table.info.keys,
        "langs": table.info.langs,
        "fields": table.info.fields,
        "text_bytes": text_bytes,
        "on_duplicate": on_duplicate.label(),
        "policy": table.info.policy_label(),
        "duplicates_dropped": table.dropped_duplicates.len(),
        "duplicates_same_entry": table.duplicates_same_entry,
        "duplicates_cross_entry": table.duplicates_cross_entry,
        "duplicate_keys": table.dropped_duplicates,
        "empty_keys_dropped": table.dropped_empty.len(),
        "empty_keys": table.dropped_empty,
        "unreachable_keys_dropped": table.unreachable_keys.len(),
        "unreachable_keys": table.unreachable_keys,
        "drop_list": a.drop_keys.map(|f| serde_json::json!({"file": f, "keys": drop_n})),
        "dropped_by_list": table.dropped_by_list,
        "unreachable_entries": table.unreachable_entries,
        "missing_slots": table.missing_slots.len(),
        "missing_slot_list": table.missing_slots,
        "ignored_slots": table.ignored_slots,
        "probe": probe,
        "suspicious_keys": suspicious,
        "general_probe": general_report,
        "dataset_sha256": sha,
        "trunk_hash": hex64(trunk),
        "old_len": report.old_len,
        "new_len": report.new_len,
        "tensors_added": report.tensors_added,
        "lineage_seq": after.header.lineage.last().map(|e| e.seq),
        "auto_routable": false,
    });
    println!("{}", serde_json::to_string_pretty(&j)?);
    Ok(())
}

// ───────────────────────── lookup-policy ─────────────────────────

/// `cortiq lookup-policy <file> --id X --policy router_and_key|key_first
/// [--keep-gate]`: switch the routing policy of an existing lookup record
/// by ONE header-only tail append (`update_header_append`) with a lineage
/// event `lookup_policy` — no rebuild, the table, the trunk and the
/// router's calibration stay (the policy is not part of `skills_hash`). A
/// record whose EFFECTIVE policy already is the requested one is left
/// alone (no append; an absent field is `router_and_key`).
///
/// The gate (review KF-2): an ACTIVE record measured without the
/// key_first step (its `gate` is not a route-eval summary with
/// `key_first_step` covering this record, [`crate::knowledge::gate_covers_key_first`])
/// goes to `stale_regate` when it is switched TO key_first — key_first
/// adds accepts the old G3 never counted — and stays out of auto-routing
/// until `route-eval` (which applies the key_first step) and
/// `skill-gate --status active` re-gate it. `--keep-gate` keeps it active
/// (the operator vouches). A switch back to router_and_key keeps the
/// status: key_first's accepts are a superset of the router's. JSON report
/// on stdout.
pub fn cmd_lookup_policy(path: &str, id: &str, policy: &str, keep_gate: bool) -> anyhow::Result<()> {
    let to = LookupPolicy::parse(policy.trim()).map_err(|e| anyhow::anyhow!("--policy: {e}"))?;
    let model = CmfModel::open(path).with_context(|| path.to_string())?;
    let rec = model
        .header
        .skills
        .iter()
        .find(|s| s.id == id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "lookup-policy: skill '{id}' not in {path} (header.skills: {:?})",
                model.header.skills.iter().map(|s| &s.id).collect::<Vec<_>>()
            )
        })?;
    anyhow::ensure!(
        rec.kind.as_deref() == Some(skill_kind::LOOKUP),
        "lookup-policy: skill '{id}' is a {} record — the policy belongs to a lookup record",
        rec.kind.as_deref().unwrap_or("v1")
    );
    let info = rec
        .lookup
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("lookup-policy: skill '{id}' has no `lookup`"))?;
    let from_label = info.policy_label().to_string();
    let from_known = LookupPolicy::is_known(info);
    let from_effective = LookupPolicy::of(info);
    let trunk = model.trunk_hash();
    let hash_before = router::skills_hash(&model.header);
    let routable_before = rec.is_auto_routable();
    let status_before = rec.status.clone();
    let gate_covers = rec
        .gate
        .as_ref()
        .is_some_and(|g| crate::knowledge::gate_covers_key_first(g, id));
    let seq = model.header.lineage.last().map_or(0, |e| e.seq + 1);
    drop(model);
    // The EFFECTIVE policy decides (review KF-6): an absent field already
    // is router_and_key — nothing to append. An unknown value (a newer
    // writer's) is replaced by the requested one.
    let unchanged = from_known && from_effective == to;
    let regate = !unchanged
        && to == LookupPolicy::KeyFirst
        && status_before.as_deref() == Some("active")
        && !gate_covers
        && !keep_gate;
    let status_after = if regate {
        Some("stale_regate".to_string())
    } else {
        status_before.clone()
    };
    let (report, lineage_seq) = if unchanged {
        (None, None)
    } else {
        let event = LineageEvent::now(
            seq,
            "lookup_policy",
            serde_json::json!({
                "id": id,
                "policy_from": from_label,
                "policy_to": to.label(),
                "status_from": status_before,
                "status_to": status_after,
                "keep_gate": keep_gate,
                "gate_covers_key_first": gate_covers,
            }),
        );
        let (id_s, to_s) = (id.to_string(), to.label().to_string());
        let r = CmfModel::update_header_append(path, move |h| {
            if let Some(s) = h.skills.iter_mut().find(|s| s.id == id_s) {
                if let Some(li) = s.lookup.as_mut() {
                    li.policy = Some(to_s);
                }
                if regate {
                    s.status = Some("stale_regate".into());
                }
            }
            h.lineage.push(event);
        })
        .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        (Some(r), Some(seq))
    };
    let after = CmfModel::open(path).with_context(|| path.to_string())?;
    let rec = after
        .header
        .skills
        .iter()
        .find(|s| s.id == id)
        .expect("the record survives a header update");
    let now = rec.lookup.as_ref().map(LookupPolicy::of);
    anyhow::ensure!(
        now == Some(to),
        "lookup-policy: {path} reads policy {now:?} after the update — this must not happen"
    );
    anyhow::ensure!(
        after.trunk_hash() == trunk,
        "trunk hash changed ({} → {}) — this must not happen",
        hex64(trunk),
        hex64(after.trunk_hash())
    );
    let stale = router::skills_hash(&after.header) != hash_before;
    let j = serde_json::json!({
        "file": path,
        "skill": id,
        "policy_from": from_label,
        "policy_to": to.label(),
        "changed": !unchanged,
        "lineage_seq": lineage_seq,
        "old_len": report.as_ref().map(|r| r.old_len),
        "new_len": report.as_ref().map(|r| r.new_len),
        "trunk_hash": hex64(trunk),
        "status": rec.status,
        "status_from": status_before,
        "regate_required": regate,
        "keep_gate": keep_gate,
        "gate_covers_key_first": gate_covers,
        "auto_routable": rec.is_auto_routable(),
        "auto_routable_before": routable_before,
        "calibration_stale": stale,
    });
    println!("{}", serde_json::to_string_pretty(&j)?);
    if unchanged {
        eprintln!("lookup-policy: '{id}' already reads as {} — nothing appended", to.label());
    }
    if regate {
        eprintln!(
            "lookup-policy: '{id}' was active with a gate measured WITHOUT the key_first step — \
             it is now stale_regate (not auto-routable). Re-gate: cortiq route-eval {path} \
             --prompts-jsonl <general-eval.jsonl> --expect backbone --include-quarantine --json \
             > gate.json && cortiq skill-gate {path} --id {id} --gate gate.json --status active"
        );
    } else if to == LookupPolicy::KeyFirst && keep_gate && !gate_covers && !unchanged {
        eprintln!(
            "warning: lookup-policy: --keep-gate — '{id}' stays {:?} with a gate measured without \
             the key_first step (its false-accept bound does not cover key_first)",
            rec.status
        );
    }
    Ok(())
}


// ───────────────────────── route-fit ─────────────────────────

pub struct RouteFitArgs<'a> {
    /// The file with the skill record (updated in place, header-only append).
    pub model: &'a str,
    pub id: &'a str,
    /// In-scope prompts of the skill (JSONL `{"prompt"}` / `{"text"}` /
    /// strings, or one prompt per line).
    pub skill_prompts: &'a str,
    /// General prompts (the backbone class), same formats.
    pub general_prompts: &'a str,
    /// φ layer (default: the file's current `router.phi.layer`).
    pub phi_layer: Option<usize>,
    /// PCA rank of both descriptors (clamped to train − 1).
    pub rank: usize,
    /// At most this many prompts per set, in file order (0 = all).
    pub max: usize,
    /// `router.margin` (default: the current router's, else 0.05).
    pub margin: Option<f32>,
    /// Target in-scope false-positive rate of the novelty flag.
    pub target_fpr: f32,
}

/// A prompt set: JSONL (`{"prompt"}` / `{"text"}` / a bare string per
/// line) or plain text (one prompt per line); blank prompts skipped.
/// Returns the prompts and the sha256 of the FILE (what `router.measured`
/// records and `route-eval` refuses as a gate set).
pub fn read_prompt_set(path: &str) -> anyhow::Result<(Vec<String>, String)> {
    let bytes = std::fs::read(path).with_context(|| path.to_string())?;
    let sha = sha256_hex(&bytes);
    let text = std::str::from_utf8(&bytes).with_context(|| format!("{path}: not UTF-8"))?;
    let jsonl = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .is_some_and(|l| {
            matches!(
                l.trim_start().as_bytes().first().copied(),
                Some(b'{') | Some(b'"')
            )
        });
    let mut prompts = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let p = if jsonl {
            let v: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("{path}:{}: invalid JSONL", ln + 1))?;
            v.as_str()
                .or_else(|| v.get("prompt").and_then(|x| x.as_str()))
                .or_else(|| v.get("text").and_then(|x| x.as_str()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{path}:{}: a JSONL prompt is a string or has \"prompt\" / \"text\"",
                        ln + 1
                    )
                })?
                .to_string()
        } else {
            line.trim().to_string()
        };
        if !p.trim().is_empty() {
            prompts.push(p);
        }
    }
    anyhow::ensure!(!prompts.is_empty(), "{path}: no prompts");
    Ok((prompts, sha))
}

/// Canonical router-v2 φ of every prompt on the backbone (raw span mean;
/// the fit normalises).
fn phi_set(p: &mut Pipeline, spec: &PhiSpec, prompts: &[String], what: &str) -> Vec<Vec<f32>> {
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(prompts.len());
    for (i, text) in prompts.iter().enumerate() {
        let q = p.tokenizer.encode_plain(text);
        let (ids, span) = router::phi_span_ids(spec, &q);
        out.push(p.probe_phi_span(&ids, spec.layer, span));
        if (i + 1) % 250 == 0 || i + 1 == prompts.len() {
            eprintln!(
                "route-fit: φ {what} {}/{} [{:.1} s]",
                i + 1,
                prompts.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    out
}

fn desc_json(d: &SelectionDescriptor, n: usize) -> serde_json::Value {
    serde_json::json!({
        "n": n,
        "train": n - d.holdout_n.unwrap_or(0),
        "holdout": d.holdout_n,
        "rank": d.rank,
        "err_mean": d.err_mean,
        "err_std": d.err_std,
    })
}

/// Tolerance of the fit ↔ runtime unit-φ parity (max |Δ| per component)
/// — the trainer's `PHI_PARITY_TOL`.
pub const PHI_PARITY_TOL: f32 = 1e-5;
/// Tolerance of the unit errors `E_base` / `E_skill` between the runtime's
/// decision and the one recomputed from the fit's φ.
pub const E_PARITY_TOL: f32 = 1e-4;
/// Holdout prompts per class the post-fit runtime check replays.
pub const RUNTIME_CHECK_PER_CLASS: usize = 8;

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Before the header is written, a FRESH backbone pipeline replays held-out
/// prompts of both classes the way `route_request` will: `encode_plain` →
/// `phi_span_ids` → `probe_phi_span` → the decision under the new policy
/// and calibration (quarantined classes scored). The unit φ must agree
/// with the fit's within `tol` per component, the decision must be the
/// same and `E_base` / `E_skill` within [`E_PARITY_TOL`] — otherwise the
/// descriptors and the calibration would describe another φ than the
/// served one, and nothing is written. The check runs on the backend of
/// THIS process (`phi_backend` in `router.measured` records it): a fit on
/// the CPU and a serve on a coop-GEMM GPU differ by ~1e-4 in φ, which
/// `route-eval` / `serve` warn about when they see another backend.
fn runtime_parity_check(
    model: &Arc<CmfModel>,
    header: &CmfHeader,
    cases: &[(&str, &[f32])],
    tol: f32,
) -> anyhow::Result<serde_json::Value> {
    let policy = header
        .router
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("runtime check: no router in the fitted header"))?;
    let mut pipe = Pipeline::from_model(model, SamplerConfig::default())?;
    let opts = RouteOptions {
        include_quarantine: true,
    };
    let (mut worst_phi, mut worst_e, mut to_skill) = (0f32, 0f32, 0usize);
    for (k, (text, phi_fit)) in cases.iter().enumerate() {
        let q = pipe.tokenizer.encode_plain(text);
        let (ids, span) = router::phi_span_ids(&policy.phi, &q);
        let phi_rt = pipe.probe_phi_span(&ids, policy.phi.layer, span);
        let (u_rt, u_fit) = (unit(&phi_rt), unit(phi_fit));
        anyhow::ensure!(
            u_rt.len() == u_fit.len(),
            "runtime check: holdout prompt #{k} — φ has {} components in the runtime, {} in the \
             fit",
            u_rt.len(),
            u_fit.len()
        );
        let d = u_rt
            .iter()
            .zip(&u_fit)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        anyhow::ensure!(
            d.is_finite() && d <= tol,
            "runtime check: holdout prompt #{k} — the runtime's unit φ differs from the fit's by \
             {d:.3e} (> {tol:.0e}): the descriptors and the calibration would describe another φ \
             than the served one ({text:?})"
        );
        worst_phi = worst_phi.max(d);
        let rt = router::route_policy_with(header, &phi_rt, opts);
        let fit = router::route_policy_with(header, phi_fit, opts);
        anyhow::ensure!(
            rt.target == fit.target,
            "runtime check: holdout prompt #{k} — the runtime routes to {} ({}), the fit's φ to \
             {} ({}) ({text:?})",
            rt.target_label(),
            rt.reason,
            fit.target_label(),
            fit.reason
        );
        for (what, a, b) in [
            ("E_base", rt.e_base(), fit.e_base()),
            ("E_skill", rt.e_skill(), fit.e_skill()),
        ] {
            if let (Some(a), Some(b)) = (a, b) {
                let de = (a - b).abs();
                anyhow::ensure!(
                    de.is_finite() && de <= E_PARITY_TOL,
                    "runtime check: holdout prompt #{k} — {what} {a:.6} in the runtime vs \
                     {b:.6} recomputed from the fit's φ (Δ {de:.3e} > {E_PARITY_TOL:.0e}) \
                     ({text:?})"
                );
                worst_e = worst_e.max(de);
            }
        }
        to_skill += usize::from(rt.skill().is_some());
    }
    Ok(serde_json::json!({
        "prompts": cases.len(),
        "decisions_equal": cases.len(),
        "to_skill": to_skill,
        "to_backbone": cases.len() - to_skill,
        "max_unit_phi_delta": worst_phi,
        "max_e_delta": worst_e,
        "tol": tol,
        "e_tol": E_PARITY_TOL,
        "phi_backend": phi_backend_label(),
        "reference": "fresh backbone pipeline of this process, route_policy_with, quarantine scored",
    }))
}

pub fn cmd_route_fit(a: RouteFitArgs<'_>) -> anyhow::Result<()> {
    anyhow::ensure!(a.rank >= 1, "--rank must be ≥ 1");
    anyhow::ensure!(
        a.target_fpr.is_finite() && (0.0..1.0).contains(&a.target_fpr),
        "--target-fpr {} must be in [0, 1)",
        a.target_fpr
    );
    let model = Arc::new(CmfModel::open(a.model).with_context(|| a.model.to_string())?);
    let at = model
        .header
        .skills
        .iter()
        .position(|s| s.id == a.id)
        .ok_or_else(|| anyhow::anyhow!("route-fit: skill '{}' not in {}", a.id, a.model))?;
    let rec = &model.header.skills[at];
    anyhow::ensure!(
        rec.is_v2(),
        "route-fit: skill '{}' is a v1 record (no kind) — router v2 routes v2 records only",
        a.id
    );
    anyhow::ensure!(
        rec.status.as_deref() != Some("retired"),
        "route-fit: skill '{}' is retired",
        a.id
    );
    let layer = match (a.phi_layer, &model.header.router) {
        (Some(l), _) => l,
        (None, Some(r)) => r.phi.layer,
        (None, None) => anyhow::bail!(
            "route-fit: --phi-layer is required ({} declares no router yet)",
            a.model
        ),
    };
    let num_layers = model.header.arch.num_layers;
    anyhow::ensure!(
        layer < num_layers,
        "--phi-layer {layer}: the model has {num_layers} layers"
    );
    if let Some(&lmin) = rec.layers.iter().min() {
        anyhow::ensure!(
            layer < lmin,
            "--phi-layer {layer} must be < min(layers) {lmin} of skill '{}' (φ must not depend \
             on the active skill)",
            a.id
        );
    }
    let margin = a
        .margin
        .or_else(|| model.header.router.as_ref().map(|r| r.margin))
        .unwrap_or(0.05);
    anyhow::ensure!(
        margin.is_finite() && margin >= 0.0,
        "--margin {margin} must be finite and ≥ 0"
    );

    let (mut skill_prompts, in_sha) = read_prompt_set(a.skill_prompts)?;
    let (mut general_prompts, general_sha) = read_prompt_set(a.general_prompts)?;
    anyhow::ensure!(
        in_sha != general_sha,
        "route-fit: --skill-prompts and --general-prompts are the same file"
    );
    let (n_skill_file, n_general_file) = (skill_prompts.len(), general_prompts.len());
    if a.max > 0 {
        skill_prompts.truncate(a.max);
        general_prompts.truncate(a.max);
    }
    for (what, n) in [
        ("--skill-prompts", skill_prompts.len()),
        ("--general-prompts", general_prompts.len()),
    ] {
        anyhow::ensure!(
            n >= 5,
            "route-fit: {what}: {n} prompts — at least 5 (20 % held out, ≥ 4 in the fit)"
        );
    }

    // φ by the runtime, on the backbone (no overlay), the cmf-im-v1 frame.
    let mut probe = Pipeline::from_model(&model, SamplerConfig::default())?;
    let spec = router::cmf_im_v1_phi_spec(&probe.tokenizer, layer)
        .map_err(|e| anyhow::anyhow!("route-fit: {e}"))?;
    if let Some(r) = &model.header.router {
        if r.phi != spec {
            let others: Vec<&str> = router::calibration_classes(&model.header)
                .into_iter()
                .filter(|s| s.id != a.id)
                .map(|s| s.id.as_str())
                .collect();
            anyhow::ensure!(
                others.is_empty(),
                "route-fit: {} routes on another φ frame (layer {}, prefix {:?}, suffix {:?}) \
                 and the calibrated classes {others:?} were fitted on it — keep --phi-layer {} \
                 or refit them together",
                a.model,
                r.phi.layer,
                r.phi.prefix_ids,
                r.phi.suffix_ids,
                r.phi.layer
            );
            eprintln!(
                "route-fit: the φ frame changes (layer {} → {layer}); the backbone descriptor \
                 is refitted with it",
                r.phi.layer
            );
        }
    }
    let skill_phi = phi_set(&mut probe, &spec, &skill_prompts, "skill");
    let general_phi = phi_set(&mut probe, &spec, &general_prompts, "general");
    drop(probe);

    let skill_desc = router::fit_descriptor(&skill_phi, layer, a.rank)
        .ok_or_else(|| anyhow::anyhow!("route-fit: the skill φ samples do not fit"))?;
    let base_desc = router::fit_descriptor(&general_phi, layer, a.rank)
        .ok_or_else(|| anyhow::anyhow!("route-fit: the general φ samples do not fit"))?;

    // The new header in memory: descriptor + policy, then the calibration
    // over every class of the file and the binding hash — one tail append.
    let mut header = model.header.clone();
    header.skills[at].selection = Some(skill_desc.clone());
    header.router = Some(RouterPolicy {
        version: 2,
        policy: "backbone_gated".into(),
        granularity: "request".into(),
        phi: spec.clone(),
        base: base_desc.clone(),
        margin,
        skills_hash: "0000000000000000".into(),
        measured: None,
    });
    let (cal, mut measured) = router::calibrate_v2(&header, a.target_fpr)
        .map_err(|e| anyhow::anyhow!("route-fit: {e}"))?;
    let hash = router::skills_hash(&header);
    measured["general_sha256"] = serde_json::json!(general_sha);
    measured["in_sha256"] = serde_json::json!(in_sha);
    measured["fitted_skill"] = serde_json::json!(a.id);
    measured["phi_layer"] = serde_json::json!(layer);
    measured["rank"] = serde_json::json!(a.rank);
    measured["n_skill_prompts"] = serde_json::json!(skill_prompts.len());
    measured["n_general_prompts"] = serde_json::json!(general_prompts.len());
    measured["fitter"] = serde_json::json!("cortiq route-fit");
    // The backend φ was computed on: a serve on another one recomputes φ
    // with another numerical class (coop GEMM ≈ tf32) and warns.
    measured["phi_backend"] = serde_json::json!(phi_backend_label());
    measured["cmf_gpu"] = serde_json::json!(std::env::var("CMF_GPU").ok());
    let classes: Vec<String> = router::calibration_classes(&header)
        .iter()
        .map(|s| s.id.clone())
        .collect();
    // Every ACTIVE v2 record — the fitted one included — was gated under
    // the previous descriptors and calibration: its measured false-accept
    // belongs to another decision surface. It is re-gated (stale_regate →
    // route-eval → skill-gate) before it auto-routes again, as the trainer
    // does on a commit that recalibrates.
    let stale: Vec<String> = model
        .header
        .skills
        .iter()
        .filter(|s| s.is_v2() && s.status.as_deref() == Some("active"))
        .map(|s| s.id.clone())
        .collect();
    let seq = header.lineage.last().map_or(0, |e| e.seq + 1);
    let event = LineageEvent::now(
        seq,
        "router_fitted",
        serde_json::json!({
            "id": a.id,
            "phi_layer": layer,
            "rank": a.rank,
            "margin": margin,
            "target_fpr": a.target_fpr,
            "n_skill": skill_prompts.len(),
            "n_general": general_prompts.len(),
            "in_sha256": in_sha,
            "general_sha256": general_sha,
            "skills_hash": hex64(hash),
            "classes": classes,
            "in_scope_recall": measured["in_scope_recall"],
            "false_accept": measured["false_accept"],
            "stale_regate": stale,
            "phi_backend": phi_backend_label(),
        }),
    );
    // The runtime replays held-out prompts of both classes against the
    // header about to be written; a disagreement refuses before a byte
    // moves.
    let mut check_header = header.clone();
    if let Some(r) = check_header.router.as_mut() {
        r.skills_hash = hex64(hash);
    }
    check_header.routing = Some(cal.clone());
    // The held-out prompts are the LAST holdout_count(n) of each set
    // (fit_descriptor); the first RUNTIME_CHECK_PER_CLASS of them replay.
    let tail = |n: usize| {
        let hold = router::holdout_count(n);
        (n - hold..n).take(RUNTIME_CHECK_PER_CLASS)
    };
    let mut cases: Vec<(&str, &[f32])> = Vec::new();
    for i in tail(skill_prompts.len()) {
        cases.push((skill_prompts[i].as_str(), skill_phi[i].as_slice()));
    }
    for i in tail(general_prompts.len()) {
        cases.push((general_prompts[i].as_str(), general_phi[i].as_slice()));
    }
    let runtime_check = runtime_parity_check(&model, &check_header, &cases, PHI_PARITY_TOL)
        .map_err(|e| anyhow::anyhow!("route-fit: {e}"))?;
    drop(check_header);
    let old_hash = model.header.router.as_ref().map(|r| r.skills_hash.clone());
    drop(model);
    let (policy_router, cal_w, measured_w, id_s, stale_w) = (
        header.router.clone().expect("set above"),
        cal.clone(),
        measured.clone(),
        a.id.to_string(),
        stale.clone(),
    );
    let report = CmfModel::update_header_append(a.model, move |h| {
        if let Some(s) = h.skills.iter_mut().find(|s| s.id == id_s) {
            s.selection = Some(skill_desc);
        }
        for s in h.skills.iter_mut() {
            if stale_w.contains(&s.id) {
                s.status = Some("stale_regate".into());
            }
        }
        let mut r = policy_router;
        r.skills_hash = hex64(hash);
        r.measured = Some(measured_w);
        h.router = Some(r);
        h.routing = Some(cal_w);
        h.lineage.push(event);
    })
    .map_err(|e| anyhow::anyhow!("route-fit: {}: {e}", a.model))?;
    let after = CmfModel::open(a.model).with_context(|| a.model.to_string())?;
    let bound = router::skills_hash(&after.header);
    anyhow::ensure!(
        bound == hash,
        "route-fit: the written file hashes its descriptors to {} but the calibration was bound \
         to {} — this must not happen",
        hex64(bound),
        hex64(hash)
    );
    let rec = after
        .header
        .skills
        .iter()
        .find(|s| s.id == a.id)
        .expect("just written");
    let j = serde_json::json!({
        "file": a.model,
        "skill": a.id,
        "kind": rec.kind,
        "status": rec.status,
        "phi": {
            "layer": spec.layer, "pool": spec.pool, "norm": spec.norm,
            "prefix_ids": spec.prefix_ids, "suffix_ids": spec.suffix_ids,
        },
        "skill_prompts": {
            "file": a.skill_prompts, "sha256": in_sha, "in_file": n_skill_file,
            "descriptor": rec.selection.as_ref().map(|d| desc_json(d, skill_prompts.len())),
        },
        "general_prompts": {
            "file": a.general_prompts, "sha256": general_sha, "in_file": n_general_file,
            "descriptor": desc_json(&base_desc, general_prompts.len()),
        },
        "margin": margin,
        "target_fpr": a.target_fpr,
        "temperature": cal.temperature,
        "novelty_theta": cal.novelty_theta,
        "samples": cal.samples,
        "in_scope_recall": measured["in_scope_recall"],
        "false_accept": measured["false_accept"],
        "false_accept_upper95": measured["false_accept_upper95"],
        "n_in": measured["n_in"],
        "n_general": measured["n_general"],
        "per_skill": measured["per_skill"],
        "measured": measured,
        "skills_hash": hex64(hash),
        "skills_hash_before": old_hash,
        "classes": router::calibration_classes(&after.header).iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
        "stale_regate": stale,
        "runtime_check": runtime_check,
        "phi_backend": phi_backend_label(),
        "auto_routable": rec.is_auto_routable(),
        "lineage_seq": seq,
        "old_len": report.old_len,
        "new_len": report.new_len,
    });
    println!("{}", serde_json::to_string_pretty(&j)?);
    eprintln!(
        "route-fit: skill '{}' @ φ layer {layer}, rank {} — in-scope recall {:.4} on {} held-out, \
         false-accept {:.4} (CP95 {:.4}) on {} general held-out; θ {:.4}, T {:.4}; skills_hash {}; \
         φ on {}{}",
        a.id,
        a.rank,
        measured["in_scope_recall"].as_f64().unwrap_or(f64::NAN),
        measured["n_in"],
        measured["false_accept"].as_f64().unwrap_or(f64::NAN),
        measured["false_accept_upper95"].as_f64().unwrap_or(f64::NAN),
        measured["n_general"],
        cal.novelty_theta,
        cal.temperature,
        hex64(hash),
        phi_backend_label(),
        if rec.is_auto_routable() {
            ""
        } else {
            " (not auto-routable yet: route-eval on disjoint sets, then skill-gate --status \
             active)"
        }
    );
    if !stale.is_empty() {
        eprintln!(
            "route-fit: {} active record(s) set to stale_regate — their gates were measured \
             under the previous router: {:?}; re-gate each (route-eval --include-quarantine on \
             the disjoint sets, skill-gate --status active) before it auto-routes again",
            stale.len(),
            stale
        );
    }
    Ok(())
}
