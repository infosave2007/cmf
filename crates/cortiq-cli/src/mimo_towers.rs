//! MiMo-V2 tower conversion: the companion file (`--mimo-towers mm-only`)
//! and the tower half of a single-file multimodal conversion
//! (`--mimo-towers multimodal`).
//!
//! The names, shapes and blob layout are the contract in
//! `cortiq_engine::mimo_mm` (module docs there); this file only decides
//! codecs and moves bytes.
//!
//! Codec policy (per tensor, first rule that applies):
//!   1. RVQ codebooks (`audio_tokenizer.encoder.quantizer.*._codebook.embed`)
//!      → F32 (the source is F32: a byte copy; argmin over distances, never
//!      rounded).
//!   2. `speech_embeddings.*` (gather tables) and every rank≠2 tensor
//!      (patch embed, convs, norms, biases, vision sinks) → EXACT: the source
//!      float encoding copied verbatim (BF16 in the release). F16 is NOT
//!      exact from BF16: measured on the release towers, 195 467 of the
//!      1 297.6 M kept BF16 values change (|Δ| ≤ 2^-25 on the 177 364 of
//!      them in visual/audio_encoder/speech: all below F16's subnormal
//!      step; `scripts/check_mimo_mm_cmf.py --f16-census`), so an F16 copy
//!      would break the bit-exact parity gates while saving nothing (BF16
//!      is 2 bytes too).
//!   3. 2-D tower matrices → a `--tensor-quant` override, else the tower
//!      base of the requested profile: q4tp for `auto`/q4tp/q2tp/q1*/vbit,
//!      otherwise the profile itself (q8_2f as the quality fallback). `f16`
//!      (profile or override) means the exact rule-2 copy — the lossless
//!      development companion.

use crate::convert::{
    Quant, SafeTensors, open_safetensors, parse_quant, quant_name, quantize_2d,
    tensor_quant_override, to_f32,
};
use cortiq_core::format::{CMF_VERSION, CmfHeader, CmfStreamWriter, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use cortiq_engine::mimo_mm::{
    AUDIO_TOKENIZER_CONFIG_BLOB, AUDIO_TOKENIZER_PREFIX, MIMO_BASE_ARCH, MIMO_MM_ARCH,
    MM_CONFIG_BLOB, MimoTokenIds, MimoTowerGroup, is_codebook, mimo_tower_inventory,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

/// Relative location of the audio tokenizer inside a MiMo checkpoint dir.
pub(crate) const AUDIO_TOKENIZER_DIR: &str = "audio_tokenizer";

/// Codebook training state dropped from the tokenizer file.
const AT_DROPPED_SUFFIXES: [&str; 3] = [
    "._codebook.cluster_size",
    "._codebook.embed_avg",
    "._codebook.inited",
];

/// The base codec of the 2-D tower matrices under a text profile. Towers
/// are small and always active; a 2-bit/1-bit/variable text profile never
/// carries over to them.
pub(crate) fn tower_base_quant(profile: Quant) -> Quant {
    match profile {
        Quant::F16 | Quant::Q8_2f | Quant::Q8Row | Quant::Q4TiledP | Quant::Q4Tiled => profile,
        _ => Quant::Q4TiledP,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TowerCodec {
    F32,
    /// The source float bytes verbatim (BF16 / F16 / F32).
    Exact,
    Matrix(Quant),
}

/// Rule category, for the provenance summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Category {
    Matrices,
    Tables,
    Codebooks,
    Other,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Self::Matrices => "matrices",
            Self::Tables => "tables",
            Self::Codebooks => "codebooks",
            Self::Other => "other",
        }
    }
}

fn category(name: &str, shape: &[usize]) -> Category {
    if is_codebook(name) {
        Category::Codebooks
    } else if name.starts_with("speech_embeddings.") {
        Category::Tables
    } else if shape.len() != 2 {
        Category::Other
    } else {
        Category::Matrices
    }
}

fn tower_codec(name: &str, shape: &[usize], profile: Quant) -> TowerCodec {
    match category(name, shape) {
        Category::Codebooks => TowerCodec::F32,
        Category::Tables | Category::Other => TowerCodec::Exact,
        Category::Matrices => {
            match tensor_quant_override(name).unwrap_or_else(|| tower_base_quant(profile)) {
                Quant::F16 => TowerCodec::Exact,
                q => TowerCodec::Matrix(q),
            }
        }
    }
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The CMF dtype that holds a safetensors float dtype byte-for-byte.
fn exact_dtype(src_dtype: &str) -> Option<TensorDtype> {
    match src_dtype {
        "BF16" => Some(TensorDtype::Bf16),
        "F16" => Some(TensorDtype::F16),
        "F32" => Some(TensorDtype::F32),
        _ => None,
    }
}

/// Encode one tower tensor. `raw` is the source payload in `src_dtype`;
/// `vals` caches its f32 decode, made only when a codec needs the values.
fn encode_tower(
    name: &str,
    shape: &[usize],
    src_dtype: &str,
    raw: &[u8],
    vals: &mut Option<Vec<f32>>,
    profile: Quant,
) -> anyhow::Result<TensorSpec> {
    let codec = tower_codec(name, shape, profile);
    let exact = exact_dtype(src_dtype);
    let needs_values = match codec {
        TowerCodec::Matrix(_) => true,
        TowerCodec::F32 => exact != Some(TensorDtype::F32),
        TowerCodec::Exact => exact.is_none(),
    };
    if needs_values && vals.is_none() {
        *vals = Some(to_f32(src_dtype, raw)?);
    }
    let (dtype, data) = match (codec, exact) {
        (TowerCodec::F32, Some(TensorDtype::F32)) => (TensorDtype::F32, raw.to_vec()),
        (TowerCodec::Exact, Some(d)) => (d, raw.to_vec()),
        (TowerCodec::F32 | TowerCodec::Exact, _) => {
            (TensorDtype::F32, f32_bytes(vals.as_ref().unwrap()))
        }
        (TowerCodec::Matrix(q), _) => quantize_2d(q, vals.as_ref().unwrap(), shape[0], shape[1]),
    };
    Ok(TensorSpec {
        name: name.to_string(),
        dtype,
        shape: shape.to_vec(),
        data,
    })
}

/// What one output received, for the provenance block.
#[derive(Default, Clone, Debug)]
pub(crate) struct TowerStats {
    /// (group, category) → dtype name → count.
    codecs: BTreeMap<(MimoTowerGroup, Category), BTreeMap<&'static str, usize>>,
    counts: BTreeMap<MimoTowerGroup, usize>,
    bytes: BTreeMap<MimoTowerGroup, u64>,
}

impl TowerStats {
    fn record(&mut self, spec: &TensorSpec) {
        let Some(g) = MimoTowerGroup::of(&spec.name) else {
            return;
        };
        *self
            .codecs
            .entry((g, category(&spec.name, &spec.shape)))
            .or_default()
            .entry(spec.dtype.name())
            .or_default() += 1;
        *self.counts.entry(g).or_default() += 1;
        *self.bytes.entry(g).or_default() += spec.data.len() as u64;
    }

    /// `{group: {category: dtype | {dtype: n, …}}}`.
    fn codec_json(&self) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for ((g, c), hist) in &self.codecs {
            let v = if hist.len() == 1 {
                serde_json::json!(hist.keys().next().unwrap())
            } else {
                serde_json::json!(hist)
            };
            out.entry(g.label())
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .unwrap()
                .insert(c.label().into(), v);
        }
        serde_json::Value::Object(out)
    }

    fn counts_json(&self) -> serde_json::Value {
        serde_json::json!(
            self.counts
                .iter()
                .map(|(g, n)| (g.label(), *n))
                .collect::<BTreeMap<_, _>>()
        )
    }

    fn bytes_json(&self) -> serde_json::Value {
        serde_json::json!(
            self.bytes
                .iter()
                .map(|(g, n)| (g.label(), *n))
                .collect::<BTreeMap<_, _>>()
        )
    }
}

/// Encode one tower tensor into every active output (also the multimodal
/// path of the text converter). `raw` is the source payload in `src_dtype`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_tower(
    batches: &mut [Vec<TensorSpec>],
    profiles: &[Quant],
    active: &[bool],
    stats: &mut [TowerStats],
    name: &str,
    shape: &[usize],
    src_dtype: &str,
    raw: &[u8],
) -> anyhow::Result<()> {
    let numel: usize = shape.iter().product();
    let mut vals = None;
    for (i, (batch, &profile)) in batches.iter_mut().zip(profiles).enumerate() {
        if !active[i] {
            continue;
        }
        let spec = encode_tower(name, shape, src_dtype, raw, &mut vals, profile)?;
        let width = match spec.dtype {
            TensorDtype::F32 => Some(4),
            TensorDtype::F16 | TensorDtype::Bf16 => Some(2),
            _ => None,
        };
        if let Some(w) = width {
            anyhow::ensure!(
                spec.data.len() == numel * w,
                "tower tensor '{name}': {} bytes for shape {shape:?} ({})",
                spec.data.len(),
                spec.dtype.name()
            );
        }
        stats[i].record(&spec);
        batch.push(spec);
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// HF download metadata (`.cache/huggingface/download/<file>.metadata`,
/// first line = commit) — the source revision when the checkpoint came
/// from `hf download --local-dir`.
fn hf_revision(dir: &Path, file: &str) -> Option<String> {
    let p = dir
        .join(".cache/huggingface/download")
        .join(format!("{file}.metadata"));
    let text = fs::read_to_string(p).ok()?;
    let line = text.lines().next()?.trim();
    (line.len() == 40 && line.bytes().all(|b| b.is_ascii_hexdigit())).then(|| line.to_string())
}

/// Everything the tower tail of a conversion needs from the checkpoint dir.
pub(crate) struct TowerSource {
    pub(crate) dir: PathBuf,
    pub(crate) config_bytes: Vec<u8>,
    pub(crate) at_config_bytes: Vec<u8>,
    pub(crate) config: serde_json::Value,
    /// `(name, shape)` in write order.
    pub(crate) inventory: Vec<(String, Vec<usize>)>,
    at_file: SafeTensors,
    /// Source tensors of the tokenizer file that were dropped.
    at_dropped_decoder: usize,
    at_dropped_state: usize,
}

impl TowerSource {
    /// Read and validate the configs and the audio tokenizer file. Errors
    /// early — before a multi-hour text conversion starts.
    pub(crate) fn open(dir: &Path) -> anyhow::Result<Self> {
        let config_bytes = fs::read(dir.join("config.json"))
            .map_err(|e| anyhow::anyhow!("read {}/config.json: {e}", dir.display()))?;
        let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let mt = config.get("model_type").and_then(|v| v.as_str());
        anyhow::ensure!(
            mt == Some(MIMO_BASE_ARCH),
            "--mimo-towers: model_type {mt:?} is not {MIMO_BASE_ARCH}"
        );
        MimoTokenIds::from_config(&config)
            .and_then(|t| t.check_pinned("config.json"))
            .map_err(|e| anyhow::anyhow!(e))?;
        let at_dir = dir.join(AUDIO_TOKENIZER_DIR);
        let at_config_bytes = fs::read(at_dir.join("config.json")).map_err(|e| {
            anyhow::anyhow!(
                "read {}/config.json: {e} (the audio tokenizer is part of the towers; \
                 download {AUDIO_TOKENIZER_DIR}/ from the same repo)",
                at_dir.display()
            )
        })?;
        let at_config: serde_json::Value = serde_json::from_slice(&at_config_bytes)?;
        let inventory =
            mimo_tower_inventory(&config, &at_config).map_err(|e| anyhow::anyhow!(e))?;
        let at_file = open_safetensors(&at_dir.join("model.safetensors"))?;
        // Classify every tokenizer tensor: encoder (kept unless training
        // state), decoder (dropped), anything else is an unknown layout.
        let (mut dec, mut state) = (0usize, 0usize);
        for t in &at_file.tensors {
            if t.name.starts_with("decoder.") {
                dec += 1;
            } else if t.name.starts_with("encoder.") {
                if AT_DROPPED_SUFFIXES.iter().any(|s| t.name.ends_with(s)) {
                    state += 1;
                }
            } else {
                anyhow::bail!(
                    "{AUDIO_TOKENIZER_DIR}/model.safetensors: tensor '{}' is neither encoder.* nor decoder.*",
                    t.name
                );
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            config_bytes,
            at_config_bytes,
            config,
            inventory,
            at_file,
            at_dropped_decoder: dec,
            at_dropped_state: state,
        })
    }

    /// Canonical CMF name of an audio tokenizer source tensor, or None when
    /// it is dropped.
    fn at_name(raw: &str) -> Option<String> {
        if !raw.starts_with("encoder.") || AT_DROPPED_SUFFIXES.iter().any(|s| raw.ends_with(s)) {
            return None;
        }
        Some(format!("{AUDIO_TOKENIZER_PREFIX}{raw}"))
    }

    /// Emit the audio tokenizer encoder and the two config blobs into every
    /// active output (the tail of a multimodal conversion).
    pub(crate) fn emit_tail(
        &self,
        batches: &mut [Vec<TensorSpec>],
        profiles: &[Quant],
        active: &[bool],
        stats: &mut [TowerStats],
    ) -> anyhow::Result<()> {
        for (i, batch) in batches.iter_mut().enumerate() {
            if active[i] {
                batch.extend(self.blobs());
            }
        }
        let want: HashMap<&str, &[usize]> = self
            .inventory
            .iter()
            .filter(|(n, _)| MimoTowerGroup::of(n) == Some(MimoTowerGroup::AudioTokenizer))
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let mut seen = 0usize;
        for t in &self.at_file.tensors {
            let Some(name) = Self::at_name(&t.name) else {
                continue;
            };
            let shape = want.get(name.as_str()).ok_or_else(|| {
                anyhow::anyhow!(
                    "audio tokenizer tensor '{}' is not in the tower layout",
                    t.name
                )
            })?;
            anyhow::ensure!(
                t.shape.as_slice() == *shape,
                "audio tokenizer '{}': shape {:?}, expected {shape:?}",
                t.name,
                t.shape
            );
            emit_tower(
                batches,
                profiles,
                active,
                stats,
                &name,
                &t.shape,
                &t.dtype,
                self.at_file.bytes(t),
            )?;
            seen += 1;
        }
        anyhow::ensure!(
            seen == want.len(),
            "audio tokenizer: {seen} encoder tensors, the layout needs {}",
            want.len()
        );
        Ok(())
    }

    fn blobs(&self) -> [TensorSpec; 2] {
        let blob = |name: &str, b: &[u8]| TensorSpec {
            name: name.into(),
            dtype: TensorDtype::U8,
            shape: vec![b.len()],
            data: b.to_vec(),
        };
        [
            blob(MM_CONFIG_BLOB, &self.config_bytes),
            blob(AUDIO_TOKENIZER_CONFIG_BLOB, &self.at_config_bytes),
        ]
    }

    /// `provenance.mimo_mm` for one output.
    pub(crate) fn provenance(&self, towers: &str, stats: &TowerStats) -> serde_json::Value {
        let tokens = MimoTokenIds::PINNED
            .named()
            .iter()
            .map(|(n, id)| (n.to_string(), *id))
            .collect::<BTreeMap<_, _>>();
        serde_json::json!({
            "base_arch": MIMO_BASE_ARCH,
            "towers": towers,
            "hidden_size": self.config.get("hidden_size"),
            "special_tokens": tokens,
            "source_revision": hf_revision(&self.dir, "config.json"),
            "audio_tokenizer_revision": hf_revision(&self.dir, "audio_tokenizer/config.json"),
            "config_sha256": sha256_hex(&self.config_bytes),
            "audio_tokenizer_config_sha256": sha256_hex(&self.at_config_bytes),
            "codec": stats.codec_json(),
            "tensor_counts": stats.counts_json(),
            "payload_bytes": stats.bytes_json(),
            "dropped": {
                "audio_tokenizer.decoder": self.at_dropped_decoder,
                "audio_tokenizer.codebook_state": self.at_dropped_state,
            },
            "codec_policy": "2-D matrices: --tensor-quant override, else q4tp (q8_2f/q8/q4t profiles carry over, \
                             f16 = exact source copy); speech_embeddings + rank!=2: exact source copy (bf16); \
                             RVQ codebooks: f32",
            "gates": {},
        })
    }
}

/// The source files holding `visual.*`, `audio_encoder.*` and
/// `speech_embeddings.*` (from the index, or the single-file checkpoint).
fn tower_shards(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    let idx: serde_json::Value = serde_json::from_slice(
        &fs::read(&index).map_err(|e| anyhow::anyhow!("read {}: {e}", index.display()))?,
    )?;
    let map = idx["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{}: no weight_map", index.display()))?;
    let mut files: Vec<String> = map
        .iter()
        .filter(|(k, _)| MimoTowerGroup::of(k).is_some())
        .filter_map(|(_, v)| v.as_str().map(String::from))
        .collect();
    files.sort();
    files.dedup();
    anyhow::ensure!(
        !files.is_empty(),
        "{}: no visual.* / audio_encoder.* / speech_embeddings.* tensors",
        index.display()
    );
    Ok(files.iter().map(|f| dir.join(f)).collect())
}

/// Header architecture of the companion: the MiMo arch name and the LLM
/// hidden size the towers project into; it has no decoder layers.
fn companion_arch(config: &serde_json::Value) -> anyhow::Result<ModelArch> {
    let hidden = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("config.json: no hidden_size"))?;
    let vocab = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": MIMO_MM_ARCH,
        "hidden_size": hidden,
        "intermediate_size": 0,
        "num_layers": 0,
        "num_attention_heads": 0,
        "num_kv_heads": 0,
        "head_dim": 0,
        "vocab_size": vocab,
        "layer_types": [],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 0,
    }))?;
    Ok(arch)
}

fn quant_type(q: Quant) -> QuantType {
    match q {
        Quant::F16 => QuantType::F16,
        Quant::Q8_2f => QuantType::Q8_2f,
        Quant::Q8Row => QuantType::Q8Row,
        _ => QuantType::Q4Block,
    }
}

/// `cortiq convert --mimo-towers mm-only`: write the companion
/// `<stem>.mm.cmf` for each requested profile (`auto` = q4tp), reading only
/// the shard(s) that hold the towers and `audio_tokenizer/`.
pub fn run_convert_mimo_mm(
    model: &str,
    outputs: &[(String, String)],
    mut progress: impl FnMut(f32),
) -> anyhow::Result<()> {
    let dir = Path::new(model);
    anyhow::ensure!(
        dir.join("config.json").exists(),
        "--mimo-towers mm-only needs a local checkpoint dir (config.json, the index, the shard \
         holding visual.*, and {AUDIO_TOKENIZER_DIR}/); '{model}' is not one"
    );
    anyhow::ensure!(!outputs.is_empty(), "at least one output is required");
    let profiles: Vec<Quant> = outputs
        .iter()
        .map(|(q, _)| {
            if q.trim().eq_ignore_ascii_case(crate::convert::AUTO_QUANT) {
                Ok(Quant::Q4TiledP)
            } else {
                parse_quant(q)
            }
        })
        .collect::<anyhow::Result<_>>()?;
    for (_, p) in outputs {
        if !p.ends_with(cortiq_engine::mimo_mm::MM_SUFFIX) {
            eprintln!(
                "  note: '{p}' does not end in {} — sibling discovery will not find it (use --mm)",
                cortiq_engine::mimo_mm::MM_SUFFIX
            );
        }
    }
    let src = TowerSource::open(dir)?;
    let shards: Vec<SafeTensors> = tower_shards(dir)?
        .iter()
        .map(|p| open_safetensors(p))
        .collect::<anyhow::Result<_>>()?;
    // name → (shard, tensor index) for the three main-file groups.
    let mut located: HashMap<&str, (usize, usize)> = HashMap::new();
    for (si, f) in shards.iter().enumerate() {
        for (ti, t) in f.tensors.iter().enumerate() {
            if MimoTowerGroup::of(&t.name).is_some() {
                anyhow::ensure!(
                    located.insert(t.name.as_str(), (si, ti)).is_none(),
                    "tower tensor '{}' appears twice in the source",
                    t.name
                );
            }
        }
    }
    let main_inv: Vec<&(String, Vec<usize>)> = src
        .inventory
        .iter()
        .filter(|(n, _)| MimoTowerGroup::of(n) != Some(MimoTowerGroup::AudioTokenizer))
        .collect();
    let missing: Vec<&str> = main_inv
        .iter()
        .filter(|(n, _)| !located.contains_key(n.as_str()))
        .map(|(n, _)| n.as_str())
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{} tower tensors missing from the checkpoint, e.g. {:?}",
        missing.len(),
        &missing[..missing.len().min(4)]
    );
    anyhow::ensure!(
        located.len() == main_inv.len(),
        "checkpoint has {} tower tensors, the layout expects {} — unknown tower tensor(s)",
        located.len(),
        main_inv.len()
    );

    let n_total = src.inventory.len() + 2;
    let reserve = CmfStreamWriter::head_reserve_for(2 * n_total, 96);
    let mut writers: Vec<CmfStreamWriter> = outputs
        .iter()
        .map(|(_, p)| {
            CmfStreamWriter::new(p, reserve).map_err(|e| anyhow::anyhow!("create {p}: {e}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let active = vec![true; outputs.len()];
    let mut stats = vec![TowerStats::default(); outputs.len()];
    let mut batches: Vec<Vec<TensorSpec>> = outputs.iter().map(|_| Vec::new()).collect();
    let drain = |batches: &mut [Vec<TensorSpec>], writers: &mut [CmfStreamWriter]| {
        for (b, w) in batches.iter_mut().zip(writers.iter_mut()) {
            for t in b.drain(..) {
                w.push(&t.name, t.dtype, &t.shape, &t.data)
                    .map_err(|e| anyhow::anyhow!("write '{}': {e}", t.name))?;
            }
        }
        anyhow::Ok(())
    };
    for (k, (name, shape)) in main_inv.iter().enumerate() {
        let (si, ti) = located[name.as_str()];
        let t = &shards[si].tensors[ti];
        anyhow::ensure!(
            &t.shape == shape,
            "'{name}': source shape {:?}, expected {shape:?}",
            t.shape
        );
        emit_tower(
            &mut batches,
            &profiles,
            &active,
            &mut stats,
            name,
            shape,
            &t.dtype,
            shards[si].bytes(t),
        )?;
        drain(&mut batches, &mut writers)?;
        progress((k + 1) as f32 / n_total as f32);
    }
    src.emit_tail(&mut batches, &profiles, &active, &mut stats)?;
    drain(&mut batches, &mut writers)?;

    let arch = companion_arch(&src.config)?;
    for (i, w) in writers.into_iter().enumerate() {
        let mut prov = serde_json::json!({
            "tool": "cortiq convert --mimo-towers mm-only",
            "source_model": model,
            "weight_quant": quant_name(profiles[i]),
        });
        prov["mimo_mm"] = src.provenance("mm-only", &stats[i]);
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch: arch.clone(),
            quant_type: quant_type(profiles[i]),
            provenance: Some(prov),
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
            routing: None,
        };
        w.finish(&header, None, None)
            .map_err(|e| anyhow::anyhow!("write {}: {e}", outputs[i].1))?;
        eprintln!(
            "  {}: {} tower tensors + 2 config blobs, codec {}",
            outputs[i].1,
            stats[i].counts.values().sum::<usize>(),
            stats[i].codec_json(),
        );
    }
    progress(1.0);
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cortiq_core::CmfModel;
    use cortiq_engine::mimo_mm::MimoMm;

    #[test]
    fn tower_codec_policy() {
        let _env = crate::convert::tests::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::convert::set_tensor_quant_overrides(&[]).unwrap();
        let q = Quant::Q4TiledP;
        assert_eq!(
            tower_codec("visual.blocks.0.attn.qkv.weight", &[3072, 1280], q),
            TowerCodec::Matrix(Quant::Q4TiledP)
        );
        assert_eq!(
            tower_codec("visual.patch_embed.proj.weight", &[1280, 3, 2, 16, 16], q),
            TowerCodec::Exact
        );
        assert_eq!(
            tower_codec("speech_embeddings.3.weight", &[1280, 1024], q),
            TowerCodec::Exact
        );
        assert_eq!(
            tower_codec(
                "audio_tokenizer.encoder.quantizer.vq.layers.4._codebook.embed",
                &[128, 1024],
                Quant::F16
            ),
            TowerCodec::F32
        );
        assert_eq!(
            tower_codec(
                "audio_encoder.projection.mlp.0.weight",
                &[16384, 4096],
                Quant::F16
            ),
            TowerCodec::Exact
        );
        assert_eq!(
            tower_codec("visual.merger.mlp.0.weight", &[5120, 5120], Quant::Q2TiledP),
            TowerCodec::Matrix(Quant::Q4TiledP)
        );
        assert_eq!(
            tower_codec("visual.merger.mlp.0.weight", &[5120, 5120], Quant::Q8_2f),
            TowerCodec::Matrix(Quant::Q8_2f)
        );
    }

    #[test]
    fn exact_codec_copies_the_source_bytes() {
        let _env = crate::convert::tests::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::convert::set_tensor_quant_overrides(&[]).unwrap();
        // 2^-20·(1 + 2^-7) is a bf16 value F16 cannot hold (below its
        // 2^-24 subnormal step): the exact codec keeps the bf16 bits.
        let tiny = 2f32.powi(-20) * (1.0 + 2f32.powi(-7));
        let raw = bf16_bytes(&[0.5, tiny]);
        let mut vals = None;
        let spec = encode_tower(
            "visual.blocks.1.attn.sinks",
            &[2],
            "BF16",
            &raw,
            &mut vals,
            Quant::Q4TiledP,
        )
        .unwrap();
        assert_eq!(
            (spec.dtype, spec.data.as_slice()),
            (TensorDtype::Bf16, &raw[..])
        );
        assert!(vals.is_none(), "an exact copy never decodes");
        // A codebook from a BF16 source is widened to F32.
        let spec = encode_tower(
            "audio_tokenizer.encoder.quantizer.vq.layers.0._codebook.embed",
            &[1, 2],
            "BF16",
            &raw,
            &mut vals,
            Quant::Q4TiledP,
        )
        .unwrap();
        assert_eq!(spec.dtype, TensorDtype::F32);
        assert_eq!(spec.data, f32_bytes(&[0.5, tiny]));
    }

    pub(crate) type Raw = (String, &'static str, Vec<usize>, Vec<u8>);

    fn raw_safetensors(tensors: &[Raw]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        out
    }

    /// Deterministic bf16-exact values (with a few bf16-only tiny ones in
    /// `tiny_at`, to exercise the F32 fallback).
    fn bf16_vals(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i * 31 + seed * 17) % 101) as f32 / 101.0 - 0.5;
                f32::from_bits((x * 0.5).to_bits() & 0xFFFF_0000)
            })
            .collect()
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// A tiny MiMo tower config: vision depth 3 (types [−1, 0, 1]), hidden
    /// 64, 4 heads / 2 kv × 16; audio 2 local layers of 32 over 3 channels;
    /// tokenizer 2 layers of 32 with 3 codebooks.
    pub(crate) fn tiny_configs() -> (serde_json::Value, serde_json::Value) {
        let cfg = serde_json::json!({
            "model_type": "mimo_v2",
            "hidden_size": 64,
            "vocab_size": 151680,
            "vision_start_token_id": 151652, "vision_end_token_id": 151653,
            "image_token_id": 151655, "video_token_id": 151656,
            "audio_token_id": 151669, "audio_start_token_id": 151673, "audio_end_token_id": 151674,
            "processor_config": {"video_start_token_id": 151670, "video_end_token_id": 151671},
            "vision_config": {
                "depth": 3, "hidden_size": 64, "intermediate_size": 96, "num_heads": 4,
                "num_key_value_heads": 2, "qk_channels": 16, "out_hidden_size": 64,
                "in_chans": 3, "patch_size": 4, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "fullatt_block_indexes": [0],
                "vit_window_attn_types": [-1, 0, 1], "visual_token_window_size": 4,
                "use_sink": true, "hidden_act": "silu", "window_size": 128
            },
            "audio_config": {
                "audio_channels": 3, "group_size": 4, "input_local_dim": 32,
                "input_local_layers": 2, "input_local_attn_heads": 2, "input_local_head_dim": 16,
                "input_local_intermediate_size": 64, "out_hidden_size": 64,
                "projection_layers": 2, "rope_theta": 640000, "add_post_norm": true,
                "speech_vocab_size": "40", "speech_zeroemb_idx": "32"
            }
        });
        let at = serde_json::json!({
            "d_model": 32, "encoder_layers": 2, "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64, "n_mels": 8, "kernel_size": 3, "stride_size": 2,
            "avg_pooler": 2, "encoder_skip_layer_id": 1, "encoder_causal": true,
            "encoder_attn_window_size": [4, 0], "hybrid_attention": true, "swa_per_block": 2,
            "rope_theta": 10000, "num_quantizers": 3, "codebook_size": [32, 16, 8],
            "sampling_rate": 24000, "hop_length": 240, "nfft": 960, "window_size": 960,
            "ln_type": "LayerNorm", "scale_embedding": false
        });
        (cfg, at)
    }

    /// Tensors of the tiny tower checkpoint: `(main-shard tensors,
    /// audio-tokenizer-file tensors incl. dropped ones, expected CMF values)`.
    /// `visual.blocks.1.attn.sinks` carries one bf16 value F16 cannot hold.
    #[allow(clippy::type_complexity)]
    pub(crate) fn tower_fixture_tensors()
    -> (Vec<Raw>, Vec<Raw>, HashMap<String, (Vec<usize>, Vec<f32>)>) {
        let (cfg, at) = tiny_configs();
        let inv = mimo_tower_inventory(&cfg, &at).unwrap();
        let mut main: Vec<Raw> = Vec::new();
        let mut tok: Vec<Raw> = Vec::new();
        let mut want = HashMap::new();
        for (k, (name, shape)) in inv.iter().enumerate() {
            let n: usize = shape.iter().product();
            if is_codebook(name) {
                let v: Vec<f32> = (0..n)
                    .map(|i| (i as f32 * 0.37 + k as f32).sin() * 1e-3)
                    .collect();
                let raw = name
                    .strip_prefix(AUDIO_TOKENIZER_PREFIX)
                    .unwrap()
                    .to_string();
                tok.push((raw, "F32", shape.clone(), f32_bytes(&v)));
                want.insert(name.clone(), (shape.clone(), v));
                continue;
            }
            let mut v = bf16_vals(n, k);
            if name == "visual.blocks.1.attn.sinks" {
                v[1] = 2f32.powi(-20) * (1.0 + 2f32.powi(-7));
            }
            let bytes = bf16_bytes(&v);
            match name.strip_prefix(AUDIO_TOKENIZER_PREFIX) {
                Some(raw) => tok.push((raw.to_string(), "BF16", shape.clone(), bytes)),
                None => main.push((name.clone(), "BF16", shape.clone(), bytes)),
            }
            want.insert(name.clone(), (shape.clone(), v));
        }
        // Dropped tokenizer tensors.
        for q in 0..3 {
            let p = format!("encoder.quantizer.vq.layers.{q}._codebook");
            tok.push((format!("{p}.cluster_size"), "F32", vec![4], vec![0; 16]));
            tok.push((format!("{p}.inited"), "F32", vec![1], vec![0; 4]));
            tok.push((format!("{p}.embed_avg"), "F32", vec![4, 2], vec![0; 32]));
        }
        tok.push((
            "decoder.layers.0.fc1.weight".into(),
            "BF16",
            vec![4, 4],
            vec![0; 32],
        ));
        (main, tok, want)
    }

    /// Write `audio_tokenizer/{config.json, model.safetensors}` into `dir`.
    pub(crate) fn write_audio_tokenizer(dir: &Path, tok: &[Raw]) {
        let (_, at) = tiny_configs();
        fs::create_dir_all(dir.join(AUDIO_TOKENIZER_DIR)).unwrap();
        fs::write(
            dir.join(AUDIO_TOKENIZER_DIR).join("config.json"),
            serde_json::to_vec_pretty(&at).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join(AUDIO_TOKENIZER_DIR).join("model.safetensors"),
            raw_safetensors(tok),
        )
        .unwrap();
    }

    /// A tiny MiMo checkpoint dir holding only the towers (index + one
    /// shard + audio_tokenizer/), plus one text tensor that must be ignored.
    fn write_tower_fixture(tag: &str) -> (PathBuf, HashMap<String, (Vec<usize>, Vec<f32>)>) {
        let dir = std::env::temp_dir().join(format!("cortiq-mimo-mm-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let (cfg, _) = tiny_configs();
        let (mut main, tok, want) = tower_fixture_tensors();
        main.push(("model.norm.weight".into(), "BF16", vec![64], vec![0; 128]));
        let shard = "model_pp0_ep0_shard0.safetensors";
        fs::write(dir.join(shard), raw_safetensors(&main)).unwrap();
        let mut wm = serde_json::Map::new();
        for (n, ..) in &main {
            wm.insert(n.clone(), serde_json::json!(shard));
        }
        // An expert shard that does not exist: mm-only must not open it.
        wm.insert(
            "model.layers.1.mlp.experts.0.up_proj.weight".into(),
            serde_json::json!("model_pp0_ep1_shard0.safetensors"),
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map": wm}).to_string(),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&cfg).unwrap(),
        )
        .unwrap();
        write_audio_tokenizer(&dir, &tok);
        (dir, want)
    }

    #[test]
    fn mm_only_writes_the_exact_companion() {
        // --tensor-quant overrides are process-global.
        let _env = crate::convert::tests::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::convert::set_tensor_quant_overrides(&[]).unwrap();
        let (dir, want) = write_tower_fixture("companion");
        let q = dir.join("x.mm.cmf");
        let f = dir.join("x-f16.mm.cmf");
        run_convert_mimo_mm(
            dir.to_str().unwrap(),
            &[
                ("auto".into(), q.to_str().unwrap().into()),
                ("f16".into(), f.to_str().unwrap().into()),
            ],
            |_| {},
        )
        .unwrap();
        let (cfg, at) = tiny_configs();
        let inv = mimo_tower_inventory(&cfg, &at).unwrap();
        for path in [&q, &f] {
            let m = std::sync::Arc::new(CmfModel::open(path).unwrap());
            assert!(m.verify().is_empty(), "{:?}", m.verify());
            assert_eq!(m.arch().arch_name, MIMO_MM_ARCH);
            assert_eq!(m.tensors.len(), inv.len() + 2);
            let mm = MimoMm::from_model(&m).unwrap();
            assert_eq!(mm.hidden_size, 64);
            let prov = mm.provenance.as_ref().unwrap();
            assert_eq!(prov["base_arch"], "mimo_v2");
            assert_eq!(prov["dropped"]["audio_tokenizer.decoder"], 1);
            assert_eq!(prov["dropped"]["audio_tokenizer.codebook_state"], 9);
            // Config blobs are the source bytes.
            assert_eq!(
                m.tensor_bytes(MM_CONFIG_BLOB).unwrap(),
                fs::read(dir.join("config.json")).unwrap().as_slice()
            );
            assert!(m.tensor("model.norm.weight").is_none());
            for (name, (shape, vals)) in &want {
                let e = m.tensor(name).unwrap_or_else(|| panic!("{name}"));
                assert_eq!(&e.shape, shape, "{name}");
                let bytes = m.tensor_bytes(name).unwrap();
                let cat = category(name, shape);
                let is_q = path == &q && cat == Category::Matrices;
                if is_codebook(name) {
                    assert_eq!(e.dtype, TensorDtype::F32, "{name}");
                    assert_eq!(bytes, f32_bytes(vals).as_slice(), "{name}");
                } else if is_q {
                    assert_eq!(e.dtype, TensorDtype::Q4TiledP, "{name}");
                    let back = mm.f32(name).unwrap();
                    let dot: f64 = back.iter().zip(vals).map(|(a, b)| (a * b) as f64).sum();
                    let na: f64 = back.iter().map(|a| (a * a) as f64).sum::<f64>().sqrt();
                    let nb: f64 = vals.iter().map(|b| (b * b) as f64).sum::<f64>().sqrt();
                    assert!(dot / (na * nb) > 0.99, "{name}: cos {}", dot / (na * nb));
                } else {
                    // Exact: the bf16 source bytes, including the value
                    // F16 could not hold (visual.blocks.1.attn.sinks[1]).
                    assert_eq!(e.dtype, TensorDtype::Bf16, "{name}");
                    assert_eq!(bytes, bf16_bytes(vals).as_slice(), "{name}");
                    assert_eq!(&mm.f32(name).unwrap(), vals, "{name}");
                }
            }
        }
        let prov_q = MimoMm::open(&q).unwrap().provenance.unwrap();
        assert_eq!(prov_q["codec"]["visual"]["matrices"], "q4tp");
        assert_eq!(prov_q["codec"]["speech_embeddings"]["tables"], "bf16");
        assert_eq!(prov_q["codec"]["visual"]["other"], "bf16");
        assert_eq!(prov_q["codec"]["audio_tokenizer"]["codebooks"], "f32");
        assert_eq!(prov_q["tensor_counts"]["visual"], 3 * 12 + 2 + 1 + 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mm_only_refuses_a_checkpoint_without_the_audio_tokenizer() {
        let (dir, _) = write_tower_fixture("noat");
        fs::remove_dir_all(dir.join(AUDIO_TOKENIZER_DIR)).unwrap();
        let out = dir.join("y.mm.cmf");
        let err = run_convert_mimo_mm(
            dir.to_str().unwrap(),
            &[("auto".into(), out.to_str().unwrap().into())],
            |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("audio_tokenizer"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }
}
