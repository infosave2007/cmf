//! Export a genome checkpoint into a `.cmf` container the runtime loads:
//! tensor names as the loader expects them (`vmf_attn.*` for the hybrid_k
//! mixer, `self_attn.*` for the anchor, `mlp.shared_expert.*` +
//! `mlp.experts.{e}.*` + `mlp.desc.*` for the routed experts,
//! `lm_head.clusters.weight` for the hierarchical head), the arch block, and
//! our tokenizer.json in the VOCAB section.

use crate::model::{EmbryoCfg, LayerOffs, Layout};
use crate::train::Checkpoint;
use cortiq_core::format::{CmfHeader, CmfModel, TensorEntry, TensorSpec, TokenizerBundle};
use cortiq_core::knowledge::{GenomeInfo, LineageEvent, hex64, trunk_hash};
use cortiq_core::types::TensorDtype;
use std::path::Path;

/// Genome statuses `export --genome-status` accepts (spec §9.2; `rejected`
/// is a verdict a gate writes later, never a birth status).
pub const EXPORT_GENOME_STATUSES: &[&str] = &["pre_chat", "candidate", "sealed"];

/// The frozen-genome identity an export stamps into `header.genome`
/// (bit `GENOME`): generation 0, no parent, the trunk hash of the bytes
/// written and of the f32 master, lineage `[birth]`.
#[derive(Clone, Debug)]
pub struct ExportGenome {
    pub id: String,
    /// `pre_chat` | `candidate` | `sealed`.
    pub status: String,
}

impl ExportGenome {
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.id.is_empty() && !self.id.chars().any(|c| c.is_whitespace() || c.is_control()),
            "genome id '{}' must be non-empty without whitespace",
            self.id
        );
        anyhow::ensure!(
            EXPORT_GENOME_STATUSES.contains(&self.status.as_str()),
            "genome status '{}' is not one of {EXPORT_GENOME_STATUSES:?}",
            self.status
        );
        Ok(())
    }
}

/// Directory entries (hash64 of the payload; offsets irrelevant) of the
/// specs as a writer would record them — the input of the core
/// `trunk_hash`.
pub fn spec_entries(specs: &[TensorSpec]) -> Vec<TensorEntry> {
    specs
        .iter()
        .map(|t| TensorEntry {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            off: 0,
            nbytes: t.data.len() as u64,
            shard: 0,
            hash: cortiq_core::hash64(&t.data),
        })
        .collect()
}

fn f32_bytes(x: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(x.len() * 4);
    for f in x {
        v.extend_from_slice(&f.to_le_bytes());
    }
    v
}

fn f16_bytes(x: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(x.len() * 2);
    for f in x {
        v.extend_from_slice(&cortiq_core::quant::f32_to_f16(*f).to_le_bytes());
    }
    v
}

pub(crate) fn spec(name: &str, shape: &[usize], data: &[f32], export_dtype: TensorDtype) -> TensorSpec {
    assert_eq!(
        shape.iter().product::<usize>(),
        data.len(),
        "{name}: shape/len mismatch"
    );
    // The f16 profile follows the runtime converter's conservative rule:
    // learned matrices (including the tied embedding/head and conv kernels)
    // are half precision, while 1-D norms/biases and routing descriptors stay
    // exact.  Descriptor rounding can change the winning expert even when the
    // forward error itself is tiny, so it is deliberately not part of this
    // storage optimization.
    let use_f16 = export_dtype == TensorDtype::F16 && shape.len() >= 2 && !name.contains(".desc.");
    let (dtype, data) = if use_f16 {
        (TensorDtype::F16, f16_bytes(data))
    } else {
        (TensorDtype::F32, f32_bytes(data))
    };
    TensorSpec {
        name: name.to_string(),
        dtype,
        shape: shape.to_vec(),
        data,
    }
}

/// The runtime arch block for a genome (JSON → ModelArch through serde so
/// every optional field takes its default).
pub fn arch_json(cfg: &EmbryoCfg) -> serde_json::Value {
    // `export_with_dtype` runs the fallible preflight before this helper.  A
    // direct caller still gets a loud failure instead of a malformed legacy
    // descriptor that would silently change the operator.
    let phase_delta_layers = cfg
        .phase_delta_layers_for_export()
        .expect("invalid Phase-Delta selector");
    let layer_types: Vec<&str> = (0..cfg.layers)
        .map(|l| {
            if cfg.is_anchor(l) {
                if cfg.anchor_bounded() {
                    "BoundedAttention"
                } else {
                    "FullAttention"
                }
            } else {
                "LinearAttention"
            }
        })
        .collect();
    let mut j = serde_json::json!({
        "arch_name": "cortiq_embryo",
        "hidden_size": cfg.hidden,
        "intermediate_size": cfg.inter,
        "num_layers": cfg.layers,
        "num_attention_heads": cfg.anchor_q_heads,
        "num_kv_heads": cfg.anchor_kv_heads,
        "head_dim": cfg.anchor_hd,
        "vocab_size": cfg.vocab,
        "layer_types": layer_types,
        "rms_norm_eps": cfg.norm_eps as f64,
        "rope_theta": cfg.rope_base as f64,
        "tie_word_embeddings": true,
        "max_position_embeddings": 131072,
        "linear_core": if cfg.is_gdn_mixer() {
            // the runtime's faithful GDN operator: `linear_core::gdn_step`
            // reads exactly these fields (loader.rs `"gated_delta_net"`)
            serde_json::json!({
                "kind": "gated_delta_net",
                "num_heads": cfg.gdn_heads,
                "value_head_dim": cfg.gdn_dv,
            })
        } else {
            serde_json::json!({
                "kind": if phase_delta_layers.is_empty() { "vmf_phase" } else { "vmf_phase_delta_v1" },
                "num_heads": cfg.heads,
                "nphase": cfg.nphase,
                "value_head_dim": cfg.dv,
                "phase_delta_layers": if phase_delta_layers.is_empty() { serde_json::Value::Null } else { serde_json::json!(phase_delta_layers) },
            })
        },
        "linear_num_key_heads": if cfg.is_gdn_mixer() { cfg.gdn_heads } else { cfg.heads },
        "linear_num_value_heads": if cfg.is_gdn_mixer() { cfg.gdn_heads } else { cfg.heads },
        "linear_key_head_dim": if cfg.is_gdn_mixer() { cfg.gdn_dk } else { cfg.nphase },
        "linear_value_head_dim": if cfg.is_gdn_mixer() { cfg.gdn_dv } else { cfg.dv },
    });
    if cfg.is_gdn_mixer() {
        j["linear_conv_kernel_dim"] = serde_json::json!(crate::model::GDN_CONV_K);
    }
    if cfg.experts > 0 {
        j["moe"] = serde_json::json!({
            "num_experts": cfg.experts,
            "top_k": 1,
            "moe_intermediate_size": cfg.inter,
            "norm_topk_prob": true,
            "shared_expert_intermediate_size": cfg.inter,
            "router_resonance": true,
        });
    }
    if cfg.head_clusters > 0 {
        j["head_clusters"] = serde_json::json!(cfg.head_clusters);
    }
    if cfg.anchor_bounded() {
        // The operator record (docs/EMBRYO_BOUNDED_ANCHOR.md §1): served
        // window + trained NoPE sinks, relative rope inside the window, no
        // far field. `train_windows` is informational for readers.
        let mut ac = serde_json::json!({
            "kind": "swa_sink_v1",
            "window": cfg.anchor_window,
            "sink": cfg.anchor_sink,
            "rope": "relative_in_window",
            "sink_scores": "nope",
        });
        if !cfg.anchor_train_windows.is_empty() {
            ac["train_windows"] = serde_json::json!(cfg.anchor_train_windows);
        }
        j["anchor_core"] = ac;
    }
    j
}

/// Validate every exporter option whose semantics are not represented by the
/// native Embryo CMF schema yet.  This deliberately runs before `Layout::new`
/// and before `CmfModel::write`: an unsupported experiment must fail without
/// creating a partial/truncated output file.
fn validate_export_cfg(cfg: &EmbryoCfg) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(
        !cfg.gdn_lane,
        "Embryo export does not support the experimental gdn_lane tail yet"
    );
    anyhow::ensure!(
        !cfg.gqa_lane,
        "Embryo export does not support the experimental gqa_lane tail yet"
    );
    anyhow::ensure!(
        !cfg.router_smooth_k4,
        "Embryo export does not support router_smooth_k4 semantics yet"
    );
    anyhow::ensure!(
        cfg.router_top2_margin.is_none(),
        "Embryo export does not support router_top2_margin semantics yet"
    );
    // Bounded anchor: `sink + window ≤ 160` (the format's kernel ceiling),
    // train windows inside the served window, a valid explicit schedule.
    cfg.check_anchor().map_err(|e| anyhow::anyhow!(e))?;
    // GDN mixer geometry (the runtime executes any nk == nv, dk, dv; the
    // trainer's own limits are the stricter ones).
    cfg.check_gdn().map_err(|e| anyhow::anyhow!(e))?;
    anyhow::ensure!(
        !cfg.gdn_beta_one,
        "gdn_beta_one is a trainer-only control arm (β ≡ 1 has no runtime operator); it cannot be exported"
    );
    if cfg.anchor_bounded() {
        anyhow::ensure!(
            (0..cfg.layers).any(|l| cfg.is_anchor(l)),
            "anchor_window > 0 but the genome has no anchor layer to bound"
        );
    }
    let selected = cfg
        .phase_delta_layers_for_export()
        .map_err(|e| anyhow::anyhow!(e))?;
    if !selected.is_empty() {
        anyhow::ensure!(
            !cfg.learn_decay,
            "Phase-Delta export requires the checkpoint's fixed decay grid; learned decay is not encoded"
        );
        anyhow::ensure!(cfg.heads > 0, "Phase-Delta requires at least one head");
        anyhow::ensure!(cfg.nphase > 0, "Phase-Delta requires nphase > 0");
        anyhow::ensure!(cfg.dv > 0, "Phase-Delta requires value width > 0");
        anyhow::ensure!(
            cfg.conv_k == 0 || cfg.conv_k >= 2,
            "Phase-Delta convolution kernel must be 0 or at least 2"
        );
    }
    Ok(selected)
}

/// Write `out` from a checkpoint (+ tokenizer.json bytes).
pub fn export(ck: &Checkpoint, tokenizer_json: &[u8], out: &Path) -> anyhow::Result<()> {
    export_with_dtype(ck, tokenizer_json, out, TensorDtype::F32)
}

/// Export with a file storage profile. F32 is the exact trainer baseline;
/// F16 stores learned matrices in half precision and keeps routing metadata,
/// norms and biases exact.
pub fn export_with_dtype(
    ck: &Checkpoint,
    tokenizer_json: &[u8],
    out: &Path,
    export_dtype: TensorDtype,
) -> anyhow::Result<()> {
    export_genome(ck, tokenizer_json, out, export_dtype, None)
}

/// [`export_with_dtype`] that, with `genome`, writes the frozen-genome
/// block (`header.genome` + `lineage: [birth]`, bit `GENOME`): the file's
/// `trunk_hash`, and the `master_trunk_hash` of the f32 export of the same
/// checkpoint (equal to it for f32; for f16 the f32 trunk is hashed in
/// memory, never written). Both come from the core `trunk_hash`.
pub fn export_genome(
    ck: &Checkpoint,
    tokenizer_json: &[u8],
    out: &Path,
    export_dtype: TensorDtype,
    genome: Option<&ExportGenome>,
) -> anyhow::Result<()> {
    refuse_genome_overwrite(out)?;
    let (header, specs) = build_export(ck, tokenizer_json, export_dtype, genome)?;
    CmfModel::write(out, &header, &specs, None, Some(tokenizer_json))?;
    Ok(())
}

/// A genome file is NEVER rewritten (docs §9: records are appended to a
/// copy through `CmfModel::append_skill`; a new trunk is a new file). The
/// core writers truncate their target, so every export checks first: when
/// `out` exists and is a CMF file whose envelope carries the `GENOME` bit OR
/// whose header JSON carries a `genome`, a `lineage` or appended `segments`,
/// refuse. The check is on content, not on the bit number alone: files
/// written by a pre-0.8.1 embryo-o1 build carry GENOME on bit 8 (today's
/// PRISM_AFFINE), and a file with appended records loses them on a rewrite.
/// A missing or unreadable file, or a file that is not CMF, passes.
pub fn refuse_genome_overwrite(out: &Path) -> anyhow::Result<()> {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(out) else {
        return Ok(());
    };
    let mut head = [0u8; 16];
    if f.read_exact(&mut head).is_err() || head[0..4] != cortiq_core::CMF_MAGIC {
        return Ok(());
    }
    drop(f);
    let required = u32::from_le_bytes([head[12], head[13], head[14], head[15]]);
    // An unreadable header falls back to the envelope bit alone.
    let by_content = matches!(
        cortiq_core::format::peek_header(out),
        Ok(Some((_, header))) if cortiq_core::format::header_json_is_genome(&header)
    );
    anyhow::ensure!(
        required & cortiq_core::format::features::GENOME == 0 && !by_content,
        "refusing to overwrite {}: it carries a frozen genome (GENOME bit, or a genome/lineage/segments \
         record in its header — pre-0.8.1 embryo-o1 files carry GENOME on bit 8) and a genome file is \
         never rewritten — write a new file, or append a record with append_skill / \
         update_header_append; a pre-0.8.1 file is migrated with `cortiq migrate-embryo-bits`, not \
         re-exported (a re-export drops appended records and the lineage)",
        out.display()
    );
    Ok(())
}

/// The trunk hash of the f32 export of `ck` — what a genome file's
/// `master_trunk_hash` is and what v2 skill records bind to.
pub fn master_trunk_hash(ck: &Checkpoint, tokenizer_json: &[u8]) -> anyhow::Result<u64> {
    let (header, specs) = build_export(ck, tokenizer_json, TensorDtype::F32, None)?;
    Ok(trunk_hash(
        &header,
        &spec_entries(&specs),
        Some(tokenizer_json),
    ))
}

/// The export in memory: header + tensors exactly as `export_genome`
/// writes them (the vocab section is `tokenizer_json`).
pub fn build_export(
    ck: &Checkpoint,
    tokenizer_json: &[u8],
    export_dtype: TensorDtype,
    genome: Option<&ExportGenome>,
) -> anyhow::Result<(CmfHeader, Vec<TensorSpec>)> {
    build_export_sources(ck, tokenizer_json, export_dtype, genome).map(|(h, t, _)| (h, t))
}

/// Where a half-precision tensor of an export was read from: its offset in
/// `ck.params` (None when the tensor is not a slice of the parameter arena).
#[derive(Clone, Debug)]
pub struct F16Source {
    pub name: String,
    pub arena_offset: Option<usize>,
    pub len: usize,
}

/// Offset of `s` inside `arena` in elements (pointer arithmetic only).
fn arena_offset(arena: &[f32], s: &[f32]) -> Option<usize> {
    let (a, b) = (arena.as_ptr() as usize, s.as_ptr() as usize);
    let sz = std::mem::size_of::<f32>();
    let (sb, ab) = (std::mem::size_of_val(s), std::mem::size_of_val(arena));
    (b >= a && (b - a) % sz == 0 && b + sb <= a + ab).then(|| (b - a) / sz)
}

/// The parameters the SERVED trunk of a `dtype` export computes with:
/// `ck.params` with every tensor the export stores in half precision
/// rounded f32 → f16 → f32 by exactly the export's own rule (the source
/// ranges are recorded while the export is built, so the rule cannot
/// drift). F32 returns the parameters unchanged. A bake over an f16 genome
/// probes φ and polishes its FFNs on these, i.e. on the trunk the runtime
/// executes (spec §4: φ identical in trainer and runtime).
pub fn served_params(
    ck: &Checkpoint,
    tokenizer_json: &[u8],
    dtype: TensorDtype,
) -> anyhow::Result<Vec<f32>> {
    let mut p = ck.params.clone();
    if dtype == TensorDtype::F32 {
        return Ok(p);
    }
    let (_, _, sources) = build_export_sources(ck, tokenizer_json, dtype, None)?;
    for s in sources {
        let off = s.arena_offset.ok_or_else(|| {
            anyhow::anyhow!(
                "{}: stored as f16 but not read from the parameter arena (cannot round it)",
                s.name
            )
        })?;
        for x in &mut p[off..off + s.len] {
            *x = cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(*x));
        }
    }
    Ok(p)
}

/// [`build_export`] plus the arena source of every tensor it stores as f16.
pub fn build_export_sources(
    ck: &Checkpoint,
    tokenizer_json: &[u8],
    export_dtype: TensorDtype,
    genome: Option<&ExportGenome>,
) -> anyhow::Result<(CmfHeader, Vec<TensorSpec>, Vec<F16Source>)> {
    anyhow::ensure!(
        matches!(export_dtype, TensorDtype::F32 | TensorDtype::F16),
        "Embryo export dtype must be f32 or f16"
    );
    if let Some(g) = genome {
        g.check()?;
    }
    // Keep this before Layout::new and all output work.  In particular, the
    // unsupported append-only experiment tails below must never leave a
    // partially written CMF behind.
    let _phase_delta_layers = validate_export_cfg(&ck.cfg)?;
    let cfg = &ck.cfg;
    let lay = Layout::new(cfg);
    let p = &ck.params;
    let h = cfg.hidden;
    let sl = |off: usize, n: usize| &p[off..off + n];
    let mut t: Vec<TensorSpec> = Vec::new();
    let spec_f32 = |name: &str, shape: &[usize], data: &[f32]| spec(name, shape, data, TensorDtype::F32);
    let f16_sources = std::cell::RefCell::new(Vec::<F16Source>::new());
    let spec = |name: &str, shape: &[usize], data: &[f32]| {
        let s = spec(name, shape, data, export_dtype);
        if s.dtype == TensorDtype::F16 {
            f16_sources.borrow_mut().push(F16Source {
                name: name.to_string(),
                arena_offset: arena_offset(p, data),
                len: data.len(),
            });
        }
        s
    };
    t.push(spec(
        "model.embed_tokens.weight",
        &[cfg.vocab, h],
        sl(lay.embed, cfg.vocab * h),
    ));
    t.push(spec("model.norm.weight", &[h], sl(lay.final_norm, h)));
    if cfg.head_clusters > 0 {
        t.push(spec(
            "lm_head.clusters.weight",
            &[cfg.head_clusters, h],
            sl(lay.head_clusters, cfg.head_clusters * h),
        ));
    }
    let desc = |name: &str| {
        ck.extras
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, x)| x.as_slice())
    };
    let (mu, u, bias) = (desc("desc.mu"), desc("desc.u"), desc("desc.bias"));
    let decay = crate::ops::hk_decay_grid(cfg.heads, cfg.nphase, cfg.horizon_min, cfg.horizon_max);
    // A_log: decay = exp(−exp(A_log)) → A_log = ln(−ln γ)
    let a_log: Vec<f32> = decay
        .iter()
        .map(|g| (-(*g as f64).ln()).ln() as f32)
        .collect();
    for (l, lo) in lay.layers.iter().enumerate() {
        let pf = format!("model.layers.{l}.");
        let ffn = match lo {
            LayerOffs::Mixer {
                ln1,
                wq,
                wk,
                wv,
                wkap,
                wo,
                alog: _,
                conv,
                ln2,
                ffn,
            } => {
                t.push(spec(
                    &format!("{pf}input_layernorm.weight"),
                    &[h],
                    sl(*ln1, h),
                ));
                t.push(spec(
                    &format!("{pf}post_attention_layernorm.weight"),
                    &[h],
                    sl(*ln2, h),
                ));
                let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
                t.push(spec(
                    &format!("{pf}vmf_attn.thq.weight"),
                    &[nh * nph, h],
                    sl(*wq, nh * nph * h),
                ));
                t.push(spec(
                    &format!("{pf}vmf_attn.thk.weight"),
                    &[nh * nph, h],
                    sl(*wk, nh * nph * h),
                ));
                t.push(spec(
                    &format!("{pf}vmf_attn.v_proj.weight"),
                    &[nh * dv, h],
                    sl(*wv, nh * dv * h),
                ));
                t.push(spec(
                    &format!("{pf}vmf_attn.out_proj.weight"),
                    &[h, nh * dv],
                    sl(*wo, h * nh * dv),
                ));
                t.push(spec(
                    &format!("{pf}vmf_attn.A_log"),
                    &[nh * 2 * nph],
                    &a_log,
                ));
                // κ gate: the trainer's padded [kappa_ld, H] → real rows; fixed bias
                t.push(spec(
                    &format!("{pf}vmf_attn.k_gate.weight"),
                    &[nh, h],
                    sl(*wkap, nh * h),
                ));
                t.push(spec(
                    &format!("{pf}vmf_attn.k_gate.bias"),
                    &[nh],
                    &vec![cfg.kappa_bias; nh],
                ));
                if *conv != usize::MAX {
                    // the Qwen/LFM conv1d convention: [channels, 1, k]
                    t.push(spec(
                        &format!("{pf}vmf_attn.conv1d.weight"),
                        &[h, 1, cfg.conv_k],
                        sl(*conv, h * cfg.conv_k),
                    ));
                }
                ffn
            }
            LayerOffs::Gdn {
                ln1,
                in_qkv,
                in_z,
                in_a,
                in_b,
                wo,
                conv,
                alog,
                dt_bias,
                norm,
                ln2,
                ffn,
            } => {
                let (nv, dv) = (cfg.gdn_heads, cfg.gdn_dv);
                let c_dim = cfg.gdn_c_dim();
                t.push(spec(
                    &format!("{pf}input_layernorm.weight"),
                    &[h],
                    sl(*ln1, h),
                ));
                t.push(spec(
                    &format!("{pf}post_attention_layernorm.weight"),
                    &[h],
                    sl(*ln2, h),
                ));
                // `linear_attn.*` — the loader's gated_delta_net names, 1:1
                // (loader.rs `load_linear_attn`); the padded control rows
                // are sliced to the live `nv` rows, the padded scalars to nv.
                t.push(spec(
                    &format!("{pf}linear_attn.in_proj_qkv.weight"),
                    &[c_dim, h],
                    sl(*in_qkv, c_dim * h),
                ));
                t.push(spec(
                    &format!("{pf}linear_attn.in_proj_z.weight"),
                    &[nv * dv, h],
                    sl(*in_z, nv * dv * h),
                ));
                t.push(spec(
                    &format!("{pf}linear_attn.in_proj_a.weight"),
                    &[nv, h],
                    sl(*in_a, nv * h),
                ));
                t.push(spec(
                    &format!("{pf}linear_attn.in_proj_b.weight"),
                    &[nv, h],
                    sl(*in_b, nv * h),
                ));
                t.push(spec(
                    &format!("{pf}linear_attn.out_proj.weight"),
                    &[h, nv * dv],
                    sl(*wo, h * nv * dv),
                ));
                // conv taps, decay and norm are operator state, never
                // quantized: f32 in every storage profile
                t.push(spec_f32(
                    &format!("{pf}linear_attn.conv1d.weight"),
                    &[c_dim, 1, crate::model::GDN_CONV_K],
                    sl(*conv, c_dim * crate::model::GDN_CONV_K),
                ));
                t.push(spec_f32(&format!("{pf}linear_attn.A_log"), &[nv], sl(*alog, nv)));
                t.push(spec_f32(
                    &format!("{pf}linear_attn.dt_bias"),
                    &[nv],
                    sl(*dt_bias, nv),
                ));
                t.push(spec_f32(
                    &format!("{pf}linear_attn.norm.weight"),
                    &[dv],
                    sl(*norm, dv),
                ));
                ffn
            }
            LayerOffs::Anchor {
                ln1,
                wq,
                wk,
                wv,
                wo,
                sink_k,
                sink_v,
                ln2,
                ffn,
            } => {
                t.push(spec(
                    &format!("{pf}input_layernorm.weight"),
                    &[h],
                    sl(*ln1, h),
                ));
                t.push(spec(
                    &format!("{pf}post_attention_layernorm.weight"),
                    &[h],
                    sl(*ln2, h),
                ));
                let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
                t.push(spec(
                    &format!("{pf}self_attn.q_proj.weight"),
                    &[qh * hd, h],
                    sl(*wq, qh * hd * h),
                ));
                t.push(spec(
                    &format!("{pf}self_attn.k_proj.weight"),
                    &[kvh * hd, h],
                    sl(*wk, kvh * hd * h),
                ));
                t.push(spec(
                    &format!("{pf}self_attn.v_proj.weight"),
                    &[kvh * hd, h],
                    sl(*wv, kvh * hd * h),
                ));
                t.push(spec(
                    &format!("{pf}self_attn.o_proj.weight"),
                    &[h, qh * hd],
                    sl(*wo, h * qh * hd),
                ));
                if cfg.anchor_bounded() && cfg.anchor_sink > 0 {
                    // Sink vectors are weights of the operator, never
                    // quantized: f32 in every storage profile.
                    let s_n = kvh * cfg.anchor_sink * hd;
                    t.push(spec_f32(
                        &format!("{pf}self_attn.sink_k.weight"),
                        &[kvh, cfg.anchor_sink, hd],
                        sl(*sink_k, s_n),
                    ));
                    t.push(spec_f32(
                        &format!("{pf}self_attn.sink_v.weight"),
                        &[kvh, cfg.anchor_sink, hd],
                        sl(*sink_v, s_n),
                    ));
                }
                ffn
            }
        };
        let i = cfg.inter;
        if cfg.experts == 0 {
            t.push(spec(
                &format!("{pf}mlp.gate_proj.weight"),
                &[i, h],
                sl(ffn.wg, i * h),
            ));
            t.push(spec(
                &format!("{pf}mlp.up_proj.weight"),
                &[i, h],
                sl(ffn.wu, i * h),
            ));
            t.push(spec(
                &format!("{pf}mlp.down_proj.weight"),
                &[h, i],
                sl(ffn.wd, h * i),
            ));
        } else {
            t.push(spec(
                &format!("{pf}mlp.shared_expert.gate_proj.weight"),
                &[i, h],
                sl(ffn.wg, i * h),
            ));
            t.push(spec(
                &format!("{pf}mlp.shared_expert.up_proj.weight"),
                &[i, h],
                sl(ffn.wu, i * h),
            ));
            t.push(spec(
                &format!("{pf}mlp.shared_expert.down_proj.weight"),
                &[h, i],
                sl(ffn.wd, h * i),
            ));
            let ne = cfg.experts;
            for e in 0..ne {
                let base = ffn.experts + e * 3 * h * i;
                t.push(spec(
                    &format!("{pf}mlp.experts.{e}.gate_proj.weight"),
                    &[i, h],
                    sl(base, i * h),
                ));
                t.push(spec(
                    &format!("{pf}mlp.experts.{e}.up_proj.weight"),
                    &[i, h],
                    sl(base + i * h, i * h),
                ));
                t.push(spec(
                    &format!("{pf}mlp.experts.{e}.down_proj.weight"),
                    &[h, i],
                    sl(base + 2 * i * h, h * i),
                ));
            }
            // per-expert descriptor records (append-only growth: a new
            // expert = new tensors, nothing rewritten); no gate placeholder —
            // the loader keys the resonance MoE on experts.0 + the arch flag
            let k = crate::model::MOE_K;
            let mu_l = mu
                .map(|m| &m[l * ne * h..(l + 1) * ne * h])
                .ok_or_else(|| anyhow::anyhow!("checkpoint has no expert descriptors (desc.mu)"))?;
            for e in 0..ne {
                t.push(spec(
                    &format!("{pf}mlp.experts.{e}.desc.mu"),
                    &[h],
                    &mu_l[e * h..(e + 1) * h],
                ));
                if let Some(u) = u {
                    let base = (l * ne + e) * k * h;
                    t.push(spec(
                        &format!("{pf}mlp.experts.{e}.desc.u"),
                        &[k, h],
                        &u[base..base + k * h],
                    ));
                }
                let b = bias.map(|b| b[l * ne + e]).unwrap_or(0.0);
                t.push(spec(&format!("{pf}mlp.experts.{e}.desc.bias"), &[1], &[b]));
            }
        }
    }
    let arch: cortiq_core::types::ModelArch = serde_json::from_value(arch_json(cfg))?;
    let tok: serde_json::Value = serde_json::from_slice(tokenizer_json)?;
    let eos = tok["added_tokens"]
        .as_array()
        .and_then(|a| a.iter().find(|t| t["content"] == "<|endoftext|>"))
        .and_then(|t| t["id"].as_u64())
        .map(|i| i as u32);
    let mut provenance = serde_json::json!({
        "producer": "cortiq-embryo",
        "step": ck.step,
        "export_dtype": export_dtype.name(),
        "trainer_cfg": cfg,
    });
    if let Some(g) = genome {
        provenance["genome"] = serde_json::json!(g.id);
    }
    let mut header = CmfHeader {
        format: "cmf".into(),
        version: 2,
        arch,
        quant_type: match export_dtype {
            TensorDtype::F16 => cortiq_core::types::QuantType::F16,
            _ => cortiq_core::types::QuantType::F32,
        },
        provenance: Some(provenance),
        tokenizer_config: Some(TokenizerBundle {
            chat_template: None,
            eos_token_ids: eos.into_iter().collect(),
            bos_token_id: None,
            pad_token_id: None,
        }),
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
    if let Some(g) = genome {
        let encoding = export_dtype.name();
        let trunk = trunk_hash(&header, &spec_entries(&t), Some(tokenizer_json));
        let master = if export_dtype == TensorDtype::F32 {
            trunk
        } else {
            master_trunk_hash(ck, tokenizer_json)?
        };
        let mut info = GenomeInfo::birth(g.id.clone(), g.status.clone(), encoding);
        info.trunk_hash = hex64(trunk);
        info.master_trunk_hash = hex64(master);
        header.genome = Some(info);
        header.lineage = vec![LineageEvent::now(
            0,
            "birth",
            serde_json::json!({
                "producer": "cortiq-embryo export",
                "step": ck.step,
                "encoding": encoding,
                "status": g.status,
            }),
        )];
    }
    Ok((header, t, f16_sources.into_inner()))
}
