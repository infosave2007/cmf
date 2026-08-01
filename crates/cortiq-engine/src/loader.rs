//! Weight loader: CMF tensor directory → Pipeline.
//!
//! Storage rule: models WITH task masks are dequantized to f32 (masked
//! execution needs f32 row access; skill files are small by design).
//! Models without masks keep quantized matrices zero-copy from the mmap
//! (`QTensor::Mapped`) — this is what lets a 15B file run in a few GB
//! of RSS instead of 60 GB of f32.
//!
//! Layer kinds come from `arch.layer_types`: FullAttention loads
//! `self_attn.*` (with auto-detected Qwen3.5 extras: per-head qk-norm by
//! tensor presence, output gate by q_proj row count); LinearAttention
//! loads the canonical core `vmf_attn.*` (folded at convert time).

use crate::kv_cache::LayerKvCache;
use crate::linear_core::{
    GdnCfg, GdnWeights, ShortConvCfg, ShortConvWeights, VmfPhaseCfg, VmfPhaseWeights,
};
use crate::pipeline::{
    AttnKind, DenseFfn, FfnKind, LayerWeights, MoeFfn, MtpModule, Pipeline, PipelineWeights,
};
use crate::qtensor::QTensor;
use crate::sampler::SamplerConfig;
use crate::tokenizer::Tokenizer;
use cortiq_core::quant::dequant_tensor;
use cortiq_core::{CmfError, CmfModel, LayerType, ModelArch};
use std::sync::Arc;

/// Tensor source selector (spec §9): backbone, one skill's overlay, or
/// a soft superposition of top-m skills (claim 14 working tensors).
pub enum Overlay<'a> {
    None,
    One(&'a str),
    /// (skill_id, weight); weights sum to 1 (softmax(−E/T) upstream).
    Blend(&'a [(String, f32)]),
}

impl Overlay<'_> {
    fn blend_touches(&self, model: &CmfModel, name: &str) -> bool {
        match self {
            Overlay::Blend(list) => list
                .iter()
                .any(|(sid, _)| model.tensor(&format!("skill.{sid}.{name}")).is_some()),
            _ => false,
        }
    }
}

fn dequant_by_name(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("tensor '{name}' not found in CMF directory"))?;
    let mut out = vec![0.0f32; entry.n_elems()];
    dequant_tensor(entry, model.entry_bytes(entry), &mut out)?;
    Ok(out)
}

/// Weighted working tensor (claim 14): Σ wᵢ·Tᵢ, where Tᵢ is the
/// skill's replacement when present, else the backbone tensor.
fn blend_f32(model: &CmfModel, name: &str, list: &[(String, f32)]) -> Result<Vec<f32>, String> {
    let mut acc: Option<Vec<f32>> = None;
    for (sid, w) in list {
        let sname = format!("skill.{sid}.{name}");
        let src = if model.tensor(&sname).is_some() {
            &sname
        } else {
            name
        };
        let t = dequant_by_name(model, src)?;
        match &mut acc {
            None => {
                let mut t = t;
                for v in t.iter_mut() {
                    *v *= w;
                }
                acc = Some(t);
            }
            Some(a) => {
                for (av, tv) in a.iter_mut().zip(&t) {
                    *av += w * tv;
                }
            }
        }
    }
    acc.ok_or_else(|| "empty blend".into())
}

/// Dequantize a tensor fully into f32 (norms, masked models).
pub(crate) fn load_f32(model: &CmfModel, name: &str, ov: &Overlay) -> Result<Vec<f32>, String> {
    if ov.blend_touches(model, name) {
        if let Overlay::Blend(list) = ov {
            return blend_f32(model, name, list);
        }
    }
    let skill = match ov {
        Overlay::One(s) => Some(*s),
        _ => None,
    };
    let entry = model
        .resolve_tensor(name, skill)
        .ok_or_else(|| format!("tensor '{name}' not found in CMF directory"))?;
    let bytes = model.entry_bytes(entry);
    let mut out = vec![0.0f32; entry.n_elems()];
    dequant_tensor(entry, bytes, &mut out)?;
    Ok(out)
}

/// Build one layer's FFN (dense or MoE) under a given overlay. Shared
/// by the static loader AND dynamic per-token skill switching
/// (`Pipeline::set_active_skill`): switching skills = rebuilding the
/// FFN of the touched layers, cheap because Mapped tensors are just
/// re-resolved mmap pointers (no dequant, no copy).
pub(crate) fn build_layer_ffn(
    model: &Arc<CmfModel>,
    arch: &ModelArch,
    li: usize,
    force_f32: bool,
    ov: &Overlay,
) -> Result<FfnKind, CmfError> {
    build_ffn_at(model, arch, &format!("model.layers.{li}."), force_f32, ov)
}

/// The FFN under an arbitrary prefix. Split out of `build_layer_ffn` so the
/// MTP block can reuse it: Qwen3.6's MTP layer carries a full MoE mlp
/// (router + 256 experts + shared expert), not the dense one the head was
/// first written against.
pub(crate) fn build_ffn_at(
    model: &Arc<CmfModel>,
    arch: &ModelArch,
    prefix: &str,
    force_f32: bool,
    ov: &Overlay,
) -> Result<FfnKind, CmfError> {
    let prefix = prefix.to_string();
    let load_dense = |p: &str| -> Result<DenseFfn, CmfError> {
        let gate_proj = load_matrix(model, &format!("{p}gate_proj.weight"), force_f32, ov)?;
        let up_proj = load_matrix(model, &format!("{p}up_proj.weight"), force_f32, ov)?;
        let down_proj = load_matrix(model, &format!("{p}down_proj.weight"), force_f32, ov)?;
        // FFN triple invariant (holds for dense and each MoE expert;
        // enforced loudly so a malformed defrag/repack — spec §11 — fails
        // at load instead of silently mis-computing). inter' is per-layer.
        let inter = gate_proj.rows();
        if up_proj.rows() != inter || down_proj.cols() != inter {
            return Err(CmfError::Parse(format!(
                "{p}: FFN dims disagree (gate.rows={inter}, up.rows={}, \
                 down.cols={}); all three must equal inter'",
                up_proj.rows(),
                down_proj.cols()
            )));
        }
        if down_proj.rows() != arch.hidden_size {
            return Err(CmfError::Parse(format!(
                "{p}: down_proj.rows={} != hidden_size={}",
                down_proj.rows(),
                arch.hidden_size
            )));
        }
        Ok(DenseFfn {
            gate_proj,
            up_proj,
            down_proj,
            act: crate::pipeline::Act::from_arch_full(arch),
        })
    };
    let router_name = format!("{prefix}mlp.gate.weight");
    if model.tensor(&router_name).is_none() {
        return Ok(FfnKind::Dense(load_dense(&format!("{prefix}mlp."))?));
    }
    let cfg = arch.moe.as_ref().ok_or_else(|| {
        CmfError::Parse(format!(
            "{router_name} present but header has no arch.moe block"
        ))
    })?;
    // Experts enumerate by TENSOR PRESENCE up to the header count — a
    // moe-defrag'd specialist keeps a per-layer contiguous prefix of
    // renumbered experts (fewer than arch.moe.num_experts), with the
    // router rows sliced to match.
    let mut experts = Vec::new();
    for e in 0..cfg.num_experts {
        if model
            .tensor(&format!("{prefix}mlp.experts.{e}.gate_proj.weight"))
            .is_none()
        {
            break;
        }
        experts.push(load_dense(&format!("{prefix}mlp.experts.{e}."))?);
    }
    if experts.is_empty() {
        return Err(CmfError::Parse(format!(
            "{prefix}: router present but no expert tensors"
        )));
    }
    let shared = if model
        .tensor(&format!("{prefix}mlp.shared_expert.gate_proj.weight"))
        .is_some()
    {
        let gate_name = format!("{prefix}mlp.shared_expert_gate.weight");
        Some((
            load_dense(&format!("{prefix}mlp.shared_expert."))?,
            if model.tensor(&gate_name).is_some() {
                Some(load_matrix(model, &gate_name, force_f32, ov)?)
            } else {
                None
            },
        ))
    } else {
        None
    };
    // LFM2-MoE selection bias (`mlp.expert_bias`): present iff the model
    // routes with a bias; loaded by tensor presence.
    let bias_name = format!("{prefix}mlp.expert_bias");
    let expert_bias = if model.tensor(&bias_name).is_some() {
        Some(load_f32(model, &bias_name, ov).map_err(CmfError::Parse)?)
    } else {
        None
    };
    // CMF_MOE_TOPK=N (opt-in): route to fewer experts than the header
    // asks. MoE decode is memory-bound — every selected expert streams
    // its three matrices per token — so halving k halves that traffic;
    // the renormalized top-k keeps the mixture a proper average.
    // Quality is the experiment — measure ppl before trusting.
    let top_k = std::env::var("CMF_MOE_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| k >= 1 && k <= cfg.top_k)
        .inspect(|k| tracing::info!("MoE top_k override: {} (header {})", k, cfg.top_k))
        .unwrap_or(cfg.top_k);
    // CMF_MOE_TAU=0.x (opt-in): adaptive routing — see MoeFfn::route_tau.
    let route_tau = std::env::var("CMF_MOE_TAU")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|&t| t > 0.0 && t < 1.0)
        .inspect(|t| tracing::info!("MoE adaptive routing: tau {t}"));
    let mask = moe_task_mask(&prefix, experts.len());
    let router = load_matrix(model, &router_name, force_f32, ov)?;
    if router.rows() != experts.len() {
        return Err(CmfError::Parse(format!(
            "{router_name}: {} rows != {} experts",
            router.rows(),
            experts.len()
        )));
    }
    let top_k = top_k.min(experts.len());
    // Gemma-4: per-expert weight scale after the top-k renorm; its
    // presence also marks the scale-less-rms router input (the folded
    // router gain — see the converter).
    let pes_name = format!("{prefix}mlp.per_expert_scale");
    let per_expert_scale = if model.tensor(&pes_name).is_some() {
        Some(load_f32(model, &pes_name, ov).map_err(CmfError::Parse)?)
    } else {
        None
    };
    let router_input_norm = per_expert_scale.is_some();
    let moe = MoeFfn {
        router,
        experts,
        top_k,
        route_tau,
        norm_topk_prob: cfg.norm_topk_prob,
        router_sigmoid: cfg.router_sigmoid,
        expert_bias,
        routed_scaling: cfg.routed_scaling_factor.unwrap_or(1.0),
        shared,
        stats: std::cell::RefCell::new(Vec::new()),
        act_sq: std::cell::RefCell::new(Vec::new()),
        act_rows: std::cell::RefCell::new(Vec::new()),
        mask,
        per_expert_scale,
        router_input_norm,
    };
    // Gemma-4 dual-branch layer: a dense MLP coexists with the routed
    // experts, each branch inside its own norm sandwich.
    if model
        .tensor(&format!("{prefix}mlp.gate_proj.weight"))
        .is_some()
    {
        let norm = |suffix: &str| -> Result<Vec<f32>, CmfError> {
            load_f32(model, &format!("{prefix}{suffix}.weight"), ov).map_err(CmfError::Parse)
        };
        return Ok(FfnKind::DenseMoe(Box::new(
            crate::pipeline::DenseMoeFfn {
                dense: load_dense(&format!("{prefix}mlp."))?,
                moe,
                post_norm_1: norm("post_feedforward_layernorm_1")?,
                pre_norm_2: norm("pre_feedforward_layernorm_2")?,
                post_norm_2: norm("post_feedforward_layernorm_2")?,
            },
        )));
    }
    Ok(FfnKind::Moe(moe))
}

/// Task mask over routed experts (opt-in, experimental): DTG-MA applied
/// to MoE. `CMF_MOE_MASK=<stats.json>` points at a claim-12 B-field dump
/// (`CMF_MOE_STATS` output — per-layer expert-selection counts from a
/// task-representative run); `CMF_MOE_MASK_COVER` (default 0.9) keeps,
/// per layer, the smallest top set of experts reaching that fraction of
/// the recorded routing mass. Selection then happens over the allowed
/// set only (softmax renormalizes). Gate any real use on a ppl A/B.
pub(crate) fn moe_task_mask(prefix: &str, ne: usize) -> Option<Vec<bool>> {
    use std::sync::OnceLock;
    static CFG: OnceLock<Option<(std::collections::HashMap<usize, Vec<u64>>, f64)>> =
        OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let path = std::env::var("CMF_MOE_MASK").ok()?;
        let cover = std::env::var("CMF_MOE_MASK_COVER")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&c| c > 0.0 && c <= 1.0)
            .unwrap_or(0.9);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| tracing::warn!("CMF_MOE_MASK: cannot read {path}: {e}"))
            .ok()?;
        let map: std::collections::HashMap<String, Vec<u64>> =
            serde_json::from_str(&text)
                .map_err(|e| tracing::warn!("CMF_MOE_MASK: bad JSON in {path}: {e}"))
                .ok()?;
        tracing::info!("MoE task mask: {path}, cover {cover}");
        Some((
            map.into_iter()
                .filter_map(|(k, v)| Some((k.parse::<usize>().ok()?, v)))
                .collect(),
            cover,
        ))
    });
    let (stats, cover) = cfg.as_ref()?;
    // The layer index rides in the tensor prefix ("model.layers.N.").
    let li: usize = prefix
        .split("layers.")
        .nth(1)?
        .split('.')
        .next()?
        .parse()
        .ok()?;
    let counts = stats.get(&li)?;
    if counts.len() != ne {
        tracing::warn!("CMF_MOE_MASK: layer {li} has {} counts, model has {ne} experts — skipped", counts.len());
        return None;
    }
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..ne).collect();
    order.sort_unstable_by_key(|&e| std::cmp::Reverse(counts[e]));
    let mut mask = vec![false; ne];
    let mut acc = 0u64;
    let mut kept = 0usize;
    for &e in &order {
        mask[e] = true;
        acc += counts[e];
        kept += 1;
        if (acc as f64) >= cover * (total as f64) {
            break;
        }
    }
    tracing::info!("MoE task mask L{li}: {kept}/{ne} experts for {:.0}% mass", cover * 100.0);
    Some(mask)
}

fn load_matrix(
    model: &Arc<CmfModel>,
    name: &str,
    force_f32: bool,
    ov: &Overlay,
) -> Result<QTensor, CmfError> {
    // Claim 14: a blended working tensor is materialized in f32 and
    // held resident (the overlay-cache slot); single skills stay
    // zero-copy pointers into the mmap.
    if ov.blend_touches(model, name) {
        if let Overlay::Blend(list) = ov {
            let entry = model
                .tensor(name)
                .ok_or_else(|| CmfError::MissingTensor(name.to_string()))?;
            let data =
                blend_f32(model, name, list).map_err(|e| CmfError::Parse(format!("blend: {e}")))?;
            return Ok(QTensor::from_f32(data, entry.shape[0], entry.shape[1]));
        }
    }
    let skill = match ov {
        Overlay::One(s) => Some(*s),
        _ => None,
    };
    // Tensor-source indirection (spec §9): the skill's replacement is
    // read in place of the backbone tensor — either/or, never a sum.
    let name: &str = &match skill {
        Some(sid) if model.tensor(&format!("skill.{sid}.{name}")).is_some() => {
            format!("skill.{sid}.{name}")
        }
        _ => name.to_string(),
    };
    let err = |e: String| CmfError::Parse(format!("weight loading: {e}"));
    if force_f32 {
        let entry = model
            .tensor(name)
            .ok_or_else(|| CmfError::MissingTensor(name.to_string()))?;
        if entry.shape.len() != 2 {
            return Err(err(format!("'{name}' is not 2-D")));
        }
        let data = load_f32(model, name, &Overlay::None).map_err(err)?;
        Ok(QTensor::from_f32(data, entry.shape[0], entry.shape[1]))
    } else {
        QTensor::from_model(model, name).map_err(err)
    }
}

impl Pipeline {
    /// Build a runnable pipeline from an opened CMF model.
    pub fn from_model(
        model: &Arc<CmfModel>,
        sampler_config: SamplerConfig,
    ) -> Result<Self, CmfError> {
        Self::from_model_with_skill(model, sampler_config, None)
    }

    /// Same, with a skill overlaid (spec §9): every layer tensor is
    /// resolved through tensor-source indirection — the skill's
    /// full-shape replacement is read in place of the backbone tensor.
    /// No per-skill model is ever assembled: Mapped tensors are
    /// pointers into the one shared mmap.
    pub fn from_model_with_skill(
        model: &Arc<CmfModel>,
        sampler_config: SamplerConfig,
        skill: Option<&str>,
    ) -> Result<Self, CmfError> {
        match skill {
            Some(s) => Self::from_model_with_overlay(model, sampler_config, &Overlay::One(s)),
            None => Self::from_model_with_overlay(model, sampler_config, &Overlay::None),
        }
    }

    /// Soft superposition (claim 14): working tensors accumulated from
    /// the given (skill, weight) list — softmax(−E/T) upstream.
    pub fn from_model_with_blend(
        model: &Arc<CmfModel>,
        sampler_config: SamplerConfig,
        blend: &[(String, f32)],
    ) -> Result<Self, CmfError> {
        Self::from_model_with_overlay(model, sampler_config, &Overlay::Blend(blend))
    }

    fn from_model_with_overlay(
        model: &Arc<CmfModel>,
        sampler_config: SamplerConfig,
        ov: &Overlay,
    ) -> Result<Self, CmfError> {
        let skill = match ov {
            Overlay::One(s) => Some(*s),
            _ => None,
        };
        if let Some(sid) = skill {
            let known = model.header.skills.iter().any(|s| s.id == sid)
                || model.skill_tensors(sid).next().is_some();
            if !known {
                return Err(CmfError::Parse(format!(
                    "skill '{sid}' not in this container (header.skills: {:?})",
                    model
                        .header
                        .skills
                        .iter()
                        .map(|s| &s.id)
                        .collect::<Vec<_>>()
                )));
            }
            tracing::info!(
                "skill '{sid}': {} replacement tensors overlaid",
                model.skill_tensors(sid).count()
            );
        }
        let arch = model.arch().clone();
        let err = |e: String| CmfError::Parse(format!("weight loading: {e}"));
        if let Some(heads) = &arch.attention_heads_per_layer {
            if heads.len() != arch.num_layers {
                return Err(CmfError::Parse(format!(
                    "arch.attention_heads_per_layer has {} entries, expected {}",
                    heads.len(),
                    arch.num_layers
                )));
            }
            if let Some((li, &nh)) = heads
                .iter()
                .enumerate()
                .find(|(_, nh)| **nh == 0 || **nh % arch.num_kv_heads != 0)
            {
                return Err(CmfError::Parse(format!(
                    "layer {li} has {nh} Q heads, which must be nonzero and divisible by {} KV heads",
                    arch.num_kv_heads
                )));
            }
        }
        if arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::SlidingAttention))
            && arch.sliding_window.is_none()
        {
            return Err(CmfError::Parse(
                "model has SlidingAttention layers but no arch.sliding_window".into(),
            ));
        }

        // Masks × quantized mmap: only ATTENTION keeps f32 (the head-mask
        // path needs f32 slices). FFN masks now run sparse directly on the
        // quant bytes (sparse_ffn_quant), and embed/lm_head are never
        // masked — so a masked model runs at quantized RSS, not the old
        // whole-model-f32 blowup.
        let masks_present = !model.masks.masks.is_empty();
        let force_f32 = masks_present; // attention only (head masks)

        // ── Tokenizer: embedded → sidecar → byte-level fallback ──
        let mut tokenizer = if let Some(vocab_bytes) = &model.vocab {
            Tokenizer::from_bytes(vocab_bytes)
                .map_err(|e| CmfError::Parse(format!("embedded tokenizer: {e}")))?
        } else {
            let sidecar = model.path.with_file_name("tokenizer.json");
            if sidecar.exists() {
                Tokenizer::from_file(&sidecar)
                    .map_err(|e| CmfError::Parse(format!("sidecar tokenizer: {e}")))?
            } else {
                tracing::warn!("no tokenizer in file or sidecar — using byte-level fallback");
                Tokenizer::byte_level()
            }
        };
        // Chat/eos bundle (spec §6.1): the FILE defines chat behavior.
        if let Some(tc) = &model.header.tokenizer_config {
            tokenizer.chat_template = tc.chat_template.clone();
            tokenizer.extra_eos.extend(tc.eos_token_ids.iter().copied());
            if tokenizer.bos_token_id.is_none() {
                tokenizer.bos_token_id = tc.bos_token_id;
            }
            tracing::info!(
                "chat bundle: template {} chars, {} stop ids",
                tc.chat_template.as_deref().map(str::len).unwrap_or(0),
                tc.eos_token_ids.len()
            );
        }
        // Gemma's contract requires <bos> at sequence start, but its
        // tokenizer.json post-processor does not add it (the chat
        // template does). Raw prompts need it too — word salad without.
        if arch.arch_name.to_lowercase().contains("gemma") && tokenizer.bos_token_id.is_some() {
            tokenizer.add_bos = true;
        }

        // ── Top-level weights (never masked → always quantized) ──
        let embed_tokens = load_matrix(model, "model.embed_tokens.weight", false, ov)?;
        let final_norm = load_f32(model, "model.norm.weight", ov).map_err(err)?;
        let lm_head = if model.tensor("lm_head.weight").is_some() {
            load_matrix(model, "lm_head.weight", false, ov)?
        } else if arch.tie_word_embeddings {
            // Tied: reuse the embedding matrix (re-open, cheap for Mapped).
            load_matrix(model, "model.embed_tokens.weight", false, ov)?
        } else {
            return Err(CmfError::MissingTensor(
                "lm_head.weight (and tie_word_embeddings is false)".into(),
            ));
        };

        // ── Linear-core geometry (required if any linear layer exists) ──
        let has_linear = arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::LinearAttention));
        let mut vmf_cfg = None;
        let mut gdn_cfg = None;
        if has_linear {
            let lc = arch.linear_core.as_ref().ok_or_else(|| {
                CmfError::Parse(
                    "model has LinearAttention layers but no arch.linear_core — \
                     reconvert with the current converter"
                        .into(),
                )
            })?;
            let need = |v: Option<usize>, name: &str| {
                v.ok_or_else(|| CmfError::Parse(format!("linear core needs arch.{name}")))
            };
            match lc.kind.as_str() {
                "vmf_phase" => {
                    vmf_cfg = Some(VmfPhaseCfg {
                        num_heads: lc.num_heads,
                        nphase: need(lc.nphase, "linear_core.nphase")?,
                        value_head_dim: lc.value_head_dim,
                        hidden_size: arch.hidden_size,
                        // θ-mass (η′): default 0 (massless); CMF_PHASE_MASS
                        // widens the phase kernel for folded-unhealed models.
                        phase_mass: std::env::var("CMF_PHASE_MASS")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0.0),
                    });
                }
                "gated_delta_net" => {
                    gdn_cfg = Some(GdnCfg {
                        num_v_heads: lc.num_heads,
                        num_k_heads: need(arch.linear_num_key_heads, "linear_num_key_heads")?,
                        key_head_dim: need(arch.linear_key_head_dim, "linear_key_head_dim")?,
                        value_head_dim: lc.value_head_dim,
                        conv_kernel: need(arch.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
                        hidden_size: arch.hidden_size,
                        rms_eps: arch.rms_norm_eps,
                    });
                }
                other => {
                    return Err(CmfError::Parse(format!(
                        "unknown linear core '{other}' (this runtime executes: \
                         gated_delta_net, vmf_phase)"
                    )));
                }
            }
        }

        // ── KDA geometry (Kimi Linear / Kimi-K3 delta-attention layers) ──
        let has_kda = arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::Kda));
        let kda_cfg = if has_kda {
            let need = |v: Option<usize>, name: &str| {
                v.ok_or_else(|| CmfError::Parse(format!("KDA core needs arch.{name}")))
            };
            Some(crate::linear_core::KdaCfg {
                num_heads: need(arch.linear_num_key_heads, "linear_num_key_heads")?,
                head_k_dim: need(arch.linear_key_head_dim, "linear_key_head_dim")?,
                head_v_dim: need(arch.linear_value_head_dim, "linear_value_head_dim")?,
                conv_kernel: need(arch.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
                hidden_size: arch.hidden_size,
                rms_eps: arch.rms_norm_eps,
            })
        } else {
            None
        };

        // ── Short-convolution geometry (LFM2 conv mixer layers) ──
        let has_short_conv = arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::ShortConv));
        let short_conv_cfg = if has_short_conv {
            Some(ShortConvCfg {
                hidden_size: arch.hidden_size,
                kernel: arch.linear_conv_kernel_dim.ok_or_else(|| {
                    CmfError::Parse(
                        "model has ShortConv layers but no arch.linear_conv_kernel_dim — \
                         reconvert with the current converter"
                            .into(),
                    )
                })?,
            })
        } else {
            None
        };

        // ── Layers ──
        let load_full_attn = |prefix: &str, layer: Option<usize>| -> Result<AttnKind, CmfError> {
            let t = |suffix: &str| load_matrix(model, &format!("{prefix}{suffix}"), force_f32, ov);
            let n = |suffix: &str| -> Option<Vec<f32>> {
                model
                    .tensor(&format!("{prefix}{suffix}"))
                    .and_then(|_| load_f32(model, &format!("{prefix}{suffix}"), ov).ok())
            };
            // DeepSeek-V2 MLA: the latent projections replace the k/v pair.
            if let Some(mla) = arch.mla.as_ref() {
                // Compressed q (K3/V3): q_a → rms → q_b; direct otherwise.
                let (q_proj, q_a, q_a_norm) = if mla.q_lora_rank.is_some() {
                    (
                        t("self_attn.q_b_proj.weight")?,
                        Some(t("self_attn.q_a_proj.weight")?),
                        Some(n("self_attn.q_a_layernorm.weight").ok_or_else(|| {
                            CmfError::Parse(format!("{prefix}: MLA needs q_a_layernorm"))
                        })?),
                    )
                } else {
                    (t("self_attn.q_proj.weight")?, None, None)
                };
                let hd = mla.qk_rope_head_dim + mla.qk_nope_head_dim;
                let nh = q_proj.rows() / hd;
                // YaRN mscale²: DeepSeek corrects the softmax scale by
                // (0.1·mscale_all_dim·ln(factor)+1)².
                let mut scale = 1.0 / (hd as f32).sqrt();
                if let Some(y) = arch.yarn.as_ref() {
                    if let Some(m) = y.mscale_all_dim.filter(|&m| m > 0.0) {
                        let ms = 0.1 * m * y.factor.ln() + 1.0;
                        scale *= ms * ms;
                    }
                }
                return Ok(AttnKind::Mla(Box::new(crate::pipeline::MlaWeights {
                    q_proj,
                    q_a,
                    q_a_norm,
                    kv_a: t("self_attn.kv_a_proj_with_mqa.weight")?,
                    kv_a_norm: n("self_attn.kv_a_layernorm.weight").ok_or_else(|| {
                        CmfError::Parse(format!("{prefix}: MLA needs kv_a_layernorm"))
                    })?,
                    kv_b: t("self_attn.kv_b_proj.weight")?,
                    o_proj: t("self_attn.o_proj.weight")?,
                    nh,
                    qk_rope: mla.qk_rope_head_dim,
                    qk_nope: mla.qk_nope_head_dim,
                    v_dim: mla.v_head_dim,
                    lora: mla.kv_lora_rank,
                    scale,
                    nope: mla.nope,
                })));
            }
            let wq = t("self_attn.q_proj.weight")?;
            let nh = layer
                .and_then(|li| {
                    arch.attention_heads_per_layer
                        .as_ref()
                        .and_then(|v| v.get(li).copied())
                })
                .unwrap_or(arch.num_attention_heads);
            // Qwen3.5 output gate: q_proj rows = 2·nh·hd (per-head [q; gate]).
            // Gemma-4 global layers legitimately have nh·global_head_dim
            // rows (which can equal 2·nh·hd) — never gated.
            let output_gate = arch.global_head_dim.is_none() && wq.rows() == 2 * nh * arch.head_dim;
            // Gemma-4 global layers run MQA at global_head_dim — their
            // q_proj legitimately carries nh·ghd rows.
            let is_global_layer = arch.global_head_dim.is_some()
                && layer.is_some_and(|li| {
                    arch.sliding_window_pattern
                        .is_some_and(|p| p > 0 && (li + 1) % p == 0)
                });
            let expect = if is_global_layer {
                nh * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                nh * arch.head_dim
            };
            if !output_gate && wq.rows() != expect {
                return Err(CmfError::Parse(format!(
                    "{prefix}self_attn.q_proj.weight rows={} != heads({nh}) * head_dim({})",
                    wq.rows(),
                    expect / nh.max(1)
                )));
            }
            let gate_name = format!("{prefix}self_attn.g_proj.weight");
            let softplus_gate = if model.tensor(&gate_name).is_some() {
                let gate = load_matrix(model, &gate_name, force_f32, ov)?;
                if gate.cols() != arch.hidden_size {
                    return Err(CmfError::Parse(format!(
                        "{gate_name} cols={} != hidden_size ({})",
                        gate.cols(),
                        arch.hidden_size
                    )));
                }
                let per_head = if gate.rows() == nh {
                    true
                } else if gate.rows() == nh * arch.head_dim {
                    false
                } else {
                    return Err(CmfError::Parse(format!(
                        "{gate_name} rows={} must equal heads ({nh}) or heads*head_dim ({})",
                        gate.rows(),
                        nh * arch.head_dim
                    )));
                };
                Some((gate, per_head))
            } else {
                None
            };
            // Qwen2-family projection biases (by tensor presence).
            let bias = match (
                n("self_attn.q_proj.bias"),
                n("self_attn.k_proj.bias"),
                n("self_attn.v_proj.bias"),
            ) {
                (Some(a), Some(b), Some(c)) => Some((a, b, c)),
                _ => None,
            };
            Ok(AttnKind::Full {
                wq,
                wk: t("self_attn.k_proj.weight")?,
                wv: t("self_attn.v_proj.weight")?,
                wo: t("self_attn.o_proj.weight")?,
                q_norm: n("self_attn.q_norm.weight"),
                k_norm: n("self_attn.k_norm.weight"),
                output_gate,
                softplus_gate,
                bias,
            })
        };

        let load_linear_attn = |prefix: &str| -> Result<AttnKind, CmfError> {
            if gdn_cfg.is_some() {
                // Faithful vendor operator: tensor names 1:1 with the source.
                let t = |suffix: &str| {
                    load_matrix(
                        model,
                        &format!("{prefix}linear_attn.{suffix}"),
                        force_f32,
                        ov,
                    )
                };
                let f = |suffix: &str| {
                    load_f32(model, &format!("{prefix}linear_attn.{suffix}"), ov).map_err(err)
                };
                return Ok(AttnKind::LinearGdn(GdnWeights {
                    in_proj_qkv: t("in_proj_qkv.weight")?,
                    in_proj_z: t("in_proj_z.weight")?,
                    in_proj_a: t("in_proj_a.weight")?,
                    in_proj_b: t("in_proj_b.weight")?,
                    conv1d: f("conv1d.weight")?,
                    a_log: f("A_log")?,
                    dt_bias: f("dt_bias")?,
                    norm: f("norm.weight")?,
                    out_proj: t("out_proj.weight")?,
                }));
            }
            let t = |suffix: &str| {
                load_matrix(model, &format!("{prefix}vmf_attn.{suffix}"), force_f32, ov)
            };
            let a_log = load_f32(model, &format!("{prefix}vmf_attn.A_log"), ov).map_err(err)?;
            // Selective-write gate κ (hybrid_k core): optional by tensor
            // presence — files without it run the classic phase kernel
            // bit-identically.
            let k_gate = if model
                .tensor(&format!("{prefix}vmf_attn.k_gate.weight"))
                .is_some()
            {
                Some((
                    t("k_gate.weight")?,
                    load_f32(model, &format!("{prefix}vmf_attn.k_gate.bias"), ov).map_err(err)?,
                ))
            } else {
                None
            };
            Ok(AttnKind::Linear(VmfPhaseWeights {
                thq: t("thq.weight")?,
                thk: t("thk.weight")?,
                v_proj: t("v_proj.weight")?,
                out_proj: t("out_proj.weight")?,
                decay: a_log.iter().map(|&a| (-(a as f64).exp()).exp()).collect(),
                k_gate,
            }))
        };

        // LFM2 short-conv mixer: in_proj [3·hidden, hidden], a depthwise
        // conv (stored f16 as `[hidden, 1, kernel]` → flattened taps), and
        // out_proj [hidden, hidden]. Names canonicalized at convert time.
        let load_short_conv = |prefix: &str| -> Result<AttnKind, CmfError> {
            let t = |suffix: &str| {
                load_matrix(
                    model,
                    &format!("{prefix}short_conv.{suffix}"),
                    force_f32,
                    ov,
                )
            };
            Ok(AttnKind::ShortConv(ShortConvWeights {
                in_proj: t("in_proj.weight")?,
                conv: load_f32(model, &format!("{prefix}short_conv.conv.weight"), ov)
                    .map_err(err)?,
                out_proj: t("out_proj.weight")?,
            }))
        };

        // KDA layer (Kimi Linear / Kimi-K3): faithful vendor tensors under
        // the `kda_attn.` canonical prefix. The output gate is full-rank
        // (g_proj, K3) or low-rank (g_a/g_b, Kimi-Linear-48B) by presence.
        let load_kda = |prefix: &str| -> Result<AttnKind, CmfError> {
            let t = |suffix: &str| {
                load_matrix(model, &format!("{prefix}kda_attn.{suffix}"), force_f32, ov)
            };
            let f = |suffix: &str| {
                load_f32(model, &format!("{prefix}kda_attn.{suffix}"), ov).map_err(err)
            };
            let gate = if model
                .tensor(&format!("{prefix}kda_attn.g_proj.weight"))
                .is_some()
            {
                crate::linear_core::KdaOutGate::Full(t("g_proj.weight")?)
            } else {
                crate::linear_core::KdaOutGate::LowRank(
                    t("g_a_proj.weight")?,
                    t("g_b_proj.weight")?,
                )
            };
            Ok(AttnKind::Kda(Box::new(crate::linear_core::KdaWeights {
                q_proj: t("q_proj.weight")?,
                k_proj: t("k_proj.weight")?,
                v_proj: t("v_proj.weight")?,
                conv_q: f("q_conv1d.weight")?,
                conv_k: f("k_conv1d.weight")?,
                conv_v: f("v_conv1d.weight")?,
                f_a: t("f_a_proj.weight")?,
                f_b: t("f_b_proj.weight")?,
                dt_bias: f("dt_bias")?,
                a_log: f("A_log")?,
                b_proj: t("b_proj.weight")?,
                gate,
                o_norm: f("o_norm.weight")?,
                o_proj: t("o_proj.weight")?,
                gate_lower_bound: arch.kda_gate_lower_bound.map(|v| v as f32),
            })))
        };

        fn anyhow_like(ok: bool) -> Result<(), ()> {
            if ok { Ok(()) } else { Err(()) }
        }
        let mut layers = Vec::with_capacity(arch.num_layers);
        let is_g3n = arch.g3n.is_some();
        // Architectures that load their own layer stack below. DeepSeek-V4
        // has none of the canonical projections — no q/k/v/o_proj, no
        // per-layer gate_proj — so the generic loop would demand
        // `self_attn.q_proj.weight` and fail before its own loader ever ran.
        let owns_its_layers = is_g3n || arch.arch_name == "deepseek_v4";
        for li in 0..(if owns_its_layers { 0 } else { arch.num_layers }) {
            let prefix = format!("model.layers.{li}.");
            let attn = match arch.layer_types.get(li) {
                Some(LayerType::LinearAttention) => load_linear_attn(&prefix)?,
                Some(LayerType::Kda) => load_kda(&prefix)?,
                Some(LayerType::ShortConv) => load_short_conv(&prefix)?,
                _ => load_full_attn(&prefix, Some(li))?,
            };
            // Gemma-2/3 sandwich: `pre_feedforward_layernorm` present →
            // it is the pre-FFN norm, and post_attention/post_feedforward
            // норms apply to the branch OUTPUTS before their residuals.
            let pre_ffn = format!("{prefix}pre_feedforward_layernorm.weight");
            let sandwich = model.tensor(&pre_ffn).is_some();
            layers.push(LayerWeights {
                input_norm: load_f32(model, &format!("{prefix}input_layernorm.weight"), ov)
                    .map_err(err)?,
                post_norm: if sandwich {
                    load_f32(model, &pre_ffn, ov).map_err(err)?
                } else {
                    load_f32(
                        model,
                        &format!("{prefix}post_attention_layernorm.weight"),
                        ov,
                    )
                    .map_err(err)?
                },
                attn_out_norm: if sandwich {
                    Some(
                        load_f32(
                            model,
                            &format!("{prefix}post_attention_layernorm.weight"),
                            ov,
                        )
                        .map_err(err)?,
                    )
                } else {
                    None
                },
                ffn_out_norm: if sandwich {
                    Some(
                        load_f32(
                            model,
                            &format!("{prefix}post_feedforward_layernorm.weight"),
                            ov,
                        )
                        .map_err(err)?,
                    )
                } else {
                    None
                },
                // Gemma-4: learned scalar multiplying the layer output.
                layer_scale: model
                    .tensor(&format!("{prefix}layer_scalar"))
                    .and_then(|_| {
                        load_f32(model, &format!("{prefix}layer_scalar"), ov)
                            .ok()
                            .and_then(|v| v.first().copied())
                    }),
                // FFN always quantized — masks run sparse on quant bytes.
                ffn: build_layer_ffn(model, &arch, li, false, ov)?,
                attn,
            });
        }

        // ── MTP head (optional, spec §2.1) ──
        //
        // The header declaring an MTP head is not the same as the file
        // carrying one. DeepSeek-V4's config announces a next-token predictor
        // whose weights the converter does not map (they are spelled `mtp.N.*`
        // and have none of the canonical projections), so demanding
        // `model.mtp.layers.0.self_attn.q_proj.weight` failed a model that is
        // otherwise complete. Presence in the directory decides.
        let mtp_present = model
            .tensor("model.mtp.layers.0.self_attn.q_proj.weight")
            .is_some()
            || model.tensor("model.mtp.eh_proj.weight").is_some();
        if arch.mtp.is_some() && !mtp_present {
            tracing::info!(
                "header declares an MTP head but the file carries none — \
                 loading without it"
            );
        }
        let mtp = if let Some(cfg) = arch.mtp.as_ref().filter(|_| mtp_present) {
            if cfg.num_layers != 1 {
                return Err(CmfError::Parse(format!(
                    "MTP with {} blocks not supported yet (only 1)",
                    cfg.num_layers
                )));
            }
            let p = "model.mtp.";
            let attn = load_full_attn("model.mtp.layers.0.", None)?;
            Some(MtpModule {
                enorm: load_f32(model, &format!("{p}enorm.weight"), ov).map_err(err)?,
                hnorm: load_f32(model, &format!("{p}hnorm.weight"), ov).map_err(err)?,
                eh_proj: load_matrix(model, &format!("{p}eh_proj.weight"), false, ov)?,
                layer: LayerWeights {
                    attn_out_norm: None,
                    ffn_out_norm: None,
                    layer_scale: None,
                    input_norm: load_f32(model, &format!("{p}layers.0.input_layernorm.weight"), ov)
                        .map_err(err)?,
                    post_norm: load_f32(
                        model,
                        &format!("{p}layers.0.post_attention_layernorm.weight"),
                        ov,
                    )
                    .map_err(err)?,
                    // Whatever the block actually carries: DeepSeek's MTP
                    // layer is dense, Qwen3.6's is a full MoE (router + 256
                    // experts + shared). Same builder as a backbone layer.
                    ffn: build_ffn_at(model, &arch, &format!("{p}layers.0."), false, ov)?,
                    attn,
                },
                final_norm: load_f32(model, &format!("{p}norm.weight"), ov).map_err(err)?,
                kv: LayerKvCache::new(arch.num_kv_heads, arch.head_dim),
            })
        } else {
            None
        };

        tracing::info!(
            "Pipeline loaded: {} | {}L ({} linear) | {:.2}B params | storage: {} | MTP: {}",
            arch.arch_name,
            arch.num_layers,
            arch.layer_types
                .iter()
                .filter(|t| matches!(t, LayerType::LinearAttention))
                .count(),
            model.total_param_count() as f64 / 1e9,
            if force_f32 {
                "f32 (masked)"
            } else {
                "quantized mmap"
            },
            if mtp.is_some() { "yes" } else { "no" }
        );

        // KV window: the descriptor's max, capped for dev-box safety;
        // CMF_MAX_SEQ overrides the cap (long-context runs).
        let cap = std::env::var("CMF_MAX_SEQ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8192);
        let max_seq_len = arch.max_position_embeddings.min(cap);

        // Looped Transformer: total virtual layers = physical × num_loops.
        let total_layers = arch.num_layers * arch.num_loops;

        let mut pipeline = Pipeline::new(
            tokenizer,
            PipelineWeights {
                embed_tokens,
                layers,
                lm_head,
                final_norm,
            },
            arch.hidden_size,
            arch.intermediate_size,
            arch.num_attention_heads,
            arch.num_kv_heads,
            arch.head_dim,
            total_layers,
            arch.num_layers, // physical layers in weights
            arch.loop_final_norm,
            arch.vocab_size,
            arch.rms_norm_eps,
            arch.rope_theta as f32,
            arch.norm_style,
            max_seq_len,
            sampler_config,
        );
        let rotary = ((arch.head_dim as f32 * arch.partial_rotary_factor) as usize).max(2);
        pipeline.set_rotary(rotary, arch.rope_theta as f32);
        pipeline.attention_heads_per_layer = arch.attention_heads_per_layer.clone();
        if let Some(yarn) = &arch.yarn {
            pipeline.inv_freq = std::sync::Arc::new(crate::attention::yarn_inv_freq(
                rotary,
                arch.rope_theta as f32,
                yarn.factor,
                yarn.original_max_position_embeddings,
                yarn.beta_fast,
                yarn.beta_slow,
            ));
            pipeline.rope_scale = yarn.attention_factor;
        }
        // Gemma-family extras: embedding scale, attention-scale
        // override, and (Gemma-3) sliding-window layers with their own
        // local RoPE base.
        pipeline.embed_multiplier = arch.embed_multiplier;
        pipeline.logit_multiplier = arch.logit_multiplier;
        if let Some(qpas) = arch.query_pre_attn_scalar {
            pipeline.attn_scale = 1.0 / (qpas as f32).sqrt();
        }
        if let (Some(w), Some(p)) = (arch.sliding_window, arch.sliding_window_pattern) {
            pipeline.swa = Some((w, p));
            if let Some(base) = arch.rope_local_base_freq {
                pipeline.inv_freq_local = Some(std::sync::Arc::new(
                    crate::attention::rope_inv_freq(rotary, base as f32),
                ));
            }
        }
        let explicit_sliding: Vec<bool> = arch
            .layer_types
            .iter()
            .map(|t| matches!(t, cortiq_core::LayerType::SlidingAttention))
            .collect();
        if explicit_sliding.iter().any(|&v| v) {
            pipeline.sliding_layers = Some(explicit_sliding);
            if let Some(w) = arch.sliding_window {
                pipeline.swa = Some((w, usize::MAX));
            }
            let local_rotary = ((arch.head_dim as f32
                * arch
                    .local_partial_rotary_factor
                    .unwrap_or(arch.partial_rotary_factor))
                as usize)
                .max(2);
            pipeline.rotary_dim_local = Some(local_rotary);
            if let Some(base) = arch.rope_local_base_freq {
                pipeline.inv_freq_local = Some(std::sync::Arc::new(
                    crate::attention::rope_inv_freq(local_rotary, base as f32),
                ));
            }
        }
        // Gemma-4: global layers run their own geometry (MQA at
        // global_head_dim) with a proportional RoPE — the first
        // factor·head_dim dims rotate, the zero-padded tail is identity.
        if let (Some(ghd), Some(gkv)) = (arch.global_head_dim, arch.num_global_kv_heads) {
            pipeline.global_attn = Some((ghd, gkv));
            let prf = arch.global_partial_rotary_factor.unwrap_or(1.0);
            let half = ghd / 2;
            let ra = (((prf * ghd as f32) as usize) / 2).min(half);
            let mut f = vec![0.0f32; half];
            for (i, slot) in f.iter_mut().enumerate().take(ra) {
                *slot = 1.0 / (arch.rope_theta as f32).powf(2.0 * i as f32 / ghd as f32);
            }
            pipeline.inv_freq_global = Some(std::sync::Arc::new(f));
            // Re-shape the global layers' KV storage to their geometry.
            // An explicit layer_types map wins over the numeric pattern
            // (explicit tags set swa's pattern to usize::MAX, which
            // would otherwise leave every global cache mis-shaped).
            let global_at = |li: usize| -> bool {
                match &pipeline.sliding_layers {
                    Some(map) => !map.get(li).copied().unwrap_or(false),
                    None => pipeline
                        .swa
                        .map(|(_, p)| p > 0 && p != usize::MAX && (li + 1) % p == 0)
                        .unwrap_or(false),
                }
            };
            for li in 0..arch.num_layers {
                if global_at(li) {
                    pipeline.kv_cache.layers[li] = crate::kv_cache::LayerKvCache::new(gkv, ghd);
                }
            }
        }
        // MLA (DeepSeek-V2): the expand-to-MHA cache holds nh heads of
        // rope+nope dims; rotary covers the rope prefix.
        if let Some(mla) = arch.mla.as_ref() {
            let hd = mla.qk_rope_head_dim + mla.qk_nope_head_dim;
            pipeline.head_dim = hd;
            pipeline.num_kv_heads = arch.num_attention_heads;
            pipeline.rotary_dim = mla.qk_rope_head_dim;
            let half = mla.qk_rope_head_dim / 2;
            let mut f = vec![0.0f32; half];
            for (i, slot) in f.iter_mut().enumerate() {
                *slot = 1.0
                    / (arch.rope_theta as f32)
                        .powf(2.0 * i as f32 / mla.qk_rope_head_dim as f32);
            }
            pipeline.inv_freq = std::sync::Arc::new(f);
            for li in 0..arch.num_layers {
                pipeline.kv_cache.layers[li] =
                    crate::kv_cache::LayerKvCache::new(arch.num_attention_heads, hd);
            }
        }
        // Per-frequency rope divisors (MiniCPM3 longrope short_factor):
        // served at the native window with the trained per-dim factors.
        // Applied after every inv_freq build (plain, YaRN, MLA).
        if let Some(fac) = &arch.rope_freq_factors {
            let mut f = pipeline.inv_freq.as_ref().clone();
            for (i, v) in f.iter_mut().enumerate() {
                if let Some(&d) = fac.get(i) {
                    *v /= d as f32;
                }
            }
            pipeline.inv_freq = std::sync::Arc::new(f);
        }
        pipeline.attn_v_norm = arch.attn_v_norm;
        pipeline.final_softcap = arch.final_logit_softcapping.map(|c| c as f32);
        pipeline.attn_softcap = arch.attn_logit_softcapping.unwrap_or(0.0) as f32;
        pipeline.vmf_cfg = vmf_cfg;
        pipeline.gdn_cfg = gdn_cfg;
        pipeline.kda_cfg = kda_cfg;
        if let Some(gc) = arch.g3n.as_ref() {
            use crate::g3n::{G3nAltUp, G3nGlobals, G3nLaurel, G3nLayer};
            anyhow_like(gc.altup_num_inputs == crate::g3n::ALTUP_N).map_err(|_| {
                CmfError::Parse(format!(
                    "g3n: altup_num_inputs {} != supported {}",
                    gc.altup_num_inputs,
                    crate::g3n::ALTUP_N
                ))
            })?;
            let t = |name: &str| load_matrix(model, name, force_f32, ov);
            let f = |name: &str| load_f32(model, name, ov).map_err(err);
            let mut altup_proj = Vec::new();
            let mut altup_unembed = Vec::new();
            for i in 0..crate::g3n::ALTUP_N - 1 {
                altup_proj.push(t(&format!("model.altup_projections.{i}.weight"))?);
                altup_unembed.push(t(&format!("model.altup_unembed_projections.{i}.weight"))?);
            }
            let first_shared = arch.num_layers.saturating_sub(gc.num_kv_shared_layers);
            let sliding_of = |li: usize| {
                matches!(
                    arch.layer_types.get(li),
                    Some(cortiq_core::LayerType::SlidingAttention)
                )
            };
            let mut g3n_layers = Vec::with_capacity(arch.num_layers);
            for li in 0..arch.num_layers {
                let pfx = format!("model.layers.{li}.");
                let shared = li >= first_shared && first_shared > 0;
                let share_src = if shared {
                    let want = sliding_of(li);
                    (0..first_shared).rev().find(|&j| sliding_of(j) == want)
                } else {
                    None
                };
                g3n_layers.push(G3nLayer {
                    altup: G3nAltUp {
                        router_norm: f(&format!("{pfx}altup.router_norm.weight"))?,
                        modality_router: t(&format!("{pfx}altup.modality_router.weight"))?,
                        prediction_coefs: t(&format!("{pfx}altup.prediction_coefs.weight"))?,
                        correction_coefs: t(&format!("{pfx}altup.correction_coefs.weight"))?,
                        correct_output_scale: f(&format!("{pfx}altup.correct_output_scale"))?,
                    },
                    laurel: G3nLaurel {
                        left: t(&format!("{pfx}laurel.linear_left.weight"))?,
                        right: t(&format!("{pfx}laurel.linear_right.weight"))?,
                        post_norm: f(&format!("{pfx}laurel.post_laurel_norm.weight"))?,
                    },
                    input_norm: f(&format!("{pfx}input_layernorm.weight"))?,
                    post_attn_norm: f(&format!("{pfx}post_attention_layernorm.weight"))?,
                    pre_ffw_norm: f(&format!("{pfx}pre_feedforward_layernorm.weight"))?,
                    post_ffw_norm: f(&format!("{pfx}post_feedforward_layernorm.weight"))?,
                    wq: t(&format!("{pfx}self_attn.q_proj.weight"))?,
                    wk: if shared {
                        None
                    } else {
                        Some(t(&format!("{pfx}self_attn.k_proj.weight"))?)
                    },
                    wv: if shared {
                        None
                    } else {
                        Some(t(&format!("{pfx}self_attn.v_proj.weight"))?)
                    },
                    wo: t(&format!("{pfx}self_attn.o_proj.weight"))?,
                    q_norm: f(&format!("{pfx}self_attn.q_norm.weight"))?,
                    k_norm: if shared {
                        None
                    } else {
                        Some(f(&format!("{pfx}self_attn.k_norm.weight"))?)
                    },
                    kv_share_src: share_src,
                    sliding: sliding_of(li),
                    gate: t(&format!("{pfx}mlp.gate_proj.weight"))?,
                    up: t(&format!("{pfx}mlp.up_proj.weight"))?,
                    down: t(&format!("{pfx}mlp.down_proj.weight"))?,
                    sparsity: gc
                        .activation_sparsity
                        .get(li)
                        .copied()
                        .unwrap_or(0.0),
                    ple_gate: t(&format!("{pfx}per_layer_input_gate.weight"))?,
                    ple_proj: t(&format!("{pfx}per_layer_projection.weight"))?,
                    post_ple_norm: f(&format!("{pfx}post_per_layer_input_norm.weight"))?,
                });
            }
            let hd = arch.head_dim;
            let globals = G3nGlobals {
                altup_proj,
                altup_unembed,
                ple_embed: t("model.embed_tokens_per_layer.weight")?,
                ple_model_proj: t("model.per_layer_model_projection.weight")?,
                ple_norm: f("model.per_layer_projection_norm.weight")?,
                ple_vocab: gc.ple_vocab,
                ple_dim: gc.ple_dim,
                num_layers: arch.num_layers,
                hidden: arch.hidden_size,
                rms_eps: arch.rms_norm_eps,
                inv_freq_local: crate::attention::rope_inv_freq(
                    hd,
                    arch.rope_local_base_freq.unwrap_or(10_000.0) as f32,
                ),
                inv_freq_global: crate::attention::rope_inv_freq(hd, arch.rope_theta as f32),
                window: arch.sliding_window.unwrap_or(512),
            };
            pipeline.g3n = Some(Box::new((globals, g3n_layers)));
        }
        // DeepSeek-V4: its own stack, selected by the arch name the
        // converter wrote. Loading failure is fatal rather than a silent
        // fallback — the generic loop cannot represent this model at all,
        // so a fallback would decode noise.
        if arch.arch_name == "deepseek_v4" {
            let moe = arch
                .moe
                .as_ref()
                .ok_or_else(|| CmfError::Parse("deepseek_v4: no moe config".into()))?;
            let cfg = crate::dsv4::Dsv4Cfg {
                dim: arch.hidden_size,
                n_heads: arch.num_attention_heads,
                head_dim: arch.head_dim,
                // The rope tail: `partial_rotary_factor` carries it when the
                // conversion recorded it (rd/head_dim), which the tensors
                // cannot reveal. Files converted before that carry 1.0,
                // meaning "unset" here rather than "rotate everything" —
                // for those the release's 64 stands in, which is what they
                // were converted from.
                rope_head_dim: if arch.partial_rotary_factor < 1.0 {
                    (((arch.head_dim as f32 * arch.partial_rotary_factor) as usize) & !1)
                        .clamp(2, arch.head_dim)
                } else {
                    64.min(arch.head_dim)
                },
                // The LoRA ranks and the group count ARE visible in the
                // weights, and reading them there means a re-tuned
                // checkpoint loads without touching this code.
                q_lora_rank: 0,
                o_lora_rank: 0,
                // Derived below from wo_a's shape — the attention output is
                // n_heads*head_dim wide and wo_a takes one group of it per
                // row block, so groups = width / wo_a.cols(). A pinned 8 is
                // right for the release and wrong for anything else, which
                // is exactly what made a toy checkpoint impossible to
                // compare against the reference.
                o_groups: 8,
                hc_mult: 4,
                hc_sinkhorn_iters: 20,
                hc_eps: 1e-6,
                norm_eps: arch.rms_norm_eps as f32,
                n_routed_experts: moe.num_experts,
                top_k: moe.top_k,
                moe_inter: moe.moe_intermediate_size,
                route_scale: moe.routed_scaling_factor.unwrap_or(1.0) as f32,
                // config.json's `swiglu_limit`, which the header has no
                // field for. The release ships 10.0; a checkpoint that
                // retunes it would need this read from the config, so it
                // sits next to the other pinned constants rather than
                // hiding inside the expert.
                swiglu_limit: 10.0,
                window: arch.sliding_window.unwrap_or(128),
                index_topk: 512,
                vocab: arch.vocab_size,
            };
            let (g, dl) = crate::dsv4::load(model, &cfg, arch.num_layers)
                .map_err(|e| CmfError::Parse(format!("deepseek_v4: {e}")))?;
            // Read the ranks off the weights that define them: wq_a's
            // rows ARE q_lora_rank, and wo_b's columns are groups x
            // o_lora_rank. A header field could disagree with the file;
            // these cannot.
            let mut cfg = cfg;
            if let Some(l0) = dl.first() {
                cfg.q_lora_rank = l0.wq_a.rows();
                let attn_width = arch.num_attention_heads * arch.head_dim;
                if l0.wo_a.cols() > 0 && attn_width % l0.wo_a.cols() == 0 {
                    cfg.o_groups = (attn_width / l0.wo_a.cols()).max(1);
                }
                cfg.o_lora_rank = l0.wo_b.cols() / cfg.o_groups.max(1);
                cfg.hc_mult = (l0.hc_attn_fn.len() / l0.hc_attn_base.len().max(1)) / cfg.dim.max(1);
                if cfg.hc_mult == 0 {
                    cfg.hc_mult = 4;
                }
            }
            // RoPE rides only the last `rope_head_dim` of each head, and the
            // reference builds its frequencies over THAT width — not over
            // head_dim, which is 512 here. The generic path above sized them
            // by head_dim, giving 1/base^(2i/512) where 1/base^(2i/64) is
            // wanted: every position rotated by the wrong angle.
            //
            // YaRN is applied unconditionally by the reference (its guard is
            // `original_seq_len > 0`, not the sequence length), so it belongs
            // in these frequencies too. Older configs spell the key `type`
            // rather than `rope_type`; when the header carries no profile the
            // release's own numbers stand in, which is better than silently
            // decoding with unscaled frequencies.
            let (yf, yo, ybf, ybs) = match &arch.yarn {
                Some(y) => (
                    y.factor,
                    y.original_max_position_embeddings,
                    y.beta_fast,
                    y.beta_slow,
                ),
                None => {
                    tracing::warn!(
                        "deepseek_v4: the header carries no YaRN profile — \
                         falling back to the release's (factor 16, original \
                         65536, beta 32/1). Re-converting with a build that \
                         reads rope_scaling.type would make this exact."
                    );
                    (16.0, 65536, 32.0, 1.0)
                }
            };
            pipeline.inv_freq = std::sync::Arc::new(crate::attention::yarn_inv_freq(
                cfg.rope_head_dim,
                arch.rope_theta as f32,
                yf,
                yo,
                ybf,
                ybs,
            ));
            let st = crate::dsv4::Dsv4State::new(arch.num_layers);
            pipeline.dsv4 = Some(Box::new((g, dl, cfg, st)));
        }
        pipeline.short_conv_cfg = short_conv_cfg;
        pipeline.mtp = mtp;
        pipeline.install_dynamic_routing(model, false);
        // Record the load-time overlay so a later set_active_skill(None)
        // correctly reverts it (the union-diff assumes dyn_active mirrors
        // the live overlay). Blend loads have no single index to revert.
        match ov {
            Overlay::One(sid) => {
                pipeline.dyn_active = model.header.skills.iter().position(|s| &s.id == sid);
            }
            Overlay::Blend(_) => pipeline.dyn_blend_loaded = true,
            Overlay::None => {}
        }
        // B1: apply the measured confidence-calibration temperature, if the
        // file carries one (softmax(logits / T) for reported Born mass).
        if let Some(c) = &model.header.calibration {
            pipeline.set_calib_temp(c.temperature);
        }
        // O(1) Nyström attention (runtime-level, no format change):
        // env CMF_O1 decides; unset falls through to the converter hint
        // in header.provenance.o1_attn (`cortiq convert --o1`), and
        // CMF_O1=off force-disables even the hint. CLI flags override
        // later via set_o1().
        let o1 = match crate::nystrom::o1_from_env() {
            crate::nystrom::O1Env::Off => None,
            crate::nystrom::O1Env::On(cfg) => Some(cfg),
            crate::nystrom::O1Env::Unset => model
                .header
                .provenance
                .as_ref()
                .and_then(|p| p.get("o1_attn"))
                .and_then(crate::nystrom::O1Cfg::from_json),
        };
        if o1.is_some() {
            if pipeline.attn_softcap > 0.0 {
                return Err(CmfError::Parse(
                    "--o1 with attention-logit soft-capping (Gemma-2) is not supported: \
                     the streaming operator has no capped-score form"
                        .into(),
                ));
            }
            pipeline.set_o1(o1);
        }
        Ok(pipeline)
    }

    /// Record per-skill dynamic-routing metadata: which FFN layers each
    /// skill actually replaces (derived from the tensors present, not
    /// the meta `layers` field), and whether the skill is eligible for
    /// cheap dynamic switching (FFN-only). Called once at load.
    pub(crate) fn install_dynamic_routing(&mut self, model: &Arc<CmfModel>, force_f32: bool) {
        self.model = Some(model.clone());
        self.dyn_force_f32 = force_f32;
        let mut per_skill = Vec::with_capacity(model.header.skills.len());
        for sk in &model.header.skills {
            let mut ffn_layers = std::collections::BTreeSet::new();
            let mut non_ffn = false;
            let prefix = format!("skill.{}.", sk.id);
            for t in model.skill_tensors(&sk.id) {
                let rel = &t.name[prefix.len()..]; // e.g. model.layers.20.mlp.down_proj.weight
                let toks: Vec<&str> = rel.split('.').collect();
                if toks.len() >= 5 && toks[0] == "model" && toks[1] == "layers" && toks[3] == "mlp"
                {
                    if let Ok(li) = toks[2].parse::<usize>() {
                        ffn_layers.insert(li);
                        continue;
                    }
                }
                non_ffn = true; // replaces attention / embed / lm_head
            }
            if non_ffn {
                tracing::warn!(
                    "skill '{}' replaces non-FFN tensors — excluded from dynamic \
                     routing (static overlay still works)",
                    sk.id
                );
                per_skill.push(None);
            } else {
                per_skill.push(Some(ffn_layers.into_iter().collect::<Vec<_>>()));
            }
        }
        self.dyn_skill_layers = per_skill;
    }

    /// Switch the overlaid skill for subsequent forwards (dynamic
    /// routing). `idx` = index into model.header.skills; None = backbone.
    /// Rebuilds the FFN of the union of the old and new skill's touched
    /// layers with the new overlay — tensor-source indirection made
    /// dynamic. Cheap: Mapped tensors are re-resolved mmap pointers.
    /// Result is bit-identical to loading the pipeline with that skill.
    pub fn set_active_skill(&mut self, idx: Option<usize>) -> Result<(), CmfError> {
        // Overlay swap changes weights → every cached K/V is stale.
        self.kv_cache.clear();
        self.kv_history.clear();
        if self.dyn_active == idx {
            return Ok(());
        }
        let model = self.model.clone().ok_or_else(|| {
            CmfError::Parse("dynamic routing needs a model-backed pipeline".into())
        })?;
        let mut union: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        if let Some(old) = self.dyn_active {
            if let Some(Some(ls)) = self.dyn_skill_layers.get(old) {
                union.extend(ls.iter().copied());
            }
        }
        let new_id: Option<String> = match idx {
            Some(n) => match self.dyn_skill_layers.get(n) {
                Some(Some(ls)) => {
                    union.extend(ls.iter().copied());
                    Some(model.header.skills[n].id.clone())
                }
                _ => {
                    return Err(CmfError::Parse(format!(
                        "skill index {n} not dynamic-eligible"
                    )));
                }
            },
            None => None,
        };
        let ov = match &new_id {
            Some(s) => Overlay::One(s),
            None => Overlay::None,
        };
        let arch = model.arch();
        for li in union {
            self.weights.layers[li].ffn =
                build_layer_ffn(&model, arch, li, self.dyn_force_f32, &ov)?;
        }
        self.dyn_active = idx;
        Ok(())
    }
}
