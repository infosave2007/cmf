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
use crate::linear_core::{GdnCfg, GdnWeights, VmfPhaseCfg, VmfPhaseWeights};
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
        let src = if model.tensor(&sname).is_some() { &sname } else { name };
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
fn load_f32(model: &CmfModel, name: &str, ov: &Overlay) -> Result<Vec<f32>, String> {
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
    let prefix = format!("model.layers.{li}.");
    let load_dense = |p: &str| -> Result<DenseFfn, CmfError> {
        Ok(DenseFfn {
            gate_proj: load_matrix(model, &format!("{p}gate_proj.weight"), force_f32, ov)?,
            up_proj: load_matrix(model, &format!("{p}up_proj.weight"), force_f32, ov)?,
            down_proj: load_matrix(model, &format!("{p}down_proj.weight"), force_f32, ov)?,
        })
    };
    let router_name = format!("{prefix}mlp.gate.weight");
    if model.tensor(&router_name).is_none() {
        return Ok(FfnKind::Dense(load_dense(&format!("{prefix}mlp."))?));
    }
    let cfg = arch.moe.as_ref().ok_or_else(|| {
        CmfError::Parse(format!("{router_name} present but header has no arch.moe block"))
    })?;
    let experts = (0..cfg.num_experts)
        .map(|e| load_dense(&format!("{prefix}mlp.experts.{e}.")))
        .collect::<Result<Vec<_>, _>>()?;
    let shared = if model
        .tensor(&format!("{prefix}mlp.shared_expert.gate_proj.weight"))
        .is_some()
    {
        Some((
            load_dense(&format!("{prefix}mlp.shared_expert."))?,
            load_matrix(model, &format!("{prefix}mlp.shared_expert_gate.weight"), force_f32, ov)?,
        ))
    } else {
        None
    };
    Ok(FfnKind::Moe(MoeFfn {
        router: load_matrix(model, &router_name, force_f32, ov)?,
        experts,
        top_k: cfg.top_k,
        norm_topk_prob: cfg.norm_topk_prob,
        shared,
        stats: std::cell::RefCell::new(Vec::new()),
    }))
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
            let data = blend_f32(model, name, list)
                .map_err(|e| CmfError::Parse(format!("blend: {e}")))?;
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
    pub fn from_model(model: &Arc<CmfModel>, sampler_config: SamplerConfig) -> Result<Self, CmfError> {
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
                    model.header.skills.iter().map(|s| &s.id).collect::<Vec<_>>()
                )));
            }
            tracing::info!(
                "skill '{sid}': {} replacement tensors overlaid",
                model.skill_tensors(sid).count()
            );
        }
        let arch = model.arch().clone();
        let err = |e: String| CmfError::Parse(format!("weight loading: {e}"));

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
                        conv_kernel: need(
                            arch.linear_conv_kernel_dim,
                            "linear_conv_kernel_dim",
                        )?,
                        hidden_size: arch.hidden_size,
                        rms_eps: arch.rms_norm_eps as f64,
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

        // ── Layers ──
        let load_full_attn = |prefix: &str| -> Result<AttnKind, CmfError> {
            let t = |suffix: &str| load_matrix(model, &format!("{prefix}{suffix}"), force_f32, ov);
            let n = |suffix: &str| -> Option<Vec<f32>> {
                model
                    .tensor(&format!("{prefix}{suffix}"))
                    .and_then(|_| load_f32(model, &format!("{prefix}{suffix}"), ov).ok())
            };
            let wq = t("self_attn.q_proj.weight")?;
            // Qwen3.5 output gate: q_proj rows = 2·nh·hd (per-head [q; gate]).
            let output_gate = wq.rows() == 2 * arch.num_attention_heads * arch.head_dim;
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
                bias,
            })
        };

        let load_linear_attn = |prefix: &str| -> Result<AttnKind, CmfError> {
            if gdn_cfg.is_some() {
                // Faithful vendor operator: tensor names 1:1 with the source.
                let t = |suffix: &str| {
                    load_matrix(model, &format!("{prefix}linear_attn.{suffix}"), force_f32, ov)
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
            let t = |suffix: &str| load_matrix(model, &format!("{prefix}vmf_attn.{suffix}"), force_f32, ov);
            let a_log = load_f32(model, &format!("{prefix}vmf_attn.A_log"), ov).map_err(err)?;
            Ok(AttnKind::Linear(VmfPhaseWeights {
                thq: t("thq.weight")?,
                thk: t("thk.weight")?,
                v_proj: t("v_proj.weight")?,
                out_proj: t("out_proj.weight")?,
                decay: a_log
                    .iter()
                    .map(|&a| (-(a as f64).exp()).exp())
                    .collect(),
            }))
        };

        let mut layers = Vec::with_capacity(arch.num_layers);
        for li in 0..arch.num_layers {
            let prefix = format!("model.layers.{li}.");
            let attn = match arch.layer_types.get(li) {
                Some(LayerType::LinearAttention) => load_linear_attn(&prefix)?,
                _ => load_full_attn(&prefix)?,
            };
            layers.push(LayerWeights {
                input_norm: load_f32(model, &format!("{prefix}input_layernorm.weight"), ov).map_err(err)?,
                post_norm: load_f32(model, &format!("{prefix}post_attention_layernorm.weight"), ov)
                    .map_err(err)?,
                // FFN always quantized — masks run sparse on quant bytes.
                ffn: build_layer_ffn(model, &arch, li, false, ov)?,
                attn,
            });
        }

        // ── MTP head (optional, spec §2.1) ──
        let mtp = if let Some(cfg) = &arch.mtp {
            if cfg.num_layers != 1 {
                return Err(CmfError::Parse(format!(
                    "MTP with {} blocks not supported yet (only 1)",
                    cfg.num_layers
                )));
            }
            let p = "model.mtp.";
            let attn = load_full_attn("model.mtp.layers.0.")?;
            Some(MtpModule {
                enorm: load_f32(model, &format!("{p}enorm.weight"), ov).map_err(err)?,
                hnorm: load_f32(model, &format!("{p}hnorm.weight"), ov).map_err(err)?,
                eh_proj: load_matrix(model, &format!("{p}eh_proj.weight"), false, ov)?,
                layer: LayerWeights {
                    input_norm: load_f32(model, &format!("{p}layers.0.input_layernorm.weight"), ov)
                        .map_err(err)?,
                    post_norm: load_f32(
                        model,
                        &format!("{p}layers.0.post_attention_layernorm.weight"),
                        ov,
                    )
                    .map_err(err)?,
                    ffn: FfnKind::Dense(DenseFfn {
                        gate_proj: load_matrix(model, &format!("{p}layers.0.mlp.gate_proj.weight"), false, ov)?,
                        up_proj: load_matrix(model, &format!("{p}layers.0.mlp.up_proj.weight"), false, ov)?,
                        down_proj: load_matrix(model, &format!("{p}layers.0.mlp.down_proj.weight"), false, ov)?,
                    }),
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
            if force_f32 { "f32 (masked)" } else { "quantized mmap" },
            if mtp.is_some() { "yes" } else { "no" }
        );

        // KV window: the descriptor's max, capped for dev-box safety;
        // CMF_MAX_SEQ overrides the cap (long-context runs).
        let cap = std::env::var("CMF_MAX_SEQ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8192);
        let max_seq_len = arch.max_position_embeddings.min(cap);

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
            arch.num_layers,
            arch.vocab_size,
            arch.rms_norm_eps,
            arch.rope_theta as f32,
            arch.norm_style,
            max_seq_len,
            sampler_config,
        );
        let rotary = ((arch.head_dim as f32 * arch.partial_rotary_factor) as usize).max(2);
        pipeline.set_rotary(rotary, arch.rope_theta as f32);
        pipeline.vmf_cfg = vmf_cfg;
        pipeline.gdn_cfg = gdn_cfg;
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
        Ok(pipeline)
    }

    /// Record per-skill dynamic-routing metadata: which FFN layers each
    /// skill actually replaces (derived from the tensors present, not
    /// the meta `layers` field), and whether the skill is eligible for
    /// cheap dynamic switching (FFN-only). Called once at load.
    pub(crate) fn install_dynamic_routing(
        &mut self,
        model: &Arc<CmfModel>,
        force_f32: bool,
    ) {
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
                if toks.len() >= 5
                    && toks[0] == "model"
                    && toks[1] == "layers"
                    && toks[3] == "mlp"
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
        if self.dyn_active == idx {
            return Ok(());
        }
        let model = self
            .model
            .clone()
            .ok_or_else(|| CmfError::Parse("dynamic routing needs a model-backed pipeline".into()))?;
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
                    )))
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
