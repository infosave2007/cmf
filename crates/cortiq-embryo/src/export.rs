//! Export a genome checkpoint into a `.cmf` container the runtime loads:
//! tensor names as the loader expects them (`vmf_attn.*` for the hybrid_k
//! mixer, `self_attn.*` for the anchor, `mlp.shared_expert.*` +
//! `mlp.experts.{e}.*` + `mlp.desc.*` for the routed experts,
//! `lm_head.clusters.weight` for the hierarchical head), the arch block, and
//! our tokenizer.json in the VOCAB section.

use crate::model::{EmbryoCfg, LayerOffs, Layout};
use crate::train::Checkpoint;
use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec, TokenizerBundle};
use cortiq_core::types::TensorDtype;
use std::path::Path;

fn f32_bytes(x: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(x.len() * 4);
    for f in x {
        v.extend_from_slice(&f.to_le_bytes());
    }
    v
}

fn spec(name: &str, shape: &[usize], data: &[f32]) -> TensorSpec {
    assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}: shape/len mismatch");
    TensorSpec { name: name.to_string(), dtype: TensorDtype::F32, shape: shape.to_vec(), data: f32_bytes(data) }
}

/// The runtime arch block for a genome (JSON → ModelArch through serde so
/// every optional field takes its default).
pub fn arch_json(cfg: &EmbryoCfg) -> serde_json::Value {
    let layer_types: Vec<&str> =
        (0..cfg.layers).map(|l| if cfg.is_anchor(l) { "FullAttention" } else { "LinearAttention" }).collect();
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
        "linear_core": {"kind": "vmf_phase", "num_heads": cfg.heads, "nphase": cfg.nphase, "value_head_dim": cfg.dv},
        "linear_num_key_heads": cfg.heads,
        "linear_num_value_heads": cfg.heads,
        "linear_key_head_dim": cfg.nphase,
        "linear_value_head_dim": cfg.dv,
    });
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
    j
}

/// Write `out` from a checkpoint (+ tokenizer.json bytes).
pub fn export(ck: &Checkpoint, tokenizer_json: &[u8], out: &Path) -> anyhow::Result<()> {
    let cfg = &ck.cfg;
    let lay = Layout::new(cfg);
    let p = &ck.params;
    let h = cfg.hidden;
    let sl = |off: usize, n: usize| &p[off..off + n];
    let mut t: Vec<TensorSpec> = Vec::new();
    t.push(spec("model.embed_tokens.weight", &[cfg.vocab, h], sl(lay.embed, cfg.vocab * h)));
    t.push(spec("model.norm.weight", &[h], sl(lay.final_norm, h)));
    if cfg.head_clusters > 0 {
        t.push(spec("lm_head.clusters.weight", &[cfg.head_clusters, h], sl(lay.head_clusters, cfg.head_clusters * h)));
    }
    let desc = |name: &str| ck.extras.iter().find(|(n, _)| n == name).map(|(_, x)| x.as_slice());
    let (mu, u, bias) = (desc("desc.mu"), desc("desc.u"), desc("desc.bias"));
    let decay = crate::ops::hk_decay_grid(cfg.heads, cfg.nphase, cfg.horizon_min, cfg.horizon_max);
    // A_log: decay = exp(−exp(A_log)) → A_log = ln(−ln γ)
    let a_log: Vec<f32> = decay.iter().map(|g| (-(*g as f64).ln()).ln() as f32).collect();
    for (l, lo) in lay.layers.iter().enumerate() {
        let pf = format!("model.layers.{l}.");
        let ffn = match lo {
            LayerOffs::Mixer { ln1, wq, wk, wv, wkap, wo, ln2, ffn } => {
                t.push(spec(&format!("{pf}input_layernorm.weight"), &[h], sl(*ln1, h)));
                t.push(spec(&format!("{pf}post_attention_layernorm.weight"), &[h], sl(*ln2, h)));
                let (nh, nph, dv) = (cfg.heads, cfg.nphase, cfg.dv);
                t.push(spec(&format!("{pf}vmf_attn.thq.weight"), &[nh * nph, h], sl(*wq, nh * nph * h)));
                t.push(spec(&format!("{pf}vmf_attn.thk.weight"), &[nh * nph, h], sl(*wk, nh * nph * h)));
                t.push(spec(&format!("{pf}vmf_attn.v_proj.weight"), &[nh * dv, h], sl(*wv, nh * dv * h)));
                t.push(spec(&format!("{pf}vmf_attn.out_proj.weight"), &[h, nh * dv], sl(*wo, h * nh * dv)));
                t.push(spec(&format!("{pf}vmf_attn.A_log"), &[nh * 2 * nph], &a_log));
                // κ gate: the trainer's padded [kappa_ld, H] → real rows; fixed bias
                t.push(spec(&format!("{pf}vmf_attn.k_gate.weight"), &[nh, h], sl(*wkap, nh * h)));
                t.push(spec(&format!("{pf}vmf_attn.k_gate.bias"), &[nh], &vec![cfg.kappa_bias; nh]));
                ffn
            }
            LayerOffs::Anchor { ln1, wq, wk, wv, wo, ln2, ffn } => {
                t.push(spec(&format!("{pf}input_layernorm.weight"), &[h], sl(*ln1, h)));
                t.push(spec(&format!("{pf}post_attention_layernorm.weight"), &[h], sl(*ln2, h)));
                let (qh, kvh, hd) = (cfg.anchor_q_heads, cfg.anchor_kv_heads, cfg.anchor_hd);
                t.push(spec(&format!("{pf}self_attn.q_proj.weight"), &[qh * hd, h], sl(*wq, qh * hd * h)));
                t.push(spec(&format!("{pf}self_attn.k_proj.weight"), &[kvh * hd, h], sl(*wk, kvh * hd * h)));
                t.push(spec(&format!("{pf}self_attn.v_proj.weight"), &[kvh * hd, h], sl(*wv, kvh * hd * h)));
                t.push(spec(&format!("{pf}self_attn.o_proj.weight"), &[h, qh * hd], sl(*wo, h * qh * hd)));
                ffn
            }
        };
        let i = cfg.inter;
        if cfg.experts == 0 {
            t.push(spec(&format!("{pf}mlp.gate_proj.weight"), &[i, h], sl(ffn.wg, i * h)));
            t.push(spec(&format!("{pf}mlp.up_proj.weight"), &[i, h], sl(ffn.wu, i * h)));
            t.push(spec(&format!("{pf}mlp.down_proj.weight"), &[h, i], sl(ffn.wd, h * i)));
        } else {
            t.push(spec(&format!("{pf}mlp.shared_expert.gate_proj.weight"), &[i, h], sl(ffn.wg, i * h)));
            t.push(spec(&format!("{pf}mlp.shared_expert.up_proj.weight"), &[i, h], sl(ffn.wu, i * h)));
            t.push(spec(&format!("{pf}mlp.shared_expert.down_proj.weight"), &[h, i], sl(ffn.wd, h * i)));
            let ne = cfg.experts;
            for e in 0..ne {
                let base = ffn.experts + e * 3 * h * i;
                t.push(spec(&format!("{pf}mlp.experts.{e}.gate_proj.weight"), &[i, h], sl(base, i * h)));
                t.push(spec(&format!("{pf}mlp.experts.{e}.up_proj.weight"), &[i, h], sl(base + i * h, i * h)));
                t.push(spec(&format!("{pf}mlp.experts.{e}.down_proj.weight"), &[h, i], sl(base + 2 * i * h, h * i)));
            }
            // per-expert descriptor records (append-only growth: a new
            // expert = new tensors, nothing rewritten); no gate placeholder —
            // the loader keys the resonance MoE on experts.0 + the arch flag
            let k = crate::model::MOE_K;
            let mu_l = mu.map(|m| &m[l * ne * h..(l + 1) * ne * h]).ok_or_else(|| anyhow::anyhow!("checkpoint has no expert descriptors (desc.mu)"))?;
            for e in 0..ne {
                t.push(spec(&format!("{pf}mlp.experts.{e}.desc.mu"), &[h], &mu_l[e * h..(e + 1) * h]));
                if let Some(u) = u {
                    let base = (l * ne + e) * k * h;
                    t.push(spec(&format!("{pf}mlp.experts.{e}.desc.u"), &[k, h], &u[base..base + k * h]));
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
    let header = CmfHeader {
        format: "cmf".into(),
        version: 2,
        arch,
        quant_type: cortiq_core::types::QuantType::F32,
        provenance: Some(serde_json::json!({
            "producer": "cortiq-embryo",
            "genome": "embryo-0",
            "step": ck.step,
            "trainer_cfg": cfg,
        })),
        tokenizer_config: Some(TokenizerBundle { chat_template: None, eos_token_ids: eos.into_iter().collect(), bos_token_id: None, pad_token_id: None }),
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write(out, &header, &t, None, Some(tokenizer_json))?;
    Ok(())
}
