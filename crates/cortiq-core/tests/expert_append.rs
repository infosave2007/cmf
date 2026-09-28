//! `kind = "expert_append"` (spec §9.5.1): new MoE experts appended over a
//! frozen resonance-routed genome — the per-layer chain rule, the layout
//! the reader computes, every writer / reader refusal, the descriptor
//! value rules, and the trunk hash that growth never moves.

use cortiq_core::format::{CmfStreamWriter, TokenizerBundle, features};
use cortiq_core::knowledge::{expert_leaf, skill_kind};
use cortiq_core::mask::{MaskCatalog, MaskPriority, TaskMask};
use cortiq_core::types::{LayerType, ModelArch, MoeConfig, NormStyle, QuantType};
use cortiq_core::{
    CmfHeader, CmfModel, ExpertAppend, GenomeInfo, LineageEvent, SelectionDescriptor, SkillBound,
    SkillOverride, SkillRecord, StateEffect, TensorDtype, TensorSpec, expert_append_base,
    expert_append_layout, genome_moe_experts, hash64, moe_layers, trunk_expert_rank,
};
use std::path::{Path, PathBuf};

const H: usize = 64;
const I: usize = 128;
const L: usize = 4;
const V: usize = 10;
/// Trunk experts per MoE layer.
const E0: usize = 4;
/// Descriptor rank of the trunk.
const K: usize = 2;
/// Layer 0 is dense; the rest carry experts.
const MOE_LAYERS: [usize; 3] = [1, 2, 3];

// ───────────────────────── fixture ─────────────────────────

fn arch(moe: bool) -> ModelArch {
    ModelArch {
        arch_name: "tiny-resonance-moe".into(),
        hidden_size: H,
        intermediate_size: I,
        num_layers: L,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        vocab_size: V,
        layer_types: vec![LayerType::FullAttention; L],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: moe.then_some(MoeConfig {
            num_experts: E0,
            top_k: 1,
            moe_intermediate_size: I,
            norm_topk_prob: false,
            shared_expert_intermediate_size: None,
            router_sigmoid: false,
            routed_scaling_factor: None,
            router_resonance: true,
        }),
        qwen4_exp: None,
        deepseek_v41: None,
        anchor_core: None,
        linear_core: None,
        head_clusters: None,
        max_position_embeddings: 256,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_v_norm: false,
        qk_norm_after_rope: false,
        num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        loop_final_norm: false,
        prism_hadamard: None,
        kv_heads_per_layer: None,
        v_head_dim: None,
    }
}

fn genome_header(moe: bool) -> CmfHeader {
    let a = arch(moe);
    let mut h: CmfHeader = serde_json::from_value(serde_json::json!({
        "version": 2,
        "arch": serde_json::to_value(&a).unwrap(),
        "quant_type": "F32",
    }))
    .unwrap();
    h.arch = a;
    h.quant_type = QuantType::F32;
    h.tokenizer_config = Some(TokenizerBundle {
        chat_template: None,
        eos_token_ids: vec![2],
        bos_token_id: None,
        pad_token_id: None,
    });
    h.genome = Some(GenomeInfo::birth("embryo-moe-test", "sealed", "f32"));
    h.lineage = vec![LineageEvent::now(0, "birth", serde_json::json!({}))];
    h
}

fn f32s(n: usize, seed: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| ((i as f32 * 0.37 + seed).sin()).to_le_bytes())
        .collect()
}

fn spec(name: &str, shape: &[usize], seed: f32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::F32,
        shape: shape.to_vec(),
        data: f32s(shape.iter().product(), seed),
    }
}

fn f32_spec(name: &str, shape: &[usize], v: &[f32]) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::F32,
        shape: shape.to_vec(),
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// `with_u`: the trunk experts carry `desc.u [K, H]` (rank K) or none (rank 0).
fn trunk_specs(moe: bool, with_u: bool) -> Vec<TensorSpec> {
    let mut t = vec![spec("model.embed_tokens.weight", &[V, H], 0.1)];
    for l in 0..L {
        let pf = format!("model.layers.{l}.mlp.");
        if !moe || !MOE_LAYERS.contains(&l) {
            t.push(spec(&format!("{pf}up_proj.weight"), &[I, H], 1.0 + l as f32));
            t.push(spec(&format!("{pf}down_proj.weight"), &[H, I], 2.0 + l as f32));
            continue;
        }
        for e in 0..E0 {
            let s = 10.0 * l as f32 + e as f32;
            let pe = format!("{pf}experts.{e}.");
            t.push(spec(&format!("{pe}gate_proj.weight"), &[I, H], s + 0.1));
            t.push(spec(&format!("{pe}up_proj.weight"), &[I, H], s + 0.2));
            t.push(spec(&format!("{pe}down_proj.weight"), &[H, I], s + 0.3));
            t.push(spec(&format!("{pe}desc.mu"), &[H], s + 0.4));
            if with_u {
                t.push(spec(&format!("{pe}desc.u"), &[K, H], s + 0.5));
            }
            t.push(f32_spec(&format!("{pe}desc.bias"), &[1], &[0.0]));
        }
    }
    t.push(spec("model.norm.weight", &[H], 3.0));
    t
}

fn masks() -> MaskCatalog {
    let a = arch(true);
    let mut ffn = vec![0u8; a.ffn_mask_bytes()];
    for i in 0..I / 2 {
        ffn[i / 8] |= 1 << (i % 8);
    }
    let head = vec![0b11u8; a.head_mask_bytes()];
    MaskCatalog {
        masks: vec![TaskMask {
            task_id: 0,
            name: "general".into(),
            description: None,
            sparsity: 0.5,
            quality: None,
            ffn_masks: vec![ffn; L],
            head_masks: vec![head; L],
            layer_gates: vec![true; L],
            expert_masks: Vec::new(),
            parent: None,
            has_hot_pack: false,
            priority: MaskPriority::Fallback,
        }],
        default_task: "general".into(),
    }
}

const VOCAB: &[u8] = br#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cmf-expert-append-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// F0: a sealed resonance-MoE genome (rank K) with masks and vocab.
fn write_f0(path: &Path) {
    CmfModel::write(
        path,
        &genome_header(true),
        &trunk_specs(true, true),
        Some(&masks()),
        Some(VOCAB),
    )
    .unwrap();
}

/// The `expert_append` record `id` over `layers` with `count` experts each,
/// its tensors built from the reader's own layout plan (bias 0, shell 1.5).
fn grown(model: &CmfModel, id: &str, layers: &[usize], count: usize) -> (SkillRecord, Vec<TensorSpec>) {
    let g = model.header.genome.clone().expect("genome");
    let rank = trunk_expert_rank(&model.header, &model.tensors, layers[0]).unwrap_or(K);
    let record = SkillRecord {
        id: id.into(),
        layers: layers.to_vec(),
        kind: Some(skill_kind::EXPERT_APPEND.into()),
        experts: Some(ExpertAppend {
            count,
            shell_quantile: 0.99,
            rank,
        }),
        bound: Some(SkillBound {
            genome_id: g.id,
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash,
        }),
        status: Some("quarantine".into()),
        origin: Some(serde_json::json!({"trigger": "user_corpus", "K": count})),
        ..Default::default()
    };
    let plan = expert_append_layout(&model.header, &model.tensors, model.header.skills.len(), &record)
        .expect("layout of a valid record");
    let tensors = plan
        .iter()
        .map(|p| match p.leaf {
            expert_leaf::BIAS => f32_spec(&p.name, &p.shape, &[0.0]),
            expert_leaf::SHELL => f32_spec(&p.name, &p.shape, &[1.5]),
            _ => spec(&p.name, &p.shape, 100.0 + p.expert as f32),
        })
        .collect();
    (record, tensors)
}

fn append(path: &Path, rec: SkillRecord, ts: &[TensorSpec]) -> Result<usize, String> {
    CmfModel::append_skill(path, rec, ts, None, None, None)
        .map(|r| r.tensors_added)
        .map_err(|e| e.to_string())
}

fn open_err(p: &Path) -> String {
    match CmfModel::open(p) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{} opened but must be refused", p.display()),
    }
}

/// Publish a MUTATED header without writer-side validation (tests `open()`).
fn raw_header_swap(src: &Path, dst: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let mut bytes = std::fs::read(src).unwrap();
    let m = CmfModel::open(src).unwrap();
    let mut v = serde_json::to_value(&m.header).unwrap();
    drop(m);
    mutate(&mut v);
    let js = serde_json::to_vec(&v).unwrap();
    let off = bytes.len() as u64;
    bytes.extend_from_slice(&js);
    bytes[0x10..0x18].copy_from_slice(&off.to_le_bytes());
    bytes[0x18..0x20].copy_from_slice(&(js.len() as u64).to_le_bytes());
    bytes[0x70..0x78].copy_from_slice(&hash64(&js).to_le_bytes());
    std::fs::write(dst, &bytes).unwrap();
}

fn specs_of(m: &CmfModel) -> Vec<TensorSpec> {
    m.tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: m.entry_bytes(e).to_vec(),
        })
        .collect()
}

fn grown_name(id: &str, l: usize, e: usize, leaf: &str) -> String {
    cortiq_core::knowledge::expert_append_tensor_name(id, l, e, leaf)
}

// ───────────────────────── kinds ─────────────────────────

#[test]
fn expert_append_is_implemented_not_reserved() {
    assert!(skill_kind::IMPLEMENTED.contains(&skill_kind::EXPERT_APPEND));
    assert!(!skill_kind::RESERVED.contains(&"expert_append"));
    assert_eq!(skill_kind::EXPERT_APPEND, "expert_append");
}

// ───────────────────────── valid records ─────────────────────────

/// All MoE layers, then a subset on top: the writer fills `moe_experts`,
/// computes the state effect, the chain gives every record its per-layer
/// indices, the trunk never moves.
#[test]
fn expert_append_grows_all_layers_and_a_subset_over_an_unchanged_trunk() {
    let dir = tempdir("grow");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let m0 = CmfModel::open(&f0).unwrap();
    let g0 = m0.header.genome.clone().unwrap();
    assert_eq!(g0.moe_experts, Some(E0), "the writer filled E0 from the arch");
    assert_eq!(genome_moe_experts(&m0.header).unwrap(), E0);
    assert_eq!(moe_layers(&m0.tensors), MOE_LAYERS.to_vec());
    for l in MOE_LAYERS {
        assert_eq!(trunk_expert_rank(&m0.header, &m0.tensors, l).unwrap(), K);
        assert_eq!(expert_append_base(&m0.header, 0, l).unwrap(), E0);
    }
    assert!(trunk_expert_rank(&m0.header, &m0.tensors, 0)
        .unwrap_err()
        .to_string()
        .contains("layer 0 is not an MoE layer"));
    let b0 = std::fs::read(&f0).unwrap();
    let h0 = m0.trunk_hash();
    let n0 = m0.tensors.len();
    drop(m0);

    // F1 = F0 + "herbs": 2 experts in every MoE layer.
    let f1 = dir.join("f1.cmf");
    std::fs::copy(&f0, &f1).unwrap();
    let m = CmfModel::open(&f1).unwrap();
    let (rec, ts) = grown(&m, "herbs", &MOE_LAYERS, 2);
    drop(m);
    assert_eq!(ts.len(), MOE_LAYERS.len() * 2 * expert_leaf::ALL.len());
    assert_eq!(append(&f1, rec, &ts).unwrap(), ts.len());

    let m1 = CmfModel::open(&f1).unwrap();
    let b1 = std::fs::read(&f1).unwrap();
    assert_eq!(&b1[128..b0.len()], &b0[128..], "true tail append");
    assert_eq!(m1.trunk_hash(), h0, "G1: growth never touches the trunk hash");
    assert_eq!(m1.header.genome, Some(g0.clone()));
    assert_eq!(m1.tensors.len(), n0 + ts.len());
    assert!(m1.verify().is_empty(), "{:?}", m1.verify());
    assert_ne!(m1.required_features & features::SKILLS_V2, 0);
    let r = &m1.header.skills[0];
    assert_eq!(r.kind.as_deref(), Some("expert_append"));
    assert_eq!(
        r.state_effect,
        Some(StateEffect {
            first_affected_layer: 1,
            switch: "sequence_start".into(),
            state_bytes_added: 0,
        }),
        "the writer computed the state effect"
    );
    assert!(!r.is_auto_routable(), "grown experts are not a route target");
    // Expert indices: E0 + k in every grown layer.
    for l in MOE_LAYERS {
        for k in 0..2 {
            for leaf in expert_leaf::ALL {
                let t = m1
                    .tensor(&grown_name("herbs", l, E0 + k, leaf))
                    .unwrap_or_else(|| panic!("layer {l} expert {} {leaf}", E0 + k));
                let trunk = m1.tensor(&cortiq_core::knowledge::expert_tensor_name(l, 0, leaf));
                match *leaf {
                    expert_leaf::BIAS | expert_leaf::SHELL => assert_eq!(t.shape, vec![1]),
                    expert_leaf::MU => assert_eq!(t.shape, vec![H]),
                    _ => assert_eq!(t.shape, trunk.unwrap().shape, "{leaf} like the trunk's expert 0"),
                }
            }
        }
        assert!(m1.tensor(&grown_name("herbs", l, E0 + 2, expert_leaf::MU)).is_none());
    }
    // The reader's plan for the mounted record names exactly its tensors.
    let plan = expert_append_layout(&m1.header, &m1.tensors, 0, &m1.header.skills[0]).unwrap();
    let mut planned: Vec<&str> = plan.iter().map(|p| p.name.as_str()).collect();
    let mut actual: Vec<&str> = m1
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("skill.herbs."))
        .map(|t| t.name.as_str())
        .collect();
    planned.sort_unstable();
    actual.sort_unstable();
    assert_eq!(planned, actual);
    drop(m1);

    // F2 = F1 + "spices": one expert in layer 3 only → index E0 + 2 there;
    // layers 1 and 2 keep E0 + 2 experts.
    let f2 = dir.join("f2.cmf");
    std::fs::copy(&f1, &f2).unwrap();
    let m = CmfModel::open(&f2).unwrap();
    assert_eq!(expert_append_base(&m.header, 1, 3).unwrap(), E0 + 2);
    assert_eq!(expert_append_base(&m.header, 1, 1).unwrap(), E0 + 2);
    let (rec, ts) = grown(&m, "spices", &[3], 1);
    drop(m);
    assert_eq!(ts.len(), expert_leaf::ALL.len());
    assert!(ts.iter().all(|t| t.name.starts_with("skill.spices.model.layers.3.mlp.experts.6.")));
    append(&f2, rec, &ts).unwrap();
    let m2 = CmfModel::open(&f2).unwrap();
    assert_eq!(m2.trunk_hash(), h0);
    assert_eq!(&std::fs::read(&f2).unwrap()[128..b1.len()], &b1[128..]);
    assert_eq!(m2.header.skills.len(), 2);
    assert_eq!(
        m2.header.skills[1].state_effect.as_ref().unwrap().first_affected_layer,
        3
    );
    // A third record would start after both in layer 3, after herbs elsewhere.
    let at = m2.header.skills.len();
    assert_eq!(expert_append_base(&m2.header, at, 3).unwrap(), E0 + 3);
    assert_eq!(expert_append_base(&m2.header, at, 2).unwrap(), E0 + 2);
    assert_eq!(m2.header.lineage.last().unwrap().event, "skill_committed");
    assert!(m2.verify().is_empty());
    drop(m2);

    // Status moves (quarantine → active) do not touch the chain.
    CmfModel::update_header_append(&f2, |h| {
        h.skills[0].status = Some("active".into());
        h.skills[1].status = Some("stale_regate".into());
    })
    .unwrap();
    let m2 = CmfModel::open(&f2).unwrap();
    assert_eq!(expert_append_base(&m2.header, 2, 3).unwrap(), E0 + 3);
    assert_eq!(m2.trunk_hash(), h0);
    std::fs::remove_dir_all(dir).ok();
}

/// A trunk without `desc.u` grows rank-0 experts: no `desc.u` in the record.
#[test]
fn rank_zero_trunk_grows_experts_without_desc_u() {
    let dir = tempdir("rank0");
    let f0 = dir.join("f0.cmf");
    CmfModel::write(
        &f0,
        &genome_header(true),
        &trunk_specs(true, false),
        None,
        Some(VOCAB),
    )
    .unwrap();
    let m = CmfModel::open(&f0).unwrap();
    assert_eq!(trunk_expert_rank(&m.header, &m.tensors, 2).unwrap(), 0);
    let (rec, ts) = grown(&m, "r0", &[2], 1);
    drop(m);
    assert_eq!(rec.experts.as_ref().unwrap().rank, 0);
    assert_eq!(ts.len(), expert_leaf::ALL.len() - 1);
    assert!(ts.iter().all(|t| !t.name.ends_with(".desc.u")));
    // A desc.u the trunk does not have is refused…
    let mut extra = ts.clone();
    extra.push(spec(&grown_name("r0", 2, E0, expert_leaf::U), &[K, H], 1.0));
    let e = append(&f0, rec.clone(), &extra).unwrap_err();
    assert!(e.contains("is not part of the expert_append layout"), "{e}");
    // …and rank 2 against a rank-0 trunk too.
    let mut r2 = rec.clone();
    r2.experts.as_mut().unwrap().rank = K;
    let e = append(&f0, r2, &ts).unwrap_err();
    assert!(e.contains("experts.rank 2 != the trunk expert 0's desc.u rank 0 at layer 2"), "{e}");
    append(&f0, rec, &ts).unwrap();
    let m = CmfModel::open(&f0).unwrap();
    assert_eq!(m.header.skills[0].experts.as_ref().unwrap().rank, 0);
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── writer refusals ─────────────────────────

#[test]
fn every_writer_refusal_leaves_the_file_alone() {
    let dir = tempdir("refuse");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let bytes0 = std::fs::read(&f0).unwrap();
    let m = CmfModel::open(&f0).unwrap();
    let (rec, ts) = grown(&m, "x", &[1], 1);
    let (rec12, ts12) = grown(&m, "x", &[1, 2], 1);
    drop(m);
    let bias_at = ts
        .iter()
        .position(|t| t.name.ends_with(".desc.bias"))
        .unwrap();
    let shell_at = ts
        .iter()
        .position(|t| t.name.ends_with(".desc.shell"))
        .unwrap();
    let gate_at = ts
        .iter()
        .position(|t| t.name.ends_with(".gate_proj.weight"))
        .unwrap();

    type Mut = Box<dyn Fn(&mut SkillRecord, &mut Vec<TensorSpec>)>;
    let cases: Vec<(&str, Mut, &str)> = vec![
        (
            "no experts",
            Box::new(|r, _| r.experts = None),
            "expert_append record has no `experts`",
        ),
        (
            "overrides",
            Box::new(|r, _| {
                r.overrides.push(SkillOverride {
                    name: "model.layers.1.mlp.experts.0.up_proj.weight".into(),
                    base_hash: "0".into(),
                })
            }),
            "`overrides` must be empty",
        ),
        (
            "count 0",
            Box::new(|r, _| r.experts.as_mut().unwrap().count = 0),
            "experts.count must be ≥ 1",
        ),
        (
            "count 2 with tensors for 1",
            Box::new(|r, _| r.experts.as_mut().unwrap().count = 2),
            "missing tensor 'skill.x.model.layers.1.mlp.experts.5.gate_proj.weight' (layer 1, expert 5)",
        ),
        (
            "shell quantile",
            Box::new(|r, _| r.experts.as_mut().unwrap().shell_quantile = 1.5),
            "experts.shell_quantile 1.5 must be finite and in [0, 1]",
        ),
        (
            "rank",
            Box::new(|r, _| r.experts.as_mut().unwrap().rank = 1),
            "experts.rank 1 != the trunk expert 0's desc.u rank 2 at layer 1",
        ),
        (
            "dense layer",
            Box::new(|r, _| r.layers = vec![0]),
            "layer 0 is not an MoE layer of the trunk",
        ),
        (
            "layer out of range",
            Box::new(|r, _| r.layers = vec![9]),
            "layer 9 of 4 layers",
        ),
        (
            "no layers",
            Box::new(|r, _| r.layers = vec![]),
            "grows no layers",
        ),
        (
            "duplicate layer",
            Box::new(|r, _| r.layers = vec![1, 1]),
            "duplicate layer in [1, 1]",
        ),
        (
            "missing tensor",
            Box::new(|_, t| {
                t.pop();
            }),
            "missing tensor 'skill.x.model.layers.1.mlp.experts.4.desc.shell' (layer 1, expert 4)",
        ),
        (
            "extra tensor",
            Box::new(|_, t| t.push(spec(&grown_name("x", 1, 9, expert_leaf::MU), &[H], 1.0))),
            "tensor 'skill.x.model.layers.1.mlp.experts.9.desc.mu' is not part of the \
             expert_append layout (layers [1], count 1)",
        ),
        (
            "tensors of an undeclared layer",
            Box::new(|_, t| {
                t.push(spec(&grown_name("x", 2, 4, expert_leaf::MU), &[H], 1.0));
            }),
            "tensor 'skill.x.model.layers.2.mlp.experts.4.desc.mu' is not part of the \
             expert_append layout",
        ),
        (
            "gate shape",
            Box::new(move |_, t| t[gate_at].shape = vec![H, I]),
            "'skill.x.model.layers.1.mlp.experts.4.gate_proj.weight' shape [64, 128] != \
             [128, 64] (the trunk's expert 0 of layer 1)",
        ),
        (
            "bias NaN",
            Box::new(move |_, t| t[bias_at].data = f32::NAN.to_le_bytes().to_vec()),
            "'skill.x.model.layers.1.mlp.experts.4.desc.bias' is NaN — a grown expert's \
             desc.bias must be a finite f32",
        ),
        (
            "bias -inf",
            Box::new(move |_, t| t[bias_at].data = f32::NEG_INFINITY.to_le_bytes().to_vec()),
            "'skill.x.model.layers.1.mlp.experts.4.desc.bias' is -inf — a grown expert's \
             desc.bias must be a finite f32",
        ),
        (
            "shell NaN",
            Box::new(move |_, t| t[shell_at].data = f32::NAN.to_le_bytes().to_vec()),
            "desc.shell' is NaN — desc.shell must be a finite f32 threshold",
        ),
        (
            "shell +inf",
            Box::new(move |_, t| t[shell_at].data = f32::INFINITY.to_le_bytes().to_vec()),
            "desc.shell' is inf — desc.shell must be a finite f32 threshold",
        ),
        (
            "bias not f32",
            Box::new(move |_, t| {
                t[bias_at].dtype = TensorDtype::F16;
                t[bias_at].data = vec![0, 0];
            }),
            "'skill.x.model.layers.1.mlp.experts.4.desc.bias' must be f32 (is F16)",
        ),
        (
            "selection",
            Box::new(|r, _| {
                r.selection = Some(SelectionDescriptor {
                    metric: "mse_unit".into(),
                    phi_layer: 0,
                    mean: String::new(),
                    basis: String::new(),
                    rank: 0,
                    err_mean: None,
                    err_std: None,
                    holdout: None,
                    holdout_n: None,
                })
            }),
            "carries no `selection`",
        ),
        (
            "transparent switch",
            Box::new(|r, _| {
                r.state_effect = Some(StateEffect {
                    first_affected_layer: 1,
                    switch: "transparent".into(),
                    state_bytes_added: 0,
                })
            }),
            "state_effect.switch 'transparent' — grown experts are part of the organism, the \
             record's switch is 'sequence_start'",
        ),
        (
            "first affected layer",
            Box::new(|r, _| {
                r.state_effect = Some(StateEffect {
                    first_affected_layer: 2,
                    switch: "sequence_start".into(),
                    state_bytes_added: 0,
                })
            }),
            "state_effect.first_affected_layer 2 != min(layers) 1",
        ),
        (
            "state bytes",
            Box::new(|r, _| {
                r.state_effect = Some(StateEffect {
                    first_affected_layer: 1,
                    switch: "sequence_start".into(),
                    state_bytes_added: 8,
                })
            }),
            "state_effect.state_bytes_added 8 — an expert_append record adds no per-sequence state",
        ),
        (
            "wrong genome",
            Box::new(|r, _| r.bound.as_mut().unwrap().genome_id = "other".into()),
            "bound.genome_id 'other' != genome.id 'embryo-moe-test'",
        ),
        (
            "unknown status",
            Box::new(|r, _| r.status = Some("grown".into())),
            "status 'grown' is not one of",
        ),
    ];
    for (name, mutate, needle) in cases {
        let (mut r, mut t) = (rec.clone(), ts.clone());
        mutate(&mut r, &mut t);
        let e = append(&f0, r, &t).unwrap_err();
        assert!(e.contains(needle), "case '{name}': refusal was '{e}', expected '{needle}'");
    }
    // Layers declared for two layers, tensors for one.
    let mut r = rec12.clone();
    r.layers = vec![1, 2];
    let e = append(&f0, r, &ts).unwrap_err();
    assert!(
        e.contains("missing tensor 'skill.x.model.layers.2.mlp.experts.4.gate_proj.weight'"),
        "{e}"
    );
    // Tensors for two layers, one declared.
    let mut r = rec12;
    r.layers = vec![1];
    let e = append(&f0, r, &ts12).unwrap_err();
    assert!(e.contains("skill.x.model.layers.2.mlp.experts.4.gate_proj.weight' is not part"), "{e}");
    assert_eq!(std::fs::read(&f0).unwrap(), bytes0, "refused appends write nothing");
    assert!(CmfModel::open(&f0).unwrap().header.skills.is_empty());

    // A dense genome (no MoE) and a gated MoE genome take no expert_append.
    let dense = dir.join("dense.cmf");
    CmfModel::write(&dense, &genome_header(false), &trunk_specs(false, false), None, Some(VOCAB))
        .unwrap();
    let bound_to = |p: &Path, mut r: SkillRecord| {
        let g = CmfModel::open(p).unwrap().header.genome.clone().unwrap();
        r.bound = Some(SkillBound {
            genome_id: g.id,
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash,
        });
        r
    };
    let m = CmfModel::open(&dense).unwrap();
    assert_eq!(m.header.genome.as_ref().unwrap().moe_experts, None);
    assert!(genome_moe_experts(&m.header).is_err());
    drop(m);
    let e = append(&dense, bound_to(&dense, rec.clone()), &[]).unwrap_err();
    assert!(e.contains("needs a resonance-routed MoE trunk"), "{e}");
    let mut gated = genome_header(true);
    gated.arch.moe.as_mut().unwrap().router_resonance = false;
    let gp = dir.join("gated.cmf");
    CmfModel::write(&gp, &gated, &trunk_specs(true, true), None, Some(VOCAB)).unwrap();
    assert_eq!(CmfModel::open(&gp).unwrap().header.genome.as_ref().unwrap().moe_experts, None);
    let e = append(&gp, bound_to(&gp, rec), &ts).unwrap_err();
    assert!(e.contains("needs a resonance-routed MoE trunk"), "{e}");
    std::fs::remove_dir_all(dir).ok();
}

/// Full rewrites (`write`, the streaming writer) enforce the same rules on
/// the payload they write.
#[test]
fn full_rewrites_check_descriptor_values_too() {
    let dir = tempdir("rewrite");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let m = CmfModel::open(&f0).unwrap();
    let (rec, ts) = grown(&m, "herbs", &[2], 1);
    drop(m);
    append(&f0, rec, &ts).unwrap();
    let m1 = CmfModel::open(&f0).unwrap();
    let specs = specs_of(&m1);
    let header = m1.header.clone();
    let masks = m1.masks.clone();
    let h1 = m1.trunk_hash();
    drop(m1);

    // Compaction of a grown file keeps everything.
    let c = dir.join("c.cmf");
    CmfModel::write(&c, &header, &specs, Some(&masks), Some(VOCAB)).unwrap();
    let mc = CmfModel::open(&c).unwrap();
    assert_eq!(mc.trunk_hash(), h1);
    assert_eq!(mc.header.skills[0].kind.as_deref(), Some("expert_append"));
    drop(mc);

    let bias = specs
        .iter()
        .position(|t| t.name.ends_with(".desc.bias") && t.name.starts_with("skill."))
        .unwrap();
    let mut bad = specs.clone();
    bad[bias].data = f32::INFINITY.to_le_bytes().to_vec();
    let e = CmfModel::write(dir.join("bad.cmf"), &header, &bad, Some(&masks), Some(VOCAB))
        .unwrap_err()
        .to_string();
    assert!(e.contains("desc.bias must be a finite f32"), "{e}");
    // A finite non-zero bias (the source's balancing bias, `bias_mode =
    // source`) is accepted by every writer and by open().
    let mut src_bias = specs.clone();
    src_bias[bias].data = (-0.25f32).to_le_bytes().to_vec();
    let sb = dir.join("src-bias.cmf");
    CmfModel::write(&sb, &header, &src_bias, Some(&masks), Some(VOCAB)).unwrap();
    let msb = CmfModel::open(&sb).unwrap();
    assert_eq!(msb.trunk_hash(), h1);
    let t = msb.tensor(&src_bias[bias].name).unwrap();
    let off = msb.entry_abs_offset(t).unwrap();
    let raw = std::fs::read(&sb).unwrap();
    assert_eq!(f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]), -0.25);
    drop(msb);

    let sp = dir.join("stream.cmf");
    let mut w = CmfStreamWriter::new(&sp, CmfStreamWriter::head_reserve_for(specs.len(), 64)).unwrap();
    for t in &bad {
        w.push(&t.name, t.dtype, &t.shape, &t.data).unwrap();
    }
    let e = w.finish(&header, Some(&masks), Some(VOCAB)).unwrap_err().to_string();
    assert!(e.contains("desc.bias must be a finite f32"), "stream: {e}");
    let mut w = CmfStreamWriter::new(&sp, CmfStreamWriter::head_reserve_for(specs.len(), 64)).unwrap();
    for t in &specs {
        w.push(&t.name, t.dtype, &t.shape, &t.data).unwrap();
    }
    w.finish(&header, Some(&masks), Some(VOCAB)).unwrap();
    assert_eq!(CmfModel::open(&sp).unwrap().trunk_hash(), h1);
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── reader refusals ─────────────────────────

#[test]
fn open_refuses_a_broken_record_and_tampered_descriptor_values() {
    let dir = tempdir("open");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let m = CmfModel::open(&f0).unwrap();
    let (rec, ts) = grown(&m, "herbs", &[1, 3], 2);
    drop(m);
    append(&f0, rec, &ts).unwrap();
    let p = dir.join("bad.cmf");

    type Mutation = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(&str, Mutation, &str)> = vec![
        (
            "moe_experts drifted",
            Box::new(|v| v["genome"]["moe_experts"] = 5.into()),
            "genome 'embryo-moe-test': moe_experts 5 != arch.moe.num_experts 4",
        ),
        (
            "experts removed",
            Box::new(|v| {
                v["skills"][0].as_object_mut().unwrap().remove("experts");
            }),
            "expert_append record has no `experts`",
        ),
        (
            "count grew",
            Box::new(|v| v["skills"][0]["experts"]["count"] = 3.into()),
            "missing tensor 'skill.herbs.model.layers.1.mlp.experts.6.gate_proj.weight'",
        ),
        (
            "count shrank",
            Box::new(|v| v["skills"][0]["experts"]["count"] = 1.into()),
            "tensor 'skill.herbs.model.layers.1.mlp.experts.5.gate_proj.weight' is not part of \
             the expert_append layout (layers [1, 3], count 1)",
        ),
        (
            "layer dropped",
            Box::new(|v| v["skills"][0]["layers"] = serde_json::json!([1])),
            "tensor 'skill.herbs.model.layers.3.mlp.experts.4.gate_proj.weight' is not part",
        ),
        (
            "layer added",
            Box::new(|v| v["skills"][0]["layers"] = serde_json::json!([1, 2, 3])),
            "missing tensor 'skill.herbs.model.layers.2.mlp.experts.4.gate_proj.weight'",
        ),
        (
            "switch relaxed",
            Box::new(|v| v["skills"][0]["state_effect"]["switch"] = "transparent".into()),
            "the record's switch is 'sequence_start'",
        ),
        (
            "experts on a v1 record",
            Box::new(|v| {
                let legacy = serde_json::json!({"id": "legacy", "layers": [1],
                    "experts": {"count": 1, "shell_quantile": 0.5, "rank": 2}});
                v["skills"].as_array_mut().unwrap().push(legacy);
            }),
            "v2 fields (overrides/bound/state_effect/status/gate/prompt_contract/origin/experts/\
             lookup) on a record without `kind`",
        ),
    ];
    for (name, mutate, needle) in cases {
        raw_header_swap(&f0, &p, mutate);
        let e = open_err(&p);
        assert!(e.contains(needle), "case '{name}': refusal was '{e}', expected '{needle}'");
    }

    // Skill payloads are outside the trunk hash: the value check catches a
    // bias / shell changed on disk.
    let m = CmfModel::open(&f0).unwrap();
    let bias = m
        .tensor(&grown_name("herbs", 3, E0 + 1, expert_leaf::BIAS))
        .unwrap();
    let shell = m
        .tensor(&grown_name("herbs", 1, E0, expert_leaf::SHELL))
        .unwrap();
    let (ba, sa) = (
        m.entry_abs_offset(bias).unwrap(),
        m.entry_abs_offset(shell).unwrap(),
    );
    drop(m);
    let bytes = std::fs::read(&f0).unwrap();
    let mut t = bytes.clone();
    t[ba..ba + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(&p, &t).unwrap();
    let e = open_err(&p);
    assert!(e.contains("experts.5.desc.bias' is NaN — a grown expert's desc.bias must be a finite f32"), "{e}");
    // a finite bias (the source's, `bias_mode = source`) is a payload
    // value the format accepts: open() succeeds
    let mut t = bytes.clone();
    t[ba..ba + 4].copy_from_slice(&(-0.25f32).to_le_bytes());
    std::fs::write(&p, &t).unwrap();
    CmfModel::open(&p).unwrap();
    let mut t = bytes.clone();
    t[sa..sa + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(&p, &t).unwrap();
    let e = open_err(&p);
    assert!(e.contains("experts.4.desc.shell' is NaN"), "{e}");
    // A shell value change that stays finite is a legitimate payload edit
    // for `verify` to flag, never a refusal at open.
    let mut t = bytes.clone();
    t[sa..sa + 4].copy_from_slice(&2.5f32.to_le_bytes());
    std::fs::write(&p, &t).unwrap();
    let m = CmfModel::open(&p).unwrap();
    assert!(m.verify().iter().any(|s| s.contains("desc.shell")), "{:?}", m.verify());
    std::fs::remove_dir_all(dir).ok();
}

/// The chain is retired from its tail: a live record behind a retired one
/// in the same layer is refused (header update and open), the last record
/// of a layer may go, then the one before it; a new record cannot grow a
/// layer whose chain ends in a retired record.
#[test]
fn retiring_is_allowed_only_from_the_tail_of_a_layers_chain() {
    let dir = tempdir("retire");
    let f = dir.join("f.cmf");
    write_f0(&f);
    let m = CmfModel::open(&f).unwrap();
    let (r1, t1) = grown(&m, "a", &[1, 3], 1);
    drop(m);
    append(&f, r1, &t1).unwrap();
    let m = CmfModel::open(&f).unwrap();
    let (r2, t2) = grown(&m, "b", &[3], 1);
    let (r3, t3) = grown(&m, "c", &[2], 1);
    drop(m);
    append(&f, r2, &t2).unwrap();
    append(&f, r3, &t3).unwrap();
    let len = std::fs::metadata(&f).unwrap().len();

    // 'a' is behind 'b' in layer 3: cannot retire yet.
    let e = CmfModel::update_header_append(&f, |h| h.skills[0].status = Some("retired".into()))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("skill 'b': expert_append chain at layer 3 is broken — the earlier record \
                    'a' is retired while this one is live"),
        "{e}"
    );
    assert_eq!(std::fs::metadata(&f).unwrap().len(), len, "refusal wrote nothing");
    let p = dir.join("swap.cmf");
    raw_header_swap(&f, &p, |v| v["skills"][0]["status"] = "retired".into());
    assert!(open_err(&p).contains("chain at layer 3 is broken"));

    // 'c' (layer 2, alone) and 'b' (tail of layer 3) may retire; then 'a'.
    CmfModel::update_header_append(&f, |h| h.skills[2].status = Some("retired".into())).unwrap();
    CmfModel::update_header_append(&f, |h| h.skills[1].status = Some("retired".into())).unwrap();
    let m = CmfModel::open(&f).unwrap();
    // Planning after retired records: refused per layer.
    let e = expert_append_base(&m.header, 3, 3).unwrap_err().to_string();
    assert!(e.contains("record 'b' before position 3 is retired"), "{e}");
    assert_eq!(expert_append_base(&m.header, 3, 1).unwrap(), E0 + 1, "layer 1 chain is live");
    let (rn, tn) = grown(&m, "n", &[1], 1);
    drop(m);
    append(&f, rn.clone(), &tn).unwrap();
    // 'n' now sits behind 'a' in layer 1: 'a' cannot retire before 'n'.
    let e = CmfModel::update_header_append(&f, |h| h.skills[0].status = Some("retired".into()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("skill 'n': expert_append chain at layer 1 is broken"), "{e}");
    CmfModel::update_header_append(&f, |h| h.skills[3].status = Some("retired".into())).unwrap();
    CmfModel::update_header_append(&f, |h| h.skills[0].status = Some("retired".into())).unwrap();
    let m = CmfModel::open(&f).unwrap();
    assert!(m.header.skills.iter().all(|s| s.status.as_deref() == Some("retired")));
    assert!(m.verify().is_empty());
    // Every layer's chain now ends in a retired record: no new growth there.
    for l in [1, 2, 3] {
        assert!(expert_append_base(&m.header, 4, l).is_err(), "layer {l}");
    }
    let g = m.header.genome.clone().unwrap();
    drop(m);
    let again = SkillRecord {
        id: "z".into(),
        bound: Some(SkillBound {
            genome_id: g.id,
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash,
        }),
        ..rn
    };
    let e = append(&f, again, &[]).unwrap_err();
    assert!(
        e.contains("skill 'z': expert_append chain at layer 1 is broken — the earlier record \
                    'a' is retired while this one is live"),
        "{e}"
    );
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── trunk hash ─────────────────────────

/// Growth never changes the trunk hash; the mask catalog does (NF-1).
#[test]
fn trunk_hash_ignores_growth_and_covers_masks() {
    let dir = tempdir("hash");
    let with = dir.join("with.cmf");
    let without = dir.join("without.cmf");
    write_f0(&with);
    CmfModel::write(
        &without,
        &genome_header(true),
        &trunk_specs(true, true),
        None,
        Some(VOCAB),
    )
    .unwrap();
    let (hw, hn) = (
        CmfModel::open(&with).unwrap().trunk_hash(),
        CmfModel::open(&without).unwrap().trunk_hash(),
    );
    assert_ne!(hw, hn, "masks are part of the trunk's execution");
    for (p, h) in [(&with, hw), (&without, hn)] {
        let m = CmfModel::open(p).unwrap();
        let (rec, ts) = grown(&m, "herbs", &MOE_LAYERS, 1);
        drop(m);
        append(p, rec, &ts).unwrap();
        let m = CmfModel::open(p).unwrap();
        assert_eq!(m.trunk_hash(), h, "{}", p.display());
        assert_eq!(m.header.genome.as_ref().unwrap().trunk_hash, cortiq_core::knowledge::hex64(h));
        assert_eq!(
            m.trunk_hash(),
            cortiq_core::trunk_hash_exec(&m.header, &m.tensors, Some(VOCAB), m.exec_hashes())
        );
        // The mask section changed under the grown file: refused.
        let bytes = std::fs::read(p).unwrap();
        let u64at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let (off, len) = (u64at(0x40) as usize, u64at(0x48) as usize);
        if len > 0 {
            let mut t = bytes.clone();
            t[off + len - 1] ^= 0x01;
            let q = dir.join("tamper.cmf");
            std::fs::write(&q, &t).unwrap();
            assert!(open_err(&q).contains("trunk_hash mismatch"));
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// NF-1 on the header-update path: a file WITH a mask catalog and WITHOUT
/// a genome receives its genome through `update_header_append`. The
/// appended header is sealed with the hash `open()` recomputes — mask
/// section included. (The writer took `model.exec`, which `open()` fills
/// only for GENOME files: the genome was sealed WITHOUT its masks and the
/// commit's re-open refused it with "trunk_hash mismatch" and rolled the
/// append back; a mask-free file passed — a rule that depended on the
/// presence of a section, not on NF-1.)
#[test]
fn update_header_append_births_a_genome_over_a_mask_catalog() {
    let dir = tempdir("birth");
    for (tag, catalog) in [("masked", Some(masks())), ("bare", None)] {
        let p = dir.join(format!("{tag}.cmf"));
        let mut h = genome_header(true);
        h.genome = None;
        h.lineage.clear();
        CmfModel::write(&p, &h, &trunk_specs(true, true), catalog.as_ref(), Some(VOCAB))
            .unwrap_or_else(|e| panic!("{tag}: write: {e}"));
        let before = CmfModel::open(&p).unwrap();
        assert!(before.header.genome.is_none());
        assert_eq!(before.exec_hashes().masks.is_some(), catalog.is_some(), "{tag}");
        let exec_before = before.exec_hashes();
        drop(before);

        let report = CmfModel::update_header_append(&p, |h| {
            h.genome = Some(GenomeInfo::birth("born-by-update", "sealed", "f32"));
            h.lineage.push(LineageEvent::now(
                0,
                "birth",
                serde_json::json!({"how": "update_header_append"}),
            ));
        })
        .unwrap_or_else(|e| panic!("{tag}: the genome birth was refused: {e}"));
        assert_eq!(report.tensors_added, 0);

        let m = CmfModel::open(&p).unwrap_or_else(|e| panic!("{tag}: re-open: {e}"));
        let g = m.header.genome.as_ref().expect("genome born");
        assert_eq!(g.id, "born-by-update");
        assert_eq!(g.moe_experts, Some(E0), "E0 filled at the seal");
        assert_eq!(m.exec_hashes(), exec_before, "{tag}: the sections did not move");
        let sealed = cortiq_core::knowledge::parse_hex64(&g.trunk_hash).unwrap();
        let want =
            cortiq_core::trunk_hash_exec(&m.header, &m.tensors, Some(VOCAB), m.exec_hashes());
        assert_eq!(sealed, want, "{tag}: sealed with the hash open() recomputes");
        assert_eq!(m.trunk_hash(), want);
        assert_eq!(report.trunk_hash, Some(want));
        assert_eq!(g.master_trunk_hash, g.trunk_hash, "f32 encoding: master = trunk");
        let no_exec = cortiq_core::trunk_hash(&m.header, &m.tensors, Some(VOCAB));
        if catalog.is_some() {
            assert_ne!(want, no_exec, "the mask section is inside the sealed hash");
        } else {
            assert_eq!(want, no_exec, "no sections: the pre-NF-1 hash");
        }
        // The born genome behaves as one: growth appends over it and the
        // trunk hash stays put.
        let (rec, ts) = grown(&m, "herbs", &MOE_LAYERS, 1);
        drop(m);
        append(&p, rec, &ts).unwrap_or_else(|e| panic!("{tag}: growth over the born genome: {e}"));
        let m = CmfModel::open(&p).unwrap();
        assert_eq!(m.trunk_hash(), want, "{tag}");
        assert_eq!(m.header.skills.len(), 1);
    }
    std::fs::remove_dir_all(dir).ok();
}
