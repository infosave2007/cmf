//! Per-visit FFN masks for Looped Transformers — the mask area's loop
//! layout, pinned at the codec level.
//!
//! The runtime indexes task masks by the VIRTUAL layer (`kv_cache` and
//! the FFN mask check both walk `li` over physical × loops), but the
//! mask area used to store one FFN row per PHYSICAL layer. On a looped
//! model that left every row past the first pass missing, and
//! `ffn_active_count` answers 0 for a missing row — so the sparse path
//! silently zeroed the entire second pass's FFN. These tests hold the
//! repaired contract from both directions.

use cortiq_core::mask::{decode_masks_section, encode_masks_section, MaskCatalog, TaskMask};
use cortiq_core::types::{LayerType, ModelArch, NormStyle};

fn arch(loops: usize) -> ModelArch {
    ModelArch {
        arch_name: "tiny-looped".into(),
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 2,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 10,
        layer_types: vec![LayerType::FullAttention; 2],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: 64,
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
        num_loops: loops,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        loop_final_norm: loops > 1,
    }
}

fn mask_with_ffn(rows: Vec<Vec<u8>>) -> TaskMask {
    TaskMask {
        task_id: 1,
        name: "t".into(),
        description: None,
        sparsity: 0.0,
        quality: None,
        ffn_masks: rows,
        head_masks: vec![vec![0b11]; 2],
        layer_gates: vec![true; 2],
        expert_masks: vec![],
        parent: None,
        has_hot_pack: false,
        priority: cortiq_core::mask::MaskPriority::Normal,
    }
}

/// Distinct per-visit rows survive the round trip exactly: what pass 0
/// masks and what pass 1 masks are different sets, and they stay so.
#[test]
fn per_visit_rows_round_trip() {
    let a = arch(2);
    // 2 physical layers × 2 loops = 4 rows, all different.
    let rows = vec![
        vec![0b1111_0000, 0xFF],
        vec![0b0000_1111, 0xFF],
        vec![0xFF, 0b1111_0000],
        vec![0xFF, 0b0000_1111],
    ];
    let cat = MaskCatalog {
        masks: vec![mask_with_ffn(rows.clone())],
        default_task: String::new(),
    };
    let bytes = encode_masks_section(&cat, &a).unwrap();
    let back = decode_masks_section(&bytes, &a).unwrap();
    assert_eq!(back.masks[0].ffn_masks, rows, "per-visit rows changed in flight");
    // And the virtual-layer indexing the runtime does is now in range
    // for every visit.
    for vl in 0..4 {
        assert!(
            back.masks[0].ffn_active_count(vl) > 0,
            "visit row {vl} reads as empty — the second pass would lose its FFN"
        );
    }
}

/// A mask authored per PHYSICAL layer (every pre-loop writer) reaches a
/// looped runtime replicated to every pass — the semantics its author
/// had, not a zeroed second pass.
#[test]
fn physical_rows_replicate_to_every_pass() {
    let a = arch(2);
    let phys = vec![vec![0b1010_1010, 0x0F], vec![0b0101_0101, 0xF0]];
    let cat = MaskCatalog {
        masks: vec![mask_with_ffn(phys.clone())],
        default_task: String::new(),
    };
    let bytes = encode_masks_section(&cat, &a).unwrap();
    let back = decode_masks_section(&bytes, &a).unwrap();
    let m = &back.masks[0];
    assert_eq!(m.ffn_masks.len(), 4, "looped file must carry virtual rows");
    assert_eq!(m.ffn_masks[2], phys[0], "pass 1, layer 0 must mirror the physical row");
    assert_eq!(m.ffn_masks[3], phys[1], "pass 1, layer 1 must mirror the physical row");
}

/// An unlooped model's mask area is byte-identical to what it always
/// was — the extension costs existing files nothing.
#[test]
fn unlooped_layout_is_unchanged() {
    let a1 = arch(1);
    let rows = vec![vec![0xFF, 0xFF], vec![0xFF, 0x0F]];
    let cat = MaskCatalog {
        masks: vec![mask_with_ffn(rows.clone())],
        default_task: String::new(),
    };
    let bytes = encode_masks_section(&cat, &a1).unwrap();
    let back = decode_masks_section(&bytes, &a1).unwrap();
    assert_eq!(back.masks[0].ffn_masks, rows);
    assert_eq!(
        back.masks[0].ffn_masks.len(),
        2,
        "unlooped: one row per physical layer, as ever"
    );
}

/// A physical-length gate list on a looped model must read as ALIVE for
/// every visit: the runtime asks `layer_alive(virtual)`, and the old
/// false-default silently killed the whole second pass — a specialist
/// that produced CJK soup the moment its mask went active, while every
/// batched scorer (which never consults gates) said the file was fine.
#[test]
fn physical_gates_replicate_and_absent_gates_read_alive() {
    let a = arch(2);
    let cat = MaskCatalog {
        masks: vec![mask_with_ffn(vec![vec![0xFF, 0xFF], vec![0xFF, 0xFF]])],
        default_task: String::new(),
    };
    let bytes = encode_masks_section(&cat, &a).unwrap();
    let back = decode_masks_section(&bytes, &a).unwrap();
    let m = &back.masks[0];
    for vl in 0..4 {
        assert!(
            m.layer_alive(vl),
            "visit {vl} read as dead from a physical-length gate list"
        );
    }
    // And past ANY recorded gate — absence is "no restriction", not death.
    assert!(m.layer_alive(97), "an absent gate must read alive");
}
