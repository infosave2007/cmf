//! Synthetic Cortiq Embryo genomes with a natively bounded anchor, written
//! with `CmfModel::write` in the trainer's export layout (tensor names and
//! arch JSON as `crates/cortiq-embryo/src/export.rs` produces them): vmf_phase
//! mixers with κ-gate and short conv — or, for the plan's variant B,
//! `gated_delta_net` mixers (`linear_attn.*`, the names `loader.rs` reads) —
//! resonance-routed experts + shared expert, the hierarchical head, and
//! `swa_sink_v1` anchors with trained sinks. Weights are pseudo-random
//! (deterministic in the seed) — the files exercise every operator the
//! runtime executes for a real genome, at the real geometry, without a
//! checkpoint.
#![allow(dead_code)]

use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec};
use cortiq_core::types::{QuantType, TensorDtype};

/// GatedDeltaNet mixer geometry (plan §1.3: `nk = nv = 4, dk = dv = 128`,
/// conv 4 taps for Embryo-3).
#[derive(Clone, Debug)]
pub struct SynthGdn {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
}

impl SynthGdn {
    pub fn c_dim(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
}

#[derive(Clone, Debug)]
pub struct SynthGeom {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    /// Zero-based anchor layers (the rest are mixers).
    pub anchor_layers: Vec<usize>,
    pub heads: usize,
    pub nphase: usize,
    pub dv: usize,
    pub qh: usize,
    pub kvh: usize,
    pub hd: usize,
    pub experts: usize,
    pub inter: usize,
    pub head_clusters: usize,
    pub conv_k: usize,
    pub window: usize,
    pub sink: usize,
    pub seed: u64,
    /// `Some` → the mixers are `gated_delta_net` layers of this geometry
    /// (the phase fields above are then unused); `None` → vmf_phase.
    pub gdn: Option<SynthGdn>,
    /// `true` → the anchors are legacy `FullAttention` layers (growing
    /// KV, absolute RoPE, no `anchor_core`, no sinks): the pre-S1 export
    /// layout, for tests of the legacy CPU/GPU paths.
    pub legacy_full: bool,
}

impl SynthGeom {
    /// Embryo-0 (56M) geometry with the plan's variant-A anchor: 7 mixers
    /// + 1 bounded anchor on layer 7, W=128, S=4.
    pub fn embryo0_bounded() -> Self {
        Self {
            vocab: 32768,
            hidden: 384,
            layers: 8,
            anchor_layers: vec![7],
            heads: 8,
            nphase: 32,
            dv: 128,
            qh: 8,
            kvh: 2,
            hd: 128,
            experts: 4,
            inter: 768,
            head_clusters: 128,
            conv_k: 4,
            window: 128,
            sink: 4,
            seed: 20260923,
            gdn: None,
            legacy_full: false,
        }
    }

    /// Plan variant B (Embryo-3, 56M class): 6 × `gated_delta_net`
    /// (nk = nv = 4, dk = dv = 128, conv 4) + 2 × bounded anchors on
    /// layers 3 and 7, W=128, S=4; router/experts/head as Embryo-0.
    /// GDN layer ≈ 992k parameters, matching the plan's per-layer count.
    pub fn embryo3_gdn_bounded() -> Self {
        Self {
            vocab: 32768,
            hidden: 384,
            layers: 8,
            anchor_layers: vec![3, 7],
            heads: 0,
            nphase: 0,
            dv: 0,
            qh: 8,
            kvh: 2,
            hd: 128,
            experts: 4,
            inter: 768,
            head_clusters: 128,
            conv_k: 0,
            window: 128,
            sink: 4,
            seed: 20260924,
            gdn: Some(SynthGdn {
                nv: 4,
                nk: 4,
                dk: 128,
                dv: 128,
                kk: 4,
            }),
            legacy_full: false,
        }
    }

    /// Legacy Embryo-class genome for the pre-S1 paths: two layers, a
    /// vmf_phase mixer and a `FullAttention` anchor at the Embryo-0
    /// width (hidden 384 — the small matrices that make every pool
    /// dispatch a limited one), vocab 4096.
    pub fn legacy_full_small() -> Self {
        Self {
            vocab: 4096,
            hidden: 384,
            layers: 2,
            anchor_layers: vec![1],
            heads: 8,
            nphase: 32,
            dv: 128,
            qh: 8,
            kvh: 2,
            hd: 128,
            experts: 4,
            inter: 768,
            head_clusters: 64,
            conv_k: 4,
            window: 128,
            sink: 4,
            seed: 31,
            gdn: None,
            legacy_full: true,
        }
    }

    /// Embryo-0 width with a bounded anchor in two layers (one vmf_phase
    /// mixer + conv, one `swa_sink_v1` anchor), vocab 4096: the long-prefix
    /// chunk-agreement gate at a cost a test can pay.
    pub fn bounded_small() -> Self {
        let mut g = Self::legacy_full_small();
        g.legacy_full = false;
        g.seed = 37;
        g
    }

    /// The trainer's `tiny()` geometry with a bounded anchor: fast enough
    /// for a debug-build parity run, every operator still present.
    pub fn tiny_bounded() -> Self {
        Self {
            vocab: 4096,
            hidden: 64,
            layers: 2,
            anchor_layers: vec![1],
            heads: 2,
            nphase: 32,
            dv: 64,
            qh: 2,
            kvh: 1,
            hd: 64,
            experts: 2,
            inter: 128,
            head_clusters: 64,
            conv_k: 4,
            window: 128,
            sink: 4,
            seed: 7,
            gdn: None,
            legacy_full: false,
        }
    }

    /// Tiny GDN + bounded genome (2 layers: one GDN mixer, one anchor)
    /// for debug-build parity runs of the kind-4 resident path.
    pub fn tiny_gdn_bounded() -> Self {
        Self {
            vocab: 4096,
            hidden: 64,
            layers: 2,
            anchor_layers: vec![1],
            heads: 0,
            nphase: 0,
            dv: 0,
            qh: 2,
            kvh: 1,
            hd: 64,
            experts: 2,
            inter: 128,
            head_clusters: 64,
            conv_k: 0,
            window: 128,
            sink: 4,
            seed: 9,
            gdn: Some(SynthGdn {
                nv: 2,
                nk: 1,
                dk: 32,
                dv: 32,
                kk: 4,
            }),
            legacy_full: false,
        }
    }

    pub fn is_anchor(&self, l: usize) -> bool {
        self.anchor_layers.contains(&l)
    }

    pub fn tag(&self) -> String {
        let mut mixer = match &self.gdn {
            Some(g) => format!("gdn{}x{}x{}k{}", g.nv, g.dk, g.dv, g.kk),
            None => "vmf".to_string(),
        };
        if self.legacy_full {
            mixer.push_str("-legacyfull");
        }
        format!(
            "{mixer}-h{}-l{}-v{}-w{}-s{}-seed{}",
            self.hidden, self.layers, self.vocab, self.window, self.sink, self.seed
        )
    }
}

pub fn synth_f32(n: usize, salt: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(salt.wrapping_mul(1442695040888963407) ^ 0x9E3779B97F4A7C15);
            let x = (x ^ (x >> 31)).wrapping_mul(0xBF58476D1CE4E5B9);
            (((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32) * scale
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// The arch block exactly as the exporter's `arch_json` writes it (through
/// serde so every optional field takes its default).
pub fn arch_json(g: &SynthGeom) -> serde_json::Value {
    let layer_types: Vec<&str> = (0..g.layers)
        .map(|l| {
            if g.is_anchor(l) {
                if g.legacy_full {
                    "FullAttention"
                } else {
                    "BoundedAttention"
                }
            } else {
                "LinearAttention"
            }
        })
        .collect();
    // The mixer family: `gated_delta_net` carries its geometry in the
    // flat `linear_*` arch fields the loader requires (`loader.rs`
    // `need(arch.linear_num_key_heads …)`), vmf_phase in `linear_core`.
    let (linear_core, lnk, lnv, ldk, ldv, lkk) = match &g.gdn {
        Some(d) => (
            serde_json::json!({
                "kind": "gated_delta_net",
                "num_heads": d.nv,
                "nphase": serde_json::Value::Null,
                "value_head_dim": d.dv,
                "phase_delta_layers": serde_json::Value::Null,
            }),
            d.nk,
            d.nv,
            d.dk,
            d.dv,
            Some(d.kk),
        ),
        None => (
            serde_json::json!({
                "kind": "vmf_phase",
                "num_heads": g.heads,
                "nphase": g.nphase,
                "value_head_dim": g.dv,
                "phase_delta_layers": serde_json::Value::Null,
            }),
            g.heads,
            g.heads,
            g.nphase,
            g.dv,
            None,
        ),
    };
    let mut j = serde_json::json!({
        "arch_name": "cortiq_embryo",
        "hidden_size": g.hidden,
        "intermediate_size": g.inter,
        "num_layers": g.layers,
        "num_attention_heads": g.qh,
        "num_kv_heads": g.kvh,
        "head_dim": g.hd,
        "vocab_size": g.vocab,
        "layer_types": layer_types,
        "rms_norm_eps": 1e-5f64,
        "rope_theta": 10000.0f64,
        "tie_word_embeddings": true,
        "max_position_embeddings": 131072,
        "linear_core": linear_core,
        "linear_num_key_heads": lnk,
        "linear_num_value_heads": lnv,
        "linear_key_head_dim": ldk,
        "linear_value_head_dim": ldv,
        "linear_conv_kernel_dim": lkk,
        "anchor_core": if g.legacy_full {
            serde_json::Value::Null
        } else {
            serde_json::json!({
                "kind": "swa_sink_v1",
                "window": g.window,
                "sink": g.sink,
                "rope": "relative_in_window",
                "sink_scores": "nope",
                "train_windows": [g.window / 2, g.window],
            })
        },
    });
    if g.experts > 0 {
        j["moe"] = serde_json::json!({
            "num_experts": g.experts,
            "top_k": 1,
            "moe_intermediate_size": g.inter,
            "norm_topk_prob": true,
            "shared_expert_intermediate_size": g.inter,
            "router_resonance": true,
        });
    }
    if g.head_clusters > 0 {
        j["head_clusters"] = serde_json::json!(g.head_clusters);
    }
    j
}

/// Write the genome to `path`. Deterministic in `g` (same bytes every call).
pub fn write_synth_genome(path: &std::path::Path, g: &SynthGeom) {
    write_synth_genome_with(path, g, None);
}

/// [`write_synth_genome`] with an optional frozen-genome record
/// (`header.genome` + a `birth` lineage event): the writer fills the trunk
/// hashes and derives the GENOME feature bit.
pub fn write_synth_genome_with(
    path: &std::path::Path,
    g: &SynthGeom,
    genome: Option<cortiq_core::GenomeInfo>,
) {
    let h = g.hidden;
    let s0 = g.seed.wrapping_mul(1_000_003);
    let mut t: Vec<TensorSpec> = Vec::new();
    let mut spec = |name: &str, shape: &[usize], data: Vec<f32>| {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "{name}: shape/len"
        );
        t.push(TensorSpec {
            name: name.to_string(),
            dtype: TensorDtype::F32,
            shape: shape.to_vec(),
            data: f32_bytes(&data),
        });
    };
    let norm1 = |n: usize, salt: u64| -> Vec<f32> {
        synth_f32(n, salt, 0.2).iter().map(|v| 1.0 + v).collect()
    };
    // Projection scale: rows of width `cols` at U(−a, a) with a = 1.5/√cols
    // give unit-ish outputs for a unit-norm input.
    let mat = |rows: usize, cols: usize, salt: u64| -> Vec<f32> {
        synth_f32(rows * cols, salt, 1.5 / (cols as f32).sqrt())
    };
    spec(
        "model.embed_tokens.weight",
        &[g.vocab, h],
        synth_f32(g.vocab * h, s0 + 1, 0.6),
    );
    spec("model.norm.weight", &[h], norm1(h, s0 + 2));
    if g.head_clusters > 0 {
        spec(
            "lm_head.clusters.weight",
            &[g.head_clusters, h],
            mat(g.head_clusters, h, s0 + 3),
        );
    }
    // Decay grid over log-spaced horizons [8, 2048], A_log = ln(−ln γ).
    let p2 = 2 * g.nphase;
    let a_log: Vec<f32> = (0..g.heads * p2)
        .map(|i| {
            let f = (i % p2) as f64 / (p2.max(2) - 1) as f64;
            let horizon = 8.0f64 * (2048.0f64 / 8.0).powf(f);
            let gamma = (-1.0 / horizon).exp();
            (-(gamma.ln())).ln() as f32
        })
        .collect();
    for l in 0..g.layers {
        let pf = format!("model.layers.{l}.");
        let sl = s0 + 100 * (l as u64 + 1);
        spec(
            &format!("{pf}input_layernorm.weight"),
            &[h],
            norm1(h, sl + 1),
        );
        spec(
            &format!("{pf}post_attention_layernorm.weight"),
            &[h],
            norm1(h, sl + 2),
        );
        if g.is_anchor(l) {
            let (qh, kvh, hd) = (g.qh, g.kvh, g.hd);
            spec(
                &format!("{pf}self_attn.q_proj.weight"),
                &[qh * hd, h],
                mat(qh * hd, h, sl + 10),
            );
            spec(
                &format!("{pf}self_attn.k_proj.weight"),
                &[kvh * hd, h],
                mat(kvh * hd, h, sl + 11),
            );
            spec(
                &format!("{pf}self_attn.v_proj.weight"),
                &[kvh * hd, h],
                mat(kvh * hd, h, sl + 12),
            );
            spec(
                &format!("{pf}self_attn.o_proj.weight"),
                &[h, qh * hd],
                mat(h, qh * hd, sl + 13),
            );
            if !g.legacy_full {
                spec(
                    &format!("{pf}self_attn.sink_k.weight"),
                    &[kvh, g.sink, hd],
                    synth_f32(kvh * g.sink * hd, sl + 14, 0.5),
                );
                spec(
                    &format!("{pf}self_attn.sink_v.weight"),
                    &[kvh, g.sink, hd],
                    synth_f32(kvh * g.sink * hd, sl + 15, 0.5),
                );
            }
        } else if let Some(d) = &g.gdn {
            // GatedDeltaNet mixer in the loader's tensor layout
            // (`model.layers.{l}.linear_attn.*`, conv `[c_dim, 1, kk]` as
            // the Qwen/LFM convention the exporter follows).  Init as the
            // plan §1.3 prescribes: `A_log = ln(1/H)` over a log grid of
            // horizons [8, 2048], `dt_bias = ln(e−1)` (softplus → 1),
            // conv = identity on the last tap + small noise, norm ≈ 1.  Salts
            // 90..96 sit above the expert range (40 + 10·e + 0..4, e < 5).
            let (nv, nk, dk, dv, kk) = (d.nv, d.nk, d.dk, d.dv, d.kk);
            let c_dim = d.c_dim();
            let la = format!("{pf}linear_attn.");
            spec(
                &format!("{la}in_proj_qkv.weight"),
                &[c_dim, h],
                mat(c_dim, h, sl + 90),
            );
            spec(
                &format!("{la}in_proj_z.weight"),
                &[nv * dv, h],
                mat(nv * dv, h, sl + 91),
            );
            spec(
                &format!("{la}in_proj_a.weight"),
                &[nv, h],
                mat(nv, h, sl + 92),
            );
            spec(
                &format!("{la}in_proj_b.weight"),
                &[nv, h],
                mat(nv, h, sl + 93),
            );
            let mut taps = synth_f32(c_dim * kk, sl + 94, 0.2);
            for c in 0..c_dim {
                taps[c * kk + kk - 1] += 1.0;
            }
            spec(&format!("{la}conv1d.weight"), &[c_dim, 1, kk], taps);
            let a_log: Vec<f32> = (0..nv)
                .map(|i| {
                    let f = i as f64 / (nv.max(2) - 1) as f64;
                    let horizon = 8.0f64 * (2048.0f64 / 8.0).powf(f);
                    (1.0 / horizon).ln() as f32
                })
                .collect();
            spec(&format!("{la}A_log"), &[nv], a_log);
            spec(
                &format!("{la}dt_bias"),
                &[nv],
                vec![(std::f64::consts::E - 1.0).ln() as f32; nv],
            );
            spec(&format!("{la}norm.weight"), &[dv], norm1(dv, sl + 95));
            spec(
                &format!("{la}out_proj.weight"),
                &[h, nv * dv],
                mat(h, nv * dv, sl + 96),
            );
            let _ = (nk, dk);
        } else {
            let (nh, nph, dv) = (g.heads, g.nphase, g.dv);
            spec(
                &format!("{pf}vmf_attn.thq.weight"),
                &[nh * nph, h],
                mat(nh * nph, h, sl + 20),
            );
            spec(
                &format!("{pf}vmf_attn.thk.weight"),
                &[nh * nph, h],
                mat(nh * nph, h, sl + 21),
            );
            spec(
                &format!("{pf}vmf_attn.v_proj.weight"),
                &[nh * dv, h],
                mat(nh * dv, h, sl + 22),
            );
            spec(
                &format!("{pf}vmf_attn.out_proj.weight"),
                &[h, nh * dv],
                mat(h, nh * dv, sl + 23),
            );
            spec(&format!("{pf}vmf_attn.A_log"), &[nh * p2], a_log.clone());
            spec(
                &format!("{pf}vmf_attn.k_gate.weight"),
                &[nh, h],
                mat(nh, h, sl + 24),
            );
            spec(
                &format!("{pf}vmf_attn.k_gate.bias"),
                &[nh],
                vec![2.0; nh],
            );
            if g.conv_k >= 2 {
                // Identity-ish taps: last tap ≈ 1, the rest small.
                let mut taps = synth_f32(h * g.conv_k, sl + 25, 0.2);
                for c in 0..h {
                    taps[c * g.conv_k + g.conv_k - 1] += 1.0;
                }
                spec(
                    &format!("{pf}vmf_attn.conv1d.weight"),
                    &[h, 1, g.conv_k],
                    taps,
                );
            }
        }
        let i = g.inter;
        if g.experts == 0 {
            spec(&format!("{pf}mlp.gate_proj.weight"), &[i, h], mat(i, h, sl + 30));
            spec(&format!("{pf}mlp.up_proj.weight"), &[i, h], mat(i, h, sl + 31));
            spec(&format!("{pf}mlp.down_proj.weight"), &[h, i], mat(h, i, sl + 32));
        } else {
            spec(
                &format!("{pf}mlp.shared_expert.gate_proj.weight"),
                &[i, h],
                mat(i, h, sl + 30),
            );
            spec(
                &format!("{pf}mlp.shared_expert.up_proj.weight"),
                &[i, h],
                mat(i, h, sl + 31),
            );
            spec(
                &format!("{pf}mlp.shared_expert.down_proj.weight"),
                &[h, i],
                mat(h, i, sl + 32),
            );
            for e in 0..g.experts {
                let se = sl + 40 + 10 * e as u64;
                spec(
                    &format!("{pf}mlp.experts.{e}.gate_proj.weight"),
                    &[i, h],
                    mat(i, h, se),
                );
                spec(
                    &format!("{pf}mlp.experts.{e}.up_proj.weight"),
                    &[i, h],
                    mat(i, h, se + 1),
                );
                spec(
                    &format!("{pf}mlp.experts.{e}.down_proj.weight"),
                    &[h, i],
                    mat(h, i, se + 2),
                );
                // Resonance descriptors: centre, rank-16 subspace, bias.
                spec(
                    &format!("{pf}mlp.experts.{e}.desc.mu"),
                    &[h],
                    synth_f32(h, se + 3, 1.0),
                );
                spec(
                    &format!("{pf}mlp.experts.{e}.desc.u"),
                    &[16, h],
                    mat(16, h, se + 4),
                );
                spec(
                    &format!("{pf}mlp.experts.{e}.desc.bias"),
                    &[1],
                    vec![0.0],
                );
            }
        }
    }
    let arch: cortiq_core::types::ModelArch =
        serde_json::from_value(arch_json(g)).expect("synthetic arch deserializes");
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::format::CMF_VERSION,
        arch,
        quant_type: QuantType::F32,
        provenance: Some(serde_json::json!({
            "producer": "cortiq-engine tests (synthetic)",
            "genome": "embryo-0-bounded-synthetic",
        })),
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
        lineage: if genome.is_some() {
            vec![cortiq_core::LineageEvent::now(
                0,
                "birth",
                serde_json::json!({"producer": "embryo_synth", "seed": g.seed}),
            )]
        } else {
            Vec::new()
        },
        genome,
        router: None,
        segments: Vec::new(),
    };
    // A genome binds its tokenizer (core refuses a genome without VOCAB):
    // embed the byte-level synthetic one plus the ChatML markers.
    let vocab = header.genome.as_ref().map(|_| synth_vocab_json());
    CmfModel::write(path, &header, &t, None, vocab.as_deref()).expect("write synthetic genome");
}

/// The synthetic tokenizer.json a genome file embeds: the 256 byte tokens
/// `<0xNN>` = id NN (so plain text tokenizes exactly as the byte-level
/// fallback `Tokenizer::byte_level()` does) plus `<|im_start|>` (256) and
/// `<|im_end|>` (257) as added special tokens — the cmf-im-v1 frame.
pub fn synth_vocab_json() -> Vec<u8> {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..256u32)
        .map(|b| (format!("<0x{b:02X}>"), serde_json::json!(b)))
        .collect();
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 256, "content": "<|im_start|>", "special": true},
            {"id": 257, "content": "<|im_end|>", "special": true},
        ],
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }))
    .expect("synthetic vocab serializes")
}

/// Write (once) and return the path of a synthetic genome in the temp dir,
/// keyed by its geometry so two processes of a parity run share one file.
pub fn synth_genome_path(g: &SynthGeom) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cortiq-synth-bounded-{}.cmf", g.tag()));
    if !path.exists() {
        let tmp = path.with_extension("cmf.part");
        write_synth_genome(&tmp, g);
        std::fs::rename(&tmp, &path).expect("rename synthetic genome");
    }
    path
}

/// Deterministic pseudo-random token ids in `[1, vocab)`.
pub fn synth_ids(n: usize, seed: u64, vocab: usize) -> Vec<u32> {
    synth_f32(n, seed, 1.0)
        .iter()
        .map(|x| (1 + (((x + 0.5) * (vocab - 1) as f32) as usize % (vocab - 1))) as u32)
        .collect()
}
