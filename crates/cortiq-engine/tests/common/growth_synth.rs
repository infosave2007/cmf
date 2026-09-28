//! Growth records (`kind = "expert_append"`, spec §9.5.1 / §2) over a
//! synthetic resonance-routed Embryo genome, for the runtime and CLI
//! tests:
//!
//! * **F0** — a frozen genome (`header.genome`, bit GENOME): 3 layers
//!   (two `gated_delta_net` mixers + one bounded anchor), every layer an
//!   MoE of `E0 = 2` resonance-routed experts of rank 16;
//! * **F1** — F0 + one `expert_append` record (`herbs`) over
//!   [`GROWN_LAYERS`] with `count = 2` experts per layer, appended at the
//!   tail by `CmfModel::append_skill` from the format's own layout plan
//!   (`expert_append_layout`), so the file is exactly what the trainer's
//!   `--record-out` produces. Layer 0 is not grown.
//!
//! The two grown experts of every layer are designed so the tests can
//! reason about routing exactly:
//! * `k = 0` — the **far** expert: centre `μ = 100·1`, `U = 0`, shell
//!   [`FAR_SHELL`]. Its reconstruction error `‖x − μ‖²` is ≈ 6·10⁵ on any
//!   hidden state of the genome — always outside its shell (−∞ with the
//!   shell on) and never the best score with the shell off;
//! * `k = 1` — the **magnet** expert: `μ` = the trunk expert 0's centre,
//!   `U = 1.05 · U₀` (same rank), shell [`MAGNET_SHELL`] = 0. Its error is
//!   `d² − 1.1025·proj₀ < d² − proj₀` = the trunk expert 0's error, so with
//!   the shell OFF it wins every token expert 0 would have won (and never
//!   ties, lower index or not); with the shell ON its error `≈ 0.8·d² > 0`
//!   puts it outside the shell — −∞, invisible.
//!
//! Hence: shell on ⇒ the grown file's per-op forward is bit-identical to
//! F0's on every token (G2 by construction); shell off ⇒ grown wins > 0
//! and the logits move. The expert FFN weights are pseudo-random and
//! differ from every trunk expert's.
//!
//! Needs the including test crate to declare `embryo_synth` at its root.
#![allow(dead_code)]

use super::embryo_synth::{SynthGeom, synth_f32, write_synth_genome_with};
use cortiq_core::format::TensorSpec;
use cortiq_core::knowledge::{expert_leaf, skill_kind};
use cortiq_core::{
    CmfModel, ExpertAppend, GenomeInfo, SkillBound, SkillRecord, TensorDtype,
    expert_append_layout, trunk_expert_rank,
};
use std::path::{Path, PathBuf};

pub const GENOME_ID: &str = "synth-growth-genome";
pub const RECORD_ID: &str = "herbs";
/// Trunk experts per layer of [`geom`].
pub const E0: usize = 2;
/// Descriptor rank of the trunk (`desc.u [16, hidden]` in `embryo_synth`).
pub const RANK: usize = 16;
/// The layers the `herbs` record grows (layer 0 keeps `E0` experts).
pub const GROWN_LAYERS: [usize; 2] = [1, 2];
/// Grown experts per grown layer of the `herbs` record (far + magnet).
pub const COUNT: usize = 2;
pub const FAR_MU: f32 = 100.0;
pub const FAR_SHELL: f32 = 1.0;
pub const MAGNET_SHELL: f32 = 0.0;
pub const MAGNET_U_SCALE: f32 = 1.05;

/// Three layers (GDN, GDN, bounded anchor) at the trainer's tiny width,
/// every layer a 2-expert resonance MoE with a shared expert.
pub fn geom() -> SynthGeom {
    SynthGeom {
        layers: 3,
        anchor_layers: vec![2],
        seed: 41,
        ..SynthGeom::tiny_gdn_bounded()
    }
}

pub fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cmf-growth-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// F0: the frozen genome (writer fills `genome.moe_experts = E0`).
pub fn write_f0(path: &Path) {
    write_synth_genome_with(
        path,
        &geom(),
        Some(GenomeInfo::birth(GENOME_ID, "pre_chat", "f32")),
    );
}

/// f32 values of a trunk tensor.
pub fn f32s_of(model: &CmfModel, name: &str) -> Vec<f32> {
    model
        .tensor_bytes(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f32_spec(name: &str, shape: &[usize], v: &[f32]) -> TensorSpec {
    assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape/len");
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::F32,
        shape: shape.to_vec(),
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// The `expert_append` record `id` over `layers` with `count` grown
/// experts per layer (`k = 0` far, `k = 1` magnet, `k ≥ 2` far again),
/// its tensors built from the reader's own layout plan for the record's
/// position `header.skills.len()`.
pub fn grown_record(
    model: &CmfModel,
    id: &str,
    layers: &[usize],
    count: usize,
    status: &str,
) -> (SkillRecord, Vec<TensorSpec>) {
    let g = model.header.genome.clone().expect("genome record");
    let rank = trunk_expert_rank(&model.header, &model.tensors, layers[0]).expect("trunk rank");
    let record = SkillRecord {
        id: id.into(),
        name: Some(format!("synthetic growth '{id}'")),
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
        status: Some(status.into()),
        origin: Some(serde_json::json!({"trigger": "test", "K": count})),
        ..Default::default()
    };
    let at = model.header.skills.len();
    let plan = expert_append_layout(&model.header, &model.tensors, at, &record)
        .expect("layout of a valid record");
    let h = model.arch().hidden_size;
    let mut tensors = Vec::with_capacity(plan.len());
    for p in &plan {
        let base = plan
            .iter()
            .filter(|q| q.layer == p.layer)
            .map(|q| q.expert)
            .min()
            .unwrap();
        let k = p.expert - base;
        let magnet = k == 1;
        let trunk0 = |leaf: &str| {
            f32s_of(
                model,
                &cortiq_core::knowledge::expert_tensor_name(p.layer, 0, leaf),
            )
        };
        let salt = 7_000_000 + (p.layer as u64) * 1000 + (p.expert as u64) * 10;
        let n: usize = p.shape.iter().product();
        let v: Vec<f32> = match p.leaf {
            expert_leaf::GATE => synth_f32(n, salt + 1, 1.5 / (p.shape[1] as f32).sqrt()),
            expert_leaf::UP => synth_f32(n, salt + 2, 1.5 / (p.shape[1] as f32).sqrt()),
            expert_leaf::DOWN => synth_f32(n, salt + 3, 1.5 / (p.shape[1] as f32).sqrt()),
            expert_leaf::MU => {
                if magnet {
                    trunk0(expert_leaf::MU)
                } else {
                    vec![FAR_MU; h]
                }
            }
            expert_leaf::U => {
                if magnet {
                    trunk0(expert_leaf::U)
                        .iter()
                        .map(|x| x * MAGNET_U_SCALE)
                        .collect()
                } else {
                    vec![0.0; n]
                }
            }
            expert_leaf::BIAS => vec![0.0],
            expert_leaf::SHELL => vec![if magnet { MAGNET_SHELL } else { FAR_SHELL }],
            other => panic!("unexpected leaf {other}"),
        };
        tensors.push(f32_spec(&p.name, &p.shape, &v));
    }
    (record, tensors)
}

/// Append the record `id` to `path` (a copy of F0 or an already grown
/// file): `count` experts on `layers`, `status`.
pub fn append_record(path: &Path, id: &str, layers: &[usize], count: usize, status: &str) {
    let model = CmfModel::open(path).expect("open before append");
    let (rec, ts) = grown_record(&model, id, layers, count, status);
    drop(model);
    CmfModel::append_skill(path, rec, &ts, None, None, None).expect("append expert_append");
}

/// Write F0 and F1 = F0 + `herbs` (status `status`) into `dir`.
pub fn write_growth_pair(dir: &Path, status: &str) -> (PathBuf, PathBuf) {
    std::fs::create_dir_all(dir).unwrap();
    let f0 = dir.join("f0.cmf");
    let f1 = dir.join("f1.cmf");
    write_f0(&f0);
    std::fs::copy(&f0, &f1).expect("copy F0 → F1");
    append_record(&f1, RECORD_ID, &GROWN_LAYERS, COUNT, status);
    (f0, f1)
}
