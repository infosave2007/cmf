//! Task mask management — per-task neuron/head/layer masks.
//!
//! A mask selects an active subset of the shared weights (weights are
//! never modified — VMF principle: a skill is a regular core of the
//! condensate). Bit order is LSB-first: neuron `i` = bit `i % 8` of
//! byte `i / 8`; bit 1 = active. Tail bits beyond the dimension MUST
//! be zero (otherwise popcount sees phantom neurons/heads).

use crate::types::ModelArch;
use serde::{Deserialize, Serialize};

/// Held-out quality contract for a mask. `None` means "not measured" —
/// the format forbids declaring quality without a measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quality {
    /// e.g. "heldout_ppl_ratio", "heldout_acc"
    pub metric: String,
    pub value: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_dense: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_samples: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_sha256: Option<String>,
}

/// A single task mask defining which neurons/heads/layers are active.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMask {
    /// Unique task identifier
    pub task_id: u32,
    /// Human-readable task name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Overall sparsity (0.0 = no pruning, 1.0 = fully pruned)
    pub sparsity: f32,
    /// Held-out quality (None = not measured)
    #[serde(default)]
    pub quality: Option<Quality>,
    /// Per-layer FFN neuron masks (bitfield: 1 = active)
    pub ffn_masks: Vec<Vec<u8>>,
    /// Per-layer attention head masks (bitfield: 1 = active)
    pub head_masks: Vec<Vec<u8>>,
    /// Per-layer alive flags
    pub layer_gates: Vec<bool>,
    /// Per-layer MoE expert masks (bitfield: 1 = routable). Empty =
    /// no expert restriction (all experts routable) — the state of
    /// every pre-expert-mask file. Spec §5: an optional area after
    /// the layer gates, present when the meta flag is set.
    #[serde(default)]
    pub expert_masks: Vec<Vec<u8>>,
    /// Parent mask name (for delta-coded masks)
    pub parent: Option<String>,
    /// Whether this mask has a precompiled hot-pack
    pub has_hot_pack: bool,
    /// Priority level for this mask
    pub priority: MaskPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskPriority {
    Fallback,
    Normal,
    Primary,
}

impl TaskMask {
    /// Count active neurons in a specific layer's FFN.
    pub fn ffn_active_count(&self, layer_idx: usize) -> usize {
        self.ffn_masks
            .get(layer_idx)
            .map(|m| m.iter().map(|b| b.count_ones() as usize).sum())
            .unwrap_or(0)
    }

    /// Check if a specific layer is alive (not pruned).
    pub fn layer_alive(&self, layer_idx: usize) -> bool {
        self.layer_gates.get(layer_idx).copied().unwrap_or(false)
    }

    /// Count total active layers.
    pub fn active_layer_count(&self) -> usize {
        self.layer_gates.iter().filter(|&&alive| alive).count()
    }

    /// Get active neuron indices for a layer (for sparse gather).
    pub fn ffn_active_indices(&self, layer_idx: usize) -> Vec<u16> {
        let Some(mask) = self.ffn_masks.get(layer_idx) else {
            return vec![];
        };
        let mut indices = Vec::new();
        for (byte_idx, &byte) in mask.iter().enumerate() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    indices.push((byte_idx * 8 + bit) as u16);
                }
            }
        }
        indices
    }

    /// Count active attention heads in a layer.
    pub fn active_head_count(&self, layer_idx: usize) -> usize {
        self.head_masks
            .get(layer_idx)
            .map(|m| m.iter().map(|b| b.count_ones() as usize).sum())
            .unwrap_or(0)
    }

    /// Expert routability flags for a layer (true = routable), or None
    /// when the mask carries no expert restriction.
    pub fn expert_flags(&self, layer_idx: usize, num_experts: usize) -> Option<Vec<bool>> {
        let mask = self.expert_masks.get(layer_idx)?;
        if mask.is_empty() {
            return None;
        }
        Some(
            (0..num_experts)
                .map(|e| mask.get(e / 8).map(|b| b & (1 << (e % 8)) != 0).unwrap_or(false))
                .collect(),
        )
    }

    /// Active head flags for a layer (true = head is alive).
    pub fn head_flags(&self, layer_idx: usize, num_heads: usize) -> Vec<bool> {
        let mut flags = vec![true; num_heads];
        if let Some(mask) = self.head_masks.get(layer_idx) {
            for (h, flag) in flags.iter_mut().enumerate() {
                *flag = mask
                    .get(h / 8)
                    .map(|b| b & (1 << (h % 8)) != 0)
                    .unwrap_or(false);
            }
        }
        flags
    }

    /// Average active neurons across all alive layers.
    pub fn avg_active_neurons(&self) -> f64 {
        let alive_layers: Vec<_> = (0..self.layer_gates.len())
            .filter(|&i| self.layer_alive(i))
            .collect();
        if alive_layers.is_empty() {
            return 0.0;
        }
        let total: usize = alive_layers.iter().map(|&i| self.ffn_active_count(i)).sum();
        total as f64 / alive_layers.len() as f64
    }

    /// Compute union of two masks (more neurons = higher quality, less speed).
    pub fn union(&self, other: &TaskMask) -> TaskMask {
        let mut result = self.clone();
        result.name = format!("{}+{}", self.name, other.name);
        result.task_id = u32::MAX; // composite
        result.parent = None;
        result.has_hot_pack = false;
        result.quality = None; // union quality is not measured

        for (li, gate) in result.layer_gates.iter_mut().enumerate() {
            *gate = self.layer_alive(li) || other.layer_alive(li);
        }

        for (li, mask) in result.ffn_masks.iter_mut().enumerate() {
            if let Some(om) = other.ffn_masks.get(li) {
                for (byte, &ob) in mask.iter_mut().zip(om) {
                    *byte |= ob;
                }
            }
        }

        for (li, mask) in result.head_masks.iter_mut().enumerate() {
            if let Some(om) = other.head_masks.get(li) {
                for (byte, &ob) in mask.iter_mut().zip(om) {
                    *byte |= ob;
                }
            }
        }

        // Expert fields: empty = unrestricted, and unrestricted wins a
        // union; otherwise OR the routable sets.
        if result.expert_masks.is_empty() || other.expert_masks.is_empty() {
            result.expert_masks = Vec::new();
        } else {
            for (li, mask) in result.expert_masks.iter_mut().enumerate() {
                match other.expert_masks.get(li) {
                    Some(om) if !om.is_empty() && !mask.is_empty() => {
                        for (byte, &ob) in mask.iter_mut().zip(om) {
                            *byte |= ob;
                        }
                    }
                    _ => mask.clear(),
                }
            }
        }

        // Recalculate sparsity
        let total_neurons: usize = result.ffn_masks.iter().map(|m| m.len() * 8).sum();
        let active: usize = (0..result.layer_gates.len())
            .map(|i| result.ffn_active_count(i))
            .sum();
        result.sparsity = 1.0 - (active as f32 / total_neurons.max(1) as f32);

        result
    }

    /// Bitwise diff between current and new mask (for hot-swap).
    /// Compares the actual bits (XOR), not per-layer counters: two masks
    /// with equal counts but different neurons produce a full delta.
    pub fn diff(&self, other: &TaskMask) -> MaskDiff {
        let n_layers = self.layer_gates.len().max(other.layer_gates.len());
        let mut changed_layers = Vec::new();
        let mut neurons_added = 0usize;
        let mut neurons_removed = 0usize;
        let mut ffn_delta = Vec::with_capacity(n_layers);

        let empty: Vec<u8> = Vec::new();
        for li in 0..n_layers {
            let a = self.ffn_masks.get(li).unwrap_or(&empty);
            let b = other.ffn_masks.get(li).unwrap_or(&empty);
            let len = a.len().max(b.len());
            let mut delta = vec![0u8; len];
            let mut layer_changed = self.layer_alive(li) != other.layer_alive(li);

            for (bi, d) in delta.iter_mut().enumerate() {
                let av = a.get(bi).copied().unwrap_or(0);
                let bv = b.get(bi).copied().unwrap_or(0);
                let x = av ^ bv;
                *d = x;
                if x != 0 {
                    layer_changed = true;
                    neurons_added += (bv & !av).count_ones() as usize;
                    neurons_removed += (av & !bv).count_ones() as usize;
                }
            }

            // Head bits count toward "changed" too.
            let ha = self.head_masks.get(li).unwrap_or(&empty);
            let hb = other.head_masks.get(li).unwrap_or(&empty);
            if ha.len() != hb.len() || ha.iter().zip(hb).any(|(x, y)| x != y) {
                layer_changed = true;
            }

            if layer_changed {
                changed_layers.push(li);
            }
            ffn_delta.push(delta);
        }

        MaskDiff {
            changed_layers,
            neurons_added,
            neurons_removed,
            ffn_delta,
        }
    }

    /// Zero tail bits beyond the real dimensions (defensive normalization).
    pub fn normalize_tail_bits(&mut self, arch: &ModelArch) {
        for row in &mut self.ffn_masks {
            zero_tail_bits(row, arch.intermediate_size);
        }
        for row in &mut self.head_masks {
            zero_tail_bits(row, arch.num_attention_heads);
        }
    }
}

/// Zero all bits at positions >= `n_bits` in a bitfield.
pub fn zero_tail_bits(bits: &mut [u8], n_bits: usize) {
    let full_bytes = n_bits / 8;
    let rem = n_bits % 8;
    if full_bytes < bits.len() {
        if rem > 0 {
            bits[full_bytes] &= (1u8 << rem) - 1;
            for b in &mut bits[full_bytes + 1..] {
                *b = 0;
            }
        } else {
            for b in &mut bits[full_bytes..] {
                *b = 0;
            }
        }
    }
}

/// Result of diffing two masks (used for efficient hot-swap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskDiff {
    pub changed_layers: Vec<usize>,
    pub neurons_added: usize,
    pub neurons_removed: usize,
    /// Per-layer XOR bitfields — exactly which neurons flipped.
    #[serde(skip)]
    pub ffn_delta: Vec<Vec<u8>>,
}

/// Catalog of all masks in a CMF model file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskCatalog {
    pub masks: Vec<TaskMask>,
    pub default_task: String,
}

impl MaskCatalog {
    pub fn empty() -> Self {
        Self {
            masks: vec![],
            default_task: "general".to_string(),
        }
    }

    /// Find mask by name.
    pub fn get(&self, name: &str) -> Option<&TaskMask> {
        self.masks.iter().find(|m| m.name == name)
    }

    /// Get fallback mask.
    pub fn fallback(&self) -> Option<&TaskMask> {
        self.masks
            .iter()
            .find(|m| m.priority == MaskPriority::Fallback)
            .or(self.masks.first())
    }

    /// List all task names.
    pub fn task_names(&self) -> Vec<&str> {
        self.masks.iter().map(|m| m.name.as_str()).collect()
    }
}

// ───────────────────── binary masks section (§5 of the spec) ─────────────────────

/// JSON metadata part of the masks section.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MasksMeta {
    default_task: String,
    masks: Vec<MaskMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MaskMeta {
    task_id: u32,
    name: String,
    #[serde(default)]
    description: Option<String>,
    sparsity: f32,
    #[serde(default)]
    quality: Option<Quality>,
    #[serde(default)]
    parent: Option<String>,
    priority: MaskPriority,
    #[serde(default)]
    has_hot_pack: bool,
    /// Spec §5: the blob carries the optional per-layer MoE-expert
    /// bitfield area after the layer gates. Old readers that ignore
    /// this flag simply never look past the gates — additive.
    #[serde(default)]
    has_expert_fields: bool,
    /// Blob offset relative to the start of the masks section.
    blob_off: u64,
    blob_len: u64,
}

/// Encode a catalog into the binary masks section:
/// `[u32 n_masks][u32 meta_len][meta JSON][blobs, each 8-aligned]`.
/// Blob: `[n_layers × ffn_bytes][n_layers × head_bytes][gates_bytes]`.
pub fn encode_masks_section(catalog: &MaskCatalog, arch: &ModelArch) -> Result<Vec<u8>, String> {
    let ffn_b = arch.ffn_mask_bytes();
    let head_b = arch.head_mask_bytes();
    let gates_b = arch.gates_mask_bytes();
    let expert_b = arch.expert_mask_bytes();
    let blob_len = arch.mask_blob_len();

    // Build blobs first to know sizes. Blob lengths are per-mask now:
    // base, or base + the optional expert area (spec §5).
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(catalog.masks.len());
    for m in &catalog.masks {
        let has_experts = expert_b > 0 && m.expert_masks.iter().any(|e| !e.is_empty());
        let mut blob = Vec::with_capacity(if has_experts {
            arch.mask_blob_len_with_experts()
        } else {
            blob_len
        });
        for li in 0..arch.num_layers {
            let mut row = vec![0u8; ffn_b];
            if let Some(src) = m.ffn_masks.get(li) {
                let n = src.len().min(ffn_b);
                row[..n].copy_from_slice(&src[..n]);
            }
            zero_tail_bits(&mut row, arch.intermediate_size);
            blob.extend_from_slice(&row);
        }
        for li in 0..arch.num_layers {
            let mut row = vec![0u8; head_b];
            if let Some(src) = m.head_masks.get(li) {
                let n = src.len().min(head_b);
                row[..n].copy_from_slice(&src[..n]);
            }
            zero_tail_bits(&mut row, arch.num_attention_heads);
            blob.extend_from_slice(&row);
        }
        let mut gates = vec![0u8; gates_b];
        for li in 0..arch.num_layers {
            if m.layer_alive(li) {
                gates[li / 8] |= 1 << (li % 8);
            }
        }
        blob.extend_from_slice(&gates);
        if has_experts {
            let ne = arch.moe.as_ref().map(|c| c.num_experts).unwrap_or(0);
            for li in 0..arch.num_layers {
                let mut row = vec![0u8; expert_b];
                match m.expert_masks.get(li) {
                    Some(src) if !src.is_empty() => {
                        let n = src.len().min(expert_b);
                        row[..n].copy_from_slice(&src[..n]);
                        zero_tail_bits(&mut row, ne);
                    }
                    // A layer with no restriction inside a masked file:
                    // all experts routable (all-ones up to num_experts).
                    _ => {
                        for e in 0..ne {
                            row[e / 8] |= 1 << (e % 8);
                        }
                    }
                }
                blob.extend_from_slice(&row);
            }
        }
        blobs.push(blob);
    }

    // Two-pass meta serialization is fragile (JSON length depends on
    // offsets). Instead: compute meta with placeholder offsets of the
    // final width by serializing once, then patching is avoided by
    // computing the blobs area start from the meta length iteratively.
    let build_meta = |blobs_start: u64| -> MasksMeta {
        let mut metas = Vec::with_capacity(catalog.masks.len());
        let mut off = blobs_start;
        for (m, blob) in catalog.masks.iter().zip(&blobs) {
            off = off.div_ceil(8) * 8; // 8-align each blob
            metas.push(MaskMeta {
                task_id: m.task_id,
                name: m.name.clone(),
                description: m.description.clone(),
                sparsity: m.sparsity,
                quality: m.quality.clone(),
                parent: m.parent.clone(),
                priority: m.priority,
                has_hot_pack: m.has_hot_pack,
                has_expert_fields: expert_b > 0 && blob.len() > blob_len,
                blob_off: off,
                blob_len: blob.len() as u64,
            });
            off += blob.len() as u64;
        }
        MasksMeta {
            default_task: catalog.default_task.clone(),
            masks: metas,
        }
    };

    // Iterate until meta length stabilizes (offsets can change digit count).
    let mut meta_len = 0usize;
    let mut meta_json;
    loop {
        let blobs_start = 8 + meta_len as u64;
        meta_json = serde_json::to_vec(&build_meta(blobs_start))
            .map_err(|e| format!("serialize masks meta: {e}"))?;
        if meta_json.len() == meta_len {
            break;
        }
        meta_len = meta_json.len();
    }

    let meta = build_meta(8 + meta_len as u64);
    let mut out = Vec::new();
    out.extend_from_slice(&(catalog.masks.len() as u32).to_le_bytes());
    out.extend_from_slice(&(meta_len as u32).to_le_bytes());
    out.extend_from_slice(&meta_json);
    for (mm, blob) in meta.masks.iter().zip(&blobs) {
        while (out.len() as u64) < mm.blob_off {
            out.push(0);
        }
        debug_assert_eq!(out.len() as u64, mm.blob_off);
        out.extend_from_slice(blob);
    }
    Ok(out)
}

/// Decode the binary masks section into a catalog.
pub fn decode_masks_section(bytes: &[u8], arch: &ModelArch) -> Result<MaskCatalog, String> {
    if bytes.len() < 8 {
        return Err("masks section too short".into());
    }
    let n_masks = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let meta_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if 8 + meta_len > bytes.len() {
        return Err("masks meta out of bounds".into());
    }
    let meta: MasksMeta = serde_json::from_slice(&bytes[8..8 + meta_len])
        .map_err(|e| format!("masks meta JSON: {e}"))?;
    if meta.masks.len() != n_masks {
        return Err(format!(
            "masks count mismatch: envelope {} vs meta {}",
            n_masks,
            meta.masks.len()
        ));
    }

    let ffn_b = arch.ffn_mask_bytes();
    let head_b = arch.head_mask_bytes();
    let expert_b = arch.expert_mask_bytes();
    let expected_blob = arch.mask_blob_len() as u64;
    let expected_with_experts = arch.mask_blob_len_with_experts() as u64;

    let mut masks = Vec::with_capacity(n_masks);
    for mm in &meta.masks {
        let expected = if mm.has_expert_fields {
            expected_with_experts
        } else {
            expected_blob
        };
        if mm.has_expert_fields && expert_b == 0 {
            return Err(format!(
                "mask '{}': expert fields flagged but the arch has no MoE block",
                mm.name
            ));
        }
        if mm.blob_len != expected {
            return Err(format!(
                "mask '{}': blob_len {} != expected {} for arch",
                mm.name, mm.blob_len, expected
            ));
        }
        let start = usize::try_from(mm.blob_off)
            .map_err(|_| format!("mask '{}': blob offset does not fit usize", mm.name))?;
        let blob_len = usize::try_from(mm.blob_len)
            .map_err(|_| format!("mask '{}': blob length does not fit usize", mm.name))?;
        let end = start
            .checked_add(blob_len)
            .ok_or_else(|| format!("mask '{}': blob range overflows", mm.name))?;
        if end > bytes.len() {
            return Err(format!("mask '{}': blob out of bounds", mm.name));
        }
        let blob = &bytes[start..end];

        let mut ffn_masks = Vec::with_capacity(arch.num_layers);
        for li in 0..arch.num_layers {
            ffn_masks.push(blob[li * ffn_b..(li + 1) * ffn_b].to_vec());
        }
        let heads_base = arch.num_layers * ffn_b;
        let mut head_masks = Vec::with_capacity(arch.num_layers);
        for li in 0..arch.num_layers {
            head_masks
                .push(blob[heads_base + li * head_b..heads_base + (li + 1) * head_b].to_vec());
        }
        let gates_base = heads_base + arch.num_layers * head_b;
        let gates = &blob[gates_base..];
        let layer_gates: Vec<bool> = (0..arch.num_layers)
            .map(|li| gates[li / 8] & (1 << (li % 8)) != 0)
            .collect();
        let expert_masks: Vec<Vec<u8>> = if mm.has_expert_fields {
            let base = gates_base + arch.gates_mask_bytes();
            (0..arch.num_layers)
                .map(|li| blob[base + li * expert_b..base + (li + 1) * expert_b].to_vec())
                .collect()
        } else {
            Vec::new()
        };

        masks.push(TaskMask {
            task_id: mm.task_id,
            name: mm.name.clone(),
            description: mm.description.clone(),
            sparsity: mm.sparsity,
            quality: mm.quality.clone(),
            ffn_masks,
            head_masks,
            layer_gates,
            expert_masks,
            parent: mm.parent.clone(),
            has_hot_pack: mm.has_hot_pack,
            priority: mm.priority,
        });
    }

    Ok(MaskCatalog {
        masks,
        default_task: meta.default_task,
    })
}
