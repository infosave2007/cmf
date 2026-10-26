//! SAM3 **video** mode: the image detector plus the SAM2-style memory-bank
//! tracker packed in the same CMF (`trk.*`, 309 tensors, plus the tracker's
//! own FPN neck, which lives in the `fpn.sam2_convs` section).
//!
//! One frame goes through the same six steps as transformers 5.20.2
//! `Sam3VideoModel._det_track_one_frame`:
//!
//! 1. **detect** — one image forward ([`Sam3::forward`](crate::sam3::Sam3)),
//!    score `sigmoid(logit)·sigmoid(presence)`, mask-IoU NMS at
//!    `det_nms_thresh`, keep above `score_threshold_detection`;
//! 2. **tracker features** — trunk map → `fpn.sam2_convs` neck → levels
//!    288/144/72 with `conv_s0`/`conv_s1` on the two finest;
//! 3. **propagate** — per tracked object the current frame's features are
//!    fused with the memory bank (`trk.transformer.encoder.*`: 4 RoPE
//!    layers over past `maskmem_features` + object pointers) and decoded by
//!    the SAM mask decoder into 288² mask logits, object score and object
//!    pointer; small holes/sprinkles are filled;
//! 4. **plan** — IoU association with the detector
//!    (`assoc_iou_thresh` / `trk_assoc_iou_thresh` / `new_det_thresh`),
//!    hotstart keep-alive + duplicate bookkeeping, recent-occlusion
//!    suppression, non-overlap constraints and a fresh memory encode of the
//!    resulting masks;
//! 5. **execute** — new detections become conditioning frames
//!    (`_use_mask_as_output`: the detection mask IS the output, the decoder
//!    only provides the object pointer), dropped objects leave the session;
//! 6. **outputs** — per-object 288² mask logits, bilinearly interpolated to
//!    the frame and binarized at 0; box = mask bounding box (inclusive
//!    corners, like `torchvision.masks_to_boxes`); `track_id` is the session
//!    object id, stable for the object's whole life.
//!
//! Batch size 1 — one object at a time, as in the reference.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, Result};
use cortiq_core::CmfModel;

use crate::inference::gelu_erf;
use crate::pool::Pool;
use crate::sam3::{self, Sam3, Sam3Out};
use crate::sam3vid_net::{
    dense, ln_chw, resize2d, sine, sine_1d, sigm, Net, TrkFeats, EMB, HID, MEM, NO_OBJ_SCORE,
};

/// Low-resolution mask side (`low_res_mask_size`), also the detector's
/// `cfg.processor.mask_size`.
pub const MASK_SIDE: usize = 288;
/// Tracker input resolution (`tracker_config.image_size`).
const IMG: usize = 1008;
/// Memory-encoder mask target side (`mask_input_size · 4`).
const MEM_SIDE: usize = 1152;

// ───────────────────── configuration ─────────────────────

/// Every knob the video pipeline reads, from the CMF's `cfg.tracker`
/// (the vendor `config.json` of `Sam3VideoModel`). The defaults below are
/// the shipped values; [`VidCfg::from_cmf`] overrides them from the file.
#[derive(Clone, Debug, PartialEq)]
pub struct VidCfg {
    pub score_threshold_detection: f32,
    pub det_nms_thresh: f32,
    pub assoc_iou_thresh: f32,
    pub trk_assoc_iou_thresh: f32,
    pub new_det_thresh: f32,
    pub recondition_on_trk_masks: bool,
    pub recondition_every_nth_frame: i64,
    pub hotstart_delay: i64,
    pub hotstart_unmatch_thresh: usize,
    pub hotstart_dup_thresh: usize,
    pub suppress_unmatched_only_within_hotstart: bool,
    pub init_trk_keep_alive: i64,
    pub max_trk_keep_alive: i64,
    pub min_trk_keep_alive: i64,
    pub suppress_recent_occlusion_thresh: f32,
    pub decrease_keep_alive_for_empty: bool,
    pub fill_hole_area: usize,
    pub max_num_objects: usize,
    pub high_conf_thresh: f32,
    pub high_iou_thresh: f32,
    pub low_res_mask_size: usize,
    pub num_maskmem: usize,
    pub max_cond_frame_num: usize,
    pub max_object_pointers: usize,
    pub sigmoid_scale: f32,
    pub sigmoid_bias: f32,
    pub temporal_ptr_pe: bool,
    pub occlusion_spatial: bool,
    /// `multimask_output_in_sam && multimask_output_for_tracking` — the
    /// tracker decodes three masks and keeps the best-IoU one.
    pub multimask: bool,
}

impl Default for VidCfg {
    fn default() -> Self {
        Self {
            score_threshold_detection: 0.5,
            det_nms_thresh: 0.1,
            assoc_iou_thresh: 0.1,
            trk_assoc_iou_thresh: 0.5,
            new_det_thresh: 0.7,
            recondition_on_trk_masks: false,
            recondition_every_nth_frame: 16,
            hotstart_delay: 15,
            hotstart_unmatch_thresh: 8,
            hotstart_dup_thresh: 8,
            suppress_unmatched_only_within_hotstart: true,
            init_trk_keep_alive: 30,
            max_trk_keep_alive: 30,
            min_trk_keep_alive: -1,
            suppress_recent_occlusion_thresh: 0.7,
            decrease_keep_alive_for_empty: false,
            fill_hole_area: 16,
            max_num_objects: 10_000,
            high_conf_thresh: 0.8,
            high_iou_thresh: 0.8,
            low_res_mask_size: MASK_SIDE,
            num_maskmem: 7,
            max_cond_frame_num: 4,
            max_object_pointers: 16,
            sigmoid_scale: 20.0,
            sigmoid_bias: -10.0,
            temporal_ptr_pe: true,
            occlusion_spatial: true,
            multimask: true,
        }
    }
}

impl VidCfg {
    /// Parse `cfg.tracker` out of the CMF (missing keys keep the defaults).
    pub fn from_cmf(m: &CmfModel) -> Result<Self> {
        let mut c = Self::default();
        let raw = m
            .tensor("cfg.tracker")
            .ok_or_else(|| anyhow!("the CMF carries no cfg.tracker section — not a SAM3 video model"))?;
        let bytes = m.entry_bytes(raw);
        let v: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| anyhow!("cfg.tracker is not JSON: {e}"))?;
        let f = |k: &str, d: f32| v[k].as_f64().map_or(d, |x| x as f32);
        let i = |k: &str, d: i64| v[k].as_i64().unwrap_or(d);
        let b = |k: &str, d: bool| v[k].as_bool().unwrap_or(d);
        let u = |k: &str, d: usize| v[k].as_u64().map_or(d, |x| x as usize);

        c.score_threshold_detection = f("score_threshold_detection", c.score_threshold_detection);
        c.det_nms_thresh = f("det_nms_thresh", c.det_nms_thresh);
        c.assoc_iou_thresh = f("assoc_iou_thresh", c.assoc_iou_thresh);
        c.trk_assoc_iou_thresh = f("trk_assoc_iou_thresh", c.trk_assoc_iou_thresh);
        c.new_det_thresh = f("new_det_thresh", c.new_det_thresh);
        c.recondition_on_trk_masks = b("recondition_on_trk_masks", c.recondition_on_trk_masks);
        c.recondition_every_nth_frame = i("recondition_every_nth_frame", c.recondition_every_nth_frame);
        c.hotstart_delay = i("hotstart_delay", c.hotstart_delay);
        c.hotstart_unmatch_thresh = u("hotstart_unmatch_thresh", c.hotstart_unmatch_thresh);
        c.hotstart_dup_thresh = u("hotstart_dup_thresh", c.hotstart_dup_thresh);
        c.suppress_unmatched_only_within_hotstart =
            b("suppress_unmatched_only_within_hotstart", c.suppress_unmatched_only_within_hotstart);
        c.init_trk_keep_alive = i("init_trk_keep_alive", c.init_trk_keep_alive);
        c.max_trk_keep_alive = i("max_trk_keep_alive", c.max_trk_keep_alive);
        c.min_trk_keep_alive = i("min_trk_keep_alive", c.min_trk_keep_alive);
        c.suppress_recent_occlusion_thresh = f(
            "suppress_overlapping_based_on_recent_occlusion_threshold",
            c.suppress_recent_occlusion_thresh,
        );
        c.decrease_keep_alive_for_empty =
            b("decrease_trk_keep_alive_for_empty_masklets", c.decrease_keep_alive_for_empty);
        c.fill_hole_area = u("fill_hole_area", c.fill_hole_area);
        c.max_num_objects = u("max_num_objects", c.max_num_objects);
        c.high_conf_thresh = f("high_conf_thresh", c.high_conf_thresh);
        c.high_iou_thresh = f("high_iou_thresh", c.high_iou_thresh);
        c.low_res_mask_size = u("low_res_mask_size", c.low_res_mask_size);
        let t = &v["tracker_config"];
        c.num_maskmem = t["num_maskmem"].as_u64().map_or(c.num_maskmem, |x| x as usize);
        c.max_cond_frame_num = t["max_cond_frame_num"].as_u64().map_or(c.max_cond_frame_num, |x| x as usize);
        c.max_object_pointers =
            t["max_object_pointers_in_encoder"].as_u64().map_or(c.max_object_pointers, |x| x as usize);
        c.sigmoid_scale = t["sigmoid_scale_for_mem_enc"].as_f64().map_or(c.sigmoid_scale, |x| x as f32);
        c.sigmoid_bias = t["sigmoid_bias_for_mem_enc"].as_f64().map_or(c.sigmoid_bias, |x| x as f32);
        c.temporal_ptr_pe = t["enable_temporal_pos_encoding_for_object_pointers"]
            .as_bool()
            .unwrap_or(c.temporal_ptr_pe);
        c.occlusion_spatial =
            t["enable_occlusion_spatial_embedding"].as_bool().unwrap_or(c.occlusion_spatial);
        c.multimask = t["multimask_output_in_sam"].as_bool().unwrap_or(true)
            && t["multimask_output_for_tracking"].as_bool().unwrap_or(true);
        anyhow::ensure!(
            c.low_res_mask_size == MASK_SIDE,
            "unsupported low_res_mask_size {}",
            c.low_res_mask_size
        );
        anyhow::ensure!(c.num_maskmem <= 7, "num_maskmem {} > 7", c.num_maskmem);
        Ok(c)
    }
}

// ───────────────────── session state ─────────────────────

/// One object's stored output for one frame.
#[derive(Default, Clone)]
struct ObjOut {
    /// 288² mask logits.
    pred_masks: Vec<f32>,
    /// Decoder object-score logit (±10 for mask-input frames).
    obj_score: f32,
    /// 256-d object pointer.
    pointer: Vec<f32>,
    mem_feat: Option<Vec<f32>>,
    mem_pos: Option<Vec<f32>>,
}

#[derive(Default)]
struct ObjState {
    cond: BTreeMap<usize, ObjOut>,
    non_cond: BTreeMap<usize, ObjOut>,
}

/// Persistent tracking state across frames (`Sam3VideoInferenceSession`).
pub struct Session {
    /// Frames in the clip (0 = unknown/streaming); the reference caps
    /// `max_object_pointers_to_use` with it.
    pub num_frames: usize,
    /// Ordered object ids — the **track ids** the CLI reports.
    pub obj_ids: Vec<i64>,
    pub max_obj_id: i64,
    objs: HashMap<i64, ObjState>,
    /// Detection score captured when the object was created.
    score: HashMap<i64, f32>,
    /// Frame-wise tracker score (sigmoid of the object-score logit).
    trk_score_fw: HashMap<i64, HashMap<usize, f32>>,
    last_occluded: HashMap<i64, i64>,
    first_frame: HashMap<i64, i64>,
    unmatched: HashMap<i64, Vec<usize>>,
    keep_alive: HashMap<i64, i64>,
    overlap: HashMap<(i64, i64), Vec<usize>>,
    removed: HashSet<i64>,
    suppressed: HashSet<i64>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            // the reference starts `max_obj_id` at -1, so the first object
            // created gets track id 0
            max_obj_id: -1,
            num_frames: 0,
            obj_ids: Vec::new(),
            objs: HashMap::new(),
            score: HashMap::new(),
            trk_score_fw: HashMap::new(),
            last_occluded: HashMap::new(),
            first_frame: HashMap::new(),
            unmatched: HashMap::new(),
            keep_alive: HashMap::new(),
            overlap: HashMap::new(),
            removed: HashSet::new(),
            suppressed: HashSet::new(),
        }
    }
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// Object ids hidden from the outputs (suppressed / hotstart-removed).
    pub fn hidden_ids(&self) -> Vec<i64> {
        self.removed.union(&self.suppressed).copied().collect()
    }
}

// ───────────────────── outputs ─────────────────────

/// One object's observation on one frame.
#[derive(Clone, Debug)]
pub struct TrackObs {
    pub id: i64,
    pub score: f32,
    /// `xyxy` in frame pixels (inclusive corners), from the mask box.
    pub box_px: [f32; 4],
    /// Binary mask at frame resolution, row-major `width × height`.
    pub mask: Vec<u8>,
    /// `true` when this frame's mask came straight from the detector.
    pub from_detection: bool,
    /// `true` when the tracker propagated the object onto this frame.
    pub from_tracker: bool,
}

/// Everything one frame produced.
#[derive(Clone, Debug)]
pub struct FrameOut {
    pub frame: usize,
    pub tracks: Vec<TrackObs>,
    /// Raw detections above threshold this frame (before association).
    pub detections: usize,
    pub presence: f32,
    pub det_ms: f64,
    pub trk_ms: f64,
    pub total_ms: f64,
}

// ───────────────────── the model ─────────────────────

struct Det {
    score: f32,
    mask: Vec<f32>,
}

/// Detector + tracker over one CMF.
pub struct Sam3Video {
    sam: Sam3,
    pub cfg: VidCfg,
}

impl Sam3Video {
    pub fn open(path: &str) -> Result<Self> {
        let sam = Sam3::open(path)?;
        let cfg = VidCfg::from_cmf(sam.cmf())?;
        Ok(Self { sam, cfg })
    }

    /// The image detector (shared prompt text tower, timing probes).
    pub fn detector(&mut self) -> &mut Sam3 {
        &mut self.sam
    }

    fn net(&self) -> Net<'_> {
        Net {
            m: self.sam.cmf(),
            pool: self.sam.pool_handle().map(|p| p.as_ref()),
            cfg: &self.cfg,
        }
    }

    /// Run one frame: detector → propagation → planning → new objects →
    /// outputs. `frame_size` is the original `(width, height)`.
    pub fn forward(
        &mut self,
        s: &mut Session,
        frame_idx: usize,
        pixels: &[f32],
        ids: &[u32],
        attn: &[u8],
        frame_size: (usize, usize),
    ) -> Result<FrameOut> {
        let _prof = crate::cpuprof::spans::dump_on_drop(format!("video frame {frame_idx}"));
        let t0 = Instant::now();
        // ── 1. detector ────────────────────────────────────────────
        let t_fwd = Instant::now();
        let out = self.sam.forward(pixels, ids, attn)?;
        let mut det_ms = t_fwd.elapsed().as_secs_f64() * 1e3;
        let t_det = Instant::now();
        let _pd = crate::cpuprof::spans::span("vid.detect");
        let dets = self.detect(&out)?;
        drop(_pd);
        det_ms += t_det.elapsed().as_secs_f64() * 1e3;
        let presence = sigm(out.presence_logit);

        let net = self.net();
        let t_trk = Instant::now();

        // ── 2. tracker features for this frame ─────────────────────
        let feats = {
            let _p = crate::cpuprof::spans::span("vid.feats");
            net.feats(&out.trunk_feats)?
        };

        // ── 3. propagate the existing objects ──────────────────────
        let pre_ids = s.obj_ids.clone();
        let mut trk_masks: Vec<Vec<f32>> = Vec::new();
        let mut trk_scores: Vec<f32> = Vec::new();
        if !pre_ids.is_empty() {
            let _pp = crate::cpuprof::spans::span("vid.propagate");
            for &id in &pre_ids {
                let (o, _high) = self.propagate(&net, s, id, frame_idx, &feats)?;
                trk_scores.push(o.obj_score);
                trk_masks.push(o.pred_masks.clone());
                self.store(s, id, frame_idx, false, o);
            }
            drop(_pp);
            // Propagation deliberately does NOT encode memory here. The HF
            // pipeline defers it until planning has applied its constraints;
            // `plan` below owns that encode (and execution may replace it when
            // a new detection causes the tracker forward to run again).
            fill_holes(&mut trk_masks, self.cfg.fill_hole_area);
            for (i, &id) in pre_ids.iter().enumerate() {
                let v = sigm(trk_scores[i]);
                s.trk_score_fw.entry(id).or_default().insert(frame_idx, v);
            }
        }

        // ── 4. planning phase ─────────────────────────────────────
        let plan = {
            let _p = crate::cpuprof::spans::span("vid.plan");
            self.plan(&net, s, frame_idx, &dets, &trk_masks, &feats)?
        };

        // ── 5. execution phase: new objects become conditioning ───
        if !plan.new_inds.is_empty() {
            let _pe = crate::cpuprof::spans::span("vid.execute");
            let mut pending: Vec<(i64, ObjOut, Vec<f32>)> = Vec::new();
            for (k, &di) in plan.new_inds.iter().enumerate() {
                let id = plan.new_ids[k];
                let mask: Vec<f32> = dets[di]
                    .mask
                    .iter()
                    .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let (o, high) = self.use_mask_as_output(&net, &feats, &mask)?;
                self.store(s, id, frame_idx, true, o.clone());
                if !s.obj_ids.contains(&id) {
                    s.obj_ids.push(id);
                }
                pending.push((id, o, high));
            }
            // the reference tracker forward re-runs every object that has no
            // conditioning output at this frame (reconditioned ones are reused)
            for &id in &pre_ids {
                if plan.recond_ids.contains(&id) {
                    continue;
                }
                let (o, high) = self.propagate(&net, s, id, frame_idx, &feats)?;
                self.store(s, id, frame_idx, false, o.clone());
                pending.push((id, o, high));
            }
            let any_mask_input = !pending.is_empty();
            self.encode_into(&net, s, frame_idx, pending, &feats.l2, IMG, IMG, any_mask_input)?;
            drop(_pe);
        }

        // ── 6. outputs ────────────────────────────────────────────
        let tracks = {
            let _p = crate::cpuprof::spans::span("vid.outputs");
            self.build_outputs(s, frame_idx, &pre_ids, &dets, &plan, frame_size)?
        };
        let trk_ms = t_trk.elapsed().as_secs_f64() * 1e3;

        for id in &plan.removed {
            s.obj_ids.retain(|x| x != id);
            s.objs.remove(id);
        }
        s.suppressed.extend(plan.suppressed.iter().copied());
        s.removed.extend(plan.removed.iter().copied());

        Ok(FrameOut {
            frame: frame_idx,
            tracks,
            detections: dets.len(),
            presence,
            det_ms,
            trk_ms,
            total_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }

    // ── detector post-processing ─────────────────────────────────

    /// Scores → threshold → mask-IoU NMS → detections.
    fn detect(&self, out: &Sam3Out) -> Result<Vec<Det>> {
        anyhow::ensure!(
            !out.pred_boxes.is_empty() && out.pred_logits.len() == out.pred_boxes.len() / 4,
            "forward returned no scores/masks (SAM3_ONLY debug env set?)"
        );
        let presence = sigm(out.presence_logit);
        let nq = out.pred_boxes.len() / 4;
        let hw = out.pred_masks.len() / nq.max(1);
        anyhow::ensure!(hw == MASK_SIDE * MASK_SIDE, "unexpected mask shape {hw}");
        let thr = self.cfg.score_threshold_detection;
        let mut scores: Vec<f32> = out.pred_logits.iter().map(|&l| sigm(l) * presence).collect();
        let keep_inital: Vec<bool> = scores.iter().map(|&s| s > thr).collect();
        let mut keep = keep_inital.clone();

        if self.cfg.det_nms_thresh > 0.0 {
            let valid: Vec<usize> = (0..nq).filter(|&i| keep[i]).collect();
            if valid.len() > 1 {
                let bin = |q: usize| -> Vec<u8> {
                    out.pred_masks[q * hw..(q + 1) * hw]
                        .iter()
                        .map(|&v| u8::from(v > 0.0))
                        .collect()
                };
                let bins: Vec<(usize, Vec<u8>)> = valid.iter().map(|&q| (q, bin(q))).collect();
                let mut order: Vec<usize> = (0..bins.len()).collect();
                order.sort_by(|a, b| {
                    scores[bins[*a].0]
                        .partial_cmp(&scores[bins[*b].0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut alive = vec![true; bins.len()];
                for (k, &i) in order.iter().enumerate() {
                    if !alive[i] {
                        continue;
                    }
                    for &j in &order[k + 1..] {
                        if alive[j] && mask_iou(&bins[i].1, &bins[j].1) >= self.cfg.det_nms_thresh {
                            alive[j] = false;
                        }
                    }
                }
                for (idx, &(q, _)) in bins.iter().enumerate() {
                    if !alive[idx] {
                        keep[q] = false;
                        scores[q] = 0.0;
                    }
                }
            }
        }
        let mut dets = Vec::new();
        for q in 0..nq {
            if !keep[q] {
                continue;
            }
            dets.push(Det {
                score: scores[q],
                mask: out.pred_masks[q * hw..(q + 1) * hw].to_vec(),
            });
        }
        Ok(dets)
    }

    // ── tracker: one object, one frame ───────────────────────────

    /// Memory-conditioned single-frame inference for an existing object;
    /// returns the stored output plus the 1008² mask logits for the memory
    /// encoder.
    fn propagate(
        &self,
        net: &Net<'_>,
        s: &Session,
        id: i64,
        frame_idx: usize,
        feats: &TrkFeats,
    ) -> Result<(ObjOut, Vec<f32>)> {
        let max_ptr = if s.num_frames > 0 {
            s.num_frames.min(self.cfg.max_object_pointers)
        } else {
            self.cfg.max_object_pointers
        };
        let st = s.objs.get(&id).ok_or_else(|| anyhow!("unknown object {id}"))?;
        let cond = self.select_cond(st, frame_idx);
        let memories = self.gather_memories(st, frame_idx, &cond)?;
        let pointers = self.gather_pointers(st, frame_idx, &cond);
        let pix = Net::pix_rows(feats);
        // memory attention reads the neck's sine PE, the mask decoder the
        // prompt encoder's Gaussian PE
        let neck_pos = Net::neck_pos();
        let pix = {
            let _p = crate::cpuprof::spans::span("vid.mem_cond");
            self.memory_cond(net, &pix, &neck_pos, &memories, &pointers, max_ptr)?
        };
        let pos = net.image_pos()?;

        let dense_p = net.no_mask_prompt()?;
        let sparse = net.empty_sparse()?;
        let dbg = std::env::var("SAM3_VID_DEBUG").is_ok();
        if dbg {
            let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mn = pix.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = pix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[vid] f{frame_idx} obj{id}: memories={} ptr={} pix n={:.3} ({}..{:.3})",
                memories.len(),
                pointers.len(),
                norm(&pix),
                mn,
                mx
            );
        }
        let (masks, iou, obj_score, tokens) = {
            let _p = crate::cpuprof::spans::span("vid.dec");
            net.mask_decoder(
                &sam3::rows_to_chw(&pix, EMB, EMB),
                &pos,
                &dense_p,
                &sparse,
                (&feats.l0, &feats.l1),
                self.cfg.multimask,
            )?
        };
        let n_masks = masks.len() / (MASK_SIDE * MASK_SIDE);
        let best = argmax(&iou).min(n_masks.saturating_sub(1));
        if dbg {
            let (mut cmn, mut cmx, mut cpos) = (f32::INFINITY, f32::NEG_INFINITY, 0usize);
            for &v in masks.iter() {
                cmn = cmn.min(v);
                cmx = cmx.max(v);
                cpos += usize::from(v > 0.0);
            }
            let base = best * MASK_SIDE * MASK_SIDE;
            let spos = masks[base..base + MASK_SIDE * MASK_SIDE].iter().filter(|&&v| v > 0.0).count();
            eprintln!(
                "[vid] f{frame_idx} obj{id}: decoder iou={iou:?} obj_score={obj_score:.4} masks {cmn:.3}..{cmx:.3} pos={cpos}/{} chosen pos={spos}",
                masks.len()
            );
        }
        let low = masks[best * MASK_SIDE * MASK_SIDE..(best + 1) * MASK_SIDE * MASK_SIDE].to_vec();
        let tok = &tokens[best * HID..(best + 1) * HID];
        let appearing = obj_score > 0.0;
        let low = if appearing {
            low
        } else {
            if dbg {
                eprintln!("[vid] f{frame_idx} obj{id}: NOT appearing (obj_score {obj_score}) -> mask forced to NO_OBJ");
            }
            vec![NO_OBJ_SCORE; MASK_SIDE * MASK_SIDE]
        };
        if dbg {
            let pos = low.iter().filter(|&&v| v > 0.0).count();
            eprintln!("[vid] f{frame_idx} obj{id}: propagate mask pos={pos}/{}", low.len());
        }
        let high = {
            let _p = crate::cpuprof::spans::span("vid.resize");
            resize2d(&low, MASK_SIDE, MASK_SIDE, IMG, IMG)
        };
        let pointer = self.object_pointer(net, tok, appearing)?;
        Ok((
            ObjOut { pred_masks: low, obj_score, pointer, mem_feat: None, mem_pos: None },
            high,
        ))
    }

    /// `_use_mask_as_output`: the detection mask IS the output; the decoder
    /// runs once to produce the object pointer.
    fn use_mask_as_output(
        &self,
        net: &Net<'_>,
        feats: &TrkFeats,
        mask: &[f32],
    ) -> Result<(ObjOut, Vec<f32>)> {
        let appearing = mask.iter().any(|&v| v > 0.0);
        let obj_score = if appearing { 10.0 } else { -10.0 };
        // the 288² binary mask is first resampled to the tracker's 1008²
        // input, then scaled to logits (*20-10) — `_use_mask_as_output`
        let hi = resize2d(mask, MASK_SIDE, MASK_SIDE, IMG, IMG);
        let high: Vec<f32> = hi
            .iter()
            .map(|v| v * self.cfg.sigmoid_scale + self.cfg.sigmoid_bias)
            .collect();
        let low = resize2d(&high, IMG, IMG, MASK_SIDE, MASK_SIDE);

        // pointer path: 4× downsample of the 1008² mask → 288² prompt → decoder
        let down = mask_downsample(net, &hi)?;
        let down = resize2d(&down, IMG / 4, IMG / 4, MASK_SIDE, MASK_SIDE);
        let dense_p = net.mask_prompt(&down)?;
        let sparse = net.empty_sparse()?;
        let pix = Net::pix_rows(feats);
        let pos = net.image_pos()?;
        let (_m, iou, dec_score, tokens) = net.mask_decoder(
            &sam3::rows_to_chw(&pix, EMB, EMB),
            &pos,
            &dense_p,
            &sparse,
            (&feats.l0, &feats.l1),
            self.cfg.multimask,
        )?;
        let n = tokens.len() / HID;
        // `_single_frame_forward` uses the same best-IoU token for the
        // object pointer as for the selected mask (not always the last one).
        let best = argmax(&iou).min(n.saturating_sub(1));
        let tok = &tokens[best * HID..(best + 1) * HID];
        let mut pointer = self.object_pointer(net, tok, dec_score > 0.0)?;
        if !appearing {
            pointer = net.v("trk.no_obj_ptr")?;
        }
        Ok((
            ObjOut { pred_masks: low, obj_score, pointer, mem_feat: None, mem_pos: None },
            high,
        ))
    }

    fn object_pointer(&self, net: &Net<'_>, tok: &[f32], appearing: bool) -> Result<Vec<f32>> {
        let p = "trk.obj_ptr_proj.";
        // FeedForward(256,256,256,3): proj_in, layers.0, proj_out
        let mut h = dense(
            &net.w(&format!("{p}layers.0.weight"))?,
            &net.v(&format!("{p}layers.0.bias"))?,
            tok,
            1,
            net.pool,
        );
        h = relu_vec(&h);
        h = dense(
            &net.w(&format!("{p}layers.1.weight"))?,
            &net.v(&format!("{p}layers.1.bias"))?,
            &h,
            1,
            net.pool,
        );
        h = relu_vec(&h);
        let ptr = dense(
            &net.w(&format!("{p}layers.2.weight"))?,
            &net.v(&format!("{p}layers.2.bias"))?,
            &h,
            1,
            net.pool,
        );
        if appearing {
            Ok(ptr)
        } else {
            net.v("trk.no_obj_ptr")
        }
    }

    // ── memory bank ──────────────────────────────────────────────

    /// Conditioning frames closest to `frame_idx`
    /// (`_select_closest_cond_frames`).
    fn select_cond(&self, st: &ObjState, frame_idx: usize) -> Vec<usize> {
        let max = self.cfg.max_cond_frame_num;
        if max == 0 || st.cond.len() <= max {
            return st.cond.keys().copied().collect();
        }
        let mut sel: Vec<usize> = Vec::new();
        if let Some(before) = st.cond.keys().copied().filter(|&t| t < frame_idx).next_back() {
            sel.push(before);
        }
        if let Some(after) = st.cond.keys().copied().filter(|&t| t >= frame_idx).next() {
            sel.push(after);
        }
        let remain = max.saturating_sub(sel.len());
        let mut rest: Vec<usize> = st.cond.keys().copied().filter(|t| !sel.contains(t)).collect();
        rest.sort_by_key(|t| (*t as i64 - frame_idx as i64).abs());
        sel.extend(rest.into_iter().take(remain));
        sel
    }

    /// `(offset, mem_feat, mem_pos)` for every memory frame
    /// (`_gather_memory_frame_outputs` + `_build_memory_attention_inputs`).
    fn gather_memories(
        &self,
        st: &ObjState,
        frame_idx: usize,
        cond: &[usize],
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>)>> {
        let mut out = Vec::new();
        for &t in cond {
            if let Some(o) = st.cond.get(&t) {
                if let (Some(f), Some(p)) = (&o.mem_feat, &o.mem_pos) {
                    out.push((0, f.clone(), p.clone()));
                }
            }
        }
        for off in (1..self.cfg.num_maskmem).rev() {
            let prev = frame_idx as i64 - off as i64;
            if prev < 0 {
                continue;
            }
            if let Some(o) = st.non_cond.get(&(prev as usize)) {
                if let (Some(f), Some(p)) = (&o.mem_feat, &o.mem_pos) {
                    out.push((off, f.clone(), p.clone()));
                }
            }
        }
        Ok(out)
    }

    /// Object pointers of past frames (`_get_object_pointers`).
    fn gather_pointers(&self, st: &ObjState, frame_idx: usize, cond: &[usize]) -> Vec<(usize, Vec<f32>)> {
        let mut offs = Vec::new();
        let mut toks = Vec::new();
        for &t in cond {
            if t > frame_idx {
                continue;
            }
            if let Some(o) = st.cond.get(&t) {
                offs.push(frame_idx - t);
                toks.push(o.pointer.clone());
            }
        }
        for d in 1..self.cfg.max_object_pointers.max(2) {
            let prev = frame_idx as i64 - d as i64;
            if prev < 0 {
                break;
            }
            if let Some(o) = st.non_cond.get(&(prev as usize)) {
                offs.push(d);
                toks.push(o.pointer.clone());
            }
        }
        offs.into_iter().zip(toks).collect()
    }

    /// `_prepare_memory_conditioned_features`: current features fused with
    /// the memory bank through the 4-layer RoPE memory attention.
    fn memory_cond(
        &self,
        net: &Net<'_>,
        pix: &[f32],
        pos: &[f32],
        memories: &[(usize, Vec<f32>, Vec<f32>)],
        pointers: &[(usize, Vec<f32>)],
        max_ptr: usize,
    ) -> Result<Vec<f32>> {
        let n = EMB * EMB;
        if memories.is_empty() {
            // conditioning frame: features + no_memory_embedding (a zero buffer)
            return Ok(pix.to_vec());
        }
        let tpos = net.v("trk.maskmem_tpos_enc")?; // (7, MEM)
        let n_tpos = tpos.len() / MEM;
        let mut mem: Vec<f32> = Vec::new();
        let mut mem_pos: Vec<f32> = Vec::new();
        for (off, f, p) in memories {
            // cond frames (offset 0) index the LAST temporal row — the
            // reference's `tpos[offset - 1]` with negative indexing.
            let row = if *off == 0 { n_tpos - 1 } else { (*off - 1).min(n_tpos - 1) };
            for (i, v) in p.iter().enumerate() {
                mem_pos.push(v + tpos[row * MEM + i % MEM]);
            }
            mem.extend_from_slice(f);
        }
        if !pointers.is_empty() {
            let max_diff = max_ptr.saturating_sub(1).max(1) as f32;
            let offs: Vec<f32> = pointers.iter().map(|(o, _)| *o as f32 / max_diff).collect();
            let pe = sine_1d(&offs, HID);
            let tw = net.w("trk.obj_ptr_tpos_proj.weight")?;
            let tb = net.v("trk.obj_ptr_tpos_proj.bias")?;
            for (i, (_o, p)) in pointers.iter().enumerate() {
                let proj = dense(&tw, &tb, &pe[i * HID..(i + 1) * HID], 1, net.pool);
                let splits = HID / MEM;
                for s in 0..splits {
                    mem.extend_from_slice(&p[s * MEM..(s + 1) * MEM]);
                    mem_pos.extend_from_slice(&proj);
                }
            }
        }
        // Object pointers are keys/values too: their four 64-wide slices are
        // appended after the spatial memory and excluded only from RoPE.
        let mem_len = mem.len() / MEM;

        // RoPE over the 72×72 grid: dim 256, spatial half 128 → 64 freqs
        let head_dim = HID;
        let spatial = head_dim / 2;
        let mut inv = vec![0f32; spatial / 2];
        for i in 0..spatial / 2 {
            inv[i] = 1.0 / 10000f32.powf((2 * i) as f32 / spatial as f32);
        }
        let mut cos = vec![0f32; n * head_dim];
        let mut sin = vec![0f32; n * head_dim];
        for y in 0..EMB {
            for x in 0..EMB {
                let base = (y * EMB + x) * head_dim;
                // position_ids row 0 = w (x), row 1 = h (y)
                for (k, src) in [x, y].iter().enumerate() {
                    for (j, iv) in inv.iter().enumerate() {
                        let ang = *src as f32 * *iv;
                        let dst = base + k * spatial + j * 2; // repeat_interleave(2)
                        cos[dst] = ang.cos();
                        cos[dst + 1] = ang.cos();
                        sin[dst] = ang.sin();
                        sin[dst + 1] = ang.sin();
                    }
                }
            }
        }

        let mut q = pix.to_vec();
        for (a, b) in q.iter_mut().zip(pos) {
            *a += 0.1 * *b;
        }
        let dump_dir = std::env::var("SAM3_VID_DUMP").ok();
        let dump = |name: &str, v: &[f32]| {
            if let Some(d) = &dump_dir {
                let _ = std::fs::create_dir_all(d);
                let _ = std::fs::write(format!("{d}/{name}.f32"), Self::f32_bytes(v));
            }
        };
        dump("initial_seq", &q);
        dump("cos", &cos);
        dump("sin", &sin);
        dump("memory", &mem);
        dump("memory_pos", &mem_pos);
        let tb = "trk.transformer.encoder.";
        for layer in 0..4 {
            let lb = format!("{tb}layers.{layer}.");
            let mut h = q.clone();
            net.ln_rows(&mut h, &format!("{lb}norm1.weight"), &format!("{lb}norm1.bias"), 1e-5)?;
            dump(&format!("l{layer}_norm1"), &h);
            let mut ao = vec![0f32; n * HID];
            net.attn(&format!("{lb}self_attn."), &h, &h, &h, n, n, HID, HID, HID, Some((&cos, &sin, n)), &mut ao)?;
            dump(&format!("l{layer}_self"), &ao);
            for (a, c) in q.iter_mut().zip(&ao) {
                *a += *c;
            }
            dump(&format!("l{layer}_res1"), &q);
            let mut h = q.clone();
            net.ln_rows(&mut h, &format!("{lb}norm2.weight"), &format!("{lb}norm2.bias"), 1e-5)?;
            dump(&format!("l{layer}_norm2"), &h);
            // keys = memory + memory position encoding (kv_in = MEM)
            let k: Vec<f32> = mem
                .chunks(MEM)
                .zip(mem_pos.chunks(MEM))
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y))
                .collect();
            if layer == 0 {
                dump("cross_key", &k);
            }
            let mut ao = vec![0f32; n * HID];
            net.attn(
                &format!("{lb}cross_attn_image."),
                &h,
                &k,
                &mem,
                n,
                mem_len,
                HID,
                MEM,
                HID,
                Some((&cos, &sin, mem_len - pointers.len() * (HID / MEM))),
                &mut ao,
            )?;
            dump(&format!("l{layer}_cross"), &ao);
            for (a, c) in q.iter_mut().zip(&ao) {
                *a += *c;
            }
            dump(&format!("l{layer}_res2"), &q);
            let mut h = q.clone();
            net.ln_rows(&mut h, &format!("{lb}norm3.weight"), &format!("{lb}norm3.bias"), 1e-5)?;
            let a = net.lin(&format!("{lb}"), "linear1", &h, n)?;
            let a = relu_vec(&a);
            let a = net.lin(&format!("{lb}"), "linear2", &a, n)?;
            for (x, c) in q.iter_mut().zip(&a) {
                *x += *c;
            }
            dump(&format!("l{layer}_out"), &q);
        }
        net.ln_rows(&mut q, &format!("{tb}norm.weight"), &format!("{tb}norm.bias"), 1e-5)?;
        dump("final", &q);
        Ok(q)
    }

    /// Memory encoder: mask → stride-16 downsampler, fused with the pixel
    /// features, ConvNeXt fuser, projection to `mem_dim` + sine PE.
    #[allow(dead_code)]
    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    fn encode_memory(
        &self,
        net: &Net<'_>,
        pix: &[f32],        // (256,72,72) CHW
        mask_logits: &[f32], // HxW logits
        h: usize,
        w: usize,
        obj_score: f32,
        from_pts: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut m = resize2d(mask_logits, h, w, MEM_SIDE, MEM_SIDE);
        if from_pts {
            for v in m.iter_mut() {
                *v = if *v > 0.0 { 1.0 } else { 0.0 };
            }
        } else {
            for v in m.iter_mut() {
                *v = sigm(*v);
            }
        }
        for v in m.iter_mut() {
            *v = *v * self.cfg.sigmoid_scale + self.cfg.sigmoid_bias;
        }
        let dbg = std::env::var("SAM3_VID_DEBUG").is_ok();
        if dbg {
            let mn = m.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = m.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let pos = m.iter().filter(|&&v| v > 0.0).count();
            eprintln!("[mem] mask input ({}): {mn:.3}..{mx:.3} pos={}/{} scale={} bias={}", m.len(), pos, m.len(), self.cfg.sigmoid_scale, self.cfg.sigmoid_bias);
        }
        // mask downsampler: 4 × (conv3 s2 + LN + GELU), then 1×1
        let p = "trk.maskmem_backbone.mask_downsampler.encoder.";
        let mut x = m;
        let mut side = MEM_SIDE;
        let mut c_in = 1usize;
        for step in [0usize, 3, 6, 9] {
            let wgt = net.v(&format!("{p}{step}.weight"))?;
            let b = net.v(&format!("{p}{step}.bias"))?;
            x = sam3::conv2d(&wgt, &b, &x, c_in, side, side, 3, 3, 2, 1, net.pool);
            side /= 2;
            let ch = b.len();
            let lw = net.v(&format!("{p}{}.weight", step + 1))?;
            let lb = net.v(&format!("{p}{}.bias", step + 1))?;
            ln_chw(&mut x, ch, side * side, &lw, &lb, 1e-6);
            x.iter_mut().for_each(|v| *v = gelu_erf(*v));
            c_in = ch;
            if dbg {
                let mn = x.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[mem] after layer {step}: side={side} ch={ch} {mn:.3}..{mx:.3}");
            }
        }
        let w12 = net.v(&format!("{p}12.weight"))?;
        let b12 = net.v(&format!("{p}12.bias"))?;
        x = sam3::conv2d(&w12, &b12, &x, c_in, side, side, 1, 1, 1, 0, net.pool);
        anyhow::ensure!(x.len() == HID * EMB * EMB, "mask downsampler output {}", x.len());
        if dbg {
            let rms = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt() / (v.len() as f32).sqrt();
            eprintln!("[mem] mask branch rms={:.4} | pix rms={:.4}", rms(&x), rms(pix));
        }

        // pixel projection + mask
        let pw = net.v("trk.maskmem_backbone.pix_feat_proj.weight")?;
        let pb = net.v("trk.maskmem_backbone.pix_feat_proj.bias")?;
        let mut f = sam3::conv2d(&pw, &pb, pix, HID, EMB, EMB, 1, 1, 1, 0, net.pool);
        if dbg {
            let rms = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt() / (v.len() as f32).sqrt();
            eprintln!("[mem] pix_proj rms={:.4}", rms(&f));
        }
        for (a, b) in f.iter_mut().zip(&x) {
            *a += *b;
        }
        if dbg {
            let rms = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt() / (v.len() as f32).sqrt();
            eprintln!("[mem] fused pre-fuser rms={:.4}", rms(&f));
        }
        let dump_dir = std::env::var("SAM3_VID_DUMP").ok();
        if let Some(d) = &dump_dir {
            let _ = std::fs::create_dir_all(d);
            let _ = std::fs::write(format!("{d}/mask_branch.f32"), Self::f32_bytes(&x));
            let _ = std::fs::write(format!("{d}/pix_proj.f32"), Self::f32_bytes(&f));
        }
        // ConvNeXt fuser ×2 — `out = block_in + gamma * pwconv2(gelu(pwconv1(LN(dwconv(block_in)))))`
        for layer in 0..2 {
            let lp = format!("trk.maskmem_backbone.fuser.layers.{layer}.");
            let block_in = f.clone(); // the residual branch springs from the BLOCK INPUT
            let dw = net.v(&format!("{lp}dwconv.weight"))?;
            let db = net.v(&format!("{lp}dwconv.bias"))?;
            f = depthwise(&dw, &db, &f, HID, EMB, EMB, 7, net.pool);
            let nw = net.v(&format!("{lp}norm.weight"))?;
            let nb = net.v(&format!("{lp}norm.bias"))?;
            ln_chw(&mut f, HID, EMB * EMB, &nw, &nb, 1e-6);
            let ln_out = if dump_dir.is_some() { Some(f.clone()) } else { None };
            // ConvNeXt pointwise convs run over the NHWC layout: rows =
            // spatial positions, columns = channels
            let rows = EMB * EMB;
            let dw_out = if dump_dir.is_some() { Some(f.clone()) } else { None };
            let w1 = net.w(&format!("{lp}pwconv1.weight"))?;
            let b1 = net.v(&format!("{lp}pwconv1.bias"))?;
            let h_rows = sam3::chw_to_rows(&f, EMB, EMB);
            let mut hh = dense(&w1, &b1, &h_rows, rows, net.pool); // (HW, 1024)
            hh.iter_mut().for_each(|v| *v = gelu_erf(*v));
            let pw1_out = if dump_dir.is_some() { Some(hh.clone()) } else { None };
            let w2 = net.w(&format!("{lp}pwconv2.weight"))?;
            let b2 = net.v(&format!("{lp}pwconv2.bias"))?;
            let hh = dense(&w2, &b2, &hh, rows, net.pool); // (HW, 256)
            let g = net.v(&format!("{lp}gamma"))?;
            if let Some(d) = &dump_dir {
                let _ = std::fs::write(format!("{d}/fuser_gamma{layer}.f32"), Self::f32_bytes(&g));
            }
            for c in 0..HID {
                for i in 0..rows {
                    f[c * rows + i] = block_in[c * rows + i] + g[c] * hh[i * HID + c];
                }
            }
            if let Some(d) = &dump_dir {
                let _ = std::fs::write(format!("{d}/fuser_layer{layer}.f32"), Self::f32_bytes(&f));
                if layer == 0 {
                    if let Some(v) = &dw_out {
                        let _ = std::fs::write(format!("{d}/f0_dwconv.f32"), Self::f32_bytes(v));
                    }
                    if let Some(v) = &ln_out {
                        let _ = std::fs::write(format!("{d}/f0_norm.f32"), Self::f32_bytes(v));
                    }
                    if let Some(v) = &pw1_out {
                        let _ = std::fs::write(format!("{d}/f0_pw1.f32"), Self::f32_bytes(v));
                    }
                    let _ = std::fs::write(format!("{d}/f0_pw2.f32"), Self::f32_bytes(&hh));
                }
            }
        }
        if std::env::var("SAM3_VID_DEBUG").is_ok() {
            let rms = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt() / (v.len() as f32).sqrt();
            eprintln!("[mem] post-fuser rms={:.4}", rms(&f));
        }
        let ow = net.v("trk.maskmem_backbone.out_proj.weight")?;
        let ob = net.v("trk.maskmem_backbone.out_proj.bias")?;
        let mut f = sam3::conv2d(&ow, &ob, &f, HID, EMB, EMB, 1, 1, 1, 0, net.pool);
        anyhow::ensure!(f.len() == MEM * EMB * EMB, "memory projection {}", f.len());
        if self.cfg.occlusion_spatial && obj_score <= 0.0 {
            let e = net.v("trk.no_obj_embed_spatial")?;
            let hw = EMB * EMB;
            for i in 0..hw {
                for (c, ev) in e.iter().enumerate() {
                    f[c * hw + i] += ev;
                }
            }
        }
        let pos = sine(EMB, EMB, MEM / 2);
        let rows = sam3::chw_to_rows(&f, EMB, EMB);
        Ok((rows, pos))
    }

    // ── storage helpers ──────────────────────────────────────────

    fn store(&self, s: &mut Session, id: i64, frame_idx: usize, cond: bool, o: ObjOut) {
        let st = s.objs.entry(id).or_default();
        if cond {
            st.cond.insert(frame_idx, o);
        } else {
            st.non_cond.insert(frame_idx, o);
        }
    }

    /// Encode memories for the outputs just written
    /// (`_batch_encode_memories`).
    fn encode_into(
        &self,
        net: &Net<'_>,
        s: &mut Session,
        frame_idx: usize,
        pending: Vec<(i64, ObjOut, Vec<f32>)>,
        pix: &[f32],
        h: usize,
        w: usize,
        from_pts: bool,
    ) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let _p = crate::cpuprof::spans::span("vid.encode");
        for (id, o, high) in pending {
            let (mf, mp) = self.encode_memory(net, pix, &high, h, w, o.obj_score, from_pts)?;
            let st = s.objs.entry(id).or_default();
            let slot = if st.cond.contains_key(&frame_idx) {
                st.cond.get_mut(&frame_idx)
            } else {
                st.non_cond.get_mut(&frame_idx)
            };
            if let Some(slot) = slot {
                slot.mem_feat = Some(mf);
                slot.mem_pos = Some(mp);
            }
        }
        Ok(())
    }
}

// ───────────────────── planning ─────────────────────

struct Plan {
    new_inds: Vec<usize>,
    new_ids: Vec<i64>,
    recond_ids: HashSet<i64>,
    removed: HashSet<i64>,
    suppressed: HashSet<i64>,
}

impl Sam3Video {
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &self,
        net: &Net<'_>,
        s: &mut Session,
        frame_idx: usize,
        dets: &[Det],
        trk_masks: &[Vec<f32>],
        feats: &TrkFeats,
    ) -> Result<Plan> {
        let cfg = &self.cfg;
        let pre_ids: Vec<i64> = s.obj_ids.clone();
        let det_masks: Vec<&[f32]> = dets.iter().map(|d| d.mask.as_slice()).collect();
        let det_scores: Vec<f32> = dets.iter().map(|d| d.score).collect();
        let assoc = associate(&det_masks, &det_scores, trk_masks, &pre_ids, cfg);
        if std::env::var("SAM3_VID_DEBUG").is_ok() {
            let trk_pos: Vec<usize> = trk_masks.iter().map(|m| m.iter().filter(|&&v| v > 0.0).count()).collect();
            let det_pos: Vec<usize> = det_masks.iter().map(|m| m.iter().filter(|&&v| v > 0.0).count()).collect();
            eprintln!(
                "[vid] f{frame_idx} plan: dets={} (pos {:?}, scores {:?}) trk={} (pos {:?}) new={:?} unmatched={:?} empty={:?}",
                det_masks.len(), det_pos, det_scores, trk_masks.len(), trk_pos,
                assoc.new_inds, assoc.unmatched, assoc.empty
            );
        }

        // object-id budget
        let mut new_inds = assoc.new_inds;
        if pre_ids.len() + new_inds.len() > cfg.max_num_objects {
            let room = cfg.max_num_objects.saturating_sub(pre_ids.len());
            let mut by_score = new_inds.clone();
            by_score.sort_by(|a, b| {
                det_scores[*b].partial_cmp(&det_scores[*a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            new_inds = by_score.into_iter().take(room).collect();
            new_inds.sort_unstable();
        }
        let start = s.max_obj_id + 1;
        let new_ids: Vec<i64> = (0..new_inds.len() as i64).map(|k| start + k).collect();

        // ── hotstart (`_process_hotstart`) ─────────────────────────
        let mut removed: HashSet<i64> = HashSet::new();
        let mut suppressed: HashSet<i64> = HashSet::new();
        for (k, &di) in new_inds.iter().enumerate() {
            let id = new_ids[k];
            let _ = di;
            s.first_frame.entry(id).or_insert(frame_idx as i64);
            s.keep_alive.insert(id, cfg.init_trk_keep_alive);
        }
        let matched: HashSet<i64> = assoc.det_to_matched.values().flatten().copied().collect();
        for id in &matched {
            let ka = s.keep_alive.get(id).copied().unwrap_or(cfg.init_trk_keep_alive);
            s.keep_alive.insert(*id, cfg.max_trk_keep_alive.max(ka + 1));
        }
        for id in &assoc.unmatched {
            s.unmatched.entry(*id).or_default().push(frame_idx);
            let ka = s.keep_alive.get(id).copied().unwrap_or(cfg.init_trk_keep_alive);
            s.keep_alive.insert(*id, cfg.min_trk_keep_alive.max(ka - 1));
        }
        if cfg.decrease_keep_alive_for_empty {
            for id in &assoc.empty {
                let ka = s.keep_alive.get(id).copied().unwrap_or(cfg.init_trk_keep_alive);
                s.keep_alive.insert(*id, cfg.min_trk_keep_alive.max(ka - 1));
            }
        }
        let hotstart_diff = frame_idx as i64 - cfg.hotstart_delay;
        let already_removed: HashSet<i64> = s.removed.clone();
        for (id, frames) in s.unmatched.clone() {
            if already_removed.contains(&id) || removed.contains(&id) {
                continue;
            }
            if frames.len() >= cfg.hotstart_unmatch_thresh {
                if s.first_frame.get(&id).copied().unwrap_or(0) > hotstart_diff {
                    removed.insert(id);
                }
            }
            let ka = s.keep_alive.get(&id).copied().unwrap_or(0);
            if ka <= 0
                && !cfg.suppress_unmatched_only_within_hotstart
                && !already_removed.contains(&id)
                && !removed.contains(&id)
            {
                suppressed.insert(id);
            }
        }
        for ids in assoc.det_to_matched.values() {
            if ids.len() < 2 {
                continue;
            }
            let first = *ids
                .iter()
                .min_by_key(|i| s.first_frame.get(i).copied().unwrap_or(i64::MAX))
                .unwrap();
            for id in ids {
                if *id != first {
                    s.overlap.entry((first, *id)).or_default().push(frame_idx);
                }
            }
        }
        let mut pairs: Vec<((i64, i64), Vec<usize>)> = s.overlap.clone().into_iter().collect();
        pairs.sort();
        for ((_, id), frames) in pairs {
            if s.removed.contains(&id) || removed.contains(&id) {
                continue;
            }
            if s.first_frame.get(&id).copied().unwrap_or(0) > hotstart_diff
                && frames.len() >= cfg.hotstart_dup_thresh
            {
                removed.insert(id);
            }
        }
        s.removed.extend(removed.iter().copied());

        // ── periodic reconditioning (`_prepare_recondition_masks`) ──
        let mut recond_ids: HashSet<i64> = HashSet::new();
        let mut recond_masks: Vec<(usize, Vec<f32>)> = Vec::new();
        if cfg.recondition_every_nth_frame > 0
            && frame_idx as i64 % cfg.recondition_every_nth_frame == 0
            && !assoc.trk_to_high_conf.is_empty()
        {
            for (&id, &di) in &assoc.trk_to_high_conf {
                let Some(i) = pre_ids.iter().position(|x| *x == id) else { continue };
                let trk_score = s
                    .trk_score_fw
                    .get(&id)
                    .and_then(|m| m.get(&frame_idx))
                    .copied()
                    .unwrap_or(0.0);
                if trk_score <= cfg.high_conf_thresh {
                    continue;
                }
                recond_ids.insert(id);
                if cfg.recondition_on_trk_masks {
                    recond_masks.push((i, trk_masks[i].clone()));
                } else {
                    recond_masks.push((
                        i,
                        dets[di].mask.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect(),
                    ));
                }
            }
        }

        // ── recent-occlusion suppression ────────────────────────────
        let mut trk = trk_masks.to_vec();
        if cfg.suppress_recent_occlusion_thresh > 0.0 && !trk.is_empty() {
            let bins: Vec<Vec<u8>> = trk
                .iter()
                .map(|m| m.iter().map(|&v| u8::from(v > 0.0)).collect())
                .collect();
            let n = bins.len();
            let mut last: Vec<i64> = pre_ids
                .iter()
                .map(|id| {
                    s.last_occluded.get(id).copied().unwrap_or(if s.removed.contains(id) {
                        100_000
                    } else {
                        -1
                    })
                })
                .collect();
            let mut to_suppress = vec![false; n];
            if n > 1 {
                for i in 0..n {
                    for j in (i + 1)..n {
                        if mask_iou(&bins[i], &bins[j]) < cfg.suppress_recent_occlusion_thresh {
                            continue;
                        }
                        if last[i] > last[j] && last[j] > -1 {
                            to_suppress[i] = true;
                        }
                        if last[j] > last[i] && last[i] > -1 {
                            to_suppress[j] = true;
                        }
                    }
                }
            }
            for (i, m) in trk.iter_mut().enumerate() {
                if bins[i].iter().all(|&v| v == 0) || to_suppress[i] {
                    for v in m.iter_mut() {
                        *v = -10.0;
                    }
                    last[i] = frame_idx as i64;
                }
            }
            for (i, id) in pre_ids.iter().enumerate() {
                s.last_occluded.insert(*id, last[i]);
            }
        }

        // ── `_tracker_update_memories`: plan → constraints → encode ─
        let mut per_obj = trk;
        for (i, m) in &recond_masks {
            per_obj[*i] = m.clone();
        }
        for (i, &id) in pre_ids.iter().enumerate() {
            if recond_ids.contains(&id) {
                if let Some(st) = s.objs.get_mut(&id) {
                    if let Some(o) = st.non_cond.remove(&frame_idx) {
                        st.cond.insert(frame_idx, o);
                    }
                }
            }
        }
        let per_obj = suppress_area_shrink(&per_obj);
        if !per_obj.is_empty() {
            let mut pending = Vec::new();
            for (i, &id) in pre_ids.iter().enumerate() {
                let area = per_obj[i].iter().any(|&v| v > 0.0);
                let score = if area { 10.0 } else { -10.0 };
                if let Some(st) = s.objs.get_mut(&id) {
                    let slot = if st.cond.contains_key(&frame_idx) {
                        st.cond.get_mut(&frame_idx)
                    } else {
                        st.non_cond.get_mut(&frame_idx)
                    };
                    if let Some(slot) = slot {
                        slot.pred_masks = per_obj[i].clone();
                        slot.obj_score = score;
                    }
                }
                pending.push((
                    id,
                    ObjOut {
                        pred_masks: per_obj[i].clone(),
                        obj_score: score,
                        pointer: Vec::new(),
                        mem_feat: None,
                        mem_pos: None,
                    },
                    per_obj[i].clone(),
                ));
            }
            self.encode_into(net, s, frame_idx, pending, &feats.l2, MASK_SIDE, MASK_SIDE, false)?;
        }

        // ── metadata updates ───────────────────────────────────────
        for (k, &id) in new_ids.iter().enumerate() {
            s.score.insert(id, det_scores[new_inds[k]]);
            s.trk_score_fw.entry(id).or_default().insert(frame_idx, det_scores[new_inds[k]]);
        }
        for id in &removed {
            s.score.insert(*id, -1e4);
            s.trk_score_fw.entry(*id).or_default().insert(frame_idx, -1e4);
        }
        s.max_obj_id = (start + new_ids.len() as i64 - 1).max(s.max_obj_id);

        Ok(Plan { new_inds, new_ids, recond_ids, removed, suppressed })
    }
}

// ───────────────────── association ─────────────────────

struct Assoc {
    new_inds: Vec<usize>,
    unmatched: Vec<i64>,
    empty: Vec<i64>,
    det_to_matched: HashMap<usize, Vec<i64>>,
    trk_to_high_conf: HashMap<i64, usize>,
}

/// `_associate_det_trk`: detections against the propagated masklets.
fn associate(
    det_masks: &[&[f32]],
    det_scores: &[f32],
    trk_masks: &[Vec<f32>],
    trk_ids: &[i64],
    cfg: &VidCfg,
) -> Assoc {
    let blank = Assoc {
        new_inds: Vec::new(),
        unmatched: Vec::new(),
        empty: Vec::new(),
        det_to_matched: HashMap::new(),
        trk_to_high_conf: HashMap::new(),
    };
    if trk_masks.is_empty() {
        // no tracks yet: every detection above threshold starts an object
        return Assoc { new_inds: (0..det_masks.len()).collect(), ..blank };
    }
    if det_masks.is_empty() {
        let mut unmatched = Vec::new();
        let mut empty = Vec::new();
        for (i, m) in trk_masks.iter().enumerate() {
            if m.iter().any(|&v| v > 0.0) {
                unmatched.push(trk_ids[i]);
            } else {
                empty.push(trk_ids[i]);
            }
        }
        return Assoc { unmatched, empty, ..blank };
    }
    let dbin: Vec<Vec<u8>> = det_masks
        .iter()
        .map(|m| m.iter().map(|&v| u8::from(v > 0.0)).collect())
        .collect();
    let tbin: Vec<Vec<u8>> = trk_masks
        .iter()
        .map(|m| m.iter().map(|&v| u8::from(v > 0.0)).collect())
        .collect();
    let mut ious = vec![0f32; det_masks.len() * trk_masks.len()];
    for (i, d) in dbin.iter().enumerate() {
        for (j, t) in tbin.iter().enumerate() {
            ious[i * trk_masks.len() + j] = mask_iou(d, t);
        }
    }
    let at = |i: usize, j: usize| ious[i * trk_masks.len() + j];

    let mut unmatched = Vec::new();
    let mut empty = Vec::new();
    for (j, t) in tbin.iter().enumerate() {
        let nonempty = t.iter().any(|&v| v != 0);
        let matched = (0..det_masks.len()).any(|i| at(i, j) >= cfg.trk_assoc_iou_thresh);
        if !nonempty {
            empty.push(trk_ids[j]);
        } else if !matched {
            unmatched.push(trk_ids[j]);
        }
    }
    let mut new_inds = Vec::new();
    let mut det_to_matched = HashMap::new();
    let mut trk_to_high_conf = HashMap::new();
    for i in 0..det_masks.len() {
        let matches_any = (0..trk_masks.len()).any(|j| at(i, j) >= cfg.assoc_iou_thresh);
        let is_new = det_scores[i] >= cfg.new_det_thresh && !matches_any;
        if is_new {
            new_inds.push(i);
        }
        let ms: Vec<i64> = (0..trk_masks.len())
            .filter(|&j| at(i, j) >= cfg.assoc_iou_thresh)
            .map(|j| trk_ids[j])
            .collect();
        let mut best = (0usize, 0f32);
        for j in 0..trk_masks.len() {
            if at(i, j) > best.1 {
                best = (j, at(i, j));
            }
        }
        if det_scores[i] >= cfg.high_conf_thresh && !is_new && best.1 >= cfg.high_iou_thresh {
            trk_to_high_conf.insert(trk_ids[best.0], i);
        }
        det_to_matched.insert(i, ms);
    }
    Assoc { new_inds, unmatched, empty, det_to_matched, trk_to_high_conf }
}

// ───────────────────── outputs ─────────────────────

impl Sam3Video {
    #[allow(clippy::too_many_arguments)]
    fn build_outputs(
        &self,
        s: &Session,
        frame_idx: usize,
        pre_ids: &[i64],
        dets: &[Det],
        plan: &Plan,
        frame_size: (usize, usize),
    ) -> Result<Vec<TrackObs>> {
        let (fw, fh) = frame_size;
        let mut out: Vec<TrackObs> = Vec::new();
        // tracker masks of the objects that existed before this frame
        for &id in pre_ids {
            if plan.removed.contains(&id) {
                continue;
            }
            let Some(st) = s.objs.get(&id) else { continue };
            let Some(o) = st.cond.get(&frame_idx).or_else(|| st.non_cond.get(&frame_idx)) else {
                if std::env::var("SAM3_VID_DEBUG").is_ok() {
                    eprintln!("[vid] f{frame_idx} out: obj{id} has NO output at this frame");
                }
                continue;
            };
            if o.pred_masks.len() != MASK_SIDE * MASK_SIDE {
                if std::env::var("SAM3_VID_DEBUG").is_ok() {
                    eprintln!("[vid] f{frame_idx} out: obj{id} mask len {} != {}", o.pred_masks.len(), MASK_SIDE * MASK_SIDE);
                }
                continue;
            }
            if o.pred_masks.iter().all(|&v| v <= 0.0) {
                if std::env::var("SAM3_VID_DEBUG").is_ok() {
                    eprintln!("[vid] f{frame_idx} out: obj{id} mask is ALL-NEGATIVE (min {:.3}) -> no observation",
                        o.pred_masks.iter().cloned().fold(f32::INFINITY, f32::min));
                }
                continue;
            }
            let score = s.score.get(&id).copied().unwrap_or(0.0);
            push_obs(&mut out, id, score, &o.pred_masks, fw, fh, false, true);
        }
        // masks of objects created on this frame come from the detector
        for (k, &di) in plan.new_inds.iter().enumerate() {
            let id = plan.new_ids[k];
            if plan.removed.contains(&id) {
                continue;
            }
            let mut m = dets[di].mask.clone();
            fill_holes_one(&mut m, self.cfg.fill_hole_area);
            let score = s.score.get(&id).copied().unwrap_or(dets[di].score);
            push_obs(&mut out, id, score, &m, fw, fh, true, false);
        }
        // hidden objects never reach the outputs
        let hidden: HashSet<i64> = s.removed.union(&s.suppressed).copied().collect();
        out.retain(|o| !hidden.contains(&o.id));
        if out.len() > 1 {
            apply_non_overlap(&mut out);
        }
        out.sort_by_key(|o| o.id);
        Ok(out)
    }
}

fn push_obs(
    out: &mut Vec<TrackObs>,
    id: i64,
    score: f32,
    logits: &[f32],
    fw: usize,
    fh: usize,
    from_detection: bool,
    from_tracker: bool,
) {
    if logits.len() != MASK_SIDE * MASK_SIDE {
        eprintln!("[vid] push_obs: id {id} logits len {} != {MASK_SIDE}", logits.len());
        return;
    }
    let flat = resize2d(logits, MASK_SIDE, MASK_SIDE, fh, fw);
    let mask: Vec<u8> = flat.iter().map(|&v| u8::from(v > 0.0)).collect();
    if mask.iter().all(|&v| v == 0) {
        eprintln!("[vid] push_obs: id {id} empty after resize to {fw}x{fh}");
        return;
    }
    let mut x0 = usize::MAX;
    let mut y0 = usize::MAX;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for y in 0..fh {
        for x in 0..fw {
            if mask[y * fw + x] != 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    out.push(TrackObs {
        id,
        score,
        box_px: [x0 as f32, y0 as f32, x1 as f32, y1 as f32],
        mask,
        from_detection,
        from_tracker,
    });
}

/// Pixel-wise "highest score wins" between objects of one prompt group
/// (`Sam3VideoProcessor._apply_object_wise_non_overlapping_constraints`).
fn apply_non_overlap(out: &mut [TrackObs]) {
    let len = match out.first() {
        Some(o) => o.mask.len(),
        None => return,
    };
    for i in 0..len {
        let mut best = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for (k, o) in out.iter().enumerate() {
            if o.mask[i] == 0 {
                continue;
            }
            if o.score > best_score {
                best_score = o.score;
                best = k;
            }
        }
        if best == usize::MAX {
            continue;
        }
        for (k, o) in out.iter_mut().enumerate() {
            if k != best && o.mask[i] != 0 {
                o.mask[i] = 0;
            }
        }
    }
}

// ───────────────────── mask helpers ─────────────────────

/// IoU of two binary masks (empty ∪ empty = 1, as in the reference).
pub fn mask_iou(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut inter = 0u32;
    let mut union = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        let x = x != 0;
        let y = y != 0;
        inter += u32::from(x && y);
        union += u32::from(x || y);
    }
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

/// `fill_holes_in_mask_scores`: small background components become 0.1,
/// small foreground components (≤ max_area and ≤ half the mask) become -0.1.
fn fill_holes(masks: &mut [Vec<f32>], max_area: usize) {
    for m in masks.iter_mut() {
        fill_holes_one(m, max_area);
    }
}

fn fill_holes_one(mask: &mut [f32], max_area: usize) {
    if max_area == 0 || mask.len() != MASK_SIDE * MASK_SIDE {
        return;
    }
    let (w, h) = (MASK_SIDE, MASK_SIDE);
    let bg: Vec<bool> = mask.iter().map(|&v| v <= 0.0).collect();
    let areas = cc_areas(&bg, w, h);
    for i in 0..mask.len() {
        if bg[i] && areas[i] > 0 && areas[i] <= max_area {
            mask[i] = 0.1;
        }
    }
    let fg: Vec<bool> = mask.iter().map(|&v| v > 0.0).collect();
    let total: usize = fg.iter().filter(|&&v| v).count();
    let thr = (total / 2).min(max_area);
    let areas = cc_areas(&fg, w, h);
    for i in 0..mask.len() {
        if fg[i] && areas[i] > 0 && areas[i] <= thr {
            mask[i] = -0.1;
        }
    }
}

/// 4-connected component sizes (every pixel carries its component's area).
fn cc_areas(mask: &[bool], w: usize, h: usize) -> Vec<usize> {
    let mut label = vec![usize::MAX; mask.len()];
    let mut sizes: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || label[start] != usize::MAX {
            continue;
        }
        let id = sizes.len();
        stack.clear();
        stack.push(start);
        label[start] = id;
        let mut count = 0usize;
        while let Some(p) = stack.pop() {
            count += 1;
            let (x, y) = (p % w, p / w);
            for (nx, ny) in
                [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)]
            {
                if nx >= w || ny >= h {
                    continue;
                }
                let q = ny * w + nx;
                if mask[q] && label[q] == usize::MAX {
                    label[q] = id;
                    stack.push(q);
                }
            }
        }
        sizes.push(count);
    }
    label
        .into_iter()
        .map(|l| if l == usize::MAX { 0 } else { sizes[l] })
        .collect()
}

/// `_suppress_object_pw_area_shrinkage`: pixel-wise non-overlap first, then
/// drop the objects whose foreground area shrinks below 30 %.
fn suppress_area_shrink(masks: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if masks.len() <= 1 {
        return masks.to_vec();
    }
    let len = masks[0].len();
    let mut non_overlap: Vec<Vec<f32>> = masks.to_vec();
    for i in 0..len {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (k, m) in masks.iter().enumerate() {
            let v = if m[i] > 0.0 { m[i] } else { -10.0 };
            if v > best_v {
                best_v = v;
                best = k;
            }
        }
        for (k, m) in non_overlap.iter_mut().enumerate() {
            if k != best && m[i] > 0.0 {
                m[i] = -10.0;
            }
        }
    }
    let mut out = masks.to_vec();
    for (k, m) in masks.iter().enumerate() {
        let before = m.iter().filter(|&&v| v > 0.0).count();
        let after = non_overlap[k].iter().filter(|&&v| v > 0.0).count();
        if (after as f32) / (before.max(1) as f32) < 0.3 {
            for v in out[k].iter_mut() {
                if *v > -10.0 {
                    *v = -10.0;
                }
            }
        }
    }
    out
}

fn relu_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.max(0.0)).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// Depthwise conv (`w [c, 1, k, k]`, `groups = c`, padding k/2).
fn depthwise(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    c: usize,
    h: usize,
    wid: usize,
    k: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let pad = k / 2;
    let mut out = vec![0f32; c * h * wid];
    let ob = out.as_mut_ptr() as usize;
    let run = |s: usize, e: usize| {
        let o = unsafe { std::slice::from_raw_parts_mut(ob as *mut f32, c * h * wid) };
        for idx in s..e {
            let ch = idx / (h * wid);
            let rem = idx % (h * wid);
            let (y, x2) = (rem / wid, rem % wid);
            let mut acc = b[ch];
            for dy in 0..k {
                let iy = y as isize + dy as isize - pad as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for dx in 0..k {
                    let ix = x2 as isize + dx as isize - pad as isize;
                    if ix < 0 || ix >= wid as isize {
                        continue;
                    }
                    acc += w[(ch * k + dy) * k + dx]
                        * x[ch * h * wid + iy as usize * wid + ix as usize];
                }
            }
            o[idx] = acc;
        }
    };
    match pool {
        Some(p) => p.run_rows(c * h * wid, &run),
        None => run(0, c * h * wid),
    }
    out
}

/// `trk.mask_downsample`: 1008² → 252² (conv k4 s4).
fn mask_downsample(net: &Net<'_>, mask: &[f32]) -> Result<Vec<f32>> {
    let w = net.v("trk.mask_downsample.weight")?;
    let b = net.v("trk.mask_downsample.bias")?;
    Ok(sam3::conv2d(&w, &b, mask, 1, IMG, IMG, 4, 4, 4, 0, net.pool))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmf_path() -> Option<std::path::PathBuf> {
        [" /home/hermes/models/sam3-q8_2f.cmf"]
            .iter()
            .map(|s| std::path::PathBuf::from(s.trim()))
            .find(|p| p.exists())
    }

    /// The tracker's shape-sensitive half (prompt encoder, mask decoder,
    /// memory encoder, memory attention, object pointers) without paying for
    /// a detector forward: fabricate the neck features, condition one frame
    /// from a synthetic detection mask, then propagate one frame.
    #[test]
    fn tracker_memory_smoke() {
        let Some(path) = cmf_path() else {
            eprintln!("SKIP tracker_memory_smoke: cmf not found");
            return;
        };
        let m = Sam3Video::open(path.to_str().expect("path")).expect("open video model");
        let net = m.net();

        // deterministic synthetic neck features
        let fill = |len: usize, k: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let h = (i as u32).wrapping_mul(2654435761) >> 16;
                    (h % 1000) as f32 / 1000.0 * k
                })
                .collect()
        };
        let feats = TrkFeats {
            l0: fill(32 * MASK_SIDE * MASK_SIDE, 0.05),
            l1: fill(64 * 144 * 144, 0.05),
            l2: fill(HID * EMB * EMB, 0.05),
        };

        // synthetic detection: the square at (48..144)² of the 288² grid
        let mut mask = vec![0f32; MASK_SIDE * MASK_SIDE];
        for y in 48..144 {
            for x in 48..144 {
                mask[y * MASK_SIDE + x] = 1.0;
            }
        }
        let mut s = Session::new();
        s.obj_ids.push(0);
        s.max_obj_id = 0;

        let (o, high) = m.use_mask_as_output(&net, &feats, &mask).expect("mask-as-output");
        assert_eq!(o.pred_masks.len(), MASK_SIDE * MASK_SIDE, "low-res mask shape");
        assert_eq!(o.pointer.len(), HID, "object pointer width");
        assert_eq!(high.len(), IMG * IMG, "high-res mask shape");
        assert!(o.obj_score > 0.0, "object score {}", o.obj_score);
        let area = 96 * 96;
        let pos = o.pred_masks.iter().filter(|&&v| v > 0.0).count();
        assert!(
            (pos as i64 - area as i64).abs() <= area as i64 / 4,
            "mask-as-output area {pos} vs input {area}"
        );

        // conditioning frame + memory encode
        m.store(&mut s, 0, 0, true, o.clone());
        let pix = feats.l2.clone();
        m.encode_into(&net, &mut s, 0, vec![(0, o.clone(), high)], &pix, IMG, IMG, true)
            .expect("memory encode");
        assert!(s.objs[&0].cond[&0].mem_feat.is_some(), "memory features stored");
        assert_eq!(s.objs[&0].cond[&0].mem_feat.as_ref().unwrap().len(), EMB * EMB * MEM);

        // propagate one frame: memory attention + decoder
        let (o1, high1) = m.propagate(&net, &s, 0, 1, &feats).expect("propagate");
        assert_eq!(o1.pred_masks.len(), MASK_SIDE * MASK_SIDE);
        assert_eq!(high1.len(), IMG * IMG);
        let pos1 = o1.pred_masks.iter().filter(|&&v| v > 0.0).count();
        let mn = o1.pred_masks.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = o1.pred_masks.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "tracker smoke: mask-as-output {} px, propagated {} px (logit {:.2}..{:.2}), obj_score {:.3} -> {:.3}",
            pos,
            pos1,
            mn,
            mx,
            o.obj_score,
            o1.obj_score
        );
        // the neck features here are synthetic noise, so the propagated mask
        // carries no structure — what this test guards is the SHAPE of every
        // stage (prompt encoder, mask decoder, memory encoder, memory
        // attention, object pointers) plus the object-score sign flip.
        assert!(o1.obj_score.is_finite(), "object score must be finite");
        assert!(mn.is_finite() && mx.is_finite(), "mask logits must be finite");
        assert!(mx > mn, "the decoder must not emit a constant mask ({mn})");
    }

    fn f32s(p: &std::path::Path) -> Vec<f32> {
        let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn fx_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sam3_video")
    }

    /// Reference-anchored tracker check (no detector forward — the fixture
    /// supplies the real neck features, memory bank and object pointer):
    /// 1. the mask decoder on the frame-0 memory prompt must reproduce the
    ///    reference object pointer;
    /// 2. memory conditioning + decoder on frame 1 must reproduce the
    ///    reference object-score logit (+10.1, "the object is there").
    #[test]
    fn tracker_fixture_parity() {
        let fx = fx_dir();
        let Some(path) = cmf_path() else {
            eprintln!("SKIP tracker_fixture_parity: cmf not found");
            return;
        };
        if !fx.join("f1/obj0/non_cond_frame_outputs/object_score_logits.f32").exists() {
            eprintln!("SKIP tracker_fixture_parity: fixtures missing at {}", fx.display());
            return;
        }
        let m = Sam3Video::open(path.to_str().unwrap()).expect("open");
        let net = m.net();
        let fill = |len: usize| vec![0f32; len];
        let feats0 = TrkFeats { l0: fill(32 * MASK_SIDE * MASK_SIDE), l1: fill(64 * 144 * 144), l2: fill(HID * EMB * EMB) };
        let _ = &feats0;

        // ── 1. frame 0: decoder with the detection mask prompt → pointer ──
        let pix0 = f32s(&fx.join("f0/neck_l2.f32"));
        assert_eq!(pix0.len(), EMB * EMB * HID);
        let pix0_chw = pix0_to_chw(&pix0); // the fixture stores (HW, C) rows
        let hi0 = f32s(&fx.join("f0/obj0/cond_frame_outputs/high_res_masks.f32"));
        let bin: Vec<f32> = hi0.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
        assert_eq!(bin.len(), IMG * IMG);
        let ds = mask_downsample(&net, &bin).expect("mask_downsample");
        let ds = crate::sam3vid_net::resize2d(&ds, IMG / 4, IMG / 4, MASK_SIDE, MASK_SIDE);
        let dense = net.mask_prompt(&ds).expect("mask prompt");
        let sparse = net.empty_sparse().expect("sparse");
        let pos = net.image_pos().expect("pos");
        let (_m, iou0, sc0, toks) = net
            .mask_decoder(&pix0_to_chw(&pix0), &pos, &dense, &sparse, (&feats0.l0, &feats0.l1), true)
            .expect("decoder f0");
        let n = toks.len() / HID;
        let best = argmax(&iou0).min(n - 1);
        let ptr = m.object_pointer(&net, &toks[best * HID..(best + 1) * HID], sc0 > 0.0).expect("pointer");
        let want = f32s(&fx.join("f0/obj0/cond_frame_outputs/object_pointer.f32"));
        let cos = dot(&ptr, &want) / (norm(&ptr) * norm(&want) + 1e-9);
        eprintln!("fixture f0 pointer: cos={cos:.4} obj_score={sc0:.3} (ref mask prompt)");
        assert!(cos >= 0.99, "frame-0 object pointer cosine {cos:.4} vs reference");

        // ── 2. frame 1: memory conditioning + decoder → object score ──
        let pix1 = f32s(&fx.join("f1/neck_l2.f32"));
        let mem_feat = f32s(&fx.join("f0/obj0/cond_frame_outputs/maskmem_features.f32"));
        let mem_pos = f32s(&fx.join("f0/obj0/cond_frame_outputs/maskmem_pos_enc.f32"));
        let mem_feat_ref = mem_feat.clone();
        let memories = vec![(0usize, mem_feat, mem_pos)];
        let pointers = vec![(1usize, want.clone())];
        let neck_pos = Net::neck_pos();
        let cond = m.memory_cond(&net, &pix1, &neck_pos, &memories, &pointers, 8).expect("memory cond");
        let cond_ref_path = fx.join("f1/memory_conditioned.f32");
        if cond_ref_path.exists() {
            let want_cond = f32s(&cond_ref_path);
            let cond_cos = dot(&cond, &want_cond) / (norm(&cond) * norm(&want_cond) + 1e-9);
            let cond_md = cond.iter().zip(&want_cond).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("fixture f1 memory conditioning: cos={cond_cos:.6} maxdiff={cond_md:.4}");
            assert!(cond_cos >= 0.99, "frame-1 memory conditioning cosine {cond_cos:.6}");
        }
        let dense = net.no_mask_prompt().expect("no mask");
        let (_m, iou1, sc1, t1) = net
            .mask_decoder(&pix0_to_chw(&cond), &pos, &dense, &sparse, (&feats0.l0, &feats0.l1), true)
            .expect("decoder f1");
        let want_sc = f32s(&fx.join("f1/obj0/non_cond_frame_outputs/object_score_logits.f32"))[0];
        eprintln!(
            "fixture f1: obj_score ours={sc1:.3} reference={want_sc:.3} (delta {:.3}), iou={iou1:?}",
            (sc1 - want_sc).abs()
        );
        assert!(
            (sc1 - want_sc).abs() < 1.0,
            "frame-1 object score {sc1:.3} vs reference {want_sc:.3}"
        );
        assert!(sc1 > 0.0, "the tracked object must read as present (got {sc1:.3})");
        let best1 = argmax(&iou1).min(t1.len() / HID - 1);
        let ptr1 = m
            .object_pointer(&net, &t1[best1 * HID..(best1 + 1) * HID], sc1 > 0.0)
            .expect("pointer f1");
        let want_ptr1 = f32s(&fx.join("f1/obj0/non_cond_frame_outputs/object_pointer.f32"));
        for (j, tok) in t1.chunks_exact(HID).enumerate() {
            let p = m.object_pointer(&net, tok, sc1 > 0.0).expect("pointer candidate f1");
            let c = dot(&p, &want_ptr1) / (norm(&p) * norm(&want_ptr1) + 1e-9);
            eprintln!("fixture f1 pointer candidate {j}: cos={c:.5} iou={:.6}", iou1[j]);
        }
        let ptr_cos1 = dot(&ptr1, &want_ptr1) / (norm(&ptr1) * norm(&want_ptr1) + 1e-9);
        eprintln!("fixture f1 pointer: cos={ptr_cos1:.5}");

        // ── 2b. frame 2: TWO spatial memories must each get the grid RoPE ──
        // This catches a cross-attention failure that frame 1 cannot see: HF
        // repeats the 72×72 frequencies for every memory frame, while the
        // object pointers at the end stay unrotated.
        let pix2 = f32s(&fx.join("f2/neck_l2.f32"));
        let memories2 = vec![
            (0usize, mem_feat_ref.clone(), f32s(&fx.join("f0/obj0/cond_frame_outputs/maskmem_pos_enc.f32"))),
            (
                1usize,
                f32s(&fx.join("f1/obj0/non_cond_frame_outputs/maskmem_features.f32")),
                f32s(&fx.join("f1/obj0/non_cond_frame_outputs/maskmem_pos_enc.f32")),
            ),
        ];
        let pointers2 = vec![(2usize, want.clone()), (1usize, want_ptr1.clone())];
        let cond2 = m.memory_cond(&net, &pix2, &neck_pos, &memories2, &pointers2, 8).expect("memory cond f2");
        let cond2_ref_path = fx.join("f2/memory_conditioned.f32");
        if cond2_ref_path.exists() {
            let want_cond2 = f32s(&cond2_ref_path);
            let cond_cos2 = dot(&cond2, &want_cond2) / (norm(&cond2) * norm(&want_cond2) + 1e-9);
            let cond_md2 = cond2.iter().zip(&want_cond2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("fixture f2 memory conditioning: cos={cond_cos2:.6} maxdiff={cond_md2:.4}");
            assert!(cond_cos2 >= 0.99, "frame-2 memory conditioning cosine {cond_cos2:.6}");
        }
        let (_m2, iou2, sc2, t2) = net
            .mask_decoder(&pix0_to_chw(&cond2), &pos, &dense, &sparse, (&feats0.l0, &feats0.l1), true)
            .expect("decoder f2");
        let want_sc2 = f32s(&fx.join("f2/obj0/non_cond_frame_outputs/object_score_logits.f32"))[0];
        let best2 = argmax(&iou2).min(t2.len() / HID - 1);
        let ptr2 = m
            .object_pointer(&net, &t2[best2 * HID..(best2 + 1) * HID], sc2 > 0.0)
            .expect("pointer f2");
        let want_ptr2 = f32s(&fx.join("f2/obj0/non_cond_frame_outputs/object_pointer.f32"));
        let ptr_cos2 = dot(&ptr2, &want_ptr2) / (norm(&ptr2) * norm(&want_ptr2) + 1e-9);
        eprintln!("fixture f2: obj_score ours={sc2:.3} reference={want_sc2:.3}, pointer cos={ptr_cos2:.5}");
        assert!(ptr_cos1 >= 0.99, "frame-1 object pointer cosine {ptr_cos1:.5}");
        assert!((sc2 - want_sc2).abs() < 1.0, "frame-2 object score {sc2:.3} vs {want_sc2:.3}");
        assert!(ptr_cos2 >= 0.99, "frame-2 object pointer cosine {ptr_cos2:.5}");

        // ── 3. our own memory encoder vs the reference's stored memory ──
        // (the live pipeline feeds the tracker OUR encoding of frame 0, so it
        // must reproduce `maskmem_features` for the same inputs)
        for from_pts in [true, false] {
            let (mf, mp) = m
                .encode_memory(&net, &pix0_chw, &hi0, IMG, IMG, sc0, from_pts)
                .expect("encode_memory");
            let cos = dot(&mf, &mem_feat_ref) / (norm(&mf) * norm(&mem_feat_ref) + 1e-9);
            let md = mf
                .iter()
                .zip(&mem_feat_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "fixture mem encode from_pts={from_pts}: cos={cos:.5} maxdiff={md:.4} rms={:.4} (ref rms {:.4})",
                norm(&mf) / (mf.len() as f32).sqrt(),
                norm(&mem_feat_ref) / (mem_feat_ref.len() as f32).sqrt()
            );
            assert!(cos >= 0.99, "memory features cosine {cos:.5} (from_pts={from_pts})");
            let want_mp = f32s(&fx.join("f0/obj0/cond_frame_outputs/maskmem_pos_enc.f32"));
            let pos_cos = dot(&mp, &want_mp) / (norm(&mp) * norm(&want_mp) + 1e-9);
            assert!(pos_cos >= 0.99, "memory position cosine {pos_cos:.5}");
        }
    }

    fn pix0_to_chw(rows: &[f32]) -> Vec<f32> {
        sam3::rows_to_chw(rows, EMB, EMB)
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}