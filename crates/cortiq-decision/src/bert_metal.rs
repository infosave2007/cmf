//! Resident FP32 BERT graph for Apple Silicon. A single command buffer per
//! text, including pooling, normalization and requested resonance skills.
//! Only tokenization, hashing and final ranking/gates remain on the CPU.
use super::{BertDims, BertModel, Linear, Norm};
use crate::packed::Packed;
use anyhow::{Context, Result, ensure};
use metal::*;
use std::ffi::c_void;

const MSL: &str = include_str!("bert_metal.metal");
const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

struct GpuLinear {
    w: Buffer,
    b: Buffer,
    inp: usize,
    out: usize,
}
struct GpuNorm {
    w: Buffer,
    b: Buffer,
}
struct GpuLayer {
    qkv: GpuLinear,
    o: GpuLinear,
    inter: GpuLinear,
    out: GpuLinear,
    attn_norm: GpuNorm,
    out_norm: GpuNorm,
}
struct Scratch {
    ids: Buffer,
    emb: Buffer,
    x: Buffer,
    a: Buffer,
    qkv: Buffer,
    scores: Buffer,
    ctx: Buffer,
    tmp: Buffer,
    inter: Buffer,
    signal: Buffer,
}

/// Owned by Encoder behind a mutex: concurrent requests cannot race on scratch.
pub(super) struct MetalEncoder {
    dims: BertDims,
    device: Device,
    queue: CommandQueue,
    embed: ComputePipelineState,
    linear: ComputePipelineState,
    linear_direct: ComputePipelineState,
    norm: ComputePipelineState,
    norm_simd: ComputePipelineState,
    scores: ComputePipelineState,
    softmax: ComputePipelineState,
    context: ComputePipelineState,
    fused_attention: ComputePipelineState,
    pool: ComputePipelineState,
    normalize: ComputePipelineState,
    word: Buffer,
    pos: Buffer,
    type_row: Buffer,
    emb_norm: GpuNorm,
    layers: Vec<GpuLayer>,
    scratch: Scratch,
    submissions: u64,
}

impl std::fmt::Debug for MetalEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalEncoder")
            .field("device", &self.device.name())
            .field("submissions", &self.submissions)
            .finish_non_exhaustive()
    }
}

fn upload(device: &DeviceRef, values: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        SHARED,
    )
}
fn norm_weights(device: &DeviceRef, n: &Norm) -> GpuNorm {
    GpuNorm {
        w: upload(device, &n.w),
        b: upload(device, &n.b),
    }
}
fn linear_weights(device: &DeviceRef, l: &Linear) -> GpuLinear {
    GpuLinear {
        w: upload(device, &l.w),
        b: upload(device, &l.b),
        inp: l.inp,
        out: l.out,
    }
}
fn bytes<T>(enc: &ComputeCommandEncoderRef, slot: u64, value: &T) {
    enc.set_bytes(
        slot,
        std::mem::size_of::<T>() as u64,
        (value as *const T).cast::<c_void>(),
    );
}
fn buffers(enc: &ComputeCommandEncoderRef, bs: &[&Buffer]) {
    for (i, b) in bs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
}
fn threads(enc: &ComputeCommandEncoderRef, n: usize) {
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(128, 1, 1));
}

impl MetalEncoder {
    pub(super) fn new(model: &BertModel) -> Result<Self> {
        objc::rc::autoreleasepool(|| Self::init(model))
    }
    fn init(model: &BertModel) -> Result<Self> {
        let device = Device::system_default().context("no Metal device")?;
        ensure!(
            device.has_unified_memory(),
            "decision Metal requires Apple Silicon unified memory (device: {})",
            device.name()
        );
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_0);
        options.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(
                &format!("#define NORM_HIDDEN {}\n{MSL}", model.dims.hidden.min(1024)),
                &options,
            )
            .map_err(|e| anyhow::anyhow!("decision Metal shader compilation: {e}"))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("Metal {name}: {e}"))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| anyhow::anyhow!("Metal {name}: {e}"))?;
            ensure!(
                p.thread_execution_width() == 32 && p.max_total_threads_per_threadgroup() >= 128,
                "decision Metal kernel {name} requires 32-lane SIMD groups and 128-thread groups"
            );
            Ok(p)
        };
        let embed = pipeline("embedding")?;
        let linear = pipeline("linear")?;
        let linear_direct = pipeline("linear_direct")?;
        let norm = pipeline("norm")?;
        let norm_simd = pipeline("norm_simd")?;
        let scores = pipeline("attention_scores")?;
        let softmax = pipeline("attention_softmax")?;
        let context = pipeline("attention_context")?;
        let fused_attention = pipeline("attention_fused")?;
        let pool = pipeline("pool")?;
        let normalize = pipeline("normalize")?;
        let d = &model.dims;
        let scratch_buffer = |n: usize| {
            let buffer = device.new_buffer((n * 4) as u64, SHARED);
            // Inactive rows in a direct matrix tile are allocated and initialized.
            unsafe {
                std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, n * 4);
            }
            buffer
        };
        // Direct SIMD matrix loads may read up to seven inactive token rows.
        // Reserve the full tile; inactive rows never contribute to active outputs.
        let capacity = d.max_position.div_ceil(8) * 8;
        let nh = capacity * d.hidden;
        let scratch = Scratch {
            ids: scratch_buffer(d.max_position),
            signal: scratch_buffer(d.hidden + crate::signal::PHI_H_DIM),
            emb: scratch_buffer(nh),
            x: scratch_buffer(nh),
            a: scratch_buffer(nh),
            qkv: scratch_buffer(nh * 3),
            scores: scratch_buffer(d.heads * d.max_position * d.max_position),
            ctx: scratch_buffer(nh),
            tmp: scratch_buffer(nh),
            inter: scratch_buffer(capacity * d.intermediate),
        };
        let layers = model
            .layers
            .iter()
            .map(|l| {
                let w: Vec<f32> = [&l.q, &l.k, &l.v]
                    .into_iter()
                    .flat_map(|v| v.w.iter().copied())
                    .collect();
                let b: Vec<f32> = [&l.q, &l.k, &l.v]
                    .into_iter()
                    .flat_map(|v| v.b.iter().copied())
                    .collect();
                GpuLayer {
                    qkv: GpuLinear {
                        w: upload(&device, &w),
                        b: upload(&device, &b),
                        inp: d.hidden,
                        out: 3 * d.hidden,
                    },
                    o: linear_weights(&device, &l.o),
                    inter: linear_weights(&device, &l.inter),
                    out: linear_weights(&device, &l.out),
                    attn_norm: norm_weights(&device, &l.ln_attn),
                    out_norm: norm_weights(&device, &l.ln_out),
                }
            })
            .collect();
        Ok(Self {
            queue: device.new_command_queue(),
            word: upload(&device, &model.word),
            pos: upload(&device, &model.pos),
            type_row: upload(&device, &model.type_row),
            emb_norm: norm_weights(&device, &model.ln_emb),
            dims: d.clone(),
            device,
            embed,
            linear,
            linear_direct,
            norm,
            norm_simd,
            scores,
            softmax,
            context,
            fused_attention,
            pool,
            normalize,
            layers,
            scratch,
            submissions: 0,
        })
    }
    pub(super) fn submissions(&self) -> u64 {
        self.submissions
    }
    pub(super) fn device_name(&self) -> &str {
        self.device.name()
    }

    fn linear(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: &GpuLinear,
        x: &Buffer,
        y: &Buffer,
        n: usize,
        gelu: bool,
    ) {
        let direct = l.inp % 8 == 0 && l.out % 8 == 0;
        enc.set_compute_pipeline_state(if direct {
            &self.linear_direct
        } else {
            &self.linear
        });
        buffers(enc, &[x, &l.w, &l.b, y]);
        bytes(
            enc,
            4,
            &[n as u32, l.inp as u32, l.out as u32, u32::from(gelu)],
        );
        enc.dispatch_thread_groups(
            MTLSize::new(
                l.out.div_ceil(if direct { 32 } else { 16 }) as u64,
                n.div_ceil(if direct { 8 } else { 16 }) as u64,
                1,
            ),
            MTLSize::new(128, 1, 1),
        );
    }
    fn norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &GpuNorm,
        x: &Buffer,
        residual: Option<&Buffer>,
        out: &Buffer,
        n: usize,
    ) {
        let simd = self.dims.hidden <= 1024;
        enc.set_compute_pipeline_state(if simd { &self.norm_simd } else { &self.norm });
        buffers(enc, &[x, residual.unwrap_or(x), &w.w, &w.b, out]);
        bytes(enc, 5, &(self.dims.hidden as u32));
        bytes(enc, 6, &(self.dims.ln_eps as f32));
        bytes(enc, 7, &u32::from(residual.is_some()));
        enc.dispatch_thread_groups(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(if simd { 32 } else { 128 }, 1, 1),
        );
    }
    fn validate_ids(&self, ids: &[u32]) -> Result<()> {
        ensure!(
            !ids.is_empty() && ids.len() <= self.dims.max_position,
            "invalid Metal token sequence length {}",
            ids.len()
        );
        ensure!(
            ids.iter().all(|&v| (v as usize) < self.dims.vocab),
            "Metal token id outside vocabulary"
        );
        Ok(())
    }
    pub(super) fn embed_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.validate_ids(ids)?;
        objc::rc::autoreleasepool(|| self.forward(ids, &[], &[]).map(|v| v.0))
    }
    pub(super) fn embed_and_score(
        &mut self,
        ids: &[u32],
        hash: &[f32],
        packed: &[&Packed],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.validate_ids(ids)?;
        ensure!(
            hash.len() == crate::signal::PHI_H_DIM,
            "Metal hash dimension"
        );
        ensure!(hash.iter().all(|v| v.is_finite()), "non-finite Metal hash");
        for (i, p) in packed.iter().enumerate() {
            ensure!(
                p.tasks() == 0 || p.dim() == self.dims.hidden + hash.len(),
                "Metal signal dimension"
            );
            ensure!(
                !packed[..i].iter().any(|q| std::ptr::eq(*q, *p)),
                "duplicate Metal scorer"
            );
            ensure!(
                p.tasks() == 0 || p.metal.is_some(),
                "joint Metal execution requires Metal scorers"
            );
        }
        objc::rc::autoreleasepool(|| self.forward(ids, hash, packed))
    }
    fn forward(
        &mut self,
        ids: &[u32],
        hash: &[f32],
        packed: &[&Packed],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let d = &self.dims;
        let n = ids.len();
        let dims = [n as u32, d.hidden as u32, d.heads as u32, d.head_dim as u32];
        let s = &self.scratch;
        // The caller holds the Encoder mutex, and the previous buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), s.ids.contents().cast::<u32>(), n);
        }
        // Global scorer-address order also prevents deadlock when independent
        // encoders share scorers requested in different orders. Standalone
        // scorer operations never acquire the encoder lock.
        let mut order: Vec<_> = packed.iter().enumerate().collect();
        order.sort_unstable_by_key(|(_, p)| **p as *const Packed as usize);
        let mut scorers: Vec<_> = order
            .into_iter()
            .map(|(i, p)| (i, p.metal.as_ref().map(|m| m.lock())))
            .collect();
        if !hash.is_empty() {
            let target = unsafe {
                std::slice::from_raw_parts_mut(
                    s.signal.contents().cast::<f32>().add(d.hidden),
                    hash.len(),
                )
            };
            for (out, h) in target.iter_mut().zip(hash) {
                *out = crate::rows::PHI_H_WEIGHT * h;
            }
        }
        let cmd = self.queue.new_command_buffer();
        cmd.set_label("Cortiq Decision FP32 BERT");
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.embed);
        buffers(
            enc,
            &[&s.ids, &self.word, &self.pos, &self.type_row, &s.emb],
        );
        bytes(enc, 5, &dims);
        threads(enc, n * d.hidden);
        self.norm(enc, &self.emb_norm, &s.emb, None, &s.x, n);
        for l in &self.layers {
            self.linear(enc, &l.qkv, &s.x, &s.qkv, n, false);
            if d.head_dim <= 32 {
                enc.set_compute_pipeline_state(&self.fused_attention);
                buffers(enc, &[&s.qkv, &s.ctx]);
                bytes(enc, 2, &dims);
                enc.dispatch_thread_groups(
                    MTLSize::new((d.heads * n) as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&self.scores);
                buffers(enc, &[&s.qkv, &s.scores]);
                bytes(enc, 2, &dims);
                threads(enc, d.heads * n * n);
                enc.set_compute_pipeline_state(&self.softmax);
                buffers(enc, &[&s.scores]);
                bytes(enc, 1, &dims);
                threads(enc, d.heads * n);
                enc.set_compute_pipeline_state(&self.context);
                buffers(enc, &[&s.qkv, &s.scores, &s.ctx]);
                bytes(enc, 3, &dims);
                threads(enc, n * d.hidden);
            }
            self.linear(enc, &l.o, &s.ctx, &s.tmp, n, false);
            self.norm(enc, &l.attn_norm, &s.tmp, Some(&s.x), &s.a, n);
            self.linear(enc, &l.inter, &s.a, &s.inter, n, true);
            self.linear(enc, &l.out, &s.inter, &s.tmp, n, false);
            self.norm(enc, &l.out_norm, &s.tmp, Some(&s.a), &s.x, n);
        }
        enc.set_compute_pipeline_state(&self.pool);
        buffers(enc, &[&s.x, &s.signal]);
        bytes(enc, 2, &dims);
        threads(enc, d.hidden);
        enc.set_compute_pipeline_state(&self.normalize);
        buffers(enc, &[&s.signal]);
        bytes(enc, 1, &(d.hidden as u32));
        enc.dispatch_threads(MTLSize::new(32, 1, 1), MTLSize::new(32, 1, 1));
        for scorer in scorers.iter().filter_map(|(_, s)| s.as_ref()) {
            scorer.encode(enc, &s.signal);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure!(
            cmd.status() == MTLCommandBufferStatus::Completed,
            "decision Metal command failed: {:?}",
            cmd.status()
        );
        self.submissions += 1;
        let v = unsafe { std::slice::from_raw_parts(s.signal.contents().cast::<f32>(), d.hidden) }
            .to_vec();
        ensure!(
            v.iter().all(|x| x.is_finite()),
            "decision Metal returned non-finite embeddings"
        );
        let mut errors = vec![Vec::new(); packed.len()];
        for (i, scorer) in &mut scorers {
            if let Some(scorer) = scorer {
                errors[*i] = scorer.completed()?;
            }
        }
        Ok((v, errors))
    }
}
