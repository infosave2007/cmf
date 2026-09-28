//! Resident FP32 Vulkan BERT + resonance: one submission and one readback.
use super::{BertDims, BertModel, EmbeddingAndErrors, Linear, Norm};
use crate::{
    gpu_vulkan::{ContextGpu, binding, entry},
    packed::Packed,
    signal::PHI_H_DIM,
};
use anyhow::{Result, ensure};
use std::{num::NonZeroU64, sync::Arc};

struct GpuLinear {
    w: u32,
    b: u32,
    k: u32,
    m: u32,
}
struct GpuNorm {
    w: u32,
    b: u32,
}
struct Layer {
    qkv: GpuLinear,
    o: GpuLinear,
    inter: GpuLinear,
    out: GpuLinear,
    attn: GpuNorm,
    norm: GpuNorm,
}
struct Scratch {
    emb: u32,
    x: u32,
    a: u32,
    qkv: u32,
    scores: u32,
    ctx: u32,
    tmp: u32,
    inter: u32,
    signal: u32,
    total: u32,
}
struct Op {
    pipeline: usize,
    groups: [u32; 3],
    params: [u32; 16],
}

struct Profile {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}
pub(super) struct VulkanEncoder {
    context: Arc<ContextGpu>,
    dims: BertDims,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    arena: wgpu::Buffer,
    input: wgpu::Buffer,
    readback: wgpu::Buffer,
    readback_bytes: u64,
    input_host: Vec<u32>,
    graphs: Vec<Vec<Op>>,
    stride: u32,
    signal: u32,
    submissions: u64,
    profile: Option<Profile>,
}
impl std::fmt::Debug for VulkanEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEncoder")
            .field("device", &self.context.name)
            .field("submissions", &self.submissions)
            .finish_non_exhaustive()
    }
}
fn append(dst: &mut Vec<f32>, values: &[f32]) -> Result<u32> {
    let offset = u32::try_from(dst.len())?;
    dst.extend_from_slice(values);
    Ok(offset)
}
fn norm(dst: &mut Vec<f32>, n: &Norm) -> Result<GpuNorm> {
    Ok(GpuNorm {
        w: append(dst, &n.w)?,
        b: append(dst, &n.b)?,
    })
}
fn linear(dst: &mut Vec<f32>, l: &Linear) -> Result<GpuLinear> {
    Ok(GpuLinear {
        w: append(dst, &l.w)?,
        b: append(dst, &l.b)?,
        k: u32::try_from(l.inp)?,
        m: u32::try_from(l.out)?,
    })
}
impl Scratch {
    fn new(d: &BertDims) -> Result<Self> {
        let mut pos = 0usize;
        let mut allocate = |n: usize| -> Result<u32> {
            let start = u32::try_from(pos)?;
            pos = pos
                .checked_add(n)
                .ok_or_else(|| anyhow::anyhow!("Vulkan scratch overflow"))?
                .div_ceil(64)
                * 64;
            Ok(start)
        };
        let nh = d.max_position * d.hidden;
        let result = Self {
            emb: allocate(nh)?,
            x: allocate(nh)?,
            a: allocate(nh)?,
            qkv: allocate(3 * nh)?,
            scores: allocate(d.heads * d.max_position * d.max_position)?,
            ctx: allocate(nh)?,
            tmp: allocate(nh)?,
            inter: allocate(d.max_position * d.intermediate)?,
            signal: allocate(d.hidden + PHI_H_DIM)?,
            total: 0,
        };
        Ok(Self {
            total: u32::try_from(pos)?,
            ..result
        })
    }
}
fn graph(
    d: &BertDims,
    n: usize,
    s: &Scratch,
    embedding: [u32; 3],
    emb_norm: &GpuNorm,
    layers: &[Layer],
    fused_attention: bool,
) -> Vec<Op> {
    let [word, pos, kind] = embedding;
    let (n, h) = (n as u32, d.hidden as u32);
    let mut ops = Vec::new();
    let base = [
        n,
        h,
        d.heads as u32,
        d.head_dim as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        (d.ln_eps as f32).to_bits(),
        0,
        0,
        0,
    ];
    let mut push = |pipeline, groups, fields: [u32; 8]| {
        let mut p = base;
        p[4..12].copy_from_slice(&fields);
        ops.push(Op {
            pipeline,
            groups,
            params: p,
        });
    };
    // Fields: x, y, z, weight, bias, k, m, mode.
    push(
        0,
        [(n * h).div_ceil(128), 1, 1],
        [0, s.emb, kind, word, pos, 0, 0, 0],
    );
    push(
        2,
        [n, 1, 1],
        [s.emb, s.x, 0, emb_norm.w, emb_norm.b, 0, 0, 0],
    );
    for l in layers {
        let q = &l.qkv;
        push(
            1,
            [q.m.div_ceil(32), n.div_ceil(8), 1],
            [s.x, s.qkv, 0, q.w, q.b, q.k, q.m, 0],
        );
        if fused_attention {
            push(
                9,
                [(d.heads as u32 * n).div_ceil(4), 1, 1],
                [s.qkv, s.ctx, 0, 0, 0, 0, 0, 0],
            );
        } else {
            push(
                3,
                [(d.heads as u32 * n * n).div_ceil(128), 1, 1],
                [s.qkv, s.scores, 0, 0, 0, 0, 0, 0],
            );
            push(
                4,
                [(d.heads as u32 * n).div_ceil(128), 1, 1],
                [s.scores, 0, 0, 0, 0, 0, 0, 0],
            );
            push(
                5,
                [(n * h).div_ceil(128), 1, 1],
                [s.qkv, s.ctx, s.scores, 0, 0, 0, 0, 0],
            );
        }
        let q = &l.o;
        push(
            1,
            [q.m.div_ceil(32), n.div_ceil(8), 1],
            [s.ctx, s.tmp, 0, q.w, q.b, q.k, q.m, 0],
        );
        push(2, [n, 1, 1], [s.tmp, s.a, s.x, l.attn.w, l.attn.b, 0, 0, 1]);
        let q = &l.inter;
        push(
            1,
            [q.m.div_ceil(32), n.div_ceil(8), 1],
            [s.a, s.inter, 0, q.w, q.b, q.k, q.m, 1],
        );
        let q = &l.out;
        push(
            1,
            [q.m.div_ceil(32), n.div_ceil(8), 1],
            [s.inter, s.tmp, 0, q.w, q.b, q.k, q.m, 0],
        );
        push(2, [n, 1, 1], [s.tmp, s.x, s.a, l.norm.w, l.norm.b, 0, 0, 1]);
    }
    push(
        6,
        [(h + PHI_H_DIM as u32).div_ceil(128), 1, 1],
        [s.x, s.signal, d.max_position as u32, 0, 0, 0, 0, 0],
    );
    push(7, [1, 1, 1], [s.signal, 0, 0, 0, 0, 0, 0, 0]);
    ops
}
impl VulkanEncoder {
    pub(super) fn new(model: &BertModel) -> Result<Self> {
        let context = ContextGpu::get()?;
        let d = &model.dims;
        ensure!(
            d.max_position <= 4096 && d.hidden + PHI_H_DIM <= 8192,
            "Vulkan encoder dimensions exceed supported limits"
        );
        let s = Scratch::new(d)?;
        let limits = context.device.limits();
        ensure!(
            u64::from(s.total) * 4 <= limits.max_storage_buffer_binding_size,
            "Vulkan activation arena exceeds device binding limit"
        );
        let mut weights = Vec::<f32>::new();
        let word = append(&mut weights, &model.word)?;
        let pos = append(&mut weights, &model.pos)?;
        let kind = append(&mut weights, &model.type_row)?;
        let emb_norm = norm(&mut weights, &model.ln_emb)?;
        let mut layers = Vec::new();
        for l in &model.layers {
            let qkv_w = u32::try_from(weights.len())?;
            for q in [&l.q, &l.k, &l.v] {
                append(&mut weights, &q.w)?;
            }
            let qkv_b = u32::try_from(weights.len())?;
            for q in [&l.q, &l.k, &l.v] {
                append(&mut weights, &q.b)?;
            }
            layers.push(Layer {
                qkv: GpuLinear {
                    w: qkv_w,
                    b: qkv_b,
                    k: d.hidden as u32,
                    m: 3 * d.hidden as u32,
                },
                o: linear(&mut weights, &l.o)?,
                inter: linear(&mut weights, &l.inter)?,
                out: linear(&mut weights, &l.out)?,
                attn: norm(&mut weights, &l.ln_attn)?,
                norm: norm(&mut weights, &l.ln_out)?,
            });
        }
        ensure!(
            weights.len() * 4 <= limits.max_storage_buffer_binding_size as usize,
            "Vulkan encoder weights exceed binding limit"
        );
        let stride = limits.min_uniform_buffer_offset_alignment.max(64);
        let mut graphs: Vec<_> = (1..=d.max_position)
            .map(|n| {
                graph(
                    d,
                    n,
                    &s,
                    [word, pos, kind],
                    &emb_norm,
                    &layers,
                    context.nvidia && d.head_dim <= 32,
                )
            })
            .collect();
        // Narrow FP32 products benefit from subgroup dot products on RTX.
        // Longer sequences amortize a tiled GEMM; extending GEMV to 512 tokens
        // regressed the long-input acceptance test, so retain the crossover.
        if context.nvidia {
            for graph in &mut graphs {
                for op in graph {
                    if op.pipeline == 1
                        && op.params[0] <= 128
                        && op.params[9] <= 1024
                        && op.params[10] <= 3072
                    {
                        op.pipeline = 8;
                        op.groups = [op.params[10].div_ceil(4), op.params[0], 1];
                    }
                }
            }
        }
        let ops = graphs[0].len();
        let param_bytes = graphs.len() * ops * stride as usize;
        ensure!(
            param_bytes <= u32::MAX as usize && param_bytes as u64 <= limits.max_buffer_size,
            "Vulkan parameter bank too large"
        );
        let mut params = vec![0u32; param_bytes / 4];
        for (ni, graph) in graphs.iter().enumerate() {
            for (oi, op) in graph.iter().enumerate() {
                ensure!(
                    op.groups
                        .iter()
                        .all(|&g| g <= limits.max_compute_workgroups_per_dimension),
                    "Vulkan dispatch exceeds grid limit"
                );
                let start = (ni * ops + oi) * stride as usize / 4;
                params[start..start + 16].copy_from_slice(&op.params);
            }
        }
        let device = &context.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decision encoder"),
            entries: &[
                entry(0, true),
                entry(1, false),
                entry(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decision encoder"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decision BERT FP32"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bert_vulkan.wgsl").into()),
        });
        let pipelines = [
            "embedding",
            "linear",
            "norm",
            "scores",
            "softmax",
            "context",
            "pool",
            "normalize",
            "linear_short",
            "attention_fused",
        ]
        .iter()
        .map(|&name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
        if let Some(e) = pollster::block_on(scope.pop()) {
            anyhow::bail!("Vulkan BERT pipeline: {e}");
        }
        let arena = context.buffer(
            u64::from(s.total) * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let input = context.buffer(
            ((d.max_position + PHI_H_DIM) * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let weights = context.upload(bytemuck::cast_slice(&weights), wgpu::BufferUsages::STORAGE);
        let params = context.upload(bytemuck::cast_slice(&params), wgpu::BufferUsages::UNIFORM);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decision resident graph"),
            layout: &layout,
            entries: &[
                binding(0, &weights),
                binding(1, &arena),
                binding(2, &input),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: NonZeroU64::new(64),
                    }),
                },
            ],
        });
        let readback_bytes = (d.hidden * 4) as u64;
        let readback = context.buffer(
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        context.check()?;
        let profile = if std::env::var_os("CORTIQ_DECISION_VULKAN_PROFILE").is_some() {
            ensure!(
                device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
                "initialize Vulkan profiling before loading the first GPU model"
            );
            Some(Profile {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("decision timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                resolve: context.buffer(
                    256,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                ),
            })
        } else {
            None
        };
        Ok(Self {
            context,
            dims: d.clone(),
            pipelines,
            bind_group,
            arena,
            input,
            readback,
            readback_bytes,
            input_host: vec![0; d.max_position + PHI_H_DIM],
            graphs,
            stride,
            signal: s.signal,
            submissions: 0,
            profile,
        })
    }
    pub(super) fn device_name(&self) -> &str {
        &self.context.name
    }
    pub(super) fn submissions(&self) -> u64 {
        self.submissions
    }
    pub(super) fn embed_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.forward(ids, &[], &[]).map(|r| r.0)
    }
    pub(super) fn embed_and_score(
        &mut self,
        ids: &[u32],
        hash: &[f32],
        packed: &[&Packed],
    ) -> Result<EmbeddingAndErrors> {
        ensure!(
            hash.len() == PHI_H_DIM && hash.iter().all(|v| v.is_finite()),
            "invalid Vulkan hash features"
        );
        for (i, p) in packed.iter().enumerate() {
            ensure!(
                p.tasks() == 0 || p.dim() == self.dims.hidden + PHI_H_DIM,
                "Vulkan signal dimension"
            );
            ensure!(
                !packed[..i].iter().any(|q| std::ptr::eq(*q, *p)),
                "duplicate Vulkan scorer"
            );
            ensure!(
                p.tasks() == 0 || p.vulkan.is_some(),
                "joint Vulkan execution requires Vulkan scorers"
            );
        }
        self.forward(ids, hash, packed)
    }
    fn forward(
        &mut self,
        ids: &[u32],
        hash: &[f32],
        packed: &[&Packed],
    ) -> Result<EmbeddingAndErrors> {
        ensure!(
            !ids.is_empty() && ids.len() <= self.dims.max_position,
            "invalid Vulkan sequence length"
        );
        ensure!(
            ids.iter().all(|&id| (id as usize) < self.dims.vocab),
            "Vulkan token id outside vocabulary"
        );
        self.context.check()?;
        let host_start = self.profile.as_ref().map(|_| std::time::Instant::now());
        self.input_host[..ids.len()].copy_from_slice(ids);
        self.input_host[self.dims.max_position..].fill(0);
        for (out, value) in self.input_host[self.dims.max_position..]
            .iter_mut()
            .zip(hash)
        {
            *out = value.to_bits();
        }
        let mut order: Vec<_> = packed.iter().enumerate().collect();
        order.sort_unstable_by_key(|(_, p)| **p as *const Packed as usize);
        let mut scorers: Vec<_> = order
            .into_iter()
            .map(|(i, p)| (i, p.vulkan.as_ref().map(|v| v.lock())))
            .collect();
        for (_, s) in &mut scorers {
            if let Some(s) = s {
                ensure!(
                    Arc::ptr_eq(&s.context, &self.context),
                    "Vulkan scorer belongs to another device"
                );
                s.bind(&self.arena, self.signal);
            }
        }
        let total = self.dims.hidden + packed.iter().map(|p| p.tasks()).sum::<usize>();
        let result_bytes = (total * 4) as u64;
        let bytes = result_bytes + if self.profile.is_some() { 16 } else { 0 };
        if bytes > self.readback_bytes {
            self.readback = self.context.buffer(
                bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            self.readback_bytes = bytes;
        }
        self.context
            .queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&self.input_host));
        let mut cmd = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decision joint graph"),
            });
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BERT + reconstruction"),
                timestamp_writes: self
                    .profile
                    .as_ref()
                    .map(|p| wgpu::ComputePassTimestampWrites {
                        query_set: &p.queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
            });
            let graph = &self.graphs[ids.len() - 1];
            for (i, op) in graph.iter().enumerate() {
                pass.set_pipeline(&self.pipelines[op.pipeline]);
                let offset = (((ids.len() - 1) * graph.len() + i) as u32) * self.stride;
                pass.set_bind_group(0, &self.bind_group, &[offset]);
                pass.dispatch_workgroups(op.groups[0], op.groups[1], op.groups[2]);
            }
            for scorer in scorers.iter().filter_map(|(_, s)| s.as_ref()) {
                scorer.encode(&mut pass);
            }
        }
        cmd.copy_buffer_to_buffer(
            &self.arena,
            u64::from(self.signal) * 4,
            &self.readback,
            0,
            (self.dims.hidden * 4) as u64,
        );
        let mut position = self.dims.hidden;
        for scorer in scorers.iter().filter_map(|(_, s)| s.as_ref()) {
            cmd.copy_buffer_to_buffer(
                &scorer.output,
                0,
                &self.readback,
                (position * 4) as u64,
                (scorer.tasks * 4) as u64,
            );
            position += scorer.tasks;
        }
        if let Some(profile) = &self.profile {
            cmd.resolve_query_set(&profile.queries, 0..2, &profile.resolve, 0);
            cmd.copy_buffer_to_buffer(&profile.resolve, 0, &self.readback, result_bytes, 16);
        }
        let encoded = host_start.map(|_| std::time::Instant::now());
        let submission = self.context.queue.submit([cmd.finish()]);
        let submitted = host_start.map(|_| std::time::Instant::now());
        let values = self.context.readback(&self.readback, bytes, submission)?;
        ensure!(
            values[..total].iter().all(|v| v.is_finite()),
            "Vulkan returned non-finite features or errors"
        );
        if let (Some(start), Some(encoded), Some(submitted)) = (host_start, encoded, submitted) {
            let tick = |i: usize| {
                u64::from(values[i].to_bits()) | (u64::from(values[i + 1].to_bits()) << 32)
            };
            let gpu_us = tick(total + 2).saturating_sub(tick(total)) as f64
                * f64::from(self.context.queue.get_timestamp_period())
                / 1000.0;
            eprintln!(
                "VULKAN_PROFILE n={} encode_us={} submit_us={} wait_us={} gpu_us={}",
                ids.len(),
                (encoded - start).as_secs_f64() * 1e6,
                (submitted - encoded).as_secs_f64() * 1e6,
                submitted.elapsed().as_secs_f64() * 1e6,
                gpu_us
            );
        }
        self.submissions += 1;
        let mut errors = vec![Vec::new(); packed.len()];
        let mut position = self.dims.hidden;
        for (i, scorer) in &mut scorers {
            if let Some(s) = scorer {
                let output = &values[position..position + s.tasks];
                s.completed(output)?;
                errors[*i] = output.to_vec();
                position += s.tasks;
            }
        }
        Ok((values[..self.dims.hidden].to_vec(), errors))
    }
}
