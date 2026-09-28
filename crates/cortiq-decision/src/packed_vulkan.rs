//! Resident sequential reconstruction; shared by standalone and joint Vulkan paths.
use super::{LANES, Packed};
use crate::gpu_vulkan::{ContextGpu, binding, entry};
use anyhow::{Context, Result, ensure};
use std::sync::Arc;

pub(crate) struct VulkanScorer {
    pub context: Arc<ContextGpu>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    weights: wgpu::Buffer,
    metadata: wgpu::Buffer,
    input: wgpu::Buffer,
    pub output: wgpu::Buffer,
    readback: wgpu::Buffer,
    cached: Option<(wgpu::Buffer, u32, wgpu::BindGroup)>,
    dim: usize,
    pub tasks: usize,
    submissions: u64,
}
impl VulkanScorer {
    pub(super) fn new(p: &Packed) -> Result<Self> {
        ensure!(
            p.dim > 0 && p.dim <= 8192,
            "Vulkan signal dimension must be 1..8192"
        );
        let context = ContextGpu::get()?;
        ensure!(
            p.tasks <= context.device.limits().max_compute_workgroups_per_dimension as usize,
            "too many Vulkan topologies"
        );
        let mut weights = Vec::<f32>::new();
        let mut metadata = Vec::<u32>::new();
        for task in 0..p.tasks {
            let block = &p.blocks[task / LANES];
            let lane = task % LANES;
            let mean = u32::try_from(weights.len()).context("Vulkan mean offset overflow")?;
            weights.extend(block.mean.iter().map(|v| v[lane]));
            let basis = u32::try_from(weights.len()).context("Vulkan basis offset overflow")?;
            weights.extend(block.basis[..p.ranks[task] * p.dim].iter().map(|v| v[lane]));
            metadata.extend([mean, basis, u32::try_from(p.ranks[task])?, 0]);
        }
        ensure!(
            weights.len() * 4 <= context.device.limits().max_storage_buffer_binding_size as usize,
            "Vulkan topology buffer exceeds device limit"
        );
        let device = &context.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decision resonance"),
            entries: &[
                entry(0, true),
                entry(1, true),
                entry(2, true),
                entry(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decision resonance"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let source = reconstruction_shader(p.dim);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decision reconstruction FP32"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("decision reconstruction"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reconstruction"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            anyhow::bail!("Vulkan reconstruction pipeline: {e}");
        }
        let result = Self {
            weights: context.upload(bytemuck::cast_slice(&weights), wgpu::BufferUsages::STORAGE),
            metadata: context.upload(bytemuck::cast_slice(&metadata), wgpu::BufferUsages::STORAGE),
            input: context.buffer(
                (p.dim * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            output: context.buffer(
                (p.tasks * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: context.buffer(
                (p.tasks * 4) as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            context,
            pipeline,
            layout,
            dim: p.dim,
            tasks: p.tasks,
            submissions: 0,
            cached: None,
        };
        result.context.check()?;
        Ok(result)
    }
    pub(super) fn submissions(&self) -> u64 {
        self.submissions
    }
    pub(crate) fn bind(&mut self, input: &wgpu::Buffer, offset: u32) {
        if self
            .cached
            .as_ref()
            .is_some_and(|(b, o, _)| b == input && *o == offset)
        {
            return;
        }
        let params = self.context.upload(
            bytemuck::cast_slice(&[offset, 0, 0, 0]),
            wgpu::BufferUsages::UNIFORM,
        );
        let group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("decision resonance inputs"),
                layout: &self.layout,
                entries: &[
                    binding(0, input),
                    binding(1, &self.weights),
                    binding(2, &self.metadata),
                    binding(3, &self.output),
                    binding(4, &params),
                ],
            });
        self.cached = Some((input.clone(), offset, group));
    }
    pub(crate) fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            &self.cached.as_ref().expect("bound resonance input").2,
            &[],
        );
        pass.dispatch_workgroups(self.tasks as u32, 1, 1);
    }
    pub(crate) fn completed(&mut self, values: &[f32]) -> Result<()> {
        ensure!(
            values.len() == self.tasks && values.iter().all(|v| v.is_finite()),
            "Vulkan reconstruction numeric overflow or length mismatch"
        );
        self.submissions += 1;
        Ok(())
    }
    pub(super) fn errors(&mut self, x: &[f32], out: &mut [f32]) -> Result<()> {
        ensure!(
            x.len() == self.dim && out.len() == self.tasks,
            "Vulkan scorer buffer dimensions"
        );
        self.context.check()?;
        self.context
            .queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(x));
        self.bind(&self.input.clone(), 0);
        let mut cmd = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decision reconstruction"),
            });
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reconstruction"),
                timestamp_writes: None,
            });
            self.encode(&mut pass);
        }
        cmd.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, (self.tasks * 4) as u64);
        let submitted = self.context.queue.submit([cmd.finish()]);
        let values = self
            .context
            .readback(&self.readback, (self.tasks * 4) as u64, submitted)?;
        self.completed(&values)?;
        out.copy_from_slice(&values);
        Ok(())
    }
}

// Generate scalar register slots. Dynamic private-array indexing can spill the
// residual to device-local memory; each rank must keep it resident in registers.
fn reconstruction_shader(dim: usize) -> String {
    let mut init = String::new();
    let mut dot = String::new();
    let mut update = String::new();
    let mut error = String::new();
    for j in 0..dim.div_ceil(256) {
        let offset = j * 256;
        init.push_str(&format!("var r{j}=0.0; if({offset}u+lid<DIM) {{ r{j}=input[p.offset+{offset}u+lid]-weights[task.x+{offset}u+lid]; }}\n"));
        dot.push_str(&format!("var b{j}=0.0; if({offset}u+lid<DIM) {{ b{j}=weights[task.y+k*DIM+{offset}u+lid]; }} sum+=r{j}*b{j};\n"));
        update.push_str(&format!("r{j}-=c*b{j};\n"));
        error.push_str(&format!("sum+=r{j}*r{j};\n"));
    }
    include_str!("packed_vulkan.wgsl")
        .replace("SIGNAL_DIM", &format!("{dim}u"))
        .replace("// REGISTER_INIT", &init)
        .replace("// REGISTER_DOT", &dot)
        .replace("// REGISTER_UPDATE", &update)
        .replace("// REGISTER_ERROR", &error)
}
