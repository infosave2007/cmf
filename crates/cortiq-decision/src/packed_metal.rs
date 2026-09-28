//! Resident topologies and sequential-projection reconstruction on Metal.
use super::{LANES, Packed};
use anyhow::{Context, Result, ensure};
use metal::*;

pub(crate) struct MetalScorer {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    input: Buffer,
    means: Buffer,
    basis: Buffer,
    offsets: Buffer,
    ranks: Buffer,
    output: Buffer,
    dim: usize,
    tasks: usize,
    submissions: u64,
}
impl MetalScorer {
    pub(super) fn new(packed: &Packed) -> Result<Self> {
        objc::rc::autoreleasepool(|| Self::init(packed))
    }
    fn init(p: &Packed) -> Result<Self> {
        let device = Device::system_default().context("no Metal device")?;
        ensure!(
            device.has_unified_memory(),
            "decision Metal requires unified memory"
        );
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_0);
        options.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(
                &format!(
                    "#define SIGNAL_DIM {}\n{}",
                    p.dim,
                    include_str!("packed_metal.metal")
                ),
                &options,
            )
            .map_err(|e| anyhow::anyhow!("Metal resonance shader: {e}"))?;
        let fun = lib
            .get_function("reconstruction_errors", None)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&fun)
            .map_err(|e| anyhow::anyhow!(e))?;
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 128,
            "unsupported Metal resonance group size"
        );
        ensure!(
            p.dim > 0 && p.dim <= 8192,
            "signal dimension {} exceeds Metal resonance register path (1..8192)",
            p.dim
        );
        let (mut means, mut basis, mut offsets, mut ranks) = (
            Vec::<f32>::new(),
            Vec::<f32>::new(),
            Vec::<u32>::new(),
            Vec::<u32>::new(),
        );
        for task in 0..p.tasks {
            let b = &p.blocks[task / LANES];
            let lane = task % LANES;
            means.extend(b.mean.iter().map(|v| v[lane]));
            offsets.push(u32::try_from(basis.len()).context("Metal topology offset too large")?);
            ranks.push(u32::try_from(p.ranks[task])?);
            basis.extend(b.basis[..p.ranks[task] * p.dim].iter().map(|v| v[lane]));
        }
        let upload = |data: &[u8]| {
            if data.is_empty() {
                device.new_buffer(4, MTLResourceOptions::StorageModeShared)
            } else {
                device.new_buffer_with_data(
                    data.as_ptr().cast(),
                    data.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
        };
        Ok(Self {
            queue: device.new_command_queue(),
            pipeline,
            input: device.new_buffer((p.dim * 4) as u64, MTLResourceOptions::StorageModeShared),
            output: device.new_buffer((p.tasks * 4) as u64, MTLResourceOptions::StorageModeShared),
            means: upload(bytemuck::cast_slice(&means)),
            basis: upload(bytemuck::cast_slice(&basis)),
            offsets: upload(bytemuck::cast_slice(&offsets)),
            ranks: upload(bytemuck::cast_slice(&ranks)),
            dim: p.dim,
            tasks: p.tasks,
            submissions: 0,
        })
    }
    pub(super) fn submissions(&self) -> u64 {
        self.submissions
    }
    pub(super) fn errors(&mut self, x: &[f32], out: &mut [f32]) -> Result<()> {
        ensure!(
            x.len() == self.dim && out.len() == self.tasks,
            "Metal scorer buffer dimensions"
        );
        objc::rc::autoreleasepool(|| {
            // Serialized by Packed's mutex; the previous command has completed.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr(),
                    self.input.contents().cast::<f32>(),
                    x.len(),
                );
            }
            let cmd = self.queue.new_command_buffer();
            cmd.set_label("Cortiq Decision reconstruction errors");
            let enc = cmd.new_compute_command_encoder();
            self.encode(enc, &self.input);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            ensure!(
                cmd.status() == MTLCommandBufferStatus::Completed,
                "Metal reconstruction command failed: {:?}",
                cmd.status()
            );
            out.copy_from_slice(&self.completed()?);
            Ok(())
        })
    }
    /// Append work to an encoder-owned command buffer. The caller holds both
    /// mutexes until completion; no readback or second CPU/GPU round trip.
    pub(crate) fn encode(&self, enc: &ComputeCommandEncoderRef, input: &Buffer) {
        enc.set_compute_pipeline_state(&self.pipeline);
        for (i, b) in [
            input,
            &self.means,
            &self.basis,
            &self.offsets,
            &self.ranks,
            &self.output,
        ]
        .iter()
        .enumerate()
        {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        let dim = self.dim as u32;
        enc.set_bytes(6, 4, (&dim as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(self.tasks as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
    }
    pub(crate) fn completed(&mut self) -> Result<Vec<f32>> {
        self.submissions += 1;
        let result =
            unsafe { std::slice::from_raw_parts(self.output.contents().cast::<f32>(), self.tasks) };
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "Metal reconstruction numeric overflow"
        );
        Ok(result.to_vec())
    }
}
